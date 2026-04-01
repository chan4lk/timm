"""
ASMS Stage 4: MLX Micro-Transformer Architecture

Decoder-only transformer sized to the OKR agent's decision complexity:
  - 4 layers, 256 hidden dim, 4 attention heads
  - ~15M parameters
  - 512 max context length
  - Task-specific vocabulary (~8K tokens)

Designed for Apple Silicon (M-series) via MLX with Metal acceleration.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class OKRModelConfig:
    """Architecture config matched to ASMS Medium complexity (|D| ≈ 500)."""

    vocab_size: int = 8000
    max_seq_len: int = 512
    num_layers: int = 4
    hidden_dim: int = 256
    num_heads: int = 4
    ffn_dim: int = 512  # 2x hidden
    dropout: float = 0.1
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def param_count_estimate(self) -> int:
        """Rough parameter count."""
        embed = self.vocab_size * self.hidden_dim  # token embeddings
        # Per layer: attention (4 projections) + FFN (2 layers) + 2 layer norms
        attn = 4 * self.hidden_dim * self.hidden_dim  # Q, K, V, O projections
        ffn = 2 * self.hidden_dim * self.ffn_dim  # up + down projections
        ln = 4 * self.hidden_dim  # 2 layer norms per layer
        per_layer = attn + ffn + ln
        total_layers = self.num_layers * per_layer
        final_ln = self.hidden_dim  # final layer norm
        lm_head = self.vocab_size * self.hidden_dim  # output projection
        return embed + total_layers + final_ln + lm_head


class RoPE(nn.Module):
    """Rotary Position Embedding — efficient positional encoding for short contexts."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        angles = mx.outer(t, freqs)
        self._cos = mx.cos(angles)
        self._sin = mx.sin(angles)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[-2]
        cos = self._cos[offset : offset + seq_len]
        sin = self._sin[offset : offset + seq_len]

        # Reshape for broadcasting: (seq_len, dim//2) -> (1, 1, seq_len, dim//2)
        # x shape is (B, heads, seq_len, head_dim)
        cos = cos.reshape(1, 1, seq_len, -1)
        sin = sin.reshape(1, 1, seq_len, -1)

        # Split into pairs and rotate
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 : self.dim]
        rotated = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        if x.shape[-1] > self.dim:
            rotated = mx.concatenate([rotated, x[..., self.dim :]], axis=-1)
        return rotated


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking."""

    def __init__(self, config: OKRModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rope = RoPE(config.head_dim, config.max_seq_len, config.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # KV cache for inference
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache


class FeedForward(nn.Module):
    """SwiGLU feed-forward — slightly more expressive than ReLU for small models."""

    def __init__(self, config: OKRModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + SwiGLU FFN."""

    def __init__(self, config: OKRModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.hidden_dim)
        self.ffn = FeedForward(config)

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None):
        # Pre-norm attention with residual
        h, new_cache = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + h
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class OKRMicroModel(nn.Module):
    """
    ASMS Micro-Transformer for OKR agent tool-call generation.

    Architecture: Decoder-only transformer (GPT-style)
    Input: Tokenized user query + session context
    Output: Structured tool-call JSON tokens (autoregressive)
    """

    def __init__(self, config: OKRModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, cache=None) -> tuple[mx.array, list]:
        B, L = tokens.shape
        x = self.embed(tokens)

        # Causal mask
        mask = None
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(x.dtype)

            # Adjust mask shape if using KV cache
            if cache is not None and cache[0] is not None:
                offset = cache[0][0].shape[2]
                prefix_mask = mx.zeros((L, offset), dtype=x.dtype)
                mask = mx.concatenate([prefix_mask, mask], axis=-1)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, mask=mask, cache=layer_cache)
            new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def generate(
        self,
        prompt_tokens: mx.array,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        eos_token_id: int = 3,
    ) -> mx.array:
        """Autoregressive generation with KV caching."""
        tokens = prompt_tokens
        cache = None
        generated = []

        for _ in range(max_new_tokens):
            logits, cache = self(tokens, cache=cache)
            # Take last token's logits
            next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                # Top-p sampling
                sorted_logits = mx.sort(next_logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum = mx.cumsum(sorted_probs, axis=-1)
                # Zero out tokens beyond top_p
                mask = cumsum - sorted_probs > top_p
                sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)
                probs = mx.softmax(sorted_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=-1)
            else:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)

            generated.append(next_token)

            if next_token.item() == eos_token_id:
                break

            tokens = next_token

        return mx.concatenate([prompt_tokens] + generated, axis=-1) if generated else prompt_tokens

    @property
    def num_params(self) -> int:
        """Count actual model parameters."""
        nparams = sum(v.size for _, v in nn.utils.tree_flatten(self.parameters()))
        return nparams


def create_model(config: OKRModelConfig | None = None) -> OKRMicroModel:
    """Create model with default ASMS Medium config."""
    if config is None:
        config = OKRModelConfig()
    model = OKRMicroModel(config)
    return model


if __name__ == "__main__":
    config = OKRModelConfig()
    print(f"OKR Micro-Transformer Config:")
    print(f"  Layers:     {config.num_layers}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Heads:      {config.num_heads}")
    print(f"  FFN dim:    {config.ffn_dim}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max seq:    {config.max_seq_len}")
    print(f"  Est params: {config.param_count_estimate:,}")

    model = create_model(config)
    mx.eval(model.parameters())
    actual = model.num_params
    print(f"  Real params: {actual:,}")
    print(f"  Size (FP16): ~{actual * 2 / 1e6:.1f} MB")
    print(f"  Size (INT4): ~{actual * 0.5 / 1e6:.1f} MB")

    # Test forward pass
    batch = mx.zeros((1, 32), dtype=mx.int32)
    logits, _ = model(batch)
    print(f"\n  Forward pass: input {batch.shape} -> logits {logits.shape}")
    print(f"  Metal GPU: {mx.metal.is_available()}")
