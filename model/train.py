"""
ASMS Stage 5: Curriculum Training Loop on MLX

Trains the OKR micro-transformer from scratch with curriculum learning:
  Phase 1 (50%): Normal cases — learn primary decision boundaries
  Phase 2 (35%): Edge cases — refine boundary behaviour
  Phase 3 (15%): Adversarial cases — harden constraint adherence

Uses AdamW optimizer with cosine learning rate schedule.
Runs entirely on Apple Silicon via MLX Metal acceleration.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import sentencepiece as spm

from architecture import OKRModelConfig, create_model

MODEL_DIR = Path(__file__).parent
CORPUS_DIR = Path(__file__).parent.parent / "corpus" / "data"
TOKENIZER_PATH = MODEL_DIR / "tokenizer" / "okr_tokenizer.model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


def load_tokenizer() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(str(TOKENIZER_PATH))
    return sp


def format_example(example: dict) -> str:
    """Convert a corpus example into a training sequence:
    <bos>QUERY: {query} CONTEXT: {context}<tool>{tool_call_json}</tool><eos>
    """
    query = example.get("input", {}).get("query", "")
    context = json.dumps(example.get("input", {}).get("session_context", {}))
    tool_calls = json.dumps(example.get("tool_calls", []))
    methodology = json.dumps(example.get("methodology_notes", {}))
    workflow = example.get("workflow", "")

    return (
        f"QUERY: {query} CONTEXT: {context} "
        f"<workflow>{workflow}</workflow> "
        f"<tool>{tool_calls}</tool> "
        f"<score>{methodology}</score>"
    )


def load_dataset(
    corpus_file: str, sp: spm.SentencePieceProcessor, max_seq_len: int
) -> dict[str, list[list[int]]]:
    """Load and tokenize corpus, split by category for curriculum learning."""
    corpus_path = CORPUS_DIR / corpus_file
    categories = {"normal": [], "edge": [], "adversarial": []}

    with open(corpus_path) as f:
        for line in f:
            ex = json.loads(line)
            category = ex.get("_meta", {}).get("category", "normal")
            text = format_example(ex)
            # Tokenize: prepend BOS (2), append EOS (3)
            token_ids = [sp.bos_id()] + sp.Encode(text) + [sp.eos_id()]
            # Truncate or pad to max_seq_len
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            else:
                token_ids = token_ids + [sp.pad_id()] * (max_seq_len - len(token_ids))
            categories[category].append(token_ids)

    for cat, seqs in categories.items():
        print(f"  {cat}: {len(seqs)} sequences")

    return categories


def create_batches(
    sequences: list[list[int]], batch_size: int, shuffle: bool = True
) -> list[tuple[mx.array, mx.array]]:
    """Create (input, target) batches for next-token prediction."""
    if shuffle:
        indices = np.random.permutation(len(sequences))
        sequences = [sequences[i] for i in indices]

    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        if len(batch_seqs) < batch_size:
            continue  # drop incomplete last batch
        tokens = mx.array(batch_seqs)
        inputs = tokens[:, :-1]   # all tokens except last
        targets = tokens[:, 1:]   # all tokens except first (shifted right)
        batches.append((inputs, targets))

    return batches


def cosine_schedule(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def loss_fn(model, inputs, targets, pad_id: int = 0):
    """Cross-entropy loss with padding mask."""
    logits, _ = model(inputs)
    # Reshape for cross entropy: (B*L, vocab) and (B*L,)
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Compute per-token cross entropy
    log_probs = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Mask padding tokens
    mask = (targets_flat != pad_id).astype(log_probs.dtype)
    masked_loss = (log_probs * mask).sum() / mask.sum()
    return masked_loss


def train(
    corpus_file: str = "corpus.jsonl",
    batch_size: int = 32,
    epochs: int = 20,
    lr_max: float = 3e-4,
    lr_min: float = 1e-5,
    warmup_steps: int = 100,
    eval_every: int = 50,
    checkpoint_every: int = 500,
    config: OKRModelConfig | None = None,
):
    print("=" * 60)
    print("ASMS Stage 5: OKR Micro-Model Training")
    print("=" * 60)

    # Setup
    if config is None:
        config = OKRModelConfig()
    sp = load_tokenizer()
    config.vocab_size = sp.GetPieceSize()
    print(f"\nTokenizer vocab: {config.vocab_size}")

    model = create_model(config)
    mx.eval(model.parameters())
    print(f"Model params: {model.num_params:,}")

    # Load data
    print(f"\nLoading corpus from {corpus_file}...")
    categories = load_dataset(corpus_file, sp, config.max_seq_len)

    # Split each category into train/val (90/10)
    train_seqs = []
    val_seqs = []
    for cat, seqs in categories.items():
        split = int(len(seqs) * 0.9)
        train_seqs.append((cat, seqs[:split]))
        val_seqs.extend(seqs[split:])

    total_train = sum(len(s) for _, s in train_seqs)
    print(f"Train: {total_train}, Val: {len(val_seqs)}")

    # Curriculum phases
    curriculum = [
        {"name": "Phase 1: Normal", "epochs_pct": 0.50, "categories": ["normal"]},
        {"name": "Phase 2: + Edge", "epochs_pct": 0.35, "categories": ["normal", "edge"]},
        {"name": "Phase 3: + Adversarial", "epochs_pct": 0.15, "categories": ["normal", "edge", "adversarial"]},
    ]

    # Calculate total steps for LR schedule
    total_steps = 0
    for phase in curriculum:
        phase_epochs = max(1, int(epochs * phase["epochs_pct"]))
        phase_seqs = []
        for cat, seqs in train_seqs:
            if cat in phase["categories"]:
                phase_seqs.extend(seqs)
        steps_per_epoch = len(phase_seqs) // batch_size
        total_steps += phase_epochs * steps_per_epoch

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=lr_max, weight_decay=0.01)

    # Training state
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")
    start_time = time.time()

    for phase in curriculum:
        phase_name = phase["name"]
        phase_epochs = max(1, int(epochs * phase["epochs_pct"]))
        phase_cats = phase["categories"]

        # Collect sequences for this phase
        phase_seqs = []
        for cat, seqs in train_seqs:
            if cat in phase_cats:
                phase_seqs.extend(seqs)

        print(f"\n{'─' * 60}")
        print(f"{phase_name} ({phase_epochs} epochs, {len(phase_seqs)} sequences)")
        print(f"{'─' * 60}")

        for epoch in range(phase_epochs):
            batches = create_batches(phase_seqs, batch_size, shuffle=True)
            epoch_loss = 0.0
            n_batches = 0

            for inputs, targets in batches:
                # Update learning rate
                lr = cosine_schedule(global_step, warmup_steps, total_steps, lr_max, lr_min)
                optimizer.learning_rate = lr

                # Forward + backward
                loss, grads = loss_and_grad_fn(model, inputs, targets, sp.pad_id())
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                # Logging
                if global_step % eval_every == 0:
                    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                    elapsed = time.time() - start_time
                    steps_sec = global_step / elapsed
                    print(
                        f"  step {global_step:5d} | loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | {steps_sec:.1f} steps/s"
                    )

                # Checkpoint
                if global_step % checkpoint_every == 0:
                    save_checkpoint(model, config, global_step)

            # Epoch validation
            if val_seqs:
                val_loss = evaluate(model, val_seqs, batch_size, sp.pad_id())
                avg_train = epoch_loss / max(1, n_batches)
                print(
                    f"  [{phase_name}] epoch {epoch+1}/{phase_epochs} | "
                    f"train_loss {avg_train:.4f} | val_loss {val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, config, global_step, tag="best")
                    print(f"  ★ New best model (val_loss={val_loss:.4f})")

    # Final save
    save_checkpoint(model, config, global_step, tag="final")
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total steps: {global_step}")
    print(f"{'=' * 60}")


def evaluate(model, sequences: list[list[int]], batch_size: int, pad_id: int) -> float:
    """Run evaluation on held-out sequences."""
    batches = create_batches(sequences, batch_size, shuffle=False)
    total_loss = 0.0
    n = 0
    for inputs, targets in batches:
        logits, _ = model(inputs)
        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        log_probs = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        mask = (targets_flat != pad_id).astype(log_probs.dtype)
        batch_loss = (log_probs * mask).sum() / mask.sum()
        total_loss += batch_loss.item()
        n += 1
    return total_loss / max(1, n)


def save_checkpoint(model, config: OKRModelConfig, step: int, tag: str | None = None):
    """Save model weights and config."""
    name = f"step_{step}" if tag is None else tag
    path = CHECKPOINT_DIR / name
    path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)

    # Save config
    config_dict = {
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "num_layers": config.num_layers,
        "hidden_dim": config.hidden_dim,
        "num_heads": config.num_heads,
        "ffn_dim": config.ffn_dim,
        "dropout": config.dropout,
        "rope_theta": config.rope_theta,
        "step": step,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path: str | Path) -> tuple:
    """Load model from checkpoint."""
    path = Path(path)
    with open(path / "config.json") as f:
        config_dict = json.load(f)

    config = OKRModelConfig(
        vocab_size=config_dict["vocab_size"],
        max_seq_len=config_dict["max_seq_len"],
        num_layers=config_dict["num_layers"],
        hidden_dim=config_dict["hidden_dim"],
        num_heads=config_dict["num_heads"],
        ffn_dim=config_dict["ffn_dim"],
        dropout=config_dict.get("dropout", 0.1),
        rope_theta=config_dict.get("rope_theta", 10000.0),
    )

    model = create_model(config)
    weights = mx.load(str(path / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASMS Stage 5: Train OKR Micro-Model")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Corpus JSONL file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save every N steps")
    args = parser.parse_args()

    train(
        corpus_file=args.corpus,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_max=args.lr,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )
