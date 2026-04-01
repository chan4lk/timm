"""
ASMS Stage 6b: Fast Inference Wrapper

Loads the quantized OKR micro-model and provides a clean inference API.
Supports both single-shot and streaming generation with KV caching.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import sentencepiece as spm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from architecture import OKRModelConfig, create_model


class OKRInference:
    """Fast inference engine for the OKR micro-model."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)

        # Load config
        with open(self.checkpoint_path / "config.json") as f:
            config_dict = json.load(f)

        self.config = OKRModelConfig(**{
            k: v for k, v in config_dict.items()
            if k in OKRModelConfig.__dataclass_fields__
        })

        # Load model
        self.model = create_model(self.config)
        weights = mx.load(str(self.checkpoint_path / "model.safetensors"))

        # Re-apply quantization if model was quantized
        quant_config = config_dict.get("quantization")
        if quant_config:
            nn.quantize(
                self.model,
                bits=quant_config["bits"],
                group_size=quant_config["group_size"],
            )

        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())

        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = str(
                Path(__file__).parent.parent / "model" / "tokenizer" / "okr_tokenizer.model"
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

        print(f"Loaded OKR model: {self.model.num_params:,} params")
        print(f"Quantized: {quant_config is not None}")

    def format_input(self, query: str, session_context: dict | None = None) -> str:
        """Format a user query into model input format."""
        context = json.dumps(session_context or {})
        return f"QUERY: {query} CONTEXT: {context} "

    def predict(
        self,
        query: str,
        session_context: dict | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict:
        """Run inference and return parsed tool calls."""
        start = time.perf_counter()

        # Tokenize
        text = self.format_input(query, session_context)
        token_ids = [self.sp.bos_id()] + self.sp.Encode(text)
        prompt = mx.array([token_ids])

        # Generate
        output = self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=self.sp.eos_id(),
        )
        mx.eval(output)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Decode
        output_ids = output[0].tolist()
        generated_ids = output_ids[len(token_ids):]
        generated_text = self.sp.Decode(generated_ids)

        # Parse structured output
        result = self._parse_output(generated_text)
        result["_inference"] = {
            "latency_ms": round(elapsed_ms, 2),
            "input_tokens": len(token_ids),
            "output_tokens": len(generated_ids),
            "raw_output": generated_text,
        }

        return result

    def _parse_output(self, text: str) -> dict:
        """Parse model output into structured tool calls."""
        result = {"workflow": None, "tool_calls": [], "methodology_notes": {}}

        # Extract workflow
        if "<workflow>" in text and "</workflow>" in text:
            start = text.index("<workflow>") + len("<workflow>")
            end = text.index("</workflow>")
            result["workflow"] = text[start:end].strip()

        # Extract tool calls
        if "<tool>" in text and "</tool>" in text:
            start = text.index("<tool>") + len("<tool>")
            end = text.index("</tool>")
            try:
                result["tool_calls"] = json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                result["tool_calls"] = [{"raw": text[start:end].strip(), "parse_error": True}]

        # Extract methodology notes
        if "<score>" in text and "</score>" in text:
            start = text.index("<score>") + len("<score>")
            end = text.index("</score>")
            try:
                result["methodology_notes"] = json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

        return result

    def confidence_score(self, result: dict) -> float:
        """Estimate confidence for fallback routing to Claude API."""
        score = 1.0

        # Penalize parse errors
        if any(tc.get("parse_error") for tc in result.get("tool_calls", [])):
            score -= 0.5

        # Penalize missing workflow
        if not result.get("workflow"):
            score -= 0.3

        # Penalize empty tool calls
        if not result.get("tool_calls"):
            score -= 0.3

        return max(0.0, score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OKR micro-model inference")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--query", default="I want to improve customer retention by 20%")
    args = parser.parse_args()

    engine = OKRInference(args.checkpoint)
    result = engine.predict(
        query=args.query,
        session_context={"userId": "user_1", "activeCycleId": "cyc_q2_2026"},
    )

    print(f"\nQuery: {args.query}")
    print(f"Workflow: {result['workflow']}")
    print(f"Tool calls: {json.dumps(result['tool_calls'], indent=2)}")
    print(f"Methodology: {json.dumps(result['methodology_notes'], indent=2)}")
    print(f"Latency: {result['_inference']['latency_ms']}ms")
    print(f"Confidence: {engine.confidence_score(result):.2f}")
