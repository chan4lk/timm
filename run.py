"""
ASMS Pipeline Runner

End-to-end script to run the full ASMS pipeline:
  1. Generate synthetic corpus (requires ANTHROPIC_API_KEY)
  2. Train tokenizer on corpus
  3. Train micro-model with curriculum learning
  4. Quantize to INT4
  5. Run benchmark
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).parent


def run_step(name: str, cmd: list[str], cwd: str | None = None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd or str(ROOT))
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ASMS Pipeline Runner")
    parser.add_argument("--corpus-size", type=int, default=100, help="Number of training examples")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model for corpus gen")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--skip-corpus", action="store_true", help="Skip corpus generation")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")
    args = parser.parse_args()

    print("ASMS Pipeline: OKR Micro-Model on MLX")
    print(f"Corpus size: {args.corpus_size}")
    print(f"Compiler model: {args.model}")
    print(f"Training epochs: {args.epochs}")

    # Stage 3: Corpus Generation
    if not args.skip_corpus:
        run_step(
            "Stage 3: Synthetic Corpus Generation",
            [PYTHON, "corpus/generate_corpus.py",
             "-n", str(args.corpus_size),
             "--model", args.model,
             "--skip-consistency",  # fast mode for initial runs
             "-o", "corpus.jsonl"],
        )

    # Stage 3.5: Tokenizer
    if not args.skip_tokenizer:
        run_step(
            "Stage 3.5: Train Tokenizer",
            [PYTHON, "model/tokenizer/train_tokenizer.py",
             "--corpus", "corpus.jsonl",
             "--vocab-size", "1500"],  # will scale with corpus
        )

    # Stage 5: Training
    if not args.skip_training:
        run_step(
            "Stage 5: Train Micro-Model",
            [PYTHON, "model/train.py",
             "--corpus", "corpus.jsonl",
             "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size)],
        )

    # Stage 6a: Quantization
    checkpoint_path = str(ROOT / "model" / "checkpoints" / "best")
    if not args.skip_quantize:
        if (ROOT / "model" / "checkpoints" / "best").exists():
            run_step(
                "Stage 6a: INT4 Quantization",
                [PYTHON, "deploy/quantize.py", checkpoint_path],
            )
        else:
            print("\n  Skipping quantization: no 'best' checkpoint found")

    # Benchmark
    quantized_path = str(ROOT / "model" / "checkpoints" / "best_q4")
    if not args.skip_benchmark:
        bench_path = quantized_path if Path(quantized_path).exists() else checkpoint_path
        if Path(bench_path).exists():
            run_step(
                "Benchmark: Evaluate Model",
                [PYTHON, "eval/benchmark.py", bench_path],
            )
        else:
            print("\n  Skipping benchmark: no checkpoint found")

    print(f"\n{'='*60}")
    print("  ASMS Pipeline Complete")
    print(f"{'='*60}")
    print(f"\nTo run inference:")
    print(f"  {PYTHON} deploy/inference.py model/checkpoints/best --query 'your query'")
    print(f"\nTo run the full pipeline with Claude API corpus:")
    print(f"  ANTHROPIC_API_KEY=... {PYTHON} run.py --corpus-size 1000")


if __name__ == "__main__":
    main()
