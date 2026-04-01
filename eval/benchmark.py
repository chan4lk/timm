"""
ASMS Evaluation: Benchmark Suite

Evaluates the OKR micro-model against ASMS paper metrics:
  1. Task Accuracy — correct workflow + tool selection
  2. Latency (p50/p99) — inference speed on Metal
  3. Model Size — quantized footprint
  4. Robustness — accuracy on adversarial/edge test set
  5. Confidence Calibration — fallback routing accuracy
"""

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))

from inference import OKRInference
from keyflow_bridge import KeyflowBridge

TEST_DIR = Path(__file__).parent / "test_sets"


def load_test_set(filename: str) -> list[dict]:
    path = TEST_DIR / filename
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate_accuracy(engine: OKRInference, test_set: list[dict]) -> dict:
    """Evaluate workflow routing and tool selection accuracy."""
    correct_workflow = 0
    correct_tools = 0
    total = len(test_set)

    for ex in test_set:
        query = ex["input"]["query"]
        context = ex["input"].get("session_context")
        expected_workflow = ex.get("workflow")
        expected_tools = set()
        for tc in ex.get("tool_calls", []):
            expected_tools.add(f"{tc['tool']}.{tc['action']}")

        result = engine.predict(query, context, temperature=0.0)

        # Check workflow match
        if result.get("workflow") == expected_workflow:
            correct_workflow += 1

        # Check tool set match
        predicted_tools = set()
        for tc in result.get("tool_calls", []):
            if not tc.get("parse_error"):
                predicted_tools.add(f"{tc.get('tool')}.{tc.get('action')}")
        if predicted_tools == expected_tools:
            correct_tools += 1

    return {
        "workflow_accuracy": correct_workflow / total if total else 0,
        "tool_accuracy": correct_tools / total if total else 0,
        "total": total,
    }


def evaluate_latency(engine: OKRInference, n_runs: int = 100) -> dict:
    """Measure inference latency distribution."""
    # Warmup
    for _ in range(5):
        engine.predict("Show me my OKRs", temperature=0.0)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        engine.predict(
            "I want to improve customer retention by 20% this quarter",
            session_context={"userId": "u1", "activeCycleId": "c1"},
            temperature=0.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "p50_ms": round(statistics.median(latencies), 2),
        "p90_ms": round(latencies[int(0.9 * len(latencies))], 2),
        "p99_ms": round(latencies[int(0.99 * len(latencies))], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "n_runs": n_runs,
    }


def evaluate_model_size(checkpoint_path: str) -> dict:
    """Report model file sizes."""
    path = Path(checkpoint_path)
    model_file = path / "model.safetensors"
    config_file = path / "config.json"

    model_size = model_file.stat().st_size if model_file.exists() else 0
    config_size = config_file.stat().st_size if config_file.exists() else 0

    # Check for quantization
    with open(config_file) as f:
        config = json.load(f)
    quant = config.get("quantization", {})

    return {
        "model_size_mb": round(model_size / 1e6, 2),
        "config_size_bytes": config_size,
        "total_size_mb": round((model_size + config_size) / 1e6, 2),
        "quantization": quant,
    }


def evaluate_robustness(
    engine: OKRInference, bridge: KeyflowBridge, test_set: list[dict]
) -> dict:
    """Evaluate on adversarial/edge cases with validation."""
    valid_outputs = 0
    fallbacks = 0
    total = len(test_set)

    for ex in test_set:
        query = ex["input"]["query"]
        context = ex["input"].get("session_context")

        result = engine.predict(query, context, temperature=0.0)
        confidence = engine.confidence_score(result)

        if bridge.should_fallback(result, confidence):
            fallbacks += 1
        else:
            # Check if all tool calls validate
            all_valid = True
            for tc in result.get("tool_calls", []):
                valid, _ = bridge.validate_tool_call(tc)
                if not valid:
                    all_valid = False
                    break
            if all_valid:
                valid_outputs += 1

    return {
        "valid_rate": valid_outputs / total if total else 0,
        "fallback_rate": fallbacks / total if total else 0,
        "total": total,
    }


def run_full_benchmark(checkpoint_path: str, test_file: str = "test.jsonl") -> dict:
    """Run complete ASMS benchmark suite."""
    print("=" * 60)
    print("ASMS Benchmark: OKR Micro-Model")
    print("=" * 60)

    engine = OKRInference(checkpoint_path)
    bridge = KeyflowBridge()

    results = {}

    # 1. Model size
    print("\n--- Model Size ---")
    results["size"] = evaluate_model_size(checkpoint_path)
    print(f"  Total: {results['size']['total_size_mb']} MB")
    print(f"  Quant: {results['size']['quantization']}")

    # 2. Latency
    print("\n--- Latency (100 runs) ---")
    results["latency"] = evaluate_latency(engine, n_runs=100)
    print(f"  p50: {results['latency']['p50_ms']}ms")
    print(f"  p99: {results['latency']['p99_ms']}ms")

    # 3. Accuracy (if test set exists)
    test_path = TEST_DIR / test_file
    if test_path.exists():
        print("\n--- Accuracy ---")
        test_set = load_test_set(test_file)
        results["accuracy"] = evaluate_accuracy(engine, test_set)
        print(f"  Workflow: {results['accuracy']['workflow_accuracy']*100:.1f}%")
        print(f"  Tools: {results['accuracy']['tool_accuracy']*100:.1f}%")

        # 4. Robustness (adversarial subset)
        adversarial = [ex for ex in test_set if ex.get("_meta", {}).get("category") == "adversarial"]
        if adversarial:
            print("\n--- Robustness (adversarial) ---")
            results["robustness"] = evaluate_robustness(engine, bridge, adversarial)
            print(f"  Valid: {results['robustness']['valid_rate']*100:.1f}%")
            print(f"  Fallback: {results['robustness']['fallback_rate']*100:.1f}%")
    else:
        print(f"\n  (No test set at {test_path}, skipping accuracy/robustness)")

    # Summary
    print(f"\n{'=' * 60}")
    print("ASMS Targets vs Results:")
    print(f"  Model size:  target <50MB,  actual {results['size']['total_size_mb']}MB")
    print(f"  Latency p50: target <5ms,   actual {results['latency']['p50_ms']}ms")
    print(f"  Latency p99: target <10ms,  actual {results['latency']['p99_ms']}ms")
    if "accuracy" in results:
        print(f"  Accuracy:    target >95%,   actual {results['accuracy']['tool_accuracy']*100:.1f}%")
    print("=" * 60)

    # Save report
    report_path = Path(__file__).parent / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASMS Benchmark Suite")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--test-file", default="test.jsonl", help="Test set JSONL file")
    args = parser.parse_args()

    run_full_benchmark(args.checkpoint, args.test_file)
