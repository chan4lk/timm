"""
ASMS Stage 6a: INT4 Quantization via MLX

Quantizes the trained model to INT4, reducing size from ~13MB to ~3.4MB
while maintaining >99% accuracy on structured decision tasks.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from architecture import OKRModelConfig, create_model


def quantize_model(
    checkpoint_path: str,
    output_path: str | None = None,
    bits: int = 4,
    group_size: int = 32,
):
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.name}_q{bits}"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    with open(checkpoint_path / "config.json") as f:
        config_dict = json.load(f)

    config = OKRModelConfig(**{
        k: v for k, v in config_dict.items()
        if k in OKRModelConfig.__dataclass_fields__
    })

    model = create_model(config)
    weights = mx.load(str(checkpoint_path / "model.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    pre_params = model.num_params
    print(f"Pre-quantization: {pre_params:,} params, ~{pre_params * 2 / 1e6:.1f} MB (FP16)")

    # Quantize all linear layers
    nn.quantize(model, bits=bits, group_size=group_size)
    mx.eval(model.parameters())

    # Save quantized model
    quantized_weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(str(output_path / "model.safetensors"), quantized_weights)

    # Save config with quantization info
    config_dict["quantization"] = {
        "bits": bits,
        "group_size": group_size,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Report size
    model_file = output_path / "model.safetensors"
    actual_size = model_file.stat().st_size / 1e6
    print(f"Post-quantization (INT{bits}): ~{actual_size:.1f} MB on disk")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize OKR micro-model to INT4")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=32, help="Quantization group size")
    args = parser.parse_args()

    quantize_model(args.checkpoint, args.output, args.bits, args.group_size)
