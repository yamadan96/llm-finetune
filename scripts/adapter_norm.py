"""Measure how far a LoRA adapter has moved the base model.

Usage:
    uv run python scripts/adapter_norm.py checkpoints/<run> [more runs ...] \
        -o adapter_norms.md --json adapter_norms.json

For each checkpoint directory (or step-* snapshot directory) it loads
``lora_weights.pt`` and ``lora_config.json`` and computes, per LoRA layer,

    dW = (alpha / rank) * B @ A

and reports the Frobenius norm of dW summed over layers, its mean and maximum
per layer, and the same numbers split into q_proj and v_proj. The base model
is never loaded, so this runs on CPU in seconds and needs no GPU.

The point of the measure: a learning-rate sweep and a step sweep both move the
adapter, and ||dW|| puts them on one axis.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from src.evidence import write_json
from src.lora import LORA_CONFIG_FILENAME, LORA_WEIGHTS_FILENAME, load_lora_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frobenius norm of the LoRA update")
    p.add_argument("runs", nargs="+", type=Path, help="Checkpoint directories")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args(argv)


def layer_updates(state: dict[str, torch.Tensor], scaling: float) -> dict[str, float]:
    """Frobenius norm of (alpha/rank) B A for every adapted layer."""
    norms: dict[str, float] = {}
    for name in sorted(state):
        if not name.endswith("lora_A"):
            continue
        layer = name[: -len("lora_A")]
        b_key = f"{layer}lora_B"
        if b_key not in state:
            continue
        a = state[name].float()
        b = state[b_key].float()
        norms[layer.rstrip(".")] = float(torch.linalg.matrix_norm(b @ a) * scaling)
    return norms


def _module_of(layer: str) -> str:
    return layer.rsplit(".", 1)[-1] if "." in layer else layer


def adapter_norm(run_dir: Path) -> dict[str, Any]:
    config = load_lora_config(run_dir / LORA_CONFIG_FILENAME) or {}
    rank = config.get("rank")
    alpha = config.get("alpha")
    if not rank or alpha is None:
        raise ValueError(f"{run_dir}: {LORA_CONFIG_FILENAME} lacks rank/alpha")
    state = torch.load(
        run_dir / LORA_WEIGHTS_FILENAME, map_location="cpu", weights_only=True
    )
    norms = layer_updates(state, alpha / rank)
    per_module: dict[str, list[float]] = {}
    for layer, value in norms.items():
        per_module.setdefault(_module_of(layer), []).append(value)
    values = list(norms.values())
    return {
        "run": run_dir.name,
        "rank": rank,
        "alpha": alpha,
        "layers": len(values),
        "total_norm": sum(values),
        "mean_norm": sum(values) / len(values) if values else 0.0,
        "max_norm": max(values) if values else 0.0,
        "per_module": {
            module: {
                "layers": len(v),
                "total_norm": sum(v),
                "mean_norm": sum(v) / len(v),
            }
            for module, v in sorted(per_module.items())
        },
    }


def render(results: list[dict[str, Any]]) -> str:
    modules = sorted({m for r in results for m in r["per_module"]})
    lines = [
        "# LoRA update magnitude",
        "",
        "`dW = (alpha / rank) * B @ A` per adapted layer; the base model is not "
        "loaded, so these are the sizes of the update, not relative changes.",
        "",
        "| run | rank | alpha | layers | total ||dW|| | mean | max | "
        + " | ".join(f"{m} mean" for m in modules)
        + " |",
        "|" + "---|" * (7 + len(modules)),
    ]
    for r in results:
        cells = [
            r["run"],
            str(r["rank"]),
            f"{r['alpha']:g}",
            str(r["layers"]),
            f"{r['total_norm']:.2f}",
            f"{r['mean_norm']:.3f}",
            f"{r['max_norm']:.3f}",
        ]
        cells += [
            f"{r['per_module'][m]['mean_norm']:.3f}" if m in r["per_module"] else "–"
            for m in modules
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = [adapter_norm(run) for run in args.runs]
    markdown = render(results)
    if args.json:
        write_json(args.json, results)
        print(f"Wrote {args.json}")
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
