"""Plot the training loss curve stored in a metrics.json written by src.train.

Usage:
    uv run python scripts/plot_metrics.py checkpoints/<run>/metrics.json \
        -o checkpoints/<run>/loss_curve.png

Train loss is drawn per logged optimizer step (``train_loss_steps``) and
validation loss once per epoch at the epoch's last optimizer step. Metrics
files written before step logging existed fall back to per-epoch train loss.
The PNG carries no timestamp or software metadata, so the same metrics.json
and matplotlib version produce a byte-identical file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGURE_SIZE = (8.0, 4.5)
DPI = 150


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot step train loss and per-epoch validation loss "
        "from a metrics.json written by src.train"
    )
    p.add_argument("metrics", type=Path, help="Path to metrics.json")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: loss_curve.png next to metrics.json)",
    )
    return p.parse_args(argv)


def _points(entries: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    pairs = [(e["step"], e[key]) for e in entries if e.get(key) is not None]
    return [step for step, _ in pairs], [value for _, value in pairs]


def _title(metrics: dict[str, Any]) -> str:
    config = metrics.get("config", {})
    parts = [str(config.get("model_id", "model"))]
    if config.get("rank") is not None:
        parts.append(f"LoRA r={config['rank']} alpha={config.get('alpha')}")
    status = metrics.get("status")
    if status and status != "completed":
        parts.append(f"[{status}]")
    return " | ".join(parts)


def plot_metrics(metrics: dict[str, Any], output: Path) -> None:
    """Render the loss curve of ``metrics`` to ``output`` (PNG)."""
    epochs = metrics.get("epochs", [])
    step_entries = metrics.get("train_loss_steps") or []

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    if step_entries:
        steps, losses = _points(step_entries, "loss")
        ax.plot(steps, losses, linewidth=1.0, label="train loss (logged steps)")
    else:
        steps, losses = _points(epochs, "train_loss")
        ax.plot(steps, losses, marker="o", label="train loss (epoch mean)")

    val_steps, val_losses = _points(epochs, "val_loss")
    if val_steps:
        ax.plot(
            val_steps,
            val_losses,
            marker="o",
            linestyle="--",
            label="validation loss (per epoch)",
        )

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (response tokens)")
    ax.set_title(_title(metrics), fontsize=10)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Drop the "Software" tEXt chunk so the file only depends on the data
    fig.savefig(output, dpi=DPI, format="png", metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    output = args.output or args.metrics.with_name("loss_curve.png")
    plot_metrics(metrics, output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
