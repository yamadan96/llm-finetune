import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "plot_metrics.py"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _load_script():
    spec = importlib.util.spec_from_file_location("plot_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(with_steps: bool = True) -> dict:
    return {
        "status": "completed",
        "config": {"model_id": "tiny/model", "rank": 4, "alpha": 8.0},
        "epochs": [
            {"epoch": 1, "step": 4, "train_loss": 2.0, "val_loss": 1.9},
            {"epoch": 2, "step": 8, "train_loss": 1.5, "val_loss": None},
        ],
        "train_loss_steps": (
            [
                {"step": s, "epoch": 1 + (s > 4), "loss": 2.5 - 0.1 * s, "lr": 1e-4}
                for s in range(2, 9, 2)
            ]
            if with_steps
            else []
        ),
    }


@pytest.mark.parametrize("with_steps", [True, False])
def test_plot_metrics_writes_png_deterministically(tmp_path, with_steps) -> None:
    module = _load_script()
    metrics_path = tmp_path / "run" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps(_metrics(with_steps)))
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"

    assert module.main([str(metrics_path), "-o", str(first)]) == 0
    assert module.main([str(metrics_path), "-o", str(second)]) == 0

    data = first.read_bytes()
    assert data.startswith(PNG_SIGNATURE)
    assert data == second.read_bytes()
    assert b"Software" not in data


def test_plot_metrics_default_output_is_next_to_metrics(tmp_path) -> None:
    module = _load_script()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_metrics()))

    module.main([str(metrics_path)])

    assert (tmp_path / "loss_curve.png").read_bytes().startswith(PNG_SIGNATURE)


def test_plot_metrics_help_runs_as_script() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "metrics.json" in result.stdout
