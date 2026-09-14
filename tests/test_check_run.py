import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_run.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_run = _load_script()

GOOD_METRICS = {
    "status": "completed",
    "error": None,
    "config": {"val_ratio": 0.02, "model_id": "Qwen/Qwen2.5-7B-Instruct"},
    "environment": {
        "cuda": "12.1",
        "gpus": [{"index": 0, "name": "GPU-X", "total_memory_gib": 47.5}],
        "git_commit": "abc1234",
        "git_dirty": False,
    },
    "epochs": [{"epoch": 1, "step": 40, "train_loss": 1.5, "val_loss": 1.4}],
    "train_loss_steps": [
        {"step": s, "epoch": 1, "loss": 2.0 - 0.02 * s, "lr": 1e-4}
        for s in range(10, 41, 10)
    ],
    "runtime": {"total_seconds": 12.0, "tokens_trained": 300, "input_tokens": 900},
    "memory": {"max_memory_allocated_gib": 20.0, "max_memory_reserved_gib": 22.0},
}
GOOD_SAMPLES = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "generation": {"max_new_tokens": 256},
    "samples": [
        {
            "id": "a",
            "instruction": "q",
            "base_output": "base answer",
            "base_new_tokens": 5,
            "finetuned_output": "tuned answer",
            "finetuned_new_tokens": 5,
        }
    ],
}


def _run_dir(tmp_path, metrics=GOOD_METRICS, samples=GOOD_SAMPLES) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "metrics.json").write_text(json.dumps(metrics))
    if samples is not None:
        (run / "samples.json").write_text(json.dumps(samples))
    (run / "lora_config.json").write_text("{}")
    return run


def _statuses(run: Path) -> dict[str, str]:
    return {c.name: c.status for c in check_run.run_checks(run)}


def test_check_run_passes_good_pilot(tmp_path, capsys) -> None:
    run = _run_dir(tmp_path)

    assert check_run.main([str(run)]) == 0
    assert set(_statuses(run).values()) == {"PASS"}
    assert "GO" in capsys.readouterr().out


def _mutated(**changes):
    metrics = copy.deepcopy(GOOD_METRICS)
    for dotted, value in changes.items():
        *parents, leaf = dotted.split("__")
        target = metrics
        for key in parents:
            target = target[key]
        target[leaf] = value
    return metrics


@pytest.mark.parametrize(
    ("metrics", "failed_check"),
    [
        (_mutated(status="failed", error="OutOfMemoryError"), "run-completed"),
        (
            _mutated(
                train_loss_steps=[
                    {"step": 10, "epoch": 1, "loss": 1.0, "lr": 0.1},
                    {"step": 20, "epoch": 1, "loss": 1.2, "lr": 0.1},
                ]
            ),
            "loss-decreases",
        ),
        (_mutated(runtime__tokens_trained=900), "response-mask"),
        (_mutated(environment__gpus=None), "evidence"),
        (_mutated(memory__max_memory_allocated_gib=None), "evidence"),
        (_mutated(config__model_id="/cache/models/qwen"), "evidence"),
    ],
)
def test_check_run_fails_on_bad_evidence(tmp_path, metrics, failed_check) -> None:
    run = _run_dir(tmp_path, metrics=metrics)

    assert _statuses(run)[failed_check] == "FAIL"
    assert check_run.main([str(run)]) == 1


def test_check_run_flags_local_base_model_path_in_lora_config(tmp_path) -> None:
    run = _run_dir(tmp_path)
    (run / "lora_config.json").write_text(json.dumps({"base_model_id": "/cache/qwen"}))

    assert _statuses(run)["evidence"] == "FAIL"


def test_check_run_requires_samples(tmp_path) -> None:
    run = _run_dir(tmp_path, samples=None)

    assert _statuses(run)["generation"] == "FAIL"


def test_check_run_warns_on_degenerate_generation(tmp_path) -> None:
    samples = copy.deepcopy(GOOD_SAMPLES)
    samples["samples"][0]["finetuned_output"] = "同じ文です。" * 30
    samples["samples"][0]["finetuned_new_tokens"] = 256
    run = _run_dir(tmp_path, samples=samples)

    checks = {c.name: c for c in check_run.run_checks(run)}

    assert checks["generation"].status == "WARN"
    assert "repetitive" in checks["generation"].detail
    assert "cut off" in checks["generation"].detail
    assert check_run.main([str(run)]) == 0


def test_check_run_warns_on_dirty_git(tmp_path) -> None:
    run = _run_dir(tmp_path, metrics=_mutated(environment__git_dirty=True))

    assert _statuses(run)["evidence"] == "WARN"


def test_check_run_help_runs_as_script() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0
    assert "run_dir" in result.stdout
