import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_run.py"
WEIGHTS_BYTES = b"fake adapter weights"


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
    "prompts_file": "prompts/compare_ja.json",
    "prompts_sha256": json.loads(
        (REPO_ROOT / "prompts" / "compare_ja.contamination.json").read_text()
    )["prompts_sha256"],
    "generation": {"max_new_tokens": 256},
    "adapter_load": {
        "strict": True,
        "weights_file": "lora_weights.pt",
        "weights_sha256": hashlib.sha256(WEIGHTS_BYTES).hexdigest(),
        "lora_modules": 4,
        "adapter_tensors": 8,
        "checkpoint_tensors": 8,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_tensors": 0,
        "lora_B_norm_before_load": 0.0,
        "lora_B_norm_after_load": 1.5,
        "model_in_eval_mode": True,
    },
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


GOOD_MASKING = {
    "tokenizer": "Qwen/Qwen2.5-7B-Instruct",
    "ok": True,
    "missing_cases": [],
    "samples": [
        {"case": case, "row_index": str(i), "ok": True}
        for i, case in enumerate(
            [
                "no_context",
                "with_context",
                "context_truncated",
                "response_truncated",
                "forced_truncation",
            ]
        )
    ],
}


def _run_dir(
    tmp_path, metrics=GOOD_METRICS, samples=GOOD_SAMPLES, masking=GOOD_MASKING
) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "metrics.json").write_text(json.dumps(metrics))
    if samples is not None:
        (run / "samples.json").write_text(json.dumps(samples))
        (run / "samples.md").write_text("# Before/After samples\n\n## 1. a\n")
    if masking is not None:
        (run / "masking_check.json").write_text(json.dumps(masking))
    (run / "lora_config.json").write_text("{}")
    (run / "loss_curve.png").write_bytes(b"png")
    (run / "lora_weights.pt").write_bytes(WEIGHTS_BYTES)
    return run


def _statuses(run: Path) -> dict[str, str]:
    return {c.name: c.status for c in check_run.run_checks(run)}


def test_check_run_passes_good_pilot(tmp_path, capsys) -> None:
    run = _run_dir(tmp_path)

    assert check_run.main([str(run)]) == 0
    assert set(_statuses(run).values()) == {"PASS"}
    out = capsys.readouterr().out
    assert "GATE PASS" in out
    assert "not a quality judgement" in out


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
        (
            _mutated(
                train_loss_steps=[
                    {"step": 10, "epoch": 1, "loss": 2.0, "lr": 0.1},
                    {"step": 20, "epoch": 1, "loss": None, "lr": 0.1},
                    {"step": 30, "epoch": 1, "loss": 1.0, "lr": 0.1},
                ]
            ),
            "loss-decreases",
        ),
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


def test_check_run_requires_passing_masking_report(tmp_path) -> None:
    assert _statuses(_run_dir(tmp_path, masking=None))["mask-boundary"] == "FAIL"

    failed = {**GOOD_MASKING, "ok": False, "samples": [{"row_index": "7", "ok": False}]}
    assert _statuses(_run_dir(tmp_path, masking=failed))["mask-boundary"] == "FAIL"

    no_context_only = {**GOOD_MASKING, "samples": GOOD_MASKING["samples"][:1]}
    assert (
        _statuses(_run_dir(tmp_path, masking=no_context_only))["mask-boundary"]
        == "FAIL"
    )


def test_check_run_warns_on_little_vram_headroom(tmp_path) -> None:
    run = _run_dir(tmp_path, metrics=_mutated(memory__max_memory_reserved_gib=46.0))

    assert _statuses(run)["vram-headroom"] == "WARN"


def test_check_run_fails_on_empty_base_output(tmp_path) -> None:
    samples = copy.deepcopy(GOOD_SAMPLES)
    samples["samples"][0]["base_output"] = " "

    assert _statuses(_run_dir(tmp_path, samples=samples))["generation"] == "FAIL"


def test_check_run_fails_without_loss_curve(tmp_path) -> None:
    run = _run_dir(tmp_path)
    (run / "loss_curve.png").unlink()

    assert _statuses(run)["evidence"] == "FAIL"


def test_check_run_fails_when_hostname_is_recorded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(check_run.socket, "gethostname", lambda: "labhost01.example")
    metrics = _mutated(config__note="trained on labhost01")

    checks = {c.name: c for c in check_run.run_checks(_run_dir(tmp_path, metrics))}

    assert checks["evidence"].status == "FAIL"
    assert "hostname" in checks["evidence"].detail


def test_check_run_ignores_hostname_inside_generated_text(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(check_run.socket, "gethostname", lambda: "labhost01")
    samples = copy.deepcopy(GOOD_SAMPLES)
    samples["samples"][0]["finetuned_output"] = "labhost01 について"

    assert _statuses(_run_dir(tmp_path, samples=samples))["evidence"] == "PASS"


def test_check_run_fails_when_username_is_recorded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(check_run.getpass, "getuser", lambda: "labuser")
    metrics = _mutated(config__owner="labuser")

    checks = {c.name: c for c in check_run.run_checks(_run_dir(tmp_path, metrics))}

    assert checks["evidence"].status == "FAIL"
    assert "username" in checks["evidence"].detail


def test_check_run_prompt_overlap_requires_matching_report(tmp_path) -> None:
    samples = copy.deepcopy(GOOD_SAMPLES)
    assert check_run.check_prompt_overlap(samples).status == "PASS"

    samples["prompts_sha256"] = "0" * 64
    assert check_run.check_prompt_overlap(samples).status == "FAIL"

    other = {**GOOD_SAMPLES, "prompts_file": "prompts/other.json"}
    assert check_run.check_prompt_overlap(other).status == "FAIL"

    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "compare_ja.contamination.json").write_text(
        json.dumps(
            {
                "ok": False,
                "prompts_sha256": GOOD_SAMPLES["prompts_sha256"],
                "results": [{"id": "a", "ok": False}],
            }
        )
    )
    assert check_run.check_prompt_overlap(GOOD_SAMPLES, tmp_path).status == "FAIL"


def _adapter(**changes) -> dict:
    samples = copy.deepcopy(GOOD_SAMPLES)
    samples["adapter_load"].update(changes)
    return samples


@pytest.mark.parametrize(
    "changes",
    [
        {"strict": False},
        {"lora_modules": 0, "adapter_tensors": 0, "checkpoint_tensors": 0},
        {"checkpoint_tensors": 6},
        {"missing_keys": 2},
        {"unexpected_keys": 1},
        {"mismatched_tensors": 1},
        {"lora_B_norm_before_load": 0.3},
        {"lora_B_norm_after_load": 0.0},
        {"model_in_eval_mode": False},
        {"weights_sha256": "0" * 64},
    ],
)
def test_check_run_adapter_load_fails_on_bad_evidence(tmp_path, changes) -> None:
    run = _run_dir(tmp_path, samples=_adapter(**changes))

    statuses = _statuses(run)

    assert statuses["adapter-load"] == "FAIL"
    # Base/fine-tuned generation is judged separately
    assert statuses["generation"] == "PASS"


def test_check_run_adapter_load_requires_evidence(tmp_path) -> None:
    samples = copy.deepcopy(GOOD_SAMPLES)
    del samples["adapter_load"]

    assert _statuses(_run_dir(tmp_path, samples=samples))["adapter-load"] == "FAIL"


def test_check_run_adapter_load_warns_without_weights_file(tmp_path) -> None:
    run = _run_dir(tmp_path)
    (run / "lora_weights.pt").unlink()

    assert _statuses(run)["adapter-load"] == "WARN"
