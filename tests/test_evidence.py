import json
import subprocess

import pytest

import src.evidence as evidence
from src.evidence import (
    REDACTED_PATH,
    RunTracker,
    collect_environment,
    public_identifier,
    redact_paths,
    write_json,
)


@pytest.mark.parametrize(
    "value",
    [
        "/abs/cache/model",
        "~/models/qwen",
        "C:\\Users\\me\\model",
        "D:/data",
        "\\\\srv\\share",
    ],
)
def test_redact_paths_replaces_local_paths(value: str) -> None:
    assert redact_paths({"a": [value, {"b": value}]}) == {
        "a": [REDACTED_PATH, {"b": REDACTED_PATH}]
    }


@pytest.mark.parametrize(
    "value", ["Qwen/Qwen2.5-7B-Instruct", "NVIDIA RTX A6000", "2.5.1+cu121", "local:x"]
)
def test_redact_paths_keeps_regular_strings(value: str) -> None:
    assert redact_paths([value, 1.5, None]) == [value, 1.5, None]


def test_public_identifier_keeps_hub_ids_and_hides_local_dirs(tmp_path) -> None:
    local = tmp_path / "cache" / "my-model"
    local.mkdir(parents=True)

    assert public_identifier("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert public_identifier(str(local)) == "local:my-model"
    assert public_identifier("/does/not/exist/model") == "local:model"


def test_write_json_is_atomic_redacted_and_rejects_nan(tmp_path) -> None:
    path = tmp_path / "out.json"

    write_json(path, {"path": str(tmp_path), "ok": "value"})

    assert json.loads(path.read_text()) == {"path": REDACTED_PATH, "ok": "value"}
    assert not (tmp_path / "out.json.tmp").exists()
    with pytest.raises(ValueError):
        write_json(path, {"loss": float("nan")})
    # A failed write leaves the previous complete file in place
    assert json.loads(path.read_text())["ok"] == "value"


def test_collect_environment_tolerates_missing_git(monkeypatch) -> None:
    def broken_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evidence.subprocess, "run", broken_run)

    env = collect_environment()

    assert env["git_commit"] is None
    assert env["git_dirty"] is None
    assert env["torch"]


def test_collect_environment_tolerates_git_error(monkeypatch) -> None:
    def failing_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=128, stdout="", stderr="")

    monkeypatch.setattr(evidence.subprocess, "run", failing_run)

    assert collect_environment()["git_commit"] is None


def test_run_tracker_splits_train_and_eval_time(tmp_path) -> None:
    ticks = iter([0.0, 1.0, 2.0, 5.0, 5.0, 6.0, 8.0, 8.0, 9.0])
    tracker = RunTracker(
        tmp_path / "metrics.json", config={}, log_every=1, clock=lambda: next(ticks)
    )
    # writes consume ticks too: init write=1.0
    tracker.start_training()  # 2.0 -> setup 2s
    tracker.start_epoch()  # 5.0
    tracker.record_step(
        epoch=1, loss=1.0, lr=0.1, supervised_tokens=10, input_tokens=20
    )
    # record_step write reads 5.0 -> 0s of training so far
    tracker.end_epoch_training()  # 6.0 -> 1s training
    tracker.end_epoch({"epoch": 1, "val_loss": float("nan")})  # 8.0 eval 2s; write 8.0

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    runtime = metrics["runtime"]
    assert runtime["setup_seconds"] == 2.0
    assert runtime["train_seconds"] == 1.0
    assert runtime["eval_seconds"] == 2.0
    assert runtime["epoch_seconds"] == [3.0]
    assert runtime["tokens_per_second"] == 10.0
    assert metrics["epochs"] == [{"epoch": 1, "val_loss": None}]
