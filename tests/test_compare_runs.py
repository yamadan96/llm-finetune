import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compare_runs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_runs = _load_script()

METRICS = {
    "status": "completed",
    "config": {
        "lr": 2e-4,
        "seed": 42,
        "rank": 16,
        "max_train_samples": 500,
        "log_every": 10,
    },
    "epochs": [{"epoch": 1, "step": 40, "train_loss": 1.6, "val_loss": 1.5}],
    "best": {"epoch": 1, "metric": "val_loss", "loss": 1.5},
    "train_loss_steps": [
        {"step": s, "epoch": 1, "loss": 2.0 - s / 100, "lr": 1e-4}
        for s in (10, 20, 30, 40)
    ],
    "runtime": {"train_seconds": 130.0, "total_seconds": 150.0},
    "memory": {"max_memory_allocated_gib": 16.5, "max_memory_reserved_gib": 17.2},
    "environment": {"git_commit": "abc1234"},
}
SAMPLES = {
    "prompts_file": "prompts/compare_ja_20.json",
    "prompts_sha256": "1" * 64,
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "system_prompt": "s",
    "generation": {"max_new_tokens": 8, "do_sample": False},
    "samples": [
        {
            "id": "list",
            "category": "list",
            "instruction": "5つ挙げて",
            "input": "",
            "base_output": "1. A案です\n2. B案です",
            "base_new_tokens": 5,
            "finetuned_output": "1. 同じ行です\n2. 同じ行です",
            "finetuned_new_tokens": 8,
        },
        {
            "id": "sum",
            "category": "summarization",
            "instruction": "要約して",
            "input": "本文",
            "base_output": "要約です。",
            "base_new_tokens": 3,
            "finetuned_output": "要約です。",
            "finetuned_new_tokens": 3,
        },
    ],
}


def _write_run(root: Path, name: str, lr: float, **config_changes) -> Path:
    run = root / name
    (run / "eval20").mkdir(parents=True)
    metrics = copy.deepcopy(METRICS)
    metrics["config"].update(lr=lr, **config_changes)
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "eval20" / "samples.json").write_text(
        json.dumps(SAMPLES, ensure_ascii=False)
    )
    return run


def test_compare_runs_writes_table_and_side_by_side_outputs(tmp_path) -> None:
    runs = [
        _write_run(tmp_path, "lr2e-4", 2e-4),
        _write_run(tmp_path, "lr5e-5", 5e-5, log_every=5),
    ]
    judgments = tmp_path / "judgments.json"
    judgments.write_text(json.dumps({"lr2e-4": {"list": "degraded", "sum": "same"}}))
    out = tmp_path / "comparison.md"

    code = compare_runs.main(
        [
            *map(str, runs),
            "--samples",
            "eval20/samples.json",
            "--judgments",
            str(judgments),
            "-o",
            str(out),
        ]
    )

    assert code == 0
    text = out.read_text()
    assert (
        "| lr2e-4 | 0.0002 | completed | 1.5 | 1.5 | 1.7 | 1 | 0 | 1 | 1 | 130 | 150 | 17.2 | 16.5 | abc1234 |"
        in text
    )
    assert "| lr2e-4 | 0 | 1 | 1 | 0 | 0 |" in text
    assert "| lr5e-5 | 0 | 0 | 0 | 0 | 2 |" in text
    assert "**lr2e-4** (judged degraded; duplicate lines" in text
    assert "### 2. sum (summarization)" in text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": 7}, "config.seed"),
        ({"max_train_samples": 100}, "config.max_train_samples"),
    ],
)
def test_compare_runs_rejects_other_config_differences(
    tmp_path, change, message, capsys
) -> None:
    runs = [_write_run(tmp_path, "a", 2e-4), _write_run(tmp_path, "b", 1e-4, **change)]

    assert compare_runs.main([*map(str, runs), "--samples", "eval20/samples.json"]) == 1
    assert message in capsys.readouterr().out


def test_compare_runs_rejects_different_prompt_sets(tmp_path, capsys) -> None:
    runs = [_write_run(tmp_path, "a", 2e-4), _write_run(tmp_path, "b", 1e-4)]
    path = runs[1] / "eval20" / "samples.json"
    samples = json.loads(path.read_text())
    samples["prompts_sha256"] = "2" * 64
    path.write_text(json.dumps(samples))

    assert compare_runs.main([*map(str, runs), "--samples", "eval20/samples.json"]) == 1
    assert "prompts_sha256" in capsys.readouterr().out


def test_judgment_counts_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError):
        compare_runs.judgment_counts({"x": "better"}, ["x"])
