import getpass
import json
import socket
from pathlib import Path

import pytest
import torch

import src.train as train_module
from src.dataset import IGNORE_INDEX, InstructionDataset
from src.lora import LORA_CONFIG_FILENAME, apply_lora, load_lora_config
from src.model import LORA_TARGET_MODULES
from src.train import build_scheduler, parse_args, seed_everything
from tests.conftest import build_tiny_causal_lm


def _lrs(num_steps: int, warmup_ratio: float) -> list[float]:
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = build_scheduler(optimizer, num_steps, warmup_ratio)
    lrs = []
    for _ in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return lrs


def test_scheduler_warms_up_linearly_then_decays_per_step() -> None:
    lrs = _lrs(num_steps=100, warmup_ratio=0.1)

    assert lrs[0] == 0.0
    assert lrs[5] == pytest.approx(0.5)
    assert lrs[10] == pytest.approx(1.0)
    warmup, decay = lrs[:11], lrs[10:]
    assert all(a < b for a, b in zip(warmup, warmup[1:], strict=False))
    assert all(a > b for a, b in zip(decay, decay[1:], strict=False))
    assert lrs[-1] < 0.01


def test_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.seed == 42
    assert args.val_ratio == 0.02
    assert args.warmup_ratio == 0.03
    assert args.max_train_samples is None
    assert args.max_val_samples is None
    assert args.dataset_id == "kunishou/databricks-dolly-15k-ja"


@pytest.mark.parametrize("value", ["0", "-5"])
def test_parse_args_rejects_non_positive_sample_limits(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--max-train-samples", value])


def test_seed_everything_makes_dataloader_order_reproducible() -> None:
    def order() -> list[int]:
        seed_everything(123)
        loader = torch.utils.data.DataLoader(
            list(range(20)),
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(123),
        )
        return [int(b) for b in loader]

    assert order() == order()


ROWS = [{"instruction": f"q{i}", "output": f"answer {i}"} for i in range(8)]
SMOKE_ARGS = ["--epochs", "2", "--batch-size", "2", "--rank", "4", "--max-length", "96"]


@pytest.fixture
def smoke_run(tmp_path, monkeypatch, fake_tokenizer):
    """Patch model/dataset loading so ``train`` runs on CPU without downloads."""
    datasets_built: dict[str, InstructionDataset] = {}

    def fake_build_lora_model(model_id, rank, alpha, dropout, target_modules):
        model = build_tiny_causal_lm()
        apply_lora(model, target_modules, rank=rank, alpha=alpha, dropout=dropout)
        return model, fake_tokenizer

    def fake_load_datasets(
        tokenizer,
        dataset_id,
        max_length,
        val_ratio,
        seed,
        max_train_samples,
        max_val_samples,
        train_examples,
        exclude_response_truncated,
        list_rows,
    ):
        datasets_built["train"] = InstructionDataset(tokenizer, ROWS[:6], max_length)
        datasets_built["val"] = InstructionDataset(tokenizer, ROWS[6:], max_length)
        return datasets_built["train"], datasets_built["val"]

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_module, "WANDB_PROJECT", None)
    monkeypatch.setattr(train_module, "build_lora_model", fake_build_lora_model)
    monkeypatch.setattr(train_module, "load_instruction_datasets", fake_load_datasets)

    def run(*extra: str) -> dict:
        train_module.train(parse_args([*SMOKE_ARGS, *extra]))
        return json.loads((tmp_path / "metrics.json").read_text())

    run.datasets = datasets_built
    return run


def test_train_smoke_writes_best_checkpoint_config_and_metrics(
    tmp_path, smoke_run
) -> None:
    metrics = smoke_run()

    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2]
    assert all(e["val_loss"] is not None for e in metrics["epochs"])
    assert metrics["config"]["seed"] == 42
    assert metrics["config"]["num_training_steps"] == 6
    assert metrics["config"]["max_train_samples"] is None
    assert metrics["config"]["max_val_samples"] is None
    assert metrics["config"]["train_examples"] is None
    assert metrics["config"]["exclude_response_truncated"] is False
    assert metrics["config"]["train_row_ids"] == []
    assert metrics["best"]["metric"] == "val_loss"
    assert (tmp_path / "lora_weights.pt").exists()
    assert load_lora_config(tmp_path / LORA_CONFIG_FILENAME) == {
        "rank": 4,
        "alpha": 32.0,
        "dropout": 0.05,
        "target_modules": LORA_TARGET_MODULES,
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
    }


def test_train_metrics_records_environment_runtime_and_memory(smoke_run) -> None:
    metrics = smoke_run()

    assert metrics["status"] == "completed"
    assert metrics["error"] is None
    config = metrics["config"]
    assert config["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert config["dataset_id"] == "kunishou/databricks-dolly-15k-ja"
    assert config["num_trainable_params"] > 0

    env = metrics["environment"]
    assert env["torch"] == torch.__version__
    assert env["transformers"] and env["datasets"] and env["python"]
    assert "git_commit" in env and "git_dirty" in env
    if not torch.cuda.is_available():
        assert env["cuda"] is None
        assert env["gpus"] is None
        assert metrics["memory"] == {
            "max_memory_allocated_gib": None,
            "max_memory_reserved_gib": None,
            "per_device": None,
        }

    runtime = metrics["runtime"]
    assert runtime["optimizer_steps"] == 6
    assert len(runtime["epoch_seconds"]) == 2
    assert runtime["total_seconds"] >= runtime["train_seconds"] > 0
    assert runtime["setup_seconds"] >= 0
    assert runtime["eval_seconds"] >= 0
    train_samples = smoke_run.datasets["train"].samples
    epochs = 2
    expected_supervised = epochs * sum(
        int((s["labels"][1:] != IGNORE_INDEX).sum()) for s in train_samples
    )
    expected_input = epochs * sum(int(s["attention_mask"].sum()) for s in train_samples)
    assert runtime["tokens_trained"] == expected_supervised
    assert runtime["input_tokens"] == expected_input
    assert runtime["tokens_per_second"] == pytest.approx(
        expected_supervised / runtime["train_seconds"]
    )


@pytest.mark.parametrize(
    ("log_every", "expected_steps", "expected_epochs"),
    [
        ("1", [1, 2, 3, 4, 5, 6], [1, 1, 1, 2, 2, 2]),
        ("2", [2, 4, 6], [1, 2, 2]),
        ("4", [4], [2]),
    ],
)
def test_train_logs_step_losses_every_n_optimizer_steps(
    smoke_run, log_every: str, expected_steps: list[int], expected_epochs: list[int]
) -> None:
    metrics = smoke_run("--log-every", log_every)

    steps = metrics["train_loss_steps"]
    assert [e["step"] for e in steps] == expected_steps
    assert [e["epoch"] for e in steps] == expected_epochs
    assert all(set(e) == {"step", "epoch", "loss", "lr"} for e in steps)
    assert all(e["loss"] > 0 for e in steps)
    # First optimizer step runs with the warmup LR of 0
    if log_every == "1":
        assert steps[0]["lr"] == 0.0
        assert steps[1]["lr"] > 0.0


def test_train_writes_metrics_incrementally_and_keeps_evidence_on_crash(
    smoke_run, monkeypatch
) -> None:
    calls = {"n": 0}
    real_evaluate = train_module.evaluate

    def failing_evaluate(model, loader):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash in epoch 2")
        return real_evaluate(model, loader)

    monkeypatch.setattr(train_module, "evaluate", failing_evaluate)

    with pytest.raises(RuntimeError):
        smoke_run("--log-every", "1")

    metrics = json.loads((train_module.CHECKPOINT_DIR / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert metrics["error"] == "RuntimeError"
    assert [e["epoch"] for e in metrics["epochs"]] == [1]
    assert [e["step"] for e in metrics["train_loss_steps"]] == [1, 2, 3, 4, 5, 6]
    assert metrics["runtime"]["optimizer_steps"] == 6
    assert metrics["best"]["epoch"] == 1


def _string_values(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _string_values(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _string_values(v)]
    return []


def assert_no_local_details(text: str, obj) -> None:
    for value in _string_values(obj):
        assert not value.startswith("/"), value
        assert ":\\" not in value, value
        assert not value.startswith("~"), value
    assert str(Path.home()) not in text
    for secret in (getpass.getuser(), socket.gethostname()):
        if len(secret) >= 3:
            assert secret not in _string_values(obj)


def test_train_metrics_contain_no_absolute_paths(tmp_path, smoke_run) -> None:
    local_model_dir = tmp_path / "models" / "tiny-qwen"
    local_model_dir.mkdir(parents=True)

    metrics = smoke_run("--model-id", str(local_model_dir))

    text = (tmp_path / "metrics.json").read_text()
    assert str(tmp_path) not in text
    assert metrics["config"]["model_id"] == "local:tiny-qwen"
    lora_config_text = (tmp_path / LORA_CONFIG_FILENAME).read_text()
    assert str(tmp_path) not in lora_config_text
    assert json.loads(lora_config_text)["base_model_id"] == "local:tiny-qwen"
    assert_no_local_details(text, metrics)
