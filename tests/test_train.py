import json

import pytest
import torch

import src.train as train_module
from src.dataset import InstructionDataset
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


def test_train_smoke_writes_best_checkpoint_config_and_metrics(
    tmp_path, monkeypatch, fake_tokenizer
) -> None:
    rows = [{"instruction": f"q{i}", "output": f"answer {i}"} for i in range(8)]

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
    ):
        return (
            InstructionDataset(tokenizer, rows[:6], max_length),
            InstructionDataset(tokenizer, rows[6:], max_length),
        )

    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(train_module, "WANDB_PROJECT", None)
    monkeypatch.setattr(train_module, "build_lora_model", fake_build_lora_model)
    monkeypatch.setattr(train_module, "load_instruction_datasets", fake_load_datasets)
    args = parse_args(
        ["--epochs", "2", "--batch-size", "2", "--rank", "4", "--max-length", "96"]
    )

    train_module.train(args)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert [e["epoch"] for e in metrics["epochs"]] == [1, 2]
    assert all(e["val_loss"] is not None for e in metrics["epochs"])
    assert metrics["config"]["seed"] == 42
    assert metrics["config"]["num_training_steps"] == 6
    assert metrics["best"]["metric"] == "val_loss"
    assert (tmp_path / "lora_weights.pt").exists()
    assert load_lora_config(tmp_path / LORA_CONFIG_FILENAME) == {
        "rank": 4,
        "alpha": 32.0,
        "dropout": 0.05,
        "target_modules": LORA_TARGET_MODULES,
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
    }
