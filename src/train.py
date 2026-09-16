"""Training script for LoRA fine-tuning of Qwen2.5-7B-Instruct."""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from .dataset import DEFAULT_DATASET, IGNORE_INDEX, load_instruction_datasets
from .evidence import RunTracker, public_identifier
from .lora import (
    LORA_CONFIG_FILENAME,
    get_lora_params,
    save_lora_config,
    save_lora_weights,
)
from .model import (
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    build_lora_model,
    enable_gradient_checkpointing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
METRICS_FILENAME = "metrics.json"


def positive_int(value: str) -> int:
    """argparse type for integers >= 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--dropout", type=float, default=LORA_DROPOUT)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Fraction of optimizer steps used for linear LR warmup",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Fraction of the dataset held out for validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face Hub dataset id",
    )
    p.add_argument(
        "--max-train-samples",
        type=positive_int,
        default=None,
        help="Use at most N raw training rows, chosen deterministically from "
        "--seed after the train/validation split (default: all)",
    )
    p.add_argument(
        "--max-val-samples",
        type=positive_int,
        default=None,
        help="Use at most N raw validation rows, chosen the same way (default: all)",
    )
    p.add_argument(
        "--train-examples",
        type=positive_int,
        default=None,
        help="Collect exactly N usable training examples by walking the seeded "
        "order (refills after filtered rows); cannot be combined with "
        "--max-train-samples",
    )
    p.add_argument(
        "--exclude-response-truncated",
        action="store_true",
        help="Drop training rows whose response does not fit --max-length "
        "(they teach answers without a closing <|im_end|>); validation is "
        "left unchanged",
    )
    p.add_argument(
        "--list-rows",
        type=int,
        default=None,
        help="Force exactly N list-instruction rows among --train-examples "
        "(the rest are non-list rows from the same seeded order); 0 removes "
        "list supervision entirely",
    )
    p.add_argument(
        "--log-every",
        type=positive_int,
        default=10,
        help="Record the mean train loss and LR every N optimizer steps "
        "(metrics.json train_loss_steps)",
    )
    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: Optimizer, num_training_steps: int, warmup_ratio: float
) -> LambdaLR:
    """Cosine schedule with linear warmup, stepped once per optimizer step."""
    num_warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> float:
    """Token-weighted mean loss over all supervised (response) tokens."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        labels = batch["labels"].to(model.device)
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=labels,
        )
        # Causal LM loss is the mean over shifted, non-ignored label tokens
        num_tokens = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens
    return total_loss / total_tokens if total_tokens else float("nan")


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Run config for metrics.json: CLI args with shareable model/dataset ids."""
    return {
        **vars(args),
        "model_id": public_identifier(args.model_id),
        "dataset_id": public_identifier(args.dataset_id),
        "target_modules": LORA_TARGET_MODULES,
    }


def count_tokens(batch: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(supervised tokens covered by the shifted causal-LM loss, non-pad tokens)."""
    supervised = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())
    non_pad = int(batch["attention_mask"].sum().item())
    return supervised, non_pad


def train(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = RunTracker(
        CHECKPOINT_DIR / METRICS_FILENAME,
        config=run_config(args),
        log_every=args.log_every,
    )
    try:
        run_training(args, tracker)
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Only the exception type: messages may contain local paths
        tracker.finish(status, error=type(exc).__name__)
        raise
    tracker.finish("completed")


def run_training(args: argparse.Namespace, tracker: RunTracker) -> None:
    seed_everything(args.seed)

    if WANDB_PROJECT:
        import wandb

        wandb.init(project=WANDB_PROJECT, config=vars(args))

    model, tokenizer = build_lora_model(
        args.model_id,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=LORA_TARGET_MODULES,
    )
    enable_gradient_checkpointing(model)

    train_set, val_set = load_instruction_datasets(
        tokenizer,
        dataset_id=args.dataset_id,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_examples=args.train_examples,
        exclude_response_truncated=args.exclude_response_truncated,
        list_rows=args.list_rows,
    )
    if len(train_set) == 0:
        raise ValueError("Training set is empty")
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if len(val_set) > 0
        else None
    )
    if val_loader is None:
        logger.warning("No validation set; best checkpoint uses train loss instead")

    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    num_training_steps = len(loader) * args.epochs
    scheduler = build_scheduler(optimizer, num_training_steps, args.warmup_ratio)

    tracker.update_config(
        num_train_examples=len(train_set),
        num_val_examples=len(val_set),
        train_dataset_stats=train_set.stats(),
        val_dataset_stats=val_set.stats(),
        train_row_ids=sorted(train_set.row_ids),
        val_row_ids=sorted(val_set.row_ids),
        num_training_steps=num_training_steps,
        num_warmup_steps=math.ceil(num_training_steps * args.warmup_ratio),
        num_trainable_params=sum(p.numel() for p in lora_params),
    )
    tracker.start_training()

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tracker.start_epoch()
        total_loss = 0.0
        for batch in loader:
            supervised_tokens, input_tokens = count_tokens(batch)
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            lr = scheduler.get_last_lr()[0]  # LR applied by this optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            total_loss += loss_value

            entry = tracker.record_step(
                epoch=epoch,
                loss=loss_value,
                lr=lr,
                supervised_tokens=supervised_tokens,
                input_tokens=input_tokens,
            )
            if entry is not None:
                logger.info(
                    "Epoch %d step %d loss=%.4f lr=%.2e",
                    epoch,
                    entry["step"],
                    entry["loss"],
                    lr,
                )
                if WANDB_PROJECT:
                    import wandb

                    wandb.log(
                        {"train/loss": entry["loss"], "train/lr": lr, "epoch": epoch},
                        step=entry["step"],
                    )

        train_loss = total_loss / len(loader)
        tracker.end_epoch_training()
        val_loss = evaluate(model, val_loader) if val_loader is not None else None
        global_step = tracker.optimizer_steps
        logger.info(
            "Epoch %d complete | train_loss=%.4f val_loss=%s",
            epoch,
            train_loss,
            f"{val_loss:.4f}" if val_loss is not None else "n/a",
        )
        if WANDB_PROJECT:
            import wandb

            log = {"epoch/train_loss": train_loss, "epoch": epoch}
            if val_loss is not None:
                log["epoch/val_loss"] = val_loss
            wandb.log(log, step=global_step)

        selection_loss = val_loss if val_loss is not None else train_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            save_lora_weights(model, str(CHECKPOINT_DIR / "lora_weights.pt"))
            save_lora_config(
                CHECKPOINT_DIR / LORA_CONFIG_FILENAME,
                rank=args.rank,
                alpha=args.alpha,
                dropout=args.dropout,
                target_modules=LORA_TARGET_MODULES,
                base_model_id=args.model_id,
            )
            tokenizer.save_pretrained(str(CHECKPOINT_DIR))
            tracker.set_best(
                {
                    "epoch": epoch,
                    "metric": "val_loss" if val_loss is not None else "train_loss",
                    "loss": best_loss,
                }
            )
            logger.info("Saved best checkpoint (loss=%.4f)", best_loss)

        tracker.end_epoch(
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    if WANDB_PROJECT:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
