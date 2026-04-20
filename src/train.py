"""Training script for LoRA fine-tuning of Qwen2.5-7B-Instruct."""
import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import InstructionDataset
from .lora import get_lora_params, save_lora_weights
from .model import build_lora_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if WANDB_PROJECT:
        import wandb

        wandb.init(project=WANDB_PROJECT, config=vars(args))

    model, tokenizer = build_lora_model(
        args.model_id, rank=args.rank, alpha=args.alpha
    )
    model.gradient_checkpointing_enable()

    dataset = InstructionDataset(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0:
                logger.info(
                    "Epoch %d step %d loss=%.4f", epoch, step, loss.item()
                )
                if WANDB_PROJECT:
                    import wandb

                    wandb.log({"train/loss": loss.item(), "epoch": epoch})

        avg_loss = total_loss / len(loader)
        scheduler.step()
        logger.info("Epoch %d complete | avg_loss=%.4f", epoch, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_lora_weights(model, str(CHECKPOINT_DIR / "lora_weights.pt"))
            tokenizer.save_pretrained(str(CHECKPOINT_DIR))
            logger.info("Saved best checkpoint (loss=%.4f)", best_loss)

    if WANDB_PROJECT:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
