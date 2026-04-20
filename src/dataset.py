"""Japanese instruction dataset for LoRA fine-tuning."""
import logging

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "kunishou/databricks-dolly-15k-ja"
SYSTEM_PROMPT = "あなたは親切なアシスタントです。"


def format_chatml(
    instruction: str, response: str, system: str = SYSTEM_PROMPT
) -> str:
    """Format as ChatML template used by Qwen2.5."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


class InstructionDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """Tokenized instruction dataset in ChatML format."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_id: str = DEFAULT_DATASET,
        max_length: int = 512,
        split: str = "train",
    ) -> None:
        raw = load_dataset(dataset_id, split=split, trust_remote_code=True)
        logger.info("Loaded %d examples from %s", len(raw), dataset_id)

        self.samples: list[dict[str, torch.Tensor]] = []
        for row in raw:
            instruction = row.get("instruction", "") or ""
            response = row.get("output", row.get("response", "")) or ""
            if not instruction or not response:
                continue
            text = format_chatml(instruction, response)
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            # Labels: mask padding with -100 so loss ignores them
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        logger.info("Prepared %d training samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
