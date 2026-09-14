"""Japanese instruction dataset for LoRA fine-tuning."""

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "kunishou/databricks-dolly-15k-ja"
SYSTEM_PROMPT = "あなたは親切なアシスタントです。"
END_TOKEN = "<|im_end|>"
IGNORE_INDEX = -100


def format_prompt_prefix(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    """ChatML prompt up to and including the assistant header (not supervised)."""
    return (
        f"<|im_start|>system\n{system}{END_TOKEN}\n"
        f"<|im_start|>user\n{instruction}{END_TOKEN}\n"
        f"<|im_start|>assistant\n"
    )


def format_response(response: str) -> str:
    """Assistant response including the closing end token (supervised)."""
    return f"{response}{END_TOKEN}"


def format_chatml(instruction: str, response: str, system: str = SYSTEM_PROMPT) -> str:
    """Format as ChatML template used by Qwen2.5."""
    return format_prompt_prefix(instruction, system) + format_response(response)


def build_example(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
    max_length: int,
    system: str = SYSTEM_PROMPT,
) -> dict[str, torch.Tensor] | None:
    """Tokenize one example so that loss is computed only on the response.

    The prompt prefix and the response are tokenized separately and
    concatenated, so the supervision boundary is exact. Labels are -100 for
    the prefix and for padding. Returns None if truncation to ``max_length``
    removes every response token (an all -100 row would yield a NaN loss).
    """
    prefix_ids = tokenizer(
        format_prompt_prefix(instruction, system), add_special_tokens=False
    )["input_ids"]
    response_ids = tokenizer(format_response(response), add_special_tokens=False)[
        "input_ids"
    ]

    input_ids = (list(prefix_ids) + list(response_ids))[:max_length]
    labels = ([IGNORE_INDEX] * len(prefix_ids) + list(response_ids))[:max_length]
    if all(label == IGNORE_INDEX for label in labels):
        return None

    num_pad = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * num_pad
    input_ids = input_ids + [tokenizer.pad_token_id] * num_pad
    labels = labels + [IGNORE_INDEX] * num_pad
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def split_indices(
    num_examples: int, val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministically split example indices into (train, validation)."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    indices = list(range(num_examples))
    random.Random(seed).shuffle(indices)
    num_val = round(num_examples * val_ratio) if val_ratio > 0 else 0
    if val_ratio > 0 and num_examples > 1:
        num_val = min(max(num_val, 1), num_examples - 1)
    return sorted(indices[num_val:]), sorted(indices[:num_val])


def limit_indices(
    indices: Sequence[int], max_samples: int | None, seed: int
) -> list[int]:
    """Deterministically keep at most ``max_samples`` of ``indices``.

    The indices are shuffled with ``random.Random(seed)`` and the first
    ``max_samples`` are kept (returned sorted). The result depends only on the
    input indices, ``max_samples`` and ``seed``; ``None`` keeps every index.
    Increasing ``max_samples`` keeps a superset of a smaller limit.
    """
    kept = list(indices)
    if max_samples is None:
        return sorted(kept)
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    random.Random(seed).shuffle(kept)
    return sorted(kept[:max_samples])


class InstructionDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """Tokenized instruction dataset in ChatML format."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        rows: Iterable[Mapping[str, Any]],
        max_length: int = 512,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        num_truncated = 0
        for row in rows:
            instruction = row.get("instruction", "") or ""
            response = row.get("output", row.get("response", "")) or ""
            if not instruction or not response:
                continue
            example = build_example(tokenizer, instruction, response, max_length)
            if example is None:
                num_truncated += 1
                continue
            self.samples.append(example)

        if num_truncated:
            logger.warning(
                "Skipped %d examples with no response tokens after truncation",
                num_truncated,
            )
        logger.info("Prepared %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def load_instruction_datasets(
    tokenizer: PreTrainedTokenizerBase,
    dataset_id: str = DEFAULT_DATASET,
    max_length: int = 512,
    val_ratio: float = 0.02,
    seed: int = 42,
    split: str = "train",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[InstructionDataset, InstructionDataset]:
    """Load the raw dataset and build seeded train/validation datasets.

    ``max_train_samples`` / ``max_val_samples`` cap the number of raw rows taken
    from each split *after* the seeded split (see ``limit_indices``), so the
    validation rows of a pilot run are always a subset of the full run's
    validation rows. Rows skipped during tokenization (empty or fully
    truncated) are not replaced, so the final dataset can be slightly smaller.
    """
    # datasets>=4 no longer supports loading scripts or `trust_remote_code`
    raw = load_dataset(dataset_id, split=split)
    logger.info("Loaded %d examples from %s", len(raw), dataset_id)

    train_idx, val_idx = split_indices(len(raw), val_ratio, seed)
    train_idx = limit_indices(train_idx, max_train_samples, seed)
    val_idx = limit_indices(val_idx, max_val_samples, seed)
    train_set = InstructionDataset(tokenizer, raw.select(train_idx), max_length)
    val_set = InstructionDataset(tokenizer, raw.select(val_idx), max_length)
    logger.info("Split: %d train / %d validation", len(train_set), len(val_set))
    return train_set, val_set
