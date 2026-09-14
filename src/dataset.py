"""Japanese instruction dataset for LoRA fine-tuning."""

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
# Separates the instruction from the row's ``input`` (reference text such as the
# passage for closed QA or summarization) inside the single user message.
CONTEXT_HEADER = "補足情報:"
# Appended to a context that was shortened to fit --max-length
CONTEXT_TRUNCATION_MARK = "…"
# A long context is shortened so that at least max_length // 4 response tokens
# (or the whole response, if shorter) fit
RESPONSE_RESERVE_DIVISOR = 4


def format_user_message(instruction: str, context: str = "") -> str:
    """User message content for one instruction row.

    Without context the message is the instruction itself. With a non-empty
    context (the dataset's ``input`` field) the format is fixed as::

        {instruction}

        補足情報:
        {context}
    """
    context = context.strip()
    if not context:
        return instruction
    return f"{instruction}\n\n{CONTEXT_HEADER}\n{context}"


def chat_messages(
    instruction: str, context: str = "", system: str = SYSTEM_PROMPT
) -> list[dict[str, str]]:
    """Messages for ``tokenizer.apply_chat_template`` (system + one user turn)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_user_message(instruction, context)},
    ]


def format_prompt_prefix(
    instruction: str, system: str = SYSTEM_PROMPT, context: str = ""
) -> str:
    """ChatML prompt up to and including the assistant header (not supervised).

    This equals Qwen2.5's chat template applied to ``chat_messages`` with
    ``add_generation_prompt=True``; scripts/inspect_masking.py verifies that
    against the real tokenizer.
    """
    user = format_user_message(instruction, context)
    return (
        f"<|im_start|>system\n{system}{END_TOKEN}\n"
        f"<|im_start|>user\n{user}{END_TOKEN}\n"
        f"<|im_start|>assistant\n"
    )


def format_response(response: str) -> str:
    """Assistant response including the closing end token (supervised)."""
    return f"{response}{END_TOKEN}"


def format_chatml(
    instruction: str, response: str, system: str = SYSTEM_PROMPT, context: str = ""
) -> str:
    """Format as ChatML template used by Qwen2.5."""
    return format_prompt_prefix(instruction, system, context) + format_response(
        response
    )


@dataclass(frozen=True)
class ExampleInfo:
    """How ``build_example_with_info`` fitted a row into ``max_length``."""

    context_used: str
    context_truncated: bool
    response_truncated: bool


def _token_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def response_token_reserve(num_response_tokens: int, max_length: int) -> int:
    """Response tokens kept when a long context has to be shortened."""
    return min(num_response_tokens, max(1, max_length // RESPONSE_RESERVE_DIVISOR))


def fit_context(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    context: str,
    num_response_tokens: int,
    max_length: int,
    system: str = SYSTEM_PROMPT,
) -> tuple[str, list[int]] | None:
    """Shorten ``context`` so the prompt leaves room for the response.

    The prompt may use at most ``max_length - response_token_reserve(...)``
    tokens. If it is longer, the context is cut at a token boundary from the
    end and ``CONTEXT_TRUNCATION_MARK`` is appended; the prompt is then
    re-tokenized as a whole, so the supervision boundary stays exact. A prompt
    without context is never shortened. Returns ``(context_used, prefix_ids)``,
    or None if no response token can fit.
    """
    context = context.strip()
    budget = max_length - response_token_reserve(num_response_tokens, max_length)
    prefix_ids = _token_ids(
        tokenizer, format_prompt_prefix(instruction, system, context)
    )
    if len(prefix_ids) <= budget:
        return context, prefix_ids
    if not context:
        # Nothing to shorten: keep whatever part of the response still fits
        return (context, prefix_ids) if len(prefix_ids) < max_length else None

    context_ids = _token_ids(tokenizer, context)
    keep = len(context_ids) - (len(prefix_ids) - budget)
    while keep > 0:
        partial = tokenizer.decode(context_ids[:keep]).rstrip("\ufffd").rstrip()
        if partial:
            candidate = partial + CONTEXT_TRUNCATION_MARK
            prefix_ids = _token_ids(
                tokenizer, format_prompt_prefix(instruction, system, candidate)
            )
            if len(prefix_ids) <= budget:
                return candidate, prefix_ids
            keep -= len(prefix_ids) - budget
        else:
            keep -= 1
    return None


def build_example_with_info(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
    max_length: int,
    system: str = SYSTEM_PROMPT,
    context: str = "",
) -> tuple[dict[str, torch.Tensor], ExampleInfo] | None:
    """Tokenize one example so that loss is computed only on the response.

    The prompt prefix (system, user message with optional context, assistant
    header) and the response are tokenized separately and concatenated, so
    the supervision boundary is exact. Labels are -100 for the prefix and for
    padding.

    Fitting into ``max_length``:

    - a long context is shortened first (see ``fit_context``) so that at
      least ``response_token_reserve`` response tokens remain
    - the sequence is then truncated from the end, so a long response keeps
      its first tokens

    Returns None if no response token would remain (for example a long
    instruction without context), because an all -100 row would yield a NaN
    loss.
    """
    response_ids = _token_ids(tokenizer, format_response(response))
    fitted = fit_context(
        tokenizer, instruction, context, len(response_ids), max_length, system
    )
    if fitted is None:
        return None
    context_used, prefix_ids = fitted

    input_ids = (prefix_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prefix_ids) + response_ids)[:max_length]
    if all(label == IGNORE_INDEX for label in labels):
        return None

    num_pad = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * num_pad
    input_ids = input_ids + [tokenizer.pad_token_id] * num_pad
    labels = labels + [IGNORE_INDEX] * num_pad
    example = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    info = ExampleInfo(
        context_used=context_used,
        context_truncated=context_used != context.strip(),
        response_truncated=len(prefix_ids) + len(response_ids) > max_length,
    )
    return example, info


def build_example(
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
    max_length: int,
    system: str = SYSTEM_PROMPT,
    context: str = "",
) -> dict[str, torch.Tensor] | None:
    """``build_example_with_info`` without the fitting details."""
    built = build_example_with_info(
        tokenizer, instruction, response, max_length, system, context
    )
    return None if built is None else built[0]


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
        self.num_empty = 0
        self.num_truncated_away = 0
        self.num_with_context = 0
        self.num_context_truncated = 0
        self.num_response_truncated = 0
        for row in rows:
            instruction = row.get("instruction", "") or ""
            response = row.get("output", row.get("response", "")) or ""
            context = row.get("input", "") or ""
            if not instruction or not response:
                self.num_empty += 1
                continue
            built = build_example_with_info(
                tokenizer, instruction, response, max_length, context=context
            )
            if built is None:
                self.num_truncated_away += 1
                continue
            example, info = built
            self.num_with_context += bool(context.strip())
            self.num_context_truncated += info.context_truncated
            self.num_response_truncated += info.response_truncated
            self.samples.append(example)

        if self.num_truncated_away:
            logger.warning(
                "Skipped %d examples with no response tokens after truncation",
                self.num_truncated_away,
            )
        logger.info(
            "Prepared %d samples (%d with context)",
            len(self.samples),
            self.num_with_context,
        )

    def stats(self) -> dict[str, int]:
        """Row counts for the run config in metrics.json."""
        return {
            "examples": len(self.samples),
            "with_context": self.num_with_context,
            "context_truncated": self.num_context_truncated,
            "response_truncated": self.num_response_truncated,
            "skipped_empty": self.num_empty,
            "skipped_truncated": self.num_truncated_away,
        }

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
