"""Japanese instruction dataset for LoRA fine-tuning."""

import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .list_rules import is_list_instruction

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
    response_tokens: int


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
        response_tokens=len(response_ids),
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
        exclude_response_truncated: bool = False,
        target_examples: int | None = None,
        row_ids: Sequence[int] | None = None,
        row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
        response_tokens_min: int | None = None,
        response_tokens_max: int | None = None,
    ) -> None:
        """Tokenize ``rows`` in the given order.

        ``exclude_response_truncated`` drops rows whose response does not fit
        into ``max_length`` (they would teach answers without a closing
        ``<|im_end|>``). ``target_examples`` stops after that many usable
        examples, so a filtered run can be refilled from the same ordered pool
        to the same size as an unfiltered one. ``row_filter`` keeps only the rows
        it accepts, which is how a split with a fixed number of
        list-instruction rows is built. ``row_ids`` are the dataset row
        indices of ``rows``; the ids actually used are kept in
        ``self.row_ids``.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, torch.Tensor]] = []
        self.row_ids: list[int] = []
        self.num_empty = 0
        self.num_truncated_away = 0
        self.num_with_context = 0
        self.num_context_truncated = 0
        self.num_response_truncated = 0
        self.num_excluded_response_truncated = 0
        self.num_excluded_by_length = 0
        self.response_token_counts: list[int] = []
        for position, row in enumerate(rows):
            if target_examples is not None and len(self.samples) >= target_examples:
                break
            if row_filter is not None and not row_filter(row):
                continue
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
            if exclude_response_truncated and info.response_truncated:
                self.num_excluded_response_truncated += 1
                continue
            if (
                response_tokens_min is not None
                and info.response_tokens < response_tokens_min
            ) or (
                response_tokens_max is not None
                and info.response_tokens > response_tokens_max
            ):
                self.num_excluded_by_length += 1
                continue
            self.response_token_counts.append(info.response_tokens)
            if row_ids is not None and position < len(row_ids):
                self.row_ids.append(int(row_ids[position]))
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
            "excluded_response_truncated": self.num_excluded_response_truncated,
            "excluded_by_length": self.num_excluded_by_length,
            "median_response_tokens": (
                sorted(self.response_token_counts)[len(self.response_token_counts) // 2]
                if self.response_token_counts
                else 0
            ),
        }

    @classmethod
    def merged(cls, parts: Sequence["InstructionDataset"]) -> "InstructionDataset":
        """One dataset from several parts, ordered by dataset row id."""
        merged = cls(parts[0].tokenizer, [], parts[0].max_length)
        pairs = [
            (row_id, sample)
            for part in parts
            for row_id, sample in zip(part.row_ids, part.samples, strict=True)
        ]
        pairs.sort(key=lambda pair: pair[0])
        merged.row_ids = [row_id for row_id, _ in pairs]
        merged.samples = [sample for _, sample in pairs]
        for part in parts:
            for name in (
                "num_empty",
                "num_truncated_away",
                "num_with_context",
                "num_context_truncated",
                "num_response_truncated",
                "num_excluded_response_truncated",
                "num_excluded_by_length",
            ):
                setattr(merged, name, getattr(merged, name) + getattr(part, name))
            merged.response_token_counts += part.response_token_counts
        return merged

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
    train_examples: int | None = None,
    exclude_response_truncated: bool = False,
    list_rows: int | None = None,
    response_tokens_min: int | None = None,
    response_tokens_max: int | None = None,
    train_row_ids: Sequence[int] | None = None,
) -> tuple[InstructionDataset, InstructionDataset]:
    """Load the raw dataset and build seeded train/validation datasets.

    ``max_train_samples`` / ``max_val_samples`` cap the number of raw rows taken
    from each split *after* the seeded split (see ``limit_indices``), so the
    validation rows of a pilot run are always a subset of the full run's
    validation rows. Rows skipped during tokenization (empty or fully
    truncated) are not replaced, so the final dataset can be slightly smaller.

    ``train_examples`` instead walks the same seeded order and stops once that
    many usable training examples have been collected, so a run that drops
    rows (``exclude_response_truncated``, applied to the training split only)
    is refilled from the same ordered pool and keeps the size of an unfiltered
    run. The two options are mutually exclusive.

    ``train_row_ids`` trains on exactly those dataset rows (they must belong to
    the training split of this seed), which is how selection experiments hand a
    precomputed split to training; it cannot be combined with the other
    selection options.

    ``list_rows`` fixes how many training examples are list instructions (the
    rest come from the same order), making the amount of list supervision an
    independent variable.
    """
    if train_examples is not None and max_train_samples is not None:
        raise ValueError("Use either train_examples or max_train_samples, not both")
    if train_row_ids is not None and (
        train_examples is not None
        or max_train_samples is not None
        or list_rows is not None
        or response_tokens_min is not None
        or response_tokens_max is not None
    ):
        raise ValueError(
            "train_row_ids cannot be combined with other selection options"
        )
    if list_rows is not None:
        if train_examples is None:
            raise ValueError("list_rows requires train_examples")
        if list_rows > train_examples:
            raise ValueError("list_rows cannot exceed train_examples")
    # datasets>=4 no longer supports loading scripts or `trust_remote_code`
    raw = load_dataset(dataset_id, split=split)
    logger.info("Loaded %d examples from %s", len(raw), dataset_id)

    train_idx, val_idx = split_indices(len(raw), val_ratio, seed)
    if train_row_ids is not None:
        unknown = set(train_row_ids) - set(train_idx)
        if unknown:
            raise ValueError(
                f"{len(unknown)} of the given train_row_ids are not in the "
                "training split of this seed/val_ratio"
            )
        train_idx = list(train_row_ids)
    elif train_examples is not None:
        # Same seeded order as limit_indices, but consumed until the target is met
        train_idx = list(train_idx)
        random.Random(seed).shuffle(train_idx)
    else:
        train_idx = limit_indices(train_idx, max_train_samples, seed)
    val_idx = limit_indices(val_idx, max_val_samples, seed)

    def build_train(target: int | None, row_filter=None) -> InstructionDataset:
        return InstructionDataset(
            tokenizer,
            raw.select(train_idx),
            max_length,
            exclude_response_truncated=exclude_response_truncated,
            target_examples=target,
            row_ids=train_idx,
            row_filter=row_filter,
            response_tokens_min=response_tokens_min,
            response_tokens_max=response_tokens_max,
        )

    def is_list_row(row: Mapping[str, Any]) -> bool:
        return is_list_instruction(row.get("instruction", "") or "")

    if list_rows is None:
        train_set = build_train(train_examples)
    else:
        # Same ordered pool, but with a fixed number of list-instruction rows
        list_part = build_train(list_rows, is_list_row)
        rest = build_train(
            train_examples - len(list_part), lambda row: not is_list_row(row)
        )
        train_set = InstructionDataset.merged([list_part, rest])
        logger.info(
            "List supervision: %d of %d training examples are list instructions",
            len(list_part),
            len(train_set),
        )
    val_set = InstructionDataset(
        tokenizer, raw.select(val_idx), max_length, row_ids=val_idx
    )
    logger.info("Split: %d train / %d validation", len(train_set), len(val_set))
    return train_set, val_set
