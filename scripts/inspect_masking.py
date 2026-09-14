"""Inspect response-only label masking with the real tokenizer on real rows.

Usage:
    uv run python scripts/inspect_masking.py \
        --output checkpoints/<run>/masking_check.json

Downloads only the tokenizer and the dataset (no model weights). Walking the
seeded training split in shuffled order it picks real rows for each case

- ``no_context``          rows without an ``input`` field
- ``with_context``        rows whose ``input`` is placed in the user message
- ``context_truncated``   a row whose ``input`` had to be shortened
- ``response_truncated``  a row whose response is cut at ``--max-length``

plus a ``forced_truncation`` of the first row at a tiny length, builds them
with ``src.dataset.build_example_with_info`` and checks, per example:

- the supervised labels form one contiguous span that starts right after the
  prompt prefix and equals the input ids on that span
- padding is attention-masked, uses the pad id and is ignored (-100)
- the ignored prefix decodes exactly to the ChatML prompt, which equals the
  tokenizer's own chat template for the same messages
- the context (or its shortened form, a prefix of the original) is inside
  the user message, and a shortened context leaves the response reserve
- the supervised span decodes exactly to ``response + <|im_end|>`` (or is a
  prefix of the response tokens when truncated)

The report lists token counts, -100 ranges and (with ``--show-ids``) the raw
input ids and labels. Exits with status 1 if any check fails or a case could
not be found. ``scripts/check_run.py`` reads the report from the run
directory.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from src.dataset import (
    CONTEXT_HEADER,
    CONTEXT_TRUNCATION_MARK,
    DEFAULT_DATASET,
    END_TOKEN,
    IGNORE_INDEX,
    build_example_with_info,
    chat_messages,
    format_chatml,
    format_prompt_prefix,
    format_response,
    response_token_reserve,
    split_indices,
)
from src.evidence import public_identifier, write_json
from src.model import DEFAULT_BASE_MODEL

FORCED_TRUNCATION_RESPONSE_TOKENS = 3
SCHEMA_VERSION = 2
CASES = ("no_context", "with_context", "context_truncated", "response_truncated")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check response-only label masking with the real tokenizer"
    )
    p.add_argument("--tokenizer", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Rows per no_context / with_context case (truncation cases use 1)",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None, help="Write a JSON report")
    p.add_argument(
        "--show-ids", action="store_true", help="Include input_ids/labels in output"
    )
    return p.parse_args(argv)


def _ranges(flags: list[bool]) -> list[list[int]]:
    """Half-open [start, end) ranges where ``flags`` is True."""
    ranges: list[list[int]] = []
    for index, flag in enumerate(flags):
        if not flag:
            continue
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1])
    return ranges


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _template_matches(
    tokenizer: Any, instruction: str, context: str, prefix: str
) -> bool | None:
    """Compare with the tokenizer's chat template (None if it has none)."""
    if not getattr(tokenizer, "chat_template", None):
        return None
    rendered = tokenizer.apply_chat_template(
        chat_messages(instruction, context),
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered == prefix


def row_texts(row: dict[str, Any]) -> tuple[str, str, str]:
    """(instruction, context, response) as ``InstructionDataset`` reads them."""
    return (
        row.get("instruction") or "",
        row.get("input") or "",
        row.get("output") or row.get("response") or "",
    )


def inspect_example(
    tokenizer: Any,
    instruction: str,
    response: str,
    max_length: int,
    context: str = "",
    show_ids: bool = False,
) -> dict[str, Any]:
    """Build one example and verify its supervision boundary."""
    response_ids = _token_ids(tokenizer, format_response(response))
    end_ids = _token_ids(tokenizer, END_TOKEN)
    report: dict[str, Any] = {
        "max_length": max_length,
        "response_tokens": len(response_ids),
        "has_context": bool(context.strip()),
    }
    built = build_example_with_info(
        tokenizer, instruction, response, max_length, context=context
    )
    if built is None:
        report.update(ok=False, checks={"example_built": False})
        return report
    example, info = built
    prefix = format_prompt_prefix(instruction, context=info.context_used)
    prefix_ids = _token_ids(tokenizer, prefix)

    ids = example["input_ids"].tolist()
    labels = example["labels"].tolist()
    mask = example["attention_mask"].tolist()
    supervised = [label != IGNORE_INDEX for label in labels]
    supervised_ranges = _ranges(supervised)
    num_tokens = sum(mask)
    start, end = supervised_ranges[0] if supervised_ranges else (0, 0)
    supervised_ids = ids[start:end]
    decoded_prefix = tokenizer.decode(ids[:start])

    checks: dict[str, bool | None] = {
        "example_built": True,
        "supervised_is_one_span": len(supervised_ranges) == 1,
        "supervised_starts_after_prefix": start == len(prefix_ids),
        "labels_equal_input_ids_on_span": labels[start:end] == supervised_ids,
        "padding_masked_and_ignored": all(
            labels[i] == IGNORE_INDEX and ids[i] == tokenizer.pad_token_id
            for i, m in enumerate(mask)
            if m == 0
        ),
        "prefix_decodes_to_prompt": decoded_prefix == prefix,
        "prompt_matches_chat_template": _template_matches(
            tokenizer, instruction, info.context_used, prefix
        ),
        "end_token_is_single_token": len(end_ids) == 1,
    }
    if context.strip():
        checks["context_in_user_message"] = (
            f"{CONTEXT_HEADER}\n{info.context_used}{END_TOKEN}" in decoded_prefix
        )
    else:
        checks["no_context_header"] = CONTEXT_HEADER not in decoded_prefix
    if info.context_truncated:
        kept = info.context_used.removesuffix(CONTEXT_TRUNCATION_MARK)
        checks["truncated_context_is_prefix_of_input"] = context.strip().startswith(
            kept
        )
        checks["response_reserve_kept"] = end - start >= response_token_reserve(
            len(response_ids), max_length
        )
    if info.response_truncated:
        checks["supervised_is_response_prefix"] = (
            supervised_ids == response_ids[: end - start]
        )
    else:
        checks["supervised_decodes_to_response"] = tokenizer.decode(
            supervised_ids
        ) == format_response(response)
        checks["ends_with_end_token"] = supervised_ids[-1:] == end_ids
        checks["sequence_decodes_to_chatml"] = tokenizer.decode(
            ids[:num_tokens]
        ) == format_chatml(instruction, response, context=info.context_used)

    report.update(
        ok=all(value is not False for value in checks.values()),
        context_truncated=info.context_truncated,
        response_truncated=info.response_truncated,
        num_tokens=num_tokens,
        prefix_tokens=len(prefix_ids),
        supervised_tokens=end - start,
        supervised_range=[start, end],
        ignore_ranges=_ranges([not flag for flag in supervised]),
        checks=checks,
        decoded_supervised=tokenizer.decode(supervised_ids),
    )
    if show_ids:
        report.update(input_ids=ids, labels=labels)
    return report


def shuffled_train_indices(num_rows: int, val_ratio: float, seed: int) -> list[int]:
    """Training-split indices (same split as ``src.train``) in seeded order."""
    train_idx, _ = split_indices(num_rows, val_ratio, seed)
    random.Random(seed).shuffle(train_idx)
    return train_idx


def classify_row(tokenizer: Any, row: dict[str, Any], max_length: int) -> str | None:
    instruction, context, response = row_texts(row)
    if not instruction or not response:
        return None
    built = build_example_with_info(
        tokenizer, instruction, response, max_length, context=context
    )
    if built is None:
        return None
    info = built[1]
    if info.context_truncated:
        return "context_truncated"
    if info.response_truncated:
        return "response_truncated"
    return "with_context" if context.strip() else "no_context"


def inspect_cases(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    num_samples: int,
    show_ids: bool = False,
) -> dict[str, Any]:
    """Pick rows for every case from ``rows`` (in order) and inspect them."""
    quotas = {
        "no_context": num_samples,
        "with_context": num_samples,
        "context_truncated": 1,
        "response_truncated": 1,
    }
    samples: list[dict[str, Any]] = []
    for row in rows:
        if not any(quotas.values()):
            break
        case = classify_row(tokenizer, row, max_length)
        if case is None or quotas[case] == 0:
            continue
        quotas[case] -= 1
        instruction, context, response = row_texts(row)
        samples.append(
            {
                "case": case,
                "row_index": row.get("index"),
                **inspect_example(
                    tokenizer, instruction, response, max_length, context, show_ids
                ),
            }
        )

    if samples:
        first = next(r for r in rows if classify_row(tokenizer, r, max_length))
        instruction, context, response = row_texts(first)
        tiny_length = (
            len(_token_ids(tokenizer, format_prompt_prefix(instruction)))
            + FORCED_TRUNCATION_RESPONSE_TOKENS
        )
        samples.append(
            {
                "case": "forced_truncation",
                "row_index": first.get("index"),
                **inspect_example(
                    tokenizer, instruction, response, tiny_length, "", show_ids
                ),
            }
        )
    missing = [case for case, left in quotas.items() if left]
    return {
        "ok": bool(samples) and not missing and all(sample["ok"] for sample in samples),
        "missing_cases": missing,
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset_id, split="train")
    order = shuffled_train_indices(len(raw), args.val_ratio, args.seed)
    rows = raw.select(order).to_list()
    result = inspect_cases(
        tokenizer, rows, args.max_length, args.num_samples, args.show_ids
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": public_identifier(args.tokenizer),
        "dataset_id": public_identifier(args.dataset_id),
        "pad_token_id": tokenizer.pad_token_id,
        "max_length": args.max_length,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        **result,
    }

    for sample in report["samples"]:
        failed = [name for name, ok in sample["checks"].items() if ok is False]
        print(
            f"[{'PASS' if sample['ok'] else 'FAIL'}] {sample['case']} "
            f"row={sample['row_index']} tokens={sample.get('num_tokens')} "
            f"prefix={sample.get('prefix_tokens')} "
            f"supervised={sample.get('supervised_tokens')} "
            f"range={sample.get('supervised_range')} "
            f"ignored={sample.get('ignore_ranges')}"
            + (f" failed={failed}" if failed else "")
        )
        print(f"    supervised text: {sample.get('decoded_supervised', '')[:120]!r}")
    if report["missing_cases"]:
        print(f"[FAIL] no rows found for cases: {report['missing_cases']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report, redact=False)
        print(f"Wrote {args.output}")
    print("MASKING OK" if report["ok"] else "MASKING CHECK FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
