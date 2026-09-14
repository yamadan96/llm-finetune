"""Inspect response-only label masking with the real tokenizer on real rows.

Usage:
    uv run python scripts/inspect_masking.py \
        --output checkpoints/<run>/masking_check.json

Downloads only the tokenizer and the dataset (no model weights). For a few
rows of the seeded training split (plus one forced-truncation case) it builds
examples with ``src.dataset.build_example`` and checks, per example:

- the supervised labels form one contiguous span that starts right after the
  prompt prefix and equals the input ids on that span
- padding is attention-masked, uses the pad id and is ignored (-100)
- the ignored prefix decodes exactly to the ChatML prompt, the supervised span
  decodes exactly to ``response + <|im_end|>`` and ends with that single token
- the whole unpadded sequence decodes back to the full ChatML text

The report lists token counts, -100 ranges and (with ``--show-ids``) the raw
input ids and labels. Exits with status 1 if any check fails.
``scripts/check_run.py`` reads the report from the run directory.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from src.dataset import (
    DEFAULT_DATASET,
    END_TOKEN,
    IGNORE_INDEX,
    build_example,
    format_chatml,
    format_prompt_prefix,
    format_response,
    limit_indices,
    split_indices,
)
from src.evidence import public_identifier, write_json
from src.model import DEFAULT_BASE_MODEL

TRUNCATED_RESPONSE_TOKENS = 3
SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check response-only label masking with the real tokenizer"
    )
    p.add_argument("--tokenizer", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--num-samples", type=int, default=3)
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


def inspect_example(
    tokenizer: Any,
    instruction: str,
    response: str,
    max_length: int,
    show_ids: bool = False,
) -> dict[str, Any]:
    """Build one example and verify its supervision boundary."""
    example = build_example(tokenizer, instruction, response, max_length)
    prefix_ids = _token_ids(tokenizer, format_prompt_prefix(instruction))
    response_ids = _token_ids(tokenizer, format_response(response))
    end_ids = _token_ids(tokenizer, END_TOKEN)
    report: dict[str, Any] = {
        "prefix_tokens": len(prefix_ids),
        "response_tokens": len(response_ids),
        "max_length": max_length,
    }
    if example is None:
        report.update(ok=False, checks={"example_built": False})
        return report

    ids = example["input_ids"].tolist()
    labels = example["labels"].tolist()
    mask = example["attention_mask"].tolist()
    supervised = [label != IGNORE_INDEX for label in labels]
    supervised_ranges = _ranges(supervised)
    num_tokens = sum(mask)
    truncated = len(prefix_ids) + len(response_ids) > max_length
    start, end = supervised_ranges[0] if supervised_ranges else (0, 0)
    supervised_ids = ids[start:end]

    checks = {
        "example_built": True,
        "supervised_is_one_span": len(supervised_ranges) == 1,
        "supervised_starts_after_prefix": start == len(prefix_ids),
        "labels_equal_input_ids_on_span": labels[start:end] == supervised_ids,
        "padding_masked_and_ignored": all(
            labels[i] == IGNORE_INDEX and ids[i] == tokenizer.pad_token_id
            for i, m in enumerate(mask)
            if m == 0
        ),
        "prefix_decodes_to_prompt": tokenizer.decode(ids[:start])
        == format_prompt_prefix(instruction),
        "end_token_is_single_token": len(end_ids) == 1,
    }
    if truncated:
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
        ) == format_chatml(instruction, response)

    report.update(
        ok=all(checks.values()),
        truncated=truncated,
        num_tokens=num_tokens,
        supervised_tokens=end - start,
        supervised_range=[start, end],
        ignore_ranges=_ranges([not flag for flag in supervised]),
        checks=checks,
        decoded_supervised=tokenizer.decode(supervised_ids),
    )
    if show_ids:
        report.update(input_ids=ids, labels=labels)
    return report


def select_rows(
    num_rows: int, num_samples: int, val_ratio: float, seed: int
) -> list[int]:
    """The same seeded training split and subset rule as ``src.train``."""
    train_idx, _ = split_indices(num_rows, val_ratio, seed)
    return limit_indices(train_idx, num_samples, seed)


def inspect_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    show_ids: bool = False,
) -> dict[str, Any]:
    """Inspect ``rows`` plus a forced truncation of the first usable row."""
    samples = []
    warnings: list[str] = []
    for row in rows:
        instruction = row.get("instruction") or ""
        response = row.get("output") or row.get("response") or ""
        if not instruction or not response:
            continue
        entry = {
            "row_index": row.get("index"),
            **inspect_example(tokenizer, instruction, response, max_length, show_ids),
        }
        if (row.get("input") or "").strip():
            entry["dataset_input_ignored"] = True
            warnings.append(
                f"row {row.get('index')}: non-empty 'input' field is not part of "
                "the prompt (src.dataset uses instruction and output only)"
            )
        samples.append(entry)
    if samples:
        first = next(r for r in rows if r.get("instruction") and r.get("output"))
        prefix_len = samples[0]["prefix_tokens"]
        samples.append(
            {
                "row_index": first.get("index"),
                "forced_truncation": True,
                **inspect_example(
                    tokenizer,
                    first["instruction"],
                    first["output"],
                    prefix_len + TRUNCATED_RESPONSE_TOKENS,
                    show_ids,
                ),
            }
        )
    return {
        "ok": bool(samples) and all(s["ok"] for s in samples),
        "samples": samples,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset_id, split="train")
    indices = select_rows(len(raw), args.num_samples, args.val_ratio, args.seed)
    result = inspect_rows(
        tokenizer, [raw[i] for i in indices], args.max_length, args.show_ids
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": public_identifier(args.tokenizer),
        "dataset_id": public_identifier(args.dataset_id),
        "pad_token_id": tokenizer.pad_token_id,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        **result,
    }

    for sample in report["samples"]:
        failed = [name for name, ok in sample["checks"].items() if not ok]
        label = "truncation" if sample.get("forced_truncation") else "sample"
        print(
            f"[{'PASS' if sample['ok'] else 'FAIL'}] {label} row={sample['row_index']} "
            f"tokens={sample.get('num_tokens')} prefix={sample['prefix_tokens']} "
            f"supervised={sample.get('supervised_tokens')} "
            f"range={sample.get('supervised_range')} "
            f"ignored={sample.get('ignore_ranges')}"
            + (f" failed={failed}" if failed else "")
        )
        print(f"    supervised text: {sample.get('decoded_supervised')!r}")
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report, redact=False)
        print(f"Wrote {args.output}")
    print("MASKING OK" if report["ok"] else "MASKING CHECK FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
