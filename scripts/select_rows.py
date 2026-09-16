"""Pick the training rows of a run and write them to a JSON file.

Usage:
    uv run python scripts/select_rows.py --train-examples 497 \
        --selection longest --category-mix-from checkpoints/<run>/metrics.json \
        --output selections/long-matched.json

    CHECKPOINT_DIR=./checkpoints/<run> uv run python -m src.train \
        --train-row-ids selections/long-matched.json ...

Selection modes, all deterministic given ``--seed`` and ``--val-ratio``:

- ``seeded``: the first usable rows of the seeded training order (what
  ``--train-examples`` does inside training)
- ``longest`` / ``shortest``: the rows with the longest / shortest teacher
  responses

With ``--category-mix-from`` the per-category counts of a reference run are
reproduced exactly, so length can be varied *within* a fixed category mix.
Rows are only taken from the training split of the given seed, and only if
they produce a usable example; the file records the achieved length
distribution per category so the dose of the intervention is visible.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.dataset import (
    DEFAULT_DATASET,
    build_example_with_info,
    split_indices,
)
from src.evidence import public_identifier, write_json
from src.model import DEFAULT_BASE_MODEL

SELECTIONS = ("seeded", "longest", "shortest")
DEFAULT_POOL_ROWS = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select training rows for a run")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--tokenizer", default=DEFAULT_BASE_MODEL)
    p.add_argument("--split", default="train")
    p.add_argument("--train-examples", type=int, required=True)
    p.add_argument("--selection", choices=SELECTIONS, default="seeded")
    p.add_argument(
        "--category-mix-from",
        type=Path,
        default=None,
        help="metrics.json of a reference run whose per-category counts are matched",
    )
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pool-rows",
        type=int,
        default=DEFAULT_POOL_ROWS,
        help="How many rows of the seeded order to consider as candidates",
    )
    p.add_argument("-o", "--output", type=Path, required=True)
    return p.parse_args(argv)


def category_quotas(metrics_path: Path, categories: dict[int, str]) -> dict[str, int]:
    """Per-category row counts of a reference run."""
    config = json.loads(metrics_path.read_text(encoding="utf-8"))["config"]
    return dict(Counter(categories[row_id] for row_id in config["train_row_ids"]))


def select(
    candidates: list[dict[str, Any]],
    train_examples: int,
    selection: str,
    quotas: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """Rows to train on, in dataset order.

    ``candidates`` are usable rows in the seeded order, each with ``row_id``,
    ``category`` and ``response_tokens``.
    """

    def pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if selection == "seeded":
            return rows[:count]
        ordered = sorted(
            rows,
            key=lambda row: (row["response_tokens"], row["order"]),
            reverse=selection == "longest",
        )
        return ordered[:count]

    if quotas is None:
        chosen = pick(candidates, train_examples)
    else:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_category[row["category"]].append(row)
        chosen = []
        for category, count in sorted(quotas.items()):
            available = by_category.get(category, [])
            if len(available) < count:
                raise ValueError(
                    f"category {category}: {len(available)} candidates for a "
                    f"quota of {count}; raise --pool-rows"
                )
            chosen += pick(available, count)
    return sorted(chosen, key=lambda row: row["row_id"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(row["response_tokens"] for row in rows)
    per_category = defaultdict(list)
    for row in rows:
        per_category[row["category"]].append(row["response_tokens"])
    return {
        "examples": len(rows),
        "median_response_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "mean_response_tokens": sum(lengths) / len(lengths) if lengths else 0,
        "categories": {
            category: {
                "rows": len(values),
                "median_response_tokens": sorted(values)[len(values) // 2],
            }
            for category, values in sorted(per_category.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset_id, split=args.split)
    train_idx, _ = split_indices(len(raw), args.val_ratio, args.seed)
    import random

    order = list(train_idx)
    random.Random(args.seed).shuffle(order)

    categories = {row_id: raw[row_id]["category"] for row_id in train_idx}
    quotas = (
        category_quotas(args.category_mix_from, categories)
        if args.category_mix_from
        else None
    )

    candidates: list[dict[str, Any]] = []
    for position, row_id in enumerate(order[: args.pool_rows]):
        row = raw[row_id]
        built = build_example_with_info(
            tokenizer,
            row.get("instruction") or "",
            row.get("output") or "",
            args.max_length,
            context=row.get("input") or "",
        )
        if built is None:
            continue
        candidates.append(
            {
                "row_id": row_id,
                "order": position,
                "category": row["category"],
                "response_tokens": built[1].response_tokens,
            }
        )

    chosen = select(candidates, args.train_examples, args.selection, quotas)
    report = {
        "dataset_id": public_identifier(args.dataset_id),
        "tokenizer": public_identifier(args.tokenizer),
        "selection": args.selection,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_length": args.max_length,
        "pool_rows": len(candidates),
        "category_mix_matched": bool(quotas),
        "row_ids": [row["row_id"] for row in chosen],
        "summary": summarize(chosen),
    }
    write_json(args.output, report)
    summary = report["summary"]
    print(
        f"{args.selection}: {summary['examples']} rows, median response "
        f"{summary['median_response_tokens']} tokens -> {args.output}"
    )
    for category, values in summary["categories"].items():
        print(
            f"  {category:24s} {values['rows']:4d} rows, median "
            f"{values['median_response_tokens']:4d} tokens"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
