"""Audit the instruction dataset that training actually sees.

Usage:
    uv run python scripts/audit_dataset.py --output docs/dataset_audit.md \
        --json docs/dataset_audit.json

Tokenizes every row with the real tokenizer through ``build_example_with_info``
(the same code path as training) and reports, overall and per dataset
category:

- rows, usable examples, rows skipped as empty or fully truncated
- prompt and response token length distribution
- how many rows carry an ``input`` field
- truncation rates (context shortened, response cut at --max-length)
- very short responses (<= --short-response-tokens tokens)
- teacher responses that repeat lines or clauses (src.text_checks)

and, because instruction following degraded most on list and rewrite prompts,
two task slices detected from the instruction text:

- list-style instructions: whether the response is actually a multi-item list,
  its item count, and whether the items repeat
- rewrite-style instructions: how much the response differs from the input it
  is supposed to rewrite (identical, near-identical, or rewritten)

Only counts, ratios and row indices are written; no dataset text.
"""

import argparse
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.dataset import (
    DEFAULT_DATASET,
    build_example_with_info,
    format_prompt_prefix,
    format_response,
)
from src.evidence import public_identifier, write_json
from src.list_rules import (
    is_list_instruction,
    list_answer_quality,
    requested_item_count,
)
from src.model import DEFAULT_BASE_MODEL
from src.text_checks import duplicate_line_stats, repetition_findings

SCHEMA_VERSION = 1
QUANTILES = (5, 25, 50, 75, 95)
REWRITE_INSTRUCTION_RE = re.compile(
    r"書き換え|書き直|言い換え|書きかえ|直してください|修正してください|"
    r"丁寧|敬語|やさしい言葉|わかりやすく|分かりやすく|要約"
)
QUOTED_RE = re.compile(r"『(.+?)』|「(.+?)」")
NEAR_IDENTICAL_RATIO = 0.9
OVERLAP_COMPARE_CHARS = 400


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit the training dataset")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--tokenizer", default=DEFAULT_BASE_MODEL)
    p.add_argument("--split", default="train")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--short-response-tokens", type=int, default=10)
    p.add_argument("--json", type=Path, default=None, help="Write the full report")
    p.add_argument("-o", "--output", type=Path, default=None, help="Write Markdown")
    return p.parse_args(argv)


def quantiles(values: list[int]) -> dict[str, float]:
    """Percentiles (nearest rank) plus mean and max; empty input gives zeros."""
    if not values:
        return {**{f"p{q}": 0 for q in QUANTILES}, "mean": 0.0, "max": 0}
    ordered = sorted(values)
    result: dict[str, float] = {}
    for q in QUANTILES:
        index = min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))
        result[f"p{q}"] = ordered[index]
    result["mean"] = sum(ordered) / len(ordered)
    result["max"] = ordered[-1]
    return result


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def quoted_passages(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in QUOTED_RE.finditer(text)]


def list_response_shape(instruction: str, response: str) -> dict[str, Any]:
    """Shared list-answer rules plus the duplicate-line count."""
    quality = list_answer_quality(instruction, response)
    stats = duplicate_line_stats(response)
    return {
        "shape": quality["shape"],
        "items": quality["items"],
        "structured": quality["structured"],
        "count_mismatch": quality["count_mismatch"],
        "duplicate_items": quality["duplicate_items"],
        "repeated_lines": int(stats["repeated_lines"]),
    }


def rewrite_overlap(context: str, response: str) -> str:
    """How far a rewrite response moved away from the text it rewrites."""
    source, target = normalize(context), normalize(response)
    if not source or not target:
        return "no_input"
    if source == target:
        return "identical"
    ratio = SequenceMatcher(
        None, source[:OVERLAP_COMPARE_CHARS], target[:OVERLAP_COMPARE_CHARS]
    ).ratio()
    return "near_identical" if ratio >= NEAR_IDENTICAL_RATIO else "rewritten"


def audit_row(
    tokenizer: Any, row: dict[str, Any], max_length: int, short_response_tokens: int
) -> dict[str, Any] | None:
    instruction = row.get("instruction") or ""
    response = row.get("output") or row.get("response") or ""
    context = row.get("input") or ""
    if not instruction or not response:
        return {"usable": False, "reason": "empty", "category": row.get("category")}

    built = build_example_with_info(
        tokenizer, instruction, response, max_length, context=context
    )
    response_tokens = len(
        tokenizer(format_response(response), add_special_tokens=False)["input_ids"]
    )
    prompt_tokens = len(
        tokenizer(
            format_prompt_prefix(instruction, context=context), add_special_tokens=False
        )["input_ids"]
    )
    if built is None:
        return {
            "usable": False,
            "reason": "truncated_away",
            "category": row.get("category"),
            "response_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
        }
    info = built[1]
    is_list = is_list_instruction(instruction)
    requested = requested_item_count(instruction) if is_list else None
    is_rewrite = bool(REWRITE_INSTRUCTION_RE.search(instruction))
    return {
        "usable": True,
        "index": row.get("index"),
        "category": row.get("category"),
        "has_input": bool(context.strip()),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "context_truncated": info.context_truncated,
        "response_truncated": info.response_truncated,
        "short_response": response_tokens <= short_response_tokens,
        "repetition": repetition_findings(response),
        "list_instruction": is_list,
        "rewrite_instruction": is_rewrite,
        "list_shape": (list_response_shape(instruction, response) if is_list else None),
        "requested_items": requested,
        "rewrite_overlap": rewrite_overlap(context, response) if is_rewrite else None,
    }


def summarize(rows: list[dict[str, Any]], short_response_tokens: int) -> dict[str, Any]:
    usable = [r for r in rows if r["usable"]]

    def group(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(items),
            "with_input": sum(r["has_input"] for r in items),
            "context_truncated": sum(r["context_truncated"] for r in items),
            "response_truncated": sum(r["response_truncated"] for r in items),
            "short_responses": sum(r["short_response"] for r in items),
            "repetitive_responses": sum(bool(r["repetition"]) for r in items),
            "response_tokens": quantiles([r["response_tokens"] for r in items]),
            "prompt_tokens": quantiles([r["prompt_tokens"] for r in items]),
        }

    categories = sorted({r["category"] or "unknown" for r in usable})
    list_rows = [r for r in usable if r["list_instruction"]]
    rewrite_rows = [r for r in usable if r["rewrite_instruction"]]
    return {
        "rows": len(rows),
        "skipped": {
            "empty": sum(r.get("reason") == "empty" for r in rows),
            "truncated_away": sum(r.get("reason") == "truncated_away" for r in rows),
        },
        "short_response_tokens": short_response_tokens,
        "overall": group(usable),
        "by_category": {
            category: group(
                [r for r in usable if (r["category"] or "unknown") == category]
            )
            for category in categories
        },
        "repetition_kinds": dict(
            Counter(
                finding.split(" (")[0] for r in usable for finding in r["repetition"]
            )
        ),
        "list_instructions": {
            **group(list_rows),
            "shapes": dict(Counter(r["list_shape"]["shape"] for r in list_rows)),
            "structured_responses": sum(
                r["list_shape"]["structured"] for r in list_rows
            ),
            "responses_with_duplicate_items": sum(
                r["list_shape"]["duplicate_items"] > 0 for r in list_rows
            ),
            "responses_with_repeated_lines": sum(
                r["list_shape"]["repeated_lines"] > 0 for r in list_rows
            ),
            "items": quantiles([r["list_shape"]["items"] for r in list_rows]),
            "requested_count": _requested_count_summary(list_rows),
            "by_category": dict(
                Counter((r["category"] or "unknown") for r in list_rows)
            ),
        },
        "rewrite_instructions": {
            **group(rewrite_rows),
            "overlap": dict(Counter(r["rewrite_overlap"] for r in rewrite_rows)),
            "by_category": dict(
                Counter((r["category"] or "unknown") for r in rewrite_rows)
            ),
        },
    }


def _requested_count_summary(list_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For instructions that ask for N items: does the response have N items?"""
    asked = [r for r in list_rows if r["requested_items"]]
    counts = Counter()
    for row in asked:
        items = row["list_shape"]["items"]
        requested = row["requested_items"]
        if not row["list_shape"]["structured"]:
            counts["prose"] += 1
        else:
            counts[
                "exact"
                if items == requested
                else "fewer"
                if items < requested
                else "more"
            ] += 1
    return {
        "instructions_with_a_count": len(asked),
        **{key: counts[key] for key in ("exact", "fewer", "more", "prose")},
    }


def _pct(part: int, whole: int) -> str:
    return f"{part} ({part / whole:.1%})" if whole else "0"


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    overall = s["overall"]
    n = overall["examples"]
    lines = [
        "# Dataset audit",
        "",
        f"- Dataset: `{report['dataset_id']}` split `{report['split']}`, "
        f"{s['rows']} rows",
        f"- Tokenizer: `{report['tokenizer']}`, `--max-length {report['max_length']}` "
        "(the training code path, including context shortening)",
        f"- Usable examples: {n}; skipped empty {s['skipped']['empty']}, "
        f"skipped because nothing of the response fit {s['skipped']['truncated_away']}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        f"| rows with an `input` | {_pct(overall['with_input'], n)} |",
        f"| context shortened | {_pct(overall['context_truncated'], n)} |",
        f"| response cut at max-length | {_pct(overall['response_truncated'], n)} |",
        f"| response <= {s['short_response_tokens']} tokens | "
        f"{_pct(overall['short_responses'], n)} |",
        f"| response repeats lines or clauses | "
        f"{_pct(overall['repetitive_responses'], n)} |",
        "",
        "Response tokens: "
        + ", ".join(f"p{q}={overall['response_tokens'][f'p{q}']:g}" for q in QUANTILES)
        + f", mean={overall['response_tokens']['mean']:.1f}"
        + f", max={overall['response_tokens']['max']:g}",
        "Prompt tokens: "
        + ", ".join(f"p{q}={overall['prompt_tokens'][f'p{q}']:g}" for q in QUANTILES)
        + f", mean={overall['prompt_tokens']['mean']:.1f}"
        + f", max={overall['prompt_tokens']['max']:g}",
        "",
        "## By category",
        "",
        "| category | examples | with input | response p50 | response p95 | "
        f"<= {s['short_response_tokens']} tok | context shortened | response cut | "
        "repetitive |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for category, group in s["by_category"].items():
        count = group["examples"]
        lines.append(
            f"| {category} | {count} | {_pct(group['with_input'], count)} | "
            f"{group['response_tokens']['p50']:g} | {group['response_tokens']['p95']:g} | "
            f"{_pct(group['short_responses'], count)} | "
            f"{_pct(group['context_truncated'], count)} | "
            f"{_pct(group['response_truncated'], count)} | "
            f"{_pct(group['repetitive_responses'], count)} |"
        )

    list_rows = s["list_instructions"]
    listed = list_rows["examples"]
    rewrite = s["rewrite_instructions"]
    rewritten = rewrite["examples"]
    lines += [
        "",
        "## Instructions that ask for a list",
        "",
        f"- {_pct(listed, n)} of usable examples",
        "- response shape: "
        + ", ".join(
            f"{k} {_pct(v, listed)}" for k, v in sorted(list_rows["shapes"].items())
        ),
        f"- structured (>= 2 items): {_pct(list_rows['structured_responses'], listed)}"
        f", with duplicate items: "
        f"{_pct(list_rows['responses_with_duplicate_items'], listed)}",
        f"- response repeats a line: "
        f"{_pct(list_rows['responses_with_repeated_lines'], listed)}",
        f"- items per response: p50={list_rows['items']['p50']:g}, "
        f"p95={list_rows['items']['p95']:g}",
        "- instructions that ask for a specific number of items: "
        f"{_pct(list_rows['requested_count']['instructions_with_a_count'], listed)}; "
        f"response has exactly that many items "
        f"{_pct(list_rows['requested_count']['exact'], list_rows['requested_count']['instructions_with_a_count'])}, "
        f"fewer {list_rows['requested_count']['fewer']}, "
        f"more {list_rows['requested_count']['more']}, "
        f"prose {list_rows['requested_count']['prose']}",
        f"- response tokens: p50={list_rows['response_tokens']['p50']:g}, "
        f"p95={list_rows['response_tokens']['p95']:g}",
        "",
        "## Instructions that ask for a rewrite",
        "",
        f"- {_pct(rewritten, n)} of usable examples",
        "- response vs. the text to rewrite: "
        + ", ".join(f"{k} {v}" for k, v in sorted(rewrite["overlap"].items())),
        f"- response tokens: p50={rewrite['response_tokens']['p50']:g}, "
        f"p95={rewrite['response_tokens']['p95']:g}",
        "",
        "## Repetition kinds in teacher responses",
        "",
        *[
            f"- {kind}: {count}"
            for kind, count in sorted(s["repetition_kinds"].items())
        ],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset_id, split=args.split)
    rows = [
        audit_row(tokenizer, dict(row), args.max_length, args.short_response_tokens)
        for row in raw
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": public_identifier(args.dataset_id),
        "split": args.split,
        "tokenizer": public_identifier(args.tokenizer),
        "max_length": args.max_length,
        "summary": summarize(rows, args.short_response_tokens),
    }
    markdown = render_markdown(report)
    if args.json:
        write_json(args.json, report)
        print(f"Wrote {args.json}")
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
