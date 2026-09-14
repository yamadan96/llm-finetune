"""Check that comparison prompts do not overlap with the training dataset.

Usage:
    uv run python scripts/check_prompt_contamination.py \
        --output prompts/compare_ja.contamination.json

Every prompt in the prompt set is compared with the ``instruction``,
``input``, ``output`` and ``instruction + input`` fields of **all** rows of
the dataset (train and validation splits alike). Texts are normalized with
NFKC and whitespace removal before comparison.

- exact      normalized prompt equals a normalized field
- substring  prompt is contained in a field, or a field (or a quoted passage
             of the prompt) of at least ``--min-substring-chars`` characters
             is contained in the other
- similarity character 3-gram Jaccard similarity, and the fraction of the
             prompt's 3-grams that occur in the field (containment)

Exits with status 1 if any prompt has an exact or substring match or a
similarity at or above the thresholds. The report stores row indices and
scores only, no dataset text.
"""

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from src.compare import load_prompts
from src.dataset import DEFAULT_DATASET
from src.evidence import public_identifier, write_json

NGRAM = 3
FIELDS = ("instruction", "input", "output")
QUOTED_RE = re.compile(r"『(.+?)』|「(.+?)」|\"(.+?)\"")
SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check comparison prompts against the training dataset"
    )
    p.add_argument("--prompts", type=Path, default=Path("prompts/compare_ja.json"))
    p.add_argument("--dataset-id", default=DEFAULT_DATASET)
    p.add_argument("--jaccard-threshold", type=float, default=0.5)
    p.add_argument("--containment-threshold", type=float, default=0.8)
    p.add_argument("--min-substring-chars", type=int, default=20)
    p.add_argument("--output", type=Path, default=None, help="Write a JSON report")
    return p.parse_args(argv)


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def ngrams(text: str) -> frozenset[str]:
    if len(text) < NGRAM:
        return frozenset([text]) if text else frozenset()
    return frozenset(text[i : i + NGRAM] for i in range(len(text) - NGRAM + 1))


def quoted_passages(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in QUOTED_RE.finditer(text)]


def row_fields(row: dict[str, Any]) -> dict[str, str]:
    fields = {name: normalize(row.get(name) or "") for name in FIELDS}
    if fields["input"]:
        fields["instruction+input"] = fields["instruction"] + fields["input"]
    return {name: value for name, value in fields.items() if value}


def check_prompt(
    prompt: dict[str, str],
    rows: list[dict[str, Any]],
    jaccard_threshold: float,
    containment_threshold: float,
    min_substring_chars: int,
) -> dict[str, Any]:
    """Compare one prompt with every field of every row."""
    text = normalize(prompt["instruction"])
    grams = ngrams(text)
    passages = [
        p
        for p in map(normalize, quoted_passages(prompt["instruction"]))
        if len(p) >= min_substring_chars
    ]
    exact: list[dict[str, Any]] = []
    substring: list[dict[str, Any]] = []
    best_jaccard = {"score": 0.0, "row_index": None, "field": None}
    best_containment = {"score": 0.0, "row_index": None, "field": None}

    for position, row in enumerate(rows):
        row_index = row.get("index", str(position))
        for name, value in row_fields(row).items():
            hit = {"row_index": row_index, "field": name}
            if value == text:
                exact.append(hit)
            if (
                text in value
                or (len(value) >= min_substring_chars and value in text)
                or any(p in value or value == p for p in passages)
            ):
                substring.append(hit)
            value_grams = ngrams(value)
            common = len(grams & value_grams)
            if not common:
                continue
            jaccard = common / len(grams | value_grams)
            containment = common / len(grams)
            if jaccard > best_jaccard["score"]:
                best_jaccard = {"score": jaccard, **hit}
            if containment > best_containment["score"]:
                best_containment = {"score": containment, **hit}

    flagged = bool(
        exact
        or substring
        or best_jaccard["score"] >= jaccard_threshold
        or best_containment["score"] >= containment_threshold
    )
    return {
        "id": prompt["id"],
        "ok": not flagged,
        "exact_matches": exact,
        "substring_matches": substring,
        "max_jaccard": {**best_jaccard, "score": round(best_jaccard["score"], 4)},
        "max_containment": {
            **best_containment,
            "score": round(best_containment["score"], 4),
        },
    }


def _dataset_revision(dataset_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id).sha
    except Exception:  # noqa: BLE001 - offline or rate limited: not essential
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    prompts = load_prompts(args.prompts)
    raw = load_dataset(args.dataset_id, split="train")
    rows = [dict(row) for row in raw]
    results = [
        check_prompt(
            prompt,
            rows,
            args.jaccard_threshold,
            args.containment_threshold,
            args.min_substring_chars,
        )
        for prompt in prompts
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": all(r["ok"] for r in results),
        "prompts_file": args.prompts.name,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "dataset_id": public_identifier(args.dataset_id),
        "dataset_revision": _dataset_revision(args.dataset_id),
        "dataset_rows": len(rows),
        "fields": [*FIELDS, "instruction+input"],
        "normalization": "NFKC, whitespace removed",
        "thresholds": {
            "jaccard_char_3gram": args.jaccard_threshold,
            "containment_char_3gram": args.containment_threshold,
            "min_substring_chars": args.min_substring_chars,
        },
        "results": results,
    }
    for result in results:
        jac = result["max_jaccard"]
        con = result["max_containment"]
        print(
            f"[{'PASS' if result['ok'] else 'FAIL'}] {result['id']}: "
            f"exact={len(result['exact_matches'])} "
            f"substring={len(result['substring_matches'])} "
            f"max_jaccard={jac['score']:.3f} (row {jac['row_index']} {jac['field']}) "
            f"max_containment={con['score']:.3f} (row {con['row_index']} {con['field']})"
        )
    if args.output:
        write_json(args.output, report)
        print(f"Wrote {args.output}")
    print("NO OVERLAP FOUND" if report["ok"] else "OVERLAP FOUND")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
