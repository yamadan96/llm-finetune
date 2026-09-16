"""List-behaviour metrics for one or more samples.json files.

Usage:
    uv run python scripts/list_metrics.py \
        checkpoints/<run-a>/list_eval/samples.json \
        checkpoints/<run-b>/list_eval/samples.json -o list_metrics.md

For the base and the fine-tuned answer of every prompt it applies the shared
rules in ``src/list_rules.py`` and reports, per run:

- structured answers: at least two items as a marked list, several lines or a
  one-line enumeration (prose counts as unstructured)
- requested-count satisfaction: among prompts whose instruction states a
  number of items (``expected_items`` in the prompt set), how often the answer
  has exactly that many items
- duplicate items: answers containing the same item twice, and how many
  duplicated items they contain in total
- termination: answers that stopped before ``max_new_tokens``
- answer length in items and generated tokens

These are structural checks, not a judgement of content. Runs must share the
prompt set and the generation settings; the script refuses otherwise.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.list_rules import duplicate_items, response_items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List-behaviour metrics per run")
    p.add_argument("samples", nargs="+", type=Path, help="samples.json files")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args(argv)


def answer_metrics(instruction_items: int | None, answer: str) -> dict[str, Any]:
    shape, items = response_items(answer)
    duplicates = duplicate_items(items)
    structured = shape != "prose" and len(items) >= 2
    return {
        "shape": shape,
        "items": len(items),
        "structured": structured,
        "duplicates": duplicates,
        "exact_count": (
            None if instruction_items is None else len(items) == instruction_items
        ),
    }


def run_metrics(samples: dict[str, Any], key: str) -> dict[str, Any]:
    """Metrics over all prompts for ``base_output`` or ``finetuned_output``."""
    items = samples["samples"]
    max_new = samples["generation"]["max_new_tokens"]
    token_key = key.replace("_output", "_new_tokens")
    per_prompt = [
        answer_metrics(sample.get("expected_items"), sample[key]) for sample in items
    ]
    with_count = [m for m in per_prompt if m["exact_count"] is not None]
    lengths = sorted(sample[token_key] for sample in items)
    return {
        "prompts": len(items),
        "structured": sum(m["structured"] for m in per_prompt),
        "structured_rate": sum(m["structured"] for m in per_prompt) / len(items),
        "with_expected_count": len(with_count),
        "exact_count": sum(m["exact_count"] for m in with_count),
        "exact_count_rate": (
            sum(m["exact_count"] for m in with_count) / len(with_count)
            if with_count
            else None
        ),
        "answers_with_duplicates": sum(m["duplicates"] > 0 for m in per_prompt),
        "duplicate_items_total": sum(m["duplicates"] for m in per_prompt),
        "terminated": sum(sample[token_key] < max_new for sample in items),
        "median_items": sorted(m["items"] for m in per_prompt)[len(per_prompt) // 2],
        "median_new_tokens": lengths[len(lengths) // 2],
        "shapes": {
            shape: sum(m["shape"] == shape for m in per_prompt)
            for shape in ("marked_list", "multi_line", "inline_list", "prose")
        },
    }


def consistency_problems(runs: list[dict[str, Any]]) -> list[str]:
    reference = runs[0]
    problems = []
    for run in runs[1:]:
        for field in ("prompts_sha256", "generation"):
            if run["samples"].get(field) != reference["samples"].get(field):
                problems.append(f"{run['name']}: {field} differs")
    return problems


def render(runs: list[dict[str, Any]]) -> str:
    columns = [
        ("structured_rate", "structured"),
        ("exact_count_rate", "exact count"),
        ("answers_with_duplicates", "answers with duplicates"),
        ("duplicate_items_total", "duplicate items"),
        ("terminated", "terminated"),
        ("median_items", "median items"),
        ("median_new_tokens", "median tokens"),
    ]
    first = runs[0]
    n = first["finetuned"]["prompts"]
    counted = first["finetuned"]["with_expected_count"]
    lines = [
        "# List behaviour",
        "",
        f"- Prompt set: `{first['samples']['prompts_file']}` ({n} prompts, "
        f"{counted} with a requested item count), sha256 "
        f"`{first['samples']['prompts_sha256'][:12]}`",
        "- Structural rules: `src/list_rules.py`. Content is not judged here.",
        "",
        "| run | answer | " + " | ".join(title for _, title in columns) + " |",
        "|" + "---|" * (len(columns) + 2),
    ]
    for run in runs:
        for label in ("base", "finetuned"):
            metrics = run[label]
            cells = []
            for key, _ in columns:
                value = metrics[key]
                if key.endswith("_rate"):
                    cells.append("–" if value is None else f"{value:.0%}")
                elif key in {"answers_with_duplicates", "terminated"}:
                    cells.append(f"{value}/{metrics['prompts']}")
                else:
                    cells.append(str(value))
            lines.append(f"| {run['name']} | {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Answer shapes (fine-tuned)",
        "",
        "| run | "
        + " | ".join(("marked_list", "multi_line", "inline_list", "prose"))
        + " |",
        "|" + "---|" * 5,
    ]
    for run in runs:
        shapes = run["finetuned"]["shapes"]
        lines.append(
            f"| {run['name']} | "
            + " | ".join(
                str(shapes[s])
                for s in ("marked_list", "multi_line", "inline_list", "prose")
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = []
    for path in args.samples:
        samples = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            {
                "name": samples.get("checkpoint") or path.parent.name,
                "samples": samples,
                "base": run_metrics(samples, "base_output"),
                "finetuned": run_metrics(samples, "finetuned_output"),
            }
        )
    problems = consistency_problems(runs)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    markdown = render(runs)
    if args.json:
        args.json.write_text(
            json.dumps(
                [{k: run[k] for k in ("name", "base", "finetuned")} for run in runs],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
