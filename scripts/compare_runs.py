"""Compare pilot runs that differ in exactly the variables you intend to vary.

Usage:
    uv run python scripts/compare_runs.py \
        checkpoints/pilot-lr2e-4 checkpoints/pilot-lr1e-4 checkpoints/pilot-lr5e-5 \
        --vary lr --samples eval20/samples.json \
        [--judgments judgments.json] -o comparison.md

For every run directory it reads ``metrics.json`` and the samples file given
by ``--samples`` (relative to the run directory) and

1. verifies that the runs share every training config value except the
   ``--vary`` keys, and that their samples used the same prompt set, the same
   generation settings and the same base model; otherwise it exits with 1
2. writes a Markdown table: varied values, final/best validation loss, the
   mean of the last logged train losses, repetition WARNs (src.text_checks)
   for fine-tuned and base outputs, outputs cut off at max_new_tokens, empty
   outputs, runtime and peak VRAM
3. optionally adds per-run counts of manual judgments (``improved``,
   ``degraded``, ``same``/``mixed``) from a JSON file
   ``{"<run dir name>": {"<prompt id>": "improved", ...}, ...}``
4. appends every prompt with the base output and each run's fine-tuned output
   side by side for reading

No column ranks the runs: the lowest loss is not the best run by itself.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.text_checks import repetition_findings

# Config keys that legitimately differ between otherwise identical runs
RUN_SPECIFIC_CONFIG_KEYS = {"log_every"}
LAST_LOGS_FOR_TRAIN_LOSS = 3
JUDGMENT_LABELS = ("improved", "degraded", "same", "mixed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pilot runs side by side")
    p.add_argument("runs", nargs="+", type=Path, help="Run directories")
    p.add_argument(
        "--vary",
        nargs="+",
        default=["lr"],
        help="metrics.json config keys that are allowed to differ",
    )
    p.add_argument(
        "--samples",
        default="samples.json",
        help="Samples file relative to each run directory",
    )
    p.add_argument("--judgments", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    return p.parse_args(argv)


def load_run(run_dir: Path, samples_name: str) -> dict[str, Any]:
    return {
        "name": run_dir.resolve().name,
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "samples": json.loads((run_dir / samples_name).read_text(encoding="utf-8")),
    }


def consistency_problems(runs: list[dict[str, Any]], vary: list[str]) -> list[str]:
    """Differences other than the intended ones (empty list if none)."""
    problems = []
    reference = runs[0]
    ignored = set(vary) | RUN_SPECIFIC_CONFIG_KEYS
    ref_config = reference["metrics"]["config"]
    for run in runs[1:]:
        config = run["metrics"]["config"]
        for key in sorted((set(ref_config) | set(config)) - ignored):
            if ref_config.get(key) != config.get(key):
                problems.append(
                    f"{run['name']}: config.{key}={config.get(key)!r} "
                    f"differs from {reference['name']} ({ref_config.get(key)!r})"
                )
        for key in ("prompts_sha256", "generation", "model_id", "system_prompt"):
            if run["samples"].get(key) != reference["samples"].get(key):
                problems.append(f"{run['name']}: samples.{key} differs")
        ids = [s["id"] for s in run["samples"]["samples"]]
        if ids != [s["id"] for s in reference["samples"]["samples"]]:
            problems.append(f"{run['name']}: prompt ids differ")
    if len({run["name"] for run in runs}) != len(runs):
        problems.append("run directory names must be unique")
    return problems


def run_row(run: dict[str, Any], vary: list[str]) -> dict[str, Any]:
    metrics, samples = run["metrics"], run["samples"]
    items = samples["samples"]
    max_new = samples["generation"]["max_new_tokens"]
    epochs = metrics.get("epochs", [])
    losses = [
        e["loss"] for e in metrics.get("train_loss_steps", []) if e["loss"] is not None
    ]
    last = losses[-LAST_LOGS_FOR_TRAIN_LOSS:]
    runtime = metrics.get("runtime", {})
    memory = metrics.get("memory", {})
    return {
        "run": run["name"],
        **{key: metrics["config"].get(key) for key in vary},
        "status": metrics.get("status"),
        "final_val_loss": epochs[-1]["val_loss"] if epochs else None,
        "best_val_loss": (metrics.get("best") or {}).get("loss"),
        "last_train_loss": sum(last) / len(last) if last else None,
        "repetition_warn_finetuned": sum(
            bool(repetition_findings(s["finetuned_output"])) for s in items
        ),
        "repetition_warn_base": sum(
            bool(repetition_findings(s["base_output"])) for s in items
        ),
        "cut_off_finetuned": sum(s["finetuned_new_tokens"] >= max_new for s in items),
        "cut_off_base": sum(s["base_new_tokens"] >= max_new for s in items),
        "empty_finetuned": sum(not s["finetuned_output"].strip() for s in items),
        "identical_to_base": sum(
            s["finetuned_output"] == s["base_output"] for s in items
        ),
        "train_seconds": runtime.get("train_seconds"),
        "total_seconds": runtime.get("total_seconds"),
        "peak_allocated_gib": memory.get("max_memory_allocated_gib"),
        "peak_reserved_gib": memory.get("max_memory_reserved_gib"),
        "git_commit": metrics.get("environment", {}).get("git_commit"),
    }


def judgment_counts(labels: dict[str, str], prompt_ids: list[str]) -> dict[str, Any]:
    unknown = sorted(set(labels) - set(prompt_ids))
    bad = sorted({v for v in labels.values()} - set(JUDGMENT_LABELS))
    if unknown or bad:
        raise ValueError(f"judgments: unknown prompt ids {unknown}, labels {bad}")
    counts = {
        label: sum(v == label for v in labels.values()) for label in JUDGMENT_LABELS
    }
    return {**counts, "unjudged": len(prompt_ids) - len(labels)}


def _fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def _fence(text: str) -> str:
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def render(
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    vary: list[str],
    judgments: dict[str, dict[str, str]] | None,
) -> str:
    items = runs[0]["samples"]["samples"]
    n = len(items)
    columns = [
        ("run", "run"),
        *[(key, key) for key in vary],
        ("status", "status"),
        ("final_val_loss", "final val loss"),
        ("best_val_loss", "best val loss"),
        ("last_train_loss", f"train loss (last {LAST_LOGS_FOR_TRAIN_LOSS} logs)"),
        ("repetition_warn_finetuned", f"repetition WARN FT/{n}"),
        ("repetition_warn_base", f"repetition WARN base/{n}"),
        ("cut_off_finetuned", "cut off FT"),
        ("identical_to_base", "identical to base"),
        ("train_seconds", "train s"),
        ("total_seconds", "total s"),
        ("peak_reserved_gib", "peak reserved GiB"),
        ("peak_allocated_gib", "peak allocated GiB"),
        ("git_commit", "git"),
    ]
    lines = [
        "# Run comparison",
        "",
        f"- Varied: {', '.join(vary)}; all other training config values are identical "
        "(checked).",
        f"- Prompt set: `{runs[0]['samples']['prompts_file']}` ({n} prompts, sha256 "
        f"`{runs[0]['samples']['prompts_sha256'][:12]}`), identical generation settings "
        "(checked).",
        "- Repetition WARN = duplicate lines, repeated clauses or looping phrases "
        "(src/text_checks.py). No column ranks the runs.",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[key]) for key, _ in columns) + " |")

    if judgments is not None:
        lines += [
            "",
            "## Manual judgments (fine-tuned vs. base, per prompt)",
            "",
            "| run | improved | degraded | same | mixed | unjudged |",
            "|---|---|---|---|---|---|",
        ]
        ids = [s["id"] for s in items]
        for run in runs:
            counts = judgment_counts(judgments.get(run["name"], {}), ids)
            lines.append(
                f"| {run['name']} | {counts['improved']} | {counts['degraded']} | "
                f"{counts['same']} | {counts['mixed']} | {counts['unjudged']} |"
            )

    lines += ["", "## Outputs"]
    for index, base_item in enumerate(items):
        category = f" ({base_item['category']})" if base_item.get("category") else ""
        lines += ["", f"### {index + 1}. {base_item['id']}{category}", ""]
        lines.append(_fence(base_item["instruction"]))
        if base_item.get("input"):
            lines += ["", "input:", "", _fence(base_item["input"])]
        lines += ["", "**base**", "", _fence(base_item["base_output"])]
        for run in runs:
            sample = run["samples"]["samples"][index]
            findings = repetition_findings(sample["finetuned_output"])
            label = (judgments or {}).get(run["name"], {}).get(sample["id"])
            notes = "; ".join([*([f"judged {label}"] if label else []), *findings])
            lines += [
                "",
                f"**{run['name']}**" + (f" ({notes})" if notes else ""),
                "",
                _fence(sample["finetuned_output"]),
            ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [load_run(run_dir, args.samples) for run_dir in args.runs]
    problems = consistency_problems(runs, args.vary)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    rows = [run_row(run, args.vary) for run in runs]
    judgments = (
        json.loads(args.judgments.read_text(encoding="utf-8"))
        if args.judgments
        else None
    )
    markdown = render(runs, rows, args.vary, judgments)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
