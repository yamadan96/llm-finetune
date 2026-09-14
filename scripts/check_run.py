"""Go/no-go gate for a pilot run before starting the full training run.

Usage:
    uv run python scripts/check_run.py checkpoints/<run>

Reads ``metrics.json`` (from src.train) and ``samples.json`` (from
src.compare) in the run directory and checks, from the recorded evidence:

1. run-completed   the run finished (an OOM or crash is recorded as failed)
2. loss-decreases  logged train loss at the end is below the start
3. response-mask   supervised tokens are a strict, non-zero subset of tokens
4. generation      fine-tuned outputs exist, are non-empty and not degenerate
5. evidence        GPU, peak VRAM, runtime, git commit and LoRA config are
                   recorded, and no absolute local paths are stored

Exits with status 1 if any check FAILs. WARN items need a human look (for
example reading samples.md) but do not block. Passing is necessary, not
sufficient: the samples still have to be read before a full run.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evidence import looks_like_local_path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
EDGE_FRACTION = 0.1
REPETITION_MIN_CHARS = 64
REPETITION_NGRAM = 8
MIN_DISTINCT_NGRAM_RATIO = 0.3


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check pilot-run evidence (metrics.json, samples.json) "
        "before starting a full run"
    )
    p.add_argument("run_dir", type=Path, help="Checkpoint directory of the run")
    return p.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_completed(metrics: dict[str, Any]) -> Check:
    status = metrics.get("status")
    if status == "completed":
        return Check("run-completed", PASS, "status=completed")
    error = metrics.get("error")
    hint = " (CUDA out of memory)" if error and "OutOfMemory" in error else ""
    return Check("run-completed", FAIL, f"status={status} error={error}{hint}")


def check_loss_decreases(metrics: dict[str, Any]) -> Check:
    losses = [
        e["loss"]
        for e in metrics.get("train_loss_steps", [])
        if e.get("loss") is not None
    ]
    if len(losses) < 2:
        return Check(
            "loss-decreases", FAIL, f"need >= 2 logged steps, got {len(losses)}"
        )
    k = max(1, int(len(losses) * EDGE_FRACTION))
    start = sum(losses[:k]) / k
    end = sum(losses[-k:]) / k
    val = [e["val_loss"] for e in metrics.get("epochs", []) if "val_loss" in e]
    detail = f"train loss first {k} logs={start:.4f} -> last {k} logs={end:.4f}"
    if (
        val
        and any(v is None for v in val)
        and metrics.get("config", {}).get("val_ratio")
    ):
        return Check("loss-decreases", FAIL, detail + "; non-finite validation loss")
    if end >= start:
        return Check("loss-decreases", FAIL, detail)
    return Check("loss-decreases", PASS, detail)


def check_response_mask(metrics: dict[str, Any]) -> Check:
    runtime = metrics.get("runtime", {})
    supervised = runtime.get("tokens_trained") or 0
    total = runtime.get("input_tokens") or 0
    if supervised <= 0 or total <= 0:
        return Check("response-mask", FAIL, "no supervised/input token counts")
    ratio = supervised / total
    detail = f"supervised {supervised} / non-pad {total} tokens ({ratio:.1%})"
    if supervised >= total:
        return Check("response-mask", FAIL, detail + ": prompt tokens are supervised")
    return Check("response-mask", PASS, detail)


def _is_repetitive(text: str) -> bool:
    if len(text) < REPETITION_MIN_CHARS:
        return False
    grams = [
        text[i : i + REPETITION_NGRAM] for i in range(len(text) - REPETITION_NGRAM + 1)
    ]
    return len(set(grams)) / len(grams) < MIN_DISTINCT_NGRAM_RATIO


def check_generation(samples: dict[str, Any] | None) -> Check:
    if samples is None:
        return Check(
            "generation", FAIL, "samples.json missing: run `python -m src.compare`"
        )
    items = samples.get("samples", [])
    if not items:
        return Check("generation", FAIL, "samples.json has no samples")
    max_new = samples.get("generation", {}).get("max_new_tokens")
    empty = [s["id"] for s in items if not s["finetuned_output"].strip()]
    if empty:
        return Check("generation", FAIL, f"empty fine-tuned outputs: {empty}")
    repetitive = [s["id"] for s in items if _is_repetitive(s["finetuned_output"])]
    cut_off = [
        s["id"]
        for s in items
        if max_new is not None and s["finetuned_new_tokens"] >= max_new
    ]
    unchanged = all(s["finetuned_output"] == s["base_output"] for s in items)
    notes = []
    if repetitive:
        notes.append(f"repetitive: {repetitive}")
    if cut_off:
        notes.append(f"cut off at max_new_tokens: {cut_off}")
    if unchanged:
        notes.append("fine-tuned outputs identical to base for every prompt")
    if notes:
        return Check("generation", WARN, "; ".join(notes) + " (read samples.md)")
    return Check(
        "generation", PASS, f"{len(items)} non-empty outputs (read samples.md)"
    )


def _local_paths(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj] if looks_like_local_path(obj) else []
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in _local_paths(v)]
    if isinstance(obj, list):
        return [p for v in obj for p in _local_paths(v)]
    return []


def check_evidence(
    run_dir: Path, metrics: dict[str, Any], samples: dict[str, Any] | None
) -> Check:
    env = metrics.get("environment", {})
    missing = []
    if not env.get("gpus"):
        missing.append("environment.gpus")
    if env.get("cuda") is None:
        missing.append("environment.cuda")
    if metrics.get("memory", {}).get("max_memory_allocated_gib") is None:
        missing.append("memory.max_memory_allocated_gib")
    if not metrics.get("runtime", {}).get("total_seconds"):
        missing.append("runtime.total_seconds")
    if not env.get("git_commit"):
        missing.append("environment.git_commit")
    if not (run_dir / "lora_config.json").exists():
        missing.append("lora_config.json")
    if missing:
        return Check("evidence", FAIL, f"missing: {', '.join(missing)}")
    metadata = {k: v for k, v in (samples or {}).items() if k != "samples"}
    lora_config = _load_json(run_dir / "lora_config.json") or {}
    leaked = _local_paths(metrics) + _local_paths(metadata) + _local_paths(lora_config)
    if leaked:
        return Check("evidence", FAIL, f"{len(leaked)} absolute local path value(s)")
    if env.get("git_dirty") is not False:
        return Check(
            "evidence",
            WARN,
            f"git commit {env['git_commit']} but git_dirty={env.get('git_dirty')}: "
            "uncommitted code changes are not reproducible",
        )
    return Check(
        "evidence", PASS, f"git {env['git_commit']}, GPU {env['gpus'][0]['name']}"
    )


def run_checks(run_dir: Path) -> list[Check]:
    metrics = _load_json(run_dir / "metrics.json")
    if metrics is None:
        return [Check("metrics", FAIL, "metrics.json missing")]
    samples = _load_json(run_dir / "samples.json")
    return [
        check_completed(metrics),
        check_loss_decreases(metrics),
        check_response_mask(metrics),
        check_generation(samples),
        check_evidence(run_dir, metrics, samples),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = run_checks(args.run_dir)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    failed = [c for c in checks if c.status == FAIL]
    print("GO" if not failed else f"NO-GO ({len(failed)} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
