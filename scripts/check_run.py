"""Pilot gate: minimum conditions before starting the full training run.

Usage:
    uv run python scripts/inspect_masking.py --output checkpoints/<run>/masking_check.json
    uv run python scripts/check_run.py checkpoints/<run>

Reads the evidence in the run directory (``metrics.json`` from src.train,
``samples.json``/``samples.md`` from src.compare, ``loss_curve.png``,
``lora_config.json`` and ``masking_check.json`` from
scripts/inspect_masking.py) and checks:

1. run-completed     the run finished (an OOM or crash is recorded as failed)
2. loss-decreases    logged train losses are finite and lower at the end
3. response-mask     supervised tokens < non-pad tokens (count sanity check)
4. mask-boundary     the real-tokenizer masking report passed
5. vram-headroom     peak reserved VRAM leaves headroom on the GPU
6. adapter-load      src.compare strictly loaded the checkpoint: adapter
                     modules present, no missing/unexpected keys, every
                     adapter tensor bit-identical to lora_weights.pt (whose
                     SHA-256 matches the file in the run directory), lora_B
                     zero before and non-zero after loading
7. generation        base and fine-tuned generation produced non-empty output
                     for every prompt (repetitive/cut-off/unchanged -> WARN)
8. prompt-overlap    the prompt set used by src.compare has a passing
                     contamination report with the same SHA-256
                     (scripts/check_prompt_contamination.py)
9. evidence          GPU, VRAM, runtime, git commit, LoRA config and loss
                     curve are recorded, without absolute local paths, the
                     current hostname or username

A gate PASS only means the minimum conditions for starting a full run are
met. It is not a judgement of training quality: WARN items and samples.md
must still be read by a person. Exits with status 1 if any check FAILs.
"""

import argparse
import getpass
import hashlib
import json
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evidence import looks_like_local_path

REPO_ROOT = Path(__file__).resolve().parent.parent
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
EDGE_FRACTION = 0.1
REPETITION_MIN_CHARS = 64
REPETITION_NGRAM = 8
MIN_DISTINCT_NGRAM_RATIO = 0.3
MAX_RESERVED_VRAM_FRACTION = 0.9
MIN_IDENTITY_LENGTH = 3
REQUIRED_MASK_CASES = (
    "no_context",
    "with_context",
    "context_truncated",
    "response_truncated",
)


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
    non_finite = [
        e["step"] for e in metrics.get("train_loss_steps", []) if e.get("loss") is None
    ]
    if non_finite:
        return Check(
            "loss-decreases", FAIL, f"non-finite train loss at steps {non_finite[:5]}"
        )
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
    detail = (
        f"supervised {supervised} / non-pad {total} tokens ({ratio:.1%}); "
        "count sanity check only, boundaries are checked by mask-boundary"
    )
    if supervised >= total:
        return Check("response-mask", FAIL, detail + ": prompt tokens are supervised")
    return Check("response-mask", PASS, detail)


def check_mask_boundary(report: dict[str, Any] | None) -> Check:
    if report is None:
        return Check(
            "mask-boundary",
            FAIL,
            "masking_check.json missing: run scripts/inspect_masking.py --output",
        )
    samples = report.get("samples", [])
    failed = [s.get("row_index") for s in samples if not s.get("ok")]
    if not samples or failed or not report.get("ok"):
        return Check("mask-boundary", FAIL, f"failed samples: {failed or 'none run'}")
    missing = sorted(set(REQUIRED_MASK_CASES) - {s.get("case") for s in samples})
    if missing or report.get("missing_cases"):
        return Check(
            "mask-boundary",
            FAIL,
            f"cases not covered: {missing or report.get('missing_cases')} "
            "(re-run scripts/inspect_masking.py)",
        )
    detail = f"{len(samples)} examples with {report.get('tokenizer')} passed"
    if report.get("warnings"):
        return Check(
            "mask-boundary",
            WARN,
            f"{detail}; {len(report['warnings'])} warning(s), e.g. "
            f"{report['warnings'][0]}",
        )
    return Check("mask-boundary", PASS, detail)


def check_vram_headroom(metrics: dict[str, Any]) -> Check:
    gpus = metrics.get("environment", {}).get("gpus") or []
    reserved = metrics.get("memory", {}).get("max_memory_reserved_gib")
    total = sum(g.get("total_memory_gib") or 0 for g in gpus)
    if reserved is None or total <= 0:
        return Check("vram-headroom", FAIL, "no GPU memory evidence recorded")
    fraction = reserved / total
    detail = f"peak reserved {reserved:.2f} / {total:.2f} GiB ({fraction:.0%})"
    if fraction > MAX_RESERVED_VRAM_FRACTION:
        return Check("vram-headroom", WARN, detail + ": little headroom")
    return Check("vram-headroom", PASS, detail)


def check_adapter_load(run_dir: Path, samples: dict[str, Any] | None) -> Check:
    adapter = (samples or {}).get("adapter_load")
    if adapter is None:
        return Check(
            "adapter-load",
            FAIL,
            "no adapter_load evidence in samples.json (re-run src.compare)",
        )
    problems = []
    if adapter.get("strict") is not True:
        problems.append("not loaded with strict=True")
    modules = adapter.get("lora_modules") or 0
    tensors = adapter.get("adapter_tensors") or 0
    if modules <= 0:
        problems.append("no LoRA modules")
    if tensors != 2 * modules or adapter.get("checkpoint_tensors") != tensors:
        problems.append(
            f"tensor counts model={tensors} checkpoint={adapter.get('checkpoint_tensors')} "
            f"modules={modules}"
        )
    for key in ("missing_keys", "unexpected_keys", "mismatched_tensors"):
        if adapter.get(key) != 0:
            problems.append(f"{key}={adapter.get(key)}")
    if adapter.get("lora_B_norm_before_load") != 0:
        problems.append("lora_B was not zero before loading")
    if not (adapter.get("lora_B_norm_after_load") or 0) > 0:
        problems.append("lora_B is zero after loading (adapter is a no-op)")
    if adapter.get("model_in_eval_mode") is not True:
        problems.append("model not in eval mode")
    if problems:
        return Check("adapter-load", FAIL, "; ".join(problems))

    detail = f"{modules} LoRA modules, {tensors} tensors bit-identical to checkpoint"
    weights = run_dir / adapter.get("weights_file", "lora_weights.pt")
    if not weights.exists():
        return Check(
            "adapter-load",
            WARN,
            f"{detail}; {weights.name} absent, hash not re-checked",
        )
    if hashlib.sha256(weights.read_bytes()).hexdigest() != adapter.get(
        "weights_sha256"
    ):
        return Check(
            "adapter-load", FAIL, f"{weights.name} differs from the loaded file"
        )
    return Check("adapter-load", PASS, f"{detail}, sha256 matches {weights.name}")


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
    empty = [
        f"{s['id']}:{key}"
        for s in items
        for key in ("base_output", "finetuned_output")
        if not s[key].strip()
    ]
    if empty:
        return Check("generation", FAIL, f"empty outputs: {empty}")
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
        "generation",
        PASS,
        f"{len(items)} prompts with non-empty base and fine-tuned generations "
        "(read samples.md)",
    )


def check_prompt_overlap(
    samples: dict[str, Any] | None, repo_root: Path = REPO_ROOT
) -> Check:
    if samples is None:
        return Check("prompt-overlap", FAIL, "samples.json missing")
    prompts_file = repo_root / samples.get("prompts_file", "")
    report_path = prompts_file.with_name(f"{prompts_file.stem}.contamination.json")
    report = _load_json(report_path) if prompts_file.suffix == ".json" else None
    if report is None:
        return Check(
            "prompt-overlap",
            FAIL,
            f"no contamination report for {samples.get('prompts_file')}: run "
            "scripts/check_prompt_contamination.py --output",
        )
    if report.get("prompts_sha256") != samples.get("prompts_sha256"):
        return Check(
            "prompt-overlap", FAIL, "contamination report is for a different prompt set"
        )
    if not report.get("ok"):
        flagged = [r["id"] for r in report.get("results", []) if not r.get("ok")]
        return Check("prompt-overlap", FAIL, f"overlapping prompts: {flagged}")
    return Check(
        "prompt-overlap",
        PASS,
        f"{len(report.get('results', []))} prompts checked against "
        f"{report.get('dataset_rows')} rows of {report.get('dataset_id')}",
    )


def _local_paths(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj] if looks_like_local_path(obj) else []
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in _local_paths(v)]
    if isinstance(obj, list):
        return [p for v in obj for p in _local_paths(v)]
    return []


def _identity_leaks(texts: list[str]) -> list[str]:
    """Current hostname/username found as whole words in artifact metadata."""
    hostname = socket.gethostname()
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        username = ""
    candidates = [
        ("hostname", hostname),
        ("hostname", hostname.split(".")[0]),
        ("username", username),
    ]
    found: list[str] = []
    for kind, value in candidates:
        if len(value) < MIN_IDENTITY_LENGTH or kind in found:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])")
        if any(pattern.search(text) for text in texts):
            found.append(kind)
    return found


def _metadata_texts(run_dir: Path, artifacts: dict[str, Any]) -> list[str]:
    """Metadata only: generated text and dataset rows may contain any word."""
    texts = [json.dumps(value, ensure_ascii=False) for value in artifacts.values()]
    samples_md = run_dir / "samples.md"
    if samples_md.exists():
        texts.append(samples_md.read_text(encoding="utf-8").split("\n## ", 1)[0])
    return texts


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
    if not (run_dir / "loss_curve.png").exists():
        missing.append("loss_curve.png")
    if missing:
        return Check("evidence", FAIL, f"missing: {', '.join(missing)}")
    masking = _load_json(run_dir / "masking_check.json") or {}
    artifacts = {
        "metrics": metrics,
        "samples": {k: v for k, v in (samples or {}).items() if k != "samples"},
        "lora_config": _load_json(run_dir / "lora_config.json") or {},
        "masking": {k: v for k, v in masking.items() if k != "samples"},
    }
    leaked = [p for value in artifacts.values() for p in _local_paths(value)]
    if leaked:
        return Check("evidence", FAIL, f"{len(leaked)} absolute local path value(s)")
    identities = _identity_leaks(_metadata_texts(run_dir, artifacts))
    if identities:
        return Check("evidence", FAIL, f"artifacts contain the {', '.join(identities)}")
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
        check_mask_boundary(_load_json(run_dir / "masking_check.json")),
        check_vram_headroom(metrics),
        check_adapter_load(run_dir, samples),
        check_generation(samples),
        check_prompt_overlap(samples),
        check_evidence(run_dir, metrics, samples),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = run_checks(args.run_dir)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    if failed:
        print(f"GATE FAIL ({len(failed)} failed): do not start the full run")
    else:
        print(
            "GATE PASS: minimum conditions for a full run are met "
            f"(not a quality judgement; {len(warned)} WARN to review, read samples.md)"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
