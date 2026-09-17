"""Reproducible run evidence: environment, runtime, peak memory and step losses.

Everything written here is meant to be committed next to a README "Results"
section, so it must not leak machine-specific details. No absolute paths,
hostnames, usernames or environment variables are recorded, and
``write_json`` redacts any string that looks like an absolute local path as a
last line of defence.
"""

import json
import logging
import math
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3
GIT_TIMEOUT_SECONDS = 5
REDACTED_PATH = "<redacted-local-path>"
LOCAL_ID_PREFIX = "local:"
_WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


# --------------------------------------------------------------------------
# Path hygiene
# --------------------------------------------------------------------------


def looks_like_local_path(value: str) -> bool:
    """True for absolute POSIX/Windows/UNC paths and home-relative paths."""
    return value.startswith(("/", "~", "\\\\")) or bool(
        _WINDOWS_ABS_PATH_RE.match(value)
    )


def public_identifier(value: str) -> str:
    """Return a shareable model/dataset identifier.

    Hub ids such as ``Qwen/Qwen2.5-7B-Instruct`` are returned unchanged. A
    local path (absolute, home-relative or an existing relative directory) is
    reduced to ``local:<final path component>`` so no directory layout leaks.
    """
    if looks_like_local_path(value) or Path(value).exists():
        return f"{LOCAL_ID_PREFIX}{Path(value).name}"
    return value


def redact_paths(obj: Any) -> Any:
    """Recursively replace string values that look like absolute local paths."""
    if isinstance(obj, str):
        return REDACTED_PATH if looks_like_local_path(obj) else obj
    if isinstance(obj, dict):
        return {key: redact_paths(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [redact_paths(value) for value in obj]
    return obj


def _json_safe_float(value: float | None) -> float | None:
    """JSON has no NaN/Infinity; store them as null."""
    if value is None or math.isfinite(value):
        return value
    return None


def write_json(path: Path, obj: Any, redact: bool = True) -> None:
    """Atomically write JSON (a crash never leaves a torn file).

    With ``redact=True`` (default) absolute-path-looking strings are replaced.
    Callers writing free text such as generated samples redact their metadata
    themselves and pass ``redact=False``.
    """
    safe = redact_paths(obj) if redact else obj
    if safe != obj:
        logger.warning("Redacted local path strings before writing %s", path.name)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------
# Environment and memory
# --------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_revision() -> dict[str, Any]:
    """Short commit SHA and whether tracked files differ from it (or None)."""
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return {"git_commit": None, "git_dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def collect_environment() -> dict[str, Any]:
    """Software versions and GPU model; CUDA fields are null without a GPU."""
    cuda_available = torch.cuda.is_available()
    gpus = None
    if cuda_available:
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / GIB, 2),
                }
            )
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "cuda": torch.version.cuda if cuda_available else None,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpus": gpus,
        **git_revision(),
    }


def reset_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        # Resetting stats before a CUDA context exists on the device raises
        # "Invalid device argument", so create the context first
        torch.empty(0, device=f"cuda:{index}")
        torch.cuda.reset_peak_memory_stats(index)


def peak_memory() -> dict[str, Any]:
    """Peak CUDA memory since the last reset, in GiB (null on CPU)."""
    if not torch.cuda.is_available():
        return {
            "max_memory_allocated_gib": None,
            "max_memory_reserved_gib": None,
            "per_device": None,
        }
    per_device = [
        {
            "index": index,
            "max_memory_allocated_gib": round(
                torch.cuda.max_memory_allocated(index) / GIB, 3
            ),
            "max_memory_reserved_gib": round(
                torch.cuda.max_memory_reserved(index) / GIB, 3
            ),
        }
        for index in range(torch.cuda.device_count())
    ]
    return {
        "max_memory_allocated_gib": round(
            sum(d["max_memory_allocated_gib"] for d in per_device), 3
        ),
        "max_memory_reserved_gib": round(
            sum(d["max_memory_reserved_gib"] for d in per_device), 3
        ),
        "per_device": per_device,
    }


def _synchronize() -> None:
    """Wait for queued CUDA kernels so wall-clock timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Run tracker
# --------------------------------------------------------------------------


class RunTracker:
    """Accumulates run evidence and rewrites ``metrics.json`` incrementally.

    The file is written on creation, every ``log_every`` optimizer steps, after
    every epoch and when the run finishes, so a crashed run still leaves its
    config, environment, step losses and completed epochs behind.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: Path,
        config: dict[str, Any],
        log_every: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        self.path = path
        self.log_every = log_every
        self._clock = clock
        reset_peak_memory()
        self._start = clock()
        self._setup_seconds: float | None = None
        self._train_seconds = 0.0
        self._eval_seconds = 0.0
        self._epoch_seconds: list[float] = []
        self._epoch_start: float | None = None
        self._phase_start: float | None = None
        self._in_training_phase = False
        self._window_loss = 0.0
        self._window_steps = 0
        self.optimizer_steps = 0
        self.tokens_trained = 0
        self.input_tokens = 0
        self.metrics: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "status": "running",
            "error": None,
            "config": dict(config),
            "environment": collect_environment(),
            "epochs": [],
            "best": None,
            "train_loss_steps": [],
            "snapshots": [],
            "runtime": {},
            "memory": {},
        }
        self.write()

    # -- timing helpers ---------------------------------------------------

    def _now(self) -> float:
        _synchronize()
        return self._clock()

    def _current_train_seconds(self, now: float) -> float:
        in_progress = (
            now - self._phase_start
            if self._in_training_phase and self._phase_start is not None
            else 0.0
        )
        return self._train_seconds + in_progress

    def _runtime(self, now: float) -> dict[str, Any]:
        train_seconds = self._current_train_seconds(now)

        def per_second(count: int) -> float | None:
            return count / train_seconds if train_seconds > 0 else None

        return {
            "total_seconds": now - self._start,
            "setup_seconds": self._setup_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": self._eval_seconds,
            "epoch_seconds": list(self._epoch_seconds),
            "optimizer_steps": self.optimizer_steps,
            "tokens_trained": self.tokens_trained,
            "tokens_per_second": per_second(self.tokens_trained),
            "input_tokens": self.input_tokens,
            "input_tokens_per_second": per_second(self.input_tokens),
        }

    # -- lifecycle --------------------------------------------------------

    def update_config(self, **fields: Any) -> None:
        self.metrics["config"] = {**self.metrics["config"], **fields}
        self.write()

    def start_training(self) -> None:
        """Mark the end of setup (model and dataset loading)."""
        self._setup_seconds = self._now() - self._start

    def start_epoch(self) -> None:
        now = self._now()
        self._epoch_start = now
        self._phase_start = now
        self._in_training_phase = True

    def record_step(
        self,
        *,
        epoch: int,
        loss: float,
        lr: float,
        supervised_tokens: int,
        input_tokens: int,
    ) -> dict[str, Any] | None:
        """Count one optimizer step; return the log entry when one is written.

        The logged ``loss`` is the mean training loss over the optimizer steps
        since the previous entry, and ``lr`` is the learning rate used by the
        logged step.
        """
        self.optimizer_steps += 1
        self.tokens_trained += supervised_tokens
        self.input_tokens += input_tokens
        self._window_loss += loss
        self._window_steps += 1
        if self.optimizer_steps % self.log_every != 0:
            return None
        entry = {
            "step": self.optimizer_steps,
            "epoch": epoch,
            "loss": _json_safe_float(self._window_loss / self._window_steps),
            "lr": lr,
        }
        self._window_loss = 0.0
        self._window_steps = 0
        self.metrics["train_loss_steps"].append(entry)
        self.write()
        return entry

    def record_snapshot(self, *, step: int, val_loss: float | None) -> None:
        """Record an adapter snapshot taken mid-run (see --snapshot-every)."""
        self.metrics["snapshots"].append(
            {
                "step": step,
                "val_loss": _json_safe_float(val_loss),
                "seconds": self._now() - self._start,
            }
        )
        self.write()

    def end_epoch_training(self) -> None:
        """Mark the end of the training phase of the current epoch."""
        now = self._now()
        if self._in_training_phase and self._phase_start is not None:
            self._train_seconds += now - self._phase_start
        self._in_training_phase = False
        self._phase_start = now

    def end_epoch(self, entry: dict[str, Any]) -> None:
        """Append an epoch summary; time since ``end_epoch_training`` is eval."""
        if self._in_training_phase:
            self.end_epoch_training()
        now = self._now()
        if self._phase_start is not None:
            self._eval_seconds += now - self._phase_start
        if self._epoch_start is not None:
            self._epoch_seconds.append(now - self._epoch_start)
        self._epoch_start = None
        self._phase_start = None
        self.metrics["epochs"].append(
            {
                key: _json_safe_float(value) if isinstance(value, float) else value
                for key, value in entry.items()
            }
        )
        self.write()

    def set_best(self, best: dict[str, Any]) -> None:
        self.metrics["best"] = best

    def finish(self, status: str, error: str | None = None) -> None:
        """Record the final status; ``error`` should be an exception type name."""
        self.metrics["status"] = status
        self.metrics["error"] = error
        self.write()

    def write(self) -> None:
        now = self._now()
        self.metrics["runtime"] = self._runtime(now)
        self.metrics["memory"] = peak_memory()
        write_json(self.path, self.metrics)
