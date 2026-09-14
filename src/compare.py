"""Before/after generation samples for a LoRA checkpoint.

Usage:
    CHECKPOINT_DIR=./checkpoints/<run> uv run python -m src.compare

The base model is loaded once and answers every prompt with greedy decoding.
The LoRA adapter from the checkpoint is then inserted into the *same* model
instance (no second copy of the 7B weights) and the prompts are answered
again with identical settings. Results are written to ``samples.json`` and
``samples.md`` in the checkpoint directory (or ``--output-dir``).
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .dataset import SYSTEM_PROMPT, format_prompt_prefix
from .evidence import (
    collect_environment,
    public_identifier,
    redact_paths,
    write_json,
)
from .lora import LORA_CONFIG_FILENAME, LORA_WEIGHTS_FILENAME, load_lora_config
from .model import (
    DEFAULT_BASE_MODEL,
    attach_lora_checkpoint,
    load_base_model,
    resolve_lora_settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = Path("prompts") / "compare_ja.json"
DEFAULT_MAX_NEW_TOKENS = 256
SAMPLES_JSON = "samples.json"
SAMPLES_MD = "samples.md"
SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate greedy before/after samples for a LoRA checkpoint"
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("CHECKPOINT_DIR", "checkpoints")),
        help="Directory with lora_weights.pt and lora_config.json "
        "(default: $CHECKPOINT_DIR or checkpoints)",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help=f"Prompt set JSON (default: {DEFAULT_PROMPTS.as_posix()} in the repo)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write samples.json/samples.md (default: checkpoint dir)",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Base model id if the checkpoint has no lora_config.json "
        f"(default: {DEFAULT_BASE_MODEL})",
    )
    return p.parse_args(argv)


def display_path(path: Path) -> str:
    """Repository-relative (or file-name-only) form of a path for artifacts."""
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def load_prompts(path: Path) -> list[dict[str, str]]:
    """Load ``{"prompts": [{"id": ..., "instruction": ...}, ...]}``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not prompts:
        raise ValueError("Prompt file must contain a non-empty 'prompts' list")
    ids = [p.get("id") for p in prompts]
    if any(not p.get("id") or not p.get("instruction") for p in prompts):
        raise ValueError("Every prompt needs a non-empty 'id' and 'instruction'")
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt ids must be unique")
    return [{"id": p["id"], "instruction": p["instruction"]} for p in prompts]


def generation_settings(max_new_tokens: int) -> dict[str, Any]:
    """Greedy decoding settings, passed explicitly to ``generate``.

    Explicit kwargs override the model's ``generation_config.json`` (Qwen2.5
    ships ``do_sample=True``, ``temperature``, ``top_p``, ``top_k`` and
    ``repetition_penalty=1.05``), so decoding does not depend on those files.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.0,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }


@torch.inference_mode()
def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    settings: dict[str, Any],
) -> tuple[str, int]:
    """Answer one instruction with the training ChatML template.

    Returns the decoded response and the number of generated tokens.
    """
    prompt_ids = tokenizer(format_prompt_prefix(instruction), add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=model.device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=pad_token_id,
        **settings,
    )
    new_ids = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(new_ids.shape[0])


def _generate_all(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[dict[str, str]],
    settings: dict[str, Any],
    label: str,
) -> list[tuple[str, int]]:
    results = []
    for index, prompt in enumerate(prompts, 1):
        logger.info("[%s] %d/%d %s", label, index, len(prompts), prompt["id"])
        results.append(
            generate_response(model, tokenizer, prompt["instruction"], settings)
        )
    return results


def _code_fence(text: str) -> str:
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _fenced(text: str) -> str:
    fence = _code_fence(text)
    return f"{fence}text\n{text}\n{fence}"


def render_markdown(result: dict[str, Any]) -> str:
    """Human-readable view of ``samples.json`` for the README."""
    lora = result["lora_config"]
    gen = result["generation"]
    env = result["environment"]
    gpus = ", ".join(g["name"] for g in env["gpus"]) if env.get("gpus") else "none"
    max_new = gen["max_new_tokens"]
    lines = [
        "# Before/After samples",
        "",
        f"- Base model: `{result['model_id']}`",
        f"- LoRA checkpoint: `{result['checkpoint']}` "
        f"(rank {lora['rank']}, alpha {lora['alpha']}, dropout {lora['dropout']}, "
        f"target modules {', '.join(lora['target_modules'])}; "
        f"settings from {result['lora_config_source']})",
        f"- Decoding: greedy (`do_sample=False`, `num_beams=1`, "
        f"`repetition_penalty={gen['repetition_penalty']}`), "
        f"`max_new_tokens={max_new}`",
        f"- Prompt set: `{result['prompts_file']}` "
        f"(sha256 `{result['prompts_sha256'][:12]}`, {len(result['samples'])} prompts)",
        f"- System prompt: {result['system_prompt']}",
        f"- Environment: torch {env['torch']}, transformers {env['transformers']}, "
        f"GPU {gpus}, git {env.get('git_commit') or 'unknown'}",
        "",
        "Both columns come from the same loaded base model: first without the "
        "adapter, then with the LoRA weights inserted. Outputs that reached "
        "`max_new_tokens` are marked as cut off.",
    ]
    for index, sample in enumerate(result["samples"], 1):
        lines += ["", f"## {index}. {sample['id']}", "", "**Prompt**", ""]
        lines.append(_fenced(sample["instruction"]))
        for key, title in (("base", "Base model"), ("finetuned", "Fine-tuned")):
            tokens = sample[f"{key}_new_tokens"]
            cut = " — cut off at max_new_tokens" if tokens >= max_new else ""
            lines += ["", f"**{title}** ({tokens} new tokens{cut})", ""]
            lines.append(_fenced(sample[f"{key}_output"]))
    return "\n".join(lines) + "\n"


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir: Path = args.checkpoint_dir
    if not (checkpoint_dir / LORA_WEIGHTS_FILENAME).exists():
        raise FileNotFoundError(
            f"{LORA_WEIGHTS_FILENAME} not found in {checkpoint_dir}"
        )
    prompts_path = args.prompts or REPO_ROOT / DEFAULT_PROMPTS
    prompts = load_prompts(prompts_path)
    settings = generation_settings(args.max_new_tokens)
    saved_config = load_lora_config(checkpoint_dir / LORA_CONFIG_FILENAME)
    model_id, lora_settings = resolve_lora_settings(
        checkpoint_dir, args.model_id or DEFAULT_BASE_MODEL
    )

    model, tokenizer = load_base_model(model_id)
    model.eval()
    base = _generate_all(model, tokenizer, prompts, settings, "base")
    attach_lora_checkpoint(model, checkpoint_dir, lora_settings, strict=True)
    finetuned = _generate_all(model, tokenizer, prompts, settings, "fine-tuned")

    metadata = redact_paths(
        {
            "schema_version": SCHEMA_VERSION,
            "model_id": public_identifier(model_id),
            "checkpoint": checkpoint_dir.resolve().name,
            "lora_config": {
                **lora_settings,
                "base_model_id": public_identifier(model_id),
            },
            "lora_config_source": (
                LORA_CONFIG_FILENAME if saved_config is not None else "defaults"
            ),
            "prompts_file": display_path(prompts_path),
            "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
            "system_prompt": SYSTEM_PROMPT,
            "chat_template": "ChatML (src.dataset.format_prompt_prefix)",
            "generation": settings,
            "environment": collect_environment(),
        }
    )
    samples = [
        {
            "id": prompt["id"],
            "instruction": prompt["instruction"],
            "base_output": base_text,
            "base_new_tokens": base_tokens,
            "finetuned_output": ft_text,
            "finetuned_new_tokens": ft_tokens,
        }
        for prompt, (base_text, base_tokens), (ft_text, ft_tokens) in zip(
            prompts, base, finetuned, strict=True
        )
    ]
    # Generated text is kept verbatim; only the metadata is path-redacted
    result = {**metadata, "samples": samples}

    output_dir: Path = args.output_dir or checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / SAMPLES_JSON, result, redact=False)
    (output_dir / SAMPLES_MD).write_text(render_markdown(result), encoding="utf-8")
    logger.info("Wrote %s and %s", SAMPLES_JSON, SAMPLES_MD)
    return result


if __name__ == "__main__":
    run_compare(parse_args())
