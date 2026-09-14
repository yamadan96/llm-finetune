"""Model loading and LoRA application for Qwen2.5-7B-Instruct."""

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .evidence import LOCAL_ID_PREFIX
from .lora import (
    LORA_CONFIG_FILENAME,
    LORA_WEIGHTS_FILENAME,
    apply_lora,
    load_lora_config,
    load_lora_weights,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_base_model(
    model_id: str = DEFAULT_BASE_MODEL,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and tokenizer with bfloat16 and device_map=auto."""
    logger.info("Loading base model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def build_lora_model(
    model_id: str = DEFAULT_BASE_MODEL,
    rank: int = LORA_RANK,
    alpha: float = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    target_modules: list[str] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and apply LoRA adapters."""
    model, tokenizer = load_base_model(model_id)
    model = apply_lora(
        model,
        target_modules or LORA_TARGET_MODULES,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %d / %d (%.2f%%)",
        trainable,
        total,
        100 * trainable / total,
    )
    return model, tokenizer


def enable_gradient_checkpointing(model: PreTrainedModel) -> None:
    """Enable gradient checkpointing so gradients still reach LoRA params.

    The base weights (including embeddings) are frozen, so the embedding
    output does not require grad. ``enable_input_require_grads`` fixes that
    for reentrant checkpointing, and ``use_reentrant=False`` avoids relying
    on it at all, independent of the transformers default.
    """
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def resolve_lora_settings(
    checkpoint_dir: Path,
    model_id: str = DEFAULT_BASE_MODEL,
) -> tuple[str, dict[str, Any]]:
    """Return ``(base model id, adapter settings)`` for a checkpoint directory.

    Adapter hyperparameters are read from ``lora_config.json`` when present;
    older checkpoints without it fall back to the module defaults. A Hub id
    stored in the config takes precedence over ``model_id``. A checkpoint
    trained from a local model stores only ``local:<name>``; then ``model_id``
    must be a path whose final component is ``<name>``.
    """
    config = load_lora_config(checkpoint_dir / LORA_CONFIG_FILENAME)
    if config is None:
        logger.warning(
            "%s not found in %s; using default LoRA settings",
            LORA_CONFIG_FILENAME,
            checkpoint_dir,
        )
        config = {}
    saved_model_id = config.get("base_model_id")
    if saved_model_id and saved_model_id.startswith(LOCAL_ID_PREFIX):
        local_name = saved_model_id.removeprefix(LOCAL_ID_PREFIX)
        if Path(model_id).name != local_name:
            raise ValueError(
                f"Checkpoint was trained on a local model ({saved_model_id}); "
                f"pass the local model directory named '{local_name}' as model id"
            )
        saved_model_id = model_id
    if saved_model_id and saved_model_id != model_id:
        logger.warning(
            "Checkpoint was trained on %s (requested %s); using %s",
            saved_model_id,
            model_id,
            saved_model_id,
        )
        model_id = saved_model_id
    settings = {
        "rank": config.get("rank", LORA_RANK),
        "alpha": config.get("alpha", LORA_ALPHA),
        "dropout": config.get("dropout", LORA_DROPOUT),
        "target_modules": config.get("target_modules", LORA_TARGET_MODULES),
    }
    return model_id, settings


def attach_lora_checkpoint(
    model: PreTrainedModel,
    checkpoint_dir: Path,
    settings: dict[str, Any],
    strict: bool = False,
) -> PreTrainedModel:
    """Insert LoRA adapters into an already loaded base model and load weights.

    Used both by ``load_finetuned_model`` and by ``src.compare``, which reuses
    one loaded base model for before/after generation. With ``strict=True``
    every adapter parameter must be present in the checkpoint.
    """
    lora_path = checkpoint_dir / LORA_WEIGHTS_FILENAME
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    apply_lora(
        model,
        settings["target_modules"],
        rank=settings["rank"],
        alpha=settings["alpha"],
        dropout=settings["dropout"],
    )
    load_lora_weights(model, str(lora_path), strict=strict)
    # Newly created LoRA modules start in training mode (dropout active)
    model.eval()
    return model


def load_finetuned_model(
    checkpoint_dir: Path,
    model_id: str = DEFAULT_BASE_MODEL,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model + LoRA weights from checkpoint directory.

    Adapter hyperparameters are read from ``lora_config.json`` when present;
    older checkpoints without it fall back to the module defaults.
    """
    lora_path = checkpoint_dir / LORA_WEIGHTS_FILENAME
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    model_id, settings = resolve_lora_settings(checkpoint_dir, model_id)
    model, tokenizer = load_base_model(model_id)
    attach_lora_checkpoint(model, checkpoint_dir, settings)
    return model, tokenizer
