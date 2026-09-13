"""Model loading and LoRA application for Qwen2.5-7B-Instruct."""

import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .lora import (
    LORA_CONFIG_FILENAME,
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


def load_finetuned_model(
    checkpoint_dir: Path,
    model_id: str = DEFAULT_BASE_MODEL,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model + LoRA weights from checkpoint directory.

    Adapter hyperparameters are read from ``lora_config.json`` when present;
    older checkpoints without it fall back to the module defaults.
    """
    lora_path = checkpoint_dir / "lora_weights.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    config = load_lora_config(checkpoint_dir / LORA_CONFIG_FILENAME)
    if config is None:
        logger.warning(
            "%s not found in %s; using default LoRA settings",
            LORA_CONFIG_FILENAME,
            checkpoint_dir,
        )
        config = {}
    saved_model_id = config.get("base_model_id")
    if saved_model_id and saved_model_id != model_id:
        logger.warning(
            "Checkpoint was trained on %s (requested %s); using %s",
            saved_model_id,
            model_id,
            saved_model_id,
        )
        model_id = saved_model_id

    model, tokenizer = build_lora_model(
        model_id,
        rank=config.get("rank", LORA_RANK),
        alpha=config.get("alpha", LORA_ALPHA),
        dropout=config.get("dropout", LORA_DROPOUT),
        target_modules=config.get("target_modules", LORA_TARGET_MODULES),
    )
    load_lora_weights(model, str(lora_path))
    model.eval()
    return model, tokenizer
