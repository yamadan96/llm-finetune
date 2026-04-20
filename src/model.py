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

from .lora import apply_lora, load_lora_weights

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
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and apply LoRA adapters."""
    model, tokenizer = load_base_model(model_id)
    model = apply_lora(
        model, LORA_TARGET_MODULES, rank=rank, alpha=alpha, dropout=dropout
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


def load_finetuned_model(
    checkpoint_dir: Path,
    model_id: str = DEFAULT_BASE_MODEL,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model + LoRA weights from checkpoint directory."""
    model, tokenizer = build_lora_model(model_id)
    lora_path = checkpoint_dir / "lora_weights.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    load_lora_weights(model, str(lora_path))
    model.eval()
    return model, tokenizer
