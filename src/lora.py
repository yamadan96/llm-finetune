"""
LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
Paper: https://arxiv.org/abs/2106.09685

Core idea: For a pre-trained weight W ∈ R^(d×k), represent the update as:
    W + ΔW = W + BA  where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d, k)
Forward pass: h = Wx + BAx * (alpha / rank)
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .evidence import public_identifier

logger = logging.getLogger(__name__)

LORA_CONFIG_FILENAME = "lora_config.json"
LORA_WEIGHTS_FILENAME = "lora_weights.pt"


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.
    Freezes the original weight W and trains only A and B.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original frozen weight
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # LoRA matrices: A ∈ R^(r×k), B ∈ R^(d×r)
        # Kept in float32 on the same device as the frozen weight, so the
        # adapter trains in full precision even when the base model is bf16.
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=device))
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=device)
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        # A: kaiming uniform (as in paper); B: zeros so ΔW=0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear path (frozen)
        result = nn.functional.linear(x, self.weight, self.bias)
        # LoRA path: x → dropout → A → B → scale (computed in the adapter dtype)
        lora_x = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = lora_x @ self.lora_A.T @ self.lora_B.T
        return result + (lora_out * self.scaling).to(result.dtype)


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace target Linear layers with LoRALinear. Freeze all other params."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    # Materialize the list first: the loop below mutates the module tree
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Match whole name components only ("q_proj" must not match "kq_proj")
        if not any(name == t or name.endswith("." + t) for t in target_modules):
            continue

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        replaced += 1

    logger.info(
        "Replaced %d Linear layers with LoRALinear (rank=%d, alpha=%g)",
        replaced,
        rank,
        alpha,
    )
    return model


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only LoRA trainable parameters (lora_A and lora_B)."""
    return [p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA adapter weights (as detached CPU tensors)."""
    lora_state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }
    torch.save(lora_state, path)
    logger.info("Saved LoRA weights to %s (%d tensors)", path, len(lora_state))


def load_lora_weights(
    model: nn.Module, path: str, device: str = "cpu", strict: bool = False
) -> nn.Module:
    """Load LoRA adapter weights into model.

    With ``strict=True`` a checkpoint that does not cover every adapter
    parameter (or contains keys the model lacks) raises instead of warning,
    so a mismatched adapter cannot silently leave the base model unchanged.
    """
    state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if strict and (lora_missing or unexpected):
        raise RuntimeError(
            f"LoRA checkpoint does not match the model: missing={lora_missing}, "
            f"unexpected={list(unexpected)}"
        )
    if lora_missing:
        logger.warning("Missing LoRA keys: %s", lora_missing)
    logger.info("Loaded LoRA weights from %s", path)
    return model


def save_lora_config(
    path: str | Path,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
    base_model_id: str,
) -> None:
    """Save the adapter hyperparameters needed to rebuild the LoRA model.

    ``base_model_id`` is stored via ``public_identifier``: a Hub id is kept as
    is, while a local model path is saved as ``local:<name>`` so the file can
    be committed. Reloading such a checkpoint requires passing the local path
    again (see ``src.model.resolve_lora_settings``).
    """
    config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": list(target_modules),
        "base_model_id": public_identifier(base_model_id),
    }
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Saved LoRA config to %s", path)


def load_lora_config(path: str | Path) -> dict[str, Any] | None:
    """Load a saved LoRA config, or return None if the file does not exist."""
    config_path = Path(path)
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))
