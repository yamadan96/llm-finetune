"""
LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
Paper: https://arxiv.org/abs/2106.09685

Core idea: For a pre-trained weight W ∈ R^(d×k), represent the update as:
    W + ΔW = W + BA  where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d, k)
Forward pass: h = Wx + BAx * (alpha / rank)
"""
import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.lora_dropout = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

        self._init_lora_weights()

    def _init_lora_weights(self) -> None:
        # A: kaiming uniform (as in paper); B: zeros so ΔW=0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear path (frozen)
        result = nn.functional.linear(x, self.weight, self.bias)
        # LoRA path: x → dropout → A → B → scale
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling


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
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
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
    """Save only LoRA adapter weights."""
    lora_state = {
        n: p for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n
    }
    torch.save(lora_state, path)
    logger.info("Saved LoRA weights to %s (%d tensors)", path, len(lora_state))


def load_lora_weights(
    model: nn.Module, path: str, device: str = "cpu"
) -> nn.Module:
    """Load LoRA adapter weights into model."""
    state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        logger.warning("Missing LoRA keys: %s", lora_missing)
    logger.info("Loaded LoRA weights from %s", path)
    return model
