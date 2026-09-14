import json

import pytest
import torch
import torch.nn as nn

from src.lora import (
    LORA_CONFIG_FILENAME,
    LoRALinear,
    apply_lora,
    get_lora_params,
    load_lora_config,
    load_lora_weights,
    save_lora_config,
    save_lora_weights,
)


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 4)
        self.v_proj = nn.Linear(8, 4)
        self.kq_proj = nn.Linear(8, 8)  # suffix-only match must be ignored


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyAttention(), ToyAttention()])
        self.norm = nn.LayerNorm(8)


def test_lora_linear_matches_base_linear_at_init() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(16, 8)
    lora = LoRALinear(linear, rank=4, alpha=8.0)
    x = torch.randn(3, 5, 16)

    assert torch.equal(lora(x), linear(x))


def test_lora_linear_output_changes_after_b_update() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(16, 8)
    lora = LoRALinear(linear, rank=4, alpha=8.0)
    x = torch.randn(3, 16)

    with torch.no_grad():
        lora.lora_B.normal_()
    expected = linear(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * 2.0

    assert not torch.allclose(lora(x), linear(x))
    assert torch.allclose(lora(x), expected, atol=1e-6)


def test_lora_linear_scaling_is_alpha_over_rank() -> None:
    lora = LoRALinear(nn.Linear(4, 4), rank=16, alpha=32.0)

    assert lora.scaling == 2.0


def test_lora_linear_with_bfloat16_base_keeps_output_dtype() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(16, 8).to(torch.bfloat16)
    lora = LoRALinear(linear, rank=4, alpha=8.0)
    x = torch.randn(2, 16, dtype=torch.bfloat16)

    out = lora(x)

    assert lora.lora_A.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, linear(x))


def test_apply_lora_replaces_only_matching_linears_and_freezes_base() -> None:
    model = ToyModel()

    apply_lora(model, ["q_proj", "v_proj"], rank=2, alpha=4.0)

    replaced = {n for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    assert replaced == {
        "layers.0.q_proj",
        "layers.0.v_proj",
        "layers.1.q_proj",
        "layers.1.v_proj",
    }
    for layer in model.layers:
        assert type(layer.k_proj) is nn.Linear
        assert type(layer.kq_proj) is nn.Linear
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.endswith(("lora_A", "lora_B")) for n in trainable)
    assert len(trainable) == 8
    assert len(get_lora_params(model)) == 8


def test_save_and_load_lora_weights_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    source = apply_lora(ToyModel(), ["q_proj", "v_proj"], rank=2, alpha=4.0)
    with torch.no_grad():
        for p in get_lora_params(source):
            p.normal_()
    weights_path = tmp_path / "lora_weights.pt"
    save_lora_weights(source, str(weights_path))

    saved = torch.load(weights_path, weights_only=True)
    assert all(t.device.type == "cpu" and not t.requires_grad for t in saved.values())

    target = apply_lora(ToyModel(), ["q_proj", "v_proj"], rank=2, alpha=4.0)
    load_lora_weights(target, str(weights_path))

    source_state = dict(source.named_parameters())
    for name, param in target.named_parameters():
        if "lora_" in name:
            assert torch.equal(param, source_state[name])


def test_save_and_load_lora_config_round_trip(tmp_path) -> None:
    path = tmp_path / LORA_CONFIG_FILENAME
    save_lora_config(
        path,
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        base_model_id="org/base",
    )

    assert load_lora_config(path) == {
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_id": "org/base",
    }
    assert json.loads(path.read_text())["rank"] == 8


def test_load_lora_config_missing_file_returns_none(tmp_path) -> None:
    assert load_lora_config(tmp_path / LORA_CONFIG_FILENAME) is None


def test_load_lora_weights_strict_rejects_missing_adapter_keys(tmp_path) -> None:
    source = apply_lora(ToyModel(), ["q_proj"], rank=2, alpha=4.0)
    weights_path = tmp_path / "lora_weights.pt"
    save_lora_weights(source, str(weights_path))
    target = apply_lora(ToyModel(), ["q_proj", "v_proj"], rank=2, alpha=4.0)

    load_lora_weights(target, str(weights_path))  # non-strict only warns
    with pytest.raises(RuntimeError, match="does not match"):
        load_lora_weights(target, str(weights_path), strict=True)
