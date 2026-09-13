import torch

import src.model as model_module
from src.lora import LoRALinear, apply_lora, save_lora_config, save_lora_weights
from src.model import enable_gradient_checkpointing, load_finetuned_model
from tests.conftest import build_tiny_causal_lm


def test_gradient_checkpointing_reaches_lora_params_only(tiny_model) -> None:
    apply_lora(tiny_model, ["q_proj", "v_proj"], rank=4, alpha=8.0)
    enable_gradient_checkpointing(tiny_model)
    tiny_model.train()
    # Make B non-zero so lora_A also receives a non-zero gradient
    with torch.no_grad():
        for module in tiny_model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.02)
    input_ids = torch.randint(3, 100, (2, 12))

    loss = tiny_model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert tiny_model.is_gradient_checkpointing
    lora_names = [n for n, _ in tiny_model.named_parameters() if "lora_" in n]
    assert len(lora_names) == 2 * 2 * 2  # layers * (q, v) * (A, B)
    for name, param in tiny_model.named_parameters():
        if "lora_" in name:
            assert param.grad is not None, name
            assert param.grad.abs().sum() > 0, name
        else:
            assert param.grad is None, name


def test_load_finetuned_model_uses_saved_lora_config(tmp_path, monkeypatch) -> None:
    source = apply_lora(build_tiny_causal_lm(), ["q_proj"], rank=3, alpha=6.0)
    with torch.no_grad():
        for name, param in source.named_parameters():
            if "lora_" in name:
                param.normal_()
    save_lora_weights(source, str(tmp_path / "lora_weights.pt"))
    save_lora_config(
        tmp_path / "lora_config.json",
        rank=3,
        alpha=6.0,
        dropout=0.0,
        target_modules=["q_proj"],
        base_model_id="tiny/base",
    )
    requested: list[str] = []

    def fake_load_base_model(model_id):
        requested.append(model_id)
        return build_tiny_causal_lm(), None

    monkeypatch.setattr(model_module, "load_base_model", fake_load_base_model)

    model, _ = load_finetuned_model(tmp_path, model_id="other/base")

    assert requested == ["tiny/base"]
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) == 2  # q_proj only, one per layer
    assert all(m.rank == 3 and m.scaling == 2.0 for m in lora_layers)
    source_state = dict(source.named_parameters())
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert torch.equal(param, source_state[name])


def test_load_finetuned_model_without_config_uses_defaults(
    tmp_path, monkeypatch
) -> None:
    source = apply_lora(
        build_tiny_causal_lm(),
        model_module.LORA_TARGET_MODULES,
        rank=model_module.LORA_RANK,
        alpha=model_module.LORA_ALPHA,
    )
    save_lora_weights(source, str(tmp_path / "lora_weights.pt"))
    monkeypatch.setattr(
        model_module, "load_base_model", lambda model_id: (build_tiny_causal_lm(), None)
    )

    model, _ = load_finetuned_model(tmp_path)

    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) == 4
    assert all(m.rank == model_module.LORA_RANK for m in lora_layers)
