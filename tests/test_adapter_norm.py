import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.lora import apply_lora, save_lora_config, save_lora_weights
from tests.test_lora import ToyModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "adapter_norm.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("adapter_norm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter_norm = _load_script()


def _write_checkpoint(path: Path, scale: float, rank: int = 2, alpha: float = 4.0):
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = apply_lora(ToyModel(), ["q_proj", "v_proj"], rank=rank, alpha=alpha)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=scale)
    save_lora_weights(model, str(path / "lora_weights.pt"))
    save_lora_config(
        path / "lora_config.json",
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        base_model_id="tiny/base",
    )
    return model


def test_zero_b_means_zero_update(tmp_path) -> None:
    path = tmp_path / "fresh"
    _write_checkpoint(path, scale=0.0)

    result = adapter_norm.adapter_norm(path)

    assert result["total_norm"] == pytest.approx(0.0)
    assert result["layers"] == 4  # 2 layers x (q_proj, v_proj)


def test_norm_grows_with_the_adapter(tmp_path) -> None:
    small = tmp_path / "small"
    large = tmp_path / "large"
    _write_checkpoint(small, scale=0.05)
    _write_checkpoint(large, scale=0.5)

    small_result = adapter_norm.adapter_norm(small)
    large_result = adapter_norm.adapter_norm(large)

    assert large_result["total_norm"] > 5 * small_result["total_norm"]
    assert set(large_result["per_module"]) == {"q_proj", "v_proj"}


def test_scaling_uses_alpha_over_rank(tmp_path) -> None:
    a = tmp_path / "alpha4"
    b = tmp_path / "alpha8"
    _write_checkpoint(a, scale=0.1, rank=2, alpha=4.0)
    _write_checkpoint(b, scale=0.1, rank=2, alpha=8.0)

    assert adapter_norm.adapter_norm(b)["total_norm"] == pytest.approx(
        2 * adapter_norm.adapter_norm(a)["total_norm"], rel=1e-5
    )


def test_layer_updates_matches_a_manual_product() -> None:
    state = {
        "layers.0.q_proj.lora_A": torch.eye(2),
        "layers.0.q_proj.lora_B": torch.eye(2) * 3.0,
    }

    norms = adapter_norm.layer_updates(state, scaling=2.0)

    # ||2 * (3I @ I)||_F = 2 * 3 * sqrt(2)
    assert norms["layers.0.q_proj"] == pytest.approx(6 * 2**0.5)


def test_main_writes_table_and_json(tmp_path, capsys) -> None:
    run = tmp_path / "run-a"
    _write_checkpoint(run, scale=0.2)
    out = tmp_path / "norms.md"
    out_json = tmp_path / "norms.json"

    assert adapter_norm.main([str(run), "-o", str(out), "--json", str(out_json)]) == 0

    assert "| run-a |" in out.read_text()
    assert json.loads(out_json.read_text())[0]["run"] == "run-a"


def test_missing_config_raises(tmp_path) -> None:
    run = tmp_path / "broken"
    _write_checkpoint(run, scale=0.1)
    (run / "lora_config.json").unlink()

    with pytest.raises(ValueError, match="rank/alpha"):
        adapter_norm.adapter_norm(run)
