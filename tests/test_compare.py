import json

import pytest
import torch

import src.compare as compare_module
from src.compare import (
    generation_settings,
    load_prompts,
    parse_args,
    render_markdown,
    run_compare,
)
from src.dataset import format_prompt_prefix
from src.lora import apply_lora, load_lora_weights, save_lora_config, save_lora_weights
from tests.conftest import FakeTokenizer, build_tiny_causal_lm
from tests.test_train import assert_no_local_details

MAX_NEW_TOKENS = 6
PROMPTS = {
    "description": "test prompts",
    "prompts": [
        {"id": "p1", "instruction": "説明してください"},
        {"id": "p2", "instruction": "要約してください"},
        {"id": "p3", "instruction": "箇条書きで挙げてください"},
    ],
}


def _qwen_like_model():
    """Tiny model whose generation_config asks for sampling, like Qwen2.5."""
    model = build_tiny_causal_lm()
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.05
    return model


@torch.no_grad()
def _manual_greedy(model, tokenizer, instruction: str) -> str:
    ids = tokenizer(format_prompt_prefix(instruction), add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = torch.tensor([ids])
    new_ids = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids=input_ids).logits[0, -1]
        next_id = int(torch.argmax(logits))
        new_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@pytest.fixture
def checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoints" / "tiny-run"
    ckpt.mkdir(parents=True)
    trained = apply_lora(
        build_tiny_causal_lm(seed=1), ["q_proj", "v_proj"], rank=4, alpha=8.0
    )
    torch.manual_seed(123)
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "lora_" in name:
                param.normal_(std=1.0)
    save_lora_weights(trained, str(ckpt / "lora_weights.pt"))
    save_lora_config(
        ckpt / "lora_config.json",
        rank=4,
        alpha=8.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        base_model_id="tiny/base",
    )
    return ckpt


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(PROMPTS, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def base_loads(monkeypatch):
    requested: list[str] = []

    def fake_load_base_model(model_id):
        requested.append(model_id)
        return _qwen_like_model(), FakeTokenizer()

    monkeypatch.setattr(compare_module, "load_base_model", fake_load_base_model)
    return requested


def _args(checkpoint, prompts_file, output_dir):
    return parse_args(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--prompts",
            str(prompts_file),
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(MAX_NEW_TOKENS),
        ]
    )


def test_compare_writes_base_and_finetuned_samples(
    tmp_path, checkpoint, prompts_file, base_loads
) -> None:
    out = tmp_path / "out"

    run_compare(_args(checkpoint, prompts_file, out))

    # The base model is loaded exactly once, using the id from lora_config.json
    assert base_loads == ["tiny/base"]
    text = (out / "samples.json").read_text(encoding="utf-8")
    result = json.loads(text)
    assert result["model_id"] == "tiny/base"
    assert result["checkpoint"] == "tiny-run"
    assert result["lora_config"] == {
        "rank": 4,
        "alpha": 8.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "base_model_id": "tiny/base",
    }
    assert result["lora_config_source"] == "lora_config.json"
    assert result["generation"]["do_sample"] is False
    assert result["generation"]["num_beams"] == 1
    assert result["generation"]["max_new_tokens"] == MAX_NEW_TOKENS
    assert [s["id"] for s in result["samples"]] == ["p1", "p2", "p3"]
    assert all(s["base_output"] and s["finetuned_output"] for s in result["samples"])
    assert any(s["base_output"] != s["finetuned_output"] for s in result["samples"])

    markdown = (out / "samples.md").read_text(encoding="utf-8")
    for sample in result["samples"]:
        assert sample["instruction"] in markdown
        assert sample["base_output"] in markdown
        assert sample["finetuned_output"] in markdown
    assert "Base model" in markdown and "Fine-tuned" in markdown

    assert str(tmp_path) not in text
    assert str(tmp_path) not in markdown
    assert_no_local_details(text, {k: v for k, v in result.items() if k != "samples"})


def test_compare_is_greedy_and_matches_separate_loads(
    tmp_path, checkpoint, prompts_file, base_loads
) -> None:
    first = run_compare(_args(checkpoint, prompts_file, tmp_path / "a"))
    second = run_compare(_args(checkpoint, prompts_file, tmp_path / "b"))

    assert first["samples"] == second["samples"]
    assert (tmp_path / "a" / "samples.md").read_bytes() == (
        tmp_path / "b" / "samples.md"
    ).read_bytes()

    # Base outputs equal plain greedy decoding (the model's sampling and
    # repetition_penalty defaults are overridden) ...
    tokenizer = FakeTokenizer()
    base_model = _qwen_like_model().eval()
    # ... and fine-tuned outputs equal a model loaded fresh with the adapter.
    tuned_model = apply_lora(_qwen_like_model(), ["q_proj", "v_proj"], 4, 8.0, 0.05)
    load_lora_weights(tuned_model, str(checkpoint / "lora_weights.pt"), strict=True)
    tuned_model.eval()
    for sample in first["samples"]:
        assert sample["base_output"] == _manual_greedy(
            base_model, tokenizer, sample["instruction"]
        )
        assert sample["finetuned_output"] == _manual_greedy(
            tuned_model, tokenizer, sample["instruction"]
        )
        assert sample["base_new_tokens"] == MAX_NEW_TOKENS


def test_compare_rejects_checkpoint_without_weights(tmp_path, prompts_file) -> None:
    with pytest.raises(FileNotFoundError):
        run_compare(_args(tmp_path, prompts_file, tmp_path / "out"))


def test_compare_mismatched_adapter_raises(
    tmp_path, checkpoint, prompts_file, base_loads
) -> None:
    save_lora_config(
        checkpoint / "lora_config.json",
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj"],
        base_model_id="tiny/base",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        run_compare(_args(checkpoint, prompts_file, tmp_path / "out"))


def test_default_prompt_set_is_valid() -> None:
    prompts = load_prompts(compare_module.REPO_ROOT / compare_module.DEFAULT_PROMPTS)

    assert 5 <= len(prompts) <= 8
    assert len({p["id"] for p in prompts}) == len(prompts)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"prompts": []},
        {"prompts": [{"id": "a"}]},
        {"prompts": [{"id": "a", "instruction": "x"}, {"id": "a", "instruction": "y"}]},
    ],
)
def test_load_prompts_rejects_invalid_files(tmp_path, data) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        load_prompts(path)


def test_generation_settings_are_greedy() -> None:
    settings = generation_settings(32)

    assert settings["do_sample"] is False
    assert settings["num_beams"] == 1
    assert settings["repetition_penalty"] == 1.0
    with pytest.raises(ValueError):
        generation_settings(0)


def test_render_markdown_uses_longer_fence_for_backticks() -> None:
    result = {
        "model_id": "m",
        "checkpoint": "c",
        "lora_config": {"rank": 1, "alpha": 2, "dropout": 0, "target_modules": ["q"]},
        "lora_config_source": "defaults",
        "generation": {"max_new_tokens": 4, "repetition_penalty": 1.0},
        "environment": {"torch": "t", "transformers": "x", "gpus": None},
        "prompts_file": "p.json",
        "prompts_sha256": "0" * 64,
        "system_prompt": "s",
        "samples": [
            {
                "id": "a",
                "instruction": "q",
                "base_output": "```code```",
                "base_new_tokens": 4,
                "finetuned_output": "ok",
                "finetuned_new_tokens": 1,
            }
        ],
    }

    markdown = render_markdown(result)

    assert "````text\n```code```\n````" in markdown
    assert "(4 new tokens — cut off at max_new_tokens)" in markdown
    assert "(1 new tokens)" in markdown
