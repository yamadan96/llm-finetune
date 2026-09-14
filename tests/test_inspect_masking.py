import importlib.util
from pathlib import Path

from tests.conftest import PAD_ID, ReversibleTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "inspect_masking.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("inspect_masking", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect_masking = _load_script()
MAX_LENGTH = 120
LONG_CONTEXT = "長い参考文章です。" * 40
ROWS = [
    {"index": "0", "instruction": "質問です", "input": "", "output": "回答です。"},
    {"index": "1", "instruction": "要約して", "input": "短い文脈", "output": "要約。"},
    {
        "index": "2",
        "instruction": "要約して",
        "input": LONG_CONTEXT,
        "output": "要約。",
    },
    {"index": "3", "instruction": "説明して", "input": "", "output": "長い回答" * 60},
    {"index": "4", "instruction": "別の質問", "input": "", "output": "はい。"},
    {"index": "5", "instruction": "抽出して", "input": "文脈二", "output": "抽出。"},
]


def _inspect(num_samples: int = 2):
    return inspect_masking.inspect_cases(
        ReversibleTokenizer(), ROWS, MAX_LENGTH, num_samples
    )


def test_inspect_cases_covers_every_case_and_passes() -> None:
    result = _inspect()

    assert result["ok"] is True, result
    assert result["missing_cases"] == []
    cases = [s["case"] for s in result["samples"]]
    assert sorted(set(cases)) == sorted([*inspect_masking.CASES, "forced_truncation"])
    by_row = {
        s["row_index"]: s for s in result["samples"] if s["case"] != "forced_truncation"
    }
    assert by_row["1"]["checks"]["context_in_user_message"] is True
    assert by_row["0"]["checks"]["no_context_header"] is True
    assert by_row["2"]["context_truncated"] is True
    assert by_row["2"]["checks"]["response_reserve_kept"] is True
    assert by_row["3"]["response_truncated"] is True
    # Without a chat template the template comparison is skipped, not failed
    assert by_row["0"]["checks"]["prompt_matches_chat_template"] is None


def test_inspect_cases_reports_missing_cases() -> None:
    result = inspect_masking.inspect_cases(
        ReversibleTokenizer(), ROWS[:1], MAX_LENGTH, 1
    )

    assert result["ok"] is False
    assert "with_context" in result["missing_cases"]


def test_inspect_example_fails_when_boundary_is_shifted(monkeypatch) -> None:
    real_build = inspect_masking.build_example_with_info

    def shifted(tokenizer, instruction, response, max_length, context=""):
        example, info = real_build(
            tokenizer, instruction, response, max_length, context=context
        )
        labels = example["labels"].clone()
        first = int((labels != -100).nonzero()[0])
        labels[first - 1] = example["input_ids"][first - 1]
        return {**example, "labels": labels}, info

    monkeypatch.setattr(inspect_masking, "build_example_with_info", shifted)

    report = inspect_masking.inspect_example(
        ReversibleTokenizer(), "要約して", "要約。", MAX_LENGTH, "短い文脈"
    )

    assert report["ok"] is False
    assert report["checks"]["supervised_starts_after_prefix"] is False


def test_inspect_example_detects_supervised_padding(monkeypatch) -> None:
    real_build = inspect_masking.build_example_with_info

    def leaky(tokenizer, instruction, response, max_length, context=""):
        example, info = real_build(
            tokenizer, instruction, response, max_length, context=context
        )
        labels = example["labels"].clone()
        labels[-1] = PAD_ID
        return {**example, "labels": labels}, info

    monkeypatch.setattr(inspect_masking, "build_example_with_info", leaky)

    report = inspect_masking.inspect_example(ReversibleTokenizer(), "q", "a", 64)

    assert report["ok"] is False
    assert report["checks"]["padding_masked_and_ignored"] is False


def test_template_mismatch_fails(monkeypatch) -> None:
    tokenizer = ReversibleTokenizer()
    tokenizer.chat_template = "custom"
    tokenizer.apply_chat_template = lambda *a, **k: "different prompt"

    report = inspect_masking.inspect_example(tokenizer, "q", "a", 64)

    assert report["ok"] is False
    assert report["checks"]["prompt_matches_chat_template"] is False


def test_shuffled_train_indices_use_training_split() -> None:
    order = inspect_masking.shuffled_train_indices(1000, 0.02, 42)
    train_idx, val_idx = inspect_masking.split_indices(1000, 0.02, 42)

    assert sorted(order) == train_idx
    assert order != train_idx
    assert set(order).isdisjoint(val_idx)
