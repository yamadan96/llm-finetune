import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "inspect_masking.py"
SPECIALS = {"<|im_start|>": 1, "<|im_end|>": 2}
PAD_ID = 0
OFFSET = 10
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in SPECIALS))


def _load_script():
    spec = importlib.util.spec_from_file_location("inspect_masking", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect_masking = _load_script()


class ReversibleTokenizer:
    """Character tokenizer whose decode is the exact inverse of encode."""

    pad_token_id = PAD_ID
    eos_token_id = SPECIALS["<|im_end|>"]

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids: list[int] = []
        pos = 0
        for match in _SPECIAL_RE.finditer(text):
            ids += [OFFSET + ord(c) for c in text[pos : match.start()]]
            ids.append(SPECIALS[match.group()])
            pos = match.end()
        ids += [OFFSET + ord(c) for c in text[pos:]]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        names = {v: k for k, v in SPECIALS.items()}
        return "".join(
            names[i] if i in names else "" if i == PAD_ID else chr(i - OFFSET)
            for i in ids
        )


ROWS = [
    {"index": "0", "instruction": "質問です", "input": "", "output": "回答です。"},
    {"index": "1", "instruction": "要約して", "input": "文脈", "output": "要約。"},
]


def test_inspect_rows_passes_for_correct_masking() -> None:
    result = inspect_masking.inspect_rows(ReversibleTokenizer(), ROWS, max_length=96)

    assert result["ok"] is True
    normal, with_input, truncated = result["samples"]
    assert all(normal["checks"].values())
    assert normal["supervised_range"] == [
        normal["prefix_tokens"],
        normal["prefix_tokens"] + normal["response_tokens"],
    ]
    assert normal["ignore_ranges"][0] == [0, normal["prefix_tokens"]]
    assert normal["decoded_supervised"] == "回答です。<|im_end|>"
    assert truncated["forced_truncation"] is True
    assert truncated["supervised_tokens"] == inspect_masking.TRUNCATED_RESPONSE_TOKENS
    assert with_input["dataset_input_ignored"] is True
    assert len(result["warnings"]) == 1


def test_inspect_rows_fails_when_boundary_is_shifted(monkeypatch) -> None:
    real_build = inspect_masking.build_example

    def shifted_build(tokenizer, instruction, response, max_length):
        example = real_build(tokenizer, instruction, response, max_length)
        labels = example["labels"].clone()
        first = int((labels != -100).nonzero()[0])
        labels[first - 1] = example["input_ids"][
            first - 1
        ]  # supervise one prompt token
        return {**example, "labels": labels}

    monkeypatch.setattr(inspect_masking, "build_example", shifted_build)

    result = inspect_masking.inspect_rows(ReversibleTokenizer(), ROWS[:1], 96)

    assert result["ok"] is False
    assert result["samples"][0]["checks"]["supervised_starts_after_prefix"] is False


def test_inspect_example_detects_supervised_padding(monkeypatch) -> None:
    real_build = inspect_masking.build_example

    def leaky_build(tokenizer, instruction, response, max_length):
        example = real_build(tokenizer, instruction, response, max_length)
        labels = example["labels"].clone()
        labels[-1] = PAD_ID
        return {**example, "labels": labels}

    monkeypatch.setattr(inspect_masking, "build_example", leaky_build)

    report = inspect_masking.inspect_example(ReversibleTokenizer(), "q", "a", 64)

    assert report["ok"] is False
    assert report["checks"]["padding_masked_and_ignored"] is False


def test_select_rows_matches_training_split_rule() -> None:
    rows = inspect_masking.select_rows(1000, 5, val_ratio=0.02, seed=42)
    train_idx, val_idx = inspect_masking.split_indices(1000, 0.02, 42)

    assert len(rows) == 5
    assert set(rows) <= set(train_idx)
    assert set(rows).isdisjoint(val_idx)
