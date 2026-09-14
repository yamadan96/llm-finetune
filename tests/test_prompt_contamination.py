import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_prompt_contamination.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_prompt_contamination", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contamination = _load_script()

ROWS = [
    {
        "index": "0",
        "instruction": "日本の首都はどこですか？",
        "input": "",
        "output": "東京です。",
    },
    {
        "index": "1",
        "instruction": "次の文章を要約してください。",
        "input": "来週の定例会議は会議室の改装工事のため木曜日に変更します。",
        "output": "会議は木曜日に変更。",
    },
]


def _check(instruction: str) -> dict:
    return contamination.check_prompt(
        {"id": "p", "instruction": instruction},
        ROWS,
        jaccard_threshold=0.5,
        containment_threshold=0.8,
        min_substring_chars=20,
    )


def test_unrelated_prompt_passes() -> None:
    result = _check("TCP と UDP の違いを簡潔に説明してください。")

    assert result["ok"] is True
    assert result["exact_matches"] == []
    assert result["substring_matches"] == []


def test_exact_match_after_normalization_is_flagged() -> None:
    result = _check("日本の首都は どこですか?")  # half-width '?' and extra space

    assert result["ok"] is False
    assert result["exact_matches"] == [{"row_index": "0", "field": "instruction"}]


def test_quoted_passage_copied_from_dataset_input_is_flagged() -> None:
    result = _check(
        "1文で要約して。『来週の定例会議は会議室の改装工事のため木曜日に変更します。』"
    )

    assert result["ok"] is False
    assert {"row_index": "1", "field": "input"} in result["substring_matches"]


def test_near_duplicate_is_flagged_by_similarity() -> None:
    result = _check("次の文章を要約して下さい。")

    assert result["ok"] is False
    assert result["max_jaccard"]["row_index"] == "1"
    assert result["max_jaccard"]["score"] >= 0.5


@pytest.mark.parametrize(
    ("text", "expected"),
    [("abc", {"abc"}), ("ab", {"ab"}), ("", set()), ("abcd", {"abc", "bcd"})],
)
def test_ngrams(text: str, expected: set[str]) -> None:
    assert contamination.ngrams(text) == expected


def test_committed_report_matches_prompt_set() -> None:
    import hashlib

    prompts = REPO_ROOT / "prompts" / "compare_ja.json"
    report = json.loads(
        (REPO_ROOT / "prompts" / "compare_ja.contamination.json").read_text()
    )

    assert report["ok"] is True
    assert report["prompts_sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest()
    ids = [p["id"] for p in json.loads(prompts.read_text())["prompts"]]
    assert [r["id"] for r in report["results"]] == ids
