import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "list_metrics.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("list_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


list_metrics = _load_script()

SAMPLES = {
    "checkpoint": "run-a",
    "prompts_file": "prompts/list_eval_ja.json",
    "prompts_sha256": "a" * 64,
    "generation": {"max_new_tokens": 64},
    "samples": [
        {
            "id": "count-3",
            "expected_items": 3,
            "base_output": "1. A\n2. B\n3. C",
            "base_new_tokens": 12,
            "finetuned_output": "1. A\n2. A\n3. A",
            "finetuned_new_tokens": 12,
        },
        {
            "id": "count-5",
            "expected_items": 5,
            "base_output": "りんご、みかん、ぶどう、もも、なし",
            "base_new_tokens": 15,
            "finetuned_output": "くだものがいくつかあります。",
            "finetuned_new_tokens": 9,
        },
        {
            "id": "open",
            "expected_items": None,
            "base_output": "- A\n- B",
            "base_new_tokens": 8,
            "finetuned_output": "- A\n- B\n- C",
            "finetuned_new_tokens": 64,
        },
    ],
}


def test_run_metrics_base_and_finetuned() -> None:
    base = list_metrics.run_metrics(SAMPLES, "base_output")
    tuned = list_metrics.run_metrics(SAMPLES, "finetuned_output")

    assert base["structured"] == 3 and base["structured_rate"] == 1.0
    assert base["exact_count"] == 2 and base["with_expected_count"] == 2
    assert base["answers_with_duplicates"] == 0
    assert base["terminated"] == 3

    # Fine-tuned: one answer repeats an item, one collapses into prose,
    # one hits max_new_tokens
    assert tuned["structured"] == 2
    assert tuned["exact_count"] == 1
    assert tuned["answers_with_duplicates"] == 1
    assert tuned["duplicate_items_total"] == 2
    assert tuned["terminated"] == 2
    assert tuned["shapes"]["prose"] == 1


def test_unique_item_metrics() -> None:
    tuned = list_metrics.run_metrics(SAMPLES, "finetuned_output")
    base = list_metrics.run_metrics(SAMPLES, "base_output")

    # Fine-tuned: "1. A / 2. A / 3. A" is 3 attempted, 1 unique
    assert tuned["attempted_items"] == 3 + 1 + 3
    assert tuned["unique_items"] == 1 + 1 + 3
    assert tuned["unique_item_rate"] == pytest.approx(5 / 7)
    assert base["unique_item_rate"] == 1.0
    # No answer attempts >= 6 items here
    assert tuned["unique_when_many"] is None


def test_answer_metrics_counts_exact_items() -> None:
    assert list_metrics.answer_metrics(3, "1. A\n2. B\n3. C")["exact_count"] is True
    assert list_metrics.answer_metrics(3, "1. A\n2. B")["exact_count"] is False
    assert list_metrics.answer_metrics(None, "A、B")["exact_count"] is None


def test_main_writes_table_and_rejects_different_prompt_sets(tmp_path, capsys) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(SAMPLES, ensure_ascii=False))
    other = {**SAMPLES, "checkpoint": "run-b", "prompts_sha256": "b" * 64}
    b.write_text(json.dumps(other, ensure_ascii=False))
    out = tmp_path / "list_metrics.md"

    assert list_metrics.main([str(a), str(b)]) == 1
    assert "prompts_sha256 differs" in capsys.readouterr().out

    b.write_text(json.dumps({**SAMPLES, "checkpoint": "run-b"}, ensure_ascii=False))
    assert list_metrics.main([str(a), str(b), "-o", str(out)]) == 0
    text = out.read_text()
    assert "| run-a | finetuned |" in text
    assert "| run-b | base |" in text


def test_list_eval_set_is_valid() -> None:
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.compare import load_prompts
    from src.list_rules import is_list_instruction, requested_item_count

    prompts = load_prompts(REPO_ROOT / "prompts" / "list_eval_ja.json")

    assert len(prompts) == 16
    assert all(is_list_instruction(p["instruction"]) for p in prompts)
    with_count = [p for p in prompts if p["expected_items"]]
    assert len(with_count) >= 8
    # The stated count is what the shared rule reads out of the instruction
    for prompt in with_count:
        assert requested_item_count(prompt["instruction"]) == prompt["expected_items"]


def test_committed_contamination_report_matches_list_eval_set() -> None:
    import hashlib

    prompts = REPO_ROOT / "prompts" / "list_eval_ja.json"
    report = json.loads(
        (REPO_ROOT / "prompts" / "list_eval_ja.contamination.json").read_text()
    )

    assert report["ok"] is True
    assert report["prompts_sha256"] == hashlib.sha256(prompts.read_bytes()).hexdigest()
