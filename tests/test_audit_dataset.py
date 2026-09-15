import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "audit_dataset.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script()


def test_quantiles_and_empty_input() -> None:
    values = list(range(1, 101))

    q = audit.quantiles(values)

    assert (q["p5"], q["p50"], q["p95"], q["max"]) == (5, 50, 95, 100)
    assert q["mean"] == pytest.approx(50.5)
    assert audit.quantiles([])["p50"] == 0


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("工夫を5つ挙げてください", 5),
        ("３点にまとめてください", 3),
        ("три", None),
        ("理由を三つ述べてください", 3),
        ("箇条書きで挙げてください", None),
    ],
)
def test_requested_item_count(instruction: str, expected) -> None:
    assert audit.requested_item_count(instruction) == expected


@pytest.mark.parametrize(
    ("response", "shape", "items"),
    [
        ("1. りんご\n2. みかん\n3. ぶどう", "marked_list", 3),
        ("りんご、みかん、ぶどう", "inline_list", 3),
        ("最初の段落です\n次の段落です", "multi_line", 2),
        ("ひとつの文だけです。", "single_sentence", 1),
    ],
)
def test_list_response_shape(response: str, shape: str, items: int) -> None:
    result = audit.list_response_shape(response)

    assert (result["shape"], result["items"]) == (shape, items)


def test_list_response_shape_counts_repeated_lines() -> None:
    assert audit.list_response_shape("- 同じ項目\n- 同じ項目")["repeated_lines"] == 1


@pytest.mark.parametrize(
    ("context", "response", "expected"),
    [
        ("元の文です。", "元の文です。", "identical"),
        (
            "この商品はとても便利で毎日使えます",
            "この商品はとても便利で毎日使えます。",
            "near_identical",
        ),
        ("資料まだですか？", "資料をお送りいただけますと幸いです。", "rewritten"),
        ("", "何か", "no_input"),
    ],
)
def test_rewrite_overlap(context: str, response: str, expected: str) -> None:
    assert audit.rewrite_overlap(context, response) == expected


def test_audit_row_and_summary(reversible_tokenizer) -> None:
    rows = [
        {
            "index": "1",
            "category": "brainstorming",
            "instruction": "工夫を3つ挙げてください",
            "input": "",
            "output": "1. あさ\n2. ひる\n3. よる",
        },
        {
            "index": "2",
            "category": "closed_qa",
            "instruction": "答えて",
            "input": "文脈。" * 60,
            "output": "はい",
        },
        {
            "index": "3",
            "category": "summarization",
            "instruction": "丁寧に書き換えてください",
            "input": "送って",
            "output": "お送りいただけますと幸いです",
        },
        {
            "index": "4",
            "category": "open_qa",
            "instruction": "",
            "input": "",
            "output": "x",
        },
    ]

    audited = [audit.audit_row(reversible_tokenizer, row, 160, 10) for row in rows]
    summary = audit.summarize(audited, short_response_tokens=10)

    assert summary["rows"] == 4
    assert summary["skipped"]["empty"] == 1
    assert summary["overall"]["examples"] == 3
    assert summary["overall"]["with_input"] == 2
    assert summary["overall"]["context_truncated"] == 1
    assert summary["overall"]["short_responses"] == 1  # "はい"
    assert set(summary["by_category"]) == {
        "brainstorming",
        "closed_qa",
        "summarization",
    }
    listed = summary["list_instructions"]
    assert listed["examples"] == 1
    assert listed["shapes"] == {"marked_list": 1}
    assert listed["requested_count"]["instructions_with_a_count"] == 1
    assert listed["requested_count"]["exact"] == 1
    rewrite = summary["rewrite_instructions"]
    assert rewrite["examples"] == 1
    assert rewrite["overlap"] == {"rewritten": 1}


def test_committed_audit_matches_the_script_schema() -> None:
    import json

    report = json.loads((REPO_ROOT / "docs" / "dataset_audit.json").read_text())

    assert report["schema_version"] == audit.SCHEMA_VERSION
    assert report["summary"]["overall"]["examples"] > 0
    assert set(report["summary"]["list_instructions"]["shapes"]) <= {
        "marked_list",
        "inline_list",
        "multi_line",
        "single_sentence",
    }
