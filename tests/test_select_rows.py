import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "select_rows.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("select_rows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_rows = _load_script()

CANDIDATES = [
    {"row_id": 10, "order": 0, "category": "open_qa", "response_tokens": 20},
    {"row_id": 11, "order": 1, "category": "open_qa", "response_tokens": 300},
    {"row_id": 12, "order": 2, "category": "open_qa", "response_tokens": 100},
    {"row_id": 13, "order": 3, "category": "summarization", "response_tokens": 50},
    {"row_id": 14, "order": 4, "category": "summarization", "response_tokens": 400},
    {"row_id": 15, "order": 5, "category": "summarization", "response_tokens": 30},
]


def _ids(rows) -> list[int]:
    return [row["row_id"] for row in rows]


def test_seeded_selection_keeps_the_seeded_order() -> None:
    chosen = select_rows.select(CANDIDATES, 3, "seeded", None)

    assert _ids(chosen) == [10, 11, 12]


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("longest", [11, 14]), ("shortest", [10, 15])],
)
def test_length_selection_picks_the_extremes(selection: str, expected) -> None:
    assert _ids(select_rows.select(CANDIDATES, 2, selection, None)) == expected


def test_category_quotas_are_met_within_each_category() -> None:
    quotas = {"open_qa": 2, "summarization": 1}

    longest = select_rows.select(CANDIDATES, 3, "longest", quotas)
    shortest = select_rows.select(CANDIDATES, 3, "shortest", quotas)

    assert _ids(longest) == [11, 12, 14]
    assert _ids(shortest) == [10, 12, 15]
    for chosen in (longest, shortest):
        counts = {c: sum(r["category"] == c for r in chosen) for c in quotas}
        assert counts == quotas
    # Same category mix, very different lengths
    assert (
        select_rows.summarize(longest)["median_response_tokens"]
        > (select_rows.summarize(shortest)["median_response_tokens"])
    )


def test_quota_larger_than_the_pool_raises() -> None:
    with pytest.raises(ValueError, match="raise --pool-rows"):
        select_rows.select(CANDIDATES, 9, "longest", {"open_qa": 9})


def test_summarize_reports_per_category_medians() -> None:
    summary = select_rows.summarize(CANDIDATES)

    assert summary["examples"] == 6
    assert summary["categories"]["open_qa"]["rows"] == 3
    assert summary["categories"]["open_qa"]["median_response_tokens"] == 100


def test_category_quotas_from_reference_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"config": {"train_row_ids": [10, 11, 13]}}))
    categories = {10: "open_qa", 11: "open_qa", 13: "summarization"}

    assert select_rows.category_quotas(metrics, categories) == {
        "open_qa": 2,
        "summarization": 1,
    }
