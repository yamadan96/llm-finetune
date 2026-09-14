import pytest

from src.text_checks import (
    duplicate_line_stats,
    normalize_line,
    repeated_clauses,
    repetition_findings,
)

PILOT_DUPLICATE_LIST = "1. 朝のルーチンを確立する\n2. 朝のルーチンを確立する\n3. 朝のルーチンを確立する\n4. 朝のルーチンを確立する\n5. 朝のルーチンを確立する"
PILOT_LOOPING_CLAUSES = (
    "TCP は、データを送信する前に、データを分割し、各データを送信し、各データが到着したことを確認し、"
    "データを再構成するプロトコルです。たとえば、ブラウザはデータを分割し、各データを送信し、"
    "各データが到着したことを確認します。一方、UDP は、データを分割し、各データを送信し、"
    "各データが到着したことを確認しないプロトコルです。サーバーは、データを分割し、各データを送信する。"
)
HEALTHY_LIST = "1. 作業時間を決める\n2. 作業スペースを分ける\n3. 定期的に休憩する\n4. 通知を切る\n5. 1日の目標を書く"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1. 朝のルーチンを確立する", "朝のルーチンを確立する"),
        ("  - **朝の  ルーチン**。", "朝のルーチン"),
        ("・ＡＢＣ", "ABC"),
        ("## 見出し", "見出し"),
    ],
)
def test_normalize_line(line: str, expected: str) -> None:
    assert normalize_line(line) == expected


def test_duplicate_numbered_lines_are_flagged() -> None:
    findings = repetition_findings(PILOT_DUPLICATE_LIST)

    assert findings == ["duplicate lines (a line repeated up to 5x)"]
    stats = duplicate_line_stats(PILOT_DUPLICATE_LIST)
    assert stats["repeated_lines"] == 4
    assert stats["repeated_line_ratio"] == pytest.approx(0.8)


def test_a_single_repeated_line_is_flagged() -> None:
    text = "要点は次の通りです\n- 早めに寝る\n- 水を飲む\n- 早めに寝る"

    assert repetition_findings(text) == ["duplicate lines (a line repeated up to 2x)"]


def test_short_repeated_lines_use_the_ratio_rule() -> None:
    text = "はい\nはい\nはい\nいいえ"

    assert repetition_findings(text) == ["duplicate line ratio 50%"]


def test_looping_clauses_are_flagged() -> None:
    assert repeated_clauses(PILOT_LOOPING_CLAUSES)["各データを送信し"] == 3
    assert repetition_findings(PILOT_LOOPING_CLAUSES)[0].startswith("repeated clauses")


@pytest.mark.parametrize(
    "text",
    [
        HEALTHY_LIST,
        "来週の定例会議は木曜日に変更され、開始時刻は午前10時のままです。",
        "",
        "4 × 3 = 12個です。12個に5個を足すと、12 + 5 = 17個になります。",
    ],
)
def test_healthy_outputs_are_not_flagged(text: str) -> None:
    assert repetition_findings(text) == []


def test_low_ngram_diversity_is_flagged() -> None:
    text = "ありがとうございます" * 12

    assert "looping phrases (low 8-gram diversity)" in repetition_findings(text)
