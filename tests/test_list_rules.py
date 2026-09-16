import pytest

from src.list_rules import (
    duplicate_items,
    is_list_instruction,
    list_answer_quality,
    normalize_item,
    requested_item_count,
    response_items,
)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("工夫を箇条書きで5つ挙げてください", True),
        ("好きな映画をリストアップしてください", True),
        ("理由を列挙してください", True),
        ("次の文章を要約してください", False),
        ("野球とホッケーの2つのスポーツをどう分類しますか？", False),
    ],
)
def test_is_list_instruction(instruction: str, expected: bool) -> None:
    assert is_list_instruction(instruction) is expected


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("工夫を5つ挙げてください", 5),
        ("三つ挙げてください", 3),
        ("箇条書きで挙げてください", None),
        # A number that does not qualify a list request is ignored
        ("2つのスポーツをどう分類しますか？", None),
    ],
)
def test_requested_item_count(instruction: str, expected) -> None:
    assert requested_item_count(instruction) == expected


@pytest.mark.parametrize(
    ("response", "shape", "items"),
    [
        ("1. りんご\n2. みかん", "marked_list", 2),
        ("りんご\nみかん\nぶどう", "multi_line", 3),
        ("ペパロニ、ソーセージ、オニオン", "inline_list", 3),
        ("犬 猫 ウサギ オウム", "inline_list", 4),
        # Prose with commas is prose, not a nine-item list
        (
            "この映画は友情を描いた作品で、スケールが大きく、何度でも見たくなります。",
            "prose",
            1,
        ),
        ("ガドウォール", "prose", 1),
    ],
)
def test_response_items(response: str, shape: str, items: int) -> None:
    result_shape, result_items = response_items(response)

    assert (result_shape, len(result_items)) == (shape, items)


def test_normalize_item_and_duplicates() -> None:
    assert (
        normalize_item(" 1. **朝のルーチンを確立する**。") == "朝のルーチンを確立する"
    )
    assert duplicate_items(["a", "a", "b", "a"]) == 2
    assert duplicate_items([]) == 0


def test_list_answer_quality_flags() -> None:
    good = list_answer_quality("3つ挙げてください", "1. A\n2. B\n3. C")
    assert good["structured"] and not good["count_mismatch"]
    assert good["requested_items"] == 3 and good["items"] == 3

    mismatch = list_answer_quality("3つ挙げてください", "1. A\n2. B")
    assert mismatch["count_mismatch"] is True

    duplicated = list_answer_quality("3つ挙げてください", "1. A\n2. A\n3. A")
    assert duplicated["duplicate_items"] == 2 and duplicated["has_duplicates"]

    prose = list_answer_quality(
        "いくつか挙げてください", "花崗岩や玄武岩は火成岩です。"
    )
    assert prose["prose_answer"] is True and prose["count_mismatch"] is False
