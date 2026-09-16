"""Deterministic rules for list-style instructions and their answers.

Shared by the dataset audit, the training-side selection of list rows and the
evaluation metrics, so that "this is a list instruction" and "this answer is a
list of N items" mean the same thing everywhere.

The rules are intentionally conservative: an answer is only *counted* as a
structured list when its items are separated by markers, line breaks or
enumeration separators. Prose that happens to contain commas is reported as
prose, not as a list of many items, because counting items inside prose was
the main source of false positives when these rules were checked by hand
against the training data.
"""

import re
import unicodedata
from collections import Counter

# "5つ挙げてください", "箇条書きで", "リストアップ", ...
LIST_INSTRUCTION_RE = re.compile(
    r"箇条書き|リストアップ|リストにして|リストを|リストで|列挙|"
    r"挙げてください|挙げよ|挙げて下さい|並べてください"
)
# A requested item count, but only when it is attached to a "list" phrasing
REQUESTED_COUNT_RE = re.compile(
    r"([0-9一二三四五六七八九十]+)\s*(?:つ|点|個|項目|種類|件)"
)
ITEM_MARKER_RE = re.compile(r"^\s*(?:[-*+•・●◦]|\d+[.)．、]|[（(]\d+[)）])\s*")
# Separators that enumerate items on one line ("A、B、C" / "A, B, C" / "A B C")
INLINE_SEPARATOR_RE = re.compile(r"[、,，･・/／]|\s+")
SENTENCE_END_RE = re.compile(r"[。．！？]")
KANJI_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
# An inline enumeration may not contain sentence ends, its items stay short and
# none of them reads as a clause (checked by hand against the training data)
MAX_INLINE_ITEM_CHARS = 20
CLAUSE_ENDING_RE = re.compile(
    r"(?:です|ます|ました|でした|ません|である|します|なります)$"
)
MIN_ITEMS_FOR_LIST = 2


def normalize_item(text: str) -> str:
    """NFKC, markers, quotes, whitespace and trailing punctuation removed."""
    item = unicodedata.normalize("NFKC", text).strip()
    item = ITEM_MARKER_RE.sub("", item)
    item = "".join(item.split())
    quotes = "\"'「」『』【】()[]*_`"
    return item.rstrip("。.、,").strip(quotes).rstrip("。.、,")


def is_list_instruction(instruction: str) -> bool:
    """True for instructions that ask for an enumeration."""
    return bool(LIST_INSTRUCTION_RE.search(unicodedata.normalize("NFKC", instruction)))


def requested_item_count(instruction: str) -> int | None:
    """How many items the instruction asks for, or None.

    Only counts that qualify the requested items are returned: the number must
    be followed by a counter and the instruction must ask for a list.
    """
    text = unicodedata.normalize("NFKC", instruction)
    if not is_list_instruction(text):
        return None
    match = REQUESTED_COUNT_RE.search(text)
    if not match:
        return None
    digits = match.group(1)
    if digits.isdigit():
        return int(digits)
    if len(digits) == 1 and digits in KANJI_DIGITS:
        return KANJI_DIGITS[digits]
    return None


def _inline_items(line: str) -> list[str]:
    """Items of a one-line enumeration, or [] if the line reads as prose."""
    if SENTENCE_END_RE.search(line.strip().rstrip("。．！？")):
        return []
    parts = [normalize_item(part) for part in INLINE_SEPARATOR_RE.split(line)]
    items = [part for part in parts if part]
    if len(items) < MIN_ITEMS_FOR_LIST:
        return []
    if any(len(item) > MAX_INLINE_ITEM_CHARS for item in items):
        return []
    if any(CLAUSE_ENDING_RE.search(item) for item in items):
        return []
    return items


def response_items(response: str) -> tuple[str, list[str]]:
    """``(shape, items)`` of a response.

    - ``marked_list``: two or more lines starting with a bullet or a number
    - ``multi_line``: two or more non-empty lines without markers
    - ``inline_list``: one line enumerating short items
    - ``prose``: anything else (a single item or free text)
    """
    lines = [line for line in response.splitlines() if line.strip()]
    marked = [normalize_item(line) for line in lines if ITEM_MARKER_RE.match(line)]
    if len(marked) >= MIN_ITEMS_FOR_LIST:
        return "marked_list", marked
    if len(lines) >= MIN_ITEMS_FOR_LIST:
        return "multi_line", [normalize_item(line) for line in lines]
    inline = _inline_items(lines[0]) if lines else []
    if inline:
        return "inline_list", inline
    return "prose", [normalize_item(response)] if response.strip() else []


def duplicate_items(items: list[str]) -> int:
    """How many items repeat one that came before."""
    counts = Counter(item for item in items if item)
    return sum(count - 1 for count in counts.values())


def list_answer_quality(instruction: str, response: str) -> dict[str, object]:
    """Structure of a list answer and the ways it can be defective."""
    shape, items = response_items(response)
    requested = requested_item_count(instruction)
    duplicates = duplicate_items(items)
    structured = shape != "prose" and len(items) >= MIN_ITEMS_FOR_LIST
    return {
        "shape": shape,
        "items": len(items),
        "requested_items": requested,
        "duplicate_items": duplicates,
        "structured": structured,
        "count_mismatch": bool(requested) and structured and len(items) != requested,
        "prose_answer": not structured,
        "has_duplicates": duplicates > 0,
    }
