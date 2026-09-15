"""Cheap, deterministic degeneration checks for generated text.

They flag outputs a person should read (repeated lines, looping phrases);
they are not a quality metric and never decide pass/fail on their own.
"""

import re
import unicodedata
from collections import Counter

# A normalized line of at least this many characters seen twice is flagged
MIN_DUPLICATE_LINE_CHARS = 4
# ... as is an output where this share of its non-empty lines are repeats
MAX_DUPLICATE_LINE_RATIO = 0.5
MIN_LINES_FOR_RATIO = 4
# The same clause (split at 、。！？ and newlines) this long, this often
MIN_REPEATED_CLAUSE_CHARS = 6
MIN_CLAUSE_OCCURRENCES = 3
# Looping phrases inside long text: few distinct character 8-grams
NGRAM = 8
MIN_CHARS_FOR_NGRAM = 64
MIN_DISTINCT_NGRAM_RATIO = 0.3

_CLAUSE_SPLIT_RE = re.compile(r"[、。，．,!?！？\n]+")
_LIST_MARKER_RE = re.compile(r"^(?:\d+\s*[.)．、:：]|[-*+•・●◦]|#+)\s*")


def normalize_line(line: str) -> str:
    """NFKC, list markers/headings stripped, whitespace and trailing punctuation removed."""
    text = unicodedata.normalize("NFKC", line).strip()
    text = _LIST_MARKER_RE.sub("", text)
    text = "".join(text.split())
    return text.rstrip("。.、,").strip("*_`")


def duplicate_line_stats(text: str) -> dict[str, float | int]:
    """Counts of repeated normalized lines in ``text``."""
    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    counts = Counter(lines)
    repeats = sum(count - 1 for count in counts.values())
    long_repeats = [
        line
        for line, count in counts.items()
        if count >= 2 and len(line) >= MIN_DUPLICATE_LINE_CHARS
    ]
    return {
        "lines": len(lines),
        "repeated_lines": repeats,
        "repeated_line_ratio": repeats / len(lines) if lines else 0.0,
        "max_line_occurrences": max(counts.values(), default=0),
        "distinct_repeated_long_lines": len(long_repeats),
    }


def repeated_clauses(text: str) -> dict[str, int]:
    """Clauses of at least MIN_REPEATED_CLAUSE_CHARS seen MIN_CLAUSE_OCCURRENCES+ times."""
    clauses = [normalize_line(c) for c in _CLAUSE_SPLIT_RE.split(text)]
    counts = Counter(c for c in clauses if len(c) >= MIN_REPEATED_CLAUSE_CHARS)
    return {c: n for c, n in counts.items() if n >= MIN_CLAUSE_OCCURRENCES}


def _low_ngram_diversity(text: str) -> bool:
    if len(text) < MIN_CHARS_FOR_NGRAM:
        return False
    grams = [text[i : i + NGRAM] for i in range(len(text) - NGRAM + 1)]
    return len(set(grams)) / len(grams) < MIN_DISTINCT_NGRAM_RATIO


def repetition_findings(text: str) -> list[str]:
    """Reasons ``text`` looks degenerate (empty list if none)."""
    findings = []
    stats = duplicate_line_stats(text)
    if stats["distinct_repeated_long_lines"]:
        findings.append(
            f"duplicate lines (a line repeated up to {stats['max_line_occurrences']}x)"
        )
    elif (
        stats["lines"] >= MIN_LINES_FOR_RATIO
        and stats["repeated_line_ratio"] >= MAX_DUPLICATE_LINE_RATIO
    ):
        findings.append(f"duplicate line ratio {stats['repeated_line_ratio']:.0%}")
    clauses = repeated_clauses(text)
    if clauses and not stats["distinct_repeated_long_lines"]:
        findings.append(
            f"repeated clauses ({len(clauses)} clause(s) up to {max(clauses.values())}x)"
        )
    if _low_ngram_diversity(text):
        findings.append("looping phrases (low 8-gram diversity)")
    return findings
