import datasets
import pytest

import src.dataset as dataset_module
from src.dataset import (
    CONTEXT_HEADER,
    CONTEXT_TRUNCATION_MARK,
    IGNORE_INDEX,
    InstructionDataset,
    build_example,
    build_example_with_info,
    chat_messages,
    format_chatml,
    format_prompt_prefix,
    format_response,
    format_user_message,
    limit_indices,
    load_instruction_datasets,
    response_token_reserve,
    split_indices,
)
from tests.conftest import PAD_ID, SPECIAL_TOKENS

IM_END = SPECIAL_TOKENS["<|im_end|>"]


def _lengths(tokenizer, instruction: str, response: str) -> tuple[int, int]:
    prefix = tokenizer(format_prompt_prefix(instruction), add_special_tokens=False)
    resp = tokenizer(format_response(response), add_special_tokens=False)
    return len(prefix["input_ids"]), len(resp["input_ids"])


def test_format_chatml_is_prefix_plus_response() -> None:
    text = format_chatml("質問", "回答")

    assert text.endswith("<|im_start|>assistant\n回答<|im_end|>")
    assert text == format_prompt_prefix("質問") + format_response("回答")


def test_build_example_masks_prompt_and_padding(fake_tokenizer) -> None:
    prefix_len, resp_len = _lengths(fake_tokenizer, "hi", "yes")
    max_length = prefix_len + resp_len + 5

    ex = build_example(fake_tokenizer, "hi", "yes", max_length)

    assert ex is not None
    labels = ex["labels"].tolist()
    ids = ex["input_ids"].tolist()
    mask = ex["attention_mask"].tolist()
    assert len(ids) == len(labels) == len(mask) == max_length
    # Prompt prefix (system/user/assistant header) is not supervised
    assert labels[:prefix_len] == [IGNORE_INDEX] * prefix_len
    # Response tokens, including the closing <|im_end|>, are supervised
    response_slice = slice(prefix_len, prefix_len + resp_len)
    assert labels[response_slice] == ids[response_slice]
    assert labels[prefix_len + resp_len - 1] == IM_END
    # Padding is masked and ignored
    assert labels[prefix_len + resp_len :] == [IGNORE_INDEX] * 5
    assert ids[prefix_len + resp_len :] == [PAD_ID] * 5
    assert mask == [1] * (prefix_len + resp_len) + [0] * 5


def test_build_example_ids_match_full_text_tokenization(fake_tokenizer) -> None:
    prefix_len, resp_len = _lengths(fake_tokenizer, "hi", "yes")

    ex = build_example(fake_tokenizer, "hi", "yes", prefix_len + resp_len)

    assert ex is not None
    assert ex["input_ids"].tolist() == fake_tokenizer.encode_text(
        format_chatml("hi", "yes")
    )


def test_build_example_partial_response_truncation_keeps_example(
    fake_tokenizer,
) -> None:
    prefix_len, _ = _lengths(fake_tokenizer, "hi", "a long answer")
    max_length = prefix_len + 3

    ex = build_example(fake_tokenizer, "hi", "a long answer", max_length)

    assert ex is not None
    labels = ex["labels"].tolist()
    assert len(labels) == max_length
    assert all(label != IGNORE_INDEX for label in labels[prefix_len:])


@pytest.mark.parametrize("cut", [0, 1])
def test_build_example_response_fully_truncated_returns_none(
    fake_tokenizer, cut: int
) -> None:
    prefix_len, _ = _lengths(fake_tokenizer, "hi", "yes")

    assert build_example(fake_tokenizer, "hi", "yes", prefix_len - cut) is None


def test_instruction_dataset_skips_empty_and_truncated_rows(fake_tokenizer) -> None:
    prefix_len, _ = _lengths(fake_tokenizer, "q", "a")
    rows = [
        {"instruction": "q", "output": "a"},
        {"instruction": "", "output": "a"},
        {"instruction": "q", "output": None},
        {"instruction": "q" * 50, "output": "a"},  # prompt alone exceeds max_length
    ]

    dataset = InstructionDataset(fake_tokenizer, rows, max_length=prefix_len + 4)

    assert len(dataset) == 1


def test_split_indices_is_seeded_disjoint_and_complete() -> None:
    train_a, val_a = split_indices(100, 0.1, seed=42)
    train_b, val_b = split_indices(100, 0.1, seed=42)
    _, val_c = split_indices(100, 0.1, seed=7)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 10
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(100))
    assert val_a != val_c


def test_split_indices_small_dataset_keeps_one_of_each() -> None:
    train, val = split_indices(5, 0.02, seed=0)

    assert len(val) == 1
    assert len(train) == 4


def test_split_indices_zero_ratio_has_no_validation() -> None:
    assert split_indices(10, 0.0, seed=0) == (list(range(10)), [])


def test_limit_indices_none_keeps_everything() -> None:
    assert limit_indices([5, 1, 3], None, seed=0) == [1, 3, 5]


def test_limit_indices_is_deterministic_bounded_and_nested() -> None:
    indices = list(range(0, 200, 2))

    first = limit_indices(indices, 10, seed=42)
    second = limit_indices(indices, 10, seed=42)
    larger = limit_indices(indices, 30, seed=42)
    other_seed = limit_indices(indices, 10, seed=7)

    assert first == second
    assert len(first) == 10
    assert set(first) <= set(indices)
    assert set(first) <= set(larger)
    assert first != other_seed


def test_limit_indices_larger_than_input_keeps_everything() -> None:
    assert limit_indices([3, 2, 1], 10, seed=0) == [1, 2, 3]


def test_limit_indices_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        limit_indices([1, 2, 3], 0, seed=0)


def _patch_raw_dataset(monkeypatch, num_rows: int) -> list[str]:
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(num_rows)]
    requested: list[str] = []

    def fake_load_dataset(dataset_id, split):
        requested.append(dataset_id)
        return datasets.Dataset.from_list(rows)

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    return requested


def _load(fake_tokenizer, **kwargs):
    return load_instruction_datasets(
        fake_tokenizer,
        dataset_id="org/data",
        max_length=96,
        val_ratio=0.2,
        seed=3,
        **kwargs,
    )


def test_load_instruction_datasets_respects_sample_limits(
    monkeypatch, fake_tokenizer
) -> None:
    requested = _patch_raw_dataset(monkeypatch, num_rows=50)

    full_train, full_val = _load(fake_tokenizer)
    train_a, val_a = _load(fake_tokenizer, max_train_samples=7, max_val_samples=3)
    train_b, val_b = _load(fake_tokenizer, max_train_samples=7, max_val_samples=3)

    assert requested == ["org/data"] * 3
    assert (len(full_train), len(full_val)) == (40, 10)
    assert (len(train_a), len(val_a)) == (7, 3)

    def ids(dataset) -> list[list[int]]:
        return [s["input_ids"].tolist() for s in dataset.samples]

    assert ids(train_a) == ids(train_b)
    assert ids(val_a) == ids(val_b)
    # Limits are applied after the split: subsets of the full splits
    assert all(x in ids(full_train) for x in ids(train_a))
    assert all(x in ids(full_val) for x in ids(val_a))


# --- Context (dataset "input" field) formatting and masking regressions -----

CTX_MAX_LENGTH = 160


def _supervised(example) -> tuple[int, int]:
    positions = (example["labels"] != IGNORE_INDEX).nonzero().flatten().tolist()
    assert positions == list(range(positions[0], positions[-1] + 1))
    return positions[0], positions[-1] + 1


def _check_masking(tokenizer, example, prefix: str, response_text: str) -> None:
    """Prefix fully ignored, supervised span == response tokens, pads ignored."""
    ids = example["input_ids"].tolist()
    labels = example["labels"].tolist()
    mask = example["attention_mask"].tolist()
    start, end = _supervised(example)
    assert tokenizer.decode(ids[:start]) == prefix
    assert labels[:start] == [IGNORE_INDEX] * start
    assert labels[start:end] == ids[start:end]
    expected = tokenizer(format_response(response_text), add_special_tokens=False)
    assert ids[start:end] == expected["input_ids"][: end - start]
    assert all(
        labels[i] == IGNORE_INDEX and ids[i] == PAD_ID
        for i, m in enumerate(mask)
        if m == 0
    )


def test_format_user_message_without_and_with_context() -> None:
    assert format_user_message("質問", "") == "質問"
    assert format_user_message("質問", "  \n") == "質問"
    assert format_user_message("要約して", " 本文 ") == "要約して\n\n補足情報:\n本文"
    assert format_prompt_prefix("要約して", context="本文") == (
        "<|im_start|>system\nあなたは親切なアシスタントです。<|im_end|>\n"
        "<|im_start|>user\n要約して\n\n補足情報:\n本文<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert chat_messages("要約して", "本文")[1] == {
        "role": "user",
        "content": "要約して\n\n補足情報:\n本文",
    }


def test_masking_without_context(reversible_tokenizer) -> None:
    built = build_example_with_info(reversible_tokenizer, "質問", "回答です。", 96)

    assert built is not None
    example, info = built
    assert info.context_used == "" and not info.context_truncated
    assert not info.response_truncated and info.response_tokens > 0
    _check_masking(
        reversible_tokenizer, example, format_prompt_prefix("質問"), "回答です。"
    )
    assert CONTEXT_HEADER not in reversible_tokenizer.decode(example["input_ids"])


def test_masking_with_context_puts_input_in_user_message(reversible_tokenizer) -> None:
    built = build_example_with_info(
        reversible_tokenizer, "要約して", "要約。", 96, context="元の文章"
    )

    assert built is not None
    example, info = built
    prefix = format_prompt_prefix("要約して", context="元の文章")
    assert info.context_used == "元の文章" and not info.context_truncated
    _check_masking(reversible_tokenizer, example, prefix, "要約。")
    assert "補足情報:\n元の文章<|im_end|>" in reversible_tokenizer.decode(
        example["input_ids"]
    )


def test_masking_with_long_context_truncates_context_not_response(
    reversible_tokenizer,
) -> None:
    context = "とても長い参考文章。" * 50
    response = "短い要約。"

    built = build_example_with_info(
        reversible_tokenizer, "要約して", response, CTX_MAX_LENGTH, context=context
    )

    assert built is not None
    example, info = built
    assert info.context_truncated and not info.response_truncated
    assert info.context_used.endswith(CONTEXT_TRUNCATION_MARK)
    assert context.startswith(info.context_used.removesuffix(CONTEXT_TRUNCATION_MARK))
    prefix = format_prompt_prefix("要約して", context=info.context_used)
    _check_masking(reversible_tokenizer, example, prefix, response)
    # The whole response (including <|im_end|>) is still supervised
    start, end = _supervised(example)
    assert reversible_tokenizer.decode(example["input_ids"][start:end]) == (
        format_response(response)
    )
    assert int(example["attention_mask"].sum()) <= CTX_MAX_LENGTH


def test_masking_with_long_context_and_long_output_keeps_response_reserve(
    reversible_tokenizer,
) -> None:
    response = "詳しい回答" * 80

    built = build_example_with_info(
        reversible_tokenizer,
        "要約して",
        response,
        CTX_MAX_LENGTH,
        context="参考文章。" * 80,
    )

    assert built is not None
    example, info = built
    assert info.context_truncated and info.response_truncated
    start, end = _supervised(example)
    assert end == CTX_MAX_LENGTH
    assert end - start >= response_token_reserve(len(response) + 1, CTX_MAX_LENGTH)
    prefix = format_prompt_prefix("要約して", context=info.context_used)
    _check_masking(reversible_tokenizer, example, prefix, response)


def test_masking_with_short_output(reversible_tokenizer) -> None:
    built = build_example_with_info(
        reversible_tokenizer,
        "はいかいいえで答えて",
        "はい",
        CTX_MAX_LENGTH,
        context="文",
    )

    assert built is not None
    example, info = built
    start, end = _supervised(example)
    assert end - start == len("はい") + 1  # response chars + <|im_end|>
    assert example["labels"][end - 1] == IM_END
    _check_masking(
        reversible_tokenizer,
        example,
        format_prompt_prefix("はいかいいえで答えて", context="文"),
        "はい",
    )


def test_masking_with_long_output_without_context(reversible_tokenizer) -> None:
    response = "長い回答" * 100

    built = build_example_with_info(reversible_tokenizer, "説明して", response, 96)

    assert built is not None
    example, info = built
    assert info.response_truncated and not info.context_truncated
    start, end = _supervised(example)
    assert start == len(
        reversible_tokenizer(format_prompt_prefix("説明して"))["input_ids"]
    )
    assert end == 96
    assert example["labels"][end - 1] != IM_END
    _check_masking(
        reversible_tokenizer, example, format_prompt_prefix("説明して"), response
    )


def test_long_instruction_without_context_is_skipped(reversible_tokenizer) -> None:
    assert build_example(reversible_tokenizer, "長い指示" * 40, "a", 96) is None


def test_instruction_dataset_uses_input_field_and_reports_stats(
    reversible_tokenizer,
) -> None:
    rows = [
        {"instruction": "q", "input": "", "output": "a"},
        {"instruction": "要約して", "input": "本文", "output": "要約"},
        {"instruction": "要約して", "input": "長い本文。" * 60, "output": "要約"},
        {"instruction": "説明して", "input": None, "output": "回答" * 100},
        {"instruction": "", "input": "x", "output": "a"},
        {"instruction": "長い指示" * 60, "input": "", "output": "a"},
    ]

    dataset = InstructionDataset(reversible_tokenizer, rows, max_length=CTX_MAX_LENGTH)

    assert dataset.stats() == {
        "examples": 4,
        "with_context": 2,
        "context_truncated": 1,
        "response_truncated": 1,
        "skipped_empty": 1,
        "skipped_truncated": 1,
        "excluded_response_truncated": 0,
        "excluded_by_length": 0,
        "median_response_tokens": 3,
    }
    decoded = reversible_tokenizer.decode(dataset[1]["input_ids"])
    assert "補足情報:\n本文<|im_end|>" in decoded


CTX_ROWS = [
    {"index": 0, "instruction": "q0", "input": "", "output": "短い回答"},
    {"index": 1, "instruction": "q1", "input": "", "output": "長い回答" * 100},
    {"index": 2, "instruction": "q2", "input": "", "output": "短い回答2"},
    {"index": 3, "instruction": "q3", "input": "", "output": "長い回答" * 100},
    {"index": 4, "instruction": "q4", "input": "", "output": "短い回答3"},
]


def test_dataset_excludes_response_truncated_rows(reversible_tokenizer) -> None:
    dataset = InstructionDataset(
        reversible_tokenizer,
        CTX_ROWS,
        max_length=96,
        exclude_response_truncated=True,
        row_ids=[r["index"] for r in CTX_ROWS],
    )

    assert len(dataset) == 3
    assert dataset.row_ids == [0, 2, 4]
    assert dataset.stats()["excluded_response_truncated"] == 2
    assert dataset.stats()["response_truncated"] == 0


def test_dataset_target_examples_stops_early_and_refills(reversible_tokenizer) -> None:
    unfiltered = InstructionDataset(
        reversible_tokenizer, CTX_ROWS[:3], 96, row_ids=[0, 1, 2]
    )
    refilled = InstructionDataset(
        reversible_tokenizer,
        CTX_ROWS,
        max_length=96,
        exclude_response_truncated=True,
        target_examples=len(unfiltered),
        row_ids=[r["index"] for r in CTX_ROWS],
    )

    assert len(unfiltered) == 3 and unfiltered.row_ids == [0, 1, 2]
    # Same size, the dropped truncated row replaced by the next usable row
    assert len(refilled) == 3 and refilled.row_ids == [0, 2, 4]


def test_load_instruction_datasets_rejects_both_limits(
    monkeypatch, fake_tokenizer
) -> None:
    _patch_raw_dataset(monkeypatch, num_rows=10)

    with pytest.raises(ValueError, match="not both"):
        load_instruction_datasets(
            fake_tokenizer,
            dataset_id="org/data",
            max_length=96,
            val_ratio=0.2,
            seed=1,
            max_train_samples=5,
            train_examples=5,
        )


def test_load_instruction_datasets_refill_keeps_size_and_records_ids(
    monkeypatch, reversible_tokenizer
) -> None:
    rows = [
        {
            "index": i,
            "instruction": f"q{i}",
            "input": "",
            "output": "長い回答" * 100 if i % 3 == 0 else f"回答{i}",
        }
        for i in range(60)
    ]
    import datasets

    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda dataset_id, split: datasets.Dataset.from_list(rows),
    )

    def build(**kwargs):
        return load_instruction_datasets(
            reversible_tokenizer,
            dataset_id="org/data",
            max_length=96,
            val_ratio=0.1,
            seed=5,
            **kwargs,
        )

    baseline_train, baseline_val = build(train_examples=20)
    filtered_train, filtered_val = build(
        train_examples=20, exclude_response_truncated=True
    )

    assert len(baseline_train) == len(filtered_train) == 20
    assert filtered_train.stats()["response_truncated"] == 0
    assert baseline_train.stats()["response_truncated"] > 0
    assert filtered_train.stats()["excluded_response_truncated"] > 0
    # The filtered run keeps every non-truncated baseline row and adds others
    assert set(baseline_train.row_ids) - set(filtered_train.row_ids)
    assert set(filtered_train.row_ids) - set(baseline_train.row_ids)
    # Validation is untouched by the training-side filter
    assert baseline_val.row_ids == filtered_val.row_ids


LIST_ROWS = [
    {
        "index": i,
        "instruction": "例を3つ挙げてください" if i % 4 == 0 else f"q{i}",
        "input": "",
        "output": f"1. A{i}\n2. B{i}\n3. C{i}" if i % 4 == 0 else f"回答{i}",
    }
    for i in range(40)
]


def _load_list_arm(reversible_tokenizer, monkeypatch, **kwargs):
    import datasets

    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda dataset_id, split: datasets.Dataset.from_list(LIST_ROWS),
    )
    return load_instruction_datasets(
        reversible_tokenizer,
        dataset_id="org/data",
        max_length=96,
        val_ratio=0.1,
        seed=3,
        **kwargs,
    )


def test_list_rows_sets_the_amount_of_list_supervision(
    reversible_tokenizer, monkeypatch
) -> None:
    from src.list_rules import is_list_instruction

    by_index = {row["index"]: row for row in LIST_ROWS}

    def count_list(dataset) -> int:
        return sum(
            is_list_instruction(by_index[i]["instruction"]) for i in dataset.row_ids
        )

    as_is, _ = _load_list_arm(reversible_tokenizer, monkeypatch, train_examples=20)
    rich, _ = _load_list_arm(
        reversible_tokenizer, monkeypatch, train_examples=20, list_rows=8
    )
    none, _ = _load_list_arm(
        reversible_tokenizer, monkeypatch, train_examples=20, list_rows=0
    )

    assert len(as_is) == len(rich) == len(none) == 20
    assert count_list(rich) == 8
    assert count_list(none) == 0
    assert 0 < count_list(as_is) < 8
    # Every arm uses distinct rows; the composed arms are ordered by row id
    for dataset in (as_is, rich, none):
        assert len(set(dataset.row_ids)) == len(dataset.row_ids)
    for dataset in (rich, none):
        assert dataset.row_ids == sorted(dataset.row_ids)


def test_list_rows_requires_train_examples(reversible_tokenizer, monkeypatch) -> None:
    with pytest.raises(ValueError, match="requires train_examples"):
        _load_list_arm(reversible_tokenizer, monkeypatch, list_rows=5)
    with pytest.raises(ValueError, match="cannot exceed"):
        _load_list_arm(reversible_tokenizer, monkeypatch, train_examples=5, list_rows=6)


LENGTH_ROWS = [
    {
        "index": i,
        "instruction": f"q{i}",
        "input": "",
        "output": ("長い回答です。" * 6) if i % 3 == 0 else "短い",
    }
    for i in range(30)
]


def _load_length_arm(reversible_tokenizer, monkeypatch, **kwargs):
    import datasets

    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda dataset_id, split: datasets.Dataset.from_list(LENGTH_ROWS),
    )
    return load_instruction_datasets(
        reversible_tokenizer,
        dataset_id="org/data",
        max_length=128,
        val_ratio=0.1,
        seed=7,
        **kwargs,
    )


def test_response_length_filter_selects_long_or_short_answers(
    reversible_tokenizer, monkeypatch
) -> None:
    long_train, long_val = _load_length_arm(
        reversible_tokenizer, monkeypatch, train_examples=8, response_tokens_min=30
    )
    short_train, short_val = _load_length_arm(
        reversible_tokenizer, monkeypatch, train_examples=8, response_tokens_max=10
    )

    assert len(long_train) == len(short_train) == 8
    assert long_train.stats()["median_response_tokens"] >= 30
    assert short_train.stats()["median_response_tokens"] <= 10
    assert long_train.stats()["excluded_by_length"] > 0
    assert short_train.stats()["excluded_by_length"] > 0
    assert set(long_train.row_ids).isdisjoint(short_train.row_ids)
    # The filter applies to training only
    assert long_val.row_ids == short_val.row_ids
    assert long_val.stats()["excluded_by_length"] == 0
