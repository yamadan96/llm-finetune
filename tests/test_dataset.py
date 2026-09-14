import datasets
import pytest

import src.dataset as dataset_module
from src.dataset import (
    IGNORE_INDEX,
    InstructionDataset,
    build_example,
    format_chatml,
    format_prompt_prefix,
    format_response,
    limit_indices,
    load_instruction_datasets,
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
