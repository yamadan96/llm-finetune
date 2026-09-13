"""Shared CPU-only test fixtures: a tiny fake tokenizer and a tiny causal LM."""

import re
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PAD_ID = 0
SPECIAL_TOKENS = {"<|im_start|>": 1, "<|im_end|>": 2}
VOCAB_SIZE = 128
_SPECIAL_RE = re.compile("|".join(re.escape(t) for t in SPECIAL_TOKENS))


class FakeTokenizer:
    """Character-level tokenizer that keeps ChatML special tokens atomic."""

    pad_token_id = PAD_ID
    eos_token_id = SPECIAL_TOKENS["<|im_end|>"]

    def encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        pos = 0
        for match in _SPECIAL_RE.finditer(text):
            ids.extend(self._encode_plain(text[pos : match.start()]))
            ids.append(SPECIAL_TOKENS[match.group()])
            pos = match.end()
        ids.extend(self._encode_plain(text[pos:]))
        return ids

    @staticmethod
    def _encode_plain(text: str) -> list[int]:
        num_special = len(SPECIAL_TOKENS) + 1
        return [num_special + ord(c) % (VOCAB_SIZE - num_special) for c in text]

    def __call__(
        self, text: str, add_special_tokens: bool = True
    ) -> dict[str, list[int]]:
        return {"input_ids": self.encode_text(text)}

    def save_pretrained(self, path: str) -> None:
        (Path(path) / "fake_tokenizer.txt").write_text("fake", encoding="utf-8")


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def build_tiny_causal_lm(seed: int = 0) -> Qwen2ForCausalLM:
    """2-layer randomly initialized Qwen2 model with tiny dimensions."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=PAD_ID,
    )
    return Qwen2ForCausalLM(config)


@pytest.fixture
def tiny_model() -> Qwen2ForCausalLM:
    return build_tiny_causal_lm()
