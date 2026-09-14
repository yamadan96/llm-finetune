"""Shared CPU-only test fixtures: a tiny fake tokenizer and a tiny causal LM."""

import re
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PAD_ID = 0
SPECIAL_TOKENS = {"<|im_start|>": 1, "<|im_end|>": 2}
VOCAB_SIZE = 128
# Decode plain ids into CJK ideographs so decoded text never looks like a path
FIRST_DECODED_CHAR = 0x4E00
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

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        """Deterministic, lossy inverse used only to turn generated ids into text."""
        names = {v: k for k, v in SPECIAL_TOKENS.items()}
        pieces: list[str] = []
        for token_id in ids.tolist() if hasattr(ids, "tolist") else ids:
            if token_id in names or token_id == PAD_ID:
                if not skip_special_tokens:
                    pieces.append(names.get(token_id, "<pad>"))
                continue
            pieces.append(chr(FIRST_DECODED_CHAR + token_id))
        return "".join(pieces)

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


class ReversibleTokenizer:
    """Character tokenizer whose ``decode`` is the exact inverse of encoding."""

    pad_token_id = PAD_ID
    eos_token_id = SPECIAL_TOKENS["<|im_end|>"]
    chat_template = None
    offset = 10

    def __call__(
        self, text: str, add_special_tokens: bool = True
    ) -> dict[str, list[int]]:
        ids: list[int] = []
        pos = 0
        for match in _SPECIAL_RE.finditer(text):
            ids += [self.offset + ord(c) for c in text[pos : match.start()]]
            ids.append(SPECIAL_TOKENS[match.group()])
            pos = match.end()
        ids += [self.offset + ord(c) for c in text[pos:]]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        names = {v: k for k, v in SPECIAL_TOKENS.items()}
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return "".join(
            names[i] if i in names else "" if i == PAD_ID else chr(i - self.offset)
            for i in ids
        )


@pytest.fixture
def reversible_tokenizer() -> ReversibleTokenizer:
    return ReversibleTokenizer()
