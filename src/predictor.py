"""Singleton predictor for LoRA fine-tuned Qwen2.5."""
import logging
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = "あなたは親切なアシスタントです。"


class Predictor:
    """Singleton: load once, chat many times."""

    _instance: "Predictor | None" = None

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        checkpoint_dir: Path | None = None,
        base_model_id: str = DEFAULT_BASE_MODEL,
    ) -> None:
        if self._initialized:
            return

        if checkpoint_dir and (checkpoint_dir / "lora_weights.pt").exists():
            from .model import load_finetuned_model

            self.model, self.tokenizer = load_finetuned_model(
                checkpoint_dir, base_model_id
            )
            logger.info("Loaded fine-tuned model from %s", checkpoint_dir)
        else:
            from .model import load_base_model

            self.model, self.tokenizer = load_base_model(base_model_id)
            logger.info("Loaded base model (no LoRA checkpoint found)")

        self._initialized = True

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("Call initialize() first.")

        # Build ChatML prompt
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
