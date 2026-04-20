"""Gradio chat UI for LoRA fine-tuned Qwen2.5-7B-Instruct."""
import logging
import os
from pathlib import Path

import gradio as gr

from src.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR")

# Initialize predictor at startup
predictor = Predictor()
try:
    ckpt = Path(CHECKPOINT_DIR) if CHECKPOINT_DIR else None
    predictor.initialize(checkpoint_dir=ckpt)
    init_error: str | None = None
except Exception as e:
    init_error = str(e)
    logger.error("Failed to initialize predictor: %s", e)


def respond(
    message: str, history: list[dict], system_prompt: str
) -> str:
    if init_error:
        return f"initialization error: {init_error}"
    messages = [m for m in history if m["role"] in ("user", "assistant")]
    messages.append({"role": "user", "content": message})
    return predictor.chat(messages, system=system_prompt)


with gr.Blocks(title="LLM LoRA Fine-tuning Demo") as demo:
    gr.Markdown("# LLM LoRA Fine-tuning Demo")
    gr.Markdown(
        "Qwen2.5-7B-Instruct with LoRA (Hu et al. 2021) "
        "Japanese instruction fine-tuning demo."
    )
    system_prompt = gr.Textbox(
        value="あなたは親切なアシスタントです。",
        label="System Prompt",
        lines=2,
    )
    chat = gr.ChatInterface(
        fn=respond,
        additional_inputs=[system_prompt],
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(
            placeholder="Enter your message...", container=False
        ),
    )

if __name__ == "__main__":
    demo.launch()
