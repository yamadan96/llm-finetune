# LLM LoRA Fine-tuning from Scratch

**Self-implemented LoRA** applied to **Qwen2.5-7B-Instruct** for Japanese instruction fine-tuning.

LoRA core (`LoRALinear`, `apply_lora`) is implemented without any PEFT library — for learning purposes.

## How LoRA Works

```
Original:  h = Wx          (W frozen, d×k)
LoRA:      h = Wx + BAx × (α/r)

  W ∈ R^(d×k)  — frozen pre-trained weight
  A ∈ R^(r×k)  — trainable, kaiming_uniform init
  B ∈ R^(d×r)  — trainable, zeros init  →  ΔW=0 at start
  r << min(d,k) — rank (e.g. 16)
```

Only `A` and `B` are trained → **0.1% of total parameters**.

## Quick Start

```bash
git clone https://github.com/yamadan96/llm-finetune
cd llm-finetune
uv sync

# Fine-tune on Japanese Dolly dataset
CHECKPOINT_DIR=./checkpoints uv run python -m src.train \
  --epochs 3 --batch-size 2 --rank 16 --alpha 32

# Launch chat demo
CHECKPOINT_DIR=./checkpoints uv run python app.py
```

## Project Structure

```
llm-finetune/
├── src/
│   ├── lora.py      # LoRALinear / apply_lora / save+load weights
│   ├── model.py     # Qwen2.5-7B loading + LoRA application
│   ├── dataset.py   # ChatML formatting (dolly-15k-ja)
│   ├── train.py     # PyTorch training loop (no Trainer)
│   └── predictor.py # Singleton chat predictor
└── app.py           # Gradio ChatInterface
```

## Training Setup

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Dataset | kunishou/databricks-dolly-15k-ja |
| Optimizer | AdamW (lr=2e-4) |
| Scheduler | CosineAnnealingLR |
| GPU memory | ~20GB (bfloat16 + gradient checkpointing) |

## Implementation Highlights

- **No PEFT dependency** — `LoRALinear` replaces `nn.Linear` directly
- **Bare PyTorch loop** — no HuggingFace `Trainer` (educational)
- **ChatML format** — compatible with Qwen2.5 chat template
- **Optional W&B** — set `WANDB_PROJECT` env var to enable logging

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

## License

MIT
