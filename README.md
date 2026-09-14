# LLM LoRA Fine-tuning from Scratch

[![CI](https://github.com/yamadan96/llm-finetune/actions/workflows/ci.yml/badge.svg)](https://github.com/yamadan96/llm-finetune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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

Only `A` and `B` are trained. The adapter matrices are kept in float32 on the
same device as the frozen weight, even when the base model is loaded in bfloat16.

### Trainable parameters (Qwen2.5-7B-Instruct, r=16, q_proj + v_proj)

Computed from the model's
[`config.json`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json):
`hidden_size=3584`, `num_attention_heads=28` (head_dim 128),
`num_key_value_heads=4`, `num_hidden_layers=28`.

```
q_proj: 3584 → 28·128 = 3584   LoRA params = r·(3584 + 3584) = 114,688
v_proj: 3584 →  4·128 =  512   LoRA params = r·(3584 +  512) =  65,536

trainable = 28 layers × (114,688 + 65,536) = 5,046,272  (≈5.05M)
```

The base model has 7,615,616,512 parameters (Hugging Face Hub safetensors
metadata; the same count is obtained by instantiating the architecture from
`config.json`), so the adapter is ≈0.066% of the base model size.

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

For a reproducible run that records the evidence behind a results report
(config, GPU, runtime, peak VRAM, loss curve, before/after samples), follow
[docs/REPRODUCE.md](docs/REPRODUCE.md): pilot run → go/no-go check → full run.

### Platform support

- **Linux / Windows**: `torch` is installed from the PyTorch CUDA 12.1 index
  (`torch 2.5.1+cu121`). Training a 7B model requires a CUDA GPU.
- **macOS (Apple Silicon)**: `uv sync` installs the CPU/MPS build of the same
  torch 2.5 line from PyPI. This is enough to run the unit tests and develop
  locally; training the 7B model on macOS has not been tested.

## Project Structure

```
llm-finetune/
├── src/
│   ├── lora.py      # LoRALinear / apply_lora / save+load weights and config
│   ├── model.py     # Qwen2.5-7B loading + LoRA application + checkpointing
│   ├── dataset.py   # ChatML formatting, label masking, train/val split, sample limits
│   ├── train.py     # PyTorch training loop (no Trainer)
│   ├── evidence.py  # metrics.json: environment, runtime, peak VRAM, step losses
│   ├── compare.py   # Greedy before/after samples for a checkpoint
│   └── predictor.py # Singleton chat predictor
├── scripts/
│   ├── plot_metrics.py              # Loss curve PNG from metrics.json
│   ├── inspect_masking.py           # Response-only masking with the real tokenizer
│   ├── check_prompt_contamination.py # Prompt set vs. training dataset overlap
│   └── check_run.py                 # Pilot gate (minimum conditions for a full run)
├── prompts/         # Fixed prompt set for before/after samples + overlap report
├── docs/REPRODUCE.md
├── tests/           # CPU-only unit tests (no model/dataset downloads)
└── app.py           # Gradio ChatInterface
```

## Training Setup

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, v_proj |
| Dataset | kunishou/databricks-dolly-15k-ja |
| Loss | Assistant response tokens only (see below) |
| Optimizer | AdamW (lr=2e-4, weight_decay=0.01), grad clip 1.0 |
| Scheduler | Cosine with linear warmup (`--warmup-ratio 0.03`), stepped per optimizer step |
| Validation | Seeded hold-out (`--val-ratio 0.02`) |
| Seed | `--seed 42` |
| Precision | bfloat16 base weights + gradient checkpointing |
| GPU memory | Not yet measured |

Run `uv run python -m src.train --help` for all options.

## Implementation Details

### Prompt format and label masking

Each dataset row (`instruction`, optional `input`, `output`) is rendered with
Qwen2.5's ChatML chat template: one system message and one user message.

```
<|im_start|>system\n{system}<|im_end|>\n
<|im_start|>user\n{user message}<|im_end|>\n
<|im_start|>assistant\n{output}<|im_end|>
```

The user message is the instruction alone when `input` is empty. When `input`
is non-empty (about 31% of `databricks-dolly-15k-ja`, e.g. the passage for
closed QA or summarization) it is fixed as:

```
{instruction}

補足情報:
{input}
```

`scripts/inspect_masking.py` checks that this prompt is identical to
`tokenizer.apply_chat_template` of the same messages with
`add_generation_prompt=True`.

The prompt prefix (everything up to and including `<|im_start|>assistant\n`)
and the response (`{output}<|im_end|>`) are tokenized separately and
concatenated, so the supervision boundary is exact. Labels are `-100` for the
prefix and for padding, so the loss covers only the assistant response,
including the closing `<|im_end|>`.

Fitting into `--max-length`:

1. If the prompt would leave fewer than `min(response length, max_length // 4)`
   tokens for the response, the `input` text is shortened at a token boundary
   and `…` is appended. Instructions themselves are never shortened.
2. The sequence is then truncated from the end, so a long response keeps its
   first tokens.
3. Rows with no response token left are skipped, because a row with only
   `-100` labels would produce a NaN loss.

The counts of examples with context, shortened contexts, truncated responses
and skipped rows are recorded in `metrics.json` (`config.train_dataset_stats`,
`config.val_dataset_stats`). See `build_example_with_info()` in
`src/dataset.py`.

### Gradient checkpointing with a frozen base

With all base weights frozen, the embedding output does not require grad. Under
reentrant checkpointing (the default in older transformers releases) this cuts
the gradient path to the LoRA layers inside checkpointed blocks.
`enable_gradient_checkpointing()` calls `model.enable_input_require_grads()` and
`model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`,
which works regardless of the transformers default.

### Scheduler, seed and validation

- The LR warms up linearly over the first `ceil(total_steps × warmup_ratio)`
  optimizer steps and then follows a cosine decay to 0 (per step, not per epoch).
- `--seed` seeds `random`, `numpy`, `torch` and the DataLoader shuffle generator,
  and determines the train/validation split.
- After each epoch the mean validation loss (token-weighted over response
  tokens) is computed; the checkpoint with the lowest validation loss is kept.
  With `--val-ratio 0` the train loss is used instead.

### Checkpoint contents

`CHECKPOINT_DIR` contains:

| File | Content |
|---|---|
| `lora_weights.pt` | LoRA `A`/`B` tensors of the best epoch |
| `lora_config.json` | rank, alpha, dropout, target modules, base model id |
| `metrics.json` | Run config, environment (versions, GPU, git commit), runtime, peak VRAM, step and per-epoch losses; rewritten during training |
| tokenizer files | Saved via `tokenizer.save_pretrained` |
| `loss_curve.png` | `scripts/plot_metrics.py` |
| `samples.json` / `samples.md` | `python -m src.compare` (greedy base vs. fine-tuned outputs) |

`load_finetuned_model()` rebuilds the adapter from `lora_config.json`. Older
checkpoints without that file fall back to the defaults in `src/model.py`.

### Design choices

- **No PEFT dependency** — `LoRALinear` replaces `nn.Linear` directly
- **Bare PyTorch loop** — no HuggingFace `Trainer` (educational)
- **Optional W&B** — set `WANDB_PROJECT` env var to enable logging

## Tests

```bash
uv sync
uv run pytest -q
```

Tests run on CPU without downloading models or datasets. They use a tiny
character-level fake tokenizer and a 2-layer randomly initialized Qwen2 model,
and cover `LoRALinear`, `apply_lora`, weight/config save-load, label masking and
truncation, the warmup/cosine scheduler, gradient flow under gradient
checkpointing, a short end-to-end training run and its `metrics.json`
(including that no absolute paths are recorded), deterministic sample limits,
the loss-curve plot, the pilot check, and greedy before/after generation.

## Results

Training results (loss curves, evaluation scores, sample outputs) are not yet
published. `metrics.json` is written during training so that real curves can be
reported once a run has been completed. The commands that produce and check
this evidence, and which artifacts get committed, are in
[docs/REPRODUCE.md](docs/REPRODUCE.md).

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

## License

MIT — see [LICENSE](LICENSE).
