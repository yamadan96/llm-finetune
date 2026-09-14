# Reproducing a training run

This document lists the exact commands that produce the evidence for the
README "Results" section: training config, GPU, runtime, peak VRAM, loss
curve and before/after generation samples. No results are recorded here.

Target environment: a single CUDA GPU with enough memory for
Qwen2.5-7B-Instruct in bfloat16 plus LoRA training (the intended setup is one
48 GB GPU with CUDA 12.1). All commands are run from the repository root and
use relative paths only.

The workflow is **pilot → check → full run**. Do not start the full run until
the pilot passes the gate in [step 3](#3-pilot-gate).

## 1. Environment

```bash
git clone https://github.com/yamadan96/llm-finetune
cd llm-finetune
git status --short          # should print nothing: runs record the commit SHA
uv sync --locked

# Expect torch 2.5.1+cu121, CUDA 12.1 and True
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi

# Optional: download the model and dataset first, so that download time is not
# part of runtime.total_seconds (the Hugging Face cache location is never recorded)
uv run hf download Qwen/Qwen2.5-7B-Instruct
uv run hf download kunishou/databricks-dolly-15k-ja --repo-type dataset
```

Run from a clean checkout: `metrics.json` stores the short commit SHA and a
`git_dirty` flag (tracked files only), and a dirty tree is reported as a
warning by the gate.

## 2. Pilot run

A short run on a deterministic subset (500 training rows and 100 validation
rows, chosen from the seeded split) with the same model, LoRA and optimizer
settings as the full run:

```bash
CHECKPOINT_DIR=./checkpoints/pilot-500 uv run python -m src.train \
  --epochs 1 --batch-size 2 --rank 16 --alpha 32 \
  --max-train-samples 500 --max-val-samples 100 --log-every 10

uv run python scripts/plot_metrics.py ./checkpoints/pilot-500/metrics.json \
  -o ./checkpoints/pilot-500/loss_curve.png

CHECKPOINT_DIR=./checkpoints/pilot-500 uv run python -m src.compare --max-new-tokens 256

uv run python scripts/check_run.py ./checkpoints/pilot-500
```

The validation rows of the pilot are a subset of the full run's validation
rows (limits are applied after the split with the same `--seed`).

## 3. Pilot gate

`scripts/check_run.py` exits with status 1 (NO-GO) if any check fails:

| Check | Passes when |
|---|---|
| `run-completed` | `metrics.json` has `status: completed` (an OOM or crash is recorded as `failed` with the exception type) |
| `loss-decreases` | the mean of the last 10% of logged train losses is below the first 10%, and validation loss is finite |
| `response-mask` | supervised tokens are non-zero and fewer than non-pad tokens, i.e. prompt tokens are excluded from the loss |
| `generation` | `samples.json` exists and every fine-tuned output is non-empty (WARN for repetitive, cut-off or unchanged outputs) |
| `evidence` | GPU, CUDA, peak VRAM, runtime, git commit and `lora_config.json` are recorded, with no absolute local paths |

The script checks recorded evidence only. Before the full run, also:

1. **Read `samples.md`.** The fine-tuned answers must be coherent Japanese
   responses to the prompts, not broken or repetitive text.
2. **Inspect the loss mask on the real tokenizer** (downloads the tokenizer
   only). The decoded supervised part must be exactly the response followed by
   `<|im_end|>`:

   ```bash
   uv run python - <<'EOF'
   from transformers import AutoTokenizer
   from src.dataset import IGNORE_INDEX, build_example

   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   ex = build_example(tok, "日本の首都はどこですか？", "東京です。", max_length=64)
   labels = ex["labels"]
   print("supervised:", repr(tok.decode(labels[labels != IGNORE_INDEX])))
   EOF
   ```

3. **Check headroom.** Compare `memory.max_memory_reserved_gib` with
   `environment.gpus[0].total_memory_gib`. The pilot uses the same
   `--max-length` and `--batch-size` as the full run, so the per-step memory
   footprint should be similar.

If the pilot fails, fix the cause (for example `--batch-size 1` or a smaller
`--max-length` after an OOM) and repeat the pilot with a new `CHECKPOINT_DIR`.
Any changed option is recorded in `metrics.json` `config`, and the full run
must use the same options.

## 4. Full run

```bash
CHECKPOINT_DIR=./checkpoints/full-3ep uv run python -m src.train \
  --epochs 3 --batch-size 2 --rank 16 --alpha 32 --log-every 10

uv run python scripts/plot_metrics.py ./checkpoints/full-3ep/metrics.json \
  -o ./checkpoints/full-3ep/loss_curve.png

CHECKPOINT_DIR=./checkpoints/full-3ep uv run python -m src.compare --max-new-tokens 256

uv run python scripts/check_run.py ./checkpoints/full-3ep
```

The full run takes long; start it inside `tmux` or `screen` so that it
survives a disconnect. `metrics.json` is rewritten at every logged step and
after every epoch, so an interrupted run still leaves its config,
environment, step losses and completed epochs, with `status` set to `failed`
or `interrupted`.

## 5. What to commit

Commit only these files per run:

| File | Produced by | Commit |
|---|---|---|
| `metrics.json` | `src.train` | yes |
| `lora_config.json` | `src.train` | yes |
| `loss_curve.png` | `scripts/plot_metrics.py` | yes |
| `samples.json`, `samples.md` | `src.compare` | yes |
| `lora_weights.pt` | `src.train` | no (ignored via `checkpoints/**/*.pt`) |
| tokenizer files | `src.train` | no (a copy of the base model's tokenizer) |
| console logs, `wandb/` | — | no (may contain machine-specific paths) |

```bash
git add checkpoints/full-3ep/metrics.json checkpoints/full-3ep/lora_config.json \
  checkpoints/full-3ep/loss_curve.png \
  checkpoints/full-3ep/samples.json checkpoints/full-3ep/samples.md
git diff --cached --stat
```

`metrics.json` and the metadata of `samples.json` never contain absolute
paths, hostnames, usernames or environment variables; model and dataset ids
given as local paths are stored as `local:<name>`. `lora_config.json` stores
`--model-id` verbatim because it is used to reload the base model, so pass a
Hugging Face Hub id (the default) for runs that will be published. The
`evidence` check fails if any of these files contains an absolute path.

## 6. Where each Results item comes from

| Results item | Source |
|---|---|
| Training config | `metrics.json` → `config` (all CLI options, dataset sizes, steps, trainable params) and `lora_config.json` |
| GPU and software | `metrics.json` → `environment` (`gpus[].name`, `gpus[].total_memory_gib`, `cuda`, `torch`, `transformers`, `datasets`, `python`, `git_commit`) |
| Runtime | `metrics.json` → `runtime` (`total_seconds`, `epoch_seconds`, `optimizer_steps`, `tokens_per_second`) |
| Peak VRAM | `metrics.json` → `memory` (`max_memory_allocated_gib`, `max_memory_reserved_gib`) |
| Loss curve | `loss_curve.png` (from `train_loss_steps` and `epochs[].val_loss`) |
| Before/After | `samples.md` / `samples.json` |
| Reproduction | the commands in this document at `environment.git_commit` |

### Field definitions

- `train_loss_steps[]`: one entry every `--log-every` optimizer steps;
  `loss` is the mean training loss over the steps since the previous entry and
  `lr` is the learning rate used by the logged step.
- `epochs[]`: `train_loss` is the mean batch loss of the epoch; `val_loss` is
  the token-weighted mean over response tokens of the validation split.
- `runtime.total_seconds`: wall clock from the start of `src.train`, including
  model and dataset loading (`setup_seconds`) and validation
  (`eval_seconds`). `train_seconds` covers the training steps only.
- `runtime.tokens_trained`: response tokens that contribute to the loss
  (shifted labels that are not `-100`); `input_tokens` counts all non-pad
  tokens. Both per-second rates use `train_seconds`.
- `memory`: peak of PyTorch's CUDA caching allocator since the start of the
  run (`torch.cuda.max_memory_allocated` / `max_memory_reserved`), summed over
  visible GPUs. `nvidia-smi` shows more because it includes the CUDA context.
- `samples.json`: greedy decoding (`do_sample=False`, `num_beams=1`,
  `repetition_penalty=1.0`, fixed `max_new_tokens`) with the training ChatML
  template and system prompt. Base and fine-tuned outputs come from the same
  loaded model, before and after inserting the adapter. Greedy outputs are
  expected to repeat on the same GPU and software versions; bitwise equality
  across different GPUs or CUDA kernels is not guaranteed.
- `prompts/compare_ja.json`: hand-written generic instructions, not taken
  from the training dataset. `samples.json` stores its SHA-256 so a changed
  prompt set is detectable.
