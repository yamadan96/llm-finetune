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
# Response-only masking with the real Qwen tokenizer on real training rows
# (tokenizer and dataset only, no model weights; exits 1 on failure)
uv run python scripts/inspect_masking.py --num-samples 3 \
  --output ./checkpoints/pilot-500/masking_check.json

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

`scripts/check_run.py` exits with status 1 (`GATE FAIL`) if any check fails.
`GATE PASS` means only that the **minimum conditions for starting the full
run** are met; it is not a judgement of training quality.

| Check | Passes when |
|---|---|
| `run-completed` | `metrics.json` has `status: completed` (an OOM or crash is recorded as `failed` with the exception type) |
| `loss-decreases` | every logged train loss is finite and the mean of the last 10% of logged losses is below the first 10% |
| `response-mask` | supervised tokens are non-zero and fewer than non-pad tokens. This is a count sanity check only; it cannot show that the boundary is in the right place |
| `mask-boundary` | `masking_check.json` from `scripts/inspect_masking.py` passed and covers real rows without `input`, with `input`, with a shortened `input` and with a truncated response (plus a forced truncation): the ignored prefix decodes exactly to the prompt and equals the tokenizer's chat template, the `input` is inside the user message, the supervised span decodes exactly to `response<|im_end|>` (or is a prefix of the response tokens), and padding is ignored |
| `prompt-overlap` | the prompt set recorded in `samples.json` has a committed contamination report (`prompts/<name>.contamination.json`) with the same SHA-256 and no flagged prompt |
| `vram-headroom` | peak reserved VRAM is at most 90% of the GPU's total memory (WARN above) |
| `adapter-load` | `samples.json` → `adapter_load` shows a strict load: LoRA modules present, adapter tensor count equals the checkpoint's (two per module), no missing/unexpected keys, every adapter tensor bit-identical to `lora_weights.pt`, `lora_B` zero right after insertion and non-zero after loading, model in eval mode, and the recorded SHA-256 equals that of `lora_weights.pt` in the run directory |
| `generation` | base generation and fine-tuned generation both produced non-empty output for all prompts (WARN for repetitive, cut-off or unchanged outputs). This does not by itself show that the adapter was loaded; that is `adapter-load` |
| `evidence` | GPU, CUDA, peak VRAM, runtime, git commit, `lora_config.json` and `loss_curve.png` exist, and the artifact metadata contains no absolute local paths and not the current hostname or username (run the check on the training machine) |

Proceed to the full run only if, in addition to `GATE PASS`:

1. **Every WARN has been read and judged by a person.** After 1 epoch on 500
   rows, fine-tuned outputs identical to the base outputs are not by
   themselves a failure.
2. **`samples.md` has been read.** Both columns must be coherent Japanese
   responses to the prompts, not broken or repetitive text.
3. **`masking_check.json` has been looked at**, not only its `ok` flag: the
   supervised text of each example is printed by `inspect_masking.py`.

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

uv run python scripts/inspect_masking.py --num-samples 3 \
  --output ./checkpoints/full-3ep/masking_check.json
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
| `masking_check.json` | `scripts/inspect_masking.py` | yes |
| `lora_weights.pt` | `src.train` | no (ignored via `checkpoints/**/*.pt`) |
| tokenizer files | `src.train` | no (a copy of the base model's tokenizer) |
| console logs, `wandb/` | — | no (may contain machine-specific paths) |

```bash
git add checkpoints/full-3ep/metrics.json checkpoints/full-3ep/lora_config.json \
  checkpoints/full-3ep/loss_curve.png \
  checkpoints/full-3ep/samples.json checkpoints/full-3ep/samples.md \
  checkpoints/full-3ep/masking_check.json
git diff --cached --stat
```

`metrics.json` and the metadata of `samples.json` never contain absolute
paths, hostnames, usernames or environment variables; model and dataset ids
given as local paths are stored as `local:<name>`. The same applies to
`base_model_id` in `lora_config.json`; reloading such a checkpoint (for example
with `src.compare`) then requires `--model-id <local model directory>`, whose
final component must equal `<name>`. Prefer the Hugging Face Hub id (the
default) for runs that will be published. The `evidence` check fails if any of
these files contains an absolute path.

## 6. Where each Results item comes from

| Results item | Source |
|---|---|
| Training config | `metrics.json` → `config` (all CLI options, dataset sizes, steps, trainable params) and `lora_config.json` |
| GPU and software | `metrics.json` → `environment` (`gpus[].name`, `gpus[].total_memory_gib`, `cuda`, `torch`, `transformers`, `datasets`, `python`, `git_commit`) |
| Runtime | `metrics.json` → `runtime` (`total_seconds`, `epoch_seconds`, `optimizer_steps`, `tokens_per_second`) |
| Peak VRAM | `metrics.json` → `memory` (`max_memory_allocated_gib`, `max_memory_reserved_gib`) |
| Loss curve | `loss_curve.png` (from `train_loss_steps` and `epochs[].val_loss`) |
| Before/After | `samples.md` / `samples.json`: all prompts of the fixed set, unedited (see below) |
| Reproduction | the commands in this document at `environment.git_commit` |

### Before/After selection rule

The README shows the outputs for **every** prompt in `prompts/compare_ja.json`
(currently 7), in file order, exactly as written to `samples.md`, including
unchanged, cut-off or worse fine-tuned answers. Prompts are not added,
removed or reworded after a run has been seen; a changed prompt set requires
re-running `src.compare` (its SHA-256 is recorded in `samples.json`).

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
  from the training dataset. A prompt's optional `input` is placed in the
  user message with the same `補足情報:` format as the training data. `samples.json` stores its SHA-256 so a changed
  prompt set is detectable.
- `prompts/compare_ja.contamination.json`: output of
  `uv run python scripts/check_prompt_contamination.py --output prompts/compare_ja.contamination.json`.
  Every prompt is compared with the `instruction`, `input`, `output` and
  `instruction + input` fields of all dataset rows (both splits) after NFKC
  and whitespace normalization: exact match, substring match (prompt in
  field, field or quoted passage of at least 20 characters in the other) and
  character 3-gram Jaccard / containment (flagged at 0.5 / 0.8). The report
  keeps row indices and scores only. Re-run it whenever the prompt set or
  dataset revision changes.
