# Experiment A: drop training rows whose response is cut at `--max-length`

Pre-registered before the run. Results are added in a separate commit.

## Question

Training rows whose response does not fit into `--max-length` teach answers
that never reach `<|im_end|>`: 882 of 14,999 usable rows (5.9%), and 22.3% of
`summarization` rows (`docs/dataset_audit.md`). Does removing them from the
training split reduce the repetition and non-stopping seen in the pilots at
every learning rate?

## Intervention (one variable)

| | control | A |
|---|---|---|
| training rows | 497 examples, seeded order | 497 examples, seeded order, rows whose response is cut are skipped and replaced by the next usable rows from the same order |
| flags | `--train-examples 497` | `--train-examples 497 --exclude-response-truncated` |

Everything else is fixed and checked by `scripts/compare_runs.py`: base model,
`--lr 5e-5`, `--rank 16`, `--alpha 32`, `--dropout 0.05`, `--epochs 1`,
`--batch-size 2`, `--max-length 512`, `--val-ratio 0.02`, `--seed 42`,
`--log-every 10`, the validation split (100 rows, **not** filtered, so
validation loss stays comparable), the 20-prompt evaluation set and the greedy
generation settings.

The control is re-run with `--train-examples 497` rather than reusing
`pilot-500-lr5e-5` (`--max-train-samples 500`), so that both arms use the same
sampling rule and differ only in the filter. `metrics.json` records
`config.train_row_ids`, so the sample-set difference between the arms is an
artifact.

## Primary endpoints (fixed in advance)

Objective, on the fixed 20-prompt set:

1. repetition WARNs on fine-tuned outputs (control pilot at 5e-5: 2/20)
2. fine-tuned outputs cut off at `max_new_tokens` (control pilot: 0/20)
3. the four prompts that degraded at every learning rate:
   `list-remote-focus`, `list-welcome-party`, `compare-tcp-udp`,
   `list-pros-cons`

Secondary: validation loss, manual per-prompt judgments (improved / degraded /
same / mixed) for all 20 prompts, runtime and peak VRAM.

## In-domain vs. capability retention

`prompts/compare_ja_20.json` labels every prompt:

- `in_domain`: the training dataset has thousands of rows for the task
  (summarization, classification, QA, lists)
- `retention`: the training dataset barely covers it (rewriting: 9
  instructions contain 書き換え/丁寧/やさしい; arithmetic: 152; deduction
  puzzles: rare), so the prompt probes whether LoRA training damaged an
  ability the base model already had. `rewrite-polite` and `rewrite-plain`
  stay in the set for exactly this reason.

An improvement on `in_domain` prompts without an improvement on `retention`
prompts is an expected outcome, not a failure of the experiment.

## Interpretation rules

- A change in the endpoints shows the effect of **this intervention**
  (filtering plus refilling changes the sample set as well as the truncation
  rate); it does not prove that truncated responses alone caused the earlier
  degeneration.
- Validation loss alone does not decide the outcome: at 2e-4 the lowest
  validation loss came with the worst generations.
- No full run and no larger-sample pilot follow from this experiment alone.
