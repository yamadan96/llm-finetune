# Experiment D: does the length of the teacher answers set the collapse?

Pre-registered before the runs. Results are added in a separate commit.

## Why this experiment

Experiments A and C removed two data-side explanations for the list collapse
(truncated teacher responses; the amount of list supervision). What every run
does share is a change in the shape of *all* answers: the base model writes
254 median tokens on the list set with no repeated item, and after fine-tuning
on any composition the answers fall to 62-84 median tokens and 6-8 of 16
repeat an item. The training data's own answers are short: median response 63
tokens, 7.0% at most 10 tokens (`docs/dataset_audit.md`).

Hypothesis (H-length): the adapter transfers the length and the phrasing
density of the teacher answers; once answers are compressed, an instruction
asking for N items is filled with N short slots and the model repeats the
most probable slot instead of finding distinct items.

## Intervention (one variable: the length of the teacher answers)

| arm | flags | teacher answers |
|---|---|---|
| `expD-control` | `--train-examples 497` | as they fall (median 63 tokens) |
| `expD-long` | `--train-examples 497 --response-tokens-min 120` | only answers of at least 120 tokens |
| `expD-short` | `--train-examples 497 --response-tokens-max 40` | only answers of at most 40 tokens |

Both filtered arms are refilled from the same seeded order to 497 examples, so
the arms differ in the length distribution of the answers, not in size. In the
first 4,000 rows of the shuffled training pool, 1,100 rows have at least 120
response tokens and 1,428 have at most 40, so both arms are available without
exhausting the pool. The dose is the whole split: every training example
differs from the control in the filtered arms.

Fixed and checked by `scripts/compare_runs.py`: base model, `--lr 5e-5`,
`--rank 16`, `--alpha 32`, `--dropout 0.05`, `--epochs 1`, `--batch-size 2`,
`--max-length 512`, `--val-ratio 0.02`, `--seed 42`, `--log-every 10`, the
validation split (unfiltered, so validation loss stays comparable), both
evaluation sets and the greedy generation settings.

## Primary endpoints

On the fixed 16-prompt list set, from `scripts/list_metrics.py`:

1. answers containing a duplicated item (control so far: 7/16)
2. duplicate items in total (control so far: 30)
3. median generated tokens (base 254, control 84)
4. structured-answer rate and requested-count satisfaction, as guard rails

## Secondary endpoints

20-prompt retention set (repetition WARNs, manual judgments split into
in_domain and retention), validation loss, runtime, peak VRAM, and the
recorded median response length of each training split
(`config.train_dataset_stats.median_response_tokens`).

## Decision rule (fixed in advance)

- **SUCCESS**: endpoints 1-3 order with the teacher answer length
  (`long` better than `control` better than `short` on duplicates, and median
  generated length increasing with teacher length), with no new degradation on
  the retention prompts.
- **NO SUPPORT**: the arms do not order, i.e. the answer-length distribution
  of the teacher data does not drive the collapse either.
- **INCONCLUSIVE**: the arms differ in something else (checked
  automatically), the metrics disagree with the read outputs, or the
  difference rests on a single prompt.

If this is NO SUPPORT, the data-selection line of attack is exhausted at this
scale and the next cycle moves to the optimization regime (LoRA rank, epochs,
scaling), per the stop criterion in `docs/RESEARCH_LOG.md`.
