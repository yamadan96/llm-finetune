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

---

## Result: SUCCESS on the pre-registered endpoints, with a trade-off

Three runs from commit `0c379b8`, 497 training examples each, `--lr 5e-5`,
identical validation split. Median response length of the training split
(recorded in `metrics.json`): **22 tokens (short) / 67 (control) / 195
(long)**, a nine-fold span; 883 and 1,320 rows were skipped by the filters and
refilled from the same seeded order.

### Primary endpoints (16-prompt list set)

| arm | teacher median tokens | answers with a duplicated item | duplicate items | mean duplicates per answer | median generated tokens |
|---|---|---|---|---|---|
| base model | – | 0/16 | 0 | 0.00 | 254 |
| `expD-short` | 22 | 7/16 | 58 | 3.62 | 54 |
| `expD-control` | 67 | 6/16 | 29 | 1.81 | 95 |
| `expD-long` | 195 | 5/16 | 26 | 1.62 | 132 |

All three pre-registered endpoints order with the length of the teacher
answers, in the predicted direction: generated length follows teacher length
(54 -> 95 -> 132 median tokens) and repetition volume falls as teacher answers
get longer (58 -> 29 -> 26 duplicate items).

### The guard rails move the other way

| arm | structured | requested count met (9) | terminated | validation loss |
|---|---|---|---|---|
| `expD-short` | 100% | 100% | 15/16 | 1.587 |
| `expD-control` | 81% | 78% | 13/16 | 1.553 |
| `expD-long` | 88% | 78% | 12/16 | 1.552 |

Training only on short answers gives the **best** format compliance — every
answer is a list, every requested count is met, almost every answer stops —
while doubling the amount of repeated content. Training on long answers gives
longer, more varied items and the worst count compliance. Validation loss is
worst exactly where format compliance is best (1.587 for `short`), the third
dissociation between validation loss and generation behaviour in this
repository.

Retention prompts: no repetition WARN in any arm (0/7 everywhere); in-domain
WARNs 2 (control), 3 (long), 1 (short).

### What this does and does not show

- It shows a **dose-response** between the length of the teacher answers and
  both the length and the repetitiveness of the fine-tuned model's answers,
  at a dose of 100% of the training split.
- It does **not** isolate length from content. Filtering by length also shifts
  the category mix (creative_writing 28 -> 61 rows in `long` and 2 in `short`;
  closed_qa 61 -> 17 and 100). The number of list instructions stays similar
  (44 / 31 / 34), and experiment C showed that list supervision does not drive
  list behaviour, so the confound is about category composition in general,
  not about list supervision.
- The failure mode is now separable in two parts: **format compliance** (item
  counts, structure, stopping) improves with short teacher answers, while
  **content diversity** degrades with them. "Selecting shorter answers"
  therefore buys obedience and pays in repetition.

### Next

Experiment E must break the confound: hold the category mix at the control's
distribution and vary only the answer length within each category, as far as
the pool allows. If the ordering survives, "select teacher answers by length"
is a data-selection rule rather than a proxy for task mix.
