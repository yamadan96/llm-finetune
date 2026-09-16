# Experiment C: does the amount of list supervision control list behaviour?

Pre-registered before the runs. Results are added in a separate commit.

## Why this experiment

Every pilot so far collapses on list prompts: the fine-tuned model answers
"five ways to stay focused" with the same item repeated five times, at every
learning rate, and experiment A (removing training rows whose response was cut
at `--max-length`) changed nothing.

Two remaining explanations were measured on the control split of 497 examples
with the corrected rules in `src/list_rules.py` (the earlier detectors were
checked by hand and rewritten: counting items inside prose and missing
space-separated or quoted enumerations produced most of their hits):

| hypothesis | measurable intervention | rows affected |
|---|---|---|
| H-quality: defective list teacher answers teach the collapse | drop list rows whose answer is prose, repeats items or misses a stated count | 8 of 497 (1.6%) |
| H-quantity: too little list supervision, so the short-prose style of the rest of the data takes over | change how many list rows are in the split | 44 present; up to 132 available |

H-quality is below the dose this setup can resolve (1.6%, with detector
precision still imperfect), so it is not tested now. H-quantity is testable
with a large dose and is tested here.

## Intervention (one variable: number of list-instruction rows)

| arm | flags | list rows | rows different from control |
|---|---|---|---|
| `expC-control` | `--train-examples 497` | 44 (as they fall) | – |
| `expC-list-rich` | `--train-examples 497 --list-rows 132` | 132 (3x) | ~88 (17.7%) |
| `expC-list-free` | `--train-examples 497 --list-rows 0` | 0 | 44 (8.9%) |

Every arm keeps 497 training examples, so more list rows means fewer non-list
rows: the intervention changes the **composition** of the split, which is what
"amount of list supervision" means here. Fixed in all arms and checked by
`scripts/compare_runs.py`: base model, `--lr 5e-5`, `--rank 16`, `--alpha 32`,
`--dropout 0.05`, `--epochs 1`, `--batch-size 2`, `--max-length 512`,
`--val-ratio 0.02`, `--seed 42`, `--log-every 10`, the validation split, both
evaluation sets and the greedy generation settings. `config.train_row_ids` is
recorded per arm.

## Primary endpoints

Automatic, on the fixed 16-prompt list set (`prompts/list_eval_ja.json`, 9
prompts state a number of items), computed by `scripts/list_metrics.py`:

1. structured-answer rate (at least two items, not prose)
2. requested-count satisfaction rate
3. answers containing a duplicated item
4. answers that terminated before `max_new_tokens`

Reported for the base model and the fine-tuned model of every arm; the base
column is identical across arms and acts as the reference line.

## Secondary endpoints

The 20-prompt retention set: repetition WARNs, manual per-prompt judgments
split into `in_domain` and `retention`, validation loss, runtime, peak VRAM.

## Decision rule (fixed in advance)

- **SUCCESS**: on the list set, `list-rich` improves endpoints 1-3 over
  `control` *and* `control` is at least as good as `list-free` (a monotone
  ordering rich >= control >= free), with no new degradation on the retention
  prompts.
- **NO SUPPORT**: the dose is present (88 and 44 rows changed) but the list
  endpoints do not order with list supervision.
- **INCONCLUSIVE**: arms differ in something other than the intended variable,
  the metrics disagree with the read outputs, or the endpoints move by a
  single prompt only.

Single seed, one epoch, 497 examples: an ordering seen here is evidence about
this regime, not a general law. Validation loss does not decide the outcome.
