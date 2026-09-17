# Experiment E: is it the length of the answers, or the category mix?

Pre-registered before the runs. Results are added in a separate commit.

## Why this experiment

Experiment D found a dose-response: selecting the 497 training examples by
teacher answer length (median 22 / 67 / 195 tokens) ordered the fine-tuned
model's answer length (54 / 95 / 132 median tokens) and its repetition volume
(58 / 29 / 26 duplicated list items). But the length filter also moved the
category mix (creative_writing 28 -> 61 rows in the long arm and 2 in the
short arm; closed_qa 61 -> 17 and 100), so "length" and "which tasks" are
confounded.

## Intervention (length varied *within* a fixed category mix)

`scripts/select_rows.py` picks the rows and `src.train --train-row-ids` trains
on exactly them. All arms reproduce the per-category counts of the control run
of experiment D (brainstorming 77, classification 61, closed_qa 61,
creative_writing 28, general_qa 73, information_extraction 55, open_qa 96,
summarization 46 = 497 rows).

| arm | selection | expected teacher length |
|---|---|---|
| `expE-control` | `--selection seeded --category-mix-from <D control>` | dataset-typical |
| `expE-long` | `--selection longest --category-mix-from <D control>` | longest rows of each category |
| `expE-short` | `--selection shortest --category-mix-from <D control>` | shortest rows of each category |

Feasibility was checked before writing the code: in the first 8,000 rows of
the seeded training pool every category has at least twice its quota, and
taking the longest vs the shortest rows of each category gives per-category
median length ratios of 20x to 93x. The achieved per-category medians are
recorded in the selection files and reported with the result.

Fixed in all arms: base model, `--lr 5e-5`, `--rank 16`, `--alpha 32`,
`--dropout 0.05`, `--epochs 1`, `--batch-size 2`, `--max-length 512`,
`--val-ratio 0.02`, `--seed 42`, `--log-every 10`, the validation split, both
evaluation sets and the greedy generation settings.

## Primary endpoints

Unchanged from experiment D, on the 16-prompt list set:

1. answers containing a duplicated item
2. duplicate items in total
3. median generated tokens

with structured-answer rate, requested-count satisfaction and termination as
guard rails, and the 20-prompt retention set as the secondary endpoint.

## Decision rule (fixed in advance)

- **SUCCESS**: endpoints 1-3 order with teacher answer length in the same
  direction as experiment D, with the category mix held fixed. Then "prefer
  longer teacher answers to reduce repetition, shorter ones to buy format
  compliance" is a data-selection rule that does not reduce to the task mix.
- **NO SUPPORT**: the ordering disappears once the category mix is held fixed.
  Then experiment D measured category composition, and the selection rule must
  be stated in terms of categories instead.
- **INCONCLUSIVE**: the achieved per-category length gap is below 2x in most
  categories, the arms differ in something else, or the endpoints rest on a
  single prompt.

Confirmatory for D's direction but under a different, stricter design: a
matching result strengthens the claim, a null result narrows it. Single seed,
one epoch; no full-scale run follows from this experiment alone.

---

## Result: PARTIALLY SUPPORTED (length transfers, repetition does not)

Three runs from commit `9fc8a58`, 497 examples each, `--lr 5e-5`, identical
validation split, identical per-category counts (77/61/61/28/73/55/96/46).
Achieved teacher response medians: **6 / 67 / 396 tokens**, per-category gaps
5x-95x, no row shared between the long and short arms.

| arm | teacher median | generated median | answers with a duplicated item | duplicate items | structured | exact count | terminated | val loss |
|---|---|---|---|---|---|---|---|---|
| base | – | 254 | 0/16 | 0 | 100% | 67% | 8/16 | – |
| `expE-short` | 6 | 46 | 4/16 | 10 | 100% | 100% | 16/16 | 1.604 |
| `expE-control` | 67 | 68 | 7/16 | 30 | 88% | 89% | 14/16 | 1.552 |
| `expE-long` | 396 | 118 | 5/16 | 29 | 100% | 67% | 11/16 | 1.562 |

### Per-endpoint verdicts

| endpoint | experiment D | experiment E (category-matched) | verdict |
|---|---|---|---|
| median generated tokens | 54 / 95 / 132 | 46 / 68 / 118 | **SUPPORTED**: generated length follows teacher length, replicated with the category mix held fixed |
| answers with a duplicated item | 7 / 6 / 5 (short worst) | 4 / 7 / 5 (short best) | **NO SUPPORT**: the ordering does not reproduce |
| duplicate items | 58 / 29 / 26 | 10 / 30 / 29 | **NO SUPPORT**: reverses |

The pre-registered rule required all three endpoints to order, so the
experiment as a whole is **PARTIALLY SUPPORTED**, and the repetition part of
experiment D is downgraded to a category-composition artifact: D's short arm
was also shifted towards classification/closed_qa/open_qa, and once the
category mix is held fixed the extra repetition disappears.

### What replaced it

Reading the outputs and measuring item-level diversity across all six arms of
D and E gives a cleaner description of the failure:

| arm | generated median tokens | mean item length (chars) | distinct-2 over items | unique items / attempted |
|---|---|---|---|---|
| base | 254 | 43.6 | 0.55 | 79/79 |
| `expD-short` | 52 | 10.0 | 0.36 | 67/125 |
| `expE-short` | 46 | 12.1 | 0.49 | 65/75 |
| `expE-control` | 68 | 18.8 | 0.29 | 58/88 |
| `expD-control` | 90 | 21.3 | 0.30 | 55/84 |
| `expD-long` / `expE-long` | 118 | 25.2-28.4 | 0.24-0.29 | 57-58/83-87 |

Item length follows teacher length monotonically in both experiments. What
does *not* follow it is the number of **distinct** items the model can
produce. Pooling every fine-tuned answer of both experiments by how many items
the answer attempts:

| items attempted | answers | mean unique items | max unique |
|---|---|---|---|
| 1-3 | 37 | 2.5 | 3 |
| 4-5 | 29 | 3.5 | 5 |
| 6-7 | 17 | 5.0 | 7 |
| 8-12 | 8 | 5.9 | 8 |
| 13+ | 5 | 6.8 | 11 |

The base model reaches 12 unique items when it attempts 8 or more; every
fine-tuned arm saturates near 6. The visible "list collapse" is this
saturation meeting a format that demands N slots: the model fills the
remaining slots by repeating.

### Standing description

Small-data LoRA fine-tuning in this setup **improves format compliance**
(structure, requested counts, termination), **transfers answer length from the
teacher data**, and **reduces the number of distinct content items the model
can produce**, regardless of the task mix, the list supervision and the length
of the teacher answers. Validation loss separates none of it (1.552-1.604
across arms whose behaviour differs visibly).
