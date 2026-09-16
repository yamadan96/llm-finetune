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
