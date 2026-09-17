# Experiment G: is the collapse a function of how far the adapter moved?

Pre-registered before the analysis. Results are added in a separate commit.

## Why this analysis

Experiment F showed the diversity collapse is progressive in optimization
steps at a fixed learning rate. The earlier learning-rate sweep showed it is
worse at a higher learning rate at a fixed number of steps (2e-4: 9 of 20
prompts judged degraded; 5e-5: 4 of 20). Both knobs move the same thing: the
size of the LoRA update `dW = (alpha / rank) B A`.

If the distinct-item rate is a monotone function of `||dW||` across both
sweeps, then "how far the adapter travelled" is the controlling variable and
the practical rule becomes "watch the update norm, not the loss".

## Method (no training)

1. `scripts/adapter_norm.py` computes `||dW||` per adapted layer for adapters
   that already exist: the four snapshots of experiment F, its final
   checkpoint, the three learning-rate arms of the first pilot sweep, and the
   six arms of experiments D and E. The base model is never loaded.
2. The three learning-rate checkpoints are evaluated on the 16-prompt list set
   with the unchanged generation settings (generation only, no training).
3. Each checkpoint contributes one point: `||dW||` against the unique /
   attempted item rate, plus duplicate items and median generated tokens.

## Pre-registered predictions

- **H-magnitude**: the unique / attempted item rate decreases monotonically in
  `||dW||` across *both* sweeps pooled, and the step sweep and the
  learning-rate sweep fall on one curve (Spearman correlation <= -0.8 over at
  least 8 checkpoints).
- **H-steps**: the step sweep orders but the learning-rate arms sit off the
  curve, i.e. the optimizer path matters beyond its endpoint.
- **H-neither**: no ordering; the collapse is not summarized by a single
  scalar of the adapter.

## Decision rule

- **SUPPORTED (H-magnitude)**: Spearman rho <= -0.8 pooled, and the
  learning-rate arms lie within the spread of the step sweep at comparable
  `||dW||`.
- **PARTIAL (H-steps)**: each sweep orders on its own (rho <= -0.8 within a
  sweep) but they do not align when pooled.
- **NO SUPPORT (H-neither)**: pooled rho > -0.5.

Ten checkpoints, one seed: this is a correlation over existing artifacts, not
a causal test. A positive result selects the next *causal* experiment (train
to a fixed `||dW||` budget by early stopping versus by lowering the learning
rate, and compare behaviour at matched norm). Norms are compared only between
adapters with the same rank and alpha, which holds for every checkpoint here
(rank 16, alpha 32).

---

## Result: the update magnitude is not the controlling variable

Ten adapters (all rank 16, alpha 32), each evaluated on the same 16-prompt
list set; norms from `scripts/adapter_norm.py`, no training.

| checkpoint | sweep | total \|\|dW\|\| | unique / attempted | duplicate items | median tokens |
|---|---|---|---|---|---|
| step-62 | steps | 8.79 | 0.83 | 15 | 83 |
| step-124 | steps | 10.87 | 0.72 | 27 | 66 |
| step-186 | steps | 11.35 | 0.67 | 28 | 68 |
| expE-control | data | 11.40 | 0.66 | 30 | 68 |
| lr 5e-5 | lr | 11.41 | 0.67 | 30 | 85 |
| step-248 | steps | 11.41 | 0.63 | 33 | 68 |
| expE-short | data | 11.67 | **0.87** | 10 | 46 |
| expE-long | data | 11.88 | 0.67 | 29 | 118 |
| lr 1e-4 | lr | 17.69 | **0.44** | 60 | 68 |
| lr 2e-4 | lr | 30.56 | 0.55 | 38 | 68 |

Spearman correlation between `||dW||` and the unique/attempted item rate:

| set | n | rho | pre-registered meaning |
|---|---|---|---|
| pooled | 10 | **-0.53** | below SUPPORTED (-0.8), above NO SUPPORT (-0.5) |
| step sweep only | 4 | -1.00 | perfectly ordered |
| learning-rate sweep only | 3 | -0.50 | not ordered |
| data-composition arms only | 3 | +0.50 | ordered the wrong way |

**Verdict: H-magnitude is rejected**, and the pre-registered categories do not
cleanly cover the outcome: the pooled value falls between SUPPORTED and NO
SUPPORT, and PARTIAL required *each* sweep to order on its own, which the
learning-rate sweep does not. Recorded as it stands rather than re-labelled.

Two counterexamples decide it:

- **lr 2e-4** has by far the largest update (30.56, 2.7x the 5e-5 arm) but is
  *more* diverse than lr 1e-4 (0.55 vs 0.44). More movement is not
  monotonically worse.
- **expE-short** sits at essentially the same norm as its control (11.67 vs
  11.40) with a much higher unique rate (0.87 vs 0.66) and a third of the
  duplicated items. At matched movement, the training data still decides.

Within one trajectory the norm tracks diversity perfectly - but there it is
little more than a proxy for the step count, and it saturates early (8.79 ->
10.87 -> 11.35 -> 11.41 while the unique rate keeps falling 0.83 -> 0.63).

### What this leaves

The collapse is not summarized by *how far* the adapter moved. Two directions
remain, both measurable without training:

1. **Structure of the update**: effective rank / spectral concentration of
   `dW` per layer. Diversity collapse in generation may mirror rank collapse
   in the adapter.
2. **Data statistics at matched movement**: `expE-short` shows the training
   data changes diversity at a fixed norm. The lexical diversity of the
   teacher answers (type-token ratio, distinct-n) is the obvious candidate,
   and it is computable over the selections already saved.
