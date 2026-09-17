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
