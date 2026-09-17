# Experiment F: when during training does the diversity collapse happen?

Pre-registered before the run. Results are added in a separate commit.

## Why this experiment

Four data-side interventions (truncated responses, list supervision, answer
length, category mix) have failed to prevent the collapse of *distinct item
capacity*: every fine-tuned arm so far saturates near six distinct list items
where the base model reaches twelve, while format compliance improves. If the
data does not decide it, the optimization does - but "optimization" so far
means a single point: 249 steps at lr 5e-5, rank 16.

This experiment looks inside one run instead of comparing runs.

## Intervention

One training run, `--train-examples 497 --lr 5e-5 --epochs 1`, with
`--snapshot-every 62`: the adapter and its validation loss are saved after
62, 124, 186 and 248 optimizer steps, and the final checkpoint is the 249-step
state. Each snapshot is then evaluated with the unchanged 16-prompt list set
and the 20-prompt retention set. Step 0 is the base model, whose numbers are
already in every samples file.

Nothing about the data changes, so any movement across snapshots is a function
of how far the adapter has travelled.

## Primary endpoints, as a function of steps

1. **distinct-item capacity**: mean unique items among answers attempting at
   least 6 items, and unique items / attempted items over all answers
   (base: 12 unique when attempting >= 8; fine-tuned arms so far: ~6)
2. duplicate items in total
3. median generated tokens
4. validation loss at the same step (recorded by the snapshot itself)

Guard rails: structured-answer rate, requested-count satisfaction,
termination, and repetition WARNs on the retention set.

## Predictions

- **Optimization-driven (H-optimization)**: capacity falls monotonically with
  steps while validation loss also falls; the dissociation is visible inside a
  single run.
- **Early saturation (H-early)**: the collapse is already complete at step 62
  and flat afterwards, which points at the first few hundred gradient steps -
  or at the adapter's rank - rather than at prolonged training.
- **No collapse until late (H-late)**: capacity holds until the last snapshot,
  which would make early stopping a sufficient remedy and would tie the
  collapse to the end of the cosine schedule.

Each outcome changes the next experiment: H-optimization -> test early
stopping on a diversity metric; H-early -> test LoRA rank and alpha at fixed
data and steps; H-late -> test schedule and epochs.

## Decision rule

- **SUPPORTED (H-optimization)**: capacity at step 249 is lower than at step
  62 by more than one item *and* the sequence is non-increasing.
- **SUPPORTED (H-early)**: capacity at step 62 is already within one item of
  the final value, and no later snapshot recovers.
- **INCONCLUSIVE**: the curve is non-monotone by more than one item in both
  directions, i.e. the 16-prompt probe cannot resolve the dynamics; then the
  probe must be widened before any further claim.

Single run, single seed: this measures the trajectory of one adapter, not a
law. Validation loss is recorded to test the dissociation, not to decide the
verdict.
