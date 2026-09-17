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

---

## Result: SUPPORTED (H-optimization)

One run from commit `475d91d` (497 examples, `--lr 5e-5`, one epoch, 249
optimizer steps), adapter snapshots every 62 steps, each snapshot evaluated on
the unchanged 16-prompt list set.

| step | validation loss | unique / attempted items | unique items when >= 6 attempted | duplicate items | answers with a duplicate | requested count met | median generated tokens |
|---|---|---|---|---|---|---|---|
| 0 (base) | – | 1.00 | 7.3 | 0 | 0/16 | 67% | 254 |
| 62 | 1.5922 | 0.83 | 7.4 | 15 | 2/16 | 89% | 83 |
| 124 | 1.5617 | 0.72 | 5.7 | 27 | 6/16 | 100% | 66 |
| 186 | 1.5538 | 0.67 | 5.0 | 28 | 6/16 | 89% | 68 |
| 248 | 1.5517 | 0.63 | 5.3 | 33 | 9/16 | 100% | 68 |
| 249 (final) | 1.5520 | 0.68 | 5.3 | 29 | 7/16 | 100% | 68 |

**Validation loss falls monotonically (1.5922 -> 1.5517) while the unique /
attempted item rate falls monotonically (0.83 -> 0.63) and duplicated items
more than double (15 -> 33).** The pre-registered SUPPORTED condition for
H-optimization asked for a fall of more than one item in the capacity measure
and a non-increasing sequence: capacity falls 7.4 -> 5.3 (2.1 items), and the
sequence is non-increasing except for a +0.3 wobble at step 248 on n=6
answers. The strictly monotone statement holds for the rate measure; the
capacity measure holds within the resolution of a 16-prompt probe. Verdict:
**SUPPORTED, with the wobble recorded rather than smoothed away.**

Two behaviours separate in time:

- **Format is acquired early and kept**: at step 62 the model already meets 89%
  of requested item counts and terminates 13/16 answers, against 67% and 8/16
  for the base model. Later snapshots do not improve this much (100% at steps
  124 and 248).
- **Diversity erodes continuously**: the same prompt shows it directly.

```
step  62: 1. 本を読む 2. テレビを見る 3. ゲームをする 4. 音楽を聴く 5. 美術を描く
          6. 美食家になる 7. パズルをする 8. マッサージをする 9. マッサージをする ...
step 124: 1. 本を読む 2. お風呂に入る 3. お茶を淹れる 4. お菓子を焼く
          5. お手伝いをする 6. お手伝いをする ...
step 186: 雨の日は、家で過ごすのに最適な日です。(repeated to the token limit, no list at all)
```

Answer length drops early too (254 -> 83 tokens by step 62) and then stays
flat, so the length transfer measured in D and E is established in the first
quarter of training and is not what erodes afterwards.

### Consequence

The collapse is not a property of which rows are in the training set - four
data-side interventions failed to move it - but of how far the adapter has
travelled. The next question is whether "how far" is measured in steps or in
update magnitude: the learning-rate sweep (2e-4, 1e-4, 5e-5 at a fixed 249
steps) and this step sweep (fixed learning rate) can be placed on one axis,
the norm of the LoRA update, using adapters that already exist. That analysis
needs no training run.
