# Research log

What this repository is actually investigating, updated every cycle. The
question is not "can LoRA lower a loss" but:

> In small-data supervised fine-tuning of an instruction-tuned model, why does
> one specific output structure collapse, which property of the teacher data
> predicts it, and does that yield a data-selection rule that generalizes?

## Cycle 3 (2026-09-16)

**Current strongest result.** A LoRA fine-tune of Qwen2.5-7B-Instruct on 497
Japanese instruction examples improves in-domain summarization and
context-grounded QA, and at the same time destroys list answers: the model
answers "give five ways" with one item repeated five times. This happens at
lr 2e-4, 1e-4 and 5e-5, and validation loss is anti-correlated with generation
quality across those runs (lowest loss, worst generations). The evaluation
that shows it — greedy fixed-prompt before/after, repetition rules, manual
judgments split into in-domain and capability-retention prompts — is part of
the repository, not a notebook.

**What changed scientifically this cycle.** The dataset audit quantified what
the training data teaches: 7.6% of rows ask for a list, 79.9% of those answer
with a structured list, 11.3% state an item count and only 53.9% of those
match it; rewriting is present in 9 instructions out of 14,999. So the two
prompts that degrade most (rewrite) have almost no supervision, while list
prompts have supervision that is frequent but structurally noisy.

**Hypotheses killed.**
- "Training on responses cut at `--max-length` teaches non-stopping and
  repetition" — experiment A, 25 of 497 rows replaced, no change in any
  pre-registered endpoint (`docs/experiments/2026-09-16-truncation-filter.md`).
- "Lowering the learning rate fixes the collapse" — 2e-4 -> 5e-5 halves the
  number of degraded prompts but leaves list collapse and rewrite loss intact.
- "The pilot is simply too small, so more data will fix it" — not killed, but
  demoted: the failure is task-specific, not uniform, so scaling the same
  distribution is not the cheapest test.

**Current bottleneck.** We cannot yet say what property of the teacher data
predicts the collapse. The quality-based explanation (defective list answers)
affects 8 of 497 rows, below the resolution of this setup.

**Next falsifiable experiment.** Experiment C: hold the training size at 497
and vary the number of list-instruction rows (0, as-is 44, 132), measuring
list behaviour with structural metrics on a fixed 16-prompt list set
(`docs/experiments/2026-09-16-list-supervision.md`).

**Why this has the highest information gain.** It is the only available
intervention whose dose is large (88 and 44 rows changed) and whose direction
is predicted by the hypothesis: if list behaviour is set by how much list
supervision the adapter sees, the endpoints must order rich >= as-is >= free.
A null result kills the quantity explanation and leaves style transfer from
the dominant short-prose answers as the remaining candidate, which is then
testable by varying answer length rather than task mix.

**Stop / pivot criterion.** If experiment C is NO SUPPORT, stop testing data
composition at 497 examples and test the style hypothesis directly (train on a
length-stratified subset), or move to the regime question (does the collapse
disappear at 3,000 examples?) with the same endpoints. If two consecutive
interventions at a dose above 8% produce no movement in the list endpoints,
conclude that small-data SFT collapse is not addressable by selection within
this dataset and report that as the finding.
