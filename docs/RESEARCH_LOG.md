# Research log

What this repository is actually investigating, updated every cycle. The
question is not "can LoRA lower a loss" but:

> In small-data supervised fine-tuning of an instruction-tuned model, why does
> one specific output structure collapse, which property of the teacher data
> predicts it, and does that yield a data-selection rule that generalizes?

## Cycle 5 (2026-09-17, after experiment D)

**Current strongest result.** The length of the teacher answers controls both
the length and the repetitiveness of the fine-tuned model. Holding 497
examples, lr 5e-5 and everything else fixed and selecting the training split by
response length (median 22 / 67 / 195 tokens), the fine-tuned model's answers
run 54 / 95 / 132 median tokens and contain 58 / 29 / 26 duplicated list items.
The same runs separate two things that looked like one failure: training on
short answers gives the **best** format compliance (100% structured, 100% of
requested item counts met, 15/16 answers terminating) and the **worst**
repetition, while long answers give varied content and the worst count
compliance. Validation loss is worst (1.587) in the arm with the best format
compliance.

**What changed scientifically this cycle.** The first dose-response in this
repository, at a dose of 100% of the split. Data selection does move the
failure, but not through the task mix (experiment C) - through the length
distribution of the answers. "List collapse" splits into a compliance axis and
a diversity axis that move in opposite directions.

**Hypotheses killed.** Truncated teacher responses (A); list supervision
quantity (C); validation loss as a proxy for generation quality (A, C and D
all dissociate them, D with the sign reversed).

**Current bottleneck.** Length is confounded with content: filtering by length
also moves the category mix (creative_writing 28 -> 61 rows in the long arm and
2 in the short arm; closed_qa 61 -> 17 and 100). The causal claim "length, not
category" is not yet isolated.

**Next falsifiable experiment.** Experiment E: rebuild the long and short arms
with the control's category distribution (per-category quotas), varying only
length within each category, as far as the pool allows; report the achieved
per-category length gap as the dose. Endpoints unchanged.

**Why this has the highest information gain.** It is the only remaining way to
tell a data-selection rule ("prefer longer teacher answers to avoid repetition,
shorter ones to buy format compliance") from a restatement of the category
mix. Either outcome is publishable inside this repository: a surviving
ordering gives a rule, a vanishing one says category composition is the lever.

**Stop / pivot criterion.** If experiment E cannot reach a per-category length
gap of at least 2x in the majority of categories, declare the separation
untestable with this dataset and move to the optimization regime (rank,
epochs, alpha) with the same endpoints. If E reproduces the ordering, test the
rule at 3,000 examples before claiming it generalizes.

## Cycle 4 (2026-09-16, after experiment C)

**Current strongest result.** The list collapse is *not* caused by the list
training data. Holding 497 examples and lr 5e-5 fixed and varying only how
many list instructions the split contains (0, 44, 132), no list endpoint
orders with list supervision; the arm with **zero** list examples has the best
requested-count satisfaction (100%) and the fewest duplicated items. Across
all arms the same thing happens to every answer: median generated length falls
from 254 tokens (base) to 62-84, items become short templated fragments and
6-8 of 16 list answers repeat an item, while the base model repeats none.
Fine-tuning simultaneously *improves* two behaviours: meeting a requested item
count (67% -> 89-100%) and terminating before `max_new_tokens` (8/16 ->
13-15/16).

**What changed scientifically this cycle.** The collapse was re-described. It
is not "the model cannot produce lists" but "the model produces the right
number of slots, stops on time, and fills the slots with a repeated short
phrase". That is a diversity/length failure that happens to be most visible in
lists, and it is inherited from the whole training distribution rather than
from list rows.

**Hypotheses killed.**
- Truncated teacher responses cause the repetition (experiment A, 25 of 497
  rows replaced, no endpoint moved).
- The amount of list supervision sets list behaviour (experiment C, doses of
  88 and 44 rows, no ordering; zero-list arm is not worse).
- Validation loss tracks generation quality: it is 1.553 in all three arms of
  experiment C and in both arms of experiment A, while the generations differ
  visibly.

**Current bottleneck.** We have a candidate mechanism (answer-length and
item-diversity compression from the dominant short-answer style) but no
experiment yet that manipulates it directly.

**Next falsifiable experiment.** Experiment D: hold 497 examples, lr and
everything else fixed, and build the split from long teacher answers only vs
short teacher answers only (median response length in the dataset is 63
tokens, so a long arm at >= 120 tokens and a short arm at <= 40 tokens are
both available in quantity). Endpoints: the same list metrics plus generated
length; prediction, if the mechanism is length/diversity transfer: the short
arm collapses harder (more duplicate items, shorter answers), the long arm
keeps longer and more varied items.

**Why this has the highest information gain.** Length is a property of every
training row, so the dose is the whole split (100%), the largest available.
The prediction is directional and the two arms bracket the control, so a null
result would rule out the last data-side explanation and point at the
optimization regime (rank, epochs, LoRA scaling) instead of the data.

**Stop / pivot criterion.** If experiment D shows no ordering in duplicate
items or answer length, stop attributing the collapse to data selection: move
to the training regime (e.g. rank 4 vs 16 vs 64 at fixed data), and report the
data-selection conclusion as negative. If D does order, the finding becomes a
data-selection rule ("select teacher answers by length/diversity, not by task
mix") and the next step is to test it at 3,000 examples.

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
