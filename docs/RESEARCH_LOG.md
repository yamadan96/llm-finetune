# Research log

What this repository is actually investigating, updated every cycle. The
question is not "can LoRA lower a loss" but:

> In small-data supervised fine-tuning of an instruction-tuned model, why does
> one specific output structure collapse, which property of the teacher data
> predicts it, and does that yield a data-selection rule that generalizes?

## Cycle 8 (2026-09-17, after experiment G)

**Current strongest result.** Unchanged from cycle 7 (loss improves while
diversity collapses inside one run), now with a boundary on the explanation:
the magnitude of the LoRA update does **not** control the collapse. Across ten
adapters at equal rank and alpha, the Spearman correlation between ||dW|| and
the unique/attempted item rate is -0.53 pooled: perfect (-1.00) within one
training trajectory, absent (-0.50) across learning rates and reversed (+0.50)
across data compositions. The learning rate with the largest update (2e-4,
||dW|| = 30.6) is *more* diverse than 1e-4 (0.55 vs 0.44), and `expE-short`
reaches 0.87 at the same norm as its control's 0.66.

**What changed scientifically this cycle.** A scalar summary of the adapter is
ruled out, at no GPU cost. It also resurfaced a data effect that experiment E
had recorded but cycle 6 under-weighted: at matched update magnitude, teacher
answer length still changes diversity (0.87 vs 0.66 unique rate, 10 vs 30
duplicated items) - so both the optimization trajectory and the data
composition move diversity, and neither is captured by ||dW|| or by validation
loss.

**Hypotheses.** Killed: update magnitude as the controlling variable (G);
plus everything killed earlier. Surviving: diversity erodes with optimization
within a trajectory (F); teacher answer length moves both generated length and
diversity at matched movement (E, re-weighted); format compliance is acquired
early and cheaply (F).

**Biggest confound.** Every conclusion rests on a 16-prompt probe with one
seed. The E-short vs control difference (0.87 vs 0.66) is the largest data
effect measured so far and has not been replicated.

**Next falsifiable experiment (H).** Two CPU analyses over existing artifacts:
(a) effective rank / spectral concentration of dW per layer for the same ten
adapters, testing whether generation diversity mirrors the adapter's spectral
collapse; (b) lexical diversity of the teacher answers in each saved selection
(type-token ratio, distinct-2), testing whether the data-side effect at
matched norm is predicted by the diversity of the answers rather than their
length.

**Why this has the highest information gain.** Both are free, both are
falsifiable, and together they separate "the adapter collapses structurally"
from "the data teaches low-diversity answers". Whichever survives selects the
next GPU experiment: a rank/alpha sweep, or a diversity-matched data
selection.

**Stop / pivot criterion.** If neither analysis shows an ordering, stop
looking for static predictors and measure the generation process directly
(token entropy and EOS probability per step across snapshots), which needs GPU
but is the last untested mechanism.

## Cycle 7 (2026-09-17, after experiment F)

**Current strongest result.** Inside a single LoRA fine-tuning run (497
examples, lr 5e-5, 249 steps), validation loss falls monotonically
(1.5922 -> 1.5617 -> 1.5538 -> 1.5517) while the model's distinct-item rate
falls monotonically (0.83 -> 0.72 -> 0.67 -> 0.63) and duplicated items more
than double (15 -> 33). Format compliance is acquired in the first 62 steps
(89% of requested counts met, 13/16 answers terminating, against 67% and 8/16
for the base model) and does not improve much afterwards, and answer length
collapses early (254 -> 83 tokens) and then stays flat. The failure is
therefore progressive in optimization, not in data selection: the same prompt
goes from eight distinct items at step 62, to five at step 124, to a sentence
repeated to the token limit at step 186.

**What changed scientifically this cycle.** The phenomenon is now a *dynamics*
statement with a paired loss curve: "loss improves, behaviour degrades" is
demonstrated within one run rather than inferred across runs. It also shows
that the two behaviours a practitioner cares about have different time
constants - format compliance saturates early, diversity erodes late - which
is what makes early stopping a plausible remedy.

**Hypotheses.** Killed: all four data-side explanations (truncation, list
supervision, teacher length as the cause of repetition, category mix);
validation loss as a behavioural proxy. Surviving: teacher length -> generated
length (SUPPORTED twice, established early in training). New: the controlling
variable is the *magnitude of the adapter update*, which would unify the step
sweep (F) and the learning-rate sweep (2e-4 / 1e-4 / 5e-5, where the highest
learning rate gave the worst generations at equal steps).

**Biggest confound.** Steps and update magnitude are not separated yet: F
varied steps at a fixed learning rate, and the earlier sweep varied the
learning rate at fixed steps. Both move ||dW||.

**Next falsifiable experiment (G).** Compute ||dW|| = ||(alpha/r) B A|| for
every adapter already saved (four snapshots of F, the three learning-rate
arms, the six arms of D and E), evaluate the three learning-rate checkpoints
on the list set (no training, three short generation runs), and test whether
the distinct-item rate is a monotone function of ||dW|| across both sweeps.

**Why this has the highest information gain.** It costs no training, reuses
artifacts, and either unifies two sweeps under one scalar - which turns the
practical rule into "watch the update norm, not the loss" - or shows that
steps and magnitude dissociate, which would point at the optimizer path rather
than its endpoint.

**Stop / pivot criterion.** If the distinct-item rate is monotone in ||dW||
across both sweeps, the next experiment tests the rule directly: early
stopping at a fixed ||dW|| budget versus a fixed step count, measured on a
widened prompt probe with multiple seeds before any claim of generality. If
||dW|| does not order them, pivot to per-layer analysis (which modules move)
before any further training.

## Cycle 6 (2026-09-17, after experiment E)

**Current strongest result.** Small-data LoRA fine-tuning (497 examples, one
epoch, Qwen2.5-7B-Instruct) leaves *format* intact and degrades *content
diversity*: across six arms that differ in task mix, list supervision and
teacher answer length, every fine-tuned model saturates at about six distinct
list items (base model: twelve when attempting eight or more), while
requested-count compliance rises from 67% to 89-100% and termination from 8/16
to 11-16/16. Teacher answer length transfers to generated length in both
experiments (46/68/118 median tokens for teacher 6/67/396, category-matched),
but repetition does **not** follow teacher length: experiment D's ordering was
a category-composition artifact and reverses once the category mix is held
fixed.

**What changed scientifically this cycle.** The failure is no longer "list
collapse". It is a **distinct-item capacity collapse** that becomes visible
whenever a format demands more slots than the fine-tuned model has distinct
content for. Two axes are now separated and measured: length (data-controlled,
dose-response confirmed twice) and diversity (not controlled by any data-side
manipulation tried so far).

**Hypotheses.** Killed: truncated teacher responses (A); list supervision
quantity (C); teacher length as the cause of repetition (E falsifies D's
repetition ordering); validation loss as a behavioural proxy (A, C, D, E).
Surviving: teacher length -> generated length (SUPPORTED twice). New and
untested: the diversity collapse is driven by the optimization regime
(step count, LoRA rank, alpha) rather than by which rows are in the data.

**Biggest confound.** The long arms train on answers that are truncated at
`--max-length 512` (262 of 497 rows in `expE-long`), so "long teacher answers"
and "teacher answers without an end" are entangled at high dose; experiment A
ruled truncation out only at a 25-row dose.

**Next falsifiable experiment (F).** Training dynamics inside a single run:
snapshot the adapter every 62 optimizer steps (62/124/186/249) and evaluate
each snapshot with the same list metrics. Prediction if the collapse is
optimization-driven: distinct-item capacity falls monotonically with steps
while validation loss also falls, i.e. the dissociation is visible *within*
one run and does not need any data manipulation.

**Why this has the highest information gain.** It costs one training run plus
four evaluations, needs no new data design, and discriminates directly between
"which rows are in the data" (four experiments have now failed to move
diversity) and "how far the adapter has moved". A monotone fall with steps
turns the phenomenon into a statement about small-data SFT optimization; a
flat curve says the collapse is set at the first few steps and points at the
adapter's capacity or at the loss objective instead.

**Stop / pivot criterion.** If F shows the collapse is already complete at the
earliest snapshot, pivot to LoRA capacity (rank 4 / 16 / 64 at fixed data and
steps). If F shows a monotone fall, the next question is whether early
stopping on a diversity metric recovers the base model's capacity without
losing the format compliance gains - which would be the publishable rule.

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
