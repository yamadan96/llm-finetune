# Experiment A: drop training rows whose response is cut at `--max-length`

Pre-registered before the run. Results are added in a separate commit.

## Question

Training rows whose response does not fit into `--max-length` teach answers
that never reach `<|im_end|>`: 882 of 14,999 usable rows (5.9%), and 22.3% of
`summarization` rows (`docs/dataset_audit.md`). Does removing them from the
training split reduce the repetition and non-stopping seen in the pilots at
every learning rate?

## Intervention (one variable)

| | control | A |
|---|---|---|
| training rows | 497 examples, seeded order | 497 examples, seeded order, rows whose response is cut are skipped and replaced by the next usable rows from the same order |
| flags | `--train-examples 497` | `--train-examples 497 --exclude-response-truncated` |

Everything else is fixed and checked by `scripts/compare_runs.py`: base model,
`--lr 5e-5`, `--rank 16`, `--alpha 32`, `--dropout 0.05`, `--epochs 1`,
`--batch-size 2`, `--max-length 512`, `--val-ratio 0.02`, `--seed 42`,
`--log-every 10`, the validation split (100 rows, **not** filtered, so
validation loss stays comparable), the 20-prompt evaluation set and the greedy
generation settings.

The control is re-run with `--train-examples 497` rather than reusing
`pilot-500-lr5e-5` (`--max-train-samples 500`), so that both arms use the same
sampling rule and differ only in the filter. `metrics.json` records
`config.train_row_ids`, so the sample-set difference between the arms is an
artifact.

## Primary endpoints (fixed in advance)

Objective, on the fixed 20-prompt set:

1. repetition WARNs on fine-tuned outputs (control pilot at 5e-5: 2/20)
2. fine-tuned outputs cut off at `max_new_tokens` (control pilot: 0/20)
3. the four prompts that degraded at every learning rate:
   `list-remote-focus`, `list-welcome-party`, `compare-tcp-udp`,
   `list-pros-cons`

Secondary: validation loss, manual per-prompt judgments (improved / degraded /
same / mixed) for all 20 prompts, runtime and peak VRAM.

## In-domain vs. capability retention

`prompts/compare_ja_20.json` labels every prompt:

- `in_domain`: the training dataset has thousands of rows for the task
  (summarization, classification, QA, lists)
- `retention`: the training dataset barely covers it (rewriting: 9
  instructions contain 書き換え/丁寧/やさしい; arithmetic: 152; deduction
  puzzles: rare), so the prompt probes whether LoRA training damaged an
  ability the base model already had. `rewrite-polite` and `rewrite-plain`
  stay in the set for exactly this reason.

An improvement on `in_domain` prompts without an improvement on `retention`
prompts is an expected outcome, not a failure of the experiment.

## Interpretation rules

- A change in the endpoints shows the effect of **this intervention**
  (filtering plus refilling changes the sample set as well as the truncation
  rate); it does not prove that truncated responses alone caused the earlier
  degeneration.
- Validation loss alone does not decide the outcome: at 2e-4 the lowest
  validation loss came with the worst generations.
- No full run and no larger-sample pilot follow from this experiment alone.

---

## Result: NO SUPPORT

Runs on one RTX A6000 from commit `13a8abb`, both arms `--lr 5e-5
--train-examples 497`, control without and A with
`--exclude-response-truncated`. Artifacts: `checkpoints/expA-control`,
`checkpoints/expA-filtered`, `checkpoints/expA-sample-diff.json`,
`checkpoints/expA-comparison.md` on the training machine (not committed).

### Dose of the intervention

| | control | A |
|---|---|---|
| training examples | 497 | 497 |
| of which response cut at 512 | 25 (5.0%) | 0 |
| rows shared with the other arm | 472 | 472 |
| rows replaced | – | 25 removed, 25 refilled |
| validation rows | identical | identical |

So the filter did act on 25 examples, not on a handful: a null result here is
informative rather than a non-intervention.

### Primary endpoints

| endpoint | control | A |
|---|---|---|
| repetition WARNs on fine-tuned outputs | 2 / 20 | 2 / 20 |
| fine-tuned outputs cut off at `max_new_tokens` | 0 / 20 | 0 / 20 |
| `list-remote-focus` | 5 identical list items | 4 identical list items (still degenerate) |
| `list-welcome-party` | 4 identical items | byte-identical output |
| `compare-tcp-udp` | correct, concise | byte-identical output |
| `list-pros-cons` | 2+2 points, one false claim, one confused point | 2+2 points, same false claim, cleaner wording |

Secondary: validation loss 1.553 in both arms; train loss over the last three
logs 1.503 vs 1.512; peak reserved VRAM 17.25 GiB in both; training 139.5 s vs
138.2 s. 14 of the 20 fine-tuned outputs are byte-identical between the arms.

Manual judgments against the base model (by Claude, not blinded):

| run | scope | improved | degraded | same | mixed |
|---|---|---|---|---|---|
| control | all | 4 | 6 | 10 | 0 |
| A | all | 4 | 5 | 10 | 1 |
| control | in_domain (13) | 4 | 4 | 5 | 0 |
| A | in_domain (13) | 4 | 3 | 5 | 1 |
| control | retention (7) | 0 | 2 | 5 | 0 |
| A | retention (7) | 0 | 2 | 5 | 0 |

### Reading

- **NO SUPPORT**: removing the 25 training rows whose response was cut, and
  refilling with non-truncated rows from the same order, left both primary
  endpoints unchanged and did not change any of the four pre-registered
  prompts in kind. The single judgment difference (`list-pros-cons`
  degraded → mixed) is one prompt on an unblinded reading and is not treated
  as an effect.
- This says that at 497 examples, teaching answers without a closing
  `<|im_end|>` is not what drives the repetition and the collapse of list
  answers. It does not clear truncated rows at larger training sizes, and it
  is not evidence about any other data-quality hypothesis.
- Both arms still show the same picture as every earlier pilot: in-domain
  summarization and context QA improve, list prompts degenerate into repeated
  items, and the retention probes (rewrite) get worse while arithmetic and
  deduction stay intact.

### Dose of the next candidates, measured on the same 497 examples

| intervention | rows affected |
|---|---|
| B1: teacher response repeats lines or clauses | 2 (0.4%) |
| B2: list instruction asks for N items, response has a different number | 12 (2.4%) |
| responses of at most 10 tokens | 34 (6.8%) |
| list instructions answered with a single sentence | 4 (0.8%) |

B1 cannot be tested at this sample size: two rows is below the resolution of
this setup. B2 is small but measurable. Any of these either needs a larger
training sample or a differently framed intervention.
