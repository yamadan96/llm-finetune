# Dataset audit

- Dataset: `kunishou/databricks-dolly-15k-ja` split `train`, 15015 rows
- Tokenizer: `Qwen/Qwen2.5-7B-Instruct`, `--max-length 512` (the training code path, including context shortening)
- Usable examples: 14999; skipped empty 0, skipped because nothing of the response fit 16

## Overall

| metric | value |
|---|---|
| rows with an `input` | 4596 (30.6%) |
| context shortened | 1208 (8.1%) |
| response cut at max-length | 882 (5.9%) |
| response <= 10 tokens | 1047 (7.0%) |
| response repeats lines or clauses | 111 (0.7%) |

Response tokens: p5=8, p25=29, p50=63, p75=130, p95=345, mean=110.0, max=7691
Prompt tokens: p5=31, p25=37, p50=47, p75=184, p95=582, mean=158.2, max=8990

## By category

| category | examples | with input | response p50 | response p95 | <= 10 tok | context shortened | response cut | repetitive |
|---|---|---|---|---|---|---|---|---|
| brainstorming | 1768 | 0 (0.0%) | 81 | 309 | 30 (1.7%) | 0 (0.0%) | 30 (1.7%) | 15 (0.8%) |
| classification | 2131 | 0 (0.0%) | 37 | 134 | 179 (8.4%) | 0 (0.0%) | 13 (0.6%) | 25 (1.2%) |
| closed_qa | 1822 | 1822 (100.0%) | 37 | 199 | 198 (10.9%) | 374 (20.5%) | 115 (6.3%) | 1 (0.1%) |
| creative_writing | 711 | 0 (0.0%) | 185 | 803 | 5 (0.7%) | 0 (0.0%) | 105 (14.8%) | 18 (2.5%) |
| general_qa | 2182 | 0 (0.0%) | 124 | 433 | 23 (1.1%) | 0 (0.0%) | 96 (4.4%) | 13 (0.6%) |
| information_extraction | 1512 | 1512 (100.0%) | 44 | 385 | 83 (5.5%) | 439 (29.0%) | 189 (12.5%) | 10 (0.7%) |
| open_qa | 3611 | 0 (0.0%) | 50 | 279 | 523 (14.5%) | 0 (0.0%) | 52 (1.4%) | 19 (0.5%) |
| summarization | 1262 | 1262 (100.0%) | 103 | 418 | 6 (0.5%) | 395 (31.3%) | 282 (22.3%) | 10 (0.8%) |

## Instructions that ask for a list

- 1577 (10.5%) of usable examples
- response shape: inline_list 682 (43.2%), marked_list 610 (38.7%), multi_line 222 (14.1%), single_sentence 63 (4.0%)
- response repeats a line: 8 (0.5%)
- items per response: p50=5, p95=15
- instructions that ask for a specific number of items: 433 (27.5%); response has exactly that many items 209 (48.3%), fewer 57, more 167, a single sentence 21
- response tokens: p50=64, p95=268

## Instructions that ask for a rewrite

- 79 (0.5%) of usable examples
- response vs. the text to rewrite: no_input 19, rewritten 60
- response tokens: p50=115, p95=323

## Repetition kinds in teacher responses

- duplicate lines: 29
- looping phrases: 6
- repeated clauses: 78
