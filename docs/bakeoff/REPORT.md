# SLM bake-off — Phase 0 exit gate (BUILD-BRIEF M7)

Generated 2026-08-07T18:02:42.779000+00:00 · mode **measured** · 18 documents · 6 candidates

Produced by `tools/slm_bakeoff.py`. Re-running it **resumes**: the response cache replays every validated answer for free, so a re-run pays only for what this run could not measure.

> ## Read this first: the run is incomplete
>
> The models answered 48 of 72 document-runs (67% of the matrix observed); 33 of those answers also validated. 24 runs never reached a model.
>
> The gap between *answered* and *validated* is the schema-validity finding, not missing data. The gap between *attempted* and *answered* is missing data:
>
> - `x24` Error code: 402 - {'error': 'You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers. Alternatively, su
> - `x13` produced output that failed validation twice for extract_fields: 1 validation error for RawField Value error, complaint_summary must be selected line indices, n
> - `x1` produced output that failed validation twice for extract_fields: 1 validation error for RawField Value error, complaint_summary requires at least one selected l
>
> Every rate below states its own denominator and every ranking row carries `n`. Where `n` is a handful, treat the row as a data point and not a verdict. Re-running `tools/slm_bakeoff.py` resumes rather than restarts: the response cache holds every answer that already validated.

## Recommendation

**google/gemma-3-4b-it** — composite 0.7721; 13/18 documents answered, 9 of those schema-valid.

*Provisional.* 5 of this candidate's documents did not reach the model in this run. Schema validity below is scored over everything observed across runs; the content metrics are scored over the 9 answers that validated.

- schema validity **56%** over n=16 observed documents (0% on the first attempt over n=8, 1 recovered by the repair turn)
- grounding gate 90% of 20 proposed values
- line attribution 100% of 18 attributable fields
- extractive summary F1 0.67 (P 0.53 / R 0.89)
- latency and token cost **not measured**: every completed document was served from the response cache, which costs neither

### The configured models are not served

- `Qwen/Qwen3-1.7B-Instruct` — **not on the router**
- `Qwen/Qwen3-0.6B-Instruct` — **not on the router**

`SLM_MODEL` and `SLM_MODEL_FALLBACK` in `app/settings.py` and `.env.example` therefore name endpoints that return `model_not_found`. Changing them is an owner decision; the recommendation above is the replacement this run measured.

## Ranking

| # | model | answered | validated | composite | schema | grounding | attribution | summary F1 | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `google/gemma-3-4b-it` | 13/18 | 9 | 0.7721 | 0.5625 (n=16) | 0.9 | 1.0 | 0.6666 | 0.0 |
| 2 | `meta-llama/Llama-3.1-8B-Instruct` | 18/18 | 18 | 0.7482 | 1.0 (n=18) | 0.5556 | 1.0 | 0.4075 | 0.0 |
| 3 | `Qwen/Qwen3-4B-Instruct-2507` | 12/18 | 6 | 0.5048 | 0.4 (n=15) | 0.5417 | 1.0 | 0.1112 | 0.0 |
| 4 | `Qwen/Qwen2.5-7B-Instruct` | 5/18 | 0 | 0.0 | 0.0 (n=14) | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | `Qwen/Qwen3-0.6B-Instruct` | 0/0 | 0 | 0.0 | — | 0.0 | 0.0 | 0.0 | 0.0 |
| 6 | `Qwen/Qwen3-1.7B-Instruct` | 0/0 | 0 | 0.0 | — | 0.0 | 0.0 | 0.0 | 0.0 |

*answered* is how many documents reached the model at all in this run. *validated* is how many of those answers passed the contract. Schema validity carries its own `n`: it is scored over every document whose outcome has ever been observed (`schema_outcomes.json`), because the response cache stores only answers that already validated — scoring one run's cache hits alone gives 1.0 and its fresh answers alone gives 0.0. A rank built on a small `n` is not a result, and `not measured` is not a score of zero.

Composite weights: schema_validity_final 0.3, grounding_pass_rate 0.3, line_attribution_accuracy 0.2, summary_f1 0.2. Value agreement is deliberately excluded — see *Caveats*.

## Q5 — does the router honour JSON mode?

**mixed: 1/5 probed models accepted response_format=json_schema and 3/5 accepted json_object. Keep JSON mode plus Pydantic plus one repair retry as the portable baseline and use json_schema opportunistically where the chosen provider serves it**

5 of 6 candidates returned a conclusive probe. An inconclusive row is one whose requests never reached a model, and it is excluded from the verdict rather than counted as a refusal.

| model | conclusive | `json_object` accepted | yields JSON | `json_schema` accepted | JSON without a format | router advertises structured output | 1st-attempt schema validity | probed |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-1.7B-Instruct` | yes | no | no | no | no | — | — | 2026-08-07 |
| `Qwen/Qwen3-0.6B-Instruct` | yes | no | no | no | no | — | — | 2026-08-07 |
| `Qwen/Qwen3-4B-Instruct-2507` | yes | yes | yes | yes | yes | yes | 0.0 | 2026-08-07 |
| `google/gemma-3-4b-it` | no | — | — | — | — | yes | 0.0 | 2026-08-07 |
| `meta-llama/Llama-3.1-8B-Instruct` | yes | yes | yes | no | yes | yes | 0.0 | 2026-08-07 |
| `Qwen/Qwen2.5-7B-Instruct` | yes | yes | yes | no | yes | yes | 0.0 | 2026-08-07 |

Probes are remembered in `q5_probes.json` and reused, because a probe is a dated observation of the router that cannot be re-taken once the account's inference credits are gone. `--refresh-q5` re-takes them.

## Availability and what the endpoint did

- `Qwen/Qwen3-1.7B-Instruct` — **unavailable**; 0 completed, 0 endpoint failures of 0. 
  - `{"error":{"message":"The requested model 'Qwen/Qwen3-1.7B-Instruct' does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}`
- `Qwen/Qwen3-0.6B-Instruct` — **unavailable**; 0 completed, 0 endpoint failures of 0. 
  - `{"error":{"message":"The requested model 'Qwen/Qwen3-0.6B-Instruct' does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}`
- `Qwen/Qwen3-4B-Instruct-2507` — available; 6 completed, 6 endpoint failures of 18. closest live relative of the configured Qwen3-1.7B-Instruct; smallest instruct model on the router with advertised structured output
  - `x6` Error code: 402 - {'error': 'You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers. Alternatively, su
  - `x5` produced output that failed validation twice for extract_fields: 1 validation error for RawField Value error, complaint_summary must be selected line indices, n
- `google/gemma-3-4b-it` — available; 9 completed, 5 endpoint failures of 18. same weight class, different vendor and tokenizer; Gemma's Devanagari coverage is the reason it is here
  - `x5` Error code: 402 - {'error': 'You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers. Alternatively, su
  - `x4` produced output that failed validation twice for extract_fields: 1 validation error for RawField Value error, complaint_summary must be selected line indices, n
- `meta-llama/Llama-3.1-8B-Instruct` — available; 18 completed, 0 endpoint failures of 18. 8B reference point: shows what the extra parameters buy on an extractive task
- `Qwen/Qwen2.5-7B-Instruct` — available; 0 completed, 13 endpoint failures of 18. previous-generation 7B, as a check that 'newer' is doing the work
  - `x13` Error code: 402 - {'error': 'You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers. Alternatively, su
  - `x4` produced output that failed validation twice for extract_fields: 1 validation error for RawField Value error, complaint_summary must be selected line indices, n

## Per-candidate detail

### `Qwen/Qwen3-1.7B-Instruct`

Not measured: {"error":{"message":"The requested model 'Qwen/Qwen3-1.7B-Instruct' does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}.

### `Qwen/Qwen3-0.6B-Instruct`

Not measured: {"error":{"message":"The requested model 'Qwen/Qwen3-0.6B-Instruct' does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}.

### `Qwen/Qwen3-4B-Instruct-2507`

6/18 documents completed (6 served from the response cache).

| metric | value |
|---|---|
| schema validity (final, fresh answers only) | 0.4 |
| schema validity denominator | 15 |
| schema validity (first attempt) | 0.0 |
| grounding pass rate | 0.5417 |
| line attribution accuracy | 1.0 |
| value agreement — exact | 0.3077 |
| value agreement — fuzzy | 0.6923 |
| summary precision | 0.0667 |
| summary recall | 0.3333 |
| summary F1 | 0.1112 |
| fields invented where the corpus has none | 0 |
| validator-owned identifiers leaked into the prompt | 0 |
| latency p50 / p95 | 0.0s / 0.0s |
| tokens per document (mean / max) | 0.0 / 0 |

Per script:

| script | docs | exact | fuzzy | grounding |
|---|---|---|---|---|
| Deva | 2 | 0.0 | 0.5 | 0.5 |
| Latn | 4 | 0.4444 | 0.7778 | 0.5625 |

### `google/gemma-3-4b-it`

9/18 documents completed (9 served from the response cache).

| metric | value |
|---|---|
| schema validity (final, fresh answers only) | 0.5625 |
| schema validity denominator | 16 |
| schema validity (first attempt) | 0.0 |
| grounding pass rate | 0.9 |
| line attribution accuracy | 1.0 |
| value agreement — exact | 0.3889 |
| value agreement — fuzzy | 0.4444 |
| summary precision | 0.5333 |
| summary recall | 0.8889 |
| summary F1 | 0.6666 |
| fields invented where the corpus has none | 0 |
| validator-owned identifiers leaked into the prompt | 0 |
| latency p50 / p95 | 0.0s / 0.0s |
| tokens per document (mean / max) | 0.0 / 0 |

Per script:

| script | docs | exact | fuzzy | grounding |
|---|---|---|---|---|
| Deva | 4 | 0.3 | 0.4 | 0.9091 |
| Latn | 5 | 0.5 | 0.5 | 0.8889 |

### `meta-llama/Llama-3.1-8B-Instruct`

18/18 documents completed (18 served from the response cache).

| metric | value |
|---|---|
| schema validity (final, fresh answers only) | 1.0 |
| schema validity denominator | 18 |
| schema validity (first attempt) | 0.0 |
| grounding pass rate | 0.5556 |
| line attribution accuracy | 1.0 |
| value agreement — exact | 0.4615 |
| value agreement — fuzzy | 0.6154 |
| summary precision | 0.2821 |
| summary recall | 0.7333 |
| summary F1 | 0.4075 |
| fields invented where the corpus has none | 1 |
| validator-owned identifiers leaked into the prompt | 0 |
| latency p50 / p95 | 0.0s / 0.0s |
| tokens per document (mean / max) | 0.0 / 0 |

Per script:

| script | docs | exact | fuzzy | grounding |
|---|---|---|---|---|
| Deva | 6 | 0.2 | 0.4 | 0.625 |
| Latn | 12 | 0.625 | 0.75 | 0.5208 |

### `Qwen/Qwen2.5-7B-Instruct`

Not measured: no document completed.

## Prompt hygiene

- Validator-owned identifiers that reached a prompt: **0** (must be 0 — account_number, disputed_amount, ifsc_code, transaction_date, transaction_reference are sentinel-masked before serialisation, PLAN.md invariant 7).
- **Finding:** no validator claims `mobile`, so nothing masks it and it reaches the SLM verbatim. Harmless here (synthetic corpus) but it is a real complainant's phone number on the air-gapped node. Recorded for OPEN-QUESTIONS.

## Method

- Sample: 18 corpus documents, variants [0, 1, 2] of bilingual_form, complaint_form_deva, complaint_form_en, complaint_letter_en, complaint_letter_mar, email_printout at profile `clean`. Variants 3 and 4 are the locked test split and were not used.
- Input text is the `.gt.json` OCR lines, not real OCR output, so the numbers measure the SLM and not the recogniser. `tools/eval_harness.py` owns the end-to-end figure.
- The prompt is the production one (`build_extraction_prompt`), with deterministic identifiers sentinel-masked from the real validator run. `identifier leaks` above is the assertion that the masking fired.
- Grounding is the production gate (`apply_grounding_gate`) at edit distance 2.
- Each document gets a fresh 8000-token job budget, enforced before the request leaves.
- Temperature 0. Response cache on, so a rerun is free and reproducible.

## Caveats

- **Value agreement is not in the score.** The corpus ground truth carries English field values on Devanagari pages (OPEN-QUESTIONS Q6), so exact agreement penalises a model that correctly copies the Devanagari text. Line attribution is the script-neutral measure and is what the ranking uses.
- Router availability and per-provider routing move. The catalogue as seen by this run is recorded in `results.json`.
- Synthetic corpus text only. Nothing here says how these models behave on real complaint prose.

