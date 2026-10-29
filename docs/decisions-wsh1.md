# WS-H1 decisions — the two verification jobs (BUILD-BRIEF M0 + M7 bake-off)

Judgment calls made while building and running `tools/bench_ocr_cpu.py`,
`tools/slm_bakeoff.py`, `bench_results.json`, `docs/bakeoff/**` and their tests.
Same conventions as the other decision logs: nothing here contradicts a PLAN.md
invariant, and items marked **OWNER** need a decision outside this lane.

Both jobs were **run for real on this machine** — the benchmark against the
committed PP-OCRv5 ONNX weights, the bake-off against the live HF router. The
measured findings are in section 1 (M0) and section 2 (M7); the design decisions
that produced them follow.

---

## 1. M0 — CPU throughput. **GATE: PASS**, with two surprises

`bench_results.json`, 100 corpus pages sampled across all 10 types and all 6
degradation profiles, seed 1729, on an Intel i7-9750H (6 physical / 12 logical
cores, 31.8 GiB RAM), `ORT_INTRA_OP_THREADS=2`, `inter_op=1`.

| concurrency | p50 | p95 | max | pages/min |
|---|---|---|---|---|
| 1 | 0.53 s | 1.89 s | 2.56 s | 104.0 |
| **2 (target)** | **0.76 s** | **3.14 s** | 3.65 s | **135.4** |

**Gate: p95 3.14 s <= 4 s at concurrency 2 — PASS.** No degradation profile
exceeds the threshold. CPU-first is settled for Phase 0 on hardware of this
class. Concurrency 2 costs 45% on per-page latency and buys 30% on throughput,
which is the expected shape for `workers x intra_op = 4` on 6 physical cores.

### 1.1 The tail is line count, not degradation

Every profile passes, but the aggregate p95 is carried entirely by one document
type:

| type (c=2) | p50 | p95 | lines |
|---|---|---|---|
| bank_statement | 3.14 s | **3.65 s** | 98–102 |
| bilingual_form | 1.07 s | 1.21 s | 24 |
| complaint_letter_en | 0.98 s | 1.19 s | 13 |
| complaint_form_en | 1.03 s | 1.11 s | 25 |
| complaint_form_deva | 0.79 s | 0.86 s | 25 |
| cheque | 0.31 s | 0.47 s | 7 |

Detection is near-constant (p50 0.31 s, p95 0.36 s at c=2 — it always runs on a
960 px-longest-side image); recognition is linear in line count (p50 0.43 s,
p95 2.69 s) and is the whole of the variance. **Every complaint document type
finishes inside 1.3 s.** The margin to the 4 s gate is held entirely by 100-line
supporting documents, so the honest statement of the risk is "a 200-line
statement will breach it", not "degraded pages are slow". Note also that the
slowest *profile* is `clean` (p95 3.65 s) — clean pages are 2339×1654 and yield
the most detections; `bounce` is the fastest at p50 0.37 s.

**OWNER / WS-F**: the fast-path SLA should be expressed per page *with a line-count
caveat*, or supporting documents with >50 lines should be routed to a slower
queue. Extrapolating linearly, ~200 lines is where a page crosses 4 s at
concurrency 2.

### 1.2 Batching is **slower** than the looped path on this CPU

BUILD-BRIEF Fig. 3 / sec. 8 asserts batched recognition is "3–5× on the hottest
path" and PLAN.md invariant 4 mandates it. Measured over 452 real line crops
(153 Deva, 299 Latn), three timed repeats each, with a warm-up pass:

| mode | Deva total | Latn total | speedup vs looped (Deva / Latn) |
|---|---|---|---|
| `looped` (batch 1, bucketed) | 2.472 s | 4.314 s | 1.00 / 1.00 |
| `batch_2` | 2.460 s | 3.889 s | **1.00 / 1.11** |
| `batch_4` | 2.517 s | 3.892 s | 0.98 / 1.11 |
| `batch_8` | 2.845 s | 4.460 s | 0.87 / 0.97 |
| `batch_16` | 3.167 s | 5.025 s | 0.78 / 0.86 |
| `batch_24` (shipped default) | 3.570 s | 5.580 s | **0.69 / 0.77** |
| `batch_32` | 3.489 s | 5.682 s | 0.71 / 0.76 |

The tool splits each run into pad / infer / decode and records the padded tensor
volume, which is what makes the result interpretable rather than merely
surprising:

* **it is not padding waste.** Every bucketed mode pushes exactly the same
  number of padded pixels (6.7 M Deva, 10.9 M Latn) — the planner buckets
  identically at batch 1 and batch 32;
* **it is not the Python CTC decode.** Decode is 0.006–0.024 s, under 1% of the
  total, and it gets *faster* with batching;
* **it is the ONNX inference itself.** `infer_seconds_median` tracks the total
  almost exactly: 2.43 s at batch 1 rising to 3.54 s at batch 24 for Deva. A
  batch-24 tensor at bucket width 640 is 24×3×48×640×4 B ≈ 8.8 MB before any
  activation, well past this CPU's 12 MB L3; batch 1 stays resident. On a
  cache-bound CPU with only two intra-op threads there is almost no per-call
  overhead left to amortise, so the larger working set is a net loss.

**Batching remains output-identical**: `text_agreement_vs_looped` is exactly
1.0 for every batch size on both scripts, across 452 crops. That is the
assertion that matters for invariant 4's correctness; only its performance
claim is contradicted.

**D-H1-1 — invariant 4 is kept; the batch size is the open question.** Nothing
in this lane changes `OCR_BATCH_SIZE`. The batched code path stays: it is
output-identical, it is the only path that is safe when a future deployment has
more cores or more cache (where the brief's 3–5× may well be real), and a
"looped" production path is exactly what invariant 4 exists to forbid. But the
committed default of 24 costs **23–31% of recognition time on this class of
machine**, and recognition is ~60% of page time at concurrency 2.
**OWNER / WS-E2**: consider `OCR_BATCH_SIZE=4` — still a batched path, still
inside no invariant's way, worth ~11% of recognition on Latin pages and roughly
break-even on Devanagari, i.e. ~5–8% of end-to-end page latency here. Re-run
`tools/bench_ocr_cpu.py` on the target CPU node before committing to a number;
this is a hardware-shaped result, not a universal one.

Reproducibility note: two independent 100-page runs (one with light background
load, one clean) gave the same curve to within ±0.04 on every speedup ratio and
the same ordering, so the effect is not measurement noise.

### 1.3 Bucket padding changes the recognised text (accuracy finding)

The benchmark also runs `looped_exact_width` — one crop per call padded to its
*own* width, i.e. the naive PaddleOCR loop with no bucketing. It is the fastest
mode measured (1.11× Deva, 1.17× Latn, because it pushes 15–18% fewer pixels:
5.7 M vs 6.7 M on Deva, 9.0 M vs 10.9 M on Latn), and it is **not usable**: its
decoded text differs from the bucketed path on
**44% of Deva crops and 15% of Latn crops**.

Inspection of the differences shows the bucketed output is the better one — the
exact-width path drops leading characters (`"9. Transaction reference"` →
`"Transaction reference"`, `"10. Previous complaint no."` →
`" Previous complaint no."`). The PP-OCR rec heads are exported for
right-padded crops; feeding an exactly-fitting tensor is out of distribution.

**D-H1-2 — bucket padding is an accuracy mechanism, not only a batching
mechanism.** `plan_recognition_batches` must keep bucketing even at batch size
1. The benchmark reports `bucketed: true/false` per mode and excludes
`looped_exact_width` from `fastest_bucketed_mode` for exactly this reason.
**OWNER / WS-D**: `tools/eval_harness.py` could quantify the CER delta if the
question is ever reopened; this lane measured only that the outputs differ and
which direction the sampled differences ran.

---

## 2. M7 — SLM bake-off. Two hard findings, one blocked measurement

`docs/bakeoff/results.json` + `REPORT.md`, 18 corpus documents (6 types ×
variants 0–2, profile `clean`, 12 Latn / 6 Deva), 6 candidates, accumulated over
several resumed runs against a credit-exhausted account. Schema validity carries
its own `n` because it accumulates in the outcome ledger (D-H1-23); the content
metrics come from the documents whose answers validated.

| # | model | schema validity | grounding | attribution | summary F1 | composite |
|---|---|---|---|---|---|---|
| 1 | `google/gemma-3-4b-it` | 0.56 (n=16) | **0.90** of 20 | 1.00 of 18 | **0.67** | **0.772** |
| 2 | `meta-llama/Llama-3.1-8B-Instruct` | **1.00** (n=18) | 0.56 of 72 | 1.00 of 39 | 0.41 | 0.748 |
| 3 | `Qwen/Qwen3-4B-Instruct-2507` | 0.40 (n=15) | 0.54 of 24 | 1.00 of 13 | 0.11 | 0.505 |
| 4 | `Qwen/Qwen2.5-7B-Instruct` | **0.00** (n=14) | — | — | — | 0.000 |
| — | `Qwen/Qwen3-1.7B-Instruct` (configured) | not served | | | | |
| — | `Qwen/Qwen3-0.6B-Instruct` (configured fallback) | not served | | | | |

The two leaders fail in opposite directions, and the difference is operational,
not cosmetic:

* **Llama-3.1-8B answers the schema every time and over-answers the content.**
  It proposed 72 values across 18 documents — 4.0 per document, i.e. it fills
  every eligible field on every page whether or not the page has it — and only
  **56%** of those survived the grounding gate. Every rejection is a flagged
  field and an escalation, so its apparent completeness converts directly into
  human review.
* **gemma-3-4b is conservative and lands what it throws.** 20 values across its
  9 valid documents (2.2 each) at **90%** grounding, and the best extractive
  summary of the field (F1 0.67, recall 0.89). Its cost is schema discipline:
  44% of its 16 observed answers violated the contract.
* Both are perfect on **line attribution** (1.00): when a value grounds, it
  grounds on the line the corpus tagged. The models are pointing at the right
  text; the disagreement is about how much to say.
* `Qwen/Qwen2.5-7B-Instruct` produced **zero** valid extractions in 14 answers.
  Not a close call.

**Recommendation:** `google/gemma-3-4b-it` on the numbers, but the schema
failures are a *prompt* defect shared by every candidate (§2.3) rather than a
property of the model — fix that first and re-rank, because it is the one
variable that would move both leaders most. If a choice is needed today,
`meta-llama/Llama-3.1-8B-Instruct` is the safer production pick: a schema
failure discards an entire page, while an ungrounded value is caught by the
grounding gate and flagged, which is the failure mode this system is designed
around.

### 2.1 **The configured SLM models do not exist on the router**

`SLM_MODEL=Qwen/Qwen3-1.7B-Instruct` and
`SLM_MODEL_FALLBACK=Qwen/Qwen3-0.6B-Instruct` — the values in `app/settings.py`
and `.env.example`, taken from BUILD-BRIEF sec. 6 — both return:

```
400 {"error":{"message":"The requested model 'Qwen/Qwen3-1.7B-Instruct' does not exist.",
     "type":"invalid_request_error","param":"model","code":"model_not_found"}}
```

The router serves 131 models; neither is among them, and there is no Qwen3
model below 4B on it at all. **Anything that calls the SLM under the laptop
profile with the shipped defaults fails on the first request.**

**OWNER**: `SLM_MODEL` must change. The smallest live instruct model in that
family is `Qwen/Qwen3-4B-Instruct-2507` (nscale, 262 k context, advertises
structured output, $0.01/$0.03 per M tokens) and it is the only candidate that
accepted every response-format variant. Recommended replacement pair:
`SLM_MODEL=Qwen/Qwen3-4B-Instruct-2507`,
`SLM_MODEL_FALLBACK=meta-llama/Llama-3.1-8B-Instruct`. Not applied here —
`app/settings.py` is WS-B's file and `.env.example` is WS-A's.

### 2.2 **Q5 answered: JSON mode yes, grammar/schema is provider-dependent**

Measured with three raw calls per candidate outside the provider (so neither the
cache, the repair retry nor Pydantic can flatter a model): `json_object`,
no `response_format` at all as the control, and `json_schema` with `strict:true`.

| model | `json_object` | yields JSON | `json_schema` | JSON with no format |
|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 200 | yes | **200** | yes |
| `meta-llama/Llama-3.1-8B-Instruct` | 200 | yes | **400** | yes |
| `Qwen/Qwen2.5-7B-Instruct` | 200 | yes | **422** | yes |
| `Qwen/Qwen3-1.7B-Instruct` | 400 model_not_found | — | 400 | — |
| `Qwen/Qwen3-0.6B-Instruct` | 400 model_not_found | — | 400 | — |
| `google/gemma-3-4b-it` | inconclusive (402 on every variant) | | | |

**Answer to Q5: there is no GBNF interface, `response_format` is the whole
surface, `json_object` is accepted by every live model tested (3/3), and the
stronger `json_schema` form is accepted by only 1 of 3** — nscale served the
schema, Together answered 422 and the Llama route answered 400 for the same
request shape. Support is a property of the *provider the router chose*, not of
the model.

Note also the control: all three models emitted valid JSON with **no**
`response_format` at all, so JSON mode is not what is producing well-formed
JSON here — the prompt is. What JSON mode buys is a guarantee under load, not
the behaviour observed on a quiet endpoint. The router's catalogue exposes
`supports_structured_output` per provider and it matched the measurement in
every conclusive case, so it is a usable pre-flight signal — but it is per
provider, and the router picks the provider.

**Recommendation (Q5):** keep the shipped baseline — JSON mode + Pydantic + one
repair retry — as the portable path, because it is the only thing that works
everywhere including a future local llama.cpp server. Use `json_schema`
opportunistically: `app/stages/s5_slm.py` already exports
`EXTRACTION_RESPONSE_SCHEMA`, and `HFAPIProvider` already has the `json_mode`
switch to hang it on. **OWNER / WS-C**: a `response_format` *mode* setting
(`off | json_object | json_schema`) with a one-time downgrade on 422 would let
the strongest available constraint be used per deployment without a code change.

### 2.3 **The prompt does not reliably stop the model writing prose**

**Every schema failure observed** — across four different models — was a
`complaint_summary` failure, and the great majority had the same cause:

```
1 validation error for RawField
  Value error, complaint_summary must be selected line indices, not prose:
  RawField.value must be None and line_indices non-empty (PLAN.md invariant 2)
```

The remainder were an empty `line_indices` list and a citation of lines 24–25 on
a 24-line page — both also `complaint_summary`. **No other field failed the
schema anywhere in the run.**

The model returns `complaint_summary` with **both** a prose `value` and correct
`line_indices`. `SYSTEM_PROMPT_EXTRACT` rule 2 forbids this in words; the
`_EXTRACTION_JSON_SHAPE` line immediately below it shows `"value": "<verbatim
text or null>"` for every field without exception, and the models follow the
shape. The single repair turn did not recover it in any observed case.

Two structural consequences worth stating:

* **schema validity is all-or-nothing per page.** `_parse_extraction` raises on
  the first bad field, so one prose summary discards four correctly extracted
  fields. That is defensible (a partial page is a silent-data-loss risk) but it
  means the PRD P0-8 schema-validity metric is a *page* rate, not a field rate,
  and it will read much worse than field-level accuracy.
* it is the single highest-value prompt fix available, and it is cheap: make the
  JSON shape line show the summary's shape separately, or use `json_schema` with
  `"value": {"const": null}` on the summary branch.

**OWNER / WS-E3**: this is a prompt/schema defect in `app/providers/hf_api.py`,
not a model-selection outcome. Not fixed here — changing the prompt would
invalidate the response cache and the measurement in the same commit.

### 2.4 What could not be measured, and why

The HF account's inference credits are **depleted**, and the router returns:

```
402 {"error": "You have depleted your monthly included credits. Purchase pre-paid
     credits to continue using Inference Providers..."}
```

Refusal is intermittent — a couple of requests land, then the account locks out
for minutes — so the run was resumed repeatedly rather than abandoned. Schema
validity is now observed on 16, 18, 15 and 14 of 18 documents for the four
served candidates. What is missing: the documents no run has reached, one
inconclusive Q5 probe (gemma), and latency/token figures for candidates whose
documents all replayed from the cache on the final pass — a cache hit costs
neither, so both are reported as unmeasured rather than as zero.

Two things had to be built for this to accumulate instead of thrash, and both
are in §3.2: the **Q5 probe cache** (D-H1-20) and the **schema-outcome ledger**
(D-H1-23). Without the ledger a re-run reported 1.0 or 0.0 for schema validity
depending on which half of the sample the endpoint happened to serve — both
were observed before it existed.

No free route exists on the router — all 131 models report `is_free: false` on
every provider — so there is no zero-cost path to finishing. **OWNER**: top up
HF credits (or subscribe to PRO) and re-run
`.venv\Scripts\python.exe tools/slm_bakeoff.py`. It resumes: the response cache
holds every validated answer, `q5_probes.json` every probe taken, and
`schema_outcomes.json` every per-document verdict, so a re-run pays only for
what was never measured.

---

## 3. Design decisions

### 3.1 `tools/bench_ocr_cpu.py`

**D-H1-3 — the benchmark instruments the real provider through its public
seams, and never reimplements it.** `LocalProvider` takes `session_factory` and
`handwriting_detector` as documented constructor arguments; the tool passes a
session wrapper that times `run()` and a crop sink that records what
`ocr_page` produced. Nothing reaches into a private attribute and nothing
duplicates detection or bucketing logic, so the numbers describe the code that
ships. The rec head and its charset come from `provider.rec_heads.peek(script)`
— `RecHeadCache` is public and `peek` deliberately does not disturb LRU order.

**D-H1-4 — the phase timer is thread-local, not locked-and-shared.** At
concurrency 2 two workers hold two pages at once; a shared counter would
attribute one worker's detection to the other worker's page and quietly corrupt
the detect/recognise split at exactly the concurrency the gate is evaluated at.
Each worker resets, runs its page, reads its own numbers.

**D-H1-5 — pages are decoded before the clock starts.** JPEG decode of a
2339×1654 page is not free, and at concurrency 2 a worker decoding would steal
cores from another worker's timed recognition and inflate its p95 for a reason
that has nothing to do with OCR. 100 pages resident is ~1 GB, which this class
of machine has; `--pages` bounds it.

**D-H1-6 — sampling is round-robin over `(type, profile)` cells, seeded.** A
flat random 100-of-295 leaves whole cells empty often enough to matter, and a
profile with two pages in it produces a p95 nobody can interpret. The 100-page
sample lands 15–19 pages in each of the six profiles. A local `random.Random`
instance, never the global one (the WS-D global-seed leak, A-1).

**D-H1-7 — percentiles are nearest-rank, higher neighbour.** Same convention as
`tools/calibrate_quality_gate.py`. Every number in the report is an actual page
time traceable to a `doc_id` in the `pages` array; an interpolated p95 belongs
to no page, and "which document is at the p95?" is the first question anyone
asks.

**D-H1-8 — a page that raises is recorded and excluded from the percentiles.**
A typed failure is a correctness finding for another lane; letting a 5 ms
exception into a latency distribution would flatter it. `errors` carries the
`doc_id` and the message. (100/100 pages succeeded in the committed run.)

**D-H1-9 — the `bounce` profile is sampled and scored like any other.** Those
pages never reach Stage 4 in production, so including them arguably flatters the
gate — but they are also the *fastest* pages measured (p50 0.39 s, they are
small and yield few detections), so excluding them would move the p95 in the
wrong direction for the wrong reason. Per-profile numbers are reported so a
reader can take either view.

**D-H1-10 — `--dry-run` is a first-class mode, not a test hook.** It swaps in a
stub provider and synthetic pixels, exercises every line of sampling,
aggregation, gate evaluation and report construction, and stamps
`mode: "dry_run"` with a gate result of `ADVISORY` rather than `PASS`. That is
what lets `tests/unit/test_bench_tool.py` run the whole tool on a machine with
an empty `models/`, and what stops a dry run ever being mistaken for an answer.

**D-H1-11 — ONNX Runtime's log severity is raised to 3 in the tool's session
factory.** The PaddlePaddle exports carry hundreds of unused initialisers and
ORT warns about each at load; the warnings would bury the benchmark's own
output. Errors and fatals still print. The provider's own sessions are
untouched.

### 3.2 `tools/slm_bakeoff.py`

**D-H1-12 — the sample is GT OCR lines, not real OCR output.** A bake-off over
real OCR scores the recogniser and the SLM together and cannot say which one
moved. `tools/eval_harness.py` owns the end-to-end number. Consequence: these
schema-validity and grounding figures are an *upper bound* — real OCR noise will
lower both.

**D-H1-13 — variants 0–2 only.** Variants 3 and 4 are the locked test split
(TEST-DATA-SPEC sec. 7, enforced by the eval harness). A model chosen against
the held-out split is a model chosen on the test set.

**D-H1-14 — profile `clean`, and the choice is recorded rather than
randomised.** Every degradation profile of a document renders the *same* GT
lines, so profile is irrelevant to a text-only bake-off; picking one and saying
so beats sampling six and implying the result depends on it.

**D-H1-15 — the prompt is the production prompt, with the production masking.**
Each document runs the real `run_validators`, takes the real
`ValidatorReport.masked_spans`, and builds the prompt with the real
`build_extraction_prompt`. The run then *verifies* the result: `identifier_leaks`
counts GT values for validator-owned fields (account number, IFSC, UTR/RRN,
date, amount) that survived into the rendered prompt. **It is 0 across every
document and every candidate** — invariant 7 holds on the bytes, not just on the
field list. The detector is tested against a deliberately unmasked string so it
is not merely a function that never fires.

**D-H1-16 — `mobile` is reported as a separate finding, not as a leak.** No
validator claims mobile numbers, so nothing masks them and they reach the SLM
verbatim. That is not an invariant-7 violation — `mobile` is not in
`DETERMINISTIC_FIELDS` — but on the air-gapped node it is a real complainant's
phone number in a model prompt. Reported under `prompt_hygiene.unclaimed_pii_fields`
and called out in `REPORT.md`. **OWNER**: either add a mobile-number validator
(it is a trivially maskable pattern and would also become an extractable field)
or accept it explicitly. Card fragments are shorter than the 6-character
minimum needle and are not searched; the card validator already claims them.

**D-H1-17 — line attribution ranks; value agreement does not.** OPEN-QUESTIONS
**Q6** means the corpus `fields` dict carries **English** values on Devanagari
pages, so an exact-value comparison punishes a model for correctly copying the
Devanagari text it was shown. The composite score uses schema validity (0.30),
grounding pass rate (0.30), line attribution (0.20) and summary F1 (0.20) —
all script-neutral. Exact and fuzzy value agreement are still measured and
reported *per script*, so the Q6 effect is visible rather than hidden.

**D-H1-18 — endpoint failures do not count against schema validity.** A 402
says nothing about a model's ability to follow a schema. Documents are split
into `completed`, `model_responded` (the model answered, validly or not) and
`api_errors`; every rate publishes its own denominator. The report's headline
coverage number is the **observation rate** (`model_responded / attempted`), not
the completion rate: a document the model answered badly is the schema-validity
*measurement*, and reporting it as missing data would make a model that
reliably breaks the schema look untested. A schema failure is **never**
retried — retrying it would turn schema validity into a measure of
persistence — while an endpoint failure is retried `--attempts` times.

**D-H1-19 — a circuit breaker that throttles, never skips.** After
`--max-consecutive-api-errors` refusals a candidate stops *retrying*: every
remaining document is still attempted once, with no backoff. The first version
skipped them outright and that was wrong — a cached document costs the endpoint
nothing, and skipping it threw away a free replay of a measurement already paid
for. Observed directly: a candidate's grounding and summary scores collapsed to
zero on a re-run because the breaker tripped before reaching its cached
documents. One fast refusal is cheaper than the data it preserves.

**D-H1-20 — Q5 probes are persisted in `docs/bakeoff/q5_probes.json`.** A probe
is a *dated observation of the router*, not a derived number: once the account's
credits are gone it cannot be re-taken, and a later run that silently reported
"unknown" would erase a finding that was genuinely obtained. A probe is stored
only if it is **conclusive on every variant** — if the `json_schema` call was
refused with 402 while the others succeeded, storing it would freeze a billing
state into a permanent, wrong capability claim. `--refresh-q5` re-takes them.

**D-H1-23 — a schema-outcome ledger, because the response cache destroys the
statistic it was built to protect.** The cache stores **only answers that
validated**. So on a re-run the cached documents are guaranteed successes and
the freshly-called ones are exactly the previous failures: scoring the hits
reports **1.0**, scoring the fresh answers reports **0.0**. Both were observed
on real re-runs of this tool. `docs/bakeoff/schema_outcomes.json` records each
document's last observed verdict; schema validity is computed over that union,
so a resumed run improves the estimate instead of corrupting it. A fresh
observation overwrites; a cache hit only fills a gap, and abstains from
first-attempt validity because a replay says nothing about whether the repair
turn was needed. Temperature is 0, so a document's outcome is stable and
remembering it is sound. A candidate with no observed outcome scores `null`,
never `0.0` — "not measured" and "scored badly" must not be the same number in a
table someone picks a model from.

**D-H1-21 — the report leads with its own incompleteness.** `REPORT.md` opens
with a "read this first" block when the completion rate is below 1.0, the
ranking table carries `n = completed/attempted` per row, and the recommendation
is stamped *Provisional* when its candidate did not finish the sample. A
composite of 0.86 over one document and over eighteen are not the same claim,
and a reader who has to derive that from a nested table will not.

**D-H1-22 — every document passes the residency gate before its text is read.**
`require_synthetic_corpus_file` is called on each sampled page under the laptop
profile; a document without a `synthetic: true` declaration raises rather than
being skipped quietly. Tested by flipping a fixture's flag and asserting
`ResidencyViolation`.

---

## 4. Tests

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_bench_tool.py` | 48 | CLI defaults incl. the 4 s gate constant and the brief's batch sizes; seed-stable and cell-spreading sampling; Windows index paths; nearest-rank percentiles; thread-local phase timing under two real threads; `TimedSession` delegation and timing-on-raise; `CropSink` bounds and its refusal to claim handwriting; the exact-width plan; concurrency measurement incl. a provider that raises on every page; every gate branch (PASS / FAIL / boundary / ADVISORY / NOT_MEASURED); machine description; full dry-run end to end and the written file; plus assertions against the **committed** `bench_results.json` — that it answers the gate, that it compares looped against batch 16 and 32, and that batching did not change the recognised text |
| `tests/unit/test_slm_bakeoff_tool.py` | 76 | CLI; the ">= 3 candidates" requirement; that locked variants are never sampled; GT loading, tag aliasing and page class from ground truth; **the residency refusal**; value normalisation; invariant-7 leak detection *and* a positive control that it fires; unclaimed-PII separation; grounding, attribution and summary scoring against a deliberately imperfect stub; fresh-budget-per-document and the exhausted-budget refusal; endpoint retry vs never-retry-a-schema-failure; the circuit breaker throttling rather than skipping *and* still collecting cached documents; API errors excluded from schema denominators while schema failures still count as observations; **the cache-bias fix** — union scoring, cache hits abstaining from first-attempt validity, a fresh observation overwriting a replay but never the reverse, a 402 never entering the ledger; ledger round-trip and corruption tolerance; error-message collapsing; ranking weights, ordering, tie-break and `null`-scores-sort-last; every Q5 verdict branch and every conclusiveness rule incl. the partly-refused probe; probe cache round-trip; dry-run end to end; markdown rendering incl. the partial-run warning; plus assertions against the **committed** `results.json` — >= 3 candidates, both scripts, Q5 answered, and zero identifier leaks |

Both suites run offline with an empty `models/` and no `HF_TOKEN`. The
"committed artefact" tests skip cleanly when the JSON files are absent, so a
fresh clone stays green.

Full regression run: **2310 passed** in `tests/unit`, plus the 6 pre-existing
failures described below. No existing test was changed by this lane. `ruff`
clean across `tools/` and both new test files.

### Pre-existing breakage, not from this lane

`tests/unit/test_providers_local.py` (5 tests) and `tests/unit/test_s4_ocr.py`
(1 test) fail on `AttributeError: '_FakeSession' object has no attribute
'get_outputs'`. Commit **`5f78992` "local provider: reconcile rec charset with
use_space_char exports"** added `session.get_outputs()[0].shape[-1]` at
`app/providers/local.py:1141` and changed only that file — `git show --stat`
confirms the test fake was never updated. The fix is three lines in the test
double:

```python
def get_outputs(self) -> list[Any]:
    return [SimpleNamespace(shape=[1, 25, len(_CHARSET) + 1])]
```

Left alone deliberately: `tests/unit/test_providers_local.py` and
`test_s4_ocr.py` are WS-C / WS-E2 files. **OWNER / WS-C**.

---

## 5. Files

| Path | What |
|---|---|
| `tools/bench_ocr_cpu.py` | M0 spike: real `LocalProvider` over N sampled corpus pages, per-page wall clock at concurrency 1 and 2, looped vs batched at 2/4/8/16/24/32 with a pad/infer/decode split, p50/p95 per profile / type / script, machine description, gate verdict. `--dry-run` needs no weights. |
| `bench_results.json` | The committed run: 100 pages, PASS at p95 3.14 s. |
| `tools/slm_bakeoff.py` | M7 bake-off: router catalogue probe, three-variant Q5 probe, production prompt with real validator masking, grounding gate, per-script scoring, ranked recommendation, resumable. `--dry-run` needs no network. |
| `docs/bakeoff/results.json` | Machine-readable run record incl. the full endpoint-failure matrix. |
| `docs/bakeoff/REPORT.md` | The written recommendation M7 asks for, leading with its own coverage. |
| `docs/bakeoff/q5_probes.json` | Persisted router probes (dated). |
| `docs/bakeoff/schema_outcomes.json` | Per-document schema verdicts, accumulated across resumed runs. |
| `tests/unit/test_bench_tool.py` | 48 tests. |
| `tests/unit/test_slm_bakeoff_tool.py` | 76 tests. |

---

## 6. Open items handed on

| # | Item | Owner |
|---|---|---|
| H1-a | `SLM_MODEL` / `SLM_MODEL_FALLBACK` name models the router does not serve; any laptop-profile SLM call fails with `model_not_found`. Measured replacements: `meta-llama/Llama-3.1-8B-Instruct` (schema 1.00) or `google/gemma-3-4b-it` (best composite). | **OWNER** / WS-B, WS-A |
| H1-b | Q5 resolved: `json_object` accepted by 3/3 live models, `json_schema` by 1/3 (provider-dependent: nscale 200, Together 422, Llama route 400), no grammar interface. Consider a `response_format` mode setting with a 400/422 downgrade. | WS-C |
| H1-c | 17 of 19 schema failures are `complaint_summary` returned as prose *and* indices; the repair turn does not recover it; one bad field discards the whole page. No other field ever failed. Fix the JSON-shape hint or use `json_schema` with a null-valued summary branch, then re-rank — it is the single variable that would move both leading candidates most. | WS-E3 |
| H1-i | Llama-3.1-8B proposes 4.0 values per document and grounds only 56%; gemma-3-4b proposes 1.4 and grounds 90%. Every ungrounded value is a flagged field and an escalation, so candidate choice directly sets the human-review load. Worth a stated policy on which failure the service prefers. | **OWNER** / WS-F |
| H1-d | `OCR_BATCH_SIZE=24` costs 20–30% of recognition time on this CPU; batch 2–4 is the measured optimum. Re-measure on the target node before changing. | WS-E2 / **OWNER** |
| H1-e | Bucket padding changes decoded text vs an exact-width crop (44% Deva / 15% Latn); bucketing must stay even at batch 1. CER delta unquantified. | WS-D (eval harness) |
| H1-f | p95 margin is held by 98–102-line `bank_statement` pages; ~200 lines breaches 4 s at concurrency 2. SLA wording or a separate queue for long supporting documents. | WS-F / **OWNER** |
| H1-g | No validator claims `mobile`, so complainant phone numbers reach the SLM prompt unmasked. Add a validator or accept explicitly. | **OWNER** / WS-E3 |
| H1-h | HF inference credits depleted; schema validity is observed on 14–18 of 18 documents per served candidate and one Q5 probe (gemma) is still inconclusive. Top up and re-run — it resumes from the response cache, the probe cache and the outcome ledger. | **OWNER** |
| H1-j | 6 pre-existing test failures from commit `5f78992` (`_FakeSession` lacks `get_outputs`) in `tests/unit/test_providers_local.py` and `test_s4_ocr.py`. Three-line fix given in §4. | **OWNER** / WS-C |
