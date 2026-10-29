# WS-H2 decisions — the M7 exit-gate evaluation

Judgment calls made while building `tools/run_corpus_eval.py`, running it over the locked dev
split and writing `docs/eval/**`. Same conventions as the other decision files: nothing here
contradicts a PLAN.md invariant, and items marked **OWNER** need a decision outside this lane.

Lane: `tools/run_corpus_eval.py`, `docs/eval/**`, `tests/unit/test_run_corpus_eval.py`, this
file, and the Q6/Q7/Q9+ rows of `OPEN-QUESTIONS.md`. Nothing outside it was touched.

WS-H2 measures; it does not fix. Every number in `docs/eval/REPORT.md` came out of the
shipped code at `git rev-parse HEAD` on the day of the run, with no tuning in between. Where a
target is missed the remedy is named and left for its owning lane.

---

## 1. The driver

### D-H2-1 · The locked split rule is imported from the harness, never reimplemented

`tools/run_corpus_eval.py` loads `tools/eval_harness.py` as a module and calls its
`load_corpus()` and `select_documents()`. A driver with its own copy of "variants 3 and 4 are
held out" is one refactor away from writing predictions the harness refuses to score — or,
much worse, from *scoring* the held-out variants because only one of the two files was
updated. `--split all` without `--final` therefore raises the harness's own `LockedSplitError`
from inside the driver, and `--final` is passed through to `evaluate()` unchanged.

Cost: the driver imports a module that imports nothing from `app/` (WS-D adjudication A-2), so
the dependency runs one way only and the scorer stays standalone.

### D-H2-2 · Sequential, one document at a time

Concurrency would roughly halve the wall clock of a cold sweep — the SLM call is ~10-20 s of
pure network wait and the CPU is idle for all of it. It is not used, because **p50/p95
end-to-end latency is a PRD target** and a latency measured under self-inflicted contention is
not that target's number. The response cache, not parallelism, is what makes a re-run cheap.

A second reason: `LocalProvider` holds ONNX sessions and a CTranslate2 translator that this
lane has not audited for concurrent use. Measuring throughput by introducing an unaudited
race would be a poor trade for 25 minutes.

### D-H2-3 · Resumability is the response cache, not a checkpoint file

Every SLM response is keyed `sha256(model + prompt + params)` in `PROVIDER_CACHE_PATH` and
only *validated* answers are stored. An interrupted sweep therefore costs the OCR again
(~1.3 s/page) and the model calls not at all, and — the part that matters — a document whose
SLM call failed is **not** cached, so simply re-running the sweep retries exactly the failures
and hits cache for everything else. That is why the run is a first pass plus a repair pass
rather than one pass with an inner retry loop: the retry is idempotent at the sweep level and
needs no state of its own.

`--resume` (skip documents that already have a prediction file) exists for the case where even
the OCR is not worth repeating. It is off by default because a default that silently skips
work makes "the run" ambiguous.

### D-H2-4 · The per-job token budget is reset per document — without this the sweep dies at ~document 3

`HFAPIProvider` owns a single `JobBudget(JOB_TOKEN_BUDGET)` for its lifetime, and
`JOB_TOKEN_BUDGET` is 8000 — a *per-job* ceiling. One document costs roughly 800-2400 tokens,
so a sweep that never resets the budget exhausts it after three or four documents and every
subsequent extraction raises `TokenBudgetExceeded`. The orchestrator absorbs that as a
`slm_unavailable:` warning (D-F-8) and keeps going, so the failure mode is **silent**: a
plausible-looking report in which the SLM simply stopped contributing a quarter of the way in.

The driver calls `remote.reset_budget(job_id=doc_id)` before each document.
`TestSweepMechanics::test_the_token_budget_is_reset_per_document` locks it in.

This is a property of the sweep, not a bug in the provider — a per-job budget behaves
correctly for the API, which resets per job. It is recorded here because any future batch
driver has to know it. See also Q9.

### D-H2-5 · A failed document becomes a prediction with `pages: []` and `fields: {}`

Three options for a document the pipeline could not finish: write nothing (the harness reports
it "missing" and excludes it from every accuracy denominator), write a marker the harness
ignores, or write an empty-but-well-formed prediction.

Chosen: the third. `pages: []` makes the harness score every ground-truth line as a deletion;
`fields: {}` makes every ground-truth field a miss. That is what "the service produced nothing
for this page" actually costs, and burying it in a coverage footnote would flatter the CER and
the field recall of a run that failed.

The failure detail travels in an `error` block the harness ignores, and the manifest counts
failures separately so the aggregate can be read either way.

### D-H2-6 · A failed document reports **no** stage timings

The counterpart to D-H2-5. The pipeline reported no `stage_timings_ms` for a document that
raised, and synthesising an `e2e_total` from the failure's wall clock would push a 240-second
timeout into the p95 of stages that never ran. The wall clock is recorded in the manifest
(`documents[].wall_ms` and `error.wall_ms`) where it can be read as what it is.

### D-H2-7 · The per-document wall-clock budget abandons rather than kills

Windows has no `SIGALRM` and a Python thread cannot be killed. The budget is enforced by
running each document in a single-worker pool and abandoning the pool on timeout; the pool is
rebuilt and the abandoned worker is given a grace period (`--doc-grace`, default 180 s) to
finish, so two documents are never deliberately in flight at once. If the grace period expires
anyway the event is counted as `run.overlap_events` in the manifest, and the honest reading is
that latency samples after that point are contaminated. `--doc-timeout 0` disables the
watchdog and calls inline, which is what the tests and `--dry-run` use.

### D-H2-8 · Only *outage* codes arm the SLM fallback

The task says: if a model is genuinely down, fall back per settings and record it. The
question is what "down" means, because the orchestrator flattens every SLM failure into one
warning shape, `slm_unavailable:<code>`.

The bake-off measured **0.5625 schema validity** for the shipped default
(`google/gemma-3-4b-it`, `docs/bakeoff/REPORT.md`). So a run of five consecutive
`PROVIDER_RESPONSE_INVALID` warnings has probability ≈ 1.6% *per window* on a perfectly
healthy model, and over 177 documents it is close to certain. Arming the fallback on it would
swap the model mid-sweep and silently change what the report is about.

`_SLM_OUTAGE_CODES` is therefore `{PROVIDER_UNAVAILABLE, PROVIDER_RATE_LIMITED,
PROVIDER_METHOD_NOT_SUPPORTED}`. `PROVIDER_RESPONSE_INVALID` is a *model quality* result and
is measured, not routed around.

**`PROVIDER_ERROR` is deliberately excluded too**, even though it is what a router `402
Payment Required` surfaces as. Two reasons: it is the catch-all code, so including it makes
the trigger unpredictable; and a 402 is an account-level credit exhaustion that the *fallback
model shares*, so swapping would spend the fallback's reputation on a problem it cannot fix.
The right response to a 402 is to re-run the sweep after credits reset, which D-H2-3 makes
cheap. The code counts are in the manifest under `run.warning_counts` so the rate is visible
either way.

### D-H2-9 · `--no-slm` refuses **every** remote method, not just the SLM

"Validators only" has to mean no remote traffic at all, or the Tier-2 VLM keeps calling the
router for escalated pages and the run is neither offline nor free. `SLMDisabledProvider`
therefore refuses `extract_fields`, `extract_fields_batch` **and** `ocr_page_heavy` with
`ProviderMethodNotSupported`, which the orchestrator absorbs as `slm_unavailable:` and
`tier2_failed:` warnings. Escalation *decisions* are unaffected — only their execution — so
escalation precision/recall stays comparable between an SLM run and a validators-only run.

### D-H2-10 · Document-level quality verdict is the **worst** page verdict

The harness scores one verdict per document; the envelope carries one per page. A document
with one unreadable page is not an `accept`, so the document verdict is the worst page's
(`bounce` > `repair` > `accept`) and the score is that page's score. With no pages at all — an
ingest rejection — `bounce_decision.decision` speaks for the document.

This is only obviously right while the corpus is one page per document, which it is. Stated in
the module docstring as a thing the tool does not handle.

### D-H2-11 · Per-page `:pN` timing keys are passed through unedited

`_Timings` publishes both `s4_ocr` (document total) and `s4_ocr:p1` (page 1's share), and the
pipeline docstring says the harness wants both. On a single-page corpus the two are identical,
which doubles the length of the latency table for no information. They are passed through
anyway: the driver's job is to transcribe what the pipeline reported, and a tool that quietly
drops half of a stage's telemetry because *it* judged the rows redundant is a tool you cannot
trust with the rows it kept. The duplication is called out in the report instead.

### D-H2-12 · `e2e_total` is added to `timings_ms`

`ProcessingStats.duration_ms` is the pipeline's own end-to-end measure and the PRD has an
end-to-end latency target, but the harness only percentiles what is inside `timings_ms`.
Adding it under a name no stage uses (`e2e_total`) puts the target's number in the same table
as its components, at the cost of one key that is not a stage. The driver's own wall clock is
**not** added — it would double-count and it belongs in the manifest, where the cold/warm split
lives.

### D-H2-13 · Latency is split three ways by model path, not two ways by cache

A re-run over a warm cache has a p50 near 1.5 s and a cold run near 12 s, and neither is "the"
latency, so the driver records the provider cache's hit/miss delta per document and splits the
percentiles by it.

The first cut of this had two buckets, cold and warm, keyed on `cache_misses == 0`. That is
wrong, and the first real run proved it: **30 of the 177 documents bounced at the quality
gate, so no model lookup happened at all**, and they landed in "warm" with a p50 of 140 ms.
Published unqualified, that reads as "the cache makes a document 10x faster" when it actually
says "the gate refused the page before any work started".

`DocRecord.model_path` is therefore three-valued — `live` (a lookup missed, so a request went
to the router and the wall clock includes it), `cached` (every lookup hit), `none` (no lookup
happened) — and the manifest carries `run.wall_ms_per_doc_by_model_path`. The report quotes
`live` against the PRD target, because production has no cache hit for a complaint it has
never seen.

### D-H2-14 · Predictions are committed under `docs/eval/predictions/`

~1.5 MB of JSON for the dev split. Committing it means the report can be reproduced — and
challenged — without a pipeline, a model directory or an HF token, which is the whole point of
a scorer that imports nothing from `app/`. The alternative (`runs/`, gitignored) makes
`REPORT.md` a claim rather than a result.

---

## 2. What the run does not measure

Carried into `docs/eval/REPORT.md` verbatim; repeated here so the decision file stands alone.

1. **Synthetic handwriting.** Font-rendered jitter is not handwriting. Every handwritten cell
   in the recognition table validates plumbing, routing and throughput, not accuracy. The real
   number needs `iiit-indic-hw-words` and real complaint pages (`corpus/index.json`
   `caveats.handwriting`, TEST-DATA-SPEC).
2. **The locked split is held out.** Variants 3 and 4 of every type were not scored and not
   run. The dev-split numbers are therefore an *optimistic* estimate to exactly the degree
   that anything in the build was tuned against variants 0-2 — and the quality gate was
   (`docs/decisions-wse1.md` §3 calibrated on the corpus), so treat the quality-gate row as
   the most optimistic in the report.
3. **Thin field types (Q6).** `receipt`, `email_printout`, `cheque` and `photo_evidence` emit
   empty `fields` dicts from their renderers, so those four types contribute field references
   only through line links, and `gt["fields"]` carries English spellings on Devanagari pages.
   Per-type field F1 for those four is thin by construction, not by failure.

---

## 3. Open / for the owner

Recorded as OPEN-QUESTIONS rows where they move an existing question; listed here with the
evidence.

- **Q6** — measured per type; see the report's per-type field table.
- **Q7** — the MT sentinel survival rate is now reported per run from real pipeline output
  rather than from a scratch probe, which is the WS-H follow-up D-F-15 asked for.
- **Q9 (new)** — `ProcessingStats.tokens_used` is always 0 under the laptop composite, and
  `JOB_TOKEN_BUDGET` is a per-instance ceiling that a batch caller must reset itself. See the
  OPEN-QUESTIONS row.
- **Q10 (new)** — the SLM default chosen by adjudication A-11 is the bake-off composite
  winner, but its schema-validity rate is what caps SLM field recall. Blocked on Q13.
- **Q11 (new)** — evidence bounding boxes are published in the post-repair (deskewed) frame
  and `rotation_applied_deg` is written but never read. Certain metric effect, possible
  product effect on the review UI.
- **Q12 (new, P0)** — the Devanagari recognition head is never selected. The largest defect
  the gate found; root cause, proof and four remedies in `docs/eval/REPORT-2026-08-08-slmlive.md` §2.
- **Q13 (new, owner action)** — the HF account's monthly inference credits are depleted, so
  the SLM and Tier-2 halves of the gate could not be measured. Blocks two PRD targets and Q10.

---

## 4. What this lane deliberately did not do

- **No fix for Q12**, even though it is one line of routing and the temptation was
  considerable. WS-H2 measures the shipped code; a gate that patches the thing it is grading
  publishes a number for code that never ran. It is written up with proof, root cause and four
  remedies so the owning lane can act in minutes.
- **No purchase of HF credits.** Q13 is an account-level spending decision and belongs to the
  owner.
- **No tuning of any threshold** between the first sweep and the published one. The four
  sweeps this report cost differ only in the driver's own reporting defects (D-H2-13, the
  `escalated` mapping, and a misplaced `return` in `DocRecord.as_json`); the pipeline was
  untouched throughout, and the recognition and detection tables are identical across every
  sweep that was scored (overall CER 0.5635, Deva 0.9042, Latn 0.4740 in each).
