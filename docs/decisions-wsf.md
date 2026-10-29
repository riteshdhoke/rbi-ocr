# WS-F decisions — pipeline, storage, API, webhooks, Celery

Judgment calls made while building `app/pipeline.py`, `app/storage/**`,
`app/api/**`, `app/webhooks.py`, `app/tasks.py`, `alembic/**`,
`docker-compose.yml` and their tests. Same conventions as the other decision
files: nothing here contradicts a PLAN.md invariant, and items marked **OWNER**
need a decision outside this lane.

WS-F composes; it does not judge. Every decision below is about *ordering,
degradation and persistence* — where a stage's judgement had to be turned into a
composition rule, or where a rule had no owner until something had to run the
stages in sequence.

---

## 1. The orchestrator

### D-F-1 · Stage 2's script vote is not called from the pipeline

`s4_ocr.run_tier1` already votes on the script and re-recognises once when the
vote disagrees with the head that produced the text (adjudication A-6, capped at
`MAX_RECOGNITION_PASSES = 2`). Calling `identify_script` again from the
orchestrator would either duplicate the vote or — worse — produce a *second*
decision that the recognition never acted on, which is exactly the kind of
"two answers to one question" the assembler's counters were designed to avoid.

The brief's step list names Stage 2 between bypass and OCR; it is there, one
level down, and the module docstring says so.

### D-F-2 · Validators run once per **document**, not once per page

The brief's order reads "per page: … s5 validators (thread pool)". `run_validators`
takes a *sequence* of pages and owns its own `ThreadPoolExecutor`, so a per-page
call would decompose one pool of N into N pools of one and lose the concurrency
the brief asked for. The pipeline therefore runs pages through quality → OCR →
LID → classification, then validators once over every non-bounced page, then the
router per page using that report.

This also fixes an ordering the per-page reading would have broken: the router
keys on **validator failures on that page** (invariant 3), which cannot be known
until the validators have seen the page.

### D-F-3 · `quality_score_low` is defined as "the gate did not accept this page as it arrived"

WS-E2 left this open (`docs/decisions-wse2.md` E2-U6): `EscalationSignals`
consumes a boolean, Stage 1.5 publishes a score whose only threshold is the
bounce cut, and a page that reaches the router is by definition above that cut —
so as shipped, nothing could ever set the signal true. The composer has to
define it.

**Definition:** `verdict is REPAIR`, or any flag in
`pipeline.QUALITY_DEFECT_FLAGS` (blur, noise, shadow, exposure, occlusion,
coverage, resolution, skew, junk text layer). `*_REPAIRED` flags are excluded
because the repair is already counted through the verdict.

Why this and not a second numeric threshold: the Q3 calibration
(`docs/decisions-wse1.md` §3) fixed one cut with evidence behind it, and adding a
"low but not bouncing" cut beside it would be a number nobody measured. The
verdict already partitions the corpus the way WS-E2 described — `clean` 45/45
accept, `scan` 33/50 repair, `heavy` 43/50, `photo` 48/50 — so the signal fires
on exactly the "degraded but recoverable" band §4b names, and it costs 1.0 of the
2.0 escalation threshold, i.e. it still never escalates a page on its own.

### D-F-4 · Structural anomalies are a closed set of three

`no_recognised_text`, `page_class_unknown`, `junk_text_layer`. Each is a fact
about *structure*, never a confidence number (invariant 3), and the router
saturates anomalies at three, so a longer list would be inert. A page that
produced no text at all is the clearest case for a second reader.

### D-F-5 · A `DocumentRejected` becomes a bounce envelope, not an exception

PRD P0-3 makes "refused, and here is the machine-readable reason" a **product
outcome**, and P1-2 generates the complainant's auto-reply from
`bounce_decision`. Raising instead would mean every caller re-implements the
mapping from `RejectionReason` to `BounceReason` that
`BounceDecision.from_rejection` already is.

Everything else propagates unchanged. `MalwareDetected` most of all: it is
deliberately not a `DocumentRejected` (WS-E1's audit) precisely so it cannot
reach a complainant-facing template, and converting it here would undo that.
`ResidencyViolation` propagates for the same reason it has no bypass flag.
`strict_rejections=True` restores the exception for a caller that wants it.

### D-F-6 · A one-page document keeps its own bounce reason

`ALL_PAGES_BOUNCED` exists and is used only when a multi-page document bounced
for *different* reasons. A single-page complaint that bounced on
`BLUR_ABOVE_CEILING` must say so — the auto-reply tells the complainant to
rescan, and "all pages bounced" tells them nothing they can act on.

### D-F-7 · A geometric repair discards the PDF text layer

`s1b_bypass` reads rectangles from the PDF as authored. If Stage 1.5 applied
`deskew` or `auto_invert`, those coordinates describe a page that no longer
exists. Using them would put every field's evidence box in the wrong place,
which is worse than the one OCR pass rasterising costs. `GEOMETRIC_REPAIRS` is
the named set; the fall-through is logged.

### D-F-8 · An unavailable SLM costs the free-text fields and nothing else

`extract_pages` failures are caught at `ProviderError` and recorded as a
`slm_unavailable:<code>` warning. Validators own every identifier, amount and
date by invariant 7 and have already run, so the envelope is still the useful
part of the answer. Making the document fail would mean an HF rate limit throws
away an extraction that never needed the SLM.

The same reasoning applies to Tier 2 (`tier2_failed:page=N:reason=…`), where
`escalate_page` already returns a typed outcome rather than raising.

Not absorbed: OCR and MT failures. A page nobody could read is not a document
with fewer fields, and `translate_fields` already has its own advisory-failure
path (`TRANSLATION_UNAVAILABLE`).

### D-F-9 · Version 2 is a re-assembly, never a patch of version 1

When Tier 2 lands, the pipeline re-runs the validators over the merged text and
re-runs the SLM **for the escalated pages only**, then assembles again. Tier 2
exists precisely to change what the page says, so keeping version 1's validator
hits would publish identifiers extracted from text the service has since decided
was wrong. The unescalated pages keep their raw SLM fields, so the re-run costs
one page's extraction, not the document's.

`assemble.py`'s docstring already required this ("Version 2 is produced by
re-assembling … not by patching version 1"); this is where it is honoured.

### D-F-10 · The fast path names its pending pages

`is_final=false` says *that* something is outstanding. `escalation_pending:page=N`
in `warnings` says *which*, so a consumer that acts on version 1 knows which
fields may still move. It is machine-readable for the same reason the grounding
rejections are.

### D-F-11 · A shadow-sampled page never has its lines replaced

`route()` already refuses to shadow-sample a page it escalated. The pipeline
completes the other half: a shadow page runs Tier 2, produces a `ShadowDiff`,
and keeps its Tier-1 lines and `escalated=False`. A control that changes the
output is not a control.

### D-F-12 · The synthetic-data declaration for an uploaded payload

PLAN.md invariant 6 gives the corpus loader a sidecar to check. An HTTP upload
has no sidecar, and under the laptop profile it is exactly the path by which a
real scan would reach a non-Indian endpoint.

`run_document(bytes, synthetic_declared=...)` requires a positive declaration
under a remote profile, surfaced by the API as an explicit `synthetic` form
field. This is the same *shape* as the sidecar — a submitter asserting
provenance — and not a bypass: there is no value of the field that lets an
**undeclared** payload through, and the check is skipped only on `cpu_node`,
which is allowed real complaints because that is its entire purpose.

A path source still goes through `require_synthetic_corpus_file` unchanged.

**OWNER:** the declaration is a *claim*. It stops an accident, not a determined
operator. If that is not enough, the remedy is to refuse uploads entirely under
the laptop profile — which would make the API untestable on this machine and is
therefore a deployment policy, not a code change.

### D-F-13 · Stage timings are keyed `stage` and `stage:pN`

`ProcessingStats.stage_timings_ms` is one flat mapping, so the key carries the
structure. `"s4_ocr"` is the document total and `"s4_ocr:p3"` is page 3's share.
The eval harness wants both: the total to compare documents, the split to find
the page that cost the p95.

---

## 2. OPEN-QUESTIONS resolved in this lane

### D-F-14 · **Q8 — RESOLVED (a).** `IFSC_MASTER_PATH` exists and the pipeline uses it

One field in `app/settings.py` (default: the committed Q4b fixture, so nothing
changes for tests or a fresh clone) plus `pipeline._load_ifsc_master`, which
loads it through `IFSCMaster.load` — the loader WS-E3 already built to take the
real RBI list as a path change (`docs/decisions-wse3.md` D-E3-8).

Two choices worth naming:

* **A missing or malformed master raises `ConfigurationError`.** Falling back to
  the synthetic fixture would silently downgrade an authoritative deployment to
  "structurally valid, not confirmed" on every IFSC in the country, and would do
  it quietly — the one failure mode a validator must not have.
* **The load is cached by resolved path.** The real list is ~170 000 rows; a
  per-document parse would dominate the pipeline.

Locked by `TestIFSCMasterSetting`, including a temporary CSV that demonstrates
the authoritative behaviour (unknown branch → `validator_passed=False` +
`IFSC_NOT_IN_MASTER`) and the refusal to fall back.

`docker-compose.yml` points `IFSC_MASTER_PATH` at a read-only `/data/ifsc` mount.

### D-F-15 · **Q7 — RESOLVED (a), and now measured rather than deferred**

Q7 asked whether the MT sentinel loss rate is material enough to justify
splitting long fields at sentence boundaries. It was recorded as "3 of 5 probe
sentences intact" from the D-E3-3 sweep and handed to WS-H to measure. The MT
weights are on this machine, so it was measured here instead.

**Method.** The shipped `mask_entities` / `restore_entities` and the shipped
`models/mt/indic-en` CT2 INT8 weights, greedy decoding. Two populations:

1. **Corpus-grounded** — every distinct Devanagari ground-truth line from
   variants 0-2 (the locked split was not touched), 91 lines, of which 54 carry
   at least one protected entity.
2. **Density probe** — 20 constructed Hindi complaint sentences carrying 1 to 5
   entities each (52 entities), because the corpus never puts more than two in
   one line and the question is specifically about long fields.

| population | fields | intact | field rate | entities | lost | per-entity loss |
|---|---|---|---|---|---|---|
| corpus Devanagari lines (1-2 entities) | 54 | 54 | **1.00** | 72 | 0 | **0.000** |
| density probe, 1-5 entities | 20 | 19 | **0.95** | 52 | 1 | **0.019** |
| — of which 4+ entities | 5 | 5 | 1.00 | 22 | 0 | 0.000 |

The single loss was a 3-entity sentence whose amount placeholder was dropped
(`ZQ0ZQ was deactivated from my account ZQ1ZQ .`).

**Ruling: (a) — accept and keep measuring; (b) is not warranted.** The concern
Q7 records was measured against the *bracket* sentinel that D-E3-3 replaced; with
the shipped `ZQ{n}ZQ` form the loss does not reproduce at 4+ entities in this
sample, and it is zero on the text the service actually processes. Splitting
fields at sentence boundaries would add a segmentation step — and a new class of
bug where a restored entity lands in the wrong segment — to recover a rate of
about two percent on synthetic long sentences.

**Caveats, stated rather than buried.** The sample is small (74 fields, 124
entities) and the corpus population tops out at two entities per line, so the
high-density row rests on constructed sentences. Line-level survival is a proxy
for field-level: in production the translated unit is an SLM-extracted span of
these lines, which is shorter and therefore easier, not harder. The measurement
is a probe script, not a committed tool — **WS-H** should fold it into the M7
eval so the rate is reported per run rather than re-derived by hand.

Nothing in Stage 6 changed. Its behaviour on loss was already correct: discard
the translation, flag `TRANSLATION_UNAVAILABLE`, keep `value` untouched.

---

## 3. Storage

### D-F-16 · Append-only is enforced in the ORM, not by a grant

`results` and `audit_events` inherit `AppendOnly`; mapper-level `before_update` /
`before_delete` listeners raise `AppendOnlyViolation`, and a session-level
`before_flush` guard catches the bulk paths mapper events cannot see.

It has to be the ORM because **SQLite has no per-table privileges at all**, and
SQLite is the default profile — the one the tests, the eval harness and the
laptop demo all run. A rule expressed as a Postgres `REVOKE` would simply not
exist where it is most often exercised. The migration's docstring recommends the
`REVOKE` as defence in depth for a Postgres deployment; it is a runbook item, not
something that can live in a migration that must also run on SQLite.

Raw SQL still bypasses this, and is meant to: the claim is that no code path *in
this service* can rewrite the trail by accident, not that a DBA cannot.

### D-F-17 · `webhook_deliveries` is deliberately **not** append-only

It is operational state, not evidence: `attempts` and `next_attempt_at` exist to
be updated. The transitions that matter (`webhook_delivered`,
`webhook_dead_lettered`, `webhook_replayed`) are recorded in `audit_events`,
which cannot be rewritten.

### D-F-18 · Result versions must be consecutive

`record_result` refuses anything but `latest + 1`. A gap means an envelope was
published and lost; accepting it would make the trail claim a continuity it does
not have. This is also what makes `finalise_document` idempotent under
`acks_late`: a redelivery fails loudly instead of appending a duplicate.

### D-F-19 · The envelope is stored as the JSON that was served

Not as normalised columns. A schema migration can then never silently rewrite a
published result, and a replay of a webhook is byte-identical.
`latest_envelope()` revalidates through the contract on the way out, so a drift
between what was written and what the models now accept surfaces at the read
rather than in a consumer's parser.

### D-F-20 · Relationships exist for flush ordering as much as for navigation

SQLAlchemy's unit of work orders inserts from inter-mapper **relationships**, not
from raw foreign keys. Without `Job.audit_events` / `Job.deliveries` the audit
row for a job creation was emitted *before* the job, and SQLite's
`PRAGMA foreign_keys=ON` (correctly) refused it. Found by the first smoke test;
recorded because the fix looks cosmetic and is not.

### D-F-21 · A relative SQLite path is anchored to the repository root

`Settings.database_url` rewrites `sqlite:///./data/runtime/app.sqlite` to an
absolute path. Otherwise the API and a Celery worker started from different
shells silently use two different databases — a failure that presents as
"the job exists but has no results".

### D-F-22 · `alembic.ini` lives in `alembic/`, not at the repository root

The root is shared by every workstream and a file there is a merge conflict
waiting to happen. Run with `-c alembic/alembic.ini`; `script_location = %(here)s`
makes it work from any directory. `sqlalchemy.url` is empty and `env.py` reads
`Settings.database_url`, so the migration and the service cannot disagree about
which database they mean and no password lands in a tracked file.

Verified: `alembic upgrade head` builds the schema on SQLite and
`alembic check` reports **no** difference against the models.

---

## 4. API

### D-F-23 · The residency gate runs at `create_app`, before the lifespan

`create_app` refuses a violating configuration **even when a provider was
injected**. If the gate only fired inside `get_provider`, every test that injects
a mock would bypass the one check this service cannot afford to get wrong, and
the refusal would be the single behaviour the suite never exercises.
`get_provider` still runs in the lifespan, so both the configuration and the
thing it builds are checked.

### D-F-24 · A submission-level `ResidencyViolation` is 422, not the taxonomy's 500

`ResidencyViolation.http_status` is 500 because the *configuration* form of it is
a deployment fault — and that form can never reach a request handler, because
`create_app` refuses to build the application at all. The only violation that
can arrive at a handler is a submission that failed to declare itself, which is
the caller's omission and is fixable by resubmitting. Answering it with 500
would tell a client to page an operator instead of reading the message.

### D-F-25 · `MalwareDetected` is answered without naming the signature

The context carries `signature` and `scanner` because the security queue needs
them; the response body carries neither. Telling a submitter which signature
fired is a free oracle for tuning the next attempt — the same reasoning that
keeps the class out of `DocumentRejected` (WS-E1's audit).

### D-F-26 · A bounced document is HTTP 200

The envelope *is* the product, and a bounce is an outcome with a
machine-readable reason that the auto-reply is generated from. A 4xx would make
a mail gateway treat a correctly-refused attachment as a transport failure and
retry it.

### D-F-27 · `/healthz` reports `degraded`, never `unhealthy`, for missing weights

The service still ingests, bounces, validates and answers without them — it just
cannot recognise. Reporting that as "down" would make an orchestrator restart a
container whose problem is an unmounted volume. The compose health check is
written to match.

### D-F-28 · `/metrics` is unauthenticated

Every series is a counter or gauge about the *service*, never about a document.
If a metric ever needs a document identifier as a label, it is the metric that is
wrong.

### D-F-29 · The PII filter is attached to loggers **and** their handlers

A filter on a logger sees only records logged through that logger — a record
propagated up from `app.stages.s4_ocr` never touches the root logger's own
filters. A filter on a *handler* sees every record that reaches it, including
propagated ones. Both are installed; `uvicorn`'s own loggers are covered
explicitly because it installs its own handlers.

The filter **drops** an `INFO` record carrying recognised complainant data
rather than masking it: "never log raw OCR text at INFO" is a rule about the
record, and a half-masked line still tells an analyst which document had a
14-digit account number on it. At `WARNING` and above the record is kept and
masked, because losing an error is worse than storing a masked identifier.

Any Indic-script run is treated as complainant text, because the only Indic
script this service handles is the content of a complaint and there is no
pattern for "a name".

---

## 5. Celery

### D-F-30 · `process_page` claims work; recognition happens in the fan-in

**This is the one place where the distributed path is not yet the real thing,
and it is deliberate.** Phase 0/1 keeps rasterised pixels in the process that
produced them — MinIO is provisioned in `docker-compose.yml` but nothing writes
a page store yet — so shipping a page image through the broker would mean
base64-ing a ~4 MB bitmap per page for parallelism this deployment cannot
currently use.

What **is** real and exercised: the chord shape, the four-queue routing,
`acks_late`, the idempotency key `(job_id, page_no, stage)`, the
`worker_process_init` model loading, and the finaliser's refusal to append a
duplicate version. What is deferred: the page tasks doing the recognition.

Closing it is one change — put the raster in object storage and hand the key to
`process_page` — and it is not made here because a page store that only exists in
a compose file cannot be tested on this machine.
`tests/unit/test_tasks_eager.py` asserts the chord and the synchronous pipeline
produce the **same envelope**, field for field, so the seam is pinned.

**OWNER / Phase 2:** the object store.

### D-F-31 · Models load in `worker_process_init`, never in the parent

ONNX Runtime and llama.cpp sessions are not fork-safe (BUILD-BRIEF sec. 8). A
provider built in the parent and inherited through `fork` shares arena state
between children, which corrupts inference under concurrency and presents as a
model that suddenly reads badly. Asserted behaviourally by firing the signal.

### D-F-32 · Retry policy lives in the delivery row, not in Celery

`deliver_webhook` returns rather than raising, and Celery's own retry is not
used. The attempt count, the backoff and the DLQ transition are all in
`webhook_deliveries`, so the in-process thread worker and the Celery task cannot
drift on the policy — and a failed delivery does not become a task failure that
triggers a second, uncounted retry.

### D-F-33 · `visibility_timeout` exceeds `CELERY_TASK_TIME_LIMIT`

Redis redelivers any message not acknowledged within the visibility timeout. A
timeout shorter than the longest task turns one slow document into two
documents.

---

## 6. Webhooks

### D-F-34 · The timestamp is inside the MAC

`HMAC-SHA256(secret, b"<timestamp>." + body)`. Signing the body alone would let a
captured delivery be replayed later with a fresh timestamp header. The receiver
rejects anything older than `WEBHOOK_SIGNATURE_TOLERANCE_SECONDS`;
`verify_signature` is exported so the consumer documentation can point at running
code rather than at a description of it.

### D-F-35 · The DLQ stores the signed body, not the envelope

A replay re-sends it byte-identically. Re-serialising would produce a different
body — key order, float formatting — and therefore a different MAC, so a receiver
deduplicating on the signature would treat a replay as a new event.

### D-F-36 · A URL without a secret is refused

An unsigned webhook is a result envelope posted to whatever answers that URL,
with no way for the receiver to tell it came from this service. `webhook_enabled`
requires both, and `publish` raises `ConfigurationError` when a URL is set
without a secret rather than silently sending in the clear.

### D-F-37 · A 4xx retries

Only 2xx is success. A receiver returning 400 on a valid envelope is far more
often a receiver mid-deploy than a permanently malformed body, and a genuinely
rejected delivery ends up in the DLQ either way — where it is visible, rather
than dropped on the first response.

---

## 7. Tests

| File | Tests | Notes |
|---|---|---|
| `tests/unit/test_pipeline.py` | 38 | Three corpus documents (clean, bounce, escalating) plus the residency, Q8 and degradation paths. Envelopes validated against the **committed** JSON Schema. Includes the one `@pytest.mark.local_models` test. |
| `tests/unit/test_storage.py` | 43 | Append-only through both enforcement paths, idempotency, soft duplicates, version continuity, DLQ transitions, dual-mode URL handling, migration/model agreement. |
| `tests/unit/test_api.py` | 41 | `TestClient` over the real ASGI app including its lifespan: upload, idempotent resubmit, soft duplicate, healthz, metrics, residency refusal both ways, PII filter. |
| `tests/unit/test_webhooks.py` | 29 | Signature verifiable from the receiver's side, backoff, DLQ, byte-identical replay, thread mode. |
| `tests/unit/test_tasks_eager.py` | 14 | Queue topology, eager chord, and the equality assertion against the synchronous pipeline. |

Full suite after this run: **1970 passed**, 6 failed — all six pre-existing and
in another lane (see §8).

### D-F-38 · The JSON Schema checker is imported, not re-implemented

`tests/unit/test_pipeline.py` imports `validate_against_schema` from
`test_assemble`. WS-E3 recorded the decision (D-E3-5) not to add a runtime
`jsonschema` dependency to assert a property of our own generator; a second
hand-rolled copy of that checker is precisely what that decision was avoiding.

### D-F-39 · Shadow sampling is disabled in every test that is not about it

`ESCALATION_SHADOW_RATE=0.0` in the shared settings fixtures. At the default 3%,
a test asserting "this document did not escalate" fails on roughly one run in
thirty-three, on a different document each time — the worst possible flake.
`TestShadowSampling` sets it to 1.0 and asserts the behaviour directly.

### D-F-40 · The `local_models` test uses `LocalProvider`, not the laptop composite

The composite routes `extract_fields` and `ocr_page_heavy` to the HF router, and
the test must not open a socket. `LocalProvider` raises
`ProviderMethodNotSupported` for both, which the pipeline absorbs into an
`slm_unavailable` warning — so what the test asserts is exactly the Tier-1 path
on real weights: ingest, quality, batched recognition, LID, classification,
validators, assembly.

**Measured on this machine** (`complaint_form_en_v0_clean.jpg`, real
PP-OCRv5 ONNX + `lid.176.ftz` + the committed fusion weights):

| | |
|---|---|
| lines recognised | 25 |
| tier / script / language | `tier1` / `Latn` / `eng` (1.00) |
| page class | `complaint_form`, confidence 1.00 |
| fields extracted | `bank_name` (lookup, pass), `ifsc_code` `SBIN0904158` (pass), `account_number` `61136570711` (pass), `amount` `45,000` (pass), `transaction_date` `07/07/2026` (pass), `utr_rrn` `113699585184` (pass) |
| warnings | `slm_unavailable:PROVIDER_METHOD_NOT_SUPPORTED` (expected: no local SLM) |
| stage timings (ms) | ingest 77, quality 220, **OCR 1062**, LID 26, classify 304, validators 7, assemble <1 |

---

## 8. Unresolved / handed on

| # | Item | State |
|---|---|---|
| F-U1 | **Six pre-existing failures in another lane.** `tests/unit/test_providers_local.py` ×5 and `tests/unit/test_s4_ocr.py` ×1 fail with `AttributeError: '_FakeSession' object has no attribute 'get_outputs'`. The test double predates the `class_axis` reconciliation added at `app/providers/local.py:1141`; the *product* code is correct (the real-provider run above exercises exactly that path). One method on the fake fixes all six. | **WS-C lane.** Present before this run; not touched. |
| F-U2 | `process_page` / `escalate_page_task` claim work without doing it (D-F-30). Needs the object store. | Phase 2. |
| F-U3 | `docker-compose.yml` references a `Dockerfile` that does not exist. | **WS-I packaging.** |
| F-U4 | No authentication on any endpoint. The service sits behind the RBI gateway; adding a token without knowing which one is worse than not adding one. | **OWNER.** |
| F-U5 | The synthetic-data declaration is a claim, not a proof (D-F-12). | **OWNER**, deployment policy. |
| F-U6 | Q7's probe is a scratch script, not a committed tool. The numbers in §2 should be reproduced per run. | **WS-H**, fold into the M7 eval. |
| F-U7 | `MAX_WALL_CLOCK_SECONDS` is enforced inside Stage 1 only; the orchestrator does not re-check it between stages. A document that ingests fast and recognises slowly can exceed it. | Needs a deadline object threaded through the stages; deferred rather than half-done. |
| F-U8 | Nothing here has been run against Postgres. The models and every query are dialect-neutral and `alembic check` is clean on SQLite, but "clean on SQLite" is not "runs on Postgres". | First deployment. |
| F-U9 | `s5_slm`'s per-page `masked_spans` come from the validator report of the *current* pass. After Tier 2 the spans are recomputed, which is correct, but a page whose Tier-2 text moved a validator hit will send a slightly different prompt on the re-run. Deterministic, documented, not a defect — recorded so it is not rediscovered as one. | Noted. |
