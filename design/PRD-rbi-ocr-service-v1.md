# PRD — RBI Complaint Document Intelligence Service

**Version** 1.0 · **Date** 2026-08-07 · **Status** Ready for build
**Supersedes** Architecture spec v1 · **Companion** `design-review-v2.html`, `TEST-DATA-SPEC.md`, `BUILD-BRIEF.md`

---

## Problem Statement

Complaints against banks under the RBI Integrated Ombudsman Scheme arrive as email attachments — scanned forms, handwritten letters, phone photos of receipts — in English and a dozen Indian languages, mixed within single documents. Every one is currently opened, read, and keyed into a case system by a human, at a per-document cost that scales linearly with volume and a turnaround measured in days.

The cost of not solving it is not primarily labour. It is **inconsistency**: two case officers key the same complaint differently, identifiers get transposed, and regional-language complaints wait longer than English ones for no reason other than reviewer availability. That asymmetry is itself a fairness problem in a grievance-redress process.

**Constraint that shapes everything:** complainant data cannot leave Indian infrastructure or be sent to a third-party model API. Every design decision downstream of that — small local models, CPU-first, ONNX quantisation — follows from it rather than from cost optimisation.

---

## Goals

1. **Cut manual keying effort by 70%** on the primary-complaint path — measured as case-officer minutes per complaint, before vs. after, on a matched sample.
2. **Close the language turnaround gap to under 15%** — median time-to-triage for a regional-language complaint within 15% of the English median. This is the fairness goal and it is the one worth defending in a review.
3. **Zero silent identifier errors.** Any account number, IFSC, card fragment, UTR or amount that reaches the case system is either checksum-validated or flagged for human confirmation. Never a confident wrong digit.
4. **Every extracted value is attributable** — page, bounding box, source mechanism — so any figure can be traced to the pixels it came from in one click.
5. **Run entirely on in-country CPU infrastructure** at launch, with a GPU path that is a configuration change rather than a re-architecture.

---

## Non-Goals

1. **Production-grade handwritten Indic accuracy.** No open model reaches it and we will not pretend otherwise. Handwritten pages route to human review by design; that is an SLA statement, not a hidden error rate. Revisited only when the Phase 3 fine-tune produces evidence.
2. **Any judgement about a complaint's merit.** The service extracts and attributes. Categorisation is limited to the closed RB-IOS ground list. No scoring, prioritisation, or recommended disposition — that boundary is what makes it deployable inside a regulated adjudication.
3. **Abstractive summarisation.** Summaries are extractive spans only. A generated sentence that subtly misstates a grievance is worse than no summary and cannot be pointed at.
4. **Scripts without a shipped recognition head** — Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Assamese. Each is a fine-tuning project, prioritised by the week-1 volume profile, not assumed into v1.
5. **A complainant-facing UI.** The service is an API plus a reviewer queue. Anything the complainant sees belongs to the RBI CMS portal team.

---

## Target Users

| Persona | Volume | What they need |
|---|---|---|
| **Case officer** (Ombudsman office) | Primary, daily | A pre-filled case record they can verify in under 2 minutes instead of keying in 12 |
| **Review queue operator** | Primary, daily | A fast correction UI for flagged fields, with the source image beside the value |
| **Integration engineer** (RBI CMS / bank systems) | Secondary | A stable, versioned JSON contract they can build against without re-reading it every release |
| **Bank InfoSec / procurement reviewer** | Gate, once per deployment | Evidence of data residency, dependency licensing, audit trail, retention |
| **ML engineer** (internal) | Continuous | Golden-set evaluation, drift signals, a retraining loop fed by real corrections |

---

## User Stories

**Case officer**
- As a case officer, I want a complaint's identifiers pre-extracted and checksum-validated so that I verify rather than transcribe.
- As a case officer, I want to click any extracted value and see the exact region of the scan it came from so that I can confirm it without opening the PDF separately.
- As a case officer, I want regional-language complaints to arrive with English values alongside the original text so that I am not blocked on a colleague who reads Marathi.
- As a case officer, I want the complaint summary to be verbatim lines from the letter so that I can quote it into the case record without paraphrase risk.
- *(Edge)* As a case officer, when a scan is too poor to read, I want the system to have already asked the complainant for a re-scan rather than handing me a page of garbage.

**Review queue operator**
- As a review operator, I want only genuinely uncertain fields flagged so that the queue is short enough to actually clear.
- As a review operator, I want my correction to be stored as a training example so that the same mistake gets rarer.

**Integration engineer**
- As an integration engineer, I want a versioned JSON schema with an explicit `is_final` flag so that I can consume a fast partial result and reconcile the final one.
- As an integration engineer, I want every field to declare whether it came from a validator, a lookup, a model, or a human so that I can set my own trust thresholds.

**InfoSec reviewer**
- As an InfoSec reviewer, I want proof that no complainant data reaches an external endpoint, enforced in code rather than asserted in a policy.
- As an InfoSec reviewer, I want an append-only audit trail of every value and every mutation.

**ML engineer**
- As an ML engineer, I want a reproducible labelled corpus in version control so that a model change produces a comparable number.
- As an ML engineer, I want a sample of *passed* pages independently re-processed by the escalation model so that I can measure what the router is missing.

---

## Requirements

### P0 — Must have

**P0-1 · Provider abstraction with three profiles**
All model access goes through one `ModelProvider` interface. Profiles: `laptop` (SLM/VLM via HF API, OCR and MT local), `cpu_node` (all local), `gpu_node` (SLM/VLM on vLLM).
- [ ] Switching profile is a single env var; no call-site changes anywhere in the pipeline
- [ ] Identical contract tests pass against all three profiles
- [ ] Given `PROVIDER_PROFILE=laptop` and `SYNTHETIC_ONLY=false`, when the app starts, then it raises `ResidencyViolation` and refuses to serve
- [ ] Corpus loader rejects any file without a `synthetic: true` sidecar while in `laptop` profile
- [ ] `provider_profile` exposed as a Prometheus gauge

**P0-2 · Ingestion with correct idempotency**
- [ ] Idempotency key is `sha256` of attachment bytes, not `Message-ID`
- [ ] Soft-duplicate detection on (sender, normalised subject, calendar day)
- [ ] Given the same attachment arrives twice via different gateways, when both are ingested, then one job is created and the second returns the existing `job_id`
- [ ] Hard resource ceilings enforced at the sandbox: page cap (default 200), total pixel cap, decompression ratio cap, wall-clock cap. Exceeding any produces a typed rejection, never an OOM.
- [ ] ClamAV scan; archives unpacked with path-traversal protection; PDF JavaScript stripped

**P0-3 · Quality gate (Stage 1.5)**
Every page scored before any model runs, with an accept / repair / bounce decision.
- [ ] Scored on: effective resolution, skew angle, blur (variance of Laplacian), text coverage, occlusion, rotation, blankness, duplication
- [ ] Given a page at 15° skew, when the gate runs, then it is repaired by deskew and accepted
- [ ] Given a page below the readability floor, when the gate runs, then it is bounced with a machine-readable reason code and an auto-reply is queued naming the specific problem
- [ ] `quality_score` and `quality_flags` present on every page record, including accepted ones
- [ ] Bounce rate exposed as a metric; a spike means an upstream change, not a model regression

**P0-4 · Digital-PDF bypass**
- [ ] Pages with a usable text layer skip OCR entirely and are marked `ocr.tier = "bypass"`
- [ ] Heuristic guards against the "text layer is junk from a prior bad OCR" case — cross-check character-class distribution against the detected script

**P0-5 · Line-level language identification**
- [ ] `script` and `language` are attributes of the OCR **line**, not the page
- [ ] Page-level `primary_language` exists only as a derived summary with a distribution
- [ ] `is_code_mixed` set when ≥15% of lines differ in script from the page majority
- [ ] Language label set is constrained by detected script (Devanagari → hin/mar/kok/nep)
- [ ] Given a printed English form with a handwritten Marathi free-text block, when processed, then the English lines carry `eng` and the Marathi lines carry `mar`, and only the Marathi field values are translated

**P0-6 · Two-tier OCR with an evidence-based router**
- [ ] Tier 1 PP-OCRv5 with script-routed recognition heads; recognition runs **batched** (16–32 line crops per call), never looped
- [ ] Escalation triggers on validator failure, structural signals, handwriting flag, script disagreement, or quality score — **not** on raw CTC confidence alone
- [ ] Shadow sampler: a configurable 3% of *passed* pages also run through Tier 2, with outputs diffed and the delta recorded
- [ ] Escalation precision and recall reported on the golden set as first-class metrics
- [ ] Tier 2 runs in an isolated worker pool with its own SLA and never blocks the fast path

**P0-7 · Deterministic extraction for all identifiers**
- [ ] Account number, IFSC, card fragment, UTR/RRN, MICR, amounts and dates extracted by regex + checksum, never by the SLM
- [ ] IFSC validated structurally (4 alpha + `0` + 6 alnum) **and** cross-checked against the RBI master list
- [ ] `bank_name` requires both a fuzzy-match score above floor **and** agreement with the IFSC bank-code prefix; disagreement flags for review rather than picking one
- [ ] Indic numerals normalised to ASCII before amount parsing; lakh/crore multipliers handled explicitly
- [ ] Given an account number where the checksum fails, when assembled, then the field is emitted with `validator_passed: false` and flagged for review — never silently corrected

**P0-8 · Constrained SLM extraction, extractive summary**
- [ ] SLM handles only open-ended fields: complainant name/address, relief sought, summary span selection, prior-complaint reference
- [ ] Output is schema-constrained (GBNF locally, JSON mode + Pydantic + one repair retry via API); schema-validity rate recorded as a metric
- [ ] `complaint_summary` is produced by the model **selecting line indices**; the assembler concatenates verbatim OCR text. The model never composes prose.
- [ ] Grounding gate: any SLM value not present in the page OCR text within edit distance 2 is rejected and the field flagged
- [ ] Temperature 0, hard `max_tokens` cap, per-job token budget

**P0-9 · Per-field translation with entity masking**
- [ ] Only extracted field values are translated, never whole pages
- [ ] Deterministic-extractor hits are replaced with sentinels before translation and restored after
- [ ] `value` (native) is immutable and authoritative; `value_en` carries `mt_engine` and is marked advisory
- [ ] Given a Marathi complaint containing ₹45,000 and an account number, when translated, then both appear byte-identical in `value_en`

**P0-10 · Versioned result contract**
- [ ] `result_version` integer and `is_final` boolean on every envelope
- [ ] Version 1 emitted at the fast-path SLA without waiting for escalated pages; subsequent versions merge them in
- [ ] Webhook fires on every version increment, HMAC-signed, with replayable DLQ
- [ ] Every field carries `source` ∈ {regex_validator, lookup_table, slm, classifier, human} and an `evidence` block with page and bbox
- [ ] Schema published as JSON Schema; contract tests in CI; breaking changes require a major version

**P0-11 · Parallelism and thread discipline**
- [ ] Document → page fan-out via Celery chord
- [ ] Batched recognition (L2) implemented and benchmarked against the looped baseline
- [ ] Deterministic extractors run concurrently; SLM field groups issue concurrently in API profile
- [ ] `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set before any numeric import; `intra_op_num_threads = 2` in all profiles
- [ ] CI asserts `celery_concurrency × intra_op_threads ≤ physical_cores`

**P0-12 · Synthetic test corpus in version control**
- [ ] Generator produces 10 document types × 5 variants × degradation matrix, deterministic under a fixed seed
- [ ] Every page emits a `.gt.json` sidecar with per-line ground truth and document-level field ground truth
- [ ] Adversarial set included: corrupt PDF, oversized PDF, zip bomb, zero-byte, wrong MIME, password-protected, blank, inverted, non-document
- [ ] Evaluation harness computes CER/WER by script and modality, field F1, schema validity, escalation precision/recall

**P0-13 · Licence gate**
- [ ] CI fails on any dependency outside {Apache-2.0, MIT, BSD-2, BSD-3, MPL-2.0}, including transitive
- [ ] No model weights under non-commercial or copyleft terms enter the build (this excludes Surya; MinerU requires clearance)

### P1 — Should have

- **P1-1 · Human review UI** — flagged fields beside the source crop, keyboard-first, correction written to the active-learning store
- **P1-2 · Auto-reply bounce loop** — templated, bilingual, naming the specific defect ("the account number region is cut off")
- **P1-3 · Supporting-document tagging** — closed tag set, no field extraction
- **P1-4 · Complaint categorisation** against the RB-IOS ground list via a distilled encoder classifier rather than the SLM
- **P1-5 · Prompt/response cache** keyed on `sha256(model + prompt + params)` — a dev-velocity feature that pays for itself in Phase 0 alone
- **P1-6 · Drift monitoring** — mean OCR confidence, escalation rate, bounce rate, HITL rate, per-script CER on a rotating shadow sample

### P2 — Future considerations (design for, do not build)

- **P2-1 · Additional script heads** — keep the recognition head a runtime-selected, LRU-cached artefact so adding Kannada is a model drop, not a code change
- **P2-2 · GPU serving** — the provider interface already accommodates it; do not let any call site assume synchronous local inference
- **P2-3 · Bhashini integration** as a government-operated translation/OCR fallback that may satisfy residency where a commercial API cannot
- **P2-4 · Cross-complaint linking** on normalised identifiers — keep `normalized` stable and indexed so this stays cheap later
- **P2-5 · Upstream pre-submission validation** in the RBI CMS portal — strictly better than any downstream gate, entirely outside our control

---

## Models, licences and provenance

### The models

| Role | Artefact | Publisher | Licence |
|---|---|---|---|
| Text detection | PP-OCRv5 mobile det (ONNX), `models/ocr/det.onnx`, 4.83 MB | Baidu / PaddlePaddle, via the official `PaddlePaddle` org on Hugging Face | Apache-2.0 |
| Latin recognition | `en_PP-OCRv5_mobile_rec` (ONNX), `models/ocr/rec_latn.onnx`, 7.85 MB | same | Apache-2.0 |
| Devanagari recognition | `devanagari_PP-OCRv5_mobile_rec` (ONNX), `models/ocr/rec_deva.onnx`, 7.91 MB | same | Apache-2.0 |
| Language identification | fastText `lid.176.ftz`, 938 KB, 176 languages | Meta AI, from Meta's own distribution point (dl.fbaipublicfiles.com), not Hugging Face | CC-BY-SA-3.0 |
| Indic→English translation | IndicTrans2 distilled 200M, CTranslate2 format, ~808 MB fp32 on disk, INT8 at load | AI4Bharat model, pre-converted build published by `adalat-ai` on Hugging Face, pinned revision `524aade4f724dccbc6cd5ffb5e3c5c832a2baee4` | MIT |
| Page-class fusion | `models/classifier/fusion.json` — fitted coefficients, not a neural network | produced by this project | project-owned |

**Total on disk:** 837 MB across 15 files.

### Provenance and the licence flag

- `tools/fetch_models.py` writes `models/PROVENANCE.json` recording source URL, resolved revision, SHA-256 and declared licence for every file.
- Every artefact is SHA-256 pinned; `docs/AIR-GAPPED-DEPLOY.md` section 5.1 carries the table of hashes and 5.4 the verification command.
- **The one licence that needs a procurement decision:** `lid.176.ftz` is CC-BY-SA-3.0, which sits outside the project's code-dependency licence allowlist (Apache-2.0/BSD/ISC/MIT/MPL-2.0/PSF). That allowlist governs code; CC-BY-SA is neither GPL/AGPL nor non-commercial, so the downloader fetches it and flags it loudly rather than silently. Attribution and share-alike apply to the model file. `--strict-licence` refuses it. Confirm with the legal owner before production.
- No model weights are in git. They are fetched once on a connected machine and carried to the air-gapped node, or shipped in the models+corpus hand-off bundle.

### On PP-OCR being Chinese in origin — what the actual risk is

PP-OCR comes from Baidu and that is worth stating plainly rather than burying. What matters for a security review is the class of model, not its country of origin:

- **Discriminative, not generative.** PP-OCR maps pixels to a character sequence over a fixed output alphabet — 438 classes for the Latin head, 570 for the Devanagari one, enumerated in the `rec_*_dict.txt` files shipped beside the weights and readable in a text editor. It has no capacity to emit executable content, no tool use, no network access, and no notion of an instruction to follow. Prompt injection, data exfiltration and code execution — the risks people mean when they say "backdoor" about an AI model — are not available to it. The worst it can do is return wrong text.
- **Offline by construction.** It runs through ONNX Runtime on an air-gapped node. There is no callback path for anything to be exfiltrated to.
- **Apache-2.0.** The weights and the graph may be inspected, redistributed and audited.

The residual risks, stated honestly because a reviewer will ask and an absolute claim invites challenge:

1. **A poisoned model could deliberately misread.** A trojaned OCR model that mis-transcribes a specific pattern on a trigger is a documented attack class against discriminative models. What bounds it here: identifiers do not survive on OCR alone — IFSC codes are checksum-validated and looked up against a master list, account numbers and amounts go through deterministic validators, every field carries an evidence block with page and bounding box a human can open, and the committed corpus gives a measured character-error baseline that a swapped model would move.
2. **The model file is data parsed by a runtime.** Any ONNX file is input to ONNX Runtime, so a malicious graph is a parser-exploit question rather than a model-behaviour one. What bounds it: artefacts are SHA-256 pinned, fetched once from the publisher's official organisation, re-verified from the bundle manifest on arrival, and the shipped graphs contain no custom operators.
3. **Supply chain.** Mitigated by the pinned revision plus hash, and by the fact that the node never fetches anything at run time.

The country of origin does not create a code-execution path here. The meaningful question for a discriminative model is whether its output can be trusted, and that is answered by the validators, the evidence trail and the measured corpus baseline — not by provenance alone.

---

## Success Metrics

### Leading (days to weeks after launch)

| Metric | Success | Stretch | Method |
|---|---|---|---|
| Case-officer minutes per complaint | −50% | −70% | Matched-sample time study, 100 complaints, before/after |
| Field-level F1, deterministic identifiers | ≥ 0.97 | ≥ 0.99 | Golden set, weekly |
| Field-level F1, SLM fields | ≥ 0.80 | ≥ 0.88 | Golden set, weekly |
| JSON schema validity rate | ≥ 99.5% | 100% | Prometheus counter, continuous |
| Silent identifier errors | **0** | **0** | Every unvalidated identifier must be flagged; any that reaches the case system unflagged and wrong is a P1 incident |
| Bounce rate (quality gate) | 5–15% | — | Outside that band means the gate is miscalibrated in one direction or the other |
| Escalation rate | ≤ 15% | ≤ 10% | Prometheus, continuous |
| Router false-negative rate | ≤ 8% | ≤ 4% | Shadow sampler diff, weekly |
| p95 end-to-end, 5-page complaint, fast path | ≤ 90 s | ≤ 45 s | Trace histogram |
| Throughput per 16-vCPU node | ≥ 60 pages/min | ≥ 100 | Load test |

### Lagging (weeks to months)

| Metric | Success | Method |
|---|---|---|
| Median triage time, regional-language vs. English | Gap ≤ 15% | Case-system timestamps, monthly cohort |
| HITL rate | ≤ 12% of documents | Review-queue volume ÷ documents |
| Re-keying incidents (officer overrides an extracted value) | ≤ 5% of fields | Correction log |
| Cost per 1,000 pages | Within infra budget | Node-hours ÷ pages |
| InfoSec review pass | First submission | Binary |

**Evaluation cadence:** golden-set run on every merge to main; full metric review at 1 week, 1 month, and 1 quarter post-launch.

---

## Timeline & Phasing

### Phase 0 — Laptop-first proof (3 weeks) · *new*

No GPU, no real complainant data, no service code on day one.

| Week | Work | Gate |
|---|---|---|
| 1 | **Throughput spike** — PP-OCRv5 INT8 on 100 pages on the target instance type. **Volume profile** — language and file-type distribution across 1,000 real mailbox items, read-only. **Licence clearance** on the full model list. | Go/no-go on CPU-first. If p95 > 4 s/page, the hardware plan changes now, not in week 19. |
| 2 | Synthetic corpus generator; 500 labelled pages committed. Evaluation harness. `ModelProvider` interface + `laptop` profile with HF API. | Corpus reproducible from seed; harness produces CER and field F1. |
| 3 | End-to-end vertical slice on synthetic English: ingest → gate → OCR → validators → SLM via API → JSON. SLM bake-off across 3 candidate models using the HF API. | A JSON envelope from an emailed PDF, on a laptop, with no GPU. |

**Exit:** the three low-confidence assumptions from the design review have numbers attached. Model shortlist is evidence-based. Nobody has bought a GPU.

### Phase 1 — Real data, real languages (6 weeks)
Fork `docling-api` as the service skeleton. Celery/Redis/Postgres. All six stages plus the quality gate. Devanagari added. Golden set extended to 1,500 pages, of which **200 are real** — including 50 real handwritten pages, because that is the only honest measurement of the handwriting gap. Result contract frozen at 1.0.0.
**Exit:** English + Hindi + Marathi end to end on real mail; deterministic F1 ≥ 0.95; contract published; handwriting gap *quantified*.

### Phase 2 — Integration & review loop (5 weeks)
Tamil, Telugu, Urdu. Review UI. Bounce auto-reply. Categorisation classifier. Shadow sampler live. Bhashini evaluation spike.
**Exit:** HITL loop closed with corrections reaching the training store; escalation precision/recall reported.

### Phase 3 — CPU optimisation (6 weeks)
Static INT8 PTQ with a real calibration set. Batched recognition tuned. Thread pinning and cpuset isolation. LRU head cache. Indic HTR fine-tune on `iiit-indic-hw-words`. Load test to 3× peak.
**Exit:** throughput and latency targets met with **no accuracy regression beyond the quantisation gates** (CER +0.5 abs, field F1 −1.0 abs).

### Phase 4 — Production & GPU (8 weeks)
vLLM for SLM and VLM tiers. Escalation threshold lowered as headroom allows. Full observability, DR, blue/green model deploys with automatic rollback, retention automation, pen test, InfoSec review.
**Exit:** 99.5% availability over 30 days; all launch metrics at success threshold; runbooks signed off.

### Hard dependencies
- RBI IFSC master list access — **blocking P0-7**, needed by Phase 1 week 1
- Mailbox read access for the volume profile — **blocking Phase 0 week 1**
- RB-IOS ground taxonomy, confirmed and versioned — blocking P1-4
- `iiit-indic-hw-words` licence terms confirmed — blocking Phase 3 fine-tune

---

## Open Questions

**Blocking**
1. **What is the actual language and script distribution of incoming complaints?** *(Data — week 1)* If >15% falls outside Devanagari/Tamil/Telugu/Arabic, the Phase 3 fine-tune backlog changes and possibly the phase order does too.
2. **What fraction of attachments carry a usable text layer?** *(Data — week 1)* Drives the throughput model directly; 30% vs 5% is a different hardware bill.
3. **Is a bounce/auto-reply to the complainant permissible under RB-IOS process rules?** *(Legal / RBI process owner)* The quality gate's highest-value path depends on it. If not permitted, the gate degrades to flag-only and we lose the best accuracy lever we have.
4. **Retention period for raw scans?** *(Legal / bank contracts)* Determines storage sizing and the erasure cascade.

**Non-blocking**
5. Does the chosen HF inference provider expose grammar-constrained decoding, or only JSON mode? *(Engineering — Phase 0 week 3)* Changes the schema-validity baseline we carry into local serving.
6. Can `iiit-indic-hw-words` be used for a commercial fine-tune, or research only? *(Legal — before Phase 3)*
7. Should the review UI be built or bought (Label Studio vs. in-house)? *(Engineering / design — Phase 2)*
8. Is Bhashini's residency posture acceptable to bank InfoSec? *(InfoSec — Phase 2 spike)*
9. What is the escalation-rate ceiling the infra budget can absorb once GPU is available? *(Finance / engineering — Phase 4)*
