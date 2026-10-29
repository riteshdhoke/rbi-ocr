# TUNING.md — operating, tuning and debugging the RBI complaint OCR service

**Audience:** an engineer who has the code and has never run it. No prior knowledge of this
project is assumed. Every term is explained the first time it appears, every command is
copy-pasteable, and every command shows what a correct run prints.

**Companion documents.** `docs/FUNCTIONAL-SPEC.md` explains *what the product does* in plain
language — read that first if you want the shape before the knobs. `docs/API.md` is the
consumer-facing HTTP reference. `docs/AIR-GAPPED-DEPLOY.md` is how to install this on a machine
with no internet. This file is about **changing behaviour after it is installed**.

---

## Contents

1. [Before you touch anything](#1-before-you-touch-anything)
2. [The key classes map](#2-the-key-classes-map)
3. [Where configuration comes from](#3-where-configuration-comes-from)
4. [Every tunable](#4-every-tunable)
5. [The failure → remedy playbook](#5-the-failure--remedy-playbook)
6. [Re-running every calibration and evaluation tool](#6-re-running-every-calibration-and-evaluation-tool)
7. [Change control — what to re-measure after moving a knob](#7-change-control--what-to-re-measure-after-moving-a-knob)

---

## 1. Before you touch anything

### 1.1 What this service is, in four sentences

It takes a complaint document that arrived as an email attachment — a scanned PDF, a phone
photo, or a digital PDF — and returns a **structured JSON result** naming the fields it found
(account number, IFSC code, amount, dates, the complainant's name and what they are asking
for), with a pointer back to the exact place on the page each value came from. Documents too
damaged to read are **refused** with a machine-readable reason and a bilingual reply the mail
system can send back. Pages the service is unsure about are **escalated** to a heavier model,
and fields that fail a check are **flagged for a human** rather than silently corrected.

Two words you will see constantly:

- **OCR** — Optical Character Recognition, turning pixels into text.
- **SLM** — Small Language Model. A language model used here **only to pick which existing
  lines of text answer a question**, never to write prose. It is the one component that can be
  remote; everything else runs locally.

### 1.2 Run everything through the project's own Python

There is a virtual environment (a private, self-contained Python installation) at `.venv`.
Using the system `python` will import the wrong libraries or none at all.

**Step 1 — confirm the interpreter exists and is the right version.**

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe --version
```

Expected output:

```
Python 3.12.4
```

> **Troubleshooting.** `The system cannot find the path specified` means the venv was never
> created or the repository lives somewhere else. Create it with
> `C:\Python312\python.exe -m venv .venv` from the repository root, then
> `.venv\Scripts\python.exe -m pip install -r requirements.txt`. On a corporate network that
> intercepts TLS, pip will fail with `SSLError` — see §5.9.

**Step 2 — always set the output encoding on Windows before running anything that prints
Devanagari.**

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

There is no output. Without it, any command that prints Hindi or Marathi text dies with
`UnicodeEncodeError: 'charmap' codec can't encode character`. Every code path in this project
already writes files as UTF-8; this variable is only about the *console*.

**Step 3 — confirm the service can read its own configuration.**

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from app.settings import get_settings; print(get_settings().startup_banner())"
```

Expected output (yours will differ in the model name if you have changed it):

```
profile=laptop synthetic_only=true slm=google/gemma-3-4b-it hf_token=set threads=2x2
```

Read it as: running the *laptop* profile, refusing to process anything but synthetic test data,
using that model for extraction, a remote token is configured, and two worker processes × two
inference threads each. **The token value is never printed by anything in this project** — only
`set` or `unset`.

### 1.3 The rules this system will not let you tune away

These are enforced in code and asserted by tests. If a change you are making seems to require
breaking one, you are solving the wrong problem.

| # | Rule | Where it lives |
|---|---|---|
| 1 | `language` / `script` / `is_handwritten` belong to a **line**, never to a page. Page values are *derived*. | `app/stages/assemble.py` |
| 2 | The summary is **line indices the model selected**; the assembler concatenates the original OCR text verbatim. The model never writes prose. | `app/stages/s5_slm.py`, `assemble.py` |
| 3 | Escalation keys on **validator failure and structural signals**, never on raw recognition confidence. | `app/stages/s4_router.py` |
| 4 | Recognition is **bucketed and padded** before batching. Batch *size* is a tunable; bucketing is not (§4.5). | `app/providers/local.py` |
| 5 | Thread caps are set **before any numeric library is imported**. | `app/__init__.py` |
| 6 | `PROVIDER_PROFILE=laptop` with `SYNTHETIC_ONLY=false` refuses to start. There is no bypass flag. | `app/providers/factory.py` |
| 7 | Deterministic fields (account, IFSC, UTR/RRN, MICR, card fragment, amounts, dates) **never** pass through the SLM. | `app/stages/s5_validators.py` |
| 8 | A value the SLM proposes must appear in the page's own OCR text within an edit distance of 2, or it is rejected and flagged. | `app/stages/s5_slm.py` |
| 9 | A failed checksum is **flagged**, never silently corrected. | `app/stages/s5_validators.py` |
| 10 | `value` (the page's own script) is immutable; `value_en` is advisory and carries which engine produced it. | `app/stages/s6_translate.py` |
| 11 | Typed exceptions at every boundary. No bare `except`, no silent defaults. | everywhere |
| 12 | Licence gate: Apache-2.0 / MIT / BSD / MPL-2.0 / ISC / PSF only. | `models/PROVENANCE.json` |

---

## 2. The key classes map

One row per thing you are likely to need to change. "Contract it enforces" is the promise that
breaks if you change the row carelessly.

### 2.1 Configuration and process setup

| Module | Class / function | Responsibility | Contract it enforces |
|---|---|---|---|
| `app/__init__.py` | *import side effect* | Writes `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `ORT_INTRA_OP_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` into the environment **before** NumPy / ONNX Runtime / OpenCV can read them | Rule 5. Importing `numpy` before `app` silently un-caps every thread pool |
| `app/settings.py` | `Settings` | Every knob, typed, bounded and documented, read from the environment and `.env` | `SLM_MAX_TOKENS ≤ JOB_TOKEN_BUDGET`; exposes `residency_violation_reason` but does **not** raise it |
| `app/settings.py` | `get_settings()` | Process-wide singleton (cached) | Call `reset_settings_cache()` in tests after changing environment variables, or you get the old values |
| `app/stages/__init__.py` | `configure_cv2_threads()` | Caps OpenCV's own thread pool, which ignores `OMP_NUM_THREADS` | Rule 5 for OpenCV specifically |

### 2.2 Model providers — the only things that touch a model

| Module | Class / function | Responsibility | Contract it enforces |
|---|---|---|---|
| `app/providers/base.py` | `ModelProvider` (Protocol) | The eight methods every provider must implement: `detect_script`, `classify_page`, `ocr_page`, `ocr_page_heavy`, `identify_language`, `extract_fields`, `extract_fields_batch`, `translate` | A stage may only ask for these eight things. Adding a ninth is a design change, not a patch |
| `app/providers/base.py` | `JobBudget`, `Reservation` | Per-job token ceiling, thread-safe; reserve before the call, settle after | Raises `TokenBudgetExceeded`. It **accounts**, it does not count — estimation belongs to the caller |
| `app/providers/base.py` | `ModelArtifactMissing` | A named weight file is absent | The message names the file and the fetch command; never a bare `FileNotFoundError` |
| `app/providers/local.py` | `LocalProvider` | ONNX detection + recognition, fastText language ID, CTranslate2 translation — all offline | Rule 4 (bucketed padding); `REC_WIDTH_BUCKETS = (80, 160, 240, 320, 480, 640, 960, 1280, 1920)`, `REC_IMAGE_HEIGHT = 48`. **Do not lower the top edge.** It is a clamp: a crop wider than it is squeezed horizontally, and a squeezed line does not degrade gently under CTC — it drops characters and stops. At the old 640 edge, 5 of 13 lines on an ordinary English letter were over it and "We acknowledge your complaint regarding the disputed transaction." decoded as "W.", at *high* confidence, because a short decode is a confident one |
| `app/providers/hf_api.py` | `HFAPIProvider` | OpenAI-compatible client for the remote SLM and Tier-2 VLM | JSON mode + Pydantic validation + **one** repair retry; `max_retries=0` on the SDK so this class owns the only retry loop |
| `app/providers/hf_api.py` | `RetryPolicy`, `backoff_delay()`, `_is_retryable()` | Exponential backoff with jitter; 429 and 5xx retried, everything else not | A 402 (payment) and a 400 (bad model) cost exactly one round trip, never a storm |
| `app/providers/cache.py` | `ResponseCache`, `make_cache_key()` | `sha256(model + prompt + params) → response`, in SQLite | Makes an interrupted evaluation resumable at zero model cost. A cache **hit is proof** the prompt — and therefore the OCR text — is byte-identical to a previous run |
| `app/providers/factory.py` | `get_provider()`, `CompositeProvider` | Chooses local / remote / composite by profile; **enforces the residency gate** | Rule 6. `laptop` + `SYNTHETIC_ONLY=false` → `ResidencyViolation` at startup |
| `app/providers/factory.py` | `is_declared_synthetic()` | Refuses corpus files with no `synthetic: true` sidecar or directory manifest | Rule 6, at the file level |

### 2.3 The pipeline stages, in the order they run

| Module | Class / function | Responsibility | Contract it enforces |
|---|---|---|---|
| `app/pipeline.py` | `run_document()` | The only place that knows the stage order and what each stage owes the next; two-stage delivery | A `DocumentRejected` becomes a **bounce envelope**, not an exception — refusal is a product outcome |
| `app/stages/s1_ingest.py` | `ingest()`, `sniff_type()`, `ResourceCaps` | Magic-byte type sniff, resource caps, rasterise at `RASTER_DPI = 200` | `document_id` = `sha256` of the **bytes**, computed once, and it is the idempotency key |
| `app/stages/s1b_bypass.py` | `detect_bypass()`, `is_junk_text_layer()` | A usable digital PDF text layer removes recognition from the critical path | A *junk* text layer (garbled embedded fonts) is refused rather than trusted |
| `app/stages/s15_quality.py` | `QualityGate`, `QualityThresholds`, `measure()`, `deskew()`, `auto_invert()` | The accept / repair / bounce decision and the repairs themselves | A bounced page short-circuits: no recognition, no classification, no model call |
| `app/stages/s2_script.py` | `identify_script()`, `ScriptDecision` | Which writing system the page is in | Rule 1. `MIN_SCRIPT_CONFIDENCE = 0.60`, `MIN_VOTING_CHARACTERS = 8` |
| `app/stages/s2b_lid.py` | `identify_line_languages()`, `LidResult` | Language **per line** | Rule 1. Page language is never set here — `assemble` derives it |
| `app/stages/s3_classify.py` | `classify_page()`, `extract_structural_features()`, `load_fusion_weights()` | What kind of page this is (form, letter, cheque, statement…) | Falls back to a fitted structural path when the CNN artifact is absent, and says so in the log |
| `app/stages/s4_ocr.py` | `run_tier1()`, `probe_script_heads()`, `Tier1Result` | Batched recognition, plus the recognition-head probe that fixed the Devanagari routing defect (§4.10) | Rule 4; `MAX_RECOGNITION_PASSES = 2` — the probe cannot cause a loop |
| `app/stages/s4_router.py` | `route()`, `shadow_sample()`, `build_shadow_diff()` | Whether this page needs the heavier model; and the 3% shadow sample | Rule 3. `SIGNAL_POINTS` is the published weight table |
| `app/stages/s4b_vlm.py` | `escalate_page()`, `merge_tier2_lines()`, `apply_tier2()` | Tier-2 vision model, merged into Tier-1 lines by box overlap | `DEFAULT_IOU_THRESHOLD = 0.5`; duplicate line indices are rejected, not deduplicated |
| `app/stages/s5_validators.py` | `run_validators()`, `IFSCMaster`, `ValidatorHit`, `ValidatorReport` | Every deterministic identifier, by regex + checksum + lookup | Rules 7 and 9. Thread-pooled **inside** the stage, so it is called once per document, not once per page |
| `app/stages/s5_slm.py` | `build_extraction_contexts()`, `extract_pages()`, `ground_value()`, `apply_grounding_gate()` | The extractive model call and the gate on what it returns | Rules 2 and 8 |
| `app/stages/s6_translate.py` | `mask_entities()`, `translate_fields()`, `restore_entities()` | Advisory English rendering of a native-script value | Rule 10. Identifiers are masked with `ZQ{n}ZQ` sentinels before translation and restored after; a lost sentinel discards the translation |
| `app/stages/assemble.py` | `assemble_envelope()`, `derive_language_summary()`, `build_complaint_summary()`, `merge_field_candidates()` | Builds the published result; applies the grounding gate; derives every page-level summary | Rules 1, 2, 8 |
| `app/stages/s7_categorise.py` | `KeywordCategoriser`, `NotProduction`, `require_production_categoriser()` | Complaint category — **a placeholder** | Refuses to be used in production by raising `CategoriserNotProduction`, which `create_app` now calls when `REQUIRE_PRODUCTION_CATEGORISER=true` (and warns about at every start when false). It is a keyword matcher standing in for the RB-IOS taxonomy nobody has yet |

### 2.4 Service layer

| Module | Class / function | Responsibility | Contract it enforces |
|---|---|---|---|
| `app/api/routes.py` | `submit_document`, `get_job`, `healthz`, `metrics` | The HTTP surface | Upload refused above `API_MAX_UPLOAD_BYTES` **before** the body is buffered |
| `app/api/logging_filter.py` | logging filter | Redacts personal data at the log filter, not at each call site | Complainant data cannot reach the log by someone forgetting |
| `app/webhooks.py` | `WebhookDispatcher`, `sign_payload()`, `verify_signature()` | Signed delivery of each result version, with a dead-letter queue | An unsigned webhook is never sent: both a URL **and** a secret are required |
| `app/storage/repository.py` | `JobStore`, `JobRecord` | Append-only job and audit records | Same code path on SQLite and Postgres |
| `app/replies/bounce.py` | `render_bilingual_bounce_reply()`, `reply_for_envelope()` | The bilingual refusal message | No unfilled placeholder can reach a complainant; `BounceDecision.detail` (engineer text) is **never** rendered into it |
| `app/tasks.py` | Celery tasks | Distributed execution | Asserted to produce the same envelope as the synchronous path |

---

## 3. Where configuration comes from

### 3.1 Precedence, highest first

1. A real environment variable set in the shell.
2. A line in the `.env` file at the repository root.
3. The default in `app/settings.py`.

The names are identical at all three levels: the field `OCR_BATCH_SIZE` in `Settings` is the
environment variable `OCR_BATCH_SIZE` and the `.env` line `OCR_BATCH_SIZE=4`.

**Create your `.env` by copying the example** — never edit the example, and never commit `.env`:

```powershell
Copy-Item E:\pythonprojects\python_ocr\.env.example E:\pythonprojects\python_ocr\.env
```

There is no output on success.

> **Troubleshooting.** `.env` is git-ignored on purpose because it holds the remote API token.
> If `git status` ever shows `.env` as a new file, stop and check `.gitignore` before
> committing anything.

### 3.2 Seeing the values actually in effect

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import json; from app.settings import Settings, get_settings; secret={k for k,f in Settings.model_fields.items() if 'SecretStr' in str(f.annotation)}; print(json.dumps({k: str(v) for k,v in get_settings().model_dump().items() if k not in secret}, indent=1))"
```

Expected output: a JSON object of every knob and its current value, beginning

```json
{
 "PROVIDER_PROFILE": "laptop",
 "SYNTHETIC_ONLY": "True",
 "HF_BASE_URL": "https://router.huggingface.co/v1",
 "SLM_MODEL": "google/gemma-3-4b-it",
 "SLM_MODEL_FALLBACK": "meta-llama/Llama-3.1-8B-Instruct",
```

**Every secret is excluded by its declared type, not by its name or its value.** The command asks
`Settings` which fields are declared `SecretStr` — today `HF_TOKEN` and `WEBHOOK_HMAC_SECRET` —
and drops those, so a secret added later is dropped automatically and this output stays safe to
paste into a ticket. Two tempting shortcuts that are both wrong: filtering on the *name* (dropping
anything containing `TOKEN`) also hides `SLM_MAX_TOKENS` and `JOB_TOKEN_BUDGET`, two of the knobs
you most often need to read; and filtering on the *value* being a `SecretStr` instance leaks an
unset secret's field as `None`, which is harmless today and will not be the day someone sets it.

### 3.3 Two kinds of knob

- **`Settings` fields** — changeable by an operator through `.env` or the environment, no code
  change, no redeploy of the image. Everything in §4.2 – §4.9.
- **Module constants** — deliberately *not* settings, because no deployment has a different
  right answer for them, and because promoting one would put a tuning decision in the hands of
  someone who cannot re-measure it. Changing one is a code change plus a re-run of the
  evaluation. Listed in §4.10.

---

## 4. Every tunable

Each row gives the **default**, the **safe range**, and **what moving it does** — including
which direction is dangerous.

### 4.1 Reading the range column

"Range" is the *sensible* operating band, not the validator's bounds. `app/settings.py` rejects
values outside a wider hard limit (for example `OCR_BATCH_SIZE` must be 1–128); the range here
is where the numbers were measured and where the behaviour is understood.

### 4.2 Provider and residency

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `PROVIDER_PROFILE` | `laptop` | `laptop` \| `cpu_node` \| `gpu_node` | Chooses where inference happens. `laptop` sends SLM/VLM text to the remote router and keeps OCR/translation local. `cpu_node` is **fully offline** and is what an air-gapped deployment uses. `gpu_node` expects a local serving stack |
| `SYNTHETIC_ONLY` | `true` | `true` on `laptop`, `false` elsewhere | The residency gate. Setting `false` while the profile is `laptop` **refuses to start** (Rule 6) — that combination would send real complainant text to a non-Indian endpoint |
| `HF_TOKEN` | unset | a token, `laptop` only | Credential for the remote router. Held as a `SecretStr`; it is never logged, never printed and never written into any report or manifest — the manifest records only `hf_token_present: true\|false` |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | any OpenAI-compatible base URL | **Pointing this at a local `llama.cpp` or vLLM server is the entire change needed** to stop using a remote model. Nothing else in the code knows the difference. On `gpu_node` it must name an on-premises host if you set it at all: the profile asserts the deployment is air-gapped, so an off-premises endpoint is a residency violation and refuses to start. Left at the default it means "no model server here" — the node runs on the deterministic validators and never opens the socket. `cpu_node` never reads it |
| `LOCAL_ENDPOINT_HOSTS` | empty | comma-separated hostnames | Extra hosts to treat as on-premises. Loopback, RFC1918 literals, unqualified single-label names and the reserved suffixes (`.internal`, `.local`, `.lan`, `.intranet`, `.home.arpa`) are already recognised, so this is only for an in-house server behind a public-looking FQDN, e.g. `slm.rbi.gov.in`. **Not a bypass:** it changes where an endpoint is believed to be, never whether the check runs |
| `GPU_EXECUTION` | `auto` | `auto` \| `require` \| `never` | Whether the local vision and translation models use CUDA. `auto` uses the GPU only on `gpu_node` and only when the runtime exposes it — a laptop that happens to have `onnxruntime-gpu` installed keeps producing the CPU numbers the quality gate was calibrated against. **Use `require` on a real GPU node:** it turns a missing CUDA provider into a start-up failure instead of a node that works correctly and is ten times too slow, which is what a CPU-only `onnxruntime` wheel shadowing `onnxruntime-gpu` produces |
| `MODELS_DIR` | `models` | any path | Where local weights live. Set this to a shared read-only mount on a deployed node so nothing ever writes into a user's home cache |
| `IFSC_MASTER_PATH` | `app/resources/ifsc_master_sample.json` | a `.csv` or `.json` path | The bank-branch master list. **A `.csv` loads as *authoritative*: an IFSC code not in it fails the lookup and is flagged.** The committed `.json` fixture is non-authoritative, so unknown codes are "unverified" rather than "wrong". A missing or malformed file raises `ConfigurationError` — it does **not** fall back to the fixture, because that would report unverified codes as confirmed |
| `BANK_ALIASES_PATH` | `app/resources/bank_aliases.json` | a `.json` path | The bank **name** table, which `IFSC_MASTER_PATH` does not govern. Holds display names, alias spellings and `non_party_institutions` — the institutions (RBI, the Ombudsman's office, NPCI) that appear on complaints but are never the bank complained about. Widening this table *raises* accuracy rather than only recall: a bank missing from it can still fuzzy-match a bank inside it. A missing, malformed or bank-less file raises `ConfigurationError` |

### 4.3 The language model (SLM)

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `SLM_MODEL` | `google/gemma-3-4b-it` | any model the endpoint serves | Which model extracts the free-text fields. See §4.4 for how this default was chosen and why it is under review |
| `SLM_MODEL_FALLBACK` | `meta-llama/Llama-3.1-8B-Instruct` | as above | Swapped in by the evaluation driver only after **five consecutive documents** report an *outage* code. A model that answers badly is not an outage and does not trigger it |
| `SLM_MAX_TOKENS` | `512` | 128 – 1024 | Ceiling on the model's reply length. Too low truncates the JSON mid-object and the answer fails validation; too high just wastes budget, because this is an extractive task |
| `SLM_TEMPERATURE` | `0.0` | keep at `0.0` | Randomness. **Extraction must be reproducible**; anything above 0 makes the same page give different answers and invalidates the response cache's purpose |
| `JOB_TOKEN_BUDGET` | `8000` | 4000 – 32000 | Tokens one job may spend. Breach raises `TokenBudgetExceeded`. **Measured spend, 2026-08-08, 147 documents: 286 198 tokens total, median 1 914 per document, worst case 5 495** (a dense bank statement) — so one document fits inside 8 000 with room, and two do not. **Known trap:** this is a per-*job* ceiling enforced on a per-*provider-instance* budget. A batch caller that processes many documents through one provider without calling `reset_budget(job_id=…)` between them exhausts it after **two** documents at the measured median, and because the orchestrator absorbs SLM failures the symptom is a plausible-looking result set in which the model quietly stops contributing. `tools/run_corpus_eval.py` resets per document; **your batch caller must too** (recorded as open question Q9) |
| `PROVIDER_CACHE_PATH` | `.cache/provider.sqlite` | any path | The response cache. Deleting this file forces every model call to be made again — which is how you deliberately re-measure rather than replay |

### 4.4 Choosing the SLM — what the bake-off measured

`docs/bakeoff/REPORT.md` compared six candidates. Two mattered:

| model | schema validity | grounding | summary F1 | composite |
|---|---|---|---|---|
| `google/gemma-3-4b-it` | **0.5625** (n=16) | 0.90 | 0.667 | **0.7721** |
| `meta-llama/Llama-3.1-8B-Instruct` | **1.000** (n=18) | 0.556 | 0.408 | 0.7482 |

"Schema validity" is the share of answers that parse into the required JSON shape on the first
try or after one repair. **A schema failure loses the whole page's extraction**; a grounding
failure loses one value, and the grounding gate (Rule 8) is insurance the pipeline provides
anyway. That asymmetry is why the composite winner is the *weaker* practical choice, and it is
tracked as open question Q10.

**Q10 now has a measured answer, and it is worse than the bake-off suggested.** The 2026-08-08
evaluation run asked `google/gemma-3-4b-it` 147 times with credits working and **zero** refusals.
**54 answers validated — a schema validity of 0.367**, against the bake-off's 0.5625 from sixteen
observations. It varies 14-fold by document type:

| type | validity | median lines |
|---|---|---|
| `receipt` | **0.933** | 15 |
| `photo_evidence` | 0.917 | 15 |
| `complaint_form_deva` | 0.800 | 25 |
| `complaint_form_en` | 0.333 | 25 |
| `bilingual_form` / `cheque` / `complaint_letter_mar` | 0.200 | 13–24 |
| `bank_statement` / `complaint_letter_en` / `email_printout` | **0.067** | 13–99 |

Documents over 60 recognised lines validate 0.083 of the time (n = 12), but length is not the
whole story — three types at 13–14 median lines also score 0.067–0.200 while `receipt` at 15
lines scores 0.933.

**Why this is a ceiling, not a penalty.** A document whose answer fails the contract contributes
**zero** SLM fields, so free-text recall is capped at 0.367 no matter how good the surviving
answers are. Against a target of 0.80, this model cannot get there. **The practical remedy is to
set `SLM_MODEL=meta-llama/Llama-3.1-8B-Instruct` and re-run** — it measured 1.000 schema validity
at n = 18 — but that is an owner decision because it trades away grounding, and the run must be
re-scored afterwards. Evidence: `docs/eval/REPORT-2026-08-08-slmlive.md` §4.

- The brief's originally-named `Qwen/Qwen3-*-Instruct` model ids **do not exist** on the router.
- `json_object` mode was accepted by 3 of 5 probed models; `json_schema` by only 1 of 5, and
  which one depends on the backend provider the router happens to pick per request. So the
  portable baseline is **JSON mode + Pydantic validation + one repair retry**, which is what
  ships.

### 4.5 Recognition batching — and the A-10 finding, spelled out

**Read this before changing `OCR_BATCH_SIZE`.**

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `OCR_BATCH_SIZE` | **`4`** | 2 – 8 on laptop-class CPU; re-measure on the target node | Line crops sent to the recognition model per call |
| `REC_HEAD_CACHE_SIZE` | `3` | 1 – 8 | How many recognition models stay loaded in memory at once. Each is ~8 MB; with two heads installed there is nothing to evict |
| `REC_HEAD_CACHE_TTL_SECONDS` | `900` | 300 – 3600 | How long an idle recognition model stays loaded before being dropped |

**The finding (adjudication A-10, measured by `tools/bench_ocr_cpu.py`, in `bench_results.json`).**
The design brief assumed batching 16–32 crops per call would be faster. On a laptop-class CPU
it is measurably **slower**, because a large batch's padded tensor no longer fits in cache:

| batch size | Devanagari, speed vs looped | Latin, speed vs looped |
|---|---|---|
| 2 | 1.00× | **1.11×** |
| **4** | 0.98× | **1.11×** |
| 8 | 0.87× | 0.97× |
| 16 | 0.78× | 0.86× |
| 24 | **0.69×** | **0.77×** |
| 32 | 0.71× | 0.76× |

So batch 24 — the brief's value — is **23–31% slower** than processing crops one at a time. The
optimum is 2–4. The default was changed 24 → 4.

**What is *not* a tunable is the bucketing.** Crops are grouped into width buckets
`(80, 160, 240, 320, 480, 640)` and right-padded to the bucket width before the call. That is
mandatory, and the measurement says why: with bucket padding, output is **byte-identical at
every batch size** (`text_agreement = 1.000` for batch 2 through 32). Without it — the naive
per-crop loop at exact width — **44% of Devanagari crops decode differently** (agreement 0.556)
and 15% of Latin crops do (0.853), because the recognition head was trained on right-padded
input and unpadded crops are out of distribution. Rule 4 restated honestly: *bucketed, padded*
recognition is mandatory; batch size is a tunable, not a virtue.

> **Troubleshooting.** If you raise `OCR_BATCH_SIZE` and throughput drops, that is this effect,
> not a bug. If you see recognition text *change* when you change batch size, bucketing has been
> broken — that is a real defect, and `tools/bench_ocr_cpu.py` will show `text_agreement` below
> 1.0.

> **Deployment note.** All of the above was measured on an Intel i7-9750H (6 physical cores).
> The production target is a 16-vCPU node with a different cache hierarchy. **Re-run the
> benchmark there** (§6.4) before trusting the value of 4.

### 4.6 Threads and concurrency

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `OMP_NUM_THREADS` | `2` | 1 – 4 | OpenMP threads, read by NumPy and ONNX Runtime |
| `MKL_NUM_THREADS` | `2` | 1 – 4 | Intel MKL threads |
| `ORT_INTRA_OP_THREADS` | `2` | **do not exceed 2 casually** | Threads inside one ONNX operation. Every runtime independently defaults to "one thread per core", and **oversubscription costs 3–5×**, not a few percent |
| `CELERY_CONCURRENCY` | `2` | 1 – (cores ÷ intra-op) | Worker processes. The invariant is `CELERY_CONCURRENCY × ORT_INTRA_OP_THREADS ≤ physical cores` |

Check the budget against the machine you are on:

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from app.settings import get_settings; s=get_settings(); s.assert_thread_budget(); print('thread budget OK:', s.thread_budget)"
```

Expected output:

```
thread budget OK: 4
```

> **Troubleshooting.** A `ValueError` naming the two variables and the core count means you have
> oversubscribed. Lower `CELERY_CONCURRENCY` first — recognition is throughput-bound, and more
> processes each running fewer threads beats fewer processes thrashing. Measured on the
> reference laptop: concurrency 1 → 104 pages/min, concurrency 2 → 135 pages/min.

### 4.7 The quality gate

This is the accept / repair / bounce decision. Its numbers are **calibrated, not guessed** —
`docs/calibration/quality_gate.json` is the frozen calibration, produced by
`tools/calibrate_quality_gate.py` (§6.2).

**How the decision is made.** Five sub-scores are measured, combined into one `quality_score`
between 0 and 1 with fixed weights, and compared to a single threshold:

| sub-score | weight | measured from |
|---|---|---|
| sharpness | **0.45** | variance of the Laplacian over the glyph band |
| contrast | 0.25 | background level minus ink level |
| resolution | 0.20 | measured text height ÷ 0.09 in ≈ effective DPI |
| coverage | 0.06 | share of page area covered by text |
| occlusion | 0.04 | share of the page hidden by a finger, shadow edge or fold |

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `QUALITY_BOUNCE_THRESHOLD` | `0.35` | 0.25 – 0.45 | **The single most consequential number in the product.** Below this score the page is refused. Raising it bounces more documents (fewer bad extractions, more complainants told to resend); lowering it accepts more marginal pages |
| `QUALITY_BLUR_LAPLACIAN_FLOOR` | `9 500` | 6 000 – 12 000 | Blur level at which sharpness scores 0. Ramped in log space |
| `QUALITY_BLUR_LAPLACIAN_TARGET` | `26 000` | 20 000 – 40 000 | Blur level at which sharpness scores 1 |
| `QUALITY_CONTRAST_FLOOR` | `17.0` | 10 – 25 | Background-minus-ink level scoring 0 |
| `QUALITY_CONTRAST_TARGET` | `70.0` | 50 – 100 | Background-minus-ink level scoring 1 |
| `QUALITY_EFFECTIVE_DPI_FLOOR` | `30.0` | 20 – 50 | Effective DPI scoring 0 |
| `QUALITY_EFFECTIVE_DPI_TARGET` | `140.0` | 100 – 200 | Effective DPI scoring 1 |
| `QUALITY_TEXT_COVERAGE_FLOOR` | `0.004` | 0.002 – 0.01 | Below this share of the page covered in text, the page bounces as effectively blank |
| `QUALITY_SKEW_REPAIR_DEG` | `0.75` | 0.5 – 2.0 | Tilt at or above which the page is rotated straight. **Below it the resampling costs more sharpness than the rotation recovers** — this is why it is not 0 |
| `QUALITY_SKEW_BOUNCE_DEG` | `25.0` | 15 – 35 | Tilt beyond which the correction is untrustworthy and the page is refused |
| `QUALITY_MIN_PAGE_DIMENSION_PX` | `200` | 150 – 400 | Short-edge floor — one inch at the 200 DPI rasterisation resolution. Below it, this is not a document |

**What the frozen calibration achieves** (295 synthetic pages):

```
expectation   accept  repair  bounce
clean             67      28       0
degraded          25     125       0
bounce             0       0      50
```

Bounce recall **1.000**, false bounces on clean pages **0.000**, overall bounce rate **16.95%**.

> **The caveat that matters.** These thresholds were calibrated against variants 0–2 of the
> synthetic corpus — the same split the evaluation report scores. It is the most optimistic
> number in the whole report, and the held-out split (variants 3–4) is what will test it
> honestly. Do not treat 0.915 expectation-satisfied as a production forecast.

### 4.8 Escalation and the shadow sampler

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `ESCALATION_SHADOW_RATE` | `0.03` | 0.01 – 0.10 | Share of pages that **passed** and are *also* run through the heavy model, purely to measure how often the router was wrong to pass them. Raising it buys statistical confidence and costs Tier-2 capacity linearly. A shadow-sampled page is **not** an escalation and must never be counted as one |
| `CODE_MIX_THRESHOLD` | `0.15` | 0.10 – 0.30 | `is_code_mixed` becomes true when at least this share of lines differ in script from the page majority |

The escalation decision itself is a weighted score (module constants, §4.10).

### 4.9 Extraction, translation, resource caps

| Knob | Default | Range | What moving it does |
|---|---|---|---|
| `GROUNDING_MAX_EDIT_DISTANCE` | `2` | 0 – 3 | **The grounding gate.** A value the SLM proposes must appear in the page's own OCR text within this many single-character edits, after Indic digits are normalised to ASCII. `0` demands an exact match and will reject correct values on noisy pages; `3` or more starts accepting values the model invented that merely resemble something on the page. This is the mechanism that makes Rule 8 real, and 2 is where it was set deliberately |
| `MAX_PAGES_PER_DOC` | `200` | 50 – 500 | Page cap; a longer PDF is refused as `PAGE_CAP_EXCEEDED` |
| `MAX_TOTAL_PIXELS` | `500 000 000` | — | Total rasterised pixels per document — the decompression-bomb ceiling |
| `MAX_DECOMPRESSION_RATIO` | `100` | 50 – 200 | Refuses a file that expands more than this many times on decode |
| `MAX_WALL_CLOCK_SECONDS` | `900.0` | 300 – 1800 | Per-document processing ceiling. Must stay **below** `CELERY_TASK_TIME_LIMIT` (1800) or the worker is killed before the document can report its own timeout |
| `API_MAX_UPLOAD_BYTES` | `67 108 864` (64 MiB) | — | Refused before the request body is buffered, so a huge upload costs no memory |

### 4.10 Module constants — code changes, not configuration

These are not in `Settings` on purpose. Every one is overridable per function call, which is how
the tests drive them; changing the module default is a code change plus a re-run of §6.

**Recognition head probe (`app/stages/s4_ocr.py`) — the Q12 fix.**

Background, because this is the subtlest part of the system. Recognition needs to know which
script it is reading *before* it reads it. The original code bootstrapped on Latin and then
voted on the resulting text — but a Latin recognition head can only emit Latin characters, so a
Devanagari page came back as Latin gibberish, the vote said "Latin", and the second pass never
fired. The loop was closed by construction. The fix re-reads **three line crops from the
bootstrap pass with every installed recognition head**, and lets the head that reads them best
decide *by the script it actually wrote*.

| Constant | Default | What moving it does |
|---|---|---|
| `MAX_RECOGNITION_PASSES` | `2` | The hard stop that makes "falls back once" mean once. Do not raise it |
| `PROBE_HEAD_SCRIPTS` | `("Latn", "Deva")` | Which heads take part. Add a script here when you install its recognition head |
| `PROBE_LINE_COUNT` | `3` | Crops sampled — the longest line in each of three vertical bands. **Measured:** 2 crops score 141/147 pages correct, 3 crops score **146/147**, 4 crops cost 32% of a page instead of 19% for no gain |
| `PROBE_CROP_MAX_WIDTH` | `160` px | Each crop is normalised to the recogniser's 48 px input height and truncated to this width — one width bucket, 5–10 glyphs, enough to tell two scripts apart. This truncation is what took the probe from 26–94% of a page down to ~16% |
| `PROBE_SCORE_MARGIN` | `0.05` | When two heads read the same crops as *different* scripts, the winner must lead by this much. **Measured over the 28 pages where heads disagreed:** the narrowest accepted margin is 0.079, the only rejected one is 0.033, and the rest run 0.12–0.44. The gap between "coin flip" and "real decision" is empty in this data, and 0.05 sits in it |
| `PROBE_MIN_SCRIPT_CHARACTERS` | `6` | If one head cannot decode this many letters from the sheet while the other can, the one that decoded wins. **This rule is load-bearing:** on a Marathi letter the Latin head returns 4 characters of rubble against the Devanagari head's 26 |
| `PROBE_MIN_INK_STD` | `8.0` | Crops with no ink are dropped — two heads cannot be compared on blank paper |

Cost, measured: probe p50 **94 ms**, p95 119 ms — about 16% of the pre-fix recognition stage at
p50 against a 25% budget. It is the one place the pipeline pays for correct Devanagari routing.

**Escalation weights (`app/stages/s4_router.py`).**

| Constant | Default | Notes |
|---|---|---|
| `ESCALATION_SCORE_THRESHOLD` | `2.0` | Escalate at or above this total |
| `SIGNAL_POINTS` | `validator_failure 1.0`, `script_disagreement 2.0`, `quality_score_low 1.0`, `handwriting_detected 0.5`, `handwriting_majority 0.5`, `structural_anomaly 0.5` | Note `script_disagreement` alone escalates. That is deliberate: two mechanisms arguing about a page's script is exactly the unsettled page Tier 2 exists for |
| `MAX_SCORED_VALIDATOR_FAILURES` | `3` | Three broken fields and thirty are the same decision |
| `MAX_SCORED_STRUCTURAL_ANOMALIES` | `3` | As above |
| `HANDWRITING_MAJORITY_RATIO` | `0.5` | Share of lines at which a page counts as "mostly handwritten" |

**Script and language identification.**

| Constant | Module | Default | Notes |
|---|---|---|---|
| `MIN_SCRIPT_CONFIDENCE` | `s2_script.py` | `0.60` | Below this the script decision is not trusted |
| `MIN_VOTING_CHARACTERS` | `s2_script.py` | `8` | Fewer characters than this cannot carry a script vote |
| `SECOND_SCRIPT_SHARE` | `s2_script.py` | `0.25` | Share at which a second script is reported as present |
| `DEFAULT_REC_HEAD_SCRIPT` | `s2_script.py` | `"Latn"` | The bootstrap head. Only a bootstrap now — the probe decides the real one |
| `MIN_LID_CHARACTERS` | `s2b_lid.py` | `4` | **Shorter lines get no language label at all.** This is the guard against confident nonsense on a three-character line |
| `MIN_LID_CONFIDENCE` | `s2b_lid.py` | `0.20` | fastText confidence floor |

**Classification, translation, categorisation.**

| Constant | Module | Default | Notes |
|---|---|---|---|
| `HANDWRITING_SCORE_THRESHOLD` | `s3_classify.py` | `0.55` | Above this a crop counts as handwritten |
| `MIN_CLASS_CONFIDENCE` | `s3_classify.py` | `0.35` | Below this the page class is reported as unknown rather than guessed |
| `MAX_DESKEW_DEGREES` | `s3_classify.py` | `25.0` | Matches `QUALITY_SKEW_BOUNCE_DEG` — keep them equal |
| `DEFAULT_IOU_THRESHOLD` | `s4b_vlm.py` | `0.5` | Box overlap at which a Tier-2 line is treated as the same line as a Tier-1 line |
| `MT_SENTINEL_TEMPLATE` | `s6_translate.py` | `ZQ{index}ZQ` | The placeholder that protects an account number from being "translated". Its exact shape matters — an earlier bracketed form was dropped by the translation model on 2 of 5 long sentences; `ZQ{n}ZQ` survives |
| `GROUNDING_WARNING_PREFIX` | `assemble.py` | `grounding_rejected` | Prefix of the warning emitted when the gate rejects a value |
| `DEFAULT_MAX_TAGS` | `s7_categorise.py` | `3` | Placeholder categoriser only |
| `FIELD_MAX_TOKENS` | `s5_slm.py` | name 48, address 128, relief 192, prior ref 48, summary 128 | Per-field completion ceiling, so a model that starts writing an essay is cut off by the API rather than by the validator after the tokens are paid for |

---

## 5. The failure → remedy playbook

Keyed by **what you observe**, not by what is wrong. Each entry: the symptom, how to confirm it,
the fix, and how to verify the fix.

### 5.1 SYMPTOM — Devanagari (Hindi/Marathi) character error rate is high

*Character Error Rate (CER) is the share of characters that are wrong; 0.05 is good, 0.5 is
unusable.* A Devanagari CER above ~0.30 on clean pages means the Devanagari recognition model is
not being used at all.

**Confirm, in order:**

1. **Is the Devanagari model present?**

   ```powershell
   Get-ChildItem E:\pythonprojects\python_ocr\models\ocr | Select-Object Name, Length
   ```

   Expected output — all five files, with `rec_deva.onnx` around 7.9 MB:

   ```
   Name                 Length
   ----                 ------
   det.onnx            4826518
   rec_deva.onnx       7912311
   rec_deva_dict.txt      2511
   rec_latn.onnx       7848423
   rec_latn_dict.txt      1852
   ```

   If `rec_deva.onnx` is missing, fetch it (§6.6). With one head installed the probe correctly
   disables itself and you get the old Latin-only behaviour with no error.

2. **Did the probe actually run and what did it decide?** Every prediction file records it.

   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import json; d=json.load(open(r'docs\eval\predictions\complaint_form_deva_v0_clean.json',encoding='utf-8')); print([l['script'] for l in d['pages'][0]['lines'][:5]])"
   ```

   Expected output on a correctly-routed Devanagari page — `Deva`, not `Latn`:

   ```
   ['Deva', 'Deva', 'Deva', 'Deva', 'Deva']
   ```

3. **Is the text shaping library present?** Devanagari needs Raqm (complex-script text shaping)
   plus a `fribidi` DLL on `PATH`. This affects the *corpus generator*, not the service, but a
   corpus rendered without it is garbage and will make recognition look broken:

   ```powershell
   $env:PATH = "E:\pythonprojects\python_ocr\tools\assets;$env:PATH"
   E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from PIL import features; print('raqm:', features.check('raqm'))"
   ```

   Expected output:

   ```
   raqm: True
   ```

   > **Troubleshooting.** `raqm: False` means `tools/assets/fribidi-0.dll` was not on `PATH`
   > *before* Pillow was imported. The generator prepends it itself and hard-fails if the check
   > is false — that hard failure is intentional, because the alternative is 60 pages of
   > dotted-circle garbage that looks like a model problem.

**Known residual, not a bug to chase:** a page that is genuinely **half Latin and half
Devanagari** (`bilingual_form`) gains nothing from the probe, because one recognition head per
page cannot read a page in two scripts. Mixing heads *within* a page is the open follow-on.

### 5.2 SYMPTOM — bounce rate is outside the 5–15% band

Too many refusals annoys complainants; too few pushes unreadable pages into the pipeline.

**Confirm** — re-measure the gate over the corpus without changing anything:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/calibrate_quality_gate.py --check
```

Expected output when nothing has drifted:

```
calibration reproduces the committed file
```

A non-zero exit and a diff means the thresholds and the committed calibration disagree.

**Fix.** Move `QUALITY_BOUNCE_THRESHOLD` (§4.7) — down to bounce less, up to bounce more — then
re-freeze and check the targets held:

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/calibrate_quality_gate.py --write
```

Expected output:

```
wrote E:\pythonprojects\python_ocr\docs\calibration\quality_gate.json
```

**Verify.** The tool prints `TARGET MISSED: …` to standard error for each broken target. The two
that matter are `bounce_recall >= 0.8` (pages designed to bounce must bounce) and
`clean_expectation_false_bounce == 0.0` (**a clean page must never bounce**). If lowering the
threshold to fix the rate breaks bounce recall, the threshold is not your problem — the
sub-score weights are, and those live in `QualityThresholds` in `app/stages/s15_quality.py`.

> **Troubleshooting.** If the rate is out of band on *real* mail but in band on the corpus, the
> corpus is the wrong shape, not the gate. The corpus is engineered to a 16.9% bounce profile by
> construction, so "16.9%" reproducing exactly is the corpus agreeing with itself, not evidence
> about production.

### 5.3 SYMPTOM — the SLM's answers fail validation ("schema validity low")

The log shows `slm_unavailable:PROVIDER_RESPONSE_INVALID`. The model answered; the answer did
not fit the required JSON shape, twice — the first attempt and one repair retry.

**Confirm** — count the failure modes across a run's predictions:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import json,glob,collections; c=collections.Counter(w.split(':')[1] for p in glob.glob(r'docs\eval\predictions\*.json') for w in json.load(open(p,encoding='utf-8')).get('warnings',[]) if w.startswith('slm_unavailable')); print(c.most_common())"
```

Expected output — a list of codes and counts. This is the actual result of the 2026-08-08 run:

```
[('PROVIDER_RESPONSE_INVALID', 93)]
```

93 of 147 eligible documents — a schema validity of **0.367** (§4.4). No other code appeared:
**zero** `PROVIDER_ERROR`, so nothing was refused by the endpoint.

Read the code:

| code | meaning | remedy |
|---|---|---|
| `PROVIDER_RESPONSE_INVALID` | The model answered and the answer was malformed | **This is a model-choice problem.** See §4.4 — the default model measured 0.5625 schema validity; the fallback measured 1.000. Switch `SLM_MODEL` to `meta-llama/Llama-3.1-8B-Instruct` and re-measure |
| `PROVIDER_ERROR` | HTTP 4xx from the endpoint | §5.4 |
| `PROVIDER_UNAVAILABLE` / `PROVIDER_RATE_LIMITED` | The endpoint is down or throttling | Genuine outage; the driver's fallback engages after five consecutive documents |
| `PROVIDER_METHOD_NOT_SUPPORTED` | The provider does not implement extraction — e.g. running with `--no-slm` | Expected in offline runs |

**Also try, before changing model:**

- Confirm JSON mode is actually being requested. `json_object` is the portable setting;
  `json_schema` is stricter but only 1 of 5 probed models accepted it, and acceptance varies
  per request because the router load-balances across backends.
- Raise `SLM_MAX_TOKENS` if answers are being truncated mid-object — a truncated JSON object is
  indistinguishable from a malformed one at the validator.
- Do **not** add a second repair retry without measuring: each retry is a full round trip and
  doubles the cost of the model's worst behaviour.

**Related symptom — the model answers validly but the fields come back empty.** Check what the
grounding gate actually rejected before blaming it:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import json,glob,collections; c=collections.Counter(dict(x.split('=',1) for x in w.split(':')[1:] if '=' in x).get('reason') for p in glob.glob(r'docs\eval\predictions\*.json') for w in json.load(open(p,encoding='utf-8')).get('warnings',[]) if w.startswith('grounding_rejected')); print(c.most_common())"
```

Expected output — the 2026-08-08 result:

```
[('empty', 61), ('ungrounded', 3)]
```

**Read the split, not the total.** `empty` means the model returned an empty value — it declined
to answer, and that is a model problem. `ungrounded` means the value is not on the page and the
gate threw it out — that is the gate working. Here the gate rejected **3 of 104 proposed values,
2.9%**. **Do not raise `GROUNDING_MAX_EDIT_DISTANCE` in response to a high total**: on this
evidence it would recover at most three values and would weaken Rule 8, which is what makes the
output defensible.

### 5.4 SYMPTOM — HTTP 402 `Payment Required` from the model endpoint

The account's included inference credits are exhausted.

**Confirm** with a single tiny request (19 tokens — this costs essentially nothing):

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from app.settings import get_settings; from openai import OpenAI; s=get_settings(); c=OpenAI(base_url=s.HF_BASE_URL, api_key=s.HF_TOKEN.get_secret_value(), max_retries=0); r=c.chat.completions.create(model=s.SLM_MODEL, messages=[{'role':'user','content':'Reply with the single word OK.'}], max_tokens=19); print('200 OK:', r.choices[0].message.content.strip())"
```

Expected output when credits are healthy:

```
200 OK: OK
```

**This is not a code change.** It is resolved by the account owner adding pre-paid credits or a
subscription. No engineer should attempt a purchase on the owner's behalf.

> **Important diagnostic subtlety — this one has bitten this project.** A tiny probe succeeding
> does **not** prove production-sized requests will succeed. The 2026-08-07 run measured a
> 19-token probe returning 200 on **both** models while **129 of 147** production-sized requests
> returned 402: refusals arrived **per request**, interleaved throughout the run, consistent with
> the router load-balancing across backend providers where only some are on the account's plan. A
> trivially-sized health check would have declared the endpoint healthy.
>
> **Always probe at production size** — roughly 2 000 prompt tokens, `max_tokens=512`, and
> `response_format={"type":"json_object"}` — before concluding the endpoint is usable. With
> credits restored, the 2026-08-08 run saw **0 refusals in 147 requests** and both probe sizes
> agreed; the point stands for the next time they do not.

### 5.5 SYMPTOM — Tier-2 escalation always fails

`tier2_applied` is 0 and every escalated page carries `tier2_failed:…:code=PROVIDER_ERROR`.

**Confirm** — look at the actual HTTP status, because a 402 (no credit) and a 400 (no such
model) both surface as `PROVIDER_ERROR` and were once mistaken for each other:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from app.settings import get_settings; from app.providers.hf_api import DEFAULT_VLM_MODEL; print('configured Tier-2 model:', DEFAULT_VLM_MODEL); print('base url:', get_settings().HF_BASE_URL)"
```

Expected output:

```
configured Tier-2 model: PaddlePaddle/PaddleOCR-VL
base url: https://router.huggingface.co/v1
```

**Known state:** that model returns HTTP **400 `model_not_supported`** — *"not supported by any
provider you have enabled"* — deterministically, and reproducibly even against a 1×1 pixel
image. So it is nothing about page size or content. Remedies, in order of preference:

1. Name a vision model the endpoint actually serves.
2. Enable a backend provider that serves this one.
3. Run Tier 2 locally on the deployment node.
4. Drop Tier 2 from this phase and say so in the spec.

**Consequence to know about:** the shadow sampler measures the router's false-negative rate by
diffing Tier-1 against Tier-2 on passed pages. With no Tier-2 answer, every diff is empty and
that metric cannot be produced at all — for a reason entirely separate from credits.

### 5.6 SYMPTOM — a translated value lost an account number (sentinel lost)

Before translating, identifiers are replaced with sentinels (`ZQ0ZQ`, `ZQ1ZQ`, …) and restored
afterwards. If a sentinel does not come back, the translation is **discarded** and the field is
flagged `TRANSLATION_UNAVAILABLE`. That is correct behaviour — it is the alternative to
publishing a mangled account number — but it leaves the field with no English rendering.

**Confirm** — count attempts and discards straight out of the committed predictions; this needs
no models and no token:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import json,glob; f=[v for p in glob.glob(r'docs\eval\predictions\*.json') for v in json.load(open(p,encoding='utf-8')).get('fields',{}).values()]; att=[v for v in f if v.get('mt_engine')]; lost=[v for v in f if 'TRANSLATION_UNAVAILABLE' in v.get('flags',[])]; print('translations attempted:', len(att), 'discarded:', len(lost))"
```

Expected output — the discard count is the number that matters. This is the 2026-08-08 result:

```
translations attempted: 29 discarded: 0
```

**What is known.** Measured against the shipped weights over every distinct Devanagari corpus
line — 54 fields carrying 72 identifiers — **54/54 survived, 0 identifiers lost**. A density
probe of 20 constructed Hindi sentences carrying 1–5 identifiers each lost 1 of 52 (1.9%), and
the 4-or-more-identifier rows were 5/5 intact. The "3 of 5 lost" figure that appears in older
notes was measured against a *bracketed* sentinel that has since been replaced and does not
reproduce with `ZQ{n}ZQ`.

**If the discard rate rises:** the cheap fix is splitting long fields at sentence boundaries
before translating. It was rejected once because it adds a new failure class — an identifier
restored into the wrong segment — to recover ~2% on synthetic sentences. Re-open that trade
only with a measured rate, not an anecdote.

> **Observation worth knowing.** The eligibility test currently hands *ASCII* values to the
> translator too: a reference like `CMS/2025/850859` came back re-spaced as `CMS / 2025 /
> 850859`. Nothing downstream is wrong — `value` is immutable and `value_en` is advisory — but
> it is wasted work and worth tightening when someone is next in `s6_translate.py`.

### 5.7 SYMPTOM — throughput is low (pages per minute)

**Confirm** — measure this machine, do not guess:

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/bench_ocr_cpu.py
```

Expected output ends with a gate verdict:

```
gate: p95 <= 4.0 s/page at concurrency 2 -> PASS (p95 3.14 s, 135.4 pages/min)
```

**Work through these in order:**

1. **Thread oversubscription** — by far the most common cause, and it costs 3–5× rather than a
   few percent. Verify `CELERY_CONCURRENCY × ORT_INTRA_OP_THREADS ≤ physical cores` (§4.6).
2. **Something imported NumPy before `app`.** The thread caps are set by importing `app`, and
   they only take effect if nothing numeric was imported first. `bench_results.json` records
   `numeric_modules_loaded_before_app` — it must be `[]`.
3. **`OCR_BATCH_SIZE` too high.** See §4.5: 24 is 23–31% *slower* than 4 on laptop-class CPU.
4. **The recognition head probe**, if and only if pages are short. It costs ~94 ms p50, which is
   ~16% of a normal page's recognition but ~40% of a three-line cheque. Adding a
   `recognise_crops(crops, script)` method to the provider protocol would roughly halve it by
   skipping the probe sheet's re-detection; that is the single highest-value performance change
   available and is recorded as a follow-up.
5. **The remote model leg.** If p95 end-to-end is dominated by network wait, that is not this
   service's latency. Re-run with `--no-slm` to measure the local pipeline alone.

### 5.8 SYMPTOM — a short line gets the wrong language

`lid.176` (the fastText language identifier) separates Hindi from Marathi poorly on short lines —
8 of 30 Devanagari pages had Marathi read as Hindi in the last measured run.

**The guard already in place:** `MIN_LID_CHARACTERS = 4` means a line shorter than four
characters gets **no** label rather than a confident wrong one, and `MIN_LID_CONFIDENCE = 0.20`
drops low-confidence answers. Raise `MIN_LID_CHARACTERS` to trade coverage for precision.

**The structural fix** is the constraint table: `SCRIPT_LANGUAGES` in `app/providers/local.py`
restricts which languages are even considered for a given script, so a Devanagari line can never
be labelled French. Extending that table with per-document-type priors is the principled
improvement; raising thresholds is the blunt one.

> **Related open defect worth knowing before you chase this.** A line's published `script` is
> currently the *recognition head's* script, not the script of the line's own characters. On a
> Devanagari form, an ASCII account number is published as `Deva`. This was invisible while the
> head was always Latin. It means `is_code_mixed` can never become true (0 of 147 recognised
> pages report more than one line script, against 30 that genuinely mix). The fix is to classify
> each line from its own decoded codepoints — the classifier `s4_ocr` already uses for its vote —
> and keep the head's script as provenance.

### 5.9 SYMPTOM — every HTTPS request fails with `CERTIFICATE_VERIFY_FAILED`

A corporate network (or a local antivirus such as Norton) is intercepting TLS with its own
certificate authority. Windows and Node trust it; Python's `certifi` bundle does not.

**Confirm:**

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import httpx; httpx.get('https://router.huggingface.co/v1/models', timeout=15)"
```

A failure prints `SSLCertVerificationError: unable to get local issuer certificate`.

**Fix — build a bundle that contains both, for this process only:**

```powershell
$certifi = & E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "import certifi; print(certifi.where())"
$out = "$env:TEMP\ca-bundle.pem"
Get-Content $certifi, "C:\ProgramData\Norton\Antivirus\wscert.pem" | Set-Content -Encoding ascii $out
$env:SSL_CERT_FILE = $out
```

There is no output. Re-run the confirm command; it should now succeed silently.

> **Never** set `verify=False` or patch the HTTP transport. Any measurement obtained over an
> unauthenticated channel is not a measurement, and shipping verification-disabled HTTP is
> explicitly forbidden by this project's working agreements. The right long-term fix is to add
> the corporate CA to the environment's trust bundle, or to rely on `truststore` — the
> downloader tools already call `truststore.inject_into_ssl()` where it is available, so the
> Windows certificate store is honoured.

### 5.10 SYMPTOM — `ModelArtifactMissing` on startup

The message names the missing file and the command that fetches it. Confirm what is present:

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -c "from app.settings import get_settings; from app.api.routes import model_artifact_report; [print(r) for r in model_artifact_report(get_settings())]"
```

Expected output: one line per artifact with a present/absent verdict. Anything absent is fetched
per §6.6 or, on an air-gapped machine, copied in by hand per `docs/AIR-GAPPED-DEPLOY.md`.

### 5.11 SYMPTOM — the page classifier is "unavailable" in the log

```
page-classifier CNN unavailable (MobileNetV3-Small page-classifier ONNX); using the fitted structural path
```

**This is not an error.** The optional CNN artifact is absent and the system is using the fitted
structural classifier (table lines, text-line geometry, colour histogram, handwriting score),
whose weights are in `docs/calibration/fusion.json`. It is the intended offline default. Train
and place the CNN with `tools/train_page_classifier.py` only if per-type classification accuracy
is actually limiting you.

---

## 6. Re-running every calibration and evaluation tool

Each tool answers one question and writes one artifact. All are re-runnable; none needs network
access except where noted.

### 6.1 The full evaluation — recognition, fields, latency, everything

```powershell
$env:PYTHONIOENCODING = "utf-8"
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/run_corpus_eval.py --score
```

Expected output: a progress line roughly every ten documents, then the full scorecard.

```
selected 177 documents (split=dev final=False types=all profiles=all limit=None)
provider profile active | profile=laptop synthetic_only=true slm=google/gemma-3-4b-it hf_token=set threads=2x2
[1/177] complaint_form_en_v0_clean status=ok 1.9s  elapsed=2s eta=364s
...
sweep done: 177 ok, 0 failed, 1370s, cache hit rate 0.080 -> manifest E:\...\docs\eval\PREDICTIONS_MANIFEST.json
```

Writes `docs/eval/predictions/*.json` (one per document), `docs/eval/report.txt`,
`docs/eval/report.json` and `docs/eval/PREDICTIONS_MANIFEST.json`.

**Budget about 23 minutes with the SLM live and a cold cache**, and about 6 minutes with
`--no-slm`. The difference is entirely network: a live model call costs a p50 of 7.3 s of wall
clock per document against 1.1 s for a cached one and 0.15 s for a bounced page.

Useful variants:

```powershell
# Offline smoke test — no network at all, five documents
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/run_corpus_eval.py --limit 5 --no-slm --score

# List what would run, load nothing, write nothing
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/run_corpus_eval.py --dry-run

# The published run over the held-out split. Do this ONCE, on code nobody has tuned since.
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/run_corpus_eval.py --split all --final --score
```

> **Troubleshooting.**
> - **The run is slow.** With the remote model answering, each document costs a real network
>   round trip (tens of seconds), and a document whose first answer fails validation costs two.
>   The sweep is **strictly sequential on purpose** — end-to-end p95 latency is a target, and a
>   number measured under self-inflicted parallel load is not that number.
> - **It was interrupted.** Just run it again. Every model response is cached by
>   `sha256(model + prompt + params)`, so a re-run pays for the OCR and not the model. Add
>   `--resume` to also skip documents that already have a prediction file.
> - **`LockedSplitError`.** You asked for `--split test` or `--split all` without `--final`.
>   That is the guard on the held-out split working correctly.
> - **TLS failures.** §5.9.

### 6.2 The quality-gate calibration

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/calibrate_quality_gate.py --check
```

Expected output when the committed calibration still matches the code:

```
calibration reproduces the committed file
```

`--write` freezes a new calibration; `--sample 3` assesses every third page for a quick look.
Needs **no models and no network** — the quality gate is pure OpenCV.

### 6.3 Scoring predictions without running the pipeline

The scorer imports nothing from `app/`, so anyone can re-score the committed predictions with no
model directory and no token, and change the scoring rules to disagree with the report:

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/eval_harness.py --corpus corpus --predictions docs/eval/predictions
```

Expected output: the same scorecard §6.1 prints at the end, reproduced exactly.

### 6.4 The CPU throughput benchmark

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/bench_ocr_cpu.py
```

Writes `bench_results.json`. **Run this on every new deployment target** — it is what produced
the A-10 batch-size finding (§4.5) and its answer is machine-specific.

### 6.5 The model bake-off

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/slm_bakeoff.py
```

Needs network and credits. **Re-running resumes** rather than restarts: cached answers replay
free, so you pay only for what the previous run could not measure. Writes `docs/bakeoff/`.

### 6.6 Fetching models and fonts

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/fetch_models.py
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/fetch_fonts.py
```

Needs network. Records provenance and SHA-256 for every file in `models/PROVENANCE.json`. On an
air-gapped machine, do not run these — follow `docs/AIR-GAPPED-DEPLOY.md`, which lists the exact
URLs, expected sizes and hashes so files can be carried in.

> **Troubleshooting.** A `403` on a model repository means the account has not accepted that
> model's licence terms. Accept them on the model page in a browser first; no code change helps.

### 6.7 Regenerating the test corpus

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe tools/generate_test_corpus.py
```

Deterministic at seed 1729 — the same seed regenerates byte-identical images, which is what
makes "265 of 295 images unchanged" a checkable claim after a generator change. Hard-fails if
Raqm text shaping is unavailable (§5.1).

### 6.8 The test suite

```powershell
E:\pythonprojects\python_ocr\.venv\Scripts\python.exe -m pytest -q
```

Expected output ends with a count of passing tests and no failures. Tests requiring local model
weights are marked `local_models` and skip cleanly when `models/` is empty — verify that before
concluding "the suite passes".

---

## 7. Change control — what to re-measure after moving a knob

| You changed | Re-run | Because |
|---|---|---|
| Any `QUALITY_*` threshold | §6.2 `--write`, then §6.1 | The bounce rate and the accept/repair/bounce confusion matrix both move |
| `OCR_BATCH_SIZE` | §6.4, then §6.1 | Throughput moves; recognition text must **not**. If it does, bucketing is broken |
| `SLM_MODEL` or `SLM_MAX_TOKENS` | §6.5, then §6.1 | Schema validity and field recall move together |
| `GROUNDING_MAX_EDIT_DISTANCE` | §6.1 | Field precision and recall trade directly against each other here |
| `ESCALATION_SHADOW_RATE` or any `SIGNAL_POINTS` weight | §6.1 | Escalation rate, escalation precision, and Tier-2 cost |
| Any probe constant (§4.10) | §6.1 | Devanagari CER, language-ID accuracy and the recognition stage's p50 all move |
| Thread or concurrency settings | §6.4 | Throughput only; accuracy is unaffected |
| `IFSC_MASTER_PATH` to a real authoritative CSV | §6.1 | Unknown branches start **failing** the lookup instead of being unverified, so field precision changes |

**Two rules of thumb.**

1. **A net improvement that contains a cell moving the wrong way should be reported as the
   cell, not the net.** A rollup that rose while one document type collapsed is how real
   regressions hide.
2. **Never tune against the held-out split.** Variants 3 and 4 of the corpus exist to test code
   nobody has tuned against them. Every number produced on the dev split is optimistic to
   exactly the degree the system was tuned on it — and the quality gate and the recognition
   probe were both tuned there.
