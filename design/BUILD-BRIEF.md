# BUILD BRIEF — RBI Complaint Document Intelligence Service

**For:** an autonomous coding agent building Phase 0 → Phase 1
**Inputs:** this brief · `PRD-rbi-ocr-service-v1.md` · `design-review-v2.html` · `TEST-DATA-SPEC.md` · `tools/generate_test_corpus.py`
**Read all four before writing code.** Where this brief and the PRD disagree, the PRD wins on *what*, this brief wins on *how*.

---

## 0. Operating rules for the agent

1. **Build Phase 0 completely before starting Phase 1.** Phase 0 has a hard exit gate. Do not begin the Celery/Postgres work until every Phase 0 acceptance test passes.
2. **The development machine has no GPU.** Nothing you write may require one. The 1.7B SLM and the 0.9B VLM are reached over HTTP in Phase 0; everything else runs locally on CPU and must actually work there.
3. **Do not invent model behaviour.** If a model, endpoint, or library API is not verified, write the adapter against the documented interface and put the uncertainty in `OPEN-QUESTIONS.md` rather than guessing and moving on.
4. **Every module ships with its test.** No module is done until `pytest` covers its contract, including the failure path.
5. **Fail loudly at boundaries.** Typed exceptions at ingestion, validation and provider layers. Never a bare `except`. Never a silent default that masks a missing field.
6. **When you hit a blocking ambiguity, stop and record it** in `OPEN-QUESTIONS.md` with the options and your recommendation. Do not pick silently.

---

## 1. Stack (pinned)

```
Python           3.11
API              FastAPI + uvicorn, Pydantic v2
Queue            Celery 5 + Redis 7          (Phase 1)
State            PostgreSQL 16 + SQLAlchemy 2 + Alembic   (Phase 1)
Objects          MinIO (S3 API)              (Phase 1)
OCR              paddleocr (PP-OCRv5) → onnxruntime for inference
PDF              pypdfium2                   (raster + text-layer bypass)
Imaging          Pillow, opencv-python-headless, numpy
LID              fasttext-wheel + lid.176.ftz (917 KB — the .ftz, never the 126 MB .bin)
Fuzzy            rapidfuzz
MT               ctranslate2 + transformers (IndicTrans2 dist-200M, INT8)
SLM (API)        openai SDK pointed at the HF router base URL
SLM (local)      llama-cpp-python, GGUF Q4_K_M, GBNF grammar   (Phase 1+)
Test             pytest, pytest-asyncio, hypothesis (for the validators)
Lint             ruff, mypy --strict on app/contracts and app/providers
Licence gate     pip-licenses in CI
```

**Forbidden dependencies:** vLLM (GPU-only; Phase 4), Surya (non-commercial weights), anything GPL/AGPL, any cloud LLM SDK other than the HF-router-compatible OpenAI client. CI fails the build on a licence outside `{Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, MPL-2.0, ISC, PSF}`.

---

## 2. Repository layout

```
rbi-ocr-service/
├── app/
│   ├── contracts/            # Pydantic models — the single source of truth
│   │   ├── envelope.py       # ResultEnvelope, DocumentResult, Field, Evidence
│   │   ├── ocr.py            # OCRLine, OCRPage  (script/language/is_handwritten HERE)
│   │   └── errors.py         # typed rejection reasons
│   ├── providers/
│   │   ├── base.py           # ModelProvider Protocol
│   │   ├── local.py          # ONNX / CTranslate2 / llama.cpp
│   │   ├── hf_api.py         # HF Inference Providers via OpenAI-compatible client
│   │   ├── cache.py          # sha256(model+prompt+params) → SQLite
│   │   └── factory.py        # profile selection + residency gate
│   ├── stages/
│   │   ├── s1_ingest.py      # MIME, sandbox, idempotency, resource caps
│   │   ├── s1b_bypass.py     # digital-PDF text layer
│   │   ├── s15_quality.py    # QUALITY GATE — accept / repair / bounce
│   │   ├── s2_script.py      # image script ID
│   │   ├── s2b_lid.py        # line-level language ID
│   │   ├── s3_classify.py    # page class, CNN + structural features
│   │   ├── s4_ocr.py         # PP-OCRv5, BATCHED recognition
│   │   ├── s4_router.py      # escalation decision + shadow sampler
│   │   ├── s4b_vlm.py        # Tier-2 escalation
│   │   ├── s5_validators.py  # deterministic extractors — NO model
│   │   ├── s5_slm.py         # constrained SLM, extractive spans only
│   │   ├── s6_translate.py   # entity-masked per-field MT
│   │   └── assemble.py       # envelope construction + grounding gate
│   ├── pipeline.py           # sync orchestrator (Phase 0)
│   ├── tasks.py              # Celery tasks + chord (Phase 1)
│   ├── api/                  # FastAPI routers
│   └── settings.py           # pydantic-settings, all env-driven
├── tools/
│   ├── generate_test_corpus.py
│   ├── eval_harness.py
│   └── bench_ocr_cpu.py      # the week-1 throughput spike
├── tests/
│   ├── contract/             # runs against EVERY provider profile
│   ├── unit/
│   └── golden/
├── corpus/                   # generated, committed with index.json
├── docker-compose.yml
└── OPEN-QUESTIONS.md
```

---

## 3. The three contracts that must be right first

Everything else is replaceable. These three are not — get them wrong and the rework is total.

### 3.1 `ModelProvider`

```python
class ModelProvider(Protocol):
    def detect_script(self, crops: list[Image]) -> ScriptResult: ...
    def classify_page(self, page: Image, feats: StructuralFeatures) -> PageClass: ...
    def ocr_page(self, page: Image, script: str) -> list[OCRLine]: ...
    def ocr_page_heavy(self, page: Image) -> list[OCRLine]: ...        # Tier 2
    def identify_language(self, text: str, script: str) -> LangResult: ...
    def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]: ...
    def extract_fields_batch(self, ctxs: list[ExtractionContext]) -> list[dict]: ...
    def translate(self, texts: list[str], src: str, tgt: str = "eng") -> list[str]: ...
```

`extract_fields_batch` exists so the provider — not the call site — decides whether concurrency means `asyncio.gather` (API profile) or a sequential loop (local profile). Do not put `asyncio` in the stage modules.

### 3.2 `OCRLine` — language lives here

```python
class OCRLine(BaseModel):
    text: str
    bbox: list[float]              # 8 floats, clockwise from top-left
    confidence: float
    script: str                    # ISO 15924
    language: str | None           # ISO 639-3 — set by s2b, per line
    is_handwritten: bool
    line_index: int
```

Page-level `primary_language` is a **derived summary** computed in `assemble.py`. If you find yourself writing `page.language`, you have reintroduced the bug the design review reversed.

### 3.3 `Field` — provenance is mandatory

```python
class Field(BaseModel):
    value: str | None              # native script, verbatim, IMMUTABLE
    value_en: str | None = None    # advisory only; never key business logic on this
    normalized: str | None = None
    confidence: float
    source: Literal["regex_validator","lookup_table","slm","classifier","human"]
    validator_passed: bool | None = None
    evidence: Evidence             # page_no + bbox + span — never optional
    mt_engine: str | None = None
```

---

## 4. Phase 0 — milestones

### M0 · Throughput spike (do this first, before any service code)

`tools/bench_ocr_cpu.py`: run PP-OCRv5 INT8 over N pages, thread-pinned, report p50/p95 per page and pages/min at concurrency 1/2/4.

- [ ] Sets `OMP_NUM_THREADS` **before** importing numpy or onnxruntime
- [ ] Compares looped recognition against **batched** recognition (16 and 32) and reports the speedup
- [ ] Writes `bench_results.json` committed to the repo

**Gate:** if p95 > 4 s/page at target concurrency, stop and record it in `OPEN-QUESTIONS.md` as a blocking finding. Do not proceed as though CPU-first is settled.

### M1 · Corpus + harness

- [ ] `generate_test_corpus.py` runs end to end; 10 types × 5 variants × 6 profiles committed under `corpus/`
- [ ] `eval_harness.py` computes CER/WER by script × modality × profile, field F1 by source, schema validity, escalation precision/recall, **line-level** LID accuracy
- [ ] Two of five variants per type held out as a locked test split; the harness refuses to report on the held-out split unless `--final` is passed

### M2 · Provider layer + residency gate

- [ ] `ModelProvider` Protocol; `local` and `hf_api` implementations
- [ ] `hf_api` uses the OpenAI-compatible client against the HF router base URL with `HF_TOKEN`, so swapping to a local llama.cpp server later is a base-URL change
- [ ] Response cache on `sha256(model + prompt + params)` → SQLite. **This is not optional** — it is what makes Phase 0 iteration feel local instead of costing 2 s and a rate-limit slot per experiment.
- [ ] Retry with exponential backoff and jitter; typed `ProviderRateLimited`, `ProviderUnavailable`
- [ ] Per-job token budget, enforced, with a typed error on breach
- [ ] **Residency gate:**

```python
def get_provider(s: Settings) -> ModelProvider:
    if s.PROVIDER_PROFILE == "laptop" and not s.SYNTHETIC_ONLY:
        raise ResidencyViolation(
            "laptop profile routes text to a non-Indian endpoint; "
            "permitted only with SYNTHETIC_ONLY=true"
        )
    return _REGISTRY[s.PROVIDER_PROFILE](s)
```

- [ ] Corpus loader refuses any file lacking a `synthetic: true` sidecar while in `laptop` profile
- [ ] Startup banner naming the active profile; `provider_profile` Prometheus gauge
- [ ] **Test:** `PROVIDER_PROFILE=laptop`, `SYNTHETIC_ONLY=false` → app refuses to start

### M3 · Ingestion + quality gate

- [ ] MIME sniffing by magic bytes, not extension; mismatch → `MIME_MISMATCH`
- [ ] Idempotency = `sha256` of attachment bytes; soft-dupe on (sender, normalised subject, day)
- [ ] Resource caps: pages (200), total pixels, decompression ratio, wall-clock — each raising its own typed error
- [ ] Digital-PDF bypass with a junk-text-layer guard (character-class distribution vs detected script)
- [ ] Quality gate scoring resolution, skew, blur (variance of Laplacian), text coverage, occlusion, rotation, blankness, duplication → `accept` / `repair` / `bounce` with a machine-readable reason
- [ ] `quality_score` and `quality_flags` on every page, including accepted
- [ ] **Test:** all 7 committed adversarial inputs produce typed rejections; no worker crashes, no OOM
- [ ] **Test:** the `bounce` profile bounces; `clean` and `light` accept; `scan` and `heavy` accept-with-repair — measured against `quality_expectation` in the corpus

### M4 · Vision stages

- [ ] Script ID on sampled line crops, majority vote
- [ ] PP-OCRv5 det → **batched** rec (bucket by aspect ratio, pad, batch 16–32). The looped implementation may exist only as a benchmark baseline.
- [ ] Recognition heads selected at runtime from an LRU cache (`maxsize=3`, TTL 15 min) — never hard-coded, because adding Kannada later must be a model drop
- [ ] Page classifier: MobileNetV3-Small logits + structural features (line count, ink density, aspect, table-line presence, colour histogram) → logistic-regression late fusion
- [ ] Line-level LID with `lid.176.ftz`, label set constrained by detected script
- [ ] `is_code_mixed` when ≥15% of lines differ in script from the page majority
- [ ] **Test:** on `bilingual_form`, English label lines carry `eng` and handwritten value lines carry `mar`; `is_code_mixed` is true

### M5 · Extraction, translation, assembly

- [ ] Validators for IFSC (structure **and** RBI master lookup), account number, card fragment, UTR/RRN, MICR, amounts (Indic numerals normalised, lakh/crore handled), dates. Property-test these with `hypothesis`.
- [ ] `bank_name` requires fuzzy-score floor **and** agreement with the IFSC bank-code prefix; disagreement flags rather than picking
- [ ] Validators run concurrently over the OCR text (thread pool)
- [ ] SLM handles only: complainant name/address, relief sought, prior-complaint reference, **summary span indices**
- [ ] `complaint_summary` = verbatim OCR text of the model-selected line indices, concatenated by the assembler. The model must not emit prose. If your prompt asks for a summary sentence, the design has been violated.
- [ ] Grounding gate: any SLM value absent from page OCR text within edit distance 2 is rejected and flagged
- [ ] Translation: per-field only; sentinel-mask every validator hit before MT and restore after; `value` immutable; `value_en` marked advisory with `mt_engine`
- [ ] **Test:** a Marathi field containing ₹45,000 and a 14-digit account number round-trips both byte-identically through translation
- [ ] **Test:** a fabricated SLM value not present in the OCR text is rejected by the grounding gate

### M6 · Envelope + vertical slice

- [ ] `ResultEnvelope` with `result_version`, `is_final`, per-field `source` and `evidence`
- [ ] JSON Schema exported; contract tests assert backward compatibility
- [ ] Synchronous `pipeline.run(path) -> ResultEnvelope` — no Celery yet
- [ ] `POST /v1/documents` accepting an upload and returning the envelope synchronously
- [ ] **Test:** the same 40 golden prompts produce schema-valid output under both `laptop` and `cpu_node` profiles; any behavioural divergence fails CI

### M7 · Phase 0 exit gate

- [ ] Full corpus processed end to end on a laptop with no GPU
- [ ] `eval_harness` report committed: CER by script × profile, field F1 by source, schema validity, escalation precision/recall, LID accuracy
- [ ] SLM bake-off across ≥3 candidates via the HF API, with a written recommendation
- [ ] `bench_results.json` answering the CPU throughput question
- [ ] `OPEN-QUESTIONS.md` current

---

## 5. Phase 1 — after the gate

Fork **`drmingler/docling-api`** (MIT) as the service skeleton rather than assembling FastAPI + Celery + Redis + Flower + compose from scratch. It already has the queue topology, worker pools, health endpoints and monitoring. Strip its conversion logic, keep its plumbing.

- Four Celery queues, one per worker pool: `q.preprocess`, `q.vision`, `q.ocr_heavy`, `q.nlp`. **One queue per pool, never per task** — this is what keeps each worker's resident model set fixed and removes any need to load and unload models on the hot path.
- Chord fan-out over pages, fan-in at the assembler. `acks_late=True`, idempotent task bodies keyed on `(job_id, page_no, stage)`, `visibility_timeout` > p99 stage latency.
- Models load in `worker_process_init`, one instance per child. ONNX and llama.cpp sessions are **not fork-safe** — do not load in the parent.
- Two-stage delivery: emit `result_version=1, is_final=false` at the fast-path SLA; increment as escalated pages land. Webhook on every increment, HMAC-signed, replayable DLQ.
- Escalation router on **validator failure + structural signals**, never raw CTC confidence. Ship the 3% shadow sampler in the same PR as the router — a router without it cannot be evaluated.
- Local SLM via llama.cpp with GBNF grammar, replacing `hf_api`. Record the schema-validity delta; it will move.
- Postgres schema with an append-only `audit_events` table. Every field mutation records actor, timestamp, before, after.
- PII redaction at the **logging filter**, not the call site. Never log raw OCR text at INFO.

---

## 6. Environment

```bash
PROVIDER_PROFILE=laptop            # laptop | cpu_node | gpu_node
SYNTHETIC_ONLY=true                # MUST be true whenever profile=laptop
HF_TOKEN=hf_...                    # laptop profile only
HF_BASE_URL=https://router.huggingface.co/v1
SLM_MODEL=Qwen/Qwen3-1.7B-Instruct
SLM_MODEL_FALLBACK=Qwen/Qwen3-0.6B-Instruct
SLM_MAX_TOKENS=512
SLM_TEMPERATURE=0
JOB_TOKEN_BUDGET=8000
PROVIDER_CACHE_PATH=.cache/provider.sqlite

OMP_NUM_THREADS=2                  # set before any numeric import
MKL_NUM_THREADS=2
ORT_INTRA_OP_THREADS=2
CELERY_CONCURRENCY=2               # concurrency × intra_op ≤ physical cores

OCR_BATCH_SIZE=24
REC_HEAD_CACHE_SIZE=3
ESCALATION_SHADOW_RATE=0.03
QUALITY_BOUNCE_THRESHOLD=0.35
MAX_PAGES_PER_DOC=200
MAX_DECOMPRESSION_RATIO=100
```

---

## 7. Definition of done, per module

1. Typed inputs and outputs, Pydantic at every boundary
2. A unit test for the happy path **and** the failure path
3. A contract test that runs under every applicable provider profile
4. Structured logging with PII redaction applied by the filter
5. A Prometheus metric for latency and for the error class it can raise
6. Docstring stating what it does *not* handle — the negative space is the part that gets forgotten

---

## 8. Do not do

- **Do not** put language on the page. It belongs on the line. (Design review, finding A2.)
- **Do not** let the SLM write prose for `complaint_summary`. Span indices only. (A3.)
- **Do not** send an account number, IFSC, UTR or amount through the SLM. Validators own those. (P0-7.)
- **Do not** escalate on OCR confidence alone. It is uncalibrated on Devanagari conjuncts and the router will silently fail on exactly the pages it exists to catch. (A1.)
- **Do not** loop recognition over line crops. Batch them — it is 3–5× on the hottest path. (Fig. 3, level 2.)
- **Do not** raise `intra_op_num_threads` above 2 to "speed things up". Every runtime defaults to `nproc` independently; oversubscription costs 3–5×.
- **Do not** load models in the Celery parent process and rely on fork.
- **Do not** call `SYNTHETIC_ONLY=false` under the `laptop` profile, and do not add a flag that bypasses the check.
- **Do not** vendor `IITB-LEAP-OCR/Indic-OCR-API` (GPL-3.0) or any Surya weight. Read them; do not link them.
- **Do not** translate a whole page. Fields only, entity-masked.
- **Do not** silently correct a failed checksum. Flag it.
- **Do not** build the review UI, the categorisation classifier, or GPU serving in Phase 0 or 1. They are P1 and P2 for a reason.

---

## 9. First three commits

1. `tools/bench_ocr_cpu.py` + `bench_results.json` — the answer to the riskiest assumption in the whole design, obtained in one day, before any architecture is committed to code.
2. `app/contracts/` — the three contracts in §3, with their tests. Everything downstream depends on these being right.
3. `app/providers/` with the residency gate and its refusal test — because the one failure this system cannot recover from is complainant data leaving the country during development.
