# Build Plan — RBI Complaint Document Intelligence Service

Full end-to-end build (owner decision 2026-08-07): everything from BUILD-BRIEF Phase 0 + Phase 1,
plus data-independent Phase 2/3 items, built as one continuous effort. This file is the anchor for
all build agents. Read `design/BUILD-BRIEF.md` for the authoritative architecture; where this file
and the brief disagree, this file records the owner's later decision.

## Owner decisions (supersede the brief where they touch it)

| Topic | Decision |
|---|---|
| Python | 3.12 (venv at `.venv`, created from `C:\Python312`). Brief said 3.11; 3.12 approved for library currency. |
| Model downloads | Small CPU models OK locally (PP-OCRv5 ~25 MB, IndicTrans2 dist-200M INT8 ~200 MB, fasttext lid.176.ftz). SLM/VLM (≥0.9B params) never downloaded — HF Inference API only, token from `.env` (`HF_TOKEN`). |
| Infra | No Docker on dev laptop. Dual-mode: default profile uses SQLite + local filesystem + in-process execution (Celery `task_always_eager` or plain sync orchestrator). `docker-compose.yml` + real Redis/Postgres/MinIO/Celery code paths included, tested in eager mode, ready for deployment. |
| Scope | Phase 0 (M0–M7) + Phase 1 service layer + stubbable extras: bounce auto-reply templates, shadow sampler, LRU recognition-head cache, batched-rec tuning, minimal review UI (single page, flagged field beside source crop, keyboard-first), stubbed categorisation classifier behind an interface (marked NOT-PRODUCTION; RB-IOS taxonomy absent). Skipped: Bhashini, vLLM/GPU serving, real golden set, mailbox profiling (needs RBI access). |
| Air gap | Production is air-gapped. Nothing at runtime may fetch from the network except the `hf_api` provider under `laptop` profile. `local` provider loads all weights from `models/` dir populated offline. |
| OCR engine detail | PP-OCRv5 via onnxruntime. Prefer pre-converted ONNX (e.g. RapidOCR-distributed PP-OCR models, Apache-2.0) over installing full paddlepaddle if the paddle wheel is oversized; record the choice + model provenance in OPEN-QUESTIONS.md and models/README.md. Devanagari rec head required. |

## Invariants (violating any of these is a build bug — from design-review v2)

1. `language`/`script`/`is_handwritten` live on **OCRLine**, never on the page. Page-level values are derived in `assemble.py` only.
2. `complaint_summary` = SLM-selected **line indices**; assembler concatenates verbatim OCR text. The model never writes prose.
3. Escalation router keys on **validator failure + structural signals**, never raw CTC confidence. 3% shadow sampler ships in the same module.
4. Recognition is **batched** (bucket by aspect ratio, pad, 16–32 per call). A looped path may exist only inside the benchmark tool.
5. Thread caps (`OMP_NUM_THREADS=2` etc.) are set **before any numeric import** (enforced in `app/__init__.py` / entrypoints).
6. Residency gate: `PROVIDER_PROFILE=laptop` + `SYNTHETIC_ONLY=false` → `ResidencyViolation` at startup. Corpus loader under laptop profile refuses files without a `synthetic: true` sidecar. No bypass flag.
7. Deterministic fields (account, IFSC, UTR/RRN, MICR, card fragment, amounts, dates) never pass through the SLM. Validators own them.
8. Grounding gate: SLM values must appear in page OCR text within Levenshtein distance 2 (script-aware; Indic digits normalized before comparison) or the field is rejected + flagged.
9. Failed checksums are flagged (`validator_passed: false`), never silently corrected.
10. `value` (native script) is immutable; `value_en` is advisory, carries `mt_engine`. Sentinel-mask validator hits before MT, restore after.
11. Typed exceptions at every boundary. No bare `except`, no silent defaults.
12. Licence gate: Apache-2.0 / MIT / BSD-2 / BSD-3 / MPL-2.0 / ISC / PSF only. No GPL/AGPL, no Surya, no non-commercial weights.

## Repository layout (from BUILD-BRIEF §"layout", with owner extensions)

```
app/
  contracts/ (envelope.py, ocr.py, errors.py)
  providers/ (base.py, local.py, hf_api.py, cache.py, factory.py)
  stages/ (s1_ingest.py, s1b_bypass.py, s15_quality.py, s2_script.py, s2b_lid.py,
           s3_classify.py, s4_ocr.py, s4_router.py, s4b_vlm.py, s5_validators.py,
           s5_slm.py, s6_translate.py, assemble.py)
  pipeline.py  tasks.py  settings.py
  api/  storage/  review_ui/ (minimal, owner-approved extra)
tools/ (generate_test_corpus.py [fixed copy], eval_harness.py, bench_ocr_cpu.py, assets/fonts/)
tests/ (contract/ unit/ golden/)
corpus/  models/ (gitignored, README only)  docs/
docker-compose.yml  OPEN-QUESTIONS.md  PLAN.md  README.md
```

## Workstreams

- **WS-A Foundation** (done inline by orchestrator): git, venv, scaffold, fonts, base deps.
- **WS-B Contracts**: `app/contracts/` + `app/settings.py` + typed errors + tests. Everything depends on this.
- **WS-C Providers**: Protocol, `local`, `hf_api` (OpenAI-compatible client), SQLite response cache, retry/backoff, token budget, residency gate + refusal test. Depends on WS-B.
- **WS-D Corpus + Harness**: fix `tools/generate_test_corpus.py` (Windows font fallback, utf-8 writes, global-seed leak, dead code), generate corpus (10×5×6 + adversarial), implement `eval_harness.py` with locked test split. Independent of B/C.
- **WS-E Stages**: E1 ingest/quality/bypass; E2 vision (script ID, LID, classifier, batched OCR, router+shadow, VLM tier-2 via provider); E3 extraction (validators + hypothesis tests, SLM extractive, translation masking, assembly + grounding gate). Depends on B; E2/E3 also on C.
- **WS-F Service**: sync pipeline, FastAPI (`POST /v1/documents`, `GET /jobs/{id}`, webhook w/ HMAC + DLQ), storage/audit (append-only, dual-mode SQLite↔Postgres), Celery tasks (eager-tested), docker-compose. Depends on E.
- **WS-G Extras**: auto-reply templates (bilingual, machine-readable defect codes), review UI page, categorisation stub. Depends on F.
- **WS-H Verification**: `bench_ocr_cpu.py` M0 spike on this laptop (proxy for target node), full-corpus e2e run, eval report committed, SLM bake-off (≥3 candidates via HF API — needs `HF_TOKEN`), final `OPEN-QUESTIONS.md` sweep.
- **WS-I Docs + packaging** (owner additions 2026-08-07, after build verified):
  1. Standalone HTML test page (`app/review_ui/` or `docs/test-page/`): upload a document to the running API, view the envelope, field evidence, and page images — zero build tooling, works from `file://` or served by FastAPI.
  2. `docs/AIR-GAPPED-DEPLOY.md`: exact step-by-step for the office air-gapped machine — venv from local wheels (`pip download` bundle instructions), model placement into `models/`, font/fribidi presence check, corpus regeneration, smoke test, service start, verification checklist.
  3. Local model directory provision: all HF/hub-touching code must honor a `MODELS_DIR`-style setting and `HF_HOME`/`local_files_only` so nothing ever writes to `~/.cache/huggingface` implicitly; office deploys point at a local directory. Settings + docs.
  4. `docs/TUNING.md`: key classes map, every tunable (env knobs, thresholds, batch sizes, cache sizes, escalation/quality thresholds), and a failure→remedy playbook (e.g. Devanagari CER high → check raqm/fonts/rec-head; bounce rate out of 5–15% band → recalibrate s15 thresholds; SLM schema validity low → JSON-mode retry/grammar; throughput low → batching/thread caps).
  6. `docs/API.md`: consumer-facing API reference — endpoints, request/response examples, envelope field semantics, webhook contract + HMAC verification, error codes, idempotency behavior. (FastAPI also serves OpenAPI at /docs; this file is the curated narrative version.)
  7. `tools/make_bundle.py` (owner-amended 2026-08-07 evening): produces TWO zips in `dist/`, both stamped with git SHA:
     - **Code bundle** (`rbi-ocr-code-<sha>.zip`, small): all source, tests, docs, fonts + fribidi DLL, corpus *generator*, `.env.example` (NEVER `.env` — assert the token is absent from the archive), `requirements*.txt` + a `constraints.txt` freeze of the working venv for reproducible installs. NO `models/`, NO `.venv/`, NO corpus images.
     - **Corpus bundle** (`rbi-ocr-corpus-<sha>.zip`, heavy): the generated `corpus/` tree exactly as committed (images, .gt.json sidecars, index.json, adversarial set) so office can test without regenerating.
  8. Documentation register (owner requirement: ASSUME THE READER KNOWS NOTHING — every doc is numbered step-by-step, every command copy-pasteable, expected output shown, troubleshooting per step):
     - `docs/AIR-GAPPED-DEPLOY.md`: from blank Windows/Linux machine → Python install check → venv creation → pip install (online via proxy AND offline wheelhouse variants) → font/fribidi verification → model placement into `models/` with the EXACT download URLs (PaddlePaddle HF ONNX repos per artifact, lid.176.ftz source, adalat-ai CT2 IndicTrans2 repo) + expected file sizes + SHA-256s from PROVENANCE.json → corpus unzip or regeneration → `.env` creation from `.env.example` (each variable explained; HF token marked optional/dev-only, never shipped) → smoke test commands → service start → verification checklist. Target: a developer receiving the bundles cold can be running by Monday.
     - `docs/FUNCTIONAL-SPEC.md` (owner addition): short product-manager-facing document — what the system does in plain language, inputs/outputs, the quality gate + escalation + human-review flow, per-field provenance guarantees, what v1 does NOT do, headline metrics with current measured values from the eval report.
     - TUNING.md and API.md as previously specified, same zero-knowledge standard.

## Working agreements for build agents

- Windows dev box: PowerShell-safe commands, no POSIX-only paths at runtime; `pathlib` everywhere; all file I/O `encoding="utf-8"`.
- Office network intercepts TLS (corporate proxy). `.venv/pip.ini` pins trusted hosts for pip; any code that downloads (font fetcher, model fetch tooling, `hf_api` client in dev) must attempt `truststore.inject_into_ssl()` first (guarded ImportError) so the Windows cert store's corporate CA is honored. Never ship verification-disabled HTTP.
- Run everything through `.venv\Scripts\python.exe`. pytest must pass before a workstream is "done".
- Every module: typed I/O (Pydantic v2 at boundaries), happy + failure path tests, docstring stating what it does NOT handle, structured logging with PII redaction at the filter, Prometheus metric hooks (prometheus-client; no-op safe when server absent).
- Anything ambiguous: pick the sane option, record it in `OPEN-QUESTIONS.md` (options + chosen recommendation), keep building. Do not block.
- Commit at workstream boundaries with conventional messages. Never commit `.env`, `models/`, or corpus regeneratable artifacts beyond `corpus/` itself (corpus IS committed per brief).
