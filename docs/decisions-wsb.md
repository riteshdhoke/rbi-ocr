# WS-B decisions — contracts + settings

Judgment calls made while building `app/__init__.py`, `app/contracts/**`,
`app/settings.py` and their tests. Owner merges the relevant ones into
`OPEN-QUESTIONS.md`. Nothing here contradicts a PLAN.md invariant; where a
choice tightens one, that is stated.

## Thread caps (`app/__init__.py`)

| # | Decision | Rationale |
|---|---|---|
| B1 | Capped **five** variables, not three: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `ORT_INTRA_OP_THREADS` (named in BUILD-BRIEF §6) **plus** `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`. | NumPy wheels ship OpenBLAS, which reads its own knob. Leaving it uncapped reintroduces exactly the oversubscription invariant 5 exists to stop. Exposed as `app.THREAD_CAP_VARS`. |
| B2 | Existing environment values are **never overwritten**. | An operator sizing a 32-core node must stay in control. `apply_thread_caps()` returns only what it actually set, so entrypoints can log the delta. |
| B3 | A numeric module imported **before** `app` produces a `ThreadCapWarning`, not an exception. | `tools/bench_ocr_cpu.py` legitimately sets its own caps and imports NumPy first (BUILD-BRIEF M0); raising would make the correct pattern unusable. The warning names the offending modules, so a genuine ordering bug is visible in test output and worker logs. Tested in a subprocess. |
| B4 | `app/__init__.py` imports **stdlib only** (`os`, `sys`, `warnings`). | A numeric import here would defeat the file's only purpose. `tests/unit/test_settings.py` asserts that importing `app`, `app.contracts` and `app.settings` pulls in none of `numpy/onnxruntime/cv2/ctranslate2/torch/scipy/sklearn`. |

## Error taxonomy (`app/contracts/errors.py`)

| # | Decision | Rationale |
|---|---|---|
| B5 | `RejectionReason.ENCRYPTED_PDF` is canonical; `ENCRYPTED_DOCUMENT` is an **enum alias with the same value**. | TEST-DATA-SPEC §5 spells it `ENCRYPTED_DOCUMENT`, the WS-B brief spells it `ENCRYPTED_PDF`. Both names resolve to one member, so the wire vocabulary stays single-valued and neither document is wrong. |
| B6 | One concrete exception class per rejection reason, registered in `REJECTION_EXCEPTIONS`, with `code == reason.value`. | BUILD-BRIEF M3 requires each resource cap to raise "its own typed error". A test asserts the mapping is total, so a new reason cannot be added without a class. |
| B7 | Added `ProviderResponseInvalid` beyond the three provider errors named in the brief. | PRD P0-8 requires schema-validity to be a *recorded metric*; counting it needs its own class. Adding it here avoids WS-C having to edit this lane later. |
| B8 | Every error carries an advisory `http_status` ClassVar. | Lets the WS-F API map errors without a second lookup table. This module imports no web framework. |
| B9 | `ResidencyViolation` subclasses `ConfigurationError`, **not** `ProviderError`. | It must not be swallowed by provider retry/backoff logic. Asserted in `test_errors.py`. |
| B10 | Quality-gate flags and bounce reasons live in `envelope.py`, not `errors.py`. | A bounced document produces a valid envelope, not a raised exception. Keeping them out of the exception module stops anyone from raising a bounce. |

## OCR contracts (`app/contracts/ocr.py`)

| # | Decision | Rationale |
|---|---|---|
| B11 | `OCRPage` **rejects** the keys `language`, `languages`, `primary_language`, `language_distribution`, `script`, `primary_script`, `is_handwritten`, `is_code_mixed` with a message naming invariant 1. | Design-review finding A2 must be unrepresentable, not merely undocumented. A `mode="before"` validator gives a targeted message instead of a generic "extra field" error. |
| B12 | `confidence` is **also** forbidden on `OCRPage` (tightens invariant 3). | The cheapest way to guarantee nothing escalates on aggregate CTC confidence is to never materialise the number on the page. Line-level confidence remains available. |
| B13 | No derived convenience properties (`page.primary_language`, etc.) on `OCRPage`. | A property is how the bug comes back. Derivation happens in `assemble.py` onto `PageResult`. |
| B14 | `bbox` validation = exactly 8 coordinates + all finite. Corner **order** is not validated; negative coordinates are allowed. | A contract cannot know page orientation, and a bounce-profile page rotated 17° legitimately has corners off-canvas. `bbox_from_xyxy()` is provided so call sites do not hand-roll the clockwise order. |
| B15 | `line_index` must be unique within a page (contiguity is not required). | Indices are the handles the extractive summary selects by (invariant 2) and must survive line filtering. `OCRPage.line_by_index()` looks up by index, never by list position. |
| B16 | `ExtractionContext` enforces `target_fields ⊆ SLM_ELIGIBLE_FIELDS` and raises naming invariant 7 for any deterministic field. | Invariant 7 becomes structurally impossible to violate rather than a review checklist item. |
| B17 | `ExtractionContext` deliberately has **no** map of already-extracted identifier values; it carries `masked_spans: list[TextSpan]` instead. | Under the laptop profile this object crosses the network. Sentinel masking needs *locations*, not values, so account numbers and IFSCs have no home in it at all. A test asserts the field does not exist. |
| B18 | `RawField` enforces invariant 2: for `complaint_summary`, `value` must be `None` and `line_indices` non-empty. | A model that ignores its instructions and returns prose is rejected at the provider boundary, not reviewed later. |
| B19 | `StructuralFeatures.colour_histogram` is required (min length 1). | An empty histogram is a missing feature, not a legitimate all-zero page — "no silent default that masks a missing field" (BUILD-BRIEF §0.5). |

## Envelope (`app/contracts/envelope.py`)

| # | Decision | Rationale |
|---|---|---|
| B20 | `Field` carries one extension beyond BUILD-BRIEF §3.3: `flags: tuple[FieldFlag, ...]`. | Invariants 8 and 9 both say a value is "flagged"; free text is not something a review queue can filter on. Machine-readable codes make the flag actionable. |
| B21 | `VALIDATOR_FAILED` is attached **automatically** when `validator_passed is False`. | Guarantees invariant 9 rather than trusting every call site. This is derivation from a fact already declared on the same object, not a silent default: the *value* is never touched. |
| B22 | `Field` and `Evidence` are `frozen=True`; mutation goes through `with_translation()` / `with_flags()`, which revalidate. | Invariant 10 ("`value` is immutable") becomes enforced rather than asserted. Translation cannot rewrite the native-script value on any code path. |
| B23 | `ComplaintSummary` stores `spans: tuple[SummarySpan, ...]` (evidence + verbatim text) and **re-derives `text` from them**, rejecting any mismatch. | Makes invariant 2 checkable at the contract boundary. Model-written prose cannot be represented in the envelope at all. |
| B24 | `BounceReason` is one flat enum spanning ingestion rejections *and* quality defects, with `BOUNCE_REASON_FOR_REJECTION` (total) and `BOUNCE_REASON_FOR_QUALITY_FLAG` (partial by design). | The complainant receiving an auto-reply (P1-2) does not care which subsystem refused the document, and the template lookup should not have to join two enums. `*_REPAIRED` flags never map to a bounce; a test asserts that. |
| B25 | `PageResult.shadow_sampled` is separate from `escalated`, and only `escalated` forces `escalation_flag`. | A 3% shadow run through Tier 2 must not inflate the escalation rate it exists to measure (invariant 3). |
| B26 | `PageResult.primary_language` must be the argmax of `language_distribution` and appear in it. | It is a derived summary, not an independent judgement; an inconsistency means `assemble.py` has a bug. |
| B27 | `document is None` is legal **only** when `bounce_decision.decision == "bounce"`, and a bounced envelope must carry no `DocumentResult`. | A missing result must say why; it may never arrive as an empty stand-in. |
| B28 | `DocumentResult` rejects `source == "slm"` on any name in `DETERMINISTIC_FIELDS`. | Invariant 7 checked a second time at the point of publication, where the value actually leaves the service. `source == "human"` remains legal for identifiers (review-UI corrections, P1-1). |
| B29 | `created_at` must be timezone-aware. | Naive timestamps across a two-stage delivery (`result_version` 1 → n) are unreconcilable. |
| B30 | `pydantic.Field` imported as `_Field`. | `Field` is the name of the contract class in BUILD-BRIEF §3.3; the brief's name wins. |

## Schema export (`app/contracts/schema_export.py`, `docs/schema/`)

| # | Decision | Rationale |
|---|---|---|
| B31 | Schema generated in **serialization** mode, keys sorted, trailing newline, UTF-8. | Consumers read what we emit. Sorted keys make the artefact diffable and the drift test deterministic. |
| B32 | Two separate tests: *drift* (committed == regenerated) and *backward compatibility* (no `$defs`, required property or enum member may disappear). | They have different fixes — "regenerate" vs "bump `SCHEMA_VERSION` and publish a new file". One combined test would hide that. |
| B33 | No `jsonschema` dependency. | Pydantic is the validator of record; adding a runtime dependency to assert a property of our own generator is licence and supply-chain cost for no coverage. Structural comparison is what the compatibility promise actually is. |
| B34 | The internal OCR contracts are **not** published; a test asserts `OCRPage`/`OCRLine` never appear in `$defs`. | Keeps the external surface to the envelope only. |

## Settings (`app/settings.py`)

| # | Decision | Rationale |
|---|---|---|
| B35 | Field names are **UPPERCASE**, identical to the env var. | BUILD-BRIEF §2 writes the residency gate as `s.PROVIDER_PROFILE` / `s.SYNTHETIC_ONLY`; the brief's spelling wins and copy-paste from `.env.example` stays literal. |
| B36 | `laptop` + `SYNTHETIC_ONLY=false` does **not** raise in `Settings` (as instructed). `residency_violation_reason` returns the exact refusal string for WS-C's factory. | The gate belongs where a model call can originate. Keeping settings loadable also means tooling can *report* a bad configuration instead of dying while reading it. The message is written once, here, so `get_provider` and its test cannot drift apart. |
| B37 | A missing `HF_TOKEN` never raises; a token present on a non-laptop profile never raises either — it is reported via `has_stray_remote_credential`. | Tests must run with no token and no network. Raising on a stray token would break BUILD-BRIEF M6's "same goldens under both `laptop` and `cpu_node`" test on a dev laptop whose `.env` legitimately holds one. WS-C/WS-F should surface the flag in the startup banner. |
| B38 | `CELERY_CONCURRENCY × ORT_INTRA_OP_THREADS ≤ cores` is exposed as `assert_thread_budget()` for CI, not enforced on load. | A 2-core CI runner must still be able to import the package with the documented defaults (2×2=4). PRD P0-11 asks CI to assert it, which is exactly what this supports. |
| B39 | `SLM_MAX_TOKENS > JOB_TOKEN_BUDGET` **is** rejected on load. | Self-contradictory: a single call could never complete within the job budget. This one has no legitimate configuration. |
| B40 | Added knobs not present in `.env.example`: `CELERY_CONCURRENCY` (BUILD-BRIEF §6), `MAX_TOTAL_PIXELS`, `MAX_WALL_CLOCK_SECONDS` (M3 caps), `REC_HEAD_CACHE_TTL_SECONDS` (M4, 15 min), `CODE_MIX_THRESHOLD` (P0-5, 0.15), `GROUNDING_MAX_EDIT_DISTANCE` (invariant 8, 2), `DATABASE_URL`, `REDIS_URL`, `CELERY_TASK_ALWAYS_EAGER`, `WEBHOOK_HMAC_SECRET`, `MODELS_DIR`, `CORPUS_DIR`, `LOG_LEVEL`. | The invariants require them and defaults must not be invented at the call site. **`.env.example` is outside the WS-B lane — owner/orchestrator should add these lines to it.** The test only asserts `.env.example ⊆ Settings`, so it stays green either way. |
| B41 | `get_settings()` is `lru_cache`d with `reset_settings_cache()` for tests. | Every stage reads settings; re-parsing `.env` per read is waste, and an uncached singleton makes test isolation harder, not easier. |

## Test scaffolding (`tests/conftest.py`)

| # | Decision | Rationale |
|---|---|---|
| B42 | `conftest.py` inserts the repo root into `sys.path` and imports `app` before anything else. | There is no `pyproject.toml`/`pytest.ini` in the repo yet (outside this lane), and the suite must run identically under `python -m pytest`, bare `pytest` and IDE runners on Windows. |
| B43 | Every `Settings` in tests is built with `_env_file=None` and a `clean_env` fixture that deletes all settings variables. | A developer's real `.env` or shell can never change a test result. No test needs `HF_TOKEN` or the network. |
| B44 | Thread-cap behaviour is tested in **subprocesses** with a scrubbed environment. | The caps apply at import time and the test process has already imported `app`; a subprocess is the only honest way to observe the first import. |

## Open / handed to other lanes

- `.env.example` should gain the knobs in **B40** (owner lane).
- Repo has no `pyproject.toml`/`ruff.toml`/`mypy.ini`. `ruff check` and
  `mypy --strict app/contracts app/settings.py` are clean as of this
  workstream, but the configuration that pins those rule sets is outside this
  lane. Recommend `[tool.mypy] strict = true` scoped to `app/contracts`,
  `app/providers`, `app/settings.py` per BUILD-BRIEF §1.
- `QualityFlag` membership is the quality gate's vocabulary (WS-E, Stage 1.5).
  Thresholds remain OPEN-QUESTIONS Q3; the enum is additive, so calibration
  does not force a schema bump.
- `supporting_doc_tags` is `tuple[str, ...]` for now. PRD P1-3 wants a closed
  tag set; WS-G should promote it to an enum, which is a backward-compatible
  narrowing only if the current values are a subset.
- `DocumentResult` has no categorisation field yet (P1-4, RB-IOS taxonomy
  absent per PLAN.md scope). WS-G's stub should add it as an optional `Field`
  so the schema change stays additive.
