# WS-C decisions — provider layer + residency gate (BUILD-BRIEF M2)

Judgment calls made while building `app/providers/**`, `tools/fetch_models.py`,
`tests/unit/test_providers*.py` and `tests/contract/**`. Owner merges the
relevant ones into `OPEN-QUESTIONS.md`. Nothing here contradicts a PLAN.md
invariant; where a choice tightens one, that is stated. Items marked
**OWNER** need a decision outside this lane.

## Protocol and shared vocabulary (`app/providers/base.py`)

| # | Decision | Rationale |
|---|---|---|
| C1 | The layer's image currency is a **NumPy `uint8` array**, `(H, W)` grayscale or `(H, W, 3)` **BGR** (OpenCV convention). `to_page_image()` converts a PIL image. | BUILD-BRIEF §3.1 writes `Image` without saying which. Detection pre-processing, the quality gate and the classifier are all OpenCV/ONNX, so a PIL object would force a conversion on the hottest path. **WS-E should pass arrays**; `to_page_image()` exists so a PIL-holding caller does not hand-roll the RGB→BGR flip. Aliased under `TYPE_CHECKING` so `import app.providers` pulls in no NumPy. |
| C2 | Two new error classes live in `app/providers/base.py`, not `app/contracts/errors.py`: `ProviderMethodNotSupported` (501) and `ModelArtifactMissing` (503). | `app/contracts/` is WS-B's lane. Both are provider-layer concepts. `ModelArtifactMissing` subclasses `ProviderUnavailable`, whose WS-B docstring already names "the local weights are missing". **OWNER**: if these belong in the shared taxonomy, moving them is a re-export. |
| C3 | `ProviderMethodNotSupported` is distinct from `ProviderUnavailable`. | "Cannot, ever" and "should work, currently cannot" have different remedies and different retry semantics. The message names the provider that *does* implement the method, so a routing bug reads as a routing bug. |
| C4 | The per-job budget is **reserve-then-settle**, not a running total. `JobBudget.reservation()` releases on any exception. | The true cost is known only from the response's `usage`, but the ceiling must be enforced *before* the request leaves. Reserve/settle gives both, and makes a failed call cost nothing. In-flight reservations count against the ceiling, so two concurrent calls cannot each see the full remaining budget. |
| C5 | `JobBudget` is thread-safe and shared across `extract_fields_batch`. | It is the one piece of mutable state the gather shares. Tested with 40 concurrent reservations against a budget that admits exactly 10. |
| C6 | `PROVIDER_METHODS` is a module constant, asserted equal to the Protocol's members. | Lets the contract suite enumerate the surface without reflection games, and fails loudly if a ninth method is added without thought. |

## Local provider (`app/providers/local.py`)

| # | Decision | Rationale |
|---|---|---|
| C7 | **`detect_script` has no approved artifact.** The loader expects `models/script/script_id.onnx` and raises `ModelArtifactMissing` naming `detect_script_from_text()` as the Phase-0 path. | OPEN-QUESTIONS Q1 covers *OCR* weights; nothing in the approved set is an image-space script classifier, and BUILD-BRIEF §0.3 forbids inventing model behaviour. `detect_script_from_text()` (Unicode-range majority vote over Tier-1 text) needs no weights and is exact wherever the text is. **OWNER / WS-E**: either approve a script-ID artifact or have Stage 2 vote on Tier-1 output. This is a real gap, deliberately visible. |
| C8 | `is_handwritten` is `False` unless a `handwriting_detector` callable is injected. | There is no approved handwriting-detection artifact either. A heuristic here would be a silent default on a field the escalation router and the bilingual-form test both read. The hook is tested; the default is documented as a gap. **OWNER / WS-E**: needs an owner. |
| C9 | Recognition width buckets are `(80, 160, 240, 320, 480, 640)` at a fixed height of 48, with the widest edge acting as a **clamp**. | Six buckets keep padding waste low without fragmenting batches. Clamping (rather than growing the tensor) matches PaddleOCR's own handling of over-long crops. |
| C10 | `plan_recognition_batches()` is pure over `(height, width)` pairs, separate from `pad_crops_to_batch()`. | Invariant 4 is the throughput-critical rule and had to be testable without a weight file. Ten unit tests cover coverage, arity, uniform bucket width, determinism and order preservation; an integration test asserts the actual session sees `[16, 4]` for 20 crops rather than 20 calls. |
| C11 | `batch_size < 1` raises, with a message naming invariant 4 and `tools/bench_ocr_cpu.py`. | The looped path is legitimate exactly once, in the benchmark. Making the error say so is cheaper than a review catching it later. |
| C12 | DB unclip is a closed-form rectangle offset (`area * ratio / perimeter`), **not** `pyclipper`. | A detected text line is a rectangle by construction, where the two agree exactly. Avoids adding a dependency (and its licence review) for a case with an analytic answer. |
| C13 | `ctc_greedy_decode` raises when the class count disagrees with the dictionary length. | A silently mismatched dict file is the single most common way a rec head produces confident nonsense, and it is invisible in the output. |
| C14 | Lines are emitted in reading order (rows within a 10 px tolerance, then left-to-right) with contiguous `line_index` from 0. | `line_index` is the handle the extractive summary selects by (invariant 2). A two-column label/value form must not interleave. |
| C15 | `constrain_language_candidates()` returns `language=None` when nothing survives the script constraint — never the first permitted member. | PRD P0-5 constrains the label set; guessing within it would be the silent default BUILD-BRIEF §0.5 forbids. Stage 2b decides what an unlabelled line means. |
| C16 | `SCRIPT_LANGUAGES` includes eight scripts beyond Latn/Deva (Beng, Guru, Gujr, Orya, Taml, Telu, Knda, Mlym). | Additive, costless, and matches BUILD-BRIEF M4's "adding Kannada later must be a model drop". `Deva` is exactly `{hin, mar, kok, nep}` per PRD P0-5; `Latn` is `{eng}`. |
| C17 | `_default_fasttext_loader` uses `fasttext-predict`, not `fasttext-wheel` (BUILD-BRIEF §1). | Already pinned in `requirements.txt` by WS-A: `fasttext-wheel` has no cp312 Windows wheel. Same `load_model`/`predict` surface, MIT-licensed. |
| C18 | IndicTrans2 is driven through SentencePiece + CTranslate2 directly, with `INDICTRANS2_TAGS` mapping ISO 639-3 to upstream tags (`kok` → `gom_Deva`, `nep` → `npi_Deva`). | `IndicTransToolkit` is not installed and is an extra dependency + licence review. Its normaliser and entity-placeholder pass are **not** applied — WS-E's sentinel masking (invariant 10) already owns entity protection. **OWNER**: recorded as a quality risk to measure in the eval harness, not a blocker. |
| C19 | `translate()` refuses any direction other than indic→`eng` with `ProviderMethodNotSupported` carrying `src`/`tgt` in context. | `models/mt` holds the `indic-en` dist-200M direction only. Returning the input unchanged, or picking a different model, would both be silent. |
| C20 | Loaders are lazy; construction performs **no I/O**. `cv2`, `onnxruntime`, `ctranslate2`, `sentencepiece` and `fasttext` are imported at first use. | The whole test suite, the API process and the contract suite must work on a machine with an empty `models/`. Also keeps `import app.providers` cheap and reduces the number of places that can drag a numeric library in ahead of the thread caps (invariant 5). |
| C21 | `RecHeadCache` takes an injectable `clock`. | TTL behaviour is asserted without sleeping. |
| C22 | `late_fusion()` and `FusionWeights` are public and pure; `fusion.json` is the on-disk contract. | WS-E can train and retune the fusion without touching the provider, and the fusion is unit-tested without the CNN. |

## HF API provider (`app/providers/hf_api.py`)

| # | Decision | Rationale |
|---|---|---|
| C23 | `openai` client is constructed with `max_retries=0`. | Our `RetryPolicy` must be the only retry loop, or the backoff schedule under test is not the schedule in production. |
| C24 | Backoff uses **proportional** jitter (`base × [1−j, 1+j]`, j=0.25), not full jitter, and `rand` is injectable. | Keeps a job's worst-case latency reasonable to state while still breaking up the herd a chord fan-out creates. `rand=lambda: 0.5` makes the schedule exactly assertable. |
| C25 | A `Retry-After` header wins over the computed delay but is capped at `max_retry_after_seconds` (30 s). | A server asking for ten minutes must not silently hold a worker for ten minutes. |
| C26 | 429 and 5xx retry; any other 4xx does not. | A 4xx other than 429 is a request bug; retrying it burns budget and hides the defect. |
| C27 | Validator-claimed spans are replaced with `[[MASK_n]]` **inside `render_lines()`**, the last point before serialisation. | There is then no code path on which an identifier value crosses the network. Sentinels are numbered by `(line_index, start)` so the prompt — and therefore the cache key — is deterministic. ASCII by design: exotic delimiters tokenise unpredictably and some models transliterate them. |
| C28 | `build_extraction_prompt()` sorts `target_fields`. | The prompt *is* the cache key; an unordered set makes every run a miss. |
| C29 | A returned field outside `ctx.target_fields` is **dropped**; everything else invalid (prose summary, hallucinated line index, empty-string value, deterministic field name) **raises** and triggers the repair turn. | Volunteered noise should not burn a repair turn. A contract violation should. |
| C30 | Exactly one repair turn (OPEN-QUESTIONS Q5 baseline); the second failure raises `ProviderResponseInvalid`. The bad answer is echoed back and the error stated imperatively. | Q5's recommendation, implemented literally. "Explain the mistake" invites prose. `ProviderResponseInvalid` is the class PRD P0-8's schema-validity metric counts. |
| C31 | A cache hit that no longer validates is **invalidated**, not served. | The only way that happens is a contract change; serving the stale answer would hide it until production. |
| C32 | A cache hit costs **zero** budget and zero tokens. | This is why the cache makes Phase 0 iteration feel local, and it is asserted as a test. |
| C33 | `estimate_tokens()` is `ceil(len/3)`, deliberately pessimistic. | Devanagari splits hard under byte-level BPE. Over-reserving costs headroom; under-reserving spends budget already promised elsewhere. The reservation is settled from real `usage`. |
| C34 | `extract_fields_batch` uses `AsyncOpenAI` + `Semaphore` + `gather`, and runs its own loop; if a loop is already running it hands the coroutine to a worker thread. | BUILD-BRIEF §3.1: the provider owns the concurrency. A FastAPI handler must not have to know. Tested from inside a running loop. |
| C35 | A batch failure **propagates**; partial results are discarded. | A half-filled list would let the assembler publish an envelope that looks complete. A budget breach or a dead endpoint is a job-level fact. |
| C36 | `translate()` on this provider is **off by default** (`enable_translate=False`). | `value_en` carries `mt_engine` precisely so a consumer can tell IndicTrans2 from a paraphrase (invariant 10). Enabling it must be a deliberate act. |
| C37 | The VLM model id is the module constant `DEFAULT_VLM_MODEL = "PaddlePaddle/PaddleOCR-VL"`, overridable by constructor argument or a `VLM_MODEL` environment variable. | **OWNER**: OPEN-QUESTIONS Q2 is still open and the id is **unverified against the HF router**. `Settings` has no `VLM_MODEL` field and `app/settings.py` is WS-B's lane — recommend adding `VLM_MODEL` (and optionally `PROVIDER_MAX_CONCURRENCY`) there so the env-var fallback can be deleted. |
| C38 | `json_mode` is a constructor flag (default on). | OPEN-QUESTIONS Q5 leaves router `response_format` support undiscovered. If a bake-off model rejects it, flip the flag: Pydantic validation and the repair retry still apply. |
| C39 | `reset_budget(job_id=...)` swaps in a fresh budget; the provider is otherwise long-lived. | Phase 0's orchestrator is single-job. **WS-F**: Phase 1 should construct one provider per job, or pass `budget=` explicitly, rather than relying on `reset_budget`. |
| C40 | `inject_truststore()` runs in `__init__`, before either client is constructed. | OPEN-QUESTIONS D7. Idempotent, thread-safe, and never disables verification. |

## Cache (`app/providers/cache.py`)

| # | Decision | Rationale |
|---|---|---|
| C41 | `CACHE_SCHEMA_VERSION` is **both** hashed into the key and stored as a column; a row at another version is a miss. | Belt and braces: the hash means a bump cannot be defeated by an existing database file, the column means `purge_other_versions()` and disk accounting are possible. |
| C42 | Fields are joined with `\x1f` (Unit Separator) before hashing. | `("ab","c")` and `("a","bc")` must not collide. Asserted. |
| C43 | Parameters are canonicalised (`sort_keys`, no whitespace, `ensure_ascii=False`), with a `default=` fallback for non-JSON values. | Two dict literals in different order are one entry. A stray `Path` in params must not blow up key construction. |
| C44 | One connection behind an `RLock`, `check_same_thread=False`, WAL for file-backed caches. | `:memory:` must behave like a file for tests, which per-thread connections break. Write contention here is nowhere near the bottleneck. |
| C45 | The cache stores the response as an **opaque string** and never parses it; the caller revalidates every hit. | A downstream schema change cannot make the cache serve something that no longer validates. |
| C46 | No TTL and no eviction. | A model call is a pure function of its key. Redeploying a model id is `invalidate_model()`; the file lives under the gitignored `.cache/`. |

## Factory and gates (`app/providers/factory.py`)

| # | Decision | Rationale |
|---|---|---|
| C47 | Profiles are **composed**: `laptop` and `gpu_node` are a `CompositeProvider` routing `{extract_fields, extract_fields_batch, ocr_page_heavy}` to `hf_api` and the other five to `local`. `cpu_node` is `LocalProvider` alone. | PLAN.md D1 puts OCR/LID/MT on the local CPU everywhere while ≥0.9B weights are never downloaded. The split is one `REMOTE_METHODS` frozenset, so it is auditable in one place and asserted by a contract test. |
| C48 | Delegation is eight explicit forwarding methods, not `__getattr__`. | A typo becomes an `AttributeError` at review time rather than a silent fallthrough in production. |
| C49 | The residency check runs **before** the registry lookup and before any construction. | A violating configuration must not open a socket, a database or a model file on its way to being refused. Asserted with a builder that raises if called. |
| C50 | The refusal message is `Settings.residency_violation_reason` (WS-B), not a literal here. | Written once; the gate and its test cannot drift apart. |
| C51 | The synthetic-corpus gate accepts a sibling `<stem>.gt.json` **or**, as a fallback, a directory-level `MANIFEST.json`, both requiring `"synthetic": true`. | The task specifies the sibling sidecar. The committed adversarial set (`corpus/adversarial/`) carries no per-file ground truth *by construction* — a zero-byte PDF has no lines — and declares provenance in a directory `MANIFEST.json` instead. Without the fallback, WS-E's M3 adversarial tests could not load their inputs under the default profile. Both forms are positive declarations; there is no third form and no flag. A manifest blesses only its own directory (asserted). **OWNER**: confirm this widening. |
| C52 | A malformed, unreadable or non-`true` sidecar is a **refusal**, not a pass. | Failing open on a parse error is exactly how a real scan gets processed under the laptop profile. Parametrised over four malformed shapes. |
| C53 | Under a non-remote profile the corpus gate is a no-op beyond the existence check. | Processing real complaints is what an air-gapped node is *for*. The gate is about where text goes, not about being strict. |
| C54 | `require_synthetic_corpus_file` raises `FileNotFoundError` for an absent path. | There is no `RejectionReason` for "file does not exist" — that is an ingestion concern (Stage 1, WS-E) and this helper is not ingestion. |
| C55 | `provider_profile` is a `Gauge` labelled by profile, 1 for active and 0 for the rest. | Lets an alert fire on `provider_profile{profile="laptop"} == 1` in an environment that should never be on it. An info metric could not. |
| C56 | The banner warns on `is_remote_profile` and on `has_stray_remote_credential`; the token is never logged. | WS-B decision B37 asked WS-C to surface the stray-credential flag. Asserted that the secret value does not appear in the log. |
| C57 | `build_laptop_provider` raises `ConfigurationError` when `HF_TOKEN` is absent. | WS-B's settings deliberately do not raise; the provider that needs it must fail at startup, where an operator sees it, not on the first extraction. |

## Downloader (`tools/fetch_models.py`) — **not run**

| # | Decision | Rationale |
|---|---|---|
| C58 | **OPEN-QUESTIONS Q1 → option (b).** Pre-converted Apache-2.0 ONNX (default repo `SWHL/RapidOCR`) rather than installing paddlepaddle + paddle2onnx. | Avoids an oversized wheel and a licence review for a one-off conversion, per PLAN.md's OCR-engine note. |
| C59 | The tool **discovers** files by listing the repository and matching ordered regex patterns; it never hard-codes a path. | The internal layout of that repository is not something this brief can assert (BUILD-BRIEF §0.3). A reorganisation upstream produces a clear "not found" instead of a 404 on a guessed path. |
| C60 | A missing **Devanagari recognition head** is reported, not fatal. `rec_deva` is deliberately absent from `OCR_REQUIRED`. | **OWNER — this is the live half of Q1.** PP-OCRv5's published multilingual recognition heads do not obviously include Devanagari, and PaddleOCR's Devanagari head is a v3-era model. The loader in `app/providers/local.py` is written for `models/ocr/rec_deva.onnx` regardless and raises a typed `ModelArtifactMissing` naming the gap. Remedies, in preference order: (a) find a pre-converted Devanagari head in another Apache-2.0 redistribution (`--ocr-repo`); (b) convert `devanagari_PP-OCRv3_mobile_rec` with paddle2onnx on a machine that can take the paddlepaddle wheel; (c) route Devanagari pages through Tier 2 and record the CER cost. Devanagari CER is the headline metric of this system, so (c) is a last resort. |
| C61 | **`lid.176.ftz` is CC-BY-SA-3.0 — outside the PLAN.md invariant 12 allowlist.** The tool downloads it by default, flags it loudly, and records it in `models/PROVENANCE.json`; `--strict-licence` refuses it. | **OWNER.** Invariant 12 governs *code dependencies* (`pip-licenses` in CI); this is a model file. CC-BY-SA is neither GPL/AGPL nor non-commercial, so it clears the stated prohibitions, but share-alike and attribution attach to the file. Same handling as OPEN-QUESTIONS D6's LGPL fribidi DLL: recorded for the procurement trail rather than silently accepted. If the owner rejects it, `py3langid` (BSD-3) or a CLD3 wheel are the alternatives, at some accuracy cost on short Indic lines. |
| C62 | IndicTrans2 is **converted locally** (`ctranslate2.converters.transformers --quantization int8`) rather than pulled pre-converted. | No pre-converted CT2 build of `ai4bharat/indictrans2-indic-en-dist-200M` is verifiable from here, and BUILD-BRIEF §0.3 forbids guessing one. Cost: ~800 MB fp32 intermediate (discarded), ~1.5 GB free space needed, a few minutes of CPU. `--mt-repo` overrides the source if a trusted pre-converted build is identified. |
| C63 | Every run writes `models/PROVENANCE.json`: source, resolved revision, SHA-256, size, declared licence, allowlist status, and problems. | The audit trail an air-gapped deployment needs, and it makes "are these the same bytes as last time?" answerable. |
| C64 | The tool imports its destination paths from `app.providers.local`. | The downloader and the loader cannot drift; asserted by a test. |
| C65 | `--hf-home` sets `HF_HOME` and downloads use `local_dir`/`cache_dir` explicitly. | PLAN.md WS-I item 3: nothing writes to `~/.cache/huggingface` implicitly. |

## Tests

| # | Decision | Rationale |
|---|---|---|
| C66 | `MockProvider` lives in `tests/contract/mock_provider.py` and builds **real contract objects**. | A contract change breaks the mock loudly instead of letting the suite drift. `tests/contract/conftest.py` puts the directory on `sys.path` so test modules import it by name. |
| C67 | HTTP is faked at the **transport** layer (`httpx.MockTransport`), not by patching `openai`. | The real client, the real exception hierarchy and the real retry loop are exercised; only the socket is replaced. No network, no token, no HF account. |
| C68 | `sleep` and `rand` are injected, so backoff delays are asserted exactly (`[0.5]`, `[0.5, 1.0]`) and the suite stays fast. | A retry test that actually sleeps is a retry test nobody runs. |
| C69 | The batched-recognition path is tested end to end against fake ONNX sessions that record their batch sizes. | Invariant 4 is asserted where it can actually regress — at the session boundary — not only in the planner. |
| C70 | A forward guard asserts no module under `app/stages/` imports `asyncio`; it skips while that directory does not exist. | BUILD-BRIEF §3.1's rule needs an enforcement point before WS-E writes the code it constrains. |
| C71 | A test loads real files out of the committed `corpus/` through the synthetic gate. | The gate must accept what WS-D actually generated, not a theory of it. |

## Handed to other lanes

- **WS-B / owner (`app/settings.py`)**: add `VLM_MODEL` (C37); consider
  `PROVIDER_MAX_CONCURRENCY` and `PROVIDER_RETRY_MAX_ATTEMPTS`, currently
  constructor arguments with defaults.
- **WS-E**: pass NumPy BGR arrays to the provider (C1); own script detection
  (C7) and handwriting detection (C8); produce
  `models/classifier/page_classifier.onnx` + `fusion.json`; do **not** import
  `asyncio` in stage modules (C70).
- **WS-F**: one provider per job or an explicit `JobBudget` (C39); call
  `close_provider()` on shutdown; scrape `provider_profile`,
  `provider_api_*` and `provider_cache_events_total`.
- **WS-H**: the SLM bake-off should record whether the router honours
  `response_format` (Q5, C38) and settle the VLM id (Q2, C37).
- **Owner**: confirm C51 (directory-manifest fallback), C60 (Devanagari head)
  and C61 (fastText CC-BY-SA).
- Repo still has no `pyproject.toml`/`mypy.ini`. `ruff check` and
  `mypy --strict app/providers` are clean as of this workstream;
  `ruff format` is **not** clean repo-wide (WS-B and WS-D files reformat too),
  so it is not enforced here either.
