# WS-E1 decisions — ingestion, bypass, quality gate (BUILD-BRIEF M3)

Judgment calls made while building and auditing `app/stages/s1_ingest.py`,
`s1b_bypass.py`, `s15_quality.py`, `app/stages/__init__.py`, their tests,
`tests/golden/**`, `docs/calibration/quality_gate.json` and
`tools/calibrate_quality_gate.py`. Same conventions as `docs/decisions-wsc.md`:
nothing here contradicts a PLAN.md invariant, and items marked **OWNER** need a
decision outside this lane.

This run **audited and completed** work an earlier run left uncommitted. The
modules were kept; three defects were fixed, two of which were latent rather
than test-visible.

---

## 1. Defects found and fixed

| # | Defect | Fix | Why it mattered |
|---|---|---|---|
| E1-1 | `open_pdf()` classified a PDFium load failure by string-matching PDFium's error message. `FPDF_GetLastError()` is a **process-global, sticky** slot: PDFium writes it on the failure paths that classify a failure and leaves the previous value untouched on the ones that do not. A zero-page document fails to load *without* recording why, so it inherited the previous failure's code. | `_prime_pdfium_error()` loads a known-not-a-PDF buffer immediately before the real load, planting `FPDF_ERR_FORMAT` as a sentinel; the password verdict is trusted only when the code read back is *not* that sentinel. `_load_pdfium_document()` holds a module lock across prime-then-load (PDFium is not thread-safe regardless). Falls back to the old "trust the code" behaviour if the sentinel cannot be established. | Once any encrypted PDF had passed through a worker, **every** subsequent malformed PDF was reported as `ENCRYPTED_PDF` for the life of that process. `BounceReason` drives the complainant-facing auto-reply (PRD P1-2), so the sender of a truncated attachment was told to remove a password that does not exist. Visible in the suite only as an order-dependent failure — `test_pdf_with_no_pages_is_malformed` passed alone and failed after the encrypted-PDF test. Regression test: `test_malformed_pdf_stays_malformed_after_an_encrypted_one`. |
| E1-2 | `test_infected_attachment_raises_before_anything_parses_it` asserted `pytest.raises(MalwareDetected)` over `ingest(b"")`, then contradicted itself in the body with `isinstance(exc, EmptyFileError \| MalwareDetected)`. | **The test was wrong, not the module.** Zero-length is checked before the scan, deliberately and per the module docstring: nothing can be infected by nothing, and `EMPTY_FILE` is a complainant-facing bounce while `MALWARE_DETECTED` must never generate a reply. Replaced with a parametrised test over three payloads a *later* stage would reject (disguised HTML, truncated PDF, corrupt PNG) that asserts `MalwareDetected` displaces the parser's error — which is what "before anything parses it" actually means — plus `test_zero_length_is_refused_before_the_scanner_is_asked`, which counts scanner calls. | The original test proved nothing about ordering; it only proved the empty check came first, which is the opposite of its name. |
| E1-3 | `character_profile()` classified Unicode categories `Mn`/`Mc` (combining marks) as `other`, i.e. as control/private-use/unassigned junk. Roughly 40% of running Devanagari is matras, virama and anusvara. | Marks now count as **letters** and vote with the script of their block (`script_of` switched from `str.isalpha()` to a category test). Marks in the script-**inherited** blocks (U+0300–036F and friends) count as letters but abstain from the script vote, returning the new `INHERITED_SCRIPT` (`Zinh`); the distribution denominator is the attributed count, not the letter count. | A clean Marathi/Hindi text layer measured **41% "unassigned characters"** and the junk guard refused the bypass. Every Devanagari digital PDF — the pages this service exists to read — paid for an unnecessary OCR pass, and the audit trail recorded a false reason for it. Found by `test_clean_devanagari_layer_is_not_junk`. NFD-decomposed Latin had the same problem via a different route, covered by `test_inherited_marks_abstain_from_the_script_vote`. |

The E1-1 primer costs a measured **6.8 µs per ingest** — a failed load of a
13-byte buffer — against ~200 ms to rasterise one page, so it is bought for
nothing.

**E1-4** — `estimate_skew` search range is ±20° while `skew_unrecoverable_deg`
is 25°, so the `SKEW_UNRECOVERABLE` branch is unreachable from measurement
alone at default settings. It is reachable (and tested) when an operator
lowers `QUALITY_SKEW_BOUNCE_DEG`, and a >20° page instead bounces on the
quality score, which is the same outcome with a less specific reason.
**OWNER / WS-E2**: either widen `skew_search_deg` to ~30° or drop the bounce
threshold to 20° so the two agree. Not changed here because either move
re-opens the Q3 calibration.

---

## 2. Test suites written

| File | Tests | Notes |
|---|---|---|
| `tests/unit/test_s1_ingest.py` | 58 (was 52) | Pre-existing; two failures fixed, three tests added (malware ordering ×3 params, zero-length precedence, malformed-vs-encrypted regression ×3 params). |
| `tests/unit/test_s1b_bypass.py` | 73 | New. Character classification, every junk rule from a string, geometry for all four `/Rotate` values, extraction contract (`language is None` — PLAN.md invariant 1), each `BypassVerdict`, and the whole-document driver including the typed-error passthrough. |
| `tests/unit/test_s15_quality.py` | 58 | New. Pages are **synthesised from primitives** (rows of dark rectangles on light paper), not rendered from a font and not loaded from the corpus: the defect under test is the only difference between two pages, and the suite needs no font, no Raqm and no corpus. |
| `tests/golden/test_adversarial.py` | 20 | New. `corpus/adversarial/MANIFEST.json` is parsed at run time and reconciled against the directory in **both** directions, so the manifest is the source of truth rather than a table copied into the test. Also covers the three `not_included_generate_locally` cases in memory. |
| `tests/golden/test_quality_calibration.py` | 13 | New. Cheap consistency checks on the committed report, plus a live re-measurement of every 7th corpus page. |

Full suite after this run: **1811 passed**, 0 failed, 0 skipped.

---

## 3. Calibration (OPEN-QUESTIONS Q3 — resolved)

### 3.1 It is real

The committed `docs/calibration/quality_gate.json` was produced by an ad-hoc
script that did not survive. It has been **re-derived from scratch** by
`tools/calibrate_quality_gate.py` over all 295 corpus pages and reproduces the
committed file exactly — every per-page verdict, score, repair list, the
confusion matrix, the per-profile breakdown, the bounce reasons, the flag
counts and the adversarial outcomes.

| # | Decision | Rationale |
|---|---|---|
| E1-5 | `tools/calibrate_quality_gate.py` is committed with `--write` / `--check` / `--sample N`. | A calibration nobody can re-run is a number somebody typed. `--check` is the assertion that the frozen file still describes the code; it belongs in CI next to the licence gate. New file, no other lane's territory. |
| E1-6 | `score_distribution` percentiles use nearest-rank (`numpy` `method="higher"`), not linear interpolation. | This was the *only* discrepancy in the first reproduction attempt, and resolving it in favour of the original was right: every reported percentile is then an actual page score traceable to a `doc_id` in the `pages` list. An interpolated p10 belongs to no page, which makes "which document sits at the tenth percentile of `heavy`?" unanswerable — the first question anyone asks when a profile drifts. |
| E1-7 | Q3 is calibrated on the **whole** corpus, not the WS-D locked split. | The locked split (A-2, BUILD-BRIEF M1) exists to stop a *model* being tuned to its test set. Six fixed numeric thresholds have no capacity to memorise 295 pages, and holding back 40% would halve the evidence behind the one boundary that matters. Recorded explicitly so it is a decision, not an oversight. |
| E1-8 | Each corpus page is assessed with a **fresh** `QualityGate`. | Near-duplicate detection is per document, and every corpus page is its own document. Sharing one gate would flag the six degradation profiles of the same page as duplicates of each other and pollute the flag counts. |

### 3.2 Final thresholds (unchanged from the frozen set)

| Knob | Value | Env override |
|---|---|---|
| `bounce_threshold` | 0.35 | `QUALITY_BOUNCE_THRESHOLD` |
| `laplacian_floor` / `_target` | 9 500 / 26 000 (ramped in log10) | `QUALITY_BLUR_LAPLACIAN_FLOOR` / `_TARGET` |
| `contrast_floor` / `_target` | 17 / 70 | `QUALITY_CONTRAST_FLOOR` / `_TARGET` |
| `effective_dpi_floor` / `_target` | 30 / 140 | `QUALITY_EFFECTIVE_DPI_FLOOR` / `_TARGET` |
| `coverage_floor` | 0.004 | `QUALITY_TEXT_COVERAGE_FLOOR` |
| `skew_repair_deg` / `skew_unrecoverable_deg` | 0.75 / 25.0 | `QUALITY_SKEW_REPAIR_DEG` / `QUALITY_SKEW_BOUNCE_DEG` |
| `min_page_dimension_px` | 200 | `QUALITY_MIN_PAGE_DIMENSION_PX` |
| weights (res/sharp/contrast/coverage/occl) | 0.20 / 0.45 / 0.25 / 0.06 / 0.04 | in-code (structure, not tuning) |
| `subscore_floor` | 0.05 | in-code |

### 3.3 Confusion matrix, 295 pages, seed 1729, generator 1.1.0

Rows are the corpus's `quality_expectation`; columns are the gate's verdict.

| expectation | accept | repair | bounce | total |
|---|---|---|---|---|
| clean | 66 | 29 | **0** | 95 |
| degraded | 26 | 124 | **0** | 150 |
| bounce | 0 | 0 | **50** | 50 |

By degradation profile:

| profile | accept | repair | bounce | total |
|---|---|---|---|---|
| clean | 45 | 0 | 0 | 45 |
| light | 21 | 29 | 0 | 50 |
| scan | 17 | 33 | 0 | 50 |
| heavy | 7 | 43 | 0 | 50 |
| photo | 2 | 48 | 0 | 50 |
| bounce | 0 | 0 | 50 | 50 |

(`clean` has 45 rather than 50 because `photo_evidence` is forced to the
`photo` profile and has no clean baseline — corpus `index.json` caveat.)

**Targets, all met:**

| target | required | measured |
|---|---|---|
| bounce-profile recall | ≥ 0.80 | **1.00** |
| clean/light false-bounce rate | 0.00 | **0.00** |
| scan/heavy/photo accept-with-repair | no bounces | **0 bounces**, 124/150 repaired (83%) |
| deviations (expected-bounce ≠ actual-bounce) | — | **0** |

**Why 0.35 is not arbitrary.** The highest-scoring `bounce`-profile page is
0.2950 and the lowest-scoring accepted page is 0.4146 (`photo`), so the
threshold sits inside an empty interval 0.12 wide, 0.055 above the worst
accepted class and 0.065 below the best rejected one.

| profile | min | p10 | median | max |
|---|---|---|---|---|
| clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| light | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scan | 0.9789 | 1.0000 | 1.0000 | 1.0000 |
| heavy | 0.4214 | 0.9783 | 1.0000 | 1.0000 |
| photo | 0.4146 | 0.4674 | 0.5662 | 0.7932 |
| **bounce** | 0.0818 | 0.1040 | 0.1533 | **0.2950** |

All 50 bounces are `QUALITY_SCORE_BELOW_THRESHOLD`; no page hit a hard floor.
Runtime 72.4 s for 295 pages, 245 ms/page, single-threaded-ish (cv2 capped at 2).

### 3.4 Adversarial inputs (BUILD-BRIEF M3: all 7 typed, no crashes)

| file | manifest expectation | measured |
|---|---|---|
| `zero_byte.pdf` | reject `EMPTY_FILE` | `EmptyFileError` |
| `truncated.pdf` | reject `MALFORMED_PDF` | `MalformedPDFError` |
| `not_a_pdf.pdf` | reject `MIME_MISMATCH` | `MimeMismatchError` |
| `html_disguised_as_pdf.pdf` | reject `MIME_MISMATCH`, script never evaluated | `MimeMismatchError` on the magic bytes; no parser, renderer or script engine sees the payload, and the error text carries a hex prefix only |
| `tiny_lowres.png` | bounce `RESOLUTION_BELOW_FLOOR` | bounce, score 0.0564 |
| `blank_page.png` | drop page `BLANK_PAGE` | bounce, score 0.0564, no OCR billed |
| `inverted.png` | repair auto-invert, then process | repair `auto_invert`, score 0.8355, output byte-exactly `255 − input` |

Generated locally per the manifest: `password_protected.pdf` → `EncryptedPDFError`
(never cracked, not even with an empty password); `pdf_5000_pages.pdf` → page cap
at 201 pages, enforced before a bitmap is allocated; `zip_bomb.zip` → refused as
`MIME_MISMATCH` (archives are not a supported type), with the ratio guard a
future expander must reuse tested separately against a PNG bomb.

---

## 4. Audit of the prior run's edits to committed files

Both are **purely additive** — no existing symbol, default or signature changed
— and both are consistent with their modules.

| file | change | verdict |
|---|---|---|
| `app/contracts/errors.py` | `+MalwareDetected` (and its `__all__` entry). Subclasses `PipelineError`, **not** `DocumentRejected`, with `code="MALWARE_DETECTED"`, HTTP 422, and `signature`/`scanner` folded into `context` via `setdefault`. | **Correct, and the inheritance choice is the load-bearing part.** Every `DocumentRejected` member is mirrored one-for-one into `BounceReason` and therefore into a complainant-facing auto-reply; an infected attachment must not produce a reply naming the signature that fired, because that is a free oracle for tuning the next attempt. `REJECTION_EXCEPTIONS` and `exception_for()` are untouched and still total over `RejectionReason`. Asserted by `test_malware_is_not_a_bounceable_rejection`. |
| `app/settings.py` | `+10` `QUALITY_*` fields, all with defaults and bounds, in the existing quality-gate block. | **Correct.** Every default equals the corresponding `QualityThresholds` default — now asserted by `test_defaults_match_the_committed_calibration`, so the two cannot drift silently. Only *tuning* knobs are exposed; the model structure (sub-score weights, advisory flag levels, analysis geometry) deliberately stays in the stage module, because changing those invalidates the calibration rather than retuning it. `Settings.model_fields` drives `conftest.SETTINGS_ENV_VARS`, so the new variables are cleared by `clean_env` automatically. |

---

## 5. Coordination and unresolved

| # | Item | Status |
|---|---|---|
| E1-9 | **No `pyproject.toml` / `pytest.ini` was created.** No WS-E1 test touches `models/`, so the `local_models` marker is not needed in this lane and registering it would have created a file for another lane to collide with. | Whichever lane first needs the marker owns the file. Suite currently runs with zero pytest config. |
| E1-10 | `_occlusion_share` understates a large solid occlusion. The illumination normaliser (morphological close ÷ divide) treats a big uniform dark blob as *background*, so a thumb covering 26% of a page measures ~7%. | The `OCCLUSION_DETECTED` flag still fires (7% > the 0.05 flag level) and the corpus `heavy` profile raises it on 4 pages, so the verdict is right and only the magnitude is compressed. Not changed: the fix is a different metric, which re-opens the whole calibration. **OWNER / WS-E2** if occlusion magnitude is ever keyed on. |
| E1-11 | Overall bounce rate on the corpus is **16.95%**, above the 5–15% operational band `docs/TUNING.md` will name. | Not a miscalibration: the corpus is 1/6 bounce-profile pages *by construction* (TEST-DATA-SPEC), which is a stress mix, not a mail-stream estimate. The band applies to production traffic. Worth restating in `docs/TUNING.md` so nobody retunes against the corpus rate. |
| E1-12 | `ROTATION_DETECTED` / `ROTATION_REPAIRED` are defined in `QualityFlag` but never raised. | Deliberate and now asserted (`test_measure_reports_orientation_anisotropy_without_acting_on_it`). Projection anisotropy does not separate upright from 90°-rotated across the ten document types — a form's column alignment is as strong as its line spacing — so the number is measured and reported, never acted on. Stage 2's text-orientation model owns the decision. |
| E1-15 | Two independent script tables now exist: `s1b_bypass.script_of` (needs coverage of the *wrong* answers — Grek/Cyrl/Hani are there because their presence is the junk evidence) and `providers/local.script_of_char` (WS-C, used by `s2_script`). | Left separate on purpose; they answer different questions and merging them would couple the junk guard to the recognition vocabulary. Checked: `script_of_char` is purely range-based with no `isalpha()` guard, so it does **not** carry the E1-3 combining-mark defect. No other module imports `s1b_bypass`, so E1-3's fix is contained. |
| E1-16 | `OPEN-QUESTIONS.md` Q3 is still marked OPEN. | **OWNER**: flip to RESOLVED citing §3 of this file. Not edited from this lane — `OPEN-QUESTIONS.md` is shared, and every workstream is landing into it concurrently. |
| E1-13 | Three `DTZ001` lint hits remain in `tests/unit/test_s1_ingest.py` (naive `datetime` in the soft-duplicate tests). | Intentional: `soft_duplicate_key` documents that the *caller* chooses the timezone and accepts naive values, so testing with them is the point. Pre-existing; left alone. |
| E1-14 | `mypy` reports cv2-stub friction in `s15_quality.py` and `import-untyped` for `pypdfium2`. | Pre-existing and out of scope: BUILD-BRIEF §1 scopes `mypy --strict` to `app/contracts` and `app/providers` only. `import pypdfium2.raw` adds one more instance of an error the existing `import pypdfium2` already produced. |
