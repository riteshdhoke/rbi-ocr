# WS-E2 decisions — vision stages (script ID, LID, page class, OCR, router, VLM)

Judgment calls, audit findings and measurements from building and then **auditing**
`app/stages/s2_script.py`, `s2b_lid.py`, `s3_classify.py`, `s4_ocr.py`, `s4_router.py`,
`s4b_vlm.py`, `tools/train_page_classifier.py`, `models/classifier/**`,
`docs/calibration/fusion*.json` and `tests/unit/test_s2*.py test_s3*.py test_s4*.py`.
Owner merges the relevant lines into `OPEN-QUESTIONS.md`.

Lane discipline: nothing outside that list was touched. `app/settings.py`,
`app/contracts/**` and `app/providers/**` are committed lanes and were read, not edited;
`tests/conftest.py` belongs to WS-B and was used as-is.

## 0. Context — this was a resumed run

A previous agent with the same task was killed mid-run by a laptop shutdown. Its stage
modules were in the working tree and imported cleanly; `tests/unit/test_s2_script.py` was the
only test file in this lane that existed. This run therefore **audited the existing modules
and wrote the five missing test files**, rather than rewriting working code. Three source
changes were made, all minimal, all listed in §6 with the evidence that prompted them.

| File | Status after this run |
|---|---|
| `s2_script.py` | audited (A-6 conformance) — unchanged |
| `s2b_lid.py` | audited — unchanged |
| `s3_classify.py` | 1 dead-code fix, 1 docstring correction (§6a, §6b) |
| `s4_ocr.py` | 1 unused-settings fix (§6c) |
| `s4_router.py` | audited (invariant 3, A-3) + A-3 reference revision (§4c) |
| `s4b_vlm.py` | audited — unchanged |
| `tools/train_page_classifier.py` | verified end to end (§3); lint-only edit |

Tests written this run: `test_s2b_lid.py` (53), `test_s3_classify.py` (48),
`test_s4_ocr.py` (31), `test_s4_router.py` (378, mostly the enumerated truth table),
`test_s4b_vlm.py` (38). With the pre-existing `test_s2_script.py` (27) the lane runs
**575 tests in ~7 s**.

---

## 1. Test-environment decisions

| # | Decision | Why |
|---|---|---|
| E2-1 | **No test in this lane loads a real model file**, so no `@pytest.mark.local_models` marker was needed and **no `pytest.ini`/`pyproject.toml` was created** (E1 owns that file if one is ever needed). | Every model this lane touches is mocked: LID through a stub that routes deliberately polluted raw fastText output through the provider's own `constrain_language_candidates`; OCR through `LocalProvider` with an injected fake ONNX session factory and zero-byte placeholder artifacts under `tmp_path`; the CNN through a stub raising the typed `ModelArtifactMissing` the real provider raises. The suite passes with `models/` empty. |
| E2-2 | The one test that reads `models/classifier/fusion.json` **skips with a reason** when it is absent. | It asserts only that the model-bundle copy and the committed `docs/calibration/` copy agree. An empty `models/` is a supported state, and the fallback copy is exercised by every other test in the file. |
| E2-3 | Corpus tests read **variants 0-2 only**, enforced by a module constant (`READABLE_VARIANTS`) and a test that asserts the loader refuses variants 3 and 4. | Variants 3-4 are the locked test split (BUILD-BRIEF M1). A test that tunes a stage against them is the same leak as training on them. |
| E2-4 | Stage-3 corpus scoring uses **one page per document type** (10 pages, module-scoped fixture, ~2.5 s), not the 147-page training set. | The unit suite checks that the *committed artifact* classifies; reproducing the training report is `tools/train_page_classifier.py --dry-run` (~34 s) and is recorded in §3 instead. |

---

## 2. Adjudication A-6 — Stage 2 conformance audit

**Ruling:** "Stage 2 uses Unicode-range majority vote over Tier-1 text (`detect_script_from_text`)
as the Phase-0/1 mechanism; the `script_id.onnx` loader stays as an optional future drop-in."

**Verdict: conformant.** Evidence:

| Requirement | Where it is met |
|---|---|
| Text vote is the mechanism | `script_from_lines()` delegates to `app.providers.local.detect_script_from_text`; `identify_script()` defaults `prefer_image=False`, so the vote is the only path a caller gets without asking for the other one. One table of Unicode ranges, in the provider. |
| Image classifier stays optional | Reached only with `prefer_image=True` **and** a provider **and** crops. `ModelArtifactMissing` either propagates (`allow_image_fallback=False`) or is recorded in `artifact_fallback_reason` while the vote runs. |
| No guessed script | `Zyyy` + confidence 0 is a legitimate outcome; `rec_head_for_script()` maps it to the Latin **fallback head** and the decision still carries `no_script_evidence`. A hint is used only when nothing voted, and never overrides a decisive vote. |
| The circularity is disclosed | Module docstring states it, and the mitigation is real: `s4_ocr.run_tier1` re-recognises once when the vote disagrees with the head (`head_mismatch`), capped at `MAX_RECOGNITION_PASSES = 2`. |

**Measured on the committed corpus** (variants 0-2, non-bounce, ground-truth text, 147 pages):
the vote is unanimous (confidence 1.000) on every monolingual type. `bilingual_form` is the
only type that is not — see §4c, which is where that finding leads.

**Two behaviours worth naming, both deliberate, both now pinned by tests:**

1. A page Tier 1 read *nothing* from votes `Zyyy`, is not confident, and reports
   `no_script_evidence` → the router sees `script_disagreement` and escalates it. That is the
   right answer (a page nobody could read is exactly what a second reader is for) and it is
   consistent with `s4b_vlm`'s own docstring.
2. A **short** page is not confident even when unanimous (`MIN_VOTING_CHARACTERS = 8`), so a
   cheque with a MICR line and three numerals cannot route a recognition head on three
   characters of evidence.

---

## 3. Page classifier — training verified, accuracy stated honestly

### 3a. The training run is legitimate

Re-run on 2026-08-07 with `.venv/Scripts/python.exe tools/train_page_classifier.py --dry-run`
and, separately, a full `train()` whose output was compared byte-for-byte with the committed
artifacts:

| Check | Result |
|---|---|
| Committed report reproduces | **Identical in every field** to a fresh run (no diffs across the whole JSON). |
| Committed coefficients reproduce | A fresh fit equals `docs/calibration/fusion.json` **exactly**. The fit is deterministic by construction: zero initialisation, full-batch gradient descent, no stochasticity. |
| The two `fusion.json` copies agree | `models/classifier/fusion.json` == `docs/calibration/fusion.json`, byte-for-byte. Pinned by a test. |
| Only variants 0-2 were used | `TRAIN_VARIANTS = {0, 1, 2}`; report says `[0, 1, 2]`; `sample_count = 147` equals exactly the number of `variant ∈ {0,1,2} ∧ profile ≠ bounce` documents in `corpus/index.json`. |
| Locked split untouched | Variants 3-4 are **118 documents** and none is loadable through `load_samples()`'s default. |
| Cost | 33.7 s for the shipped fit plus three leave-one-variant-out folds. |

### 3b. The numbers, and what they are not

| Metric | Value |
|---|---|
| Train (fit on variants 0-2, scored on variants 0-2) | **0.9728** |
| Holdout (fit on variants 0-1, scored on variant 2) | **0.9592** |
| Leave-one-variant-out | v0 1.0000, v1 0.9184, v2 0.9592 — mean **0.9592** |
| Majority-class baseline | 0.4898 (`supporting_doc`, 72 of 147) |
| Worst per-type | `complaint_letter_en` and `email_printout`, both 0.8667 |
| Train confusion | `complaint_letter` → `supporting_doc` ×2; `supporting_doc` → `complaint_letter` ×2 |

**The honest framing, which the tests now enforce rather than trust:**

* The holdout is a **variant**, not a layout. Variants differ in bank, amounts and
  complainant — *not* in document design. So 0.9592 measures robustness to content, and is an
  **optimistic** estimate of generalisation to a document the model has never seen the shape
  of. `test_the_holdout_is_named_for_what_it_is` asserts the report keeps naming the fold.
* `photo_evidence` **has no training examples at all**: every corpus page of that type is
  ground-truthed `supporting_doc`, so the class appears in `PageClassLabel` and in the
  weights and never in the data. The shipped classifier therefore *cannot* return it — its
  coefficients are ~0 and its intercept is −6.08. Two tests hold this: one on the report's
  `labels_without_training_examples`, one asserting no corpus page is predicted
  `photo_evidence`.
* The CNN coefficient slots are **exactly 0.0**. A test asserts that, and a second asserts
  that swapping uniform CNN scores for `[0.97, 0.01, 0.01, 0.01]` leaves the fused
  distribution unchanged — so no CNN that does not exist can take credit for a structural
  result, and `late_fusion` is the same code path either way.
* Test floors are set at **0.90** for train, holdout and LOVO mean, not at the observed value.
  Pinning 0.9728 would fail on every legitimate retrain; a floor is what "documented honestly"
  means operationally — the claim is "at least this", and it is separately asserted to beat the
  majority baseline by > 0.2.

### 3c. `unknown` is a refusal, not a class

`TRAINED_LABELS` deliberately excludes `unknown`; `MIN_CLASS_CONFIDENCE = 0.35` demotes a
coin flip to `unknown` and sets `below_confidence_floor`, and `_with_unknown` **rescales and
keeps** the trained scores so a reviewer sees what the page nearly was. Tested with an
explicitly uniform weight set rather than a hunted-for image, so the case is deterministic.

---

## 4. Stage 4 router — invariant 3 and adjudication A-3

### 4a. PLAN.md invariant 3 (never raw CTC confidence) — conformant

Asserted three ways, because a behavioural test alone can only prove the paths it took:

1. **AST inspection**: the module's syntax tree contains **no** attribute read named
   `confidence`. Prose mentions of "CTC confidence" are unaffected, so the documentation can
   keep explaining the rule the test enforces.
2. **Shape of the seam**: `EscalationSignals` has no confidence field for a future edit to
   reach — the router is not given `OCRLine` objects at all, and `extra="forbid"` refuses
   them. `quality_score` is carried for the audit trail; the *signal* is the boolean, so the
   threshold lives once, in Stage 1.5.
3. **Enumerated truth table**: 320 parametrised cases over
   `validator_failures × script_disagreement × quality_low × handwriting_ratio × anomalies`,
   scored by a **second, independent implementation** of the published weight table written in
   the test file. A hidden condition added to `_score` makes the two disagree.

The 3% shadow sampler ships in the same module (BUILD-BRIEF sec. 5). It is keyed on a BLAKE2b
digest of `doc_id[:page_no]`, so a shadow run is replayable from the corpus alone; measured at
0.0297 over 20 000 keys. An escalated page is never also shadow-sampled — a page already going
to Tier 2 cannot be a control for it — and the escalation-reason metric is incremented only for
real escalations, so a shadow run cannot inflate the rate it exists to measure.

### 4b. The threshold's named consequences (all tested)

One validator failure alone does **not** escalate (a complainant can write a wrong account
number, and a bigger model will not fix that); `script_disagreement` alone **does**;
`quality_score_low` alone does not (that is the `scan`/`heavy` majority of the corpus);
handwriting alone does not, even at ratio 1.0 (the handwritten complaint letter is the normal
case). Failures saturate at 3 and anomalies at 3.

### 4c. A-3 revisited — **a third fix, from a measurement**

A-3 accepted the harness's stand-in "until the router lands" and asked WS-E2 to revisit it.
The module already carried two corrections (exclude `bounce` pages, which never reach the
router; and stop treating `light`/`scan` as "should escalate"). The audit found a third:

**Measured** on variants 0-2, non-bounce, feeding ground-truth text through
`identify_script(recognised_with=<page's primary script>)` and then `route()`:

* `bilingual_form` is the **only** type that raises a script disagreement, and it raises
  `split_vote` on **15 of 15** pages → the router escalates every one. Devanagari holds
  28.4-29.8% of the page's characters, over `SECOND_SCRIPT_SHARE = 0.25`.
* Every other type: no disagreement, no escalation, on all 132 pages.
* Under the previous reference, **9 of those 15** (profiles `clean`/`light`/`scan`) were
  labelled `not_expected` — i.e. the harness would have charged the router a false positive
  for doing the right thing.

It *is* the right thing: Tier 1 recognises a whole page with **one** head, so the four
Devanagari value lines on a bilingual form were provably read with the Latin head, and the
A-6 second pass cannot fix it (there is no per-line head). Tier 2 is the actual remedy.

**Change:** `expected_escalation()` takes `is_code_mixed: bool = False`, and a code-mixed page
is `expected` at any profile. Additive, default-preserving. `corpus/index.json` carries
`is_code_mixed` per document, so the harness has the field; `LidResult.is_code_mixed` is the
runtime equivalent.

**Handoff:** the function lives in `app/` and the harness is standalone by A-2 (it must not
import `app`), so WS-D/WS-H should add `--escalation-reference router_v1` mirroring this
table: `bounce → excluded`; `code_mixed ∨ handwriting_ratio ≥ 0.5 ∨ profile ∈ {heavy, photo}
→ expected`; else `not_expected`.

---

## 5. Stage 2b, Stage 4b — audit notes

* **PRD P0-5 is enforced twice**, by design: the provider constrains the label set and
  `s2b_lid` re-checks it, so a provider bug cannot poison the derived page summary or the MT
  routing behind it. Tested with a stub that returns `eng` for a Devanagari line and is
  overruled. An unidentifiable line stays `None` **with a machine-readable reason**
  (`too_short`, `no_permitted_candidate`, `low_confidence`, `artifact_missing`, `empty_text`,
  `unknown_script`) — never the page majority.
* **BUILD-BRIEF M4's acceptance test now runs against the corpus**, not against invented
  lines: on `bilingual_form` variants 0-2, printed English label lines carry `eng`, every
  Devanagari line is handwritten and carries `mar`, and `is_code_mixed` is true, matching the
  ground truth's own flag. (Note for a reader of the M4 wording: the form also has handwritten
  **Latin** values — account numbers, dates — so the implication runs Devanagari → handwritten,
  not the converse.)
* **The code-mix margin is thin.** `bilingual_form` has 4 Devanagari lines of 24, a minority
  share of **0.1667** against `CODE_MIX_THRESHOLD = 0.15`. Any recalibration above 1/6 flips
  the corpus's headline code-mix case from true to false. Recorded here because the number
  looks safe and is not.
* **Tier-2 isolation holds.** `escalate_page` returns a typed `Tier2Result` for rate limits,
  unreachable endpoints, token-budget breaches, invalid output, absent artifacts and
  "this profile has no VLM" — the page ships at Tier-1 quality with the reason recorded.
  `TypeError`/`ValueError` still raise, because a corrupted index space must not ship
  silently. Each of the seven merge rules has its own test; the load-bearing one is that a
  matched line keeps its **Tier-1 `line_index`**, so a summary span taken before escalation
  still resolves afterwards (PLAN.md invariant 2).

---

## 6. Source changes made this run (all minimal)

**(a) `s3_classify._text_lines_from_mask` — dead `findContours` removed.** The function called
`cv2.findContours` and discarded the result, then `_line_boxes()` computed it again. One wasted
contour pass per page on the hot path. Proof it was inert: after removal,
`tools/train_page_classifier.py --dry-run` reproduces `train_accuracy = 0.9727891156462585`
unchanged.

**(b) `s3_classify.handwriting_score` — measured claims corrected.** The docstring reported
"n=2969 lines of which 520 are handwritten … AUC 0.855 … precision 0.49, recall 0.68" for
"the committed corpus (variants 0-2, non-bounce profiles, ground-truth line boxes)". Re-measured:

| Population | n | handwritten | AUC | precision@0.55 | recall@0.55 |
|---|---|---|---|---|---|
| Lines the function can score (score > 0) | 2 969 | 520 | 0.855 | 0.494 | 0.679 |
| **Every line on those pages** | 3 675 | 600 | **0.780** | 0.494 | **0.588** |

The original numbers reproduce **exactly** — but only on the scorable subset. The 706 excluded
lines (19%, 80 of them handwritten) score `0.0`, and `s4_ocr.annotate_handwriting` turns `0.0`
into `is_handwritten=False`, so the second row is what the shipped flag actually delivers.
Per-profile the same correction applies (`clean` 0.908 → the easy subset; `photo` 0.740).
Docstring now states both rows and names the exclusion. **No behaviour changed** — this is a
documentation defect, and the operational consequence is bounded exactly as the docstring
already said: the router weights handwriting at 0.5 and never escalates on it alone.

**(c) `s4_ocr.run_tier1` — unused settings read removed.** The function did
`active = settings if settings is not None else get_settings()` and never used `active`, so a
stage with no configuration was reading global settings (and could fail for it) to no purpose.
Replaced with an explicit `del settings` and a comment, matching `s3_classify`'s existing
pattern; the parameter stays for a uniform stage signature. `OCR_BATCH_SIZE` and the
recognition-head cache are the *provider's* settings.

**(d) `tools/train_page_classifier.py` — lint only.** Dropped five `# noqa: E402` directives
that suppress a rule this repo does not enable, and replaced them with one `# noqa: I001`
plus a comment explaining that `import app` **must** precede `import numpy` (PLAN.md
invariant 5) and is therefore not to be sorted. The file is now ruff-clean; the trainer's
output is unchanged.

---

## 7. Tunables that belong in `Settings` (handoff, not done here)

`app/settings.py` is a committed lane, so these stayed module constants. Every one is
overridable per call today:

| Constant | Module | Value | Suggested setting |
|---|---|---|---|
| `MIN_SCRIPT_CONFIDENCE` | `s2_script` | 0.60 | `SCRIPT_MIN_CONFIDENCE` |
| `MIN_VOTING_CHARACTERS` | `s2_script` | 8 | `SCRIPT_MIN_VOTING_CHARS` |
| `SECOND_SCRIPT_SHARE` | `s2_script` | 0.25 | `SCRIPT_SECOND_SHARE` — drives §4c |
| `MIN_LID_CHARACTERS` / `MIN_LID_CONFIDENCE` | `s2b_lid` | 4 / 0.20 | `LID_MIN_CHARS` / `LID_MIN_CONFIDENCE` |
| `MIN_CLASS_CONFIDENCE` | `s3_classify` | 0.35 | `PAGE_CLASS_MIN_CONFIDENCE` |
| `HANDWRITING_SCORE_THRESHOLD` | `s3_classify` | 0.55 | `HANDWRITING_THRESHOLD` |
| `ESCALATION_SCORE_THRESHOLD` | `s4_router` | 2.0 | `ESCALATION_SCORE_THRESHOLD` |
| `SIGNAL_POINTS` | `s4_router` | see table | keep in code; it is a rule, not a knob |
| `DEFAULT_IOU_THRESHOLD` | `s4b_vlm` | 0.5 | keep — must equal the harness's detection IoU |

---

## 8. Unresolved / open

| # | Item | State |
|---|---|---|
| E2-U1 | `models/script/script_id.onnx` does not exist (decision C7). Stage 2 runs the A-6 text vote; the image path is written, tested against a stub, and unexercised in production. | Open by design. A drop-in closes the A-6 circularity properly. |
| E2-U2 | `models/classifier/page_classifier.onnx` (MobileNetV3-Small) does not exist. The CNN slots are fitted inert, so dropping one in **requires a retrain** — the zeros are not a placeholder that starts working. | Open by design; the retrain path is `tools/train_page_classifier.py`. |
| E2-U3 | `photo_evidence` is unlearnable from this corpus (§3b). | Needs corpus ground truth to distinguish it, or the class should be dropped from `TRAINED_LABELS` until it can be. Owner call. |
| E2-U4 | The handwriting heuristic is measured against a handwriting **font** with per-cluster jitter (adjudication A-1), not against pen-on-paper. Threshold 0.55 is provisional. | Re-cut on `iiit-indic-hw-words` before any claim about real handwriting. |
| E2-U5 | `--escalation-reference router_v1` for the eval harness (§4c handoff). | WS-D/WS-H. |
| E2-U6 | Q3 quality thresholds landed in WS-E1 after this router was written; `quality_score_low` is consumed as a boolean, so the two are decoupled, but the *rate* at which E1 sets it has not been measured against this router's 2.0 threshold. | Needs one full-corpus run (WS-H M7). |
| E2-U7 | Three tests fail in **other** lanes (`test_s1_ingest.py` ×2, `test_s5_validators_properties.py` ×1) and were present before this run. | Not WS-E2. Reported to the orchestrator. |
