# WS-D decisions — test corpus + evaluation harness

Judgment calls made while building `tools/generate_test_corpus.py`, `tools/eval_harness.py`,
`corpus/**` and `tests/corpus/**`. Owner merges the relevant lines into `OPEN-QUESTIONS.md`.

Lane: `tools/generate_test_corpus.py`, `tools/eval_harness.py`, `corpus/**`, `tests/corpus/**`,
this file. Nothing outside it was touched.

---

## 1. Generator — the fixes that were on the list

| # | Fix | What was done |
|---|---|---|
| a | Windows-safe fonts | `FALLBACK_LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"` deleted. `require_fonts()` runs once up front and raises `FontUnavailableError` naming every missing file and the role it serves. `load_font()` raises the same error rather than substituting a face. The Devanagari hard-fail is kept and its wording strengthened. A test asserts the string `DejaVu` no longer appears in the source. |
| b | Devanagari shaping guard | `tools/assets` is prepended to `PATH` (and `os.add_dll_directory`, guarded) in `_prepend_assets_to_dll_path()`, which runs **before** `from PIL import ...`. `require_text_shaping()` raises `TextShapingUnavailableError` if `features.check("raqm")` is false, quoting D6 and reporting whether `fribidi-0.dll` was found. Verified empirically: without the prepend, `features.check("raqm")` is `False` on this machine; with it, `True` (raqm 0.10.5, fribidi 1.0.9, harfbuzz 14.2.1). |
| c | UTF-8 writes | Every `write_text` call passes `encoding="utf-8"` (`*.gt.json`, `index.json`, `adversarial/MANIFEST.json`). A test decodes each artefact as UTF-8 and asserts the Devanagari is stored as real UTF-8, not `\uXXXX` escapes. |
| d | Global RNG side effect | `random.seed(rng.random())` removed. See §2 — it *was* load-bearing, so it was replaced rather than deleted. The dead double assignment at v1 lines 283-284 (`script, lang = ("Deva", ...), None` immediately overwritten) is gone. |
| e | `photo_evidence` forced profile | Behaviour kept. Documented in a comment on `FORCED_PHOTO_PROFILE`, in the `RENDERERS` table, and in `index.json` under `caveats.photo_evidence_forced_profile`. |

## 2. The global seed was load-bearing — audit result

**Do not delete it blindly.** `Canvas.text()` drew handwriting jitter from the *module-level*
`random`, not from the per-document `rng`. Removing `random.seed(...)` without rewiring
would have made every handwritten page non-reproducible while all the tests still passed
(nothing compared bytes across runs).

Replacement: a `Pen` dataclass carrying an explicit jitter RNG, seeded per document as
`random.Random(rng.random())` — the same value the v1 code fed to `random.seed()`, drawn at
the same point in the stream. So the jitter sequence is unchanged, the content/degradation
stream is unchanged, and the process-global RNG is never touched.

`tests/corpus/test_determinism.py::test_generation_does_not_touch_the_global_random_state`
locks this in by comparing `random.getstate()` across a generation run.

## 3. Deviation from the brief: Devanagari handwriting was rendering as garbage

**This is the one change outside the "exactly these fixes" list. Flagging it explicitly.**

The v1 jitter loop drew one *codepoint* at a time (`for ch in s: draw(ch)`), which bypasses
Raqm entirely. Every Devanagari matra, anusvara and virama was drawn in isolation and came
out as a dotted-circle placeholder. `राजेश कुमार शर्मा` rendered as `र◌ाज◌ेश क◌ुम◌ार शर◌्म◌ा`.
Verified visually on a crop before and after.

That affects `complaint_letter_mar` (13/13 lines) and `bilingual_form` (the Devanagari
values) — 60 of 295 pages — and it is precisely the failure mode `design/FONTS.md` refuses to
ship: *"a corpus full of □□□□ that silently trains a recogniser is worse than a corpus that
failed to build."* It also makes fix (b) incoherent: hard-failing on missing Raqm while
bypassing Raqm on every handwritten Devanagari line.

**Decision:** jitter is now applied per *grapheme cluster* (`grapheme_clusters()`), so each
cluster is drawn with one shaped `ImageDraw.text` call.

- For text with no combining marks — all Latin content in this corpus — `grapheme_clusters(s)`
  returns `list(s)`, so **Latin handwriting is byte-identical to the v1 algorithm**.
- Only Devanagari handwriting changes, and only to become correct.
- `--hand-shaping codepoint` reproduces the v1 behaviour exactly, if a v1 corpus is ever
  needed. Default is `cluster`; the mode used is recorded in `index.json`.
- `grapheme_clusters()` is deliberately simpler than UAX #29: combining marks (Mn/Mc/Me),
  ZWJ/ZWNJ and virama-joined conjuncts, which covers Devanagari and Latin. It does not
  handle emoji sequences or other Indic scripts. Stated in the docstring.

**Residual limitation, recorded in `index.json` as `caveats.hand_shaping`:** kerning and
ligation *across* clusters is absent on handwritten lines. Printed lines are drawn as whole
strings and are fully shaped. If handwritten-Devanagari CER ever looks suspiciously good or
bad, this is the first thing to check.

**If the owner wants v1 pixels instead:** regenerate with `--hand-shaping codepoint`. Nothing
else needs to change.

## 4. Generator — smaller calls

- `generator_version` bumped `1.0.0` → `1.1.0`. The generator changed; the version that
  produced a corpus has to say so.
- `generate_corpus()` extracted from `main()` so tests can drive it in-process and so the
  return value (the index payload) is testable. `main()` returns an exit code and catches
  `CorpusGenerationError` → prints and exits 2. No `sys.exit` inside library code.
- Typed errors: `CorpusGenerationError` base, with `FontUnavailableError`,
  `TextShapingUnavailableError`, `UnknownDocumentTypeError`. v1 used `sys.exit(str)`.
- Argument validation added for unknown profiles, `variants < 1` and unknown shaping modes —
  all typed, none silent.
- `index.json` gained `hand_shaping`, `types`, `variants`, `profiles`, `adversarial` and a
  `caveats` object. The original `caveat` string is unchanged and still present, so anything
  reading that key keeps working.
- `write_adversarial()` returns the file list so `index.json` records it.
- **Not changed:** `render_bilingual_form` still renders a complaint form and discards it.
  It draws no jittered text, so it consumes nothing from the pen RNG and costs only time.
  Left alone under "no behavioural redesign", with a comment saying why it is safe.

## 5. Corpus shape — 295 pages, not 300

`10 types × 5 variants × 6 profiles = 300` minus the 5 `photo_evidence` collisions = **295**,
which matches the spec's "≈290 labelled pages". `photo_evidence` has no `clean` baseline by
design. Any consumer computing per-type page counts must special-case it — said in
`index.json` and asserted in `test_corpus_shape.py`.

Counts verified: 9 types × 30 + photo_evidence × 25; 59 pages per variant; 95 `clean` /
150 `degraded` / 50 `bounce` quality expectations; `bilingual_form` is the only code-mixed
type; 295 JPEGs and 295 sidecars on disk, all referenced by `index.json`.

## 6. Bounding boxes outside the image on skewed profiles — expected, not a bug

`apply_rotation` uses `expand=False`, so the page is cropped while the ground truth keeps the
true geometry. On `bounce` (±17°) roughly a quarter of the lines have at least one corner
outside the frame. That is the honest label — clamping would corrupt the detection reference.
`test_clean_profile_bboxes_lie_inside_the_image` therefore asserts in-frame only for `clean`,
and a separate test asserts every bbox is 8 finite numbers on every profile.

Consequence for the harness: detection IoU on skewed profiles is computed on the
axis-aligned hull of a rotated quadrilateral, which **understates** IoU. Those numbers are a
floor, not an estimate. Recorded in the harness docstring.

## 7. Harness — dependency policy

Task said stdlib + rapidfuzz/numpy, no `app/` imports. Followed:

- No Pydantic, despite PLAN.md's "Pydantic v2 at every boundary". The harness has to run in a
  CI stage before app dependencies exist, and importing the app's contracts would make the
  scorer depend on the thing it scores. Boundary typing is instead `@dataclass` plus an
  explicit `validate_prediction()` that returns a list of violations, and typed exceptions
  (`EvalHarnessError` and five subclasses). **Owner: overrule if you would rather have the
  Pydantic consistency.**
- `rapidfuzz` is used for edit distance when importable, with an exact pure-Python DP
  fallback. `numpy` is not imported at all — percentiles are 6 lines of stdlib.

## 8. Harness — how the reference is built (the judgment calls that move numbers)

**Field reference comes from the ground-truth lines, not `gt["fields"]`.** `gt["fields"]` is
unusable as a primary reference: the generator strips `*_deva` keys, so a Devanagari page's
`bank_name` is `"Bank of Baroda"` while the page shows `बैंक ऑफ बड़ौदा`; and it carries keys
that were never rendered (`card_last_four` on a complaint form). Using it directly would
manufacture false negatives forever.

So: lines with a non-null `field` are authoritative. `gt["fields"]` contributes *additional
accepted spellings* for fields already on the page — which is exactly what lets an advisory
`value_en` (invariant 10) count as correct — plus a narrow recovery rule for identifiers that
are visibly embedded in a line but not their own field (`account_number` and `ifsc_code` in
the `bank_statement` header). The recovery rule is limited to a whitelist of deterministic
identifiers, requires length ≥ 6, and skips any value already claimed by another field name
(otherwise `mobile` and `complainant_contact` would double-count the same string).

Known gaps this leaves, all corpus limitations rather than harness bugs:

- `receipt`, `email_printout`, `cheque` and `photo_evidence` return `{}` from their renderers,
  so their `gt["fields"]` is empty; they contribute field references only through line links.
- `relief_sought` references a whole sentence, so a pipeline that extracts a phrase scores an
  FP. Harsh but honest for this corpus.
- `complainant_city` is aliased to the line's `complainant_address`.

**Escalation reference.** The corpus has no escalation label. `--escalation-reference`
defaults to `quality_or_hand`: `quality_expectation != "clean"` OR a handwritten-line majority
(≥ 50%). `quality` and `hand` are also selectable. The number is only comparable across runs
of the same setting; that is stated in the report and in the docstring. **This is a stand-in
and should be revisited once the router exists (relates to Q3).**

**A wrong value is both a false positive and a miss.** Claiming a field incorrectly costs
precision and recall. Counting only precision would let a confident wrong answer erase the
miss from the recall denominator.

**Recall by source is measured against the ground truth, not against what the source claimed.**
Otherwise every mechanism reports recall 1.0 and the table cannot answer "where are we losing
fields". Per-source recalls now decompose the overall recall (verified: 0.553 + 0.235 = 0.788
on the smoke run).

**Line matching** is greedy by IoU on axis-aligned boxes, then text similarity (≥ 0.5) for
leftovers so an implementation reporting no boxes still gets CER/WER. Unmatched GT lines are
scored as full deletions; unmatched predictions are counted as `spurious` and do not inflate
CER denominators.

**CER/WER** are micro-averaged (Σ edits / Σ reference length), computed on NFKC-normalised,
whitespace-collapsed text with case preserved. Field-value comparison is much more aggressive
(Devanagari digits folded to ASCII, `₹`/`Rs.` and separators dropped, casefolded) because
Stage 5 owns the normalised form and the metric asks whether the right value was found at all.

## 9. Harness — locked split

Variants **3 and 4** of every type (the generator numbers variants 0-4, so this is two of
five). `--split dev` is the default and covers variants 0-2. `--split test` and `--split all`
raise `LockedSplitError` and exit 3 unless `--final` is passed. The report always records
which variants were withheld, and the human-readable footer says so unless `--final` was used.

## 10. Verification performed

- Full corpus: `--out corpus --variants 5 --seed 1729` → 295 pages, counts as §5.
- Spot-checked `bilingual_form_v2_scan`, `complaint_letter_mar_v0_clean` and
  `complaint_form_deva_v3_bounce`: Devanagari present on Deva-typed docs, bboxes in frame on
  un-rotated pages, degradation lists match the profile matrix, line-level
  `script`/`language`/`is_handwritten` present.
- Visual check of printed and handwritten Devanagari crops before and after the shaping fix.
- Harness smoke run over all 295 pages with a perturbed-oracle prediction set: every table
  populated, CER rising monotonically with degradation (clean 0.000 → light 0.025 →
  scan 0.069 → heavy 0.198 → photo 0.290 → bounce 0.640), line recall falling the same way
  (1.000 → 0.520), latency percentiles per stage and per stage × profile.
- `tests/corpus`: 143 tests, all passing.

## 11. Open / for the owner

1. **Devanagari handwriting shaping (§3)** — the one deviation from the fix list. Regenerate
   with `--hand-shaping codepoint` if you want v1 pixels; otherwise no action.
2. **Escalation reference rule (§8)** — a stand-in until the router lands. Relates to Q3.
3. **Pydantic in the harness (§7)** — deliberately absent; overrule if consistency matters
   more than the harness running standalone.
4. **`relief_sought` and the empty-field renderers (§8)** — field metrics for `receipt`,
   `email_printout`, `cheque` and `photo_evidence` are thin. Fixing it means giving those
   renderers real `fields` dicts, which is a corpus change and outside "no behavioural
   redesign". Worth doing before M7 if field F1 by type matters.
5. **`gt["fields"]` carries English values for Devanagari pages.** Worked around in the
   harness; the clean fix is for the generator to emit both `value` and `value_en`. Also a
   corpus change, so not done here.
