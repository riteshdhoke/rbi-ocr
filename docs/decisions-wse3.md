# WS-E3 decisions — validators, SLM gate, translation, assembly

Judgment calls made while building and auditing `app/stages/s5_validators.py`,
`app/stages/s5_slm.py`, `app/stages/s6_translate.py`, `app/stages/assemble.py`,
`app/resources/**` and their tests. Owner merges the relevant lines into `OPEN-QUESTIONS.md`.

Lane: the four stage modules above, `app/resources/**`, `tests/unit/test_s5*.py`,
`tests/unit/test_s6*.py`, `tests/unit/test_assemble.py`, this file. One additive edit outside
the lane is flagged in D-E3-6.

This pass was an **audit and completion** of an interrupted build, not a rewrite. Two
implementation changes were made (D-E3-3, D-E3-4); everything else is tests and documentation.

---

## D-E3-1 · The failing Luhn property test: the test was wrong, the implementation was right

`tests/unit/test_s5_validators_properties.py::TestLuhnProperties::test_transposing_adjacent_unequal_digits_is_detected`
failed with `hypothesis.errors.FailedHealthCheck: filter_too_much` — not with a property
violation. The cause was `assume(luhn_check(pan))` over uniformly random digit strings: only
one string in ten satisfies Luhn, so nine in ten examples were discarded and hypothesis
(correctly) refused to keep generating.

**Both sides were checked before deciding which to change.**

*The implementation is correct.* Adjacent positions always have opposite parity from the
right, so transposing them trades `double(x) + y` for `double(y) + x`. The checksum moves by
`f(y) − f(x)` where `f(z) = double(z) − z`, i.e. `f(z) = z` for `z ≤ 4` and `z − 9` above it.
Modulo 10 that map is `0 1 2 3 4 6 7 8 9 0` — injective except at 0 and 9. So `09 ↔ 90` is the
**only** transposition Luhn cannot see, and the test's existing `abs(a − b) == 9` carve-out is
exactly that pair and nothing more. Confirmed empirically over 20 000 constructed PANs and
every adjacent swap in each: zero undetected transpositions outside `0 ↔ 9`.

*The property is correct.* It is not overbroad; the carve-out is a proof, not a concession.

**Fix (test side):** generate Luhn-valid numbers by construction instead of filtering for
them. `_luhn_complete(body)` appends the check digit, so the strategy is total, no example is
discarded, the health check has nothing to complain about, and the shrinker still sees the
whole domain. The check digit is computed with explicit arithmetic rather than by trying ten
candidates against `luhn_check`, so the transposition property is not tested against its own
implementation. The proof above is now in the test's docstring, where the next reader of the
`abs(a − b) == 9` line will find it.

## D-E3-2 · `_best_window` prefers the shortest window on a tie — accepted, documented

`s5_slm._best_window` scans candidate widths ascending and keeps the first window at the
lowest distance, so when a value differs from the OCR line only in its **last** character, the
grounded span is the OCR *prefix* (one character short) rather than the full line — both are
at distance 1.

Not changed, because it is consistent with the function's own exact-match short circuit:
`haystack.find(needle)` already means that a model value which is a proper substring of the
OCR line grounds to that substring, losing the surrounding characters. Making the tie-break
prefer the full-length window would leave the exact-match path inconsistent with it, for a
one-character improvement in evidence.

What matters is unchanged and is asserted directly
(`test_a_trailing_disagreement_yields_the_shorter_ocr_window`): the published `Field.value` is
always a real slice of the OCR line, never the model's string.

## D-E3-3 · **Implementation change:** the MT sentinel shape was destroyed by the real engine

The mandated `@pytest.mark.local_models` integration test (item 3 of the brief) found this,
which is the whole reason for asking for it.

`MT_SENTINEL_TEMPLATE` was `[[E{index}]]`. Against the CTranslate2 IndicTrans2 dist-200M
weights now in `models/mt/indic-en`, a Hindi field masked to
`मेरे खाते से [[E0]] ... खाता संख्या [[E1]] है।` came back as
`The account number [ addressE1 ] was incorrectly deleted from my account .` — one sentinel
dropped entirely, the other translated into English prose. `restore_entities` correctly raised
`SentinelRestoreFailed` and `translate_fields` correctly flagged `TRANSLATION_UNAVAILABLE`, so
no invariant was violated — but the consequence in production is that **every free-text field
containing an identifier would silently get no `value_en` at all**, which is exactly the set of
fields the feature exists for.

Candidate shapes were swept through the real weights on one fixture (Hindi, two placeholders):

| Shape | Survived both placeholders | Model output |
|---|---|---|
| `[[E0]]` | no | `[ addressE1 ]` |
| `<ID0>` | no | `< ID0 number` |
| `<id0>` | no | `< id1 >` |
| `__ID0__` | no | `__ID1_` (truncated) |
| `@0@` / `#0#` / `%0%` / `~0~` / `${0}` / `[0]` | no | delimiters split off, digits translated |
| `<0>` | no | dropped entirely |
| `ENTITY0` | **yes** | `ENTITY0` / `ENTITY1` |
| `XX0XX` | **yes** | `XX0XX` / `XX1XX` |
| `E0` | **yes** | `E0` / `E1` |
| `QQ0QQ` | **yes** | `QQ0QQ` / `QQ1QQ` |

The pattern is unambiguous: SentencePiece keeps an unknown **alphanumeric** run together and
does not keep punctuation glued to it. The module's original comment ("an exotic delimiter
tokenises unpredictably under byte-level BPE") had the right instinct and the wrong
conclusion — brackets *are* the exotic delimiter here.

**Change:** `MT_SENTINEL_TEMPLATE = "ZQ{index}ZQ"`; `_MARKERS` becomes `ZQ`, `ZQZQ`, …;
`_sentinel` emits `{marker}{index}{marker}`; `_restore_pattern` matches the same, still
whitespace-tolerant. Four small private functions and one constant — no call site, no other
module and no contract changed. The marker is doubled on both sides so `ZQ1ZQ` cannot be a
prefix of `ZQ10ZQ`, which is what keeps restore-by-replacement order-independent (the original
`]]` served the same purpose). `ZQ` because the bigram occurs in neither English nor romanised
Indic text; `_pick_marker` escalates if a document proves otherwise.

The integration test now passes end to end against real weights: `₹45,000` and a 14-digit
account number both come back byte-identical through a genuine Hindi→English translation.

**Residual, for WS-H:** a distilled 200M model still drops a placeholder on long sentences with
four or more entities (measured: 3 of 5 probe sentences survived intact, versus 0 of 5 with the
bracket form). The stage's behaviour on loss is correct and tested; the *rate* is a calibration
item. Raised as Q7 in `OPEN-QUESTIONS.md`.

## D-E3-4 · **Implementation change (cosmetic):** `re.subn` replacement callback

`restore_entities` used `pattern.subn(lambda _m, o=original: o, restored)` — the default-argument
idiom for avoiding loop-variable capture, which `mypy` cannot infer a type for. Replaced with a
named `_constant_replacer(replacement)` factory. Behaviour is identical; the docstring now
records *why* it is a callback rather than a replacement template (in a template, `\` and
`\g<...>` are special, and what is written back is untrusted OCR text). `mypy` is now clean on
all four lane modules.

## D-E3-5 · Envelope schema validation without a `jsonschema` dependency

`tests/unit/test_contracts_schema.py` records the standing decision that pydantic is the
validator of record and that adding `jsonschema` at runtime is a licence and supply-chain cost
for no coverage. But BUILD-BRIEF M6 wants a *really assembled* envelope checked against the
committed artefact, not against the models that generated it.

`tests/unit/test_assemble.py::validate_against_schema` is a ~50-line recursive checker covering
exactly the keyword set pydantic emits (`$ref`/`$defs`, `anyOf`, `type`, `enum`, `const`,
`properties`, `required`, `additionalProperties: false`, `items`, numeric and length bounds).
Anything outside that set raises rather than silently passing, and the committed schema was
confirmed to contain no `allOf`/`oneOf`/`prefixItems`/`patternProperties`. It was
counter-tested against seven deliberate mutations (out-of-range integer, wrong type, missing
required property, extra property, bad enum member, wrong `const`, missing nested required) and
caught all seven.

## D-E3-6 · Marker registration lives in `tests/conftest.py`, not `pytest.ini`

The `local_models` marker needs registering. The repository has neither `pytest.ini` nor
`pyproject.toml`, and per the lane brief WS-E1 owns that file if one is created — so one was
**not** created. The marker is registered by a `pytest_configure` hook in `tests/conftest.py`
(the one additive edit outside this lane, 15 lines, no fixture touched). Move it to
`[tool.pytest.ini_options] markers` when E1 adds a project file; the hook's docstring says so.

## D-E3-7 · The `local_models` test asserts strictly, on purpose

`test_hindi_field_translation_through_the_local_provider` skips with a reason when
`models/mt/indic-en/{model.bin,model.SRC,model.TGT}` are absent, so the suite passes on a fresh
clone with an empty `models/` (verified: `-m "not local_models"` and the skip path both green).

When the weights *are* present it asserts the full round trip rather than "either a translation
or a flag". Greedy decoding (`beam_size=1`) makes the outcome deterministic for fixed weights,
and a future model swap that breaks placeholder survival is precisely the regression this test
should fail on — D-E3-3 is the evidence that it is a regression worth catching.

## D-E3-8 · OPEN-QUESTIONS Q4 audit — the synthetic IFSC master covers the corpus

Audited `app/resources/ifsc_master_sample.json` against every `corpus/**/*.gt.json`
(held-out split included — this audits the reference table, not extraction quality):

* **5 / 5 bank prefixes present.** The corpus renders `SBIN`, `PUNB`, `HDFC`, `BARB`, `UBIN`
  and the master knows all five. This is the one that matters: an unknown prefix flags
  `IFSC_NOT_IN_MASTER` on every page carrying it, which would fill the review queue with the
  fixture's gaps instead of with defects.
* **15 / 15 branch rows are codes the corpus actually renders.** No dead fixture rows, so the
  exact-lookup path is genuinely exercised.
* **195 of 210 distinct corpus codes have no branch row**, and that is correct behaviour, not a
  gap: `authoritative: false` makes absence *unverifiable* rather than wrong. The hit keeps
  `validator_passed=True`, carries no `IFSC_NOT_IN_MASTER` flag, and is scored at
  `UNVERIFIED_WEIGHT` (0.9) so the envelope says "structurally valid, not confirmed". Were
  absence a failure, ~93% of the corpus would arrive pre-flagged.
* `bank_aliases.json` covers the same five banks with English and Devanagari spellings; already
  asserted by `test_committed_alias_table_covers_the_corpus_banks`.

Locked in by `tests/unit/test_s5_validators.py::TestQ4MasterCoverage` (4 tests), so a corpus
regeneration that introduces a sixth bank fails the suite instead of quietly degrading
extraction confidence.

### The real RBI list is a drop-in — the loader interface

`IFSCMaster` already accepts the real list with no code change anywhere else:

```python
from app.stages.s5_validators import IFSCMaster, run_validators

master = IFSCMaster.load("data/RBI_IFSC.csv")      # .csv -> authoritative=True
report = run_validators(pages, ifsc_master=master)  # the single injection point
```

* **`IFSCMaster.load(path, *, authoritative=None)`** dispatches on suffix and refuses anything
  that is not `.json` or `.csv` — guessing the format of the file that decides whether an
  identifier is valid is not a risk worth taking.
* **`.csv` → `from_csv`, `authoritative=True` by default.** Headers are matched
  case-insensitively against `_CSV_COLUMNS`: `IFSC`/`IFSC_CODE`/`IFSCCODE`,
  `BANK`/`BANK_NAME`/`BANKNAME`, `BRANCH`/`BRANCH_NAME`/`OFFICE`, `CENTRE`/`CENTER`/`CITY`,
  `STATE` — the RBI publication and the mirrored community CSVs disagree on case and on
  `CENTRE`/`CITY`, and accepting both is cheaper than a transform step in the deployment
  runbook. `bank_code` is derived from the first four characters. Rows whose IFSC fails
  `IFSC_STRUCTURE` are skipped and counted in `skipped_rows` rather than raising: a
  ~170 000-row public file reliably carries a handful of malformed rows, and refusing the whole
  list over them would leave the validator with no master at all. A missing IFSC column raises.
* **`.json` → `from_json`**, the committed shape (`banks[]` + `branches[]`), taking
  `authoritative` from the payload unless overridden. A payload without `branches` raises, so a
  truncated download cannot become an empty master that passes everything.
* **What changes when the list is authoritative:** an unknown branch or prefix becomes a failed
  `master_lookup` check — `validator_passed=False`, `normalized=None`, `IFSC_NOT_IN_MASTER` —
  instead of an unverified pass. Nothing else in the module has a branch on the flag.
  `test_the_real_list_is_a_path_change_not_a_code_change` demonstrates the whole flow against a
  temporary CSV.
* **Native-script bank spellings stay in `bank_aliases.json`.** The real RBI master carries
  branch rows and one Latin bank spelling; it has no Devanagari rendering, which is what an
  OCR'd complaint actually contains. Keeping the two files separate is what lets the real list
  drop in without losing alias coverage.

**Not done, and deliberately out of lane:** there is no `IFSC_MASTER_PATH` setting, so
`default_ifsc_master()` is hard-wired to the committed fixture and a deployment must pass
`ifsc_master=` at the `run_validators` call site. Adding the setting is one field in
`app/settings.py` (WS-B) plus one line in the orchestrator (WS-F). Raised as Q8.

---

## What this pass did **not** change

* No validator logic, no grounding-gate logic, no assembly logic. The interrupted build's
  modules were audited against PLAN.md invariants 1, 2, 7, 8, 9, 10 and found correct; the only
  production edits are D-E3-3 (sentinel shape, driven by measurement) and D-E3-4 (a typing
  cleanup with identical behaviour).
* No contract, provider or settings file.
* `app/providers/hf_api.py`'s separate `MASK_SENTINEL_TEMPLATE` (`[[MASK_{index}]]`, used for
  the SLM prompt) was left alone: an instruct LLM is never asked to reproduce those tokens, so
  D-E3-3's finding does not apply to it.

## Known gaps carried forward (already stated in `s5_validators.py`'s docstring)

Devanagari month names, Indian financial-year notation (`FY 2025-26`), words-as-amounts
("forty five thousand only"), signed amounts on statements, and E-13B cheque MICR bands are all
unhandled by design in Phase 0. Each is a data-driven addition, none needs a structural change.
