# Q6 decisions — corpus ground-truth field gaps

Judgment calls made while closing OPEN-QUESTIONS **Q6**: `receipt`, `email_printout`,
`cheque` and `photo_evidence` emitted empty `fields` dicts, and every document's `fields`
carried English spellings even where the page was Devanagari.

Lane: `tools/generate_test_corpus.py`, `tools/eval_harness.py`, `corpus/**`,
`tests/corpus/**`, this file, and the Q6 row of `OPEN-QUESTIONS.md`. Nothing outside it
was touched. WS-D conventions from `docs/decisions-wsd.md` apply unchanged: same seed
(1729), same variants (5) and profiles (6), same locked split (variants 3 and 4), no
process-global RNG, UTF-8 everywhere, typed errors.

Generator `1.1.0` → `1.2.0`. Harness `1.0.0` → `1.1.0`.

---

## 1. What Q6 actually was — three defects, not one

The row named two. Measuring it turned up a third, and the third was the expensive one.

| # | Defect | Effect on the M7 dev-split run |
|---|---|---|
| a | Four renderers returned `fields: {}` | those types contributed field references only through line links; per-type field F1 0.024 / 0.000 / 0.000 / 0.000 |
| b | `fields` held English values on Devanagari pages | a native-script `value` had no reference of its own; the harness had to accept the translation instead, so a pipeline that put English in the immutable `value` scored identically to one that got it right |
| c | **The corpus and the envelope use different names for four fields** | `amount`, `utr_rrn`, `card_fragment` and `prior_complaint_reference` had **no reference instance at all**, so all 158 extractions of them were false positives by construction |

(c) is what the Q6 row was describing when it wrote "`amount` 69 FP / 0 GT, `utr_rrn`
61 / 0, `card_fragment` 24 / 0" — but it read as a symptom of (a) rather than as its own
cause. It is not: the corpus *does* render an amount on nine of ten types. It calls it
`disputed_amount`.

Fixing (a) and (b) without (c) would have made the numbers **worse**: `receipt` would
have gained 90 ground-truth instances under names the pipeline never emits, adding 90
misses to a type that already scored 0.

## 2. D-Q6-1 · The four thin renderers get real `fields`, and only what they draw

`fields` is now non-empty for every type. Each key is a fact the page shows:

| type | fields | pointedly absent |
|---|---|---|
| `receipt`, `photo_evidence` | `transaction_date`, `bank_name`, `card_last_four`, `disputed_amount`, `transaction_reference` | **`account_number`** — the slip prints `XXXXXX` + the last four, so a full number would be a label no reader can satisfy |
| `email_printout` | `transaction_date`, `prior_complaint_ref` | **any amount** — the acknowledgement mentions none. Two fields is the honest count |
| `cheque` | `bank_name`, `complainant_name`, `transaction_date`, `account_number`, `ifsc_code`, `micr_code`, `disputed_amount` | **`transaction_reference`** — the bottom band prints only its first six digits |

The temptation in all three rows was to claim the field anyway and let the harness's
fuzzy matching sort it out. A ground truth that claims what the page does not show is the
same defect as one that omits what it does; it just fails in the other direction, and
silently, as a permanent false negative that looks like a model problem.

`photo_evidence` reuses `render_receipt`, so it inherits the receipt's fields without a
line of new template code. The task's fallback ("if genuinely none, give it at least
amount + date on the rendered slip") was not needed.

## 3. D-Q6-2 · `fields_bilingual`, not objects inside `fields`

The additive block is a sibling dict:

```json
"fields_bilingual": {
  "bank_name": {"value": "बैंक ऑफ बड़ौदा", "value_en": "Bank of Baroda", "script": "Deva"}
}
```

Rejected: promoting `fields` values to objects. The brief asked for `fields` to stay
backward-compatible with value strings, and it has a consumer outside this lane —
`tools/slm_bakeoff.py` reads `gt["fields"]` by key and would need changing in step. A
sibling block costs the harness ~15 lines and costs every other reader nothing.

`script` is the third key because it is the question the block exists to answer ("which
fields are Devanagari-valued") and the generator knows it for free. It is the *value's*
script, not the line's: a Marathi sentence carries an ASCII account number, and labelling
that number `Deva` would tell a scorer to expect Devanagari digits.

## 4. D-Q6-3 · The block is built by the draw calls, so it cannot lie

`Canvas.text(fld=…, fld_en=…, fld_value=…)` records into `canvas.field_values`, and
`generate_corpus` copies that onto the `GTDoc`. A field can therefore only appear in
`fields_bilingual` if a renderer really rendered it —
`test_every_bilingual_field_is_linked_to_a_line_that_was_drawn` asserts the containment
against `lines[].field` for all 295 documents.

None of the three arguments touches a pixel. That is load-bearing: it is why 265 of 295
images are byte-identical (§6).

**`fld_value` exists because a line is not a field.** `"My account number is
12345678901."` links `account_number`, but the *value* is the number. Without the
override, `fields_bilingual` would publish the sentence as the account number and the
cheque's MICR band as the IFSC code. Applied to: the letter's `account_number` /
`disputed_amount` / `transaction_reference`, the cheque's `ifsc_code` and its
amount-in-words ("only" is convention, not the amount), the receipt's `card_last_four`
(the row prints a masked PAN), the email's `prior_complaint_ref` (the subject wraps it in
"Re: Service Request …"). `relief_sought` deliberately keeps the whole sentence: there,
the sentence really is the value.

**First draw wins** when a field is drawn twice — the cheque writes its amount in words
and again in figures. The second spelling still reaches a scorer through its own
`lines[].field` link, so nothing is lost and the block stays one pair per field.

## 5. D-Q6-4 · The envelope↔corpus name gap is fixed in the harness, not by renaming the corpus

`ENVELOPE_FIELD_ALIASES` maps `amount → disputed_amount`, `utr_rrn →
transaction_reference`, `card_fragment → card_last_four`, `prior_complaint_reference →
prior_complaint_ref`. Reporting keys on the **corpus** name so TP/FP/FN line up with
`gt_occurrences`, and the table is echoed in `report["settings"]`.

**Renaming the corpus keys to the contract's vocabulary was the first instinct and is
wrong here.** `tools/slm_bakeoff.py` — outside this lane — keys its PLAN.md-invariant-7
leak check off the *corpus* names (`INVARIANT7_GT_KEYS`, `UNCLAIMED_PII_GT_KEYS`). A
rename would leave those dicts syntactically valid, matching nothing, and the check would
pass while checking nothing. A security assertion that silently stops asserting is worse
than the naming inconsistency it would have tidied.

`complaint_date` is deliberately **not** aliased to `transaction_date`: the contract
distinguishes them and the corpus renders only one of them. `micr_code` needs no alias —
both vocabularies already agree, which is why the cheque's new field uses that name.

**OWNER:** the durable fix is one vocabulary. That means renaming the corpus keys *and*
updating `slm_bakeoff.py` in the same change, which crosses lanes. Recorded rather than
done.

## 6. D-Q6-5 · The cheque is the only document whose pixels changed — 30 of 295

Regenerated at seed 1729, 5 variants, 6 profiles: **265 images byte-identical, 30
changed, all `cheque`**. Every `.gt.json` changed (that is the point) plus `index.json`.

The cheque template rendered no date, no account number and no MICR band, so three of the
five fields the brief names for it could not be linked to anything. That is a template
defect, not a hard case. Three **printed** lines were added:

| doc_ids | added | why |
|---|---|---|
| `cheque_v{0..4}_{clean,light,scan,heavy,photo,bounce}` | `DATE <transaction_date>` top-right | a cheque without a date is not a cheque |
| — as above — | `A/C No. <account_number>` under the bank name | the full number, printed as real cheques print it |
| — as above — | `MICR <9 digits>` above the existing band | `micr_code` is in `DETERMINISTIC_FIELDS` and had no reference anywhere in the corpus |

Two properties made this safe to do without disturbing the rest of the corpus:

1. **The additions are printed, not handwritten.** Only `hand=True` draws consume the
   pen's jitter RNG, so every handwritten stroke on the cheque is unchanged and the three
   new strings are the entire pixel difference.
2. **`cheque_micr_code()` derives its 9 digits from facts already drawn**
   (`ifsc_code[5:8] + account_number[:3] + transaction_reference[-3:]`) rather than
   drawing from the RNG. `make_facts()` consumes the per-document stream in a fixed
   order; one extra `randint` there would have shifted every subsequent draw and
   re-rendered all 295 pages. This is the single most fragile property in the generator
   and it is why the MICR code is a derivation and not a fact.

Rendered output was checked by eye on `cheque_v0_clean`: no collisions, all three
additions legible and in plausible positions.

**Consequence for `docs/eval/`:** the 30 committed cheque predictions were produced
against the old images and are now stale. `docs/eval/**` is not this lane, so nothing
there was touched; the cheque row in §8 is a floor for that reason.

## 7. D-Q6-6 · The harness scores native and English separately — and reports it separately

`build_field_reference` now returns `FieldReference(native, en)` instead of a bare set.
`native` is what the page shows (line links + `fields_bilingual.value`); `en` is the
English equivalent (`fields_bilingual.value_en` + `gt["fields"]`, which lands in `en`
*only* when the bilingual block marks the field Devanagari-valued).

`is_bilingual` is `not en <= native`, not `native != en`. Set inequality is the wrong
test: `native` legitimately holds extra spellings — the sentence an identifier is
embedded in, the amount written twice — and none of those makes a field bilingual. What
does is an English spelling that appears nowhere in what was rendered.

**The headline TP/FP/FN accounting was deliberately left lenient**: a match from `value`,
`value_en` or `normalized` against `native | en` is a hit, exactly as before. The
stricter question — did the *immutable* `value` carry the page's own string, or did the
pipeline put the translation there — goes into a new `fields.bilingual` report block and
its own table.

The reason is separability. Q6 is a ground-truth fix; hardening the metric in the same
change would mean the next run's field numbers moved for two reasons at once and nobody
could tell which. The counters (`value_is_native`, `value_is_translation`,
`value_en_correct`, `value_en_wrong`, `value_en_absent`) make the contract violation
visible today; promoting it into precision is a one-line change whenever the owner wants
it. **OWNER: decide before the `--final` run.**

`validate_prediction` gained one check: `value_en` must be null or a string.

## 8. Measured effect — same predictions, same pipeline, fixed reference

`tools/eval_harness.py --corpus corpus --predictions docs/eval/predictions` (dev split,
177 documents), against `docs/eval/report.json` from the M7 gate. **No pipeline code
changed between these two columns.**

| | before | after |
|---|---|---|
| field TP | 157 | **405** |
| field FP | 346 | **98** |
| field FN | 827 | 786 |
| precision | 0.312 | **0.805** |
| recall | 0.160 | **0.340** |
| F1 | 0.211 | **0.478** |
| GT instances | 984 | 1191 |

By source: `regex_validator` P 0.346 → **0.811**, `lookup_table` P 0.181 → **0.855**. The
Q6 row's estimate was "deterministic precision from 0.318 to ~0.46 depending on how Q6 is
treated"; measured, the honest figure is **0.81**, and the "depending on how" is gone —
there is no longer a defensible way to treat it that produces 0.31.

Per-type field F1 for the four thin types:

| type | before | after |
|---|---|---|
| `receipt` | 0.000 | **0.827** |
| `photo_evidence` | 0.000 | **0.815** |
| `email_printout` | 0.000 | **0.588** |
| `cheque` | 0.024 | **0.359** (floor — stale predictions, §6) |

Three fields with no reference instance at all (`amount` 0/69/0, `utr_rrn` 0/61/0,
`card_fragment` 0/24/0) are gone. `micr_code` appears for the first time, at 0/0/18 — a
real miss now measurable, where before it was invisible.

**What did not improve, and should not have:** `complaint_letter_mar` is still field F1
**0.000**, and the bilingual table reads `value_is_native: 0` over 198 bilingual
ground-truth instances with only 2 claimed. That is Q12 — the Devanagari recognition head
is never selected, so there is no Devanagari text for any field to be extracted from. Q6
made that defect *measurable*; it cannot make it go away, and a Q6 fix that moved those
numbers would be a Q6 fix that was cheating.

## 9. Verification performed

- Full regeneration, seed 1729, 5 variants, 6 profiles: 295 pages, counts unchanged
  (9 types × 30 + `photo_evidence` × 25).
- Byte-diff of every image against the committed corpus: 265 identical, 30 changed, all
  `cheque` (§6).
- `tests/corpus`: **166 passed** (was 143 + 23 new/updated).
- Harness self-tests including the CLI, `--final` and the locked-split refusal: green.
- Determinism suite unchanged and green, including
  `test_generation_does_not_touch_the_global_random_state` and the per-document
  regeneration cases (one of which is `cheque_v0_scan`).
- Visual check of the regenerated `cheque_v0_clean`.
- Full `tests/unit` re-run after the corpus regeneration, because
  `tools/slm_bakeoff.py` reads `gt["fields"]` and `email_printout` went from `{}` to two
  keys.

## 10. Open / for the owner

1. **One field vocabulary (§5).** The alias table is the fix that fits this lane; the
   durable fix crosses into `tools/slm_bakeoff.py`.
2. **Promote the bilingual check into precision (§7).** Decide before `--final`.
3. **`docs/eval/` predictions for `cheque` are stale (§6).** Re-run the sweep — which is
   needed for Q12 anyway — and the floor becomes a number.
4. **`relief_sought` still references a whole sentence.** Now deliberate and documented
   rather than incidental: for that field the sentence really is the value, so a pipeline
   extracting a phrase still scores an FP. Unchanged from WS-D §8.
