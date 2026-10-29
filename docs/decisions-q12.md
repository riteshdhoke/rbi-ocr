# Q12 decisions — the circular Devanagari script routing (P0)

What was decided, measured and rejected while fixing OPEN-QUESTIONS **Q12**, the defect
the M7 exit gate named as "the single largest defect the gate found"
(`docs/eval/REPORT-2026-08-08-slmlive.md` §2).

Lane: `app/stages/s4_ocr.py`, `app/pipeline.py` (minimal), `tests/unit/test_s4_ocr.py`
(additive), `tests/unit/test_pipeline.py` (additive), this file, the Q12 row of
`OPEN-QUESTIONS.md`. Nothing else was touched. `app/providers/**`, `app/settings.py`,
`app/stages/s2_script.py` and `app/contracts/**` are other lanes' committed files and were
read, not edited — §5 records the one place that constraint shaped the design.

---

## 0. The defect, restated precisely

```python
bootstrap = script_hint or DEFAULT_REC_HEAD_SCRIPT      # == "Latn"
lines     = provider.ocr_page(page, bootstrap)          # Latin head on a Deva page
decision  = identify_script(lines=lines, ..., recognised_with=bootstrap)
```

`app/pipeline.py` passed no `script_hint`, so `bootstrap` was `"Latn"` on every page in the
service. The A-6 vote is a Unicode-range majority over `lines`, and a Latin recognition head
can only emit Latin codepoints — `बैंकिंग लोकपाल को शिकायत` came back as `h cicc  chr`. The
vote therefore returned `Latn` **confidently**, `decision.rec_head_script != bootstrap` was
false, and the one re-recognition pass A-6 exists to trigger never ran.

The loop is closed by construction: the script is a function of the text, the text is a
function of the head, and the head was a function of nothing at all.

**Not a model gap.** `models/ocr/rec_deva.onnx` reads the same page at page-concatenated CER
**0.036**. The weights were always fine.

---

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| **D-Q12-1** | **Head selection is a comparison between heads on shared crops; the A-6 text vote is demoted to a confirmation step.** After the bootstrap pass, `probe_script_heads()` re-reads 3 of that pass's own line crops with *every installed recognition head* and compares the readings. The vote still runs, unchanged, over the winning head's full-page output. | The vote's failure mode is that it cannot disagree with the head that fed it. A comparison *between* heads can. Keeping the vote as confirmation means a wrong probe is still caught — §2 shows that path firing in production on `bilingual_form_v2_light`. |
| **D-Q12-2** | **The probe's answer is "which head read these crops better, *and what did that head say the crops were*" — not "which head scored highest".** | Selecting on score alone breaks English. PP-OCRv5's Devanagari head ships a charset covering Devanagari **and** Latin (128 Devanagari + 52 Latin characters in `rec_deva_dict.txt`) and it reads an English page *marginally more confidently* than the Latin head — measured 0.982 vs 0.976 char-weighted on `complaint_form_en_v0_clean`. That asymmetry is also what makes the probe work: the Devanagari head's transcription is a genuine script vote, because it was free to answer either way. The Latin head, whose dictionary holds one Devanagari codepoint, can only ever say `Latn`. |
| **D-Q12-3** | **Heads that agree settle it with no confidence comparison at all.** Only when they read the same crops as *different* scripts does score decide, and then only past `PROBE_SCORE_MARGIN`. | 116 of 147 dev-split documents resolve as `heads_agree`, which is exactly the set that must not move. An English page cannot regress under this rule *by construction*, not by threshold — asserted as byte-equal transcription in `test_an_english_page_is_left_exactly_as_it_was`. |
| **D-Q12-4** | **Sample 3 crops: the longest line in each of 3 vertical bands, preferring lines whose bootstrap text carries letters.** | Both rules were measured (§3). Two bands score 141/147 because a Devanagari complaint form's account number, IFSC and phone number are Latin-script identifiers that *both* heads read at confidence 1.000; three bands score 146/147. The letter preference reads the bootstrap head's text, but only to **rank crops**, never to decide a script — a distinction the docstring states explicitly. |
| **D-Q12-5** | **`PROBE_SCORE_MARGIN = 0.05`.** | Measured over the 28 dev-split pages whose two heads read the crops as *different* scripts: the narrowest margin the rule accepts is **0.079** (`complaint_letter_mar_v0_photo`, routed correctly), the only one it rejects is **0.033** (`complaint_form_deva_v1_photo`), and the rest run 0.12-0.44. The gap between "coin flip" and "real decision" is empty in this data, so 0.05 sits in it. |
| **D-Q12-6** | **One head decoding almost nothing is *evidence*, not silence.** If the other head read the same crops and could not produce `PROBE_MIN_SCRIPT_CHARACTERS` (6) letters, the head that did decode wins (`single_head_decoded`). | This was a **bug in the first implementation of this fix**, caught by running the real pipeline before writing the test. Requiring two *scoring* heads stranded `complaint_letter_mar` at CER 0.79: on a Marathi letter the Latin head returns 4 characters of rubble from 3 crops while the Devanagari head returns 26. Two heads must have *read* the sheet (`len(scores) >= 2`); they need not both succeed. |
| **D-Q12-7** | **A vote that contradicts the probe reverts to a transcription already held — no third provider call, and no loop.** `Tier1Result.rec_head` names the head that produced the final text; `rec_passes` stays a record of what the provider was *paid* for. | "Falls back once" has to mean once. Because the contradicting head is essentially always the bootstrap head, whose output is still in hand, the correction is free. `MAX_RECOGNITION_PASSES` stays 2 and is still asserted. |
| **D-Q12-8** | **A probe/vote conflict raises `script_disagreement`**, which the escalation router already keys on. | Two mechanisms arguing about a page's script is precisely the unsettled page `s2_script.py` documents Tier 2 as the safety net for. It fired once in 147 documents, so the effect on the 12.4% escalation rate is ~0.7 points. |
| **D-Q12-9** | **The probe never runs where the old mechanism was not circular**: not on a bypass page, not when the A-6 vote has already re-routed the page, not with `max_passes < 2`, not with fewer than two candidate heads. | Pay for the probe only where it can change something. `max_passes=1` is `tools/bench_ocr_cpu.py` timing a single pass, and buying an answer it cannot act on would corrupt its measurement. |
| **D-Q12-10** | **`script_prior` is a new parameter, distinct from `script_hint`.** The pipeline passes the script of a PDF text layer that a geometric repair forced it to discard. It seeds the bootstrap head and breaks a tied probe; it is **not** passed to `identify_script` as `hint`. | A hint is caller metadata about the *document*, and Stage 2 raises `hint_mismatch` when the vote contradicts it. A prior is evidence about *these pixels*, produced by no recognition head — the only non-circular script signal this pipeline has — and it should not be able to manufacture a disagreement. The API is additive; every existing call site keeps its behaviour. |
| **D-Q12-11** | **Tunables are module constants, not `Settings` fields.** | Follows Stage 2's own precedent (`app/settings.py` is a committed lane). Every one is overridable per call, and every one is listed here. |

---

## 2. Measured result — the whole dev split, both arms, through the real pipeline

147 documents (variants 0-2, non-bounce), `LocalProvider` on real weights, each document run
twice through `app.pipeline.run_document`: once with `probe_heads=False` (the pre-fix code
path, exactly) and once as shipped. Same weights, same pixels, same quality gate.

CER here is **page-concatenated** against `*.gt.json` using the harness's own
`normalize_for_cer` + `levenshtein`. It is *not* the M7 report's per-matched-line CER — the
same pre-fix Devanagari page measures 0.657 this way and 0.852 in the report — so read the
delta, not the absolute against §1 of the report.

| cell | before | after |
|---|---|---|
| **script:Deva** (30 docs) | **0.7382** | **0.1648** |
| **script:Latn** (117 docs) | 0.1807 | **0.1807** — byte-identical |
| all 147 | 0.2892 | 0.1776 |

Devanagari by degradation profile, 6 documents each:

| profile | before | after |
|---|---|---|
| clean | 0.7220 | **0.0410** |
| light | 0.7402 | 0.0656 |
| scan | 0.7330 | 0.1323 |
| heavy | 0.7449 | 0.2087 |
| photo | 0.7506 | 0.3761 |

Latin by profile — every cell unchanged to four decimal places (0.1408 / 0.1571 / 0.1751 /
0.1991 / 0.2297 for clean / light / scan / heavy / photo). **No Latin page moved at all.**

Downstream, on the same run:

| | before | after |
|---|---|---|
| recognition head matches GT script | 117/147 | **146/147** |
| page `primary_script` matches GT | 117/147 (0/30 Deva) | **146/147 (29/30 Deva)** |
| Deva-page `primary_language` | `eng` on 30/30 | Indic on 29/30; exact GT match 22/30 |
| documents costing two recognition passes | 0 | 30 |
| `script_disagreement` raised | 0 | 1 |

**The one Devanagari page that is still wrong** is `complaint_form_deva_v1_photo`: the probe
returns `inconclusive_tie` (the two heads disagree by 0.033 on the harshest degradation
profile) and the page keeps the pre-fix behaviour and the pre-fix CER. It is not made worse,
and the probe says so in `script_probe.reason` rather than guessing. **No document's CER got
worse by more than 0.02**; none got worse at all.

**Three pages resolve as `single_head_decoded`** and D-Q12-6 is what carries them:
`complaint_letter_mar_v0_clean` (Latin head 4 script characters, Devanagari head 26),
`complaint_letter_mar_v1_clean` (2 against 30), `complaint_form_deva_v1_light` (5 against 26).
All three land at CER 0.044-0.066 after; all three were 0.66-0.79 before.

**Devanagari LID is no longer 0.000, and the residual is not this fix's.** 22 of 30 Deva pages
get the exact GT language; the 8 misses are all `mar` read as `hin`. `lid.176` separating
Marathi from Hindi on short lines is a Stage-2b property that could not previously be measured
because the stage was never handed Devanagari at all. Flagged for whoever owns the LID lane —
not diagnosed here.

**Translation (Q7) is unblocked.** With the SLM stood in for (see §4), a Devanagari field value
now reaches Stage 6 and comes back with `value_en` from the real IndicTrans2 weights:
`मेरे बचत खाते से अनधिकृत इले्ट्ॉनिक लेनदेन डेबिट किया गया` →
`Unauthorized electronic transaction debited from my savings account`,
`mt_engine=indictrans2-indic-en-dist-200M-ct2-int8`.

### Latency — the probe's own cost

Wall time of `probe_script_heads` (sheet construction included), measured per page on the same
run, laptop CPU, 2 threads:

| | p50 | p90 | p95 | max |
|---|---|---|---|---|
| absolute | 94 ms | 113 ms | 119 ms | 302 ms |
| **as a share of the pre-fix `s4_ocr` stage** | **16%** | 29% | 30% | — |
| as a share of the shipped `s4_ocr` stage | 12% | 21% | 22% | — |

**Mean 18% of the pre-fix stage, budget 25%.** The tail is on documents whose OCR is unusually
cheap — a cheque with three lines costs 240 ms to recognise, so a 100 ms probe is 40% of it
while being 100 ms. Stage totals: `s4_ocr` p50 582 → 771 ms, p95 1528 → 1840 ms; end to end
p50 1046 → 1266 ms, p95 2008 → 2375 ms. Against PRD target 8 (p95 ≤ 90 s) this is noise.

---

## 3. What the design is built on — measurements, not assumptions

The M7 report offered "recognise a sample of crops with both heads and keep the
higher-confidence transcription" as remedy (b). Taken literally it does not work, and three
separate measurements shaped what shipped.

**a. Confidence alone routes English to the wrong head.** On `complaint_form_en_v0_clean` the
Devanagari head scores 0.982 against the Latin head's 0.976. Hence D-Q12-2: the script the
winning head *wrote* decides, and agreement short-circuits the comparison entirely.

**b. Charset coverage is not symmetric, and that is the whole signal.** Measured over the
`rec_*_dict.txt` files actually installed:

| head | Devanagari chars | Latin chars | digits |
|---|---|---|---|
| `rec_latn` | 1 | 52 | 10 |
| `rec_deva` | 128 | 52 | 10 |

So the Devanagari head's output is diagnostic (`own-script` share 0.96 on a Deva page, 0.00 on
an English page, 0.44-0.59 on a bilingual form) and the Latin head's is not. A "charset
coverage" score in the naive sense — *does this head's output match its own script* — is
near-constant 1.0 for the Latin head and measures nothing.

**c. Recognition cost is linear in padded crop area, not in call count.** Measured directly
against both heads:

| crops per call | width 100 | width 200 | width 320 | width 500 | width 700 |
|---|---|---|---|---|---|
| 1 | 8.9 ms | 12.1 | 15.5 | 28.5 | 30.5 |
| 4 | 27.7 | 43.5 | 58.1 | 138.0 | 146.8 |
| 16 | 114.9 | 169.7 | 224.6 | 577.1 | 560.7 |

≈7 ms per crop at width 100 and ≈35 ms at width 640, flat in batch size. The first
implementation stacked full-width crops on a tall sheet and cost 26-94% of a page. Normalising
each crop to the recogniser's own 48 px input height and **truncating to 160 px** — one width
bucket, 5-10 glyphs, enough to tell two scripts apart — is what brought it to 16%. Bucket
fragmentation matters too: four crops of mixed width are four padded calls.

**d. The sheet geometry was grid-searched** over crops ∈ {2,3,4} × width cap ∈ {160,240} ×
{single row, vertical stack}, scored against the corpus's `primary_script`:

| crops | cap | layout | accuracy | cost p50 |
|---|---|---|---|---|
| 2 | 160 | stack | 141/147 | 14% |
| 3 | 160 | **row** | **146/147** | **19%** |
| 3 | 240 | row | 146/147 | 20% |
| 4 | 240 | stack | 143/147 | 32% |

(Those percentages are from the grid harness, which used a plain `ocr_page` bootstrap rather
than the full pipeline; §2's 16% is the shipped measurement.)

---

## 4. Test decisions

| # | Decision | Why |
|---|---|---|
| **D-Q12-12** | **A two-surface mock provider** (`_TwoSurfaceProvider`) answers differently for the page and for the probe sheet, recognising the sheet by its height. | Reproducing "the Latin head emits Latin noise from Devanagari pixels" needs a provider whose answer depends on *which surface* it was asked about. No test reaches into stage internals to know which call is which. |
| **D-Q12-13** | The probe **drops crops with no ink** (`PROBE_MIN_INK_STD`). | Two heads cannot be compared on blank paper, and this is what keeps every pre-existing test in `test_s4_ocr.py` — which drives white pages — passing unmodified with the exact same provider call sequences. The additive test file is genuinely additive: no existing assertion changed. |
| **D-Q12-14** | The real-weights proof is **A/B, not absolute**: each of the two pipeline tests runs the same document through `probe_heads=False` and the shipped path and compares. | An absolute CER assertion would pass for reasons unrelated to this fix. The English test asserts **byte-equal line texts** across the two arms, which is a stronger no-regression statement than any bound. |
| **D-Q12-15** | The translation test **stands in a deterministic verbatim extractor for the SLM**, and says so. | The router SLM answers `402 Payment Required` at account level (`docs/eval/REPORT-2026-08-08-slmlive.md` §1) and restoring credits is not a code change. Everything after the SLM is real: an actual Devanagari value from the Devanagari head, the real grounding gate, real sentinel masking, real IndicTrans2 weights. Without the stand-in the assertion is unprovable on this machine, and asserting it against a mocked *translator* would prove nothing about Q12. |
| **D-Q12-16** | Both key mock tests were **mutation-checked**: reverting the letter-preference and the heads-agree rule in the source each killed exactly the intended test. | A test that passes against a broken implementation is not a test. |

`@pytest.mark.local_models` guards every real-weights test behind a check for both heads plus
`mt/indic-en/model.bin`, and the guard was verified to report all seven artifacts missing
against an empty directory. The suite passes with `models/` empty.

---

## 5. Rejected, and why

| Option | Why not |
|---|---|
| **`script_id.onnx`, the image-space classifier A-6 reserves** | It is the principled fix and it stays the right Phase-1 answer, but there is no approved artifact (decision C7) and Q12 is P0 today. `identify_script(prefer_image=True)` already calls it, so dropping the file in later is still a model drop and not a code change. |
| **Make `identify_script` distrust short low-confidence transcriptions** (report remedy c) | Cheapest and weakest. It is a threshold on the *shape* of one head's output, so it still never consults the other head, and it would push a genuinely terse Latin page (a cheque) toward `Deva` for no reason. The probe subsumes it with evidence instead of a heuristic. |
| **Per-document `script_hint` from the caller** (remedy d) | Useless for a general mailbox and no help on a mixed one. Retained only as what it already was: a bootstrap seed. |
| **Adding a `recognise_crops(crops, script)` method to the provider protocol** | It is the obvious way to make the probe cheap — it would skip the sheet's redetection entirely, halving the probe. `app/providers/**` is a committed lane and BUILD-BRIEF sec. 3.1 fixes the eight methods, so the probe is built on `ocr_page` alone. **Recommended for whoever owns WS-C next**: it would take the probe from 16% to roughly 6% of a page and remove the one piece of this design that is a workaround rather than a mechanism. |
| **Probing only the alternative head, scoring the bootstrap head from its full-page output** | Halves the probe cost again (~9% p50) by exploiting the fact that the bootstrap head has already read those lines. Rejected for this change: the two scores would then come from differently-cut crops, and "both heads read the *same* crops" is the property that makes the comparison trustworthy. Recorded as a measured option, not taken. |
| **Caching the probe verdict across pages of one document** | Correct and cheap for multi-page documents, and worth doing. Not measurable on this corpus — every document is one page — so it would have shipped unmeasured. `script_hint` already carries the mechanism if a caller wants it. |
| **Promoting the tunables to `Settings`** | Another lane's file, and none of them has a deployment-specific value. Listed here for promotion if one ever does. |

---

## 6. Follow-ups this leaves on the record

1. **Re-run the M7 gate.** Every Devanagari row in `docs/eval/REPORT.md` — CER, LID, field F1
   for `complaint_form_deva` and `complaint_letter_mar`, and the whole of §5's translation
   finding — is stale in the pipeline's favour. The re-run costs OCR only.
2. **Q7 can be measured now.** Sentinel survival was unmeasurable because Stage 6 never fired;
   it fires.
3. **`mar` read as `hin` on 8 of 30 Devanagari pages** — a Stage-2b/`lid.176` question that
   this fix made visible for the first time.
4. **`complaint_form_deva_v1_photo`** is the one page the probe declines to call. If the photo
   profile matters, that is the page to look at first; the probe records its own scores on
   `Tier1Result.script_probe`, so the evidence is in the prediction file.
5. **`recognise_crops` on the provider protocol** (§5) is the one change that would make this
   design cheaper *and* simpler.
