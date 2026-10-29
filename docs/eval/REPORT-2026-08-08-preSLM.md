# M7 exit-gate evaluation — RBI complaint document intelligence (post-Q12 / Q6 re-run)

**Run:** 2026-08-08, dev split (variants 0-2), 177 documents, 0 failures, 329 s wall clock.
**Code:** `1e6ddbe19e1feebca76d4fb7e312301c3652eb5b` plus the uncommitted Q12 and Q6 fixes
(`docs/decisions-q12.md`, `docs/decisions-q6.md`). No tuning between the fixes and this run.
**Driver:** `tools/run_corpus_eval.py` 1.0.0 · **Scorer:** `tools/eval_harness.py` **1.1.0**
· **Corpus generator:** **1.2.0** (seed 1729)
**Supersedes:** [`REPORT-2026-08-07-preQ12.md`](REPORT-2026-08-07-preQ12.md), kept verbatim
beside this file so every claim below can be checked against what it replaces.
**Artifacts:** [`PREDICTIONS_MANIFEST.json`](PREDICTIONS_MANIFEST.json) ·
[`report.json`](report.json) · [`report.txt`](report.txt) (reproduced verbatim in §13) ·
[`predictions/`](predictions) (177 files — the report regenerates from these with no models
and no token).

Reproduce the score without a pipeline:

```
.venv\Scripts\python.exe tools/eval_harness.py --corpus corpus --predictions docs/eval/predictions
```

---

## Read this first

**Q12 is fixed and the Indic half of the product exists now.** Devanagari CER **0.904 → 0.518**
overall and **0.852 → 0.046** on clean Devanagari print. Devanagari line-level LID was exactly
0.000 in every cell; it is now 0.755 (print) and 0.417 (hand). Stage 6 fired for the first
time. No Latin cell regressed. §2.

**Q6 collapsed the honest-band problem.** The previous report could only publish deterministic
field F1 as "0.29 to 0.46 depending on how you treat Q6". That band is gone: with a reference
that names the fields the pages actually carry, deterministic F1 is a single number, **0.704**,
up from 0.294. §4 separates how much of that is the honest denominator (Q6) and how much is
better recognition (Q12) — the answer is 0.294 → 0.634 → 0.704.

**The SLM half is still not measured, and Q13 is not what we thought it was.** The router now
serves *some* requests: a 19-token probe succeeds on both configured models, but **129 of the
147 production-sized extraction requests still return `402 Payment Required`** while 18 return
`200 OK`, interleaved throughout the run. So it is not a clean account-level outage — it is
per-request, and 14 of 147 documents got a usable answer. Fourteen documents is not a
measurement of SLM field F1. **Every SLM-owned metric stays NOT MEASURED, blocked on Q13.** §6.

**Two defects this run exposed that the previous one could not see.** Neither is in this
lane and neither was touched:

- **Q14 — every line on a page is stamped with the *recognition head's* script**, not the
  script its own text is in. Harmless while the head was always Latin; now that the head
  varies, ASCII identifier lines on a Devanagari form are labelled `Deva`. **0 of 147
  recognised pages report more than one line script, against 30 of 147 that genuinely mix
  scripts**, so `is_code_mixed` can never fire. §7.1.
- **Q15 — the Tier-2 VLM model is not served at all.** The escalation leg fails with HTTP
  **400 `model_not_supported`**, not 402. It was indistinguishable from the credit failure
  before, because both surfaced as `PROVIDER_ERROR`. §7.2.

**`bilingual_form` gained nothing from Q12**, by construction: one recognition head per page
cannot read a page that is half Latin and half Devanagari. Its 60 Devanagari ground-truth
lines are still unread. That is now the largest single block of residual Devanagari error. §8.

---

## 1. Scorecard against the PRD success metrics

Targets from `design/PRD-rbi-ocr-service-v1.md` § Success Metrics (leading). The **pre**
column is the previous run, so every verdict change is visible.

| # | PRD target | Success | Pre-Q12 | **Measured now** | Verdict |
|---|---|---|---|---|---|
| 1 | Field F1, deterministic identifiers | ≥ 0.97 | 0.294 | **0.704** (TP 481 / FP 109 / FN 296; P 0.815, R 0.619) | **MISS** — but a real number now, not a band |
| 2 | Field F1, SLM fields | ≥ 0.80 | 0.005 | 0.028 over **14 of 147** eligible documents | **NOT MEASURED — blocked on Q13** |
| 3 | JSON schema validity rate | ≥ 99.5% | 1.000 | **1.000** (177/177 envelopes valid, 0 driver-side violations) | **PASS** |
| 4 | Silent identifier errors | **0** | 0 | **0** — 12 fields failed a validator, all 12 carry `VALIDATOR_FAILED` | **PASS** |
| 5 | Bounce rate (quality gate) | 5-15% | 16.9% | **16.9%** (30/177), on a corpus engineered to a 16.9% bounce profile | **PASS on intent, N/A on the band** — §9 |
| 6 | Escalation rate | ≤ 15% | 12.4% | **13.6%** (24/177 router escalations) | **PASS** |
| 7 | Router false-negative rate | ≤ 8% | not measured | n = 4 shadow-sampled; Tier 2 unavailable (Q15), so 0 shadow diffs completed | **NOT MEASURED** |
| 8 | p95 end-to-end, fast path | ≤ 90 s | 8.3 s | **3.90 s** p95 (1-page documents) | **PASS**, page-count caveat in §10 |
| 9 | Throughput per 16-vCPU node | ≥ 60 pages/min | — | not this machine | **OUT OF SCOPE** (adjudication A-10) |

Metrics the BUILD-BRIEF asks for that carry no numeric PRD target:

| Metric | Pre-Q12 | **Now** |
|---|---|---|
| CER / WER overall | 0.563 / 0.690 | **0.483 / 0.628** over 4 527 lines, 74 595 reference chars |
| CER by script | Latn 0.474, Deva **0.904** | Latn 0.474, Deva **0.518** |
| CER, clean profile only | 0.284; Latn print 0.099, Deva print **0.852** | **0.136**; Latn print 0.097, Deva print **0.046** |
| Detection mean IoU / line recall | 0.480 / 0.390 | 0.481 / 0.392 (clean 0.767 / 0.955) |
| Line-level LID accuracy | 0.802 matched; Deva **0.000** | **0.860** matched; Deva print **0.755**, Deva hand **0.417** |
| Quality-gate expectation-satisfied | 0.915 | 0.915 |
| Translations attempted / discarded | **0 / 0** — stage never fired | **5 / 0** — §5 |

---

## 2. How to read the before/after — three columns, not two

Two things changed since the previous run and they move the same numbers in different
directions. Attributing the movement needs three scorings, not two:

| col | predictions | corpus + harness | isolates |
|---|---|---|---|
| **A** | previous run's | generator 1.1.0, harness 1.0.0 | the published pre-Q12 baseline |
| **B** | **previous run's, unchanged** | generator 1.2.0, harness 1.1.0 | **Q6 alone** — same pipeline output, honest reference |
| **C** | this run's | generator 1.2.0, harness 1.1.0 | **Q12 on top of Q6** |

So **A→B is the reference fix and B→C is the pipeline fix.** B is the control that
`docs/decisions-q6.md` §8 also reports; it is reproduced here because without it the
deterministic-F1 jump reads as one improvement when it is two.

One wrinkle B carries: Q6 added three printed lines to every `cheque` (date, A/C number, MICR
band), so column B scores the previous run's cheque predictions against images they were never
produced from. Every cheque number in B is a floor, which is why `cheque` improves from B to C
partly for a bookkeeping reason. Ground-truth line count rises 4 419 → 4 527 for the same
reason. Nothing else in the corpus moved: 265 of 295 images are byte-identical.

### Recognition — the headline

| metric | A pre-Q12 | B Q6-only | **C post** |
|---|---|---|---|
| CER `script:Deva` | 0.904 | 0.904 | **0.518** |
| WER `script:Deva` | 0.976 | 0.976 | **0.732** |
| CER `script:Latn` | 0.474 | 0.482 | **0.474** |
| CER OVERALL | 0.563 | 0.569 | **0.483** |
| CER `profile:clean` | 0.284 | 0.290 | **0.136** |

Devanagari, every cell:

| cell | A | B | **C** | Δ (B→C) |
|---|---|---|---|---|
| Deva print clean | 0.852 | 0.852 | **0.046** | −0.806 |
| Deva print light | 0.870 | 0.870 | **0.090** | −0.780 |
| Deva print scan | 0.853 | 0.853 | **0.151** | −0.702 |
| Deva print heavy | 0.857 | 0.857 | **0.283** | −0.574 |
| Deva print photo | 0.884 | 0.884 | **0.756** | −0.128 |
| Deva hand clean | 0.846 | 0.846 | **0.249** | −0.597 |
| Deva hand light | 0.862 | 0.862 | **0.260** | −0.602 |
| Deva hand scan | 0.909 | 0.909 | **0.700** | −0.209 |
| Deva hand photo | 0.947 | 0.947 | **0.650** | −0.297 |
| Deva hand heavy | 0.935 | 0.935 | **0.781** | −0.154 |
| Deva any bounce | 1.000 | 1.000 | 1.000 | — no OCR on a bounced page: correct |

**Latin did not regress.** Every Latin handwriting cell moves by ≤ 0.002 between B and C, and
the Latin print cells improve only because the cheque predictions stopped being stale:

| cell | B | **C** |
|---|---|---|
| Latn hand clean | 0.255 | 0.255 |
| Latn hand scan | 0.564 | 0.564 |
| Latn hand light | 0.264 | 0.265 |
| Latn hand heavy | 0.690 | 0.692 |
| Latn hand photo | 0.816 | 0.812 |
| Latn print clean | 0.113 | 0.097 |

This matches the A/B measurement in `docs/decisions-q12.md` §2 — which asserted byte-identical
Latin output — through a different scorer and a different CER definition.

---

## 3. What the Devanagari fix bought downstream

| | pre-Q12 | **now** |
|---|---|---|
| Pages recognised with a Devanagari head | 0 | **24 of 30 Devanagari-majority pages** |
| Devanagari line-level LID, print | 0.000 | **0.755** |
| Devanagari line-level LID, hand | 0.000 | **0.417** |
| Overall LID (matched lines) | 0.802 | **0.860** |
| Devanagari-valued extracted fields | **0** | **29** |
| Translations attempted | **0** | **5** |
| `complaint_letter_mar` field F1 | 0.000 | **0.185** |
| `complaint_form_deva` field F1 | 0.260 (B) | **0.475** |

Language ID, per cell:

| cell | A | B | **C** | coverage now |
|---|---|---|---|---|
| Deva print | 0.000 | 0.000 | **0.755** | 0.806 |
| Deva hand | 0.000 | 0.000 | **0.417** | 0.784 |
| Latn hand | 0.889 | 0.889 | **0.890** | 0.771 |
| Latn print | 0.939 | 0.939 | **0.905** | 0.785 |
| OVERALL | 0.802 | 0.802 | **0.860** | 0.785 |

**The Latin-print cell fell 0.939 → 0.905 and that is not noise — it is Q14.** The drop is
entirely inside one document type: `complaint_form_deva`'s Latin-print cell went **1.000 →
0.068** across its 105 Latin ground-truth lines, while every other type's Latin-print cell is
unchanged to three decimals. §7.1 has the mechanism. Net LID is well up; a real regression is
hiding inside a net improvement, which is exactly what a rollup conceals.

`Deva|hand` at 0.417 has a second cause worth separating from Q12: of the Devanagari pages
that *are* now labelled, `docs/decisions-q12.md` §2 measured 8 of 30 as `mar` read as `hin`.
That is `lid.176` separating Marathi from Hindi on short lines — a Stage-2b property that
could not be measured at all while the stage was never handed Devanagari.

---

## 4. Fields — the band collapse, and what is actually left

**PRD target 1 population** (account number, IFSC, UTR/RRN, transaction reference, amounts,
dates, card fragment, bank name, MICR):

| | A pre-Q12 | B Q6-only | **C post** |
|---|---|---|---|
| TP | 156 | 402 | **481** |
| FP | 335 | 89 | **109** |
| FN | 414 | 375 | **296** |
| GT instances | 570 | 777 | 777 |
| precision | 0.318 | **0.819** | **0.815** |
| recall | 0.274 | 0.517 | **0.619** |
| **F1** | **0.294** | **0.634** | **0.704** |

**Read the two steps separately.**

- **A→B, precision 0.318 → 0.819, and this is the band collapsing.** Nothing about the
  pipeline changed across that column. The previous report had to write "the honest
  deterministic-F1 band is 0.29 to 0.46 depending on how you treat Q6", because `amount`
  (69 FP / 0 GT), `utr_rrn` (61 / 0) and `card_fragment` (24 / 0) had **no reference instance
  at all** — the corpus rendered those facts under different names, so every extraction of
  them scored as a false positive by construction. With the alias table those three phantom
  rows are gone, FP drops 335 → 89, and there is no longer a defensible way to treat the
  reference that yields 0.31. **The band is not narrowed; it is a single number.**
- **B→C, recall 0.517 → 0.619, and that is Q12.** Precision is flat (0.819 → 0.815); the
  extra 79 true positives are identifiers on Devanagari pages that were previously unreadable.

**Precision now exceeds recall on every deterministic field**, which inverts the previous
report's most interesting finding. That report saw precision *below* recall on three of four
fields and concluded the validators were over-claiming on noisy text. With an honest reference
and readable Devanagari they are not over-claiming — they are conservative. The remaining gap
to 0.97 is recall, and recall is a recognition problem, not a validator problem.

Per field:

| field | B TP/FP/FN | **C TP/FP/FN** | B F1 | **C F1** |
|---|---|---|---|---|
| `micr_code` | 0/0/18 | **13/2/5** | n/a | **0.788** |
| `ifsc_code` | 65/6/25 | 62/7/28 | 0.807 | 0.780 |
| `card_last_four` | 22/2/11 | 22/2/11 | 0.772 | 0.772 |
| `transaction_date` | 86/41/37 | **101/43/22** | 0.688 | **0.757** |
| `bank_name` | 71/12/52 | **81/13/42** | 0.689 | **0.747** |
| `transaction_reference` | 55/6/68 | **65/7/58** | 0.598 | **0.667** |
| `account_number` | 45/11/81 | **70/16/56** | 0.495 | **0.660** |
| `disputed_amount` | 58/11/83 | **67/19/74** | 0.552 | **0.590** |
| `prior_complaint_ref` (SLM) | 2/2/70 | 4/2/68 | 0.053 | 0.103 |
| `complainant_name` (SLM) | 1/1/107 | 1/0/107 | 0.018 | 0.018 |
| `complainant_address` (SLM) | 0/1/54 | 1/0/53 | n/a | 0.036 |
| `complaint_summary` (SLM) | 0/0/90 | 0/0/90 | n/a | n/a |
| `complainant_contact` (SLM) | 0/0/54 | 0/0/54 | n/a | n/a |
| `relief_sought` (SLM) | 0/5/36 | 0/1/36 | n/a | n/a |

`micr_code` is the clearest illustration of what Q6 bought: it did not exist in the corpus
before, appeared in column B as a pure miss (0/0/18) because the reference finally named it,
and in column C scores 0.788 because the cheque template now draws it. A field went from
invisible, to a measured miss, to a measured pass, without the extractor changing.

**Invariant 9 holds.** 12 fields failed their validator; all 12 carry `VALIDATOR_FAILED`;
none reached the envelope unflagged. Verified directly over the 177 prediction files.

**The bilingual `value` slot is a contract question, not a score.** Q6 added a diagnostic for
"did the pipeline put the page's own string in the immutable `value`, or the translation":

| | B Q6-only | **C post** |
|---|---|---|
| GT bilingual field occurrences | 198 | 198 |
| claimed | 2 | **13** |
| `value` = the page's own string | **0** | **6** |
| `value` = its translation | 0 | **0** |
| `value_en` correct / wrong / absent | 0 / 0 / 2 | **2 / 1 / 10** |

Zero contract violations — no field put a translation in `value`. The block is still nearly
empty (13 of 198 claimed) because the SLM owns most bilingual fields and the SLM is blocked.
`docs/decisions-q6.md` §7 leaves the owner a call on whether to promote this into precision
before `--final`; on this evidence it would cost nothing today.

---

## 5. Translation and Q7 — the stage fires, the rate is n = 5

Stage 6 ran for the first time. Measured straight out of the committed predictions, which is
the per-run measurement Q7 asked WS-H to wire in:

| | pre-Q12 | **now** |
|---|---|---|
| Extracted fields | 503 | 599 |
| Fields whose `value` is Devanagari | **0** | **29** |
| Translations attempted | **0** | **5** |
| Returned a `value_en` | 0 | **5** |
| Discarded, flagged `TRANSLATION_UNAVAILABLE` | 0 | **0** |
| **Sentinel-survival rate** | **not measurable** | **5/5 = 1.000 (n = 5)** |

`mt_engine` is `indictrans2-indic-en-dist-200M-ct2-int8` on all five — the real weights.
Examples:

| field | value | `value_en` |
|---|---|---|
| `complainant_name` | राजेश कुमार शर्मा | Rajesh Kumar Sharma |
| `complainant_address` | पनवेल महाराष्ट् | Panvel Maharashtra |
| `relief_sought` | कृपया योग्य ती कार्यवाही करावी ही विनती | Please take the appropriate action |

**Do not read 1.000 as a rate.** n = 5. Q7 is *unblocked*, not *answered*: the WS-F
corpus-grounded figure (54/54 entities intact, `docs/decisions-wsf.md` D-F-15) remains the
best evidence on the record, and this run is consistent with it rather than a replacement for
it. The binding constraint moved from Q12 to Q13 — 24 of the 29 Devanagari-valued fields never
reached MT because they are deterministic fields that resolve by lookup or regex (`bank_name`
carries a native value and an English `normalized` from the bank table, which is not a
translation), and the SLM that owns the free-text fields answered for only 14 documents.

**One observation for the Q7 / WS-F owner, not a defect claim.** One of the five translated
fields had an *ASCII* value: `prior_complaint_reference` = `CMS/2025/850859` came back as
`CMS / 2025 / 850859`. `value` is untouched and `value_en` is advisory, so nothing downstream
is wrong, but an identifier was handed to the MT engine and came back re-spaced. Worth a look
at Stage 6's eligibility test when someone is next in that file.

---

## 6. SLM reachability — Q13 is real but it is not an outage

The previous report concluded the HF account's credits were depleted account-wide, because
both configured models answered `402`. That is no longer the shape of it.

**One probe per model, before the sweep** (19 tokens each — the cap this run held itself to):
`google/gemma-3-4b-it` → **200 OK**. `meta-llama/Llama-3.1-8B-Instruct` → **200 OK**.
Both models are reachable and the token is valid.

**The sweep itself**, 147 SLM-eligible documents:

| outcome | documents |
|---|---|
| bounced — SLM never invoked | 30 |
| **router refused with `402 Payment Required`** | **129** |
| SLM answered, failed Pydantic twice (initial + repair) | 4 |
| SLM answered and validated | **14** (8 cache hits, 6 live) |

HTTP status tally across the whole run: **18 × 200, 129 × 402** for the SLM, and the 200s are
**interleaved throughout**, not front-loaded. So this is not "credits ran out at document 3".
Production-sized requests are refused per-request while a trivial one succeeds, consistent
with the router load-balancing across backend inference providers where only some are on the
account's plan — the same mechanism Q5 measured for `json_schema` support.

**Consequences, unchanged in substance:**

- PRD target 2 (SLM field F1 ≥ 0.80) is **NOT MEASURED**. 14 of 147 documents is not a sample.
  For the record the harness scores 6 TP / 3 FP / 408 FN → P 0.667, R 0.014, F1 0.028; the
  precision is the only interesting digit and it rests on nine claims.
- **Q10 stays blocked.** Whether `gemma-3-4b-it`'s 0.5625 schema validity caps field recall
  against the fallback's 1.000 needs a run where the model is actually asked 147 times. Of the
  answers that did arrive, 14 of 18 validated — *higher* than the bake-off predicted, on a
  sample far too small to update on.
- The fallback never engaged, correctly: a 402 maps to `ProviderError`, which is not in the
  driver's outage set, so the sweep did not thrash between models. One model, one request per
  document, no retry storm — a 402 is not retryable by `_is_retryable`, so each refusal cost
  exactly one round trip.

**This needs credits or a PRO subscription on the owner's account. It is not a code change,
and this agent made no purchase and will not.**

---

## 7. Two defects this run exposed

Both were invisible before. Neither is in this lane; documented, not fixed.

### 7.1 Q14 — a line's `script` is the recognition head's script, not the line's

`app/providers/local.py::LocalProvider.ocr_page` constructs every `OCRLine` with
`script=script`, where `script` is the head the page was recognised with. The line's own text
is never consulted. While the head was unconditionally Latin this was invisible — most ground
truth is Latin, so it was accidentally right. Now that Q12 makes the head vary, it is
accidentally wrong in the other direction.

On `complaint_form_deva_v0_clean`, post-fix:

| line text | GT script | published `script` | published `language` |
|---|---|---|---|
| राजेश कुमार शर्मा | Deva | Deva | hin |
| `9106990984` | **Latn** | **Deva** | None |
| `SBIN0796790` | **Latn** | **Deva** | None |
| `77592400665811` | **Latn** | **Deva** | None |
| `28/03/2026` | **Latn** | **Deva** | None |

Measured consequences:

- `complaint_form_deva`'s Latin-print LID cell: **1.000 → 0.068** over 105 Latin ground-truth
  lines. This is the whole of the overall Latin-print decline (0.939 → 0.905).
- **0 of 147 recognised pages carry more than one distinct line script**, while **30 of 147
  non-bounce dev pages genuinely mix scripts** (15 `bilingual_form`, 15 `complaint_form_deva`).
- Therefore `is_code_mixed` — which `assemble.derive_language_summary` computes by comparing
  line scripts against the page majority, with `CODE_MIX_THRESHOLD = 0.15` — **can never be
  true**. That is the PRD checklist item "is_code_mixed set when >= 15% of lines differ in
  script from the page majority", silently unreachable.

The fix is small and is not ours: derive the line's script from its own decoded characters
(the codepoint classifier `s4_ocr` already uses for the A-6 vote) and keep the head's script as
provenance if it is worth keeping. **Lane: WS-C / WS-E2.** Raised as **Q14**.

### 7.2 Q15 — the Tier-2 VLM model is not served by any enabled provider

One diagnostic request to `PaddlePaddle/PaddleOCR-VL` returns, verbatim:

```
HTTP 400  {'message': "The requested model 'PaddlePaddle/PaddleOCR-VL' is not supported
           by any provider you have enabled.",
           'type': 'invalid_request_error', 'param': 'model', 'code': 'model_not_supported'}
```

This is **not** the credit failure. It is a 400, it is deterministic, and it reproduces on an
empty 1x1 image, so it is nothing about page size or content. In this run 28 Tier-2 attempts
were made and **`tier2_applied` is 0**; in the previous run all 25 attempts failed too, but
both legs surfaced as `PROVIDER_ERROR`, so the 402 and the 400 were indistinguishable and the
report attributed both to credits.

The escalation *decision* path is exercised and correct (§9). The escalation *outcome* path
has never once run, and it will not run against this model on this account regardless of
credits. That also means PRD target 7 cannot be measured — the shadow sampler drew 4 documents
and every diff is empty for want of a Tier-2 answer.

This reopens **Q2** (Tier-2 escalation model, which the PRD leaves unnamed and the brief
proposed as PaddleOCR-VL-0.9B). **Lane: WS-C, owner decision on the model.** Raised as **Q15**.

---

## 8. Remaining MISSes, with causes

| what misses | measured | cause | owner |
|---|---|---|---|
| Deterministic field F1 0.704 vs >= 0.97 | P 0.815 / R 0.619 | **Recall, and recall is recognition.** Precision is healthy on every field; the gap is instances never read. Concentrated in degraded profiles (photo CER 0.654, heavy 0.525) and in `bilingual_form` | WS-E2 (recognition), WS-E1 (quality/repair) |
| `bilingual_form` unchanged: field F1 0.387, CER 0.509, Deva-hand LID 0.000 | 368/368 lines recognised as `Latn`; 60 Deva GT lines unread | **One head per page cannot read a bilingual page.** The Q12 probe samples 3 crops and settles the *page*; on a genuinely half-and-half form the honest answer is "both", which the design has no way to express | WS-E2 — the natural follow-on to Q12 |
| Deva print photo 0.756, barely moved from 0.884 | 6 documents | `complaint_form_deva_v1_photo` returns `inconclusive_tie` (heads 0.033 apart, under the 0.05 margin) and keeps pre-fix behaviour by design; photo is where the probe has least signal | WS-E2 (Q12 follow-up 4) |
| `complaint_letter_mar` field F1 0.185 | CER 0.503, LID 0.538 | Recognition is fixed, but the fields on a Marathi *letter* are free text, and free text is the SLM's, which is blocked | Q13 |
| SLM field F1 | 14/147 documents | Q13 | owner |
| Router false-negative rate | 0 completed shadow diffs | Q15 — Tier 2 never lands | WS-C |
| `complaint_letter_en` field F1 0.318, 35 FP | CER 0.650 | Worst Latin CER on the corpus; an English letter is dense prose and the deterministic layer over-claims on it — the one place the previous report's over-claiming finding still holds | WS-E3 |
| `is_code_mixed` never true | 0/147 vs 30/147 | Q14 | WS-C / WS-E2 |

---

## 9. Quality gate, escalation, detection

### Quality gate — unchanged and still the strongest result

```
expectation  accept  repair  bounce
clean            42      15       0
degraded         15      75       0
bounce            0       0      30
```

Expectation-satisfied **0.915**, identical to the previous run. No `clean` page bounced; all 30
`bounce`-profile pages bounced; one document moved `accept` to `repair` within `degraded`,
which satisfies the expectation either way. The 16.9% bounce rate reproduces the corpus's
designed 16.9% almost exactly.

**The caveat is unchanged and still the sharpest one here:** the Q3 thresholds were calibrated
against this corpus using variants 0-2 — the split this report scores
(`docs/decisions-wse1.md` §3). This row is the most optimistic number in the report and the
locked split is what will test it.

### Escalation

24 documents escalated (**13.6%**, up from 12.4%), precision **0.958**, recall 0.174 against
the `quality_or_hand` stand-in reference. The two extra escalations and the single false
positive are consistent with `script_disagreement` now being reachable — Q12 raises it when
the probe and the A-6 vote disagree, which `docs/decisions-q12.md` D-Q12-8 measured firing once
in 147. Recall against a stand-in reference with no corpus label behind it (adjudication A-3)
**is not a target and should not be read as one**.

`tier2_applied` remains **0** for the reason in §7.2.

### Detection

| profile | mean IoU | line recall |
|---|---|---|
| clean | 0.767 | 0.955 |
| light | 0.543 | 0.576 |
| scan | 0.450 | 0.448 |
| heavy | 0.356 | 0.285 |
| photo | 0.266 | 0.123 |
| bounce | n/a | 0.000 |

Essentially unchanged, as expected — Q12 changes which head reads the crops, not which crops
are found. Both known measurement effects still apply and are still caveats rather than
excuses: axis-aligned IoU on rotated ground truth (WS-D §6), and the coordinate-frame mismatch
on deskewed pages raised as **Q11**, which remains open and undiagnosed.

---

## 10. Latency

| stage | pre p50 | **now p50** | pre p95 | **now p95** |
|---|---|---|---|---|
| **end-to-end** | 1 210 | **1 429** | 8 258 | **3 895** |
| `s4_ocr` | 564 | **760** | 1 540 | **1 678** |
| `s5_slm` | 279 | 227 | 9 243 | 2 506 |
| `s15_quality` | 194 | 195 | 215 | 220 |
| `s3_classify` | 154 | 161 | 298 | 306 |
| `s4b_vlm` | 1 336 | 1 414 | 1 749 | 1 698 |
| `s5_validators` | 4.8 | 5.5 | 14.3 | 15.7 |
| `s2b_lid` | 1.0 | 1.4 | 5.2 | 5.4 |

**`s4_ocr` p50 +196 ms is the Q12 probe, and it is at the edge of its budget.**
`docs/decisions-q12.md` §2 predicted p50 582 → 771 ms; measured here through a different
harness, 564 → 760 ms. That is ~26% of the pre-fix stage at p50 against a 25% budget — the
honest reading is "budget met to within measurement noise", not "comfortably inside". The probe
is the one place this pipeline pays for Q12, and `recognise_crops` on the provider protocol
(`docs/decisions-q12.md` §5) would roughly halve it.

**Do not read the end-to-end p95 improvement (8 258 → 3 895 ms) as a speed-up.** It is the SLM
leg failing faster and more often. p95 here is dominated by how long the router takes to
refuse, which is not a property of this service. PRD target 8 passes by a wide margin with the
same caveats as before: these are **one-page** documents against a 5-page target, and a 5-page
document with five *real* SLM calls on this laptop would be in the 30-60 s range. The sweep is
strictly sequential (D-H2-2), so nothing was measured under self-inflicted contention.

---

## 11. Per-type field metrics

Field TP/FP/FN aggregated per type, with that type's CER. Column B is the previous run's
predictions against the fixed reference, so B→C is Q12 alone.

| type | docs | GT | B TP/FP/FN | **C TP/FP/FN** | B F1 | **C F1** | B CER | **C CER** |
|---|---|---|---|---|---|---|---|---|
| `receipt` | 18 | 90 | 67/5/23 | 67/5/23 | 0.827 | **0.827** | 0.375 | 0.375 |
| `photo_evidence` | 15 | 75 | 53/2/22 | 53/2/22 | 0.815 | **0.815** | 0.461 | 0.461 |
| `bank_statement` | 18 | 54 | 42/17/12 | 42/17/12 | 0.743 | **0.743** | 0.418 | 0.418 |
| `cheque` | 18 | 126 | 28/2/98 | **63/3/63** | 0.359 | **0.656** | 0.567 | **0.378** |
| `complaint_form_en` | 18 | 198 | 91/1/107 | 92/1/106 | 0.628 | **0.632** | 0.488 | 0.488 |
| `email_printout` | 18 | 36 | 15/0/21 | 15/0/21 | 0.588 | **0.588** | 0.635 | 0.635 |
| `complaint_form_deva` | 18 | 198 | 30/3/168 | **62/1/136** | 0.260 | **0.475** | 0.772 | **0.364** |
| `bilingual_form` | 18 | 198 | 52/19/146 | 52/19/146 | 0.387 | **0.387** | 0.509 | 0.509 |
| `complaint_letter_en` | 18 | 108 | 27/35/81 | 27/35/81 | 0.318 | **0.318** | 0.650 | 0.650 |
| `complaint_letter_mar` | 18 | 108 | 0/14/108 | **14/29/94** | 0.000 | **0.185** | 0.900 | **0.503** |

**The four types Q6 named are no longer at the bottom of this table.** `receipt` 0.000 →
0.827, `photo_evidence` 0.000 → 0.815, `email_printout` 0.000 → 0.588 — all measured against
the same predictions that scored zero, because the reference now names the fields those pages
draw. `receipt` at CER 0.375 with field F1 0.827 is a coherent pair; CER 0.375 with F1 0.000
never was.

`cheque` 0.359 → 0.656 is the one row where B is a bookkeeping floor (its predictions were
produced against pre-Q6 images), so read that improvement as mostly artefact.

`bilingual_form` and `complaint_letter_en` are the two rows that did not move at all, and §8
gives the cause for each.

---

## 12. Run provenance

| | |
|---|---|
| Documents | 177 selected, 177 written, **0 failed**, 0 timeouts, 0 overlap events |
| Wall clock | 329 s (5 min 29 s), strictly sequential |
| Schema self-check | 0 violations on 177 written predictions |
| Provider cache | 52 rows before, 58 after; 8 hits / 167 misses (hit rate 0.046) |
| SLM tokens settled | 20 734 |
| SLM model | `google/gemma-3-4b-it`; fallback `meta-llama/Llama-3.1-8B-Instruct` configured, **not engaged** |
| Tier-2 VLM | `PaddlePaddle/PaddleOCR-VL` — **not served** (§7.2) |
| Local weights | `models/ocr/{det,rec_latn,rec_deva}.onnx`, `models/lid/lid.176.ftz`, `models/mt/indic-en` (CT2 INT8) |
| Page classifier | CNN artifact absent; fitted structural path ran for all 147 non-bounced pages |

Cache hit rate fell 0.093 → 0.046, which is expected: Q12 changes the OCR text on every
Devanagari page, so those prompts are new keys. The 8 hits are Latin pages whose transcription
is byte-identical to the previous run — themselves a small independent confirmation of the
no-regression claim in §2.

**One environment note that affects reproducibility.** Python could not complete a TLS
handshake to the router on this machine: Norton's inspection CA is installed and trusted by
the OS and by Node (`NODE_EXTRA_CA_CERTS`) but is not in `certifi`'s bundle, so every request
failed `CERTIFICATE_VERIFY_FAILED`. This run sets `SSL_CERT_FILE` to a bundle concatenating
`certifi` with the machine's existing Norton root, for the run's process only. No system
setting, no certificate store and no repository file was modified, and verification was not
disabled. Anyone reproducing this on a machine with TLS inspection will need the same.

---

## 13. The harness output, verbatim

Everything below is `tools/eval_harness.py` 1.1.0's own output, unedited, copied from
[`report.txt`](report.txt). Stage rows ending `:p1` are the per-page split of the row above;
on a single-page corpus they are identical by construction (D-H2-11).

```text
==============================================================================
RBI OCR corpus evaluation — harness 1.1.0
==============================================================================
corpus       E:\pythonprojects\python_ocr\corpus  (seed 1729, generator 1.2.0)
predictions  E:\pythonprojects\python_ocr\docs\eval\predictions
split        dev  final=False  locked variants [3, 4]
documents    177 scored / 177 selected / 118 withheld

Recognition — CER / WER by script x modality x profile
-----------------------------------------------------
cell               lines  ref_chars    CER    WER
-------------------------------------------------
Deva|hand|bounce      51       1524  1.000  1.000
Deva|hand|clean       51       1524  0.249  0.416
Deva|hand|heavy       51       1527  0.781  1.033
Deva|hand|light       51       1518  0.260  0.482
Deva|hand|photo       51       1521  0.650  0.955
Deva|hand|scan        51       1511  0.700  0.927
Deva|print|bounce     54       1035  1.000  1.000
Deva|print|clean      54       1035  0.046  0.254
Deva|print|heavy      54       1035  0.283  0.726
Deva|print|light      54       1035  0.090  0.418
Deva|print|photo      54       1035  0.756  0.945
Deva|print|scan       54       1035  0.151  0.542
Latn|hand|bounce      69       1827  1.000  1.000
Latn|hand|clean       69       1824  0.255  0.333
Latn|hand|heavy       69       1820  0.692  0.911
Latn|hand|light       69       1810  0.265  0.404
Latn|hand|photo       69       1824  0.812  1.044
Latn|hand|scan        69       1830  0.564  0.770
Latn|print|bounce    588       8110  1.000  1.000
Latn|print|clean     543       7698  0.097  0.128
Latn|print|heavy     588       8106  0.470  0.629
Latn|print|light     588       8145  0.197  0.340
Latn|print|photo     588       8123  0.607  0.786
Latn|print|scan      588       8143  0.288  0.431

Recognition — rollups
--------------------------------------------------
cell            lines  ref_chars    CER    WER
----------------------------------------------
script:Deva       630      15335  0.518  0.732
script:Latn      3897      59260  0.474  0.594
profile:bounce    762      12496  1.000  1.000
profile:clean     717      12081  0.136  0.215
profile:heavy     762      12488  0.525  0.737
profile:light     762      12508  0.205  0.378
profile:photo     762      12503  0.654  0.865
profile:scan      762      12519  0.367  0.562
OVERALL          4527      74595  0.483  0.628

Detection — IoU and line recall by profile
----------------------------------------------------------------
profile  gt_lines  matched  spurious  mean_IoU  line_recall
-----------------------------------------------------------
bounce        762        0         0       n/a        0.000
clean         717      717         9     0.767        0.955
heavy         762      675        91     0.356        0.285
light         762      754         8     0.543        0.576
photo         762      672       103     0.266        0.123
scan          762      736        37     0.450        0.448
OVERALL      4527     3554       248     0.481        0.392

Fields — precision / recall / F1 by field
--------------------------------------------------------------
field                   TP  FP   FN      P      R     F1
--------------------------------------------------------
account_number          70  16   56  0.814  0.556  0.660
bank_name               81  13   42  0.862  0.659  0.747
card_last_four          22   2   11  0.917  0.667  0.772
complainant_address      1   0   53  1.000  0.019  0.036
complainant_contact      0   0   54    n/a  0.000    n/a
complainant_name         1   0  107  1.000  0.009  0.018
complaint_summary        0   0   90    n/a  0.000    n/a
disputed_amount         67  19   74  0.779  0.475  0.590
ifsc_code               62   7   28  0.899  0.689  0.780
micr_code               13   2    5  0.867  0.722  0.788
prior_complaint_ref      4   2   68  0.667  0.056  0.103
relief_sought            0   1   36  0.000  0.000    n/a
transaction_date       101  43   22  0.701  0.821  0.757
transaction_reference   65   7   58  0.903  0.528  0.667

Fields — by source mechanism (R = share of ALL reference fields this source captured)
---------------------------------------------------------
source            TP  FP    GT      P      R     F1
---------------------------------------------------
lookup_table      81  13  1191  0.862  0.068  0.126
regex_validator  400  96  1191  0.806  0.336  0.474
slm                6   3  1191  0.667  0.005  0.010

Fields — field x source (R = share of that field's reference this source captured)
------------------------------------------------------------------------------
field|source                            TP  FP   GT      P      R     F1
------------------------------------------------------------------------
account_number|regex_validator          70  16  126  0.814  0.556  0.660
bank_name|lookup_table                  81  13  123  0.862  0.659  0.747
card_last_four|regex_validator          22   2   33  0.917  0.667  0.772
complainant_address|slm                  1   0   54  1.000  0.019  0.036
complainant_name|slm                     1   0  108  1.000  0.009  0.018
disputed_amount|regex_validator         67  19  141  0.779  0.475  0.590
ifsc_code|regex_validator               62   7   90  0.899  0.689  0.780
micr_code|regex_validator               13   2   18  0.867  0.722  0.788
prior_complaint_ref|slm                  4   2   72  0.667  0.056  0.103
relief_sought|slm                        0   1   36  0.000  0.000    n/a
transaction_date|regex_validator       101  43  123  0.701  0.821  0.757
transaction_reference|regex_validator   65   7  123  0.903  0.528  0.667

Fields — bilingual slots (fields whose native spelling differs from its English)
---------------------------------
metric                     count
--------------------------------
gt occurrences               198
claimed                       13
value = page's own string      6
value = its translation        0
value_en correct               2
value_en wrong                 1
value_en absent               10
    `value` is contractually the page's own string; a translation there is a contract violation this table counts but precision/recall cannot see.

Schema validity
---------------------
metric         value
--------------------
checked          177
valid            177
validity_rate  1.000

Escalation vs reference rule 'quality_or_hand'
------------------------------------------
TP  FP   FN  precision  recall     F1
-------------------------------------
23   1  109      0.958   0.174  0.295

Quality gate — confusion (rows: quality_expectation, cols: verdict)
--------------------------------------
expectation  accept  repair  bounce
-----------------------------------
clean            42      15       0
degraded         15      75       0
bounce            0       0      30
    expectation-satisfied rate: 0.915 over 177 documents

Language ID — LINE level
----------------------------------------------------------------------------
cell        gt_lines  matched  correct  acc_matched  acc_all  coverage
----------------------------------------------------------------------
Deva|hand        306      240      100        0.417    0.327     0.784
Deva|print       324      261      197        0.755    0.608     0.806
Latn|hand        414      319      284        0.890    0.686     0.771
Latn|print      3483     2734     2474        0.905    0.710     0.785
OVERALL         4527     3554     3055        0.860    0.675     0.785

Latency (ms) by stage
------------------------------------------
stage             n       p50       p95
---------------------------------------
assemble         28     0.260     0.501
e2e_total       177  1429.051  3895.158
s0_read         177     0.961     1.873
s15_quality     177   195.298   219.678
s15_quality:p1  177   195.298   219.678
s1_ingest       177    18.720    47.024
s2b_lid         147     1.373     5.408
s2b_lid:p1      147     1.373     5.408
s3_classify     147   161.397   306.357
s3_classify:p1  147   161.397   306.357
s4_ocr          147   760.099  1678.026
s4_ocr:p1       147   760.099  1678.026
s4b_vlm          28  1413.543  1697.502
s4b_vlm:p1       28  1413.543  1697.502
s5_slm          147   227.432  2505.679
s5_validators   147     5.538    15.742


[... `Latency (ms) by stage x profile` omitted for length; it is in report.txt ...]

CAVEAT  Font-rendered handwriting is not real handwriting. This corpus validates plumbing, schema, routing and throughput. It does NOT validate handwriting accuracy — measure that on iiit-indic-hw-words and real complaint pages.
NOTE    variants [3, 4] are the locked test split and were not scored. Pass --split all --final to include them.
```

---

## 14. The caveats this corpus always carries

Stated in full because every number above inherits them. Unchanged from the previous report
except where Q12 changed what they mean.

1. **Synthetic handwriting is not handwriting.** Every `hand` cell is font-rendered glyph
   jitter. Those cells validate plumbing, routing, batching and throughput; they do **not**
   validate handwriting accuracy. This matters more now, not less: Deva-hand-clean CER 0.249
   is a genuinely new number and it is *not* evidence the service reads handwritten
   Devanagari. The real number needs `iiit-indic-hw-words` and real complaint pages.
2. **The locked split is held out.** Variants 3 and 4 — 118 documents — were neither run nor
   scored. Everything here is dev-split and optimistic to exactly the degree the build was
   tuned against variants 0-2. The quality gate was (§9); Q12's probe geometry and its 0.05
   score margin were too (`docs/decisions-q12.md` D-Q12-4, D-Q12-5). **Q12 is now a second
   thing the locked split has to test honestly**, and publishing a final number means
   `--split all --final`, once, on code nobody has tuned since.
3. **Q6's remaining owner calls.** The corpus and the envelope still use two field
   vocabularies bridged by an alias table, and the bilingual `value`-slot check is a
   diagnostic rather than part of precision. Both are in `docs/decisions-q6.md` §10 and both
   should be settled before `--final`.

---

## 15. What to fix, in priority order

1. **Restore SLM credits, then re-run** (owner). Two PRD targets, Q10, and half of Q7's
   evidence sit behind it. The run is resumable and costs the OCR only (~5.5 min).
2. **Name a Tier-2 model the router will actually serve** (Q15, WS-C + owner). The escalation
   outcome path has never executed; PRD target 7 cannot be measured until it does.
3. **Derive a line's script from its own characters** (Q14, WS-C / WS-E2). Small change,
   restores `is_code_mixed`, and removes the one real regression in this report.
4. **Teach recognition to mix heads within a page** (WS-E2, the Q12 follow-on). This is what
   `bilingual_form` needs, and `bilingual_form` is now the largest single block of residual
   Devanagari error.
5. **Decide whether evidence boxes must be mapped back to the pre-repair frame** (Q11,
   WS-E1 / WS-G). Unchanged, still open, still undiagnosed.
6. **Re-run the escalation reference once a real label exists** (WS-E2, relates to Q3/A-3).
   Precision 0.958 against a stand-in remains encouraging and not yet meaningful.

---

## 16. Method, and how to disagree with it

- **The scorer imports nothing from `app/`** (adjudication A-2) and the predictions are
  committed. Anyone can re-score them, change the escalation reference or the IoU threshold,
  or write a different scorer, with no model directory and no token.
- **The three-column method (§2) is the only honest way to read this run.** Two independent
  changes landed at once; a two-column table would attribute all of it to whichever one the
  reader already believed in.
- **`escalated` is the router's decision**, not the envelope's `escalation_flag`, which ORs in
  "any field needs review". See `docs/decisions-wsh2.md`.
- **A failed document is scored, not skipped** — written with `pages: []` and `fields: {}` so
  its lines count as deletions. There were none on this run.
- **Recall by source decomposes the total**: each mechanism is scored against the whole ground
  truth, so `lookup_table` 0.068 + `regex_validator` 0.336 + `slm` 0.005 sums to overall
  recall. A mechanism cannot score 1.0 by extracting one field well and ignoring the rest.
- **Latency is single-threaded and sequential by design.** Parallelising would have halved the
  wall clock and invalidated PRD target 8.

Decisions and reasoning for this run: [`docs/decisions-wsh3.md`](../decisions-wsh3.md).
Prior run: [`docs/decisions-wsh2.md`](../decisions-wsh2.md). The fixes it measures:
[`docs/decisions-q12.md`](../decisions-q12.md), [`docs/decisions-q6.md`](../decisions-q6.md).
