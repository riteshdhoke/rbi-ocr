# M7 exit-gate evaluation — RBI complaint document intelligence (SLM-live re-run)

**Run:** 2026-08-08, dev split (variants 0-2), 177 documents, 0 failures, **1 370 s** wall clock
(22 min 50 s). Manifest timestamps are UTC: `21:17:06Z` → `21:39:57Z`.
**Code:** `bcd85dd950dd08cbfbf91d0131b440bd7eba37ba`. **No source file was changed for this run.**
**Driver:** `tools/run_corpus_eval.py` 1.0.0 · **Scorer:** `tools/eval_harness.py` 1.1.0
· **Corpus generator:** 1.2.0 (seed 1729)
**Supersedes:** [`REPORT-2026-08-08-preSLM.md`](REPORT-2026-08-08-preSLM.md) — the same
pipeline with the SLM unreachable. Kept beside this file so every claim below is checkable
against what it replaces. Earlier still: [`REPORT-2026-08-07-preQ12.md`](REPORT-2026-08-07-preQ12.md).
**Artifacts:** [`PREDICTIONS_MANIFEST.json`](PREDICTIONS_MANIFEST.json) ·
[`report.json`](report.json) · [`report.txt`](report.txt) (reproduced verbatim in §12) ·
[`predictions/`](predictions) (177 files — the report regenerates from these with no models
and no token).

Reproduce the score without a pipeline:

```
.venv\Scripts\python.exe tools/eval_harness.py --corpus corpus --predictions docs/eval/predictions
```

---

## Read this first

**Q13 is resolved. The endpoint refused nothing.** Across 147 SLM-eligible documents the router
returned **zero `402 Payment Required`**, against **129** in the previous run. Credits were
topped up and verified before the sweep with one 19-token probe per model *and* one
production-sized probe (2 032 prompt tokens, `max_tokens=512`, `json_object` mode) — all three
returned 200. The sweep then ran live and spent **286 198 tokens**, against 20 734 last time.
The SLM was genuinely exercised for the first time. §5.

**Every non-SLM number in this report is byte-identical to the previous run.** Recognition CER
0.4830216502446545 both times, to sixteen digits. Detection, language ID, the quality-gate
confusion matrix and the escalation counts are identical integers. That is not a coincidence to
explain away — it is the control this run was designed to have. **Nothing but SLM reachability
changed, so every delta below is the SLM and only the SLM.** §2.

**PRD target 2 is MEASURED for the first time, and it is a MISS: SLM field F1 0.119**
(P 0.675, R 0.065), up from 0.028. Not "not measured" any more — every one of the 147 eligible
documents was asked, and the answer is that this configuration is nowhere near 0.80. §3.

**The binding constraint moved from reachability to schema validity, and Q10 now has its
answer.** The model answered every request; **54 of 147 answers survived validation — a schema
validity of 0.367**. The bake-off predicted 0.5625 from 16 observations; measured over 147 it is
**worse, not better**. 93 documents produced nothing at all from the SLM leg. **Recall is capped
at 0.367 by this alone**, before any question of answer quality, so the configured model cannot
reach the 0.80 target no matter how good its surviving answers are. §4.

**The grounding gate is not the bottleneck, and that is worth saying plainly.** Of 104 values
the model proposed, the gate rejected 64 — but **61 of those were rejected as `empty`** (the
model declined to answer) and only **3 as `ungrounded`** (a value not on the page). The gate
threw out 2.9% of proposals as unsupported. Invariant 8 is doing its job at almost no cost to
recall. §4.3.

**Latency is now honest, and still passes.** End-to-end p95 **3 895 → 22 609 ms**. The previous
run's fast p95 was the router refusing quickly; this one is the model working. Against PRD
target 8 (p95 ≤ 90 s) that is still a **PASS**, with much less headroom than the old number
implied. §7.

**Two measurement-chain gaps this run made visible.** Neither is in this lane; both are
documented, not fixed. Together they account for **144 of the SLM bucket's 414 misses** — a
third of the gap — and neither is a model failure:

- **`complaint_summary` is produced but cannot be scored.** 47 documents carry a summary
  totalling 198 spans, yet the field scores **0 TP / 0 FP / 90 FN** because
  `prediction_from_result` publishes it as the integer `summary_span_count` rather than as a
  scorable field. Drafted as **Q16**. §6.1.
- **`complainant_contact` has no extractor at all.** It has 54 reference instances, it is in the
  envelope vocabulary, and it is in **neither** `SLM_ELIGIBLE_FIELDS` nor the validator table.
  54 guaranteed misses, by construction, in a bucket that is a published PRD target. Drafted as
  **Q17**. §6.2.

**Q15 is unchanged and still blocks PRD target 7.** 28 Tier-2 attempts, `tier2_applied` **0**,
HTTP 400 `model_not_supported`. Credits were never the cause. §6.3.

---

## 1. Scorecard against the PRD success metrics

Targets from `design/PRD-rbi-ocr-service-v1.md` § Success Metrics. The **pre** column is the
previous run — same code, same pixels, SLM unreachable — so every verdict change is the SLM.

| # | PRD target | Success | Pre (SLM blocked) | **Measured now** | Verdict |
|---|---|---|---|---|---|
| 1 | Field F1, deterministic identifiers | ≥ 0.97 | 0.704 | **0.704** — TP 481 / FP 109 / FN 296; P 0.815, R 0.619, **identical to sixteen digits** | **MISS**, unchanged — and see the note below |
| 2 | Field F1, SLM fields | ≥ 0.80 | 0.028, *not measured* | **0.119** (P 0.675, R 0.065) over **147 of 147** eligible documents | **MEASURED — MISS.** §3, §4 |
| 3 | JSON schema validity rate *(of the envelope)* | ≥ 99.5% | 1.000 | **1.000** (177/177 valid, 0 driver-side violations) | **PASS** |
| 4 | Silent identifier errors | **0** | 0 | **0** — 12 fields failed a validator, all 12 carry `VALIDATOR_FAILED` | **PASS** |
| 5 | Bounce rate (quality gate) | 5-15% | 16.9% | **16.9%** (30/177) on a corpus engineered to a 16.9% bounce profile | **PASS on intent, N/A on the band** — §8 |
| 6 | Escalation rate | ≤ 15% | 13.6% | **13.6%** (24/177 router escalations) | **PASS** |
| 7 | Router false-negative rate | ≤ 8% | not measured | 4 shadow-sampled; Tier 2 unavailable (Q15), 0 diffs completed | **NOT MEASURED — blocked on Q15** |
| 8 | p95 end-to-end, fast path | ≤ 90 s | 3.90 s | **22.61 s** p95 (1-page documents, SLM live) | **PASS**, §7 |
| 9 | Throughput per 16-vCPU node | ≥ 60 pages/min | — | not this machine | **OUT OF SCOPE** (adjudication A-10) |

**Target 1 did not move, and that is a result rather than an absence of one.** The SLM went from
answering 14 documents to answering 147, and **not one deterministic field changed** — same TP,
same FP, same FN, on every one of the eight identifier fields. That is invariant 7 ("deterministic
fields never pass through the SLM") holding under the only condition that could ever have
falsified it. The previous run could not make that statement, because the SLM was barely running.

Metrics the BUILD-BRIEF asks for that carry no numeric PRD target:

| Metric | Pre | **Now** |
|---|---|---|
| CER / WER overall | 0.483 / 0.628 | **0.483 / 0.628** — identical |
| CER by script | Latn 0.474, Deva 0.518 | **identical** |
| Detection mean IoU / line recall | 0.481 / 0.392 | **identical** |
| Line-level LID accuracy | 0.860 matched | **identical** |
| Quality-gate expectation-satisfied | 0.915 | **identical** |
| Translations attempted / discarded | 5 / 0 | **29 / 0** — §5.3 |
| SLM tokens settled | 20 734 | **286 198** |

---

## 2. The control: what did *not* move

Before reading any delta, here is the evidence that this run isolates one variable.

| quantity | previous run | this run | identical? |
|---|---|---|---|
| Overall CER | 0.4830216502446545 | 0.4830216502446545 | **yes, to 16 digits** |
| Overall WER | 0.6284168299281649 | 0.6284168299281649 | **yes** |
| `script:Deva` CER / WER | 0.5179002282360613 / 0.7324364723467862 | same | **yes** |
| `script:Latn` CER / WER | 0.4739959500506244 / 0.5938082804923536 | same | **yes** |
| Scored lines / reference chars | 4 527 / 74 595 | 4 527 / 74 595 | **yes** |
| Detection matched / spurious / mean IoU | 3 554 / 248 / 0.481 | same | **yes** |
| LID matched / correct / accuracy | 3 554 / 3 055 / 0.8595948227349466 | same | **yes** |
| Quality-gate confusion, all nine cells | 42/15/0 · 15/75/0 · 0/0/30 | same | **yes** |
| Escalation TP / FP / FN | 23 / 1 / 109 | same | **yes** |
| Deterministic field TP / FP / FN | 481 / 109 / 296 | same | **yes** |

Only three families of number moved: **SLM-sourced fields**, **translation counts** (which
depend on the SLM producing native-script values to translate), and **latency** (which depends
on the model actually answering). Nothing else could have.

A second, independent confirmation fell out of the provider cache: **14 lookups hit**, and the
cache key is `sha256(model + prompt + params)` where the prompt embeds the OCR text. A hit is
proof that document's transcription is byte-identical to the previous run's.

---

## 3. The SLM half, measured

### 3.1 What the model was asked, and what came back

| outcome | documents | share of eligible |
|---|---|---|
| bounced — SLM never invoked (correctly) | 30 | — |
| **SLM-eligible documents** | **147** | 100% |
| router refused (`402`, `PROVIDER_ERROR`) | **0** | **0.0%** |
| answered, failed the contract twice (initial + repair) | **93** | **63.3%** |
| **answered and validated** | **54** | **36.7%** |

Of the 54 validated answers: **47** produced a complaint summary (198 spans total) and **21**
produced at least one SLM-sourced *value* field (40 field instances). Model path per document,
from the manifest: **135 live**, **12 cached**, 30 no model call at all.

The fallback model was configured and **never engaged** — correctly. A schema-invalid answer is
not an outage code, and swapping models on it would silently change what the sweep is measuring.
One model, one request per document, one repair retry where the first answer failed.

### 3.2 SLM field scores

| field | pre TP/FP/FN | **now TP/FP/FN** | pre F1 | **now F1** |
|---|---|---|---|---|
| `prior_complaint_ref` | 4/2/68 | **13/5/59** | 0.103 | **0.289** |
| `complainant_address` | 1/0/53 | **6/3/48** | 0.036 | **0.190** |
| `complainant_name` | 1/0/107 | **7/2/101** | 0.018 | **0.120** |
| `relief_sought` | 0/1/36 | **1/3/35** | n/a | **0.050** |
| `complaint_summary` | 0/0/90 | 0/0/90 | n/a | n/a — **§6.1** |
| `complainant_contact` | 0/0/54 | 0/0/54 | n/a | n/a — **§6.2** |
| **bucket total** | **6/3/408** | **27/13/387** | **0.028** | **0.119** |

Precision is essentially flat (0.667 → 0.675) while recall quadruples (0.014 → 0.065). The model
is not becoming less careful as it answers more; there is simply far more of it.

**The honest denominator question.** Two of the six fields in this bucket cannot score above zero
for reasons that have nothing to do with the model (§6.1, §6.2). Excluding them:

| bucket | TP | FP | FN | P | R | **F1** |
|---|---|---|---|---|---|---|
| all six fields (as the PRD defines the target) | 27 | 13 | 387 | 0.675 | 0.065 | **0.119** |
| four fields the pipeline actually attempts | 27 | 13 | 243 | 0.675 | 0.100 | **0.174** |

**Both are published; the first is the verdict.** The second is what the extraction path is
worth today once the measurement chain's own gaps are set aside, and it is still 0.63 short of
the target. Quoting only the second would be scoring against a denominator chosen after seeing
the numerator.

### 3.3 Per document type — where the movement is, and where it is not

The scorer has no per-type rollup, so each row is `tools/eval_harness.py` re-run with
`--types <t>` over both prediction sets against the same corpus.

| type | docs | GT | pre TP/FP/FN | **now TP/FP/FN** | pre F1 | **now F1** | pre SLM-F1 | **now SLM-F1** | CER |
|---|---|---|---|---|---|---|---|---|---|
| `complaint_form_deva` | 18 | 198 | 62/1/136 | **78/8/120** | 0.475 | **0.549** | 0.065 | **0.328** | 0.364 |
| `complaint_form_en` | 18 | 198 | 92/1/106 | **96/1/102** | 0.632 | **0.651** | 0.065 | **0.144** | 0.488 |
| `complaint_letter_mar` | 18 | 108 | 14/29/94 | **15/30/93** | 0.185 | **0.196** | 0.000 | **0.034** | 0.503 |
| `complaint_letter_en` | 18 | 108 | 27/35/81 | 27/36/81 | 0.318 | 0.316 | 0.000 | 0.000 | 0.650 |
| `bilingual_form` | 18 | 198 | 52/19/146 | 52/20/146 | 0.387 | 0.385 | 0.000 | 0.000 | 0.509 |
| `receipt` | 18 | 90 | 67/5/23 | 67/5/23 | 0.827 | 0.827 | — | — | 0.375 |
| `photo_evidence` | 15 | 75 | 53/2/22 | 53/2/22 | 0.815 | 0.815 | — | — | 0.461 |
| `bank_statement` | 18 | 54 | 42/17/12 | 42/17/12 | 0.743 | 0.743 | — | — | 0.418 |
| `cheque` | 18 | 126 | 63/3/63 | 63/3/63 | 0.656 | 0.656 | — | — | 0.378 |
| `email_printout` | 18 | 36 | 15/0/21 | 15/0/21 | 0.588 | 0.588 | — | — | 0.635 |

**Only the types that carry free-text fields moved**, which is the shape the design predicts.
Five types are unchanged to the integer because they have no SLM-owned reference fields at all.

**The most interesting row is `complaint_form_deva`.** Its SLM sub-F1 (0.328) is **more than
double** the English form's (0.144), on pages whose recognition is worse. The cause is §4.1: the
model's answers *validate* far more often on Devanagari forms (0.800) than on English ones
(0.333). Whatever is breaking the contract is not the script.

`complaint_letter_en` and `bilingual_form` each picked up exactly **one extra false positive**
and no true positive — the only two cells in the whole report that moved the wrong way.

---

## 4. Q10, answered: schema validity is the ceiling

### 4.1 The measured rate, and how far it varies

**54 of 147 = 0.367**, against the bake-off's 0.5625 from n = 16. The small-sample estimate was
optimistic by 0.20. Per document type:

| type | eligible | validated | **rate** | median lines | median tokens |
|---|---|---|---|---|---|
| `receipt` | 15 | 14 | **0.933** | 15 | 680 |
| `photo_evidence` | 12 | 11 | **0.917** | 15 | 684 |
| `complaint_form_deva` | 15 | 12 | **0.800** | 25 | 1 934 |
| `complaint_form_en` | 15 | 5 | 0.333 | 25 | 2 047 |
| `bilingual_form` | 15 | 3 | 0.200 | 24 | 2 287 |
| `cheque` | 15 | 3 | 0.200 | 13 | 1 744 |
| `complaint_letter_mar` | 15 | 3 | 0.200 | 14 | 2 008 |
| `bank_statement` | 15 | 1 | **0.067** | 99 | 5 373 |
| `complaint_letter_en` | 15 | 1 | **0.067** | 13 | 1 828 |
| `email_printout` | 15 | 1 | **0.067** | 13 | 1 645 |

A 14-fold spread across document types on one model with one prompt.

**Length is part of it but nowhere near all of it.** Grouping every eligible document by its
recognised line count:

| lines | n | validity |
|---|---|---|
| 0-20 | 87 | 0.379 |
| 20-40 | 45 | 0.444 |
| 60-100 | 12 | **0.083** |
| 100+ | 3 | **0.000** |

Documents past 60 lines almost never validate — `bank_statement` at 99 median lines is the whole
of that band. But the median line count of a *valid* answer is 15 and of a *failed* one is 14, so
length explains the tail and not the body. Three types at 13-14 median lines score 0.067, 0.067
and 0.200 while `receipt` at 15 lines scores 0.933. **This report states the association and does
not claim the mechanism** — identifying it needs the rejected payloads, which the driver does not
retain.

### 4.2 Why this is the ceiling

A document whose answer fails the contract contributes **zero** SLM fields. So:

> recall ≤ 0.367 × (whatever the surviving answers achieve)

Even a model that answered *perfectly* whenever it answered at all would cap SLM field recall at
0.367 with this validity rate. The target is 0.80. **The configured model cannot reach it**, and
that conclusion needs no assumption about answer quality.

For contrast, the bake-off measured the fallback `meta-llama/Llama-3.1-8B-Instruct` at schema
validity **1.000** over n = 18, trading grounding (0.556 vs 0.90) and summary F1 (0.408 vs 0.667).
On this evidence that trade now looks clearly correct — and §4.3 shows the grounding it trades
away is largely insurance the pipeline provides anyway.

**Recommendation for the owner (Q10):** swap `SLM_MODEL` to the fallback and re-run. The sweep
costs ~23 minutes and the response cache makes the OCR half free. This report does not make that
change: `app/settings.py` and `.env` are outside this lane, and changing the model mid-lane would
mean this document reported two configurations at once.

### 4.3 The grounding gate rejected almost nothing — measured

64 grounding rejections sounds alarming until they are decomposed:

| rejection reason | count | meaning |
|---|---|---|
| `empty` | **61** | The model returned an empty value for that field — it declined to answer |
| `ungrounded` | **3** | The value is not on the page within edit distance 2 |

By field: `relief_sought` 37, `prior_complaint_reference` 27.

So of **104 values proposed** (40 published + 64 rejected), the gate discarded **3 as
unsupported — 2.9%**. Invariant 8 is cheap here, and the recall gap is the model declining to
answer, not the gate being harsh. Raising `GROUNDING_MAX_EDIT_DISTANCE` would recover at most
three values and would weaken the guarantee that makes the output defensible; **do not tune it in
response to this report.**

---

## 5. SLM reachability and translation

### 5.1 Q13 — resolved

| | previous run | **this run** |
|---|---|---|
| 19-token probe, `gemma-3-4b-it` | 200 OK | **200 OK** |
| 19-token probe, `Llama-3.1-8B-Instruct` | 200 OK | **200 OK** |
| **production-sized probe** (2 032 prompt tokens, `max_tokens=512`, `json_object`) | not performed | **200 OK**, 133 completion tokens |
| `402 Payment Required` during the sweep | **129 of 147** | **0 of 147** |
| tokens settled | 20 734 | **286 198** |

The previous report's diagnosis — that refusals were per-request rather than account-level —
cannot be re-tested now that there are no refusals to characterise. What is established is that
**with credits on the account, every production-sized request succeeded**, which is the outcome
that matters.

The production-sized probe is a method change worth keeping: a 19-token probe returning 200
while 129 of 147 real requests returned 402 is exactly what happened last time, and a
trivially-sized health check would have declared the endpoint healthy again.

### 5.2 Cost, for planning

| | |
|---|---|
| Tokens settled, 147 documents | **286 198** |
| Documents that spent tokens | 133 |
| Median tokens per document | **1 914** |
| Maximum | 5 495 (`bank_statement`) |
| Mean per eligible document | ≈ 1 947 |

**`JOB_TOKEN_BUDGET` is 8 000 and a single document's worst case is 5 495**, so one document fits.
The per-instance-budget trap recorded as Q9 is therefore live and material at this token scale: a
batch caller that does not call `reset_budget()` between documents exhausts 8 000 after **two**
documents at the median, not three. `tools/run_corpus_eval.py` resets per document.

### 5.3 Q7 — sentinel survival, n = 5 → n = 29

| | previous run | **now** |
|---|---|---|
| Extracted fields | 599 | **630** |
| Fields whose `value` is Devanagari | 29 | **46** |
| Translations attempted | 5 | **29** |
| Returned a `value_en` | 5 | **29** |
| Discarded, flagged `TRANSLATION_UNAVAILABLE` | 0 | **0** |
| **Sentinel-survival rate** | 5/5 (n = 5) | **29/29 = 1.000 (n = 29)** |

`mt_engine` is `indictrans2-indic-en-dist-200M-ct2-int8` on all 29 — the real weights. Examples:

| field | `value` | `value_en` |
|---|---|---|
| `complainant_name` | राजेश कुमार शर्मा | Rajesh Kumar Sharma |
| `complainant_address` | पनवेल महाराष्ट् | Panvel Maharashtra |
| `complainant_address` | पवेल, मराद | Pavella : Marad |

**n = 29 is six times the previous evidence and still not a rate.** It is consistent with WS-F's
corpus-grounded 54/54 (`docs/decisions-wsf.md` D-F-15), which remains the best evidence on the
record. Q7 stays **unblocked and unanswered**; the honest way to close it is the `--final` run,
where the locked split roughly doubles the sample again.

**Two observations for whoever is next in `s6_translate.py`,** neither a defect claim:

1. The **ASCII-value observation from the previous run reproduces**: `prior_complaint_reference`
   `CMS/2025/850859` → `CMS / 2025 / 850859`. The immutable `value` is untouched and `value_en` is
   advisory, so nothing downstream is wrong, but the eligibility test is handing identifiers to
   an MT engine.
2. A comma is being rendered as a colon: `पवेल, मराद` → `Pavella : Marad`, and
   `पनवेल, महारा्` → `Panvel : Maharashtra`. Cosmetic, in an advisory field, on already-noisy OCR.

The **bilingual `value`-slot contract still holds exactly**: of 29 claimed bilingual slots, 13
carry the page's own string and **0 carry a translation**. Zero contract violations, on more than
twice the previous sample.

| | pre | **now** |
|---|---|---|
| GT bilingual field occurrences | 198 | 198 |
| claimed | 13 | **29** |
| `value` = the page's own string | 6 | **13** |
| `value` = its translation | **0** | **0** |
| `value_en` correct / wrong / absent | 2 / 1 / 10 | **9 / 10 / 10** |

`value_en` wrong rose 1 → 10 alongside correct 2 → 9. With ten wrong out of nineteen judged, the
advisory English rendering is **close to a coin flip on this corpus** and should not be presented
to a reviewer as authoritative. It is a genuinely new finding that only became visible at this
sample size.

---

## 6. Three things this run exposed

### 6.1 Q16 (drafted) — `complaint_summary` is produced but never scored

The pipeline produced a summary for **47 documents, 198 spans total**. The harness scores
`complaint_summary` **0 TP / 0 FP / 90 FN**.

Cause: `tools/run_corpus_eval.py::prediction_from_result` publishes the summary as
`summary_span_count` — an integer — and never as an entry in the prediction's `fields` map. The
corpus reference *does* carry `complaint_summary` (90 dev-split instances), so the harness scores
it as a field nothing ever claims. **90 guaranteed misses**, and they sit inside a published PRD
target.

`tools/eval_harness.py`'s own scope note says it does not score "summary-span quality", which is
a defensible boundary — span-level quality does need the envelope's contract tests. But the
*field* is in the reference and in the target's denominator, so the current arrangement is the
worst of both: counted against the score, impossible to earn.

Two ways out, and they are not equivalent: emit the summary as a scorable field and let the
harness match it like any other, **or** remove `complaint_summary` from the reference bucket and
say in the target's definition that it is measured elsewhere. Either is fine; the present state
is not. **Lane: WS-H (driver) / WS-D (harness).** Not fixed — this lane's write scope is the
driver's *invocation*, not its source.

### 6.2 Q17 (drafted) — `complainant_contact` has no extractor

`complainant_contact` has **54 reference instances** and is scored as an SLM field. It appears in
**neither**:

- `SLM_ELIGIBLE_FIELDS`, which is `{complainant_address, complainant_name, complaint_summary, prior_complaint_reference, relief_sought}` — so the model is never asked for it; nor
- the validator table, so no deterministic mechanism produces it either.

Result: **54 guaranteed misses in a PRD target's denominator, with no code path that could ever
reduce them.** This is not a model failure and no amount of SLM tuning will move it.

It is a genuine product question, not just bookkeeping: a phone number or email address is
arguably a *deterministic* field (it has structure a regex can validate) rather than a free-text
one, and it would be better served by a validator than by the SLM. **Lane: WS-E3 (validators) or
WS-B (contracts), plus an owner call on which side of the deterministic/free-text line it belongs
on.** Not fixed.

Together, Q16 and Q17 are **144 of the SLM bucket's 414 misses — 35%**.

### 6.3 Q15 — unchanged, and now definitively not about credits

| | previous run | **this run** |
|---|---|---|
| Tier-2 attempts | 28 | **28** |
| `tier2_applied` | 0 | **0** |
| Failure | HTTP 400 `model_not_supported` | **unchanged** |

The previous report established this was a 400 and not the 402. **This run removes the last
doubt**: the account had working credits, the SLM leg on the very same router succeeded 147 times
out of 147, and the Tier-2 leg still failed 28 times out of 28. `PaddlePaddle/PaddleOCR-VL` is
not served to this account, full stop.

Consequence: **PRD target 7 remains unmeasurable.** The shadow sampler drew 4 documents and every
diff is empty for want of a Tier-2 answer. This reopens **Q2** (the PRD leaves the Tier-2 model
unnamed) and is an owner decision on the model.

### 6.4 Q14 — still open, unchanged

A line's published `script` is still the recognition head's script rather than the line's own.
Nothing in this run touched it and every LID number is identical to the previous run's. It
remains: `complaint_form_deva`'s Latin-print LID cell at 0.068, 0 of 147 pages reporting mixed
scripts against 30 that genuinely mix, `is_code_mixed` unreachable. See the previous report §7.1.

---

## 7. Latency — the first honest end-to-end numbers

| stage | pre p50 | **now p50** | pre p95 | **now p95** |
|---|---|---|---|---|
| **end-to-end** | 1 429 | **5 861** | 3 895 | **22 609** |
| `s5_slm` | 227 | **5 317** | 2 506 | **21 845** |
| `s4_ocr` | 760 | 751 | 1 678 | 1 727 |
| `s4b_vlm` | 1 414 | 1 517 | 1 698 | 1 777 |
| `s15_quality` | 195 | 191 | 220 | 252 |
| `s3_classify` | 161 | 174 | 306 | 342 |
| `assemble` | 0.26 | 0.24 | 0.50 | **61.4** |
| `s5_validators` | 5.5 | 5.5 | 15.7 | 15.5 |
| `s2b_lid` | 1.4 | 1.2 | 5.4 | 5.1 |

**Read the end-to-end rise as the system finally doing the work**, not as a regression. The
previous run's p95 of 3.9 s was dominated by how quickly the router said no. Every local stage is
unchanged within noise — `s4_ocr` p50 moved 760 → 751 ms, i.e. 1% the *helpful* direction, which
is measurement noise on a machine also doing network I/O.

`assemble` p95 0.5 → 61.4 ms is the summary being built for the first time on 47 documents. It is
0.3% of end-to-end and not worth attention.

Per-document wall clock, split by whether a model call actually went out (from the manifest):

| model path | n | p50 | p95 | max |
|---|---|---|---|---|
| **live** | 135 | 7 332 ms | 24 480 ms | 59 125 ms |
| **cached** | 12 | 1 067 ms | 6 701 ms | 12 623 ms |
| **none** (bounced) | 30 | 146 ms | 183 ms | 263 ms |

That table is the cost of the remote SLM stated plainly: **a live model call is ~6× the wall
clock of a cached one and ~50× a bounced page.** A bounced page costs 146 ms — the quality gate
earns its place on throughput alone, before any accuracy argument.

**PRD target 8 still passes**, with the same caveats as before plus one new one: these are
**one-page** documents against a 5-page target, and a 5-page document making five *real* SLM calls
on this laptop would land in the 30-110 s range — which straddles the 90 s target. **The next run
that matters for target 8 is a multi-page one**, and this corpus cannot provide it.

The sweep is strictly sequential by design, so nothing was measured under self-inflicted
contention.

---

## 8. Quality gate, escalation, detection — unchanged

All three are byte-identical to the previous run and are reproduced here only so this document
stands alone.

```
expectation  accept  repair  bounce
clean            42      15       0
degraded         15      75       0
bounce            0       0      30
```

Expectation-satisfied **0.915**. No clean page bounced; all 30 bounce-profile pages bounced. The
16.9% bounce rate reproduces the corpus's designed 16.9%.

**The caveat is unchanged and still the sharpest one in this report:** the Q3 thresholds were
calibrated against this corpus using variants 0-2 — the split this report scores. It is the most
optimistic number here, and the locked split is what will test it.

Escalation: 24 documents (**13.6%**), precision **0.958**, recall 0.174 against the
`quality_or_hand` stand-in reference. Recall against a stand-in with no corpus label behind it
(adjudication A-3) is not a target and should not be read as one. `tier2_applied` remains 0 (§6.3).

Detection by profile: clean 0.767 / 0.955, light 0.543 / 0.576, scan 0.450 / 0.448, heavy
0.356 / 0.285, photo 0.266 / 0.123, bounce n/a / 0.000. Both known measurement effects still
apply: axis-aligned IoU on rotated ground truth (WS-D §6), and the post-repair coordinate frame
raised as **Q11**, still open and still undiagnosed.

---

## 9. Remaining MISSes, with causes

| what misses | measured | cause | owner |
|---|---|---|---|
| SLM field F1 0.119 vs ≥ 0.80 | P 0.675 / R 0.065 | **Schema validity 0.367 caps recall** before answer quality is even considered | **Q10 — owner: swap the model and re-run** |
| … of which 144 of 414 misses | `complaint_summary` 90, `complainant_contact` 54 | Measurement-chain gaps, not model failures | Q16, Q17 (§6) |
| Deterministic field F1 0.704 vs ≥ 0.97 | P 0.815 / R 0.619 | **Recall, and recall is recognition.** Precision is healthy on every field; the gap is instances never read. Concentrated in degraded profiles and in `bilingual_form` | WS-E2, WS-E1 |
| `bilingual_form` field F1 0.385, CER 0.509 | 368/368 lines read as `Latn`; 60 Deva GT lines unread | **One head per page cannot read a half-and-half page.** Largest single block of residual Devanagari error | WS-E2 |
| `complaint_letter_en` F1 0.316, 36 FP | CER 0.650 | Worst Latin CER on the corpus; dense prose, and the deterministic layer over-claims on it | WS-E3 |
| Router false-negative rate | 0 completed shadow diffs | **Q15** — Tier 2 never lands | WS-C + owner |
| `is_code_mixed` never true | 0/147 vs 30/147 | **Q14** | WS-C / WS-E2 |
| `value_en` wrong on 10 of 19 judged | bilingual slot table | Advisory MT on noisy Devanagari OCR | WS-E3 / Q7 |

---

## 10. What to fix, in priority order

1. **Swap `SLM_MODEL` to `meta-llama/Llama-3.1-8B-Instruct` and re-run** (owner, Q10). This run
   turns the bake-off's schema-validity argument from a small-n suspicion into a measured
   ceiling: 0.367 caps SLM recall at less than half the target regardless of answer quality. The
   re-run costs ~23 minutes and the OCR half is free from cache.
2. **Close the two measurement-chain gaps before the `--final` run** (Q16, Q17). They are 35% of
   the SLM bucket's misses and neither is a model problem. Publishing a final SLM number without
   settling them means publishing a target scored against a denominator a third of which nothing
   can reach.
3. **Name a Tier-2 model the router will serve** (Q15, WS-C + owner). Now proven independent of
   credits. PRD target 7 cannot be measured until it lands.
4. **Derive a line's script from its own characters** (Q14, WS-C / WS-E2). Small change, restores
   `is_code_mixed`.
5. **Teach recognition to mix heads within a page** (WS-E2). What `bilingual_form` needs, and the
   largest remaining block of Devanagari error.
6. **Measure a multi-page document against PRD target 8** (WS-D). One-page p95 of 22.6 s against
   a 5-page 90 s target no longer has comfortable headroom with the SLM live.
7. **Decide whether evidence boxes must map back to the pre-repair frame** (Q11) — unchanged,
   still open.

---

## 11. Caveats — every number above inherits these

1. **Synthetic handwriting is not handwriting.** Every `hand` cell is font-rendered glyph jitter.
   Those cells validate plumbing, routing, batching and throughput; they do **not** validate
   handwriting accuracy. `Deva|hand|clean` CER 0.249 is not evidence the service reads handwritten
   Devanagari.
2. **The locked split is held out.** Variants 3 and 4 — 118 documents — were neither run nor
   scored. Everything here is dev-split and optimistic to exactly the degree the build was tuned
   against variants 0-2: the quality gate was, and so were the recognition probe's geometry and
   its 0.05 score margin.
3. **This is one model, one prompt, one day.** Schema validity 0.367 characterises
   `google/gemma-3-4b-it` on this prompt against this router on 2026-08-08. The router
   load-balances across backend providers, and Q5 already measured `json_schema` support varying
   per request. A re-run could differ; the 14× spread across document types (§4.1) is the more
   robust finding.
4. **Q6's two owner calls remain open** (`docs/decisions-q6.md` §10): whether to unify the two
   field vocabularies for real, and whether to promote the bilingual `value`-slot check into
   precision. This run measured **0 contract violations** in that slot over 29 claims, so
   promoting it still costs nothing.
5. **TLS interception.** As with the previous run, Python could not complete a TLS handshake to
   the router on this machine: Norton's inspection CA is trusted by the OS and by Node but is not
   in `certifi`'s bundle. This run set `SSL_CERT_FILE` to a scratch bundle concatenating
   `certifi` with the machine's already-installed Norton root, **for the run's process only**. No
   system setting, no certificate store and no repository file was modified, and **verification
   was never disabled**. Anyone reproducing this on a machine with TLS inspection will need the
   same — see `docs/TUNING.md` §5.9.

---

## 12. The harness output, verbatim

Everything below is `tools/eval_harness.py` 1.1.0's own output, unedited, copied from
[`report.txt`](report.txt). Stage rows ending `:p1` are the per-page split of the row above; on a
single-page corpus they are identical by construction.

```text
==============================================================================
RBI OCR corpus evaluation — harness 1.1.0
==============================================================================
corpus       E:\pythonprojects\python_ocr\corpus  (seed 1729, generator 1.2.0)
predictions  E:\pythonprojects\python_ocr\docs\eval\predictions
split        dev  final=False  locked variants [3, 4]
documents    177 scored / 177 selected / 118 withheld

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
complainant_address      6   3   48  0.667  0.111  0.190
complainant_contact      0   0   54    n/a  0.000    n/a
complainant_name         7   2  101  0.778  0.065  0.120
complaint_summary        0   0   90    n/a  0.000    n/a
disputed_amount         67  19   74  0.779  0.475  0.590
ifsc_code               62   7   28  0.899  0.689  0.780
micr_code               13   2    5  0.867  0.722  0.788
prior_complaint_ref     13   5   59  0.722  0.181  0.289
relief_sought            1   3   35  0.250  0.028  0.050
transaction_date       101  43   22  0.701  0.821  0.757
transaction_reference   65   7   58  0.903  0.528  0.667

Fields — by source mechanism (R = share of ALL reference fields this source captured)
---------------------------------------------------------
source            TP  FP    GT      P      R     F1
---------------------------------------------------
lookup_table      81  13  1191  0.862  0.068  0.126
regex_validator  400  96  1191  0.806  0.336  0.474
slm               27  13  1191  0.675  0.023  0.044

Fields — bilingual slots (fields whose native spelling differs from its English)
---------------------------------
metric                     count
--------------------------------
gt occurrences               198
claimed                       29
value = page's own string     13
value = its translation        0
value_en correct               9
value_en wrong                10
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
-------------------------------------------
stage             n       p50        p95
----------------------------------------
assemble         28     0.242     61.401
e2e_total       177  5861.198  22608.611
s0_read         177     0.996      1.380
s15_quality     177   190.668    251.774
s15_quality:p1  177   190.668    251.774
s1_ingest       177    21.784     51.993
s2b_lid         147     1.242      5.091
s2b_lid:p1      147     1.242      5.091
s3_classify     147   173.781    341.939
s3_classify:p1  147   173.781    341.939
s4_ocr          147   751.036   1727.103
s4_ocr:p1       147   751.036   1727.103
s4b_vlm          28  1517.072   1777.081
s4b_vlm:p1       28  1517.072   1777.081
s5_slm          147  5316.891  21845.131
s5_validators   147     5.530     15.510

[... `Recognition — CER / WER by script x modality x profile`, `Fields — field x source`
 and `Latency (ms) by stage x profile` omitted for length; all three are in report.txt ...]

CAVEAT  Font-rendered handwriting is not real handwriting. This corpus validates plumbing, schema, routing and throughput. It does NOT validate handwriting accuracy — measure that on iiit-indic-hw-words and real complaint pages.
NOTE    variants [3, 4] are the locked test split and were not scored. Pass --split all --final to include them.
```

---

## 13. Run provenance

| | |
|---|---|
| Documents | 177 selected, 177 written, **0 failed**, 0 timeouts, 0 overlap events |
| Wall clock | **1 370 s** (22 min 50 s), strictly sequential |
| Schema self-check | **0 violations** on 177 written predictions |
| Provider cache | 58 rows before, 99 after; **14 hits / 161 misses** (hit rate 0.080), 40 stores |
| SLM tokens settled | **286 198** |
| SLM model | `google/gemma-3-4b-it`; fallback `meta-llama/Llama-3.1-8B-Instruct` configured, **not engaged** |
| Tier-2 VLM | `PaddlePaddle/PaddleOCR-VL` — **not served** (§6.3) |
| Local weights | `models/ocr/{det,rec_latn,rec_deva}.onnx`, `models/lid/lid.176.ftz`, `models/mt/indic-en` (CT2 INT8) |
| Page classifier | CNN artifact absent; fitted structural path ran for all 147 non-bounced pages |
| Git SHA | `bcd85dd950dd08cbfbf91d0131b440bd7eba37ba` |

**Never recorded anywhere:** the HF token. The manifest carries `hf_token_present: true` and not
its value or its length.

---

## 14. Method, and how to disagree with it

- **The scorer imports nothing from `app/`** (adjudication A-2) and the predictions are committed.
  Anyone can re-score them, change the escalation reference or the IoU threshold, or write a
  different scorer, with no model directory and no token.
- **The control in §2 is what makes this run readable.** One variable changed. A report that
  could not demonstrate the other variables held constant would be asserting attribution rather
  than showing it.
- **Schema validity is reported with its denominator everywhere** (54/147, and per type with each
  type's own n = 15 or 12). A rate without an n is how 0.5625 from sixteen observations became a
  planning assumption.
- **`escalated` is the router's decision**, not the envelope's `escalation_flag`, which ORs in
  "any field needs review".
- **A failed document is scored, not skipped** — written with `pages: []` and `fields: {}` so its
  lines count as deletions. There were none on this run.
- **Recall by source decomposes the total**: each mechanism is scored against the whole ground
  truth, so `lookup_table` 0.068 + `regex_validator` 0.336 + `slm` 0.023 sums to overall recall.
- **Latency is single-threaded and sequential by design.** Parallelising would have cut the wall
  clock by more than half — every document now waits on a real network round trip — and
  invalidated PRD target 8.

Decisions and reasoning for this run: [`docs/decisions-wsi-a.md`](../decisions-wsi-a.md).
Prior runs: [`docs/decisions-wsh3.md`](../decisions-wsh3.md),
[`docs/decisions-wsh2.md`](../decisions-wsh2.md). The fixes those measured:
[`docs/decisions-q12.md`](../decisions-q12.md), [`docs/decisions-q6.md`](../decisions-q6.md).
