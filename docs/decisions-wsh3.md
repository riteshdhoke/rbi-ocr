# WS-H3 decisions — the post-Q12 / Q6 re-run of the M7 exit gate

Judgment calls made while re-running `tools/run_corpus_eval.py` over the dev split after the
Q12 (Devanagari recognition routing) and Q6 (corpus field reference) fixes landed, and while
rewriting `docs/eval/**`.

Lane: `docs/eval/**`, this file, and the **Q7 / Q12 / Q13 rows only** of `OPEN-QUESTIONS.md`.
Nothing outside it was touched — no source file, no test, no corpus file, no other
OPEN-QUESTIONS row. Two new defects surfaced and were **documented rather than fixed** (§4),
which is the rule this lane operates under.

Like WS-H2: this lane measures, it does not fix. Every number in `docs/eval/REPORT.md` came
out of the working tree as it stood, with no tuning between the fixes and the run.

---

## 1. Running the sweep

### D-H3-1 · Three scorings, not two — the before/after is decomposed, not aggregated

Two independent changes landed since the last gate: Q6 changed the **reference** and Q12
changed the **pipeline**. A two-column before/after would have credited one for the other's
work, and the direction of the error is not guessable — Q6 alone moves deterministic precision
0.318 → 0.819 without a single pipeline change.

So the report publishes three columns:

| col | predictions | corpus + harness |
|---|---|---|
| A | previous run's | generator 1.1.0, harness 1.0.0 — the published baseline |
| B | **previous run's, unchanged** | generator 1.2.0, harness 1.1.0 |
| C | this run's | generator 1.2.0, harness 1.1.0 |

**A→B is the reference fix; B→C is the pipeline fix.** Column B was produced by re-scoring a
snapshot of the previous run's committed predictions, taken before the predictions directory
was cleared. It independently reproduces `docs/decisions-q6.md` §8, which is a useful
cross-check on that lane's own arithmetic.

Column A is quoted from the previous report's own published figures rather than recomputed,
and the field buckets were reconciled against it: the deterministic aggregate comes out at
TP 156 / FP 335 / FN 414, P 0.318, R 0.274, F1 0.294 — exactly what
`REPORT-2026-08-07-preQ12.md` §3 printed. A before/after whose "before" column does not
reproduce the document it claims to supersede is not a comparison.

### D-H3-2 · The previous report is kept as a file, not only in git history

`docs/eval/REPORT-2026-08-07-preQ12.md` is a byte copy of what `REPORT.md` said before this
run. Git has it too, but a reader checking a "0.904 → 0.518" claim should not have to know the
sha, and the two files sitting side by side is what makes the claim falsifiable at a glance.

### D-H3-3 · The predictions directory was cleared, not resumed

`--resume` and a partial overwrite would both have left a directory mixing pre-Q12 and post-Q12
predictions with no marker distinguishing them, and every aggregate would have been a silent
blend of two pipelines. All 177 files were deleted and all 177 rewritten. `run.documents_ok`
is 177 and `documents_failed` is 0, so nothing was inferred from a stale file.

The 30 `cheque` predictions had to be regenerated regardless: Q6 changed those images
(`docs/decisions-q6.md` §6), so the committed cheque predictions described pages that no longer
exist. That is why column B's cheque numbers are labelled a floor in the report rather than
quoted as a result.

### D-H3-4 · One probe per model before the sweep, and then the sweep runs live

The brief capped SLM probing at one per model per run to avoid hammering a dead endpoint. That
cap was honoured **before** the sweep: one 19-token request to `google/gemma-3-4b-it` and one
to `meta-llama/Llama-3.1-8B-Instruct`. Both returned 200, which is why the sweep then ran with
the SLM live rather than with `--no-slm`.

The alternative — `--no-slm` for the whole sweep — was rejected for two reasons. It refuses
*every* remote method, so the cached SLM answers from the previous run (which the brief
explicitly called free) would not have been used; and it would have made the latency table
incomparable with the previous run by removing the network leg entirely.

Running live did not become a retry storm, and this is a property of the code rather than of
luck: a 402 maps to `ProviderError`, `hf_api._is_retryable` returns False for it, and
`PROVIDER_ERROR` is not in the driver's `_SLM_OUTAGE_CODES`, so the fallback never engaged
either. **One request per document, one model, no retries, no thrash.** 129 refusals cost 129
round trips and nothing more.

### D-H3-5 · The 402 is reported as a per-request refusal, not an account outage

The previous report concluded credits were depleted account-wide. That conclusion does not
survive this run's evidence and the report says so: a 19-token request succeeds on both models,
while 129 of 147 production-sized requests are refused, and the 18 successes are **interleaved
throughout the run** rather than front-loaded. "Credits ran out at document 3" would predict a
prefix of successes; the data shows no such prefix.

The report states the mechanism as *consistent with* router-level provider selection (the same
per-request variability Q5 measured for `json_schema` support) rather than asserting it. This
lane cannot see the router's routing table, and a measurement report should not invent one.

### D-H3-6 · SLM field F1 stays NOT MEASURED at 14 of 147 documents

The harness produces a number (P 0.667, R 0.014, F1 0.028) and the report prints it, labelled,
while marking PRD target 2 **NOT MEASURED — blocked on Q13**. Fourteen documents is not a
sample of 147, and a precision resting on nine claims is not a precision. Printing the figure
but refusing the verdict is deliberate: suppressing it would invite someone to recompute it and
assume it had been hidden.

Same reasoning keeps **Q10 blocked**. Of the answers that did arrive, 14 of 18 validated —
*better* than the bake-off's 0.5625 for this model — and updating on that would be updating on
n = 18 selected by whichever backend happened to serve the request.

---

## 2. Reading the results

### D-H3-7 · The Latin no-regression claim is made against column B, not column A

Q12's own A/B (`docs/decisions-q12.md` §2) asserted byte-identical Latin output. This run
cannot reproduce that assertion directly, because Q6 changed the cheque images and the cheque
is a Latin document — so `script:Latn` moves 0.482 → 0.474 between B and C for a reason that
has nothing to do with Q12.

The report therefore makes the no-regression claim on the cells where the comparison is clean:
every `Latn|hand` cell moves by ≤ 0.002 between B and C, and the `Latn|print` movement is
explained by the cheque predictions ceasing to be stale. Stating it that way is weaker than
"byte-identical" and is the strongest thing this run's data actually supports.

An independent confirmation fell out of the provider cache: 8 documents hit the SLM cache,
which is keyed on `sha256(model + prompt + params)` and therefore on the OCR text. A cache hit
is proof that page's transcription is byte-identical to the previous run's.

### D-H3-8 · The `Latn|print` LID drop is reported as a regression, not absorbed into the rollup

Overall LID rose 0.802 → 0.860 and `Latn|print` fell 0.939 → 0.905. It would have been easy,
and wrong, to publish only the rollup. Drilling per type located the whole drop in
`complaint_form_deva` (Latin-print cell **1.000 → 0.068**, 105 ground-truth lines), with every
other type unchanged to three decimals — which is what turned a suspicious rollup into Q14.

The general rule this run applied: **a net improvement that contains a cell moving the wrong
way gets the cell reported, not the net.**

### D-H3-9 · Sentinel survival is published as a count with its n, not as a rate

Q7 asked for the sentinel-survival rate per run, and the measurement is now wired in — it reads
`value_en`, `flags` and `mt_engine` out of the committed predictions, so it needs no models and
re-derives from artifacts anyone has. This run it returned 5 attempted, 5 intact, 0
`TRANSLATION_UNAVAILABLE`.

The report prints "5/5 = 1.000 (n = 5)" and explicitly says not to read it as a rate. Q7 is
recorded as **unblocked, not answered**, with WS-F's 54/54 corpus-grounded figure left standing
as the best evidence. A 1.000 from five samples quoted as "the sentinel survival rate" is the
kind of number that gets into a slide and never gets re-measured.

Why only 5 of 29 Devanagari-valued fields reached MT is stated rather than left as a puzzle:
24 are deterministic fields resolved by lookup or regex (`bank_name` carries a native value and
an English `normalized` from the bank table — not a translation), and the SLM that owns the
free-text fields answered for 14 documents. The binding constraint on Q7 moved from Q12 to Q13.

### D-H3-10 · Latency is reported as *worse*, because it is

`s4_ocr` p50 rose 564 → 760 ms — the Q12 probe. `docs/decisions-q12.md` budgeted 25% of the
pre-fix stage and predicted 582 → 771 ms; measured here through a different harness the same
change is +196 ms, or ~26%. The report calls that "budget met to within measurement noise"
rather than "inside budget", because 26% against a 25% budget is not inside it.

The end-to-end p95 *improved* 8 258 → 3 895 ms and the report explicitly warns against reading
that as a speed-up: it is the SLM leg failing faster and more often. A p95 dominated by how
long a router takes to say no is not a measurement of this service.

---

## 3. The TLS interception problem

### D-H3-11 · `SSL_CERT_FILE` for the run's process, and it is written down

Every request from Python failed `CERTIFICATE_VERIFY_FAILED` on this machine. Cause: Norton
installs a TLS-inspection CA which the OS trust store and Node both trust
(`NODE_EXTRA_CA_CERTS=C:\ProgramData\Norton\Antivirus\wscert.pem`), and which `certifi` — what
`httpx` and the `openai` client use — does not carry. PowerShell reached the router; Python
could not.

What was done: `SSL_CERT_FILE` was pointed at a scratch bundle concatenating `certifi`'s
`cacert.pem` with the machine's **already-installed, already-trusted** Norton root, for the
duration of the run's process only.

What was deliberately **not** done: `verify=False`, `httpx` transport patching, or any edit to
a certificate store, a system setting, `.env`, or a repository file. Disabling verification
would have made every number in this report a number obtained over an unauthenticated channel,
which is not a trade a measurement run gets to make on its own authority.

This is recorded in `docs/eval/REPORT-2026-08-08-slmlive.md` §12 as well as here, because it is a reproducibility
fact: the previous run did not need it, and anyone re-running on a machine with TLS inspection
will.

---

## 4. Two defects found, and why neither was fixed

The lane rule is that a new defect gets documented precisely and stops there. Both of these are
in other lanes' committed files; neither was touched.

### D-H3-12 · Q14 — the line `script` label is the head's, not the line's

`app/providers/local.py::LocalProvider.ocr_page` sets `script=script` on every `OCRLine`, where
`script` is the recognition head. The line's own decoded characters are never consulted.

This was invisible for as long as the head was unconditionally Latin — most ground truth is
Latin, so the label was accidentally right. Q12 made the head vary and turned it accidentally
wrong: on a Devanagari form, `9106990984`, `SBIN0796790` and `28/03/2026` are all published as
`script: Deva`.

Evidence, all from this run's committed predictions:

- `complaint_form_deva` Latin-print LID cell **1.000 → 0.068** over 105 ground-truth lines,
  and that single cell accounts for the entire overall Latin-print decline.
- **0 of 147** recognised pages carry more than one distinct line script, against **30 of 147**
  non-bounce dev pages that genuinely mix scripts (15 `bilingual_form`, 15
  `complaint_form_deva`).
- `is_code_mixed` is derived in `assemble.derive_language_summary` by comparing line scripts to
  the page majority against `CODE_MIX_THRESHOLD = 0.15`. With one script per page it **cannot
  become true**, so the PRD checklist item it implements is unreachable.

Not fixed because `app/providers/**` is a committed lane. Lane: WS-C, with WS-E2 interested.
**The OPEN-QUESTIONS row is drafted in §6 below and not inserted** — only Q7/Q12/Q13 are in
this lane's write scope.

### D-H3-13 · Q15 — the Tier-2 VLM model is not served

`PaddlePaddle/PaddleOCR-VL` returns HTTP **400** with
`code: model_not_supported`, "not supported by any provider you have enabled" — deterministic,
and reproducible against a 1×1 image, so it is nothing about the page.

The previous run could not see this. Both the SLM's 402 and the VLM's 400 arrive at the stage
as `PROVIDER_ERROR`, so `REPORT-2026-08-07-preQ12.md` attributed all 25 Tier-2 failures to
credits. They were never a credit problem. `tier2_applied` has been 0 in both runs and will
stay 0 against this model on this account no matter what the balance is.

Consequence beyond the escalation path: PRD target 7 (router false-negative rate) is measured
by shadow-sampling documents and diffing Tier-1 against Tier-2. With no Tier-2 answer the diffs
are empty — 4 sampled this run, 0 completed — so that target is unmeasurable for a reason
entirely separate from Q13.

This is Q2 reopening (Tier-2 escalation model, which the PRD leaves unnamed). Not fixed:
choosing a model is an owner call, and `app/providers/**` is a committed lane. Row drafted in
§6, not inserted.

### D-H3-14 · One thing that is *not* claimed as a defect

Stage 6 translated an ASCII value: `prior_complaint_reference` = `CMS/2025/850859` came back as
`CMS / 2025 / 850859`. The immutable `value` is untouched and `value_en` is advisory, so
nothing downstream is wrong and no invariant is broken. It is recorded in `REPORT-2026-08-08-slmlive.md` §5 as an
observation for whoever is next in `s6_translate.py`, not as a finding. Inflating it into a
defect would dilute the two that are.

---

## 5. What this run does not measure

Unchanged from WS-H2 except where noted.

1. **The locked split.** Variants 3 and 4 (118 documents) were not run and not scored. Q12's
   probe geometry and its `PROBE_SCORE_MARGIN = 0.05` were tuned against variants 0-2, so the
   locked split now has **two** tuned-against-it subsystems to test honestly, not one.
2. **Multi-page behaviour.** Every corpus document is one page. The document quality verdict is
   the worst page's, and Q12's probe settles per page with no cross-page caching — both are
   correct and both are untested by this corpus.
3. **Handwriting.** Font-rendered jitter, not handwriting. `Deva|hand|clean` CER 0.249 is new
   and is not evidence about handwritten Devanagari.
4. **SLM field quality, Q10, and the router false-negative rate** — Q13 and Q15 respectively.
5. **Throughput** — not this machine (adjudication A-10).

---

## 6. Open / for the owner

1. **Q13 — SLM credits.** Owner action, unchanged, and now better characterised: the endpoint
   is reachable and refuses production-sized requests per-request. Blocks PRD target 2 and Q10.
2. **Q15 — Tier-2 model (drafted row).** *"`PaddlePaddle/PaddleOCR-VL` returns HTTP 400
   `model_not_supported` from the HF router — 'not supported by any provider you have
   enabled'. Deterministic, reproducible on a 1×1 image, and distinct from Q13's 402; both
   surfaced as `PROVIDER_ERROR`, which is why the M7 gate misattributed 25 Tier-2 failures to
   credits. `tier2_applied` is 0 in both runs. Blocks PRD target 7 (shadow diffs are empty
   without a Tier-2 answer). Options: (a) pick a VLM the router serves, (b) enable a provider
   that serves PaddleOCR-VL, (c) run Tier 2 locally, (d) drop Tier 2 from Phase 0 and say so.
   Reopens Q2. Lane: WS-C + owner."*
3. **Q14 — line script label (drafted row).** *"`LocalProvider.ocr_page` stamps the recognition
   head's script on every `OCRLine` instead of deriving it from the line's own characters. Now
   that Q12 makes the head vary, ASCII identifiers on Devanagari pages are published as `Deva`:
   `complaint_form_deva`'s Latin-print LID cell fell 1.000 → 0.068 over 105 lines, 0 of 147
   recognised pages report mixed scripts against 30 of 147 that genuinely mix, and
   `is_code_mixed` (`CODE_MIX_THRESHOLD` 0.15) can never be true, so a PRD checklist item is
   unreachable. Fix: classify each line's script from its decoded codepoints — the classifier
   `s4_ocr` uses for the A-6 vote — and keep the head's script as provenance. Lane: WS-C /
   WS-E2."*
4. **Bilingual pages need more than one head per page** (WS-E2). `bilingual_form` is unchanged
   by Q12 — 368/368 lines read as Latin, 60 Devanagari ground-truth lines unread — and is now
   the largest single block of residual Devanagari error. This is the natural follow-on to Q12
   and is a design question, not a bug.
5. **Q6's two owner calls** (`docs/decisions-q6.md` §10) — one field vocabulary, and whether to
   promote the bilingual `value`-slot check into precision. Both should be settled before
   `--final`. This run measured 0 contract violations in that slot, so promoting it costs
   nothing today.
6. **Q11** — evidence boxes in the post-repair coordinate frame. Untouched, still open.
7. **`recognise_crops` on the provider protocol** (`docs/decisions-q12.md` §5). This run
   confirms the probe costs ~26% of `s4_ocr` p50 against a 25% budget, so the case for it is
   now measured twice rather than once.

---

## 7. What this lane deliberately did not do

- **No source, test or corpus file was modified.** The two defects in §4 are both one-file
  fixes in other lanes and both were left alone.
- **No OPEN-QUESTIONS row outside Q7/Q12/Q13 was edited**, including the two rows this run
  earned. They are drafted in §6 for the orchestrator to place.
- **No git commit.**
- **No purchase, and no attempt at one.** Q13 is resolved by an owner spending money; this
  agent does not.
- **No retry storm on a 402**, and no fallback thrash — §1, D-H3-4.
- **TLS verification was not disabled** — §3, D-H3-11.
