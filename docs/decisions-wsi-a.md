# WS-I-A decisions — the SLM-live re-run and the two metrics-bearing documents

Judgment calls made while re-running `tools/run_corpus_eval.py` over the dev split with the SLM
reachable, refreshing `docs/eval/**`, and writing `docs/FUNCTIONAL-SPEC.md` and
`docs/TUNING.md`.

Lane: `tools/run_corpus_eval.py` **invocation only** (no source edit), `docs/eval/**`,
`docs/FUNCTIONAL-SPEC.md`, `docs/TUNING.md`, this file, and the **Q7 / Q13 rows only** of
`OPEN-QUESTIONS.md`. Nothing outside it was touched — no source file, no test, no corpus file,
no other OPEN-QUESTIONS row, no `.env`, no settings. Two new defects surfaced and were
**documented rather than fixed** (§4).

Like WS-H2 and WS-H3: this lane measures, it does not fix. Every number in `docs/eval/REPORT.md`
came out of the working tree as it stood.

---

## 1. Running the sweep

### D-IA-1 · Probe at production size before spending 23 minutes

The brief said credits were topped up and verified with a `gemma-3-4b-it` probe. That was not
taken as sufficient, because **the previous run's failure looked exactly like a healthy
endpoint under a small probe**: WS-H3 measured a 19-token request returning 200 on both models
while 129 of 147 production-sized requests returned 402.

So three probes ran before the sweep, in order:

1. 19 tokens → `google/gemma-3-4b-it` → **200 OK**
2. 19 tokens → `meta-llama/Llama-3.1-8B-Instruct` → **200 OK**
3. **2 032 prompt tokens, `max_tokens=512`, `response_format={"type":"json_object"}`,
   `temperature=0.0`** → **200 OK**, 133 completion tokens, well-formed JSON

Only the third one is evidence. It cost 2 165 tokens — 0.8% of what the sweep went on to spend —
and it is the check that would have caught the previous run's condition before burning the time.

**This is now written into `docs/TUNING.md` §5.4 as the standing instruction**, because the
lesson is not "credits are fine today", it is "a trivially-sized health check on this router
proves nothing".

### D-IA-2 · The sweep ran live, sequentially, with no source change

`--score` only. No `--resume`, no `--no-slm`, no `--limit`, no flag that would have changed what
was measured. All 177 prediction files were rewritten, so nothing was inferred from a stale one;
`documents_ok` is 177 and `documents_failed` is 0.

`--resume` was specifically rejected. The predictions on disk were the *previous* run's, and
resuming would have produced a directory silently mixing SLM-blocked and SLM-live documents with
no marker distinguishing them — every aggregate a blend of two conditions.

The run took **1 370 s** against the brief's estimated ~7 minutes. The estimate assumed the OCR
was the cost. With the SLM answering, a live document costs a p50 of **7.3 s** of wall clock and
the network is the cost. The sweep was left strictly sequential anyway (D-H2-2): PRD target 8 is
a latency target and a number measured under self-inflicted parallel load is not that number.

### D-IA-3 · The previous run's predictions were snapshotted before being overwritten

`docs/eval/predictions` was copied to a scratch directory before the sweep, and
`REPORT.md`/`report.json`/`report.txt`/`PREDICTIONS_MANIFEST.json` with it. That snapshot is what
made §2 of the report — the control table — and §3.3's per-type before/after possible, both
scored through the **same** corpus and the **same** harness.

The superseded report is also kept as a file: `docs/eval/REPORT-2026-08-08-preSLM.md`, byte-copied
before the rewrite, following D-H3-2. A reader checking "0.028 → 0.119" should not have to know a
git sha.

### D-IA-4 · The fallback model was left configured and untouched

`SLM_MODEL` stayed `google/gemma-3-4b-it` for the whole sweep. The driver's automatic fallback
did not engage and should not have: it arms on five consecutive *outage* codes, and
`PROVIDER_RESPONSE_INVALID` is deliberately not in that set (a model answering badly is an
ordinary event, and swapping on it would silently change what the sweep measures).

**The report recommends swapping to the fallback and explicitly does not do it.** `app/settings.py`
and `.env` are outside this lane, and a mid-lane model change would leave one document reporting
two configurations. The recommendation is stated with the arithmetic that supports it (§4.2 of the
report) so the owner can act on it in one step.

---

## 2. Reading the results

### D-IA-5 · The control table leads, because this run has a real control

Every non-SLM number came back **byte-identical** to the previous run: overall CER agrees to
sixteen decimal digits, and detection, LID, the nine quality-gate cells, the escalation counts
and all eight deterministic field buckets agree as integers.

That is not filler. WS-H3 had to publish a three-column decomposition precisely because two
changes landed together and neither could be attributed alone. Here **one variable moved**, and
the cheapest way to make every later delta trustworthy is to show the others did not. So §2 of the
report is a table of things that did *not* change, placed before anything that did.

An independent cross-check came free: 14 provider-cache hits, on a key that is
`sha256(model + prompt + params)` with the OCR text inside the prompt. A hit is proof that
document's transcription is byte-identical to the previous run's.

### D-IA-6 · "Deterministic fields did not move" is reported as a finding, not an absence

The obvious way to write target 1 was "unchanged, still a MISS, nothing to see". That throws away
the most interesting thing the run produced.

Invariant 7 says deterministic fields never pass through the SLM. Until this run that claim had
never been tested under a condition that could falsify it — the SLM answered 14 documents. It now
answers 147, and **not one of TP 481 / FP 109 / FN 296 moved on any of the eight identifier
fields**. The report says so explicitly.

### D-IA-7 · PRD target 2 is promoted from NOT MEASURED to MEASURED — MISS

WS-H3 refused the verdict at 14 of 147 documents, correctly. This run asked all 147 and got an
answer from the router every time, so the denominator is real and the verdict is earned:
**F1 0.119, MISS**.

The distinction the report insists on: the number is *low* because the model's answers fail the
contract, **not** because anything is unmeasured. "Not measured" and "measured and bad" are
different states and the second one is actionable.

### D-IA-8 · Two SLM buckets are published, and the wider one is the verdict

Two of the six fields in the SLM bucket cannot score above zero for reasons that have nothing to
do with the model (§4 below). Excluding them lifts F1 0.119 → 0.174.

**Both are printed; the six-field number is the verdict.** Publishing only the narrower one would
be choosing a denominator after seeing the numerator. Publishing only the wider one would let two
plumbing defects masquerade as model quality. Printing both, with the reason, is the only version
that survives a reader who checks.

### D-IA-9 · Q10 is answered as a ceiling, not as a ranking

The temptation was to write "gemma scored 0.367, Llama scored 1.000, therefore switch". That is a
comparison between one measurement at n = 147 and another at n = 18, which is not a comparison.

The report makes the argument that does not depend on the second number at all:

> A document whose answer fails the contract contributes zero SLM fields, therefore
> recall ≤ 0.367 × (whatever the surviving answers achieve).

The target is 0.80. **The configured model cannot reach it under any assumption about answer
quality.** The fallback's 1.000 is then quoted as the reason to expect the swap to help, clearly
labelled as n = 18 from the bake-off.

The 14-fold spread across document types (0.933 on `receipt`, 0.067 on `bank_statement`) is
reported as an **association without a claimed mechanism**. Length correlates at the tail — 0.083
past 60 lines — but the median line count of a valid answer is 15 and of a failed one 14, so
length does not explain the body of it. Diagnosing it needs the rejected payloads, which the
driver does not retain, and inventing a mechanism from a rank correlation is how a wrong fix gets
scoped.

### D-IA-10 · The grounding gate is exonerated with the decomposition, not with a total

64 grounding rejections is an alarming-looking number and it would have been easy to report it as
one. Decomposed by reason it is **61 `empty` and 3 `ungrounded`**: the model declining to answer,
versus the gate rejecting a value not on the page.

So the gate discarded **3 of 104 proposals — 2.9%** — and the report says in terms not to raise
`GROUNDING_MAX_EDIT_DISTANCE` in response, because the recoverable amount is three values and the
cost is invariant 8. Both `docs/TUNING.md` §5.3 and the report carry the decomposition and the
"do not tune this" instruction, because a total without its split is exactly the number that gets
someone to loosen a safety gate.

### D-IA-11 · Latency is reported as much worse, and as the first honest measurement

End-to-end p95 rose **3 895 → 22 609 ms**. WS-H3 had already warned its own fast p95 was "the SLM
leg failing faster"; this run confirms it by removing the failure. Every local stage is unchanged
within noise — `s4_ocr` p50 moved 760 → 751 ms.

Two things the report adds rather than smoothing over: the per-model-path split (live 7 332 ms
p50, cached 1 067 ms, bounced 146 ms), which is the cost of the remote SLM stated plainly; and a
**new caveat on target 8** — one-page documents at 22.6 s against a five-page 90 s target no
longer has comfortable headroom, and this corpus cannot test it.

### D-IA-12 · Sentinel survival is still published as a count with its n

Q7 moved from n = 5 to **n = 29**, still 0 discards. The report prints "29/29 = 1.000 (n = 29)"
and again says not to read it as a rate, keeping WS-F's corpus-grounded 54/54 as the best evidence.
Six times more data is still not a rate, and D-H3-9's reasoning is unchanged.

One genuinely new finding did clear the sample-size bar: **`value_en` is wrong on 10 of the 19
judged bilingual slots**. At the previous n (3 judged) that was unsayable. The report states it
plainly — the advisory English rendering is close to a coin flip on this corpus and must not be
shown to a reviewer as authoritative.

---

## 3. The two documents

### D-IA-13 · `FUNCTIONAL-SPEC.md` gets the honest verdicts, phrased so a PM can repeat them

The owner's standard is "assume the reader knows nothing". Applied to a metrics table that
contains two MISSes and a BLOCKED, that means each row carries a **quotable one-line explanation**
next to the verdict, not just the number — "when it claims an identifier it is right 82% of the
time; the problem is it only finds 62% of them" rather than "F1 0.704".

Three deliberate choices in that document:

- **The MISSes are in the headline table, not in a footnote.** A PM who reads only §8.1 should
  come away knowing the free-text half does not work yet and why.
- **§8.3 names the three causes and separates the fixable from the structural**, so nobody plans
  around "the model needs tuning" when a third of the shortfall is two plumbing defects and the
  rest is a model swap.
- **§8.4 says what went right**, because a spec that only lists failures gets read as a system
  that does not work — and "not one identifier moved while the model went from 14 answers to 147"
  is the strongest safety evidence this project has produced.

The flow diagram is a single mermaid graph. ASCII was considered and rejected: the flow has two
branch points and a rejoin, and ASCII forces either a lie or a wall of pipes.

### D-IA-14 · `TUNING.md` is keyed by symptom, and every measured claim carries its measurement

The failure→remedy playbook (§5) is indexed by **what the engineer observes**, not by what is
wrong — "Devanagari CER is high", "bounce rate is out of band", "every HTTPS request fails" —
because the reader does not yet know the cause, and a table keyed on causes is only usable by
someone who does not need it.

Every tunable that has been measured carries the measurement inline rather than a citation:
`OCR_BATCH_SIZE` carries the whole A-10 speed/agreement table including the 0.556 Devanagari text
agreement without bucket padding; the probe constants carry the 146/147 grid-search result and
the 0.079-vs-0.033 margin data; `JOB_TOKEN_BUDGET` now carries this run's 286 198 tokens and
5 495-token worst case. A range without the evidence behind it invites someone to "try a bigger
number".

§4.10 exists to say plainly which knobs are **module constants and not settings**, and why
promoting one is a code change plus a re-run. Without it the natural assumption is that anything
tunable is in `.env`.

---

## 4. Two defects found, and why neither was fixed

The lane rule is that a new defect gets documented precisely and stops there. Both are in other
lanes' files. Together they are **144 of the SLM bucket's 414 misses**.

### D-IA-15 · Q16 — `complaint_summary` is produced but structurally unscoreable

The pipeline produced a summary for **47 documents, 198 spans total**. The harness scores
`complaint_summary` at **0 TP / 0 FP / 90 FN**.

`tools/run_corpus_eval.py::prediction_from_result` publishes the summary as `summary_span_count`,
an integer, and never as an entry in the prediction's `fields` map. The corpus reference does
carry `complaint_summary` (90 dev-split instances), so the harness scores a field nothing ever
claims — 90 guaranteed misses inside a published PRD target.

`tools/eval_harness.py`'s scope note already says it does not score "summary-span quality", which
is a defensible boundary. The problem is the combination: the field is in the reference and in
the target's denominator while being impossible to earn. Either emit it as a scorable field, or
remove it from the reference bucket and say in the target's definition that it is measured by the
envelope's contract tests. Not both, and not neither.

Not fixed: this lane's write scope is the driver's *invocation*, not its source, and the harness
is WS-D's. **Row drafted in §6, not inserted** — only Q7/Q13 are in this lane's write scope.

### D-IA-16 · Q17 — `complainant_contact` has no extractor anywhere

54 reference instances, scored as an SLM field, and it appears in **neither**
`SLM_ELIGIBLE_FIELDS` (`{complainant_address, complainant_name, complaint_summary,
prior_complaint_reference, relief_sought}`) **nor** the validator table. The model is never asked
for it and no rule produces it. 54 guaranteed misses that no amount of model tuning can move.

Worth noting this is a **product** question and not only bookkeeping: a phone number or email
address has structure a regex can validate, so it arguably belongs on the deterministic side of
the line rather than the free-text side — where it would also gain a checksum-style guarantee.

Not fixed: `app/contracts/` and `app/stages/s5_validators.py` are committed lanes, and which side
of the deterministic/free-text boundary the field belongs on is an owner call. Row drafted in §6.

### D-IA-17 · Two things that are *not* claimed as defects

- **The ASCII value handed to the MT engine** — `CMS/2025/850859` → `CMS / 2025 / 850859` —
  reproduces from the previous run, and a new cosmetic variant appeared (`पवेल, मराद` →
  `Pavella : Marad`, a comma rendered as a colon). `value` is immutable and `value_en` is
  advisory, so no invariant breaks and nothing downstream is wrong. Recorded in `REPORT-2026-08-08-slmlive.md` §5.3
  as an observation for whoever is next in `s6_translate.py`. Inflating either into a defect
  would dilute the two that are.
- **`assemble` p95 rising 0.5 → 61.4 ms** is the summary being built for the first time on 47
  documents. It is 0.3% of end-to-end. Reported in the latency table and explicitly called not
  worth attention.

---

## 5. What this run does not measure

1. **The locked split.** Variants 3 and 4 (118 documents) were not run and not scored. The
   quality gate and the recognition probe were both tuned against variants 0-2.
2. **Multi-page behaviour** — and this now matters more than it did. Target 8's headroom shrank
   from 3.9 s to 22.6 s against a five-page 90 s target, and every corpus document is one page.
3. **Handwriting.** Font-rendered jitter, not handwriting.
4. **The router false-negative rate** — Q15, unchanged.
5. **Whether a different prompt would fix schema validity.** Only one prompt was exercised. The
   driver does not retain rejected payloads, so the failure mode was not characterised beyond its
   rate.
6. **Throughput** — not this machine (adjudication A-10).

---

## 6. Open / for the owner

1. **Q10 — swap `SLM_MODEL` to `meta-llama/Llama-3.1-8B-Instruct` and re-run.** Now supported by
   a measured ceiling rather than a bake-off ranking: 0.367 schema validity caps free-text recall
   at less than half the target regardless of answer quality. One `.env` line plus ~23 minutes.
   **This is the highest-value single change available.**
2. **Q16 — `complaint_summary` unscoreable (drafted row).** *"The pipeline produces a complaint
   summary for 47 of 147 dev-split documents (198 spans), but
   `tools/run_corpus_eval.py::prediction_from_result` publishes it as the integer
   `summary_span_count` rather than as an entry in the prediction's `fields` map. The corpus
   reference carries `complaint_summary` (90 dev-split instances), so the harness scores it
   0 TP / 0 FP / 90 FN — 90 guaranteed misses inside PRD target 2's denominator, impossible to
   earn. `tools/eval_harness.py` separately declares summary-span quality out of scope. Options:
   (a) emit the summary as a scorable field and let the harness match it like any other,
   (b) remove `complaint_summary` from the reference bucket and state in the target's definition
   that it is measured by the envelope's contract tests, (c) leave it and subtract 90 by hand
   every time the number is quoted. Lane: WS-H (driver) / WS-D (harness). Must be settled before
   the `--final` run."*
3. **Q17 — `complainant_contact` has no extractor (drafted row).** *"`complainant_contact` has 54
   dev-split reference instances and is scored inside PRD target 2, but appears in neither
   `SLM_ELIGIBLE_FIELDS` nor the validator table: the SLM is never asked for it and no
   deterministic mechanism produces it. 54 guaranteed misses no model tuning can move. It is also
   a design question — a phone number or email address has regex-checkable structure, so it may
   belong on the deterministic side of invariant 7 rather than the free-text side. Options:
   (a) add a `validate_contact` validator and move it to the deterministic bucket, (b) add it to
   `SLM_ELIGIBLE_FIELDS`, (c) remove it from the scored reference. Lane: WS-E3 / WS-B + owner
   call. Must be settled before the `--final` run."*
4. **Q15 — Tier-2 model.** Unchanged and now definitively not a credit problem: the SLM leg on
   the same router succeeded 147/147 while the Tier-2 leg failed 28/28 with HTTP 400
   `model_not_supported`. Blocks PRD target 7. Owner decision on the model; reopens Q2.
5. **Q14 — line script label.** Unchanged; every LID number is identical to the previous run's.
6. **Q9 — token accounting and the per-instance budget.** Now quantified: median 1 914 tokens per
   document against a `JOB_TOKEN_BUDGET` of 8 000, so an un-reset batch caller exhausts it after
   **two** documents. `ProcessingStats.tokens_used` is still 0 on every envelope.
7. **Q11** — evidence boxes in the post-repair coordinate frame. Untouched, still open.
8. **Q6's two owner calls** (`docs/decisions-q6.md` §10). This run measured **0 contract
   violations** in the bilingual `value` slot over 29 claims — double the previous sample — so
   promoting that check into precision still costs nothing.
9. **A multi-page latency measurement** before target 8 is claimed as met with the SLM live.

---

## 7. What this lane deliberately did not do

- **No source, test, corpus or settings file was modified.** `tools/run_corpus_eval.py` was
  *invoked*, never edited. `SLM_MODEL` was not changed despite the report recommending it.
- **No OPEN-QUESTIONS row outside Q7/Q13 was edited.** Q16 and Q17 are drafted in §6 for the
  orchestrator to place, following D-H3's precedent.
- **No git commit.**
- **No purchase, and no attempt at one.** Credits were already on the account when this lane
  started; this agent verified they worked and spent 286 198 tokens of them, which the brief
  authorised as the point of the run.
- **TLS verification was not disabled.** `SSL_CERT_FILE` was pointed at a scratch bundle
  concatenating `certifi` with the machine's already-installed, already-trusted Norton root, for
  the run's process only. No system setting, no certificate store and no repository file was
  modified. Recorded in `REPORT-2026-08-08-slmlive.md` §11.5 and `docs/TUNING.md` §5.9 because it is a
  reproducibility fact.
- **The HF token was never printed, logged or written to any file.** The manifest records
  `hf_token_present: true` and not its value or its length; both documents show `HF_TOKEN=hf_xxx`
  placeholders only.
