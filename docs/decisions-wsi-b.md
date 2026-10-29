# WS-I(b) decisions — deployment and consumer documentation

Lane: `docs/AIR-GAPPED-DEPLOY.md`, `docs/API.md`, `constraints.txt`, this file.
Date: 2026-08-08. Machine: Windows 11, CPython 3.12.4, `.venv/Scripts/python.exe`.

Every number and every code block in the two documents was produced by running the thing it
describes on this machine. Where something could not be verified, the document says so
rather than asserting it. This file records what was decided, what was measured, and the
three defects the documentation work exposed.

---

## 1. Deliverables

| File | State |
|---|---|
| `constraints.txt` | **new**, repo root, 99 pins |
| `docs/AIR-GAPPED-DEPLOY.md` | **new**, 11 numbered sections + checklist + troubleshooting appendix |
| `docs/API.md` | **new**, 10 sections |
| `docs/decisions-wsi-b.md` | this file |

Nothing under `app/`, `tools/` or `tests/` was modified. No commits were made. No token was
printed, embedded or written anywhere; both documents show `HF_TOKEN=hf_xxx` placeholders
only, and `AIR-GAPPED-DEPLOY.md` §5.2 states explicitly that no token ships in either
bundle.

---

## 2. D-IB-1 — `constraints.txt` is a pruned freeze, and the prune is `torch`

**Decision.** `constraints.txt` is `pip freeze` of the working venv **minus `torch==2.13.0`
and `networkx==3.6.1`**, with a header explaining what a constraints file is and why those
two are absent.

**Why prune at all.** `torch` is **490 MB installed** (measured:
`.venv/Lib/site-packages/torch`, against a 1 222 MB venv total). It entered this venv only during the A-9
investigation — the abandoned "download the fp32 IndicTrans2 checkpoint and convert it to
CTranslate2 locally" path. The MT artifact that actually ships is the pre-converted
`adalat-ai` CT2 build, and nothing under `app/` or `tools/` imports `torch`.

**Why those two and nothing else.** Reverse-dependency check over the installed metadata:

```
torch     <- huggingface_hub, safetensors, transformers   (all as an OPTIONAL extra)
networkx  <- torch, and nothing else
sympy     <- fonttools, networkx, onnxruntime, torch       -> KEPT (onnxruntime needs it)
mpmath    <- sympy                                          -> KEPT (transitively required)
```

`sympy` and `mpmath` look like torch dependencies and are not: `onnxruntime` requires
`sympy`, which requires `mpmath`. Pruning them would have broken the offline install. This
is the reason the prune list is two entries and not four.

**Risk, stated plainly.** A constraints file installs nothing, so removing a pin cannot
remove a package. The only consequence is that if some future resolution *does* pull
`torch` in, its version is unpinned. The header names the one flow that needs it
(`fetch_models.py --mt-repo ai4bharat/...`) and says to pin it yourself.

**Rejected alternative — ship the raw freeze.** It is more literally "the environment we
built in", but it invites a deployer to add 490 MB to a wheelhouse for code that never
executes, and it implies a dependency the project does not have. The header preserves the
audit trail (both pruned versions are named) at no cost to reproducibility.

---

## 3. D-IB-2 — model SHA-256s were recomputed from bytes, because `PROVENANCE.json` is incomplete

**The instruction was to read the hashes from `models/PROVENANCE.json` and not invent them.
The file does not contain them.**

`tools/fetch_models.py::write_provenance` rewrites `PROVENANCE.json` **in full** on every
invocation, from that invocation's records only. The last run on this machine was
`--only mt` (the A-9 supersession), so the committed file names exactly one artifact:
`mt/indic-en`. The OCR, LID and classifier entries were overwritten out of existence.

**Decision.** The tables in `AIR-GAPPED-DEPLOY.md` §5.1 carry SHA-256 values computed
directly from the bytes on this machine, with the same `hashlib.sha256` streaming read that
`fetch_models._sha256` uses.

**The check that makes this safe.** One artifact appears in both the committed provenance
file and the recomputed set — `mt/indic-en/model.bin`. Both give
`a88c1c56918d267c4003279b2b3d4ed47f7e18a859d38ebb8491da6c422e4de8`. The two methods are the
same method, so the other twelve values are provenance-grade rather than merely plausible.

**What could not be recovered, and is documented as such.** `PROVENANCE.json` also records
the *resolved upstream revision* per repository. For the OCR and LID artifacts that
information no longer exists anywhere on this machine (`models/` is gitignored, so there is
no history to recover it from). §5.1 therefore gives repository URLs and file names but
pins a revision only for the MT repo, where it survives
(`524aade4f724dccbc6cd5ffb5e3c5c832a2baee4`). A boxed note in §5.4 tells the deployer to run
`fetch_models.py` with no `--only` flag on the staging machine and keep **that**
`PROVENANCE.json` as the audit record, not the table.

**Follow-up for whoever owns `tools/fetch_models.py`:** `write_provenance` should merge into
the existing file rather than replace it, or `--only` should refuse to write provenance at
all. As it stands, the tool's own audit trail is destroyed by the ordinary act of re-fetching
one artifact. Not fixed here — outside this lane.

**A second gap, recorded not fixed.** `models/classifier/fusion.json` is a *required*
artifact (`/healthz` reports it, `s3_classify` loads it) that `fetch_models.py` never
produces and `PROVENANCE.json` never mentions. It is produced by
`tools/train_page_classifier.py` and committed at `docs/calibration/fusion.json`. §5.1
documents it as a copy step with its hash; the identity of the two files was verified
(both `ea03e8671526701f...`, 1 442 bytes).

---

## 4. D-IB-3 — Q16: the API test page cannot work as shipped, for two independent reasons

**New open question, raised here.**

`PLAN.md` WS-I item 1 requires a standalone HTML test page that "works from `file://` or
served by FastAPI". `app/review_ui/__init__.py` states "served by FastAPI (WS-F mounts
`STATIC_DIR`)". **WS-F does not mount it.** `app/api/app.py` adds one router and four
routes; there is no `StaticFiles` mount and no CORS middleware — the module docstring
explicitly puts CORS out of scope as "a reverse-proxy concern".

**Measured:** `GET /ui/test_page.html` against `uvicorn app.api.app:create_app --factory`
returns **404**.

Consequence, both routes blocked:

- **Served by FastAPI:** impossible; nothing serves it.
- **From `file://`:** the page's origin is `null` and the API's is
  `http://127.0.0.1:8000`. A `fetch` across that boundary needs
  `Access-Control-Allow-Origin` on the response, and the service sends none — so the
  browser blocks the *response*, and the upload silently appears to do nothing.
  `app/review_ui/__init__.py` even anticipates this ("The `file://` route needs the API to
  allow the `null` origin in CORS"), but nothing provides it.

**Decision.** Document a launcher rather than modify `app/` (out of lane) and rather than
document a broken step. `AIR-GAPPED-DEPLOY.md` §9.4 gives a six-line `serve.py` the deployer
creates at the project root:

```python
from fastapi.staticfiles import StaticFiles

from app.api.app import create_app
from app.review_ui import STATIC_DIR

app = create_app()
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
```

**Verified working** before it was written into the document: with this file,
`uvicorn serve:app` served `/healthz` 200, `/ui/test_page.html` 200 (13 289 bytes),
`/ui/review_ui.css` 200 `text/css`, `/ui/review_ui.js` 200 `text/javascript`,
`/ui/review.html` 200. The shim file was deleted from the repository afterwards; the working
tree carries none of it.

**Why a same-origin mount and not a CORS header.** Mounting makes the page and the API the
same origin, so no cross-origin permission is needed at all, and `review_ui.js`'s default
API base (`http://127.0.0.1:8000`) is already correct. Adding `Access-Control-Allow-Origin:
null` would be the alternative, and `null` is the *worst* origin to allowlist — every
sandboxed iframe on the internet also presents it.

**A second, independent defect in the same page, found while writing the walkthrough.** The
upload builds its `FormData` with the file and nothing else:

```js
var form = new FormData();
form.append(fieldName || "file", file, file.name);      // review_ui.js
```

It never sends `synthetic`. `POST /v1/documents` requires that declaration under the
`laptop` profile, so **every upload from the test page to a laptop-profile instance returns
HTTP 422 `RESIDENCY_VIOLATION`** — including on the development machine the page was written
on. Against `cpu_node` the field is ignored and the page works, which is why the air-gapped
target is unaffected and why this went unnoticed. Documented in §9.4 with the workaround
(`PROVIDER_PROFILE=cpu_node` for the experiment, or `curl`).

**Q16, for `OPEN-QUESTIONS.md`** (not inserted there — outside this lane's write scope):
*The WS-I standalone test page cannot be used as specified, for two independent reasons.
(i) `STATIC_DIR` is never mounted — `app/api/app.py` adds one router and no `StaticFiles` —
so `GET /ui/test_page.html` is 404 on the shipped application, and the `file://` route is
blocked by the same-origin policy because the service sends no CORS headers (measured: 404
and a `null`-origin block respectively). (ii) `review_ui.js` posts only the file, never
`synthetic=true`, so every upload to a `laptop`-profile instance is refused 422. Options:
(a) mount `STATIC_DIR` at `/ui` in `create_app` and add a synthetic-declaration checkbox to
the page — same origin, no CORS surface, both defects closed; (b) ship a `serve.py`
launcher in the code bundle and document the profile constraint; (c) add CORS middleware
allowing the `null` origin — rejected, `null` is presented by every sandboxed iframe and
cannot be restricted. Recommendation (a), with the mount behind a setting so a production
deployment can switch the pages off. Lane: WS-F / WS-G.*

---

## 5. D-IB-4 — the test-suite section publishes the failure it actually saw

The suite is 2 599 tests. The run performed for this document returned:

```
FAILED tests/unit/test_pipeline.py::test_the_head_probe_stays_inside_its_latency_budget
1 failed, 2598 passed, 1 warning in 136.13s (0:02:16)
```

That run overlapped a live-API capture using the same cores, and the failing test asserts a
**timing** budget for the Q12 script-detection probe. Re-run alone: `1 passed in 8.56s`.

**Decision.** §8.1 quotes the failing output verbatim, names the test, explains the
contention, gives the single-test re-run command with its measured time, and then states the
acceptance rule: *that* test failing alone is acceptable; any other failure is not.

**Rejected alternative — print `2599 passed` as the expected output.** It is what an idle
machine produces, but it was not what this machine produced, and a deployer who sees
`1 failed` on a busy build agent needs to know whether to stop. Writing the clean number and
omitting the observed one would make the document less useful precisely at the moment it is
consulted.

**Observation for whoever owns the probe budget:** a wall-clock assertion in the unit suite
will fail on any loaded CI runner. A CPU-time budget, or a `pytest` marker excluding it from
parallel runs, would remove a recurring false alarm. Not this lane.

---

## 6. D-IB-5 — the API document had to contradict a naive reading of the error taxonomy

`DocumentRejected` and its ten subclasses carry `http_status = 422`, which reads as "bad
input returns 422". **Measured against a running instance, it does not.** Six inputs
submitted through `POST /v1/documents`:

| input | HTTP | `status` | `bounce_decision.reason` |
|---|---|---|---|
| zero-byte `.pdf` | **200** | `bounced` | `EMPTY_FILE` |
| truncated `.pdf` | **200** | `bounced` | `MALFORMED_PDF` |
| HTML renamed `.pdf` | **200** | `bounced` | `MIME_MISMATCH` |
| GIF renamed `.pdf` | **200** | `bounced` | `MIME_MISMATCH` |
| 60×60 px PNG | **200** | `bounced` | `RESOLUTION_BELOW_FLOOR` |
| blank-page PNG | **200** | `bounced` | `BLANK_PAGE` |

The orchestrator converts every ingestion rejection into a **bounce envelope**, which is the
right product behaviour — a refused document is an answer about the document, and the
complainant-facing auto-reply is keyed on `BounceReason`, not on an HTTP status. The `422`
on the exception class is a hint the API layer never gets to use for these.

**Decision.** `API.md` §3.1 leads with "you still get HTTP 200 and a valid envelope",
tabulates all six measured cases, and §8.1 ends with an explicit boxed statement that
ingestion problems are **not** in the HTTP error table. The two codes that genuinely do
produce non-2xx were measured too: `RESIDENCY_VIOLATION` → **422** (fresh document, no
`synthetic` field) and `UPLOAD_TOO_LARGE` → **413** (with `API_MAX_UPLOAD_BYTES=1024`).

Also measured and documented: `JOB_NOT_FOUND` → 404; idempotent resubmission → 200,
`is_new: false`, **byte-identical envelope and same `job_id`** (asserted by dict comparison,
not eyeballed); soft duplicate → `soft_duplicate_of` set with reason
`same_sender_subject_day`.

---

## 7. D-IB-6 — the consumer document states the known limitations instead of omitting them

A consumer-facing reference that describes only the happy contract is how integrations get
built on fields that cannot work. Four are named in `API.md` at the exact field they affect,
each linked to its open question:

| Where | What is said | Ref |
|---|---|---|
| §4.3 `is_code_mixed` | Can never become `true` in this release: every line carries the *recognition head's* script, not its own. 0 of 147 pages reported mixed scripts against 30 that genuinely mix. "Do not build logic on this in v1." | Q14 |
| §4.3 evidence coordinates | When `repairs_applied` contains `deskew`, every `bbox` describes the straightened raster, not the uploaded image. | Q11 |
| §4.6 `tokens_used` | Reads `0` even when tokens were spent — orchestrator and provider keep separate budgets. "Do not bill on it." | Q9 |
| §8.4 `tier2_failed` | Every escalation currently ends this way; the escalation *decision* path works, the *outcome* path has never executed. | Q15 |

**Rejected alternative — document the intended behaviour and let the reader discover the
gap.** For `is_code_mixed` in particular that would produce a routing rule that is silently
dead. The cost of saying it is one sentence; the cost of not saying it is a consumer bug
nobody can find.

---

## 8. D-IB-7 — the webhook verification snippet was executed against a real delivery

`API.md` §7.3 contains a `verify_rbi_webhook` function a consumer pastes into their code.
It imports nothing from this project, and it was **run** before being published: a throwaway
HTTP receiver on `127.0.0.1:8222`, a real document submitted through the app with
`WEBHOOK_DELIVERY_MODE=inline` and a real shared secret, and the captured delivery fed to the
function.

| case | result |
|---|---|
| genuine payload, correct secret | `True` |
| wrong secret | `False` |
| body with one byte appended | `False` |
| stale timestamp (tolerance −1) | `False` |

Captured headers, reproduced verbatim in §7.1:

```
x-rbi-signature:      sha256=0e3f871ee29ce68d4e7a6edcc09bb7b0053401e856bc913e6d76ce8b7f4d7d4c
x-rbi-timestamp:      1786137805
x-rbi-job-id:         ea7f81681afd4a8a848a21fac7686ea0
x-rbi-result-version: 1
x-rbi-delivery-id:    1
```

Also confirmed and stated in the document: the request body is the **envelope itself**, not
a wrapper (top-level keys are exactly the envelope's), and the three properties that make the
scheme work — timestamp inside the MAC, verify the raw bytes, constant-time compare — are
each given a one-sentence reason rather than asserted.

`§7.4` restates the same algorithm as language-neutral pseudo-code for consumers not writing
Python, per the brief.

---

## 8b. D-IB-7b — a documented command was found broken by testing it: the `pip.ini` BOM

The obvious PowerShell recipe for creating `.venv/pip.ini` —

```
@"
[global]
trusted-host = ...
"@ | Out-File -Encoding utf8 .venv\pip.ini
```

— **produces a file pip cannot read.** Windows PowerShell 5.1's `-Encoding utf8` writes a
byte-order mark (`EF BB BF`), pip's config parser does not strip it, and pip aborts:

```
Configuration file could not be loaded.
File contains no section headers.
file: '...\pip_test.ini', line: 1
'ï»¿[global]\n'
```

Measured by pointing `PIP_CONFIG_FILE` at the generated file and running
`pip config list`: exit **2** with a BOM, exit **0** without.

**Decision.** §3a.1 uses `-Encoding ascii` (the content is pure ASCII, so nothing is lost
and no BOM is written), shows the exact failure text in a boxed note so the symptom is
searchable, adds `pip config list` as a verification step with its expected output
(`global.trusted-host='\npypi.org\nfiles.pythonhosted.org'`, verified), and repeats the fix
in the troubleshooting appendix.

This is the clearest argument for executing documentation rather than writing it: the
`utf8` version looks more correct than the `ascii` one, and it is the broken one.

---

## 9. D-IB-8 — the air-gapped profile is `cpu_node`, said loudly and early

`.env.example` ships `PROVIDER_PROFILE=laptop` + `SYNTHETIC_ONLY=true`, which is correct for
the machine it was written on and **wrong** for the machine this guide is about. Under
`laptop` the service refuses any upload that does not declare itself synthetic, which on an
air-gapped production node would refuse every real complaint.

`AIR-GAPPED-DEPLOY.md` §7.1 opens with a boxed statement — *air-gapped nodes set
`PROVIDER_PROFILE=cpu_node`* — §7.2 gives the two `.env` files side by side, and the
troubleshooting appendix names the exact symptom ("every upload returns 422
`RESIDENCY_VIOLATION`") with the fix. `API.md` §2.1 tells the consumer the mirror image:
if you see that error, you are talking to a development instance.

Related, and stated in both documents: **the deterministic half of the product needs no
token and no network.** Every identifier and amount comes from validators (invariant 7), so
an air-gapped node with `HF_TOKEN=` empty returns a complete, correct envelope minus the
free-text fields, with `slm_unavailable:PROVIDER_METHOD_NOT_SUPPORTED` in `warnings`. That
was measured, not assumed: `run_corpus_eval.py --no-slm` over
`complaint_form_en_v0_clean` returned `quality accept 1.0`, 25 lines and six extracted
fields.

---

## 10. Measurements taken for these documents

Every one is quoted in `AIR-GAPPED-DEPLOY.md` with its section number.

| What | Result | §  |
|---|---|---|
| Python / venv | `Python 3.12.4` | 1.3, 3.0 |
| Full test suite | 2 599 tests; observed `1 failed, 2598 passed … 136.13s`; the failure passes alone in `8.56s` | 8.1 |
| Fonts + shaping | 8 faces, `raqm True`, fribidi present, Pillow 12.3.0 | 4 |
| Model artifacts | 13 files hashed; all present; MT hash matches `PROVENANCE.json` exactly | 5.1, 5.4 |
| Corpus, committed | 295 documents, generator 1.2.0, seed 1729; 298 image files | 6 |
| Corpus, regenerated | 2 m 22 s; **599 of 599 files byte-identical** to the committed corpus | 6 |
| Pipeline smoke, 2 docs, `--no-slm` | 7.5 s wall; `accept 1.0`, 25 lines, 6 fields, 1 warning | 8.2 |
| Quality-gate calibration | `calibration reproduces the committed file`, 66 s | 8.3 |
| Eval harness re-score | 1.3 s, 177 scored / 118 withheld, harness 1.1.0 | 8.4 |
| `uvicorn … --factory` start | `Application startup complete.` | 9.1 |
| `/healthz` | `"status": "ok"`, six models `present: true`, `database: "ok"` | 9.2 |
| `/metrics` | 336 lines, includes `provider_profile{profile="laptop"} 1.0` | 10 |
| `curl` upload, live | HTTP 200, `completed`, 7 fields incl. one `source: "slm"`, `warnings: []` | 9.3 |
| First-request latency | ~11.3 s (lazy model load); ~1.4 s median thereafter | 9.3 |

The HF credit situation is **not** what the last report saw: a live `POST /v1/documents`
during this work returned a validated SLM answer (`prior_complaint_reference` =
`CMS/2025/705869`, `source: "slm"`, `warnings: []`). A second, otherwise identical
submission returned `slm_unavailable:PROVIDER_RESPONSE_INVALID` — the model answered but
failed Pydantic twice. Both are documented as expected states. **No conclusion about Q10 or
Q13 is drawn from n = 2**, and none is offered.

---

## 11. Things deliberately not done

- **No `app/` change for Q16.** The static mount belongs to WS-F/WS-G. Documented with a
  verified workaround and raised as an open question instead.
- **No `OPEN-QUESTIONS.md` edit.** Q16 is drafted in §4 above for the orchestrator to insert;
  writing into another lane's register mid-flight is how registers get clobbered.
- **No `PROVENANCE.json` regeneration.** It would require a full network fetch, and the file
  belongs to `tools/fetch_models.py`'s lane. The gap is documented with the safe substitute.
- **No commit.** Per the working rules.
- **No purchase, no credential.** No token appears in any file produced by this lane.
- **`tools/make_bundle.py` was not written.** It is PLAN WS-I item 7 and a different lane;
  `AIR-GAPPED-DEPLOY.md` §0 describes the two bundles from the PLAN specification and notes
  what to do if the tool has not landed.
