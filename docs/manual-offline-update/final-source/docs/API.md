# API reference — RBI complaint document intelligence, v1

## Select fields on upload

Add the optional multipart field `requested_fields` with comma-separated names:

```sh
curl http://127.0.0.1:8000/v1/documents \
  -F 'file=@document.jpg' \
  -F 'requested_fields=complainant name,mobile,account number'
```

Add `-F 'synthetic=true'` for synthetic test data under the laptop profile.
The test page provides the same input. Names have no allowlist and are normalized
to snake_case. Blank entries and duplicate names are ignored; the list is limited
to 4,096 characters. Use descriptive names matching document labels.

In the response envelope:

- `requested_fields`: normalized requested names.
- `document.extracted_fields`: each requested name maps to an array of Field
  objects with values, confidence and evidence. Multiple distinct values remain;
  no match is an empty array.
- `document.other_fields`: additional discoveries, with their original field keys
  and the same evidence-bearing Field shape.
- `document.missing_fields`: requested names with no matching value.
- `document.fields`: the complete map, retained for existing consumers.

With no list, `extracted_fields` is empty and all discoveries appear in
`other_fields`. Bounced documents retain the requested names but have `document=null`.
Label-word matching supports `complainant name` → `name_of_complainant`; it does
not infer arbitrary synonyms. Offline semantic extraction limits still apply.

Submitting the same bytes and equivalent list reuses the latest result. Changing
the list reruns extraction, retaining the same job ID and appending result versions.
`is_new=false` means the job already existed, even when a new result was produced.
`GET /v1/jobs/{job_id}` returns the latest selection and groups. HTTP 409 means that
an extraction for the same document is still running; retry after it completes.


> Extraction update: open field discovery is now enabled by default. Offline
> results can include additional labeled fields beyond the six-field examples
> below. See [Open field extraction](OPEN-FIELD-EXTRACTION.md) for behavior,
> provenance, configuration, architecture and limits.


**Audience: a developer who has to call this service from their own system.** You do not
need to know how it works inside. This document tells you what to send, what comes back,
what every field means, and what to do when something goes wrong.

**What the service does, in one paragraph.** You POST one document — a scan, a photo or a
PDF of a banking complaint. It reads the page, decides whether the image is good enough to
work with, extracts structured fields (account number, IFSC code, amounts, dates, names,
the complaint text), and returns a JSON **result envelope**. Every extracted value carries
its own provenance: which mechanism produced it, how confident that mechanism was, which
page and which pixels it came from, and whether it passed its validity check. Nothing is
returned that cannot be pointed at on the page.

**Vocabulary used throughout:**

| Term | Meaning |
|---|---|
| **envelope** | The JSON result document. Its formal contract is `docs/schema/result_envelope.schema.json`. |
| **field** | One extracted value plus its provenance — see [§5](#5-the-field-object). |
| **evidence** | Page number, bounding box and character span identifying where a value physically is. |
| **bounce** | The service refused the document and says why in a machine-readable code. It is a normal HTTP 200 answer, not an error. |
| **escalation** | A page the first-tier reader was not confident about was sent to a heavier second-tier reader. |
| **flag** | A machine-readable marker on one field meaning "a human should look at this". |
| **deterministic field** | A value produced by a rule (regular expression, checksum, lookup table), never by a language model. |
| **SLM** | Small Language Model — used only for free-text fields, never for identifiers. |

---

## Contents

1. [Base URL, authentication and transport](#1-base-url-authentication-and-transport)
2. [Endpoints](#2-endpoints)
3. [A complete worked example](#3-a-complete-worked-example)
4. [The envelope, field by field](#4-the-envelope-field-by-field)
5. [The `Field` object](#5-the-field-object)
6. [`result_version` and `is_final` — polling versus webhooks](#6-result_version-and-is_final--polling-versus-webhooks)
7. [Webhooks: payload, headers and HMAC verification](#7-webhooks-payload-headers-and-hmac-verification)
8. [Every machine-readable code](#8-every-machine-readable-code)
9. [Idempotency and soft duplicates](#9-idempotency-and-soft-duplicates)
10. [Versioning policy](#10-versioning-policy)

---

## 1. Base URL, authentication and transport

### 1.1 Base URL

There is no hosted instance. The base URL is wherever your operations team deployed it. In
every example below, replace `http://127.0.0.1:8000` with yours.

The service also publishes machine-readable API documentation at:

- `GET /openapi.json` — the OpenAPI specification
- `GET /docs` — an interactive browser page generated from it

This file is the curated narrative version; where the two disagree, the OpenAPI document is
generated from the running code and wins on mechanics, while this file wins on *meaning*.

### 1.2 Authentication: there is none in v1

**The service performs no authentication and no authorisation.** Every endpoint is open to
anyone who can reach the port. This is deliberate, and it is a deployment requirement, not
an oversight:

> Deploy it behind your organisation's API gateway, and let the gateway do
> authentication, authorisation, TLS termination, rate limiting and CORS. Do not expose
> this service directly to any network wider than the one your gateway sits on.

The reasoning is recorded in `app/api/app.py`: adding a bearer-token scheme without knowing
which identity provider the deployment uses would create a second, weaker authentication
path that operators would have to work around, which is worse than having none behind a
gateway that already has one.

Two consequences for you as a consumer:

- **Send no credentials.** There is no `Authorization` header to set. If your gateway
  requires one, it is the gateway's, and the service never sees it.
- **`GET /metrics` is open too.** It exposes counters and gauges about the *service* and
  never about a document — no complainant data, no identifiers, no document ids. Restrict
  it at the gateway anyway, as you would any operational endpoint.

### 1.3 Transport

- Requests: `multipart/form-data` for uploads, nothing else.
- Responses: `application/json; charset=utf-8`, always, including errors.
- Encoding: UTF-8 throughout. Extracted values may contain Devanagari; do not assume ASCII.
- The service does **not** send CORS headers. A browser page can only call it if it is
  served from the same origin, or if your gateway adds the headers.

---

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/documents` | Submit one document; get the result envelope back. |
| `GET` | `/v1/jobs/{job_id}` | Re-read a job and its latest envelope. |
| `GET` | `/healthz` | Is the service up, which models are loaded, is the database reachable. |
| `GET` | `/metrics` | Prometheus text exposition. |
| `GET` | `/v1/stats` | Error rates and throughput since start-up, as JSON. |

### 2.1 `POST /v1/documents`

Submits one document. **Processing is synchronous**: the request returns when the document
has been processed, so a single call is the whole interaction for most callers. Typical
one-page latency is around 1.4 seconds (median), 3.9 seconds at the 95th percentile
(`docs/eval/REPORT-2026-08-08-slmlive.md` §10); the **first** request after the service starts takes about
10 seconds because the recognition models load lazily.

**Form fields:**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `file` | file | **yes** | The document. PDF, JPEG or PNG. Rejected above `API_MAX_UPLOAD_BYTES` (64 MB by default). |
| `synthetic` | boolean | conditional | Declares the payload to be generated test data. **Required under the `laptop` deployment profile; ignored under `cpu_node`.** See the note below. |
| `sender` | string | no | Who submitted it, typically an email address. Used only for soft-duplicate detection ([§9](#9-idempotency-and-soft-duplicates)). |
| `subject` | string | no | The mail subject. Same purpose. |

> **The `synthetic` field explained.** Under the development profile (`laptop`) the service
> may send page text to a language-model endpoint outside India. That is permitted only for
> generated test data, so the submitter must *positively declare* the payload as synthetic.
> There is no value of the field that lets an undeclared payload through, and there is no
> bypass flag. On a production `cpu_node` deployment nothing leaves the machine, the field
> is ignored, and you can omit it. **If you are calling a production deployment, you do not
> need this field.** If you get HTTP 422 `RESIDENCY_VIOLATION`, you are talking to a
> `laptop`-profile instance; add `synthetic=true` if — and only if — your data really is
> test data.

**curl:**

```bash
curl -X POST "http://127.0.0.1:8000/v1/documents" \
  -F "file=@complaint.jpg;type=image/jpeg" \
  -F "sender=complainant@example.in" \
  -F "subject=Unauthorised debit - CMS/2026/401687"
```

**Python (`requests`):**

```python
import requests

BASE = "http://127.0.0.1:8000"

with open("complaint.jpg", "rb") as fh:
    response = requests.post(
        f"{BASE}/v1/documents",
        files={"file": ("complaint.jpg", fh, "image/jpeg")},
        data={
            "sender": "complainant@example.in",
            "subject": "Unauthorised debit - CMS/2026/401687",
            # "synthetic": "true",   # only needed against a laptop-profile instance
        },
        timeout=120,
    )

response.raise_for_status()
body = response.json()

print(body["job_id"], body["status"], body["is_new"])
envelope = body["result"]
```

**Response body (`SubmissionResponse`):**

| Field | Type | Meaning |
|---|---|---|
| `job_id` | string | Opaque id for this processing job. Random; use it with `GET /v1/jobs/{job_id}`. |
| `document_id` | string | `sha256:<hex>` of the uploaded bytes. Stable: the same file always produces the same value. |
| `is_new` | boolean | `false` means this exact file was submitted before and nothing was reprocessed — see [§9](#9-idempotency-and-soft-duplicates). |
| `status` | string | `queued`, `running`, `completed`, `bounced`, or `failed`. |
| `soft_duplicate_of` | string \| null | `job_id` of an earlier job that probably repeats this complaint. Advisory. |
| `soft_duplicate_reason` | string \| null | `"same_sender_subject_day"`, or `null`. |
| `result` | object \| null | The envelope. `null` only while a queued job has produced none. |

### 2.2 `GET /v1/jobs/{job_id}`

Re-reads a job. Use it to poll, to fetch a later `result_version`, or to look at something
you processed yesterday.

**curl:**

```bash
curl "http://127.0.0.1:8000/v1/jobs/f54de5494ce74189b1b5053a82002b9b"
```

**Python:**

```python
job = requests.get(f"{BASE}/v1/jobs/{job_id}", timeout=30).json()
print(job["status"], job["result_version"], job["is_final"], job["versions"])
```

**Response body (`JobResponse`), real example:**

```json
{
  "job_id": "f54de5494ce74189b1b5053a82002b9b",
  "document_id": "sha256:a2b81e39249e9911cb64617b1a12433203933f6f1004efca341c53ac01f9ef7d",
  "status": "completed",
  "result_version": 1,
  "is_final": true,
  "created_at": "2026-08-07T21:21:36.429863",
  "updated_at": "2026-08-07T21:21:47.772537",
  "soft_duplicate_of": null,
  "soft_duplicate_reason": null,
  "error_code": null,
  "versions": [1],
  "result": { "...": "the envelope, see §3" }
}
```

| Field | Meaning |
|---|---|
| `result_version` | Highest version published so far. |
| `is_final` | `true` when no further version will be published for this job. |
| `versions` | Every version published, oldest first. `[1]` normally; `[1, 2]` when a page escalated. |
| `error_code` | Machine-readable code when `status` is `failed`; `null` otherwise. |
| `created_at` / `updated_at` | Job timestamps. **These are UTC but carry no timezone suffix** — unlike the envelope's own `created_at`, which ends in `Z`. Treat them as UTC. |
| `result` | The latest envelope. |

**404** with `{"code": "JOB_NOT_FOUND", ...}` when the id is unknown.

### 2.3 `GET /healthz`

**curl:**

```bash
curl "http://127.0.0.1:8000/healthz"
```

**Real response:**

```json
{
  "status": "ok",
  "profile": "laptop",
  "synthetic_only": true,
  "banner": "profile=laptop synthetic_only=true slm=google/gemma-3-4b-it hf_token=set threads=2x2",
  "provider_profile_gauge": {"laptop": 1.0, "cpu_node": 0.0, "gpu_node": 0.0},
  "models": [
    {"name": "ocr_detection",          "path": "...\\models\\ocr\\det.onnx",           "present": true},
    {"name": "ocr_rec_latin",          "path": "...\\models\\ocr\\rec_latn.onnx",      "present": true},
    {"name": "ocr_rec_devanagari",     "path": "...\\models\\ocr\\rec_deva.onnx",      "present": true},
    {"name": "language_id",            "path": "...\\models\\lid\\lid.176.ftz",        "present": true},
    {"name": "translation",            "path": "...\\models\\mt\\indic-en\\model.bin", "present": true},
    {"name": "page_classifier_fusion", "path": "...\\models\\classifier\\fusion.json", "present": true}
  ],
  "database": "ok",
  "webhook_enabled": false,
  "schema_version": "1.0.0"
}
```

`status` is `"ok"` or `"degraded"` — never `"unhealthy"`. **`degraded` means a model file or
the database is missing, not that the service is down**: it can still accept uploads, refuse
bad ones and run validators without weights, so reporting "down" would make an orchestrator
restart a container whose real problem is an unmounted volume. As a consumer, treat
`degraded` as "results will be incomplete; alert an operator", not as "stop sending".

`schema_version` here is the envelope schema this instance emits — check it against what
your parser expects ([§10](#10-versioning-policy)).

### 2.4 `GET /metrics`

Prometheus text exposition over the default registry: document counters, per-version
counters, provider gauges, webhook delivery counters and latency histograms. Every series
describes the service, never a document — a metric carrying a document identifier would be
a bug in the metric.

```bash
curl "http://127.0.0.1:8000/metrics"
```

```
# HELP provider_profile Active provider profile.
# TYPE provider_profile gauge
provider_profile{profile="laptop"} 1.0
provider_profile{profile="cpu_node"} 0.0
provider_profile{profile="gpu_node"} 0.0
...
```

Request-level series were added alongside the per-stage ones, because until then nothing
counted *requests* and "what is the error rate?" had no answer that did not involve reading
logs:

| Series | Type | Labels | Meaning |
|---|---|---|---|
| `http_requests_total` | counter | `method`, `route`, `status` | Every response served. |
| `http_unhandled_exceptions_total` | counter | `route` | Requests that raised past every handler — nobody *chose* that 500. |
| `http_request_seconds` | histogram | `route` | Per-request latency. |

`route` is the route **template** (`/v1/jobs/{job_id}`), never the resolved path. A job id
in a label would put a document identifier into `/metrics`, and would grow one time series
per document besides. Static assets are reported under their mount (`/ui`), and
`<unmatched>` means exactly one thing: a request for a path this service does not serve.
Keeping those three apart matters — a browser fetching a favicon is not a caller hammering
a wrong URL, and lumping them together made an idle service report a 64% error rate.

The service error rate is one expression over that counter:

```
sum(rate(http_requests_total{status=~"[45].."}[5m])) / sum(rate(http_requests_total[5m]))
```

### 2.5 `GET /v1/stats`

The same in-memory counters, arranged for a person rather than a scraper. Use `/metrics`
for Prometheus and this when the question is "is it healthy right now?".

```bash
curl -s "http://127.0.0.1:8000/v1/stats"
```

```json
{
  "uptime_seconds": 1893.4,
  "requests": 412,
  "client_errors": 7,
  "server_errors": 0,
  "unhandled_exceptions": 0,
  "error_rate": 0.016990,
  "routes": [
    {"route": "/v1/documents", "requests": 268, "client_errors": 5, "server_errors": 0, "error_rate": 0.018657},
    {"route": "/v1/jobs/{job_id}", "requests": 140, "client_errors": 2, "server_errors": 0, "error_rate": 0.014286}
  ],
  "documents": {"accepted": 261, "rejected": 7},
  "bounces": {"BLURRED": 4, "RESOLUTION_BELOW_FLOOR": 2},
  "provider_calls": {"ok": 254, "rate_limited": 3}
}
```

**Everything here is in memory and resets when the process restarts.** Nothing is written
to disk, so these are "since boot" figures and never a historical record — which is why
`uptime_seconds` ships with them. For retention, scrape `/metrics`.

**`documents` and the HTTP counters answer different questions, and you want both.** A
document the quality gate refuses is a *product* outcome: it comes back as `200` with a
bounce envelope and a machine-readable reason, so it correctly does **not** raise the HTTP
error rate. `documents.rejected` and `bounces` are where it shows up. A rising
`bounces` with a flat `error_rate` means the mail stream got worse, not the service.

---

## 3. A complete worked example

A real submission of a one-page English complaint form, captured from a running instance.
Three of the seven extracted fields are shown in full; the rest are elided with `...` and
are identical in shape.

```json
{
  "schema_version": "1.0.0",
  "result_version": 1,
  "is_final": true,
  "document_id": "sha256:a2b81e39249e9911cb64617b1a12433203933f6f1004efca341c53ac01f9ef7d",
  "job_id": "f54de5494ce74189b1b5053a82002b9b",
  "created_at": "2026-08-07T21:21:47.772537Z",
  "provider_profile": "laptop",
  "bounce_decision": { "decision": "accept", "reason": null, "detail": null },
  "escalation_flag": false,
  "pages": [
    {
      "page_no": 1,
      "quality_score": 1.0,
      "quality_flags": [],
      "quality_verdict": "accept",
      "ocr_tier": "tier1",
      "line_count": 25,
      "primary_script": "Latn",
      "primary_language": "eng",
      "language_distribution": { "eng": 1.0 },
      "is_code_mixed": false,
      "escalated": false,
      "shadow_sampled": false,
      "repairs_applied": []
    }
  ],
  "document": {
    "document_class": "complaint_form",
    "document_class_confidence": 1.0,
    "fields": {
      "bank_name": {
        "value": "Bank of Baroda",
        "value_en": null,
        "normalized": "Bank of Baroda",
        "confidence": 0.9734663452420916,
        "source": "lookup_table",
        "validator_passed": true,
        "evidence": {
          "page_no": 1,
          "bbox": [710.3, 527.7, 916.6, 527.7, 916.6, 573.6, 710.3, 573.6],
          "span": { "line_index": 9, "start": 0, "end": 14 }
        },
        "mt_engine": null,
        "flags": []
      },
      "ifsc_code": {
        "value": "BARB0522983",
        "value_en": null,
        "normalized": "BARB0522983",
        "confidence": 0.899778361753984,
        "source": "regex_validator",
        "validator_passed": true,
        "evidence": {
          "page_no": 1,
          "bbox": [709.0, 601.9, 903.2, 601.9, 903.2, 652.9, 709.0, 652.9],
          "span": { "line_index": 11, "start": 0, "end": 11 }
        },
        "mt_engine": null,
        "flags": []
      },
      "prior_complaint_reference": {
        "value": "CMS/2025/705869",
        "value_en": null,
        "normalized": null,
        "confidence": 0.99,
        "source": "slm",
        "validator_passed": null,
        "evidence": {
          "page_no": 1,
          "bbox": [707.6, 995.3, 951.3, 995.3, 951.3, 1041.6, 707.6, 1041.6],
          "span": { "line_index": 21, "start": 0, "end": 15 }
        },
        "mt_engine": null,
        "flags": []
      },
      "account_number": { "...": "source: regex_validator" },
      "amount":         { "...": "source: regex_validator" },
      "transaction_date": { "...": "source: regex_validator" },
      "utr_rrn":        { "...": "source: regex_validator" }
    },
    "complaint_summary": {
      "text": "1. Name of complainant\n2 Address\n3. Mobile\n...",
      "spans": [ { "evidence": { "...": "..." }, "text": "1. Name of complainant" } ],
      "separator": "\n"
    },
    "supporting_doc_tags": [],
    "page_count": 1,
    "primary_script": "Latn",
    "primary_language": "eng",
    "language_distribution": { "eng": 1.0 },
    "is_code_mixed": false
  },
  "processing": {
    "started_at": "2026-08-07T21:21:36.478621Z",
    "finished_at": "2026-08-07T21:21:47.772537Z",
    "duration_ms": 11293.916,
    "pages_processed": 1,
    "pages_bounced": 0,
    "pages_escalated": 0,
    "tokens_used": 0,
    "stage_timings_ms": { "s0_read": 0.01, "s1_ingest": 54.084, "s15_quality": 213.187,
                          "s4_ocr": 836.81, "s2b_lid": 55.19, "s3_classify": 284.329,
                          "s5_validators": 5.629, "s5_slm": 12098.903 }
  },
  "warnings": []
}
```

### 3.1 A bounced document

When the service refuses a document, you still get **HTTP 200** and a valid envelope. It
carries a machine-readable reason and **no** extracted content. Real example, a heavily
blurred and skewed bank statement:

```json
{
  "schema_version": "1.0.0",
  "result_version": 1,
  "is_final": true,
  "document_id": "sha256:005c107e92510d9fdf5ae2345dcda4dd75da4e38a35982217169b1d5a381b3bf",
  "job_id": "1a1a1f28d59c4036b3240176d64529b1",
  "created_at": "2026-08-07T21:18:47.269610Z",
  "provider_profile": "laptop",
  "bounce_decision": {
    "decision": "bounce",
    "reason": "QUALITY_SCORE_BELOW_THRESHOLD",
    "detail": "quality score 0.21 is below the 0.35 threshold"
  },
  "escalation_flag": false,
  "pages": [
    {
      "page_no": 1,
      "quality_score": 0.20594120363946583,
      "quality_flags": ["SKEW_DETECTED", "SKEW_REPAIRED", "BLUR_ABOVE_CEILING", "SHADOW_DETECTED"],
      "quality_verdict": "bounce",
      "ocr_tier": "none",
      "line_count": 0,
      "primary_script": null,
      "primary_language": null,
      "language_distribution": {},
      "is_code_mixed": false,
      "escalated": false,
      "shadow_sampled": false,
      "repairs_applied": ["deskew"]
    }
  ],
  "document": null,
  "processing": {
    "started_at": "2026-08-07T21:18:47.100705Z",
    "finished_at": "2026-08-07T21:18:47.269610Z",
    "duration_ms": 168.905,
    "pages_processed": 1, "pages_bounced": 1, "pages_escalated": 0, "tokens_used": 0,
    "stage_timings_ms": {"s0_read": 0.006, "s1_ingest": 10.297, "s15_quality": 158.375}
  },
  "warnings": []
}
```

**The rule to code against:** `document` is `null` **if and only if**
`bounce_decision.decision == "bounce"`. A bounced document never carries extracted content,
and an accepted one always carries a `document` object. Branch on `bounce_decision.decision`,
not on whether `document` happens to be present.

**Bad input bounces too, it does not error.** A zero-byte file, a truncated PDF, an HTML
file renamed to `.pdf`, a 60×60-pixel image and a blank page were all submitted to a
running instance; every one returned **HTTP 200** with `status: "bounced"` and these
reasons:

| Submitted | `bounce_decision.reason` | `detail` |
|---|---|---|
| zero-byte `.pdf` | `EMPTY_FILE` | `attachment is zero bytes` |
| truncated `.pdf` | `MALFORMED_PDF` | `PDF could not be parsed` |
| HTML renamed `.pdf` | `MIME_MISMATCH` | `content does not match any supported type (PDF, JPEG, PNG)` |
| GIF renamed `.pdf` | `MIME_MISMATCH` | same |
| 60×60 px PNG | `RESOLUTION_BELOW_FLOOR` | `page is 60x60 px; the floor is 200 px on the short edge` |
| blank page PNG | `BLANK_PAGE` | `page carries no ink; OCR was not billed` |

Only genuine *service* faults produce non-2xx responses — see [§8.1](#81-http-error-codes).

---

## 4. The envelope, field by field

Formal contract: `docs/schema/result_envelope.schema.json`. This section is that schema in
plain language. Required properties are marked **R**; everything else has the default shown.

### 4.1 Top level

| Field | Type | | Meaning |
|---|---|---|---|
| `schema_version` | string | `"1.0.0"` | Envelope contract version. See [§10](#10-versioning-policy). |
| `result_version` | integer ≥ 1 | **R** | Which publication of this job's result you are looking at. See [§6](#6-result_version-and-is_final--polling-versus-webhooks). |
| `is_final` | boolean | **R** | `true` when no further version will follow. |
| `document_id` | string | **R** | `sha256:<hex>` of the submitted bytes. |
| `job_id` | string \| null | `null` | The job this result belongs to. |
| `created_at` | timestamp | *(now)* | When this version was produced. Timezone-aware, UTC, ends in `Z`. |
| `provider_profile` | string | **R** | `laptop`, `cpu_node` or `gpu_node` — which deployment shape produced it. Recorded so a difference between environments is attributable. |
| `bounce_decision` | object | **R** | Accept / repair / bounce plus a machine-readable reason. §4.2. |
| `escalation_flag` | boolean | `false` | `true` when a human should look at this document — either a page escalated or a field is flagged. **This is the single boolean to route on if you only read one thing.** |
| `pages` | array | `[]` | One record per page, ascending by `page_no`. §4.3. |
| `document` | object \| null | `null` | The extracted content. `null` exactly when bounced. §4.4. |
| `processing` | object | **R** | Counters and timings. §4.6. |
| `warnings` | array of string | `[]` | Machine-readable notes about what did *not* happen. §8.4. |

### 4.2 `bounce_decision`

| Field | Type | Meaning |
|---|---|---|
| `decision` | `"accept"` \| `"repair"` \| `"bounce"` | `accept` = usable as submitted. `repair` = usable after the service straightened/cleaned it. `bounce` = refused. |
| `reason` | `BounceReason` \| null | Present **exactly when** `decision == "bounce"`; `null` otherwise. Full list in [§8.2](#82-bounce-reasons). |
| `detail` | string \| null | Human-readable elaboration. **Never parse this** — it is for display and logs. |

`repair` is not a warning. It means the service fixed the image and read it successfully;
the repairs are listed per page in `pages[].repairs_applied`.

### 4.3 `pages[]`

| Field | Type | | Meaning |
|---|---|---|---|
| `page_no` | integer ≥ 1 | **R** | 1-based page number. |
| `quality_score` | 0.0–1.0 | **R** | Overall image quality. Present on **every** page including accepted ones, so you can watch it drift. |
| `quality_flags` | array | `[]` | What the quality gate observed. Flags ending `_REPAIRED` record what was *fixed*. Full list in [§8.3](#83-quality-flags). |
| `quality_verdict` | `accept`/`repair`/`bounce` | **R** | This page's own verdict. |
| `ocr_tier` | `bypass`/`tier1`/`tier2`/`none` | **R** | How the text was obtained: `bypass` = a usable text layer already existed in the PDF; `tier1` = the fast reader; `tier2` = the heavier escalation reader; `none` = not read (bounced). |
| `line_count` | integer ≥ 0 | **R** | Text lines recognised. Always `0` when `quality_verdict` is `bounce` or `ocr_tier` is `none`. |
| `primary_script` | string \| null | `null` | ISO 15924 script code, e.g. `Latn`, `Deva`. **Derived** from the page's lines. |
| `primary_language` | string \| null | `null` | ISO 639-3 code, e.g. `eng`, `hin`, `mar`. Guaranteed to be the argmax of `language_distribution`. |
| `language_distribution` | object | `{}` | ISO 639-3 code → share of lines. Values sum to ≈ 1 when non-empty. |
| `is_code_mixed` | boolean | `false` | `true` when at least 15 % of lines are in a different script from the page majority. |
| `escalated` | boolean | `false` | This page was routed to the second-tier reader because the first tier was not trusted. |
| `shadow_sampled` | boolean | `false` | This page *also* went to the second tier as part of a 3 % random quality audit. **Not** an escalation; do not count it as one. |
| `repairs_applied` | array of string | `[]` | e.g. `["deskew"]`, `["auto_invert"]`. |

> **`is_code_mixed` and per-line `script` (Q14 — fixed).** Each line is labelled with the
> script of **its own characters**, not with the name of the recognition model that read the
> page, so a page that genuinely mixes scripts now reports both and `is_code_mixed` can
> become `true`. A line that casts no script vote at all — an account number, an amount —
> inherits the page's dominant script.
>
> Previously every line carried the reading head's script, so no page could ever report more
> than one and `is_code_mixed` was always `false` (measured: 0 of 147 pages against 30 that
> genuinely mix). That labelling was also load-bearing for the wrong reason: language
> identification is constrained by `script`, so an English line labelled `Deva` could never
> be identified as English.

> **Evidence coordinates on repaired pages (Q11 — fixed).** Every `bbox` is in the
> coordinate frame of the image **you uploaded**, including on pages where
> `repairs_applied` contains `deskew`. The page is still straightened before it is read —
> that is what makes it readable — but the boxes are mapped back through the inverse of the
> straightening transform before they are published, so drawing them on the original file
> is correct.
>
> Previously these were published in the straightened frame, which also *expands* the
> canvas: a 0.9° repair on A4 shifted every box by about 18 px as well as rotating it, and
> the visible symptom was evidence boxes sitting on the wrong field.

### 4.4 `document`

`null` exactly when bounced. Otherwise:

| Field | Type | | Meaning |
|---|---|---|---|
| `document_class` | enum | **R** | `complaint_form`, `complaint_letter`, `supporting_doc`, `photo_evidence`, `unknown`. |
| `document_class_confidence` | 0.0–1.0 | **R** | Classifier confidence. |
| `fields` | object | `{}` | Field name → `Field` object. §5. |
| `complaint_summary` | object \| null | `null` | §4.5. |
| `supporting_doc_tags` | array of string | `[]` | Closed tag set for supporting documents. No field extraction happens on those. |
| `page_count` | integer ≥ 0 | **R** | Always equals `len(pages)`. |
| `primary_script` | string \| null | `null` | Document-level derived summary. |
| `primary_language` | string \| null | `null` | Document-level derived summary. |
| `language_distribution` | object | `{}` | Document-level derived summary. |
| `is_code_mixed` | boolean | `false` | Same limitation as §4.3. |

### 4.5 `complaint_summary`

**This is not a written summary.** The model selects *line indices*; the service
concatenates the verbatim recognised text of those lines. Prose the model invented cannot
be represented — the envelope re-derives `text` from `spans` and rejects any mismatch.

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Exactly `separator.join(span.text for span in spans)`. Verified, not asserted. |
| `spans` | array (≥ 1) | Each `{ evidence, text }`, where `text` is the verbatim recognised text of the referenced span. |
| `separator` | string | `"\n"` by default. |

You can therefore safely highlight the summary on the original page: every character in it
came from a span you have coordinates for.

### 4.6 `processing`

| Field | Type | Meaning |
|---|---|---|
| `started_at` / `finished_at` | timestamp \| null | Wall-clock bounds of this version's production. |
| `duration_ms` | number \| null | End to end for this version. |
| `pages_processed` / `pages_bounced` / `pages_escalated` | integer | Counts. |
| `tokens_used` | integer | Language-model tokens attributed to this job. **Known to under-report:** the orchestrator and the model client keep separate budgets, so this reads `0` on envelopes from the current build even when tokens were spent. Do not bill on it. Tracked as **Q9**. |
| `stage_timings_ms` | object | Per-stage wall clock, e.g. `s4_ocr`, `s5_slm`. Diagnostic; stage names are not part of the stable contract. |

---

## 5. The `Field` object

Every entry in `document.fields` has this shape. **`evidence` is mandatory** — a value this
service cannot point at on the page cannot be represented at all.

| Field | Type | | Meaning |
|---|---|---|---|
| `value` | string \| null | **R** | The value **exactly as it appears on the page**, in its own script. **Immutable and authoritative.** Key your business logic on this. |
| `value_en` | string \| null | `null` | An English rendering, produced by machine translation. **Advisory only.** Never key business logic on it. |
| `normalized` | string \| null | `null` | A canonical form for indexing and matching — digits-only account number, `2026-07-07` for a date, `45000.00` for an amount. Use this for lookups and joins. |
| `confidence` | 0.0–1.0 | **R** | How confident the producing mechanism was. |
| `source` | enum | **R** | Which mechanism produced it: `regex_validator`, `lookup_table`, `slm`, `classifier`, `human`. |
| `validator_passed` | boolean \| null | `null` | `true` = passed its check (e.g. IFSC format plus master-list lookup). `false` = **failed and was kept anyway, flagged**. `null` = no validator applies to this field. |
| `evidence` | object | **R** | `{ page_no, bbox, span }`. §5.1. |
| `mt_engine` | string \| null | `null` | Which translation engine produced `value_en`. Present **exactly when** `value_en` is present. |
| `flags` | array | `[]` | Why a human should look at this field. Empty means no concern. [§8.5](#85-field-flags). |

**Five rules the service guarantees, which you can rely on:**

1. **`value` is never corrected.** A failed checksum sets `validator_passed: false` and adds
   `VALIDATOR_FAILED` to `flags`. The service never silently "fixes" a value.
2. **`value` is never a translation.** Translation only ever populates `value_en`, and
   `value_en` cannot exist without `mt_engine`.
3. **Identifiers never come from a language model.** `account_number`, `ifsc_code`,
   `utr_rrn`, `micr_code`, `card_fragment`, `amount`, `transaction_date`, `complaint_date`
   and `bank_name` are produced by rules and lookups. `source: "slm"` on any of them is
   structurally impossible — the envelope refuses to validate.
4. **Language-model values are grounded.** Before a value from the model is accepted it must
   appear in the page's own recognised text within 2 character edits (script-aware, Indic
   digits normalised first). If it does not, the value is dropped and flagged
   `GROUNDING_REJECTED`; it does not reach you as a plausible-looking invention.
5. **Anything needing attention is flagged.** `field.flags` non-empty ⇒ a human should
   check it. The document-level roll-up of this is `escalation_flag`.

### 5.1 `evidence`

| Field | Type | Meaning |
|---|---|---|
| `page_no` | integer ≥ 1 | Which page. |
| `bbox` | array of 8 numbers | Quadrilateral, `[x1,y1, x2,y2, x3,y3, x4,y4]`, clockwise from top-left, in pixels of the page raster. It is a quadrilateral rather than a rectangle because text on a photographed page is not axis-aligned. |
| `span` | object | `{ line_index, start, end }` — which recognised line, and the half-open character range within it. |

Read the coordinate caveat in §4.3 before drawing these on an uploaded image.

### 5.2 Translation in practice

When a value is in an Indic script the service may attach an English rendering. Real
examples from the evaluation run (`docs/eval/REPORT-2026-08-08-slmlive.md` §5), all with
`mt_engine: "indictrans2-indic-en-dist-200M-ct2-int8"`:

| `value` | `value_en` |
|---|---|
| राजेश कुमार शर्मा | Rajesh Kumar Sharma |
| पनवेल महाराष्ट् | Panvel Maharashtra |
| कृपया योग्य ती कार्यवाही करावी ही विनती | Please take the appropriate action |

If translation is attempted and the result cannot be trusted, the translation is **discarded**
and the field is flagged `TRANSLATION_UNAVAILABLE` — you get the native value with no
`value_en`, never a partial or corrupted one.

---

## 6. `result_version` and `is_final` — polling versus webhooks

### 6.1 The semantics

A job publishes **one or more** envelopes, numbered from 1:

- **`result_version: 1`** is published as soon as the fast path finishes. If no page needed
  escalating, it is also the last, and `is_final` is `true`.
- If one or more pages were escalated to the second-tier reader, version 1 is published with
  **`is_final: false`** and a `warnings` entry naming each pending page
  (`escalation_pending:page=3`). When the escalated pages land, the document is re-assembled
  and published as **`result_version: 2, is_final: true`**.

Guarantees:

- `result_version` starts at 1 and increases by exactly 1 per publication.
- Exactly one version per job has `is_final: true`, and it is the highest.
- **You may act on version 1.** Everything in it is real; the later version can only add or
  refine content from escalated pages. `warnings` tells you which pages are still moving.
- A version, once published, is never rewritten.

### 6.2 Which integration style to choose

| | Polling | Webhook |
|---|---|---|
| **How** | `POST /v1/documents` returns the envelope synchronously; `GET /v1/jobs/{id}` for later versions. | Configure `WEBHOOK_URL`; every version is POSTed to you as it is published. |
| **Use when** | You submit one document and want the answer in the same call. This is the normal case and covers most callers. | You need to know about version 2 without asking, or you submit in bulk. |
| **Cost** | If `is_final` is `false`, you must re-`GET` until it is `true`. | You must run an HTTPS endpoint and verify signatures. |

**Recommended polling loop** (only needed when `is_final` is `false`):

```python
import time
import requests

def wait_for_final(base: str, job_id: str, *, timeout_s: float = 300.0) -> dict:
    """Poll until the job publishes its final version. Returns the envelope."""
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    while True:
        job = requests.get(f"{base}/v1/jobs/{job_id}", timeout=30).json()
        if job["is_final"] or job["status"] in ("bounced", "failed"):
            return job["result"]
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} not final after {timeout_s}s")
        time.sleep(delay)
        delay = min(delay * 2, 15.0)      # back off; do not hammer
```

Do not poll faster than once a second, and always back off. A job whose `status` is
`bounced` or `failed` will never become final in the usual sense — check status first, as
above.

---

## 7. Webhooks: payload, headers and HMAC verification

### 7.1 What arrives

One HTTP `POST` per published `result_version`, to the URL the deployment configured.

- **Body:** the envelope itself, exactly as in §3 — not wrapped in any extra object.
  `Content-Type: application/json; charset=utf-8`. UTF-8, unescaped, so Devanagari is
  readable in your logs.
- **Bytes are canonical:** keys sorted, no whitespace padding. The same version signed twice
  produces the same bytes and therefore the same signature, so you can deduplicate on it.

**Headers, from a real delivery:**

```
content-type:           application/json; charset=utf-8
x-rbi-signature:        sha256=0e3f871ee29ce68d4e7a6edcc09bb7b0053401e856bc913e6d76ce8b7f4d7d4c
x-rbi-timestamp:        1786137805
x-rbi-job-id:           ea7f81681afd4a8a848a21fac7686ea0
x-rbi-result-version:   1
x-rbi-delivery-id:      1
```

| Header | Meaning |
|---|---|
| `X-RBI-Signature` | `sha256=<hex>` — the HMAC. §7.2. |
| `X-RBI-Timestamp` | Unix seconds when the payload was signed. **Part of the signed message**, not merely alongside it. |
| `X-RBI-Job-Id` | Convenience copy of the job id. Also in the body. |
| `X-RBI-Result-Version` | Convenience copy of the version. Also in the body. |
| `X-RBI-Delivery-Id` | The sender's row id for this attempt. Useful when asking an operator to replay one. |

**Your endpoint must answer 2xx.** Anything else — including 4xx — is treated as a failure
and retried with exponential backoff (5 attempts by default, doubling from 1 s to a 60 s
cap, with jitter). After the last attempt the delivery is parked in a dead-letter queue and
an operator can replay it. **A failed delivery never loses the result**: the envelope is
stored and `GET /v1/jobs/{id}` still serves it. A webhook is a notification, not the result.

**Redelivery is possible, so be idempotent.** Deduplicate on the pair
`(job_id, result_version)`. The service documents this obligation rather than enforcing it.

### 7.2 How the signature works

```
signature = HMAC-SHA256( key   = <shared secret>,
                         data  = b"<timestamp>." + <exact body bytes> )
```

lowercase hex, sent as `X-RBI-Signature: sha256=<hex>`.

Three properties worth understanding:

1. **The timestamp is inside the MAC.** A captured delivery cannot be replayed later with a
   fresh timestamp header — the MAC would not match. Reject anything older than your
   tolerance (the sender's default assumption is 300 seconds).
2. **Sign the raw bytes you received.** Do not parse the JSON and re-serialise it before
   verifying; key order or float formatting will differ and the MAC will fail. Read the
   body as bytes.
3. **Compare in constant time.** A plain `==` on a hex digest leaks the position of the
   first wrong byte to a patient attacker.

### 7.3 Verification — Python, ready to paste

This exact function was run against a real delivery from a live instance: it returned
`True` for the genuine payload, and `False` for a wrong secret, a body with one byte
appended, and a stale timestamp.

```python
import hashlib
import hmac
import time


def verify_rbi_webhook(
    secret: str,
    body: bytes,
    headers: dict[str, str],
    tolerance_seconds: int = 300,
) -> bool:
    """True when this delivery genuinely came from the RBI OCR service.

    Args:
        secret:  the shared secret, the same value the sender has in
                 WEBHOOK_HMAC_SECRET.
        body:    the RAW request body bytes. Do not parse and re-serialise.
        headers: request headers, keys lower-cased.
    """
    signature = headers.get("x-rbi-signature", "")
    timestamp = headers.get("x-rbi-timestamp", "")

    if not signature.startswith("sha256=") or not timestamp.isdigit():
        return False

    # Reject replays of an old, captured delivery.
    if abs(int(time.time()) - int(timestamp)) > tolerance_seconds:
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + body,
        hashlib.sha256,
    ).hexdigest()

    # Constant time: a plain == leaks where the first byte differs.
    return hmac.compare_digest(signature[len("sha256="):], expected)
```

A minimal receiver, end to end:

```python
import json

from flask import Flask, request      # any framework; Flask shown for brevity

app = Flask(__name__)
SECRET = "...the shared secret..."
seen: set[tuple[str, int]] = set()    # use a database in production

@app.post("/hooks/rbi-ocr")
def receive():
    body = request.get_data()         # RAW bytes, before any parsing
    headers = {k.lower(): v for k, v in request.headers.items()}

    if not verify_rbi_webhook(SECRET, body, headers):
        return "", 401                # do NOT process an unverified payload

    envelope = json.loads(body.decode("utf-8"))
    key = (envelope["job_id"], envelope["result_version"])
    if key in seen:
        return "", 200                # already handled; still answer 2xx
    seen.add(key)

    handle(envelope)                  # your logic
    return "", 200                    # 2xx, or the sender will retry
```

### 7.4 Verification — language-neutral pseudo-code

```
on POST /your/webhook/path:
    raw_body   := read the entire request body AS BYTES, unparsed
    sig_header := header "X-RBI-Signature"          # e.g. "sha256=ab12..."
    ts_header  := header "X-RBI-Timestamp"          # e.g. "1786137805"

    if sig_header does not start with "sha256=":  respond 401; stop
    if ts_header is not all digits:               respond 401; stop

    age := absolute_value(current_unix_seconds() - integer(ts_header))
    if age > 300:                                 respond 401; stop   # replay

    message  := bytes_of(ts_header) + bytes_of(".") + raw_body
    expected := lowercase_hex( HMAC_SHA256(key = shared_secret, data = message) )
    provided := sig_header without the leading "sha256="

    if NOT constant_time_equals(provided, expected):
        respond 401; stop                          # forged or tampered

    envelope := parse_json(raw_body)
    key      := (envelope.job_id, envelope.result_version)
    if key already processed:  respond 200; stop   # idempotent redelivery
    mark key processed
    process(envelope)
    respond 200                                    # anything else = retry
```

---

## 8. Every machine-readable code

Five separate vocabularies. Each is a closed set; a value outside it is a bug.

### 8.1 HTTP error codes

Every non-2xx response has the same body shape:

```json
{ "code": "MACHINE_READABLE_CODE", "message": "human sentence", "context": { } }
```

**Branch on `code`. Never parse `message`.**

| HTTP | `code` | What happened | What you should do |
|---|---|---|---|
| 404 | `JOB_NOT_FOUND` | No job with that id. | Check the id. It is not a retry situation. |
| 413 | `UPLOAD_TOO_LARGE` | The upload exceeds `API_MAX_UPLOAD_BYTES` (64 MB default). `context.limit` gives the ceiling. | Split or compress the document. Do not retry unchanged. |
| 422 | `RESIDENCY_VIOLATION` | You submitted to a `laptop`-profile instance without `synthetic=true`. | Add the field **only** if the data really is test data; otherwise use a production endpoint. |
| 422 | `MALWARE_DETECTED` | The scanner refused the attachment. The response deliberately names **no** signature — that would be a free oracle for tuning the next attempt. | Do not retry. Escalate to security. |
| 422 | *(FastAPI validation)* | A required form field is missing or malformed. | Fix the request. |
| 429 | `PROVIDER_RATE_LIMITED` | An upstream model endpoint rate-limited us. `context.retry_after` when known. | Retry after the given delay. |
| 429 | `TOKEN_BUDGET_EXCEEDED` | The per-job language-model token budget was exhausted. | Operational; report it. Retrying the same document will hit the same ceiling. |
| 501 | `PROVIDER_METHOD_NOT_SUPPORTED` | The deployment has no component able to do that piece of work at all — e.g. no language model is configured. Not transient. | Not retryable. Usually surfaces as a `warning` instead (§8.4), because the pipeline absorbs it and returns the deterministic fields. |
| 500 | `CONFIGURATION_ERROR` | The service is misconfigured. | Not retryable. Alert an operator. |
| 500 | `STORAGE_ERROR` | A database operation failed. | Retryable after a delay. Alert an operator. |
| 502 | `PROVIDER_ERROR` | A model provider failed in a way the pipeline did not absorb. | Retryable. |
| 502 | `PROVIDER_RESPONSE_INVALID` | A model returned output that failed validation twice. | Retryable, but usually reports itself as a `warning` instead — see §8.4. |
| 503 | `PROVIDER_UNAVAILABLE` | A model endpoint was unreachable. | Retryable with backoff. |
| 503 | `MODEL_ARTIFACT_MISSING` | A local weights file is not on the server. The message names the exact path. | Not retryable. Alert an operator; check `GET /healthz`. |
| 409 | `APPEND_ONLY_VIOLATION` | Something tried to modify an append-only audit row. | Not retryable. This is a service bug; report it. |

> **Ingestion problems are NOT in this table.** An empty file, a corrupt PDF, a wrong
> content type, an image that is too small, a blank page, too many pages, a decompression
> bomb, an encrypted PDF or a document that ran out of time all return **HTTP 200 with a
> bounce envelope**, because they are answers about the *document*, not failures of the
> *service*. See §8.2.

### 8.2 Bounce reasons

`bounce_decision.reason`, present exactly when `decision == "bounce"`. These are the codes
an automatic reply to the complainant is keyed on, so they are phrased in terms of the
document, not the subsystem.

| Code | Meaning | Consumer action |
|---|---|---|
| `EMPTY_FILE` | Zero-byte attachment. | Ask the sender to resend. |
| `MALFORMED_PDF` | The PDF could not be parsed. | Ask the sender to resend, ideally as an image. |
| `MIME_MISMATCH` | The file's actual content is not PDF, JPEG or PNG, whatever it is named. | Ask for a supported format. The disguised content is never parsed or rendered. |
| `ENCRYPTED_PDF` | Password-protected. The service never attempts to guess or crack. | Ask the sender to resend without a password. |
| `PAGE_CAP_EXCEEDED` | More pages than the configured maximum (200 default). | Ask the sender to split it. |
| `PIXEL_CAP_EXCEEDED` | The pages decode to more pixels than allowed. | Ask for a lower resolution. |
| `DECOMPRESSION_BOMB` | The file expands far more than its compressed size — a hostile-input defence. | Do not retry. Treat as suspicious. |
| `WALL_CLOCK_EXCEEDED` | Processing exceeded the per-document time limit. | Retry once; if it recurs, escalate. |
| `RESOLUTION_BELOW_FLOOR` | The image is too small to be a document. `detail` gives the measured size and the floor. | Ask for a higher-resolution scan. |
| `BLANK_PAGE` | No ink on the page. | Ask the sender to check they attached the right file. |
| `BLUR_ABOVE_CEILING` | Too blurred to read. | Ask for a steadier photo or a flatbed scan. |
| `SKEW_UNRECOVERABLE` | Rotated so far that straightening cannot be trusted. | Ask for a straight scan. |
| `TEXT_COVERAGE_BELOW_FLOOR` | Almost no text found on the page. | Ask the sender to confirm the attachment. |
| `OCCLUSION_DETECTED` | Something covers a substantial part of the page (a finger, a shadow, a sticker). | Ask for a clear photo. |
| `QUALITY_SCORE_BELOW_THRESHOLD` | Overall quality below the configured threshold. `detail` gives the score and the threshold. | Ask for a better copy. |
| `ALL_PAGES_BOUNCED` | Every page individually bounced. | As above. |
| `NO_TEXT_DETECTED` | The document parsed but no text was found anywhere. | Ask the sender to confirm the attachment. |

### 8.3 Quality flags

`pages[].quality_flags`. **Present on accepted pages too** — they are observations, not
errors, and watching their frequency is how you notice input quality drifting before
accuracy drops.

| Flag | Meaning |
|---|---|
| `RESOLUTION_BELOW_FLOOR` | Effective resolution under the floor. |
| `BLUR_ABOVE_CEILING` | Sharpness below the acceptable level. |
| `SKEW_DETECTED` | The page is tilted. |
| `SKEW_REPAIRED` | The tilt was corrected. **See the coordinate caveat in §4.3.** |
| `SKEW_UNRECOVERABLE` | The tilt is beyond safe correction. |
| `ROTATION_DETECTED` / `ROTATION_REPAIRED` | The page was sideways or upside down; corrected. |
| `INVERTED_REPAIRED` | Light-on-dark page was inverted to dark-on-light. |
| `TEXT_COVERAGE_BELOW_FLOOR` | Very little of the page is text. |
| `OCCLUSION_DETECTED` | Part of the page is covered. |
| `SHADOW_DETECTED` | Uneven lighting across the page. |
| `EXPOSURE_OUT_OF_RANGE` | Too dark or blown out. |
| `NOISE_ABOVE_CEILING` | Speckle or compression noise above the ceiling. |
| `BLANK_PAGE` | No ink. |
| `DUPLICATE_PAGE` | The same page appears more than once in the document. |
| `JUNK_TEXT_LAYER` | The PDF carried a text layer that was unusable, so the page was read as an image instead. |

Flags ending `_REPAIRED` record what was **fixed**, not what is wrong. A page can be
`accept` with several flags.

### 8.4 Warnings

`warnings[]` on the envelope. Each is `"<prefix>:<detail>"`. **This is how the service tells
you what did not happen** while still returning a usable result.

| Prefix | Example | Meaning | Consumer action |
|---|---|---|---|
| `slm_unavailable` | `slm_unavailable:PROVIDER_METHOD_NOT_SUPPORTED`<br>`slm_unavailable:PROVIDER_ERROR`<br>`slm_unavailable:PROVIDER_RESPONSE_INVALID` | The language model could not be reached, or its answer failed validation twice. **All deterministic fields are still present and correct.** You are missing free-text fields (complainant name, address, contact, relief sought, summary). | Treat the document as usable. If you need the free-text fields, queue it for a human or resubmit later. |
| `escalation_pending` | `escalation_pending:page=3` | This version was published before page 3's second-tier result landed. Always accompanied by `is_final: false`. | Act on what you have; wait for the next version for that page. |
| `tier2_failed` | `tier2_failed:page=2` | The second-tier reader was attempted for that page and did not return usable text. The first-tier result stands. | Consider the page lower-confidence. In the current build this happens on every escalation — see the note below. |

> **Known issue, `tier2_failed`.** The configured second-tier vision model is not served by
> any provider enabled on the build account, so **every** escalation currently ends in
> `tier2_failed` and no page has ever actually been improved by escalation. The escalation
> *decision* logic is exercised and correct; the escalation *outcome* is not available.
> Tracked as **Q15** in `OPEN-QUESTIONS.md`.

### 8.5 Field flags

`fields[name].flags`. Non-empty means a human should look at this value. The document-level
roll-up is `escalation_flag`.

| Flag | Meaning | Consumer action |
|---|---|---|
| `VALIDATOR_FAILED` | The value failed its validity check and was kept **as read**, not corrected. Always present when `validator_passed` is `false`. | Do not use it as an identifier without review. |
| `CHECKSUM_FAILED` | A checksum (e.g. an account or reference number) did not verify. | Review. |
| `GROUNDING_REJECTED` | A language model proposed a value that did not appear in the page's own text. | The value you see is *not* the rejected one; rejected values are dropped. Treat the field as unreliable. |
| `LOW_CONFIDENCE` | The producing mechanism was not confident. | Review. |
| `BANK_NAME_IFSC_DISAGREEMENT` | The bank named on the page and the bank implied by the IFSC code are different. | Review both fields; one was misread. |
| `IFSC_NOT_IN_MASTER` | The IFSC code is well-formed but not in the authoritative branch list. | Review. Note that if the deployment runs the synthetic sample list rather than the real RBI list, codes are reported as *unverifiable* rather than wrong. |
| `MULTIPLE_CANDIDATES` | Several plausible values were found on the page and one was chosen. | Review; the alternatives are on the page. |
| `ESCALATED_PAGE` | This value came from a page the router escalated. | Lower confidence by construction. |
| `TRANSLATION_UNAVAILABLE` | Translation was attempted and discarded as untrustworthy. `value` is intact; there is no `value_en`. | Nothing is wrong with `value`. |
| `HANDWRITTEN_SOURCE` | The value was read from handwriting. | Materially lower confidence. Review for anything financial. |

### 8.6 Job status

`status` on `SubmissionResponse` and `JobResponse`:

| Value | Meaning |
|---|---|
| `queued` | Accepted, not started. Only occurs in queued deployments. |
| `running` | In progress. |
| `completed` | Finished with a document. |
| `bounced` | Finished without a document — refused at ingestion or by the quality gate. |
| `failed` | The pipeline raised. `error_code` names the typed error. |

---

## 9. Idempotency and soft duplicates

### 9.1 Idempotency — exact, byte-level

`document_id` is `sha256:<hex>` of the uploaded bytes, and the store makes it unique.
**Submitting the same bytes twice never reprocesses the document.** The second call returns
the *same* job and the envelope the first call produced, with `is_new: false`. Verified on a
live instance: the second submission of the same file returned `is_new: false` and an
identical envelope.

This makes `POST /v1/documents` safe to retry after a network timeout: you get the original
result, not a duplicate job.

```python
body = submit(path)
if not body["is_new"]:
    print("already processed as", body["job_id"])
```

**It keys on bytes, not on content.** Two PDFs that differ only in an embedded modification
date are two documents. That is deliberate: pretending otherwise would silently drop a
resubmission in which the sender fixed a page.

### 9.2 Soft duplicates — advisory, never blocking

Separately, the service flags a job that *probably repeats* an earlier complaint:

- **Key:** `(sender, normalised subject, calendar day)`. The subject is normalised by
  Unicode NFKC, repeated removal of reply/forward prefixes (`Re: Fwd: Re:`), whitespace
  collapse and case folding. **Punctuation is kept**, because complaint references such as
  `CMS/2026/401687` live in subjects and are the most discriminating thing in them.
- **Effect:** `soft_duplicate_of` names the earlier job and `soft_duplicate_reason` is
  `"same_sender_subject_day"`.
- **It changes nothing else.** The document is still processed in full and a complete
  envelope is returned. It is a hint for your case-management system, not a rejection.

Only fires when **both** `sender` and `subject` were supplied. Omit them and the field is
always `null`.

Real example (second document from the same sender and subject on the same day):

```json
{
  "job_id": "1a1a1f28d59c4036b3240176d64529b1",
  "is_new": true,
  "status": "bounced",
  "soft_duplicate_of": "7c5b49126dd042b5904ea78f80878b2e",
  "soft_duplicate_reason": "same_sender_subject_day"
}
```

### 9.3 Which layer should deduplicate?

**This one, and it already does.** A complainant who mails the same attachment five times
costs five uploads and *one* pipeline run: the four repeats are answered from the store
without a model call, because §9.1 keys on the bytes and the uniqueness constraint is
checked inside the same transaction as the insert. There is nothing to add for that case.

Three mechanisms overlap, and it is worth knowing which covers what:

| Mechanism | Catches | Cost of a repeat |
|---|---|---|
| `document_id` (§9.1) | Byte-identical resubmission | Nothing — no OCR, no SLM |
| `soft_duplicate_of` (§9.2) | Same sender, subject and day | Full processing; advisory flag only |
| `DUPLICATE_PAGE` flag | The same page twice **inside one document** | Detected during the quality gate |

The reason this belongs here rather than in the mail gateway is that the key is a hash of
the attachment bytes, and this is the layer that has them. A caller that deduplicates on
`Message-ID` catches a redelivered *mail*, which is a different and smaller set: the same
image forwarded in a new mail, or attached to a fresh complaint, has a new `Message-ID`
and the same `document_id`. Both layers deduplicating is not wasteful — the gateway's
check is cheaper — but only this one can be authoritative.

**The gap worth knowing about.** The byte check is exact, so an image that has been
*re-encoded* between sends — a mail client recompressing a JPEG, a phone re-exporting a
photo, a scan redone at a different quality — produces different bytes, a different
`document_id`, and a full reprocess. Visually identical, technically new. Nothing in the
service currently catches that across documents: `page_hash` (a perceptual dHash) is
computed for every page and `DUPLICATE_PAGE` uses it, but only **within** a single
document; there is no cross-document index of page hashes.

Closing it would mean persisting `page_hash` per page and looking up near-matches
(Hamming distance ≤ a small threshold) on ingest. That is a contained change — the hash
and the distance function already exist and are tested — but it is a *policy* decision
first, because a near-duplicate is not proof of a duplicate complaint: two complainants
photographing the same bank notice produce near-identical pages and are two real cases.
The safe shape is to flag it the way `soft_duplicate_of` does, never to skip processing.

---

## 10. Versioning policy

### 10.1 Two version numbers, different jobs

| | Where | Meaning |
|---|---|---|
| `schema_version` | on every envelope, and on `GET /healthz` | The **contract**. What shape the JSON has. |
| `result_version` | on every envelope | Which **publication** of one job's result this is. Nothing to do with the schema. |

Current `schema_version`: **`1.0.0`**.

### 10.2 What a version change means

`schema_version` follows semantic versioning, and the rule is stated from the *consumer's*
point of view:

| Change | Bump | Examples |
|---|---|---|
| **Breaking** — a correct v1 consumer could stop working | **major** (`2.0.0`) | Removing a property; making an optional property required; removing a value from an enum; changing a property's type or meaning; renaming anything. |
| **Additive** — a correct consumer keeps working | **minor** (`1.1.0`) | A new optional property; a new value in an enum; a new warning prefix. |
| **Editorial** | **patch** (`1.0.1`) | Description and documentation changes only. |

A **major** bump means a new URL path (`/v2/...`) and a new schema file in `docs/schema/`.
It will never be delivered silently under `/v1/`.

The build enforces this: a test compares the published JSON schema against the code and
fails the build if a required property or an enum value disappears without a version bump.

### 10.3 How to be a well-behaved consumer

1. **Check `schema_version`** on the first envelope of a session (or on `GET /healthz` at
   start-up). Refuse to run against a major version you do not know.
2. **Ignore properties you do not recognise.** New optional properties arrive in minor
   versions. Do not use a strict "reject unknown fields" parser on the envelope.
3. **Handle unknown enum values.** New enum values arrive in minor versions. A new
   `BounceReason` or `FieldFlag` you have never seen means "something needs attention" —
   fall through to a safe default rather than crashing.
4. **Branch on codes, never on `message` or `detail`.** Those strings are for humans and
   change without a version bump.
5. **Do not depend on `stage_timings_ms` keys.** Stage names are diagnostic and not part of
   the stable contract.
6. **Do not depend on field-name coverage.** Which fields appear depends on what the page
   contains. Absence of `micr_code` means the page had none, not that the API changed.

---

## Related documents

| You want | Read |
|---|---|
| To install and run the service | `docs/AIR-GAPPED-DEPLOY.md` |
| The formal envelope contract | `docs/schema/result_envelope.schema.json` |
| Measured accuracy, and its caveats | `docs/eval/REPORT.md` |
| What is known-broken or undecided | `OPEN-QUESTIONS.md` (Q9, Q11, Q14, Q15 are referenced above) |
| To tune thresholds or models | `docs/TUNING.md` |
