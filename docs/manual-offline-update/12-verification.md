# 12 — Verify and start the offline project

All commands here are offline. The target must already have its original dependencies.

## 1. Compile application code

```text
.venv\Scripts\python.exe -m compileall -q app
```

Expected: no output and exit code 0. If it names a SyntaxError or IndentationError,
fix that exact file before starting the API.

## 2. Regenerate the JSON Schema; do not hand-type the generated JSON

```text
.venv\Scripts\python.exe -m app.contracts.schema_export
```

Expected: `wrote .../docs/schema/result_envelope.schema.json`.
This uses the edited local models and does not access the internet.

## 3. Run the supplied behavioral checker

Put this guide folder in `docs/manual-offline-update/`, or type `verify_update.py`
there from the supplied reference. From the project root run:

```text
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --stage core
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --stage extraction
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --stage pipeline
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --stage api
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --stage schema
```

Each completed stage prints `PASS <stage>`. A failing stage prints `FAIL <stage>`
and a traceback; stop and fix it. `--stage all` runs all five stages in sequence.
The pipeline check uses the original synthetic corpus image and an in-memory mock
provider from `tests/contract/mock_provider.py`; it needs no real model calls and
imports no pytest. Keep the original tests folder available for this check.

Core checks parsing, contract flags and requested/extra/missing grouping.
Extraction checks arbitrary labels, repeated values, guided labels, fabricated
model-value rejection, and the model parser/prompt. Pipeline checks requested-name
propagation and append-version offsets. API checks the actual multipart input
schema. Schema checks your generated schema matches the edited code.

## 4. Optional exact source comparison

If `final-source/` and `SOURCE-SHA256.json` are available alongside the checker:

```text
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --compare
```

This prints each match/difference and exits nonzero on a difference. It compares
only runtime source files and ignores CRLF-vs-LF. Comments, formatting, retained
historical docstrings and import ordering still affect the comparison. A difference
is a prompt to compare the named file with its complete reference; it does not
by itself prove a behavioral bug. Never replace your private `.env` to match a hash.

## 5. Start the API

Confirm `.env` uses `cpu_node`, has no token, and enables open extraction (step 03).
Then run:

```text
.venv\Scripts\python.exe -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Expected: `Application startup complete.` and the local server URL. Keep this
terminal open. A second copy cannot bind the same port; stop the earlier copy first.

## 6. Live upload check in a second terminal

```text
.venv\Scripts\python.exe docs/manual-offline-update/verify_update.py --url http://127.0.0.1:8000
```

The script uploads only the bundled synthetic complaint form to your loopback API.
It checks health, requested matches, a deliberately missing field, additional
fields, equivalent-list result reuse, changed-list version increments and GET
retrieval. This creates/appends synthetic test results in the local database.
No external endpoint is contacted.

Expected: `PASS live HTTP`. This requires local model weights because the live
server performs real OCR. Recognition can vary by platform: if a known label is
not recognized, inspect OCR/model status rather than inventing a field value.

## 7. Browser check

Open `http://127.0.0.1:8000/ui/test_page.html` and hard-refresh it.

- Select `corpus/complaint_form_en/complaint_form_en_v0_clean.jpg`.
- Enter `name of complainant, mobile, account number, email`.
- Upload. On `cpu_node`, no synthetic declaration is required for residency;
  this specific file is synthetic test data.
- Requested values should include Rajesh Kumar Sharma, 9372805183 and 61136570711.
- `email` should be listed as not found.
- Other fields should include address/bank/amount/reference/complaint details.
- Raw JSON contains `result.requested_fields` and
  `result.document.extracted_fields`, `other_fields`, `missing_fields`, `fields`.
- Change the list and upload the same file. The job ID stays the same; result
  version increases. Upload again with the same normalized list: version stays put.
- A quality-bounce sample still returns `document=null`, without extracted content.

The CPU profile discovers explicit labels and specialist patterns. It does not
infer every fact in narrative text, and requesting a field does not manufacture
one or add a language model to the offline machine.
