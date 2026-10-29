---
name: rbi-ocr
description: Develop, run, and troubleshoot this RBI complaint document OCR service, including offline deployment, evidence-backed field extraction, provider configuration, and API behavior. Use when working in this project.
---

# RBI OCR project

Run commands from the project root. Read the implementation relevant to the task;
older specifications describe some behavior that has since changed.

## Read the right reference

- Deployment and startup: [docs/AIR-GAPPED-DEPLOY.md](docs/AIR-GAPPED-DEPLOY.md).
- Current extraction architecture and limits:
  [docs/OPEN-FIELD-EXTRACTION.md](docs/OPEN-FIELD-EXTRACTION.md).
- HTTP contract: [docs/API.md](docs/API.md).
- Configuration and performance: [docs/TUNING.md](docs/TUNING.md) and
  [app/settings.py](app/settings.py).
- Historical decisions and unfinished work: [PLAN.md](PLAN.md) and
  [OPEN-QUESTIONS.md](OPEN-QUESTIONS.md). Treat their fixed field-ownership
  rules as historical when open extraction is enabled.

## Architecture

`app/api/app.py:create_app` constructs the FastAPI application.
`app/api/routes.py` handles uploads, stored results, health, statistics and metrics.
`app/pipeline.py` coordinates ingestion, quality checks, OCR, line-level language
identification, classification, validators, optional model extraction, translation,
categorisation and assembly.

- `app/contracts/ocr.py`: OCR lines/pages and model exchange contracts.
- `app/stages/s5_validators.py`: specialist deterministic extractors and checks.
- `app/stages/s5_discover.py`: open-vocabulary discovery from OCR labels and layout.
- `app/stages/s5_slm.py`: extraction contexts and OCR grounding of model output.
- `app/providers/`: local OCR/translation and model endpoint adapters.
- `app/stages/assemble.py`: final field merging, evidence and derived summaries.
- `app/contracts/envelope.py`: published result contract.
- `app/storage/`: jobs, result versions and document-hash deduplication.

## Extraction behavior to preserve

`EXTRACT_ALL_FIELDS=true` is the default. Do not reintroduce a field-name allowlist
in the parser, grounding gate, response schema or output contract. Existing
validators check known fields; they do not define everything extraction may return.
`EXTRACT_ALL_FIELDS=false` selects the legacy fixed-field behavior.

Open discovery copies explicit labels and values, including numbered forms and
separate OCR cells on the same row. Distinct repeated values receive numeric
suffixes. Model-assisted extraction can propose additional field names, but its
values must pass OCR grounding before publication.

Every field needs page, bounding box and character-span evidence. Preserve native
OCR text; normalization and translation belong in their separate advisory fields.
Discovering a field does not prove it valid: use `validator_passed=null` when no
specialist validation ran. Label discoveries use `source="ocr_label"`. Keep
validation-failure flags and preserve distinct candidates.

Script, language and handwriting are line attributes. Derive page/document language
summaries during assembly. Complaint summaries select OCR line indices and retain
verbatim text. Bounced pages must not contribute extracted content.

The offline CPU provider has no SLM. Its label discovery cannot infer every fact
from narrative or reconstruct arbitrary tables. Do not claim unrestricted field
names guarantee complete semantic extraction.

## Run the service

Use Python 3.12 and the explicit virtual-environment interpreter; activation is not
required. Install the pinned runtime and development dependencies together:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

On an air-gapped target, use the guide's matching-platform wheelhouse and offline
installation commands instead. Never download models or dependencies at runtime.
`setup.sh --offline` uses `./wheelhouse`; `setup.sh` takes the connected install route.

For the single-node offline configuration, use `PROVIDER_PROFILE=cpu_node`, an empty
`HF_TOKEN`, local model artifacts and eager tasks. SQLite needs no separate database
server. Keep the configured numeric-library thread caps.

```sh
.venv/bin/python -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

- Test page: `http://127.0.0.1:8000/ui/test_page.html`.
- API documentation: `http://127.0.0.1:8000/docs`.
- Health: `/healthz`; statistics: `/v1/stats`; metrics: `/metrics`.

Serve the test page through the API so uploads use the same origin. The laptop
profile permits synthetic data only and may use a remote endpoint; keep its
residency checks and identifier masking. Local profiles must retain their offline
behavior. A field-discovery change must not silently enable remote calls.

The deployment guide targets Windows/Linux. Verify platform-specific dependencies
and behavior on macOS rather than assuming the reference results reproduce.
Windows-generated corpus indexes may contain backslash separators that require
normalization on POSIX systems; record such changes when checking bundle hashes.
The Docker Compose topology is documented as unverified; use the documented
single-node startup for local operation unless container deployment is requested.

## Verify changes

Choose tests matching the changed stage. Relevant suites include:

```sh
.venv/bin/python -m pytest -q tests/unit/test_s5_discover.py tests/unit/test_s5_slm.py tests/unit/test_assemble.py tests/unit/test_pipeline.py
.venv/bin/python -m pytest -q tests/unit/test_contracts_ocr.py tests/unit/test_contracts_envelope.py tests/unit/test_contracts_schema.py tests/unit/test_providers_hf_api.py tests/unit/test_api.py tests/unit/test_settings.py
```

After a published contract change, regenerate the schema and run contract checks:

```sh
.venv/bin/python -m app.contracts.schema_export
```

For a real offline pipeline check, write outputs to a scratch location:

```sh
.venv/bin/python tools/run_corpus_eval.py --limit 2 --no-slm --out ./smoke/predictions --manifest ./smoke/MANIFEST.json --report ./smoke/report.txt --json-out ./smoke/report.json
```

Keep committed evaluation predictions and calibration references unchanged unless
the task explicitly involves recalibration. Report calibration drift and native
library crashes separately from assertion failures; do not describe an incomplete
suite as passing.

Uploads are deduplicated by document hash and requested field list. The optional
`requested_fields` multipart input is a comma-separated list that guides extraction.
The envelope records the normalized names; `document.extracted_fields` contains
arrays of requested matches, `document.other_fields` contains additional discoveries,
and `document.missing_fields` records absent requests. `document.fields` remains the
complete map. See `app/field_selection.py` for normalization and label matching.

The same file and equivalent list reuse the latest stored result. Changing the list
re-extracts that job and appends result versions; never overwrite its historical
results. A submission while extraction is in progress returns HTTP 409. Include
API re-submission, persistence, requested/extra grouping, missing values and repeated
values in verification when changing this workflow.
