# Apply this update to an existing project

This folder contains 39 changed/new project files, preserving their original paths.
It is an update package, not a complete runnable project.

1. Stop the API on the destination machine and make a separate copy of its project
   and database before replacing files.
2. Copy the CONTENTS of this backup folder into the destination project root,
   merging app/, docs/, tests/ and corpus/. Include the hidden .env.example file.
3. Keep the destination machine's private .env. Set PROVIDER_PROFILE=cpu_node,
   SYNTHETIC_ONLY=false, HF_TOKEN= (empty), EXTRACT_ALL_FIELDS=true and
   CELERY_TASK_ALWAYS_EAGER=true. Keep its existing local model and database paths.
4. Use the existing Python 3.12 virtual environment. This update adds no packages
   or model weights. Do not run online installation commands on the offline machine.
5. Compile and verify as below, then start the service and hard-refresh the test page.

Windows PowerShell commands, from the destination project root:

```text
.venv\Scripts\python.exe -m compileall -q app
.venv\Scripts\python.exe -m app.contracts.schema_export
.venv\Scripts\python.exe -m pytest -q tests/unit/test_field_selection.py tests/unit/test_s5_discover.py tests/unit/test_api.py tests/unit/test_review_ui_static.py
.venv\Scripts\python.exe -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

On Linux/macOS replace .venv\Scripts\python.exe with .venv/bin/python.
Tests need the existing development dependencies; install nothing from the internet.
Open http://127.0.0.1:8000/ui/test_page.html and enter comma-separated requested fields.

The included corpus/index.json has portable forward-slash paths. It references the
existing synthetic corpus images; images and model weights are not included.
The normalized index differs from the original assets bundle's checksum.

Private .env, .venv/, model weights, databases and smoke outputs are excluded.
BACKUP-MANIFEST.json records SHA-256 and size for each of the 39 project files.

## Letter extraction quality update

This version adds s5_letter.py and s5_evidence.py, semantic field aliases,
multi-line evidence, conservative role/geometry checks and extraction_revision
cache invalidation. It includes the current handwriting-route dependencies
(base.py, s4b_vlm.py and s4c_handwriting.py) referenced by the existing pipeline.
Existing local GPU/VLM configuration still belongs in the destination .env.
CPU-only OCR cannot reliably transcribe every handwritten address or designation.
Unreliable fields stay missing; letter content is raw OCR marked for review.
For letters, addressee means recipient and sender means closing signature.

Additional verification:

```text
.venv\Scripts\python.exe -m pytest -q tests/unit/test_letter_extraction.py
```

## Multi-line bank-logo update

Include app/stages/s5_validators.py and tests/unit/test_bank_logo.py in this update.
The validator now joins nearby observed fragments such as Union Bank / of India
when they form a complete reference alias. Do not update bank_aliases.json just to
repair a split logo: the complete Union Bank of India alias already exists.
The extractor revision changes, so old cached results are re-extracted automatically.

```text
.venv\Scripts\python.exe -m pytest -q tests/unit/test_bank_logo.py tests/unit/test_s5_validators.py
```
