# Apply this update to an existing project

This folder contains 30 changed/new project files, preserving their original paths.
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
BACKUP-MANIFEST.json records SHA-256 and size for each of the 30 project files.
