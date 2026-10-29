# 13 — Tests, documentation and troubleshooting

## Existing tests must reflect the intentionally changed behavior

The original suite contains assertions that model identifiers/custom fields must
be rejected. Those assertions describe the behavior you intentionally removed.
The `final-source/tests/unit/` folder supplies the updated tests and the two new
suites in full. They are optional to type for production runtime, but necessary to
reproduce the updated regression suite. Do not edit unrelated failing assertions
merely to make the suite green.

Original tests that need corresponding updates are in:

```text
tests/unit/test_api.py
tests/unit/test_assemble.py
tests/unit/test_contracts_envelope.py
tests/unit/test_contracts_schema.py
tests/unit/test_pipeline.py
tests/unit/test_review_ui_static.py
tests/unit/test_s5_slm.py
```

New tests are:

```text
tests/unit/test_field_selection.py
tests/unit/test_s5_discover.py
```

After typing/copying those final test versions, run:

```text
.venv\Scripts\python.exe -m pytest -q tests/unit/test_field_selection.py tests/unit/test_s5_discover.py tests/unit/test_api.py tests/unit/test_review_ui_static.py tests/unit/test_assemble.py tests/unit/test_s5_slm.py tests/unit/test_providers_hf_api.py tests/unit/test_contracts_ocr.py tests/unit/test_contracts_envelope.py tests/unit/test_contracts_schema.py tests/unit/test_pipeline.py tests/unit/test_settings.py
```

Tests require the original development dependencies already installed. Do not run
an online pip command on the offline machine to obtain them.

## Documentation/reference files

The reference folder also contains the updated SKILL.md, open-extraction guide,
API and deployment documentation, and `.env.example`. Typing those files is useful
for maintenance but is not required to execute the new feature.
The published JSON Schema must instead be regenerated in step 12.

## Troubleshooting

| Symptom | Check |
|---|---|
| ImportError for `field_selection` / `s5_discover` | New file paths and imports in steps 01–02; run from the project root. |
| `ocr_label` rejected as a source | `FieldSource` in step 05. |
| `discover_fields` / `discovered` rejected as extra input | Both contract fields and their method edits in step 04. |
| Custom names still disappear | Schema enum and grounding guard (06), provider parser and context flags (07). |
| Requested groups are absent | Root metadata and consistency method (05), pipeline publish metadata (09). |
| Requested names do not guide a model | Context target_fields and both tier call sites (09), prompt (07). |
| Changed list returns old version | API cache comparison (10), publisher version offset (09). Restart the server. |
| `result_version ... expected ...` error | Starting version from API (10); apply the offset only once in the publisher (09). |
| New input visible but upload omits it | Shared `postDocument` and upload listener (11); hard-refresh cached JavaScript. |
| All results appear in one table | `renderAll`, output containers, optional selectedRows renderer (11). |
| HTTP 409 | Another extraction for this document is running; retry after completion. |
| `ModuleNotFoundError` for an existing dependency | Confirm the venv interpreter and original offline install. No new packages are needed. |
| Corpus files look missing on Linux/macOS | Windows backslashes in corpus/index.json may need normalization. Do not modify corpus images or ground truth. |
| Calibration differs | Report the difference; do not overwrite reference calibration to hide it. |

No model file or `s5_validators.py` change is required. Keep the existing validators,
residency controls, evidence checks, append-only storage and quality bounces.
