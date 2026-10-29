# Open field extraction

`EXTRACT_ALL_FIELDS=true` is now the default. Set it to `false` to use the
legacy fixed-field behavior. This setting changes extraction, not provider
selection or network access.

## Architecture and the former restrictions

`app/pipeline.py` coordinates ingestion, quality scoring, OCR, line-level language
identification, classification, validators, optional SLM extraction, translation,
categorisation and envelope assembly. Quality bounces stop extraction.

Previously the extraction path had several independent field restrictions:

- `app/stages/s5_validators.py`: nine registered deterministic extractors for
  account number, IFSC, bank name, card fragment, UTR/RRN, MICR, amount,
  transaction date and complaint date. These remain useful specialist checks.
- `app/contracts/ocr.py`: contexts allowed only five SLM targets, and raw model
  fields could not use the nine validator names.
- `app/providers/hf_api.py`: the prompt prohibited identifiers and the parser
  discarded fields outside the requested set.
- `app/stages/s5_slm.py`: the response schema enumerated field names and the
  grounding gate rejected other names.
- `app/contracts/envelope.py`: even grounded identifiers could not have SLM provenance.
- `app/stages/assemble.py`: repeated candidates were reduced to a single winner.

The open-field mode intentionally supersedes the field-ownership restriction in
historical PLAN.md invariant 7. Evidence checks, verbatim values, provider
residency rules, bounce behavior and validation-failure flags still apply.

## Current behavior

`app/stages/s5_discover.py` discovers explicit `Label: value` pairs, multiple form
cells separated by semicolons/tabs/multiple spaces, and numbered form labels with
values on the next nearby, aligned OCR line. Separate label/value OCR cells
on the same row are also supported, including unnumbered labels. Labels may use any script or name;
there is no field-name allowlist. Keys derive from labels using Unicode-aware
snake_case. Values remain verbatim OCR text, with page, bounding box and exact
character span. These fields use `source="ocr_label"` and
`validator_passed=null`: detecting a field does not claim a specialist check passed.

Assembly keeps validator-selected canonical fields, then distinct validator and
model occurrences, then discovered labels. Identical values under the same key
are deduplicated; differing values receive `_2`, `_3`, etc. Different label
spellings may produce separate keys rather than guessing they mean the same thing.

Model extraction contexts now carry `discover_fields`. In open mode the prompt
asks for every identifiable field, the parser accepts additional names, and the
grounding gate checks all returned values against OCR text. A model-produced
identifier may have `source="slm"`, with no validation claim. Extractive summaries
still use line indices, never invented prose. The response schema accepts any
nonempty field name. Clients must also accept the new `ocr_label` source value.

For local profiles, open model extraction can see validator-owned OCR spans.
The laptop profile still masks those spans before any remote request. Selecting
open extraction does not add remote calls to `cpu_node`.

## Practical limits

The current offline CPU provider has no SLM. It can extract explicit labels and
specialist validator patterns; it cannot reliably infer arbitrary semantic
fields hidden in narrative or reconstruct every table. Wider semantic discovery
requires a configured model-capable provider (for example the existing local
model-server path under `gpu_node`). Recognition errors and configured token/job
budgets can also limit recall. Open vocabulary is not a guarantee of perfect
extraction or validation.

## Requested fields per upload

The optional `requested_fields` multipart input accepts comma-separated names, for
example `complainant name, mobile, account number`. Names are normalized to Unicode
snake_case, deduplicated and sorted; empty CSV entries are ignored. The input has a
4,096-character size limit, with no field-name allowlist.

The list guides model extraction and can identify plain labels with a value on the
next nearby OCR line in offline mode. Extraction still discovers additional fields.
Requesting fields enables open extraction for that request even if the deployment's
legacy-mode setting is otherwise disabled.

The envelope records `requested_fields`. `document.extracted_fields` maps each
requested name to an array of matching, evidence-bearing Field objects; distinct
values are preserved and aliases repeating the same value are deduplicated.
An empty array means no matching value was found. `document.missing_fields` lists
those names. `document.other_fields` contains discoveries not matched to the request.
The original complete `document.fields` map remains available for existing clients.
With an empty request list, all discoveries appear in `other_fields`.

Matching uses normalized label words, so `complainant name` matches
`name_of_complainant`. It does not guess arbitrary synonyms. Use descriptive labels:
a broad request such as `name` can match multiple different names. The offline
provider still cannot infer arbitrary semantic facts from narrative text.

Jobs remain keyed by the document's hash. The same file and equivalent field list
reuse the latest saved result. A changed field list reruns extraction on that job
and appends new versions, preserving earlier results. `is_new` describes the job,
so it stays false for these re-extractions. The latest grouped result is also
available through `GET /v1/jobs/{job_id}` and webhook delivery. A request while the
same document is already processing receives HTTP 409 and can be retried afterward.
A bounced document records the list but carries no extracted document content.

## Verification

`tests/unit/test_s5_discover.py` covers arbitrary labels, multiple cells, Unicode,
repeated values, source spans, adjacent values, numbered form labels, legacy
mode, bounced pages, model discovery of custom/identifier fields, and rejection
of invented model values. Existing contract, provider, grounding, assembly and
pipeline tests cover the surrounding path.
