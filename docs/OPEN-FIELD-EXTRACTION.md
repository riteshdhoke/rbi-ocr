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

## Letter structure and conservative discovery

`app/stages/s5_letter.py` adds evidence-backed email, postal-address block, recipient,
sender/signatory, sender designation, and letter-body recognition. These recognizers
complement arbitrary field labels; they are not an output-field allowlist.
A readable greeting identifies the recipient; a readable closing/signature identifies
the sender. Designation requires an associated readable role line. Text inside the
uploaded document remains data, including any instructions in the letter.

Layout inference now rejects label/value pairs whose writing angles differ, skips
low-confidence OCR fragments, and does not promote digit-containing address lines
to arbitrary field names. Email, sender/recipient and postal-address aliases retain
their distinct meanings. Missing or ambiguous values remain missing rather than
being replaced by unrelated bank identifiers or nearby narrative.

Multi-line addresses and letter content use `source="ocr_layout"` and carry exact
`evidence.spans` for all source lines. The original `evidence.span` remains the first
span for compatibility; the bounding box covers the selected lines. Model-proposed
multi-line values must exactly join their cited OCR lines; arbitrary rewritten
summaries and spelling repairs are rejected. A signature proposed as the addressee
is rejected when clear letter-role evidence contradicts it.

This is deliberately conservative. An address block requires both a street and a
postal destination and sufficiently readable source lines. Current deterministic
address conventions and English greeting/closing rules do not cover all languages,
layouts or international addresses. Uncertain address or designation fragments are
withheld. Raw letter content is marked for review and may still contain OCR spelling
errors; geometry and OCR confidence cannot prove a transcription is correct.

## Bank names split over logo lines

The bank validator joins up to three nearby, horizontally overlapping text
fragments only when their observed text forms a complete reference-table alias.
This handles logos such as `Union Bank` above `of India`, even if an OCR line from
another column appears between them in reading order. Distant fragments, separate
columns and genuinely distinct banks are not joined. Regulator exclusions still
apply across line breaks. Every contributing span is retained in evidence and
masked under remote-profile rules.

A suffix such as `of India` alone no longer becomes a second bank through fuzzy
matching. The full observed name is the field value; the canonical reference name
remains in `normalized`. Generic label inference skips validator-claimed lines and
unlabeled rows of three or more cells, so a bank logo or table header is not used
as an arbitrary label for its neighbor. The extractor revision was updated to
recompute cached results affected by the older bank-logo behavior.

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

Matching uses normalized label words plus a small set of explicit semantic aliases.
For example, `complainant name` matches `name_of_complainant`, and `email id` matches
`email`. Email addresses and postal addresses are separate concepts. Recipient and
sender names/designations are also separate concepts; a signature is never an
addressee merely because it is the clearest visible name. Unknown custom field
names remain supported. Ambiguous broad requests such as `name` are left unresolved
when multiple different roles match. Matching aliases are deduplicated between
requested and other groups. The offline
provider still cannot infer arbitrary semantic facts from narrative text.

Jobs remain keyed by the document's hash. The same file and equivalent field list
reuse the latest saved result only when its `extraction_revision` is current.
Older results are re-extracted once with append-only versions after an extractor
revision, so stale incorrect fields are not returned indefinitely. A changed field list reruns extraction on that job
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
