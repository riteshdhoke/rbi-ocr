# RBI Complaint Document Intelligence Service

OCR + extraction pipeline for RBI complaint email attachments: quality-gated ingestion,
script-routed CPU OCR (PP-OCRv5/ONNX), line-level language ID, deterministic identifier
validation, grounded SLM extraction (extractive only), entity-masked translation, and a
versioned evidence-bearing JSON envelope.

- Specs: `design/` (PRD, BUILD-BRIEF, TEST-DATA-SPEC, design-review v2)
- Plan & owner decisions: `PLAN.md`
- Decision log: `OPEN-QUESTIONS.md`

## Quickstart (dev, Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env    # then paste your HF_TOKEN into .env
.venv\Scripts\python tools\fetch_fonts.py
.venv\Scripts\python tools\generate_test_corpus.py --out corpus
.venv\Scripts\python -m pytest
```

Profiles: `laptop` (HF API for SLM/VLM, synthetic data only — enforced), `cpu_node`
(fully offline), `gpu_node` (local models on CUDA, plus an in-house model server if you
have one — it runs without one). See `.env.example` for all knobs. Production runs
air-gapped; nothing at runtime fetches from the network outside the `laptop` profile's
HF client, and the air-gapped profiles refuse to start if pointed off-premises.

Health and error rates: `GET /healthz`, `GET /v1/stats` (JSON, in-memory, resets on
restart) and `GET /metrics` (Prometheus).

## Configured extraction API

The upload UI at `/ui/test_page.html` exposes these options and loads
the deployment catalogue from `GET /v1/field-groups`.

Send `POST /v1/documents` as multipart form data with the attachment in
`file`, the application name in `realm`, and a JSON string in
`configurations`. Under the laptop profile, add `synthetic=true` for
synthetic uploads.

```bash
curl http://localhost:8000/v1/documents \
  -F 'file=@complaint.pdf' \
  -F 'realm=complaints' \
  -F 'configurations={
    "groups":[{"name":"banking","model":"cpu"}],
    "override_model":"cpu",
    "fields_to_extract":["complainant name"],
    "other_fields":false,
    "reactions":[],
    "file_size":5242880
  }'
```

- `fields_to_extract` adds individual field names to the selected groups.
  With `other_fields: false` (the default), field extractors use only that
  selection. An empty selection returns an empty field map. Set
  `other_fields: true` to discover extra fields in the primary extraction pass.
- The built-in groups are `banking`, `contact`, and `letter`.
  Set `API_FIELD_GROUPS` to a JSON object mapping group names to field-name
  arrays to replace the catalogue.
- A group's optional `model` is `cpu` or `gpu`. `override_model`
  applies to every group and individual field. Without an override,
  individual fields and groups without a model inherit the common explicit
  model when there is one, or use the server profile. For mixed models, the
  API combines results after all configured passes finish.
- `cpu` uses local OCR and deterministic extraction. `gpu` sends accepted
  pages to the in-house VLM, including printed pages. It requires an
  in-house `HF_BASE_URL` and the served model settings described below.
- `file_size` is an optional maximum number of upload bytes for this request.
  The service-wide maximum still applies.
- `reactions` defaults to an empty array. Use `["webhook"]` to invoke the
  service's configured result webhook.

The upload and `GET /v1/jobs/{job_id}` return the same compact shape for
configured jobs. Page numbers start at 1. Missing fields are omitted. Each
returned field uses the highest-ranked evidence-bearing match. The label is
the original document label associated with that value, or null if no label
was identified. A null validation result indicates that
no deterministic validator supplied a pass or fail for that value.

```json
{
  "job_id": "example-job-id",
  "pages_processed": [1],
  "extracted_data": {
    "ifsc_code": {
      "value": "BARB0417293",
      "confidence": 0.96,
      "source_page": 1,
      "label": "IFSC Code",
      "validation_status": {
        "method": "regex_validator",
        "passed": true,
        "flags": []
      }
    }
  }
}
```

Equivalent requests within a realm reuse a saved job. The realm is an
application identifier for job deduplication. Existing clients that submit
without `realm` and `configurations` continue to receive the versioned
envelope.

## Handwritten pages on an in-house GPU

Serve a vision model using the OpenAI-compatible endpoint from IDP2's vLLM
deployment. In RBI OCR's `.env`, set the endpoint and use the model names
advertised by that deployment. For a single Qwen server, the same model can
handle the image transcription and the text-only field extraction requests.

```dotenv
PROVIDER_PROFILE=gpu_node
HF_BASE_URL=http://127.0.0.1:8000/v1
GPU_HANDWRITING_ENABLED=true
GPU_HANDWRITING_MIN_RATIO=0.5
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLM_MAX_TOKENS=3072
SLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

The handwriting stage runs after the quality gate on raster pages without a
usable PDF text layer. The existing pixel heuristic estimates the share of
text lines that look handwritten. Pages at or above the threshold go straight
to the VLM and skip Tier-1 OCR. RBI OCR still validates fields and grounds SLM
results in the transcribed lines. If the VLM is unavailable or returns no
lines, the result records a page warning and CPU OCR stays skipped. Printed
pages and usable PDF text layers follow the regular path.

## Office hand-off

The office machine is air-gapped: it has no internet, no git and no copy of this
repository. It receives **two zip files** instead, both stamped with the git commit
they were built from so a receiver can prove they belong together.

| Bundle | Contains | Size |
|---|---|---|
| `dist/rbi-ocr-code-<shortsha>.zip` | Everything except the corpus images and the model weights: `app/`, `tools/` (with the corpus generator, the 8 fonts and `fribidi-0.dll`), `tests/`, `docs/`, `design/`, `alembic/`, `requirements*.txt`, `constraints.txt`, `.env.example`, `docker-compose.yml`, `models/README.md`. | ~3 MB |
| `dist/rbi-ocr-models-corpus-<shortsha>.zip` | `models/` (15 weight files, 837 MB — the OCR heads, detector, language identifier, translation engine and classifier coefficients) plus `corpus/` exactly as committed: 295 test documents, 295 `.gt.json` sidecars, `index.json`, and the 8-file `adversarial/` set. | ~600 MB |

**Neither bundle contains `.env`, a Hugging Face token or `.venv/`.** Bundle 2 carries the
model weights, so the receiving node needs nothing from the internet; every file in both
zips is listed with its SHA-256 in an embedded `BUNDLE_MANIFEST.json`, and
`tools/make_bundle.py --verify` re-checks them from the zips themselves.

### 1. Build both bundles

Run from the project root, on a machine with git:

```powershell
.venv\Scripts\python.exe tools\make_bundle.py
```

Expected output ends with:

```
BUNDLE OK — every assertion passed.
```

The tool asserts, and prints, that the code bundle contains no `.env`, no model
weights, no `.venv`, nothing under `corpus/`, and no string matching a Hugging Face
token (`hf_` followed by 20 or more alphanumerics — every byte of every file is
scanned, binaries included). It refuses to write an archive at all if a token is
found, so a failed run leaves nothing on disk to ship by accident. It also checks
the code bundle is under 15 MB.

The file list comes from `git ls-files`, so anything `.gitignore`d is excluded
structurally rather than by a blocklist. Untracked-but-not-ignored files **are**
included (that is how `constraints.txt` and the four hand-off documents got in
before they were committed); every one of them is printed by name and recorded in
the manifest under `untracked_included`.

### 2. Re-check the bundles at any time

```powershell
.venv\Scripts\python.exe tools\make_bundle.py --verify
```

This re-opens both zips, recomputes every SHA-256 from the bytes in the archive,
re-runs the token scan and the exclusion rules, and confirms the two zips name the
same commit. It builds nothing. Use it after copying the zips onto a USB stick, and
again on the receiving machine.

### 3. What the receiver does

Send both zips plus nothing else. The receiver follows
**`docs/AIR-GAPPED-DEPLOY.md`** from section 0; it takes them from a blank Windows or
Linux machine to a running service, with the expected output shown after every
command.

Each archive embeds its own `BUNDLE_MANIFEST.json` — git SHA, build time, file count,
a SHA-256 per file, and version stamps for the documents. Because the deploy guide
tells the operator to unzip **both** archives into the **same** directory, that name
would collide, so each archive carries a second, byte-identical copy named
`BUNDLE_MANIFEST-code.json` / `BUNDLE_MANIFEST-assets.json`. After a combined extract,
read those.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `git was not found on PATH` | The tool reads the file list from git. Build the bundles on a machine that has git; the *receiving* machine never needs it. |
| `BUNDLE BUILD FAILED` … `Hugging Face token pattern found in <file>` | A real token is in a file that git does not ignore. No archive was written. Remove the secret, or add the file to `.gitignore`, and re-run. The tool never prints the matched value. |
| `BUNDLE BUILD FAILED` … `code bundle is missing docs/…` | A required document is absent from the worktree. Restore it; do not build a bundle without it. |
| `corpus path(s) differ from HEAD` | Someone edited `corpus/` without committing. The corpus bundle is specified as "exactly as committed" — commit or restore the files, or pass `--allow-dirty-corpus` if you really mean to ship the working copy. |
| `WORKTREE DIRTY: N path(s) differ from HEAD` | A warning, not an error. The bundles are built from the **worktree**, so they contain those uncommitted changes; the short SHA names the commit they are based on, not their exact content. Commit first if you want the name to be exact. |
| The build reports an untracked file you did not expect | Read the `files : N (M untracked-but-not-ignored)` line and the `untracked_included` list in the manifest — every one of them is named. Anything that should never ship (editor state, local tooling config, scratch notes) belongs in `.gitignore`; anything that should ship belongs in a commit. `--tracked-only` excludes them all, at the cost of also excluding anything not yet committed. |
| `is not a readable zip … truncated or corrupted in transit` | The copy did not complete. Re-copy and compare file sizes with the sender. |
| The two zips have different `<shortsha>` values | They are not a matching pair. Rebuild both from one worktree; the corpus and the code that scores it are versioned together, and so are the weights and the code that loads them. |
| `models/ holds no files` | The assets bundle would ship a corpus and no weights, stranding the receiving node. Run `tools/fetch_models.py` on the connected machine first. `--allow-no-models` overrides it, for testing only. |
