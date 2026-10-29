# Manual offline update: open fields and requested field lists

This guide reproduces both changes: removing the fixed field-name limit and adding
comma-separated requested fields, grouped output, missing fields and result versions.
Follow the numbered files in order. All paths below are relative to the project root.

## Before typing

1. Stop the running API with Ctrl+C in its terminal.
2. Make a separate backup copy of the project and database. Keep its `.env` private.
3. Confirm the original project already runs on the offline machine. This update adds
   no packages, weights, Redis service or database migration. It does require the
   existing Python 3.12 environment and runtime dependencies.
4. If the project has never run there, first follow `docs/AIR-GAPPED-DEPLOY.md`'s
   offline wheelhouse route. Never use `setup.sh` without `--offline` on that machine.
5. Use a UTF-8 editor, spaces for indentation, and straight ASCII quotes in Python.
   Do not type Markdown fences (the lines containing three backticks) into source files.
6. Work from the project root. Windows commands below assume PowerShell and
   `.venv\Scripts\python.exe`. On Linux/macOS substitute `.venv/bin/python`.

Check the existing environment:

```text
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -m pip check
```

Expected: Python 3.12.x and `No broken requirements found.`
These commands do not install or download anything.

## How to use the files

Each step names a file, a function/class to find and the exact code to type. Use
function names as anchors, not line numbers. The numbered steps contain the necessary
runtime edits; the final source copies are the complete authoritative reference.
If your starting code differs substantially, compare it with the reference rather
than replacing a block you cannot identify.

| Step | What to type |
|---|---|
| 01 | New field-name normalization and grouping module |
| 02 | New offline OCR label-discovery module |
| 03 | Setting and `.env` configuration |
| 04 | OCR/model exchange contracts |
| 05 | Result contract, requested/other groups and missing fields |
| 06 | SLM context/schema/grounding changes |
| 07 | Model prompt and parser changes |
| 08 | Assembly integration |
| 09 | Pipeline integration and append-only re-extraction versions |
| 10 | Upload API input and changed-list behavior |
| 11 | Browser upload input and grouped display |
| 12 | Schema regeneration, offline checks and live verification |
| 13 | Optional test/documentation alignment and troubleshooting |

Do not start the service midway through the edits: some files depend on later steps.
Syntax-only checks can run after every file; behavioral checkpoints run at specified
milestones. A successful syntax check alone does not prove extraction is correct.

## Files supplied alongside this guide

- `final-source/`: full copies of changed runtime files, tests and documentation.
  These are references to compare against, not another installed application.
- `verify_update.py`: offline behavioral verification using existing dependencies;
  no pytest installation is needed. Full script content is provided as a file so
  it can also be typed into the target project if file transfer is unavailable.
- `SOURCE-SHA256.json`: hashes of reference files after newline normalization.
  The check script's `--compare` mode compares final source text while allowing
  Windows CRLF vs Unix LF differences. Formatting/docstring differences can fail
  this exact comparison even if behavior is equivalent.

No credentials, `.env`, virtual environment, model weights, database or uploaded
complaints are included in this guide's package.
