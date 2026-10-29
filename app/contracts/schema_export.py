"""Export the published JSON Schema for :class:`ResultEnvelope` (PRD P0-10).

Run as a module to regenerate the committed artefact::

    .venv\\Scripts\\python.exe -m app.contracts.schema_export

The file lands at ``docs/schema/result_envelope.schema.json`` and is asserted
against the live models by ``tests/unit/test_contracts_schema.py``:

* a **drift** test — the regenerated schema must equal the committed one, so a
  model change without a regenerated schema fails the build;
* a **backward-compatibility** test — no required property and no enum member
  may disappear from the committed schema without a ``schema_version`` bump.

Serialization mode is used deliberately: consumers read what we *emit*, and
serialization mode is what resolves defaults, tuples and datetimes into the
JSON shapes they will actually see.

This module does NOT handle:
  * schema *versioning policy* — bumping ``SCHEMA_VERSION`` is a human
    decision; this only writes what the models currently say;
  * publishing to a registry or serving the schema over HTTP (WS-F);
  * validating instances against the schema (pydantic is the validator; the
    JSON Schema is for external consumers);
  * schemas for the internal OCR contracts — those are not published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from app.contracts.envelope import SCHEMA_VERSION, ResultEnvelope

__all__ = ["SCHEMA_ID", "SCHEMA_PATH", "build_schema", "write_schema"]

#: Repository root, resolved from this file so CWD never matters (Windows-safe).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Committed artefact path.
SCHEMA_PATH: Final[Path] = (
    _PROJECT_ROOT / "docs" / "schema" / "result_envelope.schema.json"
)

SCHEMA_ID: Final[str] = (
    "https://rbi-ocr-service.invalid/schema/"
    f"result_envelope/{SCHEMA_VERSION}/result_envelope.schema.json"
)


def build_schema() -> dict[str, Any]:
    """Return the JSON Schema for :class:`ResultEnvelope`, deterministically."""
    schema: dict[str, Any] = ResultEnvelope.model_json_schema(
        mode="serialization",
        ref_template="#/$defs/{model}",
    )
    ordered: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "ResultEnvelope",
        "description": (
            "RBI complaint document intelligence service — published result "
            "envelope. Every field carries source and evidence; page-level "
            "language values are derived summaries."
        ),
        "x-schema-version": SCHEMA_VERSION,
    }
    ordered.update(schema)
    return ordered


def write_schema(path: Path | None = None) -> Path:
    """Write the schema to ``path`` (default :data:`SCHEMA_PATH`) as UTF-8.

    Keys are sorted and a trailing newline is written so the artefact is stable
    under git and diffable line by line.
    """
    target = path or SCHEMA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_schema(), indent=2, sort_keys=True, ensure_ascii=False
    )
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def load_committed_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the committed schema artefact.

    Raises:
        FileNotFoundError: when the artefact is missing — regenerate it with
            ``python -m app.contracts.schema_export`` rather than skipping the
            contract test.
        TypeError: when the file parses but is not a JSON object.
    """
    target = path or SCHEMA_PATH
    loaded: Any = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{target} does not contain a JSON object")
    return loaded


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    written = write_schema()
    print(f"wrote {written}")
