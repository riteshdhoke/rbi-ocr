"""JSON Schema contract tests — PRD P0-10, BUILD-BRIEF M6.

Two distinct failures, with two distinct fixes:

``test_committed_schema_matches_the_models``
    The models changed and the artefact was not regenerated. Fix: run
    ``python -m app.contracts.schema_export``.

``test_committed_schema_is_backward_compatible`` and friends
    A required property, an enum member or a definition **disappeared**. That
    is a breaking change for every consumer and needs a ``schema_version``
    bump and a new file, not a regeneration.

No ``jsonschema`` dependency: pydantic is the validator of record, and adding a
runtime dependency to assert a property of our own generator would be a licence
and supply-chain cost for no coverage.  The structural comparison below is what
the compatibility promise actually is.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.contracts.envelope import SCHEMA_VERSION, ResultEnvelope
from app.contracts.schema_export import (
    SCHEMA_PATH,
    build_schema,
    load_committed_schema,
    write_schema,
)

REGENERATE_HINT = (
    "regenerate with: .venv/Scripts/python.exe -m app.contracts.schema_export"
)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return load_committed_schema()


@pytest.fixture(scope="module")
def generated() -> dict[str, Any]:
    return build_schema()


def test_schema_artefact_is_committed(schema_path: Path) -> None:
    assert schema_path.is_file(), (
        f"{schema_path} is missing — the published contract must be in git. "
        + REGENERATE_HINT
    )
    assert schema_path == SCHEMA_PATH


def test_committed_schema_matches_the_models(
    committed: dict[str, Any], generated: dict[str, Any]
) -> None:
    assert committed == generated, (
        "the committed JSON Schema has drifted from the models. " + REGENERATE_HINT
    )


def test_schema_declares_its_identity(committed: dict[str, Any]) -> None:
    assert committed["title"] == "ResultEnvelope"
    assert committed["x-schema-version"] == SCHEMA_VERSION == "1.0.0"
    assert committed["$schema"].startswith("https://json-schema.org/draft/")
    assert SCHEMA_VERSION in committed["$id"]


def test_write_schema_is_deterministic_and_utf8(tmp_path: Path) -> None:
    first = write_schema(tmp_path / "a" / "schema.json")
    second = write_schema(tmp_path / "b" / "schema.json")
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert first.read_text(encoding="utf-8").endswith("\n")


# --------------------------------------------------------------------------- #
# Backward compatibility (the snapshot contract)
# --------------------------------------------------------------------------- #


def _required(schema: dict[str, Any], definition: str | None = None) -> set[str]:
    node = schema if definition is None else schema["$defs"][definition]
    return set(node.get("required", ()))


def _enum_values(schema: dict[str, Any], definition: str) -> set[str]:
    return set(schema["$defs"][definition]["enum"])


def test_committed_schema_is_backward_compatible(
    committed: dict[str, Any], generated: dict[str, Any]
) -> None:
    """No definition may vanish without a major ``schema_version`` bump."""
    lost = set(committed.get("$defs", {})) - set(generated.get("$defs", {}))
    assert not lost, (
        f"definitions removed from the published schema: {sorted(lost)}. "
        "This is a breaking change: bump SCHEMA_VERSION and publish a new file."
    )


def test_no_required_property_disappears(
    committed: dict[str, Any], generated: dict[str, Any]
) -> None:
    lost_root = _required(committed) - _required(generated)
    assert not lost_root, f"top-level required properties removed: {sorted(lost_root)}"
    for name in committed.get("$defs", {}):
        if name not in generated.get("$defs", {}):
            continue
        if "required" not in committed["$defs"][name]:
            continue
        lost = _required(committed, name) - _required(generated, name)
        assert not lost, (
            f"required properties removed from {name}: {sorted(lost)} — "
            "breaking change, bump SCHEMA_VERSION"
        )


def test_no_enum_member_disappears(
    committed: dict[str, Any], generated: dict[str, Any]
) -> None:
    for name, node in committed.get("$defs", {}).items():
        if "enum" not in node or name not in generated.get("$defs", {}):
            continue
        lost = set(node["enum"]) - set(generated["$defs"][name].get("enum", ()))
        assert not lost, (
            f"enum members removed from {name}: {sorted(lost)} — a consumer "
            "matching on them breaks; bump SCHEMA_VERSION"
        )


# --------------------------------------------------------------------------- #
# The guarantees the schema exists to publish
# --------------------------------------------------------------------------- #


def test_envelope_required_properties(committed: dict[str, Any]) -> None:
    assert {
        "result_version",
        "is_final",
        "document_id",
        "bounce_decision",
        "provider_profile",
    } <= _required(committed)


def test_field_evidence_is_required_in_the_published_schema(
    committed: dict[str, Any],
) -> None:
    """A consumer must be able to see that provenance is never optional."""
    assert "evidence" in _required(committed, "Field")
    assert "value" in _required(committed, "Field")
    assert "confidence" in _required(committed, "Field")
    assert "source" in _required(committed, "Field")


def test_evidence_requires_page_bbox_and_span(committed: dict[str, Any]) -> None:
    assert _required(committed, "Evidence") == {"page_no", "bbox", "span"}
    bbox = committed["$defs"]["Evidence"]["properties"]["bbox"]
    assert bbox["minItems"] == 8
    assert bbox["maxItems"] == 8


def test_source_enum_includes_ocr_label_discovery(committed: dict[str, Any]) -> None:
    source = committed["$defs"]["Field"]["properties"]["source"]
    assert set(source["enum"]) == {
        "regex_validator",
        "ocr_label",
        "lookup_table",
        "slm",
        "classifier",
        "human",
    }


def test_schema_version_is_a_constant_in_the_published_schema(
    committed: dict[str, Any],
) -> None:
    node = committed["properties"]["schema_version"]
    assert node.get("const") == "1.0.0" or node.get("enum") == ["1.0.0"]


def test_page_result_publishes_derived_language_summaries(
    committed: dict[str, Any],
) -> None:
    """Invariant 1: page language exists here, and only as a derived summary."""
    properties = committed["$defs"]["PageResult"]["properties"]
    for key in (
        "quality_score",
        "quality_flags",
        "ocr_tier",
        "primary_language",
        "language_distribution",
        "is_code_mixed",
    ):
        assert key in properties, f"PageResult.{key} missing from the schema"


def test_ocr_page_is_not_published(committed: dict[str, Any]) -> None:
    """The internal OCR contract must not leak into the external schema."""
    assert "OCRPage" not in committed.get("$defs", {})
    assert "OCRLine" not in committed.get("$defs", {})


def test_a_real_envelope_satisfies_the_published_required_keys(
    make_envelope: Callable[..., ResultEnvelope], committed: dict[str, Any]
) -> None:
    payload = make_envelope().model_dump(mode="json")
    missing = _required(committed) - set(payload)
    assert not missing, f"serialised envelope lacks required keys: {sorted(missing)}"
    unknown = set(payload) - set(committed["properties"])
    assert not unknown, f"serialised envelope has undeclared keys: {sorted(unknown)}"
