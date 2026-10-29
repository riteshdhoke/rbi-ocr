"""Shared pytest fixtures for the whole suite.

Two jobs beyond fixtures:

1. Put the repository root on ``sys.path`` before anything imports ``app``, so
   the suite runs identically under ``python -m pytest``, a bare ``pytest`` and
   an IDE runner on Windows.
2. Import :mod:`app` **first**, so the thread-cap preamble (PLAN.md invariant
   5) runs before any test drags in NumPy or ONNX Runtime.  A test session that
   imported NumPy first would emit ``ThreadCapWarning`` and, worse, would not
   be measuring the configuration production runs under.

Everything here is offline by construction: no fixture opens a socket, reads
``.env`` or needs ``HF_TOKEN``.  ``Settings`` is always built with
``_env_file=None`` so a developer's real ``.env`` can never change a test
result.

This module does NOT handle: corpus fixtures (WS-D), provider fakes (WS-C), or
FastAPI test clients (WS-F).  Those belong to the lanes that own them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from app.contracts.envelope import (
    BounceDecision,
    ComplaintSummary,
    DocumentResult,
    Evidence,
    Field,
    PageResult,
    QualityVerdict,
    ResultEnvelope,
    SummarySpan,
)
from app.contracts.ocr import (
    OCRLine,
    OCRPage,
    OCRTier,
    PageClassLabel,
    TextSpan,
    bbox_from_xyxy,
)
from app.settings import Settings

#: Every environment variable :class:`Settings` reads. Cleared by ``clean_env``
#: so a developer's shell cannot leak into an assertion.
SETTINGS_ENV_VARS: tuple[str, ...] = tuple(Settings.model_fields)


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers the suite defines.

    ``local_models`` marks a test that loads real weights from ``models/``.  Such
    a test must **skip** with a reason on a machine with an empty ``models/``
    (PLAN.md: the suite has to pass on a fresh clone), and the marker is what
    lets a run exclude them wholesale with ``-m "not local_models"``.

    Registered here rather than in ``pytest.ini``/``pyproject.toml`` only because
    the repository has neither yet; move it when one is added (WS-E3 decision
    D-E3-6).
    """
    config.addinivalue_line(
        "markers",
        "local_models: needs real model weights under models/; skips when absent.",
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def schema_path(project_root: Path) -> Path:
    """The committed JSON Schema artefact."""
    return project_root / "docs" / "schema" / "result_envelope.schema.json"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove every settings variable from the environment for one test."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch


@pytest.fixture
def make_settings(clean_env: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Build a :class:`Settings` from explicit overrides only, never ``.env``."""

    def _make(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    """Default settings: laptop profile, synthetic only, no token."""
    return make_settings()


# --------------------------------------------------------------------------- #
# Contract factories
# --------------------------------------------------------------------------- #


@pytest.fixture
def bbox() -> list[float]:
    """A valid axis-aligned bbox, clockwise from top-left."""
    return bbox_from_xyxy(110.0, 534.0, 341.0, 566.0)


@pytest.fixture
def make_ocr_line(bbox: list[float]) -> Callable[..., OCRLine]:
    """Factory for :class:`OCRLine` with sane, overridable defaults."""

    def _make(**overrides: Any) -> OCRLine:
        data: dict[str, Any] = {
            "text": "4. Name of bank",
            "bbox": list(bbox),
            "confidence": 0.94,
            "script": "Latn",
            "language": "eng",
            "is_handwritten": False,
            "line_index": 0,
        }
        data.update(overrides)
        return OCRLine(**data)

    return _make


@pytest.fixture
def make_ocr_page(make_ocr_line: Callable[..., OCRLine]) -> Callable[..., OCRPage]:
    """Factory for :class:`OCRPage`; defaults to the bilingual-form hard case."""

    def _make(**overrides: Any) -> OCRPage:
        lines: Sequence[OCRLine] = overrides.pop(
            "lines",
            [
                make_ocr_line(line_index=0),
                make_ocr_line(
                    text="बैंक ऑफ "
                    "बड़ौदा",
                    script="Deva",
                    language="mar",
                    is_handwritten=True,
                    line_index=1,
                    bbox=bbox_from_xyxy(720.0, 526.0, 968.0, 567.0),
                ),
            ],
        )
        data: dict[str, Any] = {
            "page_no": 1,
            "width_px": 1240,
            "height_px": 1754,
            "lines": list(lines),
            "tier": OCRTier.TIER1,
        }
        data.update(overrides)
        return OCRPage(**data)

    return _make


@pytest.fixture
def make_evidence(bbox: list[float]) -> Callable[..., Evidence]:
    """Factory for :class:`Evidence` — page, bbox and span."""

    def _make(**overrides: Any) -> Evidence:
        data: dict[str, Any] = {
            "page_no": 1,
            "bbox": list(bbox),
            "span": TextSpan(line_index=0, start=0, end=11),
        }
        data.update(overrides)
        return Evidence(**data)

    return _make


@pytest.fixture
def make_field(make_evidence: Callable[..., Evidence]) -> Callable[..., Field]:
    """Factory for :class:`Field`; defaults to a passing IFSC validator hit."""

    def _make(**overrides: Any) -> Field:
        data: dict[str, Any] = {
            "value": "BARB0417293",
            "normalized": "BARB0417293",
            "confidence": 0.99,
            "source": "regex_validator",
            "validator_passed": True,
            "evidence": overrides.pop("evidence", make_evidence()),
        }
        data.update(overrides)
        return Field(**data)

    return _make


@pytest.fixture
def make_summary(
    make_evidence: Callable[..., Evidence],
) -> Callable[..., ComplaintSummary]:
    """Factory for a verbatim extractive :class:`ComplaintSummary`."""

    def _make(**overrides: Any) -> ComplaintSummary:
        spans = overrides.pop(
            "spans",
            (
                SummarySpan(
                    evidence=make_evidence(span=TextSpan(line_index=3, start=0, end=22)),
                    text="The bank debited twice",
                ),
                SummarySpan(
                    evidence=make_evidence(span=TextSpan(line_index=4, start=0, end=19)),
                    text="on 14 January 2026.",
                ),
            ),
        )
        separator = overrides.pop("separator", "\n")
        text = overrides.pop(
            "text", separator.join(span.text for span in spans)
        )
        return ComplaintSummary(
            text=text, spans=tuple(spans), separator=separator, **overrides
        )

    return _make


@pytest.fixture
def make_page_result() -> Callable[..., PageResult]:
    """Factory for :class:`PageResult` — an accepted, code-mixed page."""

    def _make(**overrides: Any) -> PageResult:
        data: dict[str, Any] = {
            "page_no": 1,
            "quality_score": 0.82,
            "quality_flags": (),
            "quality_verdict": QualityVerdict.ACCEPT,
            "ocr_tier": OCRTier.TIER1,
            "line_count": 2,
            "primary_script": "Latn",
            "primary_language": "eng",
            "language_distribution": {"eng": 0.5, "mar": 0.5},
            "is_code_mixed": True,
        }
        data.update(overrides)
        return PageResult(**data)

    return _make


@pytest.fixture
def make_document_result(
    make_field: Callable[..., Field],
) -> Callable[..., DocumentResult]:
    """Factory for :class:`DocumentResult`."""

    def _make(**overrides: Any) -> DocumentResult:
        data: dict[str, Any] = {
            "document_class": PageClassLabel.COMPLAINT_FORM,
            "document_class_confidence": 0.91,
            "fields": overrides.pop("fields", {"ifsc_code": make_field()}),
            "page_count": 1,
            "primary_script": "Latn",
            "primary_language": "eng",
            "language_distribution": {"eng": 0.5, "mar": 0.5},
            "is_code_mixed": True,
        }
        data.update(overrides)
        return DocumentResult(**data)

    return _make


@pytest.fixture
def make_envelope(
    make_page_result: Callable[..., PageResult],
    make_document_result: Callable[..., DocumentResult],
) -> Callable[..., ResultEnvelope]:
    """Factory for a schema-valid :class:`ResultEnvelope` (accepted document)."""

    def _make(**overrides: Any) -> ResultEnvelope:
        pages = overrides.pop("pages", (make_page_result(),))
        data: dict[str, Any] = {
            "result_version": 1,
            "is_final": True,
            "document_id": "sha256:" + "a" * 64,
            "provider_profile": "laptop",
            "bounce_decision": BounceDecision.accepted(),
            "pages": tuple(pages),
            "document": overrides.pop(
                "document", make_document_result(page_count=len(tuple(pages)))
            ),
        }
        data.update(overrides)
        return ResultEnvelope(**data)

    return _make


__all__ = ["PROJECT_ROOT", "SETTINGS_ENV_VARS", "app"]
