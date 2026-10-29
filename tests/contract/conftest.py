"""Fixtures for the provider contract suite.

Puts this directory on ``sys.path`` so ``mock_provider`` imports by name from
every test module here, and supplies the shared factories the contract
assertions need.

This module does NOT handle: settings or envelope factories — those come from
the repository-level ``tests/conftest.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

CONTRACT_DIR = Path(__file__).resolve().parent
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from mock_provider import MockProvider

from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    OCRLine,
    PageClassLabel,
    StructuralFeatures,
    bbox_from_xyxy,
)


@pytest.fixture
def mock_provider() -> MockProvider:
    """A provider that answers every method deterministically."""
    return MockProvider(
        field_values={
            "complainant_name": "4. Name of bank",
            "relief_sought": "4. Name of bank",
        }
    )


@pytest.fixture
def make_extraction_context() -> Callable[..., ExtractionContext]:
    """Factory for a valid :class:`ExtractionContext`."""

    def _make(**overrides: Any) -> ExtractionContext:
        lines = overrides.pop(
            "lines",
            [
                OCRLine(
                    text="I was debited twice on 14 January 2026",
                    bbox=bbox_from_xyxy(100.0, 200.0, 700.0, 232.0),
                    confidence=0.9,
                    script="Latn",
                    language=None,
                    is_handwritten=False,
                    line_index=0,
                ),
                OCRLine(
                    text="and the bank did not reverse the amount",
                    bbox=bbox_from_xyxy(100.0, 240.0, 700.0, 272.0),
                    confidence=0.88,
                    script="Latn",
                    language=None,
                    is_handwritten=False,
                    line_index=1,
                ),
            ],
        )
        data: dict[str, Any] = {
            "document_id": "sha256:" + "b" * 64,
            "page_no": 1,
            "lines": list(lines),
            "page_class": PageClassLabel.COMPLAINT_FORM,
            "primary_script": "Latn",
            "primary_language": "eng",
            "target_fields": ["complainant_name", SUMMARY_FIELD_NAME],
            "max_tokens": 256,
        }
        data.update(overrides)
        return ExtractionContext(**data)

    return _make


@pytest.fixture
def structural_features() -> StructuralFeatures:
    """A plausible feature vector for ``classify_page``."""
    return StructuralFeatures(
        line_count=24,
        ink_density=0.11,
        aspect_ratio=0.707,
        has_table_lines=True,
        table_line_count=6,
        colour_histogram=[0.6, 0.3, 0.1],
        handwriting_ratio=0.25,
        text_coverage=0.42,
    )
