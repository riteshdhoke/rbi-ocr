"""Pydantic contracts — the single source of truth for every boundary.

Three modules, in dependency order:

``errors``
    Typed exception taxonomy with machine-readable codes (invariant 11).
``ocr``
    ``OCRLine`` / ``OCRPage`` and the ``ModelProvider`` exchange models.
    Language lives on the line (invariant 1).
``envelope``
    ``Field`` / ``Evidence`` / ``DocumentResult`` / ``ResultEnvelope`` — the
    only shape that leaves the service (PRD P0-10).

Importing this package runs :mod:`app`'s thread-cap preamble first
(invariant 5); nothing here imports a numeric library.

This module does NOT handle: schema publication (see
:mod:`app.contracts.schema_export`), settings (:mod:`app.settings`), or any
pipeline behaviour.
"""

from __future__ import annotations

from app.contracts.envelope import (
    BOUNCE_REASON_FOR_QUALITY_FLAG,
    BOUNCE_REASON_FOR_REJECTION,
    SCHEMA_VERSION,
    BounceDecision,
    BounceReason,
    ComplaintSummary,
    DocumentResult,
    Evidence,
    Field,
    FieldFlag,
    FieldSource,
    PageResult,
    ProcessingStats,
    QualityFlag,
    QualityVerdict,
    ResultEnvelope,
    SummarySpan,
)
from app.contracts.errors import (
    REJECTION_EXCEPTIONS,
    BlankPageError,
    ConfigurationError,
    DecompressionBombError,
    DocumentRejected,
    EmptyFileError,
    EncryptedPDFError,
    MalformedPDFError,
    MimeMismatchError,
    PageCapExceededError,
    PipelineError,
    PixelCapExceededError,
    ProviderError,
    ProviderRateLimited,
    ProviderResponseInvalid,
    ProviderUnavailable,
    RejectionReason,
    ResidencyViolation,
    ResolutionBelowFloorError,
    TokenBudgetExceeded,
    WallClockExceededError,
    exception_for,
)
from app.contracts.ocr import (
    DETERMINISTIC_FIELDS,
    SLM_ELIGIBLE_FIELDS,
    SUMMARY_FIELD_NAME,
    BBox,
    Confidence,
    ExtractionContext,
    LangCandidate,
    LangResult,
    LanguageCode,
    OCRLine,
    OCRPage,
    OCRTier,
    PageClass,
    PageClassLabel,
    RawField,
    ScriptCode,
    ScriptResult,
    ScriptVote,
    StructuralFeatures,
    TextSpan,
    bbox_corners,
    bbox_from_xyxy,
)

__all__ = [
    "BOUNCE_REASON_FOR_QUALITY_FLAG",
    "BOUNCE_REASON_FOR_REJECTION",
    "DETERMINISTIC_FIELDS",
    "REJECTION_EXCEPTIONS",
    "SCHEMA_VERSION",
    "SLM_ELIGIBLE_FIELDS",
    "SUMMARY_FIELD_NAME",
    "BBox",
    "BlankPageError",
    "BounceDecision",
    "BounceReason",
    "ComplaintSummary",
    "Confidence",
    "ConfigurationError",
    "DecompressionBombError",
    "DocumentRejected",
    "DocumentResult",
    "EmptyFileError",
    "EncryptedPDFError",
    "Evidence",
    "ExtractionContext",
    "Field",
    "FieldFlag",
    "FieldSource",
    "LangCandidate",
    "LangResult",
    "LanguageCode",
    "MalformedPDFError",
    "MimeMismatchError",
    "OCRLine",
    "OCRPage",
    "OCRTier",
    "PageCapExceededError",
    "PageClass",
    "PageClassLabel",
    "PageResult",
    "PipelineError",
    "PixelCapExceededError",
    "ProcessingStats",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderResponseInvalid",
    "ProviderUnavailable",
    "QualityFlag",
    "QualityVerdict",
    "RawField",
    "RejectionReason",
    "ResidencyViolation",
    "ResolutionBelowFloorError",
    "ResultEnvelope",
    "ScriptCode",
    "ScriptResult",
    "ScriptVote",
    "StructuralFeatures",
    "SummarySpan",
    "TextSpan",
    "TokenBudgetExceeded",
    "WallClockExceededError",
    "bbox_corners",
    "bbox_from_xyxy",
    "exception_for",
]
