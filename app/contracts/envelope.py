"""The published result contract — BUILD-BRIEF sec. 3.3 and PRD P0-10.

This is the only shape that leaves the service.  Everything about it is
designed so a consumer can decide *how much to trust each value* without
knowing anything about the pipeline:

* every :class:`Field` names its ``source`` and carries **mandatory**
  :class:`Evidence` (page, bbox, span).  A value without provenance cannot be
  represented;
* ``value`` is the immutable native-script string; ``value_en`` is advisory and
  cannot exist without an ``mt_engine`` (PLAN.md invariant 10);
* a failed checksum is ``validator_passed=False`` **plus** a flag, never a
  corrected value (invariant 9);
* ``complaint_summary`` is a verbatim concatenation of selected OCR spans, and
  :class:`ComplaintSummary` *checks the concatenation* — model-written prose
  cannot be represented (invariant 2);
* page-level ``primary_language`` / ``language_distribution`` /
  ``is_code_mixed`` appear here and only here, as **derived summaries**
  computed in ``assemble.py`` (invariant 1);
* ``result_version`` + ``is_final`` allow the fast path to publish before
  escalated pages land.

``pydantic.Field`` is imported as ``_Field`` throughout because ``Field`` is
the name of the *contract* class in BUILD-BRIEF sec. 3.3 and the brief's name
wins.

This module does NOT handle:
  * computing any derived summary — it validates internal consistency
    (``primary_language`` must be the argmax of ``language_distribution``) but
    never derives one.  ``assemble.py`` owns derivation;
  * the grounding gate (invariant 8) or Levenshtein comparison — a rejected SLM
    value never becomes a ``Field``; it is dropped and flagged upstream;
  * translation — :meth:`Field.with_translation` records an already-produced
    ``value_en``; it calls no MT engine;
  * webhook delivery, HMAC signing, or the DLQ (WS-F);
  * schema publication — see :mod:`app.contracts.schema_export`;
  * PII redaction. This object contains complainant data by construction and
    must never be logged at INFO.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as _Field

from app.contracts.errors import RejectionReason
from app.contracts.ocr import (
    BBox,
    Confidence,
    LanguageCode,
    OCRTier,
    PageClassLabel,
    PageNumber,
    ScriptCode,
    TextSpan,
    check_distribution,
)

__all__ = [
    "BOUNCE_REASON_FOR_QUALITY_FLAG",
    "BOUNCE_REASON_FOR_REJECTION",
    "SCHEMA_VERSION",
    "BounceDecision",
    "BounceReason",
    "ComplaintSummary",
    "DocumentResult",
    "Evidence",
    "Field",
    "FieldFlag",
    "FieldSource",
    "PageResult",
    "ProcessingStats",
    "QualityFlag",
    "QualityVerdict",
    "ResultEnvelope",
    "SummarySpan",
]

#: Bumped only for a breaking change; ``tests/unit/test_contracts_schema.py``
#: fails the build if a required property or enum value disappears without it.
SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

#: BUILD-BRIEF sec. 3.3, verbatim.  ``classifier`` covers the page/category
#: models; ``human`` covers a review-UI correction (P1-1).
FieldSource = Literal[
    "regex_validator",
    "ocr_label",
    "lookup_table",
    "slm",
    "classifier",
    "human",
]


class QualityVerdict(StrEnum):
    """Stage 1.5 decision (design-review A5)."""

    ACCEPT = "accept"
    REPAIR = "repair"
    BOUNCE = "bounce"


class QualityFlag(StrEnum):
    """Machine-readable quality-gate observations.

    Present on **every** page record including accepted ones (PRD P0-3), so a
    drift in flag frequency is visible before it becomes a drift in accuracy.
    Flags ending ``_REPAIRED`` record what the gate fixed, not what is wrong.
    """

    RESOLUTION_BELOW_FLOOR = "RESOLUTION_BELOW_FLOOR"
    BLUR_ABOVE_CEILING = "BLUR_ABOVE_CEILING"
    SKEW_DETECTED = "SKEW_DETECTED"
    SKEW_REPAIRED = "SKEW_REPAIRED"
    SKEW_UNRECOVERABLE = "SKEW_UNRECOVERABLE"
    ROTATION_DETECTED = "ROTATION_DETECTED"
    ROTATION_REPAIRED = "ROTATION_REPAIRED"
    INVERTED_REPAIRED = "INVERTED_REPAIRED"
    TEXT_COVERAGE_BELOW_FLOOR = "TEXT_COVERAGE_BELOW_FLOOR"
    OCCLUSION_DETECTED = "OCCLUSION_DETECTED"
    SHADOW_DETECTED = "SHADOW_DETECTED"
    EXPOSURE_OUT_OF_RANGE = "EXPOSURE_OUT_OF_RANGE"
    NOISE_ABOVE_CEILING = "NOISE_ABOVE_CEILING"
    BLANK_PAGE = "BLANK_PAGE"
    DUPLICATE_PAGE = "DUPLICATE_PAGE"
    JUNK_TEXT_LAYER = "JUNK_TEXT_LAYER"


class BounceReason(StrEnum):
    """Why a document was refused, in terms an auto-reply can name (P1-2).

    Deliberately one flat vocabulary spanning ingestion rejections and
    quality-gate defects: the complainant receiving the bounce does not care
    which subsystem produced it, and the template lookup should not have to
    join two enums.  :data:`BOUNCE_REASON_FOR_REJECTION` and
    :data:`BOUNCE_REASON_FOR_QUALITY_FLAG` are the total/partial mappings in.
    """

    # --- ingestion (mirrors RejectionReason values one-for-one) ---
    EMPTY_FILE = "EMPTY_FILE"
    MALFORMED_PDF = "MALFORMED_PDF"
    MIME_MISMATCH = "MIME_MISMATCH"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"
    PAGE_CAP_EXCEEDED = "PAGE_CAP_EXCEEDED"
    PIXEL_CAP_EXCEEDED = "PIXEL_CAP_EXCEEDED"
    DECOMPRESSION_BOMB = "DECOMPRESSION_BOMB"
    WALL_CLOCK_EXCEEDED = "WALL_CLOCK_EXCEEDED"
    RESOLUTION_BELOW_FLOOR = "RESOLUTION_BELOW_FLOOR"
    BLANK_PAGE = "BLANK_PAGE"
    # --- quality gate ---
    BLUR_ABOVE_CEILING = "BLUR_ABOVE_CEILING"
    SKEW_UNRECOVERABLE = "SKEW_UNRECOVERABLE"
    TEXT_COVERAGE_BELOW_FLOOR = "TEXT_COVERAGE_BELOW_FLOOR"
    OCCLUSION_DETECTED = "OCCLUSION_DETECTED"
    QUALITY_SCORE_BELOW_THRESHOLD = "QUALITY_SCORE_BELOW_THRESHOLD"
    # --- document level ---
    ALL_PAGES_BOUNCED = "ALL_PAGES_BOUNCED"
    NO_TEXT_DETECTED = "NO_TEXT_DETECTED"


#: Total mapping — every ingestion rejection can be explained to a complainant.
#: ``tests/unit/test_errors.py`` asserts totality so a new reason cannot be
#: added without deciding what the bounce message says.
BOUNCE_REASON_FOR_REJECTION: dict[RejectionReason, BounceReason] = {
    RejectionReason.EMPTY_FILE: BounceReason.EMPTY_FILE,
    RejectionReason.MALFORMED_PDF: BounceReason.MALFORMED_PDF,
    RejectionReason.MIME_MISMATCH: BounceReason.MIME_MISMATCH,
    RejectionReason.ENCRYPTED_PDF: BounceReason.ENCRYPTED_PDF,
    RejectionReason.PAGE_CAP_EXCEEDED: BounceReason.PAGE_CAP_EXCEEDED,
    RejectionReason.PIXEL_CAP_EXCEEDED: BounceReason.PIXEL_CAP_EXCEEDED,
    RejectionReason.DECOMPRESSION_BOMB: BounceReason.DECOMPRESSION_BOMB,
    RejectionReason.WALL_CLOCK_EXCEEDED: BounceReason.WALL_CLOCK_EXCEEDED,
    RejectionReason.RESOLUTION_BELOW_FLOOR: BounceReason.RESOLUTION_BELOW_FLOOR,
    RejectionReason.BLANK_PAGE: BounceReason.BLANK_PAGE,
}

#: Partial by design: ``*_REPAIRED`` and advisory flags never bounce anything.
BOUNCE_REASON_FOR_QUALITY_FLAG: dict[QualityFlag, BounceReason] = {
    QualityFlag.RESOLUTION_BELOW_FLOOR: BounceReason.RESOLUTION_BELOW_FLOOR,
    QualityFlag.BLUR_ABOVE_CEILING: BounceReason.BLUR_ABOVE_CEILING,
    QualityFlag.SKEW_UNRECOVERABLE: BounceReason.SKEW_UNRECOVERABLE,
    QualityFlag.TEXT_COVERAGE_BELOW_FLOOR: BounceReason.TEXT_COVERAGE_BELOW_FLOOR,
    QualityFlag.OCCLUSION_DETECTED: BounceReason.OCCLUSION_DETECTED,
    QualityFlag.BLANK_PAGE: BounceReason.BLANK_PAGE,
}


class FieldFlag(StrEnum):
    """Why a field needs a human, in machine-readable form.

    Extension to BUILD-BRIEF sec. 3.3: invariants 8 and 9 both say a value is
    "flagged", and a free-text note is not something a review queue can filter
    on.  ``VALIDATOR_FAILED`` is attached automatically whenever
    ``validator_passed is False`` so invariant 9 cannot be forgotten.
    """

    VALIDATOR_FAILED = "VALIDATOR_FAILED"
    CHECKSUM_FAILED = "CHECKSUM_FAILED"
    GROUNDING_REJECTED = "GROUNDING_REJECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    BANK_NAME_IFSC_DISAGREEMENT = "BANK_NAME_IFSC_DISAGREEMENT"
    IFSC_NOT_IN_MASTER = "IFSC_NOT_IN_MASTER"
    MULTIPLE_CANDIDATES = "MULTIPLE_CANDIDATES"
    ESCALATED_PAGE = "ESCALATED_PAGE"
    TRANSLATION_UNAVAILABLE = "TRANSLATION_UNAVAILABLE"
    HANDWRITTEN_SOURCE = "HANDWRITTEN_SOURCE"


class Evidence(BaseModel):
    """Where a value physically is: page, box, character span.

    Never optional on a :class:`Field`.  A value we cannot point at is a value
    a case officer cannot check, which is the failure mode this whole service
    exists to remove.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_no: PageNumber
    bbox: BBox
    span: TextSpan


class Field(BaseModel):
    """One extracted value with its full provenance — BUILD-BRIEF sec. 3.3.

    Frozen: ``value`` is immutable by construction (invariant 10).  Use
    :meth:`with_translation` and :meth:`with_flags`, which return revalidated
    copies, rather than mutating.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Native script, verbatim from OCR. IMMUTABLE and authoritative.
    value: str | None
    #: Advisory English rendering. Never key business logic on this.
    value_en: str | None = None
    #: Canonical form for indexing/linking (P2-4) — e.g. digits-only account.
    normalized: str | None = None
    confidence: Confidence
    source: FieldSource
    #: ``None`` when no validator applies to this field.
    validator_passed: bool | None = None
    #: Mandatory. There is no constructor path that omits it.
    evidence: Evidence
    #: Required whenever ``value_en`` is set.
    mt_engine: str | None = None
    flags: tuple[FieldFlag, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _always_flag_validator_failure(cls, data: Any) -> Any:
        """Attach ``VALIDATOR_FAILED`` whenever ``validator_passed is False``.

        This is derivation from a fact already declared on the same object, not
        a silent default: PLAN.md invariant 9 says a failed check is *always*
        flagged and never silently corrected, so the flag is guaranteed here
        rather than trusted to every call site.  ``value`` is never touched.
        """
        if isinstance(data, dict) and data.get("validator_passed") is False:
            flags = list(data.get("flags") or ())
            if FieldFlag.VALIDATOR_FAILED not in flags:
                flags.append(FieldFlag.VALIDATOR_FAILED)
            data = {**data, "flags": tuple(flags)}
        return data

    @model_validator(mode="after")
    def _provenance_rules(self) -> Self:
        if self.value_en is not None and not self.mt_engine:
            raise ValueError(
                "value_en is advisory output of a machine-translation engine "
                "and requires mt_engine (PLAN.md invariant 10)"
            )
        if self.mt_engine is not None and self.value_en is None:
            raise ValueError("mt_engine set without value_en: nothing was translated")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError(f"duplicate flags: {self.flags}")
        return self

    @property
    def needs_review(self) -> bool:
        """True when any flag is present — the review queue's filter."""
        return bool(self.flags)

    def with_translation(self, value_en: str, mt_engine: str) -> Field:
        """Return a revalidated copy carrying an advisory translation.

        ``value`` is copied through untouched; there is no code path in which
        translation can alter the native-script value (invariant 10).
        """
        data = self.model_dump()
        data["value_en"] = value_en
        data["mt_engine"] = mt_engine
        return Field.model_validate(data)

    def with_flags(self, *flags: FieldFlag) -> Field:
        """Return a revalidated copy with ``flags`` added (idempotent)."""
        merged = list(self.flags)
        for flag in flags:
            if flag not in merged:
                merged.append(flag)
        data = self.model_dump()
        data["flags"] = tuple(merged)
        return Field.model_validate(data)


class SummarySpan(BaseModel):
    """One selected OCR line span and its **verbatim** text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: Evidence
    #: Exactly the OCR text of the referenced span. Not paraphrased, not fixed.
    text: str


class ComplaintSummary(BaseModel):
    """Extractive summary — PLAN.md invariant 2, enforced not asserted.

    The model returns line indices; the assembler concatenates the verbatim OCR
    text of those lines.  :meth:`_verbatim_concatenation` re-derives ``text``
    from ``spans`` and rejects any mismatch, so a prose summary produced by a
    model that ignored its instructions cannot be represented in the envelope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    spans: tuple[SummarySpan, ...] = _Field(min_length=1)
    separator: str = "\n"

    @model_validator(mode="after")
    def _verbatim_concatenation(self) -> Self:
        expected = self.separator.join(span.text for span in self.spans)
        if self.text != expected:
            raise ValueError(
                "complaint_summary.text must be the verbatim concatenation of "
                "the selected OCR spans — the model selects line indices, it "
                "never writes prose (PLAN.md invariant 2)"
            )
        return self


class PageResult(BaseModel):
    """Per-page record: quality, tier, and the derived language summary.

    This is the *only* place page-level language may appear.  It is derived in
    ``assemble.py`` from :class:`app.contracts.ocr.OCRLine` values; the OCR
    page model refuses to carry it (invariant 1).
    """

    model_config = ConfigDict(extra="forbid")

    page_no: PageNumber
    #: Present on every page including accepted ones (PRD P0-3).
    quality_score: Confidence
    quality_flags: tuple[QualityFlag, ...] = ()
    quality_verdict: QualityVerdict
    ocr_tier: OCRTier
    line_count: int = _Field(ge=0)
    # --- derived summaries (assemble.py only) ---
    primary_script: ScriptCode | None = None
    primary_language: LanguageCode | None = None
    #: ISO 639-3 -> share of lines. Values sum to ~1 when non-empty.
    language_distribution: dict[str, float] = _Field(default_factory=dict)
    #: True when >= CODE_MIX_THRESHOLD of lines differ in script from the page
    #: majority (PRD P0-5). Computed in assemble.py; recorded here.
    is_code_mixed: bool = False
    # --- routing ---
    escalated: bool = False
    #: Ran through Tier 2 as part of the 3% shadow sample, *not* an escalation
    #: (PLAN.md invariant 3). Distinguished so shadow runs never inflate the
    #: escalation rate they exist to measure.
    shadow_sampled: bool = False
    repairs_applied: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _derived_summary_is_consistent(self) -> Self:
        check_distribution(
            self.language_distribution, "PageResult.language_distribution"
        )
        for code in self.language_distribution:
            if len(code) != 3 or not code.isalpha() or not code.islower():
                raise ValueError(
                    f"language_distribution keys must be ISO 639-3 codes, got {code!r}"
                )
        if self.primary_language is not None and self.language_distribution:
            if self.primary_language not in self.language_distribution:
                raise ValueError(
                    f"primary_language {self.primary_language!r} absent from "
                    f"language_distribution {sorted(self.language_distribution)}"
                )
            best = max(self.language_distribution.values())
            if not math.isclose(
                self.language_distribution[self.primary_language], best, rel_tol=1e-9
            ):
                raise ValueError(
                    "primary_language must be the argmax of "
                    "language_distribution; it is a derived summary, not an "
                    "independent judgement (PLAN.md invariant 1)"
                )
        if self.quality_verdict is QualityVerdict.BOUNCE and self.line_count:
            raise ValueError(
                "a bounced page must not report recognised lines: nothing runs "
                "after a bounce"
            )
        if self.ocr_tier is OCRTier.NONE and self.line_count:
            raise ValueError(f"ocr_tier=none contradicts line_count={self.line_count}")
        return self


class DocumentResult(BaseModel):
    """The extracted content of one document — fields, class, summary."""

    model_config = ConfigDict(extra="forbid")

    document_class: PageClassLabel
    document_class_confidence: Confidence
    #: Keyed by field name; every value carries its own provenance.
    fields: dict[str, Field] = _Field(default_factory=dict)
    #: Requested names map to zero or more distinct evidence-bearing values.
    extracted_fields: dict[str, list[Field]] = _Field(default_factory=dict)
    other_fields: dict[str, Field] = _Field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    complaint_summary: ComplaintSummary | None = None
    #: Closed tag set for supporting documents (PRD P1-3); no field extraction.
    supporting_doc_tags: tuple[str, ...] = ()
    page_count: int = _Field(ge=0)
    # --- derived document-level summaries (assemble.py only) ---
    primary_script: ScriptCode | None = None
    primary_language: LanguageCode | None = None
    language_distribution: dict[str, float] = _Field(default_factory=dict)
    is_code_mixed: bool = False

    @model_validator(mode="after")
    def _field_provenance_rules(self) -> Self:
        for name in self.fields:
            if not name:
                raise ValueError("field names must be non-empty")
        check_distribution(
            self.language_distribution, "DocumentResult.language_distribution"
        )
        if (
            self.primary_language is not None
            and self.language_distribution
            and self.primary_language not in self.language_distribution
        ):
            raise ValueError(
                f"primary_language {self.primary_language!r} absent from "
                "language_distribution"
            )
        return self


class BounceDecision(BaseModel):
    """Accept / repair / bounce with a **machine-readable** reason (PRD P0-3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: QualityVerdict
    #: Required exactly when ``decision`` is ``bounce``; forbidden otherwise.
    reason: BounceReason | None = None
    #: Human-readable elaboration for the auto-reply. Never parsed.
    detail: str | None = None

    @model_validator(mode="after")
    def _reason_matches_decision(self) -> Self:
        if self.decision is QualityVerdict.BOUNCE and self.reason is None:
            raise ValueError(
                "a bounce must name a machine-readable reason — the auto-reply "
                "template is keyed on it (PRD P0-3, P1-2)"
            )
        if self.decision is not QualityVerdict.BOUNCE and self.reason is not None:
            raise ValueError(
                f"decision={self.decision.value!r} must not carry a bounce "
                f"reason ({self.reason.value!r})"
            )
        return self

    @classmethod
    def accepted(cls) -> BounceDecision:
        """The common case, spelled once."""
        return cls(decision=QualityVerdict.ACCEPT)

    @classmethod
    def from_rejection(
        cls, reason: RejectionReason, detail: str | None = None
    ) -> BounceDecision:
        """Build a bounce from an ingestion :class:`RejectionReason`."""
        return cls(
            decision=QualityVerdict.BOUNCE,
            reason=BOUNCE_REASON_FOR_REJECTION[reason],
            detail=detail,
        )


class ProcessingStats(BaseModel):
    """Counters the eval harness and Prometheus read; never business data."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = _Field(default=None, ge=0.0)
    pages_processed: int = _Field(default=0, ge=0)
    pages_bounced: int = _Field(default=0, ge=0)
    pages_escalated: int = _Field(default=0, ge=0)
    tokens_used: int = _Field(default=0, ge=0)
    #: Wall-clock per stage, for the p50/p95 cut in TEST-DATA-SPEC sec. 7.
    stage_timings_ms: dict[str, float] = _Field(default_factory=dict)


class ResultEnvelope(BaseModel):
    """The versioned, evidence-bearing result — PRD P0-10.

    ``result_version`` starts at 1 and increments as escalated pages land or
    a changed requested-field list triggers another extraction;
    ``is_final`` is true only on the last one.  A consumer may act on version 1
    and reconcile later.
    """

    model_config = ConfigDict(extra="forbid")

    #: Frozen literal: a breaking change requires a new major and a new file in
    #: ``docs/schema/``.
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    result_version: int = _Field(ge=1)
    is_final: bool
    document_id: str = _Field(min_length=1)
    job_id: str | None = None
    requested_fields: tuple[str, ...] = ()
    created_at: datetime = _Field(default_factory=lambda: datetime.now(UTC))
    #: ``laptop`` | ``cpu_node`` | ``gpu_node`` — recorded so a divergence
    #: between profiles is attributable (BUILD-BRIEF M6).
    provider_profile: str = _Field(min_length=1)
    bounce_decision: BounceDecision
    #: Document needs a human because a page escalated or a field is flagged.
    escalation_flag: bool = False
    pages: tuple[PageResult, ...] = ()
    #: ``None`` only when the document bounced — never an empty stand-in.
    document: DocumentResult | None = None
    processing: ProcessingStats = _Field(default_factory=ProcessingStats)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _envelope_consistency(self) -> Self:
        from app.field_selection import parse_requested_fields, partition_fields

        self.requested_fields = parse_requested_fields(self.requested_fields)
        if self.document is not None:
            extracted, other, missing = partition_fields(
                self.document.fields, self.requested_fields
            )
            self.document.extracted_fields = extracted
            self.document.other_fields = other
            self.document.missing_fields = missing
        seen: set[int] = set()
        previous = 0
        for page in self.pages:
            if page.page_no in seen:
                raise ValueError(f"duplicate page_no {page.page_no}")
            if page.page_no < previous:
                raise ValueError(
                    "pages must be ordered by ascending page_no, got "
                    f"{page.page_no} after {previous}"
                )
            seen.add(page.page_no)
            previous = page.page_no

        bounced = self.bounce_decision.decision is QualityVerdict.BOUNCE
        if self.document is None and not bounced:
            raise ValueError(
                "document may be None only when bounce_decision.decision is "
                "'bounce' — a missing result must say why, never arrive empty "
                "(BUILD-BRIEF sec. 0.5)"
            )
        if bounced and self.document is not None:
            raise ValueError("a bounced document must not carry extracted content")
        if any(page.escalated for page in self.pages) and not self.escalation_flag:
            raise ValueError(
                "escalation_flag must be true when any page escalated; shadow "
                "sampling uses PageResult.shadow_sampled instead"
            )
        if self.document is not None and self.document.page_count != len(self.pages):
            raise ValueError(
                f"document.page_count={self.document.page_count} contradicts "
                f"{len(self.pages)} page records"
            )
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return self
