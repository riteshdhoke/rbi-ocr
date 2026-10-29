"""Typed error taxonomy for the whole service (PLAN.md invariant 11).

Every boundary — ingestion, quality gate, provider, config — raises a subclass
of :class:`PipelineError`, which carries a **machine-readable ``code``** and a
**``context`` dict**.  The code is what the API layer, the auto-reply templates
(WS-G) and the eval harness key on; the human message is never parsed.

Three families live here:

``DocumentRejected``
    One class per :class:`RejectionReason`, covering the TEST-DATA-SPEC sec. 5
    adversarial set plus the BUILD-BRIEF M3 resource ceilings.  The pass
    condition for every adversarial input is one of these, never a crashed
    worker.

``ProviderError``
    Raised by the provider layer (WS-C): rate limits, unavailability, malformed
    model output, per-job token-budget breach.

``ConfigurationError`` / ``ResidencyViolation``
    Startup-time policy failures.  ``ResidencyViolation`` is the one the system
    cannot recover from — complainant data leaving the country during
    development — and is raised by the provider factory, never bypassed.

This module does NOT handle:
  * **quality-gate flags and bounce reasons** — those are envelope *data*, not
    exceptions, and live in :mod:`app.contracts.envelope` (``QualityFlag``,
    ``BounceReason``).  A bounced document produces a valid envelope, not a
    raised error;
  * HTTP responses — ``http_status`` is a hint the API layer may use; this
    module imports no web framework;
  * logging or PII redaction — ``context`` may contain a filename or a page
    number, and callers must not put OCR text or identifiers in it;
  * retry policy — ``ProviderRateLimited.retry_after`` is data, the backoff
    lives in the provider;
  * validation failures of extracted *fields*.  A failed checksum is
    ``validator_passed=False`` plus a flag on the field (PLAN.md invariant 9),
    never an exception — flagging must not abort the document.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar

__all__ = [
    "REJECTION_EXCEPTIONS",
    "AppendOnlyViolation",
    "BlankPageError",
    "ConfigurationError",
    "DecompressionBombError",
    "DocumentRejected",
    "EmptyFileError",
    "EncryptedPDFError",
    "JobNotFound",
    "MalformedPDFError",
    "MalwareDetected",
    "MimeMismatchError",
    "PageCapExceededError",
    "PipelineError",
    "PixelCapExceededError",
    "ProviderError",
    "ProviderRateLimited",
    "ProviderResponseInvalid",
    "ProviderUnavailable",
    "RejectionReason",
    "ResidencyViolation",
    "ResolutionBelowFloorError",
    "StorageError",
    "TokenBudgetExceeded",
    "WallClockExceededError",
    "WebhookDeliveryFailed",
    "exception_for",
]


class RejectionReason(StrEnum):
    """Machine-readable reasons a document or page is refused at ingestion.

    The first five are the committed adversarial inputs of TEST-DATA-SPEC
    sec. 5; the rest are the BUILD-BRIEF M3 hard resource ceilings plus the
    encrypted-PDF case (generated locally, never committed).
    """

    EMPTY_FILE = "EMPTY_FILE"
    MALFORMED_PDF = "MALFORMED_PDF"
    MIME_MISMATCH = "MIME_MISMATCH"
    RESOLUTION_BELOW_FLOOR = "RESOLUTION_BELOW_FLOOR"
    BLANK_PAGE = "BLANK_PAGE"
    PAGE_CAP_EXCEEDED = "PAGE_CAP_EXCEEDED"
    PIXEL_CAP_EXCEEDED = "PIXEL_CAP_EXCEEDED"
    DECOMPRESSION_BOMB = "DECOMPRESSION_BOMB"
    WALL_CLOCK_EXCEEDED = "WALL_CLOCK_EXCEEDED"
    ENCRYPTED_PDF = "ENCRYPTED_PDF"

    #: TEST-DATA-SPEC sec. 5 spells this ``ENCRYPTED_DOCUMENT``; the task brief
    #: spells it ``ENCRYPTED_PDF``.  Kept as an *alias* (identical value) so
    #: either name resolves to one member and the wire code stays single-valued.
    ENCRYPTED_DOCUMENT = "ENCRYPTED_PDF"


class PipelineError(Exception):
    """Base of every typed error in the service.

    ``code`` is the stable machine-readable identifier; ``context`` is a small
    JSON-safe dict of non-sensitive facts (path, page number, limit, observed
    value).  Never place OCR text, an account number or any complainant
    identifier in ``context`` — it is logged.
    """

    code: ClassVar[str] = "PIPELINE_ERROR"
    #: Advisory mapping for the API layer.  This module never imports HTTP code.
    http_status: ClassVar[int] = 500

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, object] = dict(context or {})

    def to_dict(self) -> dict[str, object]:
        """Serialise to the shape the API and audit log emit."""
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------- #
# Ingestion / resource-cap rejections
# --------------------------------------------------------------------------- #

#: Populated by :func:`_register`; one concrete class per reason.
REJECTION_EXCEPTIONS: dict[RejectionReason, type[DocumentRejected]] = {}


class DocumentRejected(PipelineError):
    """A document or page is refused before (or instead of) model work.

    Subclasses bind exactly one :class:`RejectionReason`; ``code`` equals that
    reason's value so the wire format has a single vocabulary.
    """

    code: ClassVar[str] = "DOCUMENT_REJECTED"
    http_status: ClassVar[int] = 422
    #: Bound by every concrete subclass.
    reason: ClassVar[RejectionReason]

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.context.setdefault("reason", str(self.reason))


def _register(cls: type[DocumentRejected]) -> type[DocumentRejected]:
    """Bind a concrete rejection class into :data:`REJECTION_EXCEPTIONS`."""
    existing = REJECTION_EXCEPTIONS.get(cls.reason)
    if existing is not None:
        raise RuntimeError(
            f"duplicate rejection class for {cls.reason}: "
            f"{existing.__name__} and {cls.__name__}"
        )
    REJECTION_EXCEPTIONS[cls.reason] = cls
    return cls


@_register
class EmptyFileError(DocumentRejected):
    """Zero-byte attachment (``zero_byte.pdf``)."""

    reason: ClassVar[RejectionReason] = RejectionReason.EMPTY_FILE
    code: ClassVar[str] = "EMPTY_FILE"


@_register
class MalformedPDFError(DocumentRejected):
    """Truncated or structurally invalid PDF (``truncated.pdf``)."""

    reason: ClassVar[RejectionReason] = RejectionReason.MALFORMED_PDF
    code: ClassVar[str] = "MALFORMED_PDF"


@_register
class MimeMismatchError(DocumentRejected):
    """Magic bytes disagree with the declared type (GIF/HTML named ``.pdf``).

    Raised *before* any content is interpreted — the disguised payload is never
    parsed, scripted or rendered.
    """

    reason: ClassVar[RejectionReason] = RejectionReason.MIME_MISMATCH
    code: ClassVar[str] = "MIME_MISMATCH"


@_register
class ResolutionBelowFloorError(DocumentRejected):
    """Effective resolution below the readability floor (``tiny_lowres.png``)."""

    reason: ClassVar[RejectionReason] = RejectionReason.RESOLUTION_BELOW_FLOOR
    code: ClassVar[str] = "RESOLUTION_BELOW_FLOOR"


@_register
class BlankPageError(DocumentRejected):
    """Page carries no ink; dropped without billing OCR (``blank_page.png``)."""

    reason: ClassVar[RejectionReason] = RejectionReason.BLANK_PAGE
    code: ClassVar[str] = "BLANK_PAGE"


@_register
class PageCapExceededError(DocumentRejected):
    """More pages than ``MAX_PAGES_PER_DOC`` (default 200)."""

    reason: ClassVar[RejectionReason] = RejectionReason.PAGE_CAP_EXCEEDED
    code: ClassVar[str] = "PAGE_CAP_EXCEEDED"


@_register
class PixelCapExceededError(DocumentRejected):
    """Total rasterised pixels exceed ``MAX_TOTAL_PIXELS``."""

    reason: ClassVar[RejectionReason] = RejectionReason.PIXEL_CAP_EXCEEDED
    code: ClassVar[str] = "PIXEL_CAP_EXCEEDED"


@_register
class DecompressionBombError(DocumentRejected):
    """Decompressed:compressed ratio exceeds ``MAX_DECOMPRESSION_RATIO``."""

    reason: ClassVar[RejectionReason] = RejectionReason.DECOMPRESSION_BOMB
    code: ClassVar[str] = "DECOMPRESSION_BOMB"


@_register
class WallClockExceededError(DocumentRejected):
    """The document exceeded ``MAX_WALL_CLOCK_SECONDS`` in the sandbox."""

    reason: ClassVar[RejectionReason] = RejectionReason.WALL_CLOCK_EXCEEDED
    code: ClassVar[str] = "WALL_CLOCK_EXCEEDED"


@_register
class EncryptedPDFError(DocumentRejected):
    """Password-protected PDF; we never attempt to crack or guess."""

    reason: ClassVar[RejectionReason] = RejectionReason.ENCRYPTED_PDF
    code: ClassVar[str] = "ENCRYPTED_PDF"


def exception_for(reason: RejectionReason) -> type[DocumentRejected]:
    """Return the concrete exception class bound to ``reason``.

    Raises:
        KeyError: if a reason has no bound class — a build bug, caught by
            ``tests/unit/test_errors.py``, never a runtime surprise.
    """
    return REJECTION_EXCEPTIONS[reason]


# --------------------------------------------------------------------------- #
# Ingestion security (WS-E1)
# --------------------------------------------------------------------------- #


class MalwareDetected(PipelineError):
    """The malware scanner refused the attachment before anything parsed it.

    Deliberately **not** a :class:`DocumentRejected`.  Every member of that
    family is mirrored one-for-one into
    :class:`app.contracts.envelope.BounceReason` and therefore into a
    complainant-facing auto-reply (PRD P1-2), and an infected attachment must
    not produce a reply that tells the sender which signature fired: that is a
    free oracle for tuning the next attempt.  An infected document is
    quarantined and raised to the security queue; the sender sees the generic
    "could not be processed" path, not this code.

    ``signature`` and ``scanner`` are engine facts, never complainant data, so
    they are safe in ``context``.
    """

    code: ClassVar[str] = "MALWARE_DETECTED"
    http_status: ClassVar[int] = 422

    def __init__(
        self,
        message: str,
        *,
        signature: str | None = None,
        scanner: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.signature: str | None = signature
        self.scanner: str | None = scanner
        if signature is not None:
            self.context.setdefault("signature", signature)
        if scanner is not None:
            self.context.setdefault("scanner", scanner)


# --------------------------------------------------------------------------- #
# Provider layer (WS-C)
# --------------------------------------------------------------------------- #


class ProviderError(PipelineError):
    """Base for every failure originating in the model provider layer."""

    code: ClassVar[str] = "PROVIDER_ERROR"
    http_status: ClassVar[int] = 502


class ProviderRateLimited(ProviderError):
    """Upstream returned 429 / a quota signal.  Retryable with backoff."""

    code: ClassVar[str] = "PROVIDER_RATE_LIMITED"
    http_status: ClassVar[int] = 429

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.retry_after: float | None = retry_after
        if retry_after is not None:
            self.context.setdefault("retry_after", retry_after)


class ProviderUnavailable(ProviderError):
    """Endpoint unreachable, 5xx, timed out, or the local weights are missing."""

    code: ClassVar[str] = "PROVIDER_UNAVAILABLE"
    http_status: ClassVar[int] = 503


class ProviderResponseInvalid(ProviderError):
    """Model output failed schema validation after the one permitted repair retry.

    Extra to BUILD-BRIEF sec. 3, added here so WS-C need not reach into this
    lane: PRD P0-8 requires schema-validity to be a *recorded metric*, which
    needs a distinct error class to count.
    """

    code: ClassVar[str] = "PROVIDER_RESPONSE_INVALID"
    http_status: ClassVar[int] = 502


class TokenBudgetExceeded(ProviderError):
    """The per-job token budget (``JOB_TOKEN_BUDGET``) was exhausted."""

    code: ClassVar[str] = "TOKEN_BUDGET_EXCEEDED"
    http_status: ClassVar[int] = 429

    def __init__(
        self,
        message: str,
        *,
        budget: int | None = None,
        used: int | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.budget: int | None = budget
        self.used: int | None = used
        if budget is not None:
            self.context.setdefault("budget", budget)
        if used is not None:
            self.context.setdefault("used", used)


# --------------------------------------------------------------------------- #
# Configuration / policy
# --------------------------------------------------------------------------- #


class ConfigurationError(PipelineError):
    """Settings are internally inconsistent or incomplete for the active profile."""

    code: ClassVar[str] = "CONFIGURATION_ERROR"
    http_status: ClassVar[int] = 500


class StorageError(PipelineError):
    """A persistence-layer operation could not be completed (WS-F).

    Kept out of :class:`DocumentRejected` deliberately: a database failure is
    ours, not the complainant's, and must never produce an auto-reply telling
    the sender their attachment was at fault.
    """

    code: ClassVar[str] = "STORAGE_ERROR"
    http_status: ClassVar[int] = 500


class AppendOnlyViolation(StorageError):
    """Something tried to update or delete an append-only row.

    ``audit_events`` and ``results`` are the evidence trail: PRD P1-1 requires
    every field mutation to record actor, timestamp, before and after, and a
    trail that can be edited is not a trail.  The ORM refuses the flush rather
    than trusting a database grant to be right, so the rule holds on SQLite —
    which has no per-table permissions — exactly as it does on Postgres.
    """

    code: ClassVar[str] = "APPEND_ONLY_VIOLATION"
    http_status: ClassVar[int] = 409


class JobNotFound(StorageError):
    """No job exists with the requested identifier."""

    code: ClassVar[str] = "JOB_NOT_FOUND"
    http_status: ClassVar[int] = 404


class WebhookDeliveryFailed(PipelineError):
    """A signed result delivery exhausted its retries and went to the DLQ.

    Carries ``attempts`` and the last transport status so the replay tool can
    report why a row is parked without re-reading the response body — which may
    contain the receiver's own error text and is not ours to store.
    """

    code: ClassVar[str] = "WEBHOOK_DELIVERY_FAILED"
    http_status: ClassVar[int] = 502

    def __init__(
        self,
        message: str,
        *,
        attempts: int | None = None,
        status_code: int | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.attempts: int | None = attempts
        self.status_code: int | None = status_code
        if attempts is not None:
            self.context.setdefault("attempts", attempts)
        if status_code is not None:
            self.context.setdefault("status_code", status_code)


class ResidencyViolation(ConfigurationError):
    """Data residency would be breached by the requested configuration.

    Raised by the provider factory when ``PROVIDER_PROFILE=laptop`` and
    ``SYNTHETIC_ONLY=false`` (PLAN.md invariant 6), and by the corpus loader for
    any file lacking a ``synthetic: true`` sidecar under the laptop profile.
    **There is no bypass flag and none may be added.**
    """

    code: ClassVar[str] = "RESIDENCY_VIOLATION"
    http_status: ClassVar[int] = 500
