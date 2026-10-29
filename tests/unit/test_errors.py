"""Contract tests for the typed error taxonomy (PLAN.md invariant 11).

Covers: the code/context contract on the base exception, one bound class per
rejection reason (TEST-DATA-SPEC sec. 5 + BUILD-BRIEF M3 caps), the provider
family, ``ResidencyViolation``, and totality of the bounce-reason mapping.
"""

from __future__ import annotations

import pytest

from app.contracts.envelope import (
    BOUNCE_REASON_FOR_QUALITY_FLAG,
    BOUNCE_REASON_FOR_REJECTION,
    BounceReason,
    QualityFlag,
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

ALL_ERROR_CLASSES: tuple[type[PipelineError], ...] = (
    PipelineError,
    DocumentRejected,
    EmptyFileError,
    MalformedPDFError,
    MimeMismatchError,
    ResolutionBelowFloorError,
    BlankPageError,
    PageCapExceededError,
    PixelCapExceededError,
    DecompressionBombError,
    WallClockExceededError,
    EncryptedPDFError,
    ProviderError,
    ProviderRateLimited,
    ProviderUnavailable,
    ProviderResponseInvalid,
    TokenBudgetExceeded,
    ConfigurationError,
    ResidencyViolation,
)


# --------------------------------------------------------------------------- #
# Base contract
# --------------------------------------------------------------------------- #


def test_base_error_carries_code_message_and_context() -> None:
    err = PipelineError("boom", context={"page_no": 3})
    assert err.code == "PIPELINE_ERROR"
    assert err.message == "boom"
    assert err.context == {"page_no": 3}
    assert err.to_dict() == {
        "code": "PIPELINE_ERROR",
        "message": "boom",
        "context": {"page_no": 3},
    }


def test_context_defaults_are_not_shared_between_instances() -> None:
    first = PipelineError("a")
    second = PipelineError("b")
    first.context["leak"] = True
    assert second.context == {}


def test_to_dict_returns_a_copy_of_context() -> None:
    err = PipelineError("a", context={"k": 1})
    err.to_dict()["context"]["k"] = 999
    assert err.context == {"k": 1}


def test_every_error_class_has_a_unique_code() -> None:
    codes = [cls.code for cls in ALL_ERROR_CLASSES]
    assert len(codes) == len(set(codes)), "error codes must be unique"


def test_all_errors_are_pipeline_errors_and_exceptions() -> None:
    for cls in ALL_ERROR_CLASSES:
        assert issubclass(cls, PipelineError)
        assert issubclass(cls, Exception)


def test_repr_is_informative_without_leaking_context() -> None:
    err = MimeMismatchError("bad magic", context={"sniffed": "image/gif"})
    assert "MimeMismatchError" in repr(err)
    assert "MIME_MISMATCH" in repr(err)
    assert "image/gif" not in repr(err)


# --------------------------------------------------------------------------- #
# Rejection taxonomy
# --------------------------------------------------------------------------- #


def test_every_rejection_reason_has_exactly_one_bound_exception() -> None:
    missing = [r for r in RejectionReason if r not in REJECTION_EXCEPTIONS]
    assert missing == [], f"rejection reasons without an exception: {missing}"
    assert len(REJECTION_EXCEPTIONS) == len(list(RejectionReason))


@pytest.mark.parametrize("reason", list(RejectionReason))
def test_rejection_code_equals_reason_value(reason: RejectionReason) -> None:
    cls = exception_for(reason)
    assert cls.code == reason.value
    assert cls.reason is reason
    assert cls.http_status == 422


@pytest.mark.parametrize("reason", list(RejectionReason))
def test_rejection_records_its_reason_in_context(reason: RejectionReason) -> None:
    err = exception_for(reason)("nope", context={"path": "x.pdf"})
    assert err.context["reason"] == reason.value
    assert err.context["path"] == "x.pdf"


def test_caller_supplied_reason_context_is_not_overwritten() -> None:
    err = EmptyFileError("nope", context={"reason": "custom"})
    assert err.context["reason"] == "custom"


@pytest.mark.parametrize(
    ("cls", "reason"),
    [
        (EmptyFileError, RejectionReason.EMPTY_FILE),
        (MalformedPDFError, RejectionReason.MALFORMED_PDF),
        (MimeMismatchError, RejectionReason.MIME_MISMATCH),
        (ResolutionBelowFloorError, RejectionReason.RESOLUTION_BELOW_FLOOR),
        (BlankPageError, RejectionReason.BLANK_PAGE),
        (PageCapExceededError, RejectionReason.PAGE_CAP_EXCEEDED),
        (PixelCapExceededError, RejectionReason.PIXEL_CAP_EXCEEDED),
        (DecompressionBombError, RejectionReason.DECOMPRESSION_BOMB),
        (WallClockExceededError, RejectionReason.WALL_CLOCK_EXCEEDED),
        (EncryptedPDFError, RejectionReason.ENCRYPTED_PDF),
    ],
)
def test_adversarial_inputs_map_to_typed_rejections(
    cls: type[DocumentRejected], reason: RejectionReason
) -> None:
    """TEST-DATA-SPEC sec. 5: the pass condition is a typed rejection."""
    with pytest.raises(DocumentRejected) as excinfo:
        raise cls("rejected")
    assert excinfo.value.code == reason.value
    assert isinstance(excinfo.value, cls)


def test_encrypted_document_is_an_alias_not_a_second_code() -> None:
    """TEST-DATA-SPEC spells it ENCRYPTED_DOCUMENT; the brief, ENCRYPTED_PDF."""
    assert RejectionReason.ENCRYPTED_DOCUMENT is RejectionReason.ENCRYPTED_PDF
    assert RejectionReason("ENCRYPTED_PDF") is RejectionReason.ENCRYPTED_PDF


def test_exception_for_unknown_reason_raises_key_error() -> None:
    REJECTION_EXCEPTIONS.pop(RejectionReason.BLANK_PAGE)
    try:
        with pytest.raises(KeyError):
            exception_for(RejectionReason.BLANK_PAGE)
    finally:
        REJECTION_EXCEPTIONS[RejectionReason.BLANK_PAGE] = BlankPageError


# --------------------------------------------------------------------------- #
# Provider family
# --------------------------------------------------------------------------- #


def test_rate_limited_carries_retry_after() -> None:
    err = ProviderRateLimited("429", retry_after=1.5)
    assert err.retry_after == 1.5
    assert err.context["retry_after"] == 1.5
    assert err.http_status == 429
    assert isinstance(err, ProviderError)


def test_rate_limited_without_retry_after_omits_the_key() -> None:
    err = ProviderRateLimited("429")
    assert err.retry_after is None
    assert "retry_after" not in err.context


def test_token_budget_exceeded_carries_budget_and_usage() -> None:
    err = TokenBudgetExceeded("over", budget=8000, used=8123)
    assert (err.budget, err.used) == (8000, 8123)
    assert err.context == {"budget": 8000, "used": 8123}


def test_provider_errors_share_a_catchable_base() -> None:
    for cls in (
        ProviderRateLimited,
        ProviderUnavailable,
        ProviderResponseInvalid,
        TokenBudgetExceeded,
    ):
        with pytest.raises(ProviderError):
            raise cls("x")


# --------------------------------------------------------------------------- #
# Residency
# --------------------------------------------------------------------------- #


def test_residency_violation_is_a_configuration_error() -> None:
    err = ResidencyViolation(
        "laptop profile routes text to a non-Indian endpoint; "
        "permitted only with SYNTHETIC_ONLY=true"
    )
    assert isinstance(err, ConfigurationError)
    assert isinstance(err, PipelineError)
    assert err.code == "RESIDENCY_VIOLATION"


def test_residency_violation_is_not_catchable_as_a_provider_error() -> None:
    """It is a policy failure at startup, not a transient provider fault."""
    assert not issubclass(ResidencyViolation, ProviderError)
    with pytest.raises(ResidencyViolation):
        try:
            raise ResidencyViolation("nope")
        except ProviderError:  # pragma: no cover - must not be reached
            pytest.fail("ResidencyViolation must not be swallowed by retry logic")


# --------------------------------------------------------------------------- #
# Bounce mapping (auto-reply keys, PRD P1-2)
# --------------------------------------------------------------------------- #


def test_every_rejection_reason_maps_to_a_bounce_reason() -> None:
    missing = [r for r in RejectionReason if r not in BOUNCE_REASON_FOR_REJECTION]
    assert missing == [], f"unmapped rejection reasons: {missing}"


@pytest.mark.parametrize("reason", list(RejectionReason))
def test_bounce_mapping_preserves_the_wire_code(reason: RejectionReason) -> None:
    assert BOUNCE_REASON_FOR_REJECTION[reason].value == reason.value


def test_quality_flag_bounce_mapping_is_partial_but_valid() -> None:
    for flag, bounce in BOUNCE_REASON_FOR_QUALITY_FLAG.items():
        assert isinstance(flag, QualityFlag)
        assert isinstance(bounce, BounceReason)
    repaired = [f for f in QualityFlag if f.value.endswith("_REPAIRED")]
    assert repaired, "sanity: there are repair flags"
    for flag in repaired:
        assert flag not in BOUNCE_REASON_FOR_QUALITY_FLAG, (
            f"{flag} records a successful repair and must never bounce"
        )
