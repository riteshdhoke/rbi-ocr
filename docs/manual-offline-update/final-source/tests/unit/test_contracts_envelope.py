"""Contract tests for :mod:`app.contracts.envelope` — the published result.

Asserts the provenance guarantees a consumer relies on: evidence is mandatory,
``value`` is immutable, ``value_en`` cannot exist without ``mt_engine``, a
failed validator is always flagged, the extractive summary really is verbatim,
a bounce always names a machine-readable reason, and page-level language is a
derived summary that has to agree with its own distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.contracts.envelope import (
    SCHEMA_VERSION,
    BounceDecision,
    BounceReason,
    ComplaintSummary,
    DocumentResult,
    Evidence,
    Field,
    FieldFlag,
    PageResult,
    ProcessingStats,
    QualityFlag,
    QualityVerdict,
    ResultEnvelope,
    SummarySpan,
)
from app.contracts.errors import RejectionReason
from app.contracts.ocr import OCRTier, PageClassLabel, TextSpan, bbox_from_xyxy

# --------------------------------------------------------------------------- #
# Evidence — never optional
# --------------------------------------------------------------------------- #


def test_evidence_is_mandatory_on_every_field(
    make_evidence: Callable[..., Evidence],
) -> None:
    """A value we cannot point at is a value a case officer cannot check."""
    with pytest.raises(ValidationError) as excinfo:
        Field(
            value="BARB0417293",
            confidence=0.99,
            source="regex_validator",
        )
    assert "evidence" in str(excinfo.value)
    assert Field.model_fields["evidence"].is_required()


def test_evidence_rejects_a_null(make_field: Callable[..., Field]) -> None:
    with pytest.raises(ValidationError):
        make_field(evidence=None)


def test_evidence_carries_page_bbox_and_span(
    make_evidence: Callable[..., Evidence],
) -> None:
    evidence = make_evidence()
    assert evidence.page_no == 1
    assert len(evidence.bbox) == 8
    assert isinstance(evidence.span, TextSpan)


def test_evidence_is_frozen(make_evidence: Callable[..., Evidence]) -> None:
    evidence = make_evidence()
    with pytest.raises(ValidationError):
        evidence.page_no = 2  # type: ignore[misc]


def test_evidence_bbox_must_be_eight_floats(
    make_evidence: Callable[..., Evidence],
) -> None:
    with pytest.raises(ValidationError):
        make_evidence(bbox=[1.0, 2.0, 3.0, 4.0])


# --------------------------------------------------------------------------- #
# Field — provenance, immutability, translation
# --------------------------------------------------------------------------- #


def test_field_happy_path_and_json_round_trip(
    make_field: Callable[..., Field],
) -> None:
    field = make_field()
    assert Field.model_validate_json(field.model_dump_json()) == field


def test_value_is_immutable(make_field: Callable[..., Field]) -> None:
    """PLAN.md invariant 10: the native-script value can never be rewritten."""
    field = make_field()
    with pytest.raises(ValidationError):
        field.value = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize(
    "source",
    ["regex_validator", "lookup_table", "slm", "classifier", "human"],
)
def test_source_is_the_closed_brief_set(
    make_field: Callable[..., Field], source: str
) -> None:
    assert make_field(source=source).source == source


def test_unknown_source_is_rejected(make_field: Callable[..., Field]) -> None:
    with pytest.raises(ValidationError):
        make_field(source="vibes")


def test_value_en_requires_an_mt_engine(make_field: Callable[..., Field]) -> None:
    with pytest.raises(ValidationError, match="invariant 10"):
        make_field(value_en="Bank of Baroda")


def test_mt_engine_without_value_en_is_rejected(
    make_field: Callable[..., Field],
) -> None:
    with pytest.raises(ValidationError, match="nothing was translated"):
        make_field(mt_engine="indictrans2-dist-200M-int8")


def test_with_translation_preserves_the_native_value(
    make_field: Callable[..., Field],
) -> None:
    field = make_field(value="बँक ऑफ बडोदा", source="slm", validator_passed=None)
    translated = field.with_translation("Bank of Baroda", "indictrans2-dist-200M")
    assert translated.value == "बँक ऑफ बडोदा"
    assert field.value_en is None, "the original must not be mutated"
    assert translated.value_en == "Bank of Baroda"
    assert translated.mt_engine == "indictrans2-dist-200M"


def test_failed_validator_is_always_flagged(
    make_field: Callable[..., Field],
) -> None:
    """PLAN.md invariant 9 — flagged, never silently corrected."""
    field = make_field(value="12345678901234", validator_passed=False)
    assert FieldFlag.VALIDATOR_FAILED in field.flags
    assert field.needs_review is True
    assert field.value == "12345678901234", "the value is not repaired"


def test_failed_validator_flag_is_not_duplicated(
    make_field: Callable[..., Field],
) -> None:
    field = make_field(
        validator_passed=False,
        flags=(FieldFlag.VALIDATOR_FAILED, FieldFlag.CHECKSUM_FAILED),
    )
    assert field.flags.count(FieldFlag.VALIDATOR_FAILED) == 1
    assert len(field.flags) == 2


def test_duplicate_flags_are_rejected(make_field: Callable[..., Field]) -> None:
    with pytest.raises(ValidationError, match="duplicate flags"):
        make_field(flags=(FieldFlag.LOW_CONFIDENCE, FieldFlag.LOW_CONFIDENCE))


def test_with_flags_is_idempotent(make_field: Callable[..., Field]) -> None:
    field = make_field().with_flags(FieldFlag.GROUNDING_REJECTED)
    again = field.with_flags(FieldFlag.GROUNDING_REJECTED, FieldFlag.LOW_CONFIDENCE)
    assert again.flags == (FieldFlag.GROUNDING_REJECTED, FieldFlag.LOW_CONFIDENCE)
    assert field.flags == (FieldFlag.GROUNDING_REJECTED,)


def test_passing_field_needs_no_review(make_field: Callable[..., Field]) -> None:
    assert make_field().needs_review is False


def test_field_forbids_unknown_keys(make_field: Callable[..., Field]) -> None:
    with pytest.raises(ValidationError):
        make_field(reviewed_by="someone")


# --------------------------------------------------------------------------- #
# ComplaintSummary — invariant 2
# --------------------------------------------------------------------------- #


def test_summary_is_the_verbatim_concatenation(
    make_summary: Callable[..., ComplaintSummary],
) -> None:
    summary = make_summary()
    assert summary.text == "The bank debited twice\non 14 January 2026."
    assert len(summary.spans) == 2


def test_model_written_prose_cannot_be_represented(
    make_summary: Callable[..., ComplaintSummary],
) -> None:
    """PLAN.md invariant 2, enforced rather than asserted."""
    with pytest.raises(ValidationError, match="invariant 2"):
        make_summary(text="The complainant reports a double debit.")


def test_summary_requires_at_least_one_span(
    make_evidence: Callable[..., Evidence],
) -> None:
    with pytest.raises(ValidationError):
        ComplaintSummary(text="", spans=())


def test_summary_is_frozen_and_round_trips(
    make_summary: Callable[..., ComplaintSummary],
) -> None:
    summary = make_summary()
    assert ComplaintSummary.model_validate_json(summary.model_dump_json()) == summary
    with pytest.raises(ValidationError):
        summary.text = "x"  # type: ignore[misc]


def test_summary_span_keeps_its_own_evidence(
    make_summary: Callable[..., ComplaintSummary],
) -> None:
    span: SummarySpan = make_summary().spans[0]
    assert span.evidence.span.line_index == 3


# --------------------------------------------------------------------------- #
# BounceDecision — machine-readable reason
# --------------------------------------------------------------------------- #


def test_bounce_requires_a_machine_readable_reason() -> None:
    with pytest.raises(ValidationError, match="machine-readable reason"):
        BounceDecision(decision=QualityVerdict.BOUNCE)


def test_non_bounce_must_not_carry_a_reason() -> None:
    with pytest.raises(ValidationError, match="must not carry a bounce reason"):
        BounceDecision(
            decision=QualityVerdict.ACCEPT, reason=BounceReason.BLANK_PAGE
        )


def test_accepted_helper() -> None:
    decision = BounceDecision.accepted()
    assert decision.decision is QualityVerdict.ACCEPT
    assert decision.reason is None


@pytest.mark.parametrize("reason", list(RejectionReason))
def test_from_rejection_produces_a_bounce(reason: RejectionReason) -> None:
    decision = BounceDecision.from_rejection(reason, detail="page 1")
    assert decision.decision is QualityVerdict.BOUNCE
    assert decision.reason is not None
    assert decision.reason.value == reason.value
    assert decision.detail == "page 1"


def test_repair_verdict_is_representable() -> None:
    assert (
        BounceDecision(decision=QualityVerdict.REPAIR).decision
        is QualityVerdict.REPAIR
    )


# --------------------------------------------------------------------------- #
# PageResult — derived summaries only
# --------------------------------------------------------------------------- #


def test_page_result_happy_path(make_page_result: Callable[..., PageResult]) -> None:
    page = make_page_result()
    assert page.is_code_mixed is True
    assert PageResult.model_validate_json(page.model_dump_json()) == page


def test_quality_score_and_flags_present_on_accepted_pages(
    make_page_result: Callable[..., PageResult],
) -> None:
    """PRD P0-3: on every page record, including accepted ones."""
    page = make_page_result(
        quality_verdict=QualityVerdict.ACCEPT,
        quality_flags=(QualityFlag.SKEW_REPAIRED,),
    )
    assert page.quality_score == 0.82
    assert page.quality_flags == (QualityFlag.SKEW_REPAIRED,)


def test_language_distribution_must_be_a_distribution(
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        make_page_result(language_distribution={"eng": 0.4, "mar": 0.4})


def test_language_distribution_keys_are_iso_639_3(
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="ISO 639-3"):
        make_page_result(
            primary_language=None, language_distribution={"english": 1.0}
        )


def test_primary_language_must_be_the_argmax(
    make_page_result: Callable[..., PageResult],
) -> None:
    """It is a derived summary, not an independent judgement (invariant 1)."""
    with pytest.raises(ValidationError, match="argmax"):
        make_page_result(
            primary_language="mar",
            language_distribution={"eng": 0.8, "mar": 0.2},
        )


def test_primary_language_must_appear_in_the_distribution(
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="absent from"):
        make_page_result(
            primary_language="hin",
            language_distribution={"eng": 0.8, "mar": 0.2},
        )


def test_bounced_page_reports_no_lines(
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="must not report recognised lines"):
        make_page_result(quality_verdict=QualityVerdict.BOUNCE, line_count=4)


def test_tier_none_contradicts_recognised_lines(
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="contradicts line_count"):
        make_page_result(ocr_tier=OCRTier.NONE, line_count=2)


def test_shadow_sampling_is_distinct_from_escalation(
    make_page_result: Callable[..., PageResult],
) -> None:
    """A shadow run must not inflate the rate it exists to measure."""
    page = make_page_result(shadow_sampled=True, ocr_tier=OCRTier.TIER2)
    assert page.escalated is False
    assert page.shadow_sampled is True


def test_bypass_tier_is_representable(
    make_page_result: Callable[..., PageResult],
) -> None:
    assert make_page_result(ocr_tier=OCRTier.BYPASS).ocr_tier is OCRTier.BYPASS


# --------------------------------------------------------------------------- #
# DocumentResult — invariant 7 at the envelope boundary
# --------------------------------------------------------------------------- #


def test_document_result_happy_path(
    make_document_result: Callable[..., DocumentResult],
) -> None:
    document = make_document_result()
    assert document.document_class is PageClassLabel.COMPLAINT_FORM
    assert (
        DocumentResult.model_validate_json(document.model_dump_json()) == document
    )


@pytest.mark.parametrize(
    "field_name", ["account_number", "ifsc_code", "amount", "micr_code"]
)
def test_known_fields_can_have_unvalidated_slm_source(
    make_document_result: Callable[..., DocumentResult],
    make_field: Callable[..., Field],
    field_name: str,
) -> None:
    document = make_document_result(
        fields={field_name: make_field(source="slm", validator_passed=None)}
    )
    assert document.fields[field_name].source == "slm"
    assert document.fields[field_name].validator_passed is None


def test_open_ended_fields_may_come_from_the_slm(
    make_document_result: Callable[..., DocumentResult],
    make_field: Callable[..., Field],
) -> None:
    document = make_document_result(
        fields={
            "complainant_name": make_field(
                value="Anita Deshmukh", source="slm", validator_passed=None
            )
        }
    )
    assert document.fields["complainant_name"].source == "slm"


def test_human_corrections_are_representable_for_identifiers(
    make_document_result: Callable[..., DocumentResult],
    make_field: Callable[..., Field],
) -> None:
    """A review-UI correction (P1-1) is a legitimate source for an identifier."""
    document = make_document_result(
        fields={"account_number": make_field(value="123456789012", source="human")}
    )
    assert document.fields["account_number"].source == "human"


# --------------------------------------------------------------------------- #
# ResultEnvelope
# --------------------------------------------------------------------------- #


def test_envelope_happy_path_and_json_round_trip(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    envelope = make_envelope()
    restored = ResultEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored == envelope
    assert restored.schema_version == SCHEMA_VERSION == "1.0.0"


def test_schema_version_is_pinned(make_envelope: Callable[..., ResultEnvelope]) -> None:
    with pytest.raises(ValidationError):
        make_envelope(schema_version="2.0.0")


def test_result_version_starts_at_one(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    with pytest.raises(ValidationError):
        make_envelope(result_version=0)
    partial = make_envelope(result_version=1, is_final=False)
    assert partial.is_final is False


def test_two_stage_delivery_shape(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    """PRD P0-10: version 1 at the fast-path SLA, later versions merge in."""
    fast = make_envelope(result_version=1, is_final=False)
    final = make_envelope(result_version=2, is_final=True)
    assert (fast.result_version, fast.is_final) == (1, False)
    assert (final.result_version, final.is_final) == (2, True)


def test_missing_document_must_be_a_bounce(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    with pytest.raises(ValidationError, match="must say why"):
        make_envelope(document=None)


def test_bounced_envelope_carries_no_content(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    bounced = make_envelope(
        document=None,
        pages=(),
        bounce_decision=BounceDecision.from_rejection(RejectionReason.EMPTY_FILE),
    )
    assert bounced.document is None
    assert bounced.bounce_decision.reason is BounceReason.EMPTY_FILE
    with pytest.raises(ValidationError, match="must not carry extracted content"):
        make_envelope(
            bounce_decision=BounceDecision.from_rejection(
                RejectionReason.MALFORMED_PDF
            )
        )


def test_pages_must_be_unique_and_ordered(
    make_envelope: Callable[..., ResultEnvelope],
    make_page_result: Callable[..., PageResult],
    make_document_result: Callable[..., DocumentResult],
) -> None:
    with pytest.raises(ValidationError, match="duplicate page_no"):
        make_envelope(
            pages=(make_page_result(page_no=1), make_page_result(page_no=1)),
            document=make_document_result(page_count=2),
        )
    with pytest.raises(ValidationError, match="ascending page_no"):
        make_envelope(
            pages=(make_page_result(page_no=2), make_page_result(page_no=1)),
            document=make_document_result(page_count=2),
        )


def test_escalation_flag_must_reflect_escalated_pages(
    make_envelope: Callable[..., ResultEnvelope],
    make_page_result: Callable[..., PageResult],
) -> None:
    with pytest.raises(ValidationError, match="escalation_flag must be true"):
        make_envelope(pages=(make_page_result(escalated=True),))
    ok = make_envelope(
        pages=(make_page_result(escalated=True),), escalation_flag=True
    )
    assert ok.escalation_flag is True


def test_shadow_sampled_page_does_not_force_the_escalation_flag(
    make_envelope: Callable[..., ResultEnvelope],
    make_page_result: Callable[..., PageResult],
) -> None:
    envelope = make_envelope(pages=(make_page_result(shadow_sampled=True),))
    assert envelope.escalation_flag is False


def test_page_count_must_match_the_page_records(
    make_envelope: Callable[..., ResultEnvelope],
    make_document_result: Callable[..., DocumentResult],
) -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        make_envelope(document=make_document_result(page_count=7))


def test_created_at_must_be_timezone_aware(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        # A naive datetime is exactly what this test exists to reject.
        make_envelope(created_at=datetime(2026, 8, 7, 12, 0, 0))  # noqa: DTZ001
    aware = make_envelope(created_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=UTC))
    assert aware.created_at.tzinfo is not None


def test_envelope_forbids_unknown_keys(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    with pytest.raises(ValidationError):
        make_envelope(page_language="eng")


def test_processing_stats_default_to_zero_not_none(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    stats: ProcessingStats = make_envelope().processing
    assert stats.pages_processed == 0
    assert stats.stage_timings_ms == {}


def test_full_envelope_serialises_to_json_safe_primitives(
    make_envelope: Callable[..., ResultEnvelope],
    make_summary: Callable[..., ComplaintSummary],
    make_document_result: Callable[..., DocumentResult],
    make_field: Callable[..., Field],
) -> None:
    envelope = make_envelope(
        document=make_document_result(
            complaint_summary=make_summary(),
            fields={
                "ifsc_code": make_field(),
                "complainant_name": make_field(
                    value="Anita Deshmukh", source="slm", validator_passed=None
                ),
            },
        )
    )
    payload = envelope.model_dump(mode="json")
    assert payload["schema_version"] == "1.0.0"
    assert payload["document"]["complaint_summary"]["spans"][0]["text"]
    assert isinstance(payload["bounce_decision"]["decision"], str)
    assert ResultEnvelope.model_validate(payload) == envelope


def test_evidence_survives_a_full_round_trip(
    make_envelope: Callable[..., ResultEnvelope],
    make_document_result: Callable[..., DocumentResult],
    make_field: Callable[..., Field],
    make_evidence: Callable[..., Evidence],
) -> None:
    evidence = make_evidence(
        page_no=3,
        bbox=bbox_from_xyxy(720.0, 526.0, 968.0, 567.0),
        span=TextSpan(line_index=12, start=4, end=15),
    )
    envelope = make_envelope(
        document=make_document_result(
            fields={"ifsc_code": make_field(evidence=evidence)}
        )
    )
    restored = ResultEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored.document is not None
    assert restored.document.fields["ifsc_code"].evidence == evidence


def _all_field_names(model: Any) -> set[str]:
    return set(model.model_fields)


def test_brief_field_set_is_present_exactly() -> None:
    """BUILD-BRIEF sec. 3.3 lists nine keys; ``flags`` is the documented extra."""
    expected = {
        "value",
        "value_en",
        "normalized",
        "confidence",
        "source",
        "validator_passed",
        "evidence",
        "mt_engine",
        "flags",
    }
    assert _all_field_names(Field) == expected
