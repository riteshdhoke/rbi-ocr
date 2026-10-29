"""Unit tests for the constrained SLM pass and the grounding gate (WS-E3).

Three invariants are on trial here, and each has a test whose failure means the
design has been violated rather than a detail regressed:

**Invariant 2 — the model never writes prose.**  ``complaint_summary`` carries
line *indices* and nothing else.  The contract refuses a prose value at the
:class:`~app.contracts.ocr.RawField` boundary; the gate refuses an index that
names no line; and :class:`~app.stages.s5_slm.SLMPageResult` has no field in
which summary text could be smuggled — it only publishes
``summary_line_indices``, which the assembler later resolves.

**Invariant 7 — identifiers never reach the SLM.**  Asking for a deterministic
field raises while building the context, before anything is serialised.

**Invariant 8 — the grounding gate.**  A value must occur in *that page's* OCR
text within ``GROUNDING_MAX_EDIT_DISTANCE`` edits, compared with Indic digits
normalised, or it is dropped and reported.  BUILD-BRIEF M5's mandatory test —
"a fabricated SLM value not present in the OCR text is rejected by the grounding
gate" — is :meth:`TestGroundingGate.test_a_fabricated_value_is_rejected`.

Everything is offline: the provider is a hand-written double, no weights are
touched, and no test opens a socket.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from app.contracts.envelope import FieldFlag
from app.contracts.errors import TokenBudgetExceeded
from app.contracts.ocr import (
    DETERMINISTIC_FIELDS,
    SLM_ELIGIBLE_FIELDS,
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    OCRLine,
    OCRPage,
    OCRTier,
    PageClassLabel,
    RawField,
    TextSpan,
    bbox_from_xyxy,
)
from app.settings import Settings
from app.stages.s5_slm import (
    DEFAULT_TARGET_FIELDS,
    EXTRACTION_RESPONSE_SCHEMA,
    FIELD_MAX_TOKENS,
    PROMPT_OVERHEAD_TOKENS,
    SLM_VALUE_FIELDS,
    apply_grounding_gate,
    build_extraction_context,
    build_extraction_contexts,
    context_max_tokens,
    extract_pages,
    ground_value,
    normalise_for_grounding,
    render_numbered_lines,
)

#: The complainant name every grounding case is measured against.  Twenty-one
#: characters, no repeated bigrams — an edit lands somewhere visible.
NAME_LINE = "Ramesh Kumar Deshmukh"


# --------------------------------------------------------------------------- #
# Fixtures / doubles
# --------------------------------------------------------------------------- #


def make_line(text: str, index: int, **overrides: Any) -> OCRLine:
    """One OCR line with a distinct bbox per index, so evidence is traceable."""
    data: dict[str, Any] = {
        "text": text,
        "bbox": bbox_from_xyxy(100.0, 100.0 + 40 * index, 900.0, 132.0 + 40 * index),
        "confidence": 0.95,
        "script": "Latn",
        "language": None,
        "is_handwritten": False,
        "line_index": index,
    }
    data.update(overrides)
    return OCRLine(**data)


def make_page(
    texts: Sequence[str], page_no: int = 1, **line_overrides: Any
) -> OCRPage:
    return OCRPage(
        page_no=page_no,
        width_px=1654,
        height_px=2339,
        lines=[
            make_line(text, index, **line_overrides)
            for index, text in enumerate(texts)
        ],
        tier=OCRTier.TIER1,
    )


class StubBudget:
    """Minimal :class:`~app.stages.s5_slm.BudgetSource`."""

    def __init__(self, remaining: int) -> None:
        self._remaining = remaining

    @property
    def remaining(self) -> int:
        return self._remaining


class RecordingProvider:
    """Records the contexts it was handed and replays canned results."""

    def __init__(self, results: Sequence[dict[str, RawField]] | None = None) -> None:
        self.results = list(results) if results is not None else []
        self.calls: list[list[ExtractionContext]] = []

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        self.calls.append(list(ctxs))
        if self.results:
            return self.results
        return [{} for _ in ctxs]


@pytest.fixture
def name_page() -> OCRPage:
    """A one-page complaint with a name, an address and a Devanagari reference."""
    return make_page(
        [
            "1. Name of complainant",
            NAME_LINE,
            "Flat 402, Sunrise Towers",
            "Bandra West, Mumbai 400050",
            "पूर्व शिकायत क्रमांक २०२५/०४५",
        ]
    )


def raw(name: str, **overrides: Any) -> RawField:
    data: dict[str, Any] = {"name": name, "confidence": 0.9}
    data.update(overrides)
    return RawField(**data)


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


class TestRequestConstruction:
    def test_numbered_lines_are_rendered_with_their_indices(self) -> None:
        """The indices in the prompt are the handles the summary comes back as."""
        page = make_page(["first", "second"])
        assert render_numbered_lines(page.lines) == "[0] first\n[1] second"

    def test_legacy_mode_rejects_a_deterministic_field(
        self, settings: Settings
    ) -> None:
        """Invariant 7, enforced before anything is serialised."""
        settings = settings.model_copy(update={"EXTRACT_ALL_FIELDS": False})
        page = make_page([NAME_LINE])
        with pytest.raises(ValidationError, match="must never be sent to the SLM"):
            build_extraction_context(
                page,
                document_id="doc-1",
                page_class=PageClassLabel.COMPLAINT_FORM,
                primary_script="Latn",
                target_fields=["complainant_name", "account_number"],
                settings=settings,
            )

    def test_legacy_mode_refuses_every_deterministic_field(self, settings: Settings) -> None:
        page = make_page([NAME_LINE])
        settings = settings.model_copy(update={"EXTRACT_ALL_FIELDS": False})
        for name in sorted(DETERMINISTIC_FIELDS):
            with pytest.raises(ValidationError):
                build_extraction_context(
                    page,
                    document_id="doc-1",
                    page_class=PageClassLabel.COMPLAINT_FORM,
                    primary_script="Latn",
                    target_fields=[name],
                    settings=settings,
                )

    def test_default_targets_are_exactly_the_eligible_set(self) -> None:
        assert set(DEFAULT_TARGET_FIELDS) == SLM_ELIGIBLE_FIELDS
        assert set(SLM_VALUE_FIELDS) == SLM_ELIGIBLE_FIELDS - {SUMMARY_FIELD_NAME}
        assert list(DEFAULT_TARGET_FIELDS) == sorted(DEFAULT_TARGET_FIELDS)

    def test_empty_target_fields_is_a_caller_bug(self, settings: Settings) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            build_extraction_context(
                make_page([NAME_LINE]),
                document_id="doc-1",
                page_class=PageClassLabel.COMPLAINT_FORM,
                primary_script="Latn",
                target_fields=[],
                settings=settings,
            )

    def test_masked_spans_are_carried_into_the_context(
        self, settings: Settings
    ) -> None:
        """The mechanism by which invariant 7 is true of the bytes, not the list."""
        page = make_page(["A/c 50100234567890", NAME_LINE])
        spans = [TextSpan(line_index=0, start=4, end=18)]
        ctx = build_extraction_context(
            page,
            document_id="doc-1",
            page_class=PageClassLabel.COMPLAINT_FORM,
            primary_script="Latn",
            masked_spans=spans,
            settings=settings,
        )
        assert ctx.masked_spans == spans

    def test_a_masked_span_on_an_absent_line_is_refused(
        self, settings: Settings
    ) -> None:
        with pytest.raises(ValidationError, match="absent from page"):
            build_extraction_context(
                make_page([NAME_LINE]),
                document_id="doc-1",
                page_class=PageClassLabel.COMPLAINT_FORM,
                primary_script="Latn",
                masked_spans=[TextSpan(line_index=9, start=0, end=1)],
                settings=settings,
            )

    def test_token_ceiling_is_per_field_and_clamped(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        settings = make_settings(SLM_MAX_TOKENS=512)
        one = context_max_tokens(["complainant_name"], settings=settings)
        assert one == FIELD_MAX_TOKENS["complainant_name"] + PROMPT_OVERHEAD_TOKENS
        everything = context_max_tokens(DEFAULT_TARGET_FIELDS, settings=settings)
        assert everything == settings.SLM_MAX_TOKENS

    def test_no_headroom_refuses_before_the_request_is_built(
        self, settings: Settings
    ) -> None:
        """A breach costs zero tokens because nothing is sent."""
        with pytest.raises(TokenBudgetExceeded):
            build_extraction_context(
                make_page([NAME_LINE]),
                document_id="doc-1",
                page_class=PageClassLabel.COMPLAINT_FORM,
                primary_script="Latn",
                budget=StubBudget(0),
                settings=settings,
            )

    def test_remaining_budget_clamps_the_ceiling(self, settings: Settings) -> None:
        ctx = build_extraction_context(
            make_page([NAME_LINE]),
            document_id="doc-1",
            page_class=PageClassLabel.COMPLAINT_FORM,
            primary_script="Latn",
            target_fields=["complainant_name"],
            budget=StubBudget(50),
            settings=settings,
        )
        assert ctx.max_tokens == 50
        assert ctx.token_budget_remaining == 50

    def test_pages_without_lines_are_skipped(self, settings: Settings) -> None:
        """Paying for a request that can only return ``{}`` is what the budget
        exists to stop."""
        pages = [
            make_page([NAME_LINE], page_no=1),
            OCRPage(
                page_no=2, width_px=100, height_px=100, lines=[], tier=OCRTier.NONE
            ),
        ]
        ctxs = build_extraction_contexts(
            pages,
            document_id="doc-1",
            page_classes=PageClassLabel.COMPLAINT_FORM,
            primary_scripts="Latn",
            settings=settings,
        )
        assert [ctx.page_no for ctx in ctxs] == [1]

    def test_per_page_overrides_are_read_from_mappings(
        self, settings: Settings
    ) -> None:
        pages = [make_page([NAME_LINE], page_no=n) for n in (1, 2)]
        ctxs = build_extraction_contexts(
            pages,
            document_id="doc-1",
            page_classes={
                1: PageClassLabel.COMPLAINT_FORM,
                2: PageClassLabel.SUPPORTING_DOC,
            },
            primary_scripts={1: "Latn", 2: "Deva"},
            primary_languages={1: "eng", 2: "mar"},
            settings=settings,
        )
        assert [ctx.page_class for ctx in ctxs] == [
            PageClassLabel.COMPLAINT_FORM,
            PageClassLabel.SUPPORTING_DOC,
        ]
        assert [ctx.primary_script for ctx in ctxs] == ["Latn", "Deva"]
        assert [ctx.primary_language for ctx in ctxs] == ["eng", "mar"]

    def test_extraction_always_goes_through_the_batch_call(
        self, settings: Settings
    ) -> None:
        """One call site, so the provider owns concurrency (BUILD-BRIEF 3.1)."""
        provider = RecordingProvider([{}])
        ctxs = build_extraction_contexts(
            [make_page([NAME_LINE])],
            document_id="doc-1",
            page_classes=PageClassLabel.COMPLAINT_FORM,
            primary_scripts="Latn",
            settings=settings,
        )
        extract_pages(provider, ctxs)
        assert len(provider.calls) == 1
        assert len(provider.calls[0]) == 1

    def test_no_contexts_means_no_provider_call(self) -> None:
        provider = RecordingProvider()
        assert extract_pages(provider, []) == []
        assert provider.calls == []

    def test_a_provider_that_loses_a_page_is_a_hard_error(
        self, settings: Settings
    ) -> None:
        """A half-filled result would let the assembler publish a complete-looking
        envelope."""
        provider = RecordingProvider([{}])
        ctxs = build_extraction_contexts(
            [make_page([NAME_LINE], page_no=n) for n in (1, 2)],
            document_id="doc-1",
            page_classes=PageClassLabel.COMPLAINT_FORM,
            primary_scripts="Latn",
            settings=settings,
        )
        with pytest.raises(ValueError, match="positions must be preserved"):
            extract_pages(provider, ctxs)


# --------------------------------------------------------------------------- #
# Invariant 2 — extractive summary, indices only
# --------------------------------------------------------------------------- #


class TestExtractiveSummary:
    def test_prose_is_refused_at_the_contract_boundary(self) -> None:
        """The model cannot even represent a written summary."""
        with pytest.raises(ValidationError, match="not prose"):
            RawField(
                name=SUMMARY_FIELD_NAME,
                value="The complainant was debited twice.",
                confidence=0.9,
                line_indices=[1, 2],
            )

    def test_a_summary_without_indices_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one selected"):
            RawField(name=SUMMARY_FIELD_NAME, value=None, confidence=0.9)

    def test_the_gate_publishes_indices_and_no_text(self, name_page: OCRPage) -> None:
        """The whole payload of a summary is a tuple of ints."""
        result = apply_grounding_gate(
            {SUMMARY_FIELD_NAME: raw(SUMMARY_FIELD_NAME, line_indices=[2, 1])},
            name_page,
            max_distance=2,
        )
        assert result.summary_line_indices == (2, 1)
        assert SUMMARY_FIELD_NAME not in result.fields
        assert result.rejected == ()

    def test_the_models_ordering_within_a_page_is_preserved(
        self, name_page: OCRPage
    ) -> None:
        """The model's order *is* its reading-order judgement; re-sorting it would
        scramble a narrative that runs down the page."""
        result = apply_grounding_gate(
            {SUMMARY_FIELD_NAME: raw(SUMMARY_FIELD_NAME, line_indices=[3, 0, 2])},
            name_page,
            max_distance=2,
        )
        assert result.summary_line_indices == (3, 0, 2)

    def test_an_index_that_names_no_line_is_rejected(
        self, name_page: OCRPage
    ) -> None:
        result = apply_grounding_gate(
            {SUMMARY_FIELD_NAME: raw(SUMMARY_FIELD_NAME, line_indices=[1, 99])},
            name_page,
            max_distance=2,
        )
        assert result.summary_line_indices == ()
        assert [item.reason for item in result.rejected] == ["unknown_line_index"]
        assert result.rejected[0].line_indices == (99,)
        assert result.rejected[0].value is None
        assert result.needs_review is True

    def test_the_result_model_has_nowhere_to_put_summary_text(self) -> None:
        """Structural, not behavioural: there is no attribute to smuggle prose in."""
        from app.stages.s5_slm import SLMPageResult

        assert "summary_text" not in SLMPageResult.model_fields
        assert SLMPageResult.model_config["extra"] == "forbid"


# --------------------------------------------------------------------------- #
# Invariant 8 — the grounding gate
# --------------------------------------------------------------------------- #


class TestGroundingGate:
    def test_a_fabricated_value_is_rejected(self, name_page: OCRPage) -> None:
        """BUILD-BRIEF M5's mandatory test.

        The address below is fluent, plausible and completely absent from the
        page.  It must not become a field, and the model's string must not be
        published anywhere in the accepted output.
        """
        fabricated = "Plot 17, Shanti Nagar, Pune 411001"
        result = apply_grounding_gate(
            {"complainant_address": raw("complainant_address", value=fabricated)},
            name_page,
            max_distance=2,
        )
        assert result.fields == {}
        assert len(result.rejected) == 1
        rejection = result.rejected[0]
        assert rejection.name == "complainant_address"
        assert rejection.reason == "ungrounded"
        assert rejection.value == fabricated
        assert rejection.page_no == name_page.page_no
        assert result.needs_review is True

    def test_an_exact_match_is_grounded_and_unflagged(
        self, name_page: OCRPage
    ) -> None:
        result = apply_grounding_gate(
            {"complainant_name": raw("complainant_name", value=NAME_LINE)},
            name_page,
            max_distance=2,
        )
        field = result.fields["complainant_name"]
        assert field.value == NAME_LINE
        assert field.source == "slm"
        assert field.validator_passed is None
        assert field.flags == ()
        assert field.confidence == pytest.approx(0.9)
        assert field.evidence.page_no == 1
        assert field.evidence.span == TextSpan(line_index=1, start=0, end=len(NAME_LINE))

    def test_the_published_value_is_the_ocr_substring_not_the_model_string(
        self, name_page: OCRPage
    ) -> None:
        """When the two disagree, OCR is the authority — that is what makes
        ``Field.value`` "verbatim from OCR" structurally true."""
        result = apply_grounding_gate(
            {"complainant_name": raw("complainant_name", value="Ramesh Kumbr Deshmukh")},
            name_page,
            max_distance=2,
        )
        field = result.fields["complainant_name"]
        assert field.value == NAME_LINE
        assert FieldFlag.LOW_CONFIDENCE in field.flags
        assert field.confidence == pytest.approx(0.9 * (1 - 1 / 3))

    def test_case_differences_do_not_change_the_published_casing(
        self, name_page: OCRPage
    ) -> None:
        result = apply_grounding_gate(
            {"complainant_name": raw("complainant_name", value=NAME_LINE.upper())},
            name_page,
            max_distance=2,
        )
        assert result.fields["complainant_name"].value == NAME_LINE

    @pytest.mark.parametrize(
        ("candidate", "distance"),
        [
            (NAME_LINE, 0),
            ("Ramesh Kumbr Deshmukh", 1),
            ("Ramesh Kumbr Deshmvkh", 2),
        ],
    )
    def test_edits_within_budget_ground(
        self, name_page: OCRPage, candidate: str, distance: int
    ) -> None:
        result = ground_value(candidate, name_page.lines, max_distance=2)
        assert result.grounded is True
        assert result.distance == distance
        assert result.matched_text == NAME_LINE

    def test_a_third_edit_is_beyond_the_budget(self, name_page: OCRPage) -> None:
        """The boundary is inclusive at 2 and closed at 3 — an off-by-one here is
        the difference between catching a hallucination and publishing it."""
        assert (
            ground_value("Ramesh Kumbr Deshmvkg", name_page.lines, max_distance=2).grounded
            is False
        )

    def test_the_budget_comes_from_settings_by_default(
        self, name_page: OCRPage, make_settings: Callable[..., Settings]
    ) -> None:
        candidate = "Ramesh Kumbr Deshmvkg"
        strict = make_settings(GROUNDING_MAX_EDIT_DISTANCE=2)
        loose = make_settings(GROUNDING_MAX_EDIT_DISTANCE=3)
        proposed = {"complainant_name": raw("complainant_name", value=candidate)}
        assert apply_grounding_gate(proposed, name_page, settings=strict).fields == {}
        assert "complainant_name" in apply_grounding_gate(
            proposed, name_page, settings=loose
        ).fields

    def test_indic_digits_are_normalised_before_comparison(
        self, name_page: OCRPage
    ) -> None:
        """Invariant 8's "Indic digits normalized": ``२०२५/०४५`` and ``2025/045``
        are the same reference, and the *native* form is what gets published."""
        result = apply_grounding_gate(
            {
                "prior_complaint_reference": raw(
                    "prior_complaint_reference", value="2025/045"
                )
            },
            name_page,
            max_distance=2,
        )
        field = result.fields["prior_complaint_reference"]
        assert field.value == "२०२५/०४५"
        assert field.flags == ()
        source_line = name_page.line_by_index(field.evidence.span.line_index)
        assert (
            source_line.text[field.evidence.span.start : field.evidence.span.end]
            == field.value
        )

    def test_normalisation_preserves_length_so_offsets_survive(self) -> None:
        """Every evidence span depends on this being a 1:1 code-point map."""
        for text in ("२०२५/०४५", "Ramesh", "₹45,000", "बँक ऑफ बडोदा", "ß"):
            assert len(normalise_for_grounding(text)) == len(text)

    def test_a_devanagari_value_grounds_within_two_edits(self) -> None:
        """Script-aware, per invariant 8: the comparison is over code points, so
        a Devanagari matra confusion costs one edit exactly as a Latin typo does."""
        line = "रमेश कुमार देशमुख"
        page = make_page(["तक्रारदाराचे नाव", line], script="Deva")
        result = ground_value("रमेश कुमीर देशमुख", page.lines, max_distance=2)
        assert result.grounded is True
        assert result.distance == 1
        assert result.matched_text == line

    def test_a_trailing_disagreement_yields_the_shorter_ocr_window(self) -> None:
        """Documented behaviour, not an accident: ``_best_window`` short-circuits
        on an exact substring and otherwise keeps the first window at the lowest
        distance, so a value differing only in its last character grounds against
        the OCR prefix.  What matters — and is asserted — is that the published
        text is always a real slice of the OCR line, never the model's string."""
        line = "रमेश कुमार देशमुख"
        page = make_page(["तक्रारदाराचे नाव", line], script="Deva")
        result = ground_value("रमेश कुमार देशमुक", page.lines, max_distance=2)
        assert result.grounded is True
        assert result.matched_text is not None
        assert result.matched_text in line
        assert result.matched_text != "रमेश कुमार देशमुक"

    def test_an_empty_value_is_the_model_declining_not_a_field(
        self, name_page: OCRPage
    ) -> None:
        result = apply_grounding_gate(
            {"relief_sought": raw("relief_sought", value="   ")},
            name_page,
            max_distance=2,
        )
        assert result.fields == {}
        assert [item.reason for item in result.rejected] == ["empty"]

    def test_grounding_is_per_page(self) -> None:
        """Cross-page grounding would let a model launder page 1's value into
        page 7's fields."""
        page_two = make_page(["2. Details of transaction"], page_no=2)
        result = apply_grounding_gate(
            {"complainant_name": raw("complainant_name", value=NAME_LINE)},
            page_two,
            max_distance=2,
        )
        assert result.fields == {}
        assert result.rejected[0].reason == "ungrounded"
        assert result.rejected[0].page_no == 2

    def test_a_wrong_citation_does_not_sink_a_right_answer(
        self, name_page: OCRPage
    ) -> None:
        """The gate is about the value existing, not the model's bookkeeping."""
        result = apply_grounding_gate(
            {
                "complainant_name": raw(
                    "complainant_name", value=NAME_LINE, line_indices=[0]
                )
            },
            name_page,
            max_distance=2,
        )
        assert result.fields["complainant_name"].evidence.span.line_index == 1

    def test_legacy_mode_rejects_an_ineligible_field_name(self, name_page: OCRPage) -> None:
        result = apply_grounding_gate(
            {"customer_id": raw("customer_id", value=NAME_LINE)},
            name_page,
            max_distance=2,
            settings=Settings(EXTRACT_ALL_FIELDS=False),
        )
        assert result.fields == {}
        assert [item.reason for item in result.rejected] == ["not_eligible"]

    def test_a_handwritten_source_line_flags_the_field(self) -> None:
        page = make_page(["Sunita Patil"], is_handwritten=True)
        result = apply_grounding_gate(
            {"complainant_name": raw("complainant_name", value="Sunita Patil")},
            page,
            max_distance=2,
        )
        assert FieldFlag.HANDWRITTEN_SOURCE in result.fields["complainant_name"].flags

    def test_a_multi_line_address_grounds_on_the_cited_lines(
        self, name_page: OCRPage
    ) -> None:
        """An address legitimately spans a line break; the fallback reports whole-
        line evidence and says so rather than inventing offsets."""
        result = ground_value(
            "Flat 402, Sunrise Towers Bandra West, Mumbai 400050",
            name_page.lines,
            max_distance=2,
            preferred_line_indices=[2, 3],
        )
        assert result.grounded is True
        assert result.approximate_span is True
        assert result.distance is None
        assert result.line_index == 2

    def test_an_approximate_span_costs_half_the_confidence(
        self, name_page: OCRPage
    ) -> None:
        result = apply_grounding_gate(
            {
                "complainant_address": raw(
                    "complainant_address",
                    value="Flat 402, Sunrise Towers Bandra West, Mumbai 400050",
                    line_indices=[2, 3],
                )
            },
            name_page,
            max_distance=2,
        )
        field = result.fields["complainant_address"]
        assert field.confidence == pytest.approx(0.45)
        assert FieldFlag.LOW_CONFIDENCE in field.flags

    def test_accepted_and_rejected_fields_coexist(self, name_page: OCRPage) -> None:
        """One hallucinated field must not cost a document the ones that ground."""
        result = apply_grounding_gate(
            {
                "complainant_name": raw("complainant_name", value=NAME_LINE),
                "relief_sought": raw("relief_sought", value="Refund of ₹99,999"),
            },
            name_page,
            max_distance=2,
        )
        assert set(result.fields) == {"complainant_name"}
        assert [item.name for item in result.rejected] == ["relief_sought"]

    def test_no_proposals_is_not_a_rejection(self, name_page: OCRPage) -> None:
        result = apply_grounding_gate({}, name_page, max_distance=2)
        assert result.fields == {}
        assert result.rejected == ()
        assert result.summary_line_indices == ()
        assert result.needs_review is False


# --------------------------------------------------------------------------- #
# The published response schema must not drift from the provider's parser
# --------------------------------------------------------------------------- #


class TestResponseSchemaMirror:
    def test_schema_properties_match_the_providers_payload_model(self) -> None:
        """``EXTRACTION_RESPONSE_SCHEMA`` is what a ``json_schema`` endpoint is
        told to produce; ``_FieldPayload`` is what the client then parses.  Two
        descriptions of one wire format is one too many unless they agree."""
        from app.providers.hf_api import _FieldPayload

        item = EXTRACTION_RESPONSE_SCHEMA["properties"]["fields"]["items"]
        assert set(item["properties"]) == set(_FieldPayload.model_fields)

    def test_the_schema_accepts_open_field_names(self) -> None:
        item = EXTRACTION_RESPONSE_SCHEMA["properties"]["fields"]["items"]
        assert item["properties"]["name"] == {"type": "string", "minLength": 1}

    def test_the_schema_forbids_unknown_keys(self) -> None:
        item = EXTRACTION_RESPONSE_SCHEMA["properties"]["fields"]["items"]
        assert EXTRACTION_RESPONSE_SCHEMA["additionalProperties"] is False
        assert item["additionalProperties"] is False
