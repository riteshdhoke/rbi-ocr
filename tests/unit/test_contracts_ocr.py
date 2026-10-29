"""Contract tests for :mod:`app.contracts.ocr`.

The load-bearing assertions here are PLAN.md invariants 1, 2, 3 and 7:
language belongs to the line, the summary is line indices not prose, nothing
aggregates confidence onto a page, and deterministic fields never reach the SLM.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from app.contracts.ocr import (
    BBOX_LEN,
    DETERMINISTIC_FIELDS,
    SLM_ELIGIBLE_FIELDS,
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    LangCandidate,
    LangResult,
    OCRLine,
    OCRPage,
    OCRTier,
    PageClass,
    PageClassLabel,
    RawField,
    ScriptResult,
    ScriptVote,
    StructuralFeatures,
    TextSpan,
    bbox_corners,
    bbox_from_xyxy,
)

# --------------------------------------------------------------------------- #
# bbox
# --------------------------------------------------------------------------- #


def test_bbox_helpers_round_trip() -> None:
    box = bbox_from_xyxy(10.0, 20.0, 110.0, 60.0)
    assert box == [10.0, 20.0, 110.0, 20.0, 110.0, 60.0, 10.0, 60.0]
    assert bbox_corners(box) == [
        (10.0, 20.0),
        (110.0, 20.0),
        (110.0, 60.0),
        (10.0, 60.0),
    ]


def test_bbox_corners_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="8 floats"):
        bbox_corners([1.0, 2.0])


@pytest.mark.parametrize("count", [0, 1, 4, 6, 7, 9, 10, 16])
def test_bbox_must_be_exactly_eight_floats(
    make_ocr_line: Callable[..., OCRLine], count: int
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_ocr_line(bbox=[float(i) for i in range(count)])
    assert str(BBOX_LEN) in str(excinfo.value)


def test_bbox_accepts_exactly_eight_floats(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    line = make_ocr_line(bbox=[float(i) for i in range(BBOX_LEN)])
    assert len(line.bbox) == BBOX_LEN


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_bbox_rejects_non_finite_coordinates(
    make_ocr_line: Callable[..., OCRLine], bad: float
) -> None:
    box = bbox_from_xyxy(0.0, 0.0, 10.0, 10.0)
    box[3] = bad
    with pytest.raises(ValidationError, match="finite"):
        make_ocr_line(bbox=box)


def test_bbox_allows_negative_coordinates_after_rotation(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    """A bounce-profile page rotated 17 degrees can push corners off-canvas."""
    line = make_ocr_line(bbox=[-3.0, -1.0, 90.0, -4.0, 92.0, 30.0, -1.0, 33.0])
    assert line.bbox[0] == -3.0


# --------------------------------------------------------------------------- #
# OCRLine
# --------------------------------------------------------------------------- #


def test_ocr_line_happy_path_and_json_round_trip(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    line = make_ocr_line(
        text="बैंक ऑफ बड़ौदा",
        script="Deva",
        language="mar",
        is_handwritten=True,
        line_index=1,
    )
    restored = OCRLine.model_validate_json(line.model_dump_json())
    assert restored == line
    assert restored.text == "बैंक ऑफ बड़ौदा"
    assert restored.language == "mar"


def test_language_is_optional_until_lid_runs(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    """s2b sets it per line; it is never defaulted from the page."""
    assert make_ocr_line(language=None).language is None


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_confidence_is_bounded(
    make_ocr_line: Callable[..., OCRLine], bad: float
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_line(confidence=bad)


@pytest.mark.parametrize("bad", ["latn", "LATN", "Latin", "La", "Deva1", ""])
def test_script_must_be_iso_15924(
    make_ocr_line: Callable[..., OCRLine], bad: str
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_line(script=bad)


@pytest.mark.parametrize("bad", ["en", "ENG", "english", "e", "en_IN"])
def test_language_must_be_iso_639_3(
    make_ocr_line: Callable[..., OCRLine], bad: str
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_line(language=bad)


def test_line_index_must_be_non_negative(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_line(line_index=-1)


def test_ocr_line_forbids_unknown_fields(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_line(page_language="eng")


# --------------------------------------------------------------------------- #
# OCRPage — invariant 1 (and 3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "forbidden",
    [
        "language",
        "languages",
        "primary_language",
        "language_distribution",
        "script",
        "primary_script",
        "is_handwritten",
        "is_code_mixed",
        "confidence",
    ],
)
def test_page_refuses_page_level_language_and_confidence(
    make_ocr_page: Callable[..., OCRPage], forbidden: str
) -> None:
    """Design-review A2 (and A1): the bug must be unrepresentable."""
    with pytest.raises(ValidationError) as excinfo:
        make_ocr_page(**{forbidden: "eng"})
    message = str(excinfo.value)
    assert forbidden in message
    assert "invariant" in message


def test_page_has_no_language_attribute_at_all(
    make_ocr_page: Callable[..., OCRPage],
) -> None:
    page = make_ocr_page()
    for attribute in ("language", "primary_language", "script", "is_code_mixed"):
        assert not hasattr(page, attribute), (
            f"OCRPage.{attribute} reintroduces design-review finding A2"
        )


def test_page_keeps_per_line_language(make_ocr_page: Callable[..., OCRPage]) -> None:
    """The bilingual_form hard case: English labels, Marathi handwritten values."""
    page = make_ocr_page()
    assert [line.language for line in page.lines] == ["eng", "mar"]
    assert [line.script for line in page.lines] == ["Latn", "Deva"]
    assert [line.is_handwritten for line in page.lines] == [False, True]


def test_page_json_round_trip(make_ocr_page: Callable[..., OCRPage]) -> None:
    page = make_ocr_page()
    assert OCRPage.model_validate_json(page.model_dump_json()) == page


def test_duplicate_line_indices_are_rejected(
    make_ocr_page: Callable[..., OCRPage], make_ocr_line: Callable[..., OCRLine]
) -> None:
    with pytest.raises(ValidationError, match="duplicate line_index"):
        make_ocr_page(
            lines=[make_ocr_line(line_index=0), make_ocr_line(line_index=0)]
        )


def test_line_by_index_uses_the_index_not_the_position(
    make_ocr_page: Callable[..., OCRPage], make_ocr_line: Callable[..., OCRLine]
) -> None:
    page = make_ocr_page(
        lines=[make_ocr_line(line_index=7, text="seven"), make_ocr_line(line_index=2)]
    )
    assert page.line_by_index(7).text == "seven"
    with pytest.raises(KeyError):
        page.line_by_index(0)


@pytest.mark.parametrize("dimension", ["width_px", "height_px"])
def test_page_dimensions_must_be_positive(
    make_ocr_page: Callable[..., OCRPage], dimension: str
) -> None:
    with pytest.raises(ValidationError):
        make_ocr_page(**{dimension: 0})


def test_page_tier_is_a_closed_set(make_ocr_page: Callable[..., OCRPage]) -> None:
    assert make_ocr_page(tier="bypass").tier is OCRTier.BYPASS
    with pytest.raises(ValidationError):
        make_ocr_page(tier="tier3")


# --------------------------------------------------------------------------- #
# Provider exchange models
# --------------------------------------------------------------------------- #


def test_script_result_happy_path() -> None:
    result = ScriptResult(
        script="Deva",
        confidence=0.88,
        distribution={"Deva": 0.7, "Latn": 0.3},
        votes=[ScriptVote(script="Deva", confidence=0.9)],
        sampled_crops=10,
    )
    assert result.script == "Deva"
    assert ScriptResult.model_validate_json(result.model_dump_json()) == result


def test_script_result_distribution_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1.0"):
        ScriptResult(
            script="Deva",
            confidence=0.5,
            distribution={"Deva": 0.4, "Latn": 0.4},
            sampled_crops=2,
        )


def test_script_result_winner_must_be_in_the_distribution() -> None:
    with pytest.raises(ValidationError, match="absent from distribution"):
        ScriptResult(
            script="Beng",
            confidence=0.5,
            distribution={"Deva": 0.6, "Latn": 0.4},
            sampled_crops=2,
        )


def test_lang_result_is_constrained_by_script() -> None:
    result = LangResult(
        language="mar",
        confidence=0.71,
        script="Deva",
        candidates=[
            LangCandidate(language="mar", confidence=0.71),
            LangCandidate(language="hin", confidence=0.29),
        ],
    )
    assert result.script == "Deva"
    assert result.language == "mar"


def test_lang_result_winner_must_be_a_candidate() -> None:
    with pytest.raises(ValidationError, match="not among the candidates"):
        LangResult(
            language="eng",
            confidence=0.9,
            script="Deva",
            candidates=[LangCandidate(language="hin", confidence=0.9)],
        )


def test_lang_result_allows_undetermined() -> None:
    assert LangResult(language=None, confidence=0.0, script="Zyyy").language is None


def _features(**overrides: Any) -> StructuralFeatures:
    data: dict[str, Any] = {
        "line_count": 34,
        "ink_density": 0.12,
        "aspect_ratio": 0.707,
        "has_table_lines": True,
        "table_line_count": 12,
        "colour_histogram": [0.6, 0.3, 0.1],
    }
    data.update(overrides)
    return StructuralFeatures(**data)


def test_structural_features_happy_path() -> None:
    feats = _features()
    assert feats.has_table_lines is True
    assert StructuralFeatures.model_validate_json(feats.model_dump_json()) == feats


def test_structural_features_require_a_colour_histogram() -> None:
    with pytest.raises(ValidationError):
        _features(colour_histogram=[])


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_structural_features_histogram_bins_are_bounded(bad: float) -> None:
    with pytest.raises(ValidationError, match="colour_histogram"):
        _features(colour_histogram=[0.5, bad])


def test_structural_features_reject_contradictory_table_signals() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        _features(has_table_lines=False, table_line_count=3)


def test_page_class_scores_must_contain_the_prediction() -> None:
    good = PageClass(
        label=PageClassLabel.COMPLAINT_FORM,
        confidence=0.8,
        scores={"complaint_form": 0.8, "supporting_doc": 0.2},
    )
    assert good.label is PageClassLabel.COMPLAINT_FORM
    with pytest.raises(ValidationError, match="absent from scores"):
        PageClass(
            label=PageClassLabel.PHOTO_EVIDENCE,
            confidence=0.8,
            scores={"complaint_form": 0.8, "supporting_doc": 0.2},
        )


def test_text_span_rejects_reversed_ranges() -> None:
    assert TextSpan(line_index=0, start=3, end=3).end == 3
    with pytest.raises(ValidationError, match="precedes start"):
        TextSpan(line_index=0, start=5, end=2)


def test_text_span_is_frozen() -> None:
    span = TextSpan(line_index=0, start=0, end=1)
    with pytest.raises(ValidationError):
        span.start = 2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ExtractionContext — invariant 7
# --------------------------------------------------------------------------- #


def _context(
    make_ocr_line: Callable[..., OCRLine], **overrides: Any
) -> ExtractionContext:
    data: dict[str, Any] = {
        "document_id": "sha256:" + "b" * 64,
        "page_no": 1,
        "lines": [make_ocr_line(line_index=0), make_ocr_line(line_index=1)],
        "page_class": PageClassLabel.COMPLAINT_LETTER,
        "primary_script": "Deva",
        "target_fields": ["complainant_name", SUMMARY_FIELD_NAME],
        "max_tokens": 512,
    }
    data.update(overrides)
    return ExtractionContext(**data)


def test_extraction_context_happy_path(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    ctx = _context(make_ocr_line)
    assert set(ctx.target_fields) <= SLM_ELIGIBLE_FIELDS
    assert ExtractionContext.model_validate_json(ctx.model_dump_json()) == ctx


@pytest.mark.parametrize("field_name", sorted(DETERMINISTIC_FIELDS))
def test_deterministic_fields_can_never_be_asked_of_the_slm(
    make_ocr_line: Callable[..., OCRLine], field_name: str
) -> None:
    """PLAN.md invariant 7 — validators own identifiers, amounts and dates."""
    with pytest.raises(ValidationError, match="invariant 7"):
        _context(make_ocr_line, target_fields=[field_name])


def test_unknown_target_fields_are_rejected(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    with pytest.raises(ValidationError, match="unknown SLM target field"):
        _context(make_ocr_line, target_fields=["favourite_colour"])


def test_duplicate_target_fields_are_rejected(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    with pytest.raises(ValidationError, match="duplicate target field"):
        _context(
            make_ocr_line, target_fields=["relief_sought", "relief_sought"]
        )


def test_empty_target_fields_are_rejected(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    with pytest.raises(ValidationError):
        _context(make_ocr_line, target_fields=[])


def test_context_carries_no_identifier_values(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    """Under the laptop profile this object crosses the network.

    Validator hits are referenced by span so they can be sentinel-masked; the
    values themselves must have no home here.
    """
    assert "deterministic_fields" not in ExtractionContext.model_fields
    with pytest.raises(ValidationError):
        _context(make_ocr_line, deterministic_fields={"ifsc_code": "BARB0417293"})


def test_masked_spans_must_reference_existing_lines(
    make_ocr_line: Callable[..., OCRLine],
) -> None:
    ctx = _context(
        make_ocr_line, masked_spans=[TextSpan(line_index=1, start=0, end=4)]
    )
    assert ctx.masked_spans[0].line_index == 1
    with pytest.raises(ValidationError, match="absent from page"):
        _context(
            make_ocr_line, masked_spans=[TextSpan(line_index=99, start=0, end=4)]
        )


# --------------------------------------------------------------------------- #
# RawField — invariant 2
# --------------------------------------------------------------------------- #


def test_raw_field_happy_path() -> None:
    field = RawField(
        name="complainant_name",
        value="Anita Deshmukh",
        confidence=0.77,
        line_indices=[4],
    )
    assert RawField.model_validate_json(field.model_dump_json()) == field


def test_summary_must_be_line_indices_not_prose() -> None:
    """PLAN.md invariant 2: the model selects spans, it never writes prose."""
    with pytest.raises(ValidationError, match="invariant 2"):
        RawField(
            name=SUMMARY_FIELD_NAME,
            value="The complainant was debited twice.",
            confidence=0.8,
            line_indices=[3, 4],
        )


def test_summary_requires_at_least_one_selected_line() -> None:
    with pytest.raises(ValidationError, match="at least one selected"):
        RawField(name=SUMMARY_FIELD_NAME, confidence=0.8, line_indices=[])


def test_summary_with_indices_only_is_accepted() -> None:
    field = RawField(name=SUMMARY_FIELD_NAME, confidence=0.8, line_indices=[3, 4])
    assert field.value is None
    assert field.line_indices == [3, 4]


@pytest.mark.parametrize("field_name", sorted(DETERMINISTIC_FIELDS))
def test_raw_field_refuses_deterministic_names(field_name: str) -> None:
    with pytest.raises(ValidationError, match="invariant 7"):
        RawField(name=field_name, value="x", confidence=0.9)


def test_raw_field_rejects_duplicate_line_indices() -> None:
    with pytest.raises(ValidationError, match="duplicate line_indices"):
        RawField(
            name="relief_sought", value="refund", confidence=0.5, line_indices=[2, 2]
        )


def test_field_name_sets_are_disjoint() -> None:
    assert not (DETERMINISTIC_FIELDS & SLM_ELIGIBLE_FIELDS)
    assert SUMMARY_FIELD_NAME in SLM_ELIGIBLE_FIELDS
