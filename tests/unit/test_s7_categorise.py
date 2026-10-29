"""Stage 7 — the categorisation stub and the interface it sits behind (WS-G).

Two things are under test, and they are not the same thing:

1. **the interface** — the protocol, the ``NotProduction`` marker, the gate that
   refuses a stub in production, and the closed-set/provenance rules on the
   result. These are the parts that survive when the real RB-IOS classifier
   lands;
2. **the baseline** — determinism, tie-breaks, word boundaries, the fallback.
   Its *accuracy* is deliberately not asserted anywhere: there is no labelled
   set, and a test that pinned accuracy would be pinning an invented number.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from app.contracts.envelope import DocumentResult
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, PageClassLabel, bbox_from_xyxy
from app.stages.s7_categorise import (
    DEFAULT_MAX_TAGS,
    TAXONOMY_FILENAME,
    CategorisationRequest,
    CategorisationResult,
    Categoriser,
    CategoriserNotProduction,
    Category,
    CategoryScore,
    CategoryTaxonomy,
    KeywordCategoriser,
    NotProduction,
    NotProductionWarning,
    TaxonomyInvalid,
    categorise_document,
    default_categoriser,
    load_taxonomy,
    require_production_categoriser,
)

ATM_COMPLAINT = (
    "I tried to withdraw Rs 10,000 at the ATM on 14 January but the cash was "
    "not dispensed and my account was debited."
)
UPI_COMPLAINT_HINDI = (
    "मैंने यूपीआई से पैसे भेजे, राशि कट गई पर लाभार्थी को नहीं मिला और वापस "
    "नहीं आया।"
)
UNRELATED = "The monsoon arrived early this year and the garden is doing well."


@pytest.fixture
def taxonomy() -> CategoryTaxonomy:
    return load_taxonomy()


@pytest.fixture
def engine(taxonomy: CategoryTaxonomy) -> KeywordCategoriser:
    return KeywordCategoriser(taxonomy, warn=False)


def _request(text: str, **overrides: Any) -> CategorisationRequest:
    payload: dict[str, Any] = {"document_id": "doc-1", "lines": (text,)}
    payload.update(overrides)
    return CategorisationRequest(**payload)


def _category(code: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "label_en": code.title(),
        "label_hi": "श्रेणी",
        "description": "PLACEHOLDER fixture category.",
        "keywords_en": [],
        "keywords_hi": [],
    }
    payload.update(overrides)
    return payload


def _taxonomy_file(tmp_path: Path, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "schema": "rbios-categories/1",
        "taxonomy_id": "fixture",
        "version": "0.0.1",
        "status": "SYNTHETIC_PLACEHOLDER",
        "authoritative": False,
        "not_production": True,
        "source": "test fixture",
        "notes": "test fixture",
        "fallback_code": "OTHER",
        "categories": [
            _category("ALPHA", keywords_en=["alpha", "first"]),
            _category("BETA", keywords_en=["beta", "second"]),
            _category("OTHER"),
        ],
    }
    payload.update(overrides)
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The committed placeholder taxonomy
# --------------------------------------------------------------------------- #


def test_taxonomy_loads_from_the_committed_resource(taxonomy: CategoryTaxonomy) -> None:
    assert taxonomy.taxonomy_id == "rbios-placeholder"
    assert len(taxonomy.categories) >= 10
    assert taxonomy.fallback.code == "OTHER"


def test_taxonomy_is_loudly_marked_as_a_placeholder(
    taxonomy: CategoryTaxonomy,
) -> None:
    assert taxonomy.authoritative is False
    assert taxonomy.not_production is True
    assert "PLACEHOLDER" in taxonomy.status
    assert "PLACEHOLDER" in taxonomy.version
    assert "INVENTED" in taxonomy.source
    assert taxonomy.blocking_dependency and "RB-IOS" in taxonomy.blocking_dependency


def test_every_category_declares_itself_a_placeholder(
    taxonomy: CategoryTaxonomy,
) -> None:
    for category in taxonomy.categories:
        assert "PLACEHOLDER" in category.description, category.code


def test_taxonomy_codes_are_unique_and_upper_snake(
    taxonomy: CategoryTaxonomy,
) -> None:
    codes = [category.code for category in taxonomy.categories]
    assert len(set(codes)) == len(codes)
    for code in codes:
        assert code == code.upper()
        assert " " not in code


def test_fallback_carries_no_keywords(taxonomy: CategoryTaxonomy) -> None:
    """The fallback must be reachable only by nothing else matching."""
    assert taxonomy.fallback.keywords == ()


def test_by_code_refuses_a_code_outside_the_closed_set(
    taxonomy: CategoryTaxonomy,
) -> None:
    with pytest.raises(KeyError):
        taxonomy.by_code("NOT_A_CATEGORY")


def test_taxonomy_is_cached(taxonomy: CategoryTaxonomy) -> None:
    assert load_taxonomy() is taxonomy


def test_resource_filename_is_what_the_module_advertises() -> None:
    from app.resources import RESOURCES_DIR

    assert (RESOURCES_DIR / TAXONOMY_FILENAME).is_file()


# --------------------------------------------------------------------------- #
# Loading failures are typed, never an empty table
# --------------------------------------------------------------------------- #


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyInvalid) as excinfo:
        CategoryTaxonomy.load(tmp_path / "absent.json")
    assert excinfo.value.code == "TAXONOMY_INVALID"


def test_broken_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TaxonomyInvalid, match="not valid JSON"):
        CategoryTaxonomy.load(path)


def test_json_that_is_not_an_object_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TaxonomyInvalid, match="must be a JSON object"):
        CategoryTaxonomy.load(path)


def test_duplicate_codes_are_refused(tmp_path: Path) -> None:
    path = _taxonomy_file(
        tmp_path,
        categories=[_category("ALPHA"), _category("ALPHA"), _category("OTHER")],
    )
    with pytest.raises(TaxonomyInvalid, match="duplicate category codes"):
        CategoryTaxonomy.load(path)


def test_fallback_outside_the_set_is_refused(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path, fallback_code="MISSING_CODE")
    with pytest.raises(TaxonomyInvalid, match="not one of the categories"):
        CategoryTaxonomy.load(path)


def test_authoritative_and_not_production_cannot_both_hold(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path, authoritative=True, not_production=True)
    with pytest.raises(TaxonomyInvalid, match="self-contradictory"):
        CategoryTaxonomy.load(path)


def test_a_real_taxonomy_is_a_data_change_not_a_code_change(tmp_path: Path) -> None:
    """The drop-in path: same schema, authoritative, production."""
    path = _taxonomy_file(
        tmp_path,
        taxonomy_id="rbios-2021",
        version="2021.1",
        status="AUTHORITATIVE",
        authoritative=True,
        not_production=False,
        source="the real thing",
    )

    real = CategoryTaxonomy.load(path)

    assert real.authoritative is True
    assert real.not_production is False
    assert KeywordCategoriser(real, warn=False).taxonomy is real


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


def test_baseline_satisfies_the_protocol(engine: KeywordCategoriser) -> None:
    assert isinstance(engine, Categoriser)


def test_protocol_is_a_data_protocol(engine: KeywordCategoriser) -> None:
    """``issubclass`` cannot be used on a protocol with non-method members."""
    with pytest.raises(TypeError):
        issubclass(KeywordCategoriser, Categoriser)  # type: ignore[misc]


def test_any_object_with_the_three_members_satisfies_the_protocol() -> None:
    class Fake:
        name = "fake"
        is_production = True

        def categorise(self, request: CategorisationRequest) -> CategorisationResult:
            raise NotImplementedError

    assert isinstance(Fake(), Categoriser)


def test_marker_pins_is_production_false(engine: KeywordCategoriser) -> None:
    assert isinstance(engine, NotProduction)
    assert engine.is_production is False
    assert KeywordCategoriser.is_production is False
    assert "RB-IOS" in KeywordCategoriser.not_production_reason


def test_marker_demands_a_reason() -> None:
    with pytest.raises(TypeError, match="not_production_reason"):

        class Silent(NotProduction):
            pass


def test_constructing_the_stub_warns() -> None:
    with pytest.warns(NotProductionWarning, match="NOT a production classifier"):
        KeywordCategoriser()


def test_the_warning_can_be_suppressed_for_bulk_use() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", NotProductionWarning)
        KeywordCategoriser(warn=False)


def test_default_categoriser_is_a_singleton() -> None:
    default_categoriser.cache_clear()
    with pytest.warns(NotProductionWarning):
        first = default_categoriser()
    second = default_categoriser()
    assert first is second


def test_production_gate_refuses_the_stub(engine: KeywordCategoriser) -> None:
    with pytest.raises(CategoriserNotProduction) as excinfo:
        require_production_categoriser(engine)
    assert excinfo.value.code == "CATEGORISER_NOT_PRODUCTION"
    assert "RB-IOS" in excinfo.value.message


def test_production_gate_accepts_a_real_classifier() -> None:
    class Real:
        name = "distilled-encoder-v1"
        is_production = True

        def categorise(self, request: CategorisationRequest) -> CategorisationResult:
            raise NotImplementedError

    require_production_categoriser(Real())


def test_invalid_secondary_share_is_refused(taxonomy: CategoryTaxonomy) -> None:
    for value in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="secondary_share"):
            KeywordCategoriser(taxonomy, secondary_share=value, warn=False)


# --------------------------------------------------------------------------- #
# The baseline
# --------------------------------------------------------------------------- #


def test_english_complaint_is_tagged(engine: KeywordCategoriser) -> None:
    result = engine.categorise(_request(ATM_COMPLAINT))

    assert result.primary == "ATM_CASH_NOT_DISPENSED"
    assert result.tags[0] == result.primary
    assert 0.0 < result.confidence <= 1.0


def test_devanagari_complaint_is_tagged(engine: KeywordCategoriser) -> None:
    result = engine.categorise(_request(UPI_COMPLAINT_HINDI))

    assert result.primary == "UPI_OR_IMPS_CREDIT_FAILURE"
    assert any("यूपीआई" in keyword for keyword in result.scores[0].matched_keywords)


def test_nothing_matched_returns_the_fallback_not_a_guess(
    engine: KeywordCategoriser,
) -> None:
    result = engine.categorise(_request(UNRELATED))

    assert result.tags == ("OTHER",)
    assert result.primary == "OTHER"
    assert result.confidence == 0.0
    assert result.scores == ()


def test_empty_document_returns_the_fallback(engine: KeywordCategoriser) -> None:
    result = engine.categorise(
        CategorisationRequest(document_id="doc-empty", lines=())
    )
    assert result.tags == ("OTHER",)


def test_matching_is_case_insensitive(engine: KeywordCategoriser) -> None:
    upper = engine.categorise(_request(ATM_COMPLAINT.upper()))
    lower = engine.categorise(_request(ATM_COMPLAINT.lower()))

    assert upper.tags == lower.tags


def test_ascii_keywords_respect_word_boundaries(engine: KeywordCategoriser) -> None:
    """'atm' must not fire on 'atmosphere'."""
    result = engine.categorise(
        _request("The atmosphere in the branch was calm and the loanword was odd.")
    )
    assert "ATM_CASH_NOT_DISPENSED" not in result.tags


def test_ascii_keywords_still_match_next_to_punctuation(
    engine: KeywordCategoriser,
) -> None:
    result = engine.categorise(_request("(ATM), cash not dispensed."))
    assert result.primary == "ATM_CASH_NOT_DISPENSED"


def test_confidences_form_a_distribution(engine: KeywordCategoriser) -> None:
    result = engine.categorise(
        _request(ATM_COMPLAINT + " " + UPI_COMPLAINT_HINDI + " No reply for 30 days.")
    )

    assert len(result.scores) >= 2
    assert sum(score.confidence for score in result.scores) == pytest.approx(1.0)
    for score in result.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_scores_are_ordered_by_evidence(engine: KeywordCategoriser) -> None:
    result = engine.categorise(
        _request(ATM_COMPLAINT + " " + UPI_COMPLAINT_HINDI)
    )
    values = [score.score for score in result.scores]
    assert values == sorted(values, reverse=True)


def test_is_deterministic(engine: KeywordCategoriser) -> None:
    text = ATM_COMPLAINT + " " + UPI_COMPLAINT_HINDI
    first = engine.categorise(_request(text))
    second = engine.categorise(_request(text))
    third = KeywordCategoriser(warn=False).categorise(_request(text))

    assert first.model_dump() == second.model_dump()
    assert first.tags == third.tags


def test_ties_break_on_taxonomy_order_not_input_order(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path)
    engine = KeywordCategoriser(CategoryTaxonomy.load(path), warn=False)

    forward = engine.categorise(_request("alpha beta"))
    backward = engine.categorise(_request("beta alpha"))

    assert forward.tags == backward.tags == ("ALPHA", "BETA")


def test_a_weak_runner_up_is_dropped(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path)
    engine = KeywordCategoriser(CategoryTaxonomy.load(path), warn=False)

    result = engine.categorise(_request("alpha first beta"))

    assert result.tags == ("ALPHA",)
    assert [score.code for score in result.scores] == ["ALPHA", "BETA"]


def test_max_tags_is_respected(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path)
    engine = KeywordCategoriser(CategoryTaxonomy.load(path), warn=False)

    result = engine.categorise(_request("alpha beta", max_tags=1))

    assert result.tags == ("ALPHA",)


def test_repeating_one_word_does_not_outvote_variety(tmp_path: Path) -> None:
    path = _taxonomy_file(tmp_path)
    engine = KeywordCategoriser(CategoryTaxonomy.load(path), warn=False)

    result = engine.categorise(_request("alpha alpha alpha alpha beta second"))

    assert result.primary == "BETA"


def test_result_carries_the_provenance_that_bounds_its_use(
    engine: KeywordCategoriser,
) -> None:
    result = engine.categorise(_request(ATM_COMPLAINT))

    assert result.categoriser == "keyword-baseline-v0"
    assert result.is_production is False
    assert result.not_production_reason
    assert result.taxonomy_id == "rbios-placeholder"
    assert result.taxonomy_authoritative is False


# --------------------------------------------------------------------------- #
# Result contract
# --------------------------------------------------------------------------- #


def _result_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": "doc-1",
        "tags": ("ALPHA",),
        "primary": "ALPHA",
        "confidence": 0.5,
        "scores": (),
        "categoriser": "test",
        "is_production": False,
        "not_production_reason": "test stub",
        "taxonomy_id": "fixture",
        "taxonomy_version": "0.0.1",
        "taxonomy_authoritative": False,
    }
    payload.update(overrides)
    return payload


def test_non_production_result_must_say_why() -> None:
    with pytest.raises(ValueError, match="not_production_reason"):
        CategorisationResult(**_result_payload(not_production_reason=None))


def test_production_result_must_not_carry_a_reason() -> None:
    with pytest.raises(ValueError, match="not_production_reason set"):
        CategorisationResult(
            **_result_payload(is_production=True, taxonomy_authoritative=True)
        )


def test_production_result_needs_an_authoritative_taxonomy() -> None:
    with pytest.raises(ValueError, match="non-authoritative taxonomy"):
        CategorisationResult(
            **_result_payload(
                is_production=True,
                not_production_reason=None,
                taxonomy_authoritative=False,
            )
        )


def test_duplicate_tags_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate tags"):
        CategorisationResult(**_result_payload(tags=("ALPHA", "ALPHA")))


def test_primary_must_be_the_first_tag() -> None:
    with pytest.raises(ValueError, match="must be tags"):
        CategorisationResult(**_result_payload(tags=("ALPHA", "BETA"), primary="BETA"))


def test_tags_are_never_empty() -> None:
    with pytest.raises(ValueError):
        CategorisationResult(**_result_payload(tags=()))


def test_score_confidence_is_bounded() -> None:
    with pytest.raises(ValueError):
        CategoryScore(
            code="ALPHA", label_en="Alpha", label_hi="अ", score=1, confidence=1.4
        )


def test_category_code_pattern_is_enforced() -> None:
    with pytest.raises(ValueError):
        Category(
            code="lower case",
            label_en="x",
            label_hi="x",
            description="PLACEHOLDER",
        )


# --------------------------------------------------------------------------- #
# Request builders and the pipeline slot
# --------------------------------------------------------------------------- #


def _line(text: str, index: int) -> OCRLine:
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(10.0, 10.0 + 20 * index, 400.0, 30.0 + 20 * index),
        confidence=0.9,
        script="Latn",
        is_handwritten=False,
        line_index=index,
    )


def test_request_from_ocr_lines() -> None:
    request = CategorisationRequest.from_ocr_lines(
        "doc-2", [_line("ATM", 0), _line("cash not dispensed", 1)]
    )

    assert request.lines == ("ATM", "cash not dispensed")
    assert request.text == "ATM\ncash not dispensed"
    assert request.max_tags == DEFAULT_MAX_TAGS


def test_request_from_pages_is_ordered_by_page() -> None:
    second = OCRPage(
        page_no=2, width_px=100, height_px=100, lines=[_line("second", 0)],
        tier=OCRTier.TIER1,
    )
    first = OCRPage(
        page_no=1, width_px=100, height_px=100, lines=[_line("first", 0)],
        tier=OCRTier.TIER1,
    )

    request = CategorisationRequest.from_pages("doc-3", [second, first])

    assert request.lines == ("first", "second")


def test_request_is_a_closed_set() -> None:
    with pytest.raises(ValueError):
        CategorisationRequest(
            document_id="doc-1", lines=(), account_number="123"  # type: ignore[call-arg]
        )


def test_categorise_document_accepts_lines_or_strings(
    engine: KeywordCategoriser,
) -> None:
    from_strings = categorise_document(
        "doc-4", [ATM_COMPLAINT], categoriser=engine
    )
    from_lines = categorise_document(
        "doc-4", [_line(ATM_COMPLAINT, 0)], categoriser=engine
    )

    assert from_strings.tags == from_lines.tags == ("ATM_CASH_NOT_DISPENSED",)


def test_categorise_document_passes_the_page_class(
    engine: KeywordCategoriser,
) -> None:
    result = categorise_document(
        "doc-5",
        [ATM_COMPLAINT],
        page_class=PageClassLabel.SUPPORTING_DOC,
        categoriser=engine,
    )
    assert result.primary == "ATM_CASH_NOT_DISPENSED"


def test_tags_drop_into_the_envelope_unchanged(engine: KeywordCategoriser) -> None:
    """PRD P1-3: supporting documents get tags, not fields."""
    result = engine.categorise(_request(ATM_COMPLAINT))

    document = DocumentResult(
        document_class=PageClassLabel.SUPPORTING_DOC,
        document_class_confidence=0.8,
        page_count=1,
        supporting_doc_tags=result.tags,
    )

    assert document.supporting_doc_tags == result.tags
    assert document.fields == {}
