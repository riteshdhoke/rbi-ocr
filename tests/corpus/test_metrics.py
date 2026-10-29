"""Unit checks for the harness metric primitives, with hand-computable expectations.

Every number below is derivable by hand from the strings in the test, so a regression in
the metric code cannot hide behind a plausible-looking float.

Does NOT handle: end-to-end scoring (see test_eval_harness.py) or the locked split
(see test_locked_split.py).
"""

from __future__ import annotations

import math

import pytest

# Devanagari fixtures, chosen so codepoint counts are obvious and NFKC-stable.
BANK_DEVA = "बैंक"  # ब + ै + ं + क  = 4 codepoints
BANK_DEVA_MISSING_ANUSVARA = "बैक"  # 3 codepoints, one deletion
NAME_DEVA = "राजेश कुमार शर्मा"  # 3 whitespace tokens


def test_devanagari_fixtures_have_the_lengths_the_maths_assumes():
    assert len(BANK_DEVA) == 4
    assert len(BANK_DEVA_MISSING_ANUSVARA) == 3
    assert len(NAME_DEVA.split()) == 3


# --------------------------------------------------------------------- distance


def test_levenshtein_basics(harness):
    assert harness.levenshtein("", "") == 0
    assert harness.levenshtein("abc", "abc") == 0
    assert harness.levenshtein("kitten", "sitting") == 3
    assert harness.levenshtein("abc", "") == 3
    assert harness.levenshtein(["a", "b"], ["a", "c"]) == 1


def test_levenshtein_on_devanagari_counts_codepoints(harness):
    assert harness.levenshtein(BANK_DEVA, BANK_DEVA_MISSING_ANUSVARA) == 1


# -------------------------------------------------------------------- CER / WER


def _rate(harness, pairs):
    cell = harness.RateCell()
    for ref, hyp in pairs:
        cell.add(ref, hyp)
    return cell.as_dict()


def test_cer_and_wer_perfect(harness):
    result = _rate(harness, [("hello world", "hello world")])
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0
    assert result["lines"] == 1
    assert result["ref_chars"] == 11
    assert result["ref_words"] == 2


def test_cer_one_substitution(harness):
    result = _rate(harness, [("hello", "hallo")])
    assert result["cer"] == pytest.approx(1 / 5)
    assert result["wer"] == pytest.approx(1.0)  # the single token is wrong


def test_cer_devanagari_one_deletion(harness):
    result = _rate(harness, [(BANK_DEVA, BANK_DEVA_MISSING_ANUSVARA)])
    assert result["cer"] == pytest.approx(1 / 4)


def test_wer_devanagari_dropped_token(harness):
    result = _rate(harness, [(NAME_DEVA, "राजेश कुमार")])
    assert result["wer"] == pytest.approx(1 / 3)


def test_missing_hypothesis_is_a_full_deletion(harness):
    result = _rate(harness, [(NAME_DEVA, None)])
    assert result["cer"] == pytest.approx(1.0)
    assert result["wer"] == pytest.approx(1.0)


def test_rates_are_micro_averaged_not_averaged_per_line(harness):
    """A long correct line must outweigh a short broken one — sum(err)/sum(ref)."""
    result = _rate(harness, [("a", "b"), ("x" * 99, "x" * 99)])
    assert result["cer"] == pytest.approx(1 / 100)


def test_whitespace_is_normalised_but_case_is_not(harness):
    assert _rate(harness, [("a  b", "a b")])["cer"] == 0.0
    assert _rate(harness, [("AB", "ab")])["cer"] == pytest.approx(1.0)


def test_empty_cell_reports_none_not_zero(harness):
    empty = harness.RateCell().as_dict()
    assert empty["cer"] is None and empty["wer"] is None and empty["lines"] == 0


# ------------------------------------------------------------------ value match


@pytest.mark.parametrize(
    "left,right",
    [
        ("Rs. 45,000", "45000"),
        ("₹45,000", "45000"),
        ("४५०००", "45000"),
        ("BARB0417293", "barb0417293"),
        ("CMS/2026/123456", "CMS 2026 123456"),
        ("  spaced  out ", "spacedout"),
    ],
)
def test_normalize_value_makes_equivalent_spellings_equal(harness, left, right):
    assert harness.normalize_value(left) == harness.normalize_value(right)


def test_normalize_value_keeps_genuinely_different_values_apart(harness):
    assert harness.normalize_value("45000") != harness.normalize_value("45001")
    assert harness.normalize_value("BARB0417293") != harness.normalize_value("BARB0417294")


# ----------------------------------------------------------------------- boxes


def test_to_aabb_accepts_quad_and_rect(harness):
    assert harness.to_aabb([0, 0, 10, 0, 10, 4, 0, 4]) == (0.0, 0.0, 10.0, 4.0)
    assert harness.to_aabb([10, 4, 0, 0]) == (0.0, 0.0, 10.0, 4.0)
    assert harness.to_aabb([1, 2, 3]) is None
    assert harness.to_aabb(None) is None
    assert harness.to_aabb(["a", "b", "c", "d"]) is None


def test_iou_identical_disjoint_and_half(harness):
    box = [0, 0, 10, 0, 10, 10, 0, 10]
    assert harness.iou(box, box) == pytest.approx(1.0)
    assert harness.iou(box, [20, 20, 30, 20, 30, 30, 20, 30]) == 0.0
    # 10x10 and 10x10 offset by 5 in x: intersection 50, union 150
    assert harness.iou(box, [5, 0, 15, 0, 15, 10, 5, 10]) == pytest.approx(50 / 150)
    assert harness.iou(box, None) == 0.0


# ------------------------------------------------------------------ percentile


def test_percentile(harness):
    values = [1.0, 2.0, 3.0, 4.0]
    assert harness.percentile(values, 0.0) == 1.0
    assert harness.percentile(values, 0.5) == pytest.approx(2.5)
    assert harness.percentile(values, 1.0) == 4.0
    assert harness.percentile([7.0], 0.95) == 7.0
    assert harness.percentile([], 0.5) is None


# --------------------------------------------------------------- line matching


def _pl(harness, text, bbox=None, language=None):
    return harness.PredLine(
        text=text, bbox=bbox, confidence=0.9, script="Latn", language=language,
        is_handwritten=False, line_index=0,
    )


def test_match_lines_prefers_iou_and_is_one_to_one(harness):
    gt = [
        {"text": "alpha", "bbox": [0, 0, 10, 0, 10, 10, 0, 10]},
        {"text": "beta", "bbox": [0, 20, 10, 20, 10, 30, 0, 30]},
    ]
    pred = [
        _pl(harness, "beta", [0, 20, 10, 20, 10, 30, 0, 30]),
        _pl(harness, "alpha", [1, 0, 11, 0, 11, 10, 1, 10]),
    ]
    matches = {m.gt_index: m for m in harness.match_lines(gt, pred)}
    assert matches[0].pred_index == 1 and matches[0].matched_by == "iou"
    assert matches[1].pred_index == 0
    assert matches[1].iou == pytest.approx(1.0)


def test_match_lines_falls_back_to_text_when_boxes_are_absent(harness):
    gt = [{"text": "Bank of Baroda", "bbox": [0, 0, 10, 0, 10, 10, 0, 10]}]
    pred = [_pl(harness, "Bank of Barodo", None)]
    matches = harness.match_lines(gt, pred)
    assert len(matches) == 1
    assert matches[0].matched_by == "text"
    assert matches[0].iou == 0.0


def test_match_lines_leaves_unrelated_lines_unmatched(harness):
    gt = [{"text": "Bank of Baroda", "bbox": [0, 0, 10, 0, 10, 10, 0, 10]}]
    pred = [_pl(harness, "zzzzzzzzzzzz", None)]
    assert harness.match_lines(gt, pred) == []


# ------------------------------------------------------------ field reference


def _gt(lines, fields, bilingual=None):
    gt = {"lines": lines, "fields": fields}
    if bilingual is not None:
        gt["fields_bilingual"] = bilingual
    return gt


def test_field_reference_prefers_the_native_script_line(harness):
    gt = _gt(
        [{"text": "बैंक ऑफ बड़ौदा", "field": "bank_name", "bbox": [0, 0, 1, 0, 1, 1, 0, 1]}],
        {"bank_name": "Bank of Baroda"},
    )
    reference = harness.build_field_reference(gt)
    assert harness.normalize_value("बैंक ऑफ बड़ौदा") in reference["bank_name"].native
    # the English spelling is accepted too, which is what lets an advisory value_en count
    assert harness.normalize_value("Bank of Baroda") in reference["bank_name"].accepted


def test_field_reference_applies_the_city_to_address_alias(harness):
    gt = _gt(
        [{"text": "पनवेल, महाराष्ट्र", "field": "complainant_address", "bbox": None}],
        {"complainant_city": "Panvel, Maharashtra"},
    )
    reference = harness.build_field_reference(gt)
    assert harness.normalize_value("Panvel, Maharashtra") in reference["complainant_address"].accepted
    assert "complainant_city" not in reference


def test_field_reference_recovers_a_value_embedded_in_a_line(harness):
    gt = _gt(
        [{"text": "A/C 12345678901    IFSC BARB0417293", "field": None, "bbox": None}],
        {"account_number": "12345678901", "ifsc_code": "BARB0417293"},
    )
    reference = harness.build_field_reference(gt)
    assert reference["account_number"].accepted == {harness.normalize_value("12345678901")}
    assert reference["ifsc_code"].accepted == {harness.normalize_value("BARB0417293")}


# ------------------------------------------------- bilingual reference (Q6)


def test_a_monolingual_field_reports_the_same_answer_to_both_questions(harness):
    """native and en must never be empty; a Latin field answers both identically."""
    gt = _gt([{"text": "BARB0417293", "field": "ifsc_code", "bbox": None}], {})
    ref = harness.build_field_reference(gt)["ifsc_code"]
    assert ref.native == ref.en == {harness.normalize_value("BARB0417293")}
    assert ref.is_bilingual is False


def test_a_devanagari_field_keeps_the_english_out_of_the_native_set(harness):
    """The regression Q6 names: gt['fields'] is the translation, not what the page shows."""
    gt = _gt(
        [{"text": "बैंक ऑफ बड़ौदा", "field": "bank_name", "bbox": None}],
        {"bank_name": "Bank of Baroda"},
        {"bank_name": {"value": "बैंक ऑफ बड़ौदा", "value_en": "Bank of Baroda", "script": "Deva"}},
    )
    ref = harness.build_field_reference(gt)["bank_name"]
    assert ref.native == {harness.normalize_value("बैंक ऑफ बड़ौदा")}
    assert ref.en == {harness.normalize_value("Bank of Baroda")}
    assert ref.is_bilingual is True
    assert ref.accepted == ref.native | ref.en


def test_the_bilingual_block_recovers_a_value_its_line_only_embeds(harness):
    """`fields_bilingual` carries the identifier; the line carries the sentence."""
    gt = _gt(
        [{"text": "खाते क्रमांक 8234484955318053 आहे.", "field": "account_number", "bbox": None}],
        {},
        {
            "account_number": {
                "value": "8234484955318053",
                "value_en": "8234484955318053",
                "script": "Latn",
            }
        },
    )
    ref = harness.build_field_reference(gt)["account_number"]
    assert harness.normalize_value("8234484955318053") in ref.native
    # the whole sentence stays acceptable, so an extractive pipeline is not punished
    assert harness.normalize_value("खाते क्रमांक 8234484955318053 आहे.") in ref.native
    assert ref.is_bilingual is False  # an ASCII number is not a translation of itself


def test_field_reference_still_works_without_a_bilingual_block(harness):
    """A pre-1.2.0 corpus must still score — the block is additive, not required."""
    gt = _gt([{"text": "45,000", "field": "disputed_amount", "bbox": None}], {"disputed_amount": 45000})
    ref = harness.build_field_reference(gt)["disputed_amount"]
    assert harness.normalize_value("45000") in ref.native


def test_field_reference_ignores_keys_never_rendered_on_the_page(harness):
    gt = _gt(
        [{"text": "Signature", "field": None, "bbox": None}],
        {"card_last_four": "4242", "ground_en": "Unauthorised transaction"},
    )
    assert harness.build_field_reference(gt) == {}


def test_field_reference_does_not_duplicate_a_value_under_two_names(harness):
    """mobile and complainant_contact are the same string; only the line's name survives."""
    gt = _gt(
        [{"text": "9106990984", "field": "complainant_contact", "bbox": None}],
        {"mobile": "9106990984"},
    )
    reference = harness.build_field_reference(gt)
    assert set(reference) == {"complainant_contact"}


# ------------------------------------------------------- envelope vocabulary


def test_envelope_aliases_cover_the_four_names_the_corpus_spells_differently(harness):
    """Q6: without these, `amount` and friends score FP-by-construction on every page."""
    assert harness.ENVELOPE_FIELD_ALIASES == {
        "amount": "disputed_amount",
        "utr_rrn": "transaction_reference",
        "card_fragment": "card_last_four",
        "prior_complaint_reference": "prior_complaint_ref",
    }
    # complaint_date is a different fact from transaction_date and must not be aliased
    assert "complaint_date" not in harness.ENVELOPE_FIELD_ALIASES


# -------------------------------------------------------- escalation reference


@pytest.mark.parametrize(
    "expectation,hand_ratio,mode,expected",
    [
        ("clean", 0.0, "quality_or_hand", False),
        ("clean", 1.0, "quality_or_hand", True),
        ("degraded", 0.0, "quality_or_hand", True),
        ("bounce", 0.0, "quality", True),
        ("clean", 1.0, "quality", False),
        ("clean", 1.0, "hand", True),
        ("degraded", 0.0, "hand", False),
    ],
)
def test_reference_escalation_modes(harness, expectation, hand_ratio, mode, expected):
    total = 10
    lines = [
        {"is_handwritten": i < round(hand_ratio * total), "text": "x", "language": "eng"}
        for i in range(total)
    ]
    gt = {"quality_expectation": expectation, "lines": lines}
    assert harness.reference_escalation(gt, mode) is expected


# ---------------------------------------------------------------- PRF and misc


def test_prf_arithmetic(harness):
    prf = harness.PRF()
    prf.tp, prf.fp, prf.fn = 6, 2, 2
    result = prf.as_dict()
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(0.75)
    assert result["f1"] == pytest.approx(0.75)


def test_prf_reports_none_when_undefined(harness):
    result = harness.PRF().as_dict()
    assert result["precision"] is None and result["recall"] is None and result["f1"] is None


def test_render_table_handles_no_rows(harness):
    text = harness.render_table("Empty", ["a", "b"], [])
    assert "(no data)" in text


def test_fmt_renders_none_as_na(harness):
    assert harness._fmt(None) == "n/a"
    assert harness._fmt(0.5) == "0.500"
    assert harness._fmt(math.inf) == "inf"
