"""Bank names split across logo lines remain one evidence-backed entity."""

import pytest

from app.contracts.ocr import OCRLine, OCRPage, OCRTier, bbox_from_xyxy
from app.stages.s5_discover import discover_fields
from app.stages.s5_validators import run_validators


def line(index, text, x, y, width=160, height=28):
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(x, y, x + width, y + height),
        confidence=0.99,
        script="Latn",
        language="eng",
        is_handwritten=False,
        line_index=index,
    )


def page(lines):
    return OCRPage(
        page_no=1, width_px=1000, height_px=1200, tier=OCRTier.TIER1, lines=lines
    )


@pytest.mark.parametrize(
    ("first", "tail", "expected"),
    [
        ("Union Bank", "of India", "Union Bank of India"),
        ("State Bank", "of India", "State Bank of India"),
        ("Punjab National", "Bank", "Punjab National Bank"),
    ],
)
def test_stacked_name_joins_across_an_intervening_heading(first, tail, expected):
    p = page(
        [
            line(0, first, 80, 30),
            line(1, "BANK STATEMENT", 300, 30, 300),
            line(2, tail, 170, 57, 65, 16),
        ]
    )
    report = run_validators([p], validators=["bank_name"])
    hits = report.hits_for("bank_name")
    assert len(hits) == 1
    hit = hits[0]
    assert hit.value == expected
    assert hit.normalized == expected
    assert hit.validator_passed is True
    assert [span.line_index for span in hit.evidence.spans] == [0, 2]
    assert {span.line_index for span in report.masked_spans(1)} == {0, 2}
    for span in hit.evidence.spans:
        assert p.line_by_index(span.line_index).text[span.start : span.end]
    fields = discover_fields([p], report.fields(), requested_fields=("bank_name",))
    assert list(fields) == ["bank_name"]


def test_three_line_bank_name_uses_all_source_fragments():
    p = page(
        [line(0, "Union", 90, 20), line(1, "Bank", 90, 50), line(2, "of India", 90, 80)]
    )
    hit = run_validators([p], validators=["bank_name"]).hits_for("bank_name")[0]
    assert hit.value == "Union Bank of India"
    assert len(hit.evidence.spans) == 3


@pytest.mark.parametrize(("x", "y"), [(600, 57), (170, 700)])
def test_distant_or_separate_column_suffix_does_not_complete_name(x, y):
    p = page([line(0, "Union Bank", 80, 30), line(1, "of India", x, y, 65, 16)])
    hits = run_validators([p], validators=["bank_name"]).hits_for("bank_name")
    assert [hit.value for hit in hits] == ["Union Bank"]
    assert not hits[0].evidence.spans


def test_suffix_alone_is_not_a_bank_but_a_complete_bank_of_india_is_valid():
    p = page([line(0, "of India", 80, 30), line(1, "Bank of India", 80, 300)])
    hits = run_validators([p], validators=["bank_name"]).hits_for("bank_name")
    assert [hit.value for hit in hits] == ["Bank of India"]


def test_regulator_split_over_lines_is_still_excluded():
    p = page([line(0, "Reserve", 80, 30), line(1, "Bank of India", 80, 60)])
    assert not run_validators([p], validators=["bank_name"]).hits_for("bank_name")


def test_distinct_banks_are_not_merged_into_one_name():
    p = page(
        [line(0, "Union Bank of India", 80, 30, 240), line(1, "Bank of Baroda", 80, 65)]
    )
    hits = run_validators([p], validators=["bank_name"]).hits_for("bank_name")
    assert {hit.value for hit in hits} == {"Union Bank of India", "Bank of Baroda"}


def test_unlabeled_table_headers_are_not_each_others_values():
    p = page(
        [
            line(0, "Txn Date", 10, 300, 100),
            line(1, "Value Date", 125, 300, 100),
            line(2, "Description", 240, 300, 130),
            line(3, "Debit", 385, 300, 100),
        ]
    )
    assert not discover_fields([p])
