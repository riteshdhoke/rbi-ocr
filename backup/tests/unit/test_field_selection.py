"""Requested fields remain open-ended, grounded and separate from extra discoveries."""

import pytest

from app.contracts.envelope import Evidence, Field
from app.contracts.ocr import TextSpan, bbox_from_xyxy
from app.field_selection import parse_requested_fields, partition_fields
from app.stages.s5_discover import discover_fields
from tests.unit.test_s5_discover import page


def field(value, *, confidence=0.8, validated=None):
    return Field(
        value=value,
        confidence=confidence,
        validator_passed=validated,
        source="regex_validator" if validated is not None else "ocr_label",
        evidence=Evidence(
            page_no=1,
            bbox=bbox_from_xyxy(0, 0, 20, 20),
            span=TextSpan(line_index=0, start=0, end=len(value)),
        ),
    )


def test_csv_normalizes_spaces_case_duplicates_unicode_and_empty_entries():
    assert parse_requested_fields(" Mobile, account number, MOBILE,, मोबाइल, ") == (
        "account_number",
        "mobile",
        "मोबाइल",
    )
    assert parse_requested_fields(" , , ") == ()
    with pytest.raises(ValueError):
        parse_requested_fields("mobile, ???")


def test_requested_fields_group_distinct_values_and_leave_extra_fields():
    fields = {
        "name_of_complainant": field("Anita"),
        "mobile": field("123"),
        "mobile_2": field("456"),
        "mobile_number": field("123", validated=True),
        "policy_number": field("P-42"),
    }
    selected, other, missing = partition_fields(
        fields, parse_requested_fields("complainant name, mobile, email")
    )
    assert [f.value for f in selected["complainant_name"]] == ["Anita"]
    assert [f.value for f in selected["mobile"]] == ["123", "456"]
    assert selected["mobile"][0].validator_passed is True
    assert selected["email"] == []
    assert missing == ("email",)
    assert list(other) == ["policy_number"]


def test_blank_list_keeps_all_discoveries_as_other_fields():
    fields = {"custom_id": field("A-1")}
    selected, other, missing = partition_fields(fields, ())
    assert selected == {}
    assert other == fields
    assert not missing


def test_matching_does_not_guess_synonyms_or_substrings():
    selected, other, missing = partition_fields({"email": field("a@b.in")}, ("mail",))
    assert selected == {"mail": []}
    assert missing == ("mail",)
    assert "email" in other


def test_requested_label_guides_offline_extraction_on_next_line():
    p = page("Loyalty membership", "M-123")
    assert not discover_fields([p])
    selected = discover_fields([p], requested_fields=("loyalty_membership",))
    assert selected["loyalty_membership"].value == "M-123"
    assert selected["loyalty_membership"].evidence.span.line_index == 1


def test_numeric_label_components_are_not_confused_with_other_fields():
    selected, other, missing = partition_fields(
        {"invoice_2025": field("old")}, ("invoice_2026",)
    )
    assert selected["invoice_2026"] == []
    assert missing == ("invoice_2026",)
    assert "invoice_2025" in other


def test_requested_word_inside_narrative_is_not_a_label():
    p = page('My name is Anita', 'I called the bank yesterday')
    assert not discover_fields([p], requested_fields=('name',))
