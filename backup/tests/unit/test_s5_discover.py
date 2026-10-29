"""Open-field extraction must work with no remote model and keep evidence."""

import json

from app.contracts.envelope import QualityVerdict
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, RawField, bbox_from_xyxy
from app.providers.hf_api import _parse_extraction, build_extraction_prompt
from app.settings import Settings
from app.stages.assemble import PageAssembly, assemble_envelope
from app.stages.s5_discover import discover_fields
from app.stages.s5_slm import apply_grounding_gate, build_extraction_context


def page(*texts):
    return OCRPage(
        page_no=1,
        width_px=1000,
        height_px=1000,
        tier=OCRTier.TIER1,
        lines=[
            OCRLine(
                text=text,
                bbox=bbox_from_xyxy(10, i * 30, 900, i * 30 + 20),
                confidence=0.99,
                script="Latn",
                language="eng",
                is_handwritten=False,
                line_index=i,
            )
            for i, text in enumerate(texts)
        ],
    )


def test_arbitrary_labels_multicell_unicode_and_repeated_values():
    p = page(
        "Customer ID: C-123  Email: a@example.in",
        "Policy number: P-42",
        "Email: b@example.in",
        "Email: a@example.in",
        "मोबाइल: 9876543210",
        "Website: https://example.in:8000/a",
        "Appointment: 10:30",
    )
    fields = discover_fields([p])
    assert fields["customer_id"].value == "C-123"
    assert fields["email_2"].value == "b@example.in"
    assert "email_3" not in fields
    assert fields["policy_number"].value == "P-42"
    assert fields["मोबाइल"].value == "9876543210"
    assert fields["website"].value == "https://example.in:8000/a"
    assert fields["appointment"].value == "10:30"
    for f in fields.values():
        span = f.evidence.span
        assert f.value == p.line_by_index(span.line_index).text[span.start : span.end]
        assert f.validator_passed is None
        assert f.source == "ocr_label"


def test_empty_labels_only_take_a_nearby_value_not_another_label():
    fields = discover_fields(
        [page("Customer ID:", "C-123", "Missing:", "Email: a@b.in")]
    )
    assert fields["customer_id"].value == "C-123"
    assert "missing" not in fields
    assert fields["email"].value == "a@b.in"


def test_default_assembly_discovers_fields_and_opt_out_keeps_legacy_behavior():
    p = page("Unregistered field: visible value")
    args = {
        "document_id": "test",
        "provider_profile": "cpu_node",
        "pages": [PageAssembly(page=p)],
    }
    result = assemble_envelope(**args, settings=Settings(EXTRACT_ALL_FIELDS=True))
    assert result.document.fields["unregistered_field"].value == "visible value"
    legacy = assemble_envelope(**args, settings=Settings(EXTRACT_ALL_FIELDS=False))
    assert not legacy.document.fields


def test_bounced_page_never_contributes_discovered_content():
    result = assemble_envelope(
        document_id="test",
        provider_profile="cpu_node",
        pages=[PageAssembly(page=page(), quality_verdict=QualityVerdict.BOUNCE)],
        settings=Settings(EXTRACT_ALL_FIELDS=True),
    )
    assert result.document is None or not result.document.fields


def test_model_can_discover_custom_and_previously_validator_only_fields():
    p = page("Policy: P-42", "Account: 1234567890")
    ctx = build_extraction_context(
        p,
        document_id="test",
        page_class="unknown",
        primary_script="Latn",
        settings=Settings(EXTRACT_ALL_FIELDS=True),
    )
    assert ctx.discover_fields
    assert "ALL identifiable fields" in build_extraction_prompt(ctx)
    raw = _parse_extraction(
        json.dumps(
            {
                "fields": [
                    {
                        "name": "policy_number",
                        "value": "P-42",
                        "confidence": 0.9,
                        "line_indices": [0],
                    },
                    {
                        "name": "account_number",
                        "value": "1234567890",
                        "confidence": 0.9,
                        "line_indices": [1],
                    },
                ]
            }
        ),
        ctx,
    )
    raw["invented"] = RawField(name="invented", value="does not exist", confidence=1)
    gated = apply_grounding_gate(raw, p, settings=Settings(EXTRACT_ALL_FIELDS=True))
    assert set(gated.fields) == {"policy_number", "account_number"}
    assert gated.fields["account_number"].validator_passed is None
    assert gated.rejected[0].reason == "ungrounded"


def test_numbered_form_labels_work_without_punctuation_or_a_field_registry():
    fields = discover_fields(
        [page("1. Loyalty membership", "M-123", "2. Mobile", "9876543210")]
    )
    assert fields["loyalty_membership"].value == "M-123"
    assert fields["mobile"].value == "9876543210"


def test_ocr_form_cells_on_same_row_have_no_field_allowlist():
    p = page("1. Name of complainant", "Rajesh Sharma", "Custom membership", "M-42")
    for index, line in enumerate(p.lines):
        row = index // 2
        x = 10 if index % 2 == 0 else 500
        line.bbox = bbox_from_xyxy(x, row * 50, x + 350, row * 50 + 25)
    fields = discover_fields([p])
    assert fields["name_of_complainant"].value == "Rajesh Sharma"
    assert fields["custom_membership"].value == "M-42"
    assert fields["custom_membership"].evidence.span.line_index == 3
