"""Letter structure is tested with reference OCR, separately from image recognition."""

import pytest

from app.contracts.envelope import FieldFlag
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, RawField, bbox_from_xyxy
from app.field_selection import parse_requested_fields, partition_fields
from app.settings import Settings
from app.stages.s5_discover import discover_fields
from app.stages.s5_letter import discover_letter_fields
from app.stages.s5_slm import apply_grounding_gate


def line(index, text, x=250, y=None, confidence=0.99, width=400):
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(
            x,
            y if y is not None else index * 32,
            x + width,
            (y if y is not None else index * 32) + 22,
        ),
        confidence=confidence,
        script="Latn",
        language="eng",
        is_handwritten=True,
        line_index=index,
    )


def letter(sender="Morgan Brady", recipient="Tom"):
    # Reference transcription. This fixture does not claim the bundled OCR model
    # reads the supplied photograph without spelling errors.
    lines = [
        line(0, "YEC, LLC", x=10, y=90, width=180),
        line(1, "745 Atlantic Ave", x=10, y=122, width=180),
        line(2, "Floor 3", x=10, y=154, width=180),
        line(3, "Boston, MA 02111", x=10, y=186, width=180),
        line(4, f"Hi {recipient},", y=40),
        line(5, "Just wanted to drop a quick note to say thank you.", y=80),
        line(6, "Please email me at community@yec.co for help.", y=112),
        line(7, "We will keep you informed about new resources.", y=144),
        line(8, "Thanks again,", y=220),
        line(9, sender, y=252),
        line(10, "YEC Community Manager", y=284),
    ]
    return OCRPage(
        page_no=1, width_px=900, height_px=900, lines=lines, tier=OCRTier.TIER1
    )


@pytest.mark.parametrize(
    ("sender", "recipient"), [("Morgan Brady", "Tom"), ("Alex Rivera", "Priya")]
)
def test_extracts_letter_roles_contacts_multiline_address_and_body(sender, recipient):
    p = letter(sender, recipient)
    fields = discover_letter_fields(p)
    assert fields["sender_name"].value == sender
    assert fields["recipient_name"].value == recipient
    assert fields["sender_designation"].value == "YEC Community Manager"
    assert "recipient_designation" not in fields
    assert fields["email"].value == "community@yec.co"
    assert (
        fields["postal_address"].value
        == "YEC, LLC\n745 Atlantic Ave\nFloor 3\nBoston, MA 02111"
    )
    assert len(fields["postal_address"].evidence.spans) == 4
    assert "745 Atlantic" not in fields["letter_content"].value
    assert sender not in fields["letter_content"].value
    assert "Hi " not in fields["letter_content"].value
    assert "Please email me" in fields["letter_content"].value
    for field in fields.values():
        spans = field.evidence.spans or (field.evidence.span,)
        assert field.value == "\n".join(
            p.line_by_index(s.line_index).text[s.start : s.end] for s in spans
        )
        assert field.validator_passed is None


def test_grouping_does_not_confuse_email_with_postal_address_or_sender_with_addressee():
    fields = discover_letter_fields(letter())
    requested = parse_requested_fields(
        "address, addressee, email id, sender, what is the designation of addressee, content of this letter"
    )
    extracted, other, missing = partition_fields(fields, requested)
    assert extracted["addressee"][0].value == "Tom"
    assert extracted["sender"][0].value == "Morgan Brady"
    assert extracted["email_id"][0].value == "community@yec.co"
    assert extracted["address"][0].value.startswith("YEC, LLC\n")
    assert extracted["content_of_this_letter"]
    assert missing == ("what_is_the_designation_of_addressee",)
    assert "sender_designation" in other
    assert "email" not in other


def test_uncertain_address_or_designation_is_not_completed_from_expected_text():
    p = letter()
    p.lines[1] = p.lines[1].model_copy(
        update={"text": "745 Aantic Ave", "confidence": 0.7}
    )
    p.lines[10] = p.lines[10].model_copy(
        update={"text": "YEC Comunitg Manage", "confidence": 0.76}
    )
    fields = discover_letter_fields(p)
    assert "postal_address" not in fields
    assert "sender_designation" not in fields
    assert fields["sender_name"].value == "Morgan Brady"
    assert FieldFlag.LOW_CONFIDENCE in fields["letter_content"].flags


def test_arbitrary_skewed_envelope_fragments_are_not_labels_for_body_sentences():
    p = letter()
    stamp = line(11, "BOS", x=170, y=97, width=45)
    stamp.bbox = [170, 110, 215, 90, 225, 113, 180, 133]
    p.lines.insert(6, stamp)
    fields = discover_fields([p], requested_fields=("address", "addressee"))
    assert "bos" not in fields
    assert not any(key.startswith("745_") for key in fields)
    assert "boston_ma_02111" not in fields


def test_model_can_return_a_grounded_multiline_address_but_not_corrected_ocr():
    p = letter()
    value = ", ".join(item.text for item in p.lines[:4])
    raw = RawField(
        name="address", value=value, confidence=0.95, line_indices=[0, 1, 2, 3]
    )
    result = apply_grounding_gate(
        {"address": raw}, p, settings=Settings(EXTRACT_ALL_FIELDS=True)
    )
    assert result.fields["address"].value == "\n".join(
        item.text for item in p.lines[:4]
    )
    assert len(result.fields["address"].evidence.spans) == 4
    raw = raw.model_copy(update={"value": value.replace("02111", "99999")})
    rejected = apply_grounding_gate(
        {"address": raw}, p, settings=Settings(EXTRACT_ALL_FIELDS=True), max_distance=0
    )
    assert not rejected.fields
    assert rejected.rejected[0].reason == "ungrounded"


def test_grounded_sender_cannot_be_mislabeled_as_addressee():
    p = letter()
    raw = RawField(
        name="addressee", value="Morgan Brady", confidence=1, line_indices=[9]
    )
    result = apply_grounding_gate(
        {"addressee": raw}, p, settings=Settings(EXTRACT_ALL_FIELDS=True)
    )
    assert not result.fields
    assert result.rejected[0].reason == "role_conflict"


def test_broad_name_request_abstains_and_alias_values_do_not_leak_into_other_fields():
    fields = discover_letter_fields(letter())
    requested, _, missing = partition_fields(fields, ("name",))
    assert requested == {"name": []}
    assert missing == ("name",)
    fields["email_address"] = fields["email"]
    fields["email_id"] = fields["email"]
    selected, other, _ = partition_fields(fields, ("email_id",))
    assert len(selected["email_id"]) == 1
    assert not {"email", "email_address", "email_id"} & other.keys()
