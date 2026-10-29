"""Shape, counts and index integrity of the committed corpus.

Does NOT handle: image quality, OCR accuracy, or whether the degradations look realistic.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

EXPECTED_TYPES = [
    "complaint_form_en",
    "complaint_form_deva",
    "complaint_letter_en",
    "complaint_letter_mar",
    "bilingual_form",
    "bank_statement",
    "receipt",
    "email_printout",
    "cheque",
    "photo_evidence",
]
EXPECTED_PROFILES = ["clean", "light", "scan", "heavy", "photo", "bounce"]
VARIANTS = 5
# photo_evidence is forced to the `photo` profile, so its `clean` request collides with
# the real `photo` page and one page per variant is dropped. See FORCED_PHOTO_PROFILE.
EXPECTED_PAGES = (len(EXPECTED_TYPES) - 1) * VARIANTS * len(EXPECTED_PROFILES) + VARIANTS * 5


def test_index_header(corpus_index):
    assert corpus_index["seed"] == 1729
    assert corpus_index["synthetic"] is True
    assert corpus_index["generator_version"]
    assert corpus_index["hand_shaping"] == "cluster"
    assert "iiit-indic-hw-words" in corpus_index["caveat"]
    assert set(corpus_index["caveats"]) == {
        "handwriting",
        "hand_shaping",
        "photo_evidence_forced_profile",
        "field_ground_truth",
    }


def test_page_count_matches_spec(corpus_index):
    assert corpus_index["count"] == EXPECTED_PAGES == 295
    assert len(corpus_index["documents"]) == EXPECTED_PAGES


def test_every_type_and_profile_present(corpus_index):
    docs = corpus_index["documents"]
    by_type = Counter(d["type"] for d in docs)
    assert set(by_type) == set(EXPECTED_TYPES)
    for doc_type in EXPECTED_TYPES:
        expected = 25 if doc_type == "photo_evidence" else 30
        assert by_type[doc_type] == expected, doc_type
    assert set(Counter(d["profile"] for d in docs)) == set(EXPECTED_PROFILES)
    assert set(Counter(d["variant"] for d in docs)) == set(range(VARIANTS))


def test_photo_evidence_has_no_clean_baseline(corpus_index):
    profiles = {d["profile"] for d in corpus_index["documents"] if d["type"] == "photo_evidence"}
    assert "clean" not in profiles
    assert profiles == {"light", "scan", "heavy", "photo", "bounce"}


def test_doc_ids_unique_and_files_present(corpus_dir: Path, corpus_index):
    docs = corpus_index["documents"]
    ids = [d["doc_id"] for d in docs]
    assert len(set(ids)) == len(ids)
    for doc in docs:
        assert (corpus_dir / doc["image"]).is_file(), doc["doc_id"]
        assert (corpus_dir / doc["gt"]).is_file(), doc["doc_id"]
    assert len(list(corpus_dir.rglob("*.jpg"))) == len(docs)
    assert len(list(corpus_dir.rglob("*.gt.json"))) == len(docs)


def test_quality_expectation_follows_profile(corpus_index):
    expected = {
        "clean": "clean",
        "light": "clean",
        "scan": "degraded",
        "heavy": "degraded",
        "photo": "degraded",
        "bounce": "bounce",
    }
    for doc in corpus_index["documents"]:
        assert doc["quality_expectation"] == expected[doc["profile"]], doc["doc_id"]


def test_bilingual_form_is_the_only_code_mixed_type(corpus_index):
    mixed = {d["type"] for d in corpus_index["documents"] if d["is_code_mixed"]}
    assert mixed == {"bilingual_form"}


@pytest.mark.parametrize("doc_type", ["complaint_form_deva", "complaint_letter_mar", "bilingual_form"])
def test_devanagari_text_present(corpus_dir: Path, corpus_index, doc_type):
    docs = [d for d in corpus_index["documents"] if d["type"] == doc_type]
    assert docs
    for doc in docs:
        gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
        deva = [ln for ln in gt["lines"] if ln["script"] == "Deva"]
        assert deva, doc["doc_id"]
        # actual Devanagari codepoints, not a label with Latin text behind it
        for line in deva:
            assert any("ऀ" <= ch <= "ॿ" for ch in line["text"]), (doc["doc_id"], line["text"])


def test_line_level_language_and_handwriting_are_on_the_line(corpus_dir: Path, corpus_index):
    """Invariant 1: script/language/is_handwritten live on OCRLine, never on the page."""
    doc = next(d for d in corpus_index["documents"] if d["type"] == "bilingual_form")
    gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
    for line in gt["lines"]:
        assert {"text", "bbox", "script", "language", "is_handwritten", "field"} <= set(line)
    printed_latin = [ln for ln in gt["lines"] if not ln["is_handwritten"] and ln["script"] == "Latn"]
    hand_deva = [ln for ln in gt["lines"] if ln["is_handwritten"] and ln["script"] == "Deva"]
    assert printed_latin, "bilingual form must carry printed English label lines"
    assert hand_deva, "bilingual form must carry handwritten Devanagari value lines"
    assert all(ln["language"] == "eng" for ln in printed_latin)
    assert all(ln["language"] == "mar" for ln in hand_deva)
    assert gt["is_code_mixed"] is True


def test_bboxes_are_eight_finite_numbers(corpus_dir: Path, corpus_index):
    for doc in corpus_index["documents"]:
        gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
        for line in gt["lines"]:
            bbox = line["bbox"]
            assert len(bbox) == 8, doc["doc_id"]
            assert all(isinstance(v, (int, float)) for v in bbox), doc["doc_id"]
            assert all(math.isfinite(v) and abs(v) < 1e6 for v in bbox), doc["doc_id"]


def test_clean_profile_bboxes_lie_inside_the_image(corpus_dir: Path, corpus_index):
    """Un-rotated pages must have every box in frame.

    Skewed profiles legitimately push corners out of frame: rotation uses expand=False,
    so the page is cropped while the ground truth keeps the true geometry. That is the
    honest label, not a bug, so only `clean` is asserted here.
    """
    from PIL import Image

    clean = [d for d in corpus_index["documents"] if d["profile"] == "clean"]
    assert clean
    for doc in clean:
        gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
        with Image.open(corpus_dir / doc["image"]) as img:
            width, height = img.size
        for line in gt["lines"]:
            xs, ys = line["bbox"][0::2], line["bbox"][1::2]
            assert all(0 <= x <= width for x in xs), (doc["doc_id"], line["text"])
            assert all(0 <= y <= height for y in ys), (doc["doc_id"], line["text"])


def test_degradations_recorded_for_degraded_profiles(corpus_dir: Path, corpus_index):
    for doc in corpus_index["documents"]:
        gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
        if doc["profile"] == "clean":
            assert gt["degradations"] == [], doc["doc_id"]
        else:
            assert gt["degradations"], doc["doc_id"]
            assert any(d.startswith("skew:") for d in gt["degradations"]), doc["doc_id"]
            assert any(d.startswith("blur:") for d in gt["degradations"]), doc["doc_id"]


def test_ifsc_codes_are_structurally_valid(corpus_dir: Path, corpus_index):
    """4 alpha + '0' + 6 alnum, with a real bank prefix — so Stage 5 is genuinely exercised."""
    import re

    prefixes = {code for _, _, code in __import__("generate_test_corpus").BANKS}
    pattern = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
    checked = 0
    for doc in corpus_index["documents"]:
        gt = json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))
        ifsc = gt["fields"].get("ifsc_code")
        if not ifsc:
            continue
        assert pattern.match(ifsc), (doc["doc_id"], ifsc)
        assert ifsc[:4] in prefixes, (doc["doc_id"], ifsc)
        checked += 1
    assert checked > 0


# ------------------------------------------------------- field ground truth (Q6)

#: The minimum each type must reference, for the four that used to emit `fields: {}`.
#: OPEN-QUESTIONS Q6 — before this, these four contributed no field reference at all and
#: their per-type field F1 was 0.000 by construction.
REQUIRED_FIELDS_BY_TYPE = {
    "receipt": {"transaction_date", "bank_name", "disputed_amount", "transaction_reference",
                "card_last_four"},
    "photo_evidence": {"transaction_date", "bank_name", "disputed_amount",
                       "transaction_reference", "card_last_four"},
    "email_printout": {"transaction_date", "prior_complaint_ref"},
    "cheque": {"transaction_date", "bank_name", "account_number", "ifsc_code", "micr_code",
               "disputed_amount", "complainant_name"},
}


def _gt_of(corpus_dir: Path, doc) -> dict:
    return json.loads((corpus_dir / doc["gt"]).read_text(encoding="utf-8"))


def test_no_document_type_emits_an_empty_fields_dict(corpus_dir: Path, corpus_index):
    """Q6: receipt/email_printout/cheque/photo_evidence used to return {}."""
    seen: dict[str, set[str]] = {}
    for doc in corpus_index["documents"]:
        gt = _gt_of(corpus_dir, doc)
        assert gt["fields"], doc["doc_id"]
        assert gt["fields_bilingual"], doc["doc_id"]
        seen.setdefault(doc["type"], set()).update(gt["fields_bilingual"])
    for doc_type, required in REQUIRED_FIELDS_BY_TYPE.items():
        assert required <= seen[doc_type], (doc_type, sorted(required - seen[doc_type]))


def test_every_bilingual_field_is_linked_to_a_line_that_was_drawn(corpus_dir: Path, corpus_index):
    """A key in fields_bilingual is a promise about pixels, not a note about intent."""
    for doc in corpus_index["documents"]:
        gt = _gt_of(corpus_dir, doc)
        drawn = {ln["field"] for ln in gt["lines"] if ln["field"]}
        assert set(gt["fields_bilingual"]) <= drawn, doc["doc_id"]


def test_bilingual_entries_have_the_full_shape(corpus_dir: Path, corpus_index):
    for doc in corpus_index["documents"]:
        gt = _gt_of(corpus_dir, doc)
        for name, spec in gt["fields_bilingual"].items():
            assert set(spec) == {"value", "value_en", "script"}, (doc["doc_id"], name)
            assert isinstance(spec["value"], str) and spec["value"], (doc["doc_id"], name)
            assert isinstance(spec["value_en"], str) and spec["value_en"], (doc["doc_id"], name)
            assert spec["script"] in {"Latn", "Deva"}, (doc["doc_id"], name)


def test_devanagari_values_carry_a_distinct_english_equivalent(corpus_dir: Path, corpus_index):
    """The Q6 defect: gt fields carried English only, so a Deva page had no native value."""
    checked = 0
    for doc in corpus_index["documents"]:
        gt = _gt_of(corpus_dir, doc)
        for name, spec in gt["fields_bilingual"].items():
            if spec["script"] != "Deva":
                continue
            assert any("ऀ" <= ch <= "ॿ" for ch in spec["value"]), (doc["doc_id"], name)
            assert not any("ऀ" <= ch <= "ॿ" for ch in spec["value_en"]), (doc["doc_id"], name)
            assert spec["value"] != spec["value_en"], (doc["doc_id"], name)
            checked += 1
    assert checked > 0


def test_latin_valued_fields_have_value_equal_to_value_en(corpus_dir: Path, corpus_index):
    for doc in corpus_index["documents"]:
        gt = _gt_of(corpus_dir, doc)
        for name, spec in gt["fields_bilingual"].items():
            if spec["script"] == "Latn":
                assert spec["value"] == spec["value_en"], (doc["doc_id"], name)


@pytest.mark.parametrize(
    "doc_type,field",
    [
        ("complaint_form_deva", "bank_name"),
        ("complaint_form_deva", "complainant_name"),
        ("complaint_letter_mar", "relief_sought"),
        ("bilingual_form", "complaint_summary"),
    ],
)
def test_named_devanagari_fields_are_bilingual(corpus_dir: Path, corpus_index, doc_type, field):
    docs = [d for d in corpus_index["documents"] if d["type"] == doc_type]
    assert docs
    for doc in docs:
        spec = _gt_of(corpus_dir, doc)["fields_bilingual"][field]
        assert spec["script"] == "Deva", doc["doc_id"]


def test_the_cheque_micr_code_is_nine_digits_and_on_the_page(corpus_dir: Path, corpus_index):
    """The cheque draws a real MICR band now, so `micr_code` has a reference at last."""
    docs = [d for d in corpus_index["documents"] if d["type"] == "cheque"]
    assert docs
    for doc in docs:
        gt = _gt_of(corpus_dir, doc)
        micr = gt["fields"]["micr_code"]
        assert len(micr) == 9 and micr.isdigit(), (doc["doc_id"], micr)
        assert micr in {ln["text"] for ln in gt["lines"]}, doc["doc_id"]


def test_the_cheque_does_not_claim_a_reference_it_only_truncates(corpus_dir: Path, corpus_index):
    """The bottom band prints six digits of the RRN, so the full value is not on the page."""
    for doc in corpus_index["documents"]:
        if doc["type"] != "cheque":
            continue
        gt = _gt_of(corpus_dir, doc)
        assert "transaction_reference" not in gt["fields"], doc["doc_id"]


def test_receipt_does_not_claim_the_account_number_it_only_masks(corpus_dir: Path, corpus_index):
    """The slip prints XXXXXX + last four; a full account number would be a false label."""
    for doc in corpus_index["documents"]:
        if doc["type"] not in ("receipt", "photo_evidence"):
            continue
        gt = _gt_of(corpus_dir, doc)
        assert "account_number" not in gt["fields"], doc["doc_id"]
        assert "account_number" not in gt["fields_bilingual"], doc["doc_id"]


def test_adversarial_set_present_with_manifest(corpus_dir: Path):
    adv = corpus_dir / "adversarial"
    manifest = json.loads((adv / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in manifest["expected_behaviour"]:
        assert (adv / name).exists(), name
    assert (adv / "zero_byte.pdf").stat().st_size == 0
    assert (adv / "not_a_pdf.pdf").read_bytes()[:6] == b"GIF89a"
    assert "zip_bomb.zip" in manifest["not_included_generate_locally"]


def test_gt_files_are_utf8_and_not_ascii_escaped(corpus_dir: Path, corpus_index):
    doc = next(d for d in corpus_index["documents"] if d["type"] == "complaint_letter_mar")
    raw = (corpus_dir / doc["gt"]).read_bytes()
    assert b"\\u09" not in raw, "gt.json must be written as real UTF-8, not \\uXXXX escapes"
    text = raw.decode("utf-8")
    assert any("ऀ" <= ch <= "ॿ" for ch in text)
