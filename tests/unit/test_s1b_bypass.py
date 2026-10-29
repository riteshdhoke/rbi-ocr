"""Stage 1b — digital-PDF text-layer bypass and its junk guard.

Offline by construction, like ``test_s1_ingest``: every PDF is assembled from
:func:`tests.unit.test_s1_ingest.build_pdf`, so the suite needs no fixture files
and no PDF writer.  That matters most for the junk cases — the point of the
guard is a text layer whose *characters* are wrong while the page looks fine,
and the only honest way to produce one is to control the content stream byte by
byte.

The geometry tests are the other half.  A bbox from the bypass path and a bbox
from the detector must land in the same frame, or an evidence crop shows a
reviewer the wrong corner of the page — and that failure is silent, because both
numbers are individually plausible.
"""

from __future__ import annotations

import unicodedata

import pypdfium2 as pdfium
import pytest

from app.contracts.errors import EncryptedPDFError, MalformedPDFError
from app.contracts.ocr import OCRTier
from app.stages.s1_ingest import PDF_POINTS_PER_INCH, RASTER_DPI, open_pdf
from app.stages.s1b_bypass import (
    COMMON_SCRIPT,
    INHERITED_SCRIPT,
    UNKNOWN_SCRIPT,
    BypassThresholds,
    BypassVerdict,
    bypass_document,
    character_profile,
    detect_bypass,
    extract_text_lines,
    is_junk_text_layer,
    script_of,
    text_rect_to_bbox,
)
from tests.unit.test_s1_ingest import (
    build_encrypted_pdf,
    build_pdf,
    text_content,
)

# --------------------------------------------------------------------------- #
# Page bodies
# --------------------------------------------------------------------------- #

#: Enough Latin prose, in enough rectangles, to clear every floor in
#: :class:`BypassThresholds`.
PROSE_LINES: tuple[bytes, ...] = (
    b"COMPLAINT TO THE BANKING OMBUDSMAN",
    b"The bank debited my account twice for the same transaction",
    b"on 14 January 2026 and has not reversed the duplicate entry.",
    b"I request that the amount be credited back without delay.",
)


def prose_pdf(**kwargs: object) -> bytes:
    """A digital PDF whose text layer is unambiguously usable."""
    return build_pdf(text_content(list(PROSE_LINES), leading=40), **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def digital_pdf() -> bytes:
    return prose_pdf()


def first_page_result(data: bytes, **kwargs: object):
    """``detect_bypass`` over page 1 of ``data``."""
    with open_pdf(data) as document:
        page = document[0]
        try:
            return detect_bypass(page, 1, **kwargs)  # type: ignore[arg-type]
        finally:
            page.close()


# --------------------------------------------------------------------------- #
# Character classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("A", "Latn"),
        ("z", "Latn"),
        ("é", "Latn"),
        ("क", "Deva"),
        ("ॐ", "Deva"),
        ("অ", "Beng"),
        ("த", "Taml"),
        ("Ω", "Grek"),
        ("Д", "Cyrl"),
        ("ع", "Arab"),
        ("中", "Hani"),
        # Not letters — every one of these is ordinary form furniture.
        ("7", COMMON_SCRIPT),
        (" ", COMMON_SCRIPT),
        ("₹", COMMON_SCRIPT),
        ("/", COMMON_SCRIPT),
        ("", COMMON_SCRIPT),
        # A letter in no range the table covers.
        ("ᐁ", UNKNOWN_SCRIPT),  # CANADIAN SYLLABICS E
    ],
)
def test_script_of(char: str, expected: str) -> None:
    assert script_of(char) == expected


def test_character_profile_counts_by_unicode_category() -> None:
    stats = character_profile("Ab 7.", rect_count=3)
    assert stats.char_count == 6
    assert stats.letters == 2
    assert stats.digits == 1
    assert stats.punctuation == 1
    assert stats.whitespace == 1
    assert stats.other == 1  # U+0001, category Cc
    assert stats.rect_count == 3
    assert stats.non_space == 5
    assert stats.dominant_script == "Latn"
    assert stats.dominant_script_share == pytest.approx(1.0)


def test_character_profile_replacement_chars_count_as_other_too() -> None:
    stats = character_profile("ab�")
    assert stats.replacement == 1
    assert stats.other == 1
    assert stats.replacement_share == pytest.approx(1 / 3)


def test_character_profile_script_distribution_is_over_letters_only() -> None:
    stats = character_profile("abcd" + "कखगघ" + "1234")
    assert stats.script_distribution == {"Latn": 0.5, "Deva": 0.5}
    assert stats.dominant_script_share == pytest.approx(0.5)
    assert stats.letter_share == pytest.approx(8 / 12)
    assert stats.letter_or_digit_share == pytest.approx(1.0)


def test_character_profile_of_empty_text_is_all_zero() -> None:
    stats = character_profile("")
    assert stats.char_count == 0
    assert stats.non_space == 0
    assert stats.dominant_script is None
    # The share properties must not divide by zero.
    assert stats.other_share == 0.0
    assert stats.letter_share == 0.0
    assert stats.letter_or_digit_share == 0.0
    assert stats.replacement_share == 0.0


# --------------------------------------------------------------------------- #
# The junk guard
# --------------------------------------------------------------------------- #


CLEAN_TEXT = " ".join(line.decode() for line in PROSE_LINES)


def test_clean_latin_layer_is_not_junk() -> None:
    assert is_junk_text_layer(character_profile(CLEAN_TEXT)) is None


def test_clean_devanagari_layer_is_not_junk() -> None:
    """Regression: matras are ``Mn``/``Mc``, not ``Lo``.

    Counting them as control characters made every Marathi and Hindi digital
    PDF measure ~41% "unassigned characters" and refused the bypass on exactly
    the pages the service exists to read.
    """
    text = "बँकेने माझ्या खात्यातून दोनदा रक्कम वजा केली आहे " * 3
    stats = character_profile(text)
    assert stats.dominant_script == "Deva"
    assert stats.other == 0
    assert stats.letter_share > 0.9
    assert is_junk_text_layer(stats) is None


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("ा", "Deva"),  # U+093E DEVANAGARI VOWEL SIGN AA, category Mc
        ("ँ", "Deva"),  # U+0901 CHANDRABINDU, category Mn
        ("्", "Deva"),  # U+094D VIRAMA, category Mn
        ("ৌ", "Beng"),  # U+09CC BENGALI VOWEL SIGN AU, category Mc
        ("́", INHERITED_SCRIPT),  # COMBINING ACUTE — evidence of nothing
    ],
)
def test_combining_marks_carry_their_block_script(char: str, expected: str) -> None:
    assert script_of(char) == expected


def test_inherited_marks_abstain_from_the_script_vote() -> None:
    """NFD Latin must not scatter its accents into an unattributable bucket."""
    decomposed = unicodedata.normalize("NFD", "café résumé " * 6)
    assert any(unicodedata.category(c) == "Mn" for c in decomposed)
    stats = character_profile(decomposed)
    assert stats.dominant_script == "Latn"
    assert stats.dominant_script_share == pytest.approx(1.0)
    assert INHERITED_SCRIPT not in stats.script_distribution
    assert is_junk_text_layer(stats) is None
    # Marks count toward `letters`, so the letter-share floor is not tripped.
    assert stats.letters > sum(1 for c in decomposed if c.isalpha())


@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        pytest.param("�" * 4 + CLEAN_TEXT, "replacement", id="replacement_chars"),
        pytest.param("" * 40 + CLEAN_TEXT, "control", id="control_chars"),
        pytest.param("" * 40 + CLEAN_TEXT, "control", id="private_use"),
        pytest.param("A" * 200, "one character is", id="one_glyph_repeated"),
        pytest.param(".-.," * 60, "letters or digits", id="punctuation_soup"),
        pytest.param(
            "12345678 " * 20 + "ab", "only", id="digits_without_letters"
        ),
        pytest.param(
            "abcdefgh" + "ΑΒΓΔΕΖΗΘ" + "АБВГДЕЖЗ" + "אבגדהוזח",
            "scattered across scripts",
            id="cid_scattered_across_scripts",
        ),
        pytest.param(
            "ᐁᐂᐃᐄ " * 10,
            "unattributable",
            id="letters_in_no_known_range",
        ),
        pytest.param("   ", "no non-space", id="whitespace_only"),
    ],
)
def test_junk_layers_are_refused_with_a_reason(
    text: str, expected_fragment: str
) -> None:
    reason = is_junk_text_layer(character_profile(text))
    assert reason is not None
    assert expected_fragment in reason


def test_expected_script_turns_the_guard_into_a_stronger_test() -> None:
    """A perfectly self-consistent Latin layer on a page Stage 2 read as Deva."""
    stats = character_profile(CLEAN_TEXT)
    assert is_junk_text_layer(stats) is None
    reason = is_junk_text_layer(stats, expected_script="Deva")
    assert reason is not None
    assert "Latn" in reason and "Deva" in reason


def test_thresholds_are_tunable() -> None:
    text = "A" * 60 + "bcdefghij" * 6
    strict = BypassThresholds(max_single_char_share=0.3)
    lenient = BypassThresholds(max_single_char_share=0.95)
    assert is_junk_text_layer(character_profile(text), strict) is not None
    assert is_junk_text_layer(character_profile(text), lenient) is None


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

A4_WIDTH_PT = 595.0
A4_HEIGHT_PT = 842.0
SCALE = RASTER_DPI / PDF_POINTS_PER_INCH


def test_text_rect_to_bbox_unrotated_flips_the_y_axis() -> None:
    """PDF origin is bottom-left; raster origin is top-left."""
    bbox = text_rect_to_bbox(
        (60.0, 780.0, 300.0, 800.0),
        page_width_pt=A4_WIDTH_PT,
        page_height_pt=A4_HEIGHT_PT,
        rotation=0,
    )
    assert bbox == pytest.approx(
        [
            60.0 * SCALE,
            (A4_HEIGHT_PT - 800.0) * SCALE,
            300.0 * SCALE,
            (A4_HEIGHT_PT - 800.0) * SCALE,
            300.0 * SCALE,
            (A4_HEIGHT_PT - 780.0) * SCALE,
            60.0 * SCALE,
            (A4_HEIGHT_PT - 780.0) * SCALE,
        ]
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270, 360, -90, 450])
def test_text_rect_to_bbox_stays_inside_the_rotated_canvas(rotation: int) -> None:
    """Whatever /Rotate says, the bbox must land on the rendered bitmap."""
    if rotation % 180 == 90:
        canvas_w, canvas_h = A4_HEIGHT_PT * SCALE, A4_WIDTH_PT * SCALE
    else:
        canvas_w, canvas_h = A4_WIDTH_PT * SCALE, A4_HEIGHT_PT * SCALE

    bbox = text_rect_to_bbox(
        (60.0, 700.0, 500.0, 730.0),
        page_width_pt=A4_WIDTH_PT,
        page_height_pt=A4_HEIGHT_PT,
        rotation=rotation,
    )
    xs = bbox[0::2]
    ys = bbox[1::2]
    assert min(xs) >= -1e-6
    assert min(ys) >= -1e-6
    assert max(xs) <= canvas_w + 1e-6
    assert max(ys) <= canvas_h + 1e-6
    # Non-degenerate, and canonical clockwise-from-top-left order.
    assert max(xs) > min(xs) and max(ys) > min(ys)
    assert bbox[0] == bbox[6] and bbox[2] == bbox[4]
    assert bbox[1] == bbox[3] and bbox[5] == bbox[7]


def test_text_rect_to_bbox_rejects_a_non_quadrant_rotation() -> None:
    with pytest.raises(ValueError, match="multiple of 90"):
        text_rect_to_bbox(
            (0.0, 0.0, 10.0, 10.0),
            page_width_pt=A4_WIDTH_PT,
            page_height_pt=A4_HEIGHT_PT,
            rotation=45,
        )


def test_dpi_scales_the_bbox_linearly() -> None:
    rect = (60.0, 700.0, 500.0, 730.0)
    kwargs = {
        "page_width_pt": A4_WIDTH_PT,
        "page_height_pt": A4_HEIGHT_PT,
        "rotation": 0,
    }
    low = text_rect_to_bbox(rect, dpi=100.0, **kwargs)  # type: ignore[arg-type]
    high = text_rect_to_bbox(rect, dpi=200.0, **kwargs)  # type: ignore[arg-type]
    assert high == pytest.approx([value * 2.0 for value in low])


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_extract_text_lines_lifts_the_layer_verbatim(digital_pdf: bytes) -> None:
    with open_pdf(digital_pdf) as document:
        page = document[0]
        try:
            lines, raw = extract_text_lines(page)
        finally:
            page.close()

    joined = " ".join(line.text for line in lines)
    for source in PROSE_LINES:
        assert source.decode() in joined
    assert "COMPLAINT" in raw


def test_extracted_lines_carry_the_contract_the_ocr_path_carries(
    digital_pdf: bytes,
) -> None:
    """PLAN.md invariant 1: language is Stage 2b's, never Stage 1b's."""
    with open_pdf(digital_pdf) as document:
        page = document[0]
        try:
            lines, _ = extract_text_lines(page)
        finally:
            page.close()

    assert lines
    for index, line in enumerate(lines):
        assert line.line_index == index
        assert line.language is None
        assert line.is_handwritten is False
        assert line.confidence == 1.0
        assert line.script == "Latn"
        assert len(line.bbox) == 8
        assert line.text == line.text.strip()


def test_blank_rects_do_not_become_lines() -> None:
    """A rect that decodes to whitespace is not a line, but is still profiled."""
    data = build_pdf(text_content([b"Only one real line", b"   "], leading=40))
    with open_pdf(data) as document:
        page = document[0]
        try:
            lines, raw = extract_text_lines(page)
        finally:
            page.close()
    assert all(line.text.strip() for line in lines)
    assert len(raw) >= sum(len(line.text) for line in lines)


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


def test_a_digital_pdf_is_bypassed(digital_pdf: bytes) -> None:
    result = first_page_result(digital_pdf)
    assert result.verdict is BypassVerdict.BYPASS
    assert result.used is True
    assert result.detail is None
    assert result.page is not None
    assert result.page.tier is OCRTier.BYPASS
    assert result.page.page_no == 1
    assert result.page.dpi_estimate == RASTER_DPI
    assert result.page.width_px == pytest.approx(round(A4_WIDTH_PT * SCALE), abs=1)
    assert result.page.height_px == pytest.approx(round(A4_HEIGHT_PT * SCALE), abs=1)
    assert result.stats.dominant_script == "Latn"


def test_a_scanned_page_has_no_text_layer() -> None:
    """A content stream that draws no glyphs — the shape of every real scan."""
    box = b"0.5 0.5 0.5 rg\n100 100 400 600 re\nf\n"
    result = first_page_result(build_pdf(box))
    assert result.verdict is BypassVerdict.NO_TEXT_LAYER
    assert result.used is False
    assert result.page is None
    assert result.stats.char_count == 0


def test_a_stray_label_is_sparse_not_usable() -> None:
    result = first_page_result(build_pdf(text_content([b"Page 1"])))
    assert result.verdict is BypassVerdict.SPARSE_TEXT_LAYER
    assert result.page is None
    assert result.detail is not None and "floor is" in result.detail


def test_min_rects_floor_rejects_one_long_line() -> None:
    long_line = b"A single very long line of text that easily clears min_chars"
    result = first_page_result(build_pdf(text_content([long_line])))
    assert result.verdict is BypassVerdict.SPARSE_TEXT_LAYER


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param([b"A" * 40, b"A" * 40, b"A" * 40], id="one_glyph_repeated"),
        pytest.param(
            [b".-.-.-.-.-.-.-.-.-.-", b"..;;..;;..;;..;;", b",,,,....,,,,....,,,,"],
            id="punctuation_soup",
        ),
    ],
)
def test_a_junk_layer_is_refused_and_the_page_is_discarded(
    lines: list[bytes],
) -> None:
    """No half-usable layer: a refused bypass hands back no page at all."""
    result = first_page_result(build_pdf(text_content(lines, leading=40)))
    assert result.verdict is BypassVerdict.JUNK_TEXT_LAYER
    assert result.used is False
    assert result.page is None
    assert result.detail


def test_script_mismatch_is_distinguished_from_generic_junk(
    digital_pdf: bytes,
) -> None:
    """Same refusal operationally; different rates mean different things."""
    result = first_page_result(digital_pdf, expected_script="Deva")
    assert result.verdict is BypassVerdict.SCRIPT_MISMATCH
    assert result.page is None
    assert result.detail is not None and "Deva" in result.detail


def test_matching_expected_script_still_bypasses(digital_pdf: bytes) -> None:
    result = first_page_result(digital_pdf, expected_script="Latn")
    assert result.verdict is BypassVerdict.BYPASS


def test_thresholds_flow_through_detect_bypass() -> None:
    """A layer that is sparse by default and usable once the floor is lowered."""
    data = build_pdf(text_content([b"Ref CMS/2026/401687", b"Dated 14 Jan 2026"]))
    assert first_page_result(data).verdict is BypassVerdict.SPARSE_TEXT_LAYER
    lowered = BypassThresholds(min_chars=8, min_rects=2)
    assert (
        first_page_result(data, thresholds=lowered).verdict is BypassVerdict.BYPASS
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_bypass_geometry_lands_on_the_rendered_page(rotation: int) -> None:
    """Every bbox must sit inside the OCRPage it is attached to."""
    result = first_page_result(prose_pdf(rotate=rotation))
    assert result.verdict is BypassVerdict.BYPASS
    page = result.page
    assert page is not None
    for line in page.lines:
        xs = line.bbox[0::2]
        ys = line.bbox[1::2]
        assert min(xs) >= -1.0
        assert min(ys) >= -1.0
        assert max(xs) <= page.width_px + 1.0
        assert max(ys) <= page.height_px + 1.0


def test_rotated_pages_report_the_rotated_canvas() -> None:
    upright = first_page_result(prose_pdf(rotate=0)).page
    sideways = first_page_result(prose_pdf(rotate=90)).page
    assert upright is not None and sideways is not None
    assert (sideways.width_px, sideways.height_px) == (
        upright.height_px,
        upright.width_px,
    )


# --------------------------------------------------------------------------- #
# Whole-document driver
# --------------------------------------------------------------------------- #


def test_bypass_document_from_bytes_covers_every_page() -> None:
    data = prose_pdf(pages=3)
    results = bypass_document(data)
    assert [r.page_no for r in results] == [1, 2, 3]
    assert all(r.verdict is BypassVerdict.BYPASS for r in results)


def test_bypass_document_accepts_an_open_document(digital_pdf: bytes) -> None:
    with open_pdf(digital_pdf) as document:
        results = bypass_document(document)
        assert len(results) == 1
        # The caller's document is still usable — we borrowed it, not closed it.
        assert len(document) == 1
    assert results[0].verdict is BypassVerdict.BYPASS


def test_expected_scripts_are_applied_per_page() -> None:
    results = bypass_document(prose_pdf(pages=2), expected_scripts=["Latn", "Deva"])
    assert results[0].verdict is BypassVerdict.BYPASS
    assert results[1].verdict is BypassVerdict.SCRIPT_MISMATCH


def test_short_expected_scripts_list_leaves_later_pages_unconstrained() -> None:
    results = bypass_document(prose_pdf(pages=3), expected_scripts=["Deva"])
    assert results[0].verdict is BypassVerdict.SCRIPT_MISMATCH
    assert all(r.verdict is BypassVerdict.BYPASS for r in results[1:])


def test_bypass_document_raises_the_stage_1_errors_not_a_pdfium_error() -> None:
    with pytest.raises(EncryptedPDFError):
        bypass_document(build_encrypted_pdf())
    with pytest.raises(MalformedPDFError):
        bypass_document(b"%PDF-1.7\nnot really\n")


def test_bypass_document_does_not_leak_a_pdfium_error_type() -> None:
    """The whole reason bytes go through ``open_pdf``."""
    try:
        bypass_document(b"%PDF-1.7\ntruncated")
    except pdfium.PdfiumError:  # pragma: no cover - the bug this guards
        pytest.fail("a raw PdfiumError escaped Stage 1b")
    except MalformedPDFError:
        pass
