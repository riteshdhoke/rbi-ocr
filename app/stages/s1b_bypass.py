"""Stage 1b — digital-PDF text-layer bypass, with a junk-text-layer guard.

BUILD-BRIEF M3.  A PDF produced by a word processor already contains its text;
running OCR over a rasterisation of it is slower, less accurate and bills a
recognition pass for nothing.  This stage lifts those lines straight out of the
text layer and marks the page :attr:`~app.contracts.ocr.OCRTier.BYPASS`.

**The guard is the point.** The failure mode that makes a naive bypass worse
than no bypass is a PDF whose text layer decodes to garbage: a subset-embedded
font with a broken ``ToUnicode`` CMap, a scanner that wrote an invisible OCR
layer of its own with the wrong encoding, or a generator that emitted raw glyph
ids.  The page *looks* perfect and the extracted text is mojibake, so every
downstream stage — validators, LID, the SLM — operates on noise while reporting
high confidence.  :func:`is_junk_text_layer` compares the **character-class
distribution** of the layer against the script those characters claim to be in
and refuses the bypass when they disagree.  A refused bypass is free: the page
falls through to the raster path and is recognised normally.

Rules the guard applies (all tunable through :class:`BypassThresholds`):

* control, format, private-use, surrogate or unassigned characters above a small
  share — the signature of a broken CMap;
* U+FFFD replacement characters above a small share;
* letters spread across more than one script with no clear majority — CID
  values reinterpreted as Unicode land in unrelated blocks;
* an explicitly expected script (from Stage 2, when it has already run) that the
  layer contradicts;
* almost no letters at all, or one character repeated over most of the page.

Geometry: rectangles come from PDFium in **unrotated page coordinates**, points,
origin bottom-left.  :func:`text_rect_to_bbox` maps them into raster pixels at a
given DPI for all four values of ``/Rotate``, so a bbox from the bypass path is
in the same frame as a bbox from the detector and evidence crops line up.

This module does NOT handle:
  * deciding the page's *language* — ``OCRLine.language`` is left ``None`` for
    Stage 2b, exactly as on the OCR path (PLAN.md invariant 1);
  * script **identification from pixels** — Stage 2 owns that.  What happens
    here is script *classification of characters we already have*, which is a
    different and much easier question;
  * rasterisation — Stage 1 owns it, and this stage is deliberately usable
    before any page has been rendered so a 200-page digital PDF never pays for
    pixels;
  * quality scoring.  A bypassed page still goes through Stage 1.5 for its
    ``quality_score`` and flags (PRD P0-3 requires them on *every* page);
  * reading order or paragraph reconstruction.  Lines come out in PDFium's rect
    order, which is content-stream order, not visual order;
  * encrypted or malformed PDFs — Stage 1 rejected those before this stage runs.
"""

from __future__ import annotations

import logging
import unicodedata
from bisect import bisect_right
from collections import Counter as _Counter
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pypdfium2 as pdfium
from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.ocr import OCRLine, OCRPage, OCRTier, bbox_from_xyxy
from app.stages.s1_ingest import PDF_POINTS_PER_INCH, RASTER_DPI, open_pdf

__all__ = [
    "COMMON_SCRIPT",
    "INHERITED_SCRIPT",
    "UNKNOWN_SCRIPT",
    "BypassResult",
    "BypassThresholds",
    "BypassVerdict",
    "TextLayerStats",
    "bypass_document",
    "character_profile",
    "detect_bypass",
    "extract_text_lines",
    "is_junk_text_layer",
    "script_of",
    "text_rect_to_bbox",
]

_LOGGER: Final = logging.getLogger(__name__)

#: ISO 15924 for characters that belong to no single script (digits, spaces,
#: punctuation).  Not a failure — most of a form is punctuation.
COMMON_SCRIPT: Final[str] = "Zyyy"
#: ISO 15924 for a letter we could not attribute.  A high share of these is the
#: strongest single junk signal there is.
UNKNOWN_SCRIPT: Final[str] = "Zzzz"
#: ISO 15924 for a combining mark that takes the script of whatever it attaches
#: to.  A generic acute accent is evidence of nothing, so these characters are
#: counted as letters but abstain from the script vote.
INHERITED_SCRIPT: Final[str] = "Zinh"

#: Blocks whose combining marks are script-inherited rather than script-bearing.
#: Devanagari matras (U+093E..) are deliberately **not** here: they live inside
#: the Devanagari block and are the strongest per-character evidence that a
#: layer really is Devanagari.
_INHERITED_MARK_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x1AB0, 0x1AFF),  # ... Extended
    (0x1DC0, 0x1DFF),  # ... Supplement
    (0x20D0, 0x20FF),  # ... for Symbols
    (0xFE00, 0xFE0F),  # Variation Selectors
    (0xFE20, 0xFE2F),  # Combining Half Marks
)

#: Contiguous ``(first, last, script)`` ranges, sorted by ``first``.  Covers the
#: scripts an RBI complaint can plausibly arrive in plus the ones a mangled CMap
#: scatters into, which is why Greek, Cyrillic and CJK are here: their *presence*
#: is the evidence, not their support.
_SCRIPT_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x0041, 0x005A, "Latn"),
    (0x0061, 0x007A, "Latn"),
    (0x00C0, 0x024F, "Latn"),
    (0x0370, 0x03FF, "Grek"),
    (0x0400, 0x052F, "Cyrl"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0600, 0x06FF, "Arab"),
    (0x0700, 0x074F, "Syrc"),
    (0x0750, 0x077F, "Arab"),
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0D80, 0x0DFF, "Sinh"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x10A0, 0x10FF, "Geor"),
    (0x1E00, 0x1EFF, "Latn"),
    (0x1F00, 0x1FFF, "Grek"),
    (0x3040, 0x309F, "Hira"),
    (0x30A0, 0x30FF, "Kana"),
    (0x3400, 0x4DBF, "Hani"),
    (0x4E00, 0x9FFF, "Hani"),
    (0xA8E0, 0xA8FF, "Deva"),
    (0xAC00, 0xD7AF, "Hang"),
    (0xFB50, 0xFDFF, "Arab"),
    (0xFE70, 0xFEFF, "Arab"),
)

_RANGE_STARTS: Final[tuple[int, ...]] = tuple(r[0] for r in _SCRIPT_RANGES)

_BYPASS_DECISIONS = Counter(
    "bypass_pages_total",
    "Stage 1b decisions per page.",
    ["verdict"],
)


class BypassVerdict(StrEnum):
    """Why the bypass was taken or refused.

    Every non-``BYPASS`` value means the same operational thing — fall through
    to the raster path — but they are distinguished because the *rates* differ
    in meaning: ``NO_TEXT_LAYER`` is a scan and expected, ``JUNK_TEXT_LAYER``
    rising is a font-embedding regression somewhere upstream.
    """

    BYPASS = "bypass"
    NO_TEXT_LAYER = "no_text_layer"
    SPARSE_TEXT_LAYER = "sparse_text_layer"
    JUNK_TEXT_LAYER = "junk_text_layer"
    SCRIPT_MISMATCH = "script_mismatch"


class BypassThresholds(BaseModel):
    """Knobs of the usable-layer test and the junk guard.

    Defaults are deliberately conservative in one direction: a *false* junk
    verdict costs one OCR pass, a *missed* one poisons every downstream stage
    on that page.  When in doubt, rasterise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Minimum characters on a page for the layer to be worth using.
    min_chars: int = Field(default=48, ge=1)
    #: Minimum text rectangles (a proxy for "more than one stray label").
    min_rects: int = Field(default=2, ge=1)
    #: Share of control / format / private-use / surrogate / unassigned chars
    #: above which the layer is junk.
    max_other_share: float = Field(default=0.08, ge=0.0, le=1.0)
    #: Share of U+FFFD above which the layer is junk.
    max_replacement_share: float = Field(default=0.02, ge=0.0, le=1.0)
    #: Minimum share of the dominant script among letters.
    min_dominant_script_share: float = Field(default=0.60, ge=0.0, le=1.0)
    #: Minimum share of letters among non-space characters. Below this the page
    #: is punctuation soup — but a table of amounts is legitimately digit-heavy,
    #: so digits count toward the floor via :attr:`min_letter_or_digit_share`.
    min_letter_share: float = Field(default=0.15, ge=0.0, le=1.0)
    min_letter_or_digit_share: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Share of the single most frequent non-space character above which the
    #: layer is one glyph repeated.
    max_single_char_share: float = Field(default=0.60, ge=0.0, le=1.0)


class TextLayerStats(BaseModel):
    """Character-class and script distribution of one page's text layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    char_count: int = Field(ge=0)
    rect_count: int = Field(ge=0)
    letters: int = Field(ge=0)
    digits: int = Field(ge=0)
    punctuation: int = Field(ge=0)
    whitespace: int = Field(ge=0)
    #: Control, format, private-use, surrogate and unassigned code points.
    other: int = Field(ge=0)
    replacement: int = Field(ge=0)
    #: ISO 15924 of the most common script among letters; ``None`` when the
    #: layer has no letters at all.
    dominant_script: str | None = None
    #: Share of letters in :attr:`dominant_script`, in [0, 1].
    dominant_script_share: float = Field(default=0.0, ge=0.0, le=1.0)
    #: ISO 15924 -> share, over letters only.
    script_distribution: dict[str, float] = Field(default_factory=dict)
    #: Share of the single most frequent non-space character.
    max_single_char_share: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def non_space(self) -> int:
        """Characters that are not whitespace."""
        return self.char_count - self.whitespace

    @property
    def other_share(self) -> float:
        return self.other / self.non_space if self.non_space else 0.0

    @property
    def replacement_share(self) -> float:
        return self.replacement / self.non_space if self.non_space else 0.0

    @property
    def letter_share(self) -> float:
        return self.letters / self.non_space if self.non_space else 0.0

    @property
    def letter_or_digit_share(self) -> float:
        return (
            (self.letters + self.digits) / self.non_space if self.non_space else 0.0
        )


@dataclass(frozen=True, slots=True)
class BypassResult:
    """One page's bypass decision.

    ``page`` is populated **only** when :attr:`verdict` is
    :attr:`BypassVerdict.BYPASS`.  There is no half-usable layer: either the
    lines are trustworthy enough to skip recognition or they are discarded, and
    handing a caller a partially-junk page to "use if it likes" is how the junk
    reaches production anyway.
    """

    page_no: int
    verdict: BypassVerdict
    stats: TextLayerStats
    page: OCRPage | None = None
    #: Human-readable elaboration of a refusal. Never parsed.
    detail: str | None = None

    @property
    def used(self) -> bool:
        return self.verdict is BypassVerdict.BYPASS


# --------------------------------------------------------------------------- #
# Character classification
# --------------------------------------------------------------------------- #


def script_of(char: str) -> str:
    """ISO 15924 script of a single character.

    Returns :data:`COMMON_SCRIPT` for anything that carries no script (digits,
    punctuation, spaces, symbols), :data:`INHERITED_SCRIPT` for a generic
    combining mark, and :data:`UNKNOWN_SCRIPT` for a script-bearing character in
    no range this module knows.  Deliberately a table rather than a dependency:
    the question asked here is "do these characters agree with each other",
    which needs coverage of the *wrong* answers, not authoritative Unicode data.

    **Combining marks count.**  ``str.isalpha()`` is false for every Devanagari
    matra, virama and anusvara — they are Unicode categories ``Mn``/``Mc``, not
    ``Lo`` — and roughly 40% of running Marathi or Hindi text is made of them.
    Treating them as scriptless made a perfectly good Devanagari text layer
    measure as 41% unassigned characters, which the junk guard then refused; see
    ``docs/decisions-wse1.md``.
    """
    if not char:
        return COMMON_SCRIPT
    if unicodedata.category(char[0])[0] not in ("L", "M"):
        return COMMON_SCRIPT
    code = ord(char[0])
    for first, last in _INHERITED_MARK_RANGES:
        if first <= code <= last:
            return INHERITED_SCRIPT
    index = bisect_right(_RANGE_STARTS, code) - 1
    if index >= 0:
        first, last, script = _SCRIPT_RANGES[index]
        if first <= code <= last:
            return script
    return UNKNOWN_SCRIPT


def character_profile(text: str, *, rect_count: int = 0) -> TextLayerStats:
    """Character-class and script distribution of ``text``.

    Pure, cheap and independent of PDFium so the junk guard is testable from a
    string.  Counts use :func:`unicodedata.category`, so an unassigned code
    point (``Cn``) counts as ``other`` rather than silently as a letter.

    Combining marks (``Mn``/``Mc``/``Me``) count as **letters**: a Devanagari
    matra is part of the grapheme it modifies, not a stray control character.
    Marks that inherit their script abstain from the script vote — see
    :func:`script_of` — so the distribution is over *attributed* characters
    while :attr:`TextLayerStats.letters` is over all of them.
    """
    letters = digits = punctuation = whitespace = other = replacement = 0
    script_counts: _Counter[str] = _Counter()
    char_counts: _Counter[str] = _Counter()

    for char in text:
        category = unicodedata.category(char)
        if char == "�":
            replacement += 1
            other += 1
        elif category[0] in ("L", "M"):
            letters += 1
            script = script_of(char)
            if script != INHERITED_SCRIPT:
                script_counts[script] += 1
        elif category[0] == "N":
            digits += 1
        elif category[0] in ("P", "S"):
            punctuation += 1
        elif category[0] == "Z" or char in "\t\n\r\v\f":
            whitespace += 1
        else:
            other += 1
        if not char.isspace():
            char_counts[char] += 1

    non_space = sum(char_counts.values())
    distribution: dict[str, float] = {}
    dominant: str | None = None
    dominant_share = 0.0
    attributed = sum(script_counts.values())
    if attributed:
        distribution = {
            script: count / attributed for script, count in script_counts.items()
        }
        dominant, dominant_count = script_counts.most_common(1)[0]
        dominant_share = dominant_count / attributed
    max_single = (
        char_counts.most_common(1)[0][1] / non_space if non_space else 0.0
    )
    return TextLayerStats(
        char_count=len(text),
        rect_count=rect_count,
        letters=letters,
        digits=digits,
        punctuation=punctuation,
        whitespace=whitespace,
        other=other,
        replacement=replacement,
        dominant_script=dominant,
        dominant_script_share=dominant_share,
        script_distribution=distribution,
        max_single_char_share=max_single,
    )


def is_junk_text_layer(
    stats: TextLayerStats,
    thresholds: BypassThresholds | None = None,
    *,
    expected_script: str | None = None,
) -> str | None:
    """Return the reason the layer is junk, or ``None`` when it is usable.

    A string rather than a bool so the refusal can be logged and counted without
    the caller re-deriving which rule fired.

    ``expected_script`` is the script Stage 2 detected **from the pixels**, when
    it has already run.  Passing it turns the guard from "are these characters
    self-consistent" into "do these characters describe the page we can see",
    which is strictly stronger — but it is optional because Stage 1b is designed
    to run before any page is rendered.
    """
    limits = thresholds if thresholds is not None else BypassThresholds()
    if stats.non_space == 0:
        return "no non-space characters"
    if stats.replacement_share > limits.max_replacement_share:
        return (
            f"{stats.replacement_share:.0%} replacement characters "
            f"(> {limits.max_replacement_share:.0%})"
        )
    if stats.other_share > limits.max_other_share:
        return (
            f"{stats.other_share:.0%} control/private-use/unassigned characters "
            f"(> {limits.max_other_share:.0%})"
        )
    if stats.max_single_char_share > limits.max_single_char_share:
        return (
            f"one character is {stats.max_single_char_share:.0%} of the layer "
            f"(> {limits.max_single_char_share:.0%})"
        )
    if stats.letter_or_digit_share < limits.min_letter_or_digit_share:
        return (
            f"only {stats.letter_or_digit_share:.0%} letters or digits "
            f"(< {limits.min_letter_or_digit_share:.0%})"
        )
    if stats.letter_share < limits.min_letter_share:
        return (
            f"only {stats.letter_share:.0%} letters "
            f"(< {limits.min_letter_share:.0%})"
        )
    if stats.dominant_script is None:
        return "no letters to attribute to a script"
    if stats.dominant_script == UNKNOWN_SCRIPT:
        return "the dominant script of the letters is unattributable"
    if stats.dominant_script_share < limits.min_dominant_script_share:
        return (
            f"letters are scattered across scripts; dominant "
            f"{stats.dominant_script} is only "
            f"{stats.dominant_script_share:.0%} "
            f"(< {limits.min_dominant_script_share:.0%})"
        )
    if expected_script is not None and stats.dominant_script != expected_script:
        return (
            f"text layer is {stats.dominant_script} but the page was detected "
            f"as {expected_script}"
        )
    return None


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def text_rect_to_bbox(
    rect: tuple[float, float, float, float],
    *,
    page_width_pt: float,
    page_height_pt: float,
    rotation: int,
    dpi: float = RASTER_DPI,
) -> list[float]:
    """Map a PDFium text rectangle into raster pixels.

    ``rect`` is ``(left, bottom, right, top)`` in **unrotated** page points with
    the origin bottom-left — PDFium reports text geometry in page space
    regardless of ``/Rotate``, while ``PdfPage.get_size()`` and the rendered
    bitmap are already rotated.  Reconciling the two is this function's whole
    job, and getting it wrong is invisible until a reviewer opens an evidence
    crop of the wrong corner of the page.

    ``page_width_pt`` / ``page_height_pt`` are the **unrotated** MediaBox
    dimensions.  Returns the canonical 8-float bbox, clockwise from top-left.

    Raises:
        ValueError: on a rotation that is not a multiple of 90.
    """
    left, bottom, right, top = rect
    scale = dpi / PDF_POINTS_PER_INCH
    normalised = rotation % 360
    if normalised == 0:
        x0, y0 = left, page_height_pt - top
        x1, y1 = right, page_height_pt - bottom
    elif normalised == 90:
        x0, y0 = bottom, left
        x1, y1 = top, right
    elif normalised == 180:
        x0, y0 = page_width_pt - right, bottom
        x1, y1 = page_width_pt - left, top
    elif normalised == 270:
        x0, y0 = page_height_pt - top, page_width_pt - right
        x1, y1 = page_height_pt - bottom, page_width_pt - left
    else:
        raise ValueError(
            f"page rotation must be a multiple of 90, got {rotation}"
        )
    return bbox_from_xyxy(
        min(x0, x1) * scale,
        min(y0, y1) * scale,
        max(x0, x1) * scale,
        max(y0, y1) * scale,
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _unrotated_size(page: pdfium.PdfPage) -> tuple[float, float]:
    """MediaBox dimensions *before* ``/Rotate`` is applied."""
    width, height = page.get_size()
    if page.get_rotation() % 180 == 90:
        return height, width
    return width, height


def extract_text_lines(
    page: pdfium.PdfPage, *, dpi: float = RASTER_DPI
) -> tuple[list[OCRLine], str]:
    """Lift a page's text layer into :class:`~app.contracts.ocr.OCRLine` values.

    Returns the lines and the concatenated raw text (used for the junk guard,
    which must see the layer exactly as PDFium decoded it — including the
    control characters that make it junk).

    Field choices, each deliberate:

    * ``confidence=1.0`` — this is not a recognition estimate, it is the text
      the document author wrote.  Nothing may escalate on it anyway (PLAN.md
      invariant 3), so a fabricated sub-1.0 value would only be misleading.
    * ``is_handwritten=False`` — a glyph in a content stream was typed.
    * ``language=None`` — Stage 2b sets it per line, exactly as on the OCR path.
    """
    width_pt, height_pt = _unrotated_size(page)
    rotation = page.get_rotation()
    text_page = page.get_textpage()
    try:
        rect_count = text_page.count_rects()
        lines: list[OCRLine] = []
        chunks: list[str] = []
        for index in range(rect_count):
            rect = text_page.get_rect(index)
            text = text_page.get_text_bounded(*rect)
            chunks.append(text)
            stripped = text.strip()
            if not stripped:
                continue
            bbox = text_rect_to_bbox(
                rect,
                page_width_pt=width_pt,
                page_height_pt=height_pt,
                rotation=rotation,
                dpi=dpi,
            )
            lines.append(
                OCRLine(
                    text=stripped,
                    bbox=bbox,
                    confidence=1.0,
                    script=_line_script(stripped),
                    language=None,
                    is_handwritten=False,
                    line_index=len(lines),
                )
            )
        return lines, "\n".join(chunks)
    finally:
        text_page.close()


def _line_script(text: str) -> str:
    """Dominant script of one line, falling back to :data:`COMMON_SCRIPT`.

    Uses the same attribution as :func:`character_profile`: combining marks
    vote with the script of their block, and the ones that inherit their script
    abstain rather than dragging the line to ``Zinh``.
    """
    counts: _Counter[str] = _Counter()
    for char in text:
        script = script_of(char)
        if script not in (COMMON_SCRIPT, INHERITED_SCRIPT):
            counts[script] += 1
    if not counts:
        return COMMON_SCRIPT
    return counts.most_common(1)[0][0]


def detect_bypass(
    page: pdfium.PdfPage,
    page_no: int,
    *,
    dpi: float = RASTER_DPI,
    thresholds: BypassThresholds | None = None,
    expected_script: str | None = None,
) -> BypassResult:
    """Decide whether one page may skip recognition, and build it if so."""
    limits = thresholds if thresholds is not None else BypassThresholds()
    lines, raw_text = extract_text_lines(page, dpi=dpi)
    stats = character_profile(raw_text, rect_count=len(lines))

    def _refuse(verdict: BypassVerdict, detail: str) -> BypassResult:
        _BYPASS_DECISIONS.labels(verdict=verdict.value).inc()
        _LOGGER.debug("page %d: bypass refused (%s): %s", page_no, verdict, detail)
        return BypassResult(
            page_no=page_no, verdict=verdict, stats=stats, detail=detail
        )

    if stats.char_count == 0 or not lines:
        return _refuse(BypassVerdict.NO_TEXT_LAYER, "the page carries no text layer")
    if stats.char_count < limits.min_chars or len(lines) < limits.min_rects:
        return _refuse(
            BypassVerdict.SPARSE_TEXT_LAYER,
            f"{stats.char_count} characters in {len(lines)} rect(s); floor is "
            f"{limits.min_chars} characters in {limits.min_rects} rect(s)",
        )

    junk_reason = is_junk_text_layer(
        stats, limits, expected_script=expected_script
    )
    if junk_reason is not None:
        verdict = (
            BypassVerdict.SCRIPT_MISMATCH
            if expected_script is not None
            and stats.dominant_script is not None
            and stats.dominant_script != expected_script
            else BypassVerdict.JUNK_TEXT_LAYER
        )
        return _refuse(verdict, junk_reason)

    width_pt, height_pt = _unrotated_size(page)
    scale = dpi / PDF_POINTS_PER_INCH
    if page.get_rotation() % 180 == 90:
        width_px = int(round(height_pt * scale))
        height_px = int(round(width_pt * scale))
    else:
        width_px = int(round(width_pt * scale))
        height_px = int(round(height_pt * scale))

    ocr_page = OCRPage(
        page_no=page_no,
        width_px=max(1, width_px),
        height_px=max(1, height_px),
        lines=lines,
        tier=OCRTier.BYPASS,
        dpi_estimate=dpi,
    )
    _BYPASS_DECISIONS.labels(verdict=BypassVerdict.BYPASS.value).inc()
    return BypassResult(
        page_no=page_no,
        verdict=BypassVerdict.BYPASS,
        stats=stats,
        page=ocr_page,
    )


def bypass_document(
    source: bytes | pdfium.PdfDocument,
    *,
    dpi: float = RASTER_DPI,
    thresholds: BypassThresholds | None = None,
    expected_scripts: Sequence[str | None] | None = None,
) -> list[BypassResult]:
    """Run :func:`detect_bypass` over every page of a PDF.

    ``source`` may be the original attachment bytes (Stage 1 keeps them on
    :attr:`~app.stages.s1_ingest.IngestResult.data`) or an already-open
    document.  Bytes go through :func:`~app.stages.s1_ingest.open_pdf`, so the
    encrypted and malformed cases raise the same typed errors here as they do in
    Stage 1 rather than leaking a ``PdfiumError``.

    Raises:
        EncryptedPDFError, MalformedPDFError: when ``source`` is bytes that do
            not open.
    """
    context: AbstractContextManager[pdfium.PdfDocument]
    if isinstance(source, bytes | bytearray):
        context = open_pdf(bytes(source))
    else:
        context = nullcontext(source)

    results: list[BypassResult] = []
    with context as document:
        for index in range(len(document)):
            expected = (
                expected_scripts[index]
                if expected_scripts is not None and index < len(expected_scripts)
                else None
            )
            page = document[index]
            try:
                results.append(
                    detect_bypass(
                        page,
                        index + 1,
                        dpi=dpi,
                        thresholds=thresholds,
                        expected_script=expected,
                    )
                )
            finally:
                page.close()
    return results
