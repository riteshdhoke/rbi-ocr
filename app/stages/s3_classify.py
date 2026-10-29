"""Stage 3 — page classification: structural features + logistic late fusion.

BUILD-BRIEF M4 specifies "MobileNetV3-Small logits + structural features (line
count, ink density, aspect, table-line presence, colour histogram) →
logistic-regression late fusion".  This module implements the whole of that
**except** the CNN, which is an optional ONNX artifact
(``models/classifier/page_classifier.onnx``):

* when the artifact is present the provider runs it and fuses (that is
  ``provider.classify_page``, already written in WS-C);
* when it is absent — its state on every machine today — the provider raises a
  typed :class:`~app.providers.base.ModelArtifactMissing` and this stage falls
  back to the **structural path**, which is fitted, committed and works now.

Both paths run through the same :func:`app.providers.local.late_fusion` and the
same ``fusion.json`` contract, so adding the CNN later is a retrain plus a file
drop, not a code change.

The committed weights
---------------------
``models/classifier/fusion.json`` (with an identical copy at
``docs/calibration/fusion.json`` so an air-gapped node with an empty ``models/``
still classifies) is fitted by ``tools/train_page_classifier.py`` over corpus
**variants 0-2 only** — variants 3 and 4 are the locked test split and this
stage must never have seen them.  The CNN coefficient slots in that file are
**exactly zero**: the fit was performed with uniform CNN scores, so the
structural decision is identical whether a CNN runs or not, and a real CNN
cannot silently take credit for a structural result.  Numbers are in
``docs/decisions-wse2.md``.

This module does NOT handle:
  * document-level class.  A document's class is a fold over its pages and is
    assembled in ``assemble.py``; this stage sees one page;
  * training.  ``tools/train_page_classifier.py`` owns that, and imports its
    feature extraction from here so the two cannot drift;
  * OCR.  ``lines`` are optional; without them ``line_count``,
    ``handwriting_ratio`` and ``text_coverage`` are estimated from pixels, and
    that is the configuration the committed weights were fitted in, because
    Stage 3 runs before recognition in the PRD flow;
  * the quality gate's scores, deskew or repair (Stage 1.5, WS-E1);
  * loading the CNN — the provider owns every model file (decision C20).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict

from app.contracts.ocr import (
    OCRLine,
    PageClass,
    PageClassLabel,
    StructuralFeatures,
)
from app.providers.base import ModelArtifactMissing, ModelProvider, PageImage
from app.providers.local import (
    CLASSIFIER_FUSION_FILENAME,
    CLASSIFIER_SUBDIR,
    STRUCTURAL_FEATURE_NAMES,
    FusionWeights,
    late_fusion,
)
from app.settings import PROJECT_ROOT, Settings, get_settings

__all__ = [
    "CALIBRATION_FUSION_PATH",
    "COLOUR_HISTOGRAM_BINS",
    "HANDWRITING_SCORE_THRESHOLD",
    "MIN_HANDWRITING_CROP_PX",
    "TRAINED_LABELS",
    "ClassificationMechanism",
    "ClassificationResult",
    "classify_page",
    "colour_histogram",
    "detect_table_lines",
    "estimate_handwriting_ratio",
    "estimate_skew_degrees",
    "estimate_text_lines",
    "extract_structural_features",
    "flatten_illumination",
    "fusion_search_paths",
    "handwriting_score",
    "handwriting_score_from_mask",
    "ink_mask",
    "load_fusion_weights",
    "reset_fusion_cache",
    "structural_classify",
]

_LOG = logging.getLogger(__name__)

#: Bins per colour channel; the feature is ``3 * COLOUR_HISTOGRAM_BINS`` long.
COLOUR_HISTOGRAM_BINS: Final[int] = 4

#: The committed fallback copy of the fusion coefficients.  An air-gapped node
#: with an empty ``models/`` must still be able to classify a page, so the
#: artifact ships inside the repository as well as in the model bundle.
CALIBRATION_FUSION_PATH: Final[Path] = (
    PROJECT_ROOT / "docs" / "calibration" / CLASSIFIER_FUSION_FILENAME
)

#: Label order the committed weights were fitted in.  ``unknown`` is absent by
#: design: it is a *refusal*, produced by :func:`classify_page` when confidence
#: is below the floor, never a class the model can be trained to predict.
TRAINED_LABELS: Final[tuple[PageClassLabel, ...]] = (
    PageClassLabel.COMPLAINT_FORM,
    PageClassLabel.COMPLAINT_LETTER,
    PageClassLabel.SUPPORTING_DOC,
    PageClassLabel.PHOTO_EVIDENCE,
)

#: Below this fused confidence the page is ``unknown`` rather than a coin flip.
MIN_CLASS_CONFIDENCE: Final[float] = 0.35

#: A morphological run must span this share of the page to count as a rule.
TABLE_LINE_MIN_SPAN: Final[float] = 0.30

#: Background-estimation kernel for :func:`flatten_illumination`.  Must exceed
#: the tallest glyph and stay well below the scale of a shadow gradient; 31 px
#: is both at the 120-200 DPI the corpus renders at.
BACKGROUND_KERNEL_PX: Final[int] = 31

#: Skew below this is not worth a warpAffine.
MIN_DESKEW_DEGREES: Final[float] = 0.4

#: Above this the estimate is more likely a mis-detection than a skew — a page
#: rotated 40 degrees is a rotation problem for Stage 1.5, not a feature.
MAX_DESKEW_DEGREES: Final[float] = 25.0

#: A component taller than this share of the page is not a text line.
MAX_LINE_HEIGHT_SHARE: Final[float] = 0.06

#: Above this score a line crop is called handwritten.  See
#: :func:`handwriting_score` for the calibration and its honest limits.
HANDWRITING_SCORE_THRESHOLD: Final[float] = 0.55

#: Crops smaller than this in either dimension are not scored: a 6-pixel-tall
#: strip has no stroke statistics, and scoring it would return noise wearing the
#: confidence of a measurement.
MIN_HANDWRITING_CROP_PX: Final[int] = 12

#: Fewer glyph components than this and the crop is not scored at all.
MIN_HANDWRITING_GLYPHS: Final[int] = 4

_CLASSIFICATIONS = Counter(
    "stage_s3_classifications_total",
    "Stage-3 page classifications, by mechanism and predicted label.",
    ["mechanism", "label"],
)
_CLASSIFIER_FALLBACKS = Counter(
    "stage_s3_cnn_fallbacks_total",
    "Times the page-classifier CNN was unavailable and the structural path "
    "was used instead.",
)
_FEATURE_LATENCY = Histogram(
    "stage_s3_features_seconds", "Wall time of one structural feature pass."
)
_CLASSIFY_LATENCY = Histogram(
    "stage_s3_classify_seconds", "Wall time of one Stage-3 classification."
)

_CV2_LOCK = threading.Lock()
_CV2_MODULE: Any = None
_WEIGHT_CACHE: dict[str, FusionWeights] = {}
_WEIGHT_CACHE_LOCK = threading.Lock()


def _cv2() -> Any:
    """Import OpenCV on first use, with its thread pool already capped.

    Lazy for the same reason the provider is (decision C20): ``import cv2``
    costs a few hundred milliseconds and nothing that only fuses coefficients
    should pay it.  :func:`app.stages.configure_cv2_threads` is the one place
    PLAN.md invariant 5's OpenCV gap is closed, and it is idempotent.
    """
    global _CV2_MODULE
    with _CV2_LOCK:
        if _CV2_MODULE is None:
            from app.stages import configure_cv2_threads

            configure_cv2_threads()
            import cv2

            _CV2_MODULE = cv2
        return _CV2_MODULE


class ClassificationMechanism(StrEnum):
    """Which path produced a :class:`ClassificationResult`."""

    #: MobileNetV3-Small logits fused with structural features (provider).
    CNN_FUSION = "cnn_fusion"
    #: Structural features only, uniform CNN prior — today's path.
    STRUCTURAL_FUSION = "structural_fusion"


class ClassificationResult(BaseModel):
    """Stage-3 output: the class, how it was reached, and the evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_class: PageClass
    mechanism: ClassificationMechanism
    features: StructuralFeatures
    #: Set when the CNN was asked for and its artifact was absent.
    artifact_fallback_reason: str | None = None
    #: True when the fused confidence fell below :data:`MIN_CLASS_CONFIDENCE`
    #: and the label was demoted to ``unknown``.
    below_confidence_floor: bool = False

    @property
    def label(self) -> PageClassLabel:
        """Shorthand for ``page_class.label``."""
        return self.page_class.label

    @property
    def confidence(self) -> float:
        """Shorthand for ``page_class.confidence``."""
        return self.page_class.confidence


# --------------------------------------------------------------------------- #
# Structural features
# --------------------------------------------------------------------------- #


def _to_bgr(page: PageImage) -> Any:
    """View any accepted page image as a 3-channel BGR ``uint8`` array."""
    cv2 = _cv2()
    array = np.asarray(page)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError(f"page image must be 2-D or 3-D, got {array.shape!r}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(
            f"page image must have 1, 3 or 4 channels, got {array.shape[2]}"
        )
    return np.ascontiguousarray(array.astype(np.uint8))


def flatten_illumination(page: PageImage) -> Any:
    """Grayscale page with its background lighting divided out.

    The corpus's ``scan``, ``heavy`` and ``photo`` profiles add shadow, a fold,
    a stamp and an exposure shift by design.  A global Otsu on the raw
    grayscale reads a shadowed half-page as solid ink: measured on the committed
    corpus, ``complaint_letter_en`` at profile ``photo`` came out at
    ``ink_density=0.42`` against ``0.007`` for the same page at ``clean``.  A
    "structural" feature that swings sixty-fold with the lighting is measuring
    the lighting.

    The fix is the standard division recipe: estimate the background by closing
    the page with a kernel wider than any glyph, then divide.  Ink survives
    (it is darker than its own neighbourhood); shadows and exposure do not (they
    *are* their own neighbourhood).  A 3-pixel median blur afterwards removes
    the salt-and-pepper the noise profiles add without touching stroke shape.
    """
    cv2 = _cv2()
    grey = cv2.cvtColor(_to_bgr(page), cv2.COLOR_BGR2GRAY)
    height, width = grey.shape[:2]
    span = max(3, min(BACKGROUND_KERNEL_PX, min(height, width) // 2 * 2 + 1))
    if span % 2 == 0:
        span += 1
    background = cv2.morphologyEx(
        grey,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)),
    )
    flattened = cv2.divide(grey, background, scale=255)
    if min(height, width) >= 3:
        flattened = cv2.medianBlur(flattened, 3)
    return flattened


def ink_mask(page: PageImage, *, flatten: bool = True) -> Any:
    """Binary ink mask (1 = ink) via Otsu on the illumination-flattened page.

    Args:
        flatten: run :func:`flatten_illumination` first.  Leave it on for
            anything page-sized.  Turning it off is for a caller that has
            already flattened, not for a caller that wants raw Otsu.
    """
    cv2 = _cv2()
    source = (
        flatten_illumination(page)
        if flatten
        else cv2.cvtColor(_to_bgr(page), cv2.COLOR_BGR2GRAY)
    )
    _, binary = cv2.threshold(
        source, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask = binary.astype(np.uint8)
    if min(mask.shape[:2]) >= 8:
        # A single speck is noise, not ink. 2x2 opening costs nothing and stops
        # the noise profiles from inventing thousands of one-pixel components.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
        )
    return mask


def estimate_skew_degrees(mask: Any) -> float:
    """Dominant text-line angle in degrees, positive counter-clockwise.

    Stage 1.5 deskews a page before Stage 3 sees it, but this stage must not
    *depend* on that: the trainer reads raw corpus images, and a rule detector
    built on a strictly horizontal structuring element loses every rule at
    1.5 degrees of skew — measured on ``complaint_form_en``, 25 detected rules
    at ``clean`` fell to 0 at ``light``.  Normalising the working copy makes the
    feature a property of the layout instead of a property of the scanner.

    Estimated from the text itself: ink closed into line strips, then the median
    long-edge angle of the largest strips' minimum-area rectangles.  Returns
    ``0.0`` when there are too few strips to have an opinion.
    """
    cv2 = _cv2()
    height, width = mask.shape[:2]
    if height < 32 or width < 32:
        return 0.0
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(5, int(width * 0.03)), 1)
        ),
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    angles: list[float] = []
    for contour in contours:
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.15 or box_height > height * 0.2:
            continue
        corners = cv2.boxPoints(cv2.minAreaRect(contour))
        angles.append(_long_edge_angle(corners))
    if len(angles) < 3:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=np.float64)))


def _long_edge_angle(corners: Any) -> float:
    """Angle in degrees of the longest edge of a 4-corner box, in [-45, 45]."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    best_length = -1.0
    best_angle = 0.0
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        delta = second - first
        length = float(np.hypot(delta[0], delta[1]))
        if length > best_length:
            best_length = length
            best_angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    while best_angle > 45.0:
        best_angle -= 90.0
    while best_angle < -45.0:
        best_angle += 90.0
    return best_angle


def _deskewed_mask(page: PageImage) -> Any:
    """Ink mask rotated so its text lines are horizontal.

    Rotated **in place** (same canvas, background padding), so ``aspect_ratio``
    keeps describing the page rather than the text block's bounding box.
    """
    cv2 = _cv2()
    mask = ink_mask(page)
    angle = estimate_skew_degrees(mask)
    if abs(angle) < MIN_DESKEW_DEGREES or abs(angle) > MAX_DESKEW_DEGREES:
        return mask
    height, width = mask.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated.astype(np.uint8)


def detect_table_lines(page: PageImage) -> tuple[bool, int]:
    """Count long horizontal/vertical rules via morphology (BUILD-BRIEF M4).

    Opens the ink mask with a long 1-D structuring element in each direction —
    the standard OpenCV table-extraction recipe — and keeps components spanning
    at least :data:`TABLE_LINE_MIN_SPAN` of the page in that direction.  Text
    strokes never survive a 30%-of-page kernel; printed rules and form boxes do.

    ``has_table_lines`` is ``count > 0`` rather than "enough rules to be a
    table": :class:`~app.contracts.ocr.StructuralFeatures` requires a positive
    count to imply the boolean, and the fusion sees both the flag and
    ``log1p(count)``, so *how many* rules make a table is a coefficient the
    model fits rather than a constant this function asserts.

    Returns:
        ``(has_table_lines, table_line_count)``.
    """
    return _table_lines_from_mask(_deskewed_mask(page))


def _table_lines_from_mask(mask: Any) -> tuple[bool, int]:
    """:func:`detect_table_lines` over an already-prepared, deskewed mask."""
    cv2 = _cv2()
    height, width = mask.shape[:2]
    if height < 8 or width < 8:
        return False, 0

    horizontal_len = max(8, int(width * TABLE_LINE_MIN_SPAN))
    vertical_len = max(8, int(height * TABLE_LINE_MIN_SPAN))
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1)),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len)),
    )
    count = _count_runs(horizontal, axis="horizontal") + _count_runs(
        vertical, axis="vertical"
    )
    return count > 0, count


def _count_runs(mask: Any, *, axis: str) -> int:
    """Connected components in an opened mask that still span the page."""
    cv2 = _cv2()
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    minimum = (
        width * TABLE_LINE_MIN_SPAN
        if axis == "horizontal"
        else height * TABLE_LINE_MIN_SPAN
    )
    count = 0
    for contour in contours:
        _, _, box_width, box_height = cv2.boundingRect(contour)
        extent = box_width if axis == "horizontal" else box_height
        if extent >= minimum:
            count += 1
    return count


def colour_histogram(page: PageImage, *, bins: int = COLOUR_HISTOGRAM_BINS) -> list[float]:
    """Normalised per-channel BGR histogram, ``3 * bins`` values summing to 1.

    Deliberately coarse.  A fine histogram encodes the JPEG quality of a
    degradation profile; four bins per channel encode "is this a scanned white
    page or a photographed one", which is the signal the class actually depends
    on.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    array = _to_bgr(page).astype(np.uint16)
    quantised = np.minimum(array * bins // 256, bins - 1)
    counts: list[float] = []
    for channel in range(3):
        histogram = np.bincount(
            quantised[:, :, channel].reshape(-1), minlength=bins
        )[:bins]
        counts.extend(float(value) for value in histogram)
    total = sum(counts)
    if total <= 0:  # pragma: no cover - a zero-pixel image cannot reach here
        return [0.0] * (3 * bins)
    normalised = [value / total for value in counts]
    largest = max(range(len(normalised)), key=lambda i: normalised[i])
    normalised[largest] += 1.0 - sum(normalised)
    return [min(1.0, max(0.0, value)) for value in normalised]


def estimate_text_lines(page: PageImage) -> tuple[int, float]:
    """Estimate ``(line_count, text_coverage)`` from pixels alone.

    Stage 3 runs **before** recognition in the PRD flow, so the classifier
    cannot assume it has ``OCRLine`` objects.  Ink is closed horizontally with a
    kernel about one character-width wide, which merges glyphs into word blobs
    and words into line strips without joining adjacent lines vertically; the
    surviving components that are wider than they are tall and taller than a
    speck are counted as text lines.

    Rules detected by :func:`detect_table_lines` are removed first, so a form's
    box borders are not counted as forty text lines.
    """
    return _text_lines_from_mask(_deskewed_mask(page))


def _text_lines_from_mask(mask: Any) -> tuple[int, float]:
    """:func:`estimate_text_lines` over an already-prepared, deskewed mask."""
    cv2 = _cv2()
    height, width = mask.shape[:2]
    if height < 8 or width < 8:
        return 0, 0.0

    rules = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(8, int(width * TABLE_LINE_MIN_SPAN)), 1)
        ),
    ) | cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(8, int(height * TABLE_LINE_MIN_SPAN)))
        ),
    )
    text = cv2.subtract(mask, rules)

    kernel_width = max(3, int(width * 0.015))
    closed = cv2.morphologyEx(
        text,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    boxes = _line_boxes(closed, height=height, width=width)
    covered = sum(float(box[2] * box[3]) for box in boxes)
    return len(boxes), min(1.0, covered / float(height * width))


def _line_boxes(
    closed: Any, *, height: int, width: int
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the components that look like text lines.

    A text line is wider than a speck, narrower than the page, and no taller
    than :data:`MAX_LINE_HEIGHT_SHARE` of the page.  The height cap is what
    stops a shadow band or a stamp from being counted as one enormous line and
    dragging ``text_coverage`` from 0.06 to 0.38, which is what it did before
    the cap was tightened.
    """
    cv2 = _cv2()
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    min_height = max(2, int(height * 0.004))
    min_width = max(4, int(width * 0.02))
    max_height = max(min_height + 1, int(height * MAX_LINE_HEIGHT_SHARE))
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
        if box_height < min_height or box_width < min_width:
            continue
        if box_height > max_height:
            continue
        boxes.append((box_x, box_y, box_width, box_height))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


# --------------------------------------------------------------------------- #
# Handwriting — the pixel statistic decision C8 handed to WS-E
# --------------------------------------------------------------------------- #


def handwriting_score(crop: PageImage) -> float:
    """Structural handwriting score for one line-sized crop, in ``[0, 1]``.

    **The heuristic, stated in full.**  Print is *regular*: a machine-set line
    puts every glyph of a given class on the same baseline, at the same height,
    with the same stroke weight.  Writing does none of those.  Three scale-free
    statistics measure exactly that on the binarised crop, over its connected
    components (glyphs):

    1. **Baseline jitter** (weight 0.50) — standard deviation of the *bottom*
       edge of the comparably-sized glyphs, over the crop height.  Restricting
       to glyphs within +-30% of the median height is what makes this work: an
       unfiltered bottom-edge statistic is dominated by descenders, and measured
       on this corpus it scores AUC 0.55 unfiltered against 0.91 filtered.
    2. **Glyph-height irregularity** (weight 0.30) — coefficient of variation of
       component heights (AUC 0.81).  A font has three or four glyph heights; a
       hand has a continuum.
    3. **Stroke-width variation** (weight 0.20) — coefficient of variation of
       the distance transform's ridge values, i.e. of half the local stroke
       width.  A pen varies pressure and angle; a font does not.  It carries
       little signal *on this corpus* (AUC 0.58) because a handwriting **font**
       has a machine-uniform stroke; it is kept, at low weight, because it is
       the term that will carry the signal on real pen-on-paper input.

    Each is squashed onto ``[0, 1]`` by a fixed linear map fitted to the
    corpus's own percentiles, then averaged;
    :data:`HANDWRITING_SCORE_THRESHOLD` cuts it.  There is no learned model and
    no artifact — decision C8 refused to invent one, and this is a
    *measurement* of three named properties rather than a guess dressed up as a
    classifier.

    **Measured performance, stated plainly** (re-measured in the WS-E2 audit;
    both populations are given because only one of them is the one the shipped
    flag is applied to).  Corpus variants 0-2, non-bounce profiles,
    ground-truth line boxes:

    * **Lines this function can score** (score > 0), n=2969 of which 520 are
      handwritten: **AUC 0.855** — 0.908 at profile ``clean``, 0.740 at
      ``photo``; 0.837 on Latin, 0.780 on Devanagari, where the shirorekha
      joins a whole word into one connected component and the glyph statistics
      lose their glyphs.  At the shipped threshold of 0.55, **precision 0.494,
      recall 0.679**.
    * **Every line on those pages**, n=3675 of which 600 are handwritten:
      **AUC 0.780**, precision 0.494, **recall 0.588**.  The 706 lines
      (19% of them, 80 of them handwritten) that are too small or too sparse to
      measure score ``0.0``, and :func:`app.stages.s4_ocr.annotate_handwriting`
      turns a ``0.0`` into ``is_handwritten=False`` — so the second row, not the
      first, is what the flag delivers.  Quoting only the first would be
      grading the heuristic on the lines it found easy.

    This is a useful *signal* and it is **not** a reliable per-line classifier —
    treat ``is_handwritten`` accordingly, which is why the escalation router
    weights it at half a point and never escalates on it alone.

    **What it does not prove.**  The corpus renders "handwriting" as a
    handwriting **font** plus per-grapheme-cluster jitter (adjudication A-1), so
    what this function mostly detects is that jitter.  That is enough to
    exercise routing and this code path; it is not evidence about real
    handwriting, exactly as TEST-DATA-SPEC sec. 1 warns.  The full breakdown is
    in ``docs/decisions-wse2.md``; the threshold is provisional until it is
    re-cut on ``iiit-indic-hw-words``.

    It lives in Stage 3 rather than Stage 4 because it is a structural pixel
    statistic and because ``handwriting_ratio`` is a *classifier feature*;
    :mod:`app.stages.s4_ocr` imports it and owns the per-line policy.

    Returns:
        ``0.0`` for a crop too small or too empty to measure — never a guess.
    """
    array = np.asarray(crop)
    if array.ndim < 2:
        return 0.0
    height, width = array.shape[:2]
    if height < MIN_HANDWRITING_CROP_PX or width < MIN_HANDWRITING_CROP_PX:
        return 0.0
    return handwriting_score_from_mask(ink_mask(array))


def handwriting_score_from_mask(mask: Any) -> float:
    """:func:`handwriting_score` over an already-binarised crop.

    Separate so the page-level estimator can reuse the single deskewed mask it
    already built instead of re-thresholding twenty-five sub-images.

    Returns ``0.0`` when the crop yields fewer than
    :data:`MIN_HANDWRITING_GLYPHS` glyph components — four ink blobs are not a
    sample, and a score computed from them would be noise carrying the
    authority of a measurement.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        return 0.0
    height, width = binary.shape[:2]
    if height < MIN_HANDWRITING_CROP_PX or width < MIN_HANDWRITING_CROP_PX:
        return 0.0

    glyphs = _glyph_statistics(binary)
    if glyphs is None:
        return 0.0
    baseline_jitter, height_cv = glyphs

    score = (
        0.50 * _squash(baseline_jitter, low=0.018, high=0.055)
        + 0.30 * _squash(height_cv, low=0.13, high=0.38)
        + 0.20 * _squash(_stroke_width_cv(binary), low=0.28, high=0.45)
    )
    return float(min(1.0, max(0.0, score)))


def _glyph_statistics(mask: Any) -> tuple[float, float] | None:
    """``(baseline_jitter, height_cv)`` from the crop's glyph components.

    ``baseline_jitter`` is the std of the bottom edge of glyphs whose height is
    within +-30% of the median, normalised by crop height; restricting to
    comparable glyphs is what removes the descenders that otherwise drown the
    signal.  ``None`` when there are too few components to measure.
    """
    cv2 = _cv2()
    binary = np.asarray(mask, dtype=np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, 8
    )
    heights: list[float] = []
    bottoms: list[float] = []
    for index in range(1, count):
        _x, y, _w, component_height, area = stats[index]
        if component_height < 3 or area < 6:
            continue
        heights.append(float(component_height))
        bottoms.append(float(y + component_height))
    if len(heights) < MIN_HANDWRITING_GLYPHS:
        return None

    height_array = np.asarray(heights, dtype=np.float64)
    bottom_array = np.asarray(bottoms, dtype=np.float64)
    median = float(np.median(height_array))
    comparable = (height_array >= 0.7 * median) & (height_array <= 1.3 * median)
    if int(comparable.sum()) < MIN_HANDWRITING_GLYPHS:
        comparable = np.ones_like(height_array, dtype=bool)
    crop_height = max(1.0, float(binary.shape[0]))
    return (
        float(bottom_array[comparable].std() / crop_height),
        _coefficient_of_variation(height_array),
    )


def _stroke_width_cv(mask: Any) -> float:
    """Coefficient of variation of local stroke width, via distance transform.

    The distance transform's value at a stroke's ridge is half its local width,
    so the CV of the ridge values is a scale-free stroke-weight variation.  More
    robust than horizontal run lengths, which a slanted stroke inflates without
    the stroke being any wider.
    """
    cv2 = _cv2()
    distances = cv2.distanceTransform(
        np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 3
    )
    ridge = distances[distances > 0.9]
    if ridge.size < 20:
        return 0.0
    return _coefficient_of_variation(ridge)


def _coefficient_of_variation(values: Any) -> float:
    """``std / mean``; ``0.0`` for an empty or zero-mean sample."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    mean = float(array.mean())
    if mean <= 1e-9:
        return 0.0
    return float(array.std() / mean)


def _squash(value: float, *, low: float, high: float) -> float:
    """Linear map of ``[low, high]`` onto ``[0, 1]``, clamped outside."""
    if high <= low:
        raise ValueError(f"high ({high}) must exceed low ({low})")
    return min(1.0, max(0.0, (value - low) / (high - low)))


def estimate_handwriting_ratio(
    page: PageImage, *, threshold: float = HANDWRITING_SCORE_THRESHOLD
) -> float:
    """Share of a page's detected text lines that look handwritten.

    The pixel-only counterpart of counting ``OCRLine.is_handwritten``.  Without
    it ``handwriting_ratio`` would be a constant zero everywhere Stage 3 runs
    before Stage 4 — which is everywhere — and the classifier would be fitted
    against a feature that carries no information, while the one thing that
    actually distinguishes a handwritten complaint letter from a printed email
    printout went unmeasured.
    """
    return _handwriting_ratio_from_mask(_deskewed_mask(page), threshold=threshold)


def _handwriting_ratio_from_mask(
    mask: Any, *, threshold: float = HANDWRITING_SCORE_THRESHOLD
) -> float:
    """:func:`estimate_handwriting_ratio` over a prepared, deskewed mask."""
    cv2 = _cv2()
    height, width = mask.shape[:2]
    if height < 8 or width < 8:
        return 0.0
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(3, int(width * 0.015)), 1)
        ),
    )
    boxes = _line_boxes(closed, height=height, width=width)
    if not boxes:
        return 0.0
    scored = 0
    handwritten = 0
    for box_x, box_y, box_width, box_height in boxes:
        crop = mask[box_y : box_y + box_height, box_x : box_x + box_width]
        score = handwriting_score_from_mask(crop)
        if score <= 0.0:
            continue  # too small to measure; not evidence either way
        scored += 1
        handwritten += int(score >= threshold)
    return 0.0 if scored == 0 else handwritten / scored


def extract_structural_features(
    page: PageImage,
    *,
    lines: Sequence[OCRLine] | None = None,
    settings: Settings | None = None,
) -> StructuralFeatures:
    """Compute the layout signals the classifier and the router consume.

    Args:
        page: NumPy ``uint8`` BGR or grayscale array (decision C1).
        lines: recognised lines, when Stage 4 has already run.  Supplying them
            replaces the pixel *estimates* of ``line_count``, ``text_coverage``
            and ``handwriting_ratio`` with measurements.  **The committed fusion
            weights were fitted without them** — see the module docstring — so
            passing lines changes the feature distribution the model sees; do it
            only with weights retrained the same way.

    Raises:
        ValueError: on an empty or non-2-D/3-D image.
    """
    del settings  # accepted for a uniform stage signature; nothing is read yet
    with _FEATURE_LATENCY.time():
        array = _to_bgr(page)
        height, width = array.shape[:2]
        if height < 1 or width < 1:
            raise ValueError(f"page image is empty: shape {array.shape!r}")

        # One prepared mask for every pixel feature: flattening, denoising and
        # deskewing are expensive enough that doing them three times would show
        # up in the per-page budget.
        mask = _deskewed_mask(array)
        has_tables, table_count = _table_lines_from_mask(mask)
        estimated_lines, estimated_coverage = _text_lines_from_mask(mask)

        if lines is not None:
            line_count = len(lines)
            coverage = _coverage_from_lines(lines, width=width, height=height)
            handwriting_ratio = (
                sum(1 for line in lines if line.is_handwritten) / len(lines)
                if lines
                else 0.0
            )
        else:
            line_count = estimated_lines
            coverage = estimated_coverage
            handwriting_ratio = _handwriting_ratio_from_mask(mask)

        return StructuralFeatures(
            line_count=line_count,
            ink_density=float(np.clip(mask.mean(), 0.0, 1.0)),
            aspect_ratio=float(width) / float(height),
            has_table_lines=has_tables,
            table_line_count=table_count,
            colour_histogram=colour_histogram(array),
            handwriting_ratio=float(np.clip(handwriting_ratio, 0.0, 1.0)),
            text_coverage=float(np.clip(coverage, 0.0, 1.0)),
        )


def _coverage_from_lines(
    lines: Sequence[OCRLine], *, width: int, height: int
) -> float:
    """Share of page area inside line bounding boxes (axis-aligned envelope).

    Overlapping boxes are double-counted; the result is clamped to 1.  An exact
    union would cost a mask allocation per page for a feature the fusion barely
    weights, and the corpus's detector does not emit overlapping lines.
    """
    if not lines or width <= 0 or height <= 0:
        return 0.0
    total = 0.0
    for line in lines:
        xs = line.bbox[0::2]
        ys = line.bbox[1::2]
        total += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return min(1.0, total / float(width * height))


# --------------------------------------------------------------------------- #
# Fusion weights
# --------------------------------------------------------------------------- #


def fusion_search_paths(settings: Settings | None = None) -> tuple[Path, ...]:
    """Where :func:`load_fusion_weights` looks, in order.

    ``models/classifier/fusion.json`` first (a deployment may have retrained),
    then the copy committed under ``docs/calibration/`` so a node with an empty
    ``models/`` directory still classifies.
    """
    active = settings if settings is not None else get_settings()
    return (
        active.models_dir / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME,
        CALIBRATION_FUSION_PATH,
    )


def load_fusion_weights(
    settings: Settings | None = None, *, path: Path | None = None
) -> FusionWeights:
    """Load and cache the late-fusion coefficients.

    Cached by resolved path, because a page classifier that re-parses its
    coefficients per page is a measurable share of a 4-second budget.

    Raises:
        ModelArtifactMissing: when no candidate path exists.
        ValueError: when a file exists but is not a valid ``FusionWeights``; a
            corrupt coefficient file must not fall through to the next
            candidate and quietly classify with different weights.
    """
    candidates = (path,) if path is not None else fusion_search_paths(settings)
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        key = str(candidate.resolve())
        with _WEIGHT_CACHE_LOCK:
            cached = _WEIGHT_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            weights = FusionWeights.model_validate_json(
                candidate.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"fusion coefficients at {candidate} are unreadable or "
                f"invalid: {exc}"
            ) from exc
        with _WEIGHT_CACHE_LOCK:
            _WEIGHT_CACHE[key] = weights
        return weights
    raise ModelArtifactMissing(
        artifact="page-classifier late-fusion coefficients (fusion.json)",
        path=candidates[0] if candidates and candidates[0] else Path("fusion.json"),
        how_to_obtain=(
            "Run `python tools/train_page_classifier.py` against the committed "
            "corpus; it writes models/classifier/fusion.json and the "
            "docs/calibration/ copy that ships with the source."
        ),
    )


def reset_fusion_cache() -> None:
    """Drop cached coefficients (tests, and a retrain in a live worker)."""
    with _WEIGHT_CACHE_LOCK:
        _WEIGHT_CACHE.clear()


def uniform_cnn_scores(weights: FusionWeights) -> list[float]:
    """The neutral CNN prior used by the structural path.

    Uniform, not zeros: ``late_fusion`` treats these as probabilities, and the
    committed coefficients for these slots are exactly ``0.0``, so the value
    provably cannot change the result.  Stating the prior explicitly is what
    makes that provable.
    """
    return [1.0 / len(weights.labels)] * len(weights.labels)


def structural_classify(
    feats: StructuralFeatures, weights: FusionWeights
) -> PageClass:
    """Classify from structural features alone — pure, no image, no model."""
    return late_fusion(uniform_cnn_scores(weights), feats, weights)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def classify_page(
    page: PageImage,
    *,
    feats: StructuralFeatures | None = None,
    lines: Sequence[OCRLine] | None = None,
    provider: ModelProvider | None = None,
    weights: FusionWeights | None = None,
    settings: Settings | None = None,
    use_cnn: bool = True,
    min_confidence: float = MIN_CLASS_CONFIDENCE,
) -> ClassificationResult:
    """Classify one page, preferring the CNN and falling back to structure.

    Args:
        feats: precomputed features; extracted from ``page`` when omitted.
        provider: when given and ``use_cnn`` is true, ``classify_page`` is
            attempted first.  A :class:`~app.providers.base.ModelArtifactMissing`
            (no CNN export, or no fusion file in ``models/``) is caught and the
            structural path runs; every other provider error propagates,
            because "the model is broken" and "the model is absent" are
            different facts and only one of them has a fallback.
        min_confidence: below this the label is demoted to ``unknown`` and
            ``below_confidence_floor`` is set. The scores are still returned, so
            a reviewer sees what the page nearly was.

    Raises:
        ModelArtifactMissing: when the structural path also has no coefficients
            — neither ``models/classifier/fusion.json`` nor the committed
            ``docs/calibration/`` copy is readable. That is a broken checkout,
            not a runtime condition.
    """
    with _CLASSIFY_LATENCY.time():
        features = (
            feats
            if feats is not None
            else extract_structural_features(page, lines=lines, settings=settings)
        )
        fallback_reason: str | None = None
        fused: PageClass | None = None
        mechanism = ClassificationMechanism.STRUCTURAL_FUSION

        if use_cnn and provider is not None:
            try:
                fused = provider.classify_page(page, features)
                mechanism = ClassificationMechanism.CNN_FUSION
            except ModelArtifactMissing as exc:
                _CLASSIFIER_FALLBACKS.inc()
                fallback_reason = exc.code
                _LOG.info(
                    "page-classifier CNN unavailable (%s); using the fitted "
                    "structural path",
                    exc.artifact,
                )

        if fused is None:
            resolved = (
                weights
                if weights is not None
                else load_fusion_weights(settings)
            )
            fused = structural_classify(features, resolved)

        below_floor = fused.confidence < min_confidence
        if below_floor:
            fused = PageClass(
                label=PageClassLabel.UNKNOWN,
                confidence=fused.confidence,
                scores=_with_unknown(fused.scores),
            )

        _CLASSIFICATIONS.labels(
            mechanism=mechanism.value, label=fused.label.value
        ).inc()
        _LOG.debug(
            "stage s3 label=%s confidence=%.3f mechanism=%s lines=%d tables=%d",
            fused.label.value,
            fused.confidence,
            mechanism.value,
            features.line_count,
            features.table_line_count,
        )
        return ClassificationResult(
            page_class=fused,
            mechanism=mechanism,
            features=features,
            artifact_fallback_reason=fallback_reason,
            below_confidence_floor=below_floor,
        )


def _with_unknown(scores: Mapping[str, float]) -> dict[str, float]:
    """Add an ``unknown`` entry carrying the mass the winner did not hold.

    :class:`~app.contracts.ocr.PageClass` requires the predicted label to appear
    in ``scores`` and requires ``scores`` to be a distribution, so demoting to
    ``unknown`` means re-expressing the same distribution with the residual
    named.  The trained per-class scores are preserved and rescaled, never
    discarded: a reviewer needs to see what the page nearly was.
    """
    if not scores:
        return {PageClassLabel.UNKNOWN.value: 1.0}
    best = max(scores.values())
    residual = max(0.0, 1.0 - best)
    combined = {key: value * best for key, value in scores.items()}
    combined[PageClassLabel.UNKNOWN.value] = (
        combined.get(PageClassLabel.UNKNOWN.value, 0.0) + residual
    )
    total = sum(combined.values())
    if total <= 0:  # pragma: no cover - `best` > 0 for any real distribution
        return {PageClassLabel.UNKNOWN.value: 1.0}
    rescaled = {key: value / total for key, value in combined.items()}
    largest = max(rescaled, key=lambda key: (rescaled[key], key))
    rescaled[largest] += 1.0 - sum(rescaled.values())
    return rescaled


def _assert_feature_order() -> None:
    """Fail loudly if the provider's feature order ever changes underneath us.

    ``fusion.json`` stores coefficients **positionally**; a reordering upstream
    would silently reinterpret every one of them.
    """
    expected = (
        "line_count_log",
        "ink_density",
        "aspect_ratio",
        "has_table_lines",
        "table_line_count_log",
        "handwriting_ratio",
        "text_coverage",
    )
    if tuple(STRUCTURAL_FEATURE_NAMES) != expected:
        raise RuntimeError(
            "app.providers.local.STRUCTURAL_FEATURE_NAMES changed to "
            f"{tuple(STRUCTURAL_FEATURE_NAMES)}; the committed fusion.json "
            "stores coefficients positionally and must be retrained "
            "(tools/train_page_classifier.py)"
        )


_assert_feature_order()
