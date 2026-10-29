"""Stage 1.5 — the quality gate: accept, repair or bounce, with reasons.

BUILD-BRIEF M3 / design-review A5 / PRD P0-3.  This is the stage that decides
whether a page is worth recognising, repairs the defects that are cheap and safe
to repair, and — when it refuses — says why in a code an auto-reply template can
be keyed on.

Two rules shape everything here.

**Every page carries a score and flags, including accepted ones.**  A gate that
only records its refusals cannot be monitored: the day the mail stream drifts,
the first symptom is accepted pages whose blur metric has been sliding for a
month.  :class:`QualityAssessment` is produced for every page and its
``quality_score`` / ``quality_flags`` land on
:class:`~app.contracts.envelope.PageResult`.

**Measure, repair, then measure again.**  Skew and inversion are corrected
before the page is scored, because scoring a defect we are about to fix would
bounce documents we can actually read.  The second measurement is skipped when
no repair was applied, which is the common case.

The score is a **weighted geometric mean** of five sub-scores — effective
resolution, sharpness, contrast, text coverage, occlusion — not a weighted sum.
Readability is conjunctive: a page that is sharp, well exposed and 40 DPI is not
"mostly fine", it is unreadable, and an arithmetic mean would let four good
factors carry one fatal one over the line.  Each sub-score is floored at
:attr:`QualityThresholds.subscore_floor` so a single zero cannot make the whole
score exactly zero and destroy the ordering among bad pages.

Thresholds are calibrated against the synthetic corpus (OPEN-QUESTIONS Q3): the
``clean`` and ``light`` profiles must accept, ``scan``/``heavy``/``photo`` must
accept (repair allowed), and the ``bounce`` profile must bounce.  The numbers,
the confusion matrix they produce and the reasoning are in
``docs/decisions-wse1.md`` and ``docs/calibration/quality_gate.json``.

Repairs implemented: **deskew** (rotation correction, the ``skew`` degradation
of every profile) and **auto-invert** (the adversarial ``inverted.png``).  Both
are recorded in :attr:`QualityAssessment.repairs_applied`.

This module does NOT handle:
  * binarisation for recognition — the masks built here are analysis artefacts
    and are deliberately not handed downstream; PP-OCR's own preprocessing is
    tuned for its detector and must not be second-guessed;
  * perspective correction / dewarping.  The ``photo`` profile has perspective,
    and a four-point transform estimated from a page whose borders are partly
    outside the frame moves text more often than it straightens it.  Perspective
    is *scored* (it degrades sharpness and skew consistency) and escalated, not
    repaired;
  * 90/180/270 orientation, either detection or repair.  The projection
    anisotropy measured here does not separate upright from rotated across the
    ten corpus document types — a form's column alignment is as strong as its
    line spacing — so the number is *measured and reported* and no
    ``ROTATION_DETECTED`` flag is raised **from it**.  See
    :func:`_orientation_anisotropy` for the measurement.

    Quarter-turn detection now happens in :func:`app.stages.s4_ocr.page_is_sideways`
    instead, and that is where ``ROTATION_DETECTED`` is raised.  It runs *after*
    detection, which is the whole difference: it reads the shape of the boxes the
    text detector returned rather than trying to infer orientation from the
    raster.  Measured on the same corpus that defeated the projection statistic,
    every upright page scores 1.000 and every sideways page 0.000, on all ten
    types.  A page turned 180 degrees is still invisible to it — an upside-down
    page has horizontal lines too — and nothing is rotated in either case:
    turning a page the wrong way is strictly worse than leaving it alone, and a
    quarter-turn signal does not say which way to turn;
  * denoising, contrast stretching or any other pixel repair that makes the
    image *look* better without being reversible — every repair here is a
    rigid transform or an exact inversion, so the evidence crop a case officer
    opens is still the document that arrived;
  * OCR, text-layer inspection (Stage 1b) or escalation routing (Stage 4);
  * page *rejection* at the ingestion level — a bounce is a verdict recorded in
    the envelope, not an exception.  ``app.contracts.errors`` says the same
    thing from the other side.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.envelope import (
    BounceDecision,
    BounceReason,
    QualityFlag,
    QualityVerdict,
)
from app.settings import Settings, get_settings
from app.stages import configure_cv2_threads

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    PageImage = NDArray[np.uint8]
else:
    PageImage = Any

configure_cv2_threads()

__all__ = [
    "ANALYSIS_LONG_EDGE_PX",
    "GLYPH_HEIGHT_REFERENCE_INCHES",
    "PageQuality",
    "QualityAssessment",
    "QualityGate",
    "QualityMetrics",
    "QualitySubScores",
    "QualityThresholds",
    "auto_invert",
    "deskew",
    "deskew_transform",
    "estimate_skew",
    "hamming_distance",
    "map_bbox_to_original",
    "measure",
    "page_hash",
    "to_grayscale",
]

_LOGGER: Final = logging.getLogger(__name__)

#: Long edge every page is resampled to before analysis.  Fixing it makes the
#: pixel-domain metrics (Laplacian variance above all) comparable between a
#: 300 DPI scan and a 96 DPI phone photo, which is the whole reason the numbers
#: below can be single constants rather than a function of the input size.
ANALYSIS_LONG_EDGE_PX: Final[int] = 1200

#: Physical height of a typical body-text glyph, used to turn a measured glyph
#: height in pixels into an effective DPI.  10 pt type has a cap height of about
#: 0.093 in; 0.09 in makes 200 DPI A4 body text read back as ~195 DPI, which is
#: the calibration anchor for the corpus.
GLYPH_HEIGHT_REFERENCE_INCHES: Final[float] = 0.09

_VERDICTS = Counter(
    "quality_gate_pages_total",
    "Stage 1.5 verdicts per page.",
    ["verdict"],
)
_BOUNCES = Counter(
    "quality_gate_bounces_total",
    "Bounced pages by machine-readable reason.",
    ["reason"],
)
_REPAIRS = Counter(
    "quality_gate_repairs_total",
    "Repairs applied by kind.",
    ["repair"],
)
_SCORES = Histogram(
    "quality_gate_score",
    "Distribution of quality_score over all assessed pages.",
    buckets=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0),
)

#: Names recorded in :attr:`QualityAssessment.repairs_applied`.
REPAIR_DESKEW: Final[str] = "deskew"
REPAIR_AUTO_INVERT: Final[str] = "auto_invert"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class QualityThresholds(BaseModel):
    """Every calibrated number the gate uses, in one frozen object.

    Defaults are the values frozen by the WS-E1 calibration run over the 295-page
    synthetic corpus; ``docs/calibration/quality_gate.json`` is the machine-
    readable copy and records the confusion matrix they produce.  The
    environment-driven knobs come from :class:`~app.settings.Settings` through
    :meth:`from_settings`, so an operator retunes the gate without a code change
    (``docs/TUNING.md``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- hard floors (bounce without appeal) ---
    #: A page smaller than this on its short edge is not a document.  200 px is
    #: one inch at the 200 DPI rasterisation resolution.
    min_page_dimension_px: int = Field(default=200, gt=0)
    #: Ink share below which the page is blank.  The cleanest corpus page has
    #: 8x this, so the margin is wide.
    blank_ink_ratio: float = Field(default=0.0015, ge=0.0, le=1.0)
    #: ...and it must also have essentially no glyph-shaped components.
    blank_glyph_count: int = Field(default=8, ge=0)
    #: Text coverage below which there is nothing to recognise (but the page is
    #: not blank — a stamp, a signature, a single stray mark).
    coverage_floor: float = Field(default=0.004, ge=0.0, le=1.0)
    #: Beyond this the deskew cannot be trusted and the page bounces.
    skew_unrecoverable_deg: float = Field(default=25.0, gt=0.0)

    # --- effective resolution (calibrated: corpus text height / 0.09 in) ---
    effective_dpi_floor: float = Field(default=30.0, gt=0.0)
    effective_dpi_target: float = Field(default=140.0, gt=0.0)
    #: Advisory flag, distinct from the hard floor above.
    effective_dpi_flag: float = Field(default=90.0, gt=0.0)

    # --- sharpness (variance of the Laplacian over the glyph band) ---
    #: Ramped in **log10** space (see :func:`_log_ramp`).  9 500 is just under
    #: the sharpest ``bounce``-profile page in the corpus (11 101) and 26 000 is
    #: just under the softest ``heavy`` page (24 785), which is what puts the
    #: 0.35 verdict boundary inside the gap between the two classes.
    laplacian_floor: float = Field(default=9_500.0, ge=0.0)
    laplacian_target: float = Field(default=26_000.0, gt=0.0)
    laplacian_flag: float = Field(default=40_000.0, ge=0.0)

    # --- contrast (background minus ink, on the normalised image) ---
    contrast_floor: float = Field(default=17.0, ge=0.0)
    contrast_target: float = Field(default=70.0, gt=0.0)

    # --- coverage band ---
    coverage_good_low: float = Field(default=0.012, ge=0.0, le=1.0)
    coverage_good_high: float = Field(default=0.45, ge=0.0, le=1.0)
    coverage_ceiling: float = Field(default=0.80, ge=0.0, le=1.0)

    # --- occlusion ---
    occlusion_flag: float = Field(default=0.05, ge=0.0, le=1.0)
    occlusion_ceiling: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- advisory-only observations ---
    #: Immerkaer sigma over background pixels.  4.0 separates the corpus's 3%
    #: noise profile (5.0-6.5) from its 1.2% one (2.6-3.4).
    noise_ceiling: float = Field(default=4.0, ge=0.0)
    #: Illumination non-uniformity.  Corpus pages with no shadow measure
    #: <= 0.002; every profile that applies a shadow or an exposure ramp
    #: measures >= 0.024, so 0.02 is a two-order-of-magnitude gap, not a guess.
    shadow_ceiling: float = Field(default=0.02, ge=0.0)
    exposure_low: float = Field(default=70.0, ge=0.0, le=255.0)
    exposure_high: float = Field(default=250.0, ge=0.0, le=255.0)

    # --- repairs ---
    #: Below this the skew is not worth a resample (interpolation costs more
    #: sharpness than the rotation recovers).
    skew_repair_deg: float = Field(default=0.75, gt=0.0)
    #: Search half-range of the projection-profile skew estimator.
    skew_search_deg: float = Field(default=20.0, gt=0.0)
    #: p75 of the grayscale below which the page is treated as inverted.
    invert_background_level: float = Field(default=110.0, ge=0.0, le=255.0)

    # --- near-duplicate pages ---
    duplicate_hamming_max: int = Field(default=6, ge=0, le=64)

    # --- aggregation (exponents of the weighted geometric mean) ---
    weight_resolution: float = Field(default=0.20, ge=0.0)
    weight_sharpness: float = Field(default=0.45, ge=0.0)
    weight_contrast: float = Field(default=0.25, ge=0.0)
    weight_coverage: float = Field(default=0.06, ge=0.0)
    weight_occlusion: float = Field(default=0.04, ge=0.0)
    #: Floor applied to every sub-score before the geometric mean, so one zero
    #: cannot collapse the score and erase the ordering among bad pages.
    subscore_floor: float = Field(default=0.05, gt=0.0, le=1.0)

    #: Mirrors ``QUALITY_BOUNCE_THRESHOLD``; below this the page bounces.
    bounce_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> QualityThresholds:
        """Build from :class:`~app.settings.Settings`.

        Only the *tuning* knobs are environment-driven; the model structure —
        sub-score weights, advisory flag levels, the analysis geometry — stays
        in this module, because changing those changes what the score means and
        invalidates the calibration rather than retuning it.
        ``QUALITY_BOUNCE_THRESHOLD`` is the one BUILD-BRIEF sec. 6 names, and
        the one an operator moves first when the bounce rate leaves the 5-15%
        band (``docs/TUNING.md``).
        """
        s = settings if settings is not None else get_settings()
        return cls(
            bounce_threshold=s.QUALITY_BOUNCE_THRESHOLD,
            laplacian_floor=s.QUALITY_BLUR_LAPLACIAN_FLOOR,
            laplacian_target=s.QUALITY_BLUR_LAPLACIAN_TARGET,
            contrast_floor=s.QUALITY_CONTRAST_FLOOR,
            contrast_target=s.QUALITY_CONTRAST_TARGET,
            effective_dpi_floor=s.QUALITY_EFFECTIVE_DPI_FLOOR,
            effective_dpi_target=s.QUALITY_EFFECTIVE_DPI_TARGET,
            coverage_floor=s.QUALITY_TEXT_COVERAGE_FLOOR,
            skew_repair_deg=s.QUALITY_SKEW_REPAIR_DEG,
            skew_unrecoverable_deg=s.QUALITY_SKEW_BOUNCE_DEG,
            min_page_dimension_px=s.QUALITY_MIN_PAGE_DIMENSION_PX,
        )


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


class QualityMetrics(BaseModel):
    """Raw, unweighted measurements of one page.  No policy, no verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    #: Scale factor from source pixels to the analysis copy.
    analysis_scale: float = Field(gt=0.0)
    ink_ratio: float = Field(ge=0.0, le=1.0)
    #: Share of the page covered by text after horizontal closing.
    text_coverage: float = Field(ge=0.0, le=1.0)
    glyph_count: int = Field(ge=0)
    #: Median glyph height in **source** pixels.
    glyph_height_px: float = Field(ge=0.0)
    #: Derived from :attr:`glyph_height_px`; the "effective resolution" of
    #: BUILD-BRIEF M3, which is what matters to a recogniser, not the nominal
    #: scan DPI a device claims.
    effective_dpi: float = Field(ge=0.0)
    #: Positive means the page must be rotated counter-clockwise to straighten.
    skew_deg: float
    #: Ratio of the projection-profile score at the chosen angle to the score at
    #: 0 degrees.  ~1.0 means the estimator found nothing and the angle is noise.
    skew_confidence: float = Field(ge=0.0)
    #: Variance of the Laplacian over the dilated glyph band of the
    #: illumination-normalised image (BUILD-BRIEF M3 names this metric).
    laplacian_variance: float = Field(ge=0.0)
    #: Background level minus ink level on the normalised image, 0-255.
    contrast: float = Field(ge=0.0)
    #: Largest solid non-text blob as a share of page area (thumb, shadow band).
    occlusion_share: float = Field(ge=0.0, le=1.0)
    #: Immerkaer sigma estimate over background pixels only.
    noise_sigma: float = Field(ge=0.0)
    #: Standard deviation of the illumination estimate over its mean.
    shadow_index: float = Field(ge=0.0)
    #: Mean grayscale level of the page, 0-255.
    exposure_mean: float = Field(ge=0.0, le=255.0)
    is_inverted: bool
    is_blank: bool
    #: 1.0 = rows and columns equally structured; > 1 = horizontal text lines.
    orientation_anisotropy: float = Field(ge=0.0)
    #: 64-bit difference hash, hex, for near-duplicate page detection.
    page_hash: str = Field(min_length=16, max_length=16)


class QualitySubScores(BaseModel):
    """The five factors of the geometric mean, each in [0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution: float = Field(ge=0.0, le=1.0)
    sharpness: float = Field(ge=0.0, le=1.0)
    contrast: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    occlusion: float = Field(ge=0.0, le=1.0)


class QualityAssessment(BaseModel):
    """One page's verdict, score, flags and the evidence behind them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_no: int = Field(ge=1)
    verdict: QualityVerdict
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_flags: tuple[QualityFlag, ...] = ()
    #: Set exactly when ``verdict`` is ``bounce``.
    bounce_reason: BounceReason | None = None
    sub_scores: QualitySubScores
    metrics: QualityMetrics
    repairs_applied: tuple[str, ...] = ()
    #: Human-readable elaboration for the auto-reply. Never parsed.
    detail: str | None = None

    @property
    def bounced(self) -> bool:
        return self.verdict is QualityVerdict.BOUNCE

    def to_bounce_decision(self) -> BounceDecision:
        """The envelope-level decision this page argues for."""
        if self.bounce_reason is None:
            return BounceDecision(decision=self.verdict)
        return BounceDecision(
            decision=QualityVerdict.BOUNCE,
            reason=self.bounce_reason,
            detail=self.detail,
        )


@dataclass(frozen=True, slots=True)
class PageQuality:
    """An assessment plus the repaired pixels the rest of the pipeline uses.

    A dataclass because it carries an image; see the same note on
    :class:`app.stages.s1_ingest.RasterPage`.  ``image`` is ``None`` when the
    page bounced — nothing runs after a bounce, and handing back pixels invites
    someone to run it anyway.
    """

    assessment: QualityAssessment
    image: PageImage | None


# --------------------------------------------------------------------------- #
# Primitive image operations
# --------------------------------------------------------------------------- #


def to_grayscale(image: PageImage) -> PageImage:
    """Single-channel ``uint8`` view of a BGR, BGRA or already-gray page."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        gray = array[:, :, 0]
    else:
        raise ValueError(
            f"page image must be (H, W), (H, W, 3) or (H, W, 4), got "
            f"shape {array.shape!r}"
        )
    return np.ascontiguousarray(gray, dtype=np.uint8)


def _analysis_copy(gray: PageImage) -> tuple[PageImage, float]:
    """Resample to :data:`ANALYSIS_LONG_EDGE_PX` and report the scale used."""
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= ANALYSIS_LONG_EDGE_PX:
        return gray, 1.0
    scale = ANALYSIS_LONG_EDGE_PX / longest
    resized = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _illumination(gray: PageImage) -> tuple[PageImage, PageImage]:
    """Return ``(normalised, background)``.

    Morphological closing with a kernel wider than any glyph recovers the paper,
    including a shadow gradient or an exposure ramp; dividing by it removes the
    illumination without touching stroke contrast.  Doing this first is what
    lets a single Otsu threshold work on a phone photo of a folded page, and
    skipping it is why a naive gate reports 50% ink on the ``photo`` profile.
    """
    height, width = gray.shape[:2]
    longest = max(height, width)
    kernel_size = max(15, (int(round(longest / 24)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), longest / 60.0)
    normalised = cv2.divide(gray, np.maximum(background, 1), scale=255)
    return normalised, background


def _ink_mask(normalised: PageImage) -> PageImage:
    """Otsu on the illumination-normalised page; ink is 255."""
    smoothed = cv2.medianBlur(normalised, 3)
    _, mask = cv2.threshold(
        smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask


#: Percentile of glyph-component heights used as the page's text height.
#: **Not the median.**  A noisy page contributes hundreds of 2-3 px speckles
#: that survive thresholding; they drag a median to 3 px on a page whose text is
#: 17 px, which is the difference between "accept" and "bounce".  The upper
#: quartile sits above the speckle and below the merged-blob tail that
#: area-weighting would chase.
GLYPH_HEIGHT_PERCENTILE: Final[float] = 75.0


def _glyph_mask(ink: PageImage) -> tuple[PageImage, int, float]:
    """Keep only glyph-shaped components; report count and text height.

    Filtering by component geometry is what makes every later metric behave on a
    degraded page: punch holes, fold lines, stamps and shadow edges all survive
    the threshold and all fail these bounds.
    """
    longest = max(ink.shape[:2])
    max_height = longest / 12.0
    max_width = longest / 4.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), 8
    )
    keep: list[int] = []
    heights: list[int] = []
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if height < 2 or height > max_height or width > max_width or area < 4:
            continue
        keep.append(index)
        heights.append(height)
    if not keep:
        return np.zeros_like(ink), 0, 0.0
    mask = (np.isin(labels, keep).astype(np.uint8)) * 255
    text_height = float(np.percentile(heights, GLYPH_HEIGHT_PERCENTILE))
    return mask, len(keep), text_height


def estimate_skew(
    glyphs: PageImage, *, search_deg: float = 20.0, work_px: int = 500
) -> tuple[float, float]:
    """Estimate page skew by projection-profile maximisation.

    Returns ``(degrees, confidence)`` where ``degrees`` is the correction to
    apply (positive = rotate counter-clockwise) and ``confidence`` is the ratio
    of the profile score at that angle to the score at zero — near 1.0 means the
    estimator found no structure and the angle should not be trusted.

    Coarse 1 degree sweep then a 0.1 degree refinement, on a 500 px copy of the
    **glyph** mask: the variance of the row-projection of text is sharply peaked
    at the true angle, and running it on the raw ink mask instead is what makes
    a naive implementation report 10 degrees of error on a page with a fold.
    """
    if glyphs.size == 0:
        return 0.0, 0.0
    height, width = glyphs.shape[:2]
    scale = work_px / max(height, width)
    if scale < 1.0:
        small = cv2.resize(
            glyphs,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = glyphs
    binary = (small > 100).astype(np.float32)
    if binary.sum() < 20:
        return 0.0, 0.0
    rows, cols = binary.shape
    centre = (cols / 2.0, rows / 2.0)

    def score(angle: float) -> float:
        matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, matrix, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=0.0
        )
        projection = rotated.sum(axis=1)
        return float(((projection - projection.mean()) ** 2).sum())

    coarse = float(
        max(np.arange(-search_deg, search_deg + 1e-6, 1.0), key=score)
    )
    best = float(max(np.arange(coarse - 1.0, coarse + 1.0 + 1e-6, 0.1), key=score))
    baseline = score(0.0)
    confidence = score(best) / baseline if baseline > 0.0 else 0.0
    return round(best, 2), confidence


def _occlusion_share(ink: PageImage) -> float:
    """Largest solid, non-text blob as a share of the page."""
    longest = max(ink.shape[:2])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (solid > 0).astype(np.uint8), 8
    )
    biggest = 0
    minimum_side = longest / 8.0
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if width >= minimum_side and height >= minimum_side:
            biggest = max(biggest, area)
    return float(biggest / ink.size) if ink.size else 0.0


def _noise_sigma(gray: PageImage, ink: PageImage) -> float:
    """Immerkaer noise estimate restricted to background pixels."""
    kernel = np.array(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float64
    )
    response = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    background = cv2.erode((ink == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if int(background.sum()) < 100:
        return 0.0
    return float(
        math.sqrt(math.pi / 2.0) * np.abs(response[background]).mean() / 6.0
    )


def _orientation_anisotropy(glyphs: PageImage) -> float:
    """Row-projection structure divided by column-projection structure.

    **Reported, never acted on, and the measurement below says why.**  The
    textbook claim is that upright text has far more structure along rows than
    along columns, so the ratio detects a 90-degree rotation.  Measured over the
    corpus it does not: upright pages span 0.29 to 17.8 across the ten document
    types, because a *form* — label column beside value column — has column
    alignment as strong as its line spacing, and a statement table more so.
    Rotating the same page 90 degrees moves the ratio, but the two populations
    overlap completely, so no threshold separates them.

    The number stays because it is cheap, it is real evidence, and a page whose
    anisotropy is wildly out of family with the rest of its document is worth
    seeing in the audit trail.  A trustworthy 0/90/180/270 decision needs the
    text-orientation model in Stage 2, and no flag is raised from this value.
    """
    if glyphs.size == 0 or int((glyphs > 0).sum()) < 20:
        return 1.0
    binary = (glyphs > 0).astype(np.float64)
    rows = binary.sum(axis=1)
    cols = binary.sum(axis=0)
    row_energy = float(np.diff(rows).var())
    col_energy = float(np.diff(cols).var())
    if col_energy <= 0.0:
        return float("inf") if row_energy > 0.0 else 1.0
    return row_energy / col_energy


def page_hash(gray: PageImage) -> str:
    """64-bit difference hash of a page, as 16 hex characters.

    dHash rather than average hash: it compares adjacent columns, so it is
    invariant to the exposure and gamma differences that make two scans of the
    same page look different to a mean-based hash.
    """
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    """Bit distance between two :func:`page_hash` values.

    Raises:
        ValueError: when either value is not a 16-character hex hash.
    """
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError as exc:
        raise ValueError(
            f"page hashes must be 16 hex characters, got {left!r} and {right!r}"
        ) from exc


def measure(
    image: PageImage, thresholds: QualityThresholds | None = None
) -> QualityMetrics:
    """Measure one page.  Pure: no verdict, no repair, no state.

    Raises:
        ValueError: when ``image`` is not a 2-D or 3-D ``uint8`` page.
    """
    limits = thresholds if thresholds is not None else QualityThresholds()
    gray_full = to_grayscale(image)
    source_height, source_width = gray_full.shape[:2]
    gray, scale = _analysis_copy(gray_full)

    normalised, background = _illumination(gray)
    ink = _ink_mask(normalised)
    ink_ratio = float((ink > 0).mean())
    glyphs, glyph_count, glyph_height_work = _glyph_mask(ink)

    glyph_height_px = glyph_height_work / scale if scale > 0.0 else 0.0
    effective_dpi = (
        glyph_height_px / GLYPH_HEIGHT_REFERENCE_INCHES
        if glyph_height_px > 0.0
        else 0.0
    )

    skew_deg, skew_confidence = estimate_skew(
        glyphs, search_deg=limits.skew_search_deg
    )

    band = (
        cv2.dilate((glyphs > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    )
    if int(band.sum()) >= 50:
        laplacian = cv2.Laplacian(normalised.astype(np.float64), cv2.CV_64F, ksize=3)
        laplacian_variance = float(laplacian[band].var())
        ink_level = float(normalised[glyphs > 0].mean())
        background_pixels = normalised[ink == 0]
        background_level = (
            float(background_pixels.mean()) if background_pixels.size else 255.0
        )
        contrast = max(0.0, background_level - ink_level)
    else:
        laplacian_variance = 0.0
        contrast = 0.0

    close_width = max(3, int(round(max(gray.shape[:2]) / 120)) | 1)
    closed = cv2.morphologyEx(
        glyphs,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 3)),
    )
    text_coverage = float((closed > 0).mean())

    background_mean = float(background.mean())
    shadow_index = (
        float(background.std()) / background_mean if background_mean > 0.0 else 0.0
    )
    exposure_mean = float(gray.mean())
    is_inverted = float(np.percentile(gray, 75)) < limits.invert_background_level
    is_blank = (
        ink_ratio < limits.blank_ink_ratio and glyph_count <= limits.blank_glyph_count
    )

    return QualityMetrics(
        width_px=source_width,
        height_px=source_height,
        analysis_scale=scale,
        ink_ratio=ink_ratio,
        text_coverage=text_coverage,
        glyph_count=glyph_count,
        glyph_height_px=glyph_height_px,
        effective_dpi=effective_dpi,
        skew_deg=skew_deg,
        skew_confidence=skew_confidence,
        laplacian_variance=laplacian_variance,
        contrast=contrast,
        occlusion_share=_occlusion_share(ink),
        noise_sigma=_noise_sigma(gray, ink),
        shadow_index=shadow_index,
        exposure_mean=exposure_mean,
        is_inverted=is_inverted,
        is_blank=is_blank,
        orientation_anisotropy=min(_orientation_anisotropy(glyphs), 1e6),
        page_hash=page_hash(gray),
    )


# --------------------------------------------------------------------------- #
# Repairs
# --------------------------------------------------------------------------- #


def auto_invert(image: PageImage) -> PageImage:
    """Invert a light-on-dark page.

    Exact (``255 - x``) and therefore reversible, which is why it is allowed as
    a repair at all: the evidence crop a case officer opens is still the
    document that arrived, only readable.
    """
    return np.asarray(255 - np.asarray(image), dtype=np.uint8)


def deskew_transform(
    height: int, width: int, degrees: float
) -> tuple[Any, tuple[int, int]]:
    """The affine :func:`deskew` applies, and the canvas size it needs.

    Split out from :func:`deskew` because the inverse of this matrix is what
    turns a box found on the straightened page back into a box on the page the
    complainant actually sent (:func:`map_bbox_to_original`).  Deriving it twice
    would be two chances to disagree about the centre of rotation, and a
    half-degree disagreement is a box that sits under the wrong form field.

    Returns:
        ``(matrix, (new_width, new_height))`` — a 2x3 affine mapping original
        pixel coordinates to straightened ones.
    """
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]
    return matrix, (new_width, new_height)


def map_bbox_to_original(
    bbox: Sequence[float], *, height: int, width: int, degrees: float
) -> list[float]:
    """Map a quadrilateral found on the straightened page back to the original.

    ``height``/``width`` are the **original** page's, and ``degrees`` the
    rotation :func:`deskew` applied — the same two numbers that produced the
    straightened raster, so the inverse is exact rather than estimated.

    Why the envelope carries original-frame boxes: a case officer opens the
    file that arrived and expects the evidence box to sit on the field it
    quotes.  Publishing straightened-frame coordinates made every box on a
    repaired page appear rotated (and, because the straightening also *expands*
    the canvas, offset as well) — which is what a reviewer sees as "the box is
    on the wrong field".
    """
    matrix, _size = deskew_transform(height, width, degrees)
    inverse = cv2.invertAffineTransform(matrix)
    points = np.asarray(bbox, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    mapped = np.hstack([points, ones]) @ np.asarray(inverse, dtype=np.float64).T
    return [float(value) for value in mapped.reshape(-1)]


def deskew(image: PageImage, degrees: float) -> PageImage:
    """Rotate ``image`` by ``degrees`` counter-clockwise about its centre.

    The canvas is **expanded** so no text is cropped, and the exposed corners
    are filled with the page's own border median rather than white — a white
    wedge against a grey scan is a hard edge the detector will happily find a
    text line along.
    """
    array = np.asarray(image)
    if abs(degrees) < 1e-6:
        return np.ascontiguousarray(array)
    height, width = array.shape[:2]
    matrix, size = deskew_transform(height, width, degrees)
    fill = _border_median(array)
    return cv2.warpAffine(
        array,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _border_median(array: np.ndarray) -> Any:
    """Median colour of the outermost 8-pixel frame."""
    edge = 8
    top = array[:edge]
    bottom = array[-edge:]
    left = array[:, :edge]
    right = array[:, -edge:]
    if array.ndim == 2:
        pixels = np.concatenate(
            [top.ravel(), bottom.ravel(), left.ravel(), right.ravel()]
        )
        return float(np.median(pixels))
    stacked = np.concatenate(
        [
            top.reshape(-1, array.shape[2]),
            bottom.reshape(-1, array.shape[2]),
            left.reshape(-1, array.shape[2]),
            right.reshape(-1, array.shape[2]),
        ]
    )
    return tuple(float(v) for v in np.median(stacked, axis=0))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _ramp(value: float, floor: float, target: float) -> float:
    """Linear 0-at-``floor`` to 1-at-``target``, clamped."""
    if target <= floor:
        return 1.0 if value >= target else 0.0
    return min(1.0, max(0.0, (value - floor) / (target - floor)))


def _log_ramp(value: float, floor: float, target: float) -> float:
    """:func:`_ramp` in log10 space, for a quantity that spans decades.

    Laplacian variance runs from ~3 500 on the worst corpus page to ~310 000 on
    the best — a factor of 90.  A linear ramp across that range puts every
    degraded profile in the bottom 10% of the scale and makes the difference
    between "blurry" and "unreadable" invisible; in log space the six corpus
    degradation profiles come out almost evenly spaced, which is the shape a
    threshold can actually be set on.
    """
    return _ramp(
        math.log10(max(value, 1.0)),
        math.log10(max(floor, 1.0)),
        math.log10(max(target, 1.0)),
    )


def _coverage_score(coverage: float, limits: QualityThresholds) -> float:
    """Trapezoid: too little text is as bad as a page that is one black smear."""
    if coverage <= limits.coverage_floor:
        return 0.0
    if coverage < limits.coverage_good_low:
        return _ramp(coverage, limits.coverage_floor, limits.coverage_good_low)
    if coverage <= limits.coverage_good_high:
        return 1.0
    if coverage >= limits.coverage_ceiling:
        return 0.0
    return 1.0 - _ramp(
        coverage, limits.coverage_good_high, limits.coverage_ceiling
    )


def _sub_scores(
    metrics: QualityMetrics, limits: QualityThresholds
) -> QualitySubScores:
    return QualitySubScores(
        resolution=_ramp(
            metrics.effective_dpi,
            limits.effective_dpi_floor,
            limits.effective_dpi_target,
        ),
        sharpness=_log_ramp(
            metrics.laplacian_variance,
            limits.laplacian_floor,
            limits.laplacian_target,
        ),
        contrast=_ramp(
            metrics.contrast, limits.contrast_floor, limits.contrast_target
        ),
        coverage=_coverage_score(metrics.text_coverage, limits),
        occlusion=1.0
        - _ramp(metrics.occlusion_share, limits.occlusion_flag, limits.occlusion_ceiling),
    )


def _aggregate(scores: QualitySubScores, limits: QualityThresholds) -> float:
    """Weighted geometric mean, with every factor floored.

    Geometric, not arithmetic: see the module docstring.
    """
    weighted = (
        (scores.resolution, limits.weight_resolution),
        (scores.sharpness, limits.weight_sharpness),
        (scores.contrast, limits.weight_contrast),
        (scores.coverage, limits.weight_coverage),
        (scores.occlusion, limits.weight_occlusion),
    )
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0.0:
        raise ValueError("quality-gate weights must not all be zero")
    accumulated = sum(
        weight * math.log(max(value, limits.subscore_floor))
        for value, weight in weighted
    )
    return min(1.0, max(0.0, math.exp(accumulated / total_weight)))


def _advisory_flags(
    metrics: QualityMetrics, limits: QualityThresholds
) -> list[QualityFlag]:
    """Observations recorded on every page, accepted ones included."""
    flags: list[QualityFlag] = []
    if metrics.effective_dpi and metrics.effective_dpi < limits.effective_dpi_flag:
        flags.append(QualityFlag.RESOLUTION_BELOW_FLOOR)
    if metrics.laplacian_variance < limits.laplacian_flag:
        flags.append(QualityFlag.BLUR_ABOVE_CEILING)
    if metrics.occlusion_share >= limits.occlusion_flag:
        flags.append(QualityFlag.OCCLUSION_DETECTED)
    if metrics.shadow_index >= limits.shadow_ceiling:
        flags.append(QualityFlag.SHADOW_DETECTED)
    if metrics.noise_sigma >= limits.noise_ceiling:
        flags.append(QualityFlag.NOISE_ABOVE_CEILING)
    if not (limits.exposure_low <= metrics.exposure_mean <= limits.exposure_high):
        flags.append(QualityFlag.EXPOSURE_OUT_OF_RANGE)
    # No ROTATION_DETECTED here on purpose — see _orientation_anisotropy.
    return flags


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class QualityGate:
    """Scores, repairs and judges pages; remembers hashes within a document.

    One instance per document.  The only state is the set of page hashes seen so
    far, which is what near-duplicate detection needs and is also why the
    instance must not be shared between documents: two complainants sending the
    same bank circular are not duplicates of each other.
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds: Final[QualityThresholds] = (
            thresholds if thresholds is not None else QualityThresholds()
        )
        self._hashes: list[tuple[int, str]] = []

    # -- public API ------------------------------------------------------- #

    def reset(self) -> None:
        """Forget the page hashes seen so far (start of a new document)."""
        self._hashes.clear()

    def evaluate(
        self,
        image: PageImage,
        page_no: int,
        *,
        extra_flags: Iterable[QualityFlag] = (),
    ) -> PageQuality:
        """Assess one page, applying repairs, and return pixels plus verdict.

        ``extra_flags`` lets an earlier stage contribute an observation it alone
        can make — Stage 1b passes
        :attr:`~app.contracts.envelope.QualityFlag.JUNK_TEXT_LAYER` so the flag
        reaches the envelope through the one object that carries page quality.
        """
        limits = self.thresholds
        metrics = measure(image, limits)
        flags: list[QualityFlag] = list(extra_flags)
        repairs: list[str] = []
        working: PageImage = np.asarray(image)

        # 1. Hard floor that no repair can lift: too few pixels to read.
        if min(metrics.width_px, metrics.height_px) < limits.min_page_dimension_px:
            return self._bounce(
                page_no,
                metrics,
                flags + [QualityFlag.RESOLUTION_BELOW_FLOOR],
                BounceReason.RESOLUTION_BELOW_FLOOR,
                repairs,
                detail=(
                    f"page is {metrics.width_px}x{metrics.height_px} px; the floor "
                    f"is {limits.min_page_dimension_px} px on the short edge"
                ),
            )

        # 2. Inversion first, and re-measure: on a light-on-dark page every
        #    other metric is measuring the wrong polarity, so nothing below can
        #    be trusted until it is corrected.  The inversion is exact, so the
        #    re-measured page is still the document that arrived.
        if metrics.is_inverted:
            working = auto_invert(working)
            repairs.append(REPAIR_AUTO_INVERT)
            flags.append(QualityFlag.INVERTED_REPAIRED)
            metrics = measure(working, limits)

        skew = metrics.skew_deg
        if abs(skew) > limits.skew_unrecoverable_deg:
            return self._bounce(
                page_no,
                metrics,
                flags + [QualityFlag.SKEW_DETECTED, QualityFlag.SKEW_UNRECOVERABLE],
                BounceReason.SKEW_UNRECOVERABLE,
                repairs,
                detail=(
                    f"page is skewed {skew:+.1f} deg, beyond the "
                    f"{limits.skew_unrecoverable_deg:.0f} deg recoverable limit"
                ),
            )
        # 3. Deskew is applied to the pixels the recogniser will see, but the
        #    score keeps the pre-deskew measurement.  Rotating a page resamples
        #    it, and bilinear interpolation costs real Laplacian variance —
        #    scoring the rotated copy would charge the document for a defect we
        #    introduced while repairing it, and on the corpus that alone pushed
        #    a third of the `photo` profile below the bounce threshold.
        if abs(skew) >= limits.skew_repair_deg:
            flags.append(QualityFlag.SKEW_DETECTED)
            working = deskew(working, skew)
            repairs.append(REPAIR_DESKEW)
            flags.append(QualityFlag.SKEW_REPAIRED)
        elif not repairs:
            working = np.ascontiguousarray(np.asarray(image))

        # 4. Hard verdicts, from the measurement the score is built on.
        if metrics.is_blank:
            return self._bounce(
                page_no,
                metrics,
                flags + [QualityFlag.BLANK_PAGE],
                BounceReason.BLANK_PAGE,
                repairs,
                detail="page carries no ink; OCR was not billed",
            )
        if metrics.text_coverage <= limits.coverage_floor:
            return self._bounce(
                page_no,
                metrics,
                flags + [QualityFlag.TEXT_COVERAGE_BELOW_FLOOR],
                BounceReason.TEXT_COVERAGE_BELOW_FLOOR,
                repairs,
                detail=(
                    f"text covers {metrics.text_coverage:.2%} of the page; the "
                    f"floor is {limits.coverage_floor:.2%}"
                ),
            )

        # 5. Near-duplicate detection, against pages already seen this document.
        duplicate_of = self._duplicate_of(metrics.page_hash)
        if duplicate_of is not None:
            flags.append(QualityFlag.DUPLICATE_PAGE)
        self._hashes.append((page_no, metrics.page_hash))

        # 6. Score.
        flags.extend(_advisory_flags(metrics, limits))
        sub_scores = _sub_scores(metrics, limits)
        score = _aggregate(sub_scores, limits)
        _SCORES.observe(score)

        if score < limits.bounce_threshold:
            return self._bounce(
                page_no,
                metrics,
                flags,
                BounceReason.QUALITY_SCORE_BELOW_THRESHOLD,
                repairs,
                sub_scores=sub_scores,
                score=score,
                detail=(
                    f"quality score {score:.2f} is below the "
                    f"{limits.bounce_threshold:.2f} threshold"
                ),
            )

        verdict = QualityVerdict.REPAIR if repairs else QualityVerdict.ACCEPT
        assessment = QualityAssessment(
            page_no=page_no,
            verdict=verdict,
            quality_score=score,
            quality_flags=_dedupe(flags),
            bounce_reason=None,
            sub_scores=sub_scores,
            metrics=metrics,
            repairs_applied=tuple(repairs),
            detail=(
                f"duplicate of page {duplicate_of}"
                if duplicate_of is not None
                else None
            ),
        )
        _VERDICTS.labels(verdict=verdict.value).inc()
        for repair in repairs:
            _REPAIRS.labels(repair=repair).inc()
        return PageQuality(assessment=assessment, image=working)

    def evaluate_document(
        self, images: Sequence[PageImage], *, first_page_no: int = 1
    ) -> list[PageQuality]:
        """Assess every page of one document, in order.

        Near-duplicate detection is per document and therefore lives here; the
        instance is reset first so a gate can be reused across documents without
        the previous one's hashes leaking in.
        """
        self.reset()
        return [
            self.evaluate(image, first_page_no + offset)
            for offset, image in enumerate(images)
        ]

    # -- internals -------------------------------------------------------- #

    def _duplicate_of(self, digest: str) -> int | None:
        for page_no, seen in self._hashes:
            if hamming_distance(seen, digest) <= self.thresholds.duplicate_hamming_max:
                return page_no
        return None

    def _bounce(
        self,
        page_no: int,
        metrics: QualityMetrics,
        flags: Sequence[QualityFlag],
        reason: BounceReason,
        repairs: Sequence[str],
        *,
        sub_scores: QualitySubScores | None = None,
        score: float | None = None,
        detail: str | None = None,
    ) -> PageQuality:
        resolved_scores = (
            sub_scores
            if sub_scores is not None
            else _sub_scores(metrics, self.thresholds)
        )
        resolved_score = (
            score if score is not None else _aggregate(resolved_scores, self.thresholds)
        )
        assessment = QualityAssessment(
            page_no=page_no,
            verdict=QualityVerdict.BOUNCE,
            quality_score=resolved_score,
            quality_flags=_dedupe(flags),
            bounce_reason=reason,
            sub_scores=resolved_scores,
            metrics=metrics,
            repairs_applied=tuple(repairs),
            detail=detail,
        )
        _VERDICTS.labels(verdict=QualityVerdict.BOUNCE.value).inc()
        _BOUNCES.labels(reason=reason.value).inc()
        _LOGGER.info(
            "page %d bounced: %s (score %.2f)", page_no, reason.value, resolved_score
        )
        return PageQuality(assessment=assessment, image=None)


def _dedupe(flags: Iterable[QualityFlag]) -> tuple[QualityFlag, ...]:
    """Preserve order, drop repeats — ``PageResult`` rejects duplicates."""
    seen: set[QualityFlag] = set()
    ordered: list[QualityFlag] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            ordered.append(flag)
    return tuple(ordered)
