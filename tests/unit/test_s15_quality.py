"""Stage 1.5 — the quality gate.

Pages are **synthesised from primitives**, not loaded from the corpus and not
rendered from a font.  A row of small dark rectangles on light paper is, to
every metric this stage computes, a line of text: it has glyph-shaped connected
components, a projection profile, an ink ratio and a stroke edge.  Building them
that way buys three things a fixture image cannot:

* the defect under test is the *only* difference between two pages, so a failing
  assertion names one cause;
* the suite runs with no font, no Raqm and no corpus on disk;
* every threshold can be approached from both sides, which is the only way to
  show a boundary is where the calibration says it is.

The corpus is still the authority on whether the *thresholds* are right — that
is ``tests/golden/test_quality_calibration.py`` and
``docs/calibration/quality_gate.json``.  This file is the authority on whether
the *mechanism* is right.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.contracts.envelope import BounceReason, QualityFlag, QualityVerdict
from app.settings import Settings
from app.stages.s15_quality import (
    ANALYSIS_LONG_EDGE_PX,
    REPAIR_AUTO_INVERT,
    REPAIR_DESKEW,
    PageQuality,
    QualityGate,
    QualitySubScores,
    QualityThresholds,
    _aggregate,
    auto_invert,
    deskew,
    deskew_transform,
    estimate_skew,
    hamming_distance,
    map_bbox_to_original,
    measure,
    page_hash,
    to_grayscale,
)

# --------------------------------------------------------------------------- #
# Synthetic pages
# --------------------------------------------------------------------------- #

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
PAPER = 245
INK = 20


def text_page(
    *,
    width: int = PAGE_WIDTH,
    height: int = PAGE_HEIGHT,
    glyph_h: int = 18,
    glyph_w: int = 10,
    gap: int = 6,
    row_pitch: int = 48,
    margin: int = 110,
    ink: int = INK,
    paper: int = PAPER,
) -> np.ndarray:
    """A BGR page of glyph-shaped components arranged in text lines.

    Defaults produce a page that measures ~211 effective DPI, ~29% text coverage
    and no skew — i.e. the ``clean`` corpus profile, which the gate must accept
    with a score of 1.0 and no flags.
    """
    image = np.full((height, width, 3), paper, np.uint8)
    y = margin
    while y + glyph_h <= height - margin:
        x = margin
        while x + glyph_w <= width - margin:
            cv2.rectangle(image, (x, y), (x + glyph_w, y + glyph_h), (ink,) * 3, -1)
            x += glyph_w + gap
        y += row_pitch
    return image


def blank_page(*, paper: int = PAPER) -> np.ndarray:
    return np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), paper, np.uint8)


@pytest.fixture
def clean_page() -> np.ndarray:
    return text_page()


@pytest.fixture
def gate() -> QualityGate:
    return QualityGate()


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.zeros((40, 30), np.uint8), id="gray"),
        pytest.param(np.zeros((40, 30, 1), np.uint8), id="gray_with_axis"),
        pytest.param(np.zeros((40, 30, 3), np.uint8), id="bgr"),
        pytest.param(np.zeros((40, 30, 4), np.uint8), id="bgra"),
    ],
)
def test_to_grayscale_accepts_every_page_layout(array: np.ndarray) -> None:
    gray = to_grayscale(array)
    assert gray.shape == (40, 30)
    assert gray.dtype == np.uint8
    assert gray.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.zeros((40, 30, 2), np.uint8), id="two_channels"),
        pytest.param(np.zeros((4, 5, 6, 7), np.uint8), id="four_dimensions"),
        pytest.param(np.zeros(12, np.uint8), id="one_dimension"),
    ],
)
def test_to_grayscale_refuses_a_shape_it_cannot_mean(array: np.ndarray) -> None:
    """Typed failure, not a silent reinterpretation of somebody's tensor."""
    with pytest.raises(ValueError, match="page image must be"):
        to_grayscale(array)


def test_page_hash_is_sixteen_hex_characters(clean_page: np.ndarray) -> None:
    digest = page_hash(to_grayscale(clean_page))
    assert len(digest) == 16
    int(digest, 16)  # parses


def test_page_hash_is_stable_under_exposure_change(clean_page: np.ndarray) -> None:
    """dHash compares adjacent columns, so a gamma shift must not move it."""
    gray = to_grayscale(clean_page)
    brighter = np.clip(gray.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    assert hamming_distance(page_hash(gray), page_hash(brighter)) <= 2


def test_page_hash_separates_different_pages(clean_page: np.ndarray) -> None:
    other = text_page(glyph_w=30, gap=24, row_pitch=90)
    distance = hamming_distance(
        page_hash(to_grayscale(clean_page)), page_hash(to_grayscale(other))
    )
    assert distance > QualityThresholds().duplicate_hamming_max


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("0000000000000000", "0000000000000000", 0),
        ("0000000000000000", "000000000000000f", 4),
        ("ffffffffffffffff", "0000000000000000", 64),
    ],
)
def test_hamming_distance(left: str, right: str, expected: int) -> None:
    assert hamming_distance(left, right) == expected


def test_hamming_distance_refuses_a_non_hash() -> None:
    with pytest.raises(ValueError, match="16 hex characters"):
        hamming_distance("not-a-hash", "0000000000000000")


# --------------------------------------------------------------------------- #
# Repairs
# --------------------------------------------------------------------------- #


def test_auto_invert_is_exact_and_reversible(clean_page: np.ndarray) -> None:
    """The evidence crop must still be the document that arrived."""
    inverted = auto_invert(clean_page)
    assert np.array_equal(inverted, 255 - clean_page)
    assert np.array_equal(auto_invert(inverted), clean_page)


def test_deskew_expands_the_canvas_and_crops_no_ink(clean_page: np.ndarray) -> None:
    rotated = deskew(clean_page, 7.0)
    assert rotated.shape[0] > clean_page.shape[0]
    assert rotated.shape[1] > clean_page.shape[1]
    # Rotation is area-preserving up to interpolation; no wedge of text is lost.
    before = int((to_grayscale(clean_page) < 128).sum())
    after = int((to_grayscale(rotated) < 128).sum())
    assert after == pytest.approx(before, rel=0.05)


def test_deskew_fills_corners_with_the_page_border_not_white() -> None:
    """A white wedge against grey paper is an edge the detector would follow."""
    grey = text_page(paper=180)
    rotated = deskew(grey, 10.0)
    corner = to_grayscale(rotated)[2, 2]
    assert abs(int(corner) - 180) <= 6


def test_deskew_of_zero_degrees_is_a_no_op(clean_page: np.ndarray) -> None:
    assert np.array_equal(deskew(clean_page, 0.0), clean_page)


@pytest.mark.parametrize("degrees", [-7.0, -2.5, 0.9, 4.25])
def test_deskew_transform_matches_what_deskew_produces(
    clean_page: np.ndarray, degrees: float
) -> None:
    """The published matrix must be the one the pixels actually went through."""
    height, width = clean_page.shape[:2]
    _matrix, (new_width, new_height) = deskew_transform(height, width, degrees)
    rotated = deskew(clean_page, degrees)
    assert (rotated.shape[1], rotated.shape[0]) == (new_width, new_height)


@pytest.mark.parametrize("degrees", [-7.0, -2.5, 0.9, 4.25])
def test_map_bbox_to_original_inverts_the_deskew(degrees: float) -> None:
    """Round trip: forward through the transform, back through the inverse."""
    height, width = 2339, 1654
    matrix, _size = deskew_transform(height, width, degrees)
    original = np.array(
        [[300.0, 400.0], [900.0, 400.0], [900.0, 460.0], [300.0, 460.0]]
    )
    forward = np.hstack([original, np.ones((4, 1))]) @ np.asarray(matrix).T
    back = map_bbox_to_original(
        forward.reshape(-1).tolist(), height=height, width=width, degrees=degrees
    )
    assert np.allclose(np.asarray(back).reshape(-1, 2), original, atol=1e-6)


@pytest.mark.parametrize("degrees", [-6.0, 3.0])
def test_map_bbox_to_original_undoes_the_expansion(degrees: float) -> None:
    """The inverse must remove the canvas growth, not just the rotation.

    ``deskew`` enlarges the canvas to avoid cropping, so a box sitting at the
    origin of the straightened page maps to a *negative* offset on the original
    — the straightened page starts outside it.  A transform that rotated
    correctly but ignored the translation would leave every box shifted by half
    the growth, which on A4 at one degree is about 18 px: small enough to look
    like a near miss and large enough to sit on the wrong form field.
    """
    height, width = 2339, 1654
    _matrix, (new_width, new_height) = deskew_transform(height, width, degrees)
    assert new_width > width and new_height > height

    corner = map_bbox_to_original(
        [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        height=height,
        width=width,
        degrees=degrees,
    )
    points = np.asarray(corner).reshape(-1, 2)
    assert points.min() < 0.0


@pytest.mark.parametrize("angle", [-9.0, -3.5, -1.5, 1.5, 3.5, 9.0])
def test_estimate_skew_recovers_a_known_rotation(
    clean_page: np.ndarray, angle: float
) -> None:
    rotated = deskew(clean_page, angle)
    metrics = measure(rotated)
    # `skew_deg` is the correction to apply, so it undoes the rotation.
    assert metrics.skew_deg == pytest.approx(-angle, abs=0.6)
    assert metrics.skew_confidence > 1.5


def test_estimate_skew_reports_no_confidence_on_an_empty_mask() -> None:
    assert estimate_skew(np.zeros((0, 0), np.uint8)) == (0.0, 0.0)
    assert estimate_skew(np.zeros((400, 400), np.uint8)) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def test_measure_of_a_clean_page(clean_page: np.ndarray) -> None:
    metrics = measure(clean_page)
    assert (metrics.width_px, metrics.height_px) == (PAGE_WIDTH, PAGE_HEIGHT)
    assert metrics.analysis_scale == pytest.approx(
        ANALYSIS_LONG_EDGE_PX / PAGE_HEIGHT, rel=1e-3
    )
    assert metrics.effective_dpi > QualityThresholds().effective_dpi_target
    assert metrics.glyph_count > 100
    assert metrics.contrast > QualityThresholds().contrast_target
    assert metrics.laplacian_variance > QualityThresholds().laplacian_target
    assert metrics.is_blank is False
    assert metrics.is_inverted is False
    assert metrics.skew_deg == pytest.approx(0.0, abs=0.3)
    assert metrics.occlusion_share == pytest.approx(0.0, abs=0.01)
    assert metrics.noise_sigma < QualityThresholds().noise_ceiling
    assert metrics.shadow_index < QualityThresholds().shadow_ceiling


def test_measure_does_not_mutate_its_input(clean_page: np.ndarray) -> None:
    original = clean_page.copy()
    measure(clean_page)
    assert np.array_equal(clean_page, original)


def test_measure_detects_a_blank_page() -> None:
    metrics = measure(blank_page())
    assert metrics.is_blank is True
    assert metrics.glyph_count == 0
    assert metrics.text_coverage == 0.0


def test_measure_detects_an_inverted_page(clean_page: np.ndarray) -> None:
    assert measure(auto_invert(clean_page)).is_inverted is True


def test_measure_reports_orientation_anisotropy_without_acting_on_it(
    clean_page: np.ndarray,
) -> None:
    """Documented negative space: the number is evidence, never a flag."""
    upright = measure(clean_page)
    sideways = measure(np.ascontiguousarray(np.rot90(clean_page)))
    assert upright.orientation_anisotropy >= 0.0
    assert sideways.orientation_anisotropy >= 0.0

    gate = QualityGate()
    for image in (clean_page, np.ascontiguousarray(np.rot90(clean_page))):
        flags = gate.evaluate(image, 1).assessment.quality_flags
        assert QualityFlag.ROTATION_DETECTED not in flags
        assert QualityFlag.ROTATION_REPAIRED not in flags
        gate.reset()


def test_measure_refuses_a_shape_that_is_not_a_page() -> None:
    with pytest.raises(ValueError, match="page image must be"):
        measure(np.zeros((10, 10, 2), np.uint8))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_the_score_is_conjunctive_not_an_average() -> None:
    """One fatal factor must not be carried over the line by four good ones."""
    limits = QualityThresholds()
    scores = QualitySubScores(
        resolution=1.0, sharpness=0.0, contrast=1.0, coverage=1.0, occlusion=1.0
    )
    weighted_arithmetic = (
        1.0 * limits.weight_resolution
        + 0.0 * limits.weight_sharpness
        + 1.0 * limits.weight_contrast
        + 1.0 * limits.weight_coverage
        + 1.0 * limits.weight_occlusion
    )
    geometric = _aggregate(scores, limits)
    assert geometric < limits.bounce_threshold < weighted_arithmetic


def test_the_subscore_floor_preserves_the_ordering_among_bad_pages() -> None:
    limits = QualityThresholds()
    worse = QualitySubScores(
        resolution=0.0, sharpness=0.0, contrast=0.0, coverage=0.0, occlusion=0.0
    )
    bad = QualitySubScores(
        resolution=0.3, sharpness=0.0, contrast=0.2, coverage=1.0, occlusion=1.0
    )
    assert 0.0 < _aggregate(worse, limits) < _aggregate(bad, limits)


def test_all_zero_weights_is_a_configuration_error() -> None:
    limits = QualityThresholds(
        weight_resolution=0.0,
        weight_sharpness=0.0,
        weight_contrast=0.0,
        weight_coverage=0.0,
        weight_occlusion=0.0,
    )
    with pytest.raises(ValueError, match="must not all be zero"):
        _aggregate(QualitySubScores(**dict.fromkeys(QualitySubScores.model_fields, 1.0)), limits)


# --------------------------------------------------------------------------- #
# The gate — accept
# --------------------------------------------------------------------------- #


def test_a_clean_page_is_accepted_unrepaired(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    result = gate.evaluate(clean_page, 1)
    assert isinstance(result, PageQuality)
    assessment = result.assessment
    assert assessment.verdict is QualityVerdict.ACCEPT
    assert assessment.bounce_reason is None
    assert assessment.repairs_applied == ()
    assert assessment.quality_score == pytest.approx(1.0)
    assert assessment.quality_flags == ()
    assert result.image is not None
    assert np.array_equal(result.image, clean_page)


def test_every_page_carries_a_score_and_flags_including_accepted_ones(
    gate: QualityGate,
) -> None:
    """PRD P0-3: a gate that only records refusals cannot be monitored."""
    faint = text_page(ink=170)
    assessment = gate.evaluate(faint, 1).assessment
    assert assessment.verdict is QualityVerdict.ACCEPT
    assert 0.0 < assessment.quality_score < 1.0
    assert QualityFlag.BLUR_ABOVE_CEILING in assessment.quality_flags
    assert assessment.metrics.laplacian_variance > 0.0


def test_page_numbers_are_carried_through(gate: QualityGate) -> None:
    assert gate.evaluate(text_page(), 7).assessment.page_no == 7


def test_flags_from_an_earlier_stage_reach_the_envelope(gate: QualityGate) -> None:
    """Stage 1b's junk verdict has no other way into page quality."""
    assessment = gate.evaluate(
        text_page(), 1, extra_flags=[QualityFlag.JUNK_TEXT_LAYER]
    ).assessment
    assert QualityFlag.JUNK_TEXT_LAYER in assessment.quality_flags


def test_flags_are_deduplicated(gate: QualityGate) -> None:
    """``PageResult`` rejects repeats."""
    assessment = gate.evaluate(
        text_page(ink=170),
        1,
        extra_flags=[QualityFlag.BLUR_ABOVE_CEILING, QualityFlag.BLUR_ABOVE_CEILING],
    ).assessment
    flags = assessment.quality_flags
    assert len(flags) == len(set(flags))


# --------------------------------------------------------------------------- #
# The gate — repair
# --------------------------------------------------------------------------- #


def test_a_skewed_page_is_deskewed_not_bounced(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    result = gate.evaluate(deskew(clean_page, -4.0), 1)
    assessment = result.assessment
    assert assessment.verdict is QualityVerdict.REPAIR
    assert REPAIR_DESKEW in assessment.repairs_applied
    assert QualityFlag.SKEW_DETECTED in assessment.quality_flags
    assert QualityFlag.SKEW_REPAIRED in assessment.quality_flags
    assert result.image is not None
    # The repaired pixels are what the recogniser sees, straightened.
    assert measure(result.image).skew_deg == pytest.approx(0.0, abs=0.7)


def test_skew_below_the_repair_floor_is_left_alone(gate: QualityGate) -> None:
    """Interpolation costs more sharpness than a 0.3 degree rotation recovers."""
    barely = deskew(text_page(), -0.3)
    assessment = gate.evaluate(barely, 1).assessment
    assert assessment.repairs_applied == ()
    assert assessment.verdict is QualityVerdict.ACCEPT


def test_an_inverted_page_is_corrected_and_re_measured(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    """Every other metric measures the wrong polarity until this is fixed."""
    result = gate.evaluate(auto_invert(clean_page), 1)
    assessment = result.assessment
    assert assessment.verdict is QualityVerdict.REPAIR
    assert REPAIR_AUTO_INVERT in assessment.repairs_applied
    assert QualityFlag.INVERTED_REPAIRED in assessment.quality_flags
    # Re-measured: the metrics describe the corrected page, not the dark one.
    assert assessment.metrics.is_inverted is False
    assert assessment.quality_score == pytest.approx(1.0)
    assert result.image is not None
    assert np.array_equal(result.image, clean_page)


def test_the_score_keeps_the_pre_deskew_measurement(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    """A document must not be charged for a defect we introduced repairing it."""
    rotated = deskew(clean_page, -6.0)
    assessment = gate.evaluate(rotated, 1).assessment
    assert assessment.metrics.skew_deg == pytest.approx(6.0, abs=0.6)


# --------------------------------------------------------------------------- #
# The gate — bounce
# --------------------------------------------------------------------------- #


def test_a_blank_page_bounces_without_billing_ocr(gate: QualityGate) -> None:
    result = gate.evaluate(blank_page(), 1)
    assessment = result.assessment
    assert assessment.verdict is QualityVerdict.BOUNCE
    assert assessment.bounce_reason is BounceReason.BLANK_PAGE
    assert QualityFlag.BLANK_PAGE in assessment.quality_flags
    assert result.image is None, "nothing runs after a bounce"


def test_a_page_too_small_to_be_a_document_bounces(gate: QualityGate) -> None:
    tiny = cv2.resize(text_page(), (150, 212), interpolation=cv2.INTER_AREA)
    assessment = gate.evaluate(tiny, 1).assessment
    assert assessment.bounce_reason is BounceReason.RESOLUTION_BELOW_FLOOR
    assert QualityFlag.RESOLUTION_BELOW_FLOOR in assessment.quality_flags
    assert assessment.detail is not None and "150x212" in assessment.detail


def test_a_page_with_nothing_to_recognise_bounces(gate: QualityGate) -> None:
    """Not blank — a stamp, a signature, one stray mark — but no text."""
    page = blank_page()
    cv2.circle(page, (300, 400), 26, (30, 30, 30), -1)
    cv2.circle(page, (900, 1300), 24, (30, 30, 30), -1)
    assessment = gate.evaluate(page, 1).assessment
    assert assessment.verdict is QualityVerdict.BOUNCE
    assert assessment.bounce_reason in (
        BounceReason.TEXT_COVERAGE_BELOW_FLOOR,
        BounceReason.BLANK_PAGE,
    )


def test_an_unrecoverably_skewed_page_bounces_before_it_is_rotated() -> None:
    gate = QualityGate(QualityThresholds(skew_unrecoverable_deg=2.0))
    result = gate.evaluate(deskew(text_page(), -5.0), 1)
    assessment = result.assessment
    assert assessment.bounce_reason is BounceReason.SKEW_UNRECOVERABLE
    assert QualityFlag.SKEW_UNRECOVERABLE in assessment.quality_flags
    assert REPAIR_DESKEW not in assessment.repairs_applied
    assert result.image is None


def test_a_low_contrast_page_bounces_on_the_score(gate: QualityGate) -> None:
    assessment = gate.evaluate(text_page(ink=210), 1).assessment
    assert assessment.verdict is QualityVerdict.BOUNCE
    assert assessment.bounce_reason is BounceReason.QUALITY_SCORE_BELOW_THRESHOLD
    assert assessment.quality_score < QualityThresholds().bounce_threshold
    assert assessment.detail is not None and "below the" in assessment.detail


def test_the_bounce_threshold_is_the_knob_an_operator_moves() -> None:
    page = text_page(ink=170)  # scores ~0.89
    assert QualityGate().evaluate(page, 1).assessment.verdict is QualityVerdict.ACCEPT
    strict = QualityGate(QualityThresholds(bounce_threshold=0.95))
    assert strict.evaluate(page, 1).assessment.verdict is QualityVerdict.BOUNCE


def test_a_bounce_is_a_verdict_not_an_exception(gate: QualityGate) -> None:
    """``app.contracts.errors`` says the same thing from the other side."""
    assessment = gate.evaluate(blank_page(), 1).assessment
    decision = assessment.to_bounce_decision()
    assert decision.decision is QualityVerdict.BOUNCE
    assert decision.reason is BounceReason.BLANK_PAGE
    assert decision.detail == assessment.detail


def test_an_accepted_page_argues_for_no_bounce(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    decision = gate.evaluate(clean_page, 1).assessment.to_bounce_decision()
    assert decision.decision is QualityVerdict.ACCEPT
    assert decision.reason is None


# --------------------------------------------------------------------------- #
# Near-duplicate pages
# --------------------------------------------------------------------------- #


def test_a_repeated_page_is_flagged_not_bounced(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    first = gate.evaluate(clean_page, 1).assessment
    second = gate.evaluate(clean_page.copy(), 2).assessment
    assert QualityFlag.DUPLICATE_PAGE not in first.quality_flags
    assert QualityFlag.DUPLICATE_PAGE in second.quality_flags
    assert second.verdict is QualityVerdict.ACCEPT
    assert second.detail == "duplicate of page 1"


def test_reset_forgets_the_previous_document(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    """Two complainants sending the same bank circular are not duplicates."""
    gate.evaluate(clean_page, 1)
    gate.reset()
    assert (
        QualityFlag.DUPLICATE_PAGE
        not in gate.evaluate(clean_page, 1).assessment.quality_flags
    )


def test_evaluate_document_numbers_pages_and_resets_first(
    gate: QualityGate, clean_page: np.ndarray
) -> None:
    pages = [clean_page, text_page(glyph_w=30, gap=24, row_pitch=90), clean_page]
    results = gate.evaluate_document(pages)
    assert [r.assessment.page_no for r in results] == [1, 2, 3]
    assert QualityFlag.DUPLICATE_PAGE in results[2].assessment.quality_flags
    assert QualityFlag.DUPLICATE_PAGE not in results[1].assessment.quality_flags

    # A second document does not inherit the first one's hashes.
    again = gate.evaluate_document([clean_page])
    assert QualityFlag.DUPLICATE_PAGE not in again[0].assessment.quality_flags


def test_evaluate_document_honours_first_page_no(gate: QualityGate) -> None:
    results = gate.evaluate_document([text_page()], first_page_no=5)
    assert results[0].assessment.page_no == 5


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_thresholds_come_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        QUALITY_BOUNCE_THRESHOLD=0.5,
        QUALITY_BLUR_LAPLACIAN_FLOOR=1000.0,
        QUALITY_BLUR_LAPLACIAN_TARGET=2000.0,
        QUALITY_CONTRAST_FLOOR=5.0,
        QUALITY_CONTRAST_TARGET=50.0,
        QUALITY_EFFECTIVE_DPI_FLOOR=20.0,
        QUALITY_EFFECTIVE_DPI_TARGET=120.0,
        QUALITY_TEXT_COVERAGE_FLOOR=0.01,
        QUALITY_SKEW_REPAIR_DEG=1.5,
        QUALITY_SKEW_BOUNCE_DEG=15.0,
        QUALITY_MIN_PAGE_DIMENSION_PX=300,
    )
    limits = QualityThresholds.from_settings(settings)
    assert limits.bounce_threshold == 0.5
    assert limits.laplacian_floor == 1000.0
    assert limits.laplacian_target == 2000.0
    assert limits.contrast_floor == 5.0
    assert limits.contrast_target == 50.0
    assert limits.effective_dpi_floor == 20.0
    assert limits.effective_dpi_target == 120.0
    assert limits.coverage_floor == 0.01
    assert limits.skew_repair_deg == 1.5
    assert limits.skew_unrecoverable_deg == 15.0
    assert limits.min_page_dimension_px == 300


def test_defaults_match_the_committed_calibration() -> None:
    """The frozen numbers and the settings defaults must not drift apart."""
    from_code = QualityThresholds()
    from_settings = QualityThresholds.from_settings(Settings(_env_file=None))
    assert from_code == from_settings


def test_thresholds_are_frozen() -> None:
    limits = QualityThresholds()
    with pytest.raises(ValueError):
        limits.bounce_threshold = 0.9  # type: ignore[misc]
