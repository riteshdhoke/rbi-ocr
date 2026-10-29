"""Stage 3 — page classification: structural features and the fitted fusion.

Everything here runs with an **empty ``models/``**: the fusion coefficients are
loaded from the copy committed under ``docs/calibration/``, and the CNN — which
has no approved artifact — is represented by a provider stub that raises the
typed absence the real one raises.

The corpus is read for two purposes and only for two: to check that the shipped
weights actually classify the pages they were fitted on, and to check that the
committed training report's claims are true and honestly framed.  **Only
variants 0-2** are touched; 3 and 4 are the locked test split (BUILD-BRIEF M1)
and a stage tuned against them would be measuring itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest

from app.contracts.errors import ProviderUnavailable
from app.contracts.ocr import (
    OCRLine,
    PageClass,
    PageClassLabel,
    StructuralFeatures,
    bbox_from_xyxy,
)
from app.providers.base import ModelArtifactMissing
from app.providers.local import (
    CLASSIFIER_FUSION_FILENAME,
    CLASSIFIER_SUBDIR,
    STRUCTURAL_FEATURE_NAMES,
    FusionWeights,
    late_fusion,
)
from app.settings import Settings
from app.stages.s3_classify import (
    CALIBRATION_FUSION_PATH,
    COLOUR_HISTOGRAM_BINS,
    MIN_CLASS_CONFIDENCE,
    TRAINED_LABELS,
    ClassificationMechanism,
    classify_page,
    colour_histogram,
    detect_table_lines,
    estimate_skew_degrees,
    estimate_text_lines,
    extract_structural_features,
    fusion_search_paths,
    ink_mask,
    load_fusion_weights,
    reset_fusion_cache,
    structural_classify,
    uniform_cnn_scores,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CORPUS: Final[Path] = PROJECT_ROOT / "corpus"
REPORT_PATH: Final[Path] = (
    PROJECT_ROOT / "docs" / "calibration" / "fusion.training-report.json"
)
MODELS_FUSION_PATH: Final[Path] = (
    PROJECT_ROOT / "models" / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME
)

#: Corpus variants this file may read.  3 and 4 are the locked test split.
READABLE_VARIANTS: Final[tuple[int, ...]] = (0, 1, 2)

#: Floors the committed report must clear.  Deliberately well below the numbers
#: it actually reports (0.973 / 0.959): a test that pins the exact value fails
#: on every legitimate retrain, and a floor is what "documented honestly" means
#: operationally — the claim is "at least this", not "exactly this".
MIN_TRAIN_ACCURACY: Final[float] = 0.90
MIN_HOLDOUT_ACCURACY: Final[float] = 0.90


@pytest.fixture(autouse=True)
def _clear_weight_cache() -> Iterator[None]:
    """The module caches coefficients by path; tests must not inherit them."""
    reset_fusion_cache()
    yield
    reset_fusion_cache()


@pytest.fixture(scope="session")
def committed_weights() -> FusionWeights:
    return FusionWeights.model_validate_json(
        CALIBRATION_FUSION_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="session")
def training_report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Image fixtures — synthetic, deterministic, no corpus needed
# --------------------------------------------------------------------------- #


def _cv2() -> Any:
    import cv2

    return cv2


def _typed_page(
    *,
    height: int = 800,
    width: int = 600,
    rows: int = 14,
    rules: bool = False,
) -> np.ndarray:
    """A page of machine-set text, optionally with printed rules around it."""
    cv2 = _cv2()
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in range(rows):
        cv2.putText(
            page,
            "Name of bank and branch office",
            (40, 60 + row * 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if rules:
        for rule in range(6):
            y = 40 + rule * 120
            cv2.line(page, (20, y), (width - 20, y), (0, 0, 0), 2)
        cv2.line(page, (20, 40), (20, height - 40), (0, 0, 0), 2)
    return page


def _blank_page(height: int = 800, width: int = 600) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _shadowed(page: np.ndarray, *, factor: float = 0.45) -> np.ndarray:
    """The corpus's ``photo``/``heavy`` failure mode: half the page in shade."""
    shaded = page.astype(np.float32).copy()
    shaded[:, : page.shape[1] // 2] *= factor
    return shaded.astype(np.uint8)


def _rotated(page: np.ndarray, degrees: float) -> np.ndarray:
    cv2 = _cv2()
    height, width = page.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        page, matrix, (width, height), borderValue=(255, 255, 255)
    )


def _uniform_weights(labels: tuple[PageClassLabel, ...]) -> FusionWeights:
    """Weights that give every label the same score — a deliberate coin flip."""
    length = len(labels) + len(STRUCTURAL_FEATURE_NAMES)
    return FusionWeights(
        labels=labels,
        coefficients={label.value: tuple([0.0] * length) for label in labels},
        intercepts={label.value: 0.0 for label in labels},
    )


def _features(**overrides: Any) -> StructuralFeatures:
    data: dict[str, Any] = {
        "line_count": 20,
        "ink_density": 0.05,
        "aspect_ratio": 0.75,
        "has_table_lines": True,
        "table_line_count": 6,
        "colour_histogram": [0.25, 0.25, 0.25, 0.25] * 3,
        "handwriting_ratio": 0.1,
        "text_coverage": 0.08,
    }
    data.update(overrides)
    # The fixture histogram is uniform across 12 bins; renormalise so the
    # contract's distribution check holds whatever the caller overrode.
    total = sum(data["colour_histogram"])
    data["colour_histogram"] = [value / total for value in data["colour_histogram"]]
    return StructuralFeatures(**data)


# --------------------------------------------------------------------------- #
# Structural features (BUILD-BRIEF M4)
# --------------------------------------------------------------------------- #


def test_printed_rules_are_detected_and_counted() -> None:
    has_rules, count = detect_table_lines(_typed_page(rules=True))
    assert has_rules is True
    assert count >= 6, "six horizontal rules and a vertical one were drawn"


def test_a_page_of_prose_has_no_rules() -> None:
    assert detect_table_lines(_typed_page(rules=False)) == (False, 0)


def test_a_blank_page_has_neither_rules_nor_lines() -> None:
    blank = _blank_page()
    assert detect_table_lines(blank) == (False, 0)
    assert estimate_text_lines(blank) == (0, 0.0)


def test_text_lines_are_estimated_from_pixels_before_recognition_runs() -> None:
    """Stage 3 runs before Stage 4 in the PRD flow; it cannot assume lines."""
    count, coverage = estimate_text_lines(_typed_page(rows=14, rules=False))
    assert count == pytest.approx(14, abs=2)
    assert 0.0 < coverage < 0.5


def test_rules_are_removed_before_lines_are_counted() -> None:
    """A form's box borders must not be counted as forty text lines."""
    plain, _ = estimate_text_lines(_typed_page(rows=10, rules=False))
    ruled, _ = estimate_text_lines(_typed_page(rows=10, rules=True))
    assert plain <= ruled <= plain + 40


def test_illumination_is_flattened_so_ink_density_measures_ink_not_lighting() -> None:
    """The documented bug: raw Otsu reads a shadowed half-page as solid ink."""
    page = _typed_page(rules=False)
    shadowed = _shadowed(page)

    clean_density = float(ink_mask(page).mean())
    shadowed_density = float(ink_mask(shadowed).mean())
    raw_clean = float(ink_mask(page, flatten=False).mean())
    raw_shadowed = float(ink_mask(shadowed, flatten=False).mean())

    assert shadowed_density == pytest.approx(clean_density, abs=0.02)
    # ... and this is what it would have been without the fix.
    assert raw_shadowed > 5 * raw_clean


def test_the_features_of_a_shadowed_page_still_describe_the_page() -> None:
    page = _typed_page(rules=False)
    clean = extract_structural_features(page)
    shadowed = extract_structural_features(_shadowed(page))

    assert shadowed.line_count == clean.line_count
    assert shadowed.ink_density == pytest.approx(clean.ink_density, abs=0.02)


def test_skew_is_estimated_from_the_text_itself() -> None:
    page = _typed_page(rules=False)
    assert estimate_skew_degrees(ink_mask(page)) == pytest.approx(0.0, abs=0.5)
    assert estimate_skew_degrees(ink_mask(_rotated(page, 3.0))) == pytest.approx(
        -3.0, abs=1.0
    )


def test_rules_survive_a_skewed_scan_because_the_mask_is_deskewed_first() -> None:
    """A strictly horizontal detector loses every rule at 1.5 degrees."""
    ruled = _typed_page(rules=True)
    upright = detect_table_lines(ruled)[1]
    skewed = detect_table_lines(_rotated(ruled, 3.0))[1]
    assert upright > 0
    assert skewed >= upright - 1


def test_the_colour_histogram_is_a_coarse_normalised_distribution() -> None:
    histogram = colour_histogram(_typed_page())
    assert len(histogram) == 3 * COLOUR_HISTOGRAM_BINS
    assert sum(histogram) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in histogram)


def test_a_zero_bin_histogram_is_refused() -> None:
    with pytest.raises(ValueError, match="bins must be >= 1"):
        colour_histogram(_typed_page(), bins=0)


def test_aspect_ratio_describes_the_page_not_its_text_block() -> None:
    features = extract_structural_features(_typed_page(height=800, width=600))
    assert features.aspect_ratio == pytest.approx(0.75)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_grayscale_bgr_and_bgra_pages_are_all_accepted(channels: int) -> None:
    page = _typed_page()
    if channels == 1:
        page = page[:, :, 0]
    elif channels == 4:
        page = np.dstack([page, np.full(page.shape[:2], 255, dtype=np.uint8)])
    features = extract_structural_features(page)
    assert features.line_count > 0


def test_a_one_dimensional_image_is_refused() -> None:
    with pytest.raises(ValueError, match="must be 2-D or 3-D"):
        extract_structural_features(np.zeros((10,), dtype=np.uint8))


def test_supplying_recognised_lines_replaces_the_pixel_estimates() -> None:
    """Documented caveat: the committed weights were fitted *without* lines."""
    page = _typed_page(rows=14, rules=False)
    lines = [
        OCRLine(
            text="Name of bank",
            bbox=bbox_from_xyxy(40.0, 40.0 + 45 * index, 500.0, 70.0 + 45 * index),
            confidence=0.9,
            script="Latn",
            language=None,
            is_handwritten=index % 2 == 0,
            line_index=index,
        )
        for index in range(4)
    ]
    with_lines = extract_structural_features(page, lines=lines)
    without = extract_structural_features(page)

    assert with_lines.line_count == 4
    assert with_lines.handwriting_ratio == pytest.approx(0.5)
    assert without.line_count != 4


# --------------------------------------------------------------------------- #
# Loading the fitted coefficients
# --------------------------------------------------------------------------- #


def test_an_air_gapped_node_with_an_empty_models_dir_still_classifies(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(MODELS_DIR=tmp_path)
    weights = load_fusion_weights(settings)
    assert weights.labels == TRAINED_LABELS


def test_the_models_directory_wins_over_the_committed_copy(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """A deployment that retrained must not be overridden by the source copy."""
    settings = make_settings(MODELS_DIR=tmp_path)
    assert fusion_search_paths(settings) == (
        tmp_path / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME,
        CALIBRATION_FUSION_PATH,
    )

    retrained = _uniform_weights(TRAINED_LABELS)
    target = tmp_path / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME
    target.parent.mkdir(parents=True)
    target.write_text(retrained.model_dump_json(), encoding="utf-8")

    assert load_fusion_weights(settings).intercepts == retrained.intercepts


def test_a_corrupt_coefficient_file_fails_loudly_instead_of_falling_through(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """Quietly classifying with *different* weights is the worst outcome."""
    settings = make_settings(MODELS_DIR=tmp_path)
    target = tmp_path / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME
    target.parent.mkdir(parents=True)
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable or"):
        load_fusion_weights(settings)


def test_no_coefficients_anywhere_names_the_tool_that_makes_them(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelArtifactMissing) as excinfo:
        load_fusion_weights(path=tmp_path / "absent.json")
    assert "train_page_classifier.py" in excinfo.value.how_to_obtain


def test_coefficients_are_cached_by_path_and_the_cache_can_be_dropped(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(MODELS_DIR=tmp_path)
    first = load_fusion_weights(settings)
    assert load_fusion_weights(settings) is first
    reset_fusion_cache()
    assert load_fusion_weights(settings) is not first


def test_the_committed_copy_and_the_model_bundle_copy_agree() -> None:
    """Two copies of a coefficient file that disagree is a silent A/B test."""
    if not MODELS_FUSION_PATH.is_file():
        pytest.skip(
            "models/classifier/fusion.json is absent (empty models/ is a "
            "supported state); the committed docs/calibration copy is the one "
            "the code falls back to and is exercised by the other tests"
        )
    assert json.loads(
        MODELS_FUSION_PATH.read_text(encoding="utf-8")
    ) == json.loads(CALIBRATION_FUSION_PATH.read_text(encoding="utf-8"))


def test_the_feature_order_the_coefficients_were_fitted_in_is_pinned() -> None:
    """``fusion.json`` stores coefficients positionally."""
    assert tuple(STRUCTURAL_FEATURE_NAMES) == (
        "line_count_log",
        "ink_density",
        "aspect_ratio",
        "has_table_lines",
        "table_line_count_log",
        "handwriting_ratio",
        "text_coverage",
    )


def test_the_coefficient_vectors_have_the_length_the_labels_imply(
    committed_weights: FusionWeights,
) -> None:
    expected = len(committed_weights.labels) + len(STRUCTURAL_FEATURE_NAMES)
    for label in committed_weights.labels:
        assert len(committed_weights.coefficients[label.value]) == expected


# --------------------------------------------------------------------------- #
# The CNN slots are provably inert
# --------------------------------------------------------------------------- #


def test_the_committed_cnn_coefficients_are_exactly_zero(
    committed_weights: FusionWeights,
) -> None:
    width = len(committed_weights.labels)
    for label in committed_weights.labels:
        cnn_slots = committed_weights.coefficients[label.value][:width]
        assert cnn_slots == tuple([0.0] * width), (
            f"{label.value} would let a CNN that does not exist take credit "
            "for a structural result"
        )


def test_no_cnn_score_can_change_todays_structural_decision(
    committed_weights: FusionWeights,
) -> None:
    features = _features()
    uniform = late_fusion(
        uniform_cnn_scores(committed_weights), features, committed_weights
    )
    opinionated = late_fusion(
        [0.97, 0.01, 0.01, 0.01], features, committed_weights
    )
    assert opinionated.scores == pytest.approx(uniform.scores)
    assert opinionated.label is uniform.label


def test_the_neutral_prior_is_a_distribution(
    committed_weights: FusionWeights,
) -> None:
    prior = uniform_cnn_scores(committed_weights)
    assert len(prior) == len(committed_weights.labels)
    assert sum(prior) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The stage: CNN preferred, structure as the fitted fallback
# --------------------------------------------------------------------------- #


class _CNNProvider:
    """Provider stub for ``classify_page`` only."""

    def __init__(self, result: PageClass | Exception) -> None:
        self.result = result
        self.calls = 0

    def classify_page(
        self, page: Any, feats: StructuralFeatures
    ) -> PageClass:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _artifact_missing() -> ModelArtifactMissing:
    return ModelArtifactMissing(
        artifact="page-classifier CNN export (page_classifier.onnx)",
        path=Path("models/classifier/page_classifier.onnx"),
        how_to_obtain="No approved MobileNetV3-Small export ships in Phase 0.",
    )


def test_an_absent_cnn_falls_back_to_the_fitted_structural_path(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    provider = _CNNProvider(_artifact_missing())
    result = classify_page(
        _typed_page(rules=True),
        provider=provider,
        settings=make_settings(MODELS_DIR=tmp_path),
    )
    assert provider.calls == 1
    assert result.mechanism is ClassificationMechanism.STRUCTURAL_FUSION
    assert result.artifact_fallback_reason == "MODEL_ARTIFACT_MISSING"
    assert result.label in set(TRAINED_LABELS) | {PageClassLabel.UNKNOWN}


def test_a_broken_cnn_is_not_the_same_fact_as_an_absent_one(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """"The model is broken" has no fallback; "the model is absent" does."""
    provider = _CNNProvider(ProviderUnavailable("onnxruntime session died"))
    with pytest.raises(ProviderUnavailable):
        classify_page(
            _typed_page(),
            provider=provider,
            settings=make_settings(MODELS_DIR=tmp_path),
        )


def test_a_working_cnn_is_used_and_named(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    fused = PageClass(
        label=PageClassLabel.COMPLAINT_LETTER,
        confidence=0.88,
        scores={"complaint_letter": 0.88, "complaint_form": 0.12},
    )
    result = classify_page(
        _typed_page(),
        provider=_CNNProvider(fused),
        settings=make_settings(MODELS_DIR=tmp_path),
    )
    assert result.mechanism is ClassificationMechanism.CNN_FUSION
    assert result.label is PageClassLabel.COMPLAINT_LETTER
    assert result.artifact_fallback_reason is None


def test_use_cnn_false_never_touches_the_provider(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    provider = _CNNProvider(_artifact_missing())
    classify_page(
        _typed_page(),
        provider=provider,
        use_cnn=False,
        settings=make_settings(MODELS_DIR=tmp_path),
    )
    assert provider.calls == 0


def test_precomputed_features_are_not_recomputed() -> None:
    """A page that cannot be featurised proves the features were reused."""
    result = classify_page(
        np.zeros((0,), dtype=np.uint8),
        feats=_features(),
        weights=_uniform_weights(TRAINED_LABELS),
        use_cnn=False,
    )
    assert result.features == _features()


def test_a_coin_flip_is_demoted_to_unknown_and_says_so() -> None:
    result = classify_page(
        np.zeros((0,), dtype=np.uint8),
        feats=_features(),
        weights=_uniform_weights(TRAINED_LABELS),
        use_cnn=False,
    )
    assert result.below_confidence_floor is True
    assert result.label is PageClassLabel.UNKNOWN
    assert result.confidence < MIN_CLASS_CONFIDENCE


def test_the_demoted_scores_still_show_what_the_page_nearly_was() -> None:
    result = classify_page(
        np.zeros((0,), dtype=np.uint8),
        feats=_features(),
        weights=_uniform_weights(TRAINED_LABELS),
        use_cnn=False,
    )
    scores = result.page_class.scores
    assert sum(scores.values()) == pytest.approx(1.0)
    assert PageClassLabel.UNKNOWN.value in scores
    for label in TRAINED_LABELS:
        assert label.value in scores, "a reviewer needs the trained scores"


def test_unknown_is_a_refusal_not_a_trainable_class() -> None:
    assert PageClassLabel.UNKNOWN not in TRAINED_LABELS


def test_a_confident_page_is_not_demoted(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    result = classify_page(
        _typed_page(rules=True),
        settings=make_settings(MODELS_DIR=tmp_path),
        min_confidence=0.0,
    )
    assert result.below_confidence_floor is False
    assert result.label is not PageClassLabel.UNKNOWN


def test_structural_classify_is_pure_and_needs_no_image(
    committed_weights: FusionWeights,
) -> None:
    features = _features()
    first = structural_classify(features, committed_weights)
    second = structural_classify(features, committed_weights)
    assert first == second


# --------------------------------------------------------------------------- #
# The committed training report — what it claims, and whether it is true
# --------------------------------------------------------------------------- #


def test_the_report_states_the_split_it_trained_on(
    training_report: dict[str, Any],
) -> None:
    assert training_report["train_variants"] == [0, 1, 2]
    assert training_report["excluded_profiles"] == ["bounce"]
    assert training_report["generator"] == "tools/train_page_classifier.py"


def test_the_locked_test_split_exists_and_was_not_trained_on(
    training_report: dict[str, Any],
) -> None:
    """BUILD-BRIEF M1: variants 3 and 4 are held out and stay held out."""
    index = json.loads((CORPUS / "index.json").read_text(encoding="utf-8"))
    eligible = [
        entry
        for entry in index["documents"]
        if entry["variant"] in READABLE_VARIANTS and entry["profile"] != "bounce"
    ]
    locked = [entry for entry in index["documents"] if entry["variant"] in (3, 4)]

    assert locked, "the locked split must exist to be locked"
    assert training_report["sample_count"] == len(eligible)
    assert training_report["sample_count"] < len(index["documents"])


def test_the_reported_accuracies_clear_the_documented_floors(
    training_report: dict[str, Any],
) -> None:
    assert training_report["train_accuracy"] >= MIN_TRAIN_ACCURACY
    assert training_report["holdout_accuracy_variant_2"] >= MIN_HOLDOUT_ACCURACY
    assert training_report["leave_one_variant_out_mean"] >= MIN_HOLDOUT_ACCURACY


def test_the_accuracy_is_reported_against_a_baseline_it_actually_beats(
    training_report: dict[str, Any],
) -> None:
    """An accuracy without its majority-class baseline is not a claim."""
    baseline = training_report["majority_class_baseline"]
    assert 0.0 < baseline < 1.0
    assert training_report["holdout_accuracy_variant_2"] > baseline + 0.2


def test_the_holdout_is_named_for_what_it_is(
    training_report: dict[str, Any],
) -> None:
    """The honesty that matters: the holdout is a *variant*, not a layout.

    Variants differ in bank, amounts and complainant — not in document design —
    so the number is an optimistic estimate of generalisation, and the report
    must keep saying which fold produced it rather than presenting a single
    unqualified "accuracy".
    """
    folds = training_report["leave_one_variant_out"]
    assert set(folds) == {"variant_0", "variant_1", "variant_2"}
    assert training_report["holdout_accuracy_variant_2"] == pytest.approx(
        folds["variant_2"]
    )
    assert training_report["leave_one_variant_out_mean"] == pytest.approx(
        sum(folds.values()) / len(folds)
    )
    # Fitting and scoring on the same pages cannot be the harder number.
    assert (
        training_report["train_accuracy"]
        >= training_report["leave_one_variant_out_mean"] - 1e-9
    )


def test_the_report_admits_the_class_it_has_no_examples_of(
    training_report: dict[str, Any],
) -> None:
    """``photo_evidence`` is a label of the contract, not of the corpus.

    Every corpus ``photo_evidence`` page is ground-truthed ``supporting_doc``,
    so the shipped classifier has never seen the class and cannot predict it.
    The report says so out loud; this test keeps it saying so.
    """
    assert training_report["labels_without_training_examples"] == [
        "photo_evidence"
    ]
    assert "photo_evidence" not in training_report["labels_observed_in_corpus"]
    assert "photo_evidence" in training_report["labels"]


def test_the_report_records_that_the_cnn_slots_were_fitted_inert(
    training_report: dict[str, Any],
) -> None:
    assert "zero" in str(training_report["cnn_coefficients"]).lower()
    assert training_report["feature_names"] == list(STRUCTURAL_FEATURE_NAMES)


def test_the_per_type_breakdown_localises_the_errors(
    training_report: dict[str, Any],
) -> None:
    """A single accuracy hides which document type is broken."""
    per_type = training_report["per_type_accuracy"]
    assert len(per_type) == 10, "one entry per corpus document type"
    assert all(0.0 <= value <= 1.0 for value in per_type.values())
    assert min(per_type.values()) >= 0.75


# --------------------------------------------------------------------------- #
# The shipped weights, against the corpus pages they were fitted on
# --------------------------------------------------------------------------- #


def _sample_documents() -> list[dict[str, Any]]:
    """One page per corpus type, variant 0, cheapest available profile.

    Deterministic and small: this is a smoke test that the *committed artifact*
    classifies, not a re-run of the training report (that is
    ``tools/train_page_classifier.py --dry-run``, ~34 s).
    """
    index = json.loads((CORPUS / "index.json").read_text(encoding="utf-8"))
    by_type: dict[str, dict[str, Any]] = {}
    for entry in index["documents"]:
        if entry["variant"] != 0 or entry["profile"] == "bounce":
            continue
        current = by_type.get(entry["type"])
        if current is None or (
            current["profile"] != "clean" and entry["profile"] == "clean"
        ):
            by_type[entry["type"]] = entry
    return [by_type[key] for key in sorted(by_type)]


@pytest.fixture(scope="module")
def corpus_predictions() -> list[tuple[str, str, PageClass]]:
    """``(doc_type, ground-truth class, prediction)`` for the sampled pages.

    Module-scoped: featurising a full-page corpus image costs ~0.25 s and the
    assertions below want the same ten pages.
    """
    import cv2

    weights = FusionWeights.model_validate_json(
        CALIBRATION_FUSION_PATH.read_text(encoding="utf-8")
    )
    rows: list[tuple[str, str, PageClass]] = []
    for entry in _sample_documents():
        assert entry["variant"] in READABLE_VARIANTS
        ground_truth = json.loads(
            (CORPUS / entry["gt"]).read_text(encoding="utf-8")
        )
        image = cv2.imread(str(CORPUS / entry["image"]), cv2.IMREAD_COLOR)
        assert image is not None, f"unreadable corpus image {entry['image']}"
        rows.append(
            (
                str(entry["type"]),
                str(ground_truth["document_class"]),
                structural_classify(
                    extract_structural_features(image), weights
                ),
            )
        )
    return rows


def test_the_committed_weights_classify_the_pages_they_were_fitted_on(
    corpus_predictions: list[tuple[str, str, PageClass]],
) -> None:
    assert len(corpus_predictions) == 10, "one page per corpus document type"
    misses = [
        (doc_type, truth, prediction.label.value)
        for doc_type, truth, prediction in corpus_predictions
        if prediction.label.value != truth
    ]
    accuracy = 1.0 - len(misses) / len(corpus_predictions)
    assert accuracy >= 0.9, f"structural accuracy {accuracy:.2f}, misses={misses}"


def test_the_shipped_classifier_cannot_return_a_class_it_never_saw(
    corpus_predictions: list[tuple[str, str, PageClass]],
) -> None:
    """The practical consequence of ``labels_without_training_examples``."""
    for _doc_type, _truth, prediction in corpus_predictions:
        assert prediction.label is not PageClassLabel.PHOTO_EVIDENCE
