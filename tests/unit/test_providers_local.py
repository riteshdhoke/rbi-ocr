"""Unit tests for :mod:`app.providers.local`.

Everything here runs with **no weight files**: the pure geometry, decoding,
bucketing and label-constraint functions are exercised directly, the loaders
are exercised through the typed error they must raise, and the recognition
path is exercised end to end against injected fake ONNX sessions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.contracts.ocr import PageClassLabel, StructuralFeatures
from app.providers.base import ModelArtifactMissing, ProviderMethodNotSupported
from app.providers.local import (
    REC_IMAGE_HEIGHT,
    REC_WIDTH_BUCKETS,
    SCRIPT_LANGUAGES,
    FusionWeights,
    LocalProvider,
    RecHeadCache,
    bucket_width,
    constrain_language_candidates,
    crop_perspective,
    ctc_greedy_decode,
    db_postprocess,
    detect_script_from_text,
    late_fusion,
    normalise_fasttext_label,
    order_box_clockwise,
    pad_crops_to_batch,
    plan_recognition_batches,
    script_of_char,
    sort_boxes_reading_order,
    structural_feature_vector,
)
from app.settings import Settings

# --------------------------------------------------------------------------- #
# Bucketing — PLAN.md invariant 4
# --------------------------------------------------------------------------- #


def test_bucket_width_scales_to_the_fixed_rec_height() -> None:
    """A 96x300 crop halves to 48x150, which lands in the 160 bucket."""
    assert bucket_width(96, 300) == 160


def test_bucket_width_picks_the_first_edge_that_fits() -> None:
    assert bucket_width(REC_IMAGE_HEIGHT, 79) == 80
    assert bucket_width(REC_IMAGE_HEIGHT, 81) == 160


def test_bucket_width_clamps_over_long_crops() -> None:
    """PaddleOCR squeezes rather than growing the tensor without bound."""
    assert bucket_width(REC_IMAGE_HEIGHT, 100_000) == REC_WIDTH_BUCKETS[-1]


@pytest.mark.parametrize(("h", "w"), [(0, 10), (10, 0), (-1, 5)])
def test_bucket_width_rejects_degenerate_crops(h: int, w: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        bucket_width(h, w)


def test_bucket_width_rejects_unsorted_buckets() -> None:
    with pytest.raises(ValueError, match="ascending"):
        bucket_width(48, 100, buckets=(320, 80))


def test_plan_covers_every_crop_exactly_once() -> None:
    shapes = [(48, w) for w in (30, 300, 60, 500, 90, 1200, 45)]
    plan = plan_recognition_batches(shapes, batch_size=3)
    covered = [index for batch in plan for index in batch.indices]
    assert sorted(covered) == list(range(len(shapes)))
    assert len(covered) == len(set(covered))


def test_every_batch_shares_one_target_width() -> None:
    shapes = [(48, w) for w in (30, 300, 60, 500, 90, 1200, 45)]
    for batch in plan_recognition_batches(shapes, batch_size=8):
        widths = {bucket_width(*shapes[index]) for index in batch.indices}
        assert widths == {batch.target_width}


def test_no_batch_exceeds_the_configured_size() -> None:
    shapes = [(48, 100)] * 50
    plan = plan_recognition_batches(shapes, batch_size=16)
    assert all(len(batch.indices) <= 16 for batch in plan)
    assert sum(len(batch.indices) for batch in plan) == 50


def test_a_full_bucket_splits_into_ceil_batches() -> None:
    plan = plan_recognition_batches([(48, 100)] * 33, batch_size=16)
    assert [len(batch.indices) for batch in plan] == [16, 16, 1]


def test_plan_is_deterministic() -> None:
    """A run must be reproducible; dict ordering may not leak into batching."""
    shapes = [(48, w) for w in (500, 30, 300, 60, 1200, 45, 90)]
    first = plan_recognition_batches(shapes, batch_size=2)
    second = plan_recognition_batches(shapes, batch_size=2)
    assert first == second
    assert [batch.target_width for batch in first] == sorted(
        batch.target_width for batch in first
    )


def test_within_a_bucket_input_order_is_preserved() -> None:
    shapes = [(48, 30), (48, 300), (48, 40), (48, 50)]
    plan = plan_recognition_batches(shapes, batch_size=8)
    small = next(b for b in plan if b.target_width == 80)
    assert small.indices == (0, 2, 3)


def test_empty_input_plans_nothing() -> None:
    assert plan_recognition_batches([], batch_size=16) == ()


def test_batch_size_zero_is_refused_and_says_why() -> None:
    """Invariant 4: a looped path belongs only in the benchmark tool."""
    with pytest.raises(ValueError, match="invariant 4"):
        plan_recognition_batches([(48, 100)], batch_size=0)


def test_padding_produces_the_declared_tensor_shape() -> None:
    crops = [
        np.full((30, 60, 3), 255, dtype=np.uint8),
        np.full((48, 200, 3), 128, dtype=np.uint8),
    ]
    plan = plan_recognition_batches(
        [(30, 60), (48, 200)], batch_size=8
    )
    assert len(plan) == 2
    for batch in plan:
        tensor = pad_crops_to_batch(crops, batch)
        assert tensor.shape == (
            len(batch.indices),
            3,
            REC_IMAGE_HEIGHT,
            batch.target_width,
        )
        assert tensor.dtype == np.float32
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_padding_leaves_the_tail_at_the_pad_value() -> None:
    crops = [np.full((48, 40, 3), 255, dtype=np.uint8)]
    batch = plan_recognition_batches([(48, 40)], batch_size=1)[0]
    tensor = pad_crops_to_batch(crops, batch)
    assert np.allclose(tensor[0, :, :, 60:], 0.0)


def test_padding_accepts_grayscale_crops() -> None:
    crops = [np.full((48, 40), 200, dtype=np.uint8)]
    batch = plan_recognition_batches([(48, 40)], batch_size=1)[0]
    assert pad_crops_to_batch(crops, batch).shape[1] == 3


# --------------------------------------------------------------------------- #
# CTC decoding
# --------------------------------------------------------------------------- #


def _probs(sequence: list[int], classes: int) -> np.ndarray:
    array = np.full((1, len(sequence), classes), 0.01, dtype=np.float32)
    for step, class_id in enumerate(sequence):
        array[0, step] = 0.01
        array[0, step, class_id] = 0.9
    return array


def test_ctc_collapses_repeats_and_drops_blanks() -> None:
    charset = ["a", "b", "c"]
    decoded = ctc_greedy_decode(_probs([1, 1, 0, 1, 2], 4), charset)
    assert decoded[0][0] == "aab"


def test_ctc_confidence_is_the_mean_over_kept_steps() -> None:
    decoded = ctc_greedy_decode(_probs([1, 2], 4), ["a", "b", "c"])
    assert decoded[0][1] == pytest.approx(0.9, abs=1e-5)


def test_an_all_blank_row_decodes_to_empty_with_zero_confidence() -> None:
    decoded = ctc_greedy_decode(_probs([0, 0, 0], 4), ["a", "b", "c"])
    assert decoded[0] == ("", 0.0)


def test_a_dictionary_mismatch_is_loud() -> None:
    """The commonest way a rec head produces confident nonsense."""
    with pytest.raises(ValueError, match="do not match"):
        ctc_greedy_decode(_probs([1], 4), ["a", "b", "c", "d", "e"])


def test_ctc_rejects_the_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\(N, T, C\)"):
        ctc_greedy_decode(np.zeros((3, 4), dtype=np.float32), ["a", "b", "c"])


def test_ctc_decodes_a_whole_batch() -> None:
    array = np.concatenate([_probs([1, 2], 4), _probs([3, 3], 4)], axis=0)
    assert [text for text, _ in ctc_greedy_decode(array, ["a", "b", "c"])] == [
        "ab",
        "c",
    ]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_corners_are_ordered_clockwise_from_top_left() -> None:
    scrambled = np.array(
        [[10, 60], [50, 20], [10, 20], [50, 60]], dtype=np.float32
    )
    assert order_box_clockwise(scrambled) == [
        10.0,
        20.0,
        50.0,
        20.0,
        50.0,
        60.0,
        10.0,
        60.0,
    ]


def test_order_box_rejects_the_wrong_corner_count() -> None:
    with pytest.raises(ValueError, match="4 corners"):
        order_box_clockwise(np.zeros((3, 2), dtype=np.float32))


def test_db_postprocess_finds_a_synthetic_text_block() -> None:
    prob = np.zeros((80, 120), dtype=np.float32)
    prob[20:40, 30:100] = 0.95
    boxes = db_postprocess(prob)
    assert len(boxes) == 1
    assert len(boxes[0]) == 8
    xs = boxes[0][0::2]
    ys = boxes[0][1::2]
    assert min(xs) < 30 and max(xs) > 99
    assert min(ys) < 20 and max(ys) > 39


def test_db_postprocess_drops_low_scoring_regions() -> None:
    prob = np.zeros((80, 120), dtype=np.float32)
    prob[20:40, 30:100] = 0.35
    assert db_postprocess(prob, binary_threshold=0.3, box_threshold=0.6) == []


def test_db_postprocess_scales_back_to_page_coordinates() -> None:
    prob = np.zeros((80, 120), dtype=np.float32)
    prob[20:40, 30:100] = 0.95
    plain = db_postprocess(prob)[0]
    scaled = db_postprocess(prob, ratio_h=2.0, ratio_w=3.0)[0]
    assert scaled[0] == pytest.approx(plain[0] * 3.0)
    assert scaled[1] == pytest.approx(plain[1] * 2.0)


def test_db_postprocess_rejects_a_non_2d_map() -> None:
    with pytest.raises(ValueError, match="2-D"):
        db_postprocess(np.zeros((1, 8, 8), dtype=np.float32))


def test_crop_perspective_rectifies_the_requested_quad() -> None:
    page = np.zeros((100, 200, 3), dtype=np.uint8)
    page[20:50, 40:140] = 255
    crop = crop_perspective(page, [40, 20, 140, 20, 140, 50, 40, 50])
    assert crop.shape[0] == 30
    assert crop.shape[1] == 100
    assert crop.mean() > 200


def test_crop_perspective_rejects_a_malformed_bbox() -> None:
    page = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="8 floats"):
        crop_perspective(page, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="non-finite"):
        crop_perspective(page, [0, 0, 1, 0, 1, float("nan"), 0, 1])


def test_reading_order_is_rows_then_columns() -> None:
    """A two-column label/value form must not interleave."""
    boxes = [
        [400, 10, 600, 10, 600, 40, 400, 40],
        [10, 10, 200, 10, 200, 40, 10, 40],
        [10, 100, 200, 100, 200, 130, 10, 130],
    ]
    assert sort_boxes_reading_order(boxes) == [1, 0, 2]


def test_reading_order_tolerance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        sort_boxes_reading_order([], line_tolerance=0)


# --------------------------------------------------------------------------- #
# Script from text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("character", "expected"),
    [("A", "Latn"), ("z", "Latn"), ("क", "Deva"), ("1", None), (" ", None)],
)
def test_script_of_char(character: str, expected: str | None) -> None:
    assert script_of_char(character) == expected


def test_script_of_char_wants_exactly_one_character() -> None:
    with pytest.raises(ValueError, match="single character"):
        script_of_char("ab")


def test_devanagari_page_votes_devanagari() -> None:
    result = detect_script_from_text(["बँक", "खाते", "12345"])
    assert result.script == "Deva"
    assert result.distribution["Deva"] == pytest.approx(1.0)


def test_a_mixed_page_reports_both_scripts() -> None:
    result = detect_script_from_text(["Bank of Baroda", "बँक"])
    assert result.script == "Latn"
    assert set(result.distribution) == {"Latn", "Deva"}
    assert sum(result.distribution.values()) == pytest.approx(1.0)


def test_a_digits_only_page_is_undetermined_not_guessed() -> None:
    result = detect_script_from_text(["12345", "  ", "--"])
    assert result.script == "Zyyy"
    assert result.confidence == 0.0
    assert result.distribution == {}


# --------------------------------------------------------------------------- #
# LID label constraint — PRD P0-5
# --------------------------------------------------------------------------- #


def test_devanagari_never_resolves_to_english() -> None:
    """The whole point: a Deva line is not English however sure the model is."""
    raw = [("__label__en", 0.92), ("__label__mr", 0.05), ("__label__hi", 0.03)]
    result = constrain_language_candidates(raw, "Deva")
    assert result.language == "mar"
    assert all(c.language != "eng" for c in result.candidates)
    assert result.script == "Deva"


def test_latin_constrains_to_english() -> None:
    raw = [("__label__hi", 0.6), ("__label__en", 0.4)]
    assert constrain_language_candidates(raw, "Latn").language == "eng"


def test_no_surviving_candidate_yields_none_never_a_guess() -> None:
    raw = [("__label__zh", 0.99)]
    result = constrain_language_candidates(raw, "Deva")
    assert result.language is None
    assert result.confidence == 0.0
    assert result.candidates == []


def test_candidate_confidences_are_renormalised_over_survivors() -> None:
    raw = [("__label__en", 0.8), ("__label__hi", 0.1), ("__label__mr", 0.1)]
    result = constrain_language_candidates(raw, "Deva")
    assert sum(c.confidence for c in result.candidates) == pytest.approx(1.0)


def test_an_unknown_script_leaves_the_label_set_open() -> None:
    result = constrain_language_candidates([("__label__en", 0.9)], "Zyyy")
    assert result.language == "eng"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("__label__hi", "hin"),
        ("__label__mar", "mar"),
        ("en", "eng"),
        ("__label__zzz", "zzz"),
        ("__label__zh-cn", None),
        ("__label__q", None),
    ],
)
def test_fasttext_label_normalisation(label: str, expected: str | None) -> None:
    assert normalise_fasttext_label(label) == expected


def test_script_language_table_covers_the_phase_zero_scripts() -> None:
    assert SCRIPT_LANGUAGES["Deva"] == frozenset({"hin", "mar", "kok", "nep"})
    assert "eng" not in SCRIPT_LANGUAGES["Deva"]


# --------------------------------------------------------------------------- #
# Classifier late fusion
# --------------------------------------------------------------------------- #


def _features(**overrides: Any) -> StructuralFeatures:
    data: dict[str, Any] = {
        "line_count": 20,
        "ink_density": 0.1,
        "aspect_ratio": 0.7,
        "has_table_lines": True,
        "table_line_count": 4,
        "colour_histogram": [0.5, 0.5],
        "handwriting_ratio": 0.2,
        "text_coverage": 0.4,
    }
    data.update(overrides)
    return StructuralFeatures(**data)


def _weights(**overrides: Any) -> FusionWeights:
    labels = (PageClassLabel.COMPLAINT_FORM, PageClassLabel.SUPPORTING_DOC)
    zeros = (0.0,) * (len(labels) + 7)
    data: dict[str, Any] = {
        "labels": labels,
        "coefficients": {
            "complaint_form": (1.0, 0.0, *zeros[2:]),
            "supporting_doc": (0.0, 1.0, *zeros[2:]),
        },
        "intercepts": {"complaint_form": 0.0, "supporting_doc": 0.0},
    }
    data.update(overrides)
    return FusionWeights(**data)


def test_structural_vector_matches_the_declared_order() -> None:
    vector = structural_feature_vector(_features())
    assert len(vector) == 7
    assert vector[3] == 1.0
    assert vector[1] == pytest.approx(0.1)


def test_fusion_returns_a_valid_page_class() -> None:
    result = late_fusion([0.9, 0.1], _features(), _weights())
    assert result.label is PageClassLabel.COMPLAINT_FORM
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert result.label.value in result.scores


def test_structural_features_can_flip_the_cnn() -> None:
    """Late fusion exists so layout can override a confused image model."""
    weights = _weights(
        coefficients={
            "complaint_form": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "supporting_doc": (0.0, 1.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0),
        }
    )
    assert (
        late_fusion([0.9, 0.1], _features(has_table_lines=True), weights).label
        is PageClassLabel.SUPPORTING_DOC
    )


def test_fusion_rejects_a_score_vector_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 2 CNN scores"):
        late_fusion([0.5], _features(), _weights())


def test_fusion_rejects_coefficients_of_the_wrong_length() -> None:
    weights = _weights(
        coefficients={"complaint_form": (1.0,), "supporting_doc": (0.0,)}
    )
    with pytest.raises(ValueError, match="length"):
        late_fusion([0.5, 0.5], _features(), weights)


def test_fusion_rejects_a_label_without_coefficients() -> None:
    weights = _weights(coefficients={"complaint_form": (0.0,) * 9})
    with pytest.raises(ValueError, match="no coefficients"):
        late_fusion([0.5, 0.5], _features(), weights)


# --------------------------------------------------------------------------- #
# Recognition-head cache
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_returns_the_same_object_on_a_hit() -> None:
    cache: RecHeadCache[object] = RecHeadCache(3, 900.0)
    sentinel = object()
    assert cache.get_or_load("deva", lambda: sentinel) is sentinel
    assert cache.get_or_load("deva", lambda: object()) is sentinel
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_evicts_the_least_recently_used_head() -> None:
    """maxsize=3 is why adding Kannada later is a model drop, not a rewrite."""
    cache: RecHeadCache[str] = RecHeadCache(3, 900.0)
    for key in ("latn", "deva", "beng"):
        cache.get_or_load(key, _constant(key))
    # Touching "latn" makes "deva" the least recently used.
    cache.get_or_load("latn", _constant("latn"))
    cache.get_or_load("knda", _constant("knda"))
    assert set(cache.keys()) == {"latn", "beng", "knda"}
    assert "deva" not in cache
    assert cache.evictions == 1


def _constant(value: str) -> Callable[[], str]:
    return lambda: value


def test_entries_expire_after_the_ttl() -> None:
    clock = _Clock()
    cache: RecHeadCache[str] = RecHeadCache(3, 900.0, clock=clock)
    cache.get_or_load("latn", lambda: "first")
    clock.now = 899.0
    assert cache.get_or_load("latn", lambda: "second") == "first"
    clock.now = 1800.0
    assert cache.get_or_load("latn", lambda: "third") == "third"
    assert cache.expirations >= 1


def test_clear_drops_everything() -> None:
    cache: RecHeadCache[str] = RecHeadCache(2, 900.0)
    cache.get_or_load("a", lambda: "a")
    cache.clear()
    assert len(cache) == 0


@pytest.mark.parametrize(("maxsize", "ttl"), [(0, 10.0), (2, 0.0), (2, -1.0)])
def test_cache_rejects_nonsense_configuration(maxsize: int, ttl: float) -> None:
    with pytest.raises(ValueError):
        RecHeadCache(maxsize, ttl)


# --------------------------------------------------------------------------- #
# LocalProvider — loaders, refusals, and the batched recognition path
# --------------------------------------------------------------------------- #


@pytest.fixture
def empty_models(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> Settings:
    """Settings pointing at a models directory that does not exist."""
    return make_settings(MODELS_DIR=tmp_path / "absent")


def test_construction_touches_no_file(empty_models: Settings) -> None:
    """Constructing a provider on a laptop with no weights must be free."""
    provider = LocalProvider(empty_models)
    assert provider.models_dir.name == "absent"
    assert not provider.models_dir.exists()


@pytest.mark.parametrize(
    ("call", "expected_fragment"),
    [
        (lambda p: p.identify_language("hello", "Latn"), "lid.176.ftz"),
        (lambda p: p.ocr_page(np.zeros((8, 8, 3), np.uint8), "Latn"), "det.onnx"),
        (lambda p: p.translate(["एक"], "hin", "eng"), "model.bin"),
        (lambda p: p.detect_script([np.zeros((8, 8, 3), np.uint8)]), "script_id"),
    ],
)
def test_every_loader_raises_a_typed_artifact_error(
    empty_models: Settings,
    call: Callable[[LocalProvider], object],
    expected_fragment: str,
) -> None:
    provider = LocalProvider(empty_models)
    with pytest.raises(ModelArtifactMissing) as excinfo:
        call(provider)
    assert expected_fragment in str(excinfo.value.path)
    assert excinfo.value.how_to_obtain


def test_classifier_loader_names_its_own_provenance(
    empty_models: Settings,
) -> None:
    provider = LocalProvider(empty_models)
    with pytest.raises(ModelArtifactMissing) as excinfo:
        provider.classify_page(np.zeros((8, 8, 3), np.uint8), _features())
    assert "page_classifier.onnx" in str(excinfo.value.path)


def test_script_detection_points_at_the_no_weights_alternative(
    empty_models: Settings,
) -> None:
    """The gap is stated, with the function that needs no artifact named."""
    provider = LocalProvider(empty_models)
    with pytest.raises(ModelArtifactMissing) as excinfo:
        provider.detect_script([np.zeros((8, 8, 3), np.uint8)])
    assert "detect_script_from_text" in excinfo.value.how_to_obtain


def test_empty_dictionary_file_is_an_artifact_error(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "det.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn_dict.txt").write_text("", encoding="utf-8")
    provider = LocalProvider(
        make_settings(MODELS_DIR=tmp_path),
        session_factory=lambda path: object(),
    )
    with pytest.raises(ModelArtifactMissing, match="non-empty"):
        provider._recognition_head("Latn")


def test_slm_methods_refuse_with_a_routing_message(
    empty_models: Settings,
) -> None:
    provider = LocalProvider(empty_models)
    with pytest.raises(ProviderMethodNotSupported, match="hf_api"):
        provider.ocr_page_heavy(np.zeros((4, 4, 3), np.uint8))


def test_translate_refuses_an_unsupported_direction(
    empty_models: Settings,
) -> None:
    """models/mt holds indic->eng only; the refusal says so rather than lying."""
    provider = LocalProvider(empty_models)
    with pytest.raises(ProviderMethodNotSupported) as excinfo:
        provider.translate(["hello"], "eng", "hin")
    assert excinfo.value.context["tgt"] == "hin"


def test_translate_short_circuits_the_trivial_cases(
    empty_models: Settings,
) -> None:
    """No model is loaded to translate nothing, or English into English."""
    provider = LocalProvider(empty_models)
    assert provider.translate([], "hin", "eng") == []
    assert provider.translate(["x"], "eng", "eng") == ["x"]


def test_identify_language_returns_none_for_blank_text_without_loading(
    empty_models: Settings,
) -> None:
    provider = LocalProvider(empty_models)
    result = provider.identify_language("   \n ", "Deva")
    assert result.language is None
    assert result.script == "Deva"


# --------------------------------------------------------------------------- #
# Batched recognition, end to end, against fake sessions
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Minimal ONNX Runtime session stand-in."""

    def __init__(self, produce: Callable[[np.ndarray], np.ndarray]) -> None:
        self._produce = produce
        self.batch_sizes: list[int] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="x")]

    def get_outputs(self) -> list[Any]:
        # Symbolic class axis: the loader's use_space_char reconciliation only
        # fires on a concrete int, so fake charsets keep the +1 convention.
        return [SimpleNamespace(name="y", shape=[None, None, "C"])]

    def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = next(iter(feed.values()))
        self.batch_sizes.append(int(tensor.shape[0]))
        return [self._produce(tensor)]


_CHARSET = ["a", "b", "c"]


def _fake_detection(tensor: np.ndarray) -> np.ndarray:
    """Two horizontal text blocks in the resized coordinate frame."""
    height, width = tensor.shape[2], tensor.shape[3]
    prob = np.zeros((1, 1, height, width), dtype=np.float32)
    prob[0, 0, 10:30, 10 : width - 10] = 0.95
    prob[0, 0, 50:70, 10 : width // 2] = 0.95
    return prob


def _fake_recognition(tensor: np.ndarray) -> np.ndarray:
    """Every crop decodes to ``ab``."""
    count = tensor.shape[0]
    array = np.full((count, 4, len(_CHARSET) + 1), 0.01, dtype=np.float32)
    array[:, 0, 1] = 0.9
    array[:, 1, 0] = 0.9
    array[:, 2, 2] = 0.9
    array[:, 3, 0] = 0.9
    return array


@pytest.fixture
def fake_ocr_provider(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> tuple[LocalProvider, dict[str, _FakeSession]]:
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "det.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn_dict.txt").write_text(
        "\n".join(_CHARSET), encoding="utf-8"
    )
    sessions = {
        "det": _FakeSession(_fake_detection),
        "rec": _FakeSession(_fake_recognition),
    }

    def factory(path: Path) -> _FakeSession:
        return sessions["det"] if path.name == "det.onnx" else sessions["rec"]

    provider = LocalProvider(
        make_settings(MODELS_DIR=tmp_path, OCR_BATCH_SIZE=16),
        session_factory=factory,
    )
    return provider, sessions


def test_ocr_page_produces_indexed_lines(
    fake_ocr_provider: tuple[LocalProvider, dict[str, _FakeSession]],
) -> None:
    provider, _ = fake_ocr_provider
    page = np.full((200, 400, 3), 255, dtype=np.uint8)
    lines = provider.ocr_page(page, "Latn")
    assert len(lines) == 2
    assert [line.line_index for line in lines] == [0, 1]
    assert all(line.text == "ab" for line in lines)
    assert all(line.script == "Latn" for line in lines)
    assert all(line.language is None for line in lines)
    assert all(line.is_handwritten is False for line in lines)


def test_recognition_runs_batched_not_looped(
    fake_ocr_provider: tuple[LocalProvider, dict[str, _FakeSession]],
) -> None:
    """Invariant 4 asserted where it can regress: at the session boundary."""
    provider, sessions = fake_ocr_provider
    crops = [np.full((48, 100, 3), 255, dtype=np.uint8) for _ in range(20)]
    provider._recognise(crops, "Latn")
    assert sessions["rec"].batch_sizes == [16, 4]


def test_recognition_maps_results_back_to_input_order(
    fake_ocr_provider: tuple[LocalProvider, dict[str, _FakeSession]],
) -> None:
    """Bucketing reorders work; it must not reorder answers."""
    provider, _ = fake_ocr_provider
    crops = [
        np.full((48, width, 3), 255, dtype=np.uint8)
        for width in (600, 40, 300, 50)
    ]
    decoded = provider._recognise(crops, "Latn")
    assert len(decoded) == 4
    assert all(text == "ab" for text, _ in decoded)


def test_handwriting_detector_hook_is_honoured(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """The default is an explicit gap, not a heuristic; injecting one works."""
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "det.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn_dict.txt").write_text(
        "\n".join(_CHARSET), encoding="utf-8"
    )
    sessions = {
        "det": _FakeSession(_fake_detection),
        "rec": _FakeSession(_fake_recognition),
    }
    provider = LocalProvider(
        make_settings(MODELS_DIR=tmp_path),
        session_factory=lambda path: (
            sessions["det"] if path.name == "det.onnx" else sessions["rec"]
        ),
        handwriting_detector=lambda crop: True,
    )
    lines = provider.ocr_page(np.full((200, 400, 3), 255, np.uint8), "Latn")
    assert lines and all(line.is_handwritten for line in lines)


def test_rec_head_cache_is_used_across_calls(
    fake_ocr_provider: tuple[LocalProvider, dict[str, _FakeSession]],
) -> None:
    provider, _ = fake_ocr_provider
    crops = [np.full((48, 100, 3), 255, dtype=np.uint8)]
    provider._recognise(crops, "Latn")
    provider._recognise(crops, "Latn")
    assert provider.rec_heads.hits == 1
    assert provider.rec_heads.misses == 1


def test_recognising_nothing_is_not_an_error(
    fake_ocr_provider: tuple[LocalProvider, dict[str, _FakeSession]],
) -> None:
    provider, _ = fake_ocr_provider
    assert provider._recognise([], "Latn") == []


def test_fusion_weights_round_trip_through_json(tmp_path: Path) -> None:
    """``fusion.json`` is the on-disk contract; it must survive a round trip."""
    weights = _weights()
    path = tmp_path / "fusion.json"
    path.write_text(weights.model_dump_json(), encoding="utf-8")
    loaded = FusionWeights.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert loaded == weights
