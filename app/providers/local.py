"""CPU-local provider: PP-OCRv5 ONNX, fastText LID, CTranslate2 IndicTrans2.

Everything here runs on the air-gapped node with no network and no GPU.  Three
properties are load-bearing:

**Loaders are lazy and every one of them raises**
:class:`~app.providers.base.ModelArtifactMissing`.  Constructing a
:class:`LocalProvider` touches no file, so a laptop with an empty ``models/``
directory can import, wire and test the whole pipeline; the typed error with a
path and a remediation string arrives at the exact call that needed the weight.

**Recognition is batched** (PLAN.md invariant 4).  Crops are bucketed by aspect
ratio, padded to the bucket width and pushed through the rec head in groups of
``OCR_BATCH_SIZE``.  :func:`plan_recognition_batches` is a pure function over
crop *shapes*, so the bucketing that matters for throughput is unit-testable
without a single weight file.  No looped path exists in this module.

**Recognition heads come from an LRU cache** with a TTL
(``REC_HEAD_CACHE_SIZE`` / ``REC_HEAD_CACHE_TTL_SECONDS``), keyed by script, so
adding Kannada later is a model drop and not a code change (BUILD-BRIEF M4).

The pure geometry/decoding helpers — bucketing, CTC decoding, DB
post-processing, reading order, the script-constrained LID label filter, the
classifier late fusion — are module-level functions rather than methods, both
because they are testable in isolation and because the benchmark tool
(``tools/bench_ocr_cpu.py``, WS-H) needs them without a provider.

This module does NOT handle:
  * the SLM or the VLM.  ``extract_fields``, ``extract_fields_batch`` and
    ``ocr_page_heavy`` raise
    :class:`~app.providers.base.ProviderMethodNotSupported`; PLAN.md D1 forbids
    >=0.9B weights on the laptop and Phase 0 routes them to
    :mod:`app.providers.hf_api`.  A local llama.cpp path is Phase 1+;
  * **downloading** anything.  ``tools/fetch_models.py`` populates ``models/``
    offline; this module only reads it;
  * handwriting detection.  ``OCRLine.is_handwritten`` is ``False`` unless a
    detector is injected — there is no approved artifact for it yet, and the
    gap is recorded in ``docs/decisions-wsc.md`` rather than papered over with
    a heuristic;
  * page-level language or script (PLAN.md invariant 1).  ``ocr_page`` stamps
    the *requested* script on each line; per-line LID is ``identify_language``,
    called by Stage 2b;
  * escalation decisions.  Nothing here reads ``OCRLine.confidence`` to route
    (invariant 3);
  * thread caps.  :mod:`app` set them before NumPy loaded; the ONNX session
    options here are the inner fence only.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.errors import ConfigurationError
from app.contracts.ocr import (
    ExtractionContext,
    LangCandidate,
    LangResult,
    OCRLine,
    PageClass,
    PageClassLabel,
    RawField,
    ScriptResult,
    ScriptVote,
    StructuralFeatures,
)
from app.providers.base import (
    CropImage,
    ModelArtifactMissing,
    PageImage,
    ProviderMethodNotSupported,
)
from app.settings import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = [
    "GPU_EXECUTION_PROVIDERS",
    "INDICTRANS2_TAGS",
    "ISO_639_1_TO_3",
    "REC_HEAD_COVERAGE",
    "REC_IMAGE_HEIGHT",
    "REC_WIDTH_BUCKETS",
    "SCRIPT_LANGUAGES",
    "STRUCTURAL_FEATURE_NAMES",
    "FusionWeights",
    "LocalProvider",
    "RecHeadCache",
    "RecognitionBatch",
    "bucket_width",
    "constrain_language_candidates",
    "crop_perspective",
    "ctc_greedy_decode",
    "db_postprocess",
    "detect_script_from_text",
    "late_fusion",
    "normalise_fasttext_label",
    "order_box_clockwise",
    "pad_crops_to_batch",
    "plan_recognition_batches",
    "script_of_char",
    "sort_boxes_reading_order",
    "structural_feature_vector",
]

_LOG: Final = logging.getLogger(__name__)

#: CUDA-family ORT execution providers, most preferred first.
#:
#: **TensorRT is deliberately not here**, although ORT reports it as available
#: and it is the faster of the two when it works.  ``get_available_providers()``
#: lists what the build was *compiled* with, not what can actually load, and a
#: provider that fails to load does not fall through to the next entry — ORT
#: abandons the whole list and retries on CPU:
#:
#:     Failed to load library libonnxruntime_providers_tensorrt.so with error:
#:     libnvinfer.so.10: cannot open shared object file
#:     Falling back to ['CPUExecutionProvider'] and retrying.
#:
#: So naming TensorRT on a machine that has CUDA but not TensorRT — which is
#: every stock CUDA image, including the one this was measured on — costs the
#: GPU entirely.  Asking for CUDA alone gets CUDA.  Anyone who has TensorRT
#: installed and wants it should add it here deliberately, having checked that
#: ``libnvinfer`` loads.
GPU_EXECUTION_PROVIDERS: Final[tuple[str, ...]] = ("CUDAExecutionProvider",)

_GPU_EXECUTION_PROVIDERS: Final[tuple[str, ...]] = GPU_EXECUTION_PROVIDERS


def _ctranslate2_has_cuda(ctranslate2: Any) -> bool:
    """Whether this CTranslate2 build can see a CUDA device.

    Asked rather than assumed: ``ctranslate2`` installs the same wheel name
    with and without CUDA support, so the only reliable signal is the device
    count at runtime, and it raises on a build with no CUDA at all.
    """
    try:
        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # noqa: BLE001  # pragma: no cover
        # Deliberately blind: a CTranslate2 built without CUDA raises whatever
        # its loader happens to raise, and every one of those answers means the
        # same thing — no GPU here, use the CPU path.
        return False


# --------------------------------------------------------------------------- #
# models/ layout — the authoritative paths referenced by models/README.md
# --------------------------------------------------------------------------- #

OCR_SUBDIR: Final[str] = "ocr"
LID_SUBDIR: Final[str] = "lid"
MT_SUBDIR: Final[str] = "mt"
SCRIPT_SUBDIR: Final[str] = "script"
CLASSIFIER_SUBDIR: Final[str] = "classifier"

DET_MODEL_FILENAME: Final[str] = "det.onnx"
REC_MODEL_TEMPLATE: Final[str] = "rec_{script}.onnx"
REC_DICT_TEMPLATE: Final[str] = "rec_{script}_dict.txt"
LID_MODEL_FILENAME: Final[str] = "lid.176.ftz"
SCRIPT_MODEL_FILENAME: Final[str] = "script_id.onnx"
SCRIPT_LABELS_FILENAME: Final[str] = "script_id_labels.json"
CLASSIFIER_MODEL_FILENAME: Final[str] = "page_classifier.onnx"
CLASSIFIER_FUSION_FILENAME: Final[str] = "fusion.json"
MT_CT2_DIRNAME: Final[str] = "indic-en"
MT_CT2_MODEL_FILENAME: Final[str] = "model.bin"
MT_SPM_SRC_FILENAME: Final[str] = "model.SRC"
MT_SPM_TGT_FILENAME: Final[str] = "model.TGT"

_FETCH_HINT: Final[str] = (
    "Run `python tools/fetch_models.py` on a networked machine and copy the "
    "resulting models/ directory to this node (see docs/AIR-GAPPED-DEPLOY.md)."
)

# --------------------------------------------------------------------------- #
# Recognition geometry
# --------------------------------------------------------------------------- #

#: PP-OCRv5 recognition input height.  Width varies; height never does.
REC_IMAGE_HEIGHT: Final[int] = 48

#: Bucket edges for padded recognition widths, ascending.  A crop is scaled to
#: :data:`REC_IMAGE_HEIGHT` and padded up to the first edge that fits.  The
#: widest edge is a clamp: a crop past it is *squeezed* horizontally rather
#: than growing the tensor without bound, which is PaddleOCR's own behaviour.
#:
#: The clamp used to sit at 640 and that was too low to be a rare event.  A
#: full-width A4 line at the 200 DPI rasterisation resolution normalises to
#: roughly 1000-1700 px at height 48, so on an ordinary English complaint
#: letter **5 of 13 lines** were over the edge and squeezed by up to 1.6x.  A
#: horizontally compressed line does not degrade gently under CTC: it drops
#: characters and then stops, so "We acknowledge your complaint regarding the
#: disputed transaction." decoded as "W." — at high confidence, because a
#: short decode is a confident one.  Any measurement keyed on confidence rather
#: than on ground truth will therefore call this healthy.
#:
#: 1920 covers a full-width line with margin.  The extra edges are only ever
#: instantiated for crops that need them, so a page of short form fields still
#: batches into the small buckets and pays nothing.
REC_WIDTH_BUCKETS: Final[tuple[int, ...]] = (
    80,
    160,
    240,
    320,
    480,
    640,
    960,
    1280,
    1920,
)


class RecognitionBatch(BaseModel):
    """One padded recognition call: a bucket width and the crops that fit it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_width: int = Field(gt=0)
    #: Positions in the caller's crop sequence, in the caller's order.
    indices: tuple[int, ...] = Field(min_length=1)


def bucket_width(
    height: int,
    width: int,
    *,
    image_height: int = REC_IMAGE_HEIGHT,
    buckets: Sequence[int] = REC_WIDTH_BUCKETS,
) -> int:
    """Padded width for a crop of ``(height, width)`` after height normalisation.

    Raises:
        ValueError: on a non-positive dimension or an empty/unsorted bucket list.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"crop dimensions must be positive, got height={height}, "
            f"width={width}"
        )
    if not buckets:
        raise ValueError("buckets must not be empty")
    if list(buckets) != sorted(buckets):
        raise ValueError(f"buckets must be ascending, got {tuple(buckets)}")
    scaled = math.ceil(width * image_height / height)
    for edge in buckets:
        if scaled <= edge:
            return int(edge)
    return int(buckets[-1])


def plan_recognition_batches(
    shapes: Sequence[tuple[int, int]],
    *,
    batch_size: int,
    image_height: int = REC_IMAGE_HEIGHT,
    buckets: Sequence[int] = REC_WIDTH_BUCKETS,
) -> tuple[RecognitionBatch, ...]:
    """Group crop shapes into padded recognition batches — PLAN.md invariant 4.

    Pure: takes ``(height, width)`` pairs, never pixels, so throughput-relevant
    behaviour is testable without a weight file.  Every input index appears in
    exactly one batch; batches are ordered by ascending bucket width and, within
    a bucket, by input order, so a run is reproducible.

    Args:
        shapes: ``(height, width)`` per crop, in caller order.
        batch_size: crops per call (``OCR_BATCH_SIZE``; the tuned band is 16-32).

    Raises:
        ValueError: for ``batch_size < 1`` or a non-positive crop dimension.
    """
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1, got {batch_size}; a value of 1 is a "
            "looped path and is legitimate only in tools/bench_ocr_cpu.py "
            "(PLAN.md invariant 4)"
        )
    by_bucket: dict[int, list[int]] = {}
    for index, (height, width) in enumerate(shapes):
        target = bucket_width(
            height, width, image_height=image_height, buckets=buckets
        )
        by_bucket.setdefault(target, []).append(index)

    plan: list[RecognitionBatch] = []
    for target in sorted(by_bucket):
        members = by_bucket[target]
        for start in range(0, len(members), batch_size):
            plan.append(
                RecognitionBatch(
                    target_width=target,
                    indices=tuple(members[start : start + batch_size]),
                )
            )
    return tuple(plan)


def pad_crops_to_batch(
    crops: Sequence[CropImage],
    batch: RecognitionBatch,
    *,
    image_height: int = REC_IMAGE_HEIGHT,
) -> NDArray[np.float32]:
    """Build the ``(N, 3, H, W)`` float tensor for one :class:`RecognitionBatch`.

    Each crop is scaled to ``image_height`` preserving aspect, clipped to the
    bucket width, placed at the left of a zero canvas and normalised to
    ``[-1, 1]``.  Zero padding *after* normalisation is mid-grey to the model,
    which is what PaddleOCR's own pre-processing does.
    """
    cv2 = _cv2()
    tensor = np.zeros(
        (len(batch.indices), 3, image_height, batch.target_width),
        dtype=np.float32,
    )
    for slot, index in enumerate(batch.indices):
        crop = np.asarray(crops[index])
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"crop {index} is empty: shape {crop.shape!r}")
        scaled_width = max(
            1, min(batch.target_width, math.ceil(width * image_height / height))
        )
        resized = cv2.resize(crop, (scaled_width, image_height))
        normalised = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor[slot, :, :, :scaled_width] = normalised.transpose(2, 0, 1)
    return tensor


def ctc_greedy_decode(
    probabilities: NDArray[np.float32],
    charset: Sequence[str],
    *,
    blank_index: int = 0,
) -> list[tuple[str, float]]:
    """Greedy CTC decode of a ``(N, T, C)`` probability tensor.

    PP-OCR's ``CTCLabelDecode`` puts the blank at class 0, so class ``i`` maps
    to ``charset[i - 1]``.  Confidence is the mean of the winning probabilities
    over the timesteps that survived blank/repeat collapsing — an empty result
    scores ``0.0`` rather than ``nan``.

    Raises:
        ValueError: on the wrong rank, or a class count that disagrees with the
            charset (a silently mismatched dictionary is the single most common
            way a rec head produces confident nonsense).
    """
    array = np.asarray(probabilities, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"expected a (N, T, C) probability tensor, got shape {array.shape!r}"
        )
    expected_classes = len(charset) + 1
    if array.shape[2] != expected_classes:
        raise ValueError(
            f"rec head emits {array.shape[2]} classes but the dictionary has "
            f"{len(charset)} entries (expected {expected_classes} with the "
            "CTC blank); the model and its dict file do not match"
        )
    results: list[tuple[str, float]] = []
    best_index = array.argmax(axis=2)
    best_prob = array.max(axis=2)
    for row in range(array.shape[0]):
        characters: list[str] = []
        scores: list[float] = []
        previous = -1
        for step in range(array.shape[1]):
            class_id = int(best_index[row, step])
            if class_id != previous and class_id != blank_index:
                characters.append(charset[class_id - 1])
                scores.append(float(best_prob[row, step]))
            previous = class_id
        confidence = float(np.mean(scores)) if scores else 0.0
        results.append(("".join(characters), min(1.0, max(0.0, confidence))))
    return results


def order_box_clockwise(points: NDArray[np.float32]) -> list[float]:
    """Order 4 ``(x, y)`` corners clockwise from top-left, flattened to 8 floats.

    Uses the sum/difference rule: top-left minimises ``x + y``, bottom-right
    maximises it, top-right minimises ``y - x``.  Robust for the near-axis-
    aligned quads a document detector produces.
    """
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {array.shape[0]}")
    total = array.sum(axis=1)
    diff = array[:, 1] - array[:, 0]
    top_left = array[int(np.argmin(total))]
    bottom_right = array[int(np.argmax(total))]
    top_right = array[int(np.argmin(diff))]
    bottom_left = array[int(np.argmax(diff))]
    return [
        float(top_left[0]),
        float(top_left[1]),
        float(top_right[0]),
        float(top_right[1]),
        float(bottom_right[0]),
        float(bottom_right[1]),
        float(bottom_left[0]),
        float(bottom_left[1]),
    ]


def db_postprocess(
    probability_map: NDArray[np.float32],
    *,
    ratio_h: float = 1.0,
    ratio_w: float = 1.0,
    binary_threshold: float = 0.3,
    box_threshold: float = 0.6,
    unclip_ratio: float = 1.5,
    min_box_side: float = 3.0,
    max_candidates: int = 1000,
) -> list[list[float]]:
    """Turn a DB probability map into text-line quads (8 floats each).

    The dilation step approximates the Vatti offset PaddleOCR performs with
    ``pyclipper`` by growing the minimum-area rectangle by
    ``area * unclip_ratio / perimeter`` on each side.  For a rectangle the two
    agree exactly, and a detected text line is a rectangle by construction —
    this avoids adding a dependency for a closed-form case.

    Args:
        ratio_h / ratio_w: multiply output coordinates back into original-page
            space after detection ran on a resized image.
    """
    cv2 = _cv2()
    prob = np.asarray(probability_map, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(
            f"probability map must be 2-D, got shape {prob.shape!r}"
        )
    bitmap = (prob > binary_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes: list[list[float]] = []
    for contour in contours[:max_candidates]:
        if len(contour) < 4:
            continue
        (centre_x, centre_y), (width, height), angle = cv2.minAreaRect(contour)
        if min(width, height) < min_box_side:
            continue
        score = _box_score(prob, contour)
        if score < box_threshold:
            continue
        perimeter = 2.0 * (width + height)
        if perimeter <= 0:
            continue
        distance = (width * height) * unclip_ratio / perimeter
        expanded = (
            (centre_x, centre_y),
            (width + 2.0 * distance, height + 2.0 * distance),
            angle,
        )
        corners = cv2.boxPoints(expanded)
        ordered = order_box_clockwise(np.asarray(corners, dtype=np.float32))
        boxes.append(
            [
                value * (ratio_w if position % 2 == 0 else ratio_h)
                for position, value in enumerate(ordered)
            ]
        )
    return boxes


def _box_score(
    probability_map: NDArray[np.float32], contour: NDArray[Any]
) -> float:
    """Mean probability inside one contour — DB's box confidence."""
    cv2 = _cv2()
    height, width = probability_map.shape
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    x_min = max(0, int(points[:, 0].min()))
    x_max = min(width - 1, int(points[:, 0].max()))
    y_min = max(0, int(points[:, 1].min()))
    y_max = min(height - 1, int(points[:, 1].max()))
    if x_max < x_min or y_max < y_min:
        return 0.0
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = points.copy()
    shifted[:, 0] -= x_min
    shifted[:, 1] -= y_min
    cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 1)
    if not mask.any():
        return 0.0
    window = probability_map[y_min : y_max + 1, x_min : x_max + 1]
    return float(window[mask.astype(bool)].mean())


def crop_perspective(page: PageImage, bbox: Sequence[float]) -> CropImage:
    """Rectify one detected quad out of the page (PaddleOCR's rotate-crop).

    Raises:
        ValueError: when ``bbox`` is not 8 finite floats.
    """
    cv2 = _cv2()
    if len(bbox) != 8:
        raise ValueError(f"bbox must be 8 floats, got {len(bbox)}")
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"bbox contains a non-finite coordinate: {list(bbox)!r}")
    source = np.asarray(bbox, dtype=np.float32).reshape(4, 2)
    width = int(
        max(
            np.linalg.norm(source[0] - source[1]),
            np.linalg.norm(source[2] - source[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(source[0] - source[3]),
            np.linalg.norm(source[1] - source[2]),
        )
    )
    width = max(width, 1)
    height = max(height, 1)
    target = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    crop = cv2.warpPerspective(
        np.asarray(page),
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    return np.ascontiguousarray(crop)


def sort_boxes_reading_order(
    boxes: Sequence[Sequence[float]], *, line_tolerance: float = 10.0
) -> list[int]:
    """Indices of ``boxes`` in reading order: top-to-bottom, then left-to-right.

    ``line_tolerance`` is the vertical slack (pixels) within which two boxes
    count as the same visual row, so a two-column label/value form does not
    interleave.  Returns indices, not boxes, because ``line_index`` must be
    assignable without reordering the caller's crops.
    """
    if line_tolerance <= 0:
        raise ValueError(
            f"line_tolerance must be positive, got {line_tolerance}"
        )
    keyed: list[tuple[float, float, int]] = []
    for index, box in enumerate(boxes):
        ys = [float(box[i]) for i in range(1, 8, 2)]
        xs = [float(box[i]) for i in range(0, 8, 2)]
        keyed.append((sum(ys) / 4.0, min(xs), index))
    keyed.sort(key=lambda item: (item[0], item[1]))

    rows: list[list[tuple[float, float, int]]] = []
    for entry in keyed:
        if rows and abs(entry[0] - rows[-1][0][0]) <= line_tolerance:
            rows[-1].append(entry)
        else:
            rows.append([entry])
    ordered: list[int] = []
    for row in rows:
        row.sort(key=lambda item: item[1])
        ordered.extend(entry[2] for entry in row)
    return ordered


# --------------------------------------------------------------------------- #
# Script detection from text (no artifact required)
# --------------------------------------------------------------------------- #

#: ISO 15924 code per Unicode block, covering the scripts an RBI complaint can
#: plausibly carry.  Ranges are inclusive.  Anything unlisted is "common"
#: (digits, punctuation, whitespace) and casts no vote.
_SCRIPT_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x0041, 0x005A, "Latn"),
    (0x0061, 0x007A, "Latn"),
    (0x00C0, 0x024F, "Latn"),
    (0x0900, 0x097F, "Deva"),
    (0xA8E0, 0xA8FF, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B00, 0x0B7F, "Orya"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0600, 0x06FF, "Arab"),
)

#: ISO 15924 for "undetermined" — used when nothing casts a vote.
UNDETERMINED_SCRIPT: Final[str] = "Zyyy"


def script_of_char(character: str) -> str | None:
    """ISO 15924 code for one character, or ``None`` when it is script-neutral."""
    if len(character) != 1:
        raise ValueError(
            f"expected a single character, got {len(character)} characters"
        )
    code_point = ord(character)
    for start, end, script in _SCRIPT_RANGES:
        if start <= code_point <= end:
            return script
    return None


def _majority(values: Sequence[str]) -> str | None:
    """Most common non-empty value, ties broken alphabetically."""
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda key: (counts[key], key))


def _line_script(text: str, *, fallback: str) -> str:
    """The script of one recognised line, from its own characters.

    ``fallback`` (the reading head's script) is used only when nothing in the
    line casts a vote — an account number or an amount is digits and
    punctuation, which belong to no script.  Guessing from neighbours would be
    inventing evidence, and the value of a digit string does not depend on the
    answer.
    """
    counts: dict[str, int] = {}
    for character in text:
        script = script_of_char(character)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return fallback
    return max(counts, key=lambda key: (counts[key], key))


def detect_script_from_text(texts: Sequence[str]) -> ScriptResult:
    """Majority-vote a page's script from already-recognised text.

    Phase 0's fallback for :meth:`LocalProvider.detect_script`, which needs an
    image-space classifier artifact we have not approved.  It is exact wherever
    the text is right and needs no weights at all, which is why it lives here as
    a public function: WS-E can vote on Tier-1 output instead of blocking on a
    model that ``tools/fetch_models.py`` does not fetch.

    Returns ``Zyyy`` with zero confidence when no character casts a vote — never
    a guessed script.
    """
    counts: dict[str, int] = {}
    votes: list[ScriptVote] = []
    for text in texts:
        per_line: dict[str, int] = {}
        for character in text:
            script = script_of_char(character)
            if script is not None:
                per_line[script] = per_line.get(script, 0) + 1
                counts[script] = counts.get(script, 0) + 1
        if per_line:
            total = sum(per_line.values())
            winner = max(per_line, key=lambda key: (per_line[key], key))
            votes.append(
                ScriptVote(
                    script=winner, confidence=per_line[winner] / total
                )
            )
    if not counts:
        return ScriptResult(
            script=UNDETERMINED_SCRIPT,
            confidence=0.0,
            distribution={},
            votes=votes,
            sampled_crops=len(texts),
        )
    total_chars = sum(counts.values())
    distribution = {
        script: count / total_chars for script, count in counts.items()
    }
    distribution = _renormalise(distribution)
    winner = max(distribution, key=lambda key: (distribution[key], key))
    return ScriptResult(
        script=winner,
        confidence=distribution[winner],
        distribution=distribution,
        votes=votes,
        sampled_crops=len(texts),
    )


def _renormalise(distribution: Mapping[str, float]) -> dict[str, float]:
    """Scale a mapping to sum to exactly 1.0 within the contract's tolerance."""
    total = sum(distribution.values())
    if total <= 0:
        return {}
    scaled = {key: value / total for key, value in distribution.items()}
    # Absorb float error into the largest bucket so the contract's sum check
    # cannot fail on a legitimate distribution.
    largest = max(scaled, key=lambda key: (scaled[key], key))
    scaled[largest] += 1.0 - sum(scaled.values())
    return scaled


# --------------------------------------------------------------------------- #
# Language identification — script-constrained label sets
# --------------------------------------------------------------------------- #

#: The label set ``identify_language`` may choose from, per detected script
#: (PRD P0-5: Devanagari implies hin/mar/kok/nep, never eng).  Additive by
#: design — adding Kannada is a row here plus a rec head in ``models/ocr``.
SCRIPT_LANGUAGES: Final[Mapping[str, frozenset[str]]] = {
    "Latn": frozenset({"eng"}),
    "Deva": frozenset({"hin", "mar", "kok", "nep"}),
    "Beng": frozenset({"ben", "asm"}),
    "Guru": frozenset({"pan"}),
    "Gujr": frozenset({"guj"}),
    "Orya": frozenset({"ori"}),
    "Taml": frozenset({"tam"}),
    "Telu": frozenset({"tel"}),
    "Knda": frozenset({"kan"}),
    "Mlym": frozenset({"mal"}),
    "Arab": frozenset({"urd"}),
}

#: Which **scripts** each recognition head can actually emit, as opposed to
#: which script it is named after.  Measured from the shipped charsets, not
#: assumed: ``rec_deva_dict.txt`` holds 128 Devanagari characters *and* the
#: full 52-letter Latin alphabet, while ``rec_latn_dict.txt`` holds Latin and a
#: single stray Devanagari codepoint.  The Devanagari head is therefore a
#: strict superset — it is the only head that can read a page mixing English
#: and Hindi, which is most RBI complaint forms.
#:
#: This is what makes head selection a *coverage* question rather than a
#: majority vote.  Voting picks one script and reads the whole page with that
#: head, so on a bilingual form every line in the losing script decodes to
#: rubble (measured: Devanagari CER 0.90 under the Latin head against
#: 0.009-0.042 under the Devanagari head, on the same pixels; on Latin the two
#: are equal at 0.000-0.002, so the wider head costs nothing).
#:
#: Additive in the same way as :data:`SCRIPT_LANGUAGES`: a Tamil head that also
#: carries Latin is a row here plus the ``.onnx`` drop, with no code change.
REC_HEAD_COVERAGE: Final[Mapping[str, frozenset[str]]] = {
    "Latn": frozenset({"Latn"}),
    "Deva": frozenset({"Deva", "Latn"}),
}

#: lid.176 emits a mix of ISO 639-1 and 639-3 labels; the contract is 639-3.
ISO_639_1_TO_3: Final[Mapping[str, str]] = {
    "as": "asm",
    "bn": "ben",
    "en": "eng",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

_FASTTEXT_LABEL_PREFIX: Final[str] = "__label__"


def normalise_fasttext_label(label: str) -> str | None:
    """Map a fastText label to ISO 639-3, or ``None`` when it does not map.

    ``__label__hi`` becomes ``hin``; an already-3-letter label passes through;
    anything else (``__label__zh``, ``__label__ceb``) returns ``None`` rather
    than a guess, and the caller drops it.
    """
    stripped = label.removeprefix(_FASTTEXT_LABEL_PREFIX).strip().lower()
    if len(stripped) == 2:
        return ISO_639_1_TO_3.get(stripped)
    if len(stripped) == 3 and stripped.isalpha():
        return stripped
    return None


def constrain_language_candidates(
    raw: Sequence[tuple[str, float]],
    script: str,
    *,
    max_candidates: int = 4,
) -> LangResult:
    """Filter raw LID output to the label set ``script`` permits.

    The whole point of PRD P0-5: a Devanagari line is never English, however
    confidently a 176-language model says so, and letting that label through
    poisons the derived page summary and the MT routing behind it.

    When nothing survives the constraint the language is ``None`` with zero
    confidence — **never** the first permitted member.  A guessed language is
    exactly the silent default BUILD-BRIEF sec. 0.5 forbids; Stage 2b decides
    what to do with an unlabelled line.
    """
    allowed = SCRIPT_LANGUAGES.get(script)
    scored: dict[str, float] = {}
    for label, probability in raw:
        code = normalise_fasttext_label(label)
        if code is None:
            continue
        if allowed is not None and code not in allowed:
            continue
        scored[code] = max(scored.get(code, 0.0), float(probability))
    if not scored:
        return LangResult(
            language=None, confidence=0.0, script=script, candidates=[]
        )
    total = sum(scored.values())
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    candidates = [
        LangCandidate(
            language=code,
            confidence=min(1.0, value / total) if total > 0 else 0.0,
        )
        for code, value in ranked[:max_candidates]
    ]
    return LangResult(
        language=candidates[0].language,
        confidence=candidates[0].confidence,
        script=script,
        candidates=candidates,
    )


# --------------------------------------------------------------------------- #
# Page classifier late fusion
# --------------------------------------------------------------------------- #

#: Order of the structural half of the fusion feature vector.  Fixed, because
#: ``fusion.json`` stores coefficients positionally.
STRUCTURAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "line_count_log",
    "ink_density",
    "aspect_ratio",
    "has_table_lines",
    "table_line_count_log",
    "handwriting_ratio",
    "text_coverage",
)


def structural_feature_vector(feats: StructuralFeatures) -> list[float]:
    """The structural half of the fusion input.

    Ordered by :data:`STRUCTURAL_FEATURE_NAMES`.

    Counts are log1p-compressed so a 400-line bank statement does not swamp the
    bounded features it sits beside.
    """
    return [
        math.log1p(feats.line_count),
        feats.ink_density,
        feats.aspect_ratio,
        1.0 if feats.has_table_lines else 0.0,
        math.log1p(feats.table_line_count),
        feats.handwriting_ratio,
        feats.text_coverage,
    ]


class FusionWeights(BaseModel):
    """Coefficients of the late-fusion logistic regression (``fusion.json``).

    ``coefficients[label]`` runs over ``[*cnn_scores, *structural_features]``,
    so its length is ``len(labels) + len(STRUCTURAL_FEATURE_NAMES)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    labels: tuple[PageClassLabel, ...] = Field(min_length=2)
    coefficients: dict[str, tuple[float, ...]]
    intercepts: dict[str, float]

    def expected_length(self) -> int:
        """Coefficient-vector length implied by ``labels``."""
        return len(self.labels) + len(STRUCTURAL_FEATURE_NAMES)


def late_fusion(
    cnn_scores: Sequence[float],
    feats: StructuralFeatures,
    weights: FusionWeights,
) -> PageClass:
    """Fuse CNN probabilities with structural features into a :class:`PageClass`.

    A plain softmax over per-label linear scores — the "logistic-regression late
    fusion" of BUILD-BRIEF M4.  Pure, so the fusion can be tested and retuned
    without the CNN.

    Raises:
        ValueError: when ``cnn_scores`` or a coefficient vector has the wrong
            length, or a label is missing from the weights.
    """
    if len(cnn_scores) != len(weights.labels):
        raise ValueError(
            f"expected {len(weights.labels)} CNN scores, got {len(cnn_scores)}"
        )
    features = [float(value) for value in cnn_scores]
    features.extend(structural_feature_vector(feats))
    logits: dict[str, float] = {}
    for label in weights.labels:
        key = label.value
        coefficients = weights.coefficients.get(key)
        if coefficients is None:
            raise ValueError(f"fusion weights have no coefficients for {key!r}")
        if len(coefficients) != weights.expected_length():
            raise ValueError(
                f"coefficients for {key!r} have length {len(coefficients)}, "
                f"expected {weights.expected_length()}"
            )
        logits[key] = weights.intercepts.get(key, 0.0) + float(
            np.dot(np.asarray(coefficients, dtype=np.float64), features)
        )
    ceiling = max(logits.values())
    exponentiated = {
        key: math.exp(value - ceiling) for key, value in logits.items()
    }
    total = sum(exponentiated.values())
    scores = _renormalise(
        {key: value / total for key, value in exponentiated.items()}
    )
    winner = max(scores, key=lambda key: (scores[key], key))
    return PageClass(
        label=PageClassLabel(winner),
        confidence=scores[winner],
        scores=scores,
    )


# --------------------------------------------------------------------------- #
# Recognition-head cache
# --------------------------------------------------------------------------- #

T = TypeVar("T")


class RecHeadCache(Generic[T]):
    """LRU cache with a TTL — BUILD-BRIEF M4's runtime rec-head selection.

    ``maxsize`` is ``REC_HEAD_CACHE_SIZE`` (default 3) and ``ttl_seconds`` is
    ``REC_HEAD_CACHE_TTL_SECONDS`` (default 900).  The TTL exists so a long-
    lived worker eventually releases a head it stopped using rather than
    pinning three ONNX sessions of resident memory forever.

    Thread-safe.  ``clock`` is injectable so TTL behaviour is testable without
    sleeping.
    """

    def __init__(
        self,
        maxsize: int,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        """Return the cached value for ``key``, loading it on miss.

        ``loader`` runs outside the lock, so a slow ONNX session build does not
        block other scripts; a concurrent duplicate load is possible and
        harmless (last writer wins, the loser is garbage-collected).
        """
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry[1]
            self.misses += 1
        value = loader()
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
                self.evictions += 1
        return value

    def peek(self, key: str) -> T | None:
        """Return a live value without recording a hit or refreshing LRU order."""
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(key)
            return None if entry is None else entry[1]

    def clear(self) -> None:
        """Drop every entry (worker shutdown, tests)."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> tuple[str, ...]:
        """Live keys, least-recently-used first."""
        with self._lock:
            self._expire_locked()
            return tuple(self._entries)

    def __len__(self) -> int:
        with self._lock:
            self._expire_locked()
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.peek(key) is not None

    def _expire_locked(self) -> None:
        """Drop entries older than the TTL.  Caller holds the lock."""
        now = self._clock()
        stale = [
            key
            for key, (stored_at, _) in self._entries.items()
            if now - stored_at >= self.ttl_seconds
        ]
        for key in stale:
            del self._entries[key]
            self.expirations += 1


# --------------------------------------------------------------------------- #
# Lazy third-party imports
# --------------------------------------------------------------------------- #

_CV2_LOCK = threading.Lock()
_CV2_MODULE: Any = None


def _cv2() -> Any:
    """Import OpenCV on first use.

    Deliberately lazy: ``import cv2`` costs a few hundred milliseconds and
    ``import app.providers`` must stay cheap for the API process, which never
    touches pixels.
    """
    global _CV2_MODULE
    with _CV2_LOCK:
        if _CV2_MODULE is None:
            import cv2

            _CV2_MODULE = cv2
        return _CV2_MODULE


def _require_artifact(path: Path, artifact: str, how_to_obtain: str) -> Path:
    """Return ``path`` or raise :class:`ModelArtifactMissing`."""
    if not path.exists():
        raise ModelArtifactMissing(
            artifact=artifact, path=path, how_to_obtain=how_to_obtain
        )
    return path


# --------------------------------------------------------------------------- #
# IndicTrans2
# --------------------------------------------------------------------------- #

#: IndicTrans2 language tags, keyed by the ISO 639-3 code the contract uses.
#: Konkani is ``gom_Deva`` upstream (Goan Konkani), which is why this table
#: exists rather than a string concatenation.
INDICTRANS2_TAGS: Final[Mapping[str, str]] = {
    "asm": "asm_Beng",
    "ben": "ben_Beng",
    "eng": "eng_Latn",
    "guj": "guj_Gujr",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "kok": "gom_Deva",
    "mal": "mal_Mlym",
    "mar": "mar_Deva",
    "nep": "npi_Deva",
    "ori": "ory_Orya",
    "pan": "pan_Guru",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "urd": "urd_Arab",
}

#: The engine string stamped onto ``Field.mt_engine`` (PLAN.md invariant 10).
MT_ENGINE_NAME: Final[str] = "indictrans2-indic-en-dist-200M-ct2-int8"


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #


class LocalProvider:
    """CPU-local implementation of :class:`~app.providers.base.ModelProvider`.

    Construction is free of I/O: nothing is read from ``models/`` until a
    method needs it.  That is what lets the whole service be imported, wired
    and unit-tested on a machine with no weights at all.

    Args:
        settings: batch size, cache sizes and thread counts come from here.
        models_dir: overrides ``settings.models_dir`` (tests, alternate mounts).
        session_factory: builds an ONNX Runtime session from a path; injectable
            so the recognition path can be exercised with a fake head.
        fasttext_loader: builds the LID model from a path.
        translator_factory: builds the CTranslate2 translator from a directory.
        handwriting_detector: optional per-crop predicate; without one,
            ``OCRLine.is_handwritten`` is ``False`` and the gap is explicit
            rather than heuristic (see the module docstring).
        clock: monotonic time source for the rec-head cache TTL.
    """

    #: Name used in error messages and metrics.
    name: Final[str] = "local"

    def __init__(
        self,
        settings: Settings,
        *,
        models_dir: Path | None = None,
        session_factory: Callable[[Path], Any] | None = None,
        fasttext_loader: Callable[[Path], Any] | None = None,
        translator_factory: Callable[[Path], Any] | None = None,
        handwriting_detector: Callable[[CropImage], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.models_dir: Path = (
            Path(models_dir) if models_dir is not None else settings.models_dir
        )
        self._session_factory = session_factory or self._default_session_factory
        self._fasttext_loader = fasttext_loader or _default_fasttext_loader
        self._translator_factory = (
            translator_factory or self._default_translator_factory
        )
        self._handwriting_detector = handwriting_detector
        self._lock = threading.RLock()
        self._det_session: Any = None
        self._script_session: Any = None
        self._script_labels: tuple[str, ...] | None = None
        self._classifier_session: Any = None
        self._fusion_weights: FusionWeights | None = None
        self._lid_model: Any = None
        self._mt: tuple[Any, Any, Any] | None = None
        self.rec_heads: RecHeadCache[tuple[Any, tuple[str, ...]]] = RecHeadCache(
            maxsize=settings.REC_HEAD_CACHE_SIZE,
            ttl_seconds=float(settings.REC_HEAD_CACHE_TTL_SECONDS),
            clock=clock,
        )

    # ------------------------------------------------------------- factories
    def _execution_providers(self) -> list[str]:
        """ORT execution providers, most preferred first.

        CPU is **always** appended, never replaced: an ORT session listing only
        CUDA dies on the first op that has no CUDA kernel, and the detection
        graph has several.  Listing both is what makes the GPU an accelerator
        rather than a requirement per-operator.

        ``GPU_EXECUTION=require`` turns an absent CUDA provider into a startup
        failure.  That is the point of the mode: on a real GPU node the usual
        way to lose the GPU is a CPU-only ``onnxruntime`` wheel shadowing
        ``onnxruntime-gpu``, and the symptom is a node that works perfectly and
        is ten times too slow — which nobody notices until the queue backs up.
        """
        import onnxruntime  # type: ignore[import-untyped]

        if not self.settings.gpu_requested:
            return ["CPUExecutionProvider"]

        available = set(onnxruntime.get_available_providers())
        preferred = [name for name in _GPU_EXECUTION_PROVIDERS if name in available]
        if not preferred:
            if self.settings.gpu_required:
                raise ConfigurationError(
                    "GPU_EXECUTION=require but onnxruntime exposes no CUDA "
                    "execution provider; install onnxruntime-gpu (and make "
                    "sure the CPU-only onnxruntime wheel is uninstalled)",
                    context={
                        "available_providers": sorted(available),
                        "wanted": list(_GPU_EXECUTION_PROVIDERS),
                    },
                )
            _LOG.warning(
                "GPU requested (profile=%s) but onnxruntime exposes none of "
                "%s; falling back to CPU. Available: %s",
                self.settings.PROVIDER_PROFILE.value,
                ",".join(_GPU_EXECUTION_PROVIDERS),
                ",".join(sorted(available)),
            )
            return ["CPUExecutionProvider"]
        return [*preferred, "CPUExecutionProvider"]

    def _default_session_factory(self, path: Path) -> Any:
        """Build an ONNX Runtime session pinned to ``ORT_INTRA_OP_THREADS``.

        The environment variable set in :mod:`app` is the outer fence; these
        session options are the inner one, because ORT honours the session
        setting over the environment.  Both bound the *CPU* pool only; a CUDA
        session ignores them, which is why the thread budget of PRD P0-11 is
        still the CPU-side number on a GPU node.
        """
        # ONNX Runtime ships no py.typed marker; the session object is used
        # only through run()/get_inputs(), both already `Any` behind the
        # injectable session_factory.
        import onnxruntime  # type: ignore[import-untyped]

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.settings.ORT_INTRA_OP_THREADS
        options.inter_op_num_threads = 1
        return onnxruntime.InferenceSession(
            str(path),
            sess_options=options,
            providers=self._execution_providers(),
        )

    def _default_translator_factory(self, directory: Path) -> Any:
        """Build the CTranslate2 translator, INT8 on CPU and FP16 on GPU."""
        # CTranslate2 ships no py.typed marker.
        import ctranslate2  # type: ignore[import-untyped]

        on_gpu = self.settings.gpu_requested and _ctranslate2_has_cuda(ctranslate2)
        if self.settings.gpu_requested and not on_gpu:
            if self.settings.gpu_required:
                raise ConfigurationError(
                    "GPU_EXECUTION=require but this CTranslate2 build reports "
                    "no CUDA device; install a CUDA-enabled ctranslate2",
                    context={"directory": str(directory)},
                )
            _LOG.warning(
                "GPU requested but CTranslate2 reports no CUDA device; "
                "translating on CPU"
            )
        return ctranslate2.Translator(
            str(directory),
            device="cuda" if on_gpu else "cpu",
            # int8 is the calibrated CPU path (models/PROVENANCE.json). On CUDA
            # int8 needs compute capability >= 7.5; float16 is the portable
            # GPU choice and is not slower for a 200M-parameter model.
            compute_type="float16" if on_gpu else "int8",
            inter_threads=1,
            intra_threads=self.settings.ORT_INTRA_OP_THREADS,
        )

    # --------------------------------------------------------------- loaders
    def _detection_session(self) -> Any:
        with self._lock:
            if self._det_session is None:
                path = _require_artifact(
                    self.models_dir / OCR_SUBDIR / DET_MODEL_FILENAME,
                    "PP-OCRv5 text-detection ONNX (det)",
                    _FETCH_HINT,
                )
                self._det_session = self._session_factory(path)
            return self._det_session

    def _recognition_head(self, script: str) -> tuple[Any, tuple[str, ...]]:
        """Session + charset for ``script``, via the LRU/TTL cache."""
        key = script.lower()

        def load() -> tuple[Any, tuple[str, ...]]:
            model_path = _require_artifact(
                self.models_dir
                / OCR_SUBDIR
                / REC_MODEL_TEMPLATE.format(script=key),
                f"PP-OCRv5 recognition head for script {script!r}",
                _FETCH_HINT,
            )
            dict_path = _require_artifact(
                self.models_dir
                / OCR_SUBDIR
                / REC_DICT_TEMPLATE.format(script=key),
                f"character dictionary for the {script!r} recognition head",
                _FETCH_HINT,
            )
            charset = tuple(
                line.rstrip("\n\r")
                for line in dict_path.read_text(encoding="utf-8").splitlines()
            )
            if not charset:
                raise ModelArtifactMissing(
                    artifact=f"non-empty character dictionary for {script!r}",
                    path=dict_path,
                    how_to_obtain=(
                        "The dictionary file exists but is empty; re-copy it "
                        "from the model bundle. " + _FETCH_HINT
                    ),
                )
            session = self._session_factory(model_path)
            # PaddleOCR rec heads exported with use_space_char=true emit
            # blank + dict + space; the extracted inference.yml dict carries
            # only the characters.  Reconcile against the model's own class
            # axis so a genuinely mismatched dictionary still fails loudly
            # in ctc_greedy_decode.
            class_axis = session.get_outputs()[0].shape[-1]
            if isinstance(class_axis, int) and class_axis == len(charset) + 2:
                charset = charset + (" ",)
            return session, charset

        return self.rec_heads.get_or_load(key, load)

    def _language_model(self) -> Any:
        with self._lock:
            if self._lid_model is None:
                path = _require_artifact(
                    self.models_dir / LID_SUBDIR / LID_MODEL_FILENAME,
                    "fastText language-ID model (lid.176.ftz)",
                    _FETCH_HINT,
                )
                self._lid_model = self._fasttext_loader(path)
            return self._lid_model

    def _translation_model(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._mt is None:
                directory = self.models_dir / MT_SUBDIR / MT_CT2_DIRNAME
                _require_artifact(
                    directory / MT_CT2_MODEL_FILENAME,
                    "IndicTrans2 indic-en dist-200M, CTranslate2 INT8",
                    _FETCH_HINT,
                )
                source_spm = _require_artifact(
                    directory / MT_SPM_SRC_FILENAME,
                    "IndicTrans2 source SentencePiece model",
                    _FETCH_HINT,
                )
                target_spm = _require_artifact(
                    directory / MT_SPM_TGT_FILENAME,
                    "IndicTrans2 target SentencePiece model",
                    _FETCH_HINT,
                )
                import sentencepiece

                self._mt = (
                    self._translator_factory(directory),
                    sentencepiece.SentencePieceProcessor(
                        model_file=str(source_spm)
                    ),
                    sentencepiece.SentencePieceProcessor(
                        model_file=str(target_spm)
                    ),
                )
            return self._mt

    def _classifier(self) -> tuple[Any, FusionWeights]:
        with self._lock:
            if self._classifier_session is None or self._fusion_weights is None:
                model_path = _require_artifact(
                    self.models_dir
                    / CLASSIFIER_SUBDIR
                    / CLASSIFIER_MODEL_FILENAME,
                    "MobileNetV3-Small page-classifier ONNX",
                    "Train it with tools/train_page_classifier.py against the "
                    "committed corpus (WS-E), then place the export here.",
                )
                fusion_path = _require_artifact(
                    self.models_dir
                    / CLASSIFIER_SUBDIR
                    / CLASSIFIER_FUSION_FILENAME,
                    "late-fusion coefficients (fusion.json)",
                    "Written alongside the classifier export by "
                    "tools/train_page_classifier.py (WS-E).",
                )
                self._classifier_session = self._session_factory(model_path)
                self._fusion_weights = FusionWeights.model_validate_json(
                    fusion_path.read_text(encoding="utf-8")
                )
            return self._classifier_session, self._fusion_weights

    # -------------------------------------------------------------- protocol
    def detect_script(self, crops: Sequence[CropImage]) -> ScriptResult:
        """Image-space script vote over sampled crops.

        Requires ``models/script/script_id.onnx``, which is **not** in the
        approved download set (OPEN-QUESTIONS Q1 covers OCR weights only).
        Until it is, Stage 2 should vote on Tier-1 text with
        :func:`detect_script_from_text`, which needs no artifact; the typed
        error below says exactly that instead of returning a guess.
        """
        model_path = self.models_dir / SCRIPT_SUBDIR / SCRIPT_MODEL_FILENAME
        _require_artifact(
            model_path,
            "image-space script-identification ONNX",
            "No approved artifact exists yet (see docs/decisions-wsc.md, C7). "
            "Use app.providers.local.detect_script_from_text() on Tier-1 "
            "output, which requires no weights.",
        )
        labels_path = _require_artifact(
            self.models_dir / SCRIPT_SUBDIR / SCRIPT_LABELS_FILENAME,
            "script-identification label list",
            "Ships beside script_id.onnx.",
        )
        with self._lock:
            if self._script_session is None:
                self._script_session = self._session_factory(model_path)
                self._script_labels = tuple(
                    json.loads(labels_path.read_text(encoding="utf-8"))
                )
            session = self._script_session
            labels = self._script_labels or ()
        votes: list[ScriptVote] = []
        counts: dict[str, float] = {}
        for crop in crops:
            probabilities = _softmax(
                np.asarray(
                    session.run(None, {session.get_inputs()[0].name: _chw(crop)})[0]
                ).reshape(-1)
            )
            best = int(np.argmax(probabilities))
            script = labels[best]
            confidence = float(probabilities[best])
            votes.append(ScriptVote(script=script, confidence=confidence))
            counts[script] = counts.get(script, 0.0) + confidence
        if not counts:
            return ScriptResult(
                script=UNDETERMINED_SCRIPT,
                confidence=0.0,
                distribution={},
                votes=[],
                sampled_crops=0,
            )
        distribution = _renormalise(counts)
        winner = max(distribution, key=lambda key: (distribution[key], key))
        return ScriptResult(
            script=winner,
            confidence=distribution[winner],
            distribution=distribution,
            votes=votes,
            sampled_crops=len(votes),
        )

    def classify_page(
        self, page: PageImage, feats: StructuralFeatures
    ) -> PageClass:
        """MobileNetV3-Small logits fused with structural features."""
        session, weights = self._classifier()
        inputs = {session.get_inputs()[0].name: _chw(page, size=(224, 224))}
        logits = np.asarray(session.run(None, inputs)[0]).reshape(-1)
        return late_fusion(_softmax(logits).tolist(), feats, weights)

    def ocr_page(self, page: PageImage, script: str) -> list[OCRLine]:
        """Detection, then **batched** recognition (PLAN.md invariant 4).

        Lines come back in reading order with contiguous ``line_index`` values
        starting at 0.  ``language`` is ``None``: per-line LID is Stage 2b's
        job, and defaulting it here is precisely invariant 1's failure mode.

        ``script`` names the **head that reads the page**, and each line is
        labelled with the script of *its own characters* rather than with that
        head's name.  The two are not the same thing once a head covers more
        than one script: the Devanagari head reads a bilingual form correctly,
        and calling its English lines ``Deva`` would send Stage 2b looking for
        Hindi in ``Reserve Bank of India`` — ``languages_for_script("Deva")``
        excludes ``eng`` by design (PRD P0-5).  Labelling from the text is also
        what lets ``is_code_mixed`` ever become true.
        """
        image = np.asarray(page)
        boxes = self._detect(image)
        if not boxes:
            return []
        order = sort_boxes_reading_order(boxes)
        crops = [crop_perspective(image, boxes[index]) for index in order]
        decoded = self._recognise(crops, script)
        # Lines that cast no script vote of their own — an account number, an
        # amount — inherit the page's dominant script rather than the head's.
        # On an English form read by the Devanagari head those are the same
        # pixels either way, but "Latn" is the label a reviewer expects and the
        # one that lets Stage 2b consider English.
        voted = [
            _line_script(text, fallback="")
            for text, _confidence in decoded
            if text
        ]
        page_script = _majority(voted) or script
        lines: list[OCRLine] = []
        for position, (text, confidence) in enumerate(decoded):
            if not text:
                continue
            crop = crops[position]
            lines.append(
                OCRLine(
                    text=text,
                    bbox=[float(value) for value in boxes[order[position]]],
                    confidence=confidence,
                    script=_line_script(text, fallback=page_script),
                    language=None,
                    is_handwritten=(
                        bool(self._handwriting_detector(crop))
                        if self._handwriting_detector is not None
                        else False
                    ),
                    line_index=len(lines),
                )
            )
        return lines

    def ocr_page_heavy(self, page: PageImage) -> list[OCRLine]:
        """Tier-2 escalation is a VLM call; not local in Phase 0."""
        raise ProviderMethodNotSupported(
            "ocr_page_heavy",
            provider=self.name,
            implemented_by="app.providers.hf_api.HFAPIProvider (VLM tier 2)",
            context={
                "reason": (
                    "PLAN.md D1 forbids >=0.9B weights on the dev laptop; the "
                    "air-gapped profile needs an approved local VLM artifact "
                    "that Phase 0 does not ship"
                )
            },
        )

    def identify_language(self, text: str, script: str) -> LangResult:
        """Per-line LID with ``lid.176.ftz``, constrained by ``script``.

        Newlines are stripped because fastText treats them as document
        separators and would silently score only the first fragment.
        """
        flattened = " ".join(text.split())
        if not flattened:
            return LangResult(
                language=None, confidence=0.0, script=script, candidates=[]
            )
        model = self._language_model()
        labels, probabilities = model.predict(flattened, k=10)
        raw = [
            (str(label), float(probability))
            for label, probability in zip(labels, probabilities, strict=False)
        ]
        return constrain_language_candidates(raw, script)

    def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]:
        """The SLM is remote in Phase 0 and llama.cpp in Phase 1+."""
        raise ProviderMethodNotSupported(
            "extract_fields",
            provider=self.name,
            implemented_by="app.providers.hf_api.HFAPIProvider (SLM)",
            context={"document_id": ctx.document_id, "page_no": ctx.page_no},
        )

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        """The SLM is remote in Phase 0 and llama.cpp in Phase 1+."""
        raise ProviderMethodNotSupported(
            "extract_fields_batch",
            provider=self.name,
            implemented_by="app.providers.hf_api.HFAPIProvider (SLM)",
            context={"pages": len(ctxs)},
        )

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        """Per-field MT with IndicTrans2 indic-en dist-200M (CTranslate2 INT8).

        Empty strings short-circuit so positions are preserved without asking
        the model to translate nothing.  ``src == tgt`` returns the input
        verbatim rather than round-tripping it through a model.
        """
        if not texts:
            return []
        if src == tgt:
            return list(texts)
        source_tag = INDICTRANS2_TAGS.get(src)
        target_tag = INDICTRANS2_TAGS.get(tgt)
        if source_tag is None or target_tag is None or tgt != "eng":
            raise ProviderMethodNotSupported(
                f"translate(src={src!r}, tgt={tgt!r})",
                provider=self.name,
                implemented_by=(
                    "no configured engine; models/mt holds the indic->eng "
                    "dist-200M direction only"
                ),
                context={"src": src, "tgt": tgt},
            )
        translator, source_spm, target_spm = self._translation_model()
        positions = [index for index, text in enumerate(texts) if text.strip()]
        if not positions:
            return list(texts)
        batch = [
            [source_tag, target_tag]
            + list(source_spm.encode(texts[index], out_type=str))
            for index in positions
        ]
        results = translator.translate_batch(
            batch,
            beam_size=1,
            max_batch_size=self.settings.OCR_BATCH_SIZE,
        )
        output = list(texts)
        for slot, index in enumerate(positions):
            hypothesis = results[slot].hypotheses[0]
            output[index] = str(target_spm.decode(hypothesis))
        return output

    # --------------------------------------------------------------- private
    def _detect(self, page: PageImage) -> list[list[float]]:
        """Run DB detection and return quads in original-page coordinates."""
        cv2 = _cv2()
        session = self._detection_session()
        image = np.asarray(page)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        resized, ratio_h, ratio_w = _resize_for_detection(image, cv2)
        tensor = ((resized.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(
            2, 0, 1
        )[None, ...]
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        probability_map = np.asarray(outputs[0], dtype=np.float32)
        while probability_map.ndim > 2:
            probability_map = probability_map[0]
        boxes = db_postprocess(
            probability_map, ratio_h=ratio_h, ratio_w=ratio_w
        )
        clipped: list[list[float]] = []
        for box in boxes:
            clipped.append(
                [
                    float(
                        min(
                            max(value, 0.0),
                            (width - 1) if position % 2 == 0 else (height - 1),
                        )
                    )
                    for position, value in enumerate(box)
                ]
            )
        return clipped

    def _recognise(
        self, crops: Sequence[CropImage], script: str
    ) -> list[tuple[str, float]]:
        """Batched recognition; results are returned in ``crops`` order."""
        if not crops:
            return []
        session, charset = self._recognition_head(script)
        shapes = [
            (int(np.asarray(crop).shape[0]), int(np.asarray(crop).shape[1]))
            for crop in crops
        ]
        plan = plan_recognition_batches(
            shapes, batch_size=self.settings.OCR_BATCH_SIZE
        )
        decoded: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        input_name = session.get_inputs()[0].name
        for batch in plan:
            tensor = pad_crops_to_batch(crops, batch)
            probabilities = np.asarray(
                session.run(None, {input_name: tensor})[0], dtype=np.float32
            )
            for slot, result in enumerate(
                ctc_greedy_decode(probabilities, charset)
            ):
                decoded[batch.indices[slot]] = result
        return decoded


def _default_fasttext_loader(path: Path) -> Any:
    """Load ``lid.176.ftz`` through ``fasttext-predict`` (MIT wheels).

    ``fasttext-wheel`` in BUILD-BRIEF sec. 1 has no cp312 Windows wheel;
    ``fasttext-predict`` exposes the same ``load_model``/``predict`` surface and
    is already pinned in ``requirements.txt`` (WS-A).
    """
    # fasttext-predict ships no py.typed marker.
    import fasttext  # type: ignore[import-untyped]

    return fasttext.load_model(str(path))


def _softmax(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable softmax over a 1-D vector, idempotent-ish on inputs
    that are already probabilities only up to the usual scaling — callers pass
    logits."""
    array = np.asarray(values, dtype=np.float32)
    shifted = array - array.max()
    exponentiated = np.exp(shifted)
    return np.asarray(exponentiated / exponentiated.sum(), dtype=np.float32)


def _chw(
    image: PageImage, *, size: tuple[int, int] | None = None
) -> NDArray[np.float32]:
    """Normalise an image to a ``(1, 3, H, W)`` float tensor."""
    cv2 = _cv2()
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if size is not None:
        array = cv2.resize(array, size)
    normalised = (array.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None, ...])


def _resize_for_detection(
    image: NDArray[np.uint8], cv2: Any, *, limit_side_len: int = 960
) -> tuple[NDArray[np.uint8], float, float]:
    """Resize so the longest side is ``limit_side_len`` and both sides are /32.

    Returns the resized image plus the ``(ratio_h, ratio_w)`` needed to map
    detections back into original-page coordinates.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    scale = limit_side_len / longest if longest > limit_side_len else 1.0
    target_h = max(32, round(height * scale / 32) * 32)
    target_w = max(32, round(width * scale / 32) * 32)
    resized = cv2.resize(image, (target_w, target_h))
    return resized, height / target_h, width / target_w
