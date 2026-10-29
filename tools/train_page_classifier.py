"""Fit the page-classifier late-fusion coefficients from the committed corpus.

Named by ``app/providers/local.py``'s own ``ModelArtifactMissing`` message ("Train
it with tools/train_page_classifier.py against the committed corpus (WS-E)"), so
this file is where that promise is kept.

What it does
------------
1. Reads ``corpus/index.json`` and each page's ``.gt.json`` for its
   ``document_class`` label.
2. Keeps **variants 0, 1 and 2 only** — variants 3 and 4 are the locked test
   split (BUILD-BRIEF M1) and this model must never have seen them.
3. Drops the ``bounce`` degradation profile: those pages are rejected by the
   quality gate and never reach Stage 3, so training on them would fit the
   classifier to inputs it will never be given.
4. Extracts :class:`~app.contracts.ocr.StructuralFeatures` with
   :func:`app.stages.s3_classify.extract_structural_features` — the *same*
   function inference uses, with ``lines=None``, because Stage 3 runs before
   recognition in the PRD flow.
5. Fits a multinomial logistic regression (softmax, L2, full-batch gradient
   descent with momentum) over the 7 structural features of
   ``app.providers.local.STRUCTURAL_FEATURE_NAMES``, and writes the
   coefficients in the **raw** feature space so ``late_fusion`` needs no
   scaler.
6. Sets every **CNN coefficient slot to exactly 0.0**.  The fit used uniform
   CNN scores, so the structural decision is provably independent of a CNN that
   does not exist yet; when MobileNetV3-Small lands, retraining is what turns
   those zeros into numbers.
7. Writes ``models/classifier/fusion.json``, an identical copy at
   ``docs/calibration/fusion.json`` (so an air-gapped node with an empty
   ``models/`` still classifies), and a human-readable training report beside
   the calibration copy.

Reported accuracies
-------------------
* **train** — refit on variants 0-2, scored on variants 0-2.  This is the number
  ``tests/unit/test_s3_classify.py`` asserts a floor on.
* **holdout** — fit on variants 0-1, scored on variant 2.  Variants differ in
  bank, amounts and complainant, not in layout, so this measures robustness to
  content rather than to document design; it is an optimistic estimate of
  generalisation and the report says so.
* **leave-one-variant-out** — the mean of the three folds, for the same reason.

This tool does NOT handle: training the CNN (there is no approved artifact),
downloading anything, or touching the locked test split.  It is dev tooling and
never ships in the runtime bundle.

Usage::

    .venv/Scripts/python.exe tools/train_page_classifier.py
    .venv/Scripts/python.exe tools/train_page_classifier.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# isort/ruff would sort `app` after `numpy`; it must come first, because
# importing it is what applies the thread caps before NumPy loads its runtime
# (PLAN.md invariant 5). The order is the point, so the rule is suppressed here
# rather than the import being moved.
import app  # noqa: F401,I001  - thread caps before NumPy (PLAN.md invariant 5)
import numpy as np

from app.contracts.ocr import PageClassLabel, StructuralFeatures
from app.providers.local import (
    CLASSIFIER_FUSION_FILENAME,
    CLASSIFIER_SUBDIR,
    STRUCTURAL_FEATURE_NAMES,
    FusionWeights,
    structural_feature_vector,
)
from app.stages.s3_classify import (
    CALIBRATION_FUSION_PATH,
    TRAINED_LABELS,
    extract_structural_features,
    structural_classify,
)

_LOG = logging.getLogger("train_page_classifier")

#: Variants 3 and 4 are the locked test split (BUILD-BRIEF M1).
TRAIN_VARIANTS: Final[frozenset[int]] = frozenset({0, 1, 2})

#: Bounced pages never reach Stage 3.
EXCLUDED_PROFILES: Final[frozenset[str]] = frozenset({"bounce"})

REPORT_FILENAME: Final[str] = "fusion.training-report.json"


@dataclass(frozen=True)
class Sample:
    """One training page: its features, its label and where it came from."""

    doc_id: str
    doc_type: str
    variant: int
    profile: str
    label: PageClassLabel
    features: StructuralFeatures


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def load_samples(
    corpus_dir: Path, *, variants: frozenset[int] = TRAIN_VARIANTS
) -> list[Sample]:
    """Extract features and labels for every eligible corpus page."""
    import cv2

    index_path = corpus_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    for entry in index["documents"]:
        variant = int(entry["variant"])
        profile = str(entry["profile"])
        if variant not in variants or profile in EXCLUDED_PROFILES:
            continue
        gt_path = corpus_dir / entry["gt"]
        image_path = corpus_dir / entry["image"]
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        label = PageClassLabel(str(ground_truth["document_class"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read corpus image {image_path}")
        samples.append(
            Sample(
                doc_id=str(entry["doc_id"]),
                doc_type=str(entry["type"]),
                variant=variant,
                profile=profile,
                label=label,
                features=extract_structural_features(image),
            )
        )
        if len(samples) % 25 == 0:
            _LOG.info("featurised %d pages", len(samples))
    if not samples:
        raise ValueError(f"no eligible pages found under {corpus_dir}")
    return samples


def design_matrix(
    samples: Sequence[Sample], labels: Sequence[PageClassLabel]
) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` — raw structural features and integer class indices."""
    order = {label: index for index, label in enumerate(labels)}
    matrix = np.asarray(
        [structural_feature_vector(sample.features) for sample in samples],
        dtype=np.float64,
    )
    targets = np.asarray(
        [order[sample.label] for sample in samples], dtype=np.int64
    )
    return matrix, targets


# --------------------------------------------------------------------------- #
# Multinomial logistic regression
# --------------------------------------------------------------------------- #


def fit_softmax(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    n_classes: int,
    l2: float = 1e-3,
    iterations: int = 20_000,
    learning_rate: float = 0.5,
    momentum: float = 0.9,
    seed: int = 1729,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit softmax regression, returning coefficients in the **raw** space.

    Features are standardised internally — the raw vector mixes ``log1p`` counts
    with ratios in ``[0, 1]`` and a single learning rate cannot serve both — and
    the fitted weights are mapped back, so ``late_fusion`` never needs a scaler
    and ``fusion.json`` stays a self-contained artifact.

    Deterministic: zero initialisation and full-batch updates, so two runs on
    the same corpus produce byte-identical coefficients.

    Returns:
        ``(weights, intercepts)`` shaped ``(n_features, n_classes)`` and
        ``(n_classes,)``.
    """
    del seed  # no stochasticity to seed; kept so callers can state intent
    n_samples, n_features = matrix.shape
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (matrix - mean) / scale

    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), targets] = 1.0

    weights = np.zeros((n_features, n_classes), dtype=np.float64)
    intercepts = np.zeros(n_classes, dtype=np.float64)
    velocity_w = np.zeros_like(weights)
    velocity_b = np.zeros_like(intercepts)

    for _ in range(iterations):
        probabilities = _softmax_rows(standardised @ weights + intercepts)
        error = probabilities - one_hot
        grad_w = standardised.T @ error / n_samples + l2 * weights
        grad_b = error.mean(axis=0)
        velocity_w = momentum * velocity_w - learning_rate * grad_w
        velocity_b = momentum * velocity_b - learning_rate * grad_b
        weights += velocity_w
        intercepts += velocity_b

    raw_weights = weights / scale[:, None]
    raw_intercepts = intercepts - (mean / scale) @ weights
    return raw_weights, raw_intercepts


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def build_weights(
    raw_weights: np.ndarray,
    raw_intercepts: np.ndarray,
    labels: Sequence[PageClassLabel],
) -> FusionWeights:
    """Assemble a :class:`FusionWeights` with zeroed CNN slots."""
    n_labels = len(labels)
    coefficients: dict[str, tuple[float, ...]] = {}
    intercepts: dict[str, float] = {}
    for index, label in enumerate(labels):
        cnn_slots = [0.0] * n_labels
        structural = [float(value) for value in raw_weights[:, index]]
        coefficients[label.value] = tuple(cnn_slots + structural)
        intercepts[label.value] = float(raw_intercepts[index])
    return FusionWeights(
        labels=tuple(labels), coefficients=coefficients, intercepts=intercepts
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def accuracy(samples: Sequence[Sample], weights: FusionWeights) -> float:
    """Share of ``samples`` whose fused prediction matches the label."""
    if not samples:
        return 0.0
    correct = sum(
        1
        for sample in samples
        if structural_classify(sample.features, weights).label is sample.label
    )
    return correct / len(samples)


def confusion(
    samples: Sequence[Sample], weights: FusionWeights
) -> dict[str, dict[str, int]]:
    """``truth -> predicted -> count``, for the report."""
    matrix: dict[str, dict[str, int]] = {}
    for sample in samples:
        predicted = structural_classify(sample.features, weights).label
        matrix.setdefault(sample.label.value, {})
        row = matrix[sample.label.value]
        row[predicted.value] = row.get(predicted.value, 0) + 1
    return matrix


def per_type_accuracy(
    samples: Sequence[Sample], weights: FusionWeights
) -> dict[str, float]:
    """Accuracy per corpus document type, which is where failures localise."""
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for sample in samples:
        totals[sample.doc_type] = totals.get(sample.doc_type, 0) + 1
        predicted = structural_classify(sample.features, weights).label
        if predicted is sample.label:
            hits[sample.doc_type] = hits.get(sample.doc_type, 0) + 1
    return {
        doc_type: hits.get(doc_type, 0) / count
        for doc_type, count in sorted(totals.items())
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def train(
    corpus_dir: Path,
    *,
    l2: float,
    iterations: int,
) -> tuple[FusionWeights, dict[str, object]]:
    """Fit the shipped weights and build the training report."""
    samples = load_samples(corpus_dir)
    labels = list(TRAINED_LABELS)
    _LOG.info("featurised %d pages across %d classes", len(samples), len(labels))

    matrix, targets = design_matrix(samples, labels)
    raw_weights, raw_intercepts = fit_softmax(
        matrix, targets, n_classes=len(labels), l2=l2, iterations=iterations
    )
    weights = build_weights(raw_weights, raw_intercepts, labels)

    folds: dict[str, float] = {}
    for held_out in sorted(TRAIN_VARIANTS):
        train_split = [s for s in samples if s.variant != held_out]
        test_split = [s for s in samples if s.variant == held_out]
        fold_matrix, fold_targets = design_matrix(train_split, labels)
        fold_w, fold_b = fit_softmax(
            fold_matrix,
            fold_targets,
            n_classes=len(labels),
            l2=l2,
            iterations=iterations,
        )
        folds[f"variant_{held_out}"] = accuracy(
            test_split, build_weights(fold_w, fold_b, labels)
        )

    observed = sorted({sample.label.value for sample in samples})
    report: dict[str, object] = {
        "generator": "tools/train_page_classifier.py",
        "corpus_dir": str(corpus_dir),
        "train_variants": sorted(TRAIN_VARIANTS),
        "excluded_profiles": sorted(EXCLUDED_PROFILES),
        "sample_count": len(samples),
        "labels": [label.value for label in labels],
        "labels_observed_in_corpus": observed,
        "labels_without_training_examples": [
            label.value for label in labels if label.value not in observed
        ],
        "feature_names": list(STRUCTURAL_FEATURE_NAMES),
        "cnn_coefficients": "all zero — fitted with uniform CNN scores",
        "l2": l2,
        "iterations": iterations,
        "class_counts": _counts(sample.label.value for sample in samples),
        "majority_class_baseline": max(
            _counts(sample.label.value for sample in samples).values()
        )
        / len(samples),
        "train_accuracy": accuracy(samples, weights),
        "holdout_accuracy_variant_2": folds["variant_2"],
        "leave_one_variant_out": folds,
        "leave_one_variant_out_mean": sum(folds.values()) / len(folds),
        "per_type_accuracy": per_type_accuracy(samples, weights),
        "confusion_train": confusion(samples, weights),
    }
    return weights, report


def _counts(values: object) -> dict[str, int]:
    """Count occurrences into a plain dict (JSON-safe, ordered by key)."""
    counter: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        counter[str(value)] = counter.get(str(value), 0) + 1
    return dict(sorted(counter.items()))


def write_artifacts(
    weights: FusionWeights,
    report: dict[str, object],
    *,
    models_dir: Path,
    calibration_path: Path,
) -> tuple[Path, Path, Path]:
    """Write ``fusion.json`` twice and the report once.  Returns the paths."""
    payload = json.dumps(
        weights.model_dump(mode="json"), indent=2, ensure_ascii=False
    )
    model_path = models_dir / CLASSIFIER_SUBDIR / CLASSIFIER_FUSION_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(payload + "\n", encoding="utf-8")

    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(payload + "\n", encoding="utf-8")

    report_path = calibration_path.parent / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return model_path, calibration_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus-dir", type=Path, default=PROJECT_ROOT / "corpus"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=PROJECT_ROOT / "models"
    )
    parser.add_argument(
        "--calibration-path", type=Path, default=CALIBRATION_FUSION_PATH
    )
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit and report without writing any artifact.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    weights, report = train(
        args.corpus_dir, l2=args.l2, iterations=args.iterations
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        _LOG.info("dry run: no artifact written")
        return 0

    paths = write_artifacts(
        weights,
        report,
        models_dir=args.models_dir,
        calibration_path=args.calibration_path,
    )
    for path in paths:
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
