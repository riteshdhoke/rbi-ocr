"""Re-run the Stage 1.5 quality gate over the corpus and emit its calibration.

OPEN-QUESTIONS Q3.  ``docs/calibration/quality_gate.json`` is the frozen answer:
the thresholds, the confusion matrix they produce over the 295-page synthetic
corpus, the per-profile score distribution and the adversarial outcomes.  This
script is what produces that file, and it exists because a calibration nobody
can re-run is a number somebody typed.

Usage::

    .venv\\Scripts\\python.exe tools/calibrate_quality_gate.py            # print
    .venv\\Scripts\\python.exe tools/calibrate_quality_gate.py --write    # freeze
    .venv\\Scripts\\python.exe tools/calibrate_quality_gate.py --check    # CI diff
    .venv\\Scripts\\python.exe tools/calibrate_quality_gate.py --sample 3 # every 3rd page

``--check`` compares everything except ``generated_at`` and ``runtime`` — the
two fields that are allowed to move between runs — and exits non-zero on any
other difference, which is the assertion ``tests/golden`` runs on a sample.

This script does NOT handle:
  * choosing the thresholds.  It measures the ones in
    :class:`~app.stages.s15_quality.QualityThresholds`; moving them is a code or
    environment change, and the targets below are what say whether the move was
    an improvement;
  * OCR, or any model.  The quality gate is pure OpenCV and runs offline;
  * the locked test split.  Q3 is a *threshold* question, not a generalisation
    one, so it is calibrated on the whole corpus deliberately — see
    ``docs/decisions-wse1.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `app` must be imported before numpy and cv2 so the thread-cap preamble runs
# first (PLAN.md invariant 5).  Re-sorting this block would break that, hence
# the suppression rather than a tidier arrangement.
import app  # noqa: I001, F401
import cv2
import numpy as np

from app.contracts.errors import PipelineError
from app.stages.s1_ingest import ingest_path
from app.stages.s15_quality import QualityGate, QualityThresholds

CORPUS_DIR = PROJECT_ROOT / "corpus"
ADVERSARIAL_DIR = CORPUS_DIR / "adversarial"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "calibration" / "quality_gate.json"

#: The acceptance criteria from BUILD-BRIEF M3, as machine-readable numbers.
TARGETS: dict[str, Any] = {
    "bounce_recall_min": 0.8,
    "clean_expectation_false_bounce_max": 0.0,
    "note": (
        "a few heavy/photo pages may bounce; they are borderline by design "
        "(TEST-DATA-SPEC sec. 3)"
    ),
}

#: Pages whose quality is assessed one at a time.  Near-duplicate detection is
#: per *document*, and every corpus page is its own document, so a fresh gate
#: per page is the honest configuration — sharing one would flag the five
#: degradation profiles of the same page as duplicates of each other.
_FRESH_GATE_PER_PAGE = True


def load_image(path: Path) -> np.ndarray:
    """Decode a corpus page to BGR, through Stage 1 so the path is the real one."""
    return ingest_path(path).pages[0].image


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, taking the higher of the two neighbours.

    ``method="higher"`` rather than the NumPy default: every number reported in
    ``score_distribution`` is then an *actual page score* that can be traced to
    a ``doc_id`` in the ``pages`` list.  An interpolated p10 belongs to no page,
    which makes "which document is at the tenth percentile of the heavy
    profile?" unanswerable — and that is the first question anyone asks when a
    profile starts drifting.
    """
    return float(np.percentile(values, q, method="higher")) if values else 0.0


def assess_corpus(sample: int) -> dict[str, Any]:
    index = json.loads((CORPUS_DIR / "index.json").read_text(encoding="utf-8"))
    documents = index["documents"][::sample]

    pages: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = {}
    by_profile: dict[str, Counter[str]] = {}
    scores_by_profile: dict[str, list[float]] = {}
    bounce_reasons: Counter[str] = Counter()
    flags_by_profile: dict[str, Counter[str]] = {}

    started = time.perf_counter()
    for document in documents:
        relative = Path(*document["image"].replace("\\", "/").split("/"))
        image = load_image(CORPUS_DIR / relative)
        assessment = QualityGate(QualityThresholds()).evaluate(image, 1).assessment

        profile = document["profile"]
        expectation = document["quality_expectation"]
        verdict = assessment.verdict.value

        confusion.setdefault(expectation, Counter())[verdict] += 1
        by_profile.setdefault(profile, Counter())[verdict] += 1
        scores_by_profile.setdefault(profile, []).append(assessment.quality_score)
        flags_by_profile.setdefault(profile, Counter()).update(
            flag.value for flag in assessment.quality_flags
        )
        if assessment.bounce_reason is not None:
            bounce_reasons[assessment.bounce_reason.value] += 1

        pages.append(
            {
                "doc_id": document["doc_id"],
                "profile": profile,
                "expect": expectation,
                "verdict": verdict,
                "score": round(assessment.quality_score, 4),
                "repairs": list(assessment.repairs_applied),
            }
        )
    elapsed = time.perf_counter() - started

    total = len(pages)
    bounced = sum(1 for page in pages if page["verdict"] == "bounce")
    expected_bounce = confusion.get("bounce", Counter())
    bounce_recall = (
        expected_bounce["bounce"] / sum(expected_bounce.values())
        if sum(expected_bounce.values())
        else 0.0
    )

    def false_bounce(expectation: str) -> float:
        counts = confusion.get(expectation, Counter())
        seen = sum(counts.values())
        return counts["bounce"] / seen if seen else 0.0

    deviations = [
        page
        for page in pages
        if (page["expect"] == "bounce") != (page["verdict"] == "bounce")
    ]

    return {
        "profiles": index["profiles"],
        "seed": index["seed"],
        "generator_version": index["generator_version"],
        "page_count": total,
        "pages": pages,
        "confusion_matrix": {
            "rows": "quality_expectation",
            "columns": "quality_verdict",
            "counts": {
                expectation: dict(counts) for expectation, counts in confusion.items()
            },
        },
        "by_profile": {
            profile: dict(counts) for profile, counts in by_profile.items()
        },
        "score_distribution": {
            profile: {
                "min": round(min(values), 4),
                "p10": round(percentile(values, 10), 4),
                "median": round(percentile(values, 50), 4),
                "max": round(max(values), 4),
            }
            for profile, values in scores_by_profile.items()
        },
        "metrics": {
            "bounce_recall": round(bounce_recall, 4),
            "clean_expectation_false_bounce": round(false_bounce("clean"), 4),
            "degraded_expectation_false_bounce": round(false_bounce("degraded"), 4),
            "overall_bounce_rate": round(bounced / total, 4) if total else 0.0,
            "deviations": deviations,
        },
        "bounce_reasons": dict(bounce_reasons),
        "flags_by_profile": {
            profile: dict(counts) for profile, counts in flags_by_profile.items()
        },
        "runtime": {
            "seconds_total": round(elapsed, 1),
            "ms_per_page": round(elapsed * 1000.0 / total, 1) if total else 0.0,
        },
    }


def assess_adversarial() -> dict[str, Any]:
    """The three adversarial inputs that reach Stage 1.5 at all.

    The other four are refused by Stage 1 with a typed error and never produce
    a quality verdict; ``tests/golden/test_adversarial.py`` is where those live.
    """
    manifest = json.loads(
        (ADVERSARIAL_DIR / "MANIFEST.json").read_text(encoding="utf-8")
    )
    out: dict[str, Any] = {}
    for filename in sorted(manifest["expected_behaviour"]):
        try:
            image = load_image(ADVERSARIAL_DIR / filename)
        except PipelineError:
            continue
        assessment = QualityGate(QualityThresholds()).evaluate(image, 1).assessment
        out[filename] = {
            "verdict": assessment.verdict.value,
            "bounce_reason": (
                assessment.bounce_reason.value
                if assessment.bounce_reason is not None
                else None
            ),
            "score": round(assessment.quality_score, 4),
            "repairs": list(assessment.repairs_applied),
            "flags": [flag.value for flag in assessment.quality_flags],
        }
    return out


def build_report(sample: int = 1) -> dict[str, Any]:
    corpus = assess_corpus(sample)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "resolves": "OPEN-QUESTIONS Q3",
        "corpus": {
            "path": "corpus/",
            "pages": corpus["page_count"],
            "seed": corpus["seed"],
            "generator_version": corpus["generator_version"],
            "profiles": corpus["profiles"],
        },
        "thresholds": QualityThresholds().model_dump(),
        "targets": TARGETS,
        "confusion_matrix": corpus["confusion_matrix"],
        "by_profile": corpus["by_profile"],
        "score_distribution": corpus["score_distribution"],
        "metrics": corpus["metrics"],
        "bounce_reasons": corpus["bounce_reasons"],
        "flags_by_profile": corpus["flags_by_profile"],
        "adversarial": assess_adversarial(),
        "runtime": corpus["runtime"],
        "pages": corpus["pages"],
    }


#: Fields allowed to differ between two runs of the same calibration.
VOLATILE_KEYS = frozenset({"generated_at", "runtime"})


def stable(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in VOLATILE_KEYS}


def check_targets(report: dict[str, Any]) -> list[str]:
    """Return the acceptance criteria this report fails, if any."""
    metrics = report["metrics"]
    targets = report["targets"]
    failures: list[str] = []
    if metrics["bounce_recall"] < targets["bounce_recall_min"]:
        failures.append(
            f"bounce recall {metrics['bounce_recall']} < "
            f"{targets['bounce_recall_min']}"
        )
    if (
        metrics["clean_expectation_false_bounce"]
        > targets["clean_expectation_false_bounce_max"]
    ):
        failures.append(
            f"clean/light false bounce {metrics['clean_expectation_false_bounce']} > "
            f"{targets['clean_expectation_false_bounce_max']}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="overwrite the committed calibration"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed calibration and exit non-zero on drift",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="assess every Nth corpus page (1 = all 295)",
    )
    args = parser.parse_args(argv)
    if args.sample < 1:
        parser.error("--sample must be >= 1")

    cv2.setNumThreads(2)
    report = build_report(args.sample)

    failures = check_targets(report)
    for failure in failures:
        print(f"TARGET MISSED: {failure}", file=sys.stderr)

    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {OUTPUT_PATH}")
    elif args.check:
        committed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if stable(committed) != stable(report):
            print("CALIBRATION DRIFT against " + str(OUTPUT_PATH), file=sys.stderr)
            for key in sorted(stable(committed)):
                if stable(committed)[key] != stable(report).get(key):
                    print(f"  differs: {key}", file=sys.stderr)
            return 1
        print("calibration reproduces the committed file")
    else:
        summary = {
            "pages": report["corpus"]["pages"],
            "confusion_matrix": report["confusion_matrix"]["counts"],
            "metrics": {
                key: value
                for key, value in report["metrics"].items()
                if key != "deviations"
            },
            "deviation_count": len(report["metrics"]["deviations"]),
            "runtime": report["runtime"],
        }
        print(json.dumps(summary, indent=2))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
