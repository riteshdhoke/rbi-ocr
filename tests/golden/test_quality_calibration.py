"""Golden — the Stage 1.5 calibration is real, and still true (OPEN-QUESTIONS Q3).

BUILD-BRIEF M3 states the acceptance criteria as a confusion matrix:

    **Test:** the ``bounce`` profile bounces; ``clean`` and ``light`` accept;
    ``scan`` and ``heavy`` accept-with-repair — measured against
    ``quality_expectation`` in the corpus.

``docs/calibration/quality_gate.json`` is the frozen measurement and
``tools/calibrate_quality_gate.py`` is what produces it.  This file asserts two
different things, and keeping them apart is deliberate:

* **Cheap, always:** the committed report is internally consistent, its
  thresholds are the ones the code actually runs with, and the confusion matrix
  it records meets the targets.  These catch a hand-edited calibration and a
  threshold changed in :class:`~app.stages.s15_quality.QualityThresholds`
  without re-running the gate — the two ways a calibration goes stale.
* **Live, sampled:** the gate is re-run over every :data:`SAMPLE_STRIDE`-th
  corpus page and the *same verdicts and scores* must come back.  A stride
  rather than the whole corpus because 295 pages is 72 seconds; the stride is
  coprime with the six-profile cycle, so the sample walks every profile evenly
  instead of landing on one.  ``tools/calibrate_quality_gate.py --check`` is the
  full-corpus version of this assertion and is what runs before the file is
  re-frozen.

Offline and model-free: the quality gate is pure OpenCV, so nothing here needs
``models/`` and nothing is skipped on a machine that has none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.stages.s15_quality import QualityThresholds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = PROJECT_ROOT / "docs" / "calibration" / "quality_gate.json"
CORPUS_INDEX = PROJECT_ROOT / "corpus" / "index.json"

#: Assess every Nth corpus page in the live check.  7 is coprime with the six
#: degradation profiles, so the sample cycles through all of them rather than
#: repeatedly drawing the same one; 43 pages costs ~11 s against 72 s for 295.
SAMPLE_STRIDE = 7

pytestmark = pytest.mark.skipif(
    not CALIBRATION_PATH.is_file() or not CORPUS_INDEX.is_file(),
    reason="corpus or calibration artefact absent; run tools/generate_test_corpus.py",
)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    """Re-measure a sample of the corpus with the thresholds now in the code."""
    from tools.calibrate_quality_gate import assess_corpus

    return assess_corpus(SAMPLE_STRIDE)


# --------------------------------------------------------------------------- #
# The committed report is a measurement, not a wish
# --------------------------------------------------------------------------- #


def test_the_committed_thresholds_are_the_ones_the_code_runs(
    report: dict[str, Any],
) -> None:
    """A threshold moved without re-running the gate makes the matrix a lie."""
    assert report["thresholds"] == QualityThresholds().model_dump()


def test_the_report_covers_the_whole_corpus(report: dict[str, Any]) -> None:
    index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
    assert report["corpus"]["pages"] == index["count"]
    assert len(report["pages"]) == index["count"]
    assert report["corpus"]["seed"] == index["seed"]
    assert report["corpus"]["generator_version"] == index["generator_version"]
    assert {page["doc_id"] for page in report["pages"]} == {
        document["doc_id"] for document in index["documents"]
    }


def test_the_confusion_matrix_adds_up(report: dict[str, Any]) -> None:
    """Derived from the per-page rows, or it is decoration."""
    counts: dict[str, dict[str, int]] = {}
    by_profile: dict[str, dict[str, int]] = {}
    for page in report["pages"]:
        row = counts.setdefault(page["expect"], {})
        row[page["verdict"]] = row.get(page["verdict"], 0) + 1
        profile_row = by_profile.setdefault(page["profile"], {})
        profile_row[page["verdict"]] = profile_row.get(page["verdict"], 0) + 1
    assert counts == report["confusion_matrix"]["counts"]
    assert by_profile == report["by_profile"]


def test_the_recorded_metrics_meet_the_targets(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    targets = report["targets"]
    assert metrics["bounce_recall"] >= targets["bounce_recall_min"]
    assert (
        metrics["clean_expectation_false_bounce"]
        <= targets["clean_expectation_false_bounce_max"]
    )
    assert metrics["deviations"] == []


def test_every_bounce_carries_a_machine_readable_reason(
    report: dict[str, Any],
) -> None:
    """PRD P1-2 keys the auto-reply template on it."""
    bounced = sum(1 for page in report["pages"] if page["verdict"] == "bounce")
    assert sum(report["bounce_reasons"].values()) == bounced
    assert bounced > 0


def test_the_adversarial_outcomes_are_recorded(report: dict[str, Any]) -> None:
    """Only the three that reach Stage 1.5; the rest die in Stage 1."""
    assert set(report["adversarial"]) == {
        "blank_page.png",
        "inverted.png",
        "tiny_lowres.png",
    }
    assert report["adversarial"]["inverted.png"]["repairs"] == ["auto_invert"]
    assert report["adversarial"]["blank_page.png"]["bounce_reason"] == "BLANK_PAGE"
    assert (
        report["adversarial"]["tiny_lowres.png"]["bounce_reason"]
        == "RESOLUTION_BELOW_FLOOR"
    )


# --------------------------------------------------------------------------- #
# The gate still behaves the way the report says
# --------------------------------------------------------------------------- #


def test_the_bounce_profile_bounces(live: dict[str, Any]) -> None:
    expected = live["confusion_matrix"]["counts"].get("bounce", {})
    seen = sum(expected.values())
    assert seen > 0, "the sample drew no bounce-profile pages"
    recall = expected.get("bounce", 0) / seen
    assert recall >= 0.8, f"bounce recall {recall:.2f} is below the 0.8 target"


def test_clean_and_light_never_false_bounce(live: dict[str, Any]) -> None:
    """A false bounce is a complainant told to resend a perfectly good page."""
    clean = live["confusion_matrix"]["counts"].get("clean", {})
    assert sum(clean.values()) > 0
    assert clean.get("bounce", 0) == 0
    assert set(clean) <= {"accept", "repair"}


@pytest.mark.parametrize("profile", ["scan", "heavy", "photo"])
def test_degraded_profiles_are_accepted_with_repair(
    live: dict[str, Any], profile: str
) -> None:
    verdicts = live["by_profile"].get(profile, {})
    assert sum(verdicts.values()) > 0, f"the sample drew no {profile} pages"
    assert verdicts.get("bounce", 0) == 0
    assert set(verdicts) <= {"accept", "repair"}


def test_the_sample_reproduces_the_committed_verdicts_and_scores(
    report: dict[str, Any], live: dict[str, Any]
) -> None:
    """The strongest form: same page, same verdict, same score to four places.

    ``tools/calibrate_quality_gate.py --check`` is this assertion over all 295
    pages; the stride keeps the suite fast without weakening what is asserted
    per page.
    """
    committed = {page["doc_id"]: page for page in report["pages"]}
    assert live["pages"], "the sample assessed nothing"
    for page in live["pages"]:
        expected = committed[page["doc_id"]]
        assert page["verdict"] == expected["verdict"], page["doc_id"]
        assert page["score"] == pytest.approx(expected["score"], abs=5e-4), (
            page["doc_id"]
        )
        assert page["repairs"] == expected["repairs"], page["doc_id"]


def test_the_sample_walks_every_profile(live: dict[str, Any]) -> None:
    """Guards the stride: a sample that misses `bounce` proves nothing."""
    assert set(live["by_profile"]) == set(live["profiles"])
