"""Stage 4 — Tier-1 orchestration: batching, the bypass path, handwriting.

Three things are load-bearing and are asserted here.

**PLAN.md invariant 4 — recognition is batched, never looped.**  Asserted
twice: once against a mock provider that records the batch plan the stage's
crops produce, and once end to end through :class:`LocalProvider` with fake
ONNX sessions, where the *session* records how many crops each call carried.
A stage that looped over line crops would show 20 calls of 1 instead of 16+4.

**A bypass page never reaches the recogniser.**  A digital PDF's text layer is
better than anything recognition would produce; running it would spend the
budget to make the answer worse.

**The handwriting flag is a promotion, never a demotion.**  Decision C8 left
``is_handwritten`` at ``False`` in the provider and handed the gap here; a
provider that *does* know better must not be overwritten by a heuristic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.contracts.ocr import OCRLine, OCRTier, bbox_from_xyxy
from app.providers.base import ModelArtifactMissing
from app.providers.local import LocalProvider, plan_recognition_batches
from app.settings import Settings
from app.stages import s2_script
from app.stages import s4_ocr as s4_ocr_module
from app.stages.s2_script import DEFAULT_REC_HEAD_SCRIPT
from app.stages.s3_classify import HANDWRITING_SCORE_THRESHOLD
from app.stages.s4_ocr import (
    MAX_RECOGNITION_PASSES,
    PROBE_CROP_HEIGHT,
    PROBE_SHEET_GUTTER,
    Tier1Result,
    annotate_handwriting,
    handwriting_score,
    probe_script_heads,
    run_tier1,
)

ENGLISH = "Name of bank and branch"
MARATHI = "बँकेने माझ्या खात्यातून दोनदा पैसे कापले"


@pytest.fixture
def disjoint_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node whose recognition heads do not overlap, bootstrapping on Latin.

    The shipped Devanagari head covers Latin as well, so it is chosen for
    everything and the "read with a head that cannot represent this script"
    path is unreachable with those two artifacts. That path is still live —
    a Tamil or Bengali head carries no Latin — and it is what the second pass,
    the head probe and the ``head_mismatch`` signal exist for, so the tests
    below pin the mechanism in the configuration that reaches it.
    """
    disjoint = {"Latn": frozenset({"Latn"}), "Deva": frozenset({"Deva"})}
    monkeypatch.setattr(s2_script, "REC_HEAD_COVERAGE", disjoint)
    monkeypatch.setattr(s2_script, "DEFAULT_REC_HEAD_SCRIPT", "Latn")
    monkeypatch.setattr(s4_ocr_module, "DEFAULT_REC_HEAD_SCRIPT", "Latn")


def _line(
    text: str,
    script: str,
    index: int,
    *,
    y: float | None = None,
    height: float = 30.0,
    handwritten: bool = False,
) -> OCRLine:
    top = 10.0 + 40.0 * index if y is None else y
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(10.0, top, 260.0, top + height),
        confidence=0.9,
        script=script,
        language=None,
        is_handwritten=handwritten,
        line_index=index,
    )


class _ScriptedProvider:
    """Provider stub: one canned line set per recognition-head script.

    Also records the batch plan its crops *would* produce, so the stage's
    call pattern can be checked against PLAN.md invariant 4 without a model.
    """

    def __init__(
        self,
        responses: dict[str, list[OCRLine]],
        *,
        batch_size: int = 16,
        crop_shape: tuple[int, int] = (48, 100),
    ) -> None:
        self.responses = responses
        self.batch_size = batch_size
        self.crop_shape = crop_shape
        self.calls: list[str] = []
        self.batch_sizes: list[int] = []

    def ocr_page(self, page: Any, script: str) -> list[OCRLine]:
        self.calls.append(script)
        lines = self.responses.get(script, [])
        plan = plan_recognition_batches(
            [self.crop_shape] * len(lines), batch_size=self.batch_size
        )
        self.batch_sizes.extend(len(batch.indices) for batch in plan)
        return list(lines)


def _white_page(height: int = 400, width: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _glyph_band(
    page: np.ndarray,
    top: int,
    *,
    jitter: bool,
    glyphs: int = 12,
    seed: int = 7,
) -> None:
    """Paint a line of glyph-like blocks; ``jitter`` makes them hand-like."""
    rng = np.random.default_rng(seed)
    x = 16
    for _ in range(glyphs):
        glyph_height = 16 + (int(rng.integers(-7, 8)) if jitter else 0)
        baseline = top + 30 + (int(rng.integers(-4, 5)) if jitter else 0)
        page[baseline - glyph_height : baseline, x : x + 10] = 0
        x += 18


# --------------------------------------------------------------------------- #
# PLAN.md invariant 4 — batched recognition
# --------------------------------------------------------------------------- #


def test_the_stage_calls_the_recogniser_once_per_pass_not_once_per_line(
    settings: Settings,
) -> None:
    """A loop over line crops would show 20 calls here instead of one."""
    lines = [_line(ENGLISH, "Latn", i) for i in range(20)]
    # Head-agnostic: this is about call *count*, so key on whichever head the
    # stage bootstraps with rather than pinning a script here.
    provider = _ScriptedProvider({DEFAULT_REC_HEAD_SCRIPT: lines})

    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.calls == [DEFAULT_REC_HEAD_SCRIPT]
    assert result.rec_passes == (DEFAULT_REC_HEAD_SCRIPT,)
    assert len(result.lines) == 20
    # 20 crops at OCR_BATCH_SIZE 16 is two padded calls, never twenty.
    assert provider.batch_sizes == [16, 4]


def test_recognition_stays_batched_through_the_stage_end_to_end(
    disjoint_heads: None,
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """Invariant 4 at the session boundary, driven by the stage itself.

    Fake ONNX sessions, so this passes on a machine with an empty ``models/``:
    the artifacts below are zero-byte placeholders that only the injected
    session factory ever opens.
    """
    charset = ["a", "b", "c"]
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "det.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn.onnx").write_bytes(b"")
    (ocr_dir / "rec_latn_dict.txt").write_text("\n".join(charset), encoding="utf-8")

    class _FakeSession:
        def __init__(self, produce: Callable[[np.ndarray], np.ndarray]) -> None:
            self._produce = produce
            self.batch_sizes: list[int] = []

        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="x")]

        def get_outputs(self) -> list[Any]:
            # Symbolic class axis so the use_space_char reconciliation stays off.
            return [SimpleNamespace(name="y", shape=[None, None, "C"])]

        def run(self, _outputs: Any, feed: dict[str, np.ndarray]) -> list[Any]:
            tensor = next(iter(feed.values()))
            self.batch_sizes.append(int(tensor.shape[0]))
            return [self._produce(tensor)]

    def detection(tensor: np.ndarray) -> np.ndarray:
        height, width = tensor.shape[2], tensor.shape[3]
        probability = np.zeros((1, 1, height, width), dtype=np.float32)
        for index in range(20):
            top = 10 + index * 40
            probability[0, 0, top : top + 20, 20 : width - 20] = 0.95
        return probability

    def recognition(tensor: np.ndarray) -> np.ndarray:
        count = tensor.shape[0]
        array = np.full((count, 4, len(charset) + 1), 0.01, dtype=np.float32)
        array[:, 0, 1] = 0.9  # 'a'
        array[:, 1, 0] = 0.9  # blank
        array[:, 2, 2] = 0.9  # 'b'
        array[:, 3, 0] = 0.9  # blank
        return array

    sessions = {"det": _FakeSession(detection), "rec": _FakeSession(recognition)}
    provider = LocalProvider(
        make_settings(MODELS_DIR=tmp_path, OCR_BATCH_SIZE=16),
        session_factory=lambda path: sessions[
            "det" if path.name == "det.onnx" else "rec"
        ],
    )

    result = run_tier1(
        np.full((900, 600, 3), 255, dtype=np.uint8),
        provider=provider,
        page_no=1,
        settings=make_settings(MODELS_DIR=tmp_path, OCR_BATCH_SIZE=16),
        detect_handwriting=False,
    )

    assert len(result.lines) == 20
    # Batched, not looped: a handful of padded calls rather than one per crop.
    # The exact split follows the REC_WIDTH_BUCKETS edges these crop widths land
    # on, so it is asserted as a bound rather than a literal — pinning [16, 4]
    # made a change to the bucket ladder look like a batching regression.
    batches = sessions["rec"].batch_sizes
    assert sum(batches) == 20
    assert len(batches) <= 4
    assert max(batches) > 1
    assert sessions["det"].batch_sizes == [1]


def test_a_page_with_no_detected_lines_is_recorded_as_no_tier(
    settings: Settings,
) -> None:
    provider = _ScriptedProvider({"Latn": []})
    result = run_tier1(
        _white_page(), provider=provider, page_no=3, settings=settings
    )
    assert result.page.tier is OCRTier.NONE
    assert result.lines == []
    assert result.handwriting_measured is False


def test_a_page_tier1_read_nothing_from_is_handed_to_the_router_as_unsettled(
    settings: Settings,
) -> None:
    """Deliberate: no text is no script evidence, and the router escalates it.

    A page detection found nothing on is exactly the page a second reader
    exists for, so Stage 4 reports the disagreement rather than shipping a
    silent "Latin, nothing on it".
    """
    result = run_tier1(
        _white_page(),
        provider=_ScriptedProvider({"Latn": []}),
        page_no=1,
        settings=settings,
    )
    assert result.script_disagreement is True
    assert result.script is not None
    assert result.script.disagreement_reason == "no_script_evidence"


# --------------------------------------------------------------------------- #
# Adjudication A-6 — the two-pass script bootstrap
# --------------------------------------------------------------------------- #


def test_a_devanagari_page_read_with_the_latin_head_is_read_again(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """The wrong-head case, on a node whose heads do not cover each other.

    With the shipped artifacts the stage bootstraps on the Devanagari head and
    this page is read correctly first time; ``disjoint_heads`` is what makes a
    Latin bootstrap — and therefore the second pass — reachable.
    """
    latin_pass = [_line(MARATHI, "Deva", i) for i in range(5)]
    devanagari_pass = [_line(MARATHI, "Deva", i) for i in range(5)]
    provider = _ScriptedProvider({"Latn": latin_pass, "Deva": devanagari_pass})

    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.calls == ["Latn", "Deva"]
    assert result.rec_passes == ("Latn", "Deva")
    assert result.script_code == "Deva"
    # After the re-read, the vote and the head agree: nothing left to escalate.
    assert result.script_disagreement is False


def test_a_page_whose_vote_agrees_with_its_head_pays_nothing_for_a6(
    settings: Settings,
) -> None:
    provider = _ScriptedProvider(
        {DEFAULT_REC_HEAD_SCRIPT: [_line(ENGLISH, "Latn", i) for i in range(5)]}
    )
    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )
    # The vote says Latin and the bootstrap head covers Latin, so there is
    # nothing to disagree about and no second pass to pay for.
    assert provider.calls == [DEFAULT_REC_HEAD_SCRIPT]
    assert result.rec_passes == (DEFAULT_REC_HEAD_SCRIPT,)
    assert result.script_disagreement is False


def test_a_single_pass_run_reports_the_disagreement_it_did_not_act_on(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """``max_passes=1`` is the benchmark tool's mode; the signal still ships."""
    provider = _ScriptedProvider(
        {"Latn": [_line(MARATHI, "Deva", i) for i in range(5)]}
    )
    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        max_passes=1,
        detect_handwriting=False,
    )
    assert provider.calls == ["Latn"]
    assert result.script_disagreement is True
    assert result.script is not None
    assert result.script.disagreement_reason == "head_mismatch"


def test_a_script_hint_is_the_bootstrap_head_for_pass_one(
    settings: Settings,
) -> None:
    provider = _ScriptedProvider(
        {"Deva": [_line(MARATHI, "Deva", i) for i in range(5)]}
    )
    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        script_hint="Deva",
        detect_handwriting=False,
    )
    assert provider.calls == ["Deva"]
    assert result.rec_passes == ("Deva",)


def test_a_page_is_never_read_more_than_twice() -> None:
    assert MAX_RECOGNITION_PASSES == 2


def test_lines_leave_this_stage_without_a_language(settings: Settings) -> None:
    """PLAN.md invariant 1: only Stage 2b may label a line."""
    provider = _ScriptedProvider(
        {"Latn": [_line(ENGLISH, "Latn", i) for i in range(3)]}
    )
    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )
    assert all(line.language is None for line in result.lines)


# --------------------------------------------------------------------------- #
# The bypass path — a digital text layer skips recognition entirely
# --------------------------------------------------------------------------- #


def test_a_bypass_page_never_calls_the_recogniser(settings: Settings) -> None:
    provider = _ScriptedProvider({"Latn": [_line("never asked", "Latn", 0)]})
    bypass = [_line(ENGLISH, "Latn", i) for i in range(4)]

    result = run_tier1(
        None,
        provider=provider,
        page_no=2,
        settings=settings,
        bypass_lines=bypass,
        width_px=1240,
        height_px=1754,
    )

    assert provider.calls == []
    assert result.page.tier is OCRTier.BYPASS
    assert result.rec_passes == ()
    assert [line.text for line in result.lines] == [ENGLISH] * 4


def test_a_bypass_page_is_not_asserted_to_be_printed(settings: Settings) -> None:
    """A digital text layer has no strokes; "no handwriting" would be a claim."""
    result = run_tier1(
        None,
        provider=_ScriptedProvider({}),
        page_no=1,
        settings=settings,
        bypass_lines=[_line(ENGLISH, "Latn", 0)],
        width_px=800,
        height_px=1000,
    )
    assert result.handwriting_measured is False
    assert result.handwriting_ratio == pytest.approx(0.0)


def test_a_bypass_page_still_votes_on_its_script(settings: Settings) -> None:
    """The text layer's characters are the best script evidence there is."""
    result = run_tier1(
        None,
        provider=_ScriptedProvider({}),
        page_no=1,
        settings=settings,
        bypass_lines=[_line(MARATHI, "Deva", i) for i in range(4)],
        width_px=800,
        height_px=1000,
    )
    assert result.script_code == "Deva"


def test_an_empty_bypass_page_has_no_tier_and_no_script(
    settings: Settings,
) -> None:
    result = run_tier1(
        None,
        provider=_ScriptedProvider({}),
        page_no=1,
        settings=settings,
        bypass_lines=[],
        width_px=800,
        height_px=1000,
    )
    assert result.page.tier is OCRTier.NONE
    assert result.script is None
    assert result.script_code is None
    assert result.script_disagreement is False


def test_a_bypass_page_carries_its_page_geometry_through(
    settings: Settings,
) -> None:
    result = run_tier1(
        None,
        provider=_ScriptedProvider({}),
        page_no=7,
        settings=settings,
        bypass_lines=[_line(ENGLISH, "Latn", 0)],
        width_px=1240,
        height_px=1754,
        rotation_applied_deg=-1.5,
        dpi_estimate=200.0,
    )
    assert result.page.page_no == 7
    assert (result.page.width_px, result.page.height_px) == (1240, 1754)
    assert result.page.rotation_applied_deg == pytest.approx(-1.5)
    assert result.page.dpi_estimate == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# Handwriting — the C8 gap, closed as policy in this stage
# --------------------------------------------------------------------------- #


def test_the_score_separates_a_jittered_hand_from_machine_set_print() -> None:
    printed = _white_page(60, 260)
    _glyph_band(printed, 0, jitter=False)
    hand = _white_page(60, 260)
    _glyph_band(hand, 0, jitter=True)

    printed_score = handwriting_score(printed)
    hand_score = handwriting_score(hand)

    assert printed_score < HANDWRITING_SCORE_THRESHOLD <= hand_score
    assert hand_score > printed_score


def test_an_unmeasurable_crop_scores_zero_rather_than_guessing() -> None:
    assert handwriting_score(np.full((8, 8, 3), 255, dtype=np.uint8)) == 0.0
    assert handwriting_score(_white_page(60, 260)) == 0.0


def test_annotate_promotes_a_handwritten_line_and_reports_the_ratio() -> None:
    page = _white_page(200, 300)
    _glyph_band(page, 10, jitter=True)
    _glyph_band(page, 110, jitter=False)
    lines = [
        _line("hand", "Latn", 0, y=10.0, height=50.0),
        _line("print", "Latn", 1, y=110.0, height=50.0),
    ]

    annotated, ratio = annotate_handwriting(page, lines)

    assert [line.is_handwritten for line in annotated] == [True, False]
    assert ratio == pytest.approx(0.5)


def test_a_provider_supplied_flag_is_never_erased_by_the_heuristic() -> None:
    """The heuristic can promote a line to handwritten; it can never demote."""
    page = _white_page(200, 300)
    _glyph_band(page, 10, jitter=False)
    lines = [_line("print", "Latn", 0, y=10.0, height=50.0, handwritten=True)]

    annotated, ratio = annotate_handwriting(page, lines)

    assert annotated[0].is_handwritten is True
    assert ratio == pytest.approx(1.0)


def test_annotating_returns_new_objects_and_leaves_the_input_alone() -> None:
    page = _white_page(200, 300)
    _glyph_band(page, 10, jitter=True)
    lines = [_line("hand", "Latn", 0, y=10.0, height=50.0)]

    annotated, _ratio = annotate_handwriting(page, lines)

    assert annotated[0] is not lines[0]
    assert lines[0].is_handwritten is False


def test_a_degenerate_box_is_a_detector_artefact_not_a_document_fact() -> None:
    """A zero-area box must not take the page down with it."""
    page = _white_page(200, 300)
    degenerate = OCRLine(
        text="x",
        bbox=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        confidence=0.5,
        script="Latn",
        language=None,
        is_handwritten=False,
        line_index=0,
    )
    annotated, ratio = annotate_handwriting(page, [degenerate])
    assert annotated[0].is_handwritten is False
    assert ratio == pytest.approx(0.0)


def test_annotating_no_lines_measures_nothing() -> None:
    assert annotate_handwriting(_white_page(), []) == ([], 0.0)


def test_the_stage_measures_handwriting_when_asked_and_records_that_it_did(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    page = _white_page(200, 300)
    _glyph_band(page, 10, jitter=True)
    provider = _ScriptedProvider(
        {"Latn": [_line(ENGLISH, "Latn", 0, y=10.0, height=50.0)]}
    )

    measured = run_tier1(
        page, provider=provider, page_no=1, settings=settings
    )
    unmeasured = run_tier1(
        page,
        provider=_ScriptedProvider(
            {"Latn": [_line(ENGLISH, "Latn", 0, y=10.0, height=50.0)]}
        ),
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert measured.handwriting_measured is True
    assert measured.handwriting_ratio == pytest.approx(1.0)
    assert measured.lines[0].is_handwritten is True
    assert unmeasured.handwriting_measured is False
    assert unmeasured.lines[0].is_handwritten is False


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_recognition_without_pixels_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="needs page pixels"):
        run_tier1(
            None,
            provider=_ScriptedProvider({}),
            page_no=1,
            settings=settings,
            width_px=100,
            height_px=100,
        )


def test_page_dimensions_are_required_when_there_are_no_pixels(
    settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="page dimensions are required"):
        run_tier1(
            None,
            provider=_ScriptedProvider({}),
            page_no=1,
            settings=settings,
            bypass_lines=[_line(ENGLISH, "Latn", 0)],
        )


@pytest.mark.parametrize("passes", [0, -1])
def test_a_non_positive_pass_count_is_refused(
    passes: int, settings: Settings
) -> None:
    with pytest.raises(ValueError, match="max_passes must be >= 1"):
        run_tier1(
            _white_page(),
            provider=_ScriptedProvider({}),
            page_no=1,
            settings=settings,
            max_passes=passes,
        )


def test_negative_page_dimensions_are_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="page dimensions must be positive"):
        run_tier1(
            None,
            provider=_ScriptedProvider({}),
            page_no=1,
            settings=settings,
            bypass_lines=[_line(ENGLISH, "Latn", 0)],
            width_px=0,
            height_px=10,
        )


def test_dimensions_are_taken_from_the_array_when_not_supplied(
    settings: Settings,
) -> None:
    provider = _ScriptedProvider(
        {"Latn": [_line(ENGLISH, "Latn", 0)]}
    )
    result = run_tier1(
        _white_page(height=480, width=320),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )
    assert (result.page.width_px, result.page.height_px) == (320, 480)


def test_a_one_dimensional_page_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="must be 2-D or 3-D"):
        run_tier1(
            np.zeros((10,), dtype=np.uint8),
            provider=_ScriptedProvider({}),
            page_no=1,
            settings=settings,
        )


# --------------------------------------------------------------------------- #
# Result-model conveniences
# --------------------------------------------------------------------------- #


def test_the_result_exposes_lines_and_script_without_reaching_into_the_page(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    lines: Sequence[OCRLine] = [_line(ENGLISH, "Latn", i) for i in range(3)]
    provider = _ScriptedProvider({"Latn": list(lines)})
    result = run_tier1(
        _white_page(),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )
    assert isinstance(result, Tier1Result)
    assert result.lines == list(result.page.lines)
    assert result.script_code == "Latn"


# --------------------------------------------------------------------------- #
# OPEN-QUESTIONS Q12 — the dual-head script probe
#
# The defect these cover: head *selection* used to be a vote over one head's
# own transcription, so a Devanagari page recognised by the Latin head voted
# "Latn" on Latin noise and the Devanagari head was never called
# (docs/eval/REPORT-2026-08-08-slmlive.md §2).  Every test below drives the probe with a provider
# that answers differently for the full page and for the probe sheet, which is
# the only way to reproduce that asymmetry without weights.
# --------------------------------------------------------------------------- #

#: What the Latin head does to Devanagari pixels, transcribed from the M7
#: report's own table (docs/eval/REPORT-2026-08-08-slmlive.md §2): short, low-confidence Latin.
LATIN_NOISE = ("h cicc  chr", "TT HR 2", "y-d, H6RI")

_SHEET_HEIGHT = PROBE_CROP_HEIGHT + 2 * PROBE_SHEET_GUTTER


def _probe_line(text: str, script: str, index: int, confidence: float) -> OCRLine:
    """A line as it comes back from the probe sheet — one row, side by side."""
    left = 10.0 + 200.0 * index
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(left, 8.0, left + 160.0, 8.0 + PROBE_CROP_HEIGHT),
        confidence=confidence,
        script=script,
        language=None,
        is_handwritten=False,
        line_index=index,
    )


def _probe_reading(
    texts: Sequence[str], script: str, confidence: float
) -> list[OCRLine]:
    return [
        _probe_line(text, script, index, confidence)
        for index, text in enumerate(texts)
    ]


class _TwoSurfaceProvider:
    """Answers one way for the page and another for the probe sheet.

    The sheet is recognisable by its height — :func:`probe_script_heads` builds
    exactly one row of ``PROBE_CROP_HEIGHT``-tall crops — so no test has to
    reach into the stage's internals to know which surface it is being asked
    about.  ``page_calls`` is what the recogniser was actually paid for.
    """

    def __init__(
        self,
        page: dict[str, list[OCRLine]],
        sheet: dict[str, list[OCRLine]],
        *,
        missing: Sequence[str] = (),
    ) -> None:
        self.page = page
        self.sheet = sheet
        self.missing = set(missing)
        self.page_calls: list[str] = []
        self.sheet_calls: list[str] = []
        self.sheet_shapes: list[tuple[int, ...]] = []

    def ocr_page(self, page: Any, script: str) -> list[OCRLine]:
        if script in self.missing:
            raise ModelArtifactMissing(
                artifact=f"PP-OCRv5 recognition head for script {script!r}",
                path=Path("models/ocr") / f"rec_{script.lower()}.onnx",
                how_to_obtain="Run tools/fetch_models.py.",
            )
        if np.asarray(page).shape[0] == _SHEET_HEIGHT:
            self.sheet_calls.append(script)
            self.sheet_shapes.append(tuple(np.asarray(page).shape))
            return list(self.sheet.get(script, []))
        self.page_calls.append(script)
        return list(self.page.get(script, []))


def _inked_page(
    lines: Sequence[OCRLine], *, height: int = 400, width: int = 300
) -> np.ndarray:
    """A page carrying ink inside every line box, so probe crops are usable.

    The probe drops crops with no ink on purpose — two heads cannot be compared
    on blank paper — so a fixture that wants the probe to run has to draw
    something.
    """
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for line in lines:
        xs = [int(value) for value in line.bbox[0::2]]
        ys = [int(value) for value in line.bbox[1::2]]
        left, right = max(0, min(xs)), min(width, max(xs))
        top, bottom = max(0, min(ys)), min(height, max(ys))
        for column in range(left + 2, right - 2, 6):
            page[top + 4 : bottom - 4, column : column + 3] = 0
    return page


def _devanagari_page_fixture() -> tuple[np.ndarray, _TwoSurfaceProvider]:
    """The Q12 page: Latin head reads noise, Devanagari head reads the text."""
    noise = [_line(LATIN_NOISE[index % 3], "Latn", index) for index in range(6)]
    good = [_line(MARATHI, "Deva", index) for index in range(6)]
    provider = _TwoSurfaceProvider(
        page={"Latn": noise, "Deva": good},
        sheet={
            "Latn": _probe_reading(LATIN_NOISE, "Latn", 0.53),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.94),
        },
    )
    return _inked_page(noise), provider


def test_the_probe_picks_the_head_that_reads_the_pixels_not_the_head_that_ran(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """The Q12 regression test, in one assertion set.

    Pass one is Latin and its text is Latin noise, so the A-6 vote confirms
    ``Latn`` — that is the bug.  The probe makes both heads read the *same*
    crops, the Devanagari head wins on confidence, and the page is re-read.
    """
    page, provider = _devanagari_page_fixture()

    result = run_tier1(
        page,
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.page_calls == ["Latn", "Deva"]
    assert provider.sheet_calls == ["Latn", "Deva"]
    assert result.rec_head == "Deva"
    assert result.script_code == "Deva"
    assert [line.text for line in result.lines] == [MARATHI] * 6

    probe = result.script_probe
    assert probe is not None
    assert probe.reason == "score_margin"
    assert probe.script == "Deva"
    assert probe.head == "Deva"


def test_the_probe_scores_every_head_it_tried_and_records_them(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """``script_probe`` is observability: a human must be able to check it."""
    page, provider = _devanagari_page_fixture()

    result = run_tier1(
        page,
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    probe = result.script_probe
    assert probe is not None
    assert [entry.head for entry in probe.scores] == ["Latn", "Deva"]
    assert probe.score_for("Deva") == pytest.approx(0.94)
    assert probe.score_for("Latn") == pytest.approx(0.53)
    assert probe.score_for("Deva") > probe.score_for("Latn")
    assert probe.sampled_lines >= 1
    assert probe.unavailable_heads == ()
    assert probe.elapsed_ms >= 0.0
    # Weighted by script-bearing characters, so a digit-only line cannot vote.
    latin = next(entry for entry in probe.scores if entry.head == "Latn")
    assert latin.script_characters == sum(
        sum(1 for c in text if c.isalpha()) for text in LATIN_NOISE
    )


def test_a_latin_page_is_not_handed_to_a_devanagari_head_that_reads_it_better(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """The no-regression guard, and the reason the rule is not "highest score".

    PP-OCRv5's Devanagari head has a Latin-covering charset and does read
    English marginally more confidently than the Latin head.  Selecting on
    confidence alone would therefore move every English page onto the wrong
    head.  Both heads read this sheet *as Latin*, so they agree, and agreement
    settles it before any confidence is compared.
    """
    english = [_line(ENGLISH, "Latn", index) for index in range(5)]
    provider = _TwoSurfaceProvider(
        page={"Latn": english, "Deva": english},
        sheet={
            "Latn": _probe_reading([ENGLISH] * 3, "Latn", 0.95),
            # Higher confidence, same script: the Devanagari head reading Latin.
            "Deva": _probe_reading([ENGLISH] * 3, "Latn", 0.99),
        },
    )

    result = run_tier1(
        _inked_page(english),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.page_calls == ["Latn"]
    assert result.rec_head == "Latn"
    assert result.script_probe is not None
    assert result.script_probe.reason == "heads_agree"
    assert result.script_probe.script == "Latn"
    assert result.script_disagreement is False


def test_a_head_that_decodes_almost_nothing_is_evidence_not_silence(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """The rule that ``complaint_letter_mar`` needed.

    On a Marathi letter the Latin head returns four characters of rubble from
    three crops — too few to score — while the Devanagari head returns
    twenty-six.  Requiring *two* speakers here would throw the comparison away
    and strand the page on the Latin head, which is the defect.  One head
    failing to decode the same crops the other read cleanly **is** the
    comparison.
    """
    noise = [_line("H, ", "Latn", index) for index in range(4)]
    good = [_line(MARATHI, "Deva", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Latn": noise, "Deva": good},
        sheet={
            "Latn": _probe_reading(["$,", "HT"], "Latn", 0.55),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.92),
        },
    )

    result = run_tier1(
        _inked_page(noise),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    probe = result.script_probe
    assert probe is not None
    assert probe.reason == "single_head_decoded"
    assert probe.script == "Deva"
    latin = next(entry for entry in probe.scores if entry.head == "Latn")
    assert latin.usable is False
    assert result.rec_head == "Deva"
    assert provider.page_calls == ["Latn", "Deva"]


def test_a_head_whose_artifact_is_absent_is_recorded_and_stepped_over(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """``ModelArtifactMissing`` is a deployment fact, never a page failure."""
    english = [_line(ENGLISH, "Latn", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Latn": english},
        sheet={"Latn": _probe_reading([ENGLISH] * 3, "Latn", 0.95)},
        missing=["Deva"],
    )

    result = run_tier1(
        _inked_page(english),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    probe = result.script_probe
    assert probe is not None
    assert probe.unavailable_heads == ("Deva",)
    assert probe.reason == "no_probe_evidence"
    assert probe.head is None
    # One head cannot corroborate itself, so the page keeps today's behaviour.
    assert provider.page_calls == ["Latn"]
    assert result.rec_head == "Latn"


def test_a_single_head_install_never_builds_a_probe_sheet(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """Requirement of the fix: one installed head keeps the old code path."""
    english = [_line(ENGLISH, "Latn", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Latn": english},
        sheet={"Latn": _probe_reading([ENGLISH] * 3, "Latn", 0.95)},
    )

    result = run_tier1(
        _inked_page(english),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
        probe_scripts=("Latn",),
    )

    assert provider.sheet_calls == []
    assert result.script_probe is not None
    assert result.script_probe.reason == "too_few_heads"
    assert provider.page_calls == ["Latn"]


def test_an_upstream_prior_breaks_a_probe_that_would_otherwise_be_a_coin_flip(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """The prior's only job: two heads, two scripts, no meaningful gap."""
    noise = [_line(LATIN_NOISE[index % 3], "Latn", index) for index in range(4)]
    good = [_line(MARATHI, "Deva", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Latn": noise, "Deva": good},
        sheet={
            "Latn": _probe_reading(LATIN_NOISE, "Latn", 0.80),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.81),
        },
    )

    result = run_tier1(
        _inked_page(noise),
        provider=provider,
        page_no=1,
        settings=settings,
        script_prior="Deva",
        detect_handwriting=False,
    )

    probe = result.script_probe
    assert probe is not None
    assert probe.reason == "prior_breaks_tie"
    assert probe.prior == "Deva"
    assert result.rec_head == "Deva"


def test_a_prior_seeds_the_bootstrap_head_so_the_probe_costs_nothing(
    settings: Settings,
) -> None:
    """A decisive prior should mean the *first* pass is already right."""
    good = [_line(MARATHI, "Deva", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Deva": good},
        sheet={
            "Latn": _probe_reading(LATIN_NOISE, "Latn", 0.53),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.94),
        },
    )

    result = run_tier1(
        _inked_page(good),
        provider=provider,
        page_no=1,
        settings=settings,
        script_prior="Deva",
        detect_handwriting=False,
    )

    assert provider.page_calls == ["Deva"]
    assert result.rec_head == "Deva"
    assert result.rec_passes == ("Deva",)


def test_a_tie_with_nothing_to_break_it_leaves_the_bootstrap_head_alone(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """Inconclusive is a reportable outcome, never a guess (BUILD-BRIEF 0.5)."""
    noise = [_line(LATIN_NOISE[index % 3], "Latn", index) for index in range(4)]
    provider = _TwoSurfaceProvider(
        page={"Latn": noise, "Deva": [_line(MARATHI, "Deva", 0)]},
        sheet={
            "Latn": _probe_reading(LATIN_NOISE, "Latn", 0.80),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.82),
        },
    )

    result = run_tier1(
        _inked_page(noise),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    probe = result.script_probe
    assert probe is not None
    assert probe.reason == "inconclusive_tie"
    assert probe.head is None
    assert provider.page_calls == ["Latn"]


def test_the_a6_vote_can_overrule_the_probe_without_a_third_provider_call(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """A wrong probe is undone for free, and the router hears about it.

    The probe sends the page to the Devanagari head; that head's full-page
    output turns out to be Latin, so the confirmation vote contradicts it.  The
    stage reverts to the transcription pass one already produced — two provider
    calls, not three — and still reports the disagreement, because a page whose
    script two mechanisms argue about is exactly what Tier 2 is for.
    """
    latin_pass = [_line(ENGLISH, "Latn", index) for index in range(5)]
    deva_pass = [_line(ENGLISH, "Latn", index) for index in range(5)]
    provider = _TwoSurfaceProvider(
        page={"Latn": latin_pass, "Deva": deva_pass},
        sheet={
            "Latn": _probe_reading(LATIN_NOISE, "Latn", 0.55),
            "Deva": _probe_reading([MARATHI] * 3, "Deva", 0.95),
        },
    )

    result = run_tier1(
        _inked_page(latin_pass),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.page_calls == ["Latn", "Deva"]
    assert len(provider.page_calls) <= MAX_RECOGNITION_PASSES
    assert result.rec_passes == ("Latn", "Deva")
    assert result.rec_head == "Latn"
    assert result.script_code == "Latn"
    assert result.script_disagreement is True


def test_a_page_the_a6_vote_already_rerouted_is_not_probed(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """Pay for the probe only where the old mechanism was circular."""
    provider = _TwoSurfaceProvider(
        page={
            "Latn": [_line(MARATHI, "Deva", index) for index in range(5)],
            "Deva": [_line(MARATHI, "Deva", index) for index in range(5)],
        },
        sheet={},
    )
    lines = provider.page["Latn"]

    result = run_tier1(
        _inked_page(lines),
        provider=provider,
        page_no=1,
        settings=settings,
        detect_handwriting=False,
    )

    assert provider.sheet_calls == []
    assert result.script_probe is None
    assert provider.page_calls == ["Latn", "Deva"]


def test_a_single_pass_run_does_not_pay_for_an_answer_it_cannot_use(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """``max_passes=1`` is the benchmark tool's mode — no probe, no re-read."""
    page, provider = _devanagari_page_fixture()

    result = run_tier1(
        page,
        provider=provider,
        page_no=1,
        settings=settings,
        max_passes=1,
        detect_handwriting=False,
    )

    assert provider.sheet_calls == []
    assert provider.page_calls == ["Latn"]
    assert result.script_probe is None


def test_the_probe_can_be_turned_off_for_an_a_b_measurement(
    disjoint_heads: None,
    settings: Settings,
) -> None:
    """``probe_heads=False`` restores the pre-Q12 path exactly."""
    page, provider = _devanagari_page_fixture()

    result = run_tier1(
        page,
        provider=provider,
        page_no=1,
        settings=settings,
        probe_heads=False,
        detect_handwriting=False,
    )

    assert provider.sheet_calls == []
    assert result.script_probe is None
    assert result.rec_head == "Latn"


def test_a_blank_page_yields_no_probe_crops_rather_than_a_guess(
    settings: Settings,
) -> None:
    """Two heads cannot be compared on paper with nothing on it."""
    lines = [_line(ENGLISH, "Latn", index) for index in range(4)]
    provider = _TwoSurfaceProvider(page={"Latn": lines}, sheet={})

    probe = probe_script_heads(
        _white_page(), lines, provider=provider
    )

    assert probe.reason == "no_probe_crops"
    assert probe.head is None
    assert probe.sampled_lines == 0
    assert provider.sheet_calls == []


def test_the_probe_prefers_a_line_with_letters_over_a_bare_identifier() -> None:
    """A Devanagari form's account number reads identically on both heads.

    Sampling it would spend the probe's whole budget on a crop that cannot
    discriminate, so lines whose bootstrap text carries no letter are set aside.
    Observable here as the width of the sheet the provider was handed: the
    digits line is three times wider, and it is not the one that was used.
    """
    letters = OCRLine(
        text="Bank",
        bbox=bbox_from_xyxy(10.0, 10.0, 90.0, 50.0),
        confidence=0.9,
        script="Latn",
        language=None,
        is_handwritten=False,
        line_index=0,
    )
    digits = OCRLine(
        text="77592400665811",
        bbox=bbox_from_xyxy(10.0, 60.0, 290.0, 100.0),
        confidence=0.99,
        script="Latn",
        language=None,
        is_handwritten=False,
        line_index=1,
    )
    provider = _TwoSurfaceProvider(
        page={},
        sheet={"Latn": _probe_reading(["Bank"], "Latn", 0.9)},
    )

    probe = probe_script_heads(
        _inked_page([letters, digits]),
        [letters, digits],
        provider=provider,
        line_count=1,
        # Uncapped, so the sheet's width is the chosen crop's own width and the
        # two candidates cannot produce the same measurement.
        max_crop_width=4096,
    )

    assert probe.sampled_lines == 1
    assert provider.sheet_calls == ["Latn", "Deva"]
    gutters = 2 * PROBE_SHEET_GUTTER
    letters_width = round(80 * PROBE_CROP_HEIGHT / 40) + gutters
    digits_width = round(280 * PROBE_CROP_HEIGHT / 40) + gutters
    assert provider.sheet_shapes[0][1] == letters_width != digits_width


# --------------------------------------------------------------------------- #
# Page orientation (the signal ROTATION_DETECTED was always meant to carry)
# --------------------------------------------------------------------------- #


def _quad(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """An axis-aligned box as the 8-point quad the detector returns."""
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _boxed(quad: Sequence[float], index: int = 0) -> OCRLine:
    return OCRLine(
        text="x",
        bbox=list(quad),
        confidence=0.9,
        script="Latn",
        language=None,
        is_handwritten=False,
        line_index=index,
    )


class TestPageOrientation:
    """Measured on the committed corpus: one clean page per document type, each
    at 0/90/180/270 degrees.  Every upright page scored 1.000 and every sideways
    page 0.000, on all ten types — which is why the threshold sits at 0.5."""

    def test_a_wide_box_reads_across_the_page(self) -> None:
        assert s4_ocr_module.box_long_edge_is_horizontal(_quad(10, 10, 400, 40)) is True

    def test_a_tall_box_reads_down_the_page(self) -> None:
        assert s4_ocr_module.box_long_edge_is_horizontal(_quad(10, 10, 40, 400)) is False

    def test_a_sub_pixel_box_does_not_vote(self) -> None:
        """A stray mark has no orientation and must not be counted as evidence."""
        assert s4_ocr_module.box_long_edge_is_horizontal(_quad(10, 10, 12, 12)) is None

    def test_an_upright_page_is_not_sideways(self) -> None:
        lines = [_boxed(_quad(10, 10 + 50 * i, 400, 40 + 50 * i), i) for i in range(8)]
        assert s4_ocr_module.horizontal_box_share(lines) == 1.0
        assert s4_ocr_module.page_is_sideways(lines) is False

    def test_a_quarter_turned_page_is_sideways(self) -> None:
        lines = [_boxed(_quad(10 + 50 * i, 10, 40 + 50 * i, 400), i) for i in range(8)]
        assert s4_ocr_module.horizontal_box_share(lines) == 0.0
        assert s4_ocr_module.page_is_sideways(lines) is True

    def test_a_page_with_no_usable_boxes_makes_no_claim(self) -> None:
        """Absence of evidence is not evidence of rotation."""
        assert s4_ocr_module.horizontal_box_share([]) is None
        assert s4_ocr_module.page_is_sideways([]) is False

    def test_a_few_tall_boxes_do_not_turn_an_upright_page(self) -> None:
        """Forms have vertical rules and stamps; the majority still decides."""
        lines = [_boxed(_quad(10, 10 + 50 * i, 400, 40 + 50 * i), i) for i in range(8)]
        lines.append(_boxed(_quad(500, 10, 530, 400), 8))
        assert s4_ocr_module.page_is_sideways(lines) is False
