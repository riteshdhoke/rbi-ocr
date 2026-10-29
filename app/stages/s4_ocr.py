"""Stage 4 — Tier-1 OCR orchestration: detection, batched recognition, routing.

What this stage owns, and what it deliberately does not:

* **It calls ``provider.ocr_page(page, script)`` once per recognition pass.**
  Detection, aspect-ratio bucketing, padding and the 16-32 crop batches of
  PLAN.md **invariant 4** all live inside the provider
  (:func:`app.providers.local.plan_recognition_batches`), as does the LRU/TTL
  recognition-head cache of BUILD-BRIEF M4.  There is no loop over line crops
  anywhere in this module, and ``tests/unit/test_s4_ocr.py`` asserts the batch
  sizes the ONNX session actually sees.
* **It selects the recognition head from the pixels, then confirms with the
  text.**  Adjudication A-6 makes the script a function of the recognised text,
  and the text a function of the head — a loop that closes on itself.  The M7
  gate measured what that costs (``docs/eval/REPORT-2026-08-08-slmlive.md`` §2, OPEN-QUESTIONS
  **Q12**): every page bootstrapped on the Latin head, the A-6 vote ran on the
  Latin head's own output, and a Devanagari page voted ``Latn`` *confidently*
  because Latin noise is made of Latin codepoints.  The head was never
  reselected and Devanagari CER sat at 0.904 with weights that read the same
  page at CER 0.036.

  Head **selection** therefore no longer trusts one head's self-output.  After
  the bootstrap pass, :func:`probe_script_heads` re-reads a handful of that
  pass's own line crops with *every installed recognition head* and compares
  them — see its docstring for why that is not circular.  The A-6 text vote
  stays exactly where it was, as the **confirmation** step over the winning
  head's full-page output; when the vote contradicts the probe the stage falls
  back to the other head **once**, reusing a transcription it already holds, so
  there is no loop and no third provider call.  ``rec_passes`` records the
  provider calls, ``rec_head`` names the head that produced the final text,
  ``script_probe`` carries the probe's scores for observability, and
  ``script_disagreement`` is handed to the escalation router
  (:mod:`app.stages.s4_router`).
* **It skips OCR entirely for bypass-tier pages.**  A digital PDF's text layer
  (Stage 1b) is already better than anything recognition would produce, and
  running the recogniser over it would burn the budget to make the answer
  worse.  Pass the bypass lines in and the page comes back with
  ``tier=OCRTier.BYPASS`` and no provider call.
* **It sets ``is_handwritten`` per line** with a documented structural
  heuristic.  Decision C8 left the field at ``False`` in the provider for want
  of an approved artifact and handed the gap to WS-E; this is where it is
  closed, in the stage rather than the provider, so it works with any profile
  the factory composes.  The pixel statistic itself is
  :func:`app.stages.s3_classify.handwriting_score` — it is a structural feature
  and Stage 3 fits a classifier coefficient to it, so there is exactly one
  implementation and the classifier and the line flag can never disagree.  This
  module owns the *policy*: the threshold, the per-line flag, and the rule that
  a provider-supplied flag is never overwritten.

This module does NOT handle:
  * **language** (Stage 2b) — lines come out of here with ``language=None``,
    exactly as the provider emits them (PLAN.md invariant 1);
  * the escalation decision.  This stage *produces signals*; the rule that
    combines them is :mod:`app.stages.s4_router`, and nothing here reads
    ``OCRLine.confidence`` to route (invariant 3);
  * Tier-2 (:mod:`app.stages.s4b_vlm`);
  * the quality gate, deskew or repair (Stage 1.5, WS-E1) — a page arrives
    already repaired, and ``rotation_applied_deg`` is passed through, not
    computed;
  * page-level script or language keys — ``OCRPage`` refuses them and this
    module never builds one with them.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field

from app.contracts.ocr import OCRLine, OCRPage, OCRTier, ScriptCode
from app.providers.base import ModelArtifactMissing, ModelProvider, PageImage
from app.providers.local import REC_IMAGE_HEIGHT, crop_perspective, script_of_char
from app.settings import Settings
from app.stages.s2_script import (
    DEFAULT_REC_HEAD_SCRIPT,
    REC_HEAD_SCRIPTS,
    ScriptDecision,
    identify_script,
    rec_head_for_script,
    subsumes_heads,
)
from app.stages.s3_classify import (
    HANDWRITING_SCORE_THRESHOLD,
    handwriting_score,
)

__all__ = [
    "HANDWRITING_SCORE_THRESHOLD",
    "MAX_RECOGNITION_PASSES",
    "PROBE_CROP_HEIGHT",
    "PROBE_CROP_MAX_WIDTH",
    "PROBE_HEAD_SCRIPTS",
    "PROBE_LINE_COUNT",
    "PROBE_MIN_INK_STD",
    "PROBE_MIN_SCRIPT_CHARACTERS",
    "PROBE_SCORE_MARGIN",
    "PROBE_SHEET_GUTTER",
    "HeadScore",
    "ScriptProbe",
    "Tier1Result",
    "annotate_handwriting",
    "handwriting_score",
    "probe_script_heads",
    "run_tier1",
]

_LOG = logging.getLogger(__name__)

#: A page is re-recognised at most once.  Two passes close the A-6 bootstrap
#: gap; a third would be a loop chasing a head that does not exist.
MAX_RECOGNITION_PASSES: Final[int] = 2

# --------------------------------------------------------------------------- #
# Dual-head script probe — the Q12 tunables
#
# Module constants rather than ``Settings`` fields, matching Stage 2's
# precedent: ``app/settings.py`` is another workstream's committed lane, every
# value below is overridable per call, and the measurements that fixed each one
# are in ``docs/decisions-q12.md``.
# --------------------------------------------------------------------------- #

#: Recognition-head scripts the probe tries, in order.  These are the two heads
#: ``tools/fetch_models.py`` ships; any code in
#: :data:`~app.stages.s2_script.REC_HEAD_SCRIPTS` is legal, and a head whose
#: artifacts are absent is skipped rather than fatal (see
#: :func:`probe_script_heads`).
PROBE_HEAD_SCRIPTS: Final[tuple[str, ...]] = ("Latn", "Deva")

#: Line crops sampled for the probe: one per vertical band, longest first.
#: Measured over the 147 recognisable dev-split documents, three crops score
#: 146/147 against the corpus's ``primary_script`` while two score 141/147 —
#: two bands are too easily filled by a Devanagari form's Latin-script
#: identifier fields (``docs/decisions-q12.md`` D-Q12-4).
PROBE_LINE_COUNT: Final[int] = 3

#: Probe crops are scaled to the recogniser's own input height, so the width cap
#: below is measured in the units the rec batch planner buckets on.
PROBE_CROP_HEIGHT: Final[int] = REC_IMAGE_HEIGHT

#: Each probe crop is truncated to this many pixels **after** the height
#: normalisation.  Recognition cost is linear in padded crop area, so this is
#: the single knob that sets the probe's price: 160 px is one rec width bucket
#: and holds 5-10 glyphs, which is more than enough to tell two scripts apart.
PROBE_CROP_MAX_WIDTH: Final[int] = 160

#: Background margin around and between the crops on the probe sheet, so the
#: detector sees separated lines rather than one smear.
PROBE_SHEET_GUTTER: Final[int] = 8

#: A crop whose pixel standard deviation is below this carries no ink, and two
#: heads cannot be compared on blank paper.  Such crops are dropped; a probe
#: with no crops left is inconclusive, never a guess.
PROBE_MIN_INK_STD: Final[float] = 8.0

#: A head that decoded fewer script-bearing characters than this has not said
#: enough to be scored — the same "did enough evidence speak?" gate Stage 2
#: applies with :data:`~app.stages.s2_script.MIN_VOTING_CHARACTERS`.
PROBE_MIN_SCRIPT_CHARACTERS: Final[int] = 6

#: When two heads read the same crops as *different* scripts, the winner must
#: lead by this much confidence.  Measured separation on the dev split is 0.09
#: at the very worst and 0.20-0.40 typically, so this rejects coin-flips
#: without rejecting any real decision (``docs/decisions-q12.md`` D-Q12-5).
PROBE_SCORE_MARGIN: Final[float] = 0.05

_OCR_PAGES = Counter(
    "stage_s4_ocr_pages_total",
    "Pages through Stage 4, by tier and number of recognition passes.",
    ["tier", "passes"],
)
_OCR_LINES = Histogram(
    "stage_s4_ocr_lines_per_page",
    "Lines recognised per page.",
    buckets=(0, 1, 5, 10, 20, 40, 80, 160, 320),
)
_OCR_LATENCY = Histogram(
    "stage_s4_ocr_seconds", "Wall time of one Stage-4 Tier-1 pass."
)
_SECOND_PASS = Counter(
    "stage_s4_ocr_second_pass_total",
    "Pages re-recognised because the script vote disagreed with the head "
    "that produced the text.",
    ["from_script", "to_script"],
)
_SCRIPT_PROBES = Counter(
    "stage_s4_ocr_script_probes_total",
    "Dual-head script probes, by how they resolved and what they selected.",
    ["reason", "script"],
)
_PROBE_LATENCY = Histogram(
    "stage_s4_ocr_script_probe_seconds",
    "Wall time of one dual-head script probe, sheet construction included.",
)
_PROBE_CONFLICTS = Counter(
    "stage_s4_ocr_script_probe_conflicts_total",
    "Pages where the A-6 text vote contradicted the probe's head selection.",
    ["probe_script", "voted_script"],
)


class HeadScore(BaseModel):
    """One recognition head's reading of the probe sheet, scored.

    ``score`` is the head's mean decode confidence **weighted by
    script-bearing characters**.  Digits and punctuation are weighted zero
    deliberately: every head reads ``77592400665811`` at confidence 1.000, so
    letting account numbers into the average buys nothing and dilutes the one
    thing being measured.  A head that decoded no letters at all scores 0.0 and
    cannot win.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ISO 15924 code of the head that produced this reading.
    head: ScriptCode
    #: Script-character-weighted mean decode confidence, in [0, 1].
    score: float = Field(ge=0.0, le=1.0)
    #: Majority script of *this head's own output*; ``None`` when it decoded no
    #: script-bearing characters.  For a wide-charset head this is the probe's
    #: actual evidence — see :func:`probe_script_heads`.
    voted_script: str | None = None
    #: Script-bearing characters this head decoded (the score's weight).
    script_characters: int = Field(default=0, ge=0)
    #: All characters decoded, script-bearing or not.
    decoded_characters: int = Field(default=0, ge=0)
    #: Lines the detector found on the probe sheet for this head.
    lines: int = Field(default=0, ge=0)

    @property
    def usable(self) -> bool:
        """Whether this head said enough to take part in the decision."""
        return (
            self.voted_script is not None
            and self.script_characters >= PROBE_MIN_SCRIPT_CHARACTERS
        )


class ScriptProbe(BaseModel):
    """What the dual-head probe saw, and what it concluded.

    Carried on :attr:`Tier1Result.script_probe` for observability: the eval
    harness and the review UI both need to know *why* a page was routed to a
    head, and "the probe scored Latn 0.74 and Deva 0.93 on the same three
    crops" is an answer a human can check.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Per-head readings, in the order the heads were tried.
    scores: tuple[HeadScore, ...] = ()
    #: The script the probe settled on, or ``None`` when it was inconclusive.
    script: str | None = None
    #: The recognition head :data:`script` routes to; ``None`` when inconclusive.
    head: str | None = None
    #: Machine-readable how: ``heads_agree``, ``single_head_decoded``,
    #: ``score_margin``, ``prior_breaks_tie``, ``inconclusive_tie``,
    #: ``no_probe_crops``, ``too_few_heads``, ``no_probe_evidence``.
    reason: str = "no_probe_evidence"
    #: Line crops that made it onto the probe sheet.
    sampled_lines: int = Field(default=0, ge=0)
    #: Heads asked for whose artifacts are not installed on this node.
    unavailable_heads: tuple[str, ...] = ()
    #: Upstream script evidence handed in as a tie-breaker, if any.
    prior: str | None = None
    #: Wall time of the whole probe, sheet construction included.
    elapsed_ms: float = Field(default=0.0, ge=0.0)

    @property
    def conclusive(self) -> bool:
        """True when the probe selected a head."""
        return self.head is not None

    def score_for(self, head: str) -> float | None:
        """This head's score, or ``None`` when it was not probed."""
        for entry in self.scores:
            if entry.head == head:
                return entry.score
        return None


class Tier1Result(BaseModel):
    """Everything Stage 4 produced for one page, including its router signals.

    ``page`` is the contract object downstream stages consume; the rest is
    working evidence that the envelope does not carry but the router, the eval
    harness and the review UI all need.
    """

    model_config = ConfigDict(extra="forbid")

    page: OCRPage
    #: The Stage-2 decision that selected the head, or ``None`` for a bypass
    #: page (nothing was recognised, so nothing voted).
    script: ScriptDecision | None = None
    #: Head scripts the **provider was called with**, in order.  Length 1
    #: normally, 2 after a re-run, 0 for a bypass page.  A head that appears
    #: here was paid for; a head reused from an earlier pass does not reappear.
    rec_passes: tuple[str, ...] = ()
    #: The head whose transcription is in ``page.lines``.  Usually
    #: ``rec_passes[-1]``, but not when the A-6 confirmation vote sent the page
    #: back to a transcription an earlier pass had already produced.  ``None``
    #: for a bypass page, which no head recognised.
    rec_head: str | None = None
    #: What the dual-head probe saw before the head was chosen (Q12).  ``None``
    #: when the probe did not run: a bypass page, a single-head install, a page
    #: the A-6 vote already re-routed, or ``probe_heads=False``.
    script_probe: ScriptProbe | None = None
    #: Router signal: the voted script disagreed with the head, the vote was not
    #: decisive, or the probe and the vote contradicted each other (PLAN.md
    #: invariant 3 — a fact about disagreement, never a confidence number).
    script_disagreement: bool = False
    #: Share of lines the heuristic called handwritten, in [0, 1].
    handwriting_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: True when the handwriting heuristic could not run (no page pixels, or a
    #: bypass page), so consumers can tell "no handwriting" from "not measured".
    handwriting_measured: bool = False

    @property
    def lines(self) -> list[OCRLine]:
        """Shorthand for ``page.lines``."""
        return list(self.page.lines)

    @property
    def script_code(self) -> str | None:
        """Voted ISO 15924 script, or ``None`` on a bypass page."""
        return None if self.script is None else self.script.script


# --------------------------------------------------------------------------- #
# Handwriting detection — the heuristic decision C8 handed to WS-E
# --------------------------------------------------------------------------- #


def annotate_handwriting(
    page: PageImage,
    lines: Sequence[OCRLine],
    *,
    threshold: float = HANDWRITING_SCORE_THRESHOLD,
) -> tuple[list[OCRLine], float]:
    """Set ``is_handwritten`` per line from :func:`handwriting_score`.

    The flag is **OR-ed** with whatever the provider already set: a provider
    configured with its own ``handwriting_detector`` (the hook decision C8 left
    open) has information this heuristic does not, and a stage must not be able
    to erase it.  Monotone by construction — this can promote a line to
    handwritten, never demote one.

    Crops are rectified out of the page with the provider's own
    :func:`app.providers.local.crop_perspective`, so a skewed line is measured
    upright rather than measured through its own rotation.

    Returns:
        ``(lines, handwriting_ratio)`` — new line objects, never mutated ones.
    """
    if not lines:
        return [], 0.0
    array = np.asarray(page)
    annotated: list[OCRLine] = []
    handwritten = 0
    for line in lines:
        try:
            crop = crop_perspective(array, line.bbox)
            score = handwriting_score(crop)
        except (ValueError, IndexError) as exc:
            # A degenerate box is a detector artefact, not a document fact.
            _LOG.debug(
                "handwriting score skipped for line %d: %s", line.line_index, exc
            )
            score = 0.0
        flag = line.is_handwritten or score >= threshold
        handwritten += int(flag)
        annotated.append(
            line
            if flag == line.is_handwritten
            else line.model_copy(update={"is_handwritten": flag})
        )
    return annotated, handwritten / len(lines)


# --------------------------------------------------------------------------- #
# The dual-head script probe — OPEN-QUESTIONS Q12
# --------------------------------------------------------------------------- #

_CV2_LOCK = threading.Lock()
_CV2_MODULE: Any = None


#: Share of detected line boxes whose long edge runs across the page below which
#: the page is judged to be lying on its side.  Measured on the committed corpus
#: (one clean page per document type, each at 0/90/180/270 degrees): every
#: upright page scored **1.000** and every sideways page **0.000**, with no
#: intermediate value on any of the ten types.  The threshold sits in the middle
#: of a gap that wide because the signal is not a heuristic about ink — the text
#: detector returns *line*-shaped quads, and a line of text is wide.
#:
#: This is the measurement ``s15_quality._orientation_anisotropy`` could not
#: make from pixel projections, where "a form's column alignment is as strong as
#: its line spacing".  The difference is that this runs *after* detection, so it
#: reads the detector's own opinion of where the text is rather than trying to
#: infer it from the raster.
SIDEWAYS_HORIZONTAL_SHARE: Final[float] = 0.5

#: Boxes shorter than this on their long edge are detector noise — a stray mark
#: has no meaningful orientation and must not vote on the page's.
_MIN_ORIENTED_BOX_PX: Final[float] = 8.0


def box_long_edge_is_horizontal(bbox: Sequence[float]) -> bool | None:
    """Whether a line box's long edge runs across the page rather than down it.

    ``None`` for a degenerate or sub-pixel box, which is a detector artefact and
    not evidence about the page.
    """
    points = [(float(bbox[i]), float(bbox[i + 1])) for i in range(0, 8, 2)]
    # Two adjacent sides of the quad; the longer one is the reading direction.
    side_a = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    side_b = (points[2][0] - points[1][0], points[2][1] - points[1][1])
    len_a = math.hypot(*side_a)
    len_b = math.hypot(*side_b)
    long_side, long_len = (side_a, len_a) if len_a >= len_b else (side_b, len_b)
    if long_len < _MIN_ORIENTED_BOX_PX:
        return None
    return abs(long_side[0]) >= abs(long_side[1])


def horizontal_box_share(lines: Sequence[OCRLine]) -> float | None:
    """Share of line boxes reading across the page.  ``None`` when nothing votes."""
    verdicts = [
        verdict
        for line in lines
        if (verdict := box_long_edge_is_horizontal(line.bbox)) is not None
    ]
    if not verdicts:
        return None
    return sum(verdicts) / len(verdicts)


def page_is_sideways(lines: Sequence[OCRLine]) -> bool:
    """Whether this page appears to be rotated a quarter turn.

    Reports 90 **and** 270 without distinguishing them, and cannot see 180 at
    all: a page turned upside down still has horizontal lines.  That is a real
    limit of this signal, not an omission — separating those cases needs
    recognition, not geometry.

    Nothing here rotates anything.  The page is flagged and escalated, the same
    treatment :mod:`app.stages.s15_quality` gives perspective, because turning a
    page the wrong way is strictly worse than leaving it alone and a quarter-turn
    signal does not say which way to turn.
    """
    share = horizontal_box_share(lines)
    return share is not None and share < SIDEWAYS_HORIZONTAL_SHARE


def _cv2() -> Any:
    """Import OpenCV on first use, for the same reason Stage 3 does.

    ``import cv2`` costs a few hundred milliseconds and the API process never
    touches pixels; importing this stage must stay cheap (decision C20).
    """
    global _CV2_MODULE
    with _CV2_LOCK:
        if _CV2_MODULE is None:
            import cv2

            _CV2_MODULE = cv2
        return _CV2_MODULE


def probe_script_heads(
    page: PageImage,
    lines: Sequence[OCRLine],
    *,
    provider: ModelProvider,
    scripts: Sequence[str] = PROBE_HEAD_SCRIPTS,
    prior: str | None = None,
    line_count: int = PROBE_LINE_COUNT,
    max_crop_width: int = PROBE_CROP_MAX_WIDTH,
    margin: float = PROBE_SCORE_MARGIN,
) -> ScriptProbe:
    """Decide the page's script by making **every installed head** read the
    same crops, and comparing them.

    Why this is not the circular thing A-6 warned about
    ---------------------------------------------------
    The A-6 vote asks *"what script is this text in?"* of text one head
    produced.  A Latin head physically cannot emit Devanagari codepoints, so on
    a Devanagari page its answer is ``Latn`` by construction and the vote
    confirms the head that produced it.  That is the Q12 defect.

    This function asks a different question — *"which head reads these pixels
    better, and what does that head say the pixels are?"* — and it is
    answerable because the two heads are **not symmetric**.  PP-OCRv5's
    Devanagari head ships a charset covering Devanagari *and* Latin (128 + 52
    characters in ``rec_deva_dict.txt``), so its transcription is a genuine
    vote: handed an English page it emits Latin and says ``Latn``, handed a
    Devanagari page it emits Devanagari and says ``Deva``.  The Latin head,
    whose charset holds one Devanagari codepoint, can only ever say ``Latn``.
    The decision rule below is built around exactly that asymmetry:

    * **the heads agree** on the script of what they read — settled, no
      confidence comparison needed.  This is the common case (122 of 147
      dev-split documents) and it is why an English page cannot regress: both
      heads read it as Latin, and the fact that the Devanagari head sometimes
      reads English *marginally* more confidently never gets a vote;
    * **only one head decoded enough to vote** — the other read the same crops
      and returned rubble too short to score.  That is a comparison, not
      silence, and it is the shape a Devanagari letter takes;
    * **the heads disagree** — one of them is transcribing noise, and noise
      decodes at low confidence.  The higher-scoring head wins, but only by at
      least ``margin``; otherwise the probe says so rather than flipping a
      coin.  On the dev split the winning margin here was 0.09 at its
      narrowest and 0.20-0.40 typically;
    * **still tied** — ``prior`` (upstream, non-recognition evidence such as a
      discarded PDF text layer) breaks it, and if there is none the probe
      returns inconclusive and the caller keeps the bootstrap head.

    The sheet is a single row of at most ``line_count`` crops, each rectified
    out of the page, scaled to the recogniser's input height and truncated to
    ``max_crop_width``.  Recognition cost is linear in padded crop area, so
    that truncation — not the crop count — is what keeps the probe at ~17% of a
    full page's OCR time (``docs/decisions-q12.md`` §4).

    Args:
        page: the same pixels the bootstrap pass recognised.
        lines: that pass's output; only their **boxes** are used, never their
            text, except to prefer crops that carry letters over crops that
            carry only digits.
        scripts: candidate heads.  Codes outside
            :data:`~app.stages.s2_script.REC_HEAD_SCRIPTS` are ignored — there
            is no ``rec_zyyy.onnx`` to ask for.
        prior: upstream script evidence, used only to break an exact tie.

    Returns:
        A :class:`ScriptProbe`.  It is **never** an exception path: a missing
        head artifact, a blank page, a detector that found nothing on the sheet
        and a genuine tie all come back as an inconclusive probe carrying the
        reason, because the caller's fallback (keep the bootstrap head) is
        always available and always safe.
    """
    started = time.perf_counter()
    candidates: list[str] = []
    for script in scripts:
        head = str(script)
        if head in REC_HEAD_SCRIPTS and head not in candidates:
            candidates.append(head)

    def finish(probe: ScriptProbe) -> ScriptProbe:
        elapsed = (time.perf_counter() - started) * 1000.0
        settled = probe.model_copy(update={"elapsed_ms": elapsed})
        _PROBE_LATENCY.observe(elapsed / 1000.0)
        _SCRIPT_PROBES.labels(
            reason=settled.reason, script=settled.script or "none"
        ).inc()
        return settled

    if len(candidates) < 2:
        # One head cannot disagree with itself; probing it would reproduce Q12.
        return finish(ScriptProbe(reason="too_few_heads", prior=prior))

    sheet, sampled = _probe_sheet(
        page, lines, line_count=line_count, max_crop_width=max_crop_width
    )
    if sheet is None:
        return finish(
            ScriptProbe(reason="no_probe_crops", prior=prior, sampled_lines=0)
        )

    readings: list[HeadScore] = []
    unavailable: list[str] = []
    for head in candidates:
        try:
            probe_lines = list(provider.ocr_page(sheet, head))
        except ModelArtifactMissing as missing:
            # A single-head node is a supported deployment, not a failure.
            unavailable.append(head)
            _LOG.debug(
                "script probe skipped the %s head: %s is not installed",
                head,
                missing.artifact,
            )
            continue
        readings.append(_score_head(head, probe_lines))

    probe = _decide_probe(
        readings,
        prior=prior,
        margin=margin,
        sampled_lines=sampled,
        unavailable=tuple(unavailable),
    )
    _LOG.debug(
        "script probe crops=%d %s -> %s (%s)",
        sampled,
        {entry.head: round(entry.score, 3) for entry in probe.scores},
        probe.head,
        probe.reason,
    )
    return finish(probe)


def _score_head(head: str, lines: Sequence[OCRLine]) -> HeadScore:
    """Score one head's reading of the probe sheet — see :class:`HeadScore`."""
    weighted = 0.0
    script_characters = 0
    decoded = 0
    counts: dict[str, int] = {}
    for line in lines:
        letters = 0
        for character in line.text:
            script = script_of_char(character)
            if script is not None:
                counts[script] = counts.get(script, 0) + 1
                letters += 1
        decoded += len(line.text)
        script_characters += letters
        weighted += line.confidence * letters
    voted = (
        max(counts, key=lambda key: (counts[key], key)) if counts else None
    )
    return HeadScore(
        head=head,
        score=(weighted / script_characters) if script_characters else 0.0,
        voted_script=voted,
        script_characters=script_characters,
        decoded_characters=decoded,
        lines=len(lines),
    )


def _decide_probe(
    readings: Sequence[HeadScore],
    *,
    prior: str | None,
    margin: float,
    sampled_lines: int,
    unavailable: tuple[str, ...],
) -> ScriptProbe:
    """Apply the agree / margin / prior rule to the scored heads."""
    scores = tuple(readings)

    def settled(reason: str, script: str | None = None) -> ScriptProbe:
        return ScriptProbe(
            scores=scores,
            script=script,
            head=None if script is None else rec_head_for_script(script),
            reason=reason,
            sampled_lines=sampled_lines,
            unavailable_heads=unavailable,
            prior=prior,
        )

    if len(scores) < 2:
        # Only one head *read the sheet at all*.  That is not a comparison, it
        # is the circularity this function exists to break, so it decides
        # nothing — a node with one installed head keeps the old behaviour.
        return settled("no_probe_evidence")

    usable = [entry for entry in scores if entry.usable]
    if not usable:
        return settled("no_probe_evidence")

    if len(usable) == 1:
        # The other head read the *same* crops and could not decode enough of
        # them to vote.  That is evidence, not silence, and it is the shape a
        # Devanagari letter takes: the Latin head returns four characters of
        # rubble where the Devanagari head returns twenty-six
        # (``docs/decisions-q12.md`` D-Q12-6).  Requiring two speakers here is
        # what made this probe fail on ``complaint_letter_mar``.
        return settled("single_head_decoded", str(usable[0].voted_script))

    voted = {str(entry.voted_script) for entry in usable}
    if len(voted) == 1:
        return settled("heads_agree", voted.pop())

    ranked = sorted(usable, key=lambda entry: (-entry.score, entry.head))
    if ranked[0].score - ranked[1].score >= margin:
        return settled("score_margin", str(ranked[0].voted_script))
    if prior is not None and any(entry.voted_script == prior for entry in usable):
        return settled("prior_breaks_tie", prior)
    return settled("inconclusive_tie")


def _probe_sheet(
    page: PageImage,
    lines: Sequence[OCRLine],
    *,
    line_count: int,
    max_crop_width: int,
) -> tuple[PageImage | None, int]:
    """Lay the sampled crops out as one small image both heads can read.

    Returns ``(sheet, crop_count)``; ``(None, 0)`` when nothing usable was
    found, which is a normal outcome on a blank or near-blank page.
    """
    if line_count < 1:
        raise ValueError(f"line_count must be >= 1, got {line_count}")
    if max_crop_width < 1:
        raise ValueError(f"max_crop_width must be >= 1, got {max_crop_width}")
    array = np.asarray(page)
    crops: list[np.ndarray] = []
    for line in _sample_probe_lines(lines, line_count):
        crop = _probe_crop(array, line, max_crop_width=max_crop_width)
        if crop is not None:
            crops.append(crop)
    if not crops:
        return None, 0

    gutter = PROBE_SHEET_GUTTER
    width = sum(crop.shape[1] for crop in crops) + gutter * (len(crops) + 1)
    height = PROBE_CROP_HEIGHT + 2 * gutter
    # Shaped from the crops, not from the page: a (H, W, 1) page comes back
    # from the resize as 2-D, and a sheet shaped from the page would then
    # refuse the assignment below.
    shape = (height, width, *crops[0].shape[2:])
    sheet = np.full(shape, 255, dtype=np.uint8)
    left = gutter
    for crop in crops:
        sheet[gutter : gutter + PROBE_CROP_HEIGHT, left : left + crop.shape[1]] = crop
        left += crop.shape[1] + gutter
    return sheet, len(crops)


def _sample_probe_lines(
    lines: Sequence[OCRLine], line_count: int
) -> list[OCRLine]:
    """The longest line in each of ``line_count`` vertical bands.

    Two rules, both measured (``docs/decisions-q12.md`` D-Q12-4):

    * **prefer lines that carry letters.**  A Devanagari complaint form's
      account number, IFSC and phone number are Latin-script identifiers that
      *both* heads read perfectly; sampling them tells the probe nothing.  Lines
      whose bootstrap text has no script-bearing character are set aside and
      used only if nothing else is left.  This reads the bootstrap head's text,
      but only to *rank crops* — never to decide a script.
    * **spread across the page, longest per band.**  A form's first lines are
      its printed header; taking the top ``n`` is how a bilingual page gets
      called Latin (the same reason Stage 2 samples crops evenly).
    """
    informative = [
        line
        for line in lines
        if any(script_of_char(character) is not None for character in line.text)
    ]
    pool = informative or list(lines)
    if not pool:
        return []
    tops = [min(line.bbox[1::2]) for line in pool]
    lowest, highest = min(tops), max(tops)
    span = (highest - lowest) or 1.0
    bands: dict[int, list[OCRLine]] = {}
    for line, top in zip(pool, tops, strict=True):
        index = min(line_count - 1, int((top - lowest) / span * line_count))
        bands.setdefault(index, []).append(line)

    def widest(line: OCRLine) -> tuple[float, int]:
        xs = line.bbox[0::2]
        return max(xs) - min(xs), -line.line_index

    return [max(bands[index], key=widest) for index in sorted(bands)]


def _probe_crop(
    page: np.ndarray, line: OCRLine, *, max_crop_width: int
) -> np.ndarray | None:
    """One rectified, height-normalised, width-truncated crop, or ``None``.

    ``None`` for a degenerate box and for a crop with no ink: comparing two
    heads on blank paper measures nothing, and a detector artefact is not a
    document fact (the same rule :func:`annotate_handwriting` applies).
    """
    try:
        crop = crop_perspective(page, line.bbox)
    except (ValueError, IndexError) as exc:
        _LOG.debug("probe skipped line %d: %s", line.line_index, exc)
        return None
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    if float(np.asarray(crop, dtype=np.float32).std()) < PROBE_MIN_INK_STD:
        return None
    cv2 = _cv2()
    scale = PROBE_CROP_HEIGHT / crop.shape[0]
    width = max(8, round(crop.shape[1] * scale))
    resized = cv2.resize(crop, (width, PROBE_CROP_HEIGHT))
    return np.ascontiguousarray(resized[:, :max_crop_width])


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def run_tier1(
    page: PageImage | None,
    *,
    provider: ModelProvider,
    page_no: int,
    settings: Settings | None = None,
    script_hint: str | None = None,
    script_prior: str | None = None,
    bypass_lines: Sequence[OCRLine] | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    rotation_applied_deg: float = 0.0,
    dpi_estimate: float | None = None,
    detect_handwriting: bool = True,
    max_passes: int = MAX_RECOGNITION_PASSES,
    probe_heads: bool = True,
    probe_scripts: Sequence[str] = PROBE_HEAD_SCRIPTS,
) -> Tier1Result:
    """Recognise one page and return it with its escalation signals.

    Args:
        page: the **repaired** page as a NumPy ``uint8`` BGR array (decision
            C1).  May be ``None`` only for a bypass page, where no pixels are
            needed.
        provider: supplies ``ocr_page``; the factory composes it per profile.
        page_no: 1-based page number, carried onto :class:`OCRPage`.
        script_hint: a script the caller already believes, e.g. from a previous
            page of the same document.  Used as the **bootstrap head** for pass
            one, which is where it earns its keep — it does not override the
            vote (see :mod:`app.stages.s2_script`).
        script_prior: upstream, **non-recognition** evidence about this page's
            script — today, the script of a PDF text layer Stage 1.5's geometric
            repair forced the pipeline to discard.  Distinct from
            ``script_hint``: a hint is caller metadata about the *document*, so
            Stage 2 raises ``hint_mismatch`` when the vote contradicts it, while
            a prior is evidence about *these pixels* and is used for exactly two
            things — seeding the bootstrap head when no hint was given, and
            breaking an otherwise-tied probe.  It never overrides a decisive
            probe or a decisive vote.
        bypass_lines: text-layer lines from Stage 1b.  When given, **no
            provider call is made** and the page comes back with
            ``tier=OCRTier.BYPASS``.
        width_px / height_px: page dimensions.  Required when ``page`` is
            ``None``; otherwise taken from the array.
        detect_handwriting: run :func:`annotate_handwriting`.  Off for a bypass
            page by construction — a digital text layer has no strokes.
        max_passes: the number of **full-page provider recognition calls** this
            page may cost.  1 disables both the probe's re-read and the A-6
            re-recognition pass, which is what ``tools/bench_ocr_cpu.py`` wants
            when it is timing a single pass.
        probe_heads: run the dual-head script probe (Q12).  Off restores the
            pre-fix behaviour exactly, which is what the A/B measurement in
            ``docs/decisions-q12.md`` uses; there is no reason to turn it off in
            the service.
        probe_scripts: candidate heads for the probe, in order.

    Raises:
        ValueError: when ``page`` is ``None`` without ``bypass_lines``, when the
            page dimensions cannot be determined, or when ``max_passes < 1``.
    """
    if max_passes < 1:
        raise ValueError(f"max_passes must be >= 1, got {max_passes}")
    # Accepted for a uniform stage signature and forwarded by the orchestrator;
    # nothing here reads it. ``OCR_BATCH_SIZE`` and the recognition-head cache
    # are the provider's settings, not this stage's (PLAN.md invariant 4), and
    # reading the global settings for a value we never use would give a stage
    # with no configuration an environment dependency it does not need.
    del settings

    width, height = _page_dimensions(page, width_px, height_px)

    if bypass_lines is not None:
        return _bypass_result(
            bypass_lines,
            page_no=page_no,
            width=width,
            height=height,
            rotation_applied_deg=rotation_applied_deg,
            dpi_estimate=dpi_estimate,
        )

    if page is None:
        raise ValueError(
            "run_tier1 needs page pixels unless bypass_lines is supplied; "
            "there is no recognition without an image"
        )

    with _OCR_LATENCY.time():
        outcome = _recognise_with_script_check(
            page,
            provider=provider,
            script_hint=script_hint,
            script_prior=script_prior,
            max_passes=max_passes,
            probe_heads=probe_heads,
            probe_scripts=probe_scripts,
        )
        lines, decision, passes = outcome.lines, outcome.decision, outcome.passes

        ratio = 0.0
        measured = False
        if detect_handwriting and lines:
            lines, ratio = annotate_handwriting(page, lines)
            measured = True

        ocr_page = OCRPage(
            page_no=page_no,
            width_px=width,
            height_px=height,
            lines=lines,
            tier=OCRTier.TIER1 if lines else OCRTier.NONE,
            rotation_applied_deg=rotation_applied_deg,
            dpi_estimate=dpi_estimate,
        )

    _OCR_PAGES.labels(tier=ocr_page.tier.value, passes=str(len(passes))).inc()
    _OCR_LINES.observe(len(lines))
    _LOG.debug(
        "stage s4 page=%d lines=%d script=%s head=%s passes=%s probe=%s "
        "handwriting=%.2f",
        page_no,
        len(lines),
        decision.script,
        outcome.head,
        passes,
        None if outcome.probe is None else outcome.probe.reason,
        ratio,
    )
    return Tier1Result(
        page=ocr_page,
        script=decision,
        rec_passes=passes,
        rec_head=outcome.head,
        script_probe=outcome.probe,
        script_disagreement=decision.disagreement or outcome.probe_conflict,
        handwriting_ratio=ratio,
        handwriting_measured=measured,
    )


@dataclass(frozen=True)
class _Recognition:
    """Working state of one page's head selection — never leaves this module.

    A dataclass rather than a Pydantic model on purpose: it carries the page's
    line list on the hot path and there is nothing here to validate that
    :class:`~app.contracts.ocr.OCRLine` has not validated already.
    """

    lines: list[OCRLine]
    decision: ScriptDecision
    passes: tuple[str, ...]
    head: str
    probe: ScriptProbe | None = None
    probe_conflict: bool = False


def _recognise_with_script_check(
    page: PageImage,
    *,
    provider: ModelProvider,
    script_hint: str | None,
    script_prior: str | None,
    max_passes: int,
    probe_heads: bool,
    probe_scripts: Sequence[str],
) -> _Recognition:
    """Bootstrap, probe the heads on the pixels, then confirm with the vote.

    Three steps, in this order, and the order is the whole fix (Q12):

    1. **Bootstrap.**  One pass with the hint, the prior, or Latin.  Unavoidable
       — ``ModelProvider.ocr_page`` detects and recognises in one call, so there
       is no way to obtain line boxes without recognising something, and this is
       the pass the service already paid for before this fix.
    2. **Probe.**  :func:`probe_script_heads` re-reads a few of *that pass's own
       crops* with every installed head and compares them.  This is where the
       head is chosen, and it is chosen from the pixels rather than from one
       head's opinion of itself.  Skipped when the A-6 vote below has already
       decided to re-route the page — that path was never circular, and probing
       it would be paying for an answer we hold.
    3. **Confirm.**  The A-6 text vote runs over the winning head's full-page
       output, exactly as before.  If it contradicts the probe, the page falls
       back to the other head **once**: that head's transcription is almost
       always one we already have (the bootstrap pass), so the correction is
       free, and when it is not, it costs the one pass ``max_passes`` allows.
       There is no path back — the loop cannot re-enter.

    Both directions of "wrong" are therefore recoverable: a probe that routed a
    Latin page to the Devanagari head is caught by the vote over that head's
    Latin output, and a vote that was fooled by Latin noise is pre-empted by the
    probe.  The page carries ``script_disagreement`` when they disagreed at all,
    so the escalation router still sees an unsettled page and Tier 2 remains the
    final safety net (:mod:`app.stages.s2_script`, the A-6 weakness note).
    """
    attempts: dict[str, list[OCRLine]] = {}
    passes: list[str] = []

    def recognise(head: str) -> list[OCRLine]:
        passes.append(head)
        lines = list(provider.ocr_page(page, head))
        attempts[head] = lines
        return lines

    def vote(lines: Sequence[OCRLine], head: str) -> ScriptDecision:
        return identify_script(
            lines=list(lines), hint=script_hint, recognised_with=head
        )

    head = rec_head_for_script(
        script_hint or script_prior or DEFAULT_REC_HEAD_SCRIPT
    )
    lines = recognise(head)
    decision = vote(lines, head)

    probe: ScriptProbe | None = None
    vote_reroutes = decision.is_confident and decision.rec_head_script != head
    # A head that covers every installed script has no blind spot for another
    # head to reveal, so the probe would spend ~12-22% of this stage confirming
    # what is already settled.  The circularity it guards against is a head
    # that *cannot* emit the script in front of it and therefore votes for
    # itself; that cannot happen here.  See subsumes_heads.
    #
    # Guarded on there being more than one candidate: with a single candidate
    # the probe returns ``too_few_heads`` without reading anything, and that
    # record is the audit trail for a one-head node.  Skipping it there would
    # replace a cheap explanation with silence.
    already_settled = len(set(probe_scripts)) > 1 and subsumes_heads(
        head, probe_scripts
    )
    if (
        probe_heads
        and lines
        and max_passes >= 2
        and not vote_reroutes
        and not already_settled
    ):
        # ``max_passes < 2`` means a probe could not be acted on, and buying an
        # answer we cannot use is exactly the cost ``tools/bench_ocr_cpu.py``
        # asks to be spared when it times a single pass.
        probe = probe_script_heads(
            page,
            lines,
            provider=provider,
            scripts=probe_scripts,
            prior=script_prior,
        )
        if probe.head is not None and probe.head != head:
            _LOG.info(
                "re-recognising page with the %s head: the dual-head script "
                "probe scored it %.3f against %.3f for %s over %d shared "
                "crops (%s, OPEN-QUESTIONS Q12)",
                probe.head,
                probe.score_for(probe.head) or 0.0,
                probe.score_for(head) or 0.0,
                head,
                probe.sampled_lines,
                probe.reason,
            )
            _SECOND_PASS.labels(from_script=head, to_script=probe.head).inc()
            head = probe.head
            # ``in``, not ``or``: a head that already ran and read nothing has
            # answered, and re-asking it would be a second call for the same
            # empty answer.
            lines = attempts[head] if head in attempts else recognise(head)
            decision = vote(lines, head)

    conflict = False
    if decision.is_confident and decision.rec_head_script != head:
        target = decision.rec_head_script
        if probe is not None and probe.script is not None:
            conflict = True
            _PROBE_CONFLICTS.labels(
                probe_script=probe.script, voted_script=decision.script
            ).inc()
        if target in attempts:
            # Free: the other head already read this page.  This is the normal
            # way a mistaken probe is undone, and it can happen at most once
            # because the vote that follows is taken against the head that
            # actually produced the text.
            _LOG.info(
                "script vote %s contradicts the %s head that read the page; "
                "reverting to the transcription the %s head already produced",
                decision.script,
                head,
                target,
            )
            head = target
            lines = attempts[target]
            decision = vote(lines, head)
        elif len(passes) < max_passes:
            _LOG.info(
                "re-recognising page with the %s head: the Tier-1 script vote "
                "disagreed with the %s head that produced the text "
                "(adjudication A-6)",
                target,
                head,
            )
            _SECOND_PASS.labels(from_script=head, to_script=target).inc()
            head = target
            lines = recognise(head)
            decision = vote(lines, head)

    return _Recognition(
        lines=lines,
        decision=decision,
        passes=tuple(passes),
        head=head,
        probe=probe,
        probe_conflict=conflict,
    )


def _bypass_result(
    bypass_lines: Sequence[OCRLine],
    *,
    page_no: int,
    width: int,
    height: int,
    rotation_applied_deg: float,
    dpi_estimate: float | None,
) -> Tier1Result:
    """Wrap Stage-1b text-layer lines without touching the recogniser.

    The script is still *voted* — the text layer's characters are the best
    evidence there is, and the vote costs nothing — but ``rec_passes`` is empty
    and ``handwriting_measured`` is false, because a digital text layer has no
    strokes to measure and calling every one of its lines "printed" would be an
    assertion we cannot support.

    No head probe runs here and none is needed: Q12's circularity is "the vote
    reads a head's own output", and this text came from the PDF, not a head.
    That is also why the same text makes a good ``script_prior`` for a page
    whose layer the pipeline had to discard — see :func:`run_tier1`.
    """
    lines = list(bypass_lines)
    decision = (
        identify_script(lines=lines) if lines else None
    )
    page = OCRPage(
        page_no=page_no,
        width_px=width,
        height_px=height,
        lines=lines,
        tier=OCRTier.BYPASS if lines else OCRTier.NONE,
        rotation_applied_deg=rotation_applied_deg,
        dpi_estimate=dpi_estimate,
    )
    _OCR_PAGES.labels(tier=page.tier.value, passes="0").inc()
    _OCR_LINES.observe(len(lines))
    return Tier1Result(
        page=page,
        script=decision,
        rec_passes=(),
        script_disagreement=bool(decision and decision.disagreement),
        handwriting_ratio=0.0,
        handwriting_measured=False,
    )


def _page_dimensions(
    page: PageImage | None, width_px: int | None, height_px: int | None
) -> tuple[int, int]:
    """Resolve ``(width, height)`` from explicit arguments or the array."""
    if width_px is not None and height_px is not None:
        if width_px < 1 or height_px < 1:
            raise ValueError(
                f"page dimensions must be positive, got {width_px}x{height_px}"
            )
        return int(width_px), int(height_px)
    if page is None:
        raise ValueError(
            "page dimensions are required when no page image is supplied"
        )
    array = np.asarray(page)
    if array.ndim < 2:
        raise ValueError(f"page image must be 2-D or 3-D, got {array.shape!r}")
    height, width = array.shape[:2]
    if height < 1 or width < 1:
        raise ValueError(f"page image is empty: shape {array.shape!r}")
    return int(width), int(height)
