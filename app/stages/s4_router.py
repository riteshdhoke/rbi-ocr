"""Stage 4 router — Tier-2 escalation decision + the 3% shadow sampler.

PLAN.md **invariant 3**, restated because it is the whole design of this
module: *the router keys on validator failure and structural signals, never on
raw CTC confidence.*  Recognition confidence is uncalibrated on Devanagari
conjuncts — a wrong conjunct often scores higher than a right one — so a router
that read it would fail silently on exactly the pages it exists to catch.
There is no code path in this module that reads
:attr:`app.contracts.ocr.OCRLine.confidence`, and ``tests/unit/test_s4_router.py``
asserts that by inspecting the module source.

The rule is an **explicit additive score** rather than a chain of ``if``\\ s.
Three reasons: every signal's contribution is one number a reviewer can argue
about; the whole rule is a truth table a test can enumerate; and re-tuning is a
constant change, not a control-flow change.  The weights are:

===========================  ======  ==========================================
Signal                       Points  Why that much
===========================  ======  ==========================================
``validator_failures``       1.0 ea  A failed checksum on a structured field is
                             (max 3) the strongest evidence recognition was
                                     wrong. **One** failure is not enough on its
                                     own — a complainant can write a wrong
                                     account number, and re-reading the page in
                                     a bigger model will not fix that.
``script_disagreement``       2.0    We may have recognised with the wrong head
                                     (adjudication A-6). Sufficient alone.
``quality_score_low``         1.0    The quality gate accepted the page but
                                     flagged it; on its own that is the
                                     ``scan``/``heavy`` majority of the corpus
                                     and escalating all of it is not a router.
``handwriting_detected``      0.5    Plus 0.5 more when the page is *mostly*
                                     handwritten. Never sufficient alone: the
                                     handwritten complaint letter is the normal
                                     case, not the exception.
``structural_anomalies``      0.5 ea Table/layout anomalies, capped at 1.5.
===========================  ======  ==========================================

Escalate at **>= 2.0** (:data:`ESCALATION_SCORE_THRESHOLD`).  Every consequence
is spelled out in :func:`route`'s tests, including the ones that deliberately do
*not* escalate.

The shadow sampler
------------------
BUILD-BRIEF sec. 5: "Ship the 3% shadow sampler in the same PR as the router —
a router without it cannot be evaluated."  :func:`shadow_sample` runs Tier 2 on
a deterministic 3% of **passed** pages so the escalation rule can be scored
against what Tier 2 would have said, and it is deterministic in the document id
so a shadow run is replayable from the corpus alone — no stored random state,
no divergence between a rerun and the original.

Shadow pages are marked ``shadow_sampled`` and **not** ``escalated``
(:class:`app.contracts.envelope.PageResult` keeps the two apart) precisely so a
shadow run cannot inflate the escalation rate it exists to measure.

This module does NOT handle:
  * running Tier 2 — that is :mod:`app.stages.s4b_vlm`, and keeping the decision
    pure means the router can be exhaustively tested with no model at all;
  * producing the signals.  Validator failures arrive from Stage 5 (WS-E3),
    quality from Stage 1.5 (WS-E1), script and handwriting from Stage 4;
    :class:`EscalationSignals` is the seam and it has no dependency on either;
  * the envelope.  ``assemble.py`` copies ``escalated`` / ``shadow_sampled``
    onto :class:`~app.contracts.envelope.PageResult`;
  * measuring whether escalation *helped*.  :func:`build_shadow_diff` records
    the difference; the eval harness scores it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Self

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.ocr import Confidence, OCRLine
from app.settings import Settings, get_settings

__all__ = [
    "ESCALATION_SCORE_THRESHOLD",
    "SIGNAL_POINTS",
    "EscalationDecision",
    "EscalationReason",
    "EscalationSignals",
    "ExpectedEscalation",
    "ShadowDiff",
    "build_shadow_diff",
    "expected_escalation",
    "route",
    "shadow_sample",
]

_LOG = logging.getLogger(__name__)

#: Escalate at or above this total.  A module constant, not a ``Settings``
#: field, because ``app/settings.py`` is another workstream's committed lane;
#: ``docs/decisions-wse2.md`` hands it over for promotion to
#: ``ESCALATION_SCORE_THRESHOLD`` there.  Overridable per call.
ESCALATION_SCORE_THRESHOLD: Final[float] = 2.0

#: The published weight table.  Public so a report can print the rule it scored.
SIGNAL_POINTS: Final[dict[str, float]] = {
    "validator_failure": 1.0,
    "script_disagreement": 2.0,
    "quality_score_low": 1.0,
    "handwriting_detected": 0.5,
    "handwriting_majority": 0.5,
    "structural_anomaly": 0.5,
}

#: Validator failures beyond this add nothing; three broken fields and thirty
#: are the same decision.
MAX_SCORED_VALIDATOR_FAILURES: Final[int] = 3

#: Structural anomalies beyond this add nothing.
MAX_SCORED_STRUCTURAL_ANOMALIES: Final[int] = 3

#: A page is "mostly handwritten" at or above this share of lines.
HANDWRITING_MAJORITY_RATIO: Final[float] = 0.5

_ROUTER_DECISIONS = Counter(
    "stage_s4_router_decisions_total",
    "Escalation router decisions.",
    ["escalated", "shadow_sampled"],
)
_ROUTER_REASONS = Counter(
    "stage_s4_router_reasons_total",
    "Individual signals that contributed to an escalation.",
    ["reason"],
)
_ROUTER_SCORE = Histogram(
    "stage_s4_router_score",
    "Escalation score per page.",
    buckets=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
)


class EscalationReason(StrEnum):
    """Machine-readable contributors to an escalation decision."""

    VALIDATOR_FAILURE = "validator_failure"
    SCRIPT_DISAGREEMENT = "script_disagreement"
    QUALITY_SCORE_LOW = "quality_score_low"
    HANDWRITING_DETECTED = "handwriting_detected"
    HANDWRITING_MAJORITY = "handwriting_majority"
    STRUCTURAL_ANOMALY = "structural_anomaly"
    #: Not a signal: the page passed and was drawn for the shadow sample.
    SHADOW_SAMPLE = "shadow_sample"
    #: Not a signal: the caller forced escalation (replay, review UI, tests).
    FORCED = "forced"


class EscalationSignals(BaseModel):
    """Everything the router is allowed to see.

    Deliberately *not* an ``OCRPage``.  Validator failures arrive from Stage 5,
    which runs after this decision is first taken in the two-stage delivery
    model (BUILD-BRIEF sec. 5), so the router cannot depend on a page object
    that does not exist yet.  It also cannot accidentally read a confidence
    number, because there is no confidence number on this model to read
    (PLAN.md invariant 3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Deterministic fields whose validator rejected the recognised value
    #: (failed checksum, structure or master-list lookup) — Stage 5, WS-E3.
    validator_failures: int = Field(default=0, ge=0)
    #: Names of the failing fields, for the reason string. Never their values.
    validator_failed_fields: tuple[str, ...] = ()
    #: Layout facts the page classifier or quality gate found odd: e.g.
    #: ``"table_without_rules"``, ``"occlusion"``, ``"line_count_outlier"``.
    structural_anomalies: tuple[str, ...] = ()
    #: Stage 4's heuristic found handwriting on this page.
    handwriting_detected: bool = False
    #: Share of lines called handwritten, in [0, 1].
    handwriting_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Stage 2 could not settle the script, or it disagreed with the head that
    #: produced the text (adjudication A-6).
    script_disagreement: bool = False
    #: The quality gate accepted the page but scored it near the bounce line.
    quality_score_low: bool = False
    #: The score itself, carried for the audit trail. Never scored directly —
    #: the boolean above is the signal, so the threshold lives in one place
    #: (Stage 1.5) instead of two.
    quality_score: Confidence | None = None
    #: Set by a caller that must escalate regardless (replay, human request).
    forced: bool = False

    @model_validator(mode="after")
    def _handwriting_is_consistent(self) -> Self:
        if self.handwriting_ratio > 0.0 and not self.handwriting_detected:
            raise ValueError(
                f"handwriting_ratio={self.handwriting_ratio} with "
                "handwriting_detected=False: the ratio is the measurement "
                "behind the flag, not an independent claim"
            )
        if len(self.validator_failed_fields) > self.validator_failures:
            raise ValueError(
                f"{len(self.validator_failed_fields)} named failing fields but "
                f"validator_failures={self.validator_failures}"
            )
        return self


class EscalationDecision(BaseModel):
    """The router's verdict for one page, with the arithmetic that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    escalate: bool
    #: Ran through Tier 2 as part of the shadow sample, **not** an escalation.
    shadow_sampled: bool = False
    #: Signals that contributed, in the published table's order.
    reasons: tuple[EscalationReason, ...] = ()
    #: The additive score, kept so a tuning report can plot the distribution.
    score: float = Field(ge=0.0)
    threshold: float = Field(gt=0.0)
    #: Human-readable summary for the audit log. Never parsed, never a value.
    detail: str = ""

    @model_validator(mode="after")
    def _verdict_matches_arithmetic(self) -> Self:
        if self.escalate and self.shadow_sampled:
            raise ValueError(
                "a page cannot be both escalated and shadow-sampled: the "
                "shadow sample exists to measure the escalation rate and must "
                "not be counted into it (PLAN.md invariant 3)"
            )
        if self.escalate and not self.reasons:
            raise ValueError("an escalation must name at least one reason")
        return self

    @property
    def runs_tier2(self) -> bool:
        """Whether Tier 2 should run for this page, for either reason."""
        return self.escalate or self.shadow_sampled


def shadow_sample(
    doc_id: str,
    *,
    rate: float,
    page_no: int | None = None,
) -> bool:
    """Deterministically decide whether this page joins the shadow sample.

    Keyed on a BLAKE2b digest of ``doc_id`` (plus ``page_no`` when given), so a
    replay of the same corpus draws the same sample without storing anything.
    That is the property that makes a shadow run comparable across code
    versions: the sample is a function of the data, not of the run.

    Including ``page_no`` means a 40-page document contributes ~3% of its pages
    rather than all-or-nothing; pass ``page_no=None`` to sample whole documents.

    Args:
        rate: share in ``[0, 1]``. ``0.0`` disables sampling entirely and
            ``1.0`` samples everything, both without a special case.

    Raises:
        ValueError: for a rate outside ``[0, 1]`` or an empty ``doc_id``.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"shadow rate must be in [0, 1], got {rate}")
    if not doc_id:
        raise ValueError("doc_id must be non-empty: the sample is keyed on it")
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    key = doc_id if page_no is None else f"{doc_id}:{page_no}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    #: Uniform in [0, 1) by construction; 2**64 is exact in float64.
    fraction = int.from_bytes(digest, "big") / float(1 << 64)
    return fraction < rate


def route(
    signals: EscalationSignals,
    *,
    doc_id: str,
    page_no: int | None = None,
    settings: Settings | None = None,
    threshold: float = ESCALATION_SCORE_THRESHOLD,
    shadow_rate: float | None = None,
) -> EscalationDecision:
    """Decide whether this page escalates to Tier 2, and whether it is shadowed.

    The shadow draw happens **only for pages that did not escalate**: a page
    already going to Tier 2 cannot also be a control for it.

    Args:
        doc_id: the shadow sampler's key. Required even when the rate is 0, so
            switching sampling on cannot change a call site.
        threshold: escalate at or above this score.
        shadow_rate: overrides ``Settings.ESCALATION_SHADOW_RATE``.

    Raises:
        ValueError: for a non-positive threshold or an out-of-range rate.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    active = settings if settings is not None else get_settings()
    rate = (
        float(active.ESCALATION_SHADOW_RATE)
        if shadow_rate is None
        else float(shadow_rate)
    )

    score, reasons = _score(signals)
    escalate = signals.forced or score >= threshold
    if signals.forced and EscalationReason.FORCED not in reasons:
        reasons = (*reasons, EscalationReason.FORCED)

    sampled = (
        False if escalate else shadow_sample(doc_id, rate=rate, page_no=page_no)
    )
    if sampled:
        reasons = (*reasons, EscalationReason.SHADOW_SAMPLE)

    _ROUTER_SCORE.observe(score)
    _ROUTER_DECISIONS.labels(
        escalated=str(escalate).lower(), shadow_sampled=str(sampled).lower()
    ).inc()
    if escalate:
        for reason in reasons:
            _ROUTER_REASONS.labels(reason=reason.value).inc()

    decision = EscalationDecision(
        escalate=escalate,
        shadow_sampled=sampled,
        reasons=reasons if (escalate or sampled) else (),
        score=score,
        threshold=threshold,
        detail=_detail(signals, score, threshold, escalate, sampled),
    )
    _LOG.debug(
        "stage s4 router doc=%s page=%s score=%.2f escalate=%s shadow=%s",
        doc_id,
        page_no,
        score,
        escalate,
        sampled,
    )
    return decision


def _score(
    signals: EscalationSignals,
) -> tuple[float, tuple[EscalationReason, ...]]:
    """The published additive rule.  Pure; the only place points are assigned."""
    score = 0.0
    reasons: list[EscalationReason] = []

    failures = min(signals.validator_failures, MAX_SCORED_VALIDATOR_FAILURES)
    if failures:
        score += failures * SIGNAL_POINTS["validator_failure"]
        reasons.append(EscalationReason.VALIDATOR_FAILURE)

    if signals.script_disagreement:
        score += SIGNAL_POINTS["script_disagreement"]
        reasons.append(EscalationReason.SCRIPT_DISAGREEMENT)

    if signals.quality_score_low:
        score += SIGNAL_POINTS["quality_score_low"]
        reasons.append(EscalationReason.QUALITY_SCORE_LOW)

    if signals.handwriting_detected:
        score += SIGNAL_POINTS["handwriting_detected"]
        reasons.append(EscalationReason.HANDWRITING_DETECTED)
        if signals.handwriting_ratio >= HANDWRITING_MAJORITY_RATIO:
            score += SIGNAL_POINTS["handwriting_majority"]
            reasons.append(EscalationReason.HANDWRITING_MAJORITY)

    anomalies = min(
        len(signals.structural_anomalies), MAX_SCORED_STRUCTURAL_ANOMALIES
    )
    if anomalies:
        score += anomalies * SIGNAL_POINTS["structural_anomaly"]
        reasons.append(EscalationReason.STRUCTURAL_ANOMALY)

    return round(score, 6), tuple(reasons)


def _detail(
    signals: EscalationSignals,
    score: float,
    threshold: float,
    escalate: bool,
    sampled: bool,
) -> str:
    """One audit-log line.  Field *names* only — never a recognised value."""
    parts = [f"score={score:.2f}/{threshold:.2f}"]
    if signals.validator_failures:
        named = ",".join(signals.validator_failed_fields) or "unnamed"
        parts.append(f"validator_failures={signals.validator_failures}({named})")
    if signals.script_disagreement:
        parts.append("script_disagreement")
    if signals.quality_score_low:
        parts.append("quality_low")
    if signals.handwriting_detected:
        parts.append(f"handwriting={signals.handwriting_ratio:.2f}")
    if signals.structural_anomalies:
        parts.append(f"anomalies={','.join(signals.structural_anomalies)}")
    if signals.forced:
        parts.append("forced")
    if sampled:
        parts.append("shadow_sample")
    parts.append("escalate" if escalate else "tier1_final")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Shadow-diff records
# --------------------------------------------------------------------------- #


class ShadowDiff(BaseModel):
    """What Tier 2 saw that Tier 1 did not, for one shadow-sampled page.

    This is the record BUILD-BRIEF sec. 5 says a router cannot be evaluated
    without.  It is a *measurement*, not a judgement: nothing here decides which
    tier was right, because on a synthetic corpus the ground truth already does
    and on real pages a human does.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    page_no: int = Field(ge=1)
    #: Whether the page reached Tier 2 as an escalation or as a shadow control.
    shadow: bool = True
    tier1_line_count: int = Field(ge=0)
    tier2_line_count: int = Field(ge=0)
    #: Lines whose normalised text is identical in both tiers.
    unchanged_lines: int = Field(ge=0)
    #: Lines present in both tiers whose text differs.
    changed_lines: int = Field(ge=0)
    #: Mean normalised edit distance over the lines present in both, in [0, 1].
    mean_text_distance: float = Field(ge=0.0, le=1.0)
    #: Tier-2 lines with no Tier-1 counterpart (recall Tier 1 lacked).
    tier2_only_lines: int = Field(ge=0)
    #: Tier-1 lines with no Tier-2 counterpart (recall Tier 2 lacked).
    tier1_only_lines: int = Field(ge=0)
    #: The router's verdict, so a report can join diffs to decisions.
    escalation_score: float = Field(ge=0.0)
    escalation_reasons: tuple[EscalationReason, ...] = ()

    @property
    def changed_ratio(self) -> float:
        """Share of commonly-recognised lines whose text Tier 2 changed."""
        common = self.unchanged_lines + self.changed_lines
        return 0.0 if common == 0 else self.changed_lines / common

    @property
    def is_material(self) -> bool:
        """Whether Tier 2 disagreed enough to be worth a human's attention.

        A quarter of the shared lines changed, or a line found or lost.  Stated
        as a property rather than baked into the record so the eval harness can
        re-cut it without regenerating the diffs.
        """
        return (
            self.changed_ratio >= 0.25
            or self.tier2_only_lines > 0
            or self.tier1_only_lines > 0
        )


def build_shadow_diff(
    *,
    doc_id: str,
    page_no: int,
    tier1_lines: Sequence[OCRLine],
    tier2_lines: Sequence[OCRLine],
    decision: EscalationDecision | None = None,
    shadow: bool = True,
    iou_threshold: float = 0.5,
) -> ShadowDiff:
    """Compare Tier-1 and Tier-2 output for one page.

    Lines are paired by bounding-box overlap (the same greedy IoU match
    :mod:`app.stages.s4b_vlm` uses to merge them), because ``line_index`` is
    assigned per tier and two tiers that found different numbers of lines have
    no shared index space.
    """
    from app.stages.s4b_vlm import match_lines_by_iou  # one matcher, one place

    pairs, tier1_only, tier2_only = match_lines_by_iou(
        tier1_lines, tier2_lines, iou_threshold=iou_threshold
    )
    unchanged = 0
    changed = 0
    distances: list[float] = []
    for first, second in pairs:
        distance = _normalised_distance(first.text, second.text)
        distances.append(distance)
        if distance == 0.0:
            unchanged += 1
        else:
            changed += 1

    return ShadowDiff(
        doc_id=doc_id,
        page_no=page_no,
        shadow=shadow,
        tier1_line_count=len(tier1_lines),
        tier2_line_count=len(tier2_lines),
        unchanged_lines=unchanged,
        changed_lines=changed,
        mean_text_distance=(
            float(sum(distances) / len(distances)) if distances else 0.0
        ),
        tier2_only_lines=len(tier2_only),
        tier1_only_lines=len(tier1_only),
        escalation_score=0.0 if decision is None else decision.score,
        escalation_reasons=() if decision is None else decision.reasons,
    )


def _normalised_distance(first: str, second: str) -> float:
    """Normalised Levenshtein distance in ``[0, 1]``; whitespace-insensitive.

    Uses ``rapidfuzz`` (already a pinned dependency for the validators) rather
    than a hand-rolled DP: it is C, it is correct, and a shadow report over a
    full corpus is a lot of string pairs.
    """
    from rapidfuzz.distance import Levenshtein

    left = " ".join(first.split())
    right = " ".join(second.split())
    if not left and not right:
        return 0.0
    return float(Levenshtein.normalized_distance(left, right))


# --------------------------------------------------------------------------- #
# Adjudication A-3: the eval harness's escalation reference
# --------------------------------------------------------------------------- #


class ExpectedEscalation(StrEnum):
    """What the corpus implies Tier 2 *should* have done for a page."""

    EXPECTED = "expected"
    NOT_EXPECTED = "not_expected"
    #: The page never reaches the router, so scoring it either way is wrong.
    EXCLUDED = "excluded"


#: Degradation profiles that the router is expected to escalate.  ``heavy`` and
#: ``photo`` are TEST-DATA-SPEC sec. 3's own words ("likely escalate",
#: "escalate"); ``clean``/``light``/``scan`` are "accept" and "accept (repair)".
ESCALATION_EXPECTED_PROFILES: Final[frozenset[str]] = frozenset({"heavy", "photo"})


def expected_escalation(
    *,
    profile: str | None = None,
    quality_expectation: str | None = None,
    handwriting_ratio: float = 0.0,
    is_code_mixed: bool = False,
) -> ExpectedEscalation:
    """The escalation reference this router implies — **revisiting A-3**.

    Adjudication A-3 accepted the eval harness's stand-in
    (``quality_expectation != "clean"`` OR a handwritten majority) *until the
    router lands*, and asked WS-E2 to revisit it.  Here is the revision, and why
    the stand-in is wrong in three specific ways:

    1. **``bounce``-profile pages must be excluded, not counted.**  A bounced
       page never reaches the router — Stage 1.5 rejects it and no OCR runs — so
       counting it as "should have escalated" charges the router a false
       negative for a page it was never shown.  On the committed corpus that is
       50 of 295 documents, about 17%, which is enough to dominate the recall
       number.
    2. **``degraded`` is too coarse.**  It covers ``light``, ``scan``, ``heavy``
       and ``photo`` alike, but TEST-DATA-SPEC sec. 3 expects ``light`` and
       ``scan`` to be *accepted* (``scan`` with repair) and only ``heavy`` and
       ``photo`` to escalate.  Treating all four as "should escalate" makes 100
       pages that the design says are fine look like router misses, i.e. it
       measures the corpus rather than the router.
    3. **A code-mixed page is expected to escalate at any profile.**  Measured
       on the committed corpus (variants 0-2, non-bounce): ``bilingual_form`` is
       the only type whose script vote raises a disagreement, it raises
       ``split_vote`` on **15 of 15** of its pages, and the router escalates
       every one.  That is correct behaviour, not a false positive — Tier 1
       recognises a whole page with **one** head, so the four Devanagari value
       lines on a bilingual form were provably read with the Latin head, and
       the second recognition pass cannot fix it (there is no per-line head).
       Tier 2 is the actual remedy.  Without this clause the reference calls 9
       of those 15 pages ``not_expected`` and the harness charges the router a
       false positive for doing the right thing.

    The revised reference is therefore: **excluded** when the page bounces;
    **expected** when the profile is ``heavy``/``photo``, the page is mostly
    handwritten, or the page is code-mixed; **not expected** otherwise.
    ``profile`` is preferred and ``quality_expectation`` is the fallback for a
    caller that has only the coarse label — in which case ``degraded`` maps to
    *expected*, matching the old stand-in so the two are comparable.

    Args:
        is_code_mixed: the corpus's own ``is_code_mixed`` flag, which
            ``corpus/index.json`` carries per document (and which Stage 2b
            recomputes at runtime as :attr:`LidResult.is_code_mixed`).

    This function is deliberately here and not in ``tools/eval_harness.py``: the
    harness is standalone by adjudication A-2 and must not import ``app``.
    ``docs/decisions-wse2.md`` records the handoff — WS-D/WS-H should add a
    ``--escalation-reference router_v1`` mode that mirrors this table.
    """
    if not 0.0 <= handwriting_ratio <= 1.0:
        raise ValueError(
            f"handwriting_ratio must be in [0, 1], got {handwriting_ratio}"
        )
    if profile == "bounce" or quality_expectation == "bounce":
        return ExpectedEscalation.EXCLUDED
    if is_code_mixed or handwriting_ratio >= HANDWRITING_MAJORITY_RATIO:
        return ExpectedEscalation.EXPECTED
    if profile is not None:
        return (
            ExpectedEscalation.EXPECTED
            if profile in ESCALATION_EXPECTED_PROFILES
            else ExpectedEscalation.NOT_EXPECTED
        )
    if quality_expectation is None:
        raise ValueError(
            "expected_escalation needs a profile or a quality_expectation"
        )
    return (
        ExpectedEscalation.NOT_EXPECTED
        if quality_expectation == "clean"
        else ExpectedEscalation.EXPECTED
    )
