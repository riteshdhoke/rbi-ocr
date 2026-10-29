"""Stage 4 router — escalation truth table, invariant 3, and the shadow sampler.

Three things are asserted here that nothing else in the suite can assert:

1. **The published weight table is the rule.**  The truth table is enumerated
   exhaustively over the signal space and scored by a *second, independent*
   implementation of the table in this file.  If someone edits ``_score`` to
   add a hidden condition, the two disagree and the test fails.
2. **PLAN.md invariant 3 — never raw CTC confidence.**  Asserted structurally
   (the module's AST contains no ``.confidence`` attribute read, and
   :class:`EscalationSignals` has no confidence field for one to reach) rather
   than behaviourally, because a behavioural test can only prove the paths it
   happened to take.
3. **The shadow sampler is deterministic and ~3%.**  A shadow run that is not
   replayable from the corpus alone cannot be used to score a router across
   code versions (BUILD-BRIEF sec. 5).
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from itertools import product

import pytest

from app.contracts.ocr import OCRLine, bbox_from_xyxy
from app.settings import Settings
from app.stages import s4_router
from app.stages.s4_router import (
    ESCALATION_SCORE_THRESHOLD,
    HANDWRITING_MAJORITY_RATIO,
    MAX_SCORED_STRUCTURAL_ANOMALIES,
    MAX_SCORED_VALIDATOR_FAILURES,
    SIGNAL_POINTS,
    EscalationDecision,
    EscalationReason,
    EscalationSignals,
    ExpectedEscalation,
    build_shadow_diff,
    expected_escalation,
    route,
    shadow_sample,
)

DOC = "sha256:" + "c" * 64


def _line(text: str, index: int, *, y: float = 0.0, confidence: float = 0.9) -> OCRLine:
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(10.0, y, 210.0, y + 20.0),
        confidence=confidence,
        script="Latn",
        language=None,
        is_handwritten=False,
        line_index=index,
    )


# --------------------------------------------------------------------------- #
# PLAN.md invariant 3 — the router may not read recognition confidence
# --------------------------------------------------------------------------- #


def test_the_router_module_never_reads_a_confidence_attribute() -> None:
    """Invariant 3, asserted on the source rather than on a code path.

    ``OCRLine.confidence`` is uncalibrated on Devanagari conjuncts; a router
    that read it would fail silently on exactly the pages it exists to catch.
    Prose *mentions* of confidence are fine — this looks at the AST, so only a
    real attribute read can fail it.
    """
    tree = ast.parse(inspect.getsource(s4_router))
    reads = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "confidence"
    ]
    assert reads == [], f"router reads {reads}; PLAN.md invariant 3 forbids it"


def test_the_signals_model_has_no_recognition_confidence_field() -> None:
    """There is no confidence number on the seam for a future edit to reach."""
    assert "confidence" not in EscalationSignals.model_fields
    assert "ocr_confidence" not in EscalationSignals.model_fields
    # quality_score is carried for the audit trail only; the *signal* is the
    # boolean, so the threshold lives once, in Stage 1.5.
    assert "quality_score_low" in EscalationSignals.model_fields


def test_line_confidence_cannot_change_a_decision() -> None:
    """The behavioural counterpart: the router is not given lines at all."""
    with pytest.raises(ValueError):  # extra="forbid"
        EscalationSignals(lines=[_line("x", 0, confidence=0.01)])  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# The truth table
# --------------------------------------------------------------------------- #


def _expected_score(signals: EscalationSignals) -> float:
    """The published table, re-implemented independently of ``_score``."""
    score = 0.0
    score += (
        min(signals.validator_failures, MAX_SCORED_VALIDATOR_FAILURES)
        * SIGNAL_POINTS["validator_failure"]
    )
    if signals.script_disagreement:
        score += SIGNAL_POINTS["script_disagreement"]
    if signals.quality_score_low:
        score += SIGNAL_POINTS["quality_score_low"]
    if signals.handwriting_detected:
        score += SIGNAL_POINTS["handwriting_detected"]
        if signals.handwriting_ratio >= HANDWRITING_MAJORITY_RATIO:
            score += SIGNAL_POINTS["handwriting_majority"]
    score += (
        min(len(signals.structural_anomalies), MAX_SCORED_STRUCTURAL_ANOMALIES)
        * SIGNAL_POINTS["structural_anomaly"]
    )
    return score


@pytest.mark.parametrize(
    ("failures", "script", "quality", "hand_ratio", "anomalies"),
    list(
        product(
            (0, 1, 2, 3, 5),
            (False, True),
            (False, True),
            (0.0, 0.2, 0.5, 1.0),
            (0, 1, 3, 4),
        )
    ),
)
def test_escalation_truth_table(
    failures: int,
    script: bool,
    quality: bool,
    hand_ratio: float,
    anomalies: int,
) -> None:
    """Every combination of signals scores exactly what the table says."""
    signals = EscalationSignals(
        validator_failures=failures,
        script_disagreement=script,
        quality_score_low=quality,
        handwriting_detected=hand_ratio > 0.0,
        handwriting_ratio=hand_ratio,
        structural_anomalies=tuple(f"anomaly_{i}" for i in range(anomalies)),
    )
    decision = route(signals, doc_id=DOC, page_no=1, shadow_rate=0.0)

    expected = _expected_score(signals)
    assert decision.score == pytest.approx(expected)
    assert decision.escalate is (expected >= ESCALATION_SCORE_THRESHOLD)
    assert decision.threshold == ESCALATION_SCORE_THRESHOLD


@pytest.mark.parametrize(
    ("signals", "escalates", "why"),
    [
        (
            EscalationSignals(),
            False,
            "a clean page is not escalated",
        ),
        (
            EscalationSignals(validator_failures=1, validator_failed_fields=("ifsc_code",)),
            False,
            "one bad field can be a complainant's own error, not a misread",
        ),
        (
            EscalationSignals(validator_failures=2),
            True,
            "two failures is evidence about the recognition, not the writer",
        ),
        (
            EscalationSignals(script_disagreement=True),
            True,
            "we may have read the page with the wrong head (A-6)",
        ),
        (
            EscalationSignals(quality_score_low=True),
            False,
            "the scan/heavy majority of the corpus is quality-flagged",
        ),
        (
            EscalationSignals(handwriting_detected=True, handwriting_ratio=0.9),
            False,
            "a handwritten complaint letter is the normal case",
        ),
        (
            EscalationSignals(
                quality_score_low=True,
                handwriting_detected=True,
                handwriting_ratio=0.9,
            ),
            True,
            "a mostly-handwritten page the quality gate also flagged",
        ),
        (
            EscalationSignals(
                validator_failures=1,
                quality_score_low=True,
            ),
            True,
            "a failed field on a page we already doubt",
        ),
        (
            EscalationSignals(structural_anomalies=("table_without_rules",)),
            False,
            "one layout oddity is not enough",
        ),
        (
            EscalationSignals(
                structural_anomalies=("table_without_rules", "occlusion", "outlier"),
            ),
            False,
            "three anomalies is 1.5 points, still under the bar",
        ),
        (
            EscalationSignals(
                structural_anomalies=("a", "b", "c"),
                handwriting_detected=True,
                handwriting_ratio=0.1,
            ),
            True,
            "1.5 anomalies plus handwriting reaches 2.0",
        ),
    ],
)
def test_named_cases_from_the_published_table(
    signals: EscalationSignals, escalates: bool, why: str
) -> None:
    decision = route(signals, doc_id=DOC, page_no=1, shadow_rate=0.0)
    assert decision.escalate is escalates, why
    if escalates:
        assert decision.reasons, "an escalation must name its reasons"


def test_reasons_are_reported_in_the_published_order() -> None:
    signals = EscalationSignals(
        validator_failures=1,
        script_disagreement=True,
        quality_score_low=True,
        handwriting_detected=True,
        handwriting_ratio=0.8,
        structural_anomalies=("occlusion",),
    )
    decision = route(signals, doc_id=DOC, page_no=1, shadow_rate=0.0)
    assert decision.reasons == (
        EscalationReason.VALIDATOR_FAILURE,
        EscalationReason.SCRIPT_DISAGREEMENT,
        EscalationReason.QUALITY_SCORE_LOW,
        EscalationReason.HANDWRITING_DETECTED,
        EscalationReason.HANDWRITING_MAJORITY,
        EscalationReason.STRUCTURAL_ANOMALY,
    )


def test_validator_failures_saturate() -> None:
    """Three broken fields and thirty are the same decision."""
    many = route(
        EscalationSignals(validator_failures=30), doc_id=DOC, shadow_rate=0.0
    )
    capped = route(
        EscalationSignals(validator_failures=MAX_SCORED_VALIDATOR_FAILURES),
        doc_id=DOC,
        shadow_rate=0.0,
    )
    assert many.score == capped.score == pytest.approx(3.0)


def test_structural_anomalies_saturate() -> None:
    many = route(
        EscalationSignals(structural_anomalies=tuple(f"a{i}" for i in range(20))),
        doc_id=DOC,
        shadow_rate=0.0,
    )
    assert many.score == pytest.approx(
        MAX_SCORED_STRUCTURAL_ANOMALIES * SIGNAL_POINTS["structural_anomaly"]
    )


def test_a_forced_page_escalates_whatever_it_scores() -> None:
    decision = route(
        EscalationSignals(forced=True), doc_id=DOC, page_no=2, shadow_rate=0.0
    )
    assert decision.escalate is True
    assert decision.score == pytest.approx(0.0)
    assert EscalationReason.FORCED in decision.reasons


def test_a_page_that_did_not_escalate_reports_no_reasons() -> None:
    """Reasons are the justification for an action, not a running commentary."""
    decision = route(
        EscalationSignals(handwriting_detected=True, handwriting_ratio=0.2),
        doc_id=DOC,
        page_no=1,
        shadow_rate=0.0,
    )
    assert decision.escalate is False
    assert decision.reasons == ()
    assert decision.score == pytest.approx(0.5)


def test_the_detail_string_names_fields_never_values() -> None:
    signals = EscalationSignals(
        validator_failures=2,
        validator_failed_fields=("ifsc_code", "account_number"),
        quality_score_low=True,
        quality_score=0.4,
    )
    decision = route(signals, doc_id=DOC, page_no=1, shadow_rate=0.0)
    assert "ifsc_code" in decision.detail
    assert "account_number" in decision.detail
    assert "escalate" in decision.detail
    # The audit line carries no recognised text and no score of a field value.
    assert "BARB" not in decision.detail


def test_a_non_positive_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        route(EscalationSignals(), doc_id=DOC, threshold=0.0, shadow_rate=0.0)


# --------------------------------------------------------------------------- #
# Signal-model discipline
# --------------------------------------------------------------------------- #


def test_a_handwriting_ratio_without_the_flag_is_incoherent() -> None:
    with pytest.raises(ValueError, match="not an independent claim"):
        EscalationSignals(handwriting_ratio=0.4, handwriting_detected=False)


def test_more_named_fields_than_failures_is_refused() -> None:
    with pytest.raises(ValueError, match="named failing fields"):
        EscalationSignals(
            validator_failures=1, validator_failed_fields=("a", "b")
        )


def test_a_decision_cannot_be_both_escalated_and_shadowed() -> None:
    with pytest.raises(ValueError, match="must not be counted into it"):
        EscalationDecision(
            escalate=True,
            shadow_sampled=True,
            reasons=(EscalationReason.FORCED,),
            score=2.0,
            threshold=2.0,
        )


def test_an_escalation_must_name_a_reason() -> None:
    with pytest.raises(ValueError, match="must name at least one reason"):
        EscalationDecision(escalate=True, score=2.0, threshold=2.0)


def test_runs_tier2_covers_both_routes_to_the_vlm() -> None:
    escalated = EscalationDecision(
        escalate=True,
        reasons=(EscalationReason.SCRIPT_DISAGREEMENT,),
        score=2.0,
        threshold=2.0,
    )
    shadowed = EscalationDecision(
        escalate=False,
        shadow_sampled=True,
        reasons=(EscalationReason.SHADOW_SAMPLE,),
        score=0.0,
        threshold=2.0,
    )
    quiet = EscalationDecision(escalate=False, score=0.0, threshold=2.0)
    assert escalated.runs_tier2 is True
    assert shadowed.runs_tier2 is True
    assert quiet.runs_tier2 is False


# --------------------------------------------------------------------------- #
# The 3% shadow sampler
# --------------------------------------------------------------------------- #


def test_the_sample_is_a_function_of_the_data_not_of_the_run() -> None:
    """Replayability: the property that makes a shadow run comparable."""
    first = [shadow_sample(f"doc-{i}", rate=0.03, page_no=1) for i in range(500)]
    second = [shadow_sample(f"doc-{i}", rate=0.03, page_no=1) for i in range(500)]
    assert first == second
    assert any(first), "a 3% sample of 500 pages that draws nothing is suspect"


def test_the_sampler_lands_near_the_configured_rate() -> None:
    total = 20_000
    drawn = sum(
        shadow_sample(f"doc-{i}", rate=0.03, page_no=i % 7) for i in range(total)
    )
    observed = drawn / total
    assert 0.02 <= observed <= 0.04, f"observed shadow rate {observed:.4f}"


def test_rate_zero_and_one_need_no_special_case_at_the_call_site() -> None:
    assert shadow_sample(DOC, rate=0.0) is False
    assert shadow_sample(DOC, rate=1.0) is True


def test_page_number_changes_the_draw_so_a_long_document_is_not_all_or_nothing() -> None:
    per_page = {shadow_sample("doc-42", rate=0.5, page_no=n) for n in range(40)}
    assert per_page == {True, False}


def test_whole_document_sampling_ignores_the_page() -> None:
    assert shadow_sample("doc-42", rate=0.5) == shadow_sample("doc-42", rate=0.5)


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_an_out_of_range_rate_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match=r"shadow rate must be in \[0, 1\]"):
        shadow_sample(DOC, rate=rate)


def test_an_empty_doc_id_is_refused_because_the_sample_is_keyed_on_it() -> None:
    with pytest.raises(ValueError, match="doc_id must be non-empty"):
        shadow_sample("", rate=0.03)


def test_an_escalated_page_is_never_also_a_control_for_itself() -> None:
    decision = route(
        EscalationSignals(script_disagreement=True),
        doc_id=DOC,
        page_no=1,
        shadow_rate=1.0,
    )
    assert decision.escalate is True
    assert decision.shadow_sampled is False


def test_a_shadowed_page_is_marked_but_not_escalated() -> None:
    decision = route(
        EscalationSignals(), doc_id=DOC, page_no=1, shadow_rate=1.0
    )
    assert decision.escalate is False
    assert decision.shadow_sampled is True
    assert decision.reasons == (EscalationReason.SHADOW_SAMPLE,)
    assert decision.runs_tier2 is True


def test_a_shadowed_page_still_reports_the_signals_it_carried() -> None:
    """The sample is a control, so its signals are recorded for the comparison
    — but the escalation-reason metric is only incremented for escalations, so
    a shadow run cannot inflate the counts it exists to measure."""
    decision = route(
        EscalationSignals(handwriting_detected=True, handwriting_ratio=0.9),
        doc_id=DOC,
        page_no=1,
        shadow_rate=1.0,
    )
    assert decision.escalate is False
    assert decision.shadow_sampled is True
    assert decision.reasons == (
        EscalationReason.HANDWRITING_DETECTED,
        EscalationReason.HANDWRITING_MAJORITY,
        EscalationReason.SHADOW_SAMPLE,
    )


def test_the_rate_comes_from_settings_when_the_caller_does_not_override(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(ESCALATION_SHADOW_RATE=1.0)
    decision = route(EscalationSignals(), doc_id=DOC, page_no=1, settings=settings)
    assert decision.shadow_sampled is True

    off = make_settings(ESCALATION_SHADOW_RATE=0.0)
    assert (
        route(EscalationSignals(), doc_id=DOC, page_no=1, settings=off).shadow_sampled
        is False
    )


def test_the_shipped_default_rate_is_the_three_percent_the_brief_asks_for(
    settings: Settings,
) -> None:
    assert settings.ESCALATION_SHADOW_RATE == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# Shadow diffs — the record that makes the router scoreable
# --------------------------------------------------------------------------- #


def test_a_diff_counts_unchanged_replaced_added_and_lost_lines() -> None:
    tier1 = [
        _line("Name of bank", 0, y=0.0),
        _line("Acount numbr", 1, y=40.0),
        _line("only tier one saw this", 2, y=80.0),
    ]
    tier2 = [
        _line("Name of bank", 0, y=1.0),
        _line("Account number", 1, y=41.0),
        _line("only tier two saw this", 2, y=400.0),
    ]
    diff = build_shadow_diff(
        doc_id=DOC, page_no=1, tier1_lines=tier1, tier2_lines=tier2
    )
    assert diff.unchanged_lines == 1
    assert diff.changed_lines == 1
    assert diff.tier1_only_lines == 1
    assert diff.tier2_only_lines == 1
    assert 0.0 < diff.mean_text_distance < 1.0
    assert diff.changed_ratio == pytest.approx(0.5)
    assert diff.is_material is True


def test_an_identical_reading_is_a_zero_diff() -> None:
    lines = [_line("Name of bank", 0), _line("Branch", 1, y=40.0)]
    diff = build_shadow_diff(
        doc_id=DOC, page_no=1, tier1_lines=lines, tier2_lines=list(lines)
    )
    assert diff.changed_lines == 0
    assert diff.unchanged_lines == 2
    assert diff.mean_text_distance == pytest.approx(0.0)
    assert diff.changed_ratio == pytest.approx(0.0)
    assert diff.is_material is False


def test_whitespace_alone_is_not_a_disagreement() -> None:
    tier1 = [_line("Name  of   bank", 0)]
    tier2 = [_line("Name of bank", 0)]
    diff = build_shadow_diff(
        doc_id=DOC, page_no=1, tier1_lines=tier1, tier2_lines=tier2
    )
    assert diff.unchanged_lines == 1
    assert diff.mean_text_distance == pytest.approx(0.0)


def test_a_diff_carries_the_verdict_it_should_be_joined_to() -> None:
    decision = route(
        EscalationSignals(script_disagreement=True), doc_id=DOC, shadow_rate=0.0
    )
    diff = build_shadow_diff(
        doc_id=DOC,
        page_no=1,
        tier1_lines=[_line("a b c", 0)],
        tier2_lines=[_line("a b c", 0)],
        decision=decision,
        shadow=False,
    )
    assert diff.shadow is False
    assert diff.escalation_score == pytest.approx(2.0)
    assert EscalationReason.SCRIPT_DISAGREEMENT in diff.escalation_reasons


def test_a_page_tier2_read_from_nothing_is_all_additions() -> None:
    diff = build_shadow_diff(
        doc_id=DOC,
        page_no=1,
        tier1_lines=[],
        tier2_lines=[_line("recovered text", 0)],
    )
    assert diff.tier2_only_lines == 1
    assert diff.tier1_line_count == 0
    assert diff.is_material is True


# --------------------------------------------------------------------------- #
# Adjudication A-3 revisited — the eval harness's escalation reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("clean", ExpectedEscalation.NOT_EXPECTED),
        ("light", ExpectedEscalation.NOT_EXPECTED),
        ("scan", ExpectedEscalation.NOT_EXPECTED),
        ("heavy", ExpectedEscalation.EXPECTED),
        ("photo", ExpectedEscalation.EXPECTED),
        ("bounce", ExpectedEscalation.EXCLUDED),
    ],
)
def test_the_reference_follows_the_test_data_spec_profile_table(
    profile: str, expected: ExpectedEscalation
) -> None:
    assert expected_escalation(profile=profile) is expected


def test_a_bounced_page_is_excluded_not_counted_as_a_router_miss() -> None:
    """A-3 fix 1: the router is never shown a page Stage 1.5 rejected."""
    assert (
        expected_escalation(profile="bounce", handwriting_ratio=1.0)
        is ExpectedEscalation.EXCLUDED
    )
    assert (
        expected_escalation(quality_expectation="bounce")
        is ExpectedEscalation.EXCLUDED
    )


def test_a_mostly_handwritten_page_is_expected_to_escalate() -> None:
    assert (
        expected_escalation(profile="clean", handwriting_ratio=HANDWRITING_MAJORITY_RATIO)
        is ExpectedEscalation.EXPECTED
    )
    assert (
        expected_escalation(profile="clean", handwriting_ratio=0.49)
        is ExpectedEscalation.NOT_EXPECTED
    )


def test_a_code_mixed_page_is_expected_to_escalate_at_any_profile() -> None:
    """A-3 fix 3, and the reason it exists.

    Tier 1 recognises a whole page with one head, so a code-mixed page has
    lines that were provably read with the wrong one. The router escalates it
    on ``split_vote``; a reference that called it ``not_expected`` would charge
    the router a false positive for doing the right thing.
    """
    for profile in ("clean", "light", "scan"):
        assert (
            expected_escalation(profile=profile, is_code_mixed=True)
            is ExpectedEscalation.EXPECTED
        )
        assert (
            expected_escalation(profile=profile, is_code_mixed=False)
            is ExpectedEscalation.NOT_EXPECTED
        )


def test_a_bounced_code_mixed_page_is_still_excluded() -> None:
    """Exclusion outranks every positive signal: the page never ran."""
    assert (
        expected_escalation(profile="bounce", is_code_mixed=True)
        is ExpectedEscalation.EXCLUDED
    )


def test_the_reference_agrees_with_the_router_on_the_corpus_bilingual_case() -> None:
    """The measured case: a bilingual form escalates, and is expected to.

    ``split_vote`` is what Stage 2 raises for the committed ``bilingual_form``
    pages (Devanagari holds 28-30% of the characters, over the 25% runner-up
    share); this is that decision followed through the router.
    """
    decision = route(
        EscalationSignals(script_disagreement=True),
        doc_id=DOC,
        page_no=1,
        shadow_rate=0.0,
    )
    assert decision.escalate is True
    assert (
        expected_escalation(profile="clean", is_code_mixed=True)
        is ExpectedEscalation.EXPECTED
    )


def test_the_coarse_fallback_matches_the_old_stand_in_so_the_two_compare() -> None:
    """A-3 fix 2: ``degraded`` is the coarse label, kept comparable on purpose."""
    assert (
        expected_escalation(quality_expectation="clean")
        is ExpectedEscalation.NOT_EXPECTED
    )
    assert (
        expected_escalation(quality_expectation="degraded")
        is ExpectedEscalation.EXPECTED
    )


def test_the_reference_refuses_to_guess_without_evidence() -> None:
    with pytest.raises(ValueError, match="needs a profile or a quality_expectation"):
        expected_escalation()


def test_an_impossible_handwriting_ratio_is_refused() -> None:
    with pytest.raises(ValueError, match=r"handwriting_ratio must be in \[0, 1\]"):
        expected_escalation(profile="clean", handwriting_ratio=1.5)
