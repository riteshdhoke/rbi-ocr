"""Stage 4b — Tier-2 escalation: the merge policy and isolation from tier 1.

The two properties this file exists to protect:

* **Isolation.**  Every provider condition — rate limit, unreachable endpoint,
  token budget, invalid output, "this profile has no VLM" — comes back as a
  typed :class:`Tier2Result` carrying the *Tier-1* lines.  A page ships at
  Tier-1 quality with a recorded reason; it never fails the document.  A
  programming error still raises, because hiding a corrupted index space is
  worse than a failed page.
* **The merge policy's seven rules**, each asserted separately, with rule 2
  (Tier-1 ``line_index`` survives escalation) the one that would otherwise
  silently repoint every summary span taken before Tier 2 ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.contracts.errors import (
    ProviderError,
    ProviderRateLimited,
    ProviderResponseInvalid,
    ProviderUnavailable,
    TokenBudgetExceeded,
)
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, bbox_from_xyxy
from app.providers.base import ModelArtifactMissing, ProviderMethodNotSupported
from app.stages.s4b_vlm import (
    DEFAULT_IOU_THRESHOLD,
    MergePolicy,
    MergeResult,
    Tier2Outcome,
    Tier2Result,
    apply_tier2,
    bbox_iou,
    escalate_page,
    match_lines_by_iou,
    merge_tier2_lines,
)


def _line(
    text: str,
    index: int,
    *,
    y: float = 0.0,
    x: float = 10.0,
    width: float = 200.0,
    height: float = 20.0,
    language: str | None = None,
    handwritten: bool = False,
    confidence: float = 0.80,
    script: str = "Latn",
) -> OCRLine:
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(x, y, x + width, y + height),
        confidence=confidence,
        script=script,
        language=language,
        is_handwritten=handwritten,
        line_index=index,
    )


class _HeavyProvider:
    """Provider stub exposing only ``ocr_page_heavy``."""

    def __init__(self, result: list[OCRLine] | Exception) -> None:
        self._result = result
        self.calls = 0

    def ocr_page_heavy(self, page: Any) -> list[OCRLine]:
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return list(self._result)


def _page() -> np.ndarray:
    return np.full((200, 400, 3), 255, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_identical_boxes_have_iou_one() -> None:
    box = bbox_from_xyxy(0.0, 0.0, 10.0, 10.0)
    assert bbox_iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero() -> None:
    assert (
        bbox_iou(
            bbox_from_xyxy(0.0, 0.0, 10.0, 10.0),
            bbox_from_xyxy(50.0, 50.0, 60.0, 60.0),
        )
        == 0.0
    )


def test_half_overlap_scores_a_third() -> None:
    """Two 10x10 boxes sharing half their area: 50 / (100 + 100 - 50)."""
    assert bbox_iou(
        bbox_from_xyxy(0.0, 0.0, 10.0, 10.0),
        bbox_from_xyxy(5.0, 0.0, 15.0, 10.0),
    ) == pytest.approx(50.0 / 150.0)


def test_a_malformed_box_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="bbox must be 8 floats"):
        bbox_iou([0.0, 0.0, 1.0, 1.0], bbox_from_xyxy(0.0, 0.0, 1.0, 1.0))


def test_matching_is_greedy_one_to_one_and_deterministic() -> None:
    tier1 = [_line("a", 0, y=0.0), _line("b", 1, y=100.0)]
    tier2 = [
        _line("A", 0, y=2.0),
        _line("A again", 1, y=4.0),  # also overlaps tier1[0], slightly worse
        _line("B", 2, y=101.0),
    ]
    pairs, tier1_only, tier2_only = match_lines_by_iou(tier1, tier2)
    assert [(a.line_index, b.line_index) for a, b in pairs] == [(0, 0), (1, 2)]
    assert tier1_only == []
    assert [line.line_index for line in tier2_only] == [1]

    again = match_lines_by_iou(tier1, tier2)
    assert [(a.line_index, b.line_index) for a, b in again[0]] == [(0, 0), (1, 2)]


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_an_out_of_range_iou_threshold_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"iou_threshold must be in \(0, 1\]"):
        match_lines_by_iou([], [], iou_threshold=threshold)


# --------------------------------------------------------------------------- #
# The merge policy, rule by rule
# --------------------------------------------------------------------------- #


def test_rule_2_tier2_text_wins_but_the_tier1_index_survives() -> None:
    """The invariant: a summary span taken before escalation still resolves."""
    tier1 = [_line("Acount numbr", 7, confidence=0.4)]
    tier2 = [_line("Account number", 0, confidence=0.95)]

    merged = merge_tier2_lines(tier1, tier2)

    assert merged.policy is MergePolicy.IOU_MERGE
    assert len(merged.lines) == 1
    line = merged.lines[0]
    assert line.text == "Account number"
    assert line.confidence == pytest.approx(0.95)
    assert line.line_index == 7, "renumbering would repoint every summary span"
    assert merged.replaced == 1
    assert merged.unchanged == 0


def test_rule_3_a_replaced_line_loses_its_stale_language_label() -> None:
    tier1 = [_line("Acount numbr", 0, language="eng")]
    tier2 = [_line("Account number", 0)]

    merged = merge_tier2_lines(tier1, tier2)

    assert merged.lines[0].language is None
    assert merged.language_cleared == 1


def test_rule_3_an_unchanged_line_keeps_its_language() -> None:
    tier1 = [_line("Account number", 0, language="eng")]
    tier2 = [_line("Account number", 0)]

    merged = merge_tier2_lines(tier1, tier2)

    assert merged.lines[0].language == "eng"
    assert merged.language_cleared == 0
    assert merged.unchanged == 1


def test_rule_4_handwriting_is_or_ed_and_never_erased() -> None:
    tier1 = [_line("x y z", 0, handwritten=True)]
    tier2 = [_line("x y z", 0, handwritten=False)]
    assert merge_tier2_lines(tier1, tier2).lines[0].is_handwritten is True

    tier1 = [_line("x y z", 0, handwritten=False)]
    tier2 = [_line("x y z", 0, handwritten=True)]
    assert merge_tier2_lines(tier1, tier2).lines[0].is_handwritten is True


def test_rule_5_a_line_tier2_missed_is_kept_not_dropped() -> None:
    tier1 = [_line("kept", 0, y=0.0), _line("also kept", 1, y=100.0)]
    tier2 = [_line("kept", 0, y=1.0)]

    merged = merge_tier2_lines(tier1, tier2)

    assert [line.text for line in merged.lines] == ["kept", "also kept"]
    assert merged.retained == 1


def test_rule_6_new_tier2_lines_are_appended_after_the_highest_index() -> None:
    tier1 = [_line("first", 4, y=0.0)]
    tier2 = [_line("first", 0, y=1.0), _line("found by tier two", 1, y=150.0)]

    merged = merge_tier2_lines(tier1, tier2)

    assert [line.line_index for line in merged.lines] == [4, 5]
    assert merged.lines[1].text == "found by tier two"
    assert merged.added == 1


def test_rule_7_empty_tier2_output_leaves_tier1_completely_untouched() -> None:
    tier1 = [_line("a", 0, language="eng"), _line("b", 1, y=40.0)]

    merged = merge_tier2_lines(tier1, [])

    assert merged.policy is MergePolicy.TIER1_RETAINED
    assert merged.lines == tuple(tier1)
    assert merged.retained == 2
    assert merged.replaced == merged.added == merged.unchanged == 0


def test_tier2_recovering_a_page_tier1_could_not_read_renumbers_from_zero() -> None:
    tier2 = [_line("recovered", 3, y=0.0), _line("more", 9, y=40.0)]

    merged = merge_tier2_lines([], tier2)

    assert [line.line_index for line in merged.lines] == [0, 1]
    assert merged.added == 2


def test_the_merge_never_emits_a_duplicate_index() -> None:
    tier1 = [_line("a", 0, y=0.0), _line("b", 1, y=100.0)]
    tier2 = [_line("A", 0, y=1.0), _line("new", 1, y=300.0)]

    merged = merge_tier2_lines(tier1, tier2)

    indices = [line.line_index for line in merged.lines]
    assert len(indices) == len(set(indices))


def test_duplicate_input_indices_are_a_programming_error() -> None:
    duplicated = [_line("a", 0, y=0.0), _line("b", 0, y=100.0)]
    with pytest.raises(ValueError, match="tier1 lines carry duplicate line_index"):
        merge_tier2_lines(duplicated, [])
    with pytest.raises(ValueError, match="tier2 lines carry duplicate line_index"):
        merge_tier2_lines([], duplicated)


def test_the_merge_result_model_refuses_a_duplicated_index() -> None:
    with pytest.raises(ValueError, match="duplicate line_index"):
        MergeResult(
            lines=(_line("a", 0), _line("b", 0)),
            policy=MergePolicy.IOU_MERGE,
        )


def test_a_stricter_iou_threshold_treats_a_shifted_line_as_a_new_one() -> None:
    tier1 = [_line("shifted", 0, y=0.0, height=20.0)]
    tier2 = [_line("shifted", 0, y=12.0, height=20.0)]

    loose = merge_tier2_lines(tier1, tier2, iou_threshold=0.2)
    strict = merge_tier2_lines(tier1, tier2, iou_threshold=0.9)

    assert loose.unchanged == 1
    assert strict.retained == 1 and strict.added == 1


def test_the_default_threshold_matches_the_eval_harness() -> None:
    assert DEFAULT_IOU_THRESHOLD == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Isolation from the fast path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error",
    [
        ProviderRateLimited("429 from the router"),
        ProviderUnavailable("connection reset"),
        ProviderResponseInvalid("the VLM returned prose, not lines"),
        TokenBudgetExceeded("job budget spent"),
        ModelArtifactMissing(
            artifact="VLM weights",
            path=Path("models/vlm/model.onnx"),
            how_to_obtain="No approved local VLM artifact ships in Phase 0.",
        ),
    ],
)
def test_every_provider_condition_degrades_to_tier1_instead_of_raising(
    error: ProviderError,
) -> None:
    tier1 = [_line("a", 0, language="eng"), _line("b", 1, y=40.0)]
    provider = _HeavyProvider(error)

    result = escalate_page(_page(), tier1, provider=provider)

    assert result.outcome is Tier2Outcome.PROVIDER_ERROR
    assert result.error_code == error.code
    assert result.lines == tuple(tier1), "the page still ships at Tier-1 quality"
    assert result.merge.policy is MergePolicy.TIER1_RETAINED
    assert result.succeeded is False
    assert result.tier is OCRTier.TIER1


def test_a_profile_without_a_vlm_is_recorded_not_an_error() -> None:
    provider = _HeavyProvider(
        ProviderMethodNotSupported(
            "ocr_page_heavy",
            provider="local",
            implemented_by="app.providers.hf_api.HFAPIProvider",
        )
    )
    tier1 = [_line("a", 0)]

    result = escalate_page(_page(), tier1, provider=provider)

    assert result.outcome is Tier2Outcome.NOT_SUPPORTED
    assert result.error_code == "PROVIDER_METHOD_NOT_SUPPORTED"
    assert result.lines == tuple(tier1)


def test_a_programming_error_is_not_swallowed() -> None:
    """A bug must surface; a corrupted index space must not ship silently."""
    provider = _HeavyProvider(TypeError("bbox is a string"))
    with pytest.raises(TypeError):
        escalate_page(_page(), [_line("a", 0)], provider=provider)


def test_a_page_without_pixels_is_skipped_without_calling_the_provider() -> None:
    provider = _HeavyProvider([_line("never asked", 0)])
    tier1 = [_line("a", 0)]

    result = escalate_page(None, tier1, provider=provider)

    assert result.outcome is Tier2Outcome.SKIPPED
    assert provider.calls == 0
    assert result.lines == tuple(tier1)
    assert result.error_code is None


def test_a_vlm_that_returned_nothing_has_said_nothing() -> None:
    provider = _HeavyProvider([])
    tier1 = [_line("a", 0, language="eng")]

    result = escalate_page(_page(), tier1, provider=provider)

    assert result.outcome is Tier2Outcome.EMPTY
    assert result.merge.policy is MergePolicy.TIER1_RETAINED
    assert result.lines == tuple(tier1)
    assert result.tier is OCRTier.TIER1
    assert result.error_code is None


def test_a_successful_escalation_merges_and_keeps_the_raw_tier2_lines() -> None:
    tier1 = [_line("Acount numbr", 0, confidence=0.35)]
    heavy = [_line("Account number", 0, confidence=0.97)]
    provider = _HeavyProvider(heavy)

    result = escalate_page(_page(), tier1, provider=provider)

    assert result.outcome is Tier2Outcome.MERGED
    assert result.succeeded is True
    assert result.tier is OCRTier.TIER2
    assert result.lines[0].text == "Account number"
    # The unmerged Tier-2 output is kept so the shadow diff can compare tiers.
    assert result.tier2_lines == tuple(heavy)
    assert provider.calls == 1


def test_a_page_tier1_read_as_blank_is_still_worth_a_second_reader() -> None:
    provider = _HeavyProvider([_line("recovered text", 0)])

    result = escalate_page(_page(), [], provider=provider)

    assert result.outcome is Tier2Outcome.MERGED
    assert [line.text for line in result.lines] == ["recovered text"]
    assert provider.calls == 1


# --------------------------------------------------------------------------- #
# Result-model discipline
# --------------------------------------------------------------------------- #


def test_a_failed_outcome_must_carry_a_machine_readable_code() -> None:
    with pytest.raises(ValueError, match="must carry a machine-readable"):
        Tier2Result(
            outcome=Tier2Outcome.PROVIDER_ERROR,
            merge=MergeResult(lines=(), policy=MergePolicy.TIER1_RETAINED),
        )


def test_a_successful_outcome_may_not_carry_an_error_code() -> None:
    with pytest.raises(ValueError, match="set on a successful outcome"):
        Tier2Result(
            outcome=Tier2Outcome.MERGED,
            merge=MergeResult(lines=(), policy=MergePolicy.IOU_MERGE),
            error_code="PROVIDER_RATE_LIMITED",
        )


# --------------------------------------------------------------------------- #
# apply_tier2 — the tier stamp and the lines cannot get out of step
# --------------------------------------------------------------------------- #


def _ocr_page(lines: list[OCRLine], tier: OCRTier = OCRTier.TIER1) -> OCRPage:
    return OCRPage(
        page_no=1, width_px=400, height_px=200, lines=lines, tier=tier
    )


def test_apply_tier2_stamps_tier2_only_when_tier2_contributed() -> None:
    tier1 = [_line("Acount numbr", 0)]
    merged = escalate_page(
        _page(), tier1, provider=_HeavyProvider([_line("Account number", 0)])
    )
    page = apply_tier2(_ocr_page(tier1), merged)
    assert page.tier is OCRTier.TIER2
    assert page.lines[0].text == "Account number"


def test_apply_tier2_leaves_tier1_stamped_when_escalation_failed() -> None:
    tier1 = [_line("a", 0)]
    failed = escalate_page(
        _page(), tier1, provider=_HeavyProvider(ProviderRateLimited("429"))
    )
    page = apply_tier2(_ocr_page(tier1), failed)
    assert page.tier is OCRTier.TIER1
    assert page.lines == tier1


def test_apply_tier2_does_not_demote_a_bypass_page_that_had_no_lines() -> None:
    """A digital text layer that Tier 2 could not improve is still a bypass."""
    empty_bypass = _ocr_page([], tier=OCRTier.BYPASS)
    skipped = escalate_page(None, [], provider=_HeavyProvider([]))
    assert apply_tier2(empty_bypass, skipped).tier is OCRTier.BYPASS


def test_apply_tier2_marks_a_page_nobody_could_read_as_no_tier() -> None:
    page = _ocr_page([], tier=OCRTier.TIER1)
    nothing = escalate_page(None, [], provider=_HeavyProvider([]))
    assert apply_tier2(page, nothing).tier is OCRTier.NONE
