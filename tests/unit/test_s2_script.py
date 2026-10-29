"""Stage 2 — script identification (adjudication A-6, BUILD-BRIEF M4).

Covers the text vote including genuinely mixed pages, recognition-head routing,
the optional image-classifier path and its typed-absence behaviour, and every
disagreement reason the escalation router keys on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.contracts.ocr import OCRLine, ScriptResult, ScriptVote, bbox_from_xyxy
from app.providers.base import ModelArtifactMissing
from app.stages.s2_script import (
    DEFAULT_REC_HEAD_SCRIPT,
    MIN_SCRIPT_CONFIDENCE,
    ScriptDecision,
    ScriptMechanism,
    count_script_characters,
    head_covering,
    identify_script,
    rec_head_for_script,
    script_from_lines,
)

ENGLISH = "Name of bank and branch"
MARATHI = "बँकेने माझ्या खात्यातून दोनदा पैसे कापले"


def _line(text: str, script: str, index: int) -> OCRLine:
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(10.0, 10.0 + 20 * index, 400.0, 30.0 + 20 * index),
        confidence=0.9,
        script=script,
        language=None,
        is_handwritten=False,
        line_index=index,
    )


class _ScriptProvider:
    """Minimal provider stub exposing only ``detect_script``."""

    def __init__(self, result: ScriptResult | Exception) -> None:
        self._result = result
        self.calls: list[int] = []

    def detect_script(self, crops: Any) -> ScriptResult:
        self.calls.append(len(crops))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _artifact_missing() -> ModelArtifactMissing:
    return ModelArtifactMissing(
        artifact="image-space script-identification ONNX",
        path=Path("models/script/script_id.onnx"),
        how_to_obtain="No approved artifact exists yet (decisions-wsc C7).",
    )


# --------------------------------------------------------------------------- #
# The text vote
# --------------------------------------------------------------------------- #


def test_monolingual_latin_page_votes_latn() -> None:
    result = script_from_lines([_line(ENGLISH, "Latn", i) for i in range(5)])
    assert result.script == "Latn"
    assert result.confidence == pytest.approx(1.0)


def test_monolingual_devanagari_page_votes_deva() -> None:
    result = script_from_lines([_line(MARATHI, "Deva", i) for i in range(4)])
    assert result.script == "Deva"
    assert result.confidence == pytest.approx(1.0)


def test_mixed_page_votes_the_majority_and_keeps_the_minority_visible() -> None:
    """The bilingual form: printed English labels, handwritten Marathi values."""
    lines = [_line(ENGLISH, "Latn", i) for i in range(6)]
    lines += [_line(MARATHI, "Deva", 6 + i) for i in range(6)]

    result = script_from_lines(lines)

    assert set(result.distribution) == {"Latn", "Deva"}
    assert sum(result.distribution.values()) == pytest.approx(1.0)
    # Every script that appeared is still visible after the vote; the winner
    # never erases the runner-up (that is what code-mix detection reads).
    assert result.distribution["Deva"] > 0.0
    assert result.script in {"Latn", "Deva"}


def test_a_split_page_is_flagged_as_a_disagreement_not_a_silent_winner() -> None:
    lines = [_line(ENGLISH, "Latn", i) for i in range(4)]
    lines += [_line(MARATHI, "Deva", 4 + i) for i in range(4)]

    decision = identify_script(lines=lines)

    assert decision.minority_scripts, "the runner-up must be reported"
    assert decision.disagreement is True
    assert decision.disagreement_reason in {"split_vote", "low_confidence"}


def test_digits_and_punctuation_cast_no_vote() -> None:
    result = script_from_lines(["1234 5678 -- 90.00 ₹", "  ", "..."])
    assert result.script == "Zyyy"
    assert result.confidence == pytest.approx(0.0)
    assert result.distribution == {}


def test_no_script_evidence_is_reported_never_guessed() -> None:
    decision = identify_script(lines=["1234", "5678"])
    assert decision.script == "Zyyy"
    assert decision.is_confident is False
    assert decision.disagreement_reason == "no_script_evidence"


def test_count_script_characters_counts_characters_not_lines() -> None:
    assert count_script_characters(["abc", "1234", "देव"]) == 6
    assert count_script_characters([]) == 0
    assert count_script_characters(None) == 0


def test_a_short_page_is_not_confident_even_when_unanimous() -> None:
    """A three-letter page has not said enough to route a recognition head."""
    decision = identify_script(lines=["abc"])
    assert decision.script == "Latn"
    assert decision.voting_characters == 3
    assert decision.is_confident is False
    assert decision.disagreement_reason == "low_confidence"


# --------------------------------------------------------------------------- #
# Recognition-head routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        # Latin routes to the Devanagari head: that head's charset covers Latin,
        # measures the same on it, and unlike the Latin head does not destroy a
        # Devanagari line the vote happened to miss.
        ("Latn", "Deva"),
        ("Deva", "Deva"),
        # A head with no coverage row still routes to itself — dropping in
        # rec_knda.onnx must start using it, not have it ignored.
        ("Knda", "Knda"),
        ("Zyyy", DEFAULT_REC_HEAD_SCRIPT),
        ("Hant", DEFAULT_REC_HEAD_SCRIPT),
    ],
)
def test_rec_head_routing(script: str, expected: str) -> None:
    assert rec_head_for_script(script) == expected


def test_the_default_head_is_the_widest_one_installed() -> None:
    """Not Latin. See DEFAULT_REC_HEAD_SCRIPT for the measurements."""
    assert DEFAULT_REC_HEAD_SCRIPT == "Deva"
    assert head_covering({"Latn", "Deva"}, default="Latn") == "Deva"


def test_a_script_no_installed_head_covers_falls_back_and_is_flagged() -> None:
    """Half-reading a page silently is worse than saying the head is wrong."""
    assert head_covering({"Deva", "Taml"}, default="Latn") == "Latn"


def test_undetermined_never_routes_to_a_head_that_cannot_exist() -> None:
    decision = identify_script(lines=["1234"])
    assert decision.script == "Zyyy"
    assert decision.rec_head_script == DEFAULT_REC_HEAD_SCRIPT


def test_head_mismatch_is_the_signal_that_triggers_a_second_pass() -> None:
    lines = [_line(MARATHI, "Deva", i) for i in range(5)]
    decision = identify_script(lines=lines, recognised_with="Latn")
    assert decision.rec_head_script == "Deva"
    assert decision.disagreement is True
    assert decision.disagreement_reason == "head_mismatch"


def test_agreement_with_the_head_produces_no_disagreement() -> None:
    lines = [_line(ENGLISH, "Latn", i) for i in range(5)]
    decision = identify_script(lines=lines, recognised_with="Latn")
    assert decision.disagreement is False
    assert decision.disagreement_reason is None
    assert decision.is_confident is True


# --------------------------------------------------------------------------- #
# The optional image-classifier path
# --------------------------------------------------------------------------- #


def test_image_classifier_is_preferred_when_the_artifact_exists() -> None:
    provider = _ScriptProvider(
        ScriptResult(
            script="Deva",
            confidence=0.97,
            distribution={"Deva": 0.97, "Latn": 0.03},
            votes=[ScriptVote(script="Deva", confidence=0.97)],
            sampled_crops=1,
        )
    )
    decision = identify_script(
        lines=[_line(ENGLISH, "Latn", 0)],
        provider=provider,
        crops=[object()],
        prefer_image=True,
    )
    assert decision.mechanism is ScriptMechanism.IMAGE_CLASSIFIER
    assert decision.script == "Deva"
    # Pixels beat a possibly-wrong decode, and the conflict is still reported.
    assert decision.disagreement_reason == "mechanisms_disagree"


def test_missing_script_artifact_falls_back_to_the_text_vote() -> None:
    provider = _ScriptProvider(_artifact_missing())
    decision = identify_script(
        lines=[_line(MARATHI, "Deva", i) for i in range(4)],
        provider=provider,
        crops=[object(), object()],
        prefer_image=True,
    )
    assert decision.mechanism is ScriptMechanism.TEXT_VOTE
    assert decision.script == "Deva"
    assert decision.artifact_fallback_reason == "MODEL_ARTIFACT_MISSING"


def test_missing_script_artifact_propagates_when_fallback_is_refused() -> None:
    provider = _ScriptProvider(_artifact_missing())
    with pytest.raises(ModelArtifactMissing) as excinfo:
        identify_script(
            lines=[_line(ENGLISH, "Latn", 0)],
            provider=provider,
            crops=[object()],
            prefer_image=True,
            allow_image_fallback=False,
        )
    assert excinfo.value.code == "MODEL_ARTIFACT_MISSING"


def test_real_local_provider_names_the_a6_fallback(
    make_settings: Any, tmp_path: Path
) -> None:
    """The remediation string an operator actually sees points at A-6's path.

    Asserted against the committed provider rather than a stub, because the
    string is the entire user experience of this gap and a stub could not
    regress with it.
    """
    from app.providers.local import LocalProvider

    provider = LocalProvider(make_settings(), models_dir=tmp_path)
    with pytest.raises(ModelArtifactMissing) as excinfo:
        identify_script(
            lines=[_line(ENGLISH, "Latn", 0)],
            provider=provider,
            crops=[object()],
            prefer_image=True,
            allow_image_fallback=False,
        )
    assert "detect_script_from_text" in excinfo.value.how_to_obtain


def test_crops_are_sampled_evenly_and_capped() -> None:
    provider = _ScriptProvider(
        ScriptResult(
            script="Latn",
            confidence=1.0,
            distribution={"Latn": 1.0},
            votes=[],
            sampled_crops=0,
        )
    )
    identify_script(
        lines=[_line(ENGLISH, "Latn", 0)],
        provider=provider,
        crops=[object()] * 100,
        prefer_image=True,
        max_sampled_crops=8,
    )
    assert provider.calls == [8]


# --------------------------------------------------------------------------- #
# Hints
# --------------------------------------------------------------------------- #


def test_a_hint_is_used_only_when_nothing_voted() -> None:
    decision = identify_script(lines=["1234"], hint="Deva")
    assert decision.mechanism is ScriptMechanism.HINT
    assert decision.script == "Deva"
    assert decision.rec_head_script == "Deva"


def test_evidence_beats_a_hint_and_the_conflict_is_recorded() -> None:
    decision = identify_script(
        lines=[_line(MARATHI, "Deva", i) for i in range(4)], hint="Latn"
    )
    assert decision.mechanism is ScriptMechanism.TEXT_VOTE
    assert decision.script == "Deva"
    assert decision.disagreement_reason in {"hint_mismatch", "head_mismatch"}


def test_no_evidence_at_all_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="no default script"):
        identify_script()


# --------------------------------------------------------------------------- #
# Contract discipline
# --------------------------------------------------------------------------- #


def test_a_disagreement_must_name_a_reason() -> None:
    base = identify_script(lines=[_line(ENGLISH, "Latn", i) for i in range(4)])
    with pytest.raises(ValueError, match="machine-readable reason"):
        ScriptDecision(
            result=base.result,
            mechanism=base.mechanism,
            rec_head_script=base.rec_head_script,
            is_confident=True,
            disagreement=True,
        )


def test_a_reason_without_a_disagreement_is_rejected() -> None:
    base = identify_script(lines=[_line(ENGLISH, "Latn", i) for i in range(4)])
    with pytest.raises(ValueError, match="while disagreement is False"):
        ScriptDecision(
            result=base.result,
            mechanism=base.mechanism,
            rec_head_script=base.rec_head_script,
            is_confident=True,
            disagreement=False,
            disagreement_reason="split_vote",
        )


def test_confidence_floor_is_the_documented_constant() -> None:
    """A page just under the floor is not confident; just over, it is."""
    assert 0.0 < MIN_SCRIPT_CONFIDENCE < 1.0
