"""Stage 2 — page script identification (BUILD-BRIEF M4, adjudication A-6).

The mechanism is fixed by **adjudication A-6**: the page's script is a majority
vote over the *text* Tier-1 recognition produced, computed by
:func:`app.providers.local.detect_script_from_text`, which needs no weights at
all.  The image-space classifier (``models/script/script_id.onnx``) stays as an
**optional** drop-in: ask for it with ``prefer_image=True`` and, when the
artifact is absent, the provider raises the typed
:class:`~app.providers.base.ModelArtifactMissing` that this stage either
propagates (``allow_image_fallback=False``) or records while falling back to the
text vote.  Nothing here ever guesses a script.

The result drives **recognition-head routing**: ``ScriptDecision.rec_head_script``
is the string handed to ``provider.ocr_page(page, script)``, whose LRU/TTL head
cache lives in the provider (BUILD-BRIEF M4, decision C21).

The honest weakness of A-6, stated once here and again in
``docs/decisions-wse2.md``
--------------------------------------------------------------------------
Voting on Tier-1 text is *circular* when Tier-1 ran with the wrong head: a
Devanagari page recognised by a Latin head emits Latin-looking garbage, and the
vote will happily confirm "Latn".  Three things bound the damage, and none of
them is a guess:

1. :func:`app.stages.s4_ocr.run_tier1` runs a **second recognition pass** when
   the vote disagrees with the head that produced the text, so a page that does
   decode to Devanagari is re-recognised with the Devanagari head;
2. :attr:`ScriptDecision.disagreement` is raised for a *low-confidence or split*
   vote, and Stage 4's router keys tier-2 escalation on it — the VLM is the
   actual safety net for a page whose script we could not settle;
3. dropping ``script_id.onnx`` into ``models/script/`` closes the loop properly,
   and this module already calls it when asked.

This module does NOT handle:
  * **language**.  Script is not language; ``hin``/``mar``/``kok``/``nep`` all
    write Devanagari and only Stage 2b may choose between them (PLAN.md
    invariant 1);
  * writing anything onto a page.  A :class:`ScriptDecision` is a stage result;
    ``OCRPage`` refuses page-level script by construction, and the envelope's
    ``primary_script`` is derived in ``assemble.py``;
  * loading, converting or downloading the script-ID artifact — the provider
    owns every model file (decision C20);
  * running OCR.  It votes over text somebody else recognised, and the
    two-pass orchestration is :mod:`app.stages.s4_ocr`'s.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Final, Self

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.ocr import OCRLine, ScriptCode, ScriptResult
from app.providers.base import CropImage, ModelArtifactMissing, ModelProvider
from app.providers.local import (
    REC_HEAD_COVERAGE,
    SCRIPT_LANGUAGES,
    UNDETERMINED_SCRIPT,
    detect_script_from_text,
    script_of_char,
)

__all__ = [
    "DEFAULT_REC_HEAD_SCRIPT",
    "MIN_SCRIPT_CONFIDENCE",
    "MIN_VOTING_CHARACTERS",
    "REC_HEAD_SCRIPTS",
    "SECOND_SCRIPT_SHARE",
    "ScriptDecision",
    "ScriptMechanism",
    "count_script_characters",
    "head_covering",
    "identify_script",
    "rec_head_for_script",
    "script_from_crops",
    "script_from_lines",
    "subsumes_heads",
    "widest_coverage_head",
]

_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables.  Module constants rather than Settings fields because app/settings.py
# is another workstream's committed lane; every one is overridable per call and
# every one is listed in docs/decisions-wse2.md for promotion to `Settings`.
# --------------------------------------------------------------------------- #

#: Below this share of script-bearing characters the vote is not decisive and
#: :attr:`ScriptDecision.disagreement` is raised for the escalation router.
MIN_SCRIPT_CONFIDENCE: Final[float] = 0.60

#: A page with fewer script-bearing characters than this has not said enough to
#: be voted on — think a cheque with a MICR line and three numerals.
MIN_VOTING_CHARACTERS: Final[int] = 8

#: A runner-up holding at least this share of the characters makes the page
#: genuinely mixed, which is a disagreement even when the winner is confident.
SECOND_SCRIPT_SHARE: Final[float] = 0.25

#: Scripts for which a recognition head may exist.  Deliberately the provider's
#: own ``SCRIPT_LANGUAGES`` keys: adding Kannada is a row there plus a model
#: drop, never a code change here (BUILD-BRIEF M4).
REC_HEAD_SCRIPTS: Final[frozenset[str]] = frozenset(SCRIPT_LANGUAGES)


def _coverage_of(head: str) -> frozenset[str]:
    """Scripts ``head`` can emit; a head with no declared row covers its own.

    The fallback matters: :data:`REC_HEAD_SCRIPTS` comes from the provider's
    language table and lists every script a head *could* exist for, while
    :data:`~app.providers.local.REC_HEAD_COVERAGE` only describes the charsets
    actually measured.  Dropping in ``rec_taml.onnx`` must start using it
    immediately, not have it ignored for want of a coverage row.
    """
    return REC_HEAD_COVERAGE.get(head, frozenset({head}))


def head_covering(scripts: Iterable[str], *, default: str) -> str:
    """The best installed head whose charset covers every script in ``scripts``.

    "Best" is the **widest** covering head.  The intuition says a purpose-built
    head should beat a general one, so a Latin page should prefer the Latin
    head; measured on the corpus the two are simply equal on Latin (CER
    0.000-0.002 either way), while only the wider head can read a line in the
    script it was not told to expect.  Equal accuracy plus strictly greater
    coverage is not a trade-off, so the wider head wins.  Revisit this the day
    a narrower head measures *better*, and put the numbers next to
    :data:`DEFAULT_REC_HEAD_SCRIPT` when you do.

    Ties break alphabetically, so the choice is deterministic and never depends
    on mapping iteration order.  When nothing covers the whole set — a page
    mixing Devanagari and Tamil on a node with no head for both — ``default``
    is returned and the caller's disagreement flag carries the escalation,
    because half-reading a page silently is the one outcome worse than saying
    so.
    """
    wanted = {str(script) for script in scripts}
    candidates = [
        head for head in REC_HEAD_SCRIPTS if wanted <= _coverage_of(head)
    ]
    if not candidates:
        return default
    return max(candidates, key=lambda head: (len(_coverage_of(head)), head))


def subsumes_heads(head: str, candidates: Iterable[str]) -> bool:
    """Whether ``head`` can emit everything ``candidates`` between them can.

    This is the test for "no candidate head could tell us anything this one
    cannot", and therefore for when the dual-head probe has nothing to
    arbitrate.  ``candidates`` is the probe's own list rather than
    :data:`REC_HEAD_SCRIPTS`, which enumerates every script a head *could*
    exist for (eleven of them) and not the two or three actually installed.

    Deliberately stricter than "covers the script we voted".  A Latin head
    reading a Devanagari page emits Latin noise and the A-6 vote confirms
    ``Latn``, so that head's own verdict is exactly the circular answer the
    probe exists to catch (OPEN-QUESTIONS Q12).  A head that subsumes the
    others has no such blind spot: it could have emitted the other script and
    did not.
    """
    reachable: set[str] = set()
    for candidate in candidates:
        reachable |= _coverage_of(str(candidate))
    return bool(reachable) and reachable <= _coverage_of(head)


def widest_coverage_head(default: str = "Latn") -> str:
    """The installed head that can emit the most scripts.

    Used as the bootstrap head, where nothing is yet known about the page and
    the safest guess is the head that can represent the most of it.
    """
    if not REC_HEAD_SCRIPTS:
        return default
    return max(
        REC_HEAD_SCRIPTS, key=lambda head: (len(_coverage_of(head)), head)
    )


#: The head used when the vote produced nothing usable, and the head every page
#: bootstraps on.
#:
#: This is **the widest-coverage installed head**, not Latin.  Latin was the
#: original choice on the reasoning that an RBI form set is mostly English.
#: The trouble is that a bilingual form — the common case, not the corner one —
#: carries both scripts on one page, and a head can only emit the characters in
#: its own charset, so the Latin head turns every Devanagari line into rubble:
#: measured on the corpus, Devanagari reads at CER 0.90 under the Latin head
#: and 0.009-0.042 under the Devanagari head, on identical pixels.
#:
#: The other half of the argument is that the wider head costs nothing.  On
#: Latin the two measure the same — 0.000-0.002 either way — so there is no
#: English page that the narrower head reads better.
#:
#: (An earlier revision of this comment claimed the Devanagari head was
#: dramatically better on Latin too.  That was an artifact: the measurement
#: predated the :data:`~app.providers.local.REC_WIDTH_BUCKETS` fix, and what it
#: actually detected was long English lines being truncated by the old 640 px
#: clamp.  With that fixed the heads are equal on Latin.  Kept as a note
#: because the wrong number pointed at the right decision for the wrong
#: reason, and the next person deserves to know which is which.)
#:
#: So the rule is coverage, not majority: prefer the head that can represent
#: the most scripts, because a head that cannot emit a character can never
#: recover it, while a head with a wider charset is free on a page that does
#: not need the extra range.
DEFAULT_REC_HEAD_SCRIPT: Final[str] = widest_coverage_head()

_SCRIPT_DECISIONS = Counter(
    "stage_s2_script_decisions_total",
    "Stage-2 script decisions, by mechanism and winning script.",
    ["mechanism", "script"],
)
_SCRIPT_FALLBACKS = Counter(
    "stage_s2_script_artifact_fallbacks_total",
    "Times the image-space script classifier was unavailable and the text "
    "vote was used instead.",
)
_SCRIPT_LATENCY = Histogram(
    "stage_s2_script_seconds",
    "Wall time of one Stage-2 script identification.",
)


class ScriptMechanism(StrEnum):
    """How a :class:`ScriptDecision` was reached.  Never inferred later."""

    #: Unicode-range majority vote over Tier-1 text (adjudication A-6).
    TEXT_VOTE = "text_vote"
    #: ``models/script/script_id.onnx`` over sampled line crops.
    IMAGE_CLASSIFIER = "image_classifier"
    #: A caller-supplied hint was accepted without a vote.
    HINT = "hint"
    #: Nothing voted: no lines, or no script-bearing characters.
    NONE = "none"


class ScriptDecision(BaseModel):
    """Stage-2 output: the vote, how it was reached, and what it routes to.

    ``rec_head_script`` is the only value a caller should hand to
    ``provider.ocr_page``; ``result.script`` may legitimately be ``Zyyy``, and
    there is no ``rec_zyyy.onnx``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: ScriptResult
    mechanism: ScriptMechanism
    #: ISO 15924 code to select the recognition head with.
    rec_head_script: ScriptCode
    #: True when the winning script cleared :data:`MIN_SCRIPT_CONFIDENCE` and
    #: enough characters voted.
    is_confident: bool
    #: The escalation-router signal (PLAN.md invariant 3, never CTC confidence).
    disagreement: bool
    #: Machine-readable why, e.g. ``low_confidence``, ``split_vote``,
    #: ``mechanisms_disagree``, ``head_mismatch``. ``None`` when settled.
    disagreement_reason: str | None = None
    #: Scripts other than the winner that cleared :data:`SECOND_SCRIPT_SHARE`.
    minority_scripts: tuple[str, ...] = ()
    #: Set when the image classifier was asked for and was not available.
    artifact_fallback_reason: str | None = None
    #: Script-bearing characters that voted; 0 for the image mechanism.
    voting_characters: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _reason_matches_flag(self) -> Self:
        if self.disagreement and not self.disagreement_reason:
            raise ValueError(
                "a script disagreement must carry a machine-readable reason: "
                "the escalation router records it (PLAN.md invariant 3)"
            )
        if not self.disagreement and self.disagreement_reason:
            raise ValueError(
                f"disagreement_reason {self.disagreement_reason!r} set while "
                "disagreement is False"
            )
        return self

    @property
    def script(self) -> str:
        """Shorthand for ``result.script`` (ISO 15924, may be ``Zyyy``)."""
        return self.result.script

    @property
    def confidence(self) -> float:
        """Shorthand for ``result.confidence``."""
        return self.result.confidence


def rec_head_for_script(
    script: str, *, default: str = DEFAULT_REC_HEAD_SCRIPT
) -> str:
    """Recognition-head script code for a voted script.

    ``Zyyy`` (and any script with no configured language set, i.e. no head we
    could ever have shipped) maps to ``default``.  Returning ``Zyyy`` would send
    the provider looking for ``rec_zyyy.onnx``, and a
    :class:`~app.providers.base.ModelArtifactMissing` naming a file that cannot
    exist is a worse error message than an honest fallback plus a disagreement
    flag.

    A head **wider** than the voted script wins when one exists: a page voted
    ``Latn`` is read by the Devanagari head because that head covers Latin too,
    reads it better, and — unlike the Latin head — does not silently destroy
    any Devanagari the vote happened to miss.  See
    :data:`DEFAULT_REC_HEAD_SCRIPT` for the measurements.
    """
    if script not in REC_HEAD_SCRIPTS:
        return default
    return head_covering({script}, default=script)


def script_from_lines(lines: Sequence[OCRLine] | Sequence[str]) -> ScriptResult:
    """Majority-vote the page script from Tier-1 text (adjudication A-6).

    Accepts :class:`~app.contracts.ocr.OCRLine` objects or bare strings, so the
    same function serves Stage 4's second-pass check and the eval harness.
    Delegates the character-range work to the provider layer rather than
    re-implementing it — one table of Unicode ranges, in one place.
    """
    texts = [
        line.text if isinstance(line, OCRLine) else str(line) for line in lines
    ]
    return detect_script_from_text(texts)


def script_from_crops(
    provider: ModelProvider, crops: Sequence[CropImage]
) -> ScriptResult:
    """Image-space script vote over sampled line crops (the optional path).

    Args:
        provider: any :class:`~app.providers.base.ModelProvider`; the composite
            profiles route ``detect_script`` to the local one.
        crops: NumPy ``uint8`` BGR (or grayscale) arrays — decision C1.

    Raises:
        ModelArtifactMissing: when ``models/script/script_id.onnx`` is absent,
            which is its state on every machine today (decision C7). The caller
            decides whether that is fatal; :func:`identify_script` catches it.
    """
    return provider.detect_script(crops)


def count_script_characters(
    lines: Sequence[OCRLine] | Sequence[str] | None,
) -> int:
    """Characters that carry a script vote (letters, not digits or punctuation).

    :class:`~app.contracts.ocr.ScriptResult` records per-*line* votes and
    normalised shares, not raw counts, so the "did enough evidence speak?" gate
    has to count for itself.  Counting lines instead would make a seven-line
    cheque structurally unable to be confident about its own script.
    """
    if not lines:
        return 0
    total = 0
    for line in lines:
        text = line.text if isinstance(line, OCRLine) else str(line)
        for character in text:
            if script_of_char(character) is not None:
                total += 1
    return total


def _sample_crops(
    crops: Sequence[CropImage], limit: int
) -> tuple[CropImage, ...]:
    """Evenly sample at most ``limit`` crops, preserving order.

    Even sampling rather than the first ``limit``: a form's first lines are its
    printed English header, and sampling only those is how a bilingual page gets
    called Latin.
    """
    total = len(crops)
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if total <= limit:
        return tuple(crops)
    step = total / limit
    picked = sorted({min(total - 1, int(index * step)) for index in range(limit)})
    return tuple(crops[index] for index in picked)


def identify_script(
    *,
    lines: Sequence[OCRLine] | Sequence[str] | None = None,
    provider: ModelProvider | None = None,
    crops: Sequence[CropImage] | None = None,
    hint: str | None = None,
    prefer_image: bool = False,
    allow_image_fallback: bool = True,
    recognised_with: str | None = None,
    min_confidence: float = MIN_SCRIPT_CONFIDENCE,
    min_characters: int = MIN_VOTING_CHARACTERS,
    max_sampled_crops: int = 24,
) -> ScriptDecision:
    """Decide the page script and the recognition head it routes to.

    Order of mechanisms:

    1. ``prefer_image`` **and** a provider **and** crops: the image classifier.
       On :class:`~app.providers.base.ModelArtifactMissing` this either
       propagates (``allow_image_fallback=False``) or falls through with
       ``artifact_fallback_reason`` recorded.
    2. Tier-1 text (adjudication A-6) — the Phase-0/1 mechanism.
    3. ``hint``, when there was no text to vote on.  A hint never *overrides* a
       decisive vote; when both exist and disagree, the vote wins and
       ``disagreement`` is raised, because the hint is caller metadata and the
       vote is evidence.

    Args:
        recognised_with: the script the text in ``lines`` was recognised with.
            Supplying it turns "the vote disagrees with the head that produced
            it" into an explicit ``head_mismatch`` disagreement, which is what
            makes Stage 4's second pass and the escalation signal possible.

    Raises:
        ValueError: when neither ``lines``, ``crops`` nor ``hint`` is supplied —
            this stage has nothing to be silent about.
        ModelArtifactMissing: only with ``prefer_image=True`` and
            ``allow_image_fallback=False``.
    """
    if lines is None and crops is None and hint is None:
        raise ValueError(
            "identify_script needs lines, crops or a hint; there is no "
            "default script (BUILD-BRIEF sec. 0.5)"
        )
    with _SCRIPT_LATENCY.time():
        return _identify(
            lines=lines,
            provider=provider,
            crops=crops,
            hint=hint,
            prefer_image=prefer_image,
            allow_image_fallback=allow_image_fallback,
            recognised_with=recognised_with,
            min_confidence=min_confidence,
            min_characters=min_characters,
            max_sampled_crops=max_sampled_crops,
        )


def _identify(
    *,
    lines: Sequence[OCRLine] | Sequence[str] | None,
    provider: ModelProvider | None,
    crops: Sequence[CropImage] | None,
    hint: str | None,
    prefer_image: bool,
    allow_image_fallback: bool,
    recognised_with: str | None,
    min_confidence: float,
    min_characters: int,
    max_sampled_crops: int,
) -> ScriptDecision:
    """Body of :func:`identify_script`, split out so the timer wraps one call."""
    fallback_reason: str | None = None
    image_result: ScriptResult | None = None

    if prefer_image and provider is not None and crops:
        try:
            image_result = script_from_crops(
                provider, _sample_crops(crops, max_sampled_crops)
            )
        except ModelArtifactMissing as exc:
            if not allow_image_fallback:
                raise
            _SCRIPT_FALLBACKS.inc()
            fallback_reason = exc.code
            _LOG.info(
                "script_id artifact unavailable (%s); falling back to the "
                "Tier-1 text vote per adjudication A-6",
                exc.artifact,
            )

    text_result = script_from_lines(lines) if lines else None

    chosen, mechanism = _choose_mechanism(image_result, text_result, hint)
    voting_characters = (
        count_script_characters(lines)
        if mechanism is ScriptMechanism.TEXT_VOTE
        else 0
    )

    minority = tuple(
        sorted(
            script
            for script, share in chosen.distribution.items()
            if script != chosen.script and share >= SECOND_SCRIPT_SHARE
        )
    )
    is_confident = (
        chosen.script != UNDETERMINED_SCRIPT
        and chosen.confidence >= min_confidence
        and (
            mechanism is not ScriptMechanism.TEXT_VOTE
            or voting_characters >= min_characters
        )
    )
    reason = _disagreement_reason(
        chosen=chosen,
        mechanism=mechanism,
        is_confident=is_confident,
        minority=minority,
        image_result=image_result,
        text_result=text_result,
        hint=hint,
        recognised_with=recognised_with,
    )

    decision = ScriptDecision(
        result=chosen,
        mechanism=mechanism,
        rec_head_script=rec_head_for_script(chosen.script),
        is_confident=is_confident,
        disagreement=reason is not None,
        disagreement_reason=reason,
        minority_scripts=minority,
        artifact_fallback_reason=fallback_reason,
        voting_characters=voting_characters,
    )
    _SCRIPT_DECISIONS.labels(
        mechanism=decision.mechanism.value, script=decision.script
    ).inc()
    _LOG.debug(
        "stage s2 script=%s mechanism=%s confident=%s disagreement=%s",
        decision.script,
        decision.mechanism.value,
        decision.is_confident,
        decision.disagreement_reason,
    )
    return decision


def _choose_mechanism(
    image_result: ScriptResult | None,
    text_result: ScriptResult | None,
    hint: str | None,
) -> tuple[ScriptResult, ScriptMechanism]:
    """Pick the winning result and name the mechanism that produced it.

    The image classifier wins when it ran, because it is the only mechanism that
    is not circular (it looks at pixels, not at what a possibly-wrong head
    decoded).  The text vote is next.  A hint is used only when nothing voted.
    """
    if image_result is not None and image_result.script != UNDETERMINED_SCRIPT:
        return image_result, ScriptMechanism.IMAGE_CLASSIFIER
    if text_result is not None and text_result.script != UNDETERMINED_SCRIPT:
        return text_result, ScriptMechanism.TEXT_VOTE
    if hint is not None:
        return (
            ScriptResult(
                script=hint,
                confidence=1.0,
                distribution={hint: 1.0},
                votes=[],
                sampled_crops=0,
            ),
            ScriptMechanism.HINT,
        )
    empty = text_result or image_result
    return (
        empty
        if empty is not None
        else ScriptResult(
            script=UNDETERMINED_SCRIPT,
            confidence=0.0,
            distribution={},
            votes=[],
            sampled_crops=0,
        ),
        ScriptMechanism.NONE,
    )


def _disagreement_reason(
    *,
    chosen: ScriptResult,
    mechanism: ScriptMechanism,
    is_confident: bool,
    minority: tuple[str, ...],
    image_result: ScriptResult | None,
    text_result: ScriptResult | None,
    hint: str | None,
    recognised_with: str | None,
) -> str | None:
    """The single machine-readable reason the page's script is unsettled.

    Checked most-specific first so the reason a human reads is the one that
    matters; the router treats them all alike (invariant 3 keys on the *fact* of
    disagreement, never on a confidence number).

    ``head_mismatch`` asks whether the head that read the page **could emit**
    the script that was voted, not whether it is the head we would pick today.
    The two stopped being the same question once a head could cover more than
    one script: a Latin page read by the Latin head is read correctly, and
    routing it to Tier 2 because the Devanagari head is now preferred would
    escalate a large share of ordinary traffic for nothing.  A Devanagari page
    read by the Latin head is a real mismatch — that head has one Devanagari
    codepoint in its charset — and still raises.
    """
    if (
        image_result is not None
        and text_result is not None
        and image_result.script != UNDETERMINED_SCRIPT
        and text_result.script != UNDETERMINED_SCRIPT
        and image_result.script != text_result.script
    ):
        return "mechanisms_disagree"
    if (
        recognised_with is not None
        and chosen.script != UNDETERMINED_SCRIPT
        and chosen.script not in _coverage_of(recognised_with)
    ):
        return "head_mismatch"
    if mechanism is ScriptMechanism.NONE:
        return "no_script_evidence"
    if not is_confident:
        return "low_confidence"
    if minority:
        return "split_vote"
    if (
        hint is not None
        and chosen.script != UNDETERMINED_SCRIPT
        and hint != chosen.script
        and mechanism is not ScriptMechanism.HINT
    ):
        return "hint_mismatch"
    return None
