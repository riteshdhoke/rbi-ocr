"""Stage 2b — **line-level** language identification (PLAN.md invariant 1).

The defining RBI complaint is a printed English form carrying a handwritten
Marathi free-text block.  One page-level language label silently corrupts every
stage downstream of it, which is why ``language`` is an attribute of
:class:`~app.contracts.ocr.OCRLine` and why this module writes it **one line at
a time**.  ``OCRPage`` refuses page-level language keys by construction; this
module never tries.

Two rules do the real work:

* **The label set is constrained by the line's script** (PRD P0-5).  A
  Devanagari line is ``hin``/``mar``/``kok``/``nep``, never ``eng``, however
  confidently a 176-language model says otherwise.  The constraint is applied
  inside the provider
  (:func:`app.providers.local.constrain_language_candidates`) and **re-checked
  here**, because a provider bug must not be able to poison the derived page
  summary and the MT routing behind it.
* **An unidentifiable line stays ``None``.**  Never the page majority, never the
  first permitted member of the script's set.  A guessed language is the silent
  default BUILD-BRIEF sec. 0.5 forbids, and it is indistinguishable downstream
  from a real identification.

``is_code_mixed`` is computed here as a **page-level fact about scripts**
(``>= CODE_MIX_THRESHOLD`` of lines differ in script from the page majority,
default 15%) and returned on the stage result.  It is not written onto the
page: ``assemble.py`` copies it onto :class:`~app.contracts.envelope.PageResult`
along with the rest of the derived summary.

This module does NOT handle:
  * **page or document language** (``primary_language``,
    ``language_distribution`` on the envelope).  :attr:`LidResult` exposes the
    distribution as an *advisory* summary for callers and the eval harness;
    ``assemble.py`` remains the only writer of the envelope fields, so there is
    exactly one place the derivation can be wrong;
  * translation or transliteration (Stage 6);
  * script detection — the line already carries a script from Stage 2 / Tier-1;
  * loading ``lid.176.ftz``.  The provider owns every model file, and a missing
    one arrives as a typed
    :class:`~app.providers.base.ModelArtifactMissing`, which this stage may be
    told to absorb (``on_missing_artifact="unlabelled"``) so a page still ships
    with unlabelled lines rather than no page at all.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Self

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.ocr import (
    Confidence,
    LangCandidate,
    LangResult,
    LanguageCode,
    LineIndex,
    OCRLine,
    ScriptCode,
    check_distribution,
)
from app.providers.base import ModelArtifactMissing, ModelProvider
from app.providers.local import SCRIPT_LANGUAGES, UNDETERMINED_SCRIPT
from app.settings import Settings, get_settings

__all__ = [
    "DEFAULT_CODE_MIX_THRESHOLD",
    "MIN_LID_CHARACTERS",
    "LidResult",
    "LineLanguage",
    "identify_line_languages",
    "is_code_mixed",
    "languages_for_script",
    "script_distribution",
]

_LOG = logging.getLogger(__name__)

#: Mirrors ``Settings.CODE_MIX_THRESHOLD`` for callers that pass no settings.
DEFAULT_CODE_MIX_THRESHOLD: Final[float] = 0.15

#: fastText scores three characters as confidently as three hundred.  Below this
#: length the line is left unlabelled rather than labelled badly — a two-glyph
#: Devanagari value line cannot distinguish Marathi from Hindi and pretending
#: otherwise degrades the very metric the corpus exists to measure.
MIN_LID_CHARACTERS: Final[int] = 4

#: Below this the identification is recorded but the label is withheld.
MIN_LID_CONFIDENCE: Final[float] = 0.20

_LID_LINES = Counter(
    "stage_s2b_lid_lines_total",
    "Lines passed through line-level LID, by script and outcome.",
    ["script", "outcome"],
)
_LID_LATENCY = Histogram(
    "stage_s2b_lid_seconds",
    "Wall time of one Stage-2b pass over a page's lines.",
)


def languages_for_script(script: str) -> frozenset[str]:
    """ISO 639-3 labels permitted for ``script`` (PRD P0-5).

    An unknown script returns the empty set, which makes every candidate
    inadmissible and leaves the line unlabelled.  That is deliberate: this stage
    would otherwise be the place a new script silently acquires English lines.
    """
    return SCRIPT_LANGUAGES.get(script, frozenset())


class LineLanguage(BaseModel):
    """One line's identification, with the evidence that produced it.

    Separate from :class:`~app.contracts.ocr.OCRLine` because the line contract
    carries a label, not a *justification*: ``confidence`` and ``candidates``
    are Stage-2b working data that the eval harness and the review UI want and
    that the envelope does not carry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_index: LineIndex
    #: ``None`` when nothing survived the script constraint, the line was too
    #: short, or the model was unavailable.  Never a guess.
    language: LanguageCode | None
    confidence: Confidence
    script: ScriptCode
    candidates: tuple[LangCandidate, ...] = ()
    #: Machine-readable why, when ``language`` is ``None``: ``too_short``,
    #: ``no_permitted_candidate``, ``low_confidence``, ``artifact_missing``,
    #: ``empty_text``, ``unknown_script``.
    unlabelled_reason: str | None = None

    @model_validator(mode="after")
    def _reason_matches_label(self) -> Self:
        if self.language is None and not self.unlabelled_reason:
            raise ValueError(
                f"line {self.line_index} has no language and no reason; an "
                "unlabelled line must say why (BUILD-BRIEF sec. 0.5)"
            )
        if self.language is not None and self.unlabelled_reason:
            raise ValueError(
                f"line {self.line_index} carries both a language "
                f"({self.language!r}) and an unlabelled_reason "
                f"({self.unlabelled_reason!r})"
            )
        if self.language is not None:
            permitted = languages_for_script(self.script)
            if permitted and self.language not in permitted:
                raise ValueError(
                    f"line {self.line_index}: language {self.language!r} is "
                    f"not permitted for script {self.script!r} "
                    f"({', '.join(sorted(permitted))}) — PRD P0-5"
                )
        return self


class LidResult(BaseModel):
    """Stage-2b output: relabelled lines plus the page's code-mix facts.

    ``lines`` are **new** :class:`~app.contracts.ocr.OCRLine` objects with
    ``language`` set; the input list is never mutated, because Stage 4 may still
    hold it and a shared-mutation bug in this stage is invisible until the
    envelope is wrong.
    """

    model_config = ConfigDict(extra="forbid")

    lines: tuple[OCRLine, ...]
    per_line: tuple[LineLanguage, ...]
    #: The page's majority script, or ``Zyyy`` when no line carries one.
    majority_script: ScriptCode
    #: Share of lines per script; sums to ~1 when the page has lines.
    script_distribution: dict[str, float] = Field(default_factory=dict)
    #: Advisory: ISO 639-3 -> share of **labelled** lines. ``assemble.py``
    #: remains the only writer of the envelope's copy (PLAN.md invariant 1).
    language_distribution: dict[str, float] = Field(default_factory=dict)
    #: True when >= the code-mix threshold of lines differ in script from the
    #: page majority (PRD P0-5, BUILD-BRIEF M4).
    is_code_mixed: bool
    #: The threshold actually applied, so a report can state it.
    code_mix_threshold: float = Field(ge=0.0, le=1.0)
    #: Lines left without a language, for the escalation router and the harness.
    unlabelled_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _distributions_are_distributions(self) -> Self:
        check_distribution(self.script_distribution, "LidResult.script_distribution")
        check_distribution(
            self.language_distribution, "LidResult.language_distribution"
        )
        if len(self.lines) != len(self.per_line):
            raise ValueError(
                f"{len(self.lines)} lines but {len(self.per_line)} "
                "identifications; they are parallel by contract"
            )
        return self

    def language_of(self, line_index: int) -> str | None:
        """Language of one line by ``line_index`` (not by list position).

        Raises:
            KeyError: when no line carries that index.
        """
        for record in self.per_line:
            if record.line_index == line_index:
                return record.language
        raise KeyError(f"no line with line_index {line_index}")


def script_distribution(lines: Sequence[OCRLine]) -> dict[str, float]:
    """Share of lines per ISO 15924 script.  Empty input gives an empty map."""
    if not lines:
        return {}
    counts: dict[str, int] = {}
    for line in lines:
        counts[line.script] = counts.get(line.script, 0) + 1
    total = len(lines)
    shares = {script: count / total for script, count in counts.items()}
    return _renormalise(shares)


def is_code_mixed(
    lines: Sequence[OCRLine], *, threshold: float = DEFAULT_CODE_MIX_THRESHOLD
) -> tuple[bool, str, dict[str, float]]:
    """Whether the page mixes scripts, its majority script, and the shares.

    BUILD-BRIEF M4: true when ``>= threshold`` of lines differ in script from
    the page majority.  Counted over **lines**, not characters: the signal is
    "this page has two writing systems on it", and a single long Devanagari
    paragraph beside twenty English labels is exactly the case that must fire.

    A page with no lines is not code-mixed and has majority script ``Zyyy``;
    absence of evidence is recorded as such, never as a negative finding about
    a page nobody read.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    distribution = script_distribution(lines)
    if not distribution:
        return False, UNDETERMINED_SCRIPT, {}
    majority = max(distribution, key=lambda key: (distribution[key], key))
    minority_share = 1.0 - distribution[majority]
    return minority_share >= threshold, majority, distribution


def identify_line_languages(
    lines: Sequence[OCRLine],
    *,
    provider: ModelProvider,
    settings: Settings | None = None,
    code_mix_threshold: float | None = None,
    min_characters: int = MIN_LID_CHARACTERS,
    min_confidence: float = MIN_LID_CONFIDENCE,
    on_missing_artifact: Literal["raise", "unlabelled"] = "raise",
) -> LidResult:
    """Identify the language of **each line** and flag page-level code mixing.

    One provider call per line.  fastText is microseconds per short string on
    CPU, so batching would buy nothing and would cost the per-line script
    constraint that is the entire point of PRD P0-5 — a batched call would have
    to pick one label set for lines that legitimately need different ones.

    Args:
        lines: Tier-1 (or Tier-2) lines, each already carrying a ``script``.
        provider: supplies ``identify_language(text, script)``.
        settings: ``CODE_MIX_THRESHOLD`` is read from here when
            ``code_mix_threshold`` is not given.
        on_missing_artifact: ``"raise"`` propagates
            :class:`~app.providers.base.ModelArtifactMissing` (the air-gapped
            default — a missing LID model is a deployment fault, and hiding it
            makes every line silently unlabelled). ``"unlabelled"`` records
            ``artifact_missing`` per line and continues, which is what the
            pipeline wants when LID is not worth failing a document over.

    Raises:
        ModelArtifactMissing: when ``lid.176.ftz`` is absent and
            ``on_missing_artifact="raise"``.
        ValueError: on duplicate ``line_index`` values — indices are the handles
            the extractive summary selects by (PLAN.md invariant 2).
    """
    seen: set[int] = set()
    for line in lines:
        if line.line_index in seen:
            raise ValueError(
                f"duplicate line_index {line.line_index}: indices must be "
                "unique within a page (PLAN.md invariant 2)"
            )
        seen.add(line.line_index)

    threshold = _resolve_threshold(settings, code_mix_threshold)

    with _LID_LATENCY.time():
        records = _identify_all(
            lines,
            provider=provider,
            min_characters=min_characters,
            min_confidence=min_confidence,
            on_missing_artifact=on_missing_artifact,
        )

    relabelled = tuple(
        line.model_copy(update={"language": record.language})
        for line, record in zip(lines, records, strict=True)
    )
    mixed, majority, scripts = is_code_mixed(relabelled, threshold=threshold)
    languages = _language_distribution(records)
    unlabelled = sum(1 for record in records if record.language is None)

    _LOG.debug(
        "stage s2b lines=%d unlabelled=%d majority_script=%s code_mixed=%s",
        len(relabelled),
        unlabelled,
        majority,
        mixed,
    )
    return LidResult(
        lines=relabelled,
        per_line=records,
        majority_script=majority,
        script_distribution=scripts,
        language_distribution=languages,
        is_code_mixed=mixed,
        code_mix_threshold=threshold,
        unlabelled_count=unlabelled,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _resolve_threshold(
    settings: Settings | None, override: float | None
) -> float:
    """Explicit argument, then settings, then the module default."""
    if override is not None:
        if not 0.0 <= override <= 1.0:
            raise ValueError(
                f"code_mix_threshold must be in [0, 1], got {override}"
            )
        return override
    active = settings if settings is not None else get_settings()
    return float(active.CODE_MIX_THRESHOLD)


def _identify_all(
    lines: Sequence[OCRLine],
    *,
    provider: ModelProvider,
    min_characters: int,
    min_confidence: float,
    on_missing_artifact: Literal["raise", "unlabelled"],
) -> tuple[LineLanguage, ...]:
    """Per-line identification, with the artifact policy applied once."""
    records: list[LineLanguage] = []
    artifact_gone: str | None = None
    for line in lines:
        if artifact_gone is not None:
            records.append(
                _unlabelled(line, "artifact_missing", outcome="artifact_missing")
            )
            continue
        try:
            records.append(
                _identify_one(
                    line,
                    provider=provider,
                    min_characters=min_characters,
                    min_confidence=min_confidence,
                )
            )
        except ModelArtifactMissing as exc:
            if on_missing_artifact == "raise":
                raise
            artifact_gone = exc.code
            _LOG.warning(
                "LID artifact unavailable (%s); every line on this page is "
                "left unlabelled rather than guessed",
                exc.artifact,
            )
            records.append(
                _unlabelled(line, "artifact_missing", outcome="artifact_missing")
            )
    return tuple(records)


def _identify_one(
    line: OCRLine,
    *,
    provider: ModelProvider,
    min_characters: int,
    min_confidence: float,
) -> LineLanguage:
    """Identify one line, applying every guard that can withhold a label."""
    flattened = " ".join(line.text.split())
    if not flattened:
        return _unlabelled(line, "empty_text")
    permitted = languages_for_script(line.script)
    if not permitted:
        return _unlabelled(line, "unknown_script")
    if len(flattened) < min_characters:
        return _unlabelled(line, "too_short")

    result: LangResult = provider.identify_language(flattened, line.script)
    candidates = tuple(
        candidate
        for candidate in result.candidates
        if candidate.language in permitted
    )

    if result.language is None:
        return _unlabelled(line, "no_permitted_candidate", candidates=candidates)
    if result.language not in permitted:
        # Defence in depth: the provider already constrains the label set
        # (decision C15). If it ever stops doing so, the page summary and the MT
        # routing behind it must not be the place we find out.
        _LOG.warning(
            "provider returned language %r for script %r, which PRD P0-5 does "
            "not permit; dropping the label",
            result.language,
            line.script,
        )
        return _unlabelled(
            line, "no_permitted_candidate", candidates=candidates
        )
    if result.confidence < min_confidence:
        return _unlabelled(line, "low_confidence", candidates=candidates)

    _LID_LINES.labels(script=line.script, outcome="labelled").inc()
    return LineLanguage(
        line_index=line.line_index,
        language=result.language,
        confidence=result.confidence,
        script=line.script,
        candidates=candidates,
    )


def _unlabelled(
    line: OCRLine,
    reason: str,
    *,
    candidates: tuple[LangCandidate, ...] = (),
    outcome: str = "unlabelled",
) -> LineLanguage:
    """Build the "no label, and here is why" record."""
    _LID_LINES.labels(script=line.script, outcome=outcome).inc()
    return LineLanguage(
        line_index=line.line_index,
        language=None,
        confidence=0.0,
        script=line.script,
        candidates=candidates,
        unlabelled_reason=reason,
    )


def _language_distribution(
    records: Sequence[LineLanguage],
) -> dict[str, float]:
    """Share of **labelled** lines per language.

    Unlabelled lines are excluded rather than bucketed under ``und``: a share
    that mixes "this line is Marathi" with "we could not tell" is not a language
    distribution, and the envelope's argmax rule would then be meaningless.
    """
    counts: dict[str, int] = {}
    for record in records:
        if record.language is not None:
            counts[record.language] = counts.get(record.language, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return _renormalise({code: count / total for code, count in counts.items()})


def _renormalise(distribution: Mapping[str, float]) -> dict[str, float]:
    """Scale to sum to exactly 1.0 within the contract's tolerance."""
    total = sum(distribution.values())
    if total <= 0:
        return {}
    scaled = {key: value / total for key, value in distribution.items()}
    largest = max(scaled, key=lambda key: (scaled[key], key))
    scaled[largest] += 1.0 - sum(scaled.values())
    return scaled
