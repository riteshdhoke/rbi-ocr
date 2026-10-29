"""Open-field mode is controlled by ``Settings.EXTRACT_ALL_FIELDS`` (default true).
Unknown model field names pass through the same OCR grounding gate. The fixed
field ownership and target lists described below apply to legacy mode only.
See ``docs/OPEN-FIELD-EXTRACTION.md`` for the current extraction architecture.

Stage 5b — the constrained SLM pass, and the grounding gate that polices it.

The SLM is allowed to do exactly one thing: **point at text it was shown**.  It
gets five fields (PRD P0-8, :data:`~app.contracts.ocr.SLM_ELIGIBLE_FIELDS`) and
for the fifth — ``complaint_summary`` — it returns line *indices* and no prose
at all (PLAN.md invariant 2, enforced by ``RawField`` at the contract boundary).

Everything an identifier-shaped value could be is already gone before the
request is built: :class:`~app.contracts.ocr.ExtractionContext` refuses
deterministic field names (invariant 7) and ``masked_spans`` — taken straight
from :meth:`~app.stages.s5_validators.ValidatorReport.masked_spans` — tells the
provider which character ranges to overwrite with sentinels, so no account
number, IFSC, amount or date crosses the network even under the ``laptop``
profile.

**The grounding gate** (invariant 8) lives here and is the reason the rest can
be trusted.  A value the model returns is accepted only if it occurs in that
page's OCR text within Levenshtein distance
``Settings.GROUNDING_MAX_EDIT_DISTANCE`` (2), compared script-aware with Indic
digits normalised first.  Two consequences worth stating plainly:

* an ungrounded value is **dropped and flagged**, never trimmed into shape;
* a grounded value's published ``value`` is the **OCR substring**, not the
  model's string.  When the two differ by a character or two, OCR is the
  authority — ``Field.value`` promises "verbatim from OCR" and this is what
  makes that promise structurally true rather than aspirational.  The
  disagreement is recorded as ``LOW_CONFIDENCE`` plus the measured distance.

Concurrency is the provider's (BUILD-BRIEF sec. 3.1): this module calls
``extract_fields_batch`` and never imports ``asyncio``.

This module does NOT handle:
  * building the wire prompt or applying the sentinels.  ``render_lines`` /
    ``build_extraction_prompt`` live in the provider, at the last point before
    serialisation, so there is exactly one masker and the prompt (which *is* the
    response-cache key) is not fragmented by a second renderer.
    :func:`render_numbered_lines` here is the stage-side view used for
    debugging and for the grounding haystack; it never leaves the process;
  * deterministic fields — :mod:`app.stages.s5_validators` owns them and this
    module refuses to ask for them;
  * writing the envelope, or turning a rejection into an escalation.  It reports
    :class:`RejectedField` objects and :mod:`app.stages.assemble` decides what
    the document says about them;
  * translation (:mod:`app.stages.s6_translate`);
  * retries, JSON repair, caching or token accounting *inside* a call — the
    provider owns all four.  This module sizes the request and hands over the
    remaining budget;
  * grounding a value that appears on a *different* page than the one it was
    extracted from.  Cross-page grounding would let a model launder a value from
    page 1 into page 7's fields; the gate is deliberately per-page.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict
from pydantic import Field as _Field
from rapidfuzz.distance import Levenshtein

from app.contracts.envelope import Evidence, Field, FieldFlag
from app.contracts.errors import TokenBudgetExceeded
from app.contracts.ocr import (
    SLM_ELIGIBLE_FIELDS,
    SUMMARY_FIELD_NAME,
    Confidence,
    ExtractionContext,
    LineIndex,
    OCRLine,
    OCRPage,
    PageClassLabel,
    RawField,
    TextSpan,
)
from app.settings import Settings, get_settings
from app.stages.s5_validators import normalise_digits

__all__ = [
    "DEFAULT_TARGET_FIELDS",
    "EXTRACTION_RESPONSE_SCHEMA",
    "FIELD_MAX_TOKENS",
    "PROMPT_OVERHEAD_TOKENS",
    "SLM_VALUE_FIELDS",
    "BudgetSource",
    "GroundingResult",
    "RejectedField",
    "SLMPageResult",
    "SupportsExtraction",
    "apply_grounding_gate",
    "build_extraction_context",
    "build_extraction_contexts",
    "context_max_tokens",
    "extract_pages",
    "ground_value",
    "normalise_for_grounding",
    "render_numbered_lines",
]

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# What the SLM may be asked for
# --------------------------------------------------------------------------- #

#: The eligible fields minus the summary — the ones that carry a text value.
SLM_VALUE_FIELDS: Final[tuple[str, ...]] = tuple(
    sorted(SLM_ELIGIBLE_FIELDS - {SUMMARY_FIELD_NAME})
)

#: Sorted so a context (and therefore the prompt, and therefore the cache key)
#: is deterministic — the provider sorts too, but a caller reading this list
#: should see the same order the model does.
DEFAULT_TARGET_FIELDS: Final[tuple[str, ...]] = tuple(sorted(SLM_ELIGIBLE_FIELDS))

#: Completion-token ceiling per field.  These are generous for an *extractive*
#: task: a name is a handful of tokens and an address rarely more than thirty.
#: They exist so a model that starts writing an essay is cut off by the API
#: rather than by the schema validator after the tokens are paid for.
FIELD_MAX_TOKENS: Final[Mapping[str, int]] = {
    "complainant_name": 48,
    "complainant_address": 128,
    "relief_sought": 192,
    "prior_complaint_reference": 48,
    SUMMARY_FIELD_NAME: 128,
}

#: Tokens reserved for the JSON envelope around the values themselves —
#: ``{"fields": [{"name": ..., "confidence": ..., "line_indices": [...]}]}``.
PROMPT_OVERHEAD_TOKENS: Final[int] = 64

#: JSON Schema for the extraction response, mirroring the shape
#: ``app.providers.hf_api`` asks for and validates.  Exported for endpoints that
#: accept a ``json_schema`` response format, and asserted against the provider's
#: parser in the tests so the two cannot drift.
EXTRACTION_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fields"],
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "confidence"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "value": {
                        "type": ["string", "null"],
                        "description": (
                            "A verbatim substring of the numbered lines. Must be "
                            "null for complaint_summary (PLAN.md invariant 2)."
                        ),
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "line_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "span": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": ["line_index", "start", "end"],
                        "properties": {
                            "line_index": {"type": "integer", "minimum": 0},
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 0},
                        },
                    },
                },
            },
        }
    },
}


class SupportsExtraction(Protocol):
    """The one slice of ``ModelProvider`` this stage needs.

    Narrower than the full protocol on purpose: a test double for this stage
    should not have to implement OCR, and the narrowness documents that nothing
    here touches pixels.
    """

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        ...


class BudgetSource(Protocol):
    """The read side of :class:`~app.providers.base.JobBudget`."""

    @property
    def remaining(self) -> int:
        ...


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def render_numbered_lines(lines: Sequence[OCRLine]) -> str:
    """``[index] text`` per line — the stage-side view of a page.

    Used for the grounding haystack and for debug logging.  **Not** the wire
    prompt: that is ``app.providers.hf_api.render_lines``, which additionally
    replaces validator-claimed spans with sentinels.  Keeping the masker in one
    place is why this function deliberately does not mask — a second, subtly
    different masker is how an identifier eventually escapes.
    """
    return "\n".join(f"[{line.line_index}] {line.text}" for line in lines)


def context_max_tokens(
    target_fields: Sequence[str], *, settings: Settings | None = None
) -> int:
    """Completion ceiling for one page's request.

    Sum of the per-field ceilings plus JSON overhead, clamped to
    ``SLM_MAX_TOKENS``.  Per-field rather than one global number because asking
    for 512 tokens to extract a name is how a Phase-0 job budget evaporates in
    six pages.
    """
    active = settings if settings is not None else get_settings()
    wanted = sum(FIELD_MAX_TOKENS.get(name, 64) for name in target_fields)
    return max(1, min(active.SLM_MAX_TOKENS, wanted + PROMPT_OVERHEAD_TOKENS))


def build_extraction_context(
    page: OCRPage,
    *,
    document_id: str,
    page_class: PageClassLabel,
    primary_script: str,
    primary_language: str | None = None,
    target_fields: Sequence[str] | None = None,
    masked_spans: Sequence[TextSpan] = (),
    budget: BudgetSource | None = None,
    settings: Settings | None = None,
) -> ExtractionContext:
    """Build one page's :class:`~app.contracts.ocr.ExtractionContext`.

    ``masked_spans`` should be
    :meth:`~app.stages.s5_validators.ValidatorReport.masked_spans` for this page.
    Passing them is not optional in spirit: they are the mechanism by which
    invariant 7's "identifiers never reach the SLM" is true of the *bytes*, not
    only of the field list.

    Raises:
        TokenBudgetExceeded: when ``budget`` has no headroom left. Refusing here
            rather than sending a request that will be rejected keeps the cost
            of a breach at zero.
        ValueError: for an empty ``target_fields`` — an extraction request with
            nothing to extract is a caller bug.
    """
    active = settings if settings is not None else get_settings()
    fields = tuple(target_fields) if target_fields is not None else DEFAULT_TARGET_FIELDS
    if not fields:
        raise ValueError("target_fields must name at least one field to extract")
    max_tokens = (active.SLM_MAX_TOKENS if active.EXTRACT_ALL_FIELDS
                  else context_max_tokens(fields, settings=active))
    remaining: int | None = None
    if budget is not None:
        remaining = int(budget.remaining)
        if remaining <= 0:
            raise TokenBudgetExceeded(
                "no token headroom left for SLM extraction on page "
                f"{page.page_no}",
                budget=None,
                used=None,
                context={"page_no": page.page_no, "document_id": document_id},
            )
        max_tokens = max(1, min(max_tokens, remaining))
    return ExtractionContext(
        document_id=document_id,
        page_no=page.page_no,
        lines=list(page.lines),
        page_class=page_class,
        primary_script=primary_script,
        primary_language=primary_language,
        discover_fields=active.EXTRACT_ALL_FIELDS,
        target_fields=list(fields),
        masked_spans=list(masked_spans),
        max_tokens=max_tokens,
        token_budget_remaining=remaining,
    )


def build_extraction_contexts(
    pages: Sequence[OCRPage],
    *,
    document_id: str,
    page_classes: Mapping[int, PageClassLabel] | PageClassLabel,
    primary_scripts: Mapping[int, str] | str,
    primary_languages: Mapping[int, str | None] | str | None = None,
    target_fields: Sequence[str] | None = None,
    masked_spans: Mapping[int, Sequence[TextSpan]] | None = None,
    budget: BudgetSource | None = None,
    settings: Settings | None = None,
) -> list[ExtractionContext]:
    """Contexts for every page that has recognised text.

    Pages with no lines are skipped: there is nothing to point at, and paying
    for a request that can only return an empty object is the kind of cost the
    per-job budget exists to stop.
    """
    contexts: list[ExtractionContext] = []
    for page in pages:
        if not page.lines:
            continue
        contexts.append(
            build_extraction_context(
                page,
                document_id=document_id,
                page_class=_per_page(page_classes, page.page_no),
                primary_script=_per_page(primary_scripts, page.page_no),
                primary_language=_per_page(primary_languages, page.page_no),
                target_fields=target_fields,
                masked_spans=(masked_spans or {}).get(page.page_no, ()),
                budget=budget,
                settings=settings,
            )
        )
    return contexts


def _per_page(value: Any, page_no: int) -> Any:
    """Read a per-page override out of a mapping, or use a scalar for all pages."""
    if isinstance(value, Mapping):
        return value.get(page_no)
    return value


def extract_pages(
    provider: SupportsExtraction, ctxs: Sequence[ExtractionContext]
) -> list[dict[str, RawField]]:
    """Run the SLM over every context — the provider owns the concurrency.

    Always ``extract_fields_batch``, even for one page, so there is a single
    call site and the provider (not this stage) decides whether concurrency
    means ``asyncio.gather`` or a sequential loop (BUILD-BRIEF sec. 3.1).

    Errors propagate: a token-budget breach or a dead endpoint is a job-level
    fact, and a half-filled result would let the assembler publish an envelope
    that looks complete.
    """
    if not ctxs:
        return []
    results = provider.extract_fields_batch(list(ctxs))
    if len(results) != len(ctxs):
        raise ValueError(
            f"provider returned {len(results)} extraction results for "
            f"{len(ctxs)} pages; positions must be preserved"
        )
    return results


# --------------------------------------------------------------------------- #
# The grounding gate (PLAN.md invariant 8)
# --------------------------------------------------------------------------- #

def _fold_char(character: str) -> str:
    """Lowercase one character, unless doing so would change its length.

    ``"ß".lower()`` is ``"ss"``; folding it would move every offset after it and
    silently invalidate the evidence span.  Such characters are left alone — the
    mapping stays strictly 1:1.
    """
    lowered = character.lower()
    return lowered if len(lowered) == 1 else character


def normalise_for_grounding(text: str) -> str:
    """Length-preserving normalisation used by the gate.

    Two transformations, both strictly one code point in / one code point out:

    * Indic decimal digits to ASCII (invariant 8 says "Indic digits normalized
      before comparison" — ``४५०००`` and ``45000`` are the same number and a
      model that echoes one for the other is grounded, not hallucinating);
    * case folding, skipped for the handful of characters whose lowercase form
      is longer than one code point.

    Deliberately does *not* apply NFC/NFKC: those change string length, which
    would invalidate every character offset the evidence depends on.
    :func:`ground_value` retries under full Unicode normalisation as a second
    pass and reports whole-line evidence when that is what succeeded.
    """
    return "".join(_fold_char(character) for character in normalise_digits(text))


def _unicode_normalise(text: str) -> str:
    """NFKC form, for the offset-free second pass only."""
    return unicodedata.normalize("NFKC", normalise_for_grounding(text))


class GroundingResult(BaseModel):
    """Where (and how well) a model-proposed value sits in the OCR text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounded: bool
    #: Levenshtein distance to the closest window, or ``None`` when nothing was
    #: within the budget (the exact distance of a miss is not interesting and
    #: computing it exactly would cost a full scan).
    distance: int | None = None
    line_index: LineIndex | None = None
    start: int | None = None
    end: int | None = None
    #: The OCR substring that matched — the string that becomes ``Field.value``.
    matched_text: str | None = None
    #: True when only the Unicode-normalised second pass succeeded, so the span
    #: covers the whole line rather than the exact characters.
    approximate_span: bool = False


def _best_window(
    needle: str, haystack: str, max_distance: int
) -> tuple[int, int, int] | None:
    """Closest window of ``haystack`` to ``needle`` within ``max_distance``.

    Returns ``(start, end, distance)`` or ``None``.  Windows of every length in
    ``[len(needle) - d, len(needle) + d]`` are tried because an edit budget of
    ``d`` admits values that are up to ``d`` characters shorter or longer than
    what OCR produced.  Exact-match short-circuits.
    """
    needle_length = len(needle)
    if needle_length == 0 or not haystack:
        return None
    exact = haystack.find(needle)
    if exact >= 0:
        return exact, exact + needle_length, 0
    best: tuple[int, int, int] | None = None
    lowest_width = max(1, needle_length - max_distance)
    highest_width = min(len(haystack), needle_length + max_distance)
    for width in range(lowest_width, highest_width + 1):
        for start in range(len(haystack) - width + 1):
            distance = Levenshtein.distance(
                needle, haystack[start : start + width], score_cutoff=max_distance
            )
            if distance > max_distance:
                continue
            if best is None or distance < best[2]:
                best = (start, start + width, distance)
                if distance == 0:
                    return best
    return best


def ground_value(
    value: str,
    lines: Sequence[OCRLine],
    *,
    max_distance: int = 2,
    preferred_line_indices: Sequence[int] = (),
) -> GroundingResult:
    """Locate ``value`` in a page's OCR text within ``max_distance`` edits.

    Search order, and why:

    1. **The lines the model cited**, if any.  A model that says "this came from
       line 7" and is right about line 7 is doing what it was asked; checking
       there first makes the common case cheap and keeps the evidence honest.
    2. **Every other line on the page.**  A right answer with a wrong citation
       is still a right answer; the gate is about the value existing, not about
       the model's bookkeeping.
    3. **The cited lines joined**, for a value that legitimately spans a line
       break (an address usually does).
    4. **A Unicode-normalised (NFKC) retry**, which catches the Devanagari
       nukta and ligature forms that normalise to different code-point counts.
       Because that pass changes string lengths it cannot report exact offsets,
       so the result carries ``approximate_span=True`` and the whole line.

    An empty or whitespace-only ``value`` is never grounded: it is the model
    declining to answer, and it must not become a field.
    """
    needle = normalise_for_grounding(value).strip()
    if not needle:
        return GroundingResult(grounded=False)

    by_index = {line.line_index: line for line in lines}
    cited_indices = [
        index for index in preferred_line_indices if index in by_index
    ]
    cited = [by_index[index] for index in cited_indices]
    seen = set(cited_indices)
    remaining = [line for line in lines if line.line_index not in seen]

    for line in [*cited, *remaining]:
        haystack = normalise_for_grounding(line.text)
        window = _best_window(needle, haystack, max_distance)
        if window is None:
            continue
        start, end, distance = window
        return GroundingResult(
            grounded=True,
            distance=distance,
            line_index=line.line_index,
            start=start,
            end=end,
            matched_text=line.text[start:end],
        )

    if len(cited) > 1:
        joined = " ".join(line.text for line in cited)
        if _best_window(needle, normalise_for_grounding(joined), max_distance):
            first = cited[0]
            return GroundingResult(
                grounded=True,
                distance=None,
                line_index=first.line_index,
                start=0,
                end=len(first.text),
                matched_text=first.text,
                approximate_span=True,
            )

    normalised_needle = _unicode_normalise(value).strip()
    if normalised_needle and normalised_needle != needle:
        for line in lines:
            if _best_window(
                normalised_needle, _unicode_normalise(line.text), max_distance
            ):
                return GroundingResult(
                    grounded=True,
                    distance=None,
                    line_index=line.line_index,
                    start=0,
                    end=len(line.text),
                    matched_text=line.text,
                    approximate_span=True,
                )
    return GroundingResult(grounded=False)


class RejectedField(BaseModel):
    """One model value the gate refused, and why.

    Kept as data rather than an exception: a hallucinated ``relief_sought`` must
    not abort a document whose validators found everything else.  The assembler
    turns these into warnings and an escalation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_no: int = _Field(ge=1)
    name: str = _Field(min_length=1)
    #: The model's string, kept for the review queue and the eval harness.  It
    #: is never published as a ``Field.value``.
    value: str | None
    #: ``ungrounded`` | ``empty`` | ``unknown_line_index`` | ``not_eligible``.
    reason: str
    line_indices: tuple[int, ...] = ()


class SLMPageResult(BaseModel):
    """Everything the gate let through for one page, plus what it stopped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page_no: int = _Field(ge=1)
    #: Grounded fields, ``source="slm"``, ready for the envelope.
    fields: dict[str, Field] = _Field(default_factory=dict)
    #: Line indices selected for the extractive summary, in the model's reading
    #: order, already checked to exist on this page.  The **assembler**
    #: concatenates their text (invariant 2); this module never does.
    summary_line_indices: tuple[LineIndex, ...] = ()
    rejected: tuple[RejectedField, ...] = ()

    @property
    def needs_review(self) -> bool:
        """True when anything was rejected or any accepted field is flagged."""
        return bool(self.rejected) or any(
            field.needs_review for field in self.fields.values()
        )


def apply_grounding_gate(
    raw_fields: Mapping[str, RawField],
    page: OCRPage,
    *,
    max_distance: int | None = None,
    settings: Settings | None = None,
) -> SLMPageResult:
    """Promote model output to envelope fields, refusing anything ungrounded.

    This is the whole of invariant 8 in one function.  For each proposed field:

    * a name outside :data:`~app.contracts.ocr.SLM_ELIGIBLE_FIELDS` is rejected
      (``not_eligible``) — the provider already drops unknown names, so reaching
      here means a different provider was wired in;
    * ``complaint_summary`` contributes **line indices only**, each checked to
      exist on this page.  No text is read, produced or compared: the assembler
      owns the concatenation;
    * any other field must ground.  The published ``value`` is the OCR
      substring, not the model's string; a non-zero edit distance additionally
      flags ``LOW_CONFIDENCE``.

    ``Field.validator_passed`` stays ``None`` throughout: no deterministic
    validator applies to a name or an address, and reporting ``True`` would
    claim a check that never ran.
    """
    active = settings if settings is not None else get_settings()
    budget = (
        max_distance
        if max_distance is not None
        else active.GROUNDING_MAX_EDIT_DISTANCE
    )
    known = {line.line_index for line in page.lines}
    by_index = {line.line_index: line for line in page.lines}

    accepted: dict[str, Field] = {}
    rejected: list[RejectedField] = []
    summary_indices: tuple[int, ...] = ()

    for name in sorted(raw_fields):
        raw = raw_fields[name]
        if not active.EXTRACT_ALL_FIELDS and name not in SLM_ELIGIBLE_FIELDS:
            rejected.append(
                RejectedField(
                    page_no=page.page_no,
                    name=name,
                    value=raw.value,
                    reason="not_eligible",
                    line_indices=tuple(raw.line_indices),
                )
            )
            continue
        if name == SUMMARY_FIELD_NAME:
            unknown = [i for i in raw.line_indices if i not in known]
            if unknown:
                rejected.append(
                    RejectedField(
                        page_no=page.page_no,
                        name=name,
                        value=None,
                        reason="unknown_line_index",
                        line_indices=tuple(unknown),
                    )
                )
                continue
            summary_indices = tuple(raw.line_indices)
            continue
        if raw.value is None or not raw.value.strip():
            rejected.append(
                RejectedField(
                    page_no=page.page_no,
                    name=name,
                    value=raw.value,
                    reason="empty",
                    line_indices=tuple(raw.line_indices),
                )
            )
            continue
        result = ground_value(
            raw.value,
            page.lines,
            max_distance=budget,
            preferred_line_indices=raw.line_indices,
        )
        if not result.grounded or result.line_index is None:
            _LOG.warning(
                "grounding gate rejected field %s on page %s (edit distance > %d)",
                name,
                page.page_no,
                budget,
            )
            rejected.append(
                RejectedField(
                    page_no=page.page_no,
                    name=name,
                    value=raw.value,
                    reason="ungrounded",
                    line_indices=tuple(raw.line_indices),
                )
            )
            continue
        line = by_index[result.line_index]
        flags: list[FieldFlag] = []
        if result.distance is None or result.distance > 0:
            flags.append(FieldFlag.LOW_CONFIDENCE)
        if line.is_handwritten:
            flags.append(FieldFlag.HANDWRITTEN_SOURCE)
        accepted[name] = Field(
            value=result.matched_text,
            confidence=_confidence_of(raw, result, budget),
            source="slm",
            validator_passed=None,
            evidence=Evidence(
                page_no=page.page_no,
                bbox=[float(value) for value in line.bbox],
                span=TextSpan(
                    line_index=result.line_index,
                    start=result.start or 0,
                    end=result.end if result.end is not None else len(line.text),
                ),
            ),
            flags=tuple(flags),
        )
    return SLMPageResult(
        page_no=page.page_no,
        fields=accepted,
        summary_line_indices=summary_indices,
        rejected=tuple(rejected),
    )


def _confidence_of(
    raw: RawField, result: GroundingResult, budget: int
) -> Confidence:
    """Discount the model's self-reported confidence by the grounding distance.

    A model's own confidence is not calibrated and is not evidence.  What *is*
    evidence is how far its answer sat from the pixels: an exact match keeps the
    reported number, and every edit of slack costs a proportional share of it.
    """
    if result.distance is None:
        penalty = 0.5
    else:
        penalty = 1.0 - (result.distance / (budget + 1)) if budget >= 0 else 1.0
    return max(0.0, min(1.0, float(raw.confidence) * penalty))
