"""Envelope construction — where every derived value is derived, and only here.

This module is the single writer of the four things the rest of the pipeline is
forbidden to compute:

**Page and document language** (PLAN.md invariant 1).  ``script``, ``language``
and ``is_handwritten`` live on :class:`~app.contracts.ocr.OCRLine`;
:class:`~app.contracts.ocr.OCRPage` actively rejects them.  The page-level
``primary_language`` / ``language_distribution`` / ``is_code_mixed`` that a
consumer sees are *summaries* computed in :func:`derive_language_summary` from
the line labels Stage 2b assigned, and they exist nowhere else in the codebase.

**``complaint_summary``** (invariant 2).  The SLM returns line indices; this
module looks those lines up and concatenates their **verbatim OCR text**.  The
model's prose, if it produced any, was already refused at the ``RawField``
boundary; what remains is a selection, and
:class:`~app.contracts.envelope.ComplaintSummary` re-derives the concatenation
and rejects any mismatch.  Nothing here can write a sentence.

**The grounding gate** (invariant 8).  Raw model output arrives on
:attr:`PageAssembly.raw_fields` and passes through
:func:`~app.stages.s5_slm.apply_grounding_gate` on its way in.  This is the
enforcement point: there is no other route from a provider's answer into a
:class:`~app.contracts.envelope.Field`.  A rejected value becomes a warning and
an escalation, never a field.

**Provenance.**  Every field arrives already carrying its ``source`` and its
mandatory ``Evidence`` — from a validator hit or from the gate's grounding
match.  This module never invents either, and never promotes a value it cannot
point at.

Everything else it does is bookkeeping: ``bounce_decision`` passes through
untouched, ``result_version``/``is_final`` come from the caller (the two-stage
delivery in BUILD-BRIEF sec. 5 increments them), and
:class:`~app.contracts.envelope.ProcessingStats` counters are recomputed from
the page records so they cannot disagree with them.

This module does NOT handle:
  * deciding *whether* to bounce, escalate or repair — Stage 1.5 and the router
    own those judgements; this records them;
  * running any model, validator or translator.  A ``translator`` callable may
    be passed in (normally ``functools.partial`` over
    :func:`app.stages.s6_translate.translate_fields`) and is applied to the
    finished field map;
  * OCR, quality scoring, or page classification;
  * webhook delivery, persistence, or the audit log (WS-F);
  * merging two envelopes across ``result_version`` increments.  Version 2 is
    produced by re-assembling with the escalated pages' results, not by patching
    version 1;
  * PII redaction.  The returned object contains complainant data by
    construction and must never be logged at INFO.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict
from pydantic import Field as _Field

from app.contracts.envelope import (
    BounceDecision,
    ComplaintSummary,
    DocumentResult,
    Evidence,
    Field,
    PageResult,
    ProcessingStats,
    QualityFlag,
    QualityVerdict,
    ResultEnvelope,
    SummarySpan,
)
from app.contracts.ocr import (
    Confidence,
    LineIndex,
    OCRLine,
    OCRPage,
    PageClass,
    PageClassLabel,
    RawField,
    TextSpan,
)
from app.settings import Settings, get_settings
from app.stages.s5_discover import add_discovered, discover_fields
from app.stages.s5_slm import RejectedField, SLMPageResult, apply_grounding_gate
from app.stages.s5_validators import ValidatorReport

__all__ = [
    "GROUNDING_WARNING_PREFIX",
    "LanguageSummary",
    "PageAssembly",
    "assemble_envelope",
    "build_complaint_summary",
    "build_page_result",
    "derive_language_summary",
    "merge_field_candidates",
    "utc_now",
]

_LOG = logging.getLogger(__name__)

#: Warning prefix for a value the grounding gate refused.  Machine-readable so
#: the review queue and the eval harness can count rejections without parsing
#: prose (PRD P0-8 counts schema validity; this counts the gate).
GROUNDING_WARNING_PREFIX: Final[str] = "grounding_rejected"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


class PageAssembly(BaseModel):
    """Everything known about one page, from every stage that touched it.

    Deliberately one object rather than eight parallel sequences: a page whose
    quality flags come from one list and whose OCR comes from another, indexed
    by position, is one ``enumerate`` away from attributing a bounce to the
    wrong page.
    """

    model_config = ConfigDict(extra="forbid")

    page: OCRPage
    quality_score: Confidence = 1.0
    quality_flags: tuple[QualityFlag, ...] = ()
    quality_verdict: QualityVerdict = QualityVerdict.ACCEPT
    #: Stage 3 output for this page; ``None`` when the classifier did not run.
    page_class: PageClass | None = None
    escalated: bool = False
    #: Ran through Tier 2 as part of the 3% shadow sample, *not* an escalation
    #: (PLAN.md invariant 3).
    shadow_sampled: bool = False
    repairs_applied: tuple[str, ...] = ()
    #: Raw SLM output for this page, **before** the grounding gate.  Passing
    #: pre-gated fields is not possible on purpose: the gate runs here.
    raw_fields: dict[str, RawField] = _Field(default_factory=dict)

    @property
    def page_no(self) -> int:
        return self.page.page_no


# --------------------------------------------------------------------------- #
# Derived language summaries — invariant 1 lives and dies here
# --------------------------------------------------------------------------- #


class LanguageSummary(BaseModel):
    """The derived language/script view of a set of OCR lines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_script: str | None = None
    primary_language: str | None = None
    language_distribution: dict[str, float] = _Field(default_factory=dict)
    script_distribution: dict[str, float] = _Field(default_factory=dict)
    is_code_mixed: bool = False
    #: Lines that carry no language label yet.  Reported rather than guessed:
    #: an unlabelled line is Stage 2b declining to choose, and folding it into
    #: the majority would manufacture agreement.
    unlabelled_lines: int = _Field(default=0, ge=0)


def _renormalise(counts: Mapping[str, int]) -> dict[str, float]:
    """Turn counts into a distribution that sums to exactly 1.0.

    The contract checks the sum to 1e-6; float division of, say, thirds does not
    reach that on its own, so the residual is absorbed into the largest bucket.
    """
    total = sum(counts.values())
    if total <= 0:
        return {}
    shares = {key: value / total for key, value in counts.items()}
    largest = max(shares, key=lambda key: (shares[key], key))
    shares[largest] += 1.0 - math.fsum(shares.values())
    return shares


def _argmax(distribution: Mapping[str, float]) -> str | None:
    """Largest share, ties broken lexicographically so runs are reproducible."""
    if not distribution:
        return None
    best = max(distribution.values())
    return min(
        key
        for key, value in distribution.items()
        if math.isclose(value, best, rel_tol=1e-9)
    )


def derive_language_summary(
    lines: Sequence[OCRLine], *, code_mix_threshold: float
) -> LanguageSummary:
    """Summarise a set of lines' scripts and languages.

    This is the *only* implementation of the page/document language rollup in
    the service (invariant 1).  Two deliberate choices:

    * lines with ``language is None`` are counted in ``unlabelled_lines`` and
      excluded from ``language_distribution``.  Stage 2b returns ``None`` when
      nothing survived the script constraint; treating that as agreement with
      the majority would invent a label the LID refused to give;
    * ``is_code_mixed`` keys on **script**, not language, and compares against
      ``code_mix_threshold`` (``CODE_MIX_THRESHOLD``, 0.15 by default) — PRD
      P0-5's definition, and the one the bilingual-form case needs: an English
      label line beside a handwritten Marathi value is code-mixed whether or not
      the LID managed to label either.
    """
    script_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    unlabelled = 0
    for line in lines:
        script_counts[line.script] = script_counts.get(line.script, 0) + 1
        if line.language is None:
            unlabelled += 1
            continue
        language_counts[line.language] = language_counts.get(line.language, 0) + 1

    script_distribution = _renormalise(script_counts)
    language_distribution = _renormalise(language_counts)
    primary_script = _argmax(script_distribution)
    primary_language = _argmax(language_distribution)
    minority_share = (
        1.0 - script_distribution[primary_script] if primary_script is not None else 0.0
    )
    return LanguageSummary(
        primary_script=primary_script,
        primary_language=primary_language,
        language_distribution=language_distribution,
        script_distribution=script_distribution,
        is_code_mixed=bool(lines) and minority_share >= code_mix_threshold,
        unlabelled_lines=unlabelled,
    )


# --------------------------------------------------------------------------- #
# Page records
# --------------------------------------------------------------------------- #


def build_page_result(
    assembly: PageAssembly, *, code_mix_threshold: float
) -> PageResult:
    """One :class:`~app.contracts.envelope.PageResult`, summaries included.

    Raises:
        ValueError: through the contract, when the recorded verdict contradicts
            the recognised lines (a bounced page that somehow has text). That is
            a pipeline bug and must not be smoothed over here.
    """
    summary = derive_language_summary(
        assembly.page.lines, code_mix_threshold=code_mix_threshold
    )
    return PageResult(
        page_no=assembly.page_no,
        quality_score=assembly.quality_score,
        quality_flags=assembly.quality_flags,
        quality_verdict=assembly.quality_verdict,
        ocr_tier=assembly.page.tier,
        line_count=len(assembly.page.lines),
        primary_script=summary.primary_script,
        primary_language=summary.primary_language,
        language_distribution=summary.language_distribution,
        is_code_mixed=summary.is_code_mixed,
        escalated=assembly.escalated,
        shadow_sampled=assembly.shadow_sampled,
        repairs_applied=assembly.repairs_applied,
    )


# --------------------------------------------------------------------------- #
# The extractive summary — invariant 2
# --------------------------------------------------------------------------- #


def build_complaint_summary(
    selections: Sequence[tuple[int, Sequence[LineIndex]]],
    pages: Mapping[int, OCRPage],
    *,
    separator: str = "\n",
) -> ComplaintSummary | None:
    """Concatenate the selected lines' **verbatim** OCR text.

    ``selections`` is ``[(page_no, line_indices), ...]``.  The assembler — not
    the model — reads the text; the model only ever chose indices.  Order is
    ascending page number, then the model's order within a page, because the
    model's ordering *is* its reading-order judgement and re-sorting it would
    scramble a narrative that runs down the page.

    Returns ``None`` when nothing was selected or every selected line is empty:
    an empty summary is absence, and the envelope models absence as ``None``
    rather than an empty string.

    Raises:
        KeyError: when a selection names a page or line that does not exist.
            The gate already checked line indices against their own page, so
            reaching this means the caller mismatched pages, which is a bug.
    """
    spans: list[SummarySpan] = []
    for page_no, indices in sorted(selections, key=lambda item: item[0]):
        page = pages[page_no]
        for line_index in indices:
            line = page.line_by_index(line_index)
            if not line.text:
                continue
            spans.append(
                SummarySpan(
                    evidence=Evidence(
                        page_no=page_no,
                        bbox=[float(value) for value in line.bbox],
                        span=TextSpan(
                            line_index=line_index, start=0, end=len(line.text)
                        ),
                    ),
                    text=line.text,
                )
            )
    if not spans:
        return None
    return ComplaintSummary(
        text=separator.join(span.text for span in spans),
        spans=tuple(spans),
        separator=separator,
    )


# --------------------------------------------------------------------------- #
# Field merging
# --------------------------------------------------------------------------- #


def merge_field_candidates(
    *candidate_maps: Mapping[str, Field],
) -> dict[str, Field]:
    """Merge per-source field maps, highest confidence winning.

    Ties break on earliest evidence (page, line, span start) so a document that
    repeats a value on pages 1 and 4 always cites page 1.  Later maps do not
    silently overwrite earlier ones: this is a comparison, not an update, so the
    caller's argument order carries no hidden authority.
    """
    best: dict[str, Field] = {}
    for candidates in candidate_maps:
        for name, field in candidates.items():
            incumbent = best.get(name)
            if incumbent is None or _field_rank(field) < _field_rank(incumbent):
                best[name] = field
    return best


def _field_rank(field: Field) -> tuple[float, int, int, int]:
    return (
        -round(field.confidence, 6),
        field.evidence.page_no,
        field.evidence.span.line_index,
        field.evidence.span.start,
    )


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #


def assemble_envelope(
    *,
    document_id: str,
    provider_profile: str,
    pages: Sequence[PageAssembly],
    validator_report: ValidatorReport | None = None,
    document_class: PageClass | None = None,
    bounce_decision: BounceDecision | None = None,
    translator: Callable[[dict[str, Field]], Mapping[str, Field]] | None = None,
    job_id: str | None = None,
    result_version: int = 1,
    is_final: bool = True,
    processing: ProcessingStats | None = None,
    warnings: Sequence[str] = (),
    supporting_doc_tags: Sequence[str] = (),
    summary_separator: str = "\n",
    settings: Settings | None = None,
    requested_fields: Sequence[str] = (),
) -> ResultEnvelope:
    """Build the published :class:`~app.contracts.envelope.ResultEnvelope`.

    The order of operations is the contract of this function:

    1. per-page records, with language summaries derived from line labels;
    2. **the grounding gate**, page by page, over ``PageAssembly.raw_fields``;
    3. deterministic validator fields merged with the grounded SLM fields;
    4. the optional ``translator`` applied to the merged map (``value``
       untouched, ``value_en`` advisory);
    5. ``complaint_summary`` concatenated from the surviving line selections;
    6. document-level language summary over every page's lines;
    7. counters recomputed from the page records.

    Args:
        bounce_decision: passed straight through.  When it is a bounce the
            envelope carries no ``document`` at all — a refused document must
            say why, not arrive empty (BUILD-BRIEF sec. 0.5).
        translator: applied to the finished field map; normally
            ``functools.partial(translate_fields, provider, src_language=...,
            mt_engine=...)``.  Kept as a callable so this module imports no
            provider and needs no MT engine to be tested.
        processing: caller-supplied timings.  Page counters are **always**
            recomputed here so they cannot contradict the page records.

    Raises:
        ValueError: when ``pages`` contains duplicate page numbers, or when a
            page record contradicts itself (raised by the contract).
    """
    active = settings if settings is not None else get_settings()
    threshold = active.CODE_MIX_THRESHOLD

    ordered = sorted(pages, key=lambda item: item.page_no)
    numbers = [item.page_no for item in ordered]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"duplicate page numbers in assembly input: {numbers}")

    page_results = tuple(
        build_page_result(item, code_mix_threshold=threshold) for item in ordered
    )

    gate_results: list[SLMPageResult] = [
        apply_grounding_gate(item.raw_fields, item.page, settings=active)
        for item in ordered
    ]
    rejections: tuple[RejectedField, ...] = tuple(
        rejected for result in gate_results for rejected in result.rejected
    )

    decision = bounce_decision or BounceDecision.accepted()
    all_warnings = [*warnings, *(_warning_for(item) for item in rejections)]
    if rejections:
        # Counts only: the rejected strings are unverified model output and this
        # line is INFO-level (PLAN.md: never log raw OCR text or model values).
        _LOG.info(
            "grounding gate rejected %d value(s) across %d page(s)",
            len(rejections),
            len({item.page_no for item in rejections}),
        )

    document: DocumentResult | None = None
    fields: dict[str, Field] = {}
    if decision.decision is not QualityVerdict.BOUNCE:
        validator_fields = (
            validator_report.fields() if validator_report is not None else {}
        )
        slm_fields = merge_field_candidates(*(result.fields for result in gate_results))
        fields = merge_field_candidates(validator_fields, slm_fields)
        if active.EXTRACT_ALL_FIELDS:
            # Known validators remain preferred for their canonical keys; keep
            # other occurrences instead of collapsing them to one winner.
            fields = dict(validator_fields)
            if validator_report is not None:
                for hit in validator_report.hits:
                    add_discovered(fields, hit.field_name, hit.to_field())
            for result in gate_results:
                for name, candidate in result.fields.items():
                    add_discovered(fields, name, candidate)
            fields = discover_fields(
                [
                    item.page
                    for item in ordered
                    if item.quality_verdict is not QualityVerdict.BOUNCE
                ],
                fields,
                requested_fields=requested_fields,
            )
        if translator is not None:
            fields = dict(translator(fields))

        pages_by_no = {item.page_no: item.page for item in ordered}
        summary = build_complaint_summary(
            [
                (result.page_no, result.summary_line_indices)
                for result in gate_results
                if result.summary_line_indices
            ],
            pages_by_no,
            separator=summary_separator,
        )
        document_summary = derive_language_summary(
            [line for item in ordered for line in item.page.lines],
            code_mix_threshold=threshold,
        )
        label, label_confidence = _document_class(document_class, ordered)
        document = DocumentResult(
            document_class=label,
            document_class_confidence=label_confidence,
            fields=fields,
            complaint_summary=summary,
            supporting_doc_tags=tuple(supporting_doc_tags),
            page_count=len(page_results),
            primary_script=document_summary.primary_script,
            primary_language=document_summary.primary_language,
            language_distribution=document_summary.language_distribution,
            is_code_mixed=document_summary.is_code_mixed,
        )

    escalation_flag = (
        any(result.escalated for result in page_results)
        or any(field.needs_review for field in fields.values())
        or bool(rejections)
    )

    stats = _processing_stats(processing, page_results)

    return ResultEnvelope(
        result_version=result_version,
        is_final=is_final,
        document_id=document_id,
        job_id=job_id,
        provider_profile=provider_profile,
        requested_fields=tuple(requested_fields),
        bounce_decision=decision,
        escalation_flag=escalation_flag,
        pages=page_results,
        document=document,
        processing=stats,
        warnings=tuple(all_warnings),
    )


def _warning_for(rejected: RejectedField) -> str:
    """Machine-readable warning for one gate rejection.

    The model's string is **not** included: it is unverified text that failed to
    appear in the document, and the warning list is the one part of the envelope
    a log aggregator is likely to print.
    """
    return (
        f"{GROUNDING_WARNING_PREFIX}:page={rejected.page_no}"
        f":field={rejected.name}:reason={rejected.reason}"
    )


def _document_class(
    given: PageClass | None, pages: Sequence[PageAssembly]
) -> tuple[PageClassLabel, float]:
    """Document class: the caller's, else a confidence-weighted page vote.

    Returns ``(UNKNOWN, 0.0)`` when no page was classified.  ``unknown`` with
    zero confidence is the honest answer; picking the first page's label and
    reporting its confidence as the document's would be a fabrication.
    """
    if given is not None:
        return given.label, given.confidence
    weights: dict[PageClassLabel, float] = {}
    for item in pages:
        if item.page_class is None:
            continue
        label = item.page_class.label
        weights[label] = weights.get(label, 0.0) + float(item.page_class.confidence)
    if not weights:
        return PageClassLabel.UNKNOWN, 0.0
    total = math.fsum(weights.values())
    winner = min(
        (label for label in weights if weights[label] == max(weights.values())),
        key=lambda label: label.value,
    )
    return winner, max(0.0, min(1.0, weights[winner] / total))


def _processing_stats(
    given: ProcessingStats | None, pages: Sequence[PageResult]
) -> ProcessingStats:
    """Counters recomputed from the page records; timings from the caller.

    A caller may report how long the job took; it may not report how many pages
    bounced, because the page records already say and two answers to one
    question is one answer too many.
    """
    base = given.model_copy() if given is not None else ProcessingStats()
    base.pages_processed = len(pages)
    base.pages_bounced = sum(
        1 for page in pages if page.quality_verdict is QualityVerdict.BOUNCE
    )
    base.pages_escalated = sum(1 for page in pages if page.escalated)
    if (
        base.duration_ms is None
        and base.started_at is not None
        and base.finished_at is not None
    ):
        base.duration_ms = max(
            0.0,
            (base.finished_at - base.started_at).total_seconds() * 1000.0,
        )
    return base


def utc_now() -> datetime:
    """Timezone-aware ``now``, so ``ProcessingStats`` timestamps are comparable."""
    return datetime.now(UTC)
