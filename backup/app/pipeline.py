"""The synchronous orchestrator — BUILD-BRIEF M6, one document in, envelopes out.

Everything under ``app/stages/`` is a pure-ish function of its inputs; this
module is the only place that knows what order they run in and what each one
owes the next.  It has no model of its own and makes no judgement a stage could
have made: quality decides ``accept``/``repair``/``bounce``, the router decides
escalation, the validators decide validity, and ``assemble`` decides the shape.

**The order, and why each edge exists.**

1. ``s1_ingest`` — magic-byte sniff, resource caps, rasterisation.  Its
   ``document_id`` (``sha256`` of the *bytes*) is the idempotency key the API
   and the storage layer key on, so it is computed once, here.
2. ``s1b_bypass`` (PDF only) — a usable digital text layer removes recognition
   from the critical path entirely.
3. ``s15_quality`` — per page.  A ``bounce`` **short-circuits that page**: no
   recognition, no classification, no model call, and the reason travels as a
   machine-readable :class:`~app.contracts.envelope.BounceReason`.  When *every*
   page bounces the document bounces, and the envelope carries no
   ``document`` at all (BUILD-BRIEF sec. 0.5).
4. ``s4_ocr.run_tier1`` — batched recognition (invariant 4).  Stage 2's script
   vote is *inside* it (adjudication A-6 gives it the second recognition pass),
   so this module deliberately does not vote a second time.  It owes Stage 4 one
   thing the stage cannot see for itself: when a geometric repair forced the
   text layer to be discarded, that layer's *characters* are still a script
   signal no recognition head produced, and they travel as ``script_prior`` for
   the head probe (OPEN-QUESTIONS **Q12**).
5. ``s2b_lid`` — per-line language.  Page-level language is never set here; it
   is derived in ``assemble`` and nowhere else (invariant 1).
6. ``s3_classify`` — page class from pixels plus structural features.
7. ``s5_validators`` — **document-level**, thread-pooled inside the stage.  It
   runs once over every page rather than per page, because ``run_validators``
   owns its own pool and a per-page call would serialise it into N pools of one.
8. ``s4_router`` — per page, from validator failures and structural signals.
   Never from recognition confidence (invariant 3).  The 3% shadow sampler is
   the router's own, and a shadow page is *not* an escalation.
9. ``s4b_vlm`` — Tier 2 for escalated pages only.
10. ``s5_slm`` — extractive, span-indexed, budget-capped.
11. ``s6_translate`` — per field, entity-masked, advisory.
12. ``assemble`` — the grounding gate, the summary, the derived summaries.

**Two-stage delivery** (BUILD-BRIEF sec. 5).  When any page escalates, the
document is assembled twice: ``result_version=1, is_final=false`` at Tier-1
quality with an ``escalation_pending`` warning naming each page, then Tier 2
runs and the document is re-assembled as ``result_version=2, is_final=true``.
Version 2 is a *re-assembly*, never a patch of version 1 — the validators and
the SLM see the merged text, because Tier 2 exists precisely to change what the
page says.  ``on_envelope`` fires once per version and is what the webhook
dispatcher subscribes to.

**Failure policy.**  A :class:`~app.contracts.errors.DocumentRejected` becomes a
bounce envelope rather than an exception: PRD P0-3 makes "refused, and here is
the machine-readable reason" a *product* outcome, and the auto-reply (P1-2) is
generated from ``bounce_decision``.  Pass ``strict_rejections=True`` to get the
exception instead.  Everything else propagates —
:class:`~app.contracts.errors.MalwareDetected` most of all, because it must
never reach a complainant-facing template, and
:class:`~app.contracts.errors.ResidencyViolation`, which nothing may swallow.

This module does NOT handle:
  * persistence, job records or the audit trail (``app/storage/``);
  * HTTP, multipart parsing or status codes (``app/api/``);
  * webhook signing, retries or the DLQ (``app/webhooks.py``) — it only calls
    ``on_envelope``;
  * distribution across workers (``app/tasks.py``); the Celery chord calls the
    same stage functions in the same order and is asserted to produce the same
    envelope;
  * choosing or constructing a provider, or the residency gate on model traffic
    (``app.providers.factory``);
  * PII redaction.  The returned envelopes contain complainant data by
    construction; the logging filter installed by ``app/api`` is what keeps
    them out of the log.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

from prometheus_client import Counter, Histogram

from app.contracts.envelope import (
    BounceDecision,
    BounceReason,
    Field,
    ProcessingStats,
    QualityFlag,
    QualityVerdict,
    ResultEnvelope,
)
from app.contracts.errors import (
    ConfigurationError,
    DocumentRejected,
    ProviderError,
    ResidencyViolation,
)
from app.contracts.ocr import (
    OCRLine,
    OCRPage,
    OCRTier,
    PageClass,
    PageClassLabel,
    RawField,
)
from app.providers.base import JobBudget, ModelProvider, PageImage
from app.providers.factory import require_synthetic_corpus_file
from app.providers.local import UNDETERMINED_SCRIPT
from app.settings import Settings, get_settings
from app.stages.assemble import (
    PageAssembly,
    assemble_envelope,
    derive_language_summary,
    utc_now,
)
from app.stages.s1_ingest import IngestResult, RasterPage, Scanner, ingest
from app.stages.s1b_bypass import BypassResult, BypassVerdict, bypass_document
from app.stages.s2_script import identify_script
from app.stages.s2b_lid import identify_line_languages
from app.stages.s3_classify import ClassificationResult, classify_page
from app.stages.s4_ocr import Tier1Result, page_is_sideways, run_tier1
from app.stages.s4_router import (
    EscalationDecision,
    EscalationSignals,
    build_shadow_diff,
    route,
)
from app.stages.s4b_vlm import Tier2Result, apply_tier2, escalate_page
from app.stages.s5_slm import build_extraction_contexts, extract_pages
from app.stages.s5_validators import (
    BankTable,
    IFSCMaster,
    ValidatorReport,
    load_bank_table,
    run_validators,
)
from app.stages.s6_translate import translate_fields
from app.stages.s15_quality import (
    REPAIR_AUTO_INVERT,
    REPAIR_DESKEW,
    QualityGate,
    QualityThresholds,
    map_bbox_to_original,
)

__all__ = [
    "ESCALATION_PENDING_PREFIX",
    "GEOMETRIC_REPAIRS",
    "QUALITY_DEFECT_FLAGS",
    "PageOutcome",
    "PipelineResult",
    "run",
    "run_document",
]

_LOG = logging.getLogger(__name__)

#: Machine-readable warning naming a page whose Tier-2 result has not landed
#: yet.  ``is_final=false`` says *that* something is pending; this says *which*,
#: so a consumer acting on version 1 knows which fields may still move.
ESCALATION_PENDING_PREFIX: Final[str] = "escalation_pending"

#: Warning prefix for an SLM that could not be reached.  The document still
#: ships: validators own every identifier and amount (invariant 7), so an
#: unavailable SLM costs the free-text fields and nothing else.
SLM_UNAVAILABLE_PREFIX: Final[str] = "slm_unavailable"

#: Warning prefix for a Tier-2 attempt that did not produce usable lines.
TIER2_FAILED_PREFIX: Final[str] = "tier2_failed"

#: Repairs that move pixels.  A page that had one cannot reuse the PDF text
#: layer's coordinates, because those describe the page *before* the rotation.
GEOMETRIC_REPAIRS: Final[frozenset[str]] = frozenset(
    {REPAIR_DESKEW, REPAIR_AUTO_INVERT}
)

#: Quality observations that make ``EscalationSignals.quality_score_low`` true.
#:
#: WS-E2 left this open (decision E2-U6): the router consumes a boolean and
#: Stage 1.5 publishes a score whose only threshold is the bounce cut, so a page
#: that reaches the router is by definition above it.  The composer has to
#: define the signal, and it is defined here as **"the gate did not accept this
#: page as it arrived"** — a repair was applied, or a defect ceiling fired.
#: That reproduces the band WS-E2 documented (``docs/decisions-wse2.md`` sec. 4b
#: names the ``scan``/``heavy`` majority) without inventing a second numeric
#: threshold beside the calibrated one.  ``*_REPAIRED`` flags are excluded
#: because the repair itself is already counted through the verdict.
QUALITY_DEFECT_FLAGS: Final[frozenset[QualityFlag]] = frozenset(
    {
        QualityFlag.BLUR_ABOVE_CEILING,
        QualityFlag.NOISE_ABOVE_CEILING,
        QualityFlag.SHADOW_DETECTED,
        QualityFlag.EXPOSURE_OUT_OF_RANGE,
        QualityFlag.OCCLUSION_DETECTED,
        QualityFlag.TEXT_COVERAGE_BELOW_FLOOR,
        QualityFlag.RESOLUTION_BELOW_FLOOR,
        QualityFlag.SKEW_DETECTED,
        QualityFlag.JUNK_TEXT_LAYER,
    }
)

_DOCUMENTS = Counter(
    "pipeline_documents_total",
    "Documents through the synchronous orchestrator.",
    ["outcome"],
)
_VERSIONS = Counter(
    "pipeline_result_versions_total",
    "Envelopes published, by result_version.",
    ["version"],
)
_DURATION = Histogram(
    "pipeline_document_seconds",
    "Wall clock per document, fast path to final.",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)
_STAGE_SECONDS = Histogram(
    "pipeline_stage_seconds",
    "Wall clock per stage invocation.",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0),
)


# --------------------------------------------------------------------------- #
# Per-page working state
# --------------------------------------------------------------------------- #


@dataclass
class PageOutcome:
    """Everything the orchestrator learned about one page.

    Mutable and deliberately not a Pydantic model: it carries a NumPy page
    image, it is rewritten in place when Tier 2 lands, and it never leaves this
    module.  The published shape is :class:`~app.stages.assemble.PageAssembly`,
    built from it by :meth:`to_assembly`.
    """

    page_no: int
    page: OCRPage
    quality_score: float
    quality_flags: tuple[QualityFlag, ...]
    quality_verdict: QualityVerdict
    repairs_applied: tuple[str, ...]
    image: PageImage | None = None
    bounce_reason: BounceReason | None = None
    bounce_detail: str | None = None
    page_class: PageClass | None = None
    classification: ClassificationResult | None = None
    tier1: Tier1Result | None = None
    escalation: EscalationDecision | None = None
    escalated: bool = False
    shadow_sampled: bool = False
    tier2: Tier2Result | None = None
    structural_anomalies: tuple[str, ...] = ()
    raw_fields: dict[str, RawField] = field(default_factory=dict)

    @property
    def bounced(self) -> bool:
        """True when the quality gate refused this page."""
        return self.quality_verdict is QualityVerdict.BOUNCE

    @property
    def lines(self) -> list[OCRLine]:
        """The page's current lines (Tier 2's, once it has landed)."""
        return self.page.lines

    def to_assembly(self) -> PageAssembly:
        """The published per-page input to :func:`assemble_envelope`.

        This is the one place page coordinates cross from the pipeline's
        internal frame into the published contract, so it is where a repaired
        page's boxes are mapped back onto the document as submitted (Q11).
        Everything upstream — detection, classification, Tier 2 — works on the
        straightened raster and agrees with itself; everything downstream sees
        the frame the reviewer's own copy of the file is in.
        """
        return PageAssembly(
            page=self._page_in_original_frame(),
            quality_score=self.quality_score,
            quality_flags=self.quality_flags,
            quality_verdict=self.quality_verdict,
            page_class=self.page_class,
            escalated=self.escalated,
            shadow_sampled=self.shadow_sampled,
            repairs_applied=self.repairs_applied,
            raw_fields=dict(self.raw_fields),
        )

    def _page_in_original_frame(self) -> OCRPage:
        """``self.page`` with every bbox mapped back onto the submitted image.

        A no-op — the same object — unless deskew actually ran, which is both
        the common case and the case the corpus exercises, so nothing pays for
        this on a straight page.
        """
        page = self.page
        if not page.rotation_applied_deg or not page.lines:
            return page
        # The gate rotates by ``metrics.skew_deg`` (``deskew(working, skew)``)
        # while the envelope reports ``rotation_applied_deg = -skew_deg`` — the
        # correction from the *page's* point of view rather than the image's.
        # The transform to invert is the one deskew actually applied, so negate
        # back. Getting this backwards rotates boxes by twice the skew instead
        # of zero, which looks plausible and is wrong.
        degrees = -page.rotation_applied_deg
        height, width = page.height_px, page.width_px
        remapped = [
            line.model_copy(
                update={
                    "bbox": map_bbox_to_original(
                        line.bbox, height=height, width=width, degrees=degrees
                    )
                }
            )
            for line in page.lines
        ]
        return page.model_copy(update={"lines": remapped})


@dataclass(frozen=True)
class PipelineResult:
    """Every envelope one document produced, in publication order.

    One element for a document with no escalations; two when the fast path
    published before Tier 2 landed.  ``result_version`` increases by exactly one
    per element and only the last is ``is_final``.
    """

    envelopes: tuple[ResultEnvelope, ...]
    ingest: IngestResult | None = None
    pages: tuple[PageOutcome, ...] = ()

    def __post_init__(self) -> None:
        if not self.envelopes:
            raise ValueError("a pipeline run always publishes at least one envelope")

    @property
    def fast_path(self) -> ResultEnvelope:
        """The first envelope published (``result_version=1``)."""
        return self.envelopes[0]

    @property
    def final(self) -> ResultEnvelope:
        """The last envelope published; always ``is_final=True``."""
        return self.envelopes[-1]

    @property
    def document_id(self) -> str:
        """``sha256:<hex>`` of the submitted bytes."""
        return self.final.document_id


# --------------------------------------------------------------------------- #
# Stage timing
# --------------------------------------------------------------------------- #


class _Timings:
    """Wall clock per stage, aggregated and per page.

    ``ProcessingStats.stage_timings_ms`` is one flat mapping, so the key carries
    the structure: ``"s4_ocr"`` is the document total and ``"s4_ocr:p3"`` is
    page 3's share.  The eval harness wants both — the total to compare
    documents, the per-page split to find the page that cost the p95.
    """

    __slots__ = ("_totals",)

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, page_no: int | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.add(name, elapsed * 1000.0, page_no=page_no)
            _STAGE_SECONDS.labels(stage=name).observe(elapsed)

    def add(
        self, name: str, milliseconds: float, *, page_no: int | None = None
    ) -> None:
        """Record ``milliseconds`` against a stage (and optionally one page)."""
        self._totals[name] = self._totals.get(name, 0.0) + milliseconds
        if page_no is not None:
            key = f"{name}:p{page_no}"
            self._totals[key] = self._totals.get(key, 0.0) + milliseconds

    def snapshot(self) -> dict[str, float]:
        """A copy safe to hand to a Pydantic model."""
        return {key: round(value, 3) for key, value in self._totals.items()}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def run(
    source: Path | str | bytes,
    *,
    provider: ModelProvider,
    settings: Settings | None = None,
    **kwargs: object,
) -> ResultEnvelope:
    """Process one document and return its **final** envelope.

    The convenience form of :func:`run_document` for callers that do not care
    about two-stage delivery.  Every keyword :func:`run_document` accepts is
    forwarded; pass ``on_envelope`` to observe the fast path as well.
    """
    return run_document(
        source,
        provider=provider,
        settings=settings,
        **kwargs,  # type: ignore[arg-type]
    ).final


def run_document(
    source: Path | str | bytes,
    *,
    provider: ModelProvider,
    settings: Settings | None = None,
    job_id: str | None = None,
    filename: str | None = None,
    synthetic_declared: bool | None = None,
    on_envelope: Callable[[ResultEnvelope], None] | None = None,
    ifsc_master: IFSCMaster | None = None,
    scanner: Scanner | None = None,
    strict_rejections: bool = False,
    requested_fields: Sequence[str] = (),
    starting_result_version: int = 1,
) -> PipelineResult:
    """Process one document, publishing one envelope per ``result_version``.

    Args:
        source: a filesystem path (routed through the synthetic-corpus gate) or
            the raw attachment bytes (see ``synthetic_declared``).
        provider: any :class:`~app.providers.base.ModelProvider`.  The
            orchestrator never constructs one — the residency gate lives in the
            factory and must have run before a document is submitted.
        job_id: recorded on every envelope; the storage layer's key.
        synthetic_declared: required for a **bytes** source under a remote
            profile.  A path carries its provenance in a sidecar; bytes carry
            none, so the submitter must declare it and the API surfaces that as
            an explicit form field.  This is a positive declaration in the same
            shape as the sidecar, not a bypass flag: there is no value of it
            that lets an undeclared payload through.
        on_envelope: called with each published envelope, in version order,
            **before** the next version is computed.  Webhook delivery hangs
            here.  An exception from the callback aborts the run.
        ifsc_master: overrides the master named by ``IFSC_MASTER_PATH`` (Q8).
        strict_rejections: re-raise ``DocumentRejected`` instead of returning a
            bounce envelope.
        requested_fields: user-selected field names to prioritize, while still
            discovering extra fields. Published results include both groups.
        starting_result_version: next append-only version when re-extracting
            an existing job; defaults to 1 for a new document.

    Returns:
        A :class:`PipelineResult` whose ``envelopes`` are in publication order.

    Raises:
        ResidencyViolation: an undeclared payload under a remote profile.
        MalwareDetected: the scanner refused the attachment. Never converted to
            a bounce — the reply would tell the sender which signature fired.
        DocumentRejected: only when ``strict_rejections`` is set.
        ProviderError: a model failure the stage that raised it chose not to
            absorb (the SLM and Tier 2 are absorbed; OCR and MT are not).
    """
    from app.field_selection import parse_requested_fields

    requested_fields = parse_requested_fields(requested_fields)
    if starting_result_version < 1:
        raise ValueError("starting_result_version must be positive")
    active = settings if settings is not None else get_settings()
    if requested_fields:
        active = active.model_copy(update={"EXTRACT_ALL_FIELDS": True})
    started_at = utc_now()
    wall_start = time.perf_counter()
    timings = _Timings()
    published: list[ResultEnvelope] = []

    def publish(envelope: ResultEnvelope) -> None:
        # Re-extractions append to the same job without replacing old evidence.
        envelope = ResultEnvelope.model_validate(
            {
                **envelope.model_dump(),
                "result_version": envelope.result_version + starting_result_version - 1,
                "requested_fields": requested_fields,
            }
        )
        published.append(envelope)
        _VERSIONS.labels(version=str(envelope.result_version)).inc()
        if on_envelope is not None:
            on_envelope(envelope)

    with timings.stage("s0_read"):
        data, resolved_name = _read_source(source, active, filename, synthetic_declared)

    document_id = _document_id(data)
    try:
        with timings.stage("s1_ingest"):
            ingested = ingest(
                data, filename=resolved_name, settings=active, scanner=scanner
            )
    except DocumentRejected as rejection:
        if strict_rejections:
            raise
        _DOCUMENTS.labels(outcome="rejected").inc()
        envelope = _bounce_envelope(
            document_id=document_id,
            job_id=job_id,
            settings=active,
            decision=BounceDecision.from_rejection(
                rejection.reason, detail=rejection.message
            ),
            started_at=started_at,
            timings=timings,
        )
        publish(envelope)
        _DURATION.observe(time.perf_counter() - wall_start)
        return PipelineResult(envelopes=tuple(published))

    document_id = ingested.document_id
    _LOG.info(
        "pipeline start document=%s job=%s pages=%d type=%s profile=%s",
        document_id,
        job_id,
        ingested.report.page_count,
        ingested.report.sniffed_type.value,
        active.PROVIDER_PROFILE.value,
    )

    bypasses = _bypass_results(ingested, timings)
    outcomes = [
        _process_page(
            raster,
            bypass=bypasses.get(raster.page_no),
            provider=provider,
            settings=active,
            timings=timings,
        )
        for raster in ingested.pages
    ]

    if outcomes and all(outcome.bounced for outcome in outcomes):
        _DOCUMENTS.labels(outcome="bounced").inc()
        envelope = _bounce_envelope(
            document_id=document_id,
            job_id=job_id,
            settings=active,
            decision=_document_bounce_decision(outcomes),
            started_at=started_at,
            timings=timings,
            pages=outcomes,
        )
        publish(envelope)
        _DURATION.observe(time.perf_counter() - wall_start)
        return PipelineResult(
            envelopes=tuple(published), ingest=ingested, pages=tuple(outcomes)
        )

    master = ifsc_master if ifsc_master is not None else _load_ifsc_master(active)
    bank_table = _load_bank_table(active)
    budget = JobBudget(active.JOB_TOKEN_BUDGET, job_id=job_id or document_id)
    warnings: list[str] = []

    report = _run_validators(
        outcomes, master=master, bank_table=bank_table, timings=timings
    )
    _route_pages(outcomes, document_id=document_id, report=report, settings=active)
    _extract_fields(
        outcomes,
        provider=provider,
        settings=active,
        document_id=document_id,
        report=report,
        budget=budget,
        timings=timings,
        warnings=warnings,
        requested_fields=requested_fields,
    )

    pending = [outcome.page_no for outcome in outcomes if _awaits_tier2(outcome)]
    version = 1
    envelope = _assemble(
        outcomes,
        document_id=document_id,
        job_id=job_id,
        settings=active,
        provider=provider,
        report=report,
        result_version=version,
        requested_fields=requested_fields,
        is_final=not pending,
        started_at=started_at,
        timings=timings,
        budget=budget,
        warnings=[
            *warnings,
            *(f"{ESCALATION_PENDING_PREFIX}:page={page_no}" for page_no in pending),
        ],
    )
    publish(envelope)

    if pending:
        _run_tier2(
            outcomes,
            provider=provider,
            settings=active,
            document_id=document_id,
            timings=timings,
            warnings=warnings,
        )
        report = _run_validators(
            outcomes, master=master, bank_table=bank_table, timings=timings
        )
        _extract_fields(
            outcomes,
            provider=provider,
            settings=active,
            document_id=document_id,
            report=report,
            budget=budget,
            timings=timings,
            warnings=warnings,
            requested_fields=requested_fields,
            only_pages={outcome.page_no for outcome in outcomes if outcome.escalated},
        )
        version += 1
        envelope = _assemble(
            outcomes,
            document_id=document_id,
            job_id=job_id,
            settings=active,
            provider=provider,
            report=report,
            result_version=version,
            requested_fields=requested_fields,
            is_final=True,
            started_at=started_at,
            timings=timings,
            budget=budget,
            warnings=warnings,
        )
        publish(envelope)

    _DOCUMENTS.labels(outcome="processed").inc()
    _DURATION.observe(time.perf_counter() - wall_start)
    _LOG.info(
        "pipeline done document=%s versions=%d escalated=%d",
        document_id,
        len(published),
        sum(1 for outcome in outcomes if outcome.escalated),
    )
    return PipelineResult(
        envelopes=tuple(published), ingest=ingested, pages=tuple(outcomes)
    )


# --------------------------------------------------------------------------- #
# Source resolution and the residency gate
# --------------------------------------------------------------------------- #


def _read_source(
    source: Path | str | bytes,
    settings: Settings,
    filename: str | None,
    synthetic_declared: bool | None,
) -> tuple[bytes, str | None]:
    """Resolve ``source`` to bytes, enforcing the synthetic-data declaration.

    A path is checked by :func:`~app.providers.factory.require_synthetic_corpus_file`
    — the sidecar or directory manifest is its provenance.  Bytes have no
    sidecar to read, so under a remote profile the caller must declare them.

    Raises:
        ResidencyViolation: undeclared bytes under a remote profile, or a path
            without a ``synthetic: true`` declaration.
        FileNotFoundError: the path does not exist.
    """
    if isinstance(source, (str, Path)):
        path = require_synthetic_corpus_file(Path(source), settings)
        return path.read_bytes(), filename or path.name
    if not isinstance(source, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"pipeline source must be a path or bytes, got {type(source).__name__}"
        )
    data = bytes(source)
    if settings.is_remote_profile and synthetic_declared is not True:
        raise ResidencyViolation(
            "an uploaded payload carries no synthetic-data sidecar and the "
            f"{settings.PROVIDER_PROFILE.value} profile routes text to a "
            "non-Indian endpoint; the submitter must declare the data synthetic "
            "(PLAN.md invariant 6). There is no bypass flag.",
            context={
                "profile": settings.PROVIDER_PROFILE.value,
                "filename": filename,
                "remedy": "declare synthetic data, or run under cpu_node",
            },
        )
    return data, filename


def _document_id(data: bytes) -> str:
    """``sha256:<hex>``, computed without parsing — valid even for a rejection."""
    from app.stages.s1_ingest import idempotency_key

    return idempotency_key(data)


def _load_ifsc_master(settings: Settings) -> IFSCMaster:
    """The IFSC master named by ``IFSC_MASTER_PATH`` — OPEN-QUESTIONS **Q8**.

    Resolving Q8 is one setting and this one line: WS-E3 built the loader to
    take the real RBI list as a path change (``docs/decisions-wse3.md`` D-E3-8)
    but left the orchestrator hard-wired to the committed fixture, so an
    air-gapped deployment had to edit the ``run_validators`` call site.

    A missing or malformed file **raises**.  Falling back to the synthetic
    fixture would silently downgrade an authoritative deployment to
    "structurally valid, not confirmed" on every IFSC in the country.

    Raises:
        ConfigurationError: the configured master cannot be loaded.
    """
    path = settings.ifsc_master_path
    try:
        return _cached_ifsc_master(str(path))
    except (OSError, ValueError, KeyError) as failure:
        raise ConfigurationError(
            f"IFSC_MASTER_PATH={path} could not be loaded: {failure}. The "
            "validator will not fall back to the synthetic fixture — that "
            "would report unverified IFSCs as structurally confirmed.",
            context={"path": str(path)},
        ) from failure


def _cached_ifsc_master(path: str) -> IFSCMaster:
    """Load-once cache keyed on the resolved path (a 170k-row CSV is not free)."""
    cached = _IFSC_CACHE.get(path)
    if cached is None:
        cached = IFSCMaster.load(path)
        _IFSC_CACHE[path] = cached
    return cached


_IFSC_CACHE: dict[str, IFSCMaster] = {}


def _load_bank_table(settings: Settings) -> BankTable:
    """The bank-name table named by ``BANK_ALIASES_PATH``.

    The same omission Q8 had, in the other table: the loader took a path and the
    orchestrator never passed one, so every deployment ran the committed
    five-bank fixture no matter what it was configured with.  A five-bank table
    does not merely miss the other banks — a name that is not in it can still
    fuzzy-match one that is, and be published as that bank.

    A missing or malformed file **raises**, for the same reason the IFSC master
    does: falling back would quietly restore the fixture on a node the operator
    believes is running the full list.

    Raises:
        ConfigurationError: the configured table cannot be loaded.
    """
    path = settings.bank_aliases_path
    try:
        return load_bank_table(path)
    except (OSError, ValueError, KeyError) as failure:
        raise ConfigurationError(
            f"BANK_ALIASES_PATH={path} could not be loaded: {failure}. The "
            "validator will not fall back to the committed table — that would "
            "run a deployment on a fixture it did not choose.",
            context={"path": str(path)},
        ) from failure


# --------------------------------------------------------------------------- #
# Stage 1b — the digital text layer
# --------------------------------------------------------------------------- #


def _bypass_results(
    ingested: IngestResult, timings: _Timings
) -> dict[int, BypassResult]:
    """Per-page bypass verdicts for a PDF; empty for an image.

    A raster source has no text layer at all, so the stage is skipped rather
    than run to produce ``NO_TEXT_LAYER`` on every page.
    """
    if not ingested.is_pdf:
        return {}
    with timings.stage("s1b_bypass"):
        results = bypass_document(ingested.data)
    return {result.page_no: result for result in results}


# --------------------------------------------------------------------------- #
# Per-page processing
# --------------------------------------------------------------------------- #


def _process_page(
    raster: RasterPage,
    *,
    bypass: BypassResult | None,
    provider: ModelProvider,
    settings: Settings,
    timings: _Timings,
) -> PageOutcome:
    """Quality, recognition, LID and classification for one page."""
    page_no = raster.page_no
    gate = QualityGate(QualityThresholds.from_settings(settings))
    extra_flags = (
        (QualityFlag.JUNK_TEXT_LAYER,)
        if bypass is not None and bypass.verdict is BypassVerdict.JUNK_TEXT_LAYER
        else ()
    )
    with timings.stage("s15_quality", page_no):
        quality = gate.evaluate(raster.image, page_no, extra_flags=extra_flags)
    assessment = quality.assessment

    if assessment.bounced or quality.image is None:
        _LOG.info(
            "page bounced document_page=%d reason=%s score=%.3f",
            page_no,
            assessment.bounce_reason.value if assessment.bounce_reason else "unknown",
            assessment.quality_score,
        )
        return PageOutcome(
            page_no=page_no,
            page=OCRPage(
                page_no=page_no,
                width_px=raster.width_px,
                height_px=raster.height_px,
                lines=[],
                tier=OCRTier.NONE,
            ),
            quality_score=assessment.quality_score,
            quality_flags=assessment.quality_flags,
            quality_verdict=assessment.verdict,
            repairs_applied=assessment.repairs_applied,
            bounce_reason=assessment.bounce_reason,
            bounce_detail=assessment.detail,
        )

    image = quality.image
    repairs = assessment.repairs_applied
    bypass_lines = _usable_bypass_lines(bypass, repairs)

    with timings.stage("s4_ocr", page_no):
        tier1 = run_tier1(
            None if bypass_lines is not None else image,
            provider=provider,
            page_no=page_no,
            settings=settings,
            script_prior=(
                None if bypass_lines is not None else _text_layer_script(bypass)
            ),
            bypass_lines=bypass_lines,
            width_px=raster.width_px,
            height_px=raster.height_px,
            rotation_applied_deg=(
                -assessment.metrics.skew_deg if REPAIR_DESKEW in repairs else 0.0
            ),
            dpi_estimate=raster.dpi,
        )

    page = tier1.page
    if page.lines:
        with timings.stage("s2b_lid", page_no):
            lid = identify_line_languages(
                page.lines, provider=provider, settings=settings
            )
        page = page.model_copy(update={"lines": list(lid.lines)})

    with timings.stage("s3_classify", page_no):
        classification = classify_page(
            image, lines=page.lines, provider=provider, settings=settings
        )

    outcome = PageOutcome(
        page_no=page_no,
        page=page,
        quality_score=assessment.quality_score,
        quality_flags=assessment.quality_flags,
        quality_verdict=assessment.verdict,
        repairs_applied=repairs,
        image=image,
        page_class=classification.page_class,
        classification=classification,
        tier1=tier1,
    )
    outcome.structural_anomalies = _structural_anomalies(outcome, bypass)
    if "page_sideways" in outcome.structural_anomalies:
        # ``ROTATION_DETECTED`` has been in the envelope contract since WS-B and
        # was raised by nothing: the pixel-projection measurement that was meant
        # to drive it could not separate upright from rotated, so the flag was
        # left defined and dead.  The detector's box geometry does separate them,
        # and it is the same observation the flag was always for — so it is
        # published on the page record rather than given a second name.
        outcome.quality_flags = tuple(
            dict.fromkeys((*outcome.quality_flags, QualityFlag.ROTATION_DETECTED))
        )
    return outcome


def _usable_bypass_lines(
    bypass: BypassResult | None, repairs: Sequence[str]
) -> list[OCRLine] | None:
    """The text-layer lines, when they are still true of the pixels we hold.

    A geometric repair invalidates them: ``deskew`` and ``auto_invert`` rewrite
    the raster, and the text layer's rectangles describe the page as authored.
    Recognising such a page costs one OCR pass; trusting stale coordinates would
    put every field's evidence box in the wrong place, which is worse.
    """
    if bypass is None or not bypass.used or bypass.page is None:
        return None
    if GEOMETRIC_REPAIRS.intersection(repairs):
        _LOG.info(
            "text layer discarded on page %d: quality applied %s and the layer's "
            "coordinates describe the unrepaired page",
            bypass.page_no,
            sorted(GEOMETRIC_REPAIRS.intersection(repairs)),
        )
        return None
    return list(bypass.page.lines)


def _text_layer_script(bypass: BypassResult | None) -> str | None:
    """The script of a PDF text layer this page could not use — Stage 4's prior.

    When a geometric repair forces :func:`_usable_bypass_lines` to throw the
    layer away, its *coordinates* are stale but its **characters** are not:
    they were authored, not recognised, so voting on them is the one piece of
    script evidence in this pipeline that no recognition head produced.  Stage
    4's head probe takes it as a tie-breaker (``script_prior``), which is the
    only thing this pipeline knows that the probe does not (OPEN-QUESTIONS
    **Q12**).

    ``None`` unless the vote is decisive — an undetermined or split layer is
    not evidence, and Stage 4 must not receive a guess dressed as a prior.
    """
    if bypass is None or bypass.page is None or not bypass.page.lines:
        return None
    decision = identify_script(lines=list(bypass.page.lines))
    if not decision.is_confident or decision.script == UNDETERMINED_SCRIPT:
        return None
    return decision.rec_head_script


def _structural_anomalies(
    outcome: PageOutcome, bypass: BypassResult | None
) -> tuple[str, ...]:
    """Layout-level oddities the router scores — never a confidence number.

    Each is a fact about *structure* (invariant 3): a page that produced no
    text, a page the classifier declined to name, a text layer that failed the
    junk guard.  The router saturates at three, so this list is short by design.
    """
    anomalies: list[str] = []
    if not outcome.page.lines:
        anomalies.append("no_recognised_text")
    if page_is_sideways(outcome.page.lines):
        # A quarter-turned page is not partially readable, it is unreadable, and
        # nothing upstream could see it: Stage 1 honours EXIF, so a phone photo
        # arrives upright, but a flatbed scan fed in sideways carries no EXIF
        # and every downstream number about it is meaningless.  Escalate rather
        # than rotate — the signal says "not upright", never "turn it this way".
        anomalies.append("page_sideways")
    classification = outcome.classification
    if classification is not None and (
        classification.below_confidence_floor
        or classification.page_class.label.value == "unknown"
    ):
        anomalies.append("page_class_unknown")
    if bypass is not None and bypass.verdict is BypassVerdict.JUNK_TEXT_LAYER:
        anomalies.append("junk_text_layer")
    return tuple(anomalies)


# --------------------------------------------------------------------------- #
# Stage 5 — validators (document level, thread-pooled inside the stage)
# --------------------------------------------------------------------------- #


def _run_validators(
    outcomes: Sequence[PageOutcome],
    *,
    master: IFSCMaster,
    bank_table: BankTable,
    timings: _Timings,
) -> ValidatorReport:
    """Deterministic extraction over every non-bounced page, in one pool."""
    pages = [outcome.page for outcome in outcomes if not outcome.bounced]
    with timings.stage("s5_validators"):
        return run_validators(pages, ifsc_master=master, bank_table=bank_table)


# --------------------------------------------------------------------------- #
# Stage 4 router
# --------------------------------------------------------------------------- #


def _route_pages(
    outcomes: Sequence[PageOutcome],
    *,
    document_id: str,
    report: ValidatorReport,
    settings: Settings,
) -> None:
    """Decide escalation and shadow sampling for every non-bounced page."""
    for outcome in outcomes:
        if outcome.bounced:
            continue
        failures = tuple(
            hit.field_name
            for hit in report.hits_on_page(outcome.page_no)
            if not hit.validator_passed
        )
        tier1 = outcome.tier1
        signals = EscalationSignals(
            validator_failures=len(failures),
            validator_failed_fields=failures,
            structural_anomalies=outcome.structural_anomalies,
            handwriting_detected=bool(tier1 and tier1.handwriting_ratio > 0.0),
            handwriting_ratio=tier1.handwriting_ratio if tier1 else 0.0,
            script_disagreement=bool(tier1 and tier1.script_disagreement),
            quality_score_low=_quality_low(outcome),
            quality_score=outcome.quality_score,
        )
        decision = route(
            signals,
            doc_id=document_id,
            page_no=outcome.page_no,
            settings=settings,
        )
        outcome.escalation = decision
        outcome.shadow_sampled = decision.shadow_sampled


def _quality_low(outcome: PageOutcome) -> bool:
    """Whether Stage 1.5 accepted this page as it arrived — see the constant."""
    if outcome.quality_verdict is QualityVerdict.REPAIR:
        return True
    return bool(QUALITY_DEFECT_FLAGS.intersection(outcome.quality_flags))


def _awaits_tier2(outcome: PageOutcome) -> bool:
    """True when the router asked for a second reader and it has not run."""
    return (
        outcome.escalation is not None
        and outcome.escalation.runs_tier2
        and outcome.tier2 is None
    )


# --------------------------------------------------------------------------- #
# Stage 4b — Tier 2
# --------------------------------------------------------------------------- #


def _run_tier2(
    outcomes: Sequence[PageOutcome],
    *,
    provider: ModelProvider,
    settings: Settings,
    document_id: str,
    timings: _Timings,
    warnings: list[str],
) -> None:
    """Escalate every page the router selected, then re-label its lines.

    A failed Tier-2 call is **not** a failed document: ``escalate_page`` returns
    a typed outcome for a rate limit, an absent artifact or a profile with no
    VLM, and the page ships at Tier-1 quality with the reason recorded.  A
    shadow-sampled page runs Tier 2 for the diff only; its lines are never
    replaced, because a control that changes the output is not a control.
    """
    del settings
    for outcome in outcomes:
        if not _awaits_tier2(outcome):
            continue
        with timings.stage("s4b_vlm", outcome.page_no):
            result = escalate_page(outcome.image, outcome.page.lines, provider=provider)
        outcome.tier2 = result
        if not result.succeeded:
            warnings.append(
                f"{TIER2_FAILED_PREFIX}:page={outcome.page_no}"
                f":reason={result.outcome.value}"
                + (f":code={result.error_code}" if result.error_code else "")
            )
            _LOG.warning(
                "tier 2 did not land for document=%s page=%d outcome=%s code=%s",
                document_id,
                outcome.page_no,
                result.outcome.value,
                result.error_code,
            )
            continue
        if outcome.shadow_sampled:
            diff = build_shadow_diff(
                doc_id=document_id,
                page_no=outcome.page_no,
                tier1_lines=outcome.page.lines,
                tier2_lines=result.tier2_lines,
                decision=outcome.escalation,
            )
            _LOG.info(
                "shadow diff document=%s page=%d changed=%d/%d material=%s",
                document_id,
                outcome.page_no,
                diff.changed_lines,
                diff.tier1_line_count,
                diff.is_material,
            )
            continue
        outcome.page = apply_tier2(outcome.page, result)
        outcome.escalated = True
        if outcome.page.lines:
            with timings.stage("s2b_lid", outcome.page_no):
                lid = identify_line_languages(outcome.page.lines, provider=provider)
            outcome.page = outcome.page.model_copy(update={"lines": list(lid.lines)})


# --------------------------------------------------------------------------- #
# Stage 5 — the SLM
# --------------------------------------------------------------------------- #


def _extract_fields(
    outcomes: Sequence[PageOutcome],
    *,
    provider: ModelProvider,
    settings: Settings,
    document_id: str,
    report: ValidatorReport,
    budget: JobBudget,
    timings: _Timings,
    warnings: list[str],
    only_pages: set[int] | None = None,
    requested_fields: Sequence[str] = (),
) -> None:
    """Run the extractive SLM over pages with text, absorbing provider failure.

    A provider error here costs the free-text fields and nothing else:
    identifiers, amounts and dates are the validators' by invariant 7, and they
    are already extracted.  The failure is recorded as a machine-readable
    warning so the rate is measurable rather than invisible.

    ``only_pages`` restricts the re-run after Tier 2 to the pages whose text
    actually changed; the others keep the raw fields the fast path produced.
    """
    candidates = [
        outcome
        for outcome in outcomes
        if not outcome.bounced
        and outcome.page.lines
        and (only_pages is None or outcome.page_no in only_pages)
    ]
    if not candidates:
        return

    by_page = {outcome.page_no: outcome for outcome in candidates}
    contexts = build_extraction_contexts(
        [outcome.page for outcome in candidates],
        target_fields=requested_fields or None,
        document_id=document_id,
        page_classes={
            outcome.page_no: (
                outcome.page_class.label
                if outcome.page_class is not None
                else PageClassLabel.UNKNOWN
            )
            for outcome in candidates
        },
        primary_scripts={
            outcome.page_no: _page_script(outcome) for outcome in candidates
        },
        primary_languages={
            outcome.page_no: _page_language(outcome) for outcome in candidates
        },
        masked_spans={
            outcome.page_no: (
                []
                if settings.EXTRACT_ALL_FIELDS
                and settings.PROVIDER_PROFILE.value != "laptop"
                else report.masked_spans(outcome.page_no)
            )
            for outcome in candidates
        },
        budget=budget,
        settings=settings,
    )
    if not contexts:
        return
    try:
        with timings.stage("s5_slm"):
            results = extract_pages(provider, contexts)
    except ProviderError as failure:
        warnings.append(f"{SLM_UNAVAILABLE_PREFIX}:{failure.code}")
        _LOG.warning(
            "SLM unavailable for document=%s (%s); deterministic fields stand",
            document_id,
            failure.code,
        )
        return
    for context, fields in zip(contexts, results, strict=True):
        by_page[context.page_no].raw_fields = dict(fields)


def _page_script(outcome: PageOutcome) -> str:
    """The page's dominant script, derived from its line labels (invariant 1)."""
    summary = derive_language_summary(outcome.page.lines, code_mix_threshold=1.0)
    return summary.primary_script or "Zyyy"


def _page_language(outcome: PageOutcome) -> str | None:
    """The page's dominant language, derived; ``None`` when nothing is labelled."""
    summary = derive_language_summary(outcome.page.lines, code_mix_threshold=1.0)
    return summary.primary_language


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _assemble(
    outcomes: Sequence[PageOutcome],
    *,
    document_id: str,
    job_id: str | None,
    settings: Settings,
    provider: ModelProvider,
    report: ValidatorReport,
    result_version: int,
    is_final: bool,
    started_at: datetime,
    timings: _Timings,
    budget: JobBudget,
    warnings: Sequence[str],
    requested_fields: Sequence[str] = (),
) -> ResultEnvelope:
    """Build one published envelope from the current state of every page."""
    finished_at = utc_now()
    document_summary = derive_language_summary(
        [line for outcome in outcomes for line in outcome.page.lines],
        code_mix_threshold=settings.CODE_MIX_THRESHOLD,
    )
    translator = _translator(
        provider,
        report=report,
        src_language=document_summary.primary_language,
    )
    with timings.stage("assemble"):
        envelope = assemble_envelope(
            document_id=document_id,
            provider_profile=settings.PROVIDER_PROFILE.value,
            pages=[outcome.to_assembly() for outcome in outcomes],
            validator_report=report,
            translator=translator,
            job_id=job_id,
            result_version=result_version,
            is_final=is_final,
            processing=ProcessingStats(
                started_at=started_at,
                finished_at=finished_at,
                tokens_used=budget.used,
                stage_timings_ms=timings.snapshot(),
            ),
            warnings=tuple(warnings),
            requested_fields=requested_fields,
            settings=settings,
        )
    return envelope


def _translator(
    provider: ModelProvider,
    *,
    report: ValidatorReport,
    src_language: str | None,
) -> Callable[[dict[str, Field]], Mapping[str, Field]] | None:
    """A per-field translator bound to this document, or ``None``.

    ``None`` when the document is already English or carries no language label:
    ``translate_fields`` would return the fields unchanged, and constructing the
    partial would still import the MT engine's name for nothing.

    Every validator hit's value is passed as a protected literal so an account
    number inside a free-text sentence is sentinel-masked before MT and restored
    after (invariant 10).
    """
    if src_language is None or src_language == "eng":
        return None
    from app.providers.local import MT_ENGINE_NAME

    protected = tuple(sorted({hit.value for hit in report.hits}))

    def translate(fields: dict[str, Field]) -> Mapping[str, Field]:
        return translate_fields(
            provider,
            fields,
            src_language=src_language,
            mt_engine=MT_ENGINE_NAME,
            protected_values=protected,
        )

    return translate


def _document_bounce_decision(outcomes: Sequence[PageOutcome]) -> BounceDecision:
    """The document-level bounce, keeping a single page's reason when it is one.

    A one-page document that bounced for ``BLUR_ABOVE_CEILING`` must say so —
    the auto-reply tells the complainant to rescan, not that "all pages
    bounced".  Only a multi-page document with mixed reasons falls back to
    ``ALL_PAGES_BOUNCED``.
    """
    reasons = {outcome.bounce_reason for outcome in outcomes if outcome.bounce_reason}
    if len(reasons) == 1:
        only = reasons.pop()
        detail = next(
            (o.bounce_detail for o in outcomes if o.bounce_detail is not None), None
        )
        return BounceDecision(
            decision=QualityVerdict.BOUNCE, reason=only, detail=detail
        )
    return BounceDecision(
        decision=QualityVerdict.BOUNCE,
        reason=BounceReason.ALL_PAGES_BOUNCED,
        detail=(
            f"{len(outcomes)} page(s) bounced: "
            + ", ".join(sorted(reason.value for reason in reasons))
            if reasons
            else None
        ),
    )


def _bounce_envelope(
    *,
    document_id: str,
    job_id: str | None,
    settings: Settings,
    decision: BounceDecision,
    started_at: datetime,
    timings: _Timings,
    pages: Sequence[PageOutcome] = (),
) -> ResultEnvelope:
    """A final, document-less envelope carrying the machine-readable reason."""
    return assemble_envelope(
        document_id=document_id,
        provider_profile=settings.PROVIDER_PROFILE.value,
        pages=[outcome.to_assembly() for outcome in pages],
        bounce_decision=decision,
        job_id=job_id,
        result_version=1,
        is_final=True,
        processing=ProcessingStats(
            started_at=started_at,
            finished_at=utc_now(),
            stage_timings_ms=timings.snapshot(),
        ),
        settings=settings,
    )
