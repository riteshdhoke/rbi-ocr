#!/usr/bin/env python3
"""run_corpus_eval.py — drive the **real** pipeline over the corpus and emit harness predictions.

This is the M7 exit-gate driver.  It is deliberately thin: every judgement about
the document belongs to ``app.pipeline`` and every judgement about the *score*
belongs to ``tools/eval_harness.py``.  What lives here is the third thing —
turning one :class:`~app.pipeline.PipelineResult` into one prediction file the
harness will accept, and doing that 177 times without losing the run to a single
bad page.

    .venv\\Scripts\\python.exe tools/run_corpus_eval.py                     # dev split, SLM on
    .venv\\Scripts\\python.exe tools/run_corpus_eval.py --limit 5 --no-slm  # offline smoke
    .venv\\Scripts\\python.exe tools/run_corpus_eval.py --split all --final # the published run

**The locked split is the harness's, not ours.**  ``select_documents`` is
imported from ``tools/eval_harness.py`` rather than reimplemented, so a driver
that wrote predictions for variants 3-4 while the harness refused to score them
is impossible by construction.  ``--split test``/``--split all`` raise
``LockedSplitError`` without ``--final``, exactly as the harness does, and
``--final`` is passed through to the scoring step unchanged.

**Sequential by design.**  Documents run one at a time in a single worker
thread.  Concurrency would roughly halve the wall clock of a cold run — a live
SLM call is ~10-20 s of pure network wait — but p50/p95 end-to-end latency is a PRD
target, and a number measured under self-inflicted contention is not that
target's number.  The provider response cache, not parallelism, is what makes
the *second* run cheap.

**Resumability comes from the cache, not from a checkpoint file.**  Every SLM
response is keyed by ``sha256(model + prompt + params)`` in
``PROVIDER_CACHE_PATH``, so an interrupted sweep costs the OCR again (~1.3 s per
page) and the model calls not at all.  ``--resume`` additionally skips documents
whose prediction file already exists, for when even the OCR is not worth
repeating.

**Failure is data.**  A document that raises, or that blows the per-document
wall-clock budget, is written as a prediction carrying an ``error`` block with
``pages: []`` and ``fields: {}``.  The sweep continues.  Recognition and field
recall therefore take the honest hit — the pipeline produced nothing for that
page — and the manifest counts the failures separately so they cannot hide
inside an aggregate.

**Per-document budget on Windows.**  There is no ``SIGALRM``, and a Python
thread cannot be killed.  The budget is enforced by running each document in a
single-worker pool and abandoning the worker on timeout; the pool is then
rebuilt and the abandoned thread is given a grace period to finish so two
documents are never in flight at once.  If the grace period expires the event is
counted in the manifest as ``overlap_events`` and the latency numbers after it
are, honestly, suspect.

**Never printed:** ``HF_TOKEN``.  The manifest records ``hf_token_present:
true|false`` and the token's length is not recorded either.

This module does NOT handle:
  * scoring — that is ``tools/eval_harness.py``, which this tool shells into via
    its ``evaluate()``/``render_report()`` functions when ``--score`` is set;
  * the adversarial set (``corpus/adversarial/``) — those assert typed
    rejections in ``tests/unit/test_s1_ingest.py``, not metrics;
  * multi-page documents — the corpus is one page per document, and the
    document-level quality verdict here is the *worst* page verdict, which is
    only obviously right while that holds;
  * retry of a failed document — one attempt each, so the failure rate is the
    failure rate;
  * choosing the provider profile or the residency gate (``app.providers.factory``);
  * PII redaction.  Predictions contain the corpus's synthetic complainant data
    by construction, which is exactly why the corpus is synthetic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PLAN.md invariant 5: importing ``app`` sets the thread caps, and it must
# happen before numpy/onnxruntime/cv2 are dragged in by anything below.
import app  # noqa: F401  (imported for its import-time side effect)
from app.contracts.envelope import QualityVerdict, ResultEnvelope
from app.contracts.errors import PipelineError
from app.providers.base import ModelProvider
from app.providers.cache import ResponseCache
from app.settings import Settings, get_settings

__all__ = [
    "TOOL_VERSION",
    "CorpusEvalError",
    "DocRecord",
    "DocumentRunner",
    "DocumentTimeout",
    "RunConfig",
    "SLMDisabledProvider",
    "build_manifest",
    "document_quality",
    "error_prediction",
    "load_harness",
    "main",
    "prediction_from_result",
    "run_sweep",
    "select_corpus_documents",
    "slm_outage_codes",
]

TOOL_VERSION: Final[str] = "1.0.0"

_LOG = logging.getLogger("run_corpus_eval")

#: Worst-first, so ``max`` over this ordering gives the document verdict.
_VERDICT_SEVERITY: Final[dict[str, int]] = {"accept": 0, "repair": 1, "bounce": 2}

#: Warning prefix the orchestrator emits when the SLM could not be reached.
_SLM_UNAVAILABLE: Final[str] = "slm_unavailable"

#: The ``slm_unavailable:<code>`` codes that mean **the endpoint is down**, as
#: opposed to "the model answered and the answer was wrong".  Only these arm the
#: fallback.  ``PROVIDER_RESPONSE_INVALID`` deliberately does not: the bake-off
#: measured 0.56 schema validity for the default SLM (``docs/bakeoff/REPORT.md``),
#: so a run of five invalid answers is an ordinary event for a healthy model and
#: swapping the model on it would silently change what the sweep is measuring.
_SLM_OUTAGE_CODES: Final[frozenset[str]] = frozenset(
    {"PROVIDER_UNAVAILABLE", "PROVIDER_RATE_LIMITED", "PROVIDER_METHOD_NOT_SUPPORTED"}
)


def slm_outage_codes(warnings: Sequence[str]) -> list[str]:
    """The outage codes in ``warnings`` — an empty list means the SLM is alive."""
    codes: list[str] = []
    for warning in warnings:
        if not warning.startswith(f"{_SLM_UNAVAILABLE}:"):
            continue
        code = warning.split(":", 1)[1].strip()
        if code in _SLM_OUTAGE_CODES:
            codes.append(code)
    return codes


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class CorpusEvalError(RuntimeError):
    """Base for every failure this tool raises on purpose."""


class DocumentTimeout(CorpusEvalError):
    """One document exceeded ``--doc-timeout``; the sweep continues without it."""


class HarnessUnavailableError(CorpusEvalError):
    """``tools/eval_harness.py`` could not be imported — nothing can be scored."""


# --------------------------------------------------------------------------- #
# The harness, loaded as a module (tools/ is not a package)
# --------------------------------------------------------------------------- #

_HARNESS: ModuleType | None = None


def load_harness(path: Path | None = None) -> ModuleType:
    """Import ``tools/eval_harness.py`` and cache it.

    The split rule, the corpus loader and ``validate_prediction`` all come from
    here so the driver and the scorer cannot disagree about what a run covers or
    what a valid prediction looks like.
    """
    global _HARNESS
    if _HARNESS is not None and path is None:
        return _HARNESS
    target = path or (PROJECT_ROOT / "tools" / "eval_harness.py")
    spec = importlib.util.spec_from_file_location("eval_harness", target)
    if spec is None or spec.loader is None:  # pragma: no cover - unreachable
        raise HarnessUnavailableError(f"cannot load the eval harness from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    try:
        spec.loader.exec_module(module)
    except OSError as exc:
        raise HarnessUnavailableError(f"cannot load the eval harness from {target}: {exc}") from exc
    if path is None:
        _HARNESS = module
    return module


# --------------------------------------------------------------------------- #
# Configuration + per-document record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunConfig:
    """Everything that decides *what* the sweep covers and *how hard* it tries."""

    corpus_dir: Path
    out_dir: Path
    split: str = "dev"
    final: bool = False
    limit: int | None = None
    types: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    no_slm: bool = False
    dry_run: bool = False
    resume: bool = False
    doc_timeout_s: float = 180.0
    doc_grace_s: float = 180.0
    progress_every: int = 10
    slm_failures_before_fallback: int = 5

    def as_json(self) -> dict[str, Any]:
        """JSON-safe view for the manifest."""
        return {
            "corpus_dir": str(self.corpus_dir),
            "out_dir": str(self.out_dir),
            "split": self.split,
            "final": self.final,
            "limit": self.limit,
            "types": list(self.types),
            "profiles": list(self.profiles),
            "no_slm": self.no_slm,
            "dry_run": self.dry_run,
            "resume": self.resume,
            "doc_timeout_s": self.doc_timeout_s,
            "doc_grace_s": self.doc_grace_s,
            "progress_every": self.progress_every,
            "slm_failures_before_fallback": self.slm_failures_before_fallback,
        }


@dataclass
class DocRecord:
    """What the sweep observed about one document.  Manifest row, not a metric."""

    doc_id: str
    doc_type: str
    variant: int
    profile: str
    status: str = "ok"  # ok | failed | timeout | skipped
    wall_ms: float = 0.0
    pipeline_ms: float | None = None
    line_count: int = 0
    field_count: int = 0
    result_versions: int = 0
    escalated: bool = False
    shadow_sampled: bool = False
    quality_verdict: str | None = None
    slm_tokens: int = 0
    slm_model: str | None = None
    #: Provider-cache movement attributable to this document; :attr:`model_path`
    #: is the reading of it.  Recorded per document because a sweep re-run over a
    #: warm cache and a first sweep are different latency populations.
    cache_hits: int = 0
    cache_misses: int = 0
    warnings: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def as_json(self) -> dict[str, Any]:
        payload = {
            "doc_id": self.doc_id,
            "type": self.doc_type,
            "variant": self.variant,
            "profile": self.profile,
            "status": self.status,
            "wall_ms": round(self.wall_ms, 1),
            "pipeline_ms": None if self.pipeline_ms is None else round(self.pipeline_ms, 1),
            "line_count": self.line_count,
            "field_count": self.field_count,
            "result_versions": self.result_versions,
            "escalated": self.escalated,
            "shadow_sampled": self.shadow_sampled,
            "quality_verdict": self.quality_verdict,
            "slm_tokens": self.slm_tokens,
            "slm_model": self.slm_model,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "model_path": self.model_path,
            "warnings": list(self.warnings),
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
            payload["error_message"] = self.error_message
        return payload

    @property
    def model_path(self) -> str:
        """Which latency population this document belongs to.

        ``live`` — the provider cache missed, so a request went to the router
        and the wall clock includes it.  ``cached`` — every model lookup hit.
        ``none`` — no model lookup happened at all, which is the normal case for
        a bounced page and *not* the same thing as a cache hit.  Conflating the
        last two is how a p50 of 140 ms gets published as "warm cache latency"
        when it is really "the quality gate refused the page".
        """
        if self.cache_misses > 0:
            return "live"
        if self.cache_hits > 0:
            return "cached"
        return "none"


# --------------------------------------------------------------------------- #
# Provider wrappers
# --------------------------------------------------------------------------- #


class SLMDisabledProvider:
    """``--no-slm``: a provider with **no remote traffic at all**.

    Both remote methods of the composite are refused —
    :meth:`extract_fields`/:meth:`extract_fields_batch` (the SLM) and
    :meth:`ocr_page_heavy` (the Tier-2 VLM).  The orchestrator absorbs both as
    typed warnings (``slm_unavailable:…``, ``tier2_failed:…``), so the run still
    produces a full envelope built from the validators alone, which is what a
    "validators-only" number means.

    Explicit forwarding, like ``CompositeProvider`` itself: a typo becomes an
    ``AttributeError`` at review time rather than a silent fallthrough.
    """

    def __init__(self, inner: ModelProvider) -> None:
        self.inner = inner
        self.name = f"no_slm({getattr(inner, 'name', type(inner).__name__)})"

    def _refuse(self, method: str) -> Exception:
        from app.providers.base import ProviderMethodNotSupported

        return ProviderMethodNotSupported(
            method,
            provider=self.name,
            implemented_by=f"{type(self.inner).__name__} (run without --no-slm)",
            context={"tool": "run_corpus_eval", "reason": "--no-slm"},
        )

    # ---------------------------------------------------------- forwarded
    def detect_script(self, crops: Sequence[Any]) -> Any:
        return self.inner.detect_script(crops)

    def classify_page(self, page: Any, feats: Any) -> Any:
        return self.inner.classify_page(page, feats)

    def ocr_page(self, page: Any, script: str) -> Any:
        return self.inner.ocr_page(page, script)

    def identify_language(self, text: str, script: str) -> Any:
        return self.inner.identify_language(text, script)

    def translate(self, texts: Sequence[str], src: str, tgt: str = "eng") -> Any:
        return self.inner.translate(texts, src, tgt)

    # ------------------------------------------------------------- refused
    def ocr_page_heavy(self, page: Any) -> Any:
        raise self._refuse("ocr_page_heavy")

    def extract_fields(self, ctx: Any) -> Any:
        raise self._refuse("extract_fields")

    def extract_fields_batch(self, ctxs: Sequence[Any]) -> Any:
        raise self._refuse("extract_fields_batch")

    def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if callable(closer):
            closer()

    def __repr__(self) -> str:
        return f"SLMDisabledProvider(inner={type(self.inner).__name__})"


def remote_side(provider: object) -> Any | None:
    """The object holding ``slm_model``/``budget``, or ``None``.

    ``CompositeProvider.remote`` under the laptop profile; the provider itself
    for a bare ``HFAPIProvider``; ``None`` for anything local-only or wrapped by
    :class:`SLMDisabledProvider`.
    """
    if isinstance(provider, SLMDisabledProvider):
        return None
    candidate = getattr(provider, "remote", provider)
    return candidate if hasattr(candidate, "slm_model") else None


# --------------------------------------------------------------------------- #
# Per-document wall-clock budget
# --------------------------------------------------------------------------- #


class DocumentRunner:
    """Runs one callable at a time under a wall-clock budget.

    ``timeout_s <= 0`` disables the watchdog and calls inline, which is what the
    tests and ``--dry-run`` use; there is then no thread and no ambiguity about
    what raised.
    """

    def __init__(self, timeout_s: float, grace_s: float) -> None:
        self.timeout_s = timeout_s
        self.grace_s = max(0.0, grace_s)
        self.overlap_events = 0
        self._pool: ThreadPoolExecutor | None = None
        if timeout_s > 0:
            self._pool = self._new_pool()

    @staticmethod
    def _new_pool() -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus-eval")

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; raise :class:`DocumentTimeout` if it outlives the budget."""
        if self._pool is None:
            return fn()
        future = self._pool.submit(fn)
        try:
            return future.result(timeout=self.timeout_s)
        except FutureTimeoutError:
            # A Python thread cannot be killed. Abandon the pool, then wait out
            # the grace period so the next document does not contend with it.
            self._pool.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + self.grace_s
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.5)
            if not future.done():
                self.overlap_events += 1
            self._pool = self._new_pool()
            raise DocumentTimeout(
                f"document exceeded the {self.timeout_s:.0f}s wall-clock budget"
            ) from None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


# --------------------------------------------------------------------------- #
# Envelope -> prediction
# --------------------------------------------------------------------------- #


def document_quality(envelope: ResultEnvelope) -> dict[str, Any]:
    """The document-level quality block the harness scores.

    The harness carries one verdict per document; the envelope carries one per
    page.  The document verdict is the **worst** page verdict (a document with
    one unreadable page is not an ``accept``), and the score is that page's.
    With no pages at all — an ingest rejection — the bounce decision speaks for
    the document.
    """
    if not envelope.pages:
        decision = envelope.bounce_decision.decision
        verdict = decision.value if isinstance(decision, QualityVerdict) else str(decision)
        return {"verdict": verdict, "score": 0.0, "flags": []}
    worst = max(
        envelope.pages,
        key=lambda page: _VERDICT_SEVERITY.get(str(page.quality_verdict.value), 0),
    )
    flags: list[str] = []
    for page in envelope.pages:
        for flag in page.quality_flags:
            value = str(flag.value)
            if value not in flags:
                flags.append(value)
    return {
        "verdict": str(worst.quality_verdict.value),
        "score": float(worst.quality_score),
        "flags": flags,
    }


def _line_payload(line: Any) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": [float(v) for v in line.bbox],
        "confidence": float(line.confidence),
        "script": str(line.script),
        "language": None if line.language is None else str(line.language),
        "is_handwritten": bool(line.is_handwritten),
        "line_index": int(line.line_index),
    }


def _field_payload(field: Any) -> dict[str, Any]:
    evidence = field.evidence
    return {
        "value": field.value,
        "value_en": field.value_en,
        "normalized": field.normalized,
        "confidence": float(field.confidence),
        "source": str(field.source),
        "validator_passed": field.validator_passed,
        "evidence": {
            "page_no": int(evidence.page_no),
            "bbox": [float(v) for v in evidence.bbox],
            "span": [int(evidence.span.start), int(evidence.span.end)],
            "line_index": int(evidence.span.line_index),
        },
        "mt_engine": field.mt_engine,
        "flags": [str(flag.value) for flag in field.flags],
    }


def prediction_from_result(doc_id: str, result: Any) -> dict[str, Any]:
    """Turn one :class:`~app.pipeline.PipelineResult` into a harness prediction.

    Lines come from ``result.pages`` (the orchestrator's per-page working state)
    because :class:`~app.contracts.envelope.PageResult` deliberately carries no
    text — the envelope publishes counts and derived summaries, not the OCR.
    Everything else comes from the **final** envelope.
    """
    envelope: ResultEnvelope = result.final
    outcomes = tuple(getattr(result, "pages", ()) or ())

    if outcomes:
        pages = [
            {
                "page_no": int(outcome.page_no),
                "lines": [_line_payload(line) for line in outcome.page.lines],
            }
            for outcome in outcomes
        ]
    else:
        pages = [{"page_no": int(page.page_no), "lines": []} for page in envelope.pages]

    fields: dict[str, Any] = {}
    if envelope.document is not None:
        fields = {name: _field_payload(f) for name, f in envelope.document.fields.items()}

    # The harness's ``escalated`` means "stage s4_router fired Tier 2".  The
    # envelope's ``escalation_flag`` means something broader — assemble.py ORs
    # in "any field needs review" and "the grounding gate rejected something" —
    # so scoring it against a routing reference would count every
    # failed-checksum document as an escalation.  On this run that is the
    # difference between 39 documents (22%) and 22 (12%), i.e. between missing
    # the PRD escalation target and meeting it.  The router's own decision is
    # what gets published as ``escalated``; the envelope's broader flag is kept
    # beside it as ``needs_review``.
    routed = [
        outcome
        for outcome in outcomes
        if outcome.escalation is not None and outcome.escalation.escalate
    ]
    reasons = [outcome.escalation.detail for outcome in routed]
    escalated = bool(routed) if outcomes else bool(envelope.escalation_flag)

    timings = {
        str(key): float(value)
        for key, value in envelope.processing.stage_timings_ms.items()
    }
    if envelope.processing.duration_ms is not None:
        timings["e2e_total"] = float(envelope.processing.duration_ms)

    payload: dict[str, Any] = {
        "doc_id": doc_id,
        "result_version": int(envelope.result_version),
        "is_final": bool(envelope.is_final),
        "schema_version": str(envelope.schema_version),
        "quality": document_quality(envelope),
        "escalated": escalated,
        "escalation_reason": "; ".join(reasons) if reasons else None,
        "shadow_sampled": any(outcome.shadow_sampled for outcome in outcomes),
        "pages": pages,
        "fields": fields,
        "timings_ms": timings,
        # Beyond the harness contract; ignored by the scorer, read by the report.
        "warnings": list(envelope.warnings),
        #: The envelope's broader "a human should look at this" flag.
        "needs_review": bool(envelope.escalation_flag),
        #: Tier 2 actually landed and rewrote the page (not merely routed).
        "tier2_applied": any(outcome.escalated for outcome in outcomes),
        "result_versions": len(result.envelopes),
        "document_class": (
            None if envelope.document is None else str(envelope.document.document_class.value)
        ),
        "summary_span_count": (
            0
            if envelope.document is None or envelope.document.complaint_summary is None
            else len(envelope.document.complaint_summary.spans)
        ),
    }
    return payload


def error_prediction(doc_id: str, exc: BaseException, *, wall_ms: float) -> dict[str, Any]:
    """A prediction for a document the pipeline could not finish.

    ``pages: []`` and ``fields: {}`` are deliberate: they make the harness score
    every ground-truth line as a deletion and every ground-truth field as a
    miss, which is what "the service produced nothing for this page" costs.  No
    ``timings_ms`` — the pipeline reported no stage timings, and inventing a
    total from the failure's wall clock would put a timeout into the p95 of
    stages that never ran.  The wall clock is in the manifest instead.
    """
    tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
    return {
        "doc_id": doc_id,
        "pages": [],
        "fields": {},
        "error": {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc)[:800],
            "wall_ms": round(wall_ms, 1),
            "traceback_tail": tail,
        },
    }


# --------------------------------------------------------------------------- #
# Document selection
# --------------------------------------------------------------------------- #


def select_corpus_documents(config: RunConfig, harness: ModuleType | None = None) -> list[Any]:
    """The documents this sweep covers, in corpus order.

    Split selection is the harness's :func:`select_documents` verbatim — the
    ``LockedSplitError`` it raises without ``--final`` is the point.  ``--types``
    and ``--profiles`` filter *within* the split, and ``--limit`` truncates
    after filtering so a smoke run is a prefix of the real one.
    """
    mod = harness or load_harness()
    _index, docs = mod.load_corpus(config.corpus_dir)
    selected = mod.select_documents(docs, config.split, config.final)
    if config.types:
        wanted = set(config.types)
        selected = [d for d in selected if d.doc_type in wanted]
    if config.profiles:
        wanted = set(config.profiles)
        selected = [d for d in selected if d.profile in wanted]
    if config.limit is not None:
        selected = selected[: max(0, config.limit)]
    return selected


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


@dataclass
class SweepOutcome:
    """Everything the sweep produced, for the manifest and the caller."""

    records: list[DocRecord] = dc_field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    overlap_events: int = 0
    slm_model_used: str | None = None
    slm_model_switched_to: str | None = None
    slm_switch_reason: str | None = None
    cache_stats_before: dict[str, Any] = dc_field(default_factory=dict)
    cache_stats_after: dict[str, Any] = dc_field(default_factory=dict)
    cache_rows_before: int = 0
    cache_rows_after: int = 0
    schema_violations: dict[str, list[str]] = dc_field(default_factory=dict)

    @property
    def ok(self) -> int:
        return sum(1 for r in self.records if r.status == "ok")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.records if r.status in ("failed", "timeout"))


def _cache_stats_json(cache: ResponseCache | None) -> dict[str, Any]:
    if cache is None:
        return {}
    stats = cache.stats
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "stores": stats.stores,
        "invalidations": stats.invalidations,
        "lookups": stats.lookups,
        "hit_rate": round(stats.hit_rate, 4),
    }


def run_sweep(
    config: RunConfig,
    *,
    provider: ModelProvider,
    settings: Settings,
    documents: Sequence[Any],
    cache: ResponseCache | None = None,
    run_document: Callable[..., Any] | None = None,
    harness: ModuleType | None = None,
    on_progress: Callable[[int, int, DocRecord], None] | None = None,
) -> SweepOutcome:
    """Process ``documents`` one at a time, writing one prediction file each.

    ``run_document`` is injectable so the tests can drive the whole sweep — the
    watchdog, the failure containment, the schema self-check, the manifest — with
    a fake that never loads a weight.
    """
    mod = harness or load_harness()
    runner = run_document
    if runner is None:
        from app import pipeline as _pipeline

        runner = _pipeline.run_document

    config.out_dir.mkdir(parents=True, exist_ok=True)
    outcome = SweepOutcome(
        started_at=datetime.now(UTC).isoformat(),
        cache_stats_before=_cache_stats_json(cache),
        cache_rows_before=(cache.total_rows() if cache is not None else 0),
        slm_model_used=None if config.no_slm else settings.SLM_MODEL,
    )
    remote = remote_side(provider)
    doc_runner = DocumentRunner(config.doc_timeout_s, config.doc_grace_s)
    consecutive_slm_failures = 0
    wall_start = time.perf_counter()
    total = len(documents)

    try:
        for position, doc in enumerate(documents, start=1):
            record = DocRecord(
                doc_id=doc.doc_id,
                doc_type=doc.doc_type,
                variant=int(doc.variant),
                profile=doc.profile,
                slm_model=None if config.no_slm else getattr(remote, "slm_model", None),
            )
            target = config.out_dir / f"{doc.doc_id}.json"
            if config.resume and target.is_file():
                record.status = "skipped"
                outcome.records.append(record)
                if on_progress is not None:
                    on_progress(position, total, record)
                continue

            # A fresh budget per document. The HF provider's JobBudget is
            # per-instance and JOB_TOKEN_BUDGET is a *per-job* ceiling, so a
            # sweep that never resets it exhausts the budget after ~3 documents
            # and every later extraction dies as TokenBudgetExceeded.
            tokens_before = 0
            if remote is not None and hasattr(remote, "reset_budget"):
                remote.reset_budget(job_id=doc.doc_id)
            elif remote is not None and hasattr(remote, "budget"):
                tokens_before = int(remote.budget.used)
            cache_before = cache.stats if cache is not None else None

            started = time.perf_counter()
            payload: dict[str, Any]
            try:
                result = doc_runner.call(
                    lambda d=doc: runner(  # type: ignore[misc]
                        d.image_path,
                        provider=provider,
                        settings=settings,
                        job_id=d.doc_id,
                    )
                )
                record.wall_ms = (time.perf_counter() - started) * 1000.0
                payload = prediction_from_result(doc.doc_id, result)
                envelope = result.final
                record.status = "ok"
                record.pipeline_ms = envelope.processing.duration_ms
                record.line_count = sum(len(page["lines"]) for page in payload["pages"])
                record.field_count = len(payload["fields"])
                record.result_versions = len(result.envelopes)
                record.escalated = bool(payload["escalated"])
                record.shadow_sampled = bool(payload["shadow_sampled"])
                record.quality_verdict = payload["quality"]["verdict"]
                record.warnings = tuple(envelope.warnings)
            except DocumentTimeout as exc:
                record.wall_ms = (time.perf_counter() - started) * 1000.0
                record.status = "timeout"
                record.error_type = type(exc).__name__
                record.error_message = str(exc)[:400]
                payload = error_prediction(doc.doc_id, exc, wall_ms=record.wall_ms)
                _LOG.error("doc=%s TIMEOUT after %.1fs", doc.doc_id, record.wall_ms / 1000.0)
            except (PipelineError, OSError, ValueError, RuntimeError) as exc:
                record.wall_ms = (time.perf_counter() - started) * 1000.0
                record.status = "failed"
                record.error_type = type(exc).__name__
                record.error_message = str(exc)[:400]
                payload = error_prediction(doc.doc_id, exc, wall_ms=record.wall_ms)
                _LOG.error("doc=%s FAILED %s: %s", doc.doc_id, type(exc).__name__, exc)

            if remote is not None and hasattr(remote, "budget"):
                record.slm_tokens = max(0, int(remote.budget.used) - tokens_before)
            if cache is not None and cache_before is not None:
                now = cache.stats
                record.cache_hits = now.hits - cache_before.hits
                record.cache_misses = now.misses - cache_before.misses

            violations = mod.validate_prediction(payload, doc.doc_id)
            if violations:
                outcome.schema_violations[doc.doc_id] = violations[:6]

            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            outcome.records.append(record)

            # SLM health: the orchestrator absorbs a provider failure into a
            # machine-readable warning rather than raising, so the only place a
            # sustained outage shows up is here.  Only *outage* codes count —
            # see ``_SLM_OUTAGE_CODES``.
            if not config.no_slm and remote is not None:
                if slm_outage_codes(record.warnings):
                    consecutive_slm_failures += 1
                else:
                    consecutive_slm_failures = 0
                if (
                    consecutive_slm_failures >= config.slm_failures_before_fallback
                    and outcome.slm_model_switched_to is None
                    and settings.SLM_MODEL_FALLBACK != getattr(remote, "slm_model", None)
                ):
                    previous = getattr(remote, "slm_model", None)
                    remote.slm_model = settings.SLM_MODEL_FALLBACK
                    outcome.slm_model_switched_to = settings.SLM_MODEL_FALLBACK
                    outcome.slm_switch_reason = (
                        f"{consecutive_slm_failures} consecutive documents reported an "
                        f"SLM outage code on {previous}"
                    )
                    consecutive_slm_failures = 0
                    _LOG.warning(
                        "SLM fallback engaged: %s -> %s", previous, settings.SLM_MODEL_FALLBACK
                    )

            if on_progress is not None:
                on_progress(position, total, record)
    finally:
        doc_runner.close()

    outcome.overlap_events = doc_runner.overlap_events
    outcome.duration_s = time.perf_counter() - wall_start
    outcome.finished_at = datetime.now(UTC).isoformat()
    outcome.cache_stats_after = _cache_stats_json(cache)
    outcome.cache_rows_after = cache.total_rows() if cache is not None else 0
    return outcome


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _warning_key(warning: str) -> str:
    """Group key for a pipeline warning.

    ``slm_unavailable:<CODE>`` keeps its code — the difference between "the
    endpoint was down" and "the model answered nonsense" is the whole finding.
    ``tier2_failed:page=3:reason=…`` keeps only its prefix, because the page
    number would give every occurrence its own bucket.
    """
    if warning.startswith(f"{_SLM_UNAVAILABLE}:"):
        return ":".join(warning.split(":", 2)[:2])
    return warning.split(":", 1)[0]


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    return completed.stdout.strip() or None


def _model_files(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("*")):
        if not path.is_file() or path.name in ("README.md", "PROVENANCE.json"):
            continue
        rows.append(
            {
                "path": path.relative_to(models_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _settings_snapshot(settings: Settings) -> dict[str, Any]:
    """Every knob that can move a number, and **never** a secret."""
    keys = (
        "PROVIDER_PROFILE",
        "SYNTHETIC_ONLY",
        "HF_BASE_URL",
        "SLM_MODEL",
        "SLM_MODEL_FALLBACK",
        "SLM_MAX_TOKENS",
        "SLM_TEMPERATURE",
        "JOB_TOKEN_BUDGET",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ORT_INTRA_OP_THREADS",
        "OCR_BATCH_SIZE",
        "REC_HEAD_CACHE_SIZE",
        "ESCALATION_SHADOW_RATE",
        "CODE_MIX_THRESHOLD",
        "QUALITY_BOUNCE_THRESHOLD",
        "QUALITY_BLUR_LAPLACIAN_FLOOR",
        "QUALITY_CONTRAST_FLOOR",
        "QUALITY_EFFECTIVE_DPI_FLOOR",
        "QUALITY_TEXT_COVERAGE_FLOOR",
        "QUALITY_SKEW_REPAIR_DEG",
        "QUALITY_SKEW_BOUNCE_DEG",
        "GROUNDING_MAX_EDIT_DISTANCE",
        "MAX_WALL_CLOCK_SECONDS",
        "IFSC_MASTER_PATH",
    )
    snapshot: dict[str, Any] = {}
    for key in keys:
        value = getattr(settings, key, None)
        if isinstance(value, Path):
            snapshot[key] = str(value)
        elif hasattr(value, "value"):  # StrEnum
            snapshot[key] = value.value
        else:
            snapshot[key] = value
    return snapshot


def build_manifest(
    config: RunConfig,
    outcome: SweepOutcome,
    *,
    settings: Settings,
    corpus_index: dict[str, Any],
    documents_selected: int,
) -> dict[str, Any]:
    """The run's provenance: what ran, against what, with which weights.

    The HF token is represented only by ``hf_token_present``; neither its value
    nor its length appears anywhere in this document.
    """
    verdicts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for record in outcome.records:
        statuses[record.status] = statuses.get(record.status, 0) + 1
        if record.quality_verdict:
            verdicts[record.quality_verdict] = verdicts.get(record.quality_verdict, 0) + 1
        for warning in record.warnings:
            key = _warning_key(warning)
            warning_counts[key] = warning_counts.get(key, 0) + 1

    harness = load_harness()
    ok_records = [r for r in outcome.records if r.status == "ok"]

    def _latency(rows: Sequence[DocRecord]) -> dict[str, Any]:
        values = sorted(r.wall_ms for r in rows)
        if not values:
            return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
        return {
            "n": len(values),
            "min": round(values[0], 1),
            "p50": round(harness.percentile(values, 0.50), 1),
            "p95": round(harness.percentile(values, 0.95), 1),
            "max": round(values[-1], 1),
            "mean": round(sum(values) / len(values), 1),
        }

    return {
        "tool": "tools/run_corpus_eval.py",
        "tool_version": TOOL_VERSION,
        "harness_version": getattr(harness, "HARNESS_VERSION", None),
        "git_sha": _git_sha(),
        "run": {
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at,
            "duration_s": round(outcome.duration_s, 1),
            "documents_selected": documents_selected,
            "documents_written": len(outcome.records),
            "documents_ok": outcome.ok,
            "documents_failed": outcome.failed,
            "status_counts": statuses,
            "quality_verdict_counts": verdicts,
            "warning_counts": warning_counts,
            "overlap_events": outcome.overlap_events,
            "wall_ms_per_doc": _latency(ok_records),
            "wall_ms_per_doc_by_model_path": {
                path: _latency([r for r in ok_records if r.model_path == path])
                for path in ("live", "cached", "none")
            },
            "slm_tokens_total": sum(r.slm_tokens for r in outcome.records),
        },
        "config": config.as_json(),
        "models": {
            "slm": outcome.slm_model_used,
            "slm_fallback_configured": settings.SLM_MODEL_FALLBACK,
            "slm_fallback_engaged": outcome.slm_model_switched_to,
            "slm_fallback_reason": outcome.slm_switch_reason,
            "hf_base_url": settings.HF_BASE_URL,
            "hf_token_present": settings.HF_TOKEN is not None,
            "models_dir": str(settings.MODELS_DIR),
            "files": _model_files(Path(settings.MODELS_DIR)),
            "provenance": _read_provenance(Path(settings.MODELS_DIR)),
        },
        "cache": {
            "path": str(settings.provider_cache_path),
            "enabled": True,
            "rows_before": outcome.cache_rows_before,
            "rows_after": outcome.cache_rows_after,
            "stats_before": outcome.cache_stats_before,
            "stats_after": outcome.cache_stats_after,
        },
        "corpus": {
            "dir": str(config.corpus_dir),
            "seed": corpus_index.get("seed"),
            "generator_version": corpus_index.get("generator_version"),
            "hand_shaping": corpus_index.get("hand_shaping"),
            "count": corpus_index.get("count"),
            "caveat": corpus_index.get("caveat"),
            "caveats": corpus_index.get("caveats"),
        },
        "settings": _settings_snapshot(settings),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "schema_violations": outcome.schema_violations,
        "documents": [record.as_json() for record in outcome.records],
    }


def _read_provenance(models_dir: Path) -> dict[str, Any] | None:
    path = models_dir / "PROVENANCE.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run the real pipeline over the corpus and emit eval-harness predictions.",
    )
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "corpus")
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "eval" / "predictions",
        help="directory for <doc_id>.json prediction files",
    )
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument(
        "--final",
        action="store_true",
        help="unlock the held-out variants 3-4; passed through to the harness",
    )
    ap.add_argument("--limit", type=int, default=None, help="smoke run: first N documents")
    ap.add_argument("--types", nargs="*", default=None, help="restrict to these document types")
    ap.add_argument("--profiles", nargs="*", default=None, help="restrict to these profiles")
    ap.add_argument(
        "--no-slm",
        action="store_true",
        help="validators only: refuse every remote provider method (SLM and Tier-2 VLM)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="select and report the documents, load nothing, write nothing",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip documents that already have a prediction file",
    )
    ap.add_argument("--doc-timeout", type=float, default=180.0, help="per-document seconds; <=0 disables")
    ap.add_argument("--doc-grace", type=float, default=180.0, help="seconds to wait out an abandoned document")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "docs" / "eval" / "PREDICTIONS_MANIFEST.json",
    )
    ap.add_argument(
        "--score",
        action="store_true",
        help="run the harness over the predictions when the sweep finishes",
    )
    ap.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "eval" / "report.txt")
    ap.add_argument("--json-out", type=Path, default=PROJECT_ROOT / "docs" / "eval" / "report.json")
    ap.add_argument("--log-level", default="INFO")
    return ap


def _progress_logger(config: RunConfig, total: int, started: float) -> Callable[..., None]:
    def report(position: int, _total: int, record: DocRecord) -> None:
        first_or_last = position == 1 or position == total
        if not first_or_last and position % max(1, config.progress_every) != 0:
            return
        elapsed = time.perf_counter() - started
        rate = elapsed / position if position else 0.0
        eta = rate * (total - position)
        _LOG.info(
            "[%d/%d] %s status=%s %.1fs  elapsed=%.0fs eta=%.0fs",
            position,
            total,
            record.doc_id,
            record.status,
            record.wall_ms / 1000.0,
            elapsed,
            eta,
        )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    config = RunConfig(
        corpus_dir=args.corpus,
        out_dir=args.out,
        split=args.split,
        final=bool(args.final),
        limit=args.limit,
        types=tuple(args.types or ()),
        profiles=tuple(args.profiles or ()),
        no_slm=bool(args.no_slm),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        doc_timeout_s=float(args.doc_timeout),
        doc_grace_s=float(args.doc_grace),
        progress_every=int(args.progress_every),
    )

    harness = load_harness()
    try:
        corpus_index, _all_docs = harness.load_corpus(config.corpus_dir)
        documents = select_corpus_documents(config, harness)
    except harness.EvalHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    _LOG.info(
        "selected %d documents (split=%s final=%s types=%s profiles=%s limit=%s)",
        len(documents),
        config.split,
        config.final,
        list(config.types) or "all",
        list(config.profiles) or "all",
        config.limit,
    )
    if config.dry_run:
        for doc in documents:
            print(f"{doc.doc_id}\t{doc.doc_type}\tv{doc.variant}\t{doc.profile}")
        print(f"{len(documents)} documents would run")
        return 0
    if not documents:
        print("error: no documents selected", file=sys.stderr)
        return 3

    settings = get_settings()
    cache = ResponseCache(settings.provider_cache_path)
    provider: Any = None
    try:
        from app.providers.factory import get_provider

        provider = get_provider(settings, cache=cache, log_banner=True)
        if config.no_slm:
            provider = SLMDisabledProvider(provider)
        _quiet_onnxruntime()

        started = time.perf_counter()
        outcome = run_sweep(
            config,
            provider=provider,
            settings=settings,
            documents=documents,
            cache=cache,
            harness=harness,
            on_progress=_progress_logger(config, len(documents), started),
        )
    finally:
        if provider is not None:
            from app.providers.factory import close_provider

            close_provider(provider)
        cache.close()

    manifest = build_manifest(
        config,
        outcome,
        settings=settings,
        corpus_index=corpus_index,
        documents_selected=len(documents),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _LOG.info(
        "sweep done: %d ok, %d failed, %.0fs, cache hit rate %.3f -> manifest %s",
        outcome.ok,
        outcome.failed,
        outcome.duration_s,
        outcome.cache_stats_after.get("hit_rate", 0.0),
        args.manifest,
    )

    if not args.score:
        return 0 if outcome.failed == 0 else 1

    try:
        report = harness.evaluate(
            config.corpus_dir,
            config.out_dir,
            split=config.split,
            final=config.final,
            types=list(config.types) or None,
            profiles=list(config.profiles) or None,
        )
    except harness.EvalHarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    rendered = harness.render_report(report)
    print(rendered)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if outcome.failed == 0 else 1


def _quiet_onnxruntime() -> None:
    """Silence ORT's per-initializer graph-pruning chatter.

    Several hundred WARNING lines per session start would otherwise bury the
    progress log.  Severity 3 = ERROR; a real load failure still shouts.
    """
    try:
        import onnxruntime  # type: ignore[import-untyped]

        onnxruntime.set_default_logger_severity(3)
    except (ImportError, AttributeError):  # pragma: no cover - optional
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
