"""The four endpoints.

``POST /v1/documents``
    Multipart upload, synchronous pipeline, envelope back.  Idempotent on the
    sha256 of the bytes: the same attachment returns the same job with
    ``is_new=false`` and the envelope that was already produced, and the
    pipeline does not run a second time.

``GET /v1/jobs/{job_id}``
    The job record plus its latest envelope and the list of versions published.

``GET /healthz``
    Profile banner, models present/absent, the provider gauge, database
    reachability.  The thing an air-gapped operator reads first.

``GET /metrics``
    Prometheus text exposition over the default registry — every counter the
    stages, the provider factory, the pipeline and the dispatcher already
    register.

The synthetic-data declaration deserves a note.  Under a remote profile the
pipeline refuses bytes that are not declared synthetic (PLAN.md invariant 6): a
file on disk carries its provenance in a ``.gt.json`` sidecar, an upload carries
none, so the submitter must assert it in the form.  It is a positive
declaration in the same shape as the sidecar, not a bypass — there is no value
of ``synthetic`` that lets an *undeclared* payload through, and under
``cpu_node`` the field is ignored because that node is allowed real complaints.

This module does NOT handle: authentication, the review UI, or any business
rule — every decision it makes is "which stored thing answers this request".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from app.api.schemas import (
    HealthResponse,
    JobResponse,
    ModelArtifactReport,
    RouteStats,
    StatsResponse,
    SubmissionResponse,
)
from app.contracts.envelope import ResultEnvelope
from app.contracts.errors import DocumentRejected, PipelineError, ProviderError
from app.pipeline import run_document
from app.providers.factory import PROVIDER_PROFILE_GAUGE
from app.settings import Settings
from app.storage.models import Job, JobStatus
from app.storage.repository import JobStore

__all__ = ["build_router", "model_artifact_report"]

_LOG = logging.getLogger(__name__)

#: Weights ``/healthz`` reports on, relative to ``MODELS_DIR``.  Every one is
#: optional at *startup* — the service ingests, bounces and validates without
#: them — so their absence is ``degraded``, never a failed health check.
_MODEL_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("ocr_detection", "ocr/det.onnx"),
    ("ocr_rec_latin", "ocr/rec_latn.onnx"),
    ("ocr_rec_devanagari", "ocr/rec_deva.onnx"),
    ("language_id", "lid/lid.176.ftz"),
    ("translation", "mt/indic-en/model.bin"),
    ("page_classifier_fusion", "classifier/fusion.json"),
)


def _state(request: Request) -> Any:
    return request.app.state


def _store(request: Request) -> JobStore:
    return _state(request).store


def _settings(request: Request) -> Settings:
    return _state(request).settings


def build_router() -> APIRouter:
    """Assemble the router.  A function so each app gets its own instance."""
    router = APIRouter()

    @router.post(
        "/v1/documents",
        response_model=SubmissionResponse,
        summary="Submit one document and get its result envelope",
    )
    async def submit_document(
        request: Request,
        file: Annotated[UploadFile, File(description="The attachment.")],
        synthetic: Annotated[
            bool,
            Form(
                description=(
                    "Declare the payload synthetic. Required under the laptop "
                    "profile (PLAN.md invariant 6); ignored on cpu_node."
                )
            ),
        ] = False,
        sender: Annotated[str | None, Form()] = None,
        subject: Annotated[str | None, Form()] = None,
        requested_fields: Annotated[
            str,
            Form(
                max_length=4096,
                description="Comma-separated field names to extract; additional discoveries are returned as other_fields.",
            ),
        ] = "",
        store: Annotated[JobStore, Depends(_store)] = None,  # type: ignore[assignment]
        settings: Annotated[Settings, Depends(_settings)] = None,  # type: ignore[assignment]
    ) -> SubmissionResponse:
        """Ingest, process and store one document.

        Raises:
            PipelineError: refused upload (too large), residency violation,
                malware, or a provider failure the pipeline did not absorb.
                Each is mapped to its own status by the app's handlers.
        """
        from fastapi.exceptions import RequestValidationError

        from app.field_selection import parse_requested_fields

        try:
            wanted = parse_requested_fields(requested_fields)
        except ValueError as failure:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("body", "requested_fields"),
                        "msg": str(failure),
                        "input": requested_fields,
                    }
                ]
            ) from failure
        data = await _read_upload(file, settings)
        state = _state(request)

        from app.stages.s1_ingest import idempotency_key

        document_id = idempotency_key(data)
        record = store.create_or_get_job(
            document_id=document_id,
            filename=file.filename,
            byte_size=len(data),
            content_type=file.content_type,
            sender=sender,
            subject=subject,
            provider_profile=settings.PROVIDER_PROFILE.value,
        )
        previous = store.latest_envelope(record.job.id) if not record.is_new else None
        if not record.is_new and record.job.status in {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        }:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=409,
                detail="An extraction for this document is already running. Retry after it completes.",
            )
        if (
            record.job.status != JobStatus.FAILED.value
            and previous is not None
            and previous.requested_fields == wanted
        ):
            _LOG.info(
                "idempotent resubmission | job=%s version=%d",
                record.job.id,
                record.job.latest_result_version,
            )
            return _submission(record.job, previous, False)

        job_id = record.job.id
        store.set_status(job_id, JobStatus.RUNNING)
        published: list[ResultEnvelope] = []

        def on_envelope(envelope: ResultEnvelope) -> None:
            store.record_result(job_id, envelope)
            published.append(envelope)
            state.dispatcher.publish(job_id, envelope)

        try:
            run_document(
                data,
                provider=state.provider,
                settings=settings,
                job_id=job_id,
                filename=file.filename,
                synthetic_declared=synthetic,
                requested_fields=wanted,
                starting_result_version=record.job.latest_result_version + 1,
                on_envelope=on_envelope,
            )
        except (ProviderError, DocumentRejected) as failure:
            store.set_status(
                job_id,
                JobStatus.FAILED,
                error_code=failure.code,
                error_message=failure.message,
            )
            raise
        except PipelineError as failure:
            store.set_status(
                job_id,
                JobStatus.FAILED,
                error_code=failure.code,
                error_message=failure.message,
            )
            raise

        job = store.get_job(job_id)
        return _submission(job, published[-1] if published else None, record.is_new)

    @router.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse,
        summary="The job record and its latest envelope",
    )
    async def get_job(
        job_id: str,
        store: Annotated[JobStore, Depends(_store)] = None,  # type: ignore[assignment]
    ) -> JobResponse:
        """Look one job up.

        Raises:
            JobNotFound: mapped to 404 by the application's handler.
        """
        job = store.get_job(job_id)
        return JobResponse(
            job_id=job.id,
            document_id=job.document_id,
            status=job.status,
            result_version=job.latest_result_version,
            is_final=job.is_final,
            created_at=job.created_at,
            updated_at=job.updated_at,
            soft_duplicate_of=job.soft_duplicate_of,
            soft_duplicate_reason=job.soft_duplicate_reason,
            error_code=job.error_code,
            versions=tuple(result.result_version for result in job.results),
            result=store.latest_envelope(job.id),
        )

    @router.get("/healthz", response_model=HealthResponse, summary="Liveness")
    async def healthz(request: Request) -> HealthResponse:
        """Profile banner, model inventory, provider gauge, database reachability."""
        settings = _settings(request)
        artifacts = model_artifact_report(settings)
        database = _database_state(_store(request))
        healthy = database == "ok"
        return HealthResponse(
            status="ok"
            if healthy and all(a.present for a in artifacts)
            else "degraded",
            profile=settings.PROVIDER_PROFILE.value,
            synthetic_only=settings.SYNTHETIC_ONLY,
            banner=settings.startup_banner(),
            provider_profile_gauge=_gauge_values(),
            models=artifacts,
            database=database,
            webhook_enabled=settings.webhook_enabled,
            schema_version=ResultEnvelope.model_fields["schema_version"].default,
        )

    @router.get("/metrics", summary="Prometheus exposition")
    async def metrics() -> Response:
        """The default registry, in text exposition format.

        Unauthenticated on purpose: every series here is a counter or a gauge
        about the *service*, never about a document. If a metric ever needs a
        document identifier as a label, it is the metric that is wrong.
        """
        return Response(
            content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST
        )

    @router.get("/v1/stats", response_model=StatsResponse, summary="Error rates")
    async def stats() -> StatsResponse:
        """Error rates and throughput since this process started.

        Reads the same in-memory counters ``/metrics`` exposes, arranged for a
        human rather than a scraper. Nothing is persisted: a restart zeroes
        every number here, which is why ``uptime_seconds`` ships with them.

        Carries no document identifier, for the reason given on ``/metrics``.
        """
        return build_stats()

    return router


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

#: Process start, so a rate can be read against the window that produced it.
_STARTED_AT: Final[float] = time.monotonic()


def _samples(metric_name: str) -> list[tuple[dict[str, str], float]]:
    """``(labels, value)`` for one counter in the default registry.

    Reads the registry rather than the module-level counter objects so this
    keeps working if a metric moves modules, and so it cannot accidentally
    hold a reference that keeps a collector alive.
    """
    collected: list[tuple[dict[str, str], float]] = []
    for family in REGISTRY.collect():
        if family.name != metric_name:
            continue
        for sample in family.samples:
            if sample.name.endswith("_total"):
                collected.append((dict(sample.labels), float(sample.value)))
    return collected


def _by_label(metric_name: str, label: str) -> dict[str, int]:
    """Sum one counter's values, grouped by a single label."""
    totals: dict[str, int] = {}
    for labels, value in _samples(metric_name):
        key = labels.get(label)
        if key is not None:
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def build_stats() -> StatsResponse:
    """Assemble :class:`StatsResponse` from the default registry.

    A free function so the reporting can be unit-tested without an HTTP client
    and reused by an operator script.
    """
    per_route: dict[str, dict[str, int]] = {}
    for labels, value in _samples("http_requests"):
        route = labels.get("route", "<unmatched>")
        status = labels.get("status", "")
        bucket = per_route.setdefault(
            route, {"requests": 0, "client_errors": 0, "server_errors": 0}
        )
        count = int(value)
        bucket["requests"] += count
        if status.startswith("4"):
            bucket["client_errors"] += count
        elif status.startswith("5"):
            bucket["server_errors"] += count

    routes = tuple(
        RouteStats(
            route=route,
            requests=counts["requests"],
            client_errors=counts["client_errors"],
            server_errors=counts["server_errors"],
            error_rate=_rate(
                counts["client_errors"] + counts["server_errors"],
                counts["requests"],
            ),
        )
        for route, counts in sorted(per_route.items())
    )
    requests = sum(entry.requests for entry in routes)
    client_errors = sum(entry.client_errors for entry in routes)
    server_errors = sum(entry.server_errors for entry in routes)
    unhandled = sum(
        int(value) for _labels, value in _samples("http_unhandled_exceptions")
    )

    return StatsResponse(
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 3),
        requests=requests,
        client_errors=client_errors,
        server_errors=server_errors,
        unhandled_exceptions=unhandled,
        error_rate=_rate(client_errors + server_errors, requests),
        routes=routes,
        documents=_by_label("pipeline_documents", "outcome"),
        bounces=_by_label("quality_gate_bounces", "reason"),
        provider_calls=_by_label("provider_api_requests", "outcome"),
    )


def _rate(errors: int, total: int) -> float:
    """``errors / total``, and 0.0 rather than a division by zero when idle."""
    return round(errors / total, 6) if total else 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    """Read the upload, refusing anything over ``API_MAX_UPLOAD_BYTES``.

    The cap is checked while reading rather than from ``Content-Length``, which
    a client controls, and it is separate from ``MAX_TOTAL_PIXELS``: this one
    bounds what we buffer, that one bounds what a valid file may decode to.

    Raises:
        PipelineError: the upload exceeds the cap.
    """
    limit = settings.API_MAX_UPLOAD_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _TooLarge(
                f"upload exceeds API_MAX_UPLOAD_BYTES ({limit} bytes)",
                context={"limit": limit},
            )
        chunks.append(chunk)
    return b"".join(chunks)


class _TooLarge(PipelineError):
    """The upload is bigger than this service will buffer."""

    code = "UPLOAD_TOO_LARGE"
    http_status = 413


def _submission(
    job: Job, envelope: ResultEnvelope | None, is_new: bool
) -> SubmissionResponse:
    return SubmissionResponse(
        job_id=job.id,
        document_id=job.document_id,
        is_new=is_new,
        status=job.status,
        soft_duplicate_of=job.soft_duplicate_of,
        soft_duplicate_reason=job.soft_duplicate_reason,
        result=envelope,
    )


def model_artifact_report(settings: Settings) -> tuple[ModelArtifactReport, ...]:
    """Which weights are on this node.

    Reported rather than required: an empty ``models/`` is a supported state
    (the suite runs that way), and on an air-gapped node this list is the first
    thing that tells an operator which volume did not mount.
    """
    root: Path = settings.models_dir
    return tuple(
        ModelArtifactReport(
            name=name, path=str(root / relative), present=(root / relative).exists()
        )
        for name, relative in _MODEL_ARTIFACTS
    )


def _gauge_values() -> dict[str, float]:
    """Current ``provider_profile`` gauge, read back from the registry."""
    values: dict[str, float] = {}
    for metric in PROVIDER_PROFILE_GAUGE.collect():
        for sample in metric.samples:
            profile = sample.labels.get("profile")
            if profile is not None:
                values[profile] = sample.value
    return values


def _database_state(store: JobStore) -> str:
    """``ok`` or a short reason — never the DSN, which may carry a password."""
    try:
        store.counts()
    except PipelineError as failure:
        return failure.code
    except Exception as failure:  # noqa: BLE001 - health must not raise
        return type(failure).__name__
    return "ok"
