"""The four endpoints.

``POST /v1/documents``
    Multipart upload and synchronous extraction. A configured request
    deduplicates on the attachment, realm and resolved extraction plan and
    returns a compact view. Legacy uploads return the stored envelope.

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

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Final

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import ValidationError

from app.api.schemas import (
    ExtractedValue,
    ExtractionConfiguration,
    ExtractionModel,
    ExtractionResponse,
    ExtractionValidation,
    HealthResponse,
    JobResponse,
    ModelArtifactReport,
    RouteStats,
    StatsResponse,
    SubmissionResponse,
)
from app.contracts.envelope import QualityVerdict, ResultEnvelope
from app.contracts.errors import (
    PipelineError,
    ResidencyViolation,
)
from app.pipeline import run_document
from app.providers.base import ModelProvider
from app.providers.factory import PROVIDER_PROFILE_GAUGE, get_provider
from app.settings import ProviderProfile, Settings
from app.storage.models import Job, JobStatus
from app.storage.repository import JobStore

__all__ = ["build_router", "model_artifact_report"]

_LOG = logging.getLogger(__name__)
_CONFIGURED_JOB_PREFIX = "configured:"
_CONFIGURED_ALL_PREFIX = "configured-all:"

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
        response_model=SubmissionResponse | ExtractionResponse,
        summary="Submit one document and get its result envelope",
    )
    async def submit_document(
        request: Request,
        file: Annotated[UploadFile, File(description="The attachment.")],
        realm: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        configurations: Annotated[
            str | None,
            Form(
                max_length=16384,
                description="JSON with groups, override_model, fields_to_extract, "
                "other_fields, reactions and file_size.",
            ),
        ] = None,
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
    ) -> SubmissionResponse | ExtractionResponse:
        """Ingest, process and store one document.

        Raises:
            PipelineError: refused upload (too large), residency violation,
                malware, or a provider failure the pipeline did not absorb.
                Each is mapped to its own status by the app's handlers.
        """
        from fastapi.exceptions import RequestValidationError

        from app.field_selection import EXTRACTION_REVISION, parse_requested_fields

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
        config: ExtractionConfiguration | None = None
        plans: dict[ExtractionModel | None, tuple[str, ...]] = {None: wanted}
        if realm is not None or configurations is not None:
            if not realm or not realm.strip() or configurations is None:
                raise HTTPException(422, "realm and configurations are required together")
            if wanted:
                raise HTTPException(422, "Use configurations.fields_to_extract for this request")
            config, plans = _resolve_configuration(configurations, settings)
            wanted = parse_requested_fields([name for fields in plans.values() for name in fields])
        engines = {model: _model_settings(settings, model) for model in plans}
        primary = next(iter(plans))
        if config is not None and not synthetic and any(
            engine.is_remote_profile for engine in engines.values()
        ):
            raise ResidencyViolation("The laptop profile requires synthetic=true")
        data = await _read_upload(file, settings, file_size=config.file_size if config else None)
        state = _state(request)

        from app.stages.s1_ingest import idempotency_key

        document_id = idempotency_key(data)
        job_key = document_id
        if config is not None and realm is not None:
            identity = json.dumps(
                [
                    realm.strip(), document_id, list(plans.items()),
                    config.other_fields, config.reactions, EXTRACTION_REVISION,
                    settings.PROVIDER_PROFILE.value, settings.VLM_MODEL, settings.SLM_MODEL,
                    settings.GPU_HANDWRITING_ENABLED, settings.GPU_HANDWRITING_MIN_RATIO,
                ],
                ensure_ascii=False,
            )
            prefix = _CONFIGURED_ALL_PREFIX if config.other_fields else _CONFIGURED_JOB_PREFIX
            job_key = prefix + hashlib.sha256(identity.encode()).hexdigest()
        record = store.create_or_get_job(
            document_id=job_key,
            filename=file.filename,
            byte_size=len(data),
            content_type=file.content_type,
            sender=sender,
            subject=subject,
            provider_profile=engines[primary].PROVIDER_PROFILE.value,
        )
        previous = store.latest_envelope(record.job.id) if not record.is_new else None
        if not record.is_new and record.job.status in {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        }:
            raise HTTPException(
                status_code=409,
                detail="An extraction for this document is already running. Retry after it completes.",
            )
        if (
            record.job.status != JobStatus.FAILED.value
            and previous is not None
            and previous.requested_fields == wanted
            and previous.extraction_revision == EXTRACTION_REVISION
        ):
            _LOG.info(
                "idempotent resubmission | job=%s version=%d",
                record.job.id,
                record.job.latest_result_version,
            )
            if config is not None:
                return _extraction_response(record.job, previous)
            return _submission(record.job, previous, False)

        job_id = record.job.id
        if config is None:
            store.set_status(job_id, JobStatus.RUNNING)
        elif not store.try_start_extraction(record.job):
            raise HTTPException(409, "An extraction for this document has already started")
        published: list[ResultEnvelope] = []

        def on_envelope(envelope: ResultEnvelope) -> None:
            store.record_result(job_id, envelope)
            published.append(envelope)
            if config is None or "webhook" in config.reactions:
                state.dispatcher.publish(job_id, envelope)

        try:
            results = []
            for model, fields in plans.items():
                active = engines[model]
                provider = _extraction_provider(state, settings, active)
                result = run_document(
                    data,
                    provider=provider,
                    settings=active,
                    job_id=job_id,
                    filename=file.filename,
                    synthetic_declared=synthetic,
                    requested_fields=fields,
                    other_fields=(
                        config.other_fields and model == primary if config is not None else None
                    ),
                    recognition_model=model,
                    starting_result_version=record.job.latest_result_version + 1,
                    on_envelope=on_envelope if len(plans) == 1 else None,
                )
                results.append(result.final)
            if len(results) > 1:
                on_envelope(_merge_extractions(
                    results, tuple(plans.values()), wanted, record.job.latest_result_version + 1
                ))
        except PipelineError as failure:
            store.set_status(
                job_id,
                JobStatus.FAILED,
                error_code=failure.code,
                error_message=failure.message,
            )
            raise
        except Exception:
            store.set_status(job_id, JobStatus.FAILED, error_code="EXTRACTION_FAILED")
            raise

        envelope = published[-1] if published else None
        if config is not None:
            return _extraction_response(record.job, envelope)
        return _submission(store.get_job(job_id), envelope, record.is_new)

    @router.get(
        "/v1/jobs/{job_id}",
        response_model=JobResponse | ExtractionResponse,
        summary="The job record and its latest envelope",
    )
    async def get_job(
        job_id: str,
        store: Annotated[JobStore, Depends(_store)] = None,  # type: ignore[assignment]
    ) -> JobResponse | ExtractionResponse:
        """Look one job up.

        Raises:
            JobNotFound: mapped to 404 by the application's handler.
        """
        job = store.get_job(job_id)
        envelope = store.latest_envelope(job.id)
        if job.document_id.startswith((_CONFIGURED_JOB_PREFIX, _CONFIGURED_ALL_PREFIX)):
            return _extraction_response(job, envelope)
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
            result=envelope,
        )

    @router.get("/v1/field-groups", summary="Available extraction field groups")
    def list_field_groups(
        settings: Annotated[Settings, Depends(_settings)],
    ) -> dict[str, tuple[str, ...]]:
        return settings.API_FIELD_GROUPS

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


async def _read_upload(
    file: UploadFile, settings: Settings, *, file_size: int | None = None,
) -> bytes:
    """Read the upload, refusing anything over ``API_MAX_UPLOAD_BYTES``.

    The cap is checked while reading rather than from ``Content-Length``, which
    a client controls, and it is separate from ``MAX_TOTAL_PIXELS``: this one
    bounds what we buffer, that one bounds what a valid file may decode to.

    Raises:
        PipelineError: the upload exceeds the cap.
    """
    limit = min(settings.API_MAX_UPLOAD_BYTES, file_size or settings.API_MAX_UPLOAD_BYTES)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise _TooLarge(
                f"upload exceeds the allowed file size ({limit} bytes)",
                context={"limit": limit},
            )
        chunks.append(chunk)
    return b"".join(chunks)


class _TooLarge(PipelineError):
    """The upload is bigger than this service will buffer."""

    code = "UPLOAD_TOO_LARGE"
    http_status = 413


def _resolve_configuration(
    value: str, settings: Settings,
) -> tuple[ExtractionConfiguration, dict[ExtractionModel | None, tuple[str, ...]]]:
    from app.field_selection import parse_requested_fields

    try:
        config = ExtractionConfiguration.model_validate_json(value)
    except ValidationError as failure:
        raise HTTPException(
            422, detail=failure.errors(include_context=False, include_input=False, include_url=False)
        ) from failure

    models = {group.model for group in config.groups if group.model is not None}
    default = config.override_model or (next(iter(models)) if len(models) == 1 else None)
    fields_by_model: dict[ExtractionModel | None, list[str]] = {}
    for group in config.groups:
        if group.name not in settings.API_FIELD_GROUPS:
            raise HTTPException(422, f"Unknown field group: {group.name}")
        model = config.override_model or group.model or default
        fields_by_model.setdefault(model, []).extend(settings.API_FIELD_GROUPS[group.name])
    if config.fields_to_extract or not fields_by_model:
        fields_by_model.setdefault(default, []).extend(config.fields_to_extract)

    # Discovery runs once. Prefer the default plan when present, then GPU,
    # then CPU, so rearranging the same groups preserves idempotency.
    order = dict.fromkeys((default, "gpu", "cpu", None))
    try:
        plans = {
            model: parse_requested_fields(fields_by_model[model])
            for model in order if model in fields_by_model
        }
    except ValueError as failure:
        raise HTTPException(422, str(failure)) from failure
    return config, plans


def _extraction_provider(state: Any, settings: Settings, active: Settings) -> ModelProvider:
    if not state.owns_provider or (
        active.PROVIDER_PROFILE is settings.PROVIDER_PROFILE
        and active.gpu_requested == settings.gpu_requested
        and active.gpu_required == settings.gpu_required
    ):
        return state.provider
    profile = active.PROVIDER_PROFILE
    if profile not in state.extraction_providers:
        state.extraction_providers[profile] = get_provider(active, log_banner=False)
    return state.extraction_providers[profile]


def _merge_extractions(
    results: list[ResultEnvelope],
    plans: tuple[tuple[str, ...], ...],
    requested: tuple[str, ...],
    version: int,
) -> ResultEnvelope:
    """Publish one result after all group models finish."""
    from app.field_selection import matches_field
    from app.stages.assemble import merge_field_candidates
    from app.stages.s5_discover import add_discovered

    base = next((result for result in results if result.document is not None), results[0])
    documents = [result.document for result in results if result.document is not None]
    document = base.document
    if document is not None:
        per_model = [
            {
                name: value for name, value in result.document.fields.items()
                if any(matches_field(name, field) for field in targets)
                or not any(matches_field(name, field) for field in requested)
            }
            for result, targets in zip(results, plans, strict=True)
            if result.document is not None
        ]
        fields = merge_field_candidates(*per_model)
        for candidates in per_model:
            for name, candidate in candidates.items():
                add_discovered(fields, name, candidate)
        summary = next(
            (item.complaint_summary for item in documents if item.complaint_summary is not None),
            None,
        )
        document = document.model_copy(update={"fields": fields, "complaint_summary": summary})
    timings: dict[str, float] = {}
    for result in results:
        for name, duration in result.processing.stage_timings_ms.items():
            timings[name] = timings.get(name, 0.0) + duration
    started = min(
        (result.processing.started_at for result in results if result.processing.started_at),
        default=None,
    )
    finished = max(
        (result.processing.finished_at for result in results if result.processing.finished_at),
        default=None,
    )
    stats = base.processing.model_copy(update={
        "started_at": started,
        "finished_at": finished,
        "duration_ms": (finished - started).total_seconds() * 1000 if started and finished else None,
        "tokens_used": sum(result.processing.tokens_used for result in results),
        "stage_timings_ms": timings,
    })
    return ResultEnvelope.model_validate({
        **base.model_dump(),
        "document": document,
        "processing": stats,
        "provider_profile": "mixed",
        "requested_fields": requested,
        "result_version": version,
        "escalation_flag": any(result.escalation_flag for result in results),
        "warnings": tuple(dict.fromkeys(warning for result in results for warning in result.warnings)),
    })


def _model_settings(settings: Settings, model: ExtractionModel | None) -> Settings:
    if model is None:
        return settings
    values = settings.model_dump()
    if model == "cpu":
        values.update(
            PROVIDER_PROFILE=ProviderProfile.CPU_NODE,
            GPU_EXECUTION="never",
            GPU_HANDWRITING_ENABLED=False,
            ESCALATION_SHADOW_RATE=0.0,
        )
    else:
        if not settings.slm_endpoint_configured or not settings.slm_endpoint_is_local:
            raise HTTPException(422, "GPU extraction requires an in-house HF_BASE_URL")
        values.update(
            PROVIDER_PROFILE=ProviderProfile.GPU_NODE,
            GPU_HANDWRITING_ENABLED=True,
        )
    return Settings.model_validate(values)


def _extraction_response(job: Job, envelope: ResultEnvelope | None) -> ExtractionResponse:
    pages: tuple[int, ...] = ()
    data: dict[str, ExtractedValue] = {}
    if envelope is not None:
        pages = tuple(
            page.page_no for page in envelope.pages
            if page.quality_verdict is not QualityVerdict.BOUNCE
        )
        document = envelope.document
        if document is not None:
            fields = (
                dict(document.other_fields)
                if job.document_id.startswith(_CONFIGURED_ALL_PREFIX)
                else {}
            )
            fields.update({
                name: candidates[0]
                for name, candidates in document.extracted_fields.items()
                if candidates
            })
            for name, field in fields.items():
                if field.value is not None:
                    data[name] = ExtractedValue(
                        value=field.value,
                        confidence=field.confidence,
                        source_page=field.evidence.page_no,
                        label=field.label,
                        validation_status=ExtractionValidation(
                            method=field.source,
                            passed=field.validator_passed,
                            flags=field.flags,
                        ),
                    )
    return ExtractionResponse(job_id=job.id, pages_processed=pages, extracted_data=data)


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
