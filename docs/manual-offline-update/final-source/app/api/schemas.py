"""Request and response shapes for the HTTP surface.

Deliberately thin.  :class:`~app.contracts.envelope.ResultEnvelope` **is** the
product and is returned as itself — re-modelling it here would create a second
definition of the thing the JSON Schema in ``docs/schema/`` pins, and the two
would drift on the first field anyone added.  What lives here is only the
envelope's *wrapper*: the idempotency answer, the job view, and the health
report, none of which the pipeline knows about.

This module does NOT handle: validation of the envelope (its own models do),
authentication, or pagination.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.envelope import ResultEnvelope

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "JobResponse",
    "SubmissionResponse",
]


class SubmissionResponse(BaseModel):
    """Answer to ``POST /v1/documents``.

    ``is_new=false`` with a populated ``result`` is the idempotent path: the
    same sha256 arrived before. An unchanged requested-field list reuses the
    saved result; changing that list appends a fresh extraction version.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    document_id: str
    #: False when this sha256 had already been submitted (BUILD-BRIEF M3).
    is_new: bool
    status: str
    #: The job this one probably repeats — same sender, subject and day.
    soft_duplicate_of: str | None = None
    #: Machine-readable, ``None`` when this is not a soft duplicate.
    soft_duplicate_reason: str | None = None
    #: The latest envelope. ``None`` only while a queued job has produced none.
    result: ResultEnvelope | None = None


class JobResponse(BaseModel):
    """Answer to ``GET /v1/jobs/{job_id}``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    document_id: str
    status: str
    result_version: int = Field(ge=0)
    is_final: bool
    created_at: datetime
    updated_at: datetime
    soft_duplicate_of: str | None = None
    soft_duplicate_reason: str | None = None
    error_code: str | None = None
    #: Every version this job published, oldest first.
    versions: tuple[int, ...] = ()
    result: ResultEnvelope | None = None


class ModelArtifactReport(BaseModel):
    """Which weights are on this node — the air-gapped deployment's first check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    present: bool


class HealthResponse(BaseModel):
    """Answer to ``GET /healthz``: the startup banner, in JSON.

    ``status`` is ``degraded`` rather than ``unhealthy`` when weights are
    missing: the service still ingests, still bounces, still validates and still
    answers — it just cannot recognise.  Reporting that as "down" would make an
    orchestrator restart a container whose problem is a missing volume mount.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    profile: str
    synthetic_only: bool
    banner: str
    #: 1 for the active profile, 0 for the others — mirrors the Prometheus gauge.
    provider_profile_gauge: dict[str, float]
    models: tuple[ModelArtifactReport, ...] = ()
    database: str
    webhook_enabled: bool
    schema_version: str


class RouteStats(BaseModel):
    """Counts for one route template since this process started."""

    model_config = ConfigDict(extra="forbid")

    route: str
    requests: int = Field(ge=0)
    client_errors: int = Field(ge=0, description="4xx responses.")
    server_errors: int = Field(ge=0, description="5xx responses.")
    error_rate: float = Field(
        ge=0.0, le=1.0, description="(4xx + 5xx) / requests; 0 when idle."
    )


class StatsResponse(BaseModel):
    """``GET /v1/stats`` — how the service is performing, in plain numbers.

    A readable companion to ``/metrics``, not a replacement: Prometheus text is
    what a scraper wants and this is what a person wants when the question is
    "is it healthy right now?".  Both read the same in-memory counters.

    **Everything here resets when the process restarts.**  The counters live in
    the ``prometheus_client`` default registry, which is memory-only; nothing
    is persisted, so these are "since boot" figures and never a historical
    record.  ``uptime_seconds`` is included precisely so a rate can be
    interpreted rather than guessed at.
    """

    model_config = ConfigDict(extra="forbid")

    uptime_seconds: float = Field(ge=0.0)
    requests: int = Field(ge=0)
    client_errors: int = Field(ge=0)
    server_errors: int = Field(ge=0)
    unhandled_exceptions: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    routes: tuple[RouteStats, ...] = ()
    #: Documents through the pipeline, by outcome (``accepted``/``rejected``).
    documents: dict[str, int] = Field(default_factory=dict)
    #: Pages the quality gate refused, by machine-readable reason.
    bounces: dict[str, int] = Field(default_factory=dict)
    #: Model-endpoint calls by outcome, when a remote endpoint is configured.
    provider_calls: dict[str, int] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The shape of every non-2xx body — the taxonomy's ``to_dict()``."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    context: dict[str, object] = Field(default_factory=dict)
