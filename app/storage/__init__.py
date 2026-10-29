"""Persistence: jobs, versioned results, an append-only audit trail, the DLQ.

Dual-mode by PLAN.md D2 — SQLite by default, Postgres through ``DATABASE_URL``,
one set of models for both.  Import what you need from here; the split into
:mod:`~app.storage.models`, :mod:`~app.storage.database` and
:mod:`~app.storage.repository` is an implementation detail of this package.

This package does NOT handle: running the pipeline, HTTP, webhook transport, or
object storage for page images (MinIO is provisioned in ``docker-compose.yml``
for a later phase; Phase 0/1 keeps pixels in memory for the duration of a run
and persists only the envelope).
"""

from __future__ import annotations

from app.storage.database import (
    build_engine,
    build_session_factory,
    create_all,
    dispose_engine,
    session_scope,
)
from app.storage.models import (
    APPEND_ONLY_TABLES,
    AppendOnly,
    AuditAction,
    AuditEvent,
    Base,
    DeliveryStatus,
    Job,
    JobStatus,
    Result,
    WebhookDelivery,
)
from app.storage.repository import SYSTEM_ACTOR, JobRecord, JobStore

__all__ = [
    "APPEND_ONLY_TABLES",
    "SYSTEM_ACTOR",
    "AppendOnly",
    "AuditAction",
    "AuditEvent",
    "Base",
    "DeliveryStatus",
    "Job",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "Result",
    "WebhookDelivery",
    "build_engine",
    "build_session_factory",
    "create_all",
    "dispose_engine",
    "session_scope",
]
