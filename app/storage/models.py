"""SQLAlchemy 2 models — jobs, versioned results, the audit trail, the DLQ.

PLAN.md D2 makes this dual-mode: the same declarative models run on the SQLite
file the laptop profile defaults to and on the Postgres 16 instance
``docker-compose.yml`` provisions.  Nothing here is dialect-specific — ``JSON``
is SQLAlchemy's portable type (native ``json`` on Postgres, serialised ``TEXT``
on SQLite) and every default is computed in Python rather than by the server, so
the two backends cannot drift on timestamps.

**Append-only is enforced here, in the ORM, not by a grant.**
:class:`AppendOnly` marks the tables that are evidence rather than state —
``results`` and ``audit_events`` — and mapper-level ``before_update`` /
``before_delete`` listeners refuse the flush with
:class:`~app.contracts.errors.AppendOnlyViolation`.  It has to be here because
SQLite has no per-table privileges at all, so a rule expressed as a Postgres
grant would simply not exist under the default profile — which is the profile
the tests, the eval harness and the laptop demo all run.  A correction to a
field is a **new** ``audit_events`` row and a **new** ``results`` row; the
previous version stays exactly as it was published.

The escape hatch is deliberate and narrow: raw SQL and ``Query.delete()``
bypass mapper events, so :func:`~app.storage.database.build_session_factory`
also installs a session-level guard.  Neither can stop a DBA with a psql
prompt, and neither is meant to — the point is that no code path *in this
service* can rewrite the trail by accident.

This module does NOT handle:
  * creating the schema or migrating it (``alembic/``, and
    :func:`~app.storage.database.create_all` for the SQLite default);
  * sessions, engines or transactions (``app/storage/database.py``);
  * business rules — "is this a duplicate?" is
    :mod:`app.storage.repository`'s, and the *derivation* of the soft-duplicate
    key belongs to Stage 1 (``s1_ingest.soft_duplicate_key``) so the storage
    layer cannot quietly redefine what "same complaint" means;
  * PII policy.  ``jobs.sender``/``subject`` and the stored envelope contain
    complainant data by construction; the logging filter keeps them out of the
    log and this layer never logs a row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Final

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.contracts.errors import AppendOnlyViolation

__all__ = [
    "APPEND_ONLY_TABLES",
    "AppendOnly",
    "AuditAction",
    "AuditEvent",
    "Base",
    "DeliveryStatus",
    "Job",
    "JobStatus",
    "Result",
    "WebhookDelivery",
    "assert_no_append_only_mutation",
    "utc_now",
]


def utc_now() -> datetime:
    """Timezone-aware ``now``.

    Defaults are computed in Python, not with ``server_default=func.now()``:
    SQLite's ``CURRENT_TIMESTAMP`` is naive UTC and Postgres' is the server's
    ``now()``, and a trail whose timestamps mean different things on different
    backends is worse than one with none.
    """
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every table in the service."""

    type_annotation_map: ClassVar[dict[type, Any]] = {
        dict[str, Any]: JSON,
    }


class AppendOnly:
    """Marker: rows of this table are evidence and are never rewritten.

    A mixin rather than a naming convention so the guard is a type check, and
    so a new evidence table is opted in by inheriting rather than by remembering
    to add it to a list.
    """

    __append_only__: ClassVar[bool] = True


#: Populated at import time from the mapped subclasses of :class:`AppendOnly`.
APPEND_ONLY_TABLES: Final[set[str]] = set()


class JobStatus(StrEnum):
    """Lifecycle of one submitted document."""

    #: Accepted, not yet started (Celery mode).
    QUEUED = "queued"
    RUNNING = "running"
    #: Finished with a document; ``is_final`` result stored.
    COMPLETED = "completed"
    #: Finished without a document — the quality gate or ingestion refused it.
    BOUNCED = "bounced"
    #: The pipeline raised. ``error_code`` names the typed error.
    FAILED = "failed"


class AuditAction(StrEnum):
    """Closed vocabulary of audited actions (PRD P1-1).

    Closed because the review queue and the retention policy both filter on it;
    a free-text action is a column nobody can query.
    """

    JOB_CREATED = "job_created"
    JOB_STATUS_CHANGED = "job_status_changed"
    RESULT_PUBLISHED = "result_published"
    FIELD_CORRECTED = "field_corrected"
    SOFT_DUPLICATE_LINKED = "soft_duplicate_linked"
    WEBHOOK_ENQUEUED = "webhook_enqueued"
    WEBHOOK_DELIVERED = "webhook_delivered"
    WEBHOOK_DEAD_LETTERED = "webhook_dead_lettered"
    WEBHOOK_REPLAYED = "webhook_replayed"


class DeliveryStatus(StrEnum):
    """State of one webhook delivery attempt chain."""

    PENDING = "pending"
    DELIVERED = "delivered"
    #: Retries exhausted; parked in the DLQ and replayable.
    DEAD = "dead"


class Job(Base):
    """One submitted document, keyed by the sha256 of its bytes.

    ``document_id`` is unique, which *is* the idempotency guarantee of
    BUILD-BRIEF M3: a resubmission of identical bytes finds this row instead of
    creating a second one, and the API answers ``is_new=false``.  The soft
    duplicate is a separate, weaker signal — same sender, same normalised
    subject, same calendar day — and is recorded rather than enforced, because
    two attachments to one mail are legitimately two jobs.
    """

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    #: ``sha256:<hex>`` over the attachment bytes (``s1_ingest.idempotency_key``).
    document_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value)
    filename: Mapped[str | None] = mapped_column(String(512), default=None)
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[str | None] = mapped_column(String(64), default=None)
    provider_profile: Mapped[str | None] = mapped_column(String(16), default=None)

    # --- soft duplicate: (sender, normalised subject, day) ---
    sender: Mapped[str | None] = mapped_column(String(320), default=None)
    subject: Mapped[str | None] = mapped_column(Text, default=None)
    normalized_subject: Mapped[str | None] = mapped_column(Text, default=None)
    received_day: Mapped[str | None] = mapped_column(String(10), default=None)
    #: ``sha256`` of the triple, from ``s1_ingest.soft_duplicate_key``.
    soft_duplicate_key: Mapped[str | None] = mapped_column(
        String(80), default=None, index=True
    )
    #: The earlier job this one probably repeats; ``None`` when it is the first.
    soft_duplicate_of: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id"), default=None
    )
    soft_duplicate_reason: Mapped[str | None] = mapped_column(String(64), default=None)

    latest_result_version: Mapped[int] = mapped_column(Integer, default=0)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    #: Relationships exist here for the *flush ordering* as much as for
    #: navigation: SQLAlchemy's unit of work orders inserts from inter-mapper
    #: relationships, not from raw foreign keys, so without them an audit row
    #: can be emitted before the job it references and SQLite's
    #: ``PRAGMA foreign_keys=ON`` (correctly) refuses it.
    results: Mapped[list[Result]] = relationship(
        back_populates="job", order_by="Result.result_version", lazy="selectin"
    )
    audit_events: Mapped[list[AuditEvent]] = relationship(
        back_populates="job", order_by="AuditEvent.id"
    )
    deliveries: Mapped[list[WebhookDelivery]] = relationship(
        back_populates="job", order_by="WebhookDelivery.id"
    )

    __table_args__ = (
        Index(
            "ix_jobs_soft_duplicate",
            "sender",
            "normalized_subject",
            "received_day",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"Job(id={self.id!r}, document_id={self.document_id!r}, "
            f"status={self.status!r}, v={self.latest_result_version})"
        )


class Result(Base, AppendOnly):
    """One published :class:`~app.contracts.envelope.ResultEnvelope`, verbatim.

    Append-only and unique on ``(job_id, result_version)``: version 1 is what a
    consumer acted on at the fast-path SLA, and it must still say what it said
    after Tier 2 landed.  The envelope is stored as the JSON the API served, not
    as re-normalised columns, so a schema migration can never silently rewrite a
    published result.
    """

    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    result_version: Mapped[int] = mapped_column(Integer)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    schema_version: Mapped[str] = mapped_column(String(16))
    document_id: Mapped[str] = mapped_column(String(80), index=True)
    envelope: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    job: Mapped[Job] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("job_id", "result_version", name="uq_results_job_version"),
    )

    def __repr__(self) -> str:
        return (
            f"Result(job_id={self.job_id!r}, version={self.result_version}, "
            f"final={self.is_final})"
        )


class AuditEvent(Base, AppendOnly):
    """Append-only trail: actor, timestamp, before, after (BUILD-BRIEF sec. 5).

    ``before``/``after`` are free-shaped JSON because what changed differs by
    action — a status transition carries two strings, a field correction carries
    two :class:`~app.contracts.envelope.Field` dumps.  Both may contain
    complainant data, so this table is inside the same retention boundary as
    ``results`` and is never logged.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id"), default=None, index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    #: Who or what acted: ``system``, ``celery:q.vision``, ``user:<id>``.
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(48), index=True)
    #: What was acted on: ``job``, ``result``, ``field:<name>``.
    entity: Mapped[str | None] = mapped_column(String(128), default=None)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    detail: Mapped[str | None] = mapped_column(Text, default=None)

    job: Mapped[Job | None] = relationship(back_populates="audit_events")

    def __repr__(self) -> str:
        return (
            f"AuditEvent(id={self.id}, job_id={self.job_id!r}, "
            f"action={self.action!r}, actor={self.actor!r})"
        )


class WebhookDelivery(Base):
    """One result delivery and its retry state — the replayable DLQ.

    **Not** append-only, deliberately: this is operational state, not evidence.
    ``attempts`` and ``next_attempt_at`` are meant to be updated in place, and
    the audit trail records the transitions that matter (``webhook_delivered``,
    ``webhook_dead_lettered``, ``webhook_replayed``) in the table that cannot be
    rewritten.

    The signed ``payload`` is stored so a replay re-sends **byte-identically**;
    re-serialising from the envelope would produce a different body and
    therefore a different HMAC, and a receiver that deduplicates on the
    signature would treat the replay as a new event.
    """

    __tablename__ = "webhook_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    result_version: Mapped[int] = mapped_column(Integer)
    url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(16), default=DeliveryStatus.PENDING.value, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=5)
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_status_code: Mapped[int | None] = mapped_column(Integer, default=None)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
    #: Exactly the bytes that were signed, decoded as UTF-8.
    payload: Mapped[str] = mapped_column(Text)
    signature: Mapped[str] = mapped_column(String(128))
    signed_at: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    job: Mapped[Job] = relationship(back_populates="deliveries")

    __table_args__ = (
        UniqueConstraint(
            "job_id", "result_version", name="uq_webhook_job_version"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"WebhookDelivery(job_id={self.job_id!r}, "
            f"version={self.result_version}, status={self.status!r}, "
            f"attempts={self.attempts})"
        )


# --------------------------------------------------------------------------- #
# Append-only enforcement
# --------------------------------------------------------------------------- #


def _refuse(target: object, operation: str) -> None:
    table = type(target).__tablename__  # type: ignore[attr-defined]
    raise AppendOnlyViolation(
        f"{table} is append-only: {operation} is refused. A correction is a new "
        "row — the published version must still say what it said when a "
        "consumer acted on it (BUILD-BRIEF sec. 5).",
        context={"table": table, "operation": operation},
    )


def _register_append_only(cls: type) -> None:
    """Attach the refuse-to-mutate listeners to one mapped class."""
    APPEND_ONLY_TABLES.add(cls.__tablename__)  # type: ignore[attr-defined]

    @event.listens_for(cls, "before_update")
    def _no_update(_mapper: object, _connection: object, target: object) -> None:
        _refuse(target, "UPDATE")

    @event.listens_for(cls, "before_delete")
    def _no_delete(_mapper: object, _connection: object, target: object) -> None:
        _refuse(target, "DELETE")


for _cls in (Result, AuditEvent):
    _register_append_only(_cls)


def assert_no_append_only_mutation(session: Session) -> None:
    """Refuse a flush that would rewrite evidence — the session-level net.

    Mapper events cover the ORM path; this covers what they cannot see, which is
    a ``Query.delete()``/``Query.update()`` marking objects in the session, and
    it runs *before* any SQL is emitted rather than mid-flush.  Installed by
    :func:`~app.storage.database.build_session_factory`.

    Raises:
        AppendOnlyViolation: any dirty or deleted append-only instance.
    """
    for instance in session.deleted:
        if isinstance(instance, AppendOnly):
            _refuse(instance, "DELETE")
    for instance in session.dirty:
        if isinstance(instance, AppendOnly) and session.is_modified(instance):
            _refuse(instance, "UPDATE")
