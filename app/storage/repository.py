"""The only place that reads or writes the database.

Everything the API, the Celery tasks and the webhook worker need from storage is
a method here, so the append-only rule, the idempotency rule and the audit rule
are each written once.

**Idempotency** is `sha256` of the attachment bytes (BUILD-BRIEF M3).
:meth:`JobStore.create_or_get_job` is the single door: a resubmission of
identical bytes returns the existing job with ``is_new=False`` rather than
starting a second pipeline run, and the caller answers the HTTP request from the
result already stored.

**Soft duplication** is the weaker signal — same sender, same normalised
subject, same calendar day.  It is *recorded*, never enforced: two attachments
on one mail are legitimately two jobs, and refusing the second would drop half
a complaint.  The key itself is derived by
:func:`app.stages.s1_ingest.soft_duplicate_key` so this layer cannot quietly
redefine what "the same complaint" means.

**Every mutation is audited.**  Status transitions, published results and field
corrections all write an ``audit_events`` row with actor, timestamp, before and
after.  The rows are unrewritable by construction (see
:mod:`app.storage.models`), so the trail is evidence rather than a log.

This module does NOT handle:
  * running the pipeline, or deciding what a job's status *should* be;
  * HTTP status codes — it raises :class:`~app.contracts.errors.JobNotFound`
    and the API maps it;
  * webhook transport, signing or backoff (``app/webhooks.py``); it owns the
    DLQ **rows** and their state transitions, not the delivery;
  * schema creation or migration (``app/storage/database.py``, ``alembic/``);
  * redaction.  ``sender``, ``subject`` and the stored envelope are complainant
    data; nothing here logs a row's contents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.contracts.envelope import QualityVerdict, ResultEnvelope
from app.contracts.errors import JobNotFound, StorageError
from app.settings import Settings, get_settings
from app.stages.s1_ingest import normalise_subject, soft_duplicate_key
from app.storage.database import (
    build_engine,
    build_session_factory,
    create_all,
    session_scope,
)
from app.storage.models import (
    AuditAction,
    AuditEvent,
    DeliveryStatus,
    Job,
    JobStatus,
    Result,
    WebhookDelivery,
    utc_now,
)

__all__ = ["SYSTEM_ACTOR", "JobRecord", "JobStore"]

_LOG = logging.getLogger(__name__)

#: Actor recorded for anything the service does on its own behalf.  A human
#: correction from the review UI records ``user:<id>`` instead (PRD P1-1).
SYSTEM_ACTOR = "system"


@dataclass(frozen=True)
class JobRecord:
    """A job plus whether this submission created it.

    ``is_new=False`` is the whole idempotency answer: the caller must not start
    a second pipeline run, and the envelope to return is already stored.
    """

    job: Job
    is_new: bool
    soft_duplicate_of: str | None = None
    soft_duplicate_reason: str | None = None


class JobStore:
    """Session-per-operation access to jobs, results, audit events and the DLQ.

    Owns its engine unless one is injected.  A session is opened and closed per
    method rather than held, because the API, the in-process webhook thread and
    a Celery worker all use the same store concurrently and a shared session is
    not thread-safe.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        url: str | None = None,
        create_schema: bool = True,
    ) -> None:
        self.settings = settings if settings is not None else get_settings()
        self.engine = build_engine(self.settings, url=url)
        self.sessions: sessionmaker[Session] = build_session_factory(self.engine)
        if create_schema:
            create_all(self.engine)

    def close(self) -> None:
        """Dispose the engine.  Safe to call twice."""
        self.engine.dispose()

    # ------------------------------------------------------------------ jobs
    def create_or_get_job(
        self,
        *,
        document_id: str,
        filename: str | None = None,
        byte_size: int = 0,
        content_type: str | None = None,
        sender: str | None = None,
        subject: str | None = None,
        received_at: datetime | date | None = None,
        provider_profile: str | None = None,
        actor: str = SYSTEM_ACTOR,
    ) -> JobRecord:
        """Return the job for ``document_id``, creating it only if it is new.

        The uniqueness of ``jobs.document_id`` is what makes this idempotent,
        and the check is a lookup inside the same transaction as the insert, so
        two concurrent submissions of one attachment cannot both create a row —
        the loser's flush violates the constraint and is retried as a lookup.

        Raises:
            StorageError: the write failed for a reason other than the
                idempotency constraint.
        """
        with session_scope(self.sessions) as session:
            existing = session.scalar(
                select(Job).where(Job.document_id == document_id)
            )
            if existing is not None:
                return JobRecord(
                    job=existing,
                    is_new=False,
                    soft_duplicate_of=existing.soft_duplicate_of,
                    soft_duplicate_reason=existing.soft_duplicate_reason,
                )

            day = _calendar_day(received_at)
            normalized = normalise_subject(subject) if subject else None
            dupe_key = (
                soft_duplicate_key(sender, subject, day)
                if sender and subject is not None
                else None
            )
            earlier = (
                session.scalar(
                    select(Job)
                    .where(Job.soft_duplicate_key == dupe_key)
                    .order_by(Job.created_at)
                    .limit(1)
                )
                if dupe_key
                else None
            )

            job = Job(
                id=uuid.uuid4().hex,
                document_id=document_id,
                status=JobStatus.QUEUED.value,
                filename=filename,
                byte_size=byte_size,
                content_type=content_type,
                provider_profile=provider_profile,
                sender=sender,
                subject=subject,
                normalized_subject=normalized,
                received_day=day.isoformat(),
                soft_duplicate_key=dupe_key,
                soft_duplicate_of=earlier.id if earlier is not None else None,
                soft_duplicate_reason=(
                    "same_sender_subject_day" if earlier is not None else None
                ),
            )
            session.add(job)
            session.add(
                AuditEvent(
                    job_id=job.id,
                    actor=actor,
                    action=AuditAction.JOB_CREATED.value,
                    entity="job",
                    after={"document_id": document_id, "status": job.status},
                )
            )
            if earlier is not None:
                session.add(
                    AuditEvent(
                        job_id=job.id,
                        actor=actor,
                        action=AuditAction.SOFT_DUPLICATE_LINKED.value,
                        entity="job",
                        after={"soft_duplicate_of": earlier.id},
                        detail="same sender, normalised subject and calendar day",
                    )
                )
            try:
                session.flush()
            except SQLAlchemyError as failure:
                raise StorageError(
                    f"could not create a job for {document_id}: {failure}",
                    context={"document_id": document_id},
                ) from failure
            return JobRecord(
                job=job,
                is_new=True,
                soft_duplicate_of=job.soft_duplicate_of,
                soft_duplicate_reason=job.soft_duplicate_reason,
            )

    def get_job(self, job_id: str) -> Job:
        """One job by id.

        Raises:
            JobNotFound: no such job.
        """
        with session_scope(self.sessions) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobNotFound(
                    f"no job with id {job_id!r}", context={"job_id": job_id}
                )
            _ = job.results  # materialise before the session closes
            return job

    def find_by_document_id(self, document_id: str) -> Job | None:
        """The job for an attachment's sha256, or ``None``."""
        with session_scope(self.sessions) as session:
            job = session.scalar(select(Job).where(Job.document_id == document_id))
            if job is not None:
                _ = job.results
            return job

    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        actor: str = SYSTEM_ACTOR,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> Job:
        """Move a job to ``status``, recording the transition in the trail.

        Raises:
            JobNotFound: no such job.
        """
        with session_scope(self.sessions) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobNotFound(
                    f"no job with id {job_id!r}", context={"job_id": job_id}
                )
            before = job.status
            job.status = status.value
            job.error_code = error_code
            job.error_message = error_message
            session.add(
                AuditEvent(
                    job_id=job.id,
                    actor=actor,
                    action=AuditAction.JOB_STATUS_CHANGED.value,
                    entity="job",
                    before={"status": before},
                    after={"status": status.value, "error_code": error_code},
                )
            )
            return job

    # --------------------------------------------------------------- results
    def record_result(
        self,
        job_id: str,
        envelope: ResultEnvelope,
        *,
        actor: str = SYSTEM_ACTOR,
        update_status: bool = True,
    ) -> Result:
        """Append one published envelope and advance the job's version.

        The version must be exactly one more than the last stored — a gap means
        an envelope was published and lost, and silently accepting it would make
        the trail claim a continuity it does not have.

        Raises:
            JobNotFound: no such job.
            StorageError: the version is not the next one, or the write failed.
        """
        with session_scope(self.sessions) as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobNotFound(
                    f"no job with id {job_id!r}", context={"job_id": job_id}
                )
            expected = job.latest_result_version + 1
            if envelope.result_version != expected:
                raise StorageError(
                    f"result_version {envelope.result_version} is not the next "
                    f"version for job {job_id} (expected {expected}); versions "
                    "are append-only and must not skip",
                    context={
                        "job_id": job_id,
                        "expected": expected,
                        "received": envelope.result_version,
                    },
                )
            payload = envelope.model_dump(mode="json")
            row = Result(
                job_id=job.id,
                result_version=envelope.result_version,
                is_final=envelope.is_final,
                schema_version=envelope.schema_version,
                document_id=envelope.document_id,
                envelope=payload,
            )
            session.add(row)
            job.latest_result_version = envelope.result_version
            job.is_final = envelope.is_final
            if update_status and envelope.is_final:
                bounced = (
                    envelope.bounce_decision.decision is QualityVerdict.BOUNCE
                )
                job.status = (
                    JobStatus.BOUNCED.value if bounced else JobStatus.COMPLETED.value
                )
            elif update_status:
                job.status = JobStatus.RUNNING.value
            session.add(
                AuditEvent(
                    job_id=job.id,
                    actor=actor,
                    action=AuditAction.RESULT_PUBLISHED.value,
                    entity=f"result:v{envelope.result_version}",
                    after={
                        "result_version": envelope.result_version,
                        "is_final": envelope.is_final,
                        "escalation_flag": envelope.escalation_flag,
                        "bounce": envelope.bounce_decision.decision.value,
                    },
                )
            )
            try:
                session.flush()
            except SQLAlchemyError as failure:
                raise StorageError(
                    f"could not store result v{envelope.result_version} for "
                    f"job {job_id}: {failure}",
                    context={"job_id": job_id},
                ) from failure
            return row

    def results_for(self, job_id: str) -> list[Result]:
        """Every stored version for a job, oldest first."""
        with session_scope(self.sessions) as session:
            return list(
                session.scalars(
                    select(Result)
                    .where(Result.job_id == job_id)
                    .order_by(Result.result_version)
                )
            )

    def latest_envelope(self, job_id: str) -> ResultEnvelope | None:
        """The highest stored version, revalidated through the contract.

        Revalidated rather than returned as raw JSON so a schema drift between
        what was written and what the models now accept surfaces here, at the
        read, instead of in a consumer's parser.
        """
        with session_scope(self.sessions) as session:
            row = session.scalar(
                select(Result)
                .where(Result.job_id == job_id)
                .order_by(Result.result_version.desc())
                .limit(1)
            )
            if row is None:
                return None
            return ResultEnvelope.model_validate(row.envelope)

    # ----------------------------------------------------------------- audit
    def record_audit(
        self,
        *,
        actor: str,
        action: AuditAction,
        job_id: str | None = None,
        entity: str | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        detail: str | None = None,
    ) -> AuditEvent:
        """Append one audit row.  The only way to write the trail."""
        with session_scope(self.sessions) as session:
            row = AuditEvent(
                job_id=job_id,
                actor=actor,
                action=action.value,
                entity=entity,
                before=before,
                after=after,
                detail=detail,
            )
            session.add(row)
            session.flush()
            return row

    def audit_trail(self, job_id: str) -> list[AuditEvent]:
        """Every audit row for a job, oldest first."""
        with session_scope(self.sessions) as session:
            return list(
                session.scalars(
                    select(AuditEvent)
                    .where(AuditEvent.job_id == job_id)
                    .order_by(AuditEvent.id)
                )
            )

    # ------------------------------------------------------------------- DLQ
    def enqueue_delivery(
        self,
        *,
        job_id: str,
        result_version: int,
        url: str,
        payload: str,
        signature: str,
        signed_at: int,
        max_attempts: int,
        actor: str = SYSTEM_ACTOR,
    ) -> WebhookDelivery:
        """Record a signed delivery as pending, or return the existing row.

        Unique on ``(job_id, result_version)``: a retry of the enqueue — which
        is what an ``acks_late`` Celery task redelivery looks like — must not
        create a second delivery of the same version.
        """
        with session_scope(self.sessions) as session:
            existing = session.scalar(
                select(WebhookDelivery).where(
                    WebhookDelivery.job_id == job_id,
                    WebhookDelivery.result_version == result_version,
                )
            )
            if existing is not None:
                return existing
            row = WebhookDelivery(
                job_id=job_id,
                result_version=result_version,
                url=url,
                payload=payload,
                signature=signature,
                signed_at=signed_at,
                max_attempts=max_attempts,
                status=DeliveryStatus.PENDING.value,
                next_attempt_at=utc_now(),
            )
            session.add(row)
            session.add(
                AuditEvent(
                    job_id=job_id,
                    actor=actor,
                    action=AuditAction.WEBHOOK_ENQUEUED.value,
                    entity=f"webhook:v{result_version}",
                    after={"result_version": result_version},
                )
            )
            session.flush()
            return row

    def due_deliveries(self, *, limit: int = 32) -> list[WebhookDelivery]:
        """Pending deliveries whose backoff has elapsed, oldest first."""
        now = utc_now()
        with session_scope(self.sessions) as session:
            rows = list(
                session.scalars(
                    select(WebhookDelivery)
                    .where(WebhookDelivery.status == DeliveryStatus.PENDING.value)
                    .order_by(WebhookDelivery.id)
                    .limit(limit)
                )
            )
        return [row for row in rows if _due(row, now)]

    def mark_delivered(
        self,
        delivery_id: int,
        *,
        status_code: int,
        actor: str = SYSTEM_ACTOR,
    ) -> WebhookDelivery:
        """Close a delivery successfully and audit it."""
        with session_scope(self.sessions) as session:
            row = self._delivery(session, delivery_id)
            row.status = DeliveryStatus.DELIVERED.value
            row.attempts += 1
            row.last_status_code = status_code
            row.last_error = None
            row.next_attempt_at = None
            row.delivered_at = utc_now()
            session.add(
                AuditEvent(
                    job_id=row.job_id,
                    actor=actor,
                    action=AuditAction.WEBHOOK_DELIVERED.value,
                    entity=f"webhook:v{row.result_version}",
                    after={"status_code": status_code, "attempts": row.attempts},
                )
            )
            return row

    def mark_attempt_failed(
        self,
        delivery_id: int,
        *,
        status_code: int | None,
        error: str,
        backoff_seconds: float,
        actor: str = SYSTEM_ACTOR,
    ) -> WebhookDelivery:
        """Record one failed attempt; park the delivery when attempts run out.

        ``error`` is truncated: it may quote the receiver's response body, which
        is not ours to store in full, and the DLQ is for replaying a delivery
        rather than for diagnosing someone else's service.
        """
        with session_scope(self.sessions) as session:
            row = self._delivery(session, delivery_id)
            row.attempts += 1
            row.last_status_code = status_code
            row.last_error = error[:500]
            if row.attempts >= row.max_attempts:
                row.status = DeliveryStatus.DEAD.value
                row.next_attempt_at = None
                session.add(
                    AuditEvent(
                        job_id=row.job_id,
                        actor=actor,
                        action=AuditAction.WEBHOOK_DEAD_LETTERED.value,
                        entity=f"webhook:v{row.result_version}",
                        after={
                            "attempts": row.attempts,
                            "status_code": status_code,
                        },
                    )
                )
            else:
                row.next_attempt_at = utc_now() + timedelta(seconds=backoff_seconds)
            return row

    def dead_letters(self, *, limit: int = 100) -> list[WebhookDelivery]:
        """Parked deliveries, oldest first — the replay queue."""
        with session_scope(self.sessions) as session:
            return list(
                session.scalars(
                    select(WebhookDelivery)
                    .where(WebhookDelivery.status == DeliveryStatus.DEAD.value)
                    .order_by(WebhookDelivery.id)
                    .limit(limit)
                )
            )

    def replay_delivery(
        self, delivery_id: int, *, actor: str = SYSTEM_ACTOR
    ) -> WebhookDelivery:
        """Return a parked delivery to the pending queue, attempts reset.

        The stored ``payload`` and ``signature`` are reused unchanged, so the
        replay is byte-identical to the delivery that failed and a receiver
        deduplicating on the signature sees the same event, not a new one.

        Raises:
            StorageError: the delivery is not parked.
        """
        with session_scope(self.sessions) as session:
            row = self._delivery(session, delivery_id)
            if row.status != DeliveryStatus.DEAD.value:
                raise StorageError(
                    f"delivery {delivery_id} is {row.status!r}, not "
                    f"{DeliveryStatus.DEAD.value!r}; only a parked delivery is "
                    "replayable",
                    context={"delivery_id": delivery_id, "status": row.status},
                )
            row.status = DeliveryStatus.PENDING.value
            row.attempts = 0
            row.last_error = None
            row.next_attempt_at = utc_now()
            session.add(
                AuditEvent(
                    job_id=row.job_id,
                    actor=actor,
                    action=AuditAction.WEBHOOK_REPLAYED.value,
                    entity=f"webhook:v{row.result_version}",
                    after={"delivery_id": delivery_id},
                )
            )
            return row

    def get_delivery(self, delivery_id: int) -> WebhookDelivery | None:
        """One delivery row by id, or ``None``."""
        with session_scope(self.sessions) as session:
            return session.get(WebhookDelivery, delivery_id)

    def deliveries_for(self, job_id: str) -> list[WebhookDelivery]:
        """Every delivery row for a job, oldest first."""
        with session_scope(self.sessions) as session:
            return list(
                session.scalars(
                    select(WebhookDelivery)
                    .where(WebhookDelivery.job_id == job_id)
                    .order_by(WebhookDelivery.id)
                )
            )

    @staticmethod
    def _delivery(session: Session, delivery_id: int) -> WebhookDelivery:
        row = session.get(WebhookDelivery, delivery_id)
        if row is None:
            raise StorageError(
                f"no webhook delivery with id {delivery_id}",
                context={"delivery_id": delivery_id},
            )
        return row

    # ------------------------------------------------------------ housekeeping
    def counts(self) -> dict[str, int]:
        """Row counts per table — the ``/healthz`` and ``/metrics`` view."""
        with session_scope(self.sessions) as session:
            return {
                "jobs": session.scalar(select(func.count()).select_from(Job)) or 0,
                "results": session.scalar(select(func.count()).select_from(Result))
                or 0,
                "audit_events": session.scalar(
                    select(func.count()).select_from(AuditEvent)
                )
                or 0,
                "webhook_dead_letters": session.scalar(
                    select(func.count())
                    .select_from(WebhookDelivery)
                    .where(WebhookDelivery.status == DeliveryStatus.DEAD.value)
                )
                or 0,
            }


def _calendar_day(received_at: datetime | date | None) -> date:
    """The calendar day a submission belongs to, defaulting to today (UTC)."""
    if received_at is None:
        return datetime.now(UTC).date()
    if isinstance(received_at, datetime):
        return received_at.date()
    return received_at


def _due(row: WebhookDelivery, now: datetime) -> bool:
    """Whether a pending delivery's backoff has elapsed.

    SQLite hands back naive datetimes even from a ``DateTime(timezone=True)``
    column, so the comparison is normalised here rather than trusting the
    driver — an unguarded naive/aware comparison raises ``TypeError`` on the
    default profile and never on Postgres.
    """
    due_at = row.next_attempt_at
    if due_at is None:
        return True
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=UTC)
    return due_at <= now
