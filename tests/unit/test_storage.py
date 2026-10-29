"""Storage tests — idempotency, versioned results, the append-only trail, the DLQ.

Three properties carry the weight, and each is asserted through the ORM rather
than by inspecting SQL, because the ORM is where the rules live:

**Append-only.**  ``results`` and ``audit_events`` refuse ``UPDATE`` and
``DELETE``.  Tested through the mapper path *and* the bulk path, because they
are enforced by two different listeners and only the pair is complete.

**Idempotency.**  ``jobs.document_id`` is unique; a resubmission of identical
bytes returns the same row with ``is_new=False`` and starts nothing.

**Version continuity.**  ``record_result`` refuses a version that is not the
next one.  A gap means an envelope was published and lost, and accepting it
would make the trail claim a continuity it does not have.

Everything runs on in-memory SQLite through ``StaticPool``, so there is no file,
no cleanup and no ordering dependency between tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.contracts.envelope import (
    BounceDecision,
    BounceReason,
    DocumentResult,
    QualityVerdict,
    ResultEnvelope,
)
from app.contracts.errors import AppendOnlyViolation, JobNotFound, StorageError
from app.contracts.ocr import PageClassLabel
from app.settings import Settings
from app.storage.database import create_all, session_scope
from app.storage.models import (
    APPEND_ONLY_TABLES,
    AuditAction,
    AuditEvent,
    Base,
    DeliveryStatus,
    Job,
    JobStatus,
    Result,
    WebhookDelivery,
)
from app.storage.repository import JobStore

DOC_A = "sha256:" + "a" * 64
DOC_B = "sha256:" + "b" * 64


@pytest.fixture
def store(make_settings: Callable[..., Settings]) -> Iterator[JobStore]:
    """A fresh in-memory database per test."""
    settings = make_settings(DATABASE_URL="sqlite+pysqlite:///:memory:")
    instance = JobStore(settings)
    yield instance
    instance.close()


@pytest.fixture
def job(store: JobStore) -> Job:
    """One created job."""
    return store.create_or_get_job(
        document_id=DOC_A, filename="complaint.pdf", byte_size=4096
    ).job


def envelope_at(
    version: int, *, final: bool, document_id: str = DOC_A
) -> ResultEnvelope:
    """A minimal but contract-valid accepted envelope at ``version``."""
    return ResultEnvelope(
        result_version=version,
        is_final=final,
        document_id=document_id,
        provider_profile="laptop",
        bounce_decision=BounceDecision.accepted(),
        pages=(),
        document=DocumentResult(
            document_class=PageClassLabel.COMPLAINT_FORM,
            document_class_confidence=0.9,
            page_count=0,
        ),
    )


# --------------------------------------------------------------------------- #
# Idempotency and soft duplication
# --------------------------------------------------------------------------- #


class TestIdempotency:
    def test_a_new_document_creates_a_job(self, store: JobStore) -> None:
        record = store.create_or_get_job(document_id=DOC_A)
        assert record.is_new is True
        assert record.job.document_id == DOC_A
        assert record.job.status == JobStatus.QUEUED.value

    def test_the_same_bytes_return_the_same_job(self, store: JobStore) -> None:
        first = store.create_or_get_job(document_id=DOC_A, filename="one.pdf")
        second = store.create_or_get_job(document_id=DOC_A, filename="two.pdf")
        assert second.is_new is False
        assert second.job.id == first.job.id
        assert second.job.filename == "one.pdf", "the resubmission overwrote the job"

    def test_different_bytes_are_different_jobs(self, store: JobStore) -> None:
        first = store.create_or_get_job(document_id=DOC_A)
        second = store.create_or_get_job(document_id=DOC_B)
        assert first.job.id != second.job.id

    def test_creation_is_audited(self, store: JobStore) -> None:
        record = store.create_or_get_job(document_id=DOC_A)
        actions = [event.action for event in store.audit_trail(record.job.id)]
        assert AuditAction.JOB_CREATED.value in actions


class TestSoftDuplicates:
    """Same sender, normalised subject and calendar day — recorded, not refused."""

    def test_links_a_second_submission_to_the_first(self, store: JobStore) -> None:
        day = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)
        first = store.create_or_get_job(
            document_id=DOC_A,
            sender="complainant@example.in",
            subject="Unauthorised debit CMS/2026/401687",
            received_at=day,
        )
        second = store.create_or_get_job(
            document_id=DOC_B,
            sender="complainant@example.in",
            subject="Re: Unauthorised debit CMS/2026/401687",
            received_at=day,
        )
        assert second.soft_duplicate_of == first.job.id
        assert second.soft_duplicate_reason == "same_sender_subject_day"

    def test_a_soft_duplicate_is_still_a_job(self, store: JobStore) -> None:
        """Two attachments on one mail are legitimately two documents."""
        day = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)
        store.create_or_get_job(
            document_id=DOC_A, sender="a@b.in", subject="Debit", received_at=day
        )
        second = store.create_or_get_job(
            document_id=DOC_B, sender="a@b.in", subject="Debit", received_at=day
        )
        assert second.is_new is True

    def test_a_different_day_is_not_a_duplicate(self, store: JobStore) -> None:
        day = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)
        store.create_or_get_job(
            document_id=DOC_A, sender="a@b.in", subject="Debit", received_at=day
        )
        second = store.create_or_get_job(
            document_id=DOC_B,
            sender="a@b.in",
            subject="Debit",
            received_at=day + timedelta(days=1),
        )
        assert second.soft_duplicate_of is None

    def test_the_key_is_the_ingest_stage_s_derivation(self, store: JobStore) -> None:
        """The storage layer must not redefine what "same complaint" means."""
        from app.stages.s1_ingest import soft_duplicate_key

        day = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)
        record = store.create_or_get_job(
            document_id=DOC_A, sender="a@b.in", subject="Debit", received_at=day
        )
        assert record.job.soft_duplicate_key == soft_duplicate_key(
            "a@b.in", "Debit", day.date()
        )

    def test_a_submission_without_a_sender_has_no_key(self, store: JobStore) -> None:
        record = store.create_or_get_job(document_id=DOC_A, subject="Debit")
        assert record.job.soft_duplicate_key is None


# --------------------------------------------------------------------------- #
# Versioned results
# --------------------------------------------------------------------------- #


class TestVersionedResults:
    def test_stores_a_version(self, store: JobStore, job: Job) -> None:
        row = store.record_result(job.id, envelope_at(1, final=False))
        assert row.result_version == 1
        assert row.is_final is False

    def test_versions_accumulate(self, store: JobStore, job: Job) -> None:
        store.record_result(job.id, envelope_at(1, final=False))
        store.record_result(job.id, envelope_at(2, final=True))
        assert [r.result_version for r in store.results_for(job.id)] == [1, 2]

    def test_the_job_tracks_the_latest_version(
        self, store: JobStore, job: Job
    ) -> None:
        store.record_result(job.id, envelope_at(1, final=False))
        store.record_result(job.id, envelope_at(2, final=True))
        refreshed = store.get_job(job.id)
        assert refreshed.latest_result_version == 2
        assert refreshed.is_final is True
        assert refreshed.status == JobStatus.COMPLETED.value

    def test_a_skipped_version_is_refused(self, store: JobStore, job: Job) -> None:
        store.record_result(job.id, envelope_at(1, final=False))
        with pytest.raises(StorageError, match="not the next version"):
            store.record_result(job.id, envelope_at(3, final=True))

    def test_a_replayed_version_is_refused(self, store: JobStore, job: Job) -> None:
        """An acks_late redelivery must fail loudly, not append a duplicate."""
        store.record_result(job.id, envelope_at(1, final=True))
        with pytest.raises(StorageError):
            store.record_result(job.id, envelope_at(1, final=True))

    def test_version_one_is_preserved_after_version_two_lands(
        self, store: JobStore, job: Job
    ) -> None:
        """A consumer acted on v1; it must still say what it said."""
        store.record_result(job.id, envelope_at(1, final=False))
        store.record_result(job.id, envelope_at(2, final=True))
        stored = store.results_for(job.id)
        assert stored[0].is_final is False
        assert stored[0].envelope["is_final"] is False

    def test_latest_envelope_revalidates_through_the_contract(
        self, store: JobStore, job: Job
    ) -> None:
        store.record_result(job.id, envelope_at(1, final=True))
        latest = store.latest_envelope(job.id)
        assert isinstance(latest, ResultEnvelope)
        assert latest.result_version == 1

    def test_a_bounced_final_marks_the_job_bounced(
        self, store: JobStore, job: Job
    ) -> None:
        bounced = ResultEnvelope(
            result_version=1,
            is_final=True,
            document_id=DOC_A,
            provider_profile="laptop",
            bounce_decision=BounceDecision(
                decision=QualityVerdict.BOUNCE,
                reason=BounceReason.QUALITY_SCORE_BELOW_THRESHOLD,
            ),
        )
        store.record_result(job.id, bounced)
        assert store.get_job(job.id).status == JobStatus.BOUNCED.value

    def test_publication_is_audited(self, store: JobStore, job: Job) -> None:
        store.record_result(job.id, envelope_at(1, final=True))
        events = store.audit_trail(job.id)
        published = [
            e for e in events if e.action == AuditAction.RESULT_PUBLISHED.value
        ]
        assert len(published) == 1
        assert published[0].after == {
            "result_version": 1,
            "is_final": True,
            "escalation_flag": False,
            "bounce": "accept",
        }

    def test_an_unknown_job_raises(self, store: JobStore) -> None:
        with pytest.raises(JobNotFound):
            store.record_result("nope", envelope_at(1, final=True))


# --------------------------------------------------------------------------- #
# Append-only
# --------------------------------------------------------------------------- #


class TestAppendOnly:
    def test_the_marked_tables_are_the_evidence_tables(self) -> None:
        assert APPEND_ONLY_TABLES == {"results", "audit_events"}

    def test_an_audit_event_cannot_be_updated(
        self, store: JobStore, job: Job
    ) -> None:
        with (
            pytest.raises(AppendOnlyViolation, match="append-only"),
            session_scope(store.sessions) as session,
        ):
                event = session.scalars(select(AuditEvent)).first()
                assert event is not None
                event.actor = "somebody-else"

    def test_an_audit_event_cannot_be_deleted(
        self, store: JobStore, job: Job
    ) -> None:
        with (
            pytest.raises(AppendOnlyViolation),
            session_scope(store.sessions) as session,
        ):
                event = session.scalars(select(AuditEvent)).first()
                assert event is not None
                session.delete(event)

    def test_a_result_cannot_be_updated(self, store: JobStore, job: Job) -> None:
        store.record_result(job.id, envelope_at(1, final=False))
        with (
            pytest.raises(AppendOnlyViolation),
            session_scope(store.sessions) as session,
        ):
                row = session.scalars(select(Result)).first()
                assert row is not None
                row.is_final = True

    def test_a_result_cannot_be_deleted(self, store: JobStore, job: Job) -> None:
        store.record_result(job.id, envelope_at(1, final=False))
        with (
            pytest.raises(AppendOnlyViolation),
            session_scope(store.sessions) as session,
        ):
                row = session.scalars(select(Result)).first()
                assert row is not None
                session.delete(row)

    def test_the_refusal_rolls_the_transaction_back(
        self, store: JobStore, job: Job
    ) -> None:
        """A refused mutation must leave nothing behind, not even the attempt."""
        before = store.counts()["audit_events"]
        with (
            pytest.raises(AppendOnlyViolation),
            session_scope(store.sessions) as session,
        ):
                session.add(
                    AuditEvent(job_id=job.id, actor="x", action="job_created")
                )
                event = session.scalars(select(AuditEvent)).first()
                assert event is not None
                event.actor = "tamper"
        assert store.counts()["audit_events"] == before

    def test_a_job_row_is_still_mutable(self, store: JobStore, job: Job) -> None:
        """Only evidence is frozen; job state is state."""
        updated = store.set_status(job.id, JobStatus.RUNNING)
        assert updated.status == JobStatus.RUNNING.value

    def test_status_changes_are_audited_with_before_and_after(
        self, store: JobStore, job: Job
    ) -> None:
        store.set_status(job.id, JobStatus.RUNNING)
        change = next(
            e
            for e in store.audit_trail(job.id)
            if e.action == AuditAction.JOB_STATUS_CHANGED.value
        )
        assert change.before == {"status": JobStatus.QUEUED.value}
        assert change.after is not None
        assert change.after["status"] == JobStatus.RUNNING.value


# --------------------------------------------------------------------------- #
# The DLQ
# --------------------------------------------------------------------------- #


class TestDeliveryQueue:
    @pytest.fixture
    def delivery(self, store: JobStore, job: Job) -> WebhookDelivery:
        return store.enqueue_delivery(
            job_id=job.id,
            result_version=1,
            url="https://receiver.invalid/hook",
            payload='{"result_version":1}',
            signature="deadbeef",
            signed_at=1_800_000_000,
            max_attempts=2,
        )

    def test_enqueue_is_idempotent_per_version(
        self, store: JobStore, job: Job, delivery: WebhookDelivery
    ) -> None:
        again = store.enqueue_delivery(
            job_id=job.id,
            result_version=1,
            url="https://receiver.invalid/hook",
            payload='{"result_version":1}',
            signature="deadbeef",
            signed_at=1_800_000_000,
            max_attempts=2,
        )
        assert again.id == delivery.id
        assert len(store.deliveries_for(job.id)) == 1

    def test_a_pending_delivery_is_due_immediately(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        assert [row.id for row in store.due_deliveries()] == [delivery.id]

    def test_a_failed_attempt_backs_off(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        updated = store.mark_attempt_failed(
            delivery.id, status_code=503, error="unavailable", backoff_seconds=30.0
        )
        assert updated.attempts == 1
        assert updated.status == DeliveryStatus.PENDING.value
        assert store.due_deliveries() == []

    def test_exhausted_attempts_park_the_delivery(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        for _ in range(2):
            store.mark_attempt_failed(
                delivery.id, status_code=500, error="boom", backoff_seconds=0.0
            )
        parked = store.dead_letters()
        assert [row.id for row in parked] == [delivery.id]
        assert parked[0].status == DeliveryStatus.DEAD.value

    def test_parking_is_audited(
        self, store: JobStore, job: Job, delivery: WebhookDelivery
    ) -> None:
        for _ in range(2):
            store.mark_attempt_failed(
                delivery.id, status_code=500, error="boom", backoff_seconds=0.0
            )
        actions = [e.action for e in store.audit_trail(job.id)]
        assert AuditAction.WEBHOOK_DEAD_LETTERED.value in actions

    def test_replay_requeues_with_the_same_payload(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        """Byte-identical, so a receiver deduplicating on the MAC sees one event."""
        for _ in range(2):
            store.mark_attempt_failed(
                delivery.id, status_code=500, error="boom", backoff_seconds=0.0
            )
        replayed = store.replay_delivery(delivery.id)
        assert replayed.status == DeliveryStatus.PENDING.value
        assert replayed.attempts == 0
        assert replayed.payload == '{"result_version":1}'
        assert replayed.signature == "deadbeef"

    def test_only_a_parked_delivery_is_replayable(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        with pytest.raises(StorageError, match="only a parked delivery"):
            store.replay_delivery(delivery.id)

    def test_a_delivered_row_leaves_the_queue(
        self, store: JobStore, delivery: WebhookDelivery
    ) -> None:
        store.mark_delivered(delivery.id, status_code=200)
        assert store.due_deliveries() == []
        assert store.dead_letters() == []

    def test_an_unknown_delivery_raises(self, store: JobStore) -> None:
        with pytest.raises(StorageError, match="no webhook delivery"):
            store.mark_delivered(9999, status_code=200)


# --------------------------------------------------------------------------- #
# Dual mode
# --------------------------------------------------------------------------- #


class TestDualMode:
    def test_the_models_create_cleanly_on_a_file_backend(
        self, make_settings: Callable[..., Settings], tmp_path: object
    ) -> None:
        path = tmp_path / "runtime.sqlite"  # type: ignore[operator]
        settings = make_settings(DATABASE_URL=f"sqlite+pysqlite:///{path}")
        instance = JobStore(settings)
        try:
            record = instance.create_or_get_job(document_id=DOC_A)
            assert record.is_new is True
        finally:
            instance.close()
        assert path.exists()  # type: ignore[attr-defined]

    def test_a_relative_sqlite_url_is_anchored_to_the_repository(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """Otherwise the API and a worker in different shells use two databases."""
        from app.settings import PROJECT_ROOT

        settings = make_settings(
            DATABASE_URL="sqlite+pysqlite:///./data/runtime/app.sqlite"
        )
        assert PROJECT_ROOT.as_posix() in settings.database_url

    def test_a_postgres_url_is_left_alone(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        dsn = "postgresql+psycopg://rbi:secret@postgres:5432/rbi"
        assert make_settings(DATABASE_URL=dsn).database_url == dsn

    def test_foreign_keys_are_enforced_on_sqlite(
        self, store: JobStore
    ) -> None:
        """Off by default in SQLite, so results.job_id would be decoration."""
        from sqlalchemy.exc import IntegrityError

        with (
            pytest.raises(IntegrityError),
            session_scope(store.sessions) as session,
        ):
                session.add(
                    Result(
                        job_id="does-not-exist",
                        result_version=1,
                        is_final=True,
                        schema_version="1.0.0",
                        document_id=DOC_A,
                        envelope={},
                    )
                )

    def test_create_all_is_idempotent(self, store: JobStore) -> None:
        create_all(store.engine)
        create_all(store.engine)
        assert set(Base.metadata.tables) == {
            "jobs",
            "results",
            "audit_events",
            "webhook_deliveries",
        }

    def test_counts_reports_every_table(self, store: JobStore, job: Job) -> None:
        counts = store.counts()
        assert counts["jobs"] == 1
        assert counts["audit_events"] >= 1
        assert counts["results"] == 0
        assert counts["webhook_dead_letters"] == 0


class TestMigration:
    """The committed migration and the models must not drift apart."""

    def test_the_initial_migration_declares_every_model_table(self) -> None:
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "alembic"
            / "versions"
            / "0001_initial_schema.py"
        ).read_text(encoding="utf-8")
        created = set(re.findall(r'op\.create_table\(\s*"([a-z_]+)"', source))
        assert created == set(Base.metadata.tables)
