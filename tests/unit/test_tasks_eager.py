"""Celery tests — the eager chord produces the same envelope as the sync pipeline.

That equality is the point of the whole file.  ``app/tasks.py`` exists so the
work can be distributed; the moment it becomes a *second implementation* of the
pipeline, the two drift and the deployed behaviour stops being the behaviour
that was tested.  :meth:`TestChordEquivalence.test_the_chord_and_the_pipeline_agree`
compares the two envelopes field by field, with only the fields that cannot be
equal — timings, timestamps, the job id — excluded and named.

Everything runs with ``task_always_eager``, so there is no Redis, no worker and
no broker.  ``task_eager_propagates`` is on, so a failure inside a task raises
where it happened rather than arriving as a failed ``AsyncResult`` nobody looks
at.

The queue routing is asserted directly rather than through a running worker:
one queue per pool is a *configuration* property (BUILD-BRIEF sec. 5), and a
routing table that names the wrong queue is exactly the kind of thing that only
shows up in production, where the wrong worker loads the wrong model.
"""

from __future__ import annotations

import base64
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from app.contracts.envelope import ResultEnvelope
from app.pipeline import run_document
from app.settings import Settings
from app.storage.models import JobStatus
from app.storage.repository import JobStore
from app.tasks import (
    QUEUE_NLP,
    QUEUE_OCR_HEAVY,
    QUEUE_PREPROCESS,
    QUEUE_VISION,
    build_celery_app,
    celery_app,
    process_document,
    reset_worker_context,
    set_worker_context,
)

_CONTRACT_DIR = Path(__file__).resolve().parents[1] / "contract"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from mock_provider import MockProvider

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
CLEAN_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_clean.jpg"
BOUNCE_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_bounce.jpg"

#: Fields that cannot be equal between two runs of the same document, and why.
VOLATILE = {
    "created_at",  # wall clock
    "job_id",  # a different job row per run
    "processing",  # timings and timestamps
}


@pytest.fixture
def task_settings(make_settings: Callable[..., Settings]) -> Settings:
    """Eager, in-memory, no delivery, no shadow sampling."""
    return make_settings(
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        CELERY_TASK_ALWAYS_EAGER=True,
        WEBHOOK_DELIVERY_MODE="disabled",
        ESCALATION_SHADOW_RATE=0.0,
    )


@pytest.fixture
def store(task_settings: Settings) -> Iterator[JobStore]:
    instance = JobStore(task_settings)
    yield instance
    instance.close()


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(field_values={"complainant_name": "4. Name of bank"})


@pytest.fixture
def worker_context(
    monkeypatch: pytest.MonkeyPatch,
    task_settings: Settings,
    store: JobStore,
    provider: MockProvider,
) -> Iterator[None]:
    """Wire the eager tasks to the mock provider and the in-memory store.

    ``get_settings`` is patched rather than the task bodies: the tasks read
    settings the way a worker child does, so patching the reader keeps the code
    under test identical to the code that ships.
    """
    monkeypatch.setattr("app.tasks.get_settings", lambda: task_settings)
    monkeypatch.setattr("app.settings.get_settings", lambda: task_settings)
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    set_worker_context(provider=provider, store=store)
    yield
    reset_worker_context()


def run_chord(store: JobStore, path: Path = CLEAN_DOC) -> dict[str, Any]:
    """Submit one document through the chord and return the final envelope."""
    from app.stages.s1_ingest import idempotency_key

    data = path.read_bytes()
    job = store.create_or_get_job(document_id=idempotency_key(data)).job
    return process_document.apply(
        args=(job.id, base64.b64encode(data).decode("ascii")),
        kwargs={"filename": path.name, "synthetic_declared": True},
    ).get()


def comparable(envelope: ResultEnvelope) -> dict[str, Any]:
    """The envelope minus the fields that cannot be equal across two runs."""
    payload = envelope.model_dump(mode="json")
    return {key: value for key, value in payload.items() if key not in VOLATILE}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestCeleryConfiguration:
    def test_is_eager_by_default_on_the_laptop_profile(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """PLAN.md D2: the whole path is tested without Redis."""
        application = build_celery_app(make_settings())
        assert application.conf.task_always_eager is True
        assert application.conf.task_eager_propagates is True

    def test_acks_late_with_a_visibility_timeout_above_the_task_limit(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """A shorter timeout turns one slow document into two documents."""
        settings = make_settings()
        application = build_celery_app(settings)
        assert application.conf.task_acks_late is True
        assert (
            application.conf.broker_transport_options["visibility_timeout"]
            > settings.CELERY_TASK_TIME_LIMIT
        )

    def test_one_queue_per_pool_not_per_task(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """BUILD-BRIEF sec. 5: this is what keeps a worker's model set fixed."""
        routes = build_celery_app(make_settings()).conf.task_routes
        assert routes["app.tasks.process_document"]["queue"] == QUEUE_PREPROCESS
        assert routes["app.tasks.process_page"]["queue"] == QUEUE_VISION
        assert routes["app.tasks.escalate_page_task"]["queue"] == QUEUE_OCR_HEAVY
        assert routes["app.tasks.finalise_document"]["queue"] == QUEUE_NLP
        assert {entry["queue"] for entry in routes.values()} == {
            QUEUE_PREPROCESS,
            QUEUE_VISION,
            QUEUE_OCR_HEAVY,
            QUEUE_NLP,
        }

    def test_prefetch_is_one(self, make_settings: Callable[..., Settings]) -> None:
        """Long tasks plus prefetch means one worker hoards a queue."""
        assert build_celery_app(make_settings()).conf.worker_prefetch_multiplier == 1

    def test_json_only_serialisation(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """Pickle over a broker is remote code execution with extra steps."""
        conf = build_celery_app(make_settings()).conf
        assert conf.task_serializer == "json"
        assert conf.accept_content == ["json"]

    def test_worker_concurrency_honours_the_thread_budget(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        settings = make_settings(CELERY_CONCURRENCY=2, ORT_INTRA_OP_THREADS=2)
        assert build_celery_app(settings).conf.worker_concurrency == 2
        settings.assert_thread_budget(physical_cores=4)

    def test_models_are_loaded_in_the_child_not_the_parent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ONNX sessions are not fork-safe (BUILD-BRIEF sec. 8).

        Asserted behaviourally by firing the signal Celery fires in each child
        and checking that the provider now exists — inspecting the receiver
        list is unreliable, because Celery holds its receivers weakly and
        reports them by identity rather than by name.
        """
        from celery.signals import worker_process_init

        from app import tasks

        built: list[str] = []
        monkeypatch.setattr(
            tasks, "get_settings", lambda: Settings(_env_file=None)
        )
        monkeypatch.setattr(
            "app.providers.factory.get_provider",
            lambda *_a, **_k: built.append("provider") or object(),
        )
        monkeypatch.setattr(
            "app.storage.repository.JobStore",
            lambda *_a, **_k: built.append("store") or object(),
        )
        reset_worker_context()
        try:
            worker_process_init.send(sender=None)
            assert built == ["provider", "store"]
        finally:
            reset_worker_context()


# --------------------------------------------------------------------------- #
# The chord
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("worker_context")
class TestChordEquivalence:
    def test_the_chord_produces_a_schema_valid_envelope(
        self, store: JobStore
    ) -> None:
        payload = run_chord(store)
        envelope = ResultEnvelope.model_validate(payload)
        assert envelope.is_final is True
        assert envelope.document is not None

    def test_the_chord_and_the_pipeline_agree(
        self, store: JobStore, provider: MockProvider, task_settings: Settings
    ) -> None:
        """The load-bearing assertion of this module.

        If ``app/tasks.py`` ever grows a decision the orchestrator does not
        make, the deployed behaviour stops being the behaviour that was tested,
        and this is where that shows up.
        """
        through_chord = ResultEnvelope.model_validate(run_chord(store))
        directly = run_document(
            CLEAN_DOC.read_bytes(),
            provider=MockProvider(
                field_values={"complainant_name": "4. Name of bank"}
            ),
            settings=task_settings,
            filename=CLEAN_DOC.name,
            synthetic_declared=True,
        ).final
        assert comparable(through_chord) == comparable(directly)

    def test_every_version_is_stored_by_the_finaliser(
        self, store: JobStore
    ) -> None:
        from app.stages.s1_ingest import idempotency_key

        run_chord(store)
        job = store.find_by_document_id(idempotency_key(CLEAN_DOC.read_bytes()))
        assert job is not None
        versions = [row.result_version for row in store.results_for(job.id)]
        assert versions == list(range(1, len(versions) + 1))
        assert job.status == JobStatus.COMPLETED.value

    def test_the_fan_out_claims_one_unit_per_page(
        self, store: JobStore
    ) -> None:
        """The chord header is real even though Phase 0/1 recognises in the fan-in."""
        from app.tasks import process_page

        claimed = process_page.apply(args=("job-1", 0)).get()
        assert claimed == {"job_id": "job-1", "page_index": 0, "stage": "claimed"}

    def test_a_bounced_document_goes_through_the_same_path(
        self, store: JobStore
    ) -> None:
        envelope = ResultEnvelope.model_validate(run_chord(store, BOUNCE_DOC))
        assert envelope.bounce_decision.decision.value == "bounce"
        assert envelope.document is None

    def test_the_finaliser_is_idempotent_by_refusing_a_replay(
        self, store: JobStore
    ) -> None:
        """acks_late means this task can be redelivered; it must not duplicate."""
        from app.contracts.errors import StorageError
        from app.stages.s1_ingest import idempotency_key

        run_chord(store)
        job = store.find_by_document_id(idempotency_key(CLEAN_DOC.read_bytes()))
        assert job is not None
        before = len(store.results_for(job.id))
        with pytest.raises(StorageError):
            run_chord(store)
        assert len(store.results_for(job.id)) == before


@pytest.mark.usefixtures("worker_context")
class TestWebhookTask:
    """The deployed delivery mode runs the same attempt as the thread mode."""

    def test_the_delivery_task_uses_the_dispatcher_policy(
        self,
        store: JobStore,
        task_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One retry policy for both modes, or they drift.

        The default transport is replaced at the module level rather than on an
        instance, because the task constructs its own dispatcher — which is the
        behaviour under test: a Celery worker has no object to inject into.
        """
        from app.contracts.envelope import (
            BounceDecision,
            DocumentResult,
        )
        from app.contracts.envelope import (
            ResultEnvelope as Envelope,
        )
        from app.contracts.ocr import PageClassLabel
        from app.storage.models import DeliveryStatus
        from app.tasks import deliver_webhook
        from app.webhooks import DeliveryOutcome

        settings = task_settings.model_copy(
            update={
                "WEBHOOK_URL": "https://receiver.invalid/hook",
                "WEBHOOK_DELIVERY_MODE": "celery",
                "WEBHOOK_HMAC_SECRET": "shared-secret-value",
            }
        )
        monkeypatch.setattr("app.tasks.get_settings", lambda: settings)

        sent: list[bytes] = []

        def transport(
            _url: str, body: bytes, _headers: dict[str, str], _timeout: float
        ) -> DeliveryOutcome:
            sent.append(body)
            return DeliveryOutcome(ok=True, status_code=200)

        monkeypatch.setattr("app.webhooks.httpx_transport", transport)

        document_id = "sha256:" + "d" * 64
        job = store.create_or_get_job(document_id=document_id).job
        envelope = Envelope(
            result_version=1,
            is_final=True,
            document_id=document_id,
            provider_profile="laptop",
            bounce_decision=BounceDecision.accepted(),
            document=DocumentResult(
                document_class=PageClassLabel.COMPLAINT_FORM,
                document_class_confidence=0.9,
                page_count=0,
            ),
        )
        delivery = store.enqueue_delivery(
            job_id=job.id,
            result_version=1,
            url=str(settings.WEBHOOK_URL),
            payload=envelope.model_dump_json(),
            signature="0" * 64,
            signed_at=1_800_000_000,
            max_attempts=settings.WEBHOOK_MAX_ATTEMPTS,
        )

        assert deliver_webhook.apply(args=(delivery.id,)).get() is True
        assert sent
        assert (
            store.get_delivery(delivery.id).status == DeliveryStatus.DELIVERED.value
        )
