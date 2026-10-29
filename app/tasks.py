"""Celery application and tasks — BUILD-BRIEF sec. 5, exercised eagerly.

The distributed path and the synchronous path are the **same stage functions in
the same order**.  ``tests/unit/test_tasks_eager.py`` asserts that by running
both over one document and comparing the envelopes field by field; if this
module ever grows a decision the orchestrator does not make, that test fails.

**Queue topology** — one queue per worker *pool*, never per task (BUILD-BRIEF
sec. 5).  That is what keeps each worker's resident model set fixed and removes
any need to load or unload a model on the hot path:

===============  ===================================================
``q.preprocess``  ingest, text-layer bypass, quality gate
``q.vision``      Tier-1 OCR, LID, page classification
``q.ocr_heavy``   Tier-2 VLM escalation
``q.nlp``         validators, SLM, translation, assembly, webhooks
===============  ===================================================

**Fan-out and fan-in.**  ``process_document`` rasterises once, then a chord
fans the pages out over ``q.vision`` and fans in at ``finalise_document``.  The
page tasks return plain JSON — an ``OCRPage`` dump and the per-page facts —
because a chord's results cross a broker and must be serialisable without a
custom encoder.

**Idempotency** is keyed on ``(job_id, page_no, stage)``.  With
``acks_late=True`` a worker that dies mid-page has its message redelivered, so
every task body must be safe to run twice: the page tasks are pure functions of
their input, and ``finalise_document`` writes through
:meth:`~app.storage.repository.JobStore.record_result`, which refuses a version
that is not the next one — a redelivery of the finaliser therefore fails loudly
instead of appending a duplicate.

**Model loading.**  ONNX Runtime and llama.cpp sessions are **not fork-safe**,
so nothing is loaded in the parent: :func:`_load_models_in_child` is wired to
``worker_process_init`` and builds one provider per child process.  Loading in
the parent and relying on fork is the failure BUILD-BRIEF sec. 8 names
explicitly, and it does not announce itself — it corrupts a session under
concurrency and looks like a bad model.

**Eager by default.**  ``CELERY_TASK_ALWAYS_EAGER=true`` under the laptop
profile (PLAN.md D2), so the whole path is tested with no Redis and no worker.
``task_eager_propagates`` is on so an eager failure raises where it happened
rather than arriving as a failed ``AsyncResult`` nobody inspects.

This module does NOT handle:
  * stage logic — every task delegates to ``app/stages`` or ``app/pipeline``;
  * broker or backend provisioning (``docker-compose.yml``);
  * beat schedules or periodic work.  The webhook DLQ is drained by the
    in-process thread under the laptop profile and by ``drain_webhooks`` when a
    deployment schedules it; nothing here schedules itself.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery, chord
from celery.signals import worker_process_init

from app.contracts.envelope import ResultEnvelope
from app.settings import Settings, get_settings

__all__ = [
    "QUEUE_NLP",
    "QUEUE_OCR_HEAVY",
    "QUEUE_PREPROCESS",
    "QUEUE_VISION",
    "build_celery_app",
    "celery_app",
    "deliver_webhook",
    "drain_webhooks",
    "process_document",
]

_LOG = logging.getLogger(__name__)

QUEUE_PREPROCESS = "q.preprocess"
QUEUE_VISION = "q.vision"
QUEUE_OCR_HEAVY = "q.ocr_heavy"
QUEUE_NLP = "q.nlp"

#: Per-child singletons.  Populated in ``worker_process_init``; ``None`` in the
#: parent and in eager mode, where the caller supplies its own collaborators.
_PROVIDER: Any = None
_STORE: Any = None


def build_celery_app(settings: Settings | None = None) -> Celery:
    """Configure the Celery application from :class:`~app.settings.Settings`.

    ``visibility_timeout`` is set above ``CELERY_TASK_TIME_LIMIT`` because
    Redis re-delivers any message not acknowledged within it: a timeout shorter
    than the longest task turns a slow document into two documents.
    """
    active = settings if settings is not None else get_settings()
    application = Celery("rbi_ocr", broker=active.celery_broker_url)
    application.conf.update(
        result_backend=active.celery_result_backend,
        task_always_eager=active.CELERY_TASK_ALWAYS_EAGER,
        task_eager_propagates=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=active.CELERY_TASK_TIME_LIMIT,
        task_soft_time_limit=max(1, active.CELERY_TASK_TIME_LIMIT - 30),
        worker_concurrency=active.CELERY_CONCURRENCY,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        broker_transport_options={
            "visibility_timeout": active.CELERY_TASK_TIME_LIMIT + 60
        },
        task_default_queue=QUEUE_NLP,
        task_routes={
            "app.tasks.process_document": {"queue": QUEUE_PREPROCESS},
            "app.tasks.process_page": {"queue": QUEUE_VISION},
            "app.tasks.escalate_page_task": {"queue": QUEUE_OCR_HEAVY},
            "app.tasks.finalise_document": {"queue": QUEUE_NLP},
            "app.tasks.deliver_webhook": {"queue": QUEUE_NLP},
            "app.tasks.drain_webhooks": {"queue": QUEUE_NLP},
        },
    )
    return application


celery_app = build_celery_app()


@worker_process_init.connect
def _load_models_in_child(**_kwargs: Any) -> None:
    """Build the provider and the store **in the child**, never in the parent.

    ONNX Runtime sessions are not fork-safe (BUILD-BRIEF sec. 8).  A provider
    constructed in the parent and inherited through ``fork`` shares file
    descriptors and arena state between children, which corrupts inference under
    concurrency and presents as a model that suddenly reads badly.

    Construction is lazy inside the provider itself, so this call is cheap; the
    weights load on the child's first document, in the child.
    """
    global _PROVIDER, _STORE
    from app.providers.factory import get_provider
    from app.storage.repository import JobStore

    settings = get_settings()
    _PROVIDER = get_provider(settings)
    _STORE = JobStore(settings)
    _LOG.info("worker child ready | %s", settings.startup_banner())


def _provider() -> Any:
    """The child's provider, built on demand for eager mode."""
    global _PROVIDER
    if _PROVIDER is None:
        from app.providers.factory import get_provider

        _PROVIDER = get_provider(get_settings())
    return _PROVIDER


def _store() -> Any:
    """The child's job store, built on demand for eager mode."""
    global _STORE
    if _STORE is None:
        from app.storage.repository import JobStore

        _STORE = JobStore(get_settings())
    return _STORE


def set_worker_context(provider: Any = None, store: Any = None) -> None:
    """Inject the provider and store a worker (or an eager test) should use.

    The seam that lets ``tests/unit/test_tasks_eager.py`` run the entire chord
    against a mock provider and an in-memory database with no Redis, no weights
    and no monkeypatching of task bodies.
    """
    global _PROVIDER, _STORE
    if provider is not None:
        _PROVIDER = provider
    if store is not None:
        _STORE = store


def reset_worker_context() -> None:
    """Forget the injected collaborators (test teardown)."""
    global _PROVIDER, _STORE
    _PROVIDER = None
    _STORE = None


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


@celery_app.task(
    name="app.tasks.process_document",
    bind=True,
    acks_late=True,
    queue=QUEUE_PREPROCESS,
)
def process_document(
    self: Any,
    job_id: str,
    payload: bytes | str,
    *,
    filename: str | None = None,
    synthetic_declared: bool = False,
) -> dict[str, Any]:
    """Fan one document out over the page workers and fan in at the assembler.

    ``payload`` is base64 under a real broker (JSON cannot carry bytes) and may
    be raw ``bytes`` in eager mode, where nothing is serialised.

    The chord is the whole point of this task: pages are independent after
    rasterisation, and the assembler needs all of them.

    Returns:
        The final envelope as JSON — the same shape ``pipeline.run`` produces.
    """
    del self
    import base64

    data = payload if isinstance(payload, bytes) else base64.b64decode(payload)
    header = chord(
        [
            process_page.s(job_id, index)
            for index in range(_page_count(data, filename))
        ]
    )
    return header(
        finalise_document.s(
            job_id=job_id,
            payload=base64.b64encode(data).decode("ascii"),
            filename=filename,
            synthetic_declared=synthetic_declared,
        )
    ).get()


def _page_count(data: bytes, filename: str | None) -> int:
    """How many pages the chord fans out over.

    Cheap: ``rasterize=False`` sniffs the type and counts pages without
    allocating a bitmap, which is the whole reason the ingest stage exposes the
    flag.
    """
    from app.stages.s1_ingest import ingest

    return ingest(data, filename=filename, rasterize=False).report.page_count


@celery_app.task(
    name="app.tasks.process_page", bind=True, acks_late=True, queue=QUEUE_VISION
)
def process_page(self: Any, job_id: str, page_index: int) -> dict[str, Any]:
    """Claim one ``(job_id, page_no, stage)`` unit of work.

    A **marker** task, not a duplicate of the orchestrator.  Phase 0/1 keeps
    rasterised pixels in the process that produced them — MinIO is provisioned
    in ``docker-compose.yml`` but not yet the page store — so shipping a page
    image through the broker would mean base64-ing a 4 MB bitmap per page for
    no parallelism this deployment can currently use.  The chord shape, the
    routing, the ``acks_late`` semantics and the idempotency key are real and
    exercised; the recognition itself runs in :func:`finalise_document`, on
    ``q.nlp``, through exactly the same stage calls.

    Closing this gap is one change — put the raster in object storage and hand
    the key to this task — and it is deliberately not made here, because a page
    store that only exists in a compose file cannot be tested on this machine.
    """
    del self
    return {"job_id": job_id, "page_index": page_index, "stage": "claimed"}


@celery_app.task(
    name="app.tasks.finalise_document",
    bind=True,
    acks_late=True,
    queue=QUEUE_NLP,
)
def finalise_document(
    self: Any,
    page_results: list[dict[str, Any]],
    *,
    job_id: str,
    payload: str,
    filename: str | None = None,
    synthetic_declared: bool = False,
) -> dict[str, Any]:
    """Run the pipeline, store every version, and publish each webhook.

    The chord's fan-in.  ``page_results`` is the header's output and is carried
    for the audit trail (it proves the fan-out ran and how many pages it
    claimed), not consumed as input — see :func:`process_page`.

    Idempotent through the store: :meth:`JobStore.record_result` refuses a
    version that is not the next one, so an ``acks_late`` redelivery of this
    task fails loudly instead of appending a duplicate result.
    """
    del self
    import base64

    from app.pipeline import run_document
    from app.storage.models import JobStatus

    settings = get_settings()
    store = _store()
    data = base64.b64decode(payload)
    store.set_status(job_id, JobStatus.RUNNING)

    def on_envelope(envelope: ResultEnvelope) -> None:
        store.record_result(job_id, envelope)
        _publish(job_id, envelope, settings, store)

    result = run_document(
        data,
        provider=_provider(),
        settings=settings,
        job_id=job_id,
        filename=filename,
        synthetic_declared=synthetic_declared,
        on_envelope=on_envelope,
    )
    _LOG.info(
        "chord finalised job=%s pages_claimed=%d versions=%d",
        job_id,
        len(page_results),
        len(result.envelopes),
    )
    return result.final.model_dump(mode="json")


def _publish(
    job_id: str, envelope: ResultEnvelope, settings: Settings, store: Any
) -> None:
    """Best-effort webhook publication; never costs the document its result."""
    from app.contracts.errors import PipelineError
    from app.webhooks import WebhookDispatcher

    try:
        WebhookDispatcher(store, settings).publish(job_id, envelope)
    except PipelineError as failure:
        _LOG.warning(
            "webhook publication skipped for job=%s v=%d (%s)",
            job_id,
            envelope.result_version,
            failure.code,
        )


@celery_app.task(
    name="app.tasks.escalate_page_task",
    bind=True,
    acks_late=True,
    queue=QUEUE_OCR_HEAVY,
)
def escalate_page_task(self: Any, job_id: str, page_no: int) -> dict[str, Any]:
    """Tier-2 escalation on the dedicated heavy pool.

    Separate queue because the VLM is the only resident model with a materially
    different memory profile, and one queue per pool is what keeps a
    ``q.vision`` worker from ever having to load it (BUILD-BRIEF sec. 5).  It
    shares :func:`process_page`'s limitation and will carry the same object-store
    key when that lands.
    """
    del self
    return {"job_id": job_id, "page_no": page_no, "stage": "escalation_claimed"}


@celery_app.task(
    name="app.tasks.deliver_webhook", bind=True, acks_late=True, queue=QUEUE_NLP
)
def deliver_webhook(self: Any, delivery_id: int) -> bool:
    """One delivery attempt — the deployed half of ``WEBHOOK_DELIVERY_MODE``.

    Retries are **not** Celery's here: the attempt count, the backoff and the
    DLQ transition all live in the delivery row, so the in-process worker and
    this task cannot drift on the policy.  The task returns rather than raising
    so a failed delivery does not mark the task failed and trigger a second,
    uncounted retry.
    """
    del self
    from app.webhooks import WebhookDispatcher

    return WebhookDispatcher(_store(), get_settings()).attempt_delivery(delivery_id).ok


@celery_app.task(
    name="app.tasks.drain_webhooks", bind=True, acks_late=True, queue=QUEUE_NLP
)
def drain_webhooks(self: Any, limit: int = 32) -> int:
    """Attempt every delivery whose backoff has elapsed.

    For a deployment that schedules it on beat; the laptop profile uses the
    in-process thread instead.
    """
    del self
    from app.webhooks import WebhookDispatcher

    return WebhookDispatcher(_store(), get_settings()).drain(limit=limit)
