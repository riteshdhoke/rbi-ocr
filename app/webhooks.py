"""Signed result delivery — HMAC-SHA256, exponential backoff, replayable DLQ.

BUILD-BRIEF sec. 5 makes delivery two-stage: the fast path publishes
``result_version=1, is_final=false`` and every later version fires again.  This
module is what "fires" means.

**The signature.**  ``HMAC-SHA256(secret, b"<timestamp>." + body)`` over the
**exact bytes sent**, hex, in ``X-RBI-Signature: sha256=<hex>``.  The timestamp
is inside the MAC, not merely beside it, so a captured delivery cannot be
replayed later with a fresh header; the receiver rejects anything older than
``WEBHOOK_SIGNATURE_TOLERANCE_SECONDS``.  :func:`verify_signature` is the
receiver's half and is exported so the consumer documentation can point at real
code rather than at a description of it.

**Why the body is stored.**  The DLQ keeps the serialised payload, not the
envelope, and a replay re-sends it byte-identically.  Re-serialising would
produce a different body — key order, float formatting, a ``created_at`` that
now round-trips differently — and therefore a different MAC, so a receiver
deduplicating on the signature would treat a replay as a new event.

**Two delivery modes, one code path.**  Under the laptop profile an in-process
thread drains the queue (``WEBHOOK_DELIVERY_MODE=thread``); deployed, the same
attempt runs inside a Celery task on ``q.nlp``.  Both call
:meth:`WebhookDispatcher.attempt_delivery`, so the retry policy, the backoff and
the DLQ transitions cannot diverge between them.  ``inline`` delivers on the
calling thread and is what the tests use.

**Failure is not the document's problem.**  A delivery that exhausts its
attempts is parked, and the job stays ``completed``: the envelope is stored and
``GET /v1/jobs/{id}`` serves it.  A webhook is a notification, not the result.

This module does NOT handle:
  * deciding *when* a version exists — that is ``pipeline.on_envelope``;
  * the DLQ **rows** or the audit trail (``app.storage.repository``); it owns
    the transport and the policy, the store owns the state;
  * receiver-side idempotency.  The delivery carries ``job_id`` and
    ``result_version``; deduplicating on that pair is the consumer's job and is
    documented rather than enforced from here;
  * TLS trust configuration beyond
    :func:`app.providers.base.inject_truststore` — a corporate CA is honoured,
    verification is never disabled.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Self

from prometheus_client import Counter, Histogram

from app.contracts.envelope import ResultEnvelope
from app.contracts.errors import ConfigurationError, WebhookDeliveryFailed
from app.settings import Settings, get_settings
from app.storage.models import DeliveryStatus, WebhookDelivery
from app.storage.repository import JobStore

__all__ = [
    "SIGNATURE_HEADER",
    "SIGNATURE_PREFIX",
    "TIMESTAMP_HEADER",
    "DeliveryOutcome",
    "WebhookDispatcher",
    "backoff_seconds",
    "build_headers",
    "serialise_envelope",
    "sign_payload",
    "verify_signature",
]

_LOG = logging.getLogger(__name__)

SIGNATURE_HEADER: Final[str] = "X-RBI-Signature"
TIMESTAMP_HEADER: Final[str] = "X-RBI-Timestamp"
JOB_HEADER: Final[str] = "X-RBI-Job-Id"
VERSION_HEADER: Final[str] = "X-RBI-Result-Version"
DELIVERY_HEADER: Final[str] = "X-RBI-Delivery-Id"
#: Algorithm tag, so a future rotation to a different MAC is a value change on
#: the wire rather than a silent reinterpretation of the same header.
SIGNATURE_PREFIX: Final[str] = "sha256="

_ATTEMPTS = Counter(
    "webhook_delivery_attempts_total",
    "Webhook delivery attempts.",
    ["result"],
)
_DEAD_LETTERS = Counter(
    "webhook_dead_letters_total",
    "Deliveries parked in the DLQ after exhausting their retries.",
)
_REPLAYS = Counter(
    "webhook_replays_total", "Dead-lettered deliveries returned to the queue."
)
_LATENCY = Histogram(
    "webhook_delivery_seconds",
    "Wall clock of one delivery attempt.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #


def serialise_envelope(envelope: ResultEnvelope) -> bytes:
    """The exact bytes that get signed and sent.

    ``sort_keys`` and a fixed separator set make the body a pure function of the
    envelope, so the same version signed twice produces the same MAC and a
    receiver can deduplicate on it.  ``ensure_ascii=False`` keeps Devanagari
    values readable in the receiver's log instead of expanding every character
    to six bytes of escape.
    """
    return json.dumps(
        envelope.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_payload(secret: str, payload: bytes, timestamp: int) -> str:
    """``HMAC-SHA256(secret, b"<timestamp>." + payload)`` as lowercase hex.

    Raises:
        ConfigurationError: for an empty secret — an unsigned webhook is a
            result envelope posted to whatever answers that URL.
    """
    if not secret:
        raise ConfigurationError(
            "WEBHOOK_HMAC_SECRET is empty; result delivery must be signed "
            "(BUILD-BRIEF sec. 5). Set the secret or disable delivery.",
            context={"remedy": "set WEBHOOK_HMAC_SECRET or unset WEBHOOK_URL"},
        )
    message = f"{timestamp}.".encode() + payload
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str,
    payload: bytes,
    timestamp: int,
    signature: str,
    *,
    tolerance_seconds: int = 300,
    now: int | None = None,
) -> bool:
    """The receiver's half: recompute, compare in constant time, check the age.

    Accepts the header with or without the ``sha256=`` prefix.  Comparison is
    :func:`hmac.compare_digest` — a plain ``==`` on a hex digest leaks the
    position of the first wrong byte to a patient caller.
    """
    if not secret or not signature:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - timestamp) > tolerance_seconds:
        return False
    candidate = signature.removeprefix(SIGNATURE_PREFIX).strip()
    expected = sign_payload(secret, payload, timestamp)
    return hmac.compare_digest(candidate, expected)


def build_headers(
    *,
    signature: str,
    timestamp: int,
    job_id: str,
    result_version: int,
    delivery_id: int | None = None,
) -> dict[str, str]:
    """Headers for one delivery, signature included."""
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        SIGNATURE_HEADER: f"{SIGNATURE_PREFIX}{signature}",
        TIMESTAMP_HEADER: str(timestamp),
        JOB_HEADER: job_id,
        VERSION_HEADER: str(result_version),
    }
    if delivery_id is not None:
        headers[DELIVERY_HEADER] = str(delivery_id)
    return headers


def backoff_seconds(
    attempt: int, *, base: float, maximum: float, jitter: float = 0.25
) -> float:
    """Exponential backoff with jitter, for ``attempt`` (1-based).

    Jitter is multiplicative and one-sided-up-and-down around the nominal delay.
    Without it every worker that lost the same receiver retries in lockstep and
    re-creates the outage it is backing off from.
    """
    if attempt < 1:
        raise ValueError(f"attempt is 1-based, got {attempt}")
    nominal = min(base * (2 ** (attempt - 1)), maximum)
    spread = nominal * jitter
    return max(0.0, nominal + random.uniform(-spread, spread))


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeliveryOutcome:
    """What one attempt did.  ``ok`` is the only thing the caller branches on."""

    ok: bool
    status_code: int | None = None
    error: str | None = None


#: A transport takes ``(url, body, headers, timeout)`` and reports the outcome.
#: Injected so the tests never open a socket and so a deployment can swap in a
#: client with its own proxy configuration without touching the retry policy.
Transport = Callable[[str, bytes, dict[str, str], float], DeliveryOutcome]


def httpx_transport(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> DeliveryOutcome:
    """The default transport: one POST, verification always on.

    A 2xx is success.  Everything else — including a 4xx — is a failure that
    retries, because a receiver returning 400 on a valid envelope is far more
    often a receiver that is mid-deploy than a permanently malformed body, and
    the DLQ is where a genuinely rejected delivery ends up either way.
    """
    from app.providers.base import inject_truststore

    inject_truststore()
    import httpx

    try:
        response = httpx.post(url, content=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as failure:
        return DeliveryOutcome(ok=False, error=f"{type(failure).__name__}: {failure}")
    ok = 200 <= response.status_code < 300
    return DeliveryOutcome(
        ok=ok,
        status_code=response.status_code,
        error=None if ok else f"HTTP {response.status_code}",
    )


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #


class WebhookDispatcher:
    """Signs, enqueues, delivers and retries — one policy for both modes.

    Construct one per process.  ``start()`` is only needed for the ``thread``
    mode; ``inline`` and ``celery`` need nothing running.
    """

    def __init__(
        self,
        store: JobStore,
        settings: Settings | None = None,
        *,
        transport: Transport | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        self.store = store
        self.settings = settings if settings is not None else get_settings()
        self.transport: Transport = (
            transport if transport is not None else httpx_transport
        )
        self.poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----------------------------------------------------------- publication
    def publish(self, job_id: str, envelope: ResultEnvelope) -> WebhookDelivery | None:
        """Sign and enqueue one ``result_version``; deliver per the mode.

        Returns ``None`` when delivery is not configured — that is the normal
        laptop state and must not be an error, because the envelope is stored
        and served regardless.

        Raises:
            ConfigurationError: a URL is configured without a signing secret.
        """
        if not self.settings.webhook_enabled:
            if self.settings.WEBHOOK_URL and self.settings.WEBHOOK_DELIVERY_MODE != (
                "disabled"
            ):
                raise ConfigurationError(
                    "WEBHOOK_URL is set without WEBHOOK_HMAC_SECRET; an "
                    "unsigned result delivery is refused.",
                    context={"remedy": "set WEBHOOK_HMAC_SECRET"},
                )
            return None

        secret = self.settings.WEBHOOK_HMAC_SECRET
        url = self.settings.WEBHOOK_URL
        if secret is None or url is None:  # pragma: no cover - webhook_enabled
            raise ConfigurationError(
                "webhook delivery is enabled but the URL or secret is missing",
                context={"has_url": url is not None},
            )

        body = serialise_envelope(envelope)
        timestamp = int(time.time())
        signature = sign_payload(secret.get_secret_value(), body, timestamp)
        delivery = self.store.enqueue_delivery(
            job_id=job_id,
            result_version=envelope.result_version,
            url=url,
            payload=body.decode("utf-8"),
            signature=signature,
            signed_at=timestamp,
            max_attempts=self.settings.WEBHOOK_MAX_ATTEMPTS,
        )
        mode = self.settings.WEBHOOK_DELIVERY_MODE
        if mode == "inline":
            self.attempt_delivery(delivery.id)
        elif mode == "celery":
            self._enqueue_celery(delivery.id)
        return delivery

    def _enqueue_celery(self, delivery_id: int) -> None:
        """Hand the attempt to the ``q.nlp`` worker.

        Imported lazily and failure-tolerantly: a broker that is down must not
        cost the request its response, and the row is already pending, so the
        next drain picks it up.
        """
        try:
            from app.tasks import deliver_webhook

            deliver_webhook.delay(delivery_id)
        except (ImportError, OSError, RuntimeError) as failure:
            _LOG.warning(
                "could not enqueue webhook delivery %d on the broker (%s); the "
                "row stays pending and will be retried by the next drain",
                delivery_id,
                failure,
            )

    # ------------------------------------------------------------- attempting
    def attempt_delivery(self, delivery_id: int) -> DeliveryOutcome:
        """One attempt at a pending delivery, updating its state.

        Idempotent on an already-delivered row: an ``acks_late`` Celery
        redelivery must not POST the same envelope twice.

        Raises:
            WebhookDeliveryFailed: never — it is *recorded*, not raised. The
                exception class exists for a caller that asks for the strict
                behaviour through :meth:`deliver_or_raise`.
        """
        row = self.store.get_delivery(delivery_id)
        if row is None:
            return DeliveryOutcome(ok=False, error=f"no delivery {delivery_id}")
        if row.status != DeliveryStatus.PENDING.value:
            return DeliveryOutcome(ok=True, status_code=row.last_status_code)

        body = row.payload.encode("utf-8")
        headers = build_headers(
            signature=row.signature,
            timestamp=row.signed_at,
            job_id=row.job_id,
            result_version=row.result_version,
            delivery_id=row.id,
        )
        start = time.perf_counter()
        try:
            outcome = self.transport(
                row.url, body, headers, self.settings.WEBHOOK_TIMEOUT_SECONDS
            )
        except Exception as failure:  # noqa: BLE001 - a transport may raise anything
            outcome = DeliveryOutcome(
                ok=False, error=f"{type(failure).__name__}: {failure}"
            )
        finally:
            _LATENCY.observe(time.perf_counter() - start)

        if outcome.ok:
            _ATTEMPTS.labels(result="delivered").inc()
            self.store.mark_delivered(
                delivery_id, status_code=outcome.status_code or 200
            )
            return outcome

        _ATTEMPTS.labels(result="failed").inc()
        delay = backoff_seconds(
            row.attempts + 1,
            base=self.settings.WEBHOOK_BACKOFF_BASE_SECONDS,
            maximum=self.settings.WEBHOOK_BACKOFF_MAX_SECONDS,
        )
        updated = self.store.mark_attempt_failed(
            delivery_id,
            status_code=outcome.status_code,
            error=outcome.error or "unknown transport failure",
            backoff_seconds=delay,
        )
        if updated.status == DeliveryStatus.DEAD.value:
            _DEAD_LETTERS.inc()
            _LOG.error(
                "webhook delivery %d for job %s v%d parked after %d attempts "
                "(last status %s); replay it from the DLQ",
                delivery_id,
                updated.job_id,
                updated.result_version,
                updated.attempts,
                updated.last_status_code,
            )
        return outcome

    def deliver_or_raise(self, delivery_id: int) -> None:
        """:meth:`attempt_delivery`, but a parked delivery raises.

        For a caller that wants the failure to be its problem — the replay CLI,
        and any future synchronous delivery mode.

        Raises:
            WebhookDeliveryFailed: the attempt failed.
        """
        outcome = self.attempt_delivery(delivery_id)
        if not outcome.ok:
            raise WebhookDeliveryFailed(
                f"delivery {delivery_id} failed: {outcome.error}",
                status_code=outcome.status_code,
            )

    def drain(self, *, limit: int = 32) -> int:
        """Attempt every delivery whose backoff has elapsed; return the count."""
        due = self.store.due_deliveries(limit=limit)
        for row in due:
            self.attempt_delivery(row.id)
        return len(due)

    def replay(self, delivery_id: int) -> WebhookDelivery:
        """Return a parked delivery to the queue and attempt it immediately."""
        row = self.store.replay_delivery(delivery_id)
        _REPLAYS.inc()
        self.attempt_delivery(row.id)
        return row

    def replay_all(self, *, limit: int = 100) -> int:
        """Replay every parked delivery; returns how many were requeued."""
        parked = self.store.dead_letters(limit=limit)
        for row in parked:
            self.replay(row.id)
        return len(parked)

    # ------------------------------------------------------ in-process worker
    def start(self) -> None:
        """Start the background drain thread (``thread`` mode only).

        A daemon thread: a stuck receiver must never keep the process alive
        after the API has shut down, and the queue is durable, so anything
        in flight is retried by the next process.
        """
        if self.settings.WEBHOOK_DELIVERY_MODE != "thread":
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="webhook-dispatcher", daemon=True
        )
        self._thread.start()
        _LOG.info("webhook dispatcher thread started")

    def stop(self, *, timeout: float = 5.0) -> None:
        """Ask the drain thread to finish the current attempt and exit."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.drain()
            except Exception:  # noqa: BLE001 - the worker must outlive one bad row
                _LOG.exception("webhook drain failed; continuing")
            self._stop.wait(self.poll_interval)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
