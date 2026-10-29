"""Webhook tests — the signature is verifiable, failures park, parked replays.

The transport is injected in every test, so nothing here opens a socket.  That
is not only a speed decision: the retry policy, the backoff and the DLQ
transition are the behaviour under test, and a real HTTP client would make them
a function of somebody else's timeouts.

The signature tests are written from the **receiver's** point of view on
purpose.  :func:`~app.webhooks.verify_signature` is exported so a consumer's
documentation can point at running code, and these tests are the specification
that code has to satisfy: right secret verifies, wrong secret does not, a
changed body does not, and a captured delivery replayed an hour later does not.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest

from app.contracts.envelope import (
    BounceDecision,
    DocumentResult,
    ResultEnvelope,
)
from app.contracts.errors import ConfigurationError, WebhookDeliveryFailed
from app.contracts.ocr import PageClassLabel
from app.settings import Settings
from app.storage.models import AuditAction, DeliveryStatus
from app.storage.repository import JobStore
from app.webhooks import (
    SIGNATURE_HEADER,
    SIGNATURE_PREFIX,
    TIMESTAMP_HEADER,
    DeliveryOutcome,
    WebhookDispatcher,
    backoff_seconds,
    build_headers,
    serialise_envelope,
    sign_payload,
    verify_signature,
)

SECRET = "a-shared-secret-of-reasonable-length"
DOC = "sha256:" + "c" * 64
URL = "https://receiver.invalid/rbi-hook"


def envelope(version: int = 1, *, final: bool = True) -> ResultEnvelope:
    return ResultEnvelope(
        result_version=version,
        is_final=final,
        document_id=DOC,
        provider_profile="laptop",
        bounce_decision=BounceDecision.accepted(),
        document=DocumentResult(
            document_class=PageClassLabel.COMPLAINT_FORM,
            document_class_confidence=0.9,
            page_count=0,
        ),
    )


class RecordingTransport:
    """A transport that records every call and answers from a script."""

    def __init__(self, *outcomes: DeliveryOutcome) -> None:
        self.outcomes = list(outcomes) or [DeliveryOutcome(ok=True, status_code=200)]
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> DeliveryOutcome:
        del timeout
        self.calls.append((url, body, dict(headers)))
        if len(self.outcomes) > 1:
            return self.outcomes.pop(0)
        return self.outcomes[0]


@pytest.fixture
def webhook_settings(make_settings: Callable[..., Settings]) -> Settings:
    """Inline delivery so a test never depends on a background thread."""
    return make_settings(
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        WEBHOOK_URL=URL,
        WEBHOOK_HMAC_SECRET=SECRET,
        WEBHOOK_DELIVERY_MODE="inline",
        WEBHOOK_MAX_ATTEMPTS=3,
        WEBHOOK_BACKOFF_BASE_SECONDS=0.01,
        WEBHOOK_BACKOFF_MAX_SECONDS=0.05,
    )


@pytest.fixture
def store(webhook_settings: Settings) -> Iterator[JobStore]:
    instance = JobStore(webhook_settings)
    yield instance
    instance.close()


@pytest.fixture
def job_id(store: JobStore) -> str:
    return store.create_or_get_job(document_id=DOC).job.id


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #


class TestSignature:
    def test_a_signed_payload_verifies(self) -> None:
        body = serialise_envelope(envelope())
        now = int(time.time())
        signature = sign_payload(SECRET, body, now)
        assert verify_signature(SECRET, body, now, signature, now=now) is True

    def test_the_sha256_prefix_is_accepted(self) -> None:
        body = b'{"a":1}'
        now = 1_800_000_000
        signature = sign_payload(SECRET, body, now)
        assert verify_signature(
            SECRET, body, now, f"{SIGNATURE_PREFIX}{signature}", now=now
        )

    def test_a_wrong_secret_does_not_verify(self) -> None:
        body = b'{"a":1}'
        now = 1_800_000_000
        signature = sign_payload(SECRET, body, now)
        assert verify_signature("other-secret", body, now, signature, now=now) is False

    def test_a_tampered_body_does_not_verify(self) -> None:
        now = 1_800_000_000
        signature = sign_payload(SECRET, b'{"amount":"45000"}', now)
        assert (
            verify_signature(SECRET, b'{"amount":"95000"}', now, signature, now=now)
            is False
        )

    def test_the_timestamp_is_inside_the_mac(self) -> None:
        """A capture cannot be replayed later with a fresh header."""
        body = b'{"a":1}'
        signature = sign_payload(SECRET, body, 1_800_000_000)
        assert (
            verify_signature(SECRET, body, 1_800_000_060, signature, now=1_800_000_060)
            is False
        )

    def test_an_old_delivery_is_refused(self) -> None:
        body = b'{"a":1}'
        stamped = 1_800_000_000
        signature = sign_payload(SECRET, body, stamped)
        assert (
            verify_signature(
                SECRET,
                body,
                stamped,
                signature,
                tolerance_seconds=300,
                now=stamped + 3600,
            )
            is False
        )

    def test_an_empty_secret_refuses_to_sign(self) -> None:
        with pytest.raises(ConfigurationError, match="WEBHOOK_HMAC_SECRET"):
            sign_payload("", b"{}", 1)

    def test_an_empty_secret_never_verifies(self) -> None:
        assert verify_signature("", b"{}", 1, "abc") is False

    def test_serialisation_is_stable(self) -> None:
        """The same version signed twice must produce the same MAC."""
        subject = envelope()
        assert serialise_envelope(subject) == serialise_envelope(subject)

    def test_indic_text_survives_serialisation_unescaped(self) -> None:
        body = serialise_envelope(envelope())
        assert body.decode("utf-8")  # decodes as UTF-8, no escaping assertion needed
        assert b"\\u" not in body


class TestHeaders:
    def test_carry_the_signature_job_and_version(self) -> None:
        headers = build_headers(
            signature="abc", timestamp=17, job_id="job-1", result_version=2
        )
        assert headers[SIGNATURE_HEADER] == f"{SIGNATURE_PREFIX}abc"
        assert headers[TIMESTAMP_HEADER] == "17"
        assert headers["X-RBI-Job-Id"] == "job-1"
        assert headers["X-RBI-Result-Version"] == "2"


class TestBackoff:
    def test_grows_and_is_capped(self) -> None:
        for attempt in range(1, 10):
            assert backoff_seconds(attempt, base=1.0, maximum=8.0) <= 8.0 * 1.25

    def test_is_jittered(self) -> None:
        """Lockstep retries recreate the outage they back off from."""
        draws = {backoff_seconds(3, base=1.0, maximum=60.0) for _ in range(20)}
        assert len(draws) > 1

    def test_attempt_is_one_based(self) -> None:
        with pytest.raises(ValueError, match="1-based"):
            backoff_seconds(0, base=1.0, maximum=1.0)


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestDelivery:
    def test_publishing_delivers_and_records(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(DeliveryOutcome(ok=True, status_code=202))
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        assert len(transport.calls) == 1
        assert store.get_delivery(delivery.id).status == DeliveryStatus.DELIVERED.value

    def test_the_receiver_can_verify_what_was_sent(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        """The whole contract, exercised as a consumer would."""
        transport = RecordingTransport()
        WebhookDispatcher(store, webhook_settings, transport=transport).publish(
            job_id, envelope()
        )
        _url, body, headers = transport.calls[0]
        assert verify_signature(
            SECRET,
            body,
            int(headers[TIMESTAMP_HEADER]),
            headers[SIGNATURE_HEADER],
            now=int(headers[TIMESTAMP_HEADER]),
        )

    def test_each_result_version_is_its_own_delivery(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport()
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        dispatcher.publish(job_id, envelope(1, final=False))
        dispatcher.publish(job_id, envelope(2, final=True))
        assert [row.result_version for row in store.deliveries_for(job_id)] == [1, 2]

    def test_republishing_a_version_does_not_resend(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        """An acks_late redelivery must not POST the same envelope twice."""
        transport = RecordingTransport()
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        dispatcher.publish(job_id, envelope(1))
        dispatcher.publish(job_id, envelope(1))
        assert len(transport.calls) == 1

    def test_delivery_is_skipped_when_not_configured(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        settings = make_settings(DATABASE_URL="sqlite+pysqlite:///:memory:")
        store = JobStore(settings)
        try:
            job = store.create_or_get_job(document_id=DOC).job
            transport = RecordingTransport()
            dispatcher = WebhookDispatcher(store, settings, transport=transport)
            assert dispatcher.publish(job.id, envelope()) is None
            assert transport.calls == []
        finally:
            store.close()

    def test_a_url_without_a_secret_is_refused(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """An unsigned webhook is a result envelope posted to whatever answers."""
        settings = make_settings(
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            WEBHOOK_URL=URL,
            WEBHOOK_DELIVERY_MODE="inline",
        )
        store = JobStore(settings)
        try:
            job = store.create_or_get_job(document_id=DOC).job
            dispatcher = WebhookDispatcher(
                store, settings, transport=RecordingTransport()
            )
            with pytest.raises(ConfigurationError, match="WEBHOOK_HMAC_SECRET"):
                dispatcher.publish(job.id, envelope())
        finally:
            store.close()

    def test_a_raising_transport_is_recorded_not_propagated(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        def explode(*_args: object) -> DeliveryOutcome:
            raise RuntimeError("socket exploded")

        dispatcher = WebhookDispatcher(store, webhook_settings, transport=explode)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        row = store.get_delivery(delivery.id)
        assert row.attempts == 1
        assert "RuntimeError" in (row.last_error or "")


class TestDeadLetterQueue:
    def test_retries_until_the_attempts_run_out_then_parks(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(DeliveryOutcome(ok=False, status_code=503))
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        for _ in range(webhook_settings.WEBHOOK_MAX_ATTEMPTS):
            dispatcher.attempt_delivery(delivery.id)
        row = store.get_delivery(delivery.id)
        assert row.attempts == webhook_settings.WEBHOOK_MAX_ATTEMPTS
        assert row.status == DeliveryStatus.DEAD.value
        assert [parked.id for parked in store.dead_letters()] == [delivery.id]

    def test_a_parked_delivery_is_replayable_byte_identically(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(
            DeliveryOutcome(ok=False, status_code=500),
            DeliveryOutcome(ok=False, status_code=500),
            DeliveryOutcome(ok=False, status_code=500),
            DeliveryOutcome(ok=True, status_code=200),
        )
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        for _ in range(webhook_settings.WEBHOOK_MAX_ATTEMPTS):
            dispatcher.attempt_delivery(delivery.id)
        assert store.get_delivery(delivery.id).status == DeliveryStatus.DEAD.value

        first_body = transport.calls[0][1]
        first_signature = transport.calls[0][2][SIGNATURE_HEADER]
        dispatcher.replay(delivery.id)

        assert store.get_delivery(delivery.id).status == DeliveryStatus.DELIVERED.value
        assert transport.calls[-1][1] == first_body
        assert transport.calls[-1][2][SIGNATURE_HEADER] == first_signature

    def test_replay_is_audited(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(DeliveryOutcome(ok=False, status_code=500))
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        for _ in range(webhook_settings.WEBHOOK_MAX_ATTEMPTS):
            dispatcher.attempt_delivery(delivery.id)
        dispatcher.replay(delivery.id)
        actions = [event.action for event in store.audit_trail(job_id)]
        assert AuditAction.WEBHOOK_REPLAYED.value in actions

    def test_replay_all_requeues_every_parked_delivery(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(DeliveryOutcome(ok=False, status_code=500))
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        for version in (1, 2):
            delivery = dispatcher.publish(job_id, envelope(version, final=version == 2))
            assert delivery is not None
            for _ in range(webhook_settings.WEBHOOK_MAX_ATTEMPTS):
                dispatcher.attempt_delivery(delivery.id)
        assert len(store.dead_letters()) == 2
        assert dispatcher.replay_all() == 2

    def test_deliver_or_raise_surfaces_the_failure(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport(DeliveryOutcome(ok=False, status_code=502))
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        delivery = dispatcher.publish(job_id, envelope())
        assert delivery is not None
        with pytest.raises(WebhookDeliveryFailed):
            dispatcher.deliver_or_raise(delivery.id)

    def test_drain_attempts_every_due_delivery(
        self, store: JobStore, webhook_settings: Settings, job_id: str
    ) -> None:
        transport = RecordingTransport()
        dispatcher = WebhookDispatcher(store, webhook_settings, transport=transport)
        store.enqueue_delivery(
            job_id=job_id,
            result_version=7,
            url=URL,
            payload="{}",
            signature="sig",
            signed_at=1,
            max_attempts=3,
        )
        assert dispatcher.drain() == 1
        assert len(transport.calls) == 1


class TestThreadMode:
    """The in-process worker for the laptop profile."""

    def test_the_thread_drains_the_queue(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        settings = make_settings(
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            WEBHOOK_URL=URL,
            WEBHOOK_HMAC_SECRET=SECRET,
            WEBHOOK_DELIVERY_MODE="thread",
        )
        store = JobStore(settings)
        transport = RecordingTransport()
        try:
            job = store.create_or_get_job(document_id=DOC).job
            dispatcher = WebhookDispatcher(
                store, settings, transport=transport, poll_interval=0.01
            )
            with dispatcher:
                delivery = dispatcher.publish(job.id, envelope())
                assert delivery is not None
                # Generous on purpose. The loop exits the moment the delivery
                # lands, so a healthy run costs milliseconds and this bound is
                # only reached when something is wrong. At 5 s it was not a
                # safety valve but a performance assertion about a background
                # thread competing with the rest of the suite for a CPU, and
                # it failed intermittently in full-suite runs while passing
                # every time in isolation.
                deadline = time.monotonic() + 60.0
                while time.monotonic() < deadline:
                    if (
                        store.get_delivery(delivery.id).status
                        == DeliveryStatus.DELIVERED.value
                    ):
                        break
                    time.sleep(0.01)
            assert (
                store.get_delivery(delivery.id).status
                == DeliveryStatus.DELIVERED.value
            )
        finally:
            store.close()

    def test_start_is_a_no_op_outside_thread_mode(
        self, store: JobStore, webhook_settings: Settings
    ) -> None:
        dispatcher = WebhookDispatcher(
            store, webhook_settings, transport=RecordingTransport()
        )
        dispatcher.start()
        assert dispatcher._thread is None
        dispatcher.stop()
