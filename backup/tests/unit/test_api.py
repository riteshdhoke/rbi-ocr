"""HTTP tests — upload, idempotency, health, metrics, and the residency refusal.

Everything runs through Starlette's ``TestClient``, which drives the real ASGI
application including its lifespan, so the startup ordering these tests care
about is the ordering that actually runs: the residency gate first, the
PII-redacting filter installed at application creation, the provider and the
store built in the lifespan.

The provider is the deterministic contract-suite mock and the database is
in-memory SQLite, so the file needs no network, no weights and no Redis.  What
is **not** mocked is the gate: a violating configuration must be refused even
when a provider was injected, because otherwise the one check this service
cannot afford to get wrong would be the one thing the suite never exercises.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.logging_filter import PIIRedactionFilter, install_redaction, redact
from app.contracts.errors import ResidencyViolation
from app.stages.s7_categorise import CategoriserNotProduction
from app.settings import ProviderProfile, Settings
from app.storage.models import JobStatus
from app.storage.repository import JobStore

_CONTRACT_DIR = Path(__file__).resolve().parents[1] / "contract"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from mock_provider import MockProvider

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
CLEAN_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_clean.jpg"
OTHER_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v1_clean.jpg"
BOUNCE_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_bounce.jpg"
TRUNCATED_PDF = CORPUS / "adversarial" / "truncated.pdf"


@pytest.fixture
def api_settings(make_settings: Callable[..., Settings]) -> Settings:
    """Laptop profile, in-memory database, delivery off, no shadow sampling."""
    return make_settings(
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        WEBHOOK_DELIVERY_MODE="disabled",
        ESCALATION_SHADOW_RATE=0.0,
    )


@pytest.fixture
def store(api_settings: Settings) -> Iterator[JobStore]:
    instance = JobStore(api_settings)
    yield instance
    instance.close()


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(field_values={"complainant_name": "4. Name of bank"})


@pytest.fixture
def client(
    api_settings: Settings, store: JobStore, provider: MockProvider
) -> Iterator[TestClient]:
    application = create_app(api_settings, provider=provider, store=store)
    with TestClient(application) as test_client:
        yield test_client


def upload(
    client: TestClient,
    path: Path = CLEAN_DOC,
    *,
    synthetic: bool = True,
    **form: str,
) -> object:
    """POST one corpus file as multipart."""
    return client.post(
        "/v1/documents",
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
        data={"synthetic": str(synthetic).lower(), **form},
    )


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #


class TestUpload:
    def test_returns_the_envelope(self, client: TestClient) -> None:
        response = upload(client)
        assert response.status_code == 200
        body = response.json()
        assert body["is_new"] is True
        assert body["result"]["schema_version"] == "1.0.0"
        assert body["result"]["is_final"] is True
        assert body["result"]["document"] is not None

    def test_the_document_id_is_the_sha256_of_the_upload(
        self, client: TestClient
    ) -> None:
        from app.stages.s1_ingest import idempotency_key

        body = upload(client).json()
        assert body["document_id"] == idempotency_key(CLEAN_DOC.read_bytes())

    def test_the_job_reaches_completed(
        self, client: TestClient, store: JobStore
    ) -> None:
        body = upload(client).json()
        assert store.get_job(body["job_id"]).status == JobStatus.COMPLETED.value

    def test_every_published_version_is_stored(
        self, client: TestClient, store: JobStore
    ) -> None:
        body = upload(client).json()
        versions = [r.result_version for r in store.results_for(body["job_id"])]
        assert versions == sorted(versions)
        assert versions[0] == 1

    def test_a_bounced_document_is_a_200_with_a_reason(
        self, client: TestClient
    ) -> None:
        """The envelope is the product; a bounce is an outcome, not an error."""
        body = upload(client, BOUNCE_DOC).json()
        assert body["result"]["bounce_decision"]["decision"] == "bounce"
        assert body["result"]["bounce_decision"]["reason"]
        assert body["result"]["document"] is None

    def test_a_bounced_document_marks_the_job_bounced(
        self, client: TestClient, store: JobStore
    ) -> None:
        body = upload(client, BOUNCE_DOC).json()
        assert store.get_job(body["job_id"]).status == JobStatus.BOUNCED.value

    def test_an_unreadable_pdf_bounces_rather_than_500s(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/documents",
            files={"file": ("t.pdf", TRUNCATED_PDF.read_bytes(), "application/pdf")},
            data={"synthetic": "true"},
        )
        assert response.status_code == 200
        assert response.json()["result"]["bounce_decision"]["reason"] == "MALFORMED_PDF"

    def test_an_upload_over_the_cap_is_refused(
        self, api_settings: Settings, store: JobStore, provider: MockProvider
    ) -> None:
        small = api_settings.model_copy(update={"API_MAX_UPLOAD_BYTES": 1024})
        application = create_app(small, provider=provider, store=store)
        with TestClient(application) as tiny_client:
            response = upload(tiny_client)
        assert response.status_code == 413
        assert response.json()["code"] == "UPLOAD_TOO_LARGE"

    def test_the_sender_and_subject_are_recorded(
        self, client: TestClient, store: JobStore
    ) -> None:
        body = upload(
            client, sender="complainant@example.in", subject="Re: Debit CMS/2026/1"
        ).json()
        job = store.get_job(body["job_id"])
        assert job.sender == "complainant@example.in"
        assert job.normalized_subject == "debit cms/2026/1"


class TestIdempotency:
    def test_the_same_bytes_return_the_same_job(self, client: TestClient) -> None:
        first = upload(client).json()
        second = upload(client).json()
        assert second["is_new"] is False
        assert second["job_id"] == first["job_id"]

    def test_a_resubmission_returns_the_stored_envelope(
        self, client: TestClient
    ) -> None:
        first = upload(client).json()
        second = upload(client).json()
        assert (
            second["result"]["result_version"] == first["result"]["result_version"]
        )
        assert second["result"]["document_id"] == first["result"]["document_id"]

    def test_a_resubmission_runs_nothing(
        self, client: TestClient, provider: MockProvider
    ) -> None:
        upload(client)
        calls = len(provider.calls)
        upload(client)
        assert len(provider.calls) == calls

    def test_a_soft_duplicate_is_reported_not_refused(
        self, client: TestClient
    ) -> None:
        form = {"sender": "a@b.in", "subject": "Unauthorised debit"}
        first = upload(client, CLEAN_DOC, **form).json()
        second = upload(client, OTHER_DOC, **form).json()
        assert second["is_new"] is True
        assert second["soft_duplicate_of"] == first["job_id"]
        assert second["soft_duplicate_reason"] == "same_sender_subject_day"


# --------------------------------------------------------------------------- #
# GET /v1/jobs/{id}
# --------------------------------------------------------------------------- #


class TestJobEndpoint:
    def test_returns_the_job_and_its_latest_envelope(
        self, client: TestClient
    ) -> None:
        job_id = upload(client).json()["job_id"]
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["is_final"] is True
        assert body["result"]["result_version"] == body["result_version"]

    def test_lists_every_published_version(self, client: TestClient) -> None:
        job_id = upload(client).json()["job_id"]
        versions = client.get(f"/v1/jobs/{job_id}").json()["versions"]
        assert versions == list(range(1, len(versions) + 1))

    def test_an_unknown_job_is_404(self, client: TestClient) -> None:
        response = client.get("/v1/jobs/not-a-job")
        assert response.status_code == 404
        assert response.json()["code"] == "JOB_NOT_FOUND"


# --------------------------------------------------------------------------- #
# Health and metrics
# --------------------------------------------------------------------------- #


class TestHealth:
    def test_reports_the_profile_banner(self, client: TestClient) -> None:
        body = client.get("/healthz").json()
        assert body["profile"] == "laptop"
        assert body["synthetic_only"] is True
        assert "profile=laptop" in body["banner"]

    def test_reports_every_model_artifact(self, client: TestClient) -> None:
        names = {entry["name"] for entry in client.get("/healthz").json()["models"]}
        assert {"ocr_detection", "ocr_rec_devanagari", "language_id"} <= names

    def test_missing_weights_are_degraded_not_unhealthy(
        self, api_settings: Settings, store: JobStore, provider: MockProvider, tmp_path: Path
    ) -> None:
        """A missing volume mount must not become a container restart loop."""
        empty = api_settings.model_copy(update={"MODELS_DIR": tmp_path})
        application = create_app(empty, provider=provider, store=store)
        with TestClient(application) as empty_client:
            body = empty_client.get("/healthz").json()
        assert body["status"] == "degraded"
        assert all(entry["present"] is False for entry in body["models"])

    def test_reports_the_provider_gauge(self, client: TestClient) -> None:
        gauge = client.get("/healthz").json()["provider_profile_gauge"]
        assert gauge["laptop"] == 1.0
        assert gauge["cpu_node"] == 0.0

    def test_reports_the_database_state(self, client: TestClient) -> None:
        assert client.get("/healthz").json()["database"] == "ok"

    def test_carries_no_secret(self, client: TestClient) -> None:
        body = client.get("/healthz").text
        assert "hf_" not in body.lower() or "hf_token=unset" in body.lower()


class TestMetrics:
    def test_exposes_the_prometheus_registry(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_includes_the_provider_gauge_and_pipeline_counters(
        self, client: TestClient
    ) -> None:
        upload(client)
        text = client.get("/metrics").text
        assert "provider_profile" in text
        assert "pipeline_documents_total" in text
        assert "pipeline_stage_seconds" in text


# --------------------------------------------------------------------------- #
# Residency
# --------------------------------------------------------------------------- #


class TestCategoriserGate:
    """``require_production_categoriser`` was defined, documented as the startup
    gate, tested — and called from nowhere in ``app/``."""

    def test_requiring_a_production_categoriser_refuses_the_stub(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        settings = make_settings(
            REQUIRE_PRODUCTION_CATEGORISER=True,
            DATABASE_URL="sqlite+pysqlite:///:memory:",
        )
        with pytest.raises(CategoriserNotProduction) as raised:
            create_app(settings)
        assert raised.value.code == "CATEGORISER_NOT_PRODUCTION"

    def test_the_default_starts_but_says_so_loudly(
        self,
        make_settings: Callable[..., Settings],
        provider: MockProvider,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No production classifier exists yet, so the refusal must be opt-in —
        but a node running a placeholder taxonomy says so every start."""
        settings = make_settings(
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            WEBHOOK_DELIVERY_MODE="disabled",
        )
        store = JobStore(settings)
        try:
            with caplog.at_level(logging.WARNING, logger="app.api.app"):
                create_app(settings, provider=provider, store=store)
        finally:
            store.close()
        assert any(
            "NOT a production classifier" in record.message
            for record in caplog.records
        )


class TestResidencyGate:
    def test_the_app_refuses_to_build_with_synthetic_only_false(
        self, make_settings: Callable[..., Settings]
    ) -> None:
        """BUILD-BRIEF M2's acceptance test: the app refuses to start."""
        violating = make_settings(
            SYNTHETIC_ONLY=False, DATABASE_URL="sqlite+pysqlite:///:memory:"
        )
        with pytest.raises(ResidencyViolation) as raised:
            create_app(violating)
        assert "SYNTHETIC_ONLY=true" in str(raised.value)

    def test_the_refusal_holds_even_with_an_injected_provider(
        self,
        make_settings: Callable[..., Settings],
        provider: MockProvider,
    ) -> None:
        """Otherwise the one check we cannot get wrong is never exercised."""
        violating = make_settings(
            SYNTHETIC_ONLY=False, DATABASE_URL="sqlite+pysqlite:///:memory:"
        )
        with pytest.raises(ResidencyViolation):
            create_app(violating, provider=provider)

    def test_a_cpu_node_may_run_with_synthetic_only_false(
        self, make_settings: Callable[..., Settings], provider: MockProvider
    ) -> None:
        settings = make_settings(
            PROVIDER_PROFILE=ProviderProfile.CPU_NODE,
            SYNTHETIC_ONLY=False,
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            WEBHOOK_DELIVERY_MODE="disabled",
        )
        store = JobStore(settings)
        try:
            application = create_app(settings, provider=provider, store=store)
            with TestClient(application) as node_client:
                assert node_client.get("/healthz").json()["profile"] == "cpu_node"
        finally:
            store.close()

    def test_an_undeclared_upload_is_refused_under_the_laptop_profile(
        self, client: TestClient
    ) -> None:
        response = upload(client, synthetic=False)
        assert response.status_code == 422
        assert response.json()["code"] == "RESIDENCY_VIOLATION"

    def test_an_undeclared_upload_marks_the_job_failed(
        self, client: TestClient, store: JobStore
    ) -> None:
        upload(client, synthetic=False)
        from app.stages.s1_ingest import idempotency_key

        job = store.find_by_document_id(idempotency_key(CLEAN_DOC.read_bytes()))
        assert job is not None
        assert job.status == JobStatus.FAILED.value
        assert job.error_code == "RESIDENCY_VIOLATION"


# --------------------------------------------------------------------------- #
# PII redaction at the logging filter
# --------------------------------------------------------------------------- #


class TestRedaction:
    def test_is_installed_at_application_creation(
        self, api_settings: Settings, store: JobStore, provider: MockProvider
    ) -> None:
        create_app(api_settings, provider=provider, store=store)
        root = logging.getLogger()
        assert any(isinstance(f, PIIRedactionFilter) for f in root.filters)

    def test_installation_is_idempotent(self) -> None:
        first = install_redaction()
        second = install_redaction()
        assert first is second

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("account 30294817465021 debited", "[CARD_OR_ACCOUNT]"),
            ("IFSC BARB0417293 branch", "[IFSC]"),
            ("write to complainant@example.in", "[EMAIL]"),
            ("amount ₹45,000 was", "[AMOUNT]"),
            ("बैंक ऑफ बड़ौदा", "[INDIC_TEXT]"),
        ],
    )
    def test_masks_recognised_identifiers(self, text: str, expected: str) -> None:
        masked, fired = redact(text)
        assert expected in masked
        assert fired

    def test_leaves_ordinary_text_alone(self) -> None:
        masked, fired = redact("pipeline start document=job-17 pages=3")
        assert fired == ()
        assert masked == "pipeline start document=job-17 pages=3"

    def test_drops_an_info_record_carrying_ocr_text(self) -> None:
        """"Never log raw OCR text at INFO" is a rule about the record."""
        redactor = PIIRedactionFilter()
        record = logging.LogRecord(
            "app.stages.s4_ocr",
            logging.INFO,
            "s4_ocr.py",
            1,
            "read line: %s",
            ("बैंक ऑफ बड़ौदा",),
            None,
        )
        assert redactor.filter(record) is False
        assert redactor.dropped == 1

    def test_keeps_and_masks_a_warning(self) -> None:
        """Losing an error is worse than storing a masked identifier."""
        redactor = PIIRedactionFilter()
        record = logging.LogRecord(
            "app.stages.s5_validators",
            logging.WARNING,
            "s5.py",
            1,
            "checksum failed for %s",
            ("30294817465021",),
            None,
        )
        assert redactor.filter(record) is True
        assert "[CARD_OR_ACCOUNT]" in record.getMessage()
        assert "30294817465021" not in record.getMessage()

    def test_a_malformed_format_string_does_not_break_logging(self) -> None:
        redactor = PIIRedactionFilter()
        record = logging.LogRecord(
            "x", logging.INFO, "x.py", 1, "%d items", ("not-an-int",), None
        )
        assert redactor.filter(record) is True

    def test_attaches_to_handlers_not_only_to_the_logger(self) -> None:
        """A logger filter never sees a propagated record; a handler filter does."""
        logger = logging.getLogger("test_redaction_target")
        handler = logging.NullHandler()
        logger.addHandler(handler)
        try:
            redactor = install_redaction(logger)
            assert any(f is redactor for f in handler.filters)
        finally:
            logger.removeHandler(handler)
            logger.filters.clear()


# --------------------------------------------------------------------------- #
# Stats — the readable companion to /metrics
# --------------------------------------------------------------------------- #


class TestStats:
    """``GET /v1/stats``: error rates, in memory, reset by a restart."""

    def test_counts_requests_and_computes_an_error_rate(
        self, client: TestClient
    ) -> None:
        client.get("/healthz")
        client.get("/healthz")
        client.get("/v1/jobs/no-such-job")  # 404

        payload = client.get("/v1/stats").json()
        assert payload["requests"] >= 3
        assert payload["client_errors"] >= 1
        assert payload["server_errors"] == 0
        assert 0.0 < payload["error_rate"] <= 1.0
        assert payload["uptime_seconds"] >= 0.0

    def test_routes_are_labelled_by_template_never_by_identifier(
        self, client: TestClient
    ) -> None:
        """A job id in a label would put a document identifier in the metrics.

        It would also grow one time series per document, which is the other
        reason the template is the only acceptable label.
        """
        client.get("/v1/jobs/a-very-identifying-job-id")

        routes = {entry["route"] for entry in client.get("/v1/stats").json()["routes"]}
        assert "/v1/jobs/{job_id}" in routes
        assert not any("a-very-identifying-job-id" in route for route in routes)

    def test_a_bounced_document_is_a_document_outcome_not_an_http_error(
        self, client: TestClient
    ) -> None:
        """The two numbers answer different questions, so both are reported.

        A refused document is a *product* outcome delivered as 200 with a
        bounce envelope, so it must not inflate the HTTP error rate — but it
        must still be visible, which is what ``documents`` is for.
        """
        upload(client, BOUNCE_DOC)

        payload = client.get("/v1/stats").json()
        assert payload["server_errors"] == 0
        assert sum(payload["documents"].values()) >= 1

    def test_the_document_id_never_appears_anywhere_in_the_payload(
        self, client: TestClient
    ) -> None:
        response = upload(client)
        document_id = response.json()["document_id"]  # type: ignore[attr-defined]

        assert document_id not in client.get("/v1/stats").text

    def test_static_assets_are_labelled_by_mount_not_lumped_with_404s(
        self, client: TestClient
    ) -> None:
        """A favicon must not look like a caller hammering a wrong URL.

        Static mounts do not populate ``scope["route"]``, so before they were
        matched by prefix every page asset landed in ``<unmatched>`` next to
        genuine 404s and the headline error rate read as 0.64 on an idle
        service.
        """
        client.get("/ui/test_page.html")
        client.get("/definitely-not-a-route")

        routes = {
            entry["route"]: entry
            for entry in client.get("/v1/stats").json()["routes"]
        }
        assert routes["/ui"]["client_errors"] == 0
        assert routes["<unmatched>"]["client_errors"] == 1


class TestRequestedFields:
    def test_groups_requested_missing_and_extra_fields(self, client: TestClient) -> None:
        response = upload(client, requested_fields='complainant name, missing custom field')
        assert response.status_code == 200
        result = response.json()['result']
        assert result['requested_fields'] == ['complainant_name', 'missing_custom_field']
        document = result['document']
        assert document['extracted_fields']['complainant_name']
        assert document['extracted_fields']['missing_custom_field'] == []
        assert document['missing_fields'] == ['missing_custom_field']
        assert document['other_fields']
        assert document['fields']  # Existing consumers keep the complete map.

    def test_normalized_same_list_reuses_result(self, client: TestClient, provider: MockProvider) -> None:
        first = upload(client, requested_fields='mobile, account number').json()
        calls = len(provider.calls)
        again = upload(client, requested_fields='ACCOUNT_NUMBER, mobile, MOBILE, ').json()
        assert again['job_id'] == first['job_id']
        assert again['is_new'] is False
        assert again['result']['result_version'] == first['result']['result_version']
        assert len(provider.calls) == calls

    def test_changed_list_reextracts_and_preserves_prior_versions(self, client: TestClient, store: JobStore) -> None:
        first = upload(client, requested_fields='mobile').json()
        previous_version = first['result']['result_version']
        original = store.latest_envelope(first['job_id']).model_dump(mode='json')
        changed = upload(client, requested_fields='complainant name').json()
        assert changed['job_id'] == first['job_id']
        assert changed['document_id'] == first['document_id']
        assert changed['is_new'] is False
        assert changed['result']['result_version'] > previous_version
        history = store.results_for(first['job_id'])
        assert [row.result_version for row in history] == list(range(1, len(history) + 1))
        assert history[previous_version - 1].envelope == original
        fetched = client.get('/v1/jobs/' + first['job_id']).json()['result']
        assert fetched['requested_fields'] == ['complainant_name']
        assert fetched['document']['extracted_fields']['complainant_name']

    def test_requested_list_on_bounce_is_stored_but_no_content_is_returned(self, client: TestClient) -> None:
        first = upload(client, BOUNCE_DOC, requested_fields='mobile').json()
        assert first['result']['requested_fields'] == ['mobile']
        assert first['result']['document'] is None
        changed = upload(client, BOUNCE_DOC, requested_fields='email').json()
        assert changed['result']['result_version'] > first['result']['result_version']
        assert changed['result']['requested_fields'] == ['email']
        assert changed['result']['document'] is None

    def test_empty_list_returns_all_fields_in_other_group(self, client: TestClient) -> None:
        document = upload(client, requested_fields=' , , ').json()['result']['document']
        assert document['extracted_fields'] == {}
        assert document['other_fields'] == document['fields']
        assert document['missing_fields'] == []

    def test_invalid_field_list_is_422(self, client: TestClient) -> None:
        assert upload(client, requested_fields='???').status_code == 422

    def test_request_list_has_a_length_limit(self, client: TestClient) -> None:
        assert upload(client, requested_fields='x' * 4097).status_code == 422
