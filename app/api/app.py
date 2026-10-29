"""The FastAPI application factory — BUILD-BRIEF M6 and sec. 5.

Four endpoints, and an ordering rule that matters more than any of them:
**the residency gate runs before the server accepts a connection.**
:func:`create_app` builds the provider through
:func:`app.providers.factory.get_provider` during startup, so a node configured
with ``PROVIDER_PROFILE=laptop`` and ``SYNTHETIC_ONLY=false`` fails to start
rather than serving one request and then refusing.  There is no lazy path that
would let a first upload reach a model before the check.

Second in order and second in importance: the PII-redacting logging filter is
installed at **application creation**, before any router is added, so the very
first log line the app emits is already filtered.

``POST /v1/documents`` runs the pipeline **synchronously** under the laptop
profile (PLAN.md D2: eager, in-process, no Redis).  That is a deliberate
product choice for Phase 0/1 and not a shortcut: the fast-path SLA is seconds,
the caller is a mail gateway that wants the envelope, and a queue between them
would add a second failure mode for no latency benefit at this scale.  The
Celery path exists, is exercised eagerly, and is switched on by configuration.

This module does NOT handle:
  * authentication or authorisation.  The service sits behind the RBI gateway;
    adding a token here without knowing which one is worse than not adding one;
  * TLS, CORS or rate limiting — reverse-proxy concerns;
  * the review UI (WS-G) or any HTML;
  * running the pipeline itself (``app/pipeline.py``) or persistence
    (``app/storage/``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from starlette.requests import Request
from starlette.routing import Mount

from app.contracts.errors import (
    JobNotFound,
    MalwareDetected,
    PipelineError,
    ResidencyViolation,
)
from app.providers.base import ModelProvider
from app.providers.factory import (
    close_provider,
    get_provider,
    set_provider_profile_metric,
)
from app.settings import Settings, get_settings
from app.storage.repository import JobStore
from app.webhooks import WebhookDispatcher

__all__ = ["create_app"]

_LOG = logging.getLogger(__name__)

#: Every response, by route template and status.  The service's error rate is
#: ``sum(status =~ "[45].." ) / sum(all)`` over this one counter.
_HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP responses served, by method, route template and status code.",
    ["method", "route", "status"],
)
#: Exceptions that escaped every handler.  Distinct from a 5xx the taxonomy
#: produced on purpose: this one means nobody chose the status.
_HTTP_EXCEPTIONS = Counter(
    "http_unhandled_exceptions_total",
    "Requests that raised past every exception handler.",
    ["route"],
)
_HTTP_LATENCY = Histogram(
    "http_request_seconds",
    "Wall-clock latency of one HTTP request, by route template.",
    ["route"],
)


def create_app(
    settings: Settings | None = None,
    *,
    provider: ModelProvider | None = None,
    store: JobStore | None = None,
    dispatcher: WebhookDispatcher | None = None,
) -> FastAPI:
    """Build the application.

    Every collaborator is injectable so the test client needs no network, no
    weights and no Redis — but **not** the residency gate, which runs whether or
    not a provider was injected.  A test that supplies a mock provider must
    still be refused under a violating configuration, otherwise the one check
    this service cannot afford to get wrong would be the one thing the suite
    never exercises.

    Args:
        provider: pre-built provider. When ``None`` the factory builds one at
            startup, which is also where the gate would fire.
        store: pre-built :class:`~app.storage.repository.JobStore`.
        dispatcher: pre-built webhook dispatcher.

    Raises:
        ResidencyViolation: ``PROVIDER_PROFILE=laptop`` with
            ``SYNTHETIC_ONLY=false`` (PLAN.md invariant 6).
    """
    from app.api.logging_filter import install_redaction
    from app.api.routes import build_router

    active = settings if settings is not None else get_settings()

    redactor = install_redaction()
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        install_redaction(logging.getLogger(name))

    _assert_residency(active)
    _assert_categoriser(active)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        state = application.state
        state.settings = active
        state.redactor = redactor
        state.owns_provider = provider is None
        state.owns_store = store is None
        # The gauge describes the *configuration*, not who built the provider,
        # so it is set here rather than left to the factory: an injected
        # provider must not make `provider_profile{profile="laptop"} == 1`
        # silent on a node that is in fact on the laptop profile.
        set_provider_profile_metric(active.PROVIDER_PROFILE)
        # The gate again, at the moment of construction: create_app checks the
        # configuration, this checks the thing the configuration builds.
        state.provider = provider if provider is not None else get_provider(active)
        state.store = store if store is not None else JobStore(active)
        state.dispatcher = (
            dispatcher
            if dispatcher is not None
            else WebhookDispatcher(state.store, active)
        )
        state.dispatcher.start()
        _LOG.info("api ready | %s", active.startup_banner())
        try:
            yield
        finally:
            state.dispatcher.stop()
            if state.owns_provider:
                close_provider(state.provider)
            if state.owns_store:
                state.store.close()

    application = FastAPI(
        title=active.API_TITLE,
        version="1.0.0",
        summary=(
            "Evidence-bearing extraction for RBI complaint documents. Every "
            "value carries its page, box and character span."
        ),
        lifespan=lifespan,
    )
    _install_request_metrics(application)
    application.include_router(build_router())
    # Reviewer/test pages (WS-G): same-origin so the pages need no CORS. The
    # mount is optional-by-construction — a deployment that strips the UI
    # simply ships without the directory.
    from app.review_ui import STATIC_DIR

    if STATIC_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles

        application.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")
    _install_error_handlers(application)
    return application


def _assert_residency(settings: Settings) -> None:
    """Refuse to build an application that would breach data residency.

    The provider factory raises the same error for the same reason; this
    duplicates it at *import* time so the failure happens while a human is
    watching a deploy, not on the first upload.
    """
    reason = settings.residency_violation_reason
    if reason is not None:
        raise ResidencyViolation(
            reason,
            context={
                "profile": settings.PROVIDER_PROFILE.value,
                "synthetic_only": settings.SYNTHETIC_ONLY,
                "remedy": "set SYNTHETIC_ONLY=true or change PROVIDER_PROFILE",
            },
        )


def _assert_categoriser(settings: Settings) -> None:
    """Surface — and optionally refuse — a non-production categoriser.

    ``require_production_categoriser`` was written as "the gate a deployment
    calls at startup" and then never called from anywhere in ``app/``, so the
    placeholder taxonomy would run in production with nothing but the per-result
    ``is_production=false`` marking to show for it.  A marking downstream is not
    a gate: it relies on every consumer noticing.

    The refusal is opt-in (``REQUIRE_PRODUCTION_CATEGORISER``) because no
    production classifier exists yet and raising unconditionally would refuse
    every currently legitimate deployment.  The **warning is not** — a node
    running a placeholder taxonomy says so in its startup log, every time.

    Raises:
        CategoriserNotProduction: when the setting is true and the categoriser
            is still a stub.
    """
    from app.stages.s7_categorise import (
        default_categoriser,
        require_production_categoriser,
    )

    categoriser = default_categoriser()
    if settings.REQUIRE_PRODUCTION_CATEGORISER:
        require_production_categoriser(categoriser)
        return
    if not getattr(categoriser, "is_production", False):
        _LOG.warning(
            "categoriser %r is NOT a production classifier: complaint_category "
            "and supporting_doc_tags come from a placeholder taxonomy and must "
            "not be used for routing or reporting. Set "
            "REQUIRE_PRODUCTION_CATEGORISER=true once the real RB-IOS list is "
            "installed.",
            categoriser.name,
        )


def _install_request_metrics(application: FastAPI) -> None:
    """Count every response by route and status, and time it.

    The per-stage counters already say a great deal about *documents*; nothing
    until now counted **requests**, so "what is the service's error rate?" had
    no answer that did not involve reading logs.  This closes that: the series
    below divide into exactly that number.

    Labelled with the route *template* (``/v1/jobs/{job_id}``), never the
    resolved path.  A job id in a label would put a document identifier into
    the metrics endpoint, which is the one thing ``/metrics`` promises not to
    carry, and would grow an unbounded number of time series besides.  An
    unmatched path is reported as ``<unmatched>`` for the same reason.

    In-process and non-persistent by construction: ``prometheus_client``'s
    default registry lives in memory and starts empty on every boot, so a
    restart is a reset and nothing is written to disk.
    """

    @application.middleware("http")
    async def _record(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # An exception that escapes the handlers is still a 500 to the
            # caller, so it must appear in the error rate rather than vanish.
            route = _route_of(request)
            _HTTP_REQUESTS.labels(
                method=request.method, route=route, status="500"
            ).inc()
            _HTTP_EXCEPTIONS.labels(route=route).inc()
            raise
        # Resolved after the call, because routing is what populates the scope.
        route = _route_of(request)
        _HTTP_REQUESTS.labels(
            method=request.method,
            route=route,
            status=str(response.status_code),
        ).inc()
        _HTTP_LATENCY.labels(route=route).observe(time.perf_counter() - started)
        return response


def _route_of(request: Request) -> str:
    """The route template for ``request``, or a mount prefix, or ``<unmatched>``.

    Read from the resolved route rather than ``request.url.path`` so the label
    set stays bounded and carries no identifier.

    Static mounts do not populate ``scope["route"]``, so they are matched by
    prefix and reported as the mount (``/ui``).  Without that every page asset
    landed in ``<unmatched>`` alongside genuine 404s, and a browser fetching a
    favicon was indistinguishable from a caller hammering a wrong URL — which
    made the headline error rate read as 0.64 on an idle service.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    requested = request.url.path
    for mounted in getattr(request.app, "routes", ()):
        prefix = getattr(mounted, "path", None)
        if (
            isinstance(mounted, Mount)
            and isinstance(prefix, str)
            and prefix
            and (requested == prefix or requested.startswith(prefix + "/"))
        ):
            return prefix
    return "<unmatched>"


def _install_error_handlers(application: FastAPI) -> None:
    """Map the typed taxonomy onto HTTP, with two deliberate exceptions."""

    @application.exception_handler(JobNotFound)
    async def _not_found(_request: Request, exc: JobNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content=exc.to_dict())

    @application.exception_handler(MalwareDetected)
    async def _malware(_request: Request, exc: MalwareDetected) -> JSONResponse:
        """Refuse without naming the signature.

        ``MalwareDetected.context`` carries the engine's signature name because
        the security queue needs it; the *submitter* must not get it, because
        that is a free oracle for tuning the next attempt (the same reasoning
        that keeps this class out of ``DocumentRejected``).
        """
        _LOG.error(
            "attachment refused by the scanner | scanner=%s signature=%s",
            exc.scanner,
            exc.signature,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": "the attachment could not be processed",
                "context": {},
            },
        )

    @application.exception_handler(ResidencyViolation)
    async def _residency(
        _request: Request, exc: ResidencyViolation
    ) -> JSONResponse:
        """422, not the taxonomy's 500 — deliberately, and only here.

        ``ResidencyViolation.http_status`` is 500 because the *configuration*
        form of it (``laptop`` + ``SYNTHETIC_ONLY=false``) is a deployment
        fault.  That form can never reach a request handler: :func:`create_app`
        refuses to build the application at all.  The only violation that can
        arrive here is a submission that failed to declare itself synthetic —
        the caller's omission, fixable by resubmitting with the field — and
        answering that with 500 would tell a client to page an operator instead
        of reading the message.
        """
        _LOG.error("residency gate refused a request | %s", exc.message)
        return JSONResponse(status_code=422, content=exc.to_dict())

    @application.exception_handler(PipelineError)
    async def _pipeline(_request: Request, exc: PipelineError) -> JSONResponse:
        _LOG.warning("request failed | code=%s", exc.code)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


def app_state(request: Request) -> Any:
    """The application state, for a route handler."""
    return request.app.state
