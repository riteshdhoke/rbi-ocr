"""Environment-driven configuration — BUILD-BRIEF sec. 6, PLAN.md dual-mode.

Every knob in ``.env.example`` appears here with its type, its bounds and a
description.  Field names are **UPPERCASE and identical to the environment
variable**, because BUILD-BRIEF sec. 2 writes the residency gate as
``s.PROVIDER_PROFILE`` / ``s.SYNTHETIC_ONLY`` and the brief's spelling wins.

Deliberate non-behaviour — this module **carries** values, it does not enforce
policy:

* ``PROVIDER_PROFILE=laptop`` with ``SYNTHETIC_ONLY=false`` does **not** raise
  here.  The residency gate is a *provider factory* concern (WS-C,
  BUILD-BRIEF M2) so that the refusal happens at the one place a model call can
  originate and its test can assert on ``get_provider``.  Raising in the
  settings constructor would also make the violation un-inspectable — tooling
  could not read a bad config to report it.  :attr:`Settings.residency_violation_reason`
  exposes exactly the string the factory raises, so the message lives once.
* a missing ``HF_TOKEN`` under the laptop profile does not raise either; the
  provider that needs it fails when it is constructed.  Tests must run with no
  token and no network.
* ``CELERY_CONCURRENCY x ORT_INTRA_OP_THREADS <= physical cores`` (PRD P0-11)
  is exposed as :meth:`Settings.assert_thread_budget` for CI to call, not
  enforced on load: a 2-core CI runner would otherwise be unable to import the
  package with the documented defaults.

This module does NOT handle:
  * setting the thread-cap environment variables — :mod:`app` already did that
    before this module could be imported (invariant 5); the fields here only
    *report* the values in effect;
  * the residency gate, the corpus synthetic-sidecar check, or any provider
    construction (WS-C);
  * secrets management beyond ``SecretStr`` — the ``.env`` file is gitignored
    and never logged;
  * logging configuration or Prometheus registration;
  * creating the directories it names (``PROVIDER_CACHE_PATH`` etc.); the
    component that writes there creates them.
"""

from __future__ import annotations

import ipaddress
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DEFAULT_HF_BASE_URL",
    "LOCAL_HOST_SUFFIXES",
    "PROJECT_ROOT",
    "LogLevel",
    "ProviderProfile",
    "Settings",
    "get_settings",
    "is_local_endpoint",
    "reset_settings_cache",
]

#: Repository root, resolved from this file so the CWD never matters.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The shipped default endpoint.  Its *identity* matters, not just its value:
#: an air-gapped profile that still carries this URL has not been pointed at an
#: in-house model server, which is a different situation from one deliberately
#: aimed somewhere off-premises.  See :attr:`Settings.slm_endpoint_configured`.
DEFAULT_HF_BASE_URL: Final[str] = "https://router.huggingface.co/v1"

#: DNS suffixes reserved for names that cannot resolve on the public internet.
#: A host ending in one of these is on the deploying organisation's own network
#: by definition, which is what makes them safe to treat as on-premises without
#: an operator having to enumerate every server.
LOCAL_HOST_SUFFIXES: Final[tuple[str, ...]] = (
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".home.arpa",
)


def is_local_endpoint(url: str, extra_hosts: frozenset[str] = frozenset()) -> bool:
    """Whether ``url`` names a host that cannot be off-premises.

    This is the test that decides whether an air-gapped profile is allowed to
    build a network-facing provider at all, so it **fails closed**: a name it
    cannot positively place inside the deployment is not local.  Recognised as
    local are loopback, RFC1918 / link-local / unique-local literals, the
    reserved suffixes in :data:`LOCAL_HOST_SUFFIXES`, a bare single-label
    hostname (no dot — unqualified names do not resolve publicly), and anything
    an operator names in ``LOCAL_ENDPOINT_HOSTS``.

    A public DNS name is **not** local even when it happens to resolve to a
    private address today: DNS is not under this service's control, and the
    whole point of the check is that it holds without a network round trip.
    """
    host = urlsplit(url).hostname
    if not host:
        return False
    host = host.lower().rstrip(".")
    if host in extra_hosts:
        return True
    if host == "localhost" or host.endswith(LOCAL_HOST_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Not an IP literal.  An unqualified single-label name (``slm-box``)
        # is a LAN name; a dotted public name (``router.huggingface.co``) is not.
        return "." not in host
    return bool(
        address.is_loopback or address.is_private or address.is_link_local
    )


class ProviderProfile(StrEnum):
    """Where model inference happens (PRD P0-1)."""

    #: Dev laptop: SLM/VLM over the HF router, OCR + MT local. Synthetic data only.
    LAPTOP = "laptop"
    #: Air-gapped CPU node: everything local.
    CPU_NODE = "cpu_node"
    #: GPU node: SLM/VLM on a local serving stack.
    GPU_NODE = "gpu_node"


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """All runtime configuration, read from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    # ---------------------------------------------------------------- provider
    PROVIDER_PROFILE: ProviderProfile = Field(
        default=ProviderProfile.LAPTOP,
        description="laptop | cpu_node | gpu_node (PRD P0-1).",
    )
    SYNTHETIC_ONLY: bool = Field(
        default=True,
        description=(
            "Must stay true under the laptop profile: that profile routes text "
            "to a non-Indian endpoint (PLAN.md invariant 6). No bypass flag."
        ),
    )
    HF_TOKEN: SecretStr | None = Field(
        default=None, description="HF router token; laptop profile only."
    )
    HF_BASE_URL: str = Field(
        default=DEFAULT_HF_BASE_URL,
        min_length=1,
        description=(
            "OpenAI-compatible base URL. Swapping to a local llama.cpp server "
            "is a change to this value alone (BUILD-BRIEF M2). On gpu_node, if "
            "you set this at all it must name an on-premises host — leaving "
            "the default means 'no model server here' and the node runs "
            "without an SLM rather than calling the public router. cpu_node "
            "never reads it."
        ),
    )
    LOCAL_ENDPOINT_HOSTS: str = Field(
        default="",
        description=(
            "Comma-separated extra hostnames to treat as on-premises for "
            "HF_BASE_URL, e.g. 'slm-a.rbi.gov.in'. Loopback, RFC1918 literals, "
            "unqualified names and the reserved suffixes are already local; "
            "this is for an in-house server behind a public-looking FQDN. "
            "Adding a host here is a residency assertion — it says 'this name "
            "is inside our network'. It is not a bypass: it changes where the "
            "endpoint is believed to be, never whether the check runs."
        ),
    )
    # Measured defaults (A-11 then A-12): the brief's Qwen3-Instruct ids do not
    # exist on the HF router. The full-corpus eval (docs/eval/REPORT-2026-08-08-slmlive.md,
    # 2026-08-08) measured gemma-3-4b-it schema validity at 0.367 (n=147),
    # which caps SLM recall at that level; Llama-3.1-8B-Instruct measured
    # 1.000 (n=18) with grounding 0.56 — the higher ceiling. Llama is primary,
    # gemma the fallback.
    EXTRACT_ALL_FIELDS: bool = True

    SLM_MODEL: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", min_length=1)
    SLM_MODEL_FALLBACK: str = Field(default="google/gemma-3-4b-it", min_length=1)
    SLM_MAX_TOKENS: int = Field(default=512, gt=0, le=8192)
    SLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="PRD P0-8 requires 0 for extraction; higher is a bake-off knob.",
    )
    JOB_TOKEN_BUDGET: int = Field(
        default=8000, gt=0, description="Per-job cap; breach raises TokenBudgetExceeded."
    )
    PROVIDER_CACHE_PATH: Path = Field(
        default=Path(".cache/provider.sqlite"),
        description="sha256(model+prompt+params) -> response cache (BUILD-BRIEF M2).",
    )

    # ----------------------------------------------------------------- threads
    # These mirror what app/__init__.py already placed in os.environ. They are
    # reported here, never applied from here — by the time settings load, every
    # numeric library has read the environment (invariant 5).
    OMP_NUM_THREADS: int = Field(default=2, ge=1, le=64)
    MKL_NUM_THREADS: int = Field(default=2, ge=1, le=64)
    ORT_INTRA_OP_THREADS: int = Field(
        default=2,
        ge=1,
        le=8,
        description=(
            "Never raise above 2 to 'speed things up' — every runtime defaults "
            "to nproc independently and oversubscription costs 3-5x "
            "(BUILD-BRIEF sec. 8)."
        ),
    )
    CELERY_CONCURRENCY: int = Field(
        default=2,
        ge=1,
        le=64,
        description="concurrency x intra_op <= physical cores (PRD P0-11).",
    )

    # --------------------------------------------------------------------- gpu
    GPU_EXECUTION: Literal["auto", "never", "require"] = Field(
        default="auto",
        description=(
            "Whether the local vision/MT models use CUDA. 'auto' uses the GPU "
            "only on the gpu_node profile and only when the runtime actually "
            "exposes it, falling back to CPU otherwise. 'require' refuses to "
            "start without a GPU — use it on a real GPU node so a missing "
            "onnxruntime-gpu wheel is a loud failure and not a silent 10x "
            "slowdown. 'never' pins CPU everywhere."
        ),
    )

    # ------------------------------------------------------------------- vision
    # Default 4, not the brief's 16-32: measured on-target (bench_results.json,
    # A-10) — large batches are 23-31% slower on laptop-class CPUs (cache
    # pressure), batch 2-4 optimal, output identical at every size.
    OCR_BATCH_SIZE: int = Field(
        default=4,
        ge=1,
        le=128,
        description=(
            "Line crops per recognition call. 16-32 is the tuned band; a value "
            "of 1 is a looped path and is only legitimate in the benchmark tool "
            "(PLAN.md invariant 4)."
        ),
    )
    REC_HEAD_CACHE_SIZE: int = Field(default=3, ge=1, le=32)
    REC_HEAD_CACHE_TTL_SECONDS: int = Field(
        default=900, ge=1, description="15 min per BUILD-BRIEF M4."
    )
    ESCALATION_SHADOW_RATE: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Share of PASSED pages also run through Tier 2 (invariant 3).",
    )
    CODE_MIX_THRESHOLD: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="is_code_mixed when >= this share of lines differ in script.",
    )

    # ------------------------------------------------------------ quality gate
    # OPEN-QUESTIONS Q3, resolved by the WS-E1 calibration run over the 295-page
    # synthetic corpus.  The numbers, the confusion matrix they produce and the
    # method are in docs/decisions-wse1.md and docs/calibration/quality_gate.json;
    # app/stages/s15_quality.py::QualityThresholds reads them and documents what
    # each one means.  Everything here is a *tuning* knob — the model structure
    # (sub-score weights, advisory flag levels) stays in the stage module.
    QUALITY_BOUNCE_THRESHOLD: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Below this quality_score the page bounces (OPEN-QUESTIONS Q3).",
    )
    QUALITY_BLUR_LAPLACIAN_FLOOR: float = Field(
        default=9_500.0,
        ge=0.0,
        description=(
            "Variance of the Laplacian over the glyph band at which the "
            "sharpness sub-score reaches 0. Ramped in log10 space."
        ),
    )
    QUALITY_BLUR_LAPLACIAN_TARGET: float = Field(
        default=26_000.0,
        gt=0.0,
        description="Laplacian variance at which the sharpness sub-score reaches 1.",
    )
    QUALITY_CONTRAST_FLOOR: float = Field(
        default=17.0,
        ge=0.0,
        le=255.0,
        description="Background-minus-ink level at which contrast scores 0.",
    )
    QUALITY_CONTRAST_TARGET: float = Field(
        default=70.0,
        gt=0.0,
        le=255.0,
        description="Background-minus-ink level at which contrast scores 1.",
    )
    QUALITY_EFFECTIVE_DPI_FLOOR: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Effective DPI (measured text height / 0.09 in) at which the "
            "resolution sub-score reaches 0."
        ),
    )
    QUALITY_EFFECTIVE_DPI_TARGET: float = Field(
        default=140.0,
        gt=0.0,
        description="Effective DPI at which the resolution sub-score reaches 1.",
    )
    QUALITY_TEXT_COVERAGE_FLOOR: float = Field(
        default=0.004,
        ge=0.0,
        le=1.0,
        description="Below this share of page area covered by text, the page bounces.",
    )
    QUALITY_SKEW_REPAIR_DEG: float = Field(
        default=0.75,
        gt=0.0,
        description=(
            "Skew at or above which deskew is applied. Below it the resample "
            "costs more sharpness than the rotation recovers."
        ),
    )
    QUALITY_SKEW_BOUNCE_DEG: float = Field(
        default=25.0,
        gt=0.0,
        description="Skew beyond which the correction is untrustworthy and the page bounces.",
    )
    QUALITY_MIN_PAGE_DIMENSION_PX: int = Field(
        default=200,
        gt=0,
        description=(
            "Short-edge floor below which a page is not a document "
            "(one inch at the 200 DPI rasterisation resolution)."
        ),
    )

    # ---------------------------------------------------------- resource caps
    MAX_PAGES_PER_DOC: int = Field(default=200, ge=1)
    MAX_DECOMPRESSION_RATIO: int = Field(default=100, ge=1)
    MAX_TOTAL_PIXELS: int = Field(
        default=500_000_000,
        ge=1,
        description=(
            "Total rasterised pixels per document (BUILD-BRIEF M3 pixel cap). "
            "Not yet in .env.example — see docs/decisions-wsb.md."
        ),
    )
    MAX_WALL_CLOCK_SECONDS: float = Field(
        default=900.0,
        gt=0.0,
        description="Per-document sandbox wall clock (BUILD-BRIEF M3).",
    )

    # --------------------------------------------------------------- extraction
    GROUNDING_MAX_EDIT_DISTANCE: int = Field(
        default=2,
        ge=0,
        description=(
            "Levenshtein budget for the grounding gate; script-aware, Indic "
            "digits normalised first (PLAN.md invariant 8)."
        ),
    )

    # ---------------------------------------------------------------- service
    # PLAN.md D2: SQLite + filesystem + eager by default; the Redis/Postgres
    # code paths exist and are exercised in eager mode.
    DATABASE_URL: str = Field(
        default="sqlite+pysqlite:///./data/runtime/app.sqlite", min_length=1
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0", min_length=1)
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=True)
    WEBHOOK_HMAC_SECRET: SecretStr | None = Field(default=None)

    #: OPEN-QUESTIONS **Q8**, resolved (WS-F).  ``run_validators`` takes an
    #: ``IFSCMaster``; without this setting the only way to hand it the real RBI
    #: list was to edit the orchestrator's call site.  The default is the
    #: committed synthetic fixture (Q4b) so nothing changes for tests or a fresh
    #: clone; an air-gapped deployment points it at the real ``.csv``/``.json``
    #: and gets authoritative lookups with no code change (D-E3-8).
    IFSC_MASTER_PATH: Path = Field(
        default=Path("app/resources/ifsc_master_sample.json"),
        description=(
            "IFSC master list. '.csv' loads as authoritative (unknown branch "
            "fails the lookup); '.json' takes authority from its payload."
        ),
    )

    #: The **bank-name** table, which is *not* the IFSC master.  The two were
    #: conflated once already: the deploy guide told operators to point
    #: ``IFSC_MASTER_PATH`` at the real RBI list, which reads as though bank
    #: coverage follows — it does not, because the RBI master carries branch
    #: rows and one Latin spelling per bank, never the Devanagari rendering or
    #: the acronym an OCR'd complaint actually contains.  Before this setting
    #: existed the alias table could only be changed by editing the installed
    #: package, so every deployment silently ran the five-bank fixture.
    BANK_ALIASES_PATH: Path = Field(
        default=Path("app/resources/bank_aliases.json"),
        description=(
            "Bank display names, alias spellings and the non-party institution "
            "list used by the bank_name matcher. Separate from "
            "IFSC_MASTER_PATH: that one resolves branch codes, this one "
            "resolves names."
        ),
    )

    #: The categorisation stub ships marked (``is_production=false`` plus a
    #: mandatory reason on every result) and
    #: :func:`~app.stages.s7_categorise.require_production_categoriser` exists to
    #: refuse it — but nothing called that gate, so the marking was the only
    #: thing standing between a placeholder taxonomy and a production envelope.
    #: Default ``false`` because there *is* no production categoriser yet:
    #: raising unconditionally would refuse every currently valid deployment.
    #: Startup warns loudly either way; this turns the warning into a refusal.
    REQUIRE_PRODUCTION_CATEGORISER: bool = Field(
        default=False,
        description=(
            "Refuse to start when the categoriser is not a production "
            "classifier. Set true once the real RB-IOS taxonomy is installed, "
            "so a rollback to the stub cannot silently ship placeholder tags."
        ),
    )

    # --------------------------------------------------------------- webhooks
    # BUILD-BRIEF sec. 5: two-stage delivery, HMAC-signed, replayable DLQ.
    WEBHOOK_URL: str | None = Field(
        default=None,
        description=(
            "Destination for result_version increments. None disables "
            "delivery entirely; nothing is enqueued and no DLQ row is written."
        ),
    )
    WEBHOOK_DELIVERY_MODE: Literal["thread", "celery", "inline", "disabled"] = Field(
        default="thread",
        description=(
            "thread = in-process worker (laptop profile); celery = enqueue on "
            "q.nlp; inline = deliver on the calling thread (tests); disabled = "
            "record the attempt as undeliverable and stop."
        ),
    )
    WEBHOOK_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0.0, le=300.0)
    WEBHOOK_MAX_ATTEMPTS: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Attempts before the delivery is parked in the DLQ.",
    )
    WEBHOOK_BACKOFF_BASE_SECONDS: float = Field(default=1.0, gt=0.0)
    WEBHOOK_BACKOFF_MAX_SECONDS: float = Field(default=60.0, gt=0.0)
    WEBHOOK_SIGNATURE_TOLERANCE_SECONDS: int = Field(
        default=300,
        ge=1,
        description=(
            "Age beyond which a receiver must reject a signed payload. Data "
            "for the consumer; this service only stamps the timestamp."
        ),
    )

    # ----------------------------------------------------------------- celery
    CELERY_BROKER_URL: str | None = Field(
        default=None, description="Defaults to REDIS_URL when unset."
    )
    CELERY_RESULT_BACKEND: str | None = Field(
        default=None, description="Defaults to REDIS_URL when unset."
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=1800,
        gt=0,
        description="Hard per-task ceiling; must exceed MAX_WALL_CLOCK_SECONDS.",
    )

    # -------------------------------------------------------------------- api
    API_MAX_UPLOAD_BYTES: int = Field(
        default=64 * 1024 * 1024,
        gt=0,
        description=(
            "Refused before the body is buffered. Independent of "
            "MAX_TOTAL_PIXELS, which caps what a valid upload may decode to."
        ),
    )
    API_TITLE: str = Field(default="RBI Complaint Document Intelligence", min_length=1)

    # ------------------------------------------------------------------- paths
    MODELS_DIR: Path = Field(default=Path("models"))
    CORPUS_DIR: Path = Field(default=Path("corpus"))
    LOG_LEVEL: LogLevel = Field(default="INFO")

    # ------------------------------------------------------------- validation
    @model_validator(mode="after")
    def _internal_consistency(self) -> Self:
        """Reject configurations that are self-contradictory.

        Note what is *not* here: the residency gate. See the module docstring.
        """
        if self.SLM_MAX_TOKENS > self.JOB_TOKEN_BUDGET:
            raise ValueError(
                f"SLM_MAX_TOKENS ({self.SLM_MAX_TOKENS}) exceeds "
                f"JOB_TOKEN_BUDGET ({self.JOB_TOKEN_BUDGET}): a single call "
                "could never complete within the job budget"
            )
        return self

    # -------------------------------------------------------------- accessors
    @property
    def is_remote_profile(self) -> bool:
        """True when the profile routes any text to a non-local endpoint."""
        return self.PROVIDER_PROFILE is ProviderProfile.LAPTOP

    @property
    def gpu_requested(self) -> bool:
        """Whether this configuration asks for CUDA at all.

        ``auto`` is profile-driven rather than "use whatever is installed": a
        laptop that happens to have ``onnxruntime-gpu`` in the venv must keep
        producing the CPU numbers the quality gate was calibrated against.
        """
        if self.GPU_EXECUTION == "never":
            return False
        if self.GPU_EXECUTION == "require":
            return True
        return self.PROVIDER_PROFILE is ProviderProfile.GPU_NODE

    @property
    def gpu_required(self) -> bool:
        """Whether a missing GPU is a startup failure rather than a fallback."""
        return self.GPU_EXECUTION == "require"

    @property
    def uses_remote_endpoint(self) -> bool:
        """Whether this profile's builder constructs a network-facing provider.

        ``cpu_node`` is :class:`LocalProvider` alone and never reads
        ``HF_BASE_URL``, so its value is inert and must not be judged — a
        default ``.env`` on an air-gapped CPU node still carries the router URL
        and is in no way a breach.  Keep this in step with the factory's
        ``_REGISTRY``; a contract test asserts they agree.
        """
        return self.PROVIDER_PROFILE in (
            ProviderProfile.LAPTOP,
            ProviderProfile.GPU_NODE,
        )

    @property
    def local_endpoint_hosts(self) -> frozenset[str]:
        """Operator-declared on-premises hostnames, normalised."""
        return frozenset(
            host.strip().lower().rstrip(".")
            for host in self.LOCAL_ENDPOINT_HOSTS.split(",")
            if host.strip()
        )

    @property
    def slm_endpoint_is_local(self) -> bool:
        """Whether ``HF_BASE_URL`` names a host inside the deployment."""
        return is_local_endpoint(self.HF_BASE_URL, self.local_endpoint_hosts)

    @property
    def slm_endpoint_configured(self) -> bool:
        """Whether an operator has actually pointed this node at a server.

        A node still carrying :data:`DEFAULT_HF_BASE_URL` has not been
        configured; it has merely been left alone.  Distinguishing the two is
        what lets an air-gapped deployment with **no model server at all** run
        without an SLM instead of failing to start — which is the shape of a
        genuinely self-contained node, where the only models on the box are the
        local OCR/LID/MT artifacts.
        """
        return self.HF_BASE_URL != DEFAULT_HF_BASE_URL

    @property
    def slm_available(self) -> bool:
        """Whether this configuration can call an SLM at all.

        False on an air-gapped profile with no in-house endpoint: extraction
        falls back to the deterministic validators, which is a documented,
        degraded — but correct and *offline* — mode, not an error.
        """
        if self.is_remote_profile:
            return True
        return self.slm_endpoint_configured and self.slm_endpoint_is_local

    @property
    def requires_remote_credential(self) -> bool:
        """Whether the SLM endpoint is one that needs a real API token.

        A local serving stack (vLLM, llama.cpp, Ollama, TGI) accepts any
        non-empty key and is usually run with none at all, so demanding a token
        there turns a working air-gapped node into a startup failure for no
        security benefit.  A public endpoint genuinely needs one.
        """
        return not self.slm_endpoint_is_local

    @property
    def residency_violation_reason(self) -> str | None:
        """The refusal message for the WS-C gate, or ``None`` when compliant.

        The gate itself lives in the provider factory (BUILD-BRIEF M2); the
        message lives here so it is written once and asserted once.

        Two independent ways to breach residency, because the profile name is
        an *intent* and the base URL is the *fact*:

        1. ``laptop`` is defined as routing text off-premises, so it is allowed
           only with ``SYNTHETIC_ONLY=true``;
        2. ``gpu_node`` is defined as air-gapped, so if it has been pointed at
           an endpoint **at all**, that endpoint must be on-premises.  Without
           this check a ``gpu_node`` aimed off-premises ships real complainant
           text there with no error — the profile name alone was doing the
           protecting, and it protects nothing.

        Two configurations are deliberately *not* violations:

        * ``cpu_node`` at any base URL — it never builds a client, so the value
          is inert and judging it would refuse every air-gapped CPU deployment
          that left the shipped ``.env`` alone;
        * an air-gapped profile still carrying :data:`DEFAULT_HF_BASE_URL` —
          that node has not been *configured* with a model server, it has been
          left alone.  It runs with no SLM (see :attr:`slm_available`) and
          never opens the socket, which is exactly what a self-contained box
          with no serving stack needs.  Refusing to start there would be
          refusing the most common air-gapped shape.
        """
        if self.is_remote_profile and not self.SYNTHETIC_ONLY:
            return (
                "laptop profile routes text to a non-Indian endpoint; "
                "permitted only with SYNTHETIC_ONLY=true"
            )
        if (
            self.uses_remote_endpoint
            and not self.is_remote_profile
            and self.slm_endpoint_configured
            and not self.slm_endpoint_is_local
        ):
            host = urlsplit(self.HF_BASE_URL).hostname or self.HF_BASE_URL
            return (
                f"profile {self.PROVIDER_PROFILE.value} is an air-gapped "
                f"profile but HF_BASE_URL points at {host!r}, which is not an "
                "on-premises host; real complainant text would leave the "
                "deployment. Point HF_BASE_URL at the in-house model server, "
                "or name the host in LOCAL_ENDPOINT_HOSTS if it is internal"
            )
        return None

    @property
    def hf_token_present(self) -> bool:
        """True when a non-empty ``HF_TOKEN`` is configured."""
        return bool(self.HF_TOKEN and self.HF_TOKEN.get_secret_value())

    @property
    def has_stray_remote_credential(self) -> bool:
        """A remote token is present on a profile that never reaches out.

        Reported, not raised: the same ``.env`` legitimately serves a developer
        switching profiles to run the dual-profile contract tests
        (BUILD-BRIEF M6).  On an air-gapped node this is a deployment finding,
        so the startup banner and WS-C surface it.
        """
        return self.hf_token_present and not self.is_remote_profile

    @property
    def provider_cache_path(self) -> Path:
        """``PROVIDER_CACHE_PATH`` resolved against the repository root."""
        return self._resolve(self.PROVIDER_CACHE_PATH)

    @property
    def models_dir(self) -> Path:
        """``MODELS_DIR`` resolved against the repository root."""
        return self._resolve(self.MODELS_DIR)

    @property
    def corpus_dir(self) -> Path:
        """``CORPUS_DIR`` resolved against the repository root."""
        return self._resolve(self.CORPUS_DIR)

    @property
    def ifsc_master_path(self) -> Path:
        """``IFSC_MASTER_PATH`` resolved against the repository root (Q8)."""
        return self._resolve(self.IFSC_MASTER_PATH)

    @property
    def bank_aliases_path(self) -> Path:
        """``BANK_ALIASES_PATH`` resolved against the repository root."""
        return self._resolve(self.BANK_ALIASES_PATH)

    @property
    def celery_broker_url(self) -> str:
        """Broker URL, falling back to ``REDIS_URL`` (PLAN.md D2)."""
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        """Result backend, falling back to ``REDIS_URL``."""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    @property
    def database_url(self) -> str:
        """``DATABASE_URL`` with a relative SQLite path anchored to the repo.

        Only SQLite is rewritten, and only when its path is relative: a
        ``sqlite:///./data/runtime/app.sqlite`` default that follows the current
        working directory means the API and a Celery worker started from
        different shells silently use different databases.  Every other driver
        is returned untouched — a Postgres DSN has no filesystem to anchor.
        """
        url = self.DATABASE_URL
        prefix, separator, remainder = url.partition(":///")
        if not separator or not prefix.startswith("sqlite"):
            return url
        if remainder in ("", ":memory:") or remainder.startswith(":memory:"):
            return url
        candidate = Path(remainder)
        if candidate.is_absolute():
            return url
        return f"{prefix}:///{self._resolve(candidate).as_posix()}"

    @property
    def webhook_enabled(self) -> bool:
        """True when a destination and a signing secret are both configured.

        Both are required: an unsigned webhook is a result envelope posted to
        whatever answers that URL, and the receiver has no way to tell it came
        from this service.
        """
        return bool(
            self.WEBHOOK_URL
            and self.WEBHOOK_HMAC_SECRET
            and self.WEBHOOK_HMAC_SECRET.get_secret_value()
            and self.WEBHOOK_DELIVERY_MODE != "disabled"
        )

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def thread_budget(self) -> int:
        """``CELERY_CONCURRENCY x ORT_INTRA_OP_THREADS`` (PRD P0-11)."""
        return self.CELERY_CONCURRENCY * self.ORT_INTRA_OP_THREADS

    def assert_thread_budget(self, physical_cores: int | None = None) -> None:
        """Raise when the thread budget oversubscribes the machine.

        Called by CI (PRD P0-11), never on import: a 2-core runner must still
        be able to load the documented defaults.

        Raises:
            ValueError: when ``concurrency x intra_op`` exceeds the core count.
        """
        cores = physical_cores if physical_cores is not None else _cpu_count()
        if self.thread_budget > cores:
            raise ValueError(
                f"CELERY_CONCURRENCY ({self.CELERY_CONCURRENCY}) x "
                f"ORT_INTRA_OP_THREADS ({self.ORT_INTRA_OP_THREADS}) = "
                f"{self.thread_budget} exceeds {cores} available cores; "
                "oversubscription costs 3-5x (BUILD-BRIEF sec. 8)"
            )

    def startup_banner(self) -> str:
        """One-line profile banner for the startup log (BUILD-BRIEF M2).

        Contains no secret: ``HF_TOKEN`` is reported as present/absent only.
        """
        return (
            f"profile={self.PROVIDER_PROFILE.value} "
            f"synthetic_only={str(self.SYNTHETIC_ONLY).lower()} "
            f"slm={self.SLM_MODEL} "
            f"hf_token={'set' if self.hf_token_present else 'unset'} "
            f"threads={self.CELERY_CONCURRENCY}x{self.ORT_INTRA_OP_THREADS}"
        )


def _cpu_count() -> int:
    """Usable CPU count, honouring affinity masks where the OS reports them."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:  # pragma: no cover - POSIX only
        return len(affinity(0))
    return os.cpu_count() or 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because every stage reads it; call :func:`reset_settings_cache` in
    tests after mutating the environment.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the :func:`get_settings` cache (tests and CLI entrypoints)."""
    get_settings.cache_clear()
