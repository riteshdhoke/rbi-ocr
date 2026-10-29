"""Profile selection, the residency gate, and the synthetic-corpus gate.

BUILD-BRIEF sec. 9 calls this "the one failure this system cannot recover
from": complainant data leaving the country during development.  Two gates
close that path and neither has a bypass flag, now or later.

**The residency gate.**  ``PROVIDER_PROFILE=laptop`` routes SLM and VLM traffic
to a non-Indian endpoint.  Combined with ``SYNTHETIC_ONLY=false`` that is a
breach, so :func:`get_provider` refuses to construct anything —
:class:`~app.contracts.errors.ResidencyViolation`, at startup, before a single
byte can be read.  The refusal string lives on ``Settings`` (WS-B), so the gate
and its test cannot drift apart.

**The synthetic-corpus gate.**  Under the same profile,
:func:`require_synthetic_corpus_file` refuses any input that does not carry a
sidecar declaring ``"synthetic": true``.  The provider gate alone is not
enough: a developer who sets ``SYNTHETIC_ONLY=true`` and then points the
pipeline at a real scan has satisfied the flag and violated the policy.

Profiles are **composed**, not monolithic.  PLAN.md D1 puts OCR, LID and MT on
the local CPU everywhere while >=0.9B SLM/VLM weights are never downloaded, so
the ``laptop`` provider is a :class:`CompositeProvider` that routes three
methods to :mod:`app.providers.hf_api` and five to
:mod:`app.providers.local`.  Which side owns which method is the single
``REMOTE_METHODS`` set below, so the routing is auditable in one place.

This module does NOT handle:
  * reading ``.env`` (:mod:`app.settings`) or applying thread caps (:mod:`app`);
  * starting a metrics HTTP server.  It registers a gauge; whether anything
    scrapes it is WS-F's concern, and ``prometheus_client`` is a no-op without
    a server;
  * ingestion — :func:`load_corpus_file` reads bytes and enforces residency
    only.  MIME sniffing, resource caps and idempotency are Stage 1 (WS-E);
  * validating ground-truth *content*.  It checks one boolean; the corpus
    schema belongs to WS-D;
  * provider lifetime.  Callers own the object they are handed and may call
    :func:`close_provider` when done.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from prometheus_client import Gauge

from app.contracts.errors import ResidencyViolation
from app.contracts.ocr import (
    ExtractionContext,
    LangResult,
    OCRLine,
    PageClass,
    RawField,
    ScriptResult,
    StructuralFeatures,
)
from app.providers.base import CropImage, ModelProvider, PageImage
from app.providers.cache import ResponseCache
from app.settings import ProviderProfile, Settings, get_settings

__all__ = [
    "DIRECTORY_MANIFEST_FILENAME",
    "PROVIDER_PROFILE_GAUGE",
    "SIDECAR_SUFFIX",
    "SYNTHETIC_KEY",
    "CompositeProvider",
    "build_cpu_node_provider",
    "build_gpu_node_provider",
    "build_laptop_provider",
    "close_provider",
    "get_provider",
    "is_declared_synthetic",
    "load_corpus_file",
    "log_startup_banner",
    "require_synthetic_corpus_file",
    "set_provider_profile_metric",
    "synthetic_sidecar_for",
]

_LOG = logging.getLogger(__name__)

#: One time series per profile: 1 for the active one, 0 for the rest.  A gauge
#: rather than an info metric so an alert can fire on ``provider_profile{
#: profile="laptop"} == 1`` in an environment that should never be on it.
PROVIDER_PROFILE_GAUGE = Gauge(
    "provider_profile",
    "Active model-provider profile (1 for the active profile, 0 otherwise).",
    ["profile"],
)


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


class CompositeProvider:
    """Routes each protocol method to the side of the split that owns it.

    The split is PLAN.md D1: >=0.9B SLM/VLM weights are never downloaded, so
    those three methods go remote; OCR, script ID, LID and MT are small CPU
    models and stay local, where the page pixels already are.

    Delegation is explicit — eight forwarding methods rather than
    ``__getattr__`` — so a typo in a method name is an ``AttributeError`` at
    import review time, not a silent fallthrough at 3 a.m.
    """

    #: Methods served by the remote endpoint.  The audit surface of this class.
    REMOTE_METHODS: Final[frozenset[str]] = frozenset(
        {"extract_fields", "extract_fields_batch", "ocr_page_heavy"}
    )

    def __init__(
        self,
        local: ModelProvider,
        remote: ModelProvider,
        *,
        name: str = "composite",
    ) -> None:
        self.local = local
        self.remote = remote
        self.name = name

    # ------------------------------------------------------------ local side
    def detect_script(self, crops: Sequence[CropImage]) -> ScriptResult:
        """Delegates to the local provider."""
        return self.local.detect_script(crops)

    def classify_page(
        self, page: PageImage, feats: StructuralFeatures
    ) -> PageClass:
        """Delegates to the local provider."""
        return self.local.classify_page(page, feats)

    def ocr_page(self, page: PageImage, script: str) -> list[OCRLine]:
        """Delegates to the local provider."""
        return self.local.ocr_page(page, script)

    def identify_language(self, text: str, script: str) -> LangResult:
        """Delegates to the local provider."""
        return self.local.identify_language(text, script)

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        """Delegates to the local provider (IndicTrans2 on CPU)."""
        return self.local.translate(texts, src, tgt)

    # ----------------------------------------------------------- remote side
    def ocr_page_heavy(self, page: PageImage) -> list[OCRLine]:
        """Delegates to the remote provider (Tier-2 VLM)."""
        return self.remote.ocr_page_heavy(page)

    def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]:
        """Delegates to the remote provider (SLM)."""
        return self.remote.extract_fields(ctx)

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        """Delegates to the remote provider, which owns the concurrency."""
        return self.remote.extract_fields_batch(ctxs)

    def close(self) -> None:
        """Close both sides, if they support it."""
        close_provider(self.local)
        close_provider(self.remote)

    def __repr__(self) -> str:
        return (
            f"CompositeProvider(name={self.name!r}, "
            f"local={type(self.local).__name__}, "
            f"remote={type(self.remote).__name__})"
        )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _shared_cache(settings: Settings, cache: ResponseCache | None) -> ResponseCache:
    return cache if cache is not None else ResponseCache(settings.provider_cache_path)


def build_laptop_provider(
    settings: Settings, *, cache: ResponseCache | None = None
) -> ModelProvider:
    """Dev laptop: local OCR/LID/MT, remote SLM/VLM over the HF router.

    Raises:
        ConfigurationError: when ``HF_TOKEN`` is absent — the profile cannot
            do its job without it and must say so at startup, not on the first
            extraction.
    """
    from app.providers.hf_api import HFAPIProvider
    from app.providers.local import LocalProvider

    return CompositeProvider(
        local=LocalProvider(settings),
        remote=HFAPIProvider(settings, cache=_shared_cache(settings, cache)),
        name="laptop",
    )


def build_cpu_node_provider(
    settings: Settings, *, cache: ResponseCache | None = None
) -> ModelProvider:
    """Air-gapped CPU node: everything local, nothing leaves the machine.

    ``extract_fields`` and ``ocr_page_heavy`` raise
    :class:`~app.providers.base.ProviderMethodNotSupported` here until the
    Phase 1 llama.cpp path lands (BUILD-BRIEF sec. 5).  That is the honest
    state of the profile, and a stage that needs the SLM will say so loudly
    rather than quietly returning nothing.
    """
    from app.providers.local import LocalProvider

    return LocalProvider(settings)


def build_gpu_node_provider(
    settings: Settings, *, cache: ResponseCache | None = None
) -> ModelProvider:
    """GPU node: local models on CUDA, plus an in-house SLM/VLM **if there is one**.

    Two supported shapes, and which one you get is decided by whether
    ``HF_BASE_URL`` has actually been pointed at a server:

    * **With an in-house endpoint** — a :class:`CompositeProvider`, structurally
      identical to the laptop profile.  ``HF_BASE_URL`` points at a local
      OpenAI-compatible server, which is exactly the substitution BUILD-BRIEF
      M2 designed the client around.  It must be on-premises; the residency
      gate in :func:`get_provider` has already refused it otherwise.
    * **Self-contained** — no serving stack on the box, only the local
      OCR/LID/MT artifacts.  This returns :class:`LocalProvider` alone, exactly
      like ``cpu_node``, and field extraction degrades to the deterministic
      validators.  It is the common air-gapped shape and it must not be a
      startup failure: building an ``HFAPIProvider`` against an endpoint that
      does not exist turns a working node into one that refuses to start (and,
      before the residency guard, into one that quietly called the public
      router instead).

    The GPU applies either way — it belongs to the local vision and translation
    models, not to the SLM.
    """
    from app.providers.local import LocalProvider

    local = LocalProvider(settings)
    if not settings.slm_available:
        _LOG.warning(
            "profile=gpu_node has no in-house model server (HF_BASE_URL is "
            "unset/default); field extraction will use the deterministic "
            "validators only. Point HF_BASE_URL at a local OpenAI-compatible "
            "server to enable the SLM."
        )
        return local

    from app.providers.hf_api import HFAPIProvider

    return CompositeProvider(
        local=local,
        remote=HFAPIProvider(settings, cache=_shared_cache(settings, cache)),
        name="gpu_node",
    )


#: Profile to builder.  Total over :class:`~app.settings.ProviderProfile`; a
#: test asserts that, so a new profile cannot be added without a builder.
_REGISTRY: Final[
    Mapping[ProviderProfile, Callable[..., ModelProvider]]
] = {
    ProviderProfile.LAPTOP: build_laptop_provider,
    ProviderProfile.CPU_NODE: build_cpu_node_provider,
    ProviderProfile.GPU_NODE: build_gpu_node_provider,
}


def get_provider(
    settings: Settings | None = None,
    *,
    cache: ResponseCache | None = None,
    log_banner: bool = True,
) -> ModelProvider:
    """Build the provider for the active profile — BUILD-BRIEF M2's gate.

    Order matters: the residency check runs **before** the registry lookup and
    before any construction, so a violating configuration cannot open a socket,
    a database or a model file on its way to being refused.

    Raises:
        ResidencyViolation: ``PROVIDER_PROFILE=laptop`` with
            ``SYNTHETIC_ONLY=false`` (PLAN.md invariant 6).
        ConfigurationError: the profile is unknown, or a credential the profile
            requires is absent.
    """
    active = settings if settings is not None else get_settings()

    reason = active.residency_violation_reason
    if reason is not None:
        raise ResidencyViolation(
            reason,
            context={
                "profile": active.PROVIDER_PROFILE.value,
                "synthetic_only": active.SYNTHETIC_ONLY,
                "remedy": "set SYNTHETIC_ONLY=true or change PROVIDER_PROFILE",
            },
        )

    builder = _REGISTRY.get(active.PROVIDER_PROFILE)
    if builder is None:  # pragma: no cover - unreachable while the map is total
        raise ResidencyViolation(
            f"no provider is registered for profile "
            f"{active.PROVIDER_PROFILE.value!r}; refusing to guess",
            context={"profile": active.PROVIDER_PROFILE.value},
        )

    set_provider_profile_metric(active.PROVIDER_PROFILE)
    if log_banner:
        log_startup_banner(active)
    return builder(active, cache=cache)


def close_provider(provider: object) -> None:
    """Close a provider if it holds resources.  Safe on any object."""
    closer = getattr(provider, "close", None)
    if callable(closer):
        closer()


def set_provider_profile_metric(profile: ProviderProfile) -> None:
    """Set the ``provider_profile`` gauge to 1 for ``profile``, 0 elsewhere."""
    for candidate in ProviderProfile:
        PROVIDER_PROFILE_GAUGE.labels(profile=candidate.value).set(
            1.0 if candidate is profile else 0.0
        )


def log_startup_banner(
    settings: Settings, *, logger: logging.Logger | None = None
) -> str:
    """Log and return the one-line profile banner (BUILD-BRIEF M2).

    Carries no secret — ``Settings.startup_banner`` reports ``HF_TOKEN`` as
    set/unset.  A token present on a profile that never reaches out is a
    deployment finding, so it is warned about rather than ignored.
    """
    log = logger if logger is not None else _LOG
    banner = settings.startup_banner()
    log.info("provider profile active | %s", banner)
    if settings.has_stray_remote_credential:
        log.warning(
            "HF_TOKEN is set but profile=%s never calls a remote endpoint; "
            "remove it from this node's .env",
            settings.PROVIDER_PROFILE.value,
        )
    if settings.is_remote_profile:
        log.warning(
            "profile=laptop routes SLM/VLM text to a non-Indian endpoint; "
            "synthetic data only (PLAN.md invariant 6)"
        )
    return banner


# --------------------------------------------------------------------------- #
# The synthetic-corpus gate
# --------------------------------------------------------------------------- #

#: Per-file ground-truth sidecar written by ``tools/generate_test_corpus.py``.
SIDECAR_SUFFIX: Final[str] = ".gt.json"

#: Directory-level declaration, used by the adversarial set, whose inputs have
#: no per-file ground truth *by construction* (a zero-byte PDF has no lines).
DIRECTORY_MANIFEST_FILENAME: Final[str] = "MANIFEST.json"

#: The key both forms must set to ``true``.
SYNTHETIC_KEY: Final[str] = "synthetic"


def synthetic_sidecar_for(path: Path) -> Path:
    """Sibling ground-truth path for ``path`` (``x.jpg`` -> ``x.gt.json``)."""
    return path.with_suffix(SIDECAR_SUFFIX)


def _declares_synthetic(path: Path) -> bool:
    """Whether a JSON file exists and sets ``synthetic`` to exactly ``true``.

    A malformed or unreadable sidecar is **not** a declaration.  Failing open
    on a parse error is precisely how a real document gets processed under the
    laptop profile.
    """
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get(SYNTHETIC_KEY) is True


def is_declared_synthetic(path: Path) -> tuple[bool, Path | None]:
    """Whether ``path`` is declared synthetic, and by which sidecar.

    The per-file ``.gt.json`` wins.  A directory ``MANIFEST.json`` is the
    fallback, because the adversarial inputs of TEST-DATA-SPEC sec. 5 cannot
    carry per-file ground truth and must still be loadable under the laptop
    profile.  Both are positive declarations of provenance; there is no third
    form and no flag.
    """
    sidecar = synthetic_sidecar_for(path)
    if _declares_synthetic(sidecar):
        return True, sidecar
    manifest = path.parent / DIRECTORY_MANIFEST_FILENAME
    if _declares_synthetic(manifest):
        return True, manifest
    return False, None


def require_synthetic_corpus_file(
    path: Path, settings: Settings | None = None
) -> Path:
    """Refuse ``path`` unless the active profile permits it.

    Under a non-remote profile this is a no-op beyond the existence check —
    an air-gapped node processes real complaints, which is its whole purpose.

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        ResidencyViolation: laptop profile and no ``synthetic: true`` sidecar.
    """
    active = settings if settings is not None else get_settings()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"corpus file does not exist: {target}")
    if not active.is_remote_profile:
        return target
    declared, sidecar = is_declared_synthetic(target)
    if not declared:
        raise ResidencyViolation(
            f"{target.name} carries no synthetic-data declaration and the "
            f"{active.PROVIDER_PROFILE.value} profile routes text to a "
            "non-Indian endpoint; expected a sibling "
            f"{synthetic_sidecar_for(target).name} (or a directory "
            f"{DIRECTORY_MANIFEST_FILENAME}) containing "
            f'"{SYNTHETIC_KEY}": true. There is no bypass flag.',
            context={
                "path": str(target),
                "expected_sidecar": str(synthetic_sidecar_for(target)),
                "profile": active.PROVIDER_PROFILE.value,
            },
        )
    _LOG.debug("synthetic declaration accepted for %s via %s", target, sidecar)
    return target


def load_corpus_file(path: Path, settings: Settings | None = None) -> bytes:
    """Read a corpus file after :func:`require_synthetic_corpus_file` passes.

    Returns raw bytes deliberately: Stage 1 sniffs MIME from magic bytes, and
    handing it a decoded object here would hand it a decision it must make
    itself.
    """
    return require_synthetic_corpus_file(path, settings).read_bytes()
