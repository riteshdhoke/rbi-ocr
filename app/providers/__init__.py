"""Model providers — one protocol, three profiles, two gates.

Import order matters here, so the package deliberately exposes only the light
half eagerly:

* :mod:`app.providers.base` (protocol, errors, ``JobBudget``),
  :mod:`app.providers.cache` and :mod:`app.providers.factory` import nothing
  numeric and no HTTP client, so ``import app.providers`` stays cheap for the
  API process, which never touches pixels;
* :class:`~app.providers.local.LocalProvider` (NumPy, OpenCV, ONNX Runtime) and
  :class:`~app.providers.hf_api.HFAPIProvider` (``openai``, ``httpx``) are
  resolved on first attribute access via :func:`__getattr__`, and by the
  factory only when a profile actually needs them.

That laziness is not cosmetic: PLAN.md invariant 5 requires the thread caps in
:mod:`app` to be in ``os.environ`` before NumPy or ONNX Runtime load, and the
fewer modules that drag them in incidentally, the fewer places that can go
wrong.

This package does NOT handle: stage logic, the grounding gate, ingestion, or
downloading weights (``tools/fetch_models.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.providers.base import (
    PROVIDER_METHODS,
    JobBudget,
    ModelArtifactMissing,
    ModelProvider,
    ProviderMethodNotSupported,
    Reservation,
    inject_truststore,
    to_page_image,
)
from app.providers.cache import (
    CACHE_SCHEMA_VERSION,
    CacheEntry,
    CacheStats,
    ResponseCache,
    make_cache_key,
)
from app.providers.factory import (
    CompositeProvider,
    close_provider,
    get_provider,
    is_declared_synthetic,
    load_corpus_file,
    log_startup_banner,
    require_synthetic_corpus_file,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; runtime access is lazy
    from app.providers.hf_api import HFAPIProvider
    from app.providers.local import LocalProvider

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "PROVIDER_METHODS",
    "CacheEntry",
    "CacheStats",
    "CompositeProvider",
    "HFAPIProvider",
    "JobBudget",
    "LocalProvider",
    "ModelArtifactMissing",
    "ModelProvider",
    "ProviderMethodNotSupported",
    "Reservation",
    "ResponseCache",
    "close_provider",
    "get_provider",
    "inject_truststore",
    "is_declared_synthetic",
    "load_corpus_file",
    "log_startup_banner",
    "make_cache_key",
    "require_synthetic_corpus_file",
    "to_page_image",
]

_LAZY: dict[str, str] = {
    "LocalProvider": "app.providers.local",
    "HFAPIProvider": "app.providers.hf_api",
}


def __getattr__(name: str) -> Any:
    """Resolve the heavy providers on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
