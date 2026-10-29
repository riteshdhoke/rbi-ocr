"""The ``ModelProvider`` protocol and the vocabulary every provider shares.

BUILD-BRIEF sec. 3.1 fixes eight methods.  Their whole point is that a stage
module never learns *where* a model runs:

* ``detect_script``, ``classify_page``, ``ocr_page``, ``identify_language`` and
  ``translate`` are CPU-local everywhere in Phase 0;
* ``extract_fields`` / ``extract_fields_batch`` and ``ocr_page_heavy`` are the
  SLM/VLM calls, which PLAN.md D1 forbids downloading to the dev laptop and
  therefore reach over HTTP under the ``laptop`` profile;
* ``extract_fields_batch`` exists so the **provider** decides whether
  concurrency means ``asyncio.gather`` or a sequential loop.  ``asyncio`` must
  never appear in a stage module — see the guard test in
  ``tests/contract/test_provider_contract.py``.

The image currency of this layer is a **NumPy ``uint8`` array**, ``(H, W)``
grayscale or ``(H, W, 3)`` **BGR** — the OpenCV convention — because detection
pre-processing, the quality gate and the page classifier are all OpenCV/ONNX.
BUILD-BRIEF spells the type ``Image``; it does not say *which* ``Image``, and a
PIL object would force a conversion on the hottest path.  :func:`to_page_image`
converts a PIL image for callers holding one.

Two error classes live here rather than in :mod:`app.contracts.errors`, which is
another workstream's lane:

``ProviderMethodNotSupported``
    The method is not implementable by this provider at all (a local-only
    method asked of the API client, or the reverse).  Distinct from
    :class:`~app.contracts.errors.ProviderUnavailable`, which means "should
    work, currently cannot" and is retryable.
``ModelArtifactMissing``
    Weights are absent from ``models/``.  It subclasses ``ProviderUnavailable``
    (whose docstring already names "the local weights are missing") and carries
    the path, the artifact name and **how to obtain it**, because that message
    is the entire user experience of a fresh air-gapped deployment.

This module does NOT handle:
  * constructing providers or choosing a profile — :mod:`app.providers.factory`
    owns that, including the residency gate;
  * any inference, HTTP call, model load or image decoding.  It imports no
    numeric library, no ``openai`` and no ``onnxruntime``, so ``import
    app.providers`` stays cheap and PLAN.md invariant 5 (thread caps before
    numeric imports) is never at risk from this file;
  * the grounding gate (invariant 8), validator logic, or promoting a
    :class:`~app.contracts.ocr.RawField` to an envelope ``Field`` — those are
    WS-E's, and a provider must never assert provenance it does not own;
  * counting tokens accurately.  :class:`JobBudget` accounts what it is *told*;
    the estimate used to reserve headroom lives with the provider that knows
    its own tokenizer.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, runtime_checkable

from app.contracts.errors import (
    ProviderError,
    ProviderUnavailable,
    TokenBudgetExceeded,
)
from app.contracts.ocr import (
    ExtractionContext,
    LangResult,
    OCRLine,
    PageClass,
    RawField,
    ScriptResult,
    StructuralFeatures,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps NumPy out of import
    import numpy as np
    from numpy.typing import NDArray

    #: A rasterised page: ``(H, W)`` grayscale or ``(H, W, 3)`` BGR ``uint8``.
    PageImage: TypeAlias = NDArray[np.uint8]
    #: A single detected text-line crop, same dtype and channel convention.
    CropImage: TypeAlias = NDArray[np.uint8]
else:  # pragma: no cover - annotations are postponed; this is a runtime alias
    PageImage = Any
    CropImage = Any

__all__ = [
    "PROVIDER_METHODS",
    "CropImage",
    "JobBudget",
    "ModelArtifactMissing",
    "ModelProvider",
    "PageImage",
    "ProviderMethodNotSupported",
    "Reservation",
    "inject_truststore",
    "to_page_image",
    "truststore_injected",
]

#: The eight methods of BUILD-BRIEF sec. 3.1, in brief order.  The contract
#: suite asserts every provider implements exactly these.
PROVIDER_METHODS: tuple[str, ...] = (
    "detect_script",
    "classify_page",
    "ocr_page",
    "ocr_page_heavy",
    "identify_language",
    "extract_fields",
    "extract_fields_batch",
    "translate",
)


# --------------------------------------------------------------------------- #
# Errors owned by the provider layer
# --------------------------------------------------------------------------- #


class ProviderMethodNotSupported(ProviderError):
    """This provider cannot implement this method, ever.

    Not a transient failure and **not retryable**: the API client cannot run
    PP-OCR detection, and the local provider has no SLM in Phase 0.  The
    factory composes a provider per profile precisely so a stage never sees
    this in normal operation; when it does, it is a routing bug and the
    message names the profile that *does* implement the method.
    """

    code: ClassVar[str] = "PROVIDER_METHOD_NOT_SUPPORTED"
    http_status: ClassVar[int] = 501

    def __init__(
        self,
        method: str,
        *,
        provider: str,
        implemented_by: str,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            f"{provider} does not implement {method}(); it is implemented by "
            f"{implemented_by}. Routing bug: the factory composes providers so "
            "no stage should reach this call.",
            context=context,
        )
        self.method: str = method
        self.provider: str = provider
        self.implemented_by: str = implemented_by
        self.context.setdefault("method", method)
        self.context.setdefault("provider", provider)
        self.context.setdefault("implemented_by", implemented_by)


class ModelArtifactMissing(ProviderUnavailable):
    """A weight file the local provider needs is not in ``models/``.

    Subclasses :class:`~app.contracts.errors.ProviderUnavailable` because that
    class already documents "the local weights are missing"; it is nonetheless
    **not** retryable, and the retry helper in :mod:`app.providers.hf_api`
    never sees it (it is raised only by local loaders).

    ``how_to_obtain`` is mandatory.  On an air-gapped node this string is the
    entire remediation path an operator gets, so it names the exact tool or
    artefact, never "download the model".
    """

    code: ClassVar[str] = "MODEL_ARTIFACT_MISSING"
    http_status: ClassVar[int] = 503

    def __init__(
        self,
        *,
        artifact: str,
        path: Path,
        how_to_obtain: str,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            f"model artifact {artifact!r} is missing at {path}. {how_to_obtain}",
            context=context,
        )
        self.artifact: str = artifact
        self.path: Path = path
        self.how_to_obtain: str = how_to_obtain
        self.context.setdefault("artifact", artifact)
        self.context.setdefault("path", str(path))
        self.context.setdefault("how_to_obtain", how_to_obtain)


# --------------------------------------------------------------------------- #
# Per-job token budget
# --------------------------------------------------------------------------- #


class Reservation:
    """Headroom claimed from a :class:`JobBudget` for one in-flight call.

    Reserve-then-settle exists because the true cost of a call is known only
    from the response's ``usage`` block, while the budget must be enforced
    *before* the request leaves.  A reservation that is never settled (the call
    raised) is released, so a failed request costs nothing.
    """

    __slots__ = ("_budget", "_settled", "estimated")

    def __init__(self, budget: JobBudget, estimated: int) -> None:
        self._budget = budget
        self.estimated: int = estimated
        self._settled: bool = False

    @property
    def settled(self) -> bool:
        """True once :meth:`settle` or :meth:`release` has run."""
        return self._settled

    def settle(self, actual_tokens: int) -> None:
        """Convert this reservation into ``actual_tokens`` of real usage.

        Raises:
            ValueError: on a negative count or a double settle.
        """
        if actual_tokens < 0:
            raise ValueError(f"actual_tokens must be >= 0, got {actual_tokens}")
        if self._settled:
            raise ValueError("reservation already settled")
        self._settled = True
        self._budget._settle(self.estimated, actual_tokens)

    def release(self) -> None:
        """Give the headroom back; the call did not happen."""
        if self._settled:
            return
        self._settled = True
        self._budget._settle(self.estimated, 0)


class JobBudget:
    """Per-job token ceiling (``JOB_TOKEN_BUDGET``), enforced, thread-safe.

    Thread-safe because ``extract_fields_batch`` may run many calls at once and
    the budget is the one piece of shared mutable state between them.  Breach
    raises :class:`~app.contracts.errors.TokenBudgetExceeded`.

    This class does **not** count tokens: it accounts what callers reserve and
    settle.  Estimation belongs to whoever knows the tokenizer.
    """

    __slots__ = ("_lock", "_reserved", "_used", "job_id", "limit")

    def __init__(self, limit: int, *, job_id: str | None = None) -> None:
        if limit <= 0:
            raise ValueError(f"token budget must be positive, got {limit}")
        self.limit: int = limit
        self.job_id: str | None = job_id
        self._used: int = 0
        self._reserved: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ accessors
    @property
    def used(self) -> int:
        """Tokens settled so far."""
        with self._lock:
            return self._used

    @property
    def reserved(self) -> int:
        """Tokens claimed by calls that are still in flight."""
        with self._lock:
            return self._reserved

    @property
    def remaining(self) -> int:
        """Headroom left, never negative; in-flight reservations count."""
        with self._lock:
            return max(0, self.limit - self._used - self._reserved)

    # -------------------------------------------------------------- claiming
    def reserve(self, estimated_tokens: int) -> Reservation:
        """Claim ``estimated_tokens`` of headroom.

        Raises:
            ValueError: on a negative estimate.
            TokenBudgetExceeded: when the claim would breach the ceiling.
        """
        if estimated_tokens < 0:
            raise ValueError(
                f"estimated_tokens must be >= 0, got {estimated_tokens}"
            )
        with self._lock:
            projected = self._used + self._reserved + estimated_tokens
            if projected > self.limit:
                raise TokenBudgetExceeded(
                    f"job token budget exhausted: {self._used} used + "
                    f"{self._reserved} in flight + {estimated_tokens} requested "
                    f"exceeds {self.limit}",
                    budget=self.limit,
                    used=self._used + self._reserved,
                    context=(
                        {"job_id": self.job_id} if self.job_id is not None else None
                    ),
                )
            self._reserved += estimated_tokens
        return Reservation(self, estimated_tokens)

    @contextmanager
    def reservation(self, estimated_tokens: int) -> Iterator[Reservation]:
        """Reserve for the duration of a block, releasing on any exception.

        Settle inside the block once ``usage`` is known; leaving without
        settling releases the claim, so a failed call costs no budget.
        """
        claim = self.reserve(estimated_tokens)
        try:
            yield claim
        finally:
            claim.release()

    def _settle(self, reserved: int, actual: int) -> None:
        """Internal: swap ``reserved`` headroom for ``actual`` usage."""
        with self._lock:
            self._reserved = max(0, self._reserved - reserved)
            self._used += actual

    def snapshot(self) -> dict[str, int | str | None]:
        """A JSON-safe view for logs and ``ExtractionContext`` construction."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "limit": self.limit,
                "used": self._used,
                "reserved": self._reserved,
                "remaining": max(0, self.limit - self._used - self._reserved),
            }


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class ModelProvider(Protocol):
    """BUILD-BRIEF sec. 3.1, verbatim in method set and order.

    Every implementation either performs the work or raises
    :class:`ProviderMethodNotSupported`; none may return a plausible-looking
    default (BUILD-BRIEF sec. 0.5).
    """

    def detect_script(self, crops: Sequence[CropImage]) -> ScriptResult:
        """Majority-vote the script of a page from sampled line crops."""
        ...

    def classify_page(
        self, page: PageImage, feats: StructuralFeatures
    ) -> PageClass:
        """Classify one page from pixels plus structural features."""
        ...

    def ocr_page(self, page: PageImage, script: str) -> list[OCRLine]:
        """Tier-1 recognition: detection then **batched** rec (invariant 4)."""
        ...

    def ocr_page_heavy(self, page: PageImage) -> list[OCRLine]:
        """Tier-2 escalation: the heavy VLM path."""
        ...

    def identify_language(self, text: str, script: str) -> LangResult:
        """Per-**line** LID, label set constrained by ``script`` (invariant 1)."""
        ...

    def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]:
        """SLM extraction for one page, keyed by field name."""
        ...

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        """Extraction over many pages; **the provider owns the concurrency**."""
        ...

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        """Per-field MT.  Never a whole page (BUILD-BRIEF sec. 8)."""
        ...


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

_TRUSTSTORE_LOCK = threading.Lock()
_TRUSTSTORE_INJECTED = False


def inject_truststore() -> bool:
    """Route TLS verification through the OS certificate store, if possible.

    The office network intercepts TLS with a corporate CA that lives in the
    Windows certificate store and **not** in ``certifi`` (PLAN.md working
    agreements, OPEN-QUESTIONS D7).  ``truststore`` is a dev-time convenience,
    so a missing import is not an error — it returns ``False``.

    Never disables verification, and never will.  Idempotent and thread-safe.

    Returns:
        ``True`` if the OS trust store is now in use (including on a repeat
        call), ``False`` if ``truststore`` is not installed.
    """
    global _TRUSTSTORE_INJECTED
    with _TRUSTSTORE_LOCK:
        if _TRUSTSTORE_INJECTED:
            return True
        try:
            import truststore
        except ImportError:
            return False
        truststore.inject_into_ssl()
        _TRUSTSTORE_INJECTED = True
        return True


def truststore_injected() -> bool:
    """Whether :func:`inject_truststore` has successfully run in this process."""
    with _TRUSTSTORE_LOCK:
        return _TRUSTSTORE_INJECTED


def to_page_image(image: object) -> PageImage:
    """Coerce a PIL image or array-like into this layer's :data:`PageImage`.

    Accepts anything NumPy can view as an array — NumPy arrays pass through,
    PIL images convert through the array interface.  RGB input is reordered to
    **BGR** so a caller holding a PIL image gets the same channel order the
    ONNX pre-processing expects.

    Raises:
        TypeError: when the object cannot be viewed as a numeric array.
        ValueError: when the result is not 2-D or 3-D with 1/3/4 channels.
    """
    import numpy as np  # local: keeps NumPy out of `import app.providers`

    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot convert {type(image).__name__} to a page image; pass a "
            "NumPy uint8 array or a PIL Image"
        ) from exc
    if array.ndim == 0 or array.dtype.kind not in "uifb":
        raise TypeError(
            f"cannot convert {type(image).__name__} to a page image; pass a "
            "NumPy uint8 array or a PIL Image"
        )
    if array.ndim not in (2, 3):
        raise ValueError(
            f"page image must be 2-D or 3-D, got shape {array.shape!r}"
        )
    if array.ndim == 3:
        channels = array.shape[2]
        if channels not in (1, 3, 4):
            raise ValueError(
                f"page image must have 1, 3 or 4 channels, got {channels}"
            )
        if channels == 4:
            array = array[:, :, :3]
        if channels in (3, 4) and not isinstance(image, np.ndarray):
            # PIL hands back RGB; this layer speaks BGR.
            array = array[:, :, ::-1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
