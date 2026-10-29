"""RBI complaint document-intelligence service — package root.

The only job of this module is PLAN.md **invariant 5**: CPU thread caps are set
*before any numeric library is imported*.  ONNX Runtime, OpenMP, MKL, OpenBLAS
and NumExpr each read their thread-count environment variable **once, at import
time**, and each independently defaults to ``nproc``.  Three runtimes that all
believe they own every core cost 3-5x on the hottest path (BUILD-BRIEF sec. 8),
so the caps must be in ``os.environ`` before ``numpy``/``onnxruntime``/``cv2``
are first imported.

Because every other module in the service lives under ``app.``, importing
anything from this package runs this file first, which is exactly the hook we
need.  To keep that guarantee real this module imports **stdlib only** — adding
a numeric import here would defeat its own purpose.

Entrypoints that must run before ``app`` is importable (``tools/bench_ocr_cpu.py``,
Celery ``worker_process_init``) can call :func:`apply_thread_caps` directly in
their own preamble.

Environment wins: a value already present in ``os.environ`` is never overwritten,
so an operator sizing a 32-core node stays in control.

This module does NOT handle:
  * reading ``.env`` (that is :mod:`app.settings`, which imports *after* these
    defaults are already in ``os.environ`` and therefore observes them);
  * per-session ONNX Runtime options (``intra_op_num_threads`` is passed
    explicitly by the local provider — the env var is only the outer fence);
  * ``cv2.setNumThreads`` (a runtime call, not an env var — owned by WS-E);
  * logging, settings validation, or any pipeline behaviour;
  * *forcing* a late importer to comply — if a numeric module was already
    imported before ``app``, the caps for that module are already lost and all
    we can do is warn loudly (see :func:`late_numeric_imports`).
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Final

__version__: Final[str] = "0.1.0"

#: Default value applied to every cap in :data:`THREAD_CAP_VARS` when unset.
#: 2 is the BUILD-BRIEF sec. 6 value; ``CELERY_CONCURRENCY x intra_op <= cores``.
THREAD_CAP_DEFAULT: Final[str] = "2"

#: Environment variables capped on import.  The first three are named in
#: BUILD-BRIEF sec. 6 / ``.env.example``; ``OPENBLAS_``/``NUMEXPR_`` are added
#: because NumPy wheels ship OpenBLAS and pandas/numexpr read their own knob —
#: leaving them uncapped reintroduces the oversubscription this exists to stop.
THREAD_CAP_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ORT_INTRA_OP_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Modules whose thread pools are sized at import time.  If any of these is
#: already in ``sys.modules`` when this package is first imported, the cap for
#: it arrived too late.
NUMERIC_MODULES: Final[tuple[str, ...]] = (
    "numpy",
    "onnxruntime",
    "cv2",
    "ctranslate2",
    "torch",
    "scipy",
    "sklearn",
)


class ThreadCapWarning(RuntimeWarning):
    """A numeric library was imported before ``app``, so its cap arrived late."""


def apply_thread_caps(default: str = THREAD_CAP_DEFAULT) -> dict[str, str]:
    """Set every cap in :data:`THREAD_CAP_VARS` to ``default`` **if unset**.

    Returns the mapping of variables this call actually set, so an entrypoint
    can log precisely what it changed.  Variables already present in the
    environment are left untouched and are absent from the result.
    """
    applied: dict[str, str] = {}
    for name in THREAD_CAP_VARS:
        if os.environ.get(name) is None:
            os.environ[name] = default
            applied[name] = default
    return applied


def late_numeric_imports() -> tuple[str, ...]:
    """Return the :data:`NUMERIC_MODULES` already imported in this process."""
    return tuple(name for name in NUMERIC_MODULES if name in sys.modules)


#: Variables this import actually set (empty when the operator set them all).
THREAD_CAPS_APPLIED: Final[dict[str, str]] = apply_thread_caps()

_LATE: Final[tuple[str, ...]] = late_numeric_imports()
if _LATE:
    # Deliberately a warning, not an exception: a tool that legitimately sets
    # its own caps and imports numpy first (tools/bench_ocr_cpu.py does exactly
    # that) must still be able to import `app`. Raising here would make the
    # correct pattern unusable; the warning names the offenders so a genuine
    # ordering bug is visible in test output and worker logs.
    warnings.warn(
        "app imported after numeric module(s) "
        f"{', '.join(_LATE)}: thread caps {', '.join(THREAD_CAP_VARS)} may not "
        "have been honoured by them. Set the caps before importing them "
        "(app.apply_thread_caps()) — see PLAN.md invariant 5.",
        ThreadCapWarning,
        stacklevel=2,
    )

__all__ = [
    "NUMERIC_MODULES",
    "THREAD_CAPS_APPLIED",
    "THREAD_CAP_DEFAULT",
    "THREAD_CAP_VARS",
    "ThreadCapWarning",
    "__version__",
    "apply_thread_caps",
    "late_numeric_imports",
]
