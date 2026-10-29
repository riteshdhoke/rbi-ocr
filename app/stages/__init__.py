"""Pipeline stages — one module per numbered stage of the PRD flow.

The package root deliberately imports **nothing**.  Every stage drags in a
different heavy dependency (``pypdfium2`` and Pillow for Stage 1, OpenCV for
Stage 1.5, ONNX Runtime for Stage 4), and an eager re-export here would make
``import app.stages.s5_validators`` — which needs none of them — pay for all of
them.  Import the stage you want by its module path.

The one thing that *is* here is :func:`configure_cv2_threads`.  PLAN.md
invariant 5 fences the numeric runtimes with environment variables set in
:mod:`app` before import; OpenCV is the exception, because ``cv2`` sizes its
pool at *runtime* through ``cv2.setNumThreads`` and ignores ``OMP_NUM_THREADS``
on the wheels we ship.  ``app/__init__.py`` names that gap explicitly and
assigns it to WS-E; this is where it is closed.  Every stage that imports
``cv2`` calls it at import time.

This package does NOT handle:
  * orchestration or stage ordering — that is ``app/pipeline.py`` (WS-F);
  * persistence of intermediate artefacts;
  * provider construction or the residency gate (``app.providers.factory``);
  * defining the contracts the stages exchange (``app.contracts``).
"""

from __future__ import annotations

import logging
from typing import Final

__all__ = ["configure_cv2_threads", "cv2_thread_cap"]

_LOGGER: Final = logging.getLogger(__name__)

#: Set once, by the first stage that asks.  ``None`` until then.
_CV2_THREADS: int | None = None


def configure_cv2_threads(threads: int | None = None) -> int:
    """Cap OpenCV's internal thread pool and return the value applied.

    Idempotent: the first call wins, later calls report the value already in
    force.  ``threads=None`` reads ``ORT_INTRA_OP_THREADS`` from settings, which
    is the same fence the ONNX sessions run behind — the point is that
    ``CELERY_CONCURRENCY x intra_op <= physical cores`` holds for *every*
    runtime, not just the one that was measured.

    Raises:
        ValueError: when ``threads`` is not positive.
    """
    global _CV2_THREADS
    if _CV2_THREADS is not None:
        return _CV2_THREADS
    if threads is None:
        from app.settings import get_settings

        threads = get_settings().ORT_INTRA_OP_THREADS
    if threads < 1:
        raise ValueError(f"cv2 thread cap must be >= 1, got {threads}")

    import cv2

    cv2.setNumThreads(threads)
    _CV2_THREADS = threads
    _LOGGER.debug("cv2.setNumThreads(%d)", threads)
    return threads


def cv2_thread_cap() -> int | None:
    """The cap :func:`configure_cv2_threads` applied, or ``None`` if unset."""
    return _CV2_THREADS
