"""HTTP surface — FastAPI application factory, routes, schemas, PII filter.

``from app.api import create_app`` is the whole public API of this package.
``uvicorn app.api:app`` works too: :data:`app` is built on first attribute
access so importing this package does **not** open a database or build a
provider, which is what lets the test suite import it freely.

This package does NOT handle: the pipeline, persistence, webhook transport, or
the review UI (WS-G).
"""

from __future__ import annotations

from typing import Any

from app.api.app import create_app

__all__ = ["app", "create_app"]

_APP: Any = None


def __getattr__(name: str) -> Any:
    """Build the module-level ``app`` lazily, for ``uvicorn app.api:app``.

    Deferred because :func:`create_app` runs the residency gate and constructs
    a provider: doing that at import time would make ``import app.api`` fail on
    a machine whose ``.env`` is mid-edit, and would build a database connection
    for a test that only wanted the type.
    """
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
