"""Engines, sessions and the dual-mode wiring — PLAN.md D2.

One ``DATABASE_URL`` selects the backend and nothing else changes: SQLite by
default (a file under ``data/runtime/``, created on demand), Postgres when the
environment says so.  The differences the two backends *do* have are absorbed
here rather than leaking into the models:

* **Relative SQLite paths** are resolved against the repository root by
  :attr:`~app.settings.Settings.database_url`, so the API and a worker started
  from different shells share one file rather than silently creating two.
* **Foreign keys** are off by default in SQLite.  A ``PRAGMA foreign_keys=ON``
  runs on every connection, otherwise ``results.job_id`` would be decoration.
* **Concurrency**: SQLite gets WAL and a busy timeout, because the in-process
  webhook worker writes from a second thread while a request is reading.
  ``check_same_thread=False`` is required for that and is safe here — sessions
  are per-unit-of-work and never shared between threads.
* **Pooling**: an in-memory SQLite database needs ``StaticPool`` or every
  connection gets its own empty database.  That is the shape every test uses.

This module does NOT handle:
  * schema creation in a deployment.  :func:`create_all` exists for the SQLite
    default and for tests; Postgres is migrated with Alembic
    (``alembic -c alembic/alembic.ini upgrade head``) and
    :func:`create_all` is not a substitute for it;
  * transactions beyond the ``session_scope`` unit of work;
  * any query — :mod:`app.storage.repository` owns those;
  * append-only policy, which is declared on the models; this module only
    installs the session-level half of the guard.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.contracts.errors import StorageError
from app.settings import Settings, get_settings
from app.storage.models import Base, assert_no_append_only_mutation

__all__ = [
    "build_engine",
    "build_session_factory",
    "create_all",
    "dispose_engine",
    "ensure_sqlite_parent",
    "session_scope",
]

_LOG = logging.getLogger(__name__)


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _is_memory(url: str) -> bool:
    return ":memory:" in url or "mode=memory" in url


def build_engine(settings: Settings | None = None, *, url: str | None = None) -> Engine:
    """Create the engine for ``DATABASE_URL`` (or an explicit ``url``).

    The parent directory of a SQLite file is created here: ``settings.py``
    deliberately does not create the directories it names, and the component
    that writes there is this one.

    Raises:
        StorageError: the URL cannot be turned into an engine.
    """
    active = settings if settings is not None else get_settings()
    dsn = url if url is not None else active.database_url
    kwargs: dict[str, Any] = {"future": True, "echo": False}

    if _is_sqlite(dsn):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30.0}
        if _is_memory(dsn):
            # Without StaticPool every connection opens a *different* in-memory
            # database, so a table created on one is invisible to the next.
            kwargs["poolclass"] = StaticPool
        else:
            ensure_sqlite_parent(dsn)

    try:
        engine = create_engine(dsn, **kwargs)
    except (SQLAlchemyError, ArgumentError, ValueError, ModuleNotFoundError) as failure:
        raise StorageError(
            f"cannot open a database at {_redacted(dsn)}: {failure}",
            context={"url": _redacted(dsn)},
        ) from failure

    if _is_sqlite(dsn):
        _install_sqlite_pragmas(engine, memory=_is_memory(dsn))
    _LOG.debug("database engine ready | %s", _redacted(dsn))
    return engine


def ensure_sqlite_parent(url: str) -> None:
    """Create the directory a SQLite file lives in.  No-op for other drivers.

    Exported because Alembic reaches the database without going through
    :func:`build_engine`, and ``sqlite3`` reports a missing directory as the
    thoroughly unhelpful "unable to open database file".
    """
    if not _is_sqlite(url) or _is_memory(url):
        return
    _sqlite_path(url).parent.mkdir(parents=True, exist_ok=True)


def _sqlite_path(url: str) -> Path:
    """Filesystem path inside a ``sqlite+driver:///path`` URL."""
    _, _, remainder = url.partition(":///")
    return Path(remainder.split("?", 1)[0])


def _redacted(url: str) -> str:
    """The URL with any password removed — this string reaches the log."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _credentials, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def _install_sqlite_pragmas(engine: Engine, *, memory: bool) -> None:
    """Foreign keys on, WAL for the file backend, and a real busy timeout."""

    @event.listens_for(engine, "connect")
    def _pragmas(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            if not memory:
                # WAL is meaningless (and refused) for an in-memory database.
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """A session factory with the session-level append-only guard installed.

    ``expire_on_commit=False`` so a caller can still read the object it just
    committed — the API returns a job record immediately after writing it, and
    a lazy refresh on a closed session is the classic way that turns into a
    ``DetachedInstanceError`` in production and never in a test.
    """
    factory = sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False, future=True
    )

    @event.listens_for(factory, "before_flush")
    def _guard(session: Session, _context: Any, _instances: Any) -> None:
        assert_no_append_only_mutation(session)

    return factory


def create_all(engine: Engine) -> None:
    """Create every table.  For SQLite and tests, **not** for a deployment.

    A deployment runs Alembic: ``create_all`` produces whatever the models say
    today with no record of how it got there, which is exactly what a migration
    history exists to prevent.  ``alembic/versions/0001_initial.py`` is asserted
    to agree with these models.
    """
    Base.metadata.create_all(engine)


def dispose_engine(engine: Engine) -> None:
    """Close every pooled connection.  Safe to call twice."""
    engine.dispose()


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any exception.

    Deliberately re-raises rather than translating: an
    :class:`~app.contracts.errors.AppendOnlyViolation` raised by the guard is
    the caller's bug and must arrive as itself, and a genuine database failure
    is wrapped at the repository boundary where the operation has a name.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
