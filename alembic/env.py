"""Alembic environment — one migration history, both backends.

The URL comes from :class:`app.settings.Settings`, never from ``alembic.ini``,
so the migration and the running service cannot disagree about which database
they mean and no password is ever written into a tracked file.  ``-x url=...``
overrides it for a one-off.

Two dialect differences are absorbed here:

* ``render_as_batch=True`` — SQLite cannot ``ALTER TABLE ... DROP COLUMN`` or
  alter a constraint, so Alembic has to rebuild the table.  Without it the
  first migration that changes a column runs on Postgres and fails on the
  default profile, which is the profile the tests use.
* ``compare_type=True`` — autogenerate otherwise misses a ``String(64)`` grown
  to ``String(128)``, which is exactly the sort of change that then truncates
  silently on Postgres.

This module does NOT handle: creating the database itself, seeding, or the
SQLite convenience path (``app.storage.database.create_all``, which is for
tests and is not a substitute for a migration history).
"""

from __future__ import annotations

# ruff: noqa: I001 - the sys.path fix must run between the two import blocks
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.settings import get_settings
from app.storage.database import ensure_sqlite_parent
from app.storage.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """``-x url=...``, else ``alembic.ini``, else ``Settings.database_url``."""
    override = context.get_x_argument(as_dictionary=True).get("url")
    if override:
        return str(override)
    configured = config.get_main_option("sqlalchemy.url", default="")
    if configured:
        return configured
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL without a connection (``--sql``), for a reviewed deploy."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run against a live database."""
    section = config.get_section(config.config_ini_section, {})
    url = _database_url()
    ensure_sqlite_parent(url)
    section["sqlalchemy.url"] = url
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
