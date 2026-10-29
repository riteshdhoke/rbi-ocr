"""Initial schema: jobs, results, audit_events, webhook_deliveries.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-07

Portable across SQLite and Postgres by construction (PLAN.md D2):

* ``sa.JSON`` renders as native ``JSON`` on Postgres and serialised ``TEXT`` on
  SQLite, so one column definition serves both;
* ``sa.DateTime(timezone=True)`` is honoured by Postgres and ignored by SQLite,
  which is why every timestamp is computed in Python — see
  ``app/storage/models.py::utc_now``;
* integer primary keys autoincrement on both without a sequence declaration.

**Append-only is not expressed here.**  ``results`` and ``audit_events`` are
enforced in the ORM (``app/storage/models.py``) rather than by a grant or a
trigger, because SQLite has no per-table privileges at all and a rule that only
exists on Postgres would not exist under the default profile — the one the
tests, the eval harness and the laptop demo all run.  A deployment that wants
defence in depth should additionally ``REVOKE UPDATE, DELETE`` on both tables
from the application role; that is a deployment decision and is documented in
the runbook rather than baked into a migration that must also run on SQLite.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=True),
        sa.Column("provider_profile", sa.String(length=16), nullable=True),
        sa.Column("sender", sa.String(length=320), nullable=True),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("normalized_subject", sa.Text(), nullable=True),
        sa.Column("received_day", sa.String(length=10), nullable=True),
        sa.Column("soft_duplicate_key", sa.String(length=80), nullable=True),
        sa.Column("soft_duplicate_of", sa.String(length=64), nullable=True),
        sa.Column("soft_duplicate_reason", sa.String(length=64), nullable=True),
        sa.Column("latest_result_version", sa.Integer(), nullable=False),
        sa.Column("is_final", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["soft_duplicate_of"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_document_id", "jobs", ["document_id"], unique=True)
    op.create_index(
        "ix_jobs_soft_duplicate_key", "jobs", ["soft_duplicate_key"], unique=False
    )
    op.create_index(
        "ix_jobs_soft_duplicate",
        "jobs",
        ["sender", "normalized_subject", "received_day"],
        unique=False,
    )

    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("result_version", sa.Integer(), nullable=False),
        sa.Column("is_final", sa.Boolean(), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column("document_id", sa.String(length=80), nullable=False),
        sa.Column("envelope", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "result_version", name="uq_results_job_version"
        ),
    )
    op.create_index("ix_results_job_id", "results", ["job_id"], unique=False)
    op.create_index(
        "ix_results_document_id", "results", ["document_id"], unique=False
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=48), nullable=False),
        sa.Column("entity", sa.String(length=128), nullable=True),
        sa.Column("before", sa.JSON(), nullable=True),
        sa.Column("after", sa.JSON(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_job_id", "audit_events", ["job_id"])
    op.create_index("ix_audit_events_ts", "audit_events", ["ts"])
    op.create_index("ix_audit_events_action", "audit_events", ["action"])

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("result_version", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_code", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("signature", sa.String(length=128), nullable=False),
        sa.Column("signed_at", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "result_version", name="uq_webhook_job_version"
        ),
    )
    op.create_index(
        "ix_webhook_deliveries_job_id", "webhook_deliveries", ["job_id"]
    )
    op.create_index(
        "ix_webhook_deliveries_status", "webhook_deliveries", ["status"]
    )


def downgrade() -> None:
    op.drop_table("webhook_deliveries")
    op.drop_table("audit_events")
    op.drop_table("results")
    op.drop_index("ix_jobs_soft_duplicate", table_name="jobs")
    op.drop_index("ix_jobs_soft_duplicate_key", table_name="jobs")
    op.drop_index("ix_jobs_document_id", table_name="jobs")
    op.drop_table("jobs")
