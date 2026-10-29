# 10 — Edit `app/api/routes.py`

Inside `build_router`, replace the **entire decorated** `submit_document` function
with the block below, starting at `@router.post(...)` and ending before the
`@router.get("/v1/jobs/{job_id}", ...)` endpoint. Keep that GET endpoint and all
other helpers unchanged.

The replacement block is indented four spaces because it belongs inside
`build_router`; do not remove that indentation.

```python
    @router.post(
        "/v1/documents",
        response_model=SubmissionResponse,
        summary="Submit one document and get its result envelope",
    )
    async def submit_document(
        request: Request,
        file: Annotated[UploadFile, File(description="The attachment.")],
        synthetic: Annotated[
            bool,
            Form(
                description=(
                    "Declare the payload synthetic. Required under the laptop "
                    "profile (PLAN.md invariant 6); ignored on cpu_node."
                )
            ),
        ] = False,
        sender: Annotated[str | None, Form()] = None,
        subject: Annotated[str | None, Form()] = None,
        requested_fields: Annotated[
            str,
            Form(
                max_length=4096,
                description="Comma-separated field names to extract; additional discoveries are returned as other_fields.",
            ),
        ] = "",
        store: Annotated[JobStore, Depends(_store)] = None,  # type: ignore[assignment]
        settings: Annotated[Settings, Depends(_settings)] = None,  # type: ignore[assignment]
    ) -> SubmissionResponse:
        """Ingest, process and store one document.

        Raises:
            PipelineError: refused upload (too large), residency violation,
                malware, or a provider failure the pipeline did not absorb.
                Each is mapped to its own status by the app's handlers.
        """
        from fastapi.exceptions import RequestValidationError

        from app.field_selection import parse_requested_fields

        try:
            wanted = parse_requested_fields(requested_fields)
        except ValueError as failure:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("body", "requested_fields"),
                        "msg": str(failure),
                        "input": requested_fields,
                    }
                ]
            ) from failure
        data = await _read_upload(file, settings)
        state = _state(request)

        from app.stages.s1_ingest import idempotency_key

        document_id = idempotency_key(data)
        record = store.create_or_get_job(
            document_id=document_id,
            filename=file.filename,
            byte_size=len(data),
            content_type=file.content_type,
            sender=sender,
            subject=subject,
            provider_profile=settings.PROVIDER_PROFILE.value,
        )
        previous = store.latest_envelope(record.job.id) if not record.is_new else None
        if not record.is_new and record.job.status in {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        }:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=409,
                detail="An extraction for this document is already running. Retry after it completes.",
            )
        if (
            record.job.status != JobStatus.FAILED.value
            and previous is not None
            and previous.requested_fields == wanted
        ):
            _LOG.info(
                "idempotent resubmission | job=%s version=%d",
                record.job.id,
                record.job.latest_result_version,
            )
            return _submission(record.job, previous, False)

        job_id = record.job.id
        store.set_status(job_id, JobStatus.RUNNING)
        published: list[ResultEnvelope] = []

        def on_envelope(envelope: ResultEnvelope) -> None:
            store.record_result(job_id, envelope)
            published.append(envelope)
            state.dispatcher.publish(job_id, envelope)

        try:
            run_document(
                data,
                provider=state.provider,
                settings=settings,
                job_id=job_id,
                filename=file.filename,
                synthetic_declared=synthetic,
                requested_fields=wanted,
                starting_result_version=record.job.latest_result_version + 1,
                on_envelope=on_envelope,
            )
        except (ProviderError, DocumentRejected) as failure:
            store.set_status(
                job_id,
                JobStatus.FAILED,
                error_code=failure.code,
                error_message=failure.message,
            )
            raise
        except PipelineError as failure:
            store.set_status(
                job_id,
                JobStatus.FAILED,
                error_code=failure.code,
                error_message=failure.message,
            )
            raise

        job = store.get_job(job_id)
        return _submission(job, published[-1] if published else None, record.is_new)
```

Verification points:

- `requested_fields` is a multipart form string, not a query parameter.
- CSV parsing happens before creating a job.
- Equivalent lists reuse the latest result; changed lists run extraction again.
- `starting_result_version` comes from the last stored version plus one.
- `is_new` describes job creation, not whether a new extraction ran.
- No database column or migration is added; selections are stored in envelopes.

`app/api/schemas.py` has a documentation-only update in the reference copy. It
requires no runtime edit for this feature.

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/api/routes.py
```

Milestone: run `verify_update.py --stage api`.
