# 09 — Edit `app/pipeline.py`

This is the longest integration step. Do not replace the whole file or remove any
OCR, ingestion, language, quality, residency or routing stages.

## A. `run_document` signature

Replace its signature only; leave the docstring and body below it:

```python
def run_document(
    source: Path | str | bytes,
    *,
    provider: ModelProvider,
    settings: Settings | None = None,
    job_id: str | None = None,
    filename: str | None = None,
    synthetic_declared: bool | None = None,
    on_envelope: Callable[[ResultEnvelope], None] | None = None,
    ifsc_master: IFSCMaster | None = None,
    scanner: Scanner | None = None,
    strict_rejections: bool = False,
    requested_fields: Sequence[str] = (),
    starting_result_version: int = 1,
) -> PipelineResult:
```

## B. Initialization and publication

Inside `run_document`, replace from the existing line
`active = settings if settings is not None else get_settings()` through the end
of the nested `publish` function, stopping before `with timings.stage("s0_read")`.
Use this complete replacement block:

```python
    from app.field_selection import parse_requested_fields

    requested_fields = parse_requested_fields(requested_fields)
    if starting_result_version < 1:
        raise ValueError("starting_result_version must be positive")
    active = settings if settings is not None else get_settings()
    if requested_fields:
        active = active.model_copy(update={"EXTRACT_ALL_FIELDS": True})
    started_at = utc_now()
    wall_start = time.perf_counter()
    timings = _Timings()
    published: list[ResultEnvelope] = []

    def publish(envelope: ResultEnvelope) -> None:
        # Re-extractions append to the same job without replacing old evidence.
        envelope = ResultEnvelope.model_validate(
            {
                **envelope.model_dump(),
                "result_version": envelope.result_version + starting_result_version - 1,
                "requested_fields": requested_fields,
            }
        )
        published.append(envelope)
        _VERSIONS.labels(version=str(envelope.result_version)).inc()
        if on_envelope is not None:
            on_envelope(envelope)
```

Do not add the version offset anywhere else. This nested publisher applies it
once, including to bounces, so later runs append versions instead of overwriting.

## C. Four call sites inside `run_document`

There are two `_extract_fields(...)` calls and two `_assemble(...)` calls.
Inside **each of those four calls**, add this keyword argument once:

```python
        requested_fields=requested_fields,
```

Use the same indentation as the neighboring keyword arguments. The second pair
is inside `if pending:` and therefore has four extra leading spaces. In the
second `_extract_fields` call, keep the existing `only_pages=...` argument.

Do not add this keyword to `_run_validators`, `_route_pages` or `_run_tier2`.

## D. `_extract_fields` helper

Replace the entire `_extract_fields` function with:

```python
def _extract_fields(
    outcomes: Sequence[PageOutcome],
    *,
    provider: ModelProvider,
    settings: Settings,
    document_id: str,
    report: ValidatorReport,
    budget: JobBudget,
    timings: _Timings,
    warnings: list[str],
    only_pages: set[int] | None = None,
    requested_fields: Sequence[str] = (),
) -> None:
    """Run the extractive SLM over pages with text, absorbing provider failure.

    A provider error here costs the free-text fields and nothing else:
    identifiers, amounts and dates are the validators' by invariant 7, and they
    are already extracted.  The failure is recorded as a machine-readable
    warning so the rate is measurable rather than invisible.

    ``only_pages`` restricts the re-run after Tier 2 to the pages whose text
    actually changed; the others keep the raw fields the fast path produced.
    """
    candidates = [
        outcome
        for outcome in outcomes
        if not outcome.bounced
        and outcome.page.lines
        and (only_pages is None or outcome.page_no in only_pages)
    ]
    if not candidates:
        return

    by_page = {outcome.page_no: outcome for outcome in candidates}
    contexts = build_extraction_contexts(
        [outcome.page for outcome in candidates],
        target_fields=requested_fields or None,
        document_id=document_id,
        page_classes={
            outcome.page_no: (
                outcome.page_class.label
                if outcome.page_class is not None
                else PageClassLabel.UNKNOWN
            )
            for outcome in candidates
        },
        primary_scripts={
            outcome.page_no: _page_script(outcome) for outcome in candidates
        },
        primary_languages={
            outcome.page_no: _page_language(outcome) for outcome in candidates
        },
        masked_spans={
            outcome.page_no: (
                []
                if settings.EXTRACT_ALL_FIELDS
                and settings.PROVIDER_PROFILE.value != "laptop"
                else report.masked_spans(outcome.page_no)
            )
            for outcome in candidates
        },
        budget=budget,
        settings=settings,
    )
    if not contexts:
        return
    try:
        with timings.stage("s5_slm"):
            results = extract_pages(provider, contexts)
    except ProviderError as failure:
        warnings.append(f"{SLM_UNAVAILABLE_PREFIX}:{failure.code}")
        _LOG.warning(
            "SLM unavailable for document=%s (%s); deterministic fields stand",
            document_id,
            failure.code,
        )
        return
    for context, fields in zip(contexts, results, strict=True):
        by_page[context.page_no].raw_fields = dict(fields)
```

The local profiles can see all validator spans; laptop mode retains masking.
Requested names reach the model context and open discovery remains enabled.

## E. `_assemble` helper

Replace the entire `_assemble` function with:

```python
def _assemble(
    outcomes: Sequence[PageOutcome],
    *,
    document_id: str,
    job_id: str | None,
    settings: Settings,
    provider: ModelProvider,
    report: ValidatorReport,
    result_version: int,
    is_final: bool,
    started_at: datetime,
    timings: _Timings,
    budget: JobBudget,
    warnings: Sequence[str],
    requested_fields: Sequence[str] = (),
) -> ResultEnvelope:
    """Build one published envelope from the current state of every page."""
    finished_at = utc_now()
    document_summary = derive_language_summary(
        [line for outcome in outcomes for line in outcome.page.lines],
        code_mix_threshold=settings.CODE_MIX_THRESHOLD,
    )
    translator = _translator(
        provider,
        report=report,
        src_language=document_summary.primary_language,
    )
    with timings.stage("assemble"):
        envelope = assemble_envelope(
            document_id=document_id,
            provider_profile=settings.PROVIDER_PROFILE.value,
            pages=[outcome.to_assembly() for outcome in outcomes],
            validator_report=report,
            translator=translator,
            job_id=job_id,
            result_version=result_version,
            is_final=is_final,
            processing=ProcessingStats(
                started_at=started_at,
                finished_at=finished_at,
                tokens_used=budget.used,
                stage_timings_ms=timings.snapshot(),
            ),
            warnings=tuple(warnings),
            requested_fields=requested_fields,
            settings=settings,
        )
    return envelope
```

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/pipeline.py
```

Milestone: run `verify_update.py --stage pipeline`. No real model or internet is needed for that check.
