# 08 — Edit `app/stages/assemble.py`

Add this import alongside the stage imports:

```python
from app.stages.s5_discover import add_discovered, discover_fields
```

Replace the entire `assemble_envelope` function with the following. Keep all other
functions in this module. Copying the complete function avoids missing the
requested-fields argument, repeated-value handling, or bounce exclusion.

```python
def assemble_envelope(
    *,
    document_id: str,
    provider_profile: str,
    pages: Sequence[PageAssembly],
    validator_report: ValidatorReport | None = None,
    document_class: PageClass | None = None,
    bounce_decision: BounceDecision | None = None,
    translator: Callable[[dict[str, Field]], Mapping[str, Field]] | None = None,
    job_id: str | None = None,
    result_version: int = 1,
    is_final: bool = True,
    processing: ProcessingStats | None = None,
    warnings: Sequence[str] = (),
    supporting_doc_tags: Sequence[str] = (),
    summary_separator: str = "\n",
    settings: Settings | None = None,
    requested_fields: Sequence[str] = (),
) -> ResultEnvelope:
    """Build the published :class:`~app.contracts.envelope.ResultEnvelope`.

    The order of operations is the contract of this function:

    1. per-page records, with language summaries derived from line labels;
    2. **the grounding gate**, page by page, over ``PageAssembly.raw_fields``;
    3. deterministic validator fields merged with the grounded SLM fields;
    4. the optional ``translator`` applied to the merged map (``value``
       untouched, ``value_en`` advisory);
    5. ``complaint_summary`` concatenated from the surviving line selections;
    6. document-level language summary over every page's lines;
    7. counters recomputed from the page records.

    Args:
        bounce_decision: passed straight through.  When it is a bounce the
            envelope carries no ``document`` at all — a refused document must
            say why, not arrive empty (BUILD-BRIEF sec. 0.5).
        translator: applied to the finished field map; normally
            ``functools.partial(translate_fields, provider, src_language=...,
            mt_engine=...)``.  Kept as a callable so this module imports no
            provider and needs no MT engine to be tested.
        processing: caller-supplied timings.  Page counters are **always**
            recomputed here so they cannot contradict the page records.

    Raises:
        ValueError: when ``pages`` contains duplicate page numbers, or when a
            page record contradicts itself (raised by the contract).
    """
    active = settings if settings is not None else get_settings()
    threshold = active.CODE_MIX_THRESHOLD

    ordered = sorted(pages, key=lambda item: item.page_no)
    numbers = [item.page_no for item in ordered]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"duplicate page numbers in assembly input: {numbers}")

    page_results = tuple(
        build_page_result(item, code_mix_threshold=threshold) for item in ordered
    )

    gate_results: list[SLMPageResult] = [
        apply_grounding_gate(item.raw_fields, item.page, settings=active)
        for item in ordered
    ]
    rejections: tuple[RejectedField, ...] = tuple(
        rejected for result in gate_results for rejected in result.rejected
    )

    decision = bounce_decision or BounceDecision.accepted()
    all_warnings = [*warnings, *(_warning_for(item) for item in rejections)]
    if rejections:
        # Counts only: the rejected strings are unverified model output and this
        # line is INFO-level (PLAN.md: never log raw OCR text or model values).
        _LOG.info(
            "grounding gate rejected %d value(s) across %d page(s)",
            len(rejections),
            len({item.page_no for item in rejections}),
        )

    document: DocumentResult | None = None
    fields: dict[str, Field] = {}
    if decision.decision is not QualityVerdict.BOUNCE:
        validator_fields = (
            validator_report.fields() if validator_report is not None else {}
        )
        slm_fields = merge_field_candidates(*(result.fields for result in gate_results))
        fields = merge_field_candidates(validator_fields, slm_fields)
        if active.EXTRACT_ALL_FIELDS:
            # Known validators remain preferred for their canonical keys; keep
            # other occurrences instead of collapsing them to one winner.
            fields = dict(validator_fields)
            if validator_report is not None:
                for hit in validator_report.hits:
                    add_discovered(fields, hit.field_name, hit.to_field())
            for result in gate_results:
                for name, candidate in result.fields.items():
                    add_discovered(fields, name, candidate)
            fields = discover_fields(
                [
                    item.page
                    for item in ordered
                    if item.quality_verdict is not QualityVerdict.BOUNCE
                ],
                fields,
                requested_fields=requested_fields,
            )
        if translator is not None:
            fields = dict(translator(fields))

        pages_by_no = {item.page_no: item.page for item in ordered}
        summary = build_complaint_summary(
            [
                (result.page_no, result.summary_line_indices)
                for result in gate_results
                if result.summary_line_indices
            ],
            pages_by_no,
            separator=summary_separator,
        )
        document_summary = derive_language_summary(
            [line for item in ordered for line in item.page.lines],
            code_mix_threshold=threshold,
        )
        label, label_confidence = _document_class(document_class, ordered)
        document = DocumentResult(
            document_class=label,
            document_class_confidence=label_confidence,
            fields=fields,
            complaint_summary=summary,
            supporting_doc_tags=tuple(supporting_doc_tags),
            page_count=len(page_results),
            primary_script=document_summary.primary_script,
            primary_language=document_summary.primary_language,
            language_distribution=document_summary.language_distribution,
            is_code_mixed=document_summary.is_code_mixed,
        )

    escalation_flag = (
        any(result.escalated for result in page_results)
        or any(field.needs_review for field in fields.values())
        or bool(rejections)
    )

    stats = _processing_stats(processing, page_results)

    return ResultEnvelope(
        result_version=result_version,
        is_final=is_final,
        document_id=document_id,
        job_id=job_id,
        provider_profile=provider_profile,
        requested_fields=tuple(requested_fields),
        bounce_decision=decision,
        escalation_flag=escalation_flag,
        pages=page_results,
        document=document,
        processing=stats,
        warnings=tuple(all_warnings),
    )
```

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/stages/assemble.py
```

Milestone: run `verify_update.py --stage extraction` after this step.
