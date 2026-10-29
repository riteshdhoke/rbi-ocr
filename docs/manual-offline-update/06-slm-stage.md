# 06 — Edit `app/stages/s5_slm.py`

## A. Remove the response field-name enum

In `EXTRACTION_RESPONSE_SCHEMA`, find the `name` property containing
`"enum": list(DEFAULT_TARGET_FIELDS)`. Replace that property with:

```python
                    "name": {"type": "string", "minLength": 1},
```

Do not remove the remaining JSON-shape validation or confidence limits.

## B. Context construction

Replace the entire `build_extraction_context` function with:

```python
def build_extraction_context(
    page: OCRPage,
    *,
    document_id: str,
    page_class: PageClassLabel,
    primary_script: str,
    primary_language: str | None = None,
    target_fields: Sequence[str] | None = None,
    masked_spans: Sequence[TextSpan] = (),
    budget: BudgetSource | None = None,
    settings: Settings | None = None,
) -> ExtractionContext:
    """Build one page's :class:`~app.contracts.ocr.ExtractionContext`.

    ``masked_spans`` should be
    :meth:`~app.stages.s5_validators.ValidatorReport.masked_spans` for this page.
    Passing them is not optional in spirit: they are the mechanism by which
    invariant 7's "identifiers never reach the SLM" is true of the *bytes*, not
    only of the field list.

    Raises:
        TokenBudgetExceeded: when ``budget`` has no headroom left. Refusing here
            rather than sending a request that will be rejected keeps the cost
            of a breach at zero.
        ValueError: for an empty ``target_fields`` — an extraction request with
            nothing to extract is a caller bug.
    """
    active = settings if settings is not None else get_settings()
    fields = tuple(target_fields) if target_fields is not None else DEFAULT_TARGET_FIELDS
    if not fields:
        raise ValueError("target_fields must name at least one field to extract")
    max_tokens = (active.SLM_MAX_TOKENS if active.EXTRACT_ALL_FIELDS
                  else context_max_tokens(fields, settings=active))
    remaining: int | None = None
    if budget is not None:
        remaining = int(budget.remaining)
        if remaining <= 0:
            raise TokenBudgetExceeded(
                "no token headroom left for SLM extraction on page "
                f"{page.page_no}",
                budget=None,
                used=None,
                context={"page_no": page.page_no, "document_id": document_id},
            )
        max_tokens = max(1, min(max_tokens, remaining))
    return ExtractionContext(
        document_id=document_id,
        page_no=page.page_no,
        lines=list(page.lines),
        page_class=page_class,
        primary_script=primary_script,
        primary_language=primary_language,
        discover_fields=active.EXTRACT_ALL_FIELDS,
        target_fields=list(fields),
        masked_spans=list(masked_spans),
        max_tokens=max_tokens,
        token_budget_remaining=remaining,
    )
```

## C. Grounding gate

Inside `apply_grounding_gate`, find:

```python
        if name not in SLM_ELIGIBLE_FIELDS:
```

Replace that line only with:

```python
        if not active.EXTRACT_ALL_FIELDS and name not in SLM_ELIGIBLE_FIELDS:
```

Keep the following rejection branch and all OCR grounding code. Open names are
allowed; fabricated values are still refused.

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/stages/s5_slm.py
```
