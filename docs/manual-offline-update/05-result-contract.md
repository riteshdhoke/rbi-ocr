# 05 — Edit `app/contracts/envelope.py`

## A. Sources

Replace the `FieldSource = Literal[...]` declaration with:

```python
FieldSource = Literal[
    "regex_validator",
    "ocr_label",
    "lookup_table",
    "slm",
    "classifier",
    "human",
]
```

## B. `DocumentResult`

Keep the existing `fields` declaration. Immediately after it add:

```python
    extracted_fields: dict[str, list[Field]] = _Field(default_factory=dict)
    other_fields: dict[str, Field] = _Field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
```

Replace the entire decorated `_field_provenance_rules` method with:

```python
    @model_validator(mode="after")
    def _field_provenance_rules(self) -> Self:
        for name in self.fields:
            if not name:
                raise ValueError("field names must be non-empty")
        check_distribution(
            self.language_distribution, "DocumentResult.language_distribution"
        )
        if (
            self.primary_language is not None
            and self.language_distribution
            and self.primary_language not in self.language_distribution
        ):
            raise ValueError(
                f"primary_language {self.primary_language!r} absent from "
                "language_distribution"
            )
        return self
```

The removed condition used to prohibit SLM provenance on identifier fields.
Grounded model discoveries are now allowed; `validator_passed` is not invented.
Remove the now-unused `DETERMINISTIC_FIELDS` import from this file only.

## C. `ResultEnvelope`

Immediately after `job_id: str | None = None`, add:

```python
    requested_fields: tuple[str, ...] = ()
```

Replace the entire decorated `_envelope_consistency` method with:

```python
    @model_validator(mode="after")
    def _envelope_consistency(self) -> Self:
        from app.field_selection import parse_requested_fields, partition_fields

        self.requested_fields = parse_requested_fields(self.requested_fields)
        if self.document is not None:
            extracted, other, missing = partition_fields(
                self.document.fields, self.requested_fields
            )
            self.document.extracted_fields = extracted
            self.document.other_fields = other
            self.document.missing_fields = missing
        seen: set[int] = set()
        previous = 0
        for page in self.pages:
            if page.page_no in seen:
                raise ValueError(f"duplicate page_no {page.page_no}")
            if page.page_no < previous:
                raise ValueError(
                    "pages must be ordered by ascending page_no, got "
                    f"{page.page_no} after {previous}"
                )
            seen.add(page.page_no)
            previous = page.page_no

        bounced = self.bounce_decision.decision is QualityVerdict.BOUNCE
        if self.document is None and not bounced:
            raise ValueError(
                "document may be None only when bounce_decision.decision is "
                "'bounce' — a missing result must say why, never arrive empty "
                "(BUILD-BRIEF sec. 0.5)"
            )
        if bounced and self.document is not None:
            raise ValueError("a bounced document must not carry extracted content")
        if any(page.escalated for page in self.pages) and not self.escalation_flag:
            raise ValueError(
                "escalation_flag must be true when any page escalated; shadow "
                "sampling uses PageResult.shadow_sampled instead"
            )
        if self.document is not None and self.document.page_count != len(self.pages):
            raise ValueError(
                f"document.page_count={self.document.page_count} contradicts "
                f"{len(self.pages)} page records"
            )
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return self
```

This derives both groups from the complete `document.fields` map on construction
and retrieval. Existing clients retain the complete map; requested names map to
arrays, with `[]` meaning no matching value.

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/contracts/envelope.py app/stages/s5_discover.py
```

Milestone: if `verify_update.py` is available, run it with `--stage core` as explained in step 12.
