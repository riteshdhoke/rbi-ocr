# 04 — Edit `app/contracts/ocr.py`

Keep existing OCR models and the two field-name sets. The sets remain useful for
legacy behavior; deleting them breaks imports and is not how open extraction works.

## A. `ExtractionContext`

Add this field immediately before `target_fields` (four leading spaces):

```python
    discover_fields: bool = False
```

Replace the entire decorated `_only_slm_eligible_fields` method with:

```python
    @model_validator(mode="after")
    def _only_slm_eligible_fields(self) -> Self:
        forbidden = sorted(set(self.target_fields) & DETERMINISTIC_FIELDS)
        if forbidden and not self.discover_fields:
            raise ValueError(
                f"deterministic field(s) {', '.join(forbidden)} must never be "
                "sent to the SLM — validators own them (PLAN.md invariant 7)"
            )
        unknown = sorted(set(self.target_fields) - SLM_ELIGIBLE_FIELDS)
        if unknown and not self.discover_fields:
            raise ValueError(
                f"unknown SLM target field(s) {', '.join(unknown)}; allowed: "
                f"{', '.join(sorted(SLM_ELIGIBLE_FIELDS))}"
            )
        duplicates = sorted(
            {f for f in self.target_fields if self.target_fields.count(f) > 1}
        )
        if duplicates:
            raise ValueError(
                f"duplicate target field(s): {', '.join(duplicates)}"
            )
        known_indices = {line.line_index for line in self.lines}
        for span in self.masked_spans:
            if span.line_index not in known_indices:
                raise ValueError(
                    f"masked span references line_index {span.line_index}, "
                    f"absent from page {self.page_no}"
                )
        return self
```

## B. `RawField`

Immediately after `name: str = Field(min_length=1)`, add:

```python
    discovered: bool = False
```

Replace the entire decorated `_summary_is_extractive` method with:

```python
    @model_validator(mode="after")
    def _summary_is_extractive(self) -> Self:
        if self.name in DETERMINISTIC_FIELDS and not self.discovered:
            raise ValueError(
                f"{self.name!r} is a deterministic field; the SLM must not "
                "produce it (PLAN.md invariant 7)"
            )
        if self.name == SUMMARY_FIELD_NAME:
            if self.value is not None:
                raise ValueError(
                    "complaint_summary must be selected line indices, not "
                    "prose: RawField.value must be None and line_indices "
                    "non-empty (PLAN.md invariant 2)"
                )
            if not self.line_indices:
                raise ValueError(
                    "complaint_summary requires at least one selected "
                    "line_index (PLAN.md invariant 2)"
                )
        if len(set(self.line_indices)) != len(self.line_indices):
            raise ValueError(
                f"duplicate line_indices in RawField {self.name!r}"
            )
        return self
```

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/contracts/ocr.py
```

Expected: no output. The new flags allow discoveries without weakening extractive-summary checks.
