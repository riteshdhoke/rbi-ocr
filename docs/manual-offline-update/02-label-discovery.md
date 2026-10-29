# 02 — Create `app/stages/s5_discover.py`

Type this complete new file. Imports become available after step 05, so run only the syntax check for now.

```python
"""Open-vocabulary field discovery from OCR labels, without a model or allowlist.

This complements semantic model extraction. It copies explicit ``Label: value``
pairs, including labels whose value occupies the next nearby OCR line. It does
not pretend that arbitrary narrative text can be understood without a model.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from app.contracts.envelope import Evidence, Field, FieldFlag
from app.contracts.ocr import OCRLine, OCRPage, TextSpan
from app.field_selection import field_name, matches_field

# Separate multiple form cells only at explicit layout separators. A lone colon
# inside a URL, time or value is not treated as a second label.
_NUMBERED_LABEL = re.compile(r"^\s*\d+[.)]\s+([^:：]+)$")
_CELL = re.compile(r"(?:^|[;\t]| {2,})([^:：\n;\t]{1,80})[:：]")


def _same_row_value(label: OCRLine, value: OCRLine) -> bool:
    top, bottom = min(label.bbox[1::2]), max(label.bbox[1::2])
    other_top, other_bottom = min(value.bbox[1::2]), max(value.bbox[1::2])
    overlap = min(bottom, other_bottom) - max(top, other_top)
    return (
        overlap > 0
        and overlap >= min(bottom - top, other_bottom - other_top) * 0.5
        and min(value.bbox[0::2]) >= max(label.bbox[0::2])
    )


def add_discovered(fields: dict[str, Field], name: str, candidate: Field) -> None:
    """Keep distinct occurrences rather than discarding all but the best value."""
    key, occurrence = name, 2
    while key in fields:
        if fields[key].value == candidate.value:
            return
        key = f"{name}_{occurrence}"
        occurrence += 1
    fields[key] = candidate


def discover_fields(
    pages: Sequence[OCRPage],
    existing: Mapping[str, Field] | None = None,
    requested_fields: Sequence[str] = (),
) -> dict[str, Field]:
    """Copy labeled values, preserving OCR text, locations and repeated values.

    Existing validator/model fields retain their checks and preferred names.
    Additional occurrences get numeric suffixes. Unvalidated discoveries always
    have ``validator_passed=None`` and ``source='ocr_label'``.
    """
    fields = dict(existing or {})
    for page in sorted(pages, key=lambda p: p.page_no):
        lines = page.lines
        for position, line in enumerate(lines):
            matches = list(_CELL.finditer(line.text))
            if not matches:
                numbered = _NUMBERED_LABEL.fullmatch(line.text)
                if numbered:
                    matches = [numbered]
                elif (
                    len(line.text) <= 80
                    and position + 1 < len(lines)
                    and (
                        _same_row_value(line, lines[position + 1])
                        or any(
                            matches_field(line.text, wanted)
                            and matches_field(wanted, line.text)
                            for wanted in requested_fields
                        )
                    )
                ):
                    plain_label = re.fullmatch(r"(.+)", line.text)
                    if plain_label:
                        matches = [plain_label]
            for number, match in enumerate(matches):
                label = match.group(1).strip()
                if not any(unicodedata.category(c).startswith("L") for c in label):
                    continue
                name = field_name(label)
                if not name:
                    continue
                end = (
                    matches[number + 1].start()
                    if number + 1 < len(matches)
                    else len(line.text)
                )
                start = match.end()
                while start < end and line.text[start].isspace():
                    start += 1
                while end > start and line.text[end - 1].isspace():
                    end -= 1
                source = line
                if start == end and len(matches) == 1 and position + 1 < len(lines):
                    # A label-only line may have a value directly below it. Avoid
                    # treating a distant line or the next label as its value.
                    following = lines[position + 1]
                    y_bottom = max(line.bbox[1::2])
                    y_top = min(following.bbox[1::2])
                    height = max(line.bbox[1::2]) - min(line.bbox[1::2])
                    aligned = abs(
                        min(line.bbox[0::2]) - min(following.bbox[0::2])
                    ) <= max(height * 2, 20)
                    if (
                        not _CELL.search(following.text)
                        and not _NUMBERED_LABEL.fullmatch(following.text)
                        and (
                            _same_row_value(line, following)
                            or (
                                aligned and 0 <= y_top - y_bottom <= max(height * 2, 20)
                            )
                        )
                    ):
                        source = following
                        start = len(source.text) - len(source.text.lstrip())
                        end = len(source.text.rstrip())
                if end <= start:
                    continue
                candidate = Field(
                    value=source.text[start:end],
                    confidence=min(source.confidence, 0.85),
                    source="ocr_label",
                    validator_passed=None,
                    evidence=Evidence(
                        page_no=page.page_no,
                        bbox=list(source.bbox),
                        span=TextSpan(
                            line_index=source.line_index, start=start, end=end
                        ),
                    ),
                    flags=(FieldFlag.HANDWRITTEN_SOURCE,)
                    if source.is_handwritten
                    else (),
                )
                add_discovered(fields, name, candidate)
    return fields
```

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -m py_compile app/stages/s5_discover.py
```

Expected: no output, exit code 0.
