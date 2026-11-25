"""Open-vocabulary field discovery from OCR labels, without a model or allowlist.

This complements semantic model extraction. It copies explicit ``Label: value``
pairs, including labels whose value occupies the next nearby OCR line. It does
not pretend that arbitrary narrative text can be understood without a model.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence

from app.contracts.envelope import Evidence, Field, FieldFlag
from app.contracts.ocr import OCRLine, OCRPage, TextSpan
from app.field_selection import field_name, matches_field
from app.stages.s5_letter import discover_letter_fields

# Separate multiple form cells only at explicit layout separators. A lone colon
# inside a URL, time or value is not treated as a second label.
_NUMBERED_LABEL = re.compile(r"^\s*\d+[.)]\s+([^:：]+)$")
_CELL = re.compile(r"(?:^|[;\t]| {2,})([^:：\n;\t]{1,80})[:：]")


def _same_row_value(label: OCRLine, value: OCRLine) -> bool:
    top, bottom = min(label.bbox[1::2]), max(label.bbox[1::2])
    other_top, other_bottom = min(value.bbox[1::2]), max(value.bbox[1::2])
    overlap = min(bottom, other_bottom) - max(top, other_top)
    return (
        abs(
            math.atan2(label.bbox[3] - label.bbox[1], label.bbox[2] - label.bbox[0])
            - math.atan2(value.bbox[3] - value.bbox[1], value.bbox[2] - value.bbox[0])
        )
        <= math.radians(10)
        and overlap > 0
        and overlap >= min(bottom - top, other_bottom - other_top) * 0.5
        and min(value.bbox[0::2]) >= max(label.bbox[0::2])
    )


def add_discovered(fields: dict[str, Field], name: str, candidate: Field) -> None:
    """Keep distinct occurrences rather than discarding all but the best value."""
    key, occurrence = name, 2
    while key in fields:
        existing = fields[key]
        if existing.value == candidate.value:
            if (
                existing.label is None
                and candidate.label is not None
                and existing.evidence.page_no == candidate.evidence.page_no
                and existing.evidence.span == candidate.evidence.span
            ):
                fields[key] = existing.model_copy(update={"label": candidate.label})
            return
        key = f"{name}_{occurrence}"
        occurrence += 1
    fields[key] = candidate


def discover_fields(
    pages: Sequence[OCRPage],
    existing: Mapping[str, Field] | None = None,
    requested_fields: Sequence[str] = (),
    *,
    requested_only: bool = False,
) -> dict[str, Field]:
    """Copy labeled values, preserving OCR text, locations and repeated values.

    Existing validator/model fields retain their checks and preferred names.
    Additional occurrences get numeric suffixes. Unvalidated discoveries always
    have ``validator_passed=None`` and ``source='ocr_label'``.
    """
    def wanted(name: str) -> bool:
        return not requested_only or any(
            matches_field(name, field) for field in requested_fields
        )

    fields = dict(existing or {})
    fields_by_line: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, field in fields.items():
        if field.label is None:
            fields_by_line[field.evidence.page_no, field.evidence.span.line_index].append(name)
    claimed = {
        (field.evidence.page_no, span.line_index)
        for field in fields.values()
        if field.source in {"lookup_table", "regex_validator"}
        for span in (field.evidence.spans or (field.evidence.span,))
    }
    for page in sorted(pages, key=lambda p: p.page_no):
        for name, candidate in discover_letter_fields(
            page, wanted=wanted if requested_only else None
        ).items():
            add_discovered(fields, name, candidate)
        lines = page.lines
        for position, line in enumerate(lines):
            if line.confidence < 0.8:
                continue
            matches = list(_CELL.finditer(line.text))
            if not matches:
                numbered = _NUMBERED_LABEL.fullmatch(line.text)
                if numbered:
                    matches = [numbered]
                elif (
                    (page.page_no, line.line_index) not in claimed
                    and sum(
                        1
                        for peer in lines
                        if peer is line
                        or _same_row_value(line, peer)
                        or _same_row_value(peer, line)
                    )
                    < 3
                    and len(line.text) <= 60
                    and len(line.text.split()) <= 6
                    and not any(c.isdigit() for c in line.text)
                    and not re.search(r"[.,!?;]", line.text)
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
                        following.confidence >= 0.8
                        and not _CELL.search(following.text)
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
                # A validated or model-extracted value can occupy part of this
                # cell. Keep that value's evidence and checks while adding the
                # visible label from the cell.
                for field_key in fields_by_line.get((page.page_no, source.line_index), ()):
                    field = fields[field_key]
                    span = field.evidence.span
                    if field.label is None and start <= span.start < span.end <= end:
                        fields[field_key] = field.model_copy(update={"label": label})
                if not wanted(name):
                    continue
                candidate = Field(
                    value=source.text[start:end],
                    label=label,
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
