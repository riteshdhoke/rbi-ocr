"""Construct fields from exact OCR spans, including multi-line values."""

from __future__ import annotations

from collections.abc import Sequence

from app.contracts.envelope import Evidence, Field, FieldFlag, FieldSource
from app.contracts.ocr import OCRPage, RawField, TextSpan, bbox_from_xyxy


def field_from_spans(
    page: OCRPage,
    spans: Sequence[TextSpan],
    *,
    source: FieldSource,
    confidence: float | None = None,
    review: bool = False,
) -> Field:
    if not spans:
        raise ValueError("At least one source span is required")
    lines = [page.line_by_index(span.line_index) for span in spans]
    parts = []
    for line, span in zip(lines, spans, strict=True):
        if span.end > len(line.text) or span.end <= span.start:
            raise ValueError("Evidence span must select nonempty OCR text")
        parts.append(line.text[span.start : span.end])
    flags = []
    if any(line.is_handwritten for line in lines):
        flags.append(FieldFlag.HANDWRITTEN_SOURCE)
    if review or any(line.confidence < 0.85 for line in lines):
        flags.append(FieldFlag.LOW_CONFIDENCE)
    return Field(
        value="\n".join(parts),
        source=source,
        validator_passed=None,
        confidence=min(
            [line.confidence for line in lines]
            + ([confidence] if confidence is not None else [])
        ),
        evidence=Evidence(
            page_no=page.page_no,
            bbox=bbox_from_xyxy(
                min(min(line.bbox[0::2]) for line in lines),
                min(min(line.bbox[1::2]) for line in lines),
                max(max(line.bbox[0::2]) for line in lines),
                max(max(line.bbox[1::2]) for line in lines),
            ),
            span=spans[0],
            spans=tuple(spans) if len(spans) > 1 else (),
        ),
        flags=tuple(flags),
    )


def ground_multiline(raw: RawField, page: OCRPage) -> Field | None:
    """Accept an exact join of cited OCR lines; never repair model prose."""
    if raw.value is None or len(raw.line_indices) < 2:
        return None
    known = {line.line_index: line for line in page.lines}
    if any(index not in known for index in raw.line_indices):
        return None
    lines = [known[index] for index in raw.line_indices]
    if not all(line.text.strip() for line in lines):
        return None
    texts = [line.text.strip() for line in lines]
    if raw.value.strip() not in {"\n".join(texts), " ".join(texts), ", ".join(texts)}:
        return None
    spans = [
        TextSpan(
            line_index=line.line_index,
            start=len(line.text) - len(line.text.lstrip()),
            end=len(line.text.rstrip()),
        )
        for line in lines
    ]
    return field_from_spans(page, spans, source="slm", confidence=raw.confidence)
