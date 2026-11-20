"""Conservative letter structure and contact extraction from readable OCR.

These recognizers complement arbitrary labeled fields; absent evidence produces
no value. Sender and recipient are separate roles. No OCR spelling is repaired.
"""

from __future__ import annotations

import math
import re
import statistics

from app.contracts.envelope import Field
from app.contracts.ocr import OCRLine, OCRPage, TextSpan
from app.stages.s5_evidence import field_from_spans

_EMAIL = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE
)
_GREETING = re.compile(
    r"^\s*(?:dear|hi|hello)\s+(?P<name>[^,!:;]+)[,!:]?\s*$", re.IGNORECASE
)
_CLOSING = re.compile(
    r"^\s*(?:thanks(?:\s+again)?|thank you|sincerely|regards|best regards|kind regards|yours(?:\s+(?:sincerely|faithfully|truly))?)\s*[,!.]?\s*$",
    re.IGNORECASE,
)
_STREET = re.compile(
    r"\b\d{1,6}\w?\s+.{2,65}\b(?:ave(?:nue)?|street|st|road|rd|lane|ln|drive|dr|boulevard|blvd|way|sector)\b",
    re.IGNORECASE,
)
_POSTAL = re.compile(r"(?:\b[A-Z]{2}\s*\d{5}(?:-\d{4})?\b|\b\d{6}\b)", re.IGNORECASE)
_ROLE = re.compile(
    r"\b(?:manager|director|officer|president|secretary|engineer|coordinator|founder|partner|professor|principal|head|executive)\b",
    re.IGNORECASE,
)


def _full(line: OCRLine) -> TextSpan:
    return TextSpan(
        line_index=line.line_index,
        start=len(line.text) - len(line.text.lstrip()),
        end=len(line.text.rstrip()),
    )


def _angle(line: OCRLine) -> float:
    return math.atan2(line.bbox[3] - line.bbox[1], line.bbox[2] - line.bbox[0])


def _name(text: str) -> bool:
    words = text.strip().split()
    return (
        1 <= len(words) <= 5
        and all(
            len(word.strip(".-")) >= 1 and all(c.isalpha() or c in ".-'’" for c in word)
            for word in words
        )
        and not _ROLE.search(text)
    )


def _address_blocks(page: OCRPage) -> list[Field]:
    results = []
    used = set()
    for street in page.lines:
        if street.line_index in used or not _STREET.search(street.text):
            continue
        theta = _angle(street)
        c, s = math.cos(theta), math.sin(theta)

        def position(line, c=c, s=s):
            x = sum(line.bbox[0::2]) / 4
            y = sum(line.bbox[1::2]) / 4
            return c * x + s * y, -s * x + c * y

        sx, sy = position(street)
        height = max(12, math.dist(street.bbox[0:2], street.bbox[6:8]))
        street_left = min(
            c * x + s * y
            for x, y in zip(street.bbox[0::2], street.bbox[1::2], strict=True)
        )
        street_width = (
            max(
                c * x + s * y
                for x, y in zip(street.bbox[0::2], street.bbox[1::2], strict=True)
            )
            - street_left
        )
        block = []
        for line in page.lines:
            x, y = position(line)
            aligned = abs(_angle(line) - theta) <= math.radians(12)
            small_fragment = line.text.strip().isdigit() and len(line.text.strip()) <= 3
            line_left = min(
                c * x + s * y
                for x, y in zip(line.bbox[0::2], line.bbox[1::2], strict=True)
            )
            if (
                street_left - height * 1.5
                <= line_left
                <= street_left + max(height * 2, street_width * 0.25)
                and (aligned or small_fragment)
                and abs(x - sx) <= max(100, page.width_px * 0.15)
                and -height * 1.8 <= y - sy <= height * 3.3
            ):
                block.append(line)
        block.sort(
            key=lambda line: (
                round(position(line)[1] / max(height * 0.65, 1)),
                position(line)[0],
            )
        )
        # Require both a street and a postal destination; do not complete a
        # damaged address from contextual guesses or discard unreadable parts.
        if not any(_POSTAL.search(line.text) for line in block):
            continue
        if any(line.confidence < 0.85 for line in block):
            continue
        results.append(
            field_from_spans(
                page, [_full(line) for line in block], source="ocr_layout", review=True
            )
        )
        used.update(line.line_index for line in block)
    return results


def discover_letter_fields(page: OCRPage) -> dict[str, Field]:
    fields: dict[str, Field] = {}

    def add(name: str, value: Field):
        key, i = name, 2
        while key in fields:
            if fields[key].value == value.value:
                return
            key, i = f"{name}_{i}", i + 1
        fields[key] = value

    for line in page.lines:
        if line.confidence < 0.85:
            continue
        for match in _EMAIL.finditer(line.text):
            # Sentence punctuation is outside the address, never an inferred fix.
            end = match.end()
            while end > match.start() and line.text[end - 1] in ".":
                end -= 1
            add(
                "email",
                field_from_spans(
                    page,
                    [
                        TextSpan(
                            line_index=line.line_index, start=match.start(), end=end
                        )
                    ],
                    source="ocr_layout",
                ),
            )
    for address in _address_blocks(page):
        add("postal_address", address)

    lines = page.lines
    greeting = next(
        (
            (i, _GREETING.fullmatch(line.text))
            for i, line in enumerate(lines)
            if line.confidence >= 0.85 and _GREETING.fullmatch(line.text)
        ),
        None,
    )
    if greeting is None:
        return fields
    start, match = greeting
    line = lines[start]
    name = match.group("name").strip()
    if _name(name):
        name_start = line.text.find(name, match.start("name"))
        add(
            "recipient_name",
            field_from_spans(
                page,
                [
                    TextSpan(
                        line_index=line.line_index,
                        start=name_start,
                        end=name_start + len(name),
                    )
                ],
                source="ocr_layout",
            ),
        )
    closing = next(
        (
            i
            for i in range(start + 1, len(lines))
            if _CLOSING.fullmatch(lines[i].text) and lines[i].confidence >= 0.85
        ),
        None,
    )
    if closing is None:
        return fields
    if closing + 1 < len(lines):
        signature = lines[closing + 1]
        close = lines[closing]
        height = max(close.bbox[1::2]) - min(close.bbox[1::2])
        if (
            signature.confidence >= 0.85
            and _name(signature.text)
            and 0
            <= min(signature.bbox[1::2]) - min(close.bbox[1::2])
            <= max(120, height * 3)
        ):
            add(
                "sender_name",
                field_from_spans(page, [_full(signature)], source="ocr_layout"),
            )
            if closing + 2 < len(lines):
                role = lines[closing + 2]
                if (
                    role.confidence >= 0.85
                    and len(role.text.split()) <= 10
                    and _ROLE.search(role.text)
                    and min(role.bbox[1::2]) - max(signature.bbox[1::2])
                    <= max(80, height * 2)
                ):
                    add(
                        "sender_designation",
                        field_from_spans(page, [_full(role)], source="ocr_layout"),
                    )
    height = statistics.median(
        max(item.bbox[1::2]) - min(item.bbox[1::2])
        for item in lines[start : closing + 1]
    )
    left = min(lines[start].bbox[0::2]) - height * 2
    body = [
        item
        for item in lines[start + 1 : closing]
        if min(item.bbox[0::2]) >= left
        and item.confidence >= 0.8
        and abs(_angle(item) - _angle(lines[start])) <= math.radians(12)
    ]
    if body:
        add(
            "letter_content",
            field_from_spans(
                page, [_full(item) for item in body], source="ocr_layout", review=True
            ),
        )
    return fields


def conflicting_letter_role(name: str, value: str, page: OCRPage) -> bool:
    """A grounded signature must not be silently relabeled as the addressee."""
    from app.field_selection import canonical_field

    role = canonical_field(name)
    if role not in {
        "recipient_name",
        "sender_name",
        "recipient_designation",
        "sender_designation",
    }:
        return False
    observed = discover_letter_fields(page)
    correct = observed.get(role)
    opposite_role = (
        role.replace("recipient_", "sender_")
        if role.startswith("recipient_")
        else role.replace("sender_", "recipient_")
    )
    opposite = observed.get(opposite_role)
    return bool(
        opposite
        and value.strip().casefold() == (opposite.value or "").strip().casefold()
        and (correct is None or correct.value != opposite.value)
    )
