"""Normalize user field lists and partition evidence-bearing extraction results."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.contracts.envelope import Field


def field_name(label: str) -> str:
    """Stable Unicode field key derived from a label, without a name registry."""
    label = unicodedata.normalize("NFKC", label).casefold().strip()
    label = re.sub(r"^\s*\d+[.)]\s*", "", label)
    chars = [
        c if c == "_" or unicodedata.category(c)[0] in "LNM" else "_" for c in label
    ]
    return re.sub("_+", "_", "".join(chars)).strip("_")


def parse_requested_fields(value: str | Sequence[str]) -> tuple[str, ...]:
    """Ignore empty CSV entries; case, spacing and ordering do not change a request."""
    items = value.split(",") if isinstance(value, str) else value
    names = set()
    for item in items:
        if not item.strip():
            continue
        name = field_name(item)
        if not name or not any(unicodedata.category(c)[0] in "LN" for c in name):
            raise ValueError("Each requested field must contain letters or digits.")
        names.add(name)
    return tuple(sorted(names))


def _tokens(name: str) -> set[str]:
    # These are grammatical connectors, not a field-name or synonym registry.
    return {
        word
        for word in field_name(name).split("_")
        if word and word not in {"of", "the", "and"}
    }


def matches_field(candidate: str, requested: str) -> bool:
    wanted = field_name(requested)
    tokens = _tokens(wanted)
    return field_name(candidate) == wanted or bool(
        tokens and tokens <= _tokens(candidate)
    )


def partition_fields(
    fields: Mapping[str, Field],
    requested: Sequence[str],
) -> tuple[dict[str, list[Field]], dict[str, Field], tuple[str, ...]]:
    """Return requested matches, remaining discoveries and explicitly missing names.

    Match normalized labels by their words (e.g. 'complainant name' matches
    'name_of_complainant'). Keep distinct values; repeated aliases of the same
    value use the best validated/confident evidence. Do not guess synonyms.
    """
    extracted: dict[str, list[Field]] = {}
    matched = set()
    for wanted in requested:
        candidates = [
            (name, field)
            for name, field in fields.items()
            if matches_field(name, wanted)
        ]
        candidates.sort(
            key=lambda item: (
                item[1].validator_passed is not True,
                -item[1].confidence,
                item[1].evidence.page_no,
                item[1].evidence.span.line_index,
                item[0],
            )
        )
        values: dict[str, Field] = {}
        for name, field in candidates:
            if field.value is None:
                continue
            matched.add(name)
            values.setdefault(field.value, field)
        extracted[wanted] = list(values.values())
    other = {name: field for name, field in fields.items() if name not in matched}
    missing = tuple(name for name, values in extracted.items() if not values)
    return extracted, other, missing
