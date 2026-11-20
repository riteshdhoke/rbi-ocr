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


# Version stored with results so a changed extractor does not serve stale values.
EXTRACTION_REVISION = "open-fields-bank-logo-v2"

# Semantic aliases identify a concept, not a list of allowed output fields.
# Unknown/custom labels continue through the normal label-word matcher.
_ALIASES = {
    "recipient_name": (
        "recipient_name",
        "recipient",
        "addressee",
        "addressee_name",
        "name_of_addressee",
    ),
    "sender_name": (
        "sender_name",
        "sender",
        "signatory",
        "signatory_name",
        "author",
        "signed_by",
    ),
    "recipient_designation": (
        "recipient_designation",
        "addressee_designation",
        "designation_of_addressee",
        "recipient_title",
    ),
    "sender_designation": (
        "sender_designation",
        "signatory_designation",
        "designation_of_sender",
        "sender_title",
    ),
    "mobile": ("mobile", "mobile_number"),
    "email": ("email", "email_id", "email_address", "e_mail", "e_mail_id"),
    "postal_address": (
        "address",
        "postal_address",
        "physical_address",
        "mailing_address",
        "street_address",
    ),
    "letter_content": (
        "letter_content",
        "letter_body",
        "body_of_letter",
        "content_of_this_letter",
        "content",
        "message",
    ),
    "account_number": ("account_number", "account_no", "a_c_number"),
    "transaction_reference": ("transaction_reference", "utr_rrn", "utr", "rrn"),
}


def canonical_field(name: str) -> str:
    label = field_name(name)
    # Natural questions are hints about a field, never executable instructions.
    label = re.sub(r"^(?:what_is|what_are|find|extract|give_me)_", "", label)
    tokens = _tokens(label)
    for canonical, aliases in _ALIASES.items():
        if any(tokens == _tokens(alias) for alias in aliases):
            return canonical
        base = re.sub(r"_(?:[2-9]|[1-9][0-9]{1,2})$", "", label)
        if base != label and any(_tokens(base) == _tokens(alias) for alias in aliases):
            return canonical
    return "_".join(sorted(tokens))


def matches_field(candidate: str, requested: str) -> bool:
    wanted = field_name(requested)
    candidate_family, requested_family = (
        canonical_field(candidate),
        canonical_field(wanted),
    )
    if candidate_family == requested_family:
        return True
    if candidate_family in _ALIASES and requested_family in _ALIASES:
        return False  # Email addresses and postal addresses are distinct concepts.
    tokens = _tokens(wanted)
    return bool(tokens and tokens <= _tokens(candidate))


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
        # A broad request such as 'name' must not arbitrarily select a bank,
        # recipient or sender. Distinct occurrences within one concept remain.
        if (
            _tokens(wanted) in ({"name"}, {"designation"}, {"title"}, {"reference"})
            and len({canonical_field(name) for name, _ in candidates}) > 1
            and len({field.value for _, field in candidates}) > 1
        ):
            extracted[wanted] = []
            continue
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
    other: dict[str, Field] = {}
    seen_other = set()
    for name, field in sorted(
        fields.items(),
        key=lambda item: (item[1].validator_passed is not True, -item[1].confidence),
    ):
        identity = (canonical_field(name), field.value)
        if name not in matched and identity not in seen_other:
            other[name] = field
            seen_other.add(identity)
    missing = tuple(name for name, values in extracted.items() if not values)
    return extracted, other, missing
