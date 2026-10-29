"""Stage 5 — deterministic field extraction.  **No model runs in this file.**

PLAN.md invariant 7: account numbers, IFSC codes, card fragments, UTR/RRN
references, MICR codes, amounts and dates are owned by regular expressions,
checksums and lookup tables.  They never reach the SLM, and nothing here imports
a provider.  The module's only third-party dependency is ``rapidfuzz``, used for
the one field that genuinely needs approximate matching (``bank_name``).

Three rules shape every extractor here:

**A failed check is flagged, never fixed** (invariant 9).  A hit whose Luhn
digit is wrong, whose calendar date does not exist or whose thousands grouping
is malformed comes back with ``validator_passed=False``, the matching
:class:`~app.contracts.envelope.FieldFlag`, and ``normalized=None``.  The
``value`` is always the verbatim source substring.  There is no code path that
edits a recognised character.

**Every hit points at pixels.**  A :class:`ValidatorHit` carries
:class:`~app.contracts.envelope.Evidence` — page number, the source line's
bbox, and the character span inside that line.  Character offsets survive Indic
digit normalisation because :func:`normalise_digits` is a strict 1:1 code-point
mapping, so a span found in the normalised text indexes the original text
unchanged.

**Checks are enumerated, not assumed.**  :class:`ValidatorHit.checks` lists what
actually ran.  India has no universal bank-account checksum, so an
``account_number`` hit carries ``digits`` and ``length_class`` outcomes and *no*
``checksum`` outcome at all — the absence is the honest statement, and
``validator_passed=True`` on such a hit means "structurally plausible", never
"arithmetically verified".  :data:`ACCOUNT_CHECKSUM_POLICY` spells that out for
anyone reading an envelope.

Validators run concurrently over the OCR text (BUILD-BRIEF M5) on a thread pool
in :func:`run_validators`; each one is a pure function of a single page's lines,
so the concurrency cannot change a result — output order is fixed by
``(page_no, line_index, span.start, field_name)``, not by completion order.

This module does NOT handle:
  * anything the SLM owns — complainant name/address, relief sought, prior
    complaint reference, summary spans (:mod:`app.stages.s5_slm`);
  * the grounding gate (invariant 8).  Validator hits *are* the ground truth;
    they are not checked against anything;
  * translation.  ``value`` is native-script and immutable; ``value_en`` is
    attached later by :mod:`app.stages.s6_translate` (invariant 10);
  * choosing which page a document-level field comes from — :meth:`
    ValidatorReport.fields` ranks candidates, but the envelope is built by
    :mod:`app.stages.assemble`;
  * OCR errors.  A validator can only report that what OCR produced does or does
    not satisfy a rule; recovering a mangled digit is Tier-2's job, not this
    module's;
  * the real RBI IFSC master.  :class:`IFSCMaster` is a loader with a committed
    synthetic fixture (OPEN-QUESTIONS Q4b); dropping the real CSV in is a path
    change;
  * Devanagari month names, Indian financial-year notation (``FY 2025-26``),
    words-as-amounts (``forty five thousand only``), signed amounts
    (``-4,500.00`` on a statement: the leading sign is what disambiguates a date
    component, so it is treated as one), or cheque MICR bands printed in E-13B
    symbols.  Each is recorded in ``docs/decisions-wse3.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as _Field
from rapidfuzz import fuzz

from app.contracts.envelope import Evidence, Field, FieldFlag, FieldSource
from app.contracts.ocr import (
    DETERMINISTIC_FIELDS,
    BBox,
    Confidence,
    OCRLine,
    OCRPage,
    TextSpan,
)
from app.resources import bank_aliases_path, ifsc_master_sample_path

__all__ = [
    "ACCOUNT_CHECKSUM_POLICY",
    "ACCOUNT_LENGTH_CLASSES",
    "AMOUNT_MULTIPLIERS",
    "BANK_NAME_MARGIN",
    "BANK_NAME_SCORE_FLOOR",
    "CARD_PAN_LENGTHS",
    "IFSC_STRUCTURE",
    "INDIC_DIGIT_TRANSLATION",
    "MICR_STRUCTURE",
    "REFERENCE_FORMATS",
    "VALIDATORS",
    "VALIDATOR_FIELD_NAMES",
    "AmountParse",
    "BankRecord",
    "BankTable",
    "CheckOutcome",
    "DateParse",
    "IFSCMaster",
    "IFSCRecord",
    "PageScan",
    "ValidatorHit",
    "ValidatorReport",
    "account_length_class",
    "classify_reference",
    "default_bank_records",
    "default_bank_table",
    "load_bank_table",
    "non_party_institution_names",
    "default_ifsc_master",
    "format_indian_grouping",
    "luhn_check",
    "merge_spans",
    "normalise_digits",
    "parse_amount",
    "parse_indian_date",
    "run_validators",
    "validate_account_number",
    "validate_amount",
    "validate_bank_name",
    "validate_card_fragment",
    "validate_date",
    "validate_ifsc",
    "validate_micr",
    "validate_utr_rrn",
]


# --------------------------------------------------------------------------- #
# Numeral normalisation — offset preserving by construction
# --------------------------------------------------------------------------- #

#: Code point of digit zero for every decimal script an RBI complaint can
#: plausibly carry.  Each block lays its ten digits out contiguously, which is
#: what makes a 1:1 translation table correct rather than a lucky approximation.
_DIGIT_ZEROS: Final[tuple[tuple[str, int], ...]] = (
    ("Arab", 0x0660),  # Arabic-Indic
    ("Arab-ext", 0x06F0),  # Extended Arabic-Indic (Urdu)
    ("Deva", 0x0966),
    ("Beng", 0x09E6),
    ("Guru", 0x0A66),
    ("Gujr", 0x0AE6),
    ("Orya", 0x0B66),
    ("Taml", 0x0BE6),
    ("Telu", 0x0C66),
    ("Knda", 0x0CE6),
    ("Mlym", 0x0D66),
)


def _build_digit_translation() -> dict[int, str]:
    table: dict[int, str] = {}
    for _script, zero in _DIGIT_ZEROS:
        for offset in range(10):
            table[zero + offset] = str(offset)
    return table


#: ``str.translate`` table mapping every supported Indic digit to its ASCII
#: equivalent.  **One code point in, one code point out** — this is why a span
#: located in the normalised text indexes the original text unchanged, and it is
#: the reason evidence offsets can be trusted at all.
INDIC_DIGIT_TRANSLATION: Final[Mapping[int, str]] = _build_digit_translation()


def normalise_digits(text: str) -> str:
    """Map Indic decimal digits onto ASCII, preserving length and offsets.

    Nothing else is touched: letters, punctuation, matras and ZWJ/ZWNJ all pass
    through.  Callers that need a *value* use the original string; this form is
    only ever used to find and interpret numbers.
    """
    return text.translate(dict(INDIC_DIGIT_TRANSLATION))


def _is_offset_preserving(original: str, normalised: str) -> bool:
    """Guard used by tests: normalisation must never move a character."""
    return len(original) == len(normalised)


# --------------------------------------------------------------------------- #
# Small deterministic helpers
# --------------------------------------------------------------------------- #


def luhn_check(digits: str) -> bool:
    """Mod-10 (Luhn) check over an ASCII digit string.

    Raises:
        ValueError: when ``digits`` is empty or contains a non-digit — a caller
            that has not stripped separators has a bug, and returning ``False``
            would report it as a failed card instead.
    """
    if not digits:
        raise ValueError("luhn_check needs at least one digit")
    if not digits.isdigit():
        raise ValueError(f"luhn_check expects ASCII digits only, got {digits!r}")
    total = 0
    for position, character in enumerate(reversed(digits)):
        value = ord(character) - 48
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def merge_spans(spans: Sequence[TextSpan]) -> tuple[TextSpan, ...]:
    """Merge overlapping/adjacent spans per line, in ``(line, start)`` order.

    Sentinel masking replaces a span with a fixed token; two overlapping spans
    would corrupt each other's offsets, so the masker is handed a disjoint set.
    """
    by_line: dict[int, list[TextSpan]] = {}
    for span in spans:
        by_line.setdefault(span.line_index, []).append(span)
    merged: list[TextSpan] = []
    for line_index in sorted(by_line):
        ordered = sorted(by_line[line_index], key=lambda s: (s.start, s.end))
        current_start, current_end = ordered[0].start, ordered[0].end
        for span in ordered[1:]:
            if span.start <= current_end:
                current_end = max(current_end, span.end)
                continue
            merged.append(
                TextSpan(line_index=line_index, start=current_start, end=current_end)
            )
            current_start, current_end = span.start, span.end
        merged.append(
            TextSpan(line_index=line_index, start=current_start, end=current_end)
        )
    return tuple(merged)


# --------------------------------------------------------------------------- #
# Hit model
# --------------------------------------------------------------------------- #

#: Confidence multipliers applied to the source line's OCR confidence.  Named
#: constants rather than literals so ``docs/TUNING.md`` has something to point
#: at and the eval harness can recalibrate them.
PASS_WEIGHT: Final[float] = 1.0
#: Structure passed but a stronger check could not be run (no checksum exists,
#: or the master list is not authoritative).
UNVERIFIED_WEIGHT: Final[float] = 0.9
#: A check that applies to this value failed.  Kept non-zero: the value is still
#: shown to a reviewer beside its crop, which is the whole point of flagging.
FAIL_WEIGHT: Final[float] = 0.5


class CheckOutcome(BaseModel):
    """One named check that actually ran, and what it said.

    The list of outcomes is the audit trail.  A check that is *absent* was not
    run — see :data:`ACCOUNT_CHECKSUM_POLICY` for why that distinction matters
    more here than the usual pass/fail boolean.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = _Field(min_length=1)
    passed: bool
    detail: str | None = None


class ValidatorHit(BaseModel):
    """One deterministic extraction, with its evidence and its check trail.

    Not an envelope :class:`~app.contracts.envelope.Field` yet: the assembler
    decides which candidate wins a field name and whether a document-level
    conflict adds a flag.  :meth:`to_field` performs the promotion once that is
    settled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Must be a member of :data:`~app.contracts.ocr.DETERMINISTIC_FIELDS`.
    field_name: str = _Field(min_length=1)
    #: Verbatim substring of the source line. Native script. Never edited.
    value: str = _Field(min_length=1)
    #: Canonical machine form (ASCII digits, ISO date, decimal amount).  ``None``
    #: whenever a check failed — a failed value is never canonicalised, because
    #: canonicalising it is one keystroke away from correcting it (invariant 9).
    normalized: str | None = None
    confidence: Confidence
    source: FieldSource = "regex_validator"
    evidence: Evidence
    checks: tuple[CheckOutcome, ...] = ()
    flags: tuple[FieldFlag, ...] = ()
    #: Free-text note for a human reviewer. Never parsed.
    detail: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ValidatorHit:
        if self.field_name not in DETERMINISTIC_FIELDS:
            raise ValueError(
                f"{self.field_name!r} is not a deterministic field; validators "
                f"own {sorted(DETERMINISTIC_FIELDS)} (PLAN.md invariant 7)"
            )
        if not self.validator_passed and self.normalized is not None:
            raise ValueError(
                "a hit with a failed check must not carry a normalized value: "
                "flag it, never correct it (PLAN.md invariant 9)"
            )
        if len(set(self.flags)) != len(self.flags):
            raise ValueError(f"duplicate flags: {self.flags}")
        return self

    @property
    def validator_passed(self) -> bool:
        """True when every check that ran passed.  No checks means nothing
        contradicted the match, which is how a pure-structure extractor reports
        success."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of the checks that failed, for logs and the review queue."""
        return tuple(check.name for check in self.checks if not check.passed)

    @property
    def span(self) -> TextSpan:
        """The claimed character range — what sentinel masking replaces."""
        return self.evidence.span

    def to_field(self, *, extra_flags: Sequence[FieldFlag] = ()) -> Field:
        """Promote to the published :class:`~app.contracts.envelope.Field`.

        ``VALIDATOR_FAILED`` is attached by the ``Field`` contract itself
        whenever ``validator_passed`` is false; this method does not duplicate
        that, it only carries the validator's own flags across.
        """
        flags: list[FieldFlag] = list(self.flags)
        for flag in extra_flags:
            if flag not in flags:
                flags.append(flag)
        return Field(
            value=self.value,
            normalized=self.normalized,
            confidence=self.confidence,
            source=self.source,
            validator_passed=self.validator_passed,
            evidence=self.evidence,
            flags=tuple(flags),
        )


class ValidatorReport(BaseModel):
    """Everything the deterministic pass found, across every page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: tuple[ValidatorHit, ...] = ()

    def hits_for(self, field_name: str) -> tuple[ValidatorHit, ...]:
        """Candidates for one field, best first."""
        return tuple(
            sorted(
                (hit for hit in self.hits if hit.field_name == field_name),
                key=_candidate_rank,
            )
        )

    def hits_on_page(self, page_no: int) -> tuple[ValidatorHit, ...]:
        """Every hit whose evidence points at ``page_no``, in reading order."""
        return tuple(
            sorted(
                (hit for hit in self.hits if hit.evidence.page_no == page_no),
                key=lambda hit: (hit.span.line_index, hit.span.start),
            )
        )

    def masked_spans(self, page_no: int) -> tuple[TextSpan, ...]:
        """Disjoint spans on ``page_no`` that the SLM prompt must sentinel-mask.

        Every validator hit is masked, passing or failing: a *wrong* account
        number is still an account number as far as data residency is concerned
        (PLAN.md invariant 10, and BUILD-BRIEF sec. 8's "do not send an account
        number through the SLM").
        """
        return merge_spans([hit.span for hit in self.hits_on_page(page_no)])

    def fields(self) -> dict[str, Field]:
        """Best candidate per field name, promoted to envelope fields.

        Ranking is deterministic: highest confidence, then a passing check trail
        over a failing one, then earliest evidence.  When two candidates
        disagree on the *normalised* value the winner carries
        ``MULTIPLE_CANDIDATES`` — a document that names two different account
        numbers is a review item, not a coin toss.
        """
        grouped: dict[str, list[ValidatorHit]] = {}
        for hit in self.hits:
            grouped.setdefault(hit.field_name, []).append(hit)
        result: dict[str, Field] = {}
        for name, candidates in grouped.items():
            ordered = sorted(candidates, key=_candidate_rank)
            winner = ordered[0]
            distinct = {
                hit.normalized or hit.value for hit in ordered if hit.normalized
            }
            extra: list[FieldFlag] = []
            if len(distinct) > 1:
                extra.append(FieldFlag.MULTIPLE_CANDIDATES)
            result[name] = winner.to_field(extra_flags=extra)
        return result


def _candidate_rank(hit: ValidatorHit) -> tuple[Any, ...]:
    """Deterministic ordering key — never depends on thread completion order."""
    return (
        0 if hit.validator_passed else 1,
        -round(hit.confidence, 6),
        hit.evidence.page_no,
        hit.span.line_index,
        hit.span.start,
        hit.value,
    )


# --------------------------------------------------------------------------- #
# IFSC master list — CSV or JSON, real list is a drop-in
# --------------------------------------------------------------------------- #


class BankRecord(BaseModel):
    """One bank: its IFSC prefix, its display name, and the spellings OCR sees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = _Field(pattern=r"^[A-Z]{4}$")
    name: str = _Field(min_length=1)
    #: Every string the fuzzy matcher may match against, ``name`` included.
    aliases: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _name_is_always_an_alias(cls, data: Any) -> Any:
        """Guarantee the display name is matchable without repeating it in JSON."""
        if isinstance(data, dict):
            name = data.get("name")
            aliases = list(data.get("aliases") or ())
            if isinstance(name, str) and name not in aliases:
                data = {**data, "aliases": (name, *aliases)}
        return data


class IFSCRecord(BaseModel):
    """One branch row of the master list."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ifsc: str = _Field(pattern=r"^[A-Z]{4}0[A-Z0-9]{6}$")
    bank_code: str = _Field(pattern=r"^[A-Z]{4}$")
    bank_name: str | None = None
    branch: str | None = None
    centre: str | None = None
    state: str | None = None


#: Header spellings accepted by :meth:`IFSCMaster.from_csv`, lower-cased.  The
#: RBI publication and the widely mirrored community CSVs disagree on case and
#: on ``CENTRE``/``CITY``; accepting both is cheaper than a transform step in
#: the deployment runbook.
_CSV_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "ifsc": ("ifsc", "ifsc_code", "ifsccode"),
    "bank_name": ("bank", "bank_name", "bankname"),
    "branch": ("branch", "branch_name", "office"),
    "centre": ("centre", "center", "city"),
    "state": ("state",),
}


class IFSCMaster:
    """Branch/bank lookup over an IFSC list, with an honest completeness flag.

    ``authoritative`` is the load-bearing attribute.  The committed fixture is a
    synthetic five-bank sample (OPEN-QUESTIONS Q4b), so a structurally valid
    code it has never heard of is **unverifiable**, not wrong: the hit stays
    ``validator_passed=True`` and carries ``IFSC_NOT_IN_MASTER`` only when the
    bank prefix is unknown too.  Load the real RBI list with
    ``authoritative=True`` and absence becomes a validation failure.  Nothing
    else in the module changes.

    Thread-safe by being immutable after construction; :func:`run_validators`
    shares one instance across its pool.
    """

    def __init__(
        self,
        records: Sequence[IFSCRecord],
        banks: Sequence[BankRecord] = (),
        *,
        authoritative: bool = False,
        source: str | None = None,
    ) -> None:
        self.authoritative = authoritative
        self.source = source
        self._by_ifsc: dict[str, IFSCRecord] = {
            record.ifsc.upper(): record for record in records
        }
        by_code: dict[str, BankRecord] = {bank.code: bank for bank in banks}
        for record in records:
            if record.bank_code in by_code:
                continue
            by_code[record.bank_code] = BankRecord(
                code=record.bank_code,
                name=record.bank_name or record.bank_code,
            )
        self._banks: dict[str, BankRecord] = by_code

    # ----------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self._by_ifsc)

    @property
    def banks(self) -> tuple[BankRecord, ...]:
        """Known banks, ordered by IFSC prefix."""
        return tuple(self._banks[code] for code in sorted(self._banks))

    def lookup(self, ifsc: str) -> IFSCRecord | None:
        """Exact branch row for ``ifsc`` (case-insensitive), or ``None``."""
        return self._by_ifsc.get(ifsc.strip().upper())

    def bank_for_code(self, code: str) -> BankRecord | None:
        """The bank whose IFSC prefix is ``code``, or ``None``."""
        return self._banks.get(code.strip().upper())

    def knows_prefix(self, code: str) -> bool:
        """Whether the four-letter bank prefix appears anywhere in the list."""
        return code.strip().upper() in self._banks

    # ------------------------------------------------------------- loaders
    @classmethod
    def load(
        cls, path: Path | str, *, authoritative: bool | None = None
    ) -> IFSCMaster:
        """Load from ``.json`` or ``.csv``, dispatching on the suffix.

        Raises:
            ValueError: on an unsupported suffix — guessing the format of a file
                that decides whether an identifier is valid is not a risk worth
                taking.
        """
        target = Path(path)
        suffix = target.suffix.lower()
        if suffix == ".json":
            return cls.from_json(target, authoritative=authoritative)
        if suffix == ".csv":
            return cls.from_csv(
                target,
                authoritative=True if authoritative is None else authoritative,
            )
        raise ValueError(
            f"unsupported IFSC master format {suffix!r} for {target}; "
            "expected .json or .csv"
        )

    @classmethod
    def from_json(
        cls, path: Path | str, *, authoritative: bool | None = None
    ) -> IFSCMaster:
        """Load the repository's JSON shape (see ``app/resources``).

        Raises:
            ValueError: when the payload is not an object with a ``branches``
                list — a truncated download must not silently become an empty
                master that passes everything.
        """
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "branches" not in payload:
            raise ValueError(
                f"{target} is not an IFSC master document: expected an object "
                'with a "branches" array'
            )
        banks = tuple(
            BankRecord.model_validate(entry) for entry in payload.get("banks", [])
        )
        records = tuple(
            IFSCRecord.model_validate(entry) for entry in payload["branches"]
        )
        declared = bool(payload.get("authoritative", False))
        return cls(
            records,
            banks,
            authoritative=declared if authoritative is None else authoritative,
            source=str(target),
        )

    @classmethod
    def from_csv(
        cls, path: Path | str, *, authoritative: bool = True
    ) -> IFSCMaster:
        """Load an RBI-style CSV.  Column names are matched case-insensitively.

        Rows whose IFSC does not satisfy :data:`IFSC_STRUCTURE` are skipped and
        counted in :attr:`skipped_rows` rather than raising: a 170 000-row public
        file reliably contains a handful of malformed rows, and refusing the
        whole list over them would leave the validator with no master at all.

        Raises:
            ValueError: when no IFSC column can be identified.
        """
        import csv

        target = Path(path)
        records: list[IFSCRecord] = []
        skipped = 0
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{target} has no CSV header row")
            mapping = _map_csv_columns(reader.fieldnames)
            if "ifsc" not in mapping:
                raise ValueError(
                    f"{target} has no IFSC column; looked for "
                    f"{_CSV_COLUMNS['ifsc']}, found {reader.fieldnames}"
                )
            for row in reader:
                raw = (row.get(mapping["ifsc"]) or "").strip().upper()
                if not IFSC_STRUCTURE.match(raw):
                    skipped += 1
                    continue
                records.append(
                    IFSCRecord(
                        ifsc=raw,
                        bank_code=raw[:4],
                        bank_name=_csv_value(row, mapping, "bank_name"),
                        branch=_csv_value(row, mapping, "branch"),
                        centre=_csv_value(row, mapping, "centre"),
                        state=_csv_value(row, mapping, "state"),
                    )
                )
        master = cls(records, authoritative=authoritative, source=str(target))
        master.skipped_rows = skipped
        return master

    #: Malformed rows dropped by :meth:`from_csv`.  Zero for JSON loads.
    skipped_rows: int = 0


def _map_csv_columns(fieldnames: Sequence[str]) -> dict[str, str]:
    """Map logical column names onto the header spellings actually present."""
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    mapping: dict[str, str] = {}
    for logical, candidates in _CSV_COLUMNS.items():
        for candidate in candidates:
            if candidate in lowered:
                mapping[logical] = lowered[candidate]
                break
    return mapping


def _csv_value(
    row: Mapping[str, str | None], mapping: Mapping[str, str], logical: str
) -> str | None:
    column = mapping.get(logical)
    if column is None:
        return None
    value = (row.get(column) or "").strip()
    return value or None


@lru_cache(maxsize=1)
def default_ifsc_master() -> IFSCMaster:
    """The committed synthetic master (OPEN-QUESTIONS Q4b), loaded once."""
    return IFSCMaster.from_json(ifsc_master_sample_path())


@dataclass(frozen=True)
class BankTable:
    """The bank-name reference data: who can be named, and who never can."""

    #: Banks the ``bank_name`` matcher may claim.
    records: tuple[BankRecord, ...] = ()
    #: Institutions that appear on complaints but are never the bank complained
    #: about — the regulator, the ombudsman's office, the payment operators.
    non_party: tuple[str, ...] = ()


@lru_cache(maxsize=8)
def load_bank_table(path: Path) -> BankTable:
    """Load a bank-alias table from ``path``, cached per path.

    Raises:
        ValueError: when the file parses but names no banks.  An empty table
            silently disables ``bank_name`` extraction for the whole
            deployment, which is exactly the kind of failure that shows up as
            "the field stopped being populated" three weeks later.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        BankRecord.model_validate(entry) for entry in payload.get("banks", [])
    )
    if not records:
        raise ValueError(
            f"bank alias table {path} names no banks; a table that matches "
            "nothing disables bank_name extraction entirely"
        )
    non_party = tuple(
        name
        for name in payload.get("non_party_institutions", [])
        if isinstance(name, str) and name.strip()
    )
    return BankTable(records=records, non_party=non_party)


def default_bank_table() -> BankTable:
    """The committed table (banks plus non-party institutions), loaded once.

    Kept separate from the IFSC master because the real RBI list carries branch
    rows and a single Latin bank spelling — never the Devanagari rendering an
    OCR'd complaint form actually contains.
    """
    return load_bank_table(bank_aliases_path())


def default_bank_records() -> tuple[BankRecord, ...]:
    """Bank records from the committed table."""
    return default_bank_table().records


def non_party_institution_names() -> tuple[str, ...]:
    """Non-party institution names from the committed table."""
    return default_bank_table().non_party


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

#: BUILD-BRIEF M5, verbatim: four letters, a literal zero, six alphanumerics.
IFSC_STRUCTURE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")

#: MICR line: 3-digit city + 3-digit bank + 3-digit branch.  There is no
#: check digit in the Indian MICR scheme, so structure is the whole test.
MICR_STRUCTURE: Final[re.Pattern[str]] = re.compile(r"^\d{9}$")

#: Candidate token shapes, tried in order.  Each name is reported in the hit's
#: check detail so an evaluator can slice precision by format class.
#:
#: ``utr_rtgs_22`` is structurally distinctive enough to stand alone; the other
#: two are not (a bare 12-digit run is also a plausible account number), so they
#: additionally require a reference label near the token.
REFERENCE_FORMATS: Final[tuple[tuple[str, re.Pattern[str], bool], ...]] = (
    ("utr_rtgs_22", re.compile(r"^[A-Z]{4}[A-Z]\d{17}$"), False),
    ("utr_neft_16", re.compile(r"^[A-Z]{4}[A-Z0-9]{12}$"), True),
    ("rrn_12", re.compile(r"^\d{12}$"), True),
)

#: Indian bank account numbers run 9 to 18 digits depending on the core banking
#: platform.  The class is reported, never used to reject inside the range.
ACCOUNT_LENGTH_CLASSES: Final[tuple[tuple[str, int, int], ...]] = (
    ("legacy_9_11", 9, 11),
    ("standard_12_14", 12, 14),
    ("extended_15_18", 15, 18),
)

#: Payment-card PAN lengths that carry a Luhn check digit (ISO/IEC 7812).
CARD_PAN_LENGTHS: Final[tuple[int, int]] = (13, 19)

#: What ``validator_passed=True`` does and does not promise for an account
#: number.  Quoted into ``docs/API.md`` so a consumer cannot mistake structural
#: plausibility for arithmetic verification.
ACCOUNT_CHECKSUM_POLICY: Final[str] = (
    "India has no universal bank-account check digit: the scheme is per-bank "
    "and, for most public-sector banks, unpublished. This validator therefore "
    "performs STRUCTURAL validation only - digits after separator stripping, "
    "and a length inside the 9-18 range Indian core banking platforms use. "
    "validator_passed=True means 'structurally plausible', never 'arithmetically "
    "verified'; the hit carries no 'checksum' CheckOutcome, and that absence is "
    "the machine-readable form of this sentence. When a bank-specific algorithm "
    "is supplied, add it as a further CheckOutcome - a failure then flags the "
    "field and never rewrites the digits (PLAN.md invariant 9)."
)

#: Multipliers understood after a number.  Deliberately excludes bare ``cr`` and
#: ``k``: both collide with ordinary text on a complaint form.
AMOUNT_MULTIPLIERS: Final[Mapping[str, int]] = {
    "thousand": 1_000,
    "हजार": 1_000,
    "हज़ार": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "लाख": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "करोड": 10_000_000,
    "करोड़": 10_000_000,
}

_CURRENCY_PATTERN: Final[str] = r"₹|Rs\.?|INR|रु\.?|रुपये|रुपए"
_MULTIPLIER_PATTERN: Final[str] = "|".join(
    sorted(AMOUNT_MULTIPLIERS, key=len, reverse=True)
)

_AMOUNT_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?P<cur>{_CURRENCY_PATTERN})?\s*"
    r"(?P<num>\d(?:[\d,]*\d)?(?:\.\d{1,2})?)"
    rf"\s*(?P<mult>{_MULTIPLIER_PATTERN})?",
    re.IGNORECASE,
)

#: Western (``100,000``) and Indian (``1,00,000``) grouping, plus ungrouped.
_GROUPING_WESTERN: Final[re.Pattern[str]] = re.compile(r"^\d{1,3}(?:,\d{3})*$")
_GROUPING_INDIAN: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}(?:,\d{2})*,\d{3}$")

#: Beyond this many integer digits a "number" is an identifier, not an amount.
_AMOUNT_MAX_INTEGER_DIGITS: Final[int] = 15

_DATE_NUMERIC_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d])(\d{1,2})([/\-.])(\d{1,2})\2(\d{2,4})(?![\d])"
)
_DATE_DMY_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})(?![\d])",
    re.IGNORECASE,
)
_DATE_MDY_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?![\d])",
    re.IGNORECASE,
)

_MONTH_NAMES: Final[Mapping[str, int]] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

#: Two-digit years below this pivot are 20xx, at or above it 19xx.  A complaint
#: about a 1970s transaction is not a case this service will ever see; a typo'd
#: ``26`` meaning 2026 is one it will see constantly.
_YEAR_PIVOT: Final[int] = 70

_IFSC_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{4}[0-9][A-Za-z0-9]{6}(?![A-Za-z0-9])"
)
_ALNUM_11_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]{11}(?![A-Za-z0-9])"
)
_DIGIT_RUN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d])\d[\d \-]{7,24}\d(?![\d])"
)
_MICR_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\d])\d{9}(?![\d])")
_REFERENCE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z0-9]{12,22}(?![A-Za-z0-9])"
)
_MASKED_PAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w])(?:[Xx\*•·]{4}[ \-]?){2,3}\d{4}(?![\d])"
)
_FULL_PAN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\d])\d(?:[\d \-]{11,22})\d(?![\d])"
)
_LAST4_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\d])\d{4}(?![\d])")

# --------------------------------------------------------------------------- #
# Label cues
# --------------------------------------------------------------------------- #

#: A cue is matched case-insensitively against the *label context* of a line:
#: the line's own text plus the line immediately before it in reading order.
#: Indian complaint forms are two-column (label left, value right) and the
#: reading-order sort in ``app.providers.local`` therefore emits label then
#: value in the same row, which is exactly what makes the previous line the
#: right amount of context.
ACCOUNT_CUES: Final[tuple[str, ...]] = (
    "account number", "account no", "account #", "a/c no", "a/c number",
    "a/c", "acct", "account", "खाता संख्या", "खाता क्रमांक", "खाते क्रमांक",
    "खाता", "खाते",
)
CARD_CUES: Final[tuple[str, ...]] = (
    "card", "debit card", "credit card", "कार्ड",
)
REFERENCE_CUES: Final[tuple[str, ...]] = (
    "utr", "rrn", "reference", "ref no", "ref.", "txn id", "transaction id",
    "transaction reference", "संदर्भ", "यूटीआर", "लेनदेन संदर्भ",
)
MICR_CUES: Final[tuple[str, ...]] = ("micr", "एमआईसीआर")
CURRENCY_CUES: Final[tuple[str, ...]] = (
    "₹", "rs.", "rs ", "rs)", "inr", "amount", "रु", "राशि", "रक्कम", "रुपये",
)
COMPLAINT_DATE_CUES: Final[tuple[str, ...]] = (
    "date of complaint", "complaint date", "dated", "शिकायत की तारीख",
    "शिकायत दिनांक", "तक्रार दिनांक",
)
TRANSACTION_DATE_CUES: Final[tuple[str, ...]] = (
    "date of transaction", "transaction date", "txn date", "value date",
    "लेनदेन दिनांक", "व्यवहार दिनांक", "लेनदेन की तारीख",
)


def _has_cue(context: str, cues: Sequence[str]) -> bool:
    """Case-insensitive substring test over the label context."""
    folded = context.casefold()
    return any(cue.casefold() in folded for cue in cues)


# --------------------------------------------------------------------------- #
# Page scan — the immutable input every validator sees
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageScan:
    """One page's lines, pre-normalised once for every validator to share.

    ``ascii_texts`` is :func:`normalise_digits` applied per line; because that
    mapping is 1:1, a span found there indexes ``lines[i].text`` unchanged.
    """

    page_no: int
    lines: tuple[OCRLine, ...]
    ascii_texts: tuple[str, ...]
    contexts: tuple[str, ...]
    master: IFSCMaster
    banks: tuple[BankRecord, ...]
    #: Institutions that are never the bank complained about — see
    #: :func:`non_party_institution_names`.  Travels with the table, not as a
    #: module global, so a replacement ``BANK_ALIASES_PATH`` brings its own.
    non_party: tuple[str, ...] = ()

    def evidence(self, position: int, start: int, end: int) -> Evidence:
        """Evidence for a span inside the line at list ``position``."""
        line = self.lines[position]
        bbox: BBox = [float(value) for value in line.bbox]
        return Evidence(
            page_no=self.page_no,
            bbox=bbox,
            span=TextSpan(line_index=line.line_index, start=start, end=end),
        )


def _prepare(
    page: OCRPage,
    master: IFSCMaster,
    banks: Sequence[BankRecord],
    non_party: Sequence[str],
) -> PageScan:
    lines = tuple(page.lines)
    ascii_texts = tuple(normalise_digits(line.text) for line in lines)
    contexts: list[str] = []
    for index, text in enumerate(ascii_texts):
        previous = ascii_texts[index - 1] if index else ""
        contexts.append(f"{previous} {text}".strip())
    return PageScan(
        page_no=page.page_no,
        lines=lines,
        ascii_texts=ascii_texts,
        contexts=tuple(contexts),
        master=master,
        banks=tuple(banks),
        non_party=tuple(non_party),
    )


def _confidence(line: OCRLine, weight: float) -> float:
    return max(0.0, min(1.0, float(line.confidence) * weight))


def _weight_for(checks: Sequence[CheckOutcome], *, unverified: bool) -> float:
    if any(not check.passed for check in checks):
        return FAIL_WEIGHT
    return UNVERIFIED_WEIGHT if unverified else PASS_WEIGHT


# --------------------------------------------------------------------------- #
# Pure parsers (public: hypothesis property-tests them directly)
# --------------------------------------------------------------------------- #


class AmountParse(BaseModel):
    """Outcome of reading one monetary expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Canonical decimal string with two fraction digits, or ``None`` on failure.
    normalized: str | None
    ok: bool
    #: ``grouping``/``range``/``syntax`` — machine-readable failure cause.
    reason: str | None = None
    #: Multiplier applied (``1`` when the expression carried none).
    multiplier: int = 1
    #: ``western``, ``indian`` or ``none``.
    grouping: str = "none"


def _grouping_style(integer_part: str) -> str | None:
    """``western`` / ``indian`` / ``none``, or ``None`` when grouping is broken."""
    if "," not in integer_part:
        return "none" if integer_part.isdigit() else None
    if _GROUPING_WESTERN.match(integer_part):
        return "western"
    if _GROUPING_INDIAN.match(integer_part):
        return "indian"
    return None


def parse_amount(text: str, *, multiplier_word: str | None = None) -> AmountParse:
    """Read a number written in Indian or Western grouping into a decimal.

    Accepts Indic digits (normalised first), optional ``,`` grouping in either
    convention, at most two fraction digits, and an optional lakh/crore word
    passed separately by the caller that already matched it.

    A malformed grouping (``1,0,000``) fails with ``reason="grouping"`` and no
    normalised value.  It is never silently regrouped: a comma in the wrong
    place is exactly the OCR error a reviewer needs to see (invariant 9).
    """
    candidate = normalise_digits(text).strip()
    if not candidate:
        return AmountParse(normalized=None, ok=False, reason="syntax")
    integer_part, _, fraction = candidate.partition(".")
    if fraction and (not fraction.isdigit() or len(fraction) > 2):
        return AmountParse(normalized=None, ok=False, reason="syntax")
    grouping = _grouping_style(integer_part)
    if grouping is None:
        reason = "grouping" if "," in integer_part else "syntax"
        return AmountParse(normalized=None, ok=False, reason=reason)
    digits = integer_part.replace(",", "")
    if not digits or len(digits) > _AMOUNT_MAX_INTEGER_DIGITS:
        return AmountParse(
            normalized=None, ok=False, reason="range", grouping=grouping
        )
    multiplier = 1
    if multiplier_word:
        multiplier = AMOUNT_MULTIPLIERS.get(multiplier_word.strip().lower(), 0)
        if multiplier == 0:
            return AmountParse(
                normalized=None, ok=False, reason="syntax", grouping=grouping
            )
    try:
        value = Decimal(f"{digits}.{fraction}" if fraction else digits) * multiplier
    except InvalidOperation:  # pragma: no cover - guarded by the regex above
        return AmountParse(
            normalized=None, ok=False, reason="syntax", grouping=grouping
        )
    quantised = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return AmountParse(
        normalized=f"{quantised:f}",
        ok=True,
        multiplier=multiplier,
        grouping=grouping,
    )


def format_indian_grouping(value: Decimal | int | str) -> str:
    """Render a non-negative integer amount in Indian grouping (``12,34,567``).

    Exists so the round-trip property test has a generator that produces the
    convention the parser must accept, rather than asserting against a second
    copy of the parser's own logic.

    Raises:
        ValueError: for a negative value.
    """
    number = int(Decimal(str(value)))
    if number < 0:
        raise ValueError(f"Indian grouping is defined for non-negative values, got {number}")
    digits = str(number)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join([*groups, tail])


class DateParse(BaseModel):
    """Outcome of reading one date expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ISO 8601 ``YYYY-MM-DD``, or ``None`` when the expression is not a date.
    iso: str | None
    ok: bool
    #: ``calendar``/``month``/``syntax`` — machine-readable failure cause.
    reason: str | None = None
    #: ``dmy``, ``dmy_text`` or ``mdy_text`` — which reading produced ``iso``.
    order: str | None = None


def _resolve_year(raw: str) -> int:
    year = int(raw)
    if len(raw) <= 2:
        return 1900 + year if year >= _YEAR_PIVOT else 2000 + year
    return year


def _iso_or_failure(day: int, month: int, year: int, order: str) -> DateParse:
    try:
        return DateParse(
            iso=date(year, month, day).isoformat(), ok=True, order=order
        )
    except ValueError:
        return DateParse(iso=None, ok=False, reason="calendar", order=order)


def parse_indian_date(text: str) -> DateParse:
    """Read one date token into ISO 8601, assuming day-first ordering.

    Handles ``DD/MM/YYYY``, ``DD-MM-YY``, ``DD.MM.YYYY``, ``14 January 2026``
    and ``January 14, 2026``, with Indic digits normalised first.

    **Day-first is assumed and never second-guessed.**  ``06/13/2026`` therefore
    fails with ``reason="calendar"`` rather than being re-read as 13 June:
    swapping the fields would be a silent correction of a value a human can
    check against the crop in two seconds (invariant 9).
    """
    candidate = normalise_digits(text).strip()
    numeric = _DATE_NUMERIC_RE.fullmatch(candidate)
    if numeric is not None:
        return _iso_or_failure(
            int(numeric.group(1)),
            int(numeric.group(3)),
            _resolve_year(numeric.group(4)),
            "dmy",
        )
    dmy = _DATE_DMY_TEXT_RE.fullmatch(candidate)
    if dmy is not None:
        month = _MONTH_NAMES.get(dmy.group(2).lower())
        if month is None:
            return DateParse(iso=None, ok=False, reason="month", order="dmy_text")
        return _iso_or_failure(
            int(dmy.group(1)), month, int(dmy.group(3)), "dmy_text"
        )
    mdy = _DATE_MDY_TEXT_RE.fullmatch(candidate)
    if mdy is not None:
        month = _MONTH_NAMES.get(mdy.group(1).lower())
        if month is None:
            return DateParse(iso=None, ok=False, reason="month", order="mdy_text")
        return _iso_or_failure(
            int(mdy.group(2)), month, int(mdy.group(3)), "mdy_text"
        )
    return DateParse(iso=None, ok=False, reason="syntax")


def account_length_class(digits: str) -> str | None:
    """Name the length band of an account number, or ``None`` when out of range."""
    length = len(digits)
    for name, low, high in ACCOUNT_LENGTH_CLASSES:
        if low <= length <= high:
            return name
    return None


def classify_reference(token: str) -> str | None:
    """Name the UTR/RRN format class of ``token``, or ``None``.

    Classes are structural only.  The service does not attempt to decode the
    embedded bank code, julian day or sequence: those layouts are conventions
    rather than a published standard, and reading them wrong would produce a
    confident falsehood (BUILD-BRIEF sec. 0.3).
    """
    candidate = normalise_digits(token).strip().upper()
    for name, pattern, _needs_cue in REFERENCE_FORMATS:
        if pattern.match(candidate):
            return name
    return None


# --------------------------------------------------------------------------- #
# The validators
# --------------------------------------------------------------------------- #


def validate_ifsc(scan: PageScan) -> list[ValidatorHit]:
    """IFSC: structure (``^[A-Z]{4}0[A-Z0-9]{6}$``) then master-list lookup.

    Ladder, in order:

    ==============================  ==================  ======================
    Situation                       validator_passed    flags
    ==============================  ==================  ======================
    structure fails, IFSC label     False               VALIDATOR_FAILED
    exact branch row found          True                --
    prefix known, branch unknown    True                -- (non-authoritative)
    prefix unknown                  True/False          IFSC_NOT_IN_MASTER
    ==============================  ==================  ======================

    The last row is where :attr:`IFSCMaster.authoritative` decides: against the
    real RBI list a code no branch owns is wrong; against the committed
    five-bank fixture it is merely unverifiable, and failing it would flag every
    document in the corpus.
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        line = scan.lines[position]
        seen: set[tuple[int, int]] = set()
        for match in _IFSC_TOKEN_RE.finditer(text):
            token = match.group(0).upper()
            if not IFSC_STRUCTURE.match(token):
                continue
            seen.add(match.span())
            hits.append(
                _ifsc_hit(scan, position, line, match.start(), match.end(), token)
            )
        if not _has_cue(scan.contexts[position], ("ifsc",)):
            continue
        for match in _ALNUM_11_TOKEN_RE.finditer(text):
            if match.span() in seen:
                continue
            token = match.group(0).upper()
            if IFSC_STRUCTURE.match(token) or not any(c.isalpha() for c in token):
                continue
            checks = (
                CheckOutcome(
                    name="structure",
                    passed=False,
                    detail="does not match ^[A-Z]{4}0[A-Z0-9]{6}$",
                ),
            )
            hits.append(
                ValidatorHit(
                    field_name="ifsc_code",
                    value=line.text[match.start() : match.end()],
                    normalized=None,
                    confidence=_confidence(line, FAIL_WEIGHT),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=checks,
                    detail="token sits under an IFSC label but is malformed",
                )
            )
    return hits


def _ifsc_hit(
    scan: PageScan,
    position: int,
    line: OCRLine,
    start: int,
    end: int,
    token: str,
) -> ValidatorHit:
    checks: list[CheckOutcome] = [CheckOutcome(name="structure", passed=True)]
    flags: list[FieldFlag] = []
    detail: str | None = None
    unverified = False
    record = scan.master.lookup(token)
    if record is not None:
        checks.append(
            CheckOutcome(name="master_lookup", passed=True, detail="branch row found")
        )
    elif scan.master.knows_prefix(token[:4]):
        unverified = True
        detail = (
            f"bank prefix {token[:4]} is known; branch row absent from a "
            f"{'complete' if scan.master.authoritative else 'partial'} master"
        )
        if scan.master.authoritative:
            checks.append(
                CheckOutcome(
                    name="master_lookup", passed=False, detail="no such branch"
                )
            )
            flags.append(FieldFlag.IFSC_NOT_IN_MASTER)
    else:
        unverified = True
        flags.append(FieldFlag.IFSC_NOT_IN_MASTER)
        detail = f"bank prefix {token[:4]} is not in the master list"
        if scan.master.authoritative:
            checks.append(
                CheckOutcome(
                    name="master_lookup", passed=False, detail="unknown bank prefix"
                )
            )
    passed = all(check.passed for check in checks)
    return ValidatorHit(
        field_name="ifsc_code",
        value=line.text[start:end],
        normalized=token if passed else None,
        confidence=_confidence(line, _weight_for(checks, unverified=unverified)),
        evidence=scan.evidence(position, start, end),
        checks=tuple(checks),
        flags=tuple(flags),
        detail=detail,
    )


def validate_account_number(scan: PageScan) -> list[ValidatorHit]:
    """Bank account number: digit run of 9-18 digits under an account label.

    Label-gated on purpose.  A bare eleven-digit run is equally consistent with
    an account number, a UTR and a customer id; the label is the only signal
    that distinguishes them, and guessing would produce exactly the confident
    wrong answer BUILD-BRIEF sec. 0.5 forbids.

    See :data:`ACCOUNT_CHECKSUM_POLICY` for what ``validator_passed`` promises.
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        if not _has_cue(scan.contexts[position], ACCOUNT_CUES):
            continue
        line = scan.lines[position]
        for match in _DIGIT_RUN_RE.finditer(text):
            raw = match.group(0)
            digits = re.sub(r"[ \-]", "", raw)
            if not digits.isdigit():  # pragma: no cover - regex guarantees it
                continue
            length_class = account_length_class(digits)
            checks = (
                CheckOutcome(
                    name="digits",
                    passed=True,
                    detail=f"{len(digits)} digits after separator stripping",
                ),
                CheckOutcome(
                    name="length_class",
                    passed=length_class is not None,
                    detail=length_class or "outside the 9-18 digit range",
                ),
            )
            passed = all(check.passed for check in checks)
            hits.append(
                ValidatorHit(
                    field_name="account_number",
                    value=line.text[match.start() : match.end()],
                    normalized=digits if passed else None,
                    confidence=_confidence(
                        line, _weight_for(checks, unverified=True)
                    ),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=checks,
                    detail=ACCOUNT_CHECKSUM_POLICY,
                )
            )
    return hits


def validate_card_fragment(scan: PageScan) -> list[ValidatorHit]:
    """Payment-card fragment: masked PAN, full PAN (Luhn), or a labelled last-4.

    Luhn runs **only** when the visible digits are a complete 13-19 digit PAN.
    A masked ``XXXX XXXX XXXX 1469`` or a bare labelled ``1469`` carries no
    check digit to test, so the hit records ``structure`` and no ``checksum``
    outcome — the same honest-absence convention the account validator uses.

    A full PAN whose Luhn digit is wrong comes back ``validator_passed=False``
    with ``CHECKSUM_FAILED``, ``normalized=None`` and the original characters
    untouched (invariant 9).
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        line = scan.lines[position]
        has_card_cue = _has_cue(scan.contexts[position], CARD_CUES)
        claimed: list[tuple[int, int]] = []

        for match in _MASKED_PAN_RE.finditer(text):
            claimed.append(match.span())
            last_four = match.group(0)[-4:]
            hits.append(
                ValidatorHit(
                    field_name="card_fragment",
                    value=line.text[match.start() : match.end()],
                    normalized=last_four,
                    confidence=_confidence(line, UNVERIFIED_WEIGHT),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=(
                        CheckOutcome(
                            name="structure",
                            passed=True,
                            detail="masked PAN, last four visible",
                        ),
                    ),
                    detail=(
                        "no checksum ran: Luhn needs the full PAN and only the "
                        "last four digits are visible"
                    ),
                )
            )

        for match in _FULL_PAN_RE.finditer(text):
            digits = re.sub(r"[ \-]", "", match.group(0))
            if not (CARD_PAN_LENGTHS[0] <= len(digits) <= CARD_PAN_LENGTHS[1]):
                continue
            if not has_card_cue:
                continue
            if any(
                start <= match.start() < end for start, end in claimed
            ):  # pragma: no cover - masked and full PAN cannot overlap
                continue
            claimed.append(match.span())
            passed = luhn_check(digits)
            checks = (
                CheckOutcome(
                    name="structure", passed=True, detail=f"{len(digits)}-digit PAN"
                ),
                CheckOutcome(name="checksum", passed=passed, detail="Luhn mod 10"),
            )
            hits.append(
                ValidatorHit(
                    field_name="card_fragment",
                    value=line.text[match.start() : match.end()],
                    normalized=digits[-4:] if passed else None,
                    confidence=_confidence(
                        line, _weight_for(checks, unverified=False)
                    ),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=checks,
                    flags=() if passed else (FieldFlag.CHECKSUM_FAILED,),
                    detail=None if passed else "Luhn check digit does not agree",
                )
            )

        if not has_card_cue:
            continue
        for match in _LAST4_RE.finditer(text):
            if any(start <= match.start() < end for start, end in claimed):
                continue
            hits.append(
                ValidatorHit(
                    field_name="card_fragment",
                    value=line.text[match.start() : match.end()],
                    normalized=match.group(0),
                    confidence=_confidence(line, UNVERIFIED_WEIGHT),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=(
                        CheckOutcome(
                            name="structure",
                            passed=True,
                            detail="labelled last-four fragment",
                        ),
                    ),
                    detail="no checksum ran: four digits carry no Luhn digit",
                )
            )
    return hits


def validate_utr_rrn(scan: PageScan) -> list[ValidatorHit]:
    """Transaction reference: UTR (RTGS/NEFT) or RRN, by format class.

    ``utr_rtgs_22`` is distinctive enough to be recognised anywhere on the page.
    ``utr_neft_16`` and ``rrn_12`` additionally require a reference label,
    because a bare 12-digit run is also a perfectly good account number.

    A labelled token that matches no class fails loudly rather than being
    dropped: a reference the OCR mangled is a review item, and dropping it would
    make the field silently absent instead.
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        line = scan.lines[position]
        has_cue = _has_cue(scan.contexts[position], REFERENCE_CUES)
        for match in _REFERENCE_TOKEN_RE.finditer(text):
            token = match.group(0).upper()
            format_class = classify_reference(token)
            if format_class is not None:
                needs_cue = next(
                    needs for name, _p, needs in REFERENCE_FORMATS
                    if name == format_class
                )
                if needs_cue and not has_cue:
                    continue
                checks = (
                    CheckOutcome(
                        name="format_class", passed=True, detail=format_class
                    ),
                )
                hits.append(
                    ValidatorHit(
                        field_name="utr_rrn",
                        value=line.text[match.start() : match.end()],
                        normalized=token,
                        confidence=_confidence(line, PASS_WEIGHT),
                        evidence=scan.evidence(
                            position, match.start(), match.end()
                        ),
                        checks=checks,
                        detail=f"format class {format_class}",
                    )
                )
                continue
            if not has_cue:
                continue
            hits.append(
                ValidatorHit(
                    field_name="utr_rrn",
                    value=line.text[match.start() : match.end()],
                    normalized=None,
                    confidence=_confidence(line, FAIL_WEIGHT),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=(
                        CheckOutcome(
                            name="format_class",
                            passed=False,
                            detail="matches no known UTR/RRN class",
                        ),
                    ),
                    detail="token sits under a reference label but fits no class",
                )
            )
    return hits


def validate_micr(scan: PageScan) -> list[ValidatorHit]:
    """MICR code: nine digits (3 city + 3 bank + 3 branch), label-gated.

    The Indian MICR scheme has no check digit, so structure is the entire test
    and the hit carries no ``checksum`` outcome.  Cheque MICR bands printed in
    E-13B symbols are **not** handled: recognising them needs a dedicated
    recogniser, and pretending the digits alone are the band would silently drop
    the transaction-code field.
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        if not _has_cue(scan.contexts[position], MICR_CUES):
            continue
        line = scan.lines[position]
        for match in _MICR_TOKEN_RE.finditer(text):
            token = match.group(0)
            hits.append(
                ValidatorHit(
                    field_name="micr_code",
                    value=line.text[match.start() : match.end()],
                    normalized=token,
                    confidence=_confidence(line, PASS_WEIGHT),
                    evidence=scan.evidence(position, match.start(), match.end()),
                    checks=(
                        CheckOutcome(
                            name="structure",
                            passed=bool(MICR_STRUCTURE.match(token)),
                            detail="9 digits: city(3) bank(3) branch(3)",
                        ),
                    ),
                    detail="the Indian MICR scheme defines no check digit",
                )
            )
    return hits


def validate_amount(scan: PageScan) -> list[ValidatorHit]:
    """Monetary amounts: currency marks, lakh/crore words, Indian grouping.

    A number is read as an amount when **any** of these holds:

    * it carries a currency marker (``₹``, ``Rs.``, ``INR``, ``रु.``);
    * it carries a lakh/crore/thousand multiplier word;
    * it is grouped (``4,500``) or has a fraction, and the label context names
      money;
    * it is the entire line and the label context names money — the two-column
      form case, where ``₹`` sits on the label and the cell holds bare digits.

    A bare ungrouped run with no currency context is *not* an amount, which is
    what keeps an eleven-digit account number out of this field.
    """
    hits: list[ValidatorHit] = []
    for position, text in enumerate(scan.ascii_texts):
        line = scan.lines[position]
        has_currency_cue = _has_cue(scan.contexts[position], CURRENCY_CUES)
        stripped_line = text.strip()
        for match in _AMOUNT_RE.finditer(text):
            number = match.group("num")
            if not number:  # pragma: no cover - the group is mandatory
                continue
            num_start = match.start("num")
            num_end = match.end("num")
            before = text[num_start - 1] if num_start else ""
            after = text[num_end] if num_end < len(text) else ""
            if before in {"/", "-", ":"} or after in {"/", "-", ":"}:
                continue  # a date or a time component, not an amount
            currency = match.group("cur")
            multiplier_word = match.group("mult")
            grouped = "," in number or "." in number
            standalone = match.group(0).strip() == stripped_line
            if not (
                currency
                or multiplier_word
                or (grouped and has_currency_cue)
                or (standalone and has_currency_cue)
            ):
                continue
            parsed = parse_amount(number, multiplier_word=multiplier_word)
            start, end = _trimmed_span(text, match.start(), match.end())
            checks = (
                CheckOutcome(
                    name="grouping",
                    passed=parsed.reason != "grouping",
                    detail=parsed.grouping,
                ),
                CheckOutcome(
                    name="magnitude",
                    passed=parsed.reason != "range",
                    detail=f"<= {_AMOUNT_MAX_INTEGER_DIGITS} integer digits",
                ),
                CheckOutcome(
                    name="syntax",
                    passed=parsed.reason != "syntax",
                    detail="digits, at most two fraction places",
                ),
            )
            hits.append(
                ValidatorHit(
                    field_name="amount",
                    value=line.text[start:end],
                    normalized=parsed.normalized if parsed.ok else None,
                    confidence=_confidence(
                        line, _weight_for(checks, unverified=False)
                    ),
                    evidence=scan.evidence(position, start, end),
                    checks=checks,
                    detail=(
                        f"multiplier x{parsed.multiplier}"
                        if parsed.ok and parsed.multiplier != 1
                        else None
                    ),
                )
            )
    return hits


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` past leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def validate_date(scan: PageScan) -> list[ValidatorHit]:
    """Dates in Indian notation, normalised to ISO 8601.

    The field a date lands in — ``transaction_date`` or ``complaint_date`` — is
    chosen from the label context.  When no cue matches, the hit goes to
    ``transaction_date`` (the overwhelmingly common case on a complaint form)
    and says so in ``detail``; the role is a *labelled inference*, never a
    silent one, and a reviewer sees the note beside the crop.
    """
    hits: list[ValidatorHit] = []
    patterns = (_DATE_NUMERIC_RE, _DATE_DMY_TEXT_RE, _DATE_MDY_TEXT_RE)
    for position, text in enumerate(scan.ascii_texts):
        line = scan.lines[position]
        context = scan.contexts[position]
        field_name, role_detail = _date_role(context)
        claimed: list[tuple[int, int]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                if any(
                    start < match.end() and match.start() < end
                    for start, end in claimed
                ):
                    continue
                claimed.append(match.span())
                parsed = parse_indian_date(match.group(0))
                checks = (
                    CheckOutcome(
                        name="calendar",
                        passed=parsed.ok,
                        detail=parsed.order or parsed.reason,
                    ),
                )
                hits.append(
                    ValidatorHit(
                        field_name=field_name,
                        value=line.text[match.start() : match.end()],
                        normalized=parsed.iso,
                        confidence=_confidence(
                            line, _weight_for(checks, unverified=False)
                        ),
                        evidence=scan.evidence(
                            position, match.start(), match.end()
                        ),
                        checks=checks,
                        detail=role_detail,
                    )
                )
    return hits


def _date_role(context: str) -> tuple[str, str | None]:
    if _has_cue(context, COMPLAINT_DATE_CUES):
        return "complaint_date", "role from label cue"
    if _has_cue(context, TRANSACTION_DATE_CUES):
        return "transaction_date", "role from label cue"
    return (
        "transaction_date",
        "role inferred: no date label cue matched; defaulted to transaction_date",
    )


#: Fuzzy score a candidate line must reach before it is considered a bank name.
#: Calibrated against the committed corpus: "Reserve Bank of India" (the
#: addressee of every complaint letter) scores 85.5 against "Union Bank", so the
#: floor sits above it.  ``docs/TUNING.md`` owns re-calibration.
BANK_NAME_SCORE_FLOOR: Final[float] = 88.0
#: How far the winner must beat the runner-up.  Two banks within this margin is
#: an ambiguity, not a decision.
BANK_NAME_MARGIN: Final[float] = 4.0
#: Aliases shorter than this are matched **exactly**, as standalone tokens, and
#: never fuzzily.  ``WRatio("SBIN0904158", "SBI")`` is 90 — an acronym is a
#: substring of half the identifiers on the page, and fuzzy-matching it turns an
#: IFSC code into a bank-name hit.  Observed, not theorised.
_MIN_FUZZY_ALIAS_LENGTH: Final[int] = 5
#: The aligned substring must cover at least this fraction of the alias, and be
#: at least :data:`_MIN_ALIAS_SPAN` characters.  Without it the four-character
#: label cell "Bank" on a transaction receipt scores 90 against "Bank of
#: Baroda" — ``WRatio`` is partial-ratio driven and a short needle inside a
#: short haystack is not evidence of anything.  Also observed on the corpus.
_MIN_ALIAS_COVERAGE: Final[float] = 0.6
_MIN_ALIAS_SPAN: Final[int] = 5


def validate_bank_name(scan: PageScan) -> list[ValidatorHit]:
    """Bank name by exact-then-fuzzy match, with an IFSC bank-code cross-check.

    BUILD-BRIEF M5: a score floor **and** agreement with the IFSC prefix.  Three
    mechanisms, and none overrides another —

    * **exact alias containment wins outright, longest alias first.**  A table
      that covers the real banking system necessarily contains names that nest
      inside one another (``Bank of India`` inside ``State Bank of India``,
      ``Union Bank`` inside ``City Union Bank``).  Fuzzy scoring cannot separate
      those — ``WRatio`` is partial-ratio driven, so a contained name scores ~100
      against its container — but *length* separates them exactly, because the
      longer literal is the more specific claim.  Without this rule a full table
      is strictly more dangerous than a small one;
    * otherwise ``rapidfuzz.fuzz.WRatio`` against every alias, floor
      :data:`BANK_NAME_SCORE_FLOOR`, with a :data:`BANK_NAME_MARGIN` gap over the
      runner-up.  This is the path OCR noise takes: a dropped anusvara leaves
      ``बैक ऑफ इडिया``, which is exact-matchable against nothing;
    * agreement with the four-letter prefix of any IFSC found on the same
      document.  Disagreement sets ``validator_passed=False`` and
      ``BANK_NAME_IFSC_DISAGREEMENT``.  It **never** replaces the OCR'd name
      with the master's name for that prefix, and never discards the IFSC: the
      design rule is "disagreement flags rather than picks".

    **An ambiguous match is a failure, not a footnote.**  Two banks within the
    margin (or two equally-long exact matches) yields ``validator_passed=False``,
    ``normalized=None`` and ``MULTIPLE_CANDIDATES``.  It previously published the
    winner with the flag attached and ``validator_passed=True``, which meant
    :func:`app.pipeline._route_pages` — which counts only failed hits — never
    escalated it, so nothing and nobody adjudicated the tie.  A named bank that
    is the wrong bank is worse than no bank at all; this is the same "flag it,
    never correct it" rule the contract enforces everywhere else (invariant 9).

    The cross-check itself runs in :func:`run_validators`, after every page has
    been scanned, because an IFSC on page 2 must still contradict a bank name on
    page 1.  ``value`` is the matched substring of the line, so a bank name
    inside running prose does not drag the whole sentence into the field.
    """
    hits: list[ValidatorHit] = []
    if not scan.banks:
        return hits
    aliases: list[tuple[str, BankRecord]] = [
        (alias, bank) for bank in scan.banks for alias in bank.aliases
    ]
    for position, line in enumerate(scan.lines):
        text = line.text
        if not text.strip():
            continue
        candidate = _bank_candidate(text, aliases, scan.non_party)
        if candidate is None:
            continue
        hits.append(
            _bank_hit(candidate, scan=scan, line=line, position=position)
        )
    return hits


@dataclass(frozen=True)
class _BankCandidate:
    """The bank one line claims, and how strongly."""

    bank: BankRecord
    alias: str
    start: int
    end: int
    ambiguous: bool
    #: Name of the check that decided this match, for the audit trail.
    check_name: str
    check_detail: str
    #: Populated only when ``ambiguous``; names the rival that forced the tie.
    rival_detail: str | None = None


def _bank_candidate(
    text: str,
    aliases: Sequence[tuple[str, BankRecord]],
    non_party: Sequence[str] = (),
) -> _BankCandidate | None:
    """Best bank claim on one line, or ``None`` when nothing clears the bar.

    Spans covered by a ``non_party`` match are removed before anything is
    scored.  This is not a nicety: ``Bank of India`` is a literal substring of
    ``Reserve Bank of India``, which is the addressee of every RB-IOS complaint,
    so without the mask the address block of *every* letter would yield a
    confident and wrong ``bank_name``.  The longest literal still wins — the
    regulator's name is simply longer than the bank's.
    """
    blocked = _non_party_spans(text, non_party)
    exact = sorted(
        (
            (len(alias), alias, bank, span)
            for alias, bank in aliases
            if (span := _exact_alias_span(text, alias)) is not None
            and not _is_covered(span, blocked)
        ),
        key=lambda item: (-item[0], item[2].code, item[1]),
    )
    if exact:
        longest, alias, bank, (start, end) = exact[0]
        rivals = {
            entry[2].code for entry in exact if entry[0] == longest
        } - {bank.code}
        return _BankCandidate(
            bank=bank,
            alias=alias,
            start=start,
            end=end,
            ambiguous=bool(rivals),
            check_name="exact_match",
            check_detail=f"alias {alias!r} appears verbatim",
            rival_detail=(
                f"{sorted(rivals)} match an alias of the same length ({longest})"
                if rivals
                else None
            ),
        )

    scored = sorted(
        ((_alias_score(text, alias), alias, bank) for alias, bank in aliases),
        key=lambda item: (-item[0], item[2].code, item[1]),
    )
    best_score, best_alias, best_bank = scored[0]
    if best_score < BANK_NAME_SCORE_FLOOR:
        return None
    rival = next(
        (entry for entry in scored[1:] if entry[2].code != best_bank.code), None
    )
    ambiguous = rival is not None and (best_score - rival[0]) < BANK_NAME_MARGIN
    start, end = _alias_span(text, best_alias)
    matched = end - start
    if matched < _MIN_ALIAS_SPAN or matched < _MIN_ALIAS_COVERAGE * len(best_alias):
        return None
    if _is_covered((start, end), blocked):
        return None
    return _BankCandidate(
        bank=best_bank,
        alias=best_alias,
        start=start,
        end=end,
        ambiguous=ambiguous,
        check_name="fuzzy_score",
        check_detail=f"WRatio {best_score:.1f} >= {BANK_NAME_SCORE_FLOOR}",
        rival_detail=(
            f"{rival[2].code} scores {rival[0]:.1f}, within {BANK_NAME_MARGIN} "
            f"of {best_score:.1f}"
            if ambiguous and rival is not None
            else None
        ),
    )


def _bank_hit(
    candidate: _BankCandidate, *, scan: PageScan, line: OCRLine, position: int
) -> ValidatorHit:
    """Promote a bank claim to a hit, failing it when it is ambiguous."""
    checks = [
        CheckOutcome(
            name=candidate.check_name, passed=True, detail=candidate.check_detail
        )
    ]
    if candidate.ambiguous:
        checks.append(
            CheckOutcome(
                name="unambiguous_match",
                passed=False,
                detail=(
                    f"{candidate.rival_detail}; the tie is not broken here, so "
                    "no bank is claimed"
                ),
            )
        )
    detail = f"matched alias {candidate.alias!r} (bank code {candidate.bank.code})"
    if candidate.ambiguous:
        detail += "; ambiguous, candidate shown for review but not claimed"
    return ValidatorHit(
        field_name="bank_name",
        value=line.text[candidate.start : candidate.end],
        normalized=None if candidate.ambiguous else candidate.bank.name,
        confidence=_confidence(
            line, FAIL_WEIGHT if candidate.ambiguous else PASS_WEIGHT
        ),
        source="lookup_table",
        evidence=scan.evidence(position, candidate.start, candidate.end),
        checks=tuple(checks),
        flags=(FieldFlag.MULTIPLE_CANDIDATES,) if candidate.ambiguous else (),
        detail=detail,
    )


def _non_party_spans(
    text: str, non_party: Sequence[str]
) -> tuple[tuple[int, int], ...]:
    """Every span on the line occupied by an institution that is never a party.

    Longest first, so a caller comparing lengths sees the most specific name.
    """
    spans = [
        span
        for name in non_party
        if (span := _exact_alias_span(text, name)) is not None
    ]
    return tuple(sorted(spans, key=lambda s: (s[0] - s[1], s[0])))


def _is_covered(span: tuple[int, int], blocked: Sequence[tuple[int, int]]) -> bool:
    """Whether ``span`` falls inside any blocked span.

    Containment, not overlap: a bank named *beside* the regulator on the same
    line ("complaint to the Reserve Bank of India against Canara Bank") keeps
    its own hit, because its span sits outside the regulator's.
    """
    start, end = span
    return any(lower <= start and end <= upper for lower, upper in blocked)


def _exact_alias_span(text: str, alias: str) -> tuple[int, int] | None:
    """Span of ``alias`` inside ``text`` as a boundary-delimited literal.

    ``None`` when the alias is not present verbatim.  Boundaries use ``\\w``,
    which is Unicode-aware under Python 3, so a Devanagari alias does not match
    inside a longer Devanagari run any more than a Latin one does.
    """
    found = _compile_exact_alias(alias).search(text)
    return (found.start(), found.end()) if found else None


@lru_cache(maxsize=2048)
def _compile_exact_alias(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)


def _short_alias_pattern(alias: str) -> re.Pattern[str]:
    """Standalone-token matcher for a short alias, compiled once per alias."""
    return _compile_short_alias(alias)


@lru_cache(maxsize=256)
def _compile_short_alias(alias: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE
    )


def _alias_score(text: str, alias: str) -> float:
    """Score one alias against one line.

    Long aliases go through ``rapidfuzz.fuzz.WRatio``.  Short ones (acronyms)
    are matched exactly as standalone tokens: fuzzy-matching a three-letter
    acronym against an eleven-character identifier is how ``SBIN0904158``
    becomes a bank name.
    """
    if len(alias) >= _MIN_FUZZY_ALIAS_LENGTH:
        return float(fuzz.WRatio(text, alias))
    return 100.0 if _short_alias_pattern(alias).search(text) else 0.0


def _alias_span(text: str, alias: str) -> tuple[int, int]:
    """Character range of ``alias`` inside ``text``, whitespace trimmed.

    ``partial_ratio_alignment`` is what keeps a bank name buried in running
    prose from dragging the whole sentence into the field's ``value``.
    """
    if len(alias) < _MIN_FUZZY_ALIAS_LENGTH:
        found = _short_alias_pattern(alias).search(text)
        if found is not None:
            return found.span()
        return 0, len(text)  # pragma: no cover - scored 0, never reaches here
    alignment = fuzz.partial_ratio_alignment(text, alias)
    if alignment is None:  # pragma: no cover - only for an empty needle
        return 0, len(text)
    start, end = _trimmed_span(text, alignment.src_start, alignment.src_end)
    return (start, end) if start < end else (0, len(text))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

#: Field name -> validator.  Ordered so the report's tie-breaking is stable and
#: a reader can see the whole deterministic surface in one place.
VALIDATORS: Final[Mapping[str, Callable[[PageScan], list[ValidatorHit]]]] = {
    "ifsc_code": validate_ifsc,
    "bank_name": validate_bank_name,
    "account_number": validate_account_number,
    "card_fragment": validate_card_fragment,
    "utr_rrn": validate_utr_rrn,
    "micr_code": validate_micr,
    "amount": validate_amount,
    "date": validate_date,
}

VALIDATOR_FIELD_NAMES: Final[tuple[str, ...]] = tuple(VALIDATORS)

#: Upper bound on the pool.  These are regex passes over a few kilobytes of
#: text: more threads than this buys nothing and competes with the ONNX pools
#: the thread-cap budget already accounts for (PRD P0-11).
_MAX_VALIDATOR_THREADS: Final[int] = 8


def run_validators(
    pages: Sequence[OCRPage],
    *,
    ifsc_master: IFSCMaster | None = None,
    banks: Sequence[BankRecord] | None = None,
    bank_table: BankTable | None = None,
    validators: Sequence[str] | None = None,
    max_workers: int | None = None,
) -> ValidatorReport:
    """Run every deterministic extractor over every page, concurrently.

    BUILD-BRIEF M5 asks for a thread pool; the important property is that using
    one cannot change the answer.  Each task is a pure function of one page's
    lines and shared read-only tables, and results are re-sorted into
    ``(page_no, line_index, span.start, field_name)`` order before returning, so
    a report is byte-identical whether it ran on one thread or eight.

    Args:
        pages: recognised pages.  A page with no lines contributes nothing.
        ifsc_master: defaults to the committed synthetic fixture.
        banks: bank records only.  Overrides ``bank_table``'s records while
            keeping its non-party list; mostly a test convenience.
        bank_table: the whole bank-name table — records *and* the non-party
            institutions.  Defaults to the committed one.  A deployment passes
            the table named by ``BANK_ALIASES_PATH`` here.
        validators: subset of :data:`VALIDATOR_FIELD_NAMES` to run.
        max_workers: pool size; defaults to a small bound over the task count.

    Raises:
        ValueError: for an unknown validator name — a typo must not silently
            skip an entire class of identifier.
    """
    selected = tuple(validators) if validators is not None else VALIDATOR_FIELD_NAMES
    unknown = sorted(set(selected) - set(VALIDATORS))
    if unknown:
        raise ValueError(
            f"unknown validator(s): {', '.join(unknown)}; available: "
            f"{', '.join(VALIDATOR_FIELD_NAMES)}"
        )
    master = ifsc_master if ifsc_master is not None else default_ifsc_master()
    table = bank_table if bank_table is not None else default_bank_table()
    records = tuple(banks) if banks is not None else table.records

    scans = [_prepare(page, master, records, table.non_party) for page in pages]
    tasks: list[tuple[PageScan, Callable[[PageScan], list[ValidatorHit]]]] = [
        (scan, VALIDATORS[name]) for scan in scans for name in selected
    ]
    if not tasks:
        return ValidatorReport(hits=())

    workers = max_workers or min(_MAX_VALIDATOR_THREADS, len(tasks))
    collected: list[ValidatorHit] = []
    if workers <= 1:
        for scan, validator in tasks:
            collected.extend(validator(scan))
    else:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="s5-validator"
        ) as pool:
            for result in pool.map(lambda task: task[1](task[0]), tasks):
                collected.extend(result)

    collected = _cross_check_bank_name(collected)
    collected.sort(
        key=lambda hit: (
            hit.evidence.page_no,
            hit.span.line_index,
            hit.span.start,
            hit.field_name,
            hit.value,
        )
    )
    return ValidatorReport(hits=tuple(collected))


def _cross_check_bank_name(hits: Sequence[ValidatorHit]) -> list[ValidatorHit]:
    """Flag ``bank_name``/IFSC disagreement.  Never resolves it.

    Runs document-wide, after every page: an IFSC on the cheque attached to a
    complaint must still contradict the bank named in the letter.  Both sides
    keep their values; the bank-name hit gains the flag, because that is the
    field a reviewer has to adjudicate.
    """
    prefixes = {
        hit.normalized[:4]
        for hit in hits
        if hit.field_name == "ifsc_code" and hit.normalized
    }
    if not prefixes:
        return list(hits)
    result: list[ValidatorHit] = []
    for hit in hits:
        if hit.field_name != "bank_name":
            result.append(hit)
            continue
        code = _bank_code_of(hit)
        if code is None or code in prefixes:
            result.append(hit)
            continue
        result.append(
            hit.model_copy(
                update={
                    "normalized": None,
                    "confidence": min(hit.confidence, FAIL_WEIGHT),
                    "checks": (
                        *hit.checks,
                        CheckOutcome(
                            name="ifsc_agreement",
                            passed=False,
                            detail=(
                                f"matched bank {code}, IFSC prefixes on this "
                                f"document are {sorted(prefixes)}"
                            ),
                        ),
                    ),
                    "flags": tuple(
                        dict.fromkeys(
                            (*hit.flags, FieldFlag.BANK_NAME_IFSC_DISAGREEMENT)
                        )
                    ),
                    "detail": (
                        "bank name and IFSC bank code disagree; both values are "
                        "kept verbatim and neither is preferred"
                    ),
                }
            )
        )
    return result


def _bank_code_of(hit: ValidatorHit) -> str | None:
    """Recover the bank code recorded in a ``bank_name`` hit's detail."""
    if hit.detail is None:
        return None
    match = re.search(r"bank code ([A-Z]{4})", hit.detail)
    return match.group(1) if match else None
