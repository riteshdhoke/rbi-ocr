"""Property tests for the deterministic extractors (BUILD-BRIEF M5, hypothesis).

Three kinds of property, one per class of bug these validators can have:

**Round trip.**  Anything the service can *format* it must be able to *read*,
and reading must give back the same number, date or check digit.  This is the
property that catches a grouping regex that quietly stops handling
``12,34,567``.

**Adversarial garbage.**  Arbitrary Unicode — control characters, unpaired
combining marks, right-to-left overrides, digits from six scripts — must never
raise out of a validator.  A complaint attachment is untrusted input and Stage 5
sits behind an OCR engine that will happily emit anything; the pass condition is
a typed result or no hit, never a traceback (BUILD-BRIEF sec. 0.5, PLAN.md
invariant 11).

**Boundaries.**  Length classes, edit budgets and grouping conventions are all
defined by an inclusive range, and an off-by-one at either end is invisible in
example-based tests.

Every strategy here is deterministic given hypothesis's seed; nothing sleeps,
nothing touches the network, and no weights are needed.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal
from itertools import pairwise

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.contracts.ocr import OCRLine, OCRPage, OCRTier, bbox_from_xyxy
from app.stages.s5_validators import (
    ACCOUNT_LENGTH_CLASSES,
    IFSC_STRUCTURE,
    MICR_STRUCTURE,
    account_length_class,
    classify_reference,
    format_indian_grouping,
    luhn_check,
    merge_spans,
    normalise_digits,
    parse_amount,
    parse_indian_date,
    run_validators,
)

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

#: Text drawn from every category that has ever broken a regex: control
#: characters, surrogate-free astral planes, combining marks, and the Indic
#: digit blocks the normaliser handles.
GARBAGE = st.text(
    alphabet=st.one_of(
        st.characters(),
        st.characters(min_codepoint=0x0900, max_codepoint=0x097F),
        st.characters(min_codepoint=0x0660, max_codepoint=0x066F),
        st.sampled_from(list("0123456789,.-/₹")),
    ),
    max_size=200,
)

ASCII_DIGITS = st.text(alphabet="0123456789", min_size=1, max_size=24)

BANK_PREFIXES = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=4, max_size=4
)

INDIC_ZEROS = (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66, 0x0660, 0x06F0)


def to_indic(digits: str, zero: int) -> str:
    """Render an ASCII digit string in the script whose zero is ``zero``."""
    return "".join(chr(zero + int(character)) for character in digits)


def _luhn_complete(body: str) -> str:
    """Append the check digit that makes ``body`` Luhn-valid.

    Constructing valid numbers beats filtering for them.  ``assume(luhn_check(
    pan))`` over random digit strings discards nine inputs in ten, which
    hypothesis rightly refuses as ``filter_too_much``: the health check is not
    noise, it is a warning that the surviving examples are a distorted sample of
    the domain.  This map is total, so every generated example is a valid PAN
    and the shrinker still has the whole space to work in.

    The arithmetic is written out rather than found by trying ten candidates
    against :func:`luhn_check`, so the transposition property is not tested
    against its own implementation.  The appended digit lands at position 0 from
    the right and is never doubled, which shifts every body digit up one place —
    hence ``position % 2 == 0`` here where :func:`luhn_check` uses ``== 1``.
    """
    total = 0
    for position, character in enumerate(reversed(body)):
        value = ord(character) - 48
        if position % 2 == 0:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return f"{body}{(10 - total % 10) % 10}"


def _page(texts: list[str]) -> OCRPage:
    return OCRPage(
        page_no=1,
        width_px=1000,
        height_px=1000,
        lines=[
            OCRLine(
                text=text,
                bbox=bbox_from_xyxy(0.0, 30.0 * index, 100.0, 30.0 * index + 28),
                confidence=0.9,
                script="Latn",
                language=None,
                is_handwritten=False,
                line_index=index,
            )
            for index, text in enumerate(texts)
        ],
        tier=OCRTier.TIER1,
    )


# --------------------------------------------------------------------------- #
# Numeral normalisation
# --------------------------------------------------------------------------- #


class TestDigitNormalisation:
    @given(GARBAGE)
    def test_length_is_always_preserved(self, text: str) -> None:
        """Evidence spans index the original string; a normaliser that changed
        length would silently corrupt every bbox-to-character mapping."""
        assert len(normalise_digits(text)) == len(text)

    @given(GARBAGE)
    def test_normalisation_is_idempotent(self, text: str) -> None:
        once = normalise_digits(text)
        assert normalise_digits(once) == once

    @given(ASCII_DIGITS, st.sampled_from(INDIC_ZEROS))
    def test_every_script_round_trips_to_ascii(self, digits: str, zero: int) -> None:
        assert normalise_digits(to_indic(digits, zero)) == digits

    @given(GARBAGE)
    def test_ascii_text_is_untouched(self, text: str) -> None:
        ascii_only = "".join(c for c in text if c.isascii())
        assert normalise_digits(ascii_only) == ascii_only


# --------------------------------------------------------------------------- #
# Luhn
# --------------------------------------------------------------------------- #


class TestLuhnProperties:
    @given(st.text(alphabet="0123456789", min_size=12, max_size=18))
    def test_appending_the_computed_check_digit_always_validates(
        self, body: str
    ) -> None:
        for candidate in range(10):
            if luhn_check(f"{body}{candidate}"):
                break
        else:  # pragma: no cover - a check digit always exists
            raise AssertionError("no check digit satisfied Luhn")
        assert luhn_check(f"{body}{candidate}") is True

    @given(st.text(alphabet="0123456789", min_size=13, max_size=19))
    def test_exactly_one_final_digit_validates(self, pan: str) -> None:
        body = pan[:-1]
        valid = [d for d in range(10) if luhn_check(f"{body}{d}")]
        assert len(valid) == 1

    @given(st.text(alphabet="0123456789", min_size=1, max_size=19).map(_luhn_complete))
    def test_transposing_adjacent_unequal_digits_is_detected(self, pan: str) -> None:
        """Luhn's whole purpose: catching a transposition.

        The blind spot carved out below is exactly one pair, and the exclusion
        is a proof rather than a concession.  Adjacent positions always have
        opposite parity, so a swap trades ``double(x) + y`` for ``double(y) +
        x``; the checksum moves by ``f(y) - f(x)`` where ``f(z) = double(z) - z``
        is ``z`` for ``z <= 4`` and ``z - 9`` above it.  Mod 10 that map is
        ``0 1 2 3 4 6 7 8 9 0`` — injective except at 0 and 9, so ``09 <-> 90``
        is the only invisible transposition and every other one must be caught.
        """
        for index in range(len(pan) - 1):
            if pan[index] == pan[index + 1]:
                continue
            swapped = (
                pan[:index] + pan[index + 1] + pan[index] + pan[index + 2 :]
            )
            if abs(int(pan[index]) - int(pan[index + 1])) == 9:
                continue  # 0 <-> 9: the one transposition Luhn cannot see
            assert luhn_check(swapped) is False


# --------------------------------------------------------------------------- #
# Amounts
# --------------------------------------------------------------------------- #


class TestAmountProperties:
    @given(st.integers(min_value=0, max_value=10**14 - 1))
    def test_indian_grouping_round_trips(self, value: int) -> None:
        parsed = parse_amount(format_indian_grouping(value))
        assert parsed.ok is True
        assert Decimal(parsed.normalized or "nan") == Decimal(value)

    @given(st.integers(min_value=0, max_value=10**14 - 1))
    def test_western_grouping_round_trips(self, value: int) -> None:
        parsed = parse_amount(f"{value:,}")
        assert parsed.ok is True
        assert Decimal(parsed.normalized or "nan") == Decimal(value)

    @given(st.integers(min_value=0, max_value=10**14 - 1))
    def test_ungrouped_round_trips(self, value: int) -> None:
        parsed = parse_amount(str(value))
        assert parsed.ok is True
        assert Decimal(parsed.normalized or "nan") == Decimal(value)

    @given(
        st.integers(min_value=0, max_value=10**12 - 1),
        st.sampled_from(INDIC_ZEROS),
    )
    def test_indic_digits_round_trip(self, value: int, zero: int) -> None:
        parsed = parse_amount(to_indic(str(value), zero))
        assert parsed.ok is True
        assert Decimal(parsed.normalized or "nan") == Decimal(value)

    @given(st.integers(min_value=15, max_value=40))
    def test_beyond_the_magnitude_ceiling_is_refused(self, digits: int) -> None:
        """A 15+ digit "amount" is an identifier; refusing it is what keeps an
        account number out of the amount field."""
        parsed = parse_amount("1" * (digits + 1))
        assert parsed.ok is False
        assert parsed.reason == "range"

    @given(GARBAGE)
    def test_never_raises_and_never_invents_a_value(self, text: str) -> None:
        parsed = parse_amount(text)
        assert parsed.ok is (parsed.normalized is not None)

    @given(st.integers(min_value=1, max_value=10**9))
    def test_a_failed_parse_never_produces_a_normalised_value(
        self, value: int
    ) -> None:
        broken = f"{value},0,{value}"
        parsed = parse_amount(broken)
        assume(parsed.ok is False)
        assert parsed.normalized is None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


class TestDateProperties:
    @given(st.dates(min_value=dt.date(1970, 1, 1)))
    def test_slash_notation_round_trips(self, value: dt.date) -> None:
        text = f"{value.day:02d}/{value.month:02d}/{value.year:04d}"
        parsed = parse_indian_date(text)
        assert parsed.ok is True
        assert parsed.iso == value.isoformat()

    @given(
        st.dates(min_value=dt.date(1970, 1, 1)),
        st.sampled_from(["/", "-", "."]),
    )
    def test_every_separator_behaves_the_same(
        self, value: dt.date, separator: str
    ) -> None:
        text = separator.join(
            (f"{value.day:02d}", f"{value.month:02d}", f"{value.year:04d}")
        )
        assert parse_indian_date(text).iso == value.isoformat()

    @given(st.dates(min_value=dt.date(1970, 1, 1)))
    def test_mixed_separators_are_not_a_date(self, value: dt.date) -> None:
        text = f"{value.day:02d}/{value.month:02d}-{value.year:04d}"
        assert parse_indian_date(text).ok is False

    @given(
        st.integers(min_value=1, max_value=31),
        st.integers(min_value=13, max_value=99),
        st.integers(min_value=1970, max_value=2100),
    )
    def test_a_month_beyond_twelve_is_refused_never_reinterpreted(
        self, day: int, month: int, year: int
    ) -> None:
        parsed = parse_indian_date(f"{day:02d}/{month:02d}/{year}")
        assert parsed.iso is None

    @given(GARBAGE)
    def test_never_raises(self, text: str) -> None:
        parsed = parse_indian_date(text)
        assert parsed.ok is (parsed.iso is not None)


# --------------------------------------------------------------------------- #
# Identifier structure
# --------------------------------------------------------------------------- #


class TestIdentifierStructure:
    @given(
        BANK_PREFIXES,
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=6, max_size=6),
    )
    def test_generated_ifsc_always_matches_the_structure(
        self, prefix: str, tail: str
    ) -> None:
        assert IFSC_STRUCTURE.match(f"{prefix}0{tail}")

    @given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=20))
    def test_structure_is_exactly_eleven_characters_with_a_zero_fifth(
        self, token: str
    ) -> None:
        expected = (
            len(token) == 11
            and token[:4].isalpha()
            and token[4] == "0"
            and token[5:].isalnum()
        )
        assert bool(IFSC_STRUCTURE.match(token)) is expected

    @given(ASCII_DIGITS)
    def test_micr_structure_is_exactly_nine_digits(self, digits: str) -> None:
        assert bool(MICR_STRUCTURE.match(digits)) is (len(digits) == 9)

    @given(st.integers(min_value=0, max_value=30))
    def test_account_length_classes_tile_the_range_without_gaps(
        self, length: int
    ) -> None:
        name = account_length_class("1" * length) if length else None
        low = min(bound for _n, bound, _h in ACCOUNT_LENGTH_CLASSES)
        high = max(bound for _n, _l, bound in ACCOUNT_LENGTH_CLASSES)
        assert (name is not None) is (low <= length <= high)

    @given(ASCII_DIGITS)
    def test_only_twelve_digits_are_an_rrn(self, digits: str) -> None:
        assert (classify_reference(digits) == "rrn_12") is (len(digits) == 12)

    @given(GARBAGE)
    def test_classify_reference_never_raises(self, text: str) -> None:
        assert classify_reference(text) in {
            None,
            "rrn_12",
            "utr_neft_16",
            "utr_rtgs_22",
        }


# --------------------------------------------------------------------------- #
# Whole-pipeline robustness
# --------------------------------------------------------------------------- #


class TestAdversarialPages:
    @settings(max_examples=60, deadline=None)
    @given(st.lists(GARBAGE, min_size=1, max_size=6))
    def test_no_validator_raises_on_arbitrary_text(self, texts: list[str]) -> None:
        report = run_validators([_page(texts)], max_workers=1)
        for hit in report.hits:
            assert hit.evidence.page_no == 1
            assert hit.value

    @settings(max_examples=60, deadline=None)
    @given(st.lists(GARBAGE, min_size=1, max_size=6))
    def test_every_hit_points_at_real_characters(self, texts: list[str]) -> None:
        """The span must index the *original* line, not the normalised copy."""
        page = _page(texts)
        by_index = {line.line_index: line for line in page.lines}
        for hit in run_validators([page], max_workers=1).hits:
            line = by_index[hit.span.line_index]
            assert hit.span.end <= len(line.text)
            assert line.text[hit.span.start : hit.span.end] == hit.value

    @settings(max_examples=40, deadline=None)
    @given(st.lists(GARBAGE, min_size=1, max_size=5))
    def test_masked_spans_are_always_disjoint(self, texts: list[str]) -> None:
        spans = run_validators([_page(texts)], max_workers=1).masked_spans(1)
        for left, right in pairwise(spans):
            assert (left.line_index, left.end) <= (right.line_index, right.start)

    @settings(max_examples=40, deadline=None)
    @given(st.lists(GARBAGE, min_size=1, max_size=5))
    def test_a_failed_hit_never_carries_a_normalised_value(
        self, texts: list[str]
    ) -> None:
        for hit in run_validators([_page(texts)], max_workers=1).hits:
            if not hit.validator_passed:
                assert hit.normalized is None

    @settings(max_examples=30, deadline=None)
    @given(st.lists(GARBAGE, min_size=1, max_size=4))
    def test_concurrency_never_changes_the_report(self, texts: list[str]) -> None:
        page = _page(texts)
        assert (
            run_validators([page], max_workers=1).model_dump()
            == run_validators([page], max_workers=8).model_dump()
        )


class TestMergeSpansProperties:
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=3),
                st.integers(min_value=0, max_value=40),
                st.integers(min_value=0, max_value=40),
            ),
            min_size=1,
            max_size=25,
        )
    )
    def test_merged_spans_are_disjoint_ordered_and_cover_every_input(
        self, raw: list[tuple[int, int, int]]
    ) -> None:
        from app.contracts.ocr import TextSpan

        spans = [
            TextSpan(line_index=line, start=min(a, b), end=max(a, b))
            for line, a, b in raw
        ]
        merged = merge_spans(spans)
        keyed = [(s.line_index, s.start, s.end) for s in merged]
        assert keyed == sorted(keyed)
        for left, right in pairwise(merged):
            if left.line_index == right.line_index:
                assert left.end < right.start
        for span in spans:
            assert any(
                other.line_index == span.line_index
                and other.start <= span.start
                and span.end <= other.end
                for other in merged
            )


class TestBankNameRobustness:
    @given(BANK_PREFIXES, st.integers(min_value=100000, max_value=999999))
    def test_an_ifsc_code_is_never_read_as_a_bank_name(
        self, prefix: str, tail: int
    ) -> None:
        """The short-alias trap: ``WRatio("SBIN0904158", "SBI")`` is 90."""
        code = f"{prefix}0{tail}"
        report = run_validators([_page(["5. IFSC code", code])], max_workers=1)
        for hit in report.hits_for("bank_name"):
            assert not re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", hit.value)
