"""Unit tests for the deterministic extractors (WS-E3, BUILD-BRIEF M5).

The load-bearing assertions here are the ones about *not* doing things: not
correcting a failed checksum, not normalising a value that failed a check, not
guessing which of two identical-looking digit runs is an account number, not
picking a winner when the bank name and the IFSC disagree.

Realistic fixtures come from the committed corpus ground truth, variants 0-2
only — variants 3 and 4 are the locked test split (BUILD-BRIEF M1).
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from app.contracts.envelope import Evidence, FieldFlag
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, TextSpan, bbox_from_xyxy
from app.resources import ifsc_master_sample_path
from app.stages.s5_validators import (
    ACCOUNT_CHECKSUM_POLICY,
    BankRecord,
    CheckOutcome,
    IFSCMaster,
    ValidatorHit,
    account_length_class,
    classify_reference,
    default_bank_records,
    default_ifsc_master,
    format_indian_grouping,
    luhn_check,
    merge_spans,
    normalise_digits,
    parse_amount,
    parse_indian_date,
    run_validators,
)

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "corpus"
#: BUILD-BRIEF M1 locks variants 3 and 4 as the held-out test split.
OPEN_VARIANTS = ("v0", "v1", "v2")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def make_line(text: str, index: int, **overrides: object) -> OCRLine:
    """One OCR line with a distinct bbox per index, so evidence is traceable."""
    data: dict[str, object] = {
        "text": text,
        "bbox": bbox_from_xyxy(100.0, 100.0 + 40 * index, 900.0, 132.0 + 40 * index),
        "confidence": 1.0,
        "script": "Latn",
        "language": None,
        "is_handwritten": False,
        "line_index": index,
    }
    data.update(overrides)
    return OCRLine(**data)  # type: ignore[arg-type]


def make_page(texts: Sequence[str], page_no: int = 1) -> OCRPage:
    return OCRPage(
        page_no=page_no,
        width_px=1654,
        height_px=2339,
        lines=[make_line(text, index) for index, text in enumerate(texts)],
        tier=OCRTier.TIER1,
    )


def hits_for(texts: Sequence[str], field: str, **kwargs: object) -> list[ValidatorHit]:
    report = run_validators([make_page(texts)], **kwargs)  # type: ignore[arg-type]
    return list(report.hits_for(field))


def corpus_pages() -> list[tuple[str, OCRPage, dict[str, object]]]:
    """Every open-split ``*_clean`` ground truth, as OCR pages."""
    documents: list[tuple[str, OCRPage, dict[str, object]]] = []
    for path in sorted(CORPUS_ROOT.glob("*/*_clean.gt.json")):
        if not any(f"_{variant}_" in path.name for variant in OPEN_VARIANTS):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("lines"):
            continue
        lines = [
            OCRLine(
                text=entry["text"],
                bbox=[float(value) for value in entry["bbox"]],
                confidence=0.95,
                script=entry["script"],
                language=entry["language"],
                is_handwritten=entry["is_handwritten"],
                line_index=index,
            )
            for index, entry in enumerate(payload["lines"])
        ]
        documents.append(
            (
                path.name,
                OCRPage(
                    page_no=1,
                    width_px=1654,
                    height_px=2339,
                    lines=lines,
                    tier=OCRTier.TIER1,
                ),
                payload.get("fields") or {},
            )
        )
    return documents


# --------------------------------------------------------------------------- #
# Numerals
# --------------------------------------------------------------------------- #


class TestNormaliseDigits:
    def test_devanagari_digits_become_ascii(self) -> None:
        assert normalise_digits("४५,०००") == "45,000"

    @pytest.mark.parametrize(
        "text",
        [
            "٤٥",  # Arabic-Indic
            "۴۵",  # Extended Arabic-Indic
            "৪৫",  # Bengali
            "੪੫",  # Gurmukhi
            "૪૫",  # Gujarati
            "௪௫",  # Tamil
        ],
    )
    def test_every_supported_script_maps(self, text: str) -> None:
        assert normalise_digits(text) == "45"

    def test_length_is_preserved_so_offsets_stay_valid(self) -> None:
        source = "खाते क्रमांक ३३३९०३७६८०६ आहे."
        assert len(normalise_digits(source)) == len(source)

    def test_non_digits_pass_through_untouched(self) -> None:
        source = "बैंक ऑफ बड़ौदा — Bank of Baroda"
        assert normalise_digits(source) == source


# --------------------------------------------------------------------------- #
# Luhn
# --------------------------------------------------------------------------- #


class TestLuhn:
    def test_known_valid_pan(self) -> None:
        assert luhn_check("4539578763621486") is True

    def test_single_digit_error_is_caught(self) -> None:
        assert luhn_check("4539578763621487") is False

    @pytest.mark.parametrize("bad", ["", "12 34", "abcd"])
    def test_rejects_unstripped_input(self, bad: str) -> None:
        with pytest.raises(ValueError):
            luhn_check(bad)


# --------------------------------------------------------------------------- #
# IFSC
# --------------------------------------------------------------------------- #


class TestIFSCMaster:
    def test_committed_fixture_is_not_authoritative(self) -> None:
        master = default_ifsc_master()
        assert master.authoritative is False
        assert len(master) >= 15
        assert {bank.code for bank in master.banks} == {
            "SBIN",
            "PUNB",
            "HDFC",
            "BARB",
            "UBIN",
        }

    def test_lookup_is_case_insensitive(self) -> None:
        master = default_ifsc_master()
        assert master.lookup("sbin0152060") is not None
        assert master.lookup("SBIN0152060") is not None

    def test_unknown_branch_is_none_not_a_guess(self) -> None:
        assert default_ifsc_master().lookup("SBIN0999999") is None

    def test_load_dispatches_on_suffix(self) -> None:
        master = IFSCMaster.load(ifsc_master_sample_path())
        assert len(master) == len(default_ifsc_master())

    def test_load_refuses_an_unknown_format(self, tmp_path: Path) -> None:
        target = tmp_path / "master.xlsx"
        target.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported IFSC master format"):
            IFSCMaster.load(target)

    def test_csv_drop_in_is_authoritative_by_default(self, tmp_path: Path) -> None:
        target = tmp_path / "IFSC.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["IFSC", "BANK", "BRANCH", "CENTRE", "STATE"])
            writer.writerow(["SBIN0000001", "State Bank of India", "MAIN", "MUMBAI", "MH"])
            writer.writerow(["garbage", "Nowhere Bank", "", "", ""])
        master = IFSCMaster.from_csv(target)
        assert master.authoritative is True
        assert len(master) == 1
        assert master.skipped_rows == 1
        assert master.lookup("SBIN0000001") is not None

    def test_csv_without_an_ifsc_column_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "wrong.csv"
        target.write_text("bank,branch\nX,Y\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no IFSC column"):
            IFSCMaster.from_csv(target)

    def test_truncated_json_is_refused_not_treated_as_empty(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "master.json"
        target.write_text('{"banks": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="not an IFSC master document"):
            IFSCMaster.from_json(target)


class TestIFSCValidator:
    def test_structurally_valid_code_is_found(self) -> None:
        hits = hits_for(["5. IFSC code", "SBIN0152060"], "ifsc_code")
        assert len(hits) == 1
        assert hits[0].value == "SBIN0152060"
        assert hits[0].normalized == "SBIN0152060"
        assert hits[0].validator_passed is True
        assert hits[0].flags == ()

    def test_branch_row_found_records_the_lookup(self) -> None:
        hits = hits_for(["IFSC", "SBIN0152060"], "ifsc_code")
        names = {check.name for check in hits[0].checks}
        assert names == {"structure", "master_lookup"}

    def test_known_prefix_unknown_branch_is_unverified_not_wrong(self) -> None:
        hits = hits_for(["IFSC code", "SBIN0999999"], "ifsc_code")
        assert hits[0].validator_passed is True
        assert FieldFlag.IFSC_NOT_IN_MASTER not in hits[0].flags
        assert "prefix SBIN is known" in (hits[0].detail or "")

    def test_unknown_prefix_is_flagged(self) -> None:
        hits = hits_for(["IFSC code", "ZZZZ0123456"], "ifsc_code")
        assert FieldFlag.IFSC_NOT_IN_MASTER in hits[0].flags

    def test_authoritative_master_turns_absence_into_a_failure(self) -> None:
        master = IFSCMaster(
            records=[],  # the real list would have this branch; this one does not
            banks=[BankRecord(code="SBIN", name="State Bank of India")],
            authoritative=True,
        )
        hits = hits_for(["IFSC code", "SBIN0152060"], "ifsc_code", ifsc_master=master)
        assert hits[0].validator_passed is False
        assert hits[0].normalized is None
        assert FieldFlag.IFSC_NOT_IN_MASTER in hits[0].flags
        assert hits[0].to_field().flags.count(FieldFlag.VALIDATOR_FAILED) == 1

    def test_malformed_token_under_an_ifsc_label_is_flagged_not_dropped(self) -> None:
        hits = hits_for(["5. IFSC code", "SBIN1152060"], "ifsc_code")
        assert len(hits) == 1
        assert hits[0].validator_passed is False
        assert hits[0].value == "SBIN1152060"
        assert hits[0].normalized is None

    def test_malformed_token_without_a_label_is_not_reported(self) -> None:
        assert hits_for(["random text SBIN1152060 here"], "ifsc_code") == []

    def test_lowercase_ocr_output_is_accepted_and_uppercased_in_normalized(
        self,
    ) -> None:
        hits = hits_for(["IFSC", "sbin0152060"], "ifsc_code")
        assert hits[0].value == "sbin0152060"
        assert hits[0].normalized == "SBIN0152060"


# --------------------------------------------------------------------------- #
# Account number
# --------------------------------------------------------------------------- #


class TestAccountNumber:
    def test_labelled_run_is_extracted(self) -> None:
        hits = hits_for(["6. Account number", "61136570711"], "account_number")
        assert len(hits) == 1
        assert hits[0].normalized == "61136570711"
        assert hits[0].validator_passed is True

    def test_marathi_label_is_recognised(self) -> None:
        hits = hits_for(["खाते क्रमांक 33390376806 आहे."], "account_number")
        assert hits[0].normalized == "33390376806"

    def test_indic_digits_normalise_but_value_stays_native(self) -> None:
        hits = hits_for(["खाते क्रमांक ३३३९०३७६८०६ आहे."], "account_number")
        assert hits[0].value == "३३३९०३७६८०६"
        assert hits[0].normalized == "33390376806"

    def test_a_mobile_number_is_not_an_account_number(self) -> None:
        assert hits_for(["3. Mobile", "9372805183"], "account_number") == []

    def test_no_checksum_outcome_is_recorded_because_none_exists(self) -> None:
        hits = hits_for(["Account number 61136570711"], "account_number")
        assert {check.name for check in hits[0].checks} == {"digits", "length_class"}
        assert hits[0].detail == ACCOUNT_CHECKSUM_POLICY

    @pytest.mark.parametrize(
        ("digits", "expected"),
        [
            ("1" * 8, None),
            ("1" * 9, "legacy_9_11"),
            ("1" * 11, "legacy_9_11"),
            ("1" * 12, "standard_12_14"),
            ("1" * 18, "extended_15_18"),
            ("1" * 19, None),
        ],
    )
    def test_length_class_boundaries(self, digits: str, expected: str | None) -> None:
        assert account_length_class(digits) == expected

    def test_out_of_range_length_fails_without_normalising(self) -> None:
        hits = hits_for(["Account number 1234567890123456789012"], "account_number")
        assert hits[0].validator_passed is False
        assert hits[0].normalized is None
        assert FieldFlag.VALIDATOR_FAILED in hits[0].to_field().flags

    def test_separators_are_stripped_for_the_canonical_form_only(self) -> None:
        hits = hits_for(["A/C No 6113 6570 711"], "account_number")
        assert hits[0].value == "6113 6570 711"
        assert hits[0].normalized == "61136570711"


# --------------------------------------------------------------------------- #
# Card fragment
# --------------------------------------------------------------------------- #


class TestCardFragment:
    def test_masked_pan_yields_last_four_and_no_checksum(self) -> None:
        hits = hits_for(["Card", "XXXX XXXX XXXX 4039"], "card_fragment")
        assert hits[0].value == "XXXX XXXX XXXX 4039"
        assert hits[0].normalized == "4039"
        assert all(check.name != "checksum" for check in hits[0].checks)

    def test_full_pan_with_a_valid_luhn_passes(self) -> None:
        hits = hits_for(["Debit card 4539578763621486"], "card_fragment")
        assert hits[0].validator_passed is True
        assert hits[0].normalized == "1486"

    def test_failed_luhn_is_flagged_and_never_corrected(self) -> None:
        hits = hits_for(["Debit card 4539578763621487"], "card_fragment")
        hit = hits[0]
        assert hit.validator_passed is False
        assert hit.value == "4539578763621487", "the digits must survive verbatim"
        assert hit.normalized is None
        assert FieldFlag.CHECKSUM_FAILED in hit.flags
        field = hit.to_field()
        assert field.validator_passed is False
        assert FieldFlag.VALIDATOR_FAILED in field.flags
        assert field.value == "4539578763621487"

    def test_labelled_last_four(self) -> None:
        hits = hits_for(["Card ending 1469"], "card_fragment")
        assert hits[0].normalized == "1469"

    def test_bare_four_digits_without_a_card_label_are_ignored(self) -> None:
        assert hits_for(["Cheque 1469"], "card_fragment") == []


# --------------------------------------------------------------------------- #
# UTR / RRN
# --------------------------------------------------------------------------- #


class TestReference:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("113699585184", "rrn_12"),
            ("SBIN012345678901", "utr_neft_16"),
            ("SBINR12345678901234567", "utr_rtgs_22"),
            ("SHORT", None),
            ("113699585184X", None),
        ],
    )
    def test_format_classes(self, token: str, expected: str | None) -> None:
        assert classify_reference(token) == expected

    def test_rrn_needs_a_label(self) -> None:
        assert hits_for(["Some line 113699585184"], "utr_rrn") == []
        hits = hits_for(["9. Transaction reference", "113699585184"], "utr_rrn")
        assert hits[0].normalized == "113699585184"
        assert hits[0].detail == "format class rrn_12"

    def test_rtgs_utr_is_distinctive_enough_to_stand_alone(self) -> None:
        hits = hits_for(["Credited via SBINR12345678901234567 today"], "utr_rrn")
        assert hits[0].normalized == "SBINR12345678901234567"

    def test_labelled_token_of_no_known_class_fails_loudly(self) -> None:
        hits = hits_for(["UTR", "ABCDEFGHIJKLMN"], "utr_rrn")
        assert hits[0].validator_passed is False
        assert hits[0].normalized is None


# --------------------------------------------------------------------------- #
# MICR
# --------------------------------------------------------------------------- #


class TestMICR:
    def test_labelled_nine_digits(self) -> None:
        hits = hits_for(["MICR code", "400002011"], "micr_code")
        assert hits[0].normalized == "400002011"
        assert hits[0].validator_passed is True

    def test_unlabelled_nine_digits_are_not_a_micr_code(self) -> None:
        assert hits_for(["Reference 400002011"], "micr_code") == []


# --------------------------------------------------------------------------- #
# Amounts
# --------------------------------------------------------------------------- #


class TestAmount:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("45000", "45000.00"),
            ("45,000", "45000.00"),
            ("1,00,000", "100000.00"),
            ("12,34,567", "1234567.00"),
            ("100,000", "100000.00"),
            ("4,500.50", "4500.50"),
            ("४५,०००", "45000.00"),
        ],
    )
    def test_parse_accepts_both_grouping_conventions(
        self, text: str, expected: str
    ) -> None:
        parsed = parse_amount(text)
        assert parsed.ok is True
        assert parsed.normalized == expected

    @pytest.mark.parametrize("text", ["1,0,000", "45,00", "1,2345"])
    def test_malformed_grouping_is_never_regrouped(self, text: str) -> None:
        parsed = parse_amount(text)
        assert parsed.ok is False
        assert parsed.reason == "grouping"
        assert parsed.normalized is None

    def test_multiplier_words(self) -> None:
        assert parse_amount("2.5", multiplier_word="lakh").normalized == "250000.00"
        assert parse_amount("1", multiplier_word="crore").normalized == "10000000.00"
        assert parse_amount("45", multiplier_word="हजार").normalized == "45000.00"

    def test_rupee_sign_extracts_from_a_line(self) -> None:
        hits = hits_for(["An amount of ₹45,000 was debited"], "amount")
        assert hits[0].value == "₹45,000"
        assert hits[0].normalized == "45000.00"

    def test_rs_variant_and_label_only_currency_cue(self) -> None:
        hits = hits_for(["7. Amount in dispute (Rs.)", "4,500"], "amount")
        assert hits[0].normalized == "4500.00"

    def test_bare_digits_with_no_currency_context_are_not_an_amount(self) -> None:
        assert hits_for(["Reference number", "61136570711"], "amount") == []

    def test_a_date_is_not_an_amount(self) -> None:
        assert hits_for(["Amount", "07/07/2026"], "amount") == []

    def test_lakh_in_running_text(self) -> None:
        hits = hits_for(["I lost 2.5 lakh from the account"], "amount")
        assert hits[0].normalized == "250000.00"
        assert hits[0].detail == "multiplier x100000"

    def test_malformed_grouping_under_a_currency_mark_is_flagged(self) -> None:
        hits = hits_for(["Rs. 1,0,000 debited"], "amount")
        assert hits[0].validator_passed is False
        assert hits[0].normalized is None
        assert "1,0,000" in hits[0].value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (500, "500"),
            (4500, "4,500"),
            (45000, "45,000"),
            (100000, "1,00,000"),
            (1234567, "12,34,567"),
        ],
    )
    def test_indian_grouping_formatter(self, value: int, expected: str) -> None:
        assert format_indian_grouping(value) == expected

    def test_formatter_refuses_negatives(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            format_indian_grouping(Decimal(-1))


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


class TestDates:
    @pytest.mark.parametrize(
        ("text", "iso"),
        [
            ("07/07/2026", "2026-07-07"),
            ("18-06-2026", "2026-06-18"),
            ("06.09.2026", "2026-09-06"),
            ("14 January 2026", "2026-01-14"),
            ("14 Jan 2026", "2026-01-14"),
            ("January 14, 2026", "2026-01-14"),
            ("०६/०९/२०२६", "2026-09-06"),
            ("06/09/26", "2026-09-06"),
            ("06/09/95", "1995-09-06"),
        ],
    )
    def test_indian_notations_normalise_to_iso(self, text: str, iso: str) -> None:
        parsed = parse_indian_date(text)
        assert parsed.ok is True
        assert parsed.iso == iso

    def test_day_first_is_assumed_and_never_second_guessed(self) -> None:
        parsed = parse_indian_date("06/13/2026")
        assert parsed.ok is False
        assert parsed.reason == "calendar"
        assert parsed.iso is None

    def test_impossible_calendar_date_fails(self) -> None:
        assert parse_indian_date("31/02/2026").ok is False

    def test_role_comes_from_the_label_cue(self) -> None:
        transaction = hits_for(["8. Date of transaction", "07/07/2026"], "transaction_date")
        assert transaction[0].detail == "role from label cue"
        complaint = hits_for(["Date of complaint", "07/07/2026"], "complaint_date")
        assert complaint[0].normalized == "2026-07-07"

    def test_unlabelled_date_defaults_to_transaction_and_says_so(self) -> None:
        hits = hits_for(["Date", "13/11/2026"], "transaction_date")
        assert "role inferred" in (hits[0].detail or "")

    def test_invalid_date_on_the_page_is_flagged_not_corrected(self) -> None:
        hits = hits_for(["Date of transaction", "31/02/2026"], "transaction_date")
        assert hits[0].validator_passed is False
        assert hits[0].value == "31/02/2026"
        assert hits[0].normalized is None


# --------------------------------------------------------------------------- #
# Bank name
# --------------------------------------------------------------------------- #


class TestBankName:
    def test_exact_english_name(self) -> None:
        hits = hits_for(["4. Name of bank", "State Bank of India"], "bank_name")
        assert len(hits) == 1
        assert hits[0].value == "State Bank of India"
        assert hits[0].normalized == "State Bank of India"
        assert hits[0].source == "lookup_table"

    def test_devanagari_alias(self) -> None:
        hits = hits_for(["4. बैंक का नाम", "पंजाब नेशनल बैंक"], "bank_name")
        assert hits[0].normalized == "Punjab National Bank"

    def test_name_inside_prose_yields_only_the_name(self) -> None:
        hits = hits_for(
            ["I hold a savings account with Bank of Baroda."], "bank_name"
        )
        assert hits[0].value == "Bank of Baroda"

    def test_one_ocr_error_still_matches(self) -> None:
        hits = hits_for(["State Bank of lndia"], "bank_name")
        assert hits[0].normalized == "State Bank of India"

    def test_a_label_line_is_not_a_bank_name(self) -> None:
        assert hits_for(["4. Name of bank"], "bank_name") == []

    def test_the_addressee_of_every_complaint_is_not_a_bank_name(self) -> None:
        assert hits_for(["Reserve Bank of India"], "bank_name") == []

    def test_a_short_acronym_never_fuzzy_matches_an_identifier(self) -> None:
        """``WRatio("SBIN0904158", "SBI")`` is 90; exact matching is why this
        does not become a bank-name hit."""
        assert hits_for(["IFSC code", "SBIN0904158"], "bank_name") == []

    def test_a_bare_label_cell_is_not_a_bank_name(self) -> None:
        """A receipt's four-character "Bank" cell scores 90 against "Bank of
        Baroda"; the alias-coverage floor is what stops it becoming a hit."""
        hits = hits_for(["Bank", "State Bank of India"], "bank_name")
        assert [hit.value for hit in hits] == ["State Bank of India"]

    def test_agreement_with_the_ifsc_prefix_leaves_the_hit_clean(self) -> None:
        hits = hits_for(
            ["Bank of Baroda", "IFSC code", "BARB0144620"], "bank_name"
        )
        assert hits[0].flags == ()
        assert hits[0].validator_passed is True

    def test_disagreement_flags_and_picks_neither_side(self) -> None:
        report = run_validators(
            [make_page(["State Bank of India", "IFSC code", "BARB0144620"])]
        )
        bank = report.hits_for("bank_name")[0]
        ifsc = report.hits_for("ifsc_code")[0]
        assert FieldFlag.BANK_NAME_IFSC_DISAGREEMENT in bank.flags
        assert bank.validator_passed is False
        assert bank.value == "State Bank of India", "the OCR'd name is kept"
        assert bank.normalized is None, "and no bank is chosen for it"
        assert ifsc.normalized == "BARB0144620", "the IFSC is not discarded either"

    def test_no_bank_table_means_no_bank_hits(self) -> None:
        report = run_validators([make_page(["State Bank of India"])], banks=[])
        assert report.hits_for("bank_name") == ()

    def test_committed_alias_table_covers_the_corpus_banks(self) -> None:
        codes = {bank.code for bank in default_bank_records()}
        assert {"SBIN", "PUNB", "HDFC", "BARB", "UBIN"} <= codes, (
            "the five banks the committed corpus renders must stay covered"
        )
        for bank in default_bank_records():
            assert bank.name in bank.aliases

    def test_committed_alias_table_covers_the_banking_system(self) -> None:
        """The table is sized to the country, not to the corpus.

        A five-bank table is not merely incomplete: a name absent from it can
        still fuzzy-match a name inside it and be published as that bank, so
        narrow coverage is an accuracy defect and not only a recall one.
        """
        codes = {bank.code for bank in default_bank_records()}
        # Every public sector bank, which is where complaint volume actually is.
        assert {
            "SBIN", "PUNB", "BARB", "UBIN", "BKID", "CNRB",
            "CBIN", "IDIB", "IOBA", "MAHB", "UCBA", "PSIB",
        } <= codes
        # The private banks that carry the retail card and UPI volume.
        assert {"HDFC", "ICIC", "UTIB", "KKBK", "INDB", "IDFB"} <= codes
        assert len(codes) >= 60, f"table covers only {len(codes)} banks"

    def test_a_bank_absent_from_the_table_is_never_reported_as_a_neighbour(
        self,
    ) -> None:
        """The defect this table was widened to fix.

        With a five-bank table, ``City Union Bank`` scored 97 against the alias
        ``Union Bank`` and was published as *Union Bank of India* — a different
        real bank — at confidence 0.970 with ``validator_passed=True`` and **no
        flag**, so nothing downstream could tell it from a correct answer.
        """
        report = run_validators([make_page(["City Union Bank", "Main Branch"])])
        hit = report.hits_for("bank_name")[0]
        assert hit.normalized == "City Union Bank Limited"
        assert hit.validator_passed is True

    @pytest.mark.parametrize(
        ("letterhead", "expected"),
        [
            ("Bank of India", "Bank of India"),
            ("State Bank of India", "State Bank of India"),
            ("City Union Bank", "City Union Bank Limited"),
            ("Union Bank of India", "Union Bank of India"),
            ("Central Bank of India", "Central Bank of India"),
        ],
    )
    def test_nested_bank_names_resolve_to_the_longest_literal(
        self, letterhead: str, expected: str
    ) -> None:
        """``Bank of India`` nests inside ``State Bank of India``.

        Fuzzy scoring cannot separate a name from its container — ``WRatio`` is
        partial-ratio driven, so the contained name scores ~100 against the
        container.  Length can, and does.
        """
        report = run_validators([make_page([letterhead, "Main Branch"])])
        hit = report.hits_for("bank_name")[0]
        assert hit.normalized == expected
        assert hit.validator_passed is True

    def test_the_regulator_is_never_reported_as_the_complained_bank(self) -> None:
        """Every RB-IOS complaint is addressed to the Reserve Bank of India.

        ``Bank of India`` is a literal substring of it, so without the non-party
        mask the address block of *every* letter yields a confident wrong bank.
        """
        report = run_validators(
            [
                make_page(
                    ["To,", "The Banking Ombudsman", "Reserve Bank of India"]
                )
            ]
        )
        assert report.hits_for("bank_name") == ()

    def test_a_bank_named_beside_the_regulator_still_registers(self) -> None:
        """The mask is containment, not overlap — it must not swallow the line."""
        report = run_validators(
            [
                make_page(
                    ["Complaint to the Reserve Bank of India against Canara Bank"]
                )
            ]
        )
        hit = report.hits_for("bank_name")[0]
        assert hit.normalized == "Canara Bank"

    def test_an_ambiguous_match_fails_rather_than_picking_a_winner(self) -> None:
        """A tie is escalated, not resolved by sort order.

        This previously published the winner with ``validator_passed=True`` and
        a ``MULTIPLE_CANDIDATES`` flag — but the router counts only *failed*
        hits, so the flag reached no queue and nobody adjudicated the tie.
        """
        table = [
            BankRecord(code="AAAA", name="Alpha Bank"),
            BankRecord(code="BBBB", name="Alpen Bank"),
        ]
        report = run_validators(
            [make_page(["Alpha Bank and Alpen Bank, Mumbai"])], banks=table
        )
        hit = report.hits_for("bank_name")[0]
        assert hit.validator_passed is False, "an unbroken tie must fail"
        assert hit.normalized is None, "and must not claim either bank"
        assert FieldFlag.MULTIPLE_CANDIDATES in hit.flags
        assert "unambiguous_match" in hit.failed_checks

    def test_ocr_noise_still_reaches_the_bank_through_the_fuzzy_path(self) -> None:
        """Exact matching must not displace fuzzy matching, only precede it.

        A dropped anusvara leaves ``बैक ऑफ इडिया``, which matches no alias
        verbatim — this is the path that recovers it.
        """
        report = run_validators([make_page(["बैक ऑफ इडिया"])])
        hit = report.hits_for("bank_name")[0]
        assert hit.normalized == "Bank of India"

    def test_every_alias_is_unique_to_one_bank(self) -> None:
        """Two banks sharing an alias would make that name permanently ambiguous."""
        seen: dict[str, str] = {}
        for bank in default_bank_records():
            for alias in bank.aliases:
                key = alias.casefold()
                assert key not in seen or seen[key] == bank.code, (
                    f"alias {alias!r} is claimed by both {seen.get(key)} "
                    f"and {bank.code}"
                )
                seen[key] = bank.code


# --------------------------------------------------------------------------- #
# Report / orchestration
# --------------------------------------------------------------------------- #


class TestRunValidators:
    def test_thread_pool_cannot_change_the_answer(self) -> None:
        page = make_page(
            [
                "4. Name of bank",
                "Bank of Baroda",
                "5. IFSC code",
                "BARB0144620",
                "6. Account number",
                "61136570711",
                "7. Amount in dispute (Rs.)",
                "45,000",
                "8. Date of transaction",
                "07/07/2026",
            ]
        )
        sequential = run_validators([page], max_workers=1)
        concurrent = run_validators([page], max_workers=8)
        assert sequential.model_dump() == concurrent.model_dump()

    def test_unknown_validator_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown validator"):
            run_validators([make_page(["x"])], validators=["ifsc_code", "nonsense"])

    def test_empty_input_is_an_empty_report(self) -> None:
        assert run_validators([]).hits == ()

    def test_hits_are_ordered_by_evidence(self) -> None:
        report = run_validators(
            [
                make_page(["IFSC code", "BARB0144620"], page_no=1),
                make_page(["Account number", "61136570711"], page_no=2),
            ]
        )
        pages = [hit.evidence.page_no for hit in report.hits]
        assert pages == sorted(pages)

    def test_fields_picks_the_best_candidate_and_flags_disagreement(self) -> None:
        report = run_validators(
            [
                make_page(
                    [
                        "6. Account number",
                        "61136570711",
                        "Account number",
                        "99999999999",
                    ]
                )
            ]
        )
        field = report.fields()["account_number"]
        assert FieldFlag.MULTIPLE_CANDIDATES in field.flags

    def test_masked_spans_are_disjoint_and_cover_every_hit(self) -> None:
        report = run_validators(
            [make_page(["Account number 61136570711 IFSC BARB0144620"])]
        )
        spans = report.masked_spans(1)
        assert spans
        for left, right in pairwise(spans):
            assert left.line_index < right.line_index or left.end <= right.start

    def test_masked_spans_include_failing_hits(self) -> None:
        """A wrong account number is still an account number for residency."""
        report = run_validators([make_page(["Account number 12345678901234567890123"])])
        assert report.hits_for("account_number")[0].validator_passed is False
        assert report.masked_spans(1)

    def test_a_page_with_no_lines_contributes_nothing(self) -> None:
        page = OCRPage(
            page_no=1, width_px=100, height_px=100, lines=[], tier=OCRTier.NONE
        )
        assert run_validators([page]).hits == ()


class TestMergeSpans:
    def test_overlaps_are_merged_per_line(self) -> None:
        merged = merge_spans(
            [
                TextSpan(line_index=0, start=0, end=5),
                TextSpan(line_index=0, start=3, end=9),
                TextSpan(line_index=0, start=20, end=25),
                TextSpan(line_index=1, start=0, end=4),
            ]
        )
        assert [(s.line_index, s.start, s.end) for s in merged] == [
            (0, 0, 9),
            (0, 20, 25),
            (1, 0, 4),
        ]


class TestValidatorHitContract:
    def test_a_non_deterministic_field_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a deterministic field"):
            ValidatorHit(
                field_name="complainant_name",
                value="x",
                confidence=1.0,
                evidence=_evidence(),
            )

    def test_a_failed_hit_may_not_carry_a_normalized_value(self) -> None:
        with pytest.raises(ValueError, match="flag it, never correct it"):
            ValidatorHit(
                field_name="amount",
                value="1,0,000",
                normalized="10000.00",
                confidence=0.5,
                evidence=_evidence(),
                checks=(CheckOutcome(name="grouping", passed=False),),
            )


def _evidence() -> Evidence:
    return Evidence(
        page_no=1,
        bbox=bbox_from_xyxy(0.0, 0.0, 10.0, 10.0),
        span=TextSpan(line_index=0, start=0, end=1),
    )


# --------------------------------------------------------------------------- #
# Against the committed corpus
# --------------------------------------------------------------------------- #


class TestAgainstCorpus:
    def test_corpus_is_present(self) -> None:
        assert corpus_pages(), "corpus/ must be committed for these tests"

    @pytest.mark.parametrize(
        ("name", "page", "expected"),
        [
            pytest.param(name, page, expected, id=name)
            for name, page, expected in corpus_pages()
        ],
    )
    def test_identifiers_rendered_on_the_page_are_all_recovered(
        self, name: str, page: OCRPage, expected: dict[str, object]
    ) -> None:
        rendered = " ".join(line.text for line in page.lines)
        fields = run_validators([page]).fields()
        for gt_key, field_name in (
            ("ifsc_code", "ifsc_code"),
            ("account_number", "account_number"),
            ("bank_name", "bank_name"),
        ):
            value = expected.get(gt_key)
            if not value or str(value) not in rendered:
                continue  # this renderer does not put the fact on the page
            assert field_name in fields, f"{name}: {field_name} missing"
            assert fields[field_name].normalized == str(value)

    def test_no_corpus_page_produces_a_checksum_failure(self) -> None:
        """The generator emits structurally valid identifiers (TEST-DATA-SPEC
        sec. 2), so a failure here is a validator bug, not a corpus defect."""
        for name, page, _expected in corpus_pages():
            for hit in run_validators([page]).hits:
                assert FieldFlag.CHECKSUM_FAILED not in hit.flags, name


# --------------------------------------------------------------------------- #
# OPEN-QUESTIONS Q4 — the synthetic master must cover the corpus it is for
# --------------------------------------------------------------------------- #


def corpus_ifsc_codes() -> set[str]:
    """Every IFSC code that appears anywhere in the committed ground truth.

    Read from the whole corpus, held-out split included: this is an audit of the
    *reference table*, not a measurement of extraction quality, so excluding the
    locked variants would only make the audit weaker.
    """
    found: set[str] = set()
    for path in sorted(CORPUS_ROOT.glob("*/*.gt.json")):
        found |= set(
            re.findall(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", path.read_text(encoding="utf-8"))
        )
    return found


class TestQ4MasterCoverage:
    """The committed fixture is option (b) of OPEN-QUESTIONS Q4: a synthetic
    master covering the corpus banks, plus a loader that takes the real RBI list
    as a drop-in.  These tests are what "covering" has to mean."""

    def test_every_bank_prefix_in_the_corpus_is_known(self) -> None:
        """A prefix the master has never heard of flags ``IFSC_NOT_IN_MASTER`` on
        every page that carries it — which would flood the review queue with the
        fixture's own gaps rather than with defects."""
        master = default_ifsc_master()
        prefixes = {code[:4] for code in corpus_ifsc_codes()}
        assert prefixes, "corpus/ must carry IFSC codes for this audit"
        assert prefixes == {"SBIN", "PUNB", "HDFC", "BARB", "UBIN"}
        missing = sorted(p for p in prefixes if not master.knows_prefix(p))
        assert missing == [], f"IFSC prefixes absent from the master: {missing}"

    def test_every_branch_row_is_a_code_the_corpus_actually_renders(self) -> None:
        """The other direction: a fixture row nobody exercises is dead weight and
        hides the fact that the lookup path is never taken."""
        master = default_ifsc_master()
        codes = corpus_ifsc_codes()
        rows = {
            row for row in codes if master.lookup(row) is not None
        }
        assert len(rows) == len(master), (
            "every ifsc_master_sample.json branch row should be a code the "
            "corpus renders"
        )

    def test_a_corpus_code_with_no_branch_row_is_unverifiable_not_wrong(
        self,
    ) -> None:
        """The load-bearing consequence of ``authoritative=false``.  Only 15 of
        the corpus's ~210 codes have branch rows; if absence were a failure, ~93%
        of the corpus would arrive pre-flagged."""
        master = default_ifsc_master()
        unlisted = sorted(
            code for code in corpus_ifsc_codes() if master.lookup(code) is None
        )
        assert unlisted, "expected the synthetic master to be a partial list"
        hits = hits_for(["5. IFSC code", unlisted[0]], "ifsc_code")
        assert len(hits) == 1
        assert hits[0].validator_passed is True
        assert FieldFlag.IFSC_NOT_IN_MASTER not in hits[0].flags
        assert hits[0].confidence < 1.0  # unverified, and the score says so

    def test_the_real_list_is_a_path_change_not_a_code_change(
        self, tmp_path: Path
    ) -> None:
        """The drop-in contract: hand :meth:`IFSCMaster.load` an RBI-style CSV and
        the same validators become authoritative, with no edit anywhere else."""
        target = tmp_path / "RBI_IFSC.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["IFSC", "BANK", "BRANCH", "CITY", "STATE"])
            writer.writerow(
                ["SBIN0152060", "State Bank of India", "PANVEL", "PANVEL", "MH"]
            )
        real = IFSCMaster.load(target)
        assert real.authoritative is True
        listed = hits_for(
            ["5. IFSC code", "SBIN0152060"], "ifsc_code", ifsc_master=real
        )
        assert listed[0].validator_passed is True
        unlisted = hits_for(
            ["5. IFSC code", "SBIN0999999"], "ifsc_code", ifsc_master=real
        )
        assert unlisted[0].validator_passed is False
        assert unlisted[0].normalized is None
