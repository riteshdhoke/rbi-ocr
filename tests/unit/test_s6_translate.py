"""Unit tests for entity-masked per-field translation (WS-E3, BUILD-BRIEF M5).

The mandatory test of this module is
:meth:`TestSentinelRoundTrip.test_a_marathi_field_round_trips_an_amount_and_an_account`
— BUILD-BRIEF M5: *"a Marathi field containing ₹45,000 and a 14-digit account
number round-trips both byte-identically through translation"*.  Everything else
exists to make that guarantee non-accidental:

* the digits never reach the engine (they are replaced by sentinels), so the
  round trip cannot depend on a 200M-parameter model behaving;
* what is written back is the **stored original**, so an engine that pads,
  re-spaces or re-cases a placeholder still restores byte-identical text;
* an engine that *loses* a sentinel produces no ``value_en`` at all.  There is no
  partial restore, because a translation missing one identifier reads exactly as
  trustworthy as one missing none.

Invariant 10 is the other subject: ``value`` is immutable, ``value_en`` is
advisory and cannot exist without ``mt_engine``.

All doubles, no weights — except the one test marked ``local_models``, which
runs a real Hindi->English field translation through
:class:`app.providers.local.LocalProvider` and skips cleanly when ``models/mt``
is empty.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.contracts.envelope import Evidence, Field, FieldFlag
from app.contracts.errors import ProviderUnavailable
from app.contracts.ocr import SUMMARY_FIELD_NAME, TextSpan, bbox_from_xyxy
from app.stages.s6_translate import (
    DEFAULT_PROTECT_PATTERNS,
    MT_SENTINEL_TEMPLATE,
    TRANSLATABLE_FIELDS,
    SentinelCollision,
    SentinelRestoreFailed,
    _sentinel,
    mask_entities,
    restore_entities,
    translate_fields,
    translate_text,
)

#: The translator doubles below have to speak the exact wire shape the module
#: emits, so they use its own formatter.  Re-spelling ``ZQ0ZQ`` in the tests
#: would let the sentinel shape change while every double kept passing.
SENTINEL_RE = re.compile(r"[A-Z]+\d+[A-Z]+")

#: Matches a sentinel only when no further sentinel follows it.
_LAST_SENTINEL = re.compile(rf"{SENTINEL_RE.pattern}(?!.*{SENTINEL_RE.pattern})", re.DOTALL)

#: The BUILD-BRIEF M5 case, verbatim in shape: a Marathi sentence carrying a
#: rupee amount and a 14-digit account number.
MARATHI_FIELD = (
    "माझ्या खात्यातून ₹45,000 डेबिट झाले, "
    "खाते क्रमांक 50100234567890 आहे."
)
AMOUNT = "₹45,000"
ACCOUNT = "50100234567890"

MT_ENGINE = "indictrans2-indic-en-dist-200M-ct2-int8"


# --------------------------------------------------------------------------- #
# Translator doubles
# --------------------------------------------------------------------------- #


class EchoTranslator:
    """Returns its input untouched — the sentinels survive trivially."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str, str]] = []

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        self.calls.append((list(texts), src, tgt))
        return list(texts)


class ParaphrasingTranslator:
    """Rewrites everything except the sentinels, reorders and re-spaces them.

    Real engines do exactly this: they move a placeholder to wherever English
    word order puts it and pad it with spaces.  Tolerating that while still
    restoring the *stored* original is the point of
    :func:`~app.stages.s6_translate._restore_pattern`.
    """

    def __init__(self, template: str = "amount {0} debited from account {1}") -> None:
        self.template = template
        self.calls: list[list[str]] = []

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        self.calls.append(list(texts))
        output: list[str] = []
        for text in texts:
            found = SENTINEL_RE.findall(text)
            spaced = [
                re.sub(r"(\d+)", r" \1 ", sentinel) for sentinel in found
            ]
            output.append(
                self.template.format(*spaced)
                if len(spaced) >= self.template.count("{")
                else " ".join(["translated", *spaced])
            )
        return output


class SentinelDroppingTranslator:
    """Loses the last sentinel — the failure mode that must discard the output."""

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        return [_LAST_SENTINEL.sub("", text) for text in texts]


class UnavailableTranslator:
    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        raise ProviderUnavailable("no MT engine on this node")


class MiscountingTranslator:
    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        return list(texts)[:-1]


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #


def make_field(value: str, **overrides: object) -> Field:
    data: dict[str, object] = {
        "value": value,
        "confidence": 0.9,
        "source": "slm",
        "evidence": Evidence(
            page_no=1,
            bbox=bbox_from_xyxy(100.0, 100.0, 900.0, 132.0),
            span=TextSpan(line_index=0, start=0, end=len(value)),
        ),
    }
    data.update(overrides)
    return Field(**data)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #


class TestMasking:
    def test_the_amount_and_the_account_are_both_masked(self) -> None:
        mask = mask_entities(MARATHI_FIELD)
        assert mask.tokens == (AMOUNT, ACCOUNT)
        assert mask.text == (
            f"माझ्या खात्यातून {_sentinel(mask.marker, 0)} डेबिट झाले, "
            f"खाते क्रमांक {_sentinel(mask.marker, 1)} आहे."
        )

    def test_the_sentinel_carries_no_delimiter_punctuation(self) -> None:
        """The measured constraint (D-E3-3): IndicTrans2 splits on bracket-style
        delimiters and mangles what is inside, so the placeholder is a bare
        alphanumeric run or it does not survive at all."""
        rendered = MT_SENTINEL_TEMPLATE.format(index=0)
        assert rendered.isalnum()
        assert rendered == _sentinel("ZQ", 0)

    def test_no_digit_survives_into_the_masked_text(self) -> None:
        """The whole defence: a sequence model cannot rewrite what it never sees."""
        masked = mask_entities(MARATHI_FIELD).text
        digits = [c for c in masked if c.isdigit()]
        assert digits == list("01")  # only the sentinel indices remain

    def test_sentinels_are_numbered_left_to_right(self) -> None:
        """Deterministic masked text keeps the provider response cache coherent."""
        mask = mask_entities("A 111111 B 222222 C 333333")
        expected = " ".join(
            f"{letter} {_sentinel(mask.marker, index)}"
            for index, letter in enumerate("ABC")
        )
        assert mask.text == expected
        assert mask.tokens == ("111111", "222222", "333333")

    def test_a_single_digit_index_is_not_a_prefix_of_a_double_digit_one(self) -> None:
        """Restore-by-replacement is order-independent only because of this."""
        text = " x ".join(f"{n:06d}" for n in range(12))
        mask = mask_entities(text)
        assert len(mask.tokens) == 12
        assert restore_entities(mask.text, mask) == text

    def test_masking_is_deterministic(self) -> None:
        assert mask_entities(MARATHI_FIELD) == mask_entities(MARATHI_FIELD)

    def test_indic_digits_are_masked_too(self) -> None:
        """The masker and the validators share one definition of "digit"."""
        mask = mask_entities("माझ्या खात्यातून ₹४५,००० डेबिट झाले")
        assert mask.tokens == ("₹४५,०००",)

    @pytest.mark.parametrize(
        "text",
        [
            "IFSC BARB0152449 please",
            "UTR AXIS0123456789012 refers",
            "dated 14/01/2026 sir",
            "A/c 50100234567890",
            "Rs. 1,00,000 debited",
            "charge of 45.50 applied",
        ],
    )
    def test_every_protected_shape_is_recognised(self, text: str) -> None:
        assert mask_entities(text).masked_count == 1

    def test_a_validator_hit_is_masked_even_when_no_pattern_matches(self) -> None:
        """``protected_values`` is how a value the patterns would miss is still
        protected — over-masking costs a stiffer translation, under-masking costs
        a corrupted identifier."""
        text = "Ref ABCD/22 for the complaint"
        assert mask_entities(text).masked_count == 0
        mask = mask_entities(text, protected_values=["ABCD/22"])
        assert mask.tokens == ("ABCD/22",)

    def test_masked_spans_never_overlap(self) -> None:
        """Overlapping spans would corrupt each other's offsets."""
        mask = mask_entities("Rs. 45,000 on 14/01/2026 to 50100234567890")
        assert restore_entities(mask.text, mask) == (
            "Rs. 45,000 on 14/01/2026 to 50100234567890"
        )
        assert len(mask.tokens) == len(set(mask.tokens))

    def test_a_marker_already_in_the_text_is_avoided(self) -> None:
        source = f"literal {_sentinel('ZQ', 0)} and 50100234567890"
        mask = mask_entities(source)
        assert mask.marker == "ZQZQ"
        assert _sentinel("ZQZQ", 0) in mask.text
        assert restore_entities(mask.text, mask) == source

    def test_exhausting_every_marker_raises_rather_than_guessing(self) -> None:
        """An ambiguous restore could splice an identifier into the wrong slot."""
        crafted = "ZQ" * 8
        with pytest.raises(SentinelCollision):
            mask_entities(crafted + " 50100234567890")

    def test_empty_text_is_not_a_failure(self) -> None:
        mask = mask_entities("")
        assert mask.text == ""
        assert mask.tokens == ()

    def test_patterns_are_exported_for_tuning(self) -> None:
        assert DEFAULT_PROTECT_PATTERNS
        assert all(hasattr(p, "finditer") for p in DEFAULT_PROTECT_PATTERNS)


# --------------------------------------------------------------------------- #
# The mandatory round trip
# --------------------------------------------------------------------------- #


class TestSentinelRoundTrip:
    def test_a_marathi_field_round_trips_an_amount_and_an_account(self) -> None:
        """BUILD-BRIEF M5's mandatory test, byte-identically, through a model
        that rewrites every word it is given and re-spaces the placeholders."""
        provider = ParaphrasingTranslator()
        [translated] = translate_text(provider, [MARATHI_FIELD], src="mar")

        assert translated is not None
        assert AMOUNT in translated
        assert ACCOUNT in translated
        # Byte-identical, not merely "the digits are all there somewhere".
        assert translated.count(AMOUNT) == 1
        assert translated.count(ACCOUNT) == 1
        # The engine never saw either.
        sent = provider.calls[0][0]
        assert AMOUNT not in sent
        assert ACCOUNT not in sent

    def test_an_untouched_translation_restores_the_source_exactly(self) -> None:
        [translated] = translate_text(EchoTranslator(), [MARATHI_FIELD], src="mar")
        assert translated == MARATHI_FIELD

    def test_restore_is_the_exact_inverse_of_mask(self) -> None:
        for text in (
            MARATHI_FIELD,
            "Rs. 45,000 debited on 14/01/2026",
            "no entities here at all",
            "",
        ):
            mask = mask_entities(text)
            assert restore_entities(mask.text, mask) == text

    def test_a_lost_sentinel_raises_rather_than_half_restoring(self) -> None:
        mask = mask_entities(MARATHI_FIELD)
        damaged = mask.text.replace(_sentinel(mask.marker, 1), "")
        with pytest.raises(SentinelRestoreFailed) as failure:
            restore_entities(damaged, mask)
        assert failure.value.context["sentinel_index"] == 1
        assert failure.value.code == "SENTINEL_RESTORE_FAILED"

    def test_a_lost_sentinel_yields_none_not_a_mangled_string(self) -> None:
        results = translate_text(
            SentinelDroppingTranslator(), [MARATHI_FIELD], src="mar"
        )
        assert results == [None]

    def test_positions_are_preserved_across_a_mixed_batch(self) -> None:
        texts = ["ठीक आहे", MARATHI_FIELD, "आणखी काही नाही"]
        results = translate_text(EchoTranslator(), texts, src="mar")
        assert len(results) == len(texts)
        assert results[1] == MARATHI_FIELD

    def test_a_provider_that_loses_a_row_is_a_hard_error(self) -> None:
        with pytest.raises(ValueError, match="positions must be preserved"):
            translate_text(MiscountingTranslator(), ["a", "b"], src="mar")


# --------------------------------------------------------------------------- #
# Field-level translation — invariant 10
# --------------------------------------------------------------------------- #


class TestTranslateFields:
    def test_value_is_immutable_and_value_en_is_tagged(self) -> None:
        fields = {"relief_sought": make_field(MARATHI_FIELD)}
        result = translate_fields(
            ParaphrasingTranslator(),
            fields,
            src_language="mar",
            mt_engine=MT_ENGINE,
        )
        translated = result["relief_sought"]
        assert translated.value == MARATHI_FIELD
        assert translated.value_en is not None
        assert translated.mt_engine == MT_ENGINE
        assert AMOUNT in translated.value_en
        assert ACCOUNT in translated.value_en
        # The input map is untouched: translation returns a new map.
        assert fields["relief_sought"].value_en is None

    def test_an_empty_engine_tag_is_refused(self) -> None:
        """A blank tag is worse than none: it claims provenance it cannot show."""
        with pytest.raises(ValueError, match="mt_engine is mandatory"):
            translate_fields(
                EchoTranslator(),
                {"relief_sought": make_field("काही")},
                src_language="mar",
                mt_engine="   ",
            )

    def test_identifier_fields_are_never_translated(self) -> None:
        """"BARB0417293" has no English rendering, and offering one would invite
        a consumer to read ``value_en`` where only ``value`` is authoritative."""
        for name in ("ifsc_code", "account_number", "amount", "utr_rrn"):
            assert name not in TRANSLATABLE_FIELDS
        provider = EchoTranslator()
        fields = {
            "ifsc_code": make_field("BARB0152449", source="regex_validator"),
            "amount": make_field("₹45,000", source="regex_validator"),
        }
        result = translate_fields(
            provider, fields, src_language="mar", mt_engine=MT_ENGINE
        )
        assert provider.calls == []
        assert all(field.value_en is None for field in result.values())

    def test_the_summary_is_never_translated(self) -> None:
        """It is a verbatim concatenation of OCR spans; an English rendering
        would not survive the contract's re-derivation (invariant 2)."""
        assert SUMMARY_FIELD_NAME not in TRANSLATABLE_FIELDS

    def test_an_english_document_is_not_round_tripped_through_an_engine(self) -> None:
        provider = EchoTranslator()
        result = translate_fields(
            provider,
            {"relief_sought": make_field("Please refund the amount")},
            src_language="eng",
            mt_engine=MT_ENGINE,
        )
        assert provider.calls == []
        assert result["relief_sought"].value_en is None
        assert result["relief_sought"].mt_engine is None

    def test_a_pure_identifier_value_is_skipped(self) -> None:
        """Masking leaves nothing but sentinels — there is no text to translate."""
        provider = EchoTranslator()
        result = translate_fields(
            provider,
            {"prior_complaint_reference": make_field("50100234567890")},
            src_language="mar",
            mt_engine=MT_ENGINE,
        )
        assert provider.calls == []
        assert result["prior_complaint_reference"].value_en is None

    def test_an_existing_translation_is_left_alone(self) -> None:
        provider = EchoTranslator()
        existing = make_field("काही").with_translation("something", "other-engine")
        result = translate_fields(
            provider,
            {"relief_sought": existing},
            src_language="mar",
            mt_engine=MT_ENGINE,
        )
        assert provider.calls == []
        assert result["relief_sought"].mt_engine == "other-engine"

    def test_an_unavailable_engine_flags_rather_than_fails_the_document(self) -> None:
        """Translation is advisory; a missing engine must not cost a document its
        extracted identifiers."""
        result = translate_fields(
            UnavailableTranslator(),
            {"relief_sought": make_field(MARATHI_FIELD)},
            src_language="mar",
            mt_engine=MT_ENGINE,
        )
        field = result["relief_sought"]
        assert field.value == MARATHI_FIELD
        assert field.value_en is None
        assert FieldFlag.TRANSLATION_UNAVAILABLE in field.flags

    def test_strict_mode_re_raises_the_provider_error(self) -> None:
        with pytest.raises(ProviderUnavailable):
            translate_fields(
                UnavailableTranslator(),
                {"relief_sought": make_field(MARATHI_FIELD)},
                src_language="mar",
                mt_engine=MT_ENGINE,
                strict=True,
            )

    def test_a_lost_sentinel_flags_the_field_and_publishes_no_translation(
        self,
    ) -> None:
        result = translate_fields(
            SentinelDroppingTranslator(),
            {"relief_sought": make_field(MARATHI_FIELD)},
            src_language="mar",
            mt_engine=MT_ENGINE,
        )
        field = result["relief_sought"]
        assert field.value == MARATHI_FIELD
        assert field.value_en is None
        assert field.mt_engine is None
        assert FieldFlag.TRANSLATION_UNAVAILABLE in field.flags

    def test_validator_values_can_be_protected_explicitly(self) -> None:
        provider = ParaphrasingTranslator(template="reference {0} noted")
        result = translate_fields(
            provider,
            {"relief_sought": make_field("कृपया ABCD/22 पहा")},
            src_language="mar",
            mt_engine=MT_ENGINE,
            protected_values=["ABCD/22"],
        )
        assert "ABCD/22" in (result["relief_sought"].value_en or "")

    def test_a_provider_that_loses_a_field_is_a_hard_error(self) -> None:
        with pytest.raises(ValueError, match="positions must be preserved"):
            translate_fields(
                MiscountingTranslator(),
                {
                    "relief_sought": make_field("कृपया परतावा द्या"),
                    "complainant_address": make_field("मुंबई शहर"),
                },
                src_language="mar",
                mt_engine=MT_ENGINE,
            )

    def test_the_engine_tag_is_the_providers_canonical_name(self) -> None:
        """A consumer must be able to tell IndicTrans2 apart from an LLM
        paraphrase; that only works if the tag is not invented per call site."""
        from app.providers.local import MT_ENGINE_NAME

        assert MT_ENGINE_NAME == MT_ENGINE


# --------------------------------------------------------------------------- #
# Real weights — opt-in, skipped on a machine with an empty models/
# --------------------------------------------------------------------------- #


def _mt_model_dir() -> Path | None:
    """The CTranslate2 IndicTrans2 directory, if this machine actually has it."""
    from app.providers.local import (
        MT_CT2_DIRNAME,
        MT_CT2_MODEL_FILENAME,
        MT_SPM_SRC_FILENAME,
        MT_SPM_TGT_FILENAME,
        MT_SUBDIR,
    )
    from app.settings import Settings

    directory = Settings(_env_file=None).models_dir / MT_SUBDIR / MT_CT2_DIRNAME
    required = (
        MT_CT2_MODEL_FILENAME,
        MT_SPM_SRC_FILENAME,
        MT_SPM_TGT_FILENAME,
    )
    if all((directory / name).is_file() for name in required):
        return directory
    return None


@pytest.mark.local_models
def test_hindi_field_translation_through_the_local_provider() -> None:
    """End-to-end Stage 6 over the real IndicTrans2 CT2 INT8 weights.

    Everything above this line proves the *contract* with a double.  This proves
    the contract holds against the engine that will actually run: a real
    Hindi->English field translation, with the amount and the account number
    masked, translated and restored.

    Skipped — never failed — when ``models/mt`` is empty, which is the state of a
    fresh clone and of CI (PLAN.md: the suite must pass with no weights present).
    """
    if _mt_model_dir() is None:
        pytest.skip(
            "models/mt/indic-en is absent; fetch it with tools/fetch_models.py "
            "to run the local-weights integration test"
        )
    pytest.importorskip("ctranslate2")
    pytest.importorskip("sentencepiece")

    from app.providers.local import MT_ENGINE_NAME, LocalProvider
    from app.settings import Settings

    provider = LocalProvider(Settings(_env_file=None))
    hindi = (
        "मेरे खाते से ₹45,000 गलत तरीके से काट लिए गए, "
        "खाता संख्या 50100234567890 है।"
    )
    result = translate_fields(
        provider,
        {"relief_sought": make_field(hindi)},
        src_language="hin",
        mt_engine=MT_ENGINE_NAME,
    )
    field = result["relief_sought"]

    # Invariant 10, against a real engine: the native value is byte-identical.
    assert field.value == hindi
    assert field.value_en is not None, (
        "IndicTrans2 dropped a sentinel; the stage correctly discarded the "
        "translation, but this fixture is meant to survive it"
    )
    assert field.mt_engine == MT_ENGINE_NAME
    # The masked entities came back byte-identically through a real model.
    assert AMOUNT in field.value_en
    assert ACCOUNT in field.value_en
    # And something was actually translated, rather than the source echoed.
    assert field.value_en != hindi
    assert any(character.isascii() and character.isalpha() for character in field.value_en)
