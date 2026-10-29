"""Unit tests for envelope assembly (WS-E3, BUILD-BRIEF M6).

``assemble.py`` is the single writer of everything derived, so these tests are
mostly about *provenance of derivation*:

**Invariant 1 — language is derived, never carried.**
:class:`~app.contracts.ocr.OCRPage` refuses a page-level language;
``PageResult.primary_language`` exists only because
:func:`~app.stages.assemble.derive_language_summary` computed it from the
line-level labels Stage 2b assigned.  The bilingual-form case (English printed
labels, handwritten Marathi values) is the one that has to come out right.

**Invariant 2 — the assembler writes the summary, from indices.**  The model
selected line numbers; the text in the envelope is looked up and concatenated
here, and :class:`~app.contracts.envelope.ComplaintSummary` re-derives the
concatenation and rejects a mismatch.

**Invariant 8 — the gate is inside the assembler.**  There is no route from a
provider's answer into a published :class:`~app.contracts.envelope.Field` that
skips :func:`~app.stages.s5_slm.apply_grounding_gate`, so a fabricated value
becomes a machine-readable warning rather than content.

The schema check validates a *really assembled* envelope against the committed
``docs/schema/result_envelope.schema.json``.  It uses the small validator in
this module rather than the ``jsonschema`` package, for the reason recorded in
``tests/unit/test_contracts_schema.py``: pydantic is the validator of record and
a runtime dependency added to assert a property of our own generator is a
licence and supply-chain cost for no coverage.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.contracts.envelope import (
    BounceDecision,
    BounceReason,
    Evidence,
    Field,
    FieldFlag,
    ProcessingStats,
    QualityFlag,
    QualityVerdict,
    ResultEnvelope,
)
from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    OCRLine,
    OCRPage,
    OCRTier,
    PageClass,
    PageClassLabel,
    RawField,
    TextSpan,
    bbox_from_xyxy,
)
from app.settings import Settings
from app.stages.assemble import (
    GROUNDING_WARNING_PREFIX,
    PageAssembly,
    assemble_envelope,
    build_complaint_summary,
    build_page_result,
    derive_language_summary,
    merge_field_candidates,
    utc_now,
)
from app.stages.s5_validators import run_validators

NAME_LINE = "Ramesh Kumar Deshmukh"


# --------------------------------------------------------------------------- #
# A minimal JSON Schema validator — see the module docstring for why
# --------------------------------------------------------------------------- #

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    expected = _JSON_TYPES.get(name)
    if expected is None:  # pragma: no cover - unknown keyword is a test bug
        raise AssertionError(f"unhandled JSON Schema type {name!r}")
    return isinstance(value, expected) and not isinstance(value, bool)


def validate_against_schema(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$"
) -> list[str]:
    """Validate ``value`` against the subset of JSON Schema pydantic emits.

    Handles ``$ref``/``$defs``, ``anyOf``, ``type``, ``enum``, ``const``,
    ``properties``, ``required``, ``additionalProperties: false``, ``items`` and
    the numeric/length bounds.  ``tests/unit/test_contracts_schema.py`` asserts
    the committed artefact uses nothing else, so an unhandled keyword raises
    rather than silently passing.
    """
    errors: list[str] = []
    if "$ref" in schema:
        target = schema["$ref"].removeprefix("#/$defs/")
        return validate_against_schema(value, root["$defs"][target], root, path)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if not validate_against_schema(value, option, root, path):
                return []
        return [f"{path}: matches no branch of anyOf"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    declared = schema.get("type")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_type_ok(value, name) for name in names):
            errors.append(f"{path}: {type(value).__name__} is not {names}")
            return errors
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected property {name!r}")
        for name, child in value.items():
            if name in properties:
                errors.extend(
                    validate_against_schema(
                        child, properties[name], root, f"{path}.{name}"
                    )
                )
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(
                validate_against_schema(
                    item, schema["items"], root, f"{path}[{index}]"
                )
            )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: shorter than minLength")
    if isinstance(value, list) and len(value) < schema.get("minItems", 0):
        errors.append(f"{path}: fewer than minItems")
    return errors


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_line(text: str, index: int, **overrides: Any) -> OCRLine:
    data: dict[str, Any] = {
        "text": text,
        "bbox": bbox_from_xyxy(100.0, 100.0 + 40 * index, 900.0, 132.0 + 40 * index),
        "confidence": 0.94,
        "script": "Latn",
        "language": "eng",
        "is_handwritten": False,
        "line_index": index,
    }
    data.update(overrides)
    return OCRLine(**data)


def make_page(lines: Sequence[OCRLine], page_no: int = 1) -> OCRPage:
    return OCRPage(
        page_no=page_no,
        width_px=1654,
        height_px=2339,
        lines=list(lines),
        tier=OCRTier.TIER1,
    )


@pytest.fixture
def bilingual_page() -> OCRPage:
    """BUILD-BRIEF M4's hard case: printed English labels, handwritten Marathi."""
    return make_page(
        [
            make_line("1. Name of complainant", 0),
            make_line(
                "रमेश कुमार देशमुख",
                1,
                script="Deva",
                language="mar",
                is_handwritten=True,
            ),
            make_line("2. Nature of complaint", 2),
            make_line(
                "खात्यातून दुप्पट रक्कम कापली गेली",
                3,
                script="Deva",
                language="mar",
                is_handwritten=True,
            ),
        ]
    )


@pytest.fixture
def latin_page() -> OCRPage:
    return make_page(
        [
            make_line("1. Name of complainant", 0),
            make_line(NAME_LINE, 1),
            make_line("The bank debited twice", 2),
            make_line("on 14 January 2026.", 3),
            make_line("IFSC code BARB0152449", 4),
        ]
    )


@pytest.fixture
def schema_document(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def make_field(value: str, *, page_no: int = 1, line_index: int = 0, **kw: Any) -> Field:
    data: dict[str, Any] = {
        "value": value,
        "confidence": 0.9,
        "source": "slm",
        "evidence": Evidence(
            page_no=page_no,
            bbox=bbox_from_xyxy(100.0, 100.0, 900.0, 132.0),
            span=TextSpan(line_index=line_index, start=0, end=len(value)),
        ),
    }
    data.update(kw)
    return Field(**data)


# --------------------------------------------------------------------------- #
# Invariant 1 — derived language
# --------------------------------------------------------------------------- #


class TestDerivedLanguage:
    def test_a_page_may_not_carry_a_language_of_its_own(self) -> None:
        """The bug design-review A2 reversed, still reversed."""
        with pytest.raises(ValueError, match="must not carry page-level"):
            OCRPage(
                page_no=1,
                width_px=100,
                height_px=100,
                lines=[],
                tier=OCRTier.NONE,
                primary_language="eng",
            )

    def test_the_page_summary_is_computed_from_line_labels(
        self, bilingual_page: OCRPage
    ) -> None:
        summary = derive_language_summary(
            bilingual_page.lines, code_mix_threshold=0.15
        )
        assert summary.language_distribution == {"eng": 0.5, "mar": 0.5}
        assert summary.script_distribution == {"Latn": 0.5, "Deva": 0.5}
        assert summary.primary_language == "eng"  # tie, broken lexicographically
        assert summary.is_code_mixed is True
        assert summary.unlabelled_lines == 0

    def test_code_mixing_keys_on_script_not_language(self) -> None:
        """PRD P0-5: an English label beside a Marathi value is code-mixed whether
        or not the LID managed to label either line."""
        page = make_page(
            [
                make_line("Name", 0),
                make_line("नाव", 1, script="Deva", language=None),
                make_line("Address", 2),
                make_line("Bank", 3),
            ]
        )
        summary = derive_language_summary(page.lines, code_mix_threshold=0.15)
        assert summary.is_code_mixed is True
        assert summary.language_distribution == {"eng": 1.0}
        assert summary.unlabelled_lines == 1

    def test_an_unlabelled_line_is_reported_not_folded_into_the_majority(
        self,
    ) -> None:
        """Stage 2b returning ``None`` is a refusal to choose; counting it as
        agreement would manufacture a label the LID declined to give."""
        page = make_page(
            [make_line("Name", 0), make_line("???", 1, language=None)]
        )
        summary = derive_language_summary(page.lines, code_mix_threshold=0.15)
        assert summary.unlabelled_lines == 1
        assert summary.language_distribution == {"eng": 1.0}

    @pytest.mark.parametrize("count", [3, 6, 7, 11])
    def test_distributions_sum_to_exactly_one(self, count: int) -> None:
        """Thirds do not sum to 1.0 in binary float, and the contract checks to
        1e-6; the residual has to be absorbed, not hoped away."""
        codes = (
            "hin", "mar", "eng", "kok", "nep",
            "tam", "tel", "ben", "guj", "kan", "mal",
        )
        lines = [
            make_line(f"line {index}", index, language=codes[index % count])
            for index in range(count)
        ]
        summary = derive_language_summary(lines, code_mix_threshold=0.15)
        assert math.fsum(summary.language_distribution.values()) == 1.0

    def test_no_lines_means_no_summary_rather_than_a_guess(self) -> None:
        summary = derive_language_summary([], code_mix_threshold=0.15)
        assert summary.primary_language is None
        assert summary.primary_script is None
        assert summary.language_distribution == {}
        assert summary.is_code_mixed is False

    def test_the_page_record_carries_the_derived_summary(
        self, bilingual_page: OCRPage
    ) -> None:
        record = build_page_result(
            PageAssembly(page=bilingual_page, quality_score=0.8),
            code_mix_threshold=0.15,
        )
        assert record.primary_language == "eng"
        assert record.language_distribution == {"eng": 0.5, "mar": 0.5}
        assert record.is_code_mixed is True
        assert record.line_count == 4
        assert record.ocr_tier is OCRTier.TIER1

    def test_the_document_summary_spans_every_page(
        self, bilingual_page: OCRPage, settings: Settings
    ) -> None:
        second = make_page(
            [
                make_line("अर्जदाराचा पत्ता", 0, script="Deva", language="mar"),
                make_line("नागपूर", 1, script="Deva", language="mar"),
            ],
            page_no=2,
        )
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(page=bilingual_page),
                PageAssembly(page=second),
            ],
            settings=settings,
        )
        assert envelope.document is not None
        assert envelope.document.primary_language == "mar"
        assert envelope.document.language_distribution == {
            "eng": pytest.approx(1 / 3),
            "mar": pytest.approx(2 / 3),
        }
        assert envelope.document.is_code_mixed is True


# --------------------------------------------------------------------------- #
# Invariant 2 — the assembler writes the summary
# --------------------------------------------------------------------------- #


class TestComplaintSummary:
    def test_selected_lines_are_concatenated_verbatim(
        self, latin_page: OCRPage
    ) -> None:
        summary = build_complaint_summary([(1, [2, 3])], {1: latin_page})
        assert summary is not None
        assert summary.text == "The bank debited twice\non 14 January 2026."
        assert [span.text for span in summary.spans] == [
            "The bank debited twice",
            "on 14 January 2026.",
        ]

    def test_every_span_carries_evidence_pointing_at_its_line(
        self, latin_page: OCRPage
    ) -> None:
        summary = build_complaint_summary([(1, [2, 3])], {1: latin_page})
        assert summary is not None
        for span in summary.spans:
            line = latin_page.line_by_index(span.evidence.span.line_index)
            assert span.text == line.text
            assert span.evidence.span.end == len(line.text)
            assert span.evidence.bbox == list(line.bbox)

    def test_the_models_order_within_a_page_is_kept(
        self, latin_page: OCRPage
    ) -> None:
        """Re-sorting would scramble a narrative that runs down the page."""
        summary = build_complaint_summary([(1, [3, 2])], {1: latin_page})
        assert summary is not None
        assert summary.text == "on 14 January 2026.\nThe bank debited twice"

    def test_pages_are_concatenated_in_ascending_order(
        self, latin_page: OCRPage
    ) -> None:
        second = make_page([make_line("Continued overleaf.", 0)], page_no=2)
        summary = build_complaint_summary(
            [(2, [0]), (1, [2])], {1: latin_page, 2: second}
        )
        assert summary is not None
        assert summary.text == "The bank debited twice\nContinued overleaf."

    def test_nothing_selected_is_absence_not_an_empty_string(
        self, latin_page: OCRPage
    ) -> None:
        assert build_complaint_summary([], {1: latin_page}) is None
        assert build_complaint_summary([(1, [])], {1: latin_page}) is None

    def test_empty_lines_are_skipped(self) -> None:
        page = make_page([make_line("", 0), make_line("Real text", 1)])
        summary = build_complaint_summary([(1, [0, 1])], {1: page})
        assert summary is not None
        assert summary.text == "Real text"
        assert len(summary.spans) == 1

    def test_a_selection_naming_a_missing_line_is_a_caller_bug(
        self, latin_page: OCRPage
    ) -> None:
        with pytest.raises(KeyError):
            build_complaint_summary([(1, [99])], {1: latin_page})

    def test_the_assembler_resolves_indices_end_to_end(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        """The model returned two integers; the envelope carries the OCR text."""
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    raw_fields={
                        SUMMARY_FIELD_NAME: RawField(
                            name=SUMMARY_FIELD_NAME,
                            confidence=0.8,
                            line_indices=[2, 3],
                        )
                    },
                )
            ],
            settings=settings,
        )
        assert envelope.document is not None
        summary = envelope.document.complaint_summary
        assert summary is not None
        assert summary.text == "The bank debited twice\non 14 January 2026."
        assert SUMMARY_FIELD_NAME not in envelope.document.fields

    def test_a_prose_summary_cannot_reach_the_assembler(self) -> None:
        with pytest.raises(ValueError, match="not prose"):
            RawField(
                name=SUMMARY_FIELD_NAME,
                value="The complainant was debited twice.",
                confidence=0.8,
                line_indices=[2],
            )


# --------------------------------------------------------------------------- #
# Invariant 8 — the gate runs here, and nowhere else
# --------------------------------------------------------------------------- #


class TestGroundingInsideTheAssembler:
    def test_a_fabricated_value_becomes_a_warning_not_a_field(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    raw_fields={
                        "complainant_address": RawField(
                            name="complainant_address",
                            value="Plot 17, Shanti Nagar, Pune 411001",
                            confidence=0.95,
                        )
                    },
                )
            ],
            settings=settings,
        )
        assert envelope.document is not None
        assert "complainant_address" not in envelope.document.fields
        assert envelope.warnings == (
            (
                f"{GROUNDING_WARNING_PREFIX}:page=1"
                ":field=complainant_address:reason=ungrounded"
            ),
        )
        assert envelope.escalation_flag is True

    def test_the_warning_never_repeats_the_unverified_string(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        """The warning list is the part of the envelope a log aggregator prints."""
        fabricated = "Plot 17, Shanti Nagar, Pune 411001"
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    raw_fields={
                        "complainant_address": RawField(
                            name="complainant_address",
                            value=fabricated,
                            confidence=0.95,
                        )
                    },
                )
            ],
            settings=settings,
        )
        assert all(fabricated not in warning for warning in envelope.warnings)

    def test_a_grounded_value_is_published_with_slm_provenance(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    raw_fields={
                        "complainant_name": RawField(
                            name="complainant_name",
                            value=NAME_LINE,
                            confidence=0.95,
                            line_indices=[1],
                        )
                    },
                )
            ],
            settings=settings,
        )
        assert envelope.document is not None
        field = envelope.document.fields["complainant_name"]
        assert field.value == NAME_LINE
        assert field.source == "slm"
        assert field.evidence.span.line_index == 1
        assert envelope.warnings == ()

    def test_the_assembly_input_cannot_carry_pre_gated_fields(self) -> None:
        """Structural: there is no ``fields`` on :class:`PageAssembly`, so the
        only route in is ``raw_fields`` — which the gate reads."""
        assert "fields" not in PageAssembly.model_fields
        assert "raw_fields" in PageAssembly.model_fields


# --------------------------------------------------------------------------- #
# Field merging and provenance
# --------------------------------------------------------------------------- #


class TestFieldMerging:
    def test_highest_confidence_wins(self) -> None:
        low = make_field("A", confidence=0.4)
        high = make_field("B", confidence=0.9)
        assert merge_field_candidates({"x": low}, {"x": high})["x"] is high
        assert merge_field_candidates({"x": high}, {"x": low})["x"] is high

    def test_ties_cite_the_earliest_evidence(self) -> None:
        first = make_field("A", page_no=1, line_index=3)
        later = make_field("A", page_no=4, line_index=0)
        merged = merge_field_candidates({"x": later}, {"x": first})
        assert merged["x"].evidence.page_no == 1

    def test_validator_fields_and_slm_fields_coexist(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        report = run_validators([latin_page], max_workers=1)
        assert report.hits_for("ifsc_code")
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    raw_fields={
                        "complainant_name": RawField(
                            name="complainant_name",
                            value=NAME_LINE,
                            confidence=0.95,
                        )
                    },
                )
            ],
            validator_report=report,
            settings=settings,
        )
        assert envelope.document is not None
        fields = envelope.document.fields
        assert fields["ifsc_code"].value == "BARB0152449"
        assert fields["ifsc_code"].source == "regex_validator"
        assert fields["complainant_name"].source == "slm"

    def test_known_field_can_carry_unvalidated_slm_provenance(self) -> None:
        from app.contracts.envelope import DocumentResult

        document = DocumentResult(
            document_class=PageClassLabel.COMPLAINT_FORM,
            document_class_confidence=0.9,
            fields={"account_number": make_field("50100234567890")},
            page_count=1,
        )
        assert document.fields["account_number"].source == "slm"
        assert document.fields["account_number"].validator_passed is None


# --------------------------------------------------------------------------- #
# Translation hand-off — invariant 10
# --------------------------------------------------------------------------- #


class TestTranslatorHandoff:
    def test_the_translator_sees_the_merged_map_and_value_stays_native(
        self, bilingual_page: OCRPage, settings: Settings
    ) -> None:
        seen: list[dict[str, Field]] = []

        def translator(fields: dict[str, Field]) -> dict[str, Field]:
            seen.append(dict(fields))
            return {
                name: field.with_translation("Ramesh Kumar Deshmukh", "fake-mt")
                for name, field in fields.items()
            }

        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=bilingual_page,
                    raw_fields={
                        "complainant_name": RawField(
                            name="complainant_name",
                            value="रमेश कुमार देशमुख",
                            confidence=0.9,
                            line_indices=[1],
                        )
                    },
                )
            ],
            translator=translator,
            settings=settings,
        )
        assert envelope.document is not None
        field = envelope.document.fields["complainant_name"]
        assert field.value == "रमेश कुमार देशमुख"
        assert field.value_en == "Ramesh Kumar Deshmukh"
        assert field.mt_engine == "fake-mt"
        assert set(seen[0]) == {"complainant_name", "name_of_complainant", "nature_of_complaint"}

    def test_the_module_needs_no_mt_engine_to_be_tested(self) -> None:
        """``assemble`` imports no provider — the translator is a callable."""
        import app.stages.assemble as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "app.providers" not in source


# --------------------------------------------------------------------------- #
# Envelope shape, counters and refusals
# --------------------------------------------------------------------------- #


class TestEnvelope:
    def test_the_assembled_envelope_validates_against_the_committed_schema(
        self,
        latin_page: OCRPage,
        bilingual_page: OCRPage,
        schema_document: dict[str, Any],
        settings: Settings,
    ) -> None:
        """BUILD-BRIEF M6: what the pipeline actually emits is what consumers were
        promised — checked against the artefact in ``docs/schema/``, not against
        the models that generated it."""
        second = make_page(list(bilingual_page.lines), page_no=2)
        report = run_validators([latin_page], max_workers=1)
        envelope = assemble_envelope(
            document_id="sha256:" + "a" * 64,
            job_id="job-7",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    quality_score=0.88,
                    quality_flags=(QualityFlag.SKEW_REPAIRED,),
                    quality_verdict=QualityVerdict.REPAIR,
                    page_class=PageClass(
                        label=PageClassLabel.COMPLAINT_FORM, confidence=0.91
                    ),
                    repairs_applied=("deskew",),
                    raw_fields={
                        "complainant_name": RawField(
                            name="complainant_name",
                            value=NAME_LINE,
                            confidence=0.95,
                            line_indices=[1],
                        ),
                        SUMMARY_FIELD_NAME: RawField(
                            name=SUMMARY_FIELD_NAME,
                            confidence=0.8,
                            line_indices=[2, 3],
                        ),
                    },
                ),
                PageAssembly(
                    page=second,
                    escalated=True,
                    shadow_sampled=False,
                    page_class=PageClass(
                        label=PageClassLabel.COMPLAINT_FORM, confidence=0.77
                    ),
                ),
            ],
            validator_report=report,
            processing=ProcessingStats(started_at=utc_now(), finished_at=utc_now()),
            settings=settings,
        )
        payload = json.loads(envelope.model_dump_json())
        errors = validate_against_schema(payload, schema_document, schema_document)
        assert errors == []
        # And the artefact is not merely permissive: it round-trips.
        assert ResultEnvelope.model_validate(payload) == envelope

    def test_page_counters_are_recomputed_not_trusted(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        """The page records already say how many bounced; two answers to one
        question is one answer too many."""
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(page=latin_page),
                PageAssembly(page=make_page(list(latin_page.lines), 2), escalated=True),
            ],
            processing=ProcessingStats(
                pages_processed=99, pages_escalated=0, tokens_used=140
            ),
            settings=settings,
        )
        assert envelope.processing.pages_processed == 2
        assert envelope.processing.pages_escalated == 1
        assert envelope.processing.pages_bounced == 0
        assert envelope.processing.tokens_used == 140

    def test_an_escalated_page_sets_the_document_flag(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=latin_page, escalated=True)],
            settings=settings,
        )
        assert envelope.escalation_flag is True

    def test_a_shadow_sample_is_not_an_escalation(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        """Invariant 3: shadow runs must never inflate the rate they measure."""
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=latin_page, shadow_sampled=True)],
            settings=settings,
        )
        assert envelope.pages[0].shadow_sampled is True
        assert envelope.pages[0].escalated is False
        assert envelope.escalation_flag is False

    def test_a_bounced_document_carries_no_content(
        self, settings: Settings
    ) -> None:
        blank = OCRPage(
            page_no=1, width_px=100, height_px=100, lines=[], tier=OCRTier.NONE
        )
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=blank,
                    quality_verdict=QualityVerdict.BOUNCE,
                    quality_flags=(QualityFlag.BLANK_PAGE,),
                    quality_score=0.0,
                )
            ],
            bounce_decision=BounceDecision(
                decision=QualityVerdict.BOUNCE, reason=BounceReason.BLANK_PAGE
            ),
            settings=settings,
        )
        assert envelope.document is None
        assert envelope.bounce_decision.reason is BounceReason.BLANK_PAGE
        assert envelope.processing.pages_bounced == 1

    def test_duplicate_page_numbers_are_refused(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        with pytest.raises(ValueError, match="duplicate page numbers"):
            assemble_envelope(
                document_id="doc-1",
                provider_profile="laptop",
                pages=[PageAssembly(page=latin_page), PageAssembly(page=latin_page)],
                settings=settings,
            )

    def test_pages_are_ordered_regardless_of_input_order(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(page=make_page(list(latin_page.lines), 3)),
                PageAssembly(page=latin_page),
                PageAssembly(page=make_page(list(latin_page.lines), 2)),
            ],
            settings=settings,
        )
        assert [page.page_no for page in envelope.pages] == [1, 2, 3]

    def test_an_unclassified_document_says_unknown_rather_than_guessing(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=latin_page)],
            settings=settings,
        )
        assert envelope.document is not None
        assert envelope.document.document_class is PageClassLabel.UNKNOWN
        assert envelope.document.document_class_confidence == 0.0

    def test_the_document_class_is_a_confidence_weighted_page_vote(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[
                PageAssembly(
                    page=latin_page,
                    page_class=PageClass(
                        label=PageClassLabel.COMPLAINT_LETTER, confidence=0.9
                    ),
                ),
                PageAssembly(
                    page=make_page(list(latin_page.lines), 2),
                    page_class=PageClass(
                        label=PageClassLabel.SUPPORTING_DOC, confidence=0.3
                    ),
                ),
            ],
            settings=settings,
        )
        assert envelope.document is not None
        assert envelope.document.document_class is PageClassLabel.COMPLAINT_LETTER
        assert envelope.document.document_class_confidence == pytest.approx(0.75)

    def test_a_flagged_field_escalates_the_document(
        self, settings: Settings
    ) -> None:
        page = make_page(
            [make_line("Card 4111 1111 1111 1112 debited", 0)]
        )
        report = run_validators([page], max_workers=1)
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=page)],
            validator_report=report,
            settings=settings,
        )
        assert envelope.document is not None
        card = envelope.document.fields["card_fragment"]
        assert card.validator_passed is False
        assert FieldFlag.VALIDATOR_FAILED in card.flags
        assert card.normalized is None
        assert envelope.escalation_flag is True

    def test_versioning_is_the_callers_to_set(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        """Two-stage delivery: version 1 is published before escalated pages land."""
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=latin_page)],
            result_version=2,
            is_final=False,
            settings=settings,
        )
        assert envelope.result_version == 2
        assert envelope.is_final is False
        assert envelope.schema_version == "1.0.0"

    def test_duration_is_derived_when_only_timestamps_are_given(
        self, latin_page: OCRPage, settings: Settings
    ) -> None:
        started = utc_now()
        envelope = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=latin_page)],
            processing=ProcessingStats(started_at=started, finished_at=started),
            settings=settings,
        )
        assert envelope.processing.duration_ms == 0.0

    def test_code_mix_threshold_comes_from_settings(
        self, bilingual_page: OCRPage, make_settings: Callable[..., Settings]
    ) -> None:
        loose = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=bilingual_page)],
            settings=make_settings(CODE_MIX_THRESHOLD=0.15),
        )
        strict = assemble_envelope(
            document_id="doc-1",
            provider_profile="laptop",
            pages=[PageAssembly(page=bilingual_page)],
            settings=make_settings(CODE_MIX_THRESHOLD=0.9),
        )
        assert loose.pages[0].is_code_mixed is True
        assert strict.pages[0].is_code_mixed is False
