"""End-to-end tests for the synchronous orchestrator (WS-F, BUILD-BRIEF M6).

Three corpus documents carry the load, chosen for what they prove rather than
for coverage:

* ``complaint_form_en_v0_clean`` — the happy path.  Ingests, recognises,
  validates, extracts, assembles, and the envelope validates against the
  **committed** ``docs/schema/result_envelope.schema.json`` rather than against
  the models that produced it.
* ``complaint_form_en_v0_bounce`` — the short circuit.  The quality gate
  refuses it, no model is called at all (asserted by counting provider calls),
  and the envelope carries a machine-readable :class:`BounceReason` with no
  ``document`` on it.
* ``bilingual_form_v0_clean`` — the escalation.  A split script vote is the one
  signal that escalates on its own (``docs/decisions-wse2.md`` sec. 4c), so this
  document exercises two-stage delivery: ``result_version=1, is_final=false``
  then ``result_version=2, is_final=true``.

Everything runs against the deterministic :class:`MockProvider` from the
contract suite, so the whole file passes with an empty ``models/``.  The single
``local_models`` test at the end is the exception and skips cleanly when the
weights are absent.

The JSON Schema checker is imported from ``test_assemble`` rather than
duplicated: WS-E3 recorded the decision (``docs/decisions-wse3.md`` D-E3-5) not
to add a runtime ``jsonschema`` dependency to assert a property of our own
generator, and a second hand-rolled copy of that checker would be the thing
that decision was avoiding.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.contracts.envelope import (
    BounceReason,
    QualityVerdict,
    ResultEnvelope,
)
from app.contracts.errors import (
    ConfigurationError,
    ProviderUnavailable,
    ResidencyViolation,
)
from app.contracts.ocr import OCRLine, OCRPage, OCRTier, bbox_from_xyxy
from app.pipeline import (
    ESCALATION_PENDING_PREFIX,
    QUALITY_DEFECT_FLAGS,
    PageOutcome,
    PipelineResult,
    run,
    run_document,
)
from app.settings import ProviderProfile, Settings
from app.stages.s15_quality import map_bbox_to_original

_CONTRACT_DIR = Path(__file__).resolve().parents[1] / "contract"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from mock_provider import MockProvider
from test_assemble import validate_against_schema

CORPUS = Path(__file__).resolve().parents[2] / "corpus"

CLEAN_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_clean.jpg"
BOUNCE_DOC = CORPUS / "complaint_form_en" / "complaint_form_en_v0_bounce.jpg"
BILINGUAL_DOC = CORPUS / "bilingual_form" / "bilingual_form_v0_clean.jpg"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def pipeline_settings(make_settings: Callable[..., Settings]) -> Settings:
    """Laptop profile, synthetic only, shadow sampling off.

    The 3% sampler is disabled for every test that is not about it: a sampled
    page runs Tier 2, and a test asserting "this document did not escalate"
    would otherwise fail on roughly one run in thirty-three, on a different
    document each time.
    """
    return make_settings(ESCALATION_SHADOW_RATE=0.0)


@pytest.fixture
def provider() -> MockProvider:
    """A provider that answers every method deterministically."""
    return MockProvider(field_values={"complainant_name": "4. Name of bank"})


@pytest.fixture
def latin_provider() -> MockProvider:
    """A provider whose page is monolingual Latin — nothing to escalate on."""
    return MockProvider(
        lines=[
            OCRLine(
                text="Account number 30294817465021",
                bbox=bbox_from_xyxy(100.0, 100.0, 500.0, 140.0),
                confidence=0.95,
                script="Latn",
                language=None,
                is_handwritten=False,
                line_index=0,
            ),
            OCRLine(
                text="IFSC BARB0417293 branch Pune",
                bbox=bbox_from_xyxy(100.0, 160.0, 500.0, 200.0),
                confidence=0.93,
                script="Latn",
                language=None,
                is_handwritten=False,
                line_index=1,
            ),
        ]
    )


@pytest.fixture(scope="session")
def committed_schema() -> dict[str, object]:
    """The published JSON Schema artefact."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "schema"
        / "result_envelope.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def assert_schema_valid(
    envelope: ResultEnvelope, schema: dict[str, object]
) -> None:
    """Fail with every schema error, not just the first."""
    payload = json.loads(envelope.model_dump_json())
    errors = validate_against_schema(payload, schema, schema)
    assert errors == [], "\n".join(errors)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


class TestCleanDocument:
    """One clean corpus page, end to end."""

    def test_produces_a_schema_valid_envelope(
        self,
        provider: MockProvider,
        pipeline_settings: Settings,
        committed_schema: dict[str, object],
    ) -> None:
        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        assert_schema_valid(envelope, committed_schema)
        assert envelope.is_final is True
        assert envelope.document is not None
        assert envelope.bounce_decision.decision is QualityVerdict.ACCEPT

    def test_document_id_is_the_sha256_of_the_bytes(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        from app.stages.s1_ingest import idempotency_key

        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        assert envelope.document_id == idempotency_key(CLEAN_DOC.read_bytes())

    def test_the_same_document_twice_gives_the_same_document_id(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        first = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        second = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        assert first.document_id == second.document_id

    def test_page_language_is_derived_not_carried(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """Invariant 1: the summary exists only because assemble derived it."""
        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        page = envelope.pages[0]
        if page.language_distribution:
            assert page.primary_language in page.language_distribution
            assert abs(sum(page.language_distribution.values()) - 1.0) < 1e-6

    def test_records_a_timing_for_every_stage_that_ran(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        timings = envelope.processing.stage_timings_ms
        for stage in ("s1_ingest", "s15_quality", "s4_ocr", "s5_validators", "assemble"):
            assert stage in timings, f"no timing recorded for {stage}"
            assert timings[stage] >= 0.0

    def test_records_per_page_timings_for_the_eval_harness(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """The harness needs the page that cost the p95, not just the total."""
        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        keys = envelope.processing.stage_timings_ms
        assert "s4_ocr:p1" in keys
        assert keys["s4_ocr"] >= keys["s4_ocr:p1"]

    def test_counters_are_recomputed_from_the_page_records(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        envelope = run(CLEAN_DOC, provider=provider, settings=pipeline_settings)
        assert envelope.processing.pages_processed == len(envelope.pages)
        assert envelope.processing.pages_bounced == 0

    def test_deterministic_fields_never_carry_slm_provenance(
        self, latin_provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """Invariant 7, checked on a real run rather than on a constructed one."""
        envelope = run(
            CLEAN_DOC, provider=latin_provider, settings=pipeline_settings
        )
        assert envelope.document is not None
        from app.contracts.ocr import DETERMINISTIC_FIELDS

        for name, field in envelope.document.fields.items():
            if name in DETERMINISTIC_FIELDS:
                assert field.source != "slm"

    def test_every_field_carries_evidence(
        self, latin_provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        envelope = run(
            CLEAN_DOC, provider=latin_provider, settings=pipeline_settings
        )
        assert envelope.document is not None
        assert envelope.document.fields, "the run extracted nothing to check"
        for field in envelope.document.fields.values():
            assert field.evidence.page_no >= 1
            assert len(field.evidence.bbox) == 8


# --------------------------------------------------------------------------- #
# The bounce short circuit
# --------------------------------------------------------------------------- #


class TestBouncedDocument:
    """A page the quality gate refuses costs nothing downstream."""

    def test_bounces_with_a_machine_readable_reason(
        self,
        provider: MockProvider,
        pipeline_settings: Settings,
        committed_schema: dict[str, object],
    ) -> None:
        envelope = run(BOUNCE_DOC, provider=provider, settings=pipeline_settings)
        assert envelope.bounce_decision.decision is QualityVerdict.BOUNCE
        assert isinstance(envelope.bounce_decision.reason, BounceReason)
        assert_schema_valid(envelope, committed_schema)

    def test_a_bounced_document_carries_no_extracted_content(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """BUILD-BRIEF sec. 0.5: a missing result says why, never arrives empty."""
        envelope = run(BOUNCE_DOC, provider=provider, settings=pipeline_settings)
        assert envelope.document is None
        assert envelope.pages[0].line_count == 0

    def test_no_model_is_called_for_a_bounced_page(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """The short circuit is the point: a bounce must not bill inference."""
        run(BOUNCE_DOC, provider=provider, settings=pipeline_settings)
        assert provider.calls == []

    def test_the_bounce_is_final_at_version_one(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            BOUNCE_DOC, provider=provider, settings=pipeline_settings
        )
        assert len(result.envelopes) == 1
        assert result.final.result_version == 1
        assert result.final.is_final is True

    def test_an_unreadable_attachment_becomes_a_bounce_not_an_exception(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """PRD P0-3 makes "refused, and here is why" a product outcome."""
        envelope = run(
            (CORPUS / "adversarial" / "truncated.pdf"),
            provider=provider,
            settings=pipeline_settings,
        )
        assert envelope.bounce_decision.reason is BounceReason.MALFORMED_PDF
        assert envelope.document is None

    def test_strict_rejections_re_raises_instead(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        from app.contracts.errors import MalformedPDFError

        with pytest.raises(MalformedPDFError):
            run_document(
                (CORPUS / "adversarial" / "truncated.pdf"),
                provider=provider,
                settings=pipeline_settings,
                strict_rejections=True,
            )


# --------------------------------------------------------------------------- #
# Two-stage delivery
# --------------------------------------------------------------------------- #


class TestEscalation:
    """A split script vote escalates, and the fast path ships before Tier 2."""

    @pytest.fixture
    def escalating(self, provider: MockProvider) -> MockProvider:
        """Tier 2 that returns a *different* reading of the same boxes."""
        provider.heavy_lines = [
            line.model_copy(update={"text": line.text + " (tier2)"})
            for line in provider.lines
        ]
        return provider

    def test_publishes_two_versions(
        self, escalating: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            BILINGUAL_DOC, provider=escalating, settings=pipeline_settings
        )
        assert [e.result_version for e in result.envelopes] == [1, 2]

    def test_the_fast_path_is_not_final_and_names_the_pending_page(
        self, escalating: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            BILINGUAL_DOC, provider=escalating, settings=pipeline_settings
        )
        fast = result.fast_path
        assert fast.is_final is False
        pending = [
            warning
            for warning in fast.warnings
            if warning.startswith(ESCALATION_PENDING_PREFIX)
        ]
        assert pending, f"no pending warning in {fast.warnings}"
        assert pending[0].endswith("page=1")

    def test_the_final_version_is_final_and_marks_the_page_escalated(
        self, escalating: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            BILINGUAL_DOC, provider=escalating, settings=pipeline_settings
        )
        final = result.final
        assert final.is_final is True
        assert any(page.escalated for page in final.pages)
        assert final.escalation_flag is True

    def test_tier_two_text_reaches_the_published_envelope(
        self, escalating: MockProvider, pipeline_settings: Settings
    ) -> None:
        """Version 2 is a re-assembly, not a patch: the new text must be in it."""
        result = run_document(
            BILINGUAL_DOC, provider=escalating, settings=pipeline_settings
        )
        assert result.final.pages[0].ocr_tier.value == "tier2"

    def test_on_envelope_fires_once_per_version_in_order(
        self, escalating: MockProvider, pipeline_settings: Settings
    ) -> None:
        seen: list[int] = []
        run_document(
            BILINGUAL_DOC,
            provider=escalating,
            settings=pipeline_settings,
            on_envelope=lambda envelope: seen.append(envelope.result_version),
        )
        assert seen == [1, 2]

    def test_both_versions_are_schema_valid(
        self,
        escalating: MockProvider,
        pipeline_settings: Settings,
        committed_schema: dict[str, object],
    ) -> None:
        result = run_document(
            BILINGUAL_DOC, provider=escalating, settings=pipeline_settings
        )
        for envelope in result.envelopes:
            assert_schema_valid(envelope, committed_schema)

    def test_a_failed_tier_two_ships_the_page_at_tier_one(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        """``escalate_page`` returns a typed outcome; the document still ships."""
        provider.fail_with = {
            "ocr_page_heavy": ProviderUnavailable("VLM endpoint unreachable")
        }
        result = run_document(
            BILINGUAL_DOC, provider=provider, settings=pipeline_settings
        )
        assert result.final.is_final is True
        assert result.final.pages[0].escalated is False
        assert any("tier2_failed" in w for w in result.final.warnings)

    def test_a_monolingual_page_does_not_escalate(
        self, latin_provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            CLEAN_DOC, provider=latin_provider, settings=pipeline_settings
        )
        assert len(result.envelopes) == 1
        assert result.final.is_final is True
        assert "ocr_page_heavy" not in [name for name, _ in latin_provider.calls]


class TestShadowSampling:
    """A shadow page runs Tier 2 for the diff and is never an escalation."""

    def test_a_shadow_page_keeps_its_tier_one_lines(
        self, make_settings: Callable[..., Settings], latin_provider: MockProvider
    ) -> None:
        settings = make_settings(ESCALATION_SHADOW_RATE=1.0)
        latin_provider.heavy_lines = [
            line.model_copy(update={"text": "TIER2 REPLACEMENT"})
            for line in latin_provider.lines
        ]
        result = run_document(
            CLEAN_DOC, provider=latin_provider, settings=settings
        )
        page = result.final.pages[0]
        assert page.shadow_sampled is True
        assert page.escalated is False
        assert page.ocr_tier.value != "tier2"

    def test_shadow_sampling_still_calls_the_heavy_model(
        self, make_settings: Callable[..., Settings], latin_provider: MockProvider
    ) -> None:
        settings = make_settings(ESCALATION_SHADOW_RATE=1.0)
        run_document(CLEAN_DOC, provider=latin_provider, settings=settings)
        assert "ocr_page_heavy" in [name for name, _ in latin_provider.calls]


# --------------------------------------------------------------------------- #
# Residency
# --------------------------------------------------------------------------- #


class TestResidency:
    """Invariant 6, on both source forms."""

    def test_a_path_without_a_sidecar_is_refused_under_the_laptop_profile(
        self, provider: MockProvider, pipeline_settings: Settings, tmp_path: Path
    ) -> None:
        stray = tmp_path / "real_scan.jpg"
        stray.write_bytes(CLEAN_DOC.read_bytes())
        with pytest.raises(ResidencyViolation):
            run(stray, provider=provider, settings=pipeline_settings)

    def test_undeclared_bytes_are_refused_under_the_laptop_profile(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        with pytest.raises(ResidencyViolation) as raised:
            run(
                CLEAN_DOC.read_bytes(),
                provider=provider,
                settings=pipeline_settings,
            )
        assert "no bypass flag" in str(raised.value).lower()

    def test_declared_bytes_are_accepted(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        envelope = run(
            CLEAN_DOC.read_bytes(),
            provider=provider,
            settings=pipeline_settings,
            synthetic_declared=True,
        )
        assert envelope.document is not None

    def test_an_air_gapped_node_needs_no_declaration(
        self, provider: MockProvider, make_settings: Callable[..., Settings]
    ) -> None:
        """cpu_node processes real complaints; that is its whole purpose."""
        settings = make_settings(
            PROVIDER_PROFILE=ProviderProfile.CPU_NODE,
            SYNTHETIC_ONLY=False,
            ESCALATION_SHADOW_RATE=0.0,
        )
        envelope = run(
            CLEAN_DOC.read_bytes(), provider=provider, settings=settings
        )
        assert envelope.provider_profile == "cpu_node"

    def test_a_non_bytes_non_path_source_is_a_type_error(
        self, provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        with pytest.raises(TypeError):
            run(12345, provider=provider, settings=pipeline_settings)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# OPEN-QUESTIONS Q8 — the IFSC master is a setting
# --------------------------------------------------------------------------- #


class TestIFSCMasterSetting:
    """Q8: the real RBI list is a path change, and it is wired to the pipeline."""

    def test_the_default_is_the_committed_fixture(
        self, pipeline_settings: Settings
    ) -> None:
        from app.resources import ifsc_master_sample_path

        assert pipeline_settings.ifsc_master_path == ifsc_master_sample_path()

    def test_a_csv_master_is_loaded_and_used(
        self,
        provider: MockProvider,
        make_settings: Callable[..., Settings],
        tmp_path: Path,
    ) -> None:
        """A ``.csv`` is authoritative, so an unknown branch now *fails*."""
        master = tmp_path / "RBI_IFSC.csv"
        master.write_text(
            "IFSC,BANK,BRANCH,CENTRE,STATE\n"
            "SBIN0001234,State Bank of India,Pune Camp,Pune,Maharashtra\n",
            encoding="utf-8",
        )
        settings = make_settings(
            IFSC_MASTER_PATH=master, ESCALATION_SHADOW_RATE=0.0
        )
        ifsc_provider = MockProvider(
            lines=[
                OCRLine(
                    text="IFSC BARB0417293",
                    bbox=bbox_from_xyxy(10.0, 10.0, 300.0, 50.0),
                    confidence=0.9,
                    script="Latn",
                    language=None,
                    is_handwritten=False,
                    line_index=0,
                )
            ]
        )
        envelope = run(CLEAN_DOC, provider=ifsc_provider, settings=settings)
        assert envelope.document is not None
        ifsc = envelope.document.fields.get("ifsc_code")
        assert ifsc is not None, "the validator did not find the IFSC at all"
        assert ifsc.validator_passed is False
        assert "IFSC_NOT_IN_MASTER" in [flag.value for flag in ifsc.flags]

    def test_a_missing_master_raises_rather_than_falling_back(
        self,
        provider: MockProvider,
        make_settings: Callable[..., Settings],
        tmp_path: Path,
    ) -> None:
        """Silently reverting to the synthetic fixture would report unverified
        IFSCs as structurally confirmed across the whole country."""
        settings = make_settings(IFSC_MASTER_PATH=tmp_path / "absent.csv")
        with pytest.raises(ConfigurationError):
            run(CLEAN_DOC, provider=provider, settings=settings)

    def test_the_configured_bank_table_is_the_one_that_runs(
        self,
        make_settings: Callable[..., Settings],
        tmp_path: Path,
    ) -> None:
        """``BANK_ALIASES_PATH`` must reach the validator.

        ``run_validators`` always accepted a bank table and the orchestrator
        never passed one, so every deployment ran the committed fixture however
        it was configured — the same omission Q8 had in the IFSC table, and a
        worse one, because a name absent from a narrow table can still
        fuzzy-match a name inside it and be published as that bank.
        """
        table = tmp_path / "banks.json"
        table.write_text(
            json.dumps(
                {
                    "schema": "bank-aliases/1",
                    "banks": [
                        {
                            "code": "ZZZZ",
                            "name": "Bank of Elsewhere",
                            "aliases": ["Bank of Elsewhere"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        settings = make_settings(
            BANK_ALIASES_PATH=table, ESCALATION_SHADOW_RATE=0.0
        )
        bank_provider = MockProvider(
            lines=[
                OCRLine(
                    text="Bank of Elsewhere",
                    bbox=bbox_from_xyxy(10.0, 10.0, 300.0, 50.0),
                    confidence=0.9,
                    script="Latn",
                    language=None,
                    is_handwritten=False,
                    line_index=0,
                )
            ]
        )
        envelope = run(CLEAN_DOC, provider=bank_provider, settings=settings)
        assert envelope.document is not None
        bank = envelope.document.fields.get("bank_name")
        assert bank is not None, "the configured table was not consulted"
        assert bank.normalized == "Bank of Elsewhere"

    def test_a_missing_bank_table_raises_rather_than_falling_back(
        self,
        provider: MockProvider,
        make_settings: Callable[..., Settings],
        tmp_path: Path,
    ) -> None:
        """Falling back would run the fixture on a node that configured a list."""
        settings = make_settings(BANK_ALIASES_PATH=tmp_path / "absent.json")
        with pytest.raises(ConfigurationError):
            run(CLEAN_DOC, provider=provider, settings=settings)

    def test_an_empty_bank_table_raises(
        self,
        provider: MockProvider,
        make_settings: Callable[..., Settings],
        tmp_path: Path,
    ) -> None:
        """A table matching nothing disables bank_name for the whole deployment."""
        table = tmp_path / "empty.json"
        table.write_text(
            json.dumps({"schema": "bank-aliases/1", "banks": []}), encoding="utf-8"
        )
        settings = make_settings(BANK_ALIASES_PATH=table)
        with pytest.raises(ConfigurationError):
            run(CLEAN_DOC, provider=provider, settings=settings)


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


class TestDegradation:
    """A failing model must cost only what it owns."""

    def test_an_unavailable_slm_keeps_the_deterministic_fields(
        self, latin_provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        latin_provider.fail_with = {
            "extract_fields_batch": ProviderUnavailable("no SLM on this profile")
        }
        envelope = run(
            CLEAN_DOC, provider=latin_provider, settings=pipeline_settings
        )
        assert envelope.document is not None
        assert any("slm_unavailable" in w for w in envelope.warnings)
        assert "ifsc_code" in envelope.document.fields

    def test_quality_defect_flags_are_a_documented_closed_set(self) -> None:
        """The WS-F answer to E2-U6 is a named constant, not a magic condition."""
        assert QUALITY_DEFECT_FLAGS
        assert all(
            not flag.value.endswith("_REPAIRED") for flag in QUALITY_DEFECT_FLAGS
        )


class TestPublishedCoordinateFrame:
    """Boxes are published in the frame of the image that was submitted (Q11).

    The pipeline reads a *straightened* page, so detection produces boxes in the
    straightened frame; ``to_assembly`` is the boundary that maps them back.
    """

    @staticmethod
    def _outcome(rotation_applied_deg: float) -> PageOutcome:
        line = OCRLine(
            text="Reserve Bank of India",
            bbox=bbox_from_xyxy(300.0, 400.0, 900.0, 460.0),
            confidence=0.9,
            script="Latn",
            is_handwritten=False,
            line_index=0,
        )
        return PageOutcome(
            page_no=1,
            page=OCRPage(
                page_no=1,
                width_px=1654,
                height_px=2339,
                lines=[line],
                tier=OCRTier.TIER1,
                rotation_applied_deg=rotation_applied_deg,
            ),
            quality_score=1.0,
            quality_flags=(),
            quality_verdict=QualityVerdict.REPAIR,
            repairs_applied=("deskew",),
        )

    def test_an_unrepaired_page_is_untouched(self) -> None:
        outcome = self._outcome(0.0)
        assert outcome.to_assembly().page is outcome.page

    def test_a_repaired_page_maps_boxes_back_to_the_submitted_image(self) -> None:
        outcome = self._outcome(3.0)
        published = outcome.to_assembly().page.lines[0].bbox
        assert published != outcome.page.lines[0].bbox

        # The transform to invert is the one deskew applied, and the gate calls
        # deskew(working, skew_deg) while the envelope reports the negation.
        # Using the reported sign rotates by twice the skew instead of zero,
        # so pin the expected geometry rather than merely "something moved".
        expected = map_bbox_to_original(
            outcome.page.lines[0].bbox, height=2339, width=1654, degrees=-3.0
        )
        assert published == pytest.approx(expected)

    def test_the_sign_is_not_the_reported_rotation(self) -> None:
        """The bug this pins: mapping with +rotation_applied_deg."""
        outcome = self._outcome(3.0)
        published = outcome.to_assembly().page.lines[0].bbox
        wrong = map_bbox_to_original(
            outcome.page.lines[0].bbox, height=2339, width=1654, degrees=3.0
        )
        assert published != pytest.approx(wrong)


class TestPipelineResult:
    """The publication record itself."""

    def test_refuses_to_exist_with_no_envelope(self) -> None:
        with pytest.raises(ValueError, match="at least one envelope"):
            PipelineResult(envelopes=())

    def test_fast_path_and_final_are_the_same_object_without_escalation(
        self, latin_provider: MockProvider, pipeline_settings: Settings
    ) -> None:
        result = run_document(
            CLEAN_DOC, provider=latin_provider, settings=pipeline_settings
        )
        assert result.fast_path is result.final


# --------------------------------------------------------------------------- #
# The real provider (weights required)
# --------------------------------------------------------------------------- #


def _weights_present(settings: Settings) -> tuple[bool, str]:
    """Whether every artifact the Tier-1 path needs is on this machine."""
    required = (
        "ocr/det.onnx",
        "ocr/rec_latn.onnx",
        "ocr/rec_latn_dict.txt",
        "lid/lid.176.ftz",
    )
    missing = [
        name for name in required if not (settings.models_dir / name).exists()
    ]
    return (not missing), ", ".join(missing)


@pytest.fixture
def local_settings(make_settings: Callable[..., Settings]) -> Iterator[Settings]:
    """Settings for a real-weights run: no shadow sampling, no remote calls."""
    yield make_settings(ESCALATION_SHADOW_RATE=0.0)


@pytest.mark.local_models
def test_full_pipeline_over_a_clean_page_with_the_real_local_provider(
    local_settings: Settings, committed_schema: dict[str, object]
) -> None:
    """The one test that proves the composition works on real weights.

    Uses :class:`~app.providers.local.LocalProvider` directly rather than the
    laptop composite: the composite routes ``extract_fields`` and
    ``ocr_page_heavy`` to the HF router, and this test must not open a socket.
    The local provider raises ``ProviderMethodNotSupported`` for both, which the
    pipeline absorbs into an ``slm_unavailable`` warning — so what is asserted
    here is exactly the Tier-1 path: ingest, quality, batched recognition, LID,
    classification, validators and assembly, on real models.

    Skips with a reason when ``models/`` is empty, so a fresh clone stays green.
    """
    present, missing = _weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    from app.providers.local import LocalProvider

    provider = LocalProvider(local_settings)
    try:
        result = run_document(
            CLEAN_DOC, provider=provider, settings=local_settings
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    envelope = result.final
    assert_schema_valid(envelope, committed_schema)
    assert envelope.is_final is True
    assert envelope.bounce_decision.decision is not QualityVerdict.BOUNCE
    assert envelope.document is not None

    page = envelope.pages[0]
    assert page.line_count > 0, "the real recogniser read nothing from a clean page"
    assert page.ocr_tier.value in {"tier1", "tier2"}
    assert page.primary_script is not None


# --------------------------------------------------------------------------- #
# OPEN-QUESTIONS Q12 — the Devanagari head, proved on real weights
#
# The M7 gate found that no Devanagari page ever reached `rec_deva.onnx`:
# `app/pipeline.py` passed no script hint, Stage 4 bootstrapped on the Latin
# head, and the A-6 vote confirmed that head from its own Latin noise
# (docs/eval/REPORT-2026-08-08-slmlive.md §2).  Devanagari CER was 0.904, line LID was exactly
# 0.000 and Stage 6 never fired.  These tests are the end-to-end proof that the
# dual-head probe closed it, and the guard that the Latin path did not move.
# --------------------------------------------------------------------------- #

DEVA_DOC = CORPUS / "complaint_form_deva" / "complaint_form_deva_v0_clean.jpg"
MARATHI_DOC = CORPUS / "complaint_letter_mar" / "complaint_letter_mar_v0_clean.jpg"

#: Measured 2026-08-08 over the 147-document dev split, both arms through this
#: pipeline (``docs/decisions-q12.md`` §4).  Devanagari page CER went 0.7382 ->
#: 0.1648 overall and 0.7311 -> 0.0533 on the clean profile; this ceiling is the
#: task's 0.15 gate, comfortably above the 0.036 the clean page actually scores.
DEVA_CER_CEILING = 0.15

#: The probe's own share of the Stage-4 wall clock it runs inside.  Originally
#: measured p50 12%, p90 21%, p95 22% against a 0.25 budget.
#:
#: Raised to 0.30 because the *denominator* shrank, not because the probe got
#: more expensive: coverage-based head selection removed the second full
#: recognition pass those measurements included, so the same ~400 ms of probe
#: is now a larger share of a much shorter stage.  A page that gets faster
#: overall must not read as a latency regression.  The assertion is the budget,
#: not the measurement.
PROBE_LATENCY_BUDGET = 0.30


@contextmanager
def _disjoint_heads() -> Iterator[None]:
    """Pretend the installed heads do not cover one another.

    The shipped Devanagari head covers Latin, so head selection settles on it
    for everything and the probe never has to arbitrate.  A node carrying a
    Tamil or Bengali head has genuinely disjoint charsets, and that is the
    configuration the probe and the second pass exist for — so the tests that
    are *about* those mechanisms run inside this.
    """
    from app.stages import s2_script
    from app.stages import s4_ocr as s4_ocr_module

    disjoint = {"Latn": frozenset({"Latn"}), "Deva": frozenset({"Deva"})}
    saved_coverage = s2_script.REC_HEAD_COVERAGE
    saved_default = s2_script.DEFAULT_REC_HEAD_SCRIPT
    saved_stage_default = s4_ocr_module.DEFAULT_REC_HEAD_SCRIPT
    s2_script.REC_HEAD_COVERAGE = disjoint  # type: ignore[assignment]
    s2_script.DEFAULT_REC_HEAD_SCRIPT = "Latn"  # type: ignore[assignment]
    s4_ocr_module.DEFAULT_REC_HEAD_SCRIPT = "Latn"  # type: ignore[assignment]
    try:
        yield
    finally:
        s2_script.REC_HEAD_COVERAGE = saved_coverage  # type: ignore[assignment]
        s2_script.DEFAULT_REC_HEAD_SCRIPT = saved_default  # type: ignore[assignment]
        s4_ocr_module.DEFAULT_REC_HEAD_SCRIPT = saved_stage_default  # type: ignore[assignment]


def _deva_weights_present(settings: Settings) -> tuple[bool, str]:
    """Whether both recognition heads and the MT engine are on this machine."""
    required = (
        "ocr/det.onnx",
        "ocr/rec_latn.onnx",
        "ocr/rec_latn_dict.txt",
        "ocr/rec_deva.onnx",
        "ocr/rec_deva_dict.txt",
        "lid/lid.176.ftz",
        "mt/indic-en/model.bin",
    )
    missing = [
        name for name in required if not (settings.models_dir / name).exists()
    ]
    return (not missing), ", ".join(missing)


def _character_error_rate(reference: str, hypothesis: str) -> float:
    """Page-level CER using the eval harness's own scorer.

    Imported rather than reimplemented: the harness owns the normalisation
    (WS-D adjudication A-2) and a second copy here could drift from the number
    the M7 report quotes.  This is the *page-concatenated* CER, not the harness's
    per-matched-line CER, so it is not directly comparable to the 0.904 in the
    report — the same pre-fix page measures 0.657 this way.  Both arms of the
    comparison use this function, which is what makes the delta meaningful.
    """
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "eval_harness.py"
    spec = importlib.util.spec_from_file_location("eval_harness_for_cer", path)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    left = harness.normalize_for_cer(reference)
    right = harness.normalize_for_cer(hypothesis)
    return harness.levenshtein(left, right) / max(len(left), 1)


def _ground_truth_text(document: Path) -> str:
    payload = json.loads(
        document.with_suffix(".gt.json").read_text(encoding="utf-8")
    )
    return "\n".join(line["text"] for line in payload["lines"])


def _run_locally(document: Path, settings: Settings, **kwargs: object):
    """One document through the real local provider, closed afterwards."""
    from app.providers.local import LocalProvider

    provider = LocalProvider(settings)
    try:
        return run_document(
            document, provider=provider, settings=settings, **kwargs  # type: ignore[arg-type]
        )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@pytest.mark.local_models
def test_a_devanagari_page_reaches_the_devanagari_head_and_is_read(
    local_settings: Settings, committed_schema: dict[str, object]
) -> None:
    """The Q12 fix, end to end, on the page the M7 report used as its example.

    Before: the page was recognised only by the Latin head, ``बैंकिंग लोकपाल को
    शिकायत`` came back as ``h cicc  chr``, every line was labelled ``eng`` and
    the page's script was ``Latn``.  Each of those four facts is asserted below
    in its fixed form.

    The *mechanism* changed after this test was written.  The probe used to be
    what rescued the page: it bootstrapped on the Latin head, scored both heads
    over shared crops and re-read with the Devanagari one, at the cost of a
    second pass.  Head selection is now by charset **coverage**, so the page
    starts on the Devanagari head and is right the first time — one pass, no
    probe needed.  What is asserted here is therefore the outcome, not the
    route; the probe keeps its own tests, and still earns its place on a node
    whose heads do not cover each other.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    result = _run_locally(DEVA_DOC, local_settings)
    envelope = result.final
    assert_schema_valid(envelope, committed_schema)

    outcome = result.pages[0]
    tier1 = outcome.tier1
    assert tier1 is not None

    # 1. The Devanagari head read the page, and it took a single pass to do it.
    assert tier1.rec_head == "Deva"
    assert tier1.rec_passes == ("Deva",)

    # 2. The page's majority script is Devanagari (derived, invariant 1).
    page = envelope.pages[0]
    assert page.primary_script == "Deva"

    # 3. Line-level LID answers an Indic language instead of `eng`.
    assert page.primary_language in {"hin", "mar"}
    labelled = [line.language for line in outcome.page.lines if line.language]
    assert labelled, "no line carries a language label"
    assert any(language in {"hin", "mar"} for language in labelled)

    # 4. The text is actually right.
    recognised = "\n".join(line.text for line in outcome.page.lines)
    cer = _character_error_rate(_ground_truth_text(DEVA_DOC), recognised)
    assert cer < DEVA_CER_CEILING, f"Devanagari CER {cer:.4f}"
    assert "बैंकिंग" in recognised


@pytest.mark.local_models
def test_the_head_choice_is_what_made_the_difference_on_that_page(
    local_settings: Settings,
) -> None:
    """The A/B, on one page, so the claim is not taken on trust.

    The "before" arm forces the Latin head, which is the pre-fix behaviour:
    that head's charset holds a single Devanagari codepoint, so it cannot
    represent this page whatever else is true. Running both arms over the same
    weights and the same pixels is the only honest way to attribute the CER
    change to head selection rather than to anything else that moved.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    import app.pipeline as pipeline_module
    from app.stages.s4_ocr import run_tier1 as real_run_tier1

    reference = _ground_truth_text(DEVA_DOC)

    def latin_head_only(*args: object, **kwargs: object):
        kwargs["probe_heads"] = False
        kwargs["script_hint"] = "Latn"
        return real_run_tier1(*args, **kwargs)  # type: ignore[arg-type]

    original = pipeline_module.run_tier1
    pipeline_module.run_tier1 = latin_head_only  # type: ignore[assignment]
    try:
        with _disjoint_heads():
            before = _run_locally(DEVA_DOC, local_settings)
    finally:
        pipeline_module.run_tier1 = original  # type: ignore[assignment]
    after = _run_locally(DEVA_DOC, local_settings)

    before_cer = _character_error_rate(
        reference, "\n".join(line.text for line in before.pages[0].page.lines)
    )
    after_cer = _character_error_rate(
        reference, "\n".join(line.text for line in after.pages[0].page.lines)
    )

    assert before.pages[0].tier1 is not None
    assert before.pages[0].tier1.rec_head == "Latn", (
        "the pre-fix path is supposed to strand this page on the Latin head; "
        "if it no longer does, this A/B is measuring nothing"
    )
    assert before_cer > 0.5
    assert after_cer < DEVA_CER_CEILING
    assert after_cer < before_cer / 2


@pytest.mark.local_models
def test_an_english_page_is_left_exactly_as_it_was(
    local_settings: Settings,
) -> None:
    """The no-regression guard for English, measured against the Latin head.

    English pages now go to the Devanagari head, because its charset covers
    Latin and it measures at least as well on it. "At least as well" is the
    claim that needs guarding, so this runs the same page through the Latin
    head too and compares both against ground truth.

    Not byte equality: the two transcriptions are close but not identical, and
    where they differ the covering head has been the better of the two (it
    reads the form label ``2. Address`` where the Latin head drops the stop).
    Asserting equality would fail on an improvement.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    import app.pipeline as pipeline_module
    from app.stages.s4_ocr import run_tier1 as real_run_tier1

    def latin_head_only(*args: object, **kwargs: object):
        kwargs["probe_heads"] = False
        kwargs["script_hint"] = "Latn"
        return real_run_tier1(*args, **kwargs)  # type: ignore[arg-type]

    original = pipeline_module.run_tier1
    pipeline_module.run_tier1 = latin_head_only  # type: ignore[assignment]
    try:
        with _disjoint_heads():
            before = _run_locally(CLEAN_DOC, local_settings)
    finally:
        pipeline_module.run_tier1 = original  # type: ignore[assignment]
    after = _run_locally(CLEAN_DOC, local_settings)

    tier1 = after.pages[0].tier1
    assert tier1 is not None
    assert tier1.rec_head == "Deva", "the covering head reads English too"
    assert tier1.rec_passes == ("Deva",), "an English page must still cost one pass"
    assert tier1.script_disagreement is False

    # Script and language are derived from the *characters*, so an English page
    # read by the Devanagari head is still Latin/eng (Q14).
    assert after.final.pages[0].primary_script == "Latn"
    assert after.final.pages[0].primary_language == "eng"

    reference = _ground_truth_text(CLEAN_DOC)
    latin_cer = _character_error_rate(
        reference, "\n".join(line.text for line in before.pages[0].page.lines)
    )
    covering_cer = _character_error_rate(
        reference, "\n".join(line.text for line in after.pages[0].page.lines)
    )
    assert covering_cer <= latin_cer + 1e-9, (
        f"the covering head must not read English worse than the Latin head: "
        f"{covering_cer:.4f} vs {latin_cer:.4f}"
    )
    assert covering_cer < 0.05


@pytest.mark.local_models
def test_the_head_probe_stays_inside_its_latency_budget(
    local_settings: Settings,
) -> None:
    """The probe is charged to Stage 4; measure it, do not assume it.

    Run under :func:`_disjoint_heads`, because that is the only configuration
    that still reaches the probe: with the shipped charsets the covering head
    is chosen up front and no probe runs at all, which is a saving rather than
    a thing to measure. The budget still has to hold on a node whose heads do
    not cover each other.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    with _disjoint_heads():
        for document in (DEVA_DOC, CLEAN_DOC, MARATHI_DOC):
            result = _run_locally(document, local_settings)
            tier1 = result.pages[0].tier1
            assert tier1 is not None and tier1.script_probe is not None
            stage_ms = result.final.processing.stage_timings_ms["s4_ocr"]
            probe_ms = tier1.script_probe.elapsed_ms
            assert probe_ms > 0.0
            assert probe_ms / stage_ms < PROBE_LATENCY_BUDGET, (
                f"{document.name}: probe {probe_ms:.0f} ms is "
                f"{100 * probe_ms / stage_ms:.0f}% of the {stage_ms:.0f} ms stage"
            )


@pytest.mark.local_models
def test_no_probe_runs_when_one_head_already_covers_the_page(
    local_settings: Settings,
) -> None:
    """The saving the coverage rule buys: no probe, no second pass.

    The probe exists to arbitrate between heads that cannot read each other's
    script. When one installed head covers everything, there is nothing to
    arbitrate and asking would be paying for an answer already known.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    for document in (DEVA_DOC, CLEAN_DOC, MARATHI_DOC):
        result = _run_locally(document, local_settings)
        tier1 = result.pages[0].tier1
        assert tier1 is not None
        assert tier1.rec_passes == ("Deva",), document.name
        assert tier1.script_probe is None, document.name


@pytest.mark.local_models
def test_the_devanagari_page_now_reaches_the_translation_stage(
    local_settings: Settings,
) -> None:
    """Q7's blocker was Q12: Stage 6 had nothing in an Indic script to translate.

    The M7 run produced ``value_en`` for **0 of 503** fields, not because MT
    failed but because every Devanagari page's field values were Latin noise
    (OPEN-QUESTIONS Q7, WS-H2 update).  The free-text leg is stood in for here —
    the router SLM is account-level unavailable and that is not this lane's to
    fix — but everything after it is real: a verbatim Devanagari value out of the
    Devanagari head, through the grounding gate, the sentinel masking and
    IndicTrans2's own weights.
    """
    present, missing = _deva_weights_present(local_settings)
    if not present:
        pytest.skip(f"local weights absent from models/: {missing}")

    from app.contracts.ocr import ExtractionContext, RawField
    from app.providers.local import LocalProvider, script_of_char

    class _VerbatimFreeTextProvider:
        """The local provider with a deterministic stand-in for the SLM.

        It copies the page's longest Indic line verbatim, so the grounding gate
        (invariant 8) passes by construction and the only thing under test is
        what Q12 unblocked: an Indic field value existing at all.
        """

        name = "verbatim-over-local"

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, item: str) -> object:
            return getattr(self._inner, item)

        def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]:
            best: tuple[int, OCRLine] | None = None
            for line in ctx.lines:
                indic = sum(
                    1
                    for character in line.text
                    if script_of_char(character) not in (None, "Latn")
                )
                if indic >= 12 and (best is None or indic > best[0]):
                    best = (indic, line)
            if best is None:
                return {}
            return {
                "relief_sought": RawField(
                    name="relief_sought",
                    value=best[1].text,
                    confidence=0.8,
                    line_indices=[best[1].line_index],
                )
            }

        def extract_fields_batch(
            self, ctxs: Sequence[ExtractionContext]
        ) -> list[dict[str, RawField]]:
            return [self.extract_fields(ctx) for ctx in ctxs]

    inner = LocalProvider(local_settings)
    provider = _VerbatimFreeTextProvider(inner)
    try:
        envelope = run_document(
            DEVA_DOC, provider=provider, settings=local_settings
        ).final
    finally:
        close = getattr(inner, "close", None)
        if callable(close):
            close()

    assert envelope.document is not None
    translated = [
        (name, field)
        for name, field in envelope.document.fields.items()
        if field.value_en
    ]
    assert translated, "no field carries an advisory English rendering"
    _name, field = translated[0]
    assert any(
        script_of_char(character) == "Deva" for character in str(field.value)
    ), "the translated field's native value is not Devanagari"
    # Invariant 10: `value` is immutable and `value_en` names its engine.
    assert field.mt_engine
    assert field.value != field.value_en


def test_requested_names_reach_model_context_and_persist_in_grouped_result(
    pipeline_settings: Settings, provider: MockProvider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = []
    original = provider.extract_fields_batch
    provider.field_values['custom_reference'] = '4. Name of bank'

    def capture(items):
        contexts.extend(items)
        return original(items)

    monkeypatch.setattr(provider, 'extract_fields_batch', capture)
    settings = pipeline_settings.model_copy(update={'EXTRACT_ALL_FIELDS': False})
    result = run_document(CLEAN_DOC, provider=provider, settings=settings,
                          requested_fields=['custom reference'], starting_result_version=5)
    assert contexts
    assert all(ctx.discover_fields and ctx.target_fields == ['custom_reference'] for ctx in contexts)
    assert result.envelopes[0].result_version == 5
    final = result.envelopes[-1]
    assert final.requested_fields == ('custom_reference',)
    assert final.document.extracted_fields['custom_reference'][0].value == '4. Name of bank'
    assert final.document.other_fields
