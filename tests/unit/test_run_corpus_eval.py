"""Unit tests for ``tools/run_corpus_eval.py`` — the M7 exit-gate driver.

Nothing here loads a weight, opens a socket or reads a corpus JPEG.  What is
tested is the part of a sweep that can be silently wrong and would then produce
a report nobody can challenge:

* **Selection** — the locked split is the harness's rule, ``--limit`` truncates
  after filtering, and ``--split all`` without ``--final`` refuses.
* **Schema conformance** — a prediction built from a real
  :class:`~app.contracts.envelope.ResultEnvelope` passes the harness's own
  ``validate_prediction`` with zero violations.  That assertion is the contract
  between this tool and the scorer; if it fails, every metric downstream is
  measuring the wrong thing.
* **Failure containment** — one document that raises, and one that outlives the
  wall-clock budget, must each become a prediction file and must not stop the
  sweep.
* **The two things a sweep gets wrong at scale** — not resetting the per-job
  token budget (the run dies after ~3 documents), and swapping the SLM on a
  schema failure instead of an outage (the run silently changes what it
  measures).

The corpus and pipeline are both fakes, which is what lets this file run in the
same second as the rest of the suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Module under test + the harness it borrows from
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Load the driver as a module (``tools/`` is not a package)."""
    path = PROJECT_ROOT / "tools" / "run_corpus_eval.py"
    spec = importlib.util.spec_from_file_location("run_corpus_eval_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness(tool: ModuleType) -> ModuleType:
    return tool.load_harness()


# --------------------------------------------------------------------------- #
# A miniature corpus on disk
# --------------------------------------------------------------------------- #


def _write_corpus(root: Path, *, variants: int = 5) -> Path:
    """3 types x ``variants`` x 2 profiles, with usable ground truth."""
    documents = []
    for doc_type in ("complaint_form_en", "complaint_letter_mar", "receipt"):
        for variant in range(variants):
            for profile, expectation in (("clean", "clean"), ("heavy", "degraded")):
                doc_id = f"{doc_type}_v{variant}_{profile}"
                gt_rel = f"{doc_type}/{doc_id}.gt.json"
                img_rel = f"{doc_type}/{doc_id}.jpg"
                documents.append(
                    {
                        "doc_id": doc_id,
                        "type": doc_type,
                        "variant": variant,
                        "profile": profile,
                        "image": img_rel,
                        "gt": gt_rel,
                        "lines": 2,
                        "quality_expectation": expectation,
                        "is_code_mixed": False,
                        "primary_script": "Deva" if doc_type.endswith("_mar") else "Latn",
                    }
                )
                gt_path = root / gt_rel
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.write_text(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "quality_expectation": expectation,
                            "lines": [
                                {
                                    "text": "IFSC Code: BARB0417293",
                                    "bbox": [10, 10, 200, 10, 200, 40, 10, 40],
                                    "script": "Latn",
                                    "language": "eng",
                                    "is_handwritten": False,
                                    "field": "ifsc_code",
                                    "value": "BARB0417293",
                                },
                                {
                                    "text": "Amount: 45,000",
                                    "bbox": [10, 50, 200, 50, 200, 80, 10, 80],
                                    "script": "Latn",
                                    "language": "eng",
                                    "is_handwritten": False,
                                    "field": "amount",
                                    "value": "45,000",
                                },
                            ],
                            "fields": {"ifsc_code": "BARB0417293", "amount": "45,000"},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (root / img_rel).write_bytes(b"\xff\xd8\xff\xdb not-a-real-jpeg")

    (root / "index.json").write_text(
        json.dumps(
            {
                "seed": 1729,
                "synthetic": True,
                "generator_version": "1.1.0",
                "hand_shaping": "cluster",
                "count": len(documents),
                "caveat": "synthetic",
                "documents": documents,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path / "corpus")


# --------------------------------------------------------------------------- #
# A fake pipeline result
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_result(
    make_envelope: Any,
    make_page_result: Any,
    make_document_result: Any,
    make_field: Any,
    make_ocr_page: Any,
) -> Any:
    """A ``PipelineResult``-shaped object built from the suite's own fixtures."""

    def build(
        *,
        escalated: bool = False,
        verdict: str = "accept",
        fields: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
    ) -> Any:
        from app.contracts.envelope import ProcessingStats, QualityVerdict

        page = make_ocr_page(page_no=1)
        document = make_document_result(
            fields=fields
            if fields is not None
            else {
                "ifsc_code": make_field(value="BARB0417293", source="regex_validator"),
                "amount": make_field(value="45,000", source="regex_validator"),
            }
        )
        envelope = make_envelope(
            document=document,
            pages=(
                make_page_result(
                    page_no=1,
                    quality_verdict=QualityVerdict(verdict),
                    line_count=len(page.lines),
                    escalated=escalated,
                ),
            ),
            escalation_flag=escalated,
            warnings=warnings,
            processing=ProcessingStats(
                duration_ms=1234.5,
                pages_processed=1,
                stage_timings_ms={"s4_ocr": 812.0, "s4_ocr:p1": 812.0},
            ),
        )
        outcome = SimpleNamespace(
            page_no=1,
            page=page,
            shadow_sampled=False,
            escalated=escalated,
            escalation=SimpleNamespace(escalate=escalated, detail="two signals"),
        )
        return SimpleNamespace(
            envelopes=(envelope,), final=envelope, pages=(outcome,), ingest=None
        )

    return build


# --------------------------------------------------------------------------- #
# Selection and the locked split
# --------------------------------------------------------------------------- #


class TestSelection:
    def test_dev_split_excludes_the_locked_variants(
        self, tool: ModuleType, corpus: Path, tmp_path: Path
    ) -> None:
        config = tool.RunConfig(corpus_dir=corpus, out_dir=tmp_path / "out")
        selected = tool.select_corpus_documents(config)
        assert selected, "the dev split must not be empty"
        assert {d.variant for d in selected} == {0, 1, 2}
        assert len(selected) == 3 * 3 * 2  # types x dev variants x profiles

    def test_split_all_without_final_refuses(
        self, tool: ModuleType, harness: ModuleType, corpus: Path, tmp_path: Path
    ) -> None:
        config = tool.RunConfig(corpus_dir=corpus, out_dir=tmp_path / "o", split="all")
        with pytest.raises(harness.LockedSplitError):
            tool.select_corpus_documents(config)

    def test_final_unlocks_every_variant(
        self, tool: ModuleType, corpus: Path, tmp_path: Path
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "o", split="all", final=True
        )
        assert {d.variant for d in tool.select_corpus_documents(config)} == {0, 1, 2, 3, 4}

    def test_limit_truncates_after_filtering_so_a_smoke_run_is_a_prefix(
        self, tool: ModuleType, corpus: Path, tmp_path: Path
    ) -> None:
        base = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "o", types=("receipt",)
        )
        full = tool.select_corpus_documents(base)
        limited = tool.select_corpus_documents(
            tool.RunConfig(
                corpus_dir=corpus, out_dir=tmp_path / "o", types=("receipt",), limit=3
            )
        )
        assert [d.doc_id for d in limited] == [d.doc_id for d in full[:3]]
        assert all(d.doc_type == "receipt" for d in limited)

    def test_profiles_filter(self, tool: ModuleType, corpus: Path, tmp_path: Path) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "o", profiles=("heavy",)
        )
        assert {d.profile for d in tool.select_corpus_documents(config)} == {"heavy"}

    def test_dry_run_writes_nothing_and_exits_zero(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        code = tool.main(
            ["--corpus", str(corpus), "--out", str(out), "--dry-run", "--limit", "4"]
        )
        assert code == 0
        assert not out.exists(), "a dry run must not create the predictions directory"
        assert "4 documents would run" in capsys.readouterr().out

    def test_dry_run_honours_the_locked_split(
        self, tool: ModuleType, corpus: Path, tmp_path: Path
    ) -> None:
        code = tool.main(
            ["--corpus", str(corpus), "--out", str(tmp_path / "o"), "--dry-run", "--split", "test"]
        )
        assert code == 3


# --------------------------------------------------------------------------- #
# Prediction schema conformance — the contract with the scorer
# --------------------------------------------------------------------------- #


class TestPredictionSchema:
    def test_a_real_envelope_produces_a_schema_valid_prediction(
        self, tool: ModuleType, harness: ModuleType, fake_result: Any
    ) -> None:
        payload = tool.prediction_from_result("complaint_form_en_v0_clean", fake_result())
        violations = harness.validate_prediction(payload, "complaint_form_en_v0_clean")
        assert violations == [], violations

    def test_the_prediction_round_trips_through_json_unchanged(
        self, tool: ModuleType, harness: ModuleType, fake_result: Any
    ) -> None:
        payload = tool.prediction_from_result("receipt_v1_heavy", fake_result())
        reloaded = json.loads(json.dumps(payload, ensure_ascii=False))
        assert harness.validate_prediction(reloaded, "receipt_v1_heavy") == []

    def test_lines_carry_the_per_line_labels_the_harness_scores_LID_on(
        self, tool: ModuleType, fake_result: Any
    ) -> None:
        payload = tool.prediction_from_result("d", fake_result())
        line = payload["pages"][0]["lines"][0]
        assert set(line) == {
            "text",
            "bbox",
            "confidence",
            "script",
            "language",
            "is_handwritten",
            "line_index",
        }
        assert len(line["bbox"]) == 8

    def test_field_source_stays_inside_the_harness_vocabulary(
        self, tool: ModuleType, harness: ModuleType, fake_result: Any
    ) -> None:
        payload = tool.prediction_from_result("d", fake_result())
        for spec in payload["fields"].values():
            assert spec["source"] in harness.FIELD_SOURCES
            assert "page_no" in spec["evidence"] and "bbox" in spec["evidence"]

    def test_document_quality_is_the_worst_page_not_the_first(
        self, tool: ModuleType, make_envelope: Any, make_page_result: Any
    ) -> None:
        from app.contracts.envelope import QualityVerdict

        envelope = make_envelope(
            pages=(
                make_page_result(page_no=1, quality_verdict=QualityVerdict.ACCEPT),
                make_page_result(
                    page_no=2, quality_verdict=QualityVerdict.BOUNCE, line_count=0
                ),
                make_page_result(page_no=3, quality_verdict=QualityVerdict.REPAIR),
            )
        )
        assert tool.document_quality(envelope)["verdict"] == "bounce"

    def test_document_quality_falls_back_to_the_bounce_decision_with_no_pages(
        self, tool: ModuleType, make_envelope: Any
    ) -> None:
        from app.contracts.envelope import BounceDecision, BounceReason, QualityVerdict

        envelope = make_envelope(
            pages=(),
            document=None,
            bounce_decision=BounceDecision(
                decision=QualityVerdict.BOUNCE, reason=BounceReason.BLUR_ABOVE_CEILING
            ),
        )
        assert tool.document_quality(envelope)["verdict"] == "bounce"

    def test_timings_carry_an_end_to_end_total_for_the_latency_target(
        self, tool: ModuleType, fake_result: Any
    ) -> None:
        payload = tool.prediction_from_result("d", fake_result())
        assert payload["timings_ms"]["e2e_total"] == pytest.approx(1234.5)

    def test_escalation_reason_is_only_set_when_a_page_actually_escalated(
        self, tool: ModuleType, fake_result: Any
    ) -> None:
        assert tool.prediction_from_result("d", fake_result())["escalation_reason"] is None
        escalated = tool.prediction_from_result("d", fake_result(escalated=True))
        assert escalated["escalated"] is True
        assert escalated["escalation_reason"] == "two signals"

    def test_escalated_is_the_router_decision_not_the_needs_review_flag(
        self, tool: ModuleType, fake_result: Any
    ) -> None:
        """``assemble`` ORs "a field needs review" into ``escalation_flag``.

        Publishing that as ``escalated`` scores every failed-checksum document
        as a Tier-2 escalation.  On the M7 dev split it is the difference
        between 39 documents (22%, missing the PRD <=15% target) and 22 (12%,
        meeting it) — so the two have to stay separate keys.
        """
        result = fake_result(escalated=False)
        object.__setattr__(result.final, "escalation_flag", True)
        payload = tool.prediction_from_result("d", result)
        assert payload["escalated"] is False, "the router did not escalate this page"
        assert payload["needs_review"] is True
        assert payload["tier2_applied"] is False


# --------------------------------------------------------------------------- #
# Failure containment
# --------------------------------------------------------------------------- #


class TestFailureContainment:
    def test_a_raising_document_is_recorded_and_the_sweep_continues(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        from app.contracts.errors import ProviderError

        out = tmp_path / "out"
        config = tool.RunConfig(corpus_dir=corpus, out_dir=out, limit=4, doc_timeout_s=0)
        documents = tool.select_corpus_documents(config)
        calls: list[str] = []

        def runner(path: Path, **kwargs: Any) -> Any:
            calls.append(kwargs["job_id"])
            if kwargs["job_id"].endswith("_v1_clean"):
                raise ProviderError("the endpoint fell over")
            return fake_result()

        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=runner,
        )

        assert len(calls) == 4, "every document must be attempted"
        assert outcome.ok == 3 and outcome.failed == 1
        failed = [r for r in outcome.records if r.status == "failed"]
        assert failed and failed[0].error_type == "ProviderError"
        assert len(list(out.glob("*.json"))) == 4, "a failure still gets a prediction file"

    def test_a_failed_prediction_costs_recall_rather_than_disappearing(
        self, tool: ModuleType, harness: ModuleType
    ) -> None:
        payload = tool.error_prediction("d", RuntimeError("boom"), wall_ms=12.0)
        # pages: [] makes the harness score every GT line as a deletion, and
        # fields: {} makes every GT field a miss. Omitting them would quietly
        # exclude the document from both denominators.
        assert payload["pages"] == [] and payload["fields"] == {}
        parsed = harness.parse_prediction(payload, "d")
        assert parsed.has_lines_block is True
        assert parsed.has_fields_block is True

    def test_a_failed_prediction_reports_no_stage_timings(self, tool: ModuleType) -> None:
        payload = tool.error_prediction("d", RuntimeError("boom"), wall_ms=180_000.0)
        assert "timings_ms" not in payload
        assert payload["error"]["wall_ms"] == pytest.approx(180_000.0)

    def test_the_wall_clock_budget_abandons_a_hung_document(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus,
            out_dir=tmp_path / "out",
            limit=3,
            doc_timeout_s=0.3,
            doc_grace_s=2.0,
        )
        documents = tool.select_corpus_documents(config)

        def runner(path: Path, **kwargs: Any) -> Any:
            if kwargs["job_id"].endswith("_v0_heavy"):
                time.sleep(0.8)
            return fake_result()

        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=runner,
        )
        statuses = {r.doc_id: r.status for r in outcome.records}
        assert len(statuses) == 3
        timed_out = [d for d, s in statuses.items() if s == "timeout"]
        assert timed_out and timed_out[0].endswith("_v0_heavy")
        payload = json.loads((config.out_dir / timed_out[0]).with_suffix(".json").read_text("utf-8"))
        assert payload["error"]["type"] == "DocumentTimeout"

    def test_document_runner_without_a_budget_calls_inline(self, tool: ModuleType) -> None:
        runner = tool.DocumentRunner(0.0, 0.0)
        try:
            assert runner.call(lambda: "value") == "value"
            with pytest.raises(ZeroDivisionError):
                runner.call(lambda: 1 / 0)
        finally:
            runner.close()

    def test_every_written_prediction_is_schema_checked(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=2, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
        )
        assert outcome.schema_violations == {}, outcome.schema_violations

    def test_the_prediction_file_is_named_for_the_corpus_doc_id(
        self, tool: ModuleType, harness: ModuleType, corpus: Path, tmp_path: Path,
        settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=2, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
        )
        for doc in documents:
            assert harness.find_prediction_file(config.out_dir, doc.doc_id) is not None
            payload = json.loads(
                (config.out_dir / f"{doc.doc_id}.json").read_text(encoding="utf-8")
            )
            assert payload["doc_id"] == doc.doc_id


# --------------------------------------------------------------------------- #
# Resume, budget and the SLM fallback
# --------------------------------------------------------------------------- #


class TestSweepMechanics:
    def test_resume_skips_documents_that_already_have_a_prediction(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=out, limit=3, doc_timeout_s=0, resume=True
        )
        documents = tool.select_corpus_documents(config)
        (out / f"{documents[0].doc_id}.json").write_text("{}", encoding="utf-8")
        seen: list[str] = []
        tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: (seen.append(k["job_id"]), fake_result())[1],
        )
        assert documents[0].doc_id not in seen
        assert len(seen) == 2

    def test_the_token_budget_is_reset_per_document(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        """JOB_TOKEN_BUDGET is a *per-job* ceiling on a per-instance budget.

        A sweep that never resets it exhausts the budget after a handful of
        documents and every later extraction dies as ``TokenBudgetExceeded`` —
        silently, because the orchestrator absorbs SLM failures.
        """
        resets: list[str | None] = []
        remote = SimpleNamespace(
            slm_model="google/gemma-3-4b-it",
            budget=SimpleNamespace(used=0),
            reset_budget=lambda job_id=None: resets.append(job_id),
        )
        provider = SimpleNamespace(remote=remote)
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=4, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        tool.run_sweep(
            config,
            provider=provider,
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
        )
        assert resets == [d.doc_id for d in documents]

    def test_a_schema_failure_does_not_swap_the_model(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        """0.56 schema validity is the *measured* behaviour of the default SLM.

        Treating a run of invalid answers as an outage would swap the model
        mid-sweep and quietly change what the report is about.
        """
        remote = SimpleNamespace(
            slm_model="google/gemma-3-4b-it",
            budget=SimpleNamespace(used=0),
            reset_budget=lambda job_id=None: None,
        )
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=6, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(remote=remote),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(
                warnings=("slm_unavailable:PROVIDER_RESPONSE_INVALID",)
            ),
        )
        assert outcome.slm_model_switched_to is None
        assert remote.slm_model == "google/gemma-3-4b-it"

    def test_a_sustained_outage_swaps_to_the_configured_fallback(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        remote = SimpleNamespace(
            slm_model=settings.SLM_MODEL,
            budget=SimpleNamespace(used=0),
            reset_budget=lambda job_id=None: None,
        )
        config = tool.RunConfig(
            corpus_dir=corpus,
            out_dir=tmp_path / "out",
            limit=6,
            doc_timeout_s=0,
            slm_failures_before_fallback=3,
        )
        documents = tool.select_corpus_documents(config)
        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(remote=remote),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(
                warnings=("slm_unavailable:PROVIDER_UNAVAILABLE",)
            ),
        )
        assert outcome.slm_model_switched_to == settings.SLM_MODEL_FALLBACK
        assert remote.slm_model == settings.SLM_MODEL_FALLBACK
        assert "outage" in (outcome.slm_switch_reason or "")

    @pytest.mark.parametrize(
        ("warning", "expected"),
        [
            ("slm_unavailable:PROVIDER_UNAVAILABLE", ["PROVIDER_UNAVAILABLE"]),
            ("slm_unavailable:PROVIDER_RATE_LIMITED", ["PROVIDER_RATE_LIMITED"]),
            ("slm_unavailable:PROVIDER_RESPONSE_INVALID", []),
            ("slm_unavailable:PROVIDER_ERROR", []),
            ("tier2_failed:page=1:reason=x", []),
        ],
    )
    def test_only_outage_codes_arm_the_fallback(
        self, tool: ModuleType, warning: str, expected: list[str]
    ) -> None:
        assert tool.slm_outage_codes([warning]) == expected

    def test_no_slm_refuses_every_remote_method_and_forwards_the_rest(
        self, tool: ModuleType
    ) -> None:
        from app.providers.base import ProviderMethodNotSupported

        inner = SimpleNamespace(
            name="composite",
            detect_script=lambda crops: "script",
            classify_page=lambda page, feats: "class",
            ocr_page=lambda page, script: ["line"],
            identify_language=lambda text, script: "lang",
            translate=lambda texts, src, tgt="eng": list(texts),
            close=lambda: None,
        )
        provider = tool.SLMDisabledProvider(inner)
        assert provider.ocr_page(None, "Latn") == ["line"]
        assert provider.identify_language("x", "Latn") == "lang"
        for call in (
            lambda: provider.extract_fields(None),
            lambda: provider.extract_fields_batch([]),
            lambda: provider.ocr_page_heavy(None),
        ):
            with pytest.raises(ProviderMethodNotSupported):
                call()

    def test_no_slm_hides_the_slm_model_from_the_manifest(self, tool: ModuleType) -> None:
        provider = tool.SLMDisabledProvider(SimpleNamespace(name="composite"))
        assert tool.remote_side(provider) is None

    def test_remote_side_finds_the_hf_provider_behind_the_composite(
        self, tool: ModuleType
    ) -> None:
        remote = SimpleNamespace(slm_model="m")
        assert tool.remote_side(SimpleNamespace(remote=remote)) is remote
        assert tool.remote_side(remote) is remote
        assert tool.remote_side(SimpleNamespace(local=1)) is None

    def test_progress_is_reported_for_the_first_last_and_every_nth_document(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=7, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        seen: list[int] = []
        tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
            on_progress=lambda position, total, record: seen.append(position),
        )
        assert seen == list(range(1, 8))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


class TestManifest:
    def test_the_manifest_never_carries_the_token(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=2, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
        )
        manifest = tool.build_manifest(
            config,
            outcome,
            settings=settings,
            corpus_index={"seed": 1729},
            documents_selected=len(documents),
        )
        blob = json.dumps(manifest)
        assert "HF_TOKEN" not in blob
        assert "hf_token_present" in blob
        assert "**********" not in blob and "SecretStr" not in blob
        # SLM_MAX_TOKENS is a token *count*, not a credential; the credentials
        # are the two names below and neither may appear as a settings key.
        assert "HF_TOKEN" not in manifest["settings"]
        assert "WEBHOOK_HMAC_SECRET" not in manifest["settings"]

    def test_the_manifest_is_json_serialisable_and_counts_what_ran(
        self, tool: ModuleType, corpus: Path, tmp_path: Path, settings: Any, fake_result: Any
    ) -> None:
        config = tool.RunConfig(
            corpus_dir=corpus, out_dir=tmp_path / "out", limit=3, doc_timeout_s=0
        )
        documents = tool.select_corpus_documents(config)
        outcome = tool.run_sweep(
            config,
            provider=SimpleNamespace(),
            settings=settings,
            documents=documents,
            run_document=lambda p, **k: fake_result(),
        )
        manifest = json.loads(
            json.dumps(
                tool.build_manifest(
                    config,
                    outcome,
                    settings=settings,
                    corpus_index={"seed": 1729, "generator_version": "1.1.0"},
                    documents_selected=len(documents),
                ),
                ensure_ascii=False,
            )
        )
        assert manifest["run"]["documents_ok"] == 3
        assert manifest["run"]["documents_failed"] == 0
        assert manifest["run"]["quality_verdict_counts"] == {"accept": 3}
        assert manifest["corpus"]["seed"] == 1729
        assert manifest["config"]["split"] == "dev"
        # Assert the *contents* of the per-document rows, not just their count:
        # a `len(...) == 3` check passes just as happily on `[None, None, None]`,
        # which is exactly what a misplaced `return` in `DocRecord.as_json`
        # produced on the first real run.
        assert len(manifest["documents"]) == 3
        for row, doc in zip(manifest["documents"], documents, strict=True):
            assert row is not None
            assert row["doc_id"] == doc.doc_id
            assert row["status"] == "ok"
            assert row["profile"] == doc.profile
            assert row["model_path"] in ("live", "cached", "none")
            assert isinstance(row["wall_ms"], float)

    def test_model_path_separates_a_cache_hit_from_no_model_call_at_all(
        self, tool: ModuleType
    ) -> None:
        """A bounced page makes no model lookup; it is not a fast cache hit.

        Folding the two together publishes "the cache makes a document 10x
        faster" when the truth is "the quality gate refused the page".
        """
        row = tool.DocRecord(doc_id="d", doc_type="t", variant=0, profile="clean")
        assert row.model_path == "none"
        row.cache_hits = 1
        assert row.model_path == "cached"
        row.cache_misses = 1
        assert row.model_path == "live"

    def test_warning_counts_keep_the_slm_code_but_not_the_tier2_page_number(
        self, tool: ModuleType
    ) -> None:
        assert tool._warning_key("slm_unavailable:PROVIDER_RESPONSE_INVALID") == (
            "slm_unavailable:PROVIDER_RESPONSE_INVALID"
        )
        assert tool._warning_key("tier2_failed:page=3:reason=empty") == "tier2_failed"
        assert tool._warning_key("escalation_pending") == "escalation_pending"
