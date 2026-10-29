"""Unit tests for ``tools/slm_bakeoff.py`` — the M7 SLM bake-off.

Nothing here opens a socket or needs ``HF_TOKEN``: every test runs against the
stub provider the tool ships for exactly this purpose, or against fabricated
scorecards.  What is tested is the part of a bake-off that decides which model
gets shipped and would be believed without argument — the ground-truth loader,
the scoring, the ranking arithmetic, the Q5 verdict logic, the honesty of the
coverage accounting on a partial run, and the shape of ``results.json`` and
``REPORT.md``.

The residency check is tested too: the tool must refuse a document that carries
no ``synthetic: true`` declaration while the laptop profile is active, because
that check is the only thing between a real complaint and a foreign endpoint.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from app.contracts.errors import ResidencyViolation
from app.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bakeoff() -> ModuleType:
    """Load the tool as a module (``tools/`` is not a package)."""
    path = PROJECT_ROOT / "tools" / "slm_bakeoff.py"
    spec = importlib.util.spec_from_file_location("slm_bakeoff_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-document corpus with the tags and sidecars the loader expects."""
    root = tmp_path / "corpus"
    for doc_type, script, language, doc_class in (
        ("complaint_form_en", "Latn", "eng", "complaint_form"),
        ("complaint_letter_mar", "Deva", "mar", "complaint_letter"),
    ):
        folder = root / doc_type
        folder.mkdir(parents=True)
        doc_id = f"{doc_type}_v0_clean"
        payload = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "variant": 0,
            "synthetic": True,
            "document_class": doc_class,
            "primary_script": script,
            "primary_language": language,
            "is_code_mixed": False,
            "quality_expectation": "clean",
            "lines": [
                _line("Complaint to the Banking Ombudsman", script, language, None),
                _line("Asha Patil", script, language, "complainant_name"),
                _line("Flat 4B, Pune 411001", script, language, "complainant_address"),
                _line("Account 8234484955318053", script, language, "account_number"),
                _line("The bank debited twice", script, language, "complaint_summary"),
                _line("Ref CMS/2026/173628", script, language, "prior_complaint_ref"),
            ],
            "fields": {
                "complainant_name": "Asha Patil",
                "complainant_city": "Flat 4B, Pune 411001",
                "account_number": "8234484955318053",
                "prior_complaint_ref": "CMS/2026/173628",
                "mobile": "9995332248",
            },
        }
        (folder / f"{doc_id}.gt.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (folder / f"{doc_id}.jpg").write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")
    return root


def _line(
    text: str, script: str, language: str, tag: str | None
) -> dict[str, Any]:
    return {
        "text": text,
        "bbox": [10.0, 10.0, 400.0, 10.0, 400.0, 50.0, 10.0, 50.0],
        "script": script,
        "language": language,
        "is_handwritten": False,
        "field": tag,
    }


@pytest.fixture
def laptop_settings(make_settings: Any, corpus: Path) -> Settings:
    return make_settings(CORPUS_DIR=corpus.parent / "corpus")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parser_defaults(bakeoff: ModuleType) -> None:
    args = bakeoff.build_parser().parse_args([])
    assert args.model is None
    assert args.dry_run is False
    assert args.no_cache is False
    assert args.out_dir.name == "bakeoff"


def test_models_are_repeatable(bakeoff: ModuleType) -> None:
    args = bakeoff.build_parser().parse_args(["--model", "a/b", "--model", "c/d"])
    assert args.model == ["a/b", "c/d"]


def test_the_brief_requires_at_least_three_candidates(
    bakeoff: ModuleType, settings: Settings
) -> None:
    """BUILD-BRIEF M7: ">= 3 candidates via the HF API"."""
    assert len(bakeoff.default_candidates(settings)) >= 3


def test_configured_models_lead_the_candidate_list(
    bakeoff: ModuleType, settings: Settings
) -> None:
    candidates = bakeoff.default_candidates(settings)
    assert candidates[0] == settings.SLM_MODEL
    assert candidates[1] == settings.SLM_MODEL_FALLBACK
    assert len(candidates) == len(set(candidates)), "candidates must be unique"


def test_locked_test_variants_are_never_sampled(bakeoff: ModuleType) -> None:
    """Variants 3 and 4 are the locked test split (TEST-DATA-SPEC sec. 7)."""
    assert set(bakeoff.SAMPLE_VARIANTS).isdisjoint({3, 4})


# --------------------------------------------------------------------------- #
# Ground truth loading
# --------------------------------------------------------------------------- #


def test_load_gold_docs_reads_lines_and_tags(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    docs = bakeoff.load_gold_docs(
        corpus,
        settings=laptop_settings,
        types=("complaint_form_en",),
        variants=(0,),
    )
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "complaint_form_en_v0_clean"
    assert len(doc.page.lines) == 6
    assert [line.line_index for line in doc.page.lines] == list(range(6))
    assert doc.tagged_lines["complainant_name"] == (1,)
    assert doc.summary_lines == (4,)


def test_gt_tag_aliases_are_translated(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """The corpus tags ``prior_complaint_ref``; the contract says …_reference."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    assert doc.tagged_lines["prior_complaint_reference"] == (5,)


def test_expected_fields_exclude_the_summary(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    assert "complaint_summary" not in doc.expected_fields
    assert "complainant_name" in doc.expected_fields


def test_page_class_comes_from_ground_truth(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    docs = bakeoff.load_gold_docs(
        corpus,
        settings=laptop_settings,
        types=("complaint_form_en", "complaint_letter_mar"),
        variants=(0,),
    )
    assert docs[0].page_class.value == "complaint_form"
    assert docs[1].page_class.value == "complaint_letter"


def test_an_unknown_document_class_is_not_guessed(bakeoff: ModuleType) -> None:
    assert bakeoff._page_class_for({"document_class": "invoice"}).value == "unknown"


def test_limit_caps_the_sample(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    docs = bakeoff.load_gold_docs(
        corpus,
        settings=laptop_settings,
        types=("complaint_form_en", "complaint_letter_mar"),
        variants=(0,),
        limit=1,
    )
    assert len(docs) == 1


def test_a_document_without_a_synthetic_declaration_is_refused(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """The residency gate is the whole reason this run is permitted at all."""
    gt = corpus / "complaint_form_en" / "complaint_form_en_v0_clean.gt.json"
    payload = json.loads(gt.read_text(encoding="utf-8"))
    payload["synthetic"] = False
    gt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ResidencyViolation):
        bakeoff.load_gold_docs(
            corpus,
            settings=laptop_settings,
            types=("complaint_form_en",),
            variants=(0,),
        )


def test_gt_value_falls_back_to_the_tagged_line(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """``relief_sought`` has no ``fields`` entry; the tagged line is the truth."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    assert bakeoff.gt_value_for(doc, "complainant_address") == "Flat 4B, Pune 411001"
    assert bakeoff.gt_value_for(doc, "relief_sought") is None


def test_normalise_value_folds_case_digits_and_space(bakeoff: ModuleType) -> None:
    assert bakeoff.normalise_value("  Asha   PATIL ") == "asha patil"
    assert bakeoff.normalise_value("४५०००") == "45000"


# --------------------------------------------------------------------------- #
# Prompt hygiene
# --------------------------------------------------------------------------- #


def test_validator_owned_identifiers_must_not_reach_the_prompt(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """PLAN.md invariant 7, asserted on the actual rendered bytes."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(
        bakeoff.StubProvider(), doc, settings=laptop_settings
    )
    assert result.identifier_leaks == ()


def test_the_leak_detector_actually_detects(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """A detector that never fires proves nothing; make it fire."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    unmasked = "[0] Account 8234484955318053"
    assert "account_number" in bakeoff._identifier_leaks(doc, unmasked)


def test_unclaimed_pii_is_reported_separately(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    assert bakeoff._unclaimed_pii(doc, "call 9995332248") == ("mobile",)
    assert bakeoff._identifier_leaks(doc, "call 9995332248") == ()


def test_short_values_are_not_searched(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """A four-digit needle matches by accident far more often than by leak."""
    assert bakeoff._MIN_LEAK_NEEDLE >= 6


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_stub_run_scores_grounding_and_rejects_the_ungrounded(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(
        bakeoff.StubProvider(), doc, settings=laptop_settings
    )
    assert result.ok is True
    assert result.proposed_value_fields == 2
    assert result.grounded_value_fields == 1
    assert "relief_sought" in result.rejected_fields


def test_a_grounded_value_on_the_wrong_line_is_not_attributed(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """The stub always cites line 0; the name is on line 1."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(
        bakeoff.StubProvider(), doc, settings=laptop_settings
    )
    assert result.attributable_fields == 1
    assert result.attributed_fields == 0


def test_summary_indices_are_compared_against_the_tagged_lines(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(
        bakeoff.StubProvider(), doc, settings=laptop_settings
    )
    assert result.summary_expected == (4,)
    assert result.summary_predicted == (0,)


def test_each_document_gets_a_fresh_token_budget(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """One document's spend must not starve the next (BUILD-BRIEF M2)."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    provider = bakeoff.StubProvider(budget_limit=8000)
    provider.reset_budget()
    provider.budget.reserve(8000)
    result = bakeoff._run_one_document(provider, doc, settings=laptop_settings)
    assert result.ok is True
    assert provider.budget.remaining < 8000


def test_an_exhausted_budget_stops_the_call_before_it_leaves(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """Refusing here rather than sending it keeps the cost of a breach at zero."""

    class _Starved(bakeoff.StubProvider):  # type: ignore[misc, name-defined]
        def reset_budget(self, *, job_id: str | None = None) -> Any:
            budget = super().reset_budget(job_id=job_id)
            budget.reserve(budget.limit)
            return budget

    provider = _Starved(budget_limit=8000)
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(provider, doc, settings=laptop_settings)
    assert result.ok is False
    assert result.error_class == "TokenBudgetExceeded"
    assert provider.calls == 0, "nothing may be sent once the budget is gone"


# --------------------------------------------------------------------------- #
# Retries, circuit breaker and coverage honesty
# --------------------------------------------------------------------------- #


class _Flaky:
    """Fails the endpoint ``failures`` times, then answers."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def extract_fields_batch(self, ctxs: Any) -> Any:
        from app.contracts.errors import ProviderUnavailable

        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderUnavailable("endpoint refused the request")
        return [{} for _ in ctxs]


def test_an_endpoint_failure_is_retried(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    provider = _Flaky(failures=1)
    result = bakeoff._run_one_document(
        provider,
        doc,
        settings=laptop_settings,
        attempts=3,
        attempt_delay=0.0,
        sleep=lambda _seconds: None,
    )
    assert result.ok is True
    assert result.attempts == 2


class _AlwaysInvalid:
    def extract_fields_batch(self, ctxs: Any) -> Any:
        from app.contracts.errors import ProviderResponseInvalid

        raise ProviderResponseInvalid("failed validation twice")


def test_a_schema_failure_is_never_retried(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """Retrying it would make schema validity a measure of persistence."""
    doc = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )[0]
    result = bakeoff._run_one_document(
        bakeoff.StubProvider() if False else _AlwaysInvalid(),
        doc,
        settings=laptop_settings,
        attempts=5,
        attempt_delay=0.0,
        sleep=lambda _seconds: None,
    )
    assert result.attempts == 1
    assert result.responded is True, "the model answered; the answer was invalid"
    assert result.ok is False


def test_the_circuit_breaker_stops_retrying_but_still_attempts(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """Skipping outright threw away free cache replays; one attempt is cheaper."""
    docs = bakeoff.load_gold_docs(
        corpus,
        settings=laptop_settings,
        types=("complaint_form_en", "complaint_letter_mar"),
        variants=(0,),
    )
    provider = _Flaky(failures=99)
    result = bakeoff.run_candidate(
        "stub/model",
        docs,
        settings=laptop_settings,
        api_key="none",
        cache=None,
        catalogue={},
        run_q5_probe=False,
        provider_factory=lambda _name: provider,
        attempts=3,
        max_consecutive_api_errors=1,
        sleep=lambda _seconds: None,
    )
    assert result.circuit_broken is True
    assert len(result.docs) == 2, "every document is still attempted"
    assert result.docs[0].attempts == 3, "before the trip, retries apply"
    assert result.docs[1].attempts == 1, "after the trip, one attempt, no backoff"
    assert provider.calls == 4


def test_a_tripped_breaker_still_collects_cached_documents(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """The reason the breaker throttles instead of skipping."""

    class _DeadThenCached(bakeoff.StubProvider):  # type: ignore[misc, name-defined]
        def extract_fields_batch(self, ctxs: Any) -> Any:
            from app.contracts.errors import ProviderUnavailable

            self.calls += 1
            if self.calls == 1:
                raise ProviderUnavailable("endpoint refused the request")
            return super().extract_fields_batch(ctxs)

    docs = bakeoff.load_gold_docs(
        corpus,
        settings=laptop_settings,
        types=("complaint_form_en", "complaint_letter_mar"),
        variants=(0,),
    )
    result = bakeoff.run_candidate(
        "stub/model",
        docs,
        settings=laptop_settings,
        api_key="none",
        cache=None,
        catalogue={},
        run_q5_probe=False,
        provider_factory=lambda _name: _DeadThenCached(),
        max_consecutive_api_errors=1,
        sleep=lambda _seconds: None,
    )
    assert result.circuit_broken is True
    assert result.docs[1].ok is True, "the second document was still measured"


def test_api_errors_do_not_count_against_schema_validity(
    bakeoff: ModuleType,
) -> None:
    """A 402 says nothing about a model's ability to follow a schema."""
    result = bakeoff.CandidateResult(model="m", available=True)
    result.docs = [
        bakeoff.DocResult(doc_id="a", doc_type="t", script="Latn", ok=True),
        bakeoff.DocResult(
            doc_id="b",
            doc_type="t",
            script="Latn",
            ok=False,
            error_class="ProviderError",
            error="Error code: 402",
        ),
    ]
    card = bakeoff.aggregate(result)
    assert card["documents"]["model_responded"] == 1
    assert card["documents"]["api_errors"] == 1
    assert card["schema_validity_final"] == 1.0
    assert card["schema_validity_denominator"] == 1


def _mixed_result(bakeoff: ModuleType) -> Any:
    """Two cached successes and one fresh schema failure — the re-run shape."""
    result = bakeoff.CandidateResult(model="m", available=True)
    result.docs = [
        bakeoff.DocResult(
            doc_id="cached-a", doc_type="t", script="Latn", ok=True, cached=True
        ),
        bakeoff.DocResult(
            doc_id="cached-b", doc_type="t", script="Latn", ok=True, cached=True
        ),
        bakeoff.DocResult(
            doc_id="fresh-bad",
            doc_type="t",
            script="Latn",
            ok=False,
            first_attempt_valid=False,
            error_class="ProviderResponseInvalid",
            error="failed validation twice",
        ),
    ]
    return result


def test_schema_validity_scores_the_union_of_cached_and_fresh(
    bakeoff: ModuleType,
) -> None:
    """Scoring either half alone gives 1.0 or 0.0; both were observed for real."""
    card = bakeoff.aggregate(_mixed_result(bakeoff))
    assert card["schema_validity_denominator"] == 3
    assert card["schema_validity_final"] == pytest.approx(2 / 3, abs=1e-4)


def test_a_cache_hit_abstains_from_first_attempt_validity(
    bakeoff: ModuleType, tmp_path: Path
) -> None:
    """A hit proves the answer validates, not that it validated first time."""
    ledger: dict[str, dict[str, Any]] = {}
    entries = bakeoff.merge_outcomes(ledger, _mixed_result(bakeoff))
    assert entries["cached-a"]["first_attempt_valid"] is None
    assert entries["fresh-bad"]["first_attempt_valid"] is False
    card = bakeoff.aggregate(_mixed_result(bakeoff), entries)
    assert card["schema_validity_first_attempt_denominator"] == 1
    assert card["schema_validity_first_attempt"] == 0.0


def test_a_fresh_observation_overwrites_a_cached_one(
    bakeoff: ModuleType,
) -> None:
    """A measurement beats a replay; a replay never clobbers a measurement."""
    ledger: dict[str, dict[str, Any]] = {
        "m": {"a": {"valid": False, "first_attempt_valid": False, "repaired": False}}
    }
    result = bakeoff.CandidateResult(model="m", available=True)
    result.docs = [
        bakeoff.DocResult(
            doc_id="a", doc_type="t", script="Latn", ok=True, cached=True
        )
    ]
    entries = bakeoff.merge_outcomes(ledger, result)
    assert entries["a"]["valid"] is False, "a cache hit must not overwrite a record"

    result.docs = [
        bakeoff.DocResult(doc_id="a", doc_type="t", script="Latn", ok=True)
    ]
    entries = bakeoff.merge_outcomes(ledger, result)
    assert entries["a"]["valid"] is True


def test_the_outcome_ledger_round_trips(
    bakeoff: ModuleType, tmp_path: Path
) -> None:
    ledger = {"m": {"a": {"valid": True, "first_attempt_valid": True}}}
    bakeoff.save_outcomes(tmp_path, ledger)
    assert bakeoff.load_outcomes(tmp_path) == ledger


def test_a_missing_or_corrupt_ledger_is_empty_not_fatal(
    bakeoff: ModuleType, tmp_path: Path
) -> None:
    assert bakeoff.load_outcomes(tmp_path) == {}
    (tmp_path / bakeoff.OUTCOME_LEDGER_FILENAME).write_text("{oops", encoding="utf-8")
    assert bakeoff.load_outcomes(tmp_path) == {}


def test_an_endpoint_refusal_never_enters_the_ledger(
    bakeoff: ModuleType,
) -> None:
    """A 402 is not an observation of the model."""
    ledger: dict[str, dict[str, Any]] = {}
    result = bakeoff.CandidateResult(model="m", available=True)
    result.docs = [
        bakeoff.DocResult(
            doc_id="a",
            doc_type="t",
            script="Latn",
            ok=False,
            error_class="ProviderError",
            error="Error code: 402",
        )
    ]
    assert bakeoff.merge_outcomes(ledger, result) == {}


def test_a_candidate_with_no_observation_has_no_score(
    bakeoff: ModuleType,
) -> None:
    """Zero says 'scored badly'; the truth is 'not measured'."""
    card = _card("a", schema_validity_denominator=0)
    assert bakeoff.composite_score(card) is None


def test_unscored_candidates_rank_behind_measured_ones(
    bakeoff: ModuleType,
) -> None:
    measured = _card("measured", grounding_pass_rate=0.1)
    unscored = _card("unscored", schema_validity_denominator=0)
    ranked = bakeoff.rank_candidates([unscored, measured])
    assert [row["model"] for row in ranked] == ["measured", "unscored"]
    assert ranked[1]["composite_score"] is None


def test_error_messages_are_collapsed_for_counting(bakeoff: ModuleType) -> None:
    first = bakeoff._error_signature(
        "endpoint rejected the request with 402: Error code: 402 - depleted"
    )
    second = bakeoff._error_signature(
        "other model rejected: Error code: 402 - depleted"
    )
    assert first == second


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def _card(model: str, **overrides: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "model": model,
        "available": True,
        "documents": {
            "attempted": 10,
            "completed": 10,
            "model_responded": 10,
            "api_errors": 0,
            "skipped": 0,
            "cached": 0,
            "live": 10,
        },
        "schema_validity_final": 1.0,
        "schema_validity_denominator": 10,
        "grounding_pass_rate": 1.0,
        "line_attribution_accuracy": 1.0,
        "summary_f1": 1.0,
        "latency_seconds": {"p95": 1.0},
        "errors_by_message": {},
    }
    card.update(overrides)
    return card


def test_ranking_weights_sum_to_one(bakeoff: ModuleType) -> None:
    assert sum(bakeoff.RANK_WEIGHTS.values()) == pytest.approx(1.0)


def test_a_perfect_candidate_scores_one(bakeoff: ModuleType) -> None:
    assert bakeoff.composite_score(_card("a")) == 1.0


def test_an_unavailable_candidate_scores_zero(bakeoff: ModuleType) -> None:
    assert bakeoff.composite_score(_card("a", available=False)) == 0.0


def test_a_candidate_that_completed_nothing_scores_zero(
    bakeoff: ModuleType,
) -> None:
    card = _card("a")
    card["documents"]["completed"] = 0
    assert bakeoff.composite_score(card) == 0.0


def test_ranking_orders_by_score_then_latency(bakeoff: ModuleType) -> None:
    ranked = bakeoff.rank_candidates(
        [
            _card("slow", latency_seconds={"p95": 9.0}),
            _card("fast", latency_seconds={"p95": 1.0}),
            _card("worse", grounding_pass_rate=0.0),
        ]
    )
    assert [row["model"] for row in ranked] == ["fast", "slow", "worse"]
    assert ranked[0]["rank"] == 1


def test_value_agreement_is_not_in_the_score(bakeoff: ModuleType) -> None:
    """Q6: GT carries English values on Devanagari pages."""
    assert "value_agreement_exact" not in bakeoff.RANK_WEIGHTS
    assert "value_agreement_fuzzy" not in bakeoff.RANK_WEIGHTS


# --------------------------------------------------------------------------- #
# Q5
# --------------------------------------------------------------------------- #


def _q5(**overrides: Any) -> dict[str, Any]:
    row = {
        "json_object_accepted": True,
        "json_schema_accepted": True,
        "json_object_yields_json": True,
        "unconstrained_yields_json": True,
        "router_advertises_structured_output": True,
        "schema_validity_first_attempt": 1.0,
    }
    row.update(overrides)
    return row


def test_q5_says_grammar_available_when_every_model_takes_a_schema(
    bakeoff: ModuleType,
) -> None:
    assert "constrained decoding available" in bakeoff._q5_verdict([_q5(), _q5()])


def test_q5_says_json_mode_only_when_no_schema_is_taken(
    bakeoff: ModuleType,
) -> None:
    rows = [_q5(json_schema_accepted=False), _q5(json_schema_accepted=False)]
    assert "JSON mode only" in bakeoff._q5_verdict(rows)


def test_q5_reports_a_mixed_router(bakeoff: ModuleType) -> None:
    rows = [_q5(), _q5(json_schema_accepted=False)]
    assert bakeoff._q5_verdict(rows).startswith("mixed")


def test_q5_says_nothing_when_nothing_answered(bakeoff: ModuleType) -> None:
    assert "not measured" in bakeoff._q5_verdict([])


def test_json_parse_check_tolerates_a_fence(bakeoff: ModuleType) -> None:
    assert bakeoff._parses_as_json('```json\n{"a": 1}\n```') is True
    assert bakeoff._parses_as_json("not json") is False
    assert bakeoff._parses_as_json("[1, 2]") is False, "the contract is an object"
    assert bakeoff._parses_as_json(None) is False


def test_a_402_only_probe_is_not_conclusive(bakeoff: ModuleType) -> None:
    """A depleted credit balance is a billing state, not a capability claim."""
    probe = {
        "variants": {
            "json_object": {"status_code": 402, "error": "depleted"},
            "none": {"status_code": 402, "error": "depleted"},
            "json_schema": {"status_code": 402, "error": "depleted"},
        }
    }
    assert bakeoff.probe_is_conclusive(probe) is False


def test_a_model_not_found_probe_is_conclusive(bakeoff: ModuleType) -> None:
    probe = {
        "variants": {
            "json_object": {"status_code": 400, "error": "no model"},
            "none": {"status_code": 400},
            "json_schema": {"status_code": 400},
        }
    }
    assert bakeoff.probe_is_conclusive(probe) is True


def test_a_successful_probe_is_conclusive(bakeoff: ModuleType) -> None:
    probe = {
        "variants": {
            "json_object": {"status_code": 200},
            "none": {"status_code": 200},
            # 422 is the provider refusing the format — a real answer.
            "json_schema": {"status_code": 422},
        }
    }
    assert bakeoff.probe_is_conclusive(probe) is True


def test_a_partly_refused_probe_is_not_conclusive(bakeoff: ModuleType) -> None:
    """Otherwise a 402 on one variant becomes a permanent false capability claim."""
    probe = {
        "variants": {
            "json_object": {"status_code": 200},
            "none": {"status_code": 200},
            "json_schema": {"status_code": 402, "error": "depleted"},
        }
    }
    assert bakeoff.probe_is_conclusive(probe) is False


def test_a_transport_failure_is_not_conclusive(bakeoff: ModuleType) -> None:
    probe = {
        "variants": {
            "json_object": {"status_code": None, "transport_error": "ReadTimeout"},
        }
    }
    assert bakeoff.probe_is_conclusive(probe) is False


def test_a_server_error_is_not_conclusive(bakeoff: ModuleType) -> None:
    probe = {"variants": {"json_object": {"status_code": 503}}}
    assert bakeoff.probe_is_conclusive(probe) is False


def test_an_empty_probe_is_not_conclusive(bakeoff: ModuleType) -> None:
    assert bakeoff.probe_is_conclusive({}) is False


def test_inconclusive_probes_are_excluded_from_the_verdict(
    bakeoff: ModuleType,
) -> None:
    answer = bakeoff._q5_answer(
        [
            {
                "model": "measured",
                "q5_json_mode": {
                    "variants": {"json_object": {"status_code": 200}},
                    "json_object_accepted": True,
                    "json_object_yields_json": True,
                    "json_schema_accepted": False,
                    "unconstrained_yields_json": True,
                },
                "router": {},
                "schema_validity_first_attempt": 1.0,
            },
            {
                "model": "refused",
                "q5_json_mode": {
                    "variants": {"json_object": {"status_code": 402}},
                    "json_object_accepted": False,
                },
                "router": {},
                "schema_validity_first_attempt": 0.0,
            },
        ]
    )
    assert answer["models_probed"] == 1
    assert answer["models_inconclusive"] == 1
    assert "JSON mode only" in answer["verdict"]


def test_q5_probes_round_trip_through_the_cache_file(
    bakeoff: ModuleType, tmp_path: Path
) -> None:
    probes = {"a/b": {"variants": {"json_object": {"status_code": 200}}}}
    bakeoff.save_q5_cache(tmp_path, probes)
    assert bakeoff.load_q5_cache(tmp_path) == probes


def test_a_missing_or_corrupt_q5_cache_is_empty_not_fatal(
    bakeoff: ModuleType, tmp_path: Path
) -> None:
    assert bakeoff.load_q5_cache(tmp_path) == {}
    (tmp_path / bakeoff.Q5_CACHE_FILENAME).write_text("{not json", encoding="utf-8")
    assert bakeoff.load_q5_cache(tmp_path) == {}


def test_a_remembered_probe_is_reused_instead_of_re_measured(
    bakeoff: ModuleType, corpus: Path, laptop_settings: Settings
) -> None:
    """Once credits are gone the probe cannot be re-taken; do not lose it."""
    remembered = {
        "variants": {"json_object": {"status_code": 200}},
        "json_object_accepted": True,
        "probed_at": "2026-08-07T00:00:00+00:00",
    }
    docs = bakeoff.load_gold_docs(
        corpus, settings=laptop_settings, types=("complaint_form_en",), variants=(0,)
    )
    result = bakeoff.run_candidate(
        "a/b",
        docs,
        settings=laptop_settings,
        api_key="none",
        cache=None,
        catalogue={"a/b": {"advertises_structured_output": True}},
        run_q5_probe=True,
        provider_factory=lambda _name: bakeoff.StubProvider(),
        q5_cache={"a/b": remembered},
    )
    assert result.q5["probed_at"] == "2026-08-07T00:00:00+00:00"


def test_catalogue_summary_extracts_structured_output_support(
    bakeoff: ModuleType,
) -> None:
    summary = bakeoff.summarise_catalogue_entry(
        {
            "owned_by": "qwen",
            "architecture": {"input_modalities": ["text"]},
            "providers": [
                {"provider": "a", "status": "live", "supports_structured_output": False},
                {"provider": "b", "status": "live", "supports_structured_output": True},
            ],
        }
    )
    assert summary["advertises_structured_output"] is True
    assert len(summary["providers"]) == 2


# --------------------------------------------------------------------------- #
# End to end, dry run
# --------------------------------------------------------------------------- #


@pytest.fixture
def dry_report(
    bakeoff: ModuleType, laptop_settings: Settings, corpus: Path
) -> dict[str, Any]:
    args = bakeoff.build_parser().parse_args(
        ["--dry-run", "--model", "stub/one", "--model", "stub/two"]
    )
    return bakeoff.run(args, settings=laptop_settings)


def test_dry_run_produces_a_full_report(dry_report: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "mode",
        "settings",
        "sample",
        "router",
        "ranking",
        "candidates",
        "q5_answer",
        "coverage",
        "prompt_hygiene",
    ):
        assert key in dry_report, key
    assert dry_report["mode"] == "dry_run"
    assert len(dry_report["candidates"]) == 2


def test_dry_run_report_is_json_serialisable(dry_report: dict[str, Any]) -> None:
    assert json.loads(json.dumps(dry_report, ensure_ascii=False))["mode"] == "dry_run"


def test_dry_run_coverage_is_complete(dry_report: dict[str, Any]) -> None:
    assert dry_report["coverage"]["completion_rate"] == 1.0
    assert dry_report["coverage"]["observation_rate"] == 1.0
    assert dry_report["coverage"]["api_errors"] == 0


def test_a_schema_failure_still_counts_as_an_observation(
    bakeoff: ModuleType,
) -> None:
    """Otherwise a model that reliably breaks the schema reads as unmeasured."""
    card = bakeoff.aggregate(
        bakeoff.CandidateResult(
            model="m",
            available=True,
            docs=[
                bakeoff.DocResult(doc_id="a", doc_type="t", script="Latn", ok=True),
                bakeoff.DocResult(
                    doc_id="b",
                    doc_type="t",
                    script="Latn",
                    ok=False,
                    error_class="ProviderResponseInvalid",
                    error="failed validation twice",
                ),
            ],
        )
    )
    coverage = bakeoff._coverage([card])
    assert coverage["observation_rate"] == 1.0
    assert coverage["completion_rate"] == 0.5
    assert coverage["api_errors"] == 0


def test_dry_run_finds_no_invariant_seven_leak(dry_report: dict[str, Any]) -> None:
    assert dry_report["prompt_hygiene"]["invariant_7_identifier_leaks"] == 0


def test_markdown_renders_without_a_network(
    bakeoff: ModuleType, dry_report: dict[str, Any]
) -> None:
    text = bakeoff.render_markdown(dry_report)
    assert text.startswith("# SLM bake-off")
    assert "## Recommendation" in text
    assert "## Q5" in text
    assert "## Ranking" in text
    assert "## Caveats" in text
    assert text.endswith("\n")


def test_markdown_warns_loudly_on_a_partial_run(
    bakeoff: ModuleType, dry_report: dict[str, Any]
) -> None:
    partial = dict(dry_report)
    partial["coverage"] = {
        **dry_report["coverage"],
        "completion_rate": 0.2,
        "observation_rate": 0.2,
        "documents_completed": 2,
        "documents_model_responded": 2,
        "documents_attempted": 10,
        "api_errors": 8,
        "skipped": 0,
        "endpoint_failure_messages": {"Error code: 402 - depleted": 8},
    }
    text = bakeoff.render_markdown(partial)
    assert "Read this first" in text
    assert "402" in text


def test_markdown_names_the_unserved_configured_models(
    bakeoff: ModuleType, dry_report: dict[str, Any]
) -> None:
    text = bakeoff.render_markdown(dry_report)
    assert "not served" in text or "not on the router" in text


def test_main_writes_both_artefacts(
    bakeoff: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bakeoff"
    code = bakeoff.main(["--dry-run", "--docs", "1", "--out-dir", str(out)])
    assert code == 0
    assert (out / "results.json").is_file()
    assert (out / "REPORT.md").is_file()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "dry_run"


def test_no_write_leaves_no_files(bakeoff: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "bakeoff"
    assert bakeoff.main(["--dry-run", "--docs", "1", "--out-dir", str(out), "--no-write"]) == 0
    assert not out.exists()


# --------------------------------------------------------------------------- #
# The committed artefacts
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed(bakeoff: ModuleType) -> dict[str, Any]:
    path = PROJECT_ROOT / "docs" / "bakeoff" / "results.json"
    if not path.is_file():
        pytest.skip("docs/bakeoff/results.json has not been generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_results_match_the_schema(
    bakeoff: ModuleType, committed: dict[str, Any]
) -> None:
    assert committed["schema_version"] == bakeoff.REPORT_SCHEMA_VERSION
    assert committed["mode"] == "measured"
    assert committed["sample"]["documents"] >= 15, "M7 asks for ~15-20 documents"
    assert committed["sample"]["locked_test_variants_excluded"] == [3, 4]


def test_committed_results_cover_at_least_three_candidates(
    committed: dict[str, Any],
) -> None:
    assert len(committed["candidates"]) >= 3


def test_committed_results_cover_both_scripts(committed: dict[str, Any]) -> None:
    assert set(committed["sample"]["by_script"]) >= {"Deva", "Latn"}


def test_committed_results_answer_q5(committed: dict[str, Any]) -> None:
    answer = committed["q5_answer"]
    assert answer["verdict"]
    assert answer["per_model"]


def test_committed_run_leaked_no_identifier(committed: dict[str, Any]) -> None:
    """PLAN.md invariant 7, asserted against the run that actually happened."""
    assert committed["prompt_hygiene"]["invariant_7_identifier_leaks"] == 0


def test_committed_report_markdown_exists() -> None:
    path = PROJECT_ROOT / "docs" / "bakeoff" / "REPORT.md"
    if not path.is_file():
        pytest.skip("docs/bakeoff/REPORT.md has not been generated")
    text = path.read_text(encoding="utf-8")
    assert "# SLM bake-off" in text
    assert "Q5" in text
