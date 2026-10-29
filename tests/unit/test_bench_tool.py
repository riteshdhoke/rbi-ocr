"""Unit tests for ``tools/bench_ocr_cpu.py`` — the M0 throughput spike.

Nothing here loads a weight, decodes a corpus page or measures anything real.
What is tested is the part of a benchmark that can silently be wrong and would
then produce a confident number nobody can challenge: the sampling (is it
stable? does it actually spread across profiles?), the percentile convention,
the gate arithmetic, the phase timer's thread isolation, and the shape of
``bench_results.json``.

The dry-run path exists precisely so this file can exercise the whole tool end
to end on a machine with an empty ``models/``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    """Load the tool as a module (``tools/`` is not a package)."""
    path = PROJECT_ROOT / "tools" / "bench_ocr_cpu.py"
    spec = importlib.util.spec_from_file_location("bench_ocr_cpu_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def index() -> dict[str, Any]:
    """A miniature corpus index: 3 types x 3 profiles x 2 variants."""
    documents = []
    for doc_type in ("complaint_form_en", "complaint_letter_mar", "receipt"):
        for profile in ("clean", "scan", "bounce"):
            for variant in (0, 1):
                documents.append(
                    {
                        "doc_id": f"{doc_type}_v{variant}_{profile}",
                        "type": doc_type,
                        "variant": variant,
                        "profile": profile,
                        "image": f"{doc_type}\\{doc_type}_v{variant}_{profile}.jpg",
                        "lines": 12,
                        "primary_script": (
                            "Deva" if doc_type.endswith("_mar") else "Latn"
                        ),
                    }
                )
    return {
        "seed": 1729,
        "generator_version": "1.1.0",
        "count": len(documents),
        "profiles": ["clean", "scan", "bounce"],
        "documents": documents,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_defaults_match_the_brief(bench: ModuleType) -> None:
    args = bench.build_parser().parse_args([])
    assert args.pages == 100
    assert args.threshold_seconds == 4.0
    assert args.dry_run is False
    assert args.out.name == "bench_results.json"


def test_concurrency_is_repeatable(bench: ModuleType) -> None:
    args = bench.build_parser().parse_args(["--concurrency", "1", "--concurrency", "4"])
    assert args.concurrency == [1, 4]


def test_gate_threshold_is_four_seconds(bench: ModuleType) -> None:
    """BUILD-BRIEF M0 states the number; it must not drift silently."""
    assert bench.GATE_P95_SECONDS == 4.0


def test_the_brief_named_batch_sizes_are_measured(bench: ModuleType) -> None:
    assert 16 in bench.BATCH_SIZES
    assert 32 in bench.BATCH_SIZES


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_is_seed_stable(bench: ModuleType, index: dict[str, Any]) -> None:
    kwargs = {"count": 8, "corpus_dir": Path("corpus")}
    first = bench.sample_documents(index, seed=11, **kwargs)
    second = bench.sample_documents(index, seed=11, **kwargs)
    assert [doc.doc_id for doc in first] == [doc.doc_id for doc in second]


def test_a_different_seed_gives_a_different_order(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    kwargs = {"count": 8, "corpus_dir": Path("corpus")}
    first = [doc.doc_id for doc in bench.sample_documents(index, seed=1, **kwargs)]
    second = [doc.doc_id for doc in bench.sample_documents(index, seed=2, **kwargs)]
    assert first != second


def test_sampling_spreads_over_type_and_profile_cells(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    """The reason for round-robin: nine cells, nine pages, one each."""
    docs = bench.sample_documents(
        index, count=9, seed=5, corpus_dir=Path("corpus")
    )
    cells = {(doc.doc_type, doc.profile) for doc in docs}
    assert len(cells) == 9


def test_sampling_never_repeats_a_document(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    docs = bench.sample_documents(
        index, count=18, seed=3, corpus_dir=Path("corpus")
    )
    assert len(docs) == 18
    assert len({doc.doc_id for doc in docs}) == 18


def test_asking_for_more_pages_than_exist_returns_what_there_is(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    docs = bench.sample_documents(
        index, count=500, seed=3, corpus_dir=Path("corpus")
    )
    assert len(docs) == len(index["documents"])


def test_profile_filter_is_honoured(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    docs = bench.sample_documents(
        index, count=6, seed=3, corpus_dir=Path("corpus"), profiles=["clean"]
    )
    assert {doc.profile for doc in docs} == {"clean"}


def test_an_impossible_profile_filter_raises(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="no corpus documents matched"):
        bench.sample_documents(
            index, count=6, seed=3, corpus_dir=Path("corpus"), profiles=["nope"]
        )


def test_zero_pages_is_a_caller_bug(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        bench.sample_documents(index, count=0, seed=3, corpus_dir=Path("corpus"))


def test_windows_index_paths_become_real_paths(
    bench: ModuleType, index: dict[str, Any]
) -> None:
    """``index.json`` stores backslash-separated paths; POSIX must still work."""
    docs = bench.sample_documents(
        index, count=1, seed=3, corpus_dir=Path("corpus")
    )
    assert docs[0].image_path.parent.name == docs[0].doc_type
    assert "\\" not in docs[0].image_path.name


def test_a_malformed_index_is_rejected(bench: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a corpus index"):
        bench.load_index(tmp_path)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_percentile_returns_an_observed_value(bench: ModuleType) -> None:
    """Nearest-rank, higher neighbour: every reported number is a real page."""
    values = [1.0, 2.0, 3.0, 4.0]
    assert bench.percentile(values, 95) in values
    # "higher" never interpolates: the p50 of four points is the upper
    # neighbour (3.0), not the 2.5 an interpolating percentile would invent.
    assert bench.percentile(values, 50) == 3.0
    assert bench.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert bench.percentile(values, 100) == 4.0
    assert bench.percentile(values, 0) == 1.0


def test_percentile_of_nothing_is_zero(bench: ModuleType) -> None:
    assert bench.percentile([], 95) == 0.0


def test_summarise_reports_the_full_shape(bench: ModuleType) -> None:
    stats = bench.summarise([0.1, 0.2, 0.3])
    assert stats["n"] == 3
    assert stats["min"] == 0.1
    assert stats["max"] == 0.3
    assert stats["p50"] == 0.2


def test_summarise_of_nothing_says_so(bench: ModuleType) -> None:
    assert bench.summarise([]) == {"n": 0}


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #


def test_phase_timer_is_thread_local(bench: ModuleType) -> None:
    """Two workers timing two pages must not pool their detection time."""
    timer = bench.PhaseTimer()
    seen: dict[str, float] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, seconds: float) -> None:
        timer.reset()
        barrier.wait()
        timer.add("detect", seconds)
        seen[name] = timer.snapshot()["detect"]

    threads = [
        threading.Thread(target=worker, args=("a", 1.0)),
        threading.Thread(target=worker, args=("b", 5.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == {"a": 1.0, "b": 5.0}


def test_phase_timer_buckets_unknown_phases(bench: ModuleType) -> None:
    timer = bench.PhaseTimer()
    timer.reset()
    timer.add("nonsense", 2.0)
    assert timer.snapshot()["other"] == 2.0


def test_phase_of_maps_model_filenames(bench: ModuleType) -> None:
    assert bench.phase_of("det.onnx") == "detect"
    assert bench.phase_of("rec_deva.onnx") == "recognise"
    assert bench.phase_of("page_classifier.onnx") == "other"


def test_timed_session_delegates_and_times(bench: ModuleType) -> None:
    timer = bench.PhaseTimer()
    timer.reset()
    inner = SimpleNamespace(
        run=lambda names, feed: ["out"],
        get_inputs=lambda: [SimpleNamespace(name="x")],
        marker="delegated",
    )
    session = bench.TimedSession(inner, "recognise", timer)
    assert session.run(None, {"x": 1}) == ["out"]
    assert session.get_inputs()[0].name == "x"
    assert session.marker == "delegated"
    assert session.calls == 1
    assert timer.snapshot()["recognise"] >= 0.0
    assert timer.snapshot()["recognise_calls"] == 1.0


def test_timed_session_records_time_even_when_run_raises(
    bench: ModuleType,
) -> None:
    timer = bench.PhaseTimer()
    timer.reset()

    def boom(names: Any, feed: Any) -> Any:
        raise RuntimeError("session died")

    session = bench.TimedSession(SimpleNamespace(run=boom), "detect", timer)
    with pytest.raises(RuntimeError, match="session died"):
        session.run(None, {})
    assert timer.snapshot()["detect_calls"] == 1.0


def test_crop_sink_only_captures_when_enabled(bench: ModuleType) -> None:
    sink = bench.CropSink(limit=2)
    crop = np.zeros((4, 8, 3), dtype=np.uint8)
    assert sink(crop) is False
    assert sink.crops == []
    sink.enabled = True
    for _ in range(5):
        sink(crop)
    assert len(sink.crops) == 2, "the limit must bound memory across 100 pages"


def test_crop_sink_never_claims_handwriting(bench: ModuleType) -> None:
    """There is no approved detector; the benchmark must not invent one."""
    sink = bench.CropSink()
    sink.enabled = True
    assert sink(np.zeros((4, 8, 3), dtype=np.uint8)) is False


# --------------------------------------------------------------------------- #
# The looped baseline
# --------------------------------------------------------------------------- #


def test_exact_width_plan_is_one_crop_per_call(bench: ModuleType) -> None:
    crops = [
        np.zeros((48, 100, 3), dtype=np.uint8),
        np.zeros((24, 100, 3), dtype=np.uint8),
    ]
    plan = bench.exact_width_plan(crops)
    assert [batch.indices for batch in plan] == [(0,), (1,)]
    # 48-high crop keeps its width; 24-high doubles under height normalisation.
    assert [batch.target_width for batch in plan] == [100, 200]


def test_exact_width_plan_clamps_to_the_widest_bucket(bench: ModuleType) -> None:
    from app.providers.local import REC_WIDTH_BUCKETS

    plan = bench.exact_width_plan([np.zeros((10, 4000, 3), dtype=np.uint8)])
    assert plan[0].target_width == REC_WIDTH_BUCKETS[-1]


# --------------------------------------------------------------------------- #
# Concurrency measurement
# --------------------------------------------------------------------------- #


def _pages(bench: ModuleType, count: int) -> list[Any]:
    docs = [
        bench.BenchDoc(
            doc_id=f"doc_{i}",
            doc_type="complaint_form_en",
            profile="clean" if i % 2 else "scan",
            script="Latn",
            variant=0,
            image_path=Path(f"doc_{i}.jpg"),
            expected_lines=3,
        )
        for i in range(count)
    ]
    return bench.synthesise_pages(docs)


def test_measure_concurrency_reports_every_page(bench: ModuleType) -> None:
    provider = bench.DryRunProvider(lines_per_page=3, seconds_per_page=0.001)
    result = bench.measure_concurrency(
        provider, bench.PhaseTimer(), _pages(bench, 6), workers=1
    )
    assert result["pages_attempted"] == 6
    assert result["pages_measured"] == 6
    assert result["errors"] == []
    assert result["lines_per_page"]["p50"] == 3.0
    assert set(result["by_profile"]) == {"clean", "scan"}


def test_measure_concurrency_runs_the_thread_pool(bench: ModuleType) -> None:
    provider = bench.DryRunProvider(lines_per_page=2, seconds_per_page=0.001)
    result = bench.measure_concurrency(
        provider, bench.PhaseTimer(), _pages(bench, 6), workers=2
    )
    assert result["workers"] == 2
    assert result["pages_measured"] == 6
    assert provider.calls == 6


def test_a_failing_page_is_recorded_and_excluded(bench: ModuleType) -> None:
    """A typed failure is a correctness finding, not a fast page."""

    class Exploding:
        def ocr_page(self, page: Any, script: str) -> list[Any]:
            raise RuntimeError("no rec head for Latn")

    result = bench.measure_concurrency(
        Exploding(), bench.PhaseTimer(), _pages(bench, 3), workers=1
    )
    assert result["pages_measured"] == 0
    assert len(result["errors"]) == 3
    assert "no rec head" in result["errors"][0]["error"]
    assert result["per_page_seconds"] == {"n": 0}


def test_zero_workers_is_rejected(bench: ModuleType) -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        bench.measure_concurrency(
            bench.DryRunProvider(), bench.PhaseTimer(), _pages(bench, 1), workers=0
        )


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def _level(p95: float, by_profile: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "per_page_seconds": {"p50": 0.5, "p95": p95},
        "by_profile": {
            profile: {"p95": value}
            for profile, value in (by_profile or {"clean": p95}).items()
        },
        "pages_per_minute": 60.0,
    }


def test_gate_passes_under_the_threshold(bench: ModuleType) -> None:
    gate = bench.evaluate_gate({"2": _level(3.9)}, target_concurrency=2)
    assert gate["result"] == "PASS"
    assert gate["profiles_over_threshold"] == {}


def test_gate_fails_over_the_threshold(bench: ModuleType) -> None:
    gate = bench.evaluate_gate(
        {"2": _level(4.5, {"clean": 1.0, "photo": 4.5})}, target_concurrency=2
    )
    assert gate["result"] == "FAIL"
    assert gate["profiles_over_threshold"] == {"photo": 4.5}


def test_gate_is_exactly_at_the_boundary(bench: ModuleType) -> None:
    """4.0 passes; the brief says p95 > 4 s is the failure."""
    assert bench.evaluate_gate({"2": _level(4.0)}, target_concurrency=2)["result"] == (
        "PASS"
    )


def test_a_dry_run_gate_is_advisory_not_a_pass(bench: ModuleType) -> None:
    gate = bench.evaluate_gate({"2": _level(0.01)}, target_concurrency=2, advisory=True)
    assert gate["result"] == "ADVISORY"
    assert gate["advisory"] is True
    assert "stub" in gate["note"]


def test_an_unmeasured_concurrency_says_so(bench: ModuleType) -> None:
    gate = bench.evaluate_gate({"1": _level(1.0)}, target_concurrency=4)
    assert gate["result"] == "NOT_MEASURED"


# --------------------------------------------------------------------------- #
# Machine description
# --------------------------------------------------------------------------- #


def test_machine_info_is_complete_and_non_identifying(bench: ModuleType) -> None:
    info = bench.machine_info()
    assert info["cpu_model"]
    assert info["cpu_cores_logical"] >= 1
    assert info["platform"]
    assert len(info["hostname_hashed"]) == 12
    assert bench.machine_info()["hostname_hashed"] == info["hostname_hashed"]


def test_thread_configuration_reports_the_caps_in_effect(
    bench: ModuleType, settings: Any
) -> None:
    config = bench.thread_configuration(settings)
    assert config["ort_intra_op_threads"] == settings.ORT_INTRA_OP_THREADS
    assert config["ort_inter_op_threads"] == 1
    assert "OMP_NUM_THREADS" in config["environment"]
    # Captured at import, not at report time: by report time NumPy and ONNX
    # Runtime are legitimately loaded and this would always look like a breach.
    assert config["thread_caps_honoured"] is (
        not bench.NUMERIC_LOADED_BEFORE_APP
    )


# --------------------------------------------------------------------------- #
# End to end, dry run
# --------------------------------------------------------------------------- #


@pytest.fixture
def dry_report(bench: ModuleType, tmp_path: Path) -> dict[str, Any]:
    args = bench.build_parser().parse_args(
        ["--dry-run", "--pages", "6", "--out", str(tmp_path / "bench.json")]
    )
    return bench.run(args)


def test_dry_run_produces_a_full_report(dry_report: dict[str, Any]) -> None:
    for key in (
        "schema_version",
        "mode",
        "machine",
        "runtime",
        "threads",
        "corpus",
        "concurrency",
        "batching",
        "gate",
        "pages",
    ):
        assert key in dry_report, key
    assert dry_report["mode"] == "dry_run"
    assert dry_report["corpus"]["pages_sampled"] == 6


def test_dry_run_measures_both_concurrency_levels(
    dry_report: dict[str, Any],
) -> None:
    assert set(dry_report["concurrency"]) == {"1", "2"}
    for payload in dry_report["concurrency"].values():
        assert payload["pages_measured"] == 6
        assert "_timings" not in payload, "internal state must not reach the file"


def test_dry_run_pages_carry_both_levels(dry_report: dict[str, Any]) -> None:
    assert len(dry_report["pages"]) == 6
    for row in dry_report["pages"]:
        assert "seconds_c1" in row
        assert "seconds_c2" in row


def test_dry_run_report_is_json_serialisable(dry_report: dict[str, Any]) -> None:
    text = json.dumps(dry_report, ensure_ascii=False)
    assert json.loads(text)["mode"] == "dry_run"


def test_dry_run_writes_the_file_and_exits_zero(
    bench: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nested" / "bench.json"
    code = bench.main(["--dry-run", "--pages", "4", "--out", str(out)])
    assert code == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"]["result"] == "ADVISORY"
    assert "GATE" in capsys.readouterr().out


def test_no_write_leaves_no_file(bench: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    assert bench.main(["--dry-run", "--pages", "2", "--out", str(out), "--no-write"]) == 0
    assert not out.exists()


# --------------------------------------------------------------------------- #
# The committed artefact
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def committed(bench: ModuleType) -> dict[str, Any]:
    path = PROJECT_ROOT / "bench_results.json"
    if not path.is_file():
        pytest.skip("bench_results.json has not been generated on this machine")
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_results_match_the_schema(
    bench: ModuleType, committed: dict[str, Any]
) -> None:
    assert committed["schema_version"] == bench.REPORT_SCHEMA_VERSION
    assert committed["mode"] == "measured"
    assert committed["machine"]["cpu_model"]
    assert committed["corpus"]["pages_sampled"] >= 1


def test_committed_results_answer_the_gate(committed: dict[str, Any]) -> None:
    gate = committed["gate"]
    assert gate["result"] in {"PASS", "FAIL"}
    assert gate["threshold_seconds"] == 4.0
    assert gate["p95_seconds"] > 0.0


def test_committed_results_compare_looped_against_batched(
    bench: ModuleType, committed: dict[str, Any]
) -> None:
    studies = [
        study
        for study in committed["batching"].values()
        if isinstance(study, dict) and "speedup_vs_looped" in study
    ]
    assert studies, "the brief requires a looped-vs-batched comparison"
    for study in studies:
        assert "batch_16" in study["speedup_vs_looped"]
        assert "batch_32" in study["speedup_vs_looped"]


def test_batching_does_not_change_the_recognised_text(
    committed: dict[str, Any],
) -> None:
    """The load-bearing assertion behind PLAN.md invariant 4."""
    for study in committed["batching"].values():
        if not isinstance(study, dict):
            continue
        agreement = study.get("text_agreement_vs_looped", {})
        for label, value in agreement.items():
            if label.startswith("batch_"):
                assert value == 1.0, f"{label} changed the decoded text"
