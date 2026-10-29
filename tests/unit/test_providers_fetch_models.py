"""Unit tests for ``tools/fetch_models.py`` — offline parts only.

The tool is dev-only and the orchestrator runs it after review; nothing here
downloads anything or touches the network.  What is tested is the part that can
silently be wrong: argument parsing, pattern resolution against a repository
listing, the licence flag, and the provenance record a deployment audit reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fetch_models() -> ModuleType:
    """Load the tool as a module (``tools/`` is not a package)."""
    path = PROJECT_ROOT / "tools" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_parser_defaults_to_every_group(fetch_models: ModuleType) -> None:
    args = fetch_models.build_parser().parse_args([])
    assert args.only is None
    assert args.dry_run is False
    assert args.strict_licence is False
    assert args.models_dir.name == "models"


def test_groups_can_be_selected(fetch_models: ModuleType) -> None:
    args = fetch_models.build_parser().parse_args(["--only", "lid", "--only", "ocr"])
    assert args.only == ["lid", "ocr"]


def test_an_unknown_group_is_rejected(fetch_models: ModuleType) -> None:
    with pytest.raises(SystemExit):
        fetch_models.build_parser().parse_args(["--only", "slm"])


def test_the_slm_is_not_a_downloadable_group(fetch_models: ModuleType) -> None:
    """PLAN.md D1: >=0.9B weights never land on this machine."""
    assert set(fetch_models.ARTIFACT_GROUPS) == {"lid", "ocr", "mt"}


# --------------------------------------------------------------------------- #
# Pattern resolution
# --------------------------------------------------------------------------- #


_LISTING = [
    "README.md",
    "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx",
    "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx",
    "onnx/PP-OCRv4/det/ch_PP-OCRv4_mobile_det.onnx",
    "resources/dict/ppocrv5_keys.txt",
]


def test_detection_model_resolves_to_the_v5_mobile_head(
    fetch_models: ModuleType,
) -> None:
    resolved = fetch_models._first_match(
        _LISTING, fetch_models.OCR_PATTERNS["det"]
    )
    assert resolved == "onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"


def test_dictionary_resolves(fetch_models: ModuleType) -> None:
    resolved = fetch_models._first_match(
        _LISTING, fetch_models.OCR_PATTERNS["dict_latn"]
    )
    assert resolved == "resources/dict/ppocrv5_keys.txt"


def test_a_missing_devanagari_head_resolves_to_nothing(
    fetch_models: ModuleType,
) -> None:
    """OPEN-QUESTIONS Q1: the tool reports the gap, it does not invent a path."""
    assert (
        fetch_models._first_match(_LISTING, fetch_models.OCR_PATTERNS["rec_deva"])
        is None
    )


def test_pattern_preference_order_is_honoured(fetch_models: ModuleType) -> None:
    """A v5 head must win over a v4 head present in the same repository."""
    resolved = fetch_models._first_match(
        _LISTING, fetch_models.OCR_PATTERNS["det"]
    )
    assert "v5" in resolved.lower()


def test_the_required_set_excludes_devanagari(fetch_models: ModuleType) -> None:
    assert "rec_deva" not in fetch_models.OCR_REQUIRED
    assert set(fetch_models.OCR_REQUIRED) <= set(fetch_models.OCR_PATTERNS)


# --------------------------------------------------------------------------- #
# Licence bookkeeping
# --------------------------------------------------------------------------- #


def test_the_fasttext_model_is_flagged_out_of_gate(
    fetch_models: ModuleType,
) -> None:
    """CC-BY-SA-3.0 is outside PLAN.md invariant 12; it must not pass quietly."""
    assert fetch_models.LID_LICENCE not in fetch_models.LICENCE_ALLOWLIST


def test_ocr_and_mt_licences_are_inside_the_gate(
    fetch_models: ModuleType,
) -> None:
    assert fetch_models.OCR_LICENCE in fetch_models.LICENCE_ALLOWLIST
    assert fetch_models.MT_LICENCE in fetch_models.LICENCE_ALLOWLIST


def test_strict_licence_refuses_the_out_of_gate_artifact(
    fetch_models: ModuleType, tmp_path: Path
) -> None:
    report = fetch_models.FetchReport()
    fetch_models.fetch_lid(tmp_path, report, dry_run=False, strict=True)
    assert report.records[0].status == "refused"
    assert not (tmp_path / "lid").exists()


def test_dry_run_downloads_nothing(
    fetch_models: ModuleType, tmp_path: Path
) -> None:
    report = fetch_models.FetchReport()
    fetch_models.fetch_lid(tmp_path, report, dry_run=True, strict=False)
    assert report.records[0].status == "dry-run"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_provenance_records_the_audit_fields(
    fetch_models: ModuleType, tmp_path: Path
) -> None:
    report = fetch_models.FetchReport()
    report.add(
        fetch_models.ArtifactRecord(
            group="lid",
            role="fasttext-lid-176",
            destination="lid/lid.176.ftz",
            source=fetch_models.LID_URL,
            licence=fetch_models.LID_LICENCE,
            sha256="0" * 64,
            size_bytes=938013,
        )
    )
    path = fetch_models.write_provenance(tmp_path, report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["artifacts"][0]
    assert entry["licence_in_code_allowlist"] is False
    assert entry["sha256"] == "0" * 64
    assert entry["source"].endswith("lid.176.ftz")
    assert payload["licence_allowlist"] == sorted(fetch_models.LICENCE_ALLOWLIST)


def test_provenance_is_utf8_and_newline_terminated(
    fetch_models: ModuleType, tmp_path: Path
) -> None:
    path = fetch_models.write_provenance(tmp_path, fetch_models.FetchReport())
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_problems_are_surfaced_in_the_report(
    fetch_models: ModuleType, tmp_path: Path
) -> None:
    report = fetch_models.FetchReport()
    report.add(
        fetch_models.ArtifactRecord(
            group="ocr",
            role="rec_deva",
            destination="ocr/rec_deva.onnx",
            source="repo",
            licence="Apache-2.0",
            status="unavailable",
            note="no pre-converted Devanagari head found",
        )
    )
    payload = json.loads(
        fetch_models.write_provenance(tmp_path, report).read_text(encoding="utf-8")
    )
    assert payload["problems"]
    assert "Devanagari" in payload["problems"][0]


# --------------------------------------------------------------------------- #
# Destination layout agrees with the loader
# --------------------------------------------------------------------------- #


def test_destinations_match_the_loader_paths(fetch_models: ModuleType) -> None:
    """The tool imports the loader's constants, so they cannot drift."""
    from app.providers import local

    assert fetch_models.LID_MODEL_FILENAME == local.LID_MODEL_FILENAME
    assert fetch_models.DET_MODEL_FILENAME == local.DET_MODEL_FILENAME
    assert fetch_models.REC_MODEL_TEMPLATE.format(script="deva") == (
        local.REC_MODEL_TEMPLATE.format(script="deva")
    )
