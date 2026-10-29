"""Golden — the committed adversarial inputs (BUILD-BRIEF M3).

    **Test:** all 7 committed adversarial inputs produce typed rejections;
    no worker crashes, no OOM.

``corpus/adversarial/MANIFEST.json`` is the **source of truth**, not a table
copied into this file.  Every expectation is parsed out of the manifest at run
time, and the directory listing is reconciled against it in both directions, so
adding a hostile file without declaring its expected behaviour fails here — and
so does declaring one that nobody committed.

Two properties are asserted for every input:

1. the *specific* typed outcome the manifest names — the exact
   :class:`~app.contracts.errors.DocumentRejected` subclass, or the exact
   :class:`~app.contracts.envelope.BounceReason`, because a machine-readable
   reason is what the auto-reply template keys on (PRD P1-2) and "some error"
   is not a reason;
2. that **nothing else escapes**.  A ``PdfiumError``, a ``struct.error``, a
   ``MemoryError`` or a bare ``Exception`` reaching the worker is the failure
   this file exists to catch — the rejection is the easy half.

The manifest's ``not_included_generate_locally`` entries are covered too, built
in memory from the same hand-assembled PDFs the Stage 1 unit tests use, so the
caps they exercise are not silently untested just because the artefact is too
large or too hostile to commit.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.contracts.envelope import BounceReason, QualityVerdict
from app.contracts.errors import (
    DecompressionBombError,
    DocumentRejected,
    EncryptedPDFError,
    MalwareDetected,
    PageCapExceededError,
    PipelineError,
    RejectionReason,
    exception_for,
)
from app.stages.s1_ingest import IngestResult, ingest, ingest_path
from app.stages.s15_quality import PageQuality, QualityGate
from tests.unit.test_s1_ingest import build_encrypted_pdf, build_pdf, text_content

# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #

ADVERSARIAL_DIR = Path(__file__).resolve().parents[2] / "corpus" / "adversarial"
MANIFEST_PATH = ADVERSARIAL_DIR / "MANIFEST.json"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


MANIFEST: dict[str, Any] = load_manifest()
EXPECTED_BEHAVIOUR: dict[str, str] = MANIFEST["expected_behaviour"]

#: Every verb the manifest uses, mapped to what it means to this pipeline.
#: ``drop page`` and ``bounce`` are the same mechanism — Stage 1.5 returns a
#: verdict with a reason and no pixels — but the manifest distinguishes them
#: because a blank page inside a 12-page fax is dropped while a blank
#: single-page attachment bounces the document.  That distinction is WS-F's;
#: at this level both are a bounced page.
_BOUNCE_VERBS = frozenset({"bounce", "drop page"})


def parse_expectation(sentence: str) -> tuple[str, str]:
    """``"reject: MIME_MISMATCH, script never evaluated"`` -> ``("reject", ...)``.

    Deliberately strict: an unrecognised verb raises rather than being skipped,
    because a silently-ignored expectation is an untested hostile input.
    """
    verb, _, remainder = sentence.partition(":")
    verb = verb.strip().lower()
    token = remainder.strip().split(",")[0].strip()
    if verb == "reject":
        return "reject", token
    if verb in _BOUNCE_VERBS:
        return "bounce", token
    if verb == "repair":
        return "repair", token
    raise AssertionError(
        f"MANIFEST.json uses a verb this test does not implement: {sentence!r}"
    )


# --------------------------------------------------------------------------- #
# Manifest hygiene
# --------------------------------------------------------------------------- #


def test_the_manifest_declares_the_directory_synthetic() -> None:
    """A-5: the residency gate accepts a directory manifest that says so."""
    assert MANIFEST["synthetic"] is True


def test_every_committed_file_is_declared_and_every_declaration_exists() -> None:
    on_disk = {
        path.name for path in ADVERSARIAL_DIR.iterdir() if path.name != "MANIFEST.json"
    }
    declared = set(EXPECTED_BEHAVIOUR)
    assert on_disk == declared, (
        f"undeclared files: {sorted(on_disk - declared)}; "
        f"missing files: {sorted(declared - on_disk)}"
    )


def test_the_brief_asks_for_seven_and_seven_is_what_is_committed() -> None:
    """BUILD-BRIEF M3 names the count; drift in either direction is a bug."""
    assert len(EXPECTED_BEHAVIOUR) == 7


def test_every_expectation_parses() -> None:
    for sentence in EXPECTED_BEHAVIOUR.values():
        verb, token = parse_expectation(sentence)
        assert verb in {"reject", "bounce", "repair"}
        assert token


# --------------------------------------------------------------------------- #
# The inputs
# --------------------------------------------------------------------------- #


def run_ingest(path: Path) -> IngestResult:
    """Ingest, converting anything untyped into an explicit test failure."""
    try:
        return ingest_path(path)
    except PipelineError:
        raise
    except BaseException as exc:  # noqa: BLE001 - the whole point of the test
        pytest.fail(
            f"{path.name} escaped Stage 1 as {type(exc).__name__}: {exc}. "
            "Every rejection must be typed; an untyped one crashes the worker."
        )


def assess(result: IngestResult) -> list[PageQuality]:
    return QualityGate().evaluate_document([page.image for page in result.pages])


@pytest.mark.parametrize("filename", sorted(EXPECTED_BEHAVIOUR))
def test_adversarial_input_produces_its_declared_outcome(filename: str) -> None:
    path = ADVERSARIAL_DIR / filename
    assert path.is_file(), f"{filename} is declared but not committed"
    verb, token = parse_expectation(EXPECTED_BEHAVIOUR[filename])

    if verb == "reject":
        expected = exception_for(RejectionReason(token))
        with pytest.raises(expected) as excinfo:
            run_ingest(path)
        assert excinfo.value.code == token
        # A rejection carries machine-readable context, never the payload.
        assert isinstance(excinfo.value.context, dict)
        return

    result = run_ingest(path)
    assert result.pages, f"{filename} was accepted by Stage 1 but decoded no pages"
    assessments = [quality.assessment for quality in assess(result)]

    if verb == "bounce":
        reason = BounceReason(token)
        assert any(a.bounce_reason is reason for a in assessments), (
            f"{filename} expected {token}, got "
            f"{[a.bounce_reason for a in assessments]}"
        )
        for quality in assess(result):
            if quality.assessment.bounced:
                assert quality.image is None
    else:  # repair
        assert all(a.verdict is QualityVerdict.REPAIR for a in assessments)
        assert all(a.repairs_applied for a in assessments)
        assert all(quality.image is not None for quality in assess(result))


def test_no_adversarial_input_crashes_the_worker() -> None:
    """The property, stated once over the whole directory.

    ``ingest`` promises that nothing but a typed rejection escapes it for a
    malformed input; this is the assertion that promise is measured against.
    """
    for filename in sorted(EXPECTED_BEHAVIOUR):
        path = ADVERSARIAL_DIR / filename
        try:
            result = ingest_path(path)
        except (DocumentRejected, MalwareDetected):
            continue
        except BaseException as exc:  # noqa: BLE001
            pytest.fail(f"{filename} raised an untyped {type(exc).__name__}: {exc}")
        else:
            # Accepted inputs must also survive scoring without an exception.
            assess(result)


def test_the_disguised_html_is_refused_on_its_magic_bytes() -> None:
    """No parser, no renderer and no script engine ever sees the payload."""
    path = ADVERSARIAL_DIR / "html_disguised_as_pdf.pdf"
    payload = path.read_bytes()
    assert b"<" in payload, "the fixture is supposed to be markup"
    with pytest.raises(PipelineError) as excinfo:
        ingest(payload, filename="html_disguised_as_pdf.pdf")
    assert excinfo.value.code == "MIME_MISMATCH"
    # The refusal reports a hex prefix, never the attacker-controlled bytes.
    rendered = str(excinfo.value) + json.dumps(
        excinfo.value.context, default=str
    )
    assert "<script" not in rendered.lower()


def test_the_blank_page_is_not_billed_for_ocr() -> None:
    result = run_ingest(ADVERSARIAL_DIR / "blank_page.png")
    quality = assess(result)[0]
    assert quality.assessment.bounce_reason is BounceReason.BLANK_PAGE
    assert quality.image is None


def test_the_inverted_page_is_repaired_exactly() -> None:
    """Auto-invert is ``255 - x``: the evidence crop is still the document."""
    result = run_ingest(ADVERSARIAL_DIR / "inverted.png")
    quality = assess(result)[0]
    assert "auto_invert" in quality.assessment.repairs_applied
    assert quality.image is not None
    assert quality.assessment.metrics.is_inverted is False
    original = result.pages[0].image
    assert quality.assessment.repairs_applied == ("auto_invert",)
    assert np.array_equal(quality.image, 255 - original)


# --------------------------------------------------------------------------- #
# The cases the manifest says to generate locally
# --------------------------------------------------------------------------- #


def test_the_locally_generated_cases_are_the_ones_this_file_covers() -> None:
    """Keeps the manifest and this section from drifting apart."""
    assert set(MANIFEST["not_included_generate_locally"]) == {
        "zip_bomb.zip",
        "pdf_5000_pages.pdf",
        "password_protected.pdf",
    }


def test_password_protected_pdf_is_refused_never_cracked() -> None:
    with pytest.raises(EncryptedPDFError) as excinfo:
        ingest(build_encrypted_pdf(), filename="password_protected.pdf")
    assert excinfo.value.code == "ENCRYPTED_PDF"


def test_the_page_cap_triggers_before_a_bitmap_is_allocated() -> None:
    """``pdf_5000_pages.pdf`` in the manifest; 201 is enough to prove the cap."""
    oversized = build_pdf(text_content([b"page"]), pages=201)
    with pytest.raises(PageCapExceededError) as excinfo:
        ingest(oversized, filename="pdf_5000_pages.pdf")
    assert excinfo.value.context["page_count"] == 201
    assert excinfo.value.context["limit"] == 200


def test_an_archive_attachment_is_not_a_supported_type() -> None:
    """``zip_bomb.zip`` never reaches an expander: the type is refused first."""
    zip_magic = b"PK\x03\x04" + b"\x00" * 60
    with pytest.raises(PipelineError) as excinfo:
        ingest(zip_magic, filename="zip_bomb.zip")
    assert excinfo.value.code == "MIME_MISMATCH"


def test_the_decompression_guard_the_expander_would_reuse() -> None:
    """The ratio cap itself, on the one format Stage 1 does expand.

    When WS-F adds archive expansion it must call
    ``assert_decompression_ratio`` per member; this is that guard firing on a
    payload whose decoded size dwarfs its compressed size.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (5000, 5000), (255, 255, 255)).save(buffer, format="PNG")
    bomb = buffer.getvalue()
    assert len(bomb) < 200_000
    with pytest.raises(DecompressionBombError) as excinfo:
        ingest(bomb, filename="zip_bomb_equivalent.png")
    assert excinfo.value.context["ratio"] > excinfo.value.context["limit"]
