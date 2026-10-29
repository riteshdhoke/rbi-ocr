"""Stage 1 ingestion — sniffing, identity, resource caps, decoding.

Offline by construction: every PDF used here is built byte-by-byte in
:func:`build_pdf`, so the suite needs no fixtures on disk, no network and no
PDF writer library.  Building them by hand is also the only way to produce the
inputs that matter — a dangling ``/Encrypt``, a ``/JavaScript`` action, a
205-page document — without committing hostile files to the repository.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.contracts.errors import (
    DecompressionBombError,
    DocumentRejected,
    EmptyFileError,
    EncryptedPDFError,
    MalformedPDFError,
    MalwareDetected,
    MimeMismatchError,
    PageCapExceededError,
    PixelCapExceededError,
    WallClockExceededError,
)
from app.settings import Settings
from app.stages.s1_ingest import (
    RASTER_DPI,
    IngestFlag,
    NoOpScanner,
    PageSource,
    ResourceCaps,
    Scanner,
    ScanVerdict,
    SniffedType,
    assert_decompression_ratio,
    idempotency_key,
    ingest,
    ingest_path,
    normalise_subject,
    pdf_declares_encryption,
    scan_pdf_active_content,
    sniff_type,
    soft_duplicate_key,
)

# --------------------------------------------------------------------------- #
# Hand-built PDFs (imported by test_s1b_bypass.py)
# --------------------------------------------------------------------------- #


def escape_pdf_string(raw: bytes) -> bytes:
    """Escape a byte string for a PDF literal string."""
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def text_content(lines: list[bytes], *, size: int = 12, first_y: int = 780,
                 leading: int = 60) -> bytes:
    """A content stream drawing ``lines`` down the page in Helvetica."""
    parts = [b"BT\n/F1 " + str(size).encode() + b" Tf\n"]
    y = first_y
    for raw in lines:
        parts.append(
            f"1 0 0 1 60 {y} Tm ".encode() + b"(" + escape_pdf_string(raw) + b") Tj\n"
        )
        y -= leading
    parts.append(b"ET\n")
    return b"".join(parts)


def build_pdf(
    content: bytes,
    *,
    pages: int = 1,
    rotate: int = 0,
    extra_root: bytes = b"",
    trailer_extra: bytes = b"",
    extra_objects: tuple[bytes, ...] = (),
) -> bytes:
    """Assemble a minimal, valid, uncompressed PDF with a real xref table."""
    content_obj = len(extra_objects) + 4
    font_obj = content_obj + 1
    page_objs = [font_obj + 1 + i for i in range(pages)]
    kids = b" ".join(f"{n} 0 R".encode() for n in page_objs)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R" + extra_root + b" >>",
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(pages).encode() + b" >>",
        b"<< /Type /Outlines /Count 0 >>",
        *extra_objects,
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for _ in range(pages):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Rotate "
            + str(rotate).encode()
            + b" /Resources << /Font << /F1 "
            + str(font_obj).encode()
            + b" 0 R >> >> /Contents "
            + str(content_obj).encode()
            + b" 0 R >>"
        )

    out = bytearray(b"%PDF-1.7\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R"
        + trailer_extra
        + b" >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


ENCRYPT_DICT = (
    b"<< /Filter /Standard /V 1 /R 2 /O <"
    + b"AB" * 16
    + b"> /U <"
    + b"CD" * 16
    + b"> /P -1 >>"
)


def build_encrypted_pdf() -> bytes:
    """A PDF whose trailer names a standard-security-handler ``/Encrypt``.

    PDFium reports "Incorrect password"; the byte scan sees ``/Encrypt`` first.
    Both paths must land on ``ENCRYPTED_PDF``, which is what makes the check
    robust to a document that hides ``/Encrypt`` in a compressed xref stream.
    """
    return build_pdf(
        text_content([b"secret"]),
        extra_objects=(ENCRYPT_DICT,),
        trailer_extra=b" /Encrypt 4 0 R /ID [<" + b"01" * 16 + b"> <" + b"01" * 16
        + b"> ]",
    )


def png_bytes(
    width: int, height: int, colour: tuple[int, int, int] = (255, 255, 255),
    mode: str = "RGB",
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), colour[: len(mode)]).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 249, 245)).save(buffer, format="JPEG")
    return buffer.getvalue()


class InfectedScanner:
    """A :class:`Scanner` that condemns everything, for the seam test."""

    def scan(self, data: bytes, *, filename: str | None = None) -> ScanVerdict:
        return ScanVerdict(
            clean=False, scanner="fake-clamav/1.0", signature="Eicar-Test-Signature"
        )


class CountingScanner:
    """A clean :class:`Scanner` that records how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    def scan(self, data: bytes, *, filename: str | None = None) -> ScanVerdict:
        self.calls += 1
        return ScanVerdict(clean=True, scanner="counting")


class SlowClock:
    """Monotonic clock that jumps ``step`` seconds on every read."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


@pytest.fixture
def caps() -> ResourceCaps:
    return ResourceCaps.from_settings(Settings(_env_file=None))


@pytest.fixture
def simple_pdf() -> bytes:
    return build_pdf(
        text_content(
            [
                b"COMPLAINT TO THE BANKING OMBUDSMAN",
                b"Account Number: 79192906414",
                b"IFSC Code: SBIN0695301",
            ]
        )
    )


# --------------------------------------------------------------------------- #
# MIME sniffing
# --------------------------------------------------------------------------- #


def test_sniff_identifies_the_three_supported_types(simple_pdf: bytes) -> None:
    assert sniff_type(simple_pdf) is SniffedType.PDF
    assert sniff_type(png_bytes(8, 8)) is SniffedType.PNG
    assert sniff_type(jpeg_bytes(8, 8)) is SniffedType.JPEG


def test_sniff_rejects_zero_bytes() -> None:
    with pytest.raises(EmptyFileError) as excinfo:
        sniff_type(b"")
    assert excinfo.value.code == "EMPTY_FILE"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"GIF89a" + b"\x00" * 32, id="gif"),
        pytest.param(b"<html><script>alert(1)</script></html>", id="html"),
        pytest.param(b"PK\x03\x04" + b"\x00" * 32, id="zip"),
        pytest.param(b"\x00" * 64, id="nulls"),
    ],
)
def test_sniff_rejects_unsupported_content(payload: bytes) -> None:
    with pytest.raises(MimeMismatchError) as excinfo:
        sniff_type(payload)
    assert excinfo.value.code == "MIME_MISMATCH"


def test_mime_mismatch_never_echoes_the_payload() -> None:
    """The disguised payload is attacker-controlled; it must not reach a log."""
    hostile = b"<html><script>alert('xss')</script></html>"
    with pytest.raises(MimeMismatchError) as excinfo:
        sniff_type(hostile)
    serialised = repr(excinfo.value.to_dict())
    assert "script" not in serialised
    assert "alert" not in serialised
    assert excinfo.value.context["magic_prefix_hex"] == hostile[:8].hex()


def test_magic_bytes_beat_the_extension() -> None:
    """BUILD-BRIEF M3: by magic bytes, not extension. The clash is recorded."""
    result = ingest(png_bytes(400, 400), filename="statement.pdf")
    assert result.report.sniffed_type is SniffedType.PNG
    assert IngestFlag.EXTENSION_MISMATCH in result.report.flags
    assert result.report.source_extension == ".pdf"


def test_a_truthful_extension_raises_no_flag(simple_pdf: bytes) -> None:
    result = ingest(simple_pdf, filename="complaint.PDF")
    assert IngestFlag.EXTENSION_MISMATCH not in result.report.flags


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_idempotency_key_is_sha256_over_the_bytes() -> None:
    key = idempotency_key(b"hello")
    assert key.startswith("sha256:")
    assert len(key) == len("sha256:") + 64
    assert key == idempotency_key(b"hello")
    assert key != idempotency_key(b"hellp")


def test_ingest_reports_the_idempotency_key_as_document_id(simple_pdf: bytes) -> None:
    assert ingest(simple_pdf).report.document_id == idempotency_key(simple_pdf)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Re: Complaint", "complaint"),
        ("RE: FWD: Re: Complaint", "complaint"),
        ("  Fwd:   Unauthorised   debit  ", "unauthorised debit"),
        ("Complaint CMS/2026/401687", "complaint cms/2026/401687"),
        ("AW: Antwort: Beschwerde", "beschwerde"),
    ],
)
def test_normalise_subject(raw: str, expected: str) -> None:
    assert normalise_subject(raw) == expected


def test_normalise_subject_keeps_the_reference_punctuation() -> None:
    """A complaint reference is the most discriminating token in a subject."""
    assert "cms/2026/401687" in normalise_subject("Re: about CMS/2026/401687")


def test_soft_duplicate_key_collapses_the_same_complaint() -> None:
    morning = soft_duplicate_key(
        "Ravi@Example.COM", "Re: Unauthorised debit", datetime(2026, 8, 7, 9, 15)
    )
    evening = soft_duplicate_key(
        "ravi@example.com", "unauthorised debit", datetime(2026, 8, 7, 22, 40)
    )
    assert morning == evening


def test_soft_duplicate_key_separates_days_and_senders() -> None:
    base = soft_duplicate_key("a@b.com", "Debit", date(2026, 8, 7))
    assert base != soft_duplicate_key("a@b.com", "Debit", date(2026, 8, 8))
    assert base != soft_duplicate_key("c@d.com", "Debit", date(2026, 8, 7))
    assert base != soft_duplicate_key("a@b.com", "Credit", date(2026, 8, 7))
    assert base == soft_duplicate_key(
        "a@b.com", "Debit", datetime(2026, 8, 7, 3, 0)
    )


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


def test_zero_byte_attachment(caps: ResourceCaps) -> None:
    with pytest.raises(EmptyFileError):
        ingest(b"", caps=caps)


def test_truncated_pdf_is_malformed() -> None:
    with pytest.raises(MalformedPDFError) as excinfo:
        ingest(b"%PDF-1.7\n1 0 obj\n<< /Type /Cat")
    assert excinfo.value.code == "MALFORMED_PDF"


def test_encrypted_pdf_is_refused_not_cracked() -> None:
    data = build_encrypted_pdf()
    assert pdf_declares_encryption(data) is True
    with pytest.raises(EncryptedPDFError) as excinfo:
        ingest(data)
    assert excinfo.value.context["detected_by"] == "encrypt_dictionary"


def test_encrypted_detection_survives_a_hidden_encrypt_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the byte scan blinded, PDFium's password error must still land."""
    monkeypatch.setattr(
        "app.stages.s1_ingest.pdf_declares_encryption", lambda data: False
    )
    with pytest.raises(EncryptedPDFError) as excinfo:
        ingest(build_encrypted_pdf())
    assert excinfo.value.context["detected_by"] == "pdfium"


def test_pdf_with_no_pages_is_malformed() -> None:
    empty = build_pdf(b"", pages=0)
    with pytest.raises(MalformedPDFError):
        ingest(empty)


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(build_pdf(b"", pages=0), id="zero_pages"),
        pytest.param(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n", id="truncated"),
        pytest.param(b"%PDF-1.4\n" + b"\x00" * 64, id="header_only_garbage"),
    ],
)
def test_malformed_pdf_stays_malformed_after_an_encrypted_one(
    malformed: bytes,
) -> None:
    """PDFium's last-error is process-global and sticky; we must not read a ghost.

    A zero-page document fails to load *without* PDFium recording why, so the
    code read back is whatever the previous failure left in the slot.  Before
    the primer in ``_load_pdfium_document`` that made every malformed PDF report
    ``ENCRYPTED_PDF`` for the rest of the worker's life once one encrypted
    document had been seen — and ``ENCRYPTED_PDF`` is a different auto-reply.
    """
    with pytest.raises(EncryptedPDFError):
        ingest(build_encrypted_pdf())

    with pytest.raises(MalformedPDFError) as excinfo:
        ingest(malformed)
    assert excinfo.value.code == "MALFORMED_PDF"

    # ...and the encrypted verdict is still reachable afterwards.
    with pytest.raises(EncryptedPDFError):
        ingest(build_encrypted_pdf())


def test_corrupt_image_body_is_a_mime_mismatch() -> None:
    """PNG magic over a body that is not a PNG: the declaration is a lie."""
    header = b"\x89PNG\r\n\x1a\n"
    with pytest.raises(MimeMismatchError) as excinfo:
        ingest(header + b"\xde\xad\xbe\xef" * 16)
    assert excinfo.value.context["sniffed"] == "image/png"


# --------------------------------------------------------------------------- #
# Resource caps
# --------------------------------------------------------------------------- #


def test_page_cap(caps: ResourceCaps) -> None:
    data = build_pdf(text_content([b"page"]), pages=6)
    tight = caps.model_copy(update={"max_pages": 3})
    with pytest.raises(PageCapExceededError) as excinfo:
        ingest(data, caps=tight)
    assert excinfo.value.context["page_count"] == 6
    assert excinfo.value.context["limit"] == 3


def test_pixel_cap_is_enforced_before_a_bitmap_is_allocated(
    caps: ResourceCaps,
) -> None:
    data = build_pdf(text_content([b"page"]), pages=3)
    tight = caps.model_copy(update={"max_total_pixels": 1_000_000})
    with pytest.raises(PixelCapExceededError) as excinfo:
        ingest(data, caps=tight)
    assert "planned_pixels" in excinfo.value.context


def test_pixel_cap_for_images(caps: ResourceCaps) -> None:
    tight = caps.model_copy(update={"max_total_pixels": 10_000})
    with pytest.raises(PixelCapExceededError):
        ingest(png_bytes(400, 400), caps=tight)


def test_pillow_pixel_limit_is_restored_after_ingest(caps: ResourceCaps) -> None:
    """Importing this stage must not change Pillow's behaviour process-wide."""
    before = Image.MAX_IMAGE_PIXELS
    with pytest.raises(PixelCapExceededError):
        ingest(png_bytes(400, 400), caps=caps.model_copy(
            update={"max_total_pixels": 10_000}
        ))
    assert Image.MAX_IMAGE_PIXELS == before


def test_decompression_ratio_guard(caps: ResourceCaps) -> None:
    with pytest.raises(DecompressionBombError) as excinfo:
        assert_decompression_ratio(1_000, 200 * 1024 * 1024, caps)
    assert excinfo.value.context["limit"] == caps.max_decompression_ratio


def test_decompression_guard_ignores_small_but_highly_compressible_payloads(
    caps: ResourceCaps,
) -> None:
    """A blank A4 scan compresses ~700:1 and is not an attack."""
    ratio = assert_decompression_ratio(16_000, 11_600_000, caps)
    assert ratio > caps.max_decompression_ratio


def test_a_flat_png_bomb_is_refused(caps: ResourceCaps) -> None:
    data = png_bytes(5000, 5000)
    assert len(data) < 200_000
    with pytest.raises(DecompressionBombError):
        ingest(data, caps=caps)


def test_wall_clock_cap(caps: ResourceCaps, simple_pdf: bytes) -> None:
    tight = caps.model_copy(update={"max_wall_clock_seconds": 0.5})
    with pytest.raises(WallClockExceededError) as excinfo:
        ingest(simple_pdf, caps=tight, clock=SlowClock(1.0))
    assert excinfo.value.context["limit_seconds"] == 0.5


# --------------------------------------------------------------------------- #
# Malware-scan seam
# --------------------------------------------------------------------------- #


def test_noop_scanner_declares_itself_in_the_report(simple_pdf: bytes) -> None:
    """An operator reading scanner=noop knows the control is not in place."""
    assert ingest(simple_pdf).report.scanner == "noop"
    assert isinstance(NoOpScanner(), Scanner)


@pytest.mark.parametrize(
    ("payload", "filename", "would_otherwise_raise"),
    [
        pytest.param(
            b"<html><script>alert(1)</script></html>",
            "invoice.pdf",
            MimeMismatchError,
            id="html_disguised_as_pdf",
        ),
        pytest.param(
            b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n",
            "statement.pdf",
            MalformedPDFError,
            id="truncated_pdf",
        ),
        pytest.param(
            b"\x89PNG\r\n\x1a\n" + b"\xde\xad\xbe\xef" * 16,
            "evidence.png",
            MimeMismatchError,
            id="corrupt_png",
        ),
    ],
)
def test_infected_attachment_raises_before_anything_parses_it(
    payload: bytes,
    filename: str,
    would_otherwise_raise: type[Exception],
) -> None:
    """The scan is the first thing that touches the bytes after the length check.

    Each payload is one a *later* stage would reject on its own, so the
    assertion that ``MalwareDetected`` — not the parser's error — comes out is
    what proves the scanner ran before any sniffing, decoding or PDF parse.
    """
    with pytest.raises(would_otherwise_raise):
        ingest(payload, filename=filename)

    with pytest.raises(MalwareDetected) as excinfo:
        ingest(payload, filename=filename, scanner=InfectedScanner())
    assert excinfo.value.code == "MALWARE_DETECTED"


def test_zero_length_is_refused_before_the_scanner_is_asked() -> None:
    """Nothing can be infected by nothing, and the length check is cheaper.

    Ordering is deliberate and documented in the module docstring: an empty
    attachment is ``EMPTY_FILE``, which is a complainant-facing bounce, not a
    security event that must never generate a reply.
    """
    scanner = CountingScanner()
    with pytest.raises(EmptyFileError):
        ingest(b"", scanner=scanner)
    assert scanner.calls == 0


def test_infected_pdf_is_quarantined(simple_pdf: bytes) -> None:
    with pytest.raises(MalwareDetected) as excinfo:
        ingest(simple_pdf, scanner=InfectedScanner())
    assert excinfo.value.code == "MALWARE_DETECTED"
    assert excinfo.value.signature == "Eicar-Test-Signature"
    assert excinfo.value.scanner == "fake-clamav/1.0"


def test_malware_is_not_a_bounceable_rejection() -> None:
    """It must not become a complainant-facing template (see errors.py)."""
    assert not issubclass(MalwareDetected, DocumentRejected)


# --------------------------------------------------------------------------- #
# Active content
# --------------------------------------------------------------------------- #


def test_javascript_is_detected_flagged_and_never_executed() -> None:
    data = build_pdf(
        text_content([b"Please open me"]),
        extra_root=b" /OpenAction << /S /JavaScript /JS (app.alert\\(1\\);) >>",
    )
    active = scan_pdf_active_content(data)
    assert "/JavaScript" in active.markers
    assert "/OpenAction" in active.markers
    assert active.present

    result = ingest(data, filename="hostile.pdf")
    assert IngestFlag.PDF_ACTIVE_CONTENT in result.report.flags
    assert result.report.active_content.markers == active.markers
    # The page still rasterises — we strip by never evaluating, not by refusing.
    assert len(result.pages) == 1


def test_a_clean_pdf_declares_no_active_content(simple_pdf: bytes) -> None:
    assert scan_pdf_active_content(simple_pdf).present is False


def test_aa_marker_needs_a_delimiter() -> None:
    """``/AAPL`` in a font name is not an additional-actions dictionary."""
    assert "/AA" not in scan_pdf_active_content(b"%PDF-1.7 /AAPLColor").markers
    assert "/AA" in scan_pdf_active_content(b"%PDF-1.7 /AA << /O 1 0 R >>").markers


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def test_pdf_rasterises_to_bgr_pages_at_the_documented_dpi(
    simple_pdf: bytes,
) -> None:
    result = ingest(simple_pdf)
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.source is PageSource.PDF_RASTER
    assert page.dpi == RASTER_DPI
    assert page.image.dtype == np.uint8
    assert page.image.ndim == 3 and page.image.shape[2] == 3
    # A4 at 200 DPI.
    assert 1600 < page.width_px < 1700
    assert 2300 < page.height_px < 2400
    assert result.report.total_pixels == page.pixels


def test_rasterisation_can_be_skipped_for_the_bypass_decision() -> None:
    data = build_pdf(text_content([b"digital"]), pages=4)
    result = ingest(data, rasterize=False)
    assert result.pages == ()
    assert result.report.page_count == 4
    assert IngestFlag.RASTERISATION_SKIPPED in result.report.flags


def test_image_decodes_to_a_single_bgr_page() -> None:
    result = ingest(png_bytes(320, 240, (10, 20, 200)))
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.source is PageSource.IMAGE
    assert page.image.shape == (240, 320, 3)
    # BGR: the blue channel carries the 200.
    assert int(page.image[0, 0, 0]) == 200
    assert int(page.image[0, 0, 2]) == 10


def test_alpha_is_flattened_onto_white() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(buffer, format="PNG")
    result = ingest(buffer.getvalue())
    assert IngestFlag.ALPHA_FLATTENED in result.report.flags
    assert int(result.pages[0].image[0, 0, 0]) == 255


def test_ingest_path_reads_from_disk(tmp_path: Path, simple_pdf: bytes) -> None:
    target = tmp_path / "complaint.pdf"
    target.write_bytes(simple_pdf)
    result = ingest_path(target)
    assert result.report.source_extension == ".pdf"
    assert result.is_pdf is True
    assert result.data == simple_pdf


def test_result_keeps_the_bytes_for_stage_1b(simple_pdf: bytes) -> None:
    assert ingest(simple_pdf).data == simple_pdf


def test_report_is_json_safe(simple_pdf: bytes) -> None:
    dumped = ingest(simple_pdf).report.model_dump(mode="json")
    assert dumped["sniffed_type"] == "application/pdf"
    assert dumped["scanner"] == "noop"
    assert isinstance(dumped["duration_ms"], float)


def test_resource_caps_come_from_settings() -> None:
    settings = Settings(_env_file=None, MAX_PAGES_PER_DOC=7)
    assert ResourceCaps.from_settings(settings).max_pages == 7


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: b"", id="empty"),
        pytest.param(lambda: b"GIF89a" + b"\x00" * 8, id="gif"),
        pytest.param(lambda: b"%PDF-1.7\ntruncated", id="truncated"),
        pytest.param(build_encrypted_pdf, id="encrypted"),
    ],
)
def test_every_rejection_is_typed_and_carries_a_code(
    factory: Callable[[], bytes],
) -> None:
    """No bare exception may escape Stage 1 for a malformed input."""
    with pytest.raises(DocumentRejected) as excinfo:
        ingest(factory())
    assert excinfo.value.code == excinfo.value.context["reason"]
