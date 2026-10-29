"""Stage 1 — ingestion: sniff, scan, cap, decode.

BUILD-BRIEF M3.  Everything that touches attacker-supplied bytes happens here,
and the rule of the module is that **no untrusted byte is interpreted before its
type has been established from its own content**.

What it does, in order:

1. **Zero length** → :class:`~app.contracts.errors.EmptyFileError`.
2. **Malware scan** (:class:`Scanner` seam, :class:`NoOpScanner` by default) →
   :class:`~app.contracts.errors.MalwareDetected`.
3. **MIME by magic bytes, never by extension** (PDF / JPEG / PNG).  Anything
   else → :class:`~app.contracts.errors.MimeMismatchError`.  This is what makes
   ``html_disguised_as_pdf.pdf`` safe: the payload is refused on its first eight
   bytes, so no parser, no renderer and no script engine ever sees it.
4. **PDF**: encryption declared in the file or reported by PDFium →
   ``ENCRYPTED_PDF``; anything unparseable → ``MALFORMED_PDF``; active content
   (``/JavaScript``, ``/JS``, ``/OpenAction``, ``/AA``, ``/Launch``,
   ``/EmbeddedFile``, ``/RichMedia``) is **detected and recorded, never
   executed** — see :func:`scan_pdf_active_content` for why that is a strip and
   not merely a flag.
5. **Resource caps**, each with its own typed error: page count, total pixels,
   decompression ratio, wall clock.
6. **Rasterise** accepted PDF pages with PDFium at :data:`RASTER_DPI`, decode
   images with Pillow behind an explicit decompression-bomb guard.

Identity:

* ``document_id`` is ``sha256:<hex>`` over the **attachment bytes** — the
  idempotency key.  Two mails carrying the same PDF are one document.
* :func:`soft_duplicate_key` hashes ``(sender, normalised subject, calendar
  day)``.  It is a *pure function*: this module has no store to look it up in.
  Persistence and the actual dedupe decision are WS-F's.

This module does NOT handle:
  * quality (resolution, blur, skew, blankness) — that is Stage 1.5.  Stage 1
    decodes ``tiny_lowres.png`` and ``blank_page.png`` successfully and hands
    them on; refusing them is a *quality verdict*, not an ingestion rejection;
  * the digital-PDF text layer — Stage 1b.  Stage 1 keeps the original bytes on
    :class:`IngestResult` so Stage 1b can reopen the document without a second
    read from storage;
  * archives.  A ``.zip`` attachment is not a supported type and is refused at
    step 3.  When WS-F adds archive expansion it must call
    :func:`assert_decompression_ratio` per member — that guard is public for
    exactly this reason;
  * **applying** the decompression ratio to PDF rasterisation.  A legitimate
    40 KB text PDF rasterises to ~12 MB per A4 page, a ratio of ~300; capping
    that at 100 would refuse ordinary mail.  The bound that actually holds for
    PDFs is page count x pixels, and both are enforced *before* a page is
    rendered.  See ``docs/decisions-wse1.md``;
  * real virus scanning.  :class:`NoOpScanner` returns clean for everything;
    production wires ClamAV through the same protocol (no ClamAV on the dev
    laptop, so the seam is what ships and the test asserts the seam, not an
    engine);
  * OCR, classification or any model call.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import threading
import time
import unicodedata
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field

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
from app.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

    #: ``(H, W, 3)`` BGR ``uint8`` — the channel order
    #: :func:`app.providers.base.to_page_image` normalises to.
    PageImage = NDArray[np.uint8]
else:
    PageImage = Any

__all__ = [
    "DECOMPRESSION_MIN_DECODED_BYTES",
    "MAGIC_SIGNATURES",
    "PDF_ACTIVE_CONTENT_MARKERS",
    "PDF_POINTS_PER_INCH",
    "RASTER_DPI",
    "IngestFlag",
    "IngestReport",
    "IngestResult",
    "PdfActiveContent",
    "RasterPage",
    "ResourceCaps",
    "ScanVerdict",
    "Scanner",
    "NoOpScanner",
    "PageSource",
    "SniffedType",
    "WallClock",
    "assert_decompression_ratio",
    "idempotency_key",
    "ingest",
    "ingest_path",
    "normalise_subject",
    "open_pdf",
    "page_images",
    "pdf_declares_encryption",
    "scan_pdf_active_content",
    "sniff_type",
    "soft_duplicate_key",
]

_LOGGER: Final = logging.getLogger(__name__)

#: Rasterisation resolution for accepted PDFs.  200 DPI is chosen because it is
#: (a) the resolution the synthetic corpus is rendered at, so measured quality
#: thresholds transfer, and (b) ~1.4x the 150 DPI floor below which PP-OCR's
#: recognition head loses Devanagari conjuncts, while costing 44% fewer pixels
#: than 300 DPI.  A4 at 200 DPI is 1654 x 2339 = 3.9 Mpx.
RASTER_DPI: Final[float] = 200.0

#: PDF user-space unit.
PDF_POINTS_PER_INCH: Final[float] = 72.0

#: A high decoded:compressed ratio is only evidence of an attack once the
#: *decoded* size is itself large.  A blank 1654x2339 PNG compresses ~700:1 and
#: is perfectly ordinary; a 5 KB file claiming 40 000 x 40 000 pixels is not.
#: 64 MiB is ~5 A4 pages at 200 DPI in RGB.
DECOMPRESSION_MIN_DECODED_BYTES: Final[int] = 64 * 1024 * 1024

#: Magic-byte prefixes of the three supported types.  Checked at offset 0 for
#: images; the PDF header is searched in a short prefix window because the spec
#: tolerates leading bytes before ``%PDF-``.
MAGIC_SIGNATURES: Final[dict[str, bytes]] = {
    "application/pdf": b"%PDF-",
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
}

#: How far into the file ``%PDF-`` may appear (PDF 32000-1 sec. 7.5.2 allows a
#: small amount of leading junk; real-world readers use 1 KiB).
_PDF_HEADER_WINDOW: Final[int] = 1024

#: Tokens whose presence means the document carries active content.  Matched on
#: raw bytes, so a compressed object stream can hide one — which is why
#: detection is belt-and-braces and *never* the thing that keeps us safe.  The
#: thing that keeps us safe is that we never build a form environment and never
#: draw annotations (see :func:`_render_page`).
PDF_ACTIVE_CONTENT_MARKERS: Final[tuple[bytes, ...]] = (
    b"/JavaScript",
    b"/JS",
    b"/OpenAction",
    b"/AA",
    b"/Launch",
    b"/EmbeddedFile",
    b"/RichMedia",
    b"/SubmitForm",
)

#: ``/Encrypt`` followed by a dictionary or an indirect reference.  Restricting
#: the match this way keeps the literal string inside an uncompressed content
#: stream from tripping the check.
_ENCRYPT_PATTERN: Final[re.Pattern[bytes]] = re.compile(
    rb"/Encrypt\s*(?:<<|\d+\s+\d+\s+R)"
)

#: Mail subject prefixes stripped before the soft-duplicate key is built.
#: Deliberately small and explicit — a general "strip anything in brackets"
#: rule collapses distinct complaints that quote a reference in the subject.
_SUBJECT_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:re|fw|fwd|aw|antwort|rv|उत्तर)\s*(?:\[\d+\])?\s*:\s*",
    re.IGNORECASE,
)

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

_INGESTED = Counter(
    "ingest_documents_total",
    "Attachments entering Stage 1, by sniffed type and outcome.",
    ["sniffed", "outcome"],
)
_INGEST_LATENCY = Histogram(
    "ingest_duration_seconds",
    "Wall-clock duration of one Stage 1 ingestion, by sniffed type.",
    ["sniffed"],
)
_PAGES_DECODED = Counter(
    "ingest_pages_decoded_total",
    "Pages successfully decoded or rasterised, by source.",
    ["source"],
)


# --------------------------------------------------------------------------- #
# Enumerations and small value types
# --------------------------------------------------------------------------- #


class SniffedType(StrEnum):
    """The three types Stage 1 accepts, named by their IANA media type."""

    PDF = "application/pdf"
    JPEG = "image/jpeg"
    PNG = "image/png"


class PageSource(StrEnum):
    """How a :class:`RasterPage` came into existence."""

    PDF_RASTER = "pdf_raster"
    IMAGE = "image"


class IngestFlag(StrEnum):
    """Non-fatal observations recorded on :class:`IngestReport`.

    None of these rejects a document.  They exist so that a pattern — a spike
    in extension mismatches, say — is visible in the audit log before it is a
    security incident.
    """

    #: The filename's extension disagreed with the magic bytes.  Magic wins
    #: (BUILD-BRIEF M3: "by magic bytes, not extension"); the disagreement is
    #: recorded because it is the cheapest signal that mail is being crafted.
    EXTENSION_MISMATCH = "EXTENSION_MISMATCH"
    #: The PDF declared JavaScript / an auto-action / an embedded file.
    PDF_ACTIVE_CONTENT = "PDF_ACTIVE_CONTENT"
    #: EXIF said the image was rotated; the pixels were transposed to match.
    EXIF_ORIENTATION_APPLIED = "EXIF_ORIENTATION_APPLIED"
    #: The image carried an alpha channel, flattened onto white.
    ALPHA_FLATTENED = "ALPHA_FLATTENED"
    #: Rasterisation was skipped by request (``rasterize=False``).
    RASTERISATION_SKIPPED = "RASTERISATION_SKIPPED"


class ScanVerdict(BaseModel):
    """Result of a malware scan.  ``clean`` is the only field callers branch on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clean: bool
    #: Engine identifier, e.g. ``"noop"`` or ``"clamav/0.103.11"``.
    scanner: str = Field(min_length=1)
    #: Signature name when ``clean`` is false.  Never complainant data.
    signature: str | None = None


@runtime_checkable
class Scanner(Protocol):
    """Malware-scan seam.

    One method, synchronous, bytes in.  Production wires ClamAV — either
    ``clamd`` over the local socket or ``clamscan`` in a subprocess — behind
    this protocol; the air-gapped node ships the signature database with the
    deployment bundle.  There is no ClamAV on the development laptop, so what
    ships from WS-E1 is the seam and :class:`NoOpScanner`, and the test asserts
    that an infected verdict raises, not that any particular engine detects
    anything.
    """

    def scan(self, data: bytes, *, filename: str | None = None) -> ScanVerdict:
        """Return a verdict for ``data``.  Must not raise on clean input."""
        ...


class NoOpScanner:
    """Default :class:`Scanner`: reports everything clean.

    Explicit about being a placeholder — it does **not** silently pretend to be
    a scanner in the audit trail, because :attr:`ScanVerdict.scanner` is
    ``"noop"`` and the ingest report carries it into the envelope.  An operator
    reading ``scanner=noop`` on a production document knows the control is not
    in place.
    """

    name: Final[str] = "noop"

    def scan(self, data: bytes, *, filename: str | None = None) -> ScanVerdict:
        """Always clean.  ``data`` is not read."""
        return ScanVerdict(clean=True, scanner=self.name)


class PdfActiveContent(BaseModel):
    """Which active-content markers a PDF declares."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Markers found, as their PDF names (``"/JS"``), sorted, de-duplicated.
    markers: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        """True when the document declares any active content."""
        return bool(self.markers)


class ResourceCaps(BaseModel):
    """The four Stage 1 ceilings, each with its own typed error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_pages: int = Field(gt=0)
    max_total_pixels: int = Field(gt=0)
    max_decompression_ratio: int = Field(gt=0)
    max_wall_clock_seconds: float = Field(gt=0.0)
    #: Decoded-size floor below which a high ratio is compression, not a bomb.
    decompression_min_decoded_bytes: int = Field(
        default=DECOMPRESSION_MIN_DECODED_BYTES, gt=0
    )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ResourceCaps:
        """Read the caps from :class:`~app.settings.Settings`."""
        s = settings if settings is not None else get_settings()
        return cls(
            max_pages=s.MAX_PAGES_PER_DOC,
            max_total_pixels=s.MAX_TOTAL_PIXELS,
            max_decompression_ratio=s.MAX_DECOMPRESSION_RATIO,
            max_wall_clock_seconds=s.MAX_WALL_CLOCK_SECONDS,
        )


class IngestReport(BaseModel):
    """JSON-safe record of what Stage 1 did.  Carries no pixels and no PII."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ``sha256:<hex>`` over the attachment bytes — the idempotency key.
    document_id: str = Field(min_length=8)
    sniffed_type: SniffedType
    byte_size: int = Field(ge=0)
    page_count: int = Field(ge=0)
    total_pixels: int = Field(ge=0)
    raster_dpi: float | None = Field(default=None, gt=0.0)
    #: ``decoded bytes / compressed bytes``; ``None`` for PDFs, where the ratio
    #: is not the operative bound (see the module docstring).
    decompression_ratio: float | None = Field(default=None, ge=0.0)
    scanner: str = Field(min_length=1)
    active_content: PdfActiveContent = Field(default_factory=PdfActiveContent)
    flags: tuple[IngestFlag, ...] = ()
    duration_ms: float = Field(ge=0.0)
    #: Only the extension, never the full path — filenames carry names.
    source_extension: str | None = None


@dataclass(frozen=True, slots=True)
class RasterPage:
    """One decoded page.

    A dataclass rather than a Pydantic model on purpose: it carries a NumPy
    array, and a Pydantic model holding pixels invites someone to serialise a
    3.9-megapixel page into a log line.
    """

    page_no: int
    #: ``(H, W, 3)`` BGR ``uint8``.
    image: PageImage
    source: PageSource
    dpi: float | None = None

    @property
    def width_px(self) -> int:
        return int(self.image.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.image.shape[0])

    @property
    def pixels(self) -> int:
        return self.width_px * self.height_px


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Everything downstream needs, and the bytes Stage 1b reopens."""

    report: IngestReport
    pages: tuple[RasterPage, ...]
    #: Original attachment bytes.  Present for every type; Stage 1b uses them
    #: for PDFs so it does not depend on Stage 1 keeping a document handle
    #: alive across a stage boundary.
    data: bytes

    @property
    def document_id(self) -> str:
        return self.report.document_id

    @property
    def is_pdf(self) -> bool:
        return self.report.sniffed_type is SniffedType.PDF


#: Monotonic clock, injectable so the wall-clock cap is testable without sleeping.
WallClock = Callable[[], float]


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def idempotency_key(data: bytes) -> str:
    """``sha256:<hex>`` over the attachment bytes (BUILD-BRIEF M3).

    The *bytes*, not the parsed content: two mails whose PDFs differ only in a
    ``/ModDate`` are two documents, and pretending otherwise would silently drop
    a resubmission that fixed a page.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


def normalise_subject(subject: str) -> str:
    """Canonical form of a mail subject for soft-duplicate comparison.

    NFKC, reply/forward prefixes stripped repeatedly (``Re: Fwd: Re:``),
    whitespace collapsed, case-folded.  Punctuation is **kept**: complaint
    references such as ``CMS/2026/401687`` live in subjects and are the most
    discriminating thing in them.
    """
    text = unicodedata.normalize("NFKC", subject)
    while True:
        stripped = _SUBJECT_PREFIX.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return _WHITESPACE.sub(" ", text).strip().casefold()


def soft_duplicate_key(
    sender: str, subject: str, when: datetime | date
) -> str:
    """``sha256:<hex>`` over ``(sender, normalised subject, calendar day)``.

    A **pure function**.  It does not look anything up, does not decide that a
    document is a duplicate, and holds no state: the store that answers "have we
    seen this key today?" arrives with WS-F, and keeping the key derivation here
    means the storage layer cannot quietly change what "same complaint" means.

    ``when`` is truncated to its calendar day.  A timezone-aware datetime is
    used as given — the mailbox's local day is the operative one for RBI's
    same-day rule, so the caller, not this function, decides the timezone.
    """
    day = when.date() if isinstance(when, datetime) else when
    payload = "\x1f".join(
        (
            unicodedata.normalize("NFKC", sender).strip().casefold(),
            normalise_subject(subject),
            day.isoformat(),
        )
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Type sniffing
# --------------------------------------------------------------------------- #


def sniff_type(data: bytes) -> SniffedType:
    """Identify ``data`` from its magic bytes alone.

    Raises:
        EmptyFileError: on zero bytes.
        MimeMismatchError: when the content is not one of the three supported
            types.  The message carries a hex prefix, never the bytes
            themselves — the payload is attacker-controlled and must not be
            echoed into a log or an auto-reply.
    """
    if not data:
        raise EmptyFileError(
            "attachment is zero bytes", context={"byte_size": 0}
        )
    if data[:3] == MAGIC_SIGNATURES["image/jpeg"]:
        return SniffedType.JPEG
    if data[:8] == MAGIC_SIGNATURES["image/png"]:
        return SniffedType.PNG
    if MAGIC_SIGNATURES["application/pdf"] in data[:_PDF_HEADER_WINDOW]:
        return SniffedType.PDF
    raise MimeMismatchError(
        "content does not match any supported type (PDF, JPEG, PNG)",
        context={
            "byte_size": len(data),
            "magic_prefix_hex": data[:8].hex(),
            "supported": sorted(t.value for t in SniffedType),
        },
    )


_EXTENSION_TYPES: Final[dict[str, SniffedType]] = {
    ".pdf": SniffedType.PDF,
    ".jpg": SniffedType.JPEG,
    ".jpeg": SniffedType.JPEG,
    ".jpe": SniffedType.JPEG,
    ".png": SniffedType.PNG,
}


def _extension_of(filename: str | None) -> str | None:
    if not filename:
        return None
    suffix = Path(filename).suffix.lower()
    return suffix or None


# --------------------------------------------------------------------------- #
# Resource guards
# --------------------------------------------------------------------------- #


def assert_decompression_ratio(
    compressed_bytes: int, decoded_bytes: int, caps: ResourceCaps
) -> float:
    """Enforce ``MAX_DECOMPRESSION_RATIO`` and return the observed ratio.

    Public because it is the guard any *future* expansion step must reuse — the
    zip attachment WS-F may one day accept, a decompressed PDF object stream, a
    TIFF strip.  The floor
    (:attr:`ResourceCaps.decompression_min_decoded_bytes`) is what stops the
    rule from refusing a perfectly ordinary blank scan that happens to compress
    700:1.

    Raises:
        DecompressionBombError: when both the ratio and the floor are exceeded.
    """
    if compressed_bytes <= 0:
        return 0.0
    ratio = decoded_bytes / compressed_bytes
    if (
        decoded_bytes >= caps.decompression_min_decoded_bytes
        and ratio > caps.max_decompression_ratio
    ):
        raise DecompressionBombError(
            f"decoded {decoded_bytes} bytes from {compressed_bytes} "
            f"(ratio {ratio:.1f}) exceeds the {caps.max_decompression_ratio}:1 cap",
            context={
                "compressed_bytes": compressed_bytes,
                "decoded_bytes": decoded_bytes,
                "ratio": round(ratio, 2),
                "limit": caps.max_decompression_ratio,
                "floor_bytes": caps.decompression_min_decoded_bytes,
            },
        )
    return ratio


class _Deadline:
    """Wall-clock budget for one document."""

    __slots__ = ("_clock", "_limit", "_start")

    def __init__(self, limit_seconds: float, clock: WallClock) -> None:
        self._clock = clock
        self._limit = limit_seconds
        self._start = clock()

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def check(self, stage: str) -> None:
        """Raise when the budget is spent.

        Raises:
            WallClockExceededError: always, when over budget.
        """
        elapsed = self.elapsed
        if elapsed > self._limit:
            raise WallClockExceededError(
                f"ingestion exceeded {self._limit:.1f}s at {stage}",
                context={
                    "stage": stage,
                    "elapsed_seconds": round(elapsed, 3),
                    "limit_seconds": self._limit,
                },
            )


@contextmanager
def _pillow_bomb_guard(max_pixels: int) -> Iterator[None]:
    """Make Pillow's decompression-bomb defence a typed error, not a warning.

    Pillow's default is a :class:`PIL.Image.DecompressionBombWarning` above
    ``MAX_IMAGE_PIXELS`` and a hard error above twice that.  A warning is
    useless at a security boundary — it is printed once per process and then
    suppressed.  Inside this context the limit is *our* pixel cap, the warning
    is promoted to an exception, and both it and Pillow's own error surface as
    :class:`~app.contracts.errors.PixelCapExceededError`.

    The previous ``MAX_IMAGE_PIXELS`` is restored on exit, so importing this
    module cannot change Pillow's behaviour for anyone else in the process.
    """
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


# --------------------------------------------------------------------------- #
# PDF handling
# --------------------------------------------------------------------------- #


def pdf_declares_encryption(data: bytes) -> bool:
    """True when the file carries an ``/Encrypt`` entry.

    Checked on raw bytes *before* PDFium is asked to open the document, so an
    encrypted file is refused without a parse.  PDFium's own password error is
    still caught in :func:`open_pdf`, because an ``/Encrypt`` inside a
    compressed cross-reference stream is invisible to this scan.  Two
    independent checks, one verdict: we never guess or crack a password.
    """
    return _ENCRYPT_PATTERN.search(data) is not None


def scan_pdf_active_content(data: bytes) -> PdfActiveContent:
    """Record which active-content markers the raw bytes declare.

    **The choice, recorded here because the brief asks for one: we strip by
    construction and flag by record — we never rewrite the file.**

    Rewriting a hostile PDF to remove ``/JavaScript`` means parsing and
    re-serialising attacker-controlled structure with a library that was not
    written as a sanitiser, and then trusting the output.  That adds attack
    surface to remove attack surface.  Instead the *rendering* path is stripped:
    :func:`_render_page` never calls ``init_forms()``, passes
    ``may_draw_forms=False`` and ``draw_annots=False``, and takes only the page
    content stream.  No action, no JavaScript, no embedded file and no
    annotation is ever evaluated, so the marker is a fact for the audit trail
    and the security queue rather than a thing we must neutralise.

    The scan is on raw bytes and will miss a marker inside a compressed object
    stream.  That is acceptable precisely because it is not load-bearing.
    """
    found = {
        marker.decode("ascii")
        for marker in PDF_ACTIVE_CONTENT_MARKERS
        if marker in data
    }
    # "/JS" is a substring of nothing else in the list, but "/AA" would also
    # match inside a name like "/AAPL"; require a delimiter after short names.
    if "/AA" in found and not re.search(rb"/AA\s*(?:<<|\d+\s+\d+\s+R|\[)", data):
        found.discard("/AA")
    return PdfActiveContent(markers=tuple(sorted(found)))


#: A buffer that is deterministically *not* a PDF.  Loading it fails and, in
#: doing so, writes ``FPDF_ERR_FORMAT`` into PDFium's global last-error slot —
#: see :func:`_prime_pdfium_error` for why that matters.
_PDFIUM_ERROR_PRIMER: Final[bytes] = b"\x00 not a pdf \x00"

#: Serialises prime-then-load so two threads cannot interleave their primers.
#: PDFium is not thread-safe anyway; this lock is held only across the load, not
#: across the caller's use of the document.
_PDFIUM_LOAD_LOCK: Final[threading.Lock] = threading.Lock()


def _prime_pdfium_error() -> int | None:
    """Overwrite PDFium's global last-error with a known sentinel.

    ``FPDF_GetLastError()`` is **process-global and sticky**: PDFium writes it
    on the failure paths that classify a failure, and leaves the previous value
    untouched on the ones that do not.  A structurally invalid document — a PDF
    with zero pages, for instance — fails to load *without* setting an error, so
    the code read back is whatever the last failure left behind.  If that was an
    encrypted document, a plain malformed PDF is reported as
    ``ENCRYPTED_PDF``, and the complainant receives an auto-reply asking for the
    password of a file that has none (PRD P1-2 keys the template on the code).

    Priming with a buffer that is definitely not a PDF puts a known value there
    first, so :func:`open_pdf` can tell "PDFium said password" from "PDFium said
    nothing and we are reading a ghost".  Returns the sentinel, or ``None`` when
    it could not be established — in which case the caller falls back to
    trusting the reported code, which is the previous behaviour.
    """
    try:
        document = pdfium.PdfDocument(_PDFIUM_ERROR_PRIMER)
    except pdfium.PdfiumError as exc:
        sentinel = getattr(exc, "err_code", None)
        if sentinel is None:
            sentinel = int(pdfium_raw.FPDF_GetLastError())
        # A sentinel equal to the value we are trying to detect is no sentinel.
        if int(sentinel) == int(pdfium_raw.FPDF_ERR_PASSWORD):
            return None
        return int(sentinel)
    else:  # pragma: no cover - the primer is not a PDF; it cannot open
        document.close()
        return None


def _load_pdfium_document(data: bytes) -> pdfium.PdfDocument:
    """``pdfium.PdfDocument(data)`` with a trustworthy failure classification.

    Raises:
        EncryptedPDFError: PDFium reported a *fresh* password error.
        MalformedPDFError: every other load failure, including the ones PDFium
            declines to classify at all.
    """
    with _PDFIUM_LOAD_LOCK:
        sentinel = _prime_pdfium_error()
        try:
            return pdfium.PdfDocument(data)
        except pdfium.PdfiumError as exc:
            code = getattr(exc, "err_code", None)
            if code is None:  # pragma: no cover - older pypdfium2
                code = int(pdfium_raw.FPDF_GetLastError())
            code = int(code)
            stale = sentinel is not None and code == sentinel
            if not stale and code == int(pdfium_raw.FPDF_ERR_PASSWORD):
                raise EncryptedPDFError(
                    "PDF is password-protected",
                    context={"byte_size": len(data), "detected_by": "pdfium"},
                ) from exc
            raise MalformedPDFError(
                "PDF could not be parsed",
                context={
                    "byte_size": len(data),
                    "pdfium_error": str(exc),
                    "pdfium_error_code": code,
                    "pdfium_error_was_stale": stale,
                },
            ) from exc


@contextmanager
def open_pdf(data: bytes) -> Iterator[pdfium.PdfDocument]:
    """Open ``data`` as a PDF, translating PDFium failures into typed errors.

    Two independent checks, one verdict.  The raw-byte ``/Encrypt`` scan runs
    first so an encrypted file is refused without a parse; PDFium's own verdict
    is the backstop for an ``/Encrypt`` hidden inside a compressed
    cross-reference stream, and is read through :func:`_prime_pdfium_error`
    because the slot it lives in is global and sticky.

    Raises:
        EncryptedPDFError: the document is password-protected.  We never try a
            password, including the empty one.
        MalformedPDFError: truncated, structurally invalid, zero-page, or
            otherwise unopenable.
    """
    if pdf_declares_encryption(data):
        raise EncryptedPDFError(
            "PDF declares an /Encrypt dictionary; password-protected documents "
            "are refused, never cracked",
            context={"byte_size": len(data), "detected_by": "encrypt_dictionary"},
        )
    document = _load_pdfium_document(data)
    try:
        yield document
    finally:
        document.close()


def _render_page(page: pdfium.PdfPage, dpi: float) -> PageImage:
    """Rasterise one page content stream to a BGR array.

    ``may_draw_forms=False`` and ``draw_annots=False`` are the strip described
    in :func:`scan_pdf_active_content`: only the page's own content is drawn,
    never a widget, never an annotation appearance stream, never a form field
    whose value came from an action.
    """
    bitmap = page.render(
        scale=dpi / PDF_POINTS_PER_INCH,
        may_draw_forms=False,
        draw_annots=False,
    )
    try:
        array = np.asarray(bitmap.to_numpy())
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.shape[2] == 4:
            array = array[:, :, :3]
        return np.ascontiguousarray(array, dtype=np.uint8)
    finally:
        bitmap.close()


def _ingest_pdf(
    data: bytes,
    caps: ResourceCaps,
    deadline: _Deadline,
    *,
    rasterize: bool,
) -> tuple[list[RasterPage], PdfActiveContent, int, list[IngestFlag]]:
    flags: list[IngestFlag] = []
    active = scan_pdf_active_content(data)
    if active.present:
        flags.append(IngestFlag.PDF_ACTIVE_CONTENT)
        _LOGGER.warning(
            "pdf declares active content markers=%s (never executed)",
            ",".join(active.markers),
        )

    pages: list[RasterPage] = []
    total_pixels = 0
    with open_pdf(data) as document:
        page_count = len(document)
        if page_count == 0:
            raise MalformedPDFError(
                "PDF contains no pages",
                context={"byte_size": len(data), "page_count": 0},
            )
        if page_count > caps.max_pages:
            raise PageCapExceededError(
                f"{page_count} pages exceeds the {caps.max_pages}-page cap",
                context={"page_count": page_count, "limit": caps.max_pages},
            )
        deadline.check("pdf_open")

        # Budget the whole document from the declared page geometry *before*
        # allocating a single bitmap: a 200 000-point MediaBox in a 3 KB file
        # must be refused, not rendered and then noticed.
        scale = RASTER_DPI / PDF_POINTS_PER_INCH
        planned = 0
        for index in range(page_count):
            width_pt, height_pt = document.get_page_size(index)
            planned += int(round(width_pt * scale)) * int(round(height_pt * scale))
            if planned > caps.max_total_pixels:
                raise PixelCapExceededError(
                    f"rasterising {page_count} page(s) at {RASTER_DPI:g} DPI needs "
                    f"at least {planned} pixels, over the "
                    f"{caps.max_total_pixels} cap",
                    context={
                        "planned_pixels": planned,
                        "limit": caps.max_total_pixels,
                        "page_count": page_count,
                        "dpi": RASTER_DPI,
                    },
                )

        if not rasterize:
            flags.append(IngestFlag.RASTERISATION_SKIPPED)
            return [], active, page_count, flags

        for index in range(page_count):
            deadline.check(f"raster_page_{index + 1}")
            page = document[index]
            try:
                image = _render_page(page, RASTER_DPI)
            finally:
                page.close()
            total_pixels += int(image.shape[0]) * int(image.shape[1])
            if total_pixels > caps.max_total_pixels:
                raise PixelCapExceededError(
                    f"rasterised {total_pixels} pixels, over the "
                    f"{caps.max_total_pixels} cap",
                    context={
                        "total_pixels": total_pixels,
                        "limit": caps.max_total_pixels,
                        "page_no": index + 1,
                    },
                )
            pages.append(
                RasterPage(
                    page_no=index + 1,
                    image=image,
                    source=PageSource.PDF_RASTER,
                    dpi=RASTER_DPI,
                )
            )
            _PAGES_DECODED.labels(source=PageSource.PDF_RASTER.value).inc()
    return pages, active, len(pages), flags


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #


def _ingest_image(
    data: bytes,
    sniffed: SniffedType,
    caps: ResourceCaps,
    deadline: _Deadline,
) -> tuple[list[RasterPage], float, list[IngestFlag]]:
    flags: list[IngestFlag] = []
    with _pillow_bomb_guard(caps.max_total_pixels):
        try:
            with Image.open(io.BytesIO(data)) as opened:
                width, height = opened.size
                channels = len(opened.getbands()) or 1
                pixels = width * height
                if pixels > caps.max_total_pixels:
                    raise PixelCapExceededError(
                        f"image is {width}x{height} = {pixels} pixels, over the "
                        f"{caps.max_total_pixels} cap",
                        context={
                            "width_px": width,
                            "height_px": height,
                            "total_pixels": pixels,
                            "limit": caps.max_total_pixels,
                        },
                    )
                ratio = assert_decompression_ratio(
                    len(data), pixels * channels, caps
                )
                deadline.check("image_header")
                oriented = ImageOps.exif_transpose(opened)
                if oriented is None:  # pragma: no cover - Pillow always returns
                    oriented = opened
                elif oriented.size != opened.size or _has_exif_orientation(opened):
                    flags.append(IngestFlag.EXIF_ORIENTATION_APPLIED)
                if oriented.mode in ("RGBA", "LA", "PA") or (
                    oriented.mode == "P" and "transparency" in oriented.info
                ):
                    flags.append(IngestFlag.ALPHA_FLATTENED)
                    background = Image.new("RGB", oriented.size, (255, 255, 255))
                    converted = oriented.convert("RGBA")
                    background.paste(converted, mask=converted.split()[-1])
                    rgb = background
                else:
                    rgb = oriented.convert("RGB")
                array = np.asarray(rgb, dtype=np.uint8)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise PixelCapExceededError(
                "Pillow refused the image as a decompression bomb",
                context={"limit": caps.max_total_pixels, "detail": str(exc)},
            ) from exc
        except (UnidentifiedImageError, SyntaxError, ValueError) as exc:
            raise MimeMismatchError(
                f"magic bytes claim {sniffed.value} but the payload does not "
                "decode as one",
                context={
                    "sniffed": sniffed.value,
                    "byte_size": len(data),
                    "detail": type(exc).__name__,
                },
            ) from exc
        except OSError as exc:
            raise MimeMismatchError(
                f"magic bytes claim {sniffed.value} but the payload is truncated "
                "or corrupt",
                context={
                    "sniffed": sniffed.value,
                    "byte_size": len(data),
                    "detail": type(exc).__name__,
                },
            ) from exc

    deadline.check("image_decode")
    bgr = np.ascontiguousarray(array[:, :, ::-1])
    _PAGES_DECODED.labels(source=PageSource.IMAGE.value).inc()
    return (
        [RasterPage(page_no=1, image=bgr, source=PageSource.IMAGE, dpi=None)],
        ratio,
        flags,
    )


def _has_exif_orientation(image: Image.Image) -> bool:
    """True when the file declared a non-default EXIF orientation."""
    getexif = getattr(image, "getexif", None)
    if getexif is None:  # pragma: no cover - all modern Pillow images have it
        return False
    try:
        exif = getexif()
    except (OSError, ValueError, AttributeError):  # pragma: no cover
        return False
    return bool(exif) and exif.get(0x0112, 1) not in (1, None)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def ingest(
    data: bytes,
    *,
    filename: str | None = None,
    settings: Settings | None = None,
    caps: ResourceCaps | None = None,
    scanner: Scanner | None = None,
    clock: WallClock | None = None,
    rasterize: bool = True,
) -> IngestResult:
    """Sniff, scan, cap and decode one attachment.

    Args:
        data: the attachment bytes, exactly as received.
        filename: advisory only.  Used for the extension-mismatch flag and
            passed to the scanner; **never** used to decide the type.
        settings: overrides the process settings singleton.
        caps: overrides the caps derived from ``settings``.
        scanner: malware-scan seam; :class:`NoOpScanner` when omitted.
        clock: monotonic clock, injected by the wall-clock test.
        rasterize: when false, PDF pages are validated and budgeted but not
            rendered.  Stage 1b uses this to decide the bypass before paying
            for pixels it may not need.

    Raises:
        EmptyFileError, MimeMismatchError, MalwareDetected, EncryptedPDFError,
        MalformedPDFError, PageCapExceededError, PixelCapExceededError,
        DecompressionBombError, WallClockExceededError: the typed rejections of
        BUILD-BRIEF M3 and TEST-DATA-SPEC sec. 5.  Nothing else escapes this
        function for a malformed input; that is what the adversarial golden test
        asserts.
    """
    effective_settings = settings if settings is not None else get_settings()
    effective_caps = caps if caps is not None else ResourceCaps.from_settings(
        effective_settings
    )
    effective_scanner: Scanner = scanner if scanner is not None else NoOpScanner()
    deadline = _Deadline(
        effective_caps.max_wall_clock_seconds,
        clock if clock is not None else time.monotonic,
    )
    started = time.perf_counter()
    extension = _extension_of(filename)
    sniffed_label = "unknown"

    try:
        if not data:
            raise EmptyFileError(
                "attachment is zero bytes",
                context={"byte_size": 0, "extension": extension},
            )

        verdict = effective_scanner.scan(data, filename=filename)
        if not verdict.clean:
            raise MalwareDetected(
                "attachment refused by the malware scanner",
                signature=verdict.signature,
                scanner=verdict.scanner,
                context={"byte_size": len(data)},
            )
        deadline.check("scan")

        sniffed = sniff_type(data)
        sniffed_label = sniffed.value
        flags: list[IngestFlag] = []
        if extension is not None and _EXTENSION_TYPES.get(extension) not in (
            None,
            sniffed,
        ):
            flags.append(IngestFlag.EXTENSION_MISMATCH)
            _LOGGER.warning(
                "extension %s disagrees with sniffed type %s; magic bytes win",
                extension,
                sniffed.value,
            )

        active = PdfActiveContent()
        ratio: float | None = None
        if sniffed is SniffedType.PDF:
            pages, active, page_count, pdf_flags = _ingest_pdf(
                data, effective_caps, deadline, rasterize=rasterize
            )
            flags.extend(pdf_flags)
            raster_dpi: float | None = RASTER_DPI if rasterize else None
        else:
            pages, ratio, image_flags = _ingest_image(
                data, sniffed, effective_caps, deadline
            )
            flags.extend(image_flags)
            page_count = len(pages)
            raster_dpi = None
    except (DocumentRejected, MalwareDetected) as exc:
        _INGESTED.labels(sniffed=sniffed_label, outcome=exc.code).inc()
        raise

    report = IngestReport(
        document_id=idempotency_key(data),
        sniffed_type=sniffed,
        byte_size=len(data),
        page_count=page_count,
        total_pixels=sum(page.pixels for page in pages),
        raster_dpi=raster_dpi,
        decompression_ratio=ratio,
        scanner=verdict.scanner,
        active_content=active,
        flags=tuple(flags),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        source_extension=extension,
    )
    _INGESTED.labels(sniffed=sniffed_label, outcome="accepted").inc()
    _INGEST_LATENCY.labels(sniffed=sniffed_label).observe(
        report.duration_ms / 1000.0
    )
    return IngestResult(report=report, pages=tuple(pages), data=data)


def ingest_path(path: Path | str, **kwargs: Any) -> IngestResult:
    """:func:`ingest` for a file on disk.  ``filename`` defaults to its name."""
    resolved = Path(path)
    kwargs.setdefault("filename", resolved.name)
    return ingest(resolved.read_bytes(), **kwargs)


def page_images(pages: Sequence[RasterPage]) -> list[PageImage]:
    """The raw arrays of ``pages``, in order — a convenience for stage tests."""
    return [page.image for page in pages]
