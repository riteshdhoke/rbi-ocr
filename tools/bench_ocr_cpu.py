"""M0 throughput spike — can PP-OCRv5 on a CPU carry this service?

BUILD-BRIEF sec. 9 calls this "the answer to the riskiest assumption in the
whole design".  Everything downstream assumes a laptop-class CPU can OCR a
complaint page inside the fast-path SLA; if it cannot, the architecture is
wrong and no amount of service code fixes it.  So this tool measures, on the
real machine, with the real weights:

* **per-page wall clock for detection + recognition** at concurrency 1 and 2
  (a thread pool of pipeline workers; intra-op threads stay at 2 — raising them
  to "speed things up" costs 3-5x, BUILD-BRIEF sec. 8);
* **looped versus batched recognition** across batch 2 / 4 / 8 / 16 / 24 / 32,
  and the speedup — split into padding, inference and CTC decode, because
  "batching is slower" and "batching is faster but the decode dominates" call
  for opposite responses and an aggregate second cannot tell them apart.
  PLAN.md invariant 4 forbids a looped recognition path everywhere except here,
  which is why the loop below is the only one in the repository;
* **p50 / p95 per degradation profile**, so a single slow profile cannot hide
  inside an aggregate;
* the machine it ran on, because a throughput number without a CPU model is an
  anecdote.

Everything is written to ``bench_results.json`` at the repository root and the
gate — **p95 <= 4 s/page at target concurrency** — is evaluated and printed as
PASS or FAIL.  A FAIL is a blocking finding for OPEN-QUESTIONS, not a nudge.

Usage::

    .venv\\Scripts\\python.exe tools/bench_ocr_cpu.py                 # 100 pages
    .venv\\Scripts\\python.exe tools/bench_ocr_cpu.py --pages 20
    .venv\\Scripts\\python.exe tools/bench_ocr_cpu.py --dry-run       # no weights
    .venv\\Scripts\\python.exe tools/bench_ocr_cpu.py --no-write      # print only

``--dry-run`` swaps the real provider for a deterministic stand-in and
synthesises page pixels, so every line of sampling, timing, aggregation and
report construction runs on a machine with an empty ``models/`` and no corpus
decode.  That is what makes the tool unit-testable; it is **not** a measurement
and the report says so in ``mode``.

Thread caps are applied by importing :mod:`app` before NumPy, OpenCV or ONNX
Runtime (PLAN.md invariant 5).  The import block below is ordered deliberately
and must not be re-sorted.

This tool does NOT handle:
  * the quality gate, script ID, LID, classification, the SLM or translation.
    M0 is a question about *Tier-1 OCR* throughput and mixing other stages in
    would make the number unattributable;
  * page decode time.  Pages are decoded into memory before the clock starts,
    because at concurrency 2 a worker decoding JPEG would steal cores from
    another worker's timed recognition and quietly inflate its p95;
  * accuracy.  ``tools/eval_harness.py`` owns CER/WER; the only correctness
    check here is that looped and batched recognition agree on the text, which
    is a *batching* assertion, not an OCR one;
  * choosing ``OCR_BATCH_SIZE``.  It reports the curve; the value lives in
    ``app/settings.py``;
  * multi-process concurrency.  Phase 1 runs one model set per Celery child;
    this measures the in-process thread pool the Phase 0 orchestrator uses, and
    the report records which one it was.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `app` must be imported before numpy / cv2 / onnxruntime so the thread-cap
# preamble runs first (PLAN.md invariant 5).  Do not re-sort this block.
import app

#: Numeric modules already loaded when this tool imported ``app``.  Captured
#: **here**, not at report time — by then NumPy, OpenCV and ONNX Runtime are all
#: (correctly) loaded, and reporting that would read like a violation of
#: invariant 5 rather than the normal state of a running benchmark.  A non-empty
#: tuple means the caps arrived late and the timings are suspect.
NUMERIC_LOADED_BEFORE_APP: tuple[str, ...] = app.late_numeric_imports()

import numpy as np

from app.contracts.ocr import OCRLine
from app.providers.local import (
    REC_IMAGE_HEIGHT,
    REC_WIDTH_BUCKETS,
    LocalProvider,
    RecognitionBatch,
    ctc_greedy_decode,
    pad_crops_to_batch,
    plan_recognition_batches,
)
from app.settings import PROJECT_ROOT as APP_PROJECT_ROOT
from app.settings import Settings, get_settings

__all__ = [
    "BATCH_SIZES",
    "DEFAULT_OUTPUT_PATH",
    "GATE_P95_SECONDS",
    "REPORT_SCHEMA_VERSION",
    "BenchDoc",
    "CropSink",
    "DryRunProvider",
    "Harness",
    "LoadedPage",
    "PageTiming",
    "PhaseTimer",
    "RecognitionRun",
    "TimedSession",
    "TimingSessionFactory",
    "build_parser",
    "build_provider",
    "build_report",
    "evaluate_gate",
    "exact_width_plan",
    "load_pages",
    "machine_info",
    "main",
    "measure_batching",
    "measure_concurrency",
    "percentile",
    "runtime_versions",
    "sample_documents",
    "summarise",
]

#: Bumped whenever the shape of ``bench_results.json`` changes incompatibly.
REPORT_SCHEMA_VERSION: int = 1

#: BUILD-BRIEF M0: "if p95 > 4 s/page at target concurrency, stop".
GATE_P95_SECONDS: float = 4.0

#: Batch sizes compared against the looped baseline.  16 and 32 are the two the
#: brief names; 24 is the shipped ``OCR_BATCH_SIZE`` default, and 2/4/8 are here
#: because a two-point comparison cannot tell "batching does not help" from
#: "batching helps but 16 is past the optimum" — and on a cache-limited CPU
#: those are very different findings with very different remedies.
BATCH_SIZES: tuple[int, ...] = (2, 4, 8, 16, 24, 32)

DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "bench_results.json"

#: Filename prefixes that map an ONNX session onto a pipeline phase.
_DETECTION_PREFIX = "det"
_RECOGNITION_PREFIX = "rec"


# --------------------------------------------------------------------------- #
# Machine description
# --------------------------------------------------------------------------- #


def _windows_cpu_name() -> str | None:
    """CPU brand string from the registry (``winreg`` is stdlib on Windows)."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except OSError:
        return None
    return str(value).strip()


def _linux_cpu_name() -> str | None:
    """CPU brand string from ``/proc/cpuinfo``."""
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - permission-dependent
        return None
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() in ("model name", "Model name"):
            return value.strip()
    return None


def cpu_model_name() -> str:
    """Best available CPU brand string, never an empty answer."""
    for candidate in (_windows_cpu_name(), _linux_cpu_name(), platform.processor()):
        if candidate:
            return str(candidate)
    return platform.machine() or "unknown"


class _MemoryStatusEx(ctypes.Structure):
    """``MEMORYSTATUSEX`` — the Win32 physical-memory record."""

    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def total_ram_bytes() -> int | None:
    """Installed physical RAM, or ``None`` when the OS will not say.

    ``psutil`` is not a dependency (and would be a licence-gate review for a
    number two stdlib calls already answer), so this is ``GlobalMemoryStatusEx``
    on Windows and ``sysconf`` elsewhere.
    """
    if sys.platform == "win32":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        ):
            return None
        return int(status.ullTotalPhys)
    try:  # pragma: no cover - POSIX only
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        return None
    return int(page_size) * int(page_count)  # pragma: no cover


class _ProcessorInfo(ctypes.Structure):
    """``SYSTEM_LOGICAL_PROCESSOR_INFORMATION`` (the fixed-size variant)."""

    _fields_ = (
        ("ProcessorMask", ctypes.c_void_p),
        ("Relationship", ctypes.c_ulong),
        ("Reserved", ctypes.c_ulonglong * 2),
    )


def physical_core_count() -> int | None:
    """Physical cores, or ``None`` when the OS will not say.

    Physical cores matter more than logical ones here: the thread budget rule is
    ``CELERY_CONCURRENCY x ORT_INTRA_OP_THREADS <= physical cores`` (PRD P0-11),
    and a report that only knew the logical count would make a 2x2 configuration
    on a 2-core machine look compliant.
    """
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    length = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    if length.value == 0:
        return None
    entry_size = ctypes.sizeof(_ProcessorInfo)
    count = length.value // entry_size
    buffer = (_ProcessorInfo * count)()
    if not kernel32.GetLogicalProcessorInformation(
        ctypes.byref(buffer), ctypes.byref(length)
    ):
        return None
    #: ``RelationProcessorCore`` == 0.
    return sum(1 for entry in buffer if entry.Relationship == 0) or None


def machine_info() -> dict[str, Any]:
    """Everything needed to know whether this number transfers to another box."""
    ram = total_ram_bytes()
    return {
        "cpu_model": cpu_model_name(),
        "cpu_cores_physical": physical_core_count(),
        "cpu_cores_logical": os.cpu_count(),
        "ram_bytes": ram,
        "ram_gib": round(ram / (1024**3), 2) if ram is not None else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "hostname_hashed": _short_digest(platform.node()),
    }


def _short_digest(value: str) -> str:
    """A stable, non-identifying stand-in for a hostname."""
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def runtime_versions() -> dict[str, str | None]:
    """Versions of the libraries whose behaviour the numbers depend on."""
    versions: dict[str, str | None] = {"numpy": np.__version__}
    for name in ("onnxruntime", "cv2"):
        try:
            module = __import__(name)
        except ImportError:  # pragma: no cover - both are hard dependencies
            versions[name] = None
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def thread_configuration(settings: Settings) -> dict[str, Any]:
    """The caps actually in effect, as opposed to the ones we meant to set."""
    return {
        "environment": {
            name: os.environ.get(name) for name in app.THREAD_CAP_VARS
        },
        "applied_by_app_import": dict(app.THREAD_CAPS_APPLIED),
        "ort_intra_op_threads": settings.ORT_INTRA_OP_THREADS,
        "ort_inter_op_threads": 1,
        "celery_concurrency": settings.CELERY_CONCURRENCY,
        "ocr_batch_size": settings.OCR_BATCH_SIZE,
        "numeric_modules_loaded_before_app": list(NUMERIC_LOADED_BEFORE_APP),
        "thread_caps_honoured": not NUMERIC_LOADED_BEFORE_APP,
    }


# --------------------------------------------------------------------------- #
# Corpus sampling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchDoc:
    """One corpus page selected for the benchmark."""

    doc_id: str
    doc_type: str
    profile: str
    script: str
    variant: int
    image_path: Path
    expected_lines: int


def load_index(corpus_dir: Path) -> dict[str, Any]:
    """Read ``corpus/index.json``."""
    payload = json.loads((corpus_dir / "index.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "documents" not in payload:
        raise ValueError(
            f"{corpus_dir / 'index.json'} is not a corpus index "
            "(no 'documents' key); regenerate it with tools/generate_test_corpus.py"
        )
    return payload


def sample_documents(
    index: dict[str, Any],
    *,
    count: int,
    seed: int,
    corpus_dir: Path,
    profiles: Sequence[str] | None = None,
) -> list[BenchDoc]:
    """Pick ``count`` pages spread evenly over ``(type, profile)`` cells.

    Seed-stable and order-stable: the same seed on the same index always yields
    the same pages in the same order, so two runs on two machines compare like
    with like.  Selection is round-robin over shuffled cells rather than a flat
    random sample, because a flat sample of 100 from 295 leaves whole
    ``(type, profile)`` cells empty often enough to matter — and a profile with
    no pages in it is a p95 nobody can interpret.

    Raises:
        ValueError: for a non-positive ``count`` or a profile filter that
            matches nothing.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    wanted_profiles = set(profiles) if profiles else None

    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in index["documents"]:
        profile = str(entry["profile"])
        if wanted_profiles is not None and profile not in wanted_profiles:
            continue
        cells.setdefault((str(entry["type"]), profile), []).append(entry)
    if not cells:
        raise ValueError(
            f"no corpus documents matched profiles {sorted(wanted_profiles or [])}"
        )

    rng = random.Random(seed)
    ordered_cells = sorted(cells)
    rng.shuffle(ordered_cells)
    queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in ordered_cells:
        members = sorted(cells[key], key=lambda item: str(item["doc_id"]))
        rng.shuffle(members)
        queues[key] = members

    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for key in ordered_cells:
            if not queues[key]:
                continue
            chosen.append(queues[key].pop(0))
            progressed = True
            if len(chosen) == count:
                break
        if not progressed:
            break

    return [
        BenchDoc(
            doc_id=str(entry["doc_id"]),
            doc_type=str(entry["type"]),
            profile=str(entry["profile"]),
            script=str(entry.get("primary_script") or "Latn"),
            variant=int(entry.get("variant", 0)),
            image_path=corpus_dir
            / Path(*str(entry["image"]).replace("\\", "/").split("/")),
            expected_lines=int(entry.get("lines", 0)),
        )
        for entry in chosen
    ]


@dataclass
class LoadedPage:
    """A decoded corpus page, ready for the clock."""

    doc: BenchDoc
    image: np.ndarray


def load_pages(docs: Sequence[BenchDoc]) -> list[LoadedPage]:
    """Decode every sampled page through Stage 1, before any timing starts.

    Uses the production ingestion path rather than ``cv2.imread`` so the pixels
    the benchmark measures are byte-identical to the pixels the service sees.
    """
    from app.stages.s1_ingest import ingest_path

    return [LoadedPage(doc=doc, image=ingest_path(doc.image_path).pages[0].image) for doc in docs]


def synthesise_pages(docs: Sequence[BenchDoc]) -> list[LoadedPage]:
    """Deterministic stand-in pixels for ``--dry-run`` (no corpus decode)."""
    pages: list[LoadedPage] = []
    for position, doc in enumerate(docs):
        rng = np.random.default_rng(position)
        image = rng.integers(200, 255, size=(64, 96, 3), dtype=np.uint8)
        pages.append(LoadedPage(doc=doc, image=image))
    return pages


# --------------------------------------------------------------------------- #
# Instrumentation — all of it through public LocalProvider seams
# --------------------------------------------------------------------------- #


class PhaseTimer:
    """Per-thread accumulation of ONNX time, split into detect and recognise.

    Thread-local rather than locked-and-shared: at concurrency 2 two workers run
    two pages at once, and a shared counter would attribute one worker's
    detection to the other worker's page.  Each worker resets, runs its page and
    reads back its own numbers.
    """

    PHASES = ("detect", "recognise", "other")

    def __init__(self) -> None:
        self._local = threading.local()

    def _bucket(self) -> dict[str, float]:
        bucket: dict[str, float] | None = getattr(self._local, "bucket", None)
        if bucket is None:
            bucket = {phase: 0.0 for phase in self.PHASES}
            bucket.update({f"{phase}_calls": 0.0 for phase in self.PHASES})
            self._local.bucket = bucket
        return bucket

    def add(self, phase: str, seconds: float) -> None:
        """Record one ONNX ``run()`` against ``phase`` on the calling thread."""
        bucket = self._bucket()
        key = phase if phase in self.PHASES else "other"
        bucket[key] += seconds
        bucket[f"{key}_calls"] += 1.0

    def reset(self) -> None:
        """Zero the calling thread's counters."""
        bucket = self._bucket()
        for key in bucket:
            bucket[key] = 0.0

    def snapshot(self) -> dict[str, float]:
        """The calling thread's counters."""
        return dict(self._bucket())


class TimedSession:
    """An ONNX Runtime session that reports how long ``run()`` took.

    Delegates everything else, so :class:`LocalProvider` cannot tell the
    difference — which is the point: the phase split is measured *inside* the
    real ``ocr_page`` call rather than in a re-implementation of it.
    """

    def __init__(self, inner: Any, phase: str, timer: PhaseTimer) -> None:
        self.inner = inner
        self.phase = phase
        self.timer = timer
        self.calls = 0

    def run(self, output_names: Any, input_feed: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.inner.run(output_names, input_feed, **kwargs)
        finally:
            self.timer.add(self.phase, time.perf_counter() - started)
            self.calls += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class TimingSessionFactory:
    """``session_factory`` for :class:`LocalProvider` that wraps each session.

    ``session_factory`` is a documented constructor argument, so nothing here
    reaches into the provider's internals.  Sessions are built with exactly the
    options the provider itself uses — ``intra_op = ORT_INTRA_OP_THREADS`` and
    ``inter_op = 1`` — because those options are half of what is being measured.
    """

    def __init__(self, settings: Settings, timer: PhaseTimer) -> None:
        self.settings = settings
        self.timer = timer
        self.sessions: dict[str, TimedSession] = {}
        self._lock = threading.Lock()

    def __call__(self, path: Path) -> TimedSession:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self.settings.ORT_INTRA_OP_THREADS
        options.inter_op_num_threads = 1
        # The PaddlePaddle exports carry unused initialisers and ORT warns about
        # every one of them at load; three thousand lines of graph-cleanup notes
        # would bury the benchmark's own output. Errors and fatals still print.
        options.log_severity_level = 3
        inner = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        session = TimedSession(inner, phase_of(path.name), self.timer)
        with self._lock:
            self.sessions[path.name] = session
        return session


def phase_of(filename: str) -> str:
    """Map a model filename onto a pipeline phase."""
    stem = filename.lower()
    if stem.startswith(_DETECTION_PREFIX):
        return "detect"
    if stem.startswith(_RECOGNITION_PREFIX):
        return "recognise"
    return "other"


class CropSink:
    """Captures the line crops ``ocr_page`` produced, for the batching study.

    Installed through ``handwriting_detector``, a public constructor argument
    that :class:`LocalProvider` calls once per recognised line with that line's
    crop.  Always answers ``False`` — there is no approved handwriting artifact
    and the benchmark must not invent one — so the OCR output is unchanged.
    """

    def __init__(self, limit: int = 4096) -> None:
        self.limit = limit
        self.enabled = False
        self.crops: list[np.ndarray] = []
        self._lock = threading.Lock()

    def __call__(self, crop: Any) -> bool:
        if self.enabled:
            with self._lock:
                if len(self.crops) < self.limit:
                    self.crops.append(np.asarray(crop))
        return False

    def reset(self) -> None:
        with self._lock:
            self.crops.clear()


@dataclass
class Harness:
    """The provider plus the instrumentation wired into it."""

    provider: Any
    timer: PhaseTimer
    sink: CropSink
    factory: TimingSessionFactory | None = None


def build_provider(settings: Settings, *, models_dir: Path | None = None) -> Harness:
    """A real :class:`LocalProvider` with timing and crop capture attached."""
    timer = PhaseTimer()
    sink = CropSink()
    factory = TimingSessionFactory(settings, timer)
    provider = LocalProvider(
        settings,
        models_dir=models_dir,
        session_factory=factory,
        handwriting_detector=sink,
    )
    return Harness(provider=provider, timer=timer, sink=sink, factory=factory)


class DryRunProvider:
    """Deterministic stand-in for ``--dry-run``: no weights, no ONNX, no clock lies.

    ``ocr_page`` returns a fixed number of synthetic lines after a tiny
    deterministic sleep, so the sampling, aggregation, gate and report code all
    execute.  The report records ``mode: "dry_run"`` and the gate result is
    marked ``advisory`` — a dry run must never be mistaken for a measurement.
    """

    name = "dry_run"

    def __init__(
        self,
        *,
        lines_per_page: int = 6,
        seconds_per_page: float = 0.002,
        crop_size: tuple[int, int] = (32, 128),
    ) -> None:
        self.lines_per_page = lines_per_page
        self.seconds_per_page = seconds_per_page
        self.crop_size = crop_size
        self.calls = 0
        self._lock = threading.Lock()

    def ocr_page(self, page: Any, script: str) -> list[OCRLine]:
        time.sleep(self.seconds_per_page)
        with self._lock:
            self.calls += 1
        height, width = np.asarray(page).shape[:2]
        return [
            OCRLine(
                text=f"dry-run line {index}",
                bbox=[
                    0.0,
                    float(index),
                    float(width - 1),
                    float(index),
                    float(width - 1),
                    float(min(height - 1, index + 10)),
                    0.0,
                    float(min(height - 1, index + 10)),
                ],
                confidence=0.5,
                script=script,
                language=None,
                is_handwritten=False,
                line_index=index,
            )
            for index in range(self.lines_per_page)
        ]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, taking the higher neighbour.

    Every number reported is therefore an *actual page time* traceable to a
    ``doc_id`` in ``pages``.  An interpolated p95 belongs to no page, and "which
    document is at the p95 of the heavy profile?" is the first question anyone
    asks when a profile starts drifting.
    """
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="higher"))


def summarise(values: Sequence[float]) -> dict[str, float | int]:
    """n / min / p50 / p90 / p95 / p99 / max / mean, rounded for a diffable file."""
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 4),
        "p50": round(percentile(values, 50), 4),
        "p90": round(percentile(values, 90), 4),
        "p95": round(percentile(values, 95), 4),
        "p99": round(percentile(values, 99), 4),
        "max": round(float(array.max()), 4),
        "mean": round(float(array.mean()), 4),
    }


# --------------------------------------------------------------------------- #
# The measurements
# --------------------------------------------------------------------------- #


@dataclass
class PageTiming:
    """One page's result at one concurrency level."""

    doc_id: str
    doc_type: str
    profile: str
    script: str
    seconds: float
    lines: int
    detect_seconds: float
    recognise_seconds: float
    error: str | None = None


def _time_one_page(
    provider: Any, timer: PhaseTimer, page: LoadedPage
) -> PageTiming:
    """Run one page and return its wall clock plus its phase split."""
    timer.reset()
    started = time.perf_counter()
    error: str | None = None
    lines: list[OCRLine] = []
    try:
        lines = provider.ocr_page(page.image, page.doc.script)
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    phases = timer.snapshot()
    return PageTiming(
        doc_id=page.doc.doc_id,
        doc_type=page.doc.doc_type,
        profile=page.doc.profile,
        script=page.doc.script,
        seconds=elapsed,
        lines=len(lines),
        detect_seconds=float(phases.get("detect", 0.0)),
        recognise_seconds=float(phases.get("recognise", 0.0)),
        error=error,
    )


def measure_concurrency(
    provider: Any,
    timer: PhaseTimer,
    pages: Sequence[LoadedPage],
    *,
    workers: int,
) -> dict[str, Any]:
    """Per-page wall clock over ``pages`` with a pool of ``workers`` threads.

    ``workers`` models pipeline workers; ONNX intra-op threads stay wherever
    ``ORT_INTRA_OP_THREADS`` put them, so ``workers x intra_op`` is the real
    load on the machine and the report says what that product was.

    A page that raises is recorded with its error and excluded from the
    percentiles — a typed failure is a correctness finding for another lane, and
    letting a 5 ms exception into a latency distribution would flatter it.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    started = time.perf_counter()
    if workers == 1:
        timings = [_time_one_page(provider, timer, page) for page in pages]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            timings = list(
                pool.map(lambda page: _time_one_page(provider, timer, page), pages)
            )
    wall = time.perf_counter() - started

    ok = [timing for timing in timings if timing.error is None]
    seconds = [timing.seconds for timing in ok]
    by_profile: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    by_script: dict[str, list[float]] = {}
    for timing in ok:
        by_profile.setdefault(timing.profile, []).append(timing.seconds)
        by_type.setdefault(timing.doc_type, []).append(timing.seconds)
        by_script.setdefault(timing.script, []).append(timing.seconds)

    return {
        "workers": workers,
        "pages_attempted": len(timings),
        "pages_measured": len(ok),
        "wall_seconds_total": round(wall, 3),
        "pages_per_minute": round(len(ok) * 60.0 / wall, 2) if wall > 0 else 0.0,
        "per_page_seconds": summarise(seconds),
        "by_profile": {
            profile: summarise(values) for profile, values in sorted(by_profile.items())
        },
        "by_type": {
            doc_type: summarise(values) for doc_type, values in sorted(by_type.items())
        },
        "by_script": {
            script: summarise(values) for script, values in sorted(by_script.items())
        },
        "phase_seconds": {
            "detect": summarise([timing.detect_seconds for timing in ok]),
            "recognise": summarise([timing.recognise_seconds for timing in ok]),
        },
        "lines_per_page": summarise([float(timing.lines) for timing in ok]),
        "errors": [
            {"doc_id": timing.doc_id, "error": timing.error}
            for timing in timings
            if timing.error is not None
        ],
        "_timings": timings,
    }


@dataclass
class RecognitionRun:
    """One pass of recognition over a fixed crop set at one batch size."""

    texts: list[str]
    total_seconds: float
    pad_seconds: float
    infer_seconds: float
    decode_seconds: float
    padded_pixels: int


def exact_width_plan(
    crops: Sequence[np.ndarray],
) -> tuple[RecognitionBatch, ...]:
    """One single-crop batch per crop, padded to that crop's own width.

    This is the *naive* loop — what a straight port of PaddleOCR's Python
    recogniser does, and the only mode in this file that does no bucketing at
    all.  It exists so the report can separate two effects the brief bundles
    together: the win from **bucketing** (less padding) and the win from
    **batching** (fewer calls).  Without it, "looped" would silently mean
    "bucketed, one at a time" and the comparison would flatter batching.
    """
    plan: list[RecognitionBatch] = []
    for index, crop in enumerate(crops):
        height, width = int(crop.shape[0]), int(crop.shape[1])
        scaled = max(1, -(-width * REC_IMAGE_HEIGHT // max(height, 1)))
        plan.append(
            RecognitionBatch(
                target_width=min(scaled, REC_WIDTH_BUCKETS[-1]), indices=(index,)
            )
        )
    return tuple(plan)


def _recognise_with_batch_size(
    session: Any,
    charset: Sequence[str],
    crops: Sequence[np.ndarray],
    batch_size: int,
    *,
    exact_width: bool = False,
) -> RecognitionRun:
    """Recognise ``crops`` at one batch size, split into its three costs.

    ``batch_size=1`` is the **looped** baseline PLAN.md invariant 4 permits only
    here: the same production helpers, one crop per call.  Any difference in the
    returned texts between batch sizes is a padding bug, not a tuning result,
    which is why the caller compares them.

    The pad / infer / decode split exists because "batching is slower" and
    "batching is faster but the Python CTC decode dominates" demand opposite
    responses, and an aggregate second cannot tell them apart.  ``padded_pixels``
    is the tensor volume actually pushed through the head, which is the cost
    bucketing exists to minimise.

    Args:
        exact_width: use :func:`exact_width_plan` instead of the bucketed
            planner. Only meaningful with ``batch_size == 1``.
    """
    shapes = [(int(crop.shape[0]), int(crop.shape[1])) for crop in crops]
    input_name = session.get_inputs()[0].name
    decoded: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    plan = (
        exact_width_plan(crops)
        if exact_width
        else plan_recognition_batches(shapes, batch_size=batch_size)
    )
    pad_seconds = 0.0
    infer_seconds = 0.0
    decode_seconds = 0.0
    padded_pixels = 0

    started = time.perf_counter()
    for batch in plan:
        mark = time.perf_counter()
        tensor = pad_crops_to_batch(crops, batch)
        pad_seconds += time.perf_counter() - mark
        padded_pixels += int(tensor.size)

        mark = time.perf_counter()
        raw = session.run(None, {input_name: tensor})[0]
        infer_seconds += time.perf_counter() - mark

        mark = time.perf_counter()
        probabilities = np.asarray(raw, dtype=np.float32)
        for slot, result in enumerate(ctc_greedy_decode(probabilities, charset)):
            decoded[batch.indices[slot]] = result
        decode_seconds += time.perf_counter() - mark
    total = time.perf_counter() - started

    return RecognitionRun(
        texts=[text for text, _ in decoded],
        total_seconds=total,
        pad_seconds=pad_seconds,
        infer_seconds=infer_seconds,
        decode_seconds=decode_seconds,
        padded_pixels=padded_pixels,
    )


def measure_batching(
    harness: Harness,
    pages: Sequence[LoadedPage],
    *,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    repeats: int = 3,
    max_crops: int = 600,
) -> dict[str, Any]:
    """Looped versus batched recognition over real crops, per script.

    Crops come from real ``ocr_page`` calls (captured through the public
    ``handwriting_detector`` seam), so they are the production line crops with
    the production aspect-ratio spread — a synthetic crop set would make the
    bucketing look tidier than it is and inflate the batched side.

    Grouped by script because each script has its own recognition head, and a
    speedup averaged over two different ONNX graphs answers no question.
    """
    provider = harness.provider
    sink = harness.sink
    by_script: dict[str, list[LoadedPage]] = {}
    for page in pages:
        by_script.setdefault(page.doc.script, []).append(page)

    results: dict[str, Any] = {}
    for script, script_pages in sorted(by_script.items()):
        sink.reset()
        sink.enabled = True
        try:
            for page in script_pages:
                provider.ocr_page(page.image, script)
                if len(sink.crops) >= max_crops:
                    break
        finally:
            sink.enabled = False
        crops = list(sink.crops[:max_crops])
        sink.reset()
        if not crops:
            results[script] = {"crops": 0, "note": "no crops recognised for this script"}
            continue

        head = provider.rec_heads.peek(script.lower())
        if head is None:  # pragma: no cover - only if the TTL expired mid-run
            results[script] = {
                "crops": len(crops),
                "note": "recognition head left the LRU cache before measurement",
            }
            continue
        session, charset = head

        # One untimed pass so ONNX Runtime's arena and the first-touch page
        # faults are not charged to whichever batch size happens to run first.
        _recognise_with_batch_size(session, charset, crops, 8)

        modes: dict[str, Any] = {}
        texts_by_mode: dict[str, list[str]] = {}
        plans: list[tuple[str, int, bool]] = [
            ("looped_exact_width", 1, True),
            ("looped", 1, False),
            *((f"batch_{size}", size, False) for size in batch_sizes),
        ]
        for label, size, exact in plans:
            runs = [
                _recognise_with_batch_size(
                    session, charset, crops, size, exact_width=exact
                )
                for _ in range(repeats)
            ]
            durations = [run.total_seconds for run in runs]
            median = float(np.median(durations))
            texts_by_mode[label] = runs[-1].texts
            modes[label] = {
                "batch_size": size,
                "bucketed": not exact,
                "calls": len(
                    exact_width_plan(crops)
                    if exact
                    else plan_recognition_batches(
                        [(int(c.shape[0]), int(c.shape[1])) for c in crops],
                        batch_size=size,
                    )
                ),
                "seconds_median": round(median, 4),
                "seconds_min": round(min(durations), 4),
                "crops_per_second": round(len(crops) / median, 1) if median > 0 else 0.0,
                "pad_seconds_median": round(
                    float(np.median([run.pad_seconds for run in runs])), 4
                ),
                "infer_seconds_median": round(
                    float(np.median([run.infer_seconds for run in runs])), 4
                ),
                "decode_seconds_median": round(
                    float(np.median([run.decode_seconds for run in runs])), 4
                ),
                "padded_pixels": runs[-1].padded_pixels,
            }

        baseline = modes["looped"]["seconds_median"]
        baseline_infer = modes["looped"]["infer_seconds_median"]
        speedups = {
            label: round(baseline / value["seconds_median"], 2)
            if value["seconds_median"] > 0
            else None
            for label, value in modes.items()
            if label != "looped"
        }
        infer_speedups = {
            label: round(baseline_infer / value["infer_seconds_median"], 2)
            if value["infer_seconds_median"] > 0
            else None
            for label, value in modes.items()
            if label != "looped"
        }
        reference = texts_by_mode["looped"]
        agreement = {
            label: round(
                sum(1 for a, b in zip(reference, texts, strict=True) if a == b)
                / len(reference),
                4,
            )
            for label, texts in texts_by_mode.items()
            if label != "looped"
        }
        # `looped_exact_width` is excluded from "fastest": it skips the bucket
        # padding the rec head was trained with and measurably changes the
        # decoded text (see text_agreement below), so it is a diagnostic, not a
        # configuration anyone may ship.
        bucketed = {
            label: value for label, value in modes.items() if value["bucketed"]
        }
        best = min(bucketed, key=lambda label: bucketed[label]["seconds_median"])
        results[script] = {
            "crops": len(crops),
            "pages_sampled": len(script_pages),
            "repeats": repeats,
            "modes": modes,
            "speedup_vs_looped": speedups,
            "infer_only_speedup_vs_looped": infer_speedups,
            "text_agreement_vs_looped": agreement,
            "fastest_bucketed_mode": best,
            "fastest_bucketed_speedup_vs_looped": round(
                baseline / bucketed[best]["seconds_median"], 2
            )
            if bucketed[best]["seconds_median"] > 0
            else None,
            "notes": {
                "looped": (
                    "batch size 1 with bucket padding — the baseline PLAN.md "
                    "invariant 4 forbids outside this tool"
                ),
                "looped_exact_width": (
                    "batch size 1 with no bucket padding (the naive PaddleOCR "
                    "loop). Reported for diagnosis only: it is out of "
                    "distribution for a head trained on right-padded crops and "
                    "its text_agreement below is the size of that effect"
                ),
                "text_agreement_vs_looped": (
                    "share of crops decoding identically to the looped "
                    "baseline. 1.0 for every batch_N is the assertion that "
                    "batching is output-neutral"
                ),
            },
        }
    return results


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def evaluate_gate(
    concurrency: dict[str, Any],
    *,
    target_concurrency: int,
    threshold_seconds: float = GATE_P95_SECONDS,
    advisory: bool = False,
) -> dict[str, Any]:
    """BUILD-BRIEF M0's exit criterion, as a machine-readable verdict.

    ``advisory`` marks a dry run, whose numbers describe a stub and must never
    be read as an answer to the throughput question.
    """
    key = str(target_concurrency)
    level = concurrency.get(key)
    if level is None:
        return {
            "criterion": f"p95 <= {threshold_seconds} s/page at concurrency "
            f"{target_concurrency}",
            "result": "NOT_MEASURED",
            "reason": f"no run at concurrency {target_concurrency}",
        }
    p95 = float(level["per_page_seconds"].get("p95", 0.0))
    failing = {
        profile: values["p95"]
        for profile, values in level["by_profile"].items()
        if float(values.get("p95", 0.0)) > threshold_seconds
    }
    return {
        "criterion": f"p95 <= {threshold_seconds} s/page at concurrency "
        f"{target_concurrency}",
        "target_concurrency": target_concurrency,
        "threshold_seconds": threshold_seconds,
        "p95_seconds": p95,
        "p50_seconds": float(level["per_page_seconds"].get("p50", 0.0)),
        "pages_per_minute": level["pages_per_minute"],
        "profiles_over_threshold": failing,
        "result": ("ADVISORY" if advisory else ("PASS" if p95 <= threshold_seconds else "FAIL")),
        "advisory": advisory,
        "note": (
            "dry run: timings describe a stub provider, not PP-OCRv5"
            if advisory
            else "measured with the committed PP-OCRv5 ONNX weights"
        ),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(
    *,
    settings: Settings,
    index: dict[str, Any],
    docs: Sequence[BenchDoc],
    concurrency: dict[str, Any],
    batching: dict[str, Any],
    dry_run: bool,
    target_concurrency: int,
    threshold_seconds: float,
    models_dir: Path,
    started_at: datetime,
) -> dict[str, Any]:
    """Assemble ``bench_results.json``."""
    from collections import Counter

    per_page: dict[str, dict[str, Any]] = {}
    for level, payload in concurrency.items():
        for timing in payload["_timings"]:
            row = per_page.setdefault(
                timing.doc_id,
                {
                    "doc_id": timing.doc_id,
                    "type": timing.doc_type,
                    "profile": timing.profile,
                    "script": timing.script,
                },
            )
            row[f"seconds_c{level}"] = round(timing.seconds, 4)
            row[f"lines_c{level}"] = timing.lines
            if timing.error is not None:
                row.setdefault("errors", {})[f"c{level}"] = timing.error

    public_concurrency = {
        level: {key: value for key, value in payload.items() if key != "_timings"}
        for level, payload in concurrency.items()
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": "tools/bench_ocr_cpu.py",
        "answers": "BUILD-BRIEF M0 (CPU throughput spike)",
        "mode": "dry_run" if dry_run else "measured",
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "machine": machine_info(),
        "runtime": runtime_versions(),
        "threads": thread_configuration(settings),
        "corpus": {
            "path": "corpus/",
            "seed": index.get("seed"),
            "generator_version": index.get("generator_version"),
            "documents_available": index.get("count"),
            "pages_sampled": len(docs),
            "by_profile": dict(sorted(Counter(doc.profile for doc in docs).items())),
            "by_type": dict(sorted(Counter(doc.doc_type for doc in docs).items())),
            "by_script": dict(sorted(Counter(doc.script for doc in docs).items())),
        },
        "models": _model_inventory(models_dir, dry_run=dry_run),
        "concurrency": public_concurrency,
        "batching": batching,
        "gate": evaluate_gate(
            public_concurrency,
            target_concurrency=target_concurrency,
            threshold_seconds=threshold_seconds,
            advisory=dry_run,
        ),
        "pages": [per_page[doc.doc_id] for doc in docs if doc.doc_id in per_page],
    }


def _model_inventory(models_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    """Which weight files were on disk, and how big they were."""
    if dry_run:
        return {"note": "dry run — no weights loaded"}
    ocr_dir = models_dir / "ocr"
    if not ocr_dir.exists():
        return {"note": f"{ocr_dir} does not exist"}
    return {
        # POSIX separators so the committed file diffs identically whichever OS
        # produced it.
        "ocr_dir": ocr_dir.relative_to(APP_PROJECT_ROOT).as_posix()
        if ocr_dir.is_relative_to(APP_PROJECT_ROOT)
        else ocr_dir.as_posix(),
        "files": {
            path.name: path.stat().st_size
            for path in sorted(ocr_dir.iterdir())
            if path.is_file()
        },
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="bench_ocr_cpu",
        description="M0 CPU throughput spike for PP-OCRv5 (BUILD-BRIEF M0).",
    )
    parser.add_argument(
        "--pages", type=int, default=100, help="corpus pages to benchmark (default 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=1729, help="sampling seed (default 1729)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        action="append",
        default=None,
        help="worker counts to measure; repeatable (default 1 and 2)",
    )
    parser.add_argument(
        "--target-concurrency",
        type=int,
        default=None,
        help="concurrency the gate is evaluated at (default CELERY_CONCURRENCY)",
    )
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        default=GATE_P95_SECONDS,
        help=f"gate p95 ceiling in seconds (default {GATE_P95_SECONDS})",
    )
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=8,
        help="pages per script used for the looped-vs-batched study (default 8)",
    )
    parser.add_argument(
        "--batch-repeats",
        type=int,
        default=3,
        help="timed repeats per batch size (default 3)",
    )
    parser.add_argument(
        "--profiles",
        action="append",
        default=None,
        help="restrict sampling to these degradation profiles; repeatable",
    )
    parser.add_argument(
        "--skip-batching",
        action="store_true",
        help="skip the looped-vs-batched study",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stub provider and synthetic pixels; needs no weights and no corpus decode",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="where to write the report (default bench_results.json at the repo root)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the summary without writing"
    )
    return parser


def run(args: argparse.Namespace, *, settings: Settings | None = None) -> dict[str, Any]:
    """Do the whole benchmark and return the report."""
    active = settings if settings is not None else get_settings()
    started_at = datetime.now(UTC)
    corpus_dir = active.corpus_dir
    index = load_index(corpus_dir)
    docs = sample_documents(
        index,
        count=args.pages,
        seed=args.seed,
        corpus_dir=corpus_dir,
        profiles=args.profiles,
    )

    if args.dry_run:
        pages = synthesise_pages(docs)
        harness = Harness(provider=DryRunProvider(), timer=PhaseTimer(), sink=CropSink())
    else:
        import cv2

        cv2.setNumThreads(active.ORT_INTRA_OP_THREADS)
        pages = load_pages(docs)
        harness = build_provider(active)
        # One untimed page so ONNX session construction and the first-call
        # allocator warm-up are not charged to page 1 of the measurement.
        harness.provider.ocr_page(pages[0].image, pages[0].doc.script)

    levels = args.concurrency or [1, 2]
    concurrency: dict[str, Any] = {}
    for workers in levels:
        concurrency[str(workers)] = measure_concurrency(
            harness.provider, harness.timer, pages, workers=workers
        )

    batching: dict[str, Any] = {}
    if not args.skip_batching and not args.dry_run:
        subset_by_script: dict[str, list[LoadedPage]] = {}
        for page in pages:
            bucket = subset_by_script.setdefault(page.doc.script, [])
            if len(bucket) < args.batch_pages:
                bucket.append(page)
        batching = measure_batching(
            harness,
            [page for bucket in subset_by_script.values() for page in bucket],
            repeats=args.batch_repeats,
        )
    elif args.dry_run:
        batching = {"note": "dry run — the batching study needs real recognition heads"}

    target = (
        args.target_concurrency
        if args.target_concurrency is not None
        else active.CELERY_CONCURRENCY
    )
    return build_report(
        settings=active,
        index=index,
        docs=docs,
        concurrency=concurrency,
        batching=batching,
        dry_run=args.dry_run,
        target_concurrency=target,
        threshold_seconds=args.threshold_seconds,
        models_dir=active.models_dir,
        started_at=started_at,
    )


def print_summary(report: dict[str, Any]) -> None:
    """Human-readable digest of the report."""
    machine = report["machine"]
    print(
        f"machine   {machine['cpu_model']} "
        f"({machine['cpu_cores_physical']} physical / "
        f"{machine['cpu_cores_logical']} logical cores, "
        f"{machine['ram_gib']} GiB RAM)"
    )
    print(
        f"threads   intra_op={report['threads']['ort_intra_op_threads']} "
        f"inter_op=1 batch={report['threads']['ocr_batch_size']}"
    )
    corpus = report["corpus"]
    print(f"corpus    {corpus['pages_sampled']} pages  {corpus['by_profile']}")
    for level, payload in sorted(report["concurrency"].items(), key=lambda kv: int(kv[0])):
        stats = payload["per_page_seconds"]
        print(
            f"c={level}       p50={stats.get('p50')}s p95={stats.get('p95')}s "
            f"max={stats.get('max')}s  {payload['pages_per_minute']} pages/min  "
            f"({payload['pages_measured']}/{payload['pages_attempted']} ok)"
        )
    for script, study in sorted(report["batching"].items()):
        if not isinstance(study, dict) or "speedup_vs_looped" not in study:
            continue
        print(
            f"batching  {script}: {study['crops']} crops  fastest bucketed="
            f"{study['fastest_bucketed_mode']} "
            f"({study['fastest_bucketed_speedup_vs_looped']}x)"
        )
        print(f"          speedup vs looped     {study['speedup_vs_looped']}")
        print(f"          infer-only speedup    {study['infer_only_speedup_vs_looped']}")
        print(f"          text agreement        {study['text_agreement_vs_looped']}")
    gate = report["gate"]
    print(
        f"GATE      {gate['result']}  {gate['criterion']}  "
        f"measured p95={gate.get('p95_seconds')}s"
    )
    if gate.get("profiles_over_threshold"):
        print(f"          over threshold by profile: {gate['profiles_over_threshold']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns 0 on PASS/ADVISORY, 1 on FAIL."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pages < 1:
        parser.error("--pages must be >= 1")
    if args.batch_repeats < 1:
        parser.error("--batch-repeats must be >= 1")

    report = run(args)
    print_summary(report)

    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote     {args.out}")

    if report["gate"]["result"] == "FAIL":
        print(
            "BLOCKING FINDING: record this in OPEN-QUESTIONS.md — CPU-first is "
            "not settled (BUILD-BRIEF M0).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
