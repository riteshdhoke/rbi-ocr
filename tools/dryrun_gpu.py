#!/usr/bin/env python3
"""dryrun_gpu.py — prove a GPU node is really using its GPU, before it matters.

Run this on any CUDA machine (a rented box, a free Kaggle/Colab session, the
office node itself) to answer six questions in one pass:

1. Does ONNX Runtime expose a CUDA execution provider at all?
2. Does a real session actually *place* itself on CUDA, or silently fall back?
3. Does CTranslate2 see a CUDA device, so translation is not stuck on CPU?
4. Does ``GPU_EXECUTION=require`` refuse to start when CUDA is missing?
5. Does ``gpu_node`` with no model server build a local-only provider?
6. How many pages per minute, GPU versus CPU, on the same pixels?

Question 4 is the one worth travelling for.  The usual way to lose the GPU is a
CPU-only ``onnxruntime`` wheel shadowing ``onnxruntime-gpu`` — both install the
same package name — and the symptom is a node that works perfectly and is ten
times too slow, which nobody notices until the queue backs up.  ``require`` is
supposed to turn that into a loud startup failure; this checks that it does,
rather than assuming it.

Usage::

    python tools/dryrun_gpu.py                 # full run, needs models/
    python tools/dryrun_gpu.py --pages 12      # more pages for a steadier number
    python tools/dryrun_gpu.py --skip-throughput

Exit status is 0 when every check that *can* pass on this machine passed.  A
machine with no GPU is not a failure — it reports NO GPU and still checks that
the guards behave.  Only a guard that misbehaves fails the run.

This tool does NOT handle:
  * installing anything.  It reports what is installed and what that implies;
  * the SLM.  ``gpu_node`` puts the *local* vision and translation models on
    CUDA — the language model is a separate serving stack (deploy guide 7.3);
  * proving data residency.  Running this on a rented box tells you nothing
    about where complainant data may go; see the warning it prints.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `app/__init__.py` applies the thread caps at import time, and PLAN.md
# invariant 5 requires that to happen before any numeric library loads.  This
# is therefore the only module-level import of a project or numeric package in
# this file: numpy, cv2, onnxruntime and ctranslate2 are all imported inside
# the functions that use them, well after this line.  An import sorter that
# moved a numeric import above it would silently break the invariant, so there
# is nothing here for one to reorder.
import app  # noqa: F401

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
INFO = "INFO"


@dataclass
class Check:
    """One question, its verdict, and the evidence behind it."""

    name: str
    status: str
    detail: str


@dataclass
class Results:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))
        marker = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip ", INFO: " info "}[status]
        print(f"[{marker}] {name}\n         {detail}", flush=True)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]


# --------------------------------------------------------------------------- #
# 1-3: what the runtimes can see
# --------------------------------------------------------------------------- #


def check_onnxruntime(results: Results) -> bool:
    """Whether ORT exposes a CUDA provider, and which wheel is installed."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        results.add("onnxruntime installed", FAIL, f"import failed: {exc}")
        return False

    # The shipped graphs carry unused initialisers and ORT says so, once per
    # initialiser, per session. That is a hundred lines of noise around a
    # six-line verdict table, and this tool exists to be read.
    ort.set_default_logger_severity(3)  # 3 = ERROR

    available = list(ort.get_available_providers())
    cuda = [p for p in available if p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")]
    results.add(
        "onnxruntime version",
        INFO,
        f"{ort.__version__}; providers = {', '.join(available)}",
    )
    if cuda:
        results.add(
            "ORT exposes a CUDA provider",
            PASS,
            f"{', '.join(cuda)} present — the onnxruntime-gpu wheel is the one in use",
        )
        return True
    results.add(
        "ORT exposes a CUDA provider",
        SKIP,
        "CPU-only onnxruntime. On a GPU node this is the failure to fix: "
        "`pip uninstall -y onnxruntime && pip install onnxruntime-gpu` "
        "(both install the same package name, so the CPU wheel shadows the GPU one)",
    )
    return False


def check_session_placement(results: Results, cuda_available: bool) -> None:
    """A provider being *available* is not the same as a session *using* it."""
    if not cuda_available:
        results.add("A real session lands on CUDA", SKIP, "no CUDA provider to place on")
        return
    det = REPO_ROOT / "models" / "ocr" / "det.onnx"
    if not det.is_file():
        results.add("A real session lands on CUDA", SKIP, f"{det} is absent")
        return

    from app.providers.local import LocalProvider
    from app.settings import Settings

    # Build the session the way the service does, through the provider's own
    # factory, rather than by asking ORT for CUDA directly. Those are not the
    # same test: a hardcoded ["CUDAExecutionProvider", "CPU"] list succeeded on
    # a machine where the real code path fell back to CPU, because the real
    # list also named TensorRT and ORT abandons the whole list when any entry
    # fails to load. A check that does not exercise the shipping code path can
    # report green while the thing it is checking is broken.
    settings = Settings(
        _env_file=None, PROVIDER_PROFILE="gpu_node", GPU_EXECUTION="require"
    )
    provider = LocalProvider(settings)
    requested = provider._execution_providers()
    session = provider._default_session_factory(det)
    placed = list(session.get_providers())
    results.add(
        "Providers the service actually asks for",
        INFO,
        ", ".join(requested),
    )
    if placed and placed[0] != "CPUExecutionProvider":
        results.add(
            "A real session lands on CUDA",
            PASS,
            f"detection model placed on {placed[0]} (full order: {', '.join(placed)})",
        )
    else:
        results.add(
            "A real session lands on CUDA",
            FAIL,
            f"asked for CUDA, got {placed}. ORT advertised a CUDA provider and then "
            "could not load it, so it fell back to CPU — read the [E:onnxruntime] "
            "line above for the library it could not find.\n"
            f"         {_cuda_mismatch_hint()}",
        )


def _cuda_mismatch_hint() -> str:
    """Turn "it fell back to CPU" into the command that fixes it.

    The usual cause is a wheel built against a different CUDA major version
    from the one installed: ``onnxruntime-gpu`` moved its default PyPI build
    from CUDA 12 to CUDA 13, so a current wheel on a CUDA 12 box fails to
    dlopen ``libcublasLt.so.13`` and quietly runs on CPU.  Naming the two
    versions side by side is the whole diagnosis.
    """
    present = sorted(
        {
            path.name.rsplit(".so.", 1)[-1].split(".")[0]
            for root in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu")
            for path in Path(root).glob("libcublasLt.so.*")
            if ".so." in path.name
        }
    )
    installed = ", ".join(present) if present else "none found"
    return (
        f"cuBLAS major version(s) on this machine: {installed}. If that does not "
        "match the .so the error names, the wheel and the CUDA runtime disagree:\n"
        "         install a matching build, e.g. "
        '`pip install "onnxruntime-gpu==1.22.*"` for CUDA 12,\n'
        "         then re-run. `nvidia-smi` reports the driver's CUDA ceiling."
    )


def check_ctranslate2(results: Results) -> None:
    """Translation is a separate runtime and fails over independently."""
    try:
        import ctranslate2
    except ImportError as exc:
        results.add("CTranslate2 sees a CUDA device", SKIP, f"not installed: {exc}")
        return
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001 - build without CUDA symbols
        results.add(
            "CTranslate2 sees a CUDA device",
            SKIP,
            f"this build reports no CUDA support ({type(exc).__name__}); translation runs on CPU",
        )
        return
    if count > 0:
        results.add("CTranslate2 sees a CUDA device", PASS, f"{count} device(s)")
    else:
        results.add(
            "CTranslate2 sees a CUDA device",
            SKIP,
            "0 devices — translation falls back to CPU (int8). "
            "The pipeline still works; only MT is slower",
        )


# --------------------------------------------------------------------------- #
# 4-5: the guards
# --------------------------------------------------------------------------- #


def check_require_guard(results: Results) -> None:
    """``GPU_EXECUTION=require`` must fail loudly when CUDA is absent.

    Exercised by hiding the CUDA providers from the provider's own lookup
    rather than by uninstalling a wheel, so the check is the same on a GPU
    machine and a CPU one — and so running it costs nothing.
    """
    import onnxruntime as ort

    from app.contracts.errors import ConfigurationError
    from app.providers.local import LocalProvider
    from app.settings import Settings

    original = ort.get_available_providers
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[assignment]
    try:
        settings = Settings(_env_file=None, GPU_EXECUTION="require")
        provider = LocalProvider(settings)
        try:
            provider._execution_providers()
        except ConfigurationError as exc:
            results.add(
                "GPU_EXECUTION=require refuses a CPU-only runtime",
                PASS,
                f"raised ConfigurationError: {str(exc)[:90]}...",
            )
        else:
            results.add(
                "GPU_EXECUTION=require refuses a CPU-only runtime",
                FAIL,
                "it started anyway. This is the guard against a CPU-only wheel "
                "shadowing onnxruntime-gpu; without it the node is silently 10x slow",
            )
    finally:
        ort.get_available_providers = original  # type: ignore[assignment]

    # And the other direction: `auto` must degrade quietly, not raise.
    settings = Settings(_env_file=None, GPU_EXECUTION="auto", PROVIDER_PROFILE="cpu_node")
    if settings.gpu_requested:
        results.add(
            "GPU_EXECUTION=auto stays off the GPU on cpu_node",
            FAIL,
            "auto requested a GPU on a CPU profile; the quality gate was "
            "calibrated on CPU numbers and must keep producing them",
        )
    else:
        results.add(
            "GPU_EXECUTION=auto stays off the GPU on cpu_node",
            PASS,
            "auto is profile-driven, not 'use whatever wheel is installed'",
        )


def check_local_only_gpu_node(results: Results) -> None:
    """The office topology: gpu_node, no model server, nothing leaves the box."""
    from app.providers.factory import close_provider, get_provider
    from app.providers.local import LocalProvider
    from app.settings import Settings

    settings = Settings(
        _env_file=None,
        PROVIDER_PROFILE="gpu_node",
        SYNTHETIC_ONLY=False,
        HF_TOKEN=None,
    )
    try:
        provider = get_provider(settings, log_banner=False)
    except Exception as exc:  # noqa: BLE001
        results.add(
            "gpu_node with no model server starts",
            FAIL,
            f"{type(exc).__name__}: {exc}",
        )
        return
    try:
        if isinstance(provider, LocalProvider):
            results.add(
                "gpu_node with no model server starts",
                PASS,
                "built LocalProvider alone — no socket is opened, field extraction "
                "falls back to the deterministic validators",
            )
        else:
            results.add(
                "gpu_node with no model server starts",
                FAIL,
                f"expected LocalProvider, got {type(provider).__name__}",
            )
    finally:
        close_provider(provider)

    # And the residency guard, which is the reason that default is safe.
    breach = Settings(
        _env_file=None,
        PROVIDER_PROFILE="gpu_node",
        SYNTHETIC_ONLY=False,
        HF_BASE_URL="https://api.openai.com/v1",
        HF_TOKEN="sk-not-a-real-key",
    )
    if breach.residency_violation_reason:
        results.add(
            "gpu_node refuses an off-premises endpoint",
            PASS,
            "residency gate fires before any provider is constructed",
        )
    else:
        results.add(
            "gpu_node refuses an off-premises endpoint",
            FAIL,
            "an air-gapped profile accepted a public endpoint — real complainant "
            "text could leave the deployment",
        )


# --------------------------------------------------------------------------- #
# 6: is it actually faster
# --------------------------------------------------------------------------- #


def measure(pages: list[Any], mode: str) -> tuple[float, int]:
    """Recognise ``pages`` with GPU_EXECUTION=*mode*; return (seconds, lines)."""
    from app.providers.local import LocalProvider
    from app.settings import Settings

    settings = Settings(
        _env_file=None, PROVIDER_PROFILE="gpu_node", GPU_EXECUTION=mode
    )
    provider = LocalProvider(settings)
    provider.ocr_page(pages[0], "Deva")  # warm the session and the head cache
    started = time.perf_counter()
    lines = sum(len(provider.ocr_page(page, "Deva")) for page in pages)
    return time.perf_counter() - started, lines


def check_throughput(results: Results, page_count: int, cuda_available: bool) -> None:
    """Pages per minute, GPU against CPU, on the same pixels."""
    try:
        import cv2
    except ImportError as exc:
        results.add("Throughput", SKIP, f"opencv missing: {exc}")
        return

    corpus = REPO_ROOT / "corpus"
    # Every clean variant, not just ``_v0_``: a corpus generated with
    # ``--variants 50`` holds hundreds of pages, and matching only v0 silently
    # capped ``--pages 16`` at the nine document types that have one. A timing
    # run that quietly measures half what was asked for is a bad measurement.
    files = sorted(corpus.glob("*/*_clean.jpg")) + sorted(corpus.glob("*/*_clean.png"))
    files = sorted(files)[:page_count]
    if not files:
        results.add("Throughput", SKIP, f"no corpus pages under {corpus}")
        return
    if not (REPO_ROOT / "models" / "ocr" / "det.onnx").is_file():
        results.add("Throughput", SKIP, "models/ocr/det.onnx is absent")
        return

    pages = [cv2.imread(str(path)) for path in files]
    pages = [p for p in pages if p is not None]

    cpu_seconds, lines = measure(pages, "never")
    cpu_rate = len(pages) / cpu_seconds * 60.0
    results.add(
        "Throughput (CPU)",
        INFO,
        f"{len(pages)} pages, {lines} lines in {cpu_seconds:.1f}s = {cpu_rate:.0f} pages/min",
    )

    if not cuda_available:
        results.add("Throughput (GPU)", SKIP, "no CUDA provider on this machine")
        return

    gpu_seconds, gpu_lines = measure(pages, "require")
    gpu_rate = len(pages) / gpu_seconds * 60.0
    speedup = cpu_seconds / gpu_seconds if gpu_seconds else 0.0
    headline = (
        f"{len(pages)} pages, {gpu_lines} lines in {gpu_seconds:.1f}s = "
        f"{gpu_rate:.0f} pages/min  ({speedup:.2f}x CPU)"
    )
    # A ratio this close to 1.00 is the signature of both arms running on the
    # same device, not of a GPU that happens to tie. It is how a silent CPU
    # fallback looks from the outside, so it must not read as a pass.
    if 0.90 <= speedup <= 1.10:
        results.add(
            "Throughput (GPU)",
            FAIL,
            f"{headline}\n"
            "         Suspiciously equal. The usual cause is that the GPU arm "
            "fell back to CPU — scroll up for an\n"
            "         [E:onnxruntime] line about a provider library that failed "
            "to load. Two arms on the same\n"
            "         device produce exactly this ratio.",
        )
    elif speedup < 0.90:
        results.add(
            "Throughput (GPU)",
            INFO,
            f"{headline}  — slower than CPU. Small batches on a big GPU are "
            "often launch-latency bound; try --pages 32 before concluding anything",
        )
    else:
        results.add("Throughput (GPU)", PASS, headline)
    if gpu_lines != lines:
        results.add(
            "GPU and CPU read the same text",
            FAIL,
            f"CPU produced {lines} lines, GPU produced {gpu_lines} — "
            "the accelerator changed the output, which it must never do",
        )
    else:
        results.add(
            "GPU and CPU read the same text",
            PASS,
            f"both produced {lines} lines",
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove a GPU node is really using its GPU."
    )
    parser.add_argument(
        "--pages", type=int, default=8, help="Corpus pages to time (default: 8)."
    )
    parser.add_argument(
        "--skip-throughput", action="store_true", help="Run the checks, skip the timing."
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("GPU DRY RUN")
    print("=" * 74)
    print(f"repo   : {REPO_ROOT}")
    print(f"python : {sys.version.split()[0]}")
    import numpy  # imported here, after app applied the thread caps

    print(f"numpy  : {numpy.__version__}")
    print()

    results = Results()
    cuda = check_onnxruntime(results)
    check_session_placement(results, cuda)
    check_ctranslate2(results)
    check_require_guard(results)
    check_local_only_gpu_node(results)
    if not args.skip_throughput:
        check_throughput(results, args.pages, cuda)

    print()
    print("-" * 74)
    counts = {
        status: sum(1 for c in results.checks if c.status == status)
        for status in (PASS, FAIL, SKIP, INFO)
    }
    print(
        f"{counts[PASS]} passed, {counts[FAIL]} failed, "
        f"{counts[SKIP]} skipped, {counts[INFO]} informational"
    )
    if results.failed:
        print()
        print("FAILED CHECKS — do not deploy to a GPU node until these are green:")
        for check in results.failed:
            print(f"  * {check.name}: {check.detail}")
        print()
        print("GPU DRY RUN FAILED")
        return 1

    if not cuda:
        print()
        print(
            "No CUDA on this machine, so the GPU-specific checks were skipped. "
            "The guards behaved correctly, which is what a CPU box can prove."
        )
    print()
    print(
        "NOTE ON DATA: if this is a rented or shared machine, use the synthetic "
        "corpus only.\n      A green run here says the GPU path works. It says "
        "nothing about where\n      complainant data may lawfully go — that is "
        "what PROVIDER_PROFILE is for."
    )
    print()
    print("GPU DRY RUN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
