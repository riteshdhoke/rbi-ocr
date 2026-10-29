"""Dev-only downloader for the approved CPU model artifacts.

Run this on a **networked** machine, then copy ``models/`` to the air-gapped
node (``docs/AIR-GAPPED-DEPLOY.md``).  Nothing in the service ever calls it, and
the service never downloads anything at runtime (PLAN.md air-gap rule).

Scope is exactly PLAN.md decision D1 — small CPU models only:

============  ==========================================  ==============
artifact      what                                        approx. size
============  ==========================================  ==============
``lid``       fastText ``lid.176.ftz`` language ID         0.9 MB
``ocr``       PP-OCRv5 detection + recognition ONNX        10-25 MB
``mt``        IndicTrans2 indic-en dist-200M, CT2 INT8     ~200 MB
============  ==========================================  ==============

**SLM and VLM weights are never downloaded.**  Anything at or above 0.9B
parameters is reached over the HF Inference API under the ``laptop`` profile and
must arrive on a production node as an approved offline artefact.

Three deliberate design choices:

*Discovery, not guessed paths.*  OPEN-QUESTIONS Q1 chose pre-converted ONNX
(RapidOCR's Apache-2.0 redistribution) over installing the full Paddle
toolchain, but the file layout inside that repository is not something this
brief can assert.  So the tool **lists the repository and matches patterns**,
prints what it found, and writes the result into ``models/PROVENANCE.json``.
A missing Devanagari recognition head is *reported*, not papered over — the
loader in :mod:`app.providers.local` exists for it either way and raises a typed
``ModelArtifactMissing`` naming the gap.

*A procurement trail, not a licence gate.*  ``models/PROVENANCE.json`` records
source URL, resolved revision, SHA-256 and declared licence for every file.  One
artefact — fastText's ``lid.176.ftz`` — is published under **CC-BY-SA-3.0**,
which is outside the PLAN.md invariant 12 allowlist.  That allowlist governs
*code dependencies* (``pip-licenses`` in CI) and CC-BY-SA is neither GPL/AGPL
nor non-commercial, so the tool downloads it by default and flags it loudly,
exactly as OPEN-QUESTIONS D6 handled the LGPL fribidi DLL.  ``--strict-licence``
refuses anything outside the allowlist for anyone who wants the harder rule.

*Truststore first.*  The office proxy intercepts TLS with a CA in the Windows
certificate store rather than in ``certifi`` (OPEN-QUESTIONS D7).  TLS
verification is never disabled, here or anywhere.

This module does NOT handle:
  * downloading or converting any SLM/VLM;
  * training the page classifier or the script-ID model — neither has an
    approved artefact, and ``models/classifier`` / ``models/script`` stay empty
    until WS-E produces one;
  * placing files on the air-gapped node (copy ``models/`` yourself);
  * verifying that a downloaded ONNX graph is *the model it claims to be*.  It
    records the SHA-256 so a second run, or a second machine, can prove the
    bytes match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported after the sys.path insert above so this tool runs from any cwd.
from app.providers.base import inject_truststore
from app.providers.local import (
    DET_MODEL_FILENAME,
    LID_MODEL_FILENAME,
    LID_SUBDIR,
    MT_CT2_DIRNAME,
    MT_CT2_MODEL_FILENAME,
    MT_SPM_SRC_FILENAME,
    MT_SPM_TGT_FILENAME,
    MT_SUBDIR,
    OCR_SUBDIR,
    REC_DICT_TEMPLATE,
    REC_MODEL_TEMPLATE,
)

__all__ = [
    "ARTIFACT_GROUPS",
    "LICENCE_ALLOWLIST",
    "ArtifactRecord",
    "fetch_lid",
    "fetch_mt",
    "fetch_ocr",
    "main",
    "write_provenance",
]

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

#: PLAN.md invariant 12.  Governs code dependencies; model weights are recorded
#: against it and flagged when they fall outside, not silently accepted.
LICENCE_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0", "ISC", "PSF"}
)

ARTIFACT_GROUPS: Final[tuple[str, ...]] = ("lid", "ocr", "mt")

#: fastText's own distribution point for the compressed LID model.  The 917 KB
#: ``.ftz``, never the 126 MB ``.bin`` (BUILD-BRIEF sec. 1).
LID_URL: Final[str] = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)
LID_LICENCE: Final[str] = "CC-BY-SA-3.0"

#: OPEN-QUESTIONS Q1, superseding option (b): the official PaddlePaddle org on
#: Hugging Face publishes PP-OCRv5 mobile det/rec as per-artifact ONNX repos
#: under Apache-2.0 — including the Devanagari head A-7 was hunting for
#: (verified 2026-08-07: PP-OCRv5_mobile_det_onnx, en_/devanagari_/ta_/te_
#: PP-OCRv5_mobile_rec_onnx, each holding inference.onnx + inference.yml with
#: the embedded character dictionary).  "official" selects that table; passing
#: any concrete repo id via --ocr-repo falls back to the legacy
#: single-repo listing flow below.
OCR_REPO_DEFAULT: Final[str] = "official"
OCR_LICENCE: Final[str] = "Apache-2.0"

#: role -> (repo_id, filename); dict roles derive from their rec repo's yml.
OCR_OFFICIAL_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "det": ("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "inference.onnx"),
    "rec_latn": ("PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx", "inference.onnx"),
    "rec_deva": (
        "PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx",
        "inference.onnx",
    ),
    "dict_latn": ("PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx", "inference.yml"),
    "dict_deva": (
        "PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx",
        "inference.yml",
    ),
}

#: Ordered file-name patterns, most preferred first.  Matched case-insensitively
#: against the repository listing rather than hard-coded as paths, because the
#: layout of that repository is not something this brief can assert.
OCR_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "det": (
        r"pp-?ocrv5.*mobile.*det.*\.onnx$",
        r"pp-?ocrv5.*det.*\.onnx$",
        r"pp-?ocrv4.*mobile.*det.*\.onnx$",
    ),
    "rec_latn": (
        r"(latin|en).*pp-?ocrv5.*rec.*\.onnx$",
        r"pp-?ocrv5.*rec.*mobile.*\.onnx$",
        r"pp-?ocrv5.*rec.*\.onnx$",
    ),
    "rec_deva": (
        r"devanagari.*rec.*\.onnx$",
        r"deva.*rec.*\.onnx$",
    ),
    "dict_latn": (
        r"(latin|en)_dict\.txt$",
        r"ppocr(v5)?_keys.*\.txt$",
        r".*dict.*\.txt$",
    ),
    "dict_deva": (
        r"devanagari_dict\.txt$",
        r"deva.*dict.*\.txt$",
    ),
}

#: Which resolved patterns must be present for the OCR group to be "complete".
#: ``rec_deva`` is deliberately *not* required: PLAN.md wants a Devanagari head
#: and OPEN-QUESTIONS Q1 asks whether a pre-converted one exists at all.  The
#: tool reports the answer instead of assuming it.
OCR_REQUIRED: Final[tuple[str, ...]] = ("det", "rec_latn", "dict_latn")

#: IndicTrans2 distilled 200M, Indic to English.  MIT-licensed (AI4Bharat).
#: Default is a pre-converted CTranslate2 build (C62's anticipated override):
#: CT2 registers no converter for ``IndicTransConfig``, so local conversion of
#: the ai4bharat checkpoint is impossible without their fairseq toolchain.
#: The adalat-ai repo (MIT) ships the official RoPE dist-200M line already in
#: CT2 layout; INT8 happens at load time (``compute_type="int8"``).  Verified
#: 2026-08-07 with a Hindi→English smoke test on this machine.  Passing the
#: ai4bharat repo id via --mt-repo selects the legacy download+convert flow
#: (which requires torch + transformers<5 and a CT2 with IndicTrans support).
MT_REPO_DEFAULT: Final[str] = "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
MT_LICENCE: Final[str] = "MIT"

#: SentencePiece models shipped beside the checkpoint; names as upstream.
MT_TOKENIZER_CANDIDATES: Final[tuple[tuple[str, str], ...]] = (
    ("model.SRC", MT_SPM_SRC_FILENAME),
    ("model.TGT", MT_SPM_TGT_FILENAME),
    ("dict.SRC.json", "dict.SRC.json"),
    ("dict.TGT.json", "dict.TGT.json"),
)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class ArtifactRecord:
    """One resolved artefact, for ``models/PROVENANCE.json``."""

    group: str
    role: str
    destination: str
    source: str
    licence: str
    revision: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    status: str = "ok"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "role": self.role,
            "destination": self.destination,
            "source": self.source,
            "licence": self.licence,
            "licence_in_code_allowlist": self.licence in LICENCE_ALLOWLIST,
            "revision": self.revision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "note": self.note,
        }


@dataclass
class FetchReport:
    """Everything one invocation did, printed and persisted."""

    records: list[ArtifactRecord] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, record: ArtifactRecord) -> ArtifactRecord:
        self.records.append(record)
        if record.status not in {"ok", "dry-run"}:
            self.problems.append(f"{record.group}/{record.role}: {record.note}")
        return record


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(message, flush=True)


def _licence_ok(licence: str, *, strict: bool) -> bool:
    return (licence in LICENCE_ALLOWLIST) or not strict


def _first_match(names: Sequence[str], patterns: Iterable[str]) -> str | None:
    """First repository entry matching the first pattern that matches anything."""
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        matches = sorted(name for name in names if compiled.search(name))
        if matches:
            return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Group: LID
# --------------------------------------------------------------------------- #


def fetch_lid(
    models_dir: Path, report: FetchReport, *, dry_run: bool, strict: bool
) -> None:
    """Download ``lid.176.ftz`` from fastText's distribution point."""
    destination = models_dir / LID_SUBDIR / LID_MODEL_FILENAME
    record = ArtifactRecord(
        group="lid",
        role="fasttext-lid-176",
        destination=str(destination.relative_to(models_dir)),
        source=LID_URL,
        licence=LID_LICENCE,
        note=(
            "PROCUREMENT NOTE: fastText publishes its language-ID models under "
            "CC-BY-SA-3.0, which is outside the PLAN.md invariant 12 allowlist. "
            "That allowlist governs code dependencies; CC-BY-SA is neither "
            "GPL/AGPL nor non-commercial. Attribution and share-alike apply to "
            "the model file itself. Owner to confirm."
        ),
    )
    if not _licence_ok(LID_LICENCE, strict=strict):
        record.status = "refused"
        record.note = f"--strict-licence refused {LID_LICENCE}"
        report.add(record)
        return
    if dry_run:
        record.status = "dry-run"
        report.add(record)
        return

    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[lid] downloading {LID_URL}")
    with (
        httpx.Client(follow_redirects=True, timeout=120.0) as client,
        client.stream("GET", LID_URL) as response,
    ):
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 256):
                handle.write(chunk)
    record.sha256 = _sha256(destination)
    record.size_bytes = destination.stat().st_size
    report.add(record)
    _log(f"[lid] wrote {destination} ({record.size_bytes} bytes)")


# --------------------------------------------------------------------------- #
# Group: OCR
# --------------------------------------------------------------------------- #


def _extract_character_dict(yml_path: Path) -> list[str]:
    """Pull the recognition character list out of a PaddleX ``inference.yml``.

    Searches the parsed document for a ``character_dict`` list wherever it is
    nested; raises ``ValueError`` naming the file when none is found, rather
    than writing an empty dictionary the recogniser would silently mis-decode
    with.  Does NOT handle non-YAML dict formats (plain ``*_keys.txt`` files
    come through the legacy path untouched).
    """
    import yaml

    def _find(node: object) -> list[str] | None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "character_dict" and isinstance(value, list):
                    return [str(ch) for ch in value]
                found = _find(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find(item)
                if found is not None:
                    return found
        return None

    parsed = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    chars = _find(parsed)
    if not chars:
        raise ValueError(f"no character_dict found in {yml_path}")
    return chars


def _fetch_ocr_official(
    api: Any,
    targets_by_role: dict[str, Path],
    models_dir: Path,
    report: FetchReport,
    *,
    dry_run: bool,
    hf_home: Path | None,
) -> None:
    """Download the official per-artifact PaddlePaddle ONNX repos (A-7 path a)."""
    from huggingface_hub import hf_hub_download

    revisions: dict[str, str | None] = {}
    for role, (repo, filename) in OCR_OFFICIAL_SOURCES.items():
        destination = targets_by_role[role]
        if repo not in revisions:
            revisions[repo] = _repo_revision(api, repo)
        record = ArtifactRecord(
            group="ocr",
            role=role,
            destination=str(destination.relative_to(models_dir)),
            source=f"{repo}:{filename}",
            licence=OCR_LICENCE,
            revision=revisions[repo],
        )
        if dry_run:
            record.status = "dry-run"
            record.note = f"would download {repo}:{filename}"
            report.add(record)
            _log(f"[ocr] {role}: would download {repo}:{filename}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        _log(f"[ocr] {role}: downloading {repo}:{filename}")
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revisions[repo],
                local_dir=str(destination.parent / "_staging" / role),
                cache_dir=str(hf_home) if hf_home is not None else None,
            )
            if filename.endswith(".yml"):
                chars = _extract_character_dict(Path(local))
                destination.write_text(
                    "\n".join(chars) + "\n", encoding="utf-8"
                )
                record.note = f"{len(chars)} characters extracted from {filename}"
            else:
                shutil.copyfile(local, destination)
        except Exception as exc:  # noqa: BLE001 - each artefact reports independently
            record.status = "missing" if role in OCR_REQUIRED else "unavailable"
            record.note = f"{type(exc).__name__}: {exc}"
            report.add(record)
            _log(f"[ocr] {role}: FAILED — {record.note}")
            continue
        record.sha256 = _sha256(destination)
        record.size_bytes = destination.stat().st_size
        report.add(record)
        _log(f"[ocr] wrote {destination}")

    staging = models_dir / OCR_SUBDIR / "_staging"
    if staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)


def fetch_ocr(
    models_dir: Path,
    report: FetchReport,
    *,
    repo_id: str,
    dry_run: bool,
    strict: bool,
    hf_home: Path | None,
) -> None:
    """Resolve and download PP-OCR detection/recognition ONNX + dictionaries.

    Discovers the file layout by listing the repository, so a reorganisation
    upstream produces a clear "not found" rather than a 404 on a guessed path.
    """
    from huggingface_hub import HfApi, hf_hub_download

    if not _licence_ok(OCR_LICENCE, strict=strict):
        report.add(
            ArtifactRecord(
                group="ocr",
                role="all",
                destination=OCR_SUBDIR,
                source=repo_id,
                licence=OCR_LICENCE,
                status="refused",
                note=f"--strict-licence refused {OCR_LICENCE}",
            )
        )
        return

    api = HfApi()

    targets_by_role = {
        "det": models_dir / OCR_SUBDIR / DET_MODEL_FILENAME,
        "rec_latn": models_dir / OCR_SUBDIR / REC_MODEL_TEMPLATE.format(script="latn"),
        "rec_deva": models_dir / OCR_SUBDIR / REC_MODEL_TEMPLATE.format(script="deva"),
        "dict_latn": models_dir / OCR_SUBDIR / REC_DICT_TEMPLATE.format(script="latn"),
        "dict_deva": models_dir / OCR_SUBDIR / REC_DICT_TEMPLATE.format(script="deva"),
    }
    if repo_id == "official":
        _fetch_ocr_official(
            api,
            targets_by_role,
            models_dir,
            report,
            dry_run=dry_run,
            hf_home=hf_home,
        )
        return

    _log(f"[ocr] listing {repo_id}")
    names = list(api.list_repo_files(repo_id))
    revision = _repo_revision(api, repo_id)

    targets = {
        "det": models_dir / OCR_SUBDIR / DET_MODEL_FILENAME,
        "rec_latn": models_dir
        / OCR_SUBDIR
        / REC_MODEL_TEMPLATE.format(script="latn"),
        "rec_deva": models_dir
        / OCR_SUBDIR
        / REC_MODEL_TEMPLATE.format(script="deva"),
        "dict_latn": models_dir
        / OCR_SUBDIR
        / REC_DICT_TEMPLATE.format(script="latn"),
        "dict_deva": models_dir
        / OCR_SUBDIR
        / REC_DICT_TEMPLATE.format(script="deva"),
    }

    for role, patterns in OCR_PATTERNS.items():
        remote = _first_match(names, patterns)
        destination = targets[role]
        record = ArtifactRecord(
            group="ocr",
            role=role,
            destination=str(destination.relative_to(models_dir)),
            source=f"{repo_id}:{remote}" if remote else repo_id,
            licence=OCR_LICENCE,
            revision=revision,
        )
        if remote is None:
            record.status = "missing" if role in OCR_REQUIRED else "unavailable"
            record.note = (
                f"no file in {repo_id} matched {list(patterns)}. "
                + (
                    "OPEN-QUESTIONS Q1: a pre-converted Devanagari PP-OCR "
                    "recognition head could not be located. The loader in "
                    "app/providers/local.py expects "
                    f"{destination.name} and raises ModelArtifactMissing until "
                    "one is placed there — either convert "
                    "devanagari_PP-OCRv3/v5_mobile_rec with paddle2onnx on a "
                    "machine that can take the paddlepaddle wheel, or record "
                    "the gap and run Devanagari pages through Tier 2."
                    if role == "rec_deva"
                    else "Try --ocr-repo with an alternative Apache-2.0 "
                    "redistribution."
                )
            )
            report.add(record)
            _log(f"[ocr] {role}: NOT FOUND — {record.note}")
            continue
        if dry_run:
            record.status = "dry-run"
            record.note = f"would download {remote}"
            report.add(record)
            _log(f"[ocr] {role}: would download {remote}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        _log(f"[ocr] downloading {remote}")
        local = hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            revision=revision,
            local_dir=str(destination.parent / "_staging"),
            cache_dir=str(hf_home) if hf_home is not None else None,
        )
        shutil.copyfile(local, destination)
        record.sha256 = _sha256(destination)
        record.size_bytes = destination.stat().st_size
        report.add(record)
        _log(f"[ocr] wrote {destination}")

    staging = models_dir / OCR_SUBDIR / "_staging"
    if staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)


def _repo_revision(api: Any, repo_id: str) -> str | None:
    """Pin the resolved commit so PROVENANCE.json is reproducible."""
    try:
        return str(api.model_info(repo_id).sha)
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


# --------------------------------------------------------------------------- #
# Group: MT
# --------------------------------------------------------------------------- #


def fetch_mt(
    models_dir: Path,
    report: FetchReport,
    *,
    repo_id: str,
    dry_run: bool,
    strict: bool,
    hf_home: Path | None,
) -> None:
    """Download IndicTrans2 dist-200M and convert it to CTranslate2 INT8.

    The published checkpoint is fp32 (~800 MB on the wire); the INT8
    CTranslate2 output is ~200 MB and the intermediate is deleted.  Run this on
    a machine with ~1.5 GB of free space.
    """
    destination = models_dir / MT_SUBDIR / MT_CT2_DIRNAME
    record = ArtifactRecord(
        group="mt",
        role="indictrans2-indic-en-dist-200M-ct2-int8",
        destination=str(destination.relative_to(models_dir)),
        source=repo_id,
        licence=MT_LICENCE,
        note=(
            "Converted locally with ct2-transformers-converter "
            "--quantization int8; the fp32 intermediate is discarded."
        ),
    )
    if not _licence_ok(MT_LICENCE, strict=strict):
        record.status = "refused"
        record.note = f"--strict-licence refused {MT_LICENCE}"
        report.add(record)
        return
    if dry_run:
        record.status = "dry-run"
        report.add(record)
        _log(f"[mt] would download and convert {repo_id}")
        return

    from huggingface_hub import HfApi, list_repo_files, snapshot_download

    api = HfApi()
    record.revision = _repo_revision(api, repo_id)

    # Pre-converted branch: repo already carries a CTranslate2 model dir.
    remote_files = list(list_repo_files(repo_id))
    ct2_bins = [f for f in remote_files if f.endswith("ctranslate2_model/model.bin")]
    if ct2_bins:
        ct2_prefix = ct2_bins[0].rsplit("/", 1)[0]
        record.note = (
            f"pre-converted CTranslate2 build from {repo_id}:{ct2_prefix}; "
            "INT8 applied at load time (compute_type='int8')"
        )
        with tempfile.TemporaryDirectory(prefix="indictrans2-ct2-") as staging:
            _log(f"[mt] downloading pre-converted {repo_id}")
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=record.revision,
                    local_dir=staging,
                    cache_dir=str(hf_home) if hf_home is not None else None,
                )
            )
            source_dir = snapshot / Path(ct2_prefix)
            destination.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copyfile(item, destination / item.name)
                    _log(f"[mt] copied {item.name}")
            vocab_dir = source_dir / "vocab"
            for spm_name in (MT_SPM_SRC_FILENAME, MT_SPM_TGT_FILENAME):
                spm_file = vocab_dir / spm_name
                if spm_file.is_file():
                    shutil.copyfile(spm_file, destination / spm_name)
                    _log(f"[mt] copied vocab/{spm_name}")
                else:
                    report.problems.append(
                        f"mt: {spm_name} absent from {repo_id}:{ct2_prefix}/vocab; "
                        "app/providers/local.py expects it beside model.bin"
                    )
        model_bin = destination / MT_CT2_MODEL_FILENAME
        if model_bin.is_file():
            record.sha256 = _sha256(model_bin)
            record.size_bytes = sum(
                p.stat().st_size for p in destination.rglob("*") if p.is_file()
            )
        else:
            record.status = "failed"
            record.note = f"pre-converted repo produced no {MT_CT2_MODEL_FILENAME}"
        report.add(record)
        _log(f"[mt] wrote {destination}")
        return

    with tempfile.TemporaryDirectory(prefix="indictrans2-") as staging:
        _log(f"[mt] downloading {repo_id} (fp32 checkpoint, ~800 MB)")
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=record.revision,
                local_dir=staging,
                cache_dir=str(hf_home) if hf_home is not None else None,
            )
        )
        destination.mkdir(parents=True, exist_ok=True)
        _log("[mt] converting to CTranslate2 INT8")
        # Fixed argv, no shell: nothing here is user-controlled.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ctranslate2.converters.transformers",
                "--model",
                str(snapshot),
                "--output_dir",
                str(destination),
                "--quantization",
                "int8",
                "--force",
                "--trust_remote_code",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            record.status = "failed"
            record.note = (
                "ct2-transformers-converter exited "
                f"{completed.returncode}: {completed.stderr.strip()[:800]}"
            )
            report.add(record)
            _log(f"[mt] CONVERSION FAILED — {record.note}")
            return

        for remote_name, local_name in MT_TOKENIZER_CANDIDATES:
            source = snapshot / remote_name
            if source.is_file():
                shutil.copyfile(source, destination / local_name)
                _log(f"[mt] copied tokenizer file {remote_name} -> {local_name}")
            else:
                report.problems.append(
                    f"mt: tokenizer file {remote_name} absent from {repo_id}; "
                    "app/providers/local.py expects it at "
                    f"{destination / local_name}"
                )

    model_bin = destination / MT_CT2_MODEL_FILENAME
    if model_bin.is_file():
        record.sha256 = _sha256(model_bin)
        record.size_bytes = sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        )
    else:
        record.status = "failed"
        record.note = f"converter produced no {MT_CT2_MODEL_FILENAME}"
    report.add(record)
    _log(f"[mt] wrote {destination}")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def write_provenance(models_dir: Path, report: FetchReport) -> Path:
    """Write ``models/PROVENANCE.json`` — the audit trail for a deployment."""
    path = models_dir / "PROVENANCE.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "tools/fetch_models.py",
        "licence_allowlist": sorted(LICENCE_ALLOWLIST),
        "artifacts": [record.to_dict() for record in report.records],
        "problems": report.problems,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_models.py",
        description=(
            "Download the approved CPU model artifacts into models/. "
            "Dev machines only; the service never downloads at runtime."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="destination directory (default: ./models)",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=ARTIFACT_GROUPS,
        help="fetch only these groups; repeatable (default: all)",
    )
    parser.add_argument(
        "--ocr-repo",
        default=OCR_REPO_DEFAULT,
        help=f"HF repo holding pre-converted PP-OCR ONNX (default: {OCR_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--mt-repo",
        default=MT_REPO_DEFAULT,
        help=f"HF repo holding IndicTrans2 dist-200M (default: {MT_REPO_DEFAULT})",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help=(
            "cache directory for huggingface_hub; keeps downloads out of "
            "~/.cache/huggingface (PLAN.md WS-I item 3)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and report what would be fetched, download nothing",
    )
    parser.add_argument(
        "--strict-licence",
        action="store_true",
        help=(
            "refuse any artefact whose declared licence is outside the "
            "PLAN.md invariant 12 allowlist (this rejects lid.176.ftz)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)
    groups = tuple(args.only) if args.only else ARTIFACT_GROUPS
    models_dir = Path(args.models_dir).resolve()

    if inject_truststore():
        _log("[tls] using the OS certificate store (corporate CA honoured)")
    else:
        _log("[tls] truststore not installed; using certifi defaults")
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(Path(args.hf_home).resolve())
        _log(f"[hf] HF_HOME={os.environ['HF_HOME']}")

    models_dir.mkdir(parents=True, exist_ok=True)
    report = FetchReport()

    if "lid" in groups:
        fetch_lid(
            models_dir, report, dry_run=args.dry_run, strict=args.strict_licence
        )
    if "ocr" in groups:
        fetch_ocr(
            models_dir,
            report,
            repo_id=args.ocr_repo,
            dry_run=args.dry_run,
            strict=args.strict_licence,
            hf_home=args.hf_home,
        )
    if "mt" in groups:
        fetch_mt(
            models_dir,
            report,
            repo_id=args.mt_repo,
            dry_run=args.dry_run,
            strict=args.strict_licence,
            hf_home=args.hf_home,
        )

    provenance = write_provenance(models_dir, report)
    _log(f"\n[provenance] {provenance}")
    for record in report.records:
        flag = "" if record.licence in LICENCE_ALLOWLIST else "  <-- OUT OF GATE"
        _log(
            f"  {record.group:<4} {record.role:<40} {record.status:<12} "
            f"{record.licence}{flag}"
        )
    if report.problems:
        _log("\n[problems]")
        for problem in report.problems:
            _log(f"  - {problem}")
        _log(
            "\nRecord unresolved items in OPEN-QUESTIONS.md (Q1) before the "
            "air-gapped build."
        )
    return 1 if any(r.status in {"missing", "failed"} for r in report.records) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
