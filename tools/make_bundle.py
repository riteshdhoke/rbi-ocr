#!/usr/bin/env python3
"""make_bundle.py — build the two office hand-off zips (PLAN.md WS-I item 7).

Produces, in ``dist/``, both stamped with the short git SHA of ``HEAD`` so a
receiver can prove the pair belongs together:

``dist/rbi-ocr-code-<shortsha>.zip``
    Everything a developer needs to stand the service up: ``app/``, ``tools/``
    (including the corpus *generator*, the fonts and ``fribidi-0.dll``),
    ``tests/``, ``docs/``, ``design/``, ``alembic/``, ``requirements*.txt``,
    ``constraints.txt``, ``.env.example``, ``docker-compose.yml`` and
    ``models/README.md``.  **No** corpus images, **no** model weights, **no**
    ``.venv``, **no** ``.env``.  Expected well under 15 MB.

``dist/rbi-ocr-models-corpus-<shortsha>.zip``
    Everything the receiving node cannot produce for itself:

    * ``models/`` — every weight the service loads, taken from the **worktree**
      because ``.gitignore`` excludes them and no commit lists them.  Their
      guarantee is the per-file SHA-256 in the manifest rather than a git
      object id, which is the guarantee that matters anyway: it says the bytes
      unzipped are the bytes that were measured.  ~837 MB, dominated by one
      808 MB fp32 translation model.
    * ``corpus/`` exactly as committed: 295 generated documents, one
      ``.gt.json`` ground-truth sidecar each, ``index.json``, and the
      ``adversarial/`` set.  Roughly 70 MB, mostly incompressible JPEG.

    Members at or above :data:`LARGE_MEMBER_BYTES` are streamed rather than
    buffered and deflated at :data:`FAST_COMPRESS_LEVEL`; see those constants
    for why.

Both archives embed a ``BUNDLE_MANIFEST.json`` (git SHA, build time, file
count, per-file SHA-256, document versions).  Neither archive has a top-level
folder, so ``tar -xf`` in an empty directory reproduces the project root that
``docs/AIR-GAPPED-DEPLOY.md`` section 2 draws.

Usage
-----

::

    .venv\\Scripts\\python.exe tools/make_bundle.py            # build both, then verify
    .venv\\Scripts\\python.exe tools/make_bundle.py --verify   # verify what is already in dist/
    .venv\\Scripts\\python.exe tools/make_bundle.py --only code
    .venv\\Scripts\\python.exe tools/make_bundle.py --tracked-only

Exit status is 0 only if every assertion passed.  Anything else prints
``BUNDLE BUILD FAILED`` with the reasons and exits 1.

Where the file list comes from
------------------------------

``git ls-files`` is the source of truth, in two invocations:

* ``git ls-files --cached --exclude-standard`` — everything committed;
* ``git ls-files --others --exclude-standard`` — everything *not yet* committed
  that git would offer to add, i.e. untracked **and not ignored**.

The second list is what makes this correct rather than merely literal.  PLAN.md
WS-I item 7 requires ``constraints.txt`` in the code bundle, and item 8 requires
``docs/AIR-GAPPED-DEPLOY.md``, ``docs/API.md``, ``docs/FUNCTIONAL-SPEC.md`` and
``docs/TUNING.md`` — all of which exist in the working tree and none of which is
committed yet.  A tracked-only bundle would ship without the deployment guide it
is supposed to be deployed with.  ``--exclude-standard`` is what keeps the safety
property: a file git ignores can never enter either archive, so ``.env``,
``models/`` weights, ``.venv/`` and ``dist/`` itself are excluded *structurally*
rather than by a blocklist that someone has to remember to update.  Every
untracked file that is included is printed by name and recorded in the manifest
under ``untracked_included``, and ``--tracked-only`` builds the strict
committed-files-only variant for anyone who wants it.

The blocklist exists anyway (:func:`check_forbidden`), because "structurally
impossible" is a claim worth testing rather than trusting.

This module does NOT handle:
  * downloading or placing model weights — see ``tools/fetch_models.py`` and
    ``docs/AIR-GAPPED-DEPLOY.md`` section 5;
  * generating the corpus — see ``tools/generate_test_corpus.py``;
  * signing, encrypting or transporting the archives;
  * committing anything.  It reads the worktree and writes only into ``dist/``;
  * proving that an *ignored* file is safe to ignore.  It proves the opposite
    direction: nothing ignored is in the archive.
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
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

GENERATOR_VERSION = "1.0.0"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"


def bundle_manifest_alias(bundle: str) -> str:
    """Second, bundle-specific name for the manifest.

    ``docs/AIR-GAPPED-DEPLOY.md`` section 2 tells the operator to extract **both**
    zips into the **same** directory, so a single ``BUNDLE_MANIFEST.json`` at the
    root of each one collides: whichever is unzipped second silently overwrites
    the other and the operator is left unable to verify one of the two bundles
    from the extracted tree.  Each archive therefore carries the manifest twice,
    byte-identically — once under the required name and once under this one.
    Both copies are checked by ``--verify``.
    """
    return f"BUNDLE_MANIFEST-{bundle}.json"

CODE_BUNDLE_PREFIX = "rbi-ocr-code-"
DEV_CODE_BUNDLE_PREFIX = "rbi-ocr-code-dev-"
ASSETS_BUNDLE_PREFIX = "rbi-ocr-models-corpus-"
CORPUS_ROOT = "corpus/"
MODELS_ROOT = "models/"

#: Working papers: real history, and noise to someone being handed the system
#: to run.  ``--exclude-internal`` drops them and names the archive
#: ``rbi-ocr-code-dev-<sha>.zip`` so the two variants cannot be confused.
#:
#: These are *deliberations*, not specifications.  A developer needs to know
#: what the service does (docs/FUNCTIONAL-SPEC.md), how to deploy it
#: (docs/AIR-GAPPED-DEPLOY.md), how to call it (docs/API.md) and what to turn
#: (docs/TUNING.md).  Being handed sixteen files arguing about decisions that
#: were already taken invites them to reopen settled questions, or to mistake
#: a rejected option for the shipped one.
#:
#: Two of these — ``PLAN.md`` and ``OPEN-QUESTIONS.md`` — *are* in
#: :data:`REQUIRED_CODE_PATHS`, so :func:`check_required` drops them from the
#: required list when this exclusion is in force.  A file cannot coherently be
#: both a deliberation the bundle omits and a document the bundle must carry.
#: Everything else stays required, so a future addition here that removed the
#: deployment guide would still fail the build.
#: The full bundle remains the default, because the reasoning is worth keeping.
INTERNAL_WORKING_PAPERS: Final[tuple[str, ...]] = (
    "docs/decisions-",  # 16 per-workstream decision logs
    "docs/bakeoff/",  # SLM bake-off working data
    "OPEN-QUESTIONS.md",  # the running Q&A log
    "PLAN.md",  # the build plan and its owner decisions
    "bench_results.json",  # raw throughput measurements
)
# Note what is deliberately *not* excluded: the whole of docs/eval/. A developer
# should have the measured accuracy even when it is unflattering.
#
# The dated snapshots (docs/eval/REPORT-<date>-<tag>.md) *were* excluded here as
# "named after internal IDs, meaningless to someone who never saw Q12". They are
# kept now because the current REPORT.md cites them by name for things it cannot
# itself measure — most importantly the SLM path, which the 2026-08-17 run does
# not exercise at all, so its only figures live in REPORT-2026-08-08-slmlive.md.
# Excluding a document that the included one defers to leaves the developer
# holding a reference they cannot follow, on the one number most likely to be
# quoted at them. The naming convention is what carries the meaning:
#
#     docs/eval/REPORT.md              always the current evaluation
#     docs/eval/REPORT-<date>-<tag>.md a superseded run, kept for comparison
#
# Anything that renames the current report must move the old one to a dated
# name in the same commit, or the bundle ships stale numbers as current.


def is_internal_working_paper(path: str) -> bool:
    """Whether ``path`` is a deliberation rather than a specification."""
    return path.startswith(INTERNAL_WORKING_PAPERS)

#: A member at or above this size is streamed from disk rather than held in
#: memory, and compressed at :data:`FAST_COMPRESS_LEVEL`.  The assets bundle
#: carries an 808 MB fp32 translation model; buffering it the way the code and
#: corpus bundles buffer their members would cost well over a gigabyte of RAM
#: for no benefit, because nothing scans a weights file line by line.
LARGE_MEMBER_BYTES = 50 * 1024 * 1024

#: Deflate level for those large binaries.  Measured on ``model.bin``: level 1
#: reaches 67.4% of the original in ~23 s, level 9 reaches 65.2% in ~162 s.
#: Seven times the work to save 18 MB on a bundle that is already ~600 MB is
#: not a trade worth making on a machine that has to build this repeatedly.
FAST_COMPRESS_LEVEL = 1

#: Everything else — source, docs, corpus images, vocabularies — still gets the
#: best deflate the standard library offers.
BEST_COMPRESS_LEVEL = 9

#: The owner's ceiling for the code bundle.  Measured: ~9.6 MB of files, ~5 MB zipped.
DEFAULT_MAX_CODE_BYTES = 15 * 1024 * 1024

#: A Hugging Face access token.  Real ones are ``hf_`` + 34 alphanumerics; 20 is a
#: deliberately loose floor so a truncated or future-format token still trips it.
#: Matched against raw bytes, so it fires inside a binary too.
TOKEN_PATTERN = re.compile(rb"hf_[A-Za-z0-9]{20,}")

#: ``.env.example`` and the docs are allowed to *say* ``hf_xxx``; that never matches
#: the pattern above (``xxx`` is 3 characters).  There is no allowlist and no
#: exception: one match anywhere is a hard failure.

# --------------------------------------------------------------------------- #
# What must be in the code bundle, and what must never be
# --------------------------------------------------------------------------- #

#: Absent any one of these, the bundle is not the thing PLAN.md WS-I describes.
REQUIRED_CODE_PATHS: tuple[str, ...] = (
    # The four owner-standard documents (PLAN.md WS-I items 2, 4, 6, 8)
    "docs/AIR-GAPPED-DEPLOY.md",
    "docs/API.md",
    "docs/FUNCTIONAL-SPEC.md",
    "docs/TUNING.md",
    # Reproducible install
    "requirements.txt",
    "requirements-dev.txt",
    "constraints.txt",
    ".env.example",
    # Corpus generator + everything it needs to shape Devanagari (decision D6)
    "tools/generate_test_corpus.py",
    "tools/assets/fribidi-0.dll",
    "tools/assets/fonts/NotoSansDevanagari-Regular.ttf",
    # Calibration artifacts the deploy guide tells the operator to copy/compare
    "docs/calibration/fusion.json",
    "docs/calibration/quality_gate.json",
    "docs/schema/result_envelope.schema.json",
    # The models directory placeholder, so section 2's unzip layout is true
    "models/README.md",
    # Service, tests, orientation
    "app/pipeline.py",
    "app/api/app.py",
    "tests/conftest.py",
    "README.md",
    "PLAN.md",
    "OPEN-QUESTIONS.md",
    "docker-compose.yml",
    "docs/eval/REPORT.md",
)

#: ``prefix -> minimum member count``.  Catches a whole directory going missing,
#: which a per-file list would not: one representative font proves nothing about
#: the other seven.
REQUIRED_CODE_TREES: tuple[tuple[str, int], ...] = (
    ("app/", 50),
    ("tests/", 30),
    ("tools/", 15),
    ("tools/assets/fonts/", 8),
    ("docs/", 30),
    ("design/", 4),
    ("alembic/", 3),
)

#: Path prefixes that may never appear in the code bundle.
FORBIDDEN_CODE_PREFIXES: tuple[str, ...] = (
    CORPUS_ROOT,
    "dist/",
    ".venv/",
    "venv/",
    ".git/",
    "data/runtime/",
    ".cache/",
    "htmlcov/",
)

#: ``models/`` is the interesting one: the README must be in, every weight must be out.
MODELS_ALLOWED: frozenset[str] = frozenset({"models/README.md"})

#: Model-weight and serialized-tensor extensions.  None of these is a legitimate
#: member of either bundle.
WEIGHT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".onnx", ".bin", ".pt", ".pth", ".safetensors", ".ftz", ".gguf", ".ggml",
        ".npz", ".npy", ".h5", ".ckpt", ".pb", ".tflite", ".msgpack", ".caffemodel",
        ".params", ".pdiparams", ".pdmodel", ".mlmodel", ".engine", ".plan",
    }
)

#: Private-key and credential file extensions.
SECRET_SUFFIXES: frozenset[str] = frozenset(
    {".pem", ".key", ".pfx", ".p12", ".jks", ".keystore", ".ppk", ".asc", ".gpg"}
)

#: Credential files that carry no extension at all.
SECRET_BASENAMES: frozenset[str] = frozenset(
    {
        ".env", ".env.local", ".env.production", ".env.prod",
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        ".netrc", "_netrc", ".npmrc", ".pypirc", "credentials", "secrets.json",
    }
)

#: Suffixes read as text for the "N text / M binary" split in the report.  The
#: token scan itself runs over every byte of every member regardless.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".ini", ".cfg", ".toml",
        ".html", ".css", ".js", ".j2", ".mako", ".example", ".gitignore", ".sql",
        ".csv", ".bat", ".ps1", ".sh", ".env", ".in", ".rst",
    }
)

# --------------------------------------------------------------------------- #
# Corpus expectations
# --------------------------------------------------------------------------- #

EXPECTED_CORPUS_DOCUMENTS = 295
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


class BundleError(RuntimeError):
    """A bundle could not be built or did not survive its own checks.

    Typed rather than bare so a caller (or a test) can distinguish "this tool
    refused" from "this tool crashed".
    """


@dataclass(slots=True)
class GitInfo:
    """Identity of the worktree a bundle was built from."""

    sha: str
    short_sha: str
    branch: str
    commit_time: str
    dirty: bool
    dirty_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "sha": self.sha,
            "short_sha": self.short_sha,
            "branch": self.branch,
            "commit_time_utc": self.commit_time,
            "worktree_dirty": self.dirty,
            "worktree_dirty_path_count": len(self.dirty_paths),
        }


@dataclass(slots=True)
class Problem:
    """One failed assertion.  Collected, not raised, so one run reports all of them."""

    check: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.check}] {self.detail}"


@dataclass(slots=True)
class Report:
    """Accumulates problems and notes for one build or verification."""

    problems: list[Problem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, check: str, detail: str) -> None:
        self.problems.append(Problem(check, detail))

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def ok(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------- #
# git plumbing
# --------------------------------------------------------------------------- #


def _git(repo_root: Path, *args: str) -> str:
    """Run a git command in *repo_root* and return stdout, or raise ``BundleError``."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:  # git not installed / not on PATH
        raise BundleError(
            "git was not found on PATH. This tool reads the file list from git; "
            "install git or build the bundles on a machine that has it."
        ) from exc
    if completed.returncode != 0:
        raise BundleError(
            f"`git {' '.join(args)}` failed with exit {completed.returncode}: "
            f"{completed.stderr.strip() or '(no stderr)'}"
        )
    return completed.stdout


def _git_paths(repo_root: Path, *args: str) -> list[str]:
    """``git ls-files -z`` wrapper.

    ``-z`` matters: without it git quotes and escapes any path containing a
    non-ASCII byte, and the caller would then look for a file whose name has
    literal backslashes in it.
    """
    raw = _git(repo_root, "ls-files", "-z", *args)
    return sorted(p for p in raw.split("\0") if p)


def read_git_info(repo_root: Path) -> GitInfo:
    """Identify HEAD and whether the worktree matches it."""
    sha = _git(repo_root, "rev-parse", "HEAD").strip()
    short_sha = _git(repo_root, "rev-parse", "--short", "HEAD").strip()
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    commit_time = _git(repo_root, "show", "-s", "--format=%cI", "HEAD").strip()
    status = _git(repo_root, "status", "--porcelain=v1")
    dirty_paths = [line[3:] for line in status.splitlines() if line.strip()]
    return GitInfo(
        sha=sha,
        short_sha=short_sha,
        branch=branch,
        commit_time=commit_time,
        dirty=bool(dirty_paths),
        dirty_paths=dirty_paths,
    )


# --------------------------------------------------------------------------- #
# File selection
# --------------------------------------------------------------------------- #


def collect_code_files(repo_root: Path, *, tracked_only: bool = False) -> tuple[list[str], list[str]]:
    """Return ``(all_paths, untracked_paths)`` for the code bundle.

    Both lists are repo-relative POSIX paths, sorted.  ``corpus/`` is removed
    here — it is the entire content of the other bundle.
    """
    tracked = [p for p in _git_paths(repo_root, "--cached", "--exclude-standard") if not p.startswith(CORPUS_ROOT)]
    untracked: list[str] = []
    if not tracked_only:
        untracked = [
            p for p in _git_paths(repo_root, "--others", "--exclude-standard") if not p.startswith(CORPUS_ROOT)
        ]
    combined = sorted(set(tracked) | set(untracked))
    return combined, untracked


def collect_corpus_files(repo_root: Path) -> list[str]:
    """Return the committed ``corpus/`` file list.

    Deliberately ``--cached`` only.  The corpus bundle is specified as "exactly
    as committed"; picking up an untracked file under ``corpus/`` would ship a
    corpus that no commit describes.
    """
    return _git_paths(repo_root, "--cached", "--exclude-standard", "--", CORPUS_ROOT)


def corpus_worktree_is_clean(repo_root: Path) -> list[str]:
    """Return the corpus paths that differ from HEAD (empty list means clean)."""
    status = _git(repo_root, "status", "--porcelain=v1", "--", CORPUS_ROOT)
    return [line[3:] for line in status.splitlines() if line.strip()]


def collect_model_files(repo_root: Path) -> list[str]:
    """Return every file under ``models/``, read from the **worktree**.

    Not from git, and that is the whole point: ``.gitignore`` excludes
    ``models/*`` because weights are fetched rather than committed, so there is
    no commit that lists them.  The bundle's authority is therefore the
    manifest's per-file SHA-256 rather than a git object id — which is the
    right guarantee anyway, since what the receiver needs to know is that the
    bytes they unzip are the bytes that were measured.

    Returns POSIX-relative paths, sorted, so the archive order is stable.
    """
    root = repo_root / MODELS_ROOT
    if not root.is_dir():
        return []
    found = [
        path.relative_to(repo_root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return sorted(found)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_forbidden(paths: Iterable[str], report: Report, *, bundle: str) -> None:
    """Assert nothing dangerous is in the member list.

    Everything here is *already* impossible if ``--exclude-standard`` did its
    job.  It is checked anyway because the cost of a wrong ``.gitignore`` is a
    token or 800 MB of weights in a zip that leaves the building.
    """
    for path in paths:
        pure = PurePosixPath(path)
        name = pure.name
        suffix = pure.suffix.lower()
        parts = pure.parts

        if name in SECRET_BASENAMES:
            report.fail("forbidden-file", f"{bundle}: credential file present: {path}")
        if suffix in SECRET_SUFFIXES:
            report.fail("forbidden-file", f"{bundle}: private-key/credential extension: {path}")
        # Weights are the *point* of the assets bundle, and forbidden anywhere
        # else. Scoped by prefix rather than by bundle alone, so a stray .onnx
        # outside models/ still fails even in the bundle that carries weights.
        if suffix in WEIGHT_SUFFIXES and not (
            bundle == "assets" and path.startswith(MODELS_ROOT)
        ):
            report.fail("forbidden-file", f"{bundle}: model-weight extension: {path}")
        if ".venv" in parts or "venv" in parts or "site-packages" in parts:
            report.fail("forbidden-file", f"{bundle}: virtual environment content: {path}")
        if "__pycache__" in parts or suffix in {".pyc", ".pyo", ".pyd"}:
            report.fail("forbidden-file", f"{bundle}: compiled python artifact: {path}")

        if bundle == "code":
            for prefix in FORBIDDEN_CODE_PREFIXES:
                if path.startswith(prefix):
                    report.fail("forbidden-file", f"code: path under {prefix!r}: {path}")
            if path.startswith("models/") and path not in MODELS_ALLOWED:
                report.fail(
                    "forbidden-file",
                    f"code: only {sorted(MODELS_ALLOWED)} may come from models/, found: {path}",
                )
        elif not path.startswith((CORPUS_ROOT, MODELS_ROOT)):
            report.fail(
                "forbidden-file",
                f"assets: member outside corpus/ and models/: {path}",
            )


def check_required(
    paths: Sequence[str], report: Report, *, exclude_internal: bool = False
) -> None:
    """Assert the code bundle contains everything the deploy guide promises.

    ``exclude_internal`` drops the working papers from the *required* list too.
    A file cannot coherently be both "a deliberation this bundle omits" and
    "a document this bundle must contain", and the developer bundle is defined
    by the first of those. Everything else stays required, so a list that
    removed the deployment guide would still fail here.
    """
    present = set(paths)
    for required in REQUIRED_CODE_PATHS:
        if exclude_internal and is_internal_working_paper(required):
            continue
        if required not in present:
            report.fail("missing-required", f"code bundle is missing {required}")
    for prefix, minimum in REQUIRED_CODE_TREES:
        count = sum(1 for p in paths if p.startswith(prefix))
        if count < minimum:
            report.fail(
                "missing-required",
                f"code bundle has {count} file(s) under {prefix!r}, expected at least {minimum}",
            )


def scan_for_tokens(members: dict[str, bytes], report: Report, *, bundle: str) -> dict[str, int]:
    """Search every byte of every member for a Hugging Face token pattern.

    Returns ``{"text": n, "binary": m, "bytes": total, "matches": k}``.  A hit is
    a hard failure and is reported by **path and byte offset only** — the matched
    text is never printed, logged or written anywhere, because printing a leaked
    secret to a build log is the same leak with more copies.  Callers on the
    build path must refuse to write the archive when ``matches`` is non-zero.
    """
    stats = {"text": 0, "binary": 0, "bytes": 0, "matches": 0}
    for path, blob in sorted(members.items()):
        stats["bytes"] += len(blob)
        pure = PurePosixPath(path)
        is_text = pure.suffix.lower() in TEXT_SUFFIXES or pure.name in {".gitignore", ".env.example"}
        stats["text" if is_text else "binary"] += 1
        match = TOKEN_PATTERN.search(blob)
        if match is not None:
            stats["matches"] += 1
            report.fail(
                "token-scan",
                f"{bundle}: Hugging Face token pattern found in {path} at byte offset "
                f"{match.start()} (length {match.end() - match.start()}); value withheld",
            )
    return stats


def check_corpus_shape(paths: Sequence[str], read: Callable[[str], bytes], report: Report) -> dict[str, object]:
    """Assert the corpus bundle is the committed 295-document set.

    *read* is a callable ``(path) -> bytes`` so this works identically over the
    worktree during a build and over the archive during ``--verify``.
    """
    images = [
        p
        for p in paths
        if PurePosixPath(p).suffix.lower() in IMAGE_SUFFIXES and not p.startswith("corpus/adversarial/")
    ]
    ground_truth = [p for p in paths if p.endswith(".gt.json")]
    adversarial = [p for p in paths if p.startswith("corpus/adversarial/")]

    if len(images) != EXPECTED_CORPUS_DOCUMENTS:
        report.fail("corpus-shape", f"expected {EXPECTED_CORPUS_DOCUMENTS} document images, found {len(images)}")
    if len(ground_truth) != EXPECTED_CORPUS_DOCUMENTS:
        report.fail(
            "corpus-shape", f"expected {EXPECTED_CORPUS_DOCUMENTS} .gt.json sidecars, found {len(ground_truth)}"
        )
    if "corpus/index.json" not in paths:
        report.fail("corpus-shape", "corpus/index.json is missing")
    if "corpus/adversarial/MANIFEST.json" not in adversarial:
        report.fail("corpus-shape", "corpus/adversarial/MANIFEST.json is missing")
    if len(adversarial) < 2:
        report.fail("corpus-shape", f"corpus/adversarial/ has {len(adversarial)} file(s), expected the broken set")

    summary: dict[str, object] = {
        "images": len(images),
        "ground_truth_sidecars": len(ground_truth),
        "adversarial_files": len(adversarial),
    }

    if "corpus/index.json" not in paths:
        return summary

    try:
        index = json.loads(read("corpus/index.json").decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        report.fail("corpus-shape", f"corpus/index.json is not readable JSON: {exc}")
        return summary

    summary["index_count"] = index.get("count")
    summary["generator_version"] = index.get("generator_version")
    summary["seed"] = index.get("seed")
    summary["synthetic"] = index.get("synthetic")

    documents = index.get("documents", [])
    if index.get("count") != EXPECTED_CORPUS_DOCUMENTS or len(documents) != EXPECTED_CORPUS_DOCUMENTS:
        report.fail(
            "corpus-shape",
            f"index.json declares count={index.get('count')} over {len(documents)} entries, "
            f"expected {EXPECTED_CORPUS_DOCUMENTS} of each",
        )
    if index.get("synthetic") is not True:
        report.fail("corpus-shape", "index.json does not declare synthetic: true (residency gate, invariant 6)")

    # Every document the index names must actually be in the archive.  index.json was
    # written on Windows and stores backslash-separated relative paths, so normalise.
    present = set(paths)
    missing: list[str] = []
    for entry in documents:
        for key in ("image", "gt"):
            rel = str(entry.get(key, "")).replace("\\", "/")
            if not rel:
                continue
            if f"{CORPUS_ROOT}{rel}" not in present:
                missing.append(f"{entry.get('doc_id', '?')}:{key}={rel}")
    if missing:
        head = ", ".join(missing[:5])
        report.fail(
            "corpus-shape",
            f"{len(missing)} file(s) named by index.json are absent from the bundle (e.g. {head})",
        )
    summary["index_references_resolved"] = len(documents) * 2 - len(missing)
    return summary


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _abort_on_token(stats: dict[str, int], bundle: str) -> None:
    """Refuse to write an archive that contains a token pattern.

    The report already names the file and offset.  Writing the zip first and
    failing afterwards would leave a zip on disk with a secret in it, which is
    precisely the outcome the check exists to prevent.
    """
    if stats.get("matches"):
        raise BundleError(
            f"{stats['matches']} file(s) in the {bundle} bundle match the Hugging Face token "
            "pattern. No archive was written. Remove the secret from the worktree (or add the "
            "file to .gitignore) and re-run; the offending paths are listed above."
        )


def _scan_manifest_block(stats: dict[str, int]) -> dict[str, object]:
    return {
        "text_files_scanned": stats["text"],
        "binary_files_scanned": stats["binary"],
        "bytes_scanned": stats["bytes"],
        "token_pattern": "hf_[A-Za-z0-9]{20,}",
        "token_matches": stats["matches"],
    }


def _rollup_digest(digests: dict[str, str]) -> str:
    """One digest over the whole member set: ``sha256("path\\0sha\\n" * n)``.

    Order-independent because the input is sorted; sensitive to a renamed file,
    which a digest over concatenated content would not be.
    """
    roll = hashlib.sha256()
    for path in sorted(digests):
        roll.update(path.encode("utf-8"))
        roll.update(b"\0")
        roll.update(digests[path].encode("ascii"))
        roll.update(b"\n")
    return roll.hexdigest()


def collect_doc_versions(members: dict[str, bytes]) -> dict[str, object]:
    """Version-stamp the documents and the two versioned JSON artifacts.

    Markdown carries no version field, so a document's "version" is the SHA-256
    of its bytes plus its size — enough to answer "is the guide in my zip the
    guide you measured against?", which is the question a receiver actually has.
    """
    tracked_docs = (
        "docs/AIR-GAPPED-DEPLOY.md",
        "docs/API.md",
        "docs/FUNCTIONAL-SPEC.md",
        "docs/TUNING.md",
        "docs/eval/REPORT.md",
        "OPEN-QUESTIONS.md",
        "PLAN.md",
        "README.md",
        "constraints.txt",
        "requirements.txt",
        "requirements-dev.txt",
    )
    versions: dict[str, object] = {}
    for path in tracked_docs:
        blob = members.get(path)
        if blob is None:
            continue
        versions[path] = {
            "sha256": _sha256(blob),
            "bytes": len(blob),
            "lines": blob.count(b"\n") + (0 if blob.endswith(b"\n") or not blob else 1),
        }

    # Two artifacts do carry an explicit version; surface it so a human can read it.
    report_json = members.get("docs/eval/report.json")
    if report_json is not None:
        try:
            versions["eval_harness_version"] = json.loads(report_json.decode("utf-8")).get("harness_version")
        except (ValueError, UnicodeDecodeError):
            versions["eval_harness_version"] = "unreadable"
    index_json = members.get("corpus/index.json")
    if index_json is not None:
        try:
            index = json.loads(index_json.decode("utf-8"))
            versions["corpus_generator_version"] = index.get("generator_version")
            versions["corpus_seed"] = index.get("seed")
        except (ValueError, UnicodeDecodeError):
            versions["corpus_generator_version"] = "unreadable"
    return versions


def build_manifest(
    *,
    bundle: str,
    git_info: GitInfo,
    members: dict[str, bytes],
    digests: dict[str, str],
    untracked: Sequence[str],
    tracked_only: bool,
    extra: dict[str, object] | None = None,
    bundle_name: str | None = None,
    streamed_bytes: dict[str, int] | None = None,
) -> dict[str, object]:
    """Assemble ``BUNDLE_MANIFEST.json`` for one archive.

    ``streamed_bytes`` carries the sizes of members that were never buffered,
    so ``total_uncompressed_bytes`` counts them.  Without it an 808 MB weights
    file would be reported as zero bytes, and the one number a receiver uses to
    sanity-check a transfer would be the one number that is wrong.
    """
    source_of_truth = "git ls-files --cached --exclude-standard"
    if not tracked_only and bundle == "code":
        source_of_truth += " + git ls-files --others --exclude-standard"
    if bundle == "assets":
        source_of_truth = (
            "git ls-files --cached --exclude-standard -- corpus/  +  "
            "worktree scan of models/ (weights are gitignored)"
        )

    manifest: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "bundle": bundle,
        "bundle_name": bundle_name or f"rbi-ocr-{bundle}-{git_info.short_sha}.zip",
        "project": "RBI complaint document intelligence service",
        "generator": "tools/make_bundle.py",
        "generator_version": GENERATOR_VERSION,
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info.as_dict(),
        "source_of_truth": source_of_truth,
        "tracked_only": tracked_only,
        "file_count": len(members),
        "total_uncompressed_bytes": sum(
            (streamed_bytes or {}).get(name, len(blob))
            for name, blob in members.items()
        ),
        "untracked_included": list(untracked),
        "excluded": {
            "corpus": bundle == "code",
            "model_weights": bundle != "assets",
            "virtualenv": True,
            "dotenv": True,
        },
        "doc_versions": collect_doc_versions(members),
        "content_digest": _rollup_digest(digests),
        "files": dict(sorted(digests.items())),
    }
    if extra:
        manifest.update(extra)
    return manifest


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _read_members(repo_root: Path, paths: Sequence[str]) -> dict[str, bytes]:
    """Read every member into memory.

    The code bundle is ~10 MB and the corpus ~70 MB, both comfortably in RAM on
    any machine that can run this project, and holding them makes the hash, the
    token scan and the archive write agree on the same bytes rather than on
    three separate reads of a file that could change in between.
    """
    members: dict[str, bytes] = {}
    for rel in paths:
        absolute = repo_root / Path(rel)
        try:
            members[rel] = absolute.read_bytes()
        except OSError as exc:
            raise BundleError(
                f"git lists {rel} but it could not be read from the worktree: {exc}. "
                "If the file was deleted, `git rm` it or restore it before bundling."
            ) from exc
    return members


def write_zip(
    target: Path,
    members: dict[str, bytes],
    manifest: dict[str, object],
    *,
    streamed: dict[str, Path] | None = None,
) -> None:
    """Write *members* plus the manifest to *target*, deterministically ordered.

    Member names are POSIX with no top-level folder, so extracting into an empty
    directory reproduces the project root drawn in
    ``docs/AIR-GAPPED-DEPLOY.md`` section 2.

    ``streamed`` maps archive names to files copied straight from disk instead
    of being held in memory, for members too large to buffer (see
    :data:`LARGE_MEMBER_BYTES`).  Their digests must already be in *manifest* —
    :func:`digest_file` computed them on the same bytes during collection.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.partial")
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False).encode("utf-8")
    alias = bundle_manifest_alias(str(manifest.get("bundle", "unknown")))
    by_path = dict(streamed or {})
    try:
        with zipfile.ZipFile(
            tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=BEST_COMPRESS_LEVEL
        ) as archive:
            archive.writestr(_manifest_info(MANIFEST_NAME), payload)
            archive.writestr(_manifest_info(alias), payload)
            for name in sorted(set(members) | set(by_path)):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                if name in by_path:
                    source = by_path[name]
                    # `open(..., "w")` honours the level passed here rather than
                    # the archive-wide one, so a weights file is not deflated at
                    # level 9 for two percentage points.
                    with (
                        archive.open(info, "w") as sink,
                        source.open("rb") as handle,
                    ):
                        shutil.copyfileobj(handle, sink, length=4 * 1024 * 1024)
                else:
                    archive.writestr(info, members[name])
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def digest_file(path: Path) -> tuple[str, int]:
    """``(sha256, size)`` for a file, read in chunks rather than all at once."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _manifest_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_code_bundle(
    repo_root: Path,
    out_dir: Path,
    git_info: GitInfo,
    report: Report,
    *,
    tracked_only: bool = False,
    max_bytes: int = DEFAULT_MAX_CODE_BYTES,
    exclude_internal: bool = False,
) -> Path:
    """Build ``rbi-ocr-code[-dev]-<shortsha>.zip`` and run every check over it."""
    paths, untracked = collect_code_files(repo_root, tracked_only=tracked_only)
    if not paths:
        raise BundleError("git listed no files for the code bundle; is this a git worktree?")

    dropped: list[str] = []
    if exclude_internal:
        dropped = [p for p in paths if is_internal_working_paper(p)]
        paths = [p for p in paths if not is_internal_working_paper(p)]
        untracked = [u for u in untracked if not is_internal_working_paper(u)]

    check_forbidden(paths, report, bundle="code")
    # Run *after* the drop, so a list that removed a required document fails
    # here rather than shipping a bundle without its own deployment guide.
    check_required(paths, report, exclude_internal=exclude_internal)

    members = _read_members(repo_root, paths)
    stats = scan_for_tokens(members, report, bundle="code")
    _abort_on_token(stats, "code")
    digests = {path: _sha256(blob) for path, blob in members.items()}

    prefix = DEV_CODE_BUNDLE_PREFIX if exclude_internal else CODE_BUNDLE_PREFIX
    target = out_dir / f"{prefix}{git_info.short_sha}.zip"
    manifest = build_manifest(
        bundle="code",
        git_info=git_info,
        members=members,
        digests=digests,
        untracked=untracked,
        tracked_only=tracked_only,
        bundle_name=target.name,
        extra={
            "scan": _scan_manifest_block(stats),
            "max_size_bytes": max_bytes,
            "audience": "developer" if exclude_internal else "full",
            "internal_working_papers_excluded": sorted(dropped),
        },
    )
    write_zip(target, members, manifest)

    size = target.stat().st_size
    if size > max_bytes:
        report.fail(
            "size",
            f"code bundle is {size / 1_048_576:.2f} MB, over the {max_bytes / 1_048_576:.0f} MB ceiling",
        )

    report.note(f"code bundle   : {target}")
    report.note(f"  files       : {len(members)} ({len(untracked)} untracked-but-not-ignored)")
    if exclude_internal:
        report.note(
            f"  audience    : developer — {len(dropped)} internal working paper(s) excluded"
        )
    report.note(f"  uncompressed: {stats['bytes'] / 1_048_576:.2f} MB")
    report.note(f"  zipped      : {size / 1_048_576:.2f} MB  (ceiling {max_bytes / 1_048_576:.0f} MB)")
    report.note(f"  scanned     : {stats['text']} text + {stats['binary']} binary file(s) for token pattern")
    return target


def build_assets_bundle(
    repo_root: Path,
    out_dir: Path,
    git_info: GitInfo,
    report: Report,
    *,
    allow_dirty_corpus: bool = False,
    require_models: bool = True,
) -> Path:
    """Build ``rbi-ocr-models-corpus-<shortsha>.zip``.

    Two things the receiving node cannot obtain for itself, in one archive:

    * ``corpus/`` **exactly as committed** — the same rule as before, because a
      corpus no commit describes cannot be checked against anything;
    * ``models/`` **from the worktree**, because ``.gitignore`` excludes weights
      and no commit lists them. Their guarantee is the per-file SHA-256 in the
      manifest, which is the guarantee that actually matters: it says the bytes
      unzipped are the bytes that were measured.

    Model files are streamed rather than buffered and the large ones are
    deflated at :data:`FAST_COMPRESS_LEVEL` — see those constants for the
    measurements behind both choices.
    """
    dirty = corpus_worktree_is_clean(repo_root)
    if dirty:
        detail = (
            f"{len(dirty)} corpus path(s) differ from HEAD (e.g. {', '.join(dirty[:3])}). "
            "The corpus bundle is specified as 'exactly as committed'; commit or restore them, "
            "or pass --allow-dirty-corpus to ship the worktree anyway."
        )
        if allow_dirty_corpus:
            report.note(f"  WARNING     : {detail}")
        else:
            report.fail("corpus-dirty", detail)

    paths = collect_corpus_files(repo_root)
    if not paths:
        raise BundleError("git listed no files under corpus/; nothing to bundle.")

    check_forbidden(paths, report, bundle="assets")
    members = _read_members(repo_root, paths)
    shape = check_corpus_shape(paths, lambda p: members[p], report)
    stats = scan_for_tokens(members, report, bundle="assets")
    _abort_on_token(stats, "assets")
    digests = {path: _sha256(blob) for path, blob in members.items()}

    # --- models: streamed, hashed on the way past, never fully buffered -----
    model_paths = collect_model_files(repo_root)
    if not model_paths and require_models:
        raise BundleError(
            "models/ holds no files, so the bundle would ship a corpus and no "
            "weights. Run tools/fetch_models.py first, or pass --allow-no-models "
            "to build the corpus half on its own."
        )
    check_forbidden(model_paths, report, bundle="assets")
    streamed: dict[str, Path] = {}
    streamed_bytes: dict[str, int] = {}
    model_bytes = 0
    for rel in model_paths:
        absolute = repo_root / Path(rel)
        sha, size = digest_file(absolute)
        digests[rel] = sha
        model_bytes += size
        if size >= LARGE_MEMBER_BYTES:
            streamed[rel] = absolute
            streamed_bytes[rel] = size
        else:
            # Small enough to buffer, and buffering lets the token scan see it —
            # a stray credential in a config.json would be caught here.
            members[rel] = absolute.read_bytes()
    small_model_members = {
        rel: members[rel] for rel in model_paths if rel in members
    }
    model_stats = scan_for_tokens(small_model_members, report, bundle="assets")
    _abort_on_token(model_stats, "assets")

    models_block: dict[str, object] = {
        "files": len(model_paths),
        "bytes": model_bytes,
        "streamed": sorted(streamed),
        "provenance_present": (repo_root / MODELS_ROOT / "PROVENANCE.json").is_file(),
    }

    target = out_dir / f"{ASSETS_BUNDLE_PREFIX}{git_info.short_sha}.zip"
    manifest = build_manifest(
        bundle="assets",
        git_info=git_info,
        members={**members, **dict.fromkeys(streamed, b"")},
        digests=digests,
        untracked=(),
        tracked_only=True,
        bundle_name=target.name,
        streamed_bytes=streamed_bytes,
        extra={
            "corpus": shape,
            "models": models_block,
            "scan": _scan_manifest_block(stats),
        },
    )
    write_zip(target, members, manifest, streamed=streamed)

    size = target.stat().st_size
    report.note(f"assets bundle : {target}")
    report.note(f"  files       : {len(members) + len(streamed)}")
    report.note(f"  documents   : {shape.get('images')} images + {shape.get('ground_truth_sidecars')} .gt.json")
    report.note(f"  adversarial : {shape.get('adversarial_files')} file(s)")
    report.note(f"  model files : {len(model_paths)} ({model_bytes / 1_048_576:.1f} MB)")
    report.note(f"  uncompressed: {(stats['bytes'] + model_bytes) / 1_048_576:.2f} MB")
    report.note(f"  zipped      : {size / 1_048_576:.2f} MB")
    return target


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #


def verify_zip(path: Path, report: Report, *, max_bytes: int = DEFAULT_MAX_CODE_BYTES) -> dict[str, object]:
    """Re-open an archive and re-run every check against its actual bytes.

    Nothing here trusts the build: the digests are recomputed from what the zip
    contains, not copied from the manifest, and the manifest is then compared to
    them.  Returns the manifest so the caller can cross-check the two bundles.
    """
    if not path.exists():
        report.fail("verify", f"{path} does not exist")
        return {}

    try:
        archive_cm = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        # A truncated or corrupted download is the single most likely thing to be
        # wrong with a 54 MB file that crossed an air gap on a USB stick, so it
        # gets a reported failure and a remedy rather than a traceback.
        report.fail(
            "verify",
            f"{path.name} is not a readable zip ({exc}). It is most likely truncated or "
            "corrupted in transit — re-copy it and compare sizes with the sender.",
        )
        return {}

    with archive_cm as archive:
        bad = archive.testzip()
        if bad is not None:
            report.fail("verify", f"{path.name}: CRC failure in member {bad}")
            return {}
        names = archive.namelist()
        if MANIFEST_NAME not in names:
            report.fail("verify", f"{path.name}: no {MANIFEST_NAME}")
            return {}
        raw_manifest = archive.read(MANIFEST_NAME)
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            report.fail("verify", f"{path.name}: {MANIFEST_NAME} is not readable JSON: {exc}")
            return {}

        bundle = str(manifest.get("bundle", "?"))
        alias = bundle_manifest_alias(bundle)
        if alias not in names:
            report.fail(
                "verify",
                f"{path.name}: missing the collision-proof manifest copy {alias} "
                "(both zips extract into one directory, so the plain name is not enough)",
            )
        elif archive.read(alias) != raw_manifest:
            report.fail("verify", f"{path.name}: {alias} differs from {MANIFEST_NAME}")

        manifest_names = {MANIFEST_NAME, alias}
        member_names = [name for name in names if name not in manifest_names]
        # Buffer what the content checks actually read; stream-hash the rest.
        # The assets bundle carries an 808 MB weights file, and `--verify`
        # loading it into a dict would need more memory than building it did.
        members = {}
        recomputed: dict[str, str] = {}
        for name in member_names:
            info = archive.getinfo(name)
            if info.file_size >= LARGE_MEMBER_BYTES:
                digest = hashlib.sha256()
                with archive.open(name) as handle:
                    while chunk := handle.read(4 * 1024 * 1024):
                        digest.update(chunk)
                recomputed[name] = digest.hexdigest()
            else:
                blob = archive.read(name)
                members[name] = blob
                recomputed[name] = _sha256(blob)

    paths = sorted(member_names)

    for name in names:
        if name.startswith("/") or ".." in PurePosixPath(name).parts or (len(name) > 1 and name[1] == ":"):
            report.fail("verify", f"{path.name}: unsafe member name {name!r}")

    check_forbidden(paths, report, bundle=bundle)
    if bundle == "code":
        check_required(
            paths, report, exclude_internal=manifest.get("audience") == "developer"
        )
        size = path.stat().st_size
        if size > max_bytes:
            report.fail(
                "verify",
                f"{path.name} is {size / 1_048_576:.2f} MB, over the {max_bytes / 1_048_576:.0f} MB ceiling",
            )
    elif bundle == "assets":
        corpus_paths = [p for p in paths if p.startswith(CORPUS_ROOT)]
        check_corpus_shape(corpus_paths, lambda p: members[p], report)
        if not any(p.startswith(MODELS_ROOT) for p in paths):
            report.fail(
                "verify", f"{path.name}: declares bundle 'assets' but carries no models/"
            )
    else:
        report.fail("verify", f"{path.name}: manifest declares unknown bundle type {bundle!r}")

    # Only the buffered members are scanned; a weights file is not a place a
    # credential can hide in a form this pattern would find, and reading 808 MB
    # through a regex to prove that would cost minutes per verify.
    stats = scan_for_tokens(members, report, bundle=bundle)
    declared = manifest.get("files", {})
    if not isinstance(declared, dict):
        report.fail("verify", f"{path.name}: manifest 'files' is not an object")
        declared = {}
    missing = sorted(set(declared) - set(recomputed))
    extra = sorted(set(recomputed) - set(declared))
    mismatched = sorted(n for n in set(declared) & set(recomputed) if declared[n] != recomputed[n])
    if missing:
        report.fail("verify", f"{path.name}: {len(missing)} manifest entr(ies) absent from the archive: {missing[:5]}")
    if extra:
        report.fail("verify", f"{path.name}: {len(extra)} archive member(s) not in the manifest: {extra[:5]}")
    if mismatched:
        report.fail("verify", f"{path.name}: {len(mismatched)} SHA-256 mismatch(es): {mismatched[:5]}")

    # `member_names`, not `members`: the latter holds only what was small
    # enough to buffer, so counting it would report an 808 MB weights file as
    # missing from its own archive.
    if manifest.get("file_count") != len(member_names):
        report.fail(
            "verify",
            f"{path.name}: manifest file_count={manifest.get('file_count')} "
            f"but archive holds {len(member_names)}",
        )
    rollup = _rollup_digest(recomputed)
    if manifest.get("content_digest") != rollup:
        report.fail("verify", f"{path.name}: content_digest does not match the archive's members")

    report.note(f"verified      : {path.name}")
    report.note(f"  bundle      : {bundle}   sha {manifest.get('git', {}).get('short_sha', '?')}")
    streamed_count = len(member_names) - len(members)
    report.note(
        f"  members     : {len(member_names)}  (all SHA-256s recomputed and matched"
        + (f"; {streamed_count} streamed" if streamed_count else "")
        + ")"
    )
    report.note(
        f"  scanned     : {stats['text']} text + {stats['binary']} binary file(s), "
        f"{stats['matches']} token match(es)"
    )
    report.note(f"  size        : {path.stat().st_size / 1_048_576:.2f} MB")
    return manifest


def cross_check(code_manifest: dict[str, object], assets_manifest: dict[str, object], report: Report) -> None:
    """The two bundles must describe the same commit (deploy guide section 0)."""
    if not code_manifest or not assets_manifest:
        return
    code_sha = (code_manifest.get("git") or {}).get("sha")  # type: ignore[union-attr]
    assets_sha = (assets_manifest.get("git") or {}).get("sha")  # type: ignore[union-attr]
    if code_sha != assets_sha:
        report.fail("cross-check", f"code bundle is {code_sha}, assets bundle is {assets_sha} — not a matching pair")
    else:
        report.note(f"  pair        : both bundles carry git sha {code_sha}")


def find_bundles(out_dir: Path, short_sha: str | None, report: Report | None = None) -> tuple[Path | None, Path | None]:
    """Locate the two zips in *out_dir*, preferring the ones stamped *short_sha*.

    Falling back to the newest zip is convenient and also the way someone
    verifies a stale pair without noticing, so the fallback says so out loud.
    """

    def pick(prefix: str, *, reject: str | None = None) -> Path | None:
        """Newest zip with *prefix*, ignoring any that start with *reject*.

        ``reject`` exists because ``rbi-ocr-code-dev-<sha>.zip`` also begins
        with ``rbi-ocr-code-``: without it, a glob for the full bundle happily
        returns the developer one and ``--verify`` checks the wrong archive.
        """

        def eligible(path: Path) -> bool:
            return reject is None or not path.name.startswith(reject)

        if short_sha:
            exact = out_dir / f"{prefix}{short_sha}.zip"
            if exact.exists() and eligible(exact):
                return exact
        candidates = sorted(
            (p for p in out_dir.glob(f"{prefix}*.zip") if eligible(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates and short_sha and report is not None:
            report.note(
                f"  WARNING     : no {prefix}{short_sha}.zip for the current HEAD; "
                f"verifying {candidates[0].name} instead. Re-run without --verify to rebuild."
            )
        return candidates[0] if candidates else None

    # The developer variant wins when both are present: it is the one being
    # handed over, so it is the one worth re-checking.
    code = pick(DEV_CODE_BUNDLE_PREFIX) or pick(
        CODE_BUNDLE_PREFIX, reject=DEV_CODE_BUNDLE_PREFIX
    )
    return code, pick(ASSETS_BUNDLE_PREFIX)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_report(report: Report, *, header: str) -> None:
    print()
    print(header)
    print("-" * len(header))
    for note in report.notes:
        print(note)
    if report.problems:
        print()
        print(f"{len(report.problems)} PROBLEM(S):")
        for problem in report.problems:
            print(f"  {problem}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_bundle.py",
        description="Build (and verify) the code and models+corpus hand-off zips in dist/.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: the parent of tools/).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <repo-root>/dist).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Do not build. Re-open the zips already in --out and re-check everything.",
    )
    parser.add_argument(
        "--only",
        choices=("code", "assets"),
        default=None,
        help="Build/verify only one of the two bundles (default: both).",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Committed files only. Omits constraints.txt and the four WS-I documents "
        "until they are committed, so the build will fail its own required-file check.",
    )
    parser.add_argument(
        "--allow-dirty-corpus",
        action="store_true",
        help="Bundle corpus/ from the worktree even if it differs from HEAD.",
    )
    parser.add_argument(
        "--exclude-internal",
        action="store_true",
        help="Drop the internal working papers (decision logs, OPEN-QUESTIONS, "
        "PLAN, bake-off data, raw benchmarks) from the code bundle and name it "
        "rbi-ocr-code-dev-<sha>.zip. For handing the system to a developer who "
        "needs the specifications, not the deliberations.",
    )
    parser.add_argument(
        "--allow-no-models",
        action="store_true",
        help="Build the assets bundle with an empty models/ instead of failing. "
        "The receiving node then has to fetch weights itself, which an "
        "air-gapped one cannot — so this is for testing, not for a hand-off.",
    )
    parser.add_argument(
        "--max-code-mb",
        type=float,
        default=DEFAULT_MAX_CODE_BYTES / 1_048_576,
        help="Ceiling for the code bundle in MB (default: 15).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    out_dir = (args.out or repo_root / "dist").resolve()
    max_bytes = int(args.max_code_mb * 1_048_576)

    if not (repo_root / ".git").exists():
        print(f"ERROR: {repo_root} is not a git worktree (no .git). Pass --repo-root.", file=sys.stderr)
        return 1

    report = Report()
    try:
        git_info = read_git_info(repo_root)
    except BundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"repository    : {repo_root}")
    print(f"git HEAD      : {git_info.sha}  ({git_info.short_sha}) on {git_info.branch}")
    print(f"committed at  : {git_info.commit_time}")
    if git_info.dirty:
        print(
            f"WORKTREE DIRTY: {len(git_info.dirty_paths)} path(s) differ from HEAD. "
            "The bundles are built from the WORKTREE, so they contain those changes; "
            "the short SHA names the commit they are based on, not their exact content."
        )
    print(f"output        : {out_dir}")

    code_manifest: dict[str, object] = {}
    assets_manifest: dict[str, object] = {}

    try:
        if args.verify:
            code_zip, assets_zip = find_bundles(out_dir, git_info.short_sha, report)
            if args.only != "assets":
                if code_zip is None:
                    report.fail("verify", f"no {CODE_BUNDLE_PREFIX}*.zip found in {out_dir}")
                else:
                    code_manifest = verify_zip(code_zip, report, max_bytes=max_bytes)
            if args.only != "code":
                if assets_zip is None:
                    report.fail("verify", f"no {ASSETS_BUNDLE_PREFIX}*.zip found in {out_dir}")
                else:
                    assets_manifest = verify_zip(assets_zip, report, max_bytes=max_bytes)
            header = "VERIFY"
        else:
            code_zip = assets_zip = None
            if args.only != "assets":
                code_zip = build_code_bundle(
                    repo_root,
                    out_dir,
                    git_info,
                    report,
                    tracked_only=args.tracked_only,
                    max_bytes=max_bytes,
                    exclude_internal=args.exclude_internal,
                )
            if args.only != "code":
                assets_zip = build_assets_bundle(
                    repo_root,
                    out_dir,
                    git_info,
                    report,
                    allow_dirty_corpus=args.allow_dirty_corpus,
                    require_models=not args.allow_no_models,
                )
            # Always re-open what was just written: a build that cannot survive its
            # own --verify is not a build anyone should ship.
            if code_zip is not None:
                code_manifest = verify_zip(code_zip, report, max_bytes=max_bytes)
            if assets_zip is not None:
                assets_manifest = verify_zip(assets_zip, report, max_bytes=max_bytes)
            header = "BUILD + VERIFY"

        if args.only is None:
            cross_check(code_manifest, assets_manifest, report)
    except BundleError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(report, header=header)
    print()
    if report.ok:
        print("BUNDLE OK — every assertion passed.")
        return 0
    print("BUNDLE BUILD FAILED — see the problems above. Nothing should be shipped.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
