"""Unit tests for ``tools/make_bundle.py`` — the office hand-off bundler.

Nothing here builds the *real* bundles.  Every end-to-end test runs against a
miniature project laid out in a **real git repository** created in ``tmp_path``,
because the tool's whole selection mechanism is ``git ls-files`` and a fake that
returned a list would test the assertions while skipping the part that decides
what gets shipped.

What is tested is the part of a bundler that can silently be wrong and would
then be discovered by a stranger on an air-gapped machine, or not at all:

* **exclusion** — that ``corpus/``, ``.env``, model weights, ``.venv/`` and
  ``dist/`` cannot enter the code bundle, including when ``.gitignore`` is the
  only thing keeping them out;
* **inclusion** — that an untracked-but-not-ignored file (which is what
  ``constraints.txt`` and the four WS-I documents were when the tool was
  written) does get in, and is declared as such in the manifest;
* **the token scan** — that a planted token is caught, that no archive is
  written when one is, that the token's *value* never reaches the report, and
  that the ``hf_xxx`` placeholder the documents use does not false-positive;
* **manifest shape** — that every digest is the digest of the bytes actually in
  the archive, and that ``--verify`` notices when one byte changes.

The requirement lists (``REQUIRED_CODE_PATHS``, ``REQUIRED_CODE_TREES``,
``EXPECTED_CORPUS_DOCUMENTS``) describe the real project, so the end-to-end
tests monkeypatch them down to the fake tree's scale.  One test asserts the
shipped values themselves cover the WS-I deliverables, which is the part of
those constants worth locking.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: A syntactically valid Hugging Face token shape: ``hf_`` + 34 alphanumerics.
#: Fabricated for this test, matches no account, and is never printed by the tool.
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Qr"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="make_bundle.py reads the file list from git")


@pytest.fixture(scope="module")
def mb() -> ModuleType:
    """Load the tool as a module (``tools/`` is not a package)."""
    path = PROJECT_ROOT / "tools" / "make_bundle.py"
    spec = importlib.util.spec_from_file_location("make_bundle_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The fake project
# --------------------------------------------------------------------------- #


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=str(root),
        capture_output=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A miniature project with the same *shapes* the real one has.

    Two committed corpus documents, one gitignored ``.env`` holding a token, a
    gitignored model weight, a ``models/README.md`` that must survive, a
    ``.venv`` and a ``dist`` that must not, and one untracked-but-not-ignored
    document standing in for ``constraints.txt``.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")

    _write(root, ".gitignore", ".env\n.venv/\ndist/\n__pycache__/\nmodels/*\n!models/README.md\n")
    _write(root, "README.md", "# fake project\n")
    _write(root, "requirements.txt", "pydantic==2.0\n")
    _write(root, ".env.example", "HF_TOKEN=hf_xxx\nPROVIDER_PROFILE=laptop\n")
    _write(root, "app/__init__.py", "")
    _write(root, "app/pipeline.py", "def run() -> int:\n    return 1\n")
    _write(root, "tools/generate_test_corpus.py", "# generator\n")
    _write(root, "tools/assets/fonts/NotoSansDevanagari-Regular.ttf", "not really a font\n")
    _write(root, "tools/assets/fribidi-0.dll", "not really a dll\n")
    _write(root, "tests/conftest.py", "# fixtures\n")
    _write(root, "docs/AIR-GAPPED-DEPLOY.md", "# deploy\nHF_TOKEN=hf_xxx placeholder only.\n")
    _write(root, "models/README.md", "Model weights are placed here offline.\n")

    # Corpus: two documents, one adversarial file, an index that names them.
    for name in ("alpha", "beta"):
        _write(root, f"corpus/forms/{name}.jpg", f"pretend-jpeg-{name}\n")
        _write(root, f"corpus/forms/{name}.gt.json", json.dumps({"doc_id": name, "synthetic": True}))
    _write(root, "corpus/adversarial/MANIFEST.json", json.dumps({"synthetic": True}))
    _write(root, "corpus/adversarial/zero_byte.pdf", "")
    _write(
        root,
        "corpus/index.json",
        json.dumps(
            {
                "count": 2,
                "synthetic": True,
                "seed": 1729,
                "generator_version": "1.2.0",
                # Backslashes on purpose: the real index.json was written on Windows.
                "documents": [
                    {"doc_id": n, "image": f"forms\\{n}.jpg", "gt": f"forms\\{n}.gt.json"} for n in ("alpha", "beta")
                ],
            }
        ),
    )

    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")

    # Ignored, and therefore invisible to `git ls-files --exclude-standard`.
    _write(root, ".env", f"HF_TOKEN={FAKE_TOKEN}\n")
    _write(root, "models/ocr/rec_deva.onnx", "pretend weights\n")
    _write(root, "models/PROVENANCE.json", json.dumps({"artifacts": []}))
    _write(root, ".venv/Scripts/python.exe", "pretend interpreter\n")
    _write(root, "dist/stale-bundle.zip", "leftover\n")
    _write(root, "app/__pycache__/pipeline.cpython-312.pyc", "bytecode\n")

    # Untracked but NOT ignored — the constraints.txt / WS-I-documents case.
    _write(root, "constraints.txt", "pydantic==2.0\n")

    return root


@pytest.fixture
def small_requirements(mb: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scale the real project's required-content lists down to the fake tree."""
    monkeypatch.setattr(
        mb,
        "REQUIRED_CODE_PATHS",
        (
            "docs/AIR-GAPPED-DEPLOY.md",
            "constraints.txt",
            ".env.example",
            "requirements.txt",
            "tools/generate_test_corpus.py",
            "tools/assets/fribidi-0.dll",
            "models/README.md",
            "README.md",
        ),
    )
    monkeypatch.setattr(mb, "REQUIRED_CODE_TREES", (("app/", 2), ("tools/", 3), ("docs/", 1)))
    monkeypatch.setattr(mb, "EXPECTED_CORPUS_DOCUMENTS", 2)


def _build(mb: ModuleType, root: Path, **kwargs: object) -> tuple[object, Path, Path]:
    """Build both bundles in ``<root>/dist`` and return ``(report, code, assets)``."""
    report = mb.Report()
    git_info = mb.read_git_info(root)
    out = root / "dist"
    code = mb.build_code_bundle(root, out, git_info, report, **kwargs)  # type: ignore[arg-type]
    assets = mb.build_assets_bundle(root, out, git_info, report)
    return report, code, assets


def _members(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as archive:
        return set(archive.namelist())


def _manifest(zip_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read("BUNDLE_MANIFEST.json").decode("utf-8"))


# --------------------------------------------------------------------------- #
# Exclusion rules
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("small_requirements")
class TestExclusionRules:
    def test_build_succeeds_and_writes_both_zips(self, mb: ModuleType, fake_repo: Path) -> None:
        report, code, assets = _build(mb, fake_repo)
        assert report.ok, [str(p) for p in report.problems]
        assert code.exists() and assets.exists()
        assert code.name.startswith("rbi-ocr-code-")
        assert assets.name.startswith("rbi-ocr-models-corpus-")

    def test_code_bundle_contains_no_corpus_file(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        assert not [n for n in _members(code) if n.startswith("corpus/")]

    def test_gitignored_paths_never_enter_the_code_bundle(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        names = _members(code)
        for leaked in (
            ".env",
            "models/ocr/rec_deva.onnx",
            "models/PROVENANCE.json",
            ".venv/Scripts/python.exe",
            "dist/stale-bundle.zip",
            "app/__pycache__/pipeline.cpython-312.pyc",
        ):
            assert leaked not in names, f"{leaked} leaked into the code bundle"

    def test_models_readme_is_the_only_models_member(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        assert sorted(n for n in _members(code) if n.startswith("models/")) == ["models/README.md"]

    def test_untracked_but_not_ignored_files_are_included_and_declared(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        _, code, _ = _build(mb, fake_repo)
        assert "constraints.txt" in _members(code)
        assert _manifest(code)["untracked_included"] == ["constraints.txt"]

    def test_tracked_only_omits_untracked_and_then_fails_its_own_required_check(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        report = mb.Report()
        git_info = mb.read_git_info(fake_repo)
        code = mb.build_code_bundle(fake_repo, fake_repo / "dist", git_info, report, tracked_only=True)
        assert "constraints.txt" not in _members(code)
        assert _manifest(code)["untracked_included"] == []
        assert not report.ok
        assert any("constraints.txt" in p.detail for p in report.problems)

    def test_assets_bundle_holds_corpus_models_and_the_manifests_only(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        _, _, assets = _build(mb, fake_repo)
        stray = [
            n
            for n in _members(assets)
            if not n.startswith(("corpus/", "models/")) and not n.startswith("BUNDLE_MANIFEST")
        ]
        assert stray == []

    def test_assets_bundle_carries_the_gitignored_weights(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        """The whole reason it exists: an air-gapped node cannot fetch these.

        They are invisible to `git ls-files`, so the collector reads the
        worktree and the manifest's SHA-256 is their only provenance.
        """
        _, _, assets = _build(mb, fake_repo)
        members = _members(assets)
        assert "models/ocr/rec_deva.onnx" in members
        assert "models/PROVENANCE.json" in members
        assert "models/README.md" in members
        manifest = _manifest(assets)
        assert manifest["models"]["files"] == 3  # type: ignore[index]
        assert manifest["excluded"]["model_weights"] is False  # type: ignore[index]

    def test_the_code_bundle_still_refuses_weights(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        """Relaxing the weight rule for one bundle must not relax it for both."""
        _, code, _ = _build(mb, fake_repo)
        assert not any(n.endswith(".onnx") for n in _members(code))

    def test_an_empty_models_directory_is_refused(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        """A hand-off with no weights strands the node it is handed to."""
        for weight in (fake_repo / "models").rglob("*"):
            if weight.is_file():
                weight.unlink()
        with pytest.raises(mb.BundleError, match="no files"):
            mb.build_assets_bundle(
                fake_repo, fake_repo / "dist", mb.read_git_info(fake_repo), mb.Report()
            )

    def test_assets_bundle_refuses_a_corpus_that_differs_from_head(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        (fake_repo / "corpus" / "forms" / "alpha.jpg").write_text("edited after the commit\n", encoding="utf-8")
        report = mb.Report()
        git_info = mb.read_git_info(fake_repo)
        mb.build_assets_bundle(fake_repo, fake_repo / "dist", git_info, report)
        assert any(p.check == "corpus-dirty" for p in report.problems)

        permissive = mb.Report()
        mb.build_assets_bundle(fake_repo, fake_repo / "dist", git_info, permissive, allow_dirty_corpus=True)
        assert permissive.ok

    def test_archives_have_no_top_level_folder_and_posix_names(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        for archive in (code, assets):
            for name in _members(archive):
                assert "\\" not in name, f"{name} in {archive.name} uses a Windows separator"
                assert not name.startswith("/") and ".." not in Path(name).parts
        assert "app/pipeline.py" in _members(code)

    def test_forbidden_check_catches_what_gitignore_would_have_missed(self, mb: ModuleType) -> None:
        report = mb.Report()
        mb.check_forbidden(
            [
                "app/ok.py",
                "models/ocr/rec.onnx",
                "secrets/server.pem",
                "config/.env",
                ".venv/Lib/site-packages/x.py",
                "corpus/forms/a.jpg",
                "dist/old.zip",
            ],
            report,
            bundle="code",
        )
        flagged = " ".join(p.detail for p in report.problems)
        for expected in ("rec.onnx", "server.pem", ".env", "site-packages", "corpus/forms/a.jpg", "dist/old.zip"):
            assert expected in flagged
        assert "app/ok.py" not in flagged


# --------------------------------------------------------------------------- #
# Token scan
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("small_requirements")
class TestTokenScan:
    def test_a_planted_token_aborts_the_build_and_writes_no_archive(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        _write(fake_repo, "docs/notes.md", f"remember the token {FAKE_TOKEN}\n")
        _git(fake_repo, "add", "-A")
        _git(fake_repo, "commit", "-q", "-m", "oops")

        report = mb.Report()
        git_info = mb.read_git_info(fake_repo)
        with pytest.raises(mb.BundleError) as excinfo:
            mb.build_code_bundle(fake_repo, fake_repo / "dist", git_info, report)

        assert not list((fake_repo / "dist").glob("rbi-ocr-code-*.zip")), "an archive with a token was written"
        assert any(p.check == "token-scan" for p in report.problems)
        assert "docs/notes.md" in " ".join(p.detail for p in report.problems)
        # The value must never reach the report, the exception, or a log line.
        assert FAKE_TOKEN not in str(excinfo.value)
        assert FAKE_TOKEN not in " ".join(str(p) for p in report.problems)

    def test_a_token_in_a_binary_member_is_caught_too(self, mb: ModuleType, fake_repo: Path) -> None:
        (fake_repo / "tools" / "assets" / "fribidi-0.dll").write_bytes(
            b"MZ\x00\x00" + FAKE_TOKEN.encode("ascii") + b"\x00\x00"
        )
        _git(fake_repo, "add", "-A")
        _git(fake_repo, "commit", "-q", "-m", "binary token")
        report = mb.Report()
        with pytest.raises(mb.BundleError):
            mb.build_code_bundle(fake_repo, fake_repo / "dist", mb.read_git_info(fake_repo), report)
        assert "fribidi-0.dll" in " ".join(p.detail for p in report.problems)

    def test_the_hf_xxx_placeholder_the_documents_use_does_not_false_positive(self, mb: ModuleType) -> None:
        report = mb.Report()
        stats = mb.scan_for_tokens(
            {
                ".env.example": b"HF_TOKEN=hf_xxx\n",
                "docs/TUNING.md": b"Set `HF_TOKEN=hf_xxx` in .env; never commit the real value.\n",
                "docs/API.md": b"Authorization: Bearer hf_xxxxxxxxxxxxxxxxxxx\n",  # 19 chars, under the floor
            },
            report,
            bundle="code",
        )
        assert report.ok
        assert stats["matches"] == 0

    def test_the_scan_reports_a_text_binary_split_over_every_byte(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        scan = _manifest(code)["scan"]
        assert isinstance(scan, dict)
        assert scan["token_matches"] == 0
        assert scan["binary_files_scanned"] >= 1  # the .ttf and the .dll
        assert scan["bytes_scanned"] == _manifest(code)["total_uncompressed_bytes"]


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("small_requirements")
class TestManifest:
    def test_manifest_carries_the_fields_the_plan_asks_for(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        manifest = _manifest(code)
        for key in (
            "bundle",
            "built_at_utc",
            "git",
            "file_count",
            "files",
            "doc_versions",
            "content_digest",
            "source_of_truth",
            "untracked_included",
            "scan",
        ):
            assert key in manifest, f"manifest is missing {key}"
        assert manifest["bundle"] == "code"
        assert _manifest(assets)["bundle"] == "assets"
        assert len(manifest["git"]["sha"]) == 40  # type: ignore[index]
        assert manifest["git"]["short_sha"] in code.name  # type: ignore[index]

    def test_every_declared_digest_is_the_digest_of_the_archived_bytes(
        self, mb: ModuleType, fake_repo: Path
    ) -> None:
        _, code, _ = _build(mb, fake_repo)
        manifest = _manifest(code)
        with zipfile.ZipFile(code) as archive:
            for name, declared in manifest["files"].items():  # type: ignore[union-attr]
                assert hashlib.sha256(archive.read(name)).hexdigest() == declared, name
        assert manifest["file_count"] == len(manifest["files"])  # type: ignore[arg-type]

    def test_file_count_excludes_the_two_manifest_copies(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        assert _manifest(code)["file_count"] == len(_members(code)) - 2

    def test_both_manifest_copies_are_present_and_identical(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        for archive_path, label in ((code, "code"), (assets, "assets")):
            with zipfile.ZipFile(archive_path) as archive:
                plain = archive.read("BUNDLE_MANIFEST.json")
                alias = archive.read(f"BUNDLE_MANIFEST-{label}.json")
            assert plain == alias
        # The point of the alias: both survive a combined extract into one directory.
        assert "BUNDLE_MANIFEST-code.json" in _members(code)
        assert "BUNDLE_MANIFEST-assets.json" in _members(assets)
        # models/README.md is in both by design: the code bundle needs it for the
        # unzip layout, the assets bundle ships it beside the weights.
        assert _members(code) & _members(assets) == {
            "BUNDLE_MANIFEST.json",
            "models/README.md",
        }

    def test_both_bundles_name_the_same_commit(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        assert _manifest(code)["git"]["sha"] == _manifest(assets)["git"]["sha"]  # type: ignore[index]
        report = mb.Report()
        mb.cross_check(_manifest(code), _manifest(assets), report)
        assert report.ok

    def test_cross_check_rejects_a_mismatched_pair(self, mb: ModuleType) -> None:
        report = mb.Report()
        mb.cross_check({"git": {"sha": "a" * 40}}, {"git": {"sha": "b" * 40}}, report)
        assert any(p.check == "cross-check" for p in report.problems)

    def test_doc_versions_stamp_the_documents_that_are_present(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        code_versions = _manifest(code)["doc_versions"]
        assert "docs/AIR-GAPPED-DEPLOY.md" in code_versions  # type: ignore[operator]
        assert "constraints.txt" in code_versions  # type: ignore[operator]
        entry = code_versions["docs/AIR-GAPPED-DEPLOY.md"]  # type: ignore[index]
        assert set(entry) == {"sha256", "bytes", "lines"}
        # The assets bundle stamps the generator instead, from index.json.
        assert _manifest(assets)["doc_versions"]["corpus_generator_version"] == "1.2.0"  # type: ignore[index]

    def test_worktree_dirt_is_recorded_rather_than_hidden(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        git_block = _manifest(code)["git"]
        assert git_block["worktree_dirty"] is True  # type: ignore[index]
        assert git_block["worktree_dirty_path_count"] >= 1  # type: ignore[index]

    def test_rollup_digest_notices_a_rename(self, mb: ModuleType) -> None:
        a = mb._rollup_digest({"app/one.py": "aa", "app/two.py": "bb"})
        b = mb._rollup_digest({"app/one.py": "aa", "app/three.py": "bb"})
        same_order = mb._rollup_digest({"app/two.py": "bb", "app/one.py": "aa"})
        assert a != b
        assert a == same_order

    def test_corpus_summary_counts_the_documents(self, mb: ModuleType, fake_repo: Path) -> None:
        _, _, assets = _build(mb, fake_repo)
        summary = _manifest(assets)["corpus"]
        assert summary["images"] == 2  # type: ignore[index]
        assert summary["ground_truth_sidecars"] == 2  # type: ignore[index]
        assert summary["index_references_resolved"] == 4  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("small_requirements")
class TestVerify:
    def test_verify_passes_on_a_freshly_built_pair(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, assets = _build(mb, fake_repo)
        report = mb.Report()
        mb.verify_zip(code, report)
        mb.verify_zip(assets, report)
        assert report.ok, [str(p) for p in report.problems]

    def test_verify_catches_a_single_changed_byte(self, mb: ModuleType, fake_repo: Path, tmp_path: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        tampered = tmp_path / "tampered.zip"
        with zipfile.ZipFile(code) as src, zipfile.ZipFile(tampered, "w") as dst:
            for info in src.infolist():
                blob = src.read(info.filename)
                if info.filename == "app/pipeline.py":
                    blob = blob.replace(b"return 1", b"return 2")
                dst.writestr(info.filename, blob)
        report = mb.Report()
        mb.verify_zip(tampered, report)
        assert any("SHA-256 mismatch" in p.detail for p in report.problems)

    def test_verify_catches_an_added_member(self, mb: ModuleType, fake_repo: Path, tmp_path: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        tampered = tmp_path / "extra.zip"
        with zipfile.ZipFile(code) as src, zipfile.ZipFile(tampered, "w") as dst:
            for info in src.infolist():
                dst.writestr(info.filename, src.read(info.filename))
            dst.writestr("app/smuggled.py", "print('hello')\n")
        report = mb.Report()
        mb.verify_zip(tampered, report)
        assert any("not in the manifest" in p.detail for p in report.problems)

    def test_verify_catches_a_missing_member(self, mb: ModuleType, fake_repo: Path, tmp_path: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        tampered = tmp_path / "short.zip"
        with zipfile.ZipFile(code) as src, zipfile.ZipFile(tampered, "w") as dst:
            for info in src.infolist():
                if info.filename == "docs/AIR-GAPPED-DEPLOY.md":
                    continue
                dst.writestr(info.filename, src.read(info.filename))
        report = mb.Report()
        mb.verify_zip(tampered, report)
        details = " ".join(p.detail for p in report.problems)
        assert "absent from the archive" in details
        assert "missing docs/AIR-GAPPED-DEPLOY.md" in details

    def test_verify_catches_a_token_smuggled_into_an_archive(
        self, mb: ModuleType, fake_repo: Path, tmp_path: Path
    ) -> None:
        _, code, _ = _build(mb, fake_repo)
        tampered = tmp_path / "leaky.zip"
        with zipfile.ZipFile(code) as src, zipfile.ZipFile(tampered, "w") as dst:
            for info in src.infolist():
                blob = src.read(info.filename)
                if info.filename == "README.md":
                    blob = blob + FAKE_TOKEN.encode("ascii")
                dst.writestr(info.filename, blob)
        report = mb.Report()
        mb.verify_zip(tampered, report)
        assert any(p.check == "token-scan" for p in report.problems)
        assert FAKE_TOKEN not in " ".join(str(p) for p in report.problems)

    def test_verify_requires_the_collision_proof_manifest_copy(
        self, mb: ModuleType, fake_repo: Path, tmp_path: Path
    ) -> None:
        _, code, _ = _build(mb, fake_repo)
        stripped = tmp_path / "one-manifest.zip"
        with zipfile.ZipFile(code) as src, zipfile.ZipFile(stripped, "w") as dst:
            for info in src.infolist():
                if info.filename == "BUNDLE_MANIFEST-code.json":
                    continue
                dst.writestr(info.filename, src.read(info.filename))
        report = mb.Report()
        mb.verify_zip(stripped, report)
        assert any("collision-proof" in p.detail for p in report.problems)

    def test_verify_enforces_the_size_ceiling(self, mb: ModuleType, fake_repo: Path) -> None:
        _, code, _ = _build(mb, fake_repo)
        report = mb.Report()
        mb.verify_zip(code, report, max_bytes=10)
        assert any("ceiling" in p.detail for p in report.problems)


# --------------------------------------------------------------------------- #
# Corpus shape
# --------------------------------------------------------------------------- #


class TestCorpusShape:
    def test_a_short_corpus_is_rejected(self, mb: ModuleType) -> None:
        report = mb.Report()
        index = json.dumps({"count": 295, "synthetic": True, "documents": []}).encode("utf-8")
        mb.check_corpus_shape(
            ["corpus/index.json", "corpus/forms/a.jpg", "corpus/forms/a.gt.json"],
            lambda _p: index,
            report,
        )
        details = " ".join(p.detail for p in report.problems)
        assert "expected 295 document images, found 1" in details
        assert "MANIFEST.json is missing" in details

    def test_an_index_naming_an_absent_file_is_rejected(self, mb: ModuleType, monkeypatch) -> None:
        monkeypatch.setattr(mb, "EXPECTED_CORPUS_DOCUMENTS", 1)
        report = mb.Report()
        index = json.dumps(
            {
                "count": 1,
                "synthetic": True,
                "documents": [{"doc_id": "ghost", "image": "forms\\ghost.jpg", "gt": "forms\\ghost.gt.json"}],
            }
        ).encode("utf-8")
        mb.check_corpus_shape(
            [
                "corpus/index.json",
                "corpus/forms/a.jpg",
                "corpus/forms/a.gt.json",
                "corpus/adversarial/MANIFEST.json",
                "corpus/adversarial/zero_byte.pdf",
            ],
            lambda _p: index,
            report,
        )
        assert any("named by index.json are absent" in p.detail for p in report.problems)

    def test_a_corpus_that_does_not_declare_itself_synthetic_is_rejected(
        self, mb: ModuleType, monkeypatch
    ) -> None:
        """Invariant 6: the residency gate refuses a corpus without a positive declaration."""
        monkeypatch.setattr(mb, "EXPECTED_CORPUS_DOCUMENTS", 1)
        report = mb.Report()
        index = json.dumps({"count": 1, "documents": [{"doc_id": "a", "image": "a.jpg", "gt": "a.gt.json"}]})
        mb.check_corpus_shape(
            [
                "corpus/index.json",
                "corpus/a.jpg",
                "corpus/a.gt.json",
                "corpus/adversarial/MANIFEST.json",
                "corpus/adversarial/zero_byte.pdf",
            ],
            lambda _p: index.encode("utf-8"),
            report,
        )
        assert any("synthetic: true" in p.detail for p in report.problems)

    def test_windows_backslash_paths_in_the_index_still_resolve(self, mb: ModuleType, monkeypatch) -> None:
        """The committed index.json stores ``type\\doc.jpg``; the check must normalise."""
        monkeypatch.setattr(mb, "EXPECTED_CORPUS_DOCUMENTS", 1)
        report = mb.Report()
        index = json.dumps(
            {
                "count": 1,
                "synthetic": True,
                "documents": [{"doc_id": "a", "image": "forms\\a.jpg", "gt": "forms\\a.gt.json"}],
            }
        ).encode("utf-8")
        mb.check_corpus_shape(
            [
                "corpus/index.json",
                "corpus/forms/a.jpg",
                "corpus/forms/a.gt.json",
                "corpus/adversarial/MANIFEST.json",
                "corpus/adversarial/zero_byte.pdf",
            ],
            lambda _p: index,
            report,
        )
        assert report.ok, [str(p) for p in report.problems]


# --------------------------------------------------------------------------- #
# The shipped constants, and the CLI
# --------------------------------------------------------------------------- #


class TestShippedConfiguration:
    def test_required_paths_cover_the_ws_i_deliverables(self, mb: ModuleType) -> None:
        required = set(mb.REQUIRED_CODE_PATHS)
        for path in (
            "docs/AIR-GAPPED-DEPLOY.md",
            "docs/API.md",
            "docs/FUNCTIONAL-SPEC.md",
            "docs/TUNING.md",
            "constraints.txt",
            "requirements.txt",
            "requirements-dev.txt",
            ".env.example",
            "tools/generate_test_corpus.py",
            "tools/assets/fribidi-0.dll",
            "models/README.md",
        ):
            assert path in required, f"PLAN.md WS-I item 7 names {path}; the tool does not require it"

    def test_every_required_path_exists_in_this_repository(self, mb: ModuleType) -> None:
        missing = [p for p in mb.REQUIRED_CODE_PATHS if not (PROJECT_ROOT / p).exists()]
        assert missing == [], f"the tool would refuse to build: {missing}"

    def test_required_tree_prefixes_are_directories(self, mb: ModuleType) -> None:
        for prefix, minimum in mb.REQUIRED_CODE_TREES:
            assert prefix.endswith("/"), prefix
            assert minimum > 0
            assert (PROJECT_ROOT / prefix.rstrip("/")).is_dir(), prefix

    def test_the_dotenv_is_gitignored_in_this_repository(self, mb: ModuleType) -> None:
        """The bundle's safety rests on this; assert it rather than assume it.

        Skipped when this file is not sitting in a git worktree — which is exactly
        the case when the suite is run out of an extracted code bundle, where the
        absence of ``.git`` is the point rather than a problem.
        """
        assert ".env" in mb.SECRET_BASENAMES
        assert ".env\n" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        if not (PROJECT_ROOT / ".git").exists():
            pytest.skip("not a git worktree (running from an extracted bundle)")
        completed = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=str(PROJECT_ROOT), capture_output=True, check=False
        )
        assert completed.returncode == 0, ".env is NOT gitignored in this repository"

    def test_the_token_pattern_matches_a_real_shaped_token_and_not_the_placeholder(
        self, mb: ModuleType
    ) -> None:
        assert mb.TOKEN_PATTERN.search(FAKE_TOKEN.encode("ascii")) is not None
        assert mb.TOKEN_PATTERN.search(b"HF_TOKEN=hf_xxx") is None


@pytest.mark.usefixtures("small_requirements")
class TestCli:
    def test_main_builds_verifies_and_exits_zero(self, mb: ModuleType, fake_repo: Path) -> None:
        code = mb.main(["--repo-root", str(fake_repo), "--out", str(fake_repo / "dist")])
        assert code == 0
        # `dist/` already held an unrelated stale-bundle.zip; the build adds exactly two.
        assert len(list((fake_repo / "dist").glob("rbi-ocr-*.zip"))) == 2

    def test_verify_mode_reopens_what_is_already_there(self, mb: ModuleType, fake_repo: Path) -> None:
        out = fake_repo / "dist"
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(out)]) == 0
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(out), "--verify"]) == 0

    def test_verify_mode_reports_a_truncated_bundle_instead_of_crashing(
        self, mb: ModuleType, fake_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A 54 MB file that crossed an air gap on a USB stick is the likely casualty."""
        out = fake_repo / "dist"
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(out)]) == 0
        target = next(out.glob("rbi-ocr-code-*.zip"))
        target.write_bytes(target.read_bytes()[:-64])
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(out), "--verify"]) == 1
        assert "truncated or corrupted in transit" in capsys.readouterr().out

    def test_verify_mode_fails_when_nothing_has_been_built(self, mb: ModuleType, fake_repo: Path) -> None:
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(fake_repo / "empty"), "--verify"]) == 1

    def test_a_non_git_directory_is_refused(self, mb: ModuleType, tmp_path: Path) -> None:
        assert mb.main(["--repo-root", str(tmp_path)]) == 1

    def test_only_code_skips_the_assets_bundle(self, mb: ModuleType, fake_repo: Path) -> None:
        out = fake_repo / "dist"
        assert mb.main(["--repo-root", str(fake_repo), "--out", str(out), "--only", "code"]) == 0
        assert sorted(p.name.split("-")[2] for p in out.glob("rbi-ocr-*.zip")) == ["code"]

    def test_the_size_ceiling_is_enforced_from_the_command_line(self, mb: ModuleType, fake_repo: Path) -> None:
        exit_code = mb.main(
            ["--repo-root", str(fake_repo), "--out", str(fake_repo / "dist"), "--max-code-mb", "0.000001"]
        )
        assert exit_code == 1


# --------------------------------------------------------------------------- #
# The developer bundle — specifications without the deliberations
# --------------------------------------------------------------------------- #


class TestDeveloperBundle:
    """``--exclude-internal``: what a receiver needs, not how we argued to it."""

    def test_working_papers_are_dropped_and_specs_are_kept(
        self, mb: ModuleType, fake_repo: Path, small_requirements: None
    ) -> None:
        _write(fake_repo, "docs/decisions-wsz.md", "# how we argued\n")
        _write(fake_repo, "OPEN-QUESTIONS.md", "# the running log\n")
        _write(fake_repo, "PLAN.md", "# the build plan\n")
        _git(fake_repo, "add", "-A")
        _git(fake_repo, "commit", "-q", "-m", "working papers")

        report = mb.Report()
        target = mb.build_code_bundle(
            fake_repo,
            fake_repo / "dist",
            mb.read_git_info(fake_repo),
            report,
            exclude_internal=True,
        )
        assert report.ok, [p.detail for p in report.problems]
        assert target.name.startswith("rbi-ocr-code-dev-")

        members = _members(target)
        assert "docs/decisions-wsz.md" not in members
        assert "OPEN-QUESTIONS.md" not in members
        assert "PLAN.md" not in members
        # The specifications survive: a bundle without its own deployment
        # guide would be worse than one carrying a few stale arguments.
        assert "docs/AIR-GAPPED-DEPLOY.md" in members

        manifest = _manifest(target)
        assert manifest["audience"] == "developer"
        assert "PLAN.md" in manifest["internal_working_papers_excluded"]

    def test_the_full_bundle_still_carries_them(
        self, mb: ModuleType, fake_repo: Path, small_requirements: None
    ) -> None:
        """Excluding is opt-in; the reasoning is worth keeping by default."""
        _write(fake_repo, "docs/decisions-wsz.md", "# how we argued\n")
        _git(fake_repo, "add", "-A")
        _git(fake_repo, "commit", "-q", "-m", "working paper")

        report = mb.Report()
        target = mb.build_code_bundle(
            fake_repo, fake_repo / "dist", mb.read_git_info(fake_repo), report
        )
        assert target.name.startswith("rbi-ocr-code-")
        assert not target.name.startswith("rbi-ocr-code-dev-")
        assert "docs/decisions-wsz.md" in _members(target)
        assert _manifest(target)["audience"] == "full"

    def test_find_bundles_does_not_confuse_the_two(
        self, mb: ModuleType, tmp_path: Path
    ) -> None:
        """`rbi-ocr-code-dev-x.zip` also starts with `rbi-ocr-code-`.

        Without an explicit reject the glob for the full bundle returns the
        developer one, and --verify silently checks the wrong archive.
        """
        out = tmp_path / "dist"
        out.mkdir()
        (out / "rbi-ocr-code-abc1234.zip").write_bytes(b"full")
        (out / "rbi-ocr-code-dev-abc1234.zip").write_bytes(b"dev")

        code, _assets = mb.find_bundles(out, "abc1234")
        # The developer variant wins when both exist: it is the one being
        # handed over, so it is the one worth re-checking.
        assert code is not None and code.name == "rbi-ocr-code-dev-abc1234.zip"

        (out / "rbi-ocr-code-dev-abc1234.zip").unlink()
        code, _assets = mb.find_bundles(out, "abc1234")
        assert code is not None and code.name == "rbi-ocr-code-abc1234.zip"

    def test_a_required_document_can_never_be_excluded_by_accident(
        self, mb: ModuleType
    ) -> None:
        """The exclusion list may only ever name deliberations.

        Two entries (PLAN.md, OPEN-QUESTIONS.md) are deliberately in both
        lists and are dropped from the requirement when the exclusion is on.
        Anything else appearing in both would mean the developer bundle ships
        without a document the deploy guide tells its reader to open.
        """
        overlap = {
            p for p in mb.REQUIRED_CODE_PATHS if mb.is_internal_working_paper(p)
        }
        assert overlap == {"PLAN.md", "OPEN-QUESTIONS.md"}, overlap
