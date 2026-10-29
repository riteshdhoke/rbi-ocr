"""Fixtures for the WS-D corpus and evaluation-harness tests.

Local to ``tests/corpus/`` on purpose: the tools under test are standalone scripts in
``tools/``, not importable app packages, so they are put on ``sys.path`` here rather than
in a shared conftest that other workstreams also own.

Does NOT handle: importing anything from ``app/`` (the harness must work before the app
exists), network access, or model downloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
CORPUS_DIR = PROJECT_ROOT / "corpus"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_harness
import generate_test_corpus as gen


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    if not (CORPUS_DIR / "index.json").is_file():
        pytest.skip(
            "corpus/index.json not present — run "
            "tools/generate_test_corpus.py --out corpus --variants 5 --seed 1729"
        )
    return CORPUS_DIR


@pytest.fixture(scope="session")
def corpus_index(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "index.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def gen_module():
    return gen


@pytest.fixture(scope="session")
def harness():
    return eval_harness


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory) -> Path:
    """A small real corpus: 2 types x 5 variants x 1 profile, so the locked split exists."""
    out = tmp_path_factory.mktemp("mini_corpus") / "corpus"
    gen.generate_corpus(
        out,
        variants=5,
        seed=4242,
        types=["complaint_form_en", "complaint_letter_mar"],
        profiles=["clean"],
        adversarial=False,
    )
    return out
