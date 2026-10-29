"""The corpus must regenerate byte-identically from its seed.

TEST-DATA-SPEC section 8: seeding is per (seed, type, variant, profile), so any single
document regenerates without touching the rest. That property is what makes a CI failure
pointing at one page actionable, and it is the reason the process-global ``random.seed``
call in the v1 generator had to go.

Does NOT handle: cross-platform reproducibility (libjpeg and FreeType versions differ
between machines; the guarantee here is same-machine, same-environment).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

SINGLE_DOC_CASES = [
    ("complaint_form_deva", "clean"),  # printed Devanagari, no jitter
    ("bilingual_form", "clean"),  # handwritten Devanagari + Latin, heavy jitter
    ("cheque", "scan"),  # handwriting plus geometric degradation
]


@pytest.mark.parametrize("doc_type,profile", SINGLE_DOC_CASES)
def test_single_document_reproduces_committed_corpus(
    gen_module, corpus_dir: Path, tmp_path: Path, doc_type: str, profile: str
):
    """Regenerating one document alone must match the committed bytes exactly."""
    out = tmp_path / "one"
    gen_module.generate_corpus(
        out, variants=1, seed=1729, types=[doc_type], profiles=[profile], adversarial=False
    )
    doc_id = f"{doc_type}_v0_{profile}"
    for suffix in (".jpg", ".gt.json"):
        fresh = (out / doc_type / f"{doc_id}{suffix}").read_bytes()
        committed = (corpus_dir / doc_type / f"{doc_id}{suffix}").read_bytes()
        assert fresh == committed, f"{doc_id}{suffix} is not reproducible from seed 1729"


def test_two_runs_are_identical(gen_module, tmp_path: Path):
    kwargs = {
        "variants": 2,
        "seed": 99,
        "types": ["complaint_letter_mar", "receipt"],
        "profiles": ["clean", "heavy"],
        "adversarial": False,
    }
    first, second = tmp_path / "a", tmp_path / "b"
    gen_module.generate_corpus(first, **kwargs)
    gen_module.generate_corpus(second, **kwargs)

    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert len(files) > 8
    for rel in files:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_generation_does_not_touch_the_global_random_state(gen_module, tmp_path: Path):
    """The v1 generator called random.seed() per document. Nothing may do that now."""
    random.seed(20260807)
    before = random.getstate()
    expected = [random.random() for _ in range(3)]

    random.seed(20260807)
    gen_module.generate_corpus(
        tmp_path / "c", variants=1, seed=7, types=["bilingual_form"], profiles=["clean"],
        adversarial=False,
    )
    after_state_seed = random.getstate()
    assert after_state_seed == before
    assert [random.random() for _ in range(3)] == expected


def test_variant_slice_is_independent_of_the_other_variants(gen_module, tmp_path: Path):
    """Generating variants 0..2 must produce the same v0 bytes as generating 0..0."""
    narrow, wide = tmp_path / "n", tmp_path / "w"
    gen_module.generate_corpus(
        narrow, variants=1, seed=555, types=["complaint_form_en"], profiles=["light"], adversarial=False
    )
    gen_module.generate_corpus(
        wide, variants=3, seed=555, types=["complaint_form_en"], profiles=["light"], adversarial=False
    )
    name = "complaint_form_en_v0_light.jpg"
    assert (narrow / "complaint_form_en" / name).read_bytes() == (wide / "complaint_form_en" / name).read_bytes()


def test_profile_slice_is_independent_of_the_other_profiles(gen_module, tmp_path: Path):
    one, many = tmp_path / "p1", tmp_path / "p2"
    gen_module.generate_corpus(
        one, variants=1, seed=606, types=["cheque"], profiles=["bounce"], adversarial=False
    )
    gen_module.generate_corpus(
        many, variants=1, seed=606, types=["cheque"], profiles=["clean", "scan", "bounce"], adversarial=False
    )
    name = "cheque_v0_bounce.jpg"
    assert (one / "cheque" / name).read_bytes() == (many / "cheque" / name).read_bytes()


def test_different_seed_changes_the_output(gen_module, tmp_path: Path):
    a, b = tmp_path / "s1", tmp_path / "s2"
    for out, seed in ((a, 1), (b, 2)):
        gen_module.generate_corpus(
            out, variants=1, seed=seed, types=["complaint_form_en"], profiles=["scan"], adversarial=False
        )
    name = "complaint_form_en/complaint_form_en_v0_scan.jpg"
    assert (a / name).read_bytes() != (b / name).read_bytes()


def test_index_records_the_settings_used(gen_module, tmp_path: Path):
    out = tmp_path / "idx"
    payload = gen_module.generate_corpus(
        out, variants=2, seed=31337, types=["receipt"], profiles=["clean", "photo"], adversarial=False
    )
    on_disk = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert on_disk == payload
    assert on_disk["seed"] == 31337
    assert on_disk["variants"] == 2
    assert on_disk["types"] == ["receipt"]
    assert on_disk["profiles"] == ["clean", "photo"]
    assert on_disk["adversarial"] == []
    assert on_disk["count"] == 4
