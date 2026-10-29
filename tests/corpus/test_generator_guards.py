"""The generator's refusal paths, and the text-shaping machinery they protect.

FONTS.md is explicit: a corpus of tofu boxes that silently trains a recogniser is worse
than a corpus that failed to build. These tests hold the generator to that.

Does NOT handle: verifying the rendered glyphs are visually correct — that needs eyes on a
crop, and the shaping assertion below is the machine-checkable proxy for it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ------------------------------------------------------------------- shaping


def test_assets_dir_is_on_the_path_before_pil_is_used(gen_module):
    """OPEN-QUESTIONS D6: fribidi-0.dll is resolved through PATH at PIL import time."""
    assert (gen_module.ASSETS_DIR / "fribidi-0.dll").is_file()
    assert str(gen_module.ASSETS_DIR) in os.environ["PATH"].split(os.pathsep)


def test_raqm_is_available(gen_module):
    from PIL import features

    assert features.check("raqm") is True
    gen_module.require_text_shaping()  # must not raise


def test_shaping_guard_raises_with_an_explanation(gen_module, monkeypatch):
    monkeypatch.setattr(gen_module.features, "check", lambda name: False)
    with pytest.raises(gen_module.TextShapingUnavailableError) as exc:
        gen_module.require_text_shaping()
    message = str(exc.value)
    assert "raqm" in message
    assert "D6" in message
    assert "fribidi-0.dll" in message


def test_generate_corpus_refuses_without_shaping(gen_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gen_module.features, "check", lambda name: False)
    with pytest.raises(gen_module.TextShapingUnavailableError):
        gen_module.generate_corpus(tmp_path / "x", variants=1, types=["receipt"], profiles=["clean"])


# --------------------------------------------------------------------- fonts


def test_all_required_faces_are_present(gen_module):
    assert gen_module.missing_fonts() == {}
    gen_module.require_fonts()  # must not raise


def test_there_is_no_unix_fallback_left(gen_module):
    """The v1 generator fell back to /usr/share/fonts/.../DejaVuSans.ttf on Windows."""
    assert not hasattr(gen_module, "FALLBACK_LATIN")
    source = Path(gen_module.__file__).read_text(encoding="utf-8")
    assert "/usr/share/fonts" not in source
    assert "DejaVu" not in source


def test_missing_fonts_are_reported_by_name(gen_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gen_module, "FONT_DIR", tmp_path / "empty")
    missing = gen_module.missing_fonts()
    assert set(missing) == set(gen_module.FONTS)
    with pytest.raises(gen_module.FontUnavailableError) as exc:
        gen_module.require_fonts()
    message = str(exc.value)
    assert "NotoSansDevanagari-Regular.ttf" in message
    assert "Kalam-Regular.ttf" in message
    assert "Devanagari has NO fallback" in message


def test_devanagari_role_fails_loudly_rather_than_substituting(gen_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gen_module, "FONT_DIR", tmp_path / "empty")
    monkeypatch.setattr(gen_module, "_cache", {})
    with pytest.raises(gen_module.FontUnavailableError) as exc:
        gen_module.load_font("deva_print", 26)
    assert "no fallback" in str(exc.value)


def test_latin_role_also_fails_rather_than_guessing(gen_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gen_module, "FONT_DIR", tmp_path / "empty")
    monkeypatch.setattr(gen_module, "_cache", {})
    with pytest.raises(gen_module.FontUnavailableError) as exc:
        gen_module.load_font("latin_print", 26)
    assert "NotoSans-Regular.ttf" in str(exc.value)


def test_unknown_role_is_typed(gen_module):
    with pytest.raises(gen_module.FontUnavailableError):
        gen_module.load_font("klingon_print", 26)


# ---------------------------------------------------------- grapheme clusters


def test_latin_clusters_are_plain_characters(gen_module):
    for text in ("Bank of Baroda", "BARB0417293", "Rs. 45,000"):
        assert gen_module.grapheme_clusters(text) == list(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("बैंक", ["बैं", "क"]),  # base + vowel sign + anusvara
        ("शर्मा", ["श", "र्मा"]),  # virama joins the following consonant, matra follows
        ("महाराष्ट्र", ["म", "हा", "रा", "ष्ट्र"]),  # a three-consonant conjunct
        ("बड़ौदा", ["ब", "ड़ौ", "दा"]),  # nukta stays with its base
    ],
)
def test_devanagari_clusters_keep_marks_with_their_base(gen_module, text, expected):
    assert gen_module.grapheme_clusters(text) == expected


def test_clusters_never_lose_or_reorder_codepoints(gen_module):
    for text in ("राजेश कुमार शर्मा", "पनवेल, महाराष्ट्र", "Rs. 45,000 only"):
        assert "".join(gen_module.grapheme_clusters(text)) == text


def test_no_cluster_is_a_bare_combining_mark(gen_module):
    """A lone matra renders as a dotted-circle placeholder — the Devanagari tofu."""
    import unicodedata

    for text in ("बैंकिंग लोकपाल को शिकायत", "मेरे बचत खाते से अनधिकृत"):
        for cluster in gen_module.grapheme_clusters(text):
            assert unicodedata.category(cluster[0]) not in {"Mn", "Mc", "Me"}, cluster


def test_codepoint_mode_reproduces_the_v1_behaviour(gen_module):
    """The escape hatch must still exist, and must still be the broken one."""
    assert set(gen_module.HAND_SHAPING_MODES) == {"cluster", "codepoint"}
    pen_cluster = gen_module.Pen(rng=__import__("random").Random(0), hand_shaping="cluster")
    pen_codepoint = gen_module.Pen(rng=__import__("random").Random(0), hand_shaping="codepoint")
    canvases = []
    for pen in (pen_cluster, pen_codepoint):
        canvas = gen_module.Canvas(w=400, h=120, pen=pen)
        canvas.text((10, 10), "बैंक ऑफ बड़ौदा", role="deva_hand", size=32, script="Deva",
                    lang="mar", hand=True, jitter=2.0)
        canvases.append(canvas.img.tobytes())
    assert canvases[0] != canvases[1]


# ------------------------------------------------------------ argument guards


def test_unknown_type_is_typed(gen_module, tmp_path: Path):
    with pytest.raises(gen_module.UnknownDocumentTypeError) as exc:
        gen_module.generate_corpus(tmp_path / "o", types=["not_a_type"], variants=1)
    assert "not_a_type" in str(exc.value)


def test_unknown_profile_is_typed(gen_module, tmp_path: Path):
    with pytest.raises(gen_module.CorpusGenerationError):
        gen_module.generate_corpus(tmp_path / "o", types=["receipt"], profiles=["pristine"], variants=1)


def test_zero_variants_is_typed(gen_module, tmp_path: Path):
    with pytest.raises(gen_module.CorpusGenerationError):
        gen_module.generate_corpus(tmp_path / "o", types=["receipt"], variants=0)


def test_unknown_hand_shaping_is_typed(gen_module, tmp_path: Path):
    with pytest.raises(gen_module.CorpusGenerationError):
        gen_module.generate_corpus(tmp_path / "o", types=["receipt"], variants=1, hand_shaping="scribble")


def test_all_typed_errors_share_one_base(gen_module):
    for error in (
        gen_module.FontUnavailableError,
        gen_module.TextShapingUnavailableError,
        gen_module.UnknownDocumentTypeError,
    ):
        assert issubclass(error, gen_module.CorpusGenerationError)


def test_cli_reports_a_bad_type_without_a_traceback(gen_module, tmp_path: Path, capsys):
    code = gen_module.main(["--out", str(tmp_path / "o"), "--types", "nope", "--variants", "1"])
    assert code == 2
    assert "UnknownDocumentTypeError" in capsys.readouterr().err


def test_cli_generates_and_reports(gen_module, tmp_path: Path, capsys):
    out = tmp_path / "cli"
    code = gen_module.main(
        ["--out", str(out), "--types", "receipt", "--variants", "1", "--profiles", "clean",
         "--no-adversarial"]
    )
    assert code == 0
    assert (out / "index.json").is_file()
    assert not (out / "adversarial").exists()
    assert "1 pages" in capsys.readouterr().out


# --------------------------------------------------------------- file writing


def test_every_text_artefact_is_utf8(gen_module, tmp_path: Path):
    out = tmp_path / "enc"
    gen_module.generate_corpus(
        out, variants=1, seed=5, types=["complaint_letter_mar"], profiles=["clean"], adversarial=True
    )
    for path in (
        out / "index.json",
        out / "adversarial" / "MANIFEST.json",
        out / "complaint_letter_mar" / "complaint_letter_mar_v0_clean.gt.json",
    ):
        path.read_text(encoding="utf-8")  # raises UnicodeDecodeError if written as cp1252
