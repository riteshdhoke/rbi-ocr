#!/usr/bin/env python3
"""generate_test_corpus.py — synthetic RBI-complaint corpus with ground truth by construction.

Renders documents from text we control, then physically degrades them. Because every
string is drawn at a known position, per-line ground truth falls out for free: no human
labelling, fully reproducible from a seed, versionable in git.

    .venv\\Scripts\\python.exe tools/generate_test_corpus.py --out corpus --variants 5 --seed 1729
    .venv\\Scripts\\python.exe tools/generate_test_corpus.py --out corpus --types complaint_form_deva

This is the FIXED copy of ``design/generate_test_corpus.py`` (WS-D). Layout geometry,
content pools and degradation maths are unchanged from v1; the changes are Windows
safety, encoding, typed failures, removal of a process-global RNG side effect, and
grapheme-cluster-aware handwriting so Devanagari is not drawn as loose codepoints.
See ``docs/decisions-wsd.md`` for the full change list and the rationale for each.

Ground-truth field metadata (v1.2.0, OPEN-QUESTIONS Q6)
-------------------------------------------------------
Each ``*.gt.json`` carries two field blocks, and they answer different questions:

``fields``
    The document's *facts*, in their canonical spelling — the English/ASCII form, exactly
    as v1.1.0 wrote it. Values stay plain scalars, so every existing consumer
    (``tools/slm_bakeoff.py``, ``tools/eval_harness.py``) keeps working unchanged.
    Historically it also carried keys that were never drawn on the page; the four thin
    renderers (``receipt``, ``email_printout``, ``cheque``, ``photo_evidence``) now emit
    one, and every key in it for those four is a key the page really shows.

``fields_bilingual`` (new, additive)
    ``{field_name: {"value": ..., "value_en": ..., "script": ...}}`` where **value is the
    string this page actually drew, verbatim**, and ``value_en`` is its English
    equivalent taken from the parallel fact pool. On a Latin page the two are equal. It
    is built from the draw calls themselves (``Canvas.text(fld=..., fld_en=...)``), so a
    field can only appear here if the renderer really rendered it — a key here is a
    promise about pixels, not a note about intent.

    First draw wins when a field is drawn twice (the cheque writes its amount in the
    words slot and again in the numeric box); the second spelling still reaches a scorer
    through the ``lines[].field`` link.

The split exists because ``fields`` alone cannot say what a Devanagari page shows: it
holds ``"Bank of Baroda"`` for a page that reads ``बैंक ऑफ बड़ौदा``. Scoring an
immutable native-script ``value`` against an English reference manufactures either false
negatives or false credit, depending on which way the harness leans.

Fonts (see design/FONTS.md — not vendored, licences differ) live in ``tools/assets/fonts/``:

    NotoSans-Regular.ttf / -Medium.ttf / -Bold.ttf   Latin print
    NotoSansDevanagari-Regular.ttf / -Bold.ttf        Devanagari print
    Kalam-Regular.ttf                                 handwriting, Devanagari + Latin
    Caveat-Regular.ttf                                handwriting, Latin
    NotoSansMono-Regular.ttf                          statement / MICR lines

There is no font fallback of any kind. A missing face raises ``FontUnavailableError``
naming the exact files, because a corpus full of tofu boxes that silently trains a
recogniser is worse than a corpus that failed to build.

Text shaping: Pillow's Windows wheel ships Raqm but resolves ``fribidi-0.dll`` through the
process DLL search path at first use. ``tools/assets`` is prepended to ``PATH`` *before*
PIL is imported and ``features.check("raqm")`` is asserted before any page is rendered
(OPEN-QUESTIONS D6). Without Raqm, Devanagari renders as unshaped, unreordered codepoints.

What this module does NOT handle
--------------------------------
* **Real handwriting.** Handwriting is a handwriting-styled font plus per-cluster jitter.
  Any CER claim for handwritten Indic must be measured on ``iiit-indic-hw-words`` and on
  real complaint pages. The caveat is written into ``index.json`` so it travels with the data.
* **Inter-cluster shaping on handwritten lines.** Jitter is applied per grapheme cluster,
  so each cluster is shaped correctly but kerning/ligation *across* clusters is lost.
  Printed lines are drawn as whole strings and are fully shaped.
* **Scripts other than Latin and Devanagari.** No Kannada/Bengali/Tamil pools exist yet.
* **PDF output.** Pages are JPEG only. The digital-PDF bypass path (s1b) needs its own
  fixtures; the adversarial set covers malformed PDFs but not valid multi-page ones.
* **Multi-page documents.** One image == one document == one ground-truth sidecar.
* **Layout realism beyond a fixed template.** Row pitch, margins and box geometry are
  hard-coded per renderer; there is no layout randomisation across variants.
* **Licence review of the fonts.** ``design/FONTS.md`` owns that; this module only loads them.
* **Evaluation.** Scoring predictions against this corpus is ``tools/eval_harness.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------- shaping DLL path
# MUST run before PIL is imported: Pillow resolves fribidi-0.dll through the process
# DLL search path the first time _imagingft is loaded (OPEN-QUESTIONS D6).

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FONT_DIR = ASSETS_DIR / "fonts"


def _prepend_assets_to_dll_path() -> None:
    """Make ``tools/assets`` discoverable for dynamically loaded DLLs (fribidi-0.dll)."""
    if not ASSETS_DIR.is_dir():
        return
    assets = str(ASSETS_DIR)
    current = os.environ.get("PATH", "")
    if assets not in current.split(os.pathsep):
        os.environ["PATH"] = assets + os.pathsep + current
    # Python 3.8+ on Windows narrows the search path for ctypes/extension loads; PATH is
    # what Pillow's own LoadLibrary call honours, but adding both costs nothing.
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(assets)
        except OSError:
            pass  # non-fatal: PATH is the mechanism that actually matters here


_prepend_assets_to_dll_path()

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    features,
)

GENERATOR_VERSION = "1.2.0"

# ------------------------------------------------------------------ typed errors


class CorpusGenerationError(RuntimeError):
    """Base class for every failure this generator raises at a boundary."""


class FontUnavailableError(CorpusGenerationError):
    """A required font face is missing from tools/assets/fonts. There is no fallback."""


class TextShapingUnavailableError(CorpusGenerationError):
    """Pillow cannot shape complex scripts (Raqm/fribidi unavailable)."""


class UnknownDocumentTypeError(CorpusGenerationError):
    """A --types value that no renderer implements."""


# --------------------------------------------------------------------------- fonts

FONTS: dict[str, list[str]] = {
    "latin_print": ["NotoSans-Regular.ttf", "NotoSans-Medium.ttf"],
    "latin_bold": ["NotoSans-Bold.ttf"],
    "deva_print": ["NotoSansDevanagari-Regular.ttf"],
    "deva_bold": ["NotoSansDevanagari-Bold.ttf"],
    "latin_hand": ["Caveat-Regular.ttf", "Kalam-Regular.ttf"],
    "deva_hand": ["Kalam-Regular.ttf"],
    "mono": ["NotoSansMono-Regular.ttf"],
}

_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def missing_fonts() -> dict[str, list[str]]:
    """Return ``{role: [candidate filenames]}`` for every role with no usable face."""
    missing: dict[str, list[str]] = {}
    for role, names in FONTS.items():
        if not any((FONT_DIR / n).exists() for n in names):
            missing[role] = list(names)
    return missing


def require_fonts() -> None:
    """Fail once, up front, naming every missing file. Raises FontUnavailableError."""
    missing = missing_fonts()
    if not missing:
        return
    detail = "\n".join(
        f"  role {role!r:<14} needs one of: {', '.join(names)}"
        for role, names in sorted(missing.items())
    )
    extra = (
        "\nDevanagari has NO fallback. Refusing to render Devanagari with a substitute "
        "face — wrong glyphs are worse than no document."
        if any(r.startswith("deva") for r in missing)
        else ""
    )
    raise FontUnavailableError(
        f"Missing font faces in {FONT_DIR}\n{detail}\n"
        f"Fetch them with tools/fetch_fonts.py (see design/FONTS.md).{extra}"
    )


def require_text_shaping() -> None:
    """Assert Pillow can shape complex scripts. Raises TextShapingUnavailableError."""
    if features.check("raqm"):
        return
    import PIL

    raise TextShapingUnavailableError(
        "Pillow reports features.check('raqm') is False, so Devanagari would be rendered "
        "as unshaped, unreordered codepoints (matras detached, conjuncts broken).\n"
        f"Expected fribidi-0.dll in {ASSETS_DIR}, and that directory prepended to PATH "
        "before PIL was imported.\n"
        f"  fribidi-0.dll present: {(ASSETS_DIR / 'fribidi-0.dll').exists()}\n"
        f"  Pillow version:        {PIL.__version__}\n"
        "See OPEN-QUESTIONS.md D6. Refusing to generate a Devanagari corpus without shaping."
    )


def load_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a face for ``role`` at ``size``. No fallback — raises FontUnavailableError."""
    key = (role, size)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    if role not in FONTS:
        raise FontUnavailableError(
            f"No font role named {role!r}. Known roles: {', '.join(sorted(FONTS))}"
        )
    for name in FONTS[role]:
        p = FONT_DIR / name
        if p.exists():
            font = ImageFont.truetype(str(p), size)
            _cache[key] = font
            return font
    hard = (
        " Devanagari has no fallback: wrong glyphs are worse than no document."
        if role.startswith("deva")
        else ""
    )
    raise FontUnavailableError(
        f"No usable font for role {role!r} in {FONT_DIR}; expected one of: "
        f"{', '.join(FONTS[role])}.{hard} See design/FONTS.md."
    )


# ------------------------------------------------------------------ text clusters

_VIRAMA = "्"  # DEVANAGARI SIGN VIRAMA — joins the following consonant (conjuncts)
_ZERO_WIDTH = "‌‍"  # ZWNJ, ZWJ
_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})

HAND_SHAPING_MODES = ("cluster", "codepoint")


def is_devanagari(s: str) -> bool:
    """True if ``s`` contains any codepoint in the Devanagari block (U+0900-U+097F)."""
    return any("ऀ" <= ch <= "ॿ" for ch in s)


def grapheme_clusters(s: str) -> list[str]:
    """Split ``s`` into renderable units: a base character plus everything that hangs off it.

    Devanagari matras, anusvara, nukta and virama-joined conjuncts stay attached to their
    base, so each unit can be drawn with a single shaped ``ImageDraw.text`` call. For text
    with no combining marks (all Latin content here) this returns ``list(s)``, which is
    exactly what the v1 per-codepoint loop did.

    Deliberately simpler than UAX #29: it covers Devanagari and Latin, which is all this
    corpus contains. It does NOT handle emoji ZWJ sequences, regional indicators, or
    Indic scripts with reordering rules Devanagari does not share.
    """
    clusters: list[str] = []
    for ch in s:
        if not clusters:
            clusters.append(ch)
            continue
        prev = clusters[-1]
        joins = (
            unicodedata.category(ch) in _MARK_CATEGORIES
            or ch in _ZERO_WIDTH
            or prev.endswith(_VIRAMA)
            or prev[-1] in _ZERO_WIDTH
        )
        if joins:
            clusters[-1] = prev + ch
        else:
            clusters.append(ch)
    return clusters


@dataclass
class Pen:
    """Everything about *how* handwriting is drawn, threaded explicitly rather than global.

    ``rng`` is the per-document jitter stream. It is a distinct object from the content
    RNG so that a change to one cannot shift the other, and it is passed in rather than
    taken from the ``random`` module so generating a document has no process-wide effect.
    """

    rng: random.Random
    hand_shaping: str = "cluster"  # "cluster" (shaped) | "codepoint" (v1 behaviour)


# ----------------------------------------------------------------------- gt model


@dataclass
class GTLine:
    text: str
    bbox: list[float]  # [x0,y0,x1,y1,x2,y2,x3,y3] clockwise from top-left
    script: str  # ISO 15924: Latn | Deva
    language: str  # ISO 639-3: eng | hin | mar
    is_handwritten: bool
    field: str | None = None  # links the line to an extraction field


@dataclass
class GTDoc:
    doc_id: str
    doc_type: str
    variant: int
    synthetic: bool = True
    document_class: str = "complaint_form"
    primary_script: str = "Latn"
    primary_language: str = "eng"
    is_code_mixed: bool = False
    degradations: list[str] = field(default_factory=list)
    quality_expectation: str = "clean"  # clean | degraded | bounce
    lines: list[GTLine] = field(default_factory=list)
    #: canonical facts, English/ASCII spelling, plain scalars — v1.1.0-compatible
    fields: dict = field(default_factory=dict)
    #: field name -> {"value": as drawn, "value_en": English equivalent, "script": ...}
    fields_bilingual: dict = field(default_factory=dict)


class Canvas:
    """A page that records the ground truth of everything drawn on it."""

    def __init__(
        self,
        w: int = 1654,
        h: int = 2339,
        bg: tuple[int, int, int] = (252, 251, 247),
        *,
        pen: Pen,
    ) -> None:  # A4 @ 200 dpi
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img)
        self.lines: list[GTLine] = []
        #: field name -> {"value" (as drawn), "value_en", "script"}; see module docstring.
        #: Populated only by draw calls, so it can never claim a field the page lacks.
        self.field_values: dict[str, dict[str, str]] = {}
        self.pen = pen

    def text(
        self,
        xy,
        s,
        *,
        role="latin_print",
        size=26,
        script="Latn",
        lang="eng",
        hand=False,
        fld=None,
        fld_en=None,
        fld_value=None,
        fill=(20, 24, 32),
        jitter=0.0,
        record=True,
    ):
        """Draw ``s`` and record it as ground truth.

        ``fld`` links the line to an extraction field. ``fld_value`` is the field's own
        substring when the line carries more than the field (``"My account number is
        12345678901."`` links ``account_number``, but the *value* is the number);
        it defaults to the whole drawn string. ``fld_en`` is the English equivalent of
        that value, from the parallel fact pool — pass it only when the value is not
        already English.

        None of the three affects a single pixel: they exist so the ground truth can say
        what was drawn.
        """
        if not s:
            return 0
        x, y = xy
        f = load_font(role, size)
        if hand and jitter:
            # Per-unit wobble breaks the giveaway regularity of font-rendered "handwriting".
            # A unit is a grapheme cluster, not a codepoint: drawing a bare matra or virama
            # on its own produces a dotted-circle placeholder, which is the Devanagari
            # equivalent of tofu and would make the handwritten Deva ground truth a lie.
            units = list(s) if self.pen.hand_shaping == "codepoint" else grapheme_clusters(s)
            rng = self.pen.rng
            cx = x
            for unit in units:
                dy = rng.uniform(-jitter, jitter)
                self.d.text((cx, y + dy), unit, font=f, fill=fill)
                cx += f.getlength(unit) + rng.uniform(-0.4, 0.9)
            w = cx - x
        else:
            self.d.text((x, y), s, font=f, fill=fill)
            w = f.getlength(s)
        h = size * 1.28
        if record:
            self.lines.append(
                GTLine(
                    text=s,
                    bbox=[x, y, x + w, y, x + w, y + h, x, y + h],
                    script=script,
                    language=lang,
                    is_handwritten=hand,
                    field=fld,
                )
            )
            # First draw wins: the cheque writes its amount twice (words slot, then the
            # numeric box) and both are legitimate spellings, but this dict holds one
            # canonical pair per field. The other spelling still reaches a consumer
            # through the line's own `field` link.
            if fld and fld not in self.field_values:
                value = s if fld_value is None else fld_value
                self.field_values[fld] = {
                    "value": value,
                    "value_en": value if fld_en is None else fld_en,
                    # The *value's* script, which is not always the line's: a Marathi
                    # sentence carries an ASCII account number, and labelling that
                    # number Deva would tell a scorer to expect Devanagari digits.
                    "script": (
                        "Deva"
                        if is_devanagari(value)
                        else ("Latn" if script == "Deva" else script)
                    ),
                }
        return w

    def rule(self, x0, y0, x1, y1, w=2, fill=(120, 125, 135)):
        self.d.line([x0, y0, x1, y1], fill=fill, width=w)

    def box(self, x0, y0, x1, y1, w=2, fill=(90, 95, 105)):
        self.d.rectangle([x0, y0, x1, y1], outline=fill, width=w)


# ------------------------------------------------------------------- content pool

BANKS = [
    ("State Bank of India", "स्टेट बैंक ऑफ इंडिया", "SBIN"),
    ("Punjab National Bank", "पंजाब नेशनल बैंक", "PUNB"),
    ("HDFC Bank Limited", "एचडीएफसी बैंक लिमिटेड", "HDFC"),
    ("Bank of Baroda", "बैंक ऑफ बड़ौदा", "BARB"),
    ("Union Bank of India", "यूनियन बैंक ऑफ इंडिया", "UBIN"),
]
NAMES_EN = [
    "Rajesh Kumar Sharma",
    "Priya Venkatesan",
    "Abdul Rahman Shaikh",
    "Meera Joshi",
    "Sanjay Deshpande",
]
NAMES_DEVA = [
    "राजेश कुमार शर्मा",
    "प्रिया वेंकटेशन",
    "अब्दुल रहमान शेख",
    "मीरा जोशी",
    "संजय देशपांडे",
]
CITIES = [
    ("Panvel, Maharashtra", "पनवेल, महाराष्ट्र"),
    ("Lucknow, Uttar Pradesh", "लखनऊ, उत्तर प्रदेश"),
    ("Nagpur, Maharashtra", "नागपुर, महाराष्ट्र"),
    ("Patna, Bihar", "पटना, बिहार"),
    ("Indore, Madhya Pradesh", "इंदौर, मध्य प्रदेश"),
]
GROUNDS_EN = [
    "Unauthorised electronic transaction debited from my savings account",
    "Credit card annual fee levied without prior consent or intimation",
    "Failure to credit the amount of a failed ATM withdrawal within the prescribed period",
    "Non-adherence to the declared interest rate on a fixed deposit at renewal",
    "Delay in closure of loan account and release of original property documents",
]
GROUNDS_DEVA = [
    "मेरे बचत खाते से अनधिकृत इलेक्ट्रॉनिक लेनदेन डेबिट किया गया",
    "पूर्व सहमति के बिना क्रेडिट कार्ड वार्षिक शुल्क लगाया गया",
    "असफल एटीएम निकासी की राशि निर्धारित अवधि में जमा नहीं की गई",
    "सावधि जमा नवीनीकरण पर घोषित ब्याज दर का पालन नहीं किया गया",
    "ऋण खाता बंद करने और मूल संपत्ति दस्तावेज़ लौटाने में देरी",
]


def make_facts(rng: random.Random, v: int) -> dict:
    bank_en, bank_hi, code = BANKS[v % len(BANKS)]
    city_en, city_hi = CITIES[v % len(CITIES)]
    return {
        "bank_name": bank_en,
        "bank_name_deva": bank_hi,
        "ifsc_code": f"{code}0{rng.randint(100000, 999999)}",
        "account_number": "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([11, 14, 16]))),
        "card_last_four": f"{rng.randint(1000, 9999)}",
        "disputed_amount": rng.choice([4500, 12750, 45000, 98200, 250000]),
        "transaction_date": f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026",
        "transaction_reference": f"{rng.randint(10**11, 10**12 - 1)}",
        "complainant_name": NAMES_EN[v % len(NAMES_EN)],
        "complainant_name_deva": NAMES_DEVA[v % len(NAMES_DEVA)],
        "complainant_city": city_en,
        "complainant_city_deva": city_hi,
        "mobile": f"9{rng.randint(10**8, 10**9 - 1)}",
        "prior_complaint_ref": f"CMS/{rng.randint(2025, 2026)}/{rng.randint(100000, 999999)}",
        "ground_en": GROUNDS_EN[v % len(GROUNDS_EN)],
        "ground_deva": GROUNDS_DEVA[v % len(GROUNDS_DEVA)],
    }


# --------------------------------------------------------------------- renderers


def _form_header(c: Canvas, deva: bool):
    c.box(70, 70, 1584, 235, w=3)
    if deva:
        c.text(
            (110, 100), "बैंकिंग लोकपाल को शिकायत", role="deva_bold", size=40, script="Deva", lang="hin"
        )
        c.text(
            (110, 160),
            "एकीकृत लोकपाल योजना 2021 — प्रपत्र I",
            role="deva_print",
            size=26,
            script="Deva",
            lang="hin",
        )
    else:
        c.text((110, 100), "COMPLAINT TO THE BANKING OMBUDSMAN", role="latin_bold", size=38)
        c.text(
            (110, 160),
            "Reserve Bank — Integrated Ombudsman Scheme, 2021 — Form I",
            role="latin_print",
            size=25,
        )


def render_complaint_form(
    f: dict, deva: bool = False, hand_values: bool = False, rng=None, *, pen: Pen
) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(pen=pen)
    _form_header(c, deva)
    # rows: (label, value as drawn, field name, script, language, English equivalent).
    # The last element is None whenever the drawn value is already English.
    rows = (
        [
            ("1. शिकायतकर्ता का नाम", f["complainant_name_deva"], "complainant_name", "Deva", "hin", f["complainant_name"]),
            ("2. पता", f["complainant_city_deva"], "complainant_address", "Deva", "hin", f["complainant_city"]),
            ("3. मोबाइल", f["mobile"], "complainant_contact", "Latn", "eng", None),
            ("4. बैंक का नाम", f["bank_name_deva"], "bank_name", "Deva", "hin", f["bank_name"]),
            ("5. IFSC कोड", f["ifsc_code"], "ifsc_code", "Latn", "eng", None),
            ("6. खाता संख्या", f["account_number"], "account_number", "Latn", "eng", None),
            ("7. विवादित राशि (₹)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng", None),
            ("8. लेनदेन दिनांक", f["transaction_date"], "transaction_date", "Latn", "eng", None),
            ("9. लेनदेन संदर्भ", f["transaction_reference"], "transaction_reference", "Latn", "eng", None),
            ("10. पूर्व शिकायत सं.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng", None),
        ]
        if deva
        else [
            ("1. Name of complainant", f["complainant_name"], "complainant_name", "Latn", "eng", None),
            ("2. Address", f["complainant_city"], "complainant_address", "Latn", "eng", None),
            ("3. Mobile", f["mobile"], "complainant_contact", "Latn", "eng", None),
            ("4. Name of bank", f["bank_name"], "bank_name", "Latn", "eng", None),
            ("5. IFSC code", f["ifsc_code"], "ifsc_code", "Latn", "eng", None),
            ("6. Account number", f["account_number"], "account_number", "Latn", "eng", None),
            ("7. Amount in dispute (Rs.)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng", None),
            ("8. Date of transaction", f["transaction_date"], "transaction_date", "Latn", "eng", None),
            ("9. Transaction reference", f["transaction_reference"], "transaction_reference", "Latn", "eng", None),
            ("10. Previous complaint no.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng", None),
        ]
    )

    lbl_role = "deva_print" if deva else "latin_print"
    lbl_script, lbl_lang = ("Deva", "hin") if deva else ("Latn", "eng")
    y = 300
    for label, value, fld, vscript, vlang, value_en in rows:
        c.text((110, y), label, role=lbl_role, size=25, script=lbl_script, lang=lbl_lang)
        c.box(700, y - 10, 1584, y + 50, w=2)
        if hand_values:
            vrole = "deva_hand" if vscript == "Deva" else "latin_hand"
            c.text(
                (720, y - 2),
                value,
                role=vrole,
                size=32,
                script=vscript,
                lang=vlang,
                hand=True,
                fld=fld,
                fld_en=value_en,
                jitter=2.4,
                fill=(28, 38, 96),
            )
        else:
            vrole = "deva_print" if vscript == "Deva" else "latin_print"
            c.text(
                (720, y), value, role=vrole, size=26, script=vscript, lang=vlang, fld=fld, fld_en=value_en
            )
        y += 78

    gy = y + 40
    grounds_label = "11. शिकायत का विवरण" if deva else "11. Nature of complaint"
    c.text((110, gy), grounds_label, role=lbl_role, size=25, script=lbl_script, lang=lbl_lang)
    c.box(110, gy + 46, 1584, gy + 340, w=2)
    body = f["ground_deva"] if deva else f["ground_en"]
    if hand_values:
        role = "deva_hand" if deva else "latin_hand"
        c.text(
            (140, gy + 76),
            body,
            role=role,
            size=32,
            script="Deva" if deva else "Latn",
            lang="hin" if deva else "eng",
            hand=True,
            fld="complaint_summary",
            fld_en=f["ground_en"] if deva else None,
            jitter=2.6,
            fill=(28, 38, 96),
        )
    else:
        c.text(
            (140, gy + 76),
            body,
            role=lbl_role,
            size=26,
            script=lbl_script,
            lang=lbl_lang,
            fld="complaint_summary",
            fld_en=f["ground_en"] if deva else None,
        )

    sig_y = gy + 420
    c.rule(110, sig_y, 620, sig_y)
    sig_label = "हस्ताक्षर" if deva else "Signature"
    c.text((110, sig_y + 12), sig_label, role=lbl_role, size=22, script=lbl_script, lang=lbl_lang)

    gt = GTDoc(
        doc_id="",
        doc_type="",
        variant=0,
        document_class="complaint_form",
        primary_script="Deva" if deva else "Latn",
        primary_language="hin" if deva else "eng",
    )
    return c, gt, {k: v for k, v in f.items()}


def render_complaint_letter(
    f: dict, deva: bool = False, hand: bool = True, *, pen: Pen
) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(pen=pen)
    role = ("deva_hand" if deva else "latin_hand") if hand else ("deva_print" if deva else "latin_print")
    script, lang = ("Deva", "mar") if deva else ("Latn", "eng")
    size = 34 if hand else 27
    jit = 2.6 if hand else 0.0
    y = 160

    head = (
        ["सेवेत,", "मा. बँकिंग लोकपाल,", "भारतीय रिझर्व्ह बँक"]
        if deva
        else ["To,", "The Banking Ombudsman,", "Reserve Bank of India"]
    )
    for ln in head:
        c.text((150, y), ln, role=role, size=size, script=script, lang=lang, hand=hand, jitter=jit)
        y += size + 22

    y += 30
    subj_en = f"Subject: {f['ground_en']}"
    subj = f"विषय: {f['ground_deva']}" if deva else subj_en
    c.text(
        (150, y),
        subj[:78],
        role=role,
        size=size,
        script=script,
        lang=lang,
        hand=hand,
        jitter=jit,
        fld="complaint_summary",
        fld_en=subj_en[:78] if deva else None,
    )
    y += size + 40

    # The two paragraph lists are parallel line for line, so ``para_en[i]`` is the
    # English equivalent of ``para_deva[i]`` and can be published as ``value_en``.
    para_deva = [
        f"माझे {f['bank_name_deva']} येथे बचत खाते आहे.",
        f"खाते क्रमांक {f['account_number']} आहे.",
        f"दिनांक {f['transaction_date']} रोजी रु. {f['disputed_amount']:,} रक्कम",
        "माझ्या संमतीशिवाय डेबिट करण्यात आली.",
        f"संदर्भ क्रमांक {f['transaction_reference']}.",
        "बँकेकडे तक्रार करूनही निराकरण झाले नाही.",
        "कृपया योग्य ती कार्यवाही करावी ही विनंती.",
    ]
    para_en = [
        f"I hold a savings account with {f['bank_name']}.",
        f"My account number is {f['account_number']}.",
        f"On {f['transaction_date']} an amount of Rs. {f['disputed_amount']:,} was",
        "debited without my authorisation or consent.",
        f"The transaction reference is {f['transaction_reference']}.",
        "I raised this with the branch but received no resolution.",
        "I request that the amount be reversed and compensation considered.",
    ]
    para = para_deva if deva else para_en
    flds = [None, "account_number", "disputed_amount", None, "transaction_reference", None, "relief_sought"]
    # The value each linked sentence carries. `relief_sought` genuinely *is* the whole
    # sentence; the other three are identifiers embedded in one, so the field value is
    # the identifier and the sentence stays reachable through the line link.
    fld_values = {
        "account_number": f["account_number"],
        "disputed_amount": f"{f['disputed_amount']:,}",
        "transaction_reference": f["transaction_reference"],
    }

    for ln, fl, ln_en in zip(para, flds, para_en):
        value = fld_values.get(fl)
        c.text(
            (150, y), ln, role=role, size=size, script=script, lang=lang, hand=hand, jitter=jit,
            fld=fl, fld_value=value,
            # A value that is its own identifier is already English; only the sentence
            # forms (relief_sought) need the parallel pool's line.
            fld_en=(ln_en if deva and value is None else None),
        )
        y += size + 20

    y += 50
    c.text(
        (150, y),
        ("आपला विश्वासू," if deva else "Yours faithfully,"),
        role=role,
        size=size,
        script=script,
        lang=lang,
        hand=hand,
        jitter=jit,
    )
    y += size + 26
    nm = f["complainant_name_deva"] if deva else f["complainant_name"]
    c.text(
        (150, y),
        nm,
        role=role,
        size=size,
        script=script,
        lang=lang,
        hand=hand,
        jitter=jit,
        fld="complainant_name",
        fld_en=f["complainant_name"] if deva else None,
    )

    gt = GTDoc(
        doc_id="",
        doc_type="",
        variant=0,
        document_class="complaint_letter",
        primary_script=script,
        primary_language=lang,
    )
    return c, gt, f


def render_bilingual_form(f: dict, rng, *, pen: Pen) -> tuple[Canvas, GTDoc, dict]:
    """The hard case: printed English labels, handwritten Devanagari values."""
    # v1 renders the printed form first and discards it. Kept as-is: it draws no jittered
    # text, so it consumes nothing from the pen's RNG and the output is unchanged.
    render_complaint_form(f, deva=False, hand_values=False, rng=rng, pen=pen)
    # redraw values in handwritten Devanagari over the boxed regions
    c2 = Canvas(pen=pen)
    _form_header(c2, deva=False)
    # (label, value as drawn, field, script, language, handwritten, English equivalent)
    rows = [
        ("1. Name of complainant", f["complainant_name_deva"], "complainant_name", "Deva", "mar", True, f["complainant_name"]),
        ("2. Address", f["complainant_city_deva"], "complainant_address", "Deva", "mar", True, f["complainant_city"]),
        ("3. Mobile", f["mobile"], "complainant_contact", "Latn", "eng", True, None),
        ("4. Name of bank", f["bank_name_deva"], "bank_name", "Deva", "mar", True, f["bank_name"]),
        ("5. IFSC code", f["ifsc_code"], "ifsc_code", "Latn", "eng", True, None),
        ("6. Account number", f["account_number"], "account_number", "Latn", "eng", True, None),
        ("7. Amount in dispute (Rs.)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng", True, None),
        ("8. Date of transaction", f["transaction_date"], "transaction_date", "Latn", "eng", True, None),
        ("9. Transaction reference", f["transaction_reference"], "transaction_reference", "Latn", "eng", True, None),
        ("10. Previous complaint no.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng", True, None),
    ]
    y = 300
    for label, value, fld, vs, vl, hw, value_en in rows:
        c2.text((110, y), label, role="latin_print", size=25)
        c2.box(700, y - 10, 1584, y + 50, w=2)
        c2.text(
            (720, y - 2),
            value,
            role=("deva_hand" if vs == "Deva" else "latin_hand"),
            size=32,
            script=vs,
            lang=vl,
            hand=True,
            fld=fld,
            fld_en=value_en,
            jitter=2.4,
            fill=(28, 38, 96),
        )
        y += 78
    c2.text((110, y + 40), "11. Nature of complaint", role="latin_print", size=25)
    c2.box(110, y + 86, 1584, y + 380, w=2)
    c2.text(
        (140, y + 116),
        f["ground_deva"],
        role="deva_hand",
        size=32,
        script="Deva",
        lang="mar",
        hand=True,
        fld="complaint_summary",
        fld_en=f["ground_en"],
        jitter=2.6,
        fill=(28, 38, 96),
    )

    gt = GTDoc(
        doc_id="",
        doc_type="",
        variant=0,
        document_class="complaint_form",
        primary_script="Latn",
        primary_language="eng",
        is_code_mixed=True,
    )
    return c2, gt, f


def render_bank_statement(f: dict, rng, *, pen: Pen) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(pen=pen)
    c.text((110, 90), f["bank_name"], role="latin_bold", size=34, fld="bank_name")
    c.text((110, 140), "Statement of Account", role="latin_print", size=26)
    c.text((110, 182), f"A/C {f['account_number']}    IFSC {f['ifsc_code']}", role="mono", size=23)
    c.rule(110, 230, 1584, 230, w=3)
    hdr = [("Date", 120), ("Particulars", 380), ("Ref", 880), ("Debit", 1140), ("Balance", 1380)]
    for t, x in hdr:
        c.text((x, 246), t, role="latin_bold", size=23)
    c.rule(110, 292, 1584, 292)
    y, bal = 312, 187540
    for i in range(18):
        amt = f["disputed_amount"] if i == 7 else rng.randint(120, 9800)
        bal -= amt
        c.text((120, y), f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026", role="mono", size=21)
        c.text(
            (380, y),
            rng.choice(["UPI/P2M/", "NEFT/", "ATM WDL ", "POS PUR ", "IMPS/"])
            + f"{rng.randint(10**9, 10**10 - 1)}",
            role="mono",
            size=21,
        )
        c.text((880, y), f"{rng.randint(10**11, 10**12 - 1)}", role="mono", size=21)
        c.text((1140, y), f"{amt:,}.00", role="mono", size=21)
        c.text((1380, y), f"{bal:,}.00", role="mono", size=21)
        y += 40
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    # account_number and ifsc_code share the header line, so neither can own it; a scorer
    # recovers them as values embedded in that line. bank_name has a line to itself.
    return c, gt, {
        "bank_name": f["bank_name"],
        "account_number": f["account_number"],
        "ifsc_code": f["ifsc_code"],
    }


#: Receipt row label -> the field its value carries. ``A/C`` is deliberately absent: the
#: slip prints ``XXXXXX`` + the last four digits, so linking it to ``account_number``
#: would publish a masked string as the ground truth for the full number. ``Status`` is
#: a transaction outcome, not an extraction field.
RECEIPT_ROW_FIELDS = {
    "Date": "transaction_date",
    "Bank": "bank_name",
    "Card": "card_last_four",
    "Amount": "disputed_amount",
    "RRN": "transaction_reference",
}


def render_receipt(f: dict, rng, *, pen: Pen) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(w=760, h=1500, bg=(250, 249, 245), pen=pen)
    c.text((60, 70), "TRANSACTION RECEIPT", role="latin_bold", size=30)
    c.rule(60, 122, 700, 122)
    y = 150
    for k, v in [
        ("Date", f["transaction_date"]),
        ("Bank", f["bank_name"]),
        ("A/C", "XXXXXX" + f["account_number"][-4:]),
        ("Card", "XXXX XXXX XXXX " + f["card_last_four"]),
        ("Amount", f"Rs. {f['disputed_amount']:,}.00"),
        ("RRN", f["transaction_reference"]),
        ("Status", "FAILED"),
    ]:
        c.text((60, y), k, role="latin_print", size=24)
        # The Card row prints a masked PAN; the field it carries is the last four.
        c.text(
            (300, y), v, role="mono", size=24, fld=RECEIPT_ROW_FIELDS.get(k),
            fld_value=f["card_last_four"] if k == "Card" else None,
        )
        y += 52
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    # Every key here is drawn on the slip. The full account number is NOT — it is masked
    # to its last four digits — so it is not claimed. (Q6: this dict was `{}` in v1.1.0,
    # which gave receipt and photo_evidence no field reference at all.)
    return c, gt, {
        "transaction_date": f["transaction_date"],
        "bank_name": f["bank_name"],
        "card_last_four": f["card_last_four"],
        "disputed_amount": f["disputed_amount"],
        "transaction_reference": f["transaction_reference"],
    }


#: Email header label -> the field its value carries. ``From``/``To`` are mailbox
#: addresses; the corpus has no field for either, and the ``To`` local part is only the
#: complainant's first name lowercased, so it is not the complainant's name.
EMAIL_HEADER_FIELDS = {"Date": "transaction_date", "Subject": "prior_complaint_ref"}


def render_email_printout(f: dict, rng, *, pen: Pen) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(pen=pen)
    y = 110
    for k, v in [
        ("From", "customercare@bank.example.in"),
        ("To", f"{f['complainant_name'].split()[0].lower()}@example.com"),
        ("Date", f["transaction_date"]),
        ("Subject", f"Re: Service Request {f['prior_complaint_ref']}"),
    ]:
        c.text((110, y), f"{k}:", role="latin_bold", size=24)
        # The subject line wraps the reference in "Re: Service Request ..."; the field
        # value is the reference itself.
        c.text(
            (260, y), v, role="latin_print", size=24, fld=EMAIL_HEADER_FIELDS.get(k),
            fld_value=f["prior_complaint_ref"] if k == "Subject" else None,
        )
        y += 44
    c.rule(110, y + 14, 1584, y + 14)
    y += 60
    for ln in [
        "Dear Customer,",
        "We acknowledge your complaint regarding the disputed transaction.",
        f"Your service request number is {f['prior_complaint_ref']}.",
        "The matter is under investigation and we will revert within 30 days.",
        "Regards, Customer Care",
    ]:
        c.text(
            (110, y),
            ln,
            role="latin_print",
            size=25,
            fld="prior_complaint_ref" if "service request number" in ln else None,
        )
        y += 46
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    # The acknowledgement mentions no amount, no account number and no IFSC, so this
    # document legitimately has two fields and not five. Claiming more would invent a
    # reference the page cannot support — the failure mode Q6 exists to remove.
    return c, gt, {
        "transaction_date": f["transaction_date"],
        "prior_complaint_ref": f["prior_complaint_ref"],
    }


def cheque_micr_code(f: dict) -> str:
    """A 9-digit MICR city/bank/branch code, derived from facts already drawn.

    Deliberately **not** a new ``make_facts`` key and deliberately not a fresh
    ``rng`` draw: ``make_facts`` consumes the per-document RNG in a fixed order, so one
    extra ``randint`` there would shift every subsequent draw and re-render all 295
    pages. Derivation from existing digits keeps the rest of the corpus byte-identical.
    """
    return f"{f['ifsc_code'][5:8]}{f['account_number'][:3]}{f['transaction_reference'][-3:]}"


def render_cheque(f: dict, rng, *, pen: Pen) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(w=1600, h=700, bg=(240, 245, 240), pen=pen)
    c.box(20, 20, 1580, 680, w=3)
    c.text((60, 60), f["bank_name"], role="latin_bold", size=32, fld="bank_name")
    # --- printed additions, v1.2.0 (Q6) -------------------------------------------
    # A cheque with no date, no account number and no MICR band is a template defect,
    # not a hard case: the three fields the harness most wants from this document type
    # were simply never drawn. These are PRINTED draws, so they consume nothing from the
    # pen's jitter RNG and every handwritten stroke on the page is unchanged.
    c.text((1120, 72), "DATE", role="latin_print", size=24)
    c.text((1240, 70), f["transaction_date"], role="mono", size=26, fld="transaction_date")
    c.text((60, 120), "A/C No.", role="latin_print", size=22)
    c.text((200, 118), f["account_number"], role="mono", size=24, fld="account_number")
    # ------------------------------------------------------------------------------
    c.text((60, 200), "PAY", role="latin_print", size=24)
    c.text(
        (180, 190),
        f["complainant_name"],
        role="latin_hand",
        size=34,
        hand=True,
        jitter=2.2,
        fld="complainant_name",
        fill=(28, 38, 96),
    )
    c.text((60, 300), "RUPEES", role="latin_print", size=24)
    c.text(
        (220, 292),
        f"{f['disputed_amount']:,} only",
        role="latin_hand",
        size=34,
        hand=True,
        jitter=2.2,
        fld="disputed_amount",
        fld_value=f"{f['disputed_amount']:,}",  # "only" is convention, not the amount
        fill=(28, 38, 96),
    )
    c.box(1180, 270, 1540, 350, w=2)
    c.text(
        (1200, 288),
        f"{f['disputed_amount']:,}.00",
        role="latin_hand",
        size=32,
        hand=True,
        jitter=1.8,
        # The amount in figures. `fields_bilingual` keeps the words-slot spelling (first
        # draw wins); this one reaches a scorer through its own line link.
        fld="disputed_amount",
        fill=(28, 38, 96),
    )
    c.text((60, 538), "MICR", role="latin_print", size=22)
    c.text((180, 536), cheque_micr_code(f), role="mono", size=26, fld="micr_code")
    c.text(
        (60, 600),
        f"{f['transaction_reference'][:6]}  {f['ifsc_code']}  {f['account_number'][:9]}",
        role="mono",
        size=28,
        fld="ifsc_code",
        fld_value=f["ifsc_code"],  # the band also carries a cheque number and a stub
    )
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    # `transaction_reference` is deliberately absent: the bottom band prints only its
    # first six digits, so the full value is not on the page and claiming it would be a
    # reference no reader could satisfy.
    return c, gt, {
        "bank_name": f["bank_name"],
        "complainant_name": f["complainant_name"],
        "transaction_date": f["transaction_date"],
        "account_number": f["account_number"],
        "ifsc_code": f["ifsc_code"],
        "micr_code": cheque_micr_code(f),
        "disputed_amount": f["disputed_amount"],
    }


# Every renderer takes (facts, rng, pen): ``rng`` drives content and degradation choices,
# ``pen`` carries the handwriting-jitter stream and shaping mode. Two streams, so a change
# to one cannot shift the other.
RENDERERS = {
    "complaint_form_en": lambda f, r, p: render_complaint_form(f, deva=False, hand_values=False, rng=r, pen=p),
    "complaint_form_deva": lambda f, r, p: render_complaint_form(f, deva=True, hand_values=False, rng=r, pen=p),
    "complaint_letter_en": lambda f, r, p: render_complaint_letter(f, deva=False, hand=True, pen=p),
    "complaint_letter_mar": lambda f, r, p: render_complaint_letter(f, deva=True, hand=True, pen=p),
    "bilingual_form": lambda f, r, p: render_bilingual_form(f, r, pen=p),
    "bank_statement": lambda f, r, p: render_bank_statement(f, r, pen=p),
    "receipt": lambda f, r, p: render_receipt(f, r, pen=p),
    "email_printout": lambda f, r, p: render_email_printout(f, r, pen=p),
    "cheque": lambda f, r, p: render_cheque(f, r, pen=p),
    # photo_evidence reuses the receipt renderer and is FORCED to the `photo` degradation
    # profile even when `clean` is requested — see FORCED_PHOTO_PROFILE below. It
    # therefore carries the receipt's fields (date, bank, card fragment, amount, RRN);
    # they are the identifiers its template really renders, so no extra text is drawn
    # to give it a ground truth (OPEN-QUESTIONS Q6).
    "photo_evidence": lambda f, r, p: render_receipt(f, r, pen=p),
}

# Types whose `clean` request is rewritten to `photo`. Consequence (kept from v1, now
# documented): photo_evidence emits 5 pages per variant, not 6 — the rewritten `clean`
# collides with the real `photo` page and the duplicate is dropped. photo_evidence
# therefore has NO clean baseline; it exists to exercise the photographed-input path,
# and a pristine phone photo is not a condition the quality gate needs measuring on.
FORCED_PHOTO_PROFILE = {"photo_evidence"}


# ------------------------------------------------------------------- degradation


def _rotate_pt(p, cx, cy, deg):
    a = math.radians(-deg)
    x, y = p[0] - cx, p[1] - cy
    return (x * math.cos(a) - y * math.sin(a) + cx, x * math.sin(a) + y * math.cos(a) + cy)


def apply_rotation(img: Image.Image, lines: list[GTLine], deg: float):
    """Rotate the page and carry every bbox through the same transform."""
    cx, cy = img.width / 2, img.height / 2
    out = img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(248, 247, 243))
    for ln in lines:
        pts = [(ln.bbox[i], ln.bbox[i + 1]) for i in range(0, 8, 2)]
        moved = [_rotate_pt(p, cx, cy, deg) for p in pts]
        ln.bbox = [round(v, 1) for p in moved for v in p]
    return out


def apply_shadow(img, rng):
    grad = Image.new("L", img.size, 255)
    g = ImageDraw.Draw(grad)
    x0 = rng.randint(0, img.width // 2)
    for i in range(img.width):
        v = 255 - int(70 * max(0.0, 1 - abs(i - x0) / (img.width * 0.7)))
        g.line([(i, 0), (i, img.height)], fill=v)
    grad = grad.filter(ImageFilter.GaussianBlur(60))
    return Image.composite(img, Image.new("RGB", img.size, (30, 30, 34)), grad)


def apply_noise(img, rng, amount=0.03):
    px = img.load()
    n = int(img.width * img.height * amount)
    for _ in range(n):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        v = rng.choice([0, 255])
        px[x, y] = (v, v, v)
    return img


def apply_holes(img, rng, kind="punch"):
    d = ImageDraw.Draw(img)
    if kind == "punch":
        for cy in (img.height // 3, 2 * img.height // 3):
            d.ellipse([28, cy - 26, 80, cy + 26], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    else:  # staple
        d.line([(70, 70), (120, 110)], fill=(120, 120, 130), width=6)
    return img


def apply_stamp(img, rng):
    d = ImageDraw.Draw(img, "RGBA")
    x, y = rng.randint(900, 1200), rng.randint(200, 900)
    d.ellipse([x, y, x + 260, y + 260], outline=(75, 46, 131, 190), width=7)
    f = load_font("latin_bold", 28)
    d.text((x + 52, y + 112), "RECEIVED", font=f, fill=(75, 46, 131, 190))
    return img


def apply_occlusion(img, rng):
    d = ImageDraw.Draw(img)
    x = rng.randint(int(img.width * 0.55), int(img.width * 0.8))
    y = rng.randint(int(img.height * 0.3), int(img.height * 0.7))
    d.ellipse([x, y, x + 230, y + 320], fill=(196, 158, 130))
    return img


def apply_fold(img, rng):
    d = ImageDraw.Draw(img, "RGBA")
    y = img.height // rng.choice([2, 3])
    d.line([(0, y), (img.width, y)], fill=(150, 148, 140, 130), width=3)
    return img


def degrade(
    img: Image.Image, lines: list[GTLine], profile: str, rng: random.Random
) -> tuple[Image.Image, list[str]]:
    """Physical degradation. Geometric ops carry the ground truth with them."""
    applied: list[str] = []

    if profile == "clean":
        return img, applied

    if profile in ("light", "scan", "heavy", "photo", "bounce"):
        deg = {"light": 1.5, "scan": 3.5, "heavy": 8.0, "photo": 12.0, "bounce": 17.0}[profile]
        angle = rng.uniform(-deg, deg)
        img = apply_rotation(img, lines, angle)
        applied.append(f"skew:{angle:+.1f}deg")

    if profile in ("scan", "heavy", "bounce"):
        img = apply_holes(img, rng, rng.choice(["punch", "staple"]))
        applied.append("holes")
        img = apply_fold(img, rng)
        applied.append("fold")

    if profile in ("heavy", "photo", "bounce"):
        img = apply_shadow(img, rng)
        applied.append("shadow")

    if profile == "photo":
        img = apply_occlusion(img, rng)
        applied.append("thumb_occlusion")
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.78, 1.18))
        applied.append("exposure")

    if profile in ("scan", "heavy", "photo", "bounce"):
        img = apply_stamp(img, rng)
        applied.append("stamp")

    blur = {"light": 0.4, "scan": 0.8, "heavy": 1.5, "photo": 1.9, "bounce": 3.4}[profile]
    img = img.filter(ImageFilter.GaussianBlur(blur))
    applied.append(f"blur:{blur}")

    noise = {"light": 0.004, "scan": 0.012, "heavy": 0.03, "photo": 0.02, "bounce": 0.07}[profile]
    img = apply_noise(img, rng, noise)
    applied.append(f"noise:{noise}")

    dpi_scale = {"light": 1.0, "scan": 0.85, "heavy": 0.7, "photo": 0.62, "bounce": 0.42}[profile]
    if dpi_scale < 1.0:
        w, h = int(img.width * dpi_scale), int(img.height * dpi_scale)
        img = img.resize((w, h), Image.LANCZOS)
        for ln in lines:
            ln.bbox = [round(v * dpi_scale, 1) for v in ln.bbox]
        applied.append(f"downscale:{dpi_scale}")

    return img, applied


JPEG_Q = {"clean": 95, "light": 90, "scan": 72, "heavy": 55, "photo": 45, "bounce": 28}
PROFILES = ["clean", "light", "scan", "heavy", "photo", "bounce"]

HANDWRITING_CAVEAT = (
    "Font-rendered handwriting is not real handwriting. This corpus validates "
    "plumbing, schema, routing and throughput. It does NOT validate handwriting "
    "accuracy — measure that on iiit-indic-hw-words and real complaint pages."
)
PHOTO_EVIDENCE_CAVEAT = (
    "photo_evidence is forced to the `photo` degradation profile: a `clean` request for "
    "that type is rewritten to `photo`, which then collides with the real `photo` page, so "
    "photo_evidence emits 5 pages per variant instead of 6 and has no clean baseline. "
    "Intentional — the type exists to exercise the photographed-input path, and a pristine "
    "phone photo is not a condition the quality gate needs measuring on. Any consumer "
    "computing per-type page counts must special-case it."
)
FIELD_GROUND_TRUTH_CAVEAT = (
    "Two field blocks per document. `fields` holds the canonical English/ASCII facts as "
    "plain scalars (v1.1.0-compatible). `fields_bilingual` holds {value, value_en, "
    "script} where `value` is the string the page actually drew, verbatim, and "
    "`value_en` is its English equivalent from the parallel fact pool — equal to `value` "
    "on Latin pages. `fields_bilingual` is built from the draw calls, so it can never "
    "name a field the renderer did not render; `fields` may still name a fact that is "
    "on the page only in masked or truncated form. Where a field is drawn twice (the "
    "cheque's amount, in words and in figures) `fields_bilingual` keeps the first "
    "spelling and the second is reachable through lines[].field. Score a native-script "
    "claim against `value` and a translation against `value_en`."
)
HAND_SHAPING_CAVEAT = (
    "Handwritten lines are drawn one grapheme cluster at a time so per-unit jitter can be "
    "applied. Each cluster is shaped by Raqm (Devanagari matras and conjuncts are correct), "
    "but kerning and ligation ACROSS clusters is absent on handwritten lines. Printed lines "
    "are drawn as whole strings and are fully shaped."
)


# ---------------------------------------------------------------------- adversarial


def write_adversarial(out: Path) -> list[str]:
    """Inputs designed to break the ingestion path, not the models."""
    adv = out / "adversarial"
    adv.mkdir(parents=True, exist_ok=True)
    (adv / "zero_byte.pdf").write_bytes(b"")
    (adv / "truncated.pdf").write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog")
    (adv / "not_a_pdf.pdf").write_bytes(b"GIF89a" + b"\x00" * 512)
    (adv / "html_disguised_as_pdf.pdf").write_bytes(
        b"<html><script>alert(1)</script><body>invoice</body></html>"
    )
    Image.new("RGB", (60, 60), (255, 255, 255)).save(adv / "tiny_lowres.png")
    Image.new("RGB", (1654, 2339), (255, 255, 255)).save(adv / "blank_page.png")
    inv = Image.new("RGB", (1654, 2339), (12, 12, 16))
    ImageDraw.Draw(inv).text(
        (200, 400), "WHITE ON BLACK", font=load_font("latin_bold", 60), fill=(240, 240, 240)
    )
    inv.save(adv / "inverted.png")
    expected = {
        "zero_byte.pdf": "reject: EMPTY_FILE",
        "truncated.pdf": "reject: MALFORMED_PDF",
        "not_a_pdf.pdf": "reject: MIME_MISMATCH",
        "html_disguised_as_pdf.pdf": "reject: MIME_MISMATCH, script never evaluated",
        "tiny_lowres.png": "bounce: RESOLUTION_BELOW_FLOOR",
        "blank_page.png": "drop page: BLANK_PAGE, do not bill OCR",
        "inverted.png": "repair: auto-invert, then process",
    }
    manifest = {
        "purpose": "ingestion hardening — these must be rejected with typed errors, never crash a worker",
        "synthetic": True,
        "expected_behaviour": expected,
        "not_included_generate_locally": {
            "zip_bomb.zip": "42.zip or equivalent — decompression-ratio cap must trigger",
            "pdf_5000_pages.pdf": "page cap must trigger",
            "password_protected.pdf": "reject: ENCRYPTED_DOCUMENT",
        },
    }
    (adv / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sorted(expected)


# ------------------------------------------------------------------------- driver


def generate_corpus(
    out: Path,
    *,
    variants: int = 5,
    seed: int = 1729,
    types: list[str] | None = None,
    profiles: list[str] | None = None,
    adversarial: bool = True,
    hand_shaping: str = "cluster",
) -> dict:
    """Render, degrade and write the whole corpus. Returns the index payload.

    Raises FontUnavailableError, TextShapingUnavailableError, UnknownDocumentTypeError
    or CorpusGenerationError. Never returns partial success silently.
    """
    require_fonts()
    require_text_shaping()

    types = list(RENDERERS) if types is None else list(types)
    profiles = list(PROFILES) if profiles is None else list(profiles)
    unknown = [t for t in types if t not in RENDERERS]
    if unknown:
        raise UnknownDocumentTypeError(
            f"unknown type(s) {', '.join(repr(t) for t in unknown)}; known: {', '.join(RENDERERS)}"
        )
    bad_profiles = [p for p in profiles if p not in PROFILES]
    if bad_profiles:
        raise CorpusGenerationError(
            f"unknown profile(s) {', '.join(repr(p) for p in bad_profiles)}; known: {', '.join(PROFILES)}"
        )
    if hand_shaping not in HAND_SHAPING_MODES:
        raise CorpusGenerationError(
            f"unknown hand_shaping {hand_shaping!r}; known: {', '.join(HAND_SHAPING_MODES)}"
        )
    if variants < 1:
        raise CorpusGenerationError(f"variants must be >= 1, got {variants}")

    out.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    for dtype in types:
        tdir = out / dtype
        tdir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        for v in range(variants):
            for profile in profiles:
                # seed per (type, variant, profile) — reproducible and independently regenerable
                rng = random.Random(f"{seed}:{dtype}:{v}:{profile}")
                # Dedicated handwriting-jitter stream, derived from rng so it stays inside the
                # per-document seed but never touches the process-global random module.
                pen = Pen(rng=random.Random(rng.random()), hand_shaping=hand_shaping)
                facts = make_facts(rng, v)

                canvas, gt, fields = RENDERERS[dtype](facts, rng, pen)
                prof = "photo" if dtype in FORCED_PHOTO_PROFILE and profile == "clean" else profile
                img, applied = degrade(canvas.img, canvas.lines, prof, rng)

                doc_id = f"{dtype}_v{v}_{prof}"
                if doc_id in seen:  # a type may pin its profile (e.g. photo_evidence)
                    continue
                seen.add(doc_id)
                gt.doc_id = doc_id
                gt.doc_type = dtype
                gt.variant = v
                gt.degradations = applied
                gt.quality_expectation = (
                    "bounce" if prof == "bounce" else "clean" if prof in ("clean", "light") else "degraded"
                )
                gt.lines = canvas.lines
                gt.fields = {k: v2 for k, v2 in fields.items() if not k.endswith("_deva")} or fields
                # Built by the draw calls themselves, so it can only name fields the
                # renderer actually rendered. See the module docstring.
                gt.fields_bilingual = canvas.field_values

                img_path = tdir / f"{doc_id}.jpg"
                img.convert("RGB").save(img_path, quality=JPEG_Q[prof], subsampling=2)

                payload = asdict(gt)
                payload["lines"] = [asdict(ln) for ln in gt.lines]
                (tdir / f"{doc_id}.gt.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                index.append(
                    {
                        "doc_id": doc_id,
                        "type": dtype,
                        "variant": v,
                        "profile": prof,
                        "image": str(img_path.relative_to(out)),
                        "gt": str((tdir / f"{doc_id}.gt.json").relative_to(out)),
                        "lines": len(gt.lines),
                        "quality_expectation": gt.quality_expectation,
                        "is_code_mixed": gt.is_code_mixed,
                        "primary_script": gt.primary_script,
                    }
                )

    adversarial_files: list[str] = []
    if adversarial:
        adversarial_files = write_adversarial(out)

    index_payload = {
        "seed": seed,
        "synthetic": True,
        "generator_version": GENERATOR_VERSION,
        "hand_shaping": hand_shaping,
        "count": len(index),
        "caveat": HANDWRITING_CAVEAT,
        "caveats": {
            "handwriting": HANDWRITING_CAVEAT,
            "hand_shaping": HAND_SHAPING_CAVEAT,
            "photo_evidence_forced_profile": PHOTO_EVIDENCE_CAVEAT,
            "field_ground_truth": FIELD_GROUND_TRUTH_CAVEAT,
        },
        "types": types,
        "variants": variants,
        "profiles": profiles,
        "adversarial": adversarial_files,
        "documents": index,
    }
    (out / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return index_payload


# ---------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic RBI complaint corpus.")
    ap.add_argument("--out", default=Path("./corpus"), type=Path)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--types", nargs="*", default=list(RENDERERS))
    ap.add_argument("--profiles", nargs="*", default=PROFILES)
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument(
        "--hand-shaping",
        choices=HAND_SHAPING_MODES,
        default="cluster",
        help="cluster: shape each grapheme cluster (correct Devanagari). "
        "codepoint: v1 behaviour, one codepoint per draw — leaves Devanagari matras "
        "and conjuncts detached. Only use it to reproduce a v1 corpus.",
    )
    a = ap.parse_args(argv)

    try:
        payload = generate_corpus(
            a.out,
            variants=a.variants,
            seed=a.seed,
            types=a.types,
            profiles=a.profiles,
            adversarial=not a.no_adversarial,
            hand_shaping=a.hand_shaping,
        )
    except CorpusGenerationError as exc:
        print(f"\n{type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 2

    print(f"{payload['count']} pages -> {a.out}")
    print(f"  types        {len(a.types)}")
    print(f"  variants     {a.variants}")
    print(f"  profiles     {len(a.profiles)}")
    print(f"  hand shaping {a.hand_shaping}")
    print(f"  index        {a.out / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
