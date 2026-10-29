#!/usr/bin/env python3
"""
generate_test_corpus.py — synthetic RBI-complaint corpus with ground truth by construction.

Renders documents from text we control, then physically degrades them. Because every
string is drawn at a known position, per-line ground truth falls out for free: no human
labelling, fully reproducible from a seed, versionable in git.

    python3 generate_test_corpus.py --out ./corpus --variants 5 --seed 1729
    python3 generate_test_corpus.py --out ./corpus --types complaint_form_deva --variants 5

Fonts (see FONTS.md — not vendored, licences differ):
    assets/fonts/NotoSans-Regular.ttf          Latin print
    assets/fonts/NotoSansDevanagari-Regular.ttf Devanagari print
    assets/fonts/Kalam-Regular.ttf              handwriting, covers Devanagari + Latin
    assets/fonts/Caveat-Regular.ttf             handwriting, Latin
Latin falls back to DejaVuSans if Noto is absent. Devanagari has no fallback: the
script fails loudly rather than emitting boxes, because silently-wrong glyphs are
worse than a missing document.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# --------------------------------------------------------------------------- fonts

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FALLBACK_LATIN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FONTS = {
    "latin_print":  ["NotoSans-Regular.ttf", "NotoSans-Medium.ttf"],
    "latin_bold":   ["NotoSans-Bold.ttf"],
    "deva_print":   ["NotoSansDevanagari-Regular.ttf"],
    "deva_bold":    ["NotoSansDevanagari-Bold.ttf"],
    "latin_hand":   ["Caveat-Regular.ttf", "Kalam-Regular.ttf"],
    "deva_hand":    ["Kalam-Regular.ttf"],
    "mono":         ["NotoSansMono-Regular.ttf"],
}
_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    key = (role, size)
    if key in _cache:
        return _cache[key]
    for name in FONTS.get(role, []):
        p = FONT_DIR / name
        if p.exists():
            _cache[key] = ImageFont.truetype(str(p), size)
            return _cache[key]
    if role.startswith("deva"):
        sys.exit(
            f"\nMissing Devanagari font for role '{role}'.\n"
            f"Place NotoSansDevanagari-Regular.ttf and Kalam-Regular.ttf in {FONT_DIR}.\n"
            f"Refusing to render Devanagari with a fallback face — wrong glyphs are worse "
            f"than no document.\n"
        )
    if Path(FALLBACK_LATIN).exists():
        _cache[key] = ImageFont.truetype(FALLBACK_LATIN, size)
        return _cache[key]
    sys.exit(f"No usable font for role '{role}'. See FONTS.md.")


# ----------------------------------------------------------------------- gt model

@dataclass
class GTLine:
    text: str
    bbox: list[float]          # [x0,y0,x1,y1,x2,y2,x3,y3] clockwise from top-left
    script: str                # ISO 15924: Latn | Deva
    language: str              # ISO 639-3: eng | hin | mar
    is_handwritten: bool
    field: str | None = None   # links the line to an extraction field


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
    quality_expectation: str = "clean"     # clean | degraded | bounce
    lines: list[GTLine] = field(default_factory=list)
    fields: dict = field(default_factory=dict)


class Canvas:
    """A page that records the ground truth of everything drawn on it."""

    def __init__(self, w=1654, h=2339, bg=(252, 251, 247)):  # A4 @ 200 dpi
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img)
        self.lines: list[GTLine] = []

    def text(self, xy, s, *, role="latin_print", size=26, script="Latn", lang="eng",
             hand=False, fld=None, fill=(20, 24, 32), jitter=0.0, record=True):
        if not s:
            return 0
        x, y = xy
        f = load_font(role, size)
        if hand and jitter:
            # per-glyph wobble breaks the giveaway regularity of font-rendered "handwriting"
            cx = x
            for ch in s:
                dy = random.uniform(-jitter, jitter)
                self.d.text((cx, y + dy), ch, font=f, fill=fill)
                cx += f.getlength(ch) + random.uniform(-0.4, 0.9)
            w = cx - x
        else:
            self.d.text((x, y), s, font=f, fill=fill)
            w = f.getlength(s)
        h = size * 1.28
        if record:
            self.lines.append(GTLine(
                text=s,
                bbox=[x, y, x + w, y, x + w, y + h, x, y + h],
                script=script, language=lang, is_handwritten=hand, field=fld,
            ))
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
NAMES_EN = ["Rajesh Kumar Sharma", "Priya Venkatesan", "Abdul Rahman Shaikh",
            "Meera Joshi", "Sanjay Deshpande"]
NAMES_DEVA = ["राजेश कुमार शर्मा", "प्रिया वेंकटेशन", "अब्दुल रहमान शेख",
              "मीरा जोशी", "संजय देशपांडे"]
CITIES = [("Panvel, Maharashtra", "पनवेल, महाराष्ट्र"), ("Lucknow, Uttar Pradesh", "लखनऊ, उत्तर प्रदेश"),
          ("Nagpur, Maharashtra", "नागपुर, महाराष्ट्र"), ("Patna, Bihar", "पटना, बिहार"),
          ("Indore, Madhya Pradesh", "इंदौर, मध्य प्रदेश")]
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
        "transaction_date": f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2026",
        "transaction_reference": f"{rng.randint(10**11, 10**12 - 1)}",
        "complainant_name": NAMES_EN[v % len(NAMES_EN)],
        "complainant_name_deva": NAMES_DEVA[v % len(NAMES_DEVA)],
        "complainant_city": city_en,
        "complainant_city_deva": city_hi,
        "mobile": f"9{rng.randint(10**8, 10**9 - 1)}",
        "prior_complaint_ref": f"CMS/{rng.randint(2025,2026)}/{rng.randint(100000,999999)}",
        "ground_en": GROUNDS_EN[v % len(GROUNDS_EN)],
        "ground_deva": GROUNDS_DEVA[v % len(GROUNDS_DEVA)],
    }


# --------------------------------------------------------------------- renderers

def _form_header(c: Canvas, deva: bool):
    c.box(70, 70, 1584, 235, w=3)
    if deva:
        c.text((110, 100), "बैंकिंग लोकपाल को शिकायत", role="deva_bold", size=40,
               script="Deva", lang="hin")
        c.text((110, 160), "एकीकृत लोकपाल योजना 2021 — प्रपत्र I", role="deva_print",
               size=26, script="Deva", lang="hin")
    else:
        c.text((110, 100), "COMPLAINT TO THE BANKING OMBUDSMAN", role="latin_bold", size=38)
        c.text((110, 160), "Reserve Bank — Integrated Ombudsman Scheme, 2021 — Form I",
               role="latin_print", size=25)


def render_complaint_form(f: dict, deva=False, hand_values=False, rng=None) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas()
    _form_header(c, deva)
    rows = ([
        ("1. शिकायतकर्ता का नाम", f["complainant_name_deva"], "complainant_name", "Deva", "hin"),
        ("2. पता", f["complainant_city_deva"], "complainant_address", "Deva", "hin"),
        ("3. मोबाइल", f["mobile"], "complainant_contact", "Latn", "eng"),
        ("4. बैंक का नाम", f["bank_name_deva"], "bank_name", "Deva", "hin"),
        ("5. IFSC कोड", f["ifsc_code"], "ifsc_code", "Latn", "eng"),
        ("6. खाता संख्या", f["account_number"], "account_number", "Latn", "eng"),
        ("7. विवादित राशि (₹)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng"),
        ("8. लेनदेन दिनांक", f["transaction_date"], "transaction_date", "Latn", "eng"),
        ("9. लेनदेन संदर्भ", f["transaction_reference"], "transaction_reference", "Latn", "eng"),
        ("10. पूर्व शिकायत सं.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng"),
    ] if deva else [
        ("1. Name of complainant", f["complainant_name"], "complainant_name", "Latn", "eng"),
        ("2. Address", f["complainant_city"], "complainant_address", "Latn", "eng"),
        ("3. Mobile", f["mobile"], "complainant_contact", "Latn", "eng"),
        ("4. Name of bank", f["bank_name"], "bank_name", "Latn", "eng"),
        ("5. IFSC code", f["ifsc_code"], "ifsc_code", "Latn", "eng"),
        ("6. Account number", f["account_number"], "account_number", "Latn", "eng"),
        ("7. Amount in dispute (Rs.)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng"),
        ("8. Date of transaction", f["transaction_date"], "transaction_date", "Latn", "eng"),
        ("9. Transaction reference", f["transaction_reference"], "transaction_reference", "Latn", "eng"),
        ("10. Previous complaint no.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng"),
    ])

    lbl_role = "deva_print" if deva else "latin_print"
    lbl_script, lbl_lang = ("Deva", "hin") if deva else ("Latn", "eng")
    y = 300
    for label, value, fld, vscript, vlang in rows:
        c.text((110, y), label, role=lbl_role, size=25, script=lbl_script, lang=lbl_lang)
        c.box(700, y - 10, 1584, y + 50, w=2)
        if hand_values:
            vrole = "deva_hand" if vscript == "Deva" else "latin_hand"
            c.text((720, y - 2), value, role=vrole, size=32, script=vscript, lang=vlang,
                   hand=True, fld=fld, jitter=2.4, fill=(28, 38, 96))
        else:
            vrole = "deva_print" if vscript == "Deva" else "latin_print"
            c.text((720, y), value, role=vrole, size=26, script=vscript, lang=vlang, fld=fld)
        y += 78

    gy = y + 40
    grounds_label = "11. शिकायत का विवरण" if deva else "11. Nature of complaint"
    c.text((110, gy), grounds_label, role=lbl_role, size=25, script=lbl_script, lang=lbl_lang)
    c.box(110, gy + 46, 1584, gy + 340, w=2)
    body = f["ground_deva"] if deva else f["ground_en"]
    if hand_values:
        role = "deva_hand" if deva else "latin_hand"
        c.text((140, gy + 76), body, role=role, size=32,
               script="Deva" if deva else "Latn", lang="hin" if deva else "eng",
               hand=True, fld="complaint_summary", jitter=2.6, fill=(28, 38, 96))
    else:
        c.text((140, gy + 76), body, role=lbl_role, size=26,
               script=lbl_script, lang=lbl_lang, fld="complaint_summary")

    sig_y = gy + 420
    c.rule(110, sig_y, 620, sig_y)
    sig_label = "हस्ताक्षर" if deva else "Signature"
    c.text((110, sig_y + 12), sig_label, role=lbl_role, size=22,
           script=lbl_script, lang=lbl_lang)

    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="complaint_form",
               primary_script="Deva" if deva else "Latn",
               primary_language="hin" if deva else "eng")
    return c, gt, {k: v for k, v in f.items()}


def render_complaint_letter(f: dict, deva=False, hand=True) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas()
    role = ("deva_hand" if deva else "latin_hand") if hand else ("deva_print" if deva else "latin_print")
    script, lang = ("Deva", "mar" if deva else "eng"), None
    script, lang = ("Deva", "mar") if deva else ("Latn", "eng")
    size = 34 if hand else 27
    jit = 2.6 if hand else 0.0
    y = 160

    head = ["सेवेत,", "मा. बँकिंग लोकपाल,", "भारतीय रिझर्व्ह बँक"] if deva else \
           ["To,", "The Banking Ombudsman,", "Reserve Bank of India"]
    for ln in head:
        c.text((150, y), ln, role=role, size=size, script=script, lang=lang, hand=hand, jitter=jit)
        y += size + 22

    y += 30
    subj = f"विषय: {f['ground_deva']}" if deva else f"Subject: {f['ground_en']}"
    c.text((150, y), subj[:78], role=role, size=size, script=script, lang=lang,
           hand=hand, jitter=jit, fld="complaint_summary")
    y += size + 40

    if deva:
        para = [
            f"माझे {f['bank_name_deva']} येथे बचत खाते आहे.",
            f"खाते क्रमांक {f['account_number']} आहे.",
            f"दिनांक {f['transaction_date']} रोजी रु. {f['disputed_amount']:,} रक्कम",
            "माझ्या संमतीशिवाय डेबिट करण्यात आली.",
            f"संदर्भ क्रमांक {f['transaction_reference']}.",
            "बँकेकडे तक्रार करूनही निराकरण झाले नाही.",
            "कृपया योग्य ती कार्यवाही करावी ही विनंती.",
        ]
        flds = [None, "account_number", "disputed_amount", None, "transaction_reference", None, "relief_sought"]
    else:
        para = [
            f"I hold a savings account with {f['bank_name']}.",
            f"My account number is {f['account_number']}.",
            f"On {f['transaction_date']} an amount of Rs. {f['disputed_amount']:,} was",
            "debited without my authorisation or consent.",
            f"The transaction reference is {f['transaction_reference']}.",
            "I raised this with the branch but received no resolution.",
            "I request that the amount be reversed and compensation considered.",
        ]
        flds = [None, "account_number", "disputed_amount", None, "transaction_reference", None, "relief_sought"]

    for ln, fl in zip(para, flds):
        c.text((150, y), ln, role=role, size=size, script=script, lang=lang,
               hand=hand, jitter=jit, fld=fl)
        y += size + 20

    y += 50
    c.text((150, y), ("आपला विश्वासू," if deva else "Yours faithfully,"),
           role=role, size=size, script=script, lang=lang, hand=hand, jitter=jit)
    y += size + 26
    nm = f["complainant_name_deva"] if deva else f["complainant_name"]
    c.text((150, y), nm, role=role, size=size, script=script, lang=lang,
           hand=hand, jitter=jit, fld="complainant_name")

    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="complaint_letter",
               primary_script=script, primary_language=lang)
    return c, gt, f


def render_bilingual_form(f: dict, rng) -> tuple[Canvas, GTDoc, dict]:
    """The hard case: printed English labels, handwritten Devanagari values."""
    c, gt, _ = render_complaint_form(f, deva=False, hand_values=False, rng=rng)
    # redraw values in handwritten Devanagari over the boxed regions
    c2 = Canvas()
    _form_header(c2, deva=False)
    rows = [
        ("1. Name of complainant", f["complainant_name_deva"], "complainant_name", "Deva", "mar", True),
        ("2. Address", f["complainant_city_deva"], "complainant_address", "Deva", "mar", True),
        ("3. Mobile", f["mobile"], "complainant_contact", "Latn", "eng", True),
        ("4. Name of bank", f["bank_name_deva"], "bank_name", "Deva", "mar", True),
        ("5. IFSC code", f["ifsc_code"], "ifsc_code", "Latn", "eng", True),
        ("6. Account number", f["account_number"], "account_number", "Latn", "eng", True),
        ("7. Amount in dispute (Rs.)", f"{f['disputed_amount']:,}", "disputed_amount", "Latn", "eng", True),
        ("8. Date of transaction", f["transaction_date"], "transaction_date", "Latn", "eng", True),
        ("9. Transaction reference", f["transaction_reference"], "transaction_reference", "Latn", "eng", True),
        ("10. Previous complaint no.", f["prior_complaint_ref"], "prior_complaint_ref", "Latn", "eng", True),
    ]
    y = 300
    for label, value, fld, vs, vl, hw in rows:
        c2.text((110, y), label, role="latin_print", size=25)
        c2.box(700, y - 10, 1584, y + 50, w=2)
        c2.text((720, y - 2), value, role=("deva_hand" if vs == "Deva" else "latin_hand"),
                size=32, script=vs, lang=vl, hand=True, fld=fld, jitter=2.4, fill=(28, 38, 96))
        y += 78
    c2.text((110, y + 40), "11. Nature of complaint", role="latin_print", size=25)
    c2.box(110, y + 86, 1584, y + 380, w=2)
    c2.text((140, y + 116), f["ground_deva"], role="deva_hand", size=32, script="Deva",
            lang="mar", hand=True, fld="complaint_summary", jitter=2.6, fill=(28, 38, 96))

    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="complaint_form",
               primary_script="Latn", primary_language="eng", is_code_mixed=True)
    return c2, gt, f


def render_bank_statement(f: dict, rng) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas()
    c.text((110, 90), f["bank_name"], role="latin_bold", size=34)
    c.text((110, 140), "Statement of Account", role="latin_print", size=26)
    c.text((110, 182), f"A/C {f['account_number']}    IFSC {f['ifsc_code']}",
           role="mono", size=23)
    c.rule(110, 230, 1584, 230, w=3)
    hdr = [("Date", 120), ("Particulars", 380), ("Ref", 880), ("Debit", 1140), ("Balance", 1380)]
    for t, x in hdr:
        c.text((x, 246), t, role="latin_bold", size=23)
    c.rule(110, 292, 1584, 292)
    y, bal = 312, 187540
    for i in range(18):
        amt = f["disputed_amount"] if i == 7 else rng.randint(120, 9800)
        bal -= amt
        c.text((120, y), f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2026", role="mono", size=21)
        c.text((380, y), rng.choice(["UPI/P2M/", "NEFT/", "ATM WDL ", "POS PUR ", "IMPS/"]) +
               f"{rng.randint(10**9, 10**10-1)}", role="mono", size=21)
        c.text((880, y), f"{rng.randint(10**11, 10**12-1)}", role="mono", size=21)
        c.text((1140, y), f"{amt:,}.00", role="mono", size=21)
        c.text((1380, y), f"{bal:,}.00", role="mono", size=21)
        y += 40
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    return c, gt, {"account_number": f["account_number"], "ifsc_code": f["ifsc_code"]}


def render_receipt(f: dict, rng) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(w=760, h=1500, bg=(250, 249, 245))
    c.text((60, 70), "TRANSACTION RECEIPT", role="latin_bold", size=30)
    c.rule(60, 122, 700, 122)
    y = 150
    for k, v in [("Date", f["transaction_date"]), ("Bank", f["bank_name"]),
                 ("A/C", "XXXXXX" + f["account_number"][-4:]),
                 ("Card", "XXXX XXXX XXXX " + f["card_last_four"]),
                 ("Amount", f"Rs. {f['disputed_amount']:,}.00"),
                 ("RRN", f["transaction_reference"]), ("Status", "FAILED")]:
        c.text((60, y), k, role="latin_print", size=24)
        c.text((300, y), v, role="mono", size=24,
               fld={"Amount": "disputed_amount", "RRN": "transaction_reference"}.get(k))
        y += 52
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    return c, gt, {}


def render_email_printout(f: dict, rng) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas()
    y = 110
    for k, v in [("From", "customercare@bank.example.in"),
                 ("To", f"{f['complainant_name'].split()[0].lower()}@example.com"),
                 ("Date", f["transaction_date"]),
                 ("Subject", f"Re: Service Request {f['prior_complaint_ref']}")]:
        c.text((110, y), f"{k}:", role="latin_bold", size=24)
        c.text((260, y), v, role="latin_print", size=24)
        y += 44
    c.rule(110, y + 14, 1584, y + 14)
    y += 60
    for ln in ["Dear Customer,",
               "We acknowledge your complaint regarding the disputed transaction.",
               f"Your service request number is {f['prior_complaint_ref']}.",
               "The matter is under investigation and we will revert within 30 days.",
               "Regards, Customer Care"]:
        c.text((110, y), ln, role="latin_print", size=25,
               fld="prior_complaint_ref" if "service request number" in ln else None)
        y += 46
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    return c, gt, {}


def render_cheque(f: dict, rng) -> tuple[Canvas, GTDoc, dict]:
    c = Canvas(w=1600, h=700, bg=(240, 245, 240))
    c.box(20, 20, 1580, 680, w=3)
    c.text((60, 60), f["bank_name"], role="latin_bold", size=32)
    c.text((60, 200), "PAY", role="latin_print", size=24)
    c.text((180, 190), f["complainant_name"], role="latin_hand", size=34, hand=True,
           jitter=2.2, fld="complainant_name", fill=(28, 38, 96))
    c.text((60, 300), "RUPEES", role="latin_print", size=24)
    c.text((220, 292), f"{f['disputed_amount']:,} only", role="latin_hand", size=34,
           hand=True, jitter=2.2, fld="disputed_amount", fill=(28, 38, 96))
    c.box(1180, 270, 1540, 350, w=2)
    c.text((1200, 288), f"{f['disputed_amount']:,}.00", role="latin_hand", size=32,
           hand=True, jitter=1.8, fill=(28, 38, 96))
    c.text((60, 600), f"{f['transaction_reference'][:6]}  {f['ifsc_code']}  {f['account_number'][:9]}",
           role="mono", size=28, fld="ifsc_code")
    gt = GTDoc(doc_id="", doc_type="", variant=0, document_class="supporting_doc")
    return c, gt, {}


RENDERERS = {
    "complaint_form_en":    lambda f, r: render_complaint_form(f, deva=False, hand_values=False, rng=r),
    "complaint_form_deva":  lambda f, r: render_complaint_form(f, deva=True, hand_values=False, rng=r),
    "complaint_letter_en":  lambda f, r: render_complaint_letter(f, deva=False, hand=True),
    "complaint_letter_mar": lambda f, r: render_complaint_letter(f, deva=True, hand=True),
    "bilingual_form":       lambda f, r: render_bilingual_form(f, r),
    "bank_statement":       lambda f, r: render_bank_statement(f, r),
    "receipt":              lambda f, r: render_receipt(f, r),
    "email_printout":       lambda f, r: render_email_printout(f, r),
    "cheque":               lambda f, r: render_cheque(f, r),
    "photo_evidence":       lambda f, r: render_receipt(f, r),   # degraded as a phone photo
}


# ------------------------------------------------------------------- degradation

def _rotate_pt(p, cx, cy, deg):
    a = math.radians(-deg)
    x, y = p[0] - cx, p[1] - cy
    return (x * math.cos(a) - y * math.sin(a) + cx, x * math.sin(a) + y * math.cos(a) + cy)


def apply_rotation(img: Image.Image, lines: list[GTLine], deg: float):
    """Rotate the page and carry every bbox through the same transform."""
    cx, cy = img.width / 2, img.height / 2
    out = img.rotate(deg, resample=Image.BICUBIC, expand=False,
                     fillcolor=(248, 247, 243))
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


def degrade(img: Image.Image, lines: list[GTLine], profile: str, rng: random.Random) -> tuple[Image.Image, list[str]]:
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


# ---------------------------------------------------------------------- adversarial

def write_adversarial(out: Path):
    """Inputs designed to break the ingestion path, not the models."""
    adv = out / "adversarial"
    adv.mkdir(parents=True, exist_ok=True)
    (adv / "zero_byte.pdf").write_bytes(b"")
    (adv / "truncated.pdf").write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog")
    (adv / "not_a_pdf.pdf").write_bytes(b"GIF89a" + b"\x00" * 512)
    (adv / "html_disguised_as_pdf.pdf").write_bytes(
        b"<html><script>alert(1)</script><body>invoice</body></html>")
    huge = Image.new("RGB", (60, 60), (255, 255, 255))
    huge.save(adv / "tiny_lowres.png")
    Image.new("RGB", (1654, 2339), (255, 255, 255)).save(adv / "blank_page.png")
    inv = Image.new("RGB", (1654, 2339), (12, 12, 16))
    ImageDraw.Draw(inv).text((200, 400), "WHITE ON BLACK", font=load_font("latin_bold", 60),
                             fill=(240, 240, 240))
    inv.save(adv / "inverted.png")
    manifest = {
        "purpose": "ingestion hardening — these must be rejected with typed errors, never crash a worker",
        "expected_behaviour": {
            "zero_byte.pdf": "reject: EMPTY_FILE",
            "truncated.pdf": "reject: MALFORMED_PDF",
            "not_a_pdf.pdf": "reject: MIME_MISMATCH",
            "html_disguised_as_pdf.pdf": "reject: MIME_MISMATCH, script never evaluated",
            "tiny_lowres.png": "bounce: RESOLUTION_BELOW_FLOOR",
            "blank_page.png": "drop page: BLANK_PAGE, do not bill OCR",
            "inverted.png": "repair: auto-invert, then process",
        },
        "not_included_generate_locally": {
            "zip_bomb.zip": "42.zip or equivalent — decompression-ratio cap must trigger",
            "pdf_5000_pages.pdf": "page cap must trigger",
            "password_protected.pdf": "reject: ENCRYPTED_DOCUMENT",
        },
    }
    (adv / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./corpus", type=Path)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--types", nargs="*", default=list(RENDERERS))
    ap.add_argument("--profiles", nargs="*", default=PROFILES)
    ap.add_argument("--no-adversarial", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    index = []

    for dtype in a.types:
        if dtype not in RENDERERS:
            sys.exit(f"unknown type {dtype!r}; known: {', '.join(RENDERERS)}")
        tdir = a.out / dtype
        tdir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        for v in range(a.variants):
            for profile in a.profiles:
                # seed per (type, variant, profile) — reproducible and independently regenerable
                rng = random.Random(f"{a.seed}:{dtype}:{v}:{profile}")
                random.seed(rng.random())
                facts = make_facts(rng, v)

                canvas, gt, fields = RENDERERS[dtype](facts, rng)
                prof = "photo" if dtype == "photo_evidence" and profile == "clean" else profile
                img, applied = degrade(canvas.img, canvas.lines, prof, rng)

                doc_id = f"{dtype}_v{v}_{prof}"
                if doc_id in seen:      # a type may pin its profile (e.g. photo_evidence)
                    continue
                seen.add(doc_id)
                gt.doc_id = doc_id
                gt.doc_type = dtype
                gt.variant = v
                gt.degradations = applied
                gt.quality_expectation = ("bounce" if prof == "bounce"
                                          else "clean" if prof in ("clean", "light") else "degraded")
                gt.lines = canvas.lines
                gt.fields = {k: v2 for k, v2 in fields.items() if not k.endswith("_deva")} or fields

                img_path = tdir / f"{doc_id}.jpg"
                img.convert("RGB").save(img_path, quality=JPEG_Q[prof], subsampling=2)

                payload = asdict(gt)
                payload["lines"] = [asdict(l) for l in gt.lines]
                (tdir / f"{doc_id}.gt.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2))

                index.append({
                    "doc_id": doc_id, "type": dtype, "variant": v, "profile": prof,
                    "image": str(img_path.relative_to(a.out)),
                    "gt": str((tdir / f"{doc_id}.gt.json").relative_to(a.out)),
                    "lines": len(gt.lines),
                    "quality_expectation": gt.quality_expectation,
                    "is_code_mixed": gt.is_code_mixed,
                    "primary_script": gt.primary_script,
                })

    if not a.no_adversarial:
        write_adversarial(a.out)

    (a.out / "index.json").write_text(json.dumps({
        "seed": a.seed,
        "synthetic": True,
        "generator_version": "1.0.0",
        "count": len(index),
        "caveat": ("Font-rendered handwriting is not real handwriting. This corpus validates "
                   "plumbing, schema, routing and throughput. It does NOT validate handwriting "
                   "accuracy — measure that on iiit-indic-hw-words and real complaint pages."),
        "documents": index,
    }, indent=2))

    print(f"{len(index)} pages → {a.out}")
    print(f"  types      {len(a.types)}")
    print(f"  variants   {a.variants}")
    print(f"  profiles   {len(a.profiles)}")
    print(f"  index      {a.out / 'index.json'}")


if __name__ == "__main__":
    main()
