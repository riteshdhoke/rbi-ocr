#!/usr/bin/env python3
"""eval_harness.py — score pipeline predictions against the synthetic corpus.

Implements the Evaluation Harness Contract, TEST-DATA-SPEC.md section 7.

    .venv\\Scripts\\python.exe tools/eval_harness.py --corpus corpus --predictions runs/2026-08-07
    .venv\\Scripts\\python.exe tools/eval_harness.py --corpus corpus --predictions runs/x \\
            --json-out runs/x/report.json --split all --final

Standalone by design: stdlib only, with ``rapidfuzz`` used for edit distance when it is
importable and a pure-Python fallback when it is not. It imports nothing from ``app/`` so
it runs before the service exists, in a bare CI stage, and against any implementation that
can emit the JSON below.

The locked test split
---------------------
Variants 3 and 4 of every document type are the locked test split. ``--split dev``
(the default) scores variants 0-2 only. ``--split test`` and ``--split all`` refuse to run
without ``--final``, raising ``LockedSplitError``. Synthetic data makes overfitting to your
own generator embarrassingly easy; the numbers look wonderful right up until real mail
arrives.

Predictions directory
---------------------
One JSON file per document, named ``<doc_id>.json`` (``<doc_id>.pred.json`` also accepted),
where ``doc_id`` is the value from ``corpus/index.json``. A document with no file is
reported as missing and excluded from accuracy denominators, but counted in coverage.

Predictions file schema (field names follow BUILD-BRIEF section 3)
------------------------------------------------------------------
Only ``doc_id`` is required. Every other block is optional; a metric whose inputs are
absent is reported as "n/a" rather than silently scored as zero.

    {
      "doc_id": "bilingual_form_v2_scan",         // required, must match the corpus
      "result_version": 1,                         // int
      "is_final": true,                            // bool
      "schema_version": "1.0.0",                   // free-form

      "quality": {                                 // stage s15_quality
        "verdict": "accept" | "repair" | "bounce",
        "score": 0.81,
        "flags": ["skew", "low_contrast"]
      },

      "escalated": false,                          // stage s4_router (tier-2 fired)
      "escalation_reason": null,                   // free-form, reported not scored
      "shadow_sampled": false,                     // reported not scored

      "pages": [                                   // OCRPage[]; one page per document here
        {
          "page_no": 1,
          "lines": [                               // OCRLine — language lives HERE
            {
              "text": "4. Name of bank",
              "bbox": [110,534, 341,534, 341,566, 110,566],   // 8 floats, clockwise
              "confidence": 0.97,
              "script": "Latn",                    // ISO 15924
              "language": "eng",                   // ISO 639-3, or null
              "is_handwritten": false,
              "line_index": 12
            }
          ]
        }
      ],

      "fields": {                                  // Field, keyed by field name
        "ifsc_code": {
          "value": "BARB0417293",                  // native script, verbatim, immutable
          "value_en": null,                        // advisory translation
          "normalized": "BARB0417293",
          "confidence": 0.99,
          "source": "regex_validator" | "lookup_table" | "slm" | "classifier" | "human",
          "validator_passed": true,
          "evidence": {"page_no": 1, "bbox": [...], "span": [0, 11]},
          "mt_engine": null
        }
      },

      "timings_ms": {"s1_ingest": 11.4, "s4_ocr": 812.0, "s5_validators": 24.1}
    }

Ground truth this harness reads
-------------------------------
From each ``*.gt.json`` (generator >= 1.2.0):

* ``lines[]`` — ``text``, ``bbox``, ``script``, ``language``, ``is_handwritten`` and an
  optional ``field`` linking the line to an extraction field.
* ``fields`` — the document's facts in canonical English/ASCII spelling, plain scalars.
* ``fields_bilingual`` — ``{field: {"value", "value_en", "script"}}`` where ``value`` is
  the field's own substring **as the page drew it** and ``value_en`` is its English
  equivalent. Built from the generator's draw calls, so a key here is a guarantee the
  renderer rendered it. Absent on a pre-1.2.0 corpus, which this harness still reads.

A field's accepted spellings are therefore split two ways: ``native`` (what the page
shows) and ``en`` (the English equivalent, which on a Latin page is the same string). A
native-script claim is scored against ``native`` and a translation against ``en``; see
``FieldReference`` and the ``fields.bilingual`` block of the report.

Field naming: the envelope and the corpus
------------------------------------------
The corpus predates ``app/contracts/ocr.py`` and names four fields differently
(``disputed_amount`` vs ``amount``, and so on). ``ENVELOPE_FIELD_ALIASES`` maps the
envelope's name onto the corpus's before scoring, and the mapping is echoed in
``report["settings"]``. Without it those fields have no reference instance at all, so
every extraction of them is a false positive by construction — which is exactly what the
M7 gate measured (OPEN-QUESTIONS Q6: ``amount`` 69 FP / 0 GT, ``utr_rrn`` 61 / 0,
``card_fragment`` 24 / 0). Reporting is keyed on the **corpus** name so the TP/FP/FN
columns line up with ``gt_occurrences``.

Accepted intermediates (so a half-built pipeline can still be measured):

* ``lines`` at the top level, or ``ocr.lines``, instead of ``pages[].lines``.
* A 4-number ``bbox`` ``[x0,y0,x1,y1]`` instead of the 8-number quadrilateral.
* A bare string as a field value instead of the full Field object; it is scored for
  correctness and attributed to source ``"unspecified"``.

Both forms are accepted for scoring, but only the full envelope above counts towards the
schema-validity rate — that metric exists to tell you how far the real output is from the
contract, so widening it to include intermediates would defeat it.

How the reference is built (judgment calls, made explicit)
----------------------------------------------------------
* **Line matching** is greedy by IoU on the axis-aligned bounding box of each ground-truth
  quadrilateral. Predicted lines left over are matched by text similarity so an
  implementation that reports no boxes still gets CER/WER. Unmatched ground-truth lines are
  scored as full deletions; unmatched predictions are counted as spurious but do not inflate
  CER denominators.
* **Field reference** comes from the ground-truth *lines* and from ``gt["fields_bilingual"]``
  (both native script, both guaranteed on the page). ``gt["fields"]`` only contributes extra
  accepted spellings: it carries English even for Devanagari pages and may name a fact the
  page shows only masked or truncated. Its contribution lands in ``en`` rather than
  ``native`` whenever the bilingual block says the field is Devanagari-valued.
* **A hit is still a hit whichever slot carried it.** TP/FP/FN accounting accepts a match
  from ``value``, ``value_en`` or ``normalized`` against ``native | en``. The stricter
  question — did the *immutable* ``value`` carry the native string, or did the pipeline
  put the translation there — is reported separately under ``fields.bilingual`` rather
  than folded into the headline, so a corpus fix and a metric change never arrive
  together and get mistaken for each other.
* **Recall by source** is measured against the ground truth, not against what that source
  claimed, so the per-source recalls decompose the overall recall. A validator that nails
  one field and ignores the rest scores its true share, not 1.0.
* **A wrong value is both a false positive and a miss.** Claiming a field incorrectly
  costs precision *and* recall; otherwise a confident wrong answer would erase the miss.
* **Escalation reference** is ``quality_expectation != "clean"`` OR a handwritten-line
  majority. Selectable with ``--escalation-reference``; the corpus has no escalation label,
  so this is a stand-in and the number is only comparable across runs of the same setting.

What this module does NOT handle
--------------------------------
* **Rotated-box IoU.** Skewed profiles are scored on axis-aligned boxes, which understates
  IoU at high skew. Detection numbers for ``heavy``/``photo``/``bounce`` are a floor.
* **Multi-page documents.** One document is one page here; ``page_no`` is carried but never
  used to disambiguate.
* **Reading order / layout metrics.** No ordering, column or table-structure scoring.
* **Grounding-gate, translation or summary-span quality.** Those need the envelope's own
  contract tests, not corpus ground truth.
* **Statistical significance.** No confidence intervals, no significance tests. Cells with
  small ``n`` are reported with their ``n`` so you can distrust them yourself.
* **The adversarial set.** Those assert typed rejections in the ingestion tests, not metrics.
* **Deciding pass/fail.** It reports; thresholds live in CI.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

try:  # optional accelerator; the fallback below is exact, just slower
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only in bare environments
    _RF_Levenshtein = None
    _HAVE_RAPIDFUZZ = False


HARNESS_VERSION = "1.1.0"

#: Variants held out as the locked test split. Two of five, per TEST-DATA-SPEC section 7.
LOCKED_TEST_VARIANTS = frozenset({3, 4})

FIELD_SOURCES = ("regex_validator", "lookup_table", "slm", "classifier", "human")
QUALITY_VERDICTS = ("accept", "repair", "bounce")
QUALITY_EXPECTATIONS = ("clean", "degraded", "bounce")

#: expectation -> verdicts that satisfy it (TEST-DATA-SPEC section 3)
QUALITY_ACCEPTABLE: dict[str, tuple[str, ...]] = {
    "clean": ("accept",),
    "degraded": ("accept", "repair"),
    "bounce": ("bounce",),
}

#: ``gt["fields"]`` keys that are generator bookkeeping, not extraction targets
NON_FIELD_KEYS = frozenset({"ground_en", "ground_deva"})

#: ``gt["fields"]`` key -> the field name used on ground-truth lines
FIELD_KEY_ALIASES = {"complainant_city": "complainant_address"}

#: Envelope field name (``app/contracts/ocr.py``) -> the corpus's name for the same thing.
#:
#: The corpus was written before the contract and kept its own vocabulary for four fields.
#: Scoring without this mapping gives them no reference instance at all, so every
#: extraction is a false positive and every ground-truth occurrence a miss — the defect
#: OPEN-QUESTIONS Q6 measured. Deliberately NOT here: ``complaint_date``, which the
#: contract distinguishes from ``transaction_date`` and the corpus never renders.
ENVELOPE_FIELD_ALIASES = {
    "amount": "disputed_amount",
    "utr_rrn": "transaction_reference",
    "card_fragment": "card_last_four",
    "prior_complaint_reference": "prior_complaint_ref",
}

#: keys whose value may be recovered from a line it is merely embedded in
EMBEDDED_FIELD_KEYS = frozenset(
    {"account_number", "ifsc_code", "transaction_reference", "mobile", "prior_complaint_ref"}
)
EMBEDDED_MIN_LEN = 6

#: a document counts as "should have escalated" on handwriting at or above this ratio
HAND_MAJORITY_RATIO = 0.5

DEFAULT_IOU_THRESHOLD = 0.5

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_WS = re.compile(r"\s+")
_PUNCT_FOR_VALUES = re.compile(r"[,\s\-_/.:()\[\]]+")


# ----------------------------------------------------------------- typed errors


class EvalHarnessError(RuntimeError):
    """Base class for every failure this harness raises at a boundary."""


class CorpusNotFoundError(EvalHarnessError):
    """The corpus directory or its index.json is missing or unreadable."""


class CorpusIndexInvalidError(EvalHarnessError):
    """index.json parsed but does not carry the fields the harness needs."""


class PredictionsNotFoundError(EvalHarnessError):
    """The predictions directory does not exist."""


class LockedSplitError(EvalHarnessError):
    """Someone asked to score the held-out split without passing --final."""


class UnknownSplitError(EvalHarnessError):
    """A --split value that is not dev/test/all."""


# ------------------------------------------------------------------- text utils


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance over any two sequences of hashables. Exact, not approximate."""
    if _HAVE_RAPIDFUZZ:
        return int(_RF_Levenshtein.distance(a, b))
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_for_cer(s: str) -> str:
    """NFKC, trim, collapse internal whitespace. Case and script are preserved."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip()


def normalize_value(s: Any) -> str:
    """Aggressive normalisation for field-value equality.

    NFKC, Devanagari digits to ASCII, rupee sign and ``Rs``/``Rs.`` dropped, separators and
    punctuation removed, casefolded. Applied to both sides, so ``"45,000"``, ``"Rs. 45000"``
    and ``"४५०००"`` all compare equal — which is the intent, since Stage 5 owns the
    normalised form and this metric is about whether the right value was found at all.
    """
    text = unicodedata.normalize("NFKC", str(s)).translate(_DEVANAGARI_DIGITS)
    text = text.replace("₹", "")
    text = re.sub(r"\bRs\.?\b", "", text, flags=re.IGNORECASE)
    text = _PUNCT_FOR_VALUES.sub("", text)
    return text.casefold()


def tokens(s: str) -> list[str]:
    return normalize_for_cer(s).split(" ") if normalize_for_cer(s) else []


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile. ``q`` in [0, 1]. None for an empty sequence."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * q
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return float(ordered[int(k)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (k - low))


# ------------------------------------------------------------------- bbox utils


def to_aabb(bbox: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Axis-aligned (x0, y0, x1, y1) from an 8-point quad or a 4-number rect."""
    if bbox is None:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if len(vals) == 8:
        xs, ys = vals[0::2], vals[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    ra, rb = to_aabb(a) if a is not None else None, to_aabb(b) if b is not None else None
    if ra is None or rb is None:
        return 0.0
    ix0, iy0 = max(ra[0], rb[0]), max(ra[1], rb[1])
    ix1, iy1 = min(ra[2], rb[2]), min(ra[3], rb[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ra[2] - ra[0]) * max(0.0, ra[3] - ra[1])
    area_b = max(0.0, rb[2] - rb[0]) * max(0.0, rb[3] - rb[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def text_similarity(a: str, b: str) -> float:
    """0..1 similarity used only to rescue lines that carry no usable bbox."""
    na, nb = normalize_for_cer(a), normalize_for_cer(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return 1.0 - levenshtein(na, nb) / max(len(na), len(nb))


# ------------------------------------------------------------------ data models


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    doc_type: str
    variant: int
    profile: str
    quality_expectation: str
    gt_path: Path
    image_path: Path

    @property
    def is_locked(self) -> bool:
        return self.variant in LOCKED_TEST_VARIANTS


@dataclass
class PredLine:
    text: str
    bbox: list[float] | None
    confidence: float | None
    script: str | None
    language: str | None
    is_handwritten: bool | None
    line_index: int | None


@dataclass
class Prediction:
    doc_id: str
    lines: list[PredLine] = dc_field(default_factory=list)
    has_lines_block: bool = False
    fields: dict[str, dict] = dc_field(default_factory=dict)
    has_fields_block: bool = False
    quality_verdict: str | None = None
    escalated: bool | None = None
    timings_ms: dict[str, float] = dc_field(default_factory=dict)
    schema_valid: bool = False
    schema_errors: list[str] = dc_field(default_factory=list)


# --------------------------------------------------------------------- loading


def load_corpus(corpus_dir: Path) -> tuple[dict, list[CorpusDoc]]:
    """Read index.json and resolve every document's paths. Raises CorpusNotFoundError."""
    index_path = corpus_dir / "index.json"
    if not index_path.is_file():
        raise CorpusNotFoundError(
            f"No index.json at {index_path}. Generate the corpus first:\n"
            "  .venv\\Scripts\\python.exe tools/generate_test_corpus.py --out corpus "
            "--variants 5 --seed 1729"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusNotFoundError(f"Cannot read {index_path}: {exc}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("documents"), list):
        raise CorpusIndexInvalidError(f"{index_path} has no 'documents' list.")

    docs: list[CorpusDoc] = []
    for i, entry in enumerate(index["documents"]):
        try:
            docs.append(
                CorpusDoc(
                    doc_id=str(entry["doc_id"]),
                    doc_type=str(entry["type"]),
                    variant=int(entry["variant"]),
                    profile=str(entry["profile"]),
                    quality_expectation=str(entry["quality_expectation"]),
                    gt_path=corpus_dir / entry["gt"],
                    image_path=corpus_dir / entry["image"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusIndexInvalidError(f"{index_path} document[{i}] is malformed: {exc}") from exc
    return index, docs


def load_ground_truth(doc: CorpusDoc) -> dict:
    try:
        return json.loads(doc.gt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusNotFoundError(f"Cannot read ground truth {doc.gt_path}: {exc}") from exc


def find_prediction_file(predictions_dir: Path, doc_id: str) -> Path | None:
    for name in (f"{doc_id}.json", f"{doc_id}.pred.json"):
        candidate = predictions_dir / name
        if candidate.is_file():
            return candidate
    return None


def _coerce_line(raw: Any) -> PredLine | None:
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    if not isinstance(text, str):
        return None
    bbox = raw.get("bbox")
    bbox_list: list[float] | None = None
    if isinstance(bbox, (list, tuple)) and to_aabb(bbox) is not None:
        bbox_list = [float(v) for v in bbox]
    conf = raw.get("confidence")
    return PredLine(
        text=text,
        bbox=bbox_list,
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        script=raw.get("script") if isinstance(raw.get("script"), str) else None,
        language=raw.get("language") if isinstance(raw.get("language"), str) else None,
        is_handwritten=raw.get("is_handwritten") if isinstance(raw.get("is_handwritten"), bool) else None,
        line_index=raw.get("line_index") if isinstance(raw.get("line_index"), int) else None,
    )


def _extract_lines(payload: dict) -> tuple[list[Any], bool]:
    """Return (raw line dicts, whether the canonical pages[] block was used)."""
    pages = payload.get("pages")
    if isinstance(pages, list):
        raws: list[Any] = []
        for page in pages:
            if isinstance(page, dict) and isinstance(page.get("lines"), list):
                raws.extend(page["lines"])
        return raws, True
    for holder in (payload.get("ocr"), payload):
        if isinstance(holder, dict) and isinstance(holder.get("lines"), list):
            return list(holder["lines"]), False
    return [], False


def validate_prediction(payload: Any, expected_doc_id: str) -> list[str]:
    """Structural check against the canonical envelope. Returns a list of violations.

    Empty list means schema-valid. This is deliberately stricter than what ``parse``
    accepts: the schema-validity rate must measure distance from the contract.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["top level is not a JSON object"]
    doc_id = payload.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        errors.append("doc_id missing or not a non-empty string")
    elif doc_id != expected_doc_id:
        errors.append(f"doc_id {doc_id!r} does not match corpus id {expected_doc_id!r}")

    if "result_version" in payload and not isinstance(payload["result_version"], int):
        errors.append("result_version is not an int")
    if "is_final" in payload and not isinstance(payload["is_final"], bool):
        errors.append("is_final is not a bool")

    pages = payload.get("pages")
    if pages is None:
        errors.append("pages[] missing (an intermediate 'lines' block is accepted for scoring, not for schema validity)")
    elif not isinstance(pages, list):
        errors.append("pages is not a list")
    else:
        for pi, page in enumerate(pages):
            if not isinstance(page, dict):
                errors.append(f"pages[{pi}] is not an object")
                continue
            if not isinstance(page.get("page_no"), int):
                errors.append(f"pages[{pi}].page_no is not an int")
            lines = page.get("lines")
            if not isinstance(lines, list):
                errors.append(f"pages[{pi}].lines is not a list")
                continue
            for li, line in enumerate(lines):
                where = f"pages[{pi}].lines[{li}]"
                if not isinstance(line, dict):
                    errors.append(f"{where} is not an object")
                    continue
                if not isinstance(line.get("text"), str):
                    errors.append(f"{where}.text is not a string")
                bbox = line.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 8:
                    errors.append(f"{where}.bbox is not 8 numbers")
                elif not all(isinstance(v, (int, float)) for v in bbox):
                    errors.append(f"{where}.bbox contains a non-number")
                conf = line.get("confidence")
                if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                    errors.append(f"{where}.confidence is not a number")
                elif not 0.0 <= float(conf) <= 1.0:
                    errors.append(f"{where}.confidence {conf} outside [0,1]")
                if not isinstance(line.get("script"), str):
                    errors.append(f"{where}.script is not a string")
                if "language" in line and not (line["language"] is None or isinstance(line["language"], str)):
                    errors.append(f"{where}.language is neither null nor a string")
                if not isinstance(line.get("is_handwritten"), bool):
                    errors.append(f"{where}.is_handwritten is not a bool")
                if not isinstance(line.get("line_index"), int):
                    errors.append(f"{where}.line_index is not an int")

    fields = payload.get("fields")
    if fields is not None:
        if not isinstance(fields, dict):
            errors.append("fields is not an object")
        else:
            for name, spec in fields.items():
                where = f"fields[{name!r}]"
                if not isinstance(spec, dict):
                    errors.append(f"{where} is not an object (bare values are an intermediate form)")
                    continue
                if "value" not in spec or not (spec["value"] is None or isinstance(spec["value"], str)):
                    errors.append(f"{where}.value missing or not a string/null")
                if "value_en" in spec and not (
                    spec["value_en"] is None or isinstance(spec["value_en"], str)
                ):
                    errors.append(f"{where}.value_en is neither null nor a string")
                conf = spec.get("confidence")
                if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                    errors.append(f"{where}.confidence is not a number")
                if spec.get("source") not in FIELD_SOURCES:
                    errors.append(f"{where}.source {spec.get('source')!r} not in {FIELD_SOURCES}")
                ev = spec.get("evidence")
                if not isinstance(ev, dict):
                    errors.append(f"{where}.evidence missing — evidence is never optional")
                else:
                    if not isinstance(ev.get("page_no"), int):
                        errors.append(f"{where}.evidence.page_no is not an int")
                    if to_aabb(ev.get("bbox")) is None:
                        errors.append(f"{where}.evidence.bbox is not a usable box")
                if "validator_passed" in spec and not (
                    spec["validator_passed"] is None or isinstance(spec["validator_passed"], bool)
                ):
                    errors.append(f"{where}.validator_passed is neither null nor a bool")

    quality = payload.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            errors.append("quality is not an object")
        elif quality.get("verdict") not in QUALITY_VERDICTS:
            errors.append(f"quality.verdict {quality.get('verdict')!r} not in {QUALITY_VERDICTS}")

    timings = payload.get("timings_ms")
    if timings is not None:
        if not isinstance(timings, dict):
            errors.append("timings_ms is not an object")
        elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in timings.values()):
            errors.append("timings_ms contains a non-number")
    return errors


def parse_prediction(payload: Any, doc_id: str) -> Prediction:
    """Best-effort read of a prediction payload; never raises on shape."""
    pred = Prediction(doc_id=doc_id)
    pred.schema_errors = validate_prediction(payload, doc_id)
    pred.schema_valid = not pred.schema_errors
    if not isinstance(payload, dict):
        return pred

    raw_lines, _ = _extract_lines(payload)
    pred.has_lines_block = bool(raw_lines) or isinstance(payload.get("pages"), list)
    for raw in raw_lines:
        line = _coerce_line(raw)
        if line is not None:
            pred.lines.append(line)

    fields = payload.get("fields")
    if isinstance(fields, dict):
        pred.has_fields_block = True
        for name, spec in fields.items():
            if isinstance(spec, dict):
                pred.fields[str(name)] = spec
            elif isinstance(spec, str):
                pred.fields[str(name)] = {"value": spec, "source": "unspecified"}

    quality = payload.get("quality")
    if isinstance(quality, dict) and isinstance(quality.get("verdict"), str):
        pred.quality_verdict = quality["verdict"]
    elif isinstance(payload.get("quality_verdict"), str):
        pred.quality_verdict = payload["quality_verdict"]

    if isinstance(payload.get("escalated"), bool):
        pred.escalated = payload["escalated"]
    elif isinstance(payload.get("escalation"), dict) and isinstance(
        payload["escalation"].get("escalated"), bool
    ):
        pred.escalated = payload["escalation"]["escalated"]

    timings = payload.get("timings_ms")
    if isinstance(timings, dict):
        for stage, value in timings.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                pred.timings_ms[str(stage)] = float(value)
    return pred


# ------------------------------------------------------------------- reference


@dataclass
class FieldReference:
    """Accepted normalised spellings for one field, split by what the page shows.

    ``native`` holds the string the document was rendered with — the reference for an
    envelope's immutable ``value``. ``en`` holds the English equivalent, which is what an
    advisory ``value_en`` should say. On a Latin page the two sets are identical; only on
    a Devanagari page do they diverge, and that divergence is the whole reason for the
    split (a page reading ``बैंक ऑफ बड़ौदा`` used to be scored against ``"Bank of Baroda"``).
    """

    native: set[str] = dc_field(default_factory=set)
    en: set[str] = dc_field(default_factory=set)

    @property
    def accepted(self) -> set[str]:
        """Every spelling that counts as finding this field at all."""
        return self.native | self.en

    @property
    def is_bilingual(self) -> bool:
        """True when the English side says something the page itself does not.

        Set *inequality* is the wrong test: ``native`` legitimately holds extra spellings
        (the whole sentence an identifier is embedded in, the amount written twice), and
        none of those makes the field bilingual. What does is an English spelling that
        appears nowhere in what was rendered.
        """
        return bool(self.native and self.en) and not self.en <= self.native


def build_field_reference(gt: dict) -> dict[str, FieldReference]:
    """field name -> FieldReference, built from what is demonstrably on the page.

    Two authoritative sources, both native-script:

    * ground-truth *lines* with a ``field`` link — the string that was rendered;
    * ``gt["fields_bilingual"]`` (generator >= 1.2.0) — the field's own substring as
      drawn, plus its English equivalent. Absent on an older corpus, which still scores.

    ``gt["fields"]`` contributes alternative spellings only. It holds English values even
    for Devanagari pages, so it lands in ``en`` whenever the bilingual block marks the
    field Devanagari-valued, and in both sets otherwise.
    """
    reference: dict[str, FieldReference] = defaultdict(FieldReference)
    line_texts: list[str] = []
    for line in gt.get("lines", []):
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if isinstance(text, str):
            line_texts.append(text)
        name = line.get("field")
        if isinstance(name, str) and name and isinstance(text, str):
            normalised = normalize_value(text)
            if normalised:
                reference[name].native.add(normalised)

    # The generator's own record of each field's value, so a field embedded in a longer
    # line (``"My account number is 12345678901."``) has the number as a reference and
    # not only the sentence around it.
    deva_valued: set[str] = set()
    bilingual = gt.get("fields_bilingual")
    if isinstance(bilingual, dict):
        for raw_key, spec in bilingual.items():
            if not isinstance(spec, dict):
                continue
            key = FIELD_KEY_ALIASES.get(raw_key, raw_key)
            native = normalize_value(spec.get("value")) if isinstance(spec.get("value"), str) else ""
            english = normalize_value(spec.get("value_en")) if isinstance(spec.get("value_en"), str) else ""
            if native:
                reference[key].native.add(native)
            if english:
                reference[key].en.add(english)
            if spec.get("script") == "Deva":
                deva_valued.add(key)

    # values already claimed by a line-linked field; used to avoid inventing a second
    # reference field for the same string under a different name (mobile vs
    # complainant_contact), which would show up as a permanent phantom false negative.
    covered = {value for ref in reference.values() for value in ref.accepted}

    for raw_key, raw_value in (gt.get("fields") or {}).items():
        key = FIELD_KEY_ALIASES.get(raw_key, raw_key)
        if key in NON_FIELD_KEYS or raw_key in NON_FIELD_KEYS:
            continue
        value = normalize_value(raw_value)
        if not value:
            continue
        if key in reference:
            # `fields` is the canonical English/ASCII spelling. On a Devanagari-valued
            # field it is the translation, not what the page shows.
            reference[key].en.add(value)
            if key not in deva_valued:
                reference[key].native.add(value)
        elif (
            raw_key in EMBEDDED_FIELD_KEYS
            and len(value) >= EMBEDDED_MIN_LEN
            and value not in covered
            and any(value in normalize_value(t) for t in line_texts)
        ):
            # not its own line, but visibly embedded in one (e.g. the statement header)
            reference[key].native.add(value)
            reference[key].en.add(value)

    # A field with only one side populated (a line link but no bilingual entry, say) is
    # not bilingual — it is monolingual, and both questions have the same answer.
    for ref in reference.values():
        if not ref.en:
            ref.en = set(ref.native)
        elif not ref.native:
            ref.native = set(ref.en)
    return dict(reference)


def reference_escalation(gt: dict, mode: str) -> bool:
    """Whether tier-2 escalation *should* have fired, per --escalation-reference."""
    quality = gt.get("quality_expectation") != "clean"
    lines = [ln for ln in gt.get("lines", []) if isinstance(ln, dict)]
    hand = bool(lines) and (
        sum(1 for ln in lines if ln.get("is_handwritten")) / len(lines) >= HAND_MAJORITY_RATIO
    )
    if mode == "quality":
        return quality
    if mode == "hand":
        return hand
    return quality or hand


# -------------------------------------------------------------- line alignment


@dataclass
class LineMatch:
    gt_index: int
    pred_index: int
    iou: float
    matched_by: str  # "iou" | "text"


def match_lines(gt_lines: Sequence[dict], pred_lines: Sequence[PredLine]) -> list[LineMatch]:
    """One-to-one greedy alignment: IoU first, then text similarity for the leftovers."""
    matches: list[LineMatch] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    candidates: list[tuple[float, int, int]] = []
    for gi, gline in enumerate(gt_lines):
        gbox = gline.get("bbox")
        if to_aabb(gbox) is None:
            continue
        for pi, pline in enumerate(pred_lines):
            if pline.bbox is None:
                continue
            value = iou(gbox, pline.bbox)
            if value > 0.0:
                candidates.append((value, gi, pi))
    for value, gi, pi in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(LineMatch(gt_index=gi, pred_index=pi, iou=value, matched_by="iou"))

    leftovers: list[tuple[float, int, int]] = []
    for gi, gline in enumerate(gt_lines):
        if gi in used_gt:
            continue
        gtext = gline.get("text")
        if not isinstance(gtext, str):
            continue
        for pi, pline in enumerate(pred_lines):
            if pi in used_pred:
                continue
            sim = text_similarity(gtext, pline.text)
            if sim >= 0.5:
                leftovers.append((sim, gi, pi))
    for sim, gi, pi in sorted(leftovers, key=lambda c: (-c[0], c[1], c[2])):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(
            LineMatch(gt_index=gi, pred_index=pi, iou=iou(gt_lines[gi].get("bbox"), pred_lines[pi].bbox),
                      matched_by="text")
        )
    return matches


# ------------------------------------------------------------------ accumulator


class RateCell:
    """Sum of edit distances over sum of reference lengths — a micro average."""

    __slots__ = ("char_errors", "char_total", "lines", "word_errors", "word_total")

    def __init__(self) -> None:
        self.char_errors = 0
        self.char_total = 0
        self.word_errors = 0
        self.word_total = 0
        self.lines = 0

    def add(self, ref: str, hyp: str | None) -> None:
        ref_norm = normalize_for_cer(ref)
        ref_tokens = ref_norm.split(" ") if ref_norm else []
        self.lines += 1
        self.char_total += len(ref_norm)
        self.word_total += len(ref_tokens)
        if hyp is None:
            self.char_errors += len(ref_norm)
            self.word_errors += len(ref_tokens)
            return
        hyp_norm = normalize_for_cer(hyp)
        self.char_errors += levenshtein(ref_norm, hyp_norm)
        self.word_errors += levenshtein(ref_tokens, hyp_norm.split(" ") if hyp_norm else [])

    def as_dict(self) -> dict:
        return {
            "lines": self.lines,
            "ref_chars": self.char_total,
            "ref_words": self.word_total,
            "cer": (self.char_errors / self.char_total) if self.char_total else None,
            "wer": (self.word_errors / self.word_total) if self.word_total else None,
        }


class PRF:
    __slots__ = ("fn", "fp", "tp")

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def as_dict(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "precision": precision, "recall": recall, "f1": f1}


# ------------------------------------------------------------------- evaluation


def select_documents(docs: Sequence[CorpusDoc], split: str, final: bool) -> list[CorpusDoc]:
    """Apply the locked-split rule. Raises LockedSplitError / UnknownSplitError."""
    if split not in ("dev", "test", "all"):
        raise UnknownSplitError(f"unknown split {split!r}; expected dev, test or all")
    if split in ("test", "all") and not final:
        locked = sorted(LOCKED_TEST_VARIANTS)
        raise LockedSplitError(
            f"--split {split} would score the locked test split (variants {locked} of every "
            "type) and --final was not passed.\n"
            "Two of five variants per type are held out precisely so that tuning cannot "
            "reach them. Use --split dev while iterating; pass --final only for the run "
            "whose numbers you are going to publish."
        )
    if split == "dev":
        return [d for d in docs if not d.is_locked]
    if split == "test":
        return [d for d in docs if d.is_locked]
    return list(docs)


def evaluate(
    corpus_dir: Path,
    predictions_dir: Path,
    *,
    split: str = "dev",
    final: bool = False,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    escalation_reference: str = "quality_or_hand",
    types: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
) -> dict:
    """Score ``predictions_dir`` against ``corpus_dir``. Returns the machine JSON report."""
    if not predictions_dir.is_dir():
        raise PredictionsNotFoundError(f"No predictions directory at {predictions_dir}")
    index, all_docs = load_corpus(corpus_dir)
    selected = select_documents(all_docs, split, final)
    if types:
        selected = [d for d in selected if d.doc_type in set(types)]
    if profiles:
        selected = [d for d in selected if d.profile in set(profiles)]

    recognition: dict[tuple[str, str, str], RateCell] = defaultdict(RateCell)
    recognition_by_script: dict[str, RateCell] = defaultdict(RateCell)
    recognition_by_profile: dict[str, RateCell] = defaultdict(RateCell)
    recognition_overall = RateCell()

    det_iou_sum: dict[str, float] = defaultdict(float)
    det_matched: dict[str, int] = defaultdict(int)
    det_hits: dict[str, int] = defaultdict(int)
    det_gt: dict[str, int] = defaultdict(int)
    det_spurious: dict[str, int] = defaultdict(int)

    field_by_name: dict[str, PRF] = defaultdict(PRF)
    field_by_source: dict[str, PRF] = defaultdict(PRF)
    field_by_cell: dict[tuple[str, str], PRF] = defaultdict(PRF)
    # ground-truth occurrences per field, so recall by source has a real denominator:
    # "what share of the reference did THIS mechanism capture", not "of what it claimed".
    gt_field_counts: dict[str, int] = defaultdict(int)
    # Reported, never folded into precision/recall — see the module docstring.
    bilingual_counts: dict[str, int] = dict.fromkeys(
        (
            "gt_bilingual_fields",
            "claimed",
            "value_is_native",
            "value_is_translation",
            "value_en_correct",
            "value_en_wrong",
            "value_en_absent",
        ),
        0,
    )

    lid_correct: dict[tuple[str, str], int] = defaultdict(int)
    lid_matched: dict[tuple[str, str], int] = defaultdict(int)
    lid_total: dict[tuple[str, str], int] = defaultdict(int)

    escalation = PRF()
    quality_matrix: dict[tuple[str, str], int] = defaultdict(int)
    quality_correct = 0
    quality_scored = 0

    timings: dict[str, list[float]] = defaultdict(list)
    timings_by_profile: dict[tuple[str, str], list[float]] = defaultdict(list)

    schema_valid = 0
    schema_checked = 0
    schema_examples: list[dict] = []
    missing_predictions: list[str] = []
    docs_scored = 0

    for doc in selected:
        pred_path = find_prediction_file(predictions_dir, doc.doc_id)
        if pred_path is None:
            missing_predictions.append(doc.doc_id)
            continue
        try:
            payload = json.loads(pred_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            schema_checked += 1
            if len(schema_examples) < 10:
                schema_examples.append({"doc_id": doc.doc_id, "errors": [f"unreadable: {exc}"]})
            missing_predictions.append(doc.doc_id)
            continue

        gt = load_ground_truth(doc)
        pred = parse_prediction(payload, doc.doc_id)
        docs_scored += 1

        schema_checked += 1
        if pred.schema_valid:
            schema_valid += 1
        elif len(schema_examples) < 10:
            schema_examples.append({"doc_id": doc.doc_id, "errors": pred.schema_errors[:6]})

        gt_lines = [ln for ln in gt.get("lines", []) if isinstance(ln, dict)]

        # ---- recognition + detection + LID -------------------------------------
        if pred.has_lines_block:
            matches = match_lines(gt_lines, pred.lines)
            by_gt = {m.gt_index: m for m in matches}
            det_gt[doc.profile] += len(gt_lines)
            det_spurious[doc.profile] += max(0, len(pred.lines) - len(matches))

            for gi, gline in enumerate(gt_lines):
                gtext = gline.get("text")
                if not isinstance(gtext, str):
                    continue
                script = str(gline.get("script") or "unknown")
                modality = "hand" if gline.get("is_handwritten") else "print"
                match = by_gt.get(gi)
                hyp = pred.lines[match.pred_index].text if match else None

                recognition[(script, modality, doc.profile)].add(gtext, hyp)
                recognition_by_script[script].add(gtext, hyp)
                recognition_by_profile[doc.profile].add(gtext, hyp)
                recognition_overall.add(gtext, hyp)

                if match is not None:
                    det_matched[doc.profile] += 1
                    det_iou_sum[doc.profile] += match.iou
                    if match.iou >= iou_threshold:
                        det_hits[doc.profile] += 1

                gt_lang = gline.get("language")
                if isinstance(gt_lang, str) and gt_lang:
                    lid_total[(script, modality)] += 1
                    if match is not None:
                        lid_matched[(script, modality)] += 1
                        if pred.lines[match.pred_index].language == gt_lang:
                            lid_correct[(script, modality)] += 1

        # ---- fields ------------------------------------------------------------
        if pred.has_fields_block:
            reference = build_field_reference(gt)
            for name, ref in reference.items():
                gt_field_counts[name] += 1
                if ref.is_bilingual:
                    bilingual_counts["gt_bilingual_fields"] += 1
            # A prediction with the wrong value counts as BOTH a false positive and a
            # miss: the field was claimed (so precision pays) and it was not recovered
            # (so recall pays). Tracking only "was it claimed" would let a pipeline hide
            # every miss behind a confident wrong answer.
            hit_names: set[str] = set()
            for raw_name, spec in pred.fields.items():
                # The envelope's vocabulary, mapped onto the corpus's. Reporting is keyed
                # on the corpus name so TP/FP/FN line up with gt_occurrences.
                name = ENVELOPE_FIELD_ALIASES.get(raw_name, raw_name)
                source = spec.get("source") if isinstance(spec.get("source"), str) else "unspecified"
                value = spec.get("value") if isinstance(spec.get("value"), str) else ""
                value_en = spec.get("value_en") if isinstance(spec.get("value_en"), str) else ""
                normalised = spec.get("normalized") if isinstance(spec.get("normalized"), str) else ""
                candidates = {normalize_value(v) for v in (value, value_en, normalised) if v}
                ref = reference.get(name)
                accepted = ref.accepted if ref is not None else set()
                if candidates & accepted:
                    hit_names.add(name)
                    field_by_name[name].tp += 1
                    field_by_source[source].tp += 1
                    field_by_cell[(name, source)].tp += 1
                else:
                    field_by_name[name].fp += 1
                    field_by_source[source].fp += 1
                    field_by_cell[(name, source)].fp += 1

                # Which slot carried it, asked only where the two answers differ.
                if ref is None or not ref.is_bilingual:
                    continue
                bilingual_counts["claimed"] += 1
                if value:
                    normalised_value = normalize_value(value)
                    if normalised_value in ref.native:
                        bilingual_counts["value_is_native"] += 1
                    elif normalised_value in ref.en:
                        # right answer, wrong slot: `value` is contractually the page's
                        # own string and must not be replaced by its translation.
                        bilingual_counts["value_is_translation"] += 1
                if value_en:
                    key = "value_en_correct" if normalize_value(value_en) in ref.en else "value_en_wrong"
                    bilingual_counts[key] += 1
                else:
                    bilingual_counts["value_en_absent"] += 1
            for name in reference:
                if name not in hit_names:
                    field_by_name[name].fn += 1

        # ---- escalation --------------------------------------------------------
        if pred.escalated is not None:
            should = reference_escalation(gt, escalation_reference)
            if pred.escalated and should:
                escalation.tp += 1
            elif pred.escalated and not should:
                escalation.fp += 1
            elif not pred.escalated and should:
                escalation.fn += 1

        # ---- quality gate ------------------------------------------------------
        if pred.quality_verdict is not None:
            expectation = doc.quality_expectation
            quality_matrix[(expectation, pred.quality_verdict)] += 1
            quality_scored += 1
            if pred.quality_verdict in QUALITY_ACCEPTABLE.get(expectation, ()):
                quality_correct += 1

        # ---- latency -----------------------------------------------------------
        for stage, value in pred.timings_ms.items():
            timings[stage].append(value)
            timings_by_profile[(stage, doc.profile)].append(value)

    report = {
        "harness_version": HARNESS_VERSION,
        "corpus": {
            "dir": str(corpus_dir),
            "seed": index.get("seed"),
            "generator_version": index.get("generator_version"),
            "documents_total": len(all_docs),
            "caveat": index.get("caveat"),
        },
        "predictions_dir": str(predictions_dir),
        "split": {
            "name": split,
            "final": final,
            "locked_variants": sorted(LOCKED_TEST_VARIANTS),
            "documents_selected": len(selected),
            "documents_withheld": len(all_docs) - len(selected),
            "documents_scored": docs_scored,
            "missing_predictions": missing_predictions,
        },
        "settings": {
            "iou_threshold": iou_threshold,
            "escalation_reference": escalation_reference,
            "rapidfuzz": _HAVE_RAPIDFUZZ,
            "envelope_field_aliases": dict(ENVELOPE_FIELD_ALIASES),
        },
        "recognition": {
            "overall": recognition_overall.as_dict(),
            "by_script": {k: v.as_dict() for k, v in sorted(recognition_by_script.items())},
            "by_profile": {k: v.as_dict() for k, v in sorted(recognition_by_profile.items())},
            "by_script_modality_profile": {
                f"{s}|{m}|{p}": cell.as_dict() for (s, m, p), cell in sorted(recognition.items())
            },
        },
        "detection": {
            "by_profile": {
                profile: {
                    "gt_lines": det_gt[profile],
                    "matched": det_matched[profile],
                    "spurious": det_spurious[profile],
                    "mean_iou": (det_iou_sum[profile] / det_matched[profile]) if det_matched[profile] else None,
                    "line_recall": (det_hits[profile] / det_gt[profile]) if det_gt[profile] else None,
                }
                for profile in sorted(det_gt)
            },
            "overall": {
                "gt_lines": sum(det_gt.values()),
                "matched": sum(det_matched.values()),
                "spurious": sum(det_spurious.values()),
                "mean_iou": (
                    sum(det_iou_sum.values()) / sum(det_matched.values()) if sum(det_matched.values()) else None
                ),
                "line_recall": (
                    sum(det_hits.values()) / sum(det_gt.values()) if sum(det_gt.values()) else None
                ),
            },
        },
        "fields": {
            "gt_occurrences": dict(sorted(gt_field_counts.items())),
            "by_field": {k: v.as_dict() for k, v in sorted(field_by_name.items())},
            # recall here is the share of ALL reference field instances this mechanism
            # captured, so the per-source recalls sum to the overall recall.
            "by_source": {
                k: _attributed(v, sum(gt_field_counts.values())) for k, v in sorted(field_by_source.items())
            },
            "by_field_source": {
                f"{n}|{s}": _attributed(v, gt_field_counts.get(n, 0))
                for (n, s), v in sorted(field_by_cell.items())
            },
            # Diagnostic only. Counted over fields whose native spelling and English
            # equivalent differ — in practice the Devanagari-valued ones. It answers
            # "did the immutable `value` carry the page's own string, or its
            # translation", which precision and recall cannot distinguish.
            "bilingual": dict(bilingual_counts),
        },
        "schema": {
            "checked": schema_checked,
            "valid": schema_valid,
            "validity_rate": (schema_valid / schema_checked) if schema_checked else None,
            "examples": schema_examples,
        },
        "escalation": {
            "reference_rule": escalation_reference,
            **escalation.as_dict(),
        },
        "quality_gate": {
            "scored": quality_scored,
            "expectation_satisfied_rate": (quality_correct / quality_scored) if quality_scored else None,
            "confusion": {
                expectation: {
                    verdict: quality_matrix.get((expectation, verdict), 0) for verdict in QUALITY_VERDICTS
                }
                for expectation in QUALITY_EXPECTATIONS
            },
        },
        "language_id": {
            "overall": _lid_dict(
                sum(lid_correct.values()), sum(lid_matched.values()), sum(lid_total.values())
            ),
            "by_script_modality": {
                f"{s}|{m}": _lid_dict(lid_correct[(s, m)], lid_matched[(s, m)], lid_total[(s, m)])
                for (s, m) in sorted(lid_total)
            },
        },
        "latency_ms": {
            "by_stage": {
                stage: {
                    "n": len(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                }
                for stage, values in sorted(timings.items())
            },
            "by_stage_profile": {
                f"{stage}|{profile}": {
                    "n": len(values),
                    "p50": percentile(values, 0.50),
                    "p95": percentile(values, 0.95),
                }
                for (stage, profile), values in sorted(timings_by_profile.items())
            },
        },
    }
    return report


def _attributed(prf: PRF, gt_total: int) -> dict:
    """PRF where recall is measured against the ground truth, not against what was claimed.

    Without this, a mechanism that extracts one field perfectly and ignores everything else
    reports recall 1.0 — technically true of its own output, and useless for deciding where
    the pipeline is losing fields.
    """
    result = prf.as_dict()
    result["gt_total"] = gt_total
    result["fn"] = max(0, gt_total - prf.tp)
    result["recall"] = (prf.tp / gt_total) if gt_total else None
    precision, recall = result["precision"], result["recall"]
    if precision is None or recall is None or (precision + recall) == 0:
        result["f1"] = None
    else:
        result["f1"] = 2 * precision * recall / (precision + recall)
    return result


def _lid_dict(correct: int, matched: int, total: int) -> dict:
    return {
        "gt_lines": total,
        "matched": matched,
        "correct": correct,
        "accuracy_matched": (correct / matched) if matched else None,
        "accuracy_all_lines": (correct / total) if total else None,
        "coverage": (matched / total) if total else None,
    }


# ----------------------------------------------------------------- presentation


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = [f"\n{title}", "-" * min(100, sum(widths) + 3 * (len(widths) - 1))]
    if not rows:
        lines.append("  (no data)")
        return "\n".join(lines)
    for ri, row in enumerate(cells):
        rendered = "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        )
        lines.append(rendered)
        if ri == 0:
            lines.append("-" * len(rendered))
    return "\n".join(lines)


def render_report(report: dict) -> str:
    out: list[str] = []
    split = report["split"]
    out.append("=" * 78)
    out.append(f"RBI OCR corpus evaluation — harness {report['harness_version']}")
    out.append("=" * 78)
    out.append(f"corpus       {report['corpus']['dir']}  (seed {report['corpus']['seed']}, "
               f"generator {report['corpus']['generator_version']})")
    out.append(f"predictions  {report['predictions_dir']}")
    out.append(
        f"split        {split['name']}  final={split['final']}  "
        f"locked variants {split['locked_variants']}"
    )
    out.append(
        f"documents    {split['documents_scored']} scored / {split['documents_selected']} selected / "
        f"{split['documents_withheld']} withheld"
    )
    if split["missing_predictions"]:
        shown = ", ".join(split["missing_predictions"][:5])
        more = "" if len(split["missing_predictions"]) <= 5 else f" (+{len(split['missing_predictions']) - 5} more)"
        out.append(f"MISSING      {len(split['missing_predictions'])} prediction files: {shown}{more}")

    rec = report["recognition"]
    out.append(
        render_table(
            "Recognition — CER / WER by script x modality x profile",
            ["cell", "lines", "ref_chars", "CER", "WER"],
            [
                [key, v["lines"], v["ref_chars"], v["cer"], v["wer"]]
                for key, v in rec["by_script_modality_profile"].items()
            ],
        )
    )
    out.append(
        render_table(
            "Recognition — rollups",
            ["cell", "lines", "ref_chars", "CER", "WER"],
            [[f"script:{k}", v["lines"], v["ref_chars"], v["cer"], v["wer"]] for k, v in rec["by_script"].items()]
            + [[f"profile:{k}", v["lines"], v["ref_chars"], v["cer"], v["wer"]] for k, v in rec["by_profile"].items()]
            + [["OVERALL", rec["overall"]["lines"], rec["overall"]["ref_chars"], rec["overall"]["cer"],
                rec["overall"]["wer"]]],
        )
    )

    det = report["detection"]
    out.append(
        render_table(
            "Detection — IoU and line recall by profile",
            ["profile", "gt_lines", "matched", "spurious", "mean_IoU", "line_recall"],
            [
                [k, v["gt_lines"], v["matched"], v["spurious"], v["mean_iou"], v["line_recall"]]
                for k, v in det["by_profile"].items()
            ]
            + [
                [
                    "OVERALL",
                    det["overall"]["gt_lines"],
                    det["overall"]["matched"],
                    det["overall"]["spurious"],
                    det["overall"]["mean_iou"],
                    det["overall"]["line_recall"],
                ]
            ],
        )
    )

    fields = report["fields"]
    out.append(
        render_table(
            "Fields — precision / recall / F1 by field",
            ["field", "TP", "FP", "FN", "P", "R", "F1"],
            [[k, v["tp"], v["fp"], v["fn"], v["precision"], v["recall"], v["f1"]] for k, v in fields["by_field"].items()],
        )
    )
    out.append(
        render_table(
            "Fields — by source mechanism (R = share of ALL reference fields this source captured)",
            ["source", "TP", "FP", "GT", "P", "R", "F1"],
            [
                [k, v["tp"], v["fp"], v["gt_total"], v["precision"], v["recall"], v["f1"]]
                for k, v in fields["by_source"].items()
            ],
        )
    )
    out.append(
        render_table(
            "Fields — field x source (R = share of that field's reference this source captured)",
            ["field|source", "TP", "FP", "GT", "P", "R", "F1"],
            [
                [k, v["tp"], v["fp"], v["gt_total"], v["precision"], v["recall"], v["f1"]]
                for k, v in fields["by_field_source"].items()
            ],
        )
    )

    bilingual = fields.get("bilingual") or {}
    out.append(
        render_table(
            "Fields — bilingual slots (fields whose native spelling differs from its English)",
            ["metric", "count"],
            [
                ["gt occurrences", bilingual.get("gt_bilingual_fields", 0)],
                ["claimed", bilingual.get("claimed", 0)],
                ["value = page's own string", bilingual.get("value_is_native", 0)],
                ["value = its translation", bilingual.get("value_is_translation", 0)],
                ["value_en correct", bilingual.get("value_en_correct", 0)],
                ["value_en wrong", bilingual.get("value_en_wrong", 0)],
                ["value_en absent", bilingual.get("value_en_absent", 0)],
            ],
        )
    )
    out.append(
        "    `value` is contractually the page's own string; a translation there is a "
        "contract violation this table counts but precision/recall cannot see."
    )

    schema = report["schema"]
    out.append(
        render_table(
            "Schema validity",
            ["metric", "value"],
            [["checked", schema["checked"]], ["valid", schema["valid"]], ["validity_rate", schema["validity_rate"]]],
        )
    )
    for example in schema["examples"][:3]:
        out.append(f"    {example['doc_id']}: {'; '.join(example['errors'][:3])}")

    esc = report["escalation"]
    out.append(
        render_table(
            f"Escalation vs reference rule '{esc['reference_rule']}'",
            ["TP", "FP", "FN", "precision", "recall", "F1"],
            [[esc["tp"], esc["fp"], esc["fn"], esc["precision"], esc["recall"], esc["f1"]]],
        )
    )

    quality = report["quality_gate"]
    out.append(
        render_table(
            "Quality gate — confusion (rows: quality_expectation, cols: verdict)",
            ["expectation", *QUALITY_VERDICTS],
            [[exp, *[quality["confusion"][exp][v] for v in QUALITY_VERDICTS]] for exp in QUALITY_EXPECTATIONS],
        )
    )
    out.append(f"    expectation-satisfied rate: {_fmt(quality['expectation_satisfied_rate'])} "
               f"over {quality['scored']} documents")

    lid = report["language_id"]
    out.append(
        render_table(
            "Language ID — LINE level",
            ["cell", "gt_lines", "matched", "correct", "acc_matched", "acc_all", "coverage"],
            [
                [k, v["gt_lines"], v["matched"], v["correct"], v["accuracy_matched"], v["accuracy_all_lines"],
                 v["coverage"]]
                for k, v in lid["by_script_modality"].items()
            ]
            + [
                [
                    "OVERALL",
                    lid["overall"]["gt_lines"],
                    lid["overall"]["matched"],
                    lid["overall"]["correct"],
                    lid["overall"]["accuracy_matched"],
                    lid["overall"]["accuracy_all_lines"],
                    lid["overall"]["coverage"],
                ]
            ],
        )
    )

    latency = report["latency_ms"]
    out.append(
        render_table(
            "Latency (ms) by stage",
            ["stage", "n", "p50", "p95"],
            [[k, v["n"], v["p50"], v["p95"]] for k, v in latency["by_stage"].items()],
        )
    )
    if latency["by_stage_profile"]:
        out.append(
            render_table(
                "Latency (ms) by stage x profile",
                ["stage|profile", "n", "p50", "p95"],
                [[k, v["n"], v["p50"], v["p95"]] for k, v in latency["by_stage_profile"].items()],
            )
        )

    out.append("")
    out.append(f"CAVEAT  {report['corpus']['caveat']}")
    if not split["final"]:
        out.append(
            f"NOTE    variants {split['locked_variants']} are the locked test split and were not scored. "
            "Pass --split all --final to include them."
        )
    return "\n".join(out)


# ---------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score pipeline predictions against the synthetic corpus.")
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--predictions", type=Path, required=True, help="directory of <doc_id>.json files")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument(
        "--final",
        action="store_true",
        help="unlock the held-out test split (variants 3 and 4). Use only for published runs.",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    ap.add_argument(
        "--escalation-reference",
        choices=("quality", "hand", "quality_or_hand"),
        default="quality_or_hand",
    )
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true", help="write JSON only, no table")
    a = ap.parse_args(argv)

    try:
        report = evaluate(
            a.corpus,
            a.predictions,
            split=a.split,
            final=a.final,
            iou_threshold=a.iou_threshold,
            escalation_reference=a.escalation_reference,
            types=a.types,
            profiles=a.profiles,
        )
    except LockedSplitError as exc:
        print(f"\nLockedSplitError: {exc}\n", file=sys.stderr)
        return 3
    except EvalHarnessError as exc:
        print(f"\n{type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 2

    if a.json_out is not None:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not a.quiet:
        print(render_report(report))
    if a.json_out is not None:
        print(f"\nJSON report -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
