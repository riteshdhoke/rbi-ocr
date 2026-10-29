"""M7 SLM bake-off — which small instruct model should Stage 5b use?

BUILD-BRIEF M7 requires "an SLM bake-off across >=3 candidates via the HF API,
with a written recommendation", and OPEN-QUESTIONS **Q5** ("does the router
support grammars, or JSON mode only?") says to discover the answer here rather
than assume it.  This tool answers both, on real router traffic, and writes
``docs/bakeoff/results.json`` plus ``docs/bakeoff/REPORT.md``.

**Residency-conformant by construction.**  Only corpus text crosses the
network, every file is checked through
:func:`app.providers.factory.require_synthetic_corpus_file` before it is read,
and the prompt is built by the production
:func:`app.providers.hf_api.build_extraction_prompt`, which sentinel-masks every
span the deterministic validators claimed.  The run additionally *verifies* that
no GT identifier value survives into the rendered prompt and reports the count
(``identifier_leaks``); a non-zero count is a build bug, not a metric.

**What is measured, per candidate**

* router availability — is the model actually served, and by which providers,
  with what advertised ``supports_structured_output``;
* **Q5** — whether ``response_format={"type": "json_object"}`` is accepted and
  whether it changes the answer, measured against the same request without it;
* schema validity: first attempt, after the provider's single repair turn, and
  final (PRD P0-8's headline number);
* grounding-gate pass rate (PLAN.md invariant 8) — the share of proposed values
  that actually occur in the page text;
* field agreement against ground truth, two ways: **line attribution** (did the
  accepted value come from the line the corpus tagged with that field?) and
  **value agreement** (exact and fuzzy against the GT ``fields`` dict);
* extractive-summary line-index precision / recall / F1 (PLAN.md invariant 2 —
  the model returns indices, never prose);
* latency p50/p95 over uncached calls, and tokens per document against
  ``JOB_TOKEN_BUDGET``.

**Ground truth uses the OCR lines from ``.gt.json``, not real OCR output.**  A
bake-off run over real OCR would score OCR and the SLM together and could not
say which one moved; ``tools/eval_harness.py`` owns the end-to-end number.
Variants 0-2 only: variants 3 and 4 are the locked test split (TEST-DATA-SPEC
sec. 7) and a model chosen against them would be chosen on the test set.

Usage::

    .venv\\Scripts\\python.exe tools/slm_bakeoff.py                  # full run
    .venv\\Scripts\\python.exe tools/slm_bakeoff.py --model X --model Y
    .venv\\Scripts\\python.exe tools/slm_bakeoff.py --docs 6         # quick pass
    .venv\\Scripts\\python.exe tools/slm_bakeoff.py --dry-run        # no network

The response cache is **on** by default, so a rerun costs nothing and the
report is reproducible; ``--no-cache`` forces live calls.

This tool does NOT handle:
  * the VLM (OPEN-QUESTIONS Q2).  Tier-2 escalation is a separate question with
    a separate cost profile;
  * translation quality (Q7) — ``value_en`` never enters this comparison;
  * choosing the model.  It ranks and recommends; changing ``SLM_MODEL`` in
    ``app/settings.py`` and ``.env.example`` is an owner decision recorded in
    OPEN-QUESTIONS;
  * local llama.cpp / GGUF candidates.  PLAN.md D1 forbids >=0.9B weights on
    this machine, so every candidate here is reached over HTTP;
  * accuracy of the deterministic fields.  Invariant 7 keeps them away from the
    SLM entirely, and this tool asserts that rather than measuring it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `app` first: it sets the thread caps before any numeric import (invariant 5).
import app  # noqa: I001, F401

import httpx
from prometheus_client import REGISTRY
from rapidfuzz import fuzz

from app.contracts.errors import PipelineError, ProviderError, TokenBudgetExceeded
from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    OCRLine,
    OCRPage,
    OCRTier,
    PageClassLabel,
    RawField,
)
from app.providers.base import inject_truststore
from app.providers.cache import ResponseCache
from app.providers.factory import require_synthetic_corpus_file
from app.providers.hf_api import HFAPIProvider, build_extraction_prompt
from app.settings import Settings, get_settings
from app.stages.s5_slm import (
    DEFAULT_TARGET_FIELDS,
    apply_grounding_gate,
    build_extraction_context,
    extract_pages,
)
from app.stages.s5_validators import normalise_digits, run_validators

__all__ = [
    "DEFAULT_CANDIDATES",
    "FUZZY_MATCH_FLOOR",
    "GT_FIELD_ALIASES",
    "OUTCOME_LEDGER_FILENAME",
    "Q5_CACHE_FILENAME",
    "RANK_WEIGHTS",
    "REPORT_SCHEMA_VERSION",
    "SAMPLE_TYPES",
    "SAMPLE_VARIANTS",
    "CandidateResult",
    "DocResult",
    "GoldDoc",
    "StubProvider",
    "build_parser",
    "build_report",
    "composite_score",
    "fetch_router_catalogue",
    "json_mode_probe",
    "load_gold_docs",
    "load_outcomes",
    "load_q5_cache",
    "main",
    "merge_outcomes",
    "normalise_value",
    "percentile",
    "probe_is_conclusive",
    "rank_candidates",
    "render_markdown",
    "run_candidate",
    "save_outcomes",
    "save_q5_cache",
    "summarise_catalogue_entry",
]

REPORT_SCHEMA_VERSION: int = 1

OUTPUT_DIR: Path = PROJECT_ROOT / "docs" / "bakeoff"

#: Candidates beyond the two the settings name.  Chosen to span the axes that
#: plausibly matter for this task: parameter count (4B -> 8B -> 14B), vendor
#: (Qwen / Google / Meta / Microsoft), and advertised structured-output support.
#: Availability is probed, never assumed — the router's catalogue moves.
EXTRA_CANDIDATES: tuple[str, ...] = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

CANDIDATE_NOTES: Mapping[str, str] = {
    "Qwen/Qwen3-4B-Instruct-2507": (
        "closest live relative of the configured Qwen3-1.7B-Instruct; smallest "
        "instruct model on the router with advertised structured output"
    ),
    "google/gemma-3-4b-it": (
        "same weight class, different vendor and tokenizer; Gemma's Devanagari "
        "coverage is the reason it is here"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "8B reference point: shows what the extra parameters buy on an "
        "extractive task"
    ),
    "Qwen/Qwen2.5-7B-Instruct": (
        "previous-generation 7B, as a check that 'newer' is doing the work"
    ),
}

#: Corpus types sampled, chosen for SLM-eligible field coverage across scripts.
#: ``email_printout`` carries only ``prior_complaint_ref`` and is included as
#: the negative case: a model that invents a name and an address on a document
#: that has neither is disqualified regardless of how it scores elsewhere.
SAMPLE_TYPES: tuple[str, ...] = (
    "complaint_form_en",
    "complaint_form_deva",
    "complaint_letter_en",
    "complaint_letter_mar",
    "bilingual_form",
    "email_printout",
)

#: Variants 3 and 4 are the locked test split (TEST-DATA-SPEC sec. 7).
SAMPLE_VARIANTS: tuple[int, ...] = (0, 1, 2)

#: Degradation profile of the sampled pages.  Irrelevant to a text-only
#: bake-off — every profile of a document renders the same GT lines — so the
#: cheapest one is used and the choice is recorded rather than randomised.
SAMPLE_PROFILE: str = "clean"

#: Corpus ``fields`` keys that carry a contract field under a different name.
GT_FIELD_ALIASES: Mapping[str, str] = {
    "complainant_address": "complainant_city",
    "prior_complaint_reference": "prior_complaint_ref",
}

#: Corpus *line* tags that carry a contract field under a different name.
GT_LINE_TAG_ALIASES: Mapping[str, str] = {
    "prior_complaint_reference": "prior_complaint_ref",
}

#: rapidfuzz ratio at or above which a value counts as a fuzzy match.
FUZZY_MATCH_FLOOR: float = 85.0

#: Composite ranking weights.  Schema validity and grounding dominate because a
#: field that fails either is not merely wrong, it is unusable: the assembler
#: drops it and the document escalates.  Attribution and summary F1 measure
#: whether the model is pointing at the right text; value agreement is
#: deliberately *not* in the score — OPEN-QUESTIONS Q6 means the corpus GT
#: carries English values on Devanagari pages, so it would penalise exactly the
#: multilingual behaviour we want.
RANK_WEIGHTS: Mapping[str, float] = {
    "schema_validity_final": 0.30,
    "grounding_pass_rate": 0.30,
    "line_attribution_accuracy": 0.20,
    "summary_f1": 0.20,
}

_ROUTER_MODELS_PATH = "/models"


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GoldDoc:
    """One corpus document, as the SLM stage would see it plus its answers."""

    doc_id: str
    doc_type: str
    variant: int
    script: str
    language: str | None
    is_code_mixed: bool
    page: OCRPage
    page_class: PageClassLabel
    #: ``fields`` from the ``.gt.json`` — see OPEN-QUESTIONS Q6 on scripts.
    gt_fields: Mapping[str, Any]
    #: Contract field name -> the line indices the corpus tagged with it.
    tagged_lines: Mapping[str, tuple[int, ...]]
    #: Line indices tagged ``complaint_summary``.
    summary_lines: tuple[int, ...]
    image_path: Path

    @property
    def expected_fields(self) -> tuple[str, ...]:
        """SLM-eligible fields this document actually contains."""
        return tuple(
            sorted(
                name
                for name in DEFAULT_TARGET_FIELDS
                if name != SUMMARY_FIELD_NAME and self.tagged_lines.get(name)
            )
        )


def _page_class_for(payload: Mapping[str, Any]) -> PageClassLabel:
    """The corpus's own ``document_class``, or ``unknown``.

    Taken from ground truth rather than mapped from the type name: the page
    class is one of the few things the prompt tells the model about the page,
    and inventing it here would make the bake-off measure a classifier this run
    never ran.
    """
    raw = str(payload.get("document_class", "")).strip()
    try:
        return PageClassLabel(raw)
    except ValueError:
        return PageClassLabel.UNKNOWN


def load_gold_docs(
    corpus_dir: Path,
    *,
    settings: Settings,
    types: Sequence[str] = SAMPLE_TYPES,
    variants: Sequence[int] = SAMPLE_VARIANTS,
    profile: str = SAMPLE_PROFILE,
    limit: int | None = None,
) -> list[GoldDoc]:
    """Build the fixed document sample from the committed ground truth.

    Ordering is ``type`` then ``variant``, both fixed, so the sample is the same
    on every machine and every run — a bake-off whose sample moves cannot
    compare two candidates measured a day apart.

    Every document is passed through
    :func:`~app.providers.factory.require_synthetic_corpus_file` before its text
    is read.  Under the laptop profile that is the residency gate; a document
    without a ``synthetic: true`` declaration raises rather than being skipped
    quietly.
    """
    docs: list[GoldDoc] = []
    for doc_type in types:
        for variant in variants:
            doc_id = f"{doc_type}_v{variant}_{profile}"
            gt_path = corpus_dir / doc_type / f"{doc_id}.gt.json"
            if not gt_path.is_file():
                continue
            payload = json.loads(gt_path.read_text(encoding="utf-8"))
            image_path = _image_for(gt_path, payload)
            require_synthetic_corpus_file(image_path, settings)

            lines: list[OCRLine] = []
            tagged: dict[str, list[int]] = {}
            for index, raw in enumerate(payload["lines"]):
                lines.append(
                    OCRLine(
                        text=str(raw["text"]),
                        bbox=[float(value) for value in raw["bbox"]],
                        confidence=1.0,
                        script=str(raw["script"]),
                        language=raw.get("language"),
                        is_handwritten=bool(raw.get("is_handwritten", False)),
                        line_index=index,
                    )
                )
                tag = raw.get("field")
                if tag:
                    tagged.setdefault(str(tag), []).append(index)

            contract_tags = {
                name: tuple(tagged.get(GT_LINE_TAG_ALIASES.get(name, name), ()))
                for name in DEFAULT_TARGET_FIELDS
            }
            docs.append(
                GoldDoc(
                    doc_id=doc_id,
                    doc_type=doc_type,
                    variant=variant,
                    script=str(payload.get("primary_script", "Latn")),
                    language=payload.get("primary_language"),
                    is_code_mixed=bool(payload.get("is_code_mixed", False)),
                    page=OCRPage(
                        page_no=1,
                        width_px=max(
                            1, int(max(line.bbox[2] for line in lines)) + 1
                        ),
                        height_px=max(
                            1, int(max(line.bbox[5] for line in lines)) + 1
                        ),
                        lines=lines,
                        tier=OCRTier.TIER1,
                    ),
                    page_class=_page_class_for(payload),
                    gt_fields=dict(payload.get("fields", {})),
                    tagged_lines={
                        name: value
                        for name, value in contract_tags.items()
                        if name != SUMMARY_FIELD_NAME
                    },
                    summary_lines=tuple(tagged.get(SUMMARY_FIELD_NAME, ())),
                    image_path=image_path,
                )
            )
            if limit is not None and len(docs) >= limit:
                return docs
    return docs


def _image_for(gt_path: Path, payload: Mapping[str, Any]) -> Path:
    """The rendered page beside a ``.gt.json``, whatever extension it carries."""
    stem = gt_path.name.removesuffix(".gt.json")
    for candidate in sorted(gt_path.parent.glob(f"{stem}.*")):
        if candidate.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf", ".tif"):
            return candidate
    raise FileNotFoundError(
        f"no rendered page beside {gt_path} for document "
        f"{payload.get('doc_id', stem)!r}"
    )


def gt_value_for(doc: GoldDoc, contract_field: str) -> str | None:
    """The corpus ``fields`` value for a contract field, or ``None``.

    ``relief_sought`` has no ``fields`` entry, so the tagged line's own text is
    the ground truth for it.  Devanagari documents carry **English** values here
    (OPEN-QUESTIONS Q6), which is why value agreement is reported per script and
    kept out of the composite score.
    """
    key = GT_FIELD_ALIASES.get(contract_field, contract_field)
    value = doc.gt_fields.get(key)
    if value is not None:
        return str(value)
    indices = doc.tagged_lines.get(contract_field, ())
    if indices:
        by_index = {line.line_index: line for line in doc.page.lines}
        return by_index[indices[0]].text
    return None


def normalise_value(text: str) -> str:
    """Comparison form: Indic digits to ASCII, case-folded, whitespace collapsed."""
    return " ".join(normalise_digits(text).lower().split())


# --------------------------------------------------------------------------- #
# Router probes
# --------------------------------------------------------------------------- #


def fetch_router_catalogue(
    settings: Settings, *, api_key: str, timeout: float = 60.0
) -> dict[str, Any]:
    """``GET {HF_BASE_URL}/models`` — what the router actually serves today.

    Recorded verbatim (minus the parts nobody will read) because the whole point
    of Q5 is that the answer is a property of the *router on the day*, and a
    report that only says "we used model X" cannot be re-checked later.
    """
    inject_truststore()
    url = settings.HF_BASE_URL.rstrip("/") + _ROUTER_MODELS_PATH
    response = httpx.get(
        url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    return {
        "url": url,
        "status_code": response.status_code,
        "model_count": len(entries),
        "models": {
            str(entry.get("id")): summarise_catalogue_entry(entry)
            for entry in entries
            if isinstance(entry, dict)
        },
    }


def summarise_catalogue_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The provider-level facts that decide whether a candidate is usable."""
    providers = entry.get("providers") or []
    return {
        "owned_by": entry.get("owned_by"),
        "input_modalities": (entry.get("architecture") or {}).get(
            "input_modalities"
        ),
        "providers": [
            {
                "provider": item.get("provider"),
                "status": item.get("status"),
                "context_length": item.get("context_length"),
                "supports_structured_output": item.get(
                    "supports_structured_output"
                ),
                "pricing_usd_per_million": item.get("pricing"),
            }
            for item in providers
            if isinstance(item, dict)
        ],
        "advertises_structured_output": any(
            isinstance(item, dict) and item.get("supports_structured_output") is True
            for item in providers
        ),
    }


def json_mode_probe(
    settings: Settings,
    model: str,
    *,
    api_key: str,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """OPEN-QUESTIONS **Q5**, measured: does the router honour JSON mode?

    Three raw calls, deliberately outside :class:`HFAPIProvider` so nothing —
    not the cache, not the repair retry, not the Pydantic validator — can make a
    model look better than the wire does:

    1. ``response_format={"type": "json_object"}`` — is it accepted at all, and
       does the content parse as JSON?
    2. the same request with **no** ``response_format`` — the control, so a
       model that always emits JSON is not credited to JSON mode;
    3. ``response_format={"type": "json_schema", ...}`` — the stronger form
       (constrained decoding against a schema). A 400 here with a 200 above is
       exactly the "JSON mode only, no grammar" answer Q5 asks for.

    Every call is a fixed, tiny prompt so the probe costs a few tokens and is
    comparable across candidates.
    """
    inject_truststore()
    url = settings.HF_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {
            "role": "system",
            "content": "You are an extraction engine. Reply with JSON only.",
        },
        {
            "role": "user",
            "content": (
                'Return the object {"fields": [{"name": "complainant_name", '
                '"value": "Asha Patil", "confidence": 1.0, "line_indices": '
                "[0]}]} exactly."
            ),
        },
    ]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["fields"],
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "confidence"],
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "line_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                },
            }
        },
    }
    variants: dict[str, dict[str, Any] | None] = {
        "json_object": {"type": "json_object"},
        "none": None,
        "json_schema": {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "strict": True,
                "schema": schema,
            },
        },
    }

    results: dict[str, Any] = {}
    for label, response_format in variants.items():
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 96,
            "temperature": 0,
        }
        if response_format is not None:
            body["response_format"] = response_format
        started = time.perf_counter()
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            results[label] = {
                "status_code": None,
                "transport_error": f"{type(exc).__name__}: {exc}",
            }
            continue
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {
            "status_code": response.status_code,
            "latency_seconds": round(elapsed, 3),
        }
        if response.status_code != 200:
            record["error"] = response.text[:400]
            results[label] = record
            continue
        payload = response.json()
        choices = payload.get("choices") or []
        content = (
            choices[0].get("message", {}).get("content") if choices else None
        )
        record["usage"] = payload.get("usage")
        record["content_preview"] = (content or "")[:200]
        record["parses_as_json"] = _parses_as_json(content)
        results[label] = record

    honours = bool(
        results.get("json_object", {}).get("status_code") == 200
        and results.get("json_object", {}).get("parses_as_json")
    )
    return {
        "variants": results,
        "json_object_accepted": results.get("json_object", {}).get("status_code")
        == 200,
        "json_object_yields_json": honours,
        "json_schema_accepted": results.get("json_schema", {}).get("status_code")
        == 200,
        "unconstrained_yields_json": bool(
            results.get("none", {}).get("parses_as_json")
        ),
    }


#: Filename of the persisted Q5 probe results, beside the report.
Q5_CACHE_FILENAME: str = "q5_probes.json"

#: Filename of the persisted per-document schema outcomes, beside the report.
OUTCOME_LEDGER_FILENAME: str = "schema_outcomes.json"


def load_outcomes(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-document schema outcomes observed by previous runs.

    Why this file has to exist: the response cache stores **only answers that
    validated**, so on a re-run the cached documents are guaranteed successes
    and the freshly-called ones are exactly the previous failures. Scoring
    either set alone is biased — including cache hits drives schema validity to
    1.0, excluding them drives it to 0.0. Both were observed on real re-runs.

    The ledger fixes it by remembering each document's *last observed* outcome,
    so schema validity is computed over the union of everything ever seen and a
    re-run improves the estimate instead of corrupting it. Temperature is 0, so
    a document's outcome is stable and remembering it is sound.
    """
    path = out_dir / OUTCOME_LEDGER_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(model): dict(entries)
        for model, entries in payload.items()
        if isinstance(entries, dict)
    }


def save_outcomes(out_dir: Path, ledger: Mapping[str, Mapping[str, Any]]) -> None:
    """Persist the outcome ledger for the next run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUTCOME_LEDGER_FILENAME).write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def merge_outcomes(
    ledger: dict[str, dict[str, Any]], result: CandidateResult
) -> dict[str, Any]:
    """Fold one candidate's observations into ``ledger``.

    Two rules, and the asymmetry between them is the whole point:

    * a **fresh** answer overwrites whatever was there — it is a new
      measurement, and at temperature 0 it is the current truth;
    * a **cache hit** is recorded as valid *only if the document is not already
      known*. A hit is genuine evidence the document validates (that is the
      only reason it is in the cache), but it carries no first-attempt or
      repair information, so it must not overwrite a fresh record that does.

    Scoring the union is what makes schema validity stable across re-runs.
    Scoring only the cache hits gives 1.0; scoring only the fresh answers gives
    0.0, because the failures are exactly the documents the cache never stored.
    Both were observed on real re-runs before this ledger existed.

    Returns the merged per-document view for this candidate.
    """
    entries = ledger.setdefault(result.model, {})
    stamp = datetime.now(UTC).isoformat()
    for doc in result.docs:
        if not doc.responded:
            continue
        if doc.cached:
            entries.setdefault(
                doc.doc_id,
                {
                    "valid": True,
                    "first_attempt_valid": None,
                    "repaired": False,
                    "from_cache": True,
                    "observed_at": stamp,
                },
            )
            continue
        entries[doc.doc_id] = {
            "valid": doc.ok,
            "first_attempt_valid": doc.first_attempt_valid,
            "repaired": doc.repaired,
            "from_cache": False,
            "observed_at": stamp,
        }
    return entries


def load_q5_cache(out_dir: Path) -> dict[str, Any]:
    """Previously measured Q5 probes, keyed by model.

    Persisted because a probe is a *dated observation of the router*, not a
    derived number: once the account's inference credits are gone, re-measuring
    is impossible and a run that silently reported "unknown" would erase a
    finding that was genuinely obtained.  Each entry carries ``probed_at`` so a
    stale answer is visible as stale rather than passed off as current.
    """
    path = out_dir / Q5_CACHE_FILENAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_q5_cache(out_dir: Path, probes: Mapping[str, Any]) -> None:
    """Persist the Q5 probes for the next run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / Q5_CACHE_FILENAME).write_text(
        json.dumps(probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


#: Statuses that say something about the *account or the network*, not about
#: the model's structured-output support.
INCONCLUSIVE_STATUS_CODES: frozenset[int] = frozenset({402, 408, 425, 429})


def probe_is_conclusive(probe: Mapping[str, Any]) -> bool:
    """Whether a probe learned something about the router, on **every** variant.

    All three variants must have received a decisive answer.  Requiring only one
    would let a probe whose ``json_schema`` call was refused with 402 be stored
    as ``json_schema_accepted: false`` — turning a depleted credit balance into
    a permanent, wrong claim about a model's capabilities, which is the exact
    failure mode this whole function exists to prevent.  A 422 *is* decisive: it
    is the provider saying it will not take that response format.
    """
    variants = probe.get("variants") or {}
    if not variants:
        return False
    for record in variants.values():
        if not isinstance(record, dict) or record.get("transport_error"):
            return False
        status = record.get("status_code")
        if not isinstance(status, int):
            return False
        if status in INCONCLUSIVE_STATUS_CODES or status >= 500:
            return False
    return True


def _parses_as_json(content: str | None) -> bool:
    """Whether a completion body is a JSON object, fence tolerated."""
    if not content:
        return False
    text = content.strip()
    if text.startswith("```"):
        body = text[3:]
        if body.lower().startswith("json"):
            body = body[4:]
        text = body.rsplit("```", 1)[0].strip()
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


@dataclass
class DocResult:
    """One candidate's outcome on one document."""

    doc_id: str
    doc_type: str
    script: str
    ok: bool
    error: str | None = None
    error_class: str | None = None
    cached: bool = False
    latency_seconds: float = 0.0
    tokens_used: int = 0
    first_attempt_valid: bool = True
    repaired: bool = False
    proposed_value_fields: int = 0
    grounded_value_fields: int = 0
    rejected_fields: tuple[str, ...] = ()
    attributed_fields: int = 0
    attributable_fields: int = 0
    value_exact: int = 0
    value_fuzzy: int = 0
    value_compared: int = 0
    hallucinated_fields: tuple[str, ...] = ()
    summary_predicted: tuple[int, ...] = ()
    summary_expected: tuple[int, ...] = ()
    identifier_leaks: tuple[str, ...] = ()
    unclaimed_pii: tuple[str, ...] = ()
    prompt_chars: int = 0
    attempts: int = 1

    @property
    def responded(self) -> bool:
        """The model produced content, whether or not it validated.

        A 402, a rate limit or a dead socket says nothing about the model's
        ability to follow the schema, so those documents are excluded from the
        schema-validity denominator rather than counted as failures.
        """
        return self.ok or self.error_class == "ProviderResponseInvalid"

    @property
    def freshly_observed(self) -> bool:
        """A new answer from the endpoint, not a replay of a cached one.

        Used to report how much of the schema-validity denominator this
        particular run contributed. The rate itself is computed over the
        accumulated ledger (see :func:`merge_outcomes`), because neither half
        alone is unbiased: the cache stores only successes, so scoring the hits
        gives 1.0 and scoring the fresh answers gives 0.0.
        """
        return self.responded and not self.cached


@dataclass
class CandidateResult:
    """Everything measured for one candidate model."""

    model: str
    available: bool
    notes: str = ""
    catalogue: dict[str, Any] = field(default_factory=dict)
    q5: dict[str, Any] = field(default_factory=dict)
    docs: list[DocResult] = field(default_factory=list)
    fatal_error: str | None = None
    #: Tripped when the endpoint refused ``max_consecutive_api_errors`` requests
    #: in a row — a depleted credit balance or a dead route, not a model result.
    circuit_broken: bool = False


def _counter(name: str, labels: Mapping[str, str]) -> float:
    """One Prometheus counter's current value, or 0 when it has no samples yet."""
    value = REGISTRY.get_sample_value(name, dict(labels))
    return float(value) if value is not None else 0.0


def _repair_counters() -> dict[str, float]:
    return {
        result: _counter(
            "provider_api_repair_attempts_total",
            {"task": "extract_fields", "result": result},
        )
        for result in ("attempted", "recovered", "failed")
    }


def _request_counters() -> dict[str, float]:
    return {
        outcome: _counter(
            "provider_api_requests_total",
            {"task": "extract_fields", "outcome": outcome},
        )
        for outcome in ("ok", "cache_hit")
    }


#: Corpus ``fields`` keys that a deterministic validator owns and must therefore
#: be sentinel-masked out of the prompt (PLAN.md invariant 7), mapped to the
#: validator responsible.  A hit here is a **build bug**, not a model score.
INVARIANT7_GT_KEYS: Mapping[str, str] = {
    "account_number": "account_number",
    "ifsc_code": "ifsc_code",
    "transaction_reference": "utr_rrn",
    "transaction_date": "date",
    "disputed_amount": "amount",
}

#: Corpus ``fields`` keys that are personal data but that **no** validator
#: claims, so nothing masks them and they cross the network verbatim under the
#: laptop profile.  Counted separately and reported as a finding rather than a
#: failure: the corpus is synthetic, so this run leaks nothing real, but on the
#: air-gapped node it means a complainant's phone number reaches the SLM.
UNCLAIMED_PII_GT_KEYS: tuple[str, ...] = ("mobile",)

#: Shortest needle worth searching for. Below this a coincidental substring
#: match inside another number is likelier than a genuine leak.
_MIN_LEAK_NEEDLE = 6


def _find_values(
    doc: GoldDoc, prompt: str, keys: Sequence[str]
) -> tuple[str, ...]:
    """Which of ``keys``' ground-truth values appear verbatim in ``prompt``."""
    haystack = normalise_digits(prompt)
    found: list[str] = []
    for key in keys:
        raw = doc.gt_fields.get(key)
        if raw is None:
            continue
        needle = normalise_digits(str(raw))
        if len(needle) >= _MIN_LEAK_NEEDLE and needle in haystack:
            found.append(key)
    return tuple(found)


def _identifier_leaks(doc: GoldDoc, prompt: str) -> tuple[str, ...]:
    """Validator-owned identifiers that survived into the rendered prompt.

    Invariant 7 says deterministic fields never reach the SLM, and the sentinel
    masking in ``render_lines`` is what makes that true of the *bytes*.  This is
    the assertion that the masking actually fired on this document; anything it
    returns is a build bug in Stage 5 or the provider.
    """
    return _find_values(doc, prompt, tuple(INVARIANT7_GT_KEYS))


def _unclaimed_pii(doc: GoldDoc, prompt: str) -> tuple[str, ...]:
    """Personal data in the prompt that no validator claims and nothing masks."""
    return _find_values(doc, prompt, UNCLAIMED_PII_GT_KEYS)


def _score_document(
    doc: GoldDoc,
    raw_fields: Mapping[str, RawField],
    result: DocResult,
    *,
    settings: Settings,
) -> None:
    """Fill the grounding, attribution, value and summary numbers on ``result``."""
    gate = apply_grounding_gate(raw_fields, doc.page, settings=settings)

    value_names = [name for name in raw_fields if name != SUMMARY_FIELD_NAME]
    result.proposed_value_fields = len(value_names)
    result.grounded_value_fields = len(gate.fields)
    result.rejected_fields = tuple(sorted(item.name for item in gate.rejected))

    hallucinated: list[str] = []
    for name, accepted in sorted(gate.fields.items()):
        expected_lines = doc.tagged_lines.get(name, ())
        if not expected_lines:
            # The corpus says this document has no such field; a grounded value
            # is still a value the reviewer will have to delete.
            hallucinated.append(name)
            continue
        result.attributable_fields += 1
        if accepted.evidence.span.line_index in expected_lines:
            result.attributed_fields += 1

        gt_value = gt_value_for(doc, name)
        if gt_value is None:
            continue
        result.value_compared += 1
        got = normalise_value(accepted.value or "")
        want = normalise_value(gt_value)
        if got == want:
            result.value_exact += 1
            result.value_fuzzy += 1
        elif fuzz.token_set_ratio(got, want) >= FUZZY_MATCH_FLOOR:
            result.value_fuzzy += 1
        elif want and want in got:
            # The model returned the whole line and the GT value sits inside it
            # — right text, loose boundary. Counted as fuzzy, never as exact.
            result.value_fuzzy += 1
    result.hallucinated_fields = tuple(hallucinated)
    result.summary_predicted = tuple(gate.summary_line_indices)
    result.summary_expected = doc.summary_lines


def run_candidate(
    model: str,
    docs: Sequence[GoldDoc],
    *,
    settings: Settings,
    api_key: str,
    cache: ResponseCache | None,
    catalogue: Mapping[str, Any],
    json_mode: bool = True,
    timeout: float = 90.0,
    run_q5_probe: bool = True,
    provider_factory: Any = None,
    attempts: int = 1,
    attempt_delay: float = 0.0,
    max_consecutive_api_errors: int = 6,
    sleep: Any = time.sleep,
    q5_cache: dict[str, Any] | None = None,
    refresh_q5: bool = False,
) -> CandidateResult:
    """Measure one candidate over the whole document sample.

    A model the router does not serve is recorded as unavailable with the
    router's own message and contributes an entry to the report — "we could not
    test it" is an answer, and silently dropping it would leave the reader
    wondering whether it was tried.

    Two behaviours exist because a shared inference router is not a lab bench:

    * ``attempts`` / ``attempt_delay`` retry a document whose *endpoint* failed
      (402, 429, a dead socket).  A document whose *output* failed validation is
      never retried — that is the measurement;
    * ``max_consecutive_api_errors`` trips a circuit breaker.  Once the endpoint
      has refused that many requests in a row, the remaining documents are
      attempted once each with no backoff instead of being retried — the run
      stops waiting on a dead endpoint but still collects every cached replay,
      which costs the endpoint nothing.  Combined with the outcome ledger, that
      is what makes a resumed run accumulate rather than thrash.
    """
    entry = dict(catalogue.get(model, {}))
    result = CandidateResult(
        model=model,
        available=bool(entry) or provider_factory is not None,
        notes=CANDIDATE_NOTES.get(model, ""),
        catalogue=entry,
    )

    remembered = (q5_cache or {}).get(model)
    if remembered is not None and not refresh_q5 and probe_is_conclusive(remembered):
        result.q5 = dict(remembered)
    elif run_q5_probe and provider_factory is None:
        result.q5 = json_mode_probe(settings, model, api_key=api_key, timeout=timeout)
        result.q5["probed_at"] = datetime.now(UTC).isoformat()
        if q5_cache is not None and probe_is_conclusive(result.q5):
            q5_cache[model] = result.q5

    if result.q5 and not result.q5.get("json_object_accepted") and not entry:
        result.available = False
        result.fatal_error = _first_probe_error(result.q5)
        return result

    if not result.available:
        result.fatal_error = f"{model} is not in the router catalogue"
        return result

    if provider_factory is not None:
        provider = provider_factory(model)
    else:
        provider = HFAPIProvider(
            settings,
            api_key=api_key,
            cache=cache,
            enable_cache=cache is not None,
            slm_model=model,
            json_mode=json_mode,
            max_concurrency=1,
            timeout_seconds=timeout,
        )

    try:
        consecutive = 0
        for doc in docs:
            # Once the breaker has tripped, every remaining document is still
            # attempted — but exactly once, with no backoff. Skipping outright
            # was worse: a cached document costs the endpoint nothing, and
            # skipping it threw away a free replay of a measurement already
            # paid for. One fast refusal is cheaper than the data it preserves.
            tripped = consecutive >= max_consecutive_api_errors
            if tripped:
                result.circuit_broken = True
            outcome = _run_one_document(
                provider,
                doc,
                settings=settings,
                attempts=1 if tripped else attempts,
                attempt_delay=0.0 if tripped else attempt_delay,
                sleep=sleep,
            )
            result.docs.append(outcome)
            consecutive = 0 if outcome.responded else consecutive + 1
        if result.circuit_broken and result.fatal_error is None:
            result.fatal_error = (
                f"endpoint refused {max_consecutive_api_errors} consecutive "
                "requests; the remaining documents were attempted once each "
                "without retries"
            )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            # The response cache is shared across candidates and owned by the
            # caller. Detaching it before close() is what stops candidate one's
            # teardown from closing the SQLite handle candidate two needs.
            if hasattr(provider, "cache"):
                provider.cache = None
            close()
    return result


def _first_probe_error(q5: Mapping[str, Any]) -> str:
    for record in (q5.get("variants") or {}).values():
        if isinstance(record, dict) and record.get("error"):
            return str(record["error"])[:300]
        if isinstance(record, dict) and record.get("transport_error"):
            return str(record["transport_error"])[:300]
    return "no successful probe response"


def _run_one_document(
    provider: Any,
    doc: GoldDoc,
    *,
    settings: Settings,
    attempts: int = 1,
    attempt_delay: float = 0.0,
    sleep: Any = time.sleep,
) -> DocResult:
    """One document, retried only when the *endpoint* failed.

    A ``ProviderResponseInvalid`` is the model's answer and is never retried:
    retrying it would turn schema validity into a measure of persistence.
    """
    for attempt in range(1, max(1, attempts) + 1):
        result = _attempt_one_document(provider, doc, settings=settings)
        result.attempts = attempt
        if result.responded or attempt == max(1, attempts):
            return result
        if attempt_delay > 0:
            sleep(attempt_delay)
    raise AssertionError("unreachable: the loop always returns")  # pragma: no cover


def _attempt_one_document(
    provider: Any, doc: GoldDoc, *, settings: Settings
) -> DocResult:
    """One document: build the context, call the stage, score the answer."""
    result = DocResult(
        doc_id=doc.doc_id, doc_type=doc.doc_type, script=doc.script, ok=False
    )

    # Deterministic identifiers are claimed by the validators and masked before
    # the page is serialised. Running the real validators (rather than an empty
    # span list) is what makes this the production prompt.
    report = run_validators([doc.page], ifsc_master=None)
    masked = report.masked_spans(doc.page.page_no)

    reset = getattr(provider, "reset_budget", None)
    if callable(reset):
        reset(job_id=doc.doc_id)
    budget = getattr(provider, "budget", None)

    try:
        ctx: ExtractionContext = build_extraction_context(
            doc.page,
            document_id=doc.doc_id,
            page_class=doc.page_class,
            primary_script=doc.script,
            primary_language=doc.language,
            masked_spans=masked,
            budget=budget,
            settings=settings,
        )
    except TokenBudgetExceeded as exc:
        result.error = str(exc)
        result.error_class = type(exc).__name__
        return result

    prompt = build_extraction_prompt(ctx)
    result.prompt_chars = len(prompt)
    result.identifier_leaks = _identifier_leaks(doc, prompt)
    result.unclaimed_pii = _unclaimed_pii(doc, prompt)

    before_repairs = _repair_counters()
    before_requests = _request_counters()
    started = time.perf_counter()
    failure: Exception | None = None
    extracted: list[dict[str, RawField]] = []
    try:
        extracted = extract_pages(provider, [ctx])
    except (ProviderError, PipelineError) as exc:
        failure = exc

    # Read the counters on **both** paths. A page that failed validation twice
    # is precisely the page whose repair statistics matter most; taking them
    # only on success would report 100% first-attempt validity for a model that
    # never produced a valid answer.
    result.latency_seconds = time.perf_counter() - started
    after_repairs = _repair_counters()
    after_requests = _request_counters()
    result.first_attempt_valid = (
        after_repairs["attempted"] == before_repairs["attempted"]
    )
    result.repaired = after_repairs["recovered"] > before_repairs["recovered"]
    result.cached = after_requests["cache_hit"] > before_requests["cache_hit"]
    if budget is not None:
        result.tokens_used = int(budget.used)

    if failure is not None:
        result.error = f"{failure}"
        result.error_class = type(failure).__name__
        return result

    result.ok = True
    _score_document(doc, extracted[0], result, settings=settings)
    return result


class StubProvider:
    """Offline stand-in used by ``--dry-run`` and by the unit tests.

    Returns a deterministic, deliberately *imperfect* answer: one grounded
    field, one ungrounded field and a summary index that exists.  Imperfect on
    purpose — a stub that scores 1.0 everywhere would leave the aggregation code
    untested on the paths that matter.
    """

    name = "stub"

    def __init__(self, model: str = "stub/model", budget_limit: int = 8000) -> None:
        from app.providers.base import JobBudget

        self.slm_model = model
        self.budget = JobBudget(budget_limit)
        self.calls = 0

    def reset_budget(self, *, job_id: str | None = None) -> Any:
        from app.providers.base import JobBudget

        self.budget = JobBudget(self.budget.limit, job_id=job_id)
        return self.budget

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        self.calls += 1
        results: list[dict[str, RawField]] = []
        for ctx in ctxs:
            claim = self.budget.reserve(64)
            claim.settle(64)
            fields: dict[str, RawField] = {}
            if ctx.lines:
                first = ctx.lines[0]
                fields["complainant_name"] = RawField(
                    name="complainant_name",
                    value=first.text,
                    confidence=0.8,
                    line_indices=[first.line_index],
                )
                fields["relief_sought"] = RawField(
                    name="relief_sought",
                    value="a refund that appears nowhere on this page",
                    confidence=0.6,
                    line_indices=[first.line_index],
                )
                fields[SUMMARY_FIELD_NAME] = RawField(
                    name=SUMMARY_FIELD_NAME,
                    value=None,
                    confidence=0.7,
                    line_indices=[first.line_index],
                )
            results.append(fields)
        return results

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, higher neighbour — an actual observed latency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _rate(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def aggregate(
    result: CandidateResult, outcomes: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Turn one candidate's per-document results into its scorecard.

    ``outcomes`` is the merged outcome ledger for this candidate (see
    :func:`load_outcomes`). Schema validity is computed from it, over every
    document whose outcome has *ever* been observed, so a re-run against a warm
    response cache reports the accumulated truth rather than whichever half of
    the sample the endpoint happened to serve today.
    """
    docs = result.docs
    completed = [doc for doc in docs if doc.ok]
    responded = [doc for doc in docs if doc.responded]
    scoreable = [doc for doc in docs if doc.freshly_observed]
    live = [doc for doc in completed if not doc.cached]
    known: dict[str, Any] = (
        dict(outcomes)
        if outcomes
        else {
            doc.doc_id: {
                "valid": doc.ok,
                "first_attempt_valid": doc.first_attempt_valid,
                "repaired": doc.repaired,
            }
            for doc in responded
        }
    )
    # First-attempt validity has a smaller denominator than final validity: a
    # cache hit proves the document validates but says nothing about whether it
    # needed the repair turn, so it abstains rather than voting either way.
    attempt_known = [
        entry
        for entry in known.values()
        if entry.get("first_attempt_valid") is not None
    ]

    tp = fp = fn = 0
    for doc in completed:
        predicted = set(doc.summary_predicted)
        expected = set(doc.summary_expected)
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if (
        precision + recall
    ) else 0.0

    proposed = sum(doc.proposed_value_fields for doc in completed)
    grounded = sum(doc.grounded_value_fields for doc in completed)
    attributable = sum(doc.attributable_fields for doc in completed)
    attributed = sum(doc.attributed_fields for doc in completed)
    compared = sum(doc.value_compared for doc in completed)

    by_script: dict[str, dict[str, Any]] = {}
    for doc in completed:
        bucket = by_script.setdefault(
            doc.script,
            {"docs": 0, "compared": 0, "exact": 0, "fuzzy": 0, "grounded": 0, "proposed": 0},
        )
        bucket["docs"] += 1
        bucket["compared"] += doc.value_compared
        bucket["exact"] += doc.value_exact
        bucket["fuzzy"] += doc.value_fuzzy
        bucket["grounded"] += doc.grounded_value_fields
        bucket["proposed"] += doc.proposed_value_fields

    errors: dict[str, int] = {}
    messages: dict[str, int] = {}
    for doc in docs:
        if doc.error_class:
            errors[doc.error_class] = errors.get(doc.error_class, 0) + 1
        if doc.error:
            key = _error_signature(doc.error)
            messages[key] = messages.get(key, 0) + 1

    latencies = [doc.latency_seconds for doc in live]
    tokens = [doc.tokens_used for doc in completed]

    return {
        "model": result.model,
        "available": result.available,
        "notes": result.notes,
        "fatal_error": result.fatal_error,
        "circuit_broken": result.circuit_broken,
        "documents": {
            "attempted": len(docs),
            "completed": len(completed),
            "model_responded": len(responded),
            "api_errors": sum(1 for doc in docs if not doc.responded),
            "cached": sum(1 for doc in completed if doc.cached),
            "live": len(live),
        },
        "errors_by_class": dict(sorted(errors.items())),
        "errors_by_message": dict(
            sorted(messages.items(), key=lambda item: (-item[1], item[0]))
        ),
        "schema_validity_first_attempt": _rate(
            sum(1 for entry in attempt_known if entry["first_attempt_valid"]),
            len(attempt_known),
        ),
        "schema_validity_first_attempt_denominator": len(attempt_known),
        "schema_validity_final": _rate(
            sum(1 for entry in known.values() if entry.get("valid")), len(known)
        ),
        "schema_validity_denominator": len(known),
        "schema_validity_observed_this_run": len(scoreable),
        "schema_validity_note": (
            "computed over every document whose outcome has been observed by "
            "this run or a previous one (docs/bakeoff/"
            + OUTCOME_LEDGER_FILENAME
            + "). Scoring only this run's fresh answers would report 0.0 on a "
            "warm cache (the failures are exactly what the cache never stores) "
            "and scoring only its cache hits would report 1.0"
        ),
        "repair_turns_used": sum(
            1 for entry in known.values() if entry.get("repaired")
        ),
        "grounding_pass_rate": _rate(grounded, proposed),
        "value_fields_proposed": proposed,
        "value_fields_grounded": grounded,
        "line_attribution_accuracy": _rate(attributed, attributable),
        "line_attribution_denominator": attributable,
        "value_agreement_exact": _rate(
            sum(doc.value_exact for doc in completed), compared
        ),
        "value_agreement_fuzzy": _rate(
            sum(doc.value_fuzzy for doc in completed), compared
        ),
        "value_agreement_denominator": compared,
        "value_agreement_by_script": {
            script: {
                "documents": bucket["docs"],
                "exact": _rate(bucket["exact"], bucket["compared"]),
                "fuzzy": _rate(bucket["fuzzy"], bucket["compared"]),
                "grounding_pass_rate": _rate(bucket["grounded"], bucket["proposed"]),
                "compared": bucket["compared"],
            }
            for script, bucket in sorted(by_script.items())
        },
        "summary_precision": precision,
        "summary_recall": recall,
        "summary_f1": f1,
        "fields_invented_on_documents_without_them": sum(
            len(doc.hallucinated_fields) for doc in completed
        ),
        "identifier_leaks": sum(len(doc.identifier_leaks) for doc in docs),
        "unclaimed_pii_in_prompt": sum(len(doc.unclaimed_pii) for doc in docs),
        "unclaimed_pii_fields": sorted(
            {key for doc in docs for key in doc.unclaimed_pii}
        ),
        "latency_seconds": {
            "n_live": len(latencies),
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "tokens_per_document": {
            "mean": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "max": max(tokens) if tokens else 0,
            "budget": None,
        },
        "q5_json_mode": result.q5,
        "router": result.catalogue,
        "documents_detail": [
            {
                "doc_id": doc.doc_id,
                "script": doc.script,
                "ok": doc.ok,
                "cached": doc.cached,
                "latency_seconds": round(doc.latency_seconds, 3),
                "tokens": doc.tokens_used,
                "first_attempt_valid": doc.first_attempt_valid,
                "repaired": doc.repaired,
                "proposed": doc.proposed_value_fields,
                "grounded": doc.grounded_value_fields,
                "rejected": list(doc.rejected_fields),
                "attributed": f"{doc.attributed_fields}/{doc.attributable_fields}",
                "summary_predicted": list(doc.summary_predicted),
                "summary_expected": list(doc.summary_expected),
                "invented": list(doc.hallucinated_fields),
                "identifier_leaks": list(doc.identifier_leaks),
                "unclaimed_pii": list(doc.unclaimed_pii),
                "error": doc.error,
            }
            for doc in doc_sorted(result.docs)
        ],
    }


def doc_sorted(docs: Sequence[DocResult]) -> list[DocResult]:
    """Stable document order for the report."""
    return sorted(docs, key=lambda doc: doc.doc_id)


def _error_signature(message: str) -> str:
    """Collapse an error message to something countable.

    Endpoint errors carry request ids and models; without this every 402 would
    be its own bucket and the report would say "18 distinct errors" about one
    depleted credit balance.
    """
    text = " ".join(message.split())
    for marker in ("Error code: ", "produced output that failed validation"):
        position = text.find(marker)
        if position >= 0:
            text = text[position:]
            break
    return text[:160]


def composite_score(scorecard: Mapping[str, Any]) -> float | None:
    """Weighted score used for the ranking; see :data:`RANK_WEIGHTS`.

    ``None`` — not ``0.0`` — when this run made no fresh endpoint call for the
    candidate, so schema validity has no denominator. Zero would say "it scored
    badly"; the truth is "it was not measured this time", and the two must not
    be the same number in a table someone picks a model from. Re-run with
    ``--no-cache`` (or after the endpoint recovers) to get a score.
    """
    if not scorecard.get("available") or not scorecard["documents"]["completed"]:
        return 0.0
    if not scorecard.get("schema_validity_denominator"):
        return None
    return round(
        sum(
            weight * float(scorecard.get(metric, 0.0))
            for metric, weight in RANK_WEIGHTS.items()
        ),
        4,
    )


def rank_candidates(scorecards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank by composite score, then by p95 latency as the tie-break.

    Candidates with no score this run sort last, behind every measured one.
    """
    ranked = sorted(
        (
            {
                "model": card["model"],
                "composite_score": composite_score(card),
                "schema_validity_final": card.get("schema_validity_final", 0.0),
                "schema_validity_denominator": card.get(
                    "schema_validity_denominator", 0
                ),
                "grounding_pass_rate": card.get("grounding_pass_rate", 0.0),
                "line_attribution_accuracy": card.get(
                    "line_attribution_accuracy", 0.0
                ),
                "summary_f1": card.get("summary_f1", 0.0),
                "latency_p95": card.get("latency_seconds", {}).get("p95", 0.0),
                "available": card.get("available", False),
            }
            for card in scorecards
        ),
        key=lambda row: (
            row["composite_score"] is None,
            -(row["composite_score"] or 0.0),
            row["latency_p95"],
            row["model"],
        ),
    )
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return ranked


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(
    *,
    settings: Settings,
    docs: Sequence[GoldDoc],
    results: Sequence[CandidateResult],
    catalogue: Mapping[str, Any],
    dry_run: bool,
    started_at: datetime,
    outcomes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble ``docs/bakeoff/results.json``."""
    ledger = outcomes or {}
    scorecards = [
        aggregate(result, ledger.get(result.model)) for result in results
    ]
    for card in scorecards:
        card["tokens_per_document"]["budget"] = settings.JOB_TOKEN_BUDGET
    ranking = rank_candidates(scorecards)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": "tools/slm_bakeoff.py",
        "answers": ["BUILD-BRIEF M7 (SLM bake-off)", "OPEN-QUESTIONS Q5"],
        "mode": "dry_run" if dry_run else "measured",
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "settings": {
            "provider_profile": settings.PROVIDER_PROFILE.value,
            "synthetic_only": settings.SYNTHETIC_ONLY,
            "base_url": settings.HF_BASE_URL,
            "configured_slm_model": settings.SLM_MODEL,
            "configured_slm_model_fallback": settings.SLM_MODEL_FALLBACK,
            "slm_max_tokens": settings.SLM_MAX_TOKENS,
            "slm_temperature": settings.SLM_TEMPERATURE,
            "job_token_budget": settings.JOB_TOKEN_BUDGET,
            "grounding_max_edit_distance": settings.GROUNDING_MAX_EDIT_DISTANCE,
        },
        "sample": {
            "source": "corpus .gt.json OCR lines (not real OCR output)",
            "profile": SAMPLE_PROFILE,
            "variants": list(SAMPLE_VARIANTS),
            "locked_test_variants_excluded": [3, 4],
            "documents": len(docs),
            "by_type": _count(doc.doc_type for doc in docs),
            "by_script": _count(doc.script for doc in docs),
            "doc_ids": [doc.doc_id for doc in docs],
            "target_fields": list(DEFAULT_TARGET_FIELDS),
        },
        "router": {
            "url": catalogue.get("url"),
            "model_count": catalogue.get("model_count"),
            "configured_models_served": {
                settings.SLM_MODEL: settings.SLM_MODEL
                in (catalogue.get("models") or {}),
                settings.SLM_MODEL_FALLBACK: settings.SLM_MODEL_FALLBACK
                in (catalogue.get("models") or {}),
            },
        },
        "ranking_weights": dict(RANK_WEIGHTS),
        "ranking": ranking,
        "candidates": scorecards,
        "q5_answer": _q5_answer(scorecards),
        "coverage": _coverage(scorecards),
        "prompt_hygiene": {
            "invariant_7_identifier_leaks": sum(
                card["identifier_leaks"] for card in scorecards
            ),
            "masked_by_validators": sorted(INVARIANT7_GT_KEYS),
            "unclaimed_pii_fields": sorted(
                {
                    key
                    for card in scorecards
                    for key in card.get("unclaimed_pii_fields", [])
                }
            ),
            "note": (
                "invariant_7_identifier_leaks must be 0: those values are "
                "sentinel-masked before the page is serialised. The unclaimed "
                "list is personal data no validator owns, so nothing masks it "
                "and it crosses the network verbatim under the laptop profile."
            ),
        },
    }


def _coverage(scorecards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much of the intended matrix this run actually got to measure.

    Stated up front and separately from the scores, because a composite of 0.86
    over one document and a composite of 0.86 over eighteen are not the same
    claim, and a reader who has to derive that from a nested table will not.
    """
    messages: dict[str, int] = {}
    for card in scorecards:
        for message, count in card.get("errors_by_message", {}).items():
            messages[message] = messages.get(message, 0) + count
    attempted = sum(card["documents"]["attempted"] for card in scorecards)
    completed = sum(card["documents"]["completed"] for card in scorecards)
    responded = sum(card["documents"]["model_responded"] for card in scorecards)
    return {
        "candidates": len(scorecards),
        "candidates_with_any_result": sum(
            1 for card in scorecards if card["documents"]["completed"]
        ),
        "documents_attempted": attempted,
        "documents_completed": completed,
        "documents_model_responded": responded,
        "api_errors": sum(card["documents"]["api_errors"] for card in scorecards),
        # The rate that says how much of the matrix was actually *observed*. A
        # document the model answered badly is a measurement, not a gap — it is
        # the schema-validity metric — so completion rate alone would report a
        # model that reliably breaks the schema as "not measured".
        "observation_rate": _rate(responded, attempted),
        "completion_rate": _rate(completed, attempted),
        "endpoint_failure_messages": dict(
            sorted(messages.items(), key=lambda item: (-item[1], item[0]))
        ),
        "note": (
            "observation_rate is the share of the matrix the models actually "
            "answered; completion_rate is the share whose answer also validated "
            "— the difference between them is the schema-validity finding, not "
            "missing data. A candidate observed on a handful of documents is a "
            "data point, not a verdict. Re-running the tool resumes: the "
            "response cache keeps every answer that already validated."
        ),
    }


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _q5_answer(scorecards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The measured Q5 verdict, per candidate and overall."""
    per_model = {
        card["model"]: {
            "conclusive": probe_is_conclusive(card.get("q5_json_mode") or {}),
            "probed_at": (card.get("q5_json_mode") or {}).get("probed_at"),
            "json_object_accepted": card.get("q5_json_mode", {}).get(
                "json_object_accepted"
            ),
            "json_object_yields_json": card.get("q5_json_mode", {}).get(
                "json_object_yields_json"
            ),
            "json_schema_accepted": card.get("q5_json_mode", {}).get(
                "json_schema_accepted"
            ),
            "unconstrained_yields_json": card.get("q5_json_mode", {}).get(
                "unconstrained_yields_json"
            ),
            "router_advertises_structured_output": card.get("router", {}).get(
                "advertises_structured_output"
            ),
            "schema_validity_first_attempt": (
                card.get("schema_validity_first_attempt")
                if card.get("schema_validity_denominator")
                else None
            ),
        }
        for card in scorecards
    }
    # Only a *conclusive* probe counts. A model whose three probes were all
    # refused with 402 tells us about the account's credit balance, and folding
    # it in as "does not support JSON mode" would turn a billing state into a
    # capability claim.
    tested = [row for row in per_model.values() if row["conclusive"]]
    return {
        "question": "Does the HF router support grammars (GBNF) or JSON mode only?",
        "per_model": per_model,
        "models_probed": len(tested),
        "models_inconclusive": len(per_model) - len(tested),
        "json_object_accepted_count": sum(
            1 for row in tested if row["json_object_accepted"]
        ),
        "json_schema_accepted_count": sum(
            1 for row in tested if row["json_schema_accepted"]
        ),
        "verdict": _q5_verdict(tested),
    }


def _q5_verdict(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return (
            "not measured — no candidate returned a conclusive probe (every "
            "request was refused before it reached a model)"
        )
    schema_ok = sum(1 for row in rows if row["json_schema_accepted"])
    object_ok = sum(1 for row in rows if row["json_object_accepted"])
    if schema_ok == len(rows):
        return (
            "constrained decoding available: every probed model accepted "
            "response_format=json_schema, so the schema can be enforced at the "
            "endpoint and the repair retry becomes a safety net rather than a "
            "load-bearing mechanism"
        )
    if schema_ok:
        return (
            f"mixed: {schema_ok}/{len(rows)} probed models accepted "
            "response_format=json_schema and "
            f"{object_ok}/{len(rows)} accepted json_object. Keep JSON mode plus "
            "Pydantic plus one repair retry as the portable baseline and use "
            "json_schema opportunistically where the chosen provider serves it"
        )
    if object_ok:
        return (
            "JSON mode only: response_format=json_object is accepted but "
            "json_schema is not, so no grammar is available. The Q5 baseline "
            "(JSON mode + Pydantic + one repair retry) is the design, not a "
            "stopgap"
        )
    return (
        "neither json_object nor json_schema was accepted; structured output "
        "must be carried entirely by the prompt, Pydantic and the repair retry"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """``docs/bakeoff/REPORT.md`` — the written recommendation M7 asks for."""
    ranking = report["ranking"]
    cards = {card["model"]: card for card in report["candidates"]}
    winner = next(
        (row for row in ranking if (row["composite_score"] or 0.0) > 0), None
    )
    lines: list[str] = []
    add = lines.append

    add("# SLM bake-off — Phase 0 exit gate (BUILD-BRIEF M7)")
    add("")
    add(
        f"Generated {report['generated_at']} · mode **{report['mode']}** · "
        f"{report['sample']['documents']} documents · "
        f"{len(report['candidates'])} candidates"
    )
    add("")
    add(
        "Produced by `tools/slm_bakeoff.py`. Re-running it **resumes**: the "
        "response cache replays every validated answer for free, so a re-run "
        "pays only for what this run could not measure."
    )
    add("")

    coverage = report["coverage"]
    if coverage.get("observation_rate", 1.0) < 1.0:
        add("> ## Read this first: the run is incomplete")
        add(">")
        add(
            f"> The models answered {coverage['documents_model_responded']} of "
            f"{coverage['documents_attempted']} document-runs "
            f"({coverage['observation_rate']:.0%} of the matrix observed); "
            f"{coverage['documents_completed']} of those answers also "
            "validated. "
            f"{coverage['api_errors']} runs never reached a model."
        )
        add(">")
        add(
            "> The gap between *answered* and *validated* is the schema-validity "
            "finding, not missing data. The gap between *attempted* and "
            "*answered* is missing data:"
        )
        add(">")
        for message, count in list(coverage["endpoint_failure_messages"].items())[:3]:
            add(f"> - `x{count}` {message}")
        add(">")
        add(
            "> Every rate below states its own denominator and every ranking row "
            "carries `n`. Where `n` is a handful, treat the row as a data point "
            "and not a verdict. Re-running `tools/slm_bakeoff.py` resumes rather "
            "than restarts: the response cache holds every answer that already "
            "validated."
        )
        add("")

    add("## Recommendation")
    add("")
    if winner is None:
        add(
            "**No candidate completed the sample.** See *Availability* below "
            "for what the router did with each one."
        )
    else:
        card = cards[winner["model"]]
        counts = card["documents"]
        add(
            f"**{winner['model']}** — composite {winner['composite_score']}; "
            f"{counts['model_responded']}/{counts['attempted']} documents "
            f"answered, {counts['completed']} of those schema-valid."
        )
        if counts["model_responded"] < counts["attempted"]:
            add("")
            add(
                f"*Provisional.* {counts['attempted'] - counts['model_responded']} "
                "of this candidate's documents did not reach the model in this "
                "run. Schema validity below is scored over everything observed "
                "across runs; the content metrics are scored over the "
                f"{counts['completed']} answers that validated."
            )
        add("")
        add(
            f"- schema validity **{card['schema_validity_final']:.0%}** over "
            f"n={card['schema_validity_denominator']} observed documents "
            f"({card['schema_validity_first_attempt']:.0%} on the first attempt "
            f"over n={card['schema_validity_first_attempt_denominator']}, "
            f"{card['repair_turns_used']} recovered by the repair turn)"
        )
        add(
            f"- grounding gate {card['grounding_pass_rate']:.0%} of "
            f"{card['value_fields_proposed']} proposed values"
        )
        add(
            f"- line attribution {card['line_attribution_accuracy']:.0%} of "
            f"{card['line_attribution_denominator']} attributable fields"
        )
        add(
            f"- extractive summary F1 {card['summary_f1']:.2f} "
            f"(P {card['summary_precision']:.2f} / R {card['summary_recall']:.2f})"
        )
        if card["latency_seconds"]["n_live"]:
            live = card["latency_seconds"]["n_live"]
            add(
                f"- latency p50 {card['latency_seconds']['p50']}s / p95 "
                f"{card['latency_seconds']['p95']}s over {live} uncached "
                f"call{'' if live == 1 else 's'}"
                + (" — too few to be a distribution" if live < 5 else "")
            )
            add(
                f"- {card['tokens_per_document']['mean']} tokens per document "
                f"against a {card['tokens_per_document']['budget']} budget"
            )
        else:
            add(
                "- latency and token cost **not measured**: every completed "
                "document was served from the response cache, which costs "
                "neither"
            )
    add("")

    configured = report["router"]["configured_models_served"]
    if not all(configured.values()):
        add("### The configured models are not served")
        add("")
        for model, served in configured.items():
            add(f"- `{model}` — {'served' if served else '**not on the router**'}")
        add("")
        add(
            "`SLM_MODEL` and `SLM_MODEL_FALLBACK` in `app/settings.py` and "
            "`.env.example` therefore name endpoints that return "
            "`model_not_found`. Changing them is an owner decision; the "
            "recommendation above is the replacement this run measured."
        )
        add("")

    add("## Ranking")
    add("")
    add(
        "| # | model | answered | validated | composite | schema | grounding | "
        "attribution | summary F1 | p95 s |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in ranking:
        counts = cards[row["model"]]["documents"]
        composite = (
            "not measured" if row["composite_score"] is None else row["composite_score"]
        )
        schema = (
            f"{row['schema_validity_final']} (n={row['schema_validity_denominator']})"
            if row["schema_validity_denominator"]
            else "—"
        )
        add(
            f"| {row['rank']} | `{row['model']}` | "
            f"{counts['model_responded']}/{counts['attempted']} | "
            f"{counts['completed']} | "
            f"{composite} | {schema} | {row['grounding_pass_rate']} | "
            f"{row['line_attribution_accuracy']} | {row['summary_f1']} | "
            f"{row['latency_p95']} |"
        )
    add("")
    add(
        "*answered* is how many documents reached the model at all in this run. "
        "*validated* is how many of those answers passed the contract. Schema "
        "validity carries its own `n`: it is scored over every document whose "
        f"outcome has ever been observed (`{OUTCOME_LEDGER_FILENAME}`), because "
        "the response cache stores only answers that already validated — "
        "scoring one run's cache hits alone gives 1.0 and its fresh answers "
        "alone gives 0.0. A rank built on a small `n` is not a result, and "
        "`not measured` is not a score of zero."
    )
    add("")
    add(
        "Composite weights: "
        + ", ".join(f"{name} {weight}" for name, weight in RANK_WEIGHTS.items())
        + ". Value agreement is deliberately excluded — see *Caveats*."
    )
    add("")

    add("## Q5 — does the router honour JSON mode?")
    add("")
    answer = report["q5_answer"]
    add(f"**{answer['verdict']}**")
    add("")
    add(
        f"{answer['models_probed']} of "
        f"{answer['models_probed'] + answer['models_inconclusive']} candidates "
        "returned a conclusive probe. An inconclusive row is one whose requests "
        "never reached a model, and it is excluded from the verdict rather than "
        "counted as a refusal."
    )
    add("")
    add(
        "| model | conclusive | `json_object` accepted | yields JSON | "
        "`json_schema` accepted | JSON without a format | router advertises "
        "structured output | 1st-attempt schema validity | probed |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for model, row in answer["per_model"].items():
        probed = (row["probed_at"] or "—")[:10]
        # An inconclusive probe's derived booleans are all False because nothing
        # answered, not because anything was refused. Printing them as "no"
        # would state a capability the run never observed.
        def cell(key: str, _row: Mapping[str, Any] = row) -> str:
            return _yn(_row[key]) if _row["conclusive"] else "—"

        validity = row["schema_validity_first_attempt"]
        add(
            f"| `{model}` | {_yn(row['conclusive'])} | "
            f"{cell('json_object_accepted')} | "
            f"{cell('json_object_yields_json')} | "
            f"{cell('json_schema_accepted')} | "
            f"{cell('unconstrained_yields_json')} | "
            f"{_yn(row['router_advertises_structured_output'])} | "
            f"{'—' if validity is None else validity} | {probed} |"
        )
    add("")
    add(
        f"Probes are remembered in `{Q5_CACHE_FILENAME}` and reused, because a "
        "probe is a dated observation of the router that cannot be re-taken "
        "once the account's inference credits are gone. `--refresh-q5` re-takes "
        "them."
    )
    add("")

    add("## Availability and what the endpoint did")
    add("")
    for card in report["candidates"]:
        state = "available" if card["available"] else "**unavailable**"
        counts = card["documents"]
        add(
            f"- `{card['model']}` — {state}; {counts['completed']} completed, "
            f"{counts['api_errors']} endpoint failures of "
            f"{counts['attempted']}. {card['notes']}"
        )
        if card["fatal_error"]:
            add(f"  - `{card['fatal_error']}`")
        for message, count in list(card.get("errors_by_message", {}).items())[:2]:
            add(f"  - `x{count}` {message}")
    add("")

    add("## Per-candidate detail")
    add("")
    for card in report["candidates"]:
        add(f"### `{card['model']}`")
        add("")
        if not card["available"] or not card["documents"]["completed"]:
            add(f"Not measured: {card['fatal_error'] or 'no document completed'}.")
            add("")
            continue
        add(
            f"{card['documents']['completed']}/{card['documents']['attempted']} "
            f"documents completed ({card['documents']['cached']} served from the "
            "response cache)."
        )
        add("")
        add("| metric | value |")
        add("|---|---|")
        for label, key in (
            ("schema validity (final, fresh answers only)", "schema_validity_final"),
            ("schema validity denominator", "schema_validity_denominator"),
            ("schema validity (first attempt)", "schema_validity_first_attempt"),
            ("grounding pass rate", "grounding_pass_rate"),
            ("line attribution accuracy", "line_attribution_accuracy"),
            ("value agreement — exact", "value_agreement_exact"),
            ("value agreement — fuzzy", "value_agreement_fuzzy"),
            ("summary precision", "summary_precision"),
            ("summary recall", "summary_recall"),
            ("summary F1", "summary_f1"),
        ):
            add(f"| {label} | {card[key]} |")
        add(
            f"| fields invented where the corpus has none | "
            f"{card['fields_invented_on_documents_without_them']} |"
        )
        add(
            f"| validator-owned identifiers leaked into the prompt | "
            f"{card['identifier_leaks']} |"
        )
        add(
            f"| latency p50 / p95 | {card['latency_seconds']['p50']}s / "
            f"{card['latency_seconds']['p95']}s |"
        )
        add(
            f"| tokens per document (mean / max) | "
            f"{card['tokens_per_document']['mean']} / "
            f"{card['tokens_per_document']['max']} |"
        )
        add("")
        if card["value_agreement_by_script"]:
            add("Per script:")
            add("")
            add("| script | docs | exact | fuzzy | grounding |")
            add("|---|---|---|---|---|")
            for script, row in card["value_agreement_by_script"].items():
                add(
                    f"| {script} | {row['documents']} | {row['exact']} | "
                    f"{row['fuzzy']} | {row['grounding_pass_rate']} |"
                )
            add("")

    hygiene = report["prompt_hygiene"]
    add("## Prompt hygiene")
    add("")
    add(
        f"- Validator-owned identifiers that reached a prompt: "
        f"**{hygiene['invariant_7_identifier_leaks']}** "
        f"(must be 0 — {', '.join(hygiene['masked_by_validators'])} are "
        "sentinel-masked before serialisation, PLAN.md invariant 7)."
    )
    if hygiene["unclaimed_pii_fields"]:
        add(
            "- **Finding:** no validator claims "
            + ", ".join(f"`{key}`" for key in hygiene["unclaimed_pii_fields"])
            + ", so nothing masks it and it reaches the SLM verbatim. Harmless "
            "here (synthetic corpus) but it is a real complainant's phone "
            "number on the air-gapped node. Recorded for OPEN-QUESTIONS."
        )
    else:
        add("- No unmasked personal data was detected in the prompts.")
    add("")

    add("## Method")
    add("")
    add(
        f"- Sample: {report['sample']['documents']} corpus documents, variants "
        f"{report['sample']['variants']} of "
        f"{', '.join(report['sample']['by_type'])} at profile "
        f"`{report['sample']['profile']}`. Variants 3 and 4 are the locked test "
        "split and were not used."
    )
    add(
        "- Input text is the `.gt.json` OCR lines, not real OCR output, so the "
        "numbers measure the SLM and not the recogniser. `tools/eval_harness.py` "
        "owns the end-to-end figure."
    )
    add(
        "- The prompt is the production one (`build_extraction_prompt`), with "
        "deterministic identifiers sentinel-masked from the real validator run. "
        "`identifier leaks` above is the assertion that the masking fired."
    )
    add(
        "- Grounding is the production gate (`apply_grounding_gate`) at edit "
        f"distance {report['settings']['grounding_max_edit_distance']}."
    )
    add(
        f"- Each document gets a fresh {report['settings']['job_token_budget']}-token "
        "job budget, enforced before the request leaves."
    )
    add("- Temperature 0. Response cache on, so a rerun is free and reproducible.")
    add("")

    add("## Caveats")
    add("")
    add(
        "- **Value agreement is not in the score.** The corpus ground truth "
        "carries English field values on Devanagari pages (OPEN-QUESTIONS Q6), "
        "so exact agreement penalises a model that correctly copies the "
        "Devanagari text. Line attribution is the script-neutral measure and is "
        "what the ranking uses."
    )
    add(
        "- Router availability and per-provider routing move. The catalogue as "
        "seen by this run is recorded in `results.json`."
    )
    add(
        "- Synthetic corpus text only. Nothing here says how these models "
        "behave on real complaint prose."
    )
    add("")
    return "\n".join(lines) + "\n"


def _yn(value: Any) -> str:
    if value is None:
        return "—"
    return "yes" if value else "no"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="slm_bakeoff",
        description="M7 SLM bake-off over the HF router (BUILD-BRIEF M7, Q5).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="candidate model id; repeatable. Replaces the default candidate set.",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=None,
        help="cap the document sample (default: every sampled document)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the response cache and force live calls",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="send requests without response_format (the Q5 control)",
    )
    parser.add_argument(
        "--skip-q5-probe",
        action="store_true",
        help="skip the raw JSON-mode probes (a remembered probe is still used)",
    )
    parser.add_argument(
        "--refresh-q5",
        action="store_true",
        help=(
            "re-probe JSON-mode support even for models whose probe was already "
            f"answered and stored in {Q5_CACHE_FILENAME}"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=2,
        help=(
            "attempts per document when the endpoint (not the model) fails; "
            "a schema failure is never retried (default 2)"
        ),
    )
    parser.add_argument(
        "--attempt-delay",
        type=float,
        default=8.0,
        help="seconds between endpoint retries (default 8)",
    )
    parser.add_argument(
        "--max-consecutive-api-errors",
        type=int,
        default=6,
        help=(
            "stop a candidate after this many consecutive endpoint failures and "
            "skip its remaining documents (default 6). Re-run to resume: the "
            "response cache keeps every answer that already validated."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="offline stub provider; no network, no token, no weights",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="where results.json and REPORT.md are written",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the summary without writing"
    )
    return parser


def default_candidates(settings: Settings) -> list[str]:
    """Configured models first, then the extra candidates, de-duplicated."""
    ordered: list[str] = [settings.SLM_MODEL, settings.SLM_MODEL_FALLBACK]
    ordered.extend(EXTRA_CANDIDATES)
    seen: set[str] = set()
    return [model for model in ordered if not (model in seen or seen.add(model))]


DEFAULT_CANDIDATES = EXTRA_CANDIDATES


def _run_and_record(
    ledger: dict[str, dict[str, Any]],
    model: str,
    docs: Sequence[GoldDoc],
    **kwargs: Any,
) -> CandidateResult:
    """Run one candidate and fold its fresh observations into ``ledger``."""
    result = run_candidate(model, docs, **kwargs)
    merge_outcomes(ledger, result)
    return result


def run(args: argparse.Namespace, *, settings: Settings | None = None) -> dict[str, Any]:
    """Do the whole bake-off and return the report."""
    active = settings if settings is not None else get_settings()
    started_at = datetime.now(UTC)
    docs = load_gold_docs(active.corpus_dir, settings=active, limit=args.docs)
    if not docs:
        raise ValueError(
            f"no corpus documents found under {active.corpus_dir}; generate the "
            "corpus with tools/generate_test_corpus.py first"
        )
    models = args.model or default_candidates(active)

    if args.dry_run:
        catalogue: dict[str, Any] = {"url": None, "model_count": 0, "models": {}}
        results = [
            run_candidate(
                model,
                docs,
                settings=active,
                api_key="dry-run",
                cache=None,
                catalogue={},
                run_q5_probe=False,
                provider_factory=lambda name: StubProvider(
                    name, budget_limit=active.JOB_TOKEN_BUDGET
                ),
            )
            for model in models
        ]
        return build_report(
            settings=active,
            docs=docs,
            results=results,
            catalogue=catalogue,
            dry_run=True,
            started_at=started_at,
        )

    if not active.hf_token_present:
        raise ValueError(
            "HF_TOKEN is not set; the bake-off needs it (laptop profile only). "
            "Use --dry-run to exercise the tool offline."
        )
    api_key = active.HF_TOKEN.get_secret_value()  # type: ignore[union-attr]
    catalogue = fetch_router_catalogue(active, api_key=api_key)
    cache = None if args.no_cache else ResponseCache(active.provider_cache_path)
    q5_cache = load_q5_cache(args.out_dir)
    outcomes = load_outcomes(args.out_dir)

    results = []
    try:
        for model in models:
            print(f"-- {model}", flush=True)
            results.append(
                _run_and_record(
                    outcomes,
                    model,
                    docs,
                    settings=active,
                    api_key=api_key,
                    cache=cache,
                    catalogue=catalogue["models"],
                    json_mode=not args.no_json_mode,
                    timeout=args.timeout,
                    run_q5_probe=not args.skip_q5_probe,
                    attempts=args.attempts,
                    attempt_delay=args.attempt_delay,
                    max_consecutive_api_errors=args.max_consecutive_api_errors,
                    q5_cache=q5_cache,
                    refresh_q5=args.refresh_q5,
                )
            )
    finally:
        if cache is not None:
            cache.close()
        if not args.no_write:
            if q5_cache:
                save_q5_cache(args.out_dir, q5_cache)
            if outcomes:
                save_outcomes(args.out_dir, outcomes)

    return build_report(
        outcomes=outcomes,
        settings=active,
        docs=docs,
        results=results,
        catalogue=catalogue,
        dry_run=False,
        started_at=started_at,
    )


def print_summary(report: Mapping[str, Any]) -> None:
    """Human-readable digest."""
    print(
        f"sample    {report['sample']['documents']} documents "
        f"{report['sample']['by_script']}"
    )
    cards = {card["model"]: card for card in report["candidates"]}
    for row in report["ranking"]:
        counts = cards[row["model"]]["documents"]
        composite = "n/a" if row["composite_score"] is None else row["composite_score"]
        print(
            f"  #{row['rank']} {row['model']:<40} composite={composite} "
            f"schema={row['schema_validity_final']}"
            f"(n={row['schema_validity_denominator']}) "
            f"ground={row['grounding_pass_rate']} "
            f"attrib={row['line_attribution_accuracy']} "
            f"sumF1={row['summary_f1']} p95={row['latency_p95']}s "
            f"[ans={counts['model_responded']}/{counts['attempted']} "
            f"valid={counts['completed']} cached={counts['cached']} "
            f"api_err={counts['api_errors']}]"
        )
    coverage = report["coverage"]
    print(
        f"coverage  {coverage['documents_model_responded']}/"
        f"{coverage['documents_attempted']} answered "
        f"({coverage['documents_completed']} schema-valid); "
        f"{coverage['api_errors']} never reached a model"
    )
    if coverage["endpoint_failure_messages"]:
        for message, count in list(coverage["endpoint_failure_messages"].items())[:3]:
            print(f"          x{count}  {message[:110]}")
    print(f"Q5        {report['q5_answer']['verdict']}")
    leaks = sum(card["identifier_leaks"] for card in report["candidates"])
    print(f"leaks     {leaks} validator-owned identifier(s) reached a prompt (must be 0)")
    pii = report["prompt_hygiene"]["unclaimed_pii_fields"]
    if pii:
        print(f"unmasked  no validator claims {', '.join(pii)} — reaches the SLM verbatim")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns 0 unless nothing could be measured."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.docs is not None and args.docs < 1:
        parser.error("--docs must be >= 1")

    # The grounding gate warns once per rejected field. Over six candidates and
    # eighteen documents that is several hundred lines of the tool's own
    # expected behaviour, which would bury the run's actual output. The
    # rejections are all counted and reported; only the log line is suppressed.
    logging.getLogger("app.stages.s5_slm").setLevel(logging.ERROR)

    report = run(args)
    print_summary(report)

    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.out_dir / "REPORT.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
        print(f"wrote     {args.out_dir / 'results.json'}")
        print(f"wrote     {args.out_dir / 'REPORT.md'}")

    if not any((row["composite_score"] or 0.0) > 0 for row in report["ranking"]):
        print(
            "NO CANDIDATE COMPLETED THE SAMPLE — see docs/bakeoff/REPORT.md for "
            "what the router did with each one.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
