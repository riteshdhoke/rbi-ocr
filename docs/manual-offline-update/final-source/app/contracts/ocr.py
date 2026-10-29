"""OCR-layer contracts — BUILD-BRIEF sec. 3.2, and the models the
``ModelProvider`` protocol exchanges.

The load-bearing rule of this module is **PLAN.md invariant 1 / design-review
finding A2**: ``script``, ``language`` and ``is_handwritten`` are attributes of
an :class:`OCRLine`.  They are **never** attributes of an :class:`OCRPage`.
The defining RBI complaint is a printed English form with a handwritten Marathi
free-text block; one page-level language label silently corrupts every stage
downstream of it.  :class:`OCRPage` actively rejects those keys — see
``_reject_page_level_language``.

Page-level ``primary_language`` / ``language_distribution`` / ``is_code_mixed``
exist **only** as derived summaries on the envelope, computed in
``assemble.py``.  They are deliberately absent here, including as convenience
properties: a property is how the bug comes back.

This module does NOT handle:
  * image data — no Pillow/NumPy import; crops and pages are passed to the
    provider as objects the provider owns, never through these models;
  * deriving page or document language (``assemble.py``);
  * validating that ``bbox`` is genuinely clockwise from top-left — a contract
    cannot know the page orientation, so it enforces arity (exactly 8) and
    finiteness only.  Corner *order* is the detector's responsibility, and
    :func:`bbox_from_xyxy` exists so callers do not hand-roll it;
  * geometry (IoU, deskew, rotation) — that is the eval harness and WS-E;
  * confidence calibration.  Nothing may escalate on ``OCRLine.confidence``
    alone (PLAN.md invariant 3); the router keys on validator failure and
    structural signals;
  * persistence or wire framing — these are the in-process contract; the
    externally published shape is :mod:`app.contracts.envelope`.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DETERMINISTIC_FIELDS",
    "SLM_ELIGIBLE_FIELDS",
    "SUMMARY_FIELD_NAME",
    "BBox",
    "Confidence",
    "ExtractionContext",
    "LangCandidate",
    "LangResult",
    "LanguageCode",
    "LineIndex",
    "OCRLine",
    "OCRPage",
    "OCRTier",
    "PageClass",
    "PageClassLabel",
    "PageNumber",
    "RawField",
    "ScriptCode",
    "ScriptResult",
    "ScriptVote",
    "StructuralFeatures",
    "TextSpan",
    "bbox_corners",
    "bbox_from_xyxy",
    "check_distribution",
]

# --------------------------------------------------------------------------- #
# Reusable constrained scalar types
# --------------------------------------------------------------------------- #

BBOX_LEN: int = 8


def _check_bbox(value: list[float]) -> list[float]:
    """Reject bboxes that are not exactly 8 finite coordinates."""
    if len(value) != BBOX_LEN:
        raise ValueError(
            f"bbox must be exactly {BBOX_LEN} floats "
            "(4 corners, clockwise from top-left), got " + str(len(value))
        )
    for index, coordinate in enumerate(value):
        if not math.isfinite(coordinate):
            raise ValueError(
                f"bbox coordinate {index} is not finite: {coordinate!r}"
            )
    return value


#: 4 corners as ``[x0, y0, x1, y1, x2, y2, x3, y3]``, clockwise from top-left.
#: Kept a ``list[float]`` to match BUILD-BRIEF sec. 3.2 verbatim.
BBox = Annotated[
    list[float],
    Field(
        min_length=BBOX_LEN,
        max_length=BBOX_LEN,
        description="8 floats: 4 corners (x, y) clockwise from top-left.",
    ),
    AfterValidator(_check_bbox),
]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

#: ISO 15924, e.g. ``Latn``, ``Deva``, ``Zyyy`` (undetermined).
ScriptCode = Annotated[
    str,
    Field(pattern=r"^[A-Z][a-z]{3}$", description="ISO 15924 script code."),
]

#: ISO 639-3, e.g. ``eng``, ``hin``, ``mar``, ``und`` (undetermined).
LanguageCode = Annotated[
    str,
    Field(pattern=r"^[a-z]{3}$", description="ISO 639-3 language code."),
]

PageNumber = Annotated[int, Field(ge=1, description="1-based page number.")]
LineIndex = Annotated[int, Field(ge=0, description="0-based index within a page.")]


def bbox_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    """Build an axis-aligned :data:`BBox` from a top-left/bottom-right pair.

    Emits the canonical clockwise-from-top-left corner order so call sites do
    not each invent their own.
    """
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def bbox_corners(bbox: list[float]) -> list[tuple[float, float]]:
    """Return the 4 ``(x, y)`` corners of a :data:`BBox`, in stored order."""
    if len(bbox) != BBOX_LEN:
        raise ValueError(f"bbox must be {BBOX_LEN} floats, got {len(bbox)}")
    return [(bbox[i], bbox[i + 1]) for i in range(0, BBOX_LEN, 2)]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class OCRTier(StrEnum):
    """Which mechanism produced a page's lines (PRD P0-4, P0-6)."""

    #: Digital PDF text layer, no recognition run (and no OCR billed).
    BYPASS = "bypass"
    #: PP-OCRv5 detection + batched recognition.
    TIER1 = "tier1"
    #: Escalated heavy model (VLM).
    TIER2 = "tier2"
    #: The page produced no lines (bounced / blank); recorded, never guessed.
    NONE = "none"


class PageClassLabel(StrEnum):
    """Closed page/document class set, matching TEST-DATA-SPEC sec. 2 classes."""

    COMPLAINT_FORM = "complaint_form"
    COMPLAINT_LETTER = "complaint_letter"
    SUPPORTING_DOC = "supporting_doc"
    PHOTO_EVIDENCE = "photo_evidence"
    UNKNOWN = "unknown"


#: Field name of the extractive summary.  Invariant 2: the model selects line
#: indices, the assembler concatenates verbatim OCR text.  Never prose.
SUMMARY_FIELD_NAME: str = "complaint_summary"

#: Specialist validator names; exclusive ownership applies only in legacy mode.
DETERMINISTIC_FIELDS: frozenset[str] = frozenset(
    {
        "account_number",
        "ifsc_code",
        "bank_name",
        "card_fragment",
        "utr_rrn",
        "micr_code",
        "amount",
        "transaction_date",
        "complaint_date",
    }
)

#: Legacy SLM targets; open discovery can return any field name.
SLM_ELIGIBLE_FIELDS: frozenset[str] = frozenset(
    {
        "complainant_name",
        "complainant_address",
        "relief_sought",
        "prior_complaint_reference",
        SUMMARY_FIELD_NAME,
    }
)


# --------------------------------------------------------------------------- #
# The line contract
# --------------------------------------------------------------------------- #


class OCRLine(BaseModel):
    """One recognised text line — BUILD-BRIEF sec. 3.2, verbatim field set.

    ``script``, ``language`` and ``is_handwritten`` live here and nowhere else.
    ``language`` is ``None`` until Stage 2b (line-level LID) has run; it is
    never defaulted to the page's or the document's language.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: BBox
    confidence: Confidence
    script: ScriptCode
    #: ISO 639-3, set per line by ``s2b_lid``. ``None`` = not yet identified.
    language: LanguageCode | None = None
    is_handwritten: bool
    line_index: LineIndex


# --------------------------------------------------------------------------- #
# The page contract — deliberately language-free
# --------------------------------------------------------------------------- #

#: Keys that must never appear on a page.  ``confidence`` is included because
#: PLAN.md invariant 3 forbids escalating on aggregate CTC confidence; the
#: cheapest way to stop that is to never materialise the number on the page.
_PAGE_LEVEL_FORBIDDEN: frozenset[str] = frozenset(
    {
        "language",
        "languages",
        "primary_language",
        "language_distribution",
        "script",
        "primary_script",
        "is_handwritten",
        "is_code_mixed",
        "confidence",
    }
)


class OCRPage(BaseModel):
    """The recognised lines of one page, plus the geometry they refer to.

    There is intentionally no ``language``, ``script`` or ``is_handwritten``
    here.  If you need a page-level language you want the derived summary on
    :class:`app.contracts.envelope.PageResult`, computed in ``assemble.py``.
    """

    model_config = ConfigDict(extra="forbid")

    page_no: PageNumber
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    lines: list[OCRLine]
    tier: OCRTier
    #: Deskew/rotation actually applied by the quality gate, in degrees.
    rotation_applied_deg: float = 0.0
    #: Effective DPI estimate, when the source carried or implied one.
    dpi_estimate: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _reject_page_level_language(cls, data: Any) -> Any:
        """Fail loudly on the exact bug design-review finding A2 reversed."""
        if isinstance(data, dict):
            offending = sorted(_PAGE_LEVEL_FORBIDDEN.intersection(data))
            if offending:
                raise ValueError(
                    "OCRPage must not carry page-level "
                    f"{', '.join(offending)}: script/language/is_handwritten "
                    "belong to OCRLine, and page-level summaries are derived in "
                    "assemble.py onto the envelope (PLAN.md invariant 1). "
                    "Aggregate confidence is withheld so nothing can escalate "
                    "on it (invariant 3)."
                )
        return data

    @model_validator(mode="after")
    def _unique_line_indices(self) -> Self:
        """``line_index`` is the handle the extractive summary selects by."""
        seen: set[int] = set()
        for line in self.lines:
            if line.line_index in seen:
                raise ValueError(
                    f"duplicate line_index {line.line_index} on page "
                    f"{self.page_no}: indices are the handles the SLM selects "
                    "spans by (PLAN.md invariant 2) and must be unique"
                )
            seen.add(line.line_index)
        return self

    def line_by_index(self, line_index: int) -> OCRLine:
        """Look a line up by ``line_index`` (not by list position).

        Indices survive line filtering; positions do not.  ``assemble.py`` uses
        this to rebuild ``complaint_summary`` verbatim.

        Raises:
            KeyError: when no line carries that index.
        """
        for line in self.lines:
            if line.line_index == line_index:
                return line
        raise KeyError(
            f"no line with line_index {line_index} on page {self.page_no}"
        )


# --------------------------------------------------------------------------- #
# ModelProvider exchange models (BUILD-BRIEF sec. 3.1)
# --------------------------------------------------------------------------- #


class ScriptVote(BaseModel):
    """One sampled crop's script vote, kept so the majority is auditable."""

    model_config = ConfigDict(extra="forbid")

    script: ScriptCode
    confidence: Confidence


class ScriptResult(BaseModel):
    """Output of ``detect_script(crops)`` — majority vote over sampled crops."""

    model_config = ConfigDict(extra="forbid")

    script: ScriptCode
    confidence: Confidence
    #: Share of votes per script; keys are ISO 15924, values sum to ~1.
    distribution: dict[str, float] = Field(default_factory=dict)
    votes: list[ScriptVote] = Field(default_factory=list)
    sampled_crops: int = Field(ge=0)

    @model_validator(mode="after")
    def _distribution_is_a_distribution(self) -> Self:
        check_distribution(self.distribution, "ScriptResult.distribution")
        if self.distribution and self.script not in self.distribution:
            raise ValueError(
                f"winning script {self.script!r} absent from distribution "
                f"{sorted(self.distribution)}"
            )
        return self


class LangCandidate(BaseModel):
    """A candidate language with its score, from the script-constrained set."""

    model_config = ConfigDict(extra="forbid")

    language: LanguageCode
    confidence: Confidence


class LangResult(BaseModel):
    """Output of ``identify_language(text, script)`` — **per line**, not per page.

    ``script`` records which label set the result was constrained to
    (PRD P0-5: Devanagari implies hin/mar/kok/nep, never eng).
    """

    model_config = ConfigDict(extra="forbid")

    language: LanguageCode | None
    confidence: Confidence
    script: ScriptCode
    candidates: list[LangCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _winner_is_a_candidate(self) -> Self:
        if (
            self.language is not None
            and self.candidates
            and all(c.language != self.language for c in self.candidates)
        ):
            raise ValueError(
                f"chosen language {self.language!r} is not among the "
                "candidates it was chosen from"
            )
        return self


class StructuralFeatures(BaseModel):
    """Layout signals fed to the page classifier and the escalation router.

    BUILD-BRIEF M4: line count, ink density, aspect, table-line presence,
    colour histogram.  These are the *evidence-external* signals PLAN.md
    invariant 3 requires the router to key on instead of CTC confidence.
    """

    model_config = ConfigDict(extra="forbid")

    line_count: int = Field(ge=0)
    ink_density: float = Field(ge=0.0, le=1.0)
    aspect_ratio: float = Field(gt=0.0)
    has_table_lines: bool
    table_line_count: int = Field(default=0, ge=0)
    #: Normalised colour histogram bins, each in [0, 1]. Required: an empty
    #: histogram is a missing feature, not a legitimate all-zero page.
    colour_histogram: list[float] = Field(min_length=1)
    #: Share of lines flagged handwritten by the detector, in [0, 1].
    handwriting_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Share of page area covered by detected text, in [0, 1].
    text_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _histogram_bins_in_range(self) -> Self:
        for index, bin_value in enumerate(self.colour_histogram):
            if not math.isfinite(bin_value) or not 0.0 <= bin_value <= 1.0:
                raise ValueError(
                    f"colour_histogram[{index}] must be a finite value in "
                    f"[0, 1], got {bin_value!r}"
                )
        if self.table_line_count > 0 and not self.has_table_lines:
            raise ValueError(
                "table_line_count > 0 contradicts has_table_lines=False"
            )
        return self


class PageClass(BaseModel):
    """Output of ``classify_page(page, feats)`` — logit + structural late fusion."""

    model_config = ConfigDict(extra="forbid")

    label: PageClassLabel
    confidence: Confidence
    #: Per-label probabilities from the fused classifier.
    scores: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _scores_cover_the_label(self) -> Self:
        check_distribution(self.scores, "PageClass.scores")
        if self.scores and self.label.value not in self.scores:
            raise ValueError(
                f"predicted label {self.label.value!r} absent from scores "
                f"{sorted(self.scores)}"
            )
        return self


class TextSpan(BaseModel):
    """A character range within one OCR line — the atom of evidence.

    ``end`` is exclusive.  ``line_index`` refers to
    :attr:`OCRLine.line_index`, not to a position in ``OCRPage.lines``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_index: LineIndex
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(
                f"span end {self.end} precedes start {self.start}"
            )
        return self


class ExtractionContext(BaseModel):
    """Everything the SLM is allowed to see for one page (BUILD-BRIEF sec. 3.1).

    Two invariants are enforced structurally here:

    * **Legacy mode** (``discover_fields=False``) — ``target_fields`` must be a subset of
      :data:`SLM_ELIGIBLE_FIELDS`.  Asking the model for an account number,
      IFSC, UTR/RRN, MICR, card fragment, amount or date is a build bug, and
      this raises rather than sending it.
    * Open discovery permits any target name. The pipeline retains masking on
      the laptop profile; local profiles may inspect all OCR spans.
    * **No identifier egress** — there is deliberately *no* map of already
      extracted identifier values.  Under the ``laptop`` profile this object's
      contents cross the network; ``masked_spans`` tells the prompt builder
      where validators already claimed text so it can sentinel-mask it, without
      the values themselves ever entering the context.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    page_no: PageNumber
    #: The page's lines. Their ``line_index`` values are the handles the model
    #: returns for the extractive summary (invariant 2).
    lines: list[OCRLine]
    page_class: PageClassLabel
    primary_script: ScriptCode
    #: Derived hint only; the model must not rely on it for per-line decisions.
    primary_language: LanguageCode | None = None
    discover_fields: bool = False
    target_fields: list[str] = Field(min_length=1)
    #: Regions already claimed by deterministic validators, to be sentinel-masked.
    masked_spans: list[TextSpan] = Field(default_factory=list)
    max_tokens: int = Field(gt=0)
    #: Tokens still available from the per-job budget (``JOB_TOKEN_BUDGET``).
    token_budget_remaining: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _only_slm_eligible_fields(self) -> Self:
        forbidden = sorted(set(self.target_fields) & DETERMINISTIC_FIELDS)
        if forbidden and not self.discover_fields:
            raise ValueError(
                f"deterministic field(s) {', '.join(forbidden)} must never be "
                "sent to the SLM — validators own them (PLAN.md invariant 7)"
            )
        unknown = sorted(set(self.target_fields) - SLM_ELIGIBLE_FIELDS)
        if unknown and not self.discover_fields:
            raise ValueError(
                f"unknown SLM target field(s) {', '.join(unknown)}; allowed: "
                f"{', '.join(sorted(SLM_ELIGIBLE_FIELDS))}"
            )
        duplicates = sorted(
            {f for f in self.target_fields if self.target_fields.count(f) > 1}
        )
        if duplicates:
            raise ValueError(
                f"duplicate target field(s): {', '.join(duplicates)}"
            )
        known_indices = {line.line_index for line in self.lines}
        for span in self.masked_spans:
            if span.line_index not in known_indices:
                raise ValueError(
                    f"masked span references line_index {span.line_index}, "
                    f"absent from page {self.page_no}"
                )
        return self


class RawField(BaseModel):
    """One model-proposed field, **before** grounding, validation or assembly.

    This is not the envelope :class:`app.contracts.envelope.Field`: it has no
    ``evidence``, no ``source`` and no ``validator_passed``, because none of
    those are the model's to assert.  ``assemble.py`` promotes a ``RawField``
    to a ``Field`` only after the grounding gate (invariant 8) accepts it.

    Invariant 2 is enforced here: for ``complaint_summary`` the model must
    return **line indices only** and ``value`` must be ``None``.  A prose
    summary is rejected at the contract boundary, not reviewed later.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    discovered: bool = False
    value: str | None = None
    confidence: Confidence
    #: Lines the model claims the value came from — the summary's whole payload.
    line_indices: list[LineIndex] = Field(default_factory=list)
    #: Optional finer-grained location within a single line.
    span: TextSpan | None = None

    @model_validator(mode="after")
    def _summary_is_extractive(self) -> Self:
        if self.name in DETERMINISTIC_FIELDS and not self.discovered:
            raise ValueError(
                f"{self.name!r} is a deterministic field; the SLM must not "
                "produce it (PLAN.md invariant 7)"
            )
        if self.name == SUMMARY_FIELD_NAME:
            if self.value is not None:
                raise ValueError(
                    "complaint_summary must be selected line indices, not "
                    "prose: RawField.value must be None and line_indices "
                    "non-empty (PLAN.md invariant 2)"
                )
            if not self.line_indices:
                raise ValueError(
                    "complaint_summary requires at least one selected "
                    "line_index (PLAN.md invariant 2)"
                )
        if len(set(self.line_indices)) != len(self.line_indices):
            raise ValueError(
                f"duplicate line_indices in RawField {self.name!r}"
            )
        return self


def check_distribution(
    distribution: dict[str, float],
    label: str,
    tolerance: float = 1e-6,
) -> None:
    """Validate a probability mapping: finite, in [0, 1], summing to ~1."""
    if not distribution:
        return
    total = 0.0
    for key, value in distribution.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{label}[{key!r}] must be a finite value in [0, 1], "
                f"got {value!r}"
            )
        total += value
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{label} must sum to 1.0, got {total!r}")
