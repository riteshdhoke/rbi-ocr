"""Stage 4b — Tier-2 escalation through the heavy VLM, and the merge policy.

Two properties are load-bearing.

**Isolation from the fast path.**  :func:`escalate_page` is a pure function of
its arguments that **never raises into the caller for a provider condition**.
BUILD-BRIEF sec. 5 delivers results in two stages — ``result_version=1,
is_final=false`` at the fast-path SLA, incremented as escalated pages land — so
a Tier-2 call that rate-limits, times out, or turns out not to be implemented on
this profile must degrade to "Tier 1 stands", not to a failed document.  Every
provider error is caught and returned as a typed
:class:`Tier2Outcome`; a programming error (``TypeError``, ``ValueError``) is
not caught, because that is a bug and hiding it would be worse than a failed
page.

**The merge policy, stated in full.**  Tier 2 is a stronger reader, not a
different document, so its output is merged rather than substituted:

1. Lines are paired by **bounding-box IoU** at 0.5.  ``line_index`` cannot be
   used: each tier numbers its own lines and two tiers that found different
   numbers of lines share no index space.
2. For a matched pair, **Tier-2 text and confidence win** — that is what
   escalation bought — but the **Tier-1 ``line_index`` is preserved**.  This is
   the important one: ``line_index`` is the handle the extractive summary
   selects spans by (PLAN.md invariant 2), and renumbering a page after
   escalation would silently repoint every summary index taken before it.
3. A matched line whose **text changed** has its ``language`` reset to ``None``.
   The old label described text that no longer exists; Stage 2b must re-run.
   Keeping it would be exactly the silent staleness invariant 1 exists to stop.
4. ``is_handwritten`` is **OR-ed** across tiers — neither reader can prove the
   absence of handwriting, and Stage 4's heuristic measured pixels that Tier 2
   never reports on.
5. **Unmatched Tier-1 lines are kept.**  Tier 2 not finding a line is a recall
   loss, not evidence the line is absent; dropping it would lose text we have.
6. **Unmatched Tier-2 lines are appended** with fresh indices after the highest
   Tier-1 index, so existing handles keep pointing at the same text.
7. **Empty Tier-2 output leaves Tier 1 completely untouched** and is recorded as
   ``TIER1_RETAINED``. A model that returned nothing has said nothing.

This module does NOT handle:
  * deciding *whether* to escalate — :mod:`app.stages.s4_router` owns that, and
    keeping the two apart is what lets the router be exhaustively tested with no
    model at all;
  * concurrency, queueing or the two-stage webhook.  This is a synchronous pure
    function; WS-F decides when to call it and on which worker pool;
  * language identification.  Merged lines whose text changed come back with
    ``language=None`` **by design** — re-running Stage 2b is the caller's job;
  * page-level anything.  It returns lines and a typed outcome.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Self

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.errors import ProviderError
from app.contracts.ocr import OCRLine, OCRPage, OCRTier
from app.providers.base import ModelProvider, PageImage

__all__ = [
    "DEFAULT_IOU_THRESHOLD",
    "MergePolicy",
    "MergeResult",
    "Tier2Outcome",
    "Tier2Result",
    "apply_tier2",
    "bbox_iou",
    "escalate_page",
    "match_lines_by_iou",
    "merge_tier2_lines",
]

_LOG = logging.getLogger(__name__)

#: Pairing threshold for the merge.  0.5 is the eval harness's own detection
#: threshold, so "matched" means the same thing in both places.
DEFAULT_IOU_THRESHOLD: Final[float] = 0.5

_TIER2_CALLS = Counter(
    "stage_s4b_vlm_calls_total",
    "Tier-2 escalation calls, by outcome.",
    ["outcome"],
)
_TIER2_LATENCY = Histogram(
    "stage_s4b_vlm_seconds",
    "Wall time of one Tier-2 escalation, including the merge.",
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80),
)
_TIER2_MERGES = Counter(
    "stage_s4b_vlm_merge_lines_total",
    "Lines by what the merge did with them.",
    ["disposition"],
)


class Tier2Outcome(StrEnum):
    """Why a Tier-2 attempt ended the way it did.  Always recorded."""

    #: Tier 2 ran and produced lines.
    MERGED = "merged"
    #: Tier 2 ran and produced nothing; Tier 1 stands untouched.
    EMPTY = "empty"
    #: This provider cannot run a VLM at all (``cpu_node`` in Phase 0).
    NOT_SUPPORTED = "not_supported"
    #: Rate limited, unreachable, over budget, or invalid output.
    PROVIDER_ERROR = "provider_error"
    #: Nothing to escalate — the page has no pixels or no Tier-1 lines.
    SKIPPED = "skipped"


class MergePolicy(StrEnum):
    """Which branch of the documented merge policy ran."""

    #: Tier-2 lines merged into the Tier-1 index space.
    IOU_MERGE = "iou_merge"
    #: Tier 2 said nothing usable; Tier-1 output is returned verbatim.
    TIER1_RETAINED = "tier1_retained"


class MergeResult(BaseModel):
    """The merged line set plus a full accounting of what happened to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lines: tuple[OCRLine, ...]
    policy: MergePolicy
    #: Pairs where Tier 2 agreed with Tier 1 character for character.
    unchanged: int = Field(default=0, ge=0)
    #: Pairs where Tier-2 text replaced Tier-1 text.
    replaced: int = Field(default=0, ge=0)
    #: Tier-1 lines Tier 2 did not find, kept per policy rule 5.
    retained: int = Field(default=0, ge=0)
    #: Tier-2 lines with no Tier-1 counterpart, appended per rule 6.
    added: int = Field(default=0, ge=0)
    #: Merged lines whose ``language`` was cleared per rule 3.
    language_cleared: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _indices_are_unique(self) -> Self:
        seen: set[int] = set()
        for line in self.lines:
            if line.line_index in seen:
                raise ValueError(
                    f"merge produced duplicate line_index {line.line_index}; "
                    "indices are the handles the extractive summary selects by "
                    "(PLAN.md invariant 2)"
                )
            seen.add(line.line_index)
        return self


class Tier2Result(BaseModel):
    """Stage-4b output: the merged lines, the outcome, and why."""

    model_config = ConfigDict(extra="forbid")

    outcome: Tier2Outcome
    merge: MergeResult
    #: Raw Tier-2 lines before the merge, kept for the shadow diff.
    tier2_lines: tuple[OCRLine, ...] = ()
    #: Typed error code when ``outcome`` is a failure; ``None`` otherwise.
    error_code: str | None = None
    #: The error's message, for the audit log. Contains no OCR text.
    error_detail: str | None = None

    @model_validator(mode="after")
    def _failures_carry_a_code(self) -> Self:
        failed = self.outcome in (
            Tier2Outcome.NOT_SUPPORTED,
            Tier2Outcome.PROVIDER_ERROR,
        )
        if failed and not self.error_code:
            raise ValueError(
                f"outcome {self.outcome.value!r} must carry a machine-readable "
                "error_code (PLAN.md invariant 11)"
            )
        if not failed and self.error_code:
            raise ValueError(
                f"error_code {self.error_code!r} set on a successful outcome "
                f"({self.outcome.value})"
            )
        return self

    @property
    def succeeded(self) -> bool:
        """Whether Tier 2 actually ran and returned something."""
        return self.outcome is Tier2Outcome.MERGED

    @property
    def lines(self) -> tuple[OCRLine, ...]:
        """The lines the caller should use, whatever happened."""
        return self.merge.lines

    @property
    def tier(self) -> OCRTier:
        """The tier to stamp on the page: Tier 2 only if it contributed."""
        return OCRTier.TIER2 if self.succeeded else OCRTier.TIER1


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _aabb(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Axis-aligned envelope ``(x0, y0, x1, y1)`` of an 8-float quad.

    Axis-aligned rather than rotated: the merge only needs to know *which* line
    this is, and a rotated-polygon intersection is a dependency and a bug
    surface for a decision that a bounding box already settles.  The eval
    harness makes the same trade and says so.
    """
    if len(bbox) != 8:
        raise ValueError(f"bbox must be 8 floats, got {len(bbox)}")
    xs = [float(bbox[i]) for i in range(0, 8, 2)]
    ys = [float(bbox[i]) for i in range(1, 8, 2)]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection-over-union of two 8-float quads' axis-aligned envelopes."""
    ax0, ay0, ax1, ay1 = _aabb(first)
    bx0, by0, bx1, by1 = _aabb(second)
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def match_lines_by_iou(
    tier1: Sequence[OCRLine],
    tier2: Sequence[OCRLine],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[list[tuple[OCRLine, OCRLine]], list[OCRLine], list[OCRLine]]:
    """Greedy one-to-one pairing by descending IoU.

    Greedy rather than Hungarian: text lines on a page do not overlap, so the
    assignment is effectively unambiguous and an optimal matcher would buy
    nothing for a quadratic cost.  Ties break on input order, so the pairing is
    deterministic and a shadow diff is replayable.

    Returns:
        ``(pairs, tier1_only, tier2_only)``, each in input order.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(
            f"iou_threshold must be in (0, 1], got {iou_threshold}"
        )
    candidates: list[tuple[float, int, int]] = []
    for i, first in enumerate(tier1):
        for j, second in enumerate(tier2):
            score = bbox_iou(first.bbox, second.bbox)
            if score >= iou_threshold:
                candidates.append((score, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_first: set[int] = set()
    used_second: set[int] = set()
    paired: list[tuple[int, int]] = []
    for _score, i, j in candidates:
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        paired.append((i, j))
    paired.sort()

    pairs = [(tier1[i], tier2[j]) for i, j in paired]
    tier1_only = [line for i, line in enumerate(tier1) if i not in used_first]
    tier2_only = [line for j, line in enumerate(tier2) if j not in used_second]
    return pairs, tier1_only, tier2_only


# --------------------------------------------------------------------------- #
# The merge
# --------------------------------------------------------------------------- #


def merge_tier2_lines(
    tier1: Sequence[OCRLine],
    tier2: Sequence[OCRLine],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> MergeResult:
    """Merge Tier-2 output into the Tier-1 index space.

    The full policy is in the module docstring; the invariant it exists to
    protect is that a ``line_index`` handed out before escalation still points
    at the same line afterwards (PLAN.md invariant 2).

    Raises:
        ValueError: when either input has duplicate ``line_index`` values.
    """
    _reject_duplicate_indices(tier1, "tier1")
    _reject_duplicate_indices(tier2, "tier2")

    if not tier2:
        _TIER2_MERGES.labels(disposition="retained").inc(len(tier1))
        return MergeResult(
            lines=tuple(tier1),
            policy=MergePolicy.TIER1_RETAINED,
            retained=len(tier1),
        )
    if not tier1:
        renumbered = tuple(
            line.model_copy(update={"line_index": index})
            for index, line in enumerate(tier2)
        )
        _TIER2_MERGES.labels(disposition="added").inc(len(renumbered))
        return MergeResult(
            lines=renumbered,
            policy=MergePolicy.IOU_MERGE,
            added=len(renumbered),
        )

    pairs, tier1_only, tier2_only = match_lines_by_iou(
        tier1, tier2, iou_threshold=iou_threshold
    )
    replacement: dict[int, OCRLine] = {}
    unchanged = 0
    replaced = 0
    cleared = 0
    for first, second in pairs:
        merged, changed = _merge_pair(first, second)
        replacement[first.line_index] = merged
        if changed:
            replaced += 1
            cleared += int(first.language is not None)
        else:
            unchanged += 1

    lines: list[OCRLine] = [
        replacement.get(line.line_index, line) for line in tier1
    ]
    next_index = max((line.line_index for line in lines), default=-1) + 1
    for extra in tier2_only:
        lines.append(extra.model_copy(update={"line_index": next_index}))
        next_index += 1

    _TIER2_MERGES.labels(disposition="unchanged").inc(unchanged)
    _TIER2_MERGES.labels(disposition="replaced").inc(replaced)
    _TIER2_MERGES.labels(disposition="retained").inc(len(tier1_only))
    _TIER2_MERGES.labels(disposition="added").inc(len(tier2_only))
    return MergeResult(
        lines=tuple(lines),
        policy=MergePolicy.IOU_MERGE,
        unchanged=unchanged,
        replaced=replaced,
        retained=len(tier1_only),
        added=len(tier2_only),
        language_cleared=cleared,
    )


def _merge_pair(first: OCRLine, second: OCRLine) -> tuple[OCRLine, bool]:
    """Apply rules 2-4 to one matched pair.  Returns ``(line, text_changed)``."""
    changed = first.text != second.text
    return (
        first.model_copy(
            update={
                "text": second.text,
                "confidence": second.confidence,
                "script": second.script,
                # rule 3: a label that described replaced text is stale.
                "language": None if changed else first.language,
                # rule 4: neither reader can prove handwriting is absent.
                "is_handwritten": first.is_handwritten or second.is_handwritten,
            }
        ),
        changed,
    )


def _reject_duplicate_indices(lines: Sequence[OCRLine], label: str) -> None:
    """Duplicate indices make the merge's index space meaningless."""
    seen: set[int] = set()
    for line in lines:
        if line.line_index in seen:
            raise ValueError(
                f"{label} lines carry duplicate line_index {line.line_index}"
            )
        seen.add(line.line_index)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def escalate_page(
    page: PageImage | None,
    tier1_lines: Sequence[OCRLine],
    *,
    provider: ModelProvider,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Tier2Result:
    """Run Tier 2 for one page and merge the result over Tier 1.

    **Never raises for a provider condition.**  Rate limits, unreachable
    endpoints, token-budget breaches, invalid model output and "this profile has
    no VLM" all come back as a :class:`Tier2Result` whose ``merge.lines`` are
    the Tier-1 lines unchanged.  That is what keeps escalation off the fast
    path's critical section: the worst case is that the page ships at Tier-1
    quality with a recorded reason, which is the two-stage delivery model
    working as designed.

    Args:
        page: the same repaired page array Stage 4 recognised (decision C1).
            ``None`` short-circuits to :attr:`Tier2Outcome.SKIPPED`.
        tier1_lines: what Tier 1 produced. May be empty — a page where
            detection found nothing is exactly a page worth a second reader.

    Raises:
        TypeError / ValueError: only for a programming error in the merge (a
            malformed bbox, duplicate indices). Those are bugs, not runtime
            conditions, and swallowing them would hide a corrupted index space.
    """
    if page is None:
        _TIER2_CALLS.labels(outcome=Tier2Outcome.SKIPPED.value).inc()
        return Tier2Result(
            outcome=Tier2Outcome.SKIPPED,
            merge=MergeResult(
                lines=tuple(tier1_lines),
                policy=MergePolicy.TIER1_RETAINED,
                retained=len(tier1_lines),
            ),
        )

    with _TIER2_LATENCY.time():
        try:
            heavy = list(provider.ocr_page_heavy(page))
        except ProviderError as exc:
            outcome = (
                Tier2Outcome.NOT_SUPPORTED
                if exc.code == "PROVIDER_METHOD_NOT_SUPPORTED"
                else Tier2Outcome.PROVIDER_ERROR
            )
            _TIER2_CALLS.labels(outcome=outcome.value).inc()
            _LOG.warning(
                "tier-2 escalation unavailable (%s); the page ships at Tier-1 "
                "quality with the reason recorded",
                exc.code,
            )
            return Tier2Result(
                outcome=outcome,
                merge=MergeResult(
                    lines=tuple(tier1_lines),
                    policy=MergePolicy.TIER1_RETAINED,
                    retained=len(tier1_lines),
                ),
                error_code=exc.code,
                error_detail=exc.message,
            )

        merge = merge_tier2_lines(
            tier1_lines, heavy, iou_threshold=iou_threshold
        )

    outcome = Tier2Outcome.MERGED if heavy else Tier2Outcome.EMPTY
    _TIER2_CALLS.labels(outcome=outcome.value).inc()
    _LOG.debug(
        "stage s4b outcome=%s tier1=%d tier2=%d replaced=%d added=%d "
        "retained=%d",
        outcome.value,
        len(tier1_lines),
        len(heavy),
        merge.replaced,
        merge.added,
        merge.retained,
    )
    return Tier2Result(
        outcome=outcome, merge=merge, tier2_lines=tuple(heavy)
    )


def apply_tier2(page: OCRPage, result: Tier2Result) -> OCRPage:
    """Return a copy of ``page`` carrying the merged lines and the right tier.

    A convenience for the orchestrator (WS-F) so the tier stamp and the line
    swap cannot get out of step: a page whose lines came from Tier 2 must say
    ``tier=tier2``, and a page where Tier 2 failed must not.
    """
    return page.model_copy(
        update={
            "lines": list(result.lines),
            "tier": (
                result.tier
                if result.lines
                else (OCRTier.NONE if page.tier is not OCRTier.BYPASS else page.tier)
            ),
        }
    )
