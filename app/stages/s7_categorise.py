"""Stage 7 — complaint categorisation.  **THIS IS A STUB, NOT A CLASSIFIER.**

    ############################################################
    #  NOT PRODUCTION.  Owner decision D3 (PLAN.md scope row):  #
    #  "stubbed categorisation classifier behind an interface   #
    #  (marked NOT-PRODUCTION; RB-IOS taxonomy absent)".        #
    #                                                           #
    #  What ships here is a deterministic keyword lookup over a #
    #  SYNTHETIC placeholder tag list that the build agent      #
    #  invented.  It is not trained on anything, it has never   #
    #  been evaluated against a labelled set, and its accuracy  #
    #  is unknown and unmeasured.  Nothing may be routed,       #
    #  prioritised or disposed of on its output.                #
    ############################################################

Why a stub exists at all: PRD P1-4 categorises against the **RB-IOS ground
list** with a distilled encoder classifier, and the PRD's own hard-dependency
list names "RB-IOS ground taxonomy, confirmed and versioned" as *blocking* that
requirement.  The taxonomy is not available to this build.  A slot that does
not exist cannot be wired, tested or measured, so the slot exists with an
honest occupant: the interface (:class:`Categoriser`), the closed tag set
(:data:`app.resources` ``rbios_categories.json``, every entry marked
``PLACEHOLDER``), the envelope field it fills
(:attr:`app.contracts.envelope.DocumentResult.supporting_doc_tags`), and a
baseline that makes the end-to-end path run.

Replacing the stub is meant to be a *drop-in*: implement :class:`Categoriser`
with the real distilled encoder, drop in the real taxonomy JSON with
``authoritative: true`` and ``not_production: false``, and the call site does
not change.  :func:`require_production_categoriser` is the gate a deployment
calls to refuse to start with the stub still in place.

PRD Non-Goal 2 — "no scoring, prioritisation, or recommended disposition" — is
respected: :attr:`CategoryScore.score` counts *keyword evidence*, so a reviewer
can see why a tag appeared.  It is not a priority, a severity or a
recommendation, and nothing in this module ranks complaints against each other.

This module does NOT handle:
  * **being correct.**  See the banner.  Precision and recall are unmeasured;
    there is no labelled set to measure them against;
  * the SLM.  PRD P1-4 is explicit that categorisation is an encoder task, not
    a generation task, and PLAN.md invariant 7 keeps deterministic content away
    from the SLM regardless.  Nothing here calls a provider or the network;
  * field extraction of any kind — supporting documents get tags, not fields
    (PRD P1-3);
  * language identification.  It matches Latin and Devanagari keywords over
    whatever text it is handed; Stage 2b owns what language that text is;
  * loading the real RB-IOS list.  The loader will read it (same schema), but
    nobody in this build has seen it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Final, Protocol, Self, runtime_checkable

from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.errors import ConfigurationError, PipelineError
from app.contracts.ocr import Confidence, OCRLine, OCRPage, PageClassLabel
from app.resources import resource_path

__all__ = [
    "DEFAULT_MAX_TAGS",
    "SECONDARY_TAG_SHARE",
    "TAXONOMY_FILENAME",
    "CategorisationRequest",
    "CategorisationResult",
    "Categoriser",
    "CategoriserNotProduction",
    "Category",
    "CategoryScore",
    "CategoryTaxonomy",
    "KeywordCategoriser",
    "NotProduction",
    "NotProductionWarning",
    "TaxonomyInvalid",
    "categorise_document",
    "default_categoriser",
    "load_taxonomy",
    "require_production_categoriser",
]

_LOG: Final = logging.getLogger(__name__)

#: The committed placeholder tag list.  A real deployment points the loader at
#: the RB-IOS list instead; the schema is identical.
TAXONOMY_FILENAME: Final[str] = "rbios_categories.json"

#: At most this many tags per document.  A closed tag set with everything in it
#: is not a categorisation.
DEFAULT_MAX_TAGS: Final[int] = 3

#: A runner-up needs at least this share of the winner's evidence to be emitted
#: as a secondary tag.
SECONDARY_TAG_SHARE: Final[float] = 0.6

_CATEGORISATIONS = Counter(
    "stage_s7_categorisations_total",
    "Stage-7 categorisations, by categoriser and primary tag.",
    ["categoriser", "primary"],
)
_CATEGORISE_LATENCY = Histogram(
    "stage_s7_categorise_seconds",
    "Wall time of one Stage-7 categorisation.",
)


# --------------------------------------------------------------------------- #
# Typed errors owned by this stage
# --------------------------------------------------------------------------- #


class TaxonomyInvalid(PipelineError):
    """The category file is absent, unreadable, or does not match the schema.

    Loud on purpose: a categoriser that silently falls back to an empty tag set
    would emit ``supporting_doc_tags=()`` for every document, which reads like
    "nothing matched" rather than "the table never loaded".
    """

    code: ClassVar[str] = "TAXONOMY_INVALID"
    http_status: ClassVar[int] = 500


class CategoriserNotProduction(ConfigurationError):
    """A stub categoriser was used where a production one was required.

    Raised only by :func:`require_production_categoriser`, which a deployment
    calls at startup.  The stub itself never raises — it must keep working, or
    the pipeline slot it exists to prove could not be tested.
    """

    code: ClassVar[str] = "CATEGORISER_NOT_PRODUCTION"
    http_status: ClassVar[int] = 500


class NotProductionWarning(UserWarning):
    """Emitted when a :class:`NotProduction` implementation is constructed."""


# --------------------------------------------------------------------------- #
# The taxonomy
# --------------------------------------------------------------------------- #

_CODE_PATTERN: Final[str] = r"^[A-Z][A-Z0-9_]{2,63}$"


class Category(BaseModel):
    """One tag in the closed set.

    ``keywords_en`` / ``keywords_hi`` exist for the *baseline* only.  A real
    encoder classifier ignores them; they stay in the schema because deleting
    them would make the placeholder file unable to describe itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=_CODE_PATTERN)
    label_en: str = Field(min_length=1)
    label_hi: str = Field(min_length=1)
    description: str = Field(min_length=1)
    #: Matched on ASCII word boundaries after case folding.
    keywords_en: tuple[str, ...] = ()
    #: Matched as substrings: Devanagari inflection makes a boundary match miss
    #: the stem it was written for.
    keywords_hi: tuple[str, ...] = ()

    @property
    def keywords(self) -> tuple[str, ...]:
        """Every keyword, English first, in file order."""
        return self.keywords_en + self.keywords_hi


class CategoryTaxonomy(BaseModel):
    """The closed tag set and the provenance that says how far to trust it."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_name: str = Field(alias="schema", min_length=1)
    taxonomy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: str = Field(min_length=1)
    #: ``false`` for the placeholder. Mirrors the convention
    #: ``ifsc_master_sample.json`` established for the synthetic IFSC master.
    authoritative: bool
    #: ``true`` while this table must not drive a production decision.
    not_production: bool
    source: str = Field(min_length=1)
    notes: str = Field(min_length=1)
    blocking_dependency: str | None = None
    keyword_note: str | None = None
    fallback_code: str = Field(pattern=_CODE_PATTERN)
    categories: tuple[Category, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _closed_set_is_closed(self) -> Self:
        codes = [category.code for category in self.categories]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if duplicates:
            raise ValueError(f"duplicate category codes: {', '.join(duplicates)}")
        if self.fallback_code not in codes:
            raise ValueError(
                f"fallback_code {self.fallback_code!r} is not one of the "
                "categories; a fallback outside the closed set is not a "
                "closed set"
            )
        if self.authoritative and self.not_production:
            raise ValueError(
                "authoritative=true with not_production=true is "
                "self-contradictory: either the list may drive a decision or "
                "it may not"
            )
        return self

    @property
    def codes(self) -> frozenset[str]:
        """Every code in the set."""
        return frozenset(category.code for category in self.categories)

    def by_code(self, code: str) -> Category:
        """Look a category up.

        Raises:
            KeyError: when the code is outside the closed set.
        """
        for category in self.categories:
            if category.code == code:
                return category
        raise KeyError(f"{code!r} is not in taxonomy {self.taxonomy_id!r}")

    @property
    def fallback(self) -> Category:
        """The category used when nothing matched."""
        return self.by_code(self.fallback_code)

    @classmethod
    def load(cls, path: Path | None = None) -> CategoryTaxonomy:
        """Load a taxonomy from JSON.

        Args:
            path: the file to read. ``None`` loads the committed placeholder.

        Raises:
            TaxonomyInvalid: file missing, not JSON, or not matching the
                schema. Never returns an empty taxonomy.
        """
        target = path if path is not None else resource_path(TAXONOMY_FILENAME)
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise TaxonomyInvalid(
                f"category taxonomy {target} could not be read: {exc}",
                context={"path": str(target)},
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaxonomyInvalid(
                f"category taxonomy {target} is not valid JSON: {exc}",
                context={"path": str(target)},
            ) from exc
        if not isinstance(payload, Mapping):
            raise TaxonomyInvalid(
                f"category taxonomy {target} must be a JSON object",
                context={"path": str(target)},
            )
        try:
            return cls.model_validate(dict(payload))
        except ValueError as exc:
            raise TaxonomyInvalid(
                f"category taxonomy {target} does not match the schema: {exc}",
                context={"path": str(target)},
            ) from exc


@lru_cache(maxsize=4)
def load_taxonomy(path: Path | None = None) -> CategoryTaxonomy:
    """Cached :meth:`CategoryTaxonomy.load` — the table is read once."""
    return CategoryTaxonomy.load(path)


# --------------------------------------------------------------------------- #
# Request / result contracts
# --------------------------------------------------------------------------- #


class CategorisationRequest(BaseModel):
    """What a categoriser is allowed to see.

    Text only.  No images, no provider, no already-extracted identifiers: an
    account number cannot inform a category, and keeping it out means this
    stage has nothing to leak.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    #: OCR text, one entry per line, in reading order.
    lines: tuple[str, ...] = ()
    #: The page/document class, when the classifier has run. Recorded for the
    #: real model to condition on; the baseline ignores it.
    page_class: PageClassLabel = PageClassLabel.UNKNOWN
    max_tags: int = Field(default=DEFAULT_MAX_TAGS, ge=1, le=16)

    @property
    def text(self) -> str:
        """The lines joined by newlines."""
        return "\n".join(self.lines)

    @classmethod
    def from_ocr_lines(
        cls,
        document_id: str,
        lines: Sequence[OCRLine],
        *,
        page_class: PageClassLabel = PageClassLabel.UNKNOWN,
        max_tags: int = DEFAULT_MAX_TAGS,
    ) -> CategorisationRequest:
        """Build a request from recognised lines (PLAN.md invariant 1 safe:
        nothing page-level is read off them)."""
        return cls(
            document_id=document_id,
            lines=tuple(line.text for line in lines),
            page_class=page_class,
            max_tags=max_tags,
        )

    @classmethod
    def from_pages(
        cls,
        document_id: str,
        pages: Sequence[OCRPage],
        *,
        page_class: PageClassLabel = PageClassLabel.UNKNOWN,
        max_tags: int = DEFAULT_MAX_TAGS,
    ) -> CategorisationRequest:
        """Build a request from every line of every page, in page order."""
        texts: list[str] = []
        for page in sorted(pages, key=lambda page: page.page_no):
            texts.extend(line.text for line in page.lines)
        return cls(
            document_id=document_id,
            lines=tuple(texts),
            page_class=page_class,
            max_tags=max_tags,
        )


class CategoryScore(BaseModel):
    """Evidence for one category — *not* a priority or a severity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=_CODE_PATTERN)
    label_en: str = Field(min_length=1)
    label_hi: str = Field(min_length=1)
    #: Distinct keywords that matched. Counting distinct keywords rather than
    #: occurrences stops one repeated word (or one OCR stutter) outvoting a
    #: category that matched three different ones.
    score: int = Field(ge=0)
    confidence: Confidence
    matched_keywords: tuple[str, ...] = ()


class CategorisationResult(BaseModel):
    """Tags for one document, with the provenance that bounds their use.

    ``tags`` is shaped to drop straight into
    :attr:`app.contracts.envelope.DocumentResult.supporting_doc_tags`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    #: Closed-set codes, most confident first. Never empty: the fallback code
    #: is a real answer.
    tags: tuple[str, ...] = Field(min_length=1)
    #: ``tags[0]``, spelled out because most callers only want this.
    primary: str = Field(pattern=_CODE_PATTERN)
    confidence: Confidence
    scores: tuple[CategoryScore, ...] = ()
    categoriser: str = Field(min_length=1)
    #: **False** for everything in this module today.
    is_production: bool
    #: Required whenever ``is_production`` is false. A consumer that ignores
    #: this field cannot say it was not told.
    not_production_reason: str | None = None
    taxonomy_id: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    taxonomy_authoritative: bool

    @model_validator(mode="after")
    def _provenance_rules(self) -> Self:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError(f"duplicate tags: {self.tags}")
        if self.primary != self.tags[0]:
            raise ValueError(
                f"primary {self.primary!r} must be tags[0] ({self.tags[0]!r})"
            )
        if not self.is_production and not self.not_production_reason:
            raise ValueError(
                "a non-production categorisation must say why in "
                "not_production_reason (owner decision D3): a consumer cannot "
                "discount a result it was never told to discount"
            )
        if self.is_production and self.not_production_reason:
            raise ValueError(
                "not_production_reason set on a production categorisation"
            )
        if self.is_production and not self.taxonomy_authoritative:
            raise ValueError(
                "a production categorisation cannot rest on a non-"
                "authoritative taxonomy"
            )
        return self


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


@runtime_checkable
class Categoriser(Protocol):
    """What Stage 7 needs from a classifier, and nothing more.

    Three members, so that swapping the baseline for the real distilled encoder
    touches no call site.  ``isinstance`` works against this protocol (data
    members are checked by presence in 3.12); ``issubclass`` does not, and
    ``tests/unit/test_s7_categorise.py`` pins that.
    """

    #: Stable identifier recorded on every result, e.g. ``keyword-baseline-v0``.
    name: str
    #: False for anything that must not drive a decision (owner decision D3).
    is_production: bool

    def categorise(self, request: CategorisationRequest) -> CategorisationResult:
        """Tag one document from its OCR text."""
        ...


class NotProduction:
    """Marker base class for an implementation that must not drive a decision.

    Owner decision D3 asks for the stub to be *marked*, not merely documented.
    Inheriting from this class:

    * pins ``is_production = False`` as a class attribute, so
      :func:`require_production_categoriser` can refuse it without running it;
    * requires ``not_production_reason`` to be set by the subclass, so the
      reason travels on every result;
    * emits :class:`NotProductionWarning` on construction, so a process that
      accidentally builds one says so in its logs.

    The marker is deliberately not an ``ABC``: a real implementation must be
    able to satisfy :class:`Categoriser` without importing anything from here.
    """

    #: Never overridden to True. A production implementation does not inherit
    #: from this class at all.
    is_production: ClassVar[bool] = False
    #: Subclasses must set this; it lands on every result they produce.
    not_production_reason: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.not_production_reason:
            raise TypeError(
                f"{cls.__name__} inherits NotProduction and must set "
                "not_production_reason: the reason is copied onto every "
                "result it produces (owner decision D3)"
            )

    def _warn_not_production(self, *, stacklevel: int = 3) -> None:
        """Announce the stub once per construction."""
        warnings.warn(
            f"{type(self).__name__} is NOT a production classifier: "
            f"{self.not_production_reason}",
            NotProductionWarning,
            stacklevel=stacklevel,
        )
        _LOG.warning(
            "stage s7 using non-production categoriser %s: %s",
            type(self).__name__,
            self.not_production_reason,
        )


def require_production_categoriser(categoriser: Categoriser | NotProduction) -> None:
    """Refuse a stub where a real classifier is required.

    The gate a deployment calls at startup once the RB-IOS taxonomy and the
    distilled encoder exist.  :func:`app.api.app.create_app` calls it when
    ``REQUIRE_PRODUCTION_CATEGORISER`` is set, and warns instead when it is not
    — the refusal has to be opt-in while the only categoriser that exists is a
    stub, because otherwise it would refuse every legitimate deployment.

    The parameter accepts :class:`NotProduction` as well as :class:`Categoriser`
    precisely because *refusing a stub* is the point: the stub pins
    ``is_production`` as a ``ClassVar``, which does not satisfy the protocol's
    instance-variable member, and a gate that could not be handed the thing it
    exists to reject would be unusable at its only call site.

    Raises:
        CategoriserNotProduction: when ``categoriser.is_production`` is false.
    """
    if not getattr(categoriser, "is_production", False):
        reason = getattr(
            categoriser, "not_production_reason", "no reason recorded"
        )
        raise CategoriserNotProduction(
            f"categoriser {getattr(categoriser, 'name', categoriser)!r} is not "
            f"a production classifier: {reason}",
            context={
                "categoriser": str(getattr(categoriser, "name", categoriser)),
                "reason": str(reason),
            },
        )


# --------------------------------------------------------------------------- #
# The baseline
# --------------------------------------------------------------------------- #

#: A keyword made only of these characters is matched on word boundaries.
_ASCII_KEYWORD = re.compile(r"^[ -~]+$")
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Case-fold and collapse whitespace; nothing else is touched.

    Deliberately not stripping punctuation: the boundary lookarounds handle it,
    and removing it would join "atm.cash" into one token.
    """
    return _WHITESPACE.sub(" ", text.casefold())


def _keyword_matcher(keyword: str) -> re.Pattern[str] | None:
    """Compiled boundary matcher for an ASCII keyword, else ``None``.

    ``None`` means "match this as a substring" — the Devanagari path, where a
    boundary match on a stem written without its inflection would never fire.
    """
    if not _ASCII_KEYWORD.match(keyword):
        return None
    return re.compile(
        r"(?<![0-9a-z])" + re.escape(keyword.casefold()) + r"(?![0-9a-z])"
    )


class KeywordCategoriser(NotProduction):
    """Deterministic keyword baseline.  **NOT the production classifier.**

    Counts how many *distinct* keywords of each category appear in the text,
    ranks by that count with the taxonomy's own order as the tie-break, and
    emits the winner plus any runner-up holding at least
    :data:`SECONDARY_TAG_SHARE` of the winner's evidence.  Nothing matched
    means the fallback code, at confidence 0.0 — an honest "not categorised"
    rather than the least-bad guess.

    It exists so the pipeline slot, the envelope field and the tests are real.
    Its accuracy is unknown; see the module banner.
    """

    name: ClassVar[str] = "keyword-baseline-v0"
    not_production_reason: ClassVar[str] = (
        "deterministic keyword lookup over a SYNTHETIC placeholder tag list; "
        "the RB-IOS ground taxonomy (PRD P1-4 hard dependency) is not "
        "available and no labelled set exists to measure this against"
    )

    def __init__(
        self,
        taxonomy: CategoryTaxonomy | None = None,
        *,
        secondary_share: float = SECONDARY_TAG_SHARE,
        warn: bool = True,
    ) -> None:
        """
        Args:
            taxonomy: the closed tag set. ``None`` loads the committed
                placeholder.
            secondary_share: a runner-up needs this share of the winner's
                evidence to be emitted as a second tag.
            warn: emit :class:`NotProductionWarning`. Pass ``False`` only where
                the warning has already been surfaced (bulk evaluation).

        Raises:
            ValueError: when ``secondary_share`` is outside ``(0, 1]``.
            TaxonomyInvalid: when the taxonomy cannot be loaded.
        """
        if not 0.0 < secondary_share <= 1.0:
            raise ValueError(
                f"secondary_share must be in (0, 1], got {secondary_share}"
            )
        self.taxonomy: CategoryTaxonomy = taxonomy or load_taxonomy()
        self.secondary_share: float = secondary_share
        # (category index, keyword, compiled matcher or None) — built once so
        # categorising N documents compiles nothing.
        self._matchers: tuple[tuple[int, str, re.Pattern[str] | None], ...] = tuple(
            (index, keyword, _keyword_matcher(keyword))
            for index, category in enumerate(self.taxonomy.categories)
            for keyword in category.keywords
        )
        if warn:
            self._warn_not_production()

    def categorise(self, request: CategorisationRequest) -> CategorisationResult:
        """Tag one document.  Pure: same text in, same tags out, always."""
        with _CATEGORISE_LATENCY.time():
            return self._categorise(request)

    def _categorise(self, request: CategorisationRequest) -> CategorisationResult:
        haystack = _normalise(request.text)
        hits: dict[int, list[str]] = {}
        for index, keyword, matcher in self._matchers:
            found = (
                matcher.search(haystack) is not None
                if matcher is not None
                else keyword.casefold() in haystack
            )
            if found:
                hits.setdefault(index, []).append(keyword)

        total = sum(len(matched) for matched in hits.values())
        scores = tuple(
            CategoryScore(
                code=self.taxonomy.categories[index].code,
                label_en=self.taxonomy.categories[index].label_en,
                label_hi=self.taxonomy.categories[index].label_hi,
                score=len(hits[index]),
                confidence=len(hits[index]) / total if total else 0.0,
                matched_keywords=tuple(hits[index]),
            )
            # sorted by evidence, then by the taxonomy's own order: the tie
            # break must not depend on dict iteration or on the input.
            for index in sorted(hits, key=lambda i: (-len(hits[i]), i))
        )

        if not scores:
            fallback = self.taxonomy.fallback
            return self._result(
                request,
                tags=(fallback.code,),
                confidence=0.0,
                scores=(),
            )

        best = scores[0].score
        # ceil, not truncation: "at least 60% of the winner's evidence" against
        # a winner with 3 hits must mean 2, not 1.
        floor = max(1, math.ceil(best * self.secondary_share))
        tags = tuple(
            score.code for score in scores if score.score >= floor
        )[: request.max_tags]
        return self._result(
            request, tags=tags, confidence=scores[0].confidence, scores=scores
        )

    def _result(
        self,
        request: CategorisationRequest,
        *,
        tags: tuple[str, ...],
        confidence: float,
        scores: tuple[CategoryScore, ...],
    ) -> CategorisationResult:
        result = CategorisationResult(
            document_id=request.document_id,
            tags=tags,
            primary=tags[0],
            confidence=confidence,
            scores=scores,
            categoriser=self.name,
            is_production=self.is_production,
            not_production_reason=self.not_production_reason,
            taxonomy_id=self.taxonomy.taxonomy_id,
            taxonomy_version=self.taxonomy.version,
            taxonomy_authoritative=self.taxonomy.authoritative,
        )
        _CATEGORISATIONS.labels(
            categoriser=self.name, primary=result.primary
        ).inc()
        _LOG.debug(
            "stage s7 document=%s tags=%s categoriser=%s production=%s",
            request.document_id,
            result.tags,
            self.name,
            result.is_production,
        )
        return result


@lru_cache(maxsize=1)
def default_categoriser() -> KeywordCategoriser:
    """The process-wide baseline instance.

    Cached so the :class:`NotProductionWarning` is emitted once rather than per
    document — a warning nobody can silence is a warning everybody filters out.
    """
    return KeywordCategoriser()


def categorise_document(
    document_id: str,
    lines: Iterable[str] | Sequence[OCRLine],
    *,
    page_class: PageClassLabel = PageClassLabel.UNKNOWN,
    max_tags: int = DEFAULT_MAX_TAGS,
    categoriser: Categoriser | None = None,
) -> CategorisationResult:
    """Convenience entry point for the orchestrator (WS-F).

    Accepts recognised :class:`~app.contracts.ocr.OCRLine` objects or bare
    strings.  ``result.tags`` goes into
    :attr:`app.contracts.envelope.DocumentResult.supporting_doc_tags`; nothing
    else in the envelope changes.
    """
    materialised = list(lines)
    texts = tuple(
        line.text if isinstance(line, OCRLine) else str(line)
        for line in materialised
    )
    request = CategorisationRequest(
        document_id=document_id,
        lines=texts,
        page_class=page_class,
        max_tags=max_tags,
    )
    engine = categoriser if categoriser is not None else default_categoriser()
    return engine.categorise(request)
