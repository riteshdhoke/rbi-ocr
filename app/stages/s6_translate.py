"""Stage 6 — per-field, entity-masked machine translation.

PLAN.md invariant 10, restated as the shape of this module: ``value`` is the
native-script string and is immutable; ``value_en`` is **advisory** and cannot
exist without an ``mt_engine`` tag; every validator-claimed entity is replaced
with a sentinel before the text reaches the MT engine and restored
byte-identically afterwards.

The masking is not politeness — it is the only thing standing between a
200M-parameter distilled translator and a corrupted account number.  Sequence
models rewrite digits: they regroup, they transliterate, they drop a repeated
character, and they do it with perfect fluency.  So the digits never go through
the model at all; a sentinel does, and the original characters are put back from
memory:

    माझ्या खात्यातून ₹45,000 डेबिट झाले   ->  ... ZQ0ZQ ... ZQ1ZQ ...
                                          ->  MT
                                          ->  restore -> ₹45,000 verbatim

**If a sentinel does not come back, the translation is discarded.**  There is no
partial restore and no best-effort patch-up: the field keeps its native
``value``, gains ``TRANSLATION_UNAVAILABLE``, and a human sees it.  A
``value_en`` that lost an identifier would be worse than no ``value_en`` at all,
because a case officer would read it.

Identifier fields are not translated in the first place — "₹45,000.00" has no
English rendering — so :data:`TRANSLATABLE_FIELDS` is the free-text set.
Whole-page translation is forbidden (BUILD-BRIEF sec. 8); this module only ever
sees a field's value.

This module does NOT handle:
  * choosing the MT engine or loading it.  ``provider.translate`` is called and
    ``mt_engine`` is passed in by the caller that knows which engine it wired —
    ``app.providers.local.MT_ENGINE_NAME`` is the canonical string for
    IndicTrans2, and the tag exists precisely so a consumer can tell that apart
    from an LLM paraphrase;
  * translating ``complaint_summary``.  It is a verbatim concatenation of OCR
    spans (invariant 2) and the contract re-derives it from those spans; an
    English rendering would not survive that check and has no field to live in;
  * back-translation, quality estimation or any confidence for ``value_en``;
  * detecting the source language.  Stage 2b labels lines; the caller passes the
    ISO 639-3 code;
  * sentence segmentation.  A field value is one unit of text by construction;
  * making ``value_en`` authoritative.  Nothing downstream may key business
    logic on it, and nothing in this module encourages that.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ClassVar, Final, Protocol

from pydantic import BaseModel, ConfigDict

from app.contracts.envelope import Field, FieldFlag
from app.contracts.errors import PipelineError, ProviderError
from app.contracts.ocr import SLM_ELIGIBLE_FIELDS, SUMMARY_FIELD_NAME
from app.stages.s5_validators import INDIC_DIGIT_TRANSLATION

__all__ = [
    "DEFAULT_PROTECT_PATTERNS",
    "MT_SENTINEL_TEMPLATE",
    "TRANSLATABLE_FIELDS",
    "MaskResult",
    "SentinelCollision",
    "SentinelRestoreFailed",
    "SupportsTranslation",
    "mask_entities",
    "restore_entities",
    "translate_fields",
    "translate_text",
]

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Typed failures owned by this stage
# --------------------------------------------------------------------------- #


class SentinelRestoreFailed(PipelineError):
    """The MT engine did not return every sentinel it was given.

    Data, not disaster: the caller drops the translation and flags the field.
    Raised rather than returned so no code path can accidentally publish a
    half-restored string — the only way past this exception is to decide, in
    writing, what to do without a translation.
    """

    code: ClassVar[str] = "SENTINEL_RESTORE_FAILED"
    http_status: ClassVar[int] = 502


class SentinelCollision(PipelineError):
    """The source text already contains every sentinel form we can generate.

    Effectively unreachable on real complaint text; raised rather than silently
    picking a colliding marker, because a collision would make the restore
    ambiguous and could splice an identifier into the wrong position.
    """

    code: ClassVar[str] = "SENTINEL_COLLISION"
    http_status: ClassVar[int] = 500


class SupportsTranslation(Protocol):
    """The one slice of ``ModelProvider`` this stage needs."""

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        ...


# --------------------------------------------------------------------------- #
# Sentinels
# --------------------------------------------------------------------------- #

#: Default sentinel shape: **letters and digits only, no delimiter punctuation.**
#:
#: This was measured, not chosen.  An earlier ``[[E{index}]]`` form looked
#: obviously safe and was destroyed by the engine this service actually ships
#: with: IndicTrans2 dist-200M splits on the brackets, translates or drops what
#: is inside, and returns ``[ addressE1 ]`` — so *every* field carrying an
#: identifier lost its translation.  Sweeping candidate shapes through the real
#: CT2 weights, every punctuation-delimited form (``<ID0>``, ``__ID0__``,
#: ``@0@``, ``#0#``, ``${0}``, ``[0]``, ``~0~``) was mangled and every bare
#: alphanumeric form survived (see ``docs/decisions-wse3.md`` D-E3-3 for the
#: table).  SentencePiece keeps an unknown alphanumeric run together; it does not
#: keep punctuation glued to it.
#:
#: The marker is repeated on both sides so ``ZQ1ZQ`` cannot be a prefix of
#: ``ZQ10ZQ``, which is what makes restore-by-replacement order-independent.
#: ``ZQ`` because the bigram occurs in neither English nor romanised Indic text;
#: :func:`_pick_marker` escalates if a document proves otherwise.
MT_SENTINEL_TEMPLATE: Final[str] = "ZQ{index}ZQ"

#: Markers tried, in order, when the source text already contains one.
_MARKERS: Final[tuple[str, ...]] = tuple("ZQ" * width for width in range(1, 9))


def _pick_marker(text: str) -> str:
    """Choose a sentinel marker that does not already occur in ``text``.

    Raises:
        SentinelCollision: when every candidate occurs — see the class docstring.
    """
    for marker in _MARKERS:
        if marker not in text:
            return marker
    raise SentinelCollision(
        "source text contains every candidate sentinel marker; refusing to mask "
        "with an ambiguous one",
        context={"length": len(text)},
    )


def _sentinel(marker: str, index: int) -> str:
    return f"{marker}{index}{marker}"


def _restore_pattern(marker: str, index: int) -> re.Pattern[str]:
    """Tolerant matcher: an MT engine may pad the sentinel with whitespace.

    Tolerance is only about *finding* the sentinel.  What is written back is
    always the stored original, so the restored entity is byte-identical
    regardless of how the engine spaced the placeholder.
    """
    escaped = re.escape(marker)
    return re.compile(rf"{escaped}\s*{index}\s*{escaped}")


def _constant_replacer(replacement: str) -> Callable[[re.Match[str]], str]:
    """A ``re.sub`` callback that ignores the match and writes ``replacement``.

    A callback rather than a replacement *template* on purpose: in a template
    ``\\`` and ``\\g<...>`` have meaning, and what is being written back is an
    untrusted OCR substring.  Going through a function means the stored original
    is copied byte for byte, whatever characters it happens to contain.
    """

    def replace(_match: re.Match[str]) -> str:
        return replacement

    return replace


# --------------------------------------------------------------------------- #
# What counts as an entity
# --------------------------------------------------------------------------- #


def _digit_class() -> str:
    """Character class matching ASCII **and** Indic decimal digits.

    Built from the same table :mod:`app.stages.s5_validators` normalises with,
    so the masker and the validators cannot disagree about what a digit is.
    """
    characters = sorted({chr(code) for code in INDIC_DIGIT_TRANSLATION} | set("0123456789"))
    return "".join(re.escape(character) for character in characters)


_D: Final[str] = _digit_class()
_CURRENCY: Final[str] = r"(?:₹|Rs\.?|INR|रु\.?|रुपये|रुपए)"

#: Patterns whose matches are masked before MT, longest match winning.
#:
#: Deliberately wider than "whatever the validators found": a value the
#: validators skipped because it lacked a label cue is still a digit run a
#: sequence model will happily rewrite.  Over-masking costs a slightly stiffer
#: translation; under-masking costs a corrupted account number.
DEFAULT_PROTECT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # Currency-marked amount, with or without grouping.
    re.compile(rf"{_CURRENCY}\s*[{_D}](?:[{_D},.]*[{_D}])?", re.IGNORECASE),
    # IFSC code.
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{4}0[A-Za-z0-9]{6}(?![A-Za-z0-9])"),
    # Alphanumeric reference token (UTR/RRN and friends).
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,4}[0-9][A-Za-z0-9]{9,17}(?![A-Za-z0-9])"),
    # Date in Indian notation.
    re.compile(rf"(?<![{_D}])[{_D}]{{1,2}}[/\-.][{_D}]{{1,2}}[/\-.][{_D}]{{2,4}}(?![{_D}])"),
    # Any run of 6+ digits, separators allowed inside: accounts, cards, MICR,
    # RRNs, mobile numbers.
    re.compile(rf"(?<![{_D}])[{_D}][{_D} \-]{{4,}}[{_D}](?![{_D}])"),
    # Grouped number without a currency mark (``4,500`` / ``1,00,000``).
    re.compile(rf"(?<![{_D}])[{_D}]{{1,3}}(?:,[{_D}]{{2,3}})+(?:\.[{_D}]{{1,2}})?(?![{_D}])"),
    # Bare number with a decimal fraction.
    re.compile(rf"(?<![{_D}])[{_D}]+\.[{_D}]{{1,2}}(?![{_D}])"),
)


class MaskResult(BaseModel):
    """Masked text plus everything needed to put the originals back."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    #: Original substrings, indexed by sentinel number.
    tokens: tuple[str, ...] = ()
    #: Marker used for this text; :func:`restore_entities` needs it.
    marker: str = "E"

    @property
    def masked_count(self) -> int:
        """How many entities were protected."""
        return len(self.tokens)


def _collect_spans(
    text: str,
    patterns: Sequence[re.Pattern[str]],
    protected_values: Sequence[str],
) -> list[tuple[int, int]]:
    """Non-overlapping spans to mask: longest first, then leftmost."""
    candidates: list[tuple[int, int]] = []
    for value in protected_values:
        if not value:
            continue
        start = text.find(value)
        while start >= 0:
            candidates.append((start, start + len(value)))
            start = text.find(value, start + 1)
    for pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                candidates.append((start, end))
    candidates.sort(key=lambda span: (-(span[1] - span[0]), span[0]))
    chosen: list[tuple[int, int]] = []
    for start, end in candidates:
        if any(start < other_end and other_start < end for other_start, other_end in chosen):
            continue
        chosen.append((start, end))
    chosen.sort()
    return chosen


def mask_entities(
    text: str,
    *,
    protected_values: Sequence[str] = (),
    patterns: Sequence[re.Pattern[str]] = DEFAULT_PROTECT_PATTERNS,
) -> MaskResult:
    """Replace every identifier-shaped span with a numbered sentinel.

    ``protected_values`` are literal strings — normally the ``value`` of every
    :class:`~app.stages.s5_validators.ValidatorHit` that overlaps this text — and
    are masked in addition to the pattern matches, so a value the patterns would
    have missed is still protected.

    Guarantees, both property-tested:

    * ``restore_entities(mask_entities(t).text, ...)`` reproduces ``t``
      **byte-identically**;
    * sentinels are numbered left to right, so the masked text is a
      deterministic function of the input and a response cache is not
      fragmented.
    """
    if not text:
        return MaskResult(text=text, tokens=(), marker=_MARKERS[0])
    marker = _pick_marker(text)
    spans = _collect_spans(text, list(patterns), list(protected_values))
    if not spans:
        return MaskResult(text=text, tokens=(), marker=marker)
    pieces: list[str] = []
    tokens: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        pieces.append(_sentinel(marker, len(tokens)))
        tokens.append(text[start:end])
        cursor = end
    pieces.append(text[cursor:])
    return MaskResult(text="".join(pieces), tokens=tuple(tokens), marker=marker)


def restore_entities(text: str, mask: MaskResult) -> str:
    """Put the original substrings back, byte for byte.

    Raises:
        SentinelRestoreFailed: when a sentinel is missing from ``text``.  The
            caller must then discard the translation — there is no partial
            restore, because a translation missing one identifier reads exactly
            as trustworthy as one missing none.
    """
    restored = text
    for index, original in enumerate(mask.tokens):
        pattern = _restore_pattern(mask.marker, index)
        restored, replacements = pattern.subn(_constant_replacer(original), restored)
        if replacements == 0:
            raise SentinelRestoreFailed(
                f"translation lost sentinel {_sentinel(mask.marker, index)}; "
                "the protected value cannot be restored and the translation is "
                "discarded (PLAN.md invariant 10)",
                context={
                    "sentinel_index": index,
                    "sentinel_count": len(mask.tokens),
                },
            )
    return restored


# --------------------------------------------------------------------------- #
# Field-level translation
# --------------------------------------------------------------------------- #

#: Fields that can carry an advisory ``value_en``.  Identifier fields are absent
#: on purpose: "BARB0417293" has no English rendering, and offering one would
#: invite a consumer to read ``value_en`` where only ``value`` is authoritative.
TRANSLATABLE_FIELDS: Final[frozenset[str]] = frozenset(
    (SLM_ELIGIBLE_FIELDS - {SUMMARY_FIELD_NAME}) | {"bank_name"}
)

#: Language code meaning "already the target"; never sent to an engine.
_TARGET_LANGUAGE: Final[str] = "eng"


def _has_translatable_content(mask: MaskResult) -> bool:
    """True when anything but sentinels and punctuation survives masking."""
    stripped = mask.text
    for index in range(len(mask.tokens)):
        stripped = _restore_pattern(mask.marker, index).sub(" ", stripped)
    return any(character.isalpha() for character in stripped)


def translate_text(
    provider: SupportsTranslation,
    texts: Sequence[str],
    *,
    src: str,
    tgt: str = _TARGET_LANGUAGE,
    protected_values: Sequence[str] = (),
) -> list[str | None]:
    """Mask, translate and restore a batch of strings, position-preserving.

    An entry whose sentinels did not survive comes back as ``None`` rather than
    a mangled string — the caller decides what a missing translation means for
    its field.
    """
    masks = [
        mask_entities(text, protected_values=protected_values) for text in texts
    ]
    translated = provider.translate([mask.text for mask in masks], src, tgt)
    if len(translated) != len(masks):
        raise ValueError(
            f"translate() returned {len(translated)} strings for {len(masks)} "
            "inputs; positions must be preserved"
        )
    output: list[str | None] = []
    for mask, candidate in zip(masks, translated, strict=True):
        try:
            output.append(restore_entities(candidate, mask))
        except SentinelRestoreFailed as failure:
            _LOG.warning("discarding translation: %s", failure.message)
            output.append(None)
    return output


def translate_fields(
    provider: SupportsTranslation,
    fields: Mapping[str, Field],
    *,
    src_language: str,
    mt_engine: str,
    target_language: str = _TARGET_LANGUAGE,
    protected_values: Sequence[str] = (),
    translatable: Iterable[str] = TRANSLATABLE_FIELDS,
    strict: bool = False,
) -> dict[str, Field]:
    """Attach an advisory ``value_en`` to every free-text field.

    ``value`` is copied through untouched — :meth:`Field.with_translation`
    revalidates the whole object, so there is no code path on which translation
    can alter the native string (invariant 10).

    Fields are skipped, not failed, when:

    * the field is not in ``translatable`` (identifiers, summaries);
    * ``src_language`` is already the target — an English complaint does not
      need an advisory English rendering, and producing one would put an
      ``mt_engine`` tag on text no engine touched;
    * masking leaves nothing but sentinels, i.e. the value is pure identifier;
    * a translation already exists.

    Args:
        protected_values: literal strings to mask in addition to the pattern
            matches — pass the validator hits' values for this document.
        strict: re-raise a provider failure instead of flagging the fields.
            Off by default: translation is advisory and a missing MT engine must
            not cost a document its extracted identifiers.

    Raises:
        ValueError: for an empty ``mt_engine`` — invariant 10 makes the tag
            mandatory, and a blank one is worse than none.
    """
    if not mt_engine.strip():
        raise ValueError(
            "mt_engine is mandatory: value_en is advisory output and must name "
            "the engine that produced it (PLAN.md invariant 10)"
        )
    eligible = set(translatable)
    result = dict(fields)
    if src_language == target_language:
        return result

    names = [
        name
        for name, field in fields.items()
        if name in eligible
        and field.value
        and field.value_en is None
    ]
    if not names:
        return result

    masks = {
        name: mask_entities(
            fields[name].value or "", protected_values=protected_values
        )
        for name in names
    }
    names = [name for name in names if _has_translatable_content(masks[name])]
    if not names:
        return result

    try:
        translated = provider.translate(
            [masks[name].text for name in names], src_language, target_language
        )
    except ProviderError as failure:
        if strict:
            raise
        _LOG.warning(
            "translation unavailable (%s); fields keep native values only",
            failure.code,
        )
        for name in names:
            result[name] = fields[name].with_flags(
                FieldFlag.TRANSLATION_UNAVAILABLE
            )
        return result

    if len(translated) != len(names):
        raise ValueError(
            f"translate() returned {len(translated)} strings for {len(names)} "
            "fields; positions must be preserved"
        )

    for name, candidate in zip(names, translated, strict=True):
        try:
            restored = restore_entities(candidate, masks[name])
        except SentinelRestoreFailed as failure:
            _LOG.warning(
                "field %s: %s", name, failure.message
            )
            result[name] = fields[name].with_flags(
                FieldFlag.TRANSLATION_UNAVAILABLE
            )
            continue
        result[name] = fields[name].with_translation(restored, mt_engine)
    return result
