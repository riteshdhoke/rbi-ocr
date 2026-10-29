"""Rendering of bilingual bounce auto-replies (PRD P1-2).

One Jinja2 template per (:class:`~app.contracts.envelope.BounceReason`,
:class:`ReplyLanguage`) pair, under ``app/replies/templates/<lang>/<CODE>.txt.j2``.
The code is the filename, so the lookup is total by construction and
:func:`missing_templates` turns "somebody added a bounce reason and forgot the
message" into a failing test rather than a silent 500 in the mail path.

Three deliberate choices, each recorded in ``docs/decisions-wsg.md``:

**Plain text only.**  ``autoescape`` is off because these templates render
``text/plain`` bodies, where HTML escaping would corrupt an ampersand in a
complainant's own filename.  There is no HTML template; if one is ever added it
must be loaded through a *separate* environment with autoescaping on.

**Strict undefined.**  A missing context variable raises
:class:`ReplyTemplateInvalid` instead of rendering an empty gap.  Combined with
the post-render scan for surviving ``{{``/``{%`` markers, "no unfilled
placeholder ever reaches a complainant" is enforced at runtime, not just tested.

**``BounceDecision.detail`` is not rendered.**  That field is free text written
by whichever stage bounced the page, for engineers.  Piping it into a message a
member of the public receives is how internal paths, thresholds and stage names
leak.  The *code* selects the wording; the wording is committed and reviewed.

This module does NOT handle:
  * delivery, addressing, threading, or the ``Reply-To`` a mail system needs;
  * choosing the language from the document — the caller decides, and the
    default is to send **both**, because a document we could not read is also a
    document whose language we could not identify;
  * per-tenant branding.  The signature block is literal text in each layout,
    and a deployment is expected to review both layouts before going live;
  * escaping or sanitising ``attachment_name``.  It is the sender's own
    filename echoed back to them in a plain-text body; a mail layer that
    renders replies as HTML must escape it itself.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Final, Self

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2 import TemplateError as _JinjaTemplateError
from jinja2 import TemplateNotFound as _JinjaTemplateNotFound
from prometheus_client import Counter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts.envelope import BounceReason, QualityVerdict, ResultEnvelope
from app.contracts.errors import PipelineError

__all__ = [
    "DEFAULT_LANGUAGES",
    "REPLIES_DIR",
    "SUBJECT_BODY_SEPARATOR",
    "TEMPLATES_DIR",
    "TEMPLATE_SUFFIX",
    "BilingualBounceReply",
    "BounceReply",
    "ReplyContext",
    "ReplyLanguage",
    "ReplyTemplateInvalid",
    "ReplyTemplateMissing",
    "available_reasons",
    "missing_templates",
    "render_bilingual_bounce_reply",
    "render_bounce_reply",
    "reply_for_envelope",
    "template_name",
]

_LOG: Final = logging.getLogger(__name__)

#: This package's directory, resolved from ``__file__`` so the CWD never
#: matters (PLAN.md working agreements).
REPLIES_DIR: Final[Path] = Path(__file__).resolve().parent
TEMPLATES_DIR: Final[Path] = REPLIES_DIR / "templates"

TEMPLATE_SUFFIX: Final[str] = ".txt.j2"
LAYOUT_TEMPLATE_NAME: Final[str] = "_layout" + TEMPLATE_SUFFIX

#: A line of exactly this text separates the subject from the body.  Three
#: hyphens rather than a header-style ``Subject:`` prefix so that no reply text
#: can ever be mistaken for a header by a downstream mail composer.
SUBJECT_BODY_SEPARATOR: Final[str] = "---"

#: Anything left of a Jinja construct after rendering means a template bug.
_PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = ("{{", "}}", "{%", "%}")

_REPLIES_RENDERED = Counter(
    "replies_bounce_rendered_total",
    "Bounce auto-replies rendered, by reason and language.",
    ["reason", "language"],
)


class ReplyLanguage(StrEnum):
    """Languages a bounce reply is written in.

    Values are **ISO 639-3**, matching
    :data:`app.contracts.ocr.LanguageCode` everywhere else in the service, and
    each is also the name of its template directory.  Marathi is not here: the
    corpus contains it, but nobody on this build has reviewed Marathi prose for
    tone, and an unreviewed apology in a language you do not read is worse than
    the English one the recipient can at least escalate.
    """

    ENGLISH = "eng"
    HINDI = "hin"

    @property
    def directory(self) -> str:
        """Template sub-directory for this language."""
        return self.value

    @classmethod
    def coerce(cls, value: ReplyLanguage | str) -> ReplyLanguage:
        """Accept the enum, an ISO 639-3 code, or the common 2-letter tag.

        Raises:
            ValueError: for anything else. There is no "closest match"; a
                language we have not written and reviewed is not a language we
                send.
        """
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower().replace("_", "-")
        text = text.split("-", 1)[0]
        aliases = {"en": cls.ENGLISH, "eng": cls.ENGLISH, "hi": cls.HINDI, "hin": cls.HINDI}
        try:
            return aliases[text]
        except KeyError:
            raise ValueError(
                f"no reviewed bounce-reply template language for {value!r}; "
                f"available: {', '.join(member.value for member in cls)}"
            ) from None


#: English first, Hindi second — the order the combined body is assembled in.
DEFAULT_LANGUAGES: Final[tuple[ReplyLanguage, ...]] = (
    ReplyLanguage.ENGLISH,
    ReplyLanguage.HINDI,
)


# --------------------------------------------------------------------------- #
# Typed errors owned by this package
# --------------------------------------------------------------------------- #


class ReplyTemplateMissing(PipelineError):
    """No committed template exists for this reason/language pair.

    Lives here rather than in :mod:`app.contracts.errors` for the same reason
    :class:`app.providers.base.ModelArtifactMissing` does: the contracts module
    is a shared lane, and an error that only this package can raise belongs
    next to the code that raises it.
    """

    code: ClassVar[str] = "REPLY_TEMPLATE_MISSING"
    http_status: ClassVar[int] = 500


class ReplyTemplateInvalid(PipelineError):
    """A template rendered, but the result cannot be sent to a person.

    Raised for a missing context variable (``StrictUndefined``), a missing or
    duplicated subject/body separator, an empty subject or body, a multi-line
    subject, and for any Jinja marker surviving into the output.
    """

    code: ClassVar[str] = "REPLY_TEMPLATE_INVALID"
    http_status: ClassVar[int] = 500


# --------------------------------------------------------------------------- #
# Context and results
# --------------------------------------------------------------------------- #


class ReplyContext(BaseModel):
    """Everything a template may interpolate.

    Small on purpose.  Every field here ends up in a message a member of the
    public reads, so the set is closed (``extra="forbid"``) and each addition
    is a deliberate decision about what we are willing to say back.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: What the complainant should quote when they reply.  Required: a bounce
    #: without a handle back into the case is a dead end.
    reference_id: str = Field(min_length=1)
    #: The sender's own filename, echoed so they know *which* attachment.
    #: ``None`` omits the line entirely rather than printing "unknown".
    attachment_name: str | None = Field(default=None, min_length=1)
    #: 1-based pages the defect was seen on. Empty for document-level bounces.
    page_numbers: tuple[int, ...] = ()
    #: Quoted by the page-cap template; mirrors ``Settings.MAX_PAGES_PER_DOC``.
    max_pages: int = Field(default=200, ge=1)
    #: The scanning resolution the remedy asks for. 300 DPI is the floor at
    #: which Devanagari conjuncts survive recognition.
    recommended_dpi: int = Field(default=300, ge=72)

    @model_validator(mode="after")
    def _pages_are_sane(self) -> Self:
        for page_no in self.page_numbers:
            if page_no < 1:
                raise ValueError(
                    f"page_numbers are 1-based; got {page_no} "
                    "(a complainant counts pages from one)"
                )
        if len(set(self.page_numbers)) != len(self.page_numbers):
            raise ValueError(f"duplicate page numbers: {self.page_numbers}")
        if "\n" in self.reference_id or "\r" in self.reference_id:
            raise ValueError("reference_id must be a single line")
        if self.attachment_name is not None and (
            "\n" in self.attachment_name or "\r" in self.attachment_name
        ):
            raise ValueError("attachment_name must be a single line")
        return self


class BounceReply(BaseModel):
    """One rendered reply: subject, body, and what produced them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: BounceReason
    language: ReplyLanguage
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    #: Loader-relative template path, recorded so an audit can point at the
    #: exact committed wording a complainant received.
    template: str = Field(min_length=1)

    def as_text(self) -> str:
        """``Subject``/blank line/body — the shape a plain-text mailer wants."""
        return f"{self.subject}\n\n{self.body}"


class BilingualBounceReply(BaseModel):
    """The same bounce in every reviewed language, plus a combined message.

    A document we could not read is a document whose language we could not
    identify, so the default is to send both and let the reader choose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: BounceReason
    replies: tuple[BounceReply, ...] = Field(min_length=1)
    #: Subjects joined by ``" / "`` in the order the languages were requested.
    subject: str = Field(min_length=1)
    #: Bodies joined by a horizontal rule, same order.
    body: str = Field(min_length=1)

    @model_validator(mode="after")
    def _one_reply_per_language(self) -> Self:
        languages = [reply.language for reply in self.replies]
        if len(set(languages)) != len(languages):
            raise ValueError(f"duplicate languages in a bilingual reply: {languages}")
        for reply in self.replies:
            if reply.reason is not self.reason:
                raise ValueError(
                    f"reply for {reply.reason.value!r} in a bilingual reply for "
                    f"{self.reason.value!r}"
                )
        return self

    def for_language(self, language: ReplyLanguage | str) -> BounceReply:
        """The single-language reply.

        Raises:
            KeyError: when that language was not rendered.
        """
        wanted = ReplyLanguage.coerce(language)
        for reply in self.replies:
            if reply.language is wanted:
                return reply
        raise KeyError(
            f"{wanted.value!r} was not rendered; have "
            f"{[reply.language.value for reply in self.replies]}"
        )

    def as_text(self) -> str:
        """``Subject``/blank line/combined body."""
        return f"{self.subject}\n\n{self.body}"


#: Separates the language sections of a combined body.  Plain ASCII so it
#: survives every transfer encoding a mail gateway might choose.
_LANGUAGE_RULE: Final[str] = "\n\n" + "-" * 60 + "\n\n"


# --------------------------------------------------------------------------- #
# Template plumbing
# --------------------------------------------------------------------------- #


def template_name(reason: BounceReason, language: ReplyLanguage) -> str:
    """Loader-relative template path for a reason/language pair.

    Always ``/``-separated: :class:`jinja2.FileSystemLoader` uses POSIX
    separators on every platform, Windows included.
    """
    return f"{language.directory}/{reason.value}{TEMPLATE_SUFFIX}"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    """The one Jinja environment, built once per process.

    ``autoescape=False`` is deliberate and safe here: every template in this
    directory renders ``text/plain``.  See the module docstring.
    """
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR), encoding="utf-8"),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
        auto_reload=False,
    )


def _coerce_reason(reason: BounceReason | str) -> BounceReason:
    """Accept the enum or its wire value."""
    if isinstance(reason, BounceReason):
        return reason
    try:
        return BounceReason(str(reason))
    except ValueError:
        raise ReplyTemplateMissing(
            f"{reason!r} is not a BounceReason, so no auto-reply is written "
            "for it; the vocabulary is app.contracts.envelope.BounceReason",
            context={"reason": str(reason)},
        ) from None


def _split_subject_and_body(rendered: str, *, template: str) -> tuple[str, str]:
    """Split a rendered template on its lone separator line.

    Raises:
        ReplyTemplateInvalid: when the separator is missing or repeated, when
            either half is blank, or when the subject spans lines.
    """
    lines = rendered.replace("\r\n", "\n").split("\n")
    positions = [
        index for index, line in enumerate(lines) if line.strip() == SUBJECT_BODY_SEPARATOR
    ]
    if len(positions) != 1:
        raise ReplyTemplateInvalid(
            f"template {template!r} must contain exactly one "
            f"{SUBJECT_BODY_SEPARATOR!r} line separating subject from body; "
            f"found {len(positions)}",
            context={"template": template, "separators": len(positions)},
        )
    subject = "\n".join(lines[: positions[0]]).strip()
    body = "\n".join(lines[positions[0] + 1 :]).strip()
    if not subject:
        raise ReplyTemplateInvalid(
            f"template {template!r} rendered an empty subject",
            context={"template": template},
        )
    if "\n" in subject:
        raise ReplyTemplateInvalid(
            f"template {template!r} rendered a multi-line subject: {subject!r}",
            context={"template": template},
        )
    if not body:
        raise ReplyTemplateInvalid(
            f"template {template!r} rendered an empty body",
            context={"template": template},
        )
    return subject, body


def _assert_no_placeholders(text: str, *, template: str) -> None:
    """Refuse to hand back output with a surviving Jinja marker.

    ``StrictUndefined`` already catches an undefined *variable*; this catches
    the other half — a mistyped ``{{ reference_id }`` that Jinja treats as
    literal text and would otherwise mail to a complainant verbatim.
    """
    for marker in _PLACEHOLDER_MARKERS:
        if marker in text:
            raise ReplyTemplateInvalid(
                f"template {template!r} left an unrendered {marker!r} in its "
                "output; a complainant must never receive template syntax",
                context={"template": template, "marker": marker},
            )


# --------------------------------------------------------------------------- #
# Public rendering API
# --------------------------------------------------------------------------- #


def render_bounce_reply(
    reason: BounceReason | str,
    context: ReplyContext,
    *,
    language: ReplyLanguage | str = ReplyLanguage.ENGLISH,
) -> BounceReply:
    """Render the reply for one bounce reason in one language.

    Args:
        reason: the machine-readable code the quality gate or ingestion stage
            recorded on :class:`~app.contracts.envelope.BounceDecision`.
        context: the closed set of values a template may interpolate.
        language: ISO 639-3 (``eng``/``hin``), the enum, or ``en``/``hi``.

    Raises:
        ReplyTemplateMissing: no committed template for that pair — a build
            bug, and one :func:`missing_templates` catches in CI.
        ReplyTemplateInvalid: the template rendered something unsendable (an
            undefined variable, no separator, an empty half, a surviving
            placeholder).
    """
    bounce_reason = _coerce_reason(reason)
    reply_language = ReplyLanguage.coerce(language)
    name = template_name(bounce_reason, reply_language)

    try:
        template = _environment().get_template(name)
    except _JinjaTemplateNotFound as exc:
        raise ReplyTemplateMissing(
            f"no {reply_language.value} auto-reply template for bounce reason "
            f"{bounce_reason.value}; expected {TEMPLATES_DIR / name}",
            context={
                "reason": bounce_reason.value,
                "language": reply_language.value,
                "template": name,
            },
        ) from exc
    except _JinjaTemplateError as exc:
        # A syntax error surfaces here, at compile time, not at render time.
        raise ReplyTemplateInvalid(
            f"template {name!r} does not compile: {exc}",
            context={
                "reason": bounce_reason.value,
                "language": reply_language.value,
                "template": name,
            },
        ) from exc

    try:
        rendered = template.render(context.model_dump())
    except _JinjaTemplateError as exc:
        raise ReplyTemplateInvalid(
            f"template {name!r} could not be rendered: {exc}",
            context={
                "reason": bounce_reason.value,
                "language": reply_language.value,
                "template": name,
            },
        ) from exc

    _assert_no_placeholders(rendered, template=name)
    subject, body = _split_subject_and_body(rendered, template=name)

    _REPLIES_RENDERED.labels(
        reason=bounce_reason.value, language=reply_language.value
    ).inc()
    _LOG.debug(
        "rendered bounce reply reason=%s language=%s template=%s",
        bounce_reason.value,
        reply_language.value,
        name,
    )
    return BounceReply(
        reason=bounce_reason,
        language=reply_language,
        subject=subject,
        body=body,
        template=name,
    )


def render_bilingual_bounce_reply(
    reason: BounceReason | str,
    context: ReplyContext,
    *,
    languages: Sequence[ReplyLanguage | str] = DEFAULT_LANGUAGES,
) -> BilingualBounceReply:
    """Render one bounce in every requested language and combine them.

    The combined subject joins the per-language subjects with ``" / "``; the
    combined body stacks the per-language bodies, separated by a rule, in the
    order given.

    Raises:
        ValueError: when ``languages`` is empty or repeats a language.
        ReplyTemplateMissing / ReplyTemplateInvalid: as
            :func:`render_bounce_reply`.
    """
    if not languages:
        raise ValueError("a bilingual reply needs at least one language")
    resolved = [ReplyLanguage.coerce(language) for language in languages]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate languages requested: {list(languages)}")

    bounce_reason = _coerce_reason(reason)
    replies = tuple(
        render_bounce_reply(bounce_reason, context, language=language)
        for language in resolved
    )
    return BilingualBounceReply(
        reason=bounce_reason,
        replies=replies,
        subject=" / ".join(reply.subject for reply in replies),
        body=_LANGUAGE_RULE.join(reply.body for reply in replies),
    )


def reply_for_envelope(
    envelope: ResultEnvelope,
    *,
    languages: Sequence[ReplyLanguage | str] = DEFAULT_LANGUAGES,
    attachment_name: str | None = None,
    max_pages: int | None = None,
    recommended_dpi: int | None = None,
) -> BilingualBounceReply | None:
    """The reply for a bounced envelope, or ``None`` when it was accepted.

    ``reference_id`` prefers ``job_id`` over ``document_id``: the document id
    is a content hash, and asking a member of the public to quote 64 hex
    characters is asking them not to reply.

    ``page_numbers`` is derived from the pages that actually bounced, so a
    50-page document whose page 31 was unreadable says "page 31" instead of
    making the sender rescan everything.

    Note what is *not* read: ``bounce_decision.detail``. See the module
    docstring.
    """
    decision = envelope.bounce_decision
    if decision.decision is not QualityVerdict.BOUNCE or decision.reason is None:
        return None

    bounced_pages = tuple(
        page.page_no
        for page in envelope.pages
        if page.quality_verdict is QualityVerdict.BOUNCE
    )
    overrides: dict[str, object] = {}
    if max_pages is not None:
        overrides["max_pages"] = max_pages
    if recommended_dpi is not None:
        overrides["recommended_dpi"] = recommended_dpi

    context = ReplyContext(
        reference_id=envelope.job_id or envelope.document_id,
        attachment_name=attachment_name,
        page_numbers=bounced_pages,
        **overrides,  # type: ignore[arg-type]
    )
    return render_bilingual_bounce_reply(
        decision.reason, context, languages=languages
    )


# --------------------------------------------------------------------------- #
# Coverage introspection — what CI asserts on
# --------------------------------------------------------------------------- #

_CODE_FILENAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def available_reasons(language: ReplyLanguage | str) -> frozenset[BounceReason]:
    """Bounce reasons that have a committed template in ``language``.

    Reads the directory rather than a hard-coded list, so a template added
    without touching this module is picked up and a template *deleted* is
    noticed.
    """
    directory = TEMPLATES_DIR / ReplyLanguage.coerce(language).directory
    if not directory.is_dir():
        return frozenset()
    found: set[BounceReason] = set()
    for path in directory.iterdir():
        if not path.is_file() or not path.name.endswith(TEMPLATE_SUFFIX):
            continue
        stem = path.name[: -len(TEMPLATE_SUFFIX)]
        if not _CODE_FILENAME.match(stem):
            continue  # ``_layout.txt.j2`` and friends
        try:
            found.add(BounceReason(stem))
        except ValueError:
            _LOG.warning(
                "template %s names %r, which is not a BounceReason; ignored",
                path.name,
                stem,
            )
    return frozenset(found)


def missing_templates(
    languages: Iterable[ReplyLanguage | str] = DEFAULT_LANGUAGES,
) -> tuple[tuple[BounceReason, ReplyLanguage], ...]:
    """Every (reason, language) pair with no committed template.

    Empty is the only acceptable value; ``tests/unit/test_replies.py`` asserts
    it.  A bounce reason with no message is a document refused in silence.
    """
    gaps: list[tuple[BounceReason, ReplyLanguage]] = []
    for language in languages:
        resolved = ReplyLanguage.coerce(language)
        have = available_reasons(resolved)
        for reason in BounceReason:
            if reason not in have:
                gaps.append((reason, resolved))
    return tuple(gaps)
