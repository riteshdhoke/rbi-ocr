"""Complainant-facing auto-replies — PRD P1-2, PLAN.md WS-G.

The service refuses documents it cannot read (PLAN.md invariant 6 lineage:
BUILD-BRIEF M3 quality gate, ``BounceDecision``).  A refusal that only produces
a code in a JSON envelope is invisible to the person who sent the document, so
this package turns every :class:`~app.contracts.envelope.BounceReason` into a
bilingual message that **names the specific defect and the remedy** — "the
attachment is password protected; please remove the password and send it
again", not "processing failed".

Import :func:`~app.replies.bounce.render_bounce_reply` for one language,
:func:`~app.replies.bounce.render_bilingual_bounce_reply` for the English +
Hindi pair, or :func:`~app.replies.bounce.reply_for_envelope` to go straight
from a bounced :class:`~app.contracts.envelope.ResultEnvelope`.

This package does NOT handle:
  * **sending anything.**  It renders text.  No SMTP, no queue, no address book
    — and PRD open question 3 ("is a bounce/auto-reply to the complainant
    permissible under RB-IOS process rules?") is still unanswered, so a
    delivery path built on top of this must stay behind an explicit switch;
  * deciding *whether* a document bounced — that is the quality gate and the
    ingestion stage;
  * translation.  The Hindi text is written, reviewed and committed here; it is
    never produced by the MT engine, because a machine-translated apology is
    exactly the artefact nobody can be held to;
  * malware (:class:`~app.contracts.errors.MalwareDetected` is deliberately not
    a ``BounceReason`` — telling a sender which signature fired is a free
    oracle, see that class's docstring);
  * logging.  A rendered reply contains the sender's own filename and reference
    and must not be logged at INFO.
"""

from __future__ import annotations

from app.replies.bounce import (
    DEFAULT_LANGUAGES,
    SUBJECT_BODY_SEPARATOR,
    TEMPLATES_DIR,
    BilingualBounceReply,
    BounceReply,
    ReplyContext,
    ReplyLanguage,
    ReplyTemplateInvalid,
    ReplyTemplateMissing,
    available_reasons,
    missing_templates,
    render_bilingual_bounce_reply,
    render_bounce_reply,
    reply_for_envelope,
    template_name,
)

__all__ = [
    "DEFAULT_LANGUAGES",
    "SUBJECT_BODY_SEPARATOR",
    "TEMPLATES_DIR",
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
