"""PII redaction at the logging **filter** — BUILD-BRIEF sec. 5, never at the call site.

The brief is specific about the placement and it is the whole point: a rule
applied at each call site is a rule that holds until someone adds a call site.
A filter attached to the root logger sees every record from every library —
uvicorn's access log, SQLAlchemy's echo, a third-party client printing a request
body — and none of those call sites will ever be reviewed by us.

What it does, in order:

1. **Refuses whole records above a level.**  Anything logged at ``INFO`` or
   below whose message carries recognised complainant data is dropped, not
   masked: "never log raw OCR text at INFO" is a rule about the *record*, and a
   half-masked line still tells an analyst which document had a 14-digit
   account number on it.  At ``WARNING`` and above the record is kept and
   masked, because losing an error is worse than storing a masked identifier.
2. **Masks identifiers** it can recognise structurally — account numbers, IFSC,
   card PANs, UTR/RRN, MICR, amounts with a currency mark, e-mail addresses,
   Indian phone numbers — leaving enough shape to correlate two log lines
   (``[ACCOUNT:…4821]``) without publishing the value.
3. **Masks any Indic-script run.**  Devanagari in a log line is, in this
   service, complainant text: the only Devanagari the service handles is the
   content of a complaint.  There is no pattern for "a name", so the script
   itself is the signal.

What it deliberately does **not** do:

* it does not try to redact structured ``extra`` payloads recursively — a
  filter that walks arbitrary objects is a filter that raises inside logging;
  the convention is that ``extra`` carries counters and identifiers of *ours*
  (job id, page number, error code), and :class:`PipelineError.context` is
  documented the same way;
* it does not promise completeness.  It is the last line of defence, not the
  first: the first is that no stage passes OCR text to a logger at all.

This module does NOT handle: log formatting, handlers, sinks, sampling, or
shipping.  :func:`install_redaction` attaches the filter and returns it; the
deployment owns everything else.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final

__all__ = [
    "REDACTION_PATTERNS",
    "PIIRedactionFilter",
    "install_redaction",
    "redact",
]


def _pattern(expression: str) -> re.Pattern[str]:
    return re.compile(expression)


#: ``(name, pattern, replacement)``.  Order matters: the longest and most
#: specific structures match first so a card PAN is not eaten by the generic
#: long-digit-run rule.
REDACTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = (
    (
        "ifsc",
        _pattern(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
        "[IFSC]",
    ),
    (
        "card",
        _pattern(r"\b(?:\d[ -]?){12,18}\d\b"),
        "[CARD_OR_ACCOUNT]",
    ),
    (
        "account",
        _pattern(r"\b\d{9,17}\b"),
        "[ACCOUNT]",
    ),
    (
        "utr_rrn",
        _pattern(r"\b[A-Z]{2,6}\d{8,20}\b"),
        "[REFERENCE]",
    ),
    (
        "email",
        _pattern(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "[EMAIL]",
    ),
    (
        "phone",
        _pattern(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)"),
        "[PHONE]",
    ),
    (
        "amount",
        _pattern(r"(?:₹|Rs\.?|INR)\s?[\d,.०-९]+"),
        "[AMOUNT]",
    ),
    (
        "indic",
        # Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam,
        # Odia, Gurmukhi — every script this service reads from a complaint.
        _pattern(
            r"[ऀ-ॿঀ-৿਀-੿઀-૿"
            r"଀-୿஀-௿ఀ-౿ಀ-೿"
            r"ഀ-ൿ]+"
        ),
        "[INDIC_TEXT]",
    ),
)


def redact(message: str) -> tuple[str, tuple[str, ...]]:
    """Mask every recognised identifier; report which rules fired.

    Returns ``(masked_message, rule_names)``.  The rule names are what
    :class:`PIIRedactionFilter` uses to decide whether an ``INFO`` record may
    survive at all, and they are useful on their own: a rising ``indic`` count
    means a stage started logging OCR text.
    """
    fired: list[str] = []
    result = message
    for name, pattern, replacement in REDACTION_PATTERNS:
        result, count = pattern.subn(replacement, result)
        if count:
            fired.append(name)
    return result, tuple(fired)


class PIIRedactionFilter(logging.Filter):
    """Masks complainant data on every record, and drops the worst at INFO.

    Attach once, to the **root** logger, at application creation.  It is a
    filter rather than a formatter so it applies regardless of which handler or
    format a deployment configures, and it mutates ``record.msg`` after
    rendering the arguments so a lazy ``%s`` argument cannot slip past.
    """

    def __init__(
        self,
        name: str = "",
        *,
        drop_at_or_below: int = logging.INFO,
        drop_rules: frozenset[str] = frozenset({"indic", "account", "card", "ifsc"}),
    ) -> None:
        super().__init__(name)
        self.drop_at_or_below = drop_at_or_below
        self.drop_rules = drop_rules
        #: Counters a test (and ``/metrics``) can read without parsing logs.
        self.redactions: dict[str, int] = {}
        self.dropped: int = 0

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` to drop the record, ``True`` to keep it (masked)."""
        try:
            rendered = record.getMessage()
        except (TypeError, ValueError):
            # A malformed format string is the caller's bug, not ours to hide,
            # but it must not take the process down from inside logging.
            return True
        masked, fired = redact(rendered)
        if not fired:
            return True
        for rule in fired:
            self.redactions[rule] = self.redactions.get(rule, 0) + 1
        if record.levelno <= self.drop_at_or_below and self.drop_rules.intersection(
            fired
        ):
            self.dropped += 1
            return False
        record.msg = masked
        record.args = ()
        return True


def install_redaction(
    logger: logging.Logger | None = None, **kwargs: Any
) -> PIIRedactionFilter:
    """Attach the filter to ``logger`` **and to its handlers**, once.

    Both, because Python's two filter positions do different jobs and only the
    pair covers everything:

    * a filter on a *logger* sees records logged through that logger only —
      records propagated up from ``app.stages.s4_ocr`` never touch the root
      logger's own filters;
    * a filter on a *handler* sees every record that reaches that handler,
      including propagated ones, which is the coverage the brief asks for.

    Handlers added *after* this call are not covered; pass them to
    :func:`attach_to_handler`.  ``uvicorn`` installs its own handlers on its own
    loggers, so :func:`app.api.app.create_app` calls this again for those.

    Idempotent: an already-attached filter is reused rather than stacked, so a
    test that builds several apps does not double-count every redaction.
    """
    target = logger if logger is not None else logging.getLogger()
    installed: PIIRedactionFilter | None = next(
        (f for f in target.filters if isinstance(f, PIIRedactionFilter)), None
    )
    if installed is None:
        installed = PIIRedactionFilter(**kwargs)
        target.addFilter(installed)
    for handler in target.handlers:
        attach_to_handler(handler, installed)
    return installed


def attach_to_handler(
    handler: logging.Handler, redactor: PIIRedactionFilter
) -> None:
    """Put ``redactor`` on ``handler`` if it is not already there."""
    if not any(existing is redactor for existing in handler.filters):
        handler.addFilter(redactor)
