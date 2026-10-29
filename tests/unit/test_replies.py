"""Bilingual bounce auto-replies (PRD P1-2, WS-G).

Every :class:`BounceReason` is rendered in every reviewed language and the
result is checked for the things that make a message sendable to a member of
the public: no template syntax survived, the defect is named, the remedy is
named, the reference is quotable, and none of the vocabulary blames the sender.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from app.contracts.envelope import (
    BounceDecision,
    BounceReason,
    PageResult,
    QualityVerdict,
    ResultEnvelope,
)
from app.contracts.ocr import OCRTier
from app.replies import bounce as bounce_module
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

ALL_REASONS = tuple(BounceReason)
ALL_LANGUAGES = tuple(ReplyLanguage)

DEVANAGARI = range(0x0900, 0x0980)

#: Words that blame, scold or read as a machine error report.  A bounce is the
#: service asking for help, not the service telling somebody off.
DISCOURAGED_ENGLISH = (
    "invalid",
    "illegal",
    "error",
    "failure",
    "failed",
    "reject",
    "violation",
    "unacceptable",
    "corrupt",
    "you must",
    "your fault",
    "wrong of you",
    "as per",
)
DISCOURAGED_HINDI = (
    "अमान्य",
    "त्रुटि",
    "गलती",
    "अस्वीकार",
    "असफल",
    "दोषी",
)

#: Codes whose remedy is "rescan at 300 DPI" and must therefore quote the DPI.
DPI_REMEDY_REASONS = (
    BounceReason.RESOLUTION_BELOW_FLOOR,
    BounceReason.BLUR_ABOVE_CEILING,
    BounceReason.SKEW_UNRECOVERABLE,
    BounceReason.TEXT_COVERAGE_BELOW_FLOOR,
    BounceReason.OCCLUSION_DETECTED,
    BounceReason.QUALITY_SCORE_BELOW_THRESHOLD,
    BounceReason.BLANK_PAGE,
    BounceReason.ALL_PAGES_BOUNCED,
    BounceReason.NO_TEXT_DETECTED,
    BounceReason.PIXEL_CAP_EXCEEDED,
    BounceReason.DECOMPRESSION_BOMB,
    BounceReason.WALL_CLOCK_EXCEEDED,
)


@pytest.fixture
def context() -> ReplyContext:
    """A fully populated context — every optional line present."""
    return ReplyContext(
        reference_id="JOB-2026-000412",
        attachment_name="complaint scan.pdf",
        page_numbers=(2, 3),
    )


@pytest.fixture
def bare_context() -> ReplyContext:
    """Only the required field, so the optional lines must disappear cleanly."""
    return ReplyContext(reference_id="JOB-1")


def _render_all(context: ReplyContext) -> list[BounceReply]:
    return [
        render_bounce_reply(reason, context, language=language)
        for reason in ALL_REASONS
        for language in ALL_LANGUAGES
    ]


# --------------------------------------------------------------------------- #
# Coverage: no bounce reason may be refused in silence
# --------------------------------------------------------------------------- #


def test_every_bounce_reason_has_a_template_in_every_language() -> None:
    assert missing_templates(ALL_LANGUAGES) == ()


def test_available_reasons_matches_the_enum() -> None:
    for language in ALL_LANGUAGES:
        assert available_reasons(language) == frozenset(ALL_REASONS)


def test_the_enum_has_not_grown_without_a_message() -> None:
    """A guard on the count, so adding a reason lands here loudly."""
    assert len(ALL_REASONS) == 17


def test_template_files_live_where_the_lookup_looks(
) -> None:
    for reason in ALL_REASONS:
        for language in ALL_LANGUAGES:
            path = TEMPLATES_DIR / template_name(reason, language)
            assert path.is_file(), path


# --------------------------------------------------------------------------- #
# Rendering: every code, every language
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_renders_with_a_subject_and_a_body(
    reason: BounceReason, language: ReplyLanguage, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=language)

    assert reply.reason is reason
    assert reply.language is language
    assert reply.subject.strip()
    assert "\n" not in reply.subject
    assert len(reply.body.splitlines()) > 5
    assert reply.template == template_name(reason, language)


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_no_unfilled_placeholder_survives(
    reason: BounceReason, language: ReplyLanguage, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=language)
    rendered = reply.as_text()

    for marker in ("{{", "}}", "{%", "%}", "{#", "#}"):
        assert marker not in rendered, f"{marker} survived in {reply.template}"
    assert "Undefined" not in rendered
    assert "object at 0x" not in rendered


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_body_quotes_the_reference_and_the_attachment(
    reason: BounceReason, language: ReplyLanguage, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=language)

    assert context.reference_id in reply.body
    assert context.attachment_name is not None
    assert context.attachment_name in reply.body
    assert "2, 3" in reply.body


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_optional_lines_vanish_without_leaving_a_hole(
    reason: BounceReason, language: ReplyLanguage, bare_context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, bare_context, language=language)

    assert "Attachment received" not in reply.body
    assert "प्राप्त संलग्नक" not in reply.body
    assert "Pages affected" not in reply.body
    assert bare_context.reference_id in reply.body
    assert "\n\n\n" not in reply.body


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_separator_does_not_leak_into_the_body(
    reason: BounceReason, language: ReplyLanguage, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=language)
    assert SUBJECT_BODY_SEPARATOR not in [
        line.strip() for line in reply.body.splitlines()
    ]


def test_rendering_is_deterministic(context: ReplyContext) -> None:
    first = _render_all(context)
    second = _render_all(context)
    assert [reply.model_dump() for reply in first] == [
        reply.model_dump() for reply in second
    ]


# --------------------------------------------------------------------------- #
# The wording itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
def test_english_asks_rather_than_instructs(
    reason: BounceReason, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=ReplyLanguage.ENGLISH)
    text = reply.as_text().lower()

    assert "please" in text
    for word in DISCOURAGED_ENGLISH:
        assert word not in text, f"{word!r} in {reply.template}"


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
def test_hindi_asks_rather_than_instructs(
    reason: BounceReason, context: ReplyContext
) -> None:
    reply = render_bounce_reply(reason, context, language=ReplyLanguage.HINDI)
    text = reply.as_text()

    assert "कृपया" in text
    for word in DISCOURAGED_HINDI:
        assert word not in text, f"{word!r} in {reply.template}"


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
def test_hindi_is_written_in_devanagari_and_english_is_not(
    reason: BounceReason, context: ReplyContext
) -> None:
    hindi = render_bounce_reply(reason, context, language=ReplyLanguage.HINDI)
    english = render_bounce_reply(reason, context, language=ReplyLanguage.ENGLISH)

    assert any(ord(char) in DEVANAGARI for char in hindi.body)
    body_without_reference = english.body.replace(context.reference_id, "")
    assert not any(ord(char) in DEVANAGARI for char in body_without_reference)


@pytest.mark.parametrize("reason", DPI_REMEDY_REASONS, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_rescan_remedies_quote_the_dpi(
    reason: BounceReason, language: ReplyLanguage
) -> None:
    context = ReplyContext(reference_id="JOB-9", recommended_dpi=400)
    reply = render_bounce_reply(reason, context, language=language)

    assert "400" in reply.body
    assert "300" not in reply.body


@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_page_cap_quotes_the_configured_limit(language: ReplyLanguage) -> None:
    context = ReplyContext(reference_id="JOB-9", max_pages=42)
    reply = render_bounce_reply(
        BounceReason.PAGE_CAP_EXCEEDED, context, language=language
    )

    assert "42" in reply.body
    assert "200" not in reply.body


@pytest.mark.parametrize("language", ALL_LANGUAGES, ids=lambda l: l.value)
def test_encrypted_pdf_names_the_defect_and_the_remedy(
    language: ReplyLanguage, context: ReplyContext
) -> None:
    reply = render_bounce_reply(
        BounceReason.ENCRYPTED_PDF, context, language=language
    )
    needle = "password" if language is ReplyLanguage.ENGLISH else "पासवर्ड"

    assert needle in reply.body
    assert reply.body.count(needle) >= 2  # the defect and the remedy both name it


def test_encrypted_pdf_does_not_ask_for_the_password(
    context: ReplyContext,
) -> None:
    """Never invite a complainant to mail us a credential."""
    english = render_bounce_reply(
        BounceReason.ENCRYPTED_PDF, context, language=ReplyLanguage.ENGLISH
    )
    assert "do not send us" in english.body.lower()

    hindi = render_bounce_reply(
        BounceReason.ENCRYPTED_PDF, context, language=ReplyLanguage.HINDI
    )
    assert "न भेजें" in hindi.body


@pytest.mark.parametrize("reason", ALL_REASONS, ids=lambda r: r.value)
def test_the_complaint_is_not_closed_is_stated_every_time(
    reason: BounceReason, context: ReplyContext
) -> None:
    english = render_bounce_reply(reason, context, language=ReplyLanguage.ENGLISH)
    hindi = render_bounce_reply(reason, context, language=ReplyLanguage.HINDI)

    assert "has not been closed" in english.body
    assert "बंद नहीं की गई है" in hindi.body


# --------------------------------------------------------------------------- #
# Language selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", ReplyLanguage.ENGLISH),
        ("EN", ReplyLanguage.ENGLISH),
        ("eng", ReplyLanguage.ENGLISH),
        ("en-IN", ReplyLanguage.ENGLISH),
        ("en_IN", ReplyLanguage.ENGLISH),
        ("hi", ReplyLanguage.HINDI),
        ("hin", ReplyLanguage.HINDI),
        ("HI-in", ReplyLanguage.HINDI),
        (ReplyLanguage.HINDI, ReplyLanguage.HINDI),
    ],
)
def test_language_coercion(value: str, expected: ReplyLanguage) -> None:
    assert ReplyLanguage.coerce(value) is expected


@pytest.mark.parametrize("value", ["mar", "ta", "", "klingon", "hindi-ish"])
def test_unreviewed_language_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="no reviewed bounce-reply template"):
        ReplyLanguage.coerce(value)


def test_marathi_is_deliberately_absent() -> None:
    """The corpus has Marathi; nobody reviewed Marathi prose for tone."""
    assert {language.value for language in ReplyLanguage} == {"eng", "hin"}


# --------------------------------------------------------------------------- #
# Bilingual assembly
# --------------------------------------------------------------------------- #


def test_bilingual_reply_carries_both_languages(context: ReplyContext) -> None:
    reply = render_bilingual_bounce_reply(BounceReason.BLANK_PAGE, context)

    assert isinstance(reply, BilingualBounceReply)
    assert [item.language for item in reply.replies] == list(DEFAULT_LANGUAGES)
    assert reply.subject.count(" / ") == 1
    for item in reply.replies:
        assert item.subject in reply.subject
        assert item.body in reply.body
    assert "-" * 60 in reply.body


def test_bilingual_language_order_is_respected(context: ReplyContext) -> None:
    reply = render_bilingual_bounce_reply(
        BounceReason.BLANK_PAGE, context, languages=("hin", "eng")
    )
    assert [item.language.value for item in reply.replies] == ["hin", "eng"]
    assert reply.body.index(reply.replies[0].body) == 0


def test_bilingual_for_language_lookup(context: ReplyContext) -> None:
    reply = render_bilingual_bounce_reply(BounceReason.EMPTY_FILE, context)

    assert reply.for_language("hi").language is ReplyLanguage.HINDI
    assert reply.for_language(ReplyLanguage.ENGLISH).language is ReplyLanguage.ENGLISH


def test_bilingual_for_language_refuses_an_unrendered_language(
    context: ReplyContext,
) -> None:
    reply = render_bilingual_bounce_reply(
        BounceReason.EMPTY_FILE, context, languages=("eng",)
    )
    with pytest.raises(KeyError):
        reply.for_language("hin")


def test_bilingual_refuses_an_empty_or_repeated_language_list(
    context: ReplyContext,
) -> None:
    with pytest.raises(ValueError, match="at least one language"):
        render_bilingual_bounce_reply(BounceReason.EMPTY_FILE, context, languages=())
    with pytest.raises(ValueError, match="duplicate languages"):
        render_bilingual_bounce_reply(
            BounceReason.EMPTY_FILE, context, languages=("eng", "en")
        )


# --------------------------------------------------------------------------- #
# Envelope glue
# --------------------------------------------------------------------------- #


def test_accepted_envelope_produces_no_reply(
    make_envelope: Callable[..., ResultEnvelope],
) -> None:
    assert reply_for_envelope(make_envelope()) is None


def _bounced_envelope(
    reason: BounceReason,
    *,
    detail: str | None = None,
    job_id: str | None = "JOB-77",
) -> ResultEnvelope:
    pages = (
        PageResult(
            page_no=1,
            quality_score=0.9,
            quality_verdict=QualityVerdict.ACCEPT,
            ocr_tier=OCRTier.TIER1,
            line_count=4,
        ),
        PageResult(
            page_no=2,
            quality_score=0.1,
            quality_verdict=QualityVerdict.BOUNCE,
            ocr_tier=OCRTier.NONE,
            line_count=0,
        ),
    )
    return ResultEnvelope(
        result_version=1,
        is_final=True,
        document_id="sha256:" + "b" * 64,
        job_id=job_id,
        provider_profile="laptop",
        bounce_decision=BounceDecision(
            decision=QualityVerdict.BOUNCE, reason=reason, detail=detail
        ),
        pages=pages,
        document=None,
    )


def test_bounced_envelope_produces_a_bilingual_reply() -> None:
    envelope = _bounced_envelope(BounceReason.BLUR_ABOVE_CEILING)

    reply = reply_for_envelope(envelope)

    assert reply is not None
    assert reply.reason is BounceReason.BLUR_ABOVE_CEILING
    for item in reply.replies:
        assert "JOB-77" in item.body
        assert "2" in item.body  # the page that actually bounced


def test_reference_falls_back_to_the_document_id() -> None:
    envelope = _bounced_envelope(BounceReason.EMPTY_FILE, job_id=None)

    reply = reply_for_envelope(envelope)

    assert reply is not None
    assert envelope.document_id in reply.replies[0].body


def test_internal_detail_never_reaches_the_complainant() -> None:
    envelope = _bounced_envelope(
        BounceReason.BLUR_ABOVE_CEILING,
        detail="laplacian_variance=812.4 below QUALITY_BLUR_LAPLACIAN_FLOOR",
    )

    reply = reply_for_envelope(envelope)

    assert reply is not None
    assert "laplacian" not in reply.body.lower()
    assert "QUALITY_BLUR" not in reply.body


def test_envelope_overrides_reach_the_template() -> None:
    envelope = _bounced_envelope(BounceReason.PAGE_CAP_EXCEEDED)

    reply = reply_for_envelope(envelope, max_pages=12, languages=("eng",))

    assert reply is not None
    assert "12" in reply.replies[0].body


# --------------------------------------------------------------------------- #
# Context validation
# --------------------------------------------------------------------------- #


def test_reference_id_is_required() -> None:
    with pytest.raises(ValueError):
        ReplyContext(reference_id="")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_numbers": (0,)},
        {"page_numbers": (1, 1)},
        {"reference_id": "JOB\n1"},
        {"attachment_name": "a\nb"},
        {"max_pages": 0},
        {"recommended_dpi": 10},
    ],
)
def test_context_rejects_nonsense(kwargs: dict[str, object]) -> None:
    payload: dict[str, object] = {"reference_id": "JOB-1"}
    payload.update(kwargs)
    with pytest.raises(ValueError):
        ReplyContext(**payload)  # type: ignore[arg-type]


def test_context_is_a_closed_set() -> None:
    with pytest.raises(ValueError):
        ReplyContext(reference_id="JOB-1", complainant_name="A. Person")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_unknown_reason_is_a_typed_error(context: ReplyContext) -> None:
    with pytest.raises(ReplyTemplateMissing) as excinfo:
        render_bounce_reply("NOT_A_REASON", context)
    assert excinfo.value.code == "REPLY_TEMPLATE_MISSING"


@pytest.fixture
def swapped_templates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str, str], None]:
    """Point the renderer at a throwaway template tree for one test."""

    def _install(filename: str, body: str) -> None:
        directory = tmp_path / "eng"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_text(body, encoding="utf-8")
        monkeypatch.setattr(bounce_module, "TEMPLATES_DIR", tmp_path)
        bounce_module._environment.cache_clear()

    yield _install
    bounce_module._environment.cache_clear()


def test_missing_template_file_is_a_typed_error(
    swapped_templates: Callable[[str, str], None], context: ReplyContext
) -> None:
    swapped_templates("EMPTY_FILE.txt.j2", "Subject\n---\nBody")

    with pytest.raises(ReplyTemplateMissing) as excinfo:
        render_bounce_reply(BounceReason.BLANK_PAGE, context)
    assert excinfo.value.context["reason"] == "BLANK_PAGE"


def test_undefined_variable_is_a_typed_error(
    swapped_templates: Callable[[str, str], None], context: ReplyContext
) -> None:
    swapped_templates("EMPTY_FILE.txt.j2", "Subject\n---\nHello {{ nobody_defined }}")

    with pytest.raises(ReplyTemplateInvalid):
        render_bounce_reply(BounceReason.EMPTY_FILE, context)


@pytest.mark.parametrize(
    "body",
    [
        "Subject with no separator",
        "Subject\n---\n",
        "\n---\nBody only",
        "Subject\n---\nBody\n---\nAnother",
        "Line one\nLine two\n---\nBody",
    ],
)
def test_unsendable_shapes_are_refused(
    swapped_templates: Callable[[str, str], None],
    context: ReplyContext,
    body: str,
) -> None:
    swapped_templates("EMPTY_FILE.txt.j2", body)

    with pytest.raises(ReplyTemplateInvalid):
        render_bounce_reply(BounceReason.EMPTY_FILE, context)


def test_surviving_template_syntax_is_refused(
    swapped_templates: Callable[[str, str], None], context: ReplyContext
) -> None:
    """Output that still looks like a template must not be mailed."""
    swapped_templates(
        "EMPTY_FILE.txt.j2", "Subject\n---\nReference: {{ '{{' }} reference_id }}"
    )

    with pytest.raises(ReplyTemplateInvalid, match="unrendered"):
        render_bounce_reply(BounceReason.EMPTY_FILE, context)


def test_a_template_that_does_not_compile_is_refused(
    swapped_templates: Callable[[str, str], None], context: ReplyContext
) -> None:
    """A mistyped placeholder fails at compile time, before any render."""
    swapped_templates(
        "EMPTY_FILE.txt.j2", "Subject\n---\nReference: {{ reference_id }"
    )

    with pytest.raises(ReplyTemplateInvalid, match="does not compile"):
        render_bounce_reply(BounceReason.EMPTY_FILE, context)


def test_missing_language_directory_reports_no_reasons(
    swapped_templates: Callable[[str, str], None]
) -> None:
    swapped_templates("EMPTY_FILE.txt.j2", "Subject\n---\nBody")

    assert available_reasons(ReplyLanguage.HINDI) == frozenset()
    assert available_reasons(ReplyLanguage.ENGLISH) == frozenset(
        {BounceReason.EMPTY_FILE}
    )
    assert len(missing_templates(ALL_LANGUAGES)) == len(ALL_REASONS) * 2 - 1
