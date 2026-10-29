"""Stage 2b — line-level LID: script-constrained labels and the code-mix flag.

The BUILD-BRIEF M4 acceptance test is here, run against the **committed corpus
ground truth** rather than against invented lines: on ``bilingual_form``, the
printed English label lines carry ``eng``, the handwritten Devanagari value
lines carry ``mar``, and ``is_code_mixed`` is true.

Only corpus **variants 0-2** are read.  Variants 3 and 4 are the locked test
split (BUILD-BRIEF M1) and nothing in ``app/`` — including a test that shapes
what ``app/`` looks like — may be tuned against them.  :data:`READABLE_VARIANTS`
is the single place that is stated, and ``test_the_locked_split_is_never_read``
asserts it.

No model file is touched.  The provider is a stub that emulates ``lid.176.ftz``
by routing deliberately polluted raw fastText output through the provider's own
:func:`app.providers.local.constrain_language_candidates`, which is what makes
"a Devanagari line is never English however confidently the model says so"
(PRD P0-5) an assertion about real code rather than about a convenient fake.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from app.contracts.ocr import LangResult, OCRLine, bbox_from_xyxy
from app.providers.base import ModelArtifactMissing
from app.providers.local import constrain_language_candidates
from app.settings import Settings
from app.stages.s2b_lid import (
    DEFAULT_CODE_MIX_THRESHOLD,
    MIN_LID_CHARACTERS,
    LidResult,
    LineLanguage,
    identify_line_languages,
    is_code_mixed,
    languages_for_script,
    script_distribution,
)

#: Corpus variants this file may read.  3 and 4 are the locked test split.
READABLE_VARIANTS: Final[tuple[int, ...]] = (0, 1, 2)

CORPUS: Final[Path] = Path(__file__).resolve().parents[2] / "corpus"

ENGLISH = "Name of bank and branch"
MARATHI = "बँकेने माझ्या खात्यातून दोनदा पैसे कापले"


# --------------------------------------------------------------------------- #
# Fixtures and stubs
# --------------------------------------------------------------------------- #


def _line(
    text: str,
    script: str,
    index: int,
    *,
    handwritten: bool = False,
) -> OCRLine:
    return OCRLine(
        text=text,
        bbox=bbox_from_xyxy(10.0, 10.0 + 20 * index, 400.0, 30.0 + 20 * index),
        confidence=0.9,
        script=script,
        language=None,
        is_handwritten=handwritten,
        line_index=index,
    )


class _FastTextLike:
    """Emulates ``lid.176.ftz`` **through the provider's own constraint**.

    Raw output is deliberately polluted with a high-probability ``__label__en``
    on every line, exactly as a 176-language model does on short Devanagari
    strings, plus a label outside the ISO-639-3 map that must be dropped.
    """

    def __init__(
        self,
        truth: Mapping[str, str] | None = None,
        *,
        default: str | None = None,
        english_noise: float | None = 0.85,
        truth_probability: float = 0.90,
    ) -> None:
        self.truth = dict(truth or {})
        self.default = default
        self.english_noise = english_noise
        self.truth_probability = truth_probability
        self.calls: list[tuple[str, str]] = []

    def identify_language(self, text: str, script: str) -> LangResult:
        self.calls.append((text, script))
        flattened = " ".join(text.split())
        label = self.truth.get(flattened, self.default)
        raw: list[tuple[str, float]] = [
            ("__label__ceb", 0.30),  # outside the ISO 639-3 map: dropped
        ]
        if self.english_noise is not None:
            raw.insert(0, ("__label__en", self.english_noise))
        if label is not None:
            raw.insert(0, (f"__label__{label}", self.truth_probability))
        return constrain_language_candidates(raw, script)


class _RawProvider:
    """Provider stub returning a caller-supplied result, constraint bypassed."""

    def __init__(self, result: LangResult | Exception) -> None:
        self.result = result
        self.calls = 0

    def identify_language(self, text: str, script: str) -> LangResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _artifact_missing() -> ModelArtifactMissing:
    return ModelArtifactMissing(
        artifact="fastText language-identification model (lid.176.ftz)",
        path=Path("models/lid/lid.176.ftz"),
        how_to_obtain="Run tools/fetch_models.py on a networked machine.",
    )


def _load_gt(doc_type: str, variant: int, profile: str = "clean") -> dict[str, Any]:
    """Read one corpus ground-truth file, refusing the locked test split."""
    assert variant in READABLE_VARIANTS, (
        f"variant {variant} is in the locked test split (BUILD-BRIEF M1); "
        "no stage and no stage test may read it"
    )
    path = CORPUS / doc_type / f"{doc_type}_v{variant}_{profile}.gt.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _lines_from_gt(ground_truth: Mapping[str, Any]) -> list[OCRLine]:
    """Ground-truth lines as unlabelled :class:`OCRLine` objects."""
    return [
        OCRLine(
            text=entry["text"],
            bbox=[float(value) for value in entry["bbox"]],
            confidence=1.0,
            script=entry["script"],
            language=None,
            is_handwritten=bool(entry["is_handwritten"]),
            line_index=index,
        )
        for index, entry in enumerate(ground_truth["lines"])
    ]


def _truth_map(ground_truth: Mapping[str, Any]) -> dict[str, str]:
    return {
        " ".join(entry["text"].split()): entry["language"]
        for entry in ground_truth["lines"]
        if entry["language"]
    }


# --------------------------------------------------------------------------- #
# The script-constrained label set (PRD P0-5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("Latn", {"eng"}),
        ("Deva", {"hin", "mar", "kok", "nep"}),
        ("Beng", {"ben", "asm"}),
        ("Taml", {"tam"}),
        ("Arab", {"urd"}),
    ],
)
def test_the_permitted_label_set_is_the_providers_table(
    script: str, expected: set[str]
) -> None:
    assert set(languages_for_script(script)) == expected


@pytest.mark.parametrize("script", ["Zyyy", "Hant", "Cyrl"])
def test_an_unknown_script_permits_nothing_rather_than_defaulting(
    script: str,
) -> None:
    """A new script must not be the place English silently appears."""
    assert languages_for_script(script) == frozenset()


def test_a_devanagari_line_is_never_english_however_confident_the_model_is(
    settings: Settings,
) -> None:
    """PRD P0-5, the whole point of the stage."""
    provider = _FastTextLike({MARATHI: "mar"}, english_noise=0.99)
    result = identify_line_languages(
        [_line(MARATHI, "Deva", 0)], provider=provider, settings=settings
    )
    assert result.per_line[0].language == "mar"
    assert all(
        candidate.language != "eng" for candidate in result.per_line[0].candidates
    )


def test_a_provider_that_ignores_the_constraint_is_overruled_here(
    settings: Settings,
) -> None:
    """Defence in depth: a provider bug must not poison the page summary."""
    provider = _RawProvider(
        LangResult(
            language="eng", confidence=0.99, script="Deva", candidates=[]
        )
    )
    result = identify_line_languages(
        [_line(MARATHI, "Deva", 0)], provider=provider, settings=settings
    )
    assert result.per_line[0].language is None
    assert result.per_line[0].unlabelled_reason == "no_permitted_candidate"


def test_nothing_surviving_the_constraint_leaves_the_line_unlabelled(
    settings: Settings,
) -> None:
    """A Latin line the model calls Hindi is unlabelled, not relabelled eng."""
    provider = _FastTextLike({ENGLISH: "hin"}, english_noise=None)
    result = identify_line_languages(
        [_line(ENGLISH, "Latn", 0)], provider=provider, settings=settings
    )
    assert result.per_line[0].language is None
    assert result.per_line[0].unlabelled_reason == "no_permitted_candidate"


def test_a_line_in_a_script_with_no_label_set_is_unlabelled(
    settings: Settings,
) -> None:
    provider = _FastTextLike(default="rus")
    result = identify_line_languages(
        [_line("Привет банк", "Cyrl", 0)], provider=provider, settings=settings
    )
    assert result.per_line[0].unlabelled_reason == "unknown_script"
    assert provider.calls == [], "an unlabelable script never reaches the model"


# --------------------------------------------------------------------------- #
# Withheld labels always say why
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "script", "reason"),
    [
        ("", "Latn", "empty_text"),
        ("   \n  ", "Latn", "empty_text"),
        ("ab", "Latn", "too_short"),
        ("देव", "Deva", "too_short"),
    ],
)
def test_a_line_too_short_or_empty_is_recorded_not_guessed(
    text: str, script: str, reason: str, settings: Settings
) -> None:
    provider = _FastTextLike(default="eng")
    result = identify_line_languages(
        [_line(text, script, 0)], provider=provider, settings=settings
    )
    assert result.per_line[0].language is None
    assert result.per_line[0].unlabelled_reason == reason
    assert result.unlabelled_count == 1


def test_the_minimum_length_is_the_documented_constant(settings: Settings) -> None:
    short = "x" * (MIN_LID_CHARACTERS - 1)
    long = "x" * MIN_LID_CHARACTERS
    provider = _FastTextLike(default="eng")
    result = identify_line_languages(
        [_line(short, "Latn", 0), _line(long, "Latn", 1)],
        provider=provider,
        settings=settings,
    )
    assert result.per_line[0].unlabelled_reason == "too_short"
    assert result.per_line[1].language == "eng"


def test_a_low_confidence_identification_is_recorded_but_withheld(
    settings: Settings,
) -> None:
    provider = _RawProvider(
        LangResult(language="mar", confidence=0.05, script="Deva", candidates=[])
    )
    result = identify_line_languages(
        [_line(MARATHI, "Deva", 0)],
        provider=provider,
        settings=settings,
        min_confidence=0.5,
    )
    assert result.per_line[0].language is None
    assert result.per_line[0].unlabelled_reason == "low_confidence"


def test_an_unlabelled_line_must_carry_a_reason() -> None:
    with pytest.raises(ValueError, match="must say why"):
        LineLanguage(line_index=0, language=None, confidence=0.0, script="Deva")


def test_a_labelled_line_may_not_also_carry_a_reason() -> None:
    with pytest.raises(ValueError, match="carries both a language"):
        LineLanguage(
            line_index=0,
            language="mar",
            confidence=0.9,
            script="Deva",
            unlabelled_reason="too_short",
        )


def test_the_line_record_itself_refuses_an_impermissible_label() -> None:
    with pytest.raises(ValueError, match="PRD P0-5"):
        LineLanguage(line_index=0, language="eng", confidence=0.9, script="Deva")


# --------------------------------------------------------------------------- #
# The missing-artifact policy
# --------------------------------------------------------------------------- #


def test_a_missing_lid_model_is_a_deployment_fault_by_default(
    settings: Settings,
) -> None:
    provider = _RawProvider(_artifact_missing())
    with pytest.raises(ModelArtifactMissing):
        identify_line_languages(
            [_line(ENGLISH, "Latn", 0)], provider=provider, settings=settings
        )


def test_the_pipeline_may_choose_unlabelled_lines_over_a_failed_page(
    settings: Settings,
) -> None:
    provider = _RawProvider(_artifact_missing())
    lines = [_line(ENGLISH, "Latn", i) for i in range(4)]

    result = identify_line_languages(
        lines,
        provider=provider,
        settings=settings,
        on_missing_artifact="unlabelled",
    )

    assert result.unlabelled_count == 4
    assert {record.unlabelled_reason for record in result.per_line} == {
        "artifact_missing"
    }
    assert provider.calls == 1, "the model is not re-asked once it is known absent"


# --------------------------------------------------------------------------- #
# Code mixing — the page-level fact about scripts
# --------------------------------------------------------------------------- #


def test_script_distribution_sums_to_one() -> None:
    lines = [_line(ENGLISH, "Latn", i) for i in range(3)]
    lines += [_line(MARATHI, "Deva", 3 + i) for i in range(2)]
    distribution = script_distribution(lines)
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["Latn"] == pytest.approx(0.6)


def test_a_page_with_no_lines_is_not_a_negative_finding() -> None:
    mixed, majority, distribution = is_code_mixed([])
    assert mixed is False
    assert majority == "Zyyy"
    assert distribution == {}


def test_the_threshold_is_inclusive_at_the_boundary() -> None:
    """15% of lines differing is code-mixed; the brief says ``>=``."""
    lines = [_line(ENGLISH, "Latn", i) for i in range(17)]
    lines += [_line(MARATHI, "Deva", 17 + i) for i in range(3)]
    mixed, majority, _ = is_code_mixed(lines, threshold=0.15)
    assert mixed is True
    assert majority == "Latn"

    just_under = lines[:-1]  # 2 of 19 = 10.5%
    assert is_code_mixed(just_under, threshold=0.15)[0] is False


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_an_out_of_range_threshold_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        is_code_mixed([_line(ENGLISH, "Latn", 0)], threshold=threshold)


def test_the_threshold_comes_from_settings_and_can_be_overridden(
    make_settings: Callable[..., Settings],
) -> None:
    lines = [_line(ENGLISH, "Latn", i) for i in range(9)]
    lines += [_line(MARATHI, "Deva", 9)]  # 10% minority
    provider = _FastTextLike(default=None)

    lenient = identify_line_languages(
        lines, provider=provider, settings=make_settings(CODE_MIX_THRESHOLD=0.05)
    )
    strict = identify_line_languages(
        lines, provider=provider, settings=make_settings(CODE_MIX_THRESHOLD=0.5)
    )
    override = identify_line_languages(
        lines,
        provider=provider,
        settings=make_settings(CODE_MIX_THRESHOLD=0.5),
        code_mix_threshold=0.05,
    )

    assert lenient.is_code_mixed is True
    assert lenient.code_mix_threshold == pytest.approx(0.05)
    assert strict.is_code_mixed is False
    assert override.is_code_mixed is True


def test_an_out_of_range_override_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match=r"code_mix_threshold must be in \[0, 1\]"):
        identify_line_languages(
            [_line(ENGLISH, "Latn", 0)],
            provider=_FastTextLike(default="eng"),
            settings=settings,
            code_mix_threshold=2.0,
        )


def test_the_shipped_default_matches_the_brief(settings: Settings) -> None:
    assert DEFAULT_CODE_MIX_THRESHOLD == pytest.approx(0.15)
    assert settings.CODE_MIX_THRESHOLD == pytest.approx(0.15)


# --------------------------------------------------------------------------- #
# BUILD-BRIEF M4, against the committed corpus (variants 0-2 only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", READABLE_VARIANTS)
def test_bilingual_form_labels_english_labels_and_marathi_values(
    variant: int, settings: Settings
) -> None:
    """The M4 acceptance test, on the corpus page it was written for."""
    ground_truth = _load_gt("bilingual_form", variant)
    lines = _lines_from_gt(ground_truth)
    provider = _FastTextLike(_truth_map(ground_truth), english_noise=0.95)

    result = identify_line_languages(lines, provider=provider, settings=settings)

    labelled = [
        (line, record)
        for line, record in zip(result.lines, result.per_line, strict=True)
        if record.unlabelled_reason != "too_short"
    ]
    assert labelled, "the page must produce some labelled lines"
    for line, record in labelled:
        expected = "eng" if line.script == "Latn" else "mar"
        assert record.language == expected, (
            f"line {record.line_index} ({line.script}) got {record.language}"
        )
        assert line.language == expected

    # The defining hard case: printed English labels, handwritten Marathi
    # values.  (The form also has handwritten *Latin* values — account numbers,
    # dates — which is why the assertion runs Deva -> handwritten, not the
    # converse.)
    devanagari = [line for line in result.lines if line.script == "Deva"]
    assert devanagari, "the bilingual form carries Devanagari value lines"
    assert all(line.is_handwritten for line in devanagari)
    assert {line.language for line in devanagari} == {"mar"}

    printed = [line for line in result.lines if not line.is_handwritten]
    assert printed, "the bilingual form carries printed label lines"
    assert {line.script for line in printed} == {"Latn"}
    assert {line.language for line in printed} == {"eng"}


@pytest.mark.parametrize("variant", READABLE_VARIANTS)
def test_bilingual_form_is_code_mixed_exactly_as_the_ground_truth_says(
    variant: int, settings: Settings
) -> None:
    ground_truth = _load_gt("bilingual_form", variant)
    lines = _lines_from_gt(ground_truth)
    provider = _FastTextLike(_truth_map(ground_truth))

    result = identify_line_languages(lines, provider=provider, settings=settings)

    assert ground_truth["is_code_mixed"] is True
    assert result.is_code_mixed is True
    assert result.majority_script == ground_truth["primary_script"] == "Latn"
    assert set(result.script_distribution) == {"Latn", "Deva"}
    assert result.language_distribution["eng"] > result.language_distribution["mar"]


@pytest.mark.parametrize(
    ("doc_type", "script", "language"),
    [("complaint_form_en", "Latn", "eng"), ("complaint_letter_mar", "Deva", "mar")],
)
@pytest.mark.parametrize("variant", READABLE_VARIANTS)
def test_a_monolingual_page_is_not_code_mixed(
    doc_type: str, script: str, language: str, variant: int, settings: Settings
) -> None:
    ground_truth = _load_gt(doc_type, variant)
    lines = _lines_from_gt(ground_truth)
    provider = _FastTextLike(_truth_map(ground_truth))

    result = identify_line_languages(lines, provider=provider, settings=settings)

    assert ground_truth["is_code_mixed"] is False
    assert result.is_code_mixed is False
    assert result.majority_script == script
    assert set(result.language_distribution) == {language}


def test_the_locked_split_is_never_read() -> None:
    """Variants 3 and 4 exist and are deliberately outside this file's reach."""
    assert READABLE_VARIANTS == (0, 1, 2)
    for variant in (3, 4):
        assert (
            CORPUS / "bilingual_form" / f"bilingual_form_v{variant}_clean.gt.json"
        ).is_file(), "the locked split must exist to be locked"
        with pytest.raises(AssertionError, match="locked test split"):
            _load_gt("bilingual_form", variant)


# --------------------------------------------------------------------------- #
# Result-model discipline
# --------------------------------------------------------------------------- #


def test_the_input_lines_are_never_mutated(settings: Settings) -> None:
    """Stage 4 may still hold them; a shared-mutation bug shows up as a wrong
    envelope three stages later."""
    lines = [_line(ENGLISH, "Latn", 0), _line(MARATHI, "Deva", 1)]
    provider = _FastTextLike({ENGLISH: "eng", MARATHI: "mar"})

    result = identify_line_languages(lines, provider=provider, settings=settings)

    assert [line.language for line in lines] == [None, None]
    assert [line.language for line in result.lines] == ["eng", "mar"]
    assert result.lines[0] is not lines[0]


def test_duplicate_line_indices_are_refused(settings: Settings) -> None:
    lines = [_line(ENGLISH, "Latn", 0), _line(MARATHI, "Deva", 0)]
    with pytest.raises(ValueError, match="duplicate line_index"):
        identify_line_languages(
            lines, provider=_FastTextLike(default=None), settings=settings
        )


def test_language_of_resolves_by_line_index_not_position(
    settings: Settings,
) -> None:
    lines = [
        _line(ENGLISH, "Latn", 7),
        _line(MARATHI, "Deva", 3),
    ]
    provider = _FastTextLike({ENGLISH: "eng", MARATHI: "mar"})
    result = identify_line_languages(lines, provider=provider, settings=settings)

    assert result.language_of(7) == "eng"
    assert result.language_of(3) == "mar"
    with pytest.raises(KeyError, match="no line with line_index 99"):
        result.language_of(99)


def test_lines_and_identifications_are_parallel_by_contract() -> None:
    with pytest.raises(ValueError, match="they are parallel by contract"):
        LidResult(
            lines=(_line(ENGLISH, "Latn", 0),),
            per_line=(),
            majority_script="Latn",
            is_code_mixed=False,
            code_mix_threshold=0.15,
            unlabelled_count=0,
        )


def test_the_language_distribution_covers_labelled_lines_only(
    settings: Settings,
) -> None:
    """Mixing "this line is Marathi" with "we could not tell" is not a
    distribution, and the envelope's argmax rule would be meaningless."""
    lines: Sequence[OCRLine] = [
        _line(MARATHI, "Deva", 0),
        _line("ab", "Latn", 1),  # too short: unlabelled
    ]
    provider = _FastTextLike({MARATHI: "mar"})

    result = identify_line_languages(lines, provider=provider, settings=settings)

    assert result.language_distribution == pytest.approx({"mar": 1.0})
    assert result.unlabelled_count == 1
    assert sum(result.script_distribution.values()) == pytest.approx(1.0)


def test_a_page_with_no_labelled_line_has_an_empty_distribution(
    settings: Settings,
) -> None:
    provider = _FastTextLike(default=None)
    result = identify_line_languages(
        [_line(MARATHI, "Deva", 0)], provider=provider, settings=settings
    )
    assert result.language_distribution == {}
    assert result.per_line[0].language is None


def test_an_empty_page_produces_an_empty_result(settings: Settings) -> None:
    result = identify_line_languages(
        [], provider=_FastTextLike(default=None), settings=settings
    )
    assert result.lines == ()
    assert result.per_line == ()
    assert result.majority_script == "Zyyy"
    assert result.is_code_mixed is False
    assert result.unlabelled_count == 0


def test_one_provider_call_per_line_keeps_the_per_line_constraint(
    settings: Settings,
) -> None:
    """A batched call would have to pick one label set for lines that need
    different ones — that is precisely what PRD P0-5 forbids."""
    lines = [
        _line(ENGLISH, "Latn", 0),
        _line(MARATHI, "Deva", 1),
        _line(ENGLISH, "Latn", 2),
    ]
    provider = _FastTextLike({ENGLISH: "eng", MARATHI: "mar"})

    identify_line_languages(lines, provider=provider, settings=settings)

    assert [script for _text, script in provider.calls] == ["Latn", "Deva", "Latn"]


def test_newlines_inside_a_line_are_flattened_before_identification(
    settings: Settings,
) -> None:
    """fastText treats a newline as a document separator and would silently
    score only the first fragment."""
    provider = _FastTextLike({f"{ENGLISH} continued": "eng"})
    result = identify_line_languages(
        [_line(f"{ENGLISH}\ncontinued", "Latn", 0)],
        provider=provider,
        settings=settings,
    )
    assert provider.calls[0][0] == f"{ENGLISH} continued"
    assert result.per_line[0].language == "eng"
