"""A complete, deterministic in-memory ``ModelProvider`` for tests.

BUILD-BRIEF sec. 7.3 requires a contract test that runs "under every applicable
provider profile".  Two of the three real profiles cannot run in CI — one needs
weights, the other a token and a network — so the contract suite runs against
this mock plus whatever the real classes can prove offline.

Design rules, so this stays a *test double* and not a second implementation:

* every method returns contract objects built through the real Pydantic models,
  so a contract change breaks the mock loudly rather than letting the suite
  drift;
* nothing is random and nothing sleeps.  The same inputs give the same outputs
  in the same order, because the contract test asserts ordering;
* ``calls`` records ``(method, argument summary)`` so a test can assert routing
  without patching;
* ``fail_with`` makes a chosen method raise, so failure-path behaviour is
  exercised through the same object.

This module does NOT handle: any real inference, image decoding, or network
access.  It is imported by tests only and never ships.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    LangCandidate,
    LangResult,
    OCRLine,
    PageClass,
    PageClassLabel,
    RawField,
    ScriptResult,
    ScriptVote,
    StructuralFeatures,
    bbox_from_xyxy,
)

__all__ = ["MockProvider"]


class MockProvider:
    """In-memory implementation of the eight ``ModelProvider`` methods."""

    name = "mock"

    def __init__(
        self,
        *,
        script: str = "Latn",
        language: str = "eng",
        page_label: PageClassLabel = PageClassLabel.COMPLAINT_FORM,
        lines: Sequence[OCRLine] | None = None,
        heavy_lines: Sequence[OCRLine] | None = None,
        field_values: Mapping[str, str] | None = None,
        translation_prefix: str = "EN:",
        fail_with: Mapping[str, Exception] | None = None,
    ) -> None:
        self.script = script
        self.language = language
        self.page_label = page_label
        self.lines = list(lines) if lines is not None else _default_lines(script)
        self.heavy_lines = (
            list(heavy_lines) if heavy_lines is not None else self.lines
        )
        self.field_values = dict(field_values or {})
        self.translation_prefix = translation_prefix
        self.fail_with = dict(fail_with or {})
        self.calls: list[tuple[str, Any]] = []

    # ------------------------------------------------------------- internals
    def _record(self, method: str, summary: Any) -> None:
        self.calls.append((method, summary))
        failure = self.fail_with.get(method)
        if failure is not None:
            raise failure

    def method_calls(self, method: str) -> list[Any]:
        """Argument summaries recorded for one method, in call order."""
        return [summary for name, summary in self.calls if name == method]

    # -------------------------------------------------------------- protocol
    def detect_script(self, crops: Sequence[Any]) -> ScriptResult:
        """Always the configured script; the vote list mirrors the crop count."""
        self._record("detect_script", len(crops))
        return ScriptResult(
            script=self.script,
            confidence=1.0,
            distribution={self.script: 1.0},
            votes=[
                ScriptVote(script=self.script, confidence=1.0)
                for _ in range(len(crops))
            ],
            sampled_crops=len(crops),
        )

    def classify_page(self, page: Any, feats: StructuralFeatures) -> PageClass:
        """Always the configured label, with a two-entry score distribution."""
        self._record("classify_page", feats.line_count)
        other = (
            PageClassLabel.UNKNOWN
            if self.page_label is not PageClassLabel.UNKNOWN
            else PageClassLabel.SUPPORTING_DOC
        )
        return PageClass(
            label=self.page_label,
            confidence=0.9,
            scores={self.page_label.value: 0.9, other.value: 0.1},
        )

    def ocr_page(self, page: Any, script: str) -> list[OCRLine]:
        """The configured lines, restamped with the requested script."""
        self._record("ocr_page", script)
        return [
            line.model_copy(update={"script": script}) for line in self.lines
        ]

    def ocr_page_heavy(self, page: Any) -> list[OCRLine]:
        """The configured Tier-2 lines."""
        self._record("ocr_page_heavy", None)
        return list(self.heavy_lines)

    def identify_language(self, text: str, script: str) -> LangResult:
        """Empty text yields ``None``; otherwise the configured language."""
        self._record("identify_language", (text, script))
        if not text.strip():
            return LangResult(
                language=None, confidence=0.0, script=script, candidates=[]
            )
        return LangResult(
            language=self.language,
            confidence=0.95,
            script=script,
            candidates=[
                LangCandidate(language=self.language, confidence=0.95)
            ],
        )

    def extract_fields(self, ctx: ExtractionContext) -> dict[str, RawField]:
        """Return configured values for requested fields only.

        ``complaint_summary`` comes back as line indices with a ``None`` value,
        because PLAN.md invariant 2 makes any other shape unrepresentable.
        """
        self._record("extract_fields", (ctx.document_id, ctx.page_no))
        indices = [line.line_index for line in ctx.lines]
        fields: dict[str, RawField] = {}
        for name in ctx.target_fields:
            if name == SUMMARY_FIELD_NAME:
                if indices:
                    fields[name] = RawField(
                        name=name,
                        value=None,
                        confidence=0.8,
                        line_indices=indices[:2],
                    )
                continue
            value = self.field_values.get(name)
            if value is not None:
                fields[name] = RawField(
                    name=name,
                    value=value,
                    confidence=0.75,
                    line_indices=indices[:1],
                )
        return fields

    def extract_fields_batch(
        self, ctxs: Sequence[ExtractionContext]
    ) -> list[dict[str, RawField]]:
        """Sequential loop — the local-profile shape of "provider owns it"."""
        self._record("extract_fields_batch", len(ctxs))
        return [self.extract_fields(ctx) for ctx in ctxs]

    def translate(
        self, texts: Sequence[str], src: str, tgt: str = "eng"
    ) -> list[str]:
        """Prefix each non-empty string; positions are always preserved."""
        self._record("translate", (src, tgt, len(texts)))
        return [
            f"{self.translation_prefix}{text}" if text else text
            for text in texts
        ]


def _default_lines(script: str) -> list[OCRLine]:
    """Two lines: an English label and a handwritten Devanagari value."""
    return [
        OCRLine(
            text="4. Name of bank",
            bbox=bbox_from_xyxy(110.0, 534.0, 341.0, 566.0),
            confidence=0.94,
            script=script,
            language=None,
            is_handwritten=False,
            line_index=0,
        ),
        OCRLine(
            text="बैंक ऑफ "
            "बड़ौदा",
            bbox=bbox_from_xyxy(720.0, 526.0, 968.0, 567.0),
            confidence=0.71,
            script="Deva",
            language=None,
            is_handwritten=True,
            line_index=1,
        ),
    ]
