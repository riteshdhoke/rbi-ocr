"""Unit tests for :mod:`app.providers.hf_api`.

No network and no token: every HTTP interaction goes through
``httpx.MockTransport``, which means the *real* ``openai`` client, the *real*
error classes and the *real* retry loop are exercised — only the socket is
replaced.  Sleeps are captured rather than performed, so the backoff schedule
is asserted exactly and the suite stays fast.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import openai
import pytest

from app.contracts.errors import (
    ConfigurationError,
    ProviderError,
    ProviderRateLimited,
    ProviderResponseInvalid,
    ProviderUnavailable,
    TokenBudgetExceeded,
)
from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    OCRLine,
    PageClassLabel,
    TextSpan,
    bbox_from_xyxy,
)
from app.providers.base import JobBudget, ProviderMethodNotSupported
from app.providers.cache import MEMORY_PATH, ResponseCache
from app.providers.hf_api import (
    MASK_SENTINEL_TEMPLATE,
    HFAPIProvider,
    RetryPolicy,
    backoff_delay,
    build_extraction_prompt,
    classify_openai_error,
    estimate_tokens,
    render_lines,
)
from app.settings import Settings

# --------------------------------------------------------------------------- #
# Fixtures: contexts, transports, providers
# --------------------------------------------------------------------------- #


def _lines() -> list[OCRLine]:
    return [
        OCRLine(
            text="Name: Anita Deshmukh",
            bbox=bbox_from_xyxy(100.0, 200.0, 700.0, 232.0),
            confidence=0.93,
            script="Latn",
            language=None,
            is_handwritten=False,
            line_index=0,
        ),
        OCRLine(
            text="Account 123456789012 was debited twice",
            bbox=bbox_from_xyxy(100.0, 240.0, 700.0, 272.0),
            confidence=0.9,
            script="Latn",
            language=None,
            is_handwritten=False,
            line_index=1,
        ),
    ]


@pytest.fixture
def make_ctx() -> Callable[..., ExtractionContext]:
    def _make(**overrides: Any) -> ExtractionContext:
        data: dict[str, Any] = {
            "document_id": "sha256:" + "c" * 64,
            "page_no": 1,
            "lines": _lines(),
            "page_class": PageClassLabel.COMPLAINT_FORM,
            "primary_script": "Latn",
            "primary_language": "eng",
            "target_fields": ["complainant_name"],
            "max_tokens": 256,
        }
        data.update(overrides)
        return ExtractionContext(**data)

    return _make


def _completion(content: str, *, total_tokens: int = 120) -> dict[str, Any]:
    """A minimal OpenAI chat-completion body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": total_tokens // 2,
            "completion_tokens": total_tokens - total_tokens // 2,
            "total_tokens": total_tokens,
        },
    }


class _Recorder:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            return httpx.Response(200, json=_completion('{"fields": []}'))
        return self._responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def body(self, index: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[index].content.decode("utf-8"))


def _ok(content: str, *, total_tokens: int = 120) -> httpx.Response:
    return httpx.Response(200, json=_completion(content, total_tokens=total_tokens))


def _status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"message": "nope"}}, headers=headers
    )


class _Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def build_provider(
    make_settings: Callable[..., Settings],
) -> Iterator[Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]]]:
    """Build a provider wired to a scripted mock transport."""
    created: list[HFAPIProvider] = []

    def _build(
        responses: list[httpx.Response] | None = None,
        **overrides: Any,
    ) -> tuple[HFAPIProvider, _Recorder, _Sleeper]:
        recorder = _Recorder(responses or [])
        sleeper = _Sleeper()
        settings_kwargs = overrides.pop("settings_kwargs", {})
        provider = HFAPIProvider(
            make_settings(**settings_kwargs),
            api_key="test-key",
            cache=overrides.pop("cache", ResponseCache(MEMORY_PATH)),
            http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
            async_http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(recorder)
            ),
            sleep=sleeper,
            rand=lambda: 0.5,
            **overrides,
        )
        created.append(provider)
        return provider, recorder, sleeper

    yield _build
    for provider in created:
        provider.close()


# --------------------------------------------------------------------------- #
# Backoff
# --------------------------------------------------------------------------- #


def test_backoff_is_exponential_without_jitter() -> None:
    policy = RetryPolicy(initial_backoff_seconds=0.5, multiplier=2.0, jitter=0.0)
    delays = [backoff_delay(n, policy) for n in range(4)]
    assert delays == [0.5, 1.0, 2.0, 4.0]


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(
        initial_backoff_seconds=1.0, multiplier=10.0, jitter=0.0,
        max_backoff_seconds=5.0,
    )
    assert backoff_delay(6, policy) == 5.0


def test_jitter_stays_inside_the_declared_band() -> None:
    policy = RetryPolicy(initial_backoff_seconds=1.0, multiplier=1.0, jitter=0.25)
    assert backoff_delay(0, policy, rand=lambda: 0.0) == pytest.approx(0.75)
    assert backoff_delay(0, policy, rand=lambda: 1.0) == pytest.approx(1.25)
    assert backoff_delay(0, policy, rand=lambda: 0.5) == pytest.approx(1.0)


def test_negative_attempt_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="attempt"):
        backoff_delay(-1, RetryPolicy())


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def _openai_error(status: int, **headers: str) -> Exception:
    """Provoke a real ``openai`` exception, so the SDK's own dispatch is used."""
    response = httpx.Response(
        status, json={"error": {"message": "x"}}, headers=headers
    )
    client = openai.OpenAI(
        api_key="k",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _req: response)
        ),
    )
    try:
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
    except Exception as exc:  # noqa: BLE001 - the exception is the fixture
        return exc
    finally:
        client.close()
    raise AssertionError("expected the client to raise")


def test_429_maps_to_rate_limited_with_retry_after() -> None:
    error = classify_openai_error(_openai_error(429, **{"retry-after": "7"}))
    assert isinstance(error, ProviderRateLimited)
    assert error.retry_after == 7.0
    assert error.code == "PROVIDER_RATE_LIMITED"


def test_503_maps_to_unavailable() -> None:
    error = classify_openai_error(_openai_error(503))
    assert isinstance(error, ProviderUnavailable)


def test_400_maps_to_a_plain_provider_error() -> None:
    """A request bug is not retryable and must not be dressed up as one."""
    error = classify_openai_error(_openai_error(400))
    assert type(error) is ProviderError
    assert not isinstance(error, (ProviderRateLimited, ProviderUnavailable))


def test_connection_failure_maps_to_unavailable() -> None:
    request = httpx.Request("POST", "https://example.invalid/v1")
    error = classify_openai_error(
        openai.APIConnectionError(message="boom", request=request)
    )
    assert isinstance(error, ProviderUnavailable)
    assert error.context["kind"] == "connection"


def test_an_unknown_exception_still_maps_into_the_taxonomy() -> None:
    error = classify_openai_error(RuntimeError("surprise"))
    assert isinstance(error, ProviderUnavailable)
    assert error.context["exception"] == "RuntimeError"


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #


def test_lines_render_with_their_indices(
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    rendered = render_lines(make_ctx())
    assert rendered.splitlines()[0].startswith("[0] ")
    assert rendered.splitlines()[1].startswith("[1] ")


def test_masked_spans_are_replaced_before_the_text_leaves(
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """No identifier value may cross the network (invariant 7's egress half)."""
    ctx = make_ctx(masked_spans=[TextSpan(line_index=1, start=8, end=20)])
    rendered = render_lines(ctx)
    assert "123456789012" not in rendered
    assert MASK_SENTINEL_TEMPLATE.format(index=0) in rendered


def test_multiple_masks_on_one_line_are_all_applied(
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    ctx = make_ctx(
        masked_spans=[
            TextSpan(line_index=1, start=0, end=7),
            TextSpan(line_index=1, start=8, end=20),
        ]
    )
    rendered = render_lines(ctx)
    assert "123456789012" not in rendered
    assert "Account" not in rendered
    assert MASK_SENTINEL_TEMPLATE.format(index=0) in rendered
    assert MASK_SENTINEL_TEMPLATE.format(index=1) in rendered


def test_prompt_is_deterministic_regardless_of_field_order(
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """The prompt is the cache key; an unordered set makes every run a miss."""
    a = build_extraction_prompt(
        make_ctx(target_fields=["complainant_name", "relief_sought"])
    )
    b = build_extraction_prompt(
        make_ctx(target_fields=["relief_sought", "complainant_name"])
    )
    assert a == b


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcdef") >= estimate_tokens("abc")


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_a_missing_token_fails_at_construction(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        HFAPIProvider(make_settings(), enable_cache=False)


def test_zero_concurrency_is_refused(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        HFAPIProvider(
            make_settings(), api_key="k", enable_cache=False, max_concurrency=0
        )


def test_local_only_methods_refuse(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
) -> None:
    provider, recorder, _ = build_provider()
    with pytest.raises(ProviderMethodNotSupported):
        provider.classify_page(object(), None)  # type: ignore[arg-type]
    assert recorder.call_count == 0


def test_translation_is_off_unless_explicitly_enabled(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
) -> None:
    """An LLM paraphrase must never be mistaken for IndicTrans2 output."""
    provider, recorder, _ = build_provider()
    with pytest.raises(ProviderMethodNotSupported) as excinfo:
        provider.translate(["एक"], "hin")
    assert "enable_translate" in str(excinfo.value.context["reason"])
    assert recorder.call_count == 0


def test_translation_works_once_enabled(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
) -> None:
    payload = json.dumps({"translations": ["one", "two"]})
    provider, recorder, _ = build_provider([_ok(payload)], enable_translate=True)
    assert provider.translate(["एक", "दो"], "hin", "eng") == ["one", "two"]
    assert recorder.call_count == 1


def test_a_translation_of_the_wrong_arity_is_rejected(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
) -> None:
    """Positions must be preserved; a dropped string is a silent data loss."""
    short = json.dumps({"translations": ["one"]})
    good = json.dumps({"translations": ["one", "two"]})
    provider, recorder, _ = build_provider(
        [_ok(short), _ok(good)], enable_translate=True
    )
    assert provider.translate(["एक", "दो"], "hin") == ["one", "two"]
    assert recorder.call_count == 2


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

_GOOD = json.dumps(
    {
        "fields": [
            {
                "name": "complainant_name",
                "value": "Name: Anita Deshmukh",
                "confidence": 0.8,
                "line_indices": [0],
            }
        ]
    }
)


def test_extract_fields_returns_raw_fields(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD)])
    fields = provider.extract_fields(make_ctx())
    assert set(fields) == {"complainant_name"}
    assert fields["complainant_name"].value == "Name: Anita Deshmukh"
    assert recorder.call_count == 1


def test_json_mode_is_requested(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD)])
    provider.extract_fields(make_ctx())
    assert recorder.body()["response_format"] == {"type": "json_object"}


def test_fields_outside_the_request_are_dropped_not_fatal(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """Volunteered noise must not burn a repair turn."""
    payload = json.dumps(
        {
            "fields": [
                {
                    "name": "complainant_name",
                    "value": "Name: Anita Deshmukh",
                    "confidence": 0.8,
                    "line_indices": [0],
                },
                {
                    "name": "relief_sought",
                    "value": "refund",
                    "confidence": 0.5,
                    "line_indices": [1],
                },
            ]
        }
    )
    provider, recorder, _ = build_provider([_ok(payload)])
    fields = provider.extract_fields(make_ctx())
    assert set(fields) == {"complainant_name"}
    assert recorder.call_count == 1


def test_a_fenced_json_block_is_still_accepted(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(f"```json\n{_GOOD}\n```")])
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 1


def test_a_model_that_finds_nothing_returns_an_empty_mapping(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, _ = build_provider([_ok('{"fields": []}')])
    assert provider.extract_fields(make_ctx()) == {}


# --------------------------------------------------------------------------- #
# Retry and backoff against simulated 429 / 503
# --------------------------------------------------------------------------- #


def test_a_429_is_retried_then_succeeds(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, sleeper = build_provider([_status(429), _ok(_GOOD)])
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 2
    assert sleeper.delays == [0.5]


def test_a_503_is_retried_with_growing_delays(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, sleeper = build_provider(
        [_status(503), _status(503), _ok(_GOOD)]
    )
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 3
    assert sleeper.delays == [0.5, 1.0]


def test_retry_after_wins_over_the_computed_delay(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, sleeper = build_provider(
        [_status(429, **{"retry-after": "4"}), _ok(_GOOD)]
    )
    provider.extract_fields(make_ctx())
    assert sleeper.delays == [4.0]


def test_retry_after_is_capped(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """A server asking for ten minutes must not hold a worker for ten minutes."""
    provider, _, sleeper = build_provider(
        [_status(429, **{"retry-after": "600"}), _ok(_GOOD)],
        retry_policy=RetryPolicy(max_retry_after_seconds=30.0),
    )
    provider.extract_fields(make_ctx())
    assert sleeper.delays == [30.0]


def test_exhausted_retries_raise_the_typed_error(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, sleeper = build_provider(
        [_status(429) for _ in range(3)],
        retry_policy=RetryPolicy(max_attempts=3),
    )
    with pytest.raises(ProviderRateLimited):
        provider.extract_fields(make_ctx())
    assert recorder.call_count == 3
    assert len(sleeper.delays) == 2


def test_a_400_is_never_retried(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, sleeper = build_provider([_status(400)])
    with pytest.raises(ProviderError):
        provider.extract_fields(make_ctx())
    assert recorder.call_count == 1
    assert sleeper.delays == []


def test_a_failed_call_refunds_its_budget_reservation(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, _ = build_provider(
        [_status(400)], budget=JobBudget(8000)
    )
    with pytest.raises(ProviderError):
        provider.extract_fields(make_ctx())
    assert provider.budget.used == 0
    assert provider.budget.remaining == 8000


# --------------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------------- #


def test_a_breach_raises_before_the_request_leaves(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD)], budget=JobBudget(10))
    with pytest.raises(TokenBudgetExceeded):
        provider.extract_fields(make_ctx())
    assert recorder.call_count == 0


def test_usage_from_the_response_settles_the_reservation(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, _ = build_provider(
        [_ok(_GOOD, total_tokens=137)], budget=JobBudget(8000)
    )
    provider.extract_fields(make_ctx())
    assert provider.budget.used == 137
    assert provider.budget.reserved == 0


def test_a_cache_hit_costs_no_budget(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """Why the cache makes Phase 0 iteration feel local."""
    cache = ResponseCache(MEMORY_PATH)
    provider, recorder, _ = build_provider(
        [_ok(_GOOD, total_tokens=200)], cache=cache, budget=JobBudget(8000)
    )
    provider.extract_fields(make_ctx())
    spent = provider.budget.used
    provider.extract_fields(make_ctx())
    assert recorder.call_count == 1
    assert provider.budget.used == spent


def test_reset_budget_starts_a_new_job(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, _ = build_provider([_ok(_GOOD)])
    provider.extract_fields(make_ctx())
    fresh = provider.reset_budget(job_id="job-2")
    assert fresh.used == 0
    assert fresh.job_id == "job-2"


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_the_second_identical_call_never_reaches_the_endpoint(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD)])
    first = provider.extract_fields(make_ctx())
    second = provider.extract_fields(make_ctx())
    assert recorder.call_count == 1
    assert first == second


def test_a_different_page_is_a_different_key(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD), _ok(_GOOD)])
    provider.extract_fields(make_ctx(page_no=1))
    provider.extract_fields(make_ctx(page_no=2))
    assert recorder.call_count == 2


def test_a_stale_cache_entry_is_invalidated_not_served(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """A contract change must surface, not hide behind yesterday's answer."""
    cache = ResponseCache(MEMORY_PATH)
    provider, recorder, _ = build_provider([_ok(_GOOD)], cache=cache)
    ctx = make_ctx()
    provider.extract_fields(ctx)
    # Corrupt the stored payload the way a schema change would.
    key = next(iter(_all_keys(cache)))
    cache.put(key, '{"fields": [{"name": "complainant_name"}]}', model="x")
    provider.extract_fields(ctx)
    assert recorder.call_count == 2


def _all_keys(cache: ResponseCache) -> list[str]:
    with cache._guard() as conn:
        return [row["key"] for row in conn.execute("SELECT key FROM provider_cache")]


def test_caching_can_be_disabled(
    make_settings: Callable[..., Settings],
) -> None:
    provider = HFAPIProvider(
        make_settings(), api_key="k", enable_cache=False
    )
    assert provider.cache is None
    provider.close()


# --------------------------------------------------------------------------- #
# Structured output: validation and the one repair retry
# --------------------------------------------------------------------------- #


def test_malformed_json_gets_exactly_one_repair_turn(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok("not json at all"), _ok(_GOOD)])
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 2
    repair_messages = recorder.body(1)["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert "rejected by the schema validator" in repair_messages[-1]["content"]


def test_two_failures_raise_provider_response_invalid(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok("garbage"), _ok("still garbage")])
    with pytest.raises(ProviderResponseInvalid):
        provider.extract_fields(make_ctx())
    assert recorder.call_count == 2


def test_a_prose_summary_is_rejected_by_the_contract(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """Invariant 2: the model may select lines, never write them."""
    prose = json.dumps(
        {
            "fields": [
                {
                    "name": SUMMARY_FIELD_NAME,
                    "value": "The complainant was debited twice.",
                    "confidence": 0.9,
                    "line_indices": [0, 1],
                }
            ]
        }
    )
    indices = json.dumps(
        {
            "fields": [
                {
                    "name": SUMMARY_FIELD_NAME,
                    "value": None,
                    "confidence": 0.9,
                    "line_indices": [0, 1],
                }
            ]
        }
    )
    provider, recorder, _ = build_provider([_ok(prose), _ok(indices)])
    fields = provider.extract_fields(
        make_ctx(target_fields=[SUMMARY_FIELD_NAME])
    )
    assert recorder.call_count == 2
    assert fields[SUMMARY_FIELD_NAME].value is None
    assert fields[SUMMARY_FIELD_NAME].line_indices == [0, 1]


def test_persistent_prose_is_refused_outright(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    prose = json.dumps(
        {
            "fields": [
                {
                    "name": SUMMARY_FIELD_NAME,
                    "value": "A summary.",
                    "confidence": 0.9,
                    "line_indices": [0],
                }
            ]
        }
    )
    provider, _, _ = build_provider([_ok(prose), _ok(prose)])
    with pytest.raises(ProviderResponseInvalid):
        provider.extract_fields(make_ctx(target_fields=[SUMMARY_FIELD_NAME]))


def test_a_hallucinated_line_index_is_rejected(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    bad = json.dumps(
        {
            "fields": [
                {
                    "name": "complainant_name",
                    "value": "Name: Anita Deshmukh",
                    "confidence": 0.8,
                    "line_indices": [99],
                }
            ]
        }
    )
    provider, recorder, _ = build_provider([_ok(bad), _ok(_GOOD)])
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 2


def test_an_empty_string_value_is_rejected(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """"Omit a field you cannot find" — never an empty stand-in."""
    bad = json.dumps(
        {
            "fields": [
                {
                    "name": "complainant_name",
                    "value": "",
                    "confidence": 0.1,
                    "line_indices": [],
                }
            ]
        }
    )
    provider, recorder, _ = build_provider([_ok(bad), _ok(_GOOD)])
    assert provider.extract_fields(make_ctx())
    assert recorder.call_count == 2


def test_an_empty_response_body_is_a_typed_failure(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    empty = httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [],
        },
    )
    provider, _, _ = build_provider([empty])
    with pytest.raises(ProviderResponseInvalid):
        provider.extract_fields(make_ctx())


# --------------------------------------------------------------------------- #
# Batch concurrency
# --------------------------------------------------------------------------- #


def test_batch_returns_one_result_per_context_in_order(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD) for _ in range(3)])
    ctxs = [make_ctx(page_no=n) for n in (1, 2, 3)]
    results = provider.extract_fields_batch(ctxs)
    assert len(results) == 3
    assert recorder.call_count == 3
    assert all(set(result) == {"complainant_name"} for result in results)


def test_batch_of_one_takes_the_sequential_path(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, recorder, _ = build_provider([_ok(_GOOD)])
    assert len(provider.extract_fields_batch([make_ctx()])) == 1
    assert recorder.call_count == 1


def test_batch_is_empty_for_no_contexts(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
) -> None:
    provider, recorder, _ = build_provider()
    assert provider.extract_fields_batch([]) == []
    assert recorder.call_count == 0


def test_a_batch_failure_propagates_rather_than_half_filling(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """A half-filled list would let the assembler publish a complete-looking
    envelope."""
    provider, _, _ = build_provider(
        [_ok(_GOOD), _status(400), _ok(_GOOD)], max_concurrency=1
    )
    with pytest.raises(ProviderError):
        provider.extract_fields_batch(
            [make_ctx(page_no=n) for n in (1, 2, 3)]
        )


def test_batch_shares_one_budget_across_the_gather(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    provider, _, _ = build_provider(
        [_ok(_GOOD, total_tokens=100) for _ in range(3)], budget=JobBudget(8000)
    )
    provider.extract_fields_batch([make_ctx(page_no=n) for n in (1, 2, 3)])
    assert provider.budget.used == 300


def test_batch_works_from_inside_a_running_event_loop(
    build_provider: Callable[..., tuple[HFAPIProvider, _Recorder, _Sleeper]],
    make_ctx: Callable[..., ExtractionContext],
) -> None:
    """A FastAPI handler must not have to know this provider uses asyncio."""
    import asyncio

    provider, _, _ = build_provider([_ok(_GOOD) for _ in range(2)])

    async def caller() -> list[dict[str, Any]]:
        return provider.extract_fields_batch(
            [make_ctx(page_no=n) for n in (1, 2)]
        )

    assert len(asyncio.run(caller())) == 2
