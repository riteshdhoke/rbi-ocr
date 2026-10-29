"""The provider contract — asserted against every implementation we can run.

BUILD-BRIEF sec. 7.3 wants "a contract test that runs under every applicable
provider profile".  Applicable, offline, is:

* :class:`MockProvider` — the full behavioural contract, since it can answer
  every method;
* :class:`~app.providers.local.LocalProvider` — the shape contract plus the
  typed-refusal contract, since it has no weights on this machine;
* :class:`~app.providers.hf_api.HFAPIProvider` — the shape contract plus the
  local-only refusals, since its remote methods need a network;
* :class:`~app.providers.factory.CompositeProvider` — the shape contract plus
  routing.

The rule the suite exists to enforce: **every method either does the work or
raises a typed** :class:`~app.contracts.errors.ProviderError`.  ``TypeError``,
``AttributeError``, ``NotImplementedError`` and a plausible-looking default are
all contract violations (BUILD-BRIEF sec. 0.5).

This module does NOT handle: throughput, model accuracy, or anything requiring
a weight file, a token or a socket.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mock_provider import MockProvider

from app.contracts.errors import (
    ProviderError,
    ResidencyViolation,
    TokenBudgetExceeded,
)
from app.contracts.ocr import (
    SUMMARY_FIELD_NAME,
    ExtractionContext,
    LangResult,
    OCRLine,
    PageClass,
    RawField,
    ScriptResult,
)
from app.providers.base import (
    PROVIDER_METHODS,
    JobBudget,
    ModelArtifactMissing,
    ModelProvider,
    ProviderMethodNotSupported,
)
from app.providers.cache import MEMORY_PATH, ResponseCache, make_cache_key
from app.providers.factory import CompositeProvider, get_provider
from app.providers.hf_api import HFAPIProvider
from app.providers.local import LocalProvider
from app.settings import ProviderProfile, Settings

# --------------------------------------------------------------------------- #
# Provider construction, offline
# --------------------------------------------------------------------------- #


def _empty_models_settings(
    make_settings: Callable[..., Settings], tmp: Path
) -> Settings:
    return make_settings(MODELS_DIR=tmp / "no-such-models")


@pytest.fixture
def offline_providers(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> dict[str, Any]:
    """One instance of every provider class, built without weights or network."""
    settings = _empty_models_settings(make_settings, tmp_path)
    local = LocalProvider(settings)
    remote = HFAPIProvider(
        settings,
        api_key="test-key-not-a-real-token",
        cache=ResponseCache(MEMORY_PATH),
    )
    return {
        "mock": MockProvider(),
        "local": local,
        "hf_api": remote,
        "composite": CompositeProvider(local=local, remote=remote),
    }


# --------------------------------------------------------------------------- #
# Shape: the eight methods, on every implementation
# --------------------------------------------------------------------------- #


def test_protocol_lists_exactly_the_eight_brief_methods() -> None:
    """``PROVIDER_METHODS`` and the Protocol may not drift apart."""
    declared = {
        name
        for name in dir(ModelProvider)
        if not name.startswith("_") and callable(getattr(ModelProvider, name))
    }
    assert declared == set(PROVIDER_METHODS)
    assert len(PROVIDER_METHODS) == 8


@pytest.mark.parametrize(
    "key", ["mock", "local", "hf_api", "composite"]
)
def test_every_provider_satisfies_the_protocol(
    offline_providers: dict[str, Any], key: str
) -> None:
    """Runtime protocol check plus matching parameter names per method."""
    provider = offline_providers[key]
    assert isinstance(provider, ModelProvider)
    for method_name in PROVIDER_METHODS:
        expected = list(
            inspect.signature(getattr(ModelProvider, method_name)).parameters
        )
        actual = ["self", *inspect.signature(
            getattr(provider, method_name)
        ).parameters]
        assert actual == expected, (
            f"{type(provider).__name__}.{method_name} takes {actual}, the "
            f"protocol declares {expected}"
        )


# --------------------------------------------------------------------------- #
# Behaviour: the mock answers the full contract
# --------------------------------------------------------------------------- #


def test_mock_returns_contract_types(
    mock_provider: MockProvider,
    make_extraction_context: Callable[..., ExtractionContext],
    structural_features: Any,
) -> None:
    """Every answer is a real contract object, not a lookalike dict."""
    assert isinstance(mock_provider.detect_script([object()]), ScriptResult)
    assert isinstance(
        mock_provider.classify_page(object(), structural_features), PageClass
    )
    lines = mock_provider.ocr_page(object(), "Deva")
    assert lines and all(isinstance(line, OCRLine) for line in lines)
    assert all(line.script == "Deva" for line in lines)
    assert isinstance(mock_provider.ocr_page_heavy(object())[0], OCRLine)
    assert isinstance(mock_provider.identify_language("hello", "Latn"), LangResult)
    fields = mock_provider.extract_fields(make_extraction_context())
    assert all(isinstance(value, RawField) for value in fields.values())


def test_ocr_lines_never_carry_a_language_before_stage_2b(
    mock_provider: MockProvider,
) -> None:
    """Invariant 1: ``language`` is Stage 2b's, not the OCR provider's."""
    for line in mock_provider.ocr_page(object(), "Latn"):
        assert line.language is None
    for line in mock_provider.ocr_page_heavy(object()):
        assert line.language is None


def test_ocr_line_indices_are_unique_and_ordered(
    mock_provider: MockProvider,
) -> None:
    """``line_index`` is the handle the extractive summary selects by."""
    indices = [line.line_index for line in mock_provider.ocr_page(object(), "Latn")]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)


def test_summary_comes_back_as_indices_not_prose(
    mock_provider: MockProvider,
    make_extraction_context: Callable[..., ExtractionContext],
) -> None:
    """Invariant 2, enforced at the contract boundary."""
    fields = mock_provider.extract_fields(
        make_extraction_context(target_fields=[SUMMARY_FIELD_NAME])
    )
    summary = fields[SUMMARY_FIELD_NAME]
    assert summary.value is None
    assert summary.line_indices


def test_batch_preserves_order_and_arity(
    mock_provider: MockProvider,
    make_extraction_context: Callable[..., ExtractionContext],
) -> None:
    """``extract_fields_batch`` is positional: result *i* belongs to context *i*."""
    ctxs = [make_extraction_context(page_no=n) for n in (1, 2, 3)]
    results = mock_provider.extract_fields_batch(ctxs)
    assert len(results) == len(ctxs)
    seen = [page for _, page in mock_provider.method_calls("extract_fields")]
    assert seen == [1, 2, 3]


def test_batch_of_zero_contexts_is_empty_not_an_error(
    mock_provider: MockProvider,
) -> None:
    """A page set that produced nothing must not raise on the way out."""
    assert mock_provider.extract_fields_batch([]) == []


def test_translate_preserves_positions(mock_provider: MockProvider) -> None:
    """Per-field MT is positional; an empty input stays empty in place."""
    out = mock_provider.translate(["एक", "", "दो"], "hin", "eng")
    assert len(out) == 3
    assert out[1] == ""


# --------------------------------------------------------------------------- #
# Behaviour: refusals are typed on the real classes
# --------------------------------------------------------------------------- #


def test_local_provider_refuses_slm_methods_typed(
    offline_providers: dict[str, Any],
    make_extraction_context: Callable[..., ExtractionContext],
) -> None:
    """The SLM/VLM are remote in Phase 0; the refusal names who implements them."""
    local = offline_providers["local"]
    for call in (
        lambda: local.extract_fields(make_extraction_context()),
        lambda: local.extract_fields_batch([make_extraction_context()]),
        lambda: local.ocr_page_heavy(object()),
    ):
        with pytest.raises(ProviderMethodNotSupported) as excinfo:
            call()
        assert "hf_api" in excinfo.value.implemented_by


def test_hf_api_refuses_local_only_methods_typed(
    offline_providers: dict[str, Any],
) -> None:
    """Page pixels never go to a remote endpoint for work a CPU model does."""
    remote = offline_providers["hf_api"]
    for call in (
        lambda: remote.detect_script([object()]),
        lambda: remote.ocr_page(object(), "Latn"),
        lambda: remote.identify_language("text", "Latn"),
    ):
        with pytest.raises(ProviderMethodNotSupported) as excinfo:
            call()
        assert "LocalProvider" in excinfo.value.implemented_by


def test_local_provider_reports_missing_weights_with_a_remedy(
    offline_providers: dict[str, Any],
) -> None:
    """A fresh air-gapped node must be told the path and how to fix it."""
    local = offline_providers["local"]
    with pytest.raises(ModelArtifactMissing) as excinfo:
        local.identify_language("hello", "Latn")
    error = excinfo.value
    assert error.path.name.endswith(".ftz")
    assert error.how_to_obtain
    assert str(error.path) in str(error)
    assert isinstance(error, ProviderError)


@pytest.mark.parametrize("key", ["local", "hf_api", "composite"])
def test_no_method_raises_an_untyped_error(
    offline_providers: dict[str, Any],
    key: str,
    make_extraction_context: Callable[..., ExtractionContext],
    structural_features: Any,
) -> None:
    """Whatever a provider cannot do, it says so in this taxonomy.

    The remote *work* methods are skipped for the API client because they need
    a socket; everything reachable offline must fail typed or not at all.
    """
    provider = offline_providers[key]
    remote_only = {"extract_fields", "extract_fields_batch", "ocr_page_heavy"}
    calls: dict[str, Callable[[], object]] = {
        "detect_script": lambda: provider.detect_script([object()]),
        "classify_page": lambda: provider.classify_page(
            object(), structural_features
        ),
        "ocr_page": lambda: provider.ocr_page(object(), "Latn"),
        "ocr_page_heavy": lambda: provider.ocr_page_heavy(object()),
        "identify_language": lambda: provider.identify_language("hi", "Latn"),
        "extract_fields": lambda: provider.extract_fields(
            make_extraction_context()
        ),
        "extract_fields_batch": lambda: provider.extract_fields_batch(
            [make_extraction_context()]
        ),
        "translate": lambda: provider.translate(["एक"], "hin", "eng"),
    }
    for name, call in calls.items():
        if key in {"hf_api", "composite"} and name in remote_only:
            continue
        try:
            call()
        except ProviderError:
            continue
        except Exception as exc:  # noqa: BLE001 - the assertion is the point
            pytest.fail(
                f"{type(provider).__name__}.{name} raised untyped "
                f"{type(exc).__name__}: {exc}"
            )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_composite_routes_exactly_the_remote_methods() -> None:
    """The audit surface: three methods remote, five local, no fallthrough."""
    local = MockProvider(script="Latn")
    remote = MockProvider(script="Deva")
    composite = CompositeProvider(local=local, remote=remote)

    composite.detect_script([object()])
    composite.classify_page(object(), _features())
    composite.ocr_page(object(), "Latn")
    composite.identify_language("x", "Latn")
    composite.translate(["x"], "hin")
    composite.ocr_page_heavy(object())

    local_methods = {name for name, _ in local.calls}
    remote_methods = {name for name, _ in remote.calls}
    assert remote_methods <= CompositeProvider.REMOTE_METHODS
    assert not (local_methods & CompositeProvider.REMOTE_METHODS)
    assert local_methods | remote_methods <= set(PROVIDER_METHODS)


def _features() -> Any:
    from app.contracts.ocr import StructuralFeatures

    return StructuralFeatures(
        line_count=3,
        ink_density=0.1,
        aspect_ratio=0.7,
        has_table_lines=False,
        colour_histogram=[1.0],
    )


# --------------------------------------------------------------------------- #
# The gates and the budget — BUILD-BRIEF M2's named contract tests
# --------------------------------------------------------------------------- #


def test_laptop_profile_without_synthetic_only_refuses_to_start(
    make_settings: Callable[..., Settings],
) -> None:
    """BUILD-BRIEF M2's headline test: the app refuses to start."""
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.LAPTOP, SYNTHETIC_ONLY=False
    )
    with pytest.raises(ResidencyViolation) as excinfo:
        get_provider(settings, log_banner=False)
    assert excinfo.value.message == settings.residency_violation_reason
    assert excinfo.value.context["profile"] == "laptop"


def test_a_compliant_profile_still_builds(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """The gate refuses one configuration, not every configuration."""
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.CPU_NODE,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
    )
    provider = get_provider(settings, log_banner=False)
    assert isinstance(provider, ModelProvider)


def test_token_budget_breach_is_typed_and_pre_flight() -> None:
    """A breach raises before the request, and names budget and usage."""
    budget = JobBudget(100, job_id="job-1")
    budget.reserve(60).settle(60)
    with pytest.raises(TokenBudgetExceeded) as excinfo:
        budget.reserve(50)
    assert excinfo.value.budget == 100
    assert budget.used == 60


def test_cache_key_versioning_is_part_of_the_contract() -> None:
    """A schema bump must miss, not silently serve differently-framed output."""
    args = {"model": "m", "prompt": "p", "params": {"temperature": 0.0}}
    assert make_cache_key(**args, schema_version="1") != make_cache_key(
        **args, schema_version="2"
    )
    with ResponseCache(MEMORY_PATH, schema_version="1") as v1:
        key = make_cache_key(**args, schema_version="1")
        v1.put(key, "cached", model="m")
        assert v1.get(key) == "cached"


# --------------------------------------------------------------------------- #
# The rule that keeps concurrency inside the provider
# --------------------------------------------------------------------------- #


def test_stage_modules_never_import_asyncio(project_root: Path) -> None:
    """BUILD-BRIEF sec. 3.1: ``extract_fields_batch`` owns the concurrency.

    A forward guard — ``app/stages`` is WS-E's and may not exist yet.  When it
    does, an ``asyncio`` import there means the call site started making the
    decision the provider is supposed to own.
    """
    stages = project_root / "app" / "stages"
    if not stages.is_dir():
        pytest.skip("app/stages does not exist yet (WS-E)")
    offenders = [
        path.name
        for path in sorted(stages.glob("*.py"))
        if "import asyncio" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"stage module(s) {offenders} import asyncio; concurrency belongs to "
        "ModelProvider.extract_fields_batch"
    )
