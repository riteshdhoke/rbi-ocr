"""Unit tests for :mod:`app.providers.factory` — the two gates and routing.

The residency test is the one BUILD-BRIEF sec. 9 calls out by name: the failure
this system cannot recover from is complainant data leaving the country during
development, and these assertions are the only thing standing in front of it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.contracts.errors import ConfigurationError, ResidencyViolation
from app.providers.base import ModelProvider
from app.providers.cache import MEMORY_PATH, ResponseCache
from app.providers.factory import (
    DIRECTORY_MANIFEST_FILENAME,
    PROVIDER_PROFILE_GAUGE,
    SIDECAR_SUFFIX,
    CompositeProvider,
    build_cpu_node_provider,
    build_gpu_node_provider,
    build_laptop_provider,
    close_provider,
    get_provider,
    is_declared_synthetic,
    load_corpus_file,
    log_startup_banner,
    require_synthetic_corpus_file,
    set_provider_profile_metric,
    synthetic_sidecar_for,
)
from app.providers.local import LocalProvider
from app.settings import ProviderProfile, Settings

# --------------------------------------------------------------------------- #
# The residency gate
# --------------------------------------------------------------------------- #


def test_laptop_without_synthetic_only_refuses(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.LAPTOP, SYNTHETIC_ONLY=False
    )
    with pytest.raises(ResidencyViolation) as excinfo:
        get_provider(settings, log_banner=False)
    error = excinfo.value
    assert error.code == "RESIDENCY_VIOLATION"
    assert error.context["synthetic_only"] is False
    assert "SYNTHETIC_ONLY" in str(error.context["remedy"])


def test_the_gate_message_is_the_one_settings_publishes(
    make_settings: Callable[..., Settings],
) -> None:
    """Written once in WS-B's lane so the gate and its test cannot drift."""
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.LAPTOP, SYNTHETIC_ONLY=False
    )
    with pytest.raises(ResidencyViolation) as excinfo:
        get_provider(settings, log_banner=False)
    assert excinfo.value.message == settings.residency_violation_reason


def test_the_gate_runs_before_any_construction(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A violating configuration must not open a socket on its way to refusal."""
    import app.providers.factory as factory_module

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a provider was constructed despite the gate")

    registry: dict[ProviderProfile, Any] = dict(factory_module._REGISTRY)
    registry[ProviderProfile.LAPTOP] = explode
    monkeypatch.setattr(factory_module, "_REGISTRY", registry)
    with pytest.raises(ResidencyViolation):
        get_provider(
            make_settings(
                PROVIDER_PROFILE=ProviderProfile.LAPTOP, SYNTHETIC_ONLY=False
            ),
            log_banner=False,
        )


@pytest.mark.parametrize(
    "profile", [ProviderProfile.CPU_NODE, ProviderProfile.GPU_NODE]
)
def test_non_remote_profiles_may_process_real_data(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
    profile: ProviderProfile,
) -> None:
    """The gate is about *where the text goes*, not about being strict.

    ``gpu_node`` is given an on-premises ``HF_BASE_URL`` because that is what
    makes it an air-gapped profile.  Pointing it at the public router instead
    is the case ``test_gpu_node_refuses_a_public_endpoint`` covers.
    """
    settings = make_settings(
        PROVIDER_PROFILE=profile,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
        HF_BASE_URL="http://slm.internal:8000/v1",
        HF_TOKEN="token-for-the-in-house-endpoint",
    )
    provider = get_provider(
        settings, cache=ResponseCache(MEMORY_PATH), log_banner=False
    )
    assert isinstance(provider, ModelProvider)
    close_provider(provider)


def test_gpu_node_refuses_a_public_endpoint(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """An air-gapped profile pointed off-premises is a residency breach.

    The profile name states an intent; ``HF_BASE_URL`` is the fact.  Before
    this gate existed a ``gpu_node`` that kept the default base URL sent real
    complainant text to the public HF router and reported nothing at all —
    ``is_remote_profile`` was false, so no check ran.
    """
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.GPU_NODE,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
        HF_BASE_URL="https://router.huggingface.co/v1/",
        HF_TOKEN="a-real-looking-token",
    )
    with pytest.raises(ResidencyViolation, match="not an on-premises host"):
        get_provider(settings, cache=ResponseCache(MEMORY_PATH), log_banner=False)


def test_an_unconfigured_gpu_node_runs_local_only(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """No serving stack on the box is a supported shape, not a failure.

    It is also the safe one: with nothing configured the node never constructs
    a client at all, so the default (public) base URL cannot be reached even by
    accident. Field extraction falls back to the deterministic validators.
    """
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.GPU_NODE,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
    )
    provider = get_provider(
        settings, cache=ResponseCache(MEMORY_PATH), log_banner=False
    )
    assert isinstance(provider, LocalProvider)
    close_provider(provider)


def test_cpu_node_ignores_the_base_url(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """``cpu_node`` never builds a client, so its base URL is inert.

    Judging it would refuse every air-gapped CPU deployment that left the
    shipped ``.env`` default in place, which is the common case.
    """
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.CPU_NODE,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
    )
    assert settings.HF_BASE_URL == "https://router.huggingface.co/v1"
    assert settings.residency_violation_reason is None
    provider = get_provider(
        settings, cache=ResponseCache(MEMORY_PATH), log_banner=False
    )
    close_provider(provider)


def test_gpu_node_needs_no_token_for_a_local_endpoint(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """An in-house model server authenticates nothing.

    Requiring ``HF_TOKEN`` here is what turned an air-gapped GPU node into a
    startup failure: the operator removed the token precisely *because* the
    node has no internet, and the service then refused to start.
    """
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.GPU_NODE,
        SYNTHETIC_ONLY=False,
        MODELS_DIR=tmp_path,
        HF_BASE_URL="http://localhost:8000/v1",
        HF_TOKEN=None,
    )
    provider = get_provider(
        settings, cache=ResponseCache(MEMORY_PATH), log_banner=False
    )
    assert isinstance(provider, CompositeProvider)
    remote: Any = provider.remote
    assert remote.endpoint_is_local is True
    close_provider(provider)


# --------------------------------------------------------------------------- #
# Profile composition
# --------------------------------------------------------------------------- #


def test_registry_covers_every_profile() -> None:
    """A new profile cannot be added without a builder."""
    import app.providers.factory as factory_module

    assert set(factory_module._REGISTRY) == set(ProviderProfile)


def test_cpu_node_is_local_only(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    provider = build_cpu_node_provider(
        make_settings(PROVIDER_PROFILE=ProviderProfile.CPU_NODE, MODELS_DIR=tmp_path)
    )
    assert isinstance(provider, LocalProvider)


def test_cpu_node_needs_no_token(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """An air-gapped node must build with an empty .env."""
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.CPU_NODE, MODELS_DIR=tmp_path
    )
    assert settings.HF_TOKEN is None
    assert get_provider(settings, log_banner=False) is not None


def test_laptop_is_composite_with_the_declared_split(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(MODELS_DIR=tmp_path, HF_TOKEN="hf_dummy")
    provider = build_laptop_provider(settings, cache=ResponseCache(MEMORY_PATH))
    assert isinstance(provider, CompositeProvider)
    assert isinstance(provider.local, LocalProvider)
    assert type(provider.remote).__name__ == "HFAPIProvider"
    assert CompositeProvider.REMOTE_METHODS == {
        "extract_fields",
        "extract_fields_batch",
        "ocr_page_heavy",
    }
    close_provider(provider)


def test_gpu_node_is_composite_too(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """The base URL is the only difference; that is the point of M2."""
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.GPU_NODE,
        MODELS_DIR=tmp_path,
        HF_TOKEN="hf_dummy",
        HF_BASE_URL="http://slm.internal:8000/v1",
    )
    provider = build_gpu_node_provider(settings, cache=ResponseCache(MEMORY_PATH))
    assert isinstance(provider, CompositeProvider)
    remote: Any = provider.remote
    assert remote.base_url == "http://slm.internal:8000/v1"
    close_provider(provider)


def test_laptop_without_a_token_fails_loudly_at_startup(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """Not on the first extraction — at startup, where an operator sees it."""
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        build_laptop_provider(
            make_settings(MODELS_DIR=tmp_path), cache=ResponseCache(MEMORY_PATH)
        )


def test_close_provider_is_safe_on_anything() -> None:
    close_provider(object())


# --------------------------------------------------------------------------- #
# Banner and metric
# --------------------------------------------------------------------------- #


def test_banner_names_the_profile_and_hides_the_token(
    make_settings: Callable[..., Settings], caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(HF_TOKEN="hf_super_secret_value")
    with caplog.at_level(logging.INFO):
        banner = log_startup_banner(settings)
    assert "profile=laptop" in banner
    assert "hf_token=set" in banner
    assert "hf_super_secret_value" not in caplog.text
    assert "non-Indian endpoint" in caplog.text


def test_a_stray_credential_on_an_air_gapped_node_is_warned_about(
    make_settings: Callable[..., Settings], caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.CPU_NODE, HF_TOKEN="hf_dummy"
    )
    with caplog.at_level(logging.WARNING):
        log_startup_banner(settings)
    assert "never calls a remote endpoint" in caplog.text


def test_the_gauge_marks_exactly_one_profile_active() -> None:
    set_provider_profile_metric(ProviderProfile.CPU_NODE)
    values = {
        profile.value: PROVIDER_PROFILE_GAUGE.labels(
            profile=profile.value
        )._value.get()
        for profile in ProviderProfile
    }
    assert values["cpu_node"] == 1.0
    assert values["laptop"] == 0.0
    assert values["gpu_node"] == 0.0


# --------------------------------------------------------------------------- #
# The synthetic-corpus gate
# --------------------------------------------------------------------------- #


def _write_document(directory: Path, name: str = "page.png") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_bytes(b"\x89PNG\r\n\x1a\n")
    return target


def _write_sidecar(document: Path, payload: dict[str, Any]) -> Path:
    sidecar = synthetic_sidecar_for(document)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def test_sidecar_path_follows_the_generator_convention() -> None:
    assert synthetic_sidecar_for(Path("a/b/doc_v0_clean.jpg")).name == (
        "doc_v0_clean" + SIDECAR_SUFFIX
    )


def test_a_declared_synthetic_file_loads(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    document = _write_document(tmp_path)
    _write_sidecar(document, {"synthetic": True, "doc_id": "x"})
    assert load_corpus_file(document, make_settings()) == b"\x89PNG\r\n\x1a\n"


def test_an_undeclared_file_is_refused_under_the_laptop_profile(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    document = _write_document(tmp_path)
    with pytest.raises(ResidencyViolation) as excinfo:
        require_synthetic_corpus_file(document, make_settings())
    assert "no bypass flag" in excinfo.value.message.lower()
    assert excinfo.value.context["expected_sidecar"].endswith(SIDECAR_SUFFIX)


def test_synthetic_false_is_a_refusal_not_a_pass(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    document = _write_document(tmp_path)
    _write_sidecar(document, {"synthetic": False})
    with pytest.raises(ResidencyViolation):
        require_synthetic_corpus_file(document, make_settings())


@pytest.mark.parametrize(
    "payload",
    ["{ not json", '"synthetic"', "[]", '{"synthetic": "true"}'],
)
def test_a_sidecar_that_does_not_clearly_say_true_is_a_refusal(
    make_settings: Callable[..., Settings], tmp_path: Path, payload: str
) -> None:
    """Failing open on a parse error is exactly how a real scan gets through."""
    document = _write_document(tmp_path)
    synthetic_sidecar_for(document).write_text(payload, encoding="utf-8")
    with pytest.raises(ResidencyViolation):
        require_synthetic_corpus_file(document, make_settings())


def test_the_adversarial_directory_manifest_is_accepted(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """Those inputs cannot carry per-file ground truth by construction."""
    document = _write_document(tmp_path / "adversarial", "zero_byte.pdf")
    (tmp_path / "adversarial" / DIRECTORY_MANIFEST_FILENAME).write_text(
        json.dumps({"synthetic": True, "purpose": "ingestion hardening"}),
        encoding="utf-8",
    )
    assert require_synthetic_corpus_file(document, make_settings()) == document


def test_a_manifest_does_not_bless_a_sibling_directory(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    (tmp_path / "blessed").mkdir()
    (tmp_path / "blessed" / DIRECTORY_MANIFEST_FILENAME).write_text(
        json.dumps({"synthetic": True}), encoding="utf-8"
    )
    document = _write_document(tmp_path / "elsewhere")
    with pytest.raises(ResidencyViolation):
        require_synthetic_corpus_file(document, make_settings())


def test_an_air_gapped_node_needs_no_sidecar(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    """Processing real complaints is what that profile is for."""
    document = _write_document(tmp_path)
    settings = make_settings(
        PROVIDER_PROFILE=ProviderProfile.CPU_NODE, SYNTHETIC_ONLY=False
    )
    assert require_synthetic_corpus_file(document, settings) == document


def test_a_missing_file_says_so(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        require_synthetic_corpus_file(tmp_path / "absent.png", make_settings())


def test_a_directory_is_not_a_corpus_file(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        require_synthetic_corpus_file(tmp_path, make_settings())


def test_is_declared_synthetic_reports_which_sidecar_answered(
    tmp_path: Path,
) -> None:
    document = _write_document(tmp_path)
    sidecar = _write_sidecar(document, {"synthetic": True})
    declared, source = is_declared_synthetic(document)
    assert declared is True
    assert source == sidecar


def test_the_committed_corpus_passes_the_gate(
    make_settings: Callable[..., Settings], project_root: Path
) -> None:
    """The gate must accept what WS-D actually generated, not a theory of it."""
    corpus = project_root / "corpus"
    if not corpus.is_dir():
        pytest.skip("corpus/ is not present")
        return
    samples = sorted(corpus.glob("*/*.jpg"))[:3] + sorted(
        corpus.glob("*/*.png")
    )[:3]
    if not samples:
        pytest.skip("no corpus images committed yet")
        return
    settings = make_settings()
    for sample in samples:
        assert require_synthetic_corpus_file(sample, settings) == sample
