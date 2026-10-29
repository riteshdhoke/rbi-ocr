"""Tests for :mod:`app.settings` and the thread-cap preamble in :mod:`app`.

Two subjects, both configuration:

* :class:`Settings` — every ``.env.example`` knob present, typed and bounded;
  the profile is an enum; the residency *state* is exposed but **not** enforced
  here (that is WS-C's factory, BUILD-BRIEF M2);
* :mod:`app`'s thread caps — PLAN.md invariant 5.  Because the caps are applied
  at import time and this process has already imported ``app``, those tests run
  a clean subprocess.  No network, no ``HF_TOKEN``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

import app
from app.settings import (
    PROJECT_ROOT,
    ProviderProfile,
    Settings,
    get_settings,
    is_local_endpoint,
    reset_settings_cache,
)


def _env_example_keys(root: Path) -> list[str]:
    """Parse ``.env.example`` into its declared variable names."""
    keys: list[str] = []
    for raw in (root / ".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0].strip())
    return keys


# --------------------------------------------------------------------------- #
# Coverage of the documented environment
# --------------------------------------------------------------------------- #


def test_every_env_example_knob_has_a_settings_field(project_root: Path) -> None:
    declared = _env_example_keys(project_root)
    assert declared, "sanity: .env.example is not empty"
    missing = [key for key in declared if key not in Settings.model_fields]
    assert not missing, f".env.example knobs with no Settings field: {missing}"


def test_env_example_loads_and_validates(
    clean_env: pytest.MonkeyPatch, project_root: Path
) -> None:
    """The file we ship developers must itself produce a valid configuration."""
    settings = Settings(_env_file=project_root / ".env.example")
    assert settings.PROVIDER_PROFILE is ProviderProfile.LAPTOP
    assert settings.SYNTHETIC_ONLY is True
    assert settings.OCR_BATCH_SIZE == 4  # A-10: measured optimum, brief said 16-32
    assert settings.MAX_PAGES_PER_DOC == 200
    assert settings.MAX_DECOMPRESSION_RATIO == 100
    assert settings.ESCALATION_SHADOW_RATE == pytest.approx(0.03)
    assert settings.QUALITY_BOUNCE_THRESHOLD == pytest.approx(0.35)


def test_defaults_match_the_brief(settings: Settings) -> None:
    """BUILD-BRIEF sec. 6 values, so an unset environment is still correct."""
    assert settings.PROVIDER_PROFILE is ProviderProfile.LAPTOP
    assert settings.SYNTHETIC_ONLY is True
    assert settings.HF_BASE_URL == "https://router.huggingface.co/v1"
    assert settings.SLM_MAX_TOKENS == 512
    assert settings.SLM_TEMPERATURE == 0.0
    assert settings.JOB_TOKEN_BUDGET == 8000
    assert settings.OMP_NUM_THREADS == 2
    assert settings.MKL_NUM_THREADS == 2
    assert settings.ORT_INTRA_OP_THREADS == 2
    assert settings.CELERY_CONCURRENCY == 2
    assert settings.OCR_BATCH_SIZE == 4  # A-10: measured optimum, brief said 16-32
    assert settings.REC_HEAD_CACHE_SIZE == 3
    assert settings.GROUNDING_MAX_EDIT_DISTANCE == 2
    assert settings.CODE_MIX_THRESHOLD == pytest.approx(0.15)


def test_values_are_read_from_the_environment(
    make_settings: Callable[..., Settings], clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv("PROVIDER_PROFILE", "cpu_node")
    clean_env.setenv("OCR_BATCH_SIZE", "32")
    clean_env.setenv("SYNTHETIC_ONLY", "false")
    settings = make_settings()
    assert settings.PROVIDER_PROFILE is ProviderProfile.CPU_NODE
    assert settings.OCR_BATCH_SIZE == 32
    assert settings.SYNTHETIC_ONLY is False


def test_unknown_environment_variables_are_ignored(
    make_settings: Callable[..., Settings], clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv("TOTALLY_UNRELATED", "1")
    assert make_settings().PROVIDER_PROFILE is ProviderProfile.LAPTOP


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", ["laptop", "cpu_node", "gpu_node"])
def test_profile_enum_accepts_the_three_profiles(
    make_settings: Callable[..., Settings], profile: str
) -> None:
    assert make_settings(PROVIDER_PROFILE=profile).PROVIDER_PROFILE == profile


@pytest.mark.parametrize("bad", ["LAPTOP", "desktop", "gpu", "", "cpu-node"])
def test_unknown_profile_is_rejected(
    make_settings: Callable[..., Settings], bad: str
) -> None:
    with pytest.raises(ValidationError):
        make_settings(PROVIDER_PROFILE=bad)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("SLM_MAX_TOKENS", 0),
        ("SLM_MAX_TOKENS", -1),
        ("SLM_TEMPERATURE", -0.1),
        ("SLM_TEMPERATURE", 2.5),
        ("JOB_TOKEN_BUDGET", 0),
        ("ORT_INTRA_OP_THREADS", 0),
        ("ORT_INTRA_OP_THREADS", 99),
        ("OCR_BATCH_SIZE", 0),
        ("OCR_BATCH_SIZE", 1000),
        ("REC_HEAD_CACHE_SIZE", 0),
        ("ESCALATION_SHADOW_RATE", -0.01),
        ("ESCALATION_SHADOW_RATE", 1.5),
        ("QUALITY_BOUNCE_THRESHOLD", 1.5),
        ("CODE_MIX_THRESHOLD", 2.0),
        ("MAX_PAGES_PER_DOC", 0),
        ("MAX_DECOMPRESSION_RATIO", 0),
        ("MAX_TOTAL_PIXELS", 0),
        ("MAX_WALL_CLOCK_SECONDS", 0),
        ("GROUNDING_MAX_EDIT_DISTANCE", -1),
        ("LOG_LEVEL", "TRACE"),
        ("HF_BASE_URL", ""),
        ("SLM_MODEL", ""),
    ],
)
def test_out_of_range_values_are_rejected(
    make_settings: Callable[..., Settings], field: str, bad: object
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: bad})


def test_a_single_call_may_not_exceed_the_job_budget(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ValidationError, match="exceeds"):
        make_settings(SLM_MAX_TOKENS=4096, JOB_TOKEN_BUDGET=1000)


# --------------------------------------------------------------------------- #
# Residency — carried here, enforced in WS-C
# --------------------------------------------------------------------------- #


def test_laptop_with_synthetic_only_false_does_not_raise_here(
    make_settings: Callable[..., Settings],
) -> None:
    """The gate lives in the provider factory (BUILD-BRIEF M2).

    Settings must remain loadable so tooling can *report* a bad configuration
    instead of dying while trying to read it.
    """
    settings = make_settings(PROVIDER_PROFILE="laptop", SYNTHETIC_ONLY=False)
    assert settings.SYNTHETIC_ONLY is False
    assert settings.residency_violation_reason is not None
    assert "SYNTHETIC_ONLY=true" in settings.residency_violation_reason


def test_compliant_configurations_report_no_violation(
    make_settings: Callable[..., Settings],
) -> None:
    assert (
        make_settings(
            PROVIDER_PROFILE="laptop", SYNTHETIC_ONLY=True
        ).residency_violation_reason
        is None
    )
    assert (
        make_settings(
            PROVIDER_PROFILE="cpu_node", SYNTHETIC_ONLY=False
        ).residency_violation_reason
        is None
    )


def test_only_the_laptop_profile_is_remote(
    make_settings: Callable[..., Settings],
) -> None:
    assert make_settings(PROVIDER_PROFILE="laptop").is_remote_profile is True
    assert make_settings(PROVIDER_PROFILE="cpu_node").is_remote_profile is False
    assert make_settings(PROVIDER_PROFILE="gpu_node").is_remote_profile is False


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8000/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://10.4.5.6:8000/v1", True),
        ("http://192.168.1.9:8000/v1", True),
        ("http://172.16.0.3:8000/v1", True),
        ("http://slm.internal:8000/v1", True),
        ("http://ombudsman-slm.local/v1", True),
        # An unqualified single-label name cannot resolve on the public DNS.
        ("http://slm-box:8000/v1", True),
        ("https://router.huggingface.co/v1", False),
        ("https://api.openai.com/v1", False),
        # Fails closed: a public FQDN is not local just because it might
        # resolve privately today.
        ("https://slm.rbi.gov.in/v1", False),
        ("", False),
        ("not a url", False),
    ],
)
def test_is_local_endpoint(url: str, expected: bool) -> None:
    assert is_local_endpoint(url) is expected


def test_local_endpoint_hosts_declares_an_internal_fqdn(
    make_settings: Callable[..., Settings],
) -> None:
    """The escape hatch for an in-house server behind a public-looking name."""
    settings = make_settings(
        PROVIDER_PROFILE="gpu_node",
        HF_BASE_URL="https://slm.rbi.gov.in/v1",
        LOCAL_ENDPOINT_HOSTS="slm.rbi.gov.in, other-box.example",
    )
    assert settings.slm_endpoint_is_local is True
    assert settings.residency_violation_reason is None


def test_air_gapped_profile_refuses_a_public_endpoint(
    make_settings: Callable[..., Settings],
) -> None:
    """gpu_node calls an endpoint, so that endpoint must be on-premises."""
    settings = make_settings(
        PROVIDER_PROFILE="gpu_node",
        SYNTHETIC_ONLY=False,
        HF_BASE_URL="https://api.openai.com/v1",
    )
    assert settings.slm_endpoint_is_local is False
    reason = settings.residency_violation_reason
    assert reason is not None
    assert "not an on-premises host" in reason


def test_an_unconfigured_air_gapped_node_runs_without_an_slm(
    make_settings: Callable[..., Settings],
) -> None:
    """A box with no serving stack must start, not refuse.

    Still carrying the shipped default means "nobody pointed this at a server",
    which is the common self-contained shape. It runs on the deterministic
    validators and never opens the socket — refusing to start there would be
    refusing the deployment the air-gap is for.
    """
    settings = make_settings(PROVIDER_PROFILE="gpu_node", SYNTHETIC_ONLY=False)
    assert settings.slm_endpoint_configured is False
    assert settings.slm_available is False
    assert settings.residency_violation_reason is None


def test_cpu_node_base_url_is_inert(
    make_settings: Callable[..., Settings],
) -> None:
    """cpu_node builds no client, so a stale default URL is not a breach."""
    settings = make_settings(PROVIDER_PROFILE="cpu_node", SYNTHETIC_ONLY=False)
    assert settings.uses_remote_endpoint is False
    assert settings.residency_violation_reason is None


def test_only_a_public_endpoint_requires_a_credential(
    make_settings: Callable[..., Settings],
) -> None:
    assert make_settings().requires_remote_credential is True
    assert (
        make_settings(
            HF_BASE_URL="http://localhost:8000/v1"
        ).requires_remote_credential
        is False
    )


@pytest.mark.parametrize(
    ("mode", "profile", "requested", "required"),
    [
        ("auto", "laptop", False, False),
        ("auto", "cpu_node", False, False),
        ("auto", "gpu_node", True, False),
        ("never", "gpu_node", False, False),
        ("require", "laptop", True, True),
    ],
)
def test_gpu_execution_modes(
    make_settings: Callable[..., Settings],
    mode: str,
    profile: str,
    requested: bool,
    required: bool,
) -> None:
    """'auto' is profile-driven, not "use whatever wheel is installed".

    A laptop with onnxruntime-gpu in the venv must keep producing the CPU
    numbers the quality gate was calibrated against.
    """
    settings = make_settings(GPU_EXECUTION=mode, PROVIDER_PROFILE=profile)
    assert settings.gpu_requested is requested
    assert settings.gpu_required is required


def test_there_is_no_residency_bypass_knob() -> None:
    """PLAN.md invariant 6: no bypass flag, and none may be added."""
    suspicious = [
        name
        for name in Settings.model_fields
        if any(
            token in name.upper()
            for token in ("BYPASS", "SKIP_RESIDENCY", "ALLOW_REAL", "FORCE_")
        )
    ]
    assert suspicious == [], f"residency bypass knob introduced: {suspicious}"


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_token_is_absent_by_default_so_tests_never_need_one(
    settings: Settings,
) -> None:
    assert settings.HF_TOKEN is None
    assert settings.hf_token_present is False


def test_token_is_masked_in_repr_and_banner(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(HF_TOKEN="hf_supersecret_value")
    assert "hf_supersecret_value" not in repr(settings)
    assert "hf_supersecret_value" not in settings.startup_banner()
    assert settings.hf_token_present is True
    assert settings.HF_TOKEN is not None
    assert settings.HF_TOKEN.get_secret_value() == "hf_supersecret_value"


def test_stray_remote_credential_is_reported_not_fatal(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(PROVIDER_PROFILE="cpu_node", HF_TOKEN="hf_x")
    assert settings.has_stray_remote_credential is True
    assert make_settings(HF_TOKEN="hf_x").has_stray_remote_credential is False


def test_startup_banner_names_the_active_profile(settings: Settings) -> None:
    banner = settings.startup_banner()
    assert "profile=laptop" in banner
    assert "synthetic_only=true" in banner
    assert "hf_token=unset" in banner


# --------------------------------------------------------------------------- #
# Paths and thread budget
# --------------------------------------------------------------------------- #


def test_relative_paths_resolve_against_the_repository_root(
    settings: Settings,
) -> None:
    assert settings.provider_cache_path.is_absolute()
    assert settings.provider_cache_path.parent.parent == PROJECT_ROOT
    assert settings.models_dir == PROJECT_ROOT / "models"
    assert settings.corpus_dir == PROJECT_ROOT / "corpus"


def test_absolute_paths_are_left_alone(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(MODELS_DIR=tmp_path)
    assert settings.models_dir == tmp_path


def test_thread_budget_assertion(make_settings: Callable[..., Settings]) -> None:
    """PRD P0-11: concurrency x intra_op <= physical cores, checked by CI."""
    settings = make_settings(CELERY_CONCURRENCY=2, ORT_INTRA_OP_THREADS=2)
    assert settings.thread_budget == 4
    settings.assert_thread_budget(physical_cores=4)
    with pytest.raises(ValueError, match="exceeds 2 available cores"):
        settings.assert_thread_budget(physical_cores=2)


def test_thread_budget_is_not_enforced_on_load(
    make_settings: Callable[..., Settings],
) -> None:
    """A 2-core CI runner must still be able to load the documented defaults."""
    settings = make_settings(CELERY_CONCURRENCY=32, ORT_INTRA_OP_THREADS=8)
    assert settings.thread_budget == 256


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_get_settings_is_cached_and_resettable(
    clean_env: pytest.MonkeyPatch,
) -> None:
    reset_settings_cache()
    try:
        clean_env.setenv("OCR_BATCH_SIZE", "16")
        first = get_settings()
        assert first is get_settings()
        assert first.OCR_BATCH_SIZE == 16
        clean_env.setenv("OCR_BATCH_SIZE", "32")
        assert get_settings().OCR_BATCH_SIZE == 16, "cache must not re-read env"
        reset_settings_cache()
        assert get_settings().OCR_BATCH_SIZE == 32
    finally:
        reset_settings_cache()


# --------------------------------------------------------------------------- #
# Thread caps — PLAN.md invariant 5
# --------------------------------------------------------------------------- #

_PROBE = (
    "import json, os, sys; sys.path.insert(0, {root!r}); "
    "import app; "
    "print(json.dumps({{k: os.environ.get(k) for k in app.THREAD_CAP_VARS}}))"
)


def _run_probe(env: dict[str, str], code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        encoding="utf-8",
    )


def _clean_child_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in app.THREAD_CAP_VARS:
        env.pop(name, None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_importing_app_caps_threads_before_anything_else() -> None:
    result = _run_probe(
        _clean_child_env(), _PROBE.format(root=str(PROJECT_ROOT))
    )
    assert result.returncode == 0, result.stderr
    caps = json.loads(result.stdout)
    assert caps == dict.fromkeys(app.THREAD_CAP_VARS, "2")


def test_operator_supplied_values_are_never_overwritten() -> None:
    env = _clean_child_env()
    env["OMP_NUM_THREADS"] = "7"
    result = _run_probe(env, _PROBE.format(root=str(PROJECT_ROOT)))
    assert result.returncode == 0, result.stderr
    caps = json.loads(result.stdout)
    assert caps["OMP_NUM_THREADS"] == "7"
    assert caps["MKL_NUM_THREADS"] == "2"


def test_thread_cap_vars_cover_the_documented_knobs() -> None:
    assert {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "ORT_INTRA_OP_THREADS",
    } <= set(app.THREAD_CAP_VARS)


def test_apply_thread_caps_reports_only_what_it_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in app.THREAD_CAP_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    applied = app.apply_thread_caps()
    assert "OMP_NUM_THREADS" not in applied
    assert applied["MKL_NUM_THREADS"] == "2"
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_late_numeric_import_is_warned_about() -> None:
    """A numeric library imported before ``app`` means its cap arrived late."""
    code = (
        f"import sys, types, warnings; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "sys.modules['numpy'] = types.ModuleType('numpy'); "
        "warnings.simplefilter('always'); "
        "w = warnings.catch_warnings(record=True); ws = w.__enter__(); "
        "import app; w.__exit__(None, None, None); "
        "print(len(ws), ws[0].category.__name__ if ws else '', "
        "'numpy' in str(ws[0].message) if ws else '')"
    )
    result = _run_probe(_clean_child_env(), code)
    assert result.returncode == 0, result.stderr
    count, category, mentions = result.stdout.split()
    assert int(count) >= 1
    assert category == "ThreadCapWarning"
    assert mentions == "True"


def test_app_import_pulls_in_no_numeric_module() -> None:
    """Invariant 5 only holds if importing ``app`` cannot itself import NumPy."""
    code = (
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); import app; "
        "print(sorted(m for m in app.NUMERIC_MODULES if m in sys.modules))"
    )
    result = _run_probe(_clean_child_env(), code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_contracts_import_pulls_in_no_numeric_module() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
        "import app.contracts, app.settings; "
        "print(sorted(m for m in app.NUMERIC_MODULES if m in sys.modules))"
    )
    result = _run_probe(_clean_child_env(), code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
