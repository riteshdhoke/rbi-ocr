"""Unit tests for :mod:`app.providers.base` — budget, errors, image coercion."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from app.contracts.errors import (
    ProviderError,
    ProviderUnavailable,
    TokenBudgetExceeded,
)
from app.providers.base import (
    JobBudget,
    ModelArtifactMissing,
    ProviderMethodNotSupported,
    inject_truststore,
    to_page_image,
    truststore_injected,
)

# --------------------------------------------------------------------------- #
# JobBudget
# --------------------------------------------------------------------------- #


def test_budget_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        JobBudget(0)


def test_reserve_then_settle_moves_headroom_into_usage() -> None:
    budget = JobBudget(1000)
    claim = budget.reserve(400)
    assert budget.remaining == 600
    assert budget.reserved == 400
    claim.settle(250)
    assert budget.used == 250
    assert budget.reserved == 0
    assert budget.remaining == 750


def test_a_failed_call_costs_no_budget() -> None:
    """The whole reason reserve/settle exists rather than a running total."""
    budget = JobBudget(1000)
    with budget.reservation(400):
        assert budget.remaining == 600
    assert budget.used == 0
    assert budget.remaining == 1000


def test_settling_inside_the_context_manager_survives_the_release() -> None:
    budget = JobBudget(1000)
    with budget.reservation(400) as claim:
        claim.settle(120)
    assert budget.used == 120
    assert budget.reserved == 0


def test_breach_raises_typed_and_carries_the_numbers() -> None:
    budget = JobBudget(500, job_id="job-9")
    budget.reserve(300).settle(300)
    with pytest.raises(TokenBudgetExceeded) as excinfo:
        budget.reserve(300)
    error = excinfo.value
    assert error.budget == 500
    assert error.used == 300
    assert error.context["job_id"] == "job-9"
    assert isinstance(error, ProviderError)


def test_in_flight_reservations_count_against_the_ceiling() -> None:
    """Two concurrent calls must not each see the full remaining budget."""
    budget = JobBudget(100)
    budget.reserve(60)
    with pytest.raises(TokenBudgetExceeded):
        budget.reserve(60)


def test_double_settle_is_a_programming_error() -> None:
    budget = JobBudget(100)
    claim = budget.reserve(10)
    claim.settle(5)
    with pytest.raises(ValueError, match="already settled"):
        claim.settle(5)


def test_release_after_settle_does_not_refund_twice() -> None:
    budget = JobBudget(100)
    claim = budget.reserve(10)
    claim.settle(10)
    claim.release()
    assert budget.used == 10
    assert budget.reserved == 0


def test_negative_amounts_are_rejected() -> None:
    budget = JobBudget(100)
    with pytest.raises(ValueError):
        budget.reserve(-1)
    with pytest.raises(ValueError):
        budget.reserve(1).settle(-1)


def test_budget_is_thread_safe_under_concurrent_reservation() -> None:
    """``extract_fields_batch`` shares one budget across the gather."""
    budget = JobBudget(1000)
    granted: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            claim = budget.reserve(100)
        except TokenBudgetExceeded:
            return
        claim.settle(100)
        with lock:
            granted.append(100)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(granted) == 1000
    assert budget.used == 1000
    assert budget.remaining == 0


def test_snapshot_is_json_safe() -> None:
    budget = JobBudget(100, job_id="j")
    budget.reserve(10).settle(4)
    assert budget.snapshot() == {
        "job_id": "j",
        "limit": 100,
        "used": 4,
        "reserved": 0,
        "remaining": 96,
    }


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_method_not_supported_names_who_does_implement_it() -> None:
    error = ProviderMethodNotSupported(
        "ocr_page", provider="hf_api", implemented_by="LocalProvider"
    )
    assert error.code == "PROVIDER_METHOD_NOT_SUPPORTED"
    assert error.http_status == 501
    assert error.context["implemented_by"] == "LocalProvider"
    assert "LocalProvider" in error.message
    assert isinstance(error, ProviderError)


def test_artifact_missing_is_unavailable_and_carries_a_remedy() -> None:
    error = ModelArtifactMissing(
        artifact="lid.176.ftz",
        path=Path("models/lid/lid.176.ftz"),
        how_to_obtain="Run tools/fetch_models.py.",
    )
    assert isinstance(error, ProviderUnavailable)
    assert error.code == "MODEL_ARTIFACT_MISSING"
    assert error.context["artifact"] == "lid.176.ftz"
    assert error.context["how_to_obtain"].startswith("Run")
    assert "lid.176.ftz" in error.message


def test_error_dicts_are_serialisable() -> None:
    payload = ProviderMethodNotSupported(
        "translate", provider="p", implemented_by="q"
    ).to_dict()
    assert payload["code"] == "PROVIDER_METHOD_NOT_SUPPORTED"
    assert set(payload) == {"code", "message", "context"}


# --------------------------------------------------------------------------- #
# truststore
# --------------------------------------------------------------------------- #


def test_truststore_injection_is_idempotent() -> None:
    """Corporate TLS interception (D7); a repeat call must be harmless."""
    first = inject_truststore()
    second = inject_truststore()
    assert first == second
    if first:
        assert truststore_injected()


# --------------------------------------------------------------------------- #
# Image coercion
# --------------------------------------------------------------------------- #


def test_numpy_array_passes_through_unchanged() -> None:
    array = np.zeros((4, 6, 3), dtype=np.uint8)
    array[0, 0] = (10, 20, 30)
    out = to_page_image(array)
    assert out.shape == (4, 6, 3)
    assert tuple(out[0, 0]) == (10, 20, 30)


def test_grayscale_is_accepted() -> None:
    out = to_page_image(np.zeros((3, 3), dtype=np.uint8))
    assert out.ndim == 2


def test_rgba_is_reduced_to_three_channels() -> None:
    out = to_page_image(np.zeros((2, 2, 4), dtype=np.uint8))
    assert out.shape == (2, 2, 3)


def test_pil_rgb_is_reordered_to_bgr() -> None:
    """The layer speaks OpenCV BGR; a PIL caller must not silently invert."""
    from PIL import Image

    image = Image.new("RGB", (2, 2), color=(10, 20, 30))
    out = to_page_image(image)
    assert tuple(out[0, 0]) == (30, 20, 10)


def test_non_array_object_is_refused() -> None:
    with pytest.raises(TypeError, match="page image"):
        to_page_image("not an image")


def test_wrong_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        to_page_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_unsupported_channel_count_is_refused() -> None:
    with pytest.raises(ValueError, match="channels"):
        to_page_image(np.zeros((2, 2, 5), dtype=np.uint8))


def test_float_input_is_clipped_to_uint8() -> None:
    out = to_page_image(np.array([[300.0, -5.0]], dtype=np.float32))
    assert out.dtype == np.uint8
    assert out.tolist() == [[255, 0]]
