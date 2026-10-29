"""Unit tests for :mod:`app.providers.cache` — keying, versioning, threads."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.providers.cache import (
    CACHE_SCHEMA_VERSION,
    MEMORY_PATH,
    ResponseCache,
    canonical_params,
    make_cache_key,
)

_ARGS = {"model": "Qwen/Qwen3-1.7B-Instruct", "prompt": "extract fields"}


# --------------------------------------------------------------------------- #
# Key construction
# --------------------------------------------------------------------------- #


def test_key_is_a_sha256_hex_digest() -> None:
    key = make_cache_key(**_ARGS, params={"temperature": 0.0})
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_key_is_stable_across_calls() -> None:
    first = make_cache_key(**_ARGS, params={"temperature": 0.0})
    second = make_cache_key(**_ARGS, params={"temperature": 0.0})
    assert first == second


def test_parameter_order_does_not_change_the_key() -> None:
    """Canonicalisation, or every dict literal is a separate cache entry."""
    a = make_cache_key(**_ARGS, params={"temperature": 0.0, "top_p": 1.0})
    b = make_cache_key(**_ARGS, params={"top_p": 1.0, "temperature": 0.0})
    assert a == b


def test_no_params_and_empty_params_agree() -> None:
    assert make_cache_key(**_ARGS) == make_cache_key(**_ARGS, params={})


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "other/model"), ("prompt", "different prompt")],
)
def test_every_component_participates_in_the_key(field: str, value: str) -> None:
    base = make_cache_key(**_ARGS, params={"temperature": 0.0})
    changed = make_cache_key(
        **{**_ARGS, field: value}, params={"temperature": 0.0}
    )
    assert base != changed


def test_params_participate_in_the_key() -> None:
    a = make_cache_key(**_ARGS, params={"temperature": 0.0})
    b = make_cache_key(**_ARGS, params={"temperature": 0.7})
    assert a != b


def test_schema_version_participates_in_the_key() -> None:
    """A bump must miss, not serve output produced under different framing."""
    a = make_cache_key(**_ARGS, schema_version="1")
    b = make_cache_key(**_ARGS, schema_version="2")
    assert a != b


def test_field_boundaries_cannot_be_forged() -> None:
    """``model+prompt`` concatenation must not collide across the boundary."""
    a = make_cache_key(model="ab", prompt="c")
    b = make_cache_key(model="a", prompt="bc")
    assert a != b


def test_canonical_params_survives_a_non_json_value() -> None:
    """A stray Path in params must not blow up key construction."""
    rendered = canonical_params({"path": Path("a/b"), "n": 1})
    assert "PosixPath" in rendered or "WindowsPath" in rendered
    assert '"n":1' in rendered


def test_canonical_params_keeps_devanagari_verbatim() -> None:
    assert "मराठी" in canonical_params({"x": "मराठी"})


# --------------------------------------------------------------------------- #
# Store and lookup
# --------------------------------------------------------------------------- #


def test_miss_then_store_then_hit() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        key = make_cache_key(**_ARGS)
        assert cache.get(key) is None
        cache.put(key, '{"fields": []}', model="m")
        assert cache.get(key) == '{"fields": []}'
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.stores == 1
        assert cache.stats.hit_rate == pytest.approx(0.5)


def test_hit_count_increments_on_each_read() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        cache.put("k" * 64, "v", model="m")
        first = cache.get_entry("k" * 64)
        second = cache.get_entry("k" * 64)
        assert first is not None and second is not None
        assert second.hit_count == first.hit_count + 1


def test_put_replaces_rather_than_duplicates() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        cache.put("a" * 64, "first", model="m")
        cache.put("a" * 64, "second", model="m")
        assert cache.get("a" * 64) == "second"
        assert len(cache) == 1


def test_an_entry_from_another_schema_version_is_a_miss() -> None:
    """The versioning contract: old rows miss, they do not lie."""
    with ResponseCache(MEMORY_PATH, schema_version="1") as cache:
        cache.put("b" * 64, "old", model="m")
        assert cache.get("b" * 64) == "old"
        cache.schema_version = "2"
        assert cache.get("b" * 64) is None
        assert cache.total_rows() == 1
        assert len(cache) == 0


def test_purge_other_versions_removes_only_the_strangers() -> None:
    with ResponseCache(MEMORY_PATH, schema_version="1") as cache:
        cache.put("c" * 64, "old", model="m")
        cache.schema_version = "2"
        cache.put("d" * 64, "new", model="m")
        assert cache.purge_other_versions() == 1
        assert cache.total_rows() == 1
        assert cache.get("d" * 64) == "new"


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #


def test_invalidate_one_key() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        cache.put("e" * 64, "v", model="m")
        assert cache.invalidate("e" * 64) is True
        assert cache.get("e" * 64) is None
        assert cache.invalidate("e" * 64) is False


def test_invalidate_by_model_leaves_other_models_alone() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        cache.put("f" * 64, "v1", model="alpha")
        cache.put("0" * 64, "v2", model="beta")
        assert cache.invalidate_model("alpha") == 1
        assert cache.get("0" * 64) == "v2"


def test_invalidate_all_empties_the_cache() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        for index in range(3):
            cache.put(str(index) * 64, "v", model="m")
        assert cache.invalidate_all() == 3
        assert len(cache) == 0


# --------------------------------------------------------------------------- #
# Lifecycle and storage
# --------------------------------------------------------------------------- #


def test_file_backed_cache_creates_its_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "provider.sqlite"
    with ResponseCache(target) as cache:
        cache.put("9" * 64, "v", model="m")
    assert target.is_file()


def test_a_file_cache_survives_reopening(tmp_path: Path) -> None:
    """The point of the thing: yesterday's experiments are still free today."""
    target = tmp_path / "provider.sqlite"
    key = make_cache_key(**_ARGS)
    with ResponseCache(target) as cache:
        cache.put(key, "persisted", model="m")
    with ResponseCache(target) as reopened:
        assert reopened.get(key) == "persisted"


def test_use_after_close_is_refused() -> None:
    cache = ResponseCache(MEMORY_PATH)
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("a" * 64)


def test_close_is_idempotent() -> None:
    cache = ResponseCache(MEMORY_PATH)
    cache.close()
    cache.close()


def test_default_schema_version_is_the_module_constant() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        assert cache.schema_version == CACHE_SCHEMA_VERSION


def test_repr_names_the_path_and_version() -> None:
    with ResponseCache(MEMORY_PATH) as cache:
        assert "schema_version" in repr(cache)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_writers_and_readers_do_not_corrupt_the_cache() -> None:
    """``extract_fields_batch`` fans out; one cache serves the whole gather."""
    cache = ResponseCache(MEMORY_PATH)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            key = make_cache_key(model="m", prompt=f"p{index}")
            cache.put(key, f"v{index}", model="m")
            assert cache.get(key) == f"v{index}"
        except BaseException as exc:  # noqa: BLE001 - collected and re-asserted
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(cache) == 24
    cache.close()
