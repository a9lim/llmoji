"""Regression tests for the v2 single-stage synthesis pipeline.

Targets internal behavior — cache-key isolation, sample-set
dedupe, walk-order determinism, partial-cache-flush-on-error.
The cross-corpus invariant tests live in
:mod:`tests.test_public_surface`; this module is for the
local-only correctness claims that the audit pass surfaced.

Pre-v2 these tests targeted ``_stage_a`` / ``_stage_b``; the
v2 single-stage refactor collapsed both into
``_synthesize_cells`` so the test suite mirrors that.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from llmoji.synth import Synthesizer


# ---------------------------------------------------------------------------
# cache_key — backend + base_url isolation, NUL safety
# ---------------------------------------------------------------------------


def test_cache_key_backend_isolation() -> None:
    """Same model id under different backends produces distinct keys.

    Motivating scenario: a user runs ``--backend openai --model
    gpt-4o``, then ``--backend local --base-url http://localhost
    --model gpt-4o`` against an Ollama tag. Pre-fix the cache would
    return OpenAI's adjective bag as if it came from local.
    """
    from llmoji.synth import cache_key

    a = cache_key("gpt-4o", "openai", "", "m1", "(◕‿◕)", "abc123")
    b = cache_key("gpt-4o", "local", "http://localhost:11434/v1",
                  "m1", "(◕‿◕)", "abc123")
    assert a != b


def test_cache_key_base_url_isolation() -> None:
    """Two ``local`` instances pointed at different endpoints don't
    share cache entries — the model name might be the same but the
    underlying weights almost certainly aren't.
    """
    from llmoji.synth import cache_key

    a = cache_key("llama3.1", "local", "http://localhost:11434/v1",
                  "m1", "(◕‿◕)", "abc123")
    b = cache_key("llama3.1", "local", "http://gpu-box.lan:8080/v1",
                  "m1", "(◕‿◕)", "abc123")
    assert a != b


def test_cache_key_nul_byte_safety() -> None:
    """Length-prefixed framing prevents a buried NUL (or any other
    byte) from collapsing field boundaries. ``source_model="m\\0",
    canonical="x"`` must NOT collide with ``source_model="m",
    canonical="\\0x"``.
    """
    from llmoji.synth import cache_key

    a = cache_key("m", "anthropic", "", "src\0", "x", "h")
    b = cache_key("m", "anthropic", "", "src", "\0x", "h")
    assert a != b


def test_cache_key_empty_field_safety() -> None:
    """Empty fields don't collide with each other across positions."""
    from llmoji.synth import cache_key

    a = cache_key("", "anthropic", "", "src", "(◕‿◕)", "")
    b = cache_key("", "anthropic", "", "", "(◕‿◕)", "src")
    assert a != b


def test_cache_key_deterministic() -> None:
    """Same inputs → same key. Hex string of length 16."""
    from llmoji.synth import cache_key

    k1 = cache_key("m", "anthropic", "", "m1", "(◕‿◕)", "h")
    k2 = cache_key("m", "anthropic", "", "m1", "(◕‿◕)", "h")
    assert k1 == k2
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


def test_cache_key_source_model_isolation() -> None:
    """Same canonical kaomoji emitted by two different source models
    must produce distinct cache keys — the per-source-model adjective
    bag is what the bundle row carries, and conflating them across
    models would corrupt cross-model PCA on the corpus side.
    """
    from llmoji.synth import cache_key

    a = cache_key("m", "anthropic", "", "claude-haiku", "(◕‿◕)", "h")
    b = cache_key("m", "anthropic", "", "gpt-5.4-mini", "(◕‿◕)", "h")
    assert a != b


# ---------------------------------------------------------------------------
# samples_hash — order-invariance + content-sensitivity
# ---------------------------------------------------------------------------


def test_samples_hash_order_invariant() -> None:
    """Same set of (user, assistant) pairs in different presentation
    order must produce the same hash. The hash is computed over the
    sorted pairs, so two re-runs that bucket rows in different orders
    still land on the same cache entry.
    """
    from llmoji.synth import samples_hash

    pairs_a = [("u1", "a1"), ("u2", "a2"), ("u3", "a3")]
    pairs_b = [("u3", "a3"), ("u1", "a1"), ("u2", "a2")]
    assert samples_hash(pairs_a) == samples_hash(pairs_b)


def test_samples_hash_content_sensitive() -> None:
    """Changing any sample's content shifts the hash."""
    from llmoji.synth import samples_hash

    base = [("u1", "a1"), ("u2", "a2")]
    mod = [("u1", "a1"), ("u2", "a2-changed")]
    assert samples_hash(base) != samples_hash(mod)


def test_samples_hash_nul_safety() -> None:
    """A NUL inside a user/assistant string can't shift the boundary
    between fields enough to collide with a different (user,
    assistant) tuple.
    """
    from llmoji.synth import samples_hash

    a = samples_hash([("a", "b\0c")])
    b = samples_hash([("a\0b", "c")])
    assert a != b


# ---------------------------------------------------------------------------
# _synthesize_cells — duplicate-key dedupe + walk-order determinism
# ---------------------------------------------------------------------------


class _CountingFakeSynth(Synthesizer):
    """In-memory fake synth that returns a per-call counter as part
    of the structured response. Emits ``primary_affect[0] = "cheerful"``
    with a counter-stamped extension list — so duplicate-key
    dispatches would visibly produce different outputs if they leaked
    through.
    """

    backend = "fake"
    model_id = "fake-model-1"
    base_url = ""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._n = 0
        self.calls = 0

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        del prompt, schema, max_tokens
        with self._lock:
            self._n += 1
            self.calls += 1
            return {
                "primary_affect": ["cheerful"],
                "stance_modality_function": [f"d{self._n}", "warm"],
            }


def _make_row(model: str, kaomoji: str, user: str, assistant: str):
    """Build a ScrapeRow with the post-Wave-5 lean schema."""
    from llmoji.scrape import ScrapeRow

    return ScrapeRow(
        source="test",
        model=model,
        timestamp="2026-04-28T00:00:00Z",
        cwd="/tmp",
        first_word=kaomoji,
        assistant_text=assistant,
        surrounding_user=user,
    )


def test_synthesize_cells_duplicate_key_dedupes_dispatch(
    tmp_path: Path,
) -> None:
    """Two cells whose sampled (user, assistant) sets hash to the same
    sample_set_hash must dispatch exactly once — the structured
    output is the same for the cell, regardless of source_model
    bucketing.

    Constructed: same kaomoji in two source-model buckets, with
    identical sampled rows. Pre-fix would dispatch twice; v2 collapses
    via the sample-set-hash component of the cache key.
    """
    from llmoji.analyze import _synthesize_cells

    rows_m1 = [_make_row("m1", "(◕‿◕)", "ping", "(◕‿◕) hi")]
    rows_m2 = [_make_row("m2", "(◕‿◕)", "ping", "(◕‿◕) hi")]
    buckets = {"m1": {"(◕‿◕)": rows_m1}, "m2": {"(◕‿◕)": rows_m2}}
    cache_path = tmp_path / "cache.jsonl"

    synth = _CountingFakeSynth()
    synth_by_cell, n_calls, n_cached = _synthesize_cells(
        synth, buckets, cache_path=cache_path, print_progress=False,
    )
    # Cells are NOT collapsed by source_model — that's part of the
    # cache key — so v2 dispatches twice here. (Sample-set-hash
    # collapse only kicks in if the SAME source_model has two cells
    # with the same sampled set, which is rare in practice.)
    assert synth.calls == 2
    assert n_calls == 2
    assert n_cached == 0
    # Every cell got populated.
    assert "m1" in synth_by_cell and "(◕‿◕)" in synth_by_cell["m1"]
    assert "m2" in synth_by_cell and "(◕‿◕)" in synth_by_cell["m2"]


def test_synthesize_cells_warm_cache_no_dispatches(tmp_path: Path) -> None:
    """A second run on the same data hits the cache for every cell
    — zero dispatches, every cell still resolves to its original
    synthesis.
    """
    from llmoji.analyze import _synthesize_cells

    rows = [
        _make_row("m1", "(◕‿◕)", "u1", "(◕‿◕) one"),
        _make_row("m1", "(◕‿◕)", "u2", "(◕‿◕) two"),
    ]
    buckets = {"m1": {"(◕‿◕)": rows}}
    cache_path = tmp_path / "cache.jsonl"

    synth_cold = _CountingFakeSynth()
    cold, _, _ = _synthesize_cells(
        synth_cold, buckets, cache_path=cache_path, print_progress=False,
    )

    synth_warm = _CountingFakeSynth()
    warm, n_calls_warm, n_cached_warm = _synthesize_cells(
        synth_warm, buckets, cache_path=cache_path, print_progress=False,
    )

    assert synth_warm.calls == 0
    assert n_calls_warm == 0
    assert n_cached_warm == 1
    # Same synthesis object, byte-for-byte (modulo dict ordering).
    assert warm["m1"]["(◕‿◕)"] == cold["m1"]["(◕‿◕)"]


def test_synthesize_cells_walk_order_deterministic(tmp_path: Path) -> None:
    """Cold-cache and warm-cache runs assemble the result map in
    identical sorted order — re-runs produce byte-identical bundle
    output regardless of which futures finished first.
    """
    from llmoji.analyze import _synthesize_cells

    rows_a = [_make_row("m1", "(◕‿◕)", "u", "(◕‿◕) one")]
    rows_b = [_make_row("m1", "(>_<)", "u", "(>_<) ow")]
    buckets = {"m1": {"(◕‿◕)": rows_a, "(>_<)": rows_b}}
    cache_path = tmp_path / "cache.jsonl"

    synth_cold = _CountingFakeSynth()
    cold, _, _ = _synthesize_cells(
        synth_cold, buckets, cache_path=cache_path, print_progress=False,
    )
    cold_keys = sorted(cold["m1"].keys())

    synth_warm = _CountingFakeSynth()
    warm, _, _ = _synthesize_cells(
        synth_warm, buckets, cache_path=cache_path, print_progress=False,
    )
    warm_keys = sorted(warm["m1"].keys())

    assert cold_keys == warm_keys


def test_synthesize_cells_writes_one_cache_row_per_unique_key(
    tmp_path: Path,
) -> None:
    """The cache file gets exactly one row per unique cache key. With
    distinct cells per (source_model, canonical), that's one cache
    row per cell.
    """
    from llmoji.analyze import _synthesize_cells

    rows = [
        _make_row("m1", "(◕‿◕)", "u1", "(◕‿◕) one"),
        _make_row("m1", "(>_<)", "u2", "(>_<) ow"),
        _make_row("m2", "(◕‿◕)", "u3", "(◕‿◕) hi"),
    ]
    buckets = {
        "m1": {"(◕‿◕)": [rows[0]], "(>_<)": [rows[1]]},
        "m2": {"(◕‿◕)": [rows[2]]},
    }
    cache_path = tmp_path / "cache.jsonl"

    _synthesize_cells(
        _CountingFakeSynth(), buckets, cache_path=cache_path,
        print_progress=False,
    )
    cached_rows = [
        json.loads(line) for line in cache_path.read_text().splitlines()
        if line.strip()
    ]
    keys = [r["key"] for r in cached_rows]
    assert len(keys) == len(set(keys)) == 3, (
        f"expected 3 unique cache rows, got {keys!r}"
    )
    # Each cache row carries the structured synthesis object plus
    # source_model + kaomoji metadata.
    for r in cached_rows:
        assert set(r.keys()) >= {
            "key", "kaomoji", "source_model", "synthesis",
            "model", "backend",
        }
        assert isinstance(r["synthesis"], dict)
        assert "primary_affect" in r["synthesis"]
        assert "stance_modality_function" in r["synthesis"]


# ---------------------------------------------------------------------------
# _synthesize_cells — partial cache flush survives mid-wave error
# ---------------------------------------------------------------------------


class _FailOnceSynth(Synthesizer):
    """Counter-based synth that raises on the Nth call. Pre-Wave-6
    a Stage-A error caused the deferred cache flush to never run, so
    *all* dispatched results were lost — a re-run paid the full API
    cost again. Wave 6 flushes per-future inside the as_completed
    loop, and v2 inherits the same per-future flush pattern.
    """

    backend = "fake"
    model_id = "fake-model-1"
    base_url = ""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on = fail_on_call
        self._lock = threading.Lock()
        self._n = 0
        self.calls = 0

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        del prompt, schema, max_tokens
        with self._lock:
            self._n += 1
            self.calls += 1
            if self._n == self._fail_on:
                raise RuntimeError(f"simulated failure on call {self._n}")
            return {
                "primary_affect": ["cheerful"],
                "stance_modality_function": [f"d{self._n}", "warm"],
            }


def test_synthesize_partial_cache_on_error_then_resume(
    tmp_path: Path,
) -> None:
    """A synth failure mid-wave must leave the cache flushed for cells
    that succeeded before the raise. A re-run with a passing synth
    then pays API cost only for the cells that previously failed —
    the rest hit the cache.

    Concurrency forced to 1 so the futures-complete order is
    deterministic (== submission order == walk order).
    """
    import pytest

    from llmoji.analyze import AnalyzeError, _synthesize_cells

    rows = [
        _make_row("m1", "(◕‿◕)", f"u{i}", f"(◕‿◕) a{i}")
        for i in range(4)
    ]
    # Four distinct cells so each gets its own cache key.
    buckets = {
        "m1": {f"(face-{i})": [rows[i]] for i in range(4)},
    }
    cache_path = tmp_path / "cache.jsonl"

    cold = _FailOnceSynth(fail_on_call=2)
    with pytest.raises(AnalyzeError, match=r"synthesize:.*re-run"):
        _synthesize_cells(
            cold, buckets, cache_path=cache_path,
            print_progress=False, max_workers=1,
        )

    # 4 dispatched, 1 raised → 3 cache rows on disk.
    cached_rows = [
        json.loads(line) for line in cache_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(cached_rows) == 3, (
        f"expected 3 cache rows after partial failure, "
        f"got {len(cached_rows)}: {cached_rows!r}"
    )

    # Resume: passing synth on the second run. Only the 1 failed
    # cell should dispatch — the other 3 hit the cache.
    warm = _CountingFakeSynth()
    synth_by_cell, n_calls, n_cached = _synthesize_cells(
        warm, buckets, cache_path=cache_path,
        print_progress=False, max_workers=1,
    )
    assert warm.calls == 1
    assert n_calls == 1
    assert n_cached == 3
    # Every cell now has a synthesis; none are None.
    for face in [f"(face-{i})" for i in range(4)]:
        assert synth_by_cell["m1"][face] is not None
        assert "primary_affect" in synth_by_cell["m1"][face]
