"""Regression tests for the Wave 1 correctness fixes.

These tests target internal behavior (cache-key isolation,
Stage-A duplicate-key dedupe). The cross-corpus invariant tests
live in :mod:`tests.test_public_surface`; this module is for the
local-only correctness claims that the audit pass surfaced.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from llmoji.synth import Synthesizer


# ---------------------------------------------------------------------------
# cache_key — backend + base_url isolation, NUL safety
# ---------------------------------------------------------------------------


def test_cache_key_backend_isolation() -> None:
    """Same model id under different backends produces distinct keys.

    Motivating scenario: a user runs ``--backend openai --model
    gpt-4o``, then ``--backend local --base-url http://localhost
    --model gpt-4o`` against an Ollama tag. Pre-fix the cache would
    return OpenAI's paraphrase as if it came from local.
    """
    from llmoji.synth import cache_key

    a = cache_key("gpt-4o", "openai", "", "(◕‿◕)", "u", "a")
    b = cache_key("gpt-4o", "local", "http://localhost:11434/v1",
                  "(◕‿◕)", "u", "a")
    assert a != b


def test_cache_key_base_url_isolation() -> None:
    """Two ``local`` instances pointed at different endpoints don't
    share cache entries — the model name might be the same but the
    underlying weights almost certainly aren't.
    """
    from llmoji.synth import cache_key

    a = cache_key("llama3.1", "local", "http://localhost:11434/v1",
                  "(◕‿◕)", "u", "a")
    b = cache_key("llama3.1", "local", "http://gpu-box.lan:8080/v1",
                  "(◕‿◕)", "u", "a")
    assert a != b


def test_cache_key_nul_byte_safety() -> None:
    """Length-prefixed framing prevents a buried NUL (or any other
    byte) from collapsing field boundaries. ``user="a", assistant="b\\0c"``
    must NOT collide with ``user="a\\0b", assistant="c"``.
    """
    from llmoji.synth import cache_key

    a = cache_key("m", "anthropic", "", "(◕‿◕)", "a", "b\0c")
    b = cache_key("m", "anthropic", "", "(◕‿◕)", "a\0b", "c")
    assert a != b


def test_cache_key_empty_field_safety() -> None:
    """Empty fields don't collide with each other across positions."""
    from llmoji.synth import cache_key

    a = cache_key("", "anthropic", "", "(◕‿◕)", "u", "")
    b = cache_key("", "anthropic", "", "(◕‿◕)", "", "u")
    assert a != b


def test_cache_key_deterministic() -> None:
    """Same inputs → same key. Hex string of length 16."""
    from llmoji.synth import cache_key

    k1 = cache_key("m", "anthropic", "", "(◕‿◕)", "u", "a")
    k2 = cache_key("m", "anthropic", "", "(◕‿◕)", "u", "a")
    assert k1 == k2
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


# ---------------------------------------------------------------------------
# Stage-A — duplicate-key dedupe + walk-order determinism
# ---------------------------------------------------------------------------


class _CountingFakeSynth(Synthesizer):
    """In-memory fake synth that returns a per-call counter as the
    description. Emits ``"d{n}"`` for the nth call across the whole
    fixture, regardless of prompt — so duplicate-key dispatches
    would visibly produce different outputs if they leaked through.
    """

    backend = "fake"
    model_id = "fake-model-1"
    base_url = ""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0
        self.calls = 0

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        del prompt, max_tokens  # counter-only fake; ignores input
        with self._lock:
            self._n += 1
            self.calls += 1
            return f"d{self._n}"


def _make_row(model: str, kaomoji: str, user: str, assistant: str):
    """Build a ScrapeRow with all required fields — fills in the
    optional/legacy fields with sensible defaults so the test stays
    decoupled from Wave 5's schema-tightening (which drops several
    of these)."""
    from llmoji.scrape import ScrapeRow

    return ScrapeRow(
        source="test",
        session_id="",
        project_slug="",
        assistant_uuid="",
        parent_uuid=None,
        model=model,
        timestamp="2026-04-28T00:00:00Z",
        cwd="/tmp",
        git_branch=None,
        turn_index=0,
        had_thinking=False,
        first_word=kaomoji,
        assistant_text=assistant,
        surrounding_user=user,
    )


def test_stage_a_duplicate_key_dedupes_dispatch(tmp_path: Path) -> None:
    """Two sampled rows that hash to the same cache key must dispatch
    exactly once — and both walk positions must share the single
    description so cold-cache and warm-cache runs feed Stage B
    identical lists.

    Pre-fix: cold-cache run dispatched twice (got "d1" and "d2"),
    warm-cache run resolved both walks to the same cached row
    (last-write-wins, "d2"), so cold/warm Stage B input differed.
    """
    from llmoji.analyze import _stage_a

    # Two rows in the same cell with identical (canonical, user,
    # assistant). Cache key collapses to one entry.
    rows = [
        _make_row("m1", "(◕‿◕)", "ping", "(◕‿◕) hi"),
        _make_row("m1", "(◕‿◕)", "ping", "(◕‿◕) hi"),
    ]
    buckets = {"m1": {"(◕‿◕)": rows}}
    cache_path = tmp_path / "cache.jsonl"

    synth = _CountingFakeSynth()
    descs_cold, n_calls_cold, n_cached_cold = _stage_a(
        synth, buckets, cache_path=cache_path, print_progress=False,
    )
    # One unique key → one dispatch.
    assert synth.calls == 1
    assert n_calls_cold == 1
    assert n_cached_cold == 0
    # Both walk positions populated with the same description.
    assert descs_cold["m1"]["(◕‿◕)"] == ["d1", "d1"]

    # Warm-cache rerun: zero dispatches, both positions still resolve
    # to the same cached description.
    synth_warm = _CountingFakeSynth()
    descs_warm, n_calls_warm, n_cached_warm = _stage_a(
        synth_warm, buckets, cache_path=cache_path, print_progress=False,
    )
    assert synth_warm.calls == 0
    assert n_calls_warm == 0
    assert n_cached_warm == 2
    assert descs_warm["m1"]["(◕‿◕)"] == descs_cold["m1"]["(◕‿◕)"]


def test_stage_a_walk_order_deterministic(tmp_path: Path) -> None:
    """Cold-cache and warm-cache runs feed Stage B descriptions in
    identical order — the order doesn't depend on which future
    finished first.
    """
    from llmoji.analyze import _stage_a

    rows = [
        _make_row("m1", "(◕‿◕)", "u1", "(◕‿◕) one"),
        _make_row("m1", "(◕‿◕)", "u2", "(◕‿◕) two"),
        _make_row("m1", "(◕‿◕)", "u3", "(◕‿◕) three"),
    ]
    buckets = {"m1": {"(◕‿◕)": rows}}
    cache_path = tmp_path / "cache.jsonl"

    synth_cold = _CountingFakeSynth()
    descs_cold, _, _ = _stage_a(
        synth_cold, buckets, cache_path=cache_path, print_progress=False,
    )

    synth_warm = _CountingFakeSynth()
    descs_warm, _, _ = _stage_a(
        synth_warm, buckets, cache_path=cache_path, print_progress=False,
    )

    # Same list, same order, despite warm reading from cache.
    assert descs_cold["m1"]["(◕‿◕)"] == descs_warm["m1"]["(◕‿◕)"]


def test_stage_a_writes_one_cache_row_per_unique_key(tmp_path: Path) -> None:
    """The cache file gets exactly one row per unique key, even when
    multiple sampled rows in the walk share the key. Avoids a
    last-write-wins drift between cold and warm runs.
    """
    from llmoji.analyze import _stage_a

    rows = [
        _make_row("m1", "(◕‿◕)", "ping", "(◕‿◕) hi"),
        _make_row("m1", "(◕‿◕)", "ping", "(◕‿◕) hi"),
        _make_row("m1", "(◕‿◕)", "different", "(◕‿◕) bye"),
    ]
    buckets = {"m1": {"(◕‿◕)": rows}}
    cache_path = tmp_path / "cache.jsonl"

    _stage_a(
        _CountingFakeSynth(), buckets, cache_path=cache_path,
        print_progress=False,
    )
    cached_rows = [
        json.loads(line) for line in cache_path.read_text().splitlines()
        if line.strip()
    ]
    keys = [r["key"] for r in cached_rows]
    assert len(keys) == len(set(keys)) == 2, (
        f"expected 2 unique cache rows, got {keys!r}"
    )
