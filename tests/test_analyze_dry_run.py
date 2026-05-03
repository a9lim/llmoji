"""Tests for ``llmoji analyze --dry-run`` (the planning surface).

``plan_analyze()`` runs the same bucketing + sampling + cache-key
work :func:`_stage_a` would, but never imports a synth backend or
calls one. Token + cost figures are approximate; we test the
shape and the determinism, not the exact dollar amount.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


def _make_row(model: str, kaomoji: str, user: str, assistant: str) -> Any:
    """Build a ScrapeRow with the lean post-Wave-5 schema."""
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


# ---------------------------------------------------------------------------
# plan_analyze() — shape + determinism + dispatch dedup
# ---------------------------------------------------------------------------


def test_plan_analyze_returns_expected_counts() -> None:
    """Counts match the input shape."""
    from llmoji.analyze import plan_analyze

    rows = [
        _make_row("m1", "(◕‿◕)", "u1", "(◕‿◕) one"),
        _make_row("m1", "(◕‿◕)", "u2", "(◕‿◕) two"),
        _make_row("m1", "(`・ω・´)", "u3", "(`・ω・´) three"),
        _make_row("m2", "(◕‿◕)", "u4", "(◕‿◕) four"),
    ]
    plan = plan_analyze(rows, backend="anthropic")

    assert plan.total_rows == 4
    assert plan.canonical_unique == 2  # (◕‿◕) + (`・ω・´)
    assert plan.providers_seen == ["test"]
    # 3 cells: (m1, ◕‿◕), (m1, `・ω・´), (m2, ◕‿◕).
    assert sum(len(p) for p in plan.counts_by_cell.values()) == 3
    assert plan.stage_b_calls == 3


def test_plan_analyze_stage_a_dedupes_duplicate_keys() -> None:
    """Two sampled rows with identical (canonical, user, assistant)
    fold to one cache key — the same dedup :func:`_stage_a` does at
    runtime so the dry-run estimate matches what the real run would
    issue.
    """
    from llmoji.analyze import plan_analyze

    rows = [
        _make_row("m1", "(◕‿◕)", "u", "a"),
        _make_row("m1", "(◕‿◕)", "u", "a"),  # identical → same key
        _make_row("m1", "(◕‿◕)", "u", "b"),
    ]
    plan = plan_analyze(rows, backend="anthropic")
    # 3 sampled rows, 2 unique keys.
    assert plan.stage_a_max_calls == 3
    assert plan.stage_a_unique_calls == 2


def test_plan_analyze_respects_sample_cap() -> None:
    """A cell with more rows than INSTANCE_SAMPLE_CAP samples down to
    the cap — same rule the real run applies.
    """
    from llmoji.analyze import INSTANCE_SAMPLE_CAP, plan_analyze

    rows = [
        _make_row("m1", "(◕‿◕)", f"u{i}", f"a{i}")
        for i in range(INSTANCE_SAMPLE_CAP + 5)
    ]
    plan = plan_analyze(rows, backend="anthropic")
    assert plan.stage_a_max_calls == INSTANCE_SAMPLE_CAP
    assert plan.stage_a_unique_calls == INSTANCE_SAMPLE_CAP


def test_plan_analyze_does_not_import_synth_sdks() -> None:
    """Dry-run must not pull in anthropic / openai SDKs — that's the
    whole point. Walk the same input as run_analyze would, just via
    the planning code path, and assert no SDK got constructed.
    """
    import llmoji.synth as synth_module
    from llmoji.analyze import plan_analyze

    rows = [_make_row("m1", "(◕‿◕)", "u", "a")]

    sdk_constructed = []

    def fake_make_synthesizer(*args: Any, **kwargs: Any) -> Any:
        sdk_constructed.append((args, kwargs))
        raise AssertionError("plan_analyze must not call make_synthesizer")

    with patch.object(synth_module, "make_synthesizer", fake_make_synthesizer):
        plan = plan_analyze(rows, backend="anthropic")

    assert sdk_constructed == []
    assert plan.total_rows == 1


def test_plan_analyze_cost_set_for_anthropic() -> None:
    """Priced backends get a non-None cost estimate."""
    from llmoji.analyze import plan_analyze

    rows = [_make_row("m1", "(◕‿◕)", "u", "a")]
    plan = plan_analyze(rows, backend="anthropic")
    assert plan.estimated_cost_usd is not None
    assert plan.estimated_cost_usd >= 0
    assert plan.backend == "anthropic"
    # Default model id pinned in synth_prompts.
    assert "haiku" in plan.model_id.lower()


def test_plan_analyze_cost_none_for_local_backend() -> None:
    """Local backends aren't priced — estimate returns None there."""
    from llmoji.analyze import plan_analyze

    rows = [_make_row("m1", "(◕‿◕)", "u", "a")]
    plan = plan_analyze(
        rows, backend="local", base_url="http://localhost:11434/v1",
        model_id="llama3.1",
    )
    assert plan.estimated_cost_usd is None
    assert plan.backend == "local"
    assert plan.model_id == "llama3.1"


# ---------------------------------------------------------------------------
# CLI flag behavior
# ---------------------------------------------------------------------------


def test_cli_dry_run_skips_run_analyze(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` calls plan_analyze, never run_analyze."""
    from llmoji import analyze as analyze_module
    from llmoji import cli

    rows = [_make_row("m1", "(◕‿◕)", "u", "a")]
    monkeypatch.setattr(cli, "_gather_rows", lambda: iter(rows))

    run_calls = []

    def fake_run_analyze(*args: Any, **kwargs: Any) -> Any:
        run_calls.append((args, kwargs))
        raise AssertionError("--dry-run must not call run_analyze")

    monkeypatch.setattr(analyze_module, "run_analyze", fake_run_analyze)

    parser = cli._build_parser()
    args = parser.parse_args(["analyze", "--dry-run"])
    rc = cli._cmd_analyze(args)

    assert rc == 0
    assert run_calls == []
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "estimated tokens" in out.lower()
