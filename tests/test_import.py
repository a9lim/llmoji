"""Tests for ``llmoji import`` (the dedup-aware merge of native
session files into the live journal).

The user-facing CLI is ``llmoji import <provider>``; the internal
module name stays :mod:`llmoji.backfill` so the parity tests in
``test_pipeline_parity.py`` keep working without rename churn.
These tests focus on the dedup + atomic-merge behavior that
``import_provider`` adds on top of the existing ``backfill_*``
iterators.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_journal(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a JSONL journal with the canonical 6-field shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(
    ts: str = "2026-04-28T00:00:00Z",
    model: str = "m1",
    cwd: str = "/tmp",
    kaomoji: str = "(◕‿◕)",
    user_text: str = "u",
    assistant_text: str = "a",
) -> dict[str, object]:
    return {
        "ts": ts, "model": model, "cwd": cwd,
        "kaomoji": kaomoji, "user_text": user_text,
        "assistant_text": assistant_text,
    }


# ---------------------------------------------------------------------------
# dedup key correctness
# ---------------------------------------------------------------------------


def test_dedup_key_same_inputs_collide() -> None:
    from llmoji.backfill import _dedup_key_for_journal_row

    a = _dedup_key_for_journal_row("ts", "m", "assistant body")
    b = _dedup_key_for_journal_row("ts", "m", "assistant body")
    assert a == b


def test_dedup_key_different_assistant_differs() -> None:
    from llmoji.backfill import _dedup_key_for_journal_row

    a = _dedup_key_for_journal_row("ts", "m", "body one")
    b = _dedup_key_for_journal_row("ts", "m", "body two")
    assert a != b


def test_dedup_key_nul_byte_safety() -> None:
    """Length-prefixed framing prevents a buried NUL from collapsing
    field boundaries (mirrors the cache_key invariant)."""
    from llmoji.backfill import _dedup_key_for_journal_row

    a = _dedup_key_for_journal_row("ts", "m", "a\0b")
    b = _dedup_key_for_journal_row("ts", "m\0a", "b")
    assert a != b


def test_journal_dedup_keys_round_trips_existing_rows(tmp_path: Path) -> None:
    """Loading an existing journal should populate the dedup set with
    a key for every well-formed row."""
    from llmoji.backfill import _dedup_key_for_journal_row, _journal_dedup_keys

    j = tmp_path / "journal.jsonl"
    rows = [
        _row(ts="t1", assistant_text="a1"),
        _row(ts="t2", assistant_text="a2"),
    ]
    _write_journal(j, rows)
    keys = _journal_dedup_keys(j)
    expected = {
        _dedup_key_for_journal_row("t1", "m1", "a1"),
        _dedup_key_for_journal_row("t2", "m1", "a2"),
    }
    assert keys == expected


def test_journal_dedup_keys_skips_malformed_lines(tmp_path: Path) -> None:
    """Malformed rows are tolerated — the journal walk doesn't choke
    on a borked line."""
    from llmoji.backfill import _journal_dedup_keys

    j = tmp_path / "journal.jsonl"
    j.parent.mkdir(parents=True, exist_ok=True)
    with j.open("w") as f:
        f.write(json.dumps(_row(ts="t1")) + "\n")
        f.write("{not valid json\n")
        f.write(json.dumps(_row(ts="t2", assistant_text="a2")) + "\n")
    keys = _journal_dedup_keys(j)
    assert len(keys) == 2  # the two valid rows


# ---------------------------------------------------------------------------
# import_provider — happy path + dedup + atomicity
# ---------------------------------------------------------------------------


def test_import_provider_appends_novel_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running import twice on the same source files results in zero
    novel rows on the second pass — dedup catches every row the first
    run already wrote."""
    from llmoji import backfill
    from llmoji.scrape import ScrapeRow

    journal = tmp_path / "kaomoji-journal.jsonl"

    # Stub the source iterator + journal_for resolver so we don't
    # touch real ~/.claude/.codex/.hermes paths.
    fake_rows = [
        ScrapeRow(
            source="claude_code", model="m1",
            timestamp=f"2026-04-28T00:00:0{i}Z", cwd="/tmp",
            first_word="(◕‿◕)",
            assistant_text=f"text {i}",
            surrounding_user="u",
        )
        for i in range(5)
    ]
    def _fake_iter(_name: str):
        _ = _name
        return iter(fake_rows)

    def _fake_journal(_name: str) -> Path:
        _ = _name
        return journal

    monkeypatch.setattr(backfill, "_iter_rows_for_provider", _fake_iter)
    monkeypatch.setattr(backfill, "_journal_for", _fake_journal)

    # First run: empty journal, all 5 rows novel.
    r1 = backfill.import_provider("claude_code")
    assert r1.rows_seen == 5
    assert r1.rows_novel == 5
    assert journal.exists()
    n_lines = sum(
        1 for line in journal.read_text().splitlines() if line.strip()
    )
    assert n_lines == 5

    # Second run: same 5 rows, all dedup hits, journal unchanged.
    r2 = backfill.import_provider("claude_code")
    assert r2.rows_seen == 5
    assert r2.rows_novel == 0
    n_lines2 = sum(
        1 for line in journal.read_text().splitlines() if line.strip()
    )
    assert n_lines2 == 5  # unchanged


def test_import_provider_preserves_existing_live_hook_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing journal content (from the live hook) survives the
    merge. Novel rows are appended; existing rows stay where they are.
    Critical because the user runs `import` AFTER install, with weeks
    of live-hook rows already in the journal.
    """
    from llmoji import backfill
    from llmoji.scrape import ScrapeRow

    journal = tmp_path / "kaomoji-journal.jsonl"
    # Plant 2 pre-existing live-hook rows.
    _write_journal(journal, [
        _row(ts="2026-04-01T00:00:00Z", assistant_text="live-1"),
        _row(ts="2026-04-02T00:00:00Z", assistant_text="live-2"),
    ])

    fake_rows = [
        # One overlapping row — should dedup against live-1.
        ScrapeRow(
            source="claude_code", model="m1",
            timestamp="2026-04-01T00:00:00Z", cwd="/tmp",
            first_word="(◕‿◕)", assistant_text="live-1",
            surrounding_user="u",
        ),
        # One genuinely novel historical row.
        ScrapeRow(
            source="claude_code", model="m1",
            timestamp="2026-03-15T00:00:00Z", cwd="/tmp",
            first_word="(◕‿◕)", assistant_text="historical-1",
            surrounding_user="u",
        ),
    ]
    def _fake_iter(_name: str):
        _ = _name
        return iter(fake_rows)

    def _fake_journal(_name: str) -> Path:
        _ = _name
        return journal

    monkeypatch.setattr(backfill, "_iter_rows_for_provider", _fake_iter)
    monkeypatch.setattr(backfill, "_journal_for", _fake_journal)

    r = backfill.import_provider("claude_code")
    assert r.rows_seen == 2
    assert r.rows_novel == 1  # only the historical-1 row

    # Read back — live rows preserved, historical row appended.
    rows = [
        json.loads(line)
        for line in journal.read_text().splitlines()
        if line.strip()
    ]
    bodies = [row["assistant_text"] for row in rows]
    assert "live-1" in bodies
    assert "live-2" in bodies
    assert "historical-1" in bodies
    assert len(rows) == 3


def test_import_provider_dry_run_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dry_run=True`` walks the source + counts but never touches the
    journal."""
    from llmoji import backfill
    from llmoji.scrape import ScrapeRow

    journal = tmp_path / "kaomoji-journal.jsonl"
    fake_rows = [
        ScrapeRow(
            source="claude_code", model="m1",
            timestamp="2026-04-28T00:00:00Z", cwd="/tmp",
            first_word="(◕‿◕)", assistant_text="x",
            surrounding_user="u",
        ),
    ]
    def _fake_iter(_name: str):
        _ = _name
        return iter(fake_rows)

    def _fake_journal(_name: str) -> Path:
        _ = _name
        return journal

    monkeypatch.setattr(backfill, "_iter_rows_for_provider", _fake_iter)
    monkeypatch.setattr(backfill, "_journal_for", _fake_journal)

    r = backfill.import_provider("claude_code", dry_run=True)
    assert r.rows_seen == 1
    assert r.rows_novel == 1
    assert not journal.exists()


def test_import_provider_since_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--since`` drops rows whose ts is lexicographically less than
    the cutoff. ISO-8601-with-Z compares correctly under str ordering."""
    from llmoji import backfill
    from llmoji.scrape import ScrapeRow

    journal = tmp_path / "kaomoji-journal.jsonl"
    fake_rows = [
        ScrapeRow(
            source="claude_code", model="m1",
            timestamp=ts, cwd="/tmp",
            first_word="(◕‿◕)", assistant_text=f"row-{ts}",
            surrounding_user="u",
        )
        for ts in (
            "2026-01-01T00:00:00Z",
            "2026-04-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        )
    ]
    def _fake_iter(_name: str):
        _ = _name
        return iter(fake_rows)

    def _fake_journal(_name: str) -> Path:
        _ = _name
        return journal

    monkeypatch.setattr(backfill, "_iter_rows_for_provider", _fake_iter)
    monkeypatch.setattr(backfill, "_journal_for", _fake_journal)

    r = backfill.import_provider(
        "claude_code", since="2026-03-15T00:00:00Z",
    )
    # 2 rows pass the cutoff (2026-04 + 2026-06).
    assert r.rows_seen == 2
    assert r.rows_novel == 2


def test_import_provider_unknown_name_raises() -> None:
    from llmoji.backfill import import_provider

    with pytest.raises(ValueError, match="unknown provider"):
        import_provider("nonexistent")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_import_dispatches_to_import_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``llmoji import claude_code --dry-run`` calls
    ``import_provider`` with dry_run=True."""
    from llmoji import backfill, cli

    captured: dict[str, object] = {}

    def fake_import(name: str, *, since: str | None = None,
                    dry_run: bool = False) -> backfill.ImportResult:
        captured["name"] = name
        captured["since"] = since
        captured["dry_run"] = dry_run
        return backfill.ImportResult(rows_seen=10, rows_novel=4)

    monkeypatch.setattr(backfill, "import_provider", fake_import)

    parser = cli._build_parser()
    args = parser.parse_args(["import", "claude_code", "--dry-run"])
    rc = cli._cmd_import(args)

    assert rc == 0
    assert captured == {"name": "claude_code", "since": None, "dry_run": True}
    out = capsys.readouterr().out
    assert "saw 10" in out
    assert "would append 4" in out
    assert "skipped 6" in out
