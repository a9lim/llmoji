"""``llmoji import``: replaying session history into a journal without
counting a message twice."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llmoji import cli
from llmoji.capture import import_provider
from llmoji.providers import Provider
from llmoji.scrape import ScrapeRow


def _write_journal(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _jrow(ts: str, *, kaomoji: str = "(◕‿◕)", text: str = "same words") -> dict[str, Any]:
    return {"ts": ts, "model": "m1", "cwd": "", "kaomoji": kaomoji, "user_text": "u",
            "assistant_text": text}


def _replayed(ts: str, text: str = "same words", face: str = "(◕‿◕)") -> ScrapeRow:
    return ScrapeRow("claude_code-hook", "m1", ts, None, text, face, "u")


class _Fake(Provider):
    name = "fake"

    def __init__(self, journal: Path, rows: list[ScrapeRow]) -> None:
        self.journal_path = journal
        self._rows = rows

    def session_files(self) -> list[Path]:
        return [Path("session.jsonl")]

    def replay(self, path: Path) -> Iterator[ScrapeRow]:
        return iter(self._rows)


def _import(monkeypatch: pytest.MonkeyPatch, journal: Path, rows: list[ScrapeRow], **kw: Any) -> Any:
    monkeypatch.setattr("llmoji.providers.get_provider", lambda _n: _Fake(journal, rows))
    return import_provider("fake", **kw)


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_appends_only_novel_rows_and_is_rerunnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T10:00:00.000Z")])
    rows = [_replayed("2026-04-29T10:00:00.000Z"), _replayed("2026-04-29T11:00:00.000Z", "new")]
    r = _import(monkeypatch, journal, rows)
    assert (r.rows_seen, r.rows_novel) == (2, 1)
    assert [row["assistant_text"] for row in _lines(journal)] == ["same words", "new"]
    assert _import(monkeypatch, journal, rows).rows_novel == 0


def test_dry_run_counts_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    r = _import(monkeypatch, journal, [_replayed("2026-04-29T10:00:00.000Z")], dry_run=True)
    assert r.rows_novel == 1 and not journal.exists()


def test_matching_ignores_where_an_older_splitter_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Before 3.0 the face could swallow markdown (``(•ᴗ•) **``). The
    whitespace-free message text is the same however it was split."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T10:00:00.000Z", kaomoji="(•ᴗ•) **", text="Task 1** done")])
    r = _import(monkeypatch, journal, [_replayed("2026-04-29T10:00:00.000Z", "**Task 1** done", "(•ᴗ•)")])
    assert r.rows_novel == 0


def test_folds_a_fire_time_live_row_into_its_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T17:35:53Z")])
    assert _import(monkeypatch, journal, [_replayed("2026-04-29T17:35:27.011Z")]).rows_novel == 0


def test_fold_ignores_live_rows_outside_the_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T17:00:00Z"), _jrow("2026-05-02T17:00:00Z")])
    assert _import(monkeypatch, journal, [_replayed("2026-04-29T18:00:00.000Z")]).rows_novel == 1


def test_fold_needs_the_same_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T17:35:53Z", kaomoji="(╥﹏╥)")])
    assert _import(monkeypatch, journal, [_replayed("2026-04-29T17:35:27.011Z")]).rows_novel == 1


def test_only_whole_second_stamps_are_fold_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A millisecond row came from a replay; if its message isn't
    replayed any more it still mustn't absorb an earlier message."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T11:00:00.000Z")])
    assert _import(monkeypatch, journal, [_replayed("2026-04-29T10:00:00.000Z")]).rows_novel == 1


def test_one_live_row_stands_in_for_one_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T18:00:10Z")])
    rows = [_replayed("2026-04-29T17:00:00.000Z"), _replayed("2026-04-29T18:00:00.000Z")]
    assert _import(monkeypatch, journal, rows).rows_novel == 1


def test_exact_matches_are_claimed_before_folding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row an earlier import wrote matches its replay exactly; the
    fold mustn't hand it to an earlier copy of the same words first."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T18:00:00.000Z")])
    rows = [_replayed("2026-04-29T17:00:00.000Z"), _replayed("2026-04-29T18:00:00.000Z")]
    assert _import(monkeypatch, journal, rows).rows_novel == 1


def test_a_forked_sessions_copy_is_a_duplicate_not_a_fold_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T17:00:00.000Z"), _jrow("2026-04-29T19:00:00Z")])
    rows = [
        _replayed("2026-04-29T17:00:00.000Z"),
        _replayed("2026-04-29T17:00:00.000Z"),  # the fork's copy
        _replayed("2026-04-29T18:55:00.000Z"),
    ]
    r = _import(monkeypatch, journal, rows)
    assert (r.rows_seen, r.rows_novel) == (3, 0)


def test_since_gates_appends_but_not_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Message one is in the journal and before --since. It still claims
    its own row, so the same words four seconds later aren't folded into
    it."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [_jrow("2026-04-29T10:00:00.000Z", text="Done.")])
    rows = [_replayed("2026-04-29T10:00:00.000Z", "Done."), _replayed("2026-04-29T10:00:04.000Z", "Done.")]
    r = _import(monkeypatch, journal, rows, since="2026-04-29T10:00:01.000Z")
    assert (r.rows_seen, r.rows_novel) == (1, 1)


def test_matched_rows_are_replaced_by_their_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The replay is authoritative: a fire-time row gets its message time
    and model, an old split and an injected prompt are repaired, and
    lines that aren't rows are kept."""
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [
        {**_jrow("2026-04-29T17:35:53Z"), "model": ""},
        _jrow("2026-04-29T18:00:00.000Z", kaomoji="(•ᴗ•) **", text="Task** done"),
    ])
    with journal.open("a") as f:
        f.write("not json\n")
    rows = [
        _replayed("2026-04-29T17:35:27.011Z"),
        _replayed("2026-04-29T18:00:00.000Z", "**Task** done", "(•ᴗ•)"),
    ]
    r = _import(monkeypatch, journal, rows)
    assert (r.rows_novel, r.rows_updated) == (0, 2)
    lines = journal.read_text().splitlines()
    assert lines[2] == "not json"
    first, second = (json.loads(line) for line in lines[:2])
    assert (first["ts"], first["model"]) == ("2026-04-29T17:35:27.011Z", "m1")
    assert (second["kaomoji"], second["assistant_text"]) == ("(•ᴗ•)", "**Task** done")
    assert _import(monkeypatch, journal, rows).rows_updated == 0


def test_dry_run_and_since_gate_repairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = tmp_path / "j.jsonl"
    _write_journal(journal, [{**_jrow("2026-04-29T10:00:00.000Z"), "user_text": "<task-notification>x"}])
    before = journal.read_text()
    rows = [_replayed("2026-04-29T10:00:00.000Z")]
    assert _import(monkeypatch, journal, rows, dry_run=True).rows_updated == 1
    assert _import(monkeypatch, journal, rows, since="2026-05-01T00:00:00Z").rows_updated == 0
    assert journal.read_text() == before
    assert _import(monkeypatch, journal, rows).rows_updated == 1
    assert _lines(journal)[0]["user_text"] == "u"


def test_non_importable_provider_is_refused() -> None:
    with pytest.raises(ValueError, match="replayable"):
        import_provider("antigravity")


def test_end_to_end_with_a_real_transcript(sandbox_home: Path) -> None:
    transcript = sandbox_home / ".claude" / "projects" / "p" / "s.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("".join(json.dumps(e) + "\n" for e in [
        {"type": "user", "uuid": "u", "parentUuid": None, "timestamp": "2026-10-25T00:00:00.000Z",
         "message": {"content": "hi"}},
        {"type": "assistant", "uuid": "a", "parentUuid": "u", "timestamp": "2026-10-25T00:00:01.000Z",
         "cwd": "/w", "message": {"model": "claude-test", "content": [{"type": "text", "text": "(◕‿◕) hey"}]}},
    ]))
    assert import_provider("claude_code").rows_novel == 1
    journal = sandbox_home / ".claude" / "kaomoji-journal.jsonl"
    assert _lines(journal) == [{
        "ts": "2026-10-25T00:00:01.000Z", "model": "claude-test", "cwd": "/w",
        "kaomoji": "(◕‿◕)", "user_text": "hi", "assistant_text": "hey",
    }]
    assert import_provider("claude_code").rows_novel == 0


# --- CLI ---


def test_cli_without_a_harness_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["import", "--yes"]) == 2
    assert "no harnesses detected" in capsys.readouterr().err


def test_cli_imports_every_detected_importable_harness(
    sandbox_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for d in (".claude", ".codex", ".gemini/antigravity-cli"):
        (sandbox_home / d).mkdir(parents=True)
    seen: list[tuple[str, Any, bool]] = []

    def fake(name: str, *, since: Any, dry_run: bool) -> Any:
        from llmoji.capture import ImportResult

        seen.append((name, since, dry_run))
        return ImportResult(0, 0)

    monkeypatch.setattr("llmoji.capture.import_provider", fake)
    assert cli.main(["import", "--yes", "--dry-run", "--since", "2026-01-01T00:00:00Z"]) == 0
    assert seen == [("claude_code", "2026-01-01T00:00:00Z", True), ("codex", "2026-01-01T00:00:00Z", True)]


def test_cli_rejects_a_harness_with_no_history() -> None:
    with pytest.raises(SystemExit):
        cli.main(["import", "antigravity"])
