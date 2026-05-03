"""Tests for :mod:`llmoji.sources.openhands_export`.

Builds synthetic OpenHands conversation directories — per-event
JSON files at ``<conv>/events/event-NNNNN-<id>.json`` matching the
SDK's ``MessageEvent`` / ``ActionEvent`` / etc. discriminated-union
shape — and asserts the reader yields the expected
:class:`~llmoji.scrape.ScrapeRow` stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmoji.sources.openhands_export import iter_openhands_export


def _write_event(
    events_dir: Path,
    idx: int,
    *,
    kind: str,
    source: str | None = None,
    role: str | None = None,
    text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write one event-NNNNN-<id>.json file matching the SDK's shape."""
    event: dict[str, Any] = {
        "kind": kind,
        "id": f"id-{idx:05d}",
        "timestamp": f"2026-05-02T12:00:{idx:02d}.000000",
    }
    if source is not None:
        event["source"] = source
    if role is not None:
        event["llm_message"] = {
            "role": role,
            "content": [{"type": "text", "text": text or ""}],
        }
    if extra:
        event.update(extra)
    # SDK's EVENT_NAME_RE requires ``[0-9a-fA-F\-]{8,}`` for the
    # event-id portion of the filename — pad with zeros to satisfy
    # the regex regardless of test idx.
    event_id = f"{idx:08x}-0000-0000-0000-000000000000"
    fname = f"event-{idx:05d}-{event_id}.json"
    (events_dir / fname).write_text(json.dumps(event))


def _setup_conversation(root: Path) -> Path:
    """Create a conversation directory with an empty ``events/`` subdir."""
    events = root / "events"
    events.mkdir(parents=True)
    return events


def test_emits_one_row_per_kaomoji_led_agent_message(tmp_path: Path) -> None:
    """Walk a conversation with mixed user / agent / system / action
    events; emit only one row per kaomoji-led agent ``MessageEvent``,
    paired with the most recent user message text.
    """
    events = _setup_conversation(tmp_path / "conv-001")
    _write_event(
        events, 0, kind="SystemPromptEvent", source="agent",
    )
    _write_event(
        events, 1, kind="MessageEvent", source="user", role="user",
        text="how do I write a fizzbuzz?",
    )
    _write_event(
        events, 2, kind="MessageEvent", source="agent", role="assistant",
        text="(◕ω◕) here's one approach",
    )
    _write_event(
        events, 3, kind="ActionEvent", source="agent",
        extra={"action": "execute_bash", "tool_name": "bash"},
    )
    _write_event(
        events, 4, kind="ObservationEvent", source="environment",
        extra={"observation": "stdout: hello"},
    )
    _write_event(
        events, 5, kind="MessageEvent", source="agent", role="assistant",
        text="(´▽`) all done!",
    )
    _write_event(
        events, 6, kind="MessageEvent", source="agent", role="assistant",
        text="no kaomoji here, gets dropped",
    )

    rows = list(iter_openhands_export([tmp_path]))
    assert len(rows) == 2
    first, second = rows
    assert first.source == "openhands-export"
    assert first.first_word == "(◕ω◕)"
    assert first.assistant_text == "here's one approach"
    assert first.surrounding_user == "how do I write a fizzbuzz?"
    # Per-event model isn't on the wire — bundle layer slugs to
    # "unknown" downstream. Asserting None here pins the contract.
    assert first.model is None
    # Both rows from the same turn share the originating user prompt.
    assert second.first_word == "(´▽`)"
    assert second.surrounding_user == "how do I write a fizzbuzz?"


def test_event_index_drives_chronology(tmp_path: Path) -> None:
    """Events sort by the NNNNN file-index prefix, not by timestamp.
    Mixing the timestamp order vs. file order should still yield rows
    in file-index order.
    """
    events = _setup_conversation(tmp_path / "conv")
    # Index 0: user. Index 1: agent. Order on disk doesn't matter
    # because rglob returns unsorted; the reader sorts by index.
    _write_event(events, 1, kind="MessageEvent", source="agent", role="assistant",
                 text="(°▽°) second")
    _write_event(events, 0, kind="MessageEvent", source="user", role="user",
                 text="prompt")
    rows = list(iter_openhands_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(°▽°)"
    assert rows[0].surrounding_user == "prompt"


def test_recursive_walk_across_conversations(tmp_path: Path) -> None:
    """Multiple conversation directories under the export root all
    get walked; rows from different conversations are independent
    (user_text doesn't bleed across)."""
    a = _setup_conversation(tmp_path / "conv-A")
    b = _setup_conversation(tmp_path / "conv-B")
    _write_event(a, 0, kind="MessageEvent", source="user", role="user",
                 text="A prompt")
    _write_event(a, 1, kind="MessageEvent", source="agent", role="assistant",
                 text="(◕‿◕) A reply")
    _write_event(b, 0, kind="MessageEvent", source="user", role="user",
                 text="B prompt")
    _write_event(b, 1, kind="MessageEvent", source="agent", role="assistant",
                 text="(╥﹏╥) B reply")

    rows = list(iter_openhands_export([tmp_path]))
    assert len(rows) == 2
    by_face = {r.first_word: r for r in rows}
    assert by_face["(◕‿◕)"].surrounding_user == "A prompt"
    assert by_face["(╥﹏╥)"].surrounding_user == "B prompt"


def test_skips_non_message_kinds(tmp_path: Path) -> None:
    """ActionEvent / ObservationEvent / SystemPromptEvent /
    AgentErrorEvent etc. are skipped — they don't carry the
    user-visible model prose the cross-corpus contract pins on.
    """
    events = _setup_conversation(tmp_path / "conv")
    _write_event(events, 0, kind="ActionEvent", source="agent")
    _write_event(events, 1, kind="ObservationEvent", source="environment")
    _write_event(events, 2, kind="AgentErrorEvent", source="agent")
    _write_event(events, 3, kind="PauseEvent", source="user")
    rows = list(iter_openhands_export([tmp_path]))
    assert rows == []


def test_role_mismatch_dropped(tmp_path: Path) -> None:
    """A MessageEvent with ``source: "agent"`` but
    ``llm_message.role != "assistant"`` is rejected defensively —
    better drop a row than misattribute it to the wrong role.
    """
    events = _setup_conversation(tmp_path / "conv")
    _write_event(events, 0, kind="MessageEvent", source="agent", role="system",
                 text="(◕ω◕) shouldn't fire")
    rows = list(iter_openhands_export([tmp_path]))
    assert rows == []


def test_direct_events_dir(tmp_path: Path) -> None:
    """User can point directly at a single conversation's
    ``events/`` directory rather than the parent root — convenient
    for one-off conversation imports."""
    events = _setup_conversation(tmp_path / "conv")
    _write_event(events, 0, kind="MessageEvent", source="user", role="user",
                 text="q")
    _write_event(events, 1, kind="MessageEvent", source="agent", role="assistant",
                 text="(◕▿◕) a")
    rows = list(iter_openhands_export([events]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◕▿◕)"


def test_malformed_event_files_skipped(tmp_path: Path) -> None:
    """Junk files in ``events/`` shouldn't kill the walk."""
    events = _setup_conversation(tmp_path / "conv")
    _write_event(events, 0, kind="MessageEvent", source="user", role="user",
                 text="prompt")
    # Wrong filename pattern — not picked up by the regex.
    (events / "stray.json").write_text(json.dumps({"foo": "bar"}))
    # Right pattern, wrong content.
    (events / "event-00001-bad.json").write_text("{ not json")
    _write_event(events, 2, kind="MessageEvent", source="agent", role="assistant",
                 text="(◕‿◕) ok")
    rows = list(iter_openhands_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◕‿◕)"
