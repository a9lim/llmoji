"""Live capture: each provider's ``live_rows``, the journal append, and
the installed shim end to end.

A live row is built by the same decoder as the replay of the same
message, so most tests check one path; the identity tests check that a
live turn is exactly the matching slice of its session's replay.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from llmoji.capture import append_rows, row_key
from llmoji.providers import (
    AntigravityProvider,
    ClaudeCodeProvider,
    CodexProvider,
    GrokProvider,
    get_provider,
)
from llmoji.scrape import ScrapeRow


def _write(path: Path, events: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


def _faces(rows: list[ScrapeRow]) -> list[str]:
    return [r.first_word for r in rows]


def _journal(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# --- Claude Code ---


class _Claude:
    """Builds a transcript the way Claude Code writes one: one entry per
    content block, parentUuid links, tool results as user entries."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._n = 0

    def _add(self, ev: dict[str, Any]) -> str:
        self._n += 1
        uid = f"e{self._n}"
        ev.setdefault("timestamp", f"2026-10-25T00:{self._n // 60:02d}:{self._n % 60:02d}.000Z")
        self.events.append({
            "uuid": uid, "parentUuid": self.events[-1]["uuid"] if self.events else None, **ev,
        })
        return uid

    def user(self, text: str) -> _Claude:
        self._add({"type": "user", "message": {"content": text}})
        return self

    def say(self, text: str, *, model: str = "claude-test", **extra: Any) -> _Claude:
        self._add({
            "type": "assistant", "cwd": "/w",
            "message": {"model": model, "content": [{"type": "text", "text": text}]}, **extra,
        })
        return self

    def tool(self) -> _Claude:
        self._add({"type": "assistant", "cwd": "/w", "message": {
            "model": "claude-test", "content": [{"type": "tool_use", "id": "t", "name": "Bash"}],
        }})
        self._add({"type": "user", "message": {
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
        }})
        return self

    def save(self, tmp_path: Path) -> Path:
        return _write(tmp_path / "t.jsonl", self.events)


def _claude_live(path: Path, **payload: Any) -> list[ScrapeRow]:
    return ClaudeCodeProvider().live_rows({"transcript_path": str(path), "cwd": "/w", **payload})


@pytest.mark.parametrize("build,expected", [
    (lambda c: c.say("(◕‿◕) answer").tool(), ["(◕‿◕)"]),
    (lambda c: c.tool().say("(｡•̀ᴗ-)✧ found it"), ["(｡•̀ᴗ-)✧"]),
    (lambda c: c.say("(・_・) looking").tool().say("(◕‿◕) done"), ["(・_・)", "(◕‿◕)"]),
    (lambda c: c.say("plain prose").tool().say("(◕‿◕) done"), ["(◕‿◕)"]),
    (lambda c: c.tool().tool(), []),
])
def test_claude_one_row_per_kaomoji_led_text_entry(tmp_path: Path, build: Any, expected: list[str]) -> None:
    c = _Claude().user("go")
    rows = _claude_live(build(c).save(tmp_path))
    assert _faces(rows) == expected
    assert all(r.surrounding_user == "go" and r.model == "claude-test" for r in rows)


def test_claude_live_is_the_last_turn_of_the_replay(tmp_path: Path) -> None:
    path = (
        _Claude().user("first").say("(・_・) one").tool().say("(◕‿◕) two")
        .user("second").say("(＾▽＾) three").save(tmp_path)
    )
    replay = list(ClaudeCodeProvider().replay(path))
    live = _claude_live(path, last_assistant_message="(＾▽＾) three")
    assert _faces(replay) == ["(・_・)", "(◕‿◕)", "(＾▽＾)"]
    assert [row_key(r) for r in live] == [row_key(r) for r in replay[2:]]
    assert live[0].surrounding_user == "second"


def test_claude_skill_body_mid_turn_neither_ends_the_turn_nor_becomes_user_text(tmp_path: Path) -> None:
    path = (
        _Claude().user("do it").say("(・_・) loading the skill")
        .user("Base directory for this skill: /x\n\n# Skill").say("(◕‿◕) done").save(tmp_path)
    )
    rows = _claude_live(path)
    assert _faces(rows) == ["(・_・)", "(◕‿◕)"]
    assert {r.surrounding_user for r in rows} == {"do it"}


def test_claude_task_notification_starts_a_turn_but_isnt_the_prompt(tmp_path: Path) -> None:
    """Claude Code wakes the model for a notification and a Stop follows,
    so the turn starts there; the prompt is still the human's. The
    notification roots a parentUuid chain of its own, so the walk falls
    back to the latest prompt in file order."""
    c = _Claude().user("build it").say("(・_・) started")
    c.user("<task-notification>\n<task-id>x</task-id>\n</task-notification>")
    c.events[-1]["parentUuid"] = None
    c.say("(◕‿◕) the task finished")
    rows = _claude_live(c.save(tmp_path))
    assert [(r.first_word, r.surrounding_user) for r in rows] == [("(◕‿◕)", "build it")]
    assert {r.surrounding_user for r in ClaudeCodeProvider().replay(c.save(tmp_path))} == {"build it"}


def test_claude_reminder_block_in_front_of_the_prompt_is_skipped(tmp_path: Path) -> None:
    c = _Claude()
    c._add({"type": "user", "message": {"content": [
        {"type": "text", "text": "<system-reminder>\nnoise\n</system-reminder>"},
        {"type": "text", "text": "the real question"},
    ]}})
    c.say("(◕‿◕) answer")
    assert _claude_live(c.save(tmp_path))[0].surrounding_user == "the real question"


def test_claude_sidechain_turns_are_skipped(tmp_path: Path) -> None:
    c = _Claude().user("go").say("(◕‿◕) main").say("(・_・) sub", isSidechain=True)
    assert _claude_live(c.save(tmp_path)) == []
    assert _faces(list(ClaudeCodeProvider().replay(c.save(tmp_path)))) == ["(◕‿◕)"]


def test_claude_final_message_missing_from_transcript_is_taken_from_payload(tmp_path: Path) -> None:
    path = _Claude().user("go").say("(・_・) checking").tool().save(tmp_path)
    rows = _claude_live(path, last_assistant_message="(◕‿◕) all good")
    assert _faces(rows) == ["(・_・)", "(◕‿◕)"]
    final = rows[-1]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", final.timestamp)  # fire time
    assert (final.model, final.surrounding_user, final.cwd) == ("claude-test", "go", "/w")


def test_claude_final_message_already_written_is_not_doubled(tmp_path: Path) -> None:
    path = _Claude().user("go").say("(◕‿◕) all good").save(tmp_path)
    assert _faces(_claude_live(path, last_assistant_message="(◕‿◕) all good\n")) == ["(◕‿◕)"]


def test_claude_earlier_message_cannot_stand_in_for_a_missing_final_one(tmp_path: Path) -> None:
    path = _Claude().user("go").say("(◕‿◕) Done.").tool().save(tmp_path)
    rows = _claude_live(path, last_assistant_message="(◕‿◕) Done. Here's what changed.")
    assert [r.assistant_text for r in rows] == ["Done.", "Done. Here's what changed."]


# --- Codex ---


def _codex(turns: list[tuple[list[str], list[str]]], *, shape: str = "item",
           subagent: bool = False, model: str = "gpt-test") -> list[dict[str, Any]]:
    """``turns`` is ``[(user_items, agent_messages), ...]``."""
    meta: dict[str, Any] = {"cwd": "/w"}
    if subagent:
        meta["thread_source"] = "subagent"
    events: list[dict[str, Any]] = [{"type": "session_meta", "payload": meta}]
    n = 0
    for users, messages in turns:
        events.append({"type": "turn_context", "payload": {"model": model, "cwd": "/w"}})
        for u in users:
            events.append({"type": "response_item", "payload": {
                "role": "user", "content": [{"type": "input_text", "text": u}],
            }})
        for m in messages:
            n += 1
            ts = f"2026-10-25T00:00:{n:02d}.500Z"
            if shape in ("item", "both"):
                events.append({"type": "event_msg", "timestamp": ts, "payload": {
                    "type": "item_completed",
                    "item": {"type": "AgentMessage", "content": [{"type": "Text", "text": m}]},
                }})
            if shape in ("legacy", "both"):
                events.append({"type": "event_msg", "timestamp": ts, "payload": {
                    "type": "agent_message", "message": m,
                }})
            # The model-API mirror Codex also writes; a forked thread
            # replays its parent's history as these. Never read.
            events.append({"type": "response_item", "timestamp": ts, "payload": {
                "role": "assistant", "content": [{"type": "output_text", "text": m}],
            }})
    return events


def _codex_live(path: Path, **payload: Any) -> list[ScrapeRow]:
    return CodexProvider().live_rows({"transcript_path": str(path), **payload})


@pytest.mark.parametrize("shape", ["item", "legacy", "both"])
def test_codex_each_message_once_in_either_shape(tmp_path: Path, shape: str) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([(["q"], ["(・_・) a", "prose", "(◕‿◕) b"])], shape=shape))
    rows = _codex_live(path)
    assert _faces(rows) == ["(・_・)", "(◕‿◕)"]
    assert {(r.model, r.cwd, r.surrounding_user) for r in rows} == {("gpt-test", "/w", "q")}


def test_codex_injected_user_items_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([
        (["# AGENTS.md\nrules", "real prompt", "<environment_context>x"], ["(◕‿◕) ok"]),
    ]))
    assert _codex_live(path)[0].surrounding_user == "real prompt"


@pytest.mark.parametrize("injected", [
    "<skill>\n<name>x</name>", "<subagent_notification>\n{}",
    '<codex_internal_context source="goal">\nContinue', "<task-notification>\n<task-id>x",
])
def test_codex_injected_items_never_become_the_prompt(tmp_path: Path, injected: str) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([(["real prompt"], ["(・_・) one"]), ([injected], ["(◕‿◕) two"])]))
    assert _codex_live(path)[0].surrounding_user == "real prompt"


def test_codex_another_agents_brief_is_the_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([(["<task>\nAudit the module"], ["(◕‿◕) on it"])]))
    assert _codex_live(path)[0].surrounding_user == "<task>\nAudit the module"


def test_codex_live_is_the_last_turn_and_inherits_the_prompt(tmp_path: Path) -> None:
    """A turn Codex starts itself has no user item; it carries the
    prompt that set it going."""
    path = _write(tmp_path / "r.jsonl", _codex([(["start"], ["(・_・) one"]), ([], ["(◕‿◕) two"])]))
    replay = list(CodexProvider().replay(path))
    live = _codex_live(path)
    assert [row_key(r) for r in live] == [row_key(r) for r in replay[1:]]
    assert live[0].surrounding_user == "start"


def test_codex_payload_model_fills_a_turn_context_without_one(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([(["q"], ["(◕‿◕) ok"])], model=""))
    assert _codex_live(path, model="gpt-payload")[0].model == "gpt-payload"


def test_codex_subagent_rollouts_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.jsonl", _codex([(["q"], ["(◕‿◕) ok"])], subagent=True))
    assert _codex_live(path) == [] and list(CodexProvider().replay(path)) == []


def test_codex_replays_zstd_rollouts(tmp_path: Path) -> None:
    from llmoji._util import can_read_zstd

    if not can_read_zstd():
        pytest.skip("no zstd decoder")
    try:
        from compression import zstd  # type: ignore[import-not-found]

        compress = zstd.compress
    except ImportError:
        import zstandard  # type: ignore[import-not-found]

        compress = zstandard.ZstdCompressor().compress
    raw = "".join(json.dumps(e) + "\n" for e in _codex([(["q"], ["(◕‿◕) ok"])]))
    path = tmp_path / "rollout-1.jsonl.zst"
    path.write_bytes(compress(raw.encode()))
    assert _faces(list(CodexProvider().replay(path))) == ["(◕‿◕)"]


# --- Grok ---


def _grok_update(kind: str, *, text: str = "", pid: str = "", ms: int = 0, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {"agentTimestampMs": ms}
    if pid:
        meta["promptId"] = pid
    update: dict[str, Any] = {"sessionUpdate": kind, **extra}
    if kind.endswith("_chunk"):
        update["content"] = {"type": "text", "text": text}
    return {"params": {"update": update, "_meta": meta}}


def _grok_session(root: Path, sid: str, turns: list[tuple[str, str, list[list[str]]]],
                  *, live: bool = False) -> Path:
    """``turns`` is ``[(prompt_id, user, [[chunk, ...], ...]), ...]``;
    each inner list is one message's chunks. ``live`` leaves out the
    last turn's ``turn_completed``, which Grok writes after the hook."""
    updates: list[dict[str, Any]] = []
    ms = 1_761_350_400_000
    for pid, user, messages in turns:
        updates.append(_grok_update("user_message_chunk", text=user, ms=ms))
        for chunks in messages:
            ms += 1000
            updates.extend(_grok_update("agent_message_chunk", text=c, pid=pid, ms=ms) for c in chunks)
            updates.append(_grok_update("tool_call", pid=pid, ms=ms + 10))
        if not (live and pid == turns[-1][0]):
            updates.append(_grok_update("turn_completed", prompt_id=pid, ms=ms + 20,
                                        usage={"modelUsage": {"grok-turn": {"outputTokens": 5}}}))
    session = root / "sessions" / "%2Fw" / sid
    _write(session / "updates.jsonl", updates)
    _write(session / "chat_history.jsonl", [{"type": "assistant", "model_id": "grok-latest"}])
    (session / "summary.json").write_text(json.dumps({"info": {"cwd": "/w"}}))
    return session / "updates.jsonl"


def test_grok_live_reads_the_prompts_messages_and_joins_chunks(tmp_path: Path) -> None:
    turns = [
        ("p1", "first", [["(・_・) one"]]),
        ("p2", "second", [["(◕‿", "◕) two ", "parts"], ["prose"]]),
    ]
    path = _grok_session(tmp_path, "s1", turns, live=True)
    rows = GrokProvider().live_rows({"transcript_path": str(path), "promptId": "p2", "reason": "end_turn"})
    assert [(r.first_word, r.assistant_text, r.surrounding_user) for r in rows] == [
        ("(◕‿◕)", "two parts", "second"),
    ]
    # turn_completed lands after the hook, so live rows take chat_history's
    # model; a replay of the finished session has the turn's own.
    assert rows[0].model == "grok-latest"
    done = _grok_session(tmp_path / "later", "s1", turns)
    assert [r.model for r in GrokProvider().replay(done)] == ["grok-turn", "grok-turn"]


@pytest.mark.parametrize("extra", [{"reason": "shutdown"}, {"reason": "end_turn", "subagentType": "x"}])
def test_grok_teardown_and_subagent_stops_journal_nothing(tmp_path: Path, extra: dict[str, Any]) -> None:
    path = _grok_session(tmp_path, "s1", [("p1", "q", [["(◕‿◕) ok"]])])
    assert GrokProvider().live_rows({"transcript_path": str(path), "promptId": "p1", **extra}) == []


def test_grok_replay_leaves_out_subagent_sessions(tmp_path: Path, sandbox_home: Path) -> None:
    root = sandbox_home / ".grok"
    _grok_session(root, "parent", [("p1", "q", [["(◕‿◕) ok"]])])
    _grok_session(root, "child", [("c1", "task", [["(・_・) sub"]])])
    _write(root / "sessions" / "%2Fw" / "parent" / "subagents" / "meta.json", [{"id": "child"}])
    assert [p.parent.name for p in GrokProvider().session_files()] == ["parent"]


# --- Antigravity ---


def test_antigravity_reads_planner_responses_after_the_last_user_input(tmp_path: Path) -> None:
    path = _write(tmp_path / "transcript_full.jsonl", [
        {"type": "USER_INPUT", "content": "<USER_REQUEST>\nold\n</USER_REQUEST>"},
        {"type": "PLANNER_RESPONSE", "content": "(・_・) old turn", "created_at": "t0"},
        {"type": "USER_INPUT", "content": "<USER_REQUEST>\nnew prompt\n</USER_REQUEST>"},
        {"type": "PLANNER_RESPONSE", "content": "(◕‿◕) new turn", "created_at": "t1"},
        {"type": "PLANNER_RESPONSE", "content": "", "created_at": "t2"},
    ])
    payload = {"transcriptPath": str(path), "modelName": "gemini-test", "workspacePaths": ["/w"]}
    rows = AntigravityProvider().live_rows(payload)
    assert [(r.first_word, r.surrounding_user, r.model, r.cwd, r.timestamp) for r in rows] == [
        ("(◕‿◕)", "new prompt", "gemini-test", "/w", "t1"),
    ]
    assert AntigravityProvider().live_rows({**payload, "lastUserInput": "raw"})[0].surrounding_user == "raw"


# --- journal append ---


def _row(ts: str, text: str = "(◕‿◕) ok") -> ScrapeRow:
    face, _, body = text.partition(" ")
    return ScrapeRow("t", "m", ts, None, body, face, "u")


def test_append_skips_rows_a_refire_already_wrote(tmp_path: Path) -> None:
    journal = tmp_path / "j.jsonl"
    first = [_row("2026-10-25T00:00:01.000Z"), _row("2026-10-25T00:00:02.000Z", "(・_・) two")]
    assert append_rows(journal, first) == 2
    assert append_rows(journal, [*first, _row("2026-10-25T00:00:03.000Z", "(＾▽＾) three")]) == 1
    assert len(_journal(journal)) == 3


def test_append_keeps_identical_rows_once_per_copy(tmp_path: Path) -> None:
    journal = tmp_path / "j.jsonl"
    twice = [_row("2026-10-25T00:00:01Z"), _row("2026-10-25T00:00:01Z")]
    assert append_rows(journal, twice) == 2
    assert append_rows(journal, [*twice, _row("2026-10-25T00:00:01Z")]) == 1


def test_append_folds_a_fire_time_fallback_into_its_message(tmp_path: Path) -> None:
    """The final-message fallback is stamped with fire time; when a
    re-fire then finds the message in the transcript, it's the same
    message."""
    journal = tmp_path / "j.jsonl"
    append_rows(journal, [_row("2026-10-25T00:01:00Z")])
    assert append_rows(journal, [_row("2026-10-25T00:00:59.250Z")]) == 0


def test_append_matches_rows_split_by_an_older_splitter(tmp_path: Path) -> None:
    journal = tmp_path / "j.jsonl"
    journal.write_text(json.dumps({
        "ts": "2026-10-25T00:00:01.000Z", "model": "m", "cwd": "", "kaomoji": "(•ᴗ•) **",
        "user_text": "u", "assistant_text": "Task 1** done",
    }) + "\n")
    assert append_rows(journal, [_row("2026-10-25T00:00:01.000Z", "(•ᴗ•) **Task 1** done")]) == 0


# --- the installed shim ---


def _fire(provider: str, payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    p = get_provider(provider)
    p.install()
    return subprocess.run(
        ["sh", str(p.hook_path)], input=json.dumps(payload), capture_output=True, text=True,
    )


def test_shim_journals_a_turn(tmp_path: Path) -> None:
    path = _Claude().user("go").say("(◕‿◕) hello").save(tmp_path)
    r = _fire("claude_code", {"transcript_path": str(path)})
    assert (r.returncode, r.stdout) == (0, "")
    assert [row["kaomoji"] for row in _journal(ClaudeCodeProvider().journal_path)] == ["(◕‿◕)"]


def test_shim_answers_antigravity_on_stdout() -> None:
    r = _fire("antigravity", {})
    assert (r.returncode, r.stdout) == (0, "{}\n")


def test_shim_never_fails_the_harness_and_logs_the_error(sandbox_home: Path) -> None:
    r = _fire("codex", {"transcript_path": 7})  # not a path
    assert r.returncode == 0
    r = _fire("codex", "not an object")  # type: ignore[arg-type]
    assert r.returncode == 0
    p = get_provider("codex")
    r = subprocess.run(["sh", str(p.hook_path)], input="{", capture_output=True, text=True)
    assert r.returncode == 0
    log = sandbox_home / ".llmoji" / "hook-errors.log"
    assert "JSONDecodeError" in log.read_text()
