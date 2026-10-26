"""Claude Code (``~/.claude``).

The hook is a ``Stop`` handler in ``settings.json``. Transcripts live at
``projects/<project>/<session>.jsonl`` with one entry per content block,
so a tool-heavy turn holds several text entries; each kaomoji-led one
becomes a row.

- Sidechain (subagent) entries carry ``isSidechain`` and are skipped.
- A row's ``user_text`` comes from walking ``parentUuid`` back past
  tool results, skill bodies, reminders and task notifications to the
  prompt that set the work going.
- The transcript can lag the Stop hook (Claude Code documents this; on a
  real corpus it cost about a quarter of closing messages). When the
  turn's final message isn't written yet, the payload's
  ``last_assistant_message`` stands in for it, stamped with fire time
  until ``llmoji import`` folds it into its replay.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .._util import read_jsonl
from ..scrape import INJECTED_USER_PREFIXES, ScrapeRow, kaomoji_row
from ..sources._common import walk_parents_for_user_text
from ..taxonomy import split_leading_kaomoji
from .base import Provider, home_path

# A skill body or a reminder can arrive as a user entry mid-turn; neither
# starts a turn. A task notification does (Claude Code wakes the model
# for it and a Stop follows), but like them it's never the user_text.
_SKILL_BODY = "Base directory for this skill:"
_NOT_A_TURN = (_SKILL_BODY, "<system-reminder>")
INJECTED_PREFIXES = (_SKILL_BODY, *INJECTED_USER_PREFIXES)


def _first_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return str(block["text"])
    return ""


def _content(ev: dict[str, Any]) -> Any:
    msg = ev.get("message")
    return msg.get("content") if isinstance(msg, dict) else None


def _prompt_text(node: dict[str, Any]) -> str:
    """A user entry's own words: the first text block that isn't
    injected, since a reminder can ride in front of the prompt."""
    content = _content(node)
    blocks = [content] if isinstance(content, str) else [
        str(b.get("text") or "") for b in content or []
        if isinstance(b, dict) and b.get("type") == "text"
    ] if isinstance(content, list) else []
    return next((t for t in blocks if t.strip() and not t.lstrip().startswith(INJECTED_PREFIXES)), "")


def _model(ev: dict[str, Any]) -> str:
    msg = ev.get("message")
    return str(msg.get("model") or "") if isinstance(msg, dict) else ""


def _has_block(ev: dict[str, Any], kind: str) -> bool:
    content = _content(ev)
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == kind for b in content
    )


def _is_prompt(ev: dict[str, Any]) -> bool:
    """A user entry that starts a turn: not a tool result, a skill body
    or a reminder."""
    return (
        ev.get("type") == "user"
        and isinstance(_content(ev), (str, list))
        and not _has_block(ev, "tool_result")
        and not _first_text(_content(ev)).lstrip().startswith(_NOT_A_TURN)
    )


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    home_dir = home_path(".claude")
    hooks_dir = home_path(".claude", "hooks")
    settings_path = home_path(".claude", "settings.json")
    journal_path = home_path(".claude", "kaomoji-journal.jsonl")
    doc_path = home_path(".claude", "CLAUDE.md")

    def session_files(self) -> list[Path]:
        return sorted((self.home_dir / "projects").glob("**/*.jsonl"))

    def replay(self, path: Path) -> Iterator[ScrapeRow]:
        return self._rows(read_jsonl(path))

    def live_rows(self, payload: dict[str, Any]) -> list[ScrapeRow]:
        path = Path(str(payload.get("transcript_path") or ""))
        if not path.is_file():
            return []
        events = read_jsonl(path)
        assistants = [e for e in events if e.get("type") == "assistant"]
        if assistants and assistants[-1].get("isSidechain"):
            return []
        start = max(
            (i for i, e in enumerate(events) if _is_prompt(e) and not e.get("isSidechain")),
            default=0,
        )
        rows = list(self._rows(events, start))
        last = str(payload.get("last_assistant_message") or "")
        if last.strip() and not self._final_written(events[start:], last):
            main = [e for e in assistants if not e.get("isSidechain")]
            row = kaomoji_row(
                last,
                source=f"{self.name}-hook",
                model=_model(main[-1]) if main else "",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cwd=str(payload.get("cwd") or ""),
                user=self._user_text(events[-1].get("uuid"), events) or next(
                    (t for e in reversed(events) if e.get("type") == "user"
                     and (t := _prompt_text(e))), "",
                ),
            )
            if row is not None:
                rows.append(row)
        return rows

    @staticmethod
    def _user_text(start_uuid: Any, events: list[dict[str, Any]],
                   by_uuid: dict[str, dict[str, Any]] | None = None) -> str:
        """The prompt a message answers: up its parentUuid chain, past
        tool results and anything injected."""
        if by_uuid is None:
            by_uuid = {e["uuid"]: e for e in events if isinstance(e.get("uuid"), str)}
        return walk_parents_for_user_text(
            start_uuid,
            by_uuid,
            parent_field="parentUuid",
            role_check=lambda node: node.get("type") == "user",
            text_extractor=_prompt_text,
        )

    def _rows(self, events: list[dict[str, Any]], start: int = 0) -> Iterator[ScrapeRow]:
        by_uuid = {e["uuid"]: e for e in events if isinstance(e.get("uuid"), str)}
        # A task notification roots a parentUuid chain of its own, so the
        # walk can end before reaching a prompt; the latest prompt in file
        # order is the fallback.
        latest = ""
        for i, ev in enumerate(events):
            if ev.get("type") == "user":
                latest = _prompt_text(ev) or latest
            if i < start or ev.get("type") != "assistant" or ev.get("isSidechain"):
                continue
            text = _first_text(_content(ev))
            if split_leading_kaomoji(text) is None:
                continue
            row = kaomoji_row(
                text,
                source=f"{self.name}-hook",
                model=_model(ev),
                timestamp=str(ev.get("timestamp") or ""),
                cwd=str(ev.get("cwd") or ""),
                user=self._user_text(ev.get("parentUuid"), events, by_uuid) or latest,
            )
            if row is not None:
                yield row

    @staticmethod
    def _final_written(turn: list[dict[str, Any]], last: str) -> bool:
        """Whether the turn's final message is already in the transcript.

        Judged against the turn's last assistant message only (the
        entries after its last user-type entry, written back to back),
        and only if that message calls no tool: a message that does
        didn't end the turn. Then it's the final one when the payload's
        text starts with its first text block, which also covers a reply
        the transcript split across blocks. Matching any earlier message
        would let ``(◕‿◕) Done.`` hide a missing ``(◕‿◕) Done. Here's…``.
        """
        turn = [
            e for e in turn
            if e.get("type") in ("user", "assistant") and not e.get("isSidechain")
        ]
        users = [i for i, e in enumerate(turn) if e.get("type") == "user"]
        final = turn[users[-1] + 1:] if users else turn
        if not final or any(_has_block(e, "tool_use") for e in final):
            return False
        first = next(
            (t for t in (_first_text(_content(e)).strip() for e in final) if t), "",
        )
        return bool(first) and last.strip().startswith(first)
