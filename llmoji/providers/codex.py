"""Codex (``~/.codex``).

The hook is a ``Stop`` handler in ``hooks.json``, which takes Claude
Code's settings shape. Rollouts live under ``sessions/`` and
``archived_sessions/``; codex-cli zstd-compresses old ones in place.

- Each model message is its own event: an ``item_completed`` whose item
  is an ``AgentMessage`` (codex-cli 0.144+) or an ``agent_message``
  (older). A turn is read in whichever shape it carries,
  ``item_completed`` first, so a release writing both can't count a
  message twice. ``response_item`` assistant messages are never read: a
  forked thread replays its parent's history into the child as those.
- ``turn_context`` opens a turn and names its model and cwd.
- ``user_text`` is the latest real user ``response_item`` so far, so a
  turn Codex starts by itself (a goal step, a subagent result) carries
  the prompt that set it going. Injected items (AGENTS.md, environment
  context, skill bodies, subagent notifications, goal continuations) are
  skipped; a brief from another agent (``<task>``, a gaslamp consult)
  counts as the prompt.
- Subagent threads end in ``SubagentStop``, not ``Stop``; their
  rollouts say ``thread_source == "subagent"`` and are skipped whole.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .._util import read_jsonl
from ..scrape import INJECTED_USER_PREFIXES, ScrapeRow, kaomoji_row
from .base import Provider, home_path

INJECTED_PREFIXES = (
    "# AGENTS.md", "<environment_context>", "<INSTRUCTIONS>", *INJECTED_USER_PREFIXES,
)


def _payload(ev: dict[str, Any]) -> dict[str, Any]:
    p = ev.get("payload")
    return p if isinstance(p, dict) else {}


def _is_subagent(events: list[dict[str, Any]]) -> bool:
    for ev in events:
        if ev.get("type") == "session_meta":
            p = _payload(ev)
            src = p.get("source")
            return p.get("thread_source") == "subagent" or (
                isinstance(src, dict) and "subagent" in src
            )
    return False


def _agent_message(ev: dict[str, Any]) -> tuple[str, str] | None:
    """``(shape, text)`` for an agent-message event: ``"item"`` for
    ``item_completed``, ``"legacy"`` for ``agent_message``."""
    if ev.get("type") != "event_msg":
        return None
    p = _payload(ev)
    if p.get("type") == "agent_message":
        return "legacy", str(p.get("message") or "")
    item = p.get("item")
    if p.get("type") != "item_completed" or not isinstance(item, dict):
        return None
    if item.get("type") != "AgentMessage":
        return None
    content = item.get("content")
    return "item", "".join(
        str(b.get("text") or "")
        for b in (content if isinstance(content, list) else [])
        if isinstance(b, dict) and str(b.get("type") or "").lower() == "text"
    )


def _user_text(ev: dict[str, Any]) -> str:
    p = _payload(ev)
    if ev.get("type") != "response_item" or p.get("role") != "user":
        return ""
    content = p.get("content")
    text = "\n".join(
        str(b.get("text") or "")
        for b in (content if isinstance(content, list) else [])
        if isinstance(b, dict) and b.get("type") == "input_text" and b.get("text")
    )
    return "" if text.lstrip().startswith(INJECTED_PREFIXES) else text


class CodexProvider(Provider):
    name = "codex"
    home_dir = home_path(".codex")
    hooks_dir = home_path(".codex", "hooks")
    settings_path = home_path(".codex", "hooks.json")
    journal_path = home_path(".codex", "kaomoji-journal.jsonl")
    doc_path = home_path(".codex", "AGENTS.md")

    def session_files(self) -> list[Path]:
        return [
            p
            for sub, pattern in (
                ("sessions", "**/rollout-*.jsonl"),
                ("sessions", "**/rollout-*.jsonl.zst"),
                ("archived_sessions", "rollout-*.jsonl"),
                ("archived_sessions", "rollout-*.jsonl.zst"),
            )
            for p in sorted((self.home_dir / sub).glob(pattern))
        ]

    def replay(self, path: Path) -> Iterator[ScrapeRow]:
        return self._rows(read_jsonl(path))

    def live_rows(self, payload: dict[str, Any]) -> list[ScrapeRow]:
        path = Path(str(payload.get("transcript_path") or ""))
        if not path.is_file():
            return []
        return list(
            self._rows(
                read_jsonl(path),
                last_turn_only=True,
                fallback_model=str(payload.get("model") or ""),
            )
        )

    def _rows(
        self,
        events: list[dict[str, Any]],
        *,
        last_turn_only: bool = False,
        fallback_model: str = "",
    ) -> Iterator[ScrapeRow]:
        if _is_subagent(events):
            return
        session_cwd = next(
            (str(_payload(e).get("cwd") or "") for e in events if e.get("type") == "session_meta"),
            "",
        )
        # Events before the first turn_context form a turn of their own.
        turns: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [({}, [])]
        for ev in events:
            if ev.get("type") == "turn_context":
                turns.append((_payload(ev), []))
            turns[-1][1].append(ev)
        user = ""
        for i, (ctx, turn) in enumerate(turns):
            emit = not last_turn_only or i == len(turns) - 1
            messages = [m for m in map(_agent_message, turn) if m is not None]
            shape = "item" if any(s == "item" for s, _ in messages) else "legacy"
            for ev in turn:
                user = _user_text(ev) or user
                msg = _agent_message(ev)
                if not emit or msg is None or msg[0] != shape:
                    continue
                row = kaomoji_row(
                    msg[1],
                    source=f"{self.name}-hook",
                    model=str(ctx.get("model") or "") or fallback_model,
                    timestamp=str(ev.get("timestamp") or ""),
                    cwd=str(ctx.get("cwd") or "") or session_cwd,
                    user=user,
                )
                if row is not None:
                    yield row
