"""Grok Build (``~/.grok``).

Grok reads hook files from ``~/.grok/hooks/*.json`` in Claude Code's
settings shape; llmoji owns ``llmoji.json`` there and deletes it on
uninstall. The kaomoji block goes in ``~/.grok/AGENTS.md``, the global
rules file. A per-turn nudge isn't possible: Grok discards an allowing
``UserPromptSubmit`` hook's output.

- ``transcript_path`` names the session's ``updates.jsonl`` (the ACP
  update stream). ``agent_message_chunk`` updates carry
  ``_meta.promptId``; a run of consecutive chunks from one prompt is one
  message, and a turn's ``user_text`` is the ``user_message_chunk``
  text just before its first chunk.
- Grok fires a second, observe-only ``Stop`` at session teardown; only
  ``reason == "end_turn"`` is journaled.
- The Stop payload names no model, and ``turn_completed`` (whose
  ``modelUsage`` does) lands after the hook runs, so live rows take the
  latest ``model_id`` in the sibling ``chat_history.jsonl``. A replay
  prefers the turn's own ``modelUsage``.
- Subagent turns end in ``SubagentStop``. Their sessions sit in the
  normal tree, listed under the parent's ``subagents/``, and are left
  out of replays.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._util import read_jsonl
from ..scrape import ScrapeRow, kaomoji_row
from .base import Provider, SettingsCorruptError, home_path, load_json_strict


def _iso_seconds(ms: Any) -> str:
    if not isinstance(ms, (int, float)):
        return ""
    return datetime.fromtimestamp(int(ms // 1000), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _messages(
    updates: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str]], dict[str, str], dict[str, str]]:
    """``(messages, user_by_prompt, model_by_prompt)``; each message is
    ``(prompt_id, text, ts)`` in stream order."""
    messages: list[tuple[str, str, str]] = []
    users: dict[str, str] = {}
    models: dict[str, str] = {}
    user = ""
    ucur: str | None = None
    cur: list[str] | None = None
    for row in updates:
        params = row.get("params")
        if not isinstance(params, dict):
            continue
        u = params.get("update")
        meta = params.get("_meta")
        if not isinstance(u, dict) or not isinstance(meta, dict):
            continue
        kind = u.get("sessionUpdate") or ""
        content = u.get("content")
        text = str(content.get("text") or "") if isinstance(content, dict) else ""
        pid = str(meta.get("promptId") or "")
        if kind == "agent_message_chunk" and pid:
            if cur is not None and cur[0] == pid:
                cur[1] += text
            else:
                if cur is not None:
                    messages.append((cur[0], cur[1], cur[2]))
                users.setdefault(pid, user)
                cur = [pid, text, _iso_seconds(meta.get("agentTimestampMs"))]
            ucur = None
            continue
        if cur is not None:
            messages.append((cur[0], cur[1], cur[2]))
            cur = None
        if kind == "user_message_chunk":
            ucur = text if ucur is None else ucur + text
            user = ucur
            continue
        ucur = None
        usage = u.get("usage") if kind == "turn_completed" else None
        by_model = usage.get("modelUsage") if isinstance(usage, dict) else None
        if isinstance(by_model, dict) and by_model:
            # The model that wrote most of the turn.
            tokens = {
                m: (v.get("outputTokens") or 0) if isinstance(v, dict) else 0
                for m, v in by_model.items()
            }
            models[str(u.get("prompt_id") or "")] = max(tokens, key=tokens.__getitem__)
    if cur is not None:
        messages.append((cur[0], cur[1], cur[2]))
    return messages, users, models


class GrokProvider(Provider):
    name = "grok"
    home_dir = home_path(".grok")
    hooks_dir = home_path(".grok", "hooks")
    settings_path = home_path(".grok", "hooks", "llmoji.json")
    journal_path = home_path(".grok", "kaomoji-journal.jsonl")
    doc_path = home_path(".grok", "AGENTS.md")

    def session_files(self) -> list[Path]:
        root = self.home_dir / "sessions"
        children = _subagent_session_ids(root)
        return [
            p for p in sorted(root.glob("*/*/updates.jsonl"))
            if p.parent.name not in children
        ]

    def replay(self, path: Path) -> Iterator[ScrapeRow]:
        return iter(self._rows(path))

    def live_rows(self, payload: dict[str, Any]) -> list[ScrapeRow]:
        if payload.get("reason") != "end_turn" or payload.get("subagentType"):
            return []
        path = Path(str(payload.get("transcript_path") or payload.get("transcriptPath") or ""))
        prompt = str(payload.get("promptId") or "")
        if not path.is_file() or not prompt:
            return []
        return self._rows(path, prompt=prompt, cwd=str(payload.get("cwd") or ""))

    def _rows(self, path: Path, *, prompt: str = "", cwd: str = "") -> list[ScrapeRow]:
        session = path.parent
        try:
            info = json.loads((session / "summary.json").read_text()).get("info")
            cwd = str(info.get("cwd") or "") or cwd
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        latest_model = ""
        for row in read_jsonl(session / "chat_history.jsonl"):
            if row.get("type") == "assistant" and row.get("model_id"):
                latest_model = str(row["model_id"])
        messages, users, models = _messages(read_jsonl(path))
        rows = []
        for pid, text, ts in messages:
            if prompt and pid != prompt:
                continue
            row = kaomoji_row(
                text,
                source=f"{self.name}-hook",
                model=models.get(pid) or latest_model,
                timestamp=ts,
                cwd=cwd,
                user=users.get(pid, ""),
            )
            if row is not None:
                rows.append(row)
        return rows

    def _unregister(self) -> None:
        """Deregister, then delete ``llmoji.json`` once it's empty: the
        file is ours, and an empty one would sit in Grok's hook list."""
        super()._unregister()
        try:
            if self.settings_path.exists() and load_json_strict(self.settings_path) == {}:
                self.settings_path.unlink()
        except SettingsCorruptError:
            pass


def _subagent_session_ids(sessions_root: Path) -> set[str]:
    """Session ids of every subagent child under ``sessions_root``.

    The meta files under a parent's ``subagents/`` aren't documented, so
    every directory name and every string value in their JSON counts as
    a candidate id; only real session directory names are ever compared
    against the set. The parent's own id is left out.
    """
    ids: set[str] = set()
    for sub in sessions_root.glob("*/*/subagents"):
        found: set[str] = set()
        for entry in sub.rglob("*"):
            if entry.is_dir():
                found.add(entry.name)
                continue
            if entry.suffix != ".json":
                continue
            try:
                stack: list[Any] = [json.loads(entry.read_text(errors="replace"))]
            except (OSError, json.JSONDecodeError):
                continue
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
                elif isinstance(node, str):
                    found.add(node)
        found.discard(sub.parent.name)
        ids |= found
    return ids
