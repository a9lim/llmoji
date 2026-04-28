"""Claude.ai export source adapter: ``conversations.json``.

The export is a single JSON file: a list of conversation objects,
each with a ``chat_messages`` array. The top-level ``.text`` field on
each message is the canonical content; ``content[].text`` blocks are
the fallback (newer exports occasionally null out one or the other).

Anthropic's export endpoint is non-idempotent — repeated exports of
the same account sometimes return empty content for conversations a
prior export populated fully. The CLI's ``parse --provider claude.ai``
takes one or more directories and unions them by conversation UUID,
preferring whichever copy has more non-empty messages. Public model
info is not included in the export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow
from ._common import (
    dedup_by_id_keep_richest,
    kaomoji_lead_strip,
    walk_parents_for_user_text,
)


def _message_text(msg: dict[str, Any]) -> str:
    """Prefer top-level ``.text``; fall back to ``content[].text`` blocks."""
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    parts: list[str] = []
    for block in msg.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            bt = block.get("text") or ""
            if bt.strip():
                parts.append(bt)
    return "\n".join(parts)


def _iter_conversation(conv: dict[str, Any]) -> Iterator[ScrapeRow]:
    msgs = conv.get("chat_messages") or []
    if not msgs:
        return
    by_uuid: dict[str, dict[str, Any]] = {
        m["uuid"]: m for m in msgs if isinstance(m.get("uuid"), str)
    }
    session_id = str(conv.get("uuid") or "")
    project_slug = str(conv.get("name") or "") or "(unnamed)"
    # turn_index = 0-based position among assistant messages we
    # actually consider (have non-empty text). Independent of whether
    # the row passes the kaomoji filter — that way two consecutive
    # filtered-out kaomoji-less assistant turns don't compress to the
    # same index.
    turn = -1
    for m in msgs:
        if m.get("sender") != "assistant":
            continue
        text = _message_text(m)
        if not text.strip():
            continue
        turn += 1
        # Validate + strip via the shared helper so this reader and
        # the ChatGPT one stay in lockstep — see the "Journal-row
        # contract" gotcha in :file:`CLAUDE.md`.
        stripped = kaomoji_lead_strip(text)
        if stripped is None:
            continue
        first_word, body = stripped
        # Walk parent_message_uuid back to the nearest human-authored
        # message with non-empty text. Routed through the shared
        # walker in `sources._common` so this reader and the Claude
        # Code transcript backfill apply identical traversal logic;
        # only the field name and role check differ.
        parent_uuid = m.get("parent_message_uuid")
        user_text = walk_parents_for_user_text(
            parent_uuid,
            by_uuid,
            parent_field="parent_message_uuid",
            role_check=lambda node: node.get("sender") == "human",
            text_extractor=_message_text,
        )
        yield ScrapeRow(
            source="claude-ai-export",
            session_id=session_id,
            project_slug=project_slug,
            assistant_uuid=str(m.get("uuid") or ""),
            parent_uuid=parent_uuid,
            model=None,
            timestamp=str(m.get("created_at") or ""),
            cwd=None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,
            assistant_text=body,
            first_word=first_word,
            surrounding_user=user_text,
        )


def _conv_content_score(conv: dict[str, Any]) -> int:
    """Count messages with non-empty ``.text`` or ``.content`` blocks.

    Used to rank duplicate conversations across multiple exports —
    newer Claude.ai exports sometimes drop content for conversations
    earlier exports returned in full. Prefer the version with more
    filled-in messages. Caller (:func:`iter_claude_export`) narrows
    ``conv`` to ``dict`` before invoking.
    """
    score = 0
    for m in conv.get("chat_messages") or []:
        if not isinstance(m, dict):
            continue
        t = m.get("text")
        if isinstance(t, str) and t.strip():
            score += 1
            continue
        for b in m.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                bt = b.get("text") or ""
                if bt.strip():
                    score += 1
                    break
    return score


def iter_claude_export(
    export_dirs: Iterable[Path | str],
) -> Iterator[ScrapeRow]:
    """Yield kaomoji-bearing assistant messages from one or more
    Claude.ai export directories.

    Each directory is expected to contain a ``conversations.json``
    file. Conversations are unioned by UUID across directories; on
    duplicate UUIDs the version with more non-empty messages wins
    (via :func:`llmoji.sources._common.dedup_by_id_keep_richest`).
    """
    candidates: list[tuple[str, dict[str, Any], int]] = []
    for export_dir in export_dirs:
        path = Path(export_dir) / "conversations.json"
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for conv in data:
            if not isinstance(conv, dict):
                continue
            uuid = conv.get("uuid")
            if not isinstance(uuid, str):
                continue
            candidates.append((uuid, conv, _conv_content_score(conv)))

    for conv in dedup_by_id_keep_richest(candidates).values():
        yield from _iter_conversation(conv)
