"""Claude.ai data exports.

Two shapes, told apart per directory: a single ``conversations.json``
holding every conversation, or one ``<title>.json`` per conversation
beside an ``export_summary.json``. Both use the same message schema;
only the per-file shape carries a conversation ``model``.

Conversations are unioned by uuid across directories, the copy with
more non-empty messages winning: the export endpoint sometimes returns
empty content for a conversation an earlier export had in full.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow, kaomoji_row
from ._common import dedup_by_id_keep_richest, walk_parents_for_user_text


def _message_text(msg: dict[str, Any]) -> str:
    """Top-level ``text``, else the ``content[]`` text blocks: newer
    exports sometimes null out one or the other."""
    t = msg.get("text")
    if isinstance(t, str) and t.strip():
        return t
    return "\n".join(
        str(b.get("text"))
        for b in msg.get("content") or []
        if isinstance(b, dict) and b.get("type") == "text" and str(b.get("text") or "").strip()
    )


def _conversations(export_dir: Path) -> Iterator[dict[str, Any]]:
    combined = export_dir / "conversations.json"
    if combined.exists():
        try:
            data = json.loads(combined.read_text())
        except (OSError, json.JSONDecodeError):
            return
        yield from (c for c in data if isinstance(c, dict)) if isinstance(data, list) else ()
        return
    for entry in sorted(export_dir.glob("*.json")):
        if entry.name == "export_summary.json":
            continue
        try:
            conv = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(conv, dict):
            yield conv


def _rows(conv: dict[str, Any]) -> Iterator[ScrapeRow]:
    msgs = [m for m in conv.get("chat_messages") or [] if isinstance(m, dict)]
    by_uuid = {m["uuid"]: m for m in msgs if isinstance(m.get("uuid"), str)}
    model = conv.get("model")
    for m in msgs:
        if m.get("sender") != "assistant":
            continue
        row = kaomoji_row(
            _message_text(m),
            source="claude_ai_export",
            model=model if isinstance(model, str) else None,
            timestamp=str(m.get("created_at") or ""),
            cwd=None,
            user=walk_parents_for_user_text(
                m.get("parent_message_uuid"),
                by_uuid,
                parent_field="parent_message_uuid",
                role_check=lambda node: node.get("sender") == "human",
                text_extractor=_message_text,
            ),
        )
        if row is not None:
            yield row


def iter_claude_export(export_dirs: Iterable[Path | str]) -> Iterator[ScrapeRow]:
    """Kaomoji-led assistant messages from one or more export directories."""
    candidates = [
        (conv["uuid"], conv, sum(1 for m in conv.get("chat_messages") or []
                                 if isinstance(m, dict) and _message_text(m).strip()))
        for d in export_dirs
        for conv in _conversations(Path(d))
        if isinstance(conv.get("uuid"), str)
    ]
    for conv in dedup_by_id_keep_richest(candidates).values():
        yield from _rows(conv)
