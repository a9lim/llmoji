"""ChatGPT data exports: ``conversations.json``.

The export (Settings -> Data Controls -> Export Data) holds one top-level
list of conversations. Each has a ``mapping`` of node id to
``{id, message, parent, children}`` and a ``current_node`` naming the active
leaf. The tree branches when the user regenerates or edits a message and
only the chain reachable from ``current_node`` was on screen, so the reader
walks up from it via ``parent``.

A message has ``author.role``, a Unix ``create_time`` (null on the system
seed), ``content`` and ``metadata``. ``text`` content has string ``parts``;
``multimodal_text`` mixes strings with dicts (images, audio,
code-interpreter results), and a dict may carry a ``text`` field. The model
is ``metadata.model_slug``, not on the conversation.

Conversations are unioned by id across directories, the copy with more
non-empty assistant messages winning: the export pipeline sometimes returns
empty content for a conversation an earlier export had in full.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow, kaomoji_row
from ._common import dedup_by_id_keep_richest


def _message_text(msg: dict[str, Any]) -> str:
    """The message's text parts joined by newlines.

    ``text`` content has string ``parts``; ``multimodal_text`` mixes strings
    with dicts (image and audio refs, code-interpreter output), of which
    only those with a ``text`` key count. Content types without ``parts``
    yield nothing.
    """
    content = msg.get("content") or {}
    parts = content.get("parts") or []
    out: list[str] = []
    for p in parts:
        if isinstance(p, str):
            if p.strip():
                out.append(p)
        elif isinstance(p, dict):
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t)
    return "\n".join(out)


def _walk_active_branch(
    mapping: dict[str, dict[str, Any]],
    current_node: str | None,
) -> list[dict[str, Any]]:
    """The active branch in root-to-leaf order, walked up from
    ``current_node`` via ``parent``. A seen-set and a cap of
    ``len(mapping) + 16`` steps stop a malformed export with a parent cycle
    from looping forever.
    """
    if not current_node:
        return []
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: str | None = current_node
    cap = len(mapping) + 16
    while cur and cur not in seen and len(chain) < cap:
        seen.add(cur)
        node = mapping.get(cur)
        if node is None:
            break
        chain.append(node)
        cur = node.get("parent")
    chain.reverse()
    return chain


def _node_role(node: dict[str, Any]) -> str | None:
    msg = node.get("message")
    if not isinstance(msg, dict):
        return None
    return (msg.get("author") or {}).get("role")


def _format_timestamp(ct: Any) -> str:
    """ISO-8601 UTC for a Unix timestamp, or ``""`` when it isn't a number
    (the system seed, some tool nodes)."""
    if not isinstance(ct, (int, float)):
        return ""
    return (
        datetime.fromtimestamp(float(ct), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _iter_conversation_chain(
    chain: list[dict[str, Any]],
) -> Iterator[ScrapeRow]:
    """One :class:`ScrapeRow` per kaomoji-led assistant message in an
    already-walked active branch, with the latest user text before it as
    context. It takes the chain rather than the conversation so the walk
    isn't repeated after scoring.
    """
    if not chain:
        return
    last_user_text = ""
    for node in chain:
        role = _node_role(node)
        if role == "user":
            pt = _message_text(node["message"]) if isinstance(node.get("message"), dict) else ""
            if pt.strip():
                last_user_text = pt
            continue
        if role != "assistant":
            continue
        msg = node["message"]
        text = _message_text(msg)
        if not text.strip():
            continue
        model = (msg.get("metadata") or {}).get("model_slug")
        row = kaomoji_row(
            text,
            source="chatgpt_export",
            model=str(model or ""),
            timestamp=_format_timestamp(msg.get("create_time")),
            cwd=None,
            user=last_user_text,
        )
        if row is not None:
            yield row


def _conv_chain_and_score(
    conv: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """``(chain, score)`` for one conversation from a single branch walk.

    ``score`` counts the assistant nodes with non-empty text;
    :func:`iter_chatgpt_export` uses it to pick the richest copy of a
    conversation id.
    """
    mapping = conv.get("mapping")
    if not isinstance(mapping, dict):
        return [], 0
    chain = _walk_active_branch(mapping, conv.get("current_node"))
    score = 0
    for node in chain:
        if _node_role(node) != "assistant":
            continue
        msg = node.get("message")
        if isinstance(msg, dict) and _message_text(msg).strip():
            score += 1
    return chain, score


def iter_chatgpt_export(
    export_dirs: Iterable[Path | str],
) -> Iterator[ScrapeRow]:
    """Kaomoji-led assistant messages from ChatGPT export directories.

    Each directory holds a ``conversations.json``. When a conversation id
    appears in more than one export, the copy with the most non-empty
    assistant messages in its active branch wins, as in the Claude.ai
    reader.
    """
    candidates: list[tuple[str, list[dict[str, Any]], int]] = []
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
            cid = conv.get("id") or conv.get("conversation_id")
            if not isinstance(cid, str):
                continue
            chain, score = _conv_chain_and_score(conv)
            candidates.append((cid, chain, score))

    for chain in dedup_by_id_keep_richest(candidates).values():
        yield from _iter_conversation_chain(chain)
