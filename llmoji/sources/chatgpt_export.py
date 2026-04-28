"""ChatGPT data-export source adapter: ``conversations.json``.

The OpenAI export (Settings → Data Controls → Export Data) lands as
a ZIP whose canonical contents include a single ``conversations.json``
holding every conversation in the account. The file is a top-level
JSON list; each conversation is a tree of messages plus a pointer at
the active leaf:

  - ``mapping`` — dict keyed by node id. Each node is
    ``{id, message, parent, children}``. The tree branches when the
    user clicks "regenerate" or edits an earlier message; only the
    chain reachable from ``current_node`` was on screen.
  - ``current_node`` — id of the active leaf; walk up via ``parent``
    to recover the displayed conversation in reverse order.
  - top-level ``id`` / ``conversation_id`` (UUID) and ``title``.

A message node carries:

  - ``id``, ``author: {role: "user"|"assistant"|"system"|"tool", ...}``,
    ``create_time`` (Unix timestamp; may be ``null`` for the system
    seed),
  - ``content: {content_type, parts: [...]}`` — for ``"text"``,
    ``parts`` is a list of strings; for ``"multimodal_text"``,
    ``parts`` can mix strings (text segments) with dicts (images,
    audio, code-interpreter results), and the dicts may carry a
    ``"text"`` field we want.
  - ``metadata: {model_slug?, ...}`` — the model slug lives here,
    not on the conversation envelope.

Like Anthropic's Claude.ai export endpoint, OpenAI's export pipeline
is occasionally non-idempotent for old conversations (some come back
empty in a fresh export that an earlier export populated). The
union-by-conversation-id + keep-fuller-copy strategy from the
Claude.ai reader applies symmetrically here.

This reader yields one :class:`~llmoji.scrape.ScrapeRow` per
kaomoji-bearing assistant message in the active branch of every
conversation. The validate-and-strip dance is shared with
:mod:`llmoji.sources.claude_export` via
:func:`llmoji.sources._common.kaomoji_lead_strip`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow
from ._common import dedup_by_id_keep_richest, kaomoji_lead_strip


def _message_text(msg: dict[str, Any]) -> str:
    """Extract joined text content from one ChatGPT message node.

    Handles both shapes the export uses:
      * ``content_type == "text"`` — ``parts`` is a list of strings.
      * ``content_type == "multimodal_text"`` — ``parts`` can mix
        strings and dicts (image refs, audio refs, code-interpreter
        outputs); we keep dict parts that carry a ``"text"`` key,
        skip the rest.

    Other content types (``"code"``, ``"execution_output"``,
    ``"tether_browsing_display"``, ``"system_error"`` …) are skipped —
    they aren't kaomoji-bearing assistant prose by construction.
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
    """Return the active branch in root→leaf order.

    Walks from ``current_node`` up via ``parent`` and reverses. Cycle
    detection caps the walk at ``len(mapping) + 16`` slack — a
    malformed export with a self-referencing parent pointer would
    otherwise loop forever.
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
    """Normalize ChatGPT's float Unix timestamp to ISO-8601 UTC.

    Returns ``""`` when the field is missing / not numeric (true for
    the system seed message and occasionally for tool nodes).
    """
    if not isinstance(ct, (int, float)):
        return ""
    return (
        datetime.fromtimestamp(float(ct), tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _iter_conversation_chain(
    conv: dict[str, Any],
    chain: list[dict[str, Any]],
) -> Iterator[ScrapeRow]:
    """Yield :class:`ScrapeRow` per kaomoji-led assistant message in
    a pre-walked active branch.

    Caller passes the active-branch chain directly so the
    ``_walk_active_branch`` cost isn't paid twice (once for scoring,
    once for iteration).

    Tracks ``last_user_text`` in a single forward pass — replaces the
    previous ``reversed(chain[:idx])`` per-assistant scan, which was
    O(n²) over conversation length on long sessions.
    """
    if not chain:
        return
    session_id = str(conv.get("id") or conv.get("conversation_id") or "")
    project_slug = str(conv.get("title") or "") or "(unnamed)"
    # turn_index = 0-based position among assistant messages with
    # non-empty text in the active branch — independent of the
    # kaomoji filter, mirroring claude_export's semantics so a row
    # of two consecutive non-kaomoji turns doesn't compress.
    turn = -1
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
        turn += 1
        stripped = kaomoji_lead_strip(text)
        if stripped is None:
            continue
        first_word, body = stripped
        model = (msg.get("metadata") or {}).get("model_slug")
        yield ScrapeRow(
            source="chatgpt-export",
            session_id=session_id,
            project_slug=project_slug,
            assistant_uuid=str(msg.get("id") or node.get("id") or ""),
            parent_uuid=node.get("parent"),
            model=str(model) if model else None,
            timestamp=_format_timestamp(msg.get("create_time")),
            cwd=None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,
            assistant_text=body,
            first_word=first_word,
            surrounding_user=last_user_text,
        )


def _conv_chain_and_score(
    conv: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(chain, score)`` for one conversation in a single
    branch walk.

    ``score`` counts assistant nodes in the active branch with
    non-empty text — used by :func:`iter_chatgpt_export` to pick the
    richest copy when a conversation id appears in more than one
    export. Bundling the two return values lets the caller hand the
    chain straight to :func:`_iter_conversation_chain` without
    re-walking.
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
    """Yield kaomoji-bearing assistant messages from one or more
    ChatGPT export directories.

    Each directory is expected to contain a ``conversations.json``
    file (the canonical filename in OpenAI's export ZIP). When a
    conversation id appears in more than one export, the version with
    more non-empty assistant messages in its active branch wins —
    same heuristic as the Claude.ai reader. The branch walk happens
    once per conversation: scoring + iteration share the same chain.
    """
    candidates: list[
        tuple[str, tuple[dict[str, Any], list[dict[str, Any]]], int]
    ] = []
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
            candidates.append((cid, (conv, chain), score))

    for conv, chain in dedup_by_id_keep_richest(candidates).values():
        yield from _iter_conversation_chain(conv, chain)
