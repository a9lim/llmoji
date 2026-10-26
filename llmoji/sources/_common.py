"""Helpers the export readers and the Claude Code decoder share."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

_T = TypeVar("_T")


def dedup_by_id_keep_richest(
    candidates: Iterable[tuple[str, _T, int]],
) -> dict[str, _T]:
    """One payload per id from ``(id, payload, score)`` tuples: the
    highest-scoring, the first seen on a tie."""
    best: dict[str, _T] = {}
    best_score: dict[str, int] = {}
    for cid, payload, score in candidates:
        if score > best_score.get(cid, -1):
            best[cid] = payload
            best_score[cid] = score
    return best


def walk_parents_for_user_text(
    start_uuid: str | None,
    by_uuid: dict[str, dict[str, Any]],
    *,
    parent_field: str,
    role_check: Callable[[dict[str, Any]], bool],
    text_extractor: Callable[[dict[str, Any]], str],
    injected_prefixes: Sequence[str] = (),
    max_hops: int = 1000,
) -> str:
    """The nearest user text up a parent-uuid chain: the first node
    passing ``role_check`` with non-empty text that doesn't start with
    an injected prefix. ``""`` when the chain ends first. ``max_hops``
    only guards a cycle; a tool-heavy turn chains dozens of
    tool-call/tool-result pairs between a message and its prompt."""
    uuid = start_uuid
    for _ in range(max_hops):
        if not uuid:
            return ""
        node = by_uuid.get(uuid)
        if node is None:
            return ""
        if role_check(node):
            text = text_extractor(node)
            if text and text.strip():
                if not injected_prefixes or not text.startswith(
                    tuple(injected_prefixes)
                ):
                    return text
        uuid = node.get(parent_field)
    return ""
