"""Shared helpers for the per-source ``ScrapeRow`` iterators.

Every export reader does the same dance against
:func:`llmoji.taxonomy.extract`:

  1. Pull the leading kaomoji span out of the raw assistant text.
  2. Reject the message if there is no valid leading span.
  3. Strip the prefix + surrounding whitespace from the body so the
     v1.0 journal-row contract holds (``assistant_text`` never carries
     the kaomoji — it lives in the row's ``kaomoji`` field, see the
     "Journal-row contract" gotcha in :file:`CLAUDE.md`).

Lifted into one helper so adding a new export reader doesn't reach
for the strip-and-validate code path by hand and silently produce
kaomoji-bearing ``assistant_text`` rows that would force
:func:`llmoji.haiku.mask_kaomoji` to grow a second branch.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, TypeVar

from ..taxonomy import extract

_T = TypeVar("_T")


def kaomoji_lead_strip(text: str) -> tuple[str, str] | None:
    """Validate + split the leading kaomoji from a raw assistant text.

    Returns ``(first_word, body)`` on success, where ``first_word`` is
    the validated leading kaomoji span and ``body`` is the message
    text with that prefix and surrounding whitespace stripped.
    Returns ``None`` when the text doesn't open with a kaomoji that
    passes :func:`~llmoji.taxonomy.extract` (so callers can ``continue``
    cleanly in their per-message loop).

    The leading-glyph filter is enforced inside ``extract`` (via
    :func:`~llmoji.taxonomy.is_kaomoji_candidate`) — a non-empty
    ``first_word`` already guarantees ``first_word[0]`` is in
    :data:`~llmoji.taxonomy.KAOMOJI_START_CHARS`. Single source of
    truth lives in ``taxonomy``; don't re-check here.
    """
    match = extract(text)
    if not match.first_word:
        return None
    body = text.lstrip()
    if body.startswith(match.first_word):
        body = body[len(match.first_word):].lstrip()
    return match.first_word, body


def dedup_by_id_keep_richest(
    candidates: Iterable[tuple[str, _T, int]],
) -> dict[str, _T]:
    """Deduplicate ``(id, payload, score)`` tuples by id; keep the
    payload with the highest score per id.

    Used by both export readers (claude.ai + chatgpt). Each format
    can carry the same conversation across multiple export bundles
    with one version more complete than the others (the user
    re-exports after editing one branch); routing through one
    helper means the "keep the richest" tiebreaker rule can't drift
    between readers. Strict ``>`` so the first-seen payload wins
    ties.
    """
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
    """Walk a parent-uuid chain backward to the nearest human-typed
    user text. Used by both the Claude.ai export reader and the
    Claude Code transcript backfill — same algorithm with different
    field names and role/text shapes.

    Parameters:

    - ``start_uuid`` — uuid of the immediate parent of the
      kaomoji-led assistant message.
    - ``by_uuid`` — dict of uuid → message/event node, indexed once
      per file by the caller.
    - ``parent_field`` — the key on a node holding its parent uuid
      (``"parentUuid"`` for Claude Code transcripts,
      ``"parent_message_uuid"`` for Claude.ai exports).
    - ``role_check(node)`` — returns ``True`` iff the node is a
      human-authored user message (skips tool_result and other
      non-user roles cleanly).
    - ``text_extractor(node)`` — returns the message text, or empty
      string if not extractable.
    - ``injected_prefixes`` — system-injected user-role payload
      prefixes (skill activations, etc.); a candidate text starting
      with any of these is dropped and the walk continues. Empty
      tuple skips the check.
    - ``max_hops`` — bound on a pathological uuid cycle. Generous
      because a tool-heavy turn easily chains dozens of
      assistant→tool_result pairs between the kaomoji-led text and
      the originating user prompt.

    Returns the first qualifying user text, or ``""`` when the
    chain reaches a missing or null parent first.
    """
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
