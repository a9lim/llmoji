"""Claude.ai per-conversation export source adapter.

A second Claude.ai export shape: one ``<title>.json`` per
conversation in a single directory, plus an ``export_summary.json``
sibling. Each conversation file is structurally a single element of
the legacy ``conversations.json`` list — same ``chat_messages``
schema with ``parent_message_uuid`` + ``content[]`` blocks — with
two practical differences:

  * a top-level ``model`` field carries the conversation's model
    (e.g. ``claude-opus-4-7``), which the legacy combined export
    omits, so rows from this reader populate ``ScrapeRow.model``
  * the per-file shape means a partial re-export naturally lands as
    just-the-changed-files; we still dedup by ``uuid`` across input
    dirs via :func:`dedup_by_id_keep_richest` to match the legacy
    reader's repeated-export semantics.

The per-message walk is shared with
:mod:`llmoji.sources.claude_export` via
``_iter_chat_messages_conversation`` — single source of truth for
the kaomoji filter, parent-uuid walk for ``surrounding_user``, and
the content-block-text fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow
from ._common import dedup_by_id_keep_richest
from .claude_export import (
    _conv_content_score,
    _iter_chat_messages_conversation,
)


# Filenames in the export directory that aren't conversation payloads
# and must be skipped during file discovery.
_NON_CONVERSATION_FILENAMES: frozenset[str] = frozenset({
    "export_summary.json",
})


def iter_claude_export_alt(
    export_dirs: Iterable[Path | str],
) -> Iterator[ScrapeRow]:
    """Yield kaomoji-bearing assistant messages from one or more
    Claude.ai per-conversation export directories.

    Each directory holds one ``.json`` file per conversation (plus
    an ``export_summary.json`` metadata sibling, which is skipped).
    Conversations are unioned by ``uuid`` across directories; on
    duplicate ``uuid``s the version with more non-empty messages
    wins (via :func:`dedup_by_id_keep_richest`), matching the
    legacy combined-export reader's repeated-export semantics.
    """
    candidates: list[tuple[str, dict[str, Any], int]] = []
    for export_dir in export_dirs:
        path = Path(export_dir)
        if not path.is_dir():
            continue
        for entry in sorted(path.glob("*.json")):
            if entry.name in _NON_CONVERSATION_FILENAMES:
                continue
            try:
                with entry.open() as f:
                    conv = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(conv, dict):
                continue
            uuid = conv.get("uuid")
            if not isinstance(uuid, str):
                continue
            candidates.append((uuid, conv, _conv_content_score(conv)))

    for conv in dedup_by_id_keep_richest(candidates).values():
        model = conv.get("model")
        yield from _iter_chat_messages_conversation(
            conv,
            source="claude-ai-alt-export",
            model=model if isinstance(model, str) and model else None,
        )
