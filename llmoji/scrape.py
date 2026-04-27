"""Unified kaomoji-scrape schema and journal iterator.

Concrete sources live under ``llmoji.sources``:

  - ``llmoji.sources.journal`` — read any provider's
    ``~/.<harness>/kaomoji-journal.jsonl`` (the canonical 6-field
    unified row format the bash hook templates emit).
  - ``llmoji.sources.claude_export`` — read a Claude.ai data-export
    ``conversations.json`` and yield kaomoji-bearing assistant
    messages.

Kaomoji extraction uses :func:`llmoji.taxonomy.extract` (balanced-paren
span fallback; no dialect-specific dict required at scrape time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator


@dataclass
class ScrapeRow:
    """One kaomoji-bearing assistant message, source-agnostic.

    Sources fill the fields they have evidence for and leave the
    others empty / ``None``. The downstream :mod:`llmoji.analyze`
    pipeline reads only ``first_word``, ``assistant_text``, and
    ``surrounding_user``; the richer fields (``session_id``,
    ``parent_uuid``, etc.) are preserved for research-side consumers
    that want them.
    """

    # --- provenance ---
    source: str                 # e.g. "claude_code-hook", "codex-hook",
                                #      "hermes-hook", "claude-ai-export"
    session_id: str             # session/conversation UUID, if known
    project_slug: str           # working-dir basename or conversation name
    assistant_uuid: str         # message UUID (export only)
    parent_uuid: str | None     # parent message UUID (export only)

    # --- context ---
    model: str | None           # active model slug, if reported
    timestamp: str              # ISO-8601
    cwd: str | None             # working directory at hook fire
    git_branch: str | None      # claude-code transcript only
    turn_index: int             # 0-based per-source position
    had_thinking: bool          # transcript-only enrichment

    # --- content ---
    assistant_text: str         # full assistant message text (kaomoji stripped)
    first_word: str             # leading balanced-paren span (canonical
                                # kaomoji identifier in the unified schema)
    kaomoji: str | None         # taxonomy-registered form, or None
    kaomoji_label: int          # +1/-1/0 (mostly 0 outside the gemma corpus)

    # --- upstream ---
    surrounding_user: str       # latest preceding user-authored text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_all(*sources: Iterable[ScrapeRow]) -> Iterator[ScrapeRow]:
    """Chain multiple :class:`ScrapeRow` iterators.

    The end-user CLI calls this with whatever sources the local
    installation has (one journal per installed provider, plus any
    static dumps the user has parsed). Pure plumbing — no I/O, no
    canonicalization.
    """
    for src in sources:
        yield from src
