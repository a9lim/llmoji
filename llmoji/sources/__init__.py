"""Per-source kaomoji-row iterators.

Each module exposes ``iter_<name>(path) -> Iterator[ScrapeRow]``,
parsing one concrete file format:

  - :mod:`llmoji.sources.journal` — the canonical 6-field unified
    JSONL the bash hook templates emit. Parametrized by source
    label (``"claude_code"``, ``"codex"``, ``"hermes"``, …) so a
    single function handles every provider's journal.
  - :mod:`llmoji.sources.claude_export` — Claude.ai data-export
    ``conversations.json``. Static archive format, not a hook
    output.
  - :mod:`llmoji.sources.chatgpt_export` — OpenAI ChatGPT data-export
    ``conversations.json``. Same filename as Claude's, different
    schema (a tree of message nodes keyed by id, with
    ``current_node`` pointing at the active leaf).

The two static-export readers share the validate-and-strip dance
through :func:`llmoji.sources._common.kaomoji_lead_strip` so the
journal-row contract (``assistant_text`` never carries the kaomoji)
holds uniformly across sources.

Source modules are deliberately stateless: a path in, a stream of
:class:`~llmoji.scrape.ScrapeRow` out. The CLI orchestrates which
sources to read.
"""
