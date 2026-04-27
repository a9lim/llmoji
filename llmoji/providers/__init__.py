"""Provider abstraction.

A *provider* is one coding harness whose stop-event hook contract is
"run a script with the event payload on stdin and a stdout JSON
response gates the loop." Each first-class provider in v1.0
(``claude_code``, ``codex``, ``hermes``) ships a bash hook template
under :mod:`llmoji._hooks` plus a concrete subclass of
:class:`Provider` that knows where the harness keeps its hooks
directory and settings file.

Three things drive the abstraction:

  1. Where to write the hook script (``hooks_dir``).
  2. How to register it (``settings_path`` + ``settings_format``,
     plus per-provider register/unregister logic in
     :meth:`Provider.install`/``uninstall``).
  3. Where the journal lives (``journal_path``) — a published
     uniform-schema JSONL the live hook appends to and which
     ``llmoji analyze`` reads.

Two cross-cutting parameters that the hook template needs at
generation time:

  - ``kaomoji_position`` — does the harness emit one assistant
    message per turn, multiple where the kaomoji is on the first,
    or multiple where it's on the last? The shell template's
    extraction pipeline is keyed on this.
  - ``sidechain_strategy`` — how to filter subagent dispatches.
    Three modes:

    * ``"none"`` — harness has no subagent concept (Codex).
    * ``"field_flag"`` — drop events with a boolean flag set
      (Claude Code's ``isSidechain``).
    * ``"session_correlation"`` — track child session IDs from a
      companion event and drop their stop events (Hermes — the
      ``post_llm_call`` hook fires for both parent and child
      sessions; correlate against ``delegate_task``).

Generic-JSONL-append users (motivated OpenClaw owners and similar)
bypass this abstraction entirely: they handcraft a TS handler that
writes the canonical 6-field schema to
``~/.llmoji/journals/<name>.jsonl`` and ``llmoji analyze`` picks it
up via the same :func:`llmoji.sources.journal.iter_journal`
iterator. No first-class provider required for them in v1.0.
"""

from __future__ import annotations

from .base import (
    Provider,
    ProviderStatus,
    SidechainStrategy,
    KaomojiPosition,
)
from .claude_code import ClaudeCodeProvider
from .codex import CodexProvider
from .hermes import HermesProvider

# Registry order is the user-facing default order for ``llmoji status``
# and similar listings. Claude Code first because it's the most
# common, hermes last because it's the newest and least battle-tested.
PROVIDERS: dict[str, type[Provider]] = {
    "claude_code": ClaudeCodeProvider,
    "codex": CodexProvider,
    "hermes": HermesProvider,
}


def get_provider(name: str) -> Provider:
    """Look up a provider by name. Raises :class:`KeyError` on typos."""
    if name not in PROVIDERS:
        raise KeyError(
            f"unknown provider {name!r}; "
            f"known: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name]()


__all__ = [
    "Provider",
    "ProviderStatus",
    "SidechainStrategy",
    "KaomojiPosition",
    "ClaudeCodeProvider",
    "CodexProvider",
    "HermesProvider",
    "PROVIDERS",
    "get_provider",
]
