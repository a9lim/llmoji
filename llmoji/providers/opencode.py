"""opencode provider — TS plugin auto-loaded from
``~/.config/opencode/plugins/``.

opencode (https://opencode.ai) ships a TypeScript-only plugin
system. There's no shell-hook escape hatch, so this provider lives
under :class:`PluginInstaller` rather than the bash-rendering
:class:`HookInstaller` path. The rendered plugin file at
``~/.config/opencode/plugins/llmoji.ts`` registers two opencode
hooks:

  - ``experimental.chat.system.transform`` — appends the kaomoji
    nudge to every model invocation's system-prompt array (recency
    effect; opencode rebuilds the system prompt per call).
  - ``event`` — gates on completed assistant ``message.updated``,
    dedupes by message id, walks parts via the SDK, emits one
    journal row per kaomoji-led message into
    ``~/.llmoji/journals/opencode.jsonl``.

The taxonomy validator + ``leadingBracketSpan`` extractor are
spliced in at install time from
``llmoji/_plugins/_kaomoji_taxonomy.ts.partial`` — single source of
truth shared with the openclaw provider, asserted byte-identical by
``test_plugin_taxonomy_block_matches`` in ``tests/test_public_surface.py``.

Per-provider notes (vs claude_code / codex / hermes):

  - **No bash hook on disk.** ``hook_path`` points at the rendered
    ``llmoji.ts``; the bash-side machinery (``hook_template``,
    ``main_event``, ``skip_action``, ``system_injected_prefixes``)
    is unused.
  - **No settings.json edit.** opencode auto-loads any
    ``.ts`` / ``.js`` file under ``~/.config/opencode/plugins/`` —
    file presence IS registration.
  - **No separate nudge hook file.** The nudge runs inside the
    plugin via ``experimental.chat.system.transform``.
  - **Sidechain filter:** opencode doesn't expose a sidechain /
    subagent flag on the message-updated event payload as of writing,
    so subagent rows (if/when opencode adds delegation) would land
    in the journal under their own session_ids until the SDK
    surface gives us a flag to filter on.
"""

from __future__ import annotations

from pathlib import Path

from .base import PluginInstaller


class OpencodeProvider(PluginInstaller):
    name = "opencode"
    # Global plugins dir per opencode docs — auto-loaded across every
    # opencode session. Project-local ``.opencode/plugins/`` would
    # also work but isn't where ``llmoji install`` belongs (the user
    # wants the journal across every project, not just one repo).
    plugin_dir = Path.home() / ".config" / "opencode" / "plugins"
    journal_path = Path.home() / ".llmoji" / "journals" / "opencode.jsonl"
    plugin_files = [
        ("opencode.ts.tmpl", "llmoji.ts"),
    ]
