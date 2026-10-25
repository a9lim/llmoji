"""Grok Build provider (xAI's ``grok`` CLI).

Grok discovers hook files at ``~/.grok/hooks/*.json``, each holding
the same ``{"hooks": {"<Event>": [{"hooks": [{"type": "command",
"command": "..."}]}]}}`` shape Claude Code's ``settings.json`` uses.
llmoji owns one such file outright — ``~/.grok/hooks/llmoji.json`` —
so the JSON-settings helpers in :class:`HookInstaller` register into
it and uninstall deletes it once our entry is gone. Contract source:
the user guide Grok installs at ``~/.grok/docs/user-guide/10-hooks.md``
(grok 1.0.70), cross-checked against live payloads.

Per-provider quirks (vs claude_code / codex):

  - **Soft only.** Grok's ``UserPromptSubmit`` hook runs, but the
    harness discards an allowing hook's stdout — its docs list "no
    ``additionalContext``" as a current limit — so there is nowhere
    to put a per-turn nudge. ``install --hard`` refuses;
    ``install --soft`` appends the block to ``~/.grok/AGENTS.md``,
    the global rules file Grok reads before any project one.
  - **Two Stop fires per session end.** Besides the turn-end Stop
    (``reason: "end_turn"``), Grok fires an observe-only Stop at
    session teardown (``reason: "shutdown"`` / ``"channel_closed"``).
    The hook journals only ``end_turn``; walking the teardown fire
    too would land the last turn's messages twice.
  - **Turn scoping by ``promptId``.** ``transcript_path`` names the
    session's ``updates.jsonl`` (the ACP update stream). Every
    ``agent_message_chunk`` there carries ``_meta.promptId``, the
    same id the Stop payload carries, so the hook reads exactly this
    turn's messages — an auto-woken turn (a background task
    finishing) can't re-journal the turn before it. Consecutive
    chunks are joined into one message.
  - **Model from ``chat_history.jsonl``.** The Stop payload carries
    no model, and ``updates.jsonl`` only records one in
    ``turn_completed``, which is written after the Stop hook runs.
    The sibling ``chat_history.jsonl`` stamps ``model_id`` on every
    assistant row; the hook takes the latest.
  - **No sidechain filter needed.** Inside a subagent the gate fires
    as ``SubagentStop``; a ``Stop`` hook only gates the main agent.
    The hook still drops a payload carrying ``subagentType`` as
    defense-in-depth.
  - **Claude compatibility runs Claude's hooks too.** With
    ``[compat.claude] hooks = true`` (Grok's default) Grok also runs
    the hooks in ``~/.claude/settings.json``, so a claude_code install
    fires inside Grok sessions. It journals nothing there — Grok's
    ``updates.jsonl`` has none of the entry shapes the claude_code
    hook selects — and Grok drops the nudge hook's stdout. Likewise
    ``[compat.claude] agents`` reads ``~/.claude/CLAUDE.md``, so a
    claude_code ``--soft`` block also reaches Grok.
  - **Folder trust.** A global hook makes Grok ask for folder trust
    before running it in each new folder (headless ``grok -p`` counts
    as trusted for your own hooks).
"""

from __future__ import annotations

from pathlib import Path

from .base import HookInstaller, SettingsCorruptError, _load_json_strict, home_path


class GrokProvider(HookInstaller):
    name = "grok"
    hooks_dir = home_path(".grok", "hooks")
    settings_path = home_path(".grok", "hooks", "llmoji.json")
    journal_path = home_path(".grok", "kaomoji-journal.jsonl")
    # Grok's global rules file: the first place its AGENTS.md scan
    # looks, ahead of the repo-root-to-cwd walk.
    system_prompt_doc_path = home_path(".grok", "AGENTS.md")
    hook_template = "grok.sh.tmpl"
    # Per-message while-loop, like claude_code / codex.
    skip_action = "continue"
    # User text comes from ``user_message_chunk`` updates, which carry
    # what the user typed — Grok's injected ``<user_info>`` / system
    # reminders live only in ``chat_history.jsonl``.
    system_injected_prefixes: list[str] = []
    hard_unsupported_reason = (
        "Grok discards a UserPromptSubmit hook's additionalContext, so "
        "a per-turn nudge never reaches the model"
    )

    @property
    def home_dir(self) -> Path:
        return self.settings_path.parent.parent

    def _unregister(self) -> None:
        """Deregister, then delete ``llmoji.json`` once nothing is left
        in it — the file is ours, and an empty hook file would sit in
        Grok's hook list for no reason."""
        super()._unregister()
        if not self.settings_path.exists():
            return
        try:
            remaining = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return  # leave a file we can't read alone
        if remaining == {}:
            self.settings_path.unlink()
