"""Codex provider.

Hook fires under Codex's per-session hook config. Codex delivers the
final assistant text via the ``last_assistant_message`` field on the
Stop-event payload — no transcript-walking required for the assistant
side. The ``transcript_path`` field on the payload still points at
the per-session rollout JSONL, which we use to resolve the latest
real user message.

Settings file: ``~/.codex/hooks.json``. Codex's hook system (the
``codex_hooks`` feature flag, ``Stage::Stable`` and
``default_enabled: true`` in ``codex-rs/features``) accepts a
Claude-style payload — ``{"hooks": {"<Event>": [{"hooks":
[{"type": "command", "command": "..."}]}]}}`` — at this path. (Codex
also tolerates ``[[hooks.<Event>]]`` array-of-tables in
``config.toml``, but that path warns when both representations are
present, so we standardize on the JSON file.)

The ``UserPromptSubmit`` response envelope is byte-identical to
Claude Code's (verified at ``codex-rs/hooks/src/events/
user_prompt_submit.rs``), so a single shared
``claude_codex_nudge.sh.tmpl`` template serves both providers.

Per-provider quirks (vs claude_code / hermes):

  - **One row per kaomoji-led agent_message in the current turn.**
    Codex emits each model message as its own
    ``event_msg.agent_message`` event with the text on
    ``payload.message``. A tool-heavy turn writes 5–10 of these
    (``phase: "commentary"`` for progress messages, ``"final_answer"``
    for the closing one); the journal captures every kaomoji-led
    message, not just the final summary. ``last_agent_message`` on
    the Stop payload is now ignored.
  - **No subagent concept** — sidechain filtering is unnecessary.
    ``collaboration_mode`` is ``"default"`` for every observed
    turn_context.
  - **System-injected user-role prefixes:** AGENTS.md /
    ``<environment_context>`` / ``<INSTRUCTIONS>`` are injected as
    user-role response_items at session start. Drop them defensively.
"""

from __future__ import annotations

from pathlib import Path

from .base import JsonSettingsHookInstaller


class CodexProvider(JsonSettingsHookInstaller):
    name = "codex"
    hooks_dir = Path.home() / ".codex" / "hooks"
    settings_path = Path.home() / ".codex" / "hooks.json"
    journal_path = Path.home() / ".codex" / "kaomoji-journal.jsonl"
    hook_template = "codex.sh.tmpl"
    # Same per-entry-loop reasoning as ClaudeCodeProvider: the validate
    # partial sits inside a ``while read``-loop iterating every
    # event_msg.agent_message in the current turn, so ``continue`` is
    # the right skip action.
    skip_action = "continue"
    system_injected_prefixes = [
        "# AGENTS.md",
        "<environment_context>",
        "<INSTRUCTIONS>",
    ]

    # Nudge attrs (template / filename / event / message) inherited
    # from JsonSettingsHookInstaller — same wording + envelope shared with
    # Claude Code; verified byte-identical UserPromptSubmit envelope
    # on the Codex side at codex-rs/hooks/src/events/user_prompt_submit.rs.
