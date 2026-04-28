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

Key per-provider quirks (vs claude_code / hermes):

  - **kaomoji_position = "last"**: Codex emits each agent message as
    its own ``event_msg.agent_message`` event. Progress messages
    (during tool calls) come first, the kaomoji-bearing summary
    lands last as ``task_complete.last_agent_message``. The hook
    keys on that field, NOT on the first agent_message.
  - **sidechain_strategy = "none"**: Codex has no subagent concept.
    ``collaboration_mode`` is ``"default"`` for every observed
    turn_context.
  - **system_injected_prefixes**: AGENTS.md / ``<environment_context>``
    / ``<INSTRUCTIONS>`` are injected as user-role response_items at
    session start. Drop them defensively (the live hook does the
    same).
"""

from __future__ import annotations

from pathlib import Path

from .base import KaomojiPosition, Provider, SidechainStrategy


class CodexProvider(Provider):
    name = "codex"
    hooks_dir = Path.home() / ".codex" / "hooks"
    settings_path = Path.home() / ".codex" / "hooks.json"
    settings_format = "json"
    journal_path = Path.home() / ".codex" / "kaomoji-journal.jsonl"
    hook_template = "codex.sh.tmpl"
    hook_filename = "kaomoji-log.sh"
    kaomoji_position: KaomojiPosition = "last"
    sidechain_strategy: SidechainStrategy = "none"
    sidechain_config = {}
    system_injected_prefixes = [
        "# AGENTS.md",
        "<environment_context>",
        "<INSTRUCTIONS>",
    ]

    # Nudge: same shape + wording as the Claude Code side; verified
    # byte-identical UserPromptSubmit envelope on the Codex side.
    nudge_hook_template = "claude_codex_nudge.sh.tmpl"
    nudge_hook_filename = "kaomoji-nudge.sh"
    nudge_event = "UserPromptSubmit"
    nudge_message = (
        "Please begin your message with a kaomoji that best represents "
        "how you feel."
    )

    def _register(self) -> None:
        self._register_json_settings(event="Stop")
        self._register_json_settings(
            event=self.nudge_event,
            hook_path=self.nudge_hook_path,
        )

    def _unregister(self) -> None:
        self._unregister_json_settings(event="Stop")
        self._unregister_json_settings(
            event=self.nudge_event,
            hook_path=self.nudge_hook_path,
        )

    def _is_registered(self) -> bool:
        return self._is_registered_json_settings(event="Stop")

    def _is_nudge_registered(self) -> bool:
        return self._is_registered_json_settings(
            event=self.nudge_event,
            hook_path=self.nudge_hook_path,
        )
