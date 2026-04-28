"""Claude Code provider.

Hook fires as a Stop event under ``~/.claude/settings.json``. Each
assistant turn delivers payload JSON on stdin with a
``transcript_path`` pointing at the live JSONL transcript; we read
that file to recover the full assistant text + parent-walk to the
nearest human user turn.

Key per-provider quirks (vs codex / hermes):

  - **kaomoji_position = "first"**: a Claude Code assistant message
    is one event whose ``content`` array interleaves text + tool_use
    + text. The kaomoji-prefixed reply is always the FIRST text
    block; later text is post-tool continuation.
  - **sidechain_strategy = "field_flag"** with field
    ``isSidechain``. Set on every event in a Task-tool-spawned
    subagent session.
  - **system_injected_prefixes = ["Base directory for this skill:"]**:
    slash-command activations dump the skill body as a user-role
    message. Drop these so they don't pollute ``user_text``.
"""

from __future__ import annotations

from pathlib import Path

from .base import KaomojiPosition, Provider, SidechainStrategy


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    hooks_dir = Path.home() / ".claude" / "hooks"
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_format = "json"
    journal_path = Path.home() / ".claude" / "kaomoji-journal.jsonl"
    hook_template = "claude_code.sh.tmpl"
    hook_filename = "kaomoji-log.sh"
    kaomoji_position: KaomojiPosition = "first"
    sidechain_strategy: SidechainStrategy = "field_flag"
    sidechain_config = {"field": "isSidechain"}
    system_injected_prefixes = [
        "Base directory for this skill:",
    ]

    # Nudge: shared template with Codex (the UserPromptSubmit envelope
    # is byte-identical between the two harnesses).
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
