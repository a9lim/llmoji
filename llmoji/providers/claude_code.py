"""Claude Code provider.

Hook fires as a Stop event under ``~/.claude/settings.json``. Each
assistant turn delivers payload JSON on stdin with a
``transcript_path`` pointing at the live JSONL transcript; we read
that file to recover the full assistant text + parent-walk to the
nearest human user turn.

Per-provider quirks (vs codex / hermes):

  - **Kaomoji on the FIRST text-bearing entry of the current turn.**
    Each assistant content block (text / tool_use / thinking) is its
    own top-level transcript entry; one turn produces many. The
    UserPromptSubmit nudge drives the model to lead with a kaomoji,
    so the first text in the turn is the kaomoji-bearing reply.
  - **Sidechain filter via the ``isSidechain`` field flag.** Set on
    every event in a Task-tool-spawned subagent session; the hook
    drops these rows.
  - **System-injected user-role prefix:** slash-command activations
    dump the skill body as a user-role message starting with
    ``"Base directory for this skill:"``. Drop it so it doesn't
    contaminate ``user_text``.
"""

from __future__ import annotations

from pathlib import Path

from .base import Provider


class ClaudeCodeProvider(Provider):
    name = "claude_code"
    hooks_dir = Path.home() / ".claude" / "hooks"
    settings_path = Path.home() / ".claude" / "settings.json"
    journal_path = Path.home() / ".claude" / "kaomoji-journal.jsonl"
    hook_template = "claude_code.sh.tmpl"
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
