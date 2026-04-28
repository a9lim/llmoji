"""Claude Code provider.

Hook fires as a Stop event under ``~/.claude/settings.json``. Each
assistant turn delivers payload JSON on stdin with a
``transcript_path`` pointing at the live JSONL transcript; we read
that file to recover the full assistant text + parent-walk to the
nearest human user turn.

Per-provider quirks (vs codex / hermes):

  - **One row per kaomoji-led text-bearing assistant entry in the
    current turn.** Each assistant content block (text / tool_use /
    thinking) is its own top-level transcript entry; one turn
    produces many. The UserPromptSubmit nudge drives the model to
    lead each text block with a kaomoji, so a tool-heavy turn
    (text → tool_use → text → tool_use → text) emits up to one row
    per text block. Non-kaomoji text blocks are skipped without
    aborting the rest of the walk.
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

from .base import JsonSettingsHookInstaller


class ClaudeCodeProvider(JsonSettingsHookInstaller):
    name = "claude_code"
    hooks_dir = Path.home() / ".claude" / "hooks"
    settings_path = Path.home() / ".claude" / "settings.json"
    journal_path = Path.home() / ".claude" / "kaomoji-journal.jsonl"
    hook_template = "claude_code.sh.tmpl"
    # The validate partial is inlined inside a per-entry while-loop
    # in the rendered hook (one iteration per text-bearing assistant
    # entry in the current turn); ``continue`` is the right skip
    # action — a non-kaomoji entry skips its row without bailing the
    # rest of the loop. The base default ``"exit 0"`` would terminate
    # the loop's subshell on the first non-kaomoji entry, dropping
    # every later kaomoji-led entry in the same turn.
    skip_action = "continue"
    system_injected_prefixes = [
        "Base directory for this skill:",
    ]

    # Nudge attrs (template / filename / event / message) inherited
    # from JsonSettingsHookInstaller — same wording + envelope shared with
    # Codex (the UserPromptSubmit envelope is byte-identical between
    # the two harnesses).
