"""Codex provider.

Hook fires under Codex's per-session hook config. Codex delivers the
final assistant text via the ``last_assistant_message`` field on the
Stop-event payload — no transcript-walking required for the assistant
side. The ``transcript_path`` field on the payload still points at
the per-session rollout JSONL, which we use to resolve the latest
real user message.

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

from .base import (
    KaomojiPosition,
    Provider,
    SidechainStrategy,
    _atomic_write_text,
)


class CodexProvider(Provider):
    name = "codex"
    hooks_dir = Path.home() / ".codex" / "hooks"
    settings_path = Path.home() / ".codex" / "config.toml"
    settings_format = "json"  # we wrap the toml-edit logic below
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

    # Codex's settings file is TOML, not JSON. We hand-roll the
    # register/unregister logic here rather than pulling in a TOML
    # parser dependency for what amounts to two stanzas. Idempotent
    # via marker comments. If a future Codex release moves to a
    # different settings shape this is the only place to touch.

    _MARKER_BEGIN = "# >>> llmoji begin (managed) >>>"
    _MARKER_END = "# <<< llmoji end (managed) <<<"

    def _stanza(self) -> str:
        return (
            f"{self._MARKER_BEGIN}\n"
            "[hooks.stop]\n"
            f'command = "{self.hook_path}"\n'
            f"{self._MARKER_END}\n"
        )

    def _register(self) -> None:
        existing = (
            self.settings_path.read_text()
            if self.settings_path.exists()
            else ""
        )
        if self._MARKER_BEGIN in existing:
            return  # idempotent
        # Refuse to clobber: if the user already has their own
        # `[hooks.stop]` section in the file (without our marker
        # block), appending another one yields a duplicate-table
        # TOML error and hides whatever they had. Loud failure;
        # they can move/delete their config and re-run.
        if self._has_unmanaged_hooks_stop(existing):
            from .base import SettingsCorruptError
            raise SettingsCorruptError(
                self.settings_path,
                "existing [hooks.stop] section is not managed by "
                "llmoji. Remove it (or move the whole file aside) "
                "before installing the codex hook.",
            )
        sep = "\n\n" if existing and not existing.endswith("\n") else "\n"
        _atomic_write_text(self.settings_path, existing + sep + self._stanza())

    @staticmethod
    def _has_unmanaged_hooks_stop(text: str) -> bool:
        """Return True iff the TOML text contains a `[hooks.stop]`
        section that is NOT inside our managed marker block."""
        # Cheap scan: find every `[hooks.stop]` occurrence and check
        # whether any is outside any `>>> begin <<<` ... `<<< end >>>`
        # block. We don't need a full TOML parser for this — the
        # check is conservative (false-positive is fine, false-
        # negative is the bad case).
        if "[hooks.stop]" not in text:
            return False
        managed_starts = [
            i for i in range(len(text))
            if text.startswith(CodexProvider._MARKER_BEGIN, i)
        ]
        managed_ends = [
            i for i in range(len(text))
            if text.startswith(CodexProvider._MARKER_END, i)
        ]
        # Build managed spans (begin → end pairs, in order).
        spans = []
        for s, e in zip(managed_starts, managed_ends):
            if e > s:
                spans.append((s, e + len(CodexProvider._MARKER_END)))
        # Any [hooks.stop] occurrence outside every span is unmanaged.
        idx = 0
        while True:
            idx = text.find("[hooks.stop]", idx)
            if idx < 0:
                return False
            inside = any(s <= idx < e for s, e in spans)
            if not inside:
                return True
            idx += len("[hooks.stop]")

    def _unregister(self) -> None:
        if not self.settings_path.exists():
            return
        text = self.settings_path.read_text()
        if self._MARKER_BEGIN not in text:
            return
        # Strip the managed block (and a trailing blank line if any).
        before, _, rest = text.partition(self._MARKER_BEGIN)
        _, _, after = rest.partition(self._MARKER_END)
        cleaned = before.rstrip() + ("\n" + after.lstrip() if after.strip() else "")
        cleaned = cleaned.rstrip() + "\n" if cleaned.strip() else ""
        if cleaned:
            _atomic_write_text(self.settings_path, cleaned)
        else:
            self.settings_path.unlink()

    def _is_registered(self) -> bool:
        if not self.settings_path.exists():
            return False
        return self._MARKER_BEGIN in self.settings_path.read_text()
