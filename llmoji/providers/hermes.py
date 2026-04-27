"""Hermes (NousResearch hermes-agent) provider.

Hook fires as a YAML-configured ``post_llm_call`` event under
``~/.hermes/config.yaml``. The payload (stdin JSON) carries
``hook_event_name``, ``session_id``, ``cwd``, plus an ``extra``
dict containing the pre-injection ``user_message``, the final
``assistant_response`` text, the active model, the platform, and
the conversation history. All six llmoji fields land cleanly in
one event — single-event payload, so neither the first/last
disambiguation Claude Code and Codex need.

Key per-provider quirks (vs claude_code / codex):

  - **kaomoji_position = "single"**: one final-text field, no
    first/last ambiguity.
  - **sidechain_strategy = "session_correlation"**:
    ``post_llm_call`` fires for both parent and child sessions;
    child sessions are identified by correlating ``session_id``
    against ``delegate_task`` events. Concrete config: track child
    session_ids from a companion ``subagent_stop`` registration;
    drop ``post_llm_call`` events whose session_id matches a known
    child. Implemented inline in the hook template via a small
    state file at ``~/.hermes/.llmoji-children``.
  - **system_injected_prefixes = []**: hermes delivers
    ``user_message`` pre-injection — no system-injection scrubbing
    needed.

⚠ **Empirical validation pending.** The hermes provider in v1.0
is implemented from the documented hermes-agent v0.11.0 hook
contract (see plan v1.0 prereq §4) but has not yet been smoke-
tested end-to-end against a live agent. Three items still want
real-traffic verification before claiming stability:

  1. The exact ``extra.*`` keys delivered by ``post_llm_call``
     (the docs example block was for ``pre_tool_call``).
  2. That the session-correlation sidechain strategy works
     against actual ``delegate_task`` traffic.
  3. That ``user_message`` arrives clean (no system-injection
     prefixes that need filtering).

Treat the hermes hook as experimental until that validation
lands; the journal file format and CLI surface are stable across
this verification.

Hermes settings are YAML — same edit-with-marker-block strategy as
codex's TOML to avoid pulling in a YAML dependency for what
amounts to a few lines.
"""

from __future__ import annotations

from pathlib import Path

from .base import KaomojiPosition, Provider, SidechainStrategy


class HermesProvider(Provider):
    name = "hermes"
    hooks_dir = Path.home() / ".hermes" / "agent-hooks"
    settings_path = Path.home() / ".hermes" / "config.yaml"
    settings_format = "yaml"
    journal_path = Path.home() / ".hermes" / "kaomoji-journal.jsonl"
    hook_template = "hermes.sh.tmpl"
    hook_filename = "kaomoji-log.sh"
    kaomoji_position: KaomojiPosition = "single"
    sidechain_strategy: SidechainStrategy = "session_correlation"
    sidechain_config = {
        "child_state_path": str(Path.home() / ".hermes" / ".llmoji-children"),
        "correlation_event": "delegate_task",
    }
    system_injected_prefixes: list[str] = []

    _MARKER_BEGIN = "# >>> llmoji begin (managed) >>>"
    _MARKER_END = "# <<< llmoji end (managed) <<<"

    def _stanza(self) -> str:
        # YAML-shaped registration. Hermes's `hooks` block accepts a
        # mapping {event_name: [list of command paths]}.
        return (
            f"{self._MARKER_BEGIN}\n"
            "hooks:\n"
            "  post_llm_call:\n"
            f"    - {self.hook_path}\n"
            f"{self._MARKER_END}\n"
        )

    def _register(self) -> None:
        existing = (
            self.settings_path.read_text()
            if self.settings_path.exists()
            else ""
        )
        if self._MARKER_BEGIN in existing:
            return
        sep = "\n\n" if existing and not existing.endswith("\n") else "\n"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(existing + sep + self._stanza())

    def _unregister(self) -> None:
        if not self.settings_path.exists():
            return
        text = self.settings_path.read_text()
        if self._MARKER_BEGIN not in text:
            return
        before, _, rest = text.partition(self._MARKER_BEGIN)
        _, _, after = rest.partition(self._MARKER_END)
        cleaned = before.rstrip() + ("\n" + after.lstrip() if after.strip() else "")
        cleaned = cleaned.rstrip() + "\n" if cleaned.strip() else ""
        if cleaned:
            self.settings_path.write_text(cleaned)
        else:
            self.settings_path.unlink()

    def _is_registered(self) -> bool:
        if not self.settings_path.exists():
            return False
        return self._MARKER_BEGIN in self.settings_path.read_text()
