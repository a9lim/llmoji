"""Hermes (NousResearch hermes-agent) provider.

Implemented against hermes-agent v0.11.0's
[Event Hooks docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/),
cross-checked against the actual source at
``hermes-agent/agent/shell_hooks.py`` + ``hermes-agent/run_agent.py``.
The applicable mechanism is **shell hooks** under
``~/.hermes/agent-hooks/`` registered via the ``hooks:`` block in
``~/.hermes/config.yaml``. (Hermes also supports gateway hooks at
``~/.hermes/hooks/<name>/HOOK.yaml + handler.py`` and plugin hooks
registered via ``ctx.register_hook()``; for a CLI-installed
journal-logger the shell-hooks path is the lowest-friction option
because it doesn't require a Python plugin and inherits the same
fail-open / stdout-JSON contract Claude Code and Codex hooks use.)

The CLI installs **two** hooks for hermes:

  - ``post-llm-call.sh`` — main journal logger; fires after every
    assistant turn that the agent loop completes. Walks
    ``extra.conversation_history`` and emits one journal row per
    kaomoji-led assistant message in the current turn (one row per
    kaomoji-led message, same multi-emit shape Claude Code + Codex
    produce).
  - ``pre-llm-call.sh`` — UserPromptSubmit-equivalent nudge that
    injects the kaomoji-reminder context.

Stdin payload (``post_llm_call``)::

    {
      "hook_event_name": "post_llm_call",
      "tool_name":       null,
      "tool_input":      null,
      "session_id":      "...",
      "cwd":             "...",          # = Path.cwd() of agent process
      "extra": {
        "user_message":          "...",  # original, pre-injection
        "assistant_response":    "...",  # final response only
        "conversation_history":  [...],  # full message list (this is
                                         # what we walk for multi-emit)
        "model":                 "...",
        "platform":              "..."
      }
    }

Stdout: JSON. ``{}`` is no-op. Malformed JSON / non-zero exit /
timeout never abort the agent loop (fail-open).

Per-provider quirks (vs claude_code / codex):

  - **One row per kaomoji-led assistant message in the current turn**,
    walked off ``extra.conversation_history``. Pre-fix the hook only
    read ``extra.assistant_response`` (the final string) and missed
    every progress message. The slice from the latest user-role
    message to the end of the array IS the current turn — every
    assistant entry in that window is a candidate row.
  - **Subagent (delegate_task) filtering: not viable on the current
    payload contract.** ``subagent_stop`` fires from the parent
    agent's process with the **parent's** ``session_id`` (no child
    id; verified at ``hermes-agent/tools/delegate_tool.py:2120``),
    and ``post_llm_call`` doesn't expose ``parent_session_id`` either,
    so neither side carries enough info to filter children from a
    shell hook. Subagent post_llm_call events therefore land in the
    journal under their own session_ids. We'll wire a real filter
    when an upstream payload change makes one possible. The fix
    we'd want: either (a) ``subagent_stop`` carries the child id, or
    (b) ``post_llm_call`` exposes ``parent_session_id`` /
    ``is_subagent``. Both are upstream concerns.
  - ``extra.user_message`` is delivered pre-injection per the
    documented contract — no system-injected prefixes to filter.

Hermes settings are YAML — same edit-with-marker-block strategy as
codex's TOML to avoid pulling in a YAML dependency for what
amounts to a few lines.
"""

from __future__ import annotations

import re
from pathlib import Path

from .._util import atomic_write_text
from .base import HookInstaller, SettingsCorruptError


class HermesProvider(HookInstaller):
    name = "hermes"
    hooks_dir = Path.home() / ".hermes" / "agent-hooks"
    settings_path = Path.home() / ".hermes" / "config.yaml"
    journal_path = Path.home() / ".hermes" / "kaomoji-journal.jsonl"
    hook_template = "hermes.sh.tmpl"
    hook_filename = "post-llm-call.sh"
    main_event = "post_llm_call"
    # The validate partial is inlined inside a per-message
    # ``while read`` loop in the rendered hook (one iteration per
    # assistant message in the current turn); ``continue`` is the
    # right skip action — a non-kaomoji message skips its row
    # without bailing the rest of the walk. The base default
    # ``"exit 0"`` would terminate the loop's subshell on the first
    # non-kaomoji message, dropping every later kaomoji-led message
    # in the same turn. Same shape as claude_code / codex now that
    # hermes is multi-emit too. The closing ``echo '{}'; exit 0``
    # in the template body satisfies the hermes stdout-JSON contract
    # after the loop completes.
    skip_action = "continue"
    system_injected_prefixes: list[str] = []

    # Nudge: pre_llm_call with a bare ``{context: ...}`` shape (per
    # docs, "the only hook whose return value is used"). Different
    # template from the Claude/Codex shared one — Hermes wraps no
    # ``hookSpecificOutput`` envelope.
    nudge_hook_template = "hermes_nudge.sh.tmpl"
    nudge_hook_filename = "pre-llm-call.sh"
    nudge_event = "pre_llm_call"
    nudge_message = (
        "Please begin your message with a kaomoji that best represents "
        "how you feel."
    )

    _MARKER_BEGIN = "# >>> llmoji begin (managed) >>>"
    _MARKER_END = "# <<< llmoji end (managed) <<<"

    # Empty-placeholder hooks lines that Hermes ships in its default
    # config — ``hooks: {}`` and ``hooks: []`` — are functionally
    # identical to having no ``hooks:`` key at all. We replace them in
    # place with our managed stanza on install rather than refusing
    # the way we would for a populated hooks block.
    _EMPTY_HOOKS_RE = re.compile(
        r"^hooks:[ \t]*(?:\{[ \t]*\}|\[[ \t]*\])[ \t]*$",
        re.MULTILINE,
    )

    # --- YAML stanza ---

    def _stanza(self) -> str:
        # Hermes shell-hooks YAML shape per docs:
        #   hooks:
        #     pre_llm_call:
        #       - command: "<nudge>"
        #     post_llm_call:
        #       - command: "<main>"
        lines = [self._MARKER_BEGIN, "hooks:"]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            lines.append("  pre_llm_call:")
            lines.append(f'    - command: "{self.nudge_hook_path}"')
        lines.append("  post_llm_call:")
        lines.append(f'    - command: "{self.hook_path}"')
        lines.append(self._MARKER_END)
        return "\n".join(lines) + "\n"

    def _register(self) -> None:
        existing = (
            self.settings_path.read_text()
            if self.settings_path.exists()
            else ""
        )
        if self._MARKER_BEGIN in existing:
            return  # idempotent
        # Refuse to clobber a populated top-level `hooks:` key —
        # appending a fresh `hooks:` to a YAML file that already has
        # one yields a duplicate-key document; most YAML parsers
        # silently last-write-wins, which would discard the user's
        # prior hook config. Empty placeholders (``hooks: {}`` /
        # ``hooks: []``, the Hermes default) are handled below — we
        # replace those in place rather than refusing.
        if self._has_unmanaged_hooks_top_level(existing):
            raise SettingsCorruptError(
                self.settings_path,
                "existing top-level 'hooks:' key is not managed by "
                "llmoji. Add the hermes hooks under that block by "
                "hand, or move the file aside and re-run.",
            )
        empty_match = self._EMPTY_HOOKS_RE.search(existing)
        if empty_match is not None:
            # Replace the empty placeholder line with our managed
            # stanza in place. Avoids creating duplicate top-level
            # ``hooks:`` keys (which YAML parsers resolve via silent
            # last-write-wins, leaving the file ambiguous).
            new_text = (
                existing[: empty_match.start()]
                + self._stanza().rstrip()
                + existing[empty_match.end():]
            )
            atomic_write_text(self.settings_path, new_text)
            return
        sep = "\n\n" if existing and not existing.endswith("\n") else "\n"
        atomic_write_text(self.settings_path, existing + sep + self._stanza())

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
            atomic_write_text(self.settings_path, cleaned)
        else:
            self.settings_path.unlink()

    def _check_registrations(self) -> tuple[bool, bool]:
        # Single-read override: hermes wires both hooks inside one
        # marker-fenced stanza, so one file read tells us about both.
        # Default HookInstaller._check_registrations would route through
        # the JSON-settings batch (wrong for YAML); cleaner to
        # override directly.
        if not self.settings_path.exists():
            return False, False
        present = self._MARKER_BEGIN in self.settings_path.read_text()
        return present, present

    @classmethod
    def _has_unmanaged_hooks_top_level(cls, text: str) -> bool:
        """Return True iff the YAML text contains a populated
        top-level ``hooks:`` key not inside our managed marker block.

        Strategy: cut every ``BEGIN…END`` managed span out of the text
        and search the remainder for ``^hooks:`` lines. Empty
        placeholders (``hooks: {}`` / ``hooks: []``, the Hermes
        default config shape) don't count — :meth:`_register`
        replaces those in place. Conservative on the populated
        case: any ``^hooks:`` line outside our marker block that
        doesn't match :attr:`_EMPTY_HOOKS_RE` triggers refusal.
        """
        pattern = re.escape(cls._MARKER_BEGIN) + r".*?" + re.escape(cls._MARKER_END)
        unmanaged = re.sub(pattern, "", text, flags=re.DOTALL)
        for m in re.finditer(r"^hooks:.*$", unmanaged, flags=re.MULTILINE):
            if not cls._EMPTY_HOOKS_RE.match(m.group(0)):
                return True
        return False
