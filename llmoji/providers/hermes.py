"""Hermes (NousResearch hermes-agent) provider.

Implemented against hermes-agent v0.11.0's
[Event Hooks docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/).
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
    assistant turn that the agent loop completes.
  - ``subagent-stop.sh`` — companion sidechain registrar; records
    delegated child session_ids to a state file the main hook
    consults to drop subagent traffic.

Stdin payload (``post_llm_call``)::

    {
      "hook_event_name": "post_llm_call",
      "session_id": "...",
      "cwd": "...",
      "extra": {
        "user_message":          "...",
        "assistant_response":    "...",
        "model":                 "...",
        "platform":              "...",
        "conversation_history":  [...]
      }
    }

Stdin payload (``subagent_stop``)::

    {
      "hook_event_name": "subagent_stop",
      "session_id": "...",          // child session id
      "extra": {
        "parent_session_id": "...",
        "child_role":        "...",
        "child_status":      "..."
      }
    }

Stdout: JSON. ``{}`` is no-op. Malformed JSON / non-zero exit /
timeout never abort the agent loop (fail-open).

Per-provider quirks (vs claude_code / codex):

  - **Single final-text field per turn** (``extra.assistant_response``);
    no first/last ambiguity.
  - **Sidechain handling via session correlation:** a companion
    ``subagent_stop`` hook writes child session_ids to
    ``~/.hermes/.llmoji-children``; the main ``post_llm_call`` hook
    drops matching session_ids.
  - ``extra.user_message`` is delivered pre-injection per the
    documented contract — no system-injected prefixes to filter.

⚠ The hermes path was implemented from docs only. The shell hook
shape is well-documented and other providers' hooks share the same
``stdin JSON / stdout JSON / fail-open`` skeleton, but live-traffic
verification of (a) the exact ``extra.*`` keys delivered by
``post_llm_call``, (b) the companion subagent_stop event firing as
expected on real ``delegate_task`` traffic, (c) ``user_message``
arriving clean, would still be useful before claiming the hermes
provider is battle-tested.

Hermes settings are YAML — same edit-with-marker-block strategy as
codex's TOML to avoid pulling in a YAML dependency for what
amounts to a few lines.
"""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path
from string import Template

from .._util import atomic_write_text, package_version
from .base import Provider, SettingsCorruptError

CHILD_STATE_PATH = Path.home() / ".hermes" / ".llmoji-children"
SUBAGENT_STOP_HOOK_FILENAME = "subagent-stop.sh"


class HermesProvider(Provider):
    name = "hermes"
    hooks_dir = Path.home() / ".hermes" / "agent-hooks"
    settings_path = Path.home() / ".hermes" / "config.yaml"
    journal_path = Path.home() / ".hermes" / "kaomoji-journal.jsonl"
    hook_template = "hermes.sh.tmpl"
    hook_filename = "post-llm-call.sh"
    main_event = "post_llm_call"
    # Hermes shell hooks must emit stdout JSON; ``{}`` is no-op. The
    # validate-partial defaults to ``exit 0`` for claude_code/codex,
    # which would leave hermes silently violating the contract.
    skip_action = "echo '{}'; exit 0"
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

    @property
    def subagent_hook_path(self) -> Path:
        return self.hooks_dir / SUBAGENT_STOP_HOOK_FILENAME

    # --- hook rendering ---

    def render_subagent_hook(self) -> str:
        """Render the companion subagent_stop hook."""
        template_text = importlib.resources.files("llmoji._hooks").joinpath(
            "hermes_subagent_stop.sh.tmpl"
        ).read_text()
        return Template(template_text).safe_substitute(
            CHILD_STATE_PATH=str(CHILD_STATE_PATH),
            LLMOJI_VERSION=package_version(),
        )

    # --- install / uninstall override (three hooks, one config block) ---

    def install(self) -> None:
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Main hook (post_llm_call)
        self.hook_path.write_text(self.render_hook())
        self.hook_path.chmod(0o755)
        # Companion hook (subagent_stop)
        self.subagent_hook_path.write_text(self.render_subagent_hook())
        self.subagent_hook_path.chmod(0o755)
        # Nudge hook (pre_llm_call)
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            self.nudge_hook_path.write_text(self.render_nudge_hook())
            self.nudge_hook_path.chmod(0o755)
        # Init the child-state file so the main hook's `grep -qFx`
        # never errors on a missing file.
        CHILD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHILD_STATE_PATH.touch(exist_ok=True)
        # Register all three in config.yaml
        self._register()

    def uninstall(self) -> None:
        self._unregister()
        if self.hook_path.exists():
            self.hook_path.unlink()
        if self.subagent_hook_path.exists():
            self.subagent_hook_path.unlink()
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            if self.nudge_hook_path.exists():
                self.nudge_hook_path.unlink()
        # Leave the child-state file in place; user can `rm
        # ~/.hermes/.llmoji-children` if they want a clean slate.

    # --- YAML stanza ---

    def _stanza(self) -> str:
        # Hermes shell-hooks YAML shape per docs:
        #   hooks:
        #     pre_llm_call:
        #       - command: "<nudge>"
        #     post_llm_call:
        #       - command: "<main>"
        #     subagent_stop:
        #       - command: "<subagent>"
        lines = [self._MARKER_BEGIN, "hooks:"]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            lines.append("  pre_llm_call:")
            lines.append(f'    - command: "{self.nudge_hook_path}"')
        lines.append("  post_llm_call:")
        lines.append(f'    - command: "{self.hook_path}"')
        lines.append("  subagent_stop:")
        lines.append(f'    - command: "{self.subagent_hook_path}"')
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
        # Refuse to clobber an existing top-level `hooks:` key —
        # appending a fresh `hooks:` to a YAML file that already has
        # one yields a duplicate-key document; most YAML parsers
        # silently last-write-wins, which would discard the user's
        # prior hook config.
        if self._has_unmanaged_hooks_top_level(existing):
            raise SettingsCorruptError(
                self.settings_path,
                "existing top-level 'hooks:' key is not managed by "
                "llmoji. Add the hermes hooks under that block by "
                "hand, or move the file aside and re-run.",
            )
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

    def _is_registered(self) -> bool:
        if not self.settings_path.exists():
            return False
        return self._MARKER_BEGIN in self.settings_path.read_text()

    def _is_nudge_registered(self) -> bool:
        # Hermes registers all three hooks atomically inside one
        # marker-fenced YAML stanza, so the nudge is wired up iff the
        # marker is present — same check as :meth:`_is_registered`.
        return self._is_registered()

    @classmethod
    def _has_unmanaged_hooks_top_level(cls, text: str) -> bool:
        """Return True iff the YAML text contains a top-level
        ``hooks:`` key not inside our managed marker block.

        Strategy: cut every ``BEGIN…END`` managed span out of the text
        and search the remainder for a ``^hooks:`` line. Conservative
        — matches lines that start with ``hooks:`` with no leading
        whitespace, which is exactly the top-level YAML key shape
        Hermes documents.
        """
        pattern = re.escape(cls._MARKER_BEGIN) + r".*?" + re.escape(cls._MARKER_END)
        unmanaged = re.sub(pattern, "", text, flags=re.DOTALL)
        return bool(re.search(r"^hooks:", unmanaged, flags=re.MULTILINE))
