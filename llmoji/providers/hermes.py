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

  - **kaomoji_position = "single"**: one final-text field per turn
    (``extra.assistant_response``); no first/last ambiguity.
  - **sidechain_strategy = "session_correlation"**: implemented via
    the companion ``subagent_stop`` hook, which writes child
    session_ids to ``~/.hermes/.llmoji-children``. The main
    ``post_llm_call`` hook checks that file and drops matching
    session_ids.
  - **system_injected_prefixes = []**: hermes delivers
    ``extra.user_message`` pre-injection, per the documented
    contract.

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
from pathlib import Path
from string import Template

from .base import (
    KaomojiPosition,
    Provider,
    SettingsCorruptError,
    SidechainStrategy,
    _atomic_write_text,
    _package_version,
)

CHILD_STATE_PATH = Path.home() / ".hermes" / ".llmoji-children"
SUBAGENT_STOP_HOOK_FILENAME = "subagent-stop.sh"


class HermesProvider(Provider):
    name = "hermes"
    hooks_dir = Path.home() / ".hermes" / "agent-hooks"
    settings_path = Path.home() / ".hermes" / "config.yaml"
    settings_format = "yaml"
    journal_path = Path.home() / ".hermes" / "kaomoji-journal.jsonl"
    hook_template = "hermes.sh.tmpl"
    hook_filename = "post-llm-call.sh"
    kaomoji_position: KaomojiPosition = "single"
    sidechain_strategy: SidechainStrategy = "session_correlation"
    sidechain_config = {
        "child_state_path": str(CHILD_STATE_PATH),
        "correlation_event": "subagent_stop",
    }
    system_injected_prefixes: list[str] = []

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
            LLMOJI_VERSION=_package_version(),
        )

    # --- install / uninstall override (two hooks, one config block) ---

    def install(self) -> None:
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Main hook (post_llm_call)
        self.hook_path.write_text(self.render_hook())
        self.hook_path.chmod(0o755)
        # Companion hook (subagent_stop)
        self.subagent_hook_path.write_text(self.render_subagent_hook())
        self.subagent_hook_path.chmod(0o755)
        # Init the child-state file so the main hook's `grep -qFx`
        # never errors on a missing file.
        CHILD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHILD_STATE_PATH.touch(exist_ok=True)
        # Register both in config.yaml
        self._register()

    def uninstall(self) -> None:
        self._unregister()
        if self.hook_path.exists():
            self.hook_path.unlink()
        if self.subagent_hook_path.exists():
            self.subagent_hook_path.unlink()
        # Leave the child-state file in place; user can `rm
        # ~/.hermes/.llmoji-children` if they want a clean slate.

    # --- YAML stanza ---

    def _stanza(self) -> str:
        # Hermes shell-hooks YAML shape per docs:
        #   hooks:
        #     post_llm_call:
        #       - command: "<path>"
        #     subagent_stop:
        #       - command: "<path>"
        return (
            f"{self._MARKER_BEGIN}\n"
            "hooks:\n"
            "  post_llm_call:\n"
            f'    - command: "{self.hook_path}"\n'
            "  subagent_stop:\n"
            f'    - command: "{self.subagent_hook_path}"\n'
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
        _atomic_write_text(self.settings_path, existing + sep + self._stanza())

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
            _atomic_write_text(self.settings_path, cleaned)
        else:
            self.settings_path.unlink()

    def _is_registered(self) -> bool:
        if not self.settings_path.exists():
            return False
        return self._MARKER_BEGIN in self.settings_path.read_text()

    @staticmethod
    def _has_unmanaged_hooks_top_level(text: str) -> bool:
        """Return True iff the YAML text contains a top-level
        ``hooks:`` key not inside our managed marker block.

        Conservative: matches lines that start with ``hooks:`` (no
        leading whitespace, a colon, then end-of-line or whitespace).
        That's exactly the top-level YAML key shape Hermes documents.
        """
        if "\nhooks:" not in ("\n" + text):
            return False
        # Find managed spans
        managed_starts = [
            i for i in range(len(text))
            if text.startswith(HermesProvider._MARKER_BEGIN, i)
        ]
        managed_ends = [
            i for i in range(len(text))
            if text.startswith(HermesProvider._MARKER_END, i)
        ]
        spans = []
        for s, e in zip(managed_starts, managed_ends):
            if e > s:
                spans.append((s, e + len(HermesProvider._MARKER_END)))
        # Walk every "\nhooks:" line-start
        for line_start in _line_starts_matching(text, "hooks:"):
            inside = any(s <= line_start < e for s, e in spans)
            if not inside:
                return True
        return False


def _line_starts_matching(text: str, prefix: str) -> list[int]:
    """Return offsets in ``text`` of every line that starts with
    ``prefix``."""
    out = []
    if text.startswith(prefix):
        out.append(0)
    idx = 0
    while True:
        nl = text.find("\n", idx)
        if nl < 0:
            break
        line_start = nl + 1
        if text.startswith(prefix, line_start):
            out.append(line_start)
        idx = line_start
    return out
