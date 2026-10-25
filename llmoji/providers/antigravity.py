"""Antigravity provider (Google's ``agy`` CLI).

Antigravity loads plugins from ``plugins/<name>/`` under each
customization root; the global root is ``~/.gemini/config/``. A plugin
is a directory holding a ``plugin.json`` marker and an optional
``hooks.json``, so llmoji owns one outright —
``~/.gemini/config/plugins/llmoji/`` with both files plus the two hook
scripts — and uninstall removes the directory. Contract source: the
``agy-customizations`` skill Antigravity bundles at
``~/.gemini/antigravity-cli/builtin/skills/agy-customizations/docs/``
(agy 1.4.7), cross-checked against live payloads.

``hooks.json`` is keyed by hook *name*, each mapping events to
handlers; ``Stop`` and ``PreInvocation`` take a flat handler list, not
the matcher groups tool events use::

    {"llmoji": {"Stop": [{"type": "command", "command": "..."}],
                "PreInvocation": [{"type": "command", "command": "..."}]}}

Per-provider quirks (vs claude_code / codex):

  - **Every payload names the model.** Hook stdin carries
    ``modelName`` (the resolved id, e.g. ``gemini-4-argon-medium``),
    ``transcriptPath`` (``brain/<id>/.system_generated/logs/
    transcript_full.jsonl``), ``workspacePaths`` and, since agy 1.4.7,
    ``lastUserInput`` — the raw prompt, without the ``<USER_REQUEST>``
    wrapper the transcript stores it in.
  - **Transcript steps, not messages.** Each line is a step;
    ``type == "PLANNER_RESPONSE"`` with non-empty ``content`` is a
    model message, ``USER_INPUT`` opens a turn. The transcript is
    complete when ``Stop`` fires (verified live).
  - **The nudge is a per-invocation ephemeral message.** Under
    ``--hard`` a ``PreInvocation`` hook returns ``{"injectSteps":
    [{"ephemeralMessage": ...}]}`` before every model call — each call
    can open a new message, and ephemeral steps are transient by
    design (Antigravity's own docs use exactly this for a reminder).
  - **Hooks speak JSON on stdout.** Both scripts print a JSON object
    (``{}`` from the Stop hook: no ``decision`` lets the agent stop).
  - **No replay.** Transcripts carry no per-step model id (it lives in
    protobuf blobs in the conversation SQLite store), so ``llmoji
    import`` doesn't cover Antigravity; live capture is the path.
  - **Subagents unfiltered.** Whether a subagent execution fires the
    global ``Stop`` with its own transcript isn't verified; the
    conversation store's ``parent_references`` table would be the
    signal to filter on.
  - **System-prompt doc is ``~/.gemini/GEMINI.md``.** Antigravity
    reads it as global rules; ``~/.gemini/config/`` itself is guarded
    configuration, and its policy checker treats edits there by the
    agent as tampering — ``llmoji install`` runs outside the agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._util import write_json
from .base import (
    HookInstaller,
    SettingsCorruptError,
    _load_json_strict,
    _same_hook_command,
    home_path,
)

# Hook name under which our handlers live in the plugin's hooks.json.
HOOK_NAME = "llmoji"


class AntigravityProvider(HookInstaller):
    name = "antigravity"
    hooks_dir = home_path(".gemini", "config", "plugins", "llmoji")
    settings_path = home_path(".gemini", "config", "plugins", "llmoji", "hooks.json")
    journal_path = home_path(".gemini", "kaomoji-journal.jsonl")
    system_prompt_doc_path = home_path(".gemini", "GEMINI.md")
    hook_template = "antigravity.sh.tmpl"
    # Per-message while-loop, like claude_code / codex. The template
    # prints ``{}`` before the walk, so a skipped row needs no output.
    skip_action = "continue"
    # ``lastUserInput`` is the raw prompt; nothing injected to strip.
    system_injected_prefixes: list[str] = []
    nudge_hook_template = "antigravity_nudge.sh.tmpl"
    nudge_hook_filename = "kaomoji-nudge.sh"
    nudge_event = "PreInvocation"

    @property
    def home_dir(self) -> Path:
        return Path.home() / ".gemini"

    @property
    def plugin_manifest_path(self) -> Path:
        return self.settings_path.parent / "plugin.json"

    def is_present(self) -> bool:
        """``~/.gemini`` alone is also Gemini CLI's home, so look for
        a directory only Antigravity creates."""
        return (
            (self.home_dir / "antigravity-cli").exists()
            or (self.home_dir / "config").exists()
        )

    # --- registration: our own plugin.json + hooks.json ---

    def _register(self, *, include_nudge: bool = True) -> None:
        """Write the plugin marker and set our named hook in
        ``hooks.json``: ``Stop`` always, ``PreInvocation`` under
        ``--hard``. Any other hook names a user added to the file are
        kept; a ``--soft`` re-install drops a ``PreInvocation`` an
        earlier ``--hard`` left."""
        cfg = _load_json_strict(self.settings_path)
        spec: dict[str, Any] = {
            "Stop": [{"type": "command", "command": str(self.hook_path)}],
        }
        if include_nudge and self.nudge_hook_path is not None:
            spec[self.nudge_event] = [
                {"type": "command", "command": str(self.nudge_hook_path)},
            ]
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.get(HOOK_NAME) != spec:
            cfg[HOOK_NAME] = spec
            write_json(self.settings_path, cfg)
        if not self.plugin_manifest_path.exists():
            write_json(self.plugin_manifest_path, {"name": HOOK_NAME})

    def _unregister(self) -> None:
        """Drop our named hook; once ``hooks.json`` holds nothing else,
        remove it and the plugin marker. A corrupt file is left alone."""
        if self.settings_path.exists():
            try:
                cfg = _load_json_strict(self.settings_path)
            except SettingsCorruptError:
                return
            if HOOK_NAME in cfg:
                cfg.pop(HOOK_NAME)
                if cfg:
                    write_json(self.settings_path, cfg)
                else:
                    self.settings_path.unlink()
        if not self.settings_path.exists() and self.plugin_manifest_path.exists():
            self.plugin_manifest_path.unlink()

    def uninstall(self) -> None:
        super().uninstall()
        # The plugin dir, then ``plugins/`` if install created it and
        # nothing else lives there. A missing or non-empty dir stays.
        for d in (self.hooks_dir, self.hooks_dir.parent):
            try:
                d.rmdir()
            except OSError:
                break

    def _check_registrations(self) -> tuple[bool, bool]:
        if not self.plugin_manifest_path.exists():
            return False, False
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return False, False
        spec = cfg.get(HOOK_NAME)
        if not isinstance(spec, dict):
            return False, False

        def registered(event: str, path: Path | None) -> bool:
            handlers = spec.get(event)
            if path is None or not isinstance(handlers, list):
                return False
            return any(
                isinstance(h, dict)
                and isinstance(h.get("command"), str)
                and _same_hook_command(h["command"], str(path))
                for h in handlers
            )

        return (
            registered("Stop", self.hook_path),
            registered(self.nudge_event, self.nudge_hook_path),
        )
