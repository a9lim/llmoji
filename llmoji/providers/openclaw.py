"""OpenClaw provider — TS plugin bundle at
``~/.openclaw/plugins/llmoji-kaomoji/`` plus a ``config.json`` flag
flip to grant conversation-hook access.

OpenClaw (https://openclaw.ai) ships a TypeScript plugin SDK with
``definePluginEntry`` + ``api.on(hookName, handler)``. ``llm_input``
and ``llm_output`` are *conversation hooks* — gated behind a
per-plugin opt-in flag (``plugins.entries.<plugin_id>.hooks.allow
ConversationAccess``). Without the flag set, OpenClaw refuses to
fire those hooks for the plugin and the journal stays empty.

Install lifecycle:

  1. Render ``index.ts`` + ``openclaw.plugin.json`` from package
     templates into ``~/.openclaw/plugins/llmoji-kaomoji/``.
  2. Read ``~/.openclaw/config.json``, set
     ``plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess
     = true``, atomic-write back.

Uninstall is the inverse — remove the plugin dir, unset the
``llmoji-kaomoji`` entry under ``plugins.entries`` (drop the whole
sub-mapping rather than just toggling the flag, so the user's
``plugins.entries`` mapping is byte-stable when llmoji is the only
configured plugin).

Per-provider notes (vs claude_code / codex / hermes / opencode):

  - **No bash hook.** Same as opencode — ``hook_path`` points at
    ``index.ts``.
  - **Settings file is JSON, but the shape isn't the
    ``hooks``-keyed shape claude_code / codex use.** The base
    :class:`JsonSettingsHookInstaller` helpers don't apply; this
    provider walks the ``plugins.entries`` sub-tree directly.
  - **Sidechain filter is honest.** OpenClaw fires
    ``subagent_spawned`` / ``subagent_ended`` events with the
    runId, so the plugin tracks subagent runIds and drops their
    ``llm_output`` rows. Better story than claude_code's
    ``isSidechain`` field flag (which is per-event rather than
    per-run).
  - **System-injection prefix list is empty for now.** OpenClaw
    delivers the raw user prompt at ``llm_input.prompt`` per the
    hook contract; if real-traffic inspection later shows leaked
    injection prefixes, populate ``system_injected_prefixes`` and
    the TS plugin's prefix filter (currently absent — would need
    a template addition).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._util import write_json
from .base import (
    PluginInstaller,
    SettingsCorruptError,
    _load_json_strict,
)

# OpenClaw plugin id used both as the plugin directory name and the
# key under ``plugins.entries`` in ``~/.openclaw/config.json``.
# Matches the ``id`` field in the rendered ``openclaw.plugin.json``.
PLUGIN_ID = "llmoji-kaomoji"


class OpenclawProvider(PluginInstaller):
    name = "openclaw"
    # Plugin dir = ``~/.openclaw/plugins/llmoji-kaomoji/`` — OpenClaw's
    # ``plugins install <path>`` command copies the entire bundle
    # directory into this location, but ``llmoji install`` writes the
    # rendered files directly so the user doesn't need to invoke
    # OpenClaw's own CLI for setup.
    plugin_dir = Path.home() / ".openclaw" / "plugins" / PLUGIN_ID
    settings_path = Path.home() / ".openclaw" / "config.json"
    journal_path = Path.home() / ".llmoji" / "journals" / "openclaw.jsonl"
    plugin_files = [
        ("openclaw_index.ts.tmpl", "index.ts"),
        ("openclaw_plugin.json.tmpl", "openclaw.plugin.json"),
    ]

    # --- presence detection ---

    def is_present(self) -> bool:
        """OpenClaw's home dir is ``~/.openclaw/``. Match the bash
        providers' ``settings_path.parent.exists()`` rule rather than
        :class:`PluginInstaller`'s default ``plugin_dir.parent`` — the
        plugin_dir parent (``~/.openclaw/plugins/``) doesn't exist on
        a fresh OpenClaw install, so the default would false-negative.
        """
        return self.settings_path.parent.exists()

    # --- registration via plugins.entries.<id>.hooks.allowConversationAccess ---

    def _register(self) -> None:
        """Atomic read-modify-write of ``~/.openclaw/config.json`` to
        set ``plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess
        = true``. Idempotent — re-running is a no-op when the flag
        is already on. Refuses to mutate a corrupt config (loud
        :class:`SettingsCorruptError` beats silent overwrite).
        """
        cfg = _load_json_strict(self.settings_path)
        plugins_field = cfg.get("plugins")
        if plugins_field is not None and not isinstance(plugins_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing 'plugins' field is "
                f"{type(plugins_field).__name__}, not an object",
            )
        plugins: dict[str, Any] = cfg.setdefault("plugins", {})

        entries_field = plugins.get("entries")
        if entries_field is not None and not isinstance(entries_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing 'plugins.entries' is "
                f"{type(entries_field).__name__}, not an object",
            )
        entries: dict[str, Any] = plugins.setdefault("entries", {})

        entry_field = entries.get(PLUGIN_ID)
        if entry_field is not None and not isinstance(entry_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing plugins.entries[{PLUGIN_ID!r}] is "
                f"{type(entry_field).__name__}, not an object",
            )
        entry: dict[str, Any] = entries.setdefault(PLUGIN_ID, {})

        hooks_field = entry.get("hooks")
        if hooks_field is not None and not isinstance(hooks_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing plugins.entries[{PLUGIN_ID!r}].hooks is "
                f"{type(hooks_field).__name__}, not an object",
            )
        hooks: dict[str, Any] = entry.setdefault("hooks", {})

        if hooks.get("allowConversationAccess") is True:
            return  # idempotent — flag already set
        hooks["allowConversationAccess"] = True
        write_json(self.settings_path, cfg)

    def _unregister(self) -> None:
        """Inverse: drop the ``llmoji-kaomoji`` entry from
        ``plugins.entries``. No-op when the config file is missing,
        absent the entry, or unparseable (refusing to mutate a
        corrupt config matches the JSON-settings policy in the base
        class).

        We drop the whole sub-mapping rather than just toggling the
        flag back to ``false`` so a clean uninstall leaves no
        ``llmoji-kaomoji`` mention in the user's config — same
        contract as the bash providers' uninstall.
        """
        if not self.settings_path.exists():
            return
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return

        plugins = cfg.get("plugins")
        if not isinstance(plugins, dict):
            return
        entries = plugins.get("entries")
        if not isinstance(entries, dict):
            return
        if PLUGIN_ID not in entries:
            return
        entries.pop(PLUGIN_ID, None)

        # Tidy up empty containers so the config doesn't carry a
        # ``"entries": {}`` / ``"plugins": {}`` shape after removal.
        if not entries:
            plugins.pop("entries", None)
        if not plugins:
            cfg.pop("plugins", None)
        write_json(self.settings_path, cfg)

    def _check_registrations(self) -> tuple[bool, bool]:
        """Both files present on disk AND the conversation-access flag
        set in ``config.json`` = registered. Returns ``False`` on a
        corrupt or missing config — the install would have created
        the entry, so absence at status-check time is a real gap."""
        # File-presence check from the base class.
        files_ok, _ = super()._check_registrations()
        if not files_ok:
            return False, False
        if not self.settings_path.exists():
            return False, False
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return False, False
        plugins = cfg.get("plugins")
        if not isinstance(plugins, dict):
            return False, False
        entries = plugins.get("entries")
        if not isinstance(entries, dict):
            return False, False
        entry = entries.get(PLUGIN_ID)
        if not isinstance(entry, dict):
            return False, False
        hooks = entry.get("hooks")
        if not isinstance(hooks, dict):
            return False, False
        return hooks.get("allowConversationAccess") is True, False

    def _check_settings_health(self) -> str | None:
        """``None`` on parseable / absent config, else the why-string
        from :class:`SettingsCorruptError`. Mirrors the bash JSON
        providers — :func:`_load_json_strict` is the same gauntlet."""
        if not self.settings_path.exists():
            return None
        try:
            _load_json_strict(self.settings_path)
        except SettingsCorruptError as e:
            return e.why
        return None
