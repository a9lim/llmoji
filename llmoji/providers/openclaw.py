"""OpenClaw provider — TS plugin bundle at
``~/.openclaw/extensions/llmoji-kaomoji/`` plus an ``openclaw.json``
flag flip to grant conversation-hook access.

OpenClaw (https://openclaw.ai) ships a TypeScript plugin SDK with
``definePluginEntry`` + ``api.on(hookName, handler)``. ``llm_input``
and ``llm_output`` are *conversation hooks* — gated behind a
per-plugin opt-in flag (``plugins.entries.<plugin_id>.hooks.allow
ConversationAccess``). Without the flag set, OpenClaw refuses to
fire those hooks for the plugin and the journal stays empty.

Install lifecycle:

  1. Render ``index.ts`` + ``openclaw.plugin.json`` from package
     templates into ``~/.openclaw/extensions/llmoji-kaomoji/`` — the
     global plugin root OpenClaw scans (``resolveDefaultPluginExtensionsDir``
     in ``src/plugins/install-paths.ts``).
  2. Read ``~/.openclaw/openclaw.json`` (``CONFIG_FILENAME`` in
     ``src/config/paths.ts``), set
     ``plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess
     = true``, atomic-write back.

Before 2.2 the bundle went to ``~/.openclaw/plugins/llmoji-kaomoji/``
and the flag to ``~/.openclaw/config.json``. OpenClaw reads neither —
both paths were wrong from the provider's first release (1.3.0), and
OpenClaw v2026.3.1 already used the current ones — so no earlier
install ever captured anything. Install and uninstall now delete what
those versions left at the old paths.

Uninstall is the inverse — remove the plugin dir, unset the
``llmoji-kaomoji`` entry under ``plugins.entries`` (drop the whole
sub-mapping rather than just toggling the flag, so the user's
``plugins.entries`` mapping is byte-stable when llmoji is the only
configured plugin).

Per-provider notes (vs claude_code / codex / hermes / opencode):

  - **No bash hook.** Same as opencode — ``hook_path`` points at
    ``index.ts``.
  - **Settings file is JSON5 to OpenClaw, strict JSON to us.**
    OpenClaw parses ``openclaw.json`` as JSON5; a hand-edited file
    with comments or trailing commas is refused with
    :class:`SettingsCorruptError` rather than rewritten without them.
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
    home_path,
)

# OpenClaw plugin id used both as the plugin directory name and the
# key under ``plugins.entries`` in ``~/.openclaw/openclaw.json``.
# Matches the ``id`` field in the rendered ``openclaw.plugin.json``.
PLUGIN_ID = "llmoji-kaomoji"


class OpenclawProvider(PluginInstaller):
    name = "openclaw"
    # Plugin dir = ``~/.openclaw/extensions/llmoji-kaomoji/`` — OpenClaw's
    # ``plugins install <path>`` command copies the entire bundle
    # directory into this location, but ``llmoji install`` writes the
    # rendered files directly so the user doesn't need to invoke
    # OpenClaw's own CLI for setup.
    plugin_dir = home_path(".openclaw", "extensions", PLUGIN_ID)
    settings_path = home_path(".openclaw", "openclaw.json")
    journal_path = home_path(".llmoji", "journals", "openclaw.jsonl")
    # OpenClaw's persona / voice file — slot #1 of the assembled
    # system prompt per https://docs.openclaw.ai/concepts/system-prompt.
    # The workspace dir is configurable via ``agents.defaults.workspace``
    # but defaults to ``~/.openclaw/workspace/``. We target SOUL.md
    # (voice / identity) rather than AGENTS.md (procedure / SOP) for
    # the same reason as Hermes: the kaomoji-leading reminder is
    # voice-coded.
    system_prompt_doc_path = home_path(".openclaw", "workspace", "SOUL.md")
    plugin_files = [
        ("openclaw_index.ts.tmpl", "index.ts"),
        ("openclaw_plugin.json.tmpl", "openclaw.plugin.json"),
    ]
    # Where llmoji < 2.2 wrote the bundle and the flag. OpenClaw reads
    # neither path; install and uninstall clean them up.
    legacy_plugin_dir = home_path(".openclaw", "plugins", PLUGIN_ID)
    legacy_settings_path = home_path(".openclaw", "config.json")

    # --- presence detection ---

    @property
    def home_dir(self) -> Path:
        """OpenClaw's home dir is ``~/.openclaw/``. Match the bash
        providers' ``settings_path.parent`` rule rather than
        :class:`PluginInstaller`'s default ``plugin_dir.parent`` — the
        plugin_dir parent (``~/.openclaw/extensions/``) doesn't exist on
        a fresh OpenClaw install, so the default would false-negative.
        """
        return self.settings_path.parent

    # --- registration via plugins.entries.<id>.hooks.allowConversationAccess ---

    def _register(self, *, include_nudge: bool = True) -> None:
        """Atomic read-modify-write of ``~/.openclaw/openclaw.json`` to
        set ``plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess
        = true``. Idempotent — re-running is a no-op when the flag
        is already on. Refuses to mutate a corrupt config (loud
        :class:`SettingsCorruptError` beats silent overwrite).

        ``include_nudge`` is accepted for signature parity with the
        base class but doesn't change behavior here — the openclaw
        config flag is "may this plugin attach conversation hooks at
        all," which we need set under both soft and hard modes
        because both rely on the plugin's ``llm_output`` hook for
        journal capture.
        """
        del include_nudge
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

        if hooks.get("allowConversationAccess") is not True:
            hooks["allowConversationAccess"] = True
            write_json(self.settings_path, cfg)
        self._remove_legacy_install()

    def _unregister(self) -> None:
        """Inverse: drop the ``llmoji-kaomoji`` entry from
        ``plugins.entries`` (see :func:`_drop_plugin_entry`), then
        clear anything an older version left at the legacy paths."""
        _drop_plugin_entry(self.settings_path)
        self._remove_legacy_install()

    def _remove_legacy_install(self) -> None:
        """Delete what llmoji < 2.2 left at the paths OpenClaw never
        read: the two bundle files under ``~/.openclaw/plugins/
        llmoji-kaomoji/`` (and the directories, once empty) and our
        entry in ``~/.openclaw/config.json`` (and the file, once
        nothing else is in it). Anything else found there is left
        alone."""
        for _, dest_filename in self.plugin_files:
            stale = self.legacy_plugin_dir / dest_filename
            if stale.exists():
                stale.unlink()
        for d in (self.legacy_plugin_dir, self.legacy_plugin_dir.parent):
            try:
                d.rmdir()
            except OSError:
                pass
        if _drop_plugin_entry(self.legacy_settings_path) == {}:
            self.legacy_settings_path.unlink()

    def _check_registrations(self) -> tuple[bool, bool]:
        """Both files present on disk AND the conversation-access flag
        set in ``openclaw.json`` = registered. Returns ``False`` on a
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


def _drop_plugin_entry(path: Path) -> dict[str, Any] | None:
    """Remove the ``llmoji-kaomoji`` entry from ``plugins.entries`` in
    the JSON config at ``path``, tidying the containers it leaves
    empty. Returns the config as written, or ``None`` when nothing
    changed — the file is missing, unparseable (refusing to mutate a
    corrupt config matches the JSON-settings policy in the base
    class), or had no entry of ours.

    The whole sub-mapping goes rather than the flag flipping back to
    ``false``, so a clean uninstall leaves no ``llmoji-kaomoji``
    mention behind — same contract as the bash providers' uninstall.
    """
    if not path.exists():
        return None
    try:
        cfg = _load_json_strict(path)
    except SettingsCorruptError:
        return None
    plugins = cfg.get("plugins")
    if not isinstance(plugins, dict):
        return None
    entries = plugins.get("entries")
    if not isinstance(entries, dict) or PLUGIN_ID not in entries:
        return None
    entries.pop(PLUGIN_ID, None)
    if not entries:
        plugins.pop("entries", None)
    if not plugins:
        cfg.pop("plugins", None)
    write_json(path, cfg)
    return cfg
