"""Antigravity, Google's ``agy`` CLI (``~/.gemini``).

agy loads plugins from ``~/.gemini/config/plugins/<name>/``: a
``plugin.json`` marker plus a ``hooks.json`` keyed by hook name, each
mapping events to a flat handler list. llmoji owns the ``llmoji`` plugin
outright and removes it on uninstall. Contract source: the
``agy-customizations`` skill agy bundles (agy 1.4.7), cross-checked
against live payloads.

- Every Stop payload carries ``modelName``, ``transcriptPath`` (the
  step list ``brain/<id>/.system_generated/logs/transcript_full.jsonl``),
  ``workspacePaths`` and, since 1.4.7, ``lastUserInput`` — the prompt
  without the ``<USER_REQUEST>`` wrapper the transcript stores it in.
- ``PLANNER_RESPONSE`` steps with text are model messages and
  ``USER_INPUT`` opens a turn. The transcript is complete when Stop
  fires.
- Hooks answer on stdout, so the shim prints ``{}`` (no ``decision``
  lets the agent stop) before Python runs.
- No replay: transcripts carry no per-step model id, and every bundle
  row is keyed by model.
- The journal sits in agy's state dir, the one place a sandbox that
  confines agy to its own state (gaslamp's Seatbelt profile does)
  still lets the hook write.
- The kaomoji block goes in ``~/.gemini/GEMINI.md``; agy's policy
  checker guards ``~/.gemini/config/`` itself against agent edits.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .._util import read_jsonl, write_json
from ..scrape import ScrapeRow, kaomoji_row
from .base import Provider, SettingsCorruptError, home_path, load_json_strict

PLUGIN = "llmoji"
_USER_REQUEST = re.compile(r"<USER_REQUEST>\n?([\s\S]*?)\n?</USER_REQUEST>")


class AntigravityProvider(Provider):
    name = "antigravity"
    home_dir = home_path(".gemini")
    hooks_dir = home_path(".gemini", "config", "plugins", PLUGIN)
    settings_path = home_path(".gemini", "config", "plugins", PLUGIN, "hooks.json")
    journal_path = home_path(".gemini", "antigravity-cli", "kaomoji-journal.jsonl")
    doc_path = home_path(".gemini", "GEMINI.md")
    hook_stdout = "{}"

    @property
    def manifest_path(self) -> Path:
        return self.hooks_dir / "plugin.json"

    def is_present(self) -> bool:
        # ~/.gemini alone is also Gemini CLI's home.
        return (self.home_dir / "antigravity-cli").exists() or (
            self.home_dir / "config"
        ).exists()

    def live_rows(self, payload: dict[str, Any]) -> list[ScrapeRow]:
        path = Path(str(payload.get("transcriptPath") or ""))
        steps = read_jsonl(path) if path.is_file() else []
        start = max(
            (i for i, s in enumerate(steps) if s.get("type") == "USER_INPUT"),
            default=0,
        )
        user = str(payload.get("lastUserInput") or "")
        if not user and steps and isinstance(steps[start].get("content"), str):
            content = steps[start]["content"]
            m = _USER_REQUEST.search(content)
            user = m.group(1) if m else content
        workspaces = payload.get("workspacePaths")
        rows = []
        for step in steps[start:]:
            content = step.get("content")
            if step.get("type") != "PLANNER_RESPONSE" or not isinstance(content, str):
                continue
            row = kaomoji_row(
                content,
                source=f"{self.name}-hook",
                model=str(payload.get("modelName") or ""),
                timestamp=str(step.get("created_at") or ""),
                cwd=str(workspaces[0]) if isinstance(workspaces, list) and workspaces else "",
                user=user,
            )
            if row is not None:
                rows.append(row)
        return rows

    # --- registration: our own plugin.json + hooks.json ---

    def _spec(self) -> dict[str, Any]:
        return {self.hook_event: [{"type": "command", "command": str(self.hook_path)}]}

    def _register(self) -> None:
        cfg = load_json_strict(self.settings_path)
        if cfg.get(PLUGIN) != self._spec():
            cfg[PLUGIN] = self._spec()
            write_json(self.settings_path, cfg)
        if not self.manifest_path.exists():
            write_json(self.manifest_path, {"name": PLUGIN})

    def _unregister(self) -> None:
        try:
            cfg = load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return
        if cfg.pop(PLUGIN, None) is not None:
            if cfg:
                write_json(self.settings_path, cfg)
            else:
                self.settings_path.unlink()
        if not self.settings_path.exists():
            self.manifest_path.unlink(missing_ok=True)

    def _is_registered(self) -> bool:
        try:
            spec = load_json_strict(self.settings_path).get(PLUGIN)
        except SettingsCorruptError:
            return False
        handlers = spec.get(self.hook_event) if isinstance(spec, dict) else None
        return self.manifest_path.exists() and isinstance(handlers, list) and any(
            self._is_ours(h) for h in handlers
        )

    def uninstall(self) -> None:
        super().uninstall()
        # The plugin dir, then plugins/ if nothing else lives there.
        for d in (self.hooks_dir, self.hooks_dir.parent):
            try:
                d.rmdir()
            except OSError:
                break
