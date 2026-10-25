"""Install / uninstall / status for the providers 2.2 added or moved:
grok, antigravity, and openclaw's corrected paths — plus the guard
that keeps every provider's default paths inside the test sandbox.

These tests use each provider's *default* paths on purpose: since
2.2 they resolve against ``Path.home()`` when read, so under the
suite's ``sandbox_home`` fixture they land in a temp directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmoji.providers import (
    PROVIDERS,
    AntigravityProvider,
    GrokProvider,
    OpenclawProvider,
)
from llmoji.providers.base import SOFT_DOC_HEADING, HardInstallNotSupportedError


def test_every_default_path_follows_the_sandbox_home(sandbox_home: Path) -> None:
    """A provider path computed at import time would point at the
    developer's real home no matter what the fixture does — the
    2.1.2 guard missed exactly that. Every absolute path a provider
    exposes must sit under the sandbox."""
    for name, cls in PROVIDERS.items():
        p = cls()
        for attr in dir(p):
            if attr.startswith("__"):
                continue
            try:
                value = getattr(p, attr)
            except Exception:  # noqa: BLE001 — a property that needs install state
                continue
            if isinstance(value, Path) and value.is_absolute():
                assert value.is_relative_to(sandbox_home), (
                    f"{name}.{attr} = {value} escapes the sandbox home"
                )


# ---------------------------------------------------------------------------
# grok
# ---------------------------------------------------------------------------


def test_grok_soft_install_roundtrip(sandbox_home: Path) -> None:
    p = GrokProvider()
    assert not p.is_present()
    (sandbox_home / ".grok").mkdir()
    assert p.is_present()

    p.install_soft()
    hook_file = sandbox_home / ".grok" / "hooks" / "llmoji.json"
    cfg = json.loads(hook_file.read_text())
    assert cfg == {"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": str(p.hook_path)},
    ]}]}}
    assert p.hook_path.exists()
    doc = (sandbox_home / ".grok" / "AGENTS.md").read_text()
    assert doc.startswith(SOFT_DOC_HEADING)
    s = p.status()
    assert s.installed and s.soft_installed and s.main_hook_current

    p.install_soft()  # idempotent
    assert json.loads(hook_file.read_text()) == cfg

    p.uninstall()
    assert not hook_file.exists(), "an emptied hook file is deleted"
    assert not p.hook_path.exists()
    assert not (sandbox_home / ".grok" / "AGENTS.md").exists()
    assert not p.status().installed


def test_grok_hard_install_refuses_without_touching_disk(sandbox_home: Path) -> None:
    p = GrokProvider()
    with pytest.raises(HardInstallNotSupportedError, match="--soft"):
        p.install_hard()
    assert not (sandbox_home / ".grok").exists()


def test_grok_uninstall_keeps_a_hook_file_the_user_added_to(sandbox_home: Path) -> None:
    p = GrokProvider()
    p.install_soft()
    cfg = json.loads(p.settings_path.read_text())
    cfg["hooks"]["SessionStart"] = [{"hooks": [{"type": "command", "command": "echo hi"}]}]
    p.settings_path.write_text(json.dumps(cfg))
    p.uninstall()
    assert json.loads(p.settings_path.read_text()) == {"hooks": {
        "SessionStart": [{"hooks": [{"type": "command", "command": "echo hi"}]}],
    }}


# ---------------------------------------------------------------------------
# antigravity
# ---------------------------------------------------------------------------


def _agy_plugin(home: Path) -> Path:
    return home / ".gemini" / "config" / "plugins" / "llmoji"


def test_antigravity_is_present_needs_an_antigravity_dir(sandbox_home: Path) -> None:
    p = AntigravityProvider()
    (sandbox_home / ".gemini").mkdir()  # Gemini CLI alone
    assert not p.is_present()
    (sandbox_home / ".gemini" / "antigravity-cli").mkdir()
    assert p.is_present()


def test_antigravity_hard_install_roundtrip(sandbox_home: Path) -> None:
    p = AntigravityProvider()
    p.install_hard()
    plugin = _agy_plugin(sandbox_home)
    assert json.loads((plugin / "plugin.json").read_text()) == {"name": "llmoji"}
    assert json.loads((plugin / "hooks.json").read_text()) == {"llmoji": {
        "Stop": [{"type": "command", "command": str(plugin / "kaomoji-log.sh")}],
        "PreInvocation": [{"type": "command", "command": str(plugin / "kaomoji-nudge.sh")}],
    }}
    s = p.status()
    assert s.installed and s.nudge_installed and s.main_hook_current

    p.uninstall()
    assert not plugin.exists(), "the plugin directory is ours and goes entirely"
    assert not plugin.parent.exists(), "so does the plugins/ dir install created"
    assert (sandbox_home / ".gemini" / "config").exists()
    assert not p.status().installed


def test_antigravity_soft_after_hard_drops_the_nudge(sandbox_home: Path) -> None:
    p = AntigravityProvider()
    p.install_hard()
    p.install_soft()
    hooks = json.loads((_agy_plugin(sandbox_home) / "hooks.json").read_text())
    assert set(hooks["llmoji"]) == {"Stop"}
    assert (sandbox_home / ".gemini" / "GEMINI.md").read_text().startswith(SOFT_DOC_HEADING)
    s = p.status()
    assert s.soft_installed and s.main_installed and not s.nudge_installed


def test_antigravity_keeps_other_hook_names(sandbox_home: Path) -> None:
    p = AntigravityProvider()
    p.install_hard()
    hooks_file = _agy_plugin(sandbox_home) / "hooks.json"
    hooks = json.loads(hooks_file.read_text())
    hooks["mine"] = {"Stop": [{"type": "command", "command": "true"}]}
    hooks_file.write_text(json.dumps(hooks))

    p.install_hard()
    assert "mine" in json.loads(hooks_file.read_text())
    p.uninstall()
    assert json.loads(hooks_file.read_text()) == {
        "mine": {"Stop": [{"type": "command", "command": "true"}]},
    }
    # The user's hook still needs the plugin marker to load.
    assert (_agy_plugin(sandbox_home) / "plugin.json").exists()


# ---------------------------------------------------------------------------
# openclaw — corrected paths + cleanup of what < 2.2 left behind
# ---------------------------------------------------------------------------


def test_openclaw_installs_where_openclaw_looks(sandbox_home: Path) -> None:
    p = OpenclawProvider()
    p.install_hard()
    ext = sandbox_home / ".openclaw" / "extensions" / "llmoji-kaomoji"
    assert (ext / "index.ts").exists()
    assert (ext / "openclaw.plugin.json").exists()
    cfg = json.loads((sandbox_home / ".openclaw" / "openclaw.json").read_text())
    assert cfg["plugins"]["entries"]["llmoji-kaomoji"]["hooks"] == {
        "allowConversationAccess": True,
    }
    assert p.status().installed


def test_openclaw_install_clears_the_legacy_install(sandbox_home: Path) -> None:
    oc = sandbox_home / ".openclaw"
    legacy_dir = oc / "plugins" / "llmoji-kaomoji"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "index.ts").write_text("// old")
    (legacy_dir / "openclaw.plugin.json").write_text("{}")
    (oc / "config.json").write_text(json.dumps({"plugins": {"entries": {
        "llmoji-kaomoji": {"hooks": {"allowConversationAccess": True}},
    }}}))

    OpenclawProvider().install_soft()
    assert not (oc / "plugins").exists()
    assert not (oc / "config.json").exists(), "nothing else was in it"


def test_openclaw_legacy_cleanup_leaves_foreign_content(sandbox_home: Path) -> None:
    oc = sandbox_home / ".openclaw"
    legacy_dir = oc / "plugins" / "llmoji-kaomoji"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "index.ts").write_text("// old")
    (legacy_dir / "notes.txt").write_text("mine")
    (oc / "config.json").write_text(json.dumps({"theme": "dark", "plugins": {"entries": {
        "llmoji-kaomoji": {"hooks": {"allowConversationAccess": True}},
    }}}))

    OpenclawProvider().uninstall()
    assert not (legacy_dir / "index.ts").exists()
    assert (legacy_dir / "notes.txt").read_text() == "mine"
    assert json.loads((oc / "config.json").read_text()) == {"theme": "dark"}
