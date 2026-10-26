"""Install, uninstall and status, per provider and through the CLI.

Every path a provider touches derives from ``HOME``, which conftest
points at a temp directory for every test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from llmoji import cli
from llmoji._util import atomic_write_text
from llmoji.providers import PROVIDERS, SettingsCorruptError, get_provider
from llmoji.providers.base import DOC_BLOCK, _merge_doc_block, _strip_doc_block, same_command
from llmoji.synth_prompts import LEGACY_NUDGE_MESSAGES

PROSE = "# My rules\n\nBe kind.\n"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_every_provider_path_lives_under_home(sandbox_home: Path) -> None:
    """A path default evaluated at import would point at the developer's
    real home whatever the sandbox did."""
    for name in PROVIDERS:
        p = get_provider(name)
        for path in (p.home_dir, p.hooks_dir, p.settings_path, p.journal_path, p.doc_path, p.hook_path):
            assert path.is_relative_to(sandbox_home), (name, path)


@pytest.mark.parametrize("name", list(PROVIDERS))
def test_install_roundtrip(name: str) -> None:
    p = get_provider(name)
    p.doc_path.parent.mkdir(parents=True, exist_ok=True)
    p.doc_path.write_text(PROSE)
    p.install()
    p.install()  # idempotent
    s = p.status()
    assert s.healthy, s
    assert p.doc_path.read_text() == PROSE + "\n" + DOC_BLOCK + "\n"
    assert os.access(p.hook_path, os.X_OK)
    assert p.hook_path.read_text().count(f"-m llmoji.hook {name}") == 1

    p.uninstall()
    p.uninstall()
    s = p.status()
    assert not s.installed and not s.doc_current
    assert p.doc_path.read_text() == PROSE
    assert not p.hook_path.exists()


def test_json_registration_shape_and_neighbours(sandbox_home: Path) -> None:
    p = get_provider("claude_code")
    other = {"type": "command", "command": "/usr/local/bin/other.sh"}
    p.settings_path.parent.mkdir(parents=True)
    p.settings_path.write_text(json.dumps({"model": "x", "hooks": {"Stop": [{"hooks": [other]}]}}))
    p.install()
    stop = _json(p.settings_path)["hooks"]["Stop"]
    assert stop == [{"hooks": [other]}, {"hooks": [{"type": "command", "command": str(p.hook_path)}]}]
    p.uninstall()
    assert _json(p.settings_path) == {"model": "x", "hooks": {"Stop": [{"hooks": [other]}]}}


def test_uninstall_drops_an_emptied_hooks_object() -> None:
    p = get_provider("codex")
    p.install()
    p.uninstall()
    assert _json(p.settings_path) == {}


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"hooks": []}', '{"hooks": {"Stop": {}}}'])
def test_corrupt_settings_are_refused_before_anything_is_written(content: str) -> None:
    p = get_provider("claude_code")
    p.settings_path.parent.mkdir(parents=True)
    p.settings_path.write_text(content)
    with pytest.raises(SettingsCorruptError):
        p.install()
    assert p.settings_path.read_text() == content
    assert not p.hook_path.exists() and not p.doc_path.exists()
    if content.startswith(("{not", "[")):
        assert p.status().settings_error is not None


def test_a_home_relative_registration_is_recognised(sandbox_home: Path) -> None:
    """Dotfiles repos spell the registration ``$HOME/...`` so one file
    serves every machine; reading it as absent would append a duplicate
    and journal every message twice."""
    p = get_provider("claude_code")
    portable = "$HOME/.claude/hooks/kaomoji-log.sh"
    p.settings_path.parent.mkdir(parents=True)
    p.settings_path.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": portable},
    ]}]}}))
    p.install()
    assert _json(p.settings_path)["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": portable}]}]
    assert p.status().installed
    p.uninstall()
    assert _json(p.settings_path) == {}


def test_same_command() -> None:
    assert same_command("~/x/../hooks/a.sh", os.path.expanduser("~/hooks/a.sh"))
    assert not same_command("/a/hook.sh", "/b/hook.sh")
    assert not same_command("$LLMOJI_UNSET_VAR/hook.sh", "/anything/hook.sh")


def test_writes_follow_a_symlinked_settings_file(tmp_path: Path) -> None:
    """Harness config kept in a dotfiles repo reaches ``~/.claude`` as a
    link; replacing the link would detach the live file from the repo."""
    p = get_provider("claude_code")
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text("{}")
    p.settings_path.parent.mkdir(parents=True)
    p.settings_path.symlink_to(real)
    p.install()
    assert p.settings_path.is_symlink()
    assert "Stop" in _json(real)["hooks"]
    assert not list(real.parent.glob("*.llmoji-tmp"))


def test_atomic_write_creates_a_plain_file(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "a" / "f.txt", "x")
    assert (tmp_path / "a" / "f.txt").read_text() == "x"


def test_status_flags_a_hook_written_for_another_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    p = get_provider("codex")
    p.install()
    monkeypatch.setattr(sys, "executable", "/some/other/python")
    s = p.status()
    assert s.installed and not s.hook_current and not s.healthy


def test_grok_owns_its_hook_file(sandbox_home: Path) -> None:
    p = get_provider("grok")
    p.install()
    assert p.settings_path == sandbox_home / ".grok" / "hooks" / "llmoji.json"
    p.uninstall()
    assert not p.settings_path.exists()


def test_grok_uninstall_keeps_a_hook_file_the_user_added_to() -> None:
    p = get_provider("grok")
    p.install()
    cfg = _json(p.settings_path)
    cfg["hooks"]["PreToolUse"] = [{"hooks": [{"type": "command", "command": "/x.sh"}]}]
    p.settings_path.write_text(json.dumps(cfg))
    p.uninstall()
    assert list(_json(p.settings_path)["hooks"]) == ["PreToolUse"]


def test_antigravity_owns_a_plugin_and_keeps_other_hook_names(sandbox_home: Path) -> None:
    p = get_provider("antigravity")
    p.install()
    plugin = sandbox_home / ".gemini" / "config" / "plugins" / "llmoji"
    assert _json(plugin / "plugin.json") == {"name": "llmoji"}
    assert _json(p.settings_path) == {"llmoji": {"Stop": [{"type": "command", "command": str(p.hook_path)}]}}
    cfg = _json(p.settings_path)
    cfg["mine"] = {"Stop": []}
    p.settings_path.write_text(json.dumps(cfg))
    p.uninstall()
    assert _json(p.settings_path) == {"mine": {"Stop": []}}
    p.settings_path.unlink()
    p.install()
    p.uninstall()
    assert not plugin.exists()


def test_antigravity_needs_its_own_directory_to_be_detected(sandbox_home: Path) -> None:
    p = get_provider("antigravity")
    (sandbox_home / ".gemini").mkdir()
    assert not p.is_present()  # Gemini CLI's home alone
    (sandbox_home / ".gemini" / "antigravity-cli").mkdir()
    assert p.is_present()


# --- the kaomoji block ---


def test_merge_doc_block() -> None:
    assert _merge_doc_block("") == DOC_BLOCK + "\n"
    assert _merge_doc_block("x") == "x\n\n" + DOC_BLOCK + "\n"
    assert _merge_doc_block("x\n\n") == "x\n\n" + DOC_BLOCK + "\n"
    once = _merge_doc_block(PROSE)
    assert _merge_doc_block(once) == once


@pytest.mark.parametrize("old", [
    f"## Kaomoji\n\n{LEGACY_NUDGE_MESSAGES[0]}",
    f"## Kaomoji\n\n{LEGACY_NUDGE_MESSAGES[1]}",
    f"# Kaomoji\n\n{LEGACY_NUDGE_MESSAGES[0]}",
])
def test_earlier_releases_blocks_are_replaced_and_stripped(old: str) -> None:
    doc = PROSE + "\n" + old + "\n"
    assert _merge_doc_block(doc) == PROSE + "\n" + DOC_BLOCK + "\n"
    assert _strip_doc_block(doc) == PROSE


def test_a_hand_edited_block_is_left_alone() -> None:
    doc = PROSE + "\n## Kaomoji\n\nMy own words.\n"
    assert _strip_doc_block(doc) == doc


def test_uninstall_removes_a_doc_that_held_only_the_block() -> None:
    p = get_provider("claude_code")
    p.install()
    p.uninstall()
    assert not p.doc_path.exists()


# --- CLI ---


def test_cli_install_without_a_harness_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["install", "--yes"]) == 2
    assert "no harnesses detected" in capsys.readouterr().err


def test_cli_installs_every_detected_harness(sandbox_home: Path) -> None:
    (sandbox_home / ".claude").mkdir()
    (sandbox_home / ".grok").mkdir()
    assert cli.main(["install", "--yes"]) == 0
    assert get_provider("claude_code").status().healthy and get_provider("grok").status().healthy
    assert not get_provider("codex").status().installed
    assert cli.main(["uninstall", "--yes"]) == 0
    assert not get_provider("grok").status().installed


def test_cli_prompt_declined_installs_nothing(sandbox_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (sandbox_home / ".claude").mkdir()
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert cli.main(["install"]) == 2
    assert not get_provider("claude_code").hook_path.exists()


def test_cli_one_failure_doesnt_stop_the_rest(sandbox_home: Path) -> None:
    (sandbox_home / ".claude").mkdir()
    (sandbox_home / ".codex").mkdir()
    (sandbox_home / ".claude" / "settings.json").write_text("{broken")
    assert cli.main(["install", "--yes"]) == 1
    assert get_provider("codex").status().healthy


def test_cli_status_json(capsys: pytest.CaptureFixture[str]) -> None:
    get_provider("codex").install()
    assert cli.main(["status", "--json", "--stats"]) == 0
    out = json.loads(capsys.readouterr().out)
    by_name = {p["name"]: p for p in out["providers"]}
    assert list(by_name) == list(PROVIDERS)
    assert by_name["codex"]["healthy"] and not by_name["claude_code"]["installed"]
    assert out["stats"]["rows_total"] == 0


def test_cli_status_stats_counts_rows_and_faces(sandbox_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    journal = get_provider("claude_code").journal_path
    journal.parent.mkdir(parents=True)
    row = {"ts": "t", "model": "m", "cwd": "", "kaomoji": "(◕‿◕)", "user_text": "", "assistant_text": "x"}
    journal.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\nnot json\n" + json.dumps({**row, "ts": 1}) + "\n")
    assert cli.main(["status", "--json", "--stats"]) == 0
    stats = json.loads(capsys.readouterr().out)["stats"]
    assert (stats["rows_total"], stats["rows_malformed"]) == (4, 2)
    assert stats["by_canonical"] == {"(◕‿◕)": 3}
