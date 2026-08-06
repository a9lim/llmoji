"""Tests for coexisting with dotfiles kept in a repository.

Both harness homes llmoji writes into — ``~/.claude`` and
``~/.codex`` — are commonly assembled out of symlinks pointing at a
versioned checkout, so that one machine's configuration is the same
object as another's. Two things went wrong there before 2.1.1:

  - every write went through ``os.replace`` onto the *link* path,
    detaching the live file from the repository it was versioned in.
    The install reported success; the other machine never saw it.

  - a registration written ``$HOME/.claude/hooks/kaomoji-log.sh`` —
    the portable spelling, and one both harnesses resolve because a
    ``type: command`` hook runs through a shell — did not compare
    equal to llmoji's own absolute path, so ``install`` appended a
    second entry beside it and ``status`` reported the provider
    uninstalled forever.

Neither is exotic: it is what happens the first time someone keeps
their agent config in git.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# atomic_write_text — follows symlinks instead of replacing them
# ---------------------------------------------------------------------------


def test_atomic_write_follows_symlink_to_real_file(tmp_path: Path):
    from llmoji._util import atomic_write_text

    real = tmp_path / "repo" / "settings.json"
    real.parent.mkdir()
    real.write_text("old\n")
    link = tmp_path / "home" / "settings.json"
    link.parent.mkdir()
    link.symlink_to(real)

    atomic_write_text(link, "new\n")

    assert link.is_symlink(), "the link itself must survive the write"
    assert real.read_text() == "new\n", "content lands on the real file"


def test_atomic_write_follows_symlinked_parent_directory(tmp_path: Path):
    """The hooks *directory* is the usual link, not the file."""
    from llmoji._util import atomic_write_text

    real_dir = tmp_path / "repo" / "hooks"
    real_dir.mkdir(parents=True)
    link_dir = tmp_path / "home" / "hooks"
    link_dir.parent.mkdir()
    link_dir.symlink_to(real_dir)

    atomic_write_text(link_dir / "kaomoji-log.sh", "#!/bin/sh\n")

    assert link_dir.is_symlink()
    assert (real_dir / "kaomoji-log.sh").read_text() == "#!/bin/sh\n"


def test_atomic_write_leaves_no_temp_file_beside_either_path(tmp_path: Path):
    from llmoji._util import atomic_write_text

    real = tmp_path / "repo" / "f.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)

    atomic_write_text(link, "{}\n")

    strays = [p.name for p in tmp_path.rglob("*.llmoji-tmp")]
    assert strays == []


def test_atomic_write_still_creates_a_plain_file(tmp_path: Path):
    """The non-symlink path must be untouched by the fix."""
    from llmoji._util import atomic_write_text

    path = tmp_path / "nested" / "plain.json"
    atomic_write_text(path, "hi\n")
    assert path.read_text() == "hi\n"
    assert not path.is_symlink()


# ---------------------------------------------------------------------------
# Hook-command comparison — $HOME and ~ are the same registration
# ---------------------------------------------------------------------------


def test_same_hook_command_expands_home_variable(monkeypatch):
    from llmoji.providers.base import _same_hook_command

    monkeypatch.setenv("HOME", "/home/someone")
    assert _same_hook_command(
        "$HOME/.claude/hooks/kaomoji-log.sh",
        "/home/someone/.claude/hooks/kaomoji-log.sh",
    )
    assert _same_hook_command(
        "~/.claude/hooks/kaomoji-log.sh",
        "/home/someone/.claude/hooks/kaomoji-log.sh",
    )


def test_same_hook_command_normalises_redundant_separators():
    from llmoji.providers.base import _same_hook_command

    assert _same_hook_command("/a//b/../b/hook.sh", "/a/b/hook.sh")


def test_same_hook_command_rejects_genuinely_different_paths():
    from llmoji.providers.base import _same_hook_command

    assert not _same_hook_command("/a/hook.sh", "/b/hook.sh")


def test_same_hook_command_unset_variable_does_not_collapse(monkeypatch):
    """An unresolvable variable stays literal, so it fails to match —
    the conservative direction: a spurious non-match costs a
    duplicate check, a spurious match would drop a real hook."""
    from llmoji.providers.base import _same_hook_command

    monkeypatch.delenv("LLMOJI_NOT_A_REAL_VAR", raising=False)
    assert not _same_hook_command(
        "$LLMOJI_NOT_A_REAL_VAR/hook.sh", "/anything/hook.sh",
    )


# ---------------------------------------------------------------------------
# End-to-end: a portable registration is recognised, not duplicated
# ---------------------------------------------------------------------------


def _claude_in_tmp(tmp_path: Path):
    from llmoji.providers import get_provider

    p = get_provider("claude_code")
    p.hooks_dir = tmp_path / "hooks"
    p.settings_path = tmp_path / "settings.json"
    p.journal_path = tmp_path / "journal.jsonl"
    p.system_prompt_doc_path = tmp_path / "CLAUDE.md"
    return p


def _stop_entries(settings: Path) -> list:
    import json

    return json.loads(settings.read_text())["hooks"]["Stop"]


def test_install_does_not_duplicate_a_home_relative_registration(tmp_path: Path):
    import json

    p = _claude_in_tmp(tmp_path)
    portable = "$HOME_FOR_TEST/hooks/kaomoji-log.sh"
    os.environ["HOME_FOR_TEST"] = str(tmp_path)
    try:
        p.settings_path.write_text(json.dumps({
            "hooks": {"Stop": [{"hooks": [
                {"type": "command", "command": portable},
            ]}]},
        }))
        p.install_soft()
        entries = _stop_entries(p.settings_path)
        assert len(entries) == 1, "the portable entry already registers the hook"
        cmd = entries[0]["hooks"][0]["command"]
        assert cmd == portable, "and it is left in its portable spelling"
    finally:
        del os.environ["HOME_FOR_TEST"]


def test_status_reports_a_home_relative_registration_as_installed(tmp_path: Path):
    import json

    p = _claude_in_tmp(tmp_path)
    os.environ["HOME_FOR_TEST"] = str(tmp_path)
    try:
        p.install_soft()
        # Rewrite the absolute registration llmoji just made into the
        # portable spelling, exactly as a dotfiles repo would hold it.
        cfg = json.loads(p.settings_path.read_text())
        cfg["hooks"]["Stop"] = [{"hooks": [
            {"type": "command",
             "command": "$HOME_FOR_TEST/hooks/kaomoji-log.sh"},
        ]}]
        p.settings_path.write_text(json.dumps(cfg))
        assert p.status().main_installed
    finally:
        del os.environ["HOME_FOR_TEST"]


def test_uninstall_removes_a_home_relative_registration(tmp_path: Path):
    import json

    p = _claude_in_tmp(tmp_path)
    os.environ["HOME_FOR_TEST"] = str(tmp_path)
    try:
        p.settings_path.write_text(json.dumps({
            "hooks": {"Stop": [{"hooks": [
                {"type": "command",
                 "command": "$HOME_FOR_TEST/hooks/kaomoji-log.sh"},
            ]}]},
        }))
        p.install_soft()
        p.uninstall()
        cfg = json.loads(p.settings_path.read_text())
        assert "Stop" not in cfg.get("hooks", {})
    finally:
        del os.environ["HOME_FOR_TEST"]
