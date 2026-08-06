"""Suite-wide guard: no test may reach the real home directory.

``test_nudge_install_uninstall_roundtrip`` rebound three of a
provider's four filesystem attributes and then called ``uninstall()``.
The fourth, ``system_prompt_doc_path``, still pointed at the developer's
own ``~/.claude/CLAUDE.md`` — so the test stripped the soft-doc block
out of a real config file and passed. On a machine whose agent config
is symlinked into a dotfiles repository the edit lands in version
control, where it is a commit away from every other machine.

Rebinding attributes one at a time is the wrong altitude for that
class of bug: it is correct only for as long as nobody adds a fifth
path. ``HOME`` is what every provider default is derived from, so it
is the one place worth sandboxing. Point it somewhere disposable for
the duration of every test and a forgotten attribute writes into a
temp directory instead of somebody's configuration.

Tests that care about a specific home simply set it again — a later
``monkeypatch.setenv`` wins, and the sandbox is what they fall back to.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def sandbox_home(tmp_path_factory, monkeypatch) -> Path:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    # Path.home() consults USERPROFILE on Windows and HOME elsewhere;
    # set both so the guard does not depend on the platform it runs on.
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LLMOJI_HOME", str(home / ".llmoji"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home
