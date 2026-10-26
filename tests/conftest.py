"""No test may reach the real home directory.

Every provider path derives from ``HOME``, so ``HOME`` is the one place
worth sandboxing: a forgotten path lands in a temp directory instead of
somebody's harness config, which on a9's machines is symlinked into a
dotfiles repository.

Two layers. The session layer sets ``HOME`` and ``LLMOJI_HOME`` in
``os.environ`` directly, so even a test that calls ``monkeypatch.undo()``
falls back to a sandbox, not the developer's home (one did, and
overwrote a real ``~/.llmoji/.salt``). The per-test layer gives each
test a fresh home on top of it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

_VARS = ("HOME", "USERPROFILE", "LLMOJI_HOME", "XDG_CONFIG_HOME")


@pytest.fixture(autouse=True, scope="session")
def _session_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in _VARS}
    home = tmp_path_factory.mktemp("session-home")
    os.environ.update(HOME=str(home), USERPROFILE=str(home), LLMOJI_HOME=str(home / ".llmoji"))
    os.environ.pop("XDG_CONFIG_HOME", None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def sandbox_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LLMOJI_HOME", str(home / ".llmoji"))
    return home
