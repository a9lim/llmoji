"""Filesystem layout the package writes to.

Single source of truth for every path the CLI touches outside of
provider-specific harness directories. Tests can override the home
root by setting ``LLMOJI_HOME`` in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def llmoji_home() -> Path:
    """Resolve the package's home directory.

    Override with ``$LLMOJI_HOME`` for tests / non-default installs.
    Defaults to ``~/.llmoji``.
    """
    override = os.environ.get("LLMOJI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".llmoji"


def cache_dir() -> Path:
    return llmoji_home() / "cache"


def cache_per_instance_path() -> Path:
    return cache_dir() / "per_instance.jsonl"


def bundle_dir() -> Path:
    return llmoji_home() / "bundle"


def journals_dir() -> Path:
    """Generic JSONL-append landing zone for harnesses we don't ship
    a first-class provider for. End-users on OpenClaw and similar
    write directly here against the documented schema."""
    return llmoji_home() / "journals"


def state_path() -> Path:
    """Tracks which providers are installed. Touched by
    ``install``/``uninstall``."""
    return llmoji_home() / "state.json"


def ensure_home() -> Path:
    """Create the home dir + first-run ``.gitignore`` (so users who
    rsync their home into a dotfiles repo don't accidentally publish
    the cache or local journals).

    Idempotent.
    """
    home = llmoji_home()
    home.mkdir(parents=True, exist_ok=True)
    gi = home / ".gitignore"
    if not gi.exists():
        gi.write_text(
            "# llmoji cache + bundle hold per-turn paraphrases and\n"
            "# scrape outputs derived from your local conversation\n"
            "# history. Don't sync them to a public dotfiles repo.\n"
            "*\n"
            "!.gitignore\n"
        )
    return home
