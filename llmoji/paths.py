"""Where llmoji keeps its own state: ``$LLMOJI_HOME``, default
``~/.llmoji``. Harness-side paths live on each provider."""

from __future__ import annotations

import os
from pathlib import Path


def llmoji_home() -> Path:
    override = os.environ.get("LLMOJI_HOME")
    return Path(override).expanduser() if override else Path.home() / ".llmoji"


def cache_dir() -> Path:
    return llmoji_home() / "cache"


def cache_path() -> Path:
    """The per-cell synthesis cache (``llmoji.synth``)."""
    return cache_dir() / "per_cell.jsonl"


def bundle_dir() -> Path:
    return llmoji_home() / "bundle"


def journals_dir() -> Path:
    """Journals llmoji didn't capture itself: parsed exports, and any
    harness writing the 6-field schema directly."""
    return llmoji_home() / "journals"


def salt_path() -> Path:
    """Per-machine random salt behind the submitter id. Never leaves the
    machine."""
    return llmoji_home() / ".salt"


def hook_errors_path() -> Path:
    return llmoji_home() / "hook-errors.log"


def ensure_home() -> Path:
    """Create the home dir with a ``.gitignore`` that keeps the cache
    and journals out of a dotfiles repository synced from ``$HOME``."""
    home = llmoji_home()
    home.mkdir(parents=True, exist_ok=True)
    gi = home / ".gitignore"
    if not gi.exists():
        gi.write_text(
            "# llmoji's cache, bundle and journals derive from your\n"
            "# conversation history. Don't sync them to a public repo.\n"
            "*\n"
            "!.gitignore\n"
        )
    return home
