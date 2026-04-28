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


def salt_path() -> Path:
    """Per-machine random salt, persisted at ``~/.llmoji/.salt``.

    Holds 64 hex characters (32 bytes of entropy) — the seed for
    :func:`llmoji.upload.submitter_id`. The hash is what lands on
    HF; the salt itself never leaves the user's machine.

    Pre Wave 6 the same byte was wrapped in a JSON envelope at
    ``state.json`` keyed under ``"submission_token"``; the envelope
    paid for nothing and a flat ``.salt`` is honest about what's
    on disk. The dotfile prefix keeps `ls ~/.llmoji` clean.

    HookInstaller install state is read live from each harness's
    own settings file — this isn't an install registry.
    """
    return llmoji_home() / ".salt"


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
