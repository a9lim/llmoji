"""Cross-cutting utilities used by analyze, upload, providers, cli.

These helpers were originally in ``llmoji.providers.base`` but they
have nothing to do with the provider abstraction — ``upload.py``
shouldn't have to import from ``providers.base`` to write a JSON file
atomically. Lifting them keeps the dependency graph tree-shaped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via tmp-file + rename.

    ``Path.write_text`` truncates and writes in place; an interrupt
    between truncate and final flush leaves the user's settings file
    in a partially-written state. ``os.replace`` (the underlying
    ``Path.replace`` call) is POSIX-atomic on the same filesystem,
    so the file either has the old content or the new — never half.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".llmoji-tmp")
    tmp.write_text(content)
    tmp.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def package_version() -> str:
    """Resolve the installed package version, with a fallback for
    development checkouts where ``importlib.metadata`` may not see the
    package yet."""
    try:
        from importlib.metadata import version
        return version("llmoji")
    except Exception:
        return "0.0.0+dev"


def human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (B / KB / MB / GB / TB)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def sanitize_model_id_for_path(model_id: str) -> str:
    """Subfolder-safe encoding of a model id for the per-source-model
    bundle layout.

    Rules: lowercase, ``/`` → ``__``, ``:`` → ``-``, dots and digits
    preserved. Empty / falsy input → ``"unknown"`` (defensive — keeps
    rows whose ``ScrapeRow.model`` is empty from collapsing into an
    unnamed top-level path).

    Lives here rather than in ``llmoji.synth`` so ``upload.py``'s
    allowlist walker can reuse it without dragging in the synthesis
    backend imports.
    """
    if not model_id:
        return "unknown"
    return model_id.lower().replace("/", "__").replace(":", "-")
