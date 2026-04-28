"""Cross-cutting utilities used by analyze, upload, providers, cli.

These helpers were originally in ``llmoji.providers.base`` but they
have nothing to do with the provider abstraction — ``upload.py``
shouldn't have to import from ``providers.base`` to write a JSON file
atomically. Lifting them keeps the dependency graph tree-shaped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .scrape import ScrapeRow


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


def journal_line_dict(
    *,
    ts: str,
    model: str,
    cwd: str,
    kaomoji: str,
    user_text: str,
    assistant_text: str,
) -> dict[str, Any]:
    """Canonical 6-field on-disk journal row shape.

    Single source of truth for the JSONL row that the live bash hook
    templates (under :mod:`llmoji._hooks`) emit, that the
    static-export readers persist via the CLI parse path, and that
    the :mod:`llmoji.backfill` replays match. Every writer routes
    through this helper so a future schema change touches one
    function rather than four scattered ``yield {...}`` blocks.
    """
    return {
        "ts": ts,
        "model": model,
        "cwd": cwd,
        "kaomoji": kaomoji,
        "user_text": user_text,
        "assistant_text": assistant_text,
    }


def scrape_row_to_journal_line(row: ScrapeRow) -> dict[str, Any]:
    """Map a :class:`~llmoji.scrape.ScrapeRow` to the canonical
    6-field journal row dict.

    Empty / ``None`` fields normalize to ``""`` so the dict shape is
    stable regardless of which source produced the row (live hook,
    static export, backfill). Routes through :func:`journal_line_dict`
    for the schema definition.
    """
    return journal_line_dict(
        ts=row.timestamp,
        model=row.model or "",
        cwd=row.cwd or "",
        kaomoji=row.first_word,
        user_text=row.surrounding_user,
        assistant_text=row.assistant_text,
    )


def iter_bundle_data_files(
    bundle_dir: Path,
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """Walk the bundle root, yielding ``(path, parsed_rows)`` for
    every per-source-model ``<slug>.jsonl`` file.

    Manifest.json is excluded — it's bundle metadata, not per-cell
    data. Callers that need it read it separately. Used by
    :func:`llmoji.analyze._print_preview` and
    ``examples/inspect_bundle.py`` so the read-and-parse loop lives
    in one place.

    Files are sorted by name for deterministic preview ordering.
    Empty rows skip; malformed JSON raises (these files are
    written by ``analyze`` and should always parse).
    """
    paths = sorted(
        p for p in bundle_dir.iterdir()
        if p.is_file() and p.suffix == ".jsonl"
    )
    for path in paths:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        yield path, rows


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
