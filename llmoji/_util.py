"""Small helpers shared across modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .scrape import ScrapeRow


def atomic_write_text(path: Path, content: str) -> None:
    """Write via a temp file and ``os.replace``, so an interrupt leaves
    the old content or the new, never half.

    Symlinks are followed, not replaced: harness config kept in a
    dotfiles repository reaches ``~/.claude`` as links, and renaming
    onto the link would detach the live file from its repository.
    Resolving first also keeps the temp file on the target's filesystem.
    """
    target = Path(os.path.realpath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".llmoji-tmp")
    tmp.write_text(content)
    tmp.replace(target)


def write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def package_version() -> str:
    from . import __version__

    return __version__


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


JOURNAL_FIELDS = ("ts", "model", "cwd", "kaomoji", "user_text", "assistant_text")


def journal_line(row: ScrapeRow) -> dict[str, str]:
    """The 6-field on-disk journal row for ``row``."""
    return {
        "ts": row.timestamp,
        "model": row.model or "",
        "cwd": row.cwd or "",
        "kaomoji": row.first_word,
        "user_text": row.surrounding_user,
        "assistant_text": row.assistant_text,
    }


def _zstd_decoder() -> Callable[[bytes], bytes] | None:
    """The stdlib ``compression.zstd`` (Python 3.14+), else the
    ``zstandard`` package (``llmoji[zstd]``), else ``None``."""
    try:
        from compression import zstd  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        return zstd.decompress  # type: ignore[no-any-return]
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        return None

    def decompress(data: bytes) -> bytes:
        with zstandard.ZstdDecompressor().stream_reader(data) as reader:
            return reader.read()  # type: ignore[no-any-return]

    return decompress


def can_read_zstd() -> bool:
    return _zstd_decoder() is not None


def read_session_text(path: Path) -> str | None:
    """A session file's text, decompressing ``.zst`` (codex-cli
    compresses old rollouts in place). ``None`` when it can't be read."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if path.suffix == ".zst":
        decompress = _zstd_decoder()
        if decompress is None:
            return None
        try:
            data = decompress(data)
        except Exception:  # noqa: BLE001 — a corrupt frame skips the file
            return None
    return data.decode("utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The JSON objects in a JSONL session file, skipping bad lines."""
    out: list[dict[str, Any]] = []
    for line in (read_session_text(path) or "").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def flatten_synthesis(synth_obj: dict[str, Any]) -> str:
    """A synthesis object as one comma-separated adjective string,
    circumplex anchors first. ``llmoji-study`` keeps this flat view
    beside the structured one for its embedding code."""
    primary = synth_obj.get("primary_affect") or []
    extension = synth_obj.get("stance_modality_function") or []
    if not isinstance(primary, list) or not isinstance(extension, list):
        return ""
    return ", ".join([*primary, *extension])


def sanitize_model_id_for_path(model_id: str) -> str:
    """Bundle filename stem for a model id: lowercase, ``/`` → ``__``,
    ``:`` → ``-``; ``"unknown"`` for an empty id."""
    if not model_id:
        return "unknown"
    return model_id.lower().replace("/", "__").replace(":", "-")
