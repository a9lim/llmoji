"""The provider base class: one coding harness llmoji captures from.

A provider knows four things about its harness: where the journal hook
goes and how the harness registers it, which instructions file the
kaomoji request is appended to, how to read the harness's transcripts
into journal rows, and where its journal lives.

The installed hook is a shim that runs ``python -m llmoji.hook <name>``
with the interpreter that installed it. Live capture and ``llmoji
import`` therefore share every line of transcript decoding: a live row
and the replay of the same message are the same row.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .._util import atomic_write_text, write_json
from ..scrape import ScrapeRow
from ..synth_prompts import LEGACY_NUDGE_MESSAGES, NUDGE_MESSAGE

DOC_HEADING = "## Kaomoji"

# Every block install has ever appended, current first. Uninstall and
# re-install strip whichever is present; the current block must stay
# first because the pre-2.1 wording is a prefix of today's.
_DOC_BLOCKS = tuple(
    f"{heading}\n\n{message}"
    for heading in (DOC_HEADING, "# Kaomoji")
    for message in (NUDGE_MESSAGE, *LEGACY_NUDGE_MESSAGES)
)
DOC_BLOCK = _DOC_BLOCKS[0]


class _HomePath:
    """Descriptor behind :func:`home_path`: resolves against
    ``Path.home()`` on every read. An instance attribute of the same
    name shadows it, which is how tests rebind a path."""

    def __init__(self, parts: tuple[str, ...]) -> None:
        self._parts = parts

    def __get__(self, obj: object, objtype: type | None = None) -> Path:
        return Path.home().joinpath(*self._parts)


def home_path(*parts: str) -> Path:
    """A path under the user's home, resolved when read rather than at
    import, so the test suite's ``HOME`` sandbox reaches every default."""
    return cast(Path, _HomePath(parts))


class SettingsCorruptError(RuntimeError):
    """A harness settings file exists but isn't in a shape we can edit
    safely. Install refuses rather than overwrite it."""

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(
            f"refusing to edit {path}: {why}. Fix the file by hand, "
            f"or move it aside, then re-run `llmoji install`."
        )
        self.path = path
        self.why = why


@dataclass
class ProviderStatus:
    name: str
    installed: bool  # hook file present and registered with the harness
    hook_current: bool  # hook file is what install would write now
    doc_current: bool  # the current kaomoji block is in doc_path
    settings_error: str | None
    hook_path: Path
    settings_path: Path
    doc_path: Path
    journal_path: Path
    journal_bytes: int

    @property
    def healthy(self) -> bool:
        return (
            self.installed
            and self.hook_current
            and self.doc_current
            and self.settings_error is None
        )


class Provider:
    """One harness. Subclasses set the paths and implement capture;
    the JSON registration here fits Claude Code's ``settings.json``
    shape, which Codex and Grok share."""

    name: str = ""
    home_dir: Path = Path()
    hooks_dir: Path = Path()
    settings_path: Path = Path()
    journal_path: Path = Path()
    doc_path: Path = Path()
    hook_filename = "kaomoji-log.sh"
    hook_event = "Stop"
    # Printed by the shim before Python runs, for a harness that reads
    # the hook's stdout as its reply (Antigravity wants a JSON object).
    hook_stdout = ""

    @property
    def hook_path(self) -> Path:
        return self.hooks_dir / self.hook_filename

    def is_present(self) -> bool:
        return self.home_dir.exists()

    # --- capture ---

    def live_rows(self, payload: dict[str, Any]) -> list[ScrapeRow]:
        """The kaomoji-led messages of the turn the Stop hook fired on."""
        raise NotImplementedError

    def replay(self, path: Path) -> Iterator[ScrapeRow]:
        """Every kaomoji-led message in one session file."""
        return iter(())

    def session_files(self) -> list[Path]:
        """The session files ``llmoji import`` replays."""
        return []

    @property
    def importable(self) -> bool:
        return type(self).replay is not Provider.replay

    # --- install ---

    def render_hook(self) -> str:
        reply = (
            f"printf '%s\\n' {shlex.quote(self.hook_stdout)}\n"
            if self.hook_stdout else ""
        )
        return (
            "#!/bin/sh\n"
            f"# llmoji journal hook for {self.name}; "
            f"`llmoji install {self.name}` rewrites this file.\n"
            f"{reply}"
            f"{shlex.quote(sys.executable)} -I -m llmoji.hook {self.name} 2>/dev/null\n"
            "exit 0\n"
        )

    def install(self) -> None:
        """Idempotent: register the hook, write it, append the kaomoji
        block to the instructions file. Registration runs first so a
        corrupt settings file is refused before anything is written."""
        self._register()
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.hook_path.write_text(self.render_hook())
        self.hook_path.chmod(0o755)
        text = _read(self.doc_path)
        merged = _merge_doc_block(text)
        if merged != text:
            atomic_write_text(self.doc_path, merged)

    def uninstall(self) -> None:
        """Idempotent inverse of :meth:`install`. The journal stays."""
        self._unregister()
        self.hook_path.unlink(missing_ok=True)
        text = _read(self.doc_path)
        stripped = _strip_doc_block(text)
        if stripped == text:
            return
        if stripped.strip():
            atomic_write_text(self.doc_path, stripped)
        else:
            self.doc_path.unlink()

    def status(self) -> ProviderStatus:
        settings_error = self._settings_error()
        try:
            hook_current = self.hook_path.read_text() == self.render_hook()
        except OSError:
            hook_current = False
        return ProviderStatus(
            name=self.name,
            installed=(
                self.hook_path.exists()
                and settings_error is None
                and self._is_registered()
            ),
            hook_current=hook_current,
            doc_current=DOC_BLOCK in _read(self.doc_path),
            settings_error=settings_error,
            hook_path=self.hook_path,
            settings_path=self.settings_path,
            doc_path=self.doc_path,
            journal_path=self.journal_path,
            journal_bytes=(
                self.journal_path.stat().st_size
                if self.journal_path.exists() else 0
            ),
        )

    # --- registration: hooks[<event>][i].hooks[j].command ---

    def _register(self) -> None:
        cfg = load_json_strict(self.settings_path)
        hooks = cfg.get("hooks", {})
        if not isinstance(hooks, dict):
            raise SettingsCorruptError(
                self.settings_path, "'hooks' is not an object",
            )
        bucket = hooks.get(self.hook_event, [])
        if not isinstance(bucket, list):
            raise SettingsCorruptError(
                self.settings_path, f"hooks[{self.hook_event!r}] is not an array",
            )
        if any(self._is_ours(h) for h in _leaf_hooks(bucket)):
            return
        bucket.append({"hooks": [{"type": "command", "command": str(self.hook_path)}]})
        hooks[self.hook_event] = bucket
        cfg["hooks"] = hooks
        write_json(self.settings_path, cfg)

    def _unregister(self) -> None:
        try:
            cfg = load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return  # can't tell what's ours in a file we can't read
        hooks = cfg.get("hooks")
        bucket = hooks.get(self.hook_event) if isinstance(hooks, dict) else None
        if not isinstance(bucket, list):
            return
        kept: list[Any] = []
        for entry in bucket:
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if isinstance(inner, list) and any(self._is_ours(h) for h in inner):
                rest = [h for h in inner if not self._is_ours(h)]
                if rest:
                    kept.append({**entry, "hooks": rest})
                continue
            kept.append(entry)
        if kept == bucket:
            return
        assert isinstance(hooks, dict)
        if kept:
            hooks[self.hook_event] = kept
        else:
            del hooks[self.hook_event]
        if not hooks:
            del cfg["hooks"]
        write_json(self.settings_path, cfg)

    def _is_registered(self) -> bool:
        try:
            cfg = load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return False
        hooks = cfg.get("hooks")
        bucket = hooks.get(self.hook_event) if isinstance(hooks, dict) else None
        return isinstance(bucket, list) and any(
            self._is_ours(h) for h in _leaf_hooks(bucket)
        )

    def _settings_error(self) -> str | None:
        try:
            load_json_strict(self.settings_path)
        except SettingsCorruptError as e:
            return e.why
        return None

    def _is_ours(self, handler: Any) -> bool:
        return (
            isinstance(handler, dict)
            and isinstance(handler.get("command"), str)
            and same_command(handler["command"], str(self.hook_path))
        )


def _leaf_hooks(bucket: list[Any]) -> Iterator[Any]:
    for entry in bucket:
        inner = entry.get("hooks") if isinstance(entry, dict) else None
        if isinstance(inner, list):
            yield from inner


def same_command(a: str, b: str) -> bool:
    """Whether two hook commands name the same script, after expanding
    ``$VAR`` and ``~``. A registration kept in a dotfiles repository is
    usually spelled ``$HOME/.claude/hooks/kaomoji-log.sh`` so one file
    serves every machine; a literal comparison would read it as absent
    and append a duplicate that journals every message twice."""

    def norm(cmd: str) -> str:
        return os.path.normpath(os.path.expanduser(os.path.expandvars(cmd.strip())))

    return norm(a) == norm(b)


def load_json_strict(path: Path) -> dict[str, Any]:
    """``{}`` for a missing or empty file; :class:`SettingsCorruptError`
    for anything that isn't a JSON object."""
    text = _read(path)
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SettingsCorruptError(path, f"invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise SettingsCorruptError(
            path, f"top-level JSON value is {type(data).__name__}, not an object",
        )
    return data


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _strip_doc_block(text: str) -> str:
    """``text`` without the first kaomoji block install wrote, and
    without the blank line install put before it. A hand-edited block
    matches nothing and is left alone."""
    for block in _DOC_BLOCKS:
        start = text.find(block)
        if start == -1:
            continue
        end = start + len(block)
        if text[end:end + 1] == "\n":
            end += 1
        if text[max(start - 2, 0):start] == "\n\n":
            start -= 1
        return text[:start] + text[end:]
    return text


def _merge_doc_block(text: str) -> str:
    text = _strip_doc_block(text)
    if not text:
        return DOC_BLOCK + "\n"
    sep = "" if text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n"
    return text + sep + DOC_BLOCK + "\n"
