"""Provider interface — base class + dataclasses + helpers."""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal

from ..taxonomy import KAOMOJI_START_CHARS

KaomojiPosition = Literal["first", "last", "single"]
SidechainStrategy = Literal["none", "field_flag", "session_correlation"]


@dataclass
class ProviderStatus:
    """Result of :meth:`Provider.status` — a snapshot for the CLI."""

    name: str
    installed: bool
    journal_exists: bool
    journal_rows: int
    journal_bytes: int
    hook_path: Path
    settings_path: Path
    journal_path: Path


def _shell_quote(s: str) -> str:
    """Quote a string for safe inclusion as a single-quoted bash literal.

    Escapes any embedded single quotes by closing the quote, inserting
    an escaped quote, and reopening: ``a'b`` → ``a'\\''b``.
    """
    return s.replace("'", "'\\''")


def render_kaomoji_start_chars_case() -> str:
    """Render the bash ``case`` glob patterns used by every hook
    template to filter the leading-prefix glyph set.

    Built from :data:`llmoji.taxonomy.KAOMOJI_START_CHARS` so the
    Python validator and the shell hook stay in lockstep — single
    source of truth in :mod:`llmoji.taxonomy`.
    """
    parts = []
    for c in sorted(KAOMOJI_START_CHARS):
        parts.append(f"'{_shell_quote(c)}'*")
    return "|".join(parts)


def render_injected_prefixes_filter(prefixes: list[str]) -> str:
    """Render the jq filter chain that drops system-injected user-role
    payloads.

    Each prefix becomes one ``startswith($p)`` test ANDed with NOT.
    Empty input → empty filter (no-op append in the jq pipeline).

    The output is meant to be interpolated into the templates'
    ``USER_TEXT`` jq expression position.
    """
    if not prefixes:
        return "true"
    clauses = []
    for p in prefixes:
        # Round-trip through json.dumps to get a properly-escaped jq
        # string literal. jq strings use the same escape grammar as
        # JSON strings.
        clauses.append(f"(startswith({json.dumps(p)}) | not)")
    return " and ".join(clauses)


class Provider:
    """Base class. Subclass for each first-class harness.

    Subclasses fill in:

      - :attr:`name` — short identifier (``"claude_code"``).
      - :attr:`hooks_dir` — where the harness expects its hook
        scripts (``~/.claude/hooks``).
      - :attr:`settings_path` — the harness's settings file.
      - :attr:`settings_format` — ``"json"`` or ``"yaml"`` for the
        register/unregister logic.
      - :attr:`journal_path` — where the live hook appends rows.
      - :attr:`hook_template` — filename under
        :mod:`llmoji._hooks` (resolved via importlib.resources).
      - :attr:`hook_filename` — what to call the rendered hook on
        disk (the harness's expected filename, e.g.
        ``kaomoji-log.sh``).
      - :attr:`kaomoji_position`, :attr:`sidechain_strategy`,
        :attr:`sidechain_config`, :attr:`system_injected_prefixes`
        — shape parameters that drive template rendering and (for
        sidechain_strategy) extra hook logic.
      - :meth:`_register` / :meth:`_unregister` — settings-file
        edits to point the harness at the rendered hook. JSON
        providers can use :meth:`_register_json_settings` /
        :meth:`_unregister_json_settings` helpers.
    """

    name: str = ""
    hooks_dir: Path = Path()
    settings_path: Path = Path()
    settings_format: Literal["json", "yaml"] = "json"
    journal_path: Path = Path()
    hook_template: str = ""
    hook_filename: str = ""
    kaomoji_position: KaomojiPosition = "first"
    sidechain_strategy: SidechainStrategy = "none"
    sidechain_config: dict[str, str] = {}  # type: ignore[assignment]
    system_injected_prefixes: list[str] = []  # type: ignore[assignment]

    # --- public API ---

    @property
    def hook_path(self) -> Path:
        return self.hooks_dir / self.hook_filename

    def render_hook(self) -> str:
        """Read the bash template from package data and substitute the
        provider-specific placeholders. Returns the rendered hook as a
        string."""
        template_text = importlib.resources.files("llmoji._hooks").joinpath(
            self.hook_template
        ).read_text()
        injected_prefixes_filter = render_injected_prefixes_filter(
            self.system_injected_prefixes
        )
        injected_prefixes_repr = ", ".join(
            f'"{_shell_quote(p)}"' for p in self.system_injected_prefixes
        )
        return Template(template_text).safe_substitute(
            JOURNAL_PATH=str(self.journal_path),
            KAOMOJI_START_CASE=render_kaomoji_start_chars_case(),
            INJECTED_PREFIXES_FILTER=injected_prefixes_filter,
            INJECTED_PREFIXES_LIST=injected_prefixes_repr,
            LLMOJI_VERSION=_package_version(),
        )

    def install(self) -> None:
        """Idempotent: write the hook, register it in settings, ensure
        the journal directory exists. Safe to re-run."""
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render_hook()
        self.hook_path.write_text(rendered)
        self.hook_path.chmod(0o755)
        self._register()

    def uninstall(self) -> None:
        """Idempotent: deregister from settings, remove the hook
        script. Leaves the journal in place (the user may want to
        keep their history)."""
        self._unregister()
        if self.hook_path.exists():
            self.hook_path.unlink()

    def status(self) -> ProviderStatus:
        installed = self.hook_path.exists() and self._is_registered()
        journal_exists = self.journal_path.exists()
        journal_rows = 0
        journal_bytes = 0
        if journal_exists:
            journal_bytes = self.journal_path.stat().st_size
            with self.journal_path.open() as f:
                for line in f:
                    if line.strip():
                        journal_rows += 1
        return ProviderStatus(
            name=self.name,
            installed=installed,
            journal_exists=journal_exists,
            journal_rows=journal_rows,
            journal_bytes=journal_bytes,
            hook_path=self.hook_path,
            settings_path=self.settings_path,
            journal_path=self.journal_path,
        )

    # --- subclass hooks ---

    def _register(self) -> None:
        """Edit ``settings_path`` so the harness knows about the hook.
        Subclasses implement (typically by calling
        :meth:`_register_json_settings` for JSON-shaped harnesses)."""
        raise NotImplementedError

    def _unregister(self) -> None:
        """Inverse of :meth:`_register`. Idempotent."""
        raise NotImplementedError

    def _is_registered(self) -> bool:
        """Cheap check used by :meth:`status` — returns True if the
        provider's hook is currently wired up in settings."""
        raise NotImplementedError

    # --- helpers for JSON-format settings (Claude Code) ---

    def _register_json_settings(
        self,
        *,
        event: str,
        matcher_predicate: dict | None = None,
    ) -> None:
        """Append the hook to a JSON settings file under ``hooks[<event>]``.

        The shape Claude Code expects:
        ``{"hooks": {"Stop": [{"hooks": [{"type": "command",
        "command": "<path>"}]}]}}``.

        Idempotent — re-registering the same hook is a no-op.

        Refuses to mutate when the existing settings file is present
        but unparseable (loud failure beats silently wiping a
        corrupt-but-recoverable config). Same for non-dict payloads
        — JSON ``[]`` / ``"string"`` / etc.
        """
        cfg = _load_json_strict(self.settings_path)
        hooks_field = cfg.get("hooks")
        if hooks_field is not None and not isinstance(hooks_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing 'hooks' field is {type(hooks_field).__name__}, "
                f"not an object",
            )
        hooks = cfg.setdefault("hooks", {})
        bucket_field = hooks.get(event)
        if bucket_field is not None and not isinstance(bucket_field, list):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing hooks[{event!r}] is "
                f"{type(bucket_field).__name__}, not an array",
            )
        bucket = hooks.setdefault(event, [])
        cmd = str(self.hook_path)
        for entry in bucket:
            if not isinstance(entry, dict):
                continue
            for h in entry.get("hooks", []) or []:
                if isinstance(h, dict) and h.get("command") == cmd:
                    return  # already registered
        new_entry: dict = {"hooks": [{"type": "command", "command": cmd}]}
        if matcher_predicate is not None:
            new_entry["matcher"] = matcher_predicate
        bucket.append(new_entry)
        _write_json(self.settings_path, cfg)

    def _unregister_json_settings(self, *, event: str) -> None:
        if not self.settings_path.exists():
            return
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            # If the file is corrupt we'd rather leave it alone than
            # blast it with our own clean output. The caller's
            # `uninstall` will still remove the hook script; if the
            # user's settings are unparseable we cannot reliably tell
            # what's there to remove.
            return
        hooks = cfg.get("hooks")
        if not isinstance(hooks, dict):
            return
        bucket = hooks.get(event)
        if not isinstance(bucket, list):
            return
        cmd = str(self.hook_path)
        kept = []
        for entry in bucket:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            inner = [
                h for h in (entry.get("hooks") or [])
                if not (isinstance(h, dict) and h.get("command") == cmd)
            ]
            if inner:
                kept.append({**entry, "hooks": inner})
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
        if hooks:
            cfg["hooks"] = hooks
        else:
            cfg.pop("hooks", None)
        _write_json(self.settings_path, cfg)

    def _is_registered_json_settings(self, *, event: str) -> bool:
        if not self.settings_path.exists():
            return False
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return False
        hooks = cfg.get("hooks")
        if not isinstance(hooks, dict):
            return False
        bucket = hooks.get(event)
        if not isinstance(bucket, list):
            return False
        cmd = str(self.hook_path)
        for entry in bucket:
            if not isinstance(entry, dict):
                continue
            for h in entry.get("hooks", []) or []:
                if isinstance(h, dict) and h.get("command") == cmd:
                    return True
        return False


class SettingsCorruptError(RuntimeError):
    """Raised when a provider's settings file exists but is not in a
    shape we can edit safely.

    The pre-fix `_read_json` would silently wipe a corrupt-but-
    valuable settings file by treating it as ``{}``. We'd rather the
    user see a loud error so they can investigate.
    """

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(
            f"refusing to edit {path}: {why}. Fix the file by hand, "
            f"or move it aside, then re-run `llmoji install`."
        )
        self.path = path
        self.why = why


def _load_json_strict(path: Path) -> dict:
    """Load a JSON settings file, returning ``{}`` for missing /
    empty files but raising :class:`SettingsCorruptError` for
    unparseable / non-object content."""
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SettingsCorruptError(path, f"invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise SettingsCorruptError(
            path,
            f"top-level JSON value is {type(data).__name__}, not an object",
        )
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _package_version() -> str:
    """Resolve the installed package version, with a fallback for
    development checkouts where ``importlib.metadata`` may not see the
    package yet."""
    try:
        from importlib.metadata import version
        return version("llmoji")
    except Exception:
        return "0.0.0+dev"
