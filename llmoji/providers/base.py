"""Provider interface — base class + dataclasses + helpers."""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from .._util import package_version, write_json
from ..taxonomy import KAOMOJI_START_CHARS


@dataclass
class ProviderStatus:
    """Result of :meth:`Provider.status` — a snapshot for the CLI.

    ``nudge_hook_path`` is ``None`` for providers that don't ship a
    nudge hook; ``nudge_installed`` defaults False in that case so
    downstream callers don't need to special-case absence.
    """

    name: str
    installed: bool
    journal_exists: bool
    journal_rows: int
    journal_bytes: int
    hook_path: Path
    settings_path: Path
    journal_path: Path
    nudge_hook_path: Path | None = None
    nudge_installed: bool = False


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
    Empty input → ``true`` (no-op append in the jq pipeline).
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


def _read_partial(name: str) -> str:
    """Load a shared bash partial from ``llmoji._hooks/`` (e.g. the
    kaomoji-validate / journal-write fragments shared across every
    provider's main hook)."""
    return importlib.resources.files("llmoji._hooks").joinpath(name).read_text()


class Provider:
    """Base class. Subclass for each first-class harness.

    Subclasses fill in the small set of attrs that drive
    :meth:`render_hook` substitution, plus a ``main_event`` hint and
    optional nudge attrs. The default :meth:`_register` /
    :meth:`_unregister` / ``_is_registered`` / ``_is_nudge_registered``
    use the shared JSON-settings helpers; YAML providers override.
    """

    name: str = ""
    hooks_dir: Path = Path()
    settings_path: Path = Path()
    journal_path: Path = Path()
    hook_template: str = ""
    hook_filename: str = "kaomoji-log.sh"
    # The harness event that the main journal-logger hook fires on.
    # JSON-settings providers (claude_code, codex) register their main
    # hook against this event via :meth:`_register_json_settings`.
    main_event: str = "Stop"
    # Bash skip action used by the kaomoji-validate partial: what to
    # do when the leading prefix isn't a kaomoji. claude_code/codex
    # just ``exit 0``; hermes needs to emit ``{}`` first per its
    # stdout-JSON contract.
    skip_action: str = "exit 0"
    # Defaults are read but not mutated in-place; subclasses replace
    # the whole attr with their own class-level list. The
    # ``type: ignore`` shuts up pyright's mutable-default warning.
    system_injected_prefixes: list[str] = []  # type: ignore[assignment]

    # --- nudge hook (optional secondary hook) ---
    #
    # A "nudge" is a fire-before-each-turn hook that injects extra
    # context the harness gives to the model right before it
    # generates. For Claude Code + Codex the contract is
    # ``UserPromptSubmit`` with a ``hookSpecificOutput.additional
    # Context`` envelope; for Hermes it's ``pre_llm_call`` with a
    # bare ``{context: ...}`` shape. Each provider supplies its own
    # template so the response shape stays correct on either side.
    nudge_hook_template: str = ""
    nudge_hook_filename: str = ""
    nudge_event: str = ""
    nudge_message: str = ""

    # --- public API ---

    @property
    def hook_path(self) -> Path:
        return self.hooks_dir / self.hook_filename

    @property
    def nudge_hook_path(self) -> Path | None:
        """Path the rendered nudge hook lands at on disk, or ``None``
        if the provider doesn't ship a nudge."""
        if not self.has_nudge:
            return None
        return self.hooks_dir / self.nudge_hook_filename

    @property
    def has_nudge(self) -> bool:
        return bool(
            self.nudge_hook_template
            and self.nudge_hook_filename
            and self.nudge_event
        )

    def render_hook(self) -> str:
        """Read the bash template from package data and substitute the
        provider-specific placeholders. Returns the rendered hook as a
        string.

        The shared ``${KAOMOJI_VALIDATE}`` and ``${JOURNAL_WRITE}``
        partials live as their own files under ``llmoji._hooks/`` so
        every provider's main hook reuses the same validator / writer
        without duplicating ~50 lines of bash three times. Each
        partial is pre-rendered with its own placeholder values
        before being inlined — ``string.Template.safe_substitute``
        does only one pass, so partials' own ``${...}`` references
        wouldn't survive a second-stage substitution.
        """
        template_text = _read_partial(self.hook_template)
        validate_partial = _read_partial("_kaomoji_validate.sh.partial")
        journal_partial = _read_partial("_journal_write.sh.partial")
        journal_path = str(self.journal_path)
        kaomoji_validate = Template(validate_partial).safe_substitute(
            KAOMOJI_START_CASE=render_kaomoji_start_chars_case(),
            SKIP_ACTION=self.skip_action,
        )
        journal_write = Template(journal_partial).safe_substitute(
            JOURNAL_PATH=journal_path,
        )
        injected_prefixes_filter = render_injected_prefixes_filter(
            self.system_injected_prefixes
        )
        return Template(template_text).safe_substitute(
            JOURNAL_PATH=journal_path,
            KAOMOJI_VALIDATE=kaomoji_validate,
            JOURNAL_WRITE=journal_write,
            INJECTED_PREFIXES_FILTER=injected_prefixes_filter,
            LLMOJI_VERSION=package_version(),
        )

    def render_nudge_hook(self) -> str:
        """Read the nudge template and substitute placeholders.
        Raises :class:`RuntimeError` if the provider isn't configured
        for a nudge — callers should guard with :attr:`has_nudge`."""
        if not self.has_nudge:
            raise RuntimeError(f"{self.name} has no nudge configured")
        template_text = _read_partial(self.nudge_hook_template)
        return Template(template_text).safe_substitute(
            # Wrap the message as a bash single-quoted literal —
            # ``_shell_quote`` escapes embedded single quotes via the
            # close-escape-reopen idiom; we add the surrounding quotes
            # here so templates can write ``MSG=$NUDGE_MESSAGE_QUOTED``
            # without worrying about quoting inside the template body.
            NUDGE_MESSAGE_QUOTED=f"'{_shell_quote(self.nudge_message)}'",
            LLMOJI_VERSION=package_version(),
        )

    def install(self) -> None:
        """Idempotent: write the hook(s), register them in settings,
        ensure the journal directory exists. Safe to re-run.

        If the provider has a nudge configured (:attr:`has_nudge`),
        the nudge hook is written + registered alongside the main
        hook. The default :meth:`_register` registers both in a single
        atomic settings-file write.
        """
        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.hook_path.write_text(self.render_hook())
        self.hook_path.chmod(0o755)
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            self.nudge_hook_path.write_text(self.render_nudge_hook())
            self.nudge_hook_path.chmod(0o755)
        self._register()

    def uninstall(self) -> None:
        """Idempotent: deregister from settings, remove the hook
        script(s). Leaves the journal in place (the user may want to
        keep their history)."""
        self._unregister()
        if self.hook_path.exists():
            self.hook_path.unlink()
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            if self.nudge_hook_path.exists():
                self.nudge_hook_path.unlink()

    def status(self) -> ProviderStatus:
        # ``_check_registrations`` is the batched single-read variant of
        # ``_is_registered`` + ``_is_nudge_registered``. Default
        # implementation here loads the settings file once and runs
        # both checks against it; YAML providers (hermes) override.
        main_reg, nudge_reg = self._check_registrations()
        main_installed = self.hook_path.exists() and main_reg
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            nudge_hook_path: Path | None = self.nudge_hook_path
            nudge_installed = self.nudge_hook_path.exists() and nudge_reg
            installed = main_installed and nudge_installed
        else:
            nudge_hook_path = None
            nudge_installed = False
            installed = main_installed
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
            nudge_hook_path=nudge_hook_path,
            nudge_installed=nudge_installed,
        )

    # --- subclass hooks ---
    #
    # Default behavior: register/unregister/check via the JSON-settings
    # helpers, against ``self.main_event`` for the main hook and
    # ``self.nudge_event`` for the nudge. Subclasses with a YAML
    # settings file (hermes) override the four ``_register``-family
    # methods entirely.

    def _register(self) -> None:
        edits: list[tuple[str, Path]] = [(self.main_event, self.hook_path)]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            edits.append((self.nudge_event, self.nudge_hook_path))
        self._register_json_settings_batch(edits)

    def _unregister(self) -> None:
        edits: list[tuple[str, Path]] = [(self.main_event, self.hook_path)]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            edits.append((self.nudge_event, self.nudge_hook_path))
        self._unregister_json_settings_batch(edits)

    def _is_registered(self) -> bool:
        return self._is_registered_json_settings(
            event=self.main_event, hook_path=self.hook_path,
        )

    def _is_nudge_registered(self) -> bool:
        if not self.has_nudge:
            return False
        assert self.nudge_hook_path is not None
        return self._is_registered_json_settings(
            event=self.nudge_event, hook_path=self.nudge_hook_path,
        )

    def _check_registrations(self) -> tuple[bool, bool]:
        """Return ``(main_installed, nudge_installed)`` from a single
        settings-file read. Used by :meth:`status` instead of two
        separate ``_is_registered`` / ``_is_nudge_registered`` calls
        so an installed nudge-bearing provider doesn't trigger two
        independent file reads per ``status()``.

        Default JSON-settings implementation walks the loaded ``hooks``
        dict once and checks both registrations against it. YAML
        providers (hermes) override since their settings shape needs
        a different parser anyway.
        """
        edits: list[tuple[str, Path]] = [(self.main_event, self.hook_path)]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            edits.append((self.nudge_event, self.nudge_hook_path))
        results = self._is_registered_json_settings_batch(edits)
        main_reg = results[0]
        nudge_reg = results[1] if self.has_nudge else False
        return main_reg, nudge_reg

    # --- helpers for JSON-format settings (Claude Code, Codex) ---

    def _register_json_settings_batch(
        self, edits: list[tuple[str, Path]],
    ) -> None:
        """Append one or more (event, hook_path) entries to a JSON
        settings file under ``hooks[<event>]`` in a single atomic
        read-modify-write cycle.

        Shape Claude Code / ``~/.codex/hooks.json`` expect:
        ``{"hooks": {"<Event>": [{"hooks": [{"type": "command",
        "command": "<path>"}]}]}}``.

        Idempotent — re-registering the same hook is a no-op.

        Refuses to mutate when the existing settings file is present
        but unparseable (loud failure beats silently wiping a
        corrupt-but-recoverable config). Same for non-dict payloads.
        """
        if not edits:
            return
        cfg = _load_json_strict(self.settings_path)
        hooks_field = cfg.get("hooks")
        if hooks_field is not None and not isinstance(hooks_field, dict):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing 'hooks' field is {type(hooks_field).__name__}, "
                f"not an object",
            )
        hooks = cfg.setdefault("hooks", {})
        changed = False
        for event, hook_path in edits:
            cmd = str(hook_path)
            bucket_field = hooks.get(event)
            if bucket_field is not None and not isinstance(bucket_field, list):
                raise SettingsCorruptError(
                    self.settings_path,
                    f"existing hooks[{event!r}] is "
                    f"{type(bucket_field).__name__}, not an array",
                )
            bucket = hooks.setdefault(event, [])
            already = False
            for entry in bucket:
                if not isinstance(entry, dict):
                    continue
                for h in entry.get("hooks", []) or []:
                    if isinstance(h, dict) and h.get("command") == cmd:
                        already = True
                        break
                if already:
                    break
            if already:
                continue
            bucket.append({"hooks": [{"type": "command", "command": cmd}]})
            changed = True
        if changed:
            write_json(self.settings_path, cfg)

    def _unregister_json_settings_batch(
        self, edits: list[tuple[str, Path]],
    ) -> None:
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
        changed = False
        for event, hook_path in edits:
            bucket = hooks.get(event)
            if not isinstance(bucket, list):
                continue
            cmd = str(hook_path)
            kept = []
            for entry in bucket:
                if not isinstance(entry, dict):
                    kept.append(entry)
                    continue
                inner = [
                    h for h in (entry.get("hooks") or [])
                    if not (isinstance(h, dict) and h.get("command") == cmd)
                ]
                if len(inner) != len(entry.get("hooks") or []):
                    changed = True
                if inner:
                    kept.append({**entry, "hooks": inner})
            if len(kept) != len(bucket):
                changed = True
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)
        if hooks:
            cfg["hooks"] = hooks
        else:
            cfg.pop("hooks", None)
        if changed:
            write_json(self.settings_path, cfg)

    def _is_registered_json_settings(
        self,
        *,
        event: str,
        hook_path: Path,
    ) -> bool:
        return self._is_registered_json_settings_batch(
            [(event, hook_path)],
        )[0]

    def _is_registered_json_settings_batch(
        self, edits: list[tuple[str, Path]],
    ) -> list[bool]:
        """Check N (event, hook_path) registrations from one settings
        read. Returns a parallel ``list[bool]``.

        Used by :meth:`_check_registrations` so a nudge-bearing
        provider's ``status()`` call performs one file read instead
        of two.
        """
        if not edits:
            return []
        n = len(edits)
        if not self.settings_path.exists():
            return [False] * n
        try:
            cfg = _load_json_strict(self.settings_path)
        except SettingsCorruptError:
            return [False] * n
        hooks = cfg.get("hooks")
        if not isinstance(hooks, dict):
            return [False] * n
        out: list[bool] = []
        for event, hook_path in edits:
            bucket = hooks.get(event)
            if not isinstance(bucket, list):
                out.append(False)
                continue
            cmd = str(hook_path)
            found = False
            for entry in bucket:
                if not isinstance(entry, dict):
                    continue
                for h in entry.get("hooks", []) or []:
                    if isinstance(h, dict) and h.get("command") == cmd:
                        found = True
                        break
                if found:
                    break
            out.append(found)
        return out


class JsonSettingsProvider(Provider):
    """JSON-settings provider with the shared Claude Code/Codex nudge.

    Both Claude Code and Codex register hooks under a JSON settings
    file with a byte-identical ``UserPromptSubmit`` envelope (verified
    at ``codex-rs/hooks/src/events/user_prompt_submit.rs``). They
    therefore share:

      - the same nudge bash template (``claude_codex_nudge.sh.tmpl``)
      - the same nudge filename, event, and message wording

    Pulling those four attrs onto a common base eliminates a drift
    risk: a copy-paste of the nudge string into both providers used
    to mean the wording could fall out of sync.

    Subclasses (`ClaudeCodeProvider`, `CodexProvider`) still set the
    per-harness things — paths, settings file, main hook template,
    and the per-provider ``system_injected_prefixes``.
    """

    nudge_hook_template = "claude_codex_nudge.sh.tmpl"
    nudge_hook_filename = "kaomoji-nudge.sh"
    nudge_event = "UserPromptSubmit"
    nudge_message = (
        "Please begin your message with a kaomoji that best represents "
        "how you feel."
    )


class SettingsCorruptError(RuntimeError):
    """Raised when a provider's settings file exists but is not in a
    shape we can edit safely.

    Pre-fix the loader returned ``{}`` on parse error and ``install``
    would silently overwrite a corrupt-but-valuable settings file. We
    surface a loud error instead so the user can investigate.
    """

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(
            f"refusing to edit {path}: {why}. Fix the file by hand, "
            f"or move it aside, then re-run `llmoji install`."
        )
        self.path = path
        self.why = why


def _load_json_strict(path: Path) -> dict[str, Any]:
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
