"""HookInstaller interface — base class + dataclasses + helpers.

The class is named after its job: writing one harness's bash hook
script(s) and registering them in that harness's settings file.
``providers/`` stays the directory name — concrete subclasses
correspond to user-facing harness providers (claude_code, codex,
hermes) — but the abstraction itself is hook-installation, not a
generic "provider".
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from .._util import package_version, write_json
from ..taxonomy import KAOMOJI_START_CHARS


@dataclass
class ProviderStatus:
    """Result of :meth:`HookInstaller.status` — a snapshot for the CLI.

    ``installed`` rolls up main + nudge; ``main_installed`` and
    ``nudge_installed`` carry the per-piece state so the CLI can
    point at which half is broken when ``installed`` is False.
    ``nudge_hook_path`` is ``None`` for providers that don't ship a
    nudge hook; ``nudge_installed`` defaults False in that case so
    downstream callers don't need to special-case absence.

    Health-check fields (``main_hook_current``, ``nudge_hook_current``,
    ``settings_parse_error``) layer cheap diagnostics onto the
    install-presence check. ``main_hook_current`` is False when the
    hook file exists but its content doesn't match what
    :meth:`HookInstaller.render_hook` produces for the running
    package version — typically caused by upgrading llmoji without
    re-running ``install``. ``settings_parse_error`` carries the
    why-string from :class:`SettingsCorruptError` when the settings
    file is unparseable, ``None`` otherwise; the CLI surfaces it so
    a user with a hand-broken config sees the reason inline rather
    than a silent "not installed" marker.
    """

    name: str
    installed: bool
    main_installed: bool
    journal_exists: bool
    journal_bytes: int
    hook_path: Path
    settings_path: Path
    journal_path: Path
    nudge_hook_path: Path | None = None
    nudge_installed: bool = False
    main_hook_current: bool = True
    nudge_hook_current: bool = True
    settings_parse_error: str | None = None


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


def _read_plugin_data(name: str) -> str:
    """Load a TS plugin template / partial from ``llmoji._plugins/``.

    Mirrors :func:`_read_partial` but for the plugin-style providers
    (opencode / openclaw) — the rendered output is TypeScript that
    the host harness loads at runtime, not bash.
    """
    return importlib.resources.files("llmoji._plugins").joinpath(name).read_text()


def render_plugin_template(template_name: str) -> str:
    """Render a TS plugin template by splicing the canonical taxonomy
    partial between ``// BEGIN SHARED TAXONOMY`` / ``// END SHARED
    TAXONOMY`` markers and substituting ``__LLMOJI_VERSION__``.

    The marker-fence approach (instead of ``string.Template``) is
    deliberate: TS template-literal syntax (``${expr}``) collides
    with Python's ``${name}`` placeholder syntax, so a Template-style
    render would mangle every ``${...}`` interpolation in the plugin
    body. Marker splicing sidesteps the collision and keeps the
    plugin source readable in any TS-aware editor.

    The partial is read fresh each call — the cost is one filesystem
    read per render, paid at install time only. Single source of
    truth for the taxonomy block lives in
    ``llmoji/_plugins/_kaomoji_taxonomy.ts.partial``; the test suite
    asserts the rendered output round-trips bit-identically against
    that file so a stale hand-edit in either template is caught at
    CI rather than landing as a quietly-divergent TS port.
    """
    template_text = _read_plugin_data(template_name)
    partial = _read_plugin_data("_kaomoji_taxonomy.ts.partial")
    BEGIN = "// BEGIN SHARED TAXONOMY"
    END = "// END SHARED TAXONOMY"
    if BEGIN not in template_text or END not in template_text:
        # Templates without markers (e.g. the openclaw plugin.json)
        # only need version substitution.
        return template_text.replace("__LLMOJI_VERSION__", package_version())
    start = template_text.index(BEGIN)
    body_start = template_text.index("\n", start) + 1
    end_idx = template_text.index(END)
    body = partial if partial.endswith("\n") else partial + "\n"
    spliced = template_text[:body_start] + body + template_text[end_idx:]
    return spliced.replace("__LLMOJI_VERSION__", package_version())


class HookInstaller:
    """Base class — one subclass per first-class harness.

    Renamed from ``Provider`` in 1.1.x. The class's job is writing
    bash hook script(s) and registering them in a harness's
    settings file; the old name described who not what. The
    ``providers/`` directory name is kept because concrete
    subclasses do correspond to user-facing harness providers
    (claude_code, codex, hermes).

    Subclasses fill in the small set of attrs that drive
    :meth:`render_hook` substitution, plus a ``main_event`` hint and
    optional nudge attrs. The default :meth:`_register` /
    :meth:`_unregister` / :meth:`_check_registrations` use the
    shared JSON-settings helpers; YAML providers override.
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

    def is_present(self) -> bool:
        """Heuristic: does this harness's home directory exist on disk?

        Used by ``llmoji install`` (no provider arg) to decide which
        providers to install for. The signal is the same one a user
        would get from ``ls ~`` — ``~/.claude`` exists for Claude Code,
        ``~/.codex`` for Codex, ``~/.hermes`` for Hermes. Trying to be
        cleverer (distinguishing "installed" from "ran once and
        aborted") is fragile and not worth it; if a parent dir exists
        the harness has at least left a footprint, and a user who
        deleted their harness home but kept the parent dir is an edge
        case we accept the false-positive on. Subclasses are free to
        override if a more specific signal turns out to matter.
        """
        return self.settings_path.parent.exists()

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
        # ``_check_registrations`` reads the settings file once and
        # reports both main + nudge registration in a single pass.
        # Default implementation walks the JSON-settings shape; YAML
        # providers (hermes) override.
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
        # Byte-size only — counting rows would require walking the
        # whole file and ``analyze`` re-walks it via ``iter_journal``
        # immediately after, so the per-row scan here is wasted work
        # on multi-hundred-MB journals.
        journal_bytes = (
            self.journal_path.stat().st_size if journal_exists else 0
        )
        # Health checks: cheap O(few-file-reads) diagnostics on top
        # of the install-presence check. Stale-hook detection compares
        # the on-disk hook content against what render_hook() would
        # produce now, so an upgrade-without-reinstall surfaces as
        # a "stale" warning rather than silently running yesterday's
        # bash. Skip when the hook file is missing — that's already
        # captured by ``main_installed=False``.
        if main_installed:
            main_hook_current = self._is_main_hook_current()
        else:
            main_hook_current = True
        if self.has_nudge and nudge_installed:
            nudge_hook_current = self._is_nudge_hook_current()
        else:
            nudge_hook_current = True
        settings_parse_error = self._check_settings_health()
        return ProviderStatus(
            name=self.name,
            installed=installed,
            main_installed=main_installed,
            journal_exists=journal_exists,
            journal_bytes=journal_bytes,
            hook_path=self.hook_path,
            settings_path=self.settings_path,
            journal_path=self.journal_path,
            nudge_hook_path=nudge_hook_path,
            nudge_installed=nudge_installed,
            main_hook_current=main_hook_current,
            nudge_hook_current=nudge_hook_current,
            settings_parse_error=settings_parse_error,
        )

    def _is_main_hook_current(self) -> bool:
        """Hook file content matches what :meth:`render_hook` produces
        right now. False on file read errors (defensive — the user
        sees "stale" and re-runs install)."""
        try:
            return self.hook_path.read_text() == self.render_hook()
        except OSError:
            return False

    def _is_nudge_hook_current(self) -> bool:
        if self.nudge_hook_path is None:
            return True
        try:
            return self.nudge_hook_path.read_text() == self.render_nudge_hook()
        except OSError:
            return False

    def _check_settings_health(self) -> str | None:
        """``None`` if the settings file is parseable (or absent),
        else a short error message describing why parsing failed.

        Default implementation handles JSON-settings providers via
        :func:`_load_json_strict`. YAML providers (hermes) override.
        """
        if not self.settings_path.exists():
            return None
        try:
            _load_json_strict(self.settings_path)
        except SettingsCorruptError as e:
            return e.why
        return None

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

    def _check_registrations(self) -> tuple[bool, bool]:
        """Return ``(main_installed, nudge_installed)`` from a single
        settings-file read.

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
    #
    # All three batch helpers walk the same nested shape:
    #   cfg["hooks"][event][i]["hooks"][j]
    # where the leaf at ``[j]`` is ``{"type": "command", "command":
    # "<path>"}``. The dict/list/dict/dict isinstance gauntlet lives
    # in :func:`_iter_leaf_commands` so the three helpers can speak
    # in terms of "valid leaf entries" without re-implementing the
    # validation. Malformed sub-shapes are skipped silently — they
    # pass through unchanged in :meth:`_unregister_json_settings_batch`.

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
        hooks: dict[str, Any] = cfg.setdefault("hooks", {})
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
            bucket: list[Any] = hooks.setdefault(event, [])
            if any(leaf_cmd == cmd for _, _, leaf_cmd in _iter_leaf_commands(bucket)):
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
            # Group the indices to drop by their owning entry so each
            # entry's ``hooks`` list is rebuilt at most once.
            drops_per_entry: dict[int, set[int]] = {}
            for entry_idx, hook_idx, leaf_cmd in _iter_leaf_commands(bucket):
                if leaf_cmd == cmd:
                    drops_per_entry.setdefault(entry_idx, set()).add(hook_idx)
            if not drops_per_entry:
                continue
            kept: list[Any] = []
            for entry_idx, entry in enumerate(bucket):
                drops = drops_per_entry.get(entry_idx)
                if drops is None or not isinstance(entry, dict):
                    # No matches in this entry, OR entry isn't a dict
                    # we can rewrite — preserve as-is.
                    kept.append(entry)
                    continue
                # Rebuild the entry's ``hooks`` list, dropping matched
                # indices. Malformed siblings (non-dicts, dicts without
                # ``command``) are kept unchanged because the walker
                # only emitted indices for valid-and-matching leaves.
                inner_field = entry.get("hooks") or []
                inner = [
                    h for j, h in enumerate(inner_field)
                    if j not in drops
                ]
                if inner:
                    kept.append({**entry, "hooks": inner})
                # else: entry collapsed to empty hooks list, drop it
            changed = True
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)
        if not hooks:
            cfg.pop("hooks", None)
        if changed:
            write_json(self.settings_path, cfg)

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
            out.append(
                any(leaf_cmd == cmd for _, _, leaf_cmd in _iter_leaf_commands(bucket))
            )
        return out


class JsonSettingsHookInstaller(HookInstaller):
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


class PluginInstaller(HookInstaller):
    """Plugin-style provider — renders a TypeScript plugin (one or more
    files) into the harness's plugins directory rather than rendering a
    bash hook.

    Pre-1.3 these were generic-JSONL examples under ``examples/``; 1.3
    promotes the two TS-plugin harnesses (opencode, openclaw) to
    first-class so ``llmoji install <name>`` writes the rendered plugin
    automatically. The plugin code itself handles per-turn nudge
    injection via the host harness's system-prompt extension hook —
    no separate bash nudge file lands on disk.

    Subclasses populate :attr:`plugin_files` (a list of
    ``(template_name, dest_filename)`` tuples) and :attr:`plugin_dir`
    (where on disk the rendered files land). Both opencode and
    openclaw write their journal rows to the generic-JSONL path
    ``~/.llmoji/journals/<name>.jsonl`` directly from the TS plugin —
    no live shell hook in the loop, no settings.json ``hooks`` entry.

    OpenClaw additionally edits ``~/.openclaw/config.json`` to flip
    the per-plugin ``allowConversationAccess`` flag; that override
    lives in :class:`OpenclawProvider` rather than here because the
    settings-edit shape is openclaw-specific.

    The :class:`HookInstaller` interface stays the contract — the CLI
    walks every installed provider via the same ``install`` /
    ``uninstall`` / ``status`` calls regardless of installer kind.
    Bash-specific attrs (``hook_template`` / ``hook_filename`` /
    ``main_event`` / ``skip_action`` / ``system_injected_prefixes``)
    are unused for plugin providers but preserved by the class
    hierarchy so :class:`ProviderStatus` consumers don't need a kind
    discriminator.
    """

    # Bash-side attrs are unused — render_hook is overridden, the
    # main hook never lands on disk as bash.
    hook_template: str = ""
    hook_filename: str = ""
    main_event: str = "plugin"  # informational; never registered as a bash event
    skip_action: str = ""
    system_injected_prefixes: list[str] = []  # type: ignore[assignment]

    # Nudge is baked into the plugin code itself, not a separate
    # bash file. has_nudge stays False so the base class's nudge
    # writing / registration is skipped cleanly.
    nudge_hook_template: str = ""
    nudge_hook_filename: str = ""
    nudge_event: str = ""
    nudge_message: str = (
        "Please begin your message with a kaomoji that best represents "
        "how you feel."
    )

    # --- plugin-specific surface (subclass populates) ---
    #
    # ``plugin_files`` is the list of (template_name, dest_filename)
    # pairs the install writes. The first entry is the canonical
    # "main artifact" — its dest path is reused as ``hook_path`` so
    # ``ProviderStatus.hook_path`` continues to point at something
    # meaningful for the CLI's status print. Single-file plugins
    # (opencode) put one entry; multi-file bundles (openclaw) list
    # every file the host expects.

    plugin_files: list[tuple[str, str]] = []  # type: ignore[assignment]
    plugin_dir: Path = Path()

    @property
    def hook_path(self) -> Path:
        """Reuse :attr:`HookInstaller.hook_path` as the path to the
        plugin's primary file. Pre-empties to ``plugin_dir`` itself
        when no files are configured (defensive — should never happen
        on a properly-declared subclass)."""
        if not self.plugin_files:
            return self.plugin_dir
        _, dest = self.plugin_files[0]
        return self.plugin_dir / dest

    @property
    def nudge_hook_path(self) -> Path | None:
        return None

    @property
    def has_nudge(self) -> bool:
        # The plugin file itself injects the nudge via the harness's
        # per-turn system-prompt extension hook (opencode's
        # ``experimental.chat.system.transform``, openclaw's
        # ``before_prompt_build``). No separate nudge file lives on
        # disk to install or check.
        return False

    def is_present(self) -> bool:
        """Plugin providers detect via the parent harness's home dir.

        For opencode that's ``~/.config/opencode/`` (the dir that holds
        ``plugins/``); for openclaw it's ``~/.openclaw/``. Subclasses
        that need a different signal can override.

        Reusing :attr:`plugin_dir`'s parent here means a never-installed
        opencode user (no ``~/.config/opencode``) won't get autodetected
        by ``llmoji install`` (no-arg). That matches the bash providers'
        ``settings_path.parent.exists()`` rule.
        """
        return self.plugin_dir.parent.exists()

    # --- rendering ---

    def render_hook(self) -> str:
        """Render the primary plugin file (the one at :attr:`hook_path`).

        :class:`HookInstaller.status` calls this to detect a stale
        on-disk file (``main_hook_current``); reusing the same name
        keeps the staleness check uniform across installer kinds.
        """
        if not self.plugin_files:
            return ""
        template_name, _ = self.plugin_files[0]
        return render_plugin_template(template_name)

    def render_plugin_file(self, template_name: str) -> str:
        """Render any plugin file by template name. Used by multi-file
        bundles (openclaw's index.ts + plugin.json)."""
        return render_plugin_template(template_name)

    def render_nudge_hook(self) -> str:  # pragma: no cover — has_nudge False
        raise RuntimeError(
            f"{self.name} bakes its nudge into the plugin file"
        )

    # --- install / uninstall lifecycle ---

    def install(self) -> None:
        """Idempotent: write every plugin file, register if the host
        needs an explicit toggle, ensure the journal directory exists.

        File writes are simple ``write_text`` with mode 0o644 — these
        are TS / JSON files the host harness loads, not executables.
        """
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        for template_name, dest_filename in self.plugin_files:
            dest = self.plugin_dir / dest_filename
            dest.write_text(self.render_plugin_file(template_name))
        self._register()

    def uninstall(self) -> None:
        """Idempotent: deregister, remove every plugin file. Leaves
        the journal in place (mirrors :meth:`HookInstaller.uninstall`)
        and removes the plugin directory itself when empty so a
        subsequent install starts clean."""
        self._unregister()
        for _, dest_filename in self.plugin_files:
            dest = self.plugin_dir / dest_filename
            if dest.exists():
                dest.unlink()
        # Best-effort cleanup of an emptied plugin dir. Non-empty
        # rmdir raises OSError, which we swallow — the user may have
        # other unrelated files there.
        try:
            self.plugin_dir.rmdir()
        except OSError:
            pass

    # --- status helpers ---

    def _register(self) -> None:
        """No-op default — file presence IS registration for harnesses
        that auto-load plugin files (opencode). Subclasses with an
        explicit registration step (openclaw's config flag) override.
        """
        return None

    def _unregister(self) -> None:
        return None

    def _check_registrations(self) -> tuple[bool, bool]:
        """Plugin-style status check: every declared plugin file
        present on disk = registered. Subclasses with an additional
        registration check (openclaw's config flag) AND-fold their
        own check into the first slot of the returned tuple.
        """
        all_present = bool(self.plugin_files) and all(
            (self.plugin_dir / dest).exists()
            for _, dest in self.plugin_files
        )
        return all_present, False

    def _is_main_hook_current(self) -> bool:
        """Plugin staleness: every declared file's on-disk content
        matches what :meth:`render_plugin_file` produces right now.
        False on any read error or any mismatch — surfaces as
        ``main_hook_current=False`` in status, prompting a re-run.
        """
        try:
            for template_name, dest_filename in self.plugin_files:
                dest = self.plugin_dir / dest_filename
                if dest.read_text() != self.render_plugin_file(template_name):
                    return False
            return True
        except OSError:
            return False

    def _check_settings_health(self) -> str | None:
        """No managed settings file by default. Subclasses that touch
        a JSON config (openclaw) override to re-add a parseability
        check."""
        return None


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


def _iter_leaf_commands(
    bucket: list[Any],
) -> Iterator[tuple[int, int, str]]:
    """Walk a JSON-settings event bucket and yield ``(entry_idx,
    hook_idx, command)`` for every valid leaf hook entry.

    The shape this validates is
    ``bucket[entry_idx]["hooks"][hook_idx]["command"]``:

    - each ``entry`` must be a dict
    - ``entry["hooks"]`` must be a list (a missing/falsy value is
      treated as empty)
    - each leaf ``h`` must be a dict whose ``"command"`` is a string

    Anything that fails the gauntlet is skipped silently. Used by
    the three batch helpers as the single source of truth for "what
    counts as a valid registration?" — concentrating the isinstance
    chain here means a shape-validation tweak only happens once.
    """
    for entry_idx, entry in enumerate(bucket):
        if not isinstance(entry, dict):
            continue
        inner = entry.get("hooks") or []
        if not isinstance(inner, list):
            continue
        for hook_idx, h in enumerate(inner):
            if not isinstance(h, dict):
                continue
            cmd = h.get("command")
            if isinstance(cmd, str):
                yield entry_idx, hook_idx, cmd


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
