"""Hermes (NousResearch hermes-agent) provider.

Implemented against hermes-agent v0.11.0's
[Event Hooks docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/),
cross-checked against the actual source at
``hermes-agent/agent/shell_hooks.py`` + ``hermes-agent/run_agent.py``.
The applicable mechanism is **shell hooks** under
``~/.hermes/agent-hooks/`` registered via the ``hooks:`` block in
``~/.hermes/config.yaml``. (Hermes also supports gateway hooks at
``~/.hermes/hooks/<name>/HOOK.yaml + handler.py`` and plugin hooks
registered via ``ctx.register_hook()``; for a CLI-installed
journal-logger the shell-hooks path is the lowest-friction option
because it doesn't require a Python plugin and inherits the same
fail-open / stdout-JSON contract Claude Code and Codex hooks use.)

The CLI installs **two** hooks for hermes:

  - ``post-llm-call.sh`` — main journal logger; fires after every
    assistant turn that the agent loop completes. Walks
    ``extra.conversation_history`` and emits one journal row per
    kaomoji-led assistant message in the current turn (one row per
    kaomoji-led message, same multi-emit shape Claude Code + Codex
    produce).
  - ``pre-llm-call.sh`` — UserPromptSubmit-equivalent nudge that
    injects the kaomoji-reminder context.

Stdin payload (``post_llm_call``)::

    {
      "hook_event_name": "post_llm_call",
      "tool_name":       null,
      "tool_input":      null,
      "session_id":      "...",
      "cwd":             "...",          # = Path.cwd() of agent process
      "extra": {
        "user_message":          "...",  # original, pre-injection
        "assistant_response":    "...",  # final response only
        "conversation_history":  [...],  # full message list (this is
                                         # what we walk for multi-emit)
        "model":                 "...",
        "platform":              "..."
      }
    }

Stdout: JSON. ``{}`` is no-op. Malformed JSON / non-zero exit /
timeout never abort the agent loop (fail-open).

Per-provider quirks (vs claude_code / codex):

  - **One row per kaomoji-led assistant message in the current turn**,
    walked off ``extra.conversation_history``. Pre-fix the hook only
    read ``extra.assistant_response`` (the final string) and missed
    every progress message. The slice from the latest user-role
    message to the end of the array IS the current turn — every
    assistant entry in that window is a candidate row.
  - **Subagent (delegate_task) filtering: not viable on the current
    payload contract.** ``subagent_stop`` fires from the parent
    agent's process with the **parent's** ``session_id`` (no child
    id; verified at ``hermes-agent/tools/delegate_tool.py:2120``),
    and ``post_llm_call`` doesn't expose ``parent_session_id`` either,
    so neither side carries enough info to filter children from a
    shell hook. Subagent post_llm_call events therefore land in the
    journal under their own session_ids. We'll wire a real filter
    when an upstream payload change makes one possible. The fix
    we'd want: either (a) ``subagent_stop`` carries the child id, or
    (b) ``post_llm_call`` exposes ``parent_session_id`` /
    ``is_subagent``. Both are upstream concerns.
  - ``extra.user_message`` is delivered pre-injection per the
    documented contract — no system-injected prefixes to filter.

Settings live in YAML, so register/unregister can't reuse the JSON
batch helpers the base class supplies for claude_code / codex.
**ruamel.yaml is used for parsing only, never for serialization** —
1.2.x briefly tried a load-mutate-dump approach and shipped a
silent-data-corruption bug: ruamel's RoundTripDumper does not emit
backslash line-continuations in double-quoted scalars, so a string
PyYAML had wrapped at a non-whitespace boundary (e.g. inside a
kaomoji ``(◕‿◕)`` literal) gets a single space inserted at the
wrap point on round-trip per YAML 1.2 fold rules. Hermes itself
writes ``~/.hermes/config.yaml`` with PyYAML, which uses the
backslash continuation form — so loading a Hermes-written config
and re-dumping with ruamel was guaranteed to mutate any wrapped
double-quoted scalar that didn't happen to wrap at whitespace. A
1-char change to a personality prompt on a config that the user
didn't touch is the kind of silent corruption that only shows up
weeks later as "the AI is acting slightly different". Not safe.

The current implementation uses ruamel ``CommentedMap.lc`` /
``CommentedSeq.lc`` line/col marks to compute exact line ranges
for our targeted edits, then applies the edits as text splices on
the original file content. ruamel never serializes; the user's
config stays byte-stable everywhere except the lines we explicitly
insert or delete. Cost is more code (~250 vs ~70 lines for the
load-mutate-dump approach) and a refusal on a few edge shapes
(flow-style ``hooks:`` block, empty event buckets) where surgical
edits aren't well-defined.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .._util import atomic_write_text
from .base import HookInstaller, SettingsCorruptError


def _yaml_parser() -> YAML:
    """Round-trip parser used only to load + give us line/col marks.
    We never call ``.dump()`` on this — see module docstring for the
    data-corruption story that motivates the parsing-only choice.
    """
    return YAML(typ="rt")


def _is_placeholder_hooks(value: object) -> bool:
    """``hooks: ~`` / ``hooks: {}`` / ``hooks: []`` are functionally
    "no hooks configured" and we replace them with a fresh block.
    Anything else populated is real user data — either merge into
    (mapping case) or refuse (sequence/scalar case)."""
    if value is None:
        return True
    if isinstance(value, (CommentedMap, CommentedSeq)) and not value:
        return True
    return False


def _walk_to_dedent(
    lines: list[str], start: int, parent_col: int
) -> int:
    """Return the index of the first non-blank line at indent
    <= ``parent_col``, scanning forward from ``start``. EOF if none.
    Blank lines are skipped (treated as still inside the block).
    """
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent <= parent_col:
            return i
        i += 1
    return n


def _block_end_excl(
    lines: list[str], block_start: int, parent_col: int
) -> int:
    """Index *after* the last non-blank line of the YAML block whose
    first line is at ``block_start``. Trailing blank lines belong to
    the *separator* between this block and whatever comes next, not
    to the block itself — so they're excluded from the returned
    range. That keeps a user's intentional blank-line structuring
    around the block intact when we delete or insert at the boundary.
    """
    dedent = _walk_to_dedent(lines, block_start + 1, parent_col)
    j = dedent
    while j > block_start + 1 and lines[j - 1].strip() == "":
        j -= 1
    return j


class HermesProvider(HookInstaller):
    name = "hermes"
    hooks_dir = Path.home() / ".hermes" / "agent-hooks"
    settings_path = Path.home() / ".hermes" / "config.yaml"
    journal_path = Path.home() / ".hermes" / "kaomoji-journal.jsonl"
    hook_template = "hermes.sh.tmpl"
    hook_filename = "post-llm-call.sh"
    main_event = "post_llm_call"
    # The validate partial is inlined inside a per-message
    # ``while read`` loop in the rendered hook (one iteration per
    # assistant message in the current turn); ``continue`` is the
    # right skip action — a non-kaomoji message skips its row
    # without bailing the rest of the walk. The base default
    # ``"exit 0"`` would terminate the loop's subshell on the first
    # non-kaomoji message, dropping every later kaomoji-led message
    # in the same turn. Same shape as claude_code / codex now that
    # hermes is multi-emit too. The closing ``echo '{}'; exit 0``
    # in the template body satisfies the hermes stdout-JSON contract
    # after the loop completes.
    skip_action = "continue"
    system_injected_prefixes: list[str] = []

    # Nudge: pre_llm_call with a bare ``{context: ...}`` shape (per
    # docs, "the only hook whose return value is used"). Different
    # template from the Claude/Codex shared one — Hermes wraps no
    # ``hookSpecificOutput`` envelope.
    nudge_hook_template = "hermes_nudge.sh.tmpl"
    nudge_hook_filename = "pre-llm-call.sh"
    nudge_event = "pre_llm_call"
    nudge_message = (
        "Please begin your message with a kaomoji that best represents "
        "how you feel."
    )

    # PyYAML-style block-formatting defaults for fresh blocks. When we're
    # merging into an existing user-populated hooks block, we infer the
    # actual style from the user's own keys / list items instead.
    _DEFAULT_MAPPING_INDENT = 2
    _DEFAULT_LIST_INDENT = 4

    # --- public surface ---

    def _edits(self) -> list[tuple[str, Path]]:
        edits: list[tuple[str, Path]] = [(self.main_event, self.hook_path)]
        if self.has_nudge:
            assert self.nudge_hook_path is not None
            edits.append((self.nudge_event, self.nudge_hook_path))
        return edits

    def _register(self) -> None:
        text, doc = self._read_and_parse()
        new_text = self._apply_register(text, doc)
        if new_text != text:
            atomic_write_text(self.settings_path, new_text)

    def _unregister(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            text, doc = self._read_and_parse()
        except SettingsCorruptError:
            # Refuse to mutate a corrupt config on uninstall — same
            # contract as the JSON-settings path. The hook script
            # files still get removed by HookInstaller.uninstall.
            return
        new_text = self._apply_unregister(text, doc)
        if new_text == text:
            return
        if new_text.strip():
            atomic_write_text(self.settings_path, new_text)
        else:
            self.settings_path.unlink()

    def _check_registrations(self) -> tuple[bool, bool]:
        if not self.settings_path.exists():
            return False, False
        try:
            _, doc = self._read_and_parse()
        except SettingsCorruptError:
            return False, False
        hooks = doc.get("hooks")
        if not isinstance(hooks, CommentedMap):
            return False, False

        results: list[bool] = []
        for event, hook_path in self._edits():
            bucket = hooks.get(event)
            if not isinstance(bucket, CommentedSeq):
                results.append(False)
                continue
            cmd = str(hook_path)
            results.append(
                any(
                    isinstance(e, CommentedMap) and e.get("command") == cmd
                    for e in bucket
                )
            )
        main_reg = results[0]
        nudge_reg = results[1] if self.has_nudge else False
        return main_reg, nudge_reg

    # --- read+parse ---

    def _read_and_parse(self) -> tuple[str, CommentedMap]:
        if not self.settings_path.exists():
            return "", CommentedMap()
        text = self.settings_path.read_text()
        if not text.strip():
            return text, CommentedMap()
        try:
            doc = _yaml_parser().load(text)
        except YAMLError as e:
            raise SettingsCorruptError(
                self.settings_path, f"unparseable YAML ({e})"
            ) from e
        if doc is None:
            return text, CommentedMap()
        if not isinstance(doc, CommentedMap):
            raise SettingsCorruptError(
                self.settings_path,
                f"top-level YAML value is {type(doc).__name__}, "
                f"not a mapping",
            )
        return text, doc

    # --- register flow ---

    def _apply_register(self, text: str, doc: CommentedMap) -> str:
        hooks = doc.get("hooks")
        edits = self._edits()

        if hooks is None or _is_placeholder_hooks(hooks):
            return self._install_fresh_block(text, doc, hooks, edits)

        if not isinstance(hooks, CommentedMap):
            raise SettingsCorruptError(
                self.settings_path,
                f"existing 'hooks' is {type(hooks).__name__}, "
                f"not a mapping",
            )
        if hooks.fa.flow_style():
            raise SettingsCorruptError(
                self.settings_path,
                "existing 'hooks' uses flow style ({...}); convert "
                "to block style and re-run",
            )

        return self._merge_into_existing_block(text, doc, hooks, edits)

    def _install_fresh_block(
        self,
        text: str,
        doc: CommentedMap,
        existing_value: object,
        edits: list[tuple[str, Path]],
    ) -> str:
        """No real hooks block yet — write a fresh one. Replaces a
        placeholder line in place if present; otherwise appends to
        EOF. Uses PyYAML-style indent (mapping=2, list=4) since
        there's nothing to infer from."""
        block_lines = self._render_fresh_block(edits)

        if existing_value is not None or "hooks" in doc:
            # Placeholder shapes (``hooks: {}`` / ``hooks: []`` /
            # ``hooks: ~`` / ``hooks:``) are all single-line, so the
            # one-line replacement is correct.
            lines = text.split("\n")
            key_line = doc.lc.data["hooks"][0]
            new_lines = lines[:key_line] + block_lines + lines[key_line + 1:]
            return "\n".join(new_lines)

        # No hooks key in the doc at all. Append at EOF, preserving
        # whatever final-newline convention the file already has.
        block_text = "\n".join(block_lines)
        if not text:
            return block_text + "\n"
        sep = "" if text.endswith("\n") else "\n"
        return text + sep + block_text + "\n"

    def _merge_into_existing_block(
        self,
        text: str,
        doc: CommentedMap,
        hooks: CommentedMap,
        edits: list[tuple[str, Path]],
    ) -> str:
        """Surgical merge: insert a new event sub-block, append a
        list item, or skip per-edit based on what the parsed hooks
        mapping already contains. Indent style is inferred from the
        user's existing keys + list items so additions match what
        was there.
        """
        lines = text.split("\n")
        mapping_indent = self._infer_mapping_indent(hooks)
        list_indent = self._infer_list_indent(hooks, mapping_indent)

        # Plan as a list of (insert_at_line_idx, lines_to_insert).
        ops: list[tuple[int, list[str]]] = []
        for event, hook_path in edits:
            cmd = str(hook_path)
            bucket = hooks.get(event)
            if bucket is None:
                # New event sub-block at end of hooks block.
                hooks_key_line = doc.lc.data["hooks"][0]
                hooks_block_end = _block_end_excl(
                    lines, hooks_key_line, parent_col=0
                )
                new_lines = self._render_event_subblock(
                    event, [cmd], mapping_indent, list_indent
                )
                ops.append((hooks_block_end, new_lines))
            elif isinstance(bucket, CommentedSeq):
                if bucket.fa.flow_style():
                    raise SettingsCorruptError(
                        self.settings_path,
                        f"existing hooks[{event!r}] uses flow style; "
                        f"convert to block style and re-run",
                    )
                if any(
                    isinstance(e, CommentedMap) and e.get("command") == cmd
                    for e in bucket
                ):
                    continue  # idempotent skip
                if len(bucket) == 0:
                    # Empty bucket (``event: []`` or bare ``event:``)
                    # — surgical edit isn't well-defined because there's
                    # no existing item line to anchor list_indent to,
                    # and the user's intent for the empty value is
                    # ambiguous. Refuse loudly. Removing the empty key
                    # by hand (so we install fresh) is the user fix.
                    raise SettingsCorruptError(
                        self.settings_path,
                        f"existing hooks[{event!r}] is an empty list; "
                        f"remove the key (or populate it) and re-run",
                    )
                event_key_line = hooks.lc.data[event][0]
                event_block_end = _block_end_excl(
                    lines, event_key_line, parent_col=mapping_indent
                )
                ops.append(
                    (event_block_end, [self._render_list_item(cmd, list_indent)])
                )
            else:
                raise SettingsCorruptError(
                    self.settings_path,
                    f"existing hooks[{event!r}] is "
                    f"{type(bucket).__name__}, not a sequence",
                )

        if not ops:
            return text

        # Apply in reverse line order so earlier ops don't shift
        # indices for later ones.
        ops.sort(key=lambda x: x[0], reverse=True)
        for line_idx, new_lines in ops:
            lines[line_idx:line_idx] = new_lines
        return "\n".join(lines)

    # --- unregister flow ---

    def _apply_unregister(self, text: str, doc: CommentedMap) -> str:
        hooks = doc.get("hooks")
        if not isinstance(hooks, CommentedMap):
            return text
        if hooks.fa.flow_style():
            # Don't try to surgically edit a flow-style hooks block —
            # we'd need to rewrite the line and that's exactly what
            # the parsing-only design forbids.
            return text

        lines = text.split("\n")
        mapping_indent = self._infer_mapping_indent(hooks)

        deletions: list[tuple[int, int]] = []
        events_emptied: set[str] = set()

        for event, hook_path in self._edits():
            bucket = hooks.get(event)
            if not isinstance(bucket, CommentedSeq):
                continue
            cmd = str(hook_path)
            our_indices = [
                i for i, e in enumerate(bucket)
                if isinstance(e, CommentedMap) and e.get("command") == cmd
            ]
            if not our_indices:
                continue
            if len(our_indices) == len(bucket):
                # Drop the whole event sub-block (last entry was ours).
                events_emptied.add(event)
                event_key_line = hooks.lc.data[event][0]
                event_block_end = _block_end_excl(
                    lines, event_key_line, parent_col=mapping_indent
                )
                deletions.append((event_key_line, event_block_end))
            else:
                # Keep the event, drop just our matching items.
                for i in our_indices:
                    item_start = bucket.lc.data[i][0]
                    # ruamel reports value-content col; dash is 2 cols
                    # left for ``- key: value`` form. Use the dash col
                    # as parent_col when walking forward, since the
                    # next sibling item's dash sits at the same col.
                    dash_col = max(0, bucket.lc.data[i][1] - 2)
                    if i + 1 < len(bucket):
                        item_end = bucket.lc.data[i + 1][0]
                    else:
                        item_end = _block_end_excl(
                            lines, item_start, dash_col
                        )
                    deletions.append((item_start, item_end))

        if not deletions:
            return text

        # If every event ends up emptied, the hooks block has no
        # surviving entries — drop the whole thing rather than
        # leaving an empty ``hooks:`` line behind.
        all_events = set(hooks.keys())
        if events_emptied >= all_events:
            hooks_key_line = doc.lc.data["hooks"][0]
            hooks_block_end = _block_end_excl(
                lines, hooks_key_line, parent_col=0
            )
            deletions = [(hooks_key_line, hooks_block_end)]

        # Apply in reverse line order.
        deletions.sort(key=lambda x: -x[0])
        for start, end in deletions:
            del lines[start:end]
        return "\n".join(lines)

    # --- rendering ---

    def _render_fresh_block(
        self, edits: list[tuple[str, Path]]
    ) -> list[str]:
        """Generate a complete new hooks block as a list of lines.
        Used for the no-hooks-yet and placeholder cases. PyYAML's
        default block style: mapping at 2-space indent, list items
        at 4-space indent."""
        out: list[str] = ["hooks:"]
        for event, hook_path in edits:
            out.extend(self._render_event_subblock(
                event,
                [str(hook_path)],
                self._DEFAULT_MAPPING_INDENT,
                self._DEFAULT_LIST_INDENT,
            ))
        return out

    def _render_event_subblock(
        self,
        event: str,
        commands: list[str],
        mapping_indent: int,
        list_indent: int,
    ) -> list[str]:
        out = [f"{' ' * mapping_indent}{event}:"]
        for cmd in commands:
            out.append(self._render_list_item(cmd, list_indent))
        return out

    def _render_list_item(self, cmd: str, list_indent: int) -> str:
        # Hook paths are absolute filesystem paths under ``~/.hermes/``,
        # plain ASCII with no internal whitespace, no quoting characters,
        # and well under 80 chars in practice. A bare plain scalar is
        # the right form — no need for double-quoted (which is what
        # lured ruamel into the wrap-corruption bug).
        return f"{' ' * list_indent}- command: {cmd}"

    def _infer_mapping_indent(self, hooks: CommentedMap) -> int:
        """Column of the first sub-key under ``hooks:``. The user's
        chosen mapping indent for this block. Falls back to default
        if the mapping is empty (which it isn't in this code path,
        but defensive)."""
        for key in hooks:
            return hooks.lc.data[key][1]
        return self._DEFAULT_MAPPING_INDENT

    def _infer_list_indent(
        self, hooks: CommentedMap, mapping_indent: int
    ) -> int:
        """Column of the first list item ``-`` in any of the hooks's
        sub-buckets. ruamel's ``bucket.lc.data[i]`` reports the col
        of the *value content* (after ``- ``), not the col of the
        dash — the dash is two cols to the left for the standard
        ``- key: value`` form. Subtract 2 to recover the dash col,
        which is what ``_render_list_item`` writes spaces for.

        Falls back to ``mapping_indent + 2`` so a non-default
        mapping indent gets a proportional list indent (matching
        PyYAML's offset-of-2)."""
        for key in hooks:
            bucket = hooks[key]
            if isinstance(bucket, CommentedSeq) and len(bucket) > 0:
                return max(0, bucket.lc.data[0][1] - 2)
        return mapping_indent + (
            self._DEFAULT_LIST_INDENT - self._DEFAULT_MAPPING_INDENT
        )
