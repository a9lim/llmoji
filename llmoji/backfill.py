"""One-shot backfill: replay native transcripts/rollouts into a
provider's kaomoji journal.

Each provider that exposes its history on disk gets a backfill
function that walks every session file, identifies kaomoji-bearing
assistant turns, and writes them into the same JSONL journal the
live hook appends to. The journal becomes the single source of
truth for kaomoji emission per provider.

  - Claude Code: ``~/.claude/projects/**/*.jsonl`` (per-event JSONL)
  - Codex:       ``~/.codex/sessions/**/rollout-*.jsonl[.zst]`` and
                  ``~/.codex/archived_sessions/`` (rollout JSONL; codex-cli
                  zstd-compresses older rollouts in place)
  - Hermes:      ``~/.hermes/sessions/session_*.json`` (whole-session JSON
                  with full ``messages`` list — same shape the live hook
                  receives via ``extra.conversation_history``). Hermes
                  stopped writing these snapshots in v0.21.1 (2026-09-07);
                  sessions since then live only in its ``state.db``, so
                  this replays history up to that release.
  - Grok:        ``~/.grok/sessions/<cwd>/<session>/updates.jsonl`` (ACP
                  update stream, the file the live hook reads)

Per-row schema (matches the bash hook templates under
``llmoji._hooks/``):

    {ts, model, cwd, kaomoji, user_text, assistant_text}

``ts`` carries the historical event timestamp (not "now"), so
backfilled rows are chronologically meaningful for trajectory
analyses. Rows are sorted by ``ts`` per journal before writing.
Hermes session files only carry session-level timestamps
(``session_start`` / ``last_updated``); the per-message structure
doesn't preserve turn time, so every row from one Hermes session
gets the same ``ts`` (= ``last_updated``). Cross-session ordering
still holds.

Re-runs OVERWRITE the journal — the writer truncates. Pause active
sessions during a backfill, otherwise an in-flight turn could land
in both the backfill (via transcript) and the live hook within the
same second. Sub-second-grade dedup is out of scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ._util import atomic_write_text, scrape_row_to_journal_line
from .providers import ClaudeCodeProvider, CodexProvider, GrokProvider, HermesProvider
from .scrape import ScrapeRow
from .sources._common import walk_parents_for_user_text
from .taxonomy import _KAOMOJI_MAX_LEN, is_kaomoji_candidate

_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _flush_rows(rows: Iterable[ScrapeRow], journal: Path) -> int:
    """Sort :class:`~llmoji.scrape.ScrapeRow` instances by timestamp
    and persist them as canonical 6-field JSONL via
    :func:`llmoji._util.scrape_row_to_journal_line`.

    Shared tail for every ``backfill_*`` function — three providers
    converge on the same write contract (chronological order, JSONL,
    truncate-on-write). Drift here is the failure mode this helper
    exists to prevent.
    """
    materialized = sorted(rows, key=lambda r: r.timestamp)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w") as f:
        for r in materialized:
            f.write(
                json.dumps(scrape_row_to_journal_line(r), ensure_ascii=False)
                + "\n"
            )
    return len(materialized)


def _zstd_decoder() -> Any:
    """A ``bytes -> bytes`` zstd decompressor from whatever the
    interpreter offers: the stdlib ``compression.zstd`` (Python 3.14+),
    else the ``zstandard`` package (``pip install 'llmoji[zstd]'``).
    ``None`` when neither is available."""
    try:
        from compression import zstd  # type: ignore[import-not-found]
    except ImportError:
        pass
    else:
        return zstd.decompress
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        return None

    def decompress(data: bytes) -> bytes:
        with zstandard.ZstdDecompressor().stream_reader(data) as reader:
            return reader.read()

    return decompress


def _read_session_text(path: Path) -> str | None:
    """Read a session file, decompressing ``.zst``. ``None`` when the
    file can't be read, or is compressed and no zstd decoder is
    installed — :func:`_unreadable_sources` reports the latter."""
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


def kaomoji_prefix(text: str) -> str:
    """Mirror the shell hook's extraction pipeline.

    Two-stage extraction matching ``_kaomoji_validate.sh.partial``:

      Stage 1 — strip from the first ASCII letter onward. Preserves
        bracket-leading kaomoji with internal whitespace
        (``(ง •̀_•́)``, ``(｡˃ ᵕ ˂)``) which a naive whitespace-split
        would clip.
      Stage 2 — fall back to whitespace-split when stage 1 yields
        empty. That happens when position 0 is itself an ASCII
        letter — the round-6 Path B bare-kaomoji case (``T-T``,
        ``XD``, ``Q_Q``, ``e_e``).

    Validates via :func:`~llmoji.taxonomy.is_kaomoji_candidate` and
    returns ``""`` for prose, markdown-escape artifacts, and other
    garbage.
    """
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line
            break
    if not first_line:
        return ""
    stripped = first_line.lstrip()
    # Stage 1: strip at first ASCII letter, then trim trailing space.
    m = _ASCII_LETTER_RE.search(stripped)
    cut = m.start() if m else len(stripped)
    prefix = stripped[:cut].rstrip()
    if not prefix:
        # Stage 2 fallback: whitespace-split for letter-eye bare kaomoji.
        idx = 0
        while idx < len(stripped) and not stripped[idx].isspace():
            idx += 1
        prefix = stripped[:idx]
    prefix = prefix[:_KAOMOJI_MAX_LEN]
    if not is_kaomoji_candidate(prefix):
        return ""
    return prefix


def as_shell_var(text: str) -> str:
    """``text`` as a live hook holds it after ``VAR=$(... | jq -r ...)``:
    command substitution drops every trailing newline. The replay
    applies the same rule to ``assistant_text`` and ``user_text`` so a
    live row and its replay compare equal — a message ending in a
    newline otherwise never dedups."""
    return text.rstrip("\n")


def strip_leading_kaomoji(text: str, kaomoji: str) -> str:
    """Drop the leading kaomoji + surrounding whitespace from ``text``.

    Mirrors the live hooks: the text arrives through command
    substitution (trailing newlines dropped, see :func:`as_shell_var`),
    then the journal-write jq applies ``sub("^\\s+"; "") |
    ltrimstr($kaomoji) | sub("^\\s+"; "")``. The ``kaomoji`` argument is
    what was already extracted from ``text``, so the prefix match is
    guaranteed.
    """
    stripped = as_shell_var(text).lstrip()
    if stripped.startswith(kaomoji):
        return stripped[len(kaomoji):].lstrip()
    return stripped


# ---------------------------------------------------------------------------
# Claude Code transcripts: ~/.claude/projects/**/*.jsonl
# ---------------------------------------------------------------------------


def _collect_assistant_text(message: dict[str, Any]) -> str:
    """Return the first text block of one Claude Code assistant entry.

    Each Claude Code assistant entry persists one content block —
    text, tool_use, or thinking — so "first text block" reduces to
    "the entry's text" for typical traffic. Multi-text-block entries
    are uncommon; if they happen, only the first text block is
    considered (matches the live hook).
    """
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text") or ""
                if txt:
                    return str(txt)
        return ""
    if isinstance(content, str):
        return content
    return ""


# Per-provider system-injection prefix list. Skill activations
# (Claude slash-commands) inject the skill body as a user-role
# message; those aren't real user input and would otherwise
# contaminate user_text → kaomoji-axis correlations.
#
# Single source of truth is the HookInstaller subclass attribute.
# The bash live hook gets the same list rendered into its jq filter
# via ``${INJECTED_PREFIXES_FILTER}`` at install time, so Python-side
# replay (here) and shell-side live capture cannot drift — both
# read from the same ClaudeCodeProvider class attribute.
_CLAUDE_CODE_INJECTED_PREFIXES: tuple[str, ...] = tuple(
    ClaudeCodeProvider.system_injected_prefixes
)


def _claude_code_text_extractor(ev: dict[str, Any]) -> str:
    """Extract the text payload of a Claude Code transcript user
    event. Strings pass through; list-shaped content blocks return
    the first non-empty ``"text"`` block. Non-text blocks (tool_use,
    tool_result, image) collapse to ``""`` so the parent walker
    keeps climbing past them.
    """
    m = ev.get("message", {})
    content = m.get("content") if isinstance(m, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                txt = b.get("text") or ""
                if txt.strip():
                    return str(txt)
    return ""


def _resolve_user_text_claude(
    start_uuid: str | None,
    by_uuid: dict[str, dict[str, Any]],
) -> str:
    """Claude Code parent-walk for the originating user prompt.

    Thin wrapper around the shared
    :func:`llmoji.sources._common.walk_parents_for_user_text`. Skips
    tool_result parents (they have ``type=="user"`` but content is
    machine-generated) and skill-injected content (slash-command
    activations dump the skill body as a user-role message). The
    hop budget is generous because a tool-heavy turn easily chains
    dozens of ``assistant tool_use → user tool_result`` pairs
    between the kaomoji-led text and the originating user prompt;
    pre-bump the cap was 5, which dropped ``user_text`` on ~17% of
    real-corpus rows.
    """
    return walk_parents_for_user_text(
        start_uuid,
        by_uuid,
        parent_field="parentUuid",
        role_check=lambda node: node.get("type") == "user",
        text_extractor=_claude_code_text_extractor,
        injected_prefixes=_CLAUDE_CODE_INJECTED_PREFIXES,
    )


def _replay_claude_transcript(path: Path) -> Iterator[ScrapeRow]:
    """Walk a Claude Code transcript JSONL and emit one journal row
    per kaomoji-led assistant text entry.

    Mirrors the live hook's per-entry walk: every text-bearing,
    non-sidechain assistant entry whose first text block leads with
    a valid kaomoji becomes its own row. A tool-heavy turn (text →
    tool_use → text → tool_use → text) emits up to one row per text
    block. Real user events delineate turns; each row's ``user_text``
    resolves to the originating prompt of its turn via the parent-
    Uuid walk, so all rows from one turn share the same
    ``user_text`` (the parentUuid chain leads back through every
    tool_result/assistant pair to the same originating user event).
    """
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    by_uuid = {ev["uuid"]: ev for ev in events if isinstance(ev.get("uuid"), str)}

    for ev in events:
        if ev.get("type") != "assistant":
            continue
        # Skip sidechain (subagent) turns. Agent-to-agent dispatches
        # are a different mode than human-driven sessions and would
        # otherwise contaminate user_text with subagent prompts.
        if ev.get("isSidechain"):
            continue
        m = ev.get("message", {})
        if not isinstance(m, dict):
            continue
        text = _collect_assistant_text(m)
        if not text.strip():
            # Tool-use-only / thinking-only entries have no text to
            # validate — skip without affecting any other entry.
            continue
        prefix = kaomoji_prefix(text)
        if not prefix:
            continue
        user_text = _resolve_user_text_claude(ev.get("parentUuid"), by_uuid)
        yield ScrapeRow(
            source="claude_code-backfill",
            model=str(m.get("model") or "") or None,
            timestamp=str(ev.get("timestamp") or ""),
            cwd=str(ev.get("cwd") or "") or None,
            first_word=prefix,
            assistant_text=strip_leading_kaomoji(text, prefix),
            surrounding_user=as_shell_var(user_text),
        )


def backfill_claude_code(transcript_root: Path, journal: Path) -> int:
    """Walk every transcript under ``transcript_root``, write
    chronologically-sorted journal rows to ``journal``. Returns row
    count.
    """
    rows: list[ScrapeRow] = []
    paths = sorted(transcript_root.rglob("*.jsonl"))
    for path in paths:
        rows.extend(_replay_claude_transcript(path))
    return _flush_rows(rows, journal)


# ---------------------------------------------------------------------------
# Codex rollouts: ~/.codex/sessions/**/rollout-*.jsonl
# ---------------------------------------------------------------------------


# Codex injects AGENTS.md, <environment_context>, and bare
# <INSTRUCTIONS> blocks as user-role response_items at session
# start. Drop them defensively.
#
# Same single-source-of-truth pattern as the Claude side above —
# the CodexProvider class attribute is canonical and the bash live
# hook gets it rendered into the jq filter at install time.
_CODEX_INJECTED_PREFIXES: tuple[str, ...] = tuple(
    CodexProvider.system_injected_prefixes
)


def _codex_is_subagent_rollout(events: list[dict[str, Any]]) -> bool:
    """A spawned subagent thread's rollout — ``session_meta`` says so
    via ``thread_source == "subagent"`` or a ``source`` object keyed
    ``subagent``. Mirrors the live hook's ``is_subagent`` check."""
    for ev in events:
        if ev.get("type") != "session_meta":
            continue
        p = ev.get("payload") or {}
        if not isinstance(p, dict):
            return False
        src = p.get("source")
        return p.get("thread_source") == "subagent" or (
            isinstance(src, dict) and "subagent" in src
        )
    return False


def _codex_agent_message_text(ev: dict[str, Any]) -> tuple[str, str] | None:
    """``(shape, text)`` if ``ev`` is an agent-message event, else
    ``None``. ``shape`` is ``"item"`` for the current codex-cli
    ``item_completed`` / ``AgentMessage`` event and ``"legacy"`` for
    the older ``agent_message`` event."""
    if ev.get("type") != "event_msg":
        return None
    p = ev.get("payload") or {}
    if not isinstance(p, dict):
        return None
    if p.get("type") == "agent_message":
        return "legacy", str(p.get("message") or "")
    if p.get("type") != "item_completed":
        return None
    item = p.get("item") or {}
    if not isinstance(item, dict) or item.get("type") != "AgentMessage":
        return None
    content = item.get("content") or []
    if not isinstance(content, list):
        return "item", ""
    return "item", "".join(
        str(b.get("text") or "")
        for b in content
        if isinstance(b, dict) and str(b.get("type") or "").lower() == "text"
    )


def _replay_codex_rollout(path: Path) -> Iterator[ScrapeRow]:
    """Walk a Codex rollout JSONL and emit one journal row per
    kaomoji-led agent message.

    Codex emits each model message as its own event — an
    ``item_completed`` ``AgentMessage`` on current codex-cli, an
    ``agent_message`` on older releases; a tool-heavy turn emits 5–10
    of these. Each ``turn_context`` opens a turn and supplies its
    ``model`` and ``cwd``. A turn is read in whichever message shape it
    carries, item_completed first, so a release that wrote both can't
    count a message twice — the live hook's rule. ``user_text`` is the
    latest real user response_item seen so far in the walk. Subagent
    threads are skipped whole, as the live hook skips them.
    """
    raw = _read_session_text(path)
    if raw is None:
        return
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if _codex_is_subagent_rollout(events):
        return

    session_cwd = ""
    for ev in events:
        if ev.get("type") == "session_meta":
            p = ev.get("payload") or {}
            session_cwd = str(p.get("cwd") or "")
            break

    # Split into turns at each turn_context. Events before the first
    # turn_context (session_meta, injected AGENTS.md items) form a
    # leading slice with no turn context of its own.
    turns: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [({}, [])]
    for ev in events:
        if ev.get("type") == "turn_context":
            p = ev.get("payload") or {}
            turns.append((p if isinstance(p, dict) else {}, []))
        turns[-1][1].append(ev)

    latest_user = ""
    for ctx, turn in turns:
        model = str(ctx.get("model") or "")
        cwd = str(ctx.get("cwd") or session_cwd)
        shapes = {
            m[0] for m in map(_codex_agent_message_text, turn) if m is not None
        }
        shape = "item" if "item" in shapes else "legacy"
        for ev in turn:
            if ev.get("type") == "response_item":
                p = ev.get("payload") or {}
                if p.get("role") == "user":
                    content = p.get("content") or []
                    if isinstance(content, list):
                        parts = [
                            str(b.get("text") or "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "input_text"
                        ]
                        txt = "\n".join(s for s in parts if s)
                        if txt and not txt.startswith(_CODEX_INJECTED_PREFIXES):
                            latest_user = txt
                continue
            msg = _codex_agent_message_text(ev)
            if msg is None or msg[0] != shape:
                continue
            text = msg[1]
            if not text:
                continue
            prefix = kaomoji_prefix(text)
            if not prefix:
                continue
            yield ScrapeRow(
                source="codex-backfill",
                model=model or None,
                timestamp=str(ev.get("timestamp") or ""),
                cwd=cwd or None,
                first_word=prefix,
                assistant_text=strip_leading_kaomoji(text, prefix),
                surrounding_user=as_shell_var(latest_user),
            )


def backfill_codex(rollouts_root: Path, journal: Path) -> int:
    rows: list[ScrapeRow] = []
    paths = sorted(rollouts_root.rglob("rollout-*.jsonl*"))
    for path in paths:
        rows.extend(_replay_codex_rollout(path))
    return _flush_rows(rows, journal)


# ---------------------------------------------------------------------------
# Hermes sessions: ~/.hermes/sessions/session_*.json
# ---------------------------------------------------------------------------


# Hermes' documented contract is ``extra.user_message`` arrives
# pre-injection, so the prefix list is empty. Mirrors
# :data:`HermesProvider.system_injected_prefixes`. Single source of
# truth on the HermesProvider class — if that ever populates, this
# picks up the change automatically.
_HERMES_INJECTED_PREFIXES: tuple[str, ...] = tuple(
    HermesProvider.system_injected_prefixes
)


def _replay_hermes_session(path: Path) -> Iterator[ScrapeRow]:
    """Walk one Hermes session JSON and emit one journal row per
    kaomoji-led assistant message, chunked by user-role boundaries.

    Hermes session files persist the cumulative ``messages`` list
    (overwritten each turn by ``_save_session_log`` in hermes-agent's
    run_agent.py), so by the time a backfill reads one, the file
    holds every turn's traffic in chronological order. Hermes' chat
    shape is linear — one user → N assistants/tools → one user →
    ... — so user-role indices delineate turns cleanly. For each
    consecutive (user_i, user_{i+1}] slice, walk every assistant-
    role message and emit one journal row per kaomoji-led one.

    Per-message timestamps aren't persisted in the session file, so
    every row from one session gets the same ``ts`` (=
    ``last_updated``). Cross-session ordering is preserved by the
    sort in :func:`backfill_hermes`. Within a session, rows preserve
    their order in ``messages`` (chronological by construction).
    """
    try:
        raw = path.read_text(errors="replace")
    except OSError:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    messages = data.get("messages")
    if not isinstance(messages, list):
        return

    model = str(data.get("model") or "")
    # Hermes session files don't persist cwd; the live hook reads
    # ``Path.cwd()`` at fire time. Backfilled rows therefore land
    # with cwd="" (the session file simply doesn't carry it). Same
    # degradation profile as a missing transcript field elsewhere.
    cwd = ""
    ts = str(data.get("last_updated") or data.get("session_start") or "")

    # Find every user-role message index — each delineates a turn.
    # The slice from one user-role index to the next is "that turn";
    # the last user index runs to the end of the array.
    user_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    if not user_indices:
        return

    boundaries = list(zip(user_indices, user_indices[1:] + [len(messages)]))
    for start, end in boundaries:
        user_msg = messages[start]
        if not isinstance(user_msg, dict):
            continue
        user_content = user_msg.get("content")
        if not isinstance(user_content, str):
            continue
        if user_content and not user_content.startswith(_HERMES_INJECTED_PREFIXES):
            user_text = user_content
        else:
            user_text = ""
        # Walk this turn's assistants — every assistant-role message
        # in ``messages[start:end]`` whose ``content`` is a non-empty
        # string. Tool-only assistant messages (carry ``tool_calls``
        # with empty/null ``content``) are skipped naturally.
        for msg in messages[start:end]:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            prefix = kaomoji_prefix(content)
            if not prefix:
                continue
            yield ScrapeRow(
                source="hermes-backfill",
                model=model or None,
                timestamp=ts,
                cwd=cwd or None,
                first_word=prefix,
                assistant_text=strip_leading_kaomoji(content, prefix),
                surrounding_user=as_shell_var(user_text),
            )


def backfill_hermes(sessions_root: Path, journal: Path) -> int:
    """Walk every ``session_*.json`` under ``sessions_root``, write
    chronologically-sorted journal rows to ``journal``. Returns row
    count.

    Sort key is ``last_updated``, so all rows from one session land
    contiguously and sessions order by completion time. Within one
    session, rows preserve their declared order in ``messages``
    (chronological by construction — the file is the agent's
    persisted message list).
    """
    rows: list[ScrapeRow] = []
    paths = sorted(sessions_root.glob("session_*.json"))
    for path in paths:
        rows.extend(_replay_hermes_session(path))
    return _flush_rows(rows, journal)


# ---------------------------------------------------------------------------
# Grok sessions: ~/.grok/sessions/<encoded cwd>/<session id>/updates.jsonl
# ---------------------------------------------------------------------------


def _grok_ms_to_iso(ms: Any) -> str:
    """``_meta.agentTimestampMs`` → second-precision ISO-8601 UTC, the
    same string jq's ``($ms / 1000 | floor | todate)`` gives the live
    hook. ``""`` when absent."""
    if not isinstance(ms, (int, float)):
        return ""
    return datetime.fromtimestamp(int(ms // 1000), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _grok_subagent_session_ids(sessions_root: Path) -> set[str]:
    """Session ids of every subagent child under ``sessions_root``.

    Grok keeps a child's own session in the normal sessions tree and
    records it under the parent's ``subagents/`` directory. A child's
    turns end in ``SubagentStop``, never ``Stop``, so the live hook
    never journals them; the replay skips them to match. The meta file
    schema isn't documented, so every directory name under
    ``subagents/`` and every string value in its ``*.json`` files is
    taken as a candidate id — only names of real session directories
    are ever compared against the set. The parent's own id is left
    out, since a child's meta may well record who spawned it.
    """
    ids: set[str] = set()
    for sub in sessions_root.glob("*/*/subagents"):
        found: set[str] = set()
        for entry in sub.rglob("*"):
            if entry.is_dir():
                found.add(entry.name)
            elif entry.suffix == ".json":
                try:
                    data = json.loads(entry.read_text(errors="replace"))
                except (OSError, json.JSONDecodeError):
                    continue
                stack: list[Any] = [data]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        stack.extend(node.values())
                    elif isinstance(node, list):
                        stack.extend(node)
                    elif isinstance(node, str):
                        found.add(node)
        found.discard(sub.parent.name)
        ids |= found
    return ids


def _replay_grok_session(path: Path) -> Iterator[ScrapeRow]:
    """Walk one Grok ``updates.jsonl`` and emit one journal row per
    kaomoji-led agent message.

    Mirrors the live hook's fold: ``agent_message_chunk`` updates
    carrying ``_meta.promptId`` belong to that prompt's turn, a run of
    consecutive chunks from one turn is one message, and any other
    update ends it. A turn's ``user_text`` is the latest
    ``user_message_chunk`` text seen before its first chunk. Its model
    is the dominant key of the turn's ``turn_completed.modelUsage``
    (output tokens), falling back to the latest ``model_id`` in the
    sibling ``chat_history.jsonl`` — the live hook reads the latter,
    since ``turn_completed`` lands after the Stop hook runs.
    """
    raw = _read_session_text(path)
    if raw is None:
        return
    updates: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            updates.append(row)

    session_dir = path.parent
    cwd = ""
    try:
        summary = json.loads((session_dir / "summary.json").read_text())
        info = summary.get("info") if isinstance(summary, dict) else None
        if isinstance(info, dict):
            cwd = str(info.get("cwd") or "")
    except (OSError, json.JSONDecodeError):
        pass

    fallback_model = ""
    chat = _read_session_text(session_dir / "chat_history.jsonl")
    for line in (chat or "").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("type") == "assistant" and row.get("model_id"):
            fallback_model = str(row["model_id"])

    # (pid, text, ts) per message, in stream order; user text per pid.
    messages: list[list[str]] = []
    turn_user: dict[str, str] = {}
    turn_model: dict[str, str] = {}
    user = ""
    ucur: str | None = None
    cur: list[str] | None = None
    for row in updates:
        params = row.get("params") or {}
        if not isinstance(params, dict):
            continue
        u = params.get("update") or {}
        meta = params.get("_meta") or {}
        if not isinstance(u, dict) or not isinstance(meta, dict):
            continue
        kind = u.get("sessionUpdate") or ""
        content = u.get("content")
        text = str(content.get("text") or "") if isinstance(content, dict) else ""
        pid = str(meta.get("promptId") or "")
        if kind == "agent_message_chunk" and pid:
            if cur is not None and cur[0] == pid:
                cur[1] += text
            else:
                if cur is not None:
                    messages.append(cur)
                turn_user.setdefault(pid, user)
                cur = [pid, text, _grok_ms_to_iso(meta.get("agentTimestampMs"))]
            ucur = None
            continue
        if cur is not None:
            messages.append(cur)
            cur = None
        if kind == "user_message_chunk":
            ucur = text if ucur is None else ucur + text
            user = ucur
            continue
        ucur = None
        if kind == "turn_completed":
            usage = (u.get("usage") or {}).get("modelUsage") or {}
            if isinstance(usage, dict) and usage:
                turn_model[str(u.get("prompt_id") or "")] = max(
                    usage,
                    key=lambda m: (
                        usage[m].get("outputTokens") or 0
                        if isinstance(usage[m], dict) else 0
                    ),
                )
    if cur is not None:
        messages.append(cur)

    for pid, text, ts in messages:
        if not text:
            continue
        prefix = kaomoji_prefix(text)
        if not prefix:
            continue
        yield ScrapeRow(
            source="grok-backfill",
            model=turn_model.get(pid) or fallback_model or None,
            timestamp=ts,
            cwd=cwd or None,
            first_word=prefix,
            assistant_text=strip_leading_kaomoji(text, prefix),
            surrounding_user=as_shell_var(turn_user.get(pid, "")),
        )


def _grok_session_paths(sessions_root: Path) -> list[Path]:
    """Every main-agent ``updates.jsonl`` under ``sessions_root``,
    sorted — subagent children excluded (see
    :func:`_grok_subagent_session_ids`)."""
    children = _grok_subagent_session_ids(sessions_root)
    return [
        p for p in sorted(sessions_root.glob("*/*/updates.jsonl"))
        if p.parent.name not in children
    ]


def backfill_grok(sessions_root: Path, journal: Path) -> int:
    """Walk every main-agent session under ``sessions_root``, write
    chronologically-sorted journal rows to ``journal``. Returns row
    count."""
    rows: list[ScrapeRow] = []
    for path in _grok_session_paths(sessions_root):
        rows.extend(_replay_grok_session(path))
    return _flush_rows(rows, journal)


# ---------------------------------------------------------------------------
# `llmoji import` — dedup-aware merge into the live journal
# ---------------------------------------------------------------------------
#
# The ``backfill_*`` functions above truncate the journal: they're a
# pre-install one-shot intended to be run before the live hook starts
# logging, so a re-run is fine to overwrite.
#
# ``import_provider`` is the user-facing equivalent that's safe to run
# AFTER install: it walks the same source files, then dedups against
# whatever's already in the journal (live-hook rows + any prior
# imports) and APPENDS only novel rows. Atomic via temp-write +
# os.replace so a SIGINT mid-write leaves the journal with either
# the old content or the merged content, never half.
#
# Dedup key: hash of (ts, model, assistant_text). ``ts`` is set per
# event in claude_code/codex/grok sources and per session in hermes
# (whose session JSON only persists session-level timestamps);
# collisions inside one session for hermes are intentional — same
# key, same row, dedup folds them.
#
# Live rows written before 2.2 carry the hook's fire time instead of
# the message's own time, so they never share a key with the replayed
# copy of the same message. :func:`_LegacyRowPool` folds those by
# (model, assistant_text) inside a time window, one live row per
# replayed row.


@dataclass
class ImportResult:
    """One ``import_provider`` invocation's tally.

    ``rows_seen`` is everything the source files contained (after
    ``--since`` filtering). ``rows_novel`` is what made it past the
    dedup check and got appended. ``rows_seen - rows_novel`` is the
    dedup hit count — typically every row on a re-import.
    ``unreadable`` counts source files skipped because they're
    zstd-compressed and no decoder is installed.
    """

    rows_seen: int
    rows_novel: int
    unreadable: int = 0


def _dedup_key_for_journal_row(
    ts: str, model: str, assistant_text: str,
) -> str:
    """Hash key for journal-row dedup. Length-prefixed framing so a
    NUL inside any field can't collapse field boundaries (mirrors
    :func:`llmoji.synth.cache_key`'s framing rule).
    """
    h = hashlib.sha256()
    for part in (ts, model, assistant_text):
        encoded = part.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
    return h.hexdigest()[:16]


def _iter_journal_rows(journal_path: Path) -> Iterator[dict[str, Any]]:
    """Existing journal rows as dicts. Tolerates malformed lines
    (skips silently — analyze re-walks the journal anyway and surfaces
    malformed rows there)."""
    if not journal_path.exists():
        return
    with journal_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _journal_dedup_keys(journal_path: Path) -> set[str]:
    """Read existing journal rows and build the dedup-key set."""
    return {
        _dedup_key_for_journal_row(
            str(row.get("ts") or ""),
            str(row.get("model") or ""),
            str(row.get("assistant_text") or ""),
        )
        for row in _iter_journal_rows(journal_path)
    }


def _parse_ts(ts: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# How far after a message a pre-2.2 live hook could have stamped it:
# the hook ran at turn end, and long agentic turns run for hours.
_LEGACY_FIRE_WINDOW = timedelta(hours=24)
# A fire time is never before its message, but the hook truncated to
# whole seconds while transcripts keep milliseconds.
_LEGACY_FIRE_SLACK = timedelta(seconds=5)


# The only shape a pre-2.2 live hook wrote: ``date -u
# +%Y-%m-%dT%H:%M:%SZ``, whole seconds. Rows a replay wrote keep the
# transcript's milliseconds and are never fold candidates.
_FIRE_TIME_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class _LegacyRowPool:
    """Existing journal rows that a replayed row might duplicate under
    a different timestamp.

    Before 2.2 the live hooks stamped hook fire time, so the journal
    holds a message under one ``ts`` and its replay under another.
    Exact-key dedup misses that pair, and every ``llmoji import``
    after install counted the live-captured messages again. The pool
    matches a replayed row to an unclaimed existing row in fire-time
    format with the same model, kaomoji and text whose ``ts`` falls in
    ``[ts - slack, ts + window]``. Each existing row absorbs at most one
    replayed row, exact or folded, so two genuine repeats of a short
    message both land when both have their own live rows.

    It's a heuristic and can miscount in one direction: when a live
    row's own message is no longer replayable (its transcript was
    deleted) and an identical message — same model, face and text —
    sits up to a day before it, the earlier one folds into that row and
    one message goes uncounted. ``cwd`` and ``user_text`` would narrow
    that further but cost real folds: the pre-2.2 hooks and the replay
    resolve them differently often enough to matter (22 and 35 of 885
    on one real journal).

    Callers claim every exact-key match first (:meth:`claim_exact`)
    and only then try :meth:`claim_legacy` on what's left, both over
    the whole replay set; otherwise a fuzzy claim could take the row a
    later replayed row matches exactly, and drop a real message.
    """

    def __init__(self, journal_path: Path) -> None:
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._fold: dict[tuple[str, str, str], list[tuple[datetime, int]]] = defaultdict(list)
        self._claimed: set[int] = set()
        for i, row in enumerate(_iter_journal_rows(journal_path)):
            ts = str(row.get("ts") or "")
            model = str(row.get("model") or "")
            text = str(row.get("assistant_text") or "")
            self._exact[_dedup_key_for_journal_row(ts, model, text)].append(i)
            parsed = _parse_ts(ts) if _FIRE_TIME_TS_RE.match(ts) else None
            if parsed is not None:
                kaomoji = str(row.get("kaomoji") or "")
                self._fold[(model, kaomoji, text)].append((parsed, i))
        for candidates in self._fold.values():
            candidates.sort()

    def claim_exact(self, ts: str, model: str, text: str) -> bool:
        for i in self._exact.get(_dedup_key_for_journal_row(ts, model, text), ()):
            if i not in self._claimed:
                self._claimed.add(i)
                return True
        return False

    def claim_legacy(self, ts: str, model: str, kaomoji: str, text: str) -> bool:
        msg_time = _parse_ts(ts)
        if msg_time is None:
            return False
        lo = msg_time - _LEGACY_FIRE_SLACK
        hi = msg_time + _LEGACY_FIRE_WINDOW
        for t, i in self._fold.get((model, kaomoji, text), ()):
            if i not in self._claimed and lo <= t <= hi:
                self._claimed.add(i)
                return True
        return False


# Source-path resolvers for each provider. Single source of truth for
# "where on disk do this provider's session/transcript files live?"
# — the existing ``backfill_*`` callers pass the root in, but the CLI
# ``import`` path resolves via the provider class, so the convention
# pins here.
_PROVIDER_SOURCE_GLOBS: dict[str, tuple[type, tuple[tuple[str, str], ...]]] = {
    # provider_class, ((root_subpath_under_home_dir, glob_pattern), ...)
    "claude_code": (ClaudeCodeProvider, (("projects", "**/*.jsonl"),)),
    "codex": (CodexProvider, (
        ("sessions", "**/rollout-*.jsonl"),
        ("sessions", "**/rollout-*.jsonl.zst"),
        ("archived_sessions", "rollout-*.jsonl"),
        ("archived_sessions", "rollout-*.jsonl.zst"),
    )),
    "hermes": (HermesProvider, (("sessions", "session_*.json"),)),
    "grok": (GrokProvider, (("sessions", "*/*/updates.jsonl"),)),
}

# Public-ish view of which providers ``import_provider`` can replay
# from native session files. The TS-plugin providers (opencode,
# openclaw) aren't here — their hosts don't persist a transcript-on-
# disk shape we can replay, so live-hook capture is the only path
# for those. Antigravity isn't either: its transcripts carry no
# per-message model id (that lives in protobuf blobs in a SQLite
# store), and every bundle row is keyed by model. CLI uses this to
# (a) tighten ``llmoji import`` 's ``choices=`` to providers we can
# actually serve, and (b) drive the no-arg autodetect path.
IMPORTABLE_PROVIDERS: tuple[str, ...] = tuple(_PROVIDER_SOURCE_GLOBS)


def _source_paths(name: str) -> list[Path]:
    """Every source file for ``name``, in replay order: each
    ``(subpath, pattern)`` root in turn, sorted within it. Grok's
    subagent child sessions are left out (see
    :func:`_grok_session_paths`)."""
    if name not in _PROVIDER_SOURCE_GLOBS:
        raise ValueError(
            f"unknown provider {name!r}; "
            f"known: {sorted(_PROVIDER_SOURCE_GLOBS)}"
        )
    cls, sources = _PROVIDER_SOURCE_GLOBS[name]
    home = cls().home_dir
    if name == "grok":
        root = home / sources[0][0]
        return _grok_session_paths(root) if root.exists() else []
    paths: list[Path] = []
    for subpath, pattern in sources:
        root = home / subpath
        if root.exists():
            paths.extend(sorted(root.glob(pattern)))
    return paths


def _unreadable_sources(name: str) -> int:
    """How many of ``name``'s source files are zstd-compressed with no
    decoder installed — skipped by the replay, reported by the CLI."""
    if _zstd_decoder() is not None:
        return 0
    return sum(1 for p in _source_paths(name) if p.suffix == ".zst")


_REPLAYERS: dict[str, Any] = {
    "claude_code": _replay_claude_transcript,
    "codex": _replay_codex_rollout,
    "hermes": _replay_hermes_session,
    "grok": _replay_grok_session,
}


def _iter_rows_for_provider(name: str) -> Iterator[ScrapeRow]:
    """Walk the canonical source paths for ``name`` and yield every
    :class:`ScrapeRow` the corresponding ``backfill_*`` function would
    produce. Skips cleanly when a source root doesn't exist (a provider
    that's been installed but has no session files yet).
    """
    paths = _source_paths(name)  # validates ``name``
    replay = _REPLAYERS[name]
    for path in paths:
        yield from replay(path)


def _journal_for(name: str) -> Path:
    """Resolve the journal path for ``name`` via the provider class
    — single source of truth, mirrors what the live hook writes to."""
    if name not in _PROVIDER_SOURCE_GLOBS:
        raise ValueError(
            f"unknown provider {name!r}; "
            f"known: {sorted(_PROVIDER_SOURCE_GLOBS)}"
        )
    cls, _ = _PROVIDER_SOURCE_GLOBS[name]
    return cls().journal_path


def import_provider(
    name: str,
    *,
    since: str | None = None,
    dry_run: bool = False,
) -> ImportResult:
    """Replay one provider's session/transcript files into its journal,
    deduplicating against the existing journal so re-runs are
    idempotent.

    ``since``: ISO-8601 timestamp string; rows with ``ts < since`` are
    neither counted nor appended. Comparison is lexicographic on the
    raw timestamp strings, which is correct for ISO-8601-with-Z (the
    only format the sources emit). Dedup still runs over every replayed
    row, so an older message keeps its claim on the journal row that
    is its own and can't leave it for a newer one to fold into.

    ``dry_run``: walk + dedup but don't write the journal. Returned
    ``ImportResult`` carries the same counts the real run would, so
    a script can preview before committing.

    Stop the harness before importing — concurrent live-hook writes
    during the temp-file dance could be lost in the rename. A file
    lock isn't worth the complexity for a stop-the-world recovery
    operation.
    """
    journal_path = _journal_for(name)
    pool = _LegacyRowPool(journal_path)

    replayed = list(_iter_rows_for_provider(name))

    def in_window(r: ScrapeRow) -> bool:
        return since is None or r.timestamp >= since

    rows_seen = sum(1 for r in replayed if in_window(r))
    # Exact matches first, then the pre-2.2 fire-time fold over what's
    # left, in timestamp order (see _LegacyRowPool) — both over every
    # replayed row; ``since`` only gates what's appended. A replayed
    # row whose key is already taken — by a journal row or an earlier
    # replayed copy — is a duplicate of it and never reaches the fold.
    taken: set[str] = set()
    unmatched: list[ScrapeRow] = []
    for r in replayed:
        model = r.model or ""
        text = r.assistant_text or ""
        key = _dedup_key_for_journal_row(r.timestamp, model, text)
        if pool.claim_exact(r.timestamp, model, text) or key in taken:
            taken.add(key)
            continue
        taken.add(key)
        unmatched.append(r)
    novel: list[ScrapeRow] = []
    for r in sorted(unmatched, key=lambda r: r.timestamp):
        if pool.claim_legacy(r.timestamp, r.model or "", r.first_word, r.assistant_text or ""):
            continue
        if in_window(r):
            novel.append(r)
    unreadable = _unreadable_sources(name)

    if dry_run or not novel:
        return ImportResult(
            rows_seen=rows_seen, rows_novel=len(novel), unreadable=unreadable,
        )

    # Atomic merge: read existing text (if any), append novel rows
    # sorted by timestamp, write the whole concatenation via
    # tmp+rename.
    existing_text = (
        journal_path.read_text() if journal_path.exists() else ""
    )
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    new_lines = []
    for r in sorted(novel, key=lambda r: r.timestamp):
        new_lines.append(
            json.dumps(scrape_row_to_journal_line(r), ensure_ascii=False)
        )
    new_text = existing_text + "\n".join(new_lines) + "\n"
    atomic_write_text(journal_path, new_text)

    return ImportResult(
        rows_seen=rows_seen, rows_novel=len(novel), unreadable=unreadable,
    )
