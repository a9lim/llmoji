"""One-shot backfill: replay native transcripts/rollouts into a
provider's kaomoji journal.

Each provider that exposes its history on disk gets a backfill
function that walks every session file, identifies kaomoji-bearing
assistant turns, and writes them into the same JSONL journal the
live hook appends to. The journal becomes the single source of
truth for kaomoji emission per provider.

  - Claude Code: ``~/.claude/projects/**/*.jsonl`` (per-event JSONL)
  - Codex:       ``~/.codex/sessions/**/rollout-*.jsonl`` (rollout JSONL)
  - Hermes:      ``~/.hermes/sessions/session_*.json`` (whole-session JSON
                  with full ``messages`` list — same shape the live hook
                  receives via ``extra.conversation_history``)

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

import json
import re
from pathlib import Path
from typing import Any, Iterator

from ._util import journal_line_dict
from .providers import ClaudeCodeProvider, CodexProvider, HermesProvider
from .sources._common import walk_parents_for_user_text
from .taxonomy import is_kaomoji_candidate

_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _flush_rows(rows: list[dict[str, Any]], journal: Path) -> int:
    """Sort rows by ``ts`` and write them as JSONL to ``journal``.

    Shared tail for every ``backfill_*`` function — three providers
    converge on the same write contract (chronological order, JSONL,
    truncate-on-write). Drift here is the failure mode this helper
    exists to prevent.
    """
    rows.sort(key=lambda r: r["ts"])
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def kaomoji_prefix(text: str) -> str:
    """Mirror the shell hook's awk + sed pipeline.

    Take the first non-empty line, strip leading whitespace, drop
    everything from the first ASCII letter, trim trailing whitespace,
    then validate via :func:`~llmoji.taxonomy.is_kaomoji_candidate`
    (≥2 bytes, ≤32 bytes, starts with ``KAOMOJI_START_CHARS``, no
    backslash, no 4+-letter run). Returns ``""`` for prose,
    markdown-escape artifacts, and other garbage.
    """
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line
            break
    if not first_line:
        return ""
    stripped = first_line.lstrip()
    m = _ASCII_LETTER_RE.search(stripped)
    cut = m.start() if m else len(stripped)
    prefix = stripped[:cut].rstrip()
    if not is_kaomoji_candidate(prefix):
        return ""
    return prefix


def strip_leading_kaomoji(text: str, kaomoji: str) -> str:
    """Drop the leading kaomoji + surrounding whitespace from ``text``.

    Mirrors the jq pipeline both Stop hooks apply to ``assistant_text``:
    ``sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")``. The
    ``kaomoji`` argument is what was already extracted from ``text``,
    so the prefix match is guaranteed.
    """
    stripped = text.lstrip()
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
# Single source of truth is the Provider class attribute. The bash
# live hook gets the same list rendered into its jq filter via
# ``${INJECTED_PREFIXES_FILTER}`` at install time, so Python-side
# replay (here) and shell-side live capture cannot drift — both
# read from the Provider class.
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


def _replay_claude_transcript(path: Path) -> Iterator[dict[str, Any]]:
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
        yield journal_line_dict(
            ts=str(ev.get("timestamp") or ""),
            model=str(m.get("model") or ""),
            cwd=str(ev.get("cwd") or ""),
            kaomoji=prefix,
            user_text=user_text,
            assistant_text=strip_leading_kaomoji(text, prefix),
        )


def backfill_claude_code(transcript_root: Path, journal: Path) -> int:
    """Walk every transcript under ``transcript_root``, write
    chronologically-sorted journal rows to ``journal``. Returns row
    count.
    """
    rows: list[dict[str, Any]] = []
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
# the Provider class attribute is canonical and the bash live hook
# gets it rendered into the jq filter at install time.
_CODEX_INJECTED_PREFIXES: tuple[str, ...] = tuple(
    CodexProvider.system_injected_prefixes
)


def _replay_codex_rollout(path: Path) -> Iterator[dict[str, Any]]:
    """Walk a Codex rollout JSONL and emit one journal row per
    kaomoji-led ``event_msg.agent_message`` event.

    Codex emits each model message as its own ``agent_message``
    event (``payload.message`` carries the text); a tool-heavy turn
    emits 5–10 of these. Pre-fix this only emitted on
    ``task_complete.last_agent_message``, dropping every kaomoji-led
    progress message. Per-turn invariants (``model``, ``cwd``,
    ``user_text``) are tracked via ``turn_context`` and user
    response_items as we walk; ``current_turn_id`` is the most
    recent turn_context's id and applies to every subsequent
    agent_message until the next turn_context.
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

    # Index turn_context by turn_id for model + cwd lookup.
    turn_ctx: dict[str, dict[str, str]] = {}
    session_cwd = ""
    for ev in events:
        if ev.get("type") == "session_meta":
            p = ev.get("payload") or {}
            session_cwd = str(p.get("cwd") or "")
        elif ev.get("type") == "turn_context":
            p = ev.get("payload") or {}
            tid = p.get("turn_id")
            if tid:
                turn_ctx[str(tid)] = {
                    "model": str(p.get("model") or ""),
                    "cwd": str(p.get("cwd") or session_cwd),
                }

    # Walk events in order. Maintain rolling state for the current
    # turn (turn_id, latest user_text); emit per agent_message.
    # ``agent_message`` events don't carry turn_id themselves, so we
    # use the last turn_context's id as a proxy — fine because
    # rollouts are chronological and turn_context fires at turn
    # start.
    current_turn_id = ""
    latest_user = ""
    for ev in events:
        t = ev.get("type")
        if t == "turn_context":
            p = ev.get("payload") or {}
            current_turn_id = str(p.get("turn_id") or "")
            continue
        if t == "response_item":
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
        if t != "event_msg":
            continue
        p = ev.get("payload") or {}
        if p.get("type") != "agent_message":
            continue
        text = str(p.get("message") or "")
        if not text:
            continue
        prefix = kaomoji_prefix(text)
        if not prefix:
            continue
        ctx = turn_ctx.get(current_turn_id, {})
        ts = ev.get("timestamp") or ""
        yield journal_line_dict(
            ts=str(ts),
            model=ctx.get("model", ""),
            cwd=ctx.get("cwd", session_cwd),
            kaomoji=prefix,
            user_text=latest_user,
            assistant_text=strip_leading_kaomoji(text, prefix),
        )


def backfill_codex(rollouts_root: Path, journal: Path) -> int:
    rows: list[dict[str, Any]] = []
    paths = sorted(rollouts_root.rglob("rollout-*.jsonl"))
    for path in paths:
        rows.extend(_replay_codex_rollout(path))
    return _flush_rows(rows, journal)


# ---------------------------------------------------------------------------
# Hermes sessions: ~/.hermes/sessions/session_*.json
# ---------------------------------------------------------------------------


# Hermes' documented contract is ``extra.user_message`` arrives
# pre-injection, so the prefix list is empty. Mirrors
# :data:`HermesProvider.system_injected_prefixes`. Single source of
# truth on the Provider class — if that ever populates, this picks up
# the change automatically.
_HERMES_INJECTED_PREFIXES: tuple[str, ...] = tuple(
    HermesProvider.system_injected_prefixes
)


def _replay_hermes_session(path: Path) -> Iterator[dict[str, Any]]:
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
            yield journal_line_dict(
                ts=ts,
                model=model,
                cwd=cwd,
                kaomoji=prefix,
                user_text=user_text,
                assistant_text=strip_leading_kaomoji(content, prefix),
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
    rows: list[dict[str, Any]] = []
    paths = sorted(sessions_root.glob("session_*.json"))
    for path in paths:
        rows.extend(_replay_hermes_session(path))
    return _flush_rows(rows, journal)
