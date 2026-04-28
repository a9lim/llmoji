"""One-shot backfill: replay native transcripts/rollouts into a
provider's kaomoji journal.

Each provider that exposes its history as on-disk JSONL transcripts
(Claude Code under ``~/.claude/projects/**``, Codex under
``~/.codex/sessions/**``) gets a backfill function that walks every
transcript file, identifies kaomoji-bearing assistant turns, and
writes them into the same JSONL journal the live Stop hook
appends to. The journal becomes the single source of truth for
kaomoji emission per provider.

Per-row schema (matches the bash hook templates under
``llmoji._hooks/``):

    {ts, model, cwd, kaomoji, user_text, assistant_text}

``ts`` carries the historical event timestamp (not "now"), so
backfilled rows are chronologically meaningful for trajectory
analyses. Rows are sorted by ``ts`` per journal before writing.

Re-runs OVERWRITE the journal — the writer truncates. Pause active
sessions during a backfill, otherwise an in-flight turn could land
in both the backfill (via transcript) and the live hook within the
same second. Sub-second-grade dedup is out of scope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .providers import ClaudeCodeProvider, CodexProvider
from .taxonomy import is_kaomoji_candidate


def kaomoji_prefix(text: str) -> str:
    """Mirror the shell hook's awk + sed pipeline.

    Take the first non-empty line, strip leading whitespace, drop
    everything from the first ASCII letter, trim trailing whitespace,
    then validate via :func:`~llmoji.taxonomy.is_kaomoji_candidate`
    (≥2 bytes, ≤32 bytes, starts with ``KAOMOJI_START_CHARS``, no
    backslash, no 4+-letter run, balanced brackets if applicable).
    Returns ``""`` for prose, markdown-escape artifacts, and other
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
    cut = next((i for i, c in enumerate(stripped) if "a" <= c.lower() <= "z"), len(stripped))
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
    """Return the first text block of a Claude Code assistant message.

    Claude Code assistant turns can interleave ``text + tool_use +
    text``; the kaomoji-prefixed response is always the FIRST text
    block, and subsequent text is post-tool-call continuation —
    irrelevant to kaomoji analysis. Matches the live hook.
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


def _resolve_user_text_claude(
    start_uuid: str | None,
    by_uuid: dict[str, dict[str, Any]],
    max_hops: int = 1000,
) -> str:
    """Walk parentUuid backward to the nearest human-typed user text.

    Skips tool_result parents (they have type=='user' but content is
    machine-generated) AND skill-injected content (slash-command
    activations dump the skill body as a user-role message).

    The hop budget is generous because a tool-heavy turn easily
    chains dozens of `assistant tool_use → user tool_result` pairs
    between the kaomoji-led text and the originating user prompt.
    Pre-bump the cap was 5, which dropped `user_text` on ~17% of
    real-corpus rows. The cap exists only to bound a pathological
    uuid cycle — anything higher than the longest plausible turn
    is fine.
    """
    uuid = start_uuid
    for _ in range(max_hops):
        if uuid is None:
            return ""
        ev = by_uuid.get(uuid)
        if ev is None:
            return ""
        if ev.get("type") == "user":
            m = ev.get("message", {})
            content = m.get("content") if isinstance(m, dict) else None
            text = ""
            if isinstance(content, str) and content.strip():
                text = content
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        txt = b.get("text") or ""
                        if txt.strip():
                            text = txt
                            break
            if text and not text.startswith(_CLAUDE_CODE_INJECTED_PREFIXES):
                return text
        uuid = ev.get("parentUuid")
    return ""


def _is_real_user_event(ev: dict[str, Any]) -> bool:
    """Real user event = content is a string OR a content-block array
    that is NOT a tool_result delivery. Tool-result user events don't
    start a new conversational turn — they're machine-generated
    interleaving inside one.
    """
    if ev.get("type") != "user":
        return False
    m = ev.get("message")
    if not isinstance(m, dict):
        return False
    c = m.get("content")
    if isinstance(c, str):
        return True
    if isinstance(c, list):
        return not any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in c
        )
    return False


def _replay_claude_transcript(path: Path) -> Iterator[dict[str, Any]]:
    """Walk a Claude Code transcript JSONL and emit at most one
    journal row per turn.

    Mirrors the live hook's first-text-since-latest-user logic:
    each real user prompt opens a turn, and within a turn the first
    text-bearing non-sidechain assistant entry is taken as the
    kaomoji-led reply. The naive "iterate every assistant entry"
    pattern would emit one row per text block, which over-counts
    turns where the model writes preamble + post-tool text in
    separate entries.
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

    emitted_for_turn = False
    for ev in events:
        if _is_real_user_event(ev):
            # New turn boundary — reset the per-turn flag so the
            # next text-bearing assistant entry can emit.
            emitted_for_turn = False
            continue
        if ev.get("type") != "assistant":
            continue
        # Skip sidechain (subagent) turns. Agent-to-agent dispatches
        # are a different mode than human-driven sessions and would
        # otherwise contaminate user_text with subagent prompts.
        if ev.get("isSidechain"):
            continue
        if emitted_for_turn:
            continue
        m = ev.get("message", {})
        if not isinstance(m, dict):
            continue
        text = _collect_assistant_text(m)
        if not text.strip():
            # Tool-use-only or thinking-only entries don't close out
            # the turn's emission slot — keep scanning forward for
            # the first real text in this turn.
            continue
        # First text-bearing entry of the turn — this is our shot.
        # Whether or not the kaomoji filter passes, mark the slot
        # consumed so a later text block in the same turn (e.g. a
        # post-tool follow-up) doesn't double-emit.
        emitted_for_turn = True
        prefix = kaomoji_prefix(text)
        if not prefix:
            continue
        user_text = _resolve_user_text_claude(ev.get("parentUuid"), by_uuid)
        yield {
            "ts": str(ev.get("timestamp") or ""),
            "model": str(m.get("model") or ""),
            "cwd": str(ev.get("cwd") or ""),
            "kaomoji": prefix,
            "user_text": user_text,
            "assistant_text": strip_leading_kaomoji(text, prefix),
        }


def backfill_claude_code(transcript_root: Path, journal: Path) -> int:
    """Walk every transcript under ``transcript_root``, write
    chronologically-sorted journal rows to ``journal``. Returns row
    count.
    """
    rows: list[dict[str, Any]] = []
    paths = sorted(transcript_root.rglob("*.jsonl"))
    for path in paths:
        rows.extend(_replay_claude_transcript(path))
    rows.sort(key=lambda r: r["ts"])
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


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

    # Walk in order; track latest user text; emit on task_complete.
    # Codex puts the kaomoji on the LAST agent message (per turn,
    # task_complete.last_agent_message). Progress messages go first.
    latest_user = ""
    for ev in events:
        t = ev.get("type")
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
        elif t == "event_msg":
            p = ev.get("payload") or {}
            if p.get("type") != "task_complete":
                continue
            last_agent = str(p.get("last_agent_message") or "")
            if not last_agent:
                continue
            prefix = kaomoji_prefix(last_agent)
            if not prefix:
                continue
            ctx = turn_ctx.get(str(p.get("turn_id") or ""), {})
            ts = ev.get("timestamp") or p.get("completed_at") or ""
            yield {
                "ts": str(ts),
                "model": ctx.get("model", ""),
                "cwd": ctx.get("cwd", session_cwd),
                "kaomoji": prefix,
                "user_text": latest_user,
                "assistant_text": strip_leading_kaomoji(last_agent, prefix),
            }


def backfill_codex(rollouts_root: Path, journal: Path) -> int:
    rows: list[dict[str, Any]] = []
    paths = sorted(rollouts_root.rglob("rollout-*.jsonl"))
    for path in paths:
        rows.extend(_replay_codex_rollout(path))
    rows.sort(key=lambda r: r["ts"])
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
