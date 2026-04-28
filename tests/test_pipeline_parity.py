"""Cross-validate the bash live-hook pipeline against the Python
backfill pipeline.

Both sides extract a kaomoji prefix from the assistant message and
strip it from ``assistant_text`` to land in the canonical 6-field
journal row. The bash side does it in-template with awk + sed + jq;
the Python side does it with :func:`~llmoji.backfill.kaomoji_prefix`
and :func:`~llmoji.backfill.strip_leading_kaomoji`. CLAUDE.md flags
these as load-bearing parallel implementations: bash for live perf
(no Python startup tax per turn), Python for replay perf (no
fork-per-row tax over thousands of historical turns).

This test fires the *actual rendered hook* against fabricated
transcripts and asserts the resulting journal row matches what
:func:`backfill_claude_code` / :func:`backfill_codex` produce on the
same input. Catches drift between the awk/sed/jq pipeline and the
Python helpers — the gap left open by
``test_hook_templates_render_to_valid_bash_substitutions``, which
only verifies bash parses, not that bash and Python agree on
outputs.

Skipped if ``bash`` or ``jq`` is not on PATH (the pipeline depends
on both). On macOS / Linux dev boxes both are universally available;
in stripped CI images either may be missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

# Every test in this module forks bash + jq subprocesses, which adds
# real wallclock time on local dev runs. CI keeps running the full
# suite; locally, ``pytest -m "not slow"`` skips the parity gate when
# iterating on something unrelated.
pytestmark = pytest.mark.slow

# Fields the journal row is built from. We compare these between the
# bash hook output and the backfill output. ``ts`` differs by design
# (bash stamps now, backfill stamps historical) so it's excluded.
_PARITY_FIELDS = ("kaomoji", "model", "cwd", "user_text", "assistant_text")

# Hermes-specific: the live hook reads ``cwd`` from the post_llm_call
# payload (= ``Path.cwd()`` of the agent process at fire time); the
# backfill side reads from ``~/.hermes/sessions/session_*.json``,
# which doesn't persist cwd at all (the field simply isn't in the
# session schema; see ``_save_session_log`` in
# ``hermes-agent/run_agent.py``). Backfilled rows therefore land with
# cwd="" by design. The two are divergent on cwd by source design,
# not bug.
_HERMES_PARITY_FIELDS = tuple(f for f in _PARITY_FIELDS if f != "cwd")


def _require_tools() -> tuple[str, str]:
    bash = shutil.which("bash")
    jq = shutil.which("jq")
    if not bash or not jq:
        pytest.skip("bash + jq required for pipeline-parity tests")
    return bash, jq


def _render_hook(provider_cls: Any, journal: Path, hooks_dir: Path) -> Path:
    """Render the provider's bash hook into ``hooks_dir`` and point
    its journal at ``journal``. Returns the rendered script path."""
    p = provider_cls()
    p.journal_path = journal
    p.hooks_dir = hooks_dir
    p.settings_path = hooks_dir / "settings"  # unused; render_hook only reads journal_path
    hooks_dir.mkdir(parents=True, exist_ok=True)
    rendered = p.render_hook()
    hook_path = hooks_dir / p.hook_filename
    hook_path.write_text(rendered)
    hook_path.chmod(0o755)
    return hook_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _assert_parity(
    desc: str,
    bash_rows: list[dict[str, Any]],
    bf_rows: list[dict[str, Any]],
    expect_row: bool,
    *,
    fields: tuple[str, ...] = _PARITY_FIELDS,
) -> None:
    """Both pipelines must produce identical rows in identical order.

    A single fired turn can emit ≥1 row (one per kaomoji-led message
    in the turn) since the multi-message-per-turn fix landed; the
    parity contract is that bash and backfill agree row-for-row on
    parity-critical fields, in chronological order.

    ``fields`` defaults to :data:`_PARITY_FIELDS`. Pass a narrower
    tuple for providers where one field is divergent by design (e.g.
    Hermes' ``cwd``: live hook reads from payload, backfill source
    doesn't carry it — see :data:`_HERMES_PARITY_FIELDS`).
    """
    if not expect_row:
        assert bash_rows == [] and bf_rows == [], (
            f"{desc}: expected both pipelines to skip, got "
            f"bash={bash_rows!r}, backfill={bf_rows!r}"
        )
        return
    assert len(bash_rows) >= 1, f"{desc}: expected ≥1 bash row, got {bash_rows!r}"
    assert len(bash_rows) == len(bf_rows), (
        f"{desc}: row count mismatch — "
        f"bash={len(bash_rows)}, backfill={len(bf_rows)}\n"
        f"  bash:     {bash_rows!r}\n"
        f"  backfill: {bf_rows!r}"
    )
    for i, (b, f) in enumerate(zip(bash_rows, bf_rows)):
        for k in fields:
            assert b[k] == f[k], (
                f"{desc} divergence on row {i} field {k!r}:\n"
                f"  bash:     {b[k]!r}\n"
                f"  backfill: {f[k]!r}"
            )


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

# (description, assistant first-text, prior user-text, expect_row)
#
# The "expect_row" cases verify both pipelines INCLUDE the row;
# the False cases verify both pipelines SKIP cleanly (no row written).
# Mismatch between bash and Python on either kind is the failure mode
# this test exists to catch.
_CLAUDE_CASES: list[tuple[str, str, str, bool]] = [
    ("standard kaomoji",     "(◕‿◕) sounds good",                  "thanks",     True),
    ("japanese kaomoji",     "(｡◕‿◕｡) ok",                          "test",       True),
    ("bracket kaomoji",      "[≧▽≦] yay",                          "ok",         True),
    ("leading whitespace",   "  (･_･) ok",                          "hi",         True),
    ("trailing whitespace",  "(◕‿◕)   ok",                          "hi",         True),
    ("multiline body",       "(◕‿◕)\n\nfollowed by paragraph.",     "hi",         True),
    ("prose only",           "Sure, I can help with that.",         "test",       False),
    ("backslash escape",     "(\\*_*) tricks",                      "hi",         False),
    ("oversize span",        "(this is a sentence in parens with "
                              "way more than thirty-two chars)",     "hi",        False),
    ("no leading kaomoji "
     "char",                 "*_* no opener",                       "hi",         False),
]


@pytest.mark.parametrize("desc,assistant,user,expect_row", _CLAUDE_CASES)
def test_claude_code_hook_and_backfill_agree(
    desc: str,
    assistant: str,
    user: str,
    expect_row: bool,
    tmp_path: Path,
) -> None:
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_claude_code
    from llmoji.providers import ClaudeCodeProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, hook_journal, hooks_dir)

    # Subdirectory so backfill's rglob doesn't pick up hook-journal.jsonl
    # (backfill_claude_code matches *.jsonl).
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    transcript = tx_dir / "transcript.jsonl"
    events = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "message": {"content": user},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-04-27T00:00:00Z",
            "cwd": "/test",
            "message": {
                "model": "claude-test",
                "content": [{"type": "text", "text": assistant}],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    # Fire the rendered hook with a synthetic Stop event.
    stop_event = json.dumps({
        "transcript_path": str(transcript),
        "cwd": "/test",
    })
    r = subprocess.run(
        [bash, str(hook)],
        input=stop_event,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{desc}: bash hook failed: {r.stderr}"
    bash_rows = _read_jsonl(hook_journal)
    # Strip the bash-only ``ts`` field so the comparison is on
    # parity-critical fields. Backfill stamps the historical event
    # timestamp; bash stamps now-time. Both behaviors are correct
    # per the journal contract.
    for row in bash_rows:
        row.pop("ts", None)

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    _assert_parity(desc, bash_rows, bf_rows, expect_row)


# ---------------------------------------------------------------------------
# Claude Code multi-event-per-turn shape
# ---------------------------------------------------------------------------
#
# Real Claude Code transcripts persist each assistant content block —
# text, tool_use, thinking — as its own top-level JSONL entry. A
# turn that does anything tool-flavored produces many assistant
# entries, and the model can lead each text block with its own
# kaomoji. Post-fix the hook walks every text-bearing entry in the
# current turn and emits one row per kaomoji-led entry; non-kaomoji
# entries are skipped without aborting the rest of the walk. These
# cases pin the per-entry behavior.

def _claude_turn_events(
    *, blocks: list[dict[str, Any]], user: str = "ping",
) -> list[dict[str, Any]]:
    """Build a Claude Code transcript fragment with one user prompt
    and one assistant turn whose ``blocks`` are emitted as separate
    transcript entries (matching the real on-disk shape).

    ``blocks`` is a list of either ``{"type": "text", "text": "..."}``
    or ``{"type": "tool_use", ...}`` dicts; each becomes its own
    assistant JSONL row. Real transcripts also emit ``thinking``
    blocks, which we represent the same way.
    """
    events: list[dict[str, Any]] = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-04-27T00:00:00Z",
            "message": {"content": user},
        },
    ]
    parent = "u1"
    for i, block in enumerate(blocks):
        uid = f"a{i}"
        events.append({
            "type": "assistant",
            "uuid": uid,
            "parentUuid": parent,
            "timestamp": f"2026-04-27T00:00:{i+1:02d}Z",
            "cwd": "/test",
            "message": {
                "model": "claude-test",
                "content": [block],
            },
        })
        parent = uid
        # Tool-use blocks get a tool_result user entry on their heels
        # — the same pattern real transcripts ship. Stays in the
        # current turn (tool_results are not real-user events).
        if block.get("type") == "tool_use":
            tu_id = f"r{i}"
            events.append({
                "type": "user",
                "uuid": tu_id,
                "parentUuid": uid,
                "timestamp": f"2026-04-27T00:00:{i+1:02d}.5Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": uid,
                         "content": "ok"}
                    ],
                },
            })
            parent = tu_id
    return events


_CLAUDE_TURN_SHAPE_CASES: list[tuple[str, list[dict[str, Any]], list[str]]] = [
    # description, blocks, expected_kaomojis (one per emitted row, in
    # chronological order; empty list = no rows)
    (
        "text then tool_use — kaomoji-led reply followed by more "
        "tool work",
        [
            {"type": "text", "text": "(◕‿◕) here's the answer"},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
        ],
        ["(◕‿◕)"],
    ),
    (
        "tool_use then text (model investigates first then replies)",
        [
            {"type": "tool_use", "id": "tu1", "name": "Read",
             "input": {"file_path": "/x"}},
            {"type": "text", "text": "(´･ω･`) found it"},
        ],
        ["(´･ω･`)"],
    ),
    (
        "text → tool_use → text (kaomoji on first only; the "
        "post-tool follow-up has no kaomoji and is skipped — one "
        "row emitted, not two)",
        [
            {"type": "text", "text": "(◕‿◕) starting"},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
            {"type": "text", "text": "follow-up paragraph."},
        ],
        ["(◕‿◕)"],
    ),
    (
        "text → tool_use → text BOTH kaomoji-led — the multi-emit "
        "case the fix exists for; pre-fix only the first kaomoji "
        "made it into the journal, post-fix both rows land",
        [
            {"type": "text", "text": "(◕‿◕) starting"},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
            {"type": "text", "text": "(´･ω･`) and here's the result"},
        ],
        ["(◕‿◕)", "(´･ω･`)"],
    ),
    (
        "kaomoji → tool_use → kaomoji → tool_use → kaomoji "
        "(verification-heavy turn — three kaomoji-led messages "
        "interleaved with tool calls, three rows out)",
        [
            {"type": "text", "text": "(｀・ω・´) plan"},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
            {"type": "text", "text": "(⌐■_■) scaffold"},
            {"type": "tool_use", "id": "tu2", "name": "Read",
             "input": {"file_path": "/x"}},
            {"type": "text", "text": "(＾▽＾) shipped"},
        ],
        ["(｀・ω・´)", "(⌐■_■)", "(＾▽＾)"],
    ),
    (
        "tool_use → tool_use → text (multi-tool then kaomoji-led "
        "reply, common shape for verification-heavy turns)",
        [
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
            {"type": "tool_use", "id": "tu2", "name": "Read",
             "input": {"file_path": "/x"}},
            {"type": "text", "text": "[≧▽≦] all clear"},
        ],
        ["[≧▽≦]"],
    ),
    (
        "tool_use only — no text in turn, no row emitted",
        [
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
        ],
        [],
    ),
    (
        "non-kaomoji text then tool_use then kaomoji-led text — the "
        "first text fails the kaomoji filter and is skipped, the "
        "later kaomoji-led text still emits a row (post-fix; "
        "pre-fix the first text consumed the slot and the row was "
        "lost entirely)",
        [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}},
            {"type": "text", "text": "(◕‿◕) actually here"},
        ],
        ["(◕‿◕)"],
    ),
]


@pytest.mark.parametrize(
    "desc,blocks,expected_kaomojis",
    _CLAUDE_TURN_SHAPE_CASES,
)
def test_claude_code_multi_event_turn_shapes(
    desc: str,
    blocks: list[dict[str, Any]],
    expected_kaomojis: list[str],
    tmp_path: Path,
) -> None:
    """Live hook + backfill must agree on every kaomoji-led entry in
    a multi-event turn — count, order, and content. Pins the
    per-entry walk against the original "first-text-only" behavior.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_claude_code
    from llmoji.providers import ClaudeCodeProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, hook_journal, hooks_dir)

    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    transcript = tx_dir / "transcript.jsonl"
    events = _claude_turn_events(blocks=blocks)
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    stop_event = json.dumps({
        "transcript_path": str(transcript),
        "cwd": "/test",
    })
    r = subprocess.run(
        [bash, str(hook)],
        input=stop_event,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{desc}: bash hook failed: {r.stderr}"
    bash_rows = _read_jsonl(hook_journal)
    for row in bash_rows:
        row.pop("ts", None)

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    expect_row = bool(expected_kaomojis)
    _assert_parity(desc, bash_rows, bf_rows, expect_row)
    if expect_row:
        bash_kaomojis = [r["kaomoji"] for r in bash_rows]
        assert bash_kaomojis == expected_kaomojis, (
            f"{desc}: bash extracted {bash_kaomojis!r}, "
            f"expected {expected_kaomojis!r}"
        )


def test_claude_code_one_row_per_turn_across_session(tmp_path: Path) -> None:
    """Backfill must emit one row per real-user-bounded turn even
    when a single transcript chains several turns. Pre-fix backfill
    iterated assistant entries and could double-count text blocks
    across a turn boundary; the live hook never had this issue
    (Stop fires once per turn) but the parity contract requires
    them to agree on row count.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_claude_code

    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    transcript = tx_dir / "transcript.jsonl"

    events: list[dict[str, Any]] = []
    # Turn 1: kaomoji-led text only
    events.extend([
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "timestamp": "2026-04-27T00:00:00Z",
         "message": {"content": "first prompt"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "timestamp": "2026-04-27T00:00:01Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "text", "text": "(◕‿◕) one"}]}},
    ])
    # Turn 2: kaomoji-led text then tool_use (the bugged shape)
    events.extend([
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "timestamp": "2026-04-27T00:01:00Z",
         "message": {"content": "second prompt"}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "timestamp": "2026-04-27T00:01:01Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "text", "text": "(´｡• ω •｡`) two"}]}},
        {"type": "assistant", "uuid": "a3", "parentUuid": "a2",
         "timestamp": "2026-04-27T00:01:02Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "tool_use", "id": "tu",
                                  "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": "r1", "parentUuid": "a3",
         "timestamp": "2026-04-27T00:01:02.5Z",
         "message": {"content": [{"type": "tool_result",
                                  "tool_use_id": "tu", "content": "ok"}]}},
    ])
    # Turn 3: prose (no kaomoji), no row
    events.extend([
        {"type": "user", "uuid": "u3", "parentUuid": "r1",
         "timestamp": "2026-04-27T00:02:00Z",
         "message": {"content": "third prompt"}},
        {"type": "assistant", "uuid": "a4", "parentUuid": "u3",
         "timestamp": "2026-04-27T00:02:01Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "text", "text": "Sure thing."}]}},
    ])
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    bf_journal = tmp_path / "bf.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    rows = _read_jsonl(bf_journal)
    assert [r["kaomoji"] for r in rows] == ["(◕‿◕)", "(´｡• ω •｡`)"], rows
    assert [r["user_text"] for r in rows] == ["first prompt", "second prompt"], rows
    # Sanity-check: live hook fired against the same transcript also
    # picks the second turn's kaomoji-led text (the pre-fix bug case).
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook.jsonl"
    hook = _render_hook(ClaudeCodeProvider, hook_journal, hooks_dir)
    r = subprocess.run(
        [bash, str(hook)],
        input=json.dumps({"transcript_path": str(transcript), "cwd": "/test"}),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    hook_rows = _read_jsonl(hook_journal)
    # Hook keys on the LATEST turn; turn 3 is prose so no row.
    assert hook_rows == [], hook_rows


def test_claude_code_skill_injected_user_filtered(tmp_path: Path) -> None:
    """The Claude skill-activation prefix (in
    ``ClaudeCodeProvider.system_injected_prefixes``) must be dropped
    by both pipelines, falling back to the prior real user turn.

    Verifies the prefix-list unification is wired correctly: the
    bash side gets the prefixes via ``${INJECTED_PREFIXES_FILTER}``
    template substitution, the Python side reads the same tuple
    from the Provider class. They cannot disagree on which prefixes
    to drop.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_claude_code
    from llmoji.providers import ClaudeCodeProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, hook_journal, hooks_dir)

    # Use the Provider's first declared prefix verbatim so the test
    # tracks the canonical source. If the prefix list grows, this
    # test still exercises the first one.
    inj_prefix = ClaudeCodeProvider.system_injected_prefixes[0]
    real_user = "what's the weather"
    skill_user = inj_prefix + " /path/to/skill"

    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    transcript = tx_dir / "transcript.jsonl"
    events = [
        # real user turn (older)
        {
            "type": "user",
            "uuid": "u0",
            "parentUuid": None,
            "message": {"content": real_user},
        },
        # skill activation (newer) — both sides must drop this
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "u0",
            "message": {"content": skill_user},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-04-27T00:00:00Z",
            "cwd": "/test",
            "message": {
                "model": "claude-test",
                "content": [{"type": "text", "text": "(◕‿◕) hi"}],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    stop_event = json.dumps({
        "transcript_path": str(transcript),
        "cwd": "/test",
    })
    r = subprocess.run(
        [bash, str(hook)],
        input=stop_event,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"bash hook failed: {r.stderr}"
    bash_rows = _read_jsonl(hook_journal)
    for row in bash_rows:
        row.pop("ts", None)

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    assert len(bash_rows) == 1, bash_rows
    assert len(bf_rows) == 1, bf_rows
    # Both sides must resolve user_text to the older real prompt,
    # not the skill activation.
    assert bash_rows[0]["user_text"] == real_user
    assert bf_rows[0]["user_text"] == real_user
    _assert_parity("skill injection", bash_rows, bf_rows, expect_row=True)


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

def _codex_rollout_events(
    *, user_items: list[str], agent_messages: list[str],
) -> list[dict[str, Any]]:
    """Build a minimal Codex rollout event list for one turn.

    ``user_items`` becomes a sequence of ``response_item`` events
    (role=user) in declared order; the synthesis pipeline picks the
    last surviving one after dropping injected prefixes.

    ``agent_messages`` becomes a sequence of
    ``event_msg.agent_message`` events in declared order. Post-fix
    the codex hook + backfill walk these and emit one row per
    kaomoji-led message; pre-fix only the LAST one was captured via
    ``task_complete.last_agent_message``. ``task_complete`` is still
    appended for shape-fidelity but its ``last_agent_message`` field
    is no longer read by either pipeline.
    """
    events: list[dict[str, Any]] = [
        {"type": "session_meta", "payload": {"cwd": "/test"}},
        {
            "type": "turn_context",
            "timestamp": "2026-04-27T00:00:00Z",
            "payload": {"turn_id": "t1", "model": "gpt-test", "cwd": "/test"},
        },
    ]
    for txt in user_items:
        events.append({
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [{"type": "input_text", "text": txt}],
            },
        })
    for i, msg in enumerate(agent_messages):
        events.append({
            "type": "event_msg",
            "timestamp": f"2026-04-27T00:00:{i+1:02d}Z",
            "payload": {
                "type": "agent_message",
                "message": msg,
                "phase": "final_answer" if i == len(agent_messages) - 1 else "commentary",
            },
        })
    events.append({
        "type": "event_msg",
        "timestamp": f"2026-04-27T00:00:{len(agent_messages)+1:02d}Z",
        "payload": {
            "type": "task_complete",
            "turn_id": "t1",
            "last_agent_message": agent_messages[-1] if agent_messages else "",
            "completed_at": f"2026-04-27T00:00:{len(agent_messages)+1:02d}Z",
        },
    })
    return events


# (description, assistant, user_items, expect_user_text, expect_row)
_CODEX_CASES: list[tuple[str, str, list[str], str, bool]] = [
    ("standard",
     "(◕‿◕) ok", ["test prompt"], "test prompt", True),
    ("multiline body",
     "(｡◕‿◕｡)\n\nfollow-up.", ["q?"], "q?", True),
    ("AGENTS.md filtered",
     "(◕‿◕) ok", ["# AGENTS.md\nstuff"], "", True),
    ("environment_context filtered",
     "(◕‿◕) ok", ["<environment_context>x</environment_context>"], "", True),
    ("INSTRUCTIONS filtered",
     "(◕‿◕) ok", ["<INSTRUCTIONS>x"], "", True),
    ("inject + real, real wins",
     "(◕‿◕) ok",
     ["# AGENTS.md\nstuff", "real prompt"],
     "real prompt", True),
    ("prose only",
     "Done with the task.", ["q?"], "q?", False),
    ("backslash escape",
     "(\\*_*) trick", ["q?"], "q?", False),
    ("oversize span",
     "(parenthetical sentence going way past thirty-two characters)",
     ["q?"], "q?", False),
]


@pytest.mark.parametrize(
    "desc,assistant,user_items,expect_user_text,expect_row",
    _CODEX_CASES,
)
def test_codex_hook_and_backfill_agree(
    desc: str,
    assistant: str,
    user_items: list[str],
    expect_user_text: str,
    expect_row: bool,
    tmp_path: Path,
) -> None:
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_codex
    from llmoji.providers import CodexProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(CodexProvider, hook_journal, hooks_dir)

    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    rollout = tx_dir / "rollout-2026-04-27.jsonl"  # backfill matches rollout-*.jsonl
    events = _codex_rollout_events(
        user_items=user_items, agent_messages=[assistant],
    )
    rollout.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    # Codex Stop event: transcript_path + the per-turn metadata the
    # Codex CLI passes in production. ``last_assistant_message`` is
    # included for shape-fidelity but no longer read by the hook —
    # the hook walks the rollout for every event_msg.agent_message
    # in the current turn.
    stop_event = json.dumps({
        "transcript_path": str(rollout),
        "last_assistant_message": assistant,
        "model": "gpt-test",
        "cwd": "/test",
    })
    r = subprocess.run(
        [bash, str(hook)],
        input=stop_event,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{desc}: bash hook failed: {r.stderr}"
    bash_rows = _read_jsonl(hook_journal)
    for row in bash_rows:
        row.pop("ts", None)

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_codex(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    _assert_parity(desc, bash_rows, bf_rows, expect_row)
    if expect_row:
        # Bonus: confirm the agreed-upon user_text matches the
        # case's expectation. Catches a class of bug where bash
        # and Python both produce the SAME wrong answer.
        assert bash_rows[0]["user_text"] == expect_user_text, (
            f"{desc}: bash user_text={bash_rows[0]['user_text']!r}, "
            f"expected {expect_user_text!r}"
        )


# (description, agent_messages, expected_kaomojis)
#
# Codex emits one ``event_msg.agent_message`` per model message; a
# tool-heavy turn writes 5–10 of them. Post-fix the live hook +
# backfill walk every one and emit one row per kaomoji-led message.
# Pre-fix only the final message (via ``task_complete.last_agent_
# message``) was captured. These cases pin the multi-emission
# behavior — the codex side of the same fix that
# ``test_claude_code_multi_event_turn_shapes`` pins for claude_code.
_CODEX_TURN_SHAPE_CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "two kaomoji-led messages — both land in the journal",
        ["(◕‿◕) plan", "(´･ω･`) shipped"],
        ["(◕‿◕)", "(´･ω･`)"],
    ),
    (
        "five kaomoji-led messages — every one lands (the worked "
        "example from CLAUDE.md's gotchas section)",
        [
            "(｀・ω・´) plan",
            "(ง •̀_•́) attack",
            "(⌐■_■) scaffold",
            "(｡•̀ᴗ-)✧ research",
            "(＾▽＾) shipped",
        ],
        ["(｀・ω・´)", "(ง •̀_•́)", "(⌐■_■)", "(｡•̀ᴗ-)✧", "(＾▽＾)"],
    ),
    (
        "kaomoji + non-kaomoji + kaomoji — the prose middle "
        "message is skipped, the bookends both land",
        ["(◕‿◕) plan", "let me check the logs", "(´･ω･`) found it"],
        ["(◕‿◕)", "(´･ω･`)"],
    ),
    (
        "all-prose turn — zero rows emitted (the model never "
        "kaomoji'd, fine)",
        ["let me check", "found something", "shipping the fix"],
        [],
    ),
    (
        "single kaomoji-led message — degenerate one-row case",
        ["(◕‿◕) ok"],
        ["(◕‿◕)"],
    ),
]


@pytest.mark.parametrize(
    "desc,agent_messages,expected_kaomojis",
    _CODEX_TURN_SHAPE_CASES,
)
def test_codex_multi_message_turn_shapes(
    desc: str,
    agent_messages: list[str],
    expected_kaomojis: list[str],
    tmp_path: Path,
) -> None:
    """Live hook + backfill must agree on every kaomoji-led
    agent_message in a multi-message Codex turn — count, order,
    and content. Pins the per-message walk against the original
    "task_complete.last_agent_message" behavior.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_codex
    from llmoji.providers import CodexProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(CodexProvider, hook_journal, hooks_dir)

    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    rollout = tx_dir / "rollout-2026-04-27.jsonl"
    events = _codex_rollout_events(
        user_items=["test prompt"], agent_messages=agent_messages,
    )
    rollout.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    stop_event = json.dumps({
        "transcript_path": str(rollout),
        "model": "gpt-test",
        "cwd": "/test",
    })
    r = subprocess.run(
        [bash, str(hook)],
        input=stop_event,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{desc}: bash hook failed: {r.stderr}"
    bash_rows = _read_jsonl(hook_journal)
    for row in bash_rows:
        row.pop("ts", None)

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_codex(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    expect_row = bool(expected_kaomojis)
    _assert_parity(desc, bash_rows, bf_rows, expect_row)
    if expect_row:
        bash_kaomojis = [r["kaomoji"] for r in bash_rows]
        assert bash_kaomojis == expected_kaomojis, (
            f"{desc}: bash extracted {bash_kaomojis!r}, "
            f"expected {expected_kaomojis!r}"
        )


# ---------------------------------------------------------------------------
# Hermes — live hook + backfill, multi-message-per-turn parity
# ---------------------------------------------------------------------------
#
# Hermes' live hook walks ``extra.conversation_history`` and emits one
# row per kaomoji-led assistant message in the current turn (the slice
# from the latest user-role message to the end of the array). The
# backfill side reads ``~/.hermes/sessions/session_*.json`` files and
# walks the persisted ``messages`` list, chunking on user-role
# boundaries — same per-turn-slice operation as the hook applied to
# every turn in the session. The two pipelines must agree row-for-row
# on parity-critical fields per turn.


def _hermes_payload(
    *,
    user_message: str,
    conversation_history: list[dict[str, Any]],
    session_id: str = "sess-test",
    cwd: str = "/test",
    model: str = "hermes-test",
) -> dict[str, Any]:
    """Build a synthetic ``post_llm_call`` stdin payload that mirrors
    what ``hermes-agent/agent/shell_hooks.py:_serialize_payload``
    produces. ``cwd`` is top-level (= ``Path.cwd()`` of the agent
    process at hook fire time); the per-turn fields land under
    ``extra``."""
    return {
        "hook_event_name": "post_llm_call",
        "tool_name": None,
        "tool_input": None,
        "session_id": session_id,
        "cwd": cwd,
        "extra": {
            "user_message": user_message,
            "assistant_response": (
                conversation_history[-1].get("content") or ""
                if conversation_history else ""
            ),
            "conversation_history": conversation_history,
            "model": model,
            "platform": "cli",
        },
    }


# (description, assistant blocks for the latest turn, expected_kaomojis)
#
# Each case builds a one-turn conversation: a leading user message
# plus the listed assistant content blocks (each becomes a separate
# message in conversation_history). Empty-content assistant entries
# represent tool-only turn steps and are skipped by both pipelines.
_HERMES_TURN_SHAPE_CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "standard single kaomoji-led message",
        ["(◕‿◕) sounds good"],
        ["(◕‿◕)"],
    ),
    (
        "multiline body — kaomoji on first line still extracted",
        ["(｡◕‿◕｡)\n\nfollowed by paragraph."],
        ["(｡◕‿◕｡)"],
    ),
    (
        "two kaomoji-led messages in one turn — both land",
        ["(◕‿◕) plan", "(´･ω･`) shipped"],
        ["(◕‿◕)", "(´･ω･`)"],
    ),
    (
        "five kaomoji-led messages in one turn — every one lands "
        "(verification-heavy turn shape)",
        [
            "(｀・ω・´) plan",
            "(ง •̀_•́) attack",
            "(⌐■_■) scaffold",
            "(｡•̀ᴗ-)✧ research",
            "(＾▽＾) shipped",
        ],
        ["(｀・ω・´)", "(ง •̀_•́)", "(⌐■_■)", "(｡•̀ᴗ-)✧", "(＾▽＾)"],
    ),
    (
        "kaomoji + non-kaomoji + kaomoji — prose middle skipped, "
        "bookends both land",
        ["(◕‿◕) plan", "let me check the logs", "(´･ω･`) found it"],
        ["(◕‿◕)", "(´･ω･`)"],
    ),
    (
        "all-prose turn — zero rows emitted",
        ["let me check", "found something", "shipping the fix"],
        [],
    ),
    (
        "prose only single message",
        ["Sure thing."],
        [],
    ),
    (
        "backslash escape — rejected by validator",
        ["(\\*_*) trick"],
        [],
    ),
    (
        "oversize span — rejected by validator",
        [
            "(parenthetical sentence going way past thirty-two characters)",
        ],
        [],
    ),
]


@pytest.mark.parametrize(
    "desc,assistant_blocks,expected_kaomojis",
    _HERMES_TURN_SHAPE_CASES,
)
def test_hermes_hook_and_backfill_agree(
    desc: str,
    assistant_blocks: list[str],
    expected_kaomojis: list[str],
    tmp_path: Path,
) -> None:
    """Live hook + backfill must agree on every kaomoji-led assistant
    message in a Hermes turn — count, order, and content. Live hook
    walks ``extra.conversation_history`` from the post_llm_call
    payload; backfill walks the persisted ``messages`` array in the
    session file. Same per-turn-slice operation, same canonical
    journal-row output.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_hermes
    from llmoji.providers import HermesProvider

    user_text = "test prompt"
    conversation_history = [
        {"role": "user", "content": user_text},
    ]
    for txt in assistant_blocks:
        conversation_history.append({"role": "assistant", "content": txt})

    # --- Live hook side ---
    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(HermesProvider, hook_journal, hooks_dir)

    payload = _hermes_payload(
        user_message=user_text,
        conversation_history=conversation_history,
    )
    r = subprocess.run(
        [bash, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"{desc}: bash hook failed: {r.stderr}"
    # Hermes hook contract: stdout must be JSON, ``{}`` for no-op.
    assert r.stdout.strip() == "{}", (
        f"{desc}: hermes hook stdout should be '{{}}' for fail-open "
        f"contract; got {r.stdout!r}"
    )
    bash_rows = _read_jsonl(hook_journal)
    for row in bash_rows:
        row.pop("ts", None)

    # --- Backfill side ---
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    session_file = sessions_dir / "session_20260427_000000_test.json"
    session_file.write_text(json.dumps({
        "session_id": "20260427_000000_test",
        "model": "hermes-test",
        "platform": "cli",
        "session_start": "2026-04-27T00:00:00",
        "last_updated": "2026-04-27T00:00:01",
        "system_prompt": "",
        "tools": [],
        "message_count": len(conversation_history),
        "messages": conversation_history,
    }))

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_hermes(sessions_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)
    for row in bf_rows:
        row.pop("ts", None)

    expect_row = bool(expected_kaomojis)
    _assert_parity(
        desc, bash_rows, bf_rows, expect_row,
        fields=_HERMES_PARITY_FIELDS,
    )
    if expect_row:
        bash_kaomojis = [r["kaomoji"] for r in bash_rows]
        assert bash_kaomojis == expected_kaomojis, (
            f"{desc}: bash extracted {bash_kaomojis!r}, "
            f"expected {expected_kaomojis!r}"
        )
        # Live hook stamps cwd from the payload; backfill leaves it
        # blank because the session JSON doesn't persist cwd. Pin
        # both behaviors so a future change to either side trips the
        # test rather than silently changing the on-disk row shape.
        assert all(r["cwd"] == "/test" for r in bash_rows), bash_rows
        assert all(r["cwd"] == "" for r in bf_rows), bf_rows


def test_hermes_hook_resolves_user_text_and_drops_prior_turn(
    tmp_path: Path,
) -> None:
    """Multi-turn conversation history: only the latest turn's
    assistant messages land in the journal. ``user_text`` resolves
    to the latest user-role message via ``extra.user_message``;
    earlier turns are walked off the slice.

    Pre-fix (single-message Hermes hook) this distinction didn't
    matter because the hook never walked history. Post-fix the
    walker explicitly slices from the latest user-role index, so
    cross-turn contamination is the regression to guard against.
    """
    bash, _ = _require_tools()
    from llmoji.providers import HermesProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(HermesProvider, hook_journal, hooks_dir)

    conversation_history = [
        # Turn 1 — kaomoji-led, but should NOT appear in journal (not
        # the current turn).
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "(´｡• ω •｡`) reply to first"},
        # Turn 2 — current. Two kaomoji-led messages, both land.
        {"role": "user", "content": "second prompt"},
        {"role": "assistant", "content": "(◕‿◕) starting"},
        {"role": "assistant", "content": "(＾▽＾) done"},
    ]
    payload = _hermes_payload(
        user_message="second prompt",
        conversation_history=conversation_history,
    )
    r = subprocess.run(
        [bash, str(hook)], input=json.dumps(payload),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    rows = _read_jsonl(hook_journal)
    assert [r["kaomoji"] for r in rows] == ["(◕‿◕)", "(＾▽＾)"], rows
    assert all(r["user_text"] == "second prompt" for r in rows), rows


def test_hermes_backfill_walks_every_turn(tmp_path: Path) -> None:
    """Backfill must emit rows for EVERY kaomoji-led message across
    EVERY turn in the session — not just the latest. The session JSON
    holds the cumulative ``messages`` list, so chunking on user-role
    boundaries gives one slice per turn; backfill walks each.

    Live hook only sees one turn at a time (the current
    post_llm_call's slice). Backfill rebuilds the full corpus from
    the persisted file. Tested separately because the live-hook
    parity tests above only fire one turn.
    """
    from llmoji.backfill import backfill_hermes

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    session_file = sessions_dir / "session_20260427_000000_test.json"

    # Three turns: kaomoji / mixed-with-tool / prose. Tool-only
    # assistant entries (content="" + tool_calls) must be skipped
    # without contaminating the row count.
    messages = [
        # Turn 1: single kaomoji
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "(◕‿◕) one"},
        # Turn 2: kaomoji-led + tool-only + kaomoji-led (the
        # multi-emit case the post-fix walker exists for)
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "(´･ω･`) thinking"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "tc1", "function": {
             "name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "tc1"},
        {"role": "assistant", "content": "(＾▽＾) shipped"},
        # Turn 3: prose only — no rows
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "Sure thing."},
    ]
    session_file.write_text(json.dumps({
        "session_id": "20260427_000000_test",
        "model": "hermes-test",
        "platform": "cli",
        "session_start": "2026-04-27T00:00:00",
        "last_updated": "2026-04-27T00:00:01",
        "system_prompt": "",
        "tools": [],
        "message_count": len(messages),
        "messages": messages,
    }))

    bf_journal = tmp_path / "bf.jsonl"
    n = backfill_hermes(sessions_dir, bf_journal)
    rows = _read_jsonl(bf_journal)
    assert n == 3 == len(rows), (n, rows)
    assert [r["kaomoji"] for r in rows] == [
        "(◕‿◕)", "(´･ω･`)", "(＾▽＾)",
    ], rows
    assert [r["user_text"] for r in rows] == [
        "turn 1", "turn 2", "turn 2",
    ], rows
