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
# bash hook output and the backfill output. Since 2.2 ``ts`` is one of
# them: the live hooks stamp the message's own transcript time, so a
# live row and its replay are the same row and ``llmoji import``
# dedups them exactly.
_PARITY_FIELDS = ("ts", "kaomoji", "model", "cwd", "user_text", "assistant_text")

# Hermes-specific: the live hook reads ``cwd`` from the post_llm_call
# payload (= ``Path.cwd()`` of the agent process at fire time); the
# backfill side reads from ``~/.hermes/sessions/session_*.json``,
# which doesn't persist cwd at all (the field simply isn't in the
# session schema; see ``_save_session_log`` in
# ``hermes-agent/run_agent.py``). Backfilled rows therefore land with
# cwd="" by design. The two are divergent on cwd by source design,
# not bug.
#
# ``ts`` diverges the same way: Hermes session files keep no
# per-message time, so the live hook stamps fire time and the backfill
# stamps the session's ``last_updated``.
_HERMES_PARITY_FIELDS = tuple(
    f for f in _PARITY_FIELDS if f not in ("cwd", "ts")
)


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
    # Round-6 bare kaomoji: no leader char from KAOMOJI_START_CHARS,
    # but match _looks_like_bare_kaomoji shape. Bash defers to Python
    # for Path B via the ${PYTHON_INTERPRETER} subprocess fallback;
    # backfill goes through is_kaomoji_candidate directly.
    ("bare symmetric *_*",   "*_* no opener",                       "hi",         True),
    ("bare symmetric T-T",   "T-T sad",                              "hi",         True),
    ("bare symmetric ^_^",   "^_^ happy!",                           "hi",         True),
    ("bare paired >_<",      ">_< ouch",                             "hi",         True),
    ("bare 2-char XD",       "XD lol",                               "hi",         True),
    ("bare western :)",      ":) cool",                              "hi",         True),
    # Prose-shaped pseudo-bare kaomoji that should still reject.
    ("prose 2-char OK",      "OK got it",                           "hi",         False),
    ("prose contraction",    "I'll get it",                         "hi",         False),
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

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)

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

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)

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
    from the ClaudeCodeProvider class. They cannot disagree on which
    prefixes to drop.
    """
    bash, _ = _require_tools()

    from llmoji.backfill import backfill_claude_code
    from llmoji.providers import ClaudeCodeProvider

    hooks_dir = tmp_path / "hooks"
    hook_journal = tmp_path / "hook-journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, hook_journal, hooks_dir)

    # Use the ClaudeCodeProvider's first declared prefix verbatim so
    # the test tracks the canonical source. If the prefix list grows,
    # this test still exercises the first one.
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

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_claude_code(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)

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

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_codex(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)

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

    bf_journal = tmp_path / "backfill-journal.jsonl"
    backfill_codex(tx_dir, bf_journal)
    bf_rows = _read_jsonl(bf_journal)

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


# ---------------------------------------------------------------------------
# Claude Code — final message missing from the transcript at Stop time
# ---------------------------------------------------------------------------
#
# Claude Code documents that the transcript may not hold the turn's
# final assistant message yet when Stop fires, and hands the text over
# as ``last_assistant_message`` instead. On a real corpus the race
# dropped about a quarter of all kaomoji-led messages, nearly always
# the closing one.


def _fire_claude_hook(
    bash: str, hook: Path, transcript: Path, last: str | None,
) -> None:
    payload: dict[str, Any] = {"transcript_path": str(transcript), "cwd": "/test"}
    if last is not None:
        payload["last_assistant_message"] = last
    r = subprocess.run(
        [bash, str(hook)], input=json.dumps(payload),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_claude_code_hook_journals_final_message_missing_from_transcript(
    tmp_path: Path,
) -> None:
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    transcript = tmp_path / "transcript.jsonl"
    # The transcript has the turn's first message but not its last.
    events = _claude_turn_events(blocks=[
        {"type": "text", "text": "(◕‿◕) starting"},
        {"type": "tool_use", "id": "t0", "name": "Bash", "input": {}},
    ])
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    _fire_claude_hook(bash, hook, transcript, "(＾▽＾) all done\n")
    rows = _read_jsonl(journal)
    assert [r["kaomoji"] for r in rows] == ["(◕‿◕)", "(＾▽＾)"], rows
    final = rows[1]
    assert final["assistant_text"] == "all done"
    assert final["model"] == "claude-test"
    assert final["user_text"] == "ping"
    # No transcript timestamp to borrow yet: fire time, second precision.
    assert final["ts"].endswith("Z") and "." not in final["ts"]


def test_claude_code_hook_does_not_double_a_final_message_already_written(
    tmp_path: Path,
) -> None:
    """When the transcript already holds the final message, the
    payload copy is the same message and adds nothing — including
    when the transcript split it across text blocks."""
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    transcript = tmp_path / "transcript.jsonl"
    events = _claude_turn_events(blocks=[
        {"type": "text", "text": "(◕‿◕) first half"},
        {"type": "text", "text": "second half"},
    ])
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    _fire_claude_hook(bash, hook, transcript, "(◕‿◕) first half\n\nsecond half")
    rows = _read_jsonl(journal)
    assert [r["assistant_text"] for r in rows] == ["first half"], rows


# ---------------------------------------------------------------------------
# Codex — current codex-cli rollouts (item_completed / AgentMessage)
# ---------------------------------------------------------------------------


def _codex_item_events(
    agent_messages: list[str], *, legacy_too: bool = False,
) -> list[dict[str, Any]]:
    """One turn in the codex-cli 0.144+ rollout shape: each agent
    message is an ``item_completed`` event whose item is an
    ``AgentMessage``. ``legacy_too`` also writes the old
    ``agent_message`` event for every message, as a release that
    wrote both would have."""
    events: list[dict[str, Any]] = [
        {"type": "session_meta", "payload": {
            "cwd": "/test", "thread_source": "user", "source": "exec"}},
        {"type": "turn_context", "timestamp": "2026-10-25T00:00:00.000Z",
         "payload": {"turn_id": "t1", "model": "gpt-test", "cwd": "/test"}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "do the thing"}]}},
    ]
    for i, msg in enumerate(agent_messages):
        ts = f"2026-10-25T00:00:{i+1:02d}.500Z"
        events.append({"type": "event_msg", "timestamp": ts, "payload": {
            "type": "item_completed", "turn_id": "t1",
            "item": {"type": "AgentMessage", "id": f"msg_{i}",
                     "content": [{"type": "Text", "text": msg}],
                     "phase": "commentary"}}})
        # The model-API mirror Codex also persists; never read.
        events.append({"type": "response_item", "timestamp": ts, "payload": {
            "type": "message", "role": "assistant", "id": f"msg_{i}",
            "content": [{"type": "output_text", "text": msg}]}})
        if legacy_too:
            events.append({"type": "event_msg", "timestamp": ts, "payload": {
                "type": "agent_message", "message": msg}})
    return events


def _run_codex_both(
    tmp_path: Path, events: list[dict[str, Any]], *, model: str | None = "gpt-test",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bash, _ = _require_tools()
    from llmoji.backfill import backfill_codex
    from llmoji.providers import CodexProvider

    journal = tmp_path / "hook.jsonl"
    hook = _render_hook(CodexProvider, journal, tmp_path / "hooks")
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    rollout = tx_dir / "rollout-2026-10-25.jsonl"
    rollout.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    payload: dict[str, Any] = {"transcript_path": str(rollout), "cwd": "/test"}
    if model is not None:
        payload["model"] = model
    r = subprocess.run(
        [bash, str(hook)], input=json.dumps(payload),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    bf = tmp_path / "bf.jsonl"
    backfill_codex(tx_dir, bf)
    return _read_jsonl(journal), _read_jsonl(bf)


def test_codex_item_completed_rollout_parity(tmp_path: Path) -> None:
    """The shape codex-cli has written since 0.144. The pre-2.2 hook
    matched none of it and journaled nothing."""
    msgs = ["(◕‿◕) starting", "plain progress note", "(＾▽＾) done"]
    bash_rows, bf_rows = _run_codex_both(tmp_path, _codex_item_events(msgs))
    _assert_parity("codex item_completed", bash_rows, bf_rows, True)
    assert [r["kaomoji"] for r in bash_rows] == ["(◕‿◕)", "(＾▽＾)"]
    assert [r["ts"] for r in bash_rows] == [
        "2026-10-25T00:00:01.500Z", "2026-10-25T00:00:03.500Z",
    ]
    assert bash_rows[0]["user_text"] == "do the thing"


def test_codex_turn_with_both_shapes_counts_each_message_once(
    tmp_path: Path,
) -> None:
    bash_rows, bf_rows = _run_codex_both(
        tmp_path, _codex_item_events(["(◕‿◕) once"], legacy_too=True),
    )
    _assert_parity("codex dual shape", bash_rows, bf_rows, True)
    assert len(bash_rows) == 1


def test_codex_model_falls_back_to_turn_context(tmp_path: Path) -> None:
    bash_rows, bf_rows = _run_codex_both(
        tmp_path, _codex_item_events(["(◕‿◕) hi"]), model=None,
    )
    _assert_parity("codex no payload model", bash_rows, bf_rows, True)
    assert bash_rows[0]["model"] == "gpt-test"


def test_codex_subagent_rollout_is_skipped(tmp_path: Path) -> None:
    events = _codex_item_events(["(◕‿◕) from a child thread"])
    events[0]["payload"]["thread_source"] = "subagent"
    events[0]["payload"]["source"] = {"subagent": {"thread_spawn": {
        "parent_thread_id": "p", "depth": 1}}}
    bash_rows, bf_rows = _run_codex_both(tmp_path, events)
    _assert_parity("codex subagent", bash_rows, bf_rows, False)


def test_codex_backfill_reads_zstd_rollouts(tmp_path: Path) -> None:
    """codex-cli compresses older rollouts in place to
    ``rollout-*.jsonl.zst``; the replay reads them when a zstd decoder
    is installed."""
    from llmoji.backfill import _zstd_decoder, backfill_codex

    if _zstd_decoder() is None:
        pytest.skip("no zstd decoder (Python < 3.14 without zstandard)")
    try:
        from compression import zstd  # type: ignore[import-not-found]
        compress = zstd.compress
    except ImportError:
        import zstandard  # type: ignore[import-not-found]
        compress = zstandard.ZstdCompressor().compress

    tx_dir = tmp_path / "tx" / "2026" / "09" / "01"
    tx_dir.mkdir(parents=True)
    events = _codex_item_events(["(◕‿◕) archived"])
    raw = ("\n".join(json.dumps(e) for e in events) + "\n").encode()
    (tx_dir / "rollout-2026-09-01.jsonl.zst").write_bytes(compress(raw))

    bf = tmp_path / "bf.jsonl"
    assert backfill_codex(tmp_path / "tx", bf) == 1
    assert _read_jsonl(bf)[0]["assistant_text"] == "archived"


# ---------------------------------------------------------------------------
# Grok
# ---------------------------------------------------------------------------


def _grok_update(kind: str, *, text: str = "", pid: str = "",
                 ms: int = 0, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {"agentTimestampMs": ms}
    if pid:
        meta["promptId"] = pid
    update: dict[str, Any] = {"sessionUpdate": kind, **extra}
    if text or kind.endswith("_chunk"):
        update["content"] = {"type": "text", "text": text}
    return {"timestamp": ms // 1000, "method": "session/update",
            "params": {"sessionId": "s1", "update": update, "_meta": meta}}


def _grok_session(tmp_path: Path, updates: list[dict[str, Any]],
                  model: str = "grok-test-build") -> Path:
    """A Grok session directory: updates.jsonl plus the chat_history
    and summary files the hook and replay read alongside it."""
    session = tmp_path / "sessions" / "%2Ftest" / "s1"
    session.mkdir(parents=True)
    (session / "updates.jsonl").write_text(
        "\n".join(json.dumps(u) for u in updates) + "\n")
    (session / "chat_history.jsonl").write_text(json.dumps(
        {"type": "assistant", "content": "x", "model_id": model}) + "\n")
    (session / "summary.json").write_text(json.dumps(
        {"info": {"id": "s1", "cwd": "/test"}}))
    return session


def _grok_turn(pid: str, user: str, messages: list[str], start_ms: int,
               model: str = "grok-test-build") -> list[dict[str, Any]]:
    out = [_grok_update("user_message_chunk", text=user, ms=start_ms)]
    for i, msg in enumerate(messages):
        out.append(_grok_update("agent_thought_chunk", text="hmm",
                                pid=pid, ms=start_ms + 1000 * i + 500))
        out.append(_grok_update("agent_message_chunk", text=msg,
                                pid=pid, ms=start_ms + 1000 * (i + 1)))
        out.append(_grok_update("tool_call", pid=pid, toolCallId=f"c{i}",
                                ms=start_ms + 1000 * (i + 1) + 10))
    out.append(_grok_update("turn_completed", prompt_id=pid, ms=start_ms + 9000,
                            usage={"modelUsage": {model: {"outputTokens": 5}}}))
    return out


def _fire_grok(tmp_path: Path, session: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    bash, _ = _require_tools()
    from llmoji.providers import GrokProvider

    journal = tmp_path / "hook.jsonl"
    hook = _render_hook(GrokProvider, journal, tmp_path / "hooks")
    payload = {"transcript_path": str(session / "updates.jsonl"),
               "cwd": "/test", **payload}
    r = subprocess.run([bash, str(hook)], input=json.dumps(payload),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "", "a Grok Stop hook's stdout is a stop decision"
    return _read_jsonl(journal)


def test_grok_hook_and_backfill_agree(tmp_path: Path) -> None:
    from llmoji.backfill import backfill_grok

    updates = (
        _grok_turn("p1", "first prompt", ["(◕‿◕) earlier turn"], 1_792_000_000_000)
        + _grok_turn("p2", "second prompt",
                     ["(・_・) looking", "no face here", "(＾▽＾) done"],
                     1_792_000_100_000)
    )
    session = _grok_session(tmp_path, updates)
    bash_rows = _fire_grok(tmp_path, session, {"reason": "end_turn", "promptId": "p2"})
    bf = tmp_path / "bf.jsonl"
    backfill_grok(tmp_path / "sessions", bf)
    # The live hook sees one turn; the replay sees the whole session.
    bf_rows = [r for r in _read_jsonl(bf) if r["user_text"] == "second prompt"]
    _assert_parity("grok", bash_rows, bf_rows, True)
    assert [r["kaomoji"] for r in bash_rows] == ["(・_・)", "(＾▽＾)"]
    assert bash_rows[0]["ts"] == "2026-10-14T17:48:21Z"  # 1792000101000 ms, floored
    assert bash_rows[0]["model"] == "grok-test-build"
    assert len(_read_jsonl(bf)) == 3


def test_grok_hook_joins_chunks_of_one_message(tmp_path: Path) -> None:
    from llmoji.backfill import backfill_grok

    updates = [
        _grok_update("user_message_chunk", text="hi", ms=1_792_000_000_000),
        _grok_update("agent_message_chunk", text="(◕‿◕) hel", pid="p1", ms=1_792_000_001_000),
        _grok_update("agent_message_chunk", text="lo there", pid="p1", ms=1_792_000_001_200),
        _grok_update("tool_call", pid="p1", ms=1_792_000_002_000),
        _grok_update("agent_message_chunk", text="(＾▽＾) bye", pid="p1", ms=1_792_000_003_000),
    ]
    session = _grok_session(tmp_path, updates)
    bash_rows = _fire_grok(tmp_path, session, {"reason": "end_turn", "promptId": "p1"})
    bf = tmp_path / "bf.jsonl"
    backfill_grok(tmp_path / "sessions", bf)
    _assert_parity("grok chunks", bash_rows, _read_jsonl(bf), True)
    assert [r["assistant_text"] for r in bash_rows] == ["hello there", "bye"]


def test_grok_hook_ignores_session_end_stop(tmp_path: Path) -> None:
    """Grok fires a second Stop at teardown; journaling it would land
    the last turn twice."""
    session = _grok_session(tmp_path, _grok_turn("p1", "q", ["(◕‿◕) ok"], 1_792_000_000_000))
    assert _fire_grok(tmp_path, session, {"reason": "shutdown"}) == []
    assert _fire_grok(tmp_path, session, {"reason": "end_turn", "promptId": "p1",
                                          "subagentType": "explore"}) == []


def test_grok_backfill_skips_subagent_sessions(tmp_path: Path) -> None:
    from llmoji.backfill import backfill_grok

    parent = _grok_session(tmp_path, _grok_turn("p1", "q", ["(◕‿◕) parent"], 1_792_000_000_000))
    child = parent.parent / "child-1"
    child.mkdir()
    (child / "updates.jsonl").write_text("\n".join(json.dumps(u) for u in _grok_turn(
        "c1", "task", ["(•̀ᴗ•́) child"], 1_792_000_050_000)) + "\n")
    (parent / "subagents" / "a1").mkdir(parents=True)
    (parent / "subagents" / "a1" / "meta.json").write_text(json.dumps({
        "session_id": "child-1", "parent_session_id": parent.name,
        "agent_type": "explore",
    }))

    bf = tmp_path / "bf.jsonl"
    backfill_grok(tmp_path / "sessions", bf)
    assert [r["assistant_text"] for r in _read_jsonl(bf)] == ["parent"]


def test_claude_code_skill_loaded_mid_turn_keeps_earlier_messages(
    tmp_path: Path,
) -> None:
    """A skill body arrives as a user-role entry partway through a
    turn. It isn't the user speaking, so it mustn't reset the turn
    boundary: before the fix every kaomoji-led message ahead of the
    skill load was dropped from live capture."""
    bash, _ = _require_tools()
    from llmoji.backfill import backfill_claude_code
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "hook.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    transcript = tx_dir / "t.jsonl"
    events = [
        {"type": "user", "uuid": "u1", "parentUuid": None,
         "timestamp": "2026-10-25T00:00:00Z", "message": {"content": "real prompt"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "timestamp": "2026-10-25T00:00:01Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "text", "text": "(◕‿◕) before the skill"}]}},
        {"type": "user", "uuid": "u2", "parentUuid": "a1",
         "timestamp": "2026-10-25T00:00:02Z",
         "message": {"content": [{"type": "text",
                                  "text": "Base directory for this skill: /x\n\n# Skill"}]}},
        {"type": "assistant", "uuid": "a2", "parentUuid": "u2",
         "timestamp": "2026-10-25T00:00:03Z", "cwd": "/test",
         "message": {"model": "claude-test",
                     "content": [{"type": "text", "text": "(＾▽＾) after the skill"}]}},
    ]
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    _fire_claude_hook(bash, hook, transcript, None)
    bf = tmp_path / "bf.jsonl"
    backfill_claude_code(tx_dir, bf)
    bash_rows, bf_rows = _read_jsonl(journal), _read_jsonl(bf)
    _assert_parity("skill mid-turn", bash_rows, bf_rows, True)
    assert [r["assistant_text"] for r in bash_rows] == [
        "before the skill", "after the skill",
    ]
    assert {r["user_text"] for r in bash_rows} == {"real prompt"}


# ---------------------------------------------------------------------------
# Review regressions (2.2.0 release review)
# ---------------------------------------------------------------------------


def test_claude_code_earlier_message_cannot_hide_a_missing_final_one(
    tmp_path: Path,
) -> None:
    """An earlier message whose text is a prefix of the missing final
    reply must not count as the final reply being present."""
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    transcript = tmp_path / "transcript.jsonl"
    events = _claude_turn_events(blocks=[
        {"type": "text", "text": "(◕‿◕) Done."},
        {"type": "tool_use", "id": "t0", "name": "Bash", "input": {}},
    ])
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    _fire_claude_hook(bash, hook, transcript, "(◕‿◕) Done. Here are the results.")
    assert [r["assistant_text"] for r in _read_jsonl(journal)] == [
        "Done.", "Done. Here are the results.",
    ]


def test_stop_refire_does_not_rejournal_earlier_messages(tmp_path: Path) -> None:
    """Another Stop hook blocks, the agent continues under the same
    turn, Stop fires again: the second walk re-emits the first fire's
    messages, and they must not land twice."""
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    transcript = tmp_path / "transcript.jsonl"
    first = _claude_turn_events(blocks=[{"type": "text", "text": "(◕‿◕) part one"}])
    transcript.write_text("\n".join(json.dumps(e) for e in first) + "\n")
    _fire_claude_hook(bash, hook, transcript, None)
    more = _claude_turn_events(blocks=[
        {"type": "text", "text": "(◕‿◕) part one"},
        {"type": "text", "text": "(＾▽＾) part two"},
    ])
    transcript.write_text("\n".join(json.dumps(e) for e in more) + "\n")
    _fire_claude_hook(bash, hook, transcript, None)
    assert [r["assistant_text"] for r in _read_jsonl(journal)] == ["part one", "part two"]


def test_grok_stop_refire_does_not_rejournal(tmp_path: Path) -> None:
    session = _grok_session(tmp_path, _grok_turn("p1", "q", ["(◕‿◕) A"], 1_792_000_000_000))
    payload = {"reason": "end_turn", "promptId": "p1", "stopHookActive": False}
    _fire_grok(tmp_path, session, payload)
    (session / "updates.jsonl").write_text("\n".join(json.dumps(u) for u in (
        _grok_turn("p1", "q", ["(◕‿◕) A", "(＾▽＾) B"], 1_792_000_000_000)
    )) + "\n")
    rows = _fire_grok(tmp_path, session, {**payload, "stopHookActive": True})
    assert [r["assistant_text"] for r in rows] == ["A", "B"]


def test_one_run_keeps_two_identical_rows(tmp_path: Path) -> None:
    """The re-fire guard compares against the journal as it stood
    before this run, so a message genuinely repeated within one turn
    — same text, and for Hermes the same fire-time ts — still lands
    twice."""
    bash, _ = _require_tools()
    from llmoji.providers import HermesProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(HermesProvider, journal, tmp_path / "hooks")
    payload = _hermes_payload(user_message="go", conversation_history=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "(◕‿◕) checking"},
        {"role": "assistant", "content": "(◕‿◕) checking"},
    ])
    r = subprocess.run([bash, str(hook)], input=json.dumps(payload),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(_read_jsonl(journal)) == 2


def test_codex_trailing_newlines_match_between_hook_and_replay(tmp_path: Path) -> None:
    """Command substitution drops trailing newlines in the live hook;
    the replay must too, or the two copies of a message never dedup."""
    events = _codex_item_events(["(◕‿◕) done.\n\n"])
    events[2]["payload"]["content"][0]["text"] = "do the thing\n"
    bash_rows, bf_rows = _run_codex_both(tmp_path, events)
    _assert_parity("codex trailing newline", bash_rows, bf_rows, True)
    assert bash_rows[0]["assistant_text"] == "done."
    assert bash_rows[0]["user_text"] == "do the thing"


def test_codex_turn_without_a_user_message_carries_the_last_prompt(
    tmp_path: Path,
) -> None:
    """A turn Codex starts by itself (a goal step, a subagent's result)
    has no user message; its rows carry the prompt that set it going,
    in the hook and the replay alike."""
    events = _codex_item_events(["(◕‿◕) first turn"])
    events.append({"type": "turn_context", "timestamp": "2026-10-25T00:01:00.000Z",
                   "payload": {"turn_id": "t2", "model": "gpt-test", "cwd": "/test"}})
    events.append({"type": "event_msg", "timestamp": "2026-10-25T00:01:05.000Z", "payload": {
        "type": "item_completed", "turn_id": "t2",
        "item": {"type": "AgentMessage", "id": "m2",
                 "content": [{"type": "Text", "text": "(＾▽＾) goal step"}]}}})
    bash_rows, bf_rows = _run_codex_both(tmp_path, events)
    bf_rows = [r for r in bf_rows if r["ts"] >= "2026-10-25T00:01:00"]
    _assert_parity("codex goal turn", bash_rows, bf_rows, True)
    assert bash_rows[0]["user_text"] == "do the thing"


def test_stop_refire_uses_each_earlier_row_once(tmp_path: Path) -> None:
    """The re-fire guard matches as a multiset: one earlier copy of a
    row absorbs one repeat, and a further identical row still lands."""
    bash, _ = _require_tools()
    from llmoji.providers import ClaudeCodeProvider

    journal = tmp_path / "journal.jsonl"
    hook = _render_hook(ClaudeCodeProvider, journal, tmp_path / "hooks")
    transcript = tmp_path / "transcript.jsonl"
    events = _claude_turn_events(blocks=[{"type": "text", "text": "(◕‿◕) same"}])
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    _fire_claude_hook(bash, hook, transcript, None)
    twin = dict(events[-1], uuid="a-twin")  # byte-identical row: same ts, same text
    transcript.write_text("\n".join(json.dumps(e) for e in [*events, twin]) + "\n")
    _fire_claude_hook(bash, hook, transcript, None)
    assert len(_read_jsonl(journal)) == 2
