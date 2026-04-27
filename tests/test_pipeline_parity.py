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

# Fields the journal row is built from. We compare these between the
# bash hook output and the backfill output. ``ts`` differs by design
# (bash stamps now, backfill stamps historical) so it's excluded.
_PARITY_FIELDS = ("kaomoji", "model", "cwd", "user_text", "assistant_text")


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
) -> None:
    if not expect_row:
        assert bash_rows == [] and bf_rows == [], (
            f"{desc}: expected both pipelines to skip, got "
            f"bash={bash_rows!r}, backfill={bf_rows!r}"
        )
        return
    assert len(bash_rows) == 1 and len(bf_rows) == 1, (
        f"{desc}: expected 1 row each, got "
        f"bash={len(bash_rows)}, backfill={len(bf_rows)}"
    )
    b, f = bash_rows[0], bf_rows[0]
    for k in _PARITY_FIELDS:
        assert b[k] == f[k], (
            f"{desc} divergence on {k!r}:\n"
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
    *, user_items: list[str], assistant_text: str
) -> list[dict[str, Any]]:
    """Build a minimal Codex rollout event list for one turn.

    ``user_items`` becomes a sequence of ``response_item`` events
    (role=user) in declared order; the synthesis pipeline picks the
    last surviving one after dropping injected prefixes.
    """
    events: list[dict[str, Any]] = [
        {"type": "session_meta", "payload": {"cwd": "/test"}},
        {
            "type": "turn_context",
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
    events.append({
        "type": "event_msg",
        "timestamp": "2026-04-27T00:00:00Z",
        "payload": {
            "type": "task_complete",
            "turn_id": "t1",
            "last_agent_message": assistant_text,
            "completed_at": "2026-04-27T00:00:00Z",
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
        user_items=user_items, assistant_text=assistant
    )
    rollout.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    # Codex Stop event: last_assistant_message + transcript_path
    # mirrors what the Codex CLI passes the Stop hook in production.
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
