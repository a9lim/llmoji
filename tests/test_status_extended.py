"""Tests for the extended ``llmoji status`` surface.

Default ``status`` adds cheap health checks (stale-hook detection +
settings parseability) on top of the install-presence check.
``--stats`` walks journals once for kaomoji frequency tables and
row schema validation. ``--json`` emits the same shape as the human
output but parseable.

Provider install machinery is parity-tested elsewhere; these tests
fixture-build the artifacts directly (write a settings file, write a
hook file) so the assertions stay focused on the diagnostic logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# stale-hook detection
# ---------------------------------------------------------------------------


def test_main_hook_current_returns_false_for_stale_content(
    tmp_path: Path,
) -> None:
    """A hook file whose content doesn't match render_hook() is stale."""
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    p.hooks_dir = hook_dir
    p.hook_filename = "kaomoji-log.sh"
    # Plant a stale body on disk and stub render_hook to return
    # something else; the comparison should report False.
    p.hook_path.write_text("stale content\n")
    p.render_hook = lambda: "fresh content\n"  # type: ignore[method-assign]
    assert p._is_main_hook_current() is False


def test_main_hook_current_returns_true_for_matching_content(
    tmp_path: Path,
) -> None:
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    p.hooks_dir = hook_dir
    p.hook_filename = "kaomoji-log.sh"
    body = "exact rendered body\n"
    p.hook_path.write_text(body)
    p.render_hook = lambda: body  # type: ignore[method-assign]
    assert p._is_main_hook_current() is True


# ---------------------------------------------------------------------------
# settings parseability — JSON path
# ---------------------------------------------------------------------------


def test_check_settings_health_returns_none_for_missing_file(
    tmp_path: Path,
) -> None:
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    p.settings_path = tmp_path / "does-not-exist.json"
    assert p._check_settings_health() is None


def test_check_settings_health_returns_reason_for_bad_json(
    tmp_path: Path,
) -> None:
    """Unparseable JSON surfaces a non-None reason string."""
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    settings = tmp_path / "settings.json"
    settings.write_text("{ this is not json ")
    p.settings_path = settings
    why = p._check_settings_health()
    assert why is not None
    assert "invalid JSON" in why


def test_check_settings_health_returns_reason_for_non_object(
    tmp_path: Path,
) -> None:
    """JSON that parses but isn't an object is also corrupt."""
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    settings = tmp_path / "settings.json"
    settings.write_text('["not", "an", "object"]')
    p.settings_path = settings
    why = p._check_settings_health()
    assert why is not None
    assert "not an object" in why


# ---------------------------------------------------------------------------
# row schema validation
# ---------------------------------------------------------------------------


def test_validate_journal_row_accepts_canonical_shape() -> None:
    from llmoji.cli import _validate_journal_row

    row = {
        "ts": "2026-04-28T00:00:00Z",
        "model": "claude-opus-4",
        "cwd": "/tmp",
        "kaomoji": "(◕‿◕)",
        "user_text": "ping",
        "assistant_text": "(◕‿◕) hi",
    }
    assert _validate_journal_row(row) is None


def test_validate_journal_row_rejects_missing_field() -> None:
    from llmoji.cli import _validate_journal_row

    row = {
        "ts": "2026-04-28T00:00:00Z",
        "model": "m",
        "cwd": "/tmp",
        "kaomoji": "(◕‿◕)",
        # missing user_text
        "assistant_text": "hi",
    }
    why = _validate_journal_row(row)
    assert why is not None
    assert "user_text" in why


def test_validate_journal_row_rejects_wrong_type() -> None:
    from llmoji.cli import _validate_journal_row

    row = {
        "ts": "2026-04-28T00:00:00Z",
        "model": "m",
        "cwd": "/tmp",
        "kaomoji": "(◕‿◕)",
        "user_text": "ping",
        "assistant_text": 42,  # int, not str
    }
    why = _validate_journal_row(row)
    assert why is not None
    assert "assistant_text" in why
    assert "int" in why


def test_validate_journal_row_rejects_non_dict() -> None:
    from llmoji.cli import _validate_journal_row

    why = _validate_journal_row(["array", "row"])
    assert why is not None
    assert "list" in why


# ---------------------------------------------------------------------------
# stats walk — frequency tables + malformed-row counting
# ---------------------------------------------------------------------------


def _write_journal(path: Path, rows: list[dict[str, object] | str]) -> None:
    """Write a list of rows (dicts → json-encoded; raw strings → as-is)
    to a journal JSONL file. Strings let tests inject malformed rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_walk_journals_counts_and_canonicalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats walk respects the canonical bucketing (so `(◕‿◕)` and a
    near-variant fold together) and counts every well-formed row.
    """
    from llmoji import cli, paths

    # Redirect ~/.llmoji to a tmp tree so the generic-JSONL zone test
    # picks up only what we plant.
    monkeypatch.setattr(paths, "llmoji_home", lambda: tmp_path / ".llmoji")

    # Plant a generic-JSONL journal that a hookless harness might
    # produce (the zero-detected case for managed providers).
    journals_dir = tmp_path / ".llmoji" / "journals"
    rows: list[dict[str, object] | str] = [
        {
            "ts": "2026-04-28T00:00:00Z", "model": "m1", "cwd": "/tmp",
            "kaomoji": "(◕‿◕)", "user_text": "u", "assistant_text": "a",
        },
        {
            "ts": "2026-04-28T00:00:01Z", "model": "m1", "cwd": "/tmp",
            "kaomoji": "(◕‿◕)", "user_text": "u", "assistant_text": "b",
        },
        {
            "ts": "2026-04-28T00:00:02Z", "model": "m2", "cwd": "/tmp",
            "kaomoji": "(`・ω・´)", "user_text": "u", "assistant_text": "c",
        },
    ]
    _write_journal(journals_dir / "opencode.jsonl", rows)

    # Stub each managed provider class's journal_path to nonexistent
    # so the stats walk only sees the generic-JSONL plant. Concrete
    # provider classes carry ``journal_path`` as a class-level Path
    # attribute set at class-definition time, so a base-class override
    # gets shadowed — patch each subclass directly.
    from llmoji.providers import PROVIDERS as _REGISTRY

    for _name, _cls in _REGISTRY.items():
        monkeypatch.setattr(
            _cls, "journal_path", tmp_path / f"no-{_name}.jsonl",
        )

    stats = cli._walk_journals_for_stats(provider_filter=None)
    assert stats["rows_total"] == 3
    assert stats["rows_malformed"] == 0
    # Sources keyed by file stem for the generic journals.
    assert stats["by_source"] == {"opencode": 3}
    assert stats["by_source_model"] == {"m1": 2, "m2": 1}
    # Canonical bucketing — both kaomoji are distinct canonical forms.
    canon_dict = dict(stats["by_canonical_top"])
    assert canon_dict["(◕‿◕)"] == 2
    assert canon_dict["(`・ω・´)"] == 1


def test_walk_journals_counts_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad JSON + missing fields are counted as malformed and don't
    leak into the frequency tables."""
    from llmoji import cli, paths

    monkeypatch.setattr(paths, "llmoji_home", lambda: tmp_path / ".llmoji")

    journals_dir = tmp_path / ".llmoji" / "journals"
    rows: list[dict[str, object] | str] = [
        # Valid.
        {
            "ts": "2026-04-28T00:00:00Z", "model": "m1", "cwd": "/tmp",
            "kaomoji": "(◕‿◕)", "user_text": "u", "assistant_text": "a",
        },
        # Bad JSON.
        "{not valid json",
        # Missing field.
        {
            "ts": "2026-04-28T00:00:01Z", "model": "m1", "cwd": "/tmp",
            "kaomoji": "(◕‿◕)", "user_text": "u",
            # missing assistant_text
        },
    ]
    _write_journal(journals_dir / "opencode.jsonl", rows)

    from llmoji.providers import PROVIDERS as _REGISTRY

    for _name, _cls in _REGISTRY.items():
        monkeypatch.setattr(
            _cls, "journal_path", tmp_path / f"no-{_name}.jsonl",
        )

    stats = cli._walk_journals_for_stats(provider_filter=None)
    assert stats["rows_total"] == 3
    assert stats["rows_malformed"] == 2
    assert len(stats["malformed_examples"]) == 2
    # Frequency tables only count valid rows.
    assert stats["by_source"] == {"opencode": 1}


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------


def test_status_json_emits_parseable_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json`` output must round-trip through json.loads."""
    from llmoji import cli, paths

    monkeypatch.setattr(paths, "llmoji_home", lambda: tmp_path / ".llmoji")

    parser = cli._build_parser()
    args = parser.parse_args(["status", "--json"])
    rc = cli._cmd_status(args)
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "llmoji_home" in parsed
    assert "providers" in parsed
    assert isinstance(parsed["providers"], list)
    # Per-provider keys we promised:
    p0 = parsed["providers"][0]
    for key in (
        "name", "installed", "main_installed", "main_hook_current",
        "settings_parse_error", "hook_path", "settings_path",
        "journal_path", "journal_exists", "journal_bytes",
    ):
        assert key in p0, f"missing key: {key}"
