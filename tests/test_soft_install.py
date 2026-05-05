"""Tests for ``llmoji install --soft`` / ``--hard`` / ``--long``.

The 2.0 install rework split the lifecycle into two mutually-
exclusive placements that share the journal-write hook:

  - ``install_hard`` — write the journal-write Stop hook AND the
    per-turn nudge hook. The v1 behavior.
  - ``install_soft`` — write the journal-write Stop hook AND append
    a ``# Kaomoji`` heading + the nudge wording to the harness's
    persistent system-prompt doc (``~/.claude/CLAUDE.md`` /
    ``~/.codex/AGENTS.md`` / ``~/.hermes/SOUL.md`` /
    ``~/.config/opencode/AGENTS.md`` /
    ``~/.openclaw/workspace/SOUL.md``). No per-turn nudge hook.

Both modes capture journal data — the journal-write hook is the
data-capture invariant. The two modes only differ in where the
kaomoji-leading reminder is delivered (per-turn vs. identity slot).

Plus the orthogonal ``--long`` flag that swaps the v1 one-sentence
wording for the v7 introspection framing in either placement.
``--soft`` and ``--hard`` are mutually exclusive (argparse-enforced).

The tests below cover the soft-doc planner's pure functions
directly (no filesystem) and then exercise the live install paths
in a temp dir for a couple of representative providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from llmoji.providers import HookInstaller as HookInstaller  # noqa: F401


# ---------------------------------------------------------------------------
# Pure-function planner — _merge_soft_doc / _strip_soft_doc
# ---------------------------------------------------------------------------


def test_merge_into_empty_doc_writes_heading_block():
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    out = HookInstaller._merge_soft_doc("", SHORT_NUDGE_MESSAGE)
    # ``# Kaomoji`` heading + blank line + message.
    assert out.startswith(f"{SOFT_DOC_HEADING}\n\n{SHORT_NUDGE_MESSAGE}")
    # File ends with single trailing newline.
    assert out.endswith("\n")
    # No HTML comment markers — plain markdown.
    assert "<!--" not in out


def test_merge_appends_with_blank_separator_to_existing_content():
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    existing = "user prose here\nmore prose\n"
    out = HookInstaller._merge_soft_doc(existing, SHORT_NUDGE_MESSAGE)
    # User content preserved verbatim at the front.
    assert out.startswith(existing)
    # Blank-line separator between existing content and our heading.
    assert f"{existing}\n{SOFT_DOC_HEADING}" in out


def test_merge_idempotent_on_same_message():
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    once = HookInstaller._merge_soft_doc("user prose\n", SHORT_NUDGE_MESSAGE)
    twice = HookInstaller._merge_soft_doc(once, SHORT_NUDGE_MESSAGE)
    assert twice == once


def test_merge_swaps_short_to_long_block():
    """``--long`` toggle: re-running with the long message strips the
    short block and appends the long one. No disturbance to user prose.
    """
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE, SHORT_NUDGE_MESSAGE

    after_short = HookInstaller._merge_soft_doc("prose\n", SHORT_NUDGE_MESSAGE)
    after_long = HookInstaller._merge_soft_doc(after_short, LONG_NUDGE_MESSAGE)
    assert SHORT_NUDGE_MESSAGE not in after_long
    assert LONG_NUDGE_MESSAGE in after_long
    assert after_long.startswith("prose\n")


def test_strip_removes_canonical_block_and_separator():
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    after_install = HookInstaller._merge_soft_doc("user prose\n", SHORT_NUDGE_MESSAGE)
    after_strip = HookInstaller._strip_soft_doc(after_install)
    assert after_strip == "user prose\n"


def test_strip_finds_either_canonical_variant():
    """Uninstall doesn't know whether the user installed short or
    long; it tries both canonical wordings."""
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE

    after_long_install = HookInstaller._merge_soft_doc("p\n", LONG_NUDGE_MESSAGE)
    assert HookInstaller._strip_soft_doc(after_long_install) == "p\n"


def test_strip_no_op_when_no_canonical_block():
    from llmoji.providers.base import HookInstaller

    text = "totally vanilla user content\n"
    assert HookInstaller._strip_soft_doc(text) == text


def test_strip_no_op_when_block_is_hand_edited():
    """Conservative: an edited block (heading present, body
    different) survives uninstall — we won't clobber the user's
    edits."""
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING

    edited = f"prose\n\n{SOFT_DOC_HEADING}\n\nthe user changed this.\n"
    assert HookInstaller._strip_soft_doc(edited) == edited


def test_strip_idempotent():
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    after_install = HookInstaller._merge_soft_doc("prose\n", SHORT_NUDGE_MESSAGE)
    once = HookInstaller._strip_soft_doc(after_install)
    twice = HookInstaller._strip_soft_doc(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Live install paths — install_soft / install_hard / uninstall
# ---------------------------------------------------------------------------


def _bind_provider_to_tmp(
    provider: "HookInstaller", tmp_path: Path,
) -> Path:
    """Repoint a provider's filesystem attrs at a tmp dir so install
    operations don't touch the user's real home. Each provider has
    its own attribute set; this helper handles the bash + plugin
    flavors uniformly. Returns the soft-doc path so callers can
    reference it without going through ``Optional``-typed attr
    chains (the class declares it ``Path | None``).
    """
    from llmoji.providers import HookInstaller, PluginInstaller

    doc_path = tmp_path / "DOC.md"
    if isinstance(provider, PluginInstaller):
        provider.plugin_dir = tmp_path / "plugin"
        provider.journal_path = tmp_path / "journal.jsonl"
        provider.settings_path = tmp_path / "settings.json"
        doc_path = tmp_path / "SOUL.md"
        provider.system_prompt_doc_path = doc_path
        return doc_path
    assert isinstance(provider, HookInstaller)
    provider.hooks_dir = tmp_path / "hooks"
    provider.settings_path = tmp_path / "settings.json"
    provider.journal_path = tmp_path / "journal.jsonl"
    provider.system_prompt_doc_path = doc_path
    return doc_path


def test_install_soft_creates_doc_AND_journal_hook(tmp_path: Path):
    """Soft mode installs the journal-write Stop hook (so kaomoji
    capture still happens) AND appends the soft-doc block. The only
    thing missing vs hard is the per-turn nudge hook."""
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    assert not doc.exists()
    p.install_soft()
    text = doc.read_text()
    assert SHORT_NUDGE_MESSAGE in text
    assert "# Kaomoji" in text
    # Journal-write hook IS created — both modes capture data.
    assert p.hook_path.exists()
    # Per-turn nudge hook is NOT created in soft mode.
    assert p.nudge_hook_path is not None
    assert not p.nudge_hook_path.exists()


def test_install_hard_creates_journal_AND_nudge_hooks(tmp_path: Path):
    """Hard mode installs both hooks. No doc edit."""
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_hard()
    assert p.hook_path.exists()
    assert p.nudge_hook_path is not None
    assert p.nudge_hook_path.exists()
    # No doc edit in hard mode.
    if doc.exists():
        assert SHORT_NUDGE_MESSAGE not in doc.read_text()


def test_install_soft_with_long_message(tmp_path: Path):
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE, SHORT_NUDGE_MESSAGE

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.nudge_message = LONG_NUDGE_MESSAGE
    p.install_soft()
    text = doc.read_text()
    assert LONG_NUDGE_MESSAGE in text
    assert SHORT_NUDGE_MESSAGE not in text


def test_install_soft_idempotent(tmp_path: Path):
    from llmoji.providers import get_provider

    p = get_provider("codex")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_soft()
    once = doc.read_text()
    p.install_soft()
    twice = doc.read_text()
    assert once == twice


def test_install_soft_long_toggle_swaps_block(tmp_path: Path):
    """The flagship ``--long`` toggle scenario: user installs short,
    then re-runs with --long. Old block stripped, new block appended;
    surrounding prose untouched."""
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE, SHORT_NUDGE_MESSAGE

    p = get_provider("hermes")
    doc = _bind_provider_to_tmp(p, tmp_path)
    # Pre-existing user content in the doc.
    doc.write_text("# my agent\n\nbe nice.\n")

    p.nudge_message = SHORT_NUDGE_MESSAGE
    p.install_soft()
    after_short = doc.read_text()
    assert SHORT_NUDGE_MESSAGE in after_short
    assert "be nice." in after_short

    p.nudge_message = LONG_NUDGE_MESSAGE
    p.install_soft()
    after_long = doc.read_text()
    assert LONG_NUDGE_MESSAGE in after_long
    assert SHORT_NUDGE_MESSAGE not in after_long
    assert "be nice." in after_long


def test_uninstall_removes_soft_block_preserves_prose(tmp_path: Path):
    from llmoji.providers import get_provider

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    doc.write_text("user content\n")
    p.install_soft()
    # Verify install worked.
    assert "user content" in doc.read_text()
    p.uninstall()
    # User content survives; our block is gone.
    final = doc.read_text()
    assert final == "user content\n"


def test_uninstall_removes_doc_entirely_if_empty_apart_from_block(tmp_path: Path):
    """If the user never had any prose in the doc, uninstall should
    leave nothing behind (no orphan empty file)."""
    from llmoji.providers import get_provider

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_soft()
    assert doc.exists()
    p.uninstall()
    assert not doc.exists()


def test_status_surfaces_soft_install_state(tmp_path: Path):
    from llmoji.providers import get_provider

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    # Pre-soft: clean.
    s = p.status()
    assert s.soft_installed is False
    assert s.system_prompt_doc_path == doc

    p.install_soft()
    s = p.status()
    assert s.soft_installed is True
    assert s.soft_doc_current is True
    # Soft mode installs the journal-write hook but not the nudge
    # hook — ``installed`` rolls up on hard-completeness so it stays
    # False (the per-turn nudge isn't there).
    assert s.installed is False
    # But the main hook IS installed (data-capture invariant).
    assert s.main_installed is True


def test_status_treats_either_canonical_message_as_current(tmp_path: Path):
    """``soft_doc_current`` is variant-agnostic: a block carrying
    either ``SHORT_NUDGE_MESSAGE`` or ``LONG_NUDGE_MESSAGE`` reads
    as current. The CLI's ``--long`` choice isn't persisted on disk
    — re-installs are cheap and idempotent — so status can't (and
    doesn't try to) surface a "you installed short and your default
    is long now" warning.
    """
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE, SHORT_NUDGE_MESSAGE

    p = get_provider("claude_code")
    _bind_provider_to_tmp(p, tmp_path)

    # Short-installed, then queried with default (short) message —
    # current.
    p.nudge_message = SHORT_NUDGE_MESSAGE
    p.install_soft()
    s = p.status()
    assert s.soft_installed is True
    assert s.soft_doc_current is True

    # Long-installed (CLI passes --long via instance attr), queried
    # via a fresh provider that defaults to short — still current,
    # because the on-disk content matches a canonical message.
    p.nudge_message = LONG_NUDGE_MESSAGE
    p.install_soft()
    fresh = get_provider("claude_code")
    _bind_provider_to_tmp(fresh, tmp_path)  # same tmp dir → same doc
    s = fresh.status()
    assert s.soft_installed is True
    assert s.soft_doc_current is True


def test_status_flags_hand_edited_block_as_stale(tmp_path: Path):
    """A block whose heading is present but whose body matches
    NEITHER canonical wording surfaces as stale — typically because
    the user (or another tool) edited the body."""
    from llmoji.providers import get_provider
    from llmoji.providers.base import SOFT_DOC_HEADING

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    # Hand-write a block with a non-canonical body.
    doc.write_text(f"prose\n\n{SOFT_DOC_HEADING}\n\ntotally edited\n")

    s = p.status()
    # ``soft_installed`` is False here — we only claim ownership of
    # canonical-content blocks. A hand-edited body doesn't count as
    # ours and ``soft_doc_current`` reflects no install gap.
    assert s.soft_installed is False
    # But the heading is present — surface that for the user via
    # ``soft_doc_current=False`` (the heading exists but no canonical
    # body matches).
    assert s.soft_doc_current is False


def test_install_soft_for_plugin_provider(tmp_path: Path):
    """Plugin providers (opencode, openclaw) carry their own
    ``system_prompt_doc_path`` — the soft-doc edit works the same
    way regardless of installer flavor."""
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import SHORT_NUDGE_MESSAGE

    p = get_provider("opencode")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_soft()
    assert SHORT_NUDGE_MESSAGE in doc.read_text()


# ---------------------------------------------------------------------------
# Long-prompt cross-corpus guard
# ---------------------------------------------------------------------------


def test_long_nudge_message_matches_introspection_v7():
    """``LONG_NUDGE_MESSAGE`` is the v7 introspection prompt baked
    verbatim from ``llmoji-study/preambles/introspection_v7.txt``.
    The two repos must stay in lockstep — research-side analysis of
    --long submissions assumes the dataset rows were produced under
    the exact wording in the study repo's preamble file.

    The check is best-effort: skips when the study repo isn't
    checked out alongside (CI doesn't ship it), but on a dev box
    with both repos sitting next to each other this guards the
    drift case.
    """
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE

    candidates = [
        Path.home() / "Work" / "llmoji-study" / "preambles" / "introspection_v7.txt",
        Path(__file__).resolve().parents[2] / "llmoji-study" / "preambles" / "introspection_v7.txt",
    ]
    for path in candidates:
        if path.exists():
            file_text = path.read_text().strip()
            assert LONG_NUDGE_MESSAGE == file_text, (
                f"LONG_NUDGE_MESSAGE drift vs {path} — re-sync the "
                f"baked literal in synth_prompts.py with the study "
                f"repo's preamble file."
            )
            return
    pytest.skip("llmoji-study preamble not available; drift check skipped")


# ---------------------------------------------------------------------------
# CLI flag plumbing — _cmd_install validation
# ---------------------------------------------------------------------------


def test_cli_install_requires_a_mode_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``llmoji install`` with no ``--soft`` / ``--hard`` must error
    via argparse's ``mutually_exclusive_group(required=True)``. We
    catch the SystemExit argparse raises and verify the help text
    names both flags."""
    from llmoji import cli

    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["install"])
    err = capsys.readouterr().err
    assert "--soft" in err and "--hard" in err


def test_cli_install_rejects_both_mode_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--soft`` and ``--hard`` are mutually exclusive — argparse
    rejects passing both."""
    from llmoji import cli

    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["install", "--soft", "--hard"])
    err = capsys.readouterr().err
    assert "not allowed" in err.lower() or "argument" in err.lower()


def test_cli_install_one_dispatches_long_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--long`` must reach the provider as ``LONG_NUDGE_MESSAGE`` on
    the per-instance attr. Patch the install methods to record the
    nudge_message they observe at call time."""
    from llmoji import cli
    from llmoji.providers import HookInstaller
    from llmoji.synth_prompts import LONG_NUDGE_MESSAGE

    seen_messages: list[str] = []

    def fake_install_soft(self: HookInstaller) -> None:
        seen_messages.append(self.nudge_message)

    monkeypatch.setattr(HookInstaller, "install_soft", fake_install_soft)

    ok, _ = cli._install_one("claude_code", soft=True, long=True)
    assert ok
    assert seen_messages == [LONG_NUDGE_MESSAGE]


def test_cli_install_one_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmoji import cli
    from llmoji.providers import HookInstaller

    def fail(self: HookInstaller) -> None:
        _ = self
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(HookInstaller, "install_hard", fail)
    ok, err = cli._install_one("claude_code", soft=False, long=False)
    assert not ok
    assert err is not None
    assert "simulated" in err
