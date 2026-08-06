"""Tests for ``llmoji install --soft`` / ``--hard``.

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
    from llmoji.synth_prompts import NUDGE_MESSAGE

    out = HookInstaller._merge_soft_doc("", NUDGE_MESSAGE)
    # ``# Kaomoji`` heading + blank line + message.
    assert out.startswith(f"{SOFT_DOC_HEADING}\n\n{NUDGE_MESSAGE}")
    # File ends with single trailing newline.
    assert out.endswith("\n")
    # No HTML comment markers — plain markdown.
    assert "<!--" not in out


def test_merge_appends_with_blank_separator_to_existing_content():
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING
    from llmoji.synth_prompts import NUDGE_MESSAGE

    existing = "user prose here\nmore prose\n"
    out = HookInstaller._merge_soft_doc(existing, NUDGE_MESSAGE)
    # User content preserved verbatim at the front.
    assert out.startswith(existing)
    # Blank-line separator between existing content and our heading.
    assert f"{existing}\n{SOFT_DOC_HEADING}" in out


def test_merge_idempotent_on_same_message():
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import NUDGE_MESSAGE

    once = HookInstaller._merge_soft_doc("user prose\n", NUDGE_MESSAGE)
    twice = HookInstaller._merge_soft_doc(once, NUDGE_MESSAGE)
    assert twice == once


def test_merge_replaces_legacy_block():
    """A re-run after a wording change: a doc carrying a legacy block,
    merged with the current message, comes out identical to a fresh
    merge — the legacy block was cleanly replaced, not duplicated, and
    user prose is untouched.
    """
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING
    from llmoji.synth_prompts import NUDGE_MESSAGE, _LEGACY_NUDGE_MESSAGES

    after_legacy = HookInstaller._merge_soft_doc(
        "prose\n", _LEGACY_NUDGE_MESSAGES[0]
    )
    after_current = HookInstaller._merge_soft_doc(after_legacy, NUDGE_MESSAGE)
    fresh = HookInstaller._merge_soft_doc("prose\n", NUDGE_MESSAGE)
    assert after_current == fresh
    # Exactly one block — no leftover legacy heading.
    assert after_current.count(SOFT_DOC_HEADING) == 1
    assert after_current.startswith("prose\n")


def test_strip_removes_canonical_block_and_separator():
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import NUDGE_MESSAGE

    after_install = HookInstaller._merge_soft_doc("user prose\n", NUDGE_MESSAGE)
    after_strip = HookInstaller._strip_soft_doc(after_install)
    assert after_strip == "user prose\n"


def test_strip_finds_legacy_block():
    """Uninstall doesn't know which llmoji version did the install;
    it strips a block carrying any legacy wording too."""
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import _LEGACY_NUDGE_MESSAGES

    for legacy in _LEGACY_NUDGE_MESSAGES:
        after_legacy_install = HookInstaller._merge_soft_doc("p\n", legacy)
        assert HookInstaller._strip_soft_doc(after_legacy_install) == "p\n"


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
    from llmoji.synth_prompts import NUDGE_MESSAGE

    after_install = HookInstaller._merge_soft_doc("prose\n", NUDGE_MESSAGE)
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
    from llmoji.synth_prompts import NUDGE_MESSAGE

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    assert not doc.exists()
    p.install_soft()
    text = doc.read_text()
    assert NUDGE_MESSAGE in text
    assert "# Kaomoji" in text
    # Journal-write hook IS created — both modes capture data.
    assert p.hook_path.exists()
    # Per-turn nudge hook is NOT created in soft mode.
    assert p.nudge_hook_path is not None
    assert not p.nudge_hook_path.exists()


def test_install_hard_creates_journal_AND_nudge_hooks(tmp_path: Path):
    """Hard mode installs both hooks. No doc edit."""
    from llmoji.providers import get_provider
    from llmoji.synth_prompts import NUDGE_MESSAGE

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_hard()
    assert p.hook_path.exists()
    assert p.nudge_hook_path is not None
    assert p.nudge_hook_path.exists()
    # No doc edit in hard mode.
    if doc.exists():
        assert NUDGE_MESSAGE not in doc.read_text()


def test_install_soft_idempotent(tmp_path: Path):
    from llmoji.providers import get_provider

    p = get_provider("codex")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_soft()
    once = doc.read_text()
    p.install_soft()
    twice = doc.read_text()
    assert once == twice


def test_install_soft_replaces_legacy_block(tmp_path: Path):
    """Upgrade scenario: a doc carries a legacy soft-doc block from an
    older llmoji. Re-running install strips the legacy block and writes
    the current wording; surrounding prose is untouched, and the doc
    ends up with exactly one block."""
    from llmoji.providers import get_provider
    from llmoji.providers.base import HookInstaller, SOFT_DOC_HEADING
    from llmoji.synth_prompts import NUDGE_MESSAGE, _LEGACY_NUDGE_MESSAGES

    p = get_provider("hermes")
    doc = _bind_provider_to_tmp(p, tmp_path)
    # Seed the doc with user prose + a legacy block, as an older
    # llmoji would have left it.
    doc.write_text(
        HookInstaller._merge_soft_doc(
            "# my agent\n\nbe nice.\n", _LEGACY_NUDGE_MESSAGES[0]
        )
    )

    p.install_soft()
    text = doc.read_text()
    assert NUDGE_MESSAGE in text
    assert text.count(SOFT_DOC_HEADING) == 1
    assert "be nice." in text


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


def test_status_treats_legacy_message_as_current(tmp_path: Path):
    """``soft_doc_current`` is wording-agnostic across versions: a
    block carrying a legacy wording still reads as installed +
    current, so status doesn't nag after a package upgrade. The user
    picks up the new wording on the next ``install`` re-run.
    """
    from llmoji.providers import get_provider
    from llmoji.providers.base import HookInstaller
    from llmoji.synth_prompts import _LEGACY_NUDGE_MESSAGES

    p = get_provider("claude_code")
    doc = _bind_provider_to_tmp(p, tmp_path)
    # Seed a legacy block, as an older llmoji would have written it.
    doc.write_text(
        HookInstaller._merge_soft_doc("prose\n", _LEGACY_NUDGE_MESSAGES[0])
    )
    s = p.status()
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
    from llmoji.synth_prompts import NUDGE_MESSAGE

    p = get_provider("opencode")
    doc = _bind_provider_to_tmp(p, tmp_path)
    p.install_soft()
    assert NUDGE_MESSAGE in doc.read_text()


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


def test_cli_install_one_dispatches_canonical_nudge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_install_one`` installs with the single canonical
    ``NUDGE_MESSAGE`` — the provider class default. Patch the install
    method to record the nudge_message observed at call time."""
    from llmoji import cli
    from llmoji.providers import HookInstaller
    from llmoji.synth_prompts import NUDGE_MESSAGE

    seen_messages: list[str] = []

    def fake_install_soft(self: HookInstaller) -> None:
        seen_messages.append(self.nudge_message)

    monkeypatch.setattr(HookInstaller, "install_soft", fake_install_soft)

    ok, _ = cli._install_one("claude_code", soft=True)
    assert ok
    assert seen_messages == [NUDGE_MESSAGE]


def test_cli_install_one_propagates_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmoji import cli
    from llmoji.providers import HookInstaller

    def fail(self: HookInstaller) -> None:
        _ = self
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(HookInstaller, "install_hard", fail)
    ok, err = cli._install_one("claude_code", soft=False)
    assert not ok
    assert err is not None
    assert "simulated" in err
