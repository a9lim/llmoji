"""Tests for ``llmoji install`` autodetect (no-arg path).

The CLI's no-arg ``install`` enumerates registered providers via
:meth:`HookInstaller.is_present` and installs each detected harness.
The tests mock at the ``is_present`` + ``_install_one`` level so we
don't need to spin up real harness home directories — the install
machinery itself is parity-tested elsewhere
(``test_pipeline_parity``).
"""

from __future__ import annotations

from pathlib import Path
import pytest


# Pyright reports `lambda self: True` lambdas as unused-self even with
# underscore renaming, so the fixture-side stand-ins for ``is_present``
# live as named functions instead.
def _always_present(_self: object) -> bool:
    _ = _self
    return True


def _never_present(_self: object) -> bool:
    _ = _self
    return False


# ---------------------------------------------------------------------------
# is_present() — the per-provider detection signal
# ---------------------------------------------------------------------------


def test_is_present_returns_false_when_parent_missing(tmp_path: Path) -> None:
    """A provider whose settings_path.parent doesn't exist is not detected."""
    from llmoji.providers import HookInstaller

    p = HookInstaller()
    p.settings_path = tmp_path / "nonexistent" / "settings.json"
    assert p.is_present() is False


def test_is_present_returns_true_when_parent_exists(tmp_path: Path) -> None:
    """A provider whose settings_path.parent exists is detected."""
    from llmoji.providers import HookInstaller

    parent = tmp_path / "harness-home"
    parent.mkdir()
    p = HookInstaller()
    p.settings_path = parent / "settings.json"
    assert p.is_present() is True


# ---------------------------------------------------------------------------
# _cmd_install — dispatch logic
# ---------------------------------------------------------------------------


def test_install_no_args_zero_detected_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No detected harnesses → exit 2 with a useful message naming the
    parent dirs we looked at, so the user can verify their assumption.
    """
    from llmoji import cli
    from llmoji.providers import HookInstaller

    monkeypatch.setattr(HookInstaller, "is_present", _never_present)

    parser = cli._build_parser()
    args = parser.parse_args(["install"])
    rc = cli._cmd_install(args)

    assert rc == 2
    err = capsys.readouterr().err
    assert "no harnesses detected" in err


def test_install_no_args_with_yes_installs_all_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--yes`` skips the prompt; every detected provider installs."""
    from llmoji import cli
    from llmoji.providers import PROVIDERS, HookInstaller

    # Patch every concrete provider class as well as the base —
    # subclasses with their own ``is_present`` override (OpenclawProvider)
    # bypass a base-class-only patch.
    monkeypatch.setattr(HookInstaller, "is_present", _always_present)
    from llmoji.providers import PROVIDERS as _ALL
    for _cls in _ALL.values():
        monkeypatch.setattr(_cls, "is_present", _always_present)

    installed: list[str] = []

    def fake_install_one(name: str) -> tuple[bool, str | None]:
        installed.append(name)
        return True, None

    monkeypatch.setattr(cli, "_install_one", fake_install_one)

    parser = cli._build_parser()
    args = parser.parse_args(["install", "--yes"])
    rc = cli._cmd_install(args)

    assert rc == 0
    assert set(installed) == set(PROVIDERS)


def test_install_no_args_partial_failure_returns_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single provider failing doesn't abort the rest of the batch.
    Exit 1 reports the failure(s) on stderr.
    """
    from llmoji import cli
    from llmoji.providers import HookInstaller

    # Patch every concrete provider class as well as the base —
    # subclasses with their own ``is_present`` override (OpenclawProvider)
    # bypass a base-class-only patch.
    monkeypatch.setattr(HookInstaller, "is_present", _always_present)
    from llmoji.providers import PROVIDERS as _ALL
    for _cls in _ALL.values():
        monkeypatch.setattr(_cls, "is_present", _always_present)

    def fake_install_one(name: str) -> tuple[bool, str | None]:
        if name == "codex":
            return False, "RuntimeError: simulated"
        return True, None

    monkeypatch.setattr(cli, "_install_one", fake_install_one)

    parser = cli._build_parser()
    args = parser.parse_args(["install", "--yes"])
    rc = cli._cmd_install(args)

    assert rc == 1
    err = capsys.readouterr().err
    assert "codex" in err
    assert "simulated" in err


def test_install_no_args_prompt_n_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``--yes``, answering 'n' to the prompt aborts before any
    install runs.
    """
    from llmoji import cli
    from llmoji.providers import HookInstaller

    # Patch every concrete provider class as well as the base —
    # subclasses with their own ``is_present`` override (OpenclawProvider)
    # bypass a base-class-only patch.
    monkeypatch.setattr(HookInstaller, "is_present", _always_present)
    from llmoji.providers import PROVIDERS as _ALL
    for _cls in _ALL.values():
        monkeypatch.setattr(_cls, "is_present", _always_present)

    installed: list[str] = []

    def fake_install_one(name: str) -> tuple[bool, str | None]:
        installed.append(name)
        return True, None

    monkeypatch.setattr(cli, "_install_one", fake_install_one)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    parser = cli._build_parser()
    args = parser.parse_args(["install"])
    rc = cli._cmd_install(args)

    assert rc == 1
    assert installed == []


def test_install_explicit_provider_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``llmoji install <provider>`` still works exactly as before — one
    target, no autodetect, no prompt, no detection check.
    """
    from llmoji import cli

    calls: list[str] = []

    def fake_install_one(name: str) -> tuple[bool, str | None]:
        calls.append(name)
        return True, None

    monkeypatch.setattr(cli, "_install_one", fake_install_one)

    parser = cli._build_parser()
    args = parser.parse_args(["install", "claude_code"])
    rc = cli._cmd_install(args)

    assert rc == 0
    assert calls == ["claude_code"]

