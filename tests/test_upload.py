"""Shipping a bundle: the allowlist, the archive, the HF branch push.

``huggingface_hub.HfApi`` and the token decryption are faked, so nothing
here reaches the Hub or touches the real shared credential. The one test
that runs the real decryption does so with a blob and password made for
the test.
"""

from __future__ import annotations

import getpass
import io
import json
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmoji import _shared_token, paths, upload
from llmoji.upload import (
    BUNDLE_DATA_SUFFIX,
    BUNDLE_TOPLEVEL_ALLOWLIST,
    HF_REPO,
    UPLOAD_PASSWORD_ENV,
    BundleAllowlistError,
    UploadAuthError,
    bundle_files,
    submitter_id,
    upload_hf,
    write_archive,
)

PASSWORD, TOKEN = "test-password", "hf_test_shared_token"
REAL_DECRYPT = _shared_token.decrypt_with_password
ROW = {"kaomoji": "(◕‿◕)", "count": 1, "synthesis": {
    "primary_affect": ["cheery"], "stance_modality_function": ["helpful", "eager", "wry"]}}


def _bundle(tmp_path: Path, *models: str) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"llmoji_version": "test"}) + "\n")
    for name in models or ("a", "b"):
        (bundle / f"{name}.jsonl").write_text(json.dumps(ROW, ensure_ascii=False) + "\n")
    return bundle


# --- bundle_files -----------------------------------------------------------


def test_allowlist_constants_are_the_flat_shape() -> None:
    assert BUNDLE_TOPLEVEL_ALLOWLIST == ("manifest.json",)
    assert BUNDLE_DATA_SUFFIX == ".jsonl"


def test_bundle_files_lists_the_manifest_first_then_data_by_name(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "zeta", "alpha", "mid")
    assert [p.name for p in bundle_files(bundle)] == [
        "manifest.json", "alpha.jsonl", "mid.jsonl", "zeta.jsonl"]


def test_a_manifest_alone_is_a_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    for p in bundle.glob("*.jsonl"):
        p.unlink()
    assert [p.name for p in bundle_files(bundle)] == ["manifest.json"]


def test_bundle_files_needs_a_bundle(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no bundle"):
        bundle_files(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no manifest"):
        bundle_files(empty)
    (empty / "a.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="no manifest"):
        bundle_files(empty)


def _plant(bundle: Path, kind: str) -> None:
    """Put one thing outside the allowlist into ``bundle``."""
    leak = bundle.parent / "leak.jsonl"
    leak.write_text("{}\n")
    if kind == "file":
        (bundle / "stray.txt").write_text("would leak")
    elif kind == "dir":
        (bundle / "sub").mkdir()
        (bundle / "sub" / "inner.jsonl").write_text("{}\n")
    elif kind == "dir named like data":
        (bundle / "trick.jsonl").mkdir()
    elif kind == "manifest copy":
        (bundle / "manifest.json.bak").write_text("{}")
    elif kind == "gzipped data":
        (bundle / "a.jsonl.gz").write_bytes(b"\x1f\x8b")
    elif kind == "linked manifest":
        (bundle / "manifest.json").unlink()
        (bundle / "manifest.json").symlink_to(leak)
    elif kind == "linked data":
        (bundle / "linked.jsonl").symlink_to(leak)
    elif kind == "linked dir":
        (bundle / "linked").symlink_to(bundle.parent, target_is_directory=True)
    elif kind == "dangling link":
        (bundle / "dangling.jsonl").symlink_to(bundle.parent / "missing")
    else:
        raise ValueError(kind)


KINDS = ["file", "dir", "dir named like data", "manifest copy", "gzipped data",
         "linked manifest", "linked data", "linked dir", "dangling link"]


@pytest.mark.parametrize("kind", KINDS)
def test_anything_outside_the_allowlist_is_refused(tmp_path: Path, kind: str) -> None:
    bundle = _bundle(tmp_path)
    _plant(bundle, kind)
    with pytest.raises(BundleAllowlistError, match="outside the allowlist"):
        bundle_files(bundle)


def test_the_refusal_names_every_offender(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _plant(bundle, "file")
    _plant(bundle, "dir")
    with pytest.raises(BundleAllowlistError, match=r"stray\.txt, sub are outside"):
        bundle_files(bundle)


# --- write_archive ----------------------------------------------------------


def test_archive_holds_the_allowlisted_files_flat(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    out = write_archive(bundle, out_path=tmp_path / "out.tar.gz")
    assert out == tmp_path / "out.tar.gz"
    with tarfile.open(out) as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == ["manifest.json", "a.jsonl", "b.jsonl"]
        assert all(m.isfile() for m in members)
        for m in members:
            assert tar.extractfile(m).read() == (bundle / m.name).read_bytes()  # type: ignore[union-attr]


def test_archive_defaults_to_a_timestamped_file_beside_the_bundle(tmp_path: Path) -> None:
    out = write_archive(_bundle(tmp_path))
    assert out.parent == tmp_path
    assert re.fullmatch(r"bundle-\d{8}T\d{6}Z\.tar\.gz", out.name)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("kind", ["file", "dir", "linked data"])
def test_archive_refuses_a_dirty_bundle_and_writes_nothing(tmp_path: Path, kind: str) -> None:
    bundle = _bundle(tmp_path)
    _plant(bundle, kind)
    out = tmp_path / "out.tar.gz"
    with pytest.raises(BundleAllowlistError):
        write_archive(bundle, out_path=out)
    assert not out.exists()


# --- submitter_id -----------------------------------------------------------


def test_submitter_id_is_32_hex_and_stable() -> None:
    sid = submitter_id()
    assert re.fullmatch(r"[0-9a-f]{32}", sid)
    assert submitter_id() == sid
    salt = paths.salt_path().read_text().strip()
    assert salt and salt not in sid  # the salt never leaves the machine


def test_submitter_id_follows_the_salt_and_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = submitter_id()
    with monkeypatch.context() as m:
        m.setattr(upload, "package_version", lambda: "999.0.0")
        assert submitter_id() != sid
    assert submitter_id() == sid
    paths.salt_path().write_text("another-salt\n")
    assert submitter_id() != sid


def test_an_empty_salt_file_is_replaced(tmp_path: Path) -> None:
    paths.ensure_home()
    paths.salt_path().write_text("\n")
    sid = submitter_id()
    assert paths.salt_path().read_text().strip() and submitter_id() == sid


# --- upload_hf --------------------------------------------------------------


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake Hub and a fake decryption. A prompt for the password, or a
    use of the real shared blob, fails the test unless it set one up."""
    calls: list[tuple[str, Any]] = []

    class FakeApi:
        def __init__(self, token: str | None = None, **kw: Any) -> None:
            calls.append(("init", token))

        def create_branch(self, **kw: Any) -> None:
            calls.append(("create_branch", kw))

        def upload_folder(self, **kw: Any) -> SimpleNamespace:
            calls.append(("upload_folder", kw))
            return SimpleNamespace(commit_url="https://huggingface.co/datasets/x/commit/abc")

    def decrypt(password: str) -> str:
        calls.append(("decrypt", password))
        if password != PASSWORD:
            raise ValueError("wrong password for the shared token")
        return TOKEN

    def no_prompt(prompt: str = "") -> str:
        raise AssertionError(f"unexpected password prompt: {prompt!r}")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(_shared_token, "decrypt_with_password", decrypt)
    monkeypatch.setattr(_shared_token, "ENCRYPTED_TOKEN_B64", "PLACEHOLDER_BLOB")
    monkeypatch.setattr(getpass, "getpass", no_prompt)
    monkeypatch.delenv(UPLOAD_PASSWORD_ENV, raising=False)
    return SimpleNamespace(calls=calls, names=lambda: [c[0] for c in calls])


def _call(hub: SimpleNamespace, name: str) -> Any:
    [payload] = [c[1] for c in hub.calls if c[0] == name]
    return payload


def test_upload_pushes_one_commit_to_a_submission_branch(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, PASSWORD)
    result = upload_hf(bundle, confirm=False)

    assert hub.names() == ["decrypt", "init", "create_branch", "upload_folder"]
    assert _call(hub, "init") == TOKEN  # the shared token, not anything of the user's
    contributor = submitter_id()
    branch_args, push = _call(hub, "create_branch"), _call(hub, "upload_folder")
    m = re.fullmatch(rf"submission-{contributor[:12]}-(\d{{8}}T\d{{6}}Z)", branch_args["branch"])
    assert m, branch_args["branch"]
    assert branch_args == {
        "repo_id": HF_REPO, "repo_type": "dataset", "branch": branch_args["branch"],
        "exist_ok": True,
    }
    prefix = f"contributors/{contributor}/bundle-{m.group(1)}"
    assert push["path_in_repo"] == prefix
    assert (push["folder_path"], push["repo_id"], push["repo_type"]) == (
        str(bundle), HF_REPO, "dataset")
    assert push["revision"] == branch_args["branch"]
    assert push["allow_patterns"] == ["manifest.json", "*.jsonl"]
    assert not push.get("create_pr")
    assert contributor in push["commit_message"]
    assert result == {
        "submitted": True, "branch": branch_args["branch"],
        "branch_url": f"https://huggingface.co/datasets/{HF_REPO}/tree/{branch_args['branch']}",
        "commit_url": "https://huggingface.co/datasets/x/commit/abc",
        "path_in_repo": prefix, "files": ["manifest.json", "a.jsonl", "b.jsonl"],
    }
    out = capsys.readouterr().out
    assert branch_args["branch"] in out and "manifest.json (" in out
    assert TOKEN not in out and PASSWORD not in out


@pytest.mark.parametrize("env", [" test-password \n", "test-password"])
def test_password_comes_from_the_environment_stripped(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, env: str,
) -> None:
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, env)
    upload_hf(_bundle(tmp_path), confirm=False)
    assert _call(hub, "decrypt") == PASSWORD


@pytest.mark.parametrize("env", [None, "   "])
def test_password_is_prompted_for_when_the_environment_has_none(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, env: str | None,
) -> None:
    if env is not None:
        monkeypatch.setenv(UPLOAD_PASSWORD_ENV, env)
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": f" {PASSWORD}\n")
    upload_hf(_bundle(tmp_path), confirm=False)
    assert _call(hub, "decrypt") == PASSWORD


def _eof(prompt: str = "") -> str:
    raise EOFError


@pytest.mark.parametrize("prompt", [lambda p="": "", lambda p="": "  ", _eof])
def test_no_password_stops_before_any_api_call(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, prompt: Any,
) -> None:
    monkeypatch.setattr(getpass, "getpass", prompt)
    with pytest.raises(UploadAuthError, match="no upload password.*dataset card"):
        upload_hf(_bundle(tmp_path), confirm=False)
    assert hub.calls == []


def test_a_wrong_password_stops_before_any_api_call(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, "not-the-password")
    with pytest.raises(UploadAuthError, match="wrong password") as exc:
        upload_hf(_bundle(tmp_path), confirm=False)
    assert isinstance(exc.value.__cause__, ValueError)
    assert hub.names() == ["decrypt"]


@pytest.mark.parametrize("answer", ["", "n", "no", "nope", "\n"])
def test_a_declined_confirmation_makes_no_calls(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], answer: str,
) -> None:
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, PASSWORD)
    monkeypatch.setattr("sys.stdin", io.StringIO(answer + "\n"))
    assert upload_hf(_bundle(tmp_path)) == {"submitted": False}
    assert hub.calls == []  # not even the decryption
    out = capsys.readouterr().out
    assert "submit this bundle? [yes/N]" in out and out.rstrip().endswith("aborted.")


@pytest.mark.parametrize("answer", ["yes", "y", " YES "])
def test_an_accepted_confirmation_submits(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, answer: str,
) -> None:
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, PASSWORD)
    monkeypatch.setattr("sys.stdin", io.StringIO(answer + "\n"))
    assert upload_hf(_bundle(tmp_path))["submitted"] is True
    assert hub.names()[-1] == "upload_folder"


@pytest.mark.parametrize("kind, error", [
    ("file", BundleAllowlistError), ("linked data", BundleAllowlistError),
    ("no manifest", FileNotFoundError),
])
def test_a_bad_bundle_stops_before_the_password_and_the_api(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    kind: str, error: type[Exception],
) -> None:
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, PASSWORD)
    bundle = _bundle(tmp_path)
    if kind == "no manifest":
        (bundle / "manifest.json").unlink()
    else:
        _plant(bundle, kind)
    with pytest.raises(error):
        upload_hf(bundle, confirm=False)
    assert hub.calls == []


@pytest.mark.parametrize("password", ["right", "wrong"])
def test_real_decryption_with_a_blob_made_for_the_test(
    tmp_path: Path, hub: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, password: str,
) -> None:
    """The real crypto, on a blob encrypted here with a test password; the
    shipped blob is never touched."""
    blob = _shared_token.encrypt_for_release("hf_made_for_this_test", "right")
    monkeypatch.setattr(_shared_token, "decrypt_with_password", REAL_DECRYPT)
    monkeypatch.setattr(_shared_token, "ENCRYPTED_TOKEN_B64", blob)
    monkeypatch.setenv(UPLOAD_PASSWORD_ENV, password)
    if password == "wrong":
        with pytest.raises(UploadAuthError, match="wrong password"):
            upload_hf(_bundle(tmp_path), confirm=False)
        assert hub.calls == []
    else:
        upload_hf(_bundle(tmp_path), confirm=False)
        assert _call(hub, "init") == "hf_made_for_this_test"
