"""Tests for the upload_hf two-token flow + password-encrypted
shared credential.

Pre-1.2.0 used the user's HF token to open a PR with
``create_pr=True``, which put the user's HF username on every
submission. 1.2.0 splits the flow:

  1. The user's HF token is used once for ``whoami()``
     proof-of-life and discarded.
  2. The user provides an upload password (env var or interactive
     prompt) which decrypts the shared submission token shipped
     with the package as an encrypted blob.
  3. The decrypted shared token authenticates the actual
     ``upload_folder`` call. Branch author = the shared
     submission account, not the user.

These tests assert (a) whoami uses the user's token, (b)
upload_folder uses the decrypted shared token, (c) failure paths
raise meaningfully, and (d) the encryption module round-trips
correctly with constant-time MAC compare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _seed_bundle(bundle_dir: Path) -> None:
    """Plant a minimal valid bundle so the allowlist preflight
    passes."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"llmoji_version": "test"}) + "\n"
    )
    (bundle_dir / "test-model.jsonl").write_text(
        json.dumps({
            "kaomoji": "(◕‿◕)", "count": 1,
            "synthesis_description": "test"
        }) + "\n"
    )


def _install_real_blob(
    monkeypatch: pytest.MonkeyPatch, plaintext_token: str, password: str,
) -> None:
    """Encrypt ``plaintext_token`` under ``password`` and patch
    :data:`llmoji._shared_token.ENCRYPTED_TOKEN_B64` to that blob,
    so the runtime decrypt path goes through the real crypto code
    instead of the placeholder."""
    from llmoji._shared_token import encrypt_for_release

    blob = encrypt_for_release(plaintext_token, password)
    monkeypatch.setattr(
        "llmoji._shared_token.ENCRYPTED_TOKEN_B64", blob,
    )


# ---------------------------------------------------------------------------
# Encryption module — round-trip + integrity
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_round_trips() -> None:
    """encrypt_for_release(plaintext, pw) → decrypt_with_password(pw)
    yields the original plaintext."""
    from llmoji import _shared_token

    blob = _shared_token.encrypt_for_release("hf_real_xyz", "test-password")
    # Patch the module's blob so decrypt_with_password reads it.
    real = _shared_token.ENCRYPTED_TOKEN_B64
    try:
        _shared_token.ENCRYPTED_TOKEN_B64 = blob
        recovered = _shared_token.decrypt_with_password("test-password")
        assert recovered == "hf_real_xyz"
    finally:
        _shared_token.ENCRYPTED_TOKEN_B64 = real


def test_decrypt_wrong_password_raises() -> None:
    """A wrong password raises ValueError with a remediation pointer."""
    from llmoji import _shared_token

    blob = _shared_token.encrypt_for_release("hf_real_xyz", "right")
    real = _shared_token.ENCRYPTED_TOKEN_B64
    try:
        _shared_token.ENCRYPTED_TOKEN_B64 = blob
        with pytest.raises(ValueError, match="wrong password"):
            _shared_token.decrypt_with_password("wrong")
    finally:
        _shared_token.ENCRYPTED_TOKEN_B64 = real


def test_decrypt_placeholder_blob_raises() -> None:
    """The placeholder blob is detected and raises a clear "this
    wheel forgot the rotation step" error."""
    from llmoji import _shared_token

    real = _shared_token.ENCRYPTED_TOKEN_B64
    try:
        _shared_token.ENCRYPTED_TOKEN_B64 = "PLACEHOLDER_BLOB"
        with pytest.raises(ValueError, match="placeholder"):
            _shared_token.decrypt_with_password("any")
    finally:
        _shared_token.ENCRYPTED_TOKEN_B64 = real


def test_decrypt_corrupt_blob_raises() -> None:
    """Truncated / corrupt base64 raises a helpful error."""
    from llmoji import _shared_token

    real = _shared_token.ENCRYPTED_TOKEN_B64
    try:
        # Valid base64 but too short to be a real blob.
        _shared_token.ENCRYPTED_TOKEN_B64 = "AAAA"
        with pytest.raises(ValueError, match="shorter than the minimum"):
            _shared_token.decrypt_with_password("any")
    finally:
        _shared_token.ENCRYPTED_TOKEN_B64 = real


def test_generate_password_returns_random_string() -> None:
    """``generate_password`` returns a non-empty string and two calls
    return distinct values."""
    from llmoji._shared_token import generate_password

    a = generate_password()
    b = generate_password()
    assert len(a) >= 12
    assert a != b


# ---------------------------------------------------------------------------
# Token wiring — proof of life uses user token, upload uses decrypted shared
# ---------------------------------------------------------------------------


def test_upload_hf_uses_user_token_for_whoami_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof-of-life path: user token goes to whoami; the password
    decrypts the shared token; upload_folder uses the shared
    token. The user's token must never reach the upload call.
    """
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)

    USER_TOKEN = "hf_user_personal_token_xyz"
    SHARED = "hf_shared_real_token_abc"
    PASSWORD = "test-password-from-twitter"

    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: USER_TOKEN,
    )
    _install_real_blob(monkeypatch, SHARED, PASSWORD)

    hfapi_init_tokens: list[str | None] = []
    whoami_called_on: list[Any] = []
    create_branch_call_kwargs: list[dict[str, Any]] = []
    upload_call_kwargs: list[dict[str, Any]] = []

    class _FakeApi:
        def __init__(self, token: str | None = None, **_: Any) -> None:
            hfapi_init_tokens.append(token)
            self._token = token

        def whoami(self) -> dict[str, str]:
            whoami_called_on.append(self._token)
            return {"name": "fake-user"}

        def create_branch(self, **kwargs: Any) -> None:
            create_branch_call_kwargs.append(
                {**kwargs, "_api_token": self._token},
            )

        def upload_folder(self, **kwargs: Any) -> Any:
            upload_call_kwargs.append({**kwargs, "_api_token": self._token})
            commit = MagicMock()
            commit.commit_url = "https://huggingface.co/datasets/foo/commit/abc"
            commit.pr_url = None
            return commit

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    result = upload_module.upload_hf(
        bundle, repo="a9lim/llmoji", confirm=False, password=PASSWORD,
    )

    # whoami uses the user's token, exactly once.
    assert whoami_called_on == [USER_TOKEN]
    # create_branch must run before upload_folder, with the shared
    # token, on the submission branch we're about to push to. The
    # Python-API ``upload_folder(revision=...)`` does not auto-create
    # the branch (unlike the ``hf upload`` CLI) so this call is
    # load-bearing — without it, preupload_lfs_files raises
    # RevisionNotFoundError.
    assert len(create_branch_call_kwargs) == 1
    create_kwargs = create_branch_call_kwargs[0]
    assert create_kwargs["_api_token"] == SHARED
    assert create_kwargs["repo_id"] == "a9lim/llmoji"
    assert create_kwargs["repo_type"] == "dataset"
    assert create_kwargs["branch"].startswith("submission-")
    assert create_kwargs["exist_ok"] is True
    # upload_folder uses the decrypted shared token.
    assert len(upload_call_kwargs) == 1
    upload_kwargs = upload_call_kwargs[0]
    assert upload_kwargs["_api_token"] == SHARED, (
        f"upload_folder must use the decrypted shared token, not "
        f"the user token. got token={upload_kwargs['_api_token']!r}"
    )
    # User token was only ever seen by the whoami HfApi instance.
    assert hfapi_init_tokens.count(USER_TOKEN) == 1, (
        f"user token leaked into a non-whoami HfApi instance: "
        f"{hfapi_init_tokens!r}"
    )

    # Branch and upload target the same submission branch.
    assert upload_kwargs["create_pr"] is False
    assert upload_kwargs["revision"] == create_kwargs["branch"]
    assert result["submitted"] is True
    assert result["branch"].startswith("submission-")


def test_upload_hf_password_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LLMOJI_UPLOAD_PASSWORD`` env var is honored when no
    explicit password kwarg is passed (scripted-use path)."""
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)

    PASSWORD = "from-env-var"
    SHARED = "hf_env_shared"

    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: "hf_user",
    )
    _install_real_blob(monkeypatch, SHARED, PASSWORD)
    monkeypatch.setenv(upload_module.UPLOAD_PASSWORD_ENV, PASSWORD)

    captured_token: list[str | None] = []

    class _FakeApi:
        def __init__(self, token: str | None = None, **_: Any) -> None:
            self._token = token

        def whoami(self) -> dict[str, str]:
            return {"name": "fake-user"}

        def create_branch(self, **_kwargs: Any) -> None:
            return None

        def upload_folder(self, **_kwargs: Any) -> Any:
            captured_token.append(self._token)
            commit = MagicMock()
            commit.commit_url = "https://example/commit"
            commit.pr_url = None
            return commit

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeApi)

    result = upload_module.upload_hf(
        bundle, confirm=False,  # password=None → falls back to env
    )
    assert result["submitted"] is True
    assert SHARED in captured_token


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_upload_hf_raises_when_user_has_no_hf_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No user HF token → bail with a friendly message before any
    network or password prompt."""
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)
    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: None,
    )

    with pytest.raises(upload_module.HFAuthError, match="hf auth login"):
        upload_module.upload_hf(bundle, confirm=False, password="any")


def test_upload_hf_raises_when_whoami_rejects_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user token the Hub rejects → bail before any push or
    password prompt."""
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)
    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: "hf_bad_token",
    )

    class _RejectingApi:
        def __init__(self, token: str | None = None, **_: Any) -> None:
            self._token = token

        def whoami(self) -> dict[str, str]:
            raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("huggingface_hub.HfApi", _RejectingApi)

    with pytest.raises(upload_module.HFAuthError, match="proof-of-life"):
        upload_module.upload_hf(bundle, confirm=False, password="any")


def test_upload_hf_raises_on_wrong_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrong upload password → bail before any push, with a
    pointer to where the password lives."""
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)

    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: "hf_user",
    )
    _install_real_blob(monkeypatch, "hf_real", "right-password")

    class _PassingWhoami:
        def __init__(self, token: str | None = None, **_: Any) -> None:
            self._token = token

        def whoami(self) -> dict[str, str]:
            return {"name": "fake-user"}

    monkeypatch.setattr("huggingface_hub.HfApi", _PassingWhoami)

    with pytest.raises(upload_module.HFAuthError, match="wrong password"):
        upload_module.upload_hf(
            bundle, confirm=False, password="wrong-password",
        )


def test_upload_hf_raises_when_password_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No password (no env, no kwarg, no interactive prompt) →
    HFAuthError pointing at the dataset card / Twitter."""
    from llmoji import upload as upload_module

    bundle = tmp_path / "bundle"
    _seed_bundle(bundle)

    monkeypatch.setattr(
        upload_module, "_read_user_hf_token", lambda: "hf_user",
    )
    monkeypatch.delenv(upload_module.UPLOAD_PASSWORD_ENV, raising=False)
    # Stub the prompt path to return None (simulate
    # non-interactive shell or user-just-pressed-enter).
    monkeypatch.setattr(
        upload_module, "_read_upload_password",
        lambda prompt=True: None,
    )

    class _PassingWhoami:
        def __init__(self, token: str | None = None, **_: Any) -> None:
            self._token = token

        def whoami(self) -> dict[str, str]:
            return {"name": "fake-user"}

    monkeypatch.setattr("huggingface_hub.HfApi", _PassingWhoami)

    with pytest.raises(upload_module.HFAuthError, match="dataset card"):
        upload_module.upload_hf(bundle, confirm=False)


# ---------------------------------------------------------------------------
# Sanity: shared-token module has the expected surface
# ---------------------------------------------------------------------------


def test_shared_token_module_exposes_expected_surface() -> None:
    """Import surface stays stable — the module exposes
    ``ENCRYPTED_TOKEN_B64``, ``decrypt_with_password``,
    ``encrypt_for_release``, and ``generate_password``."""
    from llmoji import _shared_token

    assert hasattr(_shared_token, "ENCRYPTED_TOKEN_B64")
    assert isinstance(_shared_token.ENCRYPTED_TOKEN_B64, str)
    assert callable(_shared_token.decrypt_with_password)
    assert callable(_shared_token.encrypt_for_release)
    assert callable(_shared_token.generate_password)
