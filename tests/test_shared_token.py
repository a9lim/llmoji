"""The encrypted submission credential: round trip, integrity, failure modes.

Every blob here is made for the test, under a test password. The shipped
blob is only inspected, never decrypted.
"""

from __future__ import annotations

import base64

import pytest

from llmoji import _shared_token
from llmoji._shared_token import decrypt_with_password, encrypt_for_release, generate_password


@pytest.fixture
def blob(monkeypatch: pytest.MonkeyPatch) -> str:
    """``hf_test`` encrypted under the password ``right``, installed as the shipped blob."""
    made = encrypt_for_release("hf_test", "right")
    monkeypatch.setattr(_shared_token, "ENCRYPTED_TOKEN_B64", made)
    return made


def test_round_trip(blob: str) -> None:
    assert decrypt_with_password("right") == "hf_test"


def test_a_wrong_password_is_refused(blob: str) -> None:
    with pytest.raises(ValueError, match="wrong password"):
        decrypt_with_password("wrong")


def test_a_tampered_blob_fails_the_mac(blob: str, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = bytearray(base64.b64decode(blob))
    raw[-1] ^= 1
    monkeypatch.setattr(_shared_token, "ENCRYPTED_TOKEN_B64", base64.b64encode(raw).decode())
    with pytest.raises(ValueError, match="wrong password"):
        decrypt_with_password("right")


def test_each_encryption_is_salted() -> None:
    assert encrypt_for_release("hf_test", "right") != encrypt_for_release("hf_test", "right")


@pytest.mark.parametrize("bad, message", [
    ("PLACEHOLDER_BLOB", "placeholder"),
    ("AAAA", "shorter than the minimum"),
    ("not base64!", "not valid base64"),
])
def test_unusable_blobs_raise(monkeypatch: pytest.MonkeyPatch, bad: str, message: str) -> None:
    monkeypatch.setattr(_shared_token, "ENCRYPTED_TOKEN_B64", bad)
    with pytest.raises(ValueError, match=message):
        decrypt_with_password("any")


def test_the_shipped_blob_is_a_real_one() -> None:
    """A release that forgot the rotation step ships the placeholder."""
    shipped = _shared_token.ENCRYPTED_TOKEN_B64
    assert not shipped.startswith(_shared_token.PLACEHOLDER_BLOB_PREFIX)
    assert len(base64.b64decode(shipped, validate=True)) >= 16 + 32 + 1


def test_generated_passwords_are_random_and_paste_cleanly() -> None:
    a, b = generate_password(), generate_password()
    assert a != b and len(a) >= 12 and "=" not in a
