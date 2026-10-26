"""Encrypted shared HuggingFace credential for submission pushes to
``a9lim/llmoji``.

:mod:`llmoji.upload` pushes each submission branch with this credential, so
the contributor's HF account stays off the dataset's history. ``SECURITY.md``
has the threat model, ``AGENTS.md`` the rotation steps.

The token is encrypted under a password that is posted publicly (dataset
card, Twitter). That keeps it out of casual reach: grepping the wheel for
``hf_`` finds nothing, secret scanners see no plaintext token, and using it
takes a deliberate password lookup. It does not stop anyone who reads the
dataset card. That is accepted: the maintainer reviews every submission
branch by diff before merging, and that review is the actual abuse
mitigation.

Layout of the base64-decoded blob:

    [16 bytes salt][32 bytes HMAC-SHA256 mac][N bytes ciphertext]

with ``key = pbkdf2_hmac('sha256', password, salt, ITERATIONS)``, ciphertext
= plaintext XOR keystream (``HMAC(key, counter_be64)`` blocks for counter 0,
1, 2, ...) and ``mac = HMAC(key, ciphertext)``: encrypt-then-MAC. The cipher
is homemade but stdlib-only; weaker than AES-GCM, adequate here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

# PBKDF2-SHA256 rounds: about OWASP's 2023 figure (210k) and ~50 ms on a
# laptop. A random ~96-bit password (see :func:`generate_password`) is out
# of brute-force reach at that cost.
ITERATIONS = 200_000

# Output of ``encrypt_for_release(real_token, password)``, replaced at every
# release. A wheel that ships the placeholder is caught by
# :data:`PLACEHOLDER_BLOB_PREFIX` and fails loudly at decrypt time.
ENCRYPTED_TOKEN_B64: str = (
    "YttXqY2EWF9EHdyHWFOqrWtA6tekNKDzScfJYhnsLqxNsjqsVwwZmBsJV4K1m3m7"
    "31vfaTkRDiWePIMLyHR3pUMsvpa9hZye8zM9aQIJRLlcHmJaiw=="
)

# A blob starting with this marker is a release that skipped rotation. A
# real blob is random bytes, so it can't plausibly start with it.
PLACEHOLDER_BLOB_PREFIX = "PLACEHOLDER"


def _derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-SHA256 key (32 bytes), used for both the keystream and the MAC."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS,
    )


def _hmac_keystream(key: bytes, length: int) -> bytes:
    """``length`` bytes of HMAC-SHA256 counter keystream."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hmac.new(key, counter.to_bytes(8, "big"), "sha256").digest()
        )
        counter += 1
    return bytes(out[:length])


def encrypt_for_release(plaintext_token: str, password: str) -> str:
    """Encrypt ``plaintext_token``; returns the base64 blob for
    :data:`ENCRYPTED_TOKEN_B64`. Run at release time, never at runtime.
    """
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    plaintext = plaintext_token.encode("utf-8")
    keystream = _hmac_keystream(key, len(plaintext))
    ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream))
    mac = hmac.new(key, ciphertext, "sha256").digest()
    blob = salt + mac + ciphertext
    return base64.b64encode(blob).decode("ascii")


def decrypt_with_password(password: str) -> str:
    """Decrypt :data:`ENCRYPTED_TOKEN_B64` with ``password``.

    Raises :class:`ValueError` for the placeholder blob, a corrupt or
    truncated blob, or a wrong password (MAC mismatch).
    """
    if ENCRYPTED_TOKEN_B64.startswith(PLACEHOLDER_BLOB_PREFIX):
        raise ValueError(
            "the shared submission credential in "
            "`llmoji/_shared_token.py` is the placeholder, not a "
            "real encrypted blob. This should never happen on a "
            "released wheel; please open an issue at "
            "https://github.com/a9lim/llmoji/issues."
        )
    try:
        blob = base64.b64decode(ENCRYPTED_TOKEN_B64, validate=True)
    except (ValueError, base64.binascii.Error) as e:  # type: ignore[attr-defined]
        raise ValueError(
            f"shared credential blob is not valid base64: {e}"
        ) from e
    if len(blob) < 16 + 32 + 1:
        raise ValueError(
            "shared credential blob is shorter than the minimum "
            "(16 byte salt + 32 byte mac + 1 byte ciphertext)"
        )
    salt = blob[:16]
    expected_mac = blob[16:48]
    ciphertext = blob[48:]
    key = _derive_key(password, salt)
    actual_mac = hmac.new(key, ciphertext, "sha256").digest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        # Constant-time, so timing doesn't reveal how much of the MAC matched.
        raise ValueError(
            "wrong password (or corrupt encrypted token). The "
            "current upload password is posted on the dataset "
            "card at https://huggingface.co/datasets/a9lim/llmoji "
            "and on Twitter at https://twitter.com/_a9lim ."
        )
    keystream = _hmac_keystream(key, len(ciphertext))
    plaintext = bytes(c ^ k for c, k in zip(ciphertext, keystream))
    return plaintext.decode("utf-8")


def generate_password(num_bytes: int = 12) -> str:
    """A random 16-character URL-safe password (about 96 bits) for rotation:
    encrypt the token with it via :func:`encrypt_for_release`, paste the blob
    into ``ENCRYPTED_TOKEN_B64`` and post the password publicly.
    """
    # Drop any base64 padding so the password copies cleanly out of a tweet.
    return secrets.token_urlsafe(num_bytes).rstrip("=")
