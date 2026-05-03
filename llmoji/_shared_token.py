"""Encrypted shared HuggingFace credential for submission branch
pushes on ``a9lim/llmoji``.

The shared credential is what authenticates the actual ``upload_folder``
call. The user's own HF token is touched only once for a
``whoami()`` proof-of-life check (see :mod:`llmoji.upload`); the
push itself goes through this credential so the user's HF account
stays off the dataset's commit history. See ``SECURITY.md`` for
the full threat model and ``CLAUDE.md`` for the operational
checklist.

This is a **paper-thin barrier**, not real security. The threat
model is "casual attacker who scrapes the wheel and runs
``grep hf_``" — that attacker walks away empty. A determined
attacker can find the password (a9 posts it on the dataset card
and on Twitter) and decrypt the token; that is by design. The
goal is to require explicit, intentional retrieval of the
password rather than incidental access.

What this defends against:

  - ``pip download llmoji && grep hf_ wheel-contents`` returns
    nothing usable (token is opaque base64).
  - Automated supply-chain scanners that flag plaintext API
    tokens in package source.
  - Incidental misuse — a user who finds the package source has
    to deliberately go look up the password to reach the token.

What this does NOT defend against:

  - Anyone who reads the dataset card or follows ``@a9lim``.
    They can find the password trivially. We accept this; the
    PR review path on the dataset (the maintainer reviews each
    submission branch by diff before merging) is the actual
    abuse mitigation.

Encryption layout, base64-decoded:

    [16 bytes salt][32 bytes HMAC-SHA256 mac][N bytes ciphertext]

Where:

  - ``key = pbkdf2_hmac('sha256', password, salt, ITERATIONS)``
  - ``ciphertext = plaintext XOR keystream`` and
    ``keystream = concat(HMAC(key, counter_be64) for counter in 0,1,2,...)``
  - ``mac = HMAC(key, ciphertext)`` — encrypt-then-MAC

This is a homemade HMAC-based stream cipher with HMAC integrity.
Stdlib only (no ``cryptography`` dep). Not as strong as AES-GCM
or ChaCha20-Poly1305, but adequate for the paper-thin threat
model and crypto-doctrine-clean (PBKDF2 → HMAC-keystream →
HMAC-MAC, encrypt-then-authenticate).

Rotation is one command. See :func:`encrypt_for_release`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

# PBKDF2-SHA256 iteration count. 200k is around the high end of
# OWASP's 2023 recommendation (210k) and takes ~50ms on a modern
# laptop. Brute-forcing a random 16-character base32 password
# (~80 bits) at 50ms/attempt is infeasible; the random-string
# password format (see CLAUDE.md ops checklist) is well past
# anything reasonable.
ITERATIONS = 200_000

# Encrypted blob; replace at every release with the output of
# ``encrypt_for_release(real_token, password)``. A wheel that ships
# with the placeholder string is detected by
# :data:`PLACEHOLDER_BLOB_PREFIX` and bails loudly at decrypt time.
ENCRYPTED_TOKEN_B64: str = (
    "YttXqY2EWF9EHdyHWFOqrWtA6tekNKDzScfJYhnsLqxNsjqsVwwZmBsJV4K1m3m7"
    "31vfaTkRDiWePIMLyHR3pUMsvpa9hZye8zM9aQIJRLlcHmJaiw=="
)

# Marker prefix so the runtime can tell "this is a release that
# forgot to rotate the token" from "this is a real encrypted
# blob." The prefix is intentionally not valid base64 padding so
# real blobs (always valid base64) can't accidentally start with
# this string.
PLACEHOLDER_BLOB_PREFIX = "PLACEHOLDER"


def _derive_key(password: str, salt: bytes) -> bytes:
    """PBKDF2-SHA256 with :data:`ITERATIONS` rounds. Returns a
    32-byte key suitable for use as both the keystream PRF key
    and the MAC key (the construction is encrypt-then-MAC with
    domain-separated calls, so reusing the key across both is
    safe).
    """
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS,
    )


def _hmac_keystream(key: bytes, length: int) -> bytes:
    """Generate ``length`` bytes of keystream by concatenating
    ``HMAC-SHA256(key, counter_be64)`` for counter = 0, 1, 2, ...

    Truncates at ``length`` bytes; counters above what's needed
    are not computed.
    """
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hmac.new(key, counter.to_bytes(8, "big"), "sha256").digest()
        )
        counter += 1
    return bytes(out[:length])


def encrypt_for_release(plaintext_token: str, password: str) -> str:
    """Encrypt ``plaintext_token`` under ``password``. Returns the
    base64 string to paste into :data:`ENCRYPTED_TOKEN_B64` at
    release time.

    Run once per release / token rotation; not called on the
    runtime path.
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
    """Decrypt :data:`ENCRYPTED_TOKEN_B64` using ``password``.

    Raises :class:`ValueError` on:

      - placeholder blob (the wheel didn't get the release
        rotation step)
      - corrupt / truncated blob
      - wrong password (HMAC mismatch — constant-time compare)
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
        # Constant-time compare so a timing-attack adversary can't
        # learn how many leading bytes of the MAC matched.
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
    """Generate a random password to use for token encryption at
    release / rotation time. Returns a URL-safe base64 string;
    12 bytes = 16 base64 characters = ~96 bits of entropy.

    Run once at rotation time:

        from llmoji._shared_token import (
            encrypt_for_release, generate_password,
        )
        password = generate_password()
        blob = encrypt_for_release("hf_real_token", password)
        # paste blob into ENCRYPTED_TOKEN_B64
        # post password on dataset card + Twitter
    """
    # ``token_urlsafe`` returns base64url; strip padding so the
    # password copy-pastes cleanly out of a tweet.
    return secrets.token_urlsafe(num_bytes).rstrip("=")
