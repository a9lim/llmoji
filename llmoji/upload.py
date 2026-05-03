"""Upload a bundle to one of two targets: HF dataset or email.

Both targets are opt-in and require explicit ``--target`` selection;
no implicit default.

HF target:
    The submission flow has three secrets in play:

      1. The user's HF token, used only for a ``whoami()``
         proof-of-life check; discarded immediately, never used to
         push the bundle.
      2. An upload password, posted by the maintainer on the
         dataset card and on Twitter; the user types it (or sets
         ``$LLMOJI_UPLOAD_PASSWORD``) when running ``upload``.
      3. The shared submission HF token, encrypted under the
         upload password and shipped with the package as an
         opaque base64 blob in :mod:`llmoji._shared_token`.

    The user's password decrypts the shared token, the shared
    token authenticates the actual ``upload_folder`` call. The
    user's HF account never appears on the dataset's commit
    history or PR list.

    Pre-1.2.0 used the user's HF token directly with
    ``create_pr=True``, which put the user's HF username on every
    submission PR and contradicted the privacy claim that
    submissions can't be traced to a specific user; 1.2.0 patches
    that. The password layer is a paper-thin barrier (anyone
    determined can find the password) — see ``SECURITY.md``.

    On the dataset side, Discussions and Pull Requests are
    DISABLED on ``a9lim/llmoji`` (HF setting). This is the
    enforcement mechanism that breaks pre-1.2.0 clients: an old
    client trying ``create_pr=True`` with the user's token gets a
    clear API error from HF, the user upgrades, the new client
    succeeds via the shared credential. New clients push to a
    per-submission branch ``submission-<contributor>-<ts>`` via
    ``upload_folder(revision=branch_name, create_pr=False)``; the
    maintainer reviews the branch by diff and merges to ``main``
    by hand.

    The bundle layout on disk (and on the merged ``main``) is
    ``contributors/<hash>/bundle-<ts>/{manifest.json,<slug>.jsonl,
    ...}`` — one ``<source-model-slug>.jsonl`` per model the
    journal saw. Loose files rather than a tarball so the HF
    dataset viewer can auto-load every per-source-model
    ``*.jsonl`` via a ``data_files: contributors/**/*.jsonl``
    configs entry on the dataset card. ``upload_folder`` does a
    single atomic commit on the submission branch so partial
    uploads can't land.

    The submitter identifier is a 32-hex-char salted hash of (a
    per-machine random token + the package version), persisted at
    ``~/.llmoji/.salt`` (flat 64-hex-char file; pre-1.1.x had it
    wrapped in a JSON envelope at ``state.json``). We do NOT
    collect HF usernames or any account-bound identifier; the
    salted hash is just enough to dedup repeat submissions from
    the same machine.

Email target:
    Build a ``mailto:`` URI with the bundle path printed in the
    body and instructions to attach manually. We don't ship SMTP.
    The CLI prints the path and opens the user's mail client via
    ``open`` (macOS) or ``xdg-open`` (Linux); the user attaches the
    tarball themselves. Email keeps the tarball-as-archive shape
    because a single attachment is what an email recipient wants.
"""

from __future__ import annotations

import os
import secrets
import sys
import tarfile
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from . import paths
from ._util import atomic_write_text, package_version

DEFAULT_HF_REPO = "a9lim/llmoji"
DEFAULT_EMAIL_TO = "mx@a9l.im"

# Strict allowlist of files the bundle is permitted to ship. The
# bundle is flat: ``manifest.json`` at the top level plus one
# ``<sanitized_source_model>.jsonl`` per source model the journal
# saw. Anything else — extra top-level files of other types,
# subdirectories of any kind, symlinks — is treated as user-added
# or stale and refused (loud failure beats silent leak).
BUNDLE_TOPLEVEL_ALLOWLIST: tuple[str, ...] = ("manifest.json",)
BUNDLE_DATA_SUFFIX: str = ".jsonl"


class BundleAllowlistError(RuntimeError):
    """Raised by ``tar_bundle`` / ``upload_hf`` when the bundle
    directory holds anything outside the allowed shape (top-level
    ``manifest.json`` plus per-source-model ``<slug>.jsonl``
    files, nothing else). Loud failure is the correct response —
    silently dropping the extras on the way out would leak
    whatever the user stashed there."""


def _classify_bundle(bundle_dir: Path) -> tuple[list[Path], list[Path]]:
    """Single-pass walk. Returns ``(allowlisted_files, extras)``,
    each sorted. Missing dir → both lists empty.

    Allowlisted files = a real ``manifest.json`` at the top level
    plus each top-level ``*.jsonl`` (the per-source-model data
    file). No recursion: subdirectories are extras.

    Symlinks are rejected — ``Path.is_file()`` follows them, so a
    symlinked ``manifest.json`` or ``<model>.jsonl`` would
    otherwise pass the allowlist check and shuffle whatever the
    link points at into the upload. Loud failure beats following
    a footgun.

    Output ordering: ``manifest.json`` first, then per-source-model
    ``.jsonl`` files sorted by filename.
    """
    if not bundle_dir.exists():
        return [], []
    toplevel_allowed = set(BUNDLE_TOPLEVEL_ALLOWLIST)
    manifest: list[Path] = []
    data_files: list[Path] = []
    extras: list[Path] = []
    for p in sorted(bundle_dir.iterdir()):
        if p.is_symlink():
            extras.append(p)
            continue
        if p.is_file():
            if p.name in toplevel_allowed:
                manifest.append(p)
            elif p.suffix == BUNDLE_DATA_SUFFIX:
                data_files.append(p)
            else:
                extras.append(p)
            continue
        # Directories, sockets, FIFOs, anything else → extras.
        extras.append(p)
    # Stable ordering: manifest(s) first in allowlist order, then
    # data files in filename order.
    by_name = {p.name: p for p in manifest}
    ordered: list[Path] = [
        by_name[n] for n in BUNDLE_TOPLEVEL_ALLOWLIST if n in by_name
    ]
    ordered.extend(data_files)
    return ordered, extras


def _check_or_raise(bundle_dir: Path, op: str) -> list[Path]:
    """Common bundle-allowlist preflight. Returns the allowlisted
    files; raises :class:`BundleAllowlistError` for extras and
    :class:`FileNotFoundError` for an empty bundle."""
    allowlisted, extras = _classify_bundle(bundle_dir)
    if extras:
        joined = ", ".join(str(p.relative_to(bundle_dir)) for p in extras)
        raise BundleAllowlistError(
            f"refusing to {op} {bundle_dir} — unexpected entr(y/ies) "
            f"{joined!r} not in the bundle allowlist (top-level "
            f"{BUNDLE_TOPLEVEL_ALLOWLIST!r} plus per-source-model "
            f"{BUNDLE_DATA_SUFFIX!r} files; no subdirs or symlinks). "
            f"Remove them or re-run `llmoji analyze` (which clears "
            f"the bundle dir)."
        )
    if not allowlisted:
        raise FileNotFoundError(
            f"no allowlisted files in {bundle_dir} — run "
            f"`llmoji analyze` first"
        )
    return allowlisted


def tar_bundle(
    bundle_dir: Path, *, out_path: Path | None = None,
) -> tuple[Path, list[Path]]:
    """Tar the bundle directory. Returns ``(tarball_path, files)``.

    Strict allowlist: only ``manifest.json`` at the top level plus
    each ``<source-model>.jsonl`` is included. If the bundle holds
    anything else (subdirs, non-jsonl files, symlinks), raise
    :class:`BundleAllowlistError` — refusing to ship is the safe
    default (the user can `rm` the extras and re-tar).

    Files are added with a flat ``arcname`` (no ``bundle/`` prefix);
    the bundle dir lives at ``~/.llmoji/bundle/`` on disk and the
    tarball recipient picks their own destination directory at
    extract time.

    Returning the file list saves :func:`upload_email` from
    re-walking the bundle dir to enumerate contents for the email
    body — same set of files, single allowlist check.
    """
    if out_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = bundle_dir.parent / f"bundle-{ts}.tar.gz"
    files = _check_or_raise(bundle_dir, "tar")
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.name)
    return out_path, files


def _submission_token() -> str:
    """Per-machine random token, persisted at ``~/.llmoji/.salt``.

    Generated once on first ``upload``. Never sent anywhere — only
    used as the salt in :func:`submitter_id`. The flat 64-hex-char
    file replaces the pre-1.1.x JSON envelope at ``state.json``;
    same byte, less wrapping.
    """
    salt_path = paths.salt_path()
    if salt_path.exists():
        existing = salt_path.read_text().strip()
        if existing:
            return existing
    token = secrets.token_hex(32)
    # Atomic write — losing the salt mid-write would invalidate
    # the user's submitter id and double-credit them in the
    # dataset's per-machine dedup.
    atomic_write_text(salt_path, token + "\n")
    return token


def submitter_id() -> str:
    """Salted-hash submitter identifier. 32 hex chars (128 bits),
    stable per (machine, llmoji version).

    Length is 128 bits because the dataset README pins submission
    identity to this string and an attacker who knows the
    per-machine token (server-side compromise) shouldn't be able to
    grind a same-id submission. 64 bits would be fine for pure dedup
    but isn't crypto-collision-resistant in the formal sense.

    Public so the manifest write in :func:`llmoji.analyze.run_analyze`
    can stamp the same id the HF upload path would use, keeping the
    bundle the user inspects byte-identical to what ships.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(_submission_token().encode("ascii"))
    h.update(b"\0")
    h.update(package_version().encode("ascii"))
    return h.hexdigest()[:32]


def _confirm(message: str) -> bool:
    """Re-prompt before committing. Returns True iff user types
    ``yes``. ``--yes`` callers should bypass this in the CLI layer.
    """
    sys.stdout.write(f"{message} [yes/N] ")
    sys.stdout.flush()
    answer = sys.stdin.readline().strip().lower()
    return answer in ("yes", "y")


# ---------------------------------------------------------------------------
# HF target
# ---------------------------------------------------------------------------


class HFAuthError(RuntimeError):
    """Raised when the proof-of-life check on the user's HF token
    fails, or when the upload password is missing or wrong, or
    when the shared-token blob is the unrelased placeholder. The
    CLI surfaces the message verbatim so the user can fix their
    auth and retry.
    """


# Env var the user can set to skip the interactive password prompt
# in scripted use (CI, batch scripts, recurring uploads).
UPLOAD_PASSWORD_ENV = "LLMOJI_UPLOAD_PASSWORD"


def _read_upload_password(prompt: bool = True) -> str | None:
    """Resolve the upload password from (in order) the env var
    :data:`UPLOAD_PASSWORD_ENV`, then an interactive ``getpass``
    prompt if ``prompt`` is True. Returns ``None`` when no
    password is found and prompting is disabled (used by
    test/dry-run paths).
    """
    env = os.environ.get(UPLOAD_PASSWORD_ENV)
    if env:
        return env.strip()
    if not prompt:
        return None
    import getpass
    print(
        "to submit, please enter the upload password."
    )
    try:
        pw = getpass.getpass("upload password: ")
    except EOFError:
        return None
    return pw.strip() or None


def _read_user_hf_token() -> str | None:
    """Read the user's HF token from the canonical locations
    (``$HF_TOKEN`` env var, then ``~/.cache/huggingface/token``).
    Returns ``None`` if nothing is configured. Used ONLY for the
    proof-of-life ``whoami()`` call; never persisted, never sent
    to the upload itself.
    """
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover — huggingface_hub is a hard dep
        return None
    token = get_token()
    if token is None:
        return None
    return token


def _proof_of_life(user_token: str) -> str:
    """Call ``HfApi.whoami()`` with the user's token. Returns the
    HF username on success (used only for the friendly print
    message; not stored, not bundled). Raises :class:`HFAuthError`
    on rejection. The token itself is discarded by the caller
    immediately after this returns.
    """
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=user_token).whoami()
    except Exception as e:  # noqa: BLE001 — surfaced verbatim
        raise HFAuthError(
            f"HF proof-of-life check failed: {e}. Please verify "
            f"your HF token (run `hf auth whoami` to test) and try "
            f"again."
        ) from e
    name = info.get("name") if isinstance(info, dict) else None
    return str(name) if name else "(unknown)"


def upload_hf(
    bundle_dir: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    confirm: bool = True,
    password: str | None = None,
) -> dict[str, Any]:
    """Push the bundle to a per-submission branch on the chosen HF
    dataset repo, going through the shared submission credential so
    the user's HF account stays off the dataset.

    Two-step flow:

    1. **Proof of life.** Read the user's HF token from
       ``$HF_TOKEN`` / ``~/.cache/huggingface/token`` and call
       ``HfApi(token=user_token).whoami()``. If the token is
       missing or rejected, raise :class:`HFAuthError` with a
       friendly remediation message. The user's token is
       discarded immediately after this call; it is never used to
       authenticate the upload itself, never logged, never written
       to disk.
    2. **Submission.** Push the bundle to a per-submission branch
       ``submission-<contributor>-<ts>`` via
       ``HfApi(token=SHARED_HF_TOKEN).upload_folder(
       revision=branch_name, create_pr=False)``. The shared
       credential lives in :mod:`llmoji._shared_token` and is a
       fine-grained HF token scoped to write on the target
       dataset only. The maintainer reviews the branch by diff
       and merges to ``main`` by hand.

    Discussions and Pull Requests are DISABLED on
    ``a9lim/llmoji`` (HF dataset setting), so pre-1.2.0 clients
    that called ``create_pr=True`` fail with a clear API error
    instead of silently leaking the user's HF username on a PR.

    Pre-flight: refuse to upload if anything outside the flat
    allowlist (top-level ``manifest.json`` plus per-source-model
    ``*.jsonl`` files) is in the bundle dir, mirroring
    :func:`tar_bundle` (loud failure beats silent leak).
    ``upload_folder``'s ``allow_patterns`` is a second line of
    defense against the same class of bug.

    The dataset card's ``data_files: contributors/**/*.jsonl`` glob
    surfaces every per-source-model data file across contributors
    into the dataset viewer's single train split (after the
    maintainer merges the submission branch into ``main``).
    """
    from ._shared_token import decrypt_with_password

    files = _check_or_raise(bundle_dir, "upload")
    contributor = submitter_id()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target_prefix = f"contributors/{contributor}/bundle-{ts}"
    branch_name = f"submission-{contributor[:12]}-{ts}"

    # Proof-of-life: confirm the user has a real HF account before
    # we push anything. Their token is discarded right after.
    user_token = _read_user_hf_token()
    if user_token is None:
        raise HFAuthError(
            "No HF token found. llmoji uses your HF account as a "
            "proof-of-life check. Please "
            "run `hf auth login` and try again."
        )
    user_name = _proof_of_life(user_token)
    # Discard the user token before doing anything else. The actual
    # upload uses the shared submission credential below.
    del user_token

    print(f"target: HF dataset {repo} → branch {branch_name}/")
    print(
        f"proof of life: authenticated as HF user {user_name!r} "
        f"(token discarded; not used for the upload)"
    )
    for f in files:
        print(
            f"  {f.relative_to(bundle_dir).as_posix()} "
            f"({f.stat().st_size} bytes)"
        )
    if confirm and not _confirm("submit this bundle?"):
        print("aborted.")
        return {"submitted": False}

    # Resolve the upload password (explicit kwarg → env var →
    # interactive prompt) and decrypt the shared submission token.
    if password is None:
        password = _read_upload_password(prompt=True)
    if not password:
        raise HFAuthError(
            "no upload password provided. The current password "
            "is posted on the dataset card at "
            "https://huggingface.co/datasets/a9lim/llmoji or set "
            f"${UPLOAD_PASSWORD_ENV} for scripted use."
        )
    try:
        shared_token = decrypt_with_password(password)
    except ValueError as e:
        raise HFAuthError(str(e)) from e
    finally:
        # Discard the user-supplied password from the local frame.
        # Python won't zero the underlying string memory but the
        # name is gone from the upload scope after this block.
        del password

    from huggingface_hub import HfApi
    api = HfApi(token=shared_token)
    # Discard the decrypted token from this scope after handing it
    # to the API client. The HfApi instance keeps its own copy
    # (necessary for retries inside upload_folder) but the local
    # binding doesn't linger past the next line.
    del shared_token
    # Create the submission branch first. ``upload_folder(revision=...)``
    # requires the branch to already exist when called via the Python
    # API; the ``hf upload`` CLI wraps an auto-create step but the
    # huggingface_hub library function does not, and a missing branch
    # surfaces as ``RevisionNotFoundError`` from the preupload step.
    # ``exist_ok=True`` makes re-runs against a partially-created
    # branch idempotent (e.g. a previous upload failed mid-commit).
    api.create_branch(
        repo_id=repo,
        repo_type="dataset",
        branch=branch_name,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        folder_path=str(bundle_dir),
        path_in_repo=target_prefix,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"llmoji bundle from {contributor}",
        commit_description=(
            f"Submitted by contributor `{contributor}` via "
            f"`llmoji upload --target hf` (llmoji "
            f"{package_version()}). Bundle path in repo: "
            f"`{target_prefix}/`. Maintainer: review this branch "
            f"and merge to `main` if approved."
        ),
        # Per-submission branch; PRs are disabled on this dataset
        # (HF setting) so old clients that hit `create_pr=True`
        # error out cleanly instead of leaking the user's HF
        # username on a PR.
        revision=branch_name,
        create_pr=False,
        # Top-level manifest + every per-source-model
        # ``<slug>.jsonl`` at the bundle root. The structural
        # allowlist check above is the primary defense; this is
        # the second line.
        allow_patterns=[
            *BUNDLE_TOPLEVEL_ALLOWLIST,
            f"*{BUNDLE_DATA_SUFFIX}",
        ],
    )
    # ``upload_folder`` returns a ``CommitInfo`` whose
    # ``commit_url`` attribute points at the submission branch's
    # commit. Older huggingface_hub releases returned a bare URL
    # string from ``upload_folder`` — defensively fall through to
    # ``str()`` if the structured attrs aren't available.
    commit_url = getattr(commit_info, "commit_url", None)
    if commit_url is None and not isinstance(commit_info, str):
        commit_url = getattr(commit_info, "pr_url", None)
    if commit_url is None:
        commit_url = str(commit_info)
    branch_url = (
        f"https://huggingface.co/datasets/{repo}/tree/{branch_name}"
    )
    print(
        f"submitted to {repo} as branch {branch_name}; the "
        f"maintainer will review and merge."
    )
    print(f"  branch: {branch_url}")
    print(f"  commit: {commit_url}")
    return {
        "submitted": True,
        "repo": repo,
        "path_in_repo": target_prefix,
        "branch": branch_name,
        "branch_url": branch_url,
        "commit_url": commit_url,
        "contributor": contributor,
        "files": [f.relative_to(bundle_dir).as_posix() for f in files],
    }


# ---------------------------------------------------------------------------
# Email target
# ---------------------------------------------------------------------------


def upload_email(
    bundle_dir: Path,
    *,
    to: str = DEFAULT_EMAIL_TO,
    confirm: bool = True,
) -> dict[str, Any]:
    """Build a mailto URI and open it in the system mail client; the
    user attaches the tarball manually. We don't ship SMTP.

    Uses :func:`webbrowser.open` to launch the mailto: URL — the
    Python stdlib's cross-platform handler picks the right opener
    on macOS / Linux / Windows without per-platform branching. On
    a CI box without a registered handler it returns ``False`` and
    we fall through to printing the URI (the user can copy it into
    a mail client by hand).
    """
    tarball, files = tar_bundle(bundle_dir)
    print(f"target: email {to}")
    print(f"local tarball: {tarball} ({tarball.stat().st_size} bytes)")
    if confirm and not _confirm("open your mail client and attach this bundle?"):
        print("aborted; tarball left on disk.")
        # Aborted by the user — surface the tarball path so a
        # scripted caller can still find it, and report submitted=False
        # accurately (pre Wave 6 the abort path returned
        # submitted=True, which lied about user intent).
        return {"submitted": False, "tarball": str(tarball), "to": to}

    body = (
        "Hi,\n\n"
        f"Please find an llmoji bundle attached at:\n  {tarball}\n\n"
        "Bundle contents:\n"
    )
    for f in files:
        rel = f.relative_to(bundle_dir).as_posix()
        body += f"  - {rel} ({f.stat().st_size} bytes)\n"
    body += "\nPaste this email into your mail client and attach the tarball manually.\n"

    mailto = "mailto:" + to + "?" + urllib.parse.urlencode({
        "subject": "llmoji bundle submission",
        "body": body,
    })

    # Best-effort handoff to the system mail client. webbrowser.open
    # returns False when no handler is registered — we still print
    # the URL so the user can copy it into a mail client by hand.
    try:
        webbrowser.open(mailto)
    except webbrowser.Error:
        pass

    print(f"\nattach: {tarball}")
    print(f"mailto: {mailto[:200]}{'...' if len(mailto) > 200 else ''}")
    return {
        "submitted": True,  # we count "user instructed" as submitted
        "tarball": str(tarball),
        "to": to,
    }
