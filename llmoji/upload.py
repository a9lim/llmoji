"""Shipping a bundle: to the HF dataset, or as a tarball to send by hand.

HF: the bundle goes to a per-submission branch of ``a9lim/llmoji``
through a shared submission token, so the contributor's HF account
never appears on the dataset. The token ships encrypted in
:mod:`llmoji._shared_token`; the upload password that decrypts it is
posted on the dataset card. That makes the token a public write
credential for the dataset, scoped to it alone: a barrier against
accidents, not against a determined vandal (see SECURITY.md). The
maintainer reviews each branch by diff and merges by hand. Discussions
and pull requests are disabled on the dataset, so no client can open a
PR under its own account.

Archive: a tarball of the same files, for anyone who'd rather email it.

Either way only the flat allowlist leaves the machine: ``manifest.json``
plus top-level ``*.jsonl``, no subdirectories, no symlinks.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from . import paths
from ._util import atomic_write_text, package_version

HF_REPO = "a9lim/llmoji"
ARCHIVE_EMAIL = "mx@a9l.im"
UPLOAD_PASSWORD_ENV = "LLMOJI_UPLOAD_PASSWORD"

BUNDLE_TOPLEVEL_ALLOWLIST: tuple[str, ...] = ("manifest.json",)
BUNDLE_DATA_SUFFIX = ".jsonl"


class BundleAllowlistError(RuntimeError):
    """The bundle directory holds something outside the allowlist.
    Refusing beats silently dropping it, or shipping it."""


class UploadAuthError(RuntimeError):
    """No upload password, or one that doesn't decrypt the token."""


def bundle_files(bundle_dir: Path) -> list[Path]:
    """The files that would ship, ``manifest.json`` first.

    Raises :class:`BundleAllowlistError` for anything else in the
    directory and :class:`FileNotFoundError` when there's no manifest.
    ``is_file`` follows symlinks, so links are refused before it's asked.
    """
    if not bundle_dir.exists():
        raise FileNotFoundError(f"no bundle at {bundle_dir}; run `llmoji analyze` first")
    extras = [
        p for p in bundle_dir.iterdir()
        if p.is_symlink() or not p.is_file()
        or (p.name not in BUNDLE_TOPLEVEL_ALLOWLIST and p.suffix != BUNDLE_DATA_SUFFIX)
    ]
    if extras:
        raise BundleAllowlistError(
            f"refusing to ship {bundle_dir}: "
            f"{', '.join(sorted(p.name for p in extras))} "
            f"{'is' if len(extras) == 1 else 'are'} outside the allowlist "
            f"(manifest.json plus top-level *.jsonl). Remove them, or re-run "
            f"`llmoji analyze`, which clears the bundle directory."
        )
    manifest = bundle_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest in {bundle_dir}; run `llmoji analyze` first")
    return [manifest, *sorted(p for p in bundle_dir.iterdir() if p.suffix == BUNDLE_DATA_SUFFIX)]


def submitter_id() -> str:
    """32 hex chars: a hash of a per-machine random salt and the llmoji
    version. Enough to dedup repeat submissions from one machine; the
    salt itself never leaves it."""
    salt_path = paths.salt_path()
    salt = salt_path.read_text().strip() if salt_path.exists() else ""
    if not salt:
        salt = secrets.token_hex(32)
        atomic_write_text(salt_path, salt + "\n")
    h = hashlib.sha256(salt.encode("ascii") + b"\0" + package_version().encode("ascii"))
    return h.hexdigest()[:32]


def _confirm(message: str) -> bool:
    sys.stdout.write(f"{message} [yes/N] ")
    sys.stdout.flush()
    return sys.stdin.readline().strip().lower() in ("yes", "y")


def _password() -> str:
    pw = os.environ.get(UPLOAD_PASSWORD_ENV, "").strip()
    if not pw:
        import getpass

        try:
            pw = getpass.getpass("upload password (posted on the dataset card): ").strip()
        except EOFError:
            pw = ""
    if not pw:
        raise UploadAuthError(
            f"no upload password. It's posted on the dataset card at "
            f"https://huggingface.co/datasets/{HF_REPO}; set "
            f"${UPLOAD_PASSWORD_ENV} for scripted use."
        )
    return pw


def upload_hf(bundle_dir: Path, *, confirm: bool = True) -> dict[str, Any]:
    """Push the bundle to a new ``submission-<id>-<ts>`` branch in one
    commit, under ``contributors/<id>/bundle-<ts>/``."""
    from ._shared_token import decrypt_with_password

    files = bundle_files(bundle_dir)
    contributor = submitter_id()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = f"contributors/{contributor}/bundle-{ts}"
    branch = f"submission-{contributor[:12]}-{ts}"
    print(f"target: HF dataset {HF_REPO}, branch {branch}")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    if confirm and not _confirm("submit this bundle?"):
        print("aborted.")
        return {"submitted": False}
    try:
        token = decrypt_with_password(_password())
    except ValueError as e:
        raise UploadAuthError(str(e)) from e

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # upload_folder needs the branch to exist already.
    api.create_branch(repo_id=HF_REPO, repo_type="dataset", branch=branch, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(bundle_dir),
        path_in_repo=prefix,
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=branch,
        commit_message=f"llmoji bundle from {contributor}",
        commit_description=(
            f"Submitted by `{contributor}` with llmoji {package_version()}. "
            f"Maintainer: review this branch and merge to `main` if approved."
        ),
        # Second line of defence behind bundle_files.
        allow_patterns=[*BUNDLE_TOPLEVEL_ALLOWLIST, f"*{BUNDLE_DATA_SUFFIX}"],
    )
    branch_url = f"https://huggingface.co/datasets/{HF_REPO}/tree/{branch}"
    print(f"submitted as {branch}; the maintainer reviews and merges by hand.")
    print(f"  branch: {branch_url}")
    print(f"  commit: {commit.commit_url}")
    return {
        "submitted": True,
        "branch": branch,
        "branch_url": branch_url,
        "commit_url": commit.commit_url,
        "path_in_repo": prefix,
        "files": [f.name for f in files],
    }


def write_archive(bundle_dir: Path, *, out_path: Path | None = None) -> Path:
    """Tar the allowlisted files flat into ``out_path`` (default
    ``~/.llmoji/bundle-<ts>.tar.gz``)."""
    files = bundle_files(bundle_dir)
    if out_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = bundle_dir.parent / f"bundle-{ts}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.name)
    return out_path
