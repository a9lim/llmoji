"""Upload a bundle to one of two targets: HF dataset or email.

Both targets are opt-in and require explicit ``--target`` selection;
no implicit default. Whichever path the user takes, the bundle is
tarballed first (atomic, content-stable input).

HF target:
    Uses ``huggingface_hub.HfApi`` to commit a single tarball into a
    contributor-named subfolder of a public dataset
    (``a9lim/llmoji`` by default). The submitter's identifier is a
    16-hex-char salted hash of (a per-machine random token + the
    package version), persisted at ``~/.llmoji/state.json``. We do
    NOT collect HF usernames or any account-bound identifier — the
    salted hash is just enough to dedup repeat submissions from the
    same machine.

Email target:
    Build a ``mailto:`` URI with the bundle path printed in the
    body and instructions to attach manually. We don't ship SMTP.
    The CLI prints the path and opens the user's mail client via
    ``open`` (macOS) or ``xdg-open`` (Linux); the user attaches the
    tarball themselves.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import tarfile
import time
import urllib.parse
from pathlib import Path

from . import paths

DEFAULT_HF_REPO = "a9lim/llmoji"
DEFAULT_EMAIL_TO = "mx@a9l.im"

# Strict allowlist of files the bundle is permitted to ship.
# Anything else in `~/.llmoji/bundle/` is treated as user-added or
# stale and refused (loud failure beats silent leak). The two-file
# bundle schema is part of the v1.0 frozen public surface; bumping
# this list is a major version bump.
BUNDLE_ALLOWLIST: tuple[str, ...] = (
    "manifest.json",
    "descriptions.jsonl",
)


def _bundle_files(bundle_dir: Path) -> list[Path]:
    """Return the bundle's allowlisted files in deterministic order.

    Files outside the allowlist are skipped (and surfaced separately
    via :func:`_unexpected_bundle_files` so the caller can refuse to
    ship).
    """
    if not bundle_dir.exists():
        return []
    return sorted(
        bundle_dir / name
        for name in BUNDLE_ALLOWLIST
        if (bundle_dir / name).is_file()
    )


def _unexpected_bundle_files(bundle_dir: Path) -> list[Path]:
    """Return any files in the bundle dir that are NOT on the
    allowlist. Used to refuse `upload` when stale or user-added
    content is present."""
    if not bundle_dir.exists():
        return []
    allowed = set(BUNDLE_ALLOWLIST)
    return sorted(
        p for p in bundle_dir.iterdir()
        if p.is_file() and p.name not in allowed
    )


def tar_bundle(bundle_dir: Path, *, out_path: Path | None = None) -> Path:
    """Tar the bundle directory. Returns the tarball path.

    Strict allowlist: only ``manifest.json`` and
    ``descriptions.jsonl`` are included. If the bundle directory
    holds any other files, raise — refusing to ship is the safe
    default (the user can `rm` the extras and re-tar).
    """
    if out_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = bundle_dir.parent / f"bundle-{ts}.tar.gz"
    extras = _unexpected_bundle_files(bundle_dir)
    if extras:
        joined = ", ".join(p.name for p in extras)
        raise FileExistsError(
            f"refusing to tar {bundle_dir} — unexpected file(s) "
            f"{joined!r} are not in the v1.0 bundle allowlist "
            f"{BUNDLE_ALLOWLIST!r}. Remove them or re-run "
            f"`llmoji analyze` (which clears the bundle dir)."
        )
    files = _bundle_files(bundle_dir)
    if not files:
        raise FileNotFoundError(
            f"no allowlisted files in {bundle_dir} — run "
            f"`llmoji analyze` first"
        )
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"bundle/{f.name}")
    return out_path


def _submission_token() -> str:
    """Per-machine random token, persisted at ``~/.llmoji/state.json``.

    Generated once on first ``upload``. Never sent anywhere — only
    used as the salt in :func:`_submitter_id`.
    """
    state_path = paths.state_path()
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    if "submission_token" not in state:
        state["submission_token"] = secrets.token_hex(32)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state["submission_token"]


def _submitter_id() -> str:
    """Salted-hash submitter identifier. 32 hex chars (128 bits),
    stable per (machine, llmoji version).

    Length is 128 bits because the dataset README will pin
    submission identity to this string and an attacker who knows
    the per-machine token (server-side compromise) shouldn't be
    able to grind a same-id submission. 64 bits would be fine for
    pure dedup but isn't crypto-collision-resistant in the formal
    sense.
    """
    import hashlib

    from .providers.base import _package_version
    h = hashlib.sha256()
    h.update(_submission_token().encode("ascii"))
    h.update(b"\0")
    h.update(_package_version().encode("ascii"))
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


def upload_hf(
    bundle_dir: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    confirm: bool = True,
) -> dict:
    """Upload the bundle as a tarball under
    ``<contributor>/bundle-<ts>.tar.gz`` in the chosen HF dataset
    repo. Returns the submission metadata dict."""
    tarball = tar_bundle(bundle_dir)
    contributor = _submitter_id()
    target_path = f"contributors/{contributor}/{tarball.name}"

    print(f"target: HF dataset {repo} → {target_path}")
    print(f"local tarball: {tarball} ({tarball.stat().st_size} bytes)")
    if confirm and not _confirm("submit this bundle?"):
        print("aborted; tarball left on disk.")
        return {"submitted": False, "tarball": str(tarball)}

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(tarball),
        path_in_repo=target_path,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"llmoji bundle from {contributor}",
    )
    print(f"submitted to {repo} as {target_path}.")
    return {
        "submitted": True,
        "tarball": str(tarball),
        "repo": repo,
        "path_in_repo": target_path,
        "contributor": contributor,
    }


# ---------------------------------------------------------------------------
# Email target
# ---------------------------------------------------------------------------


def upload_email(
    bundle_dir: Path,
    *,
    to: str = DEFAULT_EMAIL_TO,
    confirm: bool = True,
) -> dict:
    """Build a mailto URI and open it in the system mail client; the
    user attaches the tarball manually. We don't ship SMTP."""
    tarball = tar_bundle(bundle_dir)
    print(f"target: email {to}")
    print(f"local tarball: {tarball} ({tarball.stat().st_size} bytes)")
    if confirm and not _confirm("open your mail client and attach this bundle?"):
        print("aborted; tarball left on disk.")
        return {"submitted": False, "tarball": str(tarball)}

    body = (
        "Hi,\n\n"
        f"Please find an llmoji bundle attached at:\n  {tarball}\n\n"
        "Bundle contents:\n"
    )
    for f in _bundle_files(bundle_dir):
        body += f"  - {f.name} ({f.stat().st_size} bytes)\n"
    body += "\nPaste this email into your mail client and attach the tarball manually.\n"

    mailto = "mailto:" + to + "?" + urllib.parse.urlencode({
        "subject": "llmoji bundle submission",
        "body": body,
    })

    if sys.platform == "darwin":
        opener = "open"
    elif sys.platform.startswith("linux"):
        opener = "xdg-open"
    else:
        opener = ""

    if opener:
        try:
            subprocess.run([opener, mailto], check=False)
        except FileNotFoundError:
            pass

    print(f"\nattach: {tarball}")
    print(f"mailto: {mailto[:200]}{'...' if len(mailto) > 200 else ''}")
    return {
        "submitted": True,  # we count "user instructed" as submitted
        "tarball": str(tarball),
        "to": to,
    }
