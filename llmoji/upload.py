"""Upload a bundle to one of two targets: HF dataset or email.

Both targets are opt-in and require explicit ``--target`` selection;
no implicit default.

HF target:
    Uses ``huggingface_hub.HfApi.upload_folder`` to commit the
    bundle's loose files into a contributor-named, timestamped
    subfolder of a public dataset (``a9lim/llmoji`` by default). The
    final repo layout is
    ``contributors/<hash>/bundle-<ts>/{manifest.json,<slug>.jsonl,...}``
    — one ``<source-model-slug>.jsonl`` per model the journal saw.
    Loose files (rather than a tarball) so the HF dataset viewer can
    auto-load every per-source-model ``*.jsonl`` directly via a
    ``data_files: contributors/**/*.jsonl`` configs entry on the
    dataset card. ``upload_folder`` does a single atomic
    commit so partial uploads can't land. The submitter's identifier
    is a 32-hex-char salted hash of (a per-machine random token +
    the package version), persisted at ``~/.llmoji/state.json``. We
    do NOT collect HF usernames or any account-bound identifier; the
    salted hash is just enough to dedup repeat submissions from the
    same machine.

Email target:
    Build a ``mailto:`` URI with the bundle path printed in the
    body and instructions to attach manually. We don't ship SMTP.
    The CLI prints the path and opens the user's mail client via
    ``open`` (macOS) or ``xdg-open`` (Linux); the user attaches the
    tarball themselves. Email keeps the tarball-as-archive shape
    because a single attachment is what an email recipient wants.
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


def tar_bundle(bundle_dir: Path, *, out_path: Path | None = None) -> Path:
    """Tar the bundle directory. Returns the tarball path.

    Strict allowlist: only ``manifest.json`` at the top level plus
    each ``<source-model>.jsonl`` is included. If the bundle holds
    anything else (subdirs, non-jsonl files, symlinks), raise
    :class:`BundleAllowlistError` — refusing to ship is the safe
    default (the user can `rm` the extras and re-tar).
    """
    if out_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = bundle_dir.parent / f"bundle-{ts}.tar.gz"
    files = _check_or_raise(bundle_dir, "tar")
    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f"bundle/{f.name}")
    return out_path


def _submission_token() -> str:
    """Per-machine random token, persisted at ``~/.llmoji/state.json``.

    Generated once on first ``upload``. Never sent anywhere — only
    used as the salt in :func:`submitter_id`.
    """
    state_path = paths.state_path()
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    if "submission_token" not in state:
        state["submission_token"] = secrets.token_hex(32)
        # Atomic write — losing the token mid-write would invalidate
        # the user's submitter id and double-credit them in the
        # dataset's per-machine dedup.
        atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    return state["submission_token"]


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


def upload_hf(
    bundle_dir: Path,
    *,
    repo: str = DEFAULT_HF_REPO,
    confirm: bool = True,
) -> dict[str, Any]:
    """Upload the bundle's loose files under
    ``contributors/<hash>/bundle-<ts>/`` in the chosen HF dataset
    repo. Returns the submission metadata dict.

    Pre-flight: refuse to upload if anything outside the flat
    allowlist (top-level ``manifest.json`` plus per-source-model
    ``*.jsonl`` files) is in the bundle dir, mirroring
    :func:`tar_bundle` (loud failure beats silent leak).
    ``upload_folder``'s ``allow_patterns`` is a second line of
    defense against the same class of bug.

    The dataset card's ``data_files: contributors/**/*.jsonl`` glob
    surfaces every per-source-model data file across contributors
    into the dataset viewer's single train split.
    """
    files = _check_or_raise(bundle_dir, "upload")
    contributor = submitter_id()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target_prefix = f"contributors/{contributor}/bundle-{ts}"

    print(f"target: HF dataset {repo} → {target_prefix}/")
    for f in files:
        print(
            f"  {f.relative_to(bundle_dir).as_posix()} "
            f"({f.stat().st_size} bytes)"
        )
    if confirm and not _confirm("submit this bundle?"):
        print("aborted.")
        return {"submitted": False}

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(bundle_dir),
        path_in_repo=target_prefix,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"llmoji bundle from {contributor}",
        # Top-level manifest + every per-source-model
        # ``<slug>.jsonl`` at the bundle root. The structural
        # allowlist check above is the primary defense; this is
        # the second line.
        allow_patterns=[
            *BUNDLE_TOPLEVEL_ALLOWLIST,
            f"*{BUNDLE_DATA_SUFFIX}",
        ],
    )
    print(f"submitted to {repo} as {target_prefix}/.")
    return {
        "submitted": True,
        "repo": repo,
        "path_in_repo": target_prefix,
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
    user attaches the tarball manually. We don't ship SMTP."""
    tarball = tar_bundle(bundle_dir)
    files, _ = _classify_bundle(bundle_dir)
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
    for f in files:
        rel = f.relative_to(bundle_dir).as_posix()
        body += f"  - {rel} ({f.stat().st_size} bytes)\n"
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
