"""llmoji command-line entry point.

Subcommands:

  install <provider>      write hook + register; idempotent
  uninstall <provider>    inverse of install; idempotent
  status                  installed providers, journal sizes, paths
  parse --provider <n> P  ingest a static export dump (e.g. claude.ai
                          conversations.json directory) into the
                          journal layer
  analyze                 scrape + canonicalize + Haiku synthesize,
                          write bundle to ~/.llmoji/bundle/
  upload --target {hf,email}  tar bundle, submit
  cache clear             delete the per-instance Haiku cache

The package keeps its on-disk state under ``$LLMOJI_HOME``
(default ``~/.llmoji``). The user's only ship-able artifact is the
bundle directory between ``analyze`` and ``upload``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import paths
from .haiku import cache_size
from .providers import PROVIDERS, get_provider
from .sources.claude_export import iter_claude_export
from .sources.journal import iter_journal


# ---------------------------------------------------------------------------
# install / uninstall / status
# ---------------------------------------------------------------------------


def _cmd_install(args: argparse.Namespace) -> int:
    p = get_provider(args.provider)
    p.install()
    s = p.status()
    print(f"installed {p.name}.")
    print(f"  hook:     {s.hook_path}")
    print(f"  settings: {s.settings_path}")
    print(f"  journal:  {s.journal_path}")
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    p = get_provider(args.provider)
    p.uninstall()
    print(f"uninstalled {p.name}. journal at {p.journal_path} preserved.")
    return 0


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _cmd_status(args: argparse.Namespace) -> int:
    home = paths.llmoji_home()
    print(f"llmoji home: {home}")
    print()
    print("providers:")
    for name in PROVIDERS:
        p = get_provider(name)
        s = p.status()
        marker = "✓" if s.installed else "·"
        kw = "installed" if s.installed else "not installed"
        rows = f"{s.journal_rows} rows" if s.journal_exists else "no journal"
        bytes_ = (
            f", {_human_bytes(s.journal_bytes)}"
            if s.journal_exists else ""
        )
        print(f"  {marker} {name:<14} {kw:<14} ({rows}{bytes_})")
        print(f"        hook:    {s.hook_path}")
        print(f"        journal: {s.journal_path}")
    n_rows, n_bytes = cache_size(paths.cache_per_instance_path())
    print()
    print(
        f"per-instance Haiku cache: {n_rows} entries, "
        f"{_human_bytes(n_bytes)} at {paths.cache_per_instance_path()}"
    )
    bundle_dir = paths.bundle_dir()
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        files = sorted(p for p in bundle_dir.iterdir() if p.is_file())
        print(f"bundle ready at {bundle_dir} ({len(files)} files):")
        for f in files:
            print(f"  - {f.name}  ({_human_bytes(f.stat().st_size)})")
    else:
        print(f"no bundle (run `llmoji analyze`).")

    # Generic-JSONL contract zone
    journals = paths.journals_dir()
    if journals.exists():
        extra = sorted(p for p in journals.glob("*.jsonl") if p.is_file())
        if extra:
            print(f"\nextra journals at {journals}:")
            for j in extra:
                lines = sum(1 for line in j.open() if line.strip())
                print(f"  - {j.name}  ({lines} rows)")
    return 0


# ---------------------------------------------------------------------------
# parse — ingest a static dump
# ---------------------------------------------------------------------------


def _cmd_parse(args: argparse.Namespace) -> int:
    if args.provider != "claude.ai":
        print(
            f"unknown --provider {args.provider!r} for parse. "
            f"v1.0 supports: claude.ai (extends as more static-dump "
            f"formats land).",
            file=sys.stderr,
        )
        return 2
    paths.ensure_home()
    out_path = paths.journals_dir() / "claude_ai_export.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w") as f:
        for row in iter_claude_export([Path(p) for p in args.paths]):
            r = {
                "ts": row.timestamp,
                "model": row.model or "",
                "cwd": row.cwd or "",
                "kaomoji": row.first_word,
                "user_text": row.surrounding_user,
                "assistant_text": row.assistant_text,
            }
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {out_path}.")
    return 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _gather_rows():
    """Iterate every installed provider's journal + any extra
    JSONLs under ``~/.llmoji/journals/``."""
    for name in PROVIDERS:
        p = get_provider(name)
        if not p.journal_path.exists():
            continue
        yield from iter_journal(p.journal_path, source=p.name)
    journals = paths.journals_dir()
    if journals.exists():
        for j in sorted(journals.glob("*.jsonl")):
            label = j.stem
            yield from iter_journal(j, source=label)


def _cmd_analyze(args: argparse.Namespace) -> int:
    # Lazy import — analyze needs anthropic, which we don't want to
    # require for `status` / `install`.
    from .analyze import run_analyze

    rows = list(_gather_rows())
    if not rows:
        print(
            "no journal rows found. install at least one provider, run "
            "some kaomoji-bearing turns, then re-run.",
            file=sys.stderr,
        )
        return 2
    result = run_analyze(rows, notes=args.notes or "")
    print()
    print(
        f"analyze done: {result.canonical_unique} canonical kaomoji from "
        f"{result.total_rows} rows; "
        f"{result.stage_a_calls_made} new Haiku calls, "
        f"{result.stage_a_calls_cached} cached, "
        f"{result.stage_b_calls_made} syntheses."
    )
    print(f"bundle: {result.bundle_dir}")
    return 0


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _cmd_upload(args: argparse.Namespace) -> int:
    from .upload import upload_email, upload_hf

    bundle_dir = paths.bundle_dir()
    if not bundle_dir.exists() or not any(bundle_dir.iterdir()):
        print(
            "no bundle to upload. run `llmoji analyze` first.",
            file=sys.stderr,
        )
        return 2

    if args.target == "hf":
        upload_hf(bundle_dir, repo=args.hf_repo, confirm=not args.yes)
    elif args.target == "email":
        upload_email(bundle_dir, to=args.email_to, confirm=not args.yes)
    else:  # pragma: no cover — argparse enforces the choices
        return 2
    return 0


# ---------------------------------------------------------------------------
# cache clear
# ---------------------------------------------------------------------------


def _cmd_cache(args: argparse.Namespace) -> int:
    if args.cache_action != "clear":
        print(f"unknown cache action {args.cache_action!r}", file=sys.stderr)
        return 2
    cache_dir = paths.cache_dir()
    if not cache_dir.exists():
        print("no cache to clear.")
        return 0
    shutil.rmtree(cache_dir)
    print(f"cleared {cache_dir}.")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmoji",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("install", help="install a provider's hook")
    sp.add_argument("provider", choices=sorted(PROVIDERS))
    sp.set_defaults(func=_cmd_install)

    sp = sub.add_parser("uninstall", help="remove a provider's hook")
    sp.add_argument("provider", choices=sorted(PROVIDERS))
    sp.set_defaults(func=_cmd_uninstall)

    sp = sub.add_parser("status", help="show installed providers + bundle state")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser(
        "parse",
        help="ingest a static export into the journal layer",
    )
    sp.add_argument("--provider", required=True,
                    help="dump format. v1.0: claude.ai")
    sp.add_argument("paths", nargs="+", type=Path,
                    help="one or more directories containing the dump file(s)")
    sp.set_defaults(func=_cmd_parse)

    sp = sub.add_parser(
        "analyze",
        help="scrape + canonicalize + Haiku synthesize → bundle",
    )
    sp.add_argument(
        "--notes", default="",
        help="optional free-form note that lands in manifest.json",
    )
    sp.set_defaults(func=_cmd_analyze)

    sp = sub.add_parser("upload", help="ship the bundle to a target")
    sp.add_argument("--target", required=True, choices=["hf", "email"])
    sp.add_argument(
        "--hf-repo", default="a9lim/llmoji",
        help="HF dataset repo to commit to (default: %(default)s)",
    )
    sp.add_argument(
        "--email-to", default="mx@a9l.im",
        help="email recipient (default: %(default)s)",
    )
    sp.add_argument(
        "--yes", action="store_true",
        help="skip the pre-submission confirmation prompt",
    )
    sp.set_defaults(func=_cmd_upload)

    sp = sub.add_parser("cache", help="manage local Haiku cache")
    sp.add_argument("cache_action", choices=["clear"])
    sp.set_defaults(func=_cmd_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
