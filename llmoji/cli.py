"""The ``llmoji`` command.

State lives under ``$LLMOJI_HOME`` (default ``~/.llmoji``); the bundle
directory between ``analyze`` and ``upload`` is the only thing that
ever leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from . import paths
from ._util import JOURNAL_FIELDS, human_bytes, journal_line
from .providers import PROVIDERS, ProviderStatus, get_provider
from .scrape import ScrapeRow
from .sources.chatgpt_export import iter_chatgpt_export
from .sources.claude_export import iter_claude_export
from .sources.journal import iter_journal
from .taxonomy import canonicalize_kaomoji

# `llmoji parse --provider <format>`: reader and the journal it writes.
_PARSERS: dict[str, tuple[Callable[[list[Path]], Iterator[ScrapeRow]], str]] = {
    "claude.ai": (iter_claude_export, "claude_ai_export.jsonl"),
    "chatgpt": (iter_chatgpt_export, "chatgpt_export.jsonl"),
}


def _journals() -> Iterator[tuple[Path, str]]:
    """Every journal ``analyze`` reads, with its source name: each
    provider's, then each file in ``~/.llmoji/journals/``."""
    for name in PROVIDERS:
        yield get_provider(name).journal_path, f"{name}-hook"
    if paths.journals_dir().exists():
        for j in sorted(paths.journals_dir().glob("*.jsonl")):
            yield j, j.stem


def _detected(names: list[str], verb: str, yes: bool) -> list[str] | None:
    """The providers in ``names`` whose harness is on this machine,
    confirmed with the user unless ``yes``. ``None`` means stop."""
    found = [n for n in names if get_provider(n).is_present()]
    if not found:
        looked = ", ".join(f"{n} ({get_provider(n).home_dir})" for n in names)
        print(f"no harnesses detected (looked for {looked}).", file=sys.stderr)
        return None
    print("detected:")
    for n in found:
        print(f"  - {n}  ({get_provider(n).home_dir})")
    if not yes:
        try:
            answer = input(f"{verb} {len(found)} harness(es)? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("aborted.")
            return None
    print()
    return found


def _each(names: list[str], action: Callable[[str], None]) -> int:
    """Run ``action`` per provider; one failure doesn't stop the rest."""
    failures = []
    for name in names:
        try:
            action(name)
        except Exception as e:  # noqa: BLE001 — reported per provider
            failures.append(f"{name}: {type(e).__name__}: {e}")
    for f in failures:
        print(f"failed: {f}", file=sys.stderr)
    return 1 if failures else 0


# --- install / uninstall / status ---


def _install(name: str) -> None:
    p = get_provider(name)
    p.install()
    print(f"installed {name}: hook {p.hook_path}, kaomoji block in {p.doc_path}")


def _uninstall(name: str) -> None:
    p = get_provider(name)
    p.uninstall()
    print(f"uninstalled {name}; journal kept at {p.journal_path}")


def _cmd_install(args: argparse.Namespace) -> int:
    names = [args.provider] if args.provider else _detected(list(PROVIDERS), "install into", args.yes)
    return 2 if names is None else _each(names, _install)


def _cmd_uninstall(args: argparse.Namespace) -> int:
    names = [args.provider] if args.provider else _detected(list(PROVIDERS), "uninstall from", args.yes)
    return 2 if names is None else _each(names, _uninstall)


def _issues(s: ProviderStatus) -> list[str]:
    if s.settings_error is not None:
        return [f"settings unparseable: {s.settings_error}"]
    if not s.installed and not s.doc_current:
        return []  # not installed
    out = []
    if not s.installed:
        out.append("hook missing or unregistered")
    elif not s.hook_current:
        out.append("hook stale (python moved, or an older llmoji wrote it)")
    if not s.doc_current:
        out.append(f"kaomoji block missing or edited in {s.doc_path}")
    return out


def _stats(provider: str | None) -> dict[str, Any]:
    rows = malformed = 0
    by_source: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_face: Counter[str] = Counter()
    for path, source in _journals():
        if provider and source != f"{provider}-hook":
            continue
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = None
                if not (isinstance(row, dict) and all(isinstance(row.get(k), str) for k in JOURNAL_FIELDS)):
                    malformed += 1
        for r in iter_journal(path, source=source):
            by_source[source] += 1
            by_model[r.model or source] += 1
            by_face[canonicalize_kaomoji(r.first_word)] += 1
    return {
        "rows_total": rows,
        "rows_malformed": malformed,
        "by_source": dict(by_source.most_common()),
        "by_source_model": dict(by_model.most_common()),
        "by_canonical": by_face.most_common(),
    }


def _cmd_status(args: argparse.Namespace) -> int:
    names = [args.provider] if args.provider else list(PROVIDERS)
    statuses = [get_provider(n).status() for n in names]
    stats = _stats(args.provider) if args.stats else None
    bundle = paths.bundle_dir()
    bundle_files = sorted(p for p in bundle.iterdir() if p.is_file()) if bundle.exists() else []
    errors = paths.hook_errors_path()
    if args.json:
        out: dict[str, Any] = {
            "llmoji_home": str(paths.llmoji_home()),
            "providers": [
                {
                    "name": s.name,
                    "healthy": s.healthy,
                    "installed": s.installed,
                    "hook_current": s.hook_current,
                    "doc_current": s.doc_current,
                    "settings_error": s.settings_error,
                    "hook_path": str(s.hook_path),
                    "settings_path": str(s.settings_path),
                    "doc_path": str(s.doc_path),
                    "journal_path": str(s.journal_path),
                    "journal_bytes": s.journal_bytes,
                }
                for s in statuses
            ],
            "hook_errors_log": str(errors) if errors.exists() else None,
            "bundle": [{"name": p.name, "bytes": p.stat().st_size} for p in bundle_files],
        }
        if stats is not None:
            out["stats"] = {**stats, "by_canonical": dict(stats["by_canonical"][: args.top])}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"llmoji home: {paths.llmoji_home()}\n")
    for s in statuses:
        issues = _issues(s)
        state = "ok" if s.healthy else "not installed" if not (s.installed or s.doc_current) else "issues"
        journal = human_bytes(s.journal_bytes) if s.journal_path.exists() else "no journal"
        print(f"  {s.name:<12} {state:<14} journal {s.journal_path} ({journal})")
        for issue in issues:
            print(f"      ! {issue} — `llmoji install {s.name}` repairs it")
    if errors.exists():
        print(f"\nhook errors logged at {errors}")
    cache = paths.cache_path()
    print(f"\nsynthesis cache: {human_bytes(cache.stat().st_size) if cache.exists() else 'empty'}")
    if bundle_files:
        print(f"bundle at {bundle}:")
        for p in bundle_files:
            print(f"  - {p.name}  ({human_bytes(p.stat().st_size)})")
    else:
        print("no bundle (run `llmoji analyze`).")
    if stats is not None:
        print(f"\nrows: {stats['rows_total']} total, {stats['rows_malformed']} malformed")
        for title, counts in (("source", stats["by_source"]), ("source model", stats["by_source_model"])):
            print(f"\nby {title}:")
            for k, v in counts.items():
                print(f"  {v:>6}  {k}")
        print(f"\ntop {args.top} faces:")
        for face, n in stats["by_canonical"][: args.top]:
            print(f"  {n:>6}  {face}")
    return 0


# --- parse / import ---


def _cmd_parse(args: argparse.Namespace) -> int:
    reader, filename = _PARSERS[args.provider]
    paths.ensure_home()
    out = paths.journals_dir() / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for row in reader(args.paths):
            f.write(json.dumps(journal_line(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows to {out}.")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from .capture import import_provider

    importable = [n for n in PROVIDERS if get_provider(n).importable]
    names = [args.provider] if args.provider else _detected(importable, "import history from", args.yes)
    if names is None:
        return 2

    def run(name: str) -> None:
        r = import_provider(name, since=args.since, dry_run=args.dry_run)
        verb = "would append" if args.dry_run else "appended"
        repaired = "would repair" if args.dry_run else "repaired"
        print(
            f"{name}: replayed {r.rows_seen} row(s); {verb} {r.rows_novel} new, "
            f"{repaired} {r.rows_updated} from their transcripts."
        )
        if r.unreadable:
            print(
                f"{name}: skipped {r.unreadable} compressed session file(s): no zstd "
                f"decoder (Python 3.14+ has one; else `pip install 'llmoji[zstd]'`)."
            )

    return _each(names, run)


# --- analyze / upload / cache ---


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import plan_analyze, run_analyze

    if args.backend == "local" and not (args.base_url and args.model):
        print("--backend local needs --base-url and --model.", file=sys.stderr)
        return 2
    if args.backend != "local" and (args.base_url or args.model):
        print(
            f"--backend {args.backend} uses its pinned model; --base-url and "
            f"--model are for --backend local.",
            file=sys.stderr,
        )
        return 2
    rows = [r for path, source in _journals() for r in iter_journal(path, source=source)]
    if not rows:
        print("no journal rows yet: install a provider and talk to it.", file=sys.stderr)
        return 2
    kw = {"backend": args.backend, "base_url": args.base_url, "model_id": args.model}
    if args.dry_run:
        plan = plan_analyze(rows, **kw)
        print(
            f"{plan.total_rows} rows, {plan.canonical_unique} faces, {plan.cell_count} "
            f"cells across {len(plan.model_counts)} source model(s); "
            f"providers: {', '.join(plan.providers_seen)}"
        )
        for sm, n in sorted(plan.model_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {sm}")
        print(
            f"\n{plan.cell_count - plan.cached} call(s) to {plan.backend}/{plan.model_id} "
            f"({plan.cached} cached), about {plan.estimated_input_tokens:,} input and "
            f"{plan.estimated_output_tokens:,} output tokens. Nothing was called."
        )
        return 0
    result = run_analyze(rows, notes=args.notes, concurrency=args.concurrency, **kw)
    print(
        f"analyze done: {result.canonical_unique} faces from {result.total_rows} rows; "
        f"{result.calls_made} new call(s), {result.calls_cached} cached. "
        f"Bundle: {result.bundle_dir}"
    )
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    from .upload import ARCHIVE_EMAIL, upload_hf, write_archive

    bundle = paths.bundle_dir()
    if args.target == "hf":
        upload_hf(bundle, confirm=not args.yes)
    else:
        path = write_archive(bundle)
        print(f"wrote {path}\nto contribute it without HF, email it to {ARCHIVE_EMAIL}.")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    if paths.cache_dir().exists():
        shutil.rmtree(paths.cache_dir())
        print(f"cleared {paths.cache_dir()}.")
    else:
        print("no cache to clear.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmoji",
        description=(
            "Collect kaomoji journals from coding agents and submit "
            "privacy-preserving aggregates. State lives in $LLMOJI_HOME "
            "(default ~/.llmoji); only the bundle between `analyze` and "
            "`upload` ever leaves the machine."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    providers = sorted(PROVIDERS)

    sp = sub.add_parser(
        "install",
        help="install the journal hook and the kaomoji request (no provider: every harness found)",
    )
    sp.add_argument("provider", nargs="?", choices=providers)
    sp.add_argument("--yes", action="store_true", help="skip the autodetect prompt")
    sp.set_defaults(func=_cmd_install)

    sp = sub.add_parser("uninstall", help="remove both; journals are kept")
    sp.add_argument("provider", nargs="?", choices=providers)
    sp.add_argument("--yes", action="store_true", help="skip the autodetect prompt")
    sp.set_defaults(func=_cmd_uninstall)

    sp = sub.add_parser("status", help="installed providers, journals, bundle")
    sp.add_argument("--provider", choices=providers)
    sp.add_argument("--stats", action="store_true", help="also count rows, models and faces")
    sp.add_argument("--top", type=int, default=20, help="faces to list under --stats")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("parse", help="ingest a claude.ai or ChatGPT data export")
    sp.add_argument("--provider", required=True, choices=sorted(_PARSERS))
    sp.add_argument("paths", nargs="+", type=Path, help="export directories")
    sp.set_defaults(func=_cmd_parse)

    sp = sub.add_parser(
        "import",
        help="replay a harness's session history into its journal (re-runnable)",
    )
    sp.add_argument(
        "provider", nargs="?", choices=sorted(n for n in PROVIDERS if get_provider(n).importable),
    )
    sp.add_argument("--since", help="ISO-8601; skip rows before it")
    sp.add_argument("--dry-run", action="store_true", help="count, don't write")
    sp.add_argument("--yes", action="store_true", help="skip the autodetect prompt")
    sp.set_defaults(func=_cmd_import)

    sp = sub.add_parser("analyze", help="synthesize the journals into a bundle")
    sp.add_argument("--notes", default="", help="free text for manifest.json")
    sp.add_argument(
        "--backend", choices=["anthropic", "openai", "local"],
        default=os.environ.get("LLMOJI_BACKEND", "anthropic"),
        help="env: LLMOJI_BACKEND",
    )
    sp.add_argument(
        "--base-url", default=os.environ.get("LLMOJI_BASE_URL"),
        help="OpenAI-compatible endpoint for --backend local. env: LLMOJI_BASE_URL",
    )
    sp.add_argument(
        "--model", default=os.environ.get("LLMOJI_MODEL"),
        help="model id for --backend local. env: LLMOJI_MODEL",
    )
    sp.add_argument(
        "--concurrency", type=int,
        help="parallel synthesis calls (default 1). env: LLMOJI_CONCURRENCY",
    )
    sp.add_argument("--dry-run", action="store_true", help="plan and estimate, call nothing")
    sp.set_defaults(func=_cmd_analyze)

    sp = sub.add_parser("upload", help="ship the bundle")
    sp.add_argument(
        "--target", required=True, choices=["hf", "archive"],
        help="hf: push to the dataset; archive: write a tarball to send by hand",
    )
    sp.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sp.set_defaults(func=_cmd_upload)

    sp = sub.add_parser("cache", help="manage the synthesis cache")
    sp.add_argument("cache_action", choices=["clear"])
    sp.set_defaults(func=_cmd_cache)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
