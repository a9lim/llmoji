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
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterator

from . import paths
from ._util import human_bytes, scrape_row_to_journal_line
from .providers import PROVIDERS, HookInstaller, ProviderStatus, get_provider
from .scrape import ScrapeRow
from .sources.chatgpt_export import iter_chatgpt_export
from .sources.claude_export import iter_claude_export
from .sources.journal import iter_journal
from .synth import cache_size


# ---------------------------------------------------------------------------
# install / uninstall / status
# ---------------------------------------------------------------------------


def _print_install_summary(p: HookInstaller, s: ProviderStatus) -> None:
    """One block of post-install output per provider."""
    print(f"installed {p.name}.")
    print(f"  hook:     {s.hook_path}")
    if s.nudge_hook_path is not None:
        print(f"  nudge:    {s.nudge_hook_path}")
    print(f"  settings: {s.settings_path}")
    print(f"  journal:  {s.journal_path}")


def _install_one(name: str) -> tuple[bool, str | None]:
    """Install a single provider by name. Returns
    ``(succeeded, error_message)``. Used by the autodetect path so
    one corrupt config doesn't take down the rest of the batch.
    """
    p = get_provider(name)
    try:
        p.install()
    except Exception as e:  # noqa: BLE001 — surfaced to the CLI verbatim
        return False, f"{type(e).__name__}: {e}"
    _print_install_summary(p, p.status())
    return True, None


def _cmd_install(args: argparse.Namespace) -> int:
    # Explicit provider: legacy single-target path.
    if args.provider is not None:
        ok, err = _install_one(args.provider)
        if not ok:
            print(f"install failed for {args.provider}: {err}", file=sys.stderr)
            return 1
        return 0

    # Autodetect path: enumerate every registered provider, install
    # for each whose harness home dir exists on disk.
    detected = [
        name for name in PROVIDERS if get_provider(name).is_present()
    ]
    if not detected:
        print(
            "no harnesses detected (looked for "
            + ", ".join(
                f"{name} ({get_provider(name).settings_path.parent})"
                for name in PROVIDERS
            )
            + "). install a supported harness, or pass an explicit "
            "provider name (e.g. `llmoji install claude_code`).",
            file=sys.stderr,
        )
        return 2

    print("detected harnesses:")
    for name in detected:
        print(f"  - {name}  ({get_provider(name).settings_path.parent})")
    if not args.yes:
        try:
            ans = input(
                f"install llmoji hooks into {len(detected)} harness(es)? [y/N] "
            ).strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("aborted.")
            return 1
    print()

    # Partial success: one corrupt config doesn't kill the rest.
    failures: list[tuple[str, str]] = []
    for name in detected:
        ok, err = _install_one(name)
        if not ok:
            failures.append((name, err or ""))
        print()
    if failures:
        print(f"{len(failures)} of {len(detected)} provider(s) failed:",
              file=sys.stderr)
        for name, err in failures:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    p = get_provider(args.provider)
    p.uninstall()
    print(f"uninstalled {p.name}. journal at {p.journal_path} preserved.")
    return 0


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
        # Journal row count costs a full file scan; analyze re-walks
        # it via iter_journal anyway, so report bytes only here.
        journal = (
            human_bytes(s.journal_bytes) if s.journal_exists else "no journal"
        )
        print(f"  {marker} {name:<14} {kw:<14} ({journal})")
        # Annotate per-piece state only on failure — the ✓ marker
        # already implies both halves are good when set.
        hook_suffix = "" if s.main_installed else "  (missing)"
        print(f"        hook:    {s.hook_path}{hook_suffix}")
        if s.nudge_hook_path is not None:
            nudge_suffix = "" if s.nudge_installed else "  (missing)"
            print(f"        nudge:   {s.nudge_hook_path}{nudge_suffix}")
        print(f"        journal: {s.journal_path}")
    cache_path = paths.cache_per_instance_path()
    n_bytes = cache_size(cache_path)
    print()
    print(
        f"per-instance synth cache: {human_bytes(n_bytes)} at {cache_path}"
    )
    bundle_dir = paths.bundle_dir()
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        # Bundle layout: manifest.json plus one
        # <source-model>.jsonl per source model the journal saw,
        # all at the top level.
        files = sorted(p for p in bundle_dir.iterdir() if p.is_file())
        n_data = sum(1 for p in files if p.suffix == ".jsonl")
        print(
            f"bundle ready at {bundle_dir} "
            f"({len(files)} file(s), {n_data} per-source-model "
            f".jsonl):"
        )
        for f in files:
            print(f"  - {f.name}  ({human_bytes(f.stat().st_size)})")
    else:
        print("no bundle (run `llmoji analyze`).")

    # Generic-JSONL contract zone
    journals = paths.journals_dir()
    if journals.exists():
        extra = sorted(p for p in journals.glob("*.jsonl") if p.is_file())
        if extra:
            print(f"\nextra journals at {journals}:")
            for j in extra:
                with j.open() as f:
                    lines = sum(1 for line in f if line.strip())
                print(f"  - {j.name}  ({lines} rows)")
    return 0


# ---------------------------------------------------------------------------
# parse — ingest a static dump
# ---------------------------------------------------------------------------


def _write_journal_rows(rows: Iterator[ScrapeRow], out_name: str) -> int:
    """Persist a stream of :class:`ScrapeRow` to a journal JSONL.

    Used by every static-export parser: the row-to-6-field
    journal-line mapping is identical across formats, only the
    upstream :class:`ScrapeRow` iterator and output filename differ.
    The schema lives in :func:`llmoji._util.scrape_row_to_journal_line`
    so a future change flows through one function.
    """
    paths.ensure_home()
    out_path = paths.journals_dir() / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(scrape_row_to_journal_line(row), ensure_ascii=False)
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows to {out_path}.")
    return 0


def _parse_claude_ai(args: argparse.Namespace) -> int:
    return _write_journal_rows(
        iter_claude_export([Path(p) for p in args.paths]),
        "claude_ai_export.jsonl",
    )


def _parse_chatgpt(args: argparse.Namespace) -> int:
    return _write_journal_rows(
        iter_chatgpt_export([Path(p) for p in args.paths]),
        "chatgpt_export.jsonl",
    )


# Registry of static-dump parsers. Adding a new format = add an
# entry. The CLI dispatches off ``--provider`` against this dict.
_PARSERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "claude.ai": _parse_claude_ai,
    "chatgpt": _parse_chatgpt,
}


def _cmd_parse(args: argparse.Namespace) -> int:
    # argparse choices=_PARSERS keys already enforces a known
    # provider; the only way args.provider lands here is through the
    # registered set.
    return _PARSERS[args.provider](args)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _gather_rows() -> Iterator[ScrapeRow]:
    """Iterate every installed provider's journal + any extra
    JSONLs under ``~/.llmoji/journals/``.

    Live-hook journals get a ``-hook`` suffix on the source name
    (``claude_code-hook``, etc.) so the source field is honest
    about where the row came from. Static-export journals at
    ``~/.llmoji/journals/<name>.jsonl`` use the file stem verbatim
    — they aren't hooks, so the suffix would lie.
    """
    for name in PROVIDERS:
        p = get_provider(name)
        if not p.journal_path.exists():
            continue
        yield from iter_journal(p.journal_path, source=f"{p.name}-hook")
    journals = paths.journals_dir()
    if journals.exists():
        for j in sorted(journals.glob("*.jsonl")):
            label = j.stem
            yield from iter_journal(j, source=label)


def _cmd_analyze(args: argparse.Namespace) -> int:
    # Lazy import — analyze pulls in the chosen backend's SDK,
    # which we don't want to require for `status` / `install`.
    from .analyze import run_analyze

    backend = args.backend
    base_url = args.base_url
    model_id = args.model

    if backend == "local":
        if not base_url or not model_id:
            print(
                "--backend local requires both --base-url and --model "
                "(or LLMOJI_BASE_URL + LLMOJI_MODEL env vars).",
                file=sys.stderr,
            )
            return 2
    else:
        # Loud failure beats silent ignore — anthropic and openai
        # backends always use the pinned snapshot, so passing
        # --model / --base-url alongside is almost certainly a
        # mistake.
        if base_url or model_id:
            print(
                f"--backend {backend} doesn't accept --base-url / --model "
                f"(both are pinned to default snapshots). "
                f"Drop those flags or switch to --backend local.",
                file=sys.stderr,
            )
            return 2

    rows = list(_gather_rows())
    if not rows:
        print(
            "no journal rows found. install at least one provider, run "
            "some kaomoji-bearing turns, then re-run.",
            file=sys.stderr,
        )
        return 2
    result = run_analyze(
        rows,
        notes=args.notes or "",
        backend=backend,
        base_url=base_url,
        model_id=model_id,
        concurrency=args.concurrency,
    )
    print()
    print(
        f"analyze done: {result.canonical_unique} canonical kaomoji from "
        f"{result.total_rows} rows; "
        f"{result.stage_a_calls_made} new synth calls, "
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
    # ``args.cache_action`` is constrained to ``["clear"]`` by
    # argparse; only one branch needed here.
    del args
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

    sp = sub.add_parser(
        "install",
        help="install a provider's hook (no arg → autodetect all)",
    )
    sp.add_argument(
        "provider", nargs="?", default=None, choices=sorted(PROVIDERS),
        help=(
            "explicit provider; omit to autodetect every harness whose "
            "home directory exists under $HOME"
        ),
    )
    sp.add_argument(
        "--yes", action="store_true",
        help="skip the autodetect confirmation prompt (no-op with explicit provider)",
    )
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
    sp.add_argument("--provider", required=True, choices=sorted(_PARSERS),
                    help="dump format")
    sp.add_argument("paths", nargs="+", type=Path,
                    help="one or more directories containing the dump file(s)")
    sp.set_defaults(func=_cmd_parse)

    sp = sub.add_parser(
        "analyze",
        help="scrape + canonicalize + synthesize → bundle",
    )
    sp.add_argument(
        "--notes", default="",
        help="optional free-form note that lands in manifest.json",
    )
    sp.add_argument(
        "--backend",
        choices=["anthropic", "openai", "local"],
        default=os.environ.get("LLMOJI_BACKEND", "anthropic"),
        help=(
            "synthesis backend. anthropic (default) and openai use "
            "their pinned default snapshots; local needs --base-url "
            "and --model. env: LLMOJI_BACKEND."
        ),
    )
    sp.add_argument(
        "--base-url",
        default=os.environ.get("LLMOJI_BASE_URL"),
        help=(
            "OpenAI-compatible base URL for --backend local "
            "(e.g. http://localhost:11434/v1 for Ollama). "
            "env: LLMOJI_BASE_URL."
        ),
    )
    sp.add_argument(
        "--model",
        default=os.environ.get("LLMOJI_MODEL"),
        help=(
            "model id for --backend local "
            "(e.g. llama3.1, qwen2.5:14b). env: LLMOJI_MODEL."
        ),
    )
    sp.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            "Stage-A/B worker count (default 1; bump if your synth "
            "rate-limit tier has headroom). env: LLMOJI_CONCURRENCY."
        ),
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
