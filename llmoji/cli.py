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
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from . import paths
from ._util import human_bytes, scrape_row_to_journal_line
from .providers import PROVIDERS, HookInstaller, ProviderStatus, get_provider
from .scrape import ScrapeRow
from .sources.chatgpt_export import iter_chatgpt_export
from .sources.claude_export import iter_claude_export
from .sources.journal import iter_journal
from .synth import cache_size
from .taxonomy import canonicalize_kaomoji


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


# Canonical 6-field journal row schema (single source of truth lives
# at :func:`llmoji._util.journal_line_dict`; pinned here for the
# stats-mode validation pass so we don't import the helper just to
# read its keyword names).
_JOURNAL_FIELDS = (
    "ts", "model", "cwd", "kaomoji", "user_text", "assistant_text",
)


def _validate_journal_row(row: object) -> str | None:
    """Return ``None`` if ``row`` matches the canonical 6-field
    schema, else a short reason. ``stats`` mode runs every parsed
    row through this so a borked-hook-write surfaces as ``"X
    malformed rows"`` instead of silently dropping into analysis.
    """
    if not isinstance(row, dict):
        return f"top-level is {type(row).__name__}, not object"
    for f in _JOURNAL_FIELDS:
        if f not in row:
            return f"missing field {f!r}"
        if not isinstance(row[f], str):
            return f"field {f!r} is {type(row[f]).__name__}, not str"
    return None


def _provider_health_summary(s: ProviderStatus) -> tuple[str, list[str]]:
    """Return ``(marker, issues)`` where ``marker`` is one of ✓ ⚠ ✗
    · — corresponding to healthy / stale / broken / not-installed —
    and ``issues`` is a list of one-liner strings to print under the
    provider header. Empty list when healthy or not installed.
    """
    issues: list[str] = []
    if not s.installed:
        return "·", issues
    if s.settings_parse_error is not None:
        issues.append(f"settings unparseable: {s.settings_parse_error}")
    if not s.main_hook_current:
        issues.append("main hook content stale (re-run install)")
    if s.nudge_hook_path is not None and not s.nudge_hook_current:
        issues.append("nudge hook content stale (re-run install)")
    if issues:
        # Settings-corrupt is a hard break; stale is a warn.
        marker = "✗" if s.settings_parse_error is not None else "⚠"
    else:
        marker = "✓"
    return marker, issues


def _walk_journals_for_stats(provider_filter: str | None) -> dict[str, Any]:
    """Walk every relevant journal once: count rows, validate each
    against the 6-field schema, tally by source / source_model /
    canonical face. Returns a dict shaped for both human + JSON
    rendering.

    ``provider_filter`` (when set) restricts the walk to one managed
    provider's hook journal — generic-JSONL ``~/.llmoji/journals/``
    files are still included since they're not provider-scoped.
    """
    rows_total = 0
    rows_malformed = 0
    malformed_examples: list[str] = []
    by_source: Counter[str] = Counter()
    by_source_model: Counter[str] = Counter()
    by_canonical: Counter[str] = Counter()

    def _scan(path: Path, source: str) -> None:
        nonlocal rows_total, rows_malformed
        if not path.exists():
            return
        with path.open() as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                rows_total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    rows_malformed += 1
                    if len(malformed_examples) < 3:
                        malformed_examples.append(
                            f"{path.name}:{lineno}  json: {e}"
                        )
                    continue
                problem = _validate_journal_row(row)
                if problem is not None:
                    rows_malformed += 1
                    if len(malformed_examples) < 3:
                        malformed_examples.append(
                            f"{path.name}:{lineno}  {problem}"
                        )
                    continue
                by_source[source] += 1
                model = row.get("model") or source
                by_source_model[model] += 1
                # Canonicalize on the way in so frequency tables roll
                # near-variants together — same bucketing analyze uses.
                canon = canonicalize_kaomoji(row["kaomoji"])
                if canon:
                    by_canonical[canon] += 1

    targets = (
        [provider_filter] if provider_filter is not None else list(PROVIDERS)
    )
    for name in targets:
        p = get_provider(name)
        _scan(p.journal_path, source=f"{name}-hook")
    if provider_filter is None:
        # Generic-JSONL contract zone.
        journals_dir = paths.journals_dir()
        if journals_dir.exists():
            for j in sorted(journals_dir.glob("*.jsonl")):
                _scan(j, source=j.stem)

    return {
        "rows_total": rows_total,
        "rows_malformed": rows_malformed,
        "malformed_examples": malformed_examples,
        "by_source": dict(by_source.most_common()),
        "by_source_model": dict(by_source_model.most_common()),
        "by_canonical_top": by_canonical.most_common(),
    }


def _print_status_human(
    home: Path,
    snapshots: list[ProviderStatus],
    stats_data: dict[str, Any] | None,
    *,
    top_n: int,
) -> None:
    print(f"llmoji home: {home}")
    print()
    print("providers:")
    for s in snapshots:
        marker, issues = _provider_health_summary(s)
        if not s.installed:
            kw = "not installed"
        elif issues:
            # Keep the install verb, surface issues separately so the
            # signal "we're set up but something's off" is honest.
            kw = "installed (issues)"
        else:
            kw = "installed"
        journal = (
            human_bytes(s.journal_bytes) if s.journal_exists else "no journal"
        )
        print(f"  {marker} {s.name:<14} {kw:<20} ({journal})")
        # Per-piece state — annotate only on failure to keep the
        # default output compact when everything's healthy.
        hook_suffix = "" if s.main_installed else "  (missing)"
        if s.main_installed and not s.main_hook_current:
            hook_suffix = "  (stale content)"
        print(f"        hook:    {s.hook_path}{hook_suffix}")
        if s.nudge_hook_path is not None:
            nudge_suffix = "" if s.nudge_installed else "  (missing)"
            if s.nudge_installed and not s.nudge_hook_current:
                nudge_suffix = "  (stale content)"
            print(f"        nudge:   {s.nudge_hook_path}{nudge_suffix}")
        settings_suffix = ""
        if s.settings_parse_error is not None:
            settings_suffix = f"  (unparseable: {s.settings_parse_error})"
        print(f"        settings:{s.settings_path}{settings_suffix}")
        print(f"        journal: {s.journal_path}")
        for issue in issues:
            print(f"        ⚠ {issue}")

    cache_path = paths.cache_per_instance_path()
    n_bytes = cache_size(cache_path)
    print()
    print(
        f"per-instance synth cache: {human_bytes(n_bytes)} at {cache_path}"
    )
    bundle_dir = paths.bundle_dir()
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
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

    # Generic-JSONL contract zone (file-level listing — distinct from
    # the stats walk below which counts rows).
    journals = paths.journals_dir()
    if journals.exists():
        extra = sorted(p for p in journals.glob("*.jsonl") if p.is_file())
        if extra:
            print(f"\nextra journals at {journals}:")
            for j in extra:
                print(f"  - {j.name}  ({human_bytes(j.stat().st_size)})")

    if stats_data is not None:
        print()
        print("--- stats ---")
        rt = stats_data["rows_total"]
        rm = stats_data["rows_malformed"]
        print(f"rows: {rt} total, {rm} malformed")
        if rm and stats_data["malformed_examples"]:
            print("malformed examples:")
            for ex in stats_data["malformed_examples"]:
                print(f"  - {ex}")
        if stats_data["by_source"]:
            print("\nby source:")
            for k, v in stats_data["by_source"].items():
                print(f"  {v:>6}  {k}")
        if stats_data["by_source_model"]:
            print("\nby source model:")
            for k, v in stats_data["by_source_model"].items():
                print(f"  {v:>6}  {k}")
        if stats_data["by_canonical_top"]:
            top = stats_data["by_canonical_top"][:top_n]
            print(f"\ntop {len(top)} canonical kaomoji:")
            for canon, n in top:
                print(f"  {n:>6}  {canon}")


def _print_status_json(
    home: Path,
    snapshots: list[ProviderStatus],
    stats_data: dict[str, Any] | None,
    *,
    top_n: int,
) -> None:
    cache_path = paths.cache_per_instance_path()
    bundle_dir = paths.bundle_dir()
    bundle_files: list[dict[str, Any]] = []
    if bundle_dir.exists():
        for p in sorted(bundle_dir.iterdir()):
            if p.is_file():
                bundle_files.append(
                    {"name": p.name, "bytes": p.stat().st_size}
                )

    out: dict[str, Any] = {
        "llmoji_home": str(home),
        "providers": [
            {
                "name": s.name,
                "installed": s.installed,
                "main_installed": s.main_installed,
                "main_hook_current": s.main_hook_current,
                "nudge_installed": s.nudge_installed,
                "nudge_hook_current": s.nudge_hook_current,
                "settings_parse_error": s.settings_parse_error,
                "hook_path": str(s.hook_path),
                "nudge_hook_path": (
                    str(s.nudge_hook_path)
                    if s.nudge_hook_path is not None else None
                ),
                "settings_path": str(s.settings_path),
                "journal_path": str(s.journal_path),
                "journal_exists": s.journal_exists,
                "journal_bytes": s.journal_bytes,
            }
            for s in snapshots
        ],
        "cache": {
            "path": str(cache_path),
            "bytes": cache_size(cache_path),
        },
        "bundle": {
            "path": str(bundle_dir),
            "files": bundle_files,
        },
    }
    if stats_data is not None:
        out["stats"] = {
            "rows_total": stats_data["rows_total"],
            "rows_malformed": stats_data["rows_malformed"],
            "malformed_examples": stats_data["malformed_examples"],
            "by_source": stats_data["by_source"],
            "by_source_model": stats_data["by_source_model"],
            "by_canonical_top": [
                {"kaomoji": c, "count": n}
                for c, n in stats_data["by_canonical_top"][:top_n]
            ],
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _cmd_status(args: argparse.Namespace) -> int:
    home = paths.llmoji_home()
    target_names = (
        [args.provider] if args.provider else list(PROVIDERS)
    )
    snapshots = [get_provider(n).status() for n in target_names]
    stats_data = (
        _walk_journals_for_stats(args.provider) if args.stats else None
    )
    if args.json:
        _print_status_json(home, snapshots, stats_data, top_n=args.top)
    else:
        _print_status_human(home, snapshots, stats_data, top_n=args.top)
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


def _print_analyze_plan(plan: Any) -> None:
    """Render an :class:`llmoji.analyze.AnalyzePlan` for the
    ``--dry-run`` path. Typed as ``Any`` so cli.py doesn't need to
    eagerly import :mod:`llmoji.analyze` (which would pull in tqdm
    + the Stage-A/B machinery before the user even runs analyze).
    """
    print("--- analyze plan (dry run, no synth calls) ---")
    print(
        f"sources: {plan.total_rows} rows / "
        f"{plan.canonical_unique} canonical kaomoji across "
        f"{len(plan.counts_by_cell)} source model(s)"
    )
    if plan.providers_seen:
        print(f"providers seen: {', '.join(plan.providers_seen)}")
    if plan.model_counts:
        # Most-rows-first so the "where's the bulk of the corpus"
        # answer is visible at a glance.
        ranked = sorted(plan.model_counts.items(), key=lambda kv: -kv[1])
        print("source model row counts:")
        for sm, n in ranked:
            print(f"  {n:>6}  {sm}")
    n_cells = sum(len(p) for p in plan.counts_by_cell.values())
    print(
        f"\nstage A: up to {plan.stage_a_max_calls} sampled rows "
        f"across {n_cells} cell(s); {plan.stage_a_unique_calls} "
        f"unique cache key(s) → that's the cold-cache call count."
    )
    print(f"stage B: {plan.stage_b_calls} cell(s) → {plan.stage_b_calls} call(s)")
    print(
        f"\nestimated tokens (approx, char/4 heuristic): "
        f"input {plan.estimated_input_tokens:,} / "
        f"output {plan.estimated_output_tokens:,}"
    )
    if plan.estimated_cost_usd is not None:
        print(
            f"estimated cost: ~${plan.estimated_cost_usd:.4f} "
            f"(backend={plan.backend}, model={plan.model_id})"
        )
    else:
        print(
            f"estimated cost: n/a — backend {plan.backend!r} not in "
            f"the rate table (local backends, or new pricing). "
            f"Edit BACKEND_RATES_USD_PER_1M_TOKENS in synth_prompts.py "
            f"to add."
        )
    print("\nrun without --dry-run to execute.")


def _cmd_analyze(args: argparse.Namespace) -> int:
    # Lazy import — analyze pulls in the chosen backend's SDK,
    # which we don't want to require for `status` / `install`.
    from .analyze import plan_analyze, run_analyze

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

    if args.dry_run:
        plan = plan_analyze(
            rows, backend=backend, base_url=base_url, model_id=model_id,
        )
        _print_analyze_plan(plan)
        return 0

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
# import — replay native session/transcript files into the journal
# ---------------------------------------------------------------------------


def _cmd_import(args: argparse.Namespace) -> int:
    """``llmoji import <provider>`` — dedup-aware merge of historical
    session/transcript files into the live journal. Internal module
    is :mod:`llmoji.backfill`; ``import`` is the user-facing verb
    because "replay session files into the journal" is the dominant
    mental model and ``backfill`` collides namewise with what users
    might expect to be "redo the analyze pass."
    """
    from .backfill import import_provider

    try:
        result = import_provider(
            args.provider,
            since=args.since,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"import failed: {e}", file=sys.stderr)
        return 2

    label = "would append" if args.dry_run else "appended"
    skipped = result.rows_seen - result.rows_novel
    print(
        f"{args.provider}: saw {result.rows_seen} row(s); {label} "
        f"{result.rows_novel}; skipped {skipped} dedup hit(s)."
    )
    if args.dry_run:
        print("(dry run — journal not modified)")
    elif result.rows_novel:
        print(
            f"journal: {get_provider(args.provider).journal_path}\n"
            f"run `llmoji analyze` to fold them into the next bundle."
        )
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
    # Don't pass __doc__ as description — it hand-rolls a Subcommands:
    # block that duplicates argparse's auto-generated listing under
    # "positional arguments". Keep the module docstring for readers of
    # the file; give argparse just the framing prose.
    p = argparse.ArgumentParser(
        prog="llmoji",
        description=(
            "llmoji command-line entry point. State lives under "
            "$LLMOJI_HOME (default ~/.llmoji); the only ship-able "
            "artifact is the bundle directory between `analyze` and "
            "`upload`."
        ),
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

    sp = sub.add_parser(
        "status",
        help=(
            "show installed providers + bundle state + cheap health "
            "checks (--stats walks journals for frequency tables + "
            "row validation)"
        ),
    )
    sp.add_argument(
        "--provider", choices=sorted(PROVIDERS), default=None,
        help="scope to a single provider (default: all)",
    )
    sp.add_argument(
        "--stats", action="store_true",
        help=(
            "walk journals once for kaomoji frequency tables + row "
            "schema validation (slower than the default summary)"
        ),
    )
    sp.add_argument(
        "--top", type=int, default=20,
        help="top N canonical kaomoji to show under --stats (default 20)",
    )
    sp.add_argument(
        "--json", action="store_true",
        help="machine-readable JSON output for CI / jq",
    )
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
    sp.add_argument(
        "--dry-run", action="store_true",
        help=(
            "print the plan + cost estimate without making any synth "
            "calls. token + cost figures are approximate (char/4 "
            "heuristic) — order-of-magnitude reliable, not a quote."
        ),
    )
    sp.set_defaults(func=_cmd_analyze)

    sp = sub.add_parser(
        "import",
        help=(
            "replay a provider's native session/transcript files into "
            "its journal (dedup-aware merge — re-runnable)"
        ),
    )
    sp.add_argument(
        "provider", choices=sorted(PROVIDERS),
        help="harness whose session files to walk",
    )
    sp.add_argument(
        "--since", default=None,
        help=(
            "ISO-8601 timestamp; skip rows with ts < SINCE "
            "(e.g. 2026-01-01T00:00:00Z)"
        ),
    )
    sp.add_argument(
        "--dry-run", action="store_true",
        help="walk + dedup but don't write the journal",
    )
    sp.set_defaults(func=_cmd_import)

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
