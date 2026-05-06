"""Single-stage analysis pipeline for the bundle (v2).

End-user pipeline (no GPU, no embedding, no axes):

  1. Iterate every installed provider's journal + any user-parsed
     static dumps; canonicalize each leading kaomoji via
     :func:`llmoji.taxonomy.canonicalize_kaomoji`; bucket by
     ``(source_model, canonical_kaomoji)``. ``source_model`` comes
     from each row's ``ScrapeRow.model`` field (the model that
     wrote the kaomoji-bearing turn) — when that's empty (some
     static-export rows don't carry a model id), fall back to the
     row's ``source`` name so the data still surfaces in the
     bundle.
  2. Per cell, sample up to :data:`INSTANCE_SAMPLE_CAP` rows
     deterministically; render them into the locked
     :data:`SYNTHESIZE_PROMPT` as numbered ``[Sample N]`` blocks
     with the leading kaomoji masked to ``[FACE]``; call the
     synthesizer's ``call_structured`` with
     :data:`SYNTHESIS_SCHEMA`. Cache hit by content-hashed
     ``(model, backend, base_url, source_model, canonical,
     sample_set_hash)`` short-circuits the API call. The single
     call returns a structured ``{primary_affect, stance_modality_
     function}`` adjective bag drawn purely from the locked
     :data:`LEXICON`.
  3. Emit a manifest + one ``<sanitized_source_model>.jsonl``
     per source model at the top of ``~/.llmoji/bundle/`` — the
     loose-files inspection gap the user reads before deciding to
     ``upload``.

The v1 pipeline was two-stage (per-instance describe → per-cell
synthesize). v2 collapses both into one call per cell. The
synthesizer sees all sampled instances at once, eliminating the
prose-from-prose paraphrase layer that compounded fluff and made
the resulting per-cell descriptions cluster as noise in PCA. The
locked output schema also forces every adjective to come from the
corpus vocabulary, removing free-form prose from the bundle
entirely.

Embedding, axis-projection, figures, clustering all live in
``llmoji-study`` — research-side. The bundle is the boundary.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from . import paths
from ._util import atomic_write_text, package_version, sanitize_model_id_for_path
from .scrape import ScrapeRow
from .synth import (
    Synthesizer,
    append_cache,
    cache_key,
    load_cache,
    make_synthesizer,
    mask_kaomoji,
    samples_hash,
)
from .synth_prompts import (
    BACKEND_RATES_USD_PER_1M_TOKENS,
    CHARS_PER_TOKEN_HEURISTIC,
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    ESTIMATE_OUTPUT_CHARS,
    LEXICON_VERSION,
    SYNTHESIS_SCHEMA,
    SYNTHESIZE_PROMPT,
)
from .taxonomy import canonicalize_kaomoji

# Per-cell sample cap. ``_sample`` returns ``min(cap, len(rows))``
# per cell — popular faces get capped, rare faces fully sampled.
# Eriskii used 4 for the original Claude-faces work; same number
# here keeps cross-corpus comparison apples-to-apples.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# Synthesis call concurrency. The Anthropic / OpenAI httpx clients
# are thread-safe, and a content-hash cache append-write is POSIX-
# atomic for sub-PIPE_BUF (4 KB) JSONL lines, so a small thread
# pool gives ~Nx wallclock speedup on cache misses with no
# coordination beyond per-future result handling on the main
# thread. Override via ``$LLMOJI_CONCURRENCY`` (>= 1) or
# ``--concurrency``.
#
# Default 1 because the org-level Haiku rate cap (50 req/min) trips
# intermittently even at concurrency=2 on multi-hundred-cell
# backfills; the SDK's ``max_retries=8`` exponential backoff
# recovers but burns wallclock. Bump if your tier has headroom. v2
# is roughly 5× lighter than v1 (one call per cell vs N+1) so even
# concurrency=1 reanalyzes orders of magnitude faster than v1.
DEFAULT_CONCURRENCY = 1


class AnalyzeError(RuntimeError):
    """Synthesis aborted with one or more failed structured-output
    calls.

    The cache is flushed for cells that succeeded before the
    failure — re-running ``llmoji analyze`` resumes from cache,
    paying API cost only for the cells that previously failed.
    """


@dataclass
class AnalyzeResult:
    """Summary stats reported back to the CLI / printed to the user."""

    total_rows: int
    canonical_unique: int
    providers_seen: list[str]
    bundle_dir: Path
    calls_made: int
    calls_cached: int


@dataclass
class AnalyzePlan:
    """``--dry-run`` snapshot — what an ``analyze`` invocation would
    compute, without making any synth calls.

    Counts assume a cold cache (worst case for cost — a warm-cache
    re-run pays for fewer dispatches). Per-cell sample counts
    respect :data:`INSTANCE_SAMPLE_CAP`. Token counts and cost are
    approximate (see :data:`CHARS_PER_TOKEN_HEURISTIC` and
    :data:`ESTIMATE_OUTPUT_CHARS` in :mod:`synth_prompts`) —
    reliable for "is this $0.04 or $4?" but not as a quote.
    """

    total_rows: int
    canonical_unique: int
    providers_seen: list[str]
    model_counts: dict[str, int]
    counts_by_cell: dict[str, dict[str, int]]
    cell_count: int          # total cells = max calls before sample-hash dedup
    unique_calls: int        # cells with distinct cache keys
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float | None  # None when backend isn't priced
    backend: str
    model_id: str


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def _bucket_by_source_model_and_canonical(
    rows: Iterable[ScrapeRow],
) -> tuple[
    dict[str, dict[str, list[ScrapeRow]]],
    list[str],
    dict[str, int],
]:
    """Group rows by (source_model, canonical_kaomoji).

    Source-model key per row: ``r.model`` if non-empty, else
    ``r.source`` (so rows whose harness didn't stamp a model id —
    e.g. static-export readers — still surface in the bundle under
    a self-documenting fallback bucket).

    Returns ``(buckets, providers_seen_sorted, model_counts)``:

    - ``buckets[source_model][canonical] -> list[ScrapeRow]``
    - ``providers_seen`` — sorted unique ``r.source`` set
    - ``model_counts`` — ``{source_model: total_rows}`` (BEFORE
      canonicalization filtering, so it matches what the journals
      actually held)
    """
    buckets: dict[str, dict[str, list[ScrapeRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    providers: set[str] = set()
    model_counts: Counter[str] = Counter()
    for r in rows:
        providers.add(r.source)
        source_model = (r.model or "").strip() or r.source
        model_counts[source_model] += 1
        canon = canonicalize_kaomoji(r.first_word)
        if not canon:
            continue
        buckets[source_model][canon].append(r)
    # Freeze the defaultdict → plain dict so downstream code can't
    # accidentally extend the structure on read.
    frozen: dict[str, dict[str, list[ScrapeRow]]] = {
        sm: dict(by_canon) for sm, by_canon in buckets.items()
    }
    return frozen, sorted(providers), dict(model_counts)


def _sample(
    rows: list[ScrapeRow],
    *,
    cap: int,
    seed_label: str,
) -> list[ScrapeRow]:
    """Deterministic uniform sampling per cell. Sort upstream by
    some stable key so re-runs hit the same instances."""
    if len(rows) <= cap:
        return list(rows)
    rng = random.Random(f"{INSTANCE_SAMPLE_SEED}:{seed_label}")
    return rng.sample(rows, cap)


def _resolve_concurrency(explicit: int | None) -> int:
    """Synthesis worker count: ``--concurrency`` (CLI) → explicit
    arg → ``$LLMOJI_CONCURRENCY`` → :data:`DEFAULT_CONCURRENCY`.
    Clamps to ``>=1``. Bad env values fall back silently to the
    default."""
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get("LLMOJI_CONCURRENCY")
    if raw is None:
        return DEFAULT_CONCURRENCY
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_CONCURRENCY


# ---------------------------------------------------------------------------
# Sample formatting + per-cell synthesis
# ---------------------------------------------------------------------------


def _format_samples(samples: list[tuple[str, str]]) -> str:
    """Render ``(user_text, masked_assistant_text)`` pairs as the
    numbered ``[Sample N]`` block series the prompt expects.

    Empty ``user_text`` → omit the ``User:`` line entirely (some
    rows from static exports don't carry a surrounding user turn,
    and the prompt should not invent one).
    """
    blocks: list[str] = []
    for i, (user, masked_assistant) in enumerate(samples, 1):
        if user:
            blocks.append(
                f"[Sample {i}]\nUser: {user}\nAssistant: {masked_assistant}"
            )
        else:
            blocks.append(f"[Sample {i}]\nAssistant: {masked_assistant}")
    return "\n\n".join(blocks)


def _synthesize_cells(
    synth: Synthesizer,
    buckets: dict[str, dict[str, list[ScrapeRow]]],
    *,
    cache_path: Path,
    print_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], int, int]:
    """For each ``(source_model, canonical)`` cell, sample ≤
    :data:`INSTANCE_SAMPLE_CAP` rows; render the prompt; call the
    backend's structured-output path; return ``(synth_by_cell,
    n_calls, n_cached)``.

    ``synth_by_cell[source_model][canonical]`` is the synthesis
    object (``{primary_affect, stance_modality_function}``) for
    the cell, sourced either from the cache (hit) or from the
    backend (miss).

    Cache-miss API calls run on a small thread pool (``max_workers``
    or ``$LLMOJI_CONCURRENCY``, default 1). Both SDK clients are
    thread-safe.

    Cache writes happen as each future succeeds, immediately, on
    the main thread inside the ``as_completed`` loop. If any
    dispatch raises, the cache is already flushed for cells that
    succeeded; we collect errors and raise :class:`AnalyzeError`
    once the loop has fully drained, so the user can re-run and
    resume from the cache. ``synth_by_cell`` is assembled in
    deterministic walk order so re-runs produce identical bundle
    contents.
    """
    cache = load_cache(cache_path)
    n_cached = 0

    # Walk every cell in deterministic (sorted source_model, sorted
    # canonical) order. Each entry records the cell identity, the
    # cache key, and either the cached synthesis (cache hit) or the
    # rendered prompt to dispatch (cache miss). After dispatch the
    # misses are populated with their ``synthesis`` in place.
    walk: list[dict[str, Any]] = []
    for source_model in sorted(buckets):
        per_canon = buckets[source_model]
        for canon in sorted(per_canon):
            sampled = _sample(
                per_canon[canon],
                cap=INSTANCE_SAMPLE_CAP,
                seed_label=f"{source_model}:{canon}",
            )
            samples = [
                (
                    (r.surrounding_user or "").strip(),
                    r.assistant_text or "",
                )
                for r in sampled
            ]
            sh = samples_hash(samples)
            key = cache_key(
                synth.model_id, synth.backend, synth.base_url,
                source_model, canon, sh,
            )
            hit = cache.get(key)
            if hit and isinstance(hit.get("synthesis"), dict):
                walk.append({
                    "sm": source_model, "canon": canon, "key": key,
                    "cached": True, "synthesis": hit["synthesis"],
                })
                n_cached += 1
                continue
            masked_samples = [
                (user, mask_kaomoji(assistant, r.first_word))
                for r, (user, assistant) in zip(sampled, samples)
            ]
            prompt = SYNTHESIZE_PROMPT.format(
                samples=_format_samples(masked_samples),
            )
            walk.append({
                "sm": source_model, "canon": canon, "key": key,
                "cached": False, "prompt": prompt, "synthesis": None,
            })

    # Group cache misses by key. Two cells with identical sampled
    # sets (rare — mostly happens when the same kaomoji shows up
    # in multiple source_model buckets with overlapping content)
    # share a key; without dedupe a cold-cache run would dispatch
    # each duplicate separately and potentially get different
    # adjective bags, while a warm-cache follow-up would read the
    # last-write-wins cache row for all duplicates. One dispatch
    # per unique key keeps cold and warm in lockstep.
    pending_by_key: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(walk):
        if not e["cached"]:
            pending_by_key[e["key"]].append(i)

    n_calls = 0
    if pending_by_key:
        workers = _resolve_concurrency(max_workers)
        if print_progress:
            print(
                f"synthesize: {n_cached} cache hit(s), "
                f"{len(pending_by_key)} dispatch(es) "
                f"({workers} workers)"
            )

        def _synth_one(prompt: str) -> dict[str, Any]:
            return synth.call_structured(prompt, schema=SYNTHESIS_SCHEMA)

        # Catching ``Exception`` (not ``BaseException``) so Ctrl-C
        # propagates naturally — the user gets the same partial-flush
        # behavior on interrupt as on a synth error, and a re-run
        # resumes from cache either way.
        errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_key = {
                pool.submit(_synth_one, walk[indices[0]]["prompt"]): key
                for key, indices in pending_by_key.items()
            }
            iterator = as_completed(future_to_key)
            if print_progress:
                iterator = tqdm(
                    iterator,
                    total=len(future_to_key),
                    desc="synthesize",
                    unit="cell",
                    dynamic_ncols=True,
                    leave=True,
                )
            for fut in iterator:
                key = future_to_key[fut]
                try:
                    synthesis = fut.result()
                except Exception as e:  # noqa: BLE001 — captured + reraised
                    errors[key] = e
                    continue
                indices = pending_by_key[key]
                for i in indices:
                    walk[i]["synthesis"] = synthesis
                row = {
                    "key": key,
                    "kaomoji": walk[indices[0]]["canon"],
                    "source_model": walk[indices[0]]["sm"],
                    "synthesis": synthesis,
                    "model": synth.model_id,
                    "backend": synth.backend,
                }
                append_cache(cache_path, row)
                cache[key] = row
                n_calls += 1

        if errors:
            first_err = next(iter(errors.values()))
            raise AnalyzeError(
                f"synthesize: {n_calls} of {len(pending_by_key)} dispatch(es) "
                f"succeeded ({len(errors)} failed). cache flushed for "
                f"successes; re-run `llmoji analyze` to resume.\n"
                f"first failure: {first_err!r}"
            ) from first_err

    # Assemble ``synth_by_cell`` in deterministic walk order. Every
    # walk entry is now populated (cache hit, or successfully
    # dispatched — we'd have raised above otherwise).
    synth_by_cell: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in walk:
        synth_by_cell[entry["sm"]][entry["canon"]] = entry["synthesis"]

    return (
        {sm: dict(p) for sm, p in synth_by_cell.items()},
        n_calls,
        n_cached,
    )


# ---------------------------------------------------------------------------
# Bundle write
# ---------------------------------------------------------------------------


def _clear_bundle_dir(bundle_dir: Path) -> None:
    """Remove every top-level entry from ``bundle_dir``.

    Stale per-source-model files from prior runs would silently leak
    into upload otherwise — the bundle is the privacy boundary, so
    we wipe everything that isn't about to be re-written. Subdirs
    from older 1.1.0 layouts are also cleared here on first analyze
    post-upgrade.

    Symlinks are unlinked, never followed: a symlinked subdir would
    cause ``shutil.rmtree`` to walk across the link and delete files
    outside the bundle dir. ``Path.is_symlink()`` is checked before
    ``is_dir()`` because a symlink-to-directory satisfies both.
    """
    import shutil
    if not bundle_dir.exists():
        return
    for p in bundle_dir.iterdir():
        if p.is_symlink():
            p.unlink()
        elif p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def _write_bundle(
    bundle_dir: Path,
    *,
    counts_by_cell: dict[str, dict[str, int]],
    synthesized_by_cell: dict[str, dict[str, dict[str, Any]]],
    providers_seen: list[str],
    model_counts: dict[str, int],
    submitter_id: str,
    synth_backend: str,
    synth_model_id: str,
    notes: str,
) -> None:
    """Write ``manifest.json`` + per-source-model
    ``<sanitized>.jsonl`` at the bundle root. Flat loose-files
    layout so the user can ``cat`` and review before ``upload``.

    ``counts_by_cell[source_model][canonical]`` and
    ``synthesized_by_cell[source_model][canonical]`` carry the same
    set of keys — one row per face per source model.
    ``synthesized_by_cell``'s value is the structured synthesis
    object (``{primary_affect, stance_modality_function}``) drawn
    from :data:`SYNTHESIS_SCHEMA`.

    ``total_synthesized_rows`` counts rows across files, so a face
    appearing in 4 source-model files contributes 4. The manifest
    carries ``lexicon_version`` so cross-corpus aggregation can
    refuse to mix lexicon-version-N cells with lexicon-version-M
    cells.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _clear_bundle_dir(bundle_dir)

    total_synth_rows = sum(
        len(per_canon) for per_canon in synthesized_by_cell.values()
    )

    manifest = {
        "llmoji_version": package_version(),
        "lexicon_version": LEXICON_VERSION,
        "synthesis_model_id": synth_model_id,
        "synthesis_backend": synth_backend,
        "submitter_id": submitter_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "providers_seen": providers_seen,
        "model_counts": model_counts,
        "total_synthesized_rows": int(total_synth_rows),
        "notes": notes,
    }
    atomic_write_text(
        bundle_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    # Reject sanitization collisions before we write — two distinct
    # source-model strings landing on the same slug (e.g. ``A/B``
    # and ``a__b`` both → ``a__b``) would silently overwrite each
    # other's per-source-model ``<slug>.jsonl``. Loud failure beats a
    # half-shipped bundle.
    slug_owners: dict[str, list[str]] = defaultdict(list)
    for source_model in synthesized_by_cell:
        slug_owners[sanitize_model_id_for_path(source_model)].append(
            source_model,
        )
    collisions = {s: o for s, o in slug_owners.items() if len(o) > 1}
    if collisions:
        details = "; ".join(
            f"{s!r} <- {sorted(owners)!r}"
            for s, owners in sorted(collisions.items())
        )
        raise ValueError(
            f"source-model slug collision in bundle write: {details}. "
            f"Two distinct ScrapeRow.model strings sanitize to the "
            f"same subfolder name; refusing to overwrite."
        )

    for source_model in sorted(synthesized_by_cell):
        slug = sanitize_model_id_for_path(source_model)
        out_path = bundle_dir / f"{slug}.jsonl"
        per_canon = synthesized_by_cell[source_model]
        counts = counts_by_cell.get(source_model, {})
        with out_path.open("w") as f:
            for canon in sorted(per_canon):
                row = {
                    "kaomoji": canon,
                    "count": int(counts.get(canon, 0)),
                    "synthesis": per_canon[canon],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_preview(
    bundle_dir: Path,
    *,
    counts_by_cell: dict[str, dict[str, int]],
    synthesized_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Print a per-source-model summary plus per-face count + the
    flattened adjective bag. The inspection gap depends on this —
    the user sees what they're about to publish before deciding to
    upload.
    """
    from ._util import flatten_synthesis

    print("\n--- bundle preview ---")
    print(f"location: {bundle_dir}")
    n_models = len(synthesized_by_cell)
    n_rows = sum(len(p) for p in synthesized_by_cell.values())
    print(f"{n_models} source-model file(s), {n_rows} synthesized row(s) total:\n")
    for source_model in sorted(synthesized_by_cell):
        per_canon = synthesized_by_cell[source_model]
        counts = counts_by_cell.get(source_model, {})
        slug = sanitize_model_id_for_path(source_model)
        print(f"  [{source_model}]  → {slug}.jsonl  ({len(per_canon)} faces)")
        for canon in sorted(per_canon, key=lambda c: -counts.get(c, 0)):
            n = counts.get(canon, 0)
            adj = flatten_synthesis(per_canon[canon])
            print(f"    n={n:>4}  {canon}  {adj}")
    print("\n--- end preview ---")
    print("review each <model>.jsonl before `llmoji upload`.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_default_model_id(backend: str, model_id: str | None) -> str:
    """Mirror :func:`llmoji.synth.make_synthesizer`'s pinning rules
    without instantiating a synthesizer (which would import an SDK).
    Used by :func:`plan_analyze` so a dry-run never touches the
    Anthropic / OpenAI clients.
    """
    if model_id is not None:
        return model_id
    if backend == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL_ID
    if backend == "openai":
        return DEFAULT_OPENAI_MODEL_ID
    # local: caller should have passed model_id; fall back to a
    # placeholder string so the cache_key hash works in dry-run mode.
    return "(local)"


def plan_analyze(
    rows: Iterable[ScrapeRow],
    *,
    backend: str = "anthropic",
    base_url: str | None = None,
    model_id: str | None = None,
) -> AnalyzePlan:
    """Build an :class:`AnalyzePlan` without calling any synth backend.

    Walks rows, buckets them, runs the same deterministic per-cell
    sampling :func:`_synthesize_cells` would, computes the
    per-cell cache keys (so identical sample sets across cells
    fold to one dispatch), and estimates token usage / cost
    against the per-1M rate table in :mod:`synth_prompts`. Backend
    SDKs are not imported (``cache_key`` and ``samples_hash`` are
    pure-Python; the synth classes that pull in SDKs are not
    invoked).
    """
    rows_list = list(rows)
    buckets, providers_seen, model_counts = (
        _bucket_by_source_model_and_canonical(rows_list)
    )
    counts_by_cell: dict[str, dict[str, int]] = {
        sm: {canon: len(rs) for canon, rs in per_canon.items()}
        for sm, per_canon in buckets.items()
    }

    resolved_model_id = _resolve_default_model_id(backend, model_id)
    base_url_str = base_url or ""

    # Walk the same cells / sample the same rows / compute the same
    # cache keys :func:`_synthesize_cells` would. Track unique keys +
    # accumulate input chars per dispatch so the estimate matches
    # what the real run would issue at cold cache.
    seen_keys: set[str] = set()
    cell_count = 0
    input_chars = 0

    for sm in sorted(buckets):
        per_canon = buckets[sm]
        for canon in sorted(per_canon):
            cell_count += 1
            sampled = _sample(
                per_canon[canon],
                cap=INSTANCE_SAMPLE_CAP,
                seed_label=f"{sm}:{canon}",
            )
            samples = [
                (
                    (r.surrounding_user or "").strip(),
                    r.assistant_text or "",
                )
                for r in sampled
            ]
            sh = samples_hash(samples)
            key = cache_key(
                resolved_model_id, backend, base_url_str,
                sm, canon, sh,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            # mask_kaomoji-equivalent: the masked text is "[FACE] "
            # + assistant minus the leading kaomoji (already stripped
            # in assistant_text by the journal contract). Length is
            # close enough for estimation.
            masked_samples = [
                (user, "[FACE] " + assistant)
                for user, assistant in samples
            ]
            input_chars += (
                len(SYNTHESIZE_PROMPT)
                + len(_format_samples(masked_samples))
            )

    unique_calls = len(seen_keys)
    output_chars = unique_calls * ESTIMATE_OUTPUT_CHARS

    input_tokens = input_chars // CHARS_PER_TOKEN_HEURISTIC
    output_tokens = output_chars // CHARS_PER_TOKEN_HEURISTIC

    rates = BACKEND_RATES_USD_PER_1M_TOKENS.get(backend)
    if rates:
        cost: float | None = (
            (input_tokens / 1_000_000) * rates["input"]
            + (output_tokens / 1_000_000) * rates["output"]
        )
    else:
        cost = None

    n_unique_canon = len({
        canon for per_canon in buckets.values() for canon in per_canon
    })

    return AnalyzePlan(
        total_rows=len(rows_list),
        canonical_unique=n_unique_canon,
        providers_seen=providers_seen,
        model_counts=model_counts,
        counts_by_cell=counts_by_cell,
        cell_count=cell_count,
        unique_calls=unique_calls,
        estimated_input_tokens=int(input_tokens),
        estimated_output_tokens=int(output_tokens),
        estimated_cost_usd=cost,
        backend=backend,
        model_id=resolved_model_id,
    )


def run_analyze(
    rows: Iterable[ScrapeRow],
    *,
    notes: str = "",
    backend: str = "anthropic",
    base_url: str | None = None,
    model_id: str | None = None,
    concurrency: int | None = None,
    print_progress: bool = True,
) -> AnalyzeResult:
    """Top-level entry point.

    The synthesizer is constructed lazily via
    :func:`llmoji.synth.make_synthesizer` so a user without the
    chosen backend's SDK installed gets a clean ImportError pointing
    at the right ``pip install`` rather than an opaque attribute
    error mid-synthesis.
    """
    synth = make_synthesizer(backend, base_url=base_url, model_id=model_id)

    paths.ensure_home()
    bundle_dir = paths.bundle_dir()
    cache_path = paths.cache_per_cell_path()

    # Surface the legacy v1 cache one time so users who upgrade
    # know they can reclaim disk space. v2 never touches this file
    # — different keying, different shape — and ``cache clear``
    # wipes it alongside the v2 cache.
    legacy_path = paths.cache_per_instance_path()
    if print_progress and legacy_path.exists():
        print(
            f"  (note: legacy per_instance cache from pre-v2 detected at "
            f"{legacy_path}; safe to delete via `llmoji cache clear`)"
        )

    rows_list = list(rows)
    buckets, providers_seen, model_counts = _bucket_by_source_model_and_canonical(
        rows_list,
    )
    # counts_by_cell[source_model][canonical] = total rows in that cell
    # (used for the ``count`` column in each per-source-model
    # ``<slug>.jsonl``).
    counts_by_cell: dict[str, dict[str, int]] = {
        sm: {canon: len(rs) for canon, rs in per_canon.items()}
        for sm, per_canon in buckets.items()
    }
    n_unique_canon = len({
        canon for per_canon in buckets.values() for canon in per_canon
    })

    if print_progress:
        print(
            f"analyze: {len(rows_list)} rows / {n_unique_canon} canonical "
            f"kaomoji across {len(buckets)} source model(s) "
            f"({len(providers_seen)} provider(s): "
            f"{', '.join(providers_seen) or '(none)'}); "
            f"backend={synth.backend} model={synth.model_id}"
        )

    synthesized_by_cell, n_calls, n_cached = _synthesize_cells(
        synth, buckets, cache_path=cache_path,
        print_progress=print_progress, max_workers=concurrency,
    )

    # Lazy import — upload is the only place that touches the .salt file,
    # but we want the submitter id stamped into the manifest so the
    # bundle the user inspects matches what would land on HF.
    from .upload import submitter_id as _submitter_id
    _write_bundle(
        bundle_dir,
        counts_by_cell=counts_by_cell,
        synthesized_by_cell=synthesized_by_cell,
        providers_seen=providers_seen,
        model_counts=model_counts,
        submitter_id=_submitter_id(),
        synth_backend=synth.backend,
        synth_model_id=synth.model_id,
        notes=notes,
    )
    if print_progress:
        _print_preview(
            bundle_dir,
            counts_by_cell=counts_by_cell,
            synthesized_by_cell=synthesized_by_cell,
        )

    return AnalyzeResult(
        total_rows=len(rows_list),
        canonical_unique=n_unique_canon,
        providers_seen=providers_seen,
        bundle_dir=bundle_dir,
        calls_made=n_calls,
        calls_cached=n_cached,
    )
