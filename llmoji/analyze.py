"""Two-stage analysis pipeline for the bundle.

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
  2. Stage A (per-instance description): for each row, run
     :func:`llmoji.synth.mask_kaomoji` and call the chosen backend
     via the :class:`~llmoji.synth.Synthesizer` instance with
     :data:`llmoji.synth_prompts.DESCRIBE_PROMPT_*`. Cache by
     content-hash + synth model id + backend so re-runs of
     ``analyze`` skip rows already described — for cost, and so a
     re-run feeds Stage B identical descriptions in the same order
     regardless of cache state. (Synthesizer prose itself is
     model-dependent and may not be byte-stable on a fresh call,
     but the cache pins it for any given input.)
  3. Stage B (per-cell synthesis): pool Stage A descriptions for
     each ``(source_model, canonical_kaomoji)`` cell; call
     :func:`llmoji.synth.synthesize_descriptions` with
     :data:`llmoji.synth_prompts.SYNTHESIZE_PROMPT`.
  4. Emit a manifest + one ``<sanitized_source_model>.jsonl``
     per source model at the top of ``~/.llmoji/bundle/`` — the
     loose-files inspection gap the user reads before deciding to
     ``upload``.

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
    synthesize_descriptions,
)
from .synth_prompts import (
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    SYNTHESIZE_PROMPT,
)
from .taxonomy import canonicalize_kaomoji

# Stage-A sample cap per ``(source_model, canonical_kaomoji)`` cell.
# ``_sample`` returns ``min(cap, len(rows))`` per cell — popular
# faces get capped, rare faces fully sampled. Eriskii used 4 for the
# original Claude-faces work; same number here keeps cross-corpus
# comparison apples-to-apples.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# Stage-A call concurrency. The Anthropic / OpenAI httpx clients
# are thread-safe, and a content-hash cache append-write is POSIX-
# atomic for sub-PIPE_BUF (4 KB) JSONL lines, so a small thread pool
# gives ~Nx wallclock speedup on cache misses with no coordination
# beyond per-future result handling on the main thread.
# Override via $LLMOJI_CONCURRENCY (>=1).
#
# Default 2 keeps us well under the 50 req/min Haiku org cap once
# retries kick in. Bumping past 2 starts to trip the cap mid-run on
# corpora with hundreds of cache misses; the SDK's exponential
# backoff (max_retries=8) recovers but burns wallclock. Set
# ``LLMOJI_CONCURRENCY=4+`` if you have a higher rate limit tier.
DEFAULT_STAGE_A_CONCURRENCY = 2


@dataclass
class AnalyzeResult:
    """Summary stats reported back to the CLI / printed to the user."""

    total_rows: int
    canonical_unique: int
    providers_seen: list[str]
    bundle_dir: Path
    stage_a_calls_made: int
    stage_a_calls_cached: int
    stage_b_calls_made: int


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
    """Stage-A worker count: explicit arg → ``$LLMOJI_CONCURRENCY``
    → :data:`DEFAULT_STAGE_A_CONCURRENCY`. Clamps to ``>=1``. Bad
    env values fall back silently to the default."""
    if explicit is not None:
        return max(1, explicit)
    raw = os.environ.get("LLMOJI_CONCURRENCY")
    if raw is None:
        return DEFAULT_STAGE_A_CONCURRENCY
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_STAGE_A_CONCURRENCY


def _print_stage_progress(
    stage: str,
    label: str,
    n_descs: int | None,
    dt: float,
    text: str,
) -> None:
    """Shared formatter for Stage-A and Stage-B per-call progress lines."""
    short = text[:70] + ("..." if len(text) > 70 else "")
    suffix = f" (n={n_descs})" if n_descs is not None else ""
    print(f"  stage-{stage} {label}{suffix}  ({dt:.1f}s)  {short}")


# ---------------------------------------------------------------------------
# Stage A — per-instance descriptions
# ---------------------------------------------------------------------------


def _stage_a(
    synth: Synthesizer,
    buckets: dict[str, dict[str, list[ScrapeRow]]],
    *,
    cache_path: Path,
    print_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[dict[str, dict[str, list[str]]], int, int]:
    """For each ``(source_model, canonical)`` cell, sample ≤
    INSTANCE_SAMPLE_CAP rows; describe each (cache hit skips the
    API call); return ``(descs_by_cell, n_calls, n_cached)``.

    ``descs_by_cell[source_model][canonical]`` is the list of Stage A
    outputs for the cell — one entry per sampled row.

    Cache-miss API calls run on a small thread pool (``max_workers``
    or ``$LLMOJI_CONCURRENCY``, default 2). Both SDK clients are
    thread-safe; cache appends happen serially on the main thread
    after all dispatched futures complete, in deterministic walk
    order, so the bundle and the cache file are identical regardless
    of the order futures finish in.
    """
    cache = load_cache(cache_path)
    n_cached = 0

    # Walk every sampled row in deterministic (sorted source_model,
    # sorted canonical, _sample-stable) order. Each entry records
    # whether it was a cache hit (carries ``description`` directly)
    # or a miss (carries ``prompt`` to be dispatched). After dispatch
    # the misses are populated with their ``description`` in place.
    walk: list[dict[str, Any]] = []
    for source_model in sorted(buckets):
        per_canon = buckets[source_model]
        for canon in sorted(per_canon):
            sampled = _sample(
                per_canon[canon],
                cap=INSTANCE_SAMPLE_CAP,
                seed_label=f"{source_model}:{canon}",
            )
            for r in sampled:
                user_text = (r.surrounding_user or "").strip()
                assistant = r.assistant_text or ""
                key = cache_key(
                    synth.model_id, synth.backend, synth.base_url,
                    canon, user_text, assistant,
                )
                hit = cache.get(key)
                if hit and "description" in hit:
                    walk.append({
                        "sm": source_model, "canon": canon, "key": key,
                        "cached": True, "description": hit["description"],
                    })
                    n_cached += 1
                    continue
                masked = mask_kaomoji(assistant, r.first_word)
                if user_text:
                    prompt = DESCRIBE_PROMPT_WITH_USER.format(
                        user_text=user_text, masked_text=masked,
                    )
                else:
                    prompt = DESCRIBE_PROMPT_NO_USER.format(masked_text=masked)
                walk.append({
                    "sm": source_model, "canon": canon, "key": key,
                    "cached": False, "prompt": prompt, "description": None,
                })

    # Group cache misses by key. Two sampled rows can share a key
    # (same canonical + user_text + assistant_text — common when one
    # turn's assistant text gets sampled into multiple cells, or when
    # the journal carries near-duplicate rows). Without this dedupe,
    # a cold-cache run would dispatch each duplicate separately and
    # potentially get different descriptions, while a warm-cache
    # follow-up would read the (single) last-write-wins cache row
    # for all duplicates — cold and warm would feed Stage B
    # different lists. One dispatch per unique key keeps cold and
    # warm in lockstep.
    pending_by_key: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(walk):
        if not e["cached"]:
            pending_by_key[e["key"]].append(i)

    if pending_by_key:
        workers = _resolve_concurrency(max_workers)

        def _describe_one(prompt: str) -> tuple[str, float]:
            t0 = time.monotonic()
            description = synth.call(prompt)
            dt = time.monotonic() - t0 if print_progress else 0.0
            return description, dt

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_key = {
                pool.submit(_describe_one, walk[indices[0]]["prompt"]): key
                for key, indices in pending_by_key.items()
            }
            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                description, dt = fut.result()
                indices = pending_by_key[key]
                for i in indices:
                    walk[i]["description"] = description
                if print_progress:
                    first = walk[indices[0]]
                    _print_stage_progress(
                        "A", f"{first['sm']}/{first['canon']}",
                        None, dt, description,
                    )

    # Serialize: assemble descs_by_cell + append cache in deterministic
    # walk order. Order matters for Stage B because SYNTHESIZE_PROMPT
    # numbers the descriptions; if Stage B sees the same list in the
    # same order across runs, it produces the same prose. The cache
    # is written once per unique key — duplicate walk entries share
    # the cached row so a warm-cache rerun resolves all duplicates
    # to the same description.
    descs_by_cell: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_calls = 0
    written_keys: set[str] = set()
    for entry in walk:
        sm = entry["sm"]
        canon = entry["canon"]
        description = entry["description"]
        descs_by_cell[sm][canon].append(description)
        key = entry["key"]
        if not entry["cached"] and key not in written_keys:
            row = {
                "key": key,
                "kaomoji": canon,
                "description": description,
                "model": synth.model_id,
                "backend": synth.backend,
            }
            append_cache(cache_path, row)
            cache[key] = row
            written_keys.add(key)
            n_calls += 1

    return _freeze_two_level(descs_by_cell), n_calls, n_cached


def _freeze_two_level(
    d: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {sm: dict(per_canon) for sm, per_canon in d.items()}


# ---------------------------------------------------------------------------
# Stage B — per-cell syntheses
# ---------------------------------------------------------------------------


def _stage_b(
    synth: Synthesizer,
    descs_by_cell: dict[str, dict[str, list[str]]],
    *,
    print_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[dict[str, dict[str, str]], int]:
    """Per ``(source_model, canonical)`` cell, pool descriptions and
    synthesize a single 1-2-sentence meaning. Returns
    ``(synthesized_by_cell, n_calls)``.

    Synthesis calls dispatch on the same thread pool Stage A uses
    (``max_workers`` or ``$LLMOJI_CONCURRENCY``, default 2). Each
    cell's synthesis is independent — no shared mutable state, no
    cache to serialize, so the dispatch is pure parallelism win.
    """
    pending: list[tuple[str, str, list[str]]] = []
    for sm in sorted(descs_by_cell):
        for canon in sorted(descs_by_cell[sm]):
            descs = descs_by_cell[sm][canon]
            if descs:
                pending.append((sm, canon, descs))
    if not pending:
        return {}, 0

    workers = _resolve_concurrency(max_workers)

    def _synth_one(
        sm: str, canon: str, descs: list[str],
    ) -> tuple[str, str, str, int, float]:
        t0 = time.monotonic()
        line = synthesize_descriptions(
            synth, descs,
            synth_prompt_template=SYNTHESIZE_PROMPT,
        )
        dt = time.monotonic() - t0 if print_progress else 0.0
        return sm, canon, line, len(descs), dt

    out: dict[str, dict[str, str]] = defaultdict(dict)
    n_calls = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_synth_one, sm, c, d) for sm, c, d in pending]
        for fut in as_completed(futures):
            sm, canon, line, n_descs, dt = fut.result()
            out[sm][canon] = line
            n_calls += 1
            if print_progress:
                _print_stage_progress(
                    "B", f"{sm}/{canon}", n_descs, dt, line,
                )
    return {sm: dict(per_canon) for sm, per_canon in out.items()}, n_calls


# ---------------------------------------------------------------------------
# Bundle write
# ---------------------------------------------------------------------------


def _clear_bundle_dir(bundle_dir: Path) -> None:
    """Remove every top-level file AND every top-level subdir from
    ``bundle_dir``. Stale per-source-model files from prior runs
    would silently leak into upload otherwise — the bundle is the
    privacy boundary, so we wipe everything that isn't about to be
    re-written. Subdirs from older 1.1.0 layouts are also cleared
    here on first analyze post-upgrade.
    """
    import shutil
    if not bundle_dir.exists():
        return
    for p in bundle_dir.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def _write_bundle(
    bundle_dir: Path,
    *,
    counts_by_cell: dict[str, dict[str, int]],
    synthesized_by_cell: dict[str, dict[str, str]],
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
    ``total_synthesized_rows`` counts rows across files, so a face
    appearing in 4 source-model files contributes 4.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _clear_bundle_dir(bundle_dir)

    total_synth_rows = sum(
        len(per_canon) for per_canon in synthesized_by_cell.values()
    )

    manifest = {
        "llmoji_version": package_version(),
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
    # other's descriptions.jsonl. Loud failure beats a half-shipped
    # bundle.
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
                    "synthesis_description": per_canon[canon],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_preview(
    bundle_dir: Path,
    *,
    counts_by_cell: dict[str, dict[str, int]],
    synthesized_by_cell: dict[str, dict[str, str]],
) -> None:
    """Print a per-source-model summary plus per-face count + first
    ~80 chars of synthesis. The inspection gap depends on this — the
    user sees what they're about to publish before deciding to
    upload.
    """
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
            short = per_canon[canon][:80].replace("\n", " ")
            print(f"    n={n:>4}  {canon}  {short}")
    print("\n--- end preview ---")
    print("review each <model>.jsonl before `llmoji upload`.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_analyze(
    rows: Iterable[ScrapeRow],
    *,
    notes: str = "",
    backend: str = "anthropic",
    base_url: str | None = None,
    model_id: str | None = None,
    print_progress: bool = True,
) -> AnalyzeResult:
    """Top-level entry point.

    The synthesizer is constructed lazily via
    :func:`llmoji.synth.make_synthesizer` so a user without the
    chosen backend's SDK installed gets a clean ImportError pointing
    at the right ``pip install`` rather than an opaque attribute
    error deep inside Stage A.
    """
    synth = make_synthesizer(backend, base_url=base_url, model_id=model_id)

    paths.ensure_home()
    bundle_dir = paths.bundle_dir()
    cache_path = paths.cache_per_instance_path()

    rows_list = list(rows)
    buckets, providers_seen, model_counts = _bucket_by_source_model_and_canonical(
        rows_list,
    )
    # counts_by_cell[source_model][canonical] = total rows in that cell
    # (used for the ``count`` column in descriptions.jsonl).
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

    descs_by_cell, n_a, n_cached = _stage_a(
        synth, buckets, cache_path=cache_path, print_progress=print_progress,
    )
    synthesized_by_cell, n_b = _stage_b(
        synth, descs_by_cell, print_progress=print_progress,
    )

    # Lazy import — upload is the only place that touches state.json,
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
        stage_a_calls_made=n_a,
        stage_a_calls_cached=n_cached,
        stage_b_calls_made=n_b,
    )
