"""Two-stage analysis pipeline for the bundle.

End-user pipeline (no GPU, no embedding, no axes):

  1. Iterate every installed provider's journal + any user-parsed
     static dumps; canonicalize each leading kaomoji via
     :func:`llmoji.taxonomy.canonicalize_kaomoji`; bucket by
     canonical form.
  2. Stage A (per-instance description): for each row, run
     :func:`llmoji.haiku.mask_kaomoji` and call Haiku via
     :func:`llmoji.haiku.call_haiku` with
     :data:`llmoji.haiku_prompts.DESCRIBE_PROMPT_*`. Cache by
     content-hash so re-runs of ``analyze`` skip rows already
     described — both for cost and so unchanged rows produce
     identical bundles.
  3. Stage B (per-canonical-kaomoji synthesis): pool Stage A
     descriptions for one canonical face; call
     :func:`llmoji.haiku.synthesize_descriptions` with
     :data:`llmoji.haiku_prompts.SYNTHESIZE_PROMPT`.
  4. Emit a manifest + ``descriptions.jsonl`` to
     ``~/.llmoji/bundle/`` (the loose-files inspection gap the user
     reads before deciding to ``upload``).

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
from ._util import package_version
from .haiku import (
    append_cache,
    cache_key,
    call_haiku,
    load_cache,
    mask_kaomoji,
    synthesize_descriptions,
)
from .haiku_prompts import (
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    HAIKU_MODEL_ID,
    SYNTHESIZE_PROMPT,
)
from .scrape import ScrapeRow
from .taxonomy import canonicalize_kaomoji

# Stage-A sample cap per canonical kaomoji. Eriskii used 4 for the
# original Claude-faces work; same number here keeps cross-corpus
# comparison apples-to-apples. Forms with fewer instances are
# fully sampled.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# Stage-A Haiku-call concurrency. The Anthropic SDK's underlying
# httpx.Client is thread-safe, and a content-hash cache append-write
# is POSIX-atomic for sub-PIPE_BUF (4 KB) JSONL lines, so a small
# thread pool gives ~Nx wallclock speedup on cache misses with no
# coordination beyond per-future result handling on the main thread.
# Override via $LLMOJI_CONCURRENCY (>=1).
DEFAULT_STAGE_A_CONCURRENCY = 4


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


def _bucket_by_canonical(
    rows: Iterable[ScrapeRow],
) -> tuple[dict[str, list[ScrapeRow]], list[str], dict[str, int]]:
    """Group rows by canonical kaomoji.

    Returns (buckets, providers_seen_sorted, journal_counts_by_source).
    """
    buckets: dict[str, list[ScrapeRow]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for r in rows:
        counts[r.source] += 1
        canon = canonicalize_kaomoji(r.first_word)
        if not canon:
            continue
        buckets[canon].append(r)
    return dict(buckets), sorted(counts), dict(counts)


def _sample(
    rows: list[ScrapeRow],
    *,
    cap: int,
    seed_label: str,
) -> list[ScrapeRow]:
    """Deterministic uniform sampling per canonical bucket. Sort
    upstream by some stable key so re-runs hit the same instances."""
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
    stage: str, canon: str, n_descs: int | None, dt: float, text: str,
) -> None:
    """Shared formatter for Stage-A and Stage-B per-call progress lines."""
    short = text[:70] + ("..." if len(text) > 70 else "")
    suffix = f" (n={n_descs})" if n_descs is not None else ""
    print(f"  stage-{stage} {canon}{suffix}  ({dt:.1f}s)  {short}")


def _stage_a(
    client: Any,
    buckets: dict[str, list[ScrapeRow]],
    *,
    cache_path: Path,
    print_progress: bool = True,
    max_workers: int | None = None,
) -> tuple[dict[str, list[str]], int, int]:
    """For each canonical bucket, sample ≤ INSTANCE_SAMPLE_CAP rows;
    Haiku-describe each (with cache hit skipping the API call);
    return ``(descriptions_by_canonical, n_calls, n_cached)``.

    Cache-miss Haiku calls run on a small thread pool
    (``max_workers`` or ``$LLMOJI_CONCURRENCY``, default 4). The
    Anthropic client is thread-safe; cache appends happen serially
    on the main thread as futures complete, so there's no append
    interleaving to worry about.
    """
    cache = load_cache(cache_path)
    descs_by_canon: dict[str, list[str]] = defaultdict(list)
    n_cached = 0
    pending: list[tuple[str, str, str]] = []  # (canon, key, prompt)

    # Pass 1: walk every sampled row, satisfy from cache where
    # possible, queue misses for parallel dispatch.
    for canon in sorted(buckets):
        sampled = _sample(buckets[canon], cap=INSTANCE_SAMPLE_CAP, seed_label=canon)
        for r in sampled:
            user_text = (r.surrounding_user or "").strip()
            assistant = r.assistant_text or ""
            key = cache_key(canon, user_text, assistant)
            hit = cache.get(key)
            if hit and "description" in hit:
                descs_by_canon[canon].append(hit["description"])
                n_cached += 1
                continue
            masked = mask_kaomoji(assistant, r.first_word)
            if user_text:
                prompt = DESCRIBE_PROMPT_WITH_USER.format(
                    user_text=user_text, masked_text=masked,
                )
            else:
                prompt = DESCRIBE_PROMPT_NO_USER.format(masked_text=masked)
            pending.append((canon, key, prompt))

    if not pending:
        return dict(descs_by_canon), 0, n_cached

    workers = _resolve_concurrency(max_workers)

    def _describe_one(canon: str, key: str, prompt: str) -> tuple[str, str, str, float]:
        t0 = time.monotonic()
        description = call_haiku(client, prompt, model_id=HAIKU_MODEL_ID)
        dt = time.monotonic() - t0 if print_progress else 0.0
        return canon, key, description, dt

    n_calls = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_describe_one, c, k, p) for c, k, p in pending]
        for fut in as_completed(futures):
            canon, key, description, dt = fut.result()
            row = {
                "key": key,
                "kaomoji": canon,
                "description": description,
                "model": HAIKU_MODEL_ID,
            }
            append_cache(cache_path, row)
            cache[key] = row
            descs_by_canon[canon].append(description)
            n_calls += 1
            if print_progress:
                _print_stage_progress("A", canon, None, dt, description)

    return dict(descs_by_canon), n_calls, n_cached


def _stage_b(
    client: Any,
    descs_by_canon: dict[str, list[str]],
    *,
    print_progress: bool = True,
) -> tuple[dict[str, str], int]:
    """Per canonical kaomoji, pool descriptions, synthesize a single
    1-2-sentence meaning. Returns ``(synthesized_by_canonical,
    n_calls)``.
    """
    out: dict[str, str] = {}
    n_calls = 0
    for canon in sorted(descs_by_canon):
        descs = descs_by_canon[canon]
        if not descs:
            continue
        t0 = time.monotonic()
        synth = synthesize_descriptions(
            client, descs,
            model_id=HAIKU_MODEL_ID,
            synth_prompt_template=SYNTHESIZE_PROMPT,
        )
        out[canon] = synth
        n_calls += 1
        if print_progress:
            _print_stage_progress(
                "B", canon, len(descs), time.monotonic() - t0, synth,
            )
    return out, n_calls


def _write_bundle(
    bundle_dir: Path,
    *,
    counts_by_canon: dict[str, int],
    synthesized_by_canon: dict[str, str],
    providers_seen: list[str],
    journal_counts: dict[str, int],
    submitter_id: str,
    notes: str,
) -> None:
    """Write manifest.json + descriptions.jsonl. Loose-files layout
    so the user can ``cat`` and review before ``upload``.

    Clears any prior contents of ``bundle_dir`` (loose files only —
    we don't recurse into subdirs in case the user has stashed
    notes there) before writing. The bundle is the privacy
    boundary; stale leftover files MUST NOT survive an `analyze`
    that's about to feed `upload`. The two-file schema is the v1.0
    frozen surface (see :data:`llmoji.upload.BUNDLE_ALLOWLIST`).
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Clear loose files in the bundle dir before writing — guarantees
    # the post-`analyze` state is exactly the two-file schema with no
    # stragglers from prior runs or out-of-band user edits. Subdirs
    # (if any) are left alone; the upload allowlist refuses to ship
    # them.
    for p in bundle_dir.iterdir():
        if p.is_file():
            p.unlink()

    manifest = {
        "llmoji_version": package_version(),
        "haiku_model_id": HAIKU_MODEL_ID,
        "submitter_id": submitter_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "providers_seen": providers_seen,
        "journal_counts": journal_counts,
        "total_rows_scraped": int(sum(counts_by_canon.values())),
        "total_kaomoji_unique_canonical": len(counts_by_canon),
        "notes": notes,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    descriptions_path = bundle_dir / "descriptions.jsonl"
    with descriptions_path.open("w") as f:
        for canon in sorted(synthesized_by_canon):
            row = {
                "kaomoji": canon,
                "count": int(counts_by_canon.get(canon, 0)),
                "haiku_synthesis_description": synthesized_by_canon[canon],
                "llmoji_version": manifest["llmoji_version"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_preview(
    bundle_dir: Path,
    *,
    counts_by_canon: dict[str, int],
    synthesized_by_canon: dict[str, str],
) -> None:
    """Print the per-face count + first ~80 chars of synthesis for
    every canonical kaomoji that's about to be shippable. The
    inspection gap depends on this — the user sees what they're
    about to publish before deciding to upload.
    """
    print("\n--- bundle preview ---")
    print(f"location: {bundle_dir}")
    print(f"per-face rows in descriptions.jsonl ({len(synthesized_by_canon)} faces):\n")
    for canon in sorted(synthesized_by_canon, key=lambda c: -counts_by_canon.get(c, 0)):
        n = counts_by_canon.get(canon, 0)
        synth = synthesized_by_canon[canon]
        short = synth[:80].replace("\n", " ")
        print(f"  n={n:>4}  {canon}  {short}")
    print("\n--- end preview ---")
    print("review descriptions.jsonl before `llmoji upload`.\n")


def run_analyze(
    rows: Iterable[ScrapeRow],
    *,
    notes: str = "",
    print_progress: bool = True,
) -> AnalyzeResult:
    """Top-level entry point. Lazy-imports anthropic so the rest of
    the package stays importable without an API key set."""
    import anthropic
    # max_retries=8 (SDK default is 2). A full re-analyze on a
    # multi-hundred-row corpus can sustain a 429 wave for tens of
    # seconds when the org-level Haiku 50 req/min cap kicks in;
    # default retries can't ride that out and the run aborts mid-
    # Stage-A. The SDK honors the response's Retry-After header
    # and uses exponential backoff between retries — bumping the
    # cap is the right knob.
    client = anthropic.Anthropic(max_retries=8)

    paths.ensure_home()
    bundle_dir = paths.bundle_dir()
    cache_path = paths.cache_per_instance_path()

    rows_list = list(rows)
    buckets, providers_seen, journal_counts = _bucket_by_canonical(rows_list)
    counts_by_canon = {canon: len(rs) for canon, rs in buckets.items()}

    if print_progress:
        print(
            f"analyze: {len(rows_list)} rows / {len(buckets)} canonical kaomoji "
            f"across {len(providers_seen)} sources "
            f"({', '.join(providers_seen) or '(none)'})"
        )

    descs_by_canon, n_a, n_cached = _stage_a(
        client, buckets, cache_path=cache_path, print_progress=print_progress,
    )
    synthesized_by_canon, n_b = _stage_b(
        client, descs_by_canon, print_progress=print_progress,
    )

    # Lazy import — upload is the only place that touches state.json,
    # but we want the submitter id stamped into the manifest so the
    # bundle the user inspects matches what would land on HF.
    from .upload import submitter_id as _submitter_id
    _write_bundle(
        bundle_dir,
        counts_by_canon=counts_by_canon,
        synthesized_by_canon=synthesized_by_canon,
        providers_seen=providers_seen,
        journal_counts=journal_counts,
        submitter_id=_submitter_id(),
        notes=notes,
    )
    if print_progress:
        _print_preview(
            bundle_dir,
            counts_by_canon=counts_by_canon,
            synthesized_by_canon=synthesized_by_canon,
        )

    return AnalyzeResult(
        total_rows=len(rows_list),
        canonical_unique=len(buckets),
        providers_seen=providers_seen,
        bundle_dir=bundle_dir,
        stage_a_calls_made=n_a,
        stage_a_calls_cached=n_cached,
        stage_b_calls_made=n_b,
    )
