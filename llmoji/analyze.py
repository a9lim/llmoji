"""Journals to bundle.

1. Bucket every row by ``(source model, canonical face)``. The source
   model is the row's model id, or its journal's name when the row has
   none.
2. Per cell, sample up to :data:`INSTANCE_SAMPLE_CAP` rows
   deterministically, mask the face, and make one structured call that
   answers with words from the locked lexicon. Cached by content hash.
3. Write ``manifest.json`` and one ``<source-model>.jsonl`` per model to
   ``~/.llmoji/bundle/`` — the files the user reads before ``upload``.

Embeddings, projections and figures are research-side, in llmoji-study.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from . import paths
from ._util import atomic_write_text, flatten_synthesis, package_version, sanitize_model_id_for_path
from .scrape import ScrapeRow
from .synth import (
    SynthesizerError,
    append_cache,
    cache_key,
    load_cache,
    make_synthesizer,
    mask_kaomoji,
    samples_hash,
    validate_synthesis,
)
from .synth_prompts import (
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    LEXICON_VERSION,
    SYNTHESIS_SCHEMA,
    SYNTHESIZE_PROMPT,
)
from .taxonomy import canonicalize_kaomoji

# Rows sampled per cell. Eriskii's original Claude-faces work used 4;
# matching it keeps the corpora comparable.
INSTANCE_SAMPLE_CAP = 4
INSTANCE_SAMPLE_SEED = 0

# One worker by default: an organisation's Haiku rate cap trips at two
# on a multi-hundred-cell run. ``--concurrency`` or $LLMOJI_CONCURRENCY.
DEFAULT_CONCURRENCY = 1

# Dry-run token estimate: about four characters per token for English,
# and a structured reply of about 120 characters.
_CHARS_PER_TOKEN = 4
_REPLY_CHARS = 120

Buckets = dict[str, dict[str, list[ScrapeRow]]]


class AnalyzeError(RuntimeError):
    """Some synthesis calls failed. Cells that succeeded are cached, so
    a re-run pays only for the rest."""


@dataclass
class AnalyzeResult:
    total_rows: int
    canonical_unique: int
    providers_seen: list[str]
    bundle_dir: Path
    calls_made: int
    calls_cached: int


@dataclass
class AnalyzePlan:
    total_rows: int
    canonical_unique: int
    providers_seen: list[str]
    model_counts: dict[str, int]
    cell_count: int
    cached: int
    estimated_input_tokens: int  # cold-cache, approximate
    estimated_output_tokens: int
    backend: str
    model_id: str


@dataclass
class _Cell:
    source_model: str
    canonical: str
    key: str
    prompt: str


def _bucket(rows: Iterable[ScrapeRow]) -> tuple[Buckets, list[str], dict[str, int]]:
    """``(buckets, providers_seen, model_counts)``. ``model_counts``
    counts every row, faces that canonicalize to nothing included."""
    buckets: dict[str, dict[str, list[ScrapeRow]]] = defaultdict(lambda: defaultdict(list))
    providers: set[str] = set()
    model_counts: Counter[str] = Counter()
    for r in rows:
        providers.add(r.source)
        source_model = (r.model or "").strip() or r.source
        model_counts[source_model] += 1
        canon = canonicalize_kaomoji(r.first_word)
        if canon:
            buckets[source_model][canon].append(r)
    return (
        {sm: dict(cells) for sm, cells in buckets.items()},
        sorted(providers),
        dict(model_counts),
    )


def _check_slugs(buckets: Buckets) -> None:
    """Two source models whose ids sanitize to one filename would
    overwrite each other's bundle file; refuse before paying for calls."""
    slugs: dict[str, set[str]] = defaultdict(set)
    for sm in buckets:
        slugs[sanitize_model_id_for_path(sm)].add(sm)
    clashes = {s: sorted(o) for s, o in slugs.items() if len(o) > 1}
    if clashes:
        raise ValueError(f"source models collide on bundle filenames: {clashes}")


def _format_samples(samples: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"[Sample {i}]\nUser: {user}\nAssistant: {assistant}" if user
        else f"[Sample {i}]\nAssistant: {assistant}"
        for i, (user, assistant) in enumerate(samples, 1)
    )


def _cells(buckets: Buckets, *, model_id: str, backend: str, base_url: str) -> Iterator[_Cell]:
    """Every cell in sorted order with its cache key and prompt. The
    sample is drawn with a per-cell seed, so a re-run sees the same rows
    and the bundle comes out byte-identical."""
    for sm in sorted(buckets):
        for canon in sorted(buckets[sm]):
            rows = buckets[sm][canon]
            if len(rows) > INSTANCE_SAMPLE_CAP:
                rng = random.Random(f"{INSTANCE_SAMPLE_SEED}:{sm}:{canon}")
                rows = rng.sample(rows, INSTANCE_SAMPLE_CAP)
            samples = [(r.surrounding_user.strip(), r.assistant_text) for r in rows]
            yield _Cell(
                source_model=sm,
                canonical=canon,
                key=cache_key(model_id, backend, base_url, sm, canon, samples_hash(samples)),
                prompt=SYNTHESIZE_PROMPT.format(samples=_format_samples(
                    [(user, mask_kaomoji(body)) for user, body in samples]
                )),
            )


def _cached(cache: dict[str, dict[str, Any]], key: str) -> dict[str, list[str]] | None:
    try:
        return validate_synthesis(cache[key]["synthesis"])
    except (KeyError, TypeError, SynthesizerError):
        return None


def _workers(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    try:
        return max(1, int(os.environ.get("LLMOJI_CONCURRENCY", DEFAULT_CONCURRENCY)))
    except ValueError:
        return DEFAULT_CONCURRENCY


def plan_analyze(
    rows: Iterable[ScrapeRow],
    *,
    backend: str = "anthropic",
    base_url: str | None = None,
    model_id: str | None = None,
) -> AnalyzePlan:
    """What :func:`run_analyze` would do, without touching a backend."""
    rows = list(rows)
    buckets, providers, model_counts = _bucket(rows)
    # The hosted backends always use their pinned model, as make_synthesizer does.
    model = {
        "anthropic": DEFAULT_ANTHROPIC_MODEL_ID, "openai": DEFAULT_OPENAI_MODEL_ID,
    }.get(backend) or model_id or "(local)"
    cache = load_cache(paths.cache_path())
    cells = list(_cells(buckets, model_id=model, backend=backend, base_url=base_url or ""))
    cold = [c for c in cells if _cached(cache, c.key) is None]
    return AnalyzePlan(
        total_rows=len(rows),
        canonical_unique=len({c.canonical for c in cells}),
        providers_seen=providers,
        model_counts=model_counts,
        cell_count=len(cells),
        cached=len(cells) - len(cold),
        estimated_input_tokens=sum(len(c.prompt) for c in cold) // _CHARS_PER_TOKEN,
        estimated_output_tokens=len(cold) * _REPLY_CHARS // _CHARS_PER_TOKEN,
        backend=backend,
        model_id=model,
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
    synth = make_synthesizer(backend, base_url=base_url, model_id=model_id)
    paths.ensure_home()
    rows = list(rows)
    buckets, providers, model_counts = _bucket(rows)
    _check_slugs(buckets)
    cache_path = paths.cache_path()
    cache = load_cache(cache_path)
    cells = list(_cells(
        buckets, model_id=synth.model_id, backend=synth.backend, base_url=synth.base_url,
    ))
    result: dict[str, dict[str, list[str]]] = {}
    pending: list[_Cell] = []
    for cell in cells:
        hit = _cached(cache, cell.key)
        if hit is None:
            pending.append(cell)
        else:
            result[cell.key] = hit
    if print_progress:
        print(
            f"analyze: {len(rows)} rows, {len(cells)} cells across "
            f"{len(buckets)} source model(s); {len(cells) - len(pending)} cached, "
            f"{len(pending)} to synthesize with {synth.backend}/{synth.model_id}"
        )

    # Ctrl-C isn't caught: the cache already holds every finished cell.
    errors: list[Exception] = []
    if pending:
        with ThreadPoolExecutor(max_workers=_workers(concurrency)) as pool:
            futures = {
                pool.submit(synth.call_structured, c.prompt, schema=SYNTHESIS_SCHEMA): c
                for c in pending
            }
            done: Iterable[Any] = as_completed(futures)
            if print_progress:
                done = tqdm(done, total=len(futures), desc="synthesize", unit="cell")
            for fut in done:
                cell = futures[fut]
                try:
                    synthesis = fut.result()
                except Exception as e:  # noqa: BLE001 — collected, reraised below
                    errors.append(e)
                    continue
                result[cell.key] = synthesis
                append_cache(cache_path, {
                    "key": cell.key,
                    "kaomoji": cell.canonical,
                    "source_model": cell.source_model,
                    "synthesis": synthesis,
                    "model": synth.model_id,
                    "backend": synth.backend,
                })
    if errors:
        raise AnalyzeError(
            f"{len(errors)} of {len(pending)} synthesis call(s) failed; the rest "
            f"are cached, so re-run `llmoji analyze` to resume. "
            f"First failure: {errors[0]!r}"
        ) from errors[0]

    from .upload import submitter_id

    bundle_dir = paths.bundle_dir()
    _write_bundle(
        bundle_dir,
        cells=cells,
        buckets=buckets,
        synthesis=result,
        manifest={
            "llmoji_version": package_version(),
            "lexicon_version": LEXICON_VERSION,
            "synthesis_model_id": synth.model_id,
            "synthesis_backend": synth.backend,
            "submitter_id": submitter_id(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "providers_seen": providers,
            "model_counts": model_counts,
            "total_synthesized_rows": len(cells),
            "notes": notes,
        },
    )
    if print_progress:
        _print_preview(bundle_dir, cells=cells, buckets=buckets, synthesis=result)
    return AnalyzeResult(
        total_rows=len(rows),
        canonical_unique=len({c.canonical for c in cells}),
        providers_seen=providers,
        bundle_dir=bundle_dir,
        calls_made=len(pending),
        calls_cached=len(cells) - len(pending),
    )


def _write_bundle(
    bundle_dir: Path,
    *,
    cells: list[_Cell],
    buckets: Buckets,
    synthesis: dict[str, dict[str, list[str]]],
    manifest: dict[str, Any],
) -> None:
    """Replace the bundle directory's contents. Everything in it is
    removed first — the bundle is the privacy boundary, and a stale file
    would ship with the next upload. Symlinks are unlinked, never
    followed."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for p in bundle_dir.iterdir():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    atomic_write_text(
        bundle_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    by_model: dict[str, list[str]] = defaultdict(list)
    for c in cells:
        by_model[c.source_model].append(json.dumps({
            "kaomoji": c.canonical,
            "count": len(buckets[c.source_model][c.canonical]),
            "synthesis": synthesis[c.key],
        }, ensure_ascii=False))
    for sm, lines in by_model.items():
        (bundle_dir / f"{sanitize_model_id_for_path(sm)}.jsonl").write_text(
            "\n".join(lines) + "\n",
        )


def _print_preview(
    bundle_dir: Path,
    *,
    cells: list[_Cell],
    buckets: Buckets,
    synthesis: dict[str, dict[str, list[str]]],
) -> None:
    print(f"\n--- bundle preview: {bundle_dir} ---")
    by_model: dict[str, list[_Cell]] = defaultdict(list)
    for c in cells:
        by_model[c.source_model].append(c)
    for sm in sorted(by_model):
        print(f"  [{sm}] -> {sanitize_model_id_for_path(sm)}.jsonl ({len(by_model[sm])} faces)")
        for c in sorted(by_model[sm], key=lambda c: -len(buckets[sm][c.canonical])):
            n = len(buckets[sm][c.canonical])
            print(f"    n={n:>4}  {c.canonical}  {flatten_synthesis(synthesis[c.key])}")
    print("--- end preview; read each <model>.jsonl before `llmoji upload` ---\n")
