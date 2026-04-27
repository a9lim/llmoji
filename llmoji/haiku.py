"""Thin wrapper around the Anthropic SDK for the two-stage Haiku
pipeline.

The actual prompt strings live in :mod:`llmoji.haiku_prompts`; this
module is just plumbing — masking, single-call helper, and a per-
instance content-hash cache so re-runs of ``llmoji analyze`` only
pay for new rows.

Cache layout (one JSONL line per cached call):

    {
      "key":          sha256(canonical_kaomoji + "\\0" + user + "\\0" + assistant)[:16],
      "kaomoji":      canonical kaomoji,
      "description":  Haiku output,
      "model":        Haiku model slug used,
    }

The cache file lives at ``~/.llmoji/cache/per_instance.jsonl`` by
default; the path is parameterized so tests can use a tmpdir.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MASK_TOKEN = "[FACE]"


def mask_kaomoji(text: str, first_word: str) -> str:
    """Prepend :data:`MASK_TOKEN` to a kaomoji-stripped assistant body.

    By v1.0 journal contract, ``assistant_text`` never includes the
    leading kaomoji — that's carried separately in the row's
    ``kaomoji`` field. Live-hook journals strip on write
    (``ltrimstr($kaomoji)`` in the bash template), the Claude.ai
    export reader strips on parse, and the generic-JSONL contract
    requires the same. So this function just prepends ``[FACE] ``
    to give Haiku the ``[FACE] <body>`` shape its DESCRIBE prompts
    promise.

    Empty ``first_word`` (no kaomoji on this row — shouldn't reach
    here in normal flow, but defensive) → pass through unchanged.
    """
    if not first_word:
        return text
    return MASK_TOKEN + " " + text.lstrip()


def call_haiku(
    client: Any,
    prompt: str,
    *,
    model_id: str,
    max_tokens: int = 200,
) -> str:
    """Single Haiku call with a pre-formatted prompt.

    Returns the assistant's first text-block content, stripped.
    Raises on API error (callers handle their own resume loops).
    ``client`` is an ``anthropic.Anthropic`` instance — we don't
    import the SDK here so importing this module doesn't require
    ``anthropic`` to be installed.
    """
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return (getattr(block, "text", "") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Per-instance content-hash cache
# ---------------------------------------------------------------------------


def cache_key(canonical_kaomoji: str, user_text: str, assistant_text: str) -> str:
    """Deterministic 16-hex-char content hash key.

    Truncated SHA-256 — collisions on a single user's corpus are
    astronomically unlikely (~2^32 entries before a ~50% collision
    probability against a 64-bit space). The cache is private to
    one machine; no security boundary depends on the hash.
    """
    h = hashlib.sha256()
    h.update(canonical_kaomoji.encode("utf-8"))
    h.update(b"\0")
    h.update((user_text or "").encode("utf-8"))
    h.update(b"\0")
    h.update((assistant_text or "").encode("utf-8"))
    return h.hexdigest()[:16]


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load the per-instance cache as ``{key: row}``. Empty / missing
    file → empty dict."""
    out: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return out
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = row.get("key")
            if isinstance(k, str):
                out[k] = row
    return out


def append_cache(cache_path: Path, row: dict[str, Any]) -> None:
    """Append one row to the cache. Caller manages the dict in
    memory; this is the disk-side persistence."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_size(cache_path: Path) -> tuple[int, int]:
    """Return ``(n_rows, n_bytes)`` for the cache file. Used by
    ``llmoji status`` so the user knows what's on disk."""
    if not cache_path.exists():
        return (0, 0)
    n_bytes = cache_path.stat().st_size
    n_rows = 0
    with cache_path.open() as f:
        for line in f:
            if line.strip():
                n_rows += 1
    return (n_rows, n_bytes)


def synthesize_descriptions(
    client: Any,
    descriptions: Iterable[str],
    *,
    model_id: str,
    synth_prompt_template: str,
    max_tokens: int = 200,
) -> str:
    """Stage B: pool per-instance descriptions for one canonical
    kaomoji, return Haiku's synthesized one-sentence meaning.

    ``descriptions`` is the list of Stage-A outputs for the same
    canonical face. ``synth_prompt_template`` is
    :data:`llmoji.haiku_prompts.SYNTHESIZE_PROMPT` (passed in
    explicitly so the call site is easy to audit).
    """
    descs = list(descriptions)
    if not descs:
        return ""
    listed = "\n".join(f"{j+1}. {d}" for j, d in enumerate(descs))
    return call_haiku(
        client,
        synth_prompt_template.format(descriptions=listed),
        model_id=model_id,
        max_tokens=max_tokens,
    )
