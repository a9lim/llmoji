"""Synthesis backend abstraction + per-instance content-hash cache.

The actual prompt strings live in :mod:`llmoji.synth_prompts`; this
module is just plumbing — masking, a per-backend single-call helper,
and a content-hash cache so re-runs of ``llmoji analyze`` only pay
for new rows.

Three first-class backends:

- **anthropic** (default): ``anthropic.Anthropic.messages.create``
  with ``max_retries=8`` so the org-level Haiku rate cap doesn't
  abort a full re-analyze.
- **openai**: ``openai.OpenAI.responses.create`` (the Responses API,
  OpenAI's recommended path for new projects). Single-shot text in,
  ``output_text`` out — no streaming, no tools.
- **local**: ``openai.OpenAI(base_url=...)`` against any
  OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp's server,
  etc.). Uses Chat Completions because that's the surface those
  servers actually mimic; Responses-API local clones aren't a
  thing yet.

Cache layout (one JSONL line per cached call):

    {
      "key":          sha256(synth_model_id + "\\0" + backend
                            + "\\0" + base_url + "\\0" + canonical_kaomoji
                            + "\\0" + user + "\\0" + assistant)[:16],
      "kaomoji":      canonical kaomoji,
      "description":  synthesizer output,
      "model":        synthesizer model id used,
      "backend":      "anthropic" | "openai" | "local",
    }

Backend + base_url + model_id are all in the key — switching
backends or pointing a local backend at a different endpoint can't
silently return paraphrases from the prior call. One-time recompute
cost on the first analyze after upgrade (existing entries miss
cleanly because the key shape changed).

The cache file lives at ``~/.llmoji/cache/per_instance.jsonl`` by
default; the path is parameterized so tests can use a tmpdir.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable

MASK_TOKEN = "[FACE]"


def mask_kaomoji(text: str, first_word: str) -> str:
    """Prepend :data:`MASK_TOKEN` to a kaomoji-stripped assistant body.

    By the journal contract, ``assistant_text`` never includes the
    leading kaomoji — that's carried separately in the row's
    ``kaomoji`` field. Live-hook journals strip on write
    (``ltrimstr($kaomoji)`` in the bash template), the static-export
    readers strip on parse, and the generic-JSONL contract requires
    the same. So this function just prepends ``[FACE] `` to give the
    synthesizer the ``[FACE] <body>`` shape its DESCRIBE prompts
    promise.

    Empty ``first_word`` (no kaomoji on this row — shouldn't reach
    here in normal flow, but defensive) → pass through unchanged.
    """
    if not first_word:
        return text
    return MASK_TOKEN + " " + text.lstrip()


# ---------------------------------------------------------------------------
# Per-instance content-hash cache
# ---------------------------------------------------------------------------


def cache_key(
    synth_model_id: str,
    backend: str,
    base_url: str,
    canonical_kaomoji: str,
    user_text: str,
    assistant_text: str,
) -> str:
    """Deterministic 16-hex-char content hash key.

    Truncated SHA-256 — collisions on a single user's corpus are
    astronomically unlikely (~2^32 entries before a ~50% collision
    probability against a 64-bit space). The cache is private to
    one machine; no security boundary depends on the hash.

    Backend and base_url are folded in alongside the model id so two
    backends sharing a model name (e.g. ``local`` running an Ollama
    tag that collides with a remote id) — or one ``local`` instance
    pointed at two different endpoints — don't share cache entries.
    The prose differs by backend; the key has to too.

    If a future federated/shared cache lands, bump from the truncated
    16-hex prefix to the full SHA-256 hexdigest — collision
    probability scales quadratically with corpus size and 64 bits is
    only safe at single-machine scale.
    """
    h = hashlib.sha256()
    h.update((synth_model_id or "").encode("utf-8"))
    h.update(b"\0")
    h.update((backend or "").encode("utf-8"))
    h.update(b"\0")
    h.update((base_url or "").encode("utf-8"))
    h.update(b"\0")
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


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class Synthesizer:
    """Base class — concrete subclasses route ``call(prompt)`` to
    their backend. The whole pipeline holds a single instance and
    calls it from N threads (the Anthropic httpx client and OpenAI's
    httpx client are both thread-safe), so subclasses must keep
    ``call`` reentrant.

    ``base_url`` is empty for the hosted backends (anthropic, openai)
    and set to the user-supplied endpoint for ``local``. It feeds
    into :func:`cache_key` so two ``local`` instances pointed at
    different endpoints don't share cache entries.
    """

    backend: str = ""
    model_id: str = ""
    base_url: str = ""

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        raise NotImplementedError


# Concrete synthesizers all defer SDK-client construction to the
# first ``call`` so the factory itself can be invoked without
# environment variables set (constructor side-effects would
# otherwise force a real ``OPENAI_API_KEY`` just to enumerate
# backends in tests, ``llmoji status``, etc.). The lazy client is
# memoized on ``self._client`` behind a per-instance lock —
# Stage A is multi-threaded and an unguarded check-then-set would
# race on the first cache-miss wave, instantiating N clients
# instead of one (and burning N OAuth flows on the openai backend).


class AnthropicSynthesizer(Synthesizer):
    """``anthropic.Anthropic`` via ``messages.create``.

    ``max_retries=8`` (vs the SDK default of 2) so a multi-hundred-row
    re-analyze can ride out a 50 req/min Haiku cap collision. The SDK
    honors the response's Retry-After header and uses exponential
    backoff between retries.
    """

    backend = "anthropic"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import anthropic
                    self._client = anthropic.Anthropic(max_retries=8)
        return self._client

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        client = self._ensure_client()
        msg = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                return (getattr(block, "text", "") or "").strip()
        return ""


class OpenAISynthesizer(Synthesizer):
    """``openai.OpenAI`` via the Responses API.

    Responses is the recommended path for new projects on the
    official OpenAI platform; for our single-shot synthesis call it's
    just ``client.responses.create(model=..., input=prompt)`` plus
    the ``.output_text`` convenience accessor.
    """

    backend = "openai"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import openai
                    self._client = openai.OpenAI(max_retries=8)
        return self._client

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        client = self._ensure_client()
        resp = client.responses.create(
            model=self.model_id,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return (resp.output_text or "").strip()


class LocalSynthesizer(Synthesizer):
    """OpenAI-compatible Chat Completions against a local endpoint.

    Ollama, vLLM, llama.cpp's HTTP server etc. all expose a
    Chat-Completions-shaped API rather than the Responses API, so
    that's what we hit here. ``api_key`` defaults to a placeholder
    (``"ollama"``) since the ``openai.OpenAI`` constructor requires
    one even when the endpoint doesn't authenticate.
    """

    backend = "local"

    def __init__(
        self, model_id: str, *, base_url: str, api_key: str = "ollama",
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self._api_key = api_key
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import openai
                    self._client = openai.OpenAI(
                        base_url=self.base_url, api_key=self._api_key,
                    )
        return self._client

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        client = self._ensure_client()
        msg = client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = msg.choices[0] if msg.choices else None
        if choice is None:
            return ""
        return (choice.message.content or "").strip()


def make_synthesizer(
    backend: str,
    *,
    base_url: str | None = None,
    model_id: str | None = None,
) -> Synthesizer:
    """Factory. Lazy-imports the SDK for the chosen backend only, so
    a user without ``openai`` installed can still ``--backend
    anthropic`` (and vice versa, modulo ``openai`` being a hard dep
    in 1.1.0).

    - ``anthropic``: ignores ``base_url`` / ``model_id`` (always uses
      the pinned Haiku snapshot from
      :data:`llmoji.synth_prompts.DEFAULT_ANTHROPIC_MODEL_ID`).
    - ``openai``: same — pinned to
      :data:`llmoji.synth_prompts.DEFAULT_OPENAI_MODEL_ID`.
    - ``local``: requires both ``base_url`` and ``model_id``.
    """
    from .synth_prompts import DEFAULT_ANTHROPIC_MODEL_ID, DEFAULT_OPENAI_MODEL_ID

    if backend == "anthropic":
        return AnthropicSynthesizer(model_id=DEFAULT_ANTHROPIC_MODEL_ID)
    if backend == "openai":
        return OpenAISynthesizer(model_id=DEFAULT_OPENAI_MODEL_ID)
    if backend == "local":
        if not base_url or not model_id:
            raise ValueError(
                "local backend requires both --base-url and --model "
                "(or LLMOJI_BASE_URL + LLMOJI_MODEL env vars)."
            )
        return LocalSynthesizer(model_id=model_id, base_url=base_url)
    raise ValueError(
        f"unknown backend {backend!r}; expected one of "
        f"'anthropic', 'openai', 'local'."
    )


def synthesize_descriptions(
    synth: Synthesizer,
    descriptions: Iterable[str],
    *,
    synth_prompt_template: str,
    max_tokens: int = 200,
) -> str:
    """Stage B: pool per-instance descriptions for one
    ``(source_model, canonical_kaomoji)`` cell, return the
    synthesizer's one-sentence meaning.

    ``descriptions`` is the list of Stage-A outputs for the cell.
    ``synth_prompt_template`` is
    :data:`llmoji.synth_prompts.SYNTHESIZE_PROMPT` (passed in
    explicitly so the call site is easy to audit).
    """
    descs = list(descriptions)
    if not descs:
        return ""
    listed = "\n".join(f"{j+1}. {d}" for j, d in enumerate(descs))
    return synth.call(
        synth_prompt_template.format(descriptions=listed),
        max_tokens=max_tokens,
    )
