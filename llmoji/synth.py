"""Synthesis backend abstraction + per-cell content-hash cache.

The locked prompt + schema + lexicon live in :mod:`llmoji.synth_prompts`;
this module is just plumbing — masking, the per-backend single-call
helpers (plain ``.call()`` for free-form text, ``.call_structured()``
for the v2 schema-validated path), the per-cell cache key + hash
helpers, and a tiny content-hash cache so re-runs of ``llmoji
analyze`` only pay for cells whose sampled (user, assistant) set
has changed since the last run.

Three first-class backends:

- **anthropic** (default): ``anthropic.Anthropic.messages.create``
  with ``max_retries=8`` so the org-level Haiku rate cap doesn't
  abort a full re-analyze. Structured-output path uses
  ``output_config={"format": {"type": "json_schema", "schema":
  ...}}`` (verified shape from the installed SDK at
  ``anthropic/types/{output_config_param,json_output_format_param}.py``).
- **openai**: ``openai.OpenAI.responses.create`` (the Responses API,
  OpenAI's recommended path for new projects). Structured-output
  path uses ``text={"format": {"type": "json_schema", "name":
  ..., "schema": ..., "strict": True}}``.
- **local**: ``openai.OpenAI(base_url=...)`` against any
  OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp's server,
  etc.). Uses Chat Completions with ``response_format={"type":
  "json_schema", "json_schema": {...}}`` first; on
  ``BadRequestError`` (endpoint doesn't support constrained
  decoding) falls back to appending ``Output strictly as JSON
  matching this schema:\\n{schema}`` to the prompt and parsing the
  unconstrained reply.

Cache layout (one JSONL line per cached call):

    {
      "key":          sha256(synth_model_id + "\\0" + backend
                            + "\\0" + base_url + "\\0" + source_model
                            + "\\0" + canonical_kaomoji + "\\0"
                            + sample_set_hash)[:16],
      "kaomoji":      canonical kaomoji,
      "source_model": source model slug,
      "synthesis":    {primary_affect, stance_modality_function},
      "model":        synthesizer model id used,
      "backend":      "anthropic" | "openai" | "local",
    }

The sample-set hash means a re-import that grows a cell's row
list past INSTANCE_SAMPLE_CAP (changing which 4 rows get sampled)
correctly cache-misses, while a re-run on identical data hits
cleanly. Backend + base_url + model_id are also in the key —
switching backends or pointing a local backend at a different
endpoint can't silently return adjective bags from the prior call.

The cache file lives at ``~/.llmoji/cache/per_cell.jsonl`` by
default; the path is parameterized so tests can use a tmpdir. Pre
v2 used ``per_instance.jsonl`` with a finer-grained key; that
file is orphaned by v2 and the first ``analyze`` run prints a
one-line notice if it's still on disk (cleanup via
``llmoji cache clear``).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .synth_prompts import DEFAULT_ANTHROPIC_MODEL_ID, DEFAULT_OPENAI_MODEL_ID

MASK_TOKEN = "[FACE]"


class SynthesizerError(RuntimeError):
    """Raised by :meth:`Synthesizer.call_structured` when the backend
    response can't be parsed (no text block, empty body, etc.).
    Distinct from SDK-level errors so the caller can decide whether
    to retry or surface."""


def mask_kaomoji(text: str, first_word: str) -> str:
    """Prepend :data:`MASK_TOKEN` to a kaomoji-stripped assistant body.

    By the journal contract, ``assistant_text`` never includes the
    leading kaomoji — that's carried separately in the row's
    ``kaomoji`` field. Live-hook journals strip on write
    (``ltrimstr($kaomoji)`` in the bash template), the static-export
    readers strip on parse, and the generic-JSONL contract requires
    the same. So this function just prepends ``[FACE] `` to give the
    synthesizer the ``[FACE] <body>`` shape its prompts promise.

    Empty ``first_word`` (no kaomoji on this row — shouldn't reach
    here in normal flow, but defensive) → pass through unchanged.
    """
    if not first_word:
        return text
    return MASK_TOKEN + " " + text.lstrip()


# ---------------------------------------------------------------------------
# Per-cell content-hash cache
# ---------------------------------------------------------------------------


def samples_hash(samples: Iterable[tuple[str, str]]) -> str:
    """Deterministic 16-hex hash of a (user, assistant) sample list.

    Sorts the tuples first so re-runs that present the samples in
    a different order still produce the same hash. Length-prefix
    framing (``len:bytes``) on every field prevents a NUL or any
    other byte buried inside a field from shifting the boundary
    between fields and collapsing distinct tuples into the same
    hash — the same pattern :func:`cache_key` uses.

    Used as the last component of :func:`cache_key` so a cell whose
    sampled set is identical to a prior run hits the cache, while
    a cell whose sampling has shifted (new rows added,
    INSTANCE_SAMPLE_CAP bumped) misses cleanly.
    """
    pairs = sorted((u or "", a or "") for u, a in samples)
    h = hashlib.sha256()
    for u, a in pairs:
        for part in (u, a):
            encoded = part.encode("utf-8")
            h.update(str(len(encoded)).encode("ascii"))
            h.update(b":")
            h.update(encoded)
    return h.hexdigest()[:16]


def cache_key(
    synth_model_id: str,
    backend: str,
    base_url: str,
    source_model: str,
    canonical_kaomoji: str,
    sample_set_hash: str,
) -> str:
    """Deterministic 16-hex-char content hash key for a per-cell call.

    Truncated SHA-256 — collisions on a single user's corpus are
    astronomically unlikely (~2^32 entries before a ~50% collision
    probability against a 64-bit space). The cache is private to
    one machine; no security boundary depends on the hash.

    Backend, base_url, and model id are folded in alongside the
    cell identity so two backends sharing a model name (e.g.
    ``local`` running an Ollama tag that collides with a remote
    id) — or one ``local`` instance pointed at two different
    endpoints — don't share cache entries. The adjective bag a
    given (model, backend, endpoint) produces for one cell is its
    own thing.

    If a future federated/shared cache lands, bump from the
    truncated 16-hex prefix to the full SHA-256 hexdigest —
    collision probability scales quadratically with corpus size and
    64 bits is only safe at single-machine scale.
    """
    h = hashlib.sha256()
    # Length-prefix each field so a NUL byte buried inside (e.g. an
    # source-model string that happens to contain ``"\0"``) can't
    # shift a field boundary and collide with a different
    # (source_model, canonical) pair. Real input is unlikely to
    # carry NULs but the framing is cheap.
    for part in (
        synth_model_id or "",
        backend or "",
        base_url or "",
        source_model or "",
        canonical_kaomoji,
        sample_set_hash or "",
    ):
        encoded = part.encode("utf-8")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
    return h.hexdigest()[:16]


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load the per-cell cache as ``{key: row}``. Empty / missing
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


def cache_size(cache_path: Path) -> int:
    """Return the cache file's size in bytes.

    Used by ``llmoji status`` so the user knows what's on disk. Pre
    Wave 4 this also walked the file to count rows; that scan is
    wasted work on multi-hundred-MB caches and the row count was
    never load-bearing — bytes is the user-facing number.
    """
    if not cache_path.exists():
        return 0
    return cache_path.stat().st_size


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class Synthesizer:
    """Base class — concrete subclasses route ``call(prompt)`` and
    ``call_structured(prompt, schema=...)`` to their backend. The
    whole pipeline holds a single instance and calls it from N
    threads (the Anthropic httpx client and OpenAI's httpx client
    are both thread-safe), so subclasses must keep both methods
    reentrant.

    ``base_url`` is empty for the hosted backends (anthropic, openai)
    and set to the user-supplied endpoint for ``local``. It feeds
    into :func:`cache_key` so two ``local`` instances pointed at
    different endpoints don't share cache entries.

    Concrete synthesizers all defer SDK-client construction to the
    first ``call`` / ``call_structured`` so the factory itself can
    be invoked without environment variables set (constructor
    side-effects would otherwise force a real ``OPENAI_API_KEY``
    just to enumerate backends in tests, ``llmoji status``, etc.).
    The lazy client is memoized on ``self._client`` behind a
    per-instance lock — synthesis is multi-threaded and an
    unguarded check-then-set would race on the first cache-miss
    wave, instantiating N clients instead of one (and burning N
    OAuth flows on the openai backend). Subclasses implement
    :meth:`_make_client` (the SDK import + constructor call); the
    base class owns the double-checked locking around it.

    The plain ``.call()`` method is kept additively from v1; the
    v2 pipeline uses ``.call_structured()`` exclusively, but
    free-form callers (future research scripts, ad-hoc CLI tools)
    keep working.
    """

    backend: str = ""
    model_id: str = ""
    base_url: str = ""

    def __init__(self) -> None:
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _make_client(self) -> Any:
        raise NotImplementedError

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._make_client()
        return self._client

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        del prompt, max_tokens
        raise NotImplementedError

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        del prompt, schema, max_tokens
        raise NotImplementedError


class AnthropicSynthesizer(Synthesizer):
    """``anthropic.Anthropic`` via ``messages.create``.

    ``max_retries=8`` (vs the SDK default of 2) so a multi-hundred-row
    re-analyze can ride out a 50 req/min Haiku cap collision. The SDK
    honors the response's Retry-After header and uses exponential
    backoff between retries.

    Structured-output path uses ``output_config={"format": {"type":
    "json_schema", "schema": ...}}`` — verified against the
    installed SDK's ``JSONOutputFormatParam`` (only fields are
    ``type`` and ``schema`` — no ``name``, ``strict``, or
    ``description`` at the format level).
    """

    backend = "anthropic"

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id

    def _make_client(self) -> Any:
        import anthropic
        return anthropic.Anthropic(max_retries=8)

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

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        msg = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    return json.loads(text)
        raise SynthesizerError(
            "AnthropicSynthesizer.call_structured: no text block in response",
        )


class OpenAISynthesizer(Synthesizer):
    """``openai.OpenAI`` via the Responses API.

    Responses is the recommended path for new projects on the
    official OpenAI platform; for our single-shot synthesis call
    it's just ``client.responses.create(model=..., input=prompt)``
    plus the ``.output_text`` convenience accessor.

    Structured-output path uses ``text={"format": {"type":
    "json_schema", "name": ..., "schema": ..., "strict": True}}``
    — verified against the installed SDK's
    ``ResponseTextConfigParam`` +
    ``ResponseFormatTextJSONSchemaConfigParam`` (``name`` is
    required by the API; ``strict`` is optional but enables strict
    schema adherence).
    """

    backend = "openai"

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id

    def _make_client(self) -> Any:
        import openai
        return openai.OpenAI(max_retries=8)

    def call(self, prompt: str, *, max_tokens: int = 200) -> str:
        client = self._ensure_client()
        resp = client.responses.create(
            model=self.model_id,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        return (resp.output_text or "").strip()

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        resp = client.responses.create(
            model=self.model_id,
            input=prompt,
            max_output_tokens=max_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "synthesis",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        text = (resp.output_text or "").strip()
        if not text:
            raise SynthesizerError(
                "OpenAISynthesizer.call_structured: empty output_text",
            )
        return json.loads(text)


class LocalSynthesizer(Synthesizer):
    """OpenAI-compatible Chat Completions against a local endpoint.

    Ollama, vLLM, llama.cpp's HTTP server etc. all expose a
    Chat-Completions-shaped API rather than the Responses API, so
    that's what we hit here. ``api_key`` defaults to a placeholder
    (``"ollama"``) since the ``openai.OpenAI`` constructor requires
    one even when the endpoint doesn't authenticate.

    Structured-output path tries ``response_format={"type":
    "json_schema", "json_schema": {...}}`` first (the
    OpenAI-compatible constrained-decoding shape Ollama / recent
    vLLM / recent llama.cpp all expose). On ``BadRequestError`` /
    ``UnprocessableEntityError`` (older builds, custom deployments
    without constrained decoding), falls back to appending ``Output
    strictly as JSON matching this schema:\\n{schema}`` to the
    prompt and parsing the unconstrained reply. The fallback is
    best-effort — schema violations there raise
    :class:`json.JSONDecodeError` to the caller.
    """

    backend = "local"

    def __init__(
        self, model_id: str, *, base_url: str, api_key: str = "ollama",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.base_url = base_url
        self._api_key = api_key

    def _make_client(self) -> Any:
        import openai
        return openai.OpenAI(base_url=self.base_url, api_key=self._api_key)

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

    def call_structured(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        import openai
        client = self._ensure_client()
        try:
            msg = client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "synthesis",
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except (
            openai.BadRequestError,
            openai.UnprocessableEntityError,
        ):
            aug = (
                prompt
                + "\n\nOutput strictly as JSON matching this schema:\n"
                + json.dumps(schema)
            )
            msg = client.chat.completions.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": aug}],
            )
        choice = msg.choices[0] if msg.choices else None
        if choice is None or not choice.message:
            raise SynthesizerError(
                "LocalSynthesizer.call_structured: empty response",
            )
        content = choice.message.content or ""
        if not content.strip():
            raise SynthesizerError(
                "LocalSynthesizer.call_structured: empty content",
            )
        return json.loads(content)


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
