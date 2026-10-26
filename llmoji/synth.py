"""Synthesis backends and the per-cell cache.

One structured call per ``(source model, face)`` cell; the prompt,
lexicon and schema are locked in :mod:`llmoji.synth_prompts`.

- **anthropic** (default): Messages API, structured output through
  ``output_config.format``.
- **openai**: Responses API, ``text.format`` json_schema in strict mode.
- **local**: any OpenAI-compatible Chat Completions endpoint (Ollama,
  vLLM, llama.cpp). Constrained ``response_format`` when the endpoint
  takes it, else the schema appended to the prompt.

Every result is checked against the lexicon before it is cached or
bundled. The bundle carries nothing but lexicon words, and a local
model decoding without constraints could otherwise put anything there,
user text included.

The cache is a JSONL at ``~/.llmoji/cache/per_cell.jsonl`` keyed by
synth model, backend, endpoint, cell and the hash of the sampled
``(user, assistant)`` pairs, so a cell whose sample shifts misses
cleanly and a changed backend never reuses another's answer.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .synth_prompts import (
    CIRCUMPLEX_ANCHORS,
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    EXTENSION_AXES,
)

MASK_TOKEN = "[FACE]"
# The pinned models think before answering; this leaves them room.
MAX_TOKENS = 4096

_FIELDS = {
    "primary_affect": frozenset(CIRCUMPLEX_ANCHORS),
    "stance_modality_function": frozenset(EXTENSION_AXES),
}


class SynthesizerError(RuntimeError):
    """A backend reply that isn't a usable synthesis."""


def mask_kaomoji(body: str) -> str:
    """The ``[FACE] <body>`` form the prompt promises. Journal bodies
    never carry the face, so the mask is just a prefix."""
    return f"{MASK_TOKEN} {body.lstrip()}"


def validate_synthesis(obj: Any) -> dict[str, list[str]]:
    """``obj`` if it is exactly the two schema fields, each a list of
    words from its half of the lexicon; :class:`SynthesizerError`
    otherwise. Word counts aren't checked: the API schemas can't express
    them, and the prompt asks for them instead."""
    if not isinstance(obj, dict) or set(obj) != set(_FIELDS):
        raise SynthesizerError(f"not a synthesis object: {obj!r}"[:300])
    for key, allowed in _FIELDS.items():
        words = obj[key]
        if not isinstance(words, list) or not all(isinstance(w, str) and w in allowed for w in words):
            raise SynthesizerError(f"{key} holds words outside the lexicon: {words!r}"[:300])
    return obj


def _digest(*parts: str) -> str:
    """16-hex sha256 over length-prefixed parts, so no byte inside a
    part can move a boundary between parts."""
    h = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        h.update(f"{len(encoded)}:".encode("ascii") + encoded)
    return h.hexdigest()[:16]


def samples_hash(samples: Iterable[tuple[str, str]]) -> str:
    """Order-independent hash of a cell's sampled ``(user, assistant)``
    pairs."""
    return _digest(*(part for pair in sorted(samples) for part in pair))


def cache_key(
    synth_model_id: str,
    backend: str,
    base_url: str,
    source_model: str,
    canonical_kaomoji: str,
    sample_set_hash: str,
) -> str:
    return _digest(
        synth_model_id, backend, base_url, source_model, canonical_kaomoji, sample_set_hash,
    )


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return out
    with cache_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                out[row["key"]] = row
    return out


def append_cache(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Synthesizer:
    """One backend. A single instance is shared across worker threads;
    the SDK client is built on first use, under a lock so the first
    wave of calls doesn't build one each."""

    backend = ""
    base_url = ""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._client_obj: Any = None
        self._lock = threading.Lock()

    def _client(self) -> Any:
        if self._client_obj is None:
            with self._lock:
                if self._client_obj is None:
                    self._client_obj = self._make_client()
        return self._client_obj

    def _make_client(self) -> Any:
        raise NotImplementedError

    def _call(self, prompt: str, schema: dict[str, Any]) -> str:
        raise NotImplementedError

    def call_structured(self, prompt: str, *, schema: dict[str, Any]) -> dict[str, list[str]]:
        text = self._call(prompt, schema).strip()
        if not text:
            raise SynthesizerError(f"{self.backend}: empty reply")
        try:
            return validate_synthesis(json.loads(text))
        except json.JSONDecodeError as e:
            raise SynthesizerError(f"{self.backend}: reply isn't JSON: {text[:200]!r}") from e


class AnthropicSynthesizer(Synthesizer):
    backend = "anthropic"

    def _make_client(self) -> Any:
        import anthropic

        # Rides out the organisation-wide rate cap on a long re-analyze.
        return anthropic.Anthropic(max_retries=8)

    def _call(self, prompt: str, schema: dict[str, Any]) -> str:
        msg = self._client().messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if msg.stop_reason != "end_turn":
            raise SynthesizerError(f"anthropic: stopped with {msg.stop_reason!r}")
        return next((b.text for b in msg.content if b.type == "text"), "")


class OpenAISynthesizer(Synthesizer):
    backend = "openai"

    def _make_client(self) -> Any:
        import openai

        return openai.OpenAI(max_retries=8)

    def _call(self, prompt: str, schema: dict[str, Any]) -> str:
        resp = self._client().responses.create(
            model=self.model_id,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "synthesis",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return resp.output_text or ""


class LocalSynthesizer(Synthesizer):
    backend = "local"

    def __init__(self, model_id: str, *, base_url: str) -> None:
        super().__init__(model_id)
        self.base_url = base_url

    def _make_client(self) -> Any:
        import openai

        # The constructor wants a key even when the endpoint takes none.
        return openai.OpenAI(base_url=self.base_url, api_key="local")

    def _call(self, prompt: str, schema: dict[str, Any]) -> str:
        import openai

        try:
            msg = self._client().chat.completions.create(
                model=self.model_id,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "synthesis", "schema": schema, "strict": True},
                },
            )
        except (openai.BadRequestError, openai.UnprocessableEntityError):
            msg = self._client().chat.completions.create(
                model=self.model_id,
                max_tokens=MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": prompt
                    + "\n\nOutput strictly as JSON matching this schema:\n"
                    + json.dumps(schema),
                }],
            )
        return (msg.choices[0].message.content or "") if msg.choices else ""


def make_synthesizer(
    backend: str, *, base_url: str | None = None, model_id: str | None = None,
) -> Synthesizer:
    """The hosted backends always use their pinned model; ``local``
    needs both ``base_url`` and ``model_id``."""
    if backend == "anthropic":
        return AnthropicSynthesizer(DEFAULT_ANTHROPIC_MODEL_ID)
    if backend == "openai":
        return OpenAISynthesizer(DEFAULT_OPENAI_MODEL_ID)
    if backend == "local":
        if not base_url or not model_id:
            raise ValueError("local backend requires both --base-url and --model")
        return LocalSynthesizer(model_id, base_url=base_url)
    raise ValueError(f"unknown backend {backend!r}; expected anthropic, openai or local")
