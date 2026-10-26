"""Synthesis backends, reply validation, cache keys and the cache file.

Fake clients stand in for the SDKs, so each test pins the exact kwargs a
backend sends and what it does with the reply. No network.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import openai
import pytest

from llmoji.synth import (
    MAX_TOKENS,
    AnthropicSynthesizer,
    LocalSynthesizer,
    OpenAISynthesizer,
    SynthesizerError,
    append_cache,
    cache_key,
    load_cache,
    make_synthesizer,
    mask_kaomoji,
    samples_hash,
    validate_synthesis,
)
from llmoji.synth_prompts import (
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    SYNTHESIS_SCHEMA,
)

GOOD = {
    "primary_affect": ["cheery", "playful"],
    "stance_modality_function": ["helpful", "eager", "wry"],
}
REPLY = json.dumps(GOOD)


def _anthropic(text: str | None = REPLY, stop_reason: str = "end_turn") -> AnthropicSynthesizer:
    blocks = [] if text is None else [SimpleNamespace(type="text", text=text)]
    s = AnthropicSynthesizer("claude-test")
    s._client_obj = SimpleNamespace(messages=SimpleNamespace(create=MagicMock(
        return_value=SimpleNamespace(content=blocks, stop_reason=stop_reason),
    )))
    return s


def _openai(output_text: str | None = REPLY) -> OpenAISynthesizer:
    s = OpenAISynthesizer("gpt-test")
    s._client_obj = SimpleNamespace(responses=SimpleNamespace(create=MagicMock(
        return_value=SimpleNamespace(output_text=output_text),
    )))
    return s


def _completion(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _local(*results: Any) -> LocalSynthesizer:
    """A local backend whose successive completions are ``results``
    (an exception instance is raised instead of returned)."""
    s = LocalSynthesizer("llama-test", base_url="http://localhost:11434/v1")
    s._client_obj = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=MagicMock(side_effect=list(results)),
    )))
    return s


def _http_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    """An instance of the SDK's error class without building an HTTP
    response: openai 3.x uses httpx2 and the 1.66 floor httpx, and
    neither is llmoji's own dependency."""
    err = cls.__new__(cls)
    Exception.__init__(err, "response_format not supported")
    err.status_code = status
    return err


# --- request shapes ---------------------------------------------------------

def test_anthropic_request_shape() -> None:
    s = _anthropic()
    assert s.call_structured("hello", schema=SYNTHESIS_SCHEMA) == GOOD
    kwargs = s._client_obj.messages.create.call_args.kwargs
    assert kwargs == {
        "model": "claude-test",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": "hello"}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": SYNTHESIS_SCHEMA},
        },
    }


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "tool_use", None])
def test_anthropic_refuses_a_stop_other_than_end_turn(stop_reason: str | None) -> None:
    """A truncated or refused reply can still parse as JSON; the stop
    reason is the only thing that says it isn't the answer."""
    s = _anthropic(stop_reason=stop_reason)  # type: ignore[arg-type]
    with pytest.raises(SynthesizerError, match="stopped with"):
        s.call_structured("hello", schema=SYNTHESIS_SCHEMA)


def test_openai_request_shape() -> None:
    s = _openai()
    assert s.call_structured("hello", schema=SYNTHESIS_SCHEMA) == GOOD
    assert s._client_obj.responses.create.call_args.kwargs == {
        "model": "gpt-test",
        "input": "hello",
        "max_output_tokens": MAX_TOKENS,
        "text": {"format": {
            "type": "json_schema", "name": "synthesis",
            "schema": SYNTHESIS_SCHEMA, "strict": True,
        }},
    }


def test_local_request_shape() -> None:
    s = _local(_completion(REPLY))
    assert s.call_structured("hello", schema=SYNTHESIS_SCHEMA) == GOOD
    create = s._client_obj.chat.completions.create
    create.assert_called_once()
    assert create.call_args.kwargs == {
        "model": "llama-test",
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "synthesis", "schema": SYNTHESIS_SCHEMA, "strict": True},
        },
    }


@pytest.mark.parametrize(
    "cls, status", [(openai.BadRequestError, 400), (openai.UnprocessableEntityError, 422)],
)
def test_local_falls_back_to_the_schema_in_the_prompt(
    cls: type[openai.APIStatusError], status: int,
) -> None:
    s = _local(_http_error(cls, status), _completion(REPLY))
    assert s.call_structured("hello", schema=SYNTHESIS_SCHEMA) == GOOD
    calls = s._client_obj.chat.completions.create.call_args_list
    assert len(calls) == 2
    retry = calls[1].kwargs
    assert "response_format" not in retry
    assert retry["model"] == "llama-test" and retry["max_tokens"] == MAX_TOKENS
    content = retry["messages"][0]["content"]
    assert content.startswith("hello\n\n")
    assert "Output strictly as JSON matching this schema:" in content
    assert json.dumps(SYNTHESIS_SCHEMA) in content


def test_local_does_not_swallow_other_errors() -> None:
    s = _local(_http_error(openai.AuthenticationError, 401))
    with pytest.raises(openai.AuthenticationError):
        s.call_structured("hello", schema=SYNTHESIS_SCHEMA)
    assert s._client_obj.chat.completions.create.call_count == 1


# --- clients ----------------------------------------------------------------

@pytest.mark.parametrize("module, attr, make, expected", [
    ("anthropic", "Anthropic", lambda: AnthropicSynthesizer("m"), {"max_retries": 8}),
    ("openai", "OpenAI", lambda: OpenAISynthesizer("m"), {"max_retries": 8}),
    ("openai", "OpenAI", lambda: LocalSynthesizer("m", base_url="http://h/v1"),
     {"base_url": "http://h/v1", "api_key": "local"}),
])
def test_client_construction(
    monkeypatch: pytest.MonkeyPatch, module: str, attr: str, make: Any, expected: dict[str, Any],
) -> None:
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(importlib.import_module(module), attr, lambda **kw: built.append(kw))
    make()._client()
    assert built == [expected]


# --- replies that aren't a synthesis ----------------------------------------

@pytest.mark.parametrize("make", [
    lambda text: _anthropic(text),
    lambda text: _openai(text),
    lambda text: _local(_completion(text)),
], ids=["anthropic", "openai", "local"])
@pytest.mark.parametrize("text", [None, "", "  \n"])
def test_empty_reply_raises(make: Any, text: str | None) -> None:
    with pytest.raises(SynthesizerError, match="empty reply"):
        make(text).call_structured("hello", schema=SYNTHESIS_SCHEMA)


def test_local_without_choices_raises() -> None:
    s = _local(SimpleNamespace(choices=[]))
    with pytest.raises(SynthesizerError, match="empty reply"):
        s.call_structured("hello", schema=SYNTHESIS_SCHEMA)


@pytest.mark.parametrize("text", ["sure! (◕‿◕)", '{"primary_affect": ["sad"', "```json\n{}\n```"])
def test_non_json_reply_raises(text: str) -> None:
    with pytest.raises(SynthesizerError, match="isn't JSON"):
        _openai(text).call_structured("hello", schema=SYNTHESIS_SCHEMA)


def test_reply_outside_the_lexicon_raises() -> None:
    bad = json.dumps({**GOOD, "primary_affect": ["ecstatic"]})
    with pytest.raises(SynthesizerError, match="outside the lexicon"):
        _anthropic(bad).call_structured("hello", schema=SYNTHESIS_SCHEMA)


# --- validate_synthesis -----------------------------------------------------

def test_validate_synthesis_accepts_a_valid_object() -> None:
    assert validate_synthesis(GOOD) is GOOD


@pytest.mark.parametrize("bad", [
    pytest.param(None, id="not-a-dict"),
    pytest.param([GOOD], id="list"),
    pytest.param({}, id="empty"),
    pytest.param({"primary_affect": ["sad"]}, id="missing-key"),
    pytest.param({**GOOD, "notes": []}, id="extra-key"),
    pytest.param({**GOOD, "primary_affect": "sad"}, id="string-not-list"),
    pytest.param({**GOOD, "stance_modality_function": None}, id="none-not-list"),
    pytest.param({**GOOD, "primary_affect": ["ecstatic"]}, id="not-in-lexicon"),
    pytest.param({**GOOD, "primary_affect": ["helpful"]}, id="extension-word-in-affect"),
    pytest.param({**GOOD, "stance_modality_function": ["sad"]}, id="anchor-in-extension"),
    pytest.param({**GOOD, "stance_modality_function": ["wry", 3]}, id="non-string-word"),
    pytest.param({**GOOD, "primary_affect": [["sad"]]}, id="nested-list"),
    pytest.param({**GOOD, "stance_modality_function": ["wry", "my password is hunter2"]},
                 id="free-text"),
])
def test_validate_synthesis_rejects(bad: Any) -> None:
    with pytest.raises(SynthesizerError):
        validate_synthesis(bad)


# --- mask_kaomoji -----------------------------------------------------------

@pytest.mark.parametrize("body, masked", [
    ("hello there", "[FACE] hello there"),
    ("  padded", "[FACE] padded"),
    ("", "[FACE] "),
])
def test_mask_kaomoji(body: str, masked: str) -> None:
    assert mask_kaomoji(body) == masked


# --- cache_key / samples_hash -----------------------------------------------

_KEY = ("m", "anthropic", "", "src", "(◕‿◕)", "h")


def test_cache_key_is_16_hex_and_deterministic() -> None:
    k = cache_key(*_KEY)
    assert k == cache_key(*_KEY)
    assert len(k) == 16 and all(c in "0123456789abcdef" for c in k)


@pytest.mark.parametrize("i, name", enumerate(
    ["model", "backend", "base_url", "source_model", "face", "sample_hash"],
))
def test_cache_key_varies_with_every_field(i: int, name: str) -> None:
    """The same model name on another backend or endpoint, or the same
    face from another source model, must not reuse an answer."""
    other = tuple(f + "x" if j == i else f for j, f in enumerate(_KEY))
    assert cache_key(*other) != cache_key(*_KEY), name


@pytest.mark.parametrize("a, b", [
    (("src\0", "x"), ("src", "\0x")),  # a NUL can't move a boundary
    (("src", ""), ("", "src")),  # nor can an empty field
    (("3:abc", "x"), ("abc", "x")),  # nor text that looks like a length prefix
])
def test_cache_key_framing(a: tuple[str, str], b: tuple[str, str]) -> None:
    assert cache_key("m", "anthropic", "", *a, "h") != cache_key("m", "anthropic", "", *b, "h")


def test_samples_hash_ignores_order() -> None:
    pairs = [("u1", "a1"), ("u2", "a2"), ("u3", "a3")]
    assert samples_hash(pairs) == samples_hash(reversed(pairs))


def test_samples_hash_is_content_sensitive() -> None:
    base = [("u1", "a1"), ("u2", "a2")]
    assert samples_hash(base) != samples_hash([("u1", "a1"), ("u2", "a2-changed")])
    assert samples_hash(base) != samples_hash(base[:1])
    assert samples_hash(base[:1]) != samples_hash(base[:1] * 2)


@pytest.mark.parametrize("a, b", [
    ([("a", "b\0c")], [("a\0b", "c")]),
    ([("a", "b")], [("ab", "")]),
    ([("a", "b"), ("c", "d")], [("a", "bc"), ("", "d")]),
])
def test_samples_hash_framing(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> None:
    assert samples_hash(a) != samples_hash(b)


# --- cache file -------------------------------------------------------------

def test_cache_round_trip_creates_parents_and_keeps_unicode(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "cache.jsonl"
    row = {"key": "k1", "kaomoji": "(◕‿◕)", "synthesis": GOOD}
    append_cache(path, row)
    assert "(◕‿◕)" in path.read_text(encoding="utf-8")  # not \u-escaped
    assert load_cache(path) == {"k1": row}


def test_load_cache_skips_unusable_lines_and_last_row_wins(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    assert load_cache(path) == {}
    path.write_text("\n".join([
        json.dumps({"key": "a", "n": 1}),
        "{truncated",
        json.dumps(["not", "a", "row"]),
        json.dumps({"no_key": True}),
        json.dumps({"key": 7}),
        json.dumps({"key": "a", "n": 2}),
    ]) + "\n")
    assert load_cache(path) == {"a": {"key": "a", "n": 2}}


# --- make_synthesizer -------------------------------------------------------

def test_make_synthesizer_dispatch() -> None:
    a = make_synthesizer("anthropic")
    o = make_synthesizer("openai")
    loc = make_synthesizer("local", base_url="http://x/v1", model_id="llama")
    assert (type(a), a.model_id, a.backend) == (
        AnthropicSynthesizer, DEFAULT_ANTHROPIC_MODEL_ID, "anthropic")
    assert (type(o), o.model_id, o.backend) == (
        OpenAISynthesizer, DEFAULT_OPENAI_MODEL_ID, "openai")
    assert (type(loc), loc.model_id, loc.backend, loc.base_url) == (
        LocalSynthesizer, "llama", "local", "http://x/v1")
    assert a.base_url == o.base_url == ""


def test_hosted_backends_use_their_pinned_model_and_build_no_client() -> None:
    a = make_synthesizer("anthropic", model_id="other", base_url="http://x")
    o = make_synthesizer("openai", model_id="other", base_url="http://x")
    assert (a.model_id, a.base_url) == (DEFAULT_ANTHROPIC_MODEL_ID, "")
    assert (o.model_id, o.base_url) == (DEFAULT_OPENAI_MODEL_ID, "")
    assert a._client_obj is None and o._client_obj is None


@pytest.mark.parametrize("kwargs", [
    {}, {"base_url": "http://x/v1"}, {"model_id": "llama"},
    {"base_url": "", "model_id": "llama"}, {"base_url": "http://x/v1", "model_id": ""},
])
def test_local_backend_needs_base_url_and_model(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="--base-url and --model"):
        make_synthesizer("local", **kwargs)


def test_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown backend 'gemini'"):
        make_synthesizer("gemini")

