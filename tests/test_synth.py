"""Tests for the per-backend ``Synthesizer.call_structured`` SDK
shapes.

These pin the exact SDK call signatures so a future SDK upgrade
that renames a parameter (Anthropic ``output_config`` → something
else, OpenAI ``text.format`` → something else) fails loudly here
rather than silently in production. Verified shapes per the
installed SDKs (``anthropic==0.97.0``, ``openai==2.33.0``).

The tests mock the SDK clients via ``unittest.mock``; no real API
calls are made. Each test asserts the exact kwargs the
synthesizer passes to ``messages.create`` /
``responses.create`` / ``chat.completions.create``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from llmoji.synth import (
    AnthropicSynthesizer,
    LocalSynthesizer,
    OpenAISynthesizer,
    SynthesizerError,
)
from llmoji.synth_prompts import SYNTHESIS_SCHEMA


# Sentinel synthesis the mocked SDK returns from a successful call.
_FAKE_RESULT = {
    "primary_affect": ["cheerful"],
    "stance_modality_function": ["warm", "sincere"],
}


# ---------------------------------------------------------------------------
# Anthropic — output_config={"format": {"type": "json_schema", ...}}
# ---------------------------------------------------------------------------


def test_anthropic_call_structured_uses_output_config() -> None:
    """``output_config={"format": {"type": "json_schema", "schema":
    SYNTHESIS_SCHEMA}}`` — no ``name``, no ``strict``, no
    ``description`` at the format level (per the installed
    ``anthropic.types.json_output_format_param.JSONOutputFormatParam``
    typeddef which carries only ``type`` and ``schema``).
    """
    synth = AnthropicSynthesizer(model_id="claude-haiku-test")
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = json.dumps(_FAKE_RESULT)
    fake_msg = MagicMock()
    fake_msg.content = [fake_block]
    client = MagicMock()
    client.messages.create.return_value = fake_msg
    synth._client = client

    out = synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)

    assert out == _FAKE_RESULT
    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-test"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    # The format object is exactly ``{type, schema}`` — no extra fields.
    assert kwargs["output_config"] == {
        "format": {"type": "json_schema", "schema": SYNTHESIS_SCHEMA},
    }


def test_anthropic_call_structured_raises_on_no_text_block() -> None:
    """If the response carries no text block, surface
    :class:`SynthesizerError` rather than crashing on JSON parse."""
    import pytest

    synth = AnthropicSynthesizer(model_id="claude-haiku-test")
    fake_msg = MagicMock()
    fake_msg.content = []  # no blocks at all
    client = MagicMock()
    client.messages.create.return_value = fake_msg
    synth._client = client

    with pytest.raises(SynthesizerError):
        synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)


# ---------------------------------------------------------------------------
# OpenAI Responses — text={"format": {"type": "json_schema", ...}}
# ---------------------------------------------------------------------------


def test_openai_call_structured_uses_text_format() -> None:
    """``text={"format": {"type": "json_schema", "name": "synthesis",
    "schema": SYNTHESIS_SCHEMA, "strict": True}}`` — ``name`` is
    required by the API; ``strict`` enables strict schema
    adherence.
    """
    synth = OpenAISynthesizer(model_id="gpt-test")
    fake_resp = MagicMock()
    fake_resp.output_text = json.dumps(_FAKE_RESULT)
    client = MagicMock()
    client.responses.create.return_value = fake_resp
    synth._client = client

    out = synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)

    assert out == _FAKE_RESULT
    client.responses.create.assert_called_once()
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["input"] == "hello"
    assert kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "synthesis",
            "schema": SYNTHESIS_SCHEMA,
            "strict": True,
        },
    }


def test_openai_call_structured_raises_on_empty_output_text() -> None:
    """Empty ``output_text`` → :class:`SynthesizerError`."""
    import pytest

    synth = OpenAISynthesizer(model_id="gpt-test")
    fake_resp = MagicMock()
    fake_resp.output_text = ""
    client = MagicMock()
    client.responses.create.return_value = fake_resp
    synth._client = client

    with pytest.raises(SynthesizerError):
        synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)


# ---------------------------------------------------------------------------
# Local — response_format={"type": "json_schema", "json_schema": {...}}
# with prompt-fallback on BadRequestError
# ---------------------------------------------------------------------------


def test_local_call_structured_uses_response_format() -> None:
    """The native path uses ``response_format={"type": "json_schema",
    "json_schema": {"name": ..., "schema": ..., "strict": True}}``
    — the OpenAI-Chat-Completions-compatible constrained-decoding
    shape that Ollama / vLLM / recent llama.cpp expose."""
    synth = LocalSynthesizer(
        model_id="llama-test", base_url="http://localhost:11434/v1",
    )
    fake_choice = MagicMock()
    fake_choice.message.content = json.dumps(_FAKE_RESULT)
    fake_msg = MagicMock()
    fake_msg.choices = [fake_choice]
    client = MagicMock()
    client.chat.completions.create.return_value = fake_msg
    synth._client = client

    out = synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)

    assert out == _FAKE_RESULT
    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "llama-test"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "synthesis",
            "schema": SYNTHESIS_SCHEMA,
            "strict": True,
        },
    }


def test_local_call_structured_falls_back_on_bad_request() -> None:
    """Endpoints without native constrained decoding raise
    :class:`openai.BadRequestError` on ``response_format``. The
    fallback re-issues the call with the schema appended to the
    prompt.
    """
    import openai

    synth = LocalSynthesizer(
        model_id="llama-test", base_url="http://localhost:11434/v1",
    )
    fake_choice = MagicMock()
    fake_choice.message.content = json.dumps(_FAKE_RESULT)
    fake_msg = MagicMock()
    fake_msg.choices = [fake_choice]
    client = MagicMock()

    # First call (native path) raises BadRequestError; second call
    # (fallback) returns successfully.
    bad_request = openai.BadRequestError(
        message="response_format not supported",
        response=MagicMock(),
        body=None,
    )
    client.chat.completions.create.side_effect = [bad_request, fake_msg]
    synth._client = client

    out = synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)

    assert out == _FAKE_RESULT
    assert client.chat.completions.create.call_count == 2
    # Second call uses the augmented prompt + drops response_format.
    second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert "response_format" not in second_kwargs
    assert "Output strictly as JSON" in second_kwargs["messages"][0]["content"]


def test_local_call_structured_raises_on_empty_response() -> None:
    """No choices in the response → :class:`SynthesizerError`."""
    import pytest

    synth = LocalSynthesizer(
        model_id="llama-test", base_url="http://localhost:11434/v1",
    )
    fake_msg = MagicMock()
    fake_msg.choices = []
    client = MagicMock()
    client.chat.completions.create.return_value = fake_msg
    synth._client = client

    with pytest.raises(SynthesizerError):
        synth.call_structured("hello", schema=SYNTHESIS_SCHEMA)


# ---------------------------------------------------------------------------
# Synthesizer.call (plain text path) — kept additively for free-form callers
# ---------------------------------------------------------------------------


def test_anthropic_call_plain_text_still_works() -> None:
    """The v1 ``.call(prompt)`` plain-text path is preserved; only
    ``.call_structured`` is new in v2.
    """
    synth = AnthropicSynthesizer(model_id="claude-haiku-test")
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "free-form reply"
    fake_msg = MagicMock()
    fake_msg.content = [fake_block]
    client = MagicMock()
    client.messages.create.return_value = fake_msg
    synth._client = client

    assert synth.call("hi") == "free-form reply"
    kwargs = client.messages.create.call_args.kwargs
    # Plain call doesn't pass output_config.
    assert "output_config" not in kwargs


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


def test_make_synthesizer_returns_each_backend() -> None:
    """The factory dispatches to each concrete subclass."""
    from llmoji.synth import make_synthesizer

    a = make_synthesizer("anthropic")
    o = make_synthesizer("openai")
    L = make_synthesizer(
        "local", base_url="http://x", model_id="llama",
    )
    assert isinstance(a, AnthropicSynthesizer)
    assert isinstance(o, OpenAISynthesizer)
    assert isinstance(L, LocalSynthesizer)
    # call_structured is on the base class so all subclasses
    # implement it (raises NotImplementedError on bare base).
    for s in (a, o, L):
        assert callable(getattr(s, "call_structured", None))


