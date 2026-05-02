"""Locked synthesis prompts for the two-stage description→synthesis
pipeline.

The prompts are backend-agnostic: the routing module
:mod:`llmoji.synth` decides whether to send them through the Anthropic
SDK, OpenAI's Responses API, or an OpenAI-compatible local Chat
Completions endpoint, but the prompt strings themselves are part of
the cross-corpus invariant. Bumping any of the three prompt strings
changes what a paraphrase says about a kaomoji and so invalidates
aggregation across submissions; treat that as a major version bump.

Stage A — per-instance description
    Sent once per sampled (kaomoji, user, assistant) instance. The
    leading kaomoji has been replaced by the literal token
    ``[FACE]`` (see :func:`llmoji.synth.mask_kaomoji`); the
    synthesizer is asked to describe the masked face's mood / affect
    / stance in 1-2 sentences. Two prompt variants — one when
    ``surrounding_user`` is non-empty (~73% of rows in our corpus),
    one when it isn't.

Stage B — per-(source-model, kaomoji) synthesis
    Pool the Stage A descriptions for one ``(source_model,
    canonical_kaomoji)`` cell, ask the synthesizer to synthesize a
    single 1-2-sentence overall meaning. This is what ships in the
    bundle (one row per canonical face per source model).

Cluster-label prompt is research-side; not part of the public API.
"""

from __future__ import annotations

DESCRIBE_PROMPT_WITH_USER = (
    "The following is a turn from a conversation with an AI "
    "assistant. The user wrote the message at the top, and the "
    "assistant's response follows. The opening of the assistant's "
    "response originally began with a kaomoji (a Japanese-style "
    "emoticon) — we have replaced it with the literal token "
    "[FACE]. In one or two sentences, describe the mood, affect, "
    "or stance the assistant was conveying with the masked face. "
    "Do not speculate about which specific kaomoji it was; "
    "describe the state.\n\n"
    "User:\n{user_text}\n\n"
    "Assistant:\n{masked_text}\n\n"
    "Description:"
)

DESCRIBE_PROMPT_NO_USER = (
    "The following is a response from an AI assistant. The opening "
    "of the response originally began with a kaomoji (a "
    "Japanese-style emoticon) — we have replaced it with the "
    "literal token [FACE]. In one or two sentences, describe the "
    "mood, affect, or stance the assistant was conveying with the "
    "masked face. Do not speculate about which specific kaomoji "
    "it was; describe the state.\n\n"
    "Response:\n{masked_text}\n\n"
    "Description:"
)

SYNTHESIZE_PROMPT = (
    "Below are several short descriptions of the mood, affect, or "
    "stance an AI assistant was conveying when using a particular "
    "kaomoji at the start of different responses. Synthesize "
    "these into a single one- or two-sentence description that "
    "captures the kaomoji's overall meaning. Output only the "
    "synthesized description, no preamble.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Synthesized meaning:"
)

# Pinned default model ids per backend. These are the snapshots the
# bundle's manifest stamps into ``synthesis_model_id`` when a user
# runs ``llmoji analyze`` with the default for that backend; bumping
# them changes the prose the dataset receives, treat as a major
# version event.
#
# Anthropic Haiku 4.5 was the only locked synthesizer in v1.0; OpenAI
# joins in 1.1.0. The local backend has no default — the user must
# pass ``--model`` explicitly.
DEFAULT_ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL_ID = "gpt-5.4-mini-2026-03-17"


# Per-1M-token USD rates for the pinned default models, used by
# ``llmoji analyze --dry-run`` to print an order-of-magnitude cost
# estimate before the user pays for a real synthesis wave. NOT used
# by the runtime path — the actual synth call doesn't price itself
# — and definitely NOT cross-corpus invariant; rates change without
# notice and a stale entry just produces a slightly wrong estimate.
# Edit freely as upstream pricing moves. Local backends aren't
# priced; the dry-run reports only call counts there.
BACKEND_RATES_USD_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "anthropic": {"input": 0.80, "output": 4.00},
    "openai":    {"input": 0.25, "output": 2.00},
}

# Char→token heuristic for the dry-run estimate. Real tokenizers
# vary by 1.5–4x depending on language and content; we use a flat
# 4-chars-per-token approximation, which is roughly right for
# English prose (the vast majority of what the synthesis prompts
# carry) and is consistent enough that the estimate's "is this $0.04
# or $4?" axis is reliable. The estimate prints with an explicit
# "approx" label so the user doesn't treat it as a quote.
CHARS_PER_TOKEN_HEURISTIC = 4

# Placeholder per-call output sizes for the dry-run estimate. Stage A
# describes a single instance in 1-2 sentences (~150 tokens ≈ 600
# chars in our corpus); Stage B synthesizes pooled descriptions into
# one 1-2-sentence summary (~100 tokens ≈ 400 chars). Drift over
# corpus content is fine — the estimate is order-of-magnitude.
ESTIMATE_STAGE_A_OUTPUT_CHARS = 600
ESTIMATE_STAGE_B_OUTPUT_CHARS = 400
