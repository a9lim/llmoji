"""Locked Haiku prompts for the two-stage description→synthesis
pipeline.

These prompts are part of the v1.0 frozen public API. Bumping any
string here changes the description text that ships in the bundle,
which would invalidate cross-corpus aggregation against the central
HF dataset; treat changes as a major version bump (``llmoji`` 2.0.0).

Stage A — per-instance description
    Sent once per sampled (kaomoji, user, assistant) instance. The
    leading kaomoji has been replaced by the literal token
    ``[FACE]`` (see :func:`llmoji.haiku.mask_kaomoji`); Haiku is
    asked to describe the masked face's mood/affect/stance in 1-2
    sentences. Two prompt variants — one when ``surrounding_user``
    is non-empty (~73% of rows in our corpus), one when it isn't.

Stage B — per-kaomoji synthesis
    Pool the Stage A descriptions for one canonical kaomoji form,
    ask Haiku to synthesize a single 1-2-sentence overall meaning.
    This is what ships in the bundle (one row per canonical face).

Cluster-label prompt is research-side; not part of v1.0 public API.
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

# Locked Haiku model id for the v1.0 corpus. Bumping invalidates
# description-corpus parity across submissions; treated as a major
# version bump.
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"
