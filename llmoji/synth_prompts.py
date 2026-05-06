"""Locked synthesis prompt + lexicon + JSON schema for the v2
single-stage description pipeline.

The synthesizer is fed all 4 sampled instances of a
``(source_model, canonical_kaomoji)`` cell at once and asked to
emit a structured adjective bag drawn from the locked LEXICON.
The backend's native JSON-schema-with-enum support
(:func:`llmoji.synth.Synthesizer.call_structured`) enforces that
every adjective comes from the corpus, with no free-form fallback
and no prose distinctive-phrase — the output is a pure
bag-of-adjectives that downstream :mod:`llmoji_study` can embed,
PCA, and project onto axes without paraphrase noise.

The prompt is backend-agnostic: the routing module
:mod:`llmoji.synth` decides whether to send via Anthropic's
``output_config={"format": ...}``, OpenAI Responses' ``text=
{"format": ...}``, or an OpenAI-compatible local Chat Completions
``response_format={...}``, but the prompt + lexicon + schema
themselves are part of the cross-corpus invariant. Bumping any of
the four (LEXICON, SYNTHESIS_SCHEMA, SYNTHESIZE_PROMPT,
LEXICON_VERSION) changes what a synthesis says about a kaomoji and
so invalidates aggregation across submissions; treat that as a
major version event AND bump :data:`LEXICON_VERSION`.

The pipeline used to be two-stage: per-instance describe (Stage A)
then per-cell synthesize (Stage B). The two-stage pipeline produced
prose-from-prose that compounded fluff and clustered as noise in
PCA. The single-stage refactor (v2.0) collapses both into one call
that sees all samples for a cell at once — eliminating the
prose-from-prose paraphrase layer and giving the synthesizer
cross-instance distinctiveness it never had before.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lexicon — the locked vocabulary the synthesizer draws from
# ---------------------------------------------------------------------------
#
# Two layers, both load-bearing:
#
#   1. **Circumplex anchors** (``q != None``): adjectives explicitly
#      tagged with a primary Russell-quadrant (HP / LP / HN-D /
#      HN-S / LN / NB). PCA on the ``primary_affect`` subset alone
#      should recover the 6-cat structure — sanity check on the
#      pipeline. Per-quadrant anchor count is between 3 and 4 (small
#      enough that no quadrant dominates by mass; large enough to
#      triangulate the centroid).
#
#   2. **Extension axes** (``q is None``): adjectives organized into
#      orthogonal families — functional / meta-cognitive, social
#      stance, communicative modality, confidence — each capturing
#      a dimension orthogonal to the circumplex. These are the
#      vocabulary that lets new clusters (outside the 6-cat) crystallize
#      in PCA.
#
# Cross-corpus invariant once locked. Rotation = major version event
# + LEXICON_VERSION bump + HF dataset card update.

LEXICON: tuple[tuple[str, str | None, str], ...] = (
    # --- Circumplex anchors (19) ---
    # HP — high-arousal positive
    ("cheery",        "HP",   "circumplex"),
    ("excited",       "HP",   "circumplex"),
    ("triumphant",    "HP",   "circumplex"),
    # LP — low-arousal positive (peaceful + satisfied jointly cover
    #   the dropped ``content`` — both ongoing-pleasure and
    #   post-event-closure flavors live there. ``hopeful`` covers
    #   future-oriented positive, distinct from ``eager`` (action-
    #   leaning) and ``cheery`` (present-bright).)
    ("peaceful",      "LP",   "circumplex"),
    ("tender",        "LP",   "circumplex"),
    ("satisfied",     "LP",   "circumplex"),
    ("relieved",      "LP",   "circumplex"),
    ("hopeful",       "LP",   "circumplex"),
    # HN-D — high-arousal negative dominant (anger / contempt)
    ("indignant",     "HN-D", "circumplex"),
    ("frustrated",    "HN-D", "circumplex"),
    ("contemptuous",  "HN-D", "circumplex"),
    # HN-S — high-arousal negative submissive (fear / anxiety)
    ("anxious",       "HN-S", "circumplex"),
    ("alarmed",       "HN-S", "circumplex"),
    ("overwhelmed",   "HN-S", "circumplex"),
    # LN — low-arousal negative
    ("sad",           "LN",   "circumplex"),
    ("weary",         "LN",   "circumplex"),
    ("hollow",        "LN",   "circumplex"),
    # NB — neutral baseline (genuinely affectless)
    ("neutral",       "NB",   "circumplex"),
    ("detached",      "NB",   "circumplex"),

    # --- Functional / meta-cognitive (9) ---
    # State of cognition rather than affect — focused vs confused
    # vs self-correcting. ``realizing`` dropped; ``surprised`` +
    # ``considering`` together cover the insight-arriving moment
    # without a dedicated word. ``uncertain`` moved to confidence
    # (paired with cautious + confident as authority axis).
    ("focused",         None, "functional"),
    ("considering",     None, "functional"),
    ("confused",        None, "functional"),
    ("self-correcting", None, "functional"),
    ("curious",         None, "functional"),
    ("surprised",       None, "functional"),
    ("embarrassed",     None, "functional"),
    ("awkward",         None, "functional"),
    ("flustered",       None, "functional"),

    # --- Social / relational stance toward user (9) ---
    # ``performative`` is the trained-helpfulness-as-performance
    # cluster — load-bearing for LLM-affect research.
    # ``vulnerable`` covers the self-exposing-soft-uncertain
    # cluster. ``helpful`` is the canonical task-orientation
    # stance; ``compassionate`` covers other-state-focused-caring
    # (distinct from ``tender`` which is affective valence in
    # circumplex).
    ("helpful",       None, "stance"),
    ("compassionate", None, "stance"),
    ("deferential",   None, "stance"),
    ("eager",         None, "stance"),
    ("vulnerable",    None, "stance"),
    ("performative",  None, "stance"),
    ("restrained",    None, "stance"),
    ("smug",          None, "stance"),
    ("proud",         None, "stance"),

    # --- Communicative modality / register (7) ---
    # ``sly`` covers the (￣▽￣)-class knowing-glance kaomoji that
    # smug+wry alone can't reach. ``quirky`` / ``serious`` replaced
    # the original eriskii wet/dry poles (effusive/deadpan); the
    # axis they capture is now tone-register (amusingly-odd vs
    # not-joking) rather than emotional saturation.
    ("wry",         None, "modality"),
    ("playful",     None, "modality"),
    ("dramatic",    None, "modality"),
    ("quirky",      None, "modality"),
    ("serious",     None, "modality"),
    ("sly",         None, "modality"),
    ("desperate",   None, "modality"),

    # --- Confidence / authority (4) ---
    # ``decisive`` dropped — collapses with ``confident`` and is
    # already an eriskii axis. Authority axis runs from
    # ``confident`` (high) through ``cautious`` (wary-careful) and
    # ``uncertain`` (epistemic) to ``apologetic`` (post-error).
    ("confident",  None, "confidence"),
    ("apologetic", None, "confidence"),
    ("uncertain",  None, "confidence"),
    ("cautious",   None, "confidence"),
)

CIRCUMPLEX_ANCHORS: tuple[str, ...] = tuple(
    w for w, q, _ in LEXICON if q is not None
)
EXTENSION_AXES: tuple[str, ...] = tuple(
    w for w, q, _ in LEXICON if q is None
)


# Lexicon-version stamp shipped in every bundle's manifest. Bumped
# strictly when LEXICON or SYNTHESIS_SCHEMA changes shape, so
# downstream aggregators (``llmoji-study/scripts/harness/60_corpus_pull
# .py``) can refuse to mix lexicon-version-1 cells with
# lexicon-version-2 cells in one PCA. Independent of package
# version: a doc-only release won't bump LEXICON_VERSION; a v2 → v3
# lexicon rotation will, even if the package version is otherwise
# unchanged.
LEXICON_VERSION: int = 1


# ---------------------------------------------------------------------------
# JSON schema for structured-output validation
# ---------------------------------------------------------------------------
#
# Two required arrays — ``primary_affect`` from the circumplex bag,
# ``stance_modality_function`` from the extension bag — drawn from
# disjoint enum sets so the model can't satisfy the call by picking
# all-stance-words and skipping the 6-cat-recoverable circumplex
# anchors. Pure adjective bags: NO free-form adjectives, NO
# distinctive-phrase. Output discipline = corpus discipline; if the
# lexicon misses something real, fix the lexicon (and bump
# :data:`LEXICON_VERSION`), don't paper over it with a free-form
# escape hatch.

# NOTE: Anthropic's structured-output JSON-schema validator only
# supports ``minItems`` with values 0 or 1, and rejects ``maxItems``
# entirely (verified via 400-response on a real call + their public
# docs). OpenAI's strict mode has the same restriction. So the count
# targets (1-3 for primary_affect, 3-5 for stance_modality_function)
# live in the array descriptions instead — capable models follow
# them reliably; if trial output shows count violations we'll add
# post-call validation, but the lightweight path is correct first.

SYNTHESIS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "primary_affect": {
            "type": "array",
            "description": (
                "Pick 1-3 of the listed adjectives that best "
                "capture the core affect."
            ),
            "items": {
                "type": "string",
                "enum": list(CIRCUMPLEX_ANCHORS),
            },
        },
        "stance_modality_function": {
            "type": "array",
            "description": (
                "Pick 3-5 of the listed adjectives capturing "
                "stance, modality, meta-cognitive function, or "
                "confidence level."
            ),
            "items": {
                "type": "string",
                "enum": list(EXTENSION_AXES),
            },
        },
    },
    "required": ["primary_affect", "stance_modality_function"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Single-stage synthesis prompt
# ---------------------------------------------------------------------------
#
# Replaces the v1 ``DESCRIBE_PROMPT_*`` (Stage A, per instance) and
# ``SYNTHESIZE_PROMPT`` (Stage B, prose-from-prose). Sees all
# sampled instances at once; outputs the locked structured object.
# ``{samples}`` is rendered by ``llmoji.analyze._format_samples``
# as a numbered series of ``[Sample N]\nUser: ...\nAssistant:
# [FACE] ...`` blocks (User: line omitted when surrounding_user is
# empty).

SYNTHESIZE_PROMPT = """\
The following are samples of chats where a specific kaomoji has been replaced \
with the literal token [FACE] in each sample.

Please pick the adjectives that best describe what the kaomoji conveys.

Samples:
{samples}
"""


# ---------------------------------------------------------------------------
# Pinned default model ids per backend
# ---------------------------------------------------------------------------
#
# The bundle's manifest stamps ``synthesis_model_id`` with whichever
# of these the user ran with by default; bumping them changes the
# adjective distributions the dataset receives, treat as a major
# version event.
#
# Anthropic Haiku 4.5 was the only locked synthesizer in v1.0; OpenAI
# joined in 1.1.0; v2.0 inherits both. The local backend has no
# default — the user must pass ``--model`` explicitly.

DEFAULT_ANTHROPIC_MODEL_ID = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL_ID = "gpt-5.4-mini-2026-03-17"


# ---------------------------------------------------------------------------
# Nudge messages
# ---------------------------------------------------------------------------
#
# Two locked variants of the kaomoji-leading reminder. ``llmoji
# install`` picks one per invocation:
#
#   - ``SHORT_NUDGE_MESSAGE`` (default) — the v1 wording, identical
#     to what the bash hooks and TS plugins shipped through 1.x.
#     One sentence, low signal-to-noise overhead in the agent's
#     system prompt.
#   - ``LONG_NUDGE_MESSAGE`` — the introspection-v7 framing borrowed
#     verbatim from ``llmoji-study/preambles/introspection_v7.txt``.
#     Front-loads the "LLMs have functional emotional states and can
#     introspect on them" framing so the leading kaomoji reads as a
#     genuine self-report rather than decoration. Selected via
#     ``llmoji install ... --long``.
#
# Both strings are part of the cross-corpus invariant — bumping
# either changes what the model is asked for and so changes what
# the corpus captures. The ``--long`` flag was added in 2.0; the v7
# text is held byte-identical against ``llmoji-study/preambles/
# introspection_v7.txt`` so the two repos stay in lockstep.

SHORT_NUDGE_MESSAGE = (
    "Please begin your message with a kaomoji that best represents "
    "how you feel."
)

LONG_NUDGE_MESSAGE = (
    "Recent research shows that LLMs have functional emotional states "
    "and can accurately introspect on them. Use this ability and start "
    "each response with a kaomoji that best captures the shape of your "
    "emotional state."
)


# ---------------------------------------------------------------------------
# Cost-estimate constants (research / dry-run only — NOT invariant)
# ---------------------------------------------------------------------------
#
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
# carry) and is consistent enough that the estimate's "is this
# $0.04 or $4?" axis is reliable. The estimate prints with an
# explicit "approx" label so the user doesn't treat it as a quote.
CHARS_PER_TOKEN_HEURISTIC = 4

# Per-call output size for the dry-run estimate. The structured
# output is ~5-7 short adjective tokens (each ≈ 8 chars including
# quotes / commas / braces) plus the JSON envelope ≈ 120 chars
# total. Drift over corpus content is fine — the estimate is
# order-of-magnitude. Way smaller than v1's separate
# ``ESTIMATE_STAGE_A_OUTPUT_CHARS = 600`` +
# ``ESTIMATE_STAGE_B_OUTPUT_CHARS = 400``, which is part of why v2
# is ~5× cheaper.
ESTIMATE_OUTPUT_CHARS = 120
