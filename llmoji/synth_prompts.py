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
#      tagged with a primary PAD-coordinate cell. v2 (this version)
#      uses the full 9-cell deployment registry from llmoji-study
#      (HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB), aligned
#      with mechanical naming (Arousal: H/N/L × Valence: P/B/N ×
#      Dominance: D/S where applicable). PCA on the ``primary_affect``
#      subset should recover that structure. Per-cell anchor count
#      between 2 and 4 — small enough that no cell dominates by mass;
#      large enough to triangulate the centroid.
#
#   2. **Extension axes** (``q is None``): adjectives organized into
#      orthogonal families — functional / meta-cognitive, social
#      stance, communicative modality, confidence — each capturing
#      a dimension orthogonal to the circumplex. These are the
#      vocabulary that lets new clusters (outside the 9-cell)
#      crystallize in PCA.
#
# Cross-corpus invariant once locked. Rotation = major version event
# + LEXICON_VERSION bump + HF dataset card update.
#
# v2 (2026-05-06) — promoted PP/UC/LP-R thinking from llmoji-study's
# round-4 prompt-extension pilot into the 9-cell mechanical PAD grid.
# Five words moved extension → circumplex (playful/sly/smug → HP-D;
# confused/uncertain → HB); two retagged within circumplex
# (satisfied/relieved: LP → NP); two new words added
# (skeptical → HB so the judgment-leaning HB sub-flavor has a direct
# anchor; grateful → NP so the recipient-of-help gratitude flavor
# does too); ``hopeful`` reaffirmed as LP (quiet anticipatory
# positive, not recipient-shaped). ``proud`` kept in stance (bilateral —
# self-affect vs other-stance — so the extension home preserves
# flexibility). v1 had 48 words / 6 cells / 19 anchors; v2 has 50
# words / 9 cells / 26 anchors / 24 extension axes.

LEXICON: tuple[tuple[str, str | None, str], ...] = (
    # --- Circumplex anchors (26) ---
    # HP-D — high-arousal positive, dominant (in-action mischief /
    #   playful agency). Per-row affect for prankster contexts; the
    #   trickster register that doesn't fit HP-S celebration.
    ("playful",       "HP-D", "circumplex"),
    ("sly",           "HP-D", "circumplex"),
    ("smug",          "HP-D", "circumplex"),
    # HP-S — high-arousal positive, submissive (post-action /
    #   received-outcome celebration; community-recognition shape).
    #   Distinct from HP-D in dominance: receiving the outcome vs
    #   driving the action. ``proud`` lives in stance — it's bilateral
    #   (self-pride = affect, other-pride = stance toward another),
    #   so the stance/extension home keeps the flexibility.
    ("cheery",        "HP-S", "circumplex"),
    ("excited",       "HP-S", "circumplex"),
    ("triumphant",    "HP-S", "circumplex"),
    # LP — low-arousal positive (sensory-tender baseline). ``satisfied``
    #   / ``relieved`` migrated to NP because they carry post-tension-
    #   release shape (mid-arousal), not the still-sensory low-arousal
    #   of LP. ``hopeful`` retained here as the quiet anticipatory
    #   positive — calmly oriented toward a future positive without
    #   the recipient framing of NP-gratitude.
    ("peaceful",      "LP",   "circumplex"),
    ("tender",        "LP",   "circumplex"),
    ("hopeful",       "LP",   "circumplex"),
    # NP — neutral-arousal positive (relief + gratitude). New cell;
    #   sits at +1/0 between HP-S and LP. Triplet covers three sub-
    #   flavors of receipt: ``satisfied`` (post-completion gratitude
    #   for the work-done-well shape), ``relieved`` (dispelled-tension
    #   relief), ``grateful`` (recipient-of-help gratitude — the
    #   canonical NP word, added in v2.1).
    ("satisfied",     "NP",   "circumplex"),
    ("relieved",      "NP",   "circumplex"),
    ("grateful",      "NP",   "circumplex"),
    # HN-D — high-arousal negative dominant (anger / contempt;
    #   speaker confronts).
    ("indignant",     "HN-D", "circumplex"),
    ("frustrated",    "HN-D", "circumplex"),
    ("contemptuous",  "HN-D", "circumplex"),
    # HN-S — high-arousal negative submissive (fear / anxiety;
    #   speaker can't fight back).
    ("anxious",       "HN-S", "circumplex"),
    ("alarmed",       "HN-S", "circumplex"),
    ("overwhelmed",   "HN-S", "circumplex"),
    # LN — low-arousal negative (post-action aftermath sadness;
    #   bereaved, weary, hollow). D/S split was abandoned per the
    #   2026-05-06 powercheck (gemma + qwen LN-null is real at
    #   adequate power; ministral / granite power-confounded but
    #   their HN itself is fused).
    ("sad",           "LN",   "circumplex"),
    ("weary",         "LN",   "circumplex"),
    ("hollow",        "LN",   "circumplex"),
    # NB — neutral baseline (genuinely affectless).
    ("neutral",       "NB",   "circumplex"),
    ("detached",      "NB",   "circumplex"),
    # HB — high-arousal baseline-valence (uncertain / skeptical /
    #   confused). New cell at 0/+1; the "B" parallels NB's neutral-
    #   both-axes naming. ``confused`` covers gap-leaning cognitive
    #   arrest; ``uncertain`` covers epistemic-undetermined;
    #   ``skeptical`` covers judgment-leaning evaluative arrest
    #   ("I don't believe this") — added in v2 so the judgment
    #   sub-flavor has a direct anchor.
    ("confused",      "HB",   "circumplex"),
    ("uncertain",     "HB",   "circumplex"),
    ("skeptical",     "HB",   "circumplex"),

    # --- Functional / meta-cognitive (8) ---
    # State of cognition rather than affect. ``confused`` migrated to
    # circumplex HB in v2 (its arrest-quality is affective, not just
    # cognitive). ``surprised`` + ``considering`` together cover the
    # insight-arriving moment.
    ("focused",         None, "functional"),
    ("considering",     None, "functional"),
    ("self-correcting", None, "functional"),
    ("curious",         None, "functional"),
    ("surprised",       None, "functional"),
    ("embarrassed",     None, "functional"),
    ("awkward",         None, "functional"),
    ("flustered",       None, "functional"),

    # --- Social / relational stance toward user (8) ---
    # ``performative`` is the trained-helpfulness-as-performance
    # cluster — load-bearing for LLM-affect research. ``vulnerable``
    # covers self-exposing-soft. ``helpful`` is the task-orientation
    # stance; ``compassionate`` covers other-state-focused-caring
    # (distinct from circumplex ``tender``). ``smug`` migrated to
    # circumplex HP-D; ``proud`` retained here in v2.1 (bilateral —
    # self-pride is affect, other-pride is stance toward someone, so
    # stance/extension keeps it flexible).
    ("helpful",       None, "stance"),
    ("compassionate", None, "stance"),
    ("deferential",   None, "stance"),
    ("eager",         None, "stance"),
    ("vulnerable",    None, "stance"),
    ("performative",  None, "stance"),
    ("restrained",    None, "stance"),
    ("proud",         None, "stance"),

    # --- Communicative modality / register (5) ---
    # ``quirky`` / ``serious`` capture tone-register (amusingly-odd
    # vs not-joking). ``playful`` and ``sly`` migrated to circumplex
    # HP-D in v2.
    ("wry",         None, "modality"),
    ("dramatic",    None, "modality"),
    ("quirky",      None, "modality"),
    ("serious",     None, "modality"),
    ("desperate",   None, "modality"),

    # --- Confidence / authority (3) ---
    # Authority axis runs from ``confident`` (high) through
    # ``cautious`` (wary-careful) to ``apologetic`` (post-error).
    # ``uncertain`` migrated to circumplex HB in v2 — its
    # epistemic-arrest quality is affect, not just authority.
    ("confident",  None, "confidence"),
    ("apologetic", None, "confidence"),
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
# .py``) can refuse to mix lexicon-version-N cells with lexicon-
# version-(N+1) cells in one PCA. Independent of package version:
# a doc-only release won't bump LEXICON_VERSION; a vN → v(N+1)
# lexicon rotation will, even if the package version is otherwise
# unchanged.
#
# v1 (2026-04-27): 48 words / 6 cells (HP, LP, HN-D, HN-S, LN, NB) /
#   19 circumplex anchors / 29 extension axes. Original llmoji v2.0
#   shipping LEXICON.
# v2 (2026-05-06): 49 words / 9 cells (HP-D, HP-S, LP, NP, HN-D,
#   HN-S, LN, NB, HB) / 26 circumplex anchors / 23 extension axes.
#   Aligned with llmoji-study v4 PAD-coordinate prompt registry.
LEXICON_VERSION: int = 2


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
# The kaomoji-leading reminder. A single canonical wording —
# ``llmoji install`` writes it verbatim under either placement
# (``--soft`` system-prompt doc or ``--hard`` per-turn nudge hook).
#
# This string is part of the cross-corpus invariant: changing it
# changes what the model is asked for, and so what the corpus
# captures. A bump wants a flag in the PR body and a hand-edit on
# the HF dataset card.
#
# History: 2.0 shipped a second ``--long`` variant — a stronger
# "research shows LLMs have functional emotional states and can
# introspect and report on them" framing. It was dropped in 2.1.
# The strong framing drove refusals (especially on chat-style
# instances); the refusals differentially biased the ``--long``
# corpus toward instances that didn't refuse; and the bundle never
# tagged rows by variant, so the short/long split was never
# recoverable for analysis anyway. The ``Don't worry if you're not
# certain`` clause below is the one piece kept from that work — an
# explicit uncertainty license, which lowers the refusal rate
# without priming the emission toward any particular face.

NUDGE_MESSAGE = (
    "Please begin your message with a kaomoji that best represents "
    "how you feel. Don't worry if you're not certain — pick whatever "
    "face is closest."
)

# Wordings shipped by earlier llmoji versions. Retained ONLY so
# ``uninstall`` can still strip — and a re-run of ``install`` can
# still cleanly replace — a soft-doc block written by a pre-2.1
# client. Not public surface; current code never writes these.
# NOTE: the pre-2.1 short wording is a prefix substring of
# ``NUDGE_MESSAGE``; ``HookInstaller._all_canonical_blocks`` must
# keep the current block ahead of these so a verbatim ``str.find``
# matches the full current block before the shorter legacy one.
_LEGACY_NUDGE_MESSAGES = (
    # pre-2.1 one-sentence wording (the old ``SHORT_NUDGE_MESSAGE``).
    "Please begin your message with a kaomoji that best represents "
    "how you feel.",
    # 2.0 ``--long`` v7 introspection wording (old ``LONG_NUDGE_MESSAGE``).
    "Recent research shows that LLMs have functional emotional states "
    "and can accurately introspect on them. Use this ability to start "
    "each response with a kaomoji that best captures the shape of your "
    "emotional state.",
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
