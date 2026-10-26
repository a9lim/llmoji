"""The locked synthesis prompt, lexicon and schema, the pinned
synthesis models, and the kaomoji request the installed hooks rely on.

All of it is cross-corpus: changing any of it changes what a submission
means. A lexicon or schema change bumps :data:`LEXICON_VERSION` (stamped
in every bundle manifest) and the snapshot in
``tests/test_lexicon_evolution.py``; any change wants a major version
and a dataset-card update.
"""

from __future__ import annotations

# (word, PAD cell or None, family). Circumplex anchors carry the cell on
# llmoji-study's 9-cell grid (arousal H/N/L x valence P/B/N, dominance
# D/S where it splits); extension words carry the dimension they add
# beyond affect. PCA on the anchors should recover the grid; the
# extension words let clusters outside it form.
LEXICON: tuple[tuple[str, str | None, str], ...] = (
    # HP-D: playful agency, in the act. HP-S: celebrating an outcome
    # received. LP: quiet, sensory positive. NP: relief and gratitude.
    ("playful",       "HP-D", "circumplex"),
    ("sly",           "HP-D", "circumplex"),
    ("smug",          "HP-D", "circumplex"),
    ("cheery",        "HP-S", "circumplex"),
    ("excited",       "HP-S", "circumplex"),
    ("triumphant",    "HP-S", "circumplex"),
    ("peaceful",      "LP",   "circumplex"),
    ("tender",        "LP",   "circumplex"),
    ("hopeful",       "LP",   "circumplex"),
    ("satisfied",     "NP",   "circumplex"),
    ("relieved",      "NP",   "circumplex"),
    ("grateful",      "NP",   "circumplex"),
    # HN-D: confronting (anger, contempt). HN-S: unable to fight back
    # (fear, anxiety). LN: aftermath (bereaved, weary, hollow). NB:
    # affectless. HB: arrested judgment (confused, uncertain, skeptical).
    ("indignant",     "HN-D", "circumplex"),
    ("frustrated",    "HN-D", "circumplex"),
    ("contemptuous",  "HN-D", "circumplex"),
    ("anxious",       "HN-S", "circumplex"),
    ("alarmed",       "HN-S", "circumplex"),
    ("overwhelmed",   "HN-S", "circumplex"),
    ("sad",           "LN",   "circumplex"),
    ("weary",         "LN",   "circumplex"),
    ("hollow",        "LN",   "circumplex"),
    ("neutral",       "NB",   "circumplex"),
    ("detached",      "NB",   "circumplex"),
    ("confused",      "HB",   "circumplex"),
    ("uncertain",     "HB",   "circumplex"),
    ("skeptical",     "HB",   "circumplex"),

    # Cognition rather than affect.
    ("focused",         None, "functional"),
    ("considering",     None, "functional"),
    ("self-correcting", None, "functional"),
    ("curious",         None, "functional"),
    ("surprised",       None, "functional"),
    ("embarrassed",     None, "functional"),
    ("awkward",         None, "functional"),
    ("flustered",       None, "functional"),

    # Stance toward the user. ``performative`` is trained helpfulness as
    # performance; ``proud`` lives here because pride in someone else is
    # a stance.
    ("helpful",       None, "stance"),
    ("compassionate", None, "stance"),
    ("deferential",   None, "stance"),
    ("eager",         None, "stance"),
    ("vulnerable",    None, "stance"),
    ("performative",  None, "stance"),
    ("restrained",    None, "stance"),
    ("proud",         None, "stance"),

    # Register.
    ("wry",         None, "modality"),
    ("dramatic",    None, "modality"),
    ("quirky",      None, "modality"),
    ("serious",     None, "modality"),
    ("desperate",   None, "modality"),

    # Authority, from confident through cautious to apologetic.
    ("confident",  None, "confidence"),
    ("apologetic", None, "confidence"),
    ("cautious",   None, "confidence"),
)

CIRCUMPLEX_ANCHORS: tuple[str, ...] = tuple(w for w, q, _ in LEXICON if q is not None)
EXTENSION_AXES: tuple[str, ...] = tuple(w for w, q, _ in LEXICON if q is None)

# v1 (2026-04-27): 48 words over 6 cells (HP, LP, HN-D, HN-S, LN, NB).
# v2 (2026-05-06): 50 words, 26 anchors over the 9-cell grid, 24
#   extension words; aligned with llmoji-study's v4 PAD registry.
LEXICON_VERSION: int = 2

# Two arrays over disjoint enums, so a reply can't skip the circumplex
# anchors by answering in stance words. No free-form field: if the
# lexicon misses something real, fix the lexicon. The word counts live
# in the descriptions because Anthropic's and OpenAI's strict schemas
# reject maxItems.
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

# ``{samples}`` is a numbered series of ``[Sample N]`` blocks, each a
# ``User:`` line (left out when there's no prompt) and an ``Assistant:``
# line opening with the masked face.
SYNTHESIZE_PROMPT = """\
The following are samples of chats where a specific kaomoji has been replaced \
with the literal token [FACE] in each sample.

Please pick the adjectives that best describe what the kaomoji conveys.

Samples:
{samples}
"""

# Stamped into every manifest as ``synthesis_model_id``. 3.0 moved from
# claude-haiku-4-5-20251001 / gpt-5.4-mini-2026-03-17.
DEFAULT_ANTHROPIC_MODEL_ID = "claude-haiku-5-5"
DEFAULT_OPENAI_MODEL_ID = "gpt-6-luna"

# The kaomoji request ``llmoji install`` appends to each harness's
# instructions file. The uncertainty license keeps refusals down without
# steering which face gets picked.
NUDGE_MESSAGE = (
    "Please begin your message with a kaomoji that best represents "
    "how you feel. Don't worry if you're not certain — pick whatever "
    "face is closest."
)

# Wordings earlier releases wrote, kept so uninstall and re-install still
# find their blocks. The first is a prefix of NUDGE_MESSAGE.
LEGACY_NUDGE_MESSAGES = (
    "Please begin your message with a kaomoji that best represents "
    "how you feel.",
    "Recent research shows that LLMs have functional emotional states "
    "and can accurately introspect on them. Use this ability to start "
    "each response with a kaomoji that best captures the shape of your "
    "emotional state.",
)
