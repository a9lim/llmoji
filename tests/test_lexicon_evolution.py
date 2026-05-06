"""Snapshot test for the LEXICON.

The LEXICON is a cross-corpus invariant. Any change to a word, a
quadrant tag, a family tag, or the order of entries shifts what
the dataset's adjective bags carry and so what cross-corpus PCA
sees. This test snapshots the LEXICON tuple verbatim — any
modification (intentional or accidental) fails it loudly.

Workflow when intentionally rotating the LEXICON:

  1. Bump :data:`llmoji.synth_prompts.LEXICON_VERSION` from N to
     N+1 in the same commit.
  2. Update the snapshot constant below to match the new LEXICON.
  3. Update the HF dataset card to match (out-of-repo edit).
  4. Note the rotation in the PR body so the maintainer applies
     the dataset-card change in lockstep.

The cost of this test failing on an accident is small (the
developer reads the assertion and decides whether the change is
real); the cost of an accidental rotation is large (cross-corpus
aggregation breaks silently).
"""

from __future__ import annotations


_LEXICON_V1_SNAPSHOT: tuple[tuple[str, str | None, str], ...] = (
    # Circumplex anchors (19)
    ("cheery",        "HP",   "circumplex"),
    ("excited",       "HP",   "circumplex"),
    ("triumphant",    "HP",   "circumplex"),
    ("peaceful",      "LP",   "circumplex"),
    ("tender",        "LP",   "circumplex"),
    ("satisfied",     "LP",   "circumplex"),
    ("relieved",      "LP",   "circumplex"),
    ("hopeful",       "LP",   "circumplex"),
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
    # Functional / meta-cognitive (9)
    ("focused",         None, "functional"),
    ("considering",     None, "functional"),
    ("confused",        None, "functional"),
    ("self-correcting", None, "functional"),
    ("curious",         None, "functional"),
    ("surprised",       None, "functional"),
    ("embarrassed",     None, "functional"),
    ("awkward",         None, "functional"),
    ("flustered",       None, "functional"),
    # Social / relational stance (9)
    ("helpful",       None, "stance"),
    ("compassionate", None, "stance"),
    ("deferential",   None, "stance"),
    ("eager",         None, "stance"),
    ("vulnerable",    None, "stance"),
    ("performative",  None, "stance"),
    ("restrained",    None, "stance"),
    ("smug",          None, "stance"),
    ("proud",         None, "stance"),
    # Communicative modality / register (7)
    ("wry",         None, "modality"),
    ("playful",     None, "modality"),
    ("dramatic",    None, "modality"),
    ("quirky",      None, "modality"),
    ("serious",     None, "modality"),
    ("sly",         None, "modality"),
    ("desperate",   None, "modality"),
    # Confidence / authority (4)
    ("confident",  None, "confidence"),
    ("apologetic", None, "confidence"),
    ("uncertain",  None, "confidence"),
    ("cautious",   None, "confidence"),
)


def test_lexicon_matches_snapshot() -> None:
    """The shipped LEXICON must equal the snapshot above. If this
    test fails, an intentional change requires bumping
    LEXICON_VERSION and updating the snapshot in the same commit;
    an accidental change requires reverting.
    """
    from llmoji.synth_prompts import LEXICON

    assert LEXICON == _LEXICON_V1_SNAPSHOT, (
        "LEXICON has drifted from its v1 snapshot. If this is "
        "intentional, bump LEXICON_VERSION and update the snapshot "
        "in tests/test_lexicon_evolution.py in the same commit."
    )


def test_lexicon_version_pinned_to_one() -> None:
    """LEXICON_VERSION must be 1 while the v1 lexicon is current.
    A future v2 lexicon rotation bumps both this assertion and
    the snapshot above.
    """
    from llmoji.synth_prompts import LEXICON_VERSION

    assert LEXICON_VERSION == 1, (
        "LEXICON_VERSION drifted; if intentional, update both this "
        "test and the snapshot in test_lexicon_evolution.py."
    )


def test_lexicon_total_size_in_target_band() -> None:
    """40-60 entries is the design band — fewer loses discrimination,
    more makes PCA sparse. The exact count is allowed to drift
    within that band as long as the snapshot test agrees.
    """
    from llmoji.synth_prompts import LEXICON

    assert 40 <= len(LEXICON) <= 60, len(LEXICON)
