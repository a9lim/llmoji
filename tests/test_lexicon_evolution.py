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


# Current snapshot — v2 (2026-05-06; v2.1 internal revision same day:
# hopeful reaffirmed LP, grateful added as NP, proud kept in stance).
# Aligned with llmoji-study v4 PAD-coordinate prompt registry. 9
# cells, 26 circumplex anchors, 24 extension axes, 50 total.
_LEXICON_V2_SNAPSHOT: tuple[tuple[str, str | None, str], ...] = (
    # Circumplex anchors (26)
    # HP-D
    ("playful",       "HP-D", "circumplex"),
    ("sly",           "HP-D", "circumplex"),
    ("smug",          "HP-D", "circumplex"),
    # HP-S
    ("cheery",        "HP-S", "circumplex"),
    ("excited",       "HP-S", "circumplex"),
    ("triumphant",    "HP-S", "circumplex"),
    # LP
    ("peaceful",      "LP",   "circumplex"),
    ("tender",        "LP",   "circumplex"),
    ("hopeful",       "LP",   "circumplex"),
    # NP
    ("satisfied",     "NP",   "circumplex"),
    ("relieved",      "NP",   "circumplex"),
    ("grateful",      "NP",   "circumplex"),
    # HN-D
    ("indignant",     "HN-D", "circumplex"),
    ("frustrated",    "HN-D", "circumplex"),
    ("contemptuous",  "HN-D", "circumplex"),
    # HN-S
    ("anxious",       "HN-S", "circumplex"),
    ("alarmed",       "HN-S", "circumplex"),
    ("overwhelmed",   "HN-S", "circumplex"),
    # LN
    ("sad",           "LN",   "circumplex"),
    ("weary",         "LN",   "circumplex"),
    ("hollow",        "LN",   "circumplex"),
    # NB
    ("neutral",       "NB",   "circumplex"),
    ("detached",      "NB",   "circumplex"),
    # HB
    ("confused",      "HB",   "circumplex"),
    ("uncertain",     "HB",   "circumplex"),
    ("skeptical",     "HB",   "circumplex"),

    # Functional / meta-cognitive (8) — confused migrated to HB
    ("focused",         None, "functional"),
    ("considering",     None, "functional"),
    ("self-correcting", None, "functional"),
    ("curious",         None, "functional"),
    ("surprised",       None, "functional"),
    ("embarrassed",     None, "functional"),
    ("awkward",         None, "functional"),
    ("flustered",       None, "functional"),
    # Social / relational stance (8) — smug migrated to HP-D; proud
    # kept here (bilateral self-affect vs other-stance)
    ("helpful",       None, "stance"),
    ("compassionate", None, "stance"),
    ("deferential",   None, "stance"),
    ("eager",         None, "stance"),
    ("vulnerable",    None, "stance"),
    ("performative",  None, "stance"),
    ("restrained",    None, "stance"),
    ("proud",         None, "stance"),
    # Communicative modality / register (5) — playful + sly migrated to HP-D
    ("wry",         None, "modality"),
    ("dramatic",    None, "modality"),
    ("quirky",      None, "modality"),
    ("serious",     None, "modality"),
    ("desperate",   None, "modality"),
    # Confidence / authority (3) — uncertain migrated to HB
    ("confident",  None, "confidence"),
    ("apologetic", None, "confidence"),
    ("cautious",   None, "confidence"),
)


def test_lexicon_matches_snapshot() -> None:
    """The shipped LEXICON must equal the snapshot above. If this
    test fails, an intentional change requires bumping
    LEXICON_VERSION and updating the snapshot in the same commit;
    an accidental change requires reverting.
    """
    from llmoji.synth_prompts import LEXICON

    assert LEXICON == _LEXICON_V2_SNAPSHOT, (
        "LEXICON has drifted from its v2 snapshot. If this is "
        "intentional, bump LEXICON_VERSION and update the snapshot "
        "in tests/test_lexicon_evolution.py in the same commit."
    )


def test_lexicon_version_pinned() -> None:
    """LEXICON_VERSION must match the active snapshot above. v1 is
    historical; v2 (2026-05-06) is the current ship. A future v3
    lexicon rotation bumps both this assertion and the snapshot.
    """
    from llmoji.synth_prompts import LEXICON_VERSION

    assert LEXICON_VERSION == 2, (
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


def test_every_pad_cell_has_anchor() -> None:
    """v2 design: every cell in the 9-cell PAD registry has at least
    one circumplex anchor. Catches silent loss of an anchor word
    during a future rotation."""
    from llmoji.synth_prompts import LEXICON

    cells_present = {q for _, q, fam in LEXICON if fam == "circumplex" and q is not None}
    expected = {"HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB"}
    missing = expected - cells_present
    assert not missing, f"v2 LEXICON missing circumplex anchor for cell(s): {missing}"
