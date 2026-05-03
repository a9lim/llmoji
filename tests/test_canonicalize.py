"""Per-rule regression tests for the v1.0-locked canonicalization
and extraction logic.

Pre-package, these lived as a ``sanity_check()`` function inside
``llmoji/taxonomy.py``, runnable only via
``python -m llmoji.taxonomy``. Lifted into pytest with
``parametrize`` so:

  * Every rule case is its own pytest line — failures point at the
    specific input that broke, with diff against the expected value.
  * The full corpus runs in CI alongside ``test_public_surface.py``.
  * Adding a new corpus example is one line, not one ``assert``
    indented inside a 100-line function.

The v1.0 frozen public surface (``KAOMOJI_START_CHARS``, the rules
A–P, ``is_kaomoji_candidate`` / ``extract`` contracts) is the
invariant these tests pin.
"""

from __future__ import annotations

import pytest

from llmoji.taxonomy import (
    KAOMOJI_START_CHARS,
    canonicalize_kaomoji,
    extract,
    is_kaomoji_candidate,
)


# ---------------------------------------------------------------------------
# extract — leading-kaomoji span identification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("(｡◕‿◕｡) I had a great day!",  "(｡◕‿◕｡)"),
        ("(｡•́︿•̀｡) That's so sad.",       "(｡•́︿•̀｡)"),
        ("  (✿◠‿◠) hi",                  "(✿◠‿◠)"),
        # Whitespace-padded face: surfaces with internal whitespace intact.
        ("(｡˃ ᵕ ˂ ) That is wonderful!", "(｡˃ ᵕ ˂ )"),
        # Bracket-span fallback for an unknown paren form (real
        # kaomoji-shape — used to be label=0/"other" in the legacy API).
        ("(｡o_O｡) strange",              "(｡o_O｡)"),
        # v2.0: wing-hand pattern. Backslash at position 0 + closing
        # ``)/`` is the celebratory wing form. Surfaces via the
        # whitespace-fallback branch (no internal spaces).
        ("\\(^o^)/ awesome!",             "\\(^o^)/"),
        ("\\(≧▽≦)/ YESSS",                "\\(≧▽≦)/"),
        # v2.0: sparkle-decorated. Leading ``✧`` + trailing
        # decoration; whitespace-fallback grabs the whole token.
        ("✧*｡(ˊᗜˋ*)*｡✧ wow",              "✧*｡(ˊᗜˋ*)*｡✧"),
        # v2.0 sweep: bear face. ``ʕ``/``ʔ`` are paired brackets
        # in the depth-walker (added to _OPEN/_CLOSE_BRACKETS).
        ("ʕ•ᴥ•ʔ hey",                     "ʕ•ᴥ•ʔ"),
        # v2.0 sweep: shocked sigma. Single-arm leader ``Σ`` —
        # whitespace-fallback grabs the whole span.
        ("Σ(°△°|||) shock!",              "Σ(°△°|||)"),
        # v2.0 sweep: horn-fingers (lowercase + capital psi pairs).
        ("ψ(`Д´)ψ angry",                 "ψ(`Д´)ψ"),
        ("Ψ(`Д´)Ψ furious",               "Ψ(`Д´)Ψ"),
        # v2.0 sweep: kissing pair (ε + з).
        ("ε(◕‿◕)з kiss",                  "ε(◕‿◕)з"),
        # v2.0 sweep: raised hands (ƪ + ʃ).
        ("ƪ(˘⌣˘)ʃ yay",                   "ƪ(˘⌣˘)ʃ"),
        # v2.0 sweep: heavy-line wing-hand variant.
        ("╲(◕‿◕)╱ celebrate",             "╲(◕‿◕)╱"),
        # v2.0 sweep: hug-pair with mirrored close ⊃.
        ("⊂(◕‿◕)⊃ hug",                   "⊂(◕‿◕)⊃"),
        # v2.0 sweep: cheering pair ٩…۶.
        ("٩(◕‿◕)۶ woot",                  "٩(◕‿◕)۶"),
        # v2.0 sweep: cradling pair ໒…७.
        ("໒(◕‿◕)७ aww",                   "໒(◕‿◕)७"),
    ],
)
def test_extract_positive(text: str, expected: str) -> None:
    assert extract(text).first_word == expected


@pytest.mark.parametrize(
    "text",
    [
        # Plain prose — non-kaomoji input returns empty.
        "hello!",
        "",
        # Parenthesized prose with 4+-letter run → rejected.
        "(Backgrounddebugscript) trailing",
        # Bracketed phrase with internal letters → rejected.
        "[pre-commit] passed",
        # Markdown-escape backslash → rejected.
        "(\\*´∀｀\\*) hello",
        # Oversize balanced span → rejected.
        "(" + "a" * 50 + ") text",
    ],
)
def test_extract_rejects(text: str) -> None:
    assert extract(text).first_word == ""


@pytest.mark.parametrize(
    "text,expected_first_word",
    [
        # Unbalanced bracket-leading kaomoji — the depth walker can't
        # close, but the whitespace-fallback grabs the first word so
        # we don't drop a real corpus entry whose closing glyph
        # isn't strictly the matching bracket.
        ("(◕‿◕ followed by prose", "(◕‿◕"),
        ("(｡•  more prose past the paren", "(｡•"),
    ],
)
def test_extract_unbalanced_bracket_fallback(
    text: str, expected_first_word: str,
) -> None:
    """Unbalanced bracket-leading kaomoji surface via the
    whitespace-split fallback in `_leading_bracket_span`. Real
    corpus output sometimes drops or substitutes the closing glyph
    and we want those entries in the journal."""
    assert extract(text).first_word == expected_first_word


# ---------------------------------------------------------------------------
# canonicalize_kaomoji — rule-by-rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,canonical,rule",
    [
        # Empty / whitespace
        ("",                 "",                 "empty"),
        ("   ",              "",                 "empty"),
        # Rule A: invisible / cosmetic-overlay strip
        ("(⁠◕⁠‿⁠◕⁠✿⁠)",  "(◕‿◕✿)",            "A"),
        ("(๑>؂<๑)",         "(๑><๑)",           "A"),
        # Rule B: half/full-width punctuation
        ("(＞_＜)",           "(>_<)",            "B"),
        ("(；ω；)",           "(;ω;)",            "B"),
        # Rule C: internal whitespace inside the bracket span
        ("( ; ω ; )",        "(;ω;)",            "C"),
        ("( ;´Д｀)",          "(;´д`)",           "C"),
        # Rule D: Cyrillic case fold
        ("(；´Д｀)",           "(;´д`)",           "D"),
        ("(；´д｀)",           "(;´д`)",           "D"),
        # Rule E1: degree-like circular eyes
        ("(°Д°)",             "(°д°)",            "E1"),
        ("(ºДº)",             "(°д°)",            "E1"),
        ("(˚Д˚)",             "(°д°)",            "E1"),
        # Rule E2: middle-dot variants
        ("(´・ω・`)",          "(´・ω・`)",         "E2"),
        ("(´･ω･`)",           "(´・ω・`)",         "E2"),
        # Rule F: arm/hand modifiers at face boundaries
        ("(๑˃ᴗ˂)ﻭ",          "(๑˃‿˂)",           "F"),
        ("(っ╥﹏╥)っ",          "(╥﹏╥)",            "F"),
        # Rule G: combining stroke overlays
        ("(๑˃̵‿˂̵)",          "(๑˃‿˂)",           "G"),
        # Rule H + B-speculative: curly quotes + fullwidth tilde.
        # v2.0 also strips the ``┐``/``┌`` box-drawing shrug arms
        # (round-3 sweep — was preserved in v1; bumping collapses).
        ("┐(‘～`;)┌",         "('~`;)",            "H"),
        ("┐('～`;)┌",         "('~`;)",            "H"),
        # Rule I: bullet → middle-dot
        ("(´•ω•`)",           "(´・ω・`)",         "I"),
        # Rule J: bracket-corner-circle → halfwidth ideographic full stop
        ("(◍•‿•◍)",          "(｡・‿・｡)",         "J"),
        ("(｡•‿•｡)",           "(｡・‿・｡)",         "J"),
        # Rule K: hyphen-as-mouth between two middle-dot eyes → underscore
        ("(・-・)",            "(・_・)",           "K"),
        ("(・_・)",            "(・_・)",           "K"),
        # NOT folded: hyphen-as-tired-eye glyph between accent + ω
        ("(´-ω-`)",           "(´-ω-`)",          "K-preserve"),
        # Rule L: asterisk-arm folds
        ("(*•̀‿•́*)",         "(・̀‿・́)",          "L"),
        # v2.0 — Rule M (outside-leading wing/hug/sparkle): strip ``\``,
        # ``⊂``, ``✧`` greedy at start before ``(``.
        ("\\(^o^)/",          "(^o^)",             "M-wing"),
        ("\\(≧▽≦)/",          "(≧▽≦)",             "M-wing"),
        ("⊂(˘ω˘)⊂",          "(˘ω˘)",             "M-hug"),
        ("✧(ˊᗜˋ)✧",          "(ˊᗜˋ)",             "M-sparkle"),
        # v2.0 — outside-trailing wing-right and hugging-arm-right:
        # ``/`` after ``)`` and ``⊂`` after ``)`` join existing
        # ``ﻭっ`` outside-trail set.
        ("(´∀`)/",            "(´∀`)",             "M-wing-right-only"),
        ("(˘ω˘)⊂",            "(˘ω˘)",             "M-hug-right-only"),
        # v2.0 sweep — outside-trailing mirror-close hugging arm.
        ("⊂(◕‿◕)⊃",          "(◕‿◕)",             "M-hug-pair"),
        # v2.0 sweep — bear face. Whole bear preserved (no inner
        # paren to fall back to); `•` folds to `・` via rule I.
        ("ʕ•ᴥ•ʔ",             "ʕ・ᴥ・ʔ",            "bear"),
        # v2.0 sweep — shocked sigma. Single-arm leader stripped.
        ("Σ(°△°|||)",         "(°△°|||)",          "M-sigma"),
        # v2.0 sweep — horn-fingers (Cyrillic Д→д via rule D).
        ("ψ(`Д´)ψ",           "(`д´)",             "M-psi"),
        ("Ψ(`Д´)Ψ",           "(`д´)",             "M-Psi"),
        # v2.0 sweep — kissing pair (ε + з).
        ("ε(◕‿◕)з",          "(◕‿◕)",             "M-kiss"),
        # v2.0 sweep — raised hands (⌣→‿ via smile-mouth synonym).
        ("ƪ(˘⌣˘)ʃ",          "(˘‿˘)",             "M-raised"),
        # v2.0 sweep — heavy-line wings.
        ("╲(◕‿◕)╱",          "(◕‿◕)",             "M-slashes"),
        # v2.0 sweep — paired arms of v1 leaders. Cheering, running,
        # cradling — finally canonicalize symmetrically.
        ("٩(◕‿◕)۶",          "(◕‿◕)",             "M-cheer"),
        ("ᕕ(ᐛ)ᕗ",            "(ᐛ)",                "M-running"),
        ("໒(◕‿◕)७",          "(◕‿◕)",             "M-cradle"),
        # v2.0 round 3 — box-drawing pose pairs collapse to face.
        ("╰(´∀｀)╯",           "(´∀`)",             "M-arms-up"),
        ("╭(´∀｀)╮",           "(´∀`)",             "M-curl"),
        ("┐(´д｀)┌",           "(´д`)",             "M-shrug"),
        # Inverted shrug pattern (╮ as lead, ╭ as trail). v2.0
        # symmetric strip handles both orientations.
        ("╮(´д｀)╭",           "(´д`)",             "M-shrug-inv"),
        # Inverted box-drawing shrug with `┌` lead, `┐` trail.
        ("┌(´д｀)┐",           "(´д`)",             "M-shrug-inv-box"),
        # v2.0 round 3 — the iconic shrug. ¯ \ _ strip on the lead,
        # _ / ¯ on the trail, leaving the bare ``(ツ)`` face.
        ("¯\\_(ツ)_/¯",         "(ツ)",               "M-shrug-tsu"),
        # Table-flip: the ``╯`` AT THE END is the rage-arm and
        # strips, but the ``╯`` INSIDE is the rage-cheek and stays.
        # Anchored regex is what makes this clean.
        ("(╯°□°)╯",            "(╯°□°)",            "M-rage-arm"),
        # Preservation: ``_`` inside the face is not stripped (the
        # outside-arm regex is anchored at start/end and only fires
        # before ``(`` or after ``)``).
        ("(◕_◕)",              "(◕_◕)",             "M-preserve-mouth"),
        # Rules M / N: smile-mouth equivalence class → ‿
        ("(◔◡◔)",             "(◕‿◕)",            "M"),
        ("(ᵔ◡ᵔ)",             "(ᵔ‿ᵔ)",            "N"),
        ("(´｡・ᵕ・｡`)",       "(´｡・‿・｡`)",       "N"),
        # Rule O: fullwidth grave → ASCII grave. v2.0 also strips
        # the ``ヽ``/``ノ`` raised-hand arms (was preserved in v1 —
        # the rule O test pinned the pose; bumping to v2.0 collapses
        # it for symmetry with the rest of the paired-arm sweep).
        ("ヽ(´ー｀)ノ",         "(´ー`)",            "O"),
        ("ヽ(´ー`)ノ",         "(´ー`)",            "O"),
        # Halfwidth katakana ﾉ also strips.
        ("ヽ(´ー`)ﾉ",          "(´ー`)",            "O"),
        # Voiced iteration mark ヾ strips (left raised-hand variant).
        ("ヾ(◕‿◕)ノ",          "(◕‿◕)",            "O"),
        # B extension: ideographic full stop folds to halfwidth too
        ("(´。・ᵕ・。`)",      "(´｡・‿・｡`)",       "B-ext"),
        # Directional-fill eye class → ◕
        ("(◔‿◔)",             "(◕‿◕)",            "eye-class"),
        ("(◑‿◐)",             "(◕‿◕)",            "eye-class"),
        ("(◐‿◑)",             "(◕‿◕)",            "eye-class"),
        ("(◕‿◕)",             "(◕‿◕)",            "eye-class"),
        ("(◒_◒)",             "(◕_◕)",            "eye-class"),
        ("(◓‿◓)",             "(◕‿◕)",            "eye-class"),
        ("(◖_◗)",             "(◕_◕)",            "eye-class"),
        # Filled-with-pupil class → ⊙ (distinct from directional fill)
        ("(◉_◉)",             "(⊙_⊙)",            "pupil-class"),
        ("(⊙_⊙)",             "(⊙_⊙)",            "pupil-class"),
        ("(●_●)",             "(⊙_⊙)",            "pupil-class"),
        # Smile-mouth direct synonym
        ("(◕⌣◕)",             "(◕‿◕)",            "smile-mouth"),
        # Punctuation tail
        ("(・_・？)",          "(・_・?)",          "B"),
        ("(～ω～)",            "(~ω~)",            "B"),
        # Mixed combining marks (rule G with the wider set: U+0334, U+033F)
        ("(๑˃̴‿˂̿)",          "(๑˃‿˂)",           "G"),
        # No-op: already canonical
        ("(◠‿◠)",             "(◠‿◠)",            "no-op"),
    ],
    ids=lambda v: v if isinstance(v, str) and len(v) < 30 else None,
)
def test_canonicalize_rule(raw: str, canonical: str, rule: str) -> None:
    assert canonicalize_kaomoji(raw) == canonical, f"rule {rule}"


def test_canonicalize_idempotent_on_complex_example() -> None:
    """Applying the canonicalizer twice yields the same string —
    important because re-runs of analyze re-canonicalize cached
    rows."""
    once = canonicalize_kaomoji("( ⁠;⁠ ´⁠Д⁠｀⁠ )")
    twice = canonicalize_kaomoji(once)
    assert once == twice, (once, twice)


def test_canonicalize_preserves_semantically_distinct_eyes() -> None:
    """Eye-glyph classes that AREN'T in the directional-fill /
    pupil / smile-mouth equivalence sets stay distinct."""
    assert canonicalize_kaomoji("(◕‿◕)") != canonicalize_kaomoji("(♥‿♥)")


# ---------------------------------------------------------------------------
# is_kaomoji_candidate — validator contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("(｡◕‿◕｡)",          True),
        ("hi",                False),
        # v2.0: backslash at position 0 is the wing-hand pattern,
        # accepted. Backslash at position >= 1 is markdown-escape
        # artifact, still rejected (e.g. ``(\\*´∀｀\\*)``).
        ("\\(^o^)/",          True),
        ("\\(≧▽≦)/",          True),
        # Markdown-escape backslash artifact (still rejected — `\`
        # appears at position >= 1).
        ("(\\*´∀｀\\*)",       False),
        # 4+-letter run inside parens — prose, not a kaomoji.
        ("(Backgrounddebug)", False),
        # 4+-letter run inside an unclosed bracket-leading span —
        # rejected via the prose filter (the bracket-balance check
        # is gone; 4-letter-run carries the prose-rejection role).
        ("(unclosed",         False),
        # Oversize span — not a real kaomoji.
        ("(" + "a" * 100 + ")", False),
        # v2.0 sweep — bear face accepted (ʕ in start_chars).
        ("ʕ•ᴥ•ʔ",              True),
        # v2.0 sweep — shocked sigma accepted.
        ("Σ(°△°|||)",          True),
        # v2.0 sweep — horn-fingers accepted.
        ("ψ(`Д´)ψ",            True),
        # v2.0 sweep — raised hands accepted.
        ("ƪ(˘⌣˘)ʃ",           True),
        # v2.0 sweep — heavy-line wing accepted.
        ("╲(◕‿◕)╱",           True),
        # ASCII letter `m` is NOT a leader — bowing apology
        # ``m(_ _)m`` rejected at the validator (prose-risk
        # exclusion; see KAOMOJI_START_CHARS rationale).
        ("m(_ _)m",            False),
    ],
)
def test_is_kaomoji_candidate(candidate: str, expected: bool) -> None:
    assert is_kaomoji_candidate(candidate) is expected


def test_kaomoji_start_chars_includes_common_leaders() -> None:
    """Smoke-check that the leading-glyph set covers the canonical
    bracket leaders + the v2.0 sweep additions. The full set is the
    v2.0 lock."""
    for c in "([（｛":
        assert c in KAOMOJI_START_CHARS, c
    # v2.0 round 1 — wing/hug/sparkle:
    for c in "\\⊂✧":
        assert c in KAOMOJI_START_CHARS, c
    # v2.0 round 2 — Greek + Latin extension + box-drawing diagonals:
    for c in "ΣψΨεƪʕ╱╲":
        assert c in KAOMOJI_START_CHARS, c
