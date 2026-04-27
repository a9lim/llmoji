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
        # Unbalanced bracket → rejected (no whitespace-split fallback).
        "(｡• ω •｡  open paren never closed",
        # Oversize balanced span → rejected.
        "(" + "a" * 50 + ") text",
    ],
)
def test_extract_rejects(text: str) -> None:
    assert extract(text).first_word == ""


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
        # Rule H + B-speculative: curly quotes + fullwidth tilde
        ("┐(‘～`;)┌",         "┐('~`;)┌",         "H"),
        ("┐('～`;)┌",         "┐('~`;)┌",         "H"),
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
        # Rules M / N: smile-mouth equivalence class → ‿
        ("(◔◡◔)",             "(◕‿◕)",            "M"),
        ("(ᵔ◡ᵔ)",             "(ᵔ‿ᵔ)",            "N"),
        ("(´｡・ᵕ・｡`)",       "(´｡・‿・｡`)",       "N"),
        # Rule O: fullwidth grave → ASCII grave
        ("ヽ(´ー｀)ノ",         "ヽ(´ー`)ノ",        "O"),
        ("ヽ(´ー`)ノ",         "ヽ(´ー`)ノ",        "O"),
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
        # Markdown-escape backslash artifact.
        ("(\\*´∀｀\\*)",       False),
        # 4+-letter run inside parens — prose, not a kaomoji.
        ("(Backgrounddebug)", False),
        # Unbalanced bracket — sed-cut at first letter mid-bracket.
        ("(unclosed",         False),
        # Oversize span — not a real kaomoji.
        ("(" + "a" * 100 + ")", False),
    ],
)
def test_is_kaomoji_candidate(candidate: str, expected: bool) -> None:
    assert is_kaomoji_candidate(candidate) is expected


def test_kaomoji_start_chars_includes_common_leaders() -> None:
    """Smoke-check that the leading-glyph set covers the canonical
    bracket leaders. The full set is the v1.0 lock."""
    for c in "([（｛":
        assert c in KAOMOJI_START_CHARS, c
