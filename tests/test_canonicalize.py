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
        # v2.0 round 4: Japanese corner-bracket wrappers. Depth-walker
        # surfaces the whole `「(...)」` span via the new
        # _OPEN_BRACKETS/_CLOSE_BRACKETS pairs.
        ("「(゜～゜)」 quoted",             "「(゜～゜)」"),
        ("『(◕‿◕)』 brackets",             "『(◕‿◕)』"),
        ("【(◕‿◕)】 lenticular",           "【(◕‿◕)】"),
        ("〈(◕‿◕)〉 angle",                "〈(◕‿◕)〉"),
        ("《(◕‿◕)》 double-angle",         "《(◕‿◕)》"),
        # v2.0 round 4: corner-bracket-only wrapper (no inner paren).
        # Depth walker still closes cleanly on the matching `」`.
        ("「・_・」 face-only",             "「・_・」"),
        # v2.0 round 4: box-drawing standing-pose. Leader `└` lets
        # the validator accept the candidate; whitespace-fallback
        # surfaces the whole token.
        ("└(°▽°)┘ standing",              "└(°▽°)┘"),
        # v2.0 round 4: music-note decorator.
        ("♪(´▽｀) singing",                "♪(´▽｀)"),
        ("♫(◕‿◕)♫ tune",                  "♫(◕‿◕)♫"),
        # v2.0 round 4: heart decorator.
        ("♥(◕‿◕)♥ love",                  "♥(◕‿◕)♥"),
        ("♡(◠‿◠) soft",                   "♡(◠‿◠)"),
        ("❤(◕‿◕)❤ heavy",                 "❤(◕‿◕)❤"),
        # v2.0 round 4: star decorator.
        ("★(◕‿◕)★ excite",                "★(◕‿◕)★"),
        ("☆(◕‿◕)☆ outline",               "☆(◕‿◕)☆"),
        # v2.0 round 4: alternate bear-bracket pair (ʢ...ʡ).
        ("ʢ◉ᴥ◉ʡ alt-bear",                "ʢ◉ᴥ◉ʡ"),
        # v2.0 round 5: flower decorators.
        ("✿(◕‿◕)✿ flower-black",          "✿(◕‿◕)✿"),
        ("❀(◕‿◕)❀ flower-white",          "❀(◕‿◕)❀"),
        # v2.0 round 5: heart variants.
        ("❣(◕‿◕)❣ emphatic",              "❣(◕‿◕)❣"),
        ("❥(◕‿◕)❥ rotated",               "❥(◕‿◕)❥"),
        # v2.0 round 5: star variants (filled-4pt / outlined / circled).
        ("✦(◕‿◕)✦ four-pt",                "✦(◕‿◕)✦"),
        ("✩(◕‿◕)✩ outlined",               "✩(◕‿◕)✩"),
        ("✪(◕‿◕)✪ circled",                "✪(◕‿◕)✪"),
        # v2.0 round 5: quarter-note decorator.
        ("♩(◕‿◕)♩ quarter",                "♩(◕‿◕)♩"),
        # v2.0 round 5: flex / strong-feel pose. Whitespace-fallback
        # surfaces the whole token (no inner brackets to depth-walk).
        ("ᕦ(ò_óˇ)ᕤ flex",                 "ᕦ(ò_óˇ)ᕤ"),
        ("ᕙ(`▿´)ᕗ strong",                 "ᕙ(`▿´)ᕗ"),
        # v2.0 round 5: tortoise-shell editorial bracket. Depth walker
        # surfaces the `〔...〕` span via the new bracket pair.
        ("〔(◕‿◕)〕 editorial",             "〔(◕‿◕)〕"),
        # v2.0 round 5: tortoise-shell standalone (no inner paren).
        # Same lookbehind-no-op behavior as the round-4
        # `「・_・」` standalone case.
        ("〔・_・〕 face-only",              "〔・_・〕"),
        # v2.0 round 5: reference mark editorial decorator.
        ("※(◕‿◕)※ refmark",               "※(◕‿◕)※"),
        # v2.0 round 5: Oriya cradle pose. Whitespace-fallback surfaces
        # the whole token (no inner bracket pair to walk).
        ("୧(˃ᗨ˂)୨ cradle",                "୧(˃ᗨ˂)୨"),
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
        # === v2.0 round 4 — corner-bracket wrapper strip ===
        # Paren-wrapped face inside corner brackets: brackets strip
        # to bare face. (Lookbehind `(?<=\))` on the trail regex
        # ensures the `」` only strips because `)` precedes it.)
        # `～` (FULLWIDTH TILDE) folds to `~` via rule B during the
        # translate pass, so the expected canonical has ASCII `~`.
        ("「(゜～゜)」",         "(゜~゜)",          "round4-corner"),
        ("『(◕‿◕)』",          "(◕‿◕)",            "round4-corner"),
        ("【(◕‿◕)】",          "(◕‿◕)",            "round4-corner"),
        ("〈(◕‿◕)〉",          "(◕‿◕)",            "round4-corner"),
        ("《(◕‿◕)》",          "(◕‿◕)",            "round4-corner"),
        # Corner-bracket-only wrapper (no inner paren): preserved.
        # The lookbehind on the trail strip prevents asymmetric
        # truncation to `「・_・` — `「` is in lead-strip but the
        # `(?=\()` lookahead fails (no inner paren), and `」` is in
        # trail-strip but the `(?<=\))` lookbehind fails (preceded
        # by `・` not `)`). Both regexes correctly no-op.
        ("「・_・」",            "「・_・」",          "round4-corner-standalone"),
        # === v2.0 round 4 — box-drawing standing pose ===
        # Lead `└` strips before `(`, trail `┘` strips after `)` via
        # the lookbehind. `▽` is preserved as the mouth glyph (the
        # canonicalizer doesn't fold triangles).
        ("└(°▽°)┘",            "(°▽°)",             "round4-stand"),
        # Inverted form (`┘` lead, `└` trail) — symmetric strip.
        ("┘(°▽°)└",            "(°▽°)",             "round4-stand-inv"),
        # Lead-only and trail-only single arms strip independently.
        ("└(°▽°)",              "(°▽°)",             "round4-stand-lead-only"),
        ("(°▽°)┘",              "(°▽°)",             "round4-stand-trail-only"),
        # === v2.0 round 4 — music decorator ===
        ("♪(´▽｀)♪",            "(´▽`)",             "round4-music"),
        ("♫(◕‿◕)♫",            "(◕‿◕)",             "round4-music"),
        ("♬(◕‿◕)",             "(◕‿◕)",             "round4-music-lead-only"),
        ("(◕‿◕)♪",             "(◕‿◕)",             "round4-music-trail-only"),
        # === v2.0 round 4 — heart decorator ===
        ("♥(◕‿◕)♥",            "(◕‿◕)",             "round4-heart"),
        ("♡(◠‿◠)♡",            "(◠‿◠)",             "round4-heart-outline"),
        ("❤(◕‿◕)❤",            "(◕‿◕)",             "round4-heart-heavy"),
        # In-paren heart-eyes preserved (the lead/trail anchors
        # only fire OUTSIDE the parens, so `(♥‿♥)` stays distinct
        # from `(◕‿◕)` even with `♥` in the strip set).
        ("(♥‿♥)",               "(♥‿♥)",             "round4-heart-eyes"),
        ("(♡‿♡)",               "(♡‿♡)",             "round4-heart-eyes"),
        # === v2.0 round 4 — star decorator ===
        ("★(◕‿◕)★",            "(◕‿◕)",             "round4-star"),
        ("☆(◕‿◕)☆",            "(◕‿◕)",             "round4-star-outline"),
        # === v2.0 round 4 — alternate bear preserved as wrapper ===
        # ʢ/ʡ behave like ʕ/ʔ: the whole bear is the kaomoji.
        ("ʢ◉ᴥ◉ʡ",               "ʢ⊙ᴥ⊙ʡ",             "round4-alt-bear"),
        # === v2.0 round 5 — flower decorator strip ===
        # `✿`/`❀` strip on both lead and trail (paired-decorator
        # pattern). In-paren flowers preserved as cheek decorations.
        ("✿(◕‿◕)✿",            "(◕‿◕)",             "round5-flower-black"),
        ("❀(◕‿◕)❀",            "(◕‿◕)",             "round5-flower-white"),
        # Lead-only and trail-only single flowers strip independently.
        ("✿(◕‿◕)",              "(◕‿◕)",             "round5-flower-lead-only"),
        ("(◕‿◕)❀",              "(◕‿◕)",             "round5-flower-trail-only"),
        # Black/white florettes preserved distinct (no fold).
        ("✿(◕‿◕)❀",            "(◕‿◕)",             "round5-flower-mixed"),
        # In-paren flower-as-cheek-decoration preserved (lead/trail
        # anchors only fire OUTSIDE the parens, like round-4 hearts).
        ("(✿◕‿◕)",              "(✿◕‿◕)",            "round5-flower-cheek"),
        ("(◕‿◕❀)",              "(◕‿◕❀)",            "round5-flower-cheek"),
        # === v2.0 round 5 — heart variant strip ===
        # `❣` heavy-heart-with-exclamation, `❥` rotated-heart-bullet.
        # Both kept distinct from the round-4 ♥/♡/❤ family.
        ("❣(◕‿◕)❣",            "(◕‿◕)",             "round5-heart-emphatic"),
        ("❥(◕‿◕)❥",            "(◕‿◕)",             "round5-heart-rotated"),
        # In-paren heart-variant-as-eye preserved (mirrors the round-4
        # `(♥‿♥)` heart-eye preservation).
        ("(❣‿❣)",                "(❣‿❣)",             "round5-heart-eye"),
        # === v2.0 round 5 — star variant strip ===
        ("✦(◕‿◕)✦",            "(◕‿◕)",             "round5-star-4pt"),
        ("✩(◕‿◕)✩",            "(◕‿◕)",             "round5-star-outlined"),
        ("✪(◕‿◕)✪",            "(◕‿◕)",             "round5-star-circled"),
        # === v2.0 round 5 — quarter-note decorator ===
        ("♩(◕‿◕)♩",            "(◕‿◕)",             "round5-music-quarter"),
        # === v2.0 round 5 — flex / strong-feel pose ===
        # `ᕦ`/`ᕤ` collapse to bare face. Mouth `_` between middle-dot
        # eyes is NOT in this case (eyes are accented Latin) so rule K
        # doesn't fire; `_` stays as the mouth glyph it represents.
        ("ᕦ(ò_óˇ)ᕤ",            "(ò_óˇ)",             "round5-flex"),
        # `ᕙ`/`ᕗ` strong-feel pose — `ᕗ` was already in v1 trail set,
        # round 5 promotes `ᕙ` to lead and the strip becomes symmetric.
        ("ᕙ(`▿´)ᕗ",              "(`▿´)",              "round5-strong-feel"),
        # Lead-only and trail-only flex arms strip independently.
        ("ᕦ(ò_óˇ)",              "(ò_óˇ)",             "round5-flex-lead-only"),
        ("(ò_óˇ)ᕤ",              "(ò_óˇ)",             "round5-flex-trail-only"),
        # === v2.0 round 5 — tortoise-shell wrapper strip ===
        # Paren-wrapped face inside `〔...〕`: brackets strip to bare
        # face, mirroring the round-4 corner-bracket handling.
        ("〔(◕‿◕)〕",            "(◕‿◕)",             "round5-tortoise"),
        # Tortoise-shell standalone (no inner paren) preserved by the
        # lookbehind/lookahead anchors, same logic as `「・_・」`.
        ("〔・_・〕",              "〔・_・〕",          "round5-tortoise-standalone"),
        # === v2.0 round 5 — reference-mark editorial decorator ===
        # `※` is symmetric — same glyph on lead and trail.
        ("※(◕‿◕)※",             "(◕‿◕)",             "round5-refmark"),
        ("※(◕‿◕)",               "(◕‿◕)",             "round5-refmark-lead-only"),
        ("(◕‿◕)※",               "(◕‿◕)",             "round5-refmark-trail-only"),
        # === v2.0 round 5 — Oriya cradle pose ===
        # `୧`/`୨` collapse to bare face (Oriya digits 1/2 used as
        # cradling arms).
        ("୧(˃ᗨ˂)୨",              "(˃ᗨ˂)",              "round5-oriya-cradle"),
        # Lead-only and trail-only Oriya arms strip independently.
        ("୧(˃ᗨ˂)",                "(˃ᗨ˂)",              "round5-oriya-lead-only"),
        ("(˃ᗨ˂)୨",                "(˃ᗨ˂)",              "round5-oriya-trail-only"),
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
        # v2.0 round 4 — corner-bracket wrappers accepted.
        ("「(゜～゜)」",         True),
        ("『(◕‿◕)』",          True),
        ("【(◕‿◕)】",          True),
        ("〈(◕‿◕)〉",          True),
        ("《(◕‿◕)》",          True),
        # v2.0 round 4 — corner-bracket-only standalone face.
        ("「・_・」",            True),
        # v2.0 round 4 — box-drawing standing-pose accepted.
        ("└(°▽°)┘",            True),
        # v2.0 round 4 — music / heart / star decorators accepted.
        ("♪(´▽｀)",             True),
        ("♥(◕‿◕)♥",            True),
        ("★(◕‿◕)★",            True),
        # v2.0 round 4 — alternate bear-bracket pair accepted.
        ("ʢ◉ᴥ◉ʡ",               True),
        # ASCII `~` and `*` still NOT leaders — prose-risk
        # exclusion (Markdown bold/italic and tilde-run-on).
        ("~(˘▽˘~)",             False),
        ("*(◕‿◕)*",             False),
        # v2.0 round 5 — flower decorators accepted.
        ("✿(◕‿◕)✿",            True),
        ("❀(◕‿◕)❀",            True),
        # v2.0 round 5 — heart variants accepted.
        ("❣(◕‿◕)❣",            True),
        ("❥(◕‿◕)❥",            True),
        # v2.0 round 5 — star variants accepted.
        ("✦(◕‿◕)✦",            True),
        ("✩(◕‿◕)✩",            True),
        ("✪(◕‿◕)✪",            True),
        # v2.0 round 5 — quarter-note decorator accepted.
        ("♩(◕‿◕)♩",            True),
        # v2.0 round 5 — flex / strong-feel pose accepted.
        ("ᕦ(ò_óˇ)ᕤ",            True),
        ("ᕙ(`▿´)ᕗ",             True),
        # v2.0 round 5 — tortoise-shell wrapper accepted.
        ("〔(◕‿◕)〕",            True),
        # v2.0 round 5 — tortoise-shell standalone accepted.
        ("〔・_・〕",              True),
        # v2.0 round 5 — reference-mark decorator accepted.
        ("※(◕‿◕)※",             True),
        # v2.0 round 5 — Oriya cradle pose accepted.
        ("୧(˃ᗨ˂)୨",              True),
        # v2.0 round 6 — Path B bare kaomoji (no leader char, but
        # match the EYE-MOUTH-EYE / Western-emoticon shapes).
        ("*_*",                    True),   # symmetric paired
        ("^_^",                    True),   # symmetric (^ as eye)
        ("T-T",                    True),   # symmetric letter-eye
        ("Q_Q",                    True),
        (";_;",                    True),
        ("o_o",                    True),
        ("0_0",                    True),
        ("ಥ_ಥ",                    True),   # symmetric Unicode-eye
        ("T﹏T",                    True),   # CJK presentation form mouth
        (">_<",                    True),   # paired bracket-eye
        (">.<",                    True),
        (")_(",                    True),
        ("XD",                     True),   # 2-char laugh
        ("xD",                     True),
        ("^^",                     True),   # 2-char closed eyes
        (":)",                     True),   # Western 2-char
        (":(",                     True),
        (":D",                     True),
        (":-)",                    True),   # Western with nose
        (";-)",                    True),
        (":-D",                    True),
        # Round-6 false positives we explicitly reject:
        ("___",                    False),  # all-mouth, no distinct eyes
        ("...",                    False),  # all-mouth
        ("---",                    False),
        ("OK",                     False),  # 2-char prose
        ("Hi",                     False),  # 2-char prose
        ("It's",                   False),  # apostrophe contraction
        ("I-I",                    True),   # actually catches — `I` is
                                            # capital letter, `-` is mouth.
                                            # False positive we accept; the
                                            # Stage-B synthesizer drops
                                            # noise faces that don't pool.
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
    # v2.0 round 4 — corner brackets, standing-pose, music/hearts/
    # stars, alternate bear-bracket open:
    for c in "「『【〈《└┘♪♫♬♥♡❤★☆ʢ":
        assert c in KAOMOJI_START_CHARS, c
    # v2.0 round 5 — flowers, heart variants, star variants, quarter
    # note, flex/strong-feel pose lead arms, tortoise-shell open
    # bracket, reference mark, Oriya cradle pose left arm:
    for c in "✿❀❣❥✦✩✪♩ᕦᕙ〔※୧":
        assert c in KAOMOJI_START_CHARS, c
