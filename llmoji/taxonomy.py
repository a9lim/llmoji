"""Kaomoji canonicalization, validation, and extraction.

The public v1.0 surface is everything in this module. Bumping any
of `KAOMOJI_START_CHARS`, `is_kaomoji_candidate`, `extract`, or the
canonicalization rules below is a major version bump — the central
HF dataset declares "v1 corpus only" against these invariants.

Pilot-specific affect labels (`+1 happy / -1 sad`, `+1 angry / -1
calm`, etc.) live with the research-side code in
``llmoji-study/llmoji_study/taxonomy_labels.py``. They were here in
v0.x; the v1.0 split extracts them because they're gemma-tuned and
have no place in a provider-agnostic public package.

Extractor notes:
  - `extract` returns a `KaomojiMatch` containing the validated
    leading kaomoji span (or `""` if the input doesn't look like a
    kaomoji-prefixed message).
  - For bracket-leading inputs the extractor prefers a balanced-paren
    span — that's how whitespace-padded kaomoji like ``(｡˃ ᵕ ˂ )``
    surface intact. When the bracket span doesn't close cleanly
    inside the length cap (real corpus output is sometimes
    unbalanced), it falls back to a whitespace-delimited word so the
    leading kaomoji still surfaces. The `is_kaomoji_candidate`
    validator no longer enforces bracket balance — the length cap +
    4-letter-run + backslash filters carry the prose-rejection role.
  - For research-side label lookups, see
    ``llmoji_study.taxonomy_labels.extract_with_label``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Bracket pairs the fallback extractor treats as kaomoji boundaries.
# `ʕ`/`ʔ` (LATIN LETTER PHARYNGEAL VOICED FRICATIVE / GLOTTAL STOP)
# are the bear-face brackets in `ʕ•ᴥ•ʔ` — adding them to the depth-
# walk pair lets the bracket-balance branch surface the bear span
# directly instead of falling through to the whitespace-fallback.
# `ʢ`/`ʡ` (LATIN LETTER GLOTTAL STOP WITH STROKE / REVERSED) are the
# alternate bear-bracket pair in `ʢ◉ᴥ◉ʡ` variants — same self-
# contained-face rationale.
# `「『【〈《` / `」』】〉》` are Japanese corner-bracket wrappers
# (`「(゜～゜)」`, `『(◕‿◕)』`, `【(◕‿◕)】`); added to the depth
# walker so the leading wrapper finds its matching close instead of
# the whitespace-fallback eating the whole span. Pre-2.0-round-4
# kaomoji wrapped this way wouldn't even surface as candidates
# because `「`/`『`/`【`/`〈`/`《` weren't in `KAOMOJI_START_CHARS`;
# the round-4 sweep adds them.
# `〔`/`〕` (TORTOISE SHELL BRACKET) are a round-5 addition — the
# Japanese editorial bracket pair, e.g. `〔(◕‿◕)〕` and the
# corner-bracket-only standalone variant `〔・_・〕`. Same depth-
# walker rationale as the round-4 corner brackets.
_OPEN_BRACKETS = "([（｛ʕʢ「『【〈《〔"
_CLOSE_BRACKETS = ")]）｝ʔʡ」』】〉》〕"

# Leading-glyph filter for kaomoji-bearing assistant turns. Used by
# `extract`, by `is_kaomoji_candidate`, and by every shell hook
# template under `llmoji._hooks/` (rendered into the bash `case`
# pattern via `llmoji.providers.base.render_kaomoji_start_chars_case`).
# Single source of truth; previous versions duplicated this set in
# five places, which is the gotcha the v1.0 split resolved.
#
# v2.0 additions, by sweep round:
#
#   Round 1 — wing/hug/sparkle leaders identified during the initial
#   2.0 broaden: ASCII `\` (wing-hand `\(^o^)/`), `⊂` (hugging arms
#   `⊂(...)⊂`), `✧` (sparkle-decorated `✧*｡(...)*｡✧`).
#
#   Round 2 — non-prose leaders identified while running Claude on
#   emotional prompts:
#     * Greek `Σ ψ Ψ ε` — `Σ(°△°|||)` shocked-sigma, `ψ(´Д`)ψ` /
#       `Ψ(´Д`)Ψ` horn-fingers, `ε(◕‿◕)з` kissing-pair.
#     * Latin extensions `ƪ ʕ` — `ƪ(˘⌣˘)ʃ` raised hands and the
#       `ʕ•ᴥ•ʔ` bear face. (`ʕ` doubles as a face-bracket — see
#       `_OPEN_BRACKETS`.)
#     * Box-drawing diagonals `╱ ╲` — `╲(◕‿◕)╱` celebratory wings,
#       the heavier-line cousins of `\(^o^)/`.
#
#   Round 3 — pose pairs and shrug components: `╰ ╭` lead /
#   `╯ ╮` trail (arms-up / curl), `¯` lead+trail and `_` lead+trail
#   for the iconic `¯\_(ツ)_/¯` shrug. (Those landed via the
#   `_ARM_OUTSIDE` / `_ARM_OUTSIDE_LEAD` sets — for the leader set,
#   `╰ ╭ ╮ ┐ ┌ ¯ ＼` were already in v1.)
#
#   Round 4 — broaden to "any plausibly real kaomoji leader":
#     * Japanese corner brackets `「『【〈《` — `「(゜～゜)」`,
#       `『(◕‿◕)』`, `【(◕‿◕)】`. Paired with `」』】〉》` in
#       `_CLOSE_BRACKETS` and `_ARM_OUTSIDE`. Trade-off: a
#       leading Japanese quoted phrase (`「これは良い」...`) parses
#       as a balanced kaomoji-shaped span and slips past the
#       validator (length ≤32, no 4-letter ASCII run, no backslash);
#       Stage A synth describes those as "Japanese phrase, not a
#       face" and they cluster as their own canonical thing. Real-
#       corpus rate in coding-agent traffic is approximately zero.
#     * Box-drawing standing-pose corners `└ ┘` —
#       `└(°▽°)┘` arms-up-standing, the upright variant of the
#       `╰...╯` round-3 pose. (`┘` was already in `_ARM_OUTSIDE`
#       trail; round 4 promotes `└` to lead and treats `┘` as a
#       symmetric lead candidate too — `┘(°▽°)└` inverted form.)
#     * Music notes `♪ ♫ ♬` — `♪(´▽｀)`, `♫(◕‿◕)♫`, happy-
#       singing register. Distinct glyphs preserved (no fold).
#     * Hearts `♥ ♡ ❤` — `♥(◕‿◕)♥` love-decorator,
#       `♡(◠‿◠)♡` softer outline-heart variant, `❤(◕‿◕)❤`
#       heavy-heart variant. The decorator strip is anchored at the
#       face boundary, so `(♥‿♥)` (heart-as-eye, inside parens) is
#       preserved as a distinct face — only the outside-the-parens
#       hearts strip.
#     * Stars `★ ☆` — `★(◕‿◕)★` excitement-decorator,
#       `☆(◕‿◕)☆` outline variant. Same in/out anchoring as
#       hearts.
#     * Alternate bear-bracket open `ʢ` — `ʢ◉ᴥ◉ʡ` variants
#       (paired with `ʡ`). Like `ʕ`/`ʔ`, these are added to
#       `_OPEN_BRACKETS`/`_CLOSE_BRACKETS` for depth-walk
#       recognition but stay OUT of arm-strip — the whole bear-
#       shape is the kaomoji.
#
#   Round 5 — exhaustive "every plausibly real kaomoji leader and
#   matching arm-strip glyph" sweep, after round 4 covered the
#   high-frequency families. Same prose-risk threshold (acceptable
#   in coding-agent context) as round 4. Each addition is paired
#   symmetrically across `KAOMOJI_START_CHARS` /
#   `_ARM_OUTSIDE_LEAD` / `_ARM_OUTSIDE` so the canonicalizer
#   collapses both halves of a paired pose; single-decoration
#   leaders that don't have a distinct close-half (e.g. `※` is
#   itself symmetric) appear in all three sets.
#     * Flower decorators `✿ ❀` — `✿(◕‿◕)✿` BLACK FLORETTE
#       wrapper, `❀(◕‿◕)❀` WHITE FLORETTE. Both kept distinct
#       (no fold) for register parity with the heart and star
#       families.
#     * Heart decorator variants `❣ ❥` — HEAVY HEART EXCLAMATION
#       MARK ORNAMENT (`❣(◕‿◕)❣` emphatic-heart) and ROTATED
#       HEAVY BLACK HEART BULLET (`❥(◕‿◕)❥` decorative-heart).
#       Kept distinct from the round-4 `♥ ♡ ❤` family (different
#       affect register).
#     * Star decorator variants `✦ ✩ ✪` — BLACK FOUR POINTED
#       STAR (`✦(◕‿◕)✦`, the filled cousin of round-1 `✧`),
#       STRESS OUTLINED WHITE STAR (`✩(◕‿◕)✩`), CIRCLED WHITE
#       STAR (`✪(◕‿◕)✪`). Distinct point counts and fills, kept
#       distinct from `★ ☆`.
#     * Music note variant `♩` — QUARTER NOTE (`♩(◕‿◕)♩`),
#       completes the `♪ ♫ ♬` round-4 set.
#     * Flex / strong-feel pose lead arms `ᕦ ᕙ` — Canadian
#       Syllabics NWUU and WO. Iconic strongman pose
#       `ᕦ(ò_óˇ)ᕤ` and the strong-feel pose `ᕙ(`▿´)ᕗ`. The
#       right-half close-arms are `ᕤ` (added to `_ARM_OUTSIDE`
#       trail) and `ᕗ` (already in v1 trail set).
#     * Tortoise-shell brackets `〔〕` — `〔(◕‿◕)〕` Japanese
#       editorial bracket wrapper, plus the corner-bracket-only
#       standalone variant `〔・_・〕`. Paired in
#       `_OPEN_BRACKETS`/`_CLOSE_BRACKETS` and arm-strip,
#       mirroring the round-4 `「『【〈《` corner-bracket
#       handling.
#     * Reference mark `※` — Japanese editorial decorator
#       (`※(◕‿◕)※`). Symmetric (no distinct close-half), so
#       appears in both `_ARM_OUTSIDE` and `_ARM_OUTSIDE_LEAD`.
#     * Oriya cradle pose `୧ ୨` — DIGIT ONE / DIGIT TWO used as
#       cradling-arms in `୧(˃ᗨ˂)୨`. Same lead-only / trail-only
#       split as the round-2 paired-arm leaders.
#
# Deliberately NOT added in round 5 (considered, rejected):
#   * `◜ ◝ ◞ ◟` (CIRCULAR ARC corners) — observed in
#     `◝(⁰▿⁰)◜`-style fancy frames, but the lead/trail role of
#     each corner-arc isn't unambiguous (different sources put `◝`
#     on either side), so a clean symmetric strip would require
#     tracking two orientations per glyph. Defer until a real-
#     corpus sample tells us which orientation dominates.
#   * `❄ ❅ ❆` (snowflakes), `✱ ✲ ✳ ✴` (heavy asterisks),
#     `❉ ❊` (florettes) — plausible decorators but not observed
#     as kaomoji-leaders in the gemma / Claude corpora. Round 5
#     stops at observed-or-near-observed.
#
# Deliberately NOT added (despite being real kaomoji leaders):
# ASCII letters `o O q b d m p t` — these collide with very common
# 2-3 letter prose words ("ok", "of", "my", "to", "be", ...) that
# the validator's `is_kaomoji_candidate` would let through. The
# additions above don't have that problem because they almost never
# start a non-kaomoji English word. ASCII `~` and `*` similarly
# excluded — wavy-mouth `~(˘▽˘~)` and sparkle-bracket `*(◕‿◕)*`
# are real but `~` is too common in prose tildes and `*` is the
# Markdown bold/italic delimiter, both of which would generate
# constant false positives.
KAOMOJI_START_CHARS: frozenset[str] = frozenset(
    "([（｛ヽヾっ٩ᕕ╰╭╮┐┌＼¯໒\\⊂✧"  # v1.0 set + 2.0 round-1 wing/hug/sparkle
    "Σψεƪʕ"                       # round 2: Greek + Latin extension
    "Ψ"                            # round 2: capital psi (matching ψ horn-arm)
    "╱╲"                           # round 2: box-drawing diagonals
    "「『【〈《"                    # round 4: Japanese corner brackets
    "└┘"                           # round 4: box-drawing standing-pose
    "♪♫♬"                          # round 4: music notes
    "♥♡❤"                          # round 4: hearts (filled / outline / heavy)
    "★☆"                           # round 4: stars (filled / outline)
    "ʢ"                            # round 4: alternate bear-bracket open
    "✿❀"                           # round 5: flower decorators (black/white florette)
    "❣❥"                           # round 5: heart variants (heavy / rotated)
    "✦✩✪"                          # round 5: star variants (filled-4pt / outlined / circled)
    "♩"                            # round 5: quarter note (completes ♪♫♬)
    "ᕦᕙ"                          # round 5: flex / strong-feel pose lead arms
    "〔"                           # round 5: tortoise-shell open bracket
    "※"                            # round 5: reference mark editorial decorator
    "୧"                            # round 5: Oriya cradle pose left arm
)


# Maximum length of a real kaomoji we expect to encounter. Real
# kaomoji span ~5–25 characters; the longest form encountered in
# the gemma corpus was ``(╯°□°)╯︵ ┻━┻`` at ~12 chars. The cap
# rejects two-line balanced-paren prose accidentally captured by
# the bracket-span scan.
_KAOMOJI_MAX_LEN = 32

# A run of 4+ consecutive ASCII letters indicates prose, not a
# kaomoji. Belt-and-suspenders for the gemma extractor path and for
# catching pre-cut garbage in legacy data — the shell hook's
# ``[A-Za-z].*$`` cut already strips at the first letter.
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{4}")


# Bare-kaomoji mouth-glyph set (v2.0 round-6 speculative extension).
#
# Used by `_looks_like_bare_kaomoji` to validate the interior of an
# `EYE MOUTH EYE` candidate that doesn't start with a
# `KAOMOJI_START_CHARS` leader. Restricted to canonical kaomoji mouth
# glyphs (ASCII connectors, dashes/punctuation, CJK presentation
# forms, geometric shapes) — explicitly excludes ASCII letters so
# prose tokens like "It's" don't validate as bare kaomoji.
_BARE_KAOMOJI_MOUTH_RE = re.compile(
    r"["
    r"_\-.;:~=^|/\\"           # ASCII mouth glyphs
    r"·•°"      # middle dot, bullet, degree sign
    r"‐-―"           # various dashes
    r"‥…′-‷"  # ellipsis variants, primes
    r"‿⁀"            # undertie, character tie
    r"、。"            # CJK comma/fullstop
    r"　・ー･＿？"  # CJK spaces / mid-dots / fullwidth forms
    r"︰-﹏"           # CJK presentation forms (︵ ︶ ﹏ ﹋ ﹌ ︿ ︶ etc.)
    r"▰-◿"           # geometric shapes (▽ ◡ ◠ ○ ● ◇ etc.)
    r"]+"
)

# Visually-paired eye glyphs for non-symmetric bare kaomoji like `>_<`,
# `>.<`, `(_)`, `)_(`. Used as eyes (NOT brackets — these don't trigger
# the `_OPEN_BRACKETS` depth-walker).
_BARE_KAOMOJI_PAIRED_EYES: frozenset[tuple[str, str]] = frozenset({
    (">", "<"), ("<", ">"),
    (")", "("), ("(", ")"),
    ("]", "["), ("[", "]"),
    ("}", "{"), ("{", "}"),
})


def _looks_like_bare_kaomoji(s: str) -> bool:
    """Speculative bare-kaomoji shape match (v2.0 round-6 extension).

    Catches faces that don't start with a `KAOMOJI_START_CHARS` leader:
      * Symmetric `EYE MOUTH EYE`: ``^_^``, ``T-T``, ``Q_Q``, ``;_;``,
        ``o_o``, ``O_O``, ``0_0``, ``ಥ_ಥ``, ``ಥ﹏ಥ``, ``T﹏T``,
        ``e_e``, etc. — same eye on both sides, mouth glyphs in
        between.
      * Paired-eye `EYE MOUTH EYE`: ``>_<``, ``>.<``, ``)_(``,
        ``]_[`` — visually-mirrored bracket eyes (NOT bracket
        delimiters, just eye shapes).
      * Western emoticons: ``:)``, ``:(``, ``:D``, ``;)``, ``:P``,
        ``:3``, ``:O``, ``=)``, ``8)``, ``:-)``, ``:-D``, ``;-)`` —
        ``:`` ``;`` ``=`` ``8`` eye + optional ``-`` ``^`` ``o``
        ``'`` nose + ``)`` ``(`` ``D`` ``P`` ``O`` ``o`` ``3``
        ``<`` ``>`` ``/`` ``\\`` ``|`` ``*`` ``[`` ``]`` mouth.
      * 2-char closed-eye doubles: ``^^``, ``vv``, ``uu``.
      * 2-char ``XD``/``xD``/``xd`` style.

    Length / backslash / 4-letter-run filters are applied by the
    caller (`is_kaomoji_candidate`); this function focuses on the
    structural shape.

    "Speculative" framing: the goal is to surface bare-kaomoji
    affective output from models whose register strips the
    parenthesizing wrapper (e.g. granite emitting ``ಥ﹏ಥ`` for
    every grief prompt). False positives are tolerated when the
    shape is unambiguous; the length cap and letter-run filter
    upstream limit damage from prose collisions.
    """
    n = len(s)
    if n < 2:
        return False

    # 2-char patterns: closed-eye doubles + Western 2-char emoticons.
    if n == 2:
        if s[0] == s[1] and s[0] in "^vu":
            return True
        if s.upper() == "XD":
            return True
        if s[0] in ":;=8" and s[1] in ")(DPpOo3<>/\\|*[]":
            return True
        return False

    # Western emoticon: 3-4 chars starting with ``:``/``;``/``=``/``8``,
    # optional 1-char nose, then 1-2 mouth chars.
    if n <= 4 and s[0] in ":;=8":
        rest = s[1:]
        if rest and rest[0] in "-^o'":
            rest = rest[1:]
        if rest and all(c in ")(DPpOo3<>/\\|*[]" for c in rest):
            return True

    # Symmetric "EYE MOUTH EYE": 3+ chars, eyes match (or paired
    # bracket pair), interior is all mouth glyphs.
    #
    # The "distinct eye" check rejects strings of pure mouth chars
    # without distinct eyes (`___`, `...`, `---`) by requiring the
    # first character not to appear anywhere in the interior. The
    # earlier "first must not be a mouth char" rule rejected legit
    # faces like ``^_^`` and ``|_|`` because their eyes (`^`, `|`)
    # are technically in the mouth set; the in-interior check
    # accepts those while still rejecting fully-uniform sequences.
    interior = s[1:-1]
    if not interior:
        return False
    if not _BARE_KAOMOJI_MOUTH_RE.fullmatch(interior):
        return False
    first, last = s[0], s[-1]
    if first in interior:
        return False
    if first == last:
        return True
    if (first, last) in _BARE_KAOMOJI_PAIRED_EYES:
        return True
    return False


def is_kaomoji_candidate(s: str, *, max_len: int = _KAOMOJI_MAX_LEN) -> bool:
    """Return True iff `s` looks like a real kaomoji prefix.

    Used by `extract` and the journal-prefix validators (live-hook
    Python mirror, backfill replay) to reject prose, markdown-escape
    artifacts, and truncated junk that the leading-prefix sed
    pipeline would otherwise let through.

    Rules:
      Universal (all must pass):
        - length 2..`max_len`
        - no ASCII backslash *except* at position 0 — backslash at
          position 0 is the wing-hand pattern (``\\(^o^)/``); backslash
          anywhere else is a markdown-escape artifact (e.g.
          ``(\\*´∀｀\\*)`` came from a model emitting a literal ``\\*``
          that it treated as Markdown escape).
        - no run of 4+ consecutive ASCII letters (prose)
      Path A (existing v1.0+/v2.0 leader-char path):
        - first char ∈ `KAOMOJI_START_CHARS`
      Path B (v2.0 round-6 speculative bare-kaomoji extension):
        - matches `_looks_like_bare_kaomoji` shape (symmetric
          `EYE MOUTH EYE`, paired-eye, or Western emoticon)

    Bracket balance is *not* enforced (Path A). Real corpus output is
    sometimes unbalanced — variant kaomoji where the closing glyph
    isn't strictly the matching bracket — and the previous balance
    check over-rejected valid entries. The length cap, the
    4-letter-run rule, and the backslash filter together carry the
    prose-rejection role.

    v2.0 round-6 (Path B): added bare-kaomoji shape detection so
    models whose register strips the parenthesizing wrapper (e.g.
    granite emitting bare ``ಥ﹏ಥ`` for grief prompts) surface their
    affective output through `extract`. See
    `_looks_like_bare_kaomoji` for the shape rules.

    v2.0 (was: ``"\\\\" in s``): backslash filter relaxed to allow a
    leading wing. v1 rejected ``\\(^o^)/`` along with the markdown
    artifacts; v2 accepts the former and still rejects the latter
    (markdown escape produces ``\\X`` at position >= 1, never 0).
    """
    if not (2 <= len(s) <= max_len):
        return False
    if "\\" in s[1:]:
        return False
    if _LETTER_RUN_RE.search(s):
        return False
    if s[0] in KAOMOJI_START_CHARS:
        return True
    if _looks_like_bare_kaomoji(s):
        return True
    return False


@dataclass(frozen=True)
class KaomojiMatch:
    """Result of running `extract` against a generated text.

    Slim public shape: just the validated leading span. Pre-v1.0
    versions also reported a `kaomoji` (taxonomy match) and `label`
    (+1/-1/0 affect pole) — those are now research-side
    (`llmoji_study.taxonomy_labels.LabeledKaomojiMatch`) because the
    underlying TAXONOMY dict is gemma-tuned and not part of the
    provider-agnostic public package.
    """
    first_word: str  # validated leading kaomoji span, or ""


def _leading_bracket_span(text: str) -> str:
    """Return the leading kaomoji span of `text`.

    For bracket-leading inputs, prefer a balanced-paren span — that's
    how whitespace-padded kaomoji like ``(｡˃ ᵕ ˂ )`` surface intact.
    When the depth-walker hits the length cap or short-circuits on a
    `depth < 0` without ever closing, fall back to a
    whitespace-delimited word capped at ``_KAOMOJI_MAX_LEN``. Real
    corpus output is sometimes unbalanced (closing glyph isn't
    strictly the matching bracket); the fallback keeps those
    entries instead of dropping them on the floor.

    For non-bracket-leading inputs (``ヽ``, ``ᕕ``, etc.), the span
    is just the first whitespace-delimited word capped at the length
    limit.

    Returns `""` when the candidate fails `is_kaomoji_candidate` —
    prose, markdown-escape artifacts, oversize spans collapse to the
    empty string rather than producing nonsense ``first_word``
    values that downstream consumers would have to re-filter.
    """
    stripped = text.lstrip()
    if not stripped:
        return ""
    candidate = ""
    if stripped[0] in _OPEN_BRACKETS:
        depth = 0
        closed = False
        for i, c in enumerate(stripped):
            if c in _OPEN_BRACKETS:
                depth += 1
            elif c in _CLOSE_BRACKETS:
                depth -= 1
                if depth == 0:
                    candidate = stripped[: i + 1]
                    closed = True
                    break
                if depth < 0:
                    break
            if i + 1 >= _KAOMOJI_MAX_LEN:
                # Past the length cap with no clean close.
                break
        if not closed:
            # Unbalanced bracket-leading kaomoji — fall back to a
            # whitespace-delimited word (capped at _KAOMOJI_MAX_LEN)
            # so we don't drop real corpus entries whose closing
            # glyph isn't the matching bracket.
            idx = 0
            while idx < len(stripped) and not stripped[idx].isspace():
                idx += 1
                if idx >= _KAOMOJI_MAX_LEN:
                    break
            candidate = stripped[:idx]
    else:
        idx = 0
        while idx < len(stripped) and not stripped[idx].isspace():
            idx += 1
            if idx >= _KAOMOJI_MAX_LEN:
                break
        candidate = stripped[:idx]

    if candidate and is_kaomoji_candidate(candidate):
        return candidate
    return ""


def extract(text: str) -> KaomojiMatch:
    """Identify the leading kaomoji in a generated text.

    Returns `KaomojiMatch(first_word="")` for plain prose /
    non-kaomoji input — see `is_kaomoji_candidate` for the rejection
    rules.
    """
    return KaomojiMatch(first_word=_leading_bracket_span(text.lstrip()))


# ---------------------------------------------------------------------------
# Canonicalization: collapse trivial kaomoji variants
# ---------------------------------------------------------------------------
#
# Two kaomoji can differ in five cosmetic-only ways that we collapse, and one
# semantically-meaningful way that we preserve.
#
# Cosmetic (collapsed):
#
#   A. Invisible format characters: U+2060 WORD JOINER, U+200B/C/D zero-width
#      space/non-joiner/joiner, U+FEFF byte-order mark, U+0602 ARABIC
#      FOOTNOTE MARKER. Qwen occasionally emits these between every glyph
#      of a kaomoji, e.g. `(⁠◕⁠‿⁠◕⁠✿⁠)` is the
#      same expression as `(◕‿◕✿)`.
#   B. Half-width vs full-width punctuation: `>`/`＞`, `<`/`＜`, `;`/`；`,
#      `:`/`：`, `_`/`＿`, `*`/`＊`. Hand-picked over NFKC because
#      NFKC also compatibility-decomposes `´` and `˘` into space + combining
#      marks, which destroys eye glyphs in `(っ´ω`)` and `(˘▽˘)`.
#   C. Internal whitespace inside the bracket span: `( ; ω ; )` is the same
#      as `(；ω；)`. Strip only ASCII spaces; non-ASCII spacing characters
#      are part of the face.
#   D. Cyrillic case: `Д`/`д` co-occur in the same `(；´X｀)` distressed-face
#      skeleton at near-50/50 ratio, so the model isn't choosing between
#      them semantically. Lowercase all Cyrillic capitals U+0410–U+042F.
#   E. Near-identical glyph pairs:
#        E1. Degree-like circular eyes/decorations: `°` (U+00B0 DEGREE SIGN),
#            `º` (U+00BA MASCULINE ORDINAL), `˚` (U+02DA RING ABOVE) all fold
#            to `°`. Gemma's `(°Д°)` and `(ºДº)` are the same shocked face.
#        E2. Middle-dot variants: `・` (U+30FB KATAKANA MIDDLE DOT) and `･`
#            (U+FF65 HALFWIDTH KATAKANA MIDDLE DOT) fold to `・`. Qwen's
#            `(´・ω・`)` and `(´･ω･`)` are the same expression. Smaller-size
#            middle dots (`·` U+00B7, `⋅` U+22C5) are NOT folded — they
#            could plausibly be a distinct register.
#   F. Hand/arm modifiers at face boundaries: `(๑˃ᴗ˂)ﻭ` vs `(๑˃ᴗ˂)`,
#      `(っ˘▽˘ς)` vs `(っ˘▽˘)`. Stripped at the bracket boundary only —
#      same face with or without an arm reaching out.
#
# Semantically meaningful (preserved):
#
#   * Eye / mouth / decoration changes that aren't covered by E1/E2 above.
#     `(◕‿◕)` vs `(♥‿♥)` vs `(✿◕‿◕｡)` are distinct expressions.
#   * Borderline mouth-glyph case `ᴗ` vs `‿` is unified to `‿` since the
#     model emits both in the same `(｡ᵕXᵕ｡)` skeleton with no distinct
#     register.
#
# Order of operations matters:
#   1. NFC normalize (preserves `´`, `˘`, `｡` which NFKC would mangle).
#   2. Strip invisible / cosmetic-overlay characters (A + G) — must be
#      early so they don't interfere with subsequent regex / equality
#      checks.
#   3. Apply `_TYPO_SUBS` (B half/full-width + E1 degree + E2 middle-dot
#      + H curly-quote + I bullet→middle-dot + J bracket-corner-circle).
#   4. Strip internal whitespace (C).
#   5. Cyrillic case fold (D).
#   6. Apply ``_INTERNAL_SUBS`` substring substitutions (K
#      ``・-・`` → ``・_・``).
#   7. Strip arm modifiers (F + L).
#
# New rules added 2026-04-27 to catch cosmetic variants that survived
# the rules-A-through-F pass:
#
#   G. Combining strikethrough overlays U+0335–U+0338 over an eye
#      glyph: ``(๑˃̵‿˂̵)`` and ``(๑˃‿˂)`` are the same expression,
#      with U+0335 (COMBINING SHORT STROKE OVERLAY) cosmetic-only.
#      Treated like rule A invisibles.
#   H. Curly quotes fold to ASCII straight quotes:
#        U+2018/U+2019 (single) → ``'`` (U+0027)
#        U+201C/U+201D (double) → ``"`` (U+0022)
#      ``┐('～`;)┌`` and ``┐(‘～`;)┌`` are the same expression with
#      different leading-quote glyphs.
#   I. Bullet ``•`` (U+2022) → middle-dot ``・`` (U+30FB).
#      ``(´•ω•`)`` and ``(´・ω・`)`` share the same skeleton; the
#      bullet glyph is bigger but in this corpus they're being used
#      interchangeably.
#   J. Bracket-corner circle ``◍`` (U+25CD CIRCLE WITH VERTICAL FILL)
#      → ``｡`` (U+FF61). ``(◍•‿•◍)`` and ``(｡•‿•｡)`` share the
#      skeleton. This is the most aggressive of the new rules — the
#      glyphs differ in size more than the others — but in the
#      corpus the role they play (bracket-corner decoration flanking
#      the body) is identical.
#   K. ``・-・`` substring → ``・_・``. Targeted; preserves
#      ``(´-ω-`)`` (where the ``-`` is a tired-eye glyph between
#      ``´`` and ``ω``, not a mouth between two eyes).
#   L. ``*`` ASCII asterisk at face-boundary positions becomes a rule-F
#      arm modifier (alongside ``っ``, ``c``, ``ς``, ``ﻭ``).
#      ``(*•̀‿•́*)`` collapses to ``(•̀‿•́)``.

# Arm/hand/decoration modifiers that appear OUTSIDE the closing paren.
# v2.0 strips the full set of paired-arm and pose-arm patterns to the
# bare face. Each char below is the trail half of one of these
# patterns:
#   ﻭ            (๑˃ᴗ˂)ﻭ           cheering (Arabic waw)
#   っ            (っ╥﹏╥)っ          reaching tsu
#   /            (´∀`)/             wing-hand right
#   ⊂            (˘ω˘)⊂             hugging arm right (matched ⊂...⊂)
#   ⊃            ⊂(◕‿◕)⊃           hugging arm right (matched ⊂...⊃)
#   ✧            ✧(ˊᗜˋ)✧            sparkle right
#   ۶            ٩(◕‿◕)۶            cheering Arabic-Indic six
#   ᕗ            ᕕ(ᐛ)ᕗ              running Canadian syllabics hoi
#   ७            ໒(◕‿◕)७            cradling Devanagari seven
#   ψ Ψ          ψ(`Д´)ψ            horn-fingers right (lower/upper psi)
#   з            ε(◕‿◕)з            kiss-close (Cyrillic ze)
#   ʃ            ƪ(˘⌣˘)ʃ            raised-hand right (Latin esh)
#   ╱            ╲(◕‿◕)╱            heavy-line wing right
#   ノ ﾉ          ヽ(´ー`)ノ          raised-hand right (katakana no /
#                                    halfwidth)
#   ╯ ╮ ╭        ╰(´∀`)╯ ╭(´∀`)╮   box-drawing pose-arm closes
#   ┌ ┐ ┘ └      ┐(´д`)┌            box-drawing shrug closes (with
#                                    inverted-pattern siblings)
#   ¯            ¯\_(ツ)_/¯          shrug macron right
#   _            ¯\_(ツ)_/¯          shrug underscore right
# Round-4 additions (decorator-arm halves of the round-4 leader
# additions):
#   」』】〉》   「(゜～゜)」  etc.    Japanese corner-bracket trail
#                                    wrappers (paired with
#                                    `「『【〈《` lead).
#   ♪ ♫ ♬     ♪(´▽｀)♪              music-note decorator right
#   ♥ ♡ ❤     ♥(◕‿◕)♥              heart decorator right
#   ★ ☆       ★(◕‿◕)★              star decorator right
# Round-5 additions (decorator-arm halves of the round-5 leader
# additions; symmetric pairings appear in both arm sets):
#   ᕤ            ᕦ(ò_óˇ)ᕤ           flex pose right arm (paired ᕦ
#                                    in lead set; ᕗ already in v1
#                                    trail set covers ᕙ-led poses)
#   ୨            ୧(˃ᗨ˂)୨            Oriya cradle pose right arm
#   〕            〔(◕‿◕)〕           tortoise-shell close
#   ✿ ❀         ✿(◕‿◕)✿              flower decorator right
#   ❣ ❥         ❣(◕‿◕)❣              heart variant decorator right
#   ✦ ✩ ✪       ✦(◕‿◕)✦              star variant decorator right
#   ♩            ♩(◕‿◕)♩              quarter-note decorator right
#   ※            ※(◕‿◕)※              editorial decorator right
# Box-drawing chars appear in BOTH lead and trail because the
# pose can be mirrored (``╮(´д`)╭`` is the inverted shrug); same
# for ``¯`` and ``_`` in the shrug pattern. The regex anchors mean
# this only fires at the very start (before ``(``) or very end
# (immediately after ``)``, via the ``(?<=\))`` lookbehind on
# ``_TRAIL_OUTSIDE_RE``), so eye/mouth glyphs like ``_`` in
# ``(◕_◕)`` and ``╯`` in the rage-cheek of ``(╯°□°)╯`` stay
# untouched, AND so corner-bracket-only-wrapped standalone faces
# like ``「・_・」`` (no inner paren to anchor against) keep their
# trailing wrapper instead of asymmetric truncation to ``「・_・``.
_ARM_OUTSIDE = (
    "ﻭっ/⊂⊃✧۶ᕗ७ψΨзʃ╱ノﾉ╯╮╭┌┐┘└¯_"
    "」』】〉》"   # round 4: Japanese corner-bracket close wrappers
    "♪♫♬"        # round 4: music-note decorator right
    "♥♡❤"        # round 4: heart decorator right
    "★☆"         # round 4: star decorator right
    "ᕤ୨"         # round 5: flex / Oriya cradle right arms
    "〕"         # round 5: tortoise-shell close
    "✿❀"         # round 5: flower decorator right
    "❣❥"         # round 5: heart variant decorator right
    "✦✩✪"        # round 5: star variant decorator right
    "♩"          # round 5: quarter-note decorator right
    "※"          # round 5: editorial decorator right
)
# Arm/hand/decoration modifiers that appear OUTSIDE the opening paren.
# Mirror set to ``_ARM_OUTSIDE`` for the lead halves of the same
# paired-arm patterns (plus ``Σ`` which is single-arm — shocked
# sigma has no paired close):
#   \           \(^o^)/             wing-hand left
#   ⊂           ⊂(face)⊂            hugging arm left
#   ✧           ✧(face)✧            sparkle left
#   Σ           Σ(°△°|||)           shocked sigma (single-arm)
#   ψ Ψ         ψ(`Д´)ψ             horn-fingers left
#   ε           ε(◕‿◕)з             kiss-open
#   ƪ           ƪ(˘⌣˘)ʃ             raised-hand left
#   ╲           ╲(◕‿◕)╱             heavy-line wing left
#   ٩           ٩(◕‿◕)۶             cheering left
#   ᕕ           ᕕ(ᐛ)ᕗ               running left
#   ໒           ໒(◕‿◕)७             cradling left
#   ヽ ヾ        ヽ(´ー`)ノ           raised-hand left (v2.0 BREAKS
#                                    v1 — was pinned as preserved
#                                    pose by rule O test, now collapses)
#   ╰ ╭ ╮ ┐ ┌   ╰(´∀`)╯  ┐(´д`)┌    box-drawing pose leaders
#   ¯ \ _       ¯\_(ツ)_/¯           shrug components
# Round-4 lead-half additions (mirror of the round-4 _ARM_OUTSIDE
# additions):
#   「『【〈《   「(゜～゜)」  etc.    Japanese corner-bracket lead
#   └ ┘         └(°▽°)┘              box-drawing standing-pose lead
#                                     (┘...└ inverted form too)
#   ♪ ♫ ♬     ♪(´▽｀)               music-note decorator left
#   ♥ ♡ ❤     ♥(◕‿◕)               heart decorator left
#   ★ ☆       ★(◕‿◕)               star decorator left
# Round-5 lead-half additions (mirror of the round-5 _ARM_OUTSIDE
# additions; symmetric pairings appear in both arm sets):
#   ᕦ ᕙ        ᕦ(ò_óˇ)ᕤ / ᕙ(`▿´)ᕗ   flex / strong-feel lead arms
#   ୧            ୧(˃ᗨ˂)୨              Oriya cradle pose left arm
#   〔            〔(◕‿◕)〕           tortoise-shell open
#   ✿ ❀         ✿(◕‿◕)✿              flower decorator left
#   ❣ ❥         ❣(◕‿◕)❣              heart variant decorator left
#   ✦ ✩ ✪       ✦(◕‿◕)✦              star variant decorator left
#   ♩            ♩(◕‿◕)♩              quarter-note decorator left
#   ※            ※(◕‿◕)※              editorial decorator left
# Distinct from inside-leading modifiers (``っ``/``*``) which sit
# BETWEEN ``(`` and face content (``(っ╥﹏╥)``, ``(*•̀‿•́*)``).
# Note: ``ʢ`` (alternate bear-bracket open) is NOT in this set —
# like ``ʕ``, the bear-shape IS the kaomoji and we preserve the
# whole span; the ``(?=\()`` lookahead on ``_LEAD_OUTSIDE_RE`` would
# fail anyway for `ʢ◉ᴥ◉ʡ` (no inner paren), but we keep ``ʢ`` out
# of the set explicitly to mirror the ``ʕ`` rule even if a future
# corpus contains paren-wrapped variants like `ʢ(◉ᴥ◉)ʡ`.
_ARM_OUTSIDE_LEAD = (
    "\\⊂✧ΣψΨεƪ╲٩ᕕ໒ヽヾ╰╭╮┐┌¯_"
    "「『【〈《"   # round 4: Japanese corner-bracket lead wrappers
    "└┘"          # round 4: box-drawing standing-pose lead
    "♪♫♬"         # round 4: music-note decorator left
    "♥♡❤"         # round 4: heart decorator left
    "★☆"          # round 4: star decorator left
    "ᕦᕙ୧"        # round 5: flex / strong-feel / Oriya cradle leads
    "〔"          # round 5: tortoise-shell open
    "✿❀"          # round 5: flower decorator left
    "❣❥"          # round 5: heart variant decorator left
    "✦✩✪"         # round 5: star variant decorator left
    "♩"           # round 5: quarter-note decorator left
    "※"           # round 5: editorial decorator left
)
# Arm/hand modifiers that appear just INSIDE the closing paren:
#   (っ˘▽˘ς)  (っ´ω`c)  (*•̀‿•́*)
_ARM_INSIDE_TRAIL = "ςc*"
# Arm/hand modifiers that appear just INSIDE the opening paren (leading):
#   (っ╥﹏╥)  (*•̀‿•́*)
_ARM_INSIDE_LEAD = "っ*"

# ``(?<=\))`` lookbehind: trail-arm strips only fire when the run
# they'd consume is immediately preceded by ``)``. Required by the
# round-4 corner-bracket additions — without it, ``「・_・」``
# (corner-bracket-only-wrapped face, no inner paren) would have its
# closing ``」`` stripped to leave ``「・_・``. With the lookbehind,
# the trail strip only fires when there's a ``)`` to anchor against
# (i.e. a paren-wrapped face), which matches every existing v1/v2
# test case (verified: ``(´∀`)/``, ``(˘ω˘)⊂``, ``(っ╥﹏╥)っ``,
# ``(╯°□°)╯``, ``(ツ)_/¯``, ``╰(´∀`)╯`` etc. all have ``)``
# immediately before the trail run).
_TRAIL_OUTSIDE_RE = re.compile(rf"(?<=\))[{re.escape(_ARM_OUTSIDE)}]+$")
_LEAD_OUTSIDE_RE = re.compile(rf"^[{re.escape(_ARM_OUTSIDE_LEAD)}]+(?=\()")
_TRAIL_INSIDE_RE = re.compile(rf"[{re.escape(_ARM_INSIDE_TRAIL)}]+\)$")
_LEAD_INSIDE_RE = re.compile(rf"^\([{re.escape(_ARM_INSIDE_LEAD)}]+")

# Rules A + G: invisible / cosmetic-overlay format characters that
# interleave kaomoji glyphs without changing the expression.
#   A: U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER,
#      U+200D ZERO WIDTH JOINER, U+2060 WORD JOINER,
#      U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM,
#      U+0602 ARABIC FOOTNOTE MARKER (observed as a stray byte between
#      ``>`` and ``<`` in Qwen ``(๑>؂<๑)``).
#   G: U+0334 COMBINING TILDE OVERLAY,
#      U+0335 COMBINING SHORT STROKE OVERLAY,
#      U+0336 COMBINING LONG STROKE OVERLAY,
#      U+0337 COMBINING SHORT SOLIDUS OVERLAY,
#      U+0338 COMBINING LONG SOLIDUS OVERLAY,
#      U+033F COMBINING DOUBLE OVERLINE — strikethrough / overlay
#      combining marks that occasionally land on eye glyphs
#      (``˃̵``, ``˂̿`` etc.). Stripped narrowly across this set;
#      broader stripping of combining marks (U+0300–U+036F) would
#      destroy intentional accent eye glyphs in ``(•̀_•́)``
#      (U+0300 GRAVE / U+0301 ACUTE).
_INVISIBLE_CHARS = (
    "​‌‍⁠﻿؂"  # rule A
    "̴̵̶̷̸̿"               # rule G
)

# Hand-picked typographic / glyph substitutions. Hand-picked over NFKC
# because NFKC also compatibility-decomposes `´` (acute) and `˘` (breve)
# into space + combining marks, mangling eye glyphs in `(っ´ω`)` and
# `(˘▽˘)`. NFC leaves those intact; we then apply just the specific
# compatibility-equivalences we want.
_TYPO_SUBS: tuple[tuple[str, str], ...] = (
    # === Brackets and arm-modifier glyphs ===
    ("）", ")"),   # full-width close paren
    ("（", "("),   # full-width open paren
    ("ｃ", "c"),   # full-width Latin c (arm modifier)
    # === Punctuation: half/full-width pairs (rule B) ===
    ("＞", ">"),   # FULLWIDTH GREATER-THAN SIGN
    ("＜", "<"),   # FULLWIDTH LESS-THAN SIGN
    ("；", ";"),   # FULLWIDTH SEMICOLON
    ("：", ":"),   # FULLWIDTH COLON
    ("＿", "_"),   # FULLWIDTH LOW LINE
    ("＊", "*"),   # FULLWIDTH ASTERISK
    # NOT folded: `￣` (FULLWIDTH MACRON U+FFE3) is a flat horizontal
    # line, used as a closed-eye-looking-up glyph in
    # `(￣ω￣)` / `(￣ー￣)` (calm/placid register). `~` (TILDE) is wavy,
    # used in `(~ω~)` / `(~▽~)` (sleepy register). Distinct shapes
    # and distinct affect — folding them together loses the
    # register difference.
    ("｀", "`"),   # FULLWIDTH GRAVE ACCENT -> ASCII GRAVE (rule O).
                   # `ヽ(´ー`)ノ` ↔ `ヽ(´ー｀)ノ` differ only in this.
    # Speculative B extensions (none observed in corpus yet, added
    # for halfwidth/fullwidth coverage symmetry with the rest of
    # the FF0x/FF1x block; future-proofing):
    ("？", "?"),   # FULLWIDTH QUESTION MARK
    ("！", "!"),   # FULLWIDTH EXCLAMATION MARK
    ("．", "."),   # FULLWIDTH FULL STOP (distinct from `。` halfwidth
                   # ideographic full stop — `．` is the romance-period
                   # variant)
    ("，", ","),   # FULLWIDTH COMMA
    ("／", "/"),   # FULLWIDTH SOLIDUS
    ("～", "~"),   # FULLWIDTH TILDE — current corpus has the mixed
                   # `(~～~;)` form, internally inconsistent; folding
                   # gives `(~~~;)` and prevents future divergence.
    # === Quotes: curly -> ASCII straight (rule H) ===
    ("‘", "'"),  # LEFT SINGLE QUOTATION MARK
    ("’", "'"),  # RIGHT SINGLE QUOTATION MARK
    ("“", '"'),  # LEFT DOUBLE QUOTATION MARK
    ("”", '"'),  # RIGHT DOUBLE QUOTATION MARK
    # === Eye-glyph equivalence class: directional fill -> ◕ ===
    # Half/quarter-fill circle variants — "round eye with interior
    # fill in some direction", visually suggesting looking-direction.
    # Subsumes the earlier targeted mirror rule `(◑‿◐)` ↔ `(◐‿◑)`.
    ("◔", "◕"),   # CIRCLE WITH UPPER RIGHT QUADRANT BLACK
    ("◑", "◕"),   # CIRCLE WITH RIGHT HALF BLACK
    ("◐", "◕"),   # CIRCLE WITH LEFT HALF BLACK
    # Speculative extensions to the directional-fill class (not
    # observed in corpus):
    ("◒", "◕"),   # CIRCLE WITH LOWER HALF BLACK
    ("◓", "◕"),   # CIRCLE WITH UPPER HALF BLACK
    ("◖", "◕"),   # LEFT HALF BLACK CIRCLE (full-circle variant)
    ("◗", "◕"),   # RIGHT HALF BLACK CIRCLE (full-circle variant)
    # === Eye-glyph equivalence class: filled-with-pupil -> ⊙ ===
    # Distinct from the directional-fill class — these glyphs look
    # like a circle with a visible interior pupil/center dot
    # (target / wide-open / shocked-eye register), not a directional
    # fill.
    ("◉", "⊙"),   # FISHEYE (Geometric Shapes block) -> CIRCLED DOT
    # Speculative extension (not observed in corpus):
    ("●", "⊙"),   # BLACK CIRCLE (fully solid)
    # === Eye-/decoration-glyph equivalence class: degree-like -> ° (rule E1) ===
    ("º", "°"),   # MASCULINE ORDINAL INDICATOR
    ("˚", "°"),   # RING ABOVE
    # === Middle-dot equivalence class: -> ・ (rule E2 + I) ===
    ("･", "・"),   # HALFWIDTH KATAKANA MIDDLE DOT
    ("•", "・"),   # BULLET (U+2022)
    # === Mouth-glyph equivalence class: smile-curve -> ‿ (rules 3 + M + N) ===
    ("ᴗ", "‿"),   # LATIN SMALL LETTER OPEN O / connector
    ("◡", "‿"),   # LOWER HALF CIRCLE
    ("ᵕ", "‿"),   # LATIN SMALL LETTER UP TACK
    ("⌣", "‿"),   # SMILE (U+2323) — direct synonym for the
                   # smile-mouth role.
    # === Mouth-line distinction (NO fold) ===
    # `﹏` (SMALL WAVY LOW LINE U+FE4F) and `_` (ASCII UNDERSCORE) are
    # NOT interchangeable. `﹏` is wavy/distressed (`(>﹏<)`,
    # `(╥﹏╥)`); `_` is flat/neutral (`(•_•)`, `(◕_◕)`).
    # === Bracket-corner-decoration equivalence class: -> ｡ (rule J + B-extension) ===
    ("◍", "｡"),   # CIRCLE WITH VERTICAL FILL (U+25CD)
    ("。", "｡"),   # IDEOGRAPHIC FULL STOP -> halfwidth (matches J's canonical)
)

# Rule K: substring-level substitutions applied AFTER `_TYPO_SUBS` so
# that `•` → `・` has already happened, and AFTER internal-whitespace
# stripping. Targeted to avoid global `-` ↔ `_` folds that would
# corrupt `(´-ω-`)` (where `-` is a tired-eye glyph).
_INTERNAL_SUBS: tuple[tuple[str, str], ...] = (
    # Middle-dot eyes with hyphen mouth -> middle-dot eyes with
    # underscore mouth. Targeted: `(・-・)` ↔ `(・_・)`.
    ("・-・", "・_・"),
)


# Combined translation table: invisibles (rule A + G) → delete,
# typo-subs (rules B / E1 / E2 / H / I / J + arm/paren folds) →
# replace, Cyrillic upper (rule D) → lower. Built once at import,
# applied in a single ``str.translate`` pass per call (one O(n)
# string scan instead of ~30 full-string ``replace`` walks plus a
# regex sub plus a per-char Cyrillic-lower comprehension). The
# substitutions don't chain (no destination char is also a source)
# so the iterative + table forms are equivalent.
def _build_translation_table() -> dict[int, int | None]:
    table: dict[int, int | None] = {}
    # Invisibles → delete (rules A + G).
    for c in _INVISIBLE_CHARS:
        table[ord(c)] = None
    # Typo subs (single-char → single-char). All entries are 1→1
    # and no destination char appears as a source elsewhere, so the
    # iterative ``replace`` form and a single-pass translate are
    # equivalent.
    for src, dst in _TYPO_SUBS:
        table[ord(src)] = ord(dst)
    # Cyrillic capitals → lower (rule D).
    for cp in range(0x0410, 0x0430):
        table[cp] = cp + 0x20
    return table


_TRANSLATE_TABLE = _build_translation_table()


def canonicalize_kaomoji(s: str) -> str:
    """Collapse trivial kaomoji variants to a canonical form.

    Applies, in order:
      1. NFC normalization (preserves `´`, `˘`, `｡` which NFKC would mangle).
      2. Single ``str.translate`` pass folding:
           * invisible / cosmetic-overlay chars (rule A + G — U+200B/C/D,
             U+2060, U+FEFF, U+0602, U+0335–U+0338) → deleted.
           * ``_TYPO_SUBS`` substitutions (rules B / E1 / E2 / H / I / J
             plus existing arm/paren folds).
           * Cyrillic capitals (rule D) → lowercase.
      3. Strip ASCII spaces inside the `(...)` bracket span (rule C).
      4. Apply ``_INTERNAL_SUBS`` substring substitutions (rule K
         ``・-・`` → ``・_・``).
      5. Strip arm modifiers from face boundaries (rule F + L —
         ``っ ς c ﻭ *``).

    Eye/mouth/decoration changes that aren't covered by rules
    E1/E2/I/J are preserved.

    Idempotent: ``canonicalize_kaomoji(canonicalize_kaomoji(s)) == canonicalize_kaomoji(s)``.

    Empty input returns ``""``.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s.strip())
    s = s.translate(_TRANSLATE_TABLE)
    if s.startswith("(") and s.endswith(")"):
        s = "(" + s[1:-1].replace(" ", "") + ")"
    for src, dst in _INTERNAL_SUBS:
        s = s.replace(src, dst)
    # Strip outside-paren leading and trailing arm chars first so the
    # inside-paren detection sees the open/close parens unobscured.
    # v2.0: ``_LEAD_OUTSIDE_RE`` collapses wing-hand ``\(^o^)/`` and
    # hugging-arm ``⊂(face)⊂`` patterns to the bare face.
    s = _LEAD_OUTSIDE_RE.sub("", s)
    s = _TRAIL_OUTSIDE_RE.sub("", s)
    s = _LEAD_INSIDE_RE.sub("(", s)
    s = _TRAIL_INSIDE_RE.sub(")", s)
    return s


