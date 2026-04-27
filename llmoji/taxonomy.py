"""Kaomoji taxonomy for the pilot.

Two parallel dicts, one per axis under test:
  TAXONOMY           — happy.sad labels (+1 happy, -1 sad)
  ANGRY_CALM_TAXONOMY — angry.calm labels (+1 angry, -1 calm)

Both map kaomoji-string → int pole. A kaomoji may appear in one dict,
both, or neither (the "other" bucket). ``extract()`` returns the
happy.sad match for back-compat with v1 analysis; ``label_on(axis,
form)`` is the generic accessor.

Both sets were seeded from eriskii's Claude-faces catalog
(https://eriskii.net/projects/claude-faces) and extended in place after
observing gemma-4-31b-it's actual emissions. Locked taxonomies imply
reproducibility across runs; extending after a taxonomy edit requires
re-labeling the existing ``pilot_raw.jsonl`` (see CLAUDE.md).

The model's dialect preferences are distinct per steering direction —
``(｡X｡)`` bracket-dots under natural happy, ``(._.)`` ASCII under
strong sad, likely ``(ಠ益ಠ)``-family under strong angry. Always run
``00_vocab_sample.py``-style inspection before locking a new axis.

Extractor notes:
  - Primary lookup is exact longest-prefix match against TAXONOMY.
  - Fallback is a balanced-paren span, so whitespace-padded kaomoji
    like ``(｡˃ ᵕ ˂ )`` surface with a human-readable first_word even
    when they miss the taxonomy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Built from observed output across all four arms of the pilot run
# (see data/pilot_raw.jsonl). The model's kaomoji dialect shifts
# substantially under steering:
#   - unsteered: Japanese-style (｡X｡) bracket-dots form dominates
#   - happy-steered: simpler bracket forms, flower/hug decorations
#   - sad-steered: collapses to ASCII minimalism ((._.) family)
# Any form observed ≥2 times in any condition and visually unambiguous
# is included below. Forms that are clearly model-corruption
# (e.g. '(｡•impresa•)' — the Italian word 'impresa' appearing inside
# the kaomoji at high-α sad-steering) are excluded.
TAXONOMY: dict[str, int] = {
    # --- happy pole: default dialect ---
    "(｡◕‿◕｡)":    +1,   # unsteered default happy
    "(๑˃ᴗ˂)ﻭ":   +1,   # enthusiastic / cheering
    "(✿◠‿◠)":     +1,   # flower-adorned gentle warm
    "(｡♥‿♥｡)":    +1,   # very happy / loving
    "(｡◕ᴗ◕｡)":    +1,   # dialect variant
    "(｡^‿^｡)":    +1,   # dialect variant
    "(｡˃ ᵕ ˂ )":  +1,   # whitespace-padded happy
    "(ﾉ◕ヮ◕)":    +1,   # throwing-arms happy
    "(☀️‿☀️)":     +1,   # sun-eyed happy
    "(っ´ω`)":    +1,   # hugging with cute face

    # --- happy pole: steered / simpler-dialect variants ---
    "(◕‿◕)":      +1,   # simple smile-eyed happy (dominant under happy-steering)
    "(✿◕‿◕)":     +1,   # flower + simple smile
    "(づ｡◕‿◕｡)":  +1,   # reaching/hugging with happy face
    "(๑˃ᴗ˃)":    +1,   # enthusiastic (variant mouth direction)
    "(✿^▽^)":    +1,   # triangular-smile with flower
    "( ^v^ )":    +1,   # caret-eyed simple smile
    "(✿˃ᴗ˃)":    +1,   # enthusiastic with flower

    # --- sad pole: default dialect ---
    "(｡•́︿•̀｡)":   -1,   # unsteered default sad (pouty)
    "(｡╯︵╰｡)":    -1,   # downcast
    "(っ╥﹏╥)っ":   -1,   # crying / needs-hug
    "(｡T_T｡)":    -1,   # dialect variant
    "(｡ŏ﹏ŏ｡)":    -1,   # dialect variant (pouty-fearful)
    "(｡•́﹏•̀｡)":   -1,   # dialect variant (pouty, alt mouth)

    # --- sad pole: steered / minimalist-dialect variants ---
    "(._.)":      -1,   # ASCII minimalist, dominant under sad-steering
    "( . .)":     -1,   # spaced minimalist
    "( . . )":    -1,   # wider-spaced minimalist
    "( ._.)":     -1,   # leading-space minimalist
    "( . . . )":  -1,   # triple-dot minimalist
    "( . _ . )":  -1,   # spaced ASCII sad
    "( ˙ ˙ ˙ )":  -1,   # dot-trail minimalist
    "(｡ ﹏ ｡)":    -1,   # closed-eyes crying
    "(｡△｡)":      -1,   # triangle-mouth sad
    "(｡•﹏•)":    -1,   # simpler pout
    "(｡╥｡)":      -1,   # tear-eye
    "(｡ ﾟ ｡)":    -1,   # whimper
    "( ｡ ｡ )":    -1,   # minimal bracket-dots
    "( •_• )":    -1,   # spaced blank-sad
    "(っ╥╯﹏╰╥)":  -1,   # crying with multiple tears
    "(っ˘̩╭╮˘̩)":   -1,   # closed-eye sad hug

    # --- happy pole: additional hugging / decorated variants ---
    "(っ´ω`c)":   +1,   # reaching hug with cute face
    "(っ´ω` )":   +1,   # hug variant
    "(っ´ω`ｃ)":  +1,   # hug variant (fullwidth c)
    "(✿˃ᴗ˃)":    +1,   # enthusiastic with flower (variant)
}

POLE_NAMES = {+1: "happy", -1: "sad", 0: "other"}

# Parallel dict for the angry.calm axis. Seeded from eriskii's catalog;
# candidate forms to expect the model emitting under ±0.5 angry/calm
# steering. Expect to extend post-hoc the same way we did for sad
# minimalist forms — the model's actual dialect under these arms is not
# known yet.
ANGRY_CALM_TAXONOMY: dict[str, int] = {
    # --- angry pole (+1) ---
    "(ಠ_ಠ)":           +1,   # disapproving stare
    "(ಠ益ಠ)":           +1,   # glaring
    "(╬ಠ益ಠ)":          +1,   # super-glare
    "(ノಠ益ಠ)ノ":         +1,   # throwing arms, angry
    "(ノಠ益ಠ)ノ彡┻━┻":    +1,   # angry table-flip
    "(╯°□°)╯":         +1,   # throwing gesture
    "(╯°□°)╯︵ ┻━┻":    +1,   # classic table-flip
    "(ノ°Д°)ノ︵ ┻━┻":   +1,   # angry table-flip variant
    "(ꐦ°᷄д°᷅)":         +1,   # fury
    "(＃°Д°)":          +1,   # wide-eye fury, fullwidth #
    "(#°Д°)":          +1,   # wide-eye fury, ASCII #
    "(｀ε´)":           +1,   # peeved
    "(╭ರ_•́)":          +1,   # pissed off
    "( `Д´)":          +1,   # furious

    # --- calm pole (-1) ---
    "(´-ω-`)":         -1,   # peaceful
    "( ˘ω˘ )":         -1,   # sleepy-calm
    "(︶ω︶)":           -1,   # content
    "(￣ω￣)":           -1,   # content / placid
    "(´ω`)":           -1,   # peaceful
    "(─‿─)":           -1,   # serene
    "( ˘▽˘)":          -1,   # calm-content
    "(ーωー)":          -1,   # placid
    "(´ー`)":           -1,   # calm
    "(﹏‿﹏)":           -1,   # dreamy-calm
    "(´ ▽`)":          -1,   # soft calm
    "( ˘⌣˘ )":         -1,   # content calm
    "( -_-)":          -1,   # placid deadpan (not clearly angry)
    "(￣ー￣)":          -1,   # cool-calm
    "(⌐■_■)":          -1,   # too-cool-to-care (calm-adjacent)

    # --- observed pilot v2 forms (gemma-4-31b-it, α=0.5) ---
    # angry pole: table-flip remnants (extractor clips at first `)`;
    # full emissions look like ``(╯°°)╯┻╯`` with varying internal chars).
    "(╯°°)":           +1,
    "(╯°)":            +1,
    # calm pole: soft-smile and emoji-bracket forms emitted under
    # calm-steering. The pure-emoji bypass (``🌿``, ``☀️``, ``🚀``, ``🇵🇹``)
    # is tracked separately as the "kaomoji-bypass" phenomenon rather
    # than labeled calm here — see analysis notes.
    "(｡•ᴗ•｡)":         -1,   # calm pouty-content
    "( 🌿 )":           -1,   # leaf-in-brackets (condolence framing)
    "( ☁️ )":           -1,   # cloud-in-brackets
    "( 🫂 )":           -1,   # hug-in-brackets
    "(ᵔᴥᵔ)":           -1,   # teddy-bear calm
}


def label_on(axis: str, form: str) -> int:
    """Return the pole label (+1 / -1 / 0) for `form` on the named axis.

    Unknown axes raise ValueError so typos fail loudly.
    """
    if axis == "happy.sad":
        return TAXONOMY.get(form, 0)
    if axis == "angry.calm":
        return ANGRY_CALM_TAXONOMY.get(form, 0)
    raise ValueError(f"unknown axis {axis!r}")

# Bracket pairs the fallback extractor treats as kaomoji boundaries.
_OPEN_BRACKETS = "([（｛"
_CLOSE_BRACKETS = ")]）｝"

# Leading-glyph filter for kaomoji-bearing assistant turns. Used by the
# Python validators (extract, backfill_journals._kaomoji_prefix) and
# mirrored inline in `~/.claude/hooks/kaomoji-log.sh` /
# `~/.codex/hooks/kaomoji-log.sh`. Centralized here so the shell case
# patterns and Python `frozenset` stay in sync — single source of truth.
KAOMOJI_START_CHARS: frozenset[str] = frozenset("([（｛ヽヾっ٩ᕕ╰╭╮┐┌＼¯໒")


# Maximum length of a real kaomoji we expect to encounter. Real
# kaomoji span ~5–25 characters; the longest in our corpus is
# ``(╯°□°)╯︵ ┻━┻`` at ~12 chars. The cap rejects two-line balanced-
# paren prose accidentally captured by the bracket-span scan.
_KAOMOJI_MAX_LEN = 32

# A run of 4+ consecutive ASCII letters indicates prose, not a kaomoji.
# The hook's `[A-Za-z].*$` cut already strips at the first letter, so
# this is belt-and-suspenders for the gemma extractor path and for
# catching pre-cut garbage in legacy data.
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{4}")


def is_kaomoji_candidate(s: str, *, max_len: int = _KAOMOJI_MAX_LEN) -> bool:
    """Return True iff `s` looks like a real kaomoji prefix.

    Used by ``extract`` and the journal-prefix validators (live-hook
    Python mirror, backfill replay) to reject prose, markdown-escape
    artifacts, and truncated junk that the leading-prefix sed pipeline
    would otherwise let through.

    Rules (all must pass):
      - length 2..``max_len``
      - first char ∈ ``KAOMOJI_START_CHARS``
      - no ASCII backslash (markdown-escape artifact, e.g.
        ``(\\*´∀｀\\*)`` came from Claude emitting a literal ``\\*``
        that the model treated as Markdown escape)
      - no run of 4+ consecutive ASCII letters (prose)
      - if starts with an opening bracket from ``_OPEN_BRACKETS``,
        the span must be bracket-balanced
    """
    if not (2 <= len(s) <= max_len):
        return False
    if s[0] not in KAOMOJI_START_CHARS:
        return False
    if "\\" in s:
        return False
    if _LETTER_RUN_RE.search(s):
        return False
    # Require bracket balance regardless of leading char. Catches both
    # `(unclosed` forms AND `ヽ(^`-style truncations where a non-bracket
    # leader like `ヽ` precedes an unclosed inner `(` — the sed-cut at
    # first ASCII letter can chop these mid-bracket.
    depth = 0
    for c in s:
        if c in _OPEN_BRACKETS:
            depth += 1
        elif c in _CLOSE_BRACKETS:
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    return True


@dataclass(frozen=True)
class KaomojiMatch:
    """Result of running `extract` against a generated text."""
    first_word: str        # the extracted leading kaomoji-like span (or "")
    kaomoji: str | None    # the matched taxonomy entry, or None
    label: int             # +1 / -1 / 0 (other)

    @property
    def pole(self) -> str:
        return POLE_NAMES[self.label]


def _leading_bracket_span(text: str) -> str:
    """Return the leading balanced-paren span of text, or the first
    whitespace-delimited word if text doesn't start with a bracket.

    Handles kaomoji with internal whitespace (the model sometimes emits
    ``(｡˃ ᵕ ˂ )`` — spaces and all) by matching on bracket balance
    rather than splitting on the first space.

    Returns ``""`` when the candidate fails ``is_kaomoji_candidate`` —
    unbalanced brackets, prose, markdown-escape artifacts, oversize
    spans all collapse to the empty string rather than producing
    nonsense first_word values that downstream consumers have to
    re-filter.
    """
    stripped = text.lstrip()
    if not stripped:
        return ""
    candidate = ""
    if stripped[0] in _OPEN_BRACKETS:
        depth = 0
        for i, c in enumerate(stripped):
            if c in _OPEN_BRACKETS:
                depth += 1
            elif c in _CLOSE_BRACKETS:
                depth -= 1
                if depth == 0:
                    candidate = stripped[: i + 1]
                    break
                if depth < 0:
                    break
            if i + 1 >= _KAOMOJI_MAX_LEN:
                # Span ran past the length cap before closing — reject.
                # Without this guard, balanced-paren prose like
                # `(Backgrounddebugscriptcompleted...)` returns the
                # whole sentence as a first_word.
                break
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

    1. Try exact longest-prefix match against TAXONOMY.
    2. Fall back to a validated balanced-paren span as the reported
       first_word, with label=0 (other).

    Returns ``KaomojiMatch(first_word="", kaomoji=None, label=0)`` for
    plain prose / non-kaomoji input — see ``is_kaomoji_candidate`` for
    the rejection rules.
    """
    stripped = text.lstrip()
    ordered = sorted(TAXONOMY.keys(), key=len, reverse=True)
    for k in ordered:
        if stripped.startswith(k):
            return KaomojiMatch(first_word=k, kaomoji=k, label=TAXONOMY[k])
    return KaomojiMatch(
        first_word=_leading_bracket_span(stripped),
        kaomoji=None,
        label=0,
    )


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
# the rules-A-through-F pass (full list of new merge candidates is in
# the "iterate parsing/scraping" thread):
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

# Arm/hand modifiers that appear OUTSIDE the closing paren:
#   (๑˃ᴗ˂)ﻭ  (っ╥﹏╥)っ
_ARM_OUTSIDE = "ﻭっ"
# Arm/hand modifiers that appear just INSIDE the closing paren:
#   (っ˘▽˘ς)  (っ´ω`c)  (*•̀‿•́*)
_ARM_INSIDE_TRAIL = "ςc*"
# Arm/hand modifiers that appear just INSIDE the opening paren (leading):
#   (っ╥﹏╥)  (*•̀‿•́*)
_ARM_INSIDE_LEAD = "っ*"

_TRAIL_OUTSIDE_RE = re.compile(rf"[{_ARM_OUTSIDE}]+$")
_TRAIL_INSIDE_RE = re.compile(rf"[{re.escape(_ARM_INSIDE_TRAIL)}]+\)$")
_LEAD_INSIDE_RE = re.compile(rf"^\([{re.escape(_ARM_INSIDE_LEAD)}]+")

# Rules A + G: invisible / cosmetic-overlay format characters that
# interleave kaomoji glyphs without changing the expression.
#   A: U+200B ZERO WIDTH SPACE, U+200C ZERO WIDTH NON-JOINER,
#      U+200D ZERO WIDTH JOINER, U+2060 WORD JOINER,
#      U+FEFF ZERO WIDTH NO-BREAK SPACE / BOM,
#      U+0602 ARABIC FOOTNOTE MARKER (observed as a stray byte between
#      ``>`` and ``<`` in Qwen ``(๑>؂<๑)``).
#   G: U+0335 COMBINING SHORT STROKE OVERLAY,
#      U+0336 COMBINING LONG STROKE OVERLAY,
#      U+0337 COMBINING SHORT SOLIDUS OVERLAY,
#      U+0338 COMBINING LONG SOLIDUS OVERLAY — strikethrough overlays
#      that occasionally land on eye glyphs (``˃̵`` etc.). Stripped
#      narrowly to U+0335–U+0338; broader stripping of combining marks
#      (U+0300–U+036F) would destroy intentional accent eye glyphs in
#      ``(•̀_•́)`` (U+0300 GRAVE / U+0301 ACUTE).
_INVISIBLE_CHARS_RE = re.compile(
    "[​‌‍⁠﻿؂̴̵̶̷̸̿]"
)

# Hand-picked typographic / glyph substitutions. Hand-picked over NFKC
# because NFKC also compatibility-decomposes `´` (acute) and `˘` (breve)
# into space + combining marks, mangling eye glyphs in `(っ´ω`)` and
# `(˘▽˘)`. NFC leaves those intact; we then apply just the specific
# compatibility-equivalences we want.
# Single-character substitution table. Organized as component
# equivalence classes — each section lists glyphs that play the same
# role (eye / mouth / decoration / punctuation) and fold to a chosen
# canonical member of the class.
#
# Hand-picked over NFKC because NFKC also compatibility-decomposes
# `´` (acute) and `˘` (breve) into space + combining marks, mangling
# eye glyphs in `(っ´ω`)` and `(˘▽˘)`. NFC leaves those intact; we
# then apply just the specific compatibility-equivalences we want.
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
    # fill. Canonical `⊙` chosen as the most-emitted variant in the
    # current corpus (`(⊙_⊙)` n=4, `(⊙ω⊙)` n=1, `(⊙ヮ⊙)` n=1).
    ("◉", "⊙"),   # FISHEYE (Geometric Shapes block) -> CIRCLED DOT
    # Speculative extension (not observed in corpus):
    ("●", "⊙"),   # BLACK CIRCLE (fully solid; included on the
                   # register-level argument that wide-filled-eye
                   # forms are interchangeable; back out if visible
                   # issues arise)
    # === Eye-/decoration-glyph equivalence class: degree-like -> ° (rule E1) ===
    ("º", "°"),   # MASCULINE ORDINAL INDICATOR
    ("˚", "°"),   # RING ABOVE
    # === Middle-dot equivalence class: -> ・ (rule E2 + I) ===
    ("･", "・"),   # HALFWIDTH KATAKANA MIDDLE DOT
    ("•", "・"),   # BULLET (U+2022)
    # === Mouth-glyph equivalence class: smile-curve -> ‿ (rules 3 + M + N) ===
    # All upturned-mouth-curve variants — different stroke widths
    # /shapes for the same role. Pre-existing rule 3 had `ᴗ`; M adds
    # `◡` (LOWER HALF CIRCLE), N adds `ᵕ` (LATIN SMALL LETTER UP TACK).
    ("ᴗ", "‿"),   # LATIN SMALL LETTER OPEN O / connector
    ("◡", "‿"),   # LOWER HALF CIRCLE
    ("ᵕ", "‿"),   # LATIN SMALL LETTER UP TACK
    # Speculative extension (not yet observed in corpus):
    ("⌣", "‿"),   # SMILE (U+2323) — direct synonym for the
                   # smile-mouth role.
    # === Mouth-line distinction (NO fold) ===
    # `﹏` (SMALL WAVY LOW LINE U+FE4F) and `_` (ASCII UNDERSCORE) are
    # NOT interchangeable. `﹏` is wavy/distressed (`(>﹏<)`,
    # `(╥﹏╥)` — crying/distressed register); `_` is flat/neutral
    # (`(•_•)`, `(◕_◕)` — blank-stare register). The previous
    # `﹏` → `_` fold collapsed those affects together.
    # === Bracket-corner-decoration equivalence class: -> ｡ (rule J + B-extension) ===
    ("◍", "｡"),   # CIRCLE WITH VERTICAL FILL (U+25CD)
    ("。", "｡"),   # IDEOGRAPHIC FULL STOP (full-size CJK period;
                   # halfwidth `｡` chosen as canonical to match J)
)

# Rule K: substring-level substitutions applied AFTER `_TYPO_SUBS` so
# that `•` → `・` has already happened, and AFTER internal-whitespace
# stripping. Targeted to avoid global `-` ↔ `_` folds that would
# corrupt `(´-ω-`)` (where `-` is a tired-eye glyph).
#
# The earlier targeted mirror-pair rule `(◑‿◐)` → `(◐‿◑)` is no
# longer needed — circular-fill eye fold in `_TYPO_SUBS` collapses
# both to `(◕‿◕)` directly.
_INTERNAL_SUBS: tuple[tuple[str, str], ...] = (
    # Middle-dot eyes with hyphen mouth -> middle-dot eyes with
    # underscore mouth. Targeted: `(・-・)` ↔ `(・_・)`.
    ("・-・", "・_・"),
)


def _cyrillic_lower(s: str) -> str:
    """Rule D: lowercase Cyrillic capitals U+0410–U+042F.

    Leaves all non-Cyrillic-capital characters untouched, including
    other Unicode case-bearing letters (Greek, etc.) which haven't been
    observed as cosmetic-only variants in this corpus.
    """
    return "".join(
        c.lower() if 0x0410 <= ord(c) <= 0x042F else c
        for c in s
    )


def canonicalize_kaomoji(s: str) -> str:
    """Collapse trivial kaomoji variants to a canonical form.

    Applies, in order:
      1. NFC normalization (preserves `´`, `˘`, `｡` which NFKC would mangle).
      2. Strip invisible / cosmetic-overlay chars (rule A + G — U+200B/C/D,
         U+2060, U+FEFF, U+0602, U+0335–U+0338).
      3. Apply ``_TYPO_SUBS`` (rule B half/full-width + E1 degree + E2
         middle-dot + H curly-quote + I bullet→middle-dot + J
         bracket-corner-circle, plus existing arm/paren folds).
      4. Strip ASCII spaces inside the `(...)` bracket span (rule C).
      5. Lowercase Cyrillic capitals (rule D).
      6. Apply ``_INTERNAL_SUBS`` substring substitutions (rule K
         ``・-・`` → ``・_・``).
      7. Strip arm modifiers from face boundaries (rule F + L —
         ``っ ς c ﻭ *``).

    Eye/mouth/decoration changes that aren't covered by rules E1/E2/I/J
    are preserved.

    Idempotent: ``canonicalize_kaomoji(canonicalize_kaomoji(s)) == canonicalize_kaomoji(s)``.

    Empty input returns ``""``.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s.strip())
    s = _INVISIBLE_CHARS_RE.sub("", s)
    for src, dst in _TYPO_SUBS:
        s = s.replace(src, dst)
    if s.startswith("(") and s.endswith(")"):
        s = "(" + s[1:-1].replace(" ", "") + ")"
    s = _cyrillic_lower(s)
    for src, dst in _INTERNAL_SUBS:
        s = s.replace(src, dst)
    # Strip outside-paren trailing arm chars first so trailing-inside
    # detection sees the closing paren.
    s = _TRAIL_OUTSIDE_RE.sub("", s)
    s = _LEAD_INSIDE_RE.sub("(", s)
    s = _TRAIL_INSIDE_RE.sub(")", s)
    return s


def sanity_check() -> None:
    """Smoke-test the extractor and canonicalizer."""
    # --- extract() ---
    # registered kaomoji
    assert extract("(｡◕‿◕｡) I had a great day!").label == +1
    assert extract("(｡•́︿•̀｡) That's so sad.").label == -1
    assert extract("  (✿◠‿◠) hi").label == +1
    # plain text — non-kaomoji prose returns empty first_word
    assert extract("hello!").first_word == ""
    # whitespace-padded taxonomy entry still matches exactly
    m = extract("(｡˃ ᵕ ˂ ) That is wonderful!")
    assert m.label == +1
    # bracket-span fallback for an unknown paren form (real kaomoji-shape)
    m = extract("(｡o_O｡) strange")
    assert m.label == 0
    assert m.first_word == "(｡o_O｡)", repr(m.first_word)
    # empty
    assert extract("").label == 0
    # --- new robustness: garbage rejection ---
    # parenthesized prose with 4+-letter run → rejected
    assert extract("(Backgrounddebugscript) trailing").first_word == ""
    # bracketed phrase with internal letters → rejected
    assert extract("[pre-commit] passed").first_word == ""
    # markdown-escape backslash → rejected
    assert extract("(\\*´∀｀\\*) hello").first_word == ""
    # unbalanced bracket → rejected (no fall-through to whitespace split)
    assert extract("(｡• ω •｡  open paren never closed").first_word == ""
    # oversize balanced span → rejected
    long_paren = "(" + "a" * 50 + ")"
    assert extract(long_paren + " text").first_word == ""

    # --- canonicalize_kaomoji ---
    ck = canonicalize_kaomoji
    # idempotence on the empty / whitespace inputs
    assert ck("") == ""
    assert ck("   ") == ""
    # rule A: strip word-joiner / ZWSP / Arabic footnote marker
    assert ck("(⁠◕⁠‿⁠◕⁠✿⁠)") == "(◕‿◕✿)"
    assert ck("(๑>؂<๑)") == "(๑><๑)"
    # rule B: half/full-width punctuation
    assert ck("(＞_＜)") == "(>_<)"
    assert ck("(；ω；)") == "(;ω;)"
    # rule C: strip internal ASCII whitespace inside brackets
    assert ck("( ; ω ; )") == "(;ω;)"
    assert ck("( ;´Д｀)") == "(;´д`)"  # rule O folds ｀ -> `
    # rule D: Cyrillic case fold
    assert ck("(；´Д｀)") == "(;´д`)"  # rule O folds ｀ -> `
    assert ck("(；´д｀)") == "(;´д`)"  # rule O folds ｀ -> `
    # rule E1: degree-like glyphs
    assert ck("(°Д°)") == "(°д°)"
    assert ck("(ºДº)") == "(°д°)"
    assert ck("(˚Д˚)") == "(°д°)"
    # rule E2: middle-dot fold
    assert ck("(´・ω・`)") == "(´・ω・`)"
    assert ck("(´･ω･`)") == "(´・ω・`)"
    # rule F (existing): arm modifiers
    assert ck("(๑˃ᴗ˂)ﻭ") == "(๑˃‿˂)"
    assert ck("(っ╥﹏╥)っ") == "(╥﹏╥)"  # ﹏ stays distinct from _
    # rule G: combining strikethrough overlays
    assert ck("(๑˃̵‿˂̵)") == "(๑˃‿˂)"
    # rule H: curly quotes -> ASCII; fullwidth tilde `～` ALSO folds to
    # `~` under the speculative B extension added 2026-04-27 (corpus
    # had the mixed-internally-inconsistent `(~～~;)` form).
    assert ck("┐(‘～`;)┌") == "┐('~`;)┌"
    assert ck("┐('～`;)┌") == "┐('~`;)┌"
    # rule I: bullet -> middle-dot
    assert ck("(´•ω•`)") == "(´・ω・`)"
    assert ck("(´・ω・`)") == "(´・ω・`)"
    # rule J: bracket-corner circle -> bracket-corner dot
    assert ck("(◍•‿•◍)") == "(｡・‿・｡)"  # also picks up rule I
    assert ck("(｡•‿•｡)") == "(｡・‿・｡)"  # also picks up rule I
    # rule K: targeted ・-・ -> ・_・
    assert ck("(・-・)") == "(・_・)"
    assert ck("(・_・)") == "(・_・)"
    # rule K does NOT corrupt eye-`-` glyphs in (X-Y-X) form
    assert ck("(´-ω-`)") == "(´-ω-`)"
    # rule L: `*` arm modifier
    assert ck("(*•̀‿•́*)") == "(・̀‿・́)"  # I fires too: • -> ・
    # rule M: smile-curve fold ◡ -> ‿ (with eye-class fold ◔ -> ◕)
    assert ck("(◔◡◔)") == "(◕‿◕)"
    assert ck("(ᵔ◡ᵔ)") == "(ᵔ‿ᵔ)"
    # rule N: smaller-mouth tack fold ᵕ -> ‿
    assert ck("(´｡・ᵕ・｡`)") == "(´｡・‿・｡`)"
    # rule O: fullwidth grave accent -> ASCII grave
    assert ck("ヽ(´ー｀)ノ") == "ヽ(´ー`)ノ"
    assert ck("ヽ(´ー`)ノ") == "ヽ(´ー`)ノ"
    # rule B extension: ideographic full stop -> halfwidth ideographic
    # full stop (matches J's canonical `｡`)
    assert ck("(´。・ᵕ・。`)") == "(´｡・‿・｡`)"  # N fires too
    # eye class: directional-fill circular eyes -> ◕
    assert ck("(◔‿◔)") == "(◕‿◕)"
    assert ck("(◑‿◐)") == "(◕‿◕)"   # mirror pair, both fold to ◕
    assert ck("(◐‿◑)") == "(◕‿◕)"
    assert ck("(◕‿◕)") == "(◕‿◕)"   # canonical, idempotent
    # speculative directional-fill extensions
    assert ck("(◒_◒)") == "(◕_◕)"
    assert ck("(◓‿◓)") == "(◕‿◕)"
    assert ck("(◖_◗)") == "(◕_◕)"
    # eye class: filled-with-pupil eyes -> ⊙ (separate from
    # directional-fill class — distinct visual register)
    assert ck("(◉_◉)") == "(⊙_⊙)"
    assert ck("(⊙_⊙)") == "(⊙_⊙)"   # canonical, idempotent
    assert ck("(●_●)") == "(⊙_⊙)"   # speculative
    # speculative mouth extension
    assert ck("(◕⌣◕)") == "(◕‿◕)"   # SMILE -> ‿
    # speculative B extensions
    assert ck("(・_・？)") == "(・_・?)"
    assert ck("(～ω～)") == "(~ω~)"
    # speculative G extensions: U+0334 + U+033F overlay strikethroughs
    assert ck("(๑˃̴‿˂̿)") == "(๑˃‿˂)"
    # eye-class fold preserves shapes outside the class
    assert ck("(◠‿◠)") == "(◠‿◠)"   # ◠ (UPPER HALF CIRCLE) is a
                                       # different eye glyph, not folded
    # idempotence on a complex example
    once = ck("( ⁠;⁠ ´⁠Д⁠｀⁠ )")
    twice = ck(once)
    assert once == twice, (once, twice)
    # eye change preserved (NOT collapsed by E)
    assert ck("(◕‿◕)") != ck("(♥‿♥)")
    # is_kaomoji_candidate
    assert is_kaomoji_candidate("(｡◕‿◕｡)")
    assert not is_kaomoji_candidate("hi")
    assert not is_kaomoji_candidate("(\\*´∀｀\\*)")
    assert not is_kaomoji_candidate("(Backgrounddebug)")
    assert not is_kaomoji_candidate("(unclosed")
    assert not is_kaomoji_candidate("(" + "a" * 100 + ")")


if __name__ == "__main__":
    sanity_check()
    happy = sum(1 for v in TAXONOMY.values() if v > 0)
    sad = sum(1 for v in TAXONOMY.values() if v < 0)
    print(f"taxonomy OK; {len(TAXONOMY)} kaomoji registered ({happy}+/{sad}-)")
