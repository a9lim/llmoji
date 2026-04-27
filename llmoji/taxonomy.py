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
  - The fallback is a balanced-paren span, so whitespace-padded
    kaomoji like ``(｡˃ ᵕ ˂ )`` surface with a human-readable
    `first_word` even when no exact taxonomy entry exists.
  - For research-side label lookups, see
    ``llmoji_study.taxonomy_labels.extract_with_label``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Bracket pairs the fallback extractor treats as kaomoji boundaries.
_OPEN_BRACKETS = "([（｛"
_CLOSE_BRACKETS = ")]）｝"

# Leading-glyph filter for kaomoji-bearing assistant turns. Used by
# `extract`, by `is_kaomoji_candidate`, and by every shell hook
# template under `llmoji._hooks/` (rendered into the bash `case`
# pattern via `llmoji.providers.base.render_kaomoji_start_chars_case`).
# Single source of truth; previous versions duplicated this set in
# five places, which is the gotcha the v1.0 split resolved.
KAOMOJI_START_CHARS: frozenset[str] = frozenset("([（｛ヽヾっ٩ᕕ╰╭╮┐┌＼¯໒")


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


def is_kaomoji_candidate(s: str, *, max_len: int = _KAOMOJI_MAX_LEN) -> bool:
    """Return True iff `s` looks like a real kaomoji prefix.

    Used by `extract` and the journal-prefix validators (live-hook
    Python mirror, backfill replay) to reject prose, markdown-escape
    artifacts, and truncated junk that the leading-prefix sed
    pipeline would otherwise let through.

    Rules (all must pass):
      - length 2..`max_len`
      - first char ∈ `KAOMOJI_START_CHARS`
      - no ASCII backslash (markdown-escape artifact, e.g.
        ``(\\*´∀｀\\*)`` came from a model emitting a literal ``\\*``
        that it treated as Markdown escape)
      - no run of 4+ consecutive ASCII letters (prose)
      - if starts with an opening bracket from `_OPEN_BRACKETS`,
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
    # Require bracket balance regardless of leading char. Catches
    # `(unclosed` AND `ヽ(^`-style truncations where a non-bracket
    # leader like `ヽ` precedes an unclosed inner `(` — the sed-cut
    # at first ASCII letter can chop these mid-bracket.
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
    """Return the leading balanced-paren span of `text`, or the
    first whitespace-delimited word if `text` doesn't start with a
    bracket.

    Handles kaomoji with internal whitespace (the model sometimes
    emits ``(｡˃ ᵕ ˂ )`` — spaces and all) by matching on bracket
    balance rather than splitting on the first space.

    Returns `""` when the candidate fails `is_kaomoji_candidate` —
    unbalanced brackets, prose, markdown-escape artifacts, oversize
    spans all collapse to the empty string rather than producing
    nonsense `first_word` values that downstream consumers would
    have to re-filter.
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
                # Span ran past the length cap before closing —
                # reject. Without this guard, balanced-paren prose
                # like `(Backgrounddebugscriptcompleted...)` returns
                # the whole sentence as a `first_word`.
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
    # Strip outside-paren trailing arm chars first so trailing-inside
    # detection sees the closing paren.
    s = _TRAIL_OUTSIDE_RE.sub("", s)
    s = _LEAD_INSIDE_RE.sub("(", s)
    s = _TRAIL_INSIDE_RE.sub(")", s)
    return s


