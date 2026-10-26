"""Kaomoji extraction, validation and canonicalization.

`split_leading_kaomoji` and `extract` find the face that opens a message,
`is_kaomoji_candidate` decides whether a span is a face, and
`canonicalize_kaomoji` folds cosmetic variants of one face to a single form.

`KAOMOJI_START_CHARS`, the validator and the canonicalization rules are
cross-corpus invariants: the shared dataset groups faces by them, so changing
any of them changes what counts as the same face. Affect labels for faces
belong to the research-side llmoji-study repo, not here.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass

# Bracket pairs the depth walkers use to find where a face closes. Beyond the
# ASCII and fullwidth pairs they include the bear-face brackets `ʕʔ` and `ʢʡ`,
# the Japanese corner brackets `「」『』【】〈〉《》` and the tortoise-shell pair
# `〔〕`, so `ʕ•ᴥ•ʔ` and `「(゜～゜)」` close at their own end instead of
# falling through to the whitespace fallback.
_OPEN_BRACKETS = "([（｛ʕʢ「『【〈《〔"
_CLOSE_BRACKETS = ")]）｝ʔʡ」』】〉》〕"

# Characters a leader-path kaomoji can start with: opening brackets, arm and
# pose leaders, and glyphs that decorate a face (hearts, stars, notes,
# flowers). The decorators and arms also appear in `_ARM_OUTSIDE_LEAD` and
# `_ARM_OUTSIDE`, which strip them back to the bare face. Used by
# `is_kaomoji_candidate`.
#
# The corner brackets `「『【〈《` are leaders, so a leading Japanese quoted
# phrase (`「これは良い」...`) forms a balanced span that passes the validator.
# The false positive is accepted; it is close to nonexistent in coding-agent
# traffic.
#
# Left out on purpose: the ASCII letters `o O q b d m p t`, which begin short
# English words ("ok", "of", "my", "to") the validator would let through; `~`,
# common in prose; and `*`, the Markdown emphasis delimiter. Bare faces such
# as `o_o` and `*_*` still pass through the bare-face path.
KAOMOJI_START_CHARS: frozenset[str] = frozenset(
    "([（｛ヽヾっ٩ᕕ╰╭╮┐┌＼¯໒\\⊂✧"  # brackets, hand and pose leaders; wing `\`, hug `⊂`, sparkle `✧`
    "Σψεƪʕ"                       # shocked sigma, horn-fingers, kiss, raised hands, bear
    "Ψ"                            # capital psi, matching ψ
    "╱╲"                           # box-drawing diagonals
    "「『【〈《"                    # Japanese corner brackets
    "└┘"                           # box-drawing standing pose
    "♪♫♬"                          # music notes
    "♥♡❤"                          # hearts: filled, outline, heavy
    "★☆"                           # stars: filled, outline
    "ʢ"                            # alternate bear bracket
    "✿❀"                           # florettes: black, white
    "❣❥"                           # heart ornaments: exclamation, rotated
    "✦✩✪"                          # stars: four-point, outlined, circled
    "♩"                            # quarter note
    "ᕦᕙ"                          # flex and strong-feel pose arms
    "〔"                           # tortoise-shell bracket
    "※"                            # reference mark
    "୧"                            # Oriya cradle pose arm
)


# Real kaomoji run about 5-25 characters. The cap also rejects the two-line
# balanced-paren prose a bracket scan can capture.
_KAOMOJI_MAX_LEN = 32

# A run of 4+ ASCII letters is prose, not a face.
_LETTER_RUN_RE = re.compile(r"[A-Za-z]{4}")


# Interior glyphs of a bare `EYE MOUTH EYE` face: ASCII connectors and
# punctuation, dashes, CJK punctuation and presentation forms, geometric
# shapes, and a few letters. The letters (`oOwW`, `ω`) give `^o^`, `OwO` and
# `>ω<`; they can't trip the 4-letter-run prose filter, since a bare face
# never has four letters in a row. `〜` is here as well as `~` because
# validation sees the raw text, before canonicalization folds `〜` to `~`.
_BARE_KAOMOJI_MOUTH_RE = re.compile(
    r"["
    r"_\-.;:~=^|/\\"           # ASCII mouth glyphs
    r"oOwW"                    # letter mouths: `^o^`, `OwO`
    r"·•°"                     # middle dot, bullet, degree sign
    r"ω"                       # Greek omega, the cat mouth
    r"ヮ"                      # katakana small wa: `^ヮ^`
    r"〜"                      # wave dash
    r"‐-―"                     # various dashes
    r"‥…′-‷"                   # ellipsis variants, primes
    r"‿⁀"                      # undertie, character tie
    r"、。"                    # CJK comma/fullstop
    r"　・ー･＿？"             # CJK spaces / mid-dots / fullwidth forms
    r"︰-﹏"                    # CJK presentation forms (︵ ︶ ﹏ ﹋ ﹌ ︿ ︶ etc.)
    r"▰-◿"                     # geometric shapes (▽ ◡ ◠ ○ ● ◇ etc.)
    r"]+"
)

# Eye pairs for asymmetric bare faces such as `>_<` and `)_(`. They are
# eyes only; unlike the bracket sets they don't drive a depth walk.
_BARE_KAOMOJI_PAIRED_EYES: frozenset[tuple[str, str]] = frozenset({
    (">", "<"), ("<", ">"),
    (")", "("), ("(", ")"),
    ("]", "["), ("[", "]"),
    ("}", "{"), ("{", "}"),
    ("\\", "/"), ("/", "\\"),  # celebration (`\o/`), facepalm (`\_/`)
    ("o", "O"), ("O", "o"),    # `o_O` / `O_o`, the "huh?" face
})

# Western emoticon parts. `<` and `>` are eyes so that `<3` parses and an
# eyebrow can precede a face.
_WESTERN_EYES = ":;=8<>"
_WESTERN_MOUTHS = ")(DPpOo3<>/\\|*[]"
_WESTERN_NOSES = "-^o'"


def _is_western_emoticon(s: str) -> bool:
    """Whether `s` is a Western emoticon: an eye from `_WESTERN_EYES`, an
    optional nose, then one or more mouth characters, 2 to 4 in all.

    Mouths made only of the eye character (`>>`, `<<`, `==`) are rejected;
    they match structurally because `>`, `<` and `=` are both eyes and mouths.
    """
    n = len(s)
    if not (2 <= n <= 4):
        return False
    if s[0] not in _WESTERN_EYES:
        return False
    rest = s[1:]
    if rest and rest[0] in _WESTERN_NOSES:
        rest = rest[1:]
    if not rest or not all(c in _WESTERN_MOUTHS for c in rest):
        return False
    if all(c == s[0] for c in rest):
        return False
    return True


# All-letter faces (`OwO`, `UwU`, ...) that the all-letter guard in
# `_looks_like_bare_kaomoji` would reject. Only `w` and `v` mouths qualify;
# other letter triples (`lol`, `mom`) are prose.
_UWU_FACES: frozenset[str] = frozenset({
    "OwO", "OWO", "owo",
    "UwU", "UWU", "uwu",
    "OvO", "OVO", "ovo",
    "UvU", "UVU", "uvu",
})


# Crying closed-eye doubles; a literal set is cheaper than a regex.
_KOREAN_CLOSED_EYE_DOUBLES = frozenset({"ㅠㅠ", "ㅜㅜ"})


def _looks_like_bare_kaomoji(s: str) -> bool:
    """Whether `s` has the shape of a face that lacks a leader glyph.

    Shapes:
      - symmetric `EYE MOUTH EYE`: `^_^`, `T-T`, `;_;`, `@_@`, `ಥ_ಥ`, `T﹏T`,
        `OωO`, `^w^`, `\\o/`
      - paired eyes: `>_<`, `>.<`, `)_(`, `\\_/`
      - Western emoticons: `:)`, `:D`, `;-)`, `8)`, `<3`
      - eyebrowed Western: `>:(`, `>:D`
      - cat wrap: `=^.^=`, `=ω.ω=`
      - closed-eye doubles: `^^`, `vv`, `uu`, `ㅠㅠ`, `ㅜㅜ`
      - laughs: `XD`, `xP`, `X3`

    The caller applies the length, backslash and letter-run filters. The
    match is permissive on purpose, since the aim is to surface faces from
    models that drop the wrapping parens; stray prose that passes becomes a
    low-count face downstream.
    """
    n = len(s)
    if n < 2:
        return False

    # Two characters: closed-eye doubles, laughs, Western pairs.
    if n == 2:
        if s[0] == s[1] and s[0] in "^vu":
            return True
        if s in _KOREAN_CLOSED_EYE_DOUBLES:
            return True
        # Laughs: `XD`, `xP`, `X3`.
        if s[0] in "xX" and s[1] in "DPpOo3":
            return True
        # Western pair. Same-char pairs (`>>`, `<<`, `==`) are not faces,
        # though they match structurally.
        if s[0] in _WESTERN_EYES and s[1] in _WESTERN_MOUTHS and s[0] != s[1]:
            return True
        return False

    # Cat wrap: `=` on both ends of a bare face. Each level drops two
    # characters, so the recursion ends well within the length cap.
    if n >= 5 and s[0] == "=" and s[-1] == "=":
        if _looks_like_bare_kaomoji(s[1:-1]):
            return True

    # Standard Western (3-4 chars).
    if _is_western_emoticon(s):
        return True

    # Eyebrowed Western: `>` or `<` before an emoticon (`>:(`, `<:D`).
    if n >= 3 and s[0] in "><" and _is_western_emoticon(s[1:]):
        return True

    # Letter-only anime faces (`OwO`, `UwU`), which the all-letter guard
    # below would reject.
    if s in _UWU_FACES:
        return True

    # Symmetric or paired `EYE MOUTH EYE`: three or more characters, the
    # interior all mouth glyphs. The first character must not occur in the
    # interior, which rejects runs like `___` and `...` but lets `^_^` and
    # `|_|` through even though their eyes are also mouth glyphs.
    interior = s[1:-1]
    if not interior:
        return False
    if not _BARE_KAOMOJI_MOUTH_RE.fullmatch(interior):
        return False
    first, last = s[0], s[-1]
    if first in interior:
        return False
    # Three-letter prose palindromes (`lol`, `mom`, `pop`, `awa`) fit the
    # shape but are not faces. Mixed-script faces (`OωO`, `ಥ﹏ಥ`) and anything
    # with a non-letter (`T-T`) pass; the all-letter faces worth keeping are
    # in `_UWU_FACES`.
    if s.isascii() and s.isalpha():
        return False
    if first == last:
        return True
    if (first, last) in _BARE_KAOMOJI_PAIRED_EYES:
        return True
    return False


def is_kaomoji_candidate(s: str, *, max_len: int = _KAOMOJI_MAX_LEN) -> bool:
    """Whether `s` looks like a kaomoji prefix rather than prose.

    All of these must hold:
      - length 2..`max_len`
      - no backslash after position 0. A leading backslash is the wing-hand
        pattern (`\\(^o^)/`); a later one is a Markdown escape that leaked
        into the text, as in `(\\*´∀｀\\*)`.
      - no run of 4+ ASCII letters
      - the leader path (first character in `KAOMOJI_START_CHARS` and some
        content, see `_has_kaomoji_content`) or the bare-face path
        (`_looks_like_bare_kaomoji`)

    Bracket balance is not enforced: corpus output is sometimes unbalanced,
    with a closing glyph that isn't the matching bracket. The length cap,
    letter-run rule and backslash rule carry the prose rejection instead.
    """
    if not (2 <= len(s) <= max_len):
        return False
    if "\\" in s[1:]:
        return False
    if _LETTER_RUN_RE.search(s):
        return False
    # Leader path. The content check is only here: the bare-face path has
    # structural rules that already exclude letter-only spans.
    if s[0] in KAOMOJI_START_CHARS and _has_kaomoji_content(s):
        return True
    if _looks_like_bare_kaomoji(s):
        return True
    return False


def _has_kaomoji_content(s: str) -> bool:
    """Whether `s` has a character other than ASCII letters and brackets.

    Rejects leader-path spans like `[a]`, `(b)` and `(test)`, which have the
    shape of a face but no face glyph. ASCII symbols and digits count as
    content (`(0_0)`), as does any non-ASCII character.
    """
    for c in s:
        if not c.isascii():
            return True
        if c.isalpha():
            continue
        if c in _OPEN_BRACKETS or c in _CLOSE_BRACKETS:
            continue
        return True
    return False


@dataclass(frozen=True)
class KaomojiMatch:
    """The leading kaomoji span found by `extract`, or "" if there is none."""
    first_word: str


def _leading_bracket_span(text: str) -> str:
    """The leading kaomoji span of `text`, or "".

    A bracket-leading face is cut at its balanced close, so whitespace-padded
    faces like `(｡˃ ᵕ ˂ )` come out whole. If the walk reaches the length cap
    or goes negative without closing (corpus output is sometimes
    unbalanced), the first whitespace-delimited word is used instead. Other
    leaders (`ヽ`, `ᕕ`, ...) take the first word. A candidate that fails
    `is_kaomoji_candidate` yields "" rather than a nonsense `first_word`.
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
            # Unbalanced: fall back to the first whitespace-delimited word.
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
    """Find the leading kaomoji in `text`; `first_word` is "" when there is none."""
    return KaomojiMatch(first_word=_leading_bracket_span(text.lstrip()))


# Characters that open a message body rather than continue a face when
# they follow it after whitespace: markdown, list numbering, quotes.
_BODY_PUNCT = frozenset(string.punctuation + "—–…“”‘’«»")


def _is_trailing_decoration(token: str) -> bool:
    """Whether a whitespace-separated token after a face still belongs
    to it, as ``┻━┻`` does in ``(╯°□°)╯︵ ┻━┻`` and ``・ﾟ`` in
    ``(ﾉ´ヮ`)ﾉ*: ・ﾟ``. Markdown (``**``, ``—``, ``1.``), ASCII words
    and any run of two letters in a row (Japanese or Cyrillic prose)
    start the body instead."""
    if all(c in _BODY_PUNCT or c.isdigit() for c in token):
        return False
    run = 0
    for c in token:
        if c.isascii() and c.isalnum():
            return False
        run = run + 1 if unicodedata.category(c).startswith("L") else 0
        if run >= 2:
            return False
    return True


def _leading_face(line: str) -> str:
    """The kaomoji opening ``line``, or ``""``.

    Tries the glyph face first: everything up to the first whitespace or
    ASCII letter outside brackets, so ``(ง •̀_•́)ง`` keeps its inner
    spaces and ``(o_O)`` its letters. When that isn't a valid candidate
    (an unclosed bracket, a bare ``T-T`` or ``XD``), the first
    whitespace-delimited word is tried instead. Trailing decoration
    tokens are then added while the face stays under the length cap.
    """
    depth = end = 0
    for i, c in enumerate(line[:_KAOMOJI_MAX_LEN]):
        if c in _OPEN_BRACKETS:
            depth += 1
        elif c in _CLOSE_BRACKETS:
            depth = max(depth - 1, 0)
        elif depth == 0 and (c.isspace() or (c.isascii() and c.isalpha())):
            break
        end = i + 1
    face = line[:end].rstrip()
    if not is_kaomoji_candidate(face):
        face = line.split(maxsplit=1)[0] if line.strip() else ""
        if not is_kaomoji_candidate(face):
            return ""
    rest = line[len(face):]
    for token in rest.split():
        if not _is_trailing_decoration(token):
            break
        extended = line[: line.index(token, len(face)) + len(token)]
        if len(extended) > _KAOMOJI_MAX_LEN:
            break
        face = extended
    return face


def split_leading_kaomoji(text: str) -> tuple[str, str] | None:
    """Split a message into ``(face, body)`` when it opens with a
    kaomoji, else ``None``.

    The one splitter every journal row is written through — live hooks,
    ``llmoji import`` and the export parsers alike. The face never spans
    lines; ``body`` is the rest of the message with the face and the
    whitespace around it removed, so nothing but whitespace is lost.
    """
    stripped = text.lstrip()
    line, sep, rest = stripped.partition("\n")
    face = _leading_face(line)
    if not face:
        return None
    return face, (line[len(face):] + sep + rest).strip()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
#
# Faces that differ only cosmetically collapse to one form; a difference in
# eyes, mouth or decoration that no rule names stays a distinct face. The
# rule letters are the ones tests/test_canonicalize.py refers to:
#
#   A   invisible format characters are deleted
#   B   halfwidth and fullwidth punctuation folds to ASCII
#   C   ASCII spaces inside the parens go: `( ; ω ; )` is `(；ω；)`;
#       non-ASCII spacing characters are part of the face
#   D   Cyrillic capitals are lowercased: `Д` and `д` alternate in the same
#       `(；´Д｀)` skeleton, so the model isn't choosing between them
#   E1  `°` `º` `˚` fold to `°`: `(°Д°)` and `(ºДº)` are one shocked face
#   E2  `・` and `･` fold to `・`
#   F   arm and hand glyphs at the face boundary are stripped, so `(๑˃ᴗ˂)ﻭ`
#       and `(っ˘▽˘ς)` are the face without its arm
#   G   combining overlays on eye glyphs are deleted
#   H   curly quotes fold to straight quotes
#   I   bullet `•` folds to `・`
#   J   `◍` folds to `｡`: `(◍•‿•◍)` and `(｡•‿•｡)` share a skeleton and the
#       corner decoration plays the same role. The most aggressive fold
#   K   `・-・` folds to `・_・`
#   L   `*` at the face boundary is an arm like `っ`: `(*•‿•*)` becomes
#       `(・‿・)` (with I)
#   M   circles filled in some direction fold to `◕`
#   N   smile-curve mouths fold to `‿`
#   O   fullwidth grave accent folds to ASCII
#
# Order: NFC first. NFKC would split `´` and `˘` into a space plus a
# combining mark and wreck eye glyphs in `(っ´ω`)` and `(˘▽˘)`, so the
# width and glyph folds are hand-picked in `_TYPO_SUBS` instead. Then the
# translate pass (A, B, D, E1, E2, G, H, I, J, M, N, O), whitespace (C),
# `_INTERNAL_SUBS` (K), and the arms (F, L).

# Rule F, outside the parens: arm, hand and decoration glyphs around a face
# strip to the bare face. `_ARM_OUTSIDE` holds the trailing halves and
# `_ARM_OUTSIDE_LEAD` the leading ones. Patterns covered:
#   \(^o^)/  ⊂(˘ω˘)⊂  ✧(ˊᗜˋ)✧           wing, hug, sparkle
#   ψ(`Д´)ψ  ε(◕‿◕)з  ƪ(˘⌣˘)ʃ           horn-fingers, kiss, raised hands
#   ヽ(´ー`)ノ  ٩(◕‿◕)۶  ᕕ(ᐛ)ᕗ           raised hands, cheering, running
#   ໒(◕‿◕)७  ୧(˃ᗨ˂)୨  ᕦ(ò_óˇ)ᕤ          cradle, flex
#   ╲(◕‿◕)╱  ╰(´∀`)╯  ┐(´д`)┌  └(°▽°)┘  box-drawing wings and poses
#   (๑˃ᴗ˂)ﻭ  (っ╥﹏╥)っ  (´∀`)/            one-sided arms
#   ¯\_(ツ)_/¯                           shrug
#   「(゜～゜)」  〔(◕‿◕)〕                  corner and tortoise-shell brackets
#   ♪(´▽｀)♪  ♥(◕‿◕)♥  ★(◕‿◕)★  ✿(◕‿◕)✿  ※(◕‿◕)※   decorators
# Box-drawing glyphs, `¯` and `_` appear in both sets because poses mirror
# (`╮(´д`)╭`). Decorators strip only outside the parens: `(♥‿♥)`, with
# hearts as eyes, is its own face.
_ARM_OUTSIDE = (
    "ﻭっ/⊂⊃✧۶ᕗ७ψΨзʃ╱ノﾉ╯╮╭┌┐┘└¯_"
    "」』】〉》"   # corner-bracket closes
    "♪♫♬"        # music notes
    "♥♡❤"        # hearts
    "★☆"         # stars
    "ᕤ୨"         # flex and Oriya cradle arms
    "〕"         # tortoise-shell close
    "✿❀"         # flowers
    "❣❥"         # heart variants
    "✦✩✪"        # star variants
    "♩"          # quarter note
    "※"          # reference mark
    "つ"         # offering hands: `(face)つ`, `(つface)つ`
    "づ"         # voiced offering hands, as `つ`
)

# Rule F, leading halves, mirroring `_ARM_OUTSIDE`, plus `Σ`, which has no
# closing half (`Σ(°△°|||)`). The bear brackets `ʕ` and `ʢ` are left out on
# purpose: the bear is the whole face, so even a paren-wrapped variant like
# `ʢ(◉ᴥ◉)ʡ` keeps its brackets. Distinct from `_ARM_INSIDE_LEAD`, which sits
# between `(` and the face.
_ARM_OUTSIDE_LEAD = (
    "\\⊂✧ΣψΨεƪ╲٩ᕕ໒ヽヾ╰╭╮┐┌¯_"
    "「『【〈《"   # corner-bracket opens
    "└┘"          # box-drawing standing pose
    "♪♫♬"         # music notes
    "♥♡❤"         # hearts
    "★☆"          # stars
    "ᕦᕙ୧"        # flex, strong-feel and Oriya cradle arms
    "〔"          # tortoise-shell open
    "✿❀"          # flowers
    "❣❥"          # heart variants
    "✦✩✪"         # star variants
    "♩"           # quarter note
    "※"           # reference mark
)

# Rules F and L, inside the closing paren: `(っ˘▽˘ς)`, `(っ´ω`c)`, `(*•‿•*)`.
_ARM_INSIDE_TRAIL = "ςc*"
# Inside the opening paren: `(っ╥﹏╥)`, `(*•‿•*)`, `(つ◕‿◕)つ`, `(づ◕‿◕)づ`.
# `つ` and `づ` are the offering-hands gesture; the other half sits outside
# the close paren, in `_ARM_OUTSIDE`.
_ARM_INSIDE_LEAD = "っ*つづ"

# Outside arms are stripped only next to a paren: leading ones right before
# `(`, trailing ones right after `)`. Eye and mouth glyphs inside a face
# (`_` in `(◕_◕)`) are never touched, and a face wrapped in corner brackets
# with no inner paren (`「・_・」`) keeps its wrapper.
_TRAIL_OUTSIDE_RE = re.compile(rf"(?<=\))[{re.escape(_ARM_OUTSIDE)}]+$")
_LEAD_OUTSIDE_RE = re.compile(rf"^[{re.escape(_ARM_OUTSIDE_LEAD)}]+(?=\()")
_TRAIL_INSIDE_RE = re.compile(rf"[{re.escape(_ARM_INSIDE_TRAIL)}]+\)$")
_LEAD_INSIDE_RE = re.compile(rf"^\([{re.escape(_ARM_INSIDE_LEAD)}]+")

# Rules A and G: characters that sit between a face's glyphs without
# changing the expression. They are deleted.
#   A: zero-width space, non-joiner and joiner (U+200B-U+200D), word joiner
#      (U+2060), BOM (U+FEFF), ARABIC FOOTNOTE MARKER (U+0602) and the
#      variation selectors U+FE00-U+FE0F, which only request text or emoji
#      presentation: `♥` with either is the same face as bare `♥`.
#   G: combining overlays U+0334-U+0338 and U+033F, which land on eye
#      glyphs (a `˃` with a stroke through it). Other combining marks stay:
#      U+0300 and U+0301 are the accents in eyes like `(•̀_•́)`.
_INVISIBLE_CHARS = (
    "​‌‍⁠﻿؂"  # rule A
    "̴̵̶̷̸̿"               # rule G
    # Built from a range: sixteen invisible characters pasted as a literal
    # can't be checked by eye.
    + "".join(chr(cp) for cp in range(0xFE00, 0xFE10))
)

# Single-character folds for the translate pass.
_TYPO_SUBS: tuple[tuple[str, str], ...] = (
    # Fullwidth brackets and an arm letter
    ("）", ")"),   # full-width close paren
    ("（", "("),   # full-width open paren
    ("ｃ", "c"),   # full-width Latin c (arm modifier)
    # Rule B: halfwidth and fullwidth punctuation
    ("＞", ">"),   # FULLWIDTH GREATER-THAN SIGN
    ("＜", "<"),   # FULLWIDTH LESS-THAN SIGN
    ("；", ";"),   # FULLWIDTH SEMICOLON
    ("：", ":"),   # FULLWIDTH COLON
    ("＿", "_"),   # FULLWIDTH LOW LINE
    ("＊", "*"),   # FULLWIDTH ASTERISK
    # Not folded: `￣` (FULLWIDTH MACRON, flat: calm) and `~` (wavy: sleepy)
    # are different registers.
    ("｀", "`"),   # FULLWIDTH GRAVE ACCENT (rule O)
    # Further fullwidth forms, for symmetry with the pairs above.
    ("？", "?"),   # FULLWIDTH QUESTION MARK
    ("！", "!"),   # FULLWIDTH EXCLAMATION MARK
    ("．", "."),   # FULLWIDTH FULL STOP
    ("，", ","),   # FULLWIDTH COMMA
    ("／", "/"),   # FULLWIDTH SOLIDUS
    ("～", "~"),   # FULLWIDTH TILDE
    # Symbols that play the same role as an ASCII glyph fold to it.
    ("〜", "~"),   # WAVE DASH (U+301C), same role as `~`
    ("［", "["),   # FULLWIDTH LEFT SQUARE BRACKET
    ("］", "]"),   # FULLWIDTH RIGHT SQUARE BRACKET
    ("｜", "|"),   # FULLWIDTH VERTICAL LINE
    # Rule H: curly quotes fold to straight quotes.
    ("‘", "'"),  # LEFT SINGLE QUOTATION MARK
    ("’", "'"),  # RIGHT SINGLE QUOTATION MARK
    ("“", '"'),  # LEFT DOUBLE QUOTATION MARK
    ("”", '"'),  # RIGHT DOUBLE QUOTATION MARK
    # Rule M: circles filled in some direction fold to `◕`.
    ("◔", "◕"),   # CIRCLE WITH UPPER RIGHT QUADRANT BLACK
    ("◑", "◕"),   # CIRCLE WITH RIGHT HALF BLACK
    ("◐", "◕"),   # CIRCLE WITH LEFT HALF BLACK
    ("◒", "◕"),   # CIRCLE WITH LOWER HALF BLACK
    ("◓", "◕"),   # CIRCLE WITH UPPER HALF BLACK
    ("◖", "◕"),   # LEFT HALF BLACK CIRCLE (full-circle variant)
    ("◗", "◕"),   # RIGHT HALF BLACK CIRCLE (full-circle variant)
    # Circles with a visible center dot fold to `⊙` (wide-open, target eyes).
    ("◉", "⊙"),   # FISHEYE (Geometric Shapes block) -> CIRCLED DOT
    ("●", "⊙"),   # BLACK CIRCLE (fully solid)
    # Rule E1: degree-like circles fold to `°`.
    ("º", "°"),   # MASCULINE ORDINAL INDICATOR
    ("˚", "°"),   # RING ABOVE
    # Rules E2 and I: middle dots and the bullet fold to `・`. Smaller dots
    # (`·` U+00B7, `⋅` U+22C5) stay distinct; they may be another register.
    ("･", "・"),   # HALFWIDTH KATAKANA MIDDLE DOT
    ("•", "・"),   # BULLET (U+2022)
    # Rule N: smile-curve mouths fold to `‿`.
    ("ᴗ", "‿"),   # LATIN SMALL LETTER OPEN O / connector
    ("◡", "‿"),   # LOWER HALF CIRCLE
    ("ᵕ", "‿"),   # LATIN SMALL LETTER UP TACK
    ("⌣", "‿"),   # SMILE (U+2323)
    # Not folded: `﹏` (wavy, distressed: `(>﹏<)`) and `_` (flat, neutral:
    # `(•_•)`) are different registers.
    # Rule J: bracket-corner decorations fold to `｡`.
    ("◍", "｡"),   # CIRCLE WITH VERTICAL FILL (U+25CD)
    ("。", "｡"),   # IDEOGRAPHIC FULL STOP
)

# Rule K: a substring fold, applied after `_TYPO_SUBS` (so `•` is already
# `・`) and after whitespace stripping. A global `-` to `_` fold would
# corrupt `(´-ω-`)`, where `-` is a tired eye, so it is limited to a hyphen
# between middle dots: `(・-・)` becomes `(・_・)`.
_INTERNAL_SUBS: tuple[tuple[str, str], ...] = (
    ("・-・", "・_・"),
)


# The per-character rules as one translation table: invisibles (A, G) are
# deleted, `_TYPO_SUBS` entries (B, E1, E2, H, I, J, M, N, O and the paren
# and arm-letter folds) are replaced, Cyrillic capitals (D) are lowercased.
# No destination is also a source, so one `str.translate` pass equals
# applying the substitutions one after another.
def _build_translation_table() -> dict[int, int | None]:
    table: dict[int, int | None] = {}
    for c in _INVISIBLE_CHARS:
        table[ord(c)] = None
    for src, dst in _TYPO_SUBS:
        table[ord(src)] = ord(dst)
    for cp in range(0x0410, 0x0430):
        table[cp] = cp + 0x20
    return table


_TRANSLATE_TABLE = _build_translation_table()


def canonicalize_kaomoji(s: str) -> str:
    """Collapse trivial kaomoji variants to one canonical form.

    Applies NFC normalization, one `str.translate` pass (rules A, B, D, E1,
    E2, G, H, I, J, M, N, O), removal of ASCII spaces inside a parenthesized
    face (C), `_INTERNAL_SUBS` (K), then arm stripping (F, L). Eye, mouth
    and decoration differences that no rule folds stay distinct.

    Idempotent. Empty input returns "".
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s.strip())
    s = s.translate(_TRANSLATE_TABLE)
    if s.startswith("(") and s.endswith(")"):
        s = "(" + s[1:-1].replace(" ", "") + ")"
    for src, dst in _INTERNAL_SUBS:
        s = s.replace(src, dst)
    # Outside arms go first so the inside patterns see the parens
    # unobscured.
    s = _LEAD_OUTSIDE_RE.sub("", s)
    s = _TRAIL_OUTSIDE_RE.sub("", s)
    s = _LEAD_INSIDE_RE.sub("(", s)
    s = _TRAIL_INSIDE_RE.sub(")", s)
    return s
