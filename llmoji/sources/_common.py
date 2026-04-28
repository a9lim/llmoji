"""Shared helpers for the per-source ``ScrapeRow`` iterators.

Every export reader does the same dance against
:func:`llmoji.taxonomy.extract`:

  1. Pull the leading kaomoji span out of the raw assistant text.
  2. Reject the message if there is no valid leading span.
  3. Strip the prefix + surrounding whitespace from the body so the
     v1.0 journal-row contract holds (``assistant_text`` never carries
     the kaomoji — it lives in the row's ``kaomoji`` field, see the
     "Journal-row contract" gotcha in :file:`CLAUDE.md`).

Lifted into one helper so adding a new export reader doesn't reach
for the strip-and-validate code path by hand and silently produce
kaomoji-bearing ``assistant_text`` rows that would force
:func:`llmoji.haiku.mask_kaomoji` to grow a second branch.
"""

from __future__ import annotations

from ..taxonomy import extract


def kaomoji_lead_strip(text: str) -> tuple[str, str] | None:
    """Validate + split the leading kaomoji from a raw assistant text.

    Returns ``(first_word, body)`` on success, where ``first_word`` is
    the validated leading kaomoji span and ``body`` is the message
    text with that prefix and surrounding whitespace stripped.
    Returns ``None`` when the text doesn't open with a kaomoji that
    passes :func:`~llmoji.taxonomy.extract` (so callers can ``continue``
    cleanly in their per-message loop).

    The leading-glyph filter is enforced inside ``extract`` (via
    :func:`~llmoji.taxonomy.is_kaomoji_candidate`) — a non-empty
    ``first_word`` already guarantees ``first_word[0]`` is in
    :data:`~llmoji.taxonomy.KAOMOJI_START_CHARS`. Single source of
    truth lives in ``taxonomy``; don't re-check here.
    """
    match = extract(text)
    if not match.first_word:
        return None
    body = text.lstrip()
    if body.startswith(match.first_word):
        body = body[len(match.first_word):].lstrip()
    return match.first_word, body
