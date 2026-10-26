"""llmoji — collect kaomoji journals from coding agents, describe each
(model, face) cell with a locked adjective lexicon, and submit the
aggregates to a shared dataset.

The cross-corpus surface (changing any of it changes what submissions
mean, so it wants a major version and a dataset-card update):

- :mod:`llmoji.taxonomy` — :data:`KAOMOJI_START_CHARS`,
  :func:`is_kaomoji_candidate`, :func:`split_leading_kaomoji` (how a
  message divides into face and body), :func:`extract` and
  :func:`canonicalize_kaomoji` (how a face is bucketed).
- :mod:`llmoji.synth_prompts` — the lexicon, schema, prompt, pinned
  synthesis models and the kaomoji request (:data:`NUDGE_MESSAGE`).
- The 6-field journal row and the bundle layout written by
  :mod:`llmoji.analyze`.
"""

from __future__ import annotations

# Single source of the version; hatch reads this literal at build time.
__version__ = "3.0.0"

from ._util import flatten_synthesis
from .scrape import ScrapeRow
from .synth_prompts import (
    CIRCUMPLEX_ANCHORS,
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    EXTENSION_AXES,
    LEXICON,
    LEXICON_VERSION,
    NUDGE_MESSAGE,
    SYNTHESIS_SCHEMA,
    SYNTHESIZE_PROMPT,
)
from .taxonomy import (
    KAOMOJI_START_CHARS,
    KaomojiMatch,
    canonicalize_kaomoji,
    extract,
    is_kaomoji_candidate,
    split_leading_kaomoji,
)

__all__ = [
    "CIRCUMPLEX_ANCHORS",
    "DEFAULT_ANTHROPIC_MODEL_ID",
    "DEFAULT_OPENAI_MODEL_ID",
    "EXTENSION_AXES",
    "KAOMOJI_START_CHARS",
    "LEXICON",
    "LEXICON_VERSION",
    "NUDGE_MESSAGE",
    "SYNTHESIS_SCHEMA",
    "SYNTHESIZE_PROMPT",
    "KaomojiMatch",
    "ScrapeRow",
    "__version__",
    "canonicalize_kaomoji",
    "extract",
    "flatten_synthesis",
    "is_kaomoji_candidate",
    "split_leading_kaomoji",
]
