"""llmoji — provider-agnostic kaomoji journal collection + canonical
Haiku synthesis + privacy-preserving aggregate submission.

The end-user CLI is :mod:`llmoji.cli`. The locked v1.0 public surface
(invariants for cross-corpus aggregation):

  - :mod:`llmoji.taxonomy` — KAOMOJI_TAXONOMY,
    :data:`~llmoji.taxonomy.KAOMOJI_START_CHARS`,
    :func:`~llmoji.taxonomy.canonicalize_kaomoji`,
    :func:`~llmoji.taxonomy.is_kaomoji_candidate`,
    :func:`~llmoji.taxonomy.extract`.
  - :mod:`llmoji.haiku_prompts` — DESCRIBE_PROMPT_*, SYNTHESIZE_PROMPT,
    HAIKU_MODEL_ID.
  - :mod:`llmoji.scrape` — :class:`~llmoji.scrape.ScrapeRow` schema.
  - :mod:`llmoji.providers` — :class:`~llmoji.providers.Provider`
    interface and the three first-class providers.

Bumping any of those is a major version bump (``llmoji`` 2.0.0).
"""

from __future__ import annotations

from .haiku_prompts import (
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    HAIKU_MODEL_ID,
    SYNTHESIZE_PROMPT,
)
from .scrape import ScrapeRow, iter_all
from .taxonomy import (
    KAOMOJI_START_CHARS,
    KaomojiMatch,
    canonicalize_kaomoji,
    extract,
    is_kaomoji_candidate,
)

try:
    from importlib.metadata import version as _v
    __version__ = _v("llmoji")
except Exception:
    __version__ = "0.0.0+dev"

__all__ = [
    "KAOMOJI_START_CHARS",
    "KaomojiMatch",
    "canonicalize_kaomoji",
    "extract",
    "is_kaomoji_candidate",
    "ScrapeRow",
    "iter_all",
    "DESCRIBE_PROMPT_WITH_USER",
    "DESCRIBE_PROMPT_NO_USER",
    "SYNTHESIZE_PROMPT",
    "HAIKU_MODEL_ID",
    "__version__",
]
