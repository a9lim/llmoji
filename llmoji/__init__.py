"""llmoji — provider-agnostic kaomoji journal collection + canonical
Haiku synthesis + privacy-preserving aggregate submission.

The end-user CLI is :mod:`llmoji.cli`. The locked v1.0 public surface
(invariants for cross-corpus aggregation):

  - :mod:`llmoji.taxonomy` — :data:`~llmoji.taxonomy.KAOMOJI_START_CHARS`,
    :func:`~llmoji.taxonomy.is_kaomoji_candidate`,
    :func:`~llmoji.taxonomy.extract` (span-only),
    :class:`~llmoji.taxonomy.KaomojiMatch`,
    :func:`~llmoji.taxonomy.canonicalize_kaomoji` (rules A–P).
    Pilot-specific affect labels (TAXONOMY / ANGRY_CALM_TAXONOMY /
    label_on / pole) live research-side at
    ``llmoji_study.taxonomy_labels``.
  - :mod:`llmoji.haiku_prompts` — DESCRIBE_PROMPT_*,
    SYNTHESIZE_PROMPT, HAIKU_MODEL_ID.
  - :mod:`llmoji.scrape` — :class:`~llmoji.scrape.ScrapeRow` schema
    (span-only; no `kaomoji` / `kaomoji_label`).
  - :mod:`llmoji.providers` — :class:`~llmoji.providers.Provider`
    interface and the three first-class providers.
  - The bundle schema written by :func:`llmoji.analyze.run_analyze`
    (``manifest.json`` + ``descriptions.jsonl``) and enforced by
    :data:`llmoji.upload.BUNDLE_ALLOWLIST`.

Bumping any of those is a major version bump (``llmoji`` 2.0.0).
"""

from __future__ import annotations

# Single source of truth for the package version. ``pyproject.toml``
# resolves it dynamically via ``[tool.hatch.version] path =
# "llmoji/__init__.py"`` (hatch parses the literal without executing
# the module, so the eager re-exports below don't fire at build time).
__version__ = "1.0.0"

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
