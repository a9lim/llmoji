"""llmoji — provider-agnostic kaomoji journal collection + canonical
synthesis + privacy-preserving aggregate submission.

The end-user CLI is :mod:`llmoji.cli`. The locked public surface
(invariants for cross-corpus aggregation):

  - :mod:`llmoji.taxonomy` — :data:`~llmoji.taxonomy.KAOMOJI_START_CHARS`,
    :func:`~llmoji.taxonomy.is_kaomoji_candidate`,
    :func:`~llmoji.taxonomy.extract` (span-only),
    :class:`~llmoji.taxonomy.KaomojiMatch`,
    :func:`~llmoji.taxonomy.canonicalize_kaomoji` (rules A–P).
    Pilot-specific affect labels (TAXONOMY / ANGRY_CALM_TAXONOMY /
    label_on / pole) live research-side at
    ``llmoji_study.taxonomy_labels``.
  - :mod:`llmoji.synth_prompts` — DESCRIBE_PROMPT_*,
    SYNTHESIZE_PROMPT, DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID.
  - :mod:`llmoji.scrape` — :class:`~llmoji.scrape.ScrapeRow` schema
    (span-only; no `kaomoji` / `kaomoji_label`).
  - :mod:`llmoji.providers` — :class:`~llmoji.providers.HookInstaller`
    interface and the three first-class providers.
  - The bundle schema written by :func:`llmoji.analyze.run_analyze`
    (top-level ``manifest.json`` + per-source-model
    ``<slug>.jsonl``) and enforced by
    :data:`llmoji.upload.BUNDLE_TOPLEVEL_ALLOWLIST` +
    :data:`llmoji.upload.BUNDLE_DATA_SUFFIX`.

Bumping any of those changes the cross-corpus invariant; treat as
a major version bump.
"""

from __future__ import annotations

# Single source of truth for the package version. ``pyproject.toml``
# resolves it dynamically via ``[tool.hatch.version] path =
# "llmoji/__init__.py"`` (hatch parses the literal without executing
# the module, so the eager re-exports below don't fire at build time).
__version__ = "2.0.0"

from .scrape import ScrapeRow, iter_all
from .synth_prompts import (
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    DESCRIBE_PROMPT_NO_USER,
    DESCRIBE_PROMPT_WITH_USER,
    SYNTHESIZE_PROMPT,
)
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
    "DEFAULT_ANTHROPIC_MODEL_ID",
    "DEFAULT_OPENAI_MODEL_ID",
    "__version__",
]
