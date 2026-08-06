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
  - :mod:`llmoji.synth_prompts` — :data:`LEXICON`,
    :data:`LEXICON_VERSION`, :data:`SYNTHESIS_SCHEMA`,
    :data:`CIRCUMPLEX_ANCHORS`, :data:`EXTENSION_AXES`,
    :data:`SYNTHESIZE_PROMPT`, :data:`DEFAULT_ANTHROPIC_MODEL_ID`,
    :data:`DEFAULT_OPENAI_MODEL_ID`.
  - :mod:`llmoji.scrape` — :class:`~llmoji.scrape.ScrapeRow` schema
    (span-only; no `kaomoji` / `kaomoji_label`).
  - :mod:`llmoji.providers` — :class:`~llmoji.providers.HookInstaller`
    interface and the three first-class providers.
  - The bundle schema written by :func:`llmoji.analyze.run_analyze`
    (top-level ``manifest.json`` carrying ``lexicon_version`` +
    per-source-model ``<slug>.jsonl`` rows shaped
    ``{kaomoji, count, synthesis: {primary_affect,
    stance_modality_function}}``) and enforced by
    :data:`llmoji.upload.BUNDLE_TOPLEVEL_ALLOWLIST` +
    :data:`llmoji.upload.BUNDLE_DATA_SUFFIX`.

Bumping any of those changes the cross-corpus invariant; treat as
a major version bump (and bump :data:`LEXICON_VERSION` if the
lexicon or schema rotates).
"""

from __future__ import annotations

# Single source of truth for the package version. ``pyproject.toml``
# resolves it dynamically via ``[tool.hatch.version] path =
# "llmoji/__init__.py"`` (hatch parses the literal without executing
# the module, so the eager re-exports below don't fire at build time).
__version__ = "2.1.1"

from ._util import flatten_synthesis
from .scrape import ScrapeRow, iter_all
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
)

__all__ = [
    "KAOMOJI_START_CHARS",
    "KaomojiMatch",
    "canonicalize_kaomoji",
    "extract",
    "is_kaomoji_candidate",
    "ScrapeRow",
    "iter_all",
    "LEXICON",
    "LEXICON_VERSION",
    "SYNTHESIS_SCHEMA",
    "CIRCUMPLEX_ANCHORS",
    "EXTENSION_AXES",
    "SYNTHESIZE_PROMPT",
    "DEFAULT_ANTHROPIC_MODEL_ID",
    "DEFAULT_OPENAI_MODEL_ID",
    "NUDGE_MESSAGE",
    "flatten_synthesis",
    "__version__",
]
