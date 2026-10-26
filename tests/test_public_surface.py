"""Lock the public surface (3.0.0).

If any test in this file fails, that's an aggregation-invariant
event: the HF dataset's aggregation rules pin against the lexicon,
schema, pinned models and the taxonomy, and changing one wants a
hand-edit on the dataset card to match. Also locked here: the provider
roster and interface, and the retired 2.x modules staying retired.

The bundle layout, hook rendering, install/uninstall and the synth
backends have their own test files.

The tests are intentionally conservative: they spot-check shape and a
few specific values rather than fingerprinting every glyph.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

import llmoji


def test_all_exports_resolve_and_version_is_3_0_0():
    assert llmoji.__version__ == "3.0.0"
    assert len(set(llmoji.__all__)) == len(llmoji.__all__)
    for name in llmoji.__all__:
        assert hasattr(llmoji, name), f"llmoji.__all__ lists missing {name!r}"
    # Everything the docstring calls cross-corpus is exported.
    assert {
        "KAOMOJI_START_CHARS", "is_kaomoji_candidate", "split_leading_kaomoji",
        "extract", "canonicalize_kaomoji", "LEXICON", "SYNTHESIS_SCHEMA",
        "NUDGE_MESSAGE", "ScrapeRow", "flatten_synthesis",
    } <= set(llmoji.__all__)


def test_taxonomy_surface_present():
    from llmoji.taxonomy import (
        KAOMOJI_START_CHARS,
        KaomojiMatch,
        canonicalize_kaomoji,
        extract,
        is_kaomoji_candidate,
        split_leading_kaomoji,
    )
    # KAOMOJI_START_CHARS: frozen set of leading-glyph chars.
    assert isinstance(KAOMOJI_START_CHARS, frozenset)
    assert "(" in KAOMOJI_START_CHARS and "[" in KAOMOJI_START_CHARS
    # canonicalize_kaomoji: idempotent on the empty string and on a sample.
    assert canonicalize_kaomoji("") == ""
    once = canonicalize_kaomoji("(；´Д｀)")
    assert canonicalize_kaomoji(once) == once
    # is_kaomoji_candidate: basic positive + negative.
    assert is_kaomoji_candidate("(｡◕‿◕｡)")
    assert not is_kaomoji_candidate("hi")
    # extract returns a span-only KaomojiMatch (the label dict lives in
    # llmoji_study.taxonomy_labels).
    m = extract("(｡◕‿◕｡) hi")
    assert isinstance(m, KaomojiMatch)
    assert m.first_word == "(｡◕‿◕｡)"
    # split_leading_kaomoji: the one face/body splitter; None without a face.
    assert split_leading_kaomoji("(｡◕‿◕｡) hi") == ("(｡◕‿◕｡)", "hi")
    assert split_leading_kaomoji("(•ᴗ•) **Done**") == ("(•ᴗ•)", "**Done**")
    assert split_leading_kaomoji("hi (｡◕‿◕｡)") is None


def test_flatten_synthesis():
    synth = {
        "primary_affect": ["cheerful"],
        "stance_modality_function": ["warm", "sincere"],
    }
    assert llmoji.flatten_synthesis(synth) == "cheerful, warm, sincere"
    assert llmoji.flatten_synthesis({}) == ""
    assert llmoji.flatten_synthesis({"primary_affect": "cheerful"}) == ""


def test_no_pilot_labels_in_public():
    """The TAXONOMY / ANGRY_CALM_TAXONOMY / label_on names are
    research-side; they must NOT appear on the public taxonomy
    module. v1.0 split locked this out."""
    import llmoji.taxonomy as tax
    for name in ("TAXONOMY", "ANGRY_CALM_TAXONOMY", "label_on", "POLE_NAMES"):
        assert not hasattr(tax, name), (
            f"llmoji.taxonomy.{name} leaked into the public package; "
            f"it belongs in llmoji_study.taxonomy_labels."
        )


def test_sanitize_model_id_for_path():
    """Bundle filename stem rule: lowercase, ``/`` → ``__``, ``:`` →
    ``-``, dots and digits preserved. Empty → ``"unknown"`` so a row
    with no model can't collapse into an unnamed path."""
    from llmoji._util import sanitize_model_id_for_path
    assert sanitize_model_id_for_path("") == "unknown"
    assert sanitize_model_id_for_path("claude-haiku-5-5") == "claude-haiku-5-5"
    assert sanitize_model_id_for_path("gpt-5.4-mini-2026-03-17") == (
        "gpt-5.4-mini-2026-03-17"
    )
    assert sanitize_model_id_for_path("meta/llama-3.1:70b") == (
        "meta__llama-3.1-70b"
    )
    # Idempotent on already-sanitized input.
    once = sanitize_model_id_for_path("Some-Model:Tag/Path")
    assert sanitize_model_id_for_path(once) == once


def test_synth_prompts_locked():
    from llmoji.synth_prompts import (
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
    # The single-stage prompt includes the {samples} substitution
    # point and the [FACE] mask token reference.
    assert "{samples}" in SYNTHESIZE_PROMPT
    assert "[FACE]" in SYNTHESIZE_PROMPT
    # Locked default model ids per backend (3.0 moved them off
    # claude-haiku-4-5-20251001 / gpt-5.4-mini-2026-03-17). Bumping
    # either is a cross-corpus invariant change: the dataset's
    # submitted adjective bags depend on which model produced them.
    assert DEFAULT_ANTHROPIC_MODEL_ID == "claude-haiku-5-5"
    assert DEFAULT_OPENAI_MODEL_ID == "gpt-6-luna"
    assert llmoji.DEFAULT_ANTHROPIC_MODEL_ID == DEFAULT_ANTHROPIC_MODEL_ID
    assert llmoji.DEFAULT_OPENAI_MODEL_ID == DEFAULT_OPENAI_MODEL_ID
    # The kaomoji request ``llmoji install`` appends to instructions files.
    assert isinstance(NUDGE_MESSAGE, str) and "kaomoji" in NUDGE_MESSAGE
    # Lexicon shape: every adjective is a unique lowercase string,
    # quadrant tags are constrained, families are non-empty.
    seen: set[str] = set()
    quadrants = {"HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB"}
    families = {"circumplex", "functional", "stance", "modality", "confidence"}
    quadrant_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for adj, q, fam in LEXICON:
        assert isinstance(adj, str) and adj == adj.lower(), adj
        assert adj not in seen, f"duplicate adjective: {adj!r}"
        seen.add(adj)
        if q is not None:
            assert q in quadrants, q
            quadrant_counts[q] = quadrant_counts.get(q, 0) + 1
            assert fam == "circumplex", (adj, fam)
        else:
            assert fam in families - {"circumplex"}, (adj, fam)
        family_counts[fam] = family_counts.get(fam, 0) + 1
    # Every quadrant has at least 2 anchors so PCA can triangulate.
    # NB intentionally has fewer flavors (absence of affect doesn't
    # need 4 distinct shapes); the affect-bearing quadrants carry 3+.
    for q in quadrants:
        floor = 2 if q == "NB" else 3
        assert quadrant_counts.get(q, 0) >= floor, (q, quadrant_counts)
    # Each extension family carries at least 2 entries, so a reply can
    # draw its stance/modality words from more than one family.
    for fam in families - {"circumplex"}:
        assert family_counts.get(fam, 0) >= 2, (fam, family_counts)
    # Schema enums match the LEXICON partition exactly. Pyright doesn't
    # track nested types through ``dict[str, object]``, so cast to Any.
    schema: Any = SYNTHESIS_SCHEMA
    primary_enum = schema["properties"]["primary_affect"]["items"]["enum"]
    extension_enum = schema["properties"][
        "stance_modality_function"]["items"]["enum"]
    assert tuple(primary_enum) == CIRCUMPLEX_ANCHORS
    assert tuple(extension_enum) == EXTENSION_AXES
    # No overlap between the two enums.
    assert not (set(primary_enum) & set(extension_enum))
    # Schema is a strict object with no free-form keys.
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(schema["required"]) == {
        "primary_affect", "stance_modality_function",
    }
    # LEXICON_VERSION must be a positive int: bundles stamp it.
    assert isinstance(LEXICON_VERSION, int) and LEXICON_VERSION >= 1


def test_scrape_row_schema():
    from dataclasses import fields
    from llmoji.scrape import ScrapeRow

    # NOTE: ``ScrapeRow`` is in-memory only; the cross-corpus
    # invariant is the on-disk 6-field journal row, not this
    # dataclass. Adding fields is forward-compat; removing or
    # renaming is fine because no external consumer reads the
    # dataclass directly.
    expected = {
        "source", "model", "timestamp", "cwd",
        "assistant_text", "first_word", "surrounding_user",
    }
    got = {f.name for f in fields(ScrapeRow)}
    assert got == expected, (
        f"ScrapeRow shape drifted; expected {expected}, got {got}"
    )
    assert llmoji.ScrapeRow is ScrapeRow


PROVIDER_INTERFACE = (
    "install", "uninstall", "status", "live_rows", "replay", "session_files",
    "importable", "render_hook", "hook_path", "doc_path", "journal_path",
)


def test_provider_interface():
    from llmoji.providers import PROVIDERS, Provider, get_provider

    # Four first-class providers; 3.0 retired hermes, opencode and openclaw.
    assert set(PROVIDERS) == {"claude_code", "codex", "grok", "antigravity"}
    for name, cls in PROVIDERS.items():
        assert issubclass(cls, Provider)
        for attr in PROVIDER_INTERFACE:
            assert hasattr(cls, attr), f"{name} lacks {attr}"
        p = get_provider(name)
        assert isinstance(p, cls)
        assert p.name == name
        for method in ("install", "uninstall", "status", "live_rows",
                       "replay", "session_files", "render_hook"):
            assert callable(getattr(p, method)), (name, method)
        assert isinstance(p.render_hook(), str)
        assert isinstance(p.importable, bool)
        for path in (p.hook_path, p.doc_path, p.journal_path):
            assert isinstance(path, Path) and path.is_absolute(), (name, path)
    # Antigravity's transcripts carry no per-step model id, so it has
    # no replay; the other three feed ``llmoji import``.
    assert {n for n, c in PROVIDERS.items() if get_provider(n).importable} == {
        "claude_code", "codex", "grok",
    }


@pytest.mark.parametrize("name", ["hermes", "opencode", "openclaw"])
def test_retired_providers_are_unknown(name: str):
    from llmoji.providers import get_provider
    with pytest.raises(KeyError):
        get_provider(name)


def _survives(module: str) -> bool:
    """Whether ``module`` still imports as code. A pre-3.0 checkout can
    leave a directory of compiled ``__pycache__`` behind, which Python
    imports as an empty namespace package; that is not the module."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False
    if getattr(mod, "__file__", None):
        return True
    return any(
        entry.name != "__pycache__"
        for root in getattr(mod, "__path__", [])
        for entry in Path(root).iterdir()
    )


@pytest.mark.parametrize(
    "module",
    [
        "llmoji.backfill",
        "llmoji.providers.hermes",
        "llmoji.providers.opencode",
        "llmoji.providers.openclaw",
        "llmoji.sources.gemini_export",
        "llmoji.sources.openhands_export",
        "llmoji.sources.claude_export_alt",
        "llmoji._hooks",
        "llmoji._plugins",
    ],
)
def test_retired_modules_are_gone(module: str):
    assert not _survives(module), f"{module} was retired in 3.0 but imports"
