"""Lock the v1.0 frozen public surface.

If any test in this file fails, that's a major-version event — the
HF dataset README declares "v1 corpus only" aggregation rules
against these invariants. Bumping invalidates cross-corpus
comparison.

The tests are intentionally conservative — they spot-check shape
and a few specific values rather than fingerprinting every glyph.
The fingerprint is documented in CLAUDE.md.
"""

from __future__ import annotations

import json


def test_taxonomy_surface_present():
    from llmoji.taxonomy import (
        KAOMOJI_START_CHARS,
        KaomojiMatch,
        TAXONOMY,
        canonicalize_kaomoji,
        extract,
        is_kaomoji_candidate,
    )
    # KAOMOJI_START_CHARS — frozen set of leading-glyph chars.
    assert isinstance(KAOMOJI_START_CHARS, frozenset)
    assert "(" in KAOMOJI_START_CHARS and "[" in KAOMOJI_START_CHARS
    # TAXONOMY — gemma-tuned + extended; non-empty, dict[str, int].
    assert TAXONOMY and all(isinstance(v, int) for v in TAXONOMY.values())
    # canonicalize_kaomoji — idempotent on the empty string.
    assert canonicalize_kaomoji("") == ""
    # And idempotent on a sample.
    once = canonicalize_kaomoji("(；´Д｀)")
    assert canonicalize_kaomoji(once) == once
    # is_kaomoji_candidate basic positive + negative.
    assert is_kaomoji_candidate("(｡◕‿◕｡)")
    assert not is_kaomoji_candidate("hi")
    # extract returns a KaomojiMatch with the public field set.
    m = extract("(｡◕‿◕｡) hi")
    assert isinstance(m, KaomojiMatch)
    assert m.label == +1


def test_haiku_prompts_locked():
    from llmoji.haiku_prompts import (
        DESCRIBE_PROMPT_NO_USER,
        DESCRIBE_PROMPT_WITH_USER,
        HAIKU_MODEL_ID,
        SYNTHESIZE_PROMPT,
    )
    # All three prompts include the [FACE] mask token reference or a
    # Synthesize tail. Smoke-checks that we're shipping the right text.
    assert "{user_text}" in DESCRIBE_PROMPT_WITH_USER
    assert "{masked_text}" in DESCRIBE_PROMPT_WITH_USER
    assert "{masked_text}" in DESCRIBE_PROMPT_NO_USER
    assert "{descriptions}" in SYNTHESIZE_PROMPT
    # Locked Haiku model id.
    assert HAIKU_MODEL_ID == "claude-haiku-4-5-20251001"


def test_scrape_row_schema():
    from dataclasses import fields
    from llmoji.scrape import ScrapeRow

    expected = {
        "source", "session_id", "project_slug", "assistant_uuid",
        "parent_uuid", "model", "timestamp", "cwd", "git_branch",
        "turn_index", "had_thinking", "assistant_text", "first_word",
        "kaomoji", "kaomoji_label", "surrounding_user",
    }
    got = {f.name for f in fields(ScrapeRow)}
    # The frozen v1.0 surface — additions are OK (forward compat),
    # removals or renames are major version events.
    missing = expected - got
    assert not missing, f"ScrapeRow missing fields: {missing}"


def test_provider_interface():
    from llmoji.providers import PROVIDERS, get_provider
    from llmoji.providers.base import Provider

    # All three first-class providers register.
    assert set(PROVIDERS) == {"claude_code", "codex", "hermes"}
    for name in PROVIDERS:
        p = get_provider(name)
        assert isinstance(p, Provider)
        # All five required attributes are non-empty.
        assert p.name == name
        assert p.hooks_dir
        assert p.settings_path
        assert p.journal_path
        assert p.hook_template
        # Render a hook and check no placeholders leaked through.
        rendered = p.render_hook()
        assert "$JOURNAL_PATH" not in rendered
        assert "$KAOMOJI_START_CASE" not in rendered
        assert "$INJECTED_PREFIXES_FILTER" not in rendered
        assert "$LLMOJI_VERSION" not in rendered


def test_bundle_schema():
    """The bundle is the only thing that leaves the user's machine.
    Lock the schema by writing a fake bundle and re-reading it.

    Manifest must be well-formed JSON with the documented keys;
    descriptions.jsonl must be one row per canonical kaomoji with
    the four documented keys.
    """
    from llmoji.analyze import _write_bundle
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td)
        _write_bundle(
            bundle,
            counts_by_canon={"(｡◕‿◕｡)": 12, "(╥﹏╥)": 3},
            synthesized_by_canon={
                "(｡◕‿◕｡)": "Warm reassurance.",
                "(╥﹏╥)": "Tearful frustration.",
            },
            providers_seen=["claude_code-hook", "codex-hook"],
            notes="test",
        )
        manifest = json.loads((bundle / "manifest.json").read_text())
        for k in (
            "llmoji_version", "generated_at", "providers_seen",
            "total_rows_scraped", "total_kaomoji_unique_canonical",
            "notes",
        ):
            assert k in manifest, f"missing manifest key: {k}"

        rows = [
            json.loads(l)
            for l in (bundle / "descriptions.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert len(rows) == 2
        for r in rows:
            assert set(r) == {
                "kaomoji", "count", "haiku_synthesis_description",
                "llmoji_version",
            }


def test_hook_templates_render_to_valid_bash_substitutions():
    """The base renderer should ignore unknown $VARS in the template
    body (so embedded jq / sed `$KAOMOJI` etc. aren't eaten by the
    Python Template substitution). string.Template's safe_substitute
    handles this — verify."""
    from llmoji.providers import get_provider
    for name in ("claude_code", "codex", "hermes"):
        p = get_provider(name)
        rendered = p.render_hook()
        # The rendered hook still contains shell-side $KAOMOJI refs
        # (they're bash variables, not Python placeholders).
        assert "KAOMOJI" in rendered
        # And no Python Template placeholder leaked.
        for placeholder in (
            "${JOURNAL_PATH}", "${KAOMOJI_START_CASE}",
            "${INJECTED_PREFIXES_FILTER}", "${LLMOJI_VERSION}",
        ):
            assert placeholder not in rendered, (
                f"{name} kept literal {placeholder}; "
                "did substitution drop the placeholder?"
            )
