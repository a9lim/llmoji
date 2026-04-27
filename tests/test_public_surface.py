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
        canonicalize_kaomoji,
        extract,
        is_kaomoji_candidate,
    )
    # KAOMOJI_START_CHARS — frozen set of leading-glyph chars.
    assert isinstance(KAOMOJI_START_CHARS, frozenset)
    assert "(" in KAOMOJI_START_CHARS and "[" in KAOMOJI_START_CHARS
    # canonicalize_kaomoji — idempotent on the empty string.
    assert canonicalize_kaomoji("") == ""
    # And idempotent on a sample.
    once = canonicalize_kaomoji("(；´Д｀)")
    assert canonicalize_kaomoji(once) == once
    # is_kaomoji_candidate basic positive + negative.
    assert is_kaomoji_candidate("(｡◕‿◕｡)")
    assert not is_kaomoji_candidate("hi")
    # extract returns a KaomojiMatch with the public field set
    # (post-v1.0: span-only; gemma-tuned label dict moved to
    # llmoji_study.taxonomy_labels).
    m = extract("(｡◕‿◕｡) hi")
    assert isinstance(m, KaomojiMatch)
    assert m.first_word == "(｡◕‿◕｡)"


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
        "surrounding_user",
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


def test_bundle_allowlist_rejects_extras():
    """`tar_bundle` must refuse to ship anything outside
    {manifest.json, descriptions.jsonl} — that's the v1.0 frozen
    bundle schema. Loud failure beats silent leak."""
    from pathlib import Path
    import tempfile

    from llmoji.analyze import _write_bundle
    from llmoji.upload import BUNDLE_ALLOWLIST, tar_bundle

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        _write_bundle(
            bundle,
            counts_by_canon={"(◕‿◕)": 1},
            synthesized_by_canon={"(◕‿◕)": "smile"},
            providers_seen=[],
            notes="",
        )
        # Add an extra — tar should refuse.
        (bundle / "stray.txt").write_text("would leak")
        try:
            tar_bundle(bundle, out_path=Path(td) / "out.tgz")
        except FileExistsError:
            pass
        else:
            raise AssertionError("tar_bundle didn't refuse extras")
        # Allowlist itself is the frozen pair.
        assert set(BUNDLE_ALLOWLIST) == {"manifest.json", "descriptions.jsonl"}


def test_corrupt_settings_refused():
    """Provider install must refuse to mutate a corrupt-but-existing
    settings file. Silently wiping a user's config is a regression
    we never want to ship."""
    from pathlib import Path
    import tempfile

    from llmoji.providers import get_provider
    from llmoji.providers.base import SettingsCorruptError

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Claude Code: malformed JSON
        cc = get_provider("claude_code")
        cc.hooks_dir = td / "claude" / "hooks"
        cc.settings_path = td / "claude" / "settings.json"
        cc.journal_path = td / "claude" / "journal.jsonl"
        cc.settings_path.parent.mkdir(parents=True, exist_ok=True)
        cc.settings_path.write_text("{ not json")
        try:
            cc.install()
        except SettingsCorruptError:
            pass
        else:
            raise AssertionError("claude_code didn't refuse corrupt JSON")
        assert cc.settings_path.read_text() == "{ not json", (
            "corrupt config was modified"
        )

        # Codex: existing [hooks.stop]
        cx = get_provider("codex")
        cx.hooks_dir = td / "codex" / "hooks"
        cx.settings_path = td / "codex" / "config.toml"
        cx.journal_path = td / "codex" / "journal.jsonl"
        cx.settings_path.parent.mkdir(parents=True, exist_ok=True)
        original = '[hooks.stop]\ncommand = "/user/own/hook.sh"\n'
        cx.settings_path.write_text(original)
        try:
            cx.install()
        except SettingsCorruptError:
            pass
        else:
            raise AssertionError("codex didn't refuse existing [hooks.stop]")
        assert cx.settings_path.read_text() == original


def test_mask_kaomoji_handles_pre_stripped_journal_rows():
    """Live-hook journals already strip the leading kaomoji from
    ``assistant_text`` (the bash hook's ``ltrimstr($kaomoji)`` is
    part of the v1.0 journal contract). ``mask_kaomoji`` must still
    return text starting with ``[FACE]`` so the Haiku DESCRIBE
    prompts' "we replaced it with [FACE]" framing matches what
    Haiku actually sees. Bug pre-fix: the pre-stripped branch fell
    through to ``return text`` and Haiku got a prompt promising a
    [FACE] that wasn't in the body."""
    from llmoji.haiku import MASK_TOKEN, mask_kaomoji

    # Journal-source row: hook already stripped, so text doesn't
    # start with first_word. Must still get masked.
    masked = mask_kaomoji("I think we should refactor.", "(◕‿◕)")
    assert masked.startswith(MASK_TOKEN), masked
    assert "I think we should refactor." in masked

    # Static-export row: kaomoji still at the head. Substituted in
    # place.
    masked = mask_kaomoji("(◕‿◕) I think we should refactor.", "(◕‿◕)")
    assert masked.startswith(MASK_TOKEN)
    assert "(◕‿◕)" not in masked

    # Empty first_word — pass through unchanged.
    assert mask_kaomoji("hello", "") == "hello"


def test_hook_templates_render_to_valid_bash_substitutions():
    """The base renderer should ignore unknown $VARS in the template
    body (so embedded jq / sed `$KAOMOJI` etc. aren't eaten by the
    Python Template substitution). string.Template's safe_substitute
    handles this — verify. Also bash -n the rendered output so a
    template syntax error fails CI rather than failing silently
    inside the user's harness post-install."""
    import shutil
    import subprocess
    import tempfile
    from llmoji.providers import get_provider
    bash = shutil.which("bash")
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
        # bash -n syntax check on the rendered output — catches
        # template-edit regressions that don't show up in pure
        # placeholder checks.
        if bash:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".sh", delete=False,
            ) as f:
                f.write(rendered)
                tmp = f.name
            r = subprocess.run(
                [bash, "-n", tmp], capture_output=True, text=True,
            )
            assert r.returncode == 0, (
                f"{name} hook failed bash -n: {r.stderr}"
            )
