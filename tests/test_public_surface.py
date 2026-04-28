"""Lock the public surface (1.1.0 amended from the 1.0 freeze:
manifest schema, descriptions row schema, the two pinned default
model ids, and the per-source-model bundle layout were all bumped).

If any test in this file fails, that's an aggregation-invariant
event — the HF dataset's aggregation rules pin against these
schemas, and changing one wants a hand-edit on the dataset card to
match.

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


def test_sanitize_model_id_for_path():
    """Subfolder-name sanitization rule: lowercase, ``/`` → ``__``,
    ``:`` → ``-``, dots and digits preserved. Empty → ``"unknown"``
    (defensive — keeps rows with empty ``ScrapeRow.model`` from
    collapsing into an unnamed top-level path)."""
    from llmoji._util import sanitize_model_id_for_path
    assert sanitize_model_id_for_path("") == "unknown"
    assert sanitize_model_id_for_path("claude-haiku-4-5-20251001") == (
        "claude-haiku-4-5-20251001"
    )
    assert sanitize_model_id_for_path("gpt-5.4-mini-2026-03-17") == (
        "gpt-5.4-mini-2026-03-17"
    )
    assert sanitize_model_id_for_path("meta/llama-3.1:70b") == (
        "meta__llama-3.1-70b"
    )
    # Idempotent on already-sanitized input.
    once = sanitize_model_id_for_path("Some-Model:Tag/Path")
    assert sanitize_model_id_for_path(once) == once


def test_make_synthesizer_dispatches():
    """Factory must return a backend-correct Synthesizer subclass
    with the pinned default model id (anthropic, openai) or the
    explicit one (local). Construction must NOT make a network
    call — these objects sit on the call site for the duration of
    a Stage A pass.
    """
    from llmoji.synth import (
        AnthropicSynthesizer,
        LocalSynthesizer,
        OpenAISynthesizer,
        make_synthesizer,
    )
    from llmoji.synth_prompts import (
        DEFAULT_ANTHROPIC_MODEL_ID,
        DEFAULT_OPENAI_MODEL_ID,
    )

    # Anthropic — pinned snapshot.
    s = make_synthesizer("anthropic")
    assert isinstance(s, AnthropicSynthesizer)
    assert s.backend == "anthropic"
    assert s.model_id == DEFAULT_ANTHROPIC_MODEL_ID

    # OpenAI — pinned snapshot.
    s = make_synthesizer("openai")
    assert isinstance(s, OpenAISynthesizer)
    assert s.backend == "openai"
    assert s.model_id == DEFAULT_OPENAI_MODEL_ID

    # Local — explicit base_url + model_id required.
    s = make_synthesizer(
        "local",
        base_url="http://localhost:11434/v1",
        model_id="llama3.1",
    )
    assert isinstance(s, LocalSynthesizer)
    assert s.backend == "local"
    assert s.model_id == "llama3.1"

    # Local without --base-url / --model → ValueError (CLI converts
    # to argparse-level error in real flow; the factory raises
    # directly).
    try:
        make_synthesizer("local")
    except ValueError:
        pass
    else:
        raise AssertionError("local backend without args didn't raise")

    # Unknown backend → ValueError.
    try:
        make_synthesizer("nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend didn't raise")


def test_synth_prompts_locked():
    from llmoji.synth_prompts import (
        DEFAULT_ANTHROPIC_MODEL_ID,
        DEFAULT_OPENAI_MODEL_ID,
        DESCRIBE_PROMPT_NO_USER,
        DESCRIBE_PROMPT_WITH_USER,
        SYNTHESIZE_PROMPT,
    )
    # All three prompts include the [FACE] mask token reference or a
    # Synthesize tail. Smoke-checks that we're shipping the right text.
    assert "{user_text}" in DESCRIBE_PROMPT_WITH_USER
    assert "{masked_text}" in DESCRIBE_PROMPT_WITH_USER
    assert "{masked_text}" in DESCRIBE_PROMPT_NO_USER
    assert "{descriptions}" in SYNTHESIZE_PROMPT
    # Locked default model ids per backend. Bumping either is a
    # cross-corpus invariant change — the dataset's submitted prose
    # depends on which snapshot produced it.
    assert DEFAULT_ANTHROPIC_MODEL_ID == "claude-haiku-4-5-20251001"
    assert DEFAULT_OPENAI_MODEL_ID == "gpt-5.4-mini-2026-03-17"


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

    Manifest must be well-formed JSON with the documented keys.
    Per-source-model subfolders each carry one ``descriptions.jsonl``
    with one row per canonical kaomoji; row keys = the three
    documented keys.
    """
    from llmoji.analyze import _write_bundle
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td)
        _write_bundle(
            bundle,
            counts_by_cell={
                "claude-sonnet-4-5-20250929": {"(｡◕‿◕｡)": 12, "(╥﹏╥)": 3},
                "gpt-5.4-mini-2026-03-17": {"(｡◕‿◕｡)": 7},
            },
            synthesized_by_cell={
                "claude-sonnet-4-5-20250929": {
                    "(｡◕‿◕｡)": "Warm reassurance.",
                    "(╥﹏╥)": "Tearful frustration.",
                },
                "gpt-5.4-mini-2026-03-17": {
                    "(｡◕‿◕｡)": "Polite enthusiasm.",
                },
            },
            providers_seen=["claude_code-hook", "codex-hook"],
            model_counts={
                "claude-sonnet-4-5-20250929": 15,
                "gpt-5.4-mini-2026-03-17": 7,
            },
            submitter_id="0" * 32,
            synth_backend="anthropic",
            synth_model_id="claude-haiku-4-5-20251001",
            notes="test",
        )
        manifest = json.loads((bundle / "manifest.json").read_text())
        for k in (
            "llmoji_version", "synthesis_model_id", "synthesis_backend",
            "submitter_id", "generated_at", "providers_seen",
            "model_counts", "total_synthesized_rows", "notes",
        ):
            assert k in manifest, f"missing manifest key: {k}"
        # Removed keys must NOT come back.
        for gone in (
            "haiku_model_id", "journal_counts", "total_rows_scraped",
            "total_kaomoji_unique_canonical",
        ):
            assert gone not in manifest, (
                f"{gone!r} should be removed in 1.1.0 but still present"
            )
        # total_synthesized_rows = sum across folders (face appearing
        # in 2 folders contributes 2).
        assert manifest["total_synthesized_rows"] == 3
        assert manifest["synthesis_backend"] == "anthropic"
        assert manifest["synthesis_model_id"] == "claude-haiku-4-5-20251001"

        # One subfolder per source model, sanitized name.
        subdirs = sorted(p.name for p in bundle.iterdir() if p.is_dir())
        assert subdirs == [
            "claude-sonnet-4-5-20250929",
            "gpt-5.4-mini-2026-03-17",
        ]
        for sub in subdirs:
            descriptions = bundle / sub / "descriptions.jsonl"
            assert descriptions.exists()
            rows = [
                json.loads(l)
                for l in descriptions.read_text().splitlines()
                if l.strip()
            ]
            assert rows, f"empty descriptions.jsonl in {sub}/"
            for r in rows:
                assert set(r) == {
                    "kaomoji", "count", "synthesis_description",
                }, f"unexpected row keys in {sub}/: {set(r)}"


def test_bundle_allowlist_rejects_extras():
    """`tar_bundle` must refuse to ship anything outside the
    structural allowlist (top-level ``manifest.json`` + each
    ``<source-model>/descriptions.jsonl``). Loud failure beats
    silent leak — covered both for an extra top-level file AND a
    stray file inside a model subfolder.
    """
    from pathlib import Path
    import tempfile

    from llmoji.analyze import _write_bundle
    from llmoji.upload import (
        BUNDLE_SUBDIR_FILE,
        BUNDLE_TOPLEVEL_ALLOWLIST,
        BundleAllowlistError,
        tar_bundle,
    )

    def _fresh_bundle(td: Path) -> Path:
        bundle = td / "bundle"
        _write_bundle(
            bundle,
            counts_by_cell={"claude-haiku-4-5-20251001": {"(◕‿◕)": 1}},
            synthesized_by_cell={"claude-haiku-4-5-20251001": {"(◕‿◕)": "smile"}},
            providers_seen=[],
            model_counts={},
            submitter_id="0" * 32,
            synth_backend="anthropic",
            synth_model_id="claude-haiku-4-5-20251001",
            notes="",
        )
        return bundle

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. extra top-level file
        bundle = _fresh_bundle(td)
        (bundle / "stray.txt").write_text("would leak")
        try:
            tar_bundle(bundle, out_path=td / "out1.tgz")
        except BundleAllowlistError:
            pass
        else:
            raise AssertionError(
                "tar_bundle didn't refuse extra top-level file",
            )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 2. extra file inside a model subfolder
        bundle = _fresh_bundle(td)
        sub = bundle / "claude-haiku-4-5-20251001"
        (sub / "extra.txt").write_text("inner stash")
        try:
            tar_bundle(bundle, out_path=td / "out2.tgz")
        except BundleAllowlistError:
            pass
        else:
            raise AssertionError(
                "tar_bundle didn't refuse extra subdir file",
            )

    # Allowlist constants are the frozen shape.
    assert BUNDLE_TOPLEVEL_ALLOWLIST == ("manifest.json",)
    assert BUNDLE_SUBDIR_FILE == "descriptions.jsonl"


def test_bundle_allowlist_rejects_symlinks_and_empty_subdirs():
    """Symlinks bypass the file/dir contract because ``Path.is_file``
    follows them; the allowlist must explicitly reject symlinks at
    every layer. Empty model subdirs misrepresent which models
    contributed; reject those too.
    """
    from pathlib import Path
    import tempfile

    from llmoji.analyze import _write_bundle
    from llmoji.upload import BundleAllowlistError, tar_bundle

    def _fresh(td: Path) -> Path:
        bundle = td / "bundle"
        _write_bundle(
            bundle,
            counts_by_cell={"claude-haiku-4-5-20251001": {"(◕‿◕)": 1}},
            synthesized_by_cell={"claude-haiku-4-5-20251001": {"(◕‿◕)": "smile"}},
            providers_seen=[],
            model_counts={},
            submitter_id="0" * 32,
            synth_backend="anthropic",
            synth_model_id="claude-haiku-4-5-20251001",
            notes="",
        )
        return bundle

    # 1. Symlinked manifest at top level.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bundle = _fresh(td)
        # Replace the real manifest with a symlink pointing somewhere else.
        target = td / "leak.json"
        target.write_text("{}")
        (bundle / "manifest.json").unlink()
        (bundle / "manifest.json").symlink_to(target)
        try:
            tar_bundle(bundle, out_path=td / "out.tgz")
        except BundleAllowlistError:
            pass
        else:
            raise AssertionError(
                "tar_bundle didn't refuse symlinked manifest",
            )

    # 2. Empty model subdir (descriptions.jsonl removed).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bundle = _fresh(td)
        sub = bundle / "claude-haiku-4-5-20251001"
        (sub / "descriptions.jsonl").unlink()
        try:
            tar_bundle(bundle, out_path=td / "out.tgz")
        except BundleAllowlistError:
            pass
        else:
            raise AssertionError(
                "tar_bundle didn't refuse empty model subdir",
            )


def test_write_bundle_rejects_slug_collision():
    """Two distinct ``ScrapeRow.model`` strings that sanitize to
    the same subfolder slug must NOT both write to the same
    ``descriptions.jsonl`` — that would silently overwrite. Loud
    failure beats a half-shipped bundle.
    """
    from pathlib import Path
    import tempfile

    from llmoji.analyze import _write_bundle

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        try:
            _write_bundle(
                bundle,
                counts_by_cell={
                    "Some-Model:Tag": {"(◕‿◕)": 1},
                    "some-model-tag": {"(◕‿◕)": 1},
                },
                synthesized_by_cell={
                    "Some-Model:Tag": {"(◕‿◕)": "a"},
                    "some-model-tag": {"(◕‿◕)": "b"},
                },
                providers_seen=[],
                model_counts={},
                submitter_id="0" * 32,
                synth_backend="anthropic",
                synth_model_id="claude-haiku-4-5-20251001",
                notes="",
            )
        except ValueError as e:
            assert "slug collision" in str(e), str(e)
        else:
            raise AssertionError(
                "_write_bundle didn't refuse slug collision",
            )


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

        # Codex: malformed hooks.json (mirrors claude_code; both providers
        # now route through the JSON helpers and Codex stores its hook
        # registrations at ~/.codex/hooks.json).
        cx = get_provider("codex")
        cx.hooks_dir = td / "codex" / "hooks"
        cx.settings_path = td / "codex" / "hooks.json"
        cx.journal_path = td / "codex" / "journal.jsonl"
        cx.settings_path.parent.mkdir(parents=True, exist_ok=True)
        original = "{ also not json"
        cx.settings_path.write_text(original)
        try:
            cx.install()
        except SettingsCorruptError:
            pass
        else:
            raise AssertionError("codex didn't refuse corrupt JSON")
        assert cx.settings_path.read_text() == original, (
            "corrupt config was modified"
        )


def test_mask_kaomoji_prepends_face_token():
    """By v1.0 journal contract every source — live hooks, the
    Claude.ai export reader, the generic-JSONL contract — strips
    the leading kaomoji from ``assistant_text`` before yielding a
    ScrapeRow; the prefix lives in the row's ``kaomoji`` field.
    ``mask_kaomoji`` therefore just prepends ``[FACE] `` so Haiku
    sees the ``[FACE] <body>`` shape the DESCRIBE prompts promise.
    """
    from llmoji.synth import MASK_TOKEN, mask_kaomoji

    # Stripped-body row (the only journal-row shape under the v1.0
    # contract): mask token gets prepended with a separating space.
    masked = mask_kaomoji("I think we should refactor.", "(◕‿◕)")
    assert masked.startswith(MASK_TOKEN + " "), masked
    assert "I think we should refactor." in masked

    # Leading whitespace on the body is normalized away — Haiku
    # shouldn't see ``[FACE]   body``.
    masked = mask_kaomoji("   leading spaces.", "(◕‿◕)")
    assert masked == MASK_TOKEN + " leading spaces."

    # Empty first_word — defensive pass-through.
    assert mask_kaomoji("hello", "") == "hello"


def test_hook_templates_render_to_valid_bash_substitutions():
    """The base renderer should ignore unknown $VARS in the template
    body (so embedded jq / sed `$KAOMOJI` etc. aren't eaten by the
    Python Template substitution). string.Template's safe_substitute
    handles this — verify. Also bash -n the rendered output so a
    template syntax error fails CI rather than failing silently
    inside the user's harness post-install.

    Same checks apply to the secondary nudge templates — providers
    that opt into a nudge get bash -n'd on the rendered output too,
    so a quoting regression on ``$NUDGE_MESSAGE_QUOTED`` fails CI
    rather than landing as a no-op in the user's harness.
    """
    import shutil
    import subprocess
    import tempfile
    from llmoji.providers import get_provider
    bash = shutil.which("bash")

    def _bash_n(label: str, rendered: str) -> None:
        if not bash:
            return
        with tempfile.NamedTemporaryFile(
            "w", suffix=".sh", delete=False,
        ) as f:
            f.write(rendered)
            tmp = f.name
        r = subprocess.run(
            [bash, "-n", tmp], capture_output=True, text=True,
        )
        assert r.returncode == 0, (
            f"{label} failed bash -n: {r.stderr}"
        )

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
        _bash_n(f"{name} main hook", rendered)

        if p.has_nudge:
            nudge = p.render_nudge_hook()
            for placeholder in (
                "${NUDGE_MESSAGE_QUOTED}", "${LLMOJI_VERSION}",
            ):
                assert placeholder not in nudge, (
                    f"{name} nudge kept literal {placeholder}"
                )
            # The shell-quoted nudge message should round-trip
            # through the bash literal back to itself when sourced.
            assert p.nudge_message in nudge or all(
                c not in p.nudge_message for c in "'"
            ), "nudge message lost in template rendering"
            _bash_n(f"{name} nudge hook", nudge)


def test_nudge_install_uninstall_roundtrip():
    """Nudge hooks install and uninstall cleanly alongside the main
    hook — no orphan files, no orphan settings entries. Idempotent
    re-install is a no-op (same as the main-hook contract).
    """
    from pathlib import Path
    import tempfile

    from llmoji.providers import get_provider

    for provider_name in ("claude_code", "codex"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = get_provider(provider_name)
            assert p.has_nudge, (
                f"{provider_name} should ship a nudge in v1.0"
            )
            p.hooks_dir = td / provider_name / "hooks"
            p.settings_path = td / provider_name / "settings.json"
            p.journal_path = td / provider_name / "journal.jsonl"

            p.install()
            nudge_path = p.nudge_hook_path
            assert nudge_path is not None
            assert p.hook_path.exists()
            assert nudge_path.exists()
            s = p.status()
            assert s.installed
            assert s.nudge_installed
            assert s.nudge_hook_path == nudge_path

            # Idempotent: install twice should yield the same file.
            settings_after_first = p.settings_path.read_text()
            p.install()
            assert p.settings_path.read_text() == settings_after_first

            p.uninstall()
            assert not p.hook_path.exists()
            assert not nudge_path.exists()
            s = p.status()
            assert not s.installed
            assert not s.nudge_installed
