# Contributing to llmoji

Thank you very much for wanting to contribute! I really appreciate any contribution you would like to make, whether it's a PR, a bug report, or hook validation. New first-class providers are especially welcome.

## Dev setup

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"
```

The dev extras pull in `pytest` and `ruff`. There is no GPU dependency or network requirement for tests.

## Running tests

```bash
pytest tests/                            # everything
pytest tests/test_canonicalize.py -v     # rule-by-rule taxonomy regression
pytest tests/test_public_surface.py -v   # cross-corpus invariant contract
pytest tests/test_install_autodetect.py  # install + uninstall no-arg dispatch
pytest tests/test_import.py              # import dedup + autodetect
pytest -m "not slow"                     # skip the bash + jq parity gate
```

The full suite runs anywhere with Python 3.11+ in under a few seconds, and is what CI exercises on every PR. The bash + jq parity tests in `tests/test_pipeline_parity.py` are marked `slow` because they fork subprocess for every case. Skip them with `-m "not slow"` while iterating on something unrelated; CI keeps running the full suite. Tests use `$LLMOJI_HOME` to override the on-disk root, so they don't touch your real `~/.llmoji`.

## Lint

CI runs `ruff check .` on every PR. Please run it locally first:

```bash
ruff check .
ruff check . --fix      # auto-fix what's fixable
```

## Adding a new provider

llmoji ships two kinds of installers.

### Bash hook providers

For harnesses that expose shell hooks (Claude Code, Codex, Hermes), a first-class provider is one bash hook template under `llmoji/_hooks/` plus one `HookInstaller` subclass under `llmoji/providers/`. The abstraction in `llmoji/providers/base.py:HookInstaller` documents what the subclass needs. Please keep the following in mind:

1. Where the harness keeps its hooks dir and settings file.
2. The harness's stop-event payload shape (kaomoji on first or last text block per turn, or single-text-field).
3. How to filter sidechain dispatches (none, field-flag, or session-id correlation).

If the harness's settings format isn't already in `base.py` (we have JSON for Claude Code and Codex, YAML for Hermes), please add a new format alongside. The settings writer must go through `atomic_write_text`. Additionally wire up the nudge: set `nudge_hook_template`, `nudge_hook_filename`, `nudge_event`, and `nudge_message`, then override `_check_registrations` if the format isn't JSON-shaped.

Please include in the PR:

- The hook template (`llmoji/_hooks/<provider>.sh.tmpl`), plus a nudge template if the response shape differs from the existing `claude_codex_nudge.sh.tmpl` or `hermes_nudge.sh.tmpl`.
- The `HookInstaller` subclass and its `system_injected_prefixes` list (empty if the harness doesn't inject system-role-as-user-text payloads).
- Test cases in `test_public_surface.py` for the rendered hook plus any new corruption-refusal path. The existing `test_nudge_install_uninstall_roundtrip` picks up a new JSON-shaped nudge automatically; YAML-shaped providers want their own coverage.
- A short note on the harness's docs version and where the kaomoji lands in the stop payload.

### TS plugin providers

For harnesses with a TypeScript-only plugin SDK (opencode, openclaw), a first-class provider is one or more TS templates under `llmoji/_plugins/` plus one `PluginInstaller` subclass under `llmoji/providers/`. Both shipped TS providers are reasonable starting points to copy from. Each template needs the following:

1. A `// BEGIN SHARED TAXONOMY` / `// END SHARED TAXONOMY` marker pair somewhere in the file. The body between them is left empty in the template. `render_plugin_template` splices in the canonical TS port from `_kaomoji_taxonomy.ts.partial` at install time. Don't paste the validator inline.
2. A `__LLMOJI_VERSION__` token wherever you want the package version stamped.
3. The actual harness-side hook registration: a per-turn nudge injection and a per-message journal write (one row per kaomoji-led assistant message, written to `~/.llmoji/journals/<harness>.jsonl` against the canonical schema).

The `PluginInstaller` subclass needs:

- `plugin_dir` — where on disk the rendered files land (typically inside `~/.<harness>/plugins/` or similar).
- `plugin_files` — list of `(template_name, dest_filename)` tuples; the first entry is the "main artifact" whose path is reused as `hook_path` for status reporting.
- `journal_path` — `~/.llmoji/journals/<harness>.jsonl` by convention.
- `is_present` — by default returns `plugin_dir.parent.exists()`; override if the harness's home dir lives elsewhere (openclaw points at `settings_path.parent` so the parent-dir check matches the bash providers).
- `_register` / `_unregister` / `_check_registrations` only need overrides if the harness needs an explicit registration step beyond file presence (e.g. flipping a config flag; see `OpenclawProvider`). If file presence is enough, inherit the defaults.

Please include in the PR:

- The TS template(s) under `llmoji/_plugins/<provider>*.ts.tmpl` (and `.json.tmpl` if the harness needs a manifest).
- The `PluginInstaller` subclass under `llmoji/providers/<provider>.py`.
- An entry in `llmoji.providers.PROVIDERS` and the `__all__` re-export.
- A round-trip install/uninstall test in `test_public_surface.py` (the existing `test_plugin_install_uninstall_roundtrip` is a template).

The journal schema does not change for a new provider, of either flavor.

## Adding a static-dump source

Static-dump readers live under `llmoji/sources/`. They convert a specific provider's export format into the canonical six-field schema. The reader is plain Python and does not interact with the harnesses.

Please strip the leading kaomoji from `assistant_text` on parse, the same way the bash hooks do (the prefix lives in the row's `kaomoji` field).

## PRs

- Please don't bump the version in your PR unless the change is intended to ship as a release. The PyPI publish workflow is triggered by a version update.
- Anything that touches `llmoji.taxonomy`, `llmoji.synth_prompts`, `HookInstaller`, the journal schema, or the bundle schema is a cross-corpus invariant change. Please flag it explicitly in the PR body. Those affect the HF dataset and need an edit on the dataset card too. The taxonomy is currently locked at the 2.0.0 sweep (broader `KAOMOJI_START_CHARS` plus the paired-arm/wing-hand/raised-hands/box-drawing-pose strip set); adding a leader char or arm modifier is itself a cross-corpus change and needs a major version bump.
- If you're porting the validator to another language for an unsupported harness, the canonical TS port at [`llmoji/_plugins/_kaomoji_taxonomy.ts.partial`](llmoji/_plugins/_kaomoji_taxonomy.ts.partial) is the reference shape. The Python module `llmoji.taxonomy` is the actual single source of truth — bash hooks render the leading-glyph case from `KAOMOJI_START_CHARS` at install time, and the TS partial gets spliced into both shipped TS plugins by `render_plugin_template`. A `test_plugin_taxonomy_block_matches` test asserts the splice is byte-identical, so a stale hand-edit on either side fails CI.
- The hermes provider in particular wants real-traffic validation. If you run hermes and are willing to share what `extra.*` keys actually arrive on `post_llm_call` and whether `subagent_stop` correlation filters cleanly, please open an issue.

## Questions

Please reach out to me or open an issue. For anything security-sensitive or privacy-sensitive, please see [SECURITY.md](SECURITY.md).
