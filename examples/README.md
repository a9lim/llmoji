# llmoji examples

Short worked examples for the paths that aren't covered by the first-class CLI surface. The two harness plugins are illustrative TypeScript (you'd lift them into your harness's plugin config) rather than something you `python` directly.

- **[`inspect_bundle.py`](inspect_bundle.py)**: load `~/.llmoji/bundle/manifest.json` plus each per-source-model `<slug>.jsonl` at the bundle root, and print a per-canonical-kaomoji preview of what would ship on `llmoji upload`. Useful as the audit step before committing.
- **[`openclaw_plugin/`](openclaw_plugin)**: a worked OpenClaw plugin (`index.ts` plus `openclaw.plugin.json`) built on the `definePluginEntry` API. Shows the generic-JSONL-append contract on a TS-plugin host: one row per kaomoji-bearing assistant turn appended to `~/.llmoji/journals/openclaw.jsonl` against the canonical six-field schema, and `llmoji analyze` picks it up automatically alongside the managed-provider journals.
- **[`opencode_plugin.ts`](opencode_plugin.ts)**: the same generic-JSONL contract for opencode (https://opencode.ai), whose plugin host auto-loads `.ts` and `.js` files from `~/.config/opencode/plugins/` or `.opencode/plugins/`. Drop the file in either directory and you get the same per-turn capture, no shell hooks involved.
- **[`_kaomoji_taxonomy.ts.partial`](_kaomoji_taxonomy.ts.partial)**: the canonical TypeScript port of `llmoji.taxonomy.is_kaomoji_candidate` and `_leading_bracket_span`, plus the `NUDGE` string. Both plugins above include this block verbatim between `// BEGIN SHARED TAXONOMY` / `// END SHARED TAXONOMY` markers; `tests/test_public_surface.py::test_examples_taxonomy_partial_matches` asserts they stay byte-identical so the two TS ports can't drift from each other or from `taxonomy.py`. Not something you copy on its own — the plugins are still self-contained.

If you write an adapter for another harness, please open a PR adding it here.
