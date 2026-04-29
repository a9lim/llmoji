# llmoji examples

Short worked examples for the paths that aren't covered by the CLI. The two harness plugins are TypeScript instead of something you `python` directly.

- **[`inspect_bundle.py`](inspect_bundle.py)**: load `~/.llmoji/bundle/manifest.json` plus each per-source-model `<slug>.jsonl` at the bundle root, and print a per-canonical-kaomoji preview of what would ship on `llmoji upload`.
- **[`openclaw_plugin/`](openclaw_plugin)**: a worked OpenClaw plugin (`index.ts` plus `openclaw.plugin.json`) built on the `definePluginEntry` API. 
- **[`opencode_plugin.ts`](opencode_plugin.ts)**: the same generic-JSONL contract for opencode (https://opencode.ai), whose plugin host auto-loads `.ts` and `.js` files from `~/.config/opencode/plugins/` or `.opencode/plugins/`.
- **[`_kaomoji_taxonomy.ts.partial`](_kaomoji_taxonomy.ts.partial)**: the canonical TypeScript port of `llmoji.taxonomy.is_kaomoji_candidate` and `_leading_bracket_span`, plus the `NUDGE` string.

If you write an adapter for another harness, please open a PR adding it here.
