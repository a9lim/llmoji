# llmoji examples

Short worked examples for the paths that aren't covered by the first-class CLI surface. The two harness plugins are illustrative TypeScript (you'd lift them into your harness's plugin config) rather than something you `python` directly.

- **[`inspect_bundle.py`](inspect_bundle.py)**: load `~/.llmoji/bundle/manifest.json` plus each per-source-model `<slug>.jsonl` at the bundle root, and print a per-canonical-kaomoji preview of what would ship on `llmoji upload`. Useful as the audit step before committing.
- **[`openclaw_plugin/`](openclaw_plugin)**: a worked OpenClaw plugin (`index.ts` plus `openclaw.plugin.json`) built on the `definePluginEntry` API. Shows the generic-JSONL-append contract on a TS-plugin host: one row per kaomoji-bearing assistant turn appended to `~/.llmoji/journals/openclaw.jsonl` against the canonical six-field schema, and `llmoji analyze` picks it up automatically alongside the managed-provider journals.
- **[`opencode_plugin.ts`](opencode_plugin.ts)**: the same generic-JSONL contract for opencode (https://opencode.ai), whose plugin host auto-loads `.ts` and `.js` files from `~/.config/opencode/plugins/` or `.opencode/plugins/`. Drop the file in either directory and you get the same per-turn capture, no shell hooks involved.

If you write an adapter for another harness, please open a PR adding it here.
