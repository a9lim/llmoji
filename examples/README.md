# llmoji examples

Short worked examples for paths that aren't covered by the CLI.

- **[`inspect_bundle.py`](inspect_bundle.py)**: load `~/.llmoji/bundle/manifest.json` plus each per-source-model `<slug>.jsonl` at the bundle root, and print a per-canonical-kaomoji preview of what would ship on `llmoji upload`.

The two TS-plugin adapters (opencode, openclaw) that used to live here as standalone files were promoted to first-class providers in 1.3 — `llmoji install opencode` / `llmoji install openclaw` now renders and writes the plugin files for you. The canonical TypeScript port of the kaomoji validator lives at [`llmoji/_plugins/_kaomoji_taxonomy.ts.partial`](../llmoji/_plugins/_kaomoji_taxonomy.ts.partial) and is spliced into both rendered plugins at install time. The validator is locked at the 2.0.0 taxonomy sweep — see [`llmoji/taxonomy.py`](../llmoji/taxonomy.py) for the full leading-glyph and arm-strip sets.

If you write an adapter for another harness, please open a PR adding the provider under [`llmoji/providers/`](../llmoji/providers/) (bash hook) or with a TS template under [`llmoji/_plugins/`](../llmoji/_plugins/) (plugin host) — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the checklist.
