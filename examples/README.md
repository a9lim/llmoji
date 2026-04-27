# llmoji examples

These are short worked examples for the two paths that aren't covered by the first-class CLI surface. Both are runnable, but the OpenClaw script is illustrative TypeScript (you'd lift it into your harness's hook config) rather than something you `python` directly.

- **[`inspect_bundle.py`](inspect_bundle.py)**: load `~/.llmoji/bundle/{manifest.json,descriptions.jsonl}` and print a per-canonical-kaomoji preview of what would ship on `llmoji upload`. Useful as the audit step before committing.
- **[`openclaw_hook.ts`](openclaw_hook.ts)**: a worked TypeScript example of the generic-JSONL-append contract for harnesses we don't ship a first-class adapter for. Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/openclaw.jsonl` against the canonical six-field schema, and `llmoji analyze` picks it up automatically.

If you write an adapter for another harness, please open a PR adding it here.
