# llmoji

[![CI](https://github.com/a9lim/llmoji/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/llmoji/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmoji)](https://pypi.org/project/llmoji/)
[![Downloads](https://img.shields.io/pypi/dm/llmoji)](https://pypi.org/project/llmoji/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/llmoji/)

Llmoji is a small CLI for collecting kaomoji journals from coding agents (Claude Code, Codex, Hermes), distilling them into per-canonical-form Haiku descriptions, and submitting privacy-preserving aggregates to a shared research corpus. If your agent is configured to start each message with a kaomoji that reflects how it's currently feeling, llmoji captures that prefix on every assistant turn, canonicalizes it to a frozen v1.0 equivalence-class scheme, and asks Haiku to write a one-sentence overall meaning per canonical face. Only the per-face synthesized line ever leaves your machine, and only when you explicitly run `llmoji upload`.

Llmoji is the data-layer-only end-user side of a two-repo split. The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is where all probe, hidden-state, MiniLM-embedding, axis-projection, and figure work lives. This package has zero dependency on torch, sentence-transformers, saklas, or matplotlib. Runtime deps are `anthropic` (for Haiku) and `huggingface_hub` (for the upload target).

Three commands carry the day-to-day flow:

- **`llmoji install <provider>`**: write a stop-event hook into your harness, idempotently
- **`llmoji analyze`**: scrape, canonicalize, and Haiku-synthesize into a local bundle
- **`llmoji upload --target {hf,email}`**: tarball the bundle and ship it

It runs on Python 3.11+. Analyze needs an Anthropic API key in `$ANTHROPIC_API_KEY`. Upload to HuggingFace needs `huggingface-cli login` (or `$HF_TOKEN`); the email target opens your system mail client and asks you to attach the tarball manually.

---

## What this is for

The shared HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) collects per-canonical-kaomoji counts and a single Haiku-paraphrased one-sentence meaning per face, across many users' coding agents. The research-side companion repo embeds those descriptions, projects them onto an emotion axis, clusters by face, and produces figures. The aggregation rules pin "v1 corpus only", so the v1.0 frozen public surface is the contract that lets a user's bundle from one machine compose with everyone else's. Bumping any item in that surface is a major version bump.

The bundle landing on disk between `analyze` and `upload` is the deliberate inspection gap. The user `cat`s `descriptions.jsonl` before deciding to ship.

---

## Quick start

```bash
pip install llmoji
llmoji install claude_code      # or: codex, hermes
```

That writes a `Stop` hook into `~/.claude/settings.json` and a hook script under `~/.claude/llmoji-hooks/`. From now on, every kaomoji-bearing assistant turn appends one row to `~/.claude/kaomoji-journal.jsonl`. The hook is read-only: it never blocks or modifies your turn.

After some traffic has accumulated:

```bash
export ANTHROPIC_API_KEY=...
llmoji status                              # check what's been logged
llmoji analyze                             # scrape + canonicalize + Haiku synthesize
cat ~/.llmoji/bundle/descriptions.jsonl    # review what would ship
llmoji upload --target hf                  # commit to a9lim/llmoji
# or:
llmoji upload --target email               # opens mailto: with attach hint
```

`analyze` caches per-instance Haiku descriptions at `~/.llmoji/cache/per_instance.jsonl` keyed by content-hash, so re-runs only pay for new rows. `llmoji cache clear` wipes it.

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependency footprint is two packages: `anthropic` and `huggingface_hub`. Hooks run in `bash` and need `jq`, both of which ship on macOS and most Linux distros.

From source:

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"      # adds pytest + ruff
```

---

## How it works

### Journal capture

Each first-class provider has a stop-event hook that fires once per assistant turn. The hook reads the harness's stop payload from stdin, extracts the assistant's reply, validates the leading glyph against `llmoji.taxonomy.KAOMOJI_START_CHARS`, strips the kaomoji from the body, and appends one JSONL row to `~/.<harness>/kaomoji-journal.jsonl`. The schema is the same six fields across every provider:

```json
{"ts": "...", "model": "...", "cwd": "...", "kaomoji": "(◕‿◕)", "user_text": "...", "assistant_text": "..."}
```

`assistant_text` always has the leading kaomoji already stripped. The prefix lives separately in the `kaomoji` field. This contract is enforced on every write path (bash hooks via jq, the Claude.ai export reader in Python, and the generic-JSONL contract for custom harnesses).

The bash hook templates live as data files under `llmoji/_hooks/` and are rendered at install time from a single Python source for the start-char set. Templates are validated by `bash -n` in CI, so a template-edit regression fails the build instead of failing silently inside a user's harness post-install.

### Two-stage Haiku pipeline

`llmoji analyze` walks every installed provider's journal plus any extra JSONL files under `~/.llmoji/journals/`, then runs a two-stage pipeline:

- **Stage A (per instance)**: for each (kaomoji, user, assistant) row sampled (cap 4 per canonical face, deterministic seed), mask the kaomoji to `[FACE]` and call Haiku with `DESCRIBE_PROMPT_WITH_USER` or `DESCRIBE_PROMPT_NO_USER`. Cache keyed by `sha256(canonical + user + assistant)[:16]` at `~/.llmoji/cache/per_instance.jsonl`. Stage-A calls run on a small thread pool (default 4 workers, `$LLMOJI_CONCURRENCY` to override). Cache appends serialize on the main thread via `as_completed`, so no append interleaving. Set `$LLMOJI_CONCURRENCY=1` to force serial dispatch when debugging.
- **Stage B (per canonical face)**: pool Stage A descriptions for one canonical kaomoji form, synthesize a single one-or-two-sentence overall meaning via `SYNTHESIZE_PROMPT`. The synthesized line is the only thing that ships in the bundle.

Embedding, axis projection, clustering, and figures are not in this package. They happen on the receiving research side, applied to either the central HF corpus or a single user's submitted bundle.

### Bundle and inspection gap

`analyze` writes to `~/.llmoji/bundle/`:

- **`manifest.json`**: package version, journal counts per provider, canonical-kaomoji counts, the Haiku model id used, your free-form `--notes`, and the salted submitter id (the same one `upload` uses).
- **`descriptions.jsonl`**: one row per canonical kaomoji, with the synthesized meaning.

The bundle is loose files, not an opaque archive. `llmoji status` shows what's there, and `cat ~/.llmoji/bundle/descriptions.jsonl` is the audit. `upload` re-prompts before committing.

`upload.tar_bundle()` uses a strict allowlist (`manifest.json`, `descriptions.jsonl`). Any other file in the bundle dir makes it raise. `analyze` clears loose files in the bundle dir before writing, so a clean run produces exactly the two-file schema.

---

## What does and does not leave your machine

| Tier                                  | Where                                | Shipped on `upload`? |
|---------------------------------------|--------------------------------------|----------------------|
| Live journal (raw user and assistant text) | `~/.<harness>/kaomoji-journal.jsonl` | Never                |
| Per-instance Haiku paraphrase         | `~/.llmoji/cache/per_instance.jsonl` | Never                |
| Bundle (counts plus per-face synthesis) | `~/.llmoji/bundle/`                | Yes                  |

The bundle ships per-canonical-kaomoji counts plus the synthesized description (one line per face) plus a salted-hash submitter id. It does not ship raw `user_text`, raw `assistant_text`, per-instance descriptions, MiniLM embeddings, or per-axis projections. Those are research-side, applied to the bundle on the receiving end.

For frequent kaomoji, the synthesis abstracts over many contexts. For singletons, the synthesized line is effectively a paraphrase of one user turn. `analyze` prints a per-face preview and `upload` re-prompts so you can review before shipping. We deliberately don't impose a count floor; filtering is an analysis-time concern on the receiving end, and shipping the raw distribution preserves more vocabulary. Please see [SECURITY.md](SECURITY.md) for the full privacy threat model.

---

## Providers

`llmoji install <provider>` writes the hook script and registers it with the harness's settings file, idempotently.

| Provider      | Hook event              | Settings format | Notes                                                        |
|---------------|-------------------------|-----------------|--------------------------------------------------------------|
| `claude_code` | Stop                    | JSON            | Stable, in daily use.                                        |
| `codex`       | Stop                    | TOML            | Stable, in daily use.                                        |
| `hermes`      | post_llm_call plus subagent_stop | YAML   | Implemented from docs; please see hermes notes below.        |

Subagent and sidechain dispatches are filtered per provider:

- `claude_code`: `isSidechain` field flag on the stop payload. Hooks drop the row.
- `codex`: no subagent concept (`collaboration_mode` is `"default"` for every observed turn_context).
- `hermes`: session-id correlation against `subagent_stop`. The companion hook records each completed child's `session_id` and the main hook drops matching ids.

`install` refuses to clobber existing config. Three corruption paths are explicitly defended:

1. Malformed JSON in `~/.claude/settings.json` raises `SettingsCorruptError`. Please fix the file by hand before re-running install.
2. An existing unmanaged `[hooks.stop]` section in `~/.codex/config.toml` (outside our marker fence) is detected and refused. Move it aside or merge by hand.
3. An existing top-level `hooks:` key in `~/.hermes/config.yaml` (outside our marker fence) is detected and refused.

In all three cases you get a `SettingsCorruptError` with a specific path and reason, and re-running install after editing is fully idempotent. Settings writes go through an atomic tmp-file plus `os.replace`, so a power loss or `SIGINT` mid-write leaves your settings file with either the old content or the new, never half.

`llmoji uninstall <provider>` removes the hook and the settings entry, idempotently. Journals and the per-instance cache are preserved (you may re-install). `llmoji cache clear` is the explicit wipe.

### Hermes notes

The hermes provider installs two hooks:

- `~/.hermes/agent-hooks/post-llm-call.sh`: main journal logger.
- `~/.hermes/agent-hooks/subagent-stop.sh`: companion that records delegated child `session_id`s to `~/.hermes/.llmoji-children` so the main hook can drop them.

Both are registered together inside a marker-fenced `hooks:` block in `~/.hermes/config.yaml`, so re-running install is idempotent and uninstall removes the stanza cleanly. This is built from the documented hermes-agent v0.11.0 hook contract. The journal schema and CLI surface are stable, but three items still want real-traffic verification before claiming stability:

1. The exact `extra.*` keys delivered by `post_llm_call` (the docs example block was for `pre_tool_call`).
2. That session-correlation against `subagent_stop` actually filters child sessions cleanly under real `delegate_task` traffic.
3. That `extra.user_message` arrives clean (no system-injection prefixes that need filtering).

Treat the hermes hooks as docs-confirmed-but-untested until that validation lands. If you're a hermes user willing to share traffic, please open an issue.

---

## Static dumps

To pull historical kaomoji turns out of a Claude.ai data export:

```bash
llmoji parse --provider claude.ai ~/Downloads/data-...-batch-0000
```

The export's `conversations.json` is parsed and rows land at `~/.llmoji/journals/claude_ai_export.jsonl`. `llmoji analyze` picks this up alongside the live provider journals.

For Claude Code or Codex history that predates installing the live hook, the historical transcripts (`~/.claude/projects/**/*.jsonl`, `~/.codex/sessions/**/rollout-*.jsonl`) can be replayed into the journals via the `llmoji.backfill` module. Please see [`llmoji-study/scripts/21_backfill_journals.py`](https://github.com/a9lim/llmoji-study/blob/main/scripts/21_backfill_journals.py) for the worked pattern.

---

## Custom harness: generic JSONL contract

For harnesses we don't ship a first-class adapter for (notably OpenClaw, whose hook contract is TS-shaped and takes the payload as a function argument rather than stdin), there's a published append contract:

- Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Strip the leading kaomoji from `assistant_text` on the way in (the prefix lives in the `kaomoji` field).
- Validate the prefix the same way the package does: `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically. No first-class adapter required. Please see [`examples/openclaw_hook.ts`](examples/openclaw_hook.ts) for a worked OpenClaw example. OpenClaw first-class support is post-v1.0.

---

## v1.0 frozen public surface

These are stable across the v1 series. The HF dataset's aggregation rules declare "v1 corpus only", so bumping any of them is a major version bump (`llmoji` 2.0.0):

- `llmoji.taxonomy.KAOMOJI_START_CHARS` (the leading-glyph filter set)
- `llmoji.taxonomy.canonicalize_kaomoji` (rules A through P)
- `llmoji.taxonomy.is_kaomoji_candidate` validator contract
- `llmoji.taxonomy.extract` and `KaomojiMatch` (span-only; no affect labels)
- `llmoji.haiku_prompts.DESCRIBE_PROMPT_WITH_USER`, `DESCRIBE_PROMPT_NO_USER`, `SYNTHESIZE_PROMPT`
- `llmoji.haiku_prompts.HAIKU_MODEL_ID` (the locked Haiku checkpoint)
- The six-field unified journal row schema
- Per-provider system-injection prefix lists
- The `llmoji.providers.Provider` interface
- The bundle schema (`manifest.json`, `descriptions.jsonl`)
- `llmoji.scrape.ScrapeRow`

Free to change without bumping the major version: cache key derivation, `INSTANCE_SAMPLE_CAP` and `INSTANCE_SAMPLE_SEED`, internal helper names beyond what's listed above, anything in `llmoji-study` (research-side).

---

## Tests

```bash
pytest tests/                      # everything
pytest tests/test_canonicalize.py  # rule-by-rule regression for canonicalize_kaomoji + extract
pytest tests/test_public_surface.py  # locks the v1.0 contract
```

The full suite runs anywhere with no GPU and no network. CI runs `ruff check .` and `pytest` on every PR.

The public-surface test exercises taxonomy invariants, haiku-prompt content checks, provider rendering plus `bash -n` validation of every hook template, the bundle allowlist, the corrupt-config refusal paths, and the unified `mask_kaomoji` prepend contract. The canonicalize tests run rule-by-rule, with each rule case its own pytest line (~70 cases total), so a regression points you at the exact rule that broke.

---

## Contributing and security

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security and privacy, please see [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).

If you use llmoji or the central corpus in published research, please cite the [`llmoji-study`](https://github.com/a9lim/llmoji-study) writeup (citation key on the repo's README once published).
