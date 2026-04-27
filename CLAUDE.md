# CLAUDE.md

## What this is

`llmoji` is a small provider-agnostic CLI for collecting kaomoji
journals from coding agents (Claude Code, Codex, Hermes), distilling
them into per-canonical-form Haiku descriptions, and submitting
privacy-preserving aggregates to a shared HF dataset for cross-corpus
research. Companion to the research-side
[`llmoji-study`](https://github.com/a9lim/llmoji-study) repo, where
all probe / hidden-state / MiniLM-embedding / axis-projection /
figure work lives.

This package is the data-layer-only end-user side: zero dependency on
saklas, torch, sentence-transformers, or matplotlib. Runtime deps are
**`anthropic`** (for Haiku) and **`huggingface_hub`** (for the upload
target). Everything else — embedding, eriskii axis projection,
clustering, figures — is research-side and lives in `llmoji-study`,
which `pip install llmoji>=1.0,<2` and reads our public surface.

## Architecture overview

End-user pipeline:

```
     ┌─────────────────┐  hook            ┌─────────────────┐
     │  user's harness │ ──────────────▶  │ ~/.<harness>/   │
     │ (claude-code,   │  6-field JSONL   │  kaomoji-       │
     │  codex, hermes) │  per kaomoji turn│  journal.jsonl  │
     └─────────────────┘                  └────────┬────────┘
                                                   │
                                                   ▼
            ┌──────────────┐  canonicalize   ┌─────────────────┐
            │ llmoji.      │ ◀────────────── │ llmoji.sources. │
            │ taxonomy     │   per row       │ journal /       │
            └──────┬───────┘                 │ claude_export   │
                   │                         └─────────────────┘
                   ▼
           ┌──────────────────┐                ┌──────────────┐
           │ llmoji.haiku     │ ─per-instance▶│ ~/.llmoji/   │
           │ DESCRIBE_PROMPT_*│  cache key    │ cache/per_   │
           └──────┬───────────┘  by content   │ instance.    │
                  │              hash         │ jsonl        │
                  ▼                           └──────────────┘
          ┌────────────────┐
          │ llmoji.haiku   │     pool by canonical
          │ SYNTHESIZE_    │ ◀── kaomoji, one
          │ PROMPT         │     synthesis per face
          └──────┬─────────┘
                 ▼
          ┌──────────────────┐
          │ ~/.llmoji/       │  manifest.json + descriptions.jsonl
          │ bundle/          │  (loose files for user inspection)
          └──────┬───────────┘
                 │  llmoji upload --target {hf,email}
                 ▼
          ┌──────────────────┐
          │ a9lim/llmoji HF  │
          │ dataset OR email │
          └──────────────────┘
```

The bundle landing on disk between `analyze` and `upload` is the
deliberate inspection gap — the user `cat`s `descriptions.jsonl`
before deciding to ship.

### Provider abstraction

`llmoji.providers.Provider` is a base class one subclass per
first-class harness. Each provider knows:

- where the harness keeps its hooks dir + settings file
- where the journal lives
- the harness's stop-event payload shape (kaomoji on first / last /
  single-text-field per turn)
- how to filter sidechain dispatches (none / field flag /
  session-id correlation)
- which prefixes mark a system-injected user-role payload

The bash hook templates live as data files under
`llmoji/_hooks/`. `Provider.render_hook()` reads the template and
substitutes `${KAOMOJI_START_CHARS}`, `${INJECTED_PREFIXES_FILTER}`,
`${JOURNAL_PATH}`, `${LLMOJI_VERSION}`. Single source of truth in
Python (the start-char set lives in `llmoji.taxonomy`,
system-injection prefixes on the Provider class).

### Two-stage Haiku pipeline

- Stage A (per instance): for each (kaomoji, user, assistant) row
  sampled (cap 4 per canonical face, deterministic seed), mask the
  kaomoji to `[FACE]` and call Haiku with
  `DESCRIBE_PROMPT_WITH_USER` / `DESCRIBE_PROMPT_NO_USER`. Cache
  keyed by `sha256(canonical + "\0" + user + "\0" + assistant)[:16]`
  at `~/.llmoji/cache/per_instance.jsonl`. Re-runs of `analyze`
  skip rows already described.

- Stage B (per canonical face): pool Stage A descriptions for one
  canonical kaomoji form, synthesize a single 1-2-sentence overall
  meaning via `SYNTHESIZE_PROMPT`. The synthesized line is the
  **only** thing that ships in the bundle.

Embedding / axis projection / clustering / figures are NOT in this
package — they happen on the receiving research side, against either
our own corpus or submitted user bundles.

## v1.0 frozen public surface

These are stable across the v1 series. The HF dataset's aggregation
rules declare "v1 corpus only"; bumping any of them is a major
version bump (`llmoji` 2.0.0):

- **`llmoji.taxonomy`**:
  - `KAOMOJI_TAXONOMY` (gemma-tuned label dict, used by `extract`)
  - `KAOMOJI_START_CHARS` (leading-glyph filter set; rules A–P
    canonicalization rules in `canonicalize_kaomoji`)
  - `is_kaomoji_candidate` validator contract
  - `extract` / `KaomojiMatch`
- **`llmoji.haiku_prompts`**:
  - `DESCRIBE_PROMPT_WITH_USER`, `DESCRIBE_PROMPT_NO_USER`
  - `SYNTHESIZE_PROMPT`
  - `HAIKU_MODEL_ID` (the locked Haiku checkpoint)
- **`llmoji.scrape.ScrapeRow`** schema (the in-memory row shape)
- **The 6-field unified journal row schema** (on-disk JSONL):
  `{ts, model, cwd, kaomoji, user_text, assistant_text}`
- **System-injection prefix lists** per provider (in
  `llmoji.providers.{claude_code,codex,hermes}`)
- **`llmoji.providers.Provider`** interface
- **Bundle schema**: `manifest.json` + `descriptions.jsonl` shape

Free to change without bumping the public API: cache key
derivation, `INSTANCE_SAMPLE_CAP` / `INSTANCE_SAMPLE_SEED`, internal
flag names beyond `--target {hf,email}`, etc.

## Commands

```
llmoji install <provider>      write hook + register; idempotent
llmoji uninstall <provider>    inverse; idempotent (journal preserved)
llmoji status                  installed providers, journal sizes, paths
llmoji parse --provider <n> P  ingest a static export dump (e.g.
                               claude.ai conversations.json) into
                               ~/.llmoji/journals/
llmoji analyze [--notes …]     scrape + canonicalize + Haiku
                               synthesize → ~/.llmoji/bundle/
llmoji upload --target {hf,email} [--yes]  tarball + ship
llmoji cache clear             wipe ~/.llmoji/cache/
```

`upload` requires `--target` (no default) and re-prompts before
committing. `--yes` skips the prompt for scripted use.

## Layout

```
llmoji/
  pyproject.toml
  README.md
  LICENSE
  CLAUDE.md
  llmoji/
    __init__.py            # public surface re-exports
    taxonomy.py            # KAOMOJI_TAXONOMY + KAOMOJI_START_CHARS
                           # + is_kaomoji_candidate + extract +
                           # canonicalize_kaomoji (frozen v1.0)
    haiku_prompts.py       # DESCRIBE_PROMPT_* + SYNTHESIZE_PROMPT
                           # + HAIKU_MODEL_ID (frozen v1.0)
    haiku.py               # mask_kaomoji + call_haiku + cache
    scrape.py              # ScrapeRow + iter_all chain helper
    sources/
      journal.py           # generic kaomoji-journal reader (any
                           # provider's ~/.<harness>/kaomoji-journal.jsonl)
      claude_export.py     # Claude.ai conversations.json reader
    backfill.py            # one-shot transcript→journal replays
                           # for Claude Code + Codex
    providers/
      base.py              # Provider base class + dataclasses +
                           # template render helpers
      claude_code.py       # JSON-shaped Stop-hook provider
      codex.py             # TOML-shaped Stop-hook provider
      hermes.py            # YAML-shaped post_llm_call provider
                           # (impl per docs; needs live-traffic
                           # validation)
    _hooks/                # bash hook templates (importlib.resources
                           # data); rendered at install time
      claude_code.sh.tmpl
      codex.sh.tmpl
      hermes.sh.tmpl
    paths.py               # ~/.llmoji home, cache, bundle, journals
    analyze.py             # the analyze pipeline (Stage A + B + bundle)
    upload.py              # tar + HF / email targets
    cli.py                 # argparse entry, [project.scripts] llmoji
  tests/                   # (place for unit tests; not present at v1.0)
```

## Gotchas

### KAOMOJI_START_CHARS sync — RESOLVED via templating

Pre-package, the start-char set lived in five places (Python
validators in this package, the equivalent in `llmoji_study`, and
inline `case` patterns in two hand-written shell hooks). Now:

- Python single source: `llmoji.taxonomy.KAOMOJI_START_CHARS`
- Shell hooks: rendered at `install` time from the `.sh.tmpl` files
  with `${KAOMOJI_START_CASE}` substituted from the Python set.
- `is_kaomoji_candidate` validates Python-side; the rendered case
  filter handles the shell-side first pass.

If you find another copy of the set, delete it and route through
`llmoji.taxonomy`.

### Per-provider kaomoji position

- **Claude Code**: kaomoji on the **first** text block of an
  assistant content array. Later text is post-tool-call continuation,
  irrelevant. The hook reads `[0].text`.
- **Codex**: kaomoji on the **last** agent message of a turn. Each
  agent message is its own `event_msg.agent_message` event;
  progress messages come first. The hook keys on
  `task_complete.last_agent_message`.
- **Hermes**: **single** final-text field per turn. No first/last
  ambiguity.

Flipping any of these would miss every multi-step turn's kaomoji.

### Sidechain strategy

- Claude Code: `field_flag` on `isSidechain`. Hooks drop the row.
- Codex: no subagent concept. `collaboration_mode` is `"default"`
  for every observed turn_context.
- Hermes: `session_correlation`. `post_llm_call` fires for both
  parent and child sessions; child sessions are identified by
  correlating `session_id` against `delegate_task` events from a
  companion `subagent_stop` registration. The current shell hook
  reads a state file at `~/.hermes/.llmoji-children`; populating
  that file is part of the hermes empirical-validation work.

### Hermes is "implemented from docs", not battle-tested

The hermes provider (template, payload extraction, sidechain
correlation strategy) is built from the documented hermes-agent
v0.11.0 contract. Three items still want real-traffic verification
before claiming stability:

1. The exact `extra.*` keys delivered by `post_llm_call` (the docs
   example block was for `pre_tool_call`).
2. That session-correlation actually filters child sessions cleanly.
3. That `extra.user_message` arrives clean — no system-injection
   prefixes that need filtering. The Provider's
   `system_injected_prefixes` is `[]` per the docs; if real traffic
   shows otherwise, populate the list and re-render.

Treat the hermes hook as experimental until that validation lands;
the journal schema and CLI surface are stable across the
verification.

### Cache directory is leakier than the bundle

`~/.llmoji/cache/per_instance.jsonl` holds Haiku-paraphrased
descriptions of single user turns. Each row IS one user-turn
paraphrase, so for a topic-narrow corpus a singleton row can leak
specifics of that turn through. Mitigations:

- Cache is **never** bundled or shipped. Only the per-canonical-face
  synthesis (Stage B) lands in the bundle.
- `llmoji status` prints the cache size and entry count so the user
  is aware it exists.
- `llmoji uninstall <provider>` does NOT touch the cache (the user
  may re-install). `llmoji cache clear` is the explicit wipe.

The bundle is the only thing that leaves the machine, and the
inspection gap (`analyze` prints a per-face preview, `upload`
re-prompts) is the consent boundary.

### Codex puts the kaomoji on the LAST agent message, Claude on the FIRST

Stated again because the templates have to be exactly right on this:

- Claude Code's assistant message is one event with interleaved
  `text + tool_use + text` content blocks; the kaomoji-prefixed
  reply is always the FIRST text block.
- Codex emits each agent message as a separate
  `event_msg.agent_message` event; the kaomoji-bearing summary
  lands last as `task_complete.last_agent_message`.

The Codex hook + Codex backfill both key on `last_agent_message`,
NOT on the first agent_message — flipping that would miss every
multi-step turn's kaomoji.

### Codex `transcript_path` carries the rollout JSONL

We use it to resolve `user_text` (Codex injects AGENTS.md /
`<environment_context>` / `<INSTRUCTIONS>` as user-role response_items
at session start; we walk the rollout to find the latest real user
turn, dropping those prefixes defensively). The backfill module
in `llmoji.backfill` mirrors this.

### Generic JSONL contract for unsupported harnesses

Motivated users on harnesses we don't ship a first-class adapter for
(notably OpenClaw — TS-shaped hooks that take the payload as a
function argument, not stdin) can write directly to
`~/.llmoji/journals/<name>.jsonl` against the canonical 6-field
schema. `llmoji analyze` picks them up automatically alongside the
managed providers' journals. Document the contract in the README
with a worked OpenClaw example when one materializes; OpenClaw
first-class is post-v1.0.

## Conventions

- Single venv at `.venv/`, pip not uv. `pip install -e ../llmoji`
  during dev; PyPI install at freeze.
- `~/.llmoji` is the on-disk root for everything the package
  manages; tests can override via `$LLMOJI_HOME`.
- Hook templates are bash, syntactically validated by `bash -n`
  during dev. Don't introduce non-bash hook formats; if a harness
  needs one, that's a post-v1.0 first-class adapter, and the
  generic-JSONL-append contract is the v1.0 path until then.
- Public-API freeze: anything in §"v1.0 frozen public surface" gets
  a major-version bump if changed. Internal helpers (everything in
  `llmoji.providers.base._*`, `llmoji.haiku.cache_key`, etc.) are
  free to evolve.
