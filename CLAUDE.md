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
substitutes `${KAOMOJI_START_CASE}`, `${INJECTED_PREFIXES_FILTER}`,
`${JOURNAL_PATH}`, `${LLMOJI_VERSION}` via `string.Template`. Single
source of truth in Python (the start-char set lives in
`llmoji.taxonomy`, system-injection prefixes on the Provider class).

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
  - `KAOMOJI_START_CHARS` (leading-glyph filter set)
  - rules A–P in `canonicalize_kaomoji`
  - `is_kaomoji_candidate` validator contract
  - `extract` / `KaomojiMatch` (span-only — no affect labels;
    gemma-tuned label dicts moved to research-side
    `llmoji_study.taxonomy_labels`)
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
    taxonomy.py            # KAOMOJI_START_CHARS + is_kaomoji_candidate
                           # + extract + KaomojiMatch (span-only)
                           # + canonicalize_kaomoji (rules A–P; frozen v1.0)
    haiku_prompts.py       # DESCRIBE_PROMPT_* + SYNTHESIZE_PROMPT
                           # + HAIKU_MODEL_ID (frozen v1.0)
    haiku.py               # mask_kaomoji + call_haiku +
                           # per-instance content-hash cache
    scrape.py              # ScrapeRow + iter_all chain helper
                           # (span-only schema; no affect labels)
    sources/
      journal.py           # generic kaomoji-journal reader (any
                           # provider's ~/.<harness>/kaomoji-journal.jsonl)
      claude_export.py     # Claude.ai conversations.json reader
    backfill.py            # one-shot transcript→journal replays
                           # for Claude Code + Codex
    providers/
      base.py              # Provider + ProviderStatus dataclasses +
                           # SettingsCorruptError + template render
                           # helpers + JSON-settings edit helpers
                           # + _atomic_write_text (tmp+rename) used
                           # by every provider's settings writer
      claude_code.py       # JSON Stop-hook provider (~/.claude/settings.json)
      codex.py             # TOML Stop-hook provider (~/.codex/config.toml,
                           # marker-fenced [hooks.stop] stanza)
      hermes.py            # YAML post_llm_call + subagent_stop dual-hook
                           # provider (~/.hermes/config.yaml,
                           # marker-fenced hooks: stanza)
    _hooks/                # bash hook templates (importlib.resources
                           # data); rendered at install time
      claude_code.sh.tmpl
      codex.sh.tmpl
      hermes.sh.tmpl                # post_llm_call (journal logger)
      hermes_subagent_stop.sh.tmpl  # subagent_stop (sidechain registrar)
    paths.py               # ~/.llmoji home, cache, bundle, journals
    analyze.py             # the analyze pipeline (Stage A + B + bundle;
                           # clears bundle dir before writing). Stage A
                           # dispatches cache-miss Haiku calls on a
                           # ThreadPoolExecutor (default 4 workers, env
                           # $LLMOJI_CONCURRENCY); cache appends serialize
                           # on the main thread via as_completed
    upload.py              # tar + HF / email targets;
                           # BUNDLE_ALLOWLIST enforces the two-file
                           # schema, refuses extras
    cli.py                 # argparse entry, [project.scripts] llmoji
  tests/
    test_public_surface.py # pytest checks locking the v1.0 contract
                           # (taxonomy / haiku_prompts / ScrapeRow /
                           # provider rendering + bash -n / bundle
                           # allowlist / corrupt-config refusal /
                           # mask_kaomoji pre-stripped branch).
    test_canonicalize.py   # parametrized rule-by-rule regression
                           # tests for canonicalize_kaomoji + extract
                           # + is_kaomoji_candidate. Each rule case is
                           # its own pytest line. ~70 cases total.
```

## Gotchas

### `mask_kaomoji` re-prepends `[FACE]` for journal-source rows

The bash hooks strip the leading kaomoji from `assistant_text`
before writing the journal row (the
`sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")` jq
chain). The `kaomoji` field carries the prefix separately. So at
`analyze` time:

  - **Live-hook journal rows**: `assistant_text` does NOT start
    with `first_word`. `mask_kaomoji` prepends `[FACE] ` so the
    Haiku DESCRIBE prompt's "we replaced it with [FACE]" framing
    matches what Haiku actually sees in the body.
  - **Static-export rows** (claude.ai conversations.json): the
    kaomoji is still at the head of `assistant_text`.
    `mask_kaomoji` substitutes `[FACE]` in place.

Either way the masked text starts with `[FACE]` whenever
`first_word` is non-empty. Pre-fix the live-hook branch fell
through to `return text` and Haiku got a prompt promising a
`[FACE]` that wasn't in the body — affected every journal row.

If you change the hook's strip-on-write behavior, audit
`mask_kaomoji` in the same diff. The cache key is on raw
`(canonical, user_text, assistant_text)` — not on the masked
output — so existing cache entries survive a fix to either side.

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
- Hermes: `session_correlation` against the `subagent_stop` event.
  `post_llm_call` fires for both parent and child sessions; the
  installed companion hook
  `~/.hermes/agent-hooks/subagent-stop.sh` records each completed
  child's `session_id` to `~/.hermes/.llmoji-children`. The main
  hook checks that file and drops matching session_ids. Both hooks
  are registered together in the YAML stanza `llmoji install
  hermes` writes; uninstall removes both.

### Provider install refuses to clobber existing config

Three corruption paths are explicitly defended:

1. **Malformed JSON in `~/.claude/settings.json`** — pre-fix the
   loader returned `None` and `install` would silently treat it as
   `{}` and rewrite. Post-fix `_load_json_strict` raises
   `SettingsCorruptError` and the user has to fix the file by hand
   before `install` will touch it.
2. **Existing `[hooks.stop]` section in `~/.codex/config.toml`** —
   appending a fresh `[hooks.stop]` block would yield invalid
   duplicate-table TOML. `CodexProvider._has_unmanaged_hooks_stop`
   detects an unmanaged stanza outside our marker fence and refuses
   to install.
3. **Existing top-level `hooks:` key in `~/.hermes/config.yaml`** —
   appending another `hooks:` makes a duplicate-key YAML doc that
   silently last-write-wins. `HermesProvider._has_unmanaged_hooks_top_level`
   refuses to install.

In all three cases the user gets a `SettingsCorruptError` with a
specific path and reason. They edit the file (move-aside or merge
by hand) and re-run.

The non-managed analogue — a user re-running `install` after
already installing once — is fully idempotent. The marker fences in
codex/hermes mean the second `install` is a no-op; the JSON-edit
path in claude_code checks for an existing entry with our command
string and skips.

Settings writes go through `_atomic_write_text` (tmp file +
`os.replace`) so a power loss / SIGINT mid-write leaves the user's
settings file with either the old content or the new — never half.
The `upload` state.json (per-machine submission token) writes the
same way.

### Bundle is allowlisted, not just-tar-everything

`upload.tar_bundle()` only ships files in `BUNDLE_ALLOWLIST`
(`manifest.json`, `descriptions.jsonl` — the v1.0 frozen schema).
Any other file in `~/.llmoji/bundle/` makes `tar_bundle` raise
`FileExistsError`. `analyze` clears loose files in the bundle dir
before writing, so a clean run produces exactly the two-file
schema. The two together mean: stale per-instance descriptions,
user-added notes, hidden state caches, etc. cannot accidentally
leak through `upload`.

### Hermes provider — two-hook design from docs

The hermes provider installs **two** hooks, both wired through the
same shell-hooks mechanism:

- `~/.hermes/agent-hooks/post-llm-call.sh` — main journal logger.
- `~/.hermes/agent-hooks/subagent-stop.sh` — companion that records
  delegated child session_ids to `~/.hermes/.llmoji-children` so
  the main hook can drop them.

Both registered in `~/.hermes/config.yaml` under the `hooks:` block,
inside our managed marker fence so re-running install is idempotent
and uninstall removes the stanza cleanly.

This is built from the documented hermes-agent v0.11.0 hook
contract (the [Event Hooks docs][hermes-hooks]). Three items still
want real-traffic verification before claiming stability:

1. The exact `extra.*` keys delivered by `post_llm_call` (the docs
   example block was for `pre_tool_call`).
2. That session-correlation against `subagent_stop` actually filters
   child sessions cleanly under real `delegate_task` traffic.
3. That `extra.user_message` arrives clean — no system-injection
   prefixes that need filtering. The Provider's
   `system_injected_prefixes` is `[]` per the docs; if real traffic
   shows otherwise, populate the list and re-render.

Treat the hermes hooks as docs-confirmed-but-untested until that
validation lands; the journal schema and CLI surface are stable
across the verification.

[hermes-hooks]: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/

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
  in the test suite (`test_hook_templates_render_to_valid_bash_substitutions`)
  so a template-edit regression fails CI rather than failing
  silently inside a user's harness post-install. Don't introduce
  non-bash hook formats; if a harness needs one, that's a post-v1.0
  first-class adapter, and the generic-JSONL-append contract is the
  v1.0 path until then.
- Stage-A Haiku calls run on a small thread pool (default 4,
  `$LLMOJI_CONCURRENCY` to override). The Anthropic httpx.Client is
  thread-safe; cache writes serialize on the main thread via
  `as_completed`, so no append interleaving. Set
  `$LLMOJI_CONCURRENCY=1` to force serial dispatch when debugging.
- Public-API freeze: anything in §"v1.0 frozen public surface" gets
  a major-version bump if changed. Internal helpers (everything in
  `llmoji.providers.base._*`, `llmoji.haiku.cache_key`, etc.) are
  free to evolve.
