# CLAUDE.md

## What this is

`llmoji` is a small provider-agnostic CLI for collecting kaomoji
journals from coding agents (Claude Code, Codex, Hermes), distilling
them into per-(source-model, canonical-face) descriptions via the
user's chosen synthesis backend, and submitting privacy-preserving
aggregates to a shared HF dataset for cross-corpus research.
Companion to the research-side
[`llmoji-study`](https://github.com/a9lim/llmoji-study) repo, where
all probe / hidden-state / MiniLM-embedding / axis-projection /
figure work lives.

This package is the data-layer-only end-user side: zero dependency on
saklas, torch, sentence-transformers, or matplotlib. Runtime deps are
**`anthropic`** (default synth backend), **`openai`** (for the
`--backend openai` Responses-API path AND the `--backend local`
OpenAI-compatible Chat-Completions path against Ollama / vLLM /
llama.cpp's HTTP server), and **`huggingface_hub`** (for the upload
target). Everything else — embedding, eriskii axis projection,
clustering, figures — is research-side and lives in `llmoji-study`,
which `pip install llmoji>=1.1,<2` and reads our public surface.

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
            └──────┬───────┘                 │ claude_export / │
                   │                         │ chatgpt_export  │
                   │                         └─────────────────┘
                   ▼
           ┌──────────────────┐                ┌──────────────┐
           │ llmoji.synth     │ ─per-instance▶│ ~/.llmoji/   │
           │ DESCRIBE_PROMPT_*│  cache key    │ cache/per_   │
           └──────┬───────────┘  by synth     │ instance.    │
                  │              model +      │ jsonl        │
                  │              content hash └──────────────┘
                  ▼
          ┌────────────────┐
          │ llmoji.synth   │     pool by (source_model,
          │ SYNTHESIZE_    │ ◀── canonical_kaomoji); one
          │ PROMPT         │     synthesis per cell
          └──────┬─────────┘
                 ▼
          ┌──────────────────┐
          │ ~/.llmoji/       │  manifest.json
          │ bundle/          │   + <source-model>/descriptions.jsonl
          │                  │   per source model
          │                  │  (loose files for user inspection)
          └──────┬───────────┘
                 │  llmoji upload --target {hf,email}
                 ▼
          ┌──────────────────┐
          │ a9lim/llmoji HF  │
          │ dataset OR email │
          └──────────────────┘
```

The bundle landing on disk between `analyze` and `upload` is the
deliberate inspection gap — the user `cat`s each
`<source-model>/descriptions.jsonl` before deciding to ship.

### Provider abstraction

`llmoji.providers.Provider` is a base class one subclass per
first-class harness. Each provider knows:

- where the harness keeps its hooks dir + settings file
  (`hooks_dir`, `settings_path`)
- where the journal lives (`journal_path`)
- which Stop-equivalent event the harness fires on (`main_event`)
- how the bash hook should bail when the kaomoji filter rejects a
  message (`skip_action` — `continue` for claude_code/codex
  because the validate partial sits inside a per-message
  `while read` loop, `echo '{}'; exit 0` for hermes's stdout-JSON
  single-shot contract)
- which prefixes mark a system-injected user-role payload
  (`system_injected_prefixes`)
- nudge attrs (template / filename / event / message) if the
  provider opts in

JSON-settings providers (claude_code, codex) inherit the default
`_register` / `_unregister` / `_is_registered` /
`_is_nudge_registered` from `Provider`; YAML-settings providers
(hermes) override the four. Past versions of this doc listed
`kaomoji_position` / `sidechain_strategy` / `sidechain_config` /
`settings_format` — those were documentary class attrs that no code
branched on, so they're gone. The behavior they described still
holds, but it lives in each provider's hook template + register
override, not in driver attrs.

The bash hook templates live as data files under `llmoji/_hooks/`,
plus two **shared partials** that every main hook inlines:

- `_kaomoji_validate.sh.partial` — the leading-prefix extractor and
  validator. Substituted with `${KAOMOJI_START_CASE}` (built from
  `llmoji.taxonomy.KAOMOJI_START_CHARS`) and `${SKIP_ACTION}`
  before being inserted as `${KAOMOJI_VALIDATE}` into the main
  template.
- `_journal_write.sh.partial` — the `jq -nc … >> $JOURNAL_PATH`
  tail. Substituted with `${JOURNAL_PATH}` then inserted as
  `${JOURNAL_WRITE}`.

`Provider.render_hook()` runs `string.Template.safe_substitute`
twice — once on each partial with its own placeholders, once on the
main template with `JOURNAL_PATH`, `KAOMOJI_VALIDATE`, `JOURNAL_WRITE`,
`INJECTED_PREFIXES_FILTER`, `LLMOJI_VERSION`. Two passes because
`safe_substitute` is single-pass and the partials' own `${...}`
references wouldn't survive a one-pass render. Single source of
truth in Python (the start-char set lives in `llmoji.taxonomy`,
system-injection prefixes on the Provider class).

### Synthesis backend abstraction

The synthesizer is one of three concrete backends, all routed
through `llmoji.synth.Synthesizer` + `llmoji.synth.make_synthesizer`:

- **`AnthropicSynthesizer`** — `anthropic.Anthropic.messages.create`
  with `max_retries=8`, default model
  `DEFAULT_ANTHROPIC_MODEL_ID` from `llmoji.synth_prompts`.
- **`OpenAISynthesizer`** — `openai.OpenAI.responses.create` (the
  Responses API; OpenAI's recommended path for new projects).
  Returns the `.output_text` convenience accessor. Default model
  `DEFAULT_OPENAI_MODEL_ID`.
- **`LocalSynthesizer`** — `openai.OpenAI(base_url=...,
  api_key="ollama").chat.completions.create`. Chat Completions
  rather than Responses because Ollama / vLLM / llama.cpp's HTTP
  server all expose Chat-Completions-shaped endpoints. No default
  model id; user must pass `--model`.

All three defer SDK-client construction to the first `.call()` so
the factory has no env-var dependency at construction time —
`make_synthesizer("openai")` works without `OPENAI_API_KEY` set
(which is exercised by `test_make_synthesizer_dispatches`).

### Two-stage synthesis pipeline

- Stage A (per instance): for each `(source_model,
  canonical_kaomoji)` cell, sample up to `INSTANCE_SAMPLE_CAP` rows
  per cell (deterministic seed = `f"{INSTANCE_SAMPLE_SEED}:
  {source_model}:{canonical}"`); mask the kaomoji to `[FACE]` and
  call the chosen synthesizer with `DESCRIBE_PROMPT_WITH_USER` or
  `DESCRIBE_PROMPT_NO_USER`. Cache keyed by `sha256(synth_model_id
  + "\0" + canonical + "\0" + user + "\0" + assistant)[:16]` at
  `~/.llmoji/cache/per_instance.jsonl`. Re-runs of `analyze` with
  the same synth model skip rows already described; switching
  synth model misses cleanly (no stale cross-model cache hits).

- Stage B (per cell): pool Stage A descriptions for one
  `(source_model, canonical_kaomoji)` cell, synthesize a single
  1-2-sentence overall meaning via `SYNTHESIZE_PROMPT`. The
  synthesized line is the **only** thing that ships in the bundle,
  and it lands in that cell's `<source-model>/descriptions.jsonl`.

Embedding / axis projection / clustering / figures are NOT in this
package — they happen on the receiving research side, against either
our own corpus or submitted user bundles.

## Cross-corpus invariant surface (1.1 amended from the 1.0 freeze)

These are the parts the HF dataset's aggregation rules pin
against. 1.1.0 amended the bundle schema, the `synth_prompts`
module name, and the default-model constants — the second user
hadn't run `analyze` yet, so the schema break was safe to take in
a minor release. Bumping anything below changes the cross-corpus
invariant; flag in the PR body and update the HF dataset card to
match.

- **`llmoji.taxonomy`**:
  - `KAOMOJI_START_CHARS` (leading-glyph filter set)
  - rules A–P in `canonicalize_kaomoji`
  - `is_kaomoji_candidate` validator contract
  - `extract` / `KaomojiMatch` (span-only — no affect labels;
    gemma-tuned label dicts moved to research-side
    `llmoji_study.taxonomy_labels`)
- **`llmoji.synth_prompts`**:
  - `DESCRIBE_PROMPT_WITH_USER`, `DESCRIBE_PROMPT_NO_USER`
  - `SYNTHESIZE_PROMPT`
  - `DEFAULT_ANTHROPIC_MODEL_ID` (pinned Haiku snapshot)
  - `DEFAULT_OPENAI_MODEL_ID` (pinned GPT-5.4 mini snapshot)
- **`llmoji.scrape.ScrapeRow`** schema (the in-memory row shape)
- **The 6-field unified journal row schema** (on-disk JSONL):
  `{ts, model, cwd, kaomoji, user_text, assistant_text}`
- **System-injection prefix lists** per provider (in
  `llmoji.providers.{claude_code,codex,hermes}`)
- **`llmoji.providers.Provider`** interface
- **Bundle schema**:
  - top-level `manifest.json` keys: `llmoji_version`,
    `synthesis_model_id`, `synthesis_backend`, `submitter_id`,
    `generated_at`, `providers_seen`, `model_counts`,
    `total_synthesized_rows`, `notes`
  - one `<sanitized_source_model>/descriptions.jsonl` per source
    model the journal saw, each row carrying
    `{kaomoji, count, synthesis_description}`
  - subfolder name = `sanitize_model_id_for_path(source_model)`
    (lowercase, `/` → `__`, `:` → `-`)

Free to change without bumping the cross-corpus invariant: cache
key derivation, `INSTANCE_SAMPLE_CAP` / `INSTANCE_SAMPLE_SEED`,
internal flag names beyond `--target {hf,email}` and `--backend
{anthropic,openai,local}`, etc.

## Commands

```
llmoji install <provider>      write hook + register; idempotent
llmoji uninstall <provider>    inverse; idempotent (journal preserved)
llmoji status                  installed providers, journal sizes, paths
llmoji parse --provider <n> P  ingest a static export dump
                               (claude.ai or chatgpt
                               conversations.json) into
                               ~/.llmoji/journals/
llmoji analyze [--notes …]     scrape + canonicalize + synthesize
[--backend …] [--model …]      → ~/.llmoji/bundle/. backend defaults
[--base-url …]                 to anthropic; openai uses Responses
                               API; local uses Chat Completions
                               (Ollama / vLLM / llama.cpp HTTP)
llmoji upload --target {hf,email} [--yes]  ship the bundle (HF: loose
                                            files via single atomic
                                            commit; email: tarball)
llmoji cache clear             wipe ~/.llmoji/cache/
```

`upload` requires `--target` (no default) and re-prompts before
committing. `--yes` skips the prompt for scripted use.

## Layout

```
llmoji/
  pyproject.toml          # PEP 621 + hatch dynamic version (reads
                          # __version__ from llmoji/__init__.py via
                          # [tool.hatch.version] regex source)
  README.md               # public-facing, voice-rewritten using the
                          # writing skill; technical-professional
                          # register, em-dashes out
  CONTRIBUTING.md         # warm public-prose welcome + dev setup +
                          # adding-a-provider checklist
  SECURITY.md             # privacy threat model (the bundle is the
                          # consent boundary; cache never ships;
                          # singleton-kaomoji caveat called out)
  LICENSE                 # MIT (PEP 639 SPDX in pyproject)
  CLAUDE.md
  .gitignore
  .github/
    dependabot.yml        # weekly pip + github-actions; anthropic /
                          # huggingface_hub bumped manually
    PULL_REQUEST_TEMPLATE.md
    ISSUE_TEMPLATE/
      config.yml          # blank issues off; security + take-down
                          # contact links
      bug_report.yml
      feature_request.yml
      new_provider.yml
    workflows/
      ci.yml              # ruff + pytest matrix (3.11–3.13 ×
                          # ubuntu+macos) + build-and-import-wheel
                          # gate that confirms hatch's regex source
                          # round-trips without executing __init__
      release.yml         # version-from-attr → tag → PyPI publish →
                          # GitHub release; same pattern as saklas
  examples/
    README.md             # warm-public-prose intro
    inspect_bundle.py     # audit-the-bundle script (the consent step)
    openclaw_hook.ts      # worked generic-JSONL-append example for
                          # the post-v1.0 OpenClaw first-class story
  llmoji/
    py.typed              # PEP 561 type marker (Typing :: Typed
                          # classifier in pyproject)
    __init__.py            # public surface re-exports
    _util.py               # cross-cutting helpers: atomic_write_text
                           # (tmp+rename), write_json, package_version,
                           # human_bytes, sanitize_model_id_for_path
                           # (subfolder-name rule for the per-source-model
                           # bundle layout — lives here rather than in
                           # synth.py so upload.py's allowlist walker can
                           # reuse it without dragging in synth backends).
                           # Imported by analyze, upload, cli, providers
                           # — kept out of providers.base so the
                           # dependency graph stays tree-shaped (upload
                           # doesn't reach into providers for io utilities).
    taxonomy.py            # KAOMOJI_START_CHARS + is_kaomoji_candidate
                           # + extract + KaomojiMatch (span-only)
                           # + canonicalize_kaomoji (rules A–P; frozen)
    synth_prompts.py       # DESCRIBE_PROMPT_* + SYNTHESIZE_PROMPT
                           # + DEFAULT_ANTHROPIC_MODEL_ID
                           # + DEFAULT_OPENAI_MODEL_ID. Locked
                           # cross-corpus invariants — bumping any
                           # changes the prose the dataset receives.
    synth.py               # mask_kaomoji + cache helpers (single
                           # JSONL keyed by synth_model_id + canon +
                           # user + assistant) + Synthesizer abstract
                           # base + AnthropicSynthesizer (messages.create
                           # with max_retries=8) + OpenAISynthesizer
                           # (Responses API, .output_text accessor) +
                           # LocalSynthesizer (OpenAI-compatible Chat
                           # Completions, api_key="ollama" placeholder)
                           # + make_synthesizer factory.
                           # All three concrete synthesizers defer SDK
                           # client construction to first .call() so
                           # the factory has no env-var dependency at
                           # construction time.
    scrape.py              # ScrapeRow + iter_all chain helper
                           # (span-only schema; no affect labels)
    sources/
      _common.py           # kaomoji_lead_strip — validate-and-strip
                           # helper shared by every static-export
                           # reader. Single source of truth for the
                           # journal-row contract on the export side
                           # (assistant_text never carries the kaomoji).
      journal.py           # generic kaomoji-journal reader (any
                           # provider's ~/.<harness>/kaomoji-journal.jsonl)
      claude_export.py     # Claude.ai conversations.json reader
                           # (linear chat_messages array)
      chatgpt_export.py    # OpenAI ChatGPT conversations.json reader.
                           # Same filename as Claude's, different
                           # schema: a tree of message nodes keyed by
                           # id, with current_node pointing at the
                           # active leaf. Walks mapping[current_node]
                           # up via parent for the displayed branch;
                           # regenerated/edited siblings stay invisible.
    backfill.py            # one-shot transcript→journal replays for
                           # Claude Code + Codex + Hermes. Hermes
                           # backfill walks ~/.hermes/sessions/
                           # session_*.json and chunks on user-role
                           # boundaries (each chunk = one turn).
    providers/
      base.py              # Provider + ProviderStatus dataclasses +
                           # SettingsCorruptError + template render
                           # helpers + JSON-settings edit helpers
                           # (batched register/unregister/is_registered
                           # against main_event + nudge_event; one
                           # read-modify-write cycle per install).
                           # Atomic settings writes go through
                           # llmoji._util.atomic_write_text.
      claude_code.py       # JSON Stop+UserPromptSubmit provider
                           # (~/.claude/settings.json); shares the
                           # nudge template with codex. Inherits the
                           # default _register family from Provider.
      codex.py             # JSON Stop+UserPromptSubmit provider
                           # (~/.codex/hooks.json); the codex_hooks
                           # feature flag is Stage::Stable, default-on
                           # in codex-rs/features. Same default
                           # _register family as claude_code.
      hermes.py            # YAML pre_llm_call + post_llm_call
                           # provider (~/.hermes/config.yaml,
                           # marker-fenced hooks: stanza). Overrides
                           # _register family. The post_llm_call hook
                           # walks extra.conversation_history for
                           # multi-emit parity with claude_code/codex.
    _hooks/                # bash hook templates (importlib.resources
                           # data); rendered at install time
      claude_code.sh.tmpl
      codex.sh.tmpl
      hermes.sh.tmpl                # post_llm_call (journal logger)
      _kaomoji_validate.sh.partial  # shared validator inlined into
                                    # every main hook via ${KAOMOJI_VALIDATE}
      _journal_write.sh.partial     # shared jq-write tail inlined via
                                    # ${JOURNAL_WRITE}
      claude_codex_nudge.sh.tmpl    # UserPromptSubmit nudge — shared
                                    # between claude_code + codex (the
                                    # response envelope is byte-identical)
      hermes_nudge.sh.tmpl          # pre_llm_call nudge (bare
                                    # {context: ...} shape)
    paths.py               # ~/.llmoji home, cache, bundle, journals,
                           # state.json (per-machine submission token).
                           # NOT an install registry — provider install
                           # state is read live from each harness's own
                           # settings file by Provider.status().
    analyze.py             # the analyze pipeline (Stage A + B + bundle;
                           # clears bundle dir AND every subdir before
                           # writing — per-source-model layout means
                           # stale subfolders would silently leak).
                           # Buckets by (source_model, canonical) where
                           # source_model = ScrapeRow.model or fall
                           # back to ScrapeRow.source when empty.
                           # Stage A dispatches cache-miss synth calls
                           # on a ThreadPoolExecutor (default 2 workers,
                           # env $LLMOJI_CONCURRENCY); cache appends
                           # serialize on the main thread via
                           # as_completed. Manifest stamps
                           # synthesis_model_id, synthesis_backend,
                           # submitter_id, model_counts (per source
                           # model) so the inspected bundle byte-matches
                           # what HF would receive.
    upload.py              # tar + HF / email targets;
                           # BUNDLE_TOPLEVEL_ALLOWLIST + BUNDLE_SUBDIR_FILE
                           # enforce the structural shape (manifest at
                           # top, descriptions.jsonl per source-model
                           # subdir, nothing else, no recursion past
                           # one level). Refuses extras (raises
                           # BundleAllowlistError). submitter_id() is
                           # public so analyze can stamp the manifest.
    cli.py                 # argparse entry, [project.scripts] llmoji.
                           # analyze takes --backend / --base-url /
                           # --model with env-var fallbacks (LLMOJI_BACKEND,
                           # LLMOJI_BASE_URL, LLMOJI_MODEL); validates
                           # that local needs both --base-url and
                           # --model and that anthropic/openai don't
                           # accept either (loud failure beats silent
                           # ignore on the pinned-snapshot path).
  tests/
    test_public_surface.py # pytest checks locking the cross-corpus
                           # invariant contract (taxonomy / synth_prompts
                           # default model ids / ScrapeRow / synthesizer
                           # factory dispatch / sanitize_model_id_for_path
                           # / provider rendering + bash -n / structural
                           # bundle allowlist / corrupt-config refusal /
                           # mask_kaomoji unified prepend contract /
                           # nudge install/uninstall round-trip).
    test_canonicalize.py   # parametrized rule-by-rule regression
                           # tests for canonicalize_kaomoji + extract
                           # + is_kaomoji_candidate. Each rule case is
                           # its own pytest line. ~70 cases total.
    test_pipeline_parity.py  # cross-validates the bash live hook vs
                           # the Python backfill on synthetic transcripts
                           # for claude_code + codex (every parity-
                           # critical field must match), plus a
                           # bash-hook-only smoke test for hermes
                           # (no Python backfill counterpart).
```

## Gotchas

### Journal-row contract: `assistant_text` never carries the kaomoji

Every source — bash hooks, Claude.ai export reader, ChatGPT export
reader, generic-JSONL contract — must persist `assistant_text` with
the leading kaomoji already stripped. The prefix lives separately
in the row's `kaomoji` field. The bash hooks enforce this via jq's
`sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")`; the two
static-export readers route through
`llmoji.sources._common.kaomoji_lead_strip`, which wraps
`taxonomy.extract` and returns `(first_word, body)` ready to drop
into a `ScrapeRow`. Future export readers should reach for the
shared helper rather than re-implementing the dance — drift between
sources is what the helper exists to prevent.

`mask_kaomoji` consequently has a single branch: prepend
`"[FACE] "` to whatever's there. No source-shape dispatch, no
substitute-in-place fallback. If you add a new source, strip on
the way in — don't push the special case into `mask_kaomoji`.

The cache key hashes `(synth_model_id, canonical, user_text,
assistant_text)`; existing cache entries from prior `parse +
analyze` runs of an export will miss after this normalization
(export rows now hash without the kaomoji). One-time re-call cost
on the next analyze.

### KAOMOJI_START_CHARS — single source of truth

- Python: `llmoji.taxonomy.KAOMOJI_START_CHARS`
- Shell hooks: `${KAOMOJI_START_CASE}` rendered at install time
  from the same set.
- `is_kaomoji_candidate` validates Python-side; the rendered case
  filter handles the shell-side first pass.

If you find another copy of the set, delete it and route through
`llmoji.taxonomy`.

### Per-provider kaomoji capture — one row per kaomoji-led message

**Both Claude Code and Codex emit N rows per turn**, one per
kaomoji-led model message. A tool-heavy turn easily writes 5–10
kaomoji-led messages interleaved with tool calls; pre-fix the
hooks emitted at most one row per turn, dropping every progress
message. Post-fix the hooks walk the transcript / rollout and emit
per-message.

- **Claude Code**: each assistant content block (text, tool_use,
  thinking) is its own top-level transcript JSONL entry; one turn
  produces many entries. The Stop hook scopes to entries at-or-
  after the latest real-user message (string content OR text-block
  array, NOT tool_result) and walks every text-bearing non-side
  chain entry in that window. Each entry's first text block runs
  through the kaomoji validator; non-kaomoji entries skip their
  row without aborting the rest of the walk. The `BOUNDARY_TS`
  query slurps the transcript once; the per-entry walk is `jq -c`
  streamed line-by-line into a `while read` loop with `SKIP_ACTION
  =continue`.
- **Codex**: each model message is its own `event_msg.agent_message`
  event with `payload.message` carrying the text and `payload.phase`
  flagging `"commentary"` (progress) vs `"final_answer"` (closing).
  The Stop hook finds the latest `turn_context` index in the
  rollout (current turn boundary), slices forward, and walks every
  agent_message in the slice. `task_complete.last_agent_message`
  is no longer read on either side. `user_text` resolves to the
  latest non-injected user response_item in the same slice.
- **Hermes**: walks `extra.conversation_history` (the full message
  list the post_llm_call payload carries), slices from the latest
  user-role message to the end of the array, emits one row per
  kaomoji-led non-empty assistant-role message in that slice. Same
  multi-emit shape as Claude Code + Codex now. Pre-fix the hook
  read only `extra.assistant_response` (the final string) and
  missed every progress message. The conversation_history field is
  always populated per `hermes-agent/run_agent.py:12492` (`list(
  messages)`), so the walker has the data it needs without a
  follow-up payload-shape change upstream. Tool-only assistant
  messages (carry `tool_calls` + empty/null `content`) are skipped
  naturally; the walker only takes string-typed non-empty content.

Per-row invariants for the multi-emission case:

- `user_text` is resolved once per turn — every row from one turn
  carries the same originating prompt.
- The cache key hashes `(synth_model_id, canonical, user_text,
  assistant_text)` — different assistant texts within the same
  turn produce different keys, so per-instance caching works
  without collisions. Switching synth model misses cleanly
  (no stale cross-model cache hits).
- Backfills (`backfill_codex`, `backfill_claude_code`,
  `backfill_hermes`) implement the same per-message walk and stay
  parity-tested against the live hooks via
  `tests/test_pipeline_parity.py`. The `_PARITY_FIELDS` contract
  holds row-for-row in chronological order. Hermes uses the
  narrower `_HERMES_PARITY_FIELDS` (excludes `cwd`) because the
  session JSON doesn't persist cwd — backfilled rows carry `""`
  there by design while the live hook stamps `Path.cwd()` from the
  agent process.
- Existing journals from before the fix are systematically thin
  (1 row per turn vs. N). On first analyze post-fix, prefer a
  one-shot `backfill_*` rebuild over the live-hook journal so
  per-canonical-face counts reflect real traffic.

### Nudge hooks — what gives the corpus its size

Each provider ships a tiny **nudge** hook alongside the journal
logger. The nudge fires before each model turn (UserPromptSubmit
on Claude/Codex, `pre_llm_call` on Hermes) and injects a fresh
"please begin your message with a kaomoji that best represents
how you feel" reminder as additional context. Without the nudge
the model drifts away from leading kaomoji over a long session;
with it the journal stays dense.

Response shapes differ:

- **Claude Code + Codex**: `{"hookSpecificOutput": {"hookEventName":
  "UserPromptSubmit", "additionalContext": "<msg>"}}`. Codex's
  envelope is byte-identical to Claude Code's (verified at
  `codex-rs/hooks/src/events/user_prompt_submit.rs`), so a single
  shared `claude_codex_nudge.sh.tmpl` template serves both. The
  `nudge_message` itself substitutes through `_shell_quote` into a
  bash single-quoted literal, so embedded apostrophes round-trip
  cleanly.
- **Hermes**: bare `{"context": "<msg>"}` — no envelope, returned
  by `pre_llm_call`, which the docs call out as "the only hook
  whose return value is used."

The base `Provider` class exposes the nudge through
`nudge_hook_template` / `nudge_hook_filename` / `nudge_event` /
`nudge_message` class attrs and a `has_nudge` predicate. Providers
that opt in get the nudge written + registered automatically by
`install`, removed by `uninstall`, and reported by `status`. Adding
a nudge to a future provider is four class-level attrs and a
`_is_nudge_registered` override.

### Sidechain strategy

- Claude Code: `field_flag` on `isSidechain`. Hooks drop the row.
- Codex: no subagent concept. `collaboration_mode` is `"default"`
  for every observed turn_context.
- Hermes: **no viable filter on the current payload contract.**
  `subagent_stop` fires from the parent agent's process with the
  **parent's** `session_id` (no child id; verified at
  `hermes-agent/tools/delegate_tool.py:2120-2127` —
  `_invoke_hook("subagent_stop", parent_session_id=..., child_role
  =..., child_summary=..., child_status=..., duration_ms=...)`),
  and `post_llm_call` doesn't expose `parent_session_id` either, so
  neither side carries enough info to filter children from a shell
  hook. We installed a companion `subagent-stop.sh` for a previous
  iteration that recorded "child session_ids"; that file was empty
  in practice because the documented child-id field doesn't exist
  in the actual payload. The companion hook + its state file are
  removed. Subagent `post_llm_call` events therefore land in the
  journal under their own session_ids until upstream gives us a
  signal — either (a) `subagent_stop` carrying the child id, or
  (b) `post_llm_call` exposing `parent_session_id` /
  `is_subagent`.

### Provider install refuses to clobber existing config

Three corruption paths are explicitly defended:

1. **Malformed JSON in `~/.claude/settings.json`** — pre-fix the
   loader returned `None` and `install` would silently treat it as
   `{}` and rewrite. Post-fix `_load_json_strict` raises
   `SettingsCorruptError` and the user has to fix the file by hand
   before `install` will touch it.
2. **Malformed JSON in `~/.codex/hooks.json`** — same defense as
   claude_code, same helper. Codex's `codex_hooks` feature flag is
   `Stage::Stable` + `default_enabled: true`, payload shape is
   byte-identical to claude_code's, so we reuse the JSON helpers.
3. **Existing top-level `hooks:` key in `~/.hermes/config.yaml`** —
   appending another `hooks:` makes a duplicate-key YAML doc that
   silently last-write-wins. `HermesProvider._has_unmanaged_hooks_top_level`
   refuses to install.

In all three cases the user gets a `SettingsCorruptError` with a
specific path and reason. They edit the file (move-aside or merge
by hand) and re-run.

The non-managed analogue — a user re-running `install` after
already installing once — is fully idempotent. The marker fence in
hermes means the second `install` is a no-op; the JSON-edit path
in claude_code and codex checks for an existing entry with our
command string and skips. Both the main and nudge hooks dedup
independently — re-installing only one of the two pairs is fine.

Settings writes go through `llmoji._util.atomic_write_text` (tmp
file + `os.replace`) so a power loss / SIGINT mid-write leaves the
user's settings file with either the old content or the new — never
half. The `upload` state.json (per-machine submission token) writes
the same way. JSON-settings providers also batch their main+nudge
edits into a single read-modify-write cycle per `install` (via
`_register_json_settings_batch`), so a SIGKILL between registering
the Stop hook and the UserPromptSubmit nudge can't leave the user
half-installed.

### Bundle is allowlisted, not just-ship-everything

Both upload paths enforce the structural allowlist:
`BUNDLE_TOPLEVEL_ALLOWLIST = ("manifest.json",)` at the top level
plus exactly one `BUNDLE_SUBDIR_FILE = "descriptions.jsonl"` inside
each per-source-model subfolder, and no recursion past one level.

- `upload.tar_bundle()` (used by the email target) raises
  `BundleAllowlistError` if the bundle dir holds anything else,
  whether at the top level or inside a model subfolder.
- `upload.upload_hf()` does the same pre-flight check AND passes
  `allow_patterns=["manifest.json", "*/descriptions.jsonl"]` to
  `HfApi.upload_folder` as a second line of defense.

`analyze` clears the bundle dir of all top-level files AND all
subdirs before writing, so a clean run produces exactly the
structural shape. The three together mean stale per-instance
descriptions, user-added notes, hidden-state caches, leftover
subfolders from a prior backend run, etc. cannot accidentally
leak through `upload`.

### HF upload is loose files, not a tarball

`upload --target hf` pushes `manifest.json` plus each
`<source-model>/descriptions.jsonl` as loose files at
`contributors/<hash>/bundle-<ts>/` via `HfApi.upload_folder`
(single atomic commit). The dataset card on the HF side has a
`configs:` YAML pointing at `contributors/**/descriptions.jsonl`,
which is what the auto-loader needs to surface the dataset viewer.
The `**` glob is recursive so the new per-source-model nesting
matches without a card change to the loader path; only the
field-by-field schema prose on the card needs hand-editing (see
the "HF dataset card" gotcha below). Uploading as a tarball
triggered HF's WebDataset auto-detection and broke the viewer
(WebDataset expects shared-prefix archives, our loose-files
bundles don't fit).

Email target keeps `tar_bundle` because a single attachment is
what the recipient wants. The local tarball at
`~/.llmoji/bundle-<ts>.tar.gz` is now an email-only artifact.

### Hermes provider — main hook + nudge, source-verified contract

The hermes provider installs **two** hooks, both wired through the
same shell-hooks mechanism:

- `~/.hermes/agent-hooks/post-llm-call.sh` — main journal logger.
  Walks `extra.conversation_history`, slices from the latest
  user-role message to the end, emits one row per kaomoji-led
  non-empty assistant-role message in that slice. Tool-only
  assistant messages (carry `tool_calls` + empty/null `content`)
  are skipped naturally — the walker only takes string-typed
  non-empty content.
- `~/.hermes/agent-hooks/pre-llm-call.sh` — nudge that injects the
  kaomoji-reminder context via the `{context: "<msg>"}` shape (per
  docs, "the only hook whose return value is used").

Both registered in `~/.hermes/config.yaml` under the `hooks:` block,
inside our managed marker fence so re-running install is idempotent
and uninstall removes the stanza cleanly.

The implementation cross-checks the documented [Event Hooks][hermes-hooks]
contract against the actual source at
`hermes-agent/agent/shell_hooks.py:_serialize_payload` (top-level
shape) and `hermes-agent/run_agent.py:12492` (`post_llm_call`
kwargs: `session_id`, `user_message`, `assistant_response`,
`conversation_history`, `model`, `platform`). The `extra.*` block
holds everything except the four reserved top-level keys
(`tool_name`, `args`, `session_id`, `parent_session_id`); `cwd` is a
top-level field set to `Path.cwd()` of the agent process at hook
fire time, NOT under `extra`.

`extra.user_message` is the original pre-injection user message
(see `original_user_message` at the call site), so
`system_injected_prefixes` stays `[]`. If a future real-traffic
inspection shows leaked injection prefixes, populate the list and
re-render — the bash hook picks the same list up via
`${INJECTED_PREFIXES_FILTER}`.

A previous iteration installed a third companion hook
(`subagent-stop.sh`) intended to record child session_ids for
sidechain filtering. Source review showed that approach is
unworkable: `subagent_stop` fires from the parent agent's process
with the **parent's** `session_id` and no child id at all
(`hermes-agent/tools/delegate_tool.py:2120-2127`), so there's
nothing to record. The companion hook is removed; subagent
filtering is documented as not currently viable (see §"Sidechain
strategy") and traffic from delegated subagents lands in the
journal under its own session_id.

[hermes-hooks]: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/

### Cache directory is leakier than the bundle

`~/.llmoji/cache/per_instance.jsonl` holds synthesizer-paraphrased
descriptions of single user turns, keyed by `(synth_model_id,
canonical, user_text, assistant_text)`. Each row IS one user-turn
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

### HF dataset card is a separate hand-maintained surface

The user-facing dataset card at
https://huggingface.co/datasets/a9lim/llmoji is a separate document
from anything in this repo. It re-states the bundle schema and the
privacy model in user-facing prose so contributors reading the
dataset page can decide whether to submit before they've ever
touched the package README.

Two coupling points to keep in mind:

- **Schema changes need both updates.** Any change to
  `bundle/manifest.json` or `<source-model>/descriptions.jsonl`
  field names is a cross-corpus invariant change (see §"Cross-corpus
  invariant surface"), so it wants a hand-edit on the HF dataset
  card so the field-by-field schema documentation doesn't go stale.
  The card is editable in-place via the HF web UI; the canonical
  surface lives there, not in this repo. The 1.1.0 schema bump
  (manifest fields renamed, per-source-model subfolder layout)
  needs a one-time card update on merge — the auto-loader glob
  (`contributors/**/descriptions.jsonl`) handles the new nesting
  via the recursive `**` without changes.
- **License split.** The package code is GPL-3.0-or-later; the
  shared corpus on HF is CC-BY-SA-4.0. `llmoji upload --target hf`
  contributes a bundle under CC-BY-SA-4.0, and the package README's
  License section calls this out so contributors aren't surprised.
  `llmoji-study` is CC-BY-SA-4.0 — it's a research artifact
  (writeups, figures, analysis pipelines) rather than a distributed
  program, and matching the corpus license keeps derivative work
  under one consistent set of terms.

## Conventions

- Single venv at `.venv/`, pip not uv. `pip install -e ../llmoji`
  during dev; PyPI install at freeze.
- `~/.llmoji` is the on-disk root for everything the package
  manages; tests can override via `$LLMOJI_HOME`.
- Hook templates are bash, syntactically validated by `bash -n`
  in the test suite (`test_hook_templates_render_to_valid_bash_substitutions`)
  so a template-edit regression fails CI rather than failing
  silently inside a user's harness post-install. Don't introduce
  non-bash hook formats; if a harness needs one, that's a post-1.0
  first-class adapter, and the generic-JSONL-append contract is the
  path until then.
- Stage-A synth calls run on a small thread pool (default 2,
  `$LLMOJI_CONCURRENCY` to override). Both the Anthropic and OpenAI
  SDKs use thread-safe httpx clients; cache writes serialize on
  the main thread via `as_completed`, so no append interleaving.
  Set `$LLMOJI_CONCURRENCY=1` to force serial dispatch when
  debugging. The default sits at 2 because the org-level Haiku
  rate limit is 50 req/min; 4 concurrent workers reliably trip it
  on a multi-hundred-row backfill, and the SDK's `max_retries=8`
  exponential backoff (set explicitly in `AnthropicSynthesizer.
  __init__` and `OpenAISynthesizer.__init__`, vs the SDK default
  of 2) recovers but burns wallclock. Per-cell sample cap
  `INSTANCE_SAMPLE_CAP` is 4 — popular faces (>4 rows in a given
  source model) get capped, rare faces fully sampled. Same value
  as Eriskii's original Claude-faces work, kept for cross-corpus
  comparability.
- Public-API freeze: anything in §"Cross-corpus invariant surface"
  is a cross-corpus invariant; bumping wants a hand-edit on the
  HF dataset card and a flag in the PR body. Internal helpers
  (everything in `llmoji._util`, leading-underscore names in
  `llmoji.providers.base`, `llmoji.synth.cache_key`, the synth
  backend classes, etc.) are free to evolve.
