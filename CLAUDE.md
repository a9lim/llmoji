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

This package is the data-layer-only end-user side: zero dependency
on saklas, torch, sentence-transformers, or matplotlib. Runtime deps
are **`anthropic`** (default synth backend), **`openai`** (the
`--backend openai` Responses-API path AND the `--backend local`
OpenAI-compatible Chat-Completions path against Ollama / vLLM /
llama.cpp's HTTP server), and **`huggingface_hub`** (upload target).
Everything else — embedding, eriskii axis projection, clustering,
figures — is research-side and lives in `llmoji-study`, which
`pip install llmoji>=1.1,<2` and reads our public surface.

## Pipeline

```
harness hook  →  ~/.<harness>/kaomoji-journal.jsonl   (6-field rows)
              →  llmoji.sources.* readers + taxonomy.canonicalize
              →  Stage A: per-instance describe (cached)
              →  Stage B: per-cell synthesize
              →  ~/.llmoji/bundle/  (manifest.json + <model>.jsonl)
              →  llmoji upload --target {hf,email}
```

The bundle on disk between `analyze` and `upload` is the deliberate
inspection gap — the user `cat`s each `<source-model>.jsonl` before
deciding to ship.

### Provider abstraction

`llmoji.providers.Provider` is the base class, one subclass per
first-class harness. JSON-settings providers (`ClaudeCodeProvider`,
`CodexProvider`) inherit from `JsonSettingsProvider` (also in
`base.py`), which supplies the default
`_register` / `_unregister` / `_is_registered` /
`_is_nudge_registered` against any `settings.json`-shaped file.
YAML-settings providers (`HermesProvider`) override the four.

Each provider declares: `hooks_dir`, `settings_path`, `journal_path`,
`main_event`, `skip_action` (`continue` for claude_code/codex —
their validator partial sits inside a per-message `while read` loop;
`echo '{}'; exit 0` for hermes's stdout-JSON single-shot contract),
`system_injected_prefixes`, and optional nudge attrs
(`nudge_hook_template` / `nudge_hook_filename` / `nudge_event` /
`nudge_message`).

Bash hook templates live as data files under `llmoji/_hooks/`, plus
two **shared partials** every main hook inlines:

- `_kaomoji_validate.sh.partial` — leading-prefix extractor and
  validator. Substituted with `${KAOMOJI_START_CASE}` (built from
  `llmoji.taxonomy.KAOMOJI_START_CHARS`) and `${SKIP_ACTION}`,
  inserted as `${KAOMOJI_VALIDATE}`.
- `_journal_write.sh.partial` — the `jq -nc … >> $JOURNAL_PATH`
  tail. Substituted with `${JOURNAL_PATH}`, inserted as
  `${JOURNAL_WRITE}`.

`Provider.render_hook()` runs `string.Template.safe_substitute`
twice — once on each partial with its own placeholders, once on
the main template with `JOURNAL_PATH`, `KAOMOJI_VALIDATE`,
`JOURNAL_WRITE`, `INJECTED_PREFIXES_FILTER`, `LLMOJI_VERSION`. Two
passes because `safe_substitute` is single-pass and the partials'
own `${...}` references wouldn't survive a one-pass render.

### Synthesis backends

Three concrete backends, all routed through `llmoji.synth.Synthesizer`
+ `llmoji.synth.make_synthesizer`:

- **`AnthropicSynthesizer`** — `anthropic.Anthropic.messages.create`
  with `max_retries=8`, default `DEFAULT_ANTHROPIC_MODEL_ID`.
- **`OpenAISynthesizer`** — `openai.OpenAI.responses.create` (the
  Responses API; OpenAI's recommended path), `.output_text`
  accessor, default `DEFAULT_OPENAI_MODEL_ID`.
- **`LocalSynthesizer`** —
  `openai.OpenAI(base_url=..., api_key="ollama").chat.completions.create`.
  Chat Completions because Ollama / vLLM / llama.cpp HTTP all expose
  Chat-Completions-shaped endpoints. No default model id; user must
  pass `--model`.

All three defer SDK-client construction to the first `.call()` so
the factory has no env-var dependency at construction time.

### Two-stage synthesis pipeline

- **Stage A (per instance)**: for each `(source_model,
  canonical_kaomoji)` cell, sample up to `INSTANCE_SAMPLE_CAP` rows
  (deterministic seed `f"{INSTANCE_SAMPLE_SEED}:{source_model}:{canonical}"`),
  mask the kaomoji to `[FACE]`, call the synthesizer with
  `DESCRIBE_PROMPT_WITH_USER` or `DESCRIBE_PROMPT_NO_USER`. Cache
  keyed by `sha256(synth_model_id + "\0" + backend + "\0" + base_url
  + "\0" + canonical + "\0" + user + "\0" + assistant)[:16]` at
  `~/.llmoji/cache/per_instance.jsonl`. Switching synth model OR
  backend OR (for `local`) endpoint misses cleanly. Within a wave,
  cache-miss API calls dispatch on a small thread pool but the
  serial walk that builds Stage B's input + appends the cache file
  runs in deterministic order — re-runs against the same journal
  feed Stage B identical descriptions in identical order.
- **Stage B (per cell)**: pool Stage A descriptions, synthesize a
  single 1–2-sentence overall meaning via `SYNTHESIZE_PROMPT`. The
  Stage B line is the **only** thing that ships — it lands in that
  cell's `<source-model>.jsonl` at the bundle root.

Embedding / axis projection / clustering / figures happen
research-side.

## Cross-corpus invariant surface

The HF dataset's aggregation rules pin against everything below.
Bumping any of these is a cross-corpus change — flag in the PR body
and update the HF dataset card to match.

- **`llmoji.taxonomy`**: `KAOMOJI_START_CHARS`, rules A–P in
  `canonicalize_kaomoji`, `is_kaomoji_candidate` validator
  contract, `extract` / `KaomojiMatch` (span-only — no affect
  labels; gemma-tuned label dicts moved to research-side
  `llmoji_study.taxonomy_labels`).
- **`llmoji.synth_prompts`**: `DESCRIBE_PROMPT_WITH_USER`,
  `DESCRIBE_PROMPT_NO_USER`, `SYNTHESIZE_PROMPT`,
  `DEFAULT_ANTHROPIC_MODEL_ID` (pinned Haiku snapshot),
  `DEFAULT_OPENAI_MODEL_ID` (pinned GPT-5.4 mini snapshot).
- **`llmoji.scrape.ScrapeRow`** schema (in-memory row shape).
- **6-field unified journal row schema** (on-disk JSONL):
  `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- **System-injection prefix lists** per provider (in
  `llmoji.providers.{claude_code,codex,hermes}`).
- **`llmoji.providers.Provider`** interface.
- **Bundle schema**:
  - `manifest.json` keys: `llmoji_version`, `synthesis_model_id`,
    `synthesis_backend`, `submitter_id`, `generated_at`,
    `providers_seen`, `model_counts`, `total_synthesized_rows`,
    `notes`.
  - one `<sanitized_source_model>.jsonl` per source model, each row
    `{kaomoji, count, synthesis_description}`.
  - filename stem = `sanitize_model_id_for_path(source_model)`
    (lowercase, `/` → `__`, `:` → `-`).

Free to evolve without bumping invariant: cache key derivation,
`INSTANCE_SAMPLE_CAP` / `INSTANCE_SAMPLE_SEED`, internal flag names
beyond `--target {hf,email}` and `--backend {anthropic,openai,local}`.

## Commands

```
llmoji install <provider>      write hook + register; idempotent
llmoji uninstall <provider>    inverse; idempotent (journal preserved)
llmoji status                  installed providers, journal sizes, paths
llmoji parse --provider <n> P  ingest a static export dump (claude.ai
                               or chatgpt conversations.json) into
                               ~/.llmoji/journals/
llmoji analyze [--notes …]     scrape + canonicalize + synthesize
[--backend …] [--model …]      → ~/.llmoji/bundle/. backend defaults
[--base-url …]                 to anthropic; openai uses Responses
                               API; local uses Chat Completions
                               (Ollama / vLLM / llama.cpp HTTP)
llmoji upload --target {hf,email} [--yes]   ship the bundle (HF: loose
                                            files via single atomic
                                            commit; email: tarball)
llmoji cache clear             wipe ~/.llmoji/cache/
```

`upload` requires `--target` and re-prompts before committing;
`--yes` skips the prompt for scripted use.

## Layout

```
llmoji/
  pyproject.toml               # PEP 621 + hatch dynamic version
                               # (regex-source from llmoji/__init__.py)
  README.md                    # public-prose, voice-rewritten
  CONTRIBUTING.md              # dev setup + adding-a-provider checklist
  SECURITY.md                  # privacy threat model
  LICENSE                      # MIT (PEP 639 SPDX)
  CLAUDE.md
  .github/                     # dependabot + PR/issue templates +
                               # ci.yml (lint + typecheck + test +
                               # build/wheel-import gate, 3.12 on
                               # ubuntu-latest; all four required by
                               # main branch protection) +
                               # release.yml (tag → PyPI → release)
  examples/                    # inspect_bundle.py (audit script);
                               # openclaw_hook.ts (generic-JSONL example)
  llmoji/
    py.typed                   # PEP 561 marker
    __init__.py                # public surface re-exports
    _util.py                   # atomic_write_text (tmp+rename),
                               # write_json, package_version,
                               # human_bytes, sanitize_model_id_for_path.
                               # Kept out of providers.base so the
                               # dependency graph stays tree-shaped.
    taxonomy.py                # KAOMOJI_START_CHARS + is_kaomoji_candidate
                               # + extract + KaomojiMatch (span-only)
                               # + canonicalize_kaomoji (rules A–P; frozen)
    synth_prompts.py           # locked cross-corpus prompts +
                               # DEFAULT_*_MODEL_ID constants
    synth.py                   # mask_kaomoji + cache helpers +
                               # Synthesizer base + Anthropic/OpenAI/
                               # Local backends + make_synthesizer
                               # (deferred SDK construction)
    scrape.py                  # ScrapeRow + iter_all chain helper
    sources/
      _common.py               # kaomoji_lead_strip — single source of
                               # truth for the on-the-way-in strip
      journal.py               # generic kaomoji-journal reader
      claude_export.py         # Claude.ai conversations.json (linear
                               # chat_messages array)
      chatgpt_export.py        # ChatGPT conversations.json — same
                               # filename, different schema (tree of
                               # nodes; walks mapping[current_node]
                               # up via parent)
    backfill.py                # one-shot transcript→journal replays
                               # for claude_code + codex + hermes;
                               # parity-tested against live hooks
    providers/
      base.py                  # Provider + JsonSettingsProvider +
                               # ProviderStatus + SettingsCorruptError
                               # + render helpers + JSON batch
                               # register/unregister/is_registered
                               # (one read-modify-write per install)
      claude_code.py           # ~/.claude/settings.json; shares the
                               # nudge template with codex
      codex.py                 # ~/.codex/hooks.json; codex_hooks
                               # feature flag is Stage::Stable +
                               # default_enabled in codex-rs/features
      hermes.py                # ~/.hermes/config.yaml YAML
                               # pre_llm_call + post_llm_call
                               # (marker-fenced hooks: stanza)
    _hooks/
      claude_code.sh.tmpl
      codex.sh.tmpl
      hermes.sh.tmpl                # post_llm_call (journal logger)
      _kaomoji_validate.sh.partial  # inlined as ${KAOMOJI_VALIDATE}
      _journal_write.sh.partial     # inlined as ${JOURNAL_WRITE}
      claude_codex_nudge.sh.tmpl    # UserPromptSubmit nudge —
                                    # byte-identical envelope, one
                                    # template for both
      hermes_nudge.sh.tmpl          # pre_llm_call nudge (bare
                                    # {context: ...} shape)
    paths.py                   # ~/.llmoji home, cache, bundle,
                               # journals, state.json (per-machine
                               # submission token). NOT an install
                               # registry — install state is read live
                               # from each harness's settings file
                               # by Provider.status().
    analyze.py                 # Stage A + B + bundle write. Buckets
                               # by (source_model, canonical) where
                               # source_model = ScrapeRow.model or
                               # falls back to ScrapeRow.source. Stage A
                               # dispatches on a ThreadPoolExecutor
                               # (default 2 workers, $LLMOJI_CONCURRENCY);
                               # cache appends serialize on the main
                               # thread via as_completed. Clears bundle
                               # of all top-level files AND any
                               # subdirs before writing.
    upload.py                  # tar + HF / email targets;
                               # BUNDLE_TOPLEVEL_ALLOWLIST +
                               # BUNDLE_DATA_SUFFIX enforce the flat
                               # shape. submitter_id() is public so
                               # analyze can stamp the manifest.
    cli.py                     # argparse entry, [project.scripts]
                               # llmoji. analyze takes
                               # --backend/--base-url/--model with env
                               # fallbacks; validates that local
                               # needs both --base-url and --model and
                               # that anthropic/openai accept neither.
  tests/
    test_public_surface.py     # cross-corpus invariant contract
    test_canonicalize.py       # parametrized rule-by-rule regression
                               # (~70 cases)
    test_chatgpt_export.py     # chatgpt tree-walker fixtures
    test_pipeline_parity.py    # bash live hook vs Python backfill on
                               # synthetic transcripts (claude_code +
                               # codex full parity, hermes excludes cwd
                               # via _HERMES_PARITY_FIELDS)
```

## Gotchas

### Journal-row contract: `assistant_text` never carries the kaomoji

Every source — bash hooks, Claude.ai export, ChatGPT export,
generic-JSONL contract — must persist `assistant_text` with the
leading kaomoji already stripped. The prefix lives separately in the
row's `kaomoji` field. Bash hooks enforce via jq's
`sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")`; the two
static-export readers route through
`llmoji.sources._common.kaomoji_lead_strip`, which wraps
`taxonomy.extract` and returns `(first_word, body)`. Future export
readers should reach for the shared helper rather than
re-implementing the dance — drift between sources is what the helper
exists to prevent.

`mask_kaomoji` consequently has a single branch: prepend `"[FACE] "`
to whatever's there. No source-shape dispatch. If you add a new
source, strip on the way in — don't push the special case into
`mask_kaomoji`.

### KAOMOJI_START_CHARS — single source of truth

Python: `llmoji.taxonomy.KAOMOJI_START_CHARS`. Shell:
`${KAOMOJI_START_CASE}` rendered at install time from the same set.
`is_kaomoji_candidate` validates Python-side; the rendered case
filter handles the shell-side first pass. If you find another copy
of the set, delete it and route through `llmoji.taxonomy`.

`is_kaomoji_candidate` enforces: length 2..32, first char in
`KAOMOJI_START_CHARS`, no ASCII backslash, no run of 4+ ASCII
letters. Bracket-balance is *not* enforced — real corpus output is
sometimes unbalanced (closing glyph isn't strictly the matching
bracket), and the length cap + 4-letter-run + backslash filters
together carry the prose-rejection role. `_leading_bracket_span`
still uses depth-walking to *locate* the closing bracket on
bracket-leading inputs, but falls back to a whitespace-delimited
word (capped at the length limit) when the depth-walker doesn't
close cleanly.

### Per-provider kaomoji capture — N rows per turn

All three providers emit one row per kaomoji-led model message. A
tool-heavy turn easily writes 5–10 kaomoji-led messages interleaved
with tool calls.

- **Claude Code**: each assistant content block (text, tool_use,
  thinking) is its own top-level transcript JSONL entry. The Stop
  hook scopes to entries at-or-after the latest real-user message
  (string content OR text-block array, NOT tool_result), walks every
  text-bearing non-sidechain entry in that window. Each entry's first
  text block runs through the kaomoji validator; non-kaomoji entries
  skip without aborting the walk. `BOUNDARY_TS` query slurps the
  transcript once; the per-entry walk is `jq -c` streamed line-by-line
  into a `while read` loop with `SKIP_ACTION=continue`.
- **Codex**: each model message is its own
  `event_msg.agent_message` event with `payload.message` carrying
  the text and `payload.phase` flagging `"commentary"` (progress) vs
  `"final_answer"` (closing). The Stop hook finds the latest
  `turn_context` index in the rollout (current turn boundary),
  slices forward, walks every agent_message in the slice. `user_text`
  resolves to the latest non-injected user response_item in the same
  slice.
- **Hermes**: walks `extra.conversation_history` (the full message
  list `post_llm_call` carries), slices from the latest user-role
  message to the end, emits one row per kaomoji-led non-empty
  assistant-role message. Tool-only assistant messages (`tool_calls`
  + empty/null `content`) are skipped naturally — the walker only
  takes string-typed non-empty content. `conversation_history` is
  always populated per `hermes-agent/run_agent.py:12492`
  (`list(messages)`).

Per-row invariants:

- `user_text` is resolved once per turn — every row from one turn
  carries the same originating prompt.
- The cache key hashes `(synth_model_id, canonical, user_text,
  assistant_text)`, so different assistant texts within a turn
  produce different keys (no collisions).
- Backfills (`backfill_codex` / `backfill_claude_code` /
  `backfill_hermes`) implement the same per-message walk and stay
  parity-tested against the live hooks via
  `tests/test_pipeline_parity.py`. `_PARITY_FIELDS` holds row-for-row
  in chronological order; Hermes uses `_HERMES_PARITY_FIELDS`
  (excludes `cwd`) because session JSON doesn't persist cwd —
  backfilled rows carry `""` there by design while the live hook
  stamps `Path.cwd()` from the agent process.

### Nudge hooks — what gives the corpus its size

Each provider ships a tiny **nudge** hook alongside the journal
logger. The nudge fires before each model turn (UserPromptSubmit on
Claude/Codex, `pre_llm_call` on Hermes) and injects a fresh "please
begin your message with a kaomoji that best represents how you feel"
reminder as additional context. Without it the model drifts away
from leading kaomoji over a long session.

Response shapes:

- **Claude Code + Codex**: `{"hookSpecificOutput": {"hookEventName":
  "UserPromptSubmit", "additionalContext": "<msg>"}}`. Codex's
  envelope is byte-identical (verified at
  `codex-rs/hooks/src/events/user_prompt_submit.rs`), so a single
  shared `claude_codex_nudge.sh.tmpl` serves both. `nudge_message`
  substitutes through `_shell_quote` into a bash single-quoted
  literal, so embedded apostrophes round-trip.
- **Hermes**: bare `{"context": "<msg>"}` — no envelope, returned by
  `pre_llm_call`, "the only hook whose return value is used."

The base `Provider` class exposes the nudge through
`nudge_hook_template` / `nudge_hook_filename` / `nudge_event` /
`nudge_message` class attrs and a `has_nudge` predicate. Providers
that opt in get the nudge written + registered automatically by
`install`, removed by `uninstall`, reported by `status`. Adding a
nudge to a future provider is four class-level attrs (and a
`_is_nudge_registered` override for non-JSON-settings providers).

### Sidechain strategy

- **Claude Code**: drop rows where `isSidechain` is true. Field-flag.
- **Codex**: no subagent concept. `collaboration_mode` is `"default"`
  for every observed turn_context.
- **Hermes**: **no viable filter on the current payload contract.**
  `subagent_stop` fires from the parent agent's process with the
  parent's `session_id` and no child id at all (verified at
  `hermes-agent/tools/delegate_tool.py:2120-2127`); `post_llm_call`
  doesn't expose `parent_session_id` either, so neither side carries
  enough info to filter children from a shell hook. Subagent
  `post_llm_call` events therefore land in the journal under their
  own session_ids until upstream gives us either (a) `subagent_stop`
  carrying the child id, or (b) `post_llm_call` exposing
  `parent_session_id` / `is_subagent`.

### Provider install refuses to clobber existing config

Three corruption paths are explicitly defended:

1. **Malformed `~/.claude/settings.json`** — `_load_json_strict`
   raises `SettingsCorruptError`; user fixes by hand before
   `install` will touch it.
2. **Malformed `~/.codex/hooks.json`** — same defense, same helper.
   Codex's `codex_hooks` feature flag is `Stage::Stable` +
   `default_enabled: true`, payload byte-identical to claude_code's,
   so the JSON helpers are reused.
3. **Existing top-level `hooks:` key in `~/.hermes/config.yaml`** —
   appending another would duplicate-key the YAML doc (silent
   last-write-wins). `HermesProvider._has_unmanaged_hooks_top_level`
   refuses.

In all three cases the user gets a `SettingsCorruptError` with path
+ reason. Edit the file (move-aside or merge by hand) and re-run.

The non-managed analogue — re-running `install` after already
installing once — is fully idempotent. The marker fence in hermes
makes the second install a no-op; the JSON-edit path in
claude_code/codex checks for an existing entry with our command
string and skips. Main and nudge dedup independently.

Settings writes go through `llmoji._util.atomic_write_text` (tmp
file + `os.replace`) so a power loss / SIGINT mid-write leaves the
user's settings file with either the old content or the new — never
half. The `upload` state.json (per-machine submission token) writes
the same way. JSON-settings providers also batch their main+nudge
edits into a single read-modify-write cycle per install (via
`_register_json_settings_batch`), so a SIGKILL between registering
the Stop hook and the UserPromptSubmit nudge can't half-install.

### Bundle is allowlisted, not just-ship-everything

Both upload paths enforce the flat allowlist:
`BUNDLE_TOPLEVEL_ALLOWLIST = ("manifest.json",)` plus
`BUNDLE_DATA_SUFFIX = ".jsonl"` for the per-source-model data files.
No subdirs, no symlinks, no other file types.

- `upload.tar_bundle()` (email target) raises `BundleAllowlistError`
  if anything else is present.
- `upload.upload_hf()` does the same pre-flight check AND passes
  `allow_patterns=["manifest.json", "*.jsonl"]` to
  `HfApi.upload_folder` as a second line of defense.

`analyze` clears the bundle dir of all top-level files AND all
subdirs before writing, so a clean run produces exactly the flat
shape. The three together mean stale per-instance descriptions,
user-added notes, hidden-state caches, leftover subfolders, etc.
cannot accidentally leak through `upload`.

### HF upload is loose files, not a tarball

`upload --target hf` pushes `manifest.json` plus each
`<source-model>.jsonl` as loose files at
`contributors/<hash>/bundle-<ts>/` via `HfApi.upload_folder` (single
atomic commit). The dataset card has a `configs:` YAML pointing at
`contributors/**/*.jsonl`, which is what the auto-loader needs to
surface the dataset viewer. The `**` is recursive (matches
`bundle-<ts>/<model>.jsonl`); the `*.jsonl` suffix matches every
data file without picking up manifests. Tarballs trigger HF's
WebDataset auto-detection and break the viewer (WebDataset expects
shared-prefix archives).

Email target keeps `tar_bundle` because a single attachment is what
the recipient wants; `~/.llmoji/bundle-<ts>.tar.gz` is now an
email-only artifact.

### Hermes payload contract — source-verified

The hermes provider installs **two** hooks:

- `~/.hermes/agent-hooks/post-llm-call.sh` — main journal logger
  (walks `extra.conversation_history`; see "Per-provider kaomoji
  capture" above).
- `~/.hermes/agent-hooks/pre-llm-call.sh` — nudge that injects the
  kaomoji-reminder context via `{context: "<msg>"}`.

Both registered in `~/.hermes/config.yaml` under `hooks:`, inside
our managed marker fence so re-running install is idempotent and
uninstall removes the stanza cleanly.

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
(`original_user_message` at the call site), so
`system_injected_prefixes` stays `[]`. If real-traffic inspection
later shows leaked injection prefixes, populate the list and
re-render — the bash hook picks the same list up via
`${INJECTED_PREFIXES_FILTER}`.

[hermes-hooks]: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/

### Cache directory is leakier than the bundle

`~/.llmoji/cache/per_instance.jsonl` holds synthesizer-paraphrased
descriptions of single user turns, keyed by `(synth_model_id,
canonical, user_text, assistant_text)`. Each row IS one user-turn
paraphrase, so for a topic-narrow corpus a singleton row can leak
specifics of that turn through. Mitigations:

- Cache is **never** bundled or shipped. Only the per-canonical-face
  Stage B synthesis lands in the bundle.
- `llmoji status` prints cache size + entry count.
- `llmoji uninstall <provider>` does NOT touch the cache (the user
  may re-install). `llmoji cache clear` is the explicit wipe.

The bundle is the only thing that leaves the machine; the inspection
gap (`analyze` prints a per-face preview, `upload` re-prompts) is
the consent boundary.

### Codex `transcript_path` carries the rollout JSONL

Used to resolve `user_text` (Codex injects AGENTS.md /
`<environment_context>` / `<INSTRUCTIONS>` as user-role response_items
at session start; we walk the rollout to find the latest real user
turn, dropping those prefixes defensively). `llmoji.backfill`
mirrors this.

### Generic JSONL contract for unsupported harnesses

Motivated users on harnesses we don't ship a first-class adapter for
(notably OpenClaw — TS-shaped hooks taking the payload as a function
argument, not stdin) can write directly to
`~/.llmoji/journals/<name>.jsonl` against the canonical 6-field
schema. `llmoji analyze` picks them up automatically alongside
managed providers' journals. OpenClaw first-class is post-v1.0;
worked example lives at `examples/openclaw_hook.ts`.

### HF dataset card is a separate hand-maintained surface

The user-facing dataset card at
https://huggingface.co/datasets/a9lim/llmoji is a separate document
from anything in this repo. It re-states the bundle schema and
privacy model in user-facing prose so contributors can decide
whether to submit before they've ever touched the package README.

Two coupling points:

- **Schema changes need both updates.** Any change to
  `manifest.json` or `<source-model>.jsonl` field names is a
  cross-corpus invariant change, so it wants a hand-edit on the HF
  dataset card so the field-by-field schema documentation doesn't go
  stale. The card is editable in-place via the HF web UI; the
  canonical surface lives there, not in this repo.
- **License split.** The package code is GPL-3.0-or-later; the
  shared corpus on HF is CC-BY-SA-4.0. `llmoji upload --target hf`
  contributes a bundle under CC-BY-SA-4.0, and the package README's
  License section calls this out so contributors aren't surprised.
  `llmoji-study` is CC-BY-SA-4.0 — research artifact (writeups,
  figures, analysis pipelines) rather than distributed program, so
  matching the corpus license keeps derivative work under one
  consistent set of terms.

## Conventions

- Single venv at `.venv/`, pip not uv. `pip install -e ../llmoji`
  during dev; PyPI install at freeze.
- `main` is branch-protected: PR-only (no direct pushes, including
  for admins), all four CI jobs (lint / typecheck / test / build)
  required green, branch up-to-date with main, conversation
  resolution required, force-push and deletion blocked. Day-to-day
  work lands on `dev`; merge to main via PR.
- `~/.llmoji` is the on-disk root for everything the package
  manages; tests can override via `$LLMOJI_HOME`.
- Hook templates are bash, syntactically validated by `bash -n` in
  the test suite (`test_hook_templates_render_to_valid_bash_substitutions`)
  so a template-edit regression fails CI rather than silently inside
  a user's harness post-install. Don't introduce non-bash hook
  formats; if a harness needs one, that's a post-1.0 first-class
  adapter, and the generic-JSONL-append contract is the path until
  then.
- Stage-A synth calls run on a small thread pool (default 2,
  `--concurrency` flag or `$LLMOJI_CONCURRENCY` to override). Both
  Anthropic and OpenAI SDKs use thread-safe httpx clients; cache
  writes serialize on the main thread after futures complete, in
  deterministic walk order, so the cache file's row order matches
  the bundle's Stage-B input order. Default is 2
  because the org-level Haiku rate limit is 50 req/min; 4 concurrent
  workers reliably trip it on a multi-hundred-row backfill, and the
  SDK's `max_retries=8` exponential backoff (set explicitly in
  `AnthropicSynthesizer.__init__` and `OpenAISynthesizer.__init__`,
  vs the SDK default of 2) recovers but burns wallclock.
  `INSTANCE_SAMPLE_CAP` is 4 — popular faces get capped, rare faces
  fully sampled. Same value as Eriskii's original Claude-faces work,
  kept for cross-corpus comparability.
- Public-API freeze: anything in §"Cross-corpus invariant surface"
  is a cross-corpus invariant; bumping wants a hand-edit on the HF
  dataset card and a flag in the PR body. Internal helpers
  (`llmoji._util`, leading-underscore names in `llmoji.providers.base`,
  `llmoji.synth.cache_key`, the synth backend classes, etc.) are
  free to evolve.
