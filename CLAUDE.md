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
      claude_code.py       # JSON Stop+UserPromptSubmit provider
                           # (~/.claude/settings.json); shares the
                           # nudge template with codex
      codex.py             # JSON Stop+UserPromptSubmit provider
                           # (~/.codex/hooks.json); the codex_hooks
                           # feature flag is Stage::Stable, default-on
                           # in codex-rs/features
      hermes.py            # YAML pre_llm_call + post_llm_call +
                           # subagent_stop tri-hook provider
                           # (~/.hermes/config.yaml, marker-fenced
                           # hooks: stanza)
    _hooks/                # bash hook templates (importlib.resources
                           # data); rendered at install time
      claude_code.sh.tmpl
      codex.sh.tmpl
      claude_codex_nudge.sh.tmpl    # UserPromptSubmit nudge — shared
                                    # between claude_code + codex (the
                                    # response envelope is byte-identical)
      hermes.sh.tmpl                # post_llm_call (journal logger)
      hermes_nudge.sh.tmpl          # pre_llm_call nudge (bare
                                    # {context: ...} shape)
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
                           # mask_kaomoji unified prepend contract).
    test_canonicalize.py   # parametrized rule-by-rule regression
                           # tests for canonicalize_kaomoji + extract
                           # + is_kaomoji_candidate. Each rule case is
                           # its own pytest line. ~70 cases total.
```

## Gotchas

### Journal-row contract: `assistant_text` never carries the kaomoji

Every source — bash hooks, Claude.ai export reader, generic-JSONL
contract — must persist `assistant_text` with the leading kaomoji
already stripped. The prefix lives separately in the row's
`kaomoji` field. The bash hooks enforce this via jq's
`sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")`;
`llmoji.sources.claude_export` does the equivalent in Python after
`taxonomy.extract`.

`mask_kaomoji` consequently has a single branch: prepend
`"[FACE] "` to whatever's there. No source-shape dispatch, no
substitute-in-place fallback. If you add a new source, strip on
the way in — don't push the special case into `mask_kaomoji`.

The cache key hashes raw `(canonical, user_text, assistant_text)`;
existing cache entries from prior `parse + analyze` runs of an
export will miss after this normalization (export rows now hash
without the kaomoji). One-time re-call cost on the next analyze.

Pre-package, the live-hook branch of `mask_kaomoji` fell through
to `return text` and Haiku got a prompt promising a `[FACE]` that
wasn't in the body — affected every journal row. The unified
contract makes that class of bug structurally impossible.

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

- **Claude Code**: kaomoji on the **first** text-bearing entry of
  the current turn. Claude Code persists each assistant content
  block — text, tool_use, thinking — as its OWN top-level entry in
  the transcript JSONL; one turn produces many assistant entries.
  The hook scopes to events at-or-after the latest real-user
  message (string content OR text-block array, NOT tool_result),
  picks the first text-bearing assistant entry, and reads its
  first text block. Naive `last(assistant)` only catches turns
  that finish on text and never resume — every text-then-tools
  turn was getting silently dropped pre-fix. (The original gotcha
  comment claiming "one event with interleaved text + tool_use +
  text content blocks" described the API response shape, not the
  on-disk transcript shape — they don't match.)
- **Codex**: kaomoji on the **last** agent message of a turn. Each
  agent message is its own `event_msg.agent_message` event;
  progress messages come first. The hook keys on
  `task_complete.last_agent_message`, which Codex itself surfaces
  on the Stop payload — no transcript walking, harness curates the
  field.
- **Hermes**: **single** final-text field per turn
  (`extra.assistant_response`). No first/last ambiguity, harness
  curates.

Codex + Hermes are structurally immune to the Claude Code bug
because their Stop payloads carry the final assistant text as a
named field. Claude Code's only delivers `transcript_path`, so
the hook owns the find-the-right-block job.

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
2. **Malformed JSON in `~/.codex/hooks.json`** — same defense as
   claude_code, same helper. Codex used to live behind a
   marker-fenced `[hooks.stop]` stanza in `~/.codex/config.toml`,
   but the canonical home for codex hook registration is
   `~/.codex/hooks.json` — Claude-style payload, same shape as
   claude_code, verified at the `codex_hooks` feature flag in
   `codex-rs/features` (`Stage::Stable`, `default_enabled: true`).
   The TOML path warned when both representations were present, so
   we standardize on the JSON file and reuse the JSON helpers.
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

Settings writes go through `_atomic_write_text` (tmp file +
`os.replace`) so a power loss / SIGINT mid-write leaves the user's
settings file with either the old content or the new — never half.
The `upload` state.json (per-machine submission token) writes the
same way.

### Bundle is allowlisted, not just-ship-everything

Both upload paths enforce `BUNDLE_ALLOWLIST` (`manifest.json`,
`descriptions.jsonl` — the v1.0 frozen schema):

- `upload.tar_bundle()` (used by the email target) raises
  `FileExistsError` if the bundle dir holds anything else.
- `upload.upload_hf()` does the same pre-flight check and ALSO
  passes `allow_patterns=list(BUNDLE_ALLOWLIST)` to
  `HfApi.upload_folder` as a second line of defense.

`analyze` clears loose files in the bundle dir before writing, so
a clean run produces exactly the two-file schema. The three
together mean stale per-instance descriptions, user-added notes,
hidden-state caches, etc. cannot accidentally leak through
`upload`.

### HF upload is loose files, not a tarball

`upload --target hf` pushes `manifest.json` + `descriptions.jsonl`
as loose files at `contributors/<hash>/bundle-<ts>/` via
`HfApi.upload_folder` (single atomic commit). The dataset card on
the HF side has a `configs:` YAML pointing at
`contributors/**/descriptions.jsonl`, which is what the auto-loader
needs to surface the dataset viewer; uploading as a tarball
triggered HF's WebDataset auto-detection and broke the viewer
(WebDataset expects shared-prefix archives, our two-file bundles
don't fit).

Email target keeps `tar_bundle` because a single attachment is
what the recipient wants. The local tarball at
`~/.llmoji/bundle-<ts>.tar.gz` is now an email-only artifact.

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

- Claude Code persists each content block — text, tool_use,
  thinking — as its OWN top-level transcript entry. The
  kaomoji-prefixed reply is on the first text-bearing entry of
  the current turn; the live hook + backfill both scope to
  events at-or-after the latest real-user message (string
  content OR text-block array, NOT tool_result) and pick the
  first text-bearing one.
- Codex emits each agent message as a separate
  `event_msg.agent_message` event; the kaomoji-bearing summary
  lands last as `task_complete.last_agent_message`.

The Codex hook + Codex backfill both key on `last_agent_message`,
NOT on the first agent_message — flipping that would miss every
multi-step turn's kaomoji. The Claude Code hook + backfill both
key on the first text-bearing entry of the current turn — flipping
to `last(assistant)` would miss every turn that resumes tool work
after the kaomoji-led reply (the v1.0 pre-fix bug, observed
silently dropping ~6 hours of journal entries on real sessions).

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
  `bundle/manifest.json` or `bundle/descriptions.jsonl` field names
  is a v1.0 frozen-surface break (see §"v1.0 frozen public
  surface"), so it bumps the package major version AND wants a
  hand-edit on the HF dataset card so the field-by-field schema
  documentation doesn't go stale. The card is editable in-place via
  the HF web UI; the canonical surface lives there, not in this
  repo.
- **License split.** The package code is GPL-3.0-or-later; the
  shared corpus on HF is CC-BY-SA-4.0. `llmoji upload --target hf`
  contributes a bundle under CC-BY-SA-4.0, and the package README's
  License section calls this out so contributors aren't surprised.
  `llmoji-study` is AGPL-3.0-or-later (the network-use clause
  matters if anyone ever runs the research pipeline as a service).

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
