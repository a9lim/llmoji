# AGENTS.md

## What this is

`llmoji` is a small provider-agnostic CLI for collecting kaomoji
journals from coding agents (Claude Code, Codex, Hermes, Grok Build,
Antigravity, opencode, openclaw), distilling them into per-(source-model, canonical-face)
descriptions via the user's chosen synthesis backend, and submitting
privacy-preserving aggregates to a shared HF dataset for cross-corpus
research. Companion to the research-side
[`llmoji-study`](https://github.com/a9lim/llmoji-study) repo, where
all probe / hidden-state / embedding / axis-projection / figure work
lives.

Data-layer-only: zero dependency on saklas, torch, or matplotlib.
Runtime deps are `anthropic` (default synth backend), `openai`
(`--backend openai` Responses-API path AND `--backend local`
OpenAI-compatible Chat-Completions path), `huggingface_hub` (upload
target), and `ruamel.yaml` (parsing-only — used by the hermes
provider for line/col marks; we never call `yaml.dump` on the loaded
doc, see "Hermes parsing-only ruamel" gotcha for the data-corruption
story that motivates it).

## Pipeline

```
harness hook  →  ~/.<harness>/kaomoji-journal.jsonl   (6-field rows)
              →  llmoji.sources.* readers + taxonomy.canonicalize
              →  per-cell single-stage synthesize (cached, structured
                 output against the locked SYNTHESIS_SCHEMA)
              →  ~/.llmoji/bundle/  (manifest.json + <model>.jsonl)
              →  llmoji upload --target {hf,email}
```

The bundle on disk between `analyze` and `upload` is the deliberate
inspection gap — the user `cat`s each `<source-model>.jsonl` before
deciding to ship.

### HookInstaller abstraction

`llmoji.providers.HookInstaller` is the base class for **bash-hook**
providers, one subclass per first-class harness. JSON-settings
providers (`ClaudeCodeProvider`, `CodexProvider`) inherit from
`JsonSettingsHookInstaller`, which supplies the default `_register`
/ `_unregister` / `_check_registrations` against any
`settings.json`-shaped file. YAML-settings providers
(`HermesProvider`) override the three.

`PluginInstaller` is a sibling base for **TS-plugin** providers
(opencode, openclaw) — harnesses whose plugin SDKs are TS-only with
no shell-hook escape hatch. It subclasses `HookInstaller` for
type-compatibility (PROVIDERS dict and `ProviderStatus` stay
one-shape) but writes rendered TypeScript instead of bash. opencode
auto-loads from `~/.config/opencode/plugins/llmoji.ts` (file
presence = registered); openclaw writes a bundle at
`~/.openclaw/plugins/llmoji-kaomoji/` and edits
`~/.openclaw/config.json` to set
`plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess = true`.

Each provider declares: `hooks_dir`, `settings_path`, `journal_path`,
`main_event`, `system_prompt_doc_path`, `skip_action` (`continue`
for the per-message walks; `echo '{}'; exit 0` for hermes's
stdout-JSON contract), `system_injected_prefixes`, optional nudge
attrs, and `hard_unsupported_reason` (set when the harness has no
per-turn injection point — grok; `install_hard` then raises
`HardInstallNotSupportedError` and no-arg `install --hard` skips the
provider by name). `home_dir` (the harness root: drives
`is_present`, the CLI's autodetect listing, and `import`'s source
roots) defaults to `settings_path.parent`; grok, antigravity,
openclaw and the plugin base override it.

Two more bash-hook providers own their registration file outright
instead of merging into a shared one: grok writes
`~/.grok/hooks/llmoji.json` (Grok scans `~/.grok/hooks/*.json`; the
JSON-settings helpers apply unchanged) and antigravity writes a whole
plugin at `~/.gemini/config/plugins/llmoji/` (`plugin.json` +
`hooks.json` keyed by hook name, plus both scripts). Uninstall
deletes what's left once our entry is gone.

Path defaults are declared with `home_path(".claude", "hooks")`, a
descriptor that reads `Path.home()` on every access. A class-body
`Path.home() / ...` is evaluated once at import, before the test
suite's `HOME` sandbox exists, so every default pointed at the
developer's real home whatever the fixture did — the 2.1.2 guard
covered runtime `Path.home()` calls only.
`test_every_default_path_follows_the_sandbox_home` fails on any
provider path that escapes the sandbox; use `home_path` for new
ones.

Bash hook templates live under `llmoji/_hooks/`. Two shared partials
inline into every main hook: `_kaomoji_validate.sh.partial`
(extractor + validator, gets `${KAOMOJI_START_CASE}` from
`KAOMOJI_START_CHARS` and `${PYTHON_INTERPRETER}` for the round-6
fallback) and `_journal_write.sh.partial` (the `jq -nc … >>
$JOURNAL_PATH` tail). `render_hook()` runs `safe_substitute` twice
— once on each partial with its own placeholders, once on the main
template — because `safe_substitute` is single-pass and the partials'
own `${...}` references wouldn't survive a one-pass render.

### Synthesis backends

Three concrete backends, all routed through
`llmoji.synth.make_synthesizer`:

- `AnthropicSynthesizer` — `anthropic.Anthropic.messages.create`,
  `max_retries=8`, default `DEFAULT_ANTHROPIC_MODEL_ID`.
- `OpenAISynthesizer` — `openai.OpenAI.responses.create` (Responses
  API, OpenAI's recommended path), `.output_text` accessor, default
  `DEFAULT_OPENAI_MODEL_ID`.
- `LocalSynthesizer` —
  `openai.OpenAI(base_url=..., api_key="ollama").chat.completions.create`.
  Chat Completions because Ollama / vLLM / llama.cpp HTTP all expose
  Chat-Completions-shaped endpoints. No default model id; user must
  pass `--model`.

All three defer SDK-client construction to first `.call()` /
`.call_structured()` so the factory has no env-var dependency at
construction time. The structured-output paths use each backend's
native JSON-schema-with-enum support — verified against the
installed SDK shapes:

- **Anthropic** — `messages.create(..., output_config={"format":
  {"type": "json_schema", "schema": SYNTHESIS_SCHEMA}})`. The
  format object has exactly two fields (`type` and `schema`) per
  `anthropic.types.json_output_format_param.JSONOutputFormatParam`
  — no `name` / `strict` / `description`.
- **OpenAI Responses** — `responses.create(..., text={"format":
  {"type": "json_schema", "name": "synthesis", "schema":
  SYNTHESIS_SCHEMA, "strict": True}})`. `name` is required;
  `strict` enables strict schema adherence.
- **Local Chat Completions** — tries `chat.completions.create(...,
  response_format={"type": "json_schema", "json_schema": {"name":
  ..., "schema": ..., "strict": True}})` first; on
  `BadRequestError` / `UnprocessableEntityError` falls back to
  appending `Output strictly as JSON matching this schema:\n
  {schema}` to the prompt and parsing the unconstrained reply.

### Single-stage synthesis pipeline

For each `(source_model, canonical_kaomoji)` cell, sample up to
`INSTANCE_SAMPLE_CAP` rows (deterministic seed
`f"{INSTANCE_SAMPLE_SEED}:{source_model}:{canonical}"`), mask the
leading kaomoji to `[FACE]`, render the samples into the locked
`SYNTHESIZE_PROMPT` as numbered `[Sample N]\nUser: ...\nAssistant:
[FACE] ...` blocks (User: line omitted for empty user_text), and
call the synthesizer's `call_structured` with `SYNTHESIS_SCHEMA`.
The single call returns a structured object
`{primary_affect, stance_modality_function}` drawn purely from
the locked `LEXICON`.

Cache keyed by `sha256(synth_model_id + "\0" + backend + "\0" +
base_url + "\0" + source_model + "\0" + canonical + "\0" +
sample_set_hash)[:16]` at `~/.llmoji/cache/per_cell.jsonl`. The
`sample_set_hash` is a length-prefixed-framed sha256 of the sorted
`(user, assistant)` pairs that fed the call, so a re-import that
changes which 4 rows get sampled cache-misses on affected cells
while stable cells keep hitting. Switching synth model OR backend
OR (for `local`) endpoint misses cleanly. Cache-miss API calls
dispatch on a small thread pool; the deterministic walk order
makes re-runs produce byte-identical bundle output.

The v1 pipeline was two-stage (per-instance describe at Stage A,
per-cell prose-from-prose synthesize at Stage B) and produced
free-form prose that clustered as noise in PCA. v2 collapses both
into one call that sees all samples for a cell at once, drops
prose entirely in favor of pure adjective bags from the locked
LEXICON, and ships ~5× fewer API calls per analyze. The legacy
`per_instance.jsonl` cache from v1 is orphaned by v2 — first run
prints a one-line cleanup notice; `llmoji cache clear` wipes both.

Embedding / axis projection / clustering / figures are
research-side at `llmoji-study`.

## Cross-corpus invariant surface

The HF dataset's aggregation rules pin against everything below.
Bumping any of these is a cross-corpus change — flag in the PR body
and update the HF dataset card to match.

- **`llmoji.taxonomy`**:
  - `KAOMOJI_START_CHARS` — Path A leader set, broadened across the
    2.0 sweep (rounds 1–5 added wing/hug/sparkle, Greek/Latin
    extensions, box-drawing, corner brackets, music notes, hearts,
    stars, alt-bear, flowers, heart/star variants, flex/strong-feel,
    tortoise-shell, reference mark, Oriya cradle). Round 6 added
    Path B `_looks_like_bare_kaomoji`. Round 7 broadened Path B with
    `ω` / `oOwW` mouth chars (catches `>ω<`, `^o^`, `OwO`, `\o/`),
    `<>` Western eyes (catches `<3`, `>:(`), eyebrow-prefix Western
    (`>:(`, `<:(`), cat-wrap `=...=` (`=^.^=`), Korean closed-eye
    doubles (`ㅠㅠ`, `ㅜㅜ`), and the `XD`-style extension to
    `[xX][DPpOo3]` (catches `xP`, `X3`). Round 8 added `つ` to the
    arm-strip sets so the offering-hands gesture `(つface)つ`
    canonicalizes to the bare face. Round 9 broadened in three small
    directions: `ヮ` (KATAKANA SMALL WA) and `〜` (WAVE DASH) joined
    the Path B mouth class for bare `^ヮ^` / `T〜T`; U+FE00–U+FE0F
    variation selectors joined the rule-A invisibles so emoji-vs-text
    presentation forms (`♥️` / `♥︎`) collapse to bare `♥`; and `づ`
    (HIRAGANA ZU) joined both arm-strip sets as the voiced cousin of
    round-8 `つ` for `(づface)づ`. Plus four more typo-sub folds:
    `〜→~`, `［→[`, `］→]`, `｜→|` (the last three are FF0x/FF1x
    halfwidth/fullwidth-pair fills, the first is a parallel ASCII-fold
    for the wave-dash glyph).
  - Round-7 false-alarm guards: an all-ASCII-alpha reject in the
    symmetric branch (rejects `lol` / `mom` / `pop` / `awa` /
    palindrome prose), an explicit `_UWU_FACES` allow-set so
    canonical anime shapes (`OwO`, `UwU`) survive the guard, a
    same-char-as-eye reject in Western (rejects `>>` / `<<` / `==`),
    and a `_has_kaomoji_content` check on Path A that requires at
    least one non-letter non-bracket char (rejects `[a]`, `(b)`,
    `(test)`).
  - `is_kaomoji_candidate` validator contract: length 2..32, no
    backslash at position ≥1 (position 0 is the wing-hand pattern),
    no run of 4+ ASCII letters, AND (Path A leader + content-bearing
    OR Path B shape). Bracket balance is *not* enforced — real
    corpus output is sometimes unbalanced; the length cap +
    4-letter-run + backslash + content filters carry the
    prose-rejection role.
  - `canonicalize_kaomoji` rules A–P + the 2.0 paired-arm strip
    sweep (`_ARM_OUTSIDE` / `_ARM_OUTSIDE_LEAD` cover wing-hand,
    hugging arms, sparkle, shocked sigma, horn-fingers, kissing,
    raised hands, paired-arm leaders `٩()۶ / ᕕ()ᗒ / ໒()७`,
    raised-hands katakana, box-drawing pose pairs, the `¯\_(ツ)_/¯`
    shrug, corner-bracket / standing-pose / music / heart / star
    decorators, plus round-5 flower / heart-variant / star-variant /
    quarter-note / flex / strong-feel / tortoise-shell /
    reference-mark / Oriya cradle decorators, plus round-8 `つ`
    offering-hands on both arm sets, plus round-9 `づ` voiced-offering
    on both arm sets).
  - `extract` / `KaomojiMatch` (span-only — affect labels are
    research-side at `llmoji_study.taxonomy_labels`).
  - Bear faces `ʕ•ᴥ•ʔ` / `ʢ◉ᴥ◉ʡ` are special: their bracket pairs go
    in `_OPEN_BRACKETS`/`_CLOSE_BRACKETS` for depth-walk recognition
    but stay OUT of arm-strip — the whole bear is the kaomoji, no
    inner `(...)` to fall back to. Same for corner-bracket-only
    standalone faces (`「・_・」`, `〔・_・〕`).
- **`llmoji.synth_prompts`**: `LEXICON` (48-entry locked
  vocabulary — circumplex anchors tagged with HP / LP / HN-D /
  HN-S / LN / NB primary quadrant + extension axes tagged
  `functional` / `stance` / `modality` / `confidence`),
  `LEXICON_VERSION` (bundle-stamped, bumped only on lexicon
  rotation — independent of package version), `SYNTHESIS_SCHEMA`
  (the JSON schema the model's structured output is validated
  against — disjoint enum sets across `primary_affect` and
  `stance_modality_function`, no free-form, no distinctive
  phrase), `SYNTHESIZE_PROMPT` (single-stage; takes `{samples}`
  rendered as numbered `[Sample N]` blocks),
  `DEFAULT_ANTHROPIC_MODEL_ID` (pinned Haiku snapshot),
  `DEFAULT_OPENAI_MODEL_ID` (pinned GPT-5.4 mini snapshot),
  `NUDGE_MESSAGE` (the single canonical kaomoji-leading reminder
  — one sentence + an explicit uncertainty license; changing it
  is a cross-corpus bump). 2.1 dropped the 2.0 `--long` second
  variant — the strong introspection framing drove refusals and
  the bundle never tagged rows by variant, so the short/long
  split was never analytically recoverable. The LEXICON itself
  is snapshot-pinned in
  `tests/test_lexicon_evolution.py`; any rotation requires both
  the snapshot update and a `LEXICON_VERSION` bump in the same
  commit.
- **6-field journal row schema** (on-disk JSONL):
  `{ts, model, cwd, kaomoji, user_text, assistant_text}`. Since 2.2
  `ts` is the message's own transcript time wherever the source has
  one (claude_code, codex, grok, antigravity), hook fire time
  otherwise (hermes, and claude_code's `last_assistant_message`
  fallback). `ts` never ships in a bundle. The
  in-memory `llmoji.scrape.ScrapeRow` (7 fields: `source, model,
  timestamp, cwd, assistant_text, first_word, surrounding_user`) is
  free to evolve.
- **System-injection prefix lists** per provider (in
  `llmoji.providers.{claude_code,codex,hermes}`).
- **`HookInstaller` / `PluginInstaller` interfaces.** 2.2 added
  `home_dir` and `hard_unsupported_reason`. 2.0 split the
  install lifecycle into `install_hard` and `install_soft` —
  mutually exclusive *placement* modes that share the journal-write
  hook. Both modes install the Stop / `post_llm_call` hook so
  capture works under either; the modes only differ in where the
  kaomoji-leading reminder is delivered. `uninstall` undoes both.
- **`--soft` vs `--hard` placement** (mutually exclusive, exactly one
  required):
  - `--hard`: journal-write hook + per-turn nudge hook
    (`UserPromptSubmit` on claude_code/codex, `pre_llm_call` on
    hermes; baked into the rendered TS plugin for opencode/openclaw).
    The v1 behavior.
  - `--soft`: journal-write hook + appends a `## Kaomoji` heading +
    the nudge text to the harness's persistent system-prompt doc.
    No per-turn nudge hook. For TS plugin providers,
    `render_plugin_template(install_nudge=False)` strips the
    per-turn nudge block out of the rendered TS via
    `// BEGIN NUDGE HOOK` / `// END NUDGE HOOK` fences.
- **Per-harness system-prompt doc paths**:
  - `claude_code` → `~/.claude/CLAUDE.md`
  - `codex` → `~/.codex/AGENTS.md`
  - `hermes` → `~/.hermes/SOUL.md` (voice slot — paired with
    AGENTS.md for procedure)
  - `grok` → `~/.grok/AGENTS.md` (global rules, read before the
    repo-root-to-cwd walk)
  - `antigravity` → `~/.gemini/GEMINI.md` (global rules; NOT under
    `~/.gemini/config/`, which Antigravity's policy checker guards)
  - `opencode` → `~/.config/opencode/AGENTS.md`
  - `openclaw` → `~/.openclaw/workspace/SOUL.md`
- **Soft-doc shape**: plain markdown, no comment fences. Block is
  `## Kaomoji\n\n<message>` appended at EOF with a blank-line
  separator. Uninstall removes the block by exact string match
  against the canonical wording plus the legacy wordings older
  versions shipped (`_LEGACY_NUDGE_MESSAGES`); a hand-edited body
  falls through and survives uninstall (conservative on the
  user's prose). The `## Kaomoji` heading is the cross-corpus anchor;
  the former `# Kaomoji` shape remains recognized for migration and
  uninstall so existing soft installs are not stranded.
- **The seven first-class providers**: `claude_code`, `codex`,
  `hermes`, `grok`, `antigravity` (bash hooks); `opencode`,
  `openclaw` (TS plugins). `providers_seen` in shipped bundles names
  these directly (as `<name>-hook`); 2.2 added `grok` and
  `antigravity`, flagged for the dataset card.
- **Bundle schema**:
  - `manifest.json` keys: `llmoji_version`, `lexicon_version`,
    `synthesis_model_id`, `synthesis_backend`, `submitter_id`,
    `generated_at`, `providers_seen`, `model_counts`,
    `total_synthesized_rows`, `notes`.
  - one `<sanitized_source_model>.jsonl` per source model, each
    row `{kaomoji, count, synthesis: {primary_affect:
    [1..3 from CIRCUMPLEX_ANCHORS], stance_modality_function:
    [2..4 from EXTENSION_AXES]}}`. The two adjective arrays draw
    from disjoint enum sets; no free-form, no distinctive phrase.
  - filename stem = `sanitize_model_id_for_path(source_model)`
    (lowercase, `/` → `__`, `:` → `-`).

Free to evolve without bumping invariant: cache key derivation,
`INSTANCE_SAMPLE_CAP` / `INSTANCE_SAMPLE_SEED`, internal flag names
beyond `--target {hf,email}` and `--backend {anthropic,openai,local}`.

## Commands

```
llmoji install <provider> --hard
                          install journal-write hook + per-turn nudge
                          hook (the v1 behavior).
llmoji install <provider> --soft
                          install journal-write hook + append
                          "## Kaomoji" + the nudge wording to the
                          harness's system-prompt doc. No per-turn
                          nudge hook.
                          --soft and --hard are mutually exclusive;
                          exactly one required. Both capture journal
                          data; only the placement of the leading-
                          kaomoji reminder differs.
llmoji install --soft|--hard [--yes]
                          no-arg autodetect: install for every harness
                          whose home dir exists. Prompts unless --yes.
                          Partial success OK — one corrupt config
                          doesn't kill the rest of the batch.
llmoji uninstall <provider>
                          inverse; idempotent (journal preserved).
llmoji uninstall [--yes]  no-arg autodetect uninstall.
llmoji status [--stats] [--top N] [--provider N] [--json]
                          installed providers, journal sizes, paths,
                          health checks (stale-hook, settings parse).
                          --stats walks journals for kaomoji
                          frequency tables + row schema validation.
llmoji parse --provider <n> P
                          ingest a static export dump into
                          ~/.llmoji/journals/. Sources: claude.ai
                          (combined conversations.json), claude.ai-
                          alternate (per-conversation .json files +
                          export_summary.json sibling; carries
                          top-level model field), chatgpt
                          (conversations.json), gemini (AI Studio
                          chunkedPrompt or Takeout MyActivity),
                          openhands (per-event JSON).
llmoji import [<provider>] [--since <ISO>] [--dry-run] [--yes]
                          replay native session/transcript files into
                          the live journal. Dedup-aware merge against
                          (ts, model, assistant_text), plus a one-to-one
                          fold of pre-2.2 fire-time live rows; atomic.
                          Stop the harness first. No-arg autodetects
                          every importable harness (claude_code, codex,
                          hermes, grok — TS plugins and antigravity
                          don't expose replayable transcripts).
                          Recommended after every taxonomy bump.
llmoji analyze [--notes …] [--backend …] [--model …] [--base-url …]
                          scrape + canonicalize + synthesize →
                          ~/.llmoji/bundle/. backend defaults to
                          anthropic; openai uses Responses API; local
                          uses Chat Completions.
llmoji analyze --dry-run  print plan + token + cost estimate (char/4
                          heuristic + per-1M rate table; approximate).
llmoji upload --target {hf,email} [--yes]
                          ship the bundle. HF: per-submission branch
                          via shared encrypted credential (user's HF
                          token used only for whoami proof-of-life,
                          discarded). email: tarball + mailto.
llmoji cache clear        wipe ~/.llmoji/cache/
```

## Layout

```
llmoji/
  pyproject.toml         # PEP 621 + hatch dynamic version
  README.md              # public-prose, voice-rewritten
  CONTRIBUTING.md        # dev setup + adding-a-provider checklist
  SECURITY.md            # privacy threat model
  AGENTS.md              # this file
  .github/               # CI (lint+typecheck+test+build, all four
                         # required by branch protection on main) +
                         # release.yml (tag → PyPI)
  examples/              # inspect_bundle.py audit script
  llmoji/
    __init__.py          # public surface re-exports
    _util.py             # atomic_write_text, write_json,
                         # package_version, journal_line_dict,
                         # sanitize_model_id_for_path,
                         # iter_bundle_data_files
    taxonomy.py          # KAOMOJI_START_CHARS, is_kaomoji_candidate,
                         # _looks_like_bare_kaomoji (Path B),
                         # extract, KaomojiMatch (span-only),
                         # canonicalize_kaomoji (rules A–P; frozen)
    synth_prompts.py     # locked cross-corpus prompts +
                         # DEFAULT_*_MODEL_ID + nudge messages
    synth.py             # mask_kaomoji + cache helpers + Synthesizer
                         # base + Anthropic/OpenAI/Local backends
    scrape.py            # ScrapeRow + iter_all chain helper
    sources/             # static-export readers (claude.ai +
                         # claude.ai-alternate, chatgpt, gemini,
                         # openhands, generic journal); all route
                         # through _common.kaomoji_lead_strip
    backfill.py          # transcript→journal replays for claude_code
                         # + codex (incl. .jsonl.zst) + hermes + grok;
                         # parity-tested. Hybrid extraction mirrors
                         # shell hook (see "Round-6 Path B" gotcha)
    providers/           # base.py (HookInstaller +
                         # JsonSettingsHookInstaller +
                         # PluginInstaller + ProviderStatus +
                         # SettingsCorruptError + home_path + render
                         # helpers) plus claude_code / codex / hermes /
                         # grok / antigravity / opencode / openclaw
                         # concrete subclasses
    _hooks/              # bash templates per harness +
                         # _kaomoji_validate.sh.partial (validator)
                         # + _journal_write.sh.partial +
                         # claude_codex_nudge.sh.tmpl (shared
                         # UserPromptSubmit) + hermes_nudge.sh.tmpl
                         # (bare {context: ...} shape) +
                         # antigravity_nudge.sh.tmpl (injectSteps
                         # ephemeralMessage)
    _plugins/            # TS plugin templates +
                         # _kaomoji_taxonomy.ts.partial (canonical
                         # TS port of validator + Path B; spliced
                         # via render_plugin_template,
                         # byte-asserted by
                         # test_plugin_taxonomy_block_matches)
    paths.py             # ~/.llmoji home, cache, bundle, journals,
                         # .salt (per-machine submission token).
                         # NOT an install registry — install state
                         # is read live from each harness's settings.
    analyze.py           # single-stage synthesize + bundle write
    upload.py            # tar + HF / email targets
    _shared_token.py     # encrypted shared HF credential
                         # (PBKDF2/HMAC-keystream, stdlib only)
    cli.py               # argparse entry, [project.scripts] llmoji
  tests/                 # public_surface, canonicalize (~70 cases),
                         # pipeline_parity (bash-vs-Python), import,
                         # new_providers (grok / antigravity /
                         # openclaw paths + HOME-sandbox guard),
                         # status_extended, soft_install,
                         # install_autodetect, source-export readers,
                         # upload_proof_of_life
```

## Gotchas

### Journal-row contract: `assistant_text` never carries the kaomoji

Every source — bash hooks, static exports, generic-JSONL contract —
must persist `assistant_text` with the leading kaomoji already
stripped. The prefix lives separately in the row's `kaomoji` field.
Bash hooks enforce via jq's
`sub("^\\s+"; "") | ltrimstr($kaomoji) | sub("^\\s+"; "")`;
static-export readers route through
`llmoji.sources._common.kaomoji_lead_strip`. Future readers should
use the shared helper rather than re-implementing it.

`mask_kaomoji` consequently has a single branch: prepend `"[FACE] "`.
No source-shape dispatch.

### KAOMOJI_START_CHARS — single source of truth

Python: `llmoji.taxonomy.KAOMOJI_START_CHARS`. Shell:
`${KAOMOJI_START_CASE}` rendered at install time from the same set.
`is_kaomoji_candidate` validates Python-side; the rendered case
filter handles the shell-side first pass. If you find another copy
of the set, delete it and route through `llmoji.taxonomy`.

`_leading_bracket_span` uses depth-walking to locate the closing
bracket on bracket-leading inputs, falling back to a whitespace-
delimited word when the depth-walker doesn't close cleanly.

### Round-6 Path B and the Python fallback in shell hooks

Round 6 added `_looks_like_bare_kaomoji` to `is_kaomoji_candidate`,
catching bare faces (`*_*`, `T-T`, `^_^`, `>_<`, `:)`, `XD`) without
a leader char. Python is the single source of truth; the TS plugin
partial ports the logic by hand (covered by
`test_plugin_taxonomy_block_matches`). Shell hooks defer to Python
via subprocess only when Path A fails:

```bash
case "$KAOMOJI" in
  $KAOMOJI_START_CASE) ;;            # Path A — fast bash path
  *)
    if ! '${PYTHON_INTERPRETER}' -c '...' "$KAOMOJI" ; then
      ${SKIP_ACTION}
    fi
    ;;
esac
```

`${PYTHON_INTERPRETER}` is `sys.executable` substituted at install
time so the hook calls the same Python (and same `taxonomy.py`) the
user has installed. The Python startup cost (~150–200ms) is paid
only on Path A misses, which under `--soft` are rare since the
system-prompt nudge keeps the model leading with kaomoji.

Shell + backfill share a two-stage hybrid extraction:

  1. Strip from the first ASCII letter onward. Preserves bracket-
     leading kaomoji with internal whitespace (`(ง •̀_•́)`,
     `(｡˃ ᵕ ˂)`) which a naive whitespace-split would clip.
  2. If stage 1 yields empty (position 0 is itself an ASCII letter),
     fall back to whitespace-split. Catches Path B letter-eye bare
     kaomoji (`T-T`, `XD`, `Q_Q`, `e_e`).

Both stages cap output at 32 chars to match `_KAOMOJI_MAX_LEN`. Same
hybrid in `_kaomoji_validate.sh.partial` and `backfill.kaomoji_prefix`;
drift is what `test_pipeline_parity.py` exists to catch.

### Per-provider kaomoji capture — N rows per turn

All providers emit one row per kaomoji-led model message. A
tool-heavy turn easily writes 5–10 rows interleaved with tool calls.

- **Claude Code**: each assistant content block (text, tool_use,
  thinking) is its own top-level transcript JSONL entry. Stop hook
  scopes to entries at-or-after the latest real-user message
  (string content OR text-block array, NOT tool_result, NOT an
  injected-prefix payload such as a skill body — before 2.2 a skill
  loaded mid-turn reset the boundary and dropped every message ahead
  of it), walks every text-bearing non-sidechain entry, and feeds
  the rows into a
  `while read` loop with `SKIP_ACTION=continue`. The transcript is
  often behind when Stop fires — Claude Code documents it, and on a
  real corpus 89 of 347 recent kaomoji-led messages (the closing one
  of a turn, almost always) never reached the journal — so the hook
  also reads the payload's `last_assistant_message` (2.1.47+) and
  appends it unless the turn's last assistant message — the run of
  assistant entries after the turn's last user-type entry — is the
  final reply already: no `tool_use` block (a message calling a tool
  didn't end the turn) and a first text block the payload text starts
  with. Any earlier message is irrelevant: an earlier
  "(◕‿◕) Done." must not hide a missing "(◕‿◕) Done. Here's …". The
  appended row has no transcript time yet and carries fire time.
- **Codex**: each model message is its own event. codex-cli 0.144+
  writes `event_msg` / `item_completed` with `item.type ==
  "AgentMessage"` (text in `item.content[].text`); older releases
  wrote `event_msg.agent_message` (`payload.message`). The paginated
  format became the default between 0.151 (app-server threads) and
  0.158 (local), and the pre-2.2 hook — which read only the legacy
  event — went silent with no error: a9's journal stops on
  2026-08-24. The hook finds the latest `turn_context` index,
  slices forward, and reads the turn in whichever shape it carries
  (item_completed wins; legacy is the fallback for a turn with
  none), so a release writing both can't double-count. It never
  reads `response_item` assistant messages: a forked thread replays
  its parent's history into the child rollout as response_items.
  `user_text` resolves to the latest non-injected user response_item
  in the rollout so far — a turn Codex starts by itself (goal step,
  subagent result) has none of its own and carries the prompt that
  set it going (before 2.2 those rows got `""`, about 12% of a9's) —
  and `model` falls back to the turn_context's when the Stop payload
  has none. On all 337 current rollouts on a9's machine the live
  rows are byte-identical to the replay's.
- **Grok**: `transcript_path` is the session's `updates.jsonl` (ACP
  update stream). The hook keeps `agent_message_chunk` updates whose
  `_meta.promptId` equals the payload's `promptId`, joins runs of
  consecutive chunks into one message, and takes `user_text` from the
  latest `user_message_chunk` before the turn's first chunk (what the
  user typed — Grok's injected `<user_info>` and reminders live only
  in `chat_history.jsonl`). The model is the latest `model_id` in the
  sibling `chat_history.jsonl`; `turn_completed.modelUsage` has it
  too, but lands after the Stop hook. Only `reason == "end_turn"`
  is journaled: Grok fires a second Stop at session teardown
  (`shutdown` / `channel_closed`) that would repeat the last turn.
- **Antigravity**: the transcript (`transcriptPath`, the
  `transcript_full.jsonl` under `brain/<id>/`) is a step list; the
  hook walks `PLANNER_RESPONSE` steps with non-empty `content` after
  the latest `USER_INPUT`. Model and prompt come from the payload
  (`modelName`, `lastUserInput` — the latter since agy 1.4.7, with a
  `<USER_REQUEST>`-stripping fallback to the step). The transcript is
  complete when Stop fires (verified live). Hooks answer in JSON on
  stdout, so the journal hook prints `{}` before anything else.
- **Hermes**: walks `extra.conversation_history` (full message list
  `post_llm_call` carries), slices from the latest user-role message
  to end, emits one row per kaomoji-led non-empty assistant message.
  Tool-only assistant messages (`tool_calls` + empty content) are
  skipped naturally.

Per-row invariants:

- `user_text` is resolved once per turn — every row from one turn
  carries the same originating prompt.
- Text fields go through bash command substitution in the live
  hooks, which drops trailing newlines; the replays apply the same
  rule (`backfill.as_shell_var`) so the two copies of a message
  compare equal.
- A Stop hook fires again when another hook blocks the stop and the
  agent continues, and the walk re-emits the turn's earlier
  messages. Rows carry message time, so a repeat from an earlier fire
  is byte-identical: the journal-write partial checks each row against
  the journal's last 256 lines as they stood when the run began, and a
  match uses that line up (multiset — one earlier copy absorbs one
  repeat, so two identical rows from one run, or a genuine third copy,
  still land). The bound only misses a re-fire that arrives after more
  than 256 rows were written since the first fire.
- The cache key hashes `(synth_model_id, backend, base_url,
  source_model, canonical, sample_set_hash)` where
  `sample_set_hash` covers the sorted `(user, assistant)` pairs
  that fed the per-cell synthesis. Cells whose sample sets shift
  on re-import (new rows added past `INSTANCE_SAMPLE_CAP`)
  cache-miss cleanly; stable cells cache-hit.
- Backfills (`backfill_codex` / `backfill_claude_code` /
  `backfill_hermes` / `backfill_grok`) implement the same
  per-message walk and stay parity-tested via
  `test_pipeline_parity.py`, `ts` included: since 2.2 a live row and
  its replay are the same row. Hermes parity uses
  `_HERMES_PARITY_FIELDS` (excludes `cwd` and `ts`) because session
  JSON persists neither per message — backfilled rows carry `""` /
  `last_updated` while the live hook stamps `Path.cwd()` / fire time.
  Antigravity has no backfill: its transcripts carry no per-step
  model id (that's in protobuf blobs in the conversation SQLite
  store), and every bundle row is keyed by model.

### Nudge hooks — what gives the corpus its size

Without the per-turn reminder the model drifts away from leading
kaomoji over a long session. `--hard` installs a nudge hook; `--soft`
puts the same wording in the system-prompt doc. Response shapes for
the hook variant:

- **Claude Code + Codex**: `{"hookSpecificOutput": {"hookEventName":
  "UserPromptSubmit", "additionalContext": "<msg>"}}`. Codex's
  envelope is byte-identical (verified at
  `codex-rs/hooks/src/events/user_prompt_submit.rs`) so a single
  shared `claude_codex_nudge.sh.tmpl` serves both. `nudge_message`
  substitutes through `_shell_quote` into a bash single-quoted
  literal.
- **Hermes**: bare `{"context": "<msg>"}` — no envelope, returned by
  `pre_llm_call`, the only hook whose return value is used.
- **Antigravity**: `{"injectSteps": [{"ephemeralMessage": "<msg>"}]}`
  from `PreInvocation`, before every model call. Ephemeral steps are
  transient by design (Antigravity's own docs use them for exactly
  this kind of reminder) and each call can open a new message. In a
  live run (Gemini 3.8 Flash, 2026-10-25) with a prompt that never
  mentioned kaomoji, both of the turn's messages led with one.
- **Grok**: none. `UserPromptSubmit` runs but Grok discards an
  allowing hook's stdout, so `--hard` is refused
  (`hard_unsupported_reason`).

The base class exposes the nudge through `nudge_hook_template` /
`nudge_hook_filename` / `nudge_event` / `nudge_message` class attrs
and a `has_nudge` predicate. Adding a nudge to a future provider is
four class-level attrs.

### Sidechain strategy

- **Claude Code**: drop rows where `isSidechain` is true (field-flag).
- **Codex**: spawned subagent threads (multi-agent mode) run
  `SubagentStop`, never the root `Stop` (codex PR #22873, May 2026),
  so the live hook doesn't see them. Their rollouts carry
  `session_meta.payload.thread_source == "subagent"` (and a `source`
  object keyed `subagent`); the hook and the backfill both skip such
  a rollout whole.
- **Hermes**: **no filter yet.** `post_llm_call` still exposes no
  `parent_session_id` / `is_subagent`. Since hermes v0.16.0
  (2026-06-05) `subagent_start` / `subagent_stop` carry
  `child_session_id`, so a filter is now possible — record child ids
  from those events, drop `post_llm_call` rows whose `session_id` is
  one — but it needs state shared across hook processes and isn't
  built.
- **Grok**: inside a subagent the gate fires as `SubagentStop`; a
  `Stop` hook only gates the main agent. The hook still drops a
  payload carrying `subagentType`. Child sessions live in the normal
  sessions tree and are listed under the parent's `subagents/`; the
  backfill skips any session whose id appears there (meta schema
  undocumented — every dir name and string value counts).
- **Antigravity**: unverified. The conversation store has a
  `parent_references` table, which would be the signal to filter on
  if subagent executions turn out to fire the global `Stop`.
- **OpenClaw**: tracks `subagent_spawned` / `subagent_ended` runIds
  in the TS plugin and drops their `llm_output` rows. Unverified
  against a live OpenClaw (until 2.2 the plugin never loaded — see
  below). Current OpenClaw names the child in `subagent_ended` by
  `targetSessionKey` (matching `subagent_spawned.childSessionKey`);
  keying the set on session keys would be the sturdier filter.

### OpenClaw paths were wrong before 2.2

OpenClaw's global plugin root is `~/.openclaw/extensions/`
(`resolveDefaultPluginExtensionsDir`, `src/plugins/install-paths.ts`)
and its config is `~/.openclaw/openclaw.json` (`CONFIG_FILENAME`,
`src/config/paths.ts`) — already so in OpenClaw v2026.3.1. 1.3.0
through 2.1.3 wrote `~/.openclaw/plugins/llmoji-kaomoji/` and
`~/.openclaw/config.json`, neither of which OpenClaw reads, so the
provider never captured anything. Install and uninstall now delete
our files and entry at the old paths (and the old config file, once
nothing else is in it). OpenClaw parses `openclaw.json` as JSON5; a
hand-commented file is refused with `SettingsCorruptError` rather
than rewritten without its comments.

### Import dedup and pre-2.2 fire-time rows

Before 2.2 every live hook stamped `ts` with hook fire time, so a
message's live row and its replay never shared a dedup key: each
`llmoji import` after install re-added every live-captured message
(on a9's machine, 773 Claude Code rows and 89 Codex rows were
doubled this way; the next import would have added 1,329 more).
`import_provider` now claims exact-key matches first, then folds
each remaining replayed row into one unclaimed existing row in
fire-time format (whole seconds, `Z`) with the same
`(model, kaomoji, assistant_text)` whose `ts` lies in
`[ts - 5s, ts + 24h]`. Every claim consumes one specific journal
row, so genuine repeats of a short message survive. The heuristic
can undercount by one when a live row's own message is no longer
replayable and an identical message sits within the day before it. Both passes run over the
whole replay set; `--since` only gates what's appended, so an older
message keeps its claim on its own row. Adding `cwd` to the identity
would cost 22 of 885 real folds on a9's Claude Code journal, and
`cwd` plus `user_text` 35. Duplicates already sitting in a journal aren't
rewritten.

### HookInstaller.install refuses to clobber

Three corruption paths defended:

1. Malformed `~/.claude/settings.json` — `_load_json_strict` raises
   `SettingsCorruptError`.
2. Malformed `~/.codex/hooks.json` — same defense, same helper
   (Codex's `hooks` payload — formerly `codex_hooks`, renamed in
   codex-cli 0.129 — is byte-identical to Claude Code's, JSON
   helpers reused).
3. Unparseable `~/.hermes/config.yaml` — ruamel raises `YAMLError`;
   `HermesProvider._read_and_parse` rewraps as `SettingsCorruptError`.
   Same defense for non-mapping top-level docs and for a populated
   `hooks:` value that isn't a mapping.

User gets path + reason. Edit by hand and re-run.

Re-running `install` is idempotent across all three: JSON-edit checks
for an existing entry with our command string and skips; Hermes'
merge does the same structural dedup at the parsed-YAML level. Main
and nudge dedup independently. Settings writes go through
`atomic_write_text` (tmp + `os.replace`) so SIGINT mid-write leaves
old or new content, never half. JSON-settings providers batch
main+nudge edits into a single read-modify-write
(`_register_json_settings_batch`); Hermes does the same single-pass
mutate-then-edit.

### Hermes parsing-only ruamel + surgical text edits

ruamel.yaml is used in `HermesProvider` ONLY for parsing — we never
call `yaml.dump`. Background: PyYAML (which Hermes itself uses to
write `~/.hermes/config.yaml`) escapes non-ASCII and uses backslash
line-continuations in double-quoted scalars to suppress the space
that YAML 1.2 fold rules would insert at non-whitespace wrap
boundaries. ruamel's `RoundTripDumper` does neither, so a string
PyYAML wrapped at the middle of a kaomoji literal `(◕‿◕)` round-trips
through ruamel as `(◕‿◕ )` (one extra space at the wrap point). A
1-char silent mutation to a personality prompt on a config the user
didn't edit is the kind of bug that surfaces weeks later as "the AI
is acting slightly different and I can't repro it."

The current implementation:

1. Read the file as text.
2. Parse with ruamel for the document tree + `lc.data` line/col
   marks.
3. Compute per-edit operations (insert ranges for `_register`,
   deletion ranges for `_unregister`) using the marks.
4. Apply edits as text splices on the original content. ruamel never
   serializes; the file stays byte-stable everywhere except lines we
   explicitly insert or delete.

Quirks of `lc.data`:

- `CommentedMap.lc.data[key]` returns
  `(key_line, key_col, value_line, value_col)` — key column is
  exactly where the key starts.
- `CommentedSeq.lc.data[i]` returns `(item_line, item_col)` — but
  `item_col` reports the column of the *value content* (after `- `),
  not the dash. For standard `- key: value`, the dash is at
  `item_col - 2`. `_infer_list_indent` applies the offset.

Surgical-edit rules:

- Placeholder shapes (`hooks: {}`, `hooks: []`, `hooks: ~`,
  `hooks:` bare) get replaced with a fresh PyYAML-style block.
- Populated mappings are merged into: insert a new event sub-block
  at end of hooks block, append a list item at end of an existing
  event's list, or skip (idempotent dedup by command path).
- Indent style is inferred from the user's existing first sub-key
  + first list item, falling back to PyYAML defaults (mapping=2,
  list=4).
- Refused shapes (raise `SettingsCorruptError`): top-level `hooks`
  that's not a mapping/placeholder, flow-style hooks block or
  bucket, empty bucket (`event: []`).

### Bundle is allowlisted, not just-ship-everything

Both upload paths enforce the flat allowlist:
`BUNDLE_TOPLEVEL_ALLOWLIST = ("manifest.json",)` plus
`BUNDLE_DATA_SUFFIX = ".jsonl"`. No subdirs, no symlinks, no other
file types.

- `upload.tar_bundle()` (email target) raises `BundleAllowlistError`
  on anything else.
- `upload.upload_hf()` does the same pre-flight check AND passes
  `allow_patterns=["manifest.json", "*.jsonl"]` to
  `HfApi.upload_folder` as a second line of defense.

`analyze` clears the bundle dir of all files + subdirs before
writing. The three together mean stale adjective bags, hidden-state
caches, leftover subfolders, etc. cannot leak through `upload`.

### HF upload — per-submission branch + shared encrypted credential

`upload --target hf` pushes `manifest.json` plus each
`<source-model>.jsonl` as loose files at
`contributors/<hash>/bundle-<ts>/` via
`HfApi.upload_folder(..., revision=branch_name, create_pr=False)`
(single atomic commit on a per-submission branch
`submission-<contributor[:12]>-<ts>`). The maintainer reviews each
branch by diff and merges to `main` by hand.

Three keys (mirrored in SECURITY.md):

1. **User's HF token** — read once via
   `huggingface_hub.get_token()`, used for one `HfApi.whoami()`
   proof-of-life call, then discarded. Never authenticates the
   upload itself.
2. **Upload password** — read from `$LLMOJI_UPLOAD_PASSWORD` or
   prompted via `getpass.getpass`. Posted on the dataset card and
   on Twitter ([@_a9lim](https://twitter.com/_a9lim)). Gates
   decryption of the shared submission credential.
3. **Shared submission HF token** — encrypted under the upload
   password and shipped in `llmoji/_shared_token.py` as
   `ENCRYPTED_TOKEN_B64`. `decrypt_with_password(password)` returns
   the plaintext at runtime. Constructed at release time via
   `encrypt_for_release(token, password)`.

Encryption: PBKDF2-SHA256 (200,000 iterations) for the KDF,
HMAC-SHA256-keystream XOR for the cipher, HMAC-SHA256 for integrity
(encrypt-then-MAC, constant-time `compare_digest`). Stdlib only.
Layout: `base64([16-byte salt][32-byte mac][N-byte ciphertext])`.
CI smoke tests catch the placeholder blob at decrypt time so a
release that forgot the rotation step bails loudly.

The dataset has Discussions and Pull Requests DISABLED so pre-1.2.0
clients (which used `create_pr=True` and would have leaked the
user's HF username) fail at the API layer. Forces the upgrade.

#### Per-release rotation

```python
from llmoji._shared_token import encrypt_for_release, generate_password
password = generate_password()
blob = encrypt_for_release("hf_<the_real_token>", password)
```

Paste `blob` into `ENCRYPTED_TOKEN_B64`, bump the package version,
release, post `password` on the dataset card and Twitter. If
rotating because of a compromise, revoke the previous fine-grained
token from a9's HF settings first.

The HF token is a fine-grained token on a9's account scoped to write
on `a9lim/llmoji` only, no expiry. A separate submissions account
isn't needed — fine-grained scoping makes the blast radius identical.

`upload_folder` returns a `CommitInfo` whose `commit_url` points at
the submission-branch commit; older `huggingface_hub` versions
returned a bare URL string, so `upload_hf` defensively unwraps both.
The dataset card has a `configs:` YAML pointing at
`contributors/**/*.jsonl` for the auto-loader; tarballs would
trigger HF's WebDataset auto-detection and break the viewer, so
email target keeps `tar_bundle` for the single-attachment shape but
HF target goes loose-file. `mailto:` handoff goes through
`webbrowser.open` (stdlib, cross-platform).

### Hermes payload contract — source-verified

The hermes provider installs two hooks under `~/.hermes/agent-hooks/`:
`post-llm-call.sh` (journal logger) and `pre-llm-call.sh` (nudge,
under `--hard`). Both registered in `~/.hermes/config.yaml`.

Cross-checked the documented [Event Hooks][hermes-hooks] contract
against the source at `hermes-agent/agent/shell_hooks.py
:_serialize_payload` (top-level shape) and
`hermes-agent/run_agent.py:12492` (`post_llm_call` kwargs:
`session_id`, `user_message`, `assistant_response`,
`conversation_history`, `model`, `platform`). The `extra.*` block
holds everything except the four reserved top-level keys
(`tool_name`, `args`, `session_id`, `parent_session_id`); `cwd` is a
top-level field, NOT under `extra`.

`extra.user_message` is the original pre-injection user message, so
`system_injected_prefixes` stays `[]`.

Hermes stopped writing `~/.hermes/sessions/session_*.json` snapshots
in v0.21.1 (tag `v2026.9.7`, commit `7a5fc1b2a9`); sessions since live
only in `state.db`. `llmoji import hermes` therefore replays history
up to that release and silently misses everything after; a
`state.db` reader (or `hermes sessions export`) is the fix, not yet
written. If real-traffic inspection
later shows leaked prefixes, populate the list and re-render — the
bash hook picks it up via `${INJECTED_PREFIXES_FILTER}`.

[hermes-hooks]: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks/

### Cache directory is locally-private

`~/.llmoji/cache/per_cell.jsonl` holds the structured adjective
bag for each `(source_model, canonical_kaomoji)` cell, keyed by
content hash. Each row carries an adjective bag drawn purely
from the locked LEXICON — no prose, no free-form, no user-text
content — so the cache itself doesn't leak per-turn specifics
the way v1's `per_instance.jsonl` could (each v1 row was a
paraphrase of a single user turn). Mitigations:

- Cache is **never** bundled or shipped. The bundle row carries
  the same adjective bag, but the cache also carries the
  `(model, backend, base_url, source_model, sample_set_hash)`
  metadata that the bundle row drops.
- `llmoji status` prints cache size.
- `llmoji uninstall <provider>` does NOT touch the cache (the
  user may re-install). `llmoji cache clear` is the explicit
  wipe (and removes the orphaned legacy v1
  `per_instance.jsonl` if still on disk).

The bundle is the only thing that leaves the machine; the
inspection gap is the consent boundary.

### Codex `transcript_path` carries the rollout JSONL

Used to resolve `user_text`. Codex injects AGENTS.md /
`<environment_context>` / `<INSTRUCTIONS>` as user-role response_items
at session start; we walk the rollout to find the latest real user
turn, dropping those prefixes defensively. `llmoji.backfill` mirrors.

codex-cli zstd-compresses older rollouts in place
(`rollout-*.jsonl.zst`) and moves archived ones to
`~/.codex/archived_sessions/`. `llmoji import codex` reads both;
decoding needs Python 3.14's `compression.zstd` or the `zstandard`
package (`llmoji[zstd]`), and files it can't decode are counted and
reported, not silently dropped.

### Generic JSONL contract for unsupported harnesses

Motivated users on unsupported harnesses can write directly to
`~/.llmoji/journals/<name>.jsonl` against the canonical 6-field
schema. `llmoji analyze` picks them up automatically. The opencode
and openclaw TS plugins are reference implementations of this
contract on a TS-plugin host. Porting to a third TS-plugin host is
copy-paste-adapt one of those templates plus a new `PluginInstaller`
subclass.

### HF dataset card is a separate hand-maintained surface

The user-facing dataset card at
[`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) is
not in this repo. It re-states the bundle schema and privacy model
in user-facing prose so contributors can decide whether to submit
before they've ever touched the package README.

Two coupling points:

- **Schema changes need both updates.** Any change to `manifest.json`
  or `<source-model>.jsonl` field names is a cross-corpus invariant
  change and wants a hand-edit on the dataset card. Editable in the
  HF web UI.
- **License split.** Package code is GPL-3.0-or-later; the shared
  corpus on HF is CC-BY-SA-4.0. `llmoji upload --target hf`
  contributes a bundle under those terms; the README's License
  section calls this out. `llmoji-study` is also CC-BY-SA-4.0
  (research artifact, not distributed program).

## Conventions

- Any Python 3.11+ environment with an editable install
  (`pip install -e ".[dev]"`, or `uv pip install -e ".[dev]"`). On
  a9's machines llmoji is an editable install in the shared uv venv,
  so the live hooks' Path B fallback runs this checkout's
  `taxonomy.py`.
- `main` is branch-protected: PR-only (no direct pushes), four CI
  jobs (lint / typecheck / test / build) required green, branch
  up-to-date, conversation resolution required, force-push and
  deletion blocked. Day-to-day work lands on `dev`; merge via PR.
- `~/.llmoji` is the on-disk root for everything the package
  manages; tests override via `$LLMOJI_HOME`.
- Bash hook templates are syntactically validated by `bash -n` in
  `test_hook_templates_render_to_valid_bash_substitutions`. TS
  plugin templates aren't bash-validated (no equivalent stdlib
  parser); rendered output is asserted to contain expected taxonomy
  via `test_plugin_taxonomy_block_matches`. A future installer
  flavor (Lua, JS-without-TS) follows the bash-vs-plugin split:
  new sibling base under `providers/base.py` plus templates under a
  new `_<flavor>/` package data dir.
- Single-stage synth calls run on a small thread pool (default 1,
  `--concurrency` flag or `$LLMOJI_CONCURRENCY` to override). Cache
  writes happen on the main thread inside the `as_completed` loop
  immediately after each future succeeds, so a mid-wave failure
  leaves the cache populated for cells that succeeded. We collect
  errors, drain the loop, and raise `AnalyzeError` with a "re-run
  to resume" message. Default 1 because the org-level Haiku rate
  cap (50 req/min) trips intermittently at concurrency=2 on
  multi-hundred-cell backfills; the SDK's `max_retries=8`
  exponential backoff (set explicitly, vs the SDK default of 2)
  recovers but burns wallclock. Bump if your tier has headroom.
  `synth_by_cell` is assembled in deterministic walk order so
  re-runs produce byte-identical bundle output; the on-disk cache
  row order is non-deterministic and that's fine because the
  cache is hash-keyed. `INSTANCE_SAMPLE_CAP` is 4 — same value as
  Eriskii's original Claude-faces work, kept for cross-corpus
  comparability. v2's single-stage shape means one call per cell
  vs v1's `(N + 1)` per cell, so a re-analyze on the same corpus
  is roughly 5× cheaper.
- Public-API freeze: anything in §"Cross-corpus invariant surface"
  is a cross-corpus invariant; bumping wants a hand-edit on the HF
  dataset card and a flag in the PR body. Internal helpers
  (`llmoji._util`, leading-underscore names in
  `llmoji.providers.base`, `llmoji.synth.cache_key`, the synth
  backend classes, etc.) are free to evolve.
