# AGENTS.md

## What this is

`llmoji` is a small provider-agnostic CLI for collecting kaomoji
journals from coding agents (Claude Code, Codex, Hermes, opencode,
openclaw), distilling them into per-(source-model, canonical-face)
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
for claude_code/codex per-message walks; `echo '{}'; exit 0` for
hermes's stdout-JSON contract), `system_injected_prefixes`, and
optional nudge attrs.

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
- **`llmoji.synth_prompts`**: `LEXICON` (46-entry locked
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
  `SHORT_NUDGE_MESSAGE` (v1 one-sentence nudge),
  `LONG_NUDGE_MESSAGE` (v7 introspection framing baked from
  `llmoji-study/preambles/introspection_v7.txt`;
  `tests/test_soft_install.py
  ::test_long_nudge_message_matches_introspection_v7` enforces
  byte-identity). The LEXICON itself is snapshot-pinned in
  `tests/test_lexicon_evolution.py`; any rotation requires both
  the snapshot update and a `LEXICON_VERSION` bump in the same
  commit.
- **6-field journal row schema** (on-disk JSONL):
  `{ts, model, cwd, kaomoji, user_text, assistant_text}`. The
  in-memory `llmoji.scrape.ScrapeRow` (7 fields: `source, model,
  timestamp, cwd, assistant_text, first_word, surrounding_user`) is
  free to evolve.
- **System-injection prefix lists** per provider (in
  `llmoji.providers.{claude_code,codex,hermes}`).
- **`HookInstaller` / `PluginInstaller` interfaces.** 2.0 split the
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
  - `--soft`: journal-write hook + appends a `# Kaomoji` heading +
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
  - `opencode` → `~/.config/opencode/AGENTS.md`
  - `openclaw` → `~/.openclaw/workspace/SOUL.md`
- **Soft-doc shape**: plain markdown, no comment fences. Block is
  `# Kaomoji\n\n<message>` appended at EOF with a blank-line
  separator. Uninstall removes the block by exact string match
  against the two canonical wordings (short / long); a hand-edited
  body falls through and survives uninstall (conservative on the
  user's prose). The `# Kaomoji` heading is the cross-corpus anchor
  — bumping it strands existing soft installs.
- **The five first-class providers**: `claude_code`, `codex`,
  `hermes` (bash hooks); `opencode`, `openclaw` (TS plugins).
  `providers_seen` in shipped bundles names these directly.
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
                          "# Kaomoji" + the nudge wording to the
                          harness's system-prompt doc. No per-turn
                          nudge hook.
                          --soft and --hard are mutually exclusive;
                          exactly one required. Both capture journal
                          data; only the placement of the leading-
                          kaomoji reminder differs.
llmoji install <provider> --soft|--hard --long
                          orthogonal to soft/hard. Swaps the v1
                          one-sentence wording for the v7
                          introspection framing.
llmoji install --soft|--hard [--long] [--yes]
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
                          (ts, model, assistant_text), atomic. Stop
                          the harness first. No-arg autodetects every
                          importable harness (claude_code, codex,
                          hermes — TS plugins don't expose replayable
                          transcripts). Recommended after every
                          taxonomy bump.
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
                         # + codex + hermes; parity-tested. Hybrid
                         # extraction mirrors shell hook (see
                         # "Round-6 Path B" gotcha)
    providers/           # base.py (HookInstaller +
                         # JsonSettingsHookInstaller +
                         # PluginInstaller + ProviderStatus +
                         # SettingsCorruptError + render helpers)
                         # plus claude_code / codex / hermes /
                         # opencode / openclaw concrete subclasses
    _hooks/              # bash templates per harness +
                         # _kaomoji_validate.sh.partial (validator)
                         # + _journal_write.sh.partial +
                         # claude_codex_nudge.sh.tmpl (shared
                         # UserPromptSubmit) + hermes_nudge.sh.tmpl
                         # (bare {context: ...} shape)
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
  (string content OR text-block array, NOT tool_result), walks every
  text-bearing non-sidechain entry. `BOUNDARY_TS` query slurps the
  transcript once; the per-entry walk is `jq -c` streamed into a
  `while read` loop with `SKIP_ACTION=continue`.
- **Codex**: each model message is its own
  `event_msg.agent_message` event with `payload.message` carrying
  the text. The Stop hook finds the latest `turn_context` index
  (current turn boundary), slices forward, walks every
  `agent_message`. `user_text` resolves to the latest non-injected
  user response_item in the same slice.
- **Hermes**: walks `extra.conversation_history` (full message list
  `post_llm_call` carries), slices from the latest user-role message
  to end, emits one row per kaomoji-led non-empty assistant message.
  Tool-only assistant messages (`tool_calls` + empty content) are
  skipped naturally.

Per-row invariants:

- `user_text` is resolved once per turn — every row from one turn
  carries the same originating prompt.
- The cache key hashes `(synth_model_id, backend, base_url,
  source_model, canonical, sample_set_hash)` where
  `sample_set_hash` covers the sorted `(user, assistant)` pairs
  that fed the per-cell synthesis. Cells whose sample sets shift
  on re-import (new rows added past `INSTANCE_SAMPLE_CAP`)
  cache-miss cleanly; stable cells cache-hit.
- Backfills (`backfill_codex` / `backfill_claude_code` /
  `backfill_hermes`) implement the same per-message walk and stay
  parity-tested via `test_pipeline_parity.py`. Hermes parity uses
  `_HERMES_PARITY_FIELDS` (excludes `cwd`) because session JSON
  doesn't persist cwd — backfilled rows carry `""` while the live
  hook stamps `Path.cwd()`.

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

The base class exposes the nudge through `nudge_hook_template` /
`nudge_hook_filename` / `nudge_event` / `nudge_message` class attrs
and a `has_nudge` predicate. Adding a nudge to a future provider is
four class-level attrs.

### Sidechain strategy

- **Claude Code**: drop rows where `isSidechain` is true (field-flag).
- **Codex**: no subagent concept; `collaboration_mode` is `"default"`
  for every observed turn_context.
- **Hermes**: **no viable filter on the current payload contract.**
  `subagent_stop` fires from the parent's process with the parent's
  `session_id` and no child id (verified at
  `hermes-agent/tools/delegate_tool.py:2120-2127`); `post_llm_call`
  doesn't expose `parent_session_id` either. Subagent
  `post_llm_call` events land under their own session_ids until
  upstream gives us either `subagent_stop` carrying the child id or
  `post_llm_call` exposing `parent_session_id` / `is_subagent`.
- **OpenClaw**: tracks `subagent_spawned` / `subagent_ended` runIds
  in the TS plugin and drops their `llm_output` rows. Cleaner story
  than the bash providers ship today.

### HookInstaller.install refuses to clobber

Three corruption paths defended:

1. Malformed `~/.claude/settings.json` — `_load_json_strict` raises
   `SettingsCorruptError`.
2. Malformed `~/.codex/hooks.json` — same defense, same helper
   (Codex's `codex_hooks` payload is byte-identical to Claude
   Code's, JSON helpers reused).
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
`system_injected_prefixes` stays `[]`. If real-traffic inspection
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

- Single venv at `.venv/`, pip not uv. `pip install -e ../llmoji`
  during dev.
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
