# AGENTS.md

## What this is

`llmoji` collects kaomoji journals from coding agents (Claude Code,
Codex, Grok Build, Antigravity), describes each (source model,
canonical face) cell with words from a locked lexicon via the user's
chosen synthesis backend, and submits those aggregates to the shared HF
dataset [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji).
Research (probes, embeddings, figures) lives in
[`llmoji-study`](https://github.com/a9lim/llmoji-study); this package is
data collection only, with no torch or matplotlib.

As of 2026-10-25 the dataset has exactly one submission, a9's own; the
users llmoji answers to are a9's machines and anyone who might submit.

## Pipeline

```
Stop hook (sh shim) → python -m llmoji.hook <provider>
    → provider.live_rows(payload) → capture.append_rows → ~/.<harness>/kaomoji-journal.jsonl
llmoji import  → provider.replay(each session file) → capture.import_provider → same journal
llmoji parse   → sources.{claude_export,chatgpt_export} → ~/.llmoji/journals/<name>.jsonl
llmoji analyze → sources.journal.iter_journal over every journal → analyze → ~/.llmoji/bundle/
llmoji upload --target {hf,archive}
```

The bundle on disk between `analyze` and `upload` is the inspection gap:
the user reads each `<source-model>.jsonl` before deciding to ship.

## Layout

```
llmoji/
  __init__.py        public surface + __version__ (hatch reads it)
  taxonomy.py        KAOMOJI_START_CHARS, is_kaomoji_candidate,
                     split_leading_kaomoji (writing), extract +
                     canonicalize_kaomoji (bucketing)
  synth_prompts.py   LEXICON, LEXICON_VERSION, SYNTHESIS_SCHEMA,
                     SYNTHESIZE_PROMPT, pinned models, NUDGE_MESSAGE
  scrape.py          ScrapeRow, kaomoji_row
  providers/         base.py (Provider, ProviderStatus, JSON
                     registration, the doc block) + one module per
                     harness: paths, transcript decoder, quirks
  capture.py         row identity, live append, import_provider
  hook.py            the hook entry point
  sources/           journal reader, claude.ai + ChatGPT export readers
  synth.py           backends, validation, per-cell cache
  analyze.py         bucketing, sampling, bundle write
  upload.py          allowlist, HF push, archive
  _shared_token.py   the encrypted shared HF credential
  cli.py, paths.py, _util.py
tests/               conftest sandboxes HOME and LLMOJI_HOME for every test
```

## Cross-corpus surface

Changing any of these changes what a submission means: flag it in the
PR body, hand-edit the dataset card, and expect a major version.

- **Taxonomy** (`llmoji.taxonomy`): `KAOMOJI_START_CHARS`,
  `is_kaomoji_candidate` (length 2–32, no backslash after position 0, no
  run of 4+ ASCII letters, and a leader char with real content or a
  bare-face shape), `split_leading_kaomoji`, `extract`,
  `canonicalize_kaomoji` (rules A–O, listed in its docstring). The
  comments in `taxonomy.py` say why each glyph and rule is there.
- **Synthesis** (`llmoji.synth_prompts`): the 50-word `LEXICON` (26
  circumplex anchors over a 9-cell PAD grid, 24 extension words),
  `SYNTHESIS_SCHEMA` (two arrays over disjoint enums, nothing
  free-form), `SYNTHESIZE_PROMPT`, the pinned models
  (`claude-haiku-5-5`, `gpt-6-luna` since 3.0), `NUDGE_MESSAGE`. A
  lexicon or schema change bumps `LEXICON_VERSION` and the snapshot in
  `tests/test_lexicon_evolution.py` in the same commit.
- **Journal row**: six string fields, `{ts, model, cwd, kaomoji,
  user_text, assistant_text}`. `ts` is the message's own time where the
  source has one. `kaomoji` is the face, `assistant_text` the rest of
  the message without it.
- **Bundle**: `manifest.json` (`llmoji_version`, `lexicon_version`,
  `synthesis_model_id`, `synthesis_backend`, `submitter_id`,
  `generated_at`, `providers_seen`, `model_counts`,
  `total_synthesized_rows`, `notes`) plus one
  `<sanitize_model_id_for_path(model)>.jsonl` per source model, rows
  `{kaomoji, count, synthesis: {primary_affect, stance_modality_function}}`.
  `providers_seen` names journals as `<provider>-hook` or by file stem.
- **The four providers and their instructions files**: `claude_code`
  (`~/.claude/CLAUDE.md`), `codex` (`~/.codex/AGENTS.md`), `grok`
  (`~/.grok/AGENTS.md`), `antigravity` (`~/.gemini/GEMINI.md`). The
  block is `## Kaomoji\n\n<NUDGE_MESSAGE>` appended at EOF.

Free to change: everything else, including the cache key derivation,
`INSTANCE_SAMPLE_CAP` / seed, `ScrapeRow`, CLI flags beyond the targets
and backends.

## Commands

```
llmoji install [<provider>] [--yes]     hook + registration + doc block;
                                        no provider: every harness found
llmoji uninstall [<provider>] [--yes]   inverse; journals kept
llmoji status [--provider P] [--stats] [--top N] [--json]
llmoji import [<provider>] [--since ISO] [--dry-run] [--yes]
llmoji parse --provider {claude.ai,chatgpt} DIR...
llmoji analyze [--backend {anthropic,openai,local}] [--base-url] [--model]
               [--concurrency N] [--notes] [--dry-run]
llmoji upload --target {hf,archive} [--yes]
llmoji cache clear
```

## Gotchas

### One splitter for writing, `extract` for bucketing

`split_leading_kaomoji` decides where a message's face ends. It takes
everything up to the first whitespace or ASCII letter outside brackets
(so `(ง •̀_•́)ง` and `(o_O)` survive whole), falls back to the first
word for bare faces (`T-T`, `XD`), then adds whitespace-separated
decoration tokens (`┻━┻`, `・ﾟ`) but stops at markdown (`**`, `—`, `1.`)
and at any two letters in a row (Japanese or Cyrillic prose). Every row
writer uses it: the providers' decoders and both export readers.

Bucketing still goes through `extract` (the balanced-bracket span) and
`canonicalize_kaomoji`. `iter_journal` re-splits each stored face, moves
any tail back into the body, and takes `extract` of the face as
`first_word`. That repairs rows llmoji wrote before 3.0, whose splitter
cut at the first ASCII letter and so let `**`, `—` or a whole line of
non-ASCII prose into the face (about 9% of a9's Claude Code rows), and
keeps every stored face in the cell it always had. Changing the splitter
moved 8 of 3,885 historical faces to a different cell, all bracket
fixes like `(・` → `(・o・)`.

### Row identity survives the split

A message's identity (`capture.row_key`) is its `ts` plus `kaomoji +
assistant_text` with all whitespace removed. It holds whichever splitter
version divided the message, so rows from older releases still match
their replays. Model isn't part of it: Grok's live rows take the model
from `chat_history.jsonl`, its replays from `turn_completed`.

### Live capture and import share the decoder

Each provider's `live_rows` runs the same decoder as its `replay`, over
the turn the hook fired on. Before 3.0 the hooks were bash/jq copies of
the Python replay, kept equal by an 1,700-line parity suite; that whole
class of drift is gone. What remains provider-specific is in each
provider module's docstring: Claude Code's transcript lag and the
`last_assistant_message` fallback, Codex's two message shapes and forked
threads, Grok's teardown Stop and promptId scoping, Antigravity's step
list and sandboxed journal location.

### Injected user text is never the prompt

Harnesses write text of their own as user-role entries: Claude Code's
task notifications, reminders and slash-command output, Codex's skill
bodies, subagent notifications and goal continuations
(`scrape.INJECTED_USER_PREFIXES`, plus each provider's own list). None
of it becomes `user_text`. In Claude Code a task notification still
starts a turn (a Stop follows it) and roots a parentUuid chain of its
own, so the decoder falls back to the latest real prompt in file order.
A brief from another agent (`<task>`, a gaslamp consult) is a real
prompt and kept. Before 3.0 about a fifth of a9's Claude Code rows
stored a notification as their prompt; `iter_journal` reads injected
`user_text` as empty.

### The hook is a shim around one interpreter

`render_hook` writes:

```sh
#!/bin/sh
# llmoji journal hook for <name>; `llmoji install <name>` rewrites this file.
'<sys.executable>' -I -m llmoji.hook <name> 2>/dev/null
exit 0
```

(Antigravity's prints `{}` first; its hooks answer on stdout.) `-I`
keeps a project's own `llmoji/` or `json.py` in the cwd from shadowing
the package. The hook never fails the harness: `llmoji.hook.main` logs
any exception to `~/.llmoji/hook-errors.log` and returns 0, and `status`
mentions the log when it exists. Rebuilding the venv at a new path
leaves the shim pointing at nothing; `status` reports the hook stale
because its text no longer matches `render_hook`. Python start-up plus
the imports costs about 45 ms per Stop.

On a9's machines llmoji is an editable install in the shared uv venv, so
the live hooks run this checkout. A broken `llmoji.hook` import on `dev`
silently stops capture everywhere until fixed (errors land in the log).

### The live append and the fold

`append_rows` checks each row against the journal's last 256 lines and
skips one it already holds, each journal line absorbing one copy, so a
Stop that fires again (another hook blocked the stop) doesn't re-journal
the turn while a message genuinely repeated still lands.

Both the append and `import_provider` also fold a row into one with the
same text whose `ts` is a whole-second fire time within
[message − 5 s, message + 24 h]. Pre-2.2 hooks stamped every row with
fire time, and Claude Code's final-message fallback still does. The fold
can undercount by one when a fire-time row's own message is gone from
disk and an identical message sits within the day before it. Import
claims exact matches over the whole replay set before folding, so a
fold can't take a row that a later replayed row matches exactly; a key
already taken (a forked session replays its parent) is a duplicate.

Import is also the migration for stored history: every journal row a
replay claims is replaced by the replay's copy when the two differ as
rows (not as text: jq wrote compact JSON). The replay is authoritative,
so fire-time stamps, missing models, pre-3.0 splits and injected
prompts are repaired wherever the transcript still exists; on a9's Mac
the first 3.0 import repaired about 940 rows. `--since` gates appends
and repairs, not claims. The live append never rewrites.

### Install refuses to clobber

A settings file that isn't a JSON object, or whose `hooks` (or the
event's bucket) has the wrong type, raises `SettingsCorruptError` before
anything is written. Writes go through `atomic_write_text`, which
follows symlinks: a9's harness config is linked from
`~/Work/meta/bootstrap`, and renaming onto a link would detach it from
the repo. A `$HOME`-spelled registration counts as ours
(`same_command`), since the dotfiles spell it that way. Grok's
`~/.grok/hooks/llmoji.json` and Antigravity's plugin directory are
llmoji's own and are deleted once empty.

Uninstall strips the current doc block or any wording or heading an
earlier release wrote (`synth_prompts.LEGACY_NUDGE_MESSAGES`, `# Kaomoji`);
a hand-edited block is left alone.

### Synthesis output is validated

`synth.validate_synthesis` requires exactly the two schema fields, each
a list of words from its half of the lexicon, on every backend reply and
every cache hit (an invalid cached row is re-dispatched). The hosted
backends constrain decoding to the schema anyway; the local backend's
fallback appends the schema to the prompt and decodes freely, and
without the check a model could put journal text into the bundle. Word
counts (1–3, 3–5) live in the schema descriptions because Anthropic's
and OpenAI's strict schemas reject `maxItems`; they aren't enforced.

The Anthropic call pins `output_config.effort: "medium"` and raises on
any `stop_reason` but `end_turn`; `MAX_TOKENS` is 4096 because both
pinned models think before answering.

### Bundle allowlist

`upload.bundle_files` allows `manifest.json` plus top-level `*.jsonl`,
nothing else, no symlinks; both targets go through it, and `upload_hf`
also passes `allow_patterns`. `analyze` empties the bundle directory
before writing, unlinking symlinks rather than following them.

### HF upload and the shared token

`upload --target hf` pushes `contributors/<id>/bundle-<ts>/` to a new
`submission-<id[:12]>-<ts>` branch with a shared fine-grained token,
write-scoped to `a9lim/llmoji`, that ships encrypted in
`_shared_token.ENCRYPTED_TOKEN_B64` (PBKDF2-SHA256 200k, HMAC keystream,
encrypt-then-MAC). The password is posted on the dataset card and
Twitter, so the token is effectively public; review of each branch
before merging to `main` is the actual protection. Discussions and PRs
are disabled on the dataset. Rotation:

```python
from llmoji._shared_token import encrypt_for_release, generate_password
password = generate_password()
blob = encrypt_for_release("hf_<token>", password)
```

Paste `blob` into `ENCRYPTED_TOKEN_B64`, release, post the password. On
a compromise, revoke the old token in a9's HF settings first.

The submitter id is a hash of the per-machine salt (`~/.llmoji/.salt`)
and the package version.

### The dataset card is a separate surface

The card at `a9lim/llmoji` restates the bundle schema, the privacy model
and the upload password, and its `configs:` YAML points the viewer at
`contributors/**/*.jsonl`. Schema, model or provider changes want a
hand-edit there. Package code is GPL-3.0-or-later; the corpus and
llmoji-study are CC-BY-SA-4.0.

### Downstream consumers

- `~/Work/meta/bootstrap/lib.sh` (`kaomoji_hooks`) reads `llmoji status
  --provider P --json` for `providers[0].healthy` and runs `llmoji
  install P` when it's false.
- `interpretability-experiments/llmoji-experiment` imports
  `canonicalize_kaomoji`, `extract`, `is_kaomoji_candidate`,
  `iter_journal`, `iter_claude_export` and the lexicon names.

## Conventions

- Python 3.11+; `pip install -e ".[dev]"`. On a9's machines llmoji is
  an editable install in the shared uv venv.
- `main` is branch-protected: PRs only, lint / typecheck / test / build
  required, branch up to date. Day-to-day work lands on `dev`. A version
  bump on `main` publishes to PyPI (`release.yml`).
- Pyright covers the whole package. Tests never reach the real home:
  `conftest.py` points `HOME` and `LLMOJI_HOME` at temp dirs, and
  provider paths are `home_path(...)` descriptors so they follow it.
- Comments say what is true now; history goes in release notes.
