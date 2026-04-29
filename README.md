# llmoji

[![CI](https://github.com/a9lim/llmoji/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/llmoji/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmoji)](https://pypi.org/project/llmoji/)
[![Downloads](https://img.shields.io/pypi/dm/llmoji)](https://pypi.org/project/llmoji/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/llmoji/)

Llmoji is a small CLI that makes your agents cuter. (´-ω-`)

Llmoji configures your agent to start each message with a kaomoji. It locally saves them, and provides tools to summarize and upload the aggregated meaning per face to contribute to a shared database.

The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is where this data is processed.

There are three main commands:

- **`llmoji install <provider>`**: writes hooks to prompt for and record kaomoji
- **`llmoji analyze`**: scrape and aggregate your logs
- **`llmoji upload --target {hf,email}`**: ship the bundle (HF: opens a PR with loose files; email: tarball)

`analyze` needs an llm to synthesize your logs. By default, it uses Anthropic Haiku and reads `$ANTHROPIC_API_KEY`; `--backend openai` uses GPT-5.4 mini and reads `$OPENAI_API_KEY`; `--backend local` runs against any OpenAI-compatible endpoint (Ollama, vLLM, etc.) and needs `--base-url` and `--model`. `upload --target hf` needs `$HF_TOKEN` with `write` scope. The email path tarballs the bundle and has you attach it manually.

---

## Reporting issues

If you notice any errors while using the program, please update to the most recent version and reinstall the hooks. If it still persists, please open an issue. This project is a work in progress and I am actively finding and fixing bugs.

本プログラムにおいて何らかのエラーが発生し、ご迷惑をおかけしましたことを深くお詫び申し上げます。恐れ入りますが、プログラムを最新バージョンに更新し、コネクタを再インストールしていただけますでしょうか。それでも問題が解決しない場合は、Issue（課題）を起票してお知らせください。本プロジェクトは現在も開発が進行中であり、バグの特定と修正に積極的に取り組んでおります。

---

## Purpose

The shared HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) collects kaomoji counts and a single summarized description per face per source model, across many users' coding agents. The companion repo processes those descriptions. After you run `analyze`, you can inspect the files yourself under `~/.llmoji/bundle/` before you choose to `upload`.

---

## Quick start

```bash
pip install llmoji
llmoji install claude_code      # or: codex, hermes
```

From now on, your agent will use kaomoji at the start of each message. 

After letting it run for a week or so:

```bash
export ANTHROPIC_API_KEY=...
llmoji status                              # check what's been logged
llmoji analyze                             # scrape + canonicalize + summarize
llmoji upload --target hf                  # opens a PR on a9lim/llmoji
# or:
llmoji upload --target email               # opens mailto:
```

You can pick a different backend for `analyze`:

```bash
export OPENAI_API_KEY=...
llmoji analyze --backend openai            # GPT-5.4 mini via the Responses API
# or:
llmoji analyze --backend local \           # any OpenAI-compatible endpoint
  --base-url http://localhost:11434/v1 \
  --model llama3.1
```

`analyze` caches per-instance descriptions at `~/.llmoji/cache/per_instance.jsonl` keyed by content hash plus the synthesis model id, backend, and base URL. `llmoji cache clear` wipes it.

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependency footprint is four packages: `anthropic`, `openai`, `huggingface_hub`, and `ruamel.yaml` (parsing-only, used by the `hermes` provider for surgical edits to `~/.hermes/config.yaml`). Hooks run in `bash` and need `jq`.

From source:

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"      # adds pytest + ruff
```

---

## How it works

### Journal capture

Llmoji first registers a `UserPromptSubmit` hook that injects a reminder on every turn, asking the model to begin its reply with a kaomoji. It then registers a `Stop` hook that fires once per assistant turn, that extracts the reply, strips the kaomoji from the body, and appends one JSONL row to `~/.<harness>/kaomoji-journal.jsonl`. The schema is the same across every provider:

```json
{"ts": "...", "model": "...", "cwd": "...", "kaomoji": "(◕‿◕)", "user_text": "...", "assistant_text": "..."}
```

### Analysis

`llmoji analyze` scrapes every installed provider's journal plus any extra JSONL files under `~/.llmoji/journals/`. For each entry a source model wrote, the chosen synthesizer model describes that specific instance. Then, it aggregates the descriptions for each unique kaomoji per model and writes an overall meaning. This summarized output is the only thing that ships in the bundle.

The synthesizer is one of three backends, chosen via `--backend`. The same synthesizer evaluates everything in a single `analyze` run, so the descriptions across source models are directly comparable.

| Backend     | API                                          | Default model                  |
|-------------|----------------------------------------------|--------------------------------|
| `anthropic` | Anthropic SDK, `messages.create`             | `claude-haiku-4-5-20251001`    |
| `openai`    | OpenAI SDK, Responses API                    | `gpt-5.4-mini-2026-03-17`      |
| `local`     | OpenAI-compatible Chat Completions endpoint  | (set via `--model`)            |

### Bundle structure

`analyze` writes to `~/.llmoji/bundle/`:

```
~/.llmoji/bundle/
  manifest.json
  claude-sonnet-4-6.jsonl
  claude-opus-4-7.jsonl
  gpt-5.5.jsonl
```

- **`manifest.json`**: package version, the synthesis backend and model id used, a salted submitter id, generation timestamp, list of providers seen, per-source-model row counts, total synthesized rows, and anything you include as `--notes`.
- **`<source-model>.jsonl`**: one row per kaomoji as that model used it, with the synthesized meaning. The filename stem is the sanitized model id (lowercase, slashes become double-underscores, colons become hyphens).

---

## Privacy

| Tier                                       | Where                                | Shipped on `upload`? |
|--------------------------------------------|--------------------------------------|----------------------|
| Raw user and assistant text                | `~/.<harness>/kaomoji-journal.jsonl` | Never                |
| Per-instance synthesizer paraphrase        | `~/.llmoji/cache/per_instance.jsonl` | Never                |
| Synthesized summaries and counts per model | `~/.llmoji/bundle/`                  | Yes                  |

Please see [SECURITY.md](SECURITY.md) for the full privacy model.

---

## Providers

`llmoji install <provider>` writes the hook script and registers it with the harness's settings file, idempotently.

| Provider      | Hook events                                       | Settings format | Notes                                                  |
|---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|
| `claude_code` | Stop, UserPromptSubmit                            | JSON            | Stable, in daily use.                                  |
| `codex`       | Stop, UserPromptSubmit                            | JSON            | Stable, in daily use.                                  |
| `hermes`      | post_llm_call, pre_llm_call                       | YAML            | Subagent traffic is not currently filtered (no child id on the upstream payload). |

`install` does not clobber existing config. `llmoji uninstall <provider>` removes the hooks and the settings entry. Journals and the per-instance cache are preserved; wipe those with `llmoji cache clear`.

### Hermes with custom hooks

`llmoji install hermes` merges its two entries into an existing populated `hooks:` block in `~/.hermes/config.yaml`. A hand-curated config with other event buckets (`tool_call:`, `subagent_stop:`, anything else) keeps every entry intact, including comments interleaved with the user's own entries. The merge is structural — each entry's `command:` field is the dedup key, so repeated installs are a no-op.

The implementation uses `ruamel.yaml` for parsing only, never for serialization. Edits apply as text splices on the original file at the line ranges ruamel's `lc.data` line/col marks pin down; the user's PyYAML-written wrap style, quoting, and surrounding comments stay byte-stable across install / uninstall. Earlier versions tried a load-mutate-dump approach via ruamel's `RoundTripDumper`; that path silently corrupted any double-quoted scalar PyYAML had wrapped at a non-whitespace boundary (kaomoji literals like `(◕‿◕)` inside personality prompts gained a single inserted space at the wrap point), and the parsing-only design is the fix.

Empty placeholder shapes (`hooks: {}` — the Hermes default — plus `hooks: []` and `hooks: ~`) are treated as "no hooks configured" and replaced with a populated block in place. The installer refuses (with a `SettingsCorruptError` calling out the shape) on:

- a top-level `hooks:` value that isn't a mapping (`hooks: enabled`, `hooks: [some_string]`)
- a flow-style hooks block (`hooks: {pre_llm_call: [...]}`)
- an event bucket that isn't a sequence
- an event bucket that's empty (`pre_llm_call: []` — surgical edit isn't well-defined when there's no anchor item to copy list-indent from)

Fix the file by hand and re-run.

`llmoji uninstall hermes` removes only entries whose `command:` field equals one of ours. The user's entries under the same event keys, and any other event buckets, stay untouched. If our entries were the only contents of an event bucket the empty bucket is dropped; if our entries were the only contents of the entire `hooks:` block the `hooks:` key is dropped.

Pre-1.2.x installs that used the old `# >>> llmoji begin (managed) >>>` / `# <<< llmoji end (managed) <<<` marker comments are handled transparently: the new install parses the structure inside the markers as a populated `hooks:` block and is idempotent against it. After a subsequent uninstall the marker comment lines themselves remain at column 0 (they're inert YAML comments outside our managed surface) and can be deleted by hand.

---

## Static dumps

To pull kaomoji out of a Claude.ai or ChatGPT data export:

```bash
llmoji parse --provider claude.ai ~/Downloads/data-...-batch-0000
llmoji parse --provider chatgpt ~/Downloads/chatgpt-export
```

Both exports happen to ship a file named `conversations.json`, with different schemas under the same filename; the parsers handle each. Rows land at `~/.llmoji/journals/claude_ai_export.jsonl` or `~/.llmoji/journals/chatgpt_export.jsonl`, and `llmoji analyze` picks them up alongside the live provider journals. The ChatGPT reader walks the message tree from `current_node` along the active branch only, so abandoned regenerations stay out of the corpus.

For Claude Code, Codex, or Hermes history that predates installing the live hook, the historical transcripts (`~/.claude/projects/**/*.jsonl`, `~/.codex/sessions/**/rollout-*.jsonl`, `~/.hermes/sessions/session_*.json`) can be replayed into the journals via the `llmoji.backfill` module.

---

## Custom harness

For harnesses we don't ship a first-class adapter for (notably OpenClaw and opencode):

- Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Strip the leading kaomoji from `assistant_text` on the way in (the prefix lives in the `kaomoji` field).
- Validate the prefix the same way the package does: `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically. Worked examples:

- [`examples/openclaw_plugin/`](examples/openclaw_plugin/) — OpenClaw plugin (`definePluginEntry` + `api.on("llm_output", …)` + `api.on("before_prompt_build", …)`, with subagent filtering via `subagent_spawned`/`subagent_ended`). Install via `openclaw plugins install <path-to-this-dir>`, then flip `plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess` to `true` in `~/.openclaw/config.json` so the conversation hooks (`llm_input`, `llm_output`) are routed to the plugin.
- [`examples/opencode_plugin.ts`](examples/opencode_plugin.ts) — opencode plugin (TS/JS plugins auto-loaded from `~/.config/opencode/plugins/`; uses the `event` and `experimental.chat.system.transform` hooks).

Both harnesses' plugin contracts are TypeScript-only with no shell-hook escape hatch, so first-class llmoji support would have to ship as a vendored plugin tree rather than the rendered-bash pattern the other providers use; the worked examples cover the same ground until then.

The Python module `llmoji.taxonomy` is the single source of truth for the validator and the leading-glyph set; rendered bash hooks (under `llmoji._hooks/`) read from it at install time. If you're porting the validator to another language for a harness like OpenClaw or opencode, please mirror the rules in `is_kaomoji_candidate` faithfully. The two TS examples above are byte-faithful ports as of llmoji v1.1.x. Bumping any of the rules is a cross-corpus invariant change on the package side and your port needs to follow.

---

## Tests

```bash
pytest tests/                      # everything
pytest tests/test_canonicalize.py  # rule-by-rule regression for canonicalize_kaomoji and extract
pytest tests/test_public_surface.py  # locks the cross-corpus invariant contract
```

The full suite runs anywhere. CI runs `ruff check .` and `pytest` on every PR.

The public-surface test exercises taxonomy invariants, synth-prompt content checks, the synthesizer factory dispatch, provider rendering plus `bash -n` validation of every hook template, the bundle allowlist, the corrupt-config refusal paths, and the unified `mask_kaomoji` prepend contract. The canonicalize tests run rule-by-rule.

---

## Prior art

Llmoji replicates and scales [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces), the original post that came up with the idea of prompting and tracking Claude's kaomoji use. The shared HuggingFace dataset extends that pipeline across many users, many harnesses, and many model releases.

---

## Contributing and security

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security and privacy, please see [SECURITY.md](SECURITY.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is CC-BY-SA-4.0. The shared corpus on [HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is also CC-BY-SA-4.0; running `llmoji upload --target hf` contributes a bundle under those terms.

If you use llmoji or the central corpus in published research, please cite this repository.
