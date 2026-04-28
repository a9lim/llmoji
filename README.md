# llmoji

[![CI](https://github.com/a9lim/llmoji/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/llmoji/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmoji)](https://pypi.org/project/llmoji/)
[![Downloads](https://img.shields.io/pypi/dm/llmoji)](https://pypi.org/project/llmoji/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/llmoji/)

Llmoji is a small CLI that makes your agents cuter. (´-ω-`)

Llmoji configures your agent to start each message with a kaomoji. It locally saves them, and provides optional tools to summarize and upload the aggregated meaning per face to contribute to a shared database.

The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is where this data is processed.

There are three main commands:

- **`llmoji install <provider>`**: writes hooks to prompt for and record kaomoji
- **`llmoji analyze`**: scrape and aggregate your logs
- **`llmoji upload --target {hf,email}`**: ship the bundle (HF: loose files; email: tarball)

`analyze` needs a synthesis backend. The default uses Anthropic Haiku and reads `$ANTHROPIC_API_KEY`; `--backend openai` uses GPT-5.4 mini and reads `$OPENAI_API_KEY`; `--backend local` runs against any OpenAI-compatible endpoint (Ollama, vLLM, etc.) and needs `--base-url` and `--model`. `upload --target hf` needs `$HF_TOKEN`. The email path tarballs the bundle and has you attach it manually.

---

## What this is for

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
llmoji upload --target hf                  # commit to a9lim/llmoji
# or:
llmoji upload --target email               # opens mailto:
```

You can pick a different synthesis backend:

```bash
export OPENAI_API_KEY=...
llmoji analyze --backend openai            # GPT-5.4 mini via the Responses API
# or:
llmoji analyze --backend local \           # any OpenAI-compatible endpoint
  --base-url http://localhost:11434/v1 \
  --model llama3.1
```

`analyze` caches per-instance descriptions at `~/.llmoji/cache/per_instance.jsonl` keyed by content hash plus synthesis model id. `llmoji cache clear` wipes it.

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependency footprint is three packages: `anthropic`, `openai`, and `huggingface_hub`. Hooks run in `bash` and need `jq`.

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

### Synthesis pipeline

`llmoji analyze` scrapes every installed provider's journal plus any extra JSONL files under `~/.llmoji/journals/`. For each entry a model wrote, the chosen synthesizer describes that specific instance. Then, it aggregates the descriptions for each unique kaomoji per model and writes an overall meaning. This summarized output is the only thing that ships in the bundle.

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
  claude-sonnet-4-5-20250929.jsonl
  claude-haiku-4-5-20251001.jsonl
  gpt-5.4-mini-2026-03-17.jsonl
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

For harnesses we don't ship a first-class adapter for (notably OpenClaw):

- Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Strip the leading kaomoji from `assistant_text` on the way in (the prefix lives in the `kaomoji` field).
- Validate the prefix the same way the package does: `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically. Please see [`examples/openclaw_hook.ts`](examples/openclaw_hook.ts) for a worked example.

The Python module `llmoji.taxonomy` is the single source of truth for the validator and the leading-glyph set; rendered bash hooks (under `llmoji._hooks/`) read from it at install time. If you're porting the validator to another language for a harness like OpenClaw, please mirror the rules in `is_kaomoji_candidate` faithfully. Bumping any of them is a cross-corpus invariant change on the package side and your port needs to follow.

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
