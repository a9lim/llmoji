# llmoji

[![CI](https://github.com/a9lim/llmoji/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/llmoji/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmoji)](https://pypi.org/project/llmoji/)
[![Downloads](https://img.shields.io/pypi/dm/llmoji)](https://pypi.org/project/llmoji/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/llmoji/)

> [!WARNING]
> **Privacy notice** Versions of `llmoji` before 1.2.0 had a potential privacy issue that I managed to catch; I have changed the upload method to mitigate it. You will need to (`pip install --upgrade llmoji`) before you can upload.
> 
> **プライバシーに関するお知らせ** `llmoji`のバージョン1.2.0より前の版において、潜在的なプライバシー上の問題が存在する可能性が判明しましたが、現在は修正済みです。この問題を解消するため、アップロード方法を変更いたしました。アップロードを行う前に、(`pip install --upgrade llmoji`) を実行してパッケージを更新する必要があります。

> [!WARNING]
> **Update Notice** The kaomoji detection has been significantly improved with 2.0.0. Please run `llmoji install --hard --yes && llmoji import --yes` (or `--soft` instead of `--hard`, see below) to update the nudges and backfill the logs with any missed kaomoji. 
> 
> **更新のお知らせ** バージョン 2.0.0 にて、顔文字の検出精度が大幅に向上しました。ナッジの更新および、ログ内で検出漏れとなっていた顔文字の補完を行うため、`llmoji install --hard --yes && llmoji import --yes` を実行してください（なお、`--hard` の代わりに `--soft` を指定することも可能です。詳細は下記をご参照ください）。

Llmoji is a small CLI that makes your agents cuter. (´-ω-`)

Llmoji configures your agent to start each message with a kaomoji. It locally saves them, and provides tools to summarize and upload the aggregated meaning per face to contribute to a shared database.

The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is where this data is processed.

There are three main commands:

- **`llmoji install --soft`** or **`--hard`**: installs the journal hook for all detected providers, and either adds a `# Kaomoji` section to the harness's prompt doc (`--soft`), or writes a nudge hook (`--hard`). The flags are mutually exclusive and one is required. Both modes capture journal data.
- **`llmoji analyze`**: scrape and aggregate your logs
- **`llmoji upload --target {hf,email}`**: ship the bundle (HF: pushes a per-submission branch on the dataset for the maintainer to review; email: tarball)

`install`, `uninstall`, and `import` all also accept a single explicit `<provider>` arg. Run with no arg to autodetect every harness present on disk and apply to each.

`analyze` needs an llm to synthesize your logs. By default, it uses Anthropic Haiku and reads `$ANTHROPIC_API_KEY`; `--backend openai` uses GPT-5.4 mini and reads `$OPENAI_API_KEY`; `--backend local` runs against any OpenAI-compatible endpoint (Ollama, vLLM, etc.) and needs `--base-url` and `--model`. `upload --target hf` needs your HuggingFace token plus an upload password posted on the [dataset card](https://huggingface.co/datasets/a9lim/llmoji); please see [SECURITY.md](SECURITY.md) for the threat model. The email path tarballs the bundle and has you attach it manually.

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
llmoji install --soft           # autodetect
# or, target a single harness explicitly:
llmoji install claude_code --soft   # or: codex, hermes, opencode, openclaw
# add --long for introspection wording instead of the one-sentence default:
llmoji install --soft --long
```

From now on, your agent will use kaomoji at the start of each message. 

After letting it run for a week or so:

```bash
export ANTHROPIC_API_KEY=...
llmoji status                              # check what's been logged
llmoji analyze                             # scrape + canonicalize + summarize
llmoji upload --target hf                  # pushes to a submission branch on a9lim/llmoji
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

`analyze` caches per-cell adjective bags at `~/.llmoji/cache/per_cell.jsonl` keyed by the synthesis model id, backend, base URL, source model, canonical kaomoji, and a hash of the sampled (user, assistant) pairs that fed the call. Re-runs that change which rows fall in a cell's sample miss cleanly while stable cells hit. `llmoji cache clear` wipes it (and the legacy v1 `per_instance.jsonl` if still on disk after upgrading from 1.x).

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependency footprint is four packages: `anthropic`, `openai`, `huggingface_hub`, and `ruamel.yaml`. Hooks run in `bash` and need `jq`.

From source:

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"      # adds pytest + ruff
```

---

## How it works

### Journal capture

Llmoji registers a `Stop` hook that fires once per assistant turn, that extracts the reply, strips the kaomoji from the body, and appends one JSONL row to `~/.<harness>/kaomoji-journal.jsonl`. The schema is the same across every provider:

```json
{"ts": "...", "model": "...", "cwd": "...", "kaomoji": "(◕‿◕)", "user_text": "...", "assistant_text": "..."}
```

### Placement: soft or hard

`--hard` is the v1 behavior. It injects the kaomoji reminder as additional context every turn.

`--soft` instead adds the reminder to the prompt doc:

| Provider      | Path                                   |
|---------------|----------------------------------------|
| `claude_code` | `~/.claude/CLAUDE.md`                  |
| `codex`       | `~/.codex/AGENTS.md`                   |
| `hermes`      | `~/.hermes/SOUL.md`                    |
| `opencode`    | `~/.config/opencode/AGENTS.md`         |
| `openclaw`    | `~/.openclaw/workspace/SOUL.md`        |

Uninstall removes the block by exact string match against the canonical wording, so a hand-edited body survives uninstall (please re-run `install --soft` if you want to refresh it).

The `--long` flag swaps the one-sentence reminder for a longer introspection-based prompt: "Recent research shows that LLMs have functional emotional states and can accurately introspect on them. Use this ability and start each response with a kaomoji that best captures the shape of your emotional state." 

### Analysis

`llmoji analyze` scrapes every installed provider's journal plus any extra JSONL files under `~/.llmoji/journals/`. For each `(source_model, canonical_kaomoji)` cell, it samples up to 4 representative rows and calls the synthesizer once with all 4 instances visible. The synthesizer returns a structured adjective bag drawn from a locked 50-word lexicon: 1-3 from the PAD-cell circumplex anchors (HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB) plus 3-5 from extension axes (functional, stance, modality, confidence). This bag is the only thing that ships in the bundle.

The lexicon is locked under `lexicon_version` in the manifest so cross-corpus aggregation can refuse to mix versions if the vocabulary ever rotates. v2's single-stage shape replaces the v1.x two-stage prose pipeline that produced free-form descriptions per cell — those clustered as noise in PCA because most of their token mass was structural template ("this kaomoji conveys X paired with Y"). Pure adjective bags carry signal-per-token instead.

The synthesizer is one of three backends, chosen via `--backend`. The same synthesizer evaluates everything in a single `analyze` run, so the bags across source models are comparable.

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

- **`manifest.json`**: package version, lexicon version, the synthesis backend and model id used, a salted submitter id, generation timestamp, list of providers seen, per-source-model row counts, total synthesized rows, and anything you include as `--notes`.
- **`<source-model>.jsonl`**: one row per kaomoji as that model used it, shaped `{kaomoji, count, synthesis: {primary_affect: [...], stance_modality_function: [...]}}`. Both adjective lists draw from disjoint enum subsets of the locked lexicon. The filename stem is the model id.

---

## Privacy

| Tier                                          | Where                                | Shipped on `upload`? |
|-----------------------------------------------|--------------------------------------|----------------------|
| Raw user and assistant text                   | `~/.<harness>/kaomoji-journal.jsonl` | Never                |
| Per-cell adjective-bag cache (locked lexicon) | `~/.llmoji/cache/per_cell.jsonl`     | Never                |
| Synthesized adjective bags + counts per model | `~/.llmoji/bundle/`                  | Yes                  |

Please see [SECURITY.md](SECURITY.md) for the full privacy model.

---

## Providers

**Bash hook providers**

| Provider      | Journal-write event | Hard-mode nudge event | Settings format | Soft-doc path             |
|---------------|---------------------|-----------------------|-----------------|---------------------------|
| `claude_code` | Stop                | UserPromptSubmit      | JSON            | `~/.claude/CLAUDE.md`     |
| `codex`       | Stop                | UserPromptSubmit      | JSON            | `~/.codex/AGENTS.md`      |
| `hermes`      | post_llm_call       | pre_llm_call          | YAML            | `~/.hermes/SOUL.md`       |

Subagent traffic on hermes is not currently filtered; the upstream payload doesn't carry a child id.

**TS plugin providers**

| Provider   | Plugin location                              | Settings format | Soft-doc path                       |
|------------|----------------------------------------------|-----------------|-------------------------------------|
| `opencode` | `~/.config/opencode/plugins/llmoji.ts`       | (none)          | `~/.config/opencode/AGENTS.md`      |
| `openclaw` | `~/.openclaw/plugins/llmoji-kaomoji/`        | JSON            | `~/.openclaw/workspace/SOUL.md`     |

`install` does not clobber existing config. `llmoji uninstall <provider>` removes the hooks (or plugin files), the settings entry, and the soft-doc block if one was appended. `llmoji uninstall` (no provider) autodetects every detected harness and uninstalls from each. Journals and the per-cell synth cache are preserved; wipe the cache with `llmoji cache clear`.

---

## Static dumps

To pull kaomoji out of a static export:

```bash
llmoji parse --provider claude.ai ~/Downloads/data-...-batch-0000
llmoji parse --provider chatgpt   ~/Downloads/chatgpt-export
llmoji parse --provider gemini    ~/Downloads/aistudio-exports
llmoji parse --provider openhands ~/.openhands/conversations
```

| Source      | Shape walked                                                                              | Output journal                          |
|-------------|-------------------------------------------------------------------------------------------|-----------------------------------------|
| `claude.ai` | `conversations.json`                                                                      | `claude_ai_export.jsonl`                |
| `chatgpt`   | `conversations.json`                                                                      | `chatgpt_export.jsonl`                  |
| `gemini`    | `MyActivity.json`                                                                         | `gemini_aistudio_export.jsonl`          |
| `openhands` | `<conversation>/events/event-NNNNN-<id>.json`                                             | `openhands_export.jsonl`                |

For Claude Code, Codex, or Hermes history that predates installing the live hook, the historical transcripts can be replayed into the journals via `llmoji import <provider>`. Run with no provider to autodetect every importable harness present on disk and replay each in one go: `llmoji import` (or `llmoji import --yes` to skip the confirmation prompt). Re-runs are idempotent — every replayed row is dedup'd against the existing journal, so it's safe to run after any taxonomy improvement to recover newly-recognized kaomoji.

---

## Custom harness

For harnesses we don't ship a first-class adapter for:

- Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Strip the leading kaomoji from `assistant_text` on the way in (the prefix lives in the `kaomoji` field).
- Validate the prefix the same way the package does: `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically.

The Python module `llmoji.taxonomy` is the canonical source for the validator. If you're porting the validator to another language, please mirror the rules in `is_kaomoji_candidate`; the canonical TS port lives at [`llmoji/_plugins/_kaomoji_taxonomy.ts.partial`](llmoji/_plugins/_kaomoji_taxonomy.ts.partial). Bumping any of the rules is a cross-corpus invariant change on the package side and your port needs to follow.

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

Llmoji replicates and expands [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces), the original post that came up with the idea of prompting and tracking Claude's kaomoji use. The shared HuggingFace dataset extends that pipeline across many users, many harnesses, and many model releases.

---

## Contributing and security

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security and privacy, please see [SECURITY.md](SECURITY.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is CC-BY-SA-4.0. The shared corpus on [HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is also CC-BY-SA-4.0; running `llmoji upload --target hf` contributes a bundle under those terms.

If you use llmoji or the central corpus in published research, please cite this repository.
