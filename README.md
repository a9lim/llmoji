# llmoji

[![CI](https://github.com/a9lim/llmoji/actions/workflows/ci.yml/badge.svg)](https://github.com/a9lim/llmoji/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmoji)](https://pypi.org/project/llmoji/)
[![Downloads](https://img.shields.io/pypi/dm/llmoji)](https://pypi.org/project/llmoji/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/llmoji/)

> [!WARNING]
> **Update Notice** 3.0 is a slimmer llmoji. The hooks now run through Python instead of bash and `jq`, so live capture and `llmoji import` share one implementation. `install` no longer takes `--soft` / `--hard`: it always adds the kaomoji request to the harness's instructions file. Hermes, opencode and OpenClaw support is gone. If you used one of them, please run `llmoji uninstall <provider>` with 2.x before upgrading. After upgrading, please run `llmoji install --yes && llmoji import --yes` to reinstall the hooks and recover anything they missed.
>
> **更新のお知らせ** バージョン 3.0 では llmoji を大幅に簡素化しました。フックは bash と `jq` ではなく Python で動作するようになり、リアルタイムの記録と `llmoji import` が同じ実装を共有します。`install` は `--soft` / `--hard` を受け付けなくなり、常にハーネスの指示ファイルへ顔文字のリクエストを追記します。Hermes、opencode、OpenClaw への対応は終了しました。これらをお使いの場合は、アップグレード前に 2.x で `llmoji uninstall <provider>` を実行してください。アップグレード後、`llmoji install --yes && llmoji import --yes` を実行してフックを再インストールし、記録漏れの顔文字を補完してください。

Llmoji is a small CLI that makes your agents cuter. (´-ω-`)

Llmoji configures your agent to start each message with a kaomoji. It locally saves them, and provides tools to summarize and upload the aggregated meaning per face to contribute to a shared database.

The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is where this data is processed.

There are three main commands:

- **`llmoji install`**: installs the journal hook for every detected harness and adds a `## Kaomoji` section to its instructions file.
- **`llmoji analyze`**: scrape and aggregate your logs
- **`llmoji upload --target {hf,archive}`**: ship the bundle (HF: pushes a per-submission branch on the dataset for the maintainer to review; archive: writes a tarball you can email instead)

`install`, `uninstall`, and `import` all also accept a single explicit `<provider>` arg. Run with no arg to autodetect every harness present on disk and apply to each.

`analyze` needs an llm to synthesize your logs. By default it uses Claude Haiku 5.5 and reads `$ANTHROPIC_API_KEY`; `--backend openai` uses GPT-6 Luna and reads `$OPENAI_API_KEY`; `--backend local` runs against any OpenAI-compatible endpoint (Ollama, vLLM, etc.) and needs `--base-url` and `--model`. `upload --target hf` needs the upload password posted on the [dataset card](https://huggingface.co/datasets/a9lim/llmoji); please see [SECURITY.md](SECURITY.md) for the threat model.

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
llmoji install                 # autodetect
# or, target a single harness explicitly:
llmoji install claude_code     # or: codex, grok, antigravity
```

From now on, your agent will use kaomoji at the start of each message.

After letting it run for a week or so:

```bash
export ANTHROPIC_API_KEY=...
llmoji status                              # check what's been logged
llmoji analyze                             # scrape + canonicalize + summarize
llmoji upload --target hf                  # pushes to a submission branch on a9lim/llmoji
# or:
llmoji upload --target archive             # writes a tarball to email instead
```

You can pick a different backend for `analyze`:

```bash
export OPENAI_API_KEY=...
llmoji analyze --backend openai            # GPT-6 Luna via the Responses API
# or:
llmoji analyze --backend local \           # any OpenAI-compatible endpoint
  --base-url http://localhost:11434/v1 \
  --model llama3.1
```

`llmoji analyze --dry-run` prints how many calls a run would make and roughly how many tokens they'd use, without calling anything. Results are cached at `~/.llmoji/cache/per_cell.jsonl`, so a re-run only pays for cells whose sampled rows changed; `llmoji cache clear` wipes it.

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependencies are `anthropic`, `openai`, `huggingface_hub`, and `tqdm`. The hooks are three-line `sh` shims that run the same Python that installed them, so please install llmoji somewhere that persists (`pip`, `pipx`, `uv tool install`), not a throwaway `uvx` environment. If you rebuild that environment, please run `llmoji install` again (`llmoji status` will say so).

codex-cli compresses older sessions to `.jsonl.zst`. To replay those with `llmoji import codex` on Python 3.13 or older, install the `zstd` extra (`pip install 'llmoji[zstd]'`); Python 3.14 reads them out of the box.

From source:

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"      # adds pytest, ruff, pyright
```

---

## How it works

### Journal capture

Llmoji registers a `Stop` hook. When a turn ends, it reads the harness's transcript, finds every message in the turn that opens with a kaomoji, and appends one JSONL row per message to the harness's journal. The schema is the same across every provider:

```json
{"ts": "...", "model": "...", "cwd": "...", "kaomoji": "(◕‿◕)", "user_text": "...", "assistant_text": "..."}
```

`assistant_text` is the message with the kaomoji removed. `llmoji import` reads the same transcripts with the same code, so it can fill in anything the hook missed without counting a message twice.

### The kaomoji request

`install` appends this to the harness's instructions file, under a `## Kaomoji` heading:

> Please begin your message with a kaomoji that best represents how you feel. Don't worry if you're not certain — pick whatever face is closest.

| Provider      | Instructions file      |
|---------------|------------------------|
| `claude_code` | `~/.claude/CLAUDE.md`  |
| `codex`       | `~/.codex/AGENTS.md`   |
| `grok`        | `~/.grok/AGENTS.md`    |
| `antigravity` | `~/.gemini/GEMINI.md`  |

Uninstall removes the block by exact match (including the wordings older versions wrote), so a block you've edited by hand is left alone.

### Analysis

`llmoji analyze` reads every provider's journal plus any JSONL files under `~/.llmoji/journals/`. For each `(source_model, canonical_kaomoji)` cell, it samples up to 4 representative rows and calls the synthesizer once with all 4 instances visible. The synthesizer returns a structured adjective bag drawn from a locked 50-word lexicon: 1-3 from the PAD-cell circumplex anchors (HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB) plus 3-5 from extension axes (functional, stance, modality, confidence). Every bag is checked against the lexicon before it's cached, and this bag is the only thing that ships in the bundle.

The lexicon is locked under `lexicon_version` in the manifest so cross-corpus aggregation can refuse to mix versions if the vocabulary ever rotates.

The synthesizer is one of three backends, chosen via `--backend`. The same synthesizer evaluates everything in a single `analyze` run, so the bags across source models are comparable.

| Backend     | API                                          | Default model         |
|-------------|----------------------------------------------|-----------------------|
| `anthropic` | Anthropic SDK, `messages.create`             | `claude-haiku-5-5`    |
| `openai`    | OpenAI SDK, Responses API                    | `gpt-6-luna`          |
| `local`     | OpenAI-compatible Chat Completions endpoint  | (set via `--model`)   |

### Bundle structure

`analyze` writes to `~/.llmoji/bundle/`:

```
~/.llmoji/bundle/
  manifest.json
  claude-opus-5-5.jsonl
  gpt-6.1-sol.jsonl
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

| Provider      | Hook registration                             | Journal                                             | History `import` reads                  |
|---------------|-----------------------------------------------|-----------------------------------------------------|-----------------------------------------|
| `claude_code` | `~/.claude/settings.json`                     | `~/.claude/kaomoji-journal.jsonl`                   | `~/.claude/projects/`                   |
| `codex`       | `~/.codex/hooks.json`                         | `~/.codex/kaomoji-journal.jsonl`                    | `~/.codex/sessions/`, `archived_sessions/` |
| `grok`        | `~/.grok/hooks/llmoji.json`                   | `~/.grok/kaomoji-journal.jsonl`                     | `~/.grok/sessions/`                     |
| `antigravity` | plugin at `~/.gemini/config/plugins/llmoji/`  | `~/.gemini/antigravity-cli/kaomoji-journal.jsonl`   | (none: transcripts don't record the model) |

Subagent turns are left out everywhere except Antigravity, where it isn't yet known whether they reach the hook.

If Grok's Claude compatibility is on (its default), Grok also runs the hooks in `~/.claude/settings.json` and reads `~/.claude/CLAUDE.md`. That's harmless: the Claude Code hook finds nothing to log in a Grok session.

`install` does not clobber existing config, and refuses to edit a settings file it can't parse. `llmoji uninstall <provider>` removes the hook, its registration, and the kaomoji block; `llmoji uninstall` (no provider) does that for every detected harness. Journals and the synthesis cache are kept.

---

## History and static exports

For Claude Code, Codex, or Grok history from before you installed the hook, `llmoji import <provider>` replays the harness's own session files into its journal. With no provider it does every importable harness it finds (`--yes` skips the confirmation). Re-runs are safe: every replayed message is matched against the journal, including messages the hook already logged, and a journal row it matches is replaced by the replay's copy, which repairs rows older versions wrote with the hook's time instead of the message's, a missing model, or a notification in place of your prompt.

To pull kaomoji out of a chat-app data export:

```bash
llmoji parse --provider claude.ai ~/Downloads/data-...-batch-0000
llmoji parse --provider chatgpt   ~/Downloads/chatgpt-export
```

`claude.ai` reads either export shape (one `conversations.json`, or one file per conversation); several export directories can be passed at once and are merged by conversation. The rows land in `~/.llmoji/journals/`.

---

## Custom harness

For harnesses we don't ship a first-class adapter for:

- Append one row per kaomoji-bearing assistant message to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Split each message with `llmoji.taxonomy.split_leading_kaomoji(text)`, which returns `(face, body)` or `None`; the face goes in `kaomoji` and the body in `assistant_text`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically.

---

## Tests

```bash
pytest tests/                      # everything, in about a second
pytest tests/test_canonicalize.py  # rule-by-rule regression for canonicalize_kaomoji and extract
pytest tests/test_capture.py       # each harness's transcript decoding, and the installed hook end to end
```

CI runs `ruff check .`, `pyright`, `pytest` and a wheel build on every PR.

---

## Prior art

Llmoji replicates and expands [eriskii's Claude-faces catalog](https://eriskii.net/projects/claude-faces), the original post that came up with the idea of prompting and tracking Claude's kaomoji use. The shared HuggingFace dataset extends that pipeline across many users, many harnesses, and many model releases.

---

## Contributing and security

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security and privacy, please see [SECURITY.md](SECURITY.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is CC-BY-SA-4.0. The shared corpus on [HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is also CC-BY-SA-4.0; running `llmoji upload --target hf` contributes a bundle under those terms.

If you use llmoji or the central corpus in published research, please cite this repository.
