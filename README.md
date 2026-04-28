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

- **`llmoji install <provider>`**: write a `Stop` hook into your harness
- **`llmoji analyze`**: scrape and aggregate your logs
- **`llmoji upload --target {hf,email}`**: ship the bundle (HF: loose files via a single atomic commit; email: tarball as one attachment)

`analyze` needs an Anthropic API key in `$ANTHROPIC_API_KEY`; `upload --target hf` needs `$HF_TOKEN`. The email path tarballs the bundle and has you attach it manually.

---

## What this is for

The shared HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) collects kaomoji counts and a single summarized description per face, across many users' coding agents. The companion repo processes those descriptions. After you run `analyze`, you can inspect the files yourself at `~/.llmoji/bundle/descriptions.jsonl` before you choose to `upload`. 

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
llmoji analyze                             # scrape + canonicalize + Haiku summarize
llmoji upload --target hf                  # commit to a9lim/llmoji
# or:
llmoji upload --target email               # opens mailto:
```

`analyze` caches Haiku descriptions at `~/.llmoji/cache/per_instance.jsonl` keyed by content-hash. `llmoji cache clear` wipes it.

---

## Install

```bash
pip install llmoji
```

This requires Python 3.11+. The runtime dependency footprint is two packages: `anthropic` and `huggingface_hub`. Hooks run in `bash` and need `jq`.
From source:

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"      # adds pytest + ruff
```

---

## How it works

### Journal capture

Each provider has a `Stop` hook that fires once per assistant turn. The hook extracts the reply, strips the kaomoji from the body, and appends one JSONL row to `~/.<harness>/kaomoji-journal.jsonl`. The schema is the same across every provider:

```json
{"ts": "...", "model": "...", "cwd": "...", "kaomoji": "(◕‿◕)", "user_text": "...", "assistant_text": "..."}
```

### Haiku pipeline

`llmoji analyze` scrapes every installed provider's journal plus any extra JSONL files under `~/.llmoji/journals/`. For each (kaomoji, user, assistant) row saved, it uses Haiku to describe that specific instance. Then, it aggregates each unique kaomoji's descriptions and uses Haiku again to summarize an overall meaning. This summarized output is the only thing that ships in the bundle.

### Bundle structure

`analyze` writes to `~/.llmoji/bundle/`:

- **`manifest.json`**: package version, journal counts per provider, kaomoji counts, the Haiku model id used, anything you include as `--notes`, and the salted submitter id.
- **`descriptions.jsonl`**: one row per kaomoji, with the summarized meaning.

---

## What is and isn't uploaded

| Tier                                  | Where                                | Shipped on `upload`? |
|---------------------------------------|--------------------------------------|----------------------|
| Raw user and assistant text           | `~/.<harness>/kaomoji-journal.jsonl` | Never                |
| Per-instance Haiku paraphrase         | `~/.llmoji/cache/per_instance.jsonl` | Never                |
| Overall Haiku summaries and counts    | `~/.llmoji/bundle/`                  | Yes                  |

Please see [SECURITY.md](SECURITY.md) for the full privacy model.

---

## Providers

`llmoji install <provider>` writes the hook script and registers it with the harness's settings file, idempotently.

| Provider      | Hook events                                       | Settings format | Notes                                                  |
|---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|
| `claude_code` | Stop, UserPromptSubmit                            | JSON            | Stable, in daily use.                                  |
| `codex`       | Stop, UserPromptSubmit                            | JSON            | Stable, in daily use.                                  |
| `hermes`      | post_llm_call, pre_llm_call, subagent_stop        | YAML            | Implemented from docs; please see hermes notes below.  |

The Stop or `post_llm_call` hook is the journal logger. It reads each completed assistant turn, pulls off the leading kaomoji prefix, and appends a row to the harness's `kaomoji-journal.jsonl`. The UserPromptSubmit or `pre_llm_call` hook is a small "nudge" that injects a reminder on every turn, asking the model to begin its reply with a kaomoji. Both are optional, you can `uninstall` either at any time. The nudge is what makes the corpus large enough to be useful; without it the model drifts away from leading kaomoji over a long session.

`install` does not clobber existing config.

`llmoji uninstall <provider>` idempotently removes the hook and the settings entry. Journals and the per-instance cache are preserved; wipe those with `llmoji cache clear`.

---

## Static dumps

To pull kaomoji out of a Claude.ai data export:

```bash
llmoji parse --provider claude.ai ~/Downloads/data-...-batch-0000
```

The export's `conversations.json` is parsed and rows land at `~/.llmoji/journals/claude_ai_export.jsonl`. `llmoji analyze` picks this up alongside the live provider journals.

For Claude Code or Codex history that predates installing the live hook, the historical transcripts (`~/.claude/projects/**/*.jsonl`, `~/.codex/sessions/**/rollout-*.jsonl`) can be replayed into the journals via the `llmoji.backfill` module.

---

## Custom harness

For harnesses we don't ship a first-class adapter for (notably OpenClaw):

- Append one row per kaomoji-bearing assistant turn to `~/.llmoji/journals/<harness>.jsonl`.
- Use the canonical six-field schema: `{ts, model, cwd, kaomoji, user_text, assistant_text}`.
- Strip the leading kaomoji from `assistant_text` on the way in (the prefix lives in the `kaomoji` field).
- Validate the prefix the same way the package does: `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/` automatically. Please see [`examples/openclaw_hook.ts`](examples/openclaw_hook.ts) for a worked example.

---

## Tests

```bash
pytest tests/                      # everything
pytest tests/test_canonicalize.py  # rule-by-rule regression for canonicalize_kaomoji + extract
pytest tests/test_public_surface.py  # locks the v1.0 contract
```

The full suite runs anywhere. CI runs `ruff check .` and `pytest` on every PR.

The public-surface test exercises taxonomy invariants, haiku-prompt content checks, provider rendering plus `bash -n` validation of every hook template, the bundle allowlist, the corrupt-config refusal paths, and the unified `mask_kaomoji` prepend contract. The canonicalize tests run rule-by-rule.

---

## Contributing and security

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup. For security and privacy, please see [SECURITY.md](SECURITY.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). The companion research repo [`llmoji-study`](https://github.com/a9lim/llmoji-study) is AGPL-3.0-or-later. The shared corpus on [HuggingFace](https://huggingface.co/datasets/a9lim/llmoji) is CC-BY-SA-4.0; running `llmoji upload --target hf` contributes a bundle under those terms.

If you use llmoji or the central corpus in published research, please cite this repository.
