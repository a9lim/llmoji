# llmoji

A small CLI for collecting kaomoji journals from coding agents (Claude
Code, Codex, Hermes), distilling them into per-canonical-form Haiku
descriptions, and submitting privacy-preserving aggregates to a shared
research corpus. Companion to the [llmoji-study][study] research repo.

[study]: https://github.com/a9lim/llmoji-study

## What it does

If your agent is configured to start each message with a kaomoji that
reflects how it's currently feeling, this package:

1. Installs a stop-event hook in your harness (`claude_code`, `codex`,
   or `hermes`).
2. Logs one row per kaomoji-bearing assistant turn to a local JSONL
   journal at `~/.<harness>/kaomoji-journal.jsonl`. Schema:

       {ts, model, cwd, kaomoji, user_text, assistant_text}

   Stays on your machine. Hooks are read-only — they never block or
   modify your turn.
3. On `llmoji analyze`, canonicalizes kaomoji to a frozen v1.0
   equivalence-class scheme, then runs a two-stage Haiku pipeline:
   a per-instance description of each masked face, then a per-face
   synthesis pooling the descriptions. The synthesized line is the
   only thing that ships.
4. Writes a bundle (`manifest.json` + `descriptions.jsonl`) to
   `~/.llmoji/bundle/` for you to inspect — loose files, no opaque
   archive — and on `llmoji upload --target {hf,email}` tarballs and
   submits to either a public HF dataset (`a9lim/llmoji`) or
   `mailto:mx@a9l.im`.

## Install

    pip install llmoji
    llmoji install claude_code      # or: codex, hermes

`llmoji status` shows what's installed, journal sizes, and whether a
bundle is ready to inspect.

## Day-to-day flow

    llmoji status                   # check what's been logged
    llmoji analyze                  # build the bundle (Haiku-driven)
    cat ~/.llmoji/bundle/descriptions.jsonl    # review
    llmoji upload --target hf       # commit to a9lim/llmoji
    # ... or:
    llmoji upload --target email    # opens mailto: with attach hint

`analyze` caches per-instance Haiku descriptions at
`~/.llmoji/cache/per_instance.jsonl` keyed by content-hash, so re-runs
only pay for new rows. `llmoji cache clear` if you ever want to wipe
it.

## What does NOT leave your machine

| Tier                  | Where                                       | Shipped? |
|-----------------------|---------------------------------------------|----------|
| Live journal          | `~/.<harness>/kaomoji-journal.jsonl`        | Never    |
| Per-instance Haiku    | `~/.llmoji/cache/per_instance.jsonl`        | Never    |
| Bundle (counts + per-face synthesis) | `~/.llmoji/bundle/`            | Yes, on `upload` |

The bundle ships per-canonical-kaomoji counts plus the synthesized
description (one line per face). It does NOT ship raw `user_text`,
raw `assistant_text`, per-instance descriptions, MiniLM embeddings,
or per-axis projections (those are research-side, applied to the
bundle on the receiving end).

For frequent kaomoji, the synthesis abstracts over many contexts. For
singletons, it IS a paraphrase of one user turn — `analyze` prints a
preview and `upload` re-prompts so you can review before shipping. We
deliberately don't impose a count floor; filtering is an analysis-time
concern on the receiving end, and shipping the raw distribution
preserves more vocabulary.

## Providers

`llmoji install <provider>` writes the hook script and registers it
with the harness's settings file, idempotently.

| Provider      | Hook event       | Settings format | Notes                          |
|---------------|------------------|-----------------|--------------------------------|
| `claude_code` | Stop             | JSON            | Stable, in daily use           |
| `codex`       | Stop             | TOML            | Stable, in daily use           |
| `hermes`      | post_llm_call    | YAML            | Implemented from docs; needs live-traffic validation before claiming stability |

Subagent / sidechain dispatches are filtered per-provider:

  - `claude_code`: `isSidechain` field flag.
  - `codex`: no subagent concept.
  - `hermes`: session-id correlation against `delegate_task`.

## Static dumps

To pull historical kaomoji turns out of a Claude.ai data export:

    llmoji parse --provider claude.ai \
      ~/Downloads/data-...-batch-0000

The export's `conversations.json` is parsed and rows land at
`~/.llmoji/journals/claude_ai_export.jsonl`. `llmoji analyze` picks
this up alongside the live provider journals.

For Claude Code or Codex history that predates installing the live
hook, the historical transcripts (`~/.claude/projects/**/*.jsonl`,
`~/.codex/sessions/**/rollout-*.jsonl`) can be replayed into the
journals via the `llmoji.backfill` module — see
[`llmoji-study/scripts/21_backfill_journals.py`][backfill] for the
pattern.

[backfill]: https://github.com/a9lim/llmoji-study/blob/main/scripts/21_backfill_journals.py

## Custom harness — generic JSONL contract

For harnesses we don't ship a first-class adapter for (notably
OpenClaw, whose hook contract is TS-shaped), there's a published
contract:

  - Append one row per kaomoji-bearing assistant turn to
    `~/.llmoji/journals/<harness>.jsonl`.
  - Use the canonical 6-field schema (`ts, model, cwd, kaomoji,
    user_text, assistant_text`).
  - Validate the leading kaomoji prefix the same way the package
    does — `llmoji.taxonomy.is_kaomoji_candidate(prefix)`.

`llmoji analyze` picks up everything under `~/.llmoji/journals/`
automatically. No first-class adapter required.

## v1.0 frozen public surface

These are stable across the v1 series. Bumping any of them is a
major version bump, and the central HF dataset's aggregation rules
declare "v1 corpus only":

  - The kaomoji canonicalization rules in `llmoji.taxonomy.canonicalize_kaomoji`
  - `llmoji.taxonomy.KAOMOJI_START_CHARS` and `is_kaomoji_candidate`
  - Haiku per-instance + synthesis prompts in `llmoji.haiku_prompts`
  - `HAIKU_MODEL_ID` (the locked Haiku checkpoint)
  - Per-provider system-injection prefix lists
  - The 6-field unified journal row schema
  - The `Provider` interface
  - The bundle schema (`manifest.json`, `descriptions.jsonl`)

Free to change without bumping the major version: the per-instance
cache key derivation, internal command flags beyond what's listed
above, anything in `llmoji-study` (research-side).

## Links

- Companion research repo: https://github.com/a9lim/llmoji-study
- Author: a9lim — `mx@a9l.im`
- License: MIT
