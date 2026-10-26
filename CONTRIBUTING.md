# Contributing to llmoji

Thank you very much for wanting to contribute! I really appreciate any contribution you would like to make, whether it's a PR, a bug report, or hook validation. New first-class providers are especially welcome.

## Dev setup

```bash
git clone https://github.com/a9lim/llmoji
cd llmoji
pip install -e ".[dev]"
```

The dev extras pull in `pytest`, `ruff` and `pyright`. There is no GPU dependency or network requirement for tests.

## Running tests

```bash
pytest tests/                            # everything, in about a second
pytest tests/test_canonicalize.py -v     # rule-by-rule taxonomy regression
pytest tests/test_capture.py -v          # transcript decoding + the installed hook
```

CI runs `ruff check .`, `pyright` and `pytest` on every PR. Tests point `HOME` and `$LLMOJI_HOME` at temp directories, so they never touch your real harness config or `~/.llmoji`.

## Adding a new provider

A provider is one `Provider` subclass under `llmoji/providers/`, registered in `llmoji.providers.PROVIDERS`. `llmoji/providers/base.py` documents the interface; `grok.py` and `antigravity.py` are small examples of a harness with its own registration file. A provider declares:

- `home_dir`, `hooks_dir`, `settings_path`, `journal_path` and `doc_path` (the harness's global instructions file), each with `home_path(...)`. A class-body `Path.home() / ...` is evaluated at import and escapes the test suite's `HOME` sandbox.
- `live_rows(payload)`: the kaomoji-led messages of the turn a Stop hook fired on, decoded from the transcript the payload points at. Build rows with `llmoji.scrape.kaomoji_row`, and stamp each with the message's own timestamp when the transcript has one, so the live row and its replay are the same row.
- If the harness keeps replayable history: `session_files()` and `replay(path)`, sharing the decoder `live_rows` uses. That makes it importable.
- If its registration isn't Claude Code's `hooks[<event>][i].hooks[j].command` shape: `_register`, `_unregister` and `_is_registered`. Settings writes go through `atomic_write_text` or `write_json`.
- `hook_stdout`, if the harness reads the hook's stdout as its reply.

Please include in the PR fixture-based tests in `tests/test_capture.py` for the transcript shapes (one turn, several messages in a turn, the turn boundary, subagents), an install round trip (covered automatically by `tests/test_install.py` once it's in `PROVIDERS`), and a short note on the harness version you checked against.

## Adding a static-export source

Export readers live under `llmoji/sources/` and turn one export format into rows; the CLI's `parse` writes them to `~/.llmoji/journals/`. Build rows with `kaomoji_row`, which splits the face from the body for you.

## PRs

- Please don't bump the version in your PR unless the change is intended to ship as a release. The PyPI publish workflow is triggered by a version update.
- Anything that touches `llmoji.taxonomy`, `llmoji.synth_prompts`, the journal schema, or the bundle schema is a cross-corpus change. Please flag it explicitly in the PR body. Those affect the HF dataset and need an edit on the dataset card too; a lexicon or schema change also bumps `LEXICON_VERSION`.

## Questions

Please reach out to me or open an issue. For anything security-sensitive or privacy-sensitive, please see [SECURITY.md](SECURITY.md).
