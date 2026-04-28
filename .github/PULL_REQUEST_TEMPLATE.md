## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Please link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `ruff check .` passes
- [ ] `pytest tests/` passes
- [ ] If touching `llmoji/_hooks/`, please confirm `bash -n` validates the rendered output
- [ ] If touching `llmoji.taxonomy`, `llmoji.synth_prompts`, the `Provider` interface, the journal schema, or the bundle schema: please flag this as a cross-corpus invariant change in the description (the HF dataset card needs a hand-edit on those)

## Notes

<!-- Anything reviewers should know: architectural decisions, followups, known limitations -->
