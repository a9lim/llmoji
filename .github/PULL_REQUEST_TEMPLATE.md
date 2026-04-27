## What

<!-- One or two sentences on what changed -->

## Why

<!-- What problem does this solve? Please link issues with "Fixes #N" if applicable -->

## Test plan

- [ ] `ruff check .` passes
- [ ] `pytest tests/` passes
- [ ] If touching `llmoji/_hooks/`, please confirm `bash -n` validates the rendered output
- [ ] If touching `llmoji.taxonomy`, `llmoji.haiku_prompts`, the `Provider` interface, the journal schema, or the bundle schema: please flag this as a v1.0 frozen-surface change in the description (gates a major version bump)

## Notes

<!-- Anything reviewers should know: architectural decisions, followups, known limitations -->
