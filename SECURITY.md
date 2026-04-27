# Security and privacy policy

## Reporting a vulnerability

If you've found a security issue in llmoji, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/llmoji/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I'll respond within a few days and aim to have a fix as soon as possible.

## Supported versions

Only the latest minor version on PyPI receives security and privacy fixes. If you're on an older version, the fix is to upgrade.

## Privacy model

llmoji is a privacy-sensitive tool. The whole point of the package is to ship aggregates from your machine to a shared corpus, so the privacy story is the threat model.

### What stays on your machine

- **Live journals** at `~/.<harness>/kaomoji-journal.jsonl`. These hold the raw `user_text` and `assistant_text` for every kaomoji-bearing turn. They never leave your machine. Hooks are append-only; nothing else writes to them.
- **Per-instance Haiku cache** at `~/.llmoji/cache/per_instance.jsonl`. Each row is a Haiku-paraphrased description of one masked-kaomoji user turn, keyed by content hash. The cache is never bundled and never shipped. Each row is, by construction, one user-turn paraphrase, so for a topic-narrow corpus a singleton row can leak specifics of that turn through paraphrase. The cache exists so re-runs of `analyze` only pay for new rows. `llmoji status` prints its size and entry count; `llmoji cache clear` is the explicit wipe.
- **Submission token** at `~/.llmoji/state.json`. A 256-bit random token generated on first `upload`, used as the salt for the submitter id. Never sent anywhere.

### What ships when you `upload`

The bundle has two files, both human-readable JSON:

- **`manifest.json`**: package version, journal counts per provider, canonical-kaomoji counts, the Haiku model id used, your free-form `--notes`, and a salted-hash submitter id.
- **`descriptions.jsonl`**: one row per canonical kaomoji, with the synthesized one-or-two-sentence meaning produced by Haiku from a pool of per-instance descriptions of that face.

The submitter id is a 32-hex-char (128-bit) salted hash of the per-machine token plus the package version. We do not collect HuggingFace usernames or any account-bound identifier. The salted hash is just enough to dedup repeat submissions from the same machine. An attacker who learns your submitter id cannot grind a same-id submission without your `state.json`.

### What does not ship

- Raw `user_text` or `assistant_text`.
- Per-instance Haiku descriptions (Stage A output).
- MiniLM embeddings or per-axis projections (those are research-side, applied to the bundle on the receiving end).
- The HuggingFace token used for upload.
- Any account-bound identifier.

### Singleton kaomoji caveat

For frequent kaomoji, the per-face Stage B synthesis pools many per-instance descriptions and so abstracts over many contexts. For a kaomoji that appears once in your corpus, the synthesized line IS effectively a paraphrase of that one user turn. Mitigations:

- `analyze` prints a per-face preview before declaring done.
- `upload` re-prompts before committing.
- The bundle is loose files in `~/.llmoji/bundle/`; `cat ~/.llmoji/bundle/descriptions.jsonl` is the audit.
- The bundle is allowlisted to two files: `tar_bundle` refuses to ship if anything else lives in the bundle dir.

We deliberately don't impose a count floor in the package. Filtering is an analysis-time concern on the receiving end, and shipping the raw distribution preserves more vocabulary. Please review `descriptions.jsonl` before running `upload` if your kaomoji distribution is long-tailed.

### Hooks are read-only

The bash hooks shipped with each provider read the harness's stop payload from stdin, append one row to a journal, and print an empty stdout. They never block the assistant turn, never modify the assistant's reply, and never call out to the network. The hook's exit code is always 0, so a malformed payload or a partially-quoted JSON value cannot stall your harness.

The hook templates are syntactically validated by `bash -n` in CI before each release.

### Settings files

`install` writes settings via an atomic tmp-file plus `os.replace`, so a power loss or `SIGINT` mid-write leaves your settings file with either the old content or the new, never half. Three pre-existing-config corruption paths are explicitly defended:

1. Malformed JSON in `~/.claude/settings.json` raises `SettingsCorruptError` rather than silently treating it as `{}`.
2. An existing unmanaged `[hooks.stop]` section in `~/.codex/config.toml` is detected and refused.
3. An existing top-level `hooks:` key in `~/.hermes/config.yaml` is detected and refused.

In all three cases you get a specific path and reason. Edit the file by hand and re-run.

## Model and API trust

`llmoji analyze` calls the Anthropic API with your `$ANTHROPIC_API_KEY`. The masked-kaomoji prompts and the user/assistant text from your journal are sent to Anthropic for paraphrasing. Please review Anthropic's data handling policy if this matters to you.

`llmoji upload --target hf` uses your HuggingFace token (via `huggingface-cli login` or `$HF_TOKEN`) to commit the bundle tarball to the central dataset. The token is not stored or echoed by llmoji; it's read by `huggingface_hub` from your standard HF credential cache.

`llmoji upload --target email` does not ship SMTP. It builds a `mailto:` URI with the bundle path printed in the body and asks you to attach the tarball manually via your system mail client.

## Receiving end

The HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) is public. Anything you ship through `llmoji upload --target hf` lands in a contributor-named subfolder (`contributors/<your-submitter-id>/bundle-<ts>.tar.gz`) and becomes publicly downloadable. Please review `descriptions.jsonl` before uploading.

If you upload a bundle and later want it removed from the dataset, please email mx@a9l.im with your submitter id (printed by `llmoji upload`) and I'll take down the matching tarballs.
