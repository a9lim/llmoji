# Security and privacy policy

## Reporting a vulnerability

If you've found a security issue in llmoji, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/llmoji/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I'll respond within a few days and aim to have a fix as soon as possible.

## Supported versions

Only the latest minor version on PyPI receives security and privacy fixes. If you're on an older version, please upgrade.

## Privacy model

llmoji is a privacy-sensitive tool. The package ships aggregates from your machine to a shared corpus so privacy is important here.

### What stays on your machine

- **Raw `user_text` or `assistant_text`** at `~/.<harness>/kaomoji-journal.jsonl`. These hold the raw data for every kaomoji-bearing turn. They never leave your machine.
- **Per-instance Haiku cache** at `~/.llmoji/cache/per_instance.jsonl`. Each row is a Haiku-paraphrased description of one turn, keyed by content hash. The cache is never bundled and never shipped. `llmoji status` prints its size and entry count; `llmoji cache clear` is the explicit wipe.
- **Submission token** at `~/.llmoji/state.json`. A 256-bit random token generated on first `upload`, used as the salt for the submitter id. Never sent anywhere.

### What ships when you `upload`

The bundle has two files, both human-readable JSON:

- **`manifest.json`**: package version, journal counts per provider, kaomoji counts, the Haiku model id used, any included `--notes`, and a salted-hash submitter id.
- **`descriptions.jsonl`**: one row per kaomoji, with the summary produced by Haiku.

The submitter id is a 32-hex-char (128-bit) salted hash of the per-machine token plus the package version. We do not collect HuggingFace usernames or any account-bound identifier.

### Singleton kaomoji caveat

For frequent kaomoji, the per-face summary pools many instances and so abstracts over many contexts. For a kaomoji that appears once, the synthesized line IS effectively a paraphrase of that one turn. Mitigations:

- `analyze` prints a per-face preview before declaring done.
- `upload` re-prompts before committing.
- The bundle is inspectable in `~/.llmoji/bundle/`
- The bundle is allowlisted to two files: `tar_bundle` refuses to ship if anything else is in the bundle dir.

Please review `~/.llmoji/bundle/descriptions.jsonl` before running `upload` if your kaomoji distribution is long-tailed.

### Hooks are read-only

The bash hooks shipped with each provider append one row to a journal. They never block the turn, modify the reply, or call out to the network. 

## Model and API trust

`llmoji analyze` calls the Anthropic API with your `$ANTHROPIC_API_KEY`. The masked-kaomoji prompts and the user/assistant text from your journal are sent to Anthropic for paraphrasing. Please review Anthropic's data handling policy if this matters to you.

`llmoji upload --target hf` uses your `$HF_TOKEN` to commit the bundle tarball to the central dataset. The token is not stored or echoed by llmoji; it's read by `huggingface_hub` from your standard HF credential cache.

`llmoji upload --target email` builds a `mailto:` URI with the bundle path printed in the body and asks you to attach the tarball manually.

## Receiving end

The HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) is public. Anything you ship through `llmoji upload --target hf` lands in a subfolder (`contributors/<your-submitter-id>/bundle-<ts>.tar.gz`) and becomes publicly downloadable. Please review `~/.llmoji/bundle/descriptions.jsonl` before uploading.

If you upload a bundle and later want it removed from the dataset, please email mx@a9l.im with your submitter id and I'll take down the matching tarballs.
