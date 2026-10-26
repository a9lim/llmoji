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
- **Per-cell synthesizer cache** at `~/.llmoji/cache/per_cell.jsonl`. Each row holds a structured adjective bag (`primary_affect` + `stance_modality_function`) drawn from the locked LEXICON, keyed by the synthesis model id, backend, base URL, source model, canonical kaomoji, and a hash of the sampled (user, assistant) pairs that fed the call. The bag itself contains no free-form text and no journal content, only adjectives from the corpus vocabulary; every bag is checked against the lexicon before it is cached. The cache is never bundled and never shipped. `llmoji status` prints its size; `llmoji cache clear` is the explicit wipe.
- **Submission token** at `~/.llmoji/.salt`. A 256-bit random token generated on first `upload`, used as the salt for the submitter id. Never sent anywhere.

### What ships when you `upload`

The bundle is human-readable JSON, laid out flat: one top-level manifest plus one `.jsonl` per source model, no subdirectories.

- **`manifest.json`**: package version, lexicon version, the synthesis backend and model id used, per-source-model row counts, total synthesized rows, list of providers seen, generation timestamp, any included `--notes`, and a salted-hash submitter id.
- **`<source-model>.jsonl`**: one row per canonical kaomoji as that model used it, shaped `{kaomoji, count, synthesis: {primary_affect: [...], stance_modality_function: [...]}}`. Both arrays draw exclusively from the locked LEXICON enum, so no free-form text and no journal content lands in a bundle row.

The submitter id is a 32-hex-char (128-bit) salted hash of the per-machine token plus the package version. We do not collect HuggingFace usernames or any account-bound identifier.

### Singleton kaomoji caveat

In v1.x the per-face summary was free-form prose, and for a kaomoji that appeared once, the synthesized line was effectively a paraphrase of that one turn — a small singleton-leak surface. v2 removed this surface entirely: the row carries only adjectives from a 50-word locked vocabulary, and llmoji refuses any synthesis that strays outside it (a local model decoding without constraints could otherwise echo journal text). A singleton kaomoji's row is a bag like `{primary_affect: ["frustrated"], stance_modality_function: ["wry", "self-correcting", "tender"]}` rather than a paraphrase of the underlying turn. Even so, please continue to review the bundle before uploading:

- `analyze` prints a per-face preview before declaring done.
- `upload` re-prompts before committing.
- The bundle is inspectable in `~/.llmoji/bundle/`.
- The bundle is allowlisted: top-level `manifest.json` plus per-model `.jsonl` files at the root. Both upload targets refuse to ship if anything else is in the bundle dir.

Please review every `~/.llmoji/bundle/<source-model>.jsonl` before running `upload`.

### Hooks are read-only

The hook installed for each provider appends rows to that harness's journal. It never blocks the turn, modifies the reply, or calls out to the network, and it always exits successfully; anything that goes wrong is logged to `~/.llmoji/hook-errors.log`.

## Model and API trust

`llmoji analyze` sends the masked-kaomoji prompts and the user and assistant text from your journal to whichever synthesis backend you pick. Three backends are supported and each one routes your data differently. Please review the relevant data-handling policy before running `analyze` against a corpus you care about.

- **`--backend anthropic` (default)**: calls the Anthropic API with your `$ANTHROPIC_API_KEY`. Your journal text goes to Anthropic for paraphrasing.
- **`--backend openai`**: calls the OpenAI Responses API with your `$OPENAI_API_KEY`. Your journal text goes to OpenAI for paraphrasing.
- **`--backend local`**: calls a local OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp's HTTP server, etc.) at the `--base-url` you pass. Your journal text stays on whatever machine the endpoint runs on; nothing is sent to a third party.

`llmoji upload --target hf` doesn't use your HuggingFace account at all. The push uses a shared submission credential, so nothing about you appears on the dataset's commit history or branch list beyond the salted submitter id.

The submission credential ships encrypted in the package and is unlocked by an upload password, read from `$LLMOJI_UPLOAD_PASSWORD` or prompted for; the current password is posted on the [dataset card](https://huggingface.co/datasets/a9lim/llmoji). Since the password is public, the credential is effectively a public write token for the dataset, scoped to it alone. The encryption only keeps the token out of plain sight in the package source. What protects the dataset is review: every submission lands on its own branch, and nothing reaches `main` until the maintainer has read the diff and merged it.

`llmoji upload --target archive` writes the same allowlisted files to a tarball and prints its path, for anyone who would rather email it.

## Receiving end

The HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) is public. Each submission lands as its own branch on the dataset; the maintainer reviews and merges to `main` by hand. Once merged, your bundle appears under `contributors/<your-submitter-id>/bundle-<ts>/` and becomes publicly downloadable. Please review every `~/.llmoji/bundle/<source-model>.jsonl` before uploading.

If you upload a bundle and later want it removed from the dataset, please email mx@a9l.im with your submitter id and I'll take down the matching folders.
