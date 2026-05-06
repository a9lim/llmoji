# Security and privacy policy

## Reporting a vulnerability

If you've found a security issue in llmoji, please report it privately rather than filing a public issue.

- **Email:** mx@a9l.im
- **GitHub:** use [private security advisories](https://github.com/a9lim/llmoji/security/advisories/new)

Please include a description, steps to reproduce, and the version you are on. I'll respond within a few days and aim to have a fix as soon as possible.

## Supported versions

Only the latest minor version on PyPI receives security and privacy fixes. If you're on an older version, please upgrade.

## Privacy notice for upgraders (1.2.0)

Versions of llmoji before 1.2.0 would have leaked your HuggingFace username on submission. 1.2.0 patches this. Please upgrade (`pip install --upgrade llmoji`).

## Privacy model

llmoji is a privacy-sensitive tool. The package ships aggregates from your machine to a shared corpus so privacy is important here.

### What stays on your machine

- **Raw `user_text` or `assistant_text`** at `~/.<harness>/kaomoji-journal.jsonl`. These hold the raw data for every kaomoji-bearing turn. They never leave your machine.
- **Per-cell synthesizer cache** at `~/.llmoji/cache/per_cell.jsonl`. Each row holds a structured adjective bag (`primary_affect` + `stance_modality_function`) drawn from the locked LEXICON, keyed by the synthesis model id, backend, base URL, source model, canonical kaomoji, and a hash of the sampled (user, assistant) pairs that fed the call. The bag itself contains no free-form text and no journal content — only adjectives from the corpus vocabulary — so the cache is materially less leak-prone than v1's per-turn paraphrase cache. The cache is never bundled and never shipped. `llmoji status` prints its size; `llmoji cache clear` is the explicit wipe (and removes the orphaned legacy v1 `per_instance.jsonl` if still on disk after upgrading from 1.x).
- **Submission token** at `~/.llmoji/.salt`. A 256-bit random token generated on first `upload`, used as the salt for the submitter id. Never sent anywhere.

### What ships when you `upload`

The bundle is human-readable JSON, laid out flat: one top-level manifest plus one `.jsonl` per source model, no subdirectories.

- **`manifest.json`**: package version, lexicon version, the synthesis backend and model id used, per-source-model row counts, total synthesized rows, list of providers seen, generation timestamp, any included `--notes`, and a salted-hash submitter id.
- **`<source-model>.jsonl`**: one row per canonical kaomoji as that model used it, shaped `{kaomoji, count, synthesis: {primary_affect: [...], stance_modality_function: [...]}}`. Both arrays draw exclusively from the locked LEXICON enum, so no free-form text and no journal content lands in a bundle row.

The submitter id is a 32-hex-char (128-bit) salted hash of the per-machine token plus the package version. We do not collect HuggingFace usernames or any account-bound identifier.

### Singleton kaomoji caveat

In v1.x the per-face summary was free-form prose, and for a kaomoji that appeared once, the synthesized line was effectively a paraphrase of that one turn — a small singleton-leak surface. v2 removed this surface entirely: the row carries only adjectives from a 48-word locked vocabulary, so a singleton kaomoji's row is a bag like `{primary_affect: ["frustrated"], stance_modality_function: ["wry", "self-correcting", "tender"]}` rather than a paraphrase of the underlying turn. Even so, please continue to review the bundle before uploading:

- `analyze` prints a per-face preview before declaring done.
- `upload` re-prompts before committing.
- The bundle is inspectable in `~/.llmoji/bundle/`.
- The bundle is allowlisted: top-level `manifest.json` plus per-model `.jsonl` files at the root. Both `upload --target hf` and `tar_bundle` (used for email) refuse to ship if anything else is in the bundle dir.

Please review every `~/.llmoji/bundle/<source-model>.jsonl` before running `upload`.

### Hooks are read-only

The bash hooks shipped with each provider append one row to a journal. They never block the turn, modify the reply, or call out to the network. 

## Model and API trust

`llmoji analyze` sends the masked-kaomoji prompts and the user and assistant text from your journal to whichever synthesis backend you pick. Three backends are supported and each one routes your data differently. Please review the relevant data-handling policy before running `analyze` against a corpus you care about.

- **`--backend anthropic` (default)**: calls the Anthropic API with your `$ANTHROPIC_API_KEY`. Your journal text goes to Anthropic for paraphrasing.
- **`--backend openai`**: calls the OpenAI Responses API with your `$OPENAI_API_KEY`. Your journal text goes to OpenAI for paraphrasing.
- **`--backend local`**: calls a local OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp's HTTP server, etc.) at the `--base-url` you pass. Your journal text stays on whatever machine the endpoint runs on; nothing is sent to a third party.

`llmoji upload --target hf` reads your HuggingFace token from `$HF_TOKEN` or `~/.cache/huggingface/token` once for an `HfApi.whoami()` proof-of-life check, then discards it. Your token is never used besides this. The push uses a shared submission credential, so your personal HF account never appears on the dataset's commit history or branch list.

The submission credential is gated behind an upload password. Llmoji reads the password from `$LLMOJI_UPLOAD_PASSWORD` or prompts you interactively; the current password is posted on the [dataset card](https://huggingface.co/datasets/a9lim/llmoji).

`llmoji upload --target email` builds a `mailto:` URI with the bundle path printed in the body and asks you to attach the tarball manually.

## Receiving end

The HuggingFace dataset at [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji) is public. Each submission lands as its own branch on the dataset; the maintainer reviews and merges to `main` by hand. Once merged, your bundle appears under `contributors/<your-submitter-id>/bundle-<ts>/` and becomes publicly downloadable. Please review every `~/.llmoji/bundle/<source-model>.jsonl` before uploading.

If you upload a bundle and later want it removed from the dataset, please email mx@a9l.im with your submitter id and I'll take down the matching folders.
