"""Inspect the bundle that ``llmoji analyze`` produced before running
``llmoji upload``.

Run after ``llmoji analyze``::

    python examples/inspect_bundle.py

Prints the manifest summary plus one line per canonical kaomoji. This
is what would ship on ``llmoji upload`` (and nothing else: the bundle
is allowlisted to ``manifest.json`` and ``descriptions.jsonl``).
"""

from __future__ import annotations

import json
import sys

from llmoji import paths


def main() -> int:
    bundle = paths.bundle_dir()
    manifest_path = bundle / "manifest.json"
    descriptions_path = bundle / "descriptions.jsonl"

    if not manifest_path.exists() or not descriptions_path.exists():
        print(
            f"no bundle at {bundle}; run `llmoji analyze` first.",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(manifest_path.read_text())
    print(f"bundle: {bundle}")
    print(f"  llmoji version:    {manifest.get('llmoji_version', '?')}")
    print(f"  haiku model:       {manifest.get('haiku_model_id', '?')}")
    print(f"  generated at:      {manifest.get('generated_at', '?')}")
    print(f"  submitter id:      {manifest.get('submitter_id', '?')}")
    print(f"  total rows:        {manifest.get('total_rows_scraped', '?')}")
    print(
        f"  canonical unique:  "
        f"{manifest.get('total_kaomoji_unique_canonical', '?')}"
    )
    if providers := manifest.get("providers_seen"):
        print(f"  providers:         {', '.join(providers)}")
    if journal_counts := manifest.get("journal_counts"):
        print("  journal counts:")
        for src, n in sorted(journal_counts.items()):
            print(f"    {src:<24} {n} rows")
    if notes := manifest.get("notes"):
        print(f"  notes:             {notes}")
    print()

    print("descriptions.jsonl (the only prose that ships):")
    print()
    with descriptions_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kao = row.get("kaomoji", "?")
            count = row.get("count", "?")
            desc = row.get("haiku_synthesis_description", "")
            print(f"  {kao}   ({count} occurrences)")
            print(f"      {desc}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
