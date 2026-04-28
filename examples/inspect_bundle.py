"""Inspect the bundle that ``llmoji analyze`` produced before running
``llmoji upload``.

Run after ``llmoji analyze``::

    python examples/inspect_bundle.py

Prints the manifest summary plus one section per source-model
subfolder, with each canonical kaomoji's row inside. This is what
would ship on ``llmoji upload`` (and nothing else: the bundle is
allowlisted to ``manifest.json`` at the top level plus each
``<source-model>/descriptions.jsonl``).
"""

from __future__ import annotations

import json
import sys

from llmoji import paths


def main() -> int:
    bundle = paths.bundle_dir()
    manifest_path = bundle / "manifest.json"

    if not manifest_path.exists():
        print(
            f"no bundle at {bundle}; run `llmoji analyze` first.",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(manifest_path.read_text())
    print(f"bundle: {bundle}")
    print(f"  llmoji version:    {manifest.get('llmoji_version', '?')}")
    print(f"  synthesis backend: {manifest.get('synthesis_backend', '?')}")
    print(f"  synthesis model:   {manifest.get('synthesis_model_id', '?')}")
    print(f"  generated at:      {manifest.get('generated_at', '?')}")
    print(f"  submitter id:      {manifest.get('submitter_id', '?')}")
    print(
        f"  total synth rows:  "
        f"{manifest.get('total_synthesized_rows', '?')}"
    )
    if providers := manifest.get("providers_seen"):
        print(f"  providers:         {', '.join(providers)}")
    if model_counts := manifest.get("model_counts"):
        print("  model counts:")
        for src_model, n in sorted(model_counts.items()):
            print(f"    {src_model:<40} {n} rows")
    if notes := manifest.get("notes"):
        print(f"  notes:             {notes}")
    print()

    subdirs = sorted(p for p in bundle.iterdir() if p.is_dir())
    if not subdirs:
        print("no source-model subfolders — bundle is empty.")
        return 2

    print("descriptions (the only prose that ships):")
    print()
    for sub in subdirs:
        descriptions_path = sub / "descriptions.jsonl"
        if not descriptions_path.exists():
            print(f"  [{sub.name}/]  (missing descriptions.jsonl)")
            continue
        rows = [
            json.loads(line)
            for line in descriptions_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"  ── {sub.name}  ({len(rows)} faces) ──")
        for row in rows:
            kao = row.get("kaomoji", "?")
            count = row.get("count", "?")
            desc = row.get("synthesis_description", "")
            print(f"    {kao}   ({count} occurrences)")
            print(f"        {desc}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
