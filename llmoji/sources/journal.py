"""Generic kaomoji-journal reader.

Every provider's ``install`` writes a bash hook that appends one row
per kaomoji-bearing assistant turn to a journal at
``~/.<harness>/kaomoji-journal.jsonl``. The schema is uniform across
providers (this is the one piece of public API every harness's hook
must produce, locked in v1.0):

    {
      "ts":             ISO-8601 UTC timestamp,
      "model":          active model slug,
      "cwd":            working directory at hook fire,
      "kaomoji":        leading non-letter prefix (≥2 bytes, validated),
      "user_text":      latest real user message (system-injection filtered),
      "assistant_text": last assistant text MINUS the leading kaomoji
    }

The ``kaomoji`` field is already the leading prefix the shell hook
captured; we still pipe it through :func:`~llmoji.taxonomy.extract` to
recover a clean balanced-paren ``first_word`` (and a taxonomy match
when one happens to exist) and to defensively reject any legacy null
or shape-incorrect rows that slipped past the hook's own filters.

This is the preferred path for every harness whose hook is
provider-managed. For motivated users on harnesses we don't ship a
first-class adapter for (e.g. OpenClaw — TS-shaped hooks), the
contract is "write a JSONL line in this exact schema to
``~/.llmoji/journals/<name>.jsonl`` and ``analyze`` will pick it up."
The same iterator handles both cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..scrape import ScrapeRow
from ..taxonomy import KAOMOJI_START_CHARS, extract


def _project_slug_from_cwd(cwd: str | None) -> str:
    if not cwd:
        return "(unknown)"
    name = Path(cwd).name
    return name or "(unknown)"


def iter_journal(
    path: Path | str,
    *,
    source: str,
) -> Iterator[ScrapeRow]:
    """Yield :class:`ScrapeRow` per kaomoji-bearing journal line.

    Parameters
    ----------
    path :
        Filesystem path to the JSONL journal. Missing files yield no
        rows (no error — a never-installed provider is a valid
        steady-state).
    source :
        Stable name for the originating provider. Becomes the
        ``ScrapeRow.source`` value with a ``"-hook"`` suffix
        (``"claude_code"`` → ``"claude_code-hook"``). The CLI uses
        this to generate per-provider breakdowns and to label
        bundles.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        lines = f.read().splitlines()
    turn = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prefix = row.get("kaomoji")
        if not prefix:
            continue
        match = extract(str(prefix))
        if not (match.first_word and match.first_word[0] in KAOMOJI_START_CHARS):
            continue
        cwd = row.get("cwd")
        yield ScrapeRow(
            source=f"{source}-hook",
            session_id="",
            project_slug=_project_slug_from_cwd(cwd),
            assistant_uuid="",
            parent_uuid=None,
            model=str(row.get("model") or "") or None,
            timestamp=str(row.get("ts") or ""),
            cwd=str(cwd) if cwd else None,
            git_branch=None,
            turn_index=turn,
            had_thinking=False,
            assistant_text=str(row.get("assistant_text") or ""),
            first_word=match.first_word,
            surrounding_user=str(row.get("user_text") or ""),
        )
        turn += 1
