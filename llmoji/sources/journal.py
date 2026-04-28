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
from ..taxonomy import extract


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
        Stable name for the originating journal. Used verbatim as
        ``ScrapeRow.source``. Live-hook callers in
        :func:`llmoji.cli._gather_rows` pass the ``-hook`` suffix
        themselves (``"claude_code-hook"``); static-export
        ``~/.llmoji/journals/<name>.jsonl`` callers pass the bare
        stem (``"claude_ai_export"``) so the bundle subfolder names
        don't lie about whether a row came from a live hook.
    """
    p = Path(path)
    if not p.exists():
        return
    turn = 0
    # Stream line-by-line — journals grow monotonically and a heavy
    # user's file can hit 100s of MB. Reading + splitlines materialized
    # the whole file in memory; the iter form scales with row count.
    with p.open() as f:
        for raw in f:
            line = raw.strip()
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
            # ``extract`` runs the candidate filter (start-char set +
            # length + bracket balance) internally; a non-empty
            # ``first_word`` is already validated. Single source of
            # truth lives in ``taxonomy``.
            if not match.first_word:
                continue
            cwd = row.get("cwd")
            yield ScrapeRow(
                source=source,
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
