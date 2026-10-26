"""Writing rows into journals: the live hook's append and ``llmoji import``.

A message's identity is its timestamp plus its text with all whitespace
removed, face and body together. That survives a change in where the
splitter divides the two, so rows written by older releases still match
their replays.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ._util import atomic_write_text, can_read_zstd, journal_line
from .scrape import ScrapeRow

_WS = re.compile(r"\s+")

# Hook fire time as hooks before 3.0 stamped every row, and as Claude
# Code's final-message fallback still does: whole seconds, no fraction.
_FIRE_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
# A fire time follows its message by up to a long agentic turn, and may
# precede it by the fraction a whole-second stamp drops.
_FIRE_WINDOW = timedelta(hours=24)
_FIRE_SLACK = timedelta(seconds=5)

# How far back the live hook looks for a row it already wrote.
TAIL_LINES = 256


def message_text(kaomoji: str, body: str) -> str:
    return _WS.sub("", kaomoji + body)


def row_key(row: ScrapeRow) -> tuple[str, str]:
    return row.timestamp, message_text(row.first_word, row.assistant_text)


def _parse_ts(ts: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


class _Pool:
    """Journal rows a new row may duplicate; each absorbs at most one,
    so a message genuinely repeated still lands once per copy.

    Besides exact matches, a row folds into one with the same text when
    one of the two carries a fire time that could belong to the other's
    message time. That fold can undercount by one when a fire-time row's
    own message is gone from disk and an identical message sits up to a
    day before it.
    """

    def __init__(self, rows: Iterable[tuple[int, dict[str, Any]]]) -> None:
        self._exact: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._by_text: dict[str, list[tuple[datetime, bool, int]]] = defaultdict(list)
        self._claimed: set[int] = set()
        for i, row in rows:
            ts = str(row.get("ts") or "")
            text = message_text(str(row.get("kaomoji") or ""), str(row.get("assistant_text") or ""))
            self._exact[(ts, text)].append(i)
            t = _parse_ts(ts)
            if t is not None:
                self._by_text[text].append((t, bool(_FIRE_TS.match(ts)), i))
        for candidates in self._by_text.values():
            candidates.sort()

    def claim_exact(self, key: tuple[str, str]) -> int | None:
        """The index of an unclaimed row with exactly this key, claimed."""
        for i in self._exact.get(key, ()):
            if i not in self._claimed:
                self._claimed.add(i)
                return i
        return None

    def claim_fold(self, key: tuple[str, str]) -> int | None:
        """The index of an unclaimed row this one folds into, claimed."""
        ts, text = key
        t = _parse_ts(ts)
        if t is None:
            return None
        fire = bool(_FIRE_TS.match(ts))
        for pt, pfire, i in self._by_text.get(text, ()):
            if i in self._claimed or pfire == fire:
                continue
            msg_t, fire_t = (pt, t) if fire else (t, pt)
            if msg_t - _FIRE_SLACK <= fire_t <= msg_t + _FIRE_WINDOW:
                self._claimed.add(i)
                return i
        return None

    def claim(self, key: tuple[str, str]) -> int | None:
        i = self.claim_exact(key)
        return i if i is not None else self.claim_fold(key)


def _parsed(lines: Iterable[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(line index, row)`` for each line holding a JSON object."""
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield i, row


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(f.tell() - 2_000_000, 0))
        return f.read().decode("utf-8", errors="replace").splitlines()[-n:]


def append_rows(journal: Path, rows: list[ScrapeRow]) -> int:
    """Append the rows not already among the journal's last lines.

    A Stop hook fires again when another hook blocks the stop and the
    agent continues, and the re-fire walks the turn's earlier messages
    again. The look-back only misses a re-fire that comes after more
    than :data:`TAIL_LINES` rows were written since the first fire.
    """
    if not rows:
        return 0
    pool = _Pool(_parsed(_tail(journal, TAIL_LINES)))
    new = [r for r in rows if pool.claim(row_key(r)) is None]
    if new:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a") as f:
            for r in new:
                f.write(json.dumps(journal_line(r), ensure_ascii=False) + "\n")
    return len(new)


@dataclass
class ImportResult:
    rows_seen: int  # replayed rows at or after ``since``
    rows_novel: int  # of those, the ones appended (or that would be)
    rows_updated: int = 0  # journal rows replaced by their replay
    unreadable: int = 0  # compressed session files with no zstd decoder


def import_provider(
    name: str, *, since: str | None = None, dry_run: bool = False,
) -> ImportResult:
    """Replay a provider's session files into its journal. Re-runnable.

    A replayed message the journal lacks is appended. One it holds
    replaces the journal's copy: the replay is authoritative, so a row a
    hook wrote with fire time, no model, an older split or an injected
    prompt is repaired wherever its transcript still exists.

    ``since`` (ISO-8601) gates what's counted, appended and replaced,
    not the matching: every replayed row claims its own journal row
    first, so an older message can't leave its row for a newer one to
    fold into.

    Stop the harness first: a live append racing the rewrite is lost.
    """
    from .providers import get_provider

    provider = get_provider(name)
    if not provider.importable:
        raise ValueError(f"{name} keeps no replayable session history")
    files = provider.session_files()
    replayed = [row for path in files for row in provider.replay(path)]
    journal = provider.journal_path
    lines = journal.read_text().splitlines() if journal.exists() else []
    pool = _Pool(_parsed(lines))

    def in_window(r: ScrapeRow) -> bool:
        return since is None or r.timestamp >= since

    # Exact matches first, over every replayed row, so a fold can't take
    # a journal row that a later replayed row matches exactly. A key
    # already taken (a forked session replays its parent's history) is a
    # duplicate and never reaches the fold.
    matched: dict[int, ScrapeRow] = {}
    taken: set[tuple[str, str]] = set()
    unmatched: list[ScrapeRow] = []
    for r in replayed:
        key = row_key(r)
        i = pool.claim_exact(key)
        if i is not None:
            matched[i] = r
        elif key not in taken:
            unmatched.append(r)
        taken.add(key)
    novel: list[ScrapeRow] = []
    for r in sorted(unmatched, key=lambda r: r.timestamp):
        i = pool.claim_fold(row_key(r))
        if i is not None:
            matched[i] = r
        elif in_window(r):
            novel.append(r)

    def dump(r: ScrapeRow) -> str:
        return json.dumps(journal_line(r), ensure_ascii=False)

    # Compared as rows, not text: jq wrote compact JSON, and formatting
    # alone isn't worth a rewrite.
    updates = {
        i: dump(r) for i, r in matched.items()
        if in_window(r) and journal_line(r) != json.loads(lines[i])
    }
    if (novel or updates) and not dry_run:
        out = [updates.get(i, line) for i, line in enumerate(lines)]
        atomic_write_text(journal, "".join(line + "\n" for line in [*out, *map(dump, novel)]))
    return ImportResult(
        rows_seen=sum(map(in_window, replayed)),
        rows_novel=len(novel),
        rows_updated=len(updates),
        unreadable=0 if can_read_zstd() else sum(p.suffix == ".zst" for p in files),
    )
