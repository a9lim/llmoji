"""Reading journals: the hooks' own and any in ``~/.llmoji/journals/``.

Every journal row has six string fields::

    {"ts", "model", "cwd", "kaomoji", "user_text", "assistant_text"}

``kaomoji`` is the face that opened the message and ``assistant_text``
the rest of it. A harness llmoji has no provider for can append rows in
this schema to ``~/.llmoji/journals/<name>.jsonl`` and ``analyze`` reads
them like any other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..scrape import ScrapeRow, is_injected
from ..taxonomy import extract, split_leading_kaomoji

# Model ids that name no model: Claude Code stamps ``<synthetic>`` on
# entries it writes itself, and on some resumed sessions' messages.
_NOT_A_MODEL = frozenset({"<synthetic>"})


def iter_journal(path: Path | str, *, source: str) -> Iterator[ScrapeRow]:
    """One :class:`ScrapeRow` per row with a valid face.

    The stored face is re-split, so rows from a writer that let markdown
    or prose into the ``kaomoji`` field (llmoji before 3.0 did, for
    ``(•ᴗ•) **Done**``) get it back in the body. ``first_word`` is the
    face's bucketing span (:func:`~llmoji.taxonomy.extract`). A
    ``user_text`` that is harness-injected text (a task notification, a
    reminder; earlier hooks stored those) reads as empty.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            split = split_leading_kaomoji(str(row.get("kaomoji") or ""))
            if split is None:
                continue
            face, tail = split
            first_word = extract(face).first_word
            if not first_word:
                continue
            body = str(row.get("assistant_text") or "")
            model = str(row.get("model") or "")
            user = str(row.get("user_text") or "")
            yield ScrapeRow(
                source=source,
                model=None if model in _NOT_A_MODEL else model or None,
                timestamp=str(row.get("ts") or ""),
                cwd=str(row.get("cwd") or "") or None,
                assistant_text=f"{tail} {body}".strip() if tail else body,
                first_word=first_word,
                surrounding_user="" if is_injected(user) else user,
            )
