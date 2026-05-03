"""OpenHands event-log source adapter.

OpenHands (https://github.com/All-Hands-AI/OpenHands) persists each
conversation as a directory of per-event JSON files at
``<conversation_dir>/events/event-NNNNN-<event_id>.json`` (verified
against the SDK's ``persistence_const.py``: ``EVENTS_DIR = "events"``,
``EVENT_FILE_PATTERN = "event-{idx:05d}-{event_id}.json"``). The
``NNNNN`` prefix is the integer event index — sort by it to recover
chronological order without parsing every file's ``timestamp``
field.

Events are typed Pydantic models with a ``kind`` discriminator
field carrying the class name (``MessageEvent``, ``ActionEvent``,
``ObservationEvent``, ``SystemPromptEvent``, ``AgentErrorEvent``, …).
We only care about ``MessageEvent`` rows where ``source == "agent"``
— those are the agent's user-visible text replies, the ones that
can carry a kaomoji prefix. Action / observation / system events
are skipped.

MessageEvent shape (verified against
``openhands-sdk/openhands/sdk/event/llm_convertible/message.py``)::

    {
      "kind": "MessageEvent",
      "id": "<uuid>",
      "timestamp": "2026-...",
      "source": "agent" | "user" | "environment",
      "llm_message": {
        "role": "assistant" | "user" | "system" | "tool",
        "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", ...}
        ],
        "reasoning_content": ...,
        ...
      },
      "llm_response_id": "...",
      "activated_skills": [...],
      "extended_content": [...],
      "sender": null
    }

Per-row notes (vs claude.ai / chatgpt / gemini export):

  - **No model attribution per event.** The model id is set at
    conversation construction time (LLM config in ``base_state.json``)
    rather than on each event, and the on-disk event files don't
    carry it. We stamp ``model=None`` so the bundle layer buckets
    OpenHands rows under the ``unknown`` slug. A follow-up could
    read ``base_state.json`` to recover the model id; not done here
    because the conversation state schema has churned more than the
    event schema and a stable parser wants more verification first.
  - **No cwd.** Events don't carry the agent process's working
    directory; same degradation as the hermes backfill.
  - **Multi-source attribution.** OpenHands routes through LiteLLM
    so a single conversation can mix Anthropic / OpenAI / local
    models. The model field stays empty here, but the diverse
    backends still flow into the corpus through the journal — the
    research-side analysis can correlate via ``base_state.json``
    after the fact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow
from ._common import kaomoji_lead_strip

# Matches the SDK's ``EVENT_NAME_RE`` from
# ``openhands-sdk/openhands/sdk/conversation/persistence_const.py``.
# Capturing the index gives us a chronological sort key without
# parsing every event's timestamp field.
_EVENT_NAME_RE = re.compile(
    r"^event-(?P<idx>\d{5})-(?P<event_id>[0-9a-fA-F\-]{8,})\.json$"
)


def _extract_text(content: Any) -> str:
    """Pull the joined text from an LLM ``Message.content`` array.

    The SDK's content list mixes ``TextContent`` and ``ImageContent``;
    we keep only the text-typed parts. Each part's ``"type"`` field
    discriminates (``"text"`` for the text variant). Non-text parts
    (images, tool-result references) are dropped — they don't carry
    the leading kaomoji we're filtering on.
    """
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") != "text":
            continue
        t = p.get("text")
        if isinstance(t, str) and t.strip():
            parts.append(t)
    return "\n".join(parts)


def _iter_conversation(events_dir: Path) -> Iterator[ScrapeRow]:
    """Walk one conversation's ``events/`` directory in chronological
    (file-index) order, emit one :class:`ScrapeRow` per kaomoji-led
    agent ``MessageEvent``.

    Tracks the most recent user-source MessageEvent's text for
    pairing with each agent row's ``user_text``. SystemPromptEvent /
    ActionEvent / ObservationEvent are skipped — they don't carry
    user-visible model prose.
    """
    files: list[tuple[int, Path]] = []
    for path in events_dir.iterdir():
        if not path.is_file():
            continue
        m = _EVENT_NAME_RE.match(path.name)
        if not m:
            continue
        files.append((int(m.group("idx")), path))
    if not files:
        return
    files.sort(key=lambda t: t[0])

    last_user_text = ""
    for _, path in files:
        try:
            with path.open() as f:
                event = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "MessageEvent":
            continue
        llm_message = event.get("llm_message")
        if not isinstance(llm_message, dict):
            continue
        text = _extract_text(llm_message.get("content"))
        if not text.strip():
            continue
        source = event.get("source")
        if source == "user":
            last_user_text = text
            continue
        if source != "agent":
            # ``environment`` source is observation playback;
            # skip silently.
            continue
        # Belt-and-braces: the SDK's MessageEvent docstring says
        # agent-source events use ``role: "assistant"`` on the wrapped
        # llm_message. Reject if we see anything else — better to
        # drop a row than misattribute it.
        if llm_message.get("role") != "assistant":
            continue
        stripped = kaomoji_lead_strip(text)
        if stripped is None:
            continue
        first_word, body = stripped
        ts = event.get("timestamp")
        yield ScrapeRow(
            # The conversation directory name is typically a UUID;
            # not user-meaningful in attribution but useful for
            # traceback if a row looks suspect.
            source="openhands-export",
            # Per-event model attribution lives in base_state.json,
            # not on the MessageEvent. Leave empty; bundle layer
            # buckets under ``unknown``.
            model=None,
            timestamp=str(ts) if isinstance(ts, str) else "",
            cwd=None,
            assistant_text=body,
            first_word=first_word,
            surrounding_user=last_user_text,
        )


def _find_event_dirs(root: Path) -> Iterator[Path]:
    """Yield every ``events/`` directory under ``root`` (recursively).

    OpenHands conversations land under whatever working directory
    the user configured — there's no fixed home dir. The user
    points :func:`iter_openhands_export` at the parent dir holding
    one or more conversation subdirectories; we walk for any
    descendant ``events/`` that contains
    ``event-NNNNN-<id>.json`` files.

    Visited in sorted path order so re-runs are deterministic.
    """
    if not root.exists():
        return
    if root.is_dir() and root.name == "events":
        # User pointed directly at one conversation's events/ dir.
        yield root
        return
    for path in sorted(root.rglob("events")):
        if path.is_dir():
            yield path


def iter_openhands_export(
    export_dirs: Iterable[Path | str],
) -> Iterator[ScrapeRow]:
    """Yield kaomoji-bearing agent messages from one or more
    OpenHands conversation roots.

    Each input directory is walked recursively for ``events/``
    sub-directories containing ``event-NNNNN-<event_id>.json``
    files. The user's typical pattern is to point at the parent dir
    holding all their conversations — e.g. ``~/.openhands/conversations``
    if running the local agent server, or wherever the user
    configured the FileStore. No fixed-home assumption.
    """
    for export_dir in export_dirs:
        for events_dir in _find_event_dirs(Path(export_dir)):
            yield from _iter_conversation(events_dir)
