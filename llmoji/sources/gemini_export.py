"""Google Gemini export source adapter — handles two formats.

Google ships **two** distinct Gemini exports, with different schemas
and different acquisition paths. This reader auto-dispatches per
file at read time so the user can mix both into one directory
without picking a sub-provider.

**1. AI Studio** (https://aistudio.google.com) — one JSON file per
conversation, downloaded individually from the UI. The developer-
facing Gemini surface. Shape::

    {
      "runSettings": {
        "model": "models/gemini-2.5-pro-exp-...",
        "temperature": ..., "topP": ..., "topK": ...,
        ...
      },
      "citations": [...],
      "systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
      "chunkedPrompt": {
        "chunks": [
          {"text": "user prompt", "role": "user"},
          {"text": "(◕ω◕) reply...", "role": "model", "tokenCount": 123},
          {"text": "intermediate thinking", "role": "model", "isThought": true, ...},
          ...
        ]
      }
    }

**2. Google Takeout MyActivity.json** — single file containing every
chat (and other product activities) for the consumer Gemini app.
Acquired from `takeout.google.com` with "Gemini Apps" selected and
JSON format chosen. Top-level is a flat list of entries; each Gemini
chat lands as its own entry::

    [
      {
        "header": "Gemini",                          # product label
        "time": "2026-...Z",                         # ISO-8601 UTC
        "title": "Asked Gemini",                     # action label
        "subtitles": [
          {"name": "User", "value": "user prompt"}   # user message
        ],
        "safeHtmlItem": [
          {"html": "<p>(◕ω◕) reply HTML...</p>"}     # response in HTML
        ],
        "products": ["Gemini Apps"], ...
      },
      ...
    ]

MyActivity entries cover every Google product the user has activity
for; the reader filters on ``"Gemini" in header`` to scope to the
consumer Gemini chat product (mirroring the convention from
upstream parser tools). Order on disk is reverse-chronological
(newest first); we reverse to chronological on emit.

The Takeout export's response field is HTML (``<p>``, ``<br>``, code
blocks, occasionally inline emphasis tags); we strip tags and decode
entities before kaomoji extraction so the prefix isn't masked by a
leading ``<p>`` or escaped ``&#40;``.

Per-row notes (vs claude.ai / chatgpt / AI Studio):

  - **AI Studio: no per-message timestamps.** Stamps every row from
    one file with the file's mtime as ISO-8601 UTC. Within-file
    ordering preserved by ``chunks[]`` array order.
  - **AI Studio: ``isThought: true`` chunks are skipped.** Private
    chain-of-thought, never user-visible.
  - **AI Studio: ``model`` from ``runSettings.model``** with
    ``models/`` prefix stripped (``gemini-2.5-pro-exp`` not
    ``models/gemini-2.5-pro-exp``).
  - **Takeout: per-entry ``time`` field** carries an ISO-8601 UTC
    timestamp so cross-entry ordering is honest. Newer files than
    the user's chat history may use sub-second precision; we keep
    the string verbatim for the journal row.
  - **Takeout: ``model`` is empty.** MyActivity doesn't track which
    specific Gemini variant served the response (Pro / Flash / etc.);
    the ``header`` field is just the product name. Rows bucket
    under ``unknown`` in the bundle layer — same degradation as
    OpenHands. A research-side join against announcement timestamps
    could backfill the model id if it matters; not done here.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..scrape import ScrapeRow
from ._common import kaomoji_lead_strip


def _is_aistudio_shape(data: Any) -> bool:
    """Heuristic: top-level dict with a ``chunkedPrompt`` mapping
    that contains a ``chunks`` list. Filters out unrelated JSON
    files the user might have left in the export directory.
    """
    if not isinstance(data, dict):
        return False
    cp = data.get("chunkedPrompt")
    if not isinstance(cp, dict):
        return False
    return isinstance(cp.get("chunks"), list)


def _is_takeout_my_activity_shape(data: Any) -> bool:
    """Heuristic: top-level list whose first entry looks like a
    Takeout MyActivity record (carries ``header`` + ``time`` keys).

    MyActivity files are large flat lists; sniffing the first dict
    with the right keys is enough to dispatch correctly. Empty lists
    fall through and yield zero rows on the Takeout walker — same
    behavior as a list with no Gemini entries, which is the right
    answer for someone whose Takeout export is for a different
    product.
    """
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    return "header" in first and "time" in first


class _HtmlTextStripper(HTMLParser):
    """Minimal HTML-to-plain-text reducer for ``safeHtmlItem.html``.

    The Takeout export's response field is HTML-encoded (``<p>``,
    ``<br>``, occasionally inline emphasis or code tags) with HTML
    entities for special characters. We need plain text with a
    leading kaomoji intact; ``html.parser.HTMLParser`` plus
    ``html.unescape`` (already applied to ``handle_data`` payloads
    by the parser) lands the body in a state ``kaomoji_lead_strip``
    can chew on without leading-bracket masking.

    Block-level tags (``p``, ``br``, ``div``, ``li``) emit a
    newline so adjacent paragraphs don't run into each other —
    matters for ``kaomoji_prefix``-style "first line only" pickers
    elsewhere in the pipeline. The validator we use here
    (``kaomoji_lead_strip`` via ``taxonomy.extract``) reads the
    leading span only, so newlines past the first don't matter.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # ``attrs`` is unused — we only care about block-level tag
        # boundaries to inject newlines. The parameter name is
        # locked by HTMLParser's base class so it can't be renamed
        # to silence pyright's unused-name warning.
        del attrs
        if tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def get_text(self) -> str:
        return "".join(self._parts)


_WHITESPACE_RUN = re.compile(r"[ \t]+")


def _html_to_text(snippet: str) -> str:
    """Strip HTML tags + decode entities; collapse runs of horizontal
    whitespace so a kaomoji like ``(◕‿◕)`` doesn't get mangled by an
    inline ``<span>...&nbsp;...</span>`` that decodes to a no-break
    space at the join boundary.

    Runs the snippet through both the HTMLParser-based stripper and
    a final ``html.unescape`` pass (entities outside tag content,
    e.g. inside attributes or between unwrapped text, do reach
    ``handle_data`` thanks to ``convert_charrefs=True`` — the second
    pass is belt-and-braces against malformed input).
    """
    parser = _HtmlTextStripper()
    try:
        parser.feed(snippet)
        parser.close()
    except Exception:
        # Malformed HTML — fall back to unescape on the raw string,
        # which still recovers the kaomoji 99% of the time.
        return html.unescape(snippet).strip()
    text = html.unescape(parser.get_text())
    # Collapse runs of horizontal whitespace; preserve newlines so
    # paragraph structure survives downstream.
    text = _WHITESPACE_RUN.sub(" ", text)
    return text.strip()


def _model_id(data: dict[str, Any]) -> str | None:
    """Pull the model id from ``runSettings.model``, stripping the
    ``models/`` prefix AI Studio adds for API-call compatibility.

    Returns ``None`` when the field is missing — the bundle layer
    sanitizes empty model strings to a fixed slug, so a missing id
    doesn't crash.
    """
    settings = data.get("runSettings")
    if not isinstance(settings, dict):
        return None
    model = settings.get("model")
    if not isinstance(model, str) or not model:
        return None
    if model.startswith("models/"):
        return model[len("models/"):]
    return model


def _file_timestamp(path: Path) -> str:
    """ISO-8601 UTC timestamp for the file. AI Studio exports don't
    carry per-chunk timestamps and the conversation envelope's
    ``createTime`` field isn't reliably populated, so we fall back
    to the file's mtime — accurate within "when the user clicked
    save", which is good enough for cross-corpus ordering.
    """
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _iter_conversation(
    path: Path,
    data: dict[str, Any],
) -> Iterator[ScrapeRow]:
    """Walk one AI Studio conversation file's ``chunks`` array,
    emit one :class:`ScrapeRow` per kaomoji-led non-thought model
    chunk. Tracks the most recent user chunk's text in a single
    forward pass so every assistant row carries the originating
    user prompt for its turn.
    """
    chunks = data["chunkedPrompt"]["chunks"]
    if not isinstance(chunks, list):
        return

    model = _model_id(data)
    ts = _file_timestamp(path)
    last_user_text = ""

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        role = chunk.get("role")
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if role == "user":
            last_user_text = text
            continue
        if role != "model":
            continue
        # Skip private chain-of-thought; only user-visible model
        # output can carry the kaomoji prefix.
        if chunk.get("isThought") is True:
            continue
        stripped = kaomoji_lead_strip(text)
        if stripped is None:
            continue
        first_word, body = stripped
        yield ScrapeRow(
            source="gemini-aistudio-export",
            model=model,
            timestamp=ts,
            cwd=None,
            assistant_text=body,
            first_word=first_word,
            surrounding_user=last_user_text,
        )


def _iter_takeout_entries(
    entries: list[Any],
) -> Iterator[ScrapeRow]:
    """Walk a Google Takeout MyActivity.json list, emit one
    :class:`ScrapeRow` per kaomoji-led Gemini chat response.

    The on-disk order is reverse-chronological (Takeout writes
    newest first); we reverse to chronological so within-file row
    order matches the live-hook journals' append-order convention.
    Each entry maps to one row when:

      - ``"Gemini" in entry["header"]`` (filter out non-Gemini
        product activity that may live in the same file)
      - ``safeHtmlItem`` HTML strips to a kaomoji-bearing leading
        span
      - ``subtitles`` carries a non-empty ``value`` (the user prompt)

    User text resolves per entry rather than per turn — MyActivity
    treats each chat round as its own entry, so the prompt-and-
    response pairing is row-local.
    """
    # Reverse to chronological order. Slice rather than mutate the
    # caller's list to keep the iterator pure.
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        header = entry.get("header")
        if not isinstance(header, str) or "Gemini" not in header:
            continue
        # Response: join every safeHtmlItem.html, strip to text.
        items = entry.get("safeHtmlItem")
        if not isinstance(items, list) or not items:
            continue
        html_parts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            h = item.get("html")
            if isinstance(h, str) and h:
                html_parts.append(h)
        if not html_parts:
            continue
        body_text = _html_to_text("\n".join(html_parts))
        if not body_text:
            continue
        stripped = kaomoji_lead_strip(body_text)
        if stripped is None:
            continue
        first_word, body = stripped

        # User prompt: first non-empty subtitle value. MyActivity's
        # subtitles[].value is plain text (no HTML escaping).
        subtitles = entry.get("subtitles") or []
        user_text = ""
        if isinstance(subtitles, list):
            for sub in subtitles:
                if not isinstance(sub, dict):
                    continue
                val = sub.get("value")
                if isinstance(val, str) and val.strip():
                    user_text = val
                    break

        ts = entry.get("time")
        yield ScrapeRow(
            source="gemini-takeout-export",
            # MyActivity doesn't carry a per-entry model id — the
            # consumer Gemini app routes between Pro / Flash / etc.
            # transparently. Bundle layer slugs to ``unknown``.
            model=None,
            timestamp=str(ts) if isinstance(ts, str) else "",
            cwd=None,
            assistant_text=body,
            first_word=first_word,
            surrounding_user=user_text,
        )


def iter_gemini_export(
    export_dirs: Iterable[Path | str],
) -> Iterator[ScrapeRow]:
    """Yield kaomoji-bearing model messages from one or more Gemini
    export directories — handles both AI Studio (one JSON per
    conversation, ``chunkedPrompt`` shape) and Google Takeout (one
    flat ``MyActivity.json`` listing every Gemini chat) automatically.

    Each directory is walked recursively for ``*.json`` files;
    each file is sniffed at read time and dispatched to the right
    walker. Files matching neither shape are skipped silently
    (so the user can mix exports with unrelated JSON without hand-
    curating the directory). Files are visited in sorted path
    order so re-runs are deterministic.

    Within a Takeout file the entries are processed
    reverse-on-disk → chronological-on-emit; AI Studio files
    process in array order (already chronological).
    """
    for export_dir in export_dirs:
        root = Path(export_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                with path.open() as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if _is_aistudio_shape(data):
                yield from _iter_conversation(path, data)
            elif _is_takeout_my_activity_shape(data):
                yield from _iter_takeout_entries(data)
            # else: unrelated JSON, skip silently
