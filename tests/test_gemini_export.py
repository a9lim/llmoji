"""Tests for :mod:`llmoji.sources.gemini_export`.

Builds synthetic AI Studio export JSON files matching the
``chunkedPrompt.chunks[]`` shape and asserts the reader yields the
expected :class:`~llmoji.scrape.ScrapeRow` stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmoji.sources.gemini_export import iter_gemini_export


def _write_export(path: Path, *, model: str, chunks: list[dict[str, Any]]) -> None:
    """Write an AI Studio-shaped JSON file with the given chunks."""
    path.write_text(
        json.dumps(
            {
                "runSettings": {"model": model, "temperature": 0.7},
                "citations": [],
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": "you are helpful"}],
                },
                "chunkedPrompt": {"chunks": chunks},
            }
        )
    )


def test_emits_one_row_per_kaomoji_led_model_chunk(tmp_path: Path) -> None:
    """Walk a single conversation; one model chunk leads with a
    kaomoji, one doesn't, one is a thought (skip), one is plain
    prose (skip). Result: exactly one row, paired with the most
    recent user chunk's text."""
    _write_export(
        tmp_path / "conv.json",
        model="models/gemini-2.5-pro-exp-2026-02",
        chunks=[
            {"role": "user", "text": "what's two plus two?"},
            {"role": "model", "text": "thinking...", "isThought": True},
            {"role": "model", "text": "(◕ω◕) it's four!"},
            {"role": "user", "text": "ok thanks"},
            {"role": "model", "text": "no kaomoji here"},  # filtered
        ],
    )

    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    r = rows[0]
    assert r.source == "gemini-aistudio-export"
    # ``models/`` prefix stripped.
    assert r.model == "gemini-2.5-pro-exp-2026-02"
    assert r.first_word == "(◕ω◕)"
    assert r.assistant_text == "it's four!"
    assert r.surrounding_user == "what's two plus two?"
    # File mtime got stamped as ISO-8601 UTC; format is
    # ``YYYY-MM-DDTHH:MM:SS...Z`` so just spot-check the prefix shape.
    assert r.timestamp.endswith("Z") or len(r.timestamp) >= 19


def test_skips_non_aistudio_json(tmp_path: Path) -> None:
    """The reader walks every ``.json`` in the directory but
    skips files that don't carry the AI Studio shape — so a user
    can dump an export plus unrelated JSON into the same dir."""
    _write_export(
        tmp_path / "ok.json",
        model="gemini-2.5-flash",
        chunks=[
            {"role": "user", "text": "hi"},
            {"role": "model", "text": "(´▽`) hello!"},
        ],
    )
    (tmp_path / "stray.json").write_text(json.dumps({"unrelated": True}))
    (tmp_path / "garbage.json").write_text("{ not valid json")
    (tmp_path / "list.json").write_text(json.dumps([1, 2, 3]))

    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(´▽`)"
    assert rows[0].model == "gemini-2.5-flash"


def test_user_text_resolves_per_turn(tmp_path: Path) -> None:
    """Multi-turn conversation: each model row carries the originating
    user prompt of its turn. Tool-heavy turns aren't really a thing in
    AI Studio (no tool calls), but the same forward-pass user-tracking
    pattern covers regular alternation cleanly."""
    _write_export(
        tmp_path / "multi.json",
        model="gemini-2.5-pro",
        chunks=[
            {"role": "user", "text": "first question"},
            {"role": "model", "text": "(◕‿◕) first answer"},
            {"role": "user", "text": "second question"},
            {"role": "model", "text": "(╥﹏╥) second answer"},
        ],
    )

    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 2
    assert rows[0].surrounding_user == "first question"
    assert rows[1].surrounding_user == "second question"


def test_missing_directory_yields_nothing(tmp_path: Path) -> None:
    """Pointing at a nonexistent directory returns cleanly — same
    contract as the other export readers."""
    rows = list(iter_gemini_export([tmp_path / "does-not-exist"]))
    assert rows == []


def test_recursive_walk(tmp_path: Path) -> None:
    """The reader walks nested directories — users often unzip an
    archive and end up with conversations in subfolders."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    _write_export(
        nested / "deep.json",
        model="gemini-1.5-pro",
        chunks=[
            {"role": "user", "text": "?"},
            {"role": "model", "text": "(°ロ°) !"},
        ],
    )
    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(°ロ°)"


# ----- consumer Gemini Takeout MyActivity.json -----


def _takeout_entry(
    *,
    header: str = "Gemini",
    time: str = "2026-04-15T10:30:00.000Z",
    title: str = "Asked Gemini",
    user_prompt: str | None = "what is the weather",
    response_html: str | None = "<p>(◕ω◕) it's sunny.</p>",
) -> dict[str, Any]:
    """Build one MyActivity.json entry mirroring the real Takeout shape."""
    entry: dict[str, Any] = {
        "header": header,
        "time": time,
        "title": title,
    }
    if user_prompt is not None:
        entry["subtitles"] = [{"name": "User", "value": user_prompt}]
    if response_html is not None:
        entry["safeHtmlItem"] = [{"html": response_html}]
    return entry


def test_takeout_emits_one_row_per_kaomoji_led_response(tmp_path: Path) -> None:
    """A flat MyActivity.json with two Gemini entries (one
    kaomoji-led, one not) plus a non-Gemini product entry. Only the
    kaomoji-led Gemini entry yields a row, with HTML stripped from
    the body and the user prompt paired."""
    activity = [
        # Reverse-chronological on disk: newest first.
        _takeout_entry(
            time="2026-04-16T09:00:00Z",
            response_html="<p>(◕ω◕) hello there!</p>",
            user_prompt="hi gemini",
        ),
        _takeout_entry(
            time="2026-04-15T10:30:00Z",
            response_html="<p>plain prose, no kaomoji here.</p>",
            user_prompt="any kaomoji?",
        ),
        # Non-Gemini entry (Search, Maps, etc.) — filtered out.
        {
            "header": "Search",
            "time": "2026-04-14T08:00:00Z",
            "title": "Searched for `kaomoji`",
        },
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))

    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    r = rows[0]
    assert r.source == "gemini-takeout-export"
    assert r.first_word == "(◕ω◕)"
    assert r.assistant_text == "hello there!"
    assert r.surrounding_user == "hi gemini"
    # Per-entry timestamp preserved verbatim.
    assert r.timestamp == "2026-04-16T09:00:00Z"
    # No model attribution available from MyActivity.
    assert r.model is None


def test_takeout_decodes_html_entities_and_strips_tags(tmp_path: Path) -> None:
    """Real Gemini responses arrive with HTML entities for special
    characters; ``(`` and ``)`` are sometimes encoded as ``&#40;`` /
    ``&#41;`` depending on serializer. The reader must decode + strip
    so the kaomoji is recoverable.
    """
    activity = [
        _takeout_entry(
            response_html=(
                "<div>&#40;&#9685;&#969;&#9685;&#41;&nbsp;encoded paren reply"
                "<br/>second line</div>"
            ),
        ),
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))
    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◕ω◕)"
    # Body has the &nbsp; (decoded to U+00A0) plus ``encoded paren reply``;
    # we don't pin the exact whitespace shape here, just that the
    # leading kaomoji and the rest of the prose round-trip.
    assert "encoded paren reply" in rows[0].assistant_text


def test_takeout_chronological_emit_order(tmp_path: Path) -> None:
    """MyActivity stores entries newest-first on disk. The reader
    emits oldest-first so within-file row order matches the
    live-hook journals' append-order convention.
    """
    activity = [
        _takeout_entry(
            time="2026-04-16T09:00:00Z",
            response_html="<p>(◕▽◕) newer</p>",
            user_prompt="newer prompt",
        ),
        _takeout_entry(
            time="2026-04-14T09:00:00Z",
            response_html="<p>(╥﹏╥) older</p>",
            user_prompt="older prompt",
        ),
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))
    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 2
    # Older first.
    assert rows[0].first_word == "(╥﹏╥)"
    assert rows[1].first_word == "(◕▽◕)"
    assert rows[0].timestamp < rows[1].timestamp


def test_takeout_skips_non_gemini_headers(tmp_path: Path) -> None:
    """``header`` field filters: only entries with ``"Gemini"`` in the
    label come through. Entries from Search, Maps, YouTube, etc. with
    the same shape but a different header are skipped."""
    activity = [
        # Non-Gemini products — silent skip.
        {
            "header": "Maps",
            "time": "2026-04-15T10:00:00Z",
            "title": "Searched for the moon",
            "safeHtmlItem": [{"html": "<p>(◕ω◕) shouldn't fire</p>"}],
        },
        {
            "header": "Search",
            "time": "2026-04-15T11:00:00Z",
            "title": "Searched for `(◕ω◕)`",
            "safeHtmlItem": [{"html": "<p>(◕ω◕) shouldn't fire</p>"}],
        },
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))
    rows = list(iter_gemini_export([tmp_path]))
    assert rows == []


def test_aistudio_and_takeout_coexist_in_one_dir(tmp_path: Path) -> None:
    """The reader auto-dispatches per-file shape, so a user can dump
    their AI Studio + Takeout exports into the same directory and run
    one parse command."""
    _write_export(
        tmp_path / "aistudio-conv.json",
        model="gemini-2.5-pro",
        chunks=[
            {"role": "user", "text": "hi"},
            {"role": "model", "text": "(´▽`) hello!"},
        ],
    )
    activity = [
        _takeout_entry(response_html="<p>(´ε｀ ) takeout reply</p>"),
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))
    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 2
    sources = {r.source for r in rows}
    assert sources == {"gemini-aistudio-export", "gemini-takeout-export"}


def test_takeout_skips_entries_without_response(tmp_path: Path) -> None:
    """Some MyActivity entries have an action (``Asked Gemini``) but
    no ``safeHtmlItem`` — partial / interrupted requests, or rows
    where the user hit "stop generating" before any response landed.
    Skip cleanly rather than emitting an empty-body row.
    """
    activity = [
        _takeout_entry(response_html=None),
        _takeout_entry(
            response_html="<p>(◕ω◕) good one</p>",
        ),
    ]
    (tmp_path / "MyActivity.json").write_text(json.dumps(activity))
    rows = list(iter_gemini_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◕ω◕)"
