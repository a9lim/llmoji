"""Tests for :mod:`llmoji.sources.claude_export_alt`.

The alt format is one ``<title>.json`` per conversation in a single
directory plus an ``export_summary.json`` sibling — same per-message
shape as the legacy combined ``conversations.json`` but split by
conversation and carrying ``model`` at the top level.

The per-message walker is shared with
:mod:`llmoji.sources.claude_export`, so coverage here focuses on
what differs between the two readers:

  * one-file-per-conversation discovery; ``export_summary.json`` is
    skipped
  * ``ScrapeRow.model`` populated from the conversation top-level
  * dedup-by-uuid + keep-fuller-copy across input dirs (a partial
    re-export should still pick the richer copy)
  * malformed / non-dict / non-conversation files are skipped
    quietly so one corrupt file doesn't kill a whole export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmoji.sources.claude_export_alt import iter_claude_export_alt


def _msg(
    uuid: str,
    sender: str,
    text: str,
    *,
    parent: str | None = None,
    created_at: str = "2026-04-26T08:35:49.481783Z",
) -> dict[str, Any]:
    """Build one ``chat_messages`` entry in the per-conversation
    export shape (same fields as the legacy combined export)."""
    return {
        "uuid": uuid,
        "text": "",
        "content": [
            {
                "start_timestamp": created_at,
                "stop_timestamp": created_at,
                "type": "text",
                "text": text,
                "citations": [],
            }
        ],
        "sender": sender,
        "index": 0,
        "created_at": created_at,
        "updated_at": created_at,
        "truncated": False,
        "attachments": [],
        "files": [],
        "sync_sources": [],
        "parent_message_uuid": parent,
    }


def _conv(
    uuid: str,
    name: str,
    msgs: list[dict[str, Any]],
    *,
    model: str | None = "claude-opus-4-7",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "uuid": uuid,
        "name": name,
        "summary": "",
        "created_at": "2026-04-26T08:35:48.594691Z",
        "updated_at": "2026-04-26T08:37:07.179647Z",
        "settings": {},
        "is_starred": False,
        "is_temporary": False,
        "platform": "claude.ai",
        "current_leaf_message_uuid": msgs[-1]["uuid"] if msgs else None,
        "chat_messages": msgs,
    }
    if model is not None:
        out["model"] = model
    return out


def _write_conv(out_dir: Path, filename: str, conv: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(conv))
    return path


def _write_summary(out_dir: Path, total: int) -> None:
    """Drop in the metadata sibling the export tool emits — the
    reader must skip it without raising."""
    (out_dir / "export_summary.json").write_text(
        json.dumps(
            {
                "export_date": "2026-05-05T17:34:44.074Z",
                "total_conversations": total,
                "successful_exports": total,
                "failed_exports": 0,
                "failed_conversations": [],
                "format": "json",
                "include_metadata": True,
            }
        )
    )


# ---------------------------------------------------------------------------
# core path: per-file discovery + kaomoji filter
# ---------------------------------------------------------------------------


def test_yields_kaomoji_led_assistant_from_per_file_dir(
    tmp_path: Path,
) -> None:
    msgs = [
        _msg("u1", "human", "what's up?"),
        _msg(
            "a1", "assistant",
            "(◕‿◕) feeling cheery, thanks for asking",
            parent="u1",
        ),
    ]
    _write_conv(tmp_path, "Chat one.json", _conv("c1", "Chat one", msgs))
    _write_summary(tmp_path, total=1)

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    r = rows[0]
    assert r.source == "claude-ai-alt-export"
    assert r.first_word == "(◕‿◕)"
    # Journal-row contract: assistant_text MUST NOT carry the kaomoji.
    assert r.assistant_text == "feeling cheery, thanks for asking"
    assert r.surrounding_user == "what's up?"
    # Per-conversation export DOES carry model — unlike the legacy combined
    # conversations.json which omits it.
    assert r.model == "claude-opus-4-7"


def test_export_summary_sibling_is_skipped(tmp_path: Path) -> None:
    """Reader must not try to parse export_summary.json as a
    conversation. Single conversation in the dir should yield exactly
    one row even though there are two .json files."""
    msgs = [
        _msg("u1", "human", "ping"),
        _msg("a1", "assistant", "(￣ω￣) pong", parent="u1"),
    ]
    _write_conv(tmp_path, "Ping conversation.json", _conv("c1", "Ping", msgs))
    _write_summary(tmp_path, total=1)

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(￣ω￣)"


def test_multiple_conversations_each_emit_rows(tmp_path: Path) -> None:
    msgs_a = [
        _msg("u1", "human", "first prompt"),
        _msg("a1", "assistant", "(•_•) first reply", parent="u1"),
    ]
    msgs_b = [
        _msg("u2", "human", "second prompt"),
        _msg("a2", "assistant", "(^_^) second reply", parent="u2"),
    ]
    _write_conv(
        tmp_path, "Convo A.json", _conv("c1", "Convo A", msgs_a),
    )
    _write_conv(
        tmp_path, "Convo B.json", _conv("c2", "Convo B", msgs_b),
    )

    rows = list(iter_claude_export_alt([tmp_path]))
    assert {r.first_word for r in rows} == {"(•_•)", "(^_^)"}
    assert {r.surrounding_user for r in rows} == {
        "first prompt", "second prompt",
    }


def test_skips_non_kaomoji_assistant_message(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "hi"),
        _msg("a1", "assistant", "Sure, here's the answer.", parent="u1"),
    ]
    _write_conv(tmp_path, "no kaomoji.json", _conv("c1", "no kaomoji", msgs))

    assert list(iter_claude_export_alt([tmp_path])) == []


def test_missing_model_field_yields_none_model(tmp_path: Path) -> None:
    """Defensive: a conversation without a top-level ``model`` (older
    versions of the alt export, partial files) should still yield rows
    with ``ScrapeRow.model = None``."""
    msgs = [
        _msg("u1", "human", "hello"),
        _msg("a1", "assistant", "(◕‿◕) hi", parent="u1"),
    ]
    _write_conv(
        tmp_path, "no model.json", _conv("c1", "no model", msgs, model=None),
    )

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    assert rows[0].model is None
    assert rows[0].first_word == "(◕‿◕)"


# ---------------------------------------------------------------------------
# parent walk for surrounding_user
# ---------------------------------------------------------------------------


def test_walks_parent_uuid_chain_for_surrounding_user(tmp_path: Path) -> None:
    """The kaomoji-led assistant message's parent chain may pass
    through earlier assistant turns; the walker should land on the
    nearest human-authored message with non-empty text."""
    msgs = [
        _msg("u1", "human", "the original question"),
        _msg("a1", "assistant", "earlier assistant turn", parent="u1"),
        _msg(
            "a2", "assistant",
            "(•‿•) the kaomoji-led follow-up",
            parent="a1",
        ),
    ]
    _write_conv(
        tmp_path, "chain.json", _conv("c1", "chain", msgs),
    )

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    assert rows[0].surrounding_user == "the original question"


# ---------------------------------------------------------------------------
# dedup across multiple input dirs
# ---------------------------------------------------------------------------


def test_dedup_across_dirs_keeps_richer_copy(tmp_path: Path) -> None:
    """Same conversation uuid in two dirs — the reader must yield
    rows from the version with more non-empty messages, matching the
    legacy combined export's repeated-export semantics."""
    sparse = [_msg("u1", "human", "")]  # empty user, no assistant
    rich = [
        _msg("u1", "human", "real question"),
        _msg("a1", "assistant", "(￣ー￣) real reply", parent="u1"),
    ]
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    _write_conv(dir_a, "convo.json", _conv("c1", "convo", sparse))
    _write_conv(dir_b, "convo.json", _conv("c1", "convo", rich))

    rows = list(iter_claude_export_alt([dir_a, dir_b]))
    assert len(rows) == 1
    assert rows[0].first_word == "(￣ー￣)"
    assert rows[0].surrounding_user == "real question"


# ---------------------------------------------------------------------------
# defensive: malformed / wrong-shape files
# ---------------------------------------------------------------------------


def test_malformed_json_file_is_skipped(tmp_path: Path) -> None:
    """A corrupt .json should not abort the rest of the export."""
    msgs = [
        _msg("u1", "human", "ok"),
        _msg("a1", "assistant", "(•_•) all good", parent="u1"),
    ]
    _write_conv(tmp_path, "good.json", _conv("c1", "good", msgs))
    (tmp_path / "broken.json").write_text("{not valid json")

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(•_•)"


def test_non_dict_top_level_is_skipped(tmp_path: Path) -> None:
    """A list at the top level (e.g. someone pasted the legacy combined
    export here by mistake) should be skipped quietly."""
    msgs = [
        _msg("u1", "human", "ok"),
        _msg("a1", "assistant", "(•_•) all good", parent="u1"),
    ]
    _write_conv(tmp_path, "good.json", _conv("c1", "good", msgs))
    (tmp_path / "list.json").write_text(json.dumps([{"uuid": "x"}]))

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1


def test_missing_uuid_field_is_skipped(tmp_path: Path) -> None:
    """Defensive against a payload that's the right shape but lacks
    a uuid (which would break dedup)."""
    msgs = [
        _msg("u1", "human", "ok"),
        _msg("a1", "assistant", "(•_•) all good", parent="u1"),
    ]
    good = _conv("c1", "good", msgs)
    bad = _conv("c2", "no uuid", msgs)
    bad.pop("uuid")
    _write_conv(tmp_path, "good.json", good)
    _write_conv(tmp_path, "no_uuid.json", bad)

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1


def test_nonexistent_dir_is_silently_skipped(tmp_path: Path) -> None:
    """One bad input path shouldn't kill the rest of the batch."""
    msgs = [
        _msg("u1", "human", "ok"),
        _msg("a1", "assistant", "(•_•) all good", parent="u1"),
    ]
    real = tmp_path / "real"
    _write_conv(real, "good.json", _conv("c1", "good", msgs))
    missing = tmp_path / "does_not_exist"

    rows = list(iter_claude_export_alt([missing, real]))
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# content-block fallback (mirrors the legacy export reader)
# ---------------------------------------------------------------------------


def test_content_block_text_fallback_when_top_text_empty(
    tmp_path: Path,
) -> None:
    """In real exports the top-level ``text`` field is often empty
    on assistant messages and the actual content lives in
    ``content[].text`` blocks. The shared walker handles this — the
    alt reader should inherit it transparently."""
    msgs = [
        _msg("u1", "human", "what's the weather"),
        # Build an assistant message with empty top-level text and
        # multiple content blocks (text + thinking + final text).
        {
            "uuid": "a1",
            "text": "",
            "content": [
                {"type": "text", "text": " "},
                {"type": "thinking", "thinking": "private chain..."},
                {
                    "type": "text",
                    "text": "(◔_◔) sunny and the answer is here",
                },
            ],
            "sender": "assistant",
            "created_at": "2026-04-26T08:35:50Z",
            "parent_message_uuid": "u1",
        },
    ]
    _write_conv(
        tmp_path, "fallback.json", _conv("c1", "fallback", msgs),
    )

    rows = list(iter_claude_export_alt([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◔_◔)"
    assert rows[0].assistant_text == "sunny and the answer is here"
