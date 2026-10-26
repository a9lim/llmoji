"""Tests for :mod:`llmoji.sources.claude_export`.

``iter_claude_export`` reads both claude.ai export shapes, told apart
per directory: a combined ``conversations.json`` (no ``model``), or one
``<title>.json`` per conversation beside ``export_summary.json`` (with
``model``). A directory holding both uses only the combined file.
Conversations union by uuid across directories, the copy with more
non-empty messages winning (first seen on a tie). Rows carry
``source == "claude_ai_export"`` and go through ``kaomoji_row``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmoji.sources.claude_export import iter_claude_export

TS = "2026-04-26T08:35:49.481783Z"


def _msg(
    uuid: str, sender: str, text: str, *, parent: str | None = None, where: str = "content",
) -> dict[str, Any]:
    """One ``chat_messages`` entry; ``where`` puts the text in the
    ``content[]`` blocks (the usual export) or the top-level ``text``."""
    return {
        "uuid": uuid,
        "text": text if where == "text" else "",
        "content": [{"type": "text", "text": text if where == "content" else ""}],
        "sender": sender,
        "created_at": TS,
        "parent_message_uuid": parent,
    }


def _conv(uuid: str, msgs: list[dict[str, Any]], *, model: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"uuid": uuid, "name": uuid, "chat_messages": msgs}
    if model is not None:
        out["model"] = model
    return out


def _exchange(uuid: str, prompt: str, reply: str) -> list[dict[str, Any]]:
    return [
        _msg(f"{uuid}-h", "human", prompt),
        _msg(f"{uuid}-a", "assistant", reply, parent=f"{uuid}-h"),
    ]


def _write_combined(out_dir: Path, convs: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversations.json").write_text(json.dumps(convs))


def _write_conv(out_dir: Path, filename: str, conv: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(conv))


# --- the two shapes ---


def test_combined_shape_yields_rows_with_model_none(tmp_path: Path) -> None:
    _write_combined(tmp_path, [
        _conv("c1", _exchange("m1", "what's up?", "(◕‿◕) feeling cheery")),
        _conv("c2", _exchange("m2", "ping", "(￣ω￣) pong")),
    ])

    rows = list(iter_claude_export([tmp_path]))
    assert [r.first_word for r in rows] == ["(◕‿◕)", "(￣ω￣)"]
    r = rows[0]
    assert r.source == "claude_ai_export"
    assert r.model is None
    assert r.cwd is None
    assert r.timestamp == TS
    # Journal-row contract: assistant_text does not carry the kaomoji.
    assert r.assistant_text == "feeling cheery"
    assert r.surrounding_user == "what's up?"


def test_per_file_shape_carries_model_and_skips_export_summary(tmp_path: Path) -> None:
    _write_conv(tmp_path, "Chat one.json", _conv(
        "c1", _exchange("m1", "hi", "(◕‿◕) hello"), model="claude-opus-4-7",
    ))
    _write_conv(tmp_path, "Chat two.json", _conv(
        "c2", _exchange("m2", "yo", "(•_•) hey"), model="claude-sonnet-4-5",
    ))
    # A summary shaped like a conversation: parsing it would add a row.
    _write_conv(tmp_path, "export_summary.json", _conv(
        "summary", _exchange("m3", "x", "(^_^) should never appear"),
        model="claude-opus-4-7",
    ))

    rows = list(iter_claude_export([tmp_path]))
    assert {(r.first_word, r.model) for r in rows} == {
        ("(◕‿◕)", "claude-opus-4-7"),
        ("(•_•)", "claude-sonnet-4-5"),
    }
    assert all(r.source == "claude_ai_export" for r in rows)


@pytest.mark.parametrize("model", [None, "", 7, ["claude"]])
def test_missing_or_non_string_model_is_none(tmp_path: Path, model: Any) -> None:
    conv = _conv("c1", _exchange("m1", "hi", "(◕‿◕) hello"))
    if model is not None:
        conv["model"] = model
    _write_conv(tmp_path, "c.json", conv)

    rows = list(iter_claude_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].model is None


def test_directory_with_both_shapes_uses_only_the_combined_file(tmp_path: Path) -> None:
    _write_combined(tmp_path, [_conv("c1", _exchange("m1", "q", "(◕‿◕) from the combined file"))])
    # Same uuid with another reply, and a conversation of its own:
    # neither per-file entry may surface.
    for uuid, reply in (("c1", "(•_•) per-file copy"), ("c2", "(^_^) per-file only")):
        _write_conv(tmp_path, f"{uuid}.json", _conv(
            uuid, _exchange(uuid, "q", reply), model="claude-opus-4-7",
        ))

    rows = list(iter_claude_export([tmp_path]))
    assert [r.assistant_text for r in rows] == ["from the combined file"]
    assert rows[0].model is None


# --- union by uuid across directories ---


def test_union_keeps_richer_copy_in_either_order(tmp_path: Path) -> None:
    """The export endpoint sometimes returns empty content for a
    conversation an earlier export had in full."""
    thin = [_msg("u1", "human", ""), _msg("a1", "assistant", "", parent="u1")]
    full = _exchange("m", "real question", "(￣ー￣) real reply")
    _write_combined(tmp_path / "thin", [_conv("c1", thin)])
    _write_conv(tmp_path / "full", "c1.json", _conv("c1", full, model="claude-opus-4-7"))

    for order in (["thin", "full"], ["full", "thin"]):
        rows = list(iter_claude_export([tmp_path / d for d in order]))
        assert len(rows) == 1, order
        assert rows[0].first_word == "(￣ー￣)"
        assert rows[0].surrounding_user == "real question"
        assert rows[0].model == "claude-opus-4-7"


def test_richness_counts_non_empty_messages_not_all_messages(tmp_path: Path) -> None:
    hollow = [_msg(f"h{i}", "human", "") for i in range(5)]
    one = _exchange("m", "q", "(◕‿◕) kept")[1:]  # the reply alone: parent missing
    _write_combined(tmp_path / "a", [_conv("c1", hollow)])
    _write_combined(tmp_path / "b", [_conv("c1", one)])

    rows = list(iter_claude_export([tmp_path / "a", tmp_path / "b"]))
    assert [r.assistant_text for r in rows] == ["kept"]


def test_union_tie_keeps_first_seen(tmp_path: Path) -> None:
    _write_combined(tmp_path / "a", [_conv("c1", _exchange("m", "q", "(◕‿◕) from a"))])
    _write_combined(tmp_path / "b", [
        _conv("c1", _exchange("m", "q", "(•_•) from b")),
        _conv("c2", _exchange("n", "q", "(^_^) only in b")),
    ])

    rows = list(iter_claude_export([tmp_path / "a", tmp_path / "b"]))
    assert sorted(r.assistant_text for r in rows) == ["from a", "only in b"]


# --- user text: parent-uuid walk ---


def test_parent_walk_lands_on_nearest_human_message(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "first question"),
        _msg("a1", "assistant", "(◕‿◕) first answer", parent="u1"),
        _msg("u2", "human", "second question"),
        _msg("a2", "assistant", "plain interlude", parent="u2"),
        _msg("a3", "assistant", "(•‿•) follow-up", parent="a2"),
    ]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    rows = list(iter_claude_export([tmp_path]))
    assert [r.surrounding_user for r in rows] == [
        "first question", "second question",
    ]


def test_parent_walk_skips_empty_human_text_and_strips(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "\n  the real prompt \n"),
        _msg("u2", "human", "  \n", parent="u1"),
        _msg("a1", "assistant", "(◕‿◕) hi", parent="u2"),
    ]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    assert list(iter_claude_export([tmp_path]))[0].surrounding_user == (
        "the real prompt"
    )


@pytest.mark.parametrize("parent", [None, "no-such-uuid"])
def test_no_reachable_human_leaves_user_empty(
    tmp_path: Path, parent: str | None,
) -> None:
    msgs = [_msg("a1", "assistant", "(◕‿◕) first words", parent=parent)]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    rows = list(iter_claude_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].surrounding_user == ""


# --- message text: top-level ``text`` vs ``content[]`` ---


def test_top_level_text_is_read(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "from text", where="text"),
        _msg("a1", "assistant", "(◕‿◕) also text", parent="u1", where="text"),
    ]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    r = list(iter_claude_export([tmp_path]))[0]
    assert r.assistant_text == "also text"
    assert r.surrounding_user == "from text"


def test_content_blocks_back_up_empty_text(tmp_path: Path) -> None:
    """Newer exports null one field or the other: with a blank ``text``,
    the text blocks stand in (thinking blocks and blank blocks do not
    count) and join with a newline."""
    reply = {
        "uuid": "a1",
        "text": "  ",
        "content": [
            {"type": "text", "text": " "},
            {"type": "thinking", "thinking": "(•_•) private chain"},
            {"type": "text", "text": "(◔_◔) sunny"},
            {"type": "text", "text": "and the answer is here"},
        ],
        "sender": "assistant",
        "created_at": TS,
        "parent_message_uuid": "u1",
    }
    _write_combined(tmp_path, [_conv("c1", [_msg("u1", "human", "weather?"), reply])])

    rows = list(iter_claude_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◔_◔)"
    assert rows[0].assistant_text == "sunny\nand the answer is here"


def test_non_empty_text_wins_over_content_blocks(tmp_path: Path) -> None:
    reply = _msg("a1", "assistant", "(•_•) from the blocks")
    reply["text"] = "(◕‿◕) from the text field"
    _write_combined(tmp_path, [_conv("c1", [reply])])

    assert list(iter_claude_export([tmp_path]))[0].assistant_text == (
        "from the text field"
    )


# --- which messages yield rows ---


def test_non_kaomoji_assistant_messages_are_skipped(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "hi"),
        _msg("a1", "assistant", "Sure, here's the answer.", parent="u1"),
        _msg("a2", "assistant", "Sure (◕‿◕)", parent="a1"),
        _msg("a3", "assistant", "", parent="a2"),
    ]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    assert list(iter_claude_export([tmp_path])) == []


def test_human_messages_never_yield_rows(tmp_path: Path) -> None:
    msgs = [
        _msg("u1", "human", "(◕‿◕) I open with a face too"),
        _msg("a1", "assistant", "no face here", parent="u1"),
    ]
    _write_combined(tmp_path, [_conv("c1", msgs)])

    assert list(iter_claude_export([tmp_path])) == []


@pytest.mark.parametrize(
    ("text", "face", "body"),
    [
        ("(•ᴗ•) **Done**", "(•ᴗ•)", "**Done**"),
        ("(•ᴗ•) **Done**\n\n- one\n- two", "(•ᴗ•)", "**Done**\n\n- one\n- two"),
        ("(◕‿◕) - item", "(◕‿◕)", "- item"),
        ("  (｡◕‿◕｡)   padded", "(｡◕‿◕｡)", "padded"),
        ("(◕‿◕)\nnext line", "(◕‿◕)", "next line"),
        ("(◕‿◕)", "(◕‿◕)", ""),
    ],
)
def test_face_and_body_split(
    tmp_path: Path, text: str, face: str, body: str,
) -> None:
    _write_combined(tmp_path, [_conv("c1", _exchange("m", "q", text))])

    rows = list(iter_claude_export([tmp_path]))
    assert len(rows) == 1
    assert (rows[0].first_word, rows[0].assistant_text) == (face, body)


# --- defensive: bad inputs don't abort the rest of the export ---


def test_bad_inputs_are_skipped(tmp_path: Path) -> None:
    """Corrupt, list-shaped and uuid-less per-file entries, and stray
    or uuid-less entries in a combined list, don't abort the export."""
    no_uuid = _conv("x", _exchange("n", "ok", "(^_^) no uuid"))
    no_uuid.pop("uuid")
    per_file = tmp_path / "per_file"
    _write_conv(per_file, "good.json", _conv("c1", _exchange("m", "ok", "(•_•) good")))
    _write_conv(per_file, "no_uuid.json", no_uuid)
    (per_file / "broken.json").write_text("{not valid json")
    (per_file / "list.json").write_text(json.dumps([{"uuid": "y"}]))
    combined = tmp_path / "combined"
    combined.mkdir()
    (combined / "conversations.json").write_text(json.dumps([
        "stray", 3, None, no_uuid,
        _conv("c2", _exchange("o", "ok", "(^_^) also good")),
    ]))

    rows = list(iter_claude_export([per_file, combined]))
    assert [r.first_word for r in rows] == ["(•_•)", "(^_^)"]


def test_missing_or_empty_directories_yield_nothing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    real = tmp_path / "real"
    _write_combined(real, [_conv("c1", _exchange("m", "q", "(•_•) here"))])

    rows = list(iter_claude_export([tmp_path / "gone", tmp_path / "empty", real]))
    assert [r.first_word for r in rows] == ["(•_•)"]
