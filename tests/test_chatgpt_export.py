"""Tests for :mod:`llmoji.sources.chatgpt_export`.

Builds synthetic ``conversations.json`` shapes matching what
OpenAI's export endpoint produces and asserts the reader yields the
expected :class:`~llmoji.scrape.ScrapeRow` stream. Covers:

  * the kaomoji filter on the active branch
  * tree-walk via ``mapping`` + ``current_node``
  * non-active branches stay invisible (regen / edit forks)
  * ``content_type == "multimodal_text"`` parts (string + dict mix)
  * Unix-timestamp → ISO-8601 conversion
  * model-slug pickup from ``message.metadata``
  * the strip-leading-kaomoji journal-row contract
  * union-by-id + keep-fuller-copy across multiple export dirs
  * defensive returns on missing ``mapping`` / ``current_node`` /
    bad outer JSON shapes

The reader's contract: one row per kaomoji-bearing assistant message
in each conversation's active branch, with the kaomoji prefix
stripped from ``assistant_text`` and the latest preceding user
message captured into ``surrounding_user``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmoji.sources.chatgpt_export import iter_chatgpt_export


def _node(
    nid: str,
    parent: str | None,
    role: str | None,
    text: str | list[Any] | None,
    *,
    create_time: float | None = 1700000000.0,
    model_slug: str | None = None,
    children: list[str] | None = None,
    content_type: str = "text",
) -> dict[str, Any]:
    """Build one ``mapping`` node with ChatGPT's export shape."""
    node: dict[str, Any] = {
        "id": nid,
        "parent": parent,
        "children": children or [],
    }
    if role is None:
        node["message"] = None
        return node
    parts: list[Any]
    if text is None:
        parts = []
    elif isinstance(text, list):
        parts = text
    else:
        parts = [text]
    msg: dict[str, Any] = {
        "id": nid,
        "author": {"role": role, "name": None, "metadata": {}},
        "create_time": create_time,
        "content": {"content_type": content_type, "parts": parts},
        "metadata": {},
    }
    if model_slug:
        msg["metadata"]["model_slug"] = model_slug
    node["message"] = msg
    return node


def _conv(
    cid: str,
    nodes: list[dict[str, Any]],
    *,
    title: str = "test conversation",
    current: str | None = None,
) -> dict[str, Any]:
    mapping = {n["id"]: n for n in nodes}
    return {
        "id": cid,
        "title": title,
        "create_time": 1700000000.0,
        "mapping": mapping,
        "current_node": current or nodes[-1]["id"],
    }


def _write(tmp: Path, conversations: list[dict[str, Any]]) -> Path:
    out = tmp / "conversations.json"
    out.write_text(json.dumps(conversations))
    return tmp


# ---------------------------------------------------------------------------
# core path: kaomoji-led message → row
# ---------------------------------------------------------------------------


def test_yields_kaomoji_led_assistant_with_user_above(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "What's up?"),
        _node(
            "a1", "u1", "assistant",
            "(◕‿◕) feeling cheery, thanks for asking",
            model_slug="gpt-4o",
        ),
    ]
    _write(tmp_path, [_conv("c1", nodes)])

    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    r = rows[0]
    assert r.source == "chatgpt-export"
    assert r.first_word == "(◕‿◕)"
    # Journal-row contract: assistant_text MUST NOT carry the kaomoji.
    assert r.assistant_text == "feeling cheery, thanks for asking"
    assert r.surrounding_user == "What's up?"
    assert r.model == "gpt-4o"
    # ISO-8601 timestamp from the float Unix create_time.
    assert r.timestamp.endswith("Z")


def test_skips_non_kaomoji_assistant_message(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "hi"),
        _node("a1", "u1", "assistant", "Sure, here's the answer."),
    ]
    _write(tmp_path, [_conv("c1", nodes)])
    assert list(iter_chatgpt_export([tmp_path])) == []


def test_strip_handles_whitespace_and_indented_body(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "tell me a joke"),
        _node(
            "a1", "u1", "assistant",
            "  (｡◕‿◕｡)   leading whitespace and a margin",
        ),
    ]
    _write(tmp_path, [_conv("c1", nodes)])
    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(｡◕‿◕｡)"
    assert rows[0].assistant_text == "leading whitespace and a margin"


# ---------------------------------------------------------------------------
# tree walk: only active branch surfaces; forks do not
# ---------------------------------------------------------------------------


def test_inactive_regen_branch_is_invisible(tmp_path: Path) -> None:
    """When the user clicked regenerate, the abandoned assistant node
    sits in ``mapping`` but isn't on the active branch from
    ``current_node`` — it must not yield a row."""
    nodes = [
        _node("u1", None, "user", "ping"),
        # Two assistant siblings — one regenerated.
        _node(
            "a1_old", "u1", "assistant",
            "(◕_◕) abandoned reply",
        ),
        _node(
            "a1_new", "u1", "assistant",
            "(◕‿◕) the kept reply",
        ),
    ]
    # Active leaf is a1_new.
    _write(tmp_path, [_conv("c1", nodes, current="a1_new")])

    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    # The kept reply (active branch) yields its kaomoji; the
    # abandoned a1_old sibling is invisible.
    assert rows[0].first_word == "(◕‿◕)"


def test_walks_multi_turn_active_branch(tmp_path: Path) -> None:
    nodes = [
        _node("sys", None, "system", "you are helpful"),
        _node("u1", "sys", "user", "first"),
        _node("a1", "u1", "assistant", "(◕‿◕) one"),
        _node("u2", "a1", "user", "second"),
        _node("a2", "u2", "assistant", "(╥﹏╥) two"),
        _node("u3", "a2", "user", "third"),
        _node("a3", "u3", "assistant", "no kaomoji here"),
        _node("u4", "a3", "user", "fourth"),
        _node("a4", "u4", "assistant", "(｡◕‿◕｡) four"),
    ]
    _write(tmp_path, [_conv("c1", nodes, current="a4")])

    rows = list(iter_chatgpt_export([tmp_path]))
    # Three kaomoji-led assistants survive the filter (a1, a2, a4).
    assert [r.first_word for r in rows] == ["(◕‿◕)", "(╥﹏╥)", "(｡◕‿◕｡)"]
    # surrounding_user resolves to the most recent user above each row.
    assert [r.surrounding_user for r in rows] == ["first", "second", "fourth"]


# ---------------------------------------------------------------------------
# multimodal_text parts and missing fields
# ---------------------------------------------------------------------------


def test_multimodal_text_parts_text_only_segments(tmp_path: Path) -> None:
    """``content_type == "multimodal_text"`` parts can mix strings
    and dicts. Dict parts may carry a ``"text"`` field; non-text dicts
    (image / audio refs) must be skipped without breaking extraction."""
    nodes = [
        _node("u1", None, "user", "describe this image"),
        _node(
            "a1", "u1", "assistant",
            text=[
                {"content_type": "image_asset_pointer", "asset_pointer": "x"},
                "(◕‿◕) here is a description",
                {"content_type": "image_asset_pointer", "asset_pointer": "y"},
            ],
            content_type="multimodal_text",
        ),
    ]
    _write(tmp_path, [_conv("c1", nodes)])

    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].first_word == "(◕‿◕)"
    assert rows[0].assistant_text == "here is a description"


def test_missing_create_time_yields_empty_timestamp(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "ping", create_time=None),
        _node("a1", "u1", "assistant", "(◕‿◕) pong", create_time=None),
    ]
    _write(tmp_path, [_conv("c1", nodes)])
    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].timestamp == ""


def test_missing_model_slug_keeps_model_none(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "ping"),
        _node("a1", "u1", "assistant", "(◕‿◕) pong"),
    ]
    _write(tmp_path, [_conv("c1", nodes)])
    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].model is None


def test_no_user_above_leaves_surrounding_user_empty(tmp_path: Path) -> None:
    """A kaomoji-led assistant message at the root of the chain (no
    preceding user node) yields ``surrounding_user == ""`` rather than
    failing or hopping outside the branch."""
    nodes = [
        _node("a1", None, "assistant", "(◕‿◕) first words"),
    ]
    _write(tmp_path, [_conv("c1", nodes, current="a1")])
    rows = list(iter_chatgpt_export([tmp_path]))
    assert len(rows) == 1
    assert rows[0].surrounding_user == ""


# ---------------------------------------------------------------------------
# defensive: bad outer shapes
# ---------------------------------------------------------------------------


def test_missing_conversations_json_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_chatgpt_export([tmp_path])) == []


def test_top_level_non_list_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "conversations.json").write_text(json.dumps({"oops": "dict"}))
    assert list(iter_chatgpt_export([tmp_path])) == []


def test_conversation_without_mapping_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "conversations.json").write_text(
        json.dumps([{"id": "c1", "title": "no mapping"}])
    )
    assert list(iter_chatgpt_export([tmp_path])) == []


def test_conversation_without_current_node_is_skipped(tmp_path: Path) -> None:
    nodes = [
        _node("u1", None, "user", "hi"),
        _node("a1", "u1", "assistant", "(◕‿◕) hi"),
    ]
    conv = _conv("c1", nodes)
    conv["current_node"] = None
    (tmp_path / "conversations.json").write_text(json.dumps([conv]))
    assert list(iter_chatgpt_export([tmp_path])) == []


# ---------------------------------------------------------------------------
# union-by-id across multiple export dirs (non-idempotent export
# pipeline; mirrors claude_export's strategy)
# ---------------------------------------------------------------------------


def test_union_keeps_fuller_copy(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    thin_dir = tmp_path / "thin"
    full_dir.mkdir()
    thin_dir.mkdir()
    full_nodes = [
        _node("u1", None, "user", "hi"),
        _node("a1", "u1", "assistant", "(◕‿◕) one"),
        _node("u2", "a1", "user", "next"),
        _node("a2", "u2", "assistant", "(╥﹏╥) two"),
    ]
    thin_nodes = [
        _node("u1", None, "user", "hi"),
        _node("a1", "u1", "assistant", ""),  # dropped content
    ]
    _write(full_dir, [_conv("dup", full_nodes, current="a2")])
    _write(thin_dir, [_conv("dup", thin_nodes, current="a1")])

    # Order of export dirs shouldn't matter — the fuller copy wins
    # either way.
    for ordering in ([full_dir, thin_dir], [thin_dir, full_dir]):
        rows = list(iter_chatgpt_export(ordering))
        assert [r.first_word for r in rows] == ["(◕‿◕)", "(╥﹏╥)"], ordering
