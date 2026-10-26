"""Where a message divides into face and body, and how the journal
reader repairs rows an older splitter divided wrongly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmoji.sources.journal import iter_journal
from llmoji.taxonomy import split_leading_kaomoji


@pytest.mark.parametrize("text,face,body", [
    ("(◕‿◕) Done.", "(◕‿◕)", "Done."),
    ("(◕‿◕)\n\nDone.\nMore.", "(◕‿◕)", "Done.\nMore."),
    ("  \n(◕‿◕) Done.", "(◕‿◕)", "Done."),
    # Markdown and prose after the face open the body.
    ("(•ᴗ•) **Task 1** done", "(•ᴗ•)", "**Task 1** done"),
    ("(◕‿◕) — done", "(◕‿◕)", "— done"),
    ("(¬_¬ ) 1. Wrong", "(¬_¬ )", "1. Wrong"),
    ("(◕‿◕) こんにちは、元気ですか", "(◕‿◕)", "こんにちは、元気ですか"),
    ("(◕‿◕) Привет мир", "(◕‿◕)", "Привет мир"),
    # Spaces and letters inside the brackets, arms and decorations after.
    ("(ง •̀_•́)ง let's go", "(ง •̀_•́)ง", "let's go"),
    ("(o_O) hmm", "(o_O)", "hmm"),
    ("(T_T) hello", "(T_T)", "hello"),
    ("(・̀‿・́)و fixed", "(・̀‿・́)و", "fixed"),
    ("(╯°□°)╯︵ ┻━┻ ugh", "(╯°□°)╯︵ ┻━┻", "ugh"),
    ("(ﾉ´ヮ`)ﾉ*: ・ﾟ Phase 1", "(ﾉ´ヮ`)ﾉ*: ・ﾟ", "Phase 1"),
    ("ᕕ( ᐛ )ᗒ go", "ᕕ( ᐛ )ᗒ", "go"),
    ("\\(^o^)/ yes", "\\(^o^)/", "yes"),
    ("ʕ•ᴥ•ʔ bear", "ʕ•ᴥ•ʔ", "bear"),
    # Bare faces.
    ("T-T why", "T-T", "why"),
    ("XD lol", "XD", "lol"),
    (":) ok", ":)", "ok"),
    ("^_^ yay", "^_^", "yay"),
    # An unclosed bracket falls back to the first word.
    ("(◕‿◕ unbalanced text", "(◕‿◕", "unbalanced text"),
])
def test_split(text: str, face: str, body: str) -> None:
    assert split_leading_kaomoji(text) == (face, body)


@pytest.mark.parametrize("text", [
    "Done.", "**Done**", "# Title", "(see below)", "Hello (◕‿◕)", "", "   ",
    "[/private/tmp/some/path/file.txt]",
])
def test_no_face(text: str) -> None:
    assert split_leading_kaomoji(text) is None


def test_nothing_but_whitespace_is_lost() -> None:
    for text in ["(•ᴗ•) **Task 1 ✅** — lore", "(ﾉ´ヮ`)ﾉ*: ・ﾟ Phase\n\n1"]:
        face, body = split_leading_kaomoji(text) or ("", "")
        assert "".join((face + body).split()) == "".join(text.split())


def _journal(tmp_path: Path, **row: str) -> Path:
    base = {"ts": "t", "model": "m", "cwd": "", "kaomoji": "(◕‿◕)", "user_text": "u", "assistant_text": "x"}
    path = tmp_path / "j.jsonl"
    path.write_text(json.dumps({**base, **row}) + "\n")
    return path


def test_reader_returns_text_an_older_splitter_put_in_the_face(tmp_path: Path) -> None:
    [row] = iter_journal(_journal(tmp_path, kaomoji="(•ᴗ•) **", assistant_text="Task 1** done"), source="s")
    assert (row.first_word, row.assistant_text) == ("(•ᴗ•)", "** Task 1** done")
    [row] = iter_journal(_journal(tmp_path, kaomoji="(◕‿◕) こんにちは", assistant_text=""), source="s")
    assert (row.first_word, row.assistant_text) == ("(◕‿◕)", "こんにちは")


def test_reader_buckets_by_the_bracket_span(tmp_path: Path) -> None:
    """``extract`` stays the bucketing step, so stored faces with arms
    land in the cells they always did."""
    [row] = iter_journal(_journal(tmp_path, kaomoji="(ง •̀_•́)ง"), source="s")
    assert row.first_word == "(ง •̀_•́)"


def test_reader_treats_synthetic_as_no_model(tmp_path: Path) -> None:
    [row] = iter_journal(_journal(tmp_path, model="<synthetic>"), source="claude_code-hook")
    assert row.model is None


def test_reader_blanks_injected_user_text(tmp_path: Path) -> None:
    """Hooks before 3.0 stored task notifications and reminders as the
    prompt; they read as no prompt at all."""
    for user in ("<task-notification>\n<task-id>x</task-id>", "  <system-reminder>\nx"):
        [row] = iter_journal(_journal(tmp_path, user_text=user), source="s")
        assert row.surrounding_user == ""
    [row] = iter_journal(_journal(tmp_path, user_text="<task>\nAudit"), source="s")
    assert row.surrounding_user == "<task>\nAudit"


def test_reader_skips_rows_without_a_face(tmp_path: Path) -> None:
    assert list(iter_journal(_journal(tmp_path, kaomoji="Done."), source="s")) == []
