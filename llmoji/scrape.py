"""The in-memory row every reader produces.

The on-disk contract is the 6-field journal row (``ts``, ``model``,
``cwd``, ``kaomoji``, ``user_text``, ``assistant_text``);
:class:`ScrapeRow` is its in-memory form and free to evolve.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import split_leading_kaomoji

# User-role text a harness injects rather than a person or an agent
# writing: completion notices, reminders, slash-command output, skill
# bodies. Never a row's ``user_text``.
INJECTED_USER_PREFIXES = (
    "<task-notification>",
    "<system-reminder>",
    "<subagent_notification>",
    "<skill>",
    "<codex_internal_context",
    "<local-command-caveat>",
    "<local-command-stdout>",
)


def is_injected(text: str) -> bool:
    return text.lstrip().startswith(INJECTED_USER_PREFIXES)


@dataclass
class ScrapeRow:
    """One kaomoji-led assistant message."""

    source: str  # journal it came from: "claude_code-hook", "chatgpt_export", ...
    model: str | None
    timestamp: str  # ISO-8601
    cwd: str | None
    assistant_text: str  # the message with its leading kaomoji removed
    first_word: str  # the leading kaomoji
    surrounding_user: str  # the user prompt the message answers


def kaomoji_row(
    text: str,
    *,
    source: str,
    model: str | None,
    timestamp: str,
    cwd: str | None,
    user: str,
) -> ScrapeRow | None:
    """A row for ``text`` if it opens with a kaomoji, else ``None``."""
    split = split_leading_kaomoji(text)
    if split is None:
        return None
    face, body = split
    return ScrapeRow(
        source=source,
        model=model or None,
        timestamp=timestamp,
        cwd=cwd or None,
        assistant_text=body,
        first_word=face,
        surrounding_user=user.strip(),
    )
