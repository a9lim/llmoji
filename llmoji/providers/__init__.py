"""The harnesses llmoji captures from, one :class:`Provider` each."""

from __future__ import annotations

from .antigravity import AntigravityProvider
from .base import Provider, ProviderStatus, SettingsCorruptError
from .claude_code import ClaudeCodeProvider
from .codex import CodexProvider
from .grok import GrokProvider

# Listing order for status and autodetect.
PROVIDERS: dict[str, type[Provider]] = {
    "claude_code": ClaudeCodeProvider,
    "codex": CodexProvider,
    "grok": GrokProvider,
    "antigravity": AntigravityProvider,
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
    return PROVIDERS[name]()


__all__ = [
    "PROVIDERS",
    "AntigravityProvider",
    "ClaudeCodeProvider",
    "CodexProvider",
    "GrokProvider",
    "Provider",
    "ProviderStatus",
    "SettingsCorruptError",
    "get_provider",
]
