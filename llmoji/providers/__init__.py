"""Hook-installer abstraction for the user-facing harness providers.

A *provider* is one coding harness whose per-turn data-capture
contract llmoji can install into. Two installer flavors live under
this package:

  - **bash hook** (claude_code, codex, hermes) — the traditional
    shape. Each provider ships a bash template under
    :mod:`llmoji._hooks`, gets rendered + written to the harness's
    hooks directory, and is registered via the harness's settings
    file (JSON for claude_code/codex, YAML for hermes). See
    :class:`HookInstaller` and :class:`JsonSettingsHookInstaller`.
  - **TS plugin** (opencode, openclaw, since 1.3) — the harness has
    no shell-hook escape hatch but does support TypeScript plugins.
    Each provider ships one or more ``.ts.tmpl`` / ``.json.tmpl``
    templates under :mod:`llmoji._plugins`, gets rendered + written
    to the harness's plugins directory, and (for openclaw) flips
    a JSON config flag to grant conversation-hook access. See
    :class:`PluginInstaller`.

Both flavors implement the same :class:`HookInstaller` interface so
the CLI walks every provider via the same install / uninstall /
status calls. :class:`PluginInstaller` is a subclass of
:class:`HookInstaller` for type-compatibility — it overrides the
bash-specific machinery while keeping :class:`ProviderStatus` and
the PROVIDERS-dict shape unchanged.

The class was named ``Provider`` in 1.1.0; 1.1.x renamed it to
``HookInstaller`` because the abstraction is about installing
per-turn capture, not about being a generic "provider". The
``providers/`` directory name stays — concrete subclasses are
still ``ClaudeCodeProvider`` / ``CodexProvider`` / ``HermesProvider``
/ ``OpencodeProvider`` / ``OpenclawProvider``.

Three things drive the abstraction:

  1. Where to write the hook script / plugin file
     (``hooks_dir`` for bash; ``plugin_dir`` for plugin).
  2. How to register it. JSON-settings bash providers (claude_code,
     codex) get the default :meth:`HookInstaller._register` /
     :meth:`HookInstaller._unregister` /
     :meth:`HookInstaller._check_registrations` against a
     ``hooks``-keyed settings shape. YAML bash providers (hermes)
     override with surgical YAML edits. Plugin providers either
     auto-register on file presence (opencode) or flip a JSON config
     flag (openclaw).
  3. Where the journal lives (``journal_path``) — a published
     uniform-schema JSONL the live hook / plugin appends to and
     which ``llmoji analyze`` reads. Bash providers write under
     ``~/.<harness>/kaomoji-journal.jsonl``; plugin providers write
     under ``~/.llmoji/journals/<name>.jsonl`` (the same generic-JSONL
     contract motivated users on unsupported harnesses can use).

Cross-corpus invariant note: 1.3 promotes opencode + openclaw to
first-class. The ``providers_seen`` list in shipped bundles now
includes ``opencode`` / ``openclaw`` rows; flag this on the dataset
card.
"""

from __future__ import annotations

from .base import HookInstaller, PluginInstaller, ProviderStatus
from .claude_code import ClaudeCodeProvider
from .codex import CodexProvider
from .hermes import HermesProvider
from .opencode import OpencodeProvider
from .openclaw import OpenclawProvider

# Registry order is the user-facing default order for ``llmoji status``
# and similar listings. Bash providers (the original three) come
# first in install-popularity order; plugin providers follow because
# they're newer and the host harnesses are less common in the corpus.
PROVIDERS: dict[str, type[HookInstaller]] = {
    "claude_code": ClaudeCodeProvider,
    "codex": CodexProvider,
    "hermes": HermesProvider,
    "opencode": OpencodeProvider,
    "openclaw": OpenclawProvider,
}


def get_provider(name: str) -> HookInstaller:
    """Look up a provider by name. Raises :class:`KeyError` on typos."""
    if name not in PROVIDERS:
        raise KeyError(
            f"unknown provider {name!r}; "
            f"known: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name]()


__all__ = [
    "HookInstaller",
    "PluginInstaller",
    "ProviderStatus",
    "ClaudeCodeProvider",
    "CodexProvider",
    "HermesProvider",
    "OpencodeProvider",
    "OpenclawProvider",
    "PROVIDERS",
    "get_provider",
]
