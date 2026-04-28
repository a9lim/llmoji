"""Bash hook templates and shared partials.

Each ``*.sh.tmpl`` is a per-provider main hook, rendered at install
time via ``string.Template`` substitution. The ``*.partial`` files
are shared fragments inlined into every provider's main hook by
:meth:`llmoji.providers.base.HookInstaller.render_hook` so the
kaomoji validator and the JSONL journal-writer live in one place.

Substitutions performed by ``HookInstaller.render_hook``:

  ``$JOURNAL_PATH``           absolute path to the journal JSONL
  ``$KAOMOJI_VALIDATE``       expanded contents of
                              ``_kaomoji_validate.sh.partial`` with
                              ``$KAOMOJI_START_CASE`` (built from
                              :data:`llmoji.taxonomy.KAOMOJI_START_CHARS`)
                              and ``$SKIP_ACTION`` (per-provider bail
                              statement) substituted in
  ``$JOURNAL_WRITE``          expanded contents of
                              ``_journal_write.sh.partial``
  ``$INJECTED_PREFIXES_FILTER`` jq expression filtering system-injected
                              user-role payloads (see
                              :func:`llmoji.providers.base.render_injected_prefixes_filter`)
  ``$LLMOJI_VERSION``         the rendering package version

The package layout intentionally keeps these as data files (under
:mod:`llmoji._hooks`, accessed via ``importlib.resources``) rather
than f-strings inside Python: makes it easy to ``cat``-and-debug the
rendered hook on the user's filesystem, and the templating boundary
is the only line where we mix Python and bash.
"""
