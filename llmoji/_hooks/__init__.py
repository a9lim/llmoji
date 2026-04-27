"""Bash hook templates.

Each ``*.sh.tmpl`` is rendered at provider-install time with
``string.Template`` substitutions:

  ``$JOURNAL_PATH``           absolute path to the journal JSONL
  ``$KAOMOJI_START_CASE``     bash ``case`` glob list, generated from
                              :data:`llmoji.taxonomy.KAOMOJI_START_CHARS`
  ``$INJECTED_PREFIXES_FILTER`` jq expression filtering system-injected
                              user-role payloads (see
                              :func:`llmoji.providers.base.render_injected_prefixes_filter`)
  ``$INJECTED_PREFIXES_LIST`` comma-separated jq-string list of the
                              same prefixes (for hermes's child-session
                              state file)
  ``$LLMOJI_VERSION``         the rendering package version

The package layout intentionally keeps these as data files (under
:mod:`llmoji._hooks`, accessed via ``importlib.resources``) rather
than f-strings inside Python: makes it easy to ``cat``-and-debug the
rendered hook on the user's filesystem, and the templating boundary
is the only line where we mix Python and bash.
"""
