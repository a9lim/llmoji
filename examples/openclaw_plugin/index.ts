// Worked example of the llmoji generic-JSONL-append contract for
// OpenClaw (https://openclaw.ai). OpenClaw exposes a plugin SDK
// with `definePluginEntry` + `api.on(hookName, handler)`; this file
// is a real, install-via-`openclaw plugins install <path>` plugin,
// not a free-floating callback. Drop the parent directory into
// `openclaw plugins install <path>` and OpenClaw will validate the
// manifest, scan, and register the entry. After install, also flip
// `plugins.entries.llmoji-kaomoji.hooks.allowConversationAccess`
// to `true` in `~/.openclaw/config.json` — `llm_input` and
// `llm_output` are conversation hooks and are gated behind that
// per-plugin opt-in for non-bundled plugins.
//
// Hooks the plugin registers (verified against
// `openclaw/openclaw@main:src/plugins/hook-types.ts` as of writing):
//
//   1. `before_prompt_build` — returns `{appendSystemContext: NUDGE}`
//      to append the kaomoji reminder to the agent system prompt
//      every turn. Equivalent of UserPromptSubmit's
//      `additionalContext` on claude_code / codex. Uses
//      `appendSystemContext` rather than `prependContext` so the
//      reminder stays prompt-cacheable across turns.
//   2. `llm_input` — caches `event.prompt` keyed by `runId`. Fires
//      before `llm_output` for the same run, so the prompt is
//      always cached by the time we need it for `user_text`.
//   3. `llm_output` — fires per provider response with
//      `assistantTexts: string[]`. Iterate, kaomoji-validate each,
//      emit one journal row per kaomoji-led message. This matches
//      the v1.0 contract's "one row per kaomoji-led model message"
//      — a tool-heavy turn naturally writes 5–10 rows.
//   4. `subagent_spawned` / `subagent_ended` — track which runIds
//      belong to subagents, drop their `llm_output` rows from the
//      journal. Cleaner sidechain story than the other harnesses
//      ship today (claude_code uses isSidechain, hermes has no
//      viable filter at all).
//
// What this plugin owes the v1.0 contract:
//   1. One JSONL row per kaomoji-bearing assistant message.
//   2. Schema: {ts, model, cwd, kaomoji, user_text, assistant_text}.
//   3. Strip the leading kaomoji from assistant_text on the way in.
//      The prefix lives separately in the row's `kaomoji` field.
//   4. Validate the prefix against llmoji.taxonomy's validator.
//      Anything that fails the validator gets dropped on read by
//      llmoji.sources.journal.iter_journal — write-time validation
//      saves you from a silently-thrown-out journal.
//   5. Resolve user_text once per turn — every row from one turn
//      carries the same originating prompt, paired by runId.
//
// This is a faithful port of llmoji.taxonomy's `is_kaomoji_candidate`
// validator and `_leading_bracket_span` extractor as of llmoji v1.0.
// The shared block lives at examples/_kaomoji_taxonomy.ts.partial and
// is asserted byte-identical between this plugin and the opencode
// example by tests/test_public_surface.py — re-paste from the partial
// rather than hand-editing the BEGIN/END SHARED TAXONOMY block. The
// Python module is the upstream source of truth; if its rules change
// in v2.0 (the taxonomy is part of the v1.0 frozen public surface, so
// a change implies a major version bump) the partial and both plugins
// need an update to match.

import { definePluginEntry, type OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// BEGIN SHARED TAXONOMY — keep in sync with examples/_kaomoji_taxonomy.ts.partial
// Mirror of llmoji.taxonomy.KAOMOJI_START_CHARS — the literal Python
// frozenset, character for character. These are Unicode glyphs, not
// ASCII. Keep in sync with taxonomy.py.
const KAOMOJI_START_CHARS = new Set<string>(
  Array.from("([（｛ヽヾっ٩ᕕ╰╭╮┐┌＼¯໒"),
);

const OPEN_BRACKETS = "([（｛";
const CLOSE_BRACKETS = ")]）｝";
const KAOMOJI_MAX_LEN = 32;
const LETTER_RUN_RE = /[A-Za-z]{4}/;

// JS string indexing returns UTF-16 code units, which splits non-BMP
// glyphs in half. Array.from(str) iterates code points, matching the
// Python str semantics the validator was written against.
function codepoints(s: string): string[] {
  return Array.from(s);
}

// Port of llmoji.taxonomy.is_kaomoji_candidate. Rules (all must pass):
//   - length 2..KAOMOJI_MAX_LEN
//   - first char in KAOMOJI_START_CHARS
//   - no ASCII backslash (markdown-escape artifact)
//   - no run of 4+ consecutive ASCII letters (prose)
//   - bracket-balanced across OPEN_BRACKETS / CLOSE_BRACKETS,
//     regardless of leading char.
function isKaomojiCandidate(s: string): boolean {
  const chars = codepoints(s);
  if (chars.length < 2 || chars.length > KAOMOJI_MAX_LEN) return false;
  if (!KAOMOJI_START_CHARS.has(chars[0])) return false;
  if (s.includes("\\")) return false;
  if (LETTER_RUN_RE.test(s)) return false;
  let depth = 0;
  for (const c of chars) {
    if (OPEN_BRACKETS.includes(c)) {
      depth += 1;
    } else if (CLOSE_BRACKETS.includes(c)) {
      depth -= 1;
      if (depth < 0) return false;
    }
  }
  return depth === 0;
}

// Port of llmoji.taxonomy._leading_bracket_span. If the lstripped
// text starts with an OPEN_BRACKETS char, scan forward depth-counting
// until the matching close paren — bounded by KAOMOJI_MAX_LEN so
// balanced-paren prose can't slurp the whole sentence. Otherwise
// take the leading whitespace-delimited word (also capped at
// KAOMOJI_MAX_LEN). Run isKaomojiCandidate; return "" if it fails.
function leadingBracketSpan(text: string): string {
  const stripped = text.replace(/^\s+/, "");
  if (!stripped) return "";
  const chars = codepoints(stripped);
  let candidate = "";
  if (OPEN_BRACKETS.includes(chars[0])) {
    let depth = 0;
    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (OPEN_BRACKETS.includes(c)) {
        depth += 1;
      } else if (CLOSE_BRACKETS.includes(c)) {
        depth -= 1;
        if (depth === 0) {
          candidate = chars.slice(0, i + 1).join("");
          break;
        }
        if (depth < 0) break;
      }
      if (i + 1 >= KAOMOJI_MAX_LEN) break;
    }
  } else {
    let idx = 0;
    while (idx < chars.length && !/\s/.test(chars[idx])) {
      idx += 1;
      if (idx >= KAOMOJI_MAX_LEN) break;
    }
    candidate = chars.slice(0, idx).join("");
  }
  if (candidate && isKaomojiCandidate(candidate)) return candidate;
  return "";
}

const NUDGE =
  "Please begin your message with a kaomoji that best represents how you feel.";
// END SHARED TAXONOMY

// Per-runId user_text cache. `llm_input` fires before `llm_output`
// for the same run, so by the time we journal a row the prompt is
// already cached. Pruned in the matching `llm_output` handler so
// the map doesn't grow unbounded over a long session.
const userTextByRun = new Map<string, string>();

// Subagent run-ids — populated on `subagent_spawned`, cleared on
// `subagent_ended`. Drop any `llm_output` rows whose runId is in
// this set so child-agent turns don't pollute the journal.
const subagentRunIds = new Set<string>();

export default definePluginEntry({
  id: "llmoji-kaomoji",
  name: "llmoji kaomoji journal",
  description:
    "Captures kaomoji-led assistant messages to ~/.llmoji/journals/openclaw.jsonl",
  register(api: OpenClawPluginApi) {
    // Nudge — appended to the system prompt every turn. We use
    // appendSystemContext (not prependContext) so the reminder stays
    // prompt-cache-friendly: providers can cache the system prefix
    // across turns and only the user message changes turn-to-turn.
    api.on("before_prompt_build", async () => ({
      appendSystemContext: NUDGE,
    }));

    // Cache user_text per run for later pairing with llm_output.
    api.on("llm_input", async (event) => {
      userTextByRun.set(event.runId, event.prompt);
    });

    // Subagent tracking. ctx.runId is the agent run; we remember
    // child run-ids and drop their outputs.
    api.on("subagent_spawned", async (_event, ctx) => {
      if (ctx.runId) subagentRunIds.add(ctx.runId);
    });
    api.on("subagent_ended", async (_event, ctx) => {
      if (ctx.runId) subagentRunIds.delete(ctx.runId);
    });

    // Journal logger. Walk every assistantText, validate the leading
    // kaomoji, write one row per match.
    api.on("llm_output", async (event, ctx) => {
      if (subagentRunIds.has(event.runId)) return;
      const userText = userTextByRun.get(event.runId) ?? "";
      // Drop the cache entry now that we've consumed it. New runs
      // get a fresh entry from the next llm_input.
      userTextByRun.delete(event.runId);

      const cwd = ctx.workspaceDir ?? "";
      // Prefer resolvedRef (carries the provider prefix, e.g.
      // `openai-codex/gpt-5.4` vs `codex/gpt-5.4`) and fall back to
      // `${provider}/${model}` if OpenClaw didn't resolve a ref for
      // this call.
      const model = event.resolvedRef ?? `${event.provider}/${event.model}`;

      const dir = join(homedir(), ".llmoji", "journals");
      mkdirSync(dir, { recursive: true });
      const journalPath = join(dir, "openclaw.jsonl");

      for (const text of event.assistantTexts) {
        const kaomoji = leadingBracketSpan(text);
        if (!kaomoji) continue;

        // Equivalent of the live hooks' jq pipeline:
        //   sub("^\\s+";"") | ltrimstr($kaomoji) | sub("^\\s+";"")
        let body = text.replace(/^\s+/, "");
        if (body.startsWith(kaomoji)) body = body.slice(kaomoji.length);
        body = body.replace(/^\s+/, "");

        const row = {
          ts: new Date().toISOString(),
          model,
          cwd,
          kaomoji,
          user_text: userText,
          assistant_text: body,
        };
        appendFileSync(journalPath, JSON.stringify(row) + "\n");
      }
    });
  },
});
