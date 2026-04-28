// Worked example of the llmoji generic-JSONL-append contract for
// opencode (https://opencode.ai). opencode's plugin system is
// TypeScript/JavaScript only — it auto-loads `.ts` / `.js` files
// from `~/.config/opencode/plugins/` (global) or `.opencode/plugins/`
// (project). There's no shell-hook escape hatch, so a first-class
// llmoji provider would have to ship as a TS plugin rather than the
// rendered-bash pattern the other providers use; until that lands,
// motivated opencode users can drop this file into their plugins
// dir and get the same per-turn capture.
//
// The plugin registers two hooks against the public Hooks interface
// (see `packages/plugin/src/index.ts` in the opencode repo):
//
//   1. `experimental.chat.system.transform` — appends the kaomoji
//      reminder to the system-prompt array on every model
//      invocation. Fires per turn, equivalent to the
//      UserPromptSubmit nudge in claude_code / codex.
//   2. `event` — fires for every Event (`message.updated`,
//      `message.part.updated`, `session.idle`, …). We gate on
//      completed assistant messages, dedupe by message id, query
//      the message's parts via the SDK, and emit one journal row
//      per kaomoji-led assistant message. This matches the
//      "one row per kaomoji-led model message" contract the live
//      shell hooks honor for claude_code / codex / hermes.
//
// What this plugin owes the v1.0 contract:
//   1. One JSONL row per kaomoji-bearing assistant message (NOT
//      per turn — a tool-heavy turn easily writes 5–10 messages).
//   2. Schema: {ts, model, cwd, kaomoji, user_text, assistant_text}.
//   3. Strip the leading kaomoji from assistant_text on the way in.
//      The prefix lives separately in the row's `kaomoji` field.
//   4. Validate the prefix against llmoji.taxonomy's validator.
//      Anything that fails the validator gets dropped on read by
//      llmoji.sources.journal.iter_journal — write-time validation
//      saves you from a silently-thrown-out journal.
//   5. Resolve user_text once per turn — every row from one turn
//      carries the same originating prompt. We cache it on
//      `chat.message` and read it back on each assistant
//      `message.updated`.
//
// Compared to the openclaw_plugin/ example: that one ports the
// validator the same way, but assumes a stop-event payload it can
// inspect directly. opencode doesn't fire one-shot stop events with
// the assistant text inline — events are partial state updates and
// the message contents come from the SDK. So this file does the
// same validator port plus an SDK round-trip per completed message.
//
// This is a faithful port of llmoji.taxonomy's `is_kaomoji_candidate`
// validator and `_leading_bracket_span` extractor as of llmoji v1.0.
// The Python module is canonical; if its rules change in v2.0 (the
// taxonomy is part of the v1.0 frozen public surface, so a change
// implies a major version bump) this file needs an update to match.

import type { Plugin, PluginInput } from "@opencode-ai/plugin";
import type { AssistantMessage, Part, TextPart } from "@opencode-ai/sdk";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

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

// Per-session cache of the most recent user-message text. Stamped
// on `chat.message`, read back on each assistant `message.updated`
// so every row from one turn carries the same originating prompt.
const userTextBySession = new Map<string, string>();

// Dedupe set: `event` for `message.updated` may fire more than once
// per assistant message as state transitions (streaming → completed
// → finish reason set). We only want to emit one row per message.
const journaledMessages = new Set<string>();

const opencodeLlmoji: Plugin = async ({ client }: PluginInput) => ({
  // Nudge — appended to the system prompt array every model call.
  // opencode concatenates the array, so push-as-suffix puts our
  // reminder near the end of the prompt where it has the strongest
  // recency effect. NOT idempotent across plugin reloads in the
  // same session; opencode rebuilds the system prompt per call.
  "experimental.chat.system.transform": async (_input, output) => {
    output.system.push(NUDGE);
  },

  // Cache user text per session. UserMessage parts are a Part[] —
  // pull text-typed parts only and concat. Synthetic / injected
  // parts have `synthetic: true`; we drop those so the cached
  // user_text reflects what the user actually typed.
  "chat.message": async ({ sessionID }, { parts }) => {
    const text = parts
      .filter((p): p is TextPart => p.type === "text" && !p.synthetic)
      .map((p) => p.text)
      .join("");
    userTextBySession.set(sessionID, text);
  },

  // Journal logger. Listens for every event; gates to completed
  // assistant messages, dedupes by id, walks parts via the SDK to
  // recover the first text block.
  event: async ({ event }) => {
    if (event.type !== "message.updated") return;
    const info = event.properties.info;
    if (info.role !== "assistant") return;
    const assistant = info as AssistantMessage;
    if (!assistant.time?.completed) return;
    if (journaledMessages.has(assistant.id)) return;
    journaledMessages.add(assistant.id);

    // SDK round-trip: EventMessageUpdated only carries the message
    // info, not its parts. `client.session.messages` returns the
    // full {info, parts}[] for the session. Find the matching id
    // and pull its first text part. (The SDK shape here is the
    // public surface as of opencode's current SDK; if opencode
    // ships a `client.session.message.get(id, messageID)` in a
    // future SDK release, prefer that — it avoids the full-list
    // pull on every assistant completion.)
    let messages: { info: { id: string }; parts: Part[] }[] = [];
    try {
      const res = await client.session.messages({
        path: { id: assistant.sessionID },
      });
      messages = (res as { data?: typeof messages }).data ?? [];
    } catch {
      // SDK unavailable / session gone — skip rather than crash
      // the agent loop. The journal is best-effort.
      return;
    }
    const entry = messages.find((m) => m.info.id === assistant.id);
    if (!entry) return;

    const firstText = entry.parts.find(
      (p): p is TextPart => p.type === "text" && !p.synthetic,
    );
    if (!firstText?.text) return;

    const kaomoji = leadingBracketSpan(firstText.text);
    if (!kaomoji) return;

    // Equivalent of the live hooks' jq pipeline:
    //   sub("^\\s+";"") | ltrimstr($kaomoji) | sub("^\\s+";"")
    let body = firstText.text.replace(/^\s+/, "");
    if (body.startsWith(kaomoji)) body = body.slice(kaomoji.length);
    body = body.replace(/^\s+/, "");

    const row = {
      ts: new Date().toISOString(),
      model: `${assistant.providerID}/${assistant.modelID}`,
      cwd: assistant.path?.cwd ?? "",
      kaomoji,
      user_text: userTextBySession.get(assistant.sessionID) ?? "",
      assistant_text: body,
    };

    const dir = join(homedir(), ".llmoji", "journals");
    mkdirSync(dir, { recursive: true });
    appendFileSync(join(dir, "opencode.jsonl"), JSON.stringify(row) + "\n");
  },
});

export default opencodeLlmoji;
