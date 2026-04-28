// Worked example of the llmoji generic-JSONL-append contract for a
// harness we don't ship a first-class adapter for (here OpenClaw,
// whose stop-event hooks take the payload as a function argument
// rather than reading from stdin).
//
// Drop this into your OpenClaw hook config as the Stop handler; it
// appends one row per kaomoji-bearing assistant turn to
// ~/.llmoji/journals/openclaw.jsonl against the canonical six-field
// schema, and `llmoji analyze` picks it up automatically alongside
// the managed providers' journals.
//
// What you owe the contract:
//   1. One JSONL row per kaomoji-bearing assistant turn.
//   2. Schema: {ts, model, cwd, kaomoji, user_text, assistant_text}.
//   3. Strip the leading kaomoji from assistant_text on the way in.
//      The prefix lives separately in the row's `kaomoji` field.
//   4. Validate the prefix against llmoji.taxonomy's validator.
//      Anything that fails the validator gets dropped on read by
//      llmoji.sources.journal.iter_journal — write-time validation
//      saves you from a silently-thrown-out journal.
//   5. Drop subagent / sidechain dispatches if the harness exposes
//      a flag for them.
//
// This is a faithful port of llmoji.taxonomy's `is_kaomoji_candidate`
// validator and `_leading_bracket_span` extractor as of llmoji v1.0.
// The Python module is canonical; if its rules change in v2.0 (the
// taxonomy is part of the v1.0 frozen public surface, so a change
// implies a major version bump) this file needs an update to match.

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
// Python str semantics the validator was written against. All length
// checks and character lookups in this port go through code-point
// arrays for that reason.
function codepoints(s: string): string[] {
  return Array.from(s);
}

// Port of llmoji.taxonomy.is_kaomoji_candidate. Rules (all must pass):
//   - length 2..KAOMOJI_MAX_LEN
//   - first char in KAOMOJI_START_CHARS
//   - no ASCII backslash (markdown-escape artifact)
//   - no run of 4+ consecutive ASCII letters (prose)
//   - bracket-balanced across OPEN_BRACKETS / CLOSE_BRACKETS,
//     regardless of leading char (catches `ヽ(^`-style truncations
//     where a non-bracket leader precedes an unclosed inner `(`).
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

// Port of llmoji.taxonomy._leading_bracket_span. If the lstripped text
// starts with an OPEN_BRACKETS char, scan forward depth-counting until
// the matching close paren — bounded by KAOMOJI_MAX_LEN so balanced-
// paren prose can't slurp the whole sentence. Otherwise take the
// leading whitespace-delimited word (also capped at KAOMOJI_MAX_LEN).
// Run isKaomojiCandidate on the candidate; return "" if it fails.
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

// NOTE: field names below are guesses against OpenClaw's stop hook
// payload; if your harness names them differently, rename here. The
// row schema we WRITE is fixed by the v1.0 contract — only the input
// adapter is harness-specific.
interface StopPayload {
  user_message?: string;
  assistant_message?: string;
  model?: string;
  cwd?: string;
  is_sidechain?: boolean;
}

export function onStop(payload: StopPayload): void {
  if (payload.is_sidechain) return;

  const assistant = payload.assistant_message ?? "";
  if (!assistant.trim()) return;

  const kaomoji = leadingBracketSpan(assistant);
  if (!kaomoji) return;

  // Equivalent of the live hooks' jq pipeline:
  //   sub("^\\s+";"") | ltrimstr($kaomoji) | sub("^\\s+";"")
  // Strip leading whitespace, then the kaomoji prefix, then any
  // whitespace that was sitting between the kaomoji and the body.
  let body = assistant.replace(/^\s+/, "");
  if (body.startsWith(kaomoji)) body = body.slice(kaomoji.length);
  body = body.replace(/^\s+/, "");

  const row = {
    ts: new Date().toISOString(),
    model: payload.model ?? "",
    cwd: payload.cwd ?? "",
    kaomoji,
    user_text: payload.user_message ?? "",
    assistant_text: body,
  };

  const dir = join(homedir(), ".llmoji", "journals");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, "openclaw.jsonl");
  appendFileSync(path, JSON.stringify(row) + "\n");
}
