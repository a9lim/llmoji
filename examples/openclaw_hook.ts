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
//   4. Validate the prefix against the start-char set used by the
//      Python validator; the simplest port is the regex below.
//   5. Drop subagent / sidechain dispatches if the harness exposes
//      a flag for them.

import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// Mirror of llmoji.taxonomy.KAOMOJI_START_CHARS leading-character set.
// If you find this drifting from the Python source of truth, please
// bump llmoji and re-render — the Python set is canonical.
const KAOMOJI_START = /^[\p{L}\p{N}\p{S}\p{P}<>([{|/\\:;.,'"!?@#$%^&*~`\-+=]/u;

interface StopPayload {
  user_message?: string;
  assistant_message?: string;
  model?: string;
  cwd?: string;
  is_sidechain?: boolean;
}

export function onStop(payload: StopPayload): void {
  if (payload.is_sidechain) return;

  const assistant = (payload.assistant_message ?? "").trimStart();
  if (!assistant) return;

  // Take the leading whitespace-delimited token as the candidate
  // kaomoji prefix.
  const m = assistant.match(/^(\S+)\s+([\s\S]*)$/);
  if (!m) return;
  const [, kaomoji, body] = m;

  // Reject if the leading char isn't in the kaomoji start-char set.
  if (!KAOMOJI_START.test(kaomoji)) return;

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
