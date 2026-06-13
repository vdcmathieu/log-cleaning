# Log cleaning

A small tool that takes AI assistant logs from many different GenAI sources — each
exported in its own messy format — and turns them into **one clean, homogeneous
format**. Whatever tool or provider a log came from (Claude Code, OpenAI Codex,
ChatGPT, …), the output looks the same and contains just **the user prompts and the
assistant's answers**, plus a short metadata header (models, providers, sessions,
working dirs, dates).

Everything that isn't part of the conversation — the model's internal
"thinking"/reasoning, tool calls, tool results, system & meta lines, injected
boilerplate — is stripped out.

```
raw-logs/            ->   clean-logs/
  alpha/                    alpha.json   + alpha.md
  beta.jsonl                beta.json    + beta.md
  gamma/                    gamma.json   + gamma.md
                            _summary.json
```

Each input is named by an id (a file like `beta.jsonl`, or a folder like `alpha/`).
That id is carried straight through to the output: `beta.jsonl` → `beta.json`, and
the `alpha/` folder → `alpha.json`.

## Usage

No dependencies — just Python 3.9+.

```bash
python3 clean_logs.py                    # raw-logs/  ->  clean-logs/
python3 clean_logs.py --raw RAW --out OUT
python3 clean_logs.py --include-sidechains   # also keep sub-agent (Task) transcripts
python3 clean_logs.py --no-merge-turns       # keep every assistant micro-turn separate
python3 clean_logs.py --no-markdown          # write JSON only, skip the .md transcripts
```

## Output format

`clean-logs/<id>.json` — the canonical, machine-readable result:

```jsonc
{
  "participant": "gamma",                 // the id, taken from the input's name
  "generated_at": "2026-06-13T21:14:18Z",
  "source": { "path": "raw-logs/gamma", "kind": "folder", "files": [ ... ] },
  "formats_detected": { "gamma/<file>.jsonl": "claude-code-v3" },
  "detection_scores": { ... },            // why each file was routed where
  "providers": ["fireworks"],
  "models": ["accounts/fireworks/routers/kimi-k2p6-turbo"],
  "stats": { "sessions": 4, "messages": 57, "user_messages": 29,
             "assistant_messages": 28, "parse_errors": 0 },
  "parse_errors": [],
  "sessions": [
    {
      "session_id": "019ead0e-...",
      "source_format": "claude-code-v3",
      "provider": "fireworks",
      "models": ["accounts/fireworks/routers/kimi-k2p6-turbo"],
      "cwd": "...",
      "started_at": "...", "ended_at": "...",
      "message_count": 14,
      "messages": [
        { "role": "user", "text": "ssh into ...", "timestamp": "..." },
        { "role": "assistant", "text": "The key commands ...",
          "timestamp": "...", "model": "...", "provider": "fireworks" }
      ]
    }
  ]
}
```

Every message is just `{role, text, timestamp}` (assistant turns also carry
`model`/`provider`). **The shape is identical no matter which tool or provider the
log came from** — OpenAI, Anthropic, xAI, Moonshot, … all collapse to this.

`clean-logs/<id>.md` is a human-readable transcript of the same content, for quickly
skimming a conversation. `clean-logs/_summary.json` is a one-glance overview of every
input.

## What counts as "clean"

**Kept:** the prompts the human typed, and the assistant's natural-language answers.
Consecutive assistant fragments are merged, so each prompt is followed by one
consolidated answer (agentic CLIs emit many small text fragments between tool calls).
Distinct user prompts are never merged together.

**Dropped:** model "thinking"/reasoning, tool calls and tool results, system/meta
records, synthetic CLI messages (e.g. login/API errors), sub-agent sidechains (unless
`--include-sidechains`), and injected boilerplate — caveats, command stdout, system
reminders, environment-context blocks, task notifications, context-compaction
summaries, and bare utility slash commands like `/clear`.

## How it handles different formats (and future ones)

Each log format is handled by a small **adapter**. An adapter says how confident it is
that it understands a file (`detect`) and how to turn that file into clean sessions
(`parse`). The file is routed to the highest-confidence adapter.

Adapters currently shipped:

| Adapter | Format |
|---|---|
| `claude-code-classic` | Claude Code `~/.claude/projects` JSONL |
| `claude-code-v3` | Newer Claude Code session log (`"version": 3`) |
| `codex-cli` | OpenAI Codex CLI log |
| `chatgpt-export` | ChatGPT "Export data" `conversations.json` |
| `generic` | Best-effort fallback for anything else |

**Future flexibility.** Two things make new/unfamiliar tools and models work without
edits:

1. The **`generic` adapter** — a last-resort handler that looks for a role and any
   text on each record. An unknown format still yields a readable transcript (flagged
   with `"source_format": "generic"` so you can see it fell through).
2. **Provider inference** — the provider name is derived from the model id when the log
   doesn't state it, covering Anthropic / OpenAI / Google / xAI / Moonshot / DeepSeek /
   Meta / Mistral / Alibaba / Cohere out of the box.

**To add a first-class adapter for a new tool:** subclass `Adapter` in `clean_logs.py`,
implement `detect(sample) -> float` and `parse(records, rel_path, opts) -> list[Session]`,
and add an instance to the `ADAPTERS` list. Nothing else changes — output stays uniform.

## Notes

- Input files are never modified; output is fully regenerated each run.
- Large files (hundreds of MB) are streamed line-by-line.
- Malformed JSON lines, BOMs, and unreadable files are tolerated and counted under each
  input's `parse_errors` rather than aborting the run.
- `detection_scores` and `formats_detected` are recorded per file so you can audit how
  every input was interpreted.
