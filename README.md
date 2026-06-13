# Log cleaning

Turns the messy, multi-format AI-usage logs that research participants sent us into
one clean, uniform per-participant file containing **only the human prompts and the
assistant's answers**, plus a small metadata header (models, providers, sessions,
working dirs, dates).

Everything the research doesn't need — the model's internal "thinking"/reasoning,
tool calls, tool results, system & meta lines, injected boilerplate — is stripped.

```
raw-logs/            ->   clean-logs/
  P001/                     P001.json   + P001.md
  P006.jsonl                P006.json   + P006.md
  P052.jsonl                P052.json   + P052.md
  P093/                     P093.json   + P093.md
  P101.jsonl                P101.json   + P101.md
                            _summary.json
```

The participant id is taken from the **name of the input** (file stem or folder
name), so `P006.jsonl` → `P006.json` and the `P093/` folder → `P093.json`.

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

`clean-logs/<PID>.json` — the canonical, machine-readable result:

```jsonc
{
  "participant": "P093",
  "generated_at": "2026-06-13T21:14:18Z",
  "source": { "path": "raw-logs/P093", "kind": "folder", "files": [ ... ] },
  "formats_detected": { "P093/<file>.jsonl": "claude-code-v3" },
  "detection_scores": { ... },          // why each file was routed where
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
      "cwd": "/Users/samuel/zurihack",
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

`clean-logs/<PID>.md` is a human-readable transcript of the same content, for
quickly skimming "what did this participant actually ask".

`clean-logs/_summary.json` is a one-glance overview of every participant.

## What counts as "clean"

**Kept:** genuine human-typed prompts, and the assistant's natural-language answers.
Consecutive same-role turns are merged, so each prompt is followed by one
consolidated answer (agentic CLIs emit many small text fragments between tool calls;
`--no-merge-turns` disables this).

**Dropped:** model "thinking"/reasoning, tool calls and tool results, system/meta
records, synthetic CLI messages (e.g. login/API errors), sub-agent sidechains (unless
`--include-sidechains`), and injected boilerplate — caveats, command stdout, system
reminders, environment-context blocks, and bare utility slash commands like `/clear`.

## How it handles different formats (and future ones)

Each log format is handled by a small **adapter**. An adapter says how confident it is
that it understands a file (`detect`) and how to turn that file into clean sessions
(`parse`). The file is routed to the highest-confidence adapter.

Adapters currently shipped:

| Adapter | Format | Seen in |
|---|---|---|
| `claude-code-classic` | Claude Code `~/.claude/projects` JSONL | P001, P006, P052 |
| `claude-code-v3` | Newer Claude Code session log (`"version": 3`) | P093 |
| `codex-cli` | OpenAI Codex CLI log | P101 |
| `chatgpt-export` | ChatGPT "Export data" `conversations.json` | (future) |
| `generic` | Best-effort fallback for anything else | (future) |

**Future flexibility.** Two things make new/unfamiliar tools and models work without
edits:

1. The **`generic` adapter** — a last-resort handler that looks for a role and any
   text on each record. An unknown future format still yields a readable transcript
   (flagged with `"source_format": "generic"` so you can see it fell through).
2. **Provider inference** — the provider name is derived from the model id when the log
   doesn't state it (`infer_provider`), covering Anthropic / OpenAI / Google / xAI /
   Moonshot / DeepSeek / Meta / Mistral / Alibaba / Cohere out of the box.

**To add a first-class adapter for a new tool:** subclass `Adapter` in `clean_logs.py`,
implement `detect(sample) -> float` and `parse(records, rel_path, opts) -> list[Session]`,
and add an instance to the `ADAPTERS` list. Nothing else changes — output stays uniform.

## Notes for reproducibility

- Input files are never modified; output is fully regenerated each run.
- Large files (e.g. the 95 MB `P006.jsonl`) are streamed line-by-line.
- Malformed JSON lines, BOMs, and unreadable files are tolerated and counted under
  each participant's `parse_errors` rather than aborting the run.
- `detection_scores` and `formats_detected` are recorded per file so you can audit how
  every input was interpreted.
