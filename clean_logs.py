#!/usr/bin/env python3
"""
clean_logs.py — Normalise heterogeneous AI assistant logs into a single clean format.

GenAI assistant logs come in many shapes (Claude Code "classic" project logs, the
newer Claude Code v3 session format, OpenAI Codex CLI logs, and — in the future —
anything else). This tool reads everything under ``raw-logs/`` and writes a clean,
uniform JSON file per input under ``clean-logs/`` containing ONLY the parts worth
reading back:

  * the prompts the human actually typed, and
  * the assistant's natural-language answers,

with a small metadata header (id, models used, providers, sessions, working dirs,
dates). Everything else — chain-of-thought / "thinking", tool calls, tool results,
system & meta lines, injected boilerplate — is stripped out.

Design goals
------------
* **Provider-agnostic output.** Anthropic, OpenAI, xAI, Moonshot, Google, ... all
  collapse to the same ``{role, text, model, ...}`` shape.
* **Flexible / future-proof.** Formats are handled by pluggable *adapters*. Each
  adapter says how confident it is that it understands a file (``detect``) and how
  to turn that file into clean sessions (``parse``). To support a brand-new tool,
  add one adapter class and register it — nothing else changes. A best-effort
  ``GenericAdapter`` catches formats nobody has written an adapter for yet, so
  unknown logs still yield *something* readable.
* **No third-party dependencies.** Pure standard library, Python 3.9+.

Usage
-----
    python3 clean_logs.py                 # raw-logs/  ->  clean-logs/
    python3 clean_logs.py --raw RAW --out OUT
    python3 clean_logs.py --include-sidechains   # keep sub-agent transcripts too
    python3 clean_logs.py --no-markdown          # JSON only, skip .md transcripts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Normalised intermediate representation (the "clean" shape everything maps to)
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """One cleaned conversational turn: a human prompt or an assistant answer."""

    role: str  # "user" | "assistant"
    text: str
    timestamp: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "text": self.text}
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.role == "assistant":
            if self.model:
                d["model"] = self.model
            if self.provider:
                d["provider"] = self.provider
        return d


@dataclass
class Session:
    """A single conversation/session, after cleaning."""

    session_id: Optional[str]
    source_file: str
    source_format: str
    provider: Optional[str] = None
    models: List[str] = field(default_factory=list)
    cwd: Optional[str] = None
    project: Optional[str] = None
    git_branch: Optional[str] = None
    cli_version: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    messages: List[Message] = field(default_factory=list)

    # -- helpers used by adapters while building up a session ----------------
    def note_model(self, model: Optional[str]) -> None:
        if model and model != "<synthetic>" and model not in self.models:
            self.models.append(model)

    def note_time(self, ts: Any) -> None:
        ts = _norm_ts(ts)
        if not ts:
            return
        if self.started_at is None or ts < self.started_at:
            self.started_at = ts
        if self.ended_at is None or ts > self.ended_at:
            self.ended_at = ts

    def add_message(self, msg: Optional[Message]) -> None:
        if msg is None or not msg.text:
            return
        msg.timestamp = _norm_ts(msg.timestamp)
        self.messages.append(msg)
        self.note_time(msg.timestamp)
        if msg.role == "assistant":
            self.note_model(msg.model)

    def to_dict(self) -> Dict[str, Any]:
        header: Dict[str, Any] = {
            "session_id": self.session_id,
            "source_file": self.source_file,
            "source_format": self.source_format,
        }
        if self.provider:
            header["provider"] = self.provider
        if self.models:
            header["models"] = self.models
        if self.cwd:
            header["cwd"] = self.cwd
        if self.project:
            header["project"] = self.project
        if self.git_branch:
            header["git_branch"] = self.git_branch
        if self.cli_version:
            header["cli_version"] = self.cli_version
        if self.started_at:
            header["started_at"] = self.started_at
        if self.ended_at:
            header["ended_at"] = self.ended_at
        header["message_count"] = len(self.messages)
        header["messages"] = [m.to_dict() for m in self.messages]
        return header


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

_PROVIDER_BY_MODEL = [
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4-mini", "openai"),
    ("codex", "openai"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("kimi", "moonshot"),
    ("moonshot", "moonshot"),
    ("deepseek", "deepseek"),
    ("llama", "meta"),
    ("mistral", "mistralai"),
    ("mixtral", "mistralai"),
    ("qwen", "alibaba"),
    ("command-r", "cohere"),
]


def _norm_ts(ts: Any) -> Optional[str]:
    """Normalise a timestamp to an ISO-8601 'Z' string.

    Different tools log time differently: ISO strings (Claude Code records, Codex)
    or epoch numbers (Claude Code v3 ``message.timestamp`` is epoch milliseconds).
    Normalising up front keeps every stored timestamp comparable and human-readable.
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        s = ts.strip()
        return s or None
    if isinstance(ts, (int, float)):
        v = float(ts)
        if v >= 1e12:  # milliseconds since epoch
            v /= 1000.0
        try:
            return (
                datetime.fromtimestamp(v, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
        except (OverflowError, OSError, ValueError):
            return None
    return None


def infer_provider(model: Optional[str]) -> Optional[str]:
    """Best-effort provider name from a model id (used when the log omits it)."""
    if not model:
        return None
    m = model.lower()
    for needle, provider in _PROVIDER_BY_MODEL:
        if needle in m:
            return provider
    return None


def _extract_text_blocks(content: Any, keep_types: Tuple[str, ...]) -> str:
    """Join the text of the wanted block types from a message ``content`` value.

    ``content`` may be a plain string or a list of ``{"type": ..., "text": ...}``
    blocks. Only blocks whose ``type`` is in ``keep_types`` contribute text;
    everything else (thinking, tool_use, tool_result, images, ...) is ignored.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") in keep_types:
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        return "\n".join(parts).strip()
    return ""


# Patterns for boilerplate that gets injected into "user" turns by the CLIs but
# was never typed by the human. These are stripped from prompt text.
_STRIP_BLOCK_PATTERNS = [
    re.compile(r"<system-reminder>.*?</system-reminder>", re.S),
    re.compile(r"<local-command-stdout>.*?</local-command-stdout>", re.S),
    re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.S),
    re.compile(r"<task-notification>.*?</task-notification>", re.S),
]

# Wrappers that mark a whole "user" turn as machine-injected context rather than a
# genuine prompt. If a turn is *only* one of these, it is dropped entirely.
_INJECTED_CONTEXT_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
    "<skills_instructions>",
    "<user_instructions>",
    "<EXPERIMENTAL_",
    "<command-output>",
    "<task-notification>",
    "[SYSTEM NOTIFICATION",
)

# Opening phrase of a CLI-generated context-compaction summary (not human input).
_COMPACT_SUMMARY_PREFIX = "This session is being continued from a previous conversation"

_INTERRUPT_RE = re.compile(r"^\[Request interrupted by user[^\]]*\]$")
_CMD_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.S)
_CMD_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.S)
_CMD_CONTENTS_RE = re.compile(r"<command-contents>(.*?)</command-contents>", re.S)


def clean_user_text(text: Optional[str]) -> Optional[str]:
    """Turn a raw 'user' turn into the human's actual prompt, or ``None`` to drop.

    Drops: interruption markers, local-command caveats, bare slash commands, and
    machine-injected context blocks. Keeps: genuine free text, and slash commands
    that carry argument text (rendered as ``"<command> <args>"``).
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    if _INTERRUPT_RE.match(t):
        return None

    low = t.lower()
    if low.startswith("caveat: the messages below") and "<command-name>" not in t:
        return None

    # CLI-generated context-compaction summary — machine text, not a human prompt.
    if t.startswith(_COMPACT_SUMMARY_PREFIX):
        return None

    # Slash command invocations: keep only if they carry argument / content text.
    if "<command-name>" in t:
        name_m = _CMD_NAME_RE.search(t)
        args_m = _CMD_ARGS_RE.search(t)
        contents_m = _CMD_CONTENTS_RE.search(t)
        name = name_m.group(1).strip() if name_m else ""
        args = args_m.group(1).strip() if args_m else ""
        if not args and contents_m:
            args = contents_m.group(1).strip()
        if not args:
            return None  # bare utility command like /clear, /model, /init
        combined = (name + " " + args).strip()
        return combined or None

    # Whole-turn injected context (Codex environment blocks etc.).
    if t.startswith(_INJECTED_CONTEXT_PREFIXES):
        return None

    for pat in _STRIP_BLOCK_PATTERNS:
        t = pat.sub("", t)
    t = t.strip()
    return t or None


# ---------------------------------------------------------------------------
# Record reading (jsonl / json, BOM-tolerant, error-tolerant)
# ---------------------------------------------------------------------------


def iter_records(path: Path) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    """Yield ``(line_no, obj, error)`` for every record in a .jsonl or .json file.

    ``.jsonl``/``.ndjson`` is streamed one object per line (memory-friendly for the
    multi-hundred-MB files). ``.json`` is loaded whole; a top-level list yields its
    elements, anything else yields a single record. A UTF-8 BOM is handled
    transparently and a malformed line yields ``(line_no, None, error_message)``.
    """
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".log"):
        try:
            fh = open(path, encoding="utf-8-sig", errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable file
            yield 0, None, "cannot open: %s" % exc
            return
        with fh:
            for i, line in enumerate(fh, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    yield i, json.loads(s), None
                except json.JSONDecodeError as exc:
                    yield i, None, str(exc)
    else:  # treat everything else as a single JSON document
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:  # ValueError covers JSON + Unicode errors
            yield 0, None, str(exc)
            return
        if isinstance(data, list):
            for i, item in enumerate(data, 1):
                yield i, item, None
        else:
            yield 1, data, None


def obj_stream(path: Path, errors: List[str]) -> Iterator[Any]:
    """Stream just the successfully parsed objects, appending any errors to a list."""
    for line_no, obj, err in iter_records(path):
        if err is not None:
            errors.append("%s:%d %s" % (path.name, line_no, err))
            continue
        yield obj


def sample_records(path: Path, n: int = 400) -> List[Any]:
    """Read up to ``n`` parsed objects from the front of a file (for format detection)."""
    out: List[Any] = []
    for _line_no, obj, _err in iter_records(path):
        if obj is not None:
            out.append(obj)
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Adapter framework
# ---------------------------------------------------------------------------


class Adapter:
    """Base class for a log-format handler.

    Subclasses implement :meth:`detect` (confidence in [0, 1] that this adapter
    understands a file, given a sample of its records) and :meth:`parse` (turn the
    full stream of records into a list of cleaned :class:`Session` objects).
    """

    name = "base"

    def detect(self, sample: List[Any]) -> float:
        raise NotImplementedError

    def parse(self, records: Iterable[Any], rel_path: str, opts: "Options") -> List[Session]:
        raise NotImplementedError


@dataclass
class Options:
    include_sidechains: bool = False
    merge_turns: bool = True


def merge_consecutive(messages: List[Message]) -> List[Message]:
    """Collapse runs of consecutive ASSISTANT turns into one answer.

    Claude Code (and agentic CLIs generally) emit an assistant 'turn' for every
    step between tool calls, so a single answer arrives as many small text
    fragments once the tool calls are stripped. Merging those restores the natural
    'one prompt -> one answer' reading.

    User turns are NEVER merged: two consecutive human turns are two distinct
    prompts (the human sent two messages, or the assistant's turn in between was
    tool-only and got stripped). Fusing them would corrupt the per-prompt counts.
    """
    out: List[Message] = []
    for m in messages:
        if out and out[-1].role == m.role == "assistant":
            prev = out[-1]
            prev.text = (prev.text + "\n\n" + m.text).strip()
            if not prev.model and m.model:
                prev.model = m.model
            if not prev.provider and m.provider:
                prev.provider = m.provider
        else:
            out.append(m)
    return out


def _dict_records(sample: List[Any]) -> List[dict]:
    return [r for r in sample if isinstance(r, dict)]


# --- Claude Code "classic" project logs ------------------------------------


class ClaudeClassicAdapter(Adapter):
    """The standard ``~/.claude/projects/<proj>/<session>.jsonl`` format.

    One line per event; conversational lines have ``type`` user/assistant/system
    and a ``message`` dict with ``role`` + ``content`` (string or block list).
    Files may concatenate many sessions (keyed by ``sessionId`` / ``_session_id``).
    """

    name = "claude-code-classic"

    def detect(self, sample: List[Any]) -> float:
        recs = _dict_records(sample)
        if not recs:
            return 0.0
        hits = 0
        for r in recs:
            score = 0
            if "parentUuid" in r:
                score += 1
            if "isSidechain" in r:
                score += 1
            if "gitBranch" in r or "userType" in r:
                score += 1
            if r.get("type") in ("user", "assistant", "system"):
                score += 1
            # v3 markers should NOT be here
            if r.get("type") in ("session", "model_change", "thinking_level_change"):
                score -= 3
            if "modelId" in r or "responseModel" in r:
                score -= 2
            if score >= 2:
                hits += 1
        return min(1.0, hits / len(recs) * 1.2)

    def parse(self, records: Iterable[Any], rel_path: str, opts: Options) -> List[Session]:
        sessions: Dict[str, Session] = {}
        order: List[str] = []

        def get_session(rec: dict) -> Session:
            sid = rec.get("_session_id") or rec.get("sessionId") or "default"
            sess = sessions.get(sid)
            if sess is None:
                sess = Session(
                    session_id=None if sid == "default" else sid,
                    source_file=rel_path,
                    source_format=self.name,
                )
                sessions[sid] = sess
                order.append(sid)
            # opportunistically fill metadata from any line that carries it
            if not sess.cwd and rec.get("cwd"):
                sess.cwd = rec["cwd"]
            if not sess.git_branch and rec.get("gitBranch"):
                sess.git_branch = rec["gitBranch"]
            if not sess.cli_version and rec.get("version"):
                sess.cli_version = rec["version"]
            if not sess.project:
                proj = rec.get("_project")
                if proj:
                    sess.project = proj
                elif rec.get("cwd"):
                    sess.project = rec["cwd"]
            return sess

        for rec in records:
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            msg = rec.get("message")
            is_msg = isinstance(msg, dict) and msg.get("role") in ("user", "assistant")
            if not is_msg:
                continue
            if rec.get("isSidechain") and not opts.include_sidechains:
                continue
            if rec.get("isMeta") or rec.get("isCompactSummary"):
                continue

            sess = get_session(rec)
            ts = rec.get("timestamp") or msg.get("timestamp")
            role = msg.get("role")

            if role == "user":
                raw = _extract_text_blocks(msg.get("content"), keep_types=("text",))
                cleaned = clean_user_text(raw)
                if cleaned:
                    sess.add_message(Message(role="user", text=cleaned, timestamp=ts))
            else:  # assistant
                model = msg.get("model")
                if model == "<synthetic>":
                    continue  # injected CLI/error messages, not real model output
                text = _extract_text_blocks(msg.get("content"), keep_types=("text",))
                if text:
                    sess.add_message(
                        Message(
                            role="assistant",
                            text=text,
                            timestamp=ts,
                            model=model,
                            provider=infer_provider(model),
                        )
                    )

        for sess in sessions.values():
            if sess.provider is None and sess.models:
                sess.provider = infer_provider(sess.models[0])
        return _finalise_sessions(sessions[s] for s in order)


# --- Claude Code v3 session logs -------------------------------------------


class ClaudeV3Adapter(Adapter):
    """Newer Claude Code session log (``"version": 3``).

    Top-level ``type`` is session/message/model_change/custom/.... Messages carry
    ``message.content`` blocks of type ``text`` / ``thinking`` / ``toolCall``
    (camelCase), and a ``toolResult`` role for tool outputs. The active model is
    announced by ``model_change`` lines (``provider`` + ``modelId``) and echoed on
    each assistant message (``provider`` / ``model``).
    """

    name = "claude-code-v3"

    def detect(self, sample: List[Any]) -> float:
        recs = _dict_records(sample)
        if not recs:
            return 0.0
        markers = 0
        for r in recs:
            if r.get("type") in ("session", "model_change", "thinking_level_change"):
                markers += 2
            if r.get("type") == "custom" and "customType" in r:
                markers += 1
            if "modelId" in r and "provider" in r:
                markers += 2
            if "responseModel" in r:
                markers += 1
            m = r.get("message")
            if isinstance(m, dict):
                if m.get("role") == "toolResult":
                    markers += 1
                content = m.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "toolCall":
                            markers += 1
                            break
        return min(1.0, markers / (len(recs) * 1.5))

    def parse(self, records: Iterable[Any], rel_path: str, opts: Options) -> List[Session]:
        sess = Session(session_id=None, source_file=rel_path, source_format=self.name)
        # session id often only in the filename; fall back to that
        stem = Path(rel_path).stem
        cur_model: Optional[str] = None
        cur_provider: Optional[str] = None

        for rec in records:
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            if rtype == "session":
                sess.session_id = rec.get("id") or sess.session_id
                sess.cwd = rec.get("cwd") or sess.cwd
                if rec.get("version") is not None:
                    sess.cli_version = "v%s" % rec.get("version")
                sess.note_time(rec.get("timestamp"))
                continue
            if rtype == "model_change":
                cur_model = rec.get("modelId") or cur_model
                cur_provider = rec.get("provider") or cur_provider
                sess.note_model(cur_model)
                if cur_provider and not sess.provider:
                    sess.provider = cur_provider
                continue
            if rtype != "message":
                continue  # custom / thinking_level_change / etc.

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            ts = rec.get("timestamp") or msg.get("timestamp")

            if role == "user":
                raw = _extract_text_blocks(msg.get("content"), keep_types=("text",))
                cleaned = clean_user_text(raw)
                if cleaned:
                    sess.add_message(Message(role="user", text=cleaned, timestamp=ts))
            elif role == "assistant":
                model = msg.get("model") or msg.get("responseModel") or cur_model
                provider = msg.get("provider") or cur_provider
                text = _extract_text_blocks(msg.get("content"), keep_types=("text",))
                if text:
                    sess.add_message(
                        Message(
                            role="assistant",
                            text=text,
                            timestamp=ts,
                            model=model,
                            provider=provider or infer_provider(model),
                        )
                    )
            # role == "toolResult" (or anything else) -> dropped

        if sess.session_id is None:
            # filenames look like 2026-..._<uuid>.jsonl
            m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", stem)
            sess.session_id = m.group(1) if m else stem
        if sess.provider is None and sess.models:
            sess.provider = infer_provider(sess.models[0])
        return _finalise_sessions([sess])


# --- OpenAI Codex CLI logs --------------------------------------------------


class CodexAdapter(Adapter):
    """OpenAI Codex CLI session log.

    Records are ``{timestamp, type, payload}`` where ``type`` is one of
    ``session_meta`` / ``turn_context`` / ``event_msg`` / ``response_item``. The
    cleanest conversational signal is ``event_msg`` with inner type
    ``user_message`` / ``agent_message`` (already free of reasoning and injected
    context). ``response_item`` messages are used only as a fallback. A single file
    may hold several sessions, each introduced by a ``session_meta`` record.
    """

    name = "codex-cli"
    _INJECTED = ("<environment_context>", "<permissions instructions>",
                 "<skills_instructions>", "<user_instructions>")

    def detect(self, sample: List[Any]) -> float:
        recs = _dict_records(sample)
        if not recs:
            return 0.0
        hits = 0
        for r in recs:
            if "payload" in r and r.get("type") in (
                "session_meta", "turn_context", "event_msg", "response_item",
            ):
                hits += 1
            elif isinstance(r.get("payload"), dict) and "model_provider" in r["payload"]:
                hits += 1
        return min(1.0, hits / len(recs) * 1.3)

    def parse(self, records: Iterable[Any], rel_path: str, opts: Options) -> List[Session]:
        sessions: List[Session] = []
        cur: Optional[Session] = None
        cur_model: Optional[str] = None
        cur_provider: Optional[str] = None
        # per-session fallback buffers (response_item messages), used only if the
        # session produced no event_msg conversational turns
        fallback: Dict[int, List[Message]] = {}

        def new_session(payload: dict, ts: Optional[str]) -> Session:
            nonlocal cur_model, cur_provider
            s = Session(
                session_id=payload.get("id"),
                source_file=rel_path,
                source_format=self.name,
                cwd=payload.get("cwd"),
                provider=payload.get("model_provider"),
                cli_version=payload.get("cli_version"),
            )
            s.note_time(ts)
            cur_provider = payload.get("model_provider")
            cur_model = None
            sessions.append(s)
            fallback[id(s)] = []
            return s

        for rec in records:
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            payload = rec.get("payload")
            ts = rec.get("timestamp")
            if not isinstance(payload, dict):
                continue

            if rtype == "session_meta":
                cur = new_session(payload, ts)
                continue

            if cur is None:
                # a stream that starts mid-session: synthesise one
                cur = Session(session_id=None, source_file=rel_path, source_format=self.name)
                sessions.append(cur)
                fallback[id(cur)] = []

            if rtype == "turn_context":
                if payload.get("model"):
                    cur_model = payload["model"]
                    cur.note_model(cur_model)
                if payload.get("cwd") and not cur.cwd:
                    cur.cwd = payload["cwd"]
                continue

            if rtype == "event_msg":
                ptype = payload.get("type")
                if ptype == "user_message":
                    cleaned = clean_user_text(payload.get("message"))
                    if cleaned:
                        cur.add_message(Message(role="user", text=cleaned, timestamp=ts))
                elif ptype == "agent_message":
                    text = (payload.get("message") or "").strip()
                    if text:
                        cur.add_message(
                            Message(
                                role="assistant",
                                text=text,
                                timestamp=ts,
                                model=cur_model,
                                provider=cur_provider or infer_provider(cur_model),
                            )
                        )
                continue

            if rtype == "response_item" and payload.get("type") == "message":
                role = payload.get("role")
                if role not in ("user", "assistant"):
                    continue
                keep = ("input_text",) if role == "user" else ("output_text", "text")
                text = _extract_text_blocks(payload.get("content"), keep_types=keep)
                if role == "user":
                    text = clean_user_text(text) or ""
                if text:
                    fallback[id(cur)].append(
                        Message(
                            role=role,
                            text=text,
                            timestamp=ts,
                            model=cur_model if role == "assistant" else None,
                            provider=(cur_provider or infer_provider(cur_model)) if role == "assistant" else None,
                        )
                    )

        # apply fallback for sessions with no event-based turns
        for s in sessions:
            if not s.messages and fallback.get(id(s)):
                for m in fallback[id(s)]:
                    s.add_message(m)
            if s.provider is None and s.models:
                s.provider = infer_provider(s.models[0])

        return _finalise_sessions(sessions)


# --- OpenAI ChatGPT data export (conversations.json) -----------------------


class ChatGPTExportAdapter(Adapter):
    """OpenAI ChatGPT "Export data" format (``conversations.json``).

    A JSON array of conversation objects; each has a ``mapping`` of node-id ->
    ``{message: {author: {role}, content: {content_type, parts}}}`` forming a tree.
    We walk the tree in ``create_time`` order and keep user/assistant text parts.
    Included speculatively because ChatGPT exports are common and were named as an
    expected source; the generic adapter still covers it if this misses.
    """

    name = "chatgpt-export"

    def detect(self, sample: List[Any]) -> float:
        recs = _dict_records(sample)
        if not recs:
            return 0.0
        hits = 0
        for r in recs:
            if isinstance(r.get("mapping"), dict) and (
                "title" in r or "create_time" in r or "conversation_id" in r
            ):
                hits += 1
        return min(1.0, hits / len(recs)) if hits else 0.0

    def parse(self, records: Iterable[Any], rel_path: str, opts: Options) -> List[Session]:
        sessions: List[Session] = []
        for convo in records:
            if not isinstance(convo, dict) or not isinstance(convo.get("mapping"), dict):
                continue
            s = Session(
                session_id=convo.get("conversation_id") or convo.get("id"),
                source_file=rel_path,
                source_format=self.name,
                project=convo.get("title"),
            )
            nodes = [n for n in convo["mapping"].values() if isinstance(n, dict) and n.get("message")]

            def sort_key(node: dict) -> float:
                msg = node.get("message") or {}
                ct = msg.get("create_time")
                return ct if isinstance(ct, (int, float)) else 0.0

            for node in sorted(nodes, key=sort_key):
                msg = node.get("message") or {}
                author = (msg.get("author") or {}).get("role")
                if author not in ("user", "assistant"):
                    continue
                content = msg.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                text = ""
                if isinstance(parts, list):
                    text = "\n".join(p for p in parts if isinstance(p, str)).strip()
                ts = None
                ct = msg.get("create_time")
                if isinstance(ct, (int, float)):
                    ts = datetime.fromtimestamp(ct, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                model = (msg.get("metadata") or {}).get("model_slug")
                if author == "user":
                    cleaned = clean_user_text(text)
                    if cleaned:
                        s.add_message(Message(role="user", text=cleaned, timestamp=ts))
                else:
                    if text:
                        s.add_message(
                            Message(role="assistant", text=text, timestamp=ts,
                                    model=model, provider=infer_provider(model) or "openai")
                        )
            if s.provider is None:
                s.provider = "openai"
            sessions.append(s)
        return _finalise_sessions(sessions)


# --- Generic best-effort fallback ------------------------------------------


class GenericAdapter(Adapter):
    """Last-resort adapter for formats nobody has written a handler for yet.

    Looks for a role and any text-bearing field on each record and extracts what it
    can. Always returns a tiny non-zero confidence so a purpose-built adapter wins,
    but an unknown future format still yields a readable transcript.
    """

    name = "generic"
    _USER = {"user", "human", "prompt"}
    _ASSISTANT = {"assistant", "model", "ai", "bot", "agent"}

    def detect(self, sample: List[Any]) -> float:
        for r in _dict_records(sample):
            if self._role_of(r) and self._text_of(r):
                return 0.1
        return 0.05

    @staticmethod
    def _role_of(rec: dict) -> Optional[str]:
        for getter in (
            rec.get("role"),
            (rec.get("message") or {}).get("role") if isinstance(rec.get("message"), dict) else None,
            (rec.get("payload") or {}).get("role") if isinstance(rec.get("payload"), dict) else None,
            (rec.get("author") or {}).get("role") if isinstance(rec.get("author"), dict) else None,
        ):
            if isinstance(getter, str):
                return getter
        return None

    @staticmethod
    def _text_of(rec: dict) -> str:
        for container in (rec, rec.get("message"), rec.get("payload")):
            if not isinstance(container, dict):
                continue
            for key in ("content", "text", "message", "parts"):
                val = container.get(key)
                txt = _extract_text_blocks(val, keep_types=("text", "input_text", "output_text"))
                if txt:
                    return txt
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return ""

    def parse(self, records: Iterable[Any], rel_path: str, opts: Options) -> List[Session]:
        s = Session(session_id=Path(rel_path).stem, source_file=rel_path, source_format=self.name)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            role = self._role_of(rec)
            if role is None:
                continue
            role_l = role.lower()
            if role_l in self._USER:
                norm = "user"
            elif role_l in self._ASSISTANT:
                norm = "assistant"
            else:
                continue
            text = self._text_of(rec)
            ts = rec.get("timestamp") or (rec.get("message") or {}).get("timestamp") if isinstance(rec.get("message"), dict) else rec.get("timestamp")
            model = None
            if isinstance(rec.get("message"), dict):
                model = rec["message"].get("model")
            model = model or rec.get("model")
            if norm == "user":
                cleaned = clean_user_text(text)
                if cleaned:
                    s.add_message(Message(role="user", text=cleaned, timestamp=ts))
            else:
                if text:
                    s.add_message(Message(role="assistant", text=text, timestamp=ts,
                                          model=model, provider=infer_provider(model)))
        return _finalise_sessions([s])


# Registry. Order matters only as a tiebreak; detection scores decide the winner.
ADAPTERS: List[Adapter] = [
    ClaudeClassicAdapter(),
    ClaudeV3Adapter(),
    CodexAdapter(),
    ChatGPTExportAdapter(),
    GenericAdapter(),
]


def _finalise_sessions(sessions: Iterable[Session]) -> List[Session]:
    """Drop empty sessions and sort each session's models for stable output."""
    out: List[Session] = []
    for s in sessions:
        if not s.messages:
            continue
        out.append(s)
    return out


def choose_adapter(sample: List[Any]) -> Tuple[Adapter, Dict[str, float]]:
    """Score every adapter against a sample and return the best plus all scores."""
    scores: Dict[str, float] = {}
    best: Optional[Adapter] = None
    best_score = -1.0
    for ad in ADAPTERS:
        try:
            sc = ad.detect(sample)
        except Exception:  # an adapter's detector must never crash the run
            sc = 0.0
        scores[ad.name] = round(sc, 3)
        if sc > best_score:
            best_score = sc
            best = ad
    return best if best is not None else GenericAdapter(), scores


# ---------------------------------------------------------------------------
# Participant-level processing
# ---------------------------------------------------------------------------


def participant_files(source: Path) -> List[Path]:
    """All log files for a participant (the file itself, or every file in the folder)."""
    if source.is_file():
        return [source]
    files: List[Path] = []
    for pattern in ("*.jsonl", "*.ndjson", "*.json", "*.log"):
        files.extend(sorted(source.rglob(pattern)))
    # de-dup while preserving order, and skip macOS noise
    seen = set()
    out = []
    for f in files:
        if f.name == ".DS_Store" or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def process_participant(source: Path, raw_root: Path, opts: Options) -> Dict[str, Any]:
    """Clean every file for one participant and return the participant document."""
    pid = source.stem if source.is_file() else source.name
    files = participant_files(source)
    all_sessions: List[Session] = []
    formats: Dict[str, str] = {}
    detect_scores: Dict[str, Dict[str, float]] = {}
    parse_errors: List[str] = []

    for f in files:
        rel = str(f.relative_to(raw_root))
        try:
            sample = sample_records(f)
        except Exception as exc:  # never let one unreadable file abort the participant
            parse_errors.append("%s: unreadable (%s)" % (rel, exc))
            formats[rel] = "error"
            continue
        if not sample:
            parse_errors.append("%s: no readable JSON records" % rel)
            formats[rel] = "empty"
            continue
        adapter, scores = choose_adapter(sample)
        formats[rel] = adapter.name
        detect_scores[rel] = scores
        try:
            sessions = adapter.parse(obj_stream(f, parse_errors), rel, opts)
        except Exception as exc:  # never let one file sink the whole participant
            parse_errors.append("%s: adapter %s failed: %s" % (rel, adapter.name, exc))
            sessions = []
        all_sessions.extend(sessions)

    if opts.merge_turns:
        for s in all_sessions:
            s.messages = merge_consecutive(s.messages)

    all_sessions.sort(key=lambda s: (s.started_at or "", s.source_file))

    models: List[str] = []
    providers: List[str] = []
    user_msgs = assistant_msgs = 0
    for s in all_sessions:
        for m in s.models:
            if m not in models:
                models.append(m)
        if s.provider and s.provider not in providers:
            providers.append(s.provider)
        for msg in s.messages:
            if msg.role == "user":
                user_msgs += 1
            else:
                assistant_msgs += 1

    return {
        "participant": pid,
        "generated_at": _now_iso(),
        "source": {
            "path": str(source.relative_to(raw_root.parent)),
            "kind": "file" if source.is_file() else "folder",
            "files": [str(f.relative_to(raw_root)) for f in files],
        },
        "formats_detected": formats,
        "detection_scores": detect_scores,
        "providers": providers,
        "models": models,
        "stats": {
            "sessions": len(all_sessions),
            "messages": user_msgs + assistant_msgs,
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "files_processed": len(files),
            "parse_errors": len(parse_errors),
        },
        "parse_errors": parse_errors,
        "sessions": [s.to_dict() for s in all_sessions],
    }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def write_markdown(doc: Dict[str, Any], path: Path) -> None:
    """A human-readable transcript next to the JSON, for quick reading."""
    lines: List[str] = []
    lines.append("# Participant %s — clean transcript" % doc["participant"])
    lines.append("")
    st = doc["stats"]
    lines.append(
        "*%d sessions · %d user prompts · %d assistant answers · models: %s*"
        % (st["sessions"], st["user_messages"], st["assistant_messages"],
           ", ".join(doc["models"]) or "unknown")
    )
    lines.append("")
    for i, sess in enumerate(doc["sessions"], 1):
        lines.append("---")
        lines.append("")
        head = "## Session %d" % i
        if sess.get("session_id"):
            head += " · `%s`" % sess["session_id"]
        lines.append(head)
        meta_bits = []
        if sess.get("models"):
            meta_bits.append("model: %s" % ", ".join(sess["models"]))
        if sess.get("provider"):
            meta_bits.append("provider: %s" % sess["provider"])
        if sess.get("project"):
            meta_bits.append("project: %s" % sess["project"])
        if sess.get("started_at"):
            meta_bits.append("started: %s" % sess["started_at"])
        meta_bits.append("format: %s" % sess.get("source_format", "?"))
        if meta_bits:
            lines.append("_" + " · ".join(meta_bits) + "_")
        lines.append("")
        for msg in sess["messages"]:
            who = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
            if msg["role"] == "assistant" and msg.get("model"):
                who += " (%s)" % msg["model"]
            lines.append("**%s:**" % who)
            lines.append("")
            lines.append(msg["text"])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(raw_root: Path, out_root: Path, opts: Options, make_markdown: bool = True) -> Dict[str, Any]:
    sources = []
    for child in sorted(raw_root.iterdir()):
        if child.name == ".DS_Store":
            continue
        if child.is_dir() or child.suffix.lower() in (".jsonl", ".ndjson", ".json", ".log"):
            sources.append(child)

    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at": _now_iso(), "participants": []}

    for source in sources:
        doc = process_participant(source, raw_root, opts)
        pid = doc["participant"]
        (out_root / ("%s.json" % pid)).write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if make_markdown:
            write_markdown(doc, out_root / ("%s.md" % pid))
        summary["participants"].append(
            {
                "participant": pid,
                "formats": sorted(set(doc["formats_detected"].values())),
                "stats": doc["stats"],
                "models": doc["models"],
                "providers": doc["providers"],
            }
        )
        st = doc["stats"]
        print(
            "  %-6s  %-22s  sessions=%-3d  user=%-4d  assistant=%-4d  errors=%d"
            % (
                pid,
                "+".join(sorted(set(doc["formats_detected"].values()))),
                st["sessions"],
                st["user_messages"],
                st["assistant_messages"],
                st["parse_errors"],
            )
        )

    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--raw", type=Path, default=here / "raw-logs", help="input folder (default: ./raw-logs)")
    parser.add_argument("--out", type=Path, default=here / "clean-logs", help="output folder (default: ./clean-logs)")
    parser.add_argument("--include-sidechains", action="store_true", help="keep sub-agent (Task) transcripts too")
    parser.add_argument("--no-merge-turns", action="store_true", help="keep every assistant micro-turn separate instead of merging into one answer")
    parser.add_argument("--no-markdown", action="store_true", help="write JSON only, skip .md transcripts")
    args = parser.parse_args(argv)

    if not args.raw.exists():
        print("error: raw folder not found: %s" % args.raw, file=sys.stderr)
        return 2

    print("Cleaning logs:  %s  ->  %s" % (args.raw, args.out))
    opts = Options(include_sidechains=args.include_sidechains, merge_turns=not args.no_merge_turns)
    summary = run(args.raw, args.out, opts, make_markdown=not args.no_markdown)
    print("Done. %d participant(s) written to %s" % (len(summary["participants"]), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
