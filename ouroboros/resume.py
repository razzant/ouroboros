"""Version-portable conversation-resume.

The Ouroboros analog of Claude Code's ``--resume`` / single_conversation. A task that
carries ``resume_from_task_id`` continues the SAME warm, prompt-cacheable conversation of a
prior completed task, instead of starting cold. Two seams:

  * ``capture(env, task_id, messages, text)`` — opt-in via ``OUROBOROS_RESUME_CAPTURE``,
    persists the final conversation (with the closing assistant turn appended) to
    ``<OUROBOROS_DATA_DIR>/state/resume/<task_id>.json``.
  * ``load_resume_turns(env, task)`` — reads ``task["resume_from_task_id"]``, guards it,
    loads the captured turns, drops the prior system turn, sanitizes each turn, and marks a
    cache breakpoint at the end; returns ``[]`` when absent/invalid.

This module is deliberately self-contained (json, os, pathlib, logging only) so it ports
verbatim across Ouroboros versions — the core only needs 3 thin hooks calling into it.

Invariants (extracted VERBATIM from feat/conversation-resume, do not change):
  * The sanitize blocklist drops provider-private reasoning/cache metadata
    (reasoning_details, reasoning_content, reasoning, response_id, signature, cache_control)
    but PRESERVES structural keys tool_calls/tool_call_id/name (H1 — dropping tool_calls
    causes provider 400s on replay).
  * The path-traversal guard rejects resume_from_task_id containing "/", "\\", a leading
    ".", or ".." and enforces containment under the resume dir (H2).
  * One ephemeral cache_control breakpoint at the end of the replayed turns.
  * Uses OUROBOROS_DATA_DIR (server-global) NOT env.drive_path for both capture and load.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict, List

log = logging.getLogger(__name__)

_RESUME_DROP_TOP = ("reasoning_details", "reasoning_content", "reasoning", "response_id",
                    "signature", "cache_control")


def sanitize_turn(m: Dict[str, Any]) -> Dict[str, Any]:
    """Resume: PRESERVE the turn's structural keys (role, content, tool_calls, tool_call_id, name — the
    engine's native OpenAI-style format) so tool_call/tool_result pairing stays valid on replay, but DROP
    provider-private reasoning/thinking + cache metadata, whose signatures do not survive replay in a fresh
    API call (mirrors the live-path _strip_openrouter_roundtrip_metadata). A completed task's turns are
    already paired, so keeping tool_calls/tool_call_id yields a valid conversation."""
    out = {k: v for k, v in m.items() if k not in _RESUME_DROP_TOP}
    out["role"] = m.get("role") or "user"
    content = m.get("content")
    if isinstance(content, list):
        kept: List[Dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                kept.append(b)
                continue
            if b.get("type") in ("thinking", "redacted_thinking", "reasoning"):
                continue
            kept.append({k: v for k, v in b.items() if k not in ("signature", "reasoning", "cache_control")})
        out["content"] = kept
    return out


def mark_cache_breakpoint(m: Dict[str, Any]) -> None:
    """Place one ephemeral cache_control breakpoint at the END of the replayed prefix so the seeded
    conversation can hit the prompt cache on the next submission (within the provider TTL)."""
    content = m.get("content")
    if isinstance(content, str):
        m["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list):
        for b in reversed(content):
            if isinstance(b, dict) and b.get("type") in ("text", "tool_result"):
                b["cache_control"] = {"type": "ephemeral"}
                break


def load_resume_turns(env: Any, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Conversation-resume — the Ouroboros analog of Claude Code's --resume / single_conversation.
    If the task carries ``resume_from_task_id``, load that prior task's captured conversation turns
    (written by capture() when OUROBOROS_RESUME_CAPTURE is set) and return them so the new task
    CONTINUES the same warm, prompt-cacheable conversation. The prior system message is dropped (the
    system prompt is rebuilt fresh each task with live drive state); reasoning blocks are stripped.
    Returns [] when not set or the capture file is missing."""
    rid = str(task.get("resume_from_task_id") or "").strip()
    if not rid:
        return []
    # Path-traversal guard: resume_from_task_id is attacker-reachable via the unauthenticated /api/tasks
    # body and is interpolated into a filesystem path, so reject anything that could escape the resume dir
    # (the primary task_id is similarly gated by validate_task_id at the gateway).
    if "/" in rid or "\\" in rid or rid.startswith(".") or ".." in rid:
        log.warning("resume: rejected unsafe resume_from_task_id %r (path-traversal guard)", rid)
        return []
    try:
        # Server-global location (shared across a server's tasks; per-task child drives are not shared).
        root = os.environ.get("OUROBOROS_DATA_DIR") or str(getattr(env, "drive_root", "") or "")
        resume_dir = (pathlib.Path(root) / "state" / "resume").resolve()
        path = (resume_dir / (rid + ".json")).resolve()
        if not str(path).startswith(str(resume_dir) + os.sep):  # containment, belt-and-suspenders
            return []
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("messages") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for m in raw:
            if isinstance(m, dict) and m.get("role") != "system":
                out.append(sanitize_turn(m))
        if out:
            mark_cache_breakpoint(out[-1])
        return out
    except Exception:
        log.debug("resume: failed to load prior conversation for %s", rid, exc_info=True)
        return []


def capture(env: Any, task_id: str, messages: List[Dict[str, Any]], text: str) -> None:
    """Conversation-resume capture (opt-in via OUROBOROS_RESUME_CAPTURE): persist the final
    conversation so a later task can continue it via resume_from_task_id — the Ouroboros
    analog of Claude Code's --resume / single_conversation. No-op unless the env var is set."""
    if not os.environ.get("OUROBOROS_RESUME_CAPTURE"):
        return
    try:
        _tid = str(task_id or "")
        if not _tid:
            return
        # Server-global location (shared across this server's tasks).
        _root = os.environ.get("OUROBOROS_DATA_DIR") or str(getattr(env, "drive_root", "") or "")
        _rdir = os.path.join(_root, "state", "resume")
        os.makedirs(_rdir, exist_ok=True)
        # run_llm_loop returns the final assistant text but does NOT append that closing
        # turn to `messages` (only intermediate tool-using turns are appended). Append it
        # so the resumed conversation contains the agent's OWN actions, not just the prompts.
        _msgs = list(messages)
        if isinstance(text, str) and text.strip() and (
            not _msgs or not (isinstance(_msgs[-1], dict) and _msgs[-1].get("role") == "assistant")
        ):
            _msgs.append({"role": "assistant", "content": text})
        with open(os.path.join(_rdir, _tid + ".json"), "w", encoding="utf-8") as _fh:
            _fh.write(json.dumps({"messages": _msgs}, ensure_ascii=False))
    except Exception:
        log.debug("resume: failed to capture conversation", exc_info=True)
