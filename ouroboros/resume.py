"""Version-portable conversation-resume.

The Ouroboros analog of Claude Code's ``--resume`` / single_conversation. A task that
carries ``resume_from_task_id`` continues the SAME warm, prompt-cacheable conversation of a
prior completed task, instead of starting cold. Two seams:

  * ``capture(env, task_id, messages, text)`` — opt-in via ``OUROBOROS_RESUME_CAPTURE``,
    persists the final conversation (with the closing assistant turn appended) to
    ``<OUROBOROS_DATA_DIR>/state/resume/<task_id>.json``.
  * ``load_resume_turns(env, task)`` — SPLICE mode (legacy, explicit opt-in): reads ``task["resume_from_task_id"]``,
    guards it, loads the captured turns, drops the prior system turn, sanitizes each turn, and
    marks a cache breakpoint at the end; returns ``[]`` when absent/invalid.
  * ``load_continuation(env, task)`` — CONTINUATION mode (the DEFAULT for resumed tasks):
    the true CC --resume analog — returns the stored conversation WITH its original system message
    verbatim (byte-stable prefix ⇒ prompt-cache hits, no fresh-state re-framing); the caller
    replaces its fresh message list and appends only the new user turn (append-only).
  * ``note_final_msg(messages, msg)`` — loop-side capture-fidelity hook: appends the FULL provider
    message for the no-tool final turn (reasoning/response_id preserved) when capture is armed.

This module is deliberately self-contained (json, os, pathlib, logging only) so it ports
verbatim across Ouroboros versions — the core only needs 4 thin hooks calling into it.

Invariants:
  * Sanitize mirrors the LIVE-path replay policy (llm.py): PRESERVE reasoning continuity —
    reasoning/reasoning_details and thinking blocks WITH their signatures are replayed
    verbatim on resume, exactly as the in-task loop replays them. Anthropic thinking-block
    signatures are live-probe-verified portable across OpenRouter providers
    (llm.py _reasoning_signature_portable_across_or_providers), and the llm layer's
    reactive 400 strip-and-retry is the safety net for any family that rejects a replayed
    signature. Only ``cache_control`` is dropped (stale breakpoints from the prior task
    would fight the fresh breakpoint placed below; max 4 per request). Structural keys
    tool_calls/tool_call_id/name are preserved (dropping tool_calls 400s on replay).
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

_RESUME_DROP_TOP = ("cache_control",)


def sanitize_turn(m: Dict[str, Any]) -> Dict[str, Any]:
    """Resume: replay the turn with LIVE-path fidelity. Structural keys (role, content,
    tool_calls, tool_call_id, name) AND reasoning continuity (assistant-level reasoning/
    reasoning_details, thinking/redacted_thinking content blocks WITH signatures) are
    preserved verbatim — same-model replay accepts them, cross-provider portability is
    live-probe-verified for Anthropic/Gemini/OpenAI families (llm.py), and the llm layer's
    reactive 400 strip-and-retry covers any family that rejects a replayed signature.
    Only ``cache_control`` is dropped: stale breakpoints from the captured task would
    fight the fresh end-of-replay breakpoint (max 4 per request)."""
    out = {k: v for k, v in m.items() if k not in _RESUME_DROP_TOP}
    out["role"] = m.get("role") or "user"
    content = m.get("content")
    if isinstance(content, list):
        kept: List[Dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                kept.append(b)
                continue
            kept.append({k: v for k, v in b.items() if k != "cache_control"})
        out["content"] = kept
    return out


def mark_cache_breakpoint(m: Dict[str, Any]) -> bool:
    """Place one ephemeral cache_control breakpoint at the END of the replayed prefix so the seeded
    conversation can hit the prompt cache on the next submission (within the provider TTL).
    Refuses empty/whitespace text (Anthropic rejects empty text blocks; a breakpoint on one is
    silently dropped by the send path — the caller should walk back to an earlier turn).
    Returns True when a breakpoint was placed."""
    content = m.get("content")
    if isinstance(content, str):
        if not content.strip():
            return False
        m["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return True
    if isinstance(content, list):
        for b in reversed(content):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result" or (
                b.get("type") == "text" and str(b.get("text") or "").strip()
            ):
                b["cache_control"] = {"type": "ephemeral"}
                return True
    return False


def _trim_chain(out: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CC-compaction analog for the resumed chain (deterministic, no LLM): when the stored
    non-system turns exceed OUROBOROS_RESUME_MAX_CHARS (default 600K chars ≈ 150K tokens),
    drop the OLDEST turns down to ~70% of the cap and insert ONE explicit omission note
    after the system message. Without this the chain grows unbounded and the provider 400s
    near the context window with no fallback (the soft cap is a documented no-op). Trimming
    is deterministic and stable between triggers, so the prefix stays cacheable in the
    stretches between trims (CC's own compaction pays the same cache invalidation)."""
    try:
        cap = int(os.environ.get("OUROBOROS_RESUME_MAX_CHARS", "600000"))
    except ValueError:
        cap = 600000
    if cap <= 0 or len(out) < 2:
        return out
    sizes = [len(json.dumps(m, ensure_ascii=False, default=str)) for m in out[1:]]
    total = sum(sizes)
    if total <= cap:
        return out
    target = int(cap * 0.7)
    kept = list(out[1:])
    dropped = 0
    while kept and total > target:
        total -= sizes[dropped]
        kept.pop(0)
        dropped += 1
    while kept and kept[0].get("role") == "tool":  # never start with an orphan tool reply
        kept.pop(0)
        dropped += 1
    log.warning("resume: chain over %d chars — dropped the %d oldest turns (kept %d)",
                cap, dropped, len(kept))
    note = {"role": "user",
            "content": f"[resume: the {dropped} earliest turns of this conversation were "
                       f"omitted to fit the context window]"}
    return [out[0], note] + kept


def _mark_prefix_breakpoint(turns: List[Dict[str, Any]]) -> None:
    """Walk backwards over the replayed turns and place ONE end-of-prefix breakpoint on the
    nearest markable (non-system, non-empty) turn."""
    for m in reversed(turns):
        if m.get("role") == "system":
            break  # never demote the system message's own breakpoints
        if mark_cache_breakpoint(m):
            return


_KNOWN_RESUME_MODES = ("", "splice", "continuation")


def _resume_mode(task: Dict[str, Any]) -> str:
    """Normalized resume_mode. DEFAULT (empty) = CONTINUATION — the true CC --resume analog
    measurably outperforms splice (6q A/B: 0.611 vs 0.144), so it is what plain resume means;
    'splice' is the explicit legacy opt-in. Unknown values warn LOUDLY (a typo must not
    silently change the mode) and fall back to the default."""
    mode = str(task.get("resume_mode") or "").strip().lower()
    if mode not in _KNOWN_RESUME_MODES:
        log.warning("resume: unknown resume_mode %r (known: %s) — using the default (continuation)",
                    mode, "/".join(m or "''" for m in _KNOWN_RESUME_MODES))
        return "continuation"
    return mode or "continuation"


def _load_capture_messages(env: Any, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Guarded shared loader: resolve ``task["resume_from_task_id"]`` to its capture file and return
    the raw stored messages list, or [] when absent/invalid. Carries the path-traversal guard —
    resume_from_task_id is attacker-reachable via the unauthenticated /api/tasks body and is
    interpolated into a filesystem path (the primary task_id is gated by validate_task_id)."""
    rid = str(task.get("resume_from_task_id") or "").strip()
    if not rid:
        return []
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
            # LOUD: a requested-but-missing capture silently cold-restarts the whole chain (the
            # prior task died before capture ran) — the one failure shape callers can't see.
            log.warning("resume: capture for %s NOT FOUND — conversation chain restarts cold", rid)
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("messages") if isinstance(data, dict) else data
        return raw if isinstance(raw, list) else []
    except Exception:
        log.debug("resume: failed to load prior conversation for %s", rid, exc_info=True)
        return []


def load_resume_turns(env: Any, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Conversation-resume, SPLICE mode (LEGACY, explicit ``resume_mode=splice`` only) — the prior
    task's captured turns are spliced between the FRESH system message and the new user turn (the
    system prompt is rebuilt each task with live drive state; the stored system turn is dropped).
    Returns [] unless splice is explicitly requested, or when resume_from_task_id is
    absent/invalid / the capture is missing. Continuation is the DEFAULT (see load_continuation)."""
    if _resume_mode(task) != "splice":
        return []
    out: List[Dict[str, Any]] = []
    for m in _load_capture_messages(env, task):
        if isinstance(m, dict) and m.get("role") != "system":
            out.append(sanitize_turn(m))
    _mark_prefix_breakpoint(out)
    return out


def load_continuation(env: Any, task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Conversation-resume, CONTINUATION mode (the DEFAULT; also explicit
    ``resume_mode=continuation``) — the
    true Claude Code --resume analog: return the FULL stored conversation with the ORIGINAL system
    message VERBATIM (byte-stable prefix ⇒ cross-task prompt-cache hits; no fresh-state re-framing,
    no Recent-chat self-echo), followed by the sanitized turns. The caller replaces its fresh
    message list with this and appends only the new user turn — append-only, CC's
    single_conversation shape. The system turn keeps its own cache_control breakpoints; stale
    per-turn breakpoints from the prior task are dropped and ONE fresh breakpoint marks the end of
    the replayed prefix. Returns [] (caller falls back to the fresh build) when the mode is not
    requested, the capture is missing, or the capture lacks its leading system turn."""
    if _resume_mode(task) != "continuation":
        return []
    raw = _load_capture_messages(env, task)
    if not raw:
        return []
    out: List[Dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            out.append(m)  # VERBATIM — preserve the original cache_control breakpoints
        else:
            out.append(sanitize_turn(m))
    if not out or out[0].get("role") != "system":
        log.warning("resume: continuation capture for %s lacks a leading system turn; "
                    "falling back to the fresh build", task.get("resume_from_task_id"))
        return []
    out = _trim_chain(out)
    _mark_prefix_breakpoint(out)
    log.info("resume: CONTINUATION of %s — %d stored turns, system verbatim",
             task.get("resume_from_task_id"), len(out) - 1)
    return out


def note_final_msg(messages: List[Dict[str, Any]], msg: Any) -> None:
    """Capture-fidelity hook for the loop's no-tool FINAL turn: run_llm_loop returns the final
    answer text WITHOUT appending the provider message, so capture() reconstructs a bare
    {role, content} turn and loses reasoning/reasoning_details/response_id (measured on a
    per-action bench run: 360/364 captured assistant turns bare while 47/58 provider finals
    carried reasoning). When capture is armed, append the FULL provider msg so the persisted
    conversation keeps the fidelity the live tool-round path already has (loop.py's
    ``messages.append(dict(msg))``). No-op unless OUROBOROS_RESUME_CAPTURE is set — normal runs
    are untouched."""
    if not os.environ.get("OUROBOROS_RESUME_CAPTURE"):
        return
    try:
        if not isinstance(msg, dict) or not isinstance(messages, list):
            return
        content = msg.get("content")
        has_content = (isinstance(content, str) and content.strip()) or (
            isinstance(content, list) and content)
        if not has_content:
            # An empty final would be captured as content:"" — the end-of-prefix breakpoint
            # then lands on an empty text block (dropped by the send path; Anthropic 400s on
            # empty text). Skip; capture()'s bare-text append covers the placeholder path.
            return
        # Allowlist (not a None-denylist): provider echo junk like tool_calls:[] or
        # annotations:[] must never enter the persisted capture — it 400s on strict lanes.
        keep = ("role", "content", "tool_calls", "tool_call_id", "name",
                "reasoning", "reasoning_details", "response_id")
        out = {k: msg[k] for k in keep if msg.get(k)}
        out["role"] = out.get("role") or "assistant"
        out["content"] = content
        messages.append(out)
    except Exception:
        log.debug("resume: note_final_msg failed", exc_info=True)


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
        # Dangling-tail guard: a task killed between the assistant tool_calls turn and its
        # tool replies would persist a tool_use with no tool_result — the replay then 400s
        # FOREVER (every later task inherits it). Synthesize aborted-tool replies instead.
        if _msgs and isinstance(_msgs[-1], dict) and _msgs[-1].get("role") == "assistant":
            for tc in (_msgs[-1].get("tool_calls") or []):
                tc_id = str((tc or {}).get("id") or "") if isinstance(tc, dict) else ""
                if tc_id:
                    _msgs.append({"role": "tool", "tool_call_id": tc_id,
                                  "content": "TOOL_ERROR: task ended before this tool ran"})
        if isinstance(text, str) and text.strip() and (
            not _msgs or not (isinstance(_msgs[-1], dict) and _msgs[-1].get("role") == "assistant")
        ):
            _msgs.append({"role": "assistant", "content": text})
        with open(os.path.join(_rdir, _tid + ".json"), "w", encoding="utf-8") as _fh:
            _fh.write(json.dumps({"messages": _msgs}, ensure_ascii=False))
    except Exception:
        log.debug("resume: failed to capture conversation", exc_info=True)
