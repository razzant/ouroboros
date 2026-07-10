"""LLM-loop tool execution: dispatch, timeouts, truncation, live logs."""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import logging

from ouroboros.config import get_finalization_grace_sec, load_settings
from ouroboros.deadline_utils import deadline_remaining_sec
from ouroboros.observability import new_call_id, persist_call
from ouroboros.tool_capabilities import (
    READ_ONLY_PARALLEL_TOOLS,
    PARALLEL_SAFE_ENQUEUE_TOOLS,
    FOREGROUND_MUTATIVE_TOOLS,
    REVIEWED_MUTATIVE_TOOLS,
    STATEFUL_BROWSER_TOOLS,
    TOOL_RESULT_LIMITS as _TOOL_RESULT_LIMITS,
    DEFAULT_TOOL_RESULT_LIMIT as _DEFAULT_TOOL_RESULT_LIMIT,
    UNTRUNCATED_TOOL_RESULTS as _UNTRUNCATED_TOOL_RESULTS,
    UNTRUNCATED_REPO_READ_PATHS as _UNTRUNCATED_REPO_READ_PATHS,
)
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import (
    append_jsonl,
    emit_log_event,
    sanitize_tool_args_for_log,
    sanitize_tool_result_for_log,
    truncate_for_log,
    utc_now_iso,
)

log = logging.getLogger(__name__)

_FAILURE_PREFIXES = (
    "⚠️ TOOL_",
    "⚠️ SHELL_",
    "⚠️ RUN_SCRIPT_",
    "⚠️ CLAUDE_CODE_",
    "⚠️ VLM_",
    "⚠️ LIGHT_MODE_",
    "⚠️ WORKSPACE_",
    "⚠️ ELEVATION_",
    "⚠️ SKILL_STATE_",
    "⚠️ SKILL_REDIRECT_",
    "⚠️ SKILL_PAYLOAD_ARG_",
    "⚠️ DATA_WRITE_",
    "⚠️ DATA_READ_BLOCKED",
    "⚠️ DATA_LIST_BLOCKED",
    "⚠️ WRITE_FILE_",
    "⚠️ EDIT_TEXT_",
    "⚠️ ARTIFACT_OUTPUT_ERROR",
    "⚠️ CORE_PROTECTION_BLOCKED",
    "⚠️ SKILL_PAYLOAD_CONTROL_BLOCKED",
    "⚠️ COGNITIVE_TOOL_REQUIRED",
    "⚠️ ROOT_REQUIRED_USER_FILES",
    "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE",
    "⚠️ RESOURCE_CONSTRAINT_BLOCKED",
    "⚠️ RESOURCE_POLICY_BLOCKED",
    "⚠️ INTEGRATE_",
)
_FAILURE_MARKERS = (
    "_BLOCKED",
    "_ERROR",
    "_FAILED",
    "_UNAVAILABLE",
    "_VIOLATION",
)
_EXIT_CODE_RE = re.compile(r"exit_code=(-?\d+)")
_SIGNAL_RE = re.compile(r"signal=([A-Z0-9_]+)")

# Reviewed mutative tools get a hard ceiling after their soft timeout.
_REVIEWED_MUTATIVE_HARD_CEILING = 1800


def _emit_live_log(tools: ToolRegistry, payload: Dict[str, Any]) -> None:
    """Emit a live log through the registry context queue. Lineage (parent/root task
    ids) is merged in so a SUBAGENT's live log routes to its root project thread
    (C4.4) — only the root is bound, so without lineage the child's log stays in main."""
    event_queue = getattr(getattr(tools, "_ctx", None), "event_queue", None)
    enriched = dict(payload)
    meta = _tool_task_metadata(tools)
    for key in ("parent_task_id", "root_task_id"):
        if meta.get(key) and not enriched.get(key):
            enriched[key] = meta.get(key)
    emit_log_event(
        event_queue,
        {"ts": utc_now_iso(), **enriched},
        log_label="tool live",
    )


def _tool_correlation(tools: ToolRegistry) -> Dict[str, Any]:
    ctx = getattr(tools, "_ctx", None)
    meta = getattr(ctx, "_current_llm_call_meta", {}) if ctx is not None else {}
    return dict(meta) if isinstance(meta, dict) else {}


def _tool_task_metadata(tools: ToolRegistry) -> Dict[str, Any]:
    ctx = getattr(tools, "_ctx", None)
    meta = getattr(ctx, "task_metadata", {}) if ctx is not None else {}
    data = dict(meta) if isinstance(meta, dict) else {}
    for key in ("parent_task_id", "root_task_id", "delegation_role"):
        if data.get(key):
            continue
        value = getattr(ctx, key, "") if ctx is not None else ""
        if value:
            data[key] = value
    if ctx is not None and getattr(ctx, "task_depth", None) is not None:
        data.setdefault("task_depth", getattr(ctx, "task_depth"))
    if ctx is not None and getattr(ctx, "budget_drive_root", ""):
        data.setdefault("budget_drive_root", getattr(ctx, "budget_drive_root"))
    return data


def _append_tool_log(tools: ToolRegistry, drive_logs: pathlib.Path, payload: Dict[str, Any]) -> None:
    meta = _tool_task_metadata(tools)
    for key in ("parent_task_id", "root_task_id", "delegation_role", "task_depth"):
        if meta.get(key) not in (None, ""):
            payload[key] = meta.get(key)
    append_jsonl(drive_logs / "tools.jsonl", payload)
    root = str(meta.get("budget_drive_root") or "").strip()
    if not root:
        return
    candidate = pathlib.Path(root).resolve(strict=False) / "logs" / "tools.jsonl"
    try:
        if candidate.resolve(strict=False) == (pathlib.Path(drive_logs) / "tools.jsonl").resolve(strict=False):
            return
    except Exception:
        pass
    append_jsonl(candidate, payload)


def _with_correlation(payload: Dict[str, Any], correlation: Dict[str, Any], *, tool_call_id: str = "") -> Dict[str, Any]:
    out = dict(payload)
    for key in ("execution_id", "round_id", "llm_call_id"):
        if correlation.get(key):
            out[key] = correlation.get(key)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    return out


_PER_CALL_TIMEOUT_TOOLS = ("run_command", "run_script")
# Structural ordering margin: the outer cap sits this far above the requested
# per-call timeout so the handler's own (cleanly-messaged) subprocess timeout
# fires first, before the outer thread-kill. Not a wait duration — a race margin.
_PER_CALL_TIMEOUT_OUTER_MARGIN_SEC = 30


def _tc_args(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort parse of a tool call's JSON arguments (for timeout resolution)."""
    try:
        raw = (tc.get("function") or {}).get("arguments")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError, AttributeError):
        pass
    return {}


def _get_tool_timeout(
    tools: ToolRegistry, tool_name: str, tool_args: Optional[Dict[str, Any]] = None
) -> int:
    """Return max(settings/env timeout, per-tool minimum).

    For ``run_command``/``run_script`` a per-call ``timeout_sec`` (or its
    ``timeout`` alias) raises this OUTER tool-execution cap so the handler's own
    deadline-clamped subprocess timeout fires first — otherwise a legitimately
    long ``run_command(timeout_sec=900)`` would be cut off at the static 360s
    entry cap before the inner timeout matters. A +30s margin lets the inner
    (cleanly-messaged) timeout win over the outer thread-kill.
    """
    settings_val = 0
    try:
        settings_val = int(load_settings().get("OUROBOROS_TOOL_TIMEOUT_SEC") or 0)
    except Exception:
        pass
    if settings_val <= 0:
        env_val = os.environ.get("OUROBOROS_TOOL_TIMEOUT_SEC")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed > 0:
                    settings_val = parsed
            except ValueError:
                pass
    per_tool = tools.get_timeout(tool_name)
    base = max(settings_val, per_tool) if settings_val > 0 else per_tool
    if tool_name in _PER_CALL_TIMEOUT_TOOLS and isinstance(tool_args, dict):
        raw = tool_args.get("timeout_sec", tool_args.get("timeout"))
        try:
            override = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            override = 0
        if override > 0:
            from ouroboros.config import get_per_call_timeout_ceiling_sec

            return min(max(base, override), get_per_call_timeout_ceiling_sec()) + _PER_CALL_TIMEOUT_OUTER_MARGIN_SEC
    return _deadline_clamped_timeout(tools, tool_name, base)


# Network/long-wait tools whose OUTER timeout is clamped to the remaining task
# deadline (v6.54.3, 1.4A): one in-flight web_search with a 540s ceiling could
# otherwise swallow the whole remaining budget — finalize_now drains only at
# round boundaries, so the task slid past its deadline mid-tool (TB2.1
# gpt2-codegolf post-mortem). Local media/process tools stay unclamped: the
# per-call/run_command machinery and the deadline milestones own those.
_DEADLINE_CLAMPED_TOOLS = frozenset({
    "web_search", "browse_page", "browser_action", "youtube_transcript",
    "wait_task", "wait_tasks",
})


def _deadline_clamped_timeout(tools: ToolRegistry, tool_name: str, base_timeout: int) -> int:
    """min(tool timeout, remaining − finalization reserve) for network/long tools.

    Without ``deadline_at`` (remaining reads 0.0) the behavior is byte-identical.
    An already-elapsed deadline is also left unclamped — the loop's forced
    finalization owns that path. The clamp never floors PAST the reserve: inside
    or near the finalization window the tool gets a near-immediate (1s) timeout —
    a fast, cleanly-messaged failure — rather than a grace-period-eating floor
    (review round 1: a 30s floor let one web call consume the reserve the clamp
    exists to protect)."""
    if tool_name not in _DEADLINE_CLAMPED_TOOLS:
        return base_timeout
    try:
        remaining = deadline_remaining_sec(getattr(tools, "_ctx", None))
    except Exception:
        return base_timeout
    if remaining <= 0:
        return base_timeout
    # v6.55.0: the plain finalization GRACE emit-window (task_pacing SSOT) — the
    # tool must return before the deadline leaves no time to emit a final answer.
    # NOT the pct reserve (an acceptance-review gate concept): clamping a tool out
    # of the last pct-of-total would strand long tasks with idle time.
    try:
        from ouroboros.task_pacing import effective_finalization_reserve_sec

        reserve = effective_finalization_reserve_sec(getattr(tools, "_ctx", None))
    except Exception:
        reserve = float(get_finalization_grace_sec())
    window = remaining - reserve
    if window >= base_timeout:
        return base_timeout
    return int(max(1.0, min(float(base_timeout), window)))


def _path_is_cognitive_artifact(tool_name: str, tool_args: Optional[Dict[str, Any]]) -> bool:
    """Return whether a read target must stay whole."""
    if not tool_args:
        return False

    raw_path = str(tool_args.get("path") or "").strip()
    if not raw_path:
        return False

    normalized = raw_path.replace("\\", "/").lstrip("./")

    if tool_name == "read_file" and str((tool_args or {}).get("root") or "active_workspace") == "runtime_data":
        return normalized.startswith("memory/") and "/_backup/" not in normalized

    if tool_name == "read_file":
        return normalized.startswith("prompts/") or normalized in _UNTRUNCATED_REPO_READ_PATHS

    return False


def _should_skip_tool_result_truncation(
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
) -> bool:
    """Canonical/cognitive reads must remain whole."""
    return tool_name in _UNTRUNCATED_TOOL_RESULTS or _path_is_cognitive_artifact(tool_name, tool_args)


def _truncate_tool_result(
    result: Any,
    tool_name: str = "",
    tool_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Cap tool result unless it is an untruncated artifact."""
    limit = _TOOL_RESULT_LIMITS.get(tool_name, _DEFAULT_TOOL_RESULT_LIMIT)
    s = str(result)
    if _should_skip_tool_result_truncation(tool_name, tool_args):
        return s
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated from {len(s)} chars, limit={limit})"


def _is_tool_execution_failure(tool_ok: bool, result: Any) -> bool:
    """Treat only executor/runtime failures as UI tool failures."""
    if not tool_ok:
        return True
    text = str(result or "")
    if text.startswith("⚠️ SHELL_REGEX_AUTO_CORRECTED"):
        remainder = text.split("\n", 1)[1] if "\n" in text else ""
        if any(prefix in remainder for prefix in _FAILURE_PREFIXES):
            return True
        return False
    if text.startswith("⚠️ REVIEW_BLOCKED") or text.startswith("⚠️ GIT_ERROR"):
        return False
    if text.startswith(_FAILURE_PREFIXES):
        return True
    first_line = text.splitlines()[0] if text.startswith("⚠️ ") else ""
    return bool(first_line and any(marker in first_line for marker in _FAILURE_MARKERS))


def _extract_result_metadata(fn_name: str, result: Any, is_error: bool) -> Dict[str, Any]:
    """Extract structured outcome facts for summaries and reflections."""
    text = str(result or "")
    status = "error" if is_error else "ok"
    if text.startswith("⚠️ TOOL_TIMEOUT"):
        status = "timeout"
    elif text.startswith("⚠️ SHELL_REGEX_AUTO_CORRECTED") and "⚠️ ARTIFACT_OUTPUT_ERROR" in text:
        status = "artifact_output_error"
    elif text.startswith("⚠️ SHELL_REGEX_AUTO_CORRECTED") and "⚠️ SHELL_EXIT_ERROR" not in text:
        status = "ok_autocorrected"
    elif text.startswith("⚠️ SHELL_EXIT_ERROR"):
        status = "non_zero_exit"
    elif text.startswith("⚠️ SHELL_CWD_BLOCKED"):
        status = "cwd_blocked"
    elif text.startswith("⚠️ SHELL_"):
        status = "shell_error"
    elif text.startswith("⚠️ RUN_SCRIPT_BLOCKED"):
        status = "run_script_blocked"
    elif text.startswith("⚠️ CLAUDE_CODE_TIMEOUT"):
        status = "timeout"
    elif text.startswith("⚠️ CLAUDE_CODE_INSTALL_ERROR"):
        status = "install_error"
    elif text.startswith("⚠️ CLAUDE_CODE_UNAVAILABLE"):
        status = "unavailable"
    elif text.startswith("⚠️ CLAUDE_CODE_"):
        status = "claude_code_error"
    elif text.startswith("⚠️ VLM_"):
        status = "vlm_error"
    elif text.startswith("⚠️ CORE_PROTECTION_BLOCKED"):
        status = "protected_blocked"
    elif text.startswith("⚠️ SKILL_PAYLOAD_CONTROL_BLOCKED"):
        status = "skill_payload_control_blocked"
    elif text.startswith("⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED") or text.startswith("⚠️ LIGHT_MODE_BLOCKED"):
        status = "light_mode_blocked"
    elif text.startswith("⚠️ COGNITIVE_TOOL_REQUIRED"):
        status = "cognitive_tool_required"
    elif text.startswith("⚠️ ROOT_REQUIRED_USER_FILES"):
        status = "root_required_user_files"
    elif text.startswith("⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE"):
        status = "root_required_active_workspace"
    elif text.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED"):
        status = "resource_constraint_blocked"
    elif text.startswith("⚠️ RESOURCE_POLICY_BLOCKED"):
        status = "resource_policy_blocked"
    elif text.startswith("⚠️ INTEGRATE_"):
        status = "integration_blocked"
    elif text.startswith("⚠️ WORKSPACE_"):
        status = "workspace_blocked"
    elif text.startswith("⚠️ ELEVATION_"):
        status = "elevation_blocked"
    elif text.startswith("⚠️ SKILL_STATE_"):
        status = "skill_state_blocked"
    elif text.startswith("⚠️ SKILL_REDIRECT_") or text.startswith("⚠️ SKILL_PAYLOAD_ARG_"):
        status = "skill_payload_blocked"
    elif text.startswith("⚠️ DATA_WRITE_") or text.startswith("⚠️ DATA_READ_BLOCKED") or text.startswith("⚠️ DATA_LIST_BLOCKED"):
        status = "data_blocked"
    elif text.startswith("⚠️ WRITE_FILE_"):
        status = "write_file_blocked"
    elif text.startswith("⚠️ EDIT_TEXT_"):
        status = "edit_text_blocked"
    elif text.startswith("⚠️ ARTIFACT_OUTPUT_ERROR"):
        status = "artifact_output_error"
    elif text.startswith("⚠️ SAFETY_VIOLATION") or text.startswith("⚠️ CRITICAL SAFETY_VIOLATION"):
        status = "safety_violation"
    elif text.startswith("⚠️ HEAL_MODE_BLOCKED"):
        status = "heal_mode_blocked"
    elif text.startswith("⚠️ GIT_VIA_SHELL_BLOCKED"):
        status = "git_via_shell_blocked"
    elif text.startswith("⚠️ ") and "_BLOCKED" in text.splitlines()[0]:
        status = "blocked"
    elif text.startswith("⚠️ ") and "_VIOLATION" in text.splitlines()[0]:
        status = "violation"
    elif text.startswith("⚠️ ") and any(marker in text.splitlines()[0] for marker in ("_ERROR", "_FAILED", "_UNAVAILABLE")):
        status = "error"

    meta: Dict[str, Any] = {"status": status}
    # Structured deliverable signal captured from the FULL result (before the trace
    # preview is truncated to 700 chars) so effect detection never misses a
    # late ARTIFACT_OUTPUTS marker (e.g. a stopped service after a long log tail).
    if not is_error and "ARTIFACT_OUTPUTS" in text:
        meta["artifact_registered"] = True
    # Same full-result capture for the swarm force-plan gate: the review
    # aggregate marker sits at the END of a long plan_task result, far past the
    # 700-char trace preview the gate used to substring-match against.
    if fn_name == "plan_task" and not is_error and "## Plan Review Results" in text and "AGGREGATE:" in text:
        meta["plan_review_aggregate"] = True
    exit_match = _EXIT_CODE_RE.search(text)
    if exit_match:
        try:
            meta["exit_code"] = int(exit_match.group(1))
        except ValueError:
            pass
    signal_match = _SIGNAL_RE.search(text)
    if signal_match:
        meta["signal"] = signal_match.group(1)
    if fn_name == "run_command" and not is_error and meta.get("exit_code") == 0:
        if status == "ok_autocorrected":
            meta["status"] = "ok_autocorrected"
        else:
            meta["status"] = "ok"
    return meta


def _execute_single_tool(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    task_id: str = "",
) -> Dict[str, Any]:
    """
    Execute a single tool call and return all needed info.

    Returns dict with: tool_call_id, fn_name, result, is_error, args_for_log, is_code_tool
    """
    requested_fn_name = tc["function"]["name"]
    fn_name = str(requested_fn_name or "").strip()
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS
    correlation = _tool_correlation(tools)

    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, ValueError) as e:
        result = f"⚠️ TOOL_ARG_ERROR: Could not parse arguments for '{requested_fn_name}': {e}"
        trace_ref = {}
        try:
            trace_ref = persist_call(
                pathlib.Path(drive_logs).parent,
                task_id=task_id,
                call_id=new_call_id("tool_arg_error"),
                call_type="tool_call",
                payload={
                    "tool": fn_name,
                    "tool_call_id": tool_call_id,
                    "parent_call_id": correlation.get("llm_call_id"),
                    "execution_id": correlation.get("execution_id"),
                    "round_id": correlation.get("round_id"),
                    "raw_arguments": tc.get("function", {}).get("arguments"),
                    "result": result,
                },
                manifest={
                    "execution_id": correlation.get("execution_id"),
                    "round_id": correlation.get("round_id"),
                    "parent_call_id": correlation.get("llm_call_id"),
                    "tool_call_id": tool_call_id,
                    "tool": fn_name,
                    "status": "arg_error",
                },
            )
        except Exception:
            log.debug("Failed to persist tool arg-error observability payload", exc_info=True)
        return {
            "tool_call_id": tool_call_id,
            "fn_name": fn_name,
            "result": result,
            "is_error": True,
            "tool_args": {},
            "args_for_log": {},
            "is_code_tool": is_code_tool,
            "trace_ref": trace_ref,
            "result_meta": _extract_result_metadata(fn_name, result, True),
        }

    args_for_log = sanitize_tool_args_for_log(fn_name, args if isinstance(args, dict) else {})

    tool_ok = True
    try:
        result = tools.execute(fn_name, args)
    except Exception as e:
        tool_ok = False
        safe_error = sanitize_tool_result_for_log(f"{type(e).__name__}: {e}")
        result = f"⚠️ TOOL_ERROR ({fn_name}): {safe_error}"
        append_jsonl(drive_logs / "events.jsonl", _with_correlation({
            "ts": utc_now_iso(), "type": "tool_error", "task_id": task_id,
            "tool": fn_name, "args": args_for_log, "error": safe_error,
        }, correlation, tool_call_id=tool_call_id))

    is_error = _is_tool_execution_failure(tool_ok, result)
    result_meta = _extract_result_metadata(fn_name, result, is_error)

    trace_ref = {}
    try:
        trace_ref = persist_call(
            pathlib.Path(drive_logs).parent,
            task_id=task_id,
            call_id=new_call_id(f"tool_{fn_name}"),
            call_type="tool_call",
            payload={
                "tool": fn_name,
                "tool_call_id": tool_call_id,
                "parent_call_id": correlation.get("llm_call_id"),
                "execution_id": correlation.get("execution_id"),
                "round_id": correlation.get("round_id"),
                "args": args,
                "result": result,
                "tool_ok": tool_ok,
                "semantic_ok": not is_error,
                "result_meta": result_meta,
            },
            manifest={
                "execution_id": correlation.get("execution_id"),
                "round_id": correlation.get("round_id"),
                "parent_call_id": correlation.get("llm_call_id"),
                "tool_call_id": tool_call_id,
                "tool": fn_name,
                "status": str(result_meta.get("status") or ("ok" if tool_ok else "exception")),
                "semantic_ok": not is_error,
            },
        )
    except Exception:
        log.debug("Failed to persist tool observability payload", exc_info=True)

    _append_tool_log(tools, drive_logs, _with_correlation({
        "ts": utc_now_iso(), "type": "tool_call", "tool": fn_name, "task_id": task_id,
        "args": args_for_log,
        "result_preview": sanitize_tool_result_for_log(truncate_for_log(result, 2000)),
        "is_error": is_error,
        "status": result_meta.get("status"),
        "args_ref": (trace_ref.get("manifest_ref") or {}).get("path") if trace_ref else None,
        "result_ref": trace_ref.get("manifest_ref") if trace_ref else None,
    }, correlation, tool_call_id=tool_call_id))

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": is_error,
        "tool_args": args if isinstance(args, dict) else {},
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
        "trace_ref": trace_ref,
        "result_meta": result_meta,
    }


class StatefulToolExecutor:
    """Thread-sticky executor for Playwright/greenlet stateful tools."""
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None

    def submit(self, fn, *args, **kwargs):
        """Submit work to the sticky thread."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stateful_tool")
        return self._executor.submit(fn, *args, **kwargs)

    def reset(self):
        """Reset the sticky thread after timeout/error."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def shutdown(self, wait=True, cancel_futures=False):
        """Shutdown the sticky executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None


def _make_timeout_result(
    fn_name: str,
    tool_call_id: str,
    is_code_tool: bool,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    reset_msg: str = "",
    correlation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and log a timeout result."""
    args_for_log = {}
    raw_args: Dict[str, Any] = {}
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
        raw_args = args if isinstance(args, dict) else {}
        args_for_log = sanitize_tool_args_for_log(fn_name, raw_args)
    except Exception:
        pass

    result = (
        f"⚠️ TOOL_TIMEOUT ({fn_name}): exceeded {timeout_sec}s limit. "
        f"The tool is still running in background but control is returned to you. "
        f"{reset_msg}Try a different approach or inform the user{' about the issue' if not reset_msg else ''}."
    )
    trace_ref = {}
    corr = dict(correlation or {})
    try:
        trace_ref = persist_call(
            pathlib.Path(drive_logs).parent,
            task_id=task_id,
            call_id=new_call_id(f"tool_timeout_{fn_name}"),
            call_type="tool_timeout",
            payload={
                "tool": fn_name,
                "tool_call_id": tool_call_id,
                "parent_call_id": corr.get("llm_call_id"),
                "execution_id": corr.get("execution_id"),
                "round_id": corr.get("round_id"),
                "timeout_sec": timeout_sec,
                "args": raw_args,
                "args_redacted_preview": args_for_log,
                "result": result,
            },
            manifest={
                "execution_id": corr.get("execution_id"),
                "round_id": corr.get("round_id"),
                "parent_call_id": corr.get("llm_call_id"),
                "tool_call_id": tool_call_id,
                "tool": fn_name,
                "status": "timeout",
                "timeout_sec": timeout_sec,
            },
        )
    except Exception:
        log.debug("Failed to persist tool timeout observability payload", exc_info=True)

    append_jsonl(drive_logs / "events.jsonl", _with_correlation({
        "ts": utc_now_iso(), "type": "tool_timeout",
        "task_id": task_id,
        "tool": fn_name, "args": args_for_log,
        "timeout_sec": timeout_sec,
        "result_ref": trace_ref.get("manifest_ref") if trace_ref else None,
    }, corr, tool_call_id=tool_call_id))
    append_jsonl(drive_logs / "tools.jsonl", _with_correlation({
        "ts": utc_now_iso(), "type": "tool_call", "tool": fn_name,
        "task_id": task_id,
        "args": args_for_log, "result_preview": result,
        "result_ref": trace_ref.get("manifest_ref") if trace_ref else None,
    }, corr, tool_call_id=tool_call_id))

    return {
        "tool_call_id": tool_call_id,
        "fn_name": fn_name,
        "result": result,
        "is_error": True,
        "args_for_log": args_for_log,
        "is_code_tool": is_code_tool,
        "trace_ref": trace_ref,
        "result_meta": _extract_result_metadata(fn_name, result, True),
    }


def _execute_with_timeout(
    tools: ToolRegistry,
    tc: Dict[str, Any],
    drive_logs: pathlib.Path,
    timeout_sec: int,
    task_id: str = "",
    stateful_executor: Optional[StatefulToolExecutor] = None,
) -> Dict[str, Any]:
    """Execute one tool call with timeout handling."""
    requested_fn_name = tc["function"]["name"]
    fn_name = str(requested_fn_name or "").strip()
    tool_call_id = tc["id"]
    is_code_tool = fn_name in tools.CODE_TOOLS
    use_stateful = stateful_executor and fn_name in STATEFUL_BROWSER_TOOLS
    started_at = time.perf_counter()
    correlation = _tool_correlation(tools)
    args_for_log = {}
    try:
        args = json.loads(tc["function"]["arguments"] or "{}")
        if isinstance(args, dict):
            args_for_log = sanitize_tool_args_for_log(fn_name, args)
    except Exception:
        pass
    _emit_live_log(tools, _with_correlation({
        "type": "tool_call_started",
        "task_id": task_id,
        "tool": fn_name,
        "timeout_sec": timeout_sec,
        "args": args_for_log,
    }, correlation, tool_call_id=tool_call_id))

    if use_stateful:
        future = stateful_executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
        try:
            result = future.result(timeout=timeout_sec)
            result_meta = result.get("result_meta") or {}
            _emit_live_log(tools, _with_correlation({
                "type": "tool_call_finished",
                "task_id": task_id,
                "tool": fn_name,
                "args": result.get("args_for_log", args_for_log),
                "duration_sec": round(time.perf_counter() - started_at, 3),
                "is_error": bool(result.get("is_error")),
                "status": result_meta.get("status"),
                "exit_code": result_meta.get("exit_code"),
                "signal": result_meta.get("signal"),
                "result_preview": sanitize_tool_result_for_log(
                    truncate_for_log(result.get("result", ""), 500)
                ),
            }, correlation, tool_call_id=tool_call_id))
            return result
        except (TimeoutError, concurrent.futures.TimeoutError):
            stateful_executor.reset()
            reset_msg = "Browser state has been reset. "
            timeout_result = _make_timeout_result(
                fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                timeout_sec, task_id, reset_msg, correlation=correlation
            )
            _emit_live_log(tools, _with_correlation({
                "type": "tool_call_timeout",
                "task_id": task_id,
                "tool": fn_name,
                "args": args_for_log,
                "duration_sec": round(time.perf_counter() - started_at, 3),
                "timeout_sec": timeout_sec,
            }, correlation, tool_call_id=tool_call_id))
            return timeout_result
    else:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_execute_single_tool, tools, tc, drive_logs, task_id)
            try:
                result = future.result(timeout=timeout_sec)
                result_meta = result.get("result_meta") or {}
                _emit_live_log(tools, _with_correlation({
                    "type": "tool_call_finished",
                    "task_id": task_id,
                    "tool": fn_name,
                    "args": result.get("args_for_log", args_for_log),
                    "duration_sec": round(time.perf_counter() - started_at, 3),
                    "is_error": bool(result.get("is_error")),
                    "status": result_meta.get("status"),
                    "exit_code": result_meta.get("exit_code"),
                    "signal": result_meta.get("signal"),
                    "result_preview": sanitize_tool_result_for_log(
                        truncate_for_log(result.get("result", ""), 500)
                    ),
                }, correlation, tool_call_id=tool_call_id))
                return result
            except (TimeoutError, concurrent.futures.TimeoutError):
                is_reviewed_mutative = fn_name in REVIEWED_MUTATIVE_TOOLS
                is_foreground_mutative = fn_name in FOREGROUND_MUTATIVE_TOOLS

                if is_reviewed_mutative or is_foreground_mutative:
                    # Review/code mutation tools must not end with ambiguous timeout.
                    try:
                        from ouroboros.tools.commit_gate import _mark_review_attempt_late
                        ctx = getattr(tools, "_ctx", None)
                        if ctx is not None and is_reviewed_mutative:
                            _mark_review_attempt_late(
                                ctx,
                                soft_timeout_sec=timeout_sec,
                                duration_sec=round(time.perf_counter() - started_at, 1),
                            )
                    except Exception:
                        log.debug("Failed to mark reviewed attempt as late_result_pending", exc_info=True)
                    _emit_live_log(tools, _with_correlation({
                        "type": "tool_call_late",
                        "task_id": task_id,
                        "tool": fn_name,
                        "args": args_for_log,
                        "soft_timeout_sec": timeout_sec,
                        "message": (
                            f"Foreground mutative tool '{fn_name}' exceeded "
                            f"{timeout_sec}s — still waiting for result "
                            + (
                                f"(hard ceiling: {_REVIEWED_MUTATIVE_HARD_CEILING}s)"
                                if is_reviewed_mutative else "(terminal wait: no background edits)"
                            )
                        ),
                    }, correlation, tool_call_id=tool_call_id))
                    if is_foreground_mutative:
                        result = future.result()
                        result_meta = result.get("result_meta") or {}
                        _emit_live_log(tools, _with_correlation({
                            "type": "tool_call_finished",
                            "task_id": task_id,
                            "tool": fn_name,
                            "args": result.get("args_for_log", args_for_log),
                            "duration_sec": round(time.perf_counter() - started_at, 3),
                            "is_error": bool(result.get("is_error")),
                            "status": result_meta.get("status"),
                            "late": True,
                            "terminal_wait": True,
                        }, correlation, tool_call_id=tool_call_id))
                        return result
                    try:
                        ceiling = max(_REVIEWED_MUTATIVE_HARD_CEILING, timeout_sec + 60)
                        remaining = max(1, ceiling - timeout_sec)
                        result = future.result(timeout=remaining)
                        result_meta = result.get("result_meta") or {}
                        _emit_live_log(tools, _with_correlation({
                            "type": "tool_call_finished",
                            "task_id": task_id,
                            "tool": fn_name,
                            "args": result.get("args_for_log", args_for_log),
                            "duration_sec": round(time.perf_counter() - started_at, 3),
                            "is_error": bool(result.get("is_error")),
                            "status": result_meta.get("status"),
                            "late": True,
                        }, correlation, tool_call_id=tool_call_id))
                        return result
                    except (TimeoutError, concurrent.futures.TimeoutError):
                        # Hard ceiling records terminal state; late real result may overwrite.
                        try:
                            from ouroboros.tools.commit_gate import _record_commit_attempt
                            ctx = getattr(tools, "_ctx", None)
                            if ctx is not None:
                                _record_commit_attempt(
                                    ctx,
                                    commit_message=str(getattr(ctx, "_current_review_commit_message", "") or ""),
                                    status="failed",
                                    block_reason="infra_failure",
                                    block_details=(
                                        f"Hard ceiling timeout ({_REVIEWED_MUTATIVE_HARD_CEILING}s). "
                                        "The underlying operation may still complete later."
                                    ),
                                    duration_sec=round(time.perf_counter() - started_at, 1),
                                    late_result_pending=True,
                                    phase="late_hard_ceiling",
                                    readiness_warnings=[
                                        "Reviewed mutative tool exceeded the hard ceiling; late result may still arrive."
                                    ],
                                    degraded_reasons=[
                                        f"hard_ceiling_timeout:{_REVIEWED_MUTATIVE_HARD_CEILING}"
                                    ],
                                )
                        except Exception:
                            pass
                        timeout_result = _make_timeout_result(
                            fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                            _REVIEWED_MUTATIVE_HARD_CEILING, task_id,
                            reset_msg=(
                                f"CRITICAL: Reviewed mutative tool hit hard ceiling "
                                f"({_REVIEWED_MUTATIVE_HARD_CEILING}s). "
                                "Check git state manually. "
                            ),
                            correlation=correlation,
                        )
                        _emit_live_log(tools, _with_correlation({
                            "type": "tool_call_timeout",
                            "task_id": task_id,
                            "tool": fn_name,
                            "args": args_for_log,
                            "duration_sec": round(time.perf_counter() - started_at, 3),
                            "timeout_sec": _REVIEWED_MUTATIVE_HARD_CEILING,
                            "hard_ceiling": True,
                        }, correlation, tool_call_id=tool_call_id))
                        return timeout_result
                else:
                    timeout_result = _make_timeout_result(
                        fn_name, tool_call_id, is_code_tool, tc, drive_logs,
                        timeout_sec, task_id, reset_msg="", correlation=correlation
                    )
                    _emit_live_log(tools, _with_correlation({
                        "type": "tool_call_timeout",
                        "task_id": task_id,
                        "tool": fn_name,
                        "args": args_for_log,
                        "duration_sec": round(time.perf_counter() - started_at, 3),
                        "timeout_sec": timeout_sec,
                    }, correlation, tool_call_id=tool_call_id))
                    return timeout_result
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


_PARALLEL_SAFE_TOOLS: frozenset[str] = READ_ONLY_PARALLEL_TOOLS | PARALLEL_SAFE_ENQUEUE_TOOLS


def tool_calls_can_run_parallel(tool_calls: List[Dict[str, Any]]) -> bool:
    """True when a tool-call round may execute in the shared ThreadPool.

    Read-only-parallel tools plus fire-and-forget enqueue tools
    (schedule_subagent) qualify; any other tool forces sequential execution.
    """
    return (
        len(tool_calls) > 1
        and all(
            str(tc.get("function", {}).get("name") or "").strip() in _PARALLEL_SAFE_TOOLS
            for tc in tool_calls
        )
    )


def handle_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: ToolRegistry,
    drive_logs: pathlib.Path,
    task_id: str,
    stateful_executor: StatefulToolExecutor,
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """Execute tool calls, append results, and return error count."""
    can_parallel = tool_calls_can_run_parallel(tool_calls)

    if not can_parallel:
        results = [
            _execute_with_timeout(tools, tc, drive_logs,
                                  _get_tool_timeout(tools, str(tc["function"]["name"] or "").strip(), _tc_args(tc)), task_id,
                                  stateful_executor)
            for tc in tool_calls
        ]
    else:
        max_workers = min(len(tool_calls), 8)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_index = {
                executor.submit(
                    _execute_with_timeout, tools, tc, drive_logs,
                    _get_tool_timeout(tools, str(tc["function"]["name"] or "").strip(), _tc_args(tc)), task_id,
                    stateful_executor,
                ): idx
                for idx, tc in enumerate(tool_calls)
            }
            results = [None] * len(tool_calls)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    tc = tool_calls[idx]
                    requested_fn_name = tc.get("function", {}).get("name", "unknown")
                    fn_name = str(requested_fn_name or "").strip()
                    safe_error = sanitize_tool_result_for_log(str(exc))
                    results[idx] = {
                        "tool_call_id": tc.get("id", ""),
                        "fn_name": fn_name,
                        "result": f"⚠️ TOOL_ERROR: Unexpected error: {safe_error}",
                        "is_error": True,
                        "tool_args": {},
                        "args_for_log": {},
                        "is_code_tool": fn_name in tools.CODE_TOOLS,
                        "result_meta": _extract_result_metadata(
                            fn_name,
                            f"⚠️ TOOL_ERROR: Unexpected error: {safe_error}",
                            True,
                        ),
                    }
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return process_tool_results(results, messages, llm_trace, emit_progress)


def process_tool_results(
    results: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> int:
    """Append tool results to messages/trace and return error count."""
    error_count = 0

    for exec_result in results:
        fn_name = exec_result["fn_name"]
        is_error = exec_result["is_error"]

        if is_error:
            error_count += 1

        truncated_result = _truncate_tool_result(
            exec_result["result"],
            tool_name=fn_name,
            tool_args=exec_result.get("tool_args"),
        )

        messages.append({
            "role": "tool",
            "tool_call_id": exec_result["tool_call_id"],
            "content": truncated_result
        })

        llm_trace["tool_calls"].append({
            "tool": fn_name,
            "args": _safe_args(exec_result["args_for_log"]),
            "result": truncate_for_log(exec_result["result"], 700),
            "is_error": is_error,
            "trace_ref": exec_result.get("trace_ref"),
            **(exec_result.get("result_meta") or {}),
        })
        if fn_name == "task_acceptance_review" and not is_error:
            raw = str(exec_result.get("result") or "")
            # The auto self-call leads with a compact improvement capsule and wraps
            # the full ReviewRunResult JSON in <full_review>...</full_review> (M5).
            # Extract that block so the FULL record still lands in review_runs even
            # when the visible result is not pure JSON — otherwise an auto review
            # that produced a capsule would record nothing and leave the objective
            # unevaluated exactly when the feedback matters most.
            payload = raw
            if "<full_review>" in raw and "</full_review>" in raw:
                payload = raw.split("<full_review>", 1)[1].rsplit("</full_review>", 1)[0].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    llm_trace.setdefault("review_runs", []).append(parsed)
                    # v6.54.4 (review round 2): dissent is recorded on the agent-called
                    # path too — merge into acceptance_decision without requiring an
                    # agent_decision envelope.
                    if parsed.get("dissent_noted"):
                        _dec = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
                        _dec.setdefault("source", "agent_task_acceptance_review_tool")
                        _dec["dissent_noted"] = True
                        llm_trace["acceptance_decision"] = _dec
                    agent_decision = parsed.get("agent_decision") if isinstance(parsed.get("agent_decision"), dict) else {}
                    if agent_decision:
                        llm_trace["acceptance_decision"] = {
                            "status": str(agent_decision.get("disposition") or ""),
                            "source": str(agent_decision.get("source") or "agent_task_acceptance_review_tool"),
                            "rationale": str(agent_decision.get("rationale") or "")[:500],
                            "agent_disposition": str(agent_decision.get("disposition") or ""),
                            "agent_rationale": str(agent_decision.get("rationale") or "")[:500],
                            # Carried forward, not overwritten (review round 3): the
                            # payload delivers dissent_noted and agent_decision together.
                            **({"dissent_noted": True} if parsed.get("dissent_noted") else {}),
                        }
                        # v6.54.4 obligations layer: apply the agent's per-obligation
                        # dispositions onto the host-collected per-task obligations.
                        ob_dispositions = agent_decision.get("obligation_dispositions")
                        if isinstance(ob_dispositions, list) and ob_dispositions:
                            by_id = {
                                str(e.get("id") or ""): e
                                for e in ob_dispositions if isinstance(e, dict)
                            }
                            for ob in (llm_trace.get("acceptance_obligations") or []):
                                if not isinstance(ob, dict):
                                    continue
                                entry = by_id.get(str(ob.get("id") or ""))
                                if entry:
                                    ob["disposition"] = str(entry.get("disposition") or "")
                                    ob["disposition_reason"] = str(entry.get("reason") or "")[:500]
                                    ob["status"] = "disposed"
            except Exception:
                log.debug("Failed to parse task_acceptance_review tool result", exc_info=True)

    return error_count


def _safe_args(v: Any) -> Any:
    """Ensure args are JSON-serializable for trace logging."""
    try:
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))
    except Exception:
        log.debug("Failed to serialize args for trace logging", exc_info=True)
        return {"_repr": repr(v)}
