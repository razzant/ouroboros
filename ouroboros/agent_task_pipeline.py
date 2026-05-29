"""Post-task result emission, memory work, reflections, and review context."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, List

from ouroboros.task_results import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    load_task_result,
    write_task_result,
)
from ouroboros.artifacts import collect_task_artifact_records, merge_artifact_records
from ouroboros.outcomes import (
    RESULT_SUCCEEDED,
    artifact_bundle_from_result,
    build_verification_ledger,
    derive_loop_outcome,
    maybe_write_verification_artifact,
)
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_review_artifact as _truncate_with_notice

log = logging.getLogger(__name__)


def _resolve_task_summary_model(default_model: str) -> str:
    prefix_to_provider = {
        "openai::": "openai",
        "anthropic::": "anthropic",
        "cloudru::": "cloudru",
        "gigachat::": "gigachat",
        "openai-compatible::": "openai-compatible",
        "openrouter::": "openrouter",
    }
    provider_env_keys: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "cloudru": ["CLOUDRU_FOUNDATION_MODELS_API_KEY"],
        "gigachat": ["GIGACHAT_CREDENTIALS"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }

    def model_has_credentials(model: str) -> bool:
        name = str(model or "").strip()
        provider = "openrouter"
        for prefix, candidate_provider in prefix_to_provider.items():
            if name.startswith(prefix):
                provider = candidate_provider
                break
        if provider == "openai-compatible":
            compat = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
            legacy_key = str(os.environ.get("OPENAI_API_KEY", "") or "").strip()
            legacy_base = str(os.environ.get("OPENAI_BASE_URL", "") or "").strip()
            return bool(compat or (legacy_key and legacy_base))
        if provider == "gigachat":
            creds = str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip()
            user = str(os.environ.get("GIGACHAT_USER", "") or "").strip()
            password = str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip()
            return bool(creds or (user and password))
        for env_key in provider_env_keys.get(provider, ["OPENROUTER_API_KEY"]):
            if str(os.environ.get(env_key, "") or "").strip():
                return True
        return False

    if model_has_credentials(default_model):
        return default_model

    for env_name in (
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_FALLBACK",
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_CODE",
    ):
        candidate = str(os.environ.get(env_name, "") or "").strip()
        if candidate and model_has_credentials(candidate):
            return candidate
    return default_model


def build_trace_summary(llm_trace: dict) -> str:
    """Return a compact human-readable summary of tool calls and agent notes."""
    tool_calls = llm_trace.get("tool_calls", []) or []
    notes = llm_trace.get("reasoning_notes", []) or []

    n = len(tool_calls)
    errors = sum(1 for tc in tool_calls if isinstance(tc, dict) and tc.get("is_error"))

    lines: list[str] = [f"## Tool trace ({n} calls, {errors} errors)"]

    if not tool_calls:
        lines.append("No tool calls.")
    else:
        def _fmt_call(idx: int, tc: dict) -> str:
            name = tc.get("tool", "unknown")
            args = tc.get("args", {})
            if isinstance(args, dict):
                parts = []
                arg_items = list(args.items())
                for k, v in arg_items[:2]:
                    v_str = str(v)
                    if len(v_str) > 60:
                        v_str = _truncate_with_notice(v_str, 60).replace("\n", " ")
                    parts.append(f"{k}={v_str!r}")
                if len(arg_items) > 2:
                    parts.append(f"⚠️ OMISSION NOTE: {len(arg_items) - 2} more args omitted")
                args_str = ", ".join(parts)
            else:
                args_str = repr(args)
                if len(args_str) > 80:
                    args_str = _truncate_with_notice(args_str, 80).replace("\n", " ")
            facts = []
            status = str(tc.get("status") or "").strip()
            if status and status != "ok":
                facts.append(f"status={status}")
            if tc.get("exit_code") not in (None, 0):
                facts.append(f"exit_code={tc.get('exit_code')}")
            if tc.get("signal"):
                facts.append(f"signal={tc.get('signal')}")
            fact_suffix = f" [{', '.join(facts)}]" if facts else ""
            suffix = " → ERROR" if tc.get("is_error") else ""
            return f"{idx}. {name}({args_str}){fact_suffix}{suffix}"

        if n > 30:
            shown = (
                [_fmt_call(i + 1, tool_calls[i]) for i in range(15)]
                + [f"⚠️ OMISSION NOTE: {n - 30} middle tool calls omitted from trace summary."]
                + [_fmt_call(n - 14 + i, tool_calls[n - 15 + i]) for i in range(15)]
            )
        else:
            shown = [_fmt_call(i + 1, tool_calls[i]) for i in range(n)]
        lines.extend(shown)

    if notes:
        lines.append("\n## Agent notes (supplementary, not source of truth)")
        lines.extend(f"- {note}" for note in notes)

    summary = "\n".join(lines)
    if len(summary) > 4000:
        summary = _truncate_with_notice(summary, 4000)
    return summary


def _update_improvement_backlog(
    env: Any,
    reflection_entry: Dict[str, Any] | None,
) -> int:
    """Persist LLM-nominated follow-up improvements into the durable backlog."""
    try:
        from ouroboros.improvement_backlog import append_backlog_items

        candidates = list((reflection_entry or {}).get("backlog_candidates") or [])
        if not candidates:
            return 0
        return append_backlog_items(env.drive_root, candidates)
    except Exception:
        log.debug("Improvement backlog update failed", exc_info=True)
        return 0


def _apply_reflection_memory_actions(
    env: Any,
    reflection_entry: Dict[str, Any] | None,
) -> int:
    """Auto-apply LLM-nominated durable memory actions from the experience review.

    Runs against ``env.drive_root``; for forked/workspace tasks the finalizer
    also invokes post-task processing with the parent drive, so learnings land
    on the canonical drive rather than a discarded child drive.
    """
    try:
        actions = list((reflection_entry or {}).get("memory_actions") or [])
        if not actions:
            return 0
        from ouroboros.reflection import apply_memory_actions

        return apply_memory_actions(env, actions)
    except Exception:
        log.debug("Reflection memory action application failed", exc_info=True)
        return 0


def _child_task_evidence(env: Any, task: Dict[str, Any], limit: int = 6000) -> str:
    """Return compact evidence from child/subagent results for parent experience review."""
    task_id = str(task.get("id") or "")
    if not task_id:
        return ""
    try:
        from ouroboros.task_results import list_task_results

        rows = []
        for item in list_task_results(env.drive_root):
            if not isinstance(item, dict):
                continue
            if str(item.get("parent_task_id") or "") != task_id and str(item.get("root_task_id") or "") != task_id:
                continue
            rows.append({
                "task_id": item.get("task_id") or item.get("id"),
                "status": item.get("status"),
                "role": item.get("role"),
                "result_status": item.get("result_status"),
                "cost_usd": item.get("cost_usd"),
                "trace_summary": _truncate_with_notice(item.get("trace_summary", ""), 800),
                "result": _truncate_with_notice(item.get("result", ""), 1600),
            })
        if not rows:
            return ""
        return _truncate_with_notice(json.dumps(rows, ensure_ascii=False, indent=2), limit)
    except Exception:
        log.debug("Failed to collect child task evidence", exc_info=True)
        return ""


def _run_post_task_processing_async(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    review_evidence: Dict[str, Any],
    drive_logs: pathlib.Path,
    *,
    blocking: bool = False,
) -> None:
    """Run best-effort LLM-heavy post-task memory work off the reply path."""
    task_snapshot = json.loads(json.dumps(task, ensure_ascii=False, default=str))
    usage_snapshot = json.loads(json.dumps(usage, ensure_ascii=False, default=str))
    trace_snapshot = json.loads(json.dumps(llm_trace, ensure_ascii=False, default=str))
    review_evidence_snapshot = json.loads(json.dumps(review_evidence, ensure_ascii=False, default=str))

    def _run() -> None:
        try:
            from ouroboros.llm import LLMClient

            llm_client = LLMClient()
            # Summary first: chat.jsonl is more durable than best-effort reflection/backlog.
            _run_task_summary(
                env,
                llm_client,
                task_snapshot,
                usage_snapshot,
                trace_snapshot,
                drive_logs,
                review_evidence=review_evidence_snapshot,
            )
            reflection_entry = _run_reflection(
                env, llm_client, task_snapshot, usage_snapshot,
                trace_snapshot, review_evidence_snapshot,
            )
            _update_improvement_backlog(env, reflection_entry)
            _apply_reflection_memory_actions(env, reflection_entry)
        except Exception:
            log.warning("Async post-task processing failed", exc_info=True)

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


def emit_task_results(
    env: Any, memory: Any, llm: Any,
    pending_events: List[Dict[str, Any]],
    task: Dict[str, Any], text: str,
    usage: Dict[str, Any], llm_trace: Dict[str, Any],
    start_time: float, drive_logs: pathlib.Path,
    ctx: Any = None,
) -> None:
    """Emit all end-of-task events to supervisor and run post-task processing."""
    loop_outcome = derive_loop_outcome(text or "", usage, llm_trace)
    result_status = str(loop_outcome.get("result_status") or "")
    reason_code = str(loop_outcome.get("reason_code") or "")
    pending_events.append({
        "type": "send_message", "chat_id": task["chat_id"],
        "text": text or "\u200b", "log_text": text or "",
        "format": "markdown",
        "task_id": task.get("id"), "ts": utc_now_iso(),
    })

    duration_sec = round(time.time() - start_time, 3)
    n_tool_calls = len(llm_trace.get("tool_calls", []))
    n_tool_errors = sum(1 for tc in llm_trace.get("tool_calls", [])
                        if isinstance(tc, dict) and tc.get("is_error"))
    try:
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "task_eval", "ok": result_status == RESULT_SUCCEEDED,
            "task_id": task.get("id"), "task_type": task.get("type"),
            "result_status": result_status,
            "reason_code": reason_code,
            "review_eligibility": str(loop_outcome.get("review_eligibility") or ""),
            "review_trigger": str(loop_outcome.get("review_trigger") or ""),
            "duration_sec": duration_sec,
            "tool_calls": n_tool_calls,
            "tool_errors": n_tool_errors,
            "response_len": len(text),
        })
    except Exception:
        log.warning("Failed to log task eval event", exc_info=True)
        pass

    pending_events.append({
        "type": "task_metrics",
        "task_id": task.get("id"), "task_type": task.get("type"),
        "result_status": result_status,
        "reason_code": reason_code,
        "duration_sec": duration_sec,
        "tool_calls": n_tool_calls, "tool_errors": n_tool_errors,
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_rounds": int(usage.get("rounds") or 0),
        "ts": utc_now_iso(),
    })

    review_evidence: Dict[str, Any] = {}
    try:
        from ouroboros.review_evidence import collect_review_evidence

        review_evidence = collect_review_evidence(
            env.drive_root,
            task_id=str(task.get("id") or ""),
            repo_dir=getattr(env, "repo_dir", None),
        )
    except Exception:
        log.debug("Failed to collect review evidence", exc_info=True)

    _store_task_result(env, task, text, usage, llm_trace, review_evidence=review_evidence)
    stored_result = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    artifact_bundle = stored_result.get("artifact_bundle") if isinstance(stored_result.get("artifact_bundle"), dict) else {}
    pending_events.append({
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        "result_status": result_status,
        "reason_code": reason_code,
        "artifact_status": stored_result.get("artifact_status") or artifact_bundle.get("status") or "",
        "artifact_bundle": artifact_bundle,
        "review_status": stored_result.get("review_status") if isinstance(stored_result.get("review_status"), dict) else {},
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        "total_rounds": int(usage.get("rounds") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "ts": utc_now_iso(),
    })
    # NOTE: task_done is NOT written to events.jsonl here.
    # It goes through EVENT_Q → supervisor _handle_task_done → append_jsonl.
    # This ensures causal ordering: send_message reaches the UI before task_done,
    # preventing the live card from collapsing before the assistant reply arrives.
    restart_reason = str(getattr(ctx, "pending_restart_reason", "") or "").strip()
    if restart_reason:
        pending_events.append({
            "type": "restart_request",
            "reason": restart_reason,
            "ts": utc_now_iso(),
        })
        try:
            ctx.pending_restart_reason = None
        except Exception:
            pass

    if str(task.get("delegation_role") or "") != "subagent":
        post_usage = dict(usage or {})
        post_usage["result_status"] = result_status
        post_usage["reason_code"] = reason_code
        _run_chat_consolidation(env, memory, llm, task, drive_logs)
        _run_scratchpad_consolidation(env, memory, llm)
        # LLM-heavy memory work stays off the reply critical path.
        _run_post_task_processing_async(
            env, task, post_usage, llm_trace, review_evidence, drive_logs,
            blocking=(
                str(task.get("type") or "") == "evolution"
                or bool(str(task.get("workspace_root") or "").strip())
                or bool(str(task.get("workspace_mode") or "").strip())
            ),
        )
        budget_drive_root = str(task.get("budget_drive_root") or "").strip()
        if budget_drive_root and str(pathlib.Path(budget_drive_root).resolve(strict=False)) != str(pathlib.Path(env.drive_root).resolve(strict=False)):
            from types import SimpleNamespace

            parent_env = SimpleNamespace(repo_dir=env.repo_dir, drive_root=pathlib.Path(budget_drive_root), drive_path=lambda rel: pathlib.Path(budget_drive_root) / rel)
            _run_post_task_processing_async(
                parent_env,
                {**task, "drive_root": budget_drive_root, "child_drive_root": str(env.drive_root)},
                post_usage,
                llm_trace,
                review_evidence,
                pathlib.Path(budget_drive_root) / "logs",
                blocking=True,
            )


def _store_task_result(env: Any, task: Dict[str, Any], text: str,
                       usage: Dict[str, Any], llm_trace: Dict[str, Any],
                       review_evidence: Dict[str, Any] | None = None) -> None:
    """Store task result for parent task retrieval."""
    try:
        trace_summary = build_trace_summary(llm_trace)
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
        loop_outcome = derive_loop_outcome(text or "", usage, llm_trace)
        result_status = str(loop_outcome.get("result_status") or "")
        reason_code = str(loop_outcome.get("reason_code") or "")
        status = (
            STATUS_FAILED
            if str(existing.get("status") or "") == STATUS_FAILED or result_status != RESULT_SUCCEEDED
            else STATUS_COMPLETED
        )
        artifact_bundle_for_ledger = artifact_bundle_from_result(existing)
        verification_ledger = build_verification_ledger(
            task=task,
            loop_outcome=loop_outcome,
            llm_trace=llm_trace,
            artifact_bundle=artifact_bundle_for_ledger,
            review_evidence=review_evidence or {},
        )
        verification_refs = maybe_write_verification_artifact(
            env.drive_root,
            str(task.get("id") or ""),
            verification_ledger,
        )
        artifacts = list(existing.get("artifacts") or []) if isinstance(existing.get("artifacts"), list) else []
        artifact_record = verification_refs.get("artifact")
        if artifact_record and artifact_record not in artifacts:
            artifacts.append(artifact_record)
        collected_artifacts = collect_task_artifact_records(env.drive_root, str(task.get("id") or ""))
        artifacts = merge_artifact_records(artifacts, collected_artifacts)
        provisional = {
            **existing,
            "artifacts": artifacts,
        }
        artifact_bundle = artifact_bundle_from_result(provisional)
        write_task_result(
            env.drive_root,
            str(task.get("id") or ""),
            status,
            result_status=result_status,
            reason_code=reason_code,
            loop_outcome=loop_outcome,
            parent_task_id=task.get("parent_task_id"),
            root_task_id=task.get("root_task_id"),
            session_id=task.get("session_id"),
            actor_id=task.get("actor_id"),
            delegation_role=task.get("delegation_role"),
            description=task.get("description"),
            context=task.get("context"),
            workspace_root=task.get("workspace_root"),
            workspace_mode=task.get("workspace_mode"),
            memory_mode=task.get("memory_mode"),
            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=text or "",
            trace_summary=trace_summary,
            trace_refs=loop_outcome.get("trace_refs") or {},
            cost_usd=round(float(usage.get("cost") or 0), 6),
            total_rounds=int(usage.get("rounds") or 0),
            review_evidence=review_evidence or {},
            verification_ledger=verification_refs.get("inline"),
            artifact_bundle=artifact_bundle,
            artifacts=artifacts,
            ts=utc_now_iso(),
        )
    except Exception as e:
        log.warning("Failed to store task result: %s", e)


_TASK_SUMMARY_PROMPT = """\
Summarize this completed task for Ouroboros's episodic memory.
Be specific about: what was tried, what worked, what failed, key decisions made.
Include file names, tool names, error messages when relevant.
Treat tool statuses and exit/signal facts as authoritative. Agent notes are supplementary only.
Never claim a tool succeeded when the trace shows non-zero exit, timeout, install_error, or any error status.
If structured review evidence contains critical/advisory findings or open obligations,
mention them individually with severity, item/tag identity, and whether they blocked
the commit, remained open, or were resolved.
If the task was trivial (0 tool calls and ≤1 round), keep it to 1-2 sentences and DO NOT add meta-reflection.
If the task was non-trivial, end with a short meta-reflection section:
- What friction, errors, or weak assumptions slowed the work?
- What should Ouroboros change in its own process or prompts to avoid repeating that class of mistake?
Keep the meta-reflection concrete and operational, not narrative.
End with: "Details: progress.jsonl + tools.jsonl for task_id={task_id}"

## Task
Goal: {goal}
Type: {task_type}
Rounds: {rounds}, Cost: ${cost:.2f}

## Execution trace
{trace_summary}

## Structured review evidence
{review_evidence}
"""


def _run_task_summary(env, llm, task, usage, llm_trace, drive_logs, review_evidence=None):
    """Generate a detailed task summary and inject it into chat.jsonl."""
    try:
        from ouroboros.consolidator import (
            CONSOLIDATION_MODEL,
            CONSOLIDATION_REASONING_EFFORT,
        )
        task_id = task.get("id", "unknown")
        n_tool_calls = len(llm_trace.get("tool_calls", []) or [])
        rounds = int(usage.get("rounds") or 0)
        cost = float(usage.get("cost") or 0)
        result_status = str(usage.get("result_status") or "")
        reason_code = str(usage.get("reason_code") or "")

        # Skip LLM summary for trivial tasks.
        if n_tool_calls == 0 and rounds <= 1:
            goal = _truncate_with_notice(task.get("text", ""), 200)
            summary_text = (
                f"Task {task_id} ({task.get('type', 'user')}): "
                f"{goal}. {rounds}r, ${cost:.2f}."
            )
            append_jsonl(drive_logs / "chat.jsonl", {
                "ts": utc_now_iso(), "direction": "system",
                "type": "task_summary", "task_id": task_id, "text": summary_text,
                "tool_calls": n_tool_calls, "rounds": rounds,
                "result_status": result_status, "reason_code": reason_code,
            })
            return

        summary_model = _resolve_task_summary_model(CONSOLIDATION_MODEL)
        goal = _truncate_with_notice(task.get("text", ""), 500)
        trace = build_trace_summary(llm_trace)
        try:
            from ouroboros.review_evidence import format_review_evidence_for_prompt
            review_section = format_review_evidence_for_prompt(review_evidence or {}, max_chars=8000)
        except Exception:
            review_section = "(review evidence unavailable)"
        prompt = _TASK_SUMMARY_PROMPT.format(
            task_id=task_id, goal=goal or "(no goal text)",
            task_type=task.get("type", "user"), rounds=rounds,
            cost=cost,
            trace_summary=_truncate_with_notice(trace, 3000),
            review_evidence=review_section,
        )
        try:
            msg, _usage = llm.chat(messages=[{"role": "user", "content": prompt}],
                                   model=summary_model,
                                   reasoning_effort=CONSOLIDATION_REASONING_EFFORT,
                                   max_tokens=16384)
            summary_text = (msg.get("content") or "").strip()
            if _usage.get("cost"):
                try:
                    from supervisor.state import update_budget_from_usage
                    update_budget_from_usage(_usage)
                except Exception:
                    pass
        except Exception:
            log.warning("Task summary LLM call failed, using fallback", exc_info=True)
            summary_text = (
                f"Task {task_id} ({task.get('type', 'user')}): "
                f"{_truncate_with_notice(goal, 200)}. {rounds}r, ${cost:.2f}."
            )
        if summary_text:
            append_jsonl(drive_logs / "chat.jsonl", {
                "ts": utc_now_iso(), "direction": "system",
                "type": "task_summary", "task_id": task_id, "text": summary_text,
                "tool_calls": n_tool_calls, "rounds": rounds,
                "result_status": result_status, "reason_code": reason_code,
            })
    except Exception:
        log.debug("Task summary generation failed (non-critical)", exc_info=True)


def _run_chat_consolidation(env, memory, llm, task, drive_logs):
    """Run dialogue-block consolidation in a daemon thread."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate
        consolidate = _c.consolidate
        chat_path = drive_logs / "chat.jsonl"
        blocks_path = env.drive_path("memory") / "dialogue_blocks.json"
        meta_path = env.drive_path("memory") / "dialogue_meta.json"
        if should_consolidate(meta_path, chat_path):
            _id, _ident, _llm, _logs = task.get("id"), memory.load_identity(), llm, drive_logs
            def _run():
                try:
                    u = consolidate(chat_path=chat_path, blocks_path=blocks_path,
                                    meta_path=meta_path, llm_client=_llm, identity_text=_ident)
                    if u:
                        append_jsonl(_logs / "events.jsonl", {"ts": utc_now_iso(),
                            "type": "chat_block_consolidation", "task_id": _id,
                            "cost_usd": round(float(u.get("cost") or 0), 6)})
                        # Daemon-thread work updates budget directly.
                        if u.get("cost") or u.get("prompt_tokens"):
                            try:
                                from supervisor.state import update_budget_from_usage
                                update_budget_from_usage(u)
                            except Exception:
                                pass
                except Exception:
                    log.warning("Chat block consolidation failed", exc_info=True)
            threading.Thread(target=_run, daemon=True).start()
    except Exception:
        log.warning("Chat block consolidation setup failed", exc_info=True)


def _run_scratchpad_consolidation(env: Any, memory: Any, llm: Any) -> None:
    """Run scratchpad consolidation in a daemon thread."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate_scratchpad
        consolidate = _c.consolidate_scratchpad
        if should_consolidate(memory):
            kb_dir = env.drive_path("memory/knowledge")
            _identity = memory.load_identity()

            def _run():
                try:
                    u = consolidate(memory, kb_dir, llm, _identity)
                    # Daemon-thread work updates budget directly.
                    if u and (u.get("cost") or u.get("prompt_tokens")):
                        try:
                            from supervisor.state import update_budget_from_usage
                            update_budget_from_usage(u)
                        except Exception:
                            pass
                except Exception:
                    log.warning("Scratchpad consolidation failed", exc_info=True)

            threading.Thread(target=_run, daemon=True).start()
    except Exception:
        log.debug("Scratchpad consolidation setup failed", exc_info=True)


def _run_reflection(env: Any, llm: Any, task: Dict[str, Any],
                    usage: Dict[str, Any], llm_trace: Dict[str, Any],
                    review_evidence: Dict[str, Any]) -> Dict[str, Any] | None:
    """Run execution reflection synchronously (process memory, Bible P1)."""
    try:
        from ouroboros.reflection import (
            should_generate_reflection, generate_reflection, append_reflection,
        )
        if should_generate_reflection(
            llm_trace,
            task=task,
            rounds=int(usage.get("rounds", 0)),
            cost_usd=float(usage.get("cost", 0.0)),
        ):
            trace_summary = build_trace_summary(llm_trace)
            child_evidence = _child_task_evidence(env, task)
            try:
                entry = generate_reflection(
                    task, llm_trace, trace_summary,
                    llm, usage,
                    review_evidence=review_evidence,
                    child_evidence=child_evidence,
                )
                append_reflection(env.drive_root, entry)
                return entry
            except Exception:
                log.warning("Execution reflection failed (non-critical)", exc_info=True)
    except Exception:
        log.debug("Execution reflection setup failed", exc_info=True)
    return None


def build_review_context(env: Any) -> str:
    """Build a compact review continuity section for the main reasoning context."""
    try:
        from ouroboros.review_state import (
            _LEGACY_CURRENT_REPO_KEY,
            compute_snapshot_hash,
            format_status_section,
            load_state,
            make_repo_key,
        )
        from ouroboros.task_continuation import list_review_continuations
        from ouroboros.task_results import load_task_result

        state = load_state(pathlib.Path(env.drive_root))
        continuations, corrupt = list_review_continuations(env.drive_root)
        repo_dir = pathlib.Path(env.repo_dir)
        repo_key = make_repo_key(repo_dir)
        snapshot_hash = compute_snapshot_hash(repo_dir)
        open_obs = state.get_open_obligations(repo_key=repo_key)
        open_debts = state.get_open_commit_readiness_debts(repo_key=repo_key)
        if (
            not state.advisory_runs
            and not state.latest_attempt()
            and not continuations
            and not corrupt
            and not open_obs
            and not open_debts
        ):
            return ""

        current_run = None
        for run in reversed(state.advisory_runs):
            if run.snapshot_hash != snapshot_hash:
                continue
            if run.repo_key not in ("", repo_key, _LEGACY_CURRENT_REPO_KEY):
                continue
            current_run = run
            break

        lines: List[str] = ["## Review Continuity", "### Live repo gate"]
        live_status = str(getattr(current_run, "status", "") or "missing")
        repo_commit_ready = bool(
            current_run is not None
            and current_run.status in ("fresh", "bypassed", "skipped")
            and not open_obs
            and not open_debts
        )
        lines.append(f"- repo_key={repo_key}")
        lines.append(f"- snapshot_hash={snapshot_hash[:12] or '(empty)'}")
        lines.append(f"- advisory_status={live_status}")
        lines.append(f"- repo_commit_ready={'yes' if repo_commit_ready else 'no'}")
        if current_run is not None:
            lines.append(f"- current_review_ts={str(current_run.ts or '')[:19]}")
            if current_run.bypass_reason:
                lines.append(f"- bypass_reason={_truncate_with_notice(current_run.bypass_reason, 220)}")
        else:
            lines.append("- no advisory run matches the current worktree snapshot")

        stale_matches_repo = not state.last_stale_repo_key or state.last_stale_repo_key == repo_key
        if state.last_stale_from_edit_ts and stale_matches_repo:
            lines.append(
                f"- stale_marker={state.last_stale_from_edit_ts[:19]}: "
                f"{_truncate_with_notice(state.last_stale_reason or 'worktree edit invalidated advisory freshness', 220)}"
            )

        if open_debts:
            lines.append("- retry_anchor=commit_readiness_debt")
            lines.append(f"- commit_readiness_debt={len(open_debts)}")
            lines.append("\n### Commit-readiness debt (start retry here)")
            for debt in open_debts:
                summary = _truncate_with_notice(getattr(debt, "summary", ""), 180).replace("\n", " ")
                lines.append(
                    f"- [{getattr(debt, 'debt_id', '')}] status={getattr(debt, 'status', '')} "
                    f"category={getattr(debt, 'category', '')} source={getattr(debt, 'source', '')}"
                )
                lines.append(f"  summary={summary}")
                if getattr(debt, "source_obligation_ids", None):
                    lines.append(f"  obligation_ids={', '.join(list(debt.source_obligation_ids or []))}")
                for evidence in list(getattr(debt, "evidence", []) or []):
                    lines.append(f"  evidence={_truncate_with_notice(evidence, 180).replace(chr(10), ' ')}")
        else:
            lines.append("- commit_readiness_debt=0")

        if open_obs:
            lines.append(f"- open_obligations={len(open_obs)}")
            for ob in open_obs:
                reason = _truncate_with_notice(getattr(ob, "reason", ""), 120).replace("\n", " ")
                lines.append(
                    f"  [{getattr(ob, 'obligation_id', '')}] "
                    f"{getattr(ob, 'item', '')}: {reason}"
                )
        else:
            lines.append("- open_obligations=0")

        scoped_continuations = [
            item for item in continuations
            if item.repo_key in ("", repo_key, _LEGACY_CURRENT_REPO_KEY)
        ]
        if scoped_continuations:
            lines.append("\n### Open review continuations")
            scoped_continuations.sort(key=lambda item: str(item.updated_ts or item.created_ts or ""), reverse=True)
            # Cap review context only with explicit OMISSION NOTEs; no silent slicing.
            _CONTINUATION_CAP = 5
            _PER_FINDING_CAP = 3
            shown_continuations = scoped_continuations[:_CONTINUATION_CAP]
            if len(scoped_continuations) > _CONTINUATION_CAP:
                lines.append(
                    f"⚠️ OMISSION NOTE: {len(scoped_continuations) - _CONTINUATION_CAP} "
                    f"older continuation(s) omitted (showing {_CONTINUATION_CAP} most recent)."
                )
            for item in shown_continuations:
                task_status = str((load_task_result(env.drive_root, item.task_id) or {}).get("status") or "missing")
                lines.append(
                    f"- task={item.task_id} status={task_status} source={item.source} "
                    f"stage={item.stage} tool={item.tool_name or 'commit_reviewed'} "
                    f"attempt={int(item.attempt or 0)}"
                )
                if item.block_reason:
                    lines.append(f"  block_reason={item.block_reason}")
                if item.readiness_warnings:
                    shown = list(item.readiness_warnings)[:_PER_FINDING_CAP]
                    for warn in shown:
                        warning = _truncate_with_notice(warn, 180).replace("\n", " ")
                        lines.append(f"  readiness_warning={warning}")
                    if len(item.readiness_warnings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.readiness_warnings) - _PER_FINDING_CAP} "
                            f"additional readiness_warning(s) omitted."
                        )
                if item.critical_findings:
                    shown = list(item.critical_findings)[:_PER_FINDING_CAP]
                    for top in shown:
                        label = str(top.get("item") or top.get("reason") or "critical finding")
                        reason = _truncate_with_notice(top.get("reason") or "", 140).replace("\n", " ")
                        lines.append(f"  critical_finding={label}: {reason}")
                    if len(item.critical_findings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.critical_findings) - _PER_FINDING_CAP} "
                            f"additional critical_finding(s) omitted."
                        )
                if item.advisory_findings:
                    shown = list(item.advisory_findings)[:_PER_FINDING_CAP]
                    for top in shown:
                        label = str(top.get("item") or top.get("reason") or "advisory finding")
                        reason = _truncate_with_notice(top.get("reason") or "", 140).replace("\n", " ")
                        lines.append(f"  advisory_finding={label}: {reason}")
                    if len(item.advisory_findings) > _PER_FINDING_CAP:
                        lines.append(
                            f"  ⚠️ OMISSION NOTE: {len(item.advisory_findings) - _PER_FINDING_CAP} "
                            f"additional advisory_finding(s) omitted."
                        )
                if item.obligation_ids:
                    lines.append(f"  obligation_ids={', '.join(item.obligation_ids)}")
        if corrupt:
            lines.append("\n### Corrupt review continuations")
            _CORRUPT_CAP = 3
            shown_corrupt = corrupt[:_CORRUPT_CAP]
            for item in shown_corrupt:
                lines.append(f"- {_truncate_with_notice(item, 220)}")
            if len(corrupt) > _CORRUPT_CAP:
                lines.append(
                    f"⚠️ OMISSION NOTE: {len(corrupt) - _CORRUPT_CAP} "
                    f"additional corrupt entry/entries omitted."
                )

        history = format_status_section(state, repo_dir=repo_dir)
        if history:
            history = history.replace("## Advisory Pre-Review Status", "### Historical review ledger")
            lines.append("\n" + history)

        return "\n".join(lines)
    except Exception:
        log.debug("Failed to build review continuity context", exc_info=True)
        return ""
