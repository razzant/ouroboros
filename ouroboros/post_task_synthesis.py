"""Post-task synthesis workers for the task pipeline (v7 L-C2 split).

The LLM-heavy best-effort memory work the post-task orchestrator
(``agent_task_pipeline._run_post_task_processing_async``) dispatches after a
task ends: the tool-trace summary, the episodic task summary, chat/scratchpad
consolidation, the execution reflection with its child-task evidence, the
durable improvement backlog and reflection memory actions, plus the shared
pre-synthesis usage snapshot and the compact review projection those prompts
embed. Extracted from agent_task_pipeline.py; the pipeline re-exports every
name, so historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import replace
from typing import Any, Dict

from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.post_task_checkpoint import is_root_post_task as _is_root_post_task
from ouroboros.synthesis_cost_text import (
    _synthesis_cost_text,
    _synthesis_cost_usd,
    _synthesis_usage_snapshot_text,
)
from ouroboros.task_finalization import sealed_final_prompt_section
from ouroboros.task_results import TASK_COST_META_FIELDS
from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.utils import truncate_review_artifact as _truncate_with_notice

log = logging.getLogger("ouroboros.agent_task_pipeline")


def build_trace_summary(llm_trace: dict) -> str:
    """Return a compact human-readable summary of tool calls and agent notes."""
    tool_calls = llm_trace.get("tool_calls", []) or []
    notes = llm_trace.get("reasoning_notes", []) or []

    n = len(tool_calls)
    # v6.57.0 — honest breakdown so a task that finished with a deliverable is not
    # mislabeled "43 errors" (the site/PB incidents): separate GENUINE unresolved
    # errors from POLICY denials, cosmetic non-zero exits, recovered errors, and
    # ignored read-only blocks. Self-learning (reflection reads this) must not be
    # poisoned by counting policy refusals or intentional probe exits as failures.
    from ouroboros.outcomes import _classify_tool_errors

    _buckets = _classify_tool_errors(llm_trace)
    _unresolved = len(_buckets.get("unresolved") or [])
    _policy = len(_buckets.get("policy_denials") or [])
    _cosmetic = len(_buckets.get("cosmetic") or [])
    _recovered = len(_buckets.get("recovered") or [])
    _ignored = len(_buckets.get("ignored") or [])
    _breakdown_bits = [f"{_unresolved} errors"]
    if _policy:
        _breakdown_bits.append(f"{_policy} policy-denied")
    if _recovered:
        _breakdown_bits.append(f"{_recovered} recovered")
    if _cosmetic:
        _breakdown_bits.append(f"{_cosmetic} cosmetic")
    if _ignored:
        _breakdown_bits.append(f"{_ignored} ignored")

    lines: list[str] = [f"## Tool trace ({n} calls, {', '.join(_breakdown_bits)})"]

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
                    if len(v_str) > 200:
                        v_str = _truncate_with_notice(v_str, 200).replace("\n", " ")
                    parts.append(f"{k}={v_str!r}")
                if len(arg_items) > 2:
                    parts.append(f"⚠️ OMISSION NOTE: {len(arg_items) - 2} more args omitted")
                args_str = ", ".join(parts)
            else:
                args_str = repr(args)
                if len(args_str) > 200:
                    args_str = _truncate_with_notice(args_str, 200).replace("\n", " ")
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
        added = append_backlog_items(env.drive_root, candidates)
        try:
            from ouroboros.improvement_backlog import groom_backlog

            groom_backlog(env.drive_root)  # size-triggered; no-op while small
        except Exception:
            log.debug("Backlog grooming failed", exc_info=True)
        return added
    except Exception:
        log.debug("Improvement backlog update failed", exc_info=True)
        return 0


def _apply_reflection_memory_actions(
    env: Any,
    reflection_entry: Dict[str, Any] | None,
    project_id: str = "",
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

        return apply_memory_actions(env, actions, project_id=project_id)
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
                "outcome_axes": normalize_outcome_axes(item),
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


def _pre_synthesis_usage_snapshot(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    """Freeze one honest, non-final root/subtree cost view for synthesis.

    Summary and reflection share this loop-local dictionary.  The existing
    terminal checkpoint remains the sole final authority after their own model
    calls settle.
    """
    snapshot = json.loads(json.dumps(usage, ensure_ascii=False, default=str))
    if not _is_root_post_task(task):
        return snapshot

    task_id = str(task.get("id") or task.get("task_id") or "")
    budget_root = pathlib.Path(
        task.get("budget_drive_root") or getattr(env, "drive_root", ".")
    )
    snapshot.update({
        "cost_snapshot_at": utc_now_iso(),
        "cost_final": False,
        "cost_with_children_partial": True,
    })
    try:
        from ouroboros.usage_accounting import usage_breakdown

        logical_root_id = str(task.get("root_task_id") or task_id)
        subtree = usage_breakdown(budget_root, root_task_id=logical_root_id)
        snapshot.update({
            "cost_usd_with_children": round(float(subtree["accounted_usd"]), 6),
            "reserved_usd": round(float(subtree["reserved_usd"]), 6),
            "unresolved_upper_bound_usd": round(
                float(subtree["unresolved_upper_bound_usd"]), 6
            ),
            "unknown_unmetered": int(subtree["unknown_unmetered"]),
            "ledger_integrity": (
                "degraded" if bool(subtree.get("integrity_degraded")) else "ok"
            ),
            "cost_accounting_status": "available",
        })
    except Exception:
        log.warning(
            "Pre-synthesis subtree cost is unavailable for %s",
            task_id or "unknown",
            exc_info=True,
        )
        snapshot.update({
            "cost_usd_with_children": None,
            "reserved_usd": None,
            "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
            "ledger_integrity": "unavailable",
            "cost_accounting_status": "unavailable",
        })
    return snapshot


def _compact_review_projection(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build the public review projection without copying raw actor output."""
    try:
        from ouroboros.review_substrate import compact_review_projection

        return compact_review_projection(llm_trace.get("review_runs") or [])
    except Exception:
        log.debug("Failed to build compact review projection", exc_info=True)
        return {"panels": []}


_TASK_SUMMARY_PROMPT = """\
Summarize this completed task for Ouroboros's episodic memory.
Be specific about: what was tried, what worked, what failed, key decisions made.
Include file names, tool names, error messages when relevant.
Treat tool statuses and exit/signal facts as authoritative. Agent notes are supplementary only.
Never claim a tool succeeded when the trace shows non-zero exit, timeout, or any error status.
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
Rounds: {rounds}, Cost: {cost_text}

{usage_snapshot}{sealed_final}## Execution trace
{trace_summary}

## Structured review evidence
{review_evidence}
"""


def _summary_row_cost_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Flat task-scope cost fields for the task_summary chat row (v6.82 P1).

    Mapped explicitly from the pre-synthesis usage snapshot
    (``_pre_synthesis_usage_snapshot``): only the snapshot's own honest keys are
    copied. Its schema deliberately differs from the full nine-field browser set
    — it carries no ``cost_usd``/``cost_accounting_error`` — and a non-root
    snapshot without accounting keys yields nothing. Never fabricates values;
    the terminal ``task_results`` checkpoint stays the final authority (history
    replay overrides these row values with it when the result file survives).
    """
    return {key: usage[key] for key in TASK_COST_META_FIELDS if key in usage}


def _run_task_summary(env, llm, task, usage, llm_trace, drive_logs, review_evidence=None,
                      sealed_final=None):
    """Generate a detailed task summary and inject it into chat.jsonl."""
    try:
        from ouroboros.projects_registry import project_thread_note_for_task

        from ouroboros.consolidator import (
            CONSOLIDATION_REASONING_EFFORT,
            _consolidation_route,
        )
        task_id = task.get("id", "unknown")
        n_tool_calls = len(llm_trace.get("tool_calls", []) or [])
        rounds = int(usage.get("rounds") or 0)
        cost_text = _synthesis_cost_text(usage)
        outcome_axes = normalize_outcome_axes(usage)
        reason_code = str(usage.get("reason_code") or "")
        review_projection = _compact_review_projection(llm_trace)

        # Skip LLM summary for trivial tasks.
        if n_tool_calls == 0 and rounds <= 1:
            goal = _truncate_with_notice(task.get("text", ""), 200)
            summary_text = (
                f"Task {task_id} ({task.get('type', 'user')}): "
                f"{goal}. {rounds}r, {cost_text}." + project_thread_note_for_task(task)
            )
            append_jsonl(drive_logs / "chat.jsonl", {
                "ts": utc_now_iso(), "direction": "system",
                "type": "task_summary", "task_id": task_id, "text": summary_text,
                "chat_id": int(task.get("chat_id") or 0),
                "tool_calls": n_tool_calls, "rounds": rounds,
                "outcome_axes": outcome_axes, "reason_code": reason_code,
                **_summary_row_cost_fields(usage),
                **({"review_projection": review_projection} if review_projection.get("panels") else {}),
            })
            return

        summary_model, summary_use_local = _consolidation_route()
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
            cost_text=cost_text,
            usage_snapshot=_synthesis_usage_snapshot_text(usage),
            sealed_final=sealed_final_prompt_section(sealed_final),
            trace_summary=_truncate_with_notice(trace, 3000),
            review_evidence=review_section,
        )
        try:
            msg, _usage = llm.chat(messages=[{"role": "user", "content": prompt}],
                                   model=summary_model,
                                   reasoning_effort=CONSOLIDATION_REASONING_EFFORT,
                                   max_tokens=16384,
                                   use_local=summary_use_local)
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
                f"{_truncate_with_notice(goal, 200)}. {rounds}r, {cost_text}."
            )
        if summary_text:
            summary_text += project_thread_note_for_task(task)
            append_jsonl(drive_logs / "chat.jsonl", {
                "ts": utc_now_iso(), "direction": "system",
                "type": "task_summary", "task_id": task_id, "text": summary_text,
                "chat_id": int(task.get("chat_id") or 0),
                "tool_calls": n_tool_calls, "rounds": rounds,
                "outcome_axes": outcome_axes, "reason_code": reason_code,
                **_summary_row_cost_fields(usage),
                **({"review_projection": review_projection} if review_projection.get("panels") else {}),
            })
    except Exception:
        log.debug("Task summary generation failed (non-critical)", exc_info=True)


def _run_chat_consolidation(env, memory, llm, task, drive_logs):
    """Run dialogue-block consolidation inside the root post-task worker."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate
        consolidate = _c.consolidate
        chat_path = drive_logs / "chat.jsonl"
        blocks_path = env.drive_path("memory") / "dialogue_blocks.json"
        meta_path = env.drive_path("memory") / "dialogue_meta.json"
        if should_consolidate(meta_path, chat_path):
            _id, _ident, _llm, _logs = task.get("id"), memory.load_identity(), llm, drive_logs
            from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

            base_scope = current_usage_scope()
            chat_scope = (
                replace(base_scope, category="consolidation", source="chat_consolidation")
                if base_scope is not None
                else UsageScope(
                    drive_root=task.get("budget_drive_root") or env.drive_root,
                    task_id=str(_id or ""),
                    root_task_id=str(task.get("root_task_id") or _id or ""),
                    category="consolidation",
                    source="chat_consolidation",
                )
            )

            with usage_scope(chat_scope):
                u = consolidate(chat_path=chat_path, blocks_path=blocks_path,
                                meta_path=meta_path, llm_client=_llm, identity_text=_ident)
            if u:
                append_jsonl(_logs / "events.jsonl", {"ts": utc_now_iso(),
                    "type": "chat_block_consolidation", "task_id": _id,
                    "cost_usd": (
                        round(float(u["cost"]), 6)
                        if u.get("cost") is not None
                        else None
                    )})
                if u.get("cost") or u.get("prompt_tokens"):
                    from supervisor.state import update_budget_from_usage
                    update_budget_from_usage(u)
    except Exception:
        log.warning("Chat block consolidation setup failed", exc_info=True)


def _run_scratchpad_consolidation(env: Any, memory: Any, llm: Any) -> None:
    """Run scratchpad consolidation inside the root post-task worker."""
    try:
        from ouroboros import consolidator as _c

        should_consolidate = _c.should_consolidate_scratchpad
        consolidate = _c.consolidate_scratchpad
        if should_consolidate(memory):
            kb_dir = env.drive_path("memory/knowledge")
            _identity = memory.load_identity()
            from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

            base_scope = current_usage_scope()
            scratch_scope = (
                replace(base_scope, category="consolidation", source="scratchpad_consolidation")
                if base_scope is not None
                else UsageScope(
                    drive_root=env.drive_root,
                    category="consolidation",
                    source="scratchpad_consolidation",
                )
            )

            with usage_scope(scratch_scope):
                u = consolidate(memory, kb_dir, llm, _identity)
            if u and (u.get("cost") or u.get("prompt_tokens")):
                from supervisor.state import update_budget_from_usage
                update_budget_from_usage(u)
    except Exception:
        log.debug("Scratchpad consolidation setup failed", exc_info=True)


def _run_reflection(env: Any, llm: Any, task: Dict[str, Any],
                    usage: Dict[str, Any], llm_trace: Dict[str, Any],
                    review_evidence: Dict[str, Any],
                    sealed_final: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """Run execution reflection synchronously (process memory, Bible P1)."""
    try:
        from ouroboros.reflection import (
            should_generate_reflection, generate_reflection, append_reflection_routed,
        )
        synthesis_cost = _synthesis_cost_usd(usage)
        if should_generate_reflection(
            llm_trace,
            task=task,
            rounds=int(usage.get("rounds", 0)),
            cost_usd=synthesis_cost,
        ):
            trace_summary = build_trace_summary(llm_trace)
            child_evidence = _child_task_evidence(env, task)
            try:
                reflection_usage = dict(usage)
                # Reflection's legacy durable cost_usd field now records this
                # same subtree snapshot instead of silently reverting to own cost.
                reflection_usage["cost"] = synthesis_cost
                entry = generate_reflection(
                    task, llm_trace, trace_summary,
                    llm, reflection_usage,
                    review_evidence=review_evidence,
                    child_evidence=child_evidence,
                    usage_snapshot_text=_synthesis_usage_snapshot_text(usage),
                    sealed_final_text=sealed_final_prompt_section(sealed_final),
                )
                append_reflection_routed(env, task, entry)
                return entry
            except Exception:
                log.warning("Execution reflection failed (non-critical)", exc_info=True)
    except Exception:
        log.debug("Execution reflection setup failed", exc_info=True)
    return None
