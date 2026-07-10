"""Post-task result emission, memory work, reflections, and review context."""

from __future__ import annotations

import json
import logging
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
    EXECUTION_BEST_EFFORT,
    EXECUTION_FAILED,
    EXECUTION_INFRA_FAILED,
    EXECUTION_OK,
    apply_receipt_absent_flag,
    artifact_bundle_from_result,
    build_verification_ledger,
    derive_loop_outcome,
    maybe_write_verification_artifact,
    normalize_outcome_axes,
)
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.subagents import build_subagent_envelope
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_review_artifact as _truncate_with_notice

log = logging.getLogger(__name__)


# Credential-aware model selection lives in the provider registry SSOT.
from ouroboros.provider_models import resolve_credentialed_model as _resolve_task_summary_model


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


def _build_swarm_efficiency(env: Any, task: Dict[str, Any]) -> Dict[str, Any] | None:
    """Compact derived swarm-efficiency rollup for a task that fanned out subagents.

    Computed from the durable ``swarm_fanout`` telemetry this task already emits
    (control.py:_emit_swarm_fanout): the number of children, the number of fan-out
    waves, the summed inter-wave latency, and the set of effective model lanes used.
    Returns None for a plain task (no fan-out), so the block only appears on real
    swarms.

    OMITTED (no reliable structured source today): ``observed_max_concurrency`` —
    child task results carry only ``ts``/``updated_at``, not a per-child running-start
    vs finish timestamp, so true overlap cannot be derived honestly here — and
    ``parent_blocked_wait_sec`` (wait_task returns prose, not a typed duration).
    """
    task_id = str(task.get("id") or task.get("task_id") or "")
    if not task_id:
        return None
    try:
        from ouroboros.utils import iter_jsonl_objects

        drive_root = getattr(env, "drive_root", None)
        if drive_root is None:
            return None
        events_path = pathlib.Path(drive_root) / "logs" / "events.jsonl"
        child_ids: set[str] = set()
        wave_count = 0
        inter_wave_latency_total = 0.0
        lanes: list[str] = []
        # Read the FULL per-task events stream (not a tail window): the swarm_fanout
        # events can occur EARLY in a long fan-out task, so a bounded tail would
        # silently undercount waves/children (P1 no-silent-loss). This runs once at
        # finalization (not a hot path) and only for fan-out tasks.
        for ev in iter_jsonl_objects(events_path):
            if ev.get("type") != "swarm_fanout":
                continue
            if str(ev.get("parent_task_id") or ev.get("task_id") or "") != task_id:
                continue
            wave_count += 1
            for tid in ev.get("task_ids") or []:
                if str(tid or "").strip():
                    child_ids.add(str(tid))
            try:
                inter_wave_latency_total += float(ev.get("inter_wave_latency_sec") or 0.0)
            except (TypeError, ValueError):
                pass
            for lane in ev.get("effective_model_lanes") or []:
                if str(lane or "").strip() and str(lane) not in lanes:
                    lanes.append(str(lane))
        if not child_ids:
            return None
        return {
            "subagent_count": len(child_ids),
            "wave_count": wave_count,
            "inter_wave_latency_sec_total": round(inter_wave_latency_total, 3),
            "lanes_used": lanes,
        }
    except Exception:
        log.debug("swarm efficiency rollup failed", exc_info=True)
        return None


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


def _run_post_task_processing_async(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    review_evidence: Dict[str, Any],
    drive_logs: pathlib.Path,
    *,
    blocking: bool = False,
) -> Dict[str, Any] | None:
    """Run best-effort LLM-heavy post-task memory work off the reply path."""
    task_snapshot = json.loads(json.dumps(task, ensure_ascii=False, default=str))
    usage_snapshot = json.loads(json.dumps(usage, ensure_ascii=False, default=str))
    trace_snapshot = json.loads(json.dumps(llm_trace, ensure_ascii=False, default=str))
    review_evidence_snapshot = json.loads(json.dumps(review_evidence, ensure_ascii=False, default=str))

    result: Dict[str, Any] = {}

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
            result["reflection_entry"] = reflection_entry
            from ouroboros.project_facts import resolve_project_id

            _pid = resolve_project_id(task_snapshot)
            # Project facts stay project-scoped, but the improvement backlog is
            # Ouroboros's GLOBAL queue of lessons about its own tooling. A
            # workspace task can reveal generic friction (bad tools, poor
            # prompts, broken lifecycle) that should feed post-task evolution
            # without leaking project facts into canonical memory.
            _update_improvement_backlog(env, reflection_entry)
            _apply_reflection_memory_actions(env, reflection_entry, project_id=_pid)
            try:
                from ouroboros.post_task_evolution import maybe_promote

                maybe_promote(env, task_snapshot, reflection_entry, llm_client)
            except Exception:
                log.debug("Post-task evolution promotion failed", exc_info=True)
        except Exception:
            log.warning("Async post-task processing failed", exc_info=True)

    if blocking:
        _run()
        return result.get("reflection_entry")
    threading.Thread(target=_run, daemon=True).start()
    return None


def _run_global_backlog_promotion_only(
    env: Any,
    task: Dict[str, Any],
    reflection_entry: Dict[str, Any] | None,
    llm: Any,
) -> None:
    """Feed canonical improvement backlog/promotion without leaking project memory."""

    if not reflection_entry:
        return
    try:
        candidates = [
            item for item in (reflection_entry.get("backlog_candidates") or [])
            if isinstance(item, dict) and str(item.get("summary") or "").strip()
        ]
        if not candidates:
            return
        sanitized_entry = {
            "reflection": "\n".join(f"- {str(item.get('summary') or '').strip()}" for item in candidates),
            "backlog_candidates": candidates,
            "memory_actions": [],
        }
        _update_improvement_backlog(env, sanitized_entry)
        from ouroboros.post_task_evolution import maybe_promote

        global_task = {
            "id": str(task.get("id") or ""),
            "type": str(task.get("type") or "task"),
            "source": "project_scoped_global_improvement",
            "metadata": {"globalized_from_project_task": True},
        }
        maybe_promote(env, global_task, sanitized_entry, llm)
    except Exception:
        log.debug("Canonical post-task promotion-only path failed", exc_info=True)


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
    # FR3 observability: apply the receipt_absent / expected_output_ungrounded objective-axis
    # flag HERE — once — so the SAME flagged loop_outcome feeds the task_eval / task_metrics /
    # task_done event stream (the day-1 monitoring metric reads it) AND the durable
    # task_result.json. _store_task_result reuses this loop_outcome (single source), so the
    # flag is no longer applied to a second, independently-derived outcome the events never saw.
    apply_receipt_absent_flag(
        loop_outcome, llm_trace, getattr(env, "drive_root", None), str(task.get("id") or ""),
        expected_output=str(task.get("expected_output") or ""),
    )
    outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
    execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
    reason_code = str(loop_outcome.get("reason_code") or "")
    # CW3 (v6.34.0): a short same-route "turn=decision" turn (ephemeral, run while the
    # main agent is busy) DELIVERS its inline answer but must not leave a durable TASK
    # RECORD \u2014 no task_result file, no task_eval ledger row. The cognitive-memory writes
    # (reflection/consolidation/letters-home) are already gated further below; this
    # closes the remaining durable task-record writes. The answer + card resolution
    # (send_message/task_done) and budget metrics still flow so the reply is visible.
    _ephemeral = bool(task.get("_ephemeral_turn"))
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
    if not _ephemeral:
        try:
            append_jsonl(drive_logs / "events.jsonl", {
                "ts": utc_now_iso(), "type": "task_eval", "ok": execution_status not in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED},
                "task_id": task.get("id"), "task_type": task.get("type"),
                "outcome_axes": outcome_axes,
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
        "outcome_axes": outcome_axes,
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

    if not _ephemeral:
        _store_task_result(env, task, text, usage, llm_trace, review_evidence=review_evidence, loop_outcome=loop_outcome)
        stored_result = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    else:
        # No durable task_result file for a transient decision turn; the card still
        # resolves via task_done below (with empty artifact/review status).
        stored_result = {}
    artifact_bundle = stored_result.get("artifact_bundle") if isinstance(stored_result.get("artifact_bundle"), dict) else {}
    pending_events.append({
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        # CW3: tells the supervisor's task_done handler to NOT synthesize a durable
        # missing-result task_result for a transient decision turn (which has none).
        "_ephemeral": _ephemeral,
        # Carry the thread so the terminal card finalizes in its project panel
        # (per-thread fan-out), not just the main chat.
        "chat_id": int(task.get("chat_id") or 0),
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "artifact_status": stored_result.get("artifact_status") or artifact_bundle.get("status") or "",
        "artifact_bundle": artifact_bundle,
        "review_status": stored_result.get("review_status") if isinstance(stored_result.get("review_status"), dict) else {},
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        # v6.57.0 (P6b): recursive cost incl. children (from the stored rollup) so the
        # parent card / Logs can show the true subtree cost, not just this task's own.
        "cost_usd_with_children": stored_result.get("cost_usd_with_children"),
        "cost_with_children_partial": bool(stored_result.get("cost_with_children_partial")),
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
        post_usage["outcome_axes"] = outcome_axes
        post_usage["reason_code"] = reason_code
        # Ephemeral same-route turns (the "turn=decision" anti-freeze path while the
        # main agent is busy) are PROHIBITED from ALL durable memory: not only
        # reflection/evolution (below) but chat/scratchpad consolidation and project
        # letters-home too — the locked main path owns those (v6.33.0 WS10
        # idempotency contract; claudexor B5). ``_ephemeral`` is computed once near
        # the top of this function (it also gates the durable task-record writes).
        if not _ephemeral:
            _run_chat_consolidation(env, memory, llm, task, drive_logs)
            _run_scratchpad_consolidation(env, memory, llm)
        from ouroboros.project_facts import resolve_project_id

        _project_scoped = bool(resolve_project_id(task))
        # A project THREAD conversation runs on the fast direct-chat lane. It is
        # project-scoped only for CONTEXT (it sees the project's knowledge/
        # journal), but it is NOT a pooled task completion: it must not block the
        # reply on LLM post-processing and must not write letters home (that
        # would turn every "как дела?" into a journal milestone + a consciousness
        # observation and stall the global chat lock). Only real pooled project
        # tasks get the letters-home + blocking treatment.
        _is_direct_chat = bool(task.get("_is_direct_chat"))
        _project_task = _project_scoped and not _is_direct_chat and not _ephemeral
        if _project_task:
            # Letters home (v6.32.0): record the cycle in the project's own
            # journal and emit a concise completion digest for consciousness
            # (project_id + full objective + outcome). Full project awareness:
            # this is a crisp "task finished" summary, not an isolation boundary —
            # the one mind already sees the project thread in its unified memory;
            # only per-cycle RAW internal facts stay in the per-project store.
            _pid = resolve_project_id(task)
            # The full objective IS the meaning of the cycle — carry it whole into
            # the journal milestone and the consciousness digest (BIBLE P1: no
            # silent/lossy clip of cognitive text). Objectives are concise by
            # nature; the task and task_results remain the durable record.
            _objective = str(
                task.get("objective") or task.get("description") or task.get("text") or ""
            )
            _exec_status = str((outcome_axes.get("execution") or {}).get("status") or "unknown")
            try:
                # Route through the shared bounded helper so the auto-milestone
                # honors the journal's durable per-row contract (over-limit gets a
                # VISIBLE pointer, never a silent slice or a raw unbounded append).
                from ouroboros.tools.project_journal import append_journal_milestone

                append_journal_milestone(
                    _pid,
                    # Compare against the canonical execution-axis constants, not raw
                    # "success"/"best_effort" — the axis value for a clean finish is
                    # EXECUTION_OK ("ok"), so the old literal never matched and every
                    # successful task was journaled as "blocked" (C9.1 seed bug).
                    "done" if _exec_status in (EXECUTION_OK, EXECUTION_BEST_EFFORT) else "blocked",
                    f"Task finished ({_exec_status}): {_objective}",
                    task_id=str(task.get("id") or ""),
                )
            except Exception:
                log.debug("project journal task-done entry failed", exc_info=True)
            # F2 (v6.39): when the SWARM ROOT (a top-level project task — no parent) finishes,
            # mirror its ephemeral task-tree ledger's durable-worthy coordination (attention
            # beacons + interface contracts) into the durable project journal once, so the
            # swarm's blockers/contracts survive the tree GC. Subagents skip this (the root
            # absorbs the whole tree); the helper no-ops when there is no ledger.
            if not str(task.get("parent_task_id") or "").strip():
                try:
                    from ouroboros.tools.project_journal import mirror_tree_coordination_to_journal

                    mirror_tree_coordination_to_journal(
                        _pid,
                        str(task.get("root_task_id") or task.get("id") or ""),
                        task_id=str(task.get("id") or ""),
                    )
                except Exception:
                    log.debug("project journal swarm-coordination mirror failed", exc_info=True)
            try:
                pending_events.append({
                    "type": "project_digest",
                    "project_id": _pid,
                    "task_id": str(task.get("id") or ""),
                    "objective": _objective,
                    "execution_status": _exec_status,
                    "objective_status": str((outcome_axes.get("objective") or {}).get("status") or "not_evaluated"),
                    "ts": utc_now_iso(),
                })
            except Exception:
                log.debug("project digest emission failed", exc_info=True)
        # LLM-heavy memory work stays off the reply critical path; ephemeral turns
        # skip reflection/evolution too (see the idempotency note above).
        if _ephemeral:
            reflection_entry = None
        else:
            reflection_entry = _run_post_task_processing_async(
                env, task, post_usage, llm_trace, review_evidence, drive_logs,
                blocking=(
                    str(task.get("type") or "") == "evolution"
                    or bool(str(task.get("workspace_root") or "").strip())
                    or bool(str(task.get("workspace_mode") or "").strip())
                    or _project_task
                ),
            )
        budget_drive_root = str(task.get("budget_drive_root") or "").strip()
        # Leak guard (Phase 3b / red-team R3.1): a project-scoped task must NEVER
        # write its learnings to the canonical parent drive — project facts live
        # only in the per-project store. Suppress the canonical dual-run for it.
        if (
            budget_drive_root
            and str(pathlib.Path(budget_drive_root).resolve(strict=False)) != str(pathlib.Path(env.drive_root).resolve(strict=False))
        ):
            from types import SimpleNamespace

            parent_env = SimpleNamespace(repo_dir=env.repo_dir, drive_root=pathlib.Path(budget_drive_root), drive_path=lambda rel: pathlib.Path(budget_drive_root) / rel)
            parent_task = {**task, "drive_root": budget_drive_root, "child_drive_root": str(env.drive_root)}
            if _project_scoped:
                _run_global_backlog_promotion_only(parent_env, parent_task, reflection_entry, llm)
            else:
                _run_post_task_processing_async(
                    parent_env,
                    parent_task,
                    post_usage,
                    llm_trace,
                    review_evidence,
                    pathlib.Path(budget_drive_root) / "logs",
                    blocking=True,
                )


def _store_task_result(env: Any, task: Dict[str, Any], text: str,
                       usage: Dict[str, Any], llm_trace: Dict[str, Any],
                       review_evidence: Dict[str, Any] | None = None,
                       loop_outcome: Dict[str, Any] | None = None) -> None:
    """Store task result for parent task retrieval.

    ``loop_outcome``, when supplied by ``emit_task_results``, is the SINGLE already-
    derived, already-receipt_absent-flagged outcome that also fed the task_eval /
    task_metrics event stream — so the persisted axes match the events exactly and we
    do not derive/flag a second time. It is only re-derived here when called without one.
    """
    try:
        trace_summary = build_trace_summary(llm_trace)
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
        if loop_outcome is None:
            loop_outcome = derive_loop_outcome(text or "", usage, llm_trace)
            # FR3: inject durable verification receipts into the trace and flag
            # receipt_absent on a clean-but-unverified effects turn — BEFORE normalize so
            # the persisted axes and the ledger agree (claudexor lockstep fix).
            apply_receipt_absent_flag(
                loop_outcome, llm_trace, env.drive_root, str(task.get("id") or ""),
                expected_output=str(task.get("expected_output") or ""),
            )
        outcome_axes = normalize_outcome_axes({"outcome_axes": loop_outcome.get("outcome_axes")})
        execution_status = str((outcome_axes.get("execution") or {}).get("status") or "")
        reason_code = str(loop_outcome.get("reason_code") or "")
        status = (
            STATUS_FAILED
            if str(existing.get("status") or "") == STATUS_FAILED
            or execution_status in {EXECUTION_FAILED, EXECUTION_INFRA_FAILED}
            else STATUS_COMPLETED
        )
        task_contract = build_task_contract(task)
        task = {**task, "task_contract": task_contract}
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
        outcome_axes = dict(outcome_axes)
        existing_artifact_axis = (
            (existing.get("outcome_axes") or {}).get("artifacts")
            if isinstance(existing.get("outcome_axes"), dict)
            else {}
        )
        artifact_axis = dict(existing_artifact_axis) if isinstance(existing_artifact_axis, dict) else {}
        if isinstance(outcome_axes.get("artifacts"), dict):
            artifact_axis.update(outcome_axes.get("artifacts") or {})
        artifact_axis["status"] = str(artifact_bundle.get("status") or artifact_axis.get("status") or "not_applicable")
        outcome_axes["artifacts"] = artifact_axis
        # B1: compact swarm-efficiency rollup, only for a task that actually fanned
        # out subagents (None for a plain task -> kwarg omitted).
        swarm_efficiency = _build_swarm_efficiency(env, task)
        subagent_envelope = task.get("subagent_envelope") if isinstance(task.get("subagent_envelope"), dict) else {}
        if str(task.get("delegation_role") or "").lower() == "subagent":
            subagent_envelope = build_subagent_envelope(
                task_id=str(task.get("id") or ""),
                parent_task_id=str(task.get("parent_task_id") or ""),
                root_task_id=str(task.get("root_task_id") or ""),
                task_group_id=str(task.get("task_group_id") or ""),
                depth=int(task.get("depth") or 0),
                role=str(task.get("role") or ""),
                requested_lane=str(task.get("requested_model_lane") or task.get("model_lane") or "auto"),
                effective_lane=str(task.get("effective_model_lane") or task.get("model_lane") or "light"),
                model=str(task.get("model") or ""),
                status=status,
                usage={
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "rounds": int(usage.get("rounds") or 0),
                },
                cost_usd=round(float(usage.get("cost") or 0), 6),
            )
        # P6b (v6.57.0): recursive cost rollup INCLUDING children. cost_usd stays this
        # task's own spend (unchanged for existing consumers); cost_usd_with_children is
        # the additive parent-visible total, computed from the canonical status root
        # (budget_drive_root) where child results live.
        _own_cost = round(float(usage.get("cost") or 0), 6)
        _cost_status_root = task.get("budget_drive_root") or env.drive_root
        try:
            from ouroboros.task_status import compute_cost_with_children

            _cost_with_children, _cost_partial = compute_cost_with_children(
                _cost_status_root, str(task.get("id") or ""), _own_cost
            )
        except Exception:
            _cost_with_children, _cost_partial = _own_cost, False
        write_task_result(
            env.drive_root,
            str(task.get("id") or ""),
            status,
            reason_code=reason_code,
            outcome_axes=outcome_axes,
            cost_usd_with_children=_cost_with_children,
            cost_with_children_partial=_cost_partial,
            task_contract=task_contract,
            loop_outcome=loop_outcome,
            project_id=str(task.get("project_id") or ""),
            parent_task_id=task.get("parent_task_id"),
            root_task_id=task.get("root_task_id"),
            session_id=task.get("session_id"),
            actor_id=task.get("actor_id"),
            delegation_role=task.get("delegation_role"),
            role=task.get("role"),
            description=task.get("description"),
            objective=task.get("objective") or task.get("description"),
            title=task.get("title"),
            expected_output=task.get("expected_output"),
            constraints=task.get("constraints"),
            context=task.get("context"),
            workspace_root=task.get("workspace_root"),
            workspace_mode=task.get("workspace_mode"),
            memory_mode=task.get("memory_mode"),
            drive_root=task.get("drive_root"),
            child_drive_root=task.get("child_drive_root") or task.get("drive_root"),
            budget_drive_root=task.get("budget_drive_root"),
            task_constraint=task.get("task_constraint"),
            model_lane=task.get("model_lane"),
            requested_model_lane=task.get("requested_model_lane"),
            effective_model_lane=task.get("effective_model_lane"),
            model=task.get("model"),
            use_local_model=task.get("use_local_model"),
            task_group_id=task.get("task_group_id"),
            task_group=task.get("task_group"),
            subagent_envelope=subagent_envelope,
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=text or "",
            final_answer=str(loop_outcome.get("final_answer") or ""),
            trace_summary=trace_summary,
            trace_refs=loop_outcome.get("trace_refs") or {},
            cost_usd=round(float(usage.get("cost") or 0), 6),
            total_rounds=int(usage.get("rounds") or 0),
            review_evidence=review_evidence or {},
            verification_ledger=verification_refs.get("inline"),
            artifact_bundle=artifact_bundle,
            artifacts=artifacts,
            **({"swarm_efficiency": swarm_efficiency} if swarm_efficiency else {}),
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
        outcome_axes = normalize_outcome_axes(usage)
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
                "chat_id": int(task.get("chat_id") or 0),
                "tool_calls": n_tool_calls, "rounds": rounds,
                "outcome_axes": outcome_axes, "reason_code": reason_code,
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
                "chat_id": int(task.get("chat_id") or 0),
                "tool_calls": n_tool_calls, "rounds": rounds,
                "outcome_axes": outcome_axes, "reason_code": reason_code,
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
