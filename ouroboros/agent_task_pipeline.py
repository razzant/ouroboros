"""Post-task result emission, memory work, reflections, and review context."""

from __future__ import annotations

import json
import functools
import logging
import pathlib
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List

from ouroboros.task_results import (
    TASK_COST_META_FIELDS,
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
from ouroboros.subagents import envelope_from_task
from ouroboros.utils import utc_now_iso, append_jsonl, truncate_review_artifact as _truncate_with_notice
from ouroboros.post_task_checkpoint import (
    POST_TASK_SYNTHESIS_INFLIGHT as _POST_TASK_SYNTHESIS_INFLIGHT,
    POST_TASK_SYNTHESIS_LOCK as _POST_TASK_SYNTHESIS_LOCK,
    is_root_post_task as _is_root_post_task,
    root_checkpoint_roots as _root_checkpoint_roots,
    root_post_task_already_completed as _root_post_task_already_completed,
    set_root_post_task_checkpoint as _set_root_post_task_checkpoint,
)

log = logging.getLogger(__name__)


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


def _build_swarm_efficiency(env: Any, task: Dict[str, Any]) -> Dict[str, Any] | None:
    """Compact derived swarm-efficiency rollup for a task that fanned out subagents.

    Computed from the durable ``swarm_fanout`` telemetry this task already emits
    (control.py:_emit_swarm_fanout): the number of children, the number of fan-out
    waves, the summed inter-wave latency, and the set of model lanes REQUESTED —
    fanout events are written before any child starts, so effective lanes are not
    knowable here; they live on each child's own dispatch record.
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
            # The lane a wave ASKED for. A fan-out event is written before any child
            # starts, so it cannot know what they ran on — that is a per-child
            # dispatch fact and lives on each child's own record.
            lane = str(ev.get("requested_model_lane") or "").strip()
            if lane and lane not in lanes:
                lanes.append(lane)
        if not child_ids:
            return None
        return {
            "subagent_count": len(child_ids),
            "wave_count": wave_count,
            "inter_wave_latency_sec_total": round(inter_wave_latency_total, 3),
            "lanes_requested": lanes,
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


def _synthesis_cost_usd(usage: Dict[str, Any]) -> float | None:
    """Prefer the subtree snapshot; preserve legacy callers without one."""
    key = "cost_usd_with_children" if "cost_usd_with_children" in usage else "cost"
    value = usage.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


def _synthesis_cost_text(usage: Dict[str, Any]) -> str:
    cost = _synthesis_cost_usd(usage)
    if cost is None:
        return "cost unavailable (non-final)" if "cost_usd_with_children" in usage else "cost unknown"
    if bool(usage.get("cost_with_children_partial")):
        return f"${cost:.2f} subtree cost snapshot (non-final)"
    return f"${cost:.2f}"


_SYNTHESIS_USAGE_PROMPT_FIELDS = (
    "cost_usd_with_children",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "unknown_unmetered",
    "ledger_integrity",
    "cost_snapshot_at",
    "cost_final",
    "cost_with_children_partial",
    "cost_accounting_status",
    "reason_code",
    "outcome_axes",
)


def _synthesis_usage_snapshot_text(usage: Dict[str, Any]) -> str:
    """Render the bounded root snapshot section shared by synthesis prompts."""
    if not (
        "cost_usd_with_children" in usage
        and str(usage.get("cost_snapshot_at") or "").strip()
        and usage.get("cost_final") is False
        and usage.get("cost_with_children_partial") is True
    ):
        return ""
    projection = {
        field: usage.get(field)
        for field in _SYNTHESIS_USAGE_PROMPT_FIELDS
    }
    payload = json.dumps(projection, ensure_ascii=False, indent=2, default=str)
    return (
        "## Shared pre-synthesis cost and outcome snapshot\n"
        "`cost_usd_with_children` is accounted subtree cost only. `reserved_usd` and\n"
        "`unresolved_upper_bound_usd` are separate non-final exposure fields; do not add\n"
        "them to or describe them as already included in the accounted total. This snapshot is non-final:\n"
        "summary/reflection calls happen after it. Never turn null/unavailable values into zero.\n"
        "`outcome_axes` is canonical task truth: never describe objective best_effort,\n"
        "degraded, or fail — or a non-pass review axis — as clean success.\n"
        f"{payload}\n\n"
    )


def _compact_review_projection(llm_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Build the public review projection without copying raw actor output."""
    try:
        from ouroboros.review_substrate import compact_review_projection

        return compact_review_projection(llm_trace.get("review_runs") or [])
    except Exception:
        log.debug("Failed to build compact review projection", exc_info=True)
        return {"panels": []}


def _run_post_task_processing_async(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    review_evidence: Dict[str, Any],
    drive_logs: pathlib.Path,
    *,
    blocking: bool = False,
    on_reflection: Callable[[Dict[str, Any] | None, Any], None] | None = None,
) -> Dict[str, Any] | None:
    """Run best-effort LLM-heavy post-task memory work off the reply path."""
    task_snapshot = json.loads(json.dumps(task, ensure_ascii=False, default=str))
    trace_snapshot = json.loads(json.dumps(llm_trace, ensure_ascii=False, default=str))
    review_evidence_snapshot = json.loads(json.dumps(review_evidence, ensure_ascii=False, default=str))

    result: Dict[str, Any] = {}
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    base_scope = current_usage_scope()
    if base_scope is not None:
        post_scope = replace(
            base_scope, category="post_task", source="post_task_synthesis",
        )
    else:
        task_id = str(task_snapshot.get("id") or "")
        post_scope = UsageScope(
            drive_root=task_snapshot.get("budget_drive_root") or env.drive_root,
            task_id=task_id,
            root_task_id=str(task_snapshot.get("root_task_id") or task_id),
            parent_task_id=str(task_snapshot.get("parent_task_id") or ""),
            category="post_task",
            source="post_task_synthesis",
        )

    post_task_key: tuple[str, str] | None = None
    if _is_root_post_task(task_snapshot):
        task_id = str(task_snapshot.get("id") or task_snapshot.get("task_id") or "")
        roots = _root_checkpoint_roots(env, task_snapshot)
        root_key = str(pathlib.Path(
            task_snapshot.get("budget_drive_root")
            or (roots[0] if roots else env.drive_root)
        ).resolve(strict=False))
        post_task_key = (root_key, task_id)
        with _POST_TASK_SYNTHESIS_LOCK:
            if post_task_key in _POST_TASK_SYNTHESIS_INFLIGHT:
                return None
            _POST_TASK_SYNTHESIS_INFLIGHT.add(post_task_key)
        _set_root_post_task_checkpoint(env, task_snapshot, "running")

    # Freeze one honest subtree view synchronously at the pre-synthesis
    # boundary. No scoped worker or consolidation/model call may start first;
    # summary and reflection then receive this same object, while the existing
    # terminal checkpoint remains the only final accounting authority.
    usage_snapshot = _pre_synthesis_usage_snapshot(env, task_snapshot, usage)

    def _run_scoped() -> None:
        checkpoint_status = "degraded"
        try:
            from ouroboros.llm import LLMClient
            from ouroboros.memory import Memory

            llm_client = LLMClient()
            task_memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
            # All late model work belongs to this one scoped worker.  This keeps
            # the root checkpoint non-final until consolidation, summary,
            # reflection, and promotion have all stopped billing.
            _run_chat_consolidation(env, task_memory, llm_client, task_snapshot, drive_logs)
            _run_scratchpad_consolidation(env, task_memory, llm_client)
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
            if on_reflection is not None:
                on_reflection(reflection_entry, llm_client)
            checkpoint_status = "completed"
        except Exception:
            log.warning("Async post-task processing failed", exc_info=True)
        finally:
            _set_root_post_task_checkpoint(env, task_snapshot, checkpoint_status)
            if post_task_key is not None:
                with _POST_TASK_SYNTHESIS_LOCK:
                    _POST_TASK_SYNTHESIS_INFLIGHT.discard(post_task_key)

    def _run() -> None:
        with usage_scope(post_scope):
            _run_scoped()

    if blocking:
        _run()
        return result.get("reflection_entry")
    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        _set_root_post_task_checkpoint(env, task_snapshot, "degraded")
        if post_task_key is not None:
            with _POST_TASK_SYNTHESIS_LOCK:
                _POST_TASK_SYNTHESIS_INFLIGHT.discard(post_task_key)
        raise
    return None


def recover_pending_root_post_task_synthesis(
    drive_root: Any, repo_dir: Any = None,
) -> int:
    """Resume an undispatched root synthesis; degrade an indeterminate one.

    The existing task-result checkpoint is the only durable authority.  The
    process-local in-flight set prevents the periodic reconciler from spawning a
    duplicate while the recovered thread is alive.  After restart only
    ``pending_once`` is replay-safe: ``running`` may have crossed a paid provider
    boundary, so it becomes terminal ``degraded`` instead of repeating calls.
    """
    from types import SimpleNamespace
    from ouroboros.task_results import list_task_results

    root = pathlib.Path(drive_root).resolve(strict=False)
    try:
        rows = list_task_results(root)
    except Exception:
        return 0
    recovered = 0
    for stored in rows:
        task_id = str(stored.get("task_id") or stored.get("id") or "")
        checkpoint = stored.get("root_phase_checkpoint")
        phase = str(checkpoint.get("post_task_synthesis") or "") if isinstance(checkpoint, dict) else ""
        if not task_id or phase not in {"pending_once", "running"}:
            continue
        task = {**stored, "id": task_id, "root_task_id": str(stored.get("root_task_id") or task_id)}
        task.setdefault("budget_drive_root", str(root))
        if not _is_root_post_task(task):
            continue
        env = SimpleNamespace(
            repo_dir=pathlib.Path(repo_dir or task.get("repo_dir") or root.parent),
            drive_root=root,
            drive_path=lambda rel, _root=root: _root / rel,
        )
        if phase == "running":
            degraded = dict(checkpoint)
            degraded.update({
                "post_task_synthesis": "degraded",
                "post_task_stop_reason": "restart_indeterminate_running",
            })
            write_task_result(
                root, task_id, str(task.get("status") or STATUS_COMPLETED),
                root_phase_checkpoint=degraded,
            )
            _set_root_post_task_checkpoint(env, task, "refresh")
            recovered += 1
            continue
        usage = {
            "cost": float(task.get("cost_usd") or 0),
            "rounds": int(task.get("total_rounds") or 0),
            "reason_code": str(task.get("reason_code") or ""),
            "outcome_axes": task.get("outcome_axes") or {},
        }
        trace = {
            "tool_calls": [],
            "reasoning_notes": [str(task.get("trace_summary") or "")],
            "recovered_post_task_synthesis": True,
        }
        _run_post_task_processing_async(
            env, task, usage, trace,
            task.get("review_evidence") if isinstance(task.get("review_evidence"), dict) else {},
            root / "logs", blocking=False,
        )
        recovered += 1
    return recovered


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


def _attach_host_mutation_projection(
    env: Any,
    task: Dict[str, Any],
    llm_trace: Dict[str, Any],
) -> None:
    """Bind durable mutation evidence to the existing outcome trace seam."""
    from ouroboros.mutation_attribution import (
        attribution_task_id,
        load_mutation_evidence_projection,
        record_terminal_mutation_candidates,
    )

    task_id = str(task.get("id") or "").strip()
    root_task_id = str(task.get("root_task_id") or task_id).strip()
    task_ids = list(dict.fromkeys(item for item in (root_task_id, task_id) if item))
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    evidence_root = pathlib.Path(
        task.get("budget_drive_root")
        or metadata.get("budget_drive_root")
        or getattr(env, "budget_drive_root", None)
        or env.drive_root
    )
    # Only the owning root task refreshes its terminal candidate snapshot at
    # outcome derivation; a subagent never rewrites the root's evidence.
    if task_id and task_id == root_task_id:
        try:
            if attribution_task_id(evidence_root, (task_id,)) == task_id:
                record_terminal_mutation_candidates(evidence_root, task_id)
        except Exception:
            log.debug("terminal mutation snapshot failed for %s", task_id, exc_info=True)
    llm_trace.pop("mutation_attribution", None)
    for candidate in task_ids:
        projection = load_mutation_evidence_projection(evidence_root, candidate)
        if projection:
            llm_trace["mutation_attribution"] = projection
            return


def _derive_host_bound_loop_outcome(
    env: Any,
    task: Dict[str, Any],
    text: str,
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive once from the current durable mutation-evidence binding."""
    _attach_host_mutation_projection(env, task, llm_trace)
    return derive_loop_outcome(text or "", usage, llm_trace)


def emit_task_results(
    env: Any, memory: Any, llm: Any,
    pending_events: List[Dict[str, Any]],
    task: Dict[str, Any], text: str,
    usage: Dict[str, Any], llm_trace: Dict[str, Any],
    start_time: float, drive_logs: pathlib.Path,
    ctx: Any = None,
) -> None:
    """Emit all end-of-task events to supervisor and run post-task processing."""
    loop_outcome = _derive_host_bound_loop_outcome(env, task, text, usage, llm_trace)
    # FR3 observability: apply the receipt_absent / expected_output_ungrounded objective-axis
    # flag HERE — once — so the SAME flagged loop_outcome feeds events and the durable
    # task_result.json. _store_task_result reuses this loop_outcome, so the
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
    # closes the remaining durable task-record writes. An inline answer + card
    # resolution (send_message/task_done) and budget metrics still flow so the reply
    # is visible. A typed routing receipt is presentation metadata attached to the
    # owner's message, not a replacement for the agent's conversational answer.
    # Agent.run_task normalizes blank model output before this boundary, so every
    # finalized response follows the same durable send_message path.
    _ephemeral = bool(task.get("_ephemeral_turn"))
    _typed_routing_action = (
        str(getattr(ctx, "_typed_routing_action_emitted", "") or "").strip()
        if ctx is not None else ""
    )
    _has_typed_routing_action = bool(_ephemeral and _typed_routing_action)
    _decision_meta = {"ephemeral_decision": True} if _ephemeral else {}
    send_event = {
        "type": "send_message", "chat_id": task["chat_id"],
        "text": text or "\u200b", "log_text": text or "",
        "format": "markdown",
        "task_id": task.get("id"), "ts": utc_now_iso(),
    }
    if _decision_meta:
        # MessageBus already carries progress_meta on both progress and final
        # frames.  The Web client uses this only to suppress the transient
        # task card; the inline conversational answer itself remains visible.
        send_event["progress_meta"] = dict(_decision_meta)
    pending_events.append(send_event)

    duration_sec = round(time.time() - start_time, 3)
    n_tool_calls = len(llm_trace.get("tool_calls", []))
    n_tool_errors = sum(1 for tc in llm_trace.get("tool_calls", [])
                        if isinstance(tc, dict) and tc.get("is_error"))
    try:
        from supervisor.state import reconstruct_task_cost

        task_cost_fields = reconstruct_task_cost(
            str(task.get("id") or ""), fields=True,
            drive_root=pathlib.Path(task.get("budget_drive_root") or env.drive_root),
        )
    except Exception:
        log.error("Task cost authority unavailable at finalization", exc_info=True)
        task_cost_fields = _unavailable_cost_fields("ledger_unavailable")
    if _is_root_post_task(task) and not _root_post_task_already_completed(env, task):
        task_cost_fields["cost_final"] = False
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
        "ephemeral_decision": _ephemeral,
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "duration_sec": duration_sec,
        "tool_calls": n_tool_calls, "tool_errors": n_tool_errors,
        **task_cost_fields,
        **({"resource_limit": dict(usage.get("resource_limit") or {})}
           if isinstance(usage.get("resource_limit"), dict) else {}),
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
        _store_task_result(
            env, task, text, usage, llm_trace, review_evidence=review_evidence,
            loop_outcome=loop_outcome, result_fields={**task_cost_fields, "duration_sec": duration_sec},
        )
        stored_result = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
    else:
        # No durable task_result file for a transient decision turn; the card still
        # resolves via task_done below (with empty artifact/review status).
        stored_result = {}
    artifact_bundle = stored_result.get("artifact_bundle") if isinstance(stored_result.get("artifact_bundle"), dict) else {}
    review_projection = stored_result.get("review_projection") or {}
    pending_events.append({
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        # CW3: tells the supervisor's task_done handler to NOT synthesize a durable
        # missing-result task_result for a transient decision turn (which has none).
        "_ephemeral": _ephemeral,
        # Presentation marker only. The supervisor's typed routing event remains
        # the action/receipt authority; this keeps a transient decision task from
        # becoming a visible task card (and a bogus "Turn into project" target).
        "ephemeral_decision": _ephemeral,
        **({"typed_routing_action": _typed_routing_action} if _has_typed_routing_action else {}),
        # Carry the thread so the terminal card finalizes in its project panel
        # (per-thread fan-out), not just the main chat.
        "chat_id": int(task.get("chat_id") or 0),
        "outcome_axes": outcome_axes,
        "reason_code": reason_code,
        "artifact_status": stored_result.get("artifact_status") or artifact_bundle.get("status") or "",
        "artifact_bundle": artifact_bundle,
        "review_status": stored_result.get("review_status") if isinstance(stored_result.get("review_status"), dict) else {},
        **({"review_projection": review_projection} if review_projection.get("panels") else {}),
        **task_cost_fields,
        # v6.57.0 (P6b): recursive cost incl. children (from the stored rollup) so the
        # parent card / Logs can show the true subtree cost, not just this task's own.
        "cost_usd_with_children": stored_result.get("cost_usd_with_children"),
        "cost_with_children_partial": bool(stored_result.get("cost_with_children_partial")),
        **({"resource_limit": dict(usage.get("resource_limit") or {})}
           if isinstance(usage.get("resource_limit"), dict) else {}),
        "ts": utc_now_iso(),
    })
    # NOTE: task_done is NOT written to events.jsonl here.
    # It goes through EVENT_Q → supervisor _handle_task_done → append_jsonl.
    # This ensures causal ordering: send_message reaches the UI before task_done,
    # preventing the live card from collapsing before the assistant reply arrives.
    restart_reason = str(getattr(ctx, "pending_restart_reason", "") or "").strip()
    evolution_restart = bool(getattr(ctx, "pending_restart_is_evolution", False))
    if restart_reason:
        if not evolution_restart:
            pending_events.append({
                "type": "restart_request", "reason": restart_reason,
                "evolution_restart": False, "ts": utc_now_iso(),
            })
        ctx.pending_restart_reason = None
        ctx.pending_restart_is_evolution = False

    if _is_root_post_task(task):
        post_usage = dict(usage or {})
        post_usage["outcome_axes"] = outcome_axes
        post_usage["reason_code"] = reason_code
        # Ephemeral same-route turns (the "turn=decision" anti-freeze path while the
        # main agent is busy) are PROHIBITED from ALL durable memory: not only
        # reflection/evolution (below) but chat/scratchpad consolidation and project
        # letters-home too — the locked main path owns those (v6.33.0 WS10
        # idempotency contract; claudexor B5). ``_ephemeral`` is computed once near
        # the top of this function (it also gates the durable task-record writes).
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
        budget_drive_root = str(task.get("budget_drive_root") or "").strip()
        split_drive = bool(
            budget_drive_root
            and str(pathlib.Path(budget_drive_root).resolve(strict=False)) != str(pathlib.Path(env.drive_root).resolve(strict=False))
        )
        parent_env = None
        parent_task = None
        if split_drive:
            from types import SimpleNamespace

            parent_env = SimpleNamespace(repo_dir=env.repo_dir, drive_root=pathlib.Path(budget_drive_root), drive_path=lambda rel: pathlib.Path(budget_drive_root) / rel)
            parent_task = {**task, "drive_root": budget_drive_root, "child_drive_root": str(env.drive_root)}

        global_reflection_callback = None
        if split_drive and _project_scoped and parent_env is not None and parent_task is not None:
            global_reflection_callback = functools.partial(
                _run_global_backlog_promotion_only, parent_env, parent_task)

        if not _ephemeral and not _root_post_task_already_completed(env, task):
            if split_drive and not _project_scoped and parent_env is not None and parent_task is not None:
                _run_post_task_processing_async(
                    parent_env,
                    parent_task,
                    post_usage,
                    llm_trace,
                    review_evidence,
                    pathlib.Path(budget_drive_root) / "logs",
                    blocking=True,
                )
            else:
                _run_post_task_processing_async(
                    env, task, post_usage, llm_trace, review_evidence, drive_logs,
                    blocking=(
                        str(task.get("type") or "") == "evolution"
                        or bool(str(task.get("workspace_root") or "").strip())
                        or bool(str(task.get("workspace_mode") or "").strip())
                        or _project_task
                    ),
                    on_reflection=global_reflection_callback,
                )


def _unavailable_cost_fields(error: str) -> Dict[str, Any]:
    """Fail-closed cost projection: ONE shape for both paths that can lose the ledger."""
    return {"cost_accounting_status": "unavailable", "cost_final": False,
            "cost_accounting_error": error, "cost_usd": None, "total_rounds": None,
            "prompt_tokens": None, "completion_tokens": None, "reserved_usd": None,
            "unresolved_upper_bound_usd": None, "unknown_unmetered": None}


def _store_task_result(env: Any, task: Dict[str, Any], text: str,
                       usage: Dict[str, Any], llm_trace: Dict[str, Any],
                       review_evidence: Dict[str, Any] | None = None,
                       loop_outcome: Dict[str, Any] | None = None,
                       result_fields: Dict[str, Any] | None = None) -> None:
    """Store task result for parent task retrieval.

    ``loop_outcome``, when supplied by ``emit_task_results``, is the SINGLE already-
    derived, already-receipt_absent-flagged outcome that also fed the task_eval /
    task_metrics event stream — so the persisted axes match the events exactly and we
    do not derive/flag a second time. It is only re-derived here when called without one.
    ``result_fields`` bundles the cost projection PLUS ``duration_sec`` (not a tenth
    parameter — the signature was already at the <8 limit); elapsed is written by THIS
    terminal writer only, so cancel/crash results show Unavailable, never a fake 0.
    """
    try:
        trace_summary = build_trace_summary(llm_trace)
        cost_fields = dict(result_fields or {})
        duration_sec = cost_fields.pop("duration_sec", None)
        cost_fields = cost_fields or _unavailable_cost_fields("ledger_projection_missing")
        existing = load_task_result(env.drive_root, str(task.get("id") or "")) or {}
        if loop_outcome is None:
            loop_outcome = _derive_host_bound_loop_outcome(env, task, text, usage, llm_trace)
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
        # B1: swarm-efficiency rollup, only for a task that fanned out (None -> omitted).
        swarm_efficiency = _build_swarm_efficiency(env, task)
        subagent_envelope = task.get("subagent_envelope") if isinstance(task.get("subagent_envelope"), dict) else {}
        if str(task.get("delegation_role") or "").lower() == "subagent":
            subagent_envelope = envelope_from_task(
                task, status=status, usage=usage, cost_usd=cost_fields.get("cost_usd"))
            if cost_fields.get("cost_accounting_status") != "available":
                subagent_envelope.update({
                    "cost_usd": None,
                    "cost_accounting_status": "unavailable",
                    "cost_final": False,
                })
        # Compatibility projection only. The physical-attempt ledger is the sole
        # monetary authority; the supervisor replaces the root projection with the
        # exact subtree total at terminal handling. Do not independently walk task
        # results here: that was a second accounting engine and could double-count.
        _own_cost = cost_fields.get("cost_usd")
        _cost_with_children = _own_cost
        _cost_partial = True
        root_phase_checkpoint: Dict[str, Any] = {}
        if _is_root_post_task(task):
            incoming_checkpoint = llm_trace.get("root_phase_checkpoint")
            existing_checkpoint = existing.get("root_phase_checkpoint")
            if isinstance(existing_checkpoint, dict) and existing_checkpoint.get("post_task_synthesis") in {"completed", "degraded"}:
                root_phase_checkpoint = dict(existing_checkpoint)
            elif isinstance(incoming_checkpoint, dict):
                root_phase_checkpoint = dict(incoming_checkpoint)
            elif isinstance(existing_checkpoint, dict):
                root_phase_checkpoint = dict(existing_checkpoint)
            else:
                root_phase_checkpoint = {
                    "phase": "task_acceptance",
                    "status": "not_required",
                    "pass_index": 0,
                }
            root_phase_checkpoint.setdefault("post_task_synthesis", "pending_once")
        review_projection = _compact_review_projection(llm_trace)
        write_task_result(
            env.drive_root,
            str(task.get("id") or ""),
            status,
            reason_code=reason_code,
            outcome_axes=outcome_axes,
            # Compatibility mirror consumed by the gateway and task_done event.
            # ``outcome_axes.review`` remains the canonical structured axis.
            review_status=dict(outcome_axes.get("review") or {}),
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
            parent_model_lane=task.get("parent_model_lane"),
            requested_executor=task.get("requested_executor"),
            effective_model_lane=task.get("effective_model_lane"),
            model=task.get("model"),
            use_local_model=task.get("use_local_model"),
            effective_executor=task.get("effective_executor"),
            executor_route=task.get("executor_route"),
            tool_profile=task.get("tool_profile"),
            capability_delta=task.get("capability_delta"),
            reasoning_effort=task.get("reasoning_effort"),
            task_group_id=task.get("task_group_id"),
            task_group=task.get("task_group"),
            subagent_envelope=subagent_envelope,
            metadata=task.get("metadata") if isinstance(task.get("metadata"), dict) else {},
            result=text or "",
            final_answer=str(loop_outcome.get("final_answer") or ""),
            trace_summary=trace_summary,
            trace_refs=loop_outcome.get("trace_refs") or {},
            **({"duration_sec": float(duration_sec)} if isinstance(duration_sec, (int, float)) and not isinstance(duration_sec, bool) else {}),
            **cost_fields,
            review_evidence=review_evidence or {},
            **({"review_projection": review_projection} if review_projection.get("panels") else {}),
            verification_ledger=verification_refs.get("inline"),
            artifact_bundle=artifact_bundle,
            artifacts=artifacts,
            **({"root_phase_checkpoint": root_phase_checkpoint} if root_phase_checkpoint else {}),
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
Rounds: {rounds}, Cost: {cost_text}

{usage_snapshot}
## Execution trace
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


def _run_task_summary(env, llm, task, usage, llm_trace, drive_logs, review_evidence=None):
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
                    review_evidence: Dict[str, Any]) -> Dict[str, Any] | None:
    """Run execution reflection synchronously (process memory, Bible P1)."""
    try:
        from ouroboros.reflection import (
            should_generate_reflection, generate_reflection, append_reflection,
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
        from ouroboros.task_continuation import (
            list_review_continuations,
            retire_settled_continuations_for_context,
        )
        from ouroboros.task_results import load_task_result

        state = load_state(pathlib.Path(env.drive_root))
        retired = retire_settled_continuations_for_context(
            env.drive_root, state, lambda tid: load_task_result(env.drive_root, tid))
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

        if retired:
            lines.append(f"- {len(retired)} settled continuation(s) archived out of context "
                         "(durable under state/review_continuations/archived/): "
                         + ", ".join(retired[:5])
                         + (f" (+{len(retired) - 5} more)" if len(retired) > 5 else ""))
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
