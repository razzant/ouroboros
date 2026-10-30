"""Post-task synthesis workers for the task pipeline (v7 L-C2 split).

The best-effort work the post-task orchestrator
(``agent_task_pipeline._run_post_task_processing_async``) dispatches after a
task ends: the tool-trace summary, the free host facts row and the fallback
memory draft or candidate, plus the shared pre-synthesis usage snapshot and
the compact review projection. No stage writes lessons or knowledge notes:
the acting mind authors those (``knowledge_write``); no stage rewrites the
scratchpad: the mind folds it (#918, ``update_scratchpad`` with ``replaces``).
Extracted from agent_task_pipeline.py; the pipeline re-exports every name, so
historical imports and monkeypatch targets keep working."""

from __future__ import annotations

import json
import logging
import pathlib

from dataclasses import replace
from typing import Any, Dict
from ouroboros.dialogue_provenance import presence_provenance_fields
from ouroboros.llm_claudexor import propagate_model_error
from ouroboros.observability import without_finalization_timing
from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.subagent_messages import initiator_meta
from ouroboros.synthesis_cost_text import _summary_row_cost_fields
from ouroboros.tool_capabilities import routing_action_for_tool
from ouroboros.utils import append_jsonl, truncate_review_artifact as _truncate_with_notice, utc_now_iso


log = logging.getLogger("ouroboros.agent_task_pipeline")


def _atp():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros import agent_task_pipeline

    return agent_task_pipeline


def task_tool_metrics(llm_trace: dict) -> dict:
    """Project recorded calls once; unknown names never become an empty census."""
    unavailable = bool(llm_trace.get("loop_evidence_unavailable"))
    calls = llm_trace.get("tool_calls") or []
    metrics = {
        "tool_calls": None if unavailable else len(calls),
        "tool_errors": None if unavailable else sum(
            1 for call in calls if isinstance(call, dict) and call.get("is_error")),
        # The addressing calls among them (tool_capabilities owns the family), so
        # a replayed block can tell a receipt-only turn from real work without
        # a client list of tool names.
        "routing_tool_calls": None if unavailable else sum(
            1 for call in calls if isinstance(call, dict) and routing_action_for_tool(call.get("tool"))),
        "completion_tool_calls": None if unavailable else sum(
            1 for call in calls if isinstance(call, dict) and call.get("completion_control") is True and not call.get("is_error")),
        "tool_call_counts": None,
    }
    if unavailable or llm_trace.get("recovered_post_task_synthesis") or not isinstance(llm_trace.get("tool_calls"), list):
        return metrics
    counts: dict[str, int] = {}
    for call in calls:
        name = call.get("tool") if isinstance(call, dict) else None
        if not isinstance(name, str) or not name.strip():
            return metrics
        name = name.strip()
        counts[name] = counts.get(name, 0) + 1
    metrics["tool_call_counts"] = counts
    return metrics


def capture_task_inputs(ctx: Any, task: dict, drive_root: Any, receipts: list) -> dict:
    """Freeze the existing owner corpus and check receipts before actor cleanup."""
    from ouroboros._outcome_receipts import verification_receipt_ledger_row
    from ouroboros.review_evidence_sections import (
        _accept_owner_directives, _accept_verification_summary,
    )

    task_id = str(task.get("id") or task.get("task_id") or "")
    result: dict = {
        "version": 1, "task_id": task_id,
        "source_ref": {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"},
        "unavailable_sections": [],
    }
    try:
        # The run's provenance comes first: a reader learns who started the run
        # and whether the owner door stamped it before it reads the first text.
        from ouroboros.dialogue_provenance import run_origin

        metadata = getattr(ctx, "task_metadata", None)
        result["run_origin"] = run_origin({
            **task, "metadata": metadata if isinstance(metadata, dict) else task.get("metadata"),
        })
    except Exception:
        result["unavailable_sections"].append("run_origin")
        log.warning("Task run origin unavailable for synthesis: %s", task_id, exc_info=True)
    try:
        result["owner_requirements_and_decisions"] = _accept_owner_directives(ctx, drive_root, task_id)
    except Exception:
        result["unavailable_sections"].append("owner_requirements_and_decisions")
        log.warning("Task owner input unavailable for synthesis: %s", task_id, exc_info=True)
    try:
        result["verification_summary"] = _accept_verification_summary(receipts)
        result["verification_receipts"] = [
            {"ts": row.get("ts"), **verification_receipt_ledger_row(row)}
            for row in receipts if isinstance(row, dict)
        ]
    except Exception:
        result["unavailable_sections"].append("verification_receipts")
        log.warning("Task verification input unavailable for synthesis: %s", task_id, exc_info=True)
    # Copy once into the normal durable completion package. Prompt workers must
    # neither retain mutable actor objects nor re-read a later task's messages.
    return json.loads(json.dumps(result, ensure_ascii=False, default=str))



def build_trace_summary(llm_trace: dict) -> str:
    """Return the bounded, human-readable preview of tool calls and agent notes.

    It is what is stored and shown (task card, parents, children): two arguments per
    call, a positional window past thirty calls, a total cut. Values stay width-bounded,
    with a disclosed omission; the trace itself keeps every call.
    """
    if llm_trace.get("loop_evidence_unavailable"):
        return "## Tool trace (call count unknown)\nThe failed loop supplied no verified execution trace."
    tool_calls = llm_trace.get("tool_calls", []) or []
    notes = llm_trace.get("reasoning_notes", []) or []

    n = len(tool_calls)
    # Honest breakdown so a task that finished with a deliverable is not mislabeled
    # "43 errors": separate GENUINE unresolved errors from POLICY denials, cosmetic
    # non-zero exits, recovered errors, and ignored read-only blocks.
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
        def _fmt_call(index: int, tc: dict) -> str:
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
            if status:
                facts.append(f"status={status}")
            if tc.get("exit_code") is not None:
                facts.append(f"exit_code={tc.get('exit_code')}")
            if tc.get("signal"):
                facts.append(f"signal={tc.get('signal')}")
            fact_suffix = f" [{', '.join(facts)}]" if facts else ""
            suffix = " → ERROR" if tc.get("is_error") else ""
            return f"{index + 1}. {name}({args_str}){fact_suffix}{suffix}"

        if n > 30:
            shown = (
                [_fmt_call(i, tool_calls[i]) for i in range(15)]
                + [f"⚠️ OMISSION NOTE: {n - 30} middle tool calls omitted from trace summary."]
                + [_fmt_call(n - 15 + i, tool_calls[n - 15 + i]) for i in range(15)]
            )
        else:
            shown = [_fmt_call(i, tool_calls[i]) for i in range(n)]
        lines.extend(shown)

    if notes:
        lines.append("\n## Agent notes (supplementary, not source of truth)")
        lines.extend(f"- {note}" for note in notes)

    summary = "\n".join(lines)
    if len(summary) > 4000:
        summary = _truncate_with_notice(summary, 4000)
    return summary


def _pre_synthesis_usage_snapshot(
    env: Any,
    task: Dict[str, Any],
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    """Freeze one honest, non-final root/subtree cost view for the post-task phase.

    The facts row reads this loop-local dictionary. The existing terminal
    checkpoint remains the sole final authority after later model calls settle.
    """
    snapshot = json.loads(json.dumps(without_finalization_timing(usage), ensure_ascii=False, default=str))
    if not _atp()._is_root_post_task(task):
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
        from ouroboros.cost_projection import COST_SCOPE_ROOT_TREE, build_cost_presentation

        logical_root_id = str(task.get("root_task_id") or task_id)
        subtree = usage_breakdown(budget_root, root_task_id=logical_root_id, custody_root=env.drive_root, include_custody=True)
        snapshot.update({
            "accounted_upper_bound_usd_with_children": round(float(subtree["accounted_usd"]), 6),
            "cost_presentation": build_cost_presentation(subtree, scope=COST_SCOPE_ROOT_TREE),
            "cost_custody": subtree.get("cost_custody"),
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
            "accounted_upper_bound_usd_with_children": None,
            "reserved_usd": None,
            "unresolved_upper_bound_usd": None,
            "unknown_unmetered": None,
            "ledger_integrity": "unavailable",
            "cost_accounting_status": "unavailable",
            "cost_presentation": None,
        })
    return snapshot


def _compact_review_projection(llm_trace: Dict[str, Any], task: Dict[str, Any] | None = None,
                               drive_root: Any = None) -> Dict[str, Any]:
    """Build the public review projection without copying raw actor output: the trace's
    acceptance runs, then this task's own review-ledger records (``task_ledger_records``)."""
    try:
        from ouroboros.review_projection import task_ledger_records
        from ouroboros.review_substrate import compact_review_projection

        records, omitted = task_ledger_records(task or {}, drive_root)
        projection = compact_review_projection(llm_trace.get("review_runs") or [], records=records)
        if records or omitted == "1+":
            projection["review_records_omitted"] = omitted
        return projection
    except Exception:
        log.debug("Failed to build compact review projection", exc_info=True)
        return {"panels": []}


def _record_task_facts(env: Any, task: Dict[str, Any], usage: Dict[str, Any],
                       llm_trace: Dict[str, Any], drive_logs: pathlib.Path) -> None:
    """Append the task's free host facts row to chat.jsonl: no model call, no prose.

    The owner removed the paid task narrative (TZ-2 decision 2=A). This row keeps
    the facts its readers take from a ``task_summary`` row when the stored result is
    gone: the direct-turn fact, origin label, typed routing action, tool metrics,
    rounds, cost and review projection. Its kind is ``host_task_facts``, never
    ``authored_root_summary``, and it persists no continuation narrative, so a
    Main continuation sees the typed ``continuation_narrative_unavailable`` gap.
    Text stays empty: existing Project and result references own navigation.
    """
    task_id = str(task.get("id") or "unknown")
    try:
        from ouroboros.project_dialogue import append_canonical_task_summary, completion_status_label, outcome_phase
        from ouroboros.task_finalization import artifact_store_roots, rescued_files_fact

        canonical_root = pathlib.Path(task.get("budget_drive_root") or drive_logs.parent)
        result_root = pathlib.Path(getattr(env, "drive_root", canonical_root))
        stored_result = _atp().load_task_result(result_root, task_id) or {}
        review_projection = _compact_review_projection(llm_trace, task, canonical_root)
        # TZ-2 C2: how many files the task rescued into its store(s) — positive, zero or
        # unknown — by stat alone; the fact discloses that no hash was computed. A split
        # non-Project root synthesizes on the canonical drive (parent env and task): its
        # actor store is then the row's recorded ``child_drive_root``, else this drive.
        canonical = result_root.resolve(strict=False) == canonical_root.resolve(strict=False)
        files_rescued = rescued_files_fact(task_id, artifact_store_roots(
            canonical_root, task_id, task=task, child_root=None if canonical else result_root))
        append_canonical_task_summary(canonical_root, retained=True, row={
            "ts": utc_now_iso(), "direction": "system", "type": "task_summary",
            "summary_kind": "host_task_facts", "summary_id": f"task-facts:{task_id}",
            "task_id": task_id, "parent_task_id": str(task.get("parent_task_id") or ""), "root_task_id": str(task.get("root_task_id") or task_id),
            "project_id": str(task.get("project_id") or ""), "chat_id": int(task.get("chat_id") or 0), "delegation_role": str(task.get("delegation_role") or ""), "role": str(task.get("role") or ""),
            "status": str(stored_result.get("status") or "completed"), "outcome": completion_status_label(stored_result, usage), "outcome_phase": outcome_phase(stored_result, usage),
            "outcome_final": False, "outcome_authority": "pre_finalization_host_facts",
            # The chat block reads its chrome, the addressing fact and the
            # origin label from this row when the task result has been pruned.
            "_is_direct_chat": bool(task.get("_is_direct_chat")), **initiator_meta(task),
            **({"typed_routing_action": str(usage["typed_routing_action"])} if usage.get("typed_routing_action") else {}),
            "text": "", **task_tool_metrics(llm_trace),
            "rounds": usage.get("rounds"),
            "outcome_axes": normalize_outcome_axes(usage), "reason_code": str(usage.get("reason_code") or ""),
            "result_ref": {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"},
            "files_rescued": files_rescued,
            **_summary_row_cost_fields(usage), **presence_provenance_fields(task),
            **({"review_projection": review_projection} if review_projection.get("panels") or review_projection.get("gaps") or review_projection.get("review_records_omitted") else {}),
        })
    except Exception:
        log.warning("Task facts row was not recorded for %s (non-critical)", task_id, exc_info=True)


POST_TASK_INTERRUPT_KINDS = frozenset({"budget_exhausted", "provider_outcome_unknown"})


def propagate_paid_interruption(error: BaseException) -> None:
    """Re-raise what must stop later paid post-work; return for an ordinary failure.

    ``propagate_model_error`` carries the control and typed unknown-provider facts;
    the wallet's ``BudgetExceeded`` and an unresolved attempt on ANY provider's
    exception chain — the consolidator's own classifier, never a broadened global
    one — are the others (TZ-2 C3). A stage adapter that only logged them let the
    coordinator run the next paid stage and write ``completed``. Anything else
    returns, so the caller isolates the failure to its own stage.
    """
    propagate_model_error(error)
    from ouroboros.transport_custody import outcome_unknown_on_chain
    from ouroboros.usage_accounting import BudgetExceeded

    if isinstance(error, BudgetExceeded) or outcome_unknown_on_chain(error):
        raise error


def post_task_interruption(control: BaseException) -> str:
    """The closed stop-reason word for a control that ended paid post-work.

    A model wait carries its ``control_reason``; a typed provider fact keeps its
    code; an unknown outcome (a dispatched attempt without a terminal provider
    fact) is always ``provider_outcome_unknown`` — the transport's own
    ``model_outcome_unknown`` spelling never reaches the checkpoint.
    """
    reason = str(getattr(control, "control_reason", "") or "")
    if reason:
        return reason
    code = str(getattr(control, "code", "") or "")
    from ouroboros.transport_custody import outcome_unknown_on_chain

    if not code or code == "model_outcome_unknown" or outcome_unknown_on_chain(control):
        return "provider_outcome_unknown"
    return code


def _post_task_paid_interruption(errors: Any) -> str:
    """The stage's typed outcome from returned error facts: '' when clean.

    Memory consolidation returns errors to keep completed chunks. Only this
    stage adapter interprets those existing facts: a budget or unknown-provider
    kind wins and stops later paid post-work; any other kind names the last
    UNRESOLVED ordinary failure, so a stage that lost a chunk reads ``degraded``
    like a stage that raised (TZ-2 C3: unfinished stages are never ``completed``).
    The history keeps every attempt; a refusal its producer answered (a split whose
    halves carry their own rows, ``resolution``) is not an unfinished stage.
    """
    rows = [row for row in (errors if isinstance(errors, list) else []) if isinstance(row, dict)]
    for row in rows:
        if row.get("kind") in POST_TASK_INTERRUPT_KINDS:
            return str(row["kind"])
    unresolved = [row for row in rows if not row.get("resolution")]
    return str((unresolved[-1].get("kind") or "stage_error")) if unresolved else ""


def _run_memory_fallback_draft(env: Any, task: Dict[str, Any], llm: Any, drive_logs: pathlib.Path,
                               llm_trace: Dict[str, Any]) -> str:
    """Run the fallback memory draft (``memory_fallback``) inside the root post-task worker.

    At most one Light call drafts an acting record (consciousness off) or an unsealed
    failed-work candidate (on), billed as consolidation; a refusal with a receipt and a
    returned failure read ``degraded``, a budget or unknown-provider kind interrupts.
    """
    try:
        from ouroboros import memory_fallback
        from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

        base_scope = current_usage_scope()
        scope = (replace(base_scope, category="consolidation", source="memory_fallback") if base_scope is not None
                 else UsageScope(drive_root=env.drive_root, category="consolidation", source="memory_fallback"))
        with usage_scope(scope):
            run = memory_fallback.run_fallback_draft(env, task, llm, drive_logs, llm_trace)
        usage = run.usage
        if usage and (usage.get("cost") or usage.get("prompt_tokens")):
            from supervisor.state import update_budget_from_usage
            update_budget_from_usage(usage)
        if run.outcome in ("consciousness_on", "not_activated", "nothing"):
            return ""
        from ouroboros.knowledge import observed_route_stamp

        reason = (run.kind if run.outcome == "refused"
                  else _post_task_paid_interruption(run.errors))
        try:
            append_jsonl(pathlib.Path(drive_logs) / "events.jsonl", {
                "ts": utc_now_iso(), "type": "memory_fallback_draft", "task_id": str(task.get("id") or ""),
                "unit": run.unit.describe() if run.unit else None, "outcome": run.outcome, "kind": run.kind or None,
                "record_id": run.record_id or None, "input_tokens": run.input_tokens, "effect": run.effect or None,
                "accounted_upper_bound_usd": round(float(usage["cost"]), 6) if usage.get("cost") is not None else None,
                "route": observed_route_stamp(usage) if usage else None})
        except Exception:
            log.debug("memory_fallback_draft event was not recorded", exc_info=True)
        return reason
    except Exception as error:
        propagate_paid_interruption(error)
        log.debug("Fallback memory draft setup failed", exc_info=True)
        return "stage_setup_failed"


# --- a Pause an earlier version saved: ended once by the maintenance pass ---

def end_saved_late_phase_pause(env: Any, task: Dict[str, Any], root: pathlib.Path, authority: Dict[str, Any],
                               *, owner_liveness: Any, claim_guard: Any, stop_requested: Any,
                               event_queue: Any) -> bool:
    """End one late phase an earlier version saved as the owner's Pause; True when ended now.

    Today a Pause ends the post-answer stages at once, so this saved remainder
    never runs: its stages are disclosed the way a Pause ends them now
    (``owner_pause:skipped=…``; a Stop that already reached it keeps
    ``owner_stopped``, a stage this version no longer has ``stage_retired``),
    then the tree's Pause settles as after any ended phase, its latch and fence
    released once nothing else remains. The answer, its status and the saved
    payload stay; the exact-record compare makes a repeat a no-op.
    """
    from ouroboros.post_task_checkpoint import (
        post_task_checkpoint_witness, post_task_synthesis_is_terminal,
        release_post_task_synthesis, reserve_post_task_synthesis,
    )

    task_id = str(task["id"])
    checkpoint = authority.get("root_phase_checkpoint") or {}
    saved = checkpoint.get("post_task_pause") if isinstance(checkpoint.get("post_task_pause"), dict) else {}
    skipped = [str(stage) for stage in (saved.get("remaining_stages") or [saved.get("stage")]) if stage]
    retired = [str(stage) for stage in saved.get("retired_stages") or () if stage]
    reason = f"{'owner_stopped' if saved.get('stopped_at') else 'owner_pause'}:skipped={','.join(skipped)}"
    if retired:
        reason += f";stage_retired:skipped={','.join(retired)}"
    key = reserve_post_task_synthesis(root, task_id, owner_liveness=owner_liveness,
                                      claim_guard=claim_guard, stop_requested=stop_requested)
    if key is None:
        return False
    try:
        updated = _atp()._publish_root_post_task_checkpoint(
            env, task, "degraded", expected=post_task_checkpoint_witness(authority), event_queue=event_queue,
            stop_requested=stop_requested, stop_reason=reason)
    finally:
        release_post_task_synthesis(key)
    if not (updated and post_task_synthesis_is_terminal(
            (updated.get("root_phase_checkpoint") or {}).get("post_task_synthesis"))):
        return False
    append_jsonl(pathlib.Path(root) / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": "saved_late_phase_ended", "task_id": task_id, "post_task_stop_reason": reason})
    log.info("Late phase of %s saved by an earlier version ended: %s", task_id, reason)
    from supervisor.owner_pause_control import late_phase_settled
    late_phase_settled(root, task_id)
    return True
