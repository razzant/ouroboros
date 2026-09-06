"""Forced finalization of a task that ran out of road: orphan notes, child claims
and the absorption gate, forced children acceptance, swarm-action enforcement,
forced services and owner-directive drain, the one forced model call, stale and
fallback candidates, the swarm router and the forced final answer.
Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import queue
import time

from typing import Any, Callable, Dict, List, Optional, Tuple
from ouroboros.loop_llm_call import forced_response_is_incomplete, forced_response_parts
from ouroboros.outcomes import REASON_DELIVERY_CONTROL_DEGRADED
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_HOST_SALVAGE, TERMINAL_ORIGIN_MODEL_FINAL
from ouroboros.tool_policy import swarm_router_turn
from ouroboros.tools.registry import ToolRegistry
from ouroboros.usage_accounting import BudgetExceeded
from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_round_limits import _RoundLimitContext


log = logging.getLogger("ouroboros.loop")


def _loop():
    """The parent loop module, read at call time.

    The loop's members stay monkeypatch-addressable at their historical
    ``ouroboros.loop`` bindings (tests rebind them there), so this leaf
    resolves every cross-reference through the module at each call instead
    of freezing whatever object a from-import saw at import time.
    """
    from ouroboros import loop

    return loop


def _load_direct_child_results(
    status_root: pathlib.Path,
    task_id: str,
    root_task_id: str,
) -> list[Dict[str, Any]]:
    """Read this task's direct children (plan review spawns none)."""

    from ouroboros.task_status import find_child_tasks

    return [
        row for row in find_child_tasks(
            pathlib.Path(status_root),
            parent_task_id=task_id,
            root_task_id=root_task_id,
            exclude_task_id=task_id,
            scope="direct",
        )
        if isinstance(row, dict)
    ]


def _direct_child_results(ctx: _RoundLimitContext) -> list[Dict[str, Any]]:
    """Read this node's direct children from the existing task-status authority."""

    try:
        status_root = ctx.status_drive_root or ctx.drive_root or pathlib.Path(ctx.drive_logs).parent
        if status_root is None or not ctx.task_id:
            return []
        return _loop()._load_direct_child_results(
            pathlib.Path(status_root),
            ctx.task_id,
            str(ctx.root_task_id or ctx.task_id),
        )
    except Exception:
        return []


def _child_disposition_state(child: Dict[str, Any]) -> str:
    """Return cancellation or the current task-tree exact-hash disposition."""

    # Explicit cancellation wins every completion race; late scratch results are
    # not recovered. Only a SETTLED ``cancelled`` counts as handled (GR2-8c):
    # ``cancel_requested`` is intent, so such a child stays cancel-pending.
    if (
        str(child.get("parent_decision") or "").strip().lower() == "cancelled"
        and str(child.get("status") or "").strip().lower() == "cancelled"
    ):
        return "cancelled"
    try:
        from ouroboros.tools.join_ledger import _current_child_result_disposition

        current = _current_child_result_disposition(child)
        if current:
            return current
    except Exception:
        pass
    return ""


def _project_child_result_dispositions(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> None:
    """Expose a compact exact-hash projection for acceptance/outcome reducers."""

    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

        current = []
        for child in _loop()._direct_child_results(ctx):
            disposition = _loop()._child_disposition_state(child)
            if disposition not in {"integrated", "irrelevant", "deferred"}:
                continue
            current.append({
                "child_task_id": str(child.get("task_id") or child.get("id") or ""),
                "disposition": disposition,
                "child_result_sha256": _child_result_sha256(child),
            })
        llm_trace["child_result_dispositions"] = {
            "current": current,
            "deferred_count": sum(row["disposition"] == "deferred" for row in current),
        }
    except Exception:
        llm_trace["child_result_dispositions"] = {"current": [], "deferred_count": 0}


def _record_forced_finalization(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    *,
    reason_code: str,
    source: str,
    candidate: Optional[DeliveryCandidate],
) -> None:
    # Forced exits bypass the normal no-tool finalization gate. Project
    # child dispositions here, after services/evidence and the candidate
    # refresh, so every forced return exposes the same terminal child-result
    # truth to the outcome reducer.
    _loop()._project_child_result_dispositions(ctx, llm_trace)
    # Common terminal recorder = the ONE seam over the LLM-seam forced
    # answer (`_forced_final_answer`) and the no-spend host-fallback fence
    # path (`_handle_budget_exceeded` -> `_forced_fallback_result`).
    _loop()._record_forced_acceptance_bypass(ctx, llm_trace, reason_code)
    ctx.accumulated_usage.setdefault("terminal_origin", TERMINAL_ORIGIN_HOST_NOTICE)
    binding = dict(candidate.acceptance_binding or {}) if candidate is not None else {}
    tools = getattr(ctx, "tools", None)
    current_fingerprint = str(
        getattr(getattr(tools, "_ctx", None), "_delivery_evidence_fingerprint", "")
        or ""
    )
    current_revision = int(
        getattr(getattr(tools, "_ctx", None), "_delivery_evidence_revision", 0)
        or 0
    )
    llm_trace["forced_finalization"] = {
        "reason_code": reason_code,
        "source": source,
        "degraded": True,
        "candidate_sha256": candidate.content_sha256 if candidate is not None else "",
        "candidate_revision": candidate.revision if candidate is not None else None,
        "evidence_revision": candidate.evidence_revision if candidate is not None else None,
        "current_evidence_revision": current_revision,
        "evidence_current": bool(
            candidate is not None
            and candidate.evidence_fingerprint == current_fingerprint
        ),
        "acceptance_status": str(binding.get("acceptance_status") or "unaccepted"),
        "acceptance_authoritative": bool(binding.get("authoritative", False)),
    }


def _forced_orphan_note(ctx: _RoundLimitContext, *, include_terminal: bool = True) -> str:
    """A bounded note listing children the parent did NOT explicitly handle
    (discard/cancel), appended to a finalization so paid child work is never
    SILENTLY orphaned (P1; P5 — no prose parsing). On a FORCED finalization
    (deadline / provider death / finalize_now, ``include_terminal=True``) the
    parent may not have seen completions: RUNNING and COMPLETED-undecided are
    both reported. On a NORMAL no-tool finalization
    (``include_terminal=False``) the agent saw every change, so only
    STILL-RUNNING undecided children — genuinely orphaned by finalizing
    mid-flight — are reported. Never raises."""
    try:
        from ouroboros.task_status import FINAL_STATUSES

        children = _loop()._direct_child_results(ctx)
        claimed = _loop()._claimed_child_dispositions(ctx)

        def _undecided(c: Dict[str, Any]) -> bool:
            if _loop()._child_disposition_state(c) in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }:
                return False  # explicitly handled
            # completed children were already surfaced via the reminder
            return include_terminal or str(c.get("status") or "").strip().lower() not in FINAL_STATUSES

        undecided = [c for c in children if _undecided(c)]
        deferred = [c for c in children if _loop()._child_disposition_state(c) == "deferred"]

        def _label(c: Dict[str, Any]) -> str:
            tid = str(c.get("task_id") or c.get("id") or "?")
            st = str(c.get("status") or "?").strip().lower()
            lifecycle = "running" if st not in FINAL_STATUSES else st
            # W2: a child whose latest decision row no longer binds the current
            # result was read and decided — say that, not "unread" (the row
            # exists, the binding did not). Only for children left UNDECIDED:
            # a carried disposition is no failed binding.
            claim = claimed.get(tid) if not _loop()._child_disposition_state(c) else None
            if claim is not None:
                disposition, row_sha = claim
                from ouroboros.tools.join_ledger import _child_result_sha256

                if _child_result_sha256(c) != row_sha:
                    detail = (
                        f"{disposition} recorded for an EARLIER result hash; the current "
                        "result is not bound — re-inspect and re-submit the current hash"
                    )
                else:
                    detail = (
                        f"{disposition} recorded for this exact result hash but not carried "
                        "by this round's disposition projection — re-submit to close it"
                    )
                return f"{tid} [{lifecycle}; {detail}]"
            terminal = str(c.get("child_status") or "").strip().lower()
            if terminal and terminal != st:
                return f"{tid} [{lifecycle}; terminal_result={terminal}]"
            return f"{tid} [{lifecycle}]"

        notes: list[str] = []
        if undecided:
            listed = "; ".join(_label(c) for c in undecided[:10])
            more = f" (+{len(undecided) - 10} more)" if len(undecided) > 10 else ""
            lead = "finalized under a hard limit with" if include_terminal else "finalized with"
            detail = (
                "running ones may be incomplete, completed ones may be UNREAD"
                if include_terminal else
                "still-running children not absorbed or discarded"
            )
            notes.append(
                f"\n\n⚠️ NOTE: {lead} {len(undecided)} child task(s) not explicitly absorbed or "
                f"discarded — {detail}: {listed}{more}. Inspect with get_task_result(<id>) / "
                f"peek_task(<id>)."
            )
        if deferred:
            listed = "; ".join(_label(c) for c in deferred[:10])
            more = f" (+{len(deferred) - 10} more)" if len(deferred) > 10 else ""
            notes.append(
                f"\n\n⚠️ DEFERRED CHILD RESULTS: {listed}{more}. These exact results were "
                "explicitly deferred, so this answer is degraded/best-effort rather than clean solved."
            )
        return "".join(notes)
    except Exception:
        return ""


def _claimed_child_dispositions(ctx: _RoundLimitContext) -> Dict[str, tuple]:
    """task_id -> (disposition, row_sha) from THIS parent's latest blackboard
    decision rows (W2). Consulted only for children the disposition projection
    left undecided: a row that exists but no longer binds is audit evidence of a
    claimed-but-failed disposition write, and the forced orphan note must say so
    instead of calling the child unread. Pure read, never raises."""
    try:
        from ouroboros.task_tree_ledger import CHILD_RESULT_DISPOSITION_TYPE, tree_ledger_rows

        status_root = (
            getattr(ctx, "status_drive_root", None)
            or getattr(ctx, "drive_root", None)
        )
        root_id = str(getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or "")
        parent_id = str(getattr(ctx, "task_id", "") or "")
        if status_root is None or not root_id or not parent_id:
            return {}
        claims: Dict[str, tuple] = {}
        for row in tree_ledger_rows(root_id, data_root=pathlib.Path(status_root)):
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if (
                str(row.get("kind") or "") == "decision"
                and str(payload.get("type") or "") == CHILD_RESULT_DISPOSITION_TYPE
                and str(row.get("task_id") or "") == parent_id
                and str(payload.get("child_task_id") or "")
            ):
                # Later rows win: the ledger is append-only and the newest decision
                # is the one whose failure to bind is worth naming.
                claims[str(payload["child_task_id"])] = (
                    str(payload.get("disposition") or ""),
                    str(payload.get("child_result_sha256") or ""),
                )
        return claims
    except Exception:
        return {}


def _undispositioned_children(ctx: _RoundLimitContext) -> list[Dict[str, Any]]:
    try:
        return [
            child for child in _loop()._direct_child_results(ctx)
            if _loop()._child_disposition_state(child) not in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }
        ]
    except Exception:
        return []


def _undecided_children_listing(undecided: list[Dict[str, Any]]) -> str:
    """Bounded ``id [status] sha256`` listing shared by the absorption
    reminder and the forced-finalization prompt."""

    from ouroboros.tools.join_ledger import _child_result_sha256

    return "; ".join(
        f"{c.get('task_id') or c.get('id') or '?'} [{c.get('status') or 'unknown'}] "
        f"sha256={_child_result_sha256(c)}"
        for c in undecided[:10]
    )


def _maybe_enforce_child_absorption_gate(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    content: Any,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]] | str]:
    undecided = _undispositioned_children(limit_ctx)
    if not undecided:
        return None
    if not getattr(tools._ctx, "_child_absorption_reminded", False):
        tools._ctx._child_absorption_reminded = True
        if content and str(content).strip():
            messages.append({"role": "assistant", "content": content})
        listed = _undecided_children_listing(undecided)
        reminder = (
            "[CHILD_ABSORPTION_REQUIRED]\n"
            "You have child result(s) without a current exact-hash disposition: "
            f"{listed}. Before a clean final answer, inspect unfinished children or record a "
            "tree_note(kind='decision') payload with type=child_result_disposition, child_task_id, "
            "disposition=integrated|irrelevant|deferred, and the shown child_result_sha256. "
            "To disposition several children in ONE call, pass a children array instead: "
            "payload={'type': 'child_result_disposition', 'children': [{'child_task_id': ..., "
            "'disposition': ..., 'child_result_sha256': ...}, ...]}. "
            "discard_child_result remains the shorthand for irrelevant. This is a bounded reminder; "
            "ignoring it will finalize best_effort, not clean."
        )
        _loop()._append_or_merge_user_message(messages, reminder)
        emit_progress("Child absorption reminder injected before final response.")
        llm_trace["reasoning_notes"].append("Child absorption reminder injected before final response.")
        return "continue"
    # Fresh snapshot for the forced prompt: child statuses may have flipped
    # since the reminder round; the model must state CURRENT statuses.
    undecided = _undispositioned_children(limit_ctx)
    text, usage, forced_trace = _loop()._forced_final_answer(
        limit_ctx,
        prompt=(
            "[FINALIZE_WITH_UNABSORBED_CHILDREN]\n"
            "You still have child results without exact dispositions and already received one "
            "child-absorption reminder. Produce an honest best-effort final answer now; name the "
            "unabsorbed or unfinished children explicitly. Current child state: "
            f"{_undecided_children_listing(undecided)}."
        ),
        fallback_text="⚠️ Finalized best-effort with undispositioned child results.",
        reason_code="children_unabsorbed",
    )
    _loop()._merge_finalization_trace(llm_trace, forced_trace)
    _run_forced_children_acceptance(
        tools, limit_ctx, text, messages, emit_progress, llm_trace,
    )
    return text, usage, llm_trace


def _run_forced_children_acceptance(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    text: str,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> None:
    """Content acceptance still runs on the forced children_unabsorbed rail (owner Q2A).

    The panel uses the ORDINARY entry point
    (`_run_task_acceptance_review_once`) after the forced answer text exists
    but BEFORE the loop seals it; the evidence packet carries the
    undispositioned children via the ctx stash. The forced rail can never
    take another model round, so a ``True`` return terminalizes here: a
    requested improvement pass downgrades to ``finalized_unaccepted``; a WAIT
    shape that never ran the panel keeps the typed acceptance-bypass verdict
    from `_record_forced_finalization`. Never raises — salvage outranks review."""
    if not str(text or "").strip():
        return
    tools_ctx = tools._ctx
    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

        # Fresh debt adjacent to the panel's own fresh subtree read: a child
        # may settle across the forced call — one packet, one moment.
        undecided = _undispositioned_children(limit_ctx)
        debt = [
            {
                "task_id": str(c.get("task_id") or c.get("id") or ""),
                "status": str(c.get("status") or "unknown"),
                "child_result_sha256": _child_result_sha256(c),
            }
            for c in undecided[:20]
            if isinstance(c, dict)
        ]
        if len(undecided) > 20:
            # Explicit omission marker: a >20-child debt list must not read as complete.
            debt.append({"omitted": len(undecided) - 20, "total": len(undecided)})
        tools_ctx._forced_undispositioned_children = debt
        another_round = _loop()._run_task_acceptance_review_once(
            tools=tools,
            content=str(text),
            task_id=limit_ctx.task_id,
            task_type=limit_ctx.task_type,
            llm_trace=llm_trace,
            drive_root=limit_ctx.drive_root,
            messages=messages,
            emit_progress=emit_progress,
        )
        if not another_round:
            return
        tools_ctx._task_acceptance_reviewed = True
        _loop()._end_task_acceptance_fence(tools_ctx, outcome="terminal")
        # This rail records its bypass BEFORE its panel, so the terminalisation
        # belongs here rather than in the bypass recorder.
        if _loop().terminalize_dangling_revision(llm_trace, rail="children_unabsorbed"):
            emit_progress(
                "Task acceptance ran on the forced rail; the requested improvement "
                "pass is unavailable, finalizing unaccepted."
            )
    except Exception:
        log.debug("Forced children_unabsorbed acceptance run failed", exc_info=True)
    finally:
        tools_ctx._forced_undispositioned_children = None


def _enforce_swarm_actions(
    content: str,
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> bool:
    """Hold normal finalization while routing or blocking plan work is open."""

    if swarm_router_turn(tools._ctx) and not _loop()._swarm_handoff_attempt(tools._ctx):
        if content.strip():
            messages.append({"role": "assistant", "content": content})
        reminder = (
            "[SWARM_ROUTING_INTENT] Admit exactly one new managed root now with "
            "promote_chat_to_task, or from Main route_to_project for a clearly matching "
            "existing Project. Do not answer inline or steer an existing task."
        )
        _loop()._append_or_merge_user_message(messages, reminder)
        llm_trace["reasoning_notes"].append(reminder)
        emit_progress("Swarm routing action required before final response.")
        return True

    decision = _loop()._force_plan_decision(tools._ctx, llm_trace)
    if decision.get("required"):
        llm_trace["force_plan_decision"] = decision
    if decision.get("allow"):
        return False
    if content.strip():
        messages.append({"role": "assistant", "content": content})
    reminder = _loop()._force_plan_reminder(decision)
    _loop()._append_or_merge_user_message(messages, reminder)
    llm_trace["reasoning_notes"].append(reminder)
    emit_progress("Plan-review action required before final response.")
    return True


_FORCED_BEST_EFFORT_TAIL = (
    "Produce your best final answer now from the verified work so far; clearly "
    "mark anything unverified or incomplete. An honest best-effort result is the "
    "expected outcome here, not a failure."
)


def _prepare_forced_prompt(
    ctx: _RoundLimitContext, prompt: str, llm_trace: Dict[str, Any],
) -> str:
    _loop()._finalize_forced_services(ctx, llm_trace)
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    return prompt + _loop()._forced_delegation_note(tools_ctx, llm_trace)


def _finalize_forced_services(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> None:
    """Finalize services and expose their stable projection before forced synthesis."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return
    _loop()._finalize_task_services(_loop()._LoopExitContext(
        tools=tools,
        drive_root=ctx.drive_root,
        task_id=ctx.task_id,
        event_queue=ctx.event_queue,
        drive_logs=ctx.drive_logs,
        accumulated_usage=ctx.accumulated_usage,
        llm_trace=llm_trace,
    ))
    _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    projection = _loop()._service_finalization_evidence(llm_trace)
    if not projection:
        return
    payload = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if ctx.forced_service_evidence_fingerprint == fingerprint:
        return
    from ouroboros.observability import redact_projection

    ctx.forced_service_evidence_fingerprint = fingerprint
    safe_payload = truncate_review_artifact(
        str(redact_projection(payload).value),
        limit=8000,
    )
    _loop()._append_or_merge_user_message(
        ctx.messages,
        "[SERVICE_FINALIZATION_EVIDENCE]\n"
        "Task services were finalized before forced synthesis. Incorporate this "
        f"evidence and disclose any failure honestly:\n{safe_payload}",
    )


def _drain_forced_owner_directives(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
) -> bool:
    """Drain typed owner input after a forced call and advance answer evidence."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return False
    incoming = ctx.incoming_messages
    if incoming is None:
        incoming = queue.Queue()
    seen = ctx.owner_msg_seen
    if not isinstance(seen, set):
        seen = set()
        ctx.owner_msg_seen = seen
    directives = getattr(tools._ctx, "_owner_directives", None)
    before = len(directives) if isinstance(directives, list) else 0
    _loop()._drain_incoming_messages(
        ctx.messages,
        incoming,
        ctx.drive_root,
        ctx.task_id,
        ctx.event_queue,
        seen,
        owner_ctx=tools._ctx,
    )
    directives = getattr(tools._ctx, "_owner_directives", None)
    after = len(directives) if isinstance(directives, list) else 0
    if after <= before:
        return False
    candidate = _loop()._live_delivery_candidate(ctx)
    binding = (
        candidate.acceptance_binding
        if isinstance(candidate, _loop().DeliveryCandidate)
        and isinstance(candidate.acceptance_binding, dict)
        else {}
    )
    if (
        binding.get("authoritative") is True
        or bool(getattr(tools._ctx, "_task_acceptance_reviewed", False))
        or bool(getattr(tools._ctx, "_task_acceptance_sealed_fence_token", None))
    ):
        _loop()._supersede_task_acceptance_for_owner_followup(tools._ctx, llm_trace)
    _loop()._delivery_evidence_state(tools, ctx, llm_trace)
    return True


def _call_forced_model_once(
    ctx: _RoundLimitContext, *, initial_messages: Any = None, admitted_request: Any = None,
) -> str:
    response_meta: Dict[str, Any] = {}
    identity = (
        "model", "provider", "candidate_raw_sha256", "candidate_raw_size_bytes",
    )
    candidate_predicate = (
        lambda actual: all(getattr(actual, key, None) == getattr(admitted_request, key, None) for key in identity)
        if admitted_request is not None else None
    )
    final_msg, _final_cost = _loop().call_llm_with_retry(
        ctx.llm,
        ctx.messages,
        ctx.active_model,
        getattr(ctx, "tool_schemas", None),
        ctx.active_effort,
        ctx.max_retries,
        ctx.drive_logs,
        ctx.task_id,
        ctx.round_idx,
        ctx.event_queue,
        ctx.accumulated_usage,
        ctx.task_type,
        use_local=ctx.active_use_local,
        deadline_ts=ctx.deadline_ts,
        response_meta_out=response_meta,
        transport_reserve_sec=0.0,
        allow_server_web_search=_loop()._server_web_allowed_by_task(
            getattr(getattr(ctx, "tools", None), "_ctx", None)
        ),
        initial_messages=initial_messages,
        candidate_predicate=candidate_predicate,
    )
    ctx.accumulated_usage["_forced_response_meta"] = response_meta
    return str((final_msg or {}).get("content") or "").strip()


def _publish_model_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    reason_code: str,
    *,
    degraded_reason: str = "",
) -> Optional[DeliveryCandidate]:
    """Replace the retained answer and old verdict."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return None
    candidate = _loop()._replace_delivery_candidate(
        tools,
        ctx,
        llm_trace,
        full_text,
        control=f"forced_replace:{reason_code}",
    )
    candidate.acceptance_binding = _loop()._forced_unaccepted_binding(
        tools, candidate, reason_code,
    )
    candidate.degraded = True
    candidate.degraded_reason = degraded_reason or reason_code
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _publish_stale_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    stale_candidate: DeliveryCandidate,
    reason_code: str,
    suffix: str,
) -> Optional[DeliveryCandidate]:
    """Preserve old text."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return None
    current_revision, _current_fingerprint = _loop()._delivery_evidence_state(
        tools, ctx, llm_trace,
    )
    disclosure = (
        "\n\n⚠️ STALE-EVIDENCE NOTICE — RESUME REQUIRED (host): The preserved "
        "answer above was produced before newer task evidence reached the loop. "
        "It has not been regenerated or accepted against that newer evidence and "
        "does not claim to incorporate it. Resume the task to produce and review "
        "a complete answer against the latest evidence."
    )
    full_text = _loop()._compose_delivery_suffix(
        _loop()._compose_delivery_suffix(stale_candidate.full_text, suffix),
        disclosure,
    )
    candidate = _loop()._replace_delivery_candidate(
        tools,
        ctx,
        llm_trace,
        full_text,
        control=f"forced_stale_preserve:{reason_code}",
    )
    # A host disclosure cannot make the preserved model text current.
    candidate.evidence_revision = stale_candidate.evidence_revision
    candidate.evidence_fingerprint = stale_candidate.evidence_fingerprint
    candidate.acceptance_binding = _loop()._forced_unaccepted_binding(
        tools, candidate, reason_code,
    )
    candidate.acceptance_binding.update({
        "evidence_revision": stale_candidate.evidence_revision,
        "current_evidence_revision": current_revision,
        "stale_evidence": True,
    })
    candidate.degraded = True
    candidate.degraded_reason = reason_code
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    ctx.delivery_candidate = candidate
    return candidate


def _forced_fallback_result(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    fallback_text: str,
    reason_code: str,
    *,
    source: str = "host_fallback",
    retained_source: str = "",
    retained_control: str = "",
    candidate_reason: str = "",
    provider_terminal: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Compose fallback."""
    router_result = _loop()._forced_swarm_router_result(ctx, llm_trace, reason_code)
    if router_result is not None:
        return router_result
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    plan_suffix = (
        _loop()._force_plan_disclosure(tool_ctx, llm_trace, forced_reason=reason_code)
        if tool_ctx is not None else ""
    )
    suffix = plan_suffix + _loop()._forced_orphan_note(ctx)
    live_candidate = _loop()._live_delivery_candidate(ctx)
    fallback_is_retained_model_text = (
        isinstance(live_candidate, _loop().DeliveryCandidate)
        and fallback_text == live_candidate.full_text
    )
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        composed = (
            candidate.model_text or candidate.full_text if provider_terminal else
            sanitize_tool_result_for_log(_loop()._compose_delivery_suffix(candidate.full_text, suffix))
        )
        ctx.accumulated_usage.update(
            terminal_origin=TERMINAL_ORIGIN_MODEL_FINAL,
            terminal_plan_review_open=bool(plan_suffix),
        )
        if composed != candidate.full_text:
            candidate = _publish_model_forced_candidate(
                ctx, llm_trace, composed, reason_code,
                degraded_reason=candidate_reason,
            )
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _loop()._record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=(
                    f"{retained_source}_with_host_suffix"
                    if retained_source else "retained_candidate_with_host_suffix"
                ),
                candidate=candidate,
            )
            return composed, ctx.accumulated_usage, llm_trace
        _loop()._degrade_retained_delivery_candidate(
            ctx,
            llm_trace,
            candidate,
            control=retained_control or f"forced_preserve:{reason_code}",
            reason_code=candidate_reason or reason_code,
        )
        ctx.accumulated_usage["_best_effort_extracted"] = True
        _loop()._record_forced_finalization(
            ctx,
            llm_trace,
            reason_code=reason_code,
            source=retained_source or "retained_candidate",
            candidate=candidate,
        )
        return candidate.full_text, ctx.accumulated_usage, llm_trace

    if fallback_is_retained_model_text and live_candidate is not None:
        candidate = _publish_stale_forced_candidate(
            ctx,
            llm_trace,
            live_candidate,
            reason_code,
            suffix,
        )
        if candidate is not None:
            if candidate_reason:
                candidate.degraded_reason = candidate_reason
                _loop()._publish_delivery_candidate(ctx.tools, candidate, llm_trace)
            if provider_terminal:
                ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_HOST_SALVAGE
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _loop()._record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=f"{source}_stale_evidence_resume_required",
                candidate=candidate,
            )
            return candidate.full_text, ctx.accumulated_usage, llm_trace

    composed = sanitize_tool_result_for_log(_loop()._compose_delivery_suffix(fallback_text, suffix))
    candidate = _publish_model_forced_candidate(
        ctx, llm_trace, composed, reason_code,
    )
    if fallback_is_retained_model_text:
        ctx.accumulated_usage["_best_effort_extracted"] = True
    if provider_terminal:
        ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_HOST_SALVAGE
    _loop()._record_forced_finalization(
        ctx,
        llm_trace,
        reason_code=reason_code,
        source=source,
        candidate=candidate,
    )
    return composed, ctx.accumulated_usage, llm_trace


def _forced_swarm_router_result(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    reason_code: str,
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Use deterministic routing text only when a real rail ends the router."""

    tools = getattr(ctx, "tools", None)
    if tools is None or not swarm_router_turn(tools._ctx):
        return None
    attempt = _loop()._swarm_handoff_attempt(tools._ctx)
    status = str(attempt.get("status") or "not_attempted")
    task_id = str(attempt.get("task_id") or "")
    if status == "scheduled":
        text = f"✅ Swarm admitted managed task {task_id}. Work continues in that task."
    elif status == "unconfirmed":
        text = (
            f"⚠️ Swarm attempted managed task {task_id}, but admission was not confirmed. "
            "No second routing event was emitted; keep the task id for reconciliation."
        )
    elif status == "rejected":
        detail = str(attempt.get("reason") or "admission rejected")
        text = f"⚠️ Swarm could not admit a new managed task ({detail}). No retry was emitted."
    else:
        text = (
            f"⚠️ Swarm reached the task-wide rail `{reason_code}` before a managed-root "
            "admission attempt completed. No inline work was published."
        )
    full_text = _loop()._compose_delivery_suffix(text, _loop()._forced_orphan_note(ctx))
    candidate = _loop()._replace_delivery_candidate(
        tools, ctx, llm_trace, full_text, control=f"forced_swarm_router:{reason_code}",
    )
    if status != "scheduled":
        candidate.degraded = True
        candidate.degraded_reason = reason_code
    _loop()._publish_delivery_candidate(tools, candidate, llm_trace)
    if status == "scheduled":
        # The short acknowledgement hit a rail, but the requested managed work
        # was already durably admitted. Keep that successful handoff truthful.
        ctx.accumulated_usage.pop("execution_status", None)
        ctx.accumulated_usage.pop("reason_code", None)
    else:
        ctx.accumulated_usage.update(execution_status="failed", reason_code=reason_code)
    _loop()._record_forced_finalization(
        ctx,
        llm_trace,
        reason_code=reason_code,
        source="host_swarm_routing_fallback",
        candidate=candidate,
    )
    return candidate.full_text, ctx.accumulated_usage, llm_trace


def _resolve_forced_delivery_control(
    tools_ctx: Any,
    extracted: str,
) -> Tuple[str, str, bool, bool]:
    """Resolve forced control; returns text, degradation, retained, replaced."""
    if tools_ctx is None or not extracted:
        return extracted, "", False, False
    candidate = getattr(tools_ctx, "_delivery_candidate", None)
    armed = bool(getattr(tools_ctx, "_delivery_control_required", False)) or (
        isinstance(candidate, _loop().DeliveryCandidate)
        and _loop()._delivery_replace_required(candidate)
    )
    resolved, retained, degraded, consumed, replaced = (
        _loop()._resolve_forced_delivery_control_body(
            extracted, candidate, armed=armed,
        )
    )
    if consumed:
        tools_ctx._delivery_control_required = False
    return (
        resolved,
        REASON_DELIVERY_CONTROL_DEGRADED if degraded else "",
        retained,
        replaced,
    )


def _forced_final_answer(
    ctx: _RoundLimitContext,
    *,
    prompt: str,
    fallback_text: str,
    reason_code: str,
    single_semantic_turn: bool = False,
    provider_terminal: bool = False,
    _prompt_prepared: bool = False,
    _initial_messages: Any = None,
    _admitted_request: Any = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Forced rail."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    if not _prompt_prepared:
        prompt = _loop()._prepare_forced_prompt(ctx, prompt, llm_trace)
    if ctx.deadline_ts is not None and time.time() >= float(ctx.deadline_ts):
        ctx.accumulated_usage.update(execution_status="failed", reason_code=reason_code)
        return _loop()._forced_fallback_result(
            ctx, llm_trace, fallback_text, reason_code,
            source=f"{reason_code}_window_elapsed",
        )
    router_result = _loop()._forced_swarm_router_result(ctx, llm_trace, reason_code)
    if router_result is not None:
        return router_result
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    _loop()._append_or_merge_user_message(ctx.messages, prompt)
    extracted = ""
    response_meta: Dict[str, Any] = {}
    for attempt in range(1 if single_semantic_turn else 2):
        try:
            ctx.accumulated_usage.pop("_forced_response_meta", None)
            if attempt == 0 and _admitted_request is not None:
                forced = _loop()._call_forced_model_once(
                    ctx, initial_messages=_initial_messages, admitted_request=_admitted_request)
            else:
                forced = _loop()._call_forced_model_once(ctx)
            extracted, response_meta = forced_response_parts(forced, ctx.accumulated_usage)
        except BudgetExceeded:
            _loop()._drain_forced_owner_directives(ctx, llm_trace)
            raise
        except Exception:
            log.warning("Failed to get final response after %s", reason_code, exc_info=True)
            extracted = ""
            response_meta = {}
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        if not _loop()._drain_forced_owner_directives(ctx, llm_trace):
            break
        if str(ctx.accumulated_usage.get("_last_llm_error_kind") or "") == "provider_outcome_unknown":
            return _loop()._forced_fallback_result(
                ctx, llm_trace,
                "⚠️ Provider outcome unknown; directive retained. Resume the task without a blind resend.",
                reason_code,
                source="provider_outcome_unknown_no_resend",
            )
        if attempt == 1:
            return _loop()._forced_fallback_result(
                ctx,
                llm_trace,
                "⚠️ Another directive arrived. Resume the task for a current answer.",
                reason_code,
                source="late_owner_directive_requires_resume",
                provider_terminal=provider_terminal,
            )
        _loop()._finalize_forced_services(ctx, llm_trace)
        _loop()._append_or_merge_user_message(
            ctx.messages,
            "[FORCED_OWNER_REFRESH] Answer all current directives; ignore the stale draft.",
        )

    # Control resolution runs BEFORE the incomplete branch: a retained candidate
    # recovered from a control body must not be discarded as a truncated draft,
    # and a stale-evidence retention keeps its own reason (#447/issue-449).
    incomplete = bool(extracted) and forced_response_is_incomplete(response_meta)
    extracted, control_degraded, retained, replaced = _resolve_forced_delivery_control(
        tools_ctx, extracted,
    )
    current = _loop()._current_delivery_candidate(ctx, llm_trace)
    if retained and current is None:
        return _loop()._forced_fallback_result(
            ctx, llm_trace, extracted, reason_code,
            source="model_control_retained", candidate_reason=control_degraded,
            provider_terminal=provider_terminal,
        )
    # A reply that still asks for a tool is a preamble on every rail, replace
    # control or not; other incompleteness may still be resolved by a replace.
    if incomplete and (
        bool(response_meta.get("tool_call_count")) or current is not None or not replaced
    ):
        return _loop()._forced_fallback_result(
            ctx, llm_trace, extracted or fallback_text, reason_code,
            source="forced_model_incomplete", candidate_reason=control_degraded,
            provider_terminal=provider_terminal,
        )
    if extracted:
        ctx.accumulated_usage["_best_effort_extracted"] = True
        plan_suffix = (
            _loop()._force_plan_disclosure(tools_ctx, llm_trace, forced_reason=reason_code)
            if tools_ctx is not None else ""
        )
        ctx.accumulated_usage["terminal_plan_review_open"] = bool(plan_suffix)
        full_text = extracted if provider_terminal else _loop()._compose_delivery_suffix(
            extracted, plan_suffix + _loop()._forced_orphan_note(ctx),
        )
        ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
        candidate = _publish_model_forced_candidate(
            ctx, llm_trace, full_text, reason_code,
            degraded_reason=control_degraded,
        )
        if control_degraded and candidate is not None:
            llm_trace.setdefault("reasoning_notes", []).append(
                "Forced finalization received an invalid delivery-control object; "
                "preserved the retained complete answer."
            )
        _loop()._record_forced_finalization(
            ctx,
            llm_trace,
            reason_code=reason_code,
            source="model",
            candidate=candidate,
        )
        return (
            candidate.full_text if candidate is not None else full_text,
            ctx.accumulated_usage,
            llm_trace,
        )
    return _loop()._forced_fallback_result(
        ctx,
        llm_trace,
        fallback_text,
        reason_code,
        provider_terminal=provider_terminal,
    )
