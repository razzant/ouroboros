"""Forced finalization of a task that ran out of road: orphan notes, child claims
and the absorption gate, forced children acceptance, swarm-action enforcement,
forced services and owner-directive drain, the one forced model call, stale and
fallback candidates, the swarm router and the forced final answer.
Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import json
import hashlib
import queue
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from ouroboros.outcomes import (
    ACCEPTANCE_FINALIZED_UNACCEPTED,
    ACCEPTANCE_REVISION_REQUESTED,
    REASON_DELIVERY_CONTROL_DEGRADED,
)
from ouroboros.tool_policy import swarm_router_turn
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import truncate_review_artifact
from ouroboros.usage_accounting import BudgetExceeded

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate
    from ouroboros.loop_round_limits import _RoundLimitContext


# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
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

    # Explicit cancellation is lifecycle authority and wins every completion
    # race. Late scratch results are not projected or recovered. Only a
    # SETTLED ``cancelled`` counts as handled (GR2-8c): the legacy
    # ``cancel_requested`` STATUS is an unsettled latch — intent, not outcome.
    # Treating it as done suppressed the handoff reminder for a child still
    # being torn down; such a child stays visible as cancel-pending until
    # custody settles it.
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
    # Forced exits bypass the normal no-tool finalization gate. Project child
    # dispositions here, after services/evidence and the returned candidate
    # have been refreshed, so every forced return exposes the same terminal
    # child-result truth to the outcome reducer.
    _loop()._project_child_result_dispositions(ctx, llm_trace)
    # Common terminal recorder = the ONE seam covering both the LLM-seam forced
    # answer (`_forced_final_answer`) and the no-spend host-fallback fence path
    # (`_handle_budget_exceeded` -> `_forced_fallback_result`).
    _loop()._record_forced_acceptance_bypass(ctx, llm_trace, reason_code)
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
    """A bounded note listing children the parent did NOT explicitly handle (discard/cancel),
    appended to a finalization so paid child work is never SILENTLY orphaned (P1; P5 — no
    prose parsing). On a FORCED finalization (deadline / provider death / finalize_now,
    ``include_terminal=True``) the parent was cut off and may not have seen completions, so
    RUNNING and COMPLETED-undecided children are both reported. On a NORMAL no-tool
    finalization (``include_terminal=False``) the agent was reminded of every change
    (including completions) before choosing to finalize, so only STILL-RUNNING undecided
    children — genuinely orphaned by finalizing mid-flight — are reported. Never raises."""
    try:
        from ouroboros.task_status import FINAL_STATUSES

        children = _loop()._direct_child_results(ctx)
        claimed = _claimed_child_dispositions(ctx)

        def _undecided(c: Dict[str, Any]) -> bool:
            if _loop()._child_disposition_state(c) in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }:
                return False  # explicitly handled
            if not include_terminal and str(c.get("status") or "").strip().lower() in FINAL_STATUSES:
                return False  # completed children were already surfaced via the reminder
            return True

        undecided = [c for c in children if _undecided(c)]
        deferred = [c for c in children if _loop()._child_disposition_state(c) == "deferred"]

        def _label(c: Dict[str, Any]) -> str:
            tid = str(c.get("task_id") or c.get("id") or "?")
            st = str(c.get("status") or "?").strip().lower()
            lifecycle = "running" if st not in FINAL_STATUSES else st
            # W2: a child whose LATEST blackboard decision row no longer binds
            # the current result was READ and decided — say that, not "unread".
            # Say only what the ledger PROVES: the row EXISTS; the binding to
            # the standing result did not. Scoped to children the projection
            # genuinely left UNDECIDED: a carried disposition (deferred /
            # integrated / irrelevant / discarded / cancelled) is not a
            # failed binding, and "re-submit to close it" would be false there.
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
        from ouroboros.tools.join_ledger import _child_result_sha256

        listed = "; ".join(
            f"{c.get('task_id') or c.get('id') or '?'} [{c.get('status') or 'unknown'}] "
            f"sha256={_child_result_sha256(c)}"
            for c in undecided[:10]
        )
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
    text, usage, forced_trace = _loop()._forced_final_answer(
        limit_ctx,
        prompt=(
            "[FINALIZE_WITH_UNABSORBED_CHILDREN]\n"
            "You still have child results without exact dispositions and already received one "
            "child-absorption reminder. Produce an honest best-effort final answer now; name the "
            "unabsorbed or unfinished children explicitly."
        ),
        fallback_text="⚠️ Finalized best-effort with undispositioned child results.",
        reason_code="children_unabsorbed",
    )
    _loop()._merge_finalization_trace(llm_trace, forced_trace)
    _run_forced_children_acceptance(
        tools, limit_ctx, undecided, text, messages, emit_progress, llm_trace,
    )
    return text, usage, llm_trace


def _run_forced_children_acceptance(
    tools: ToolRegistry,
    limit_ctx: _RoundLimitContext,
    undecided: list[Dict[str, Any]],
    text: str,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
    llm_trace: Dict[str, Any],
) -> None:
    """Content acceptance still runs on the forced children_unabsorbed rail (owner Q2A).

    The panel uses the ORDINARY entry point (`_run_task_acceptance_review_once`)
    after the forced answer text exists but BEFORE the loop seals it; the evidence
    packet carries the undispositioned children via the ctx stash. The forced rail
    can never take another model round, so a ``True`` return terminalizes here: a
    requested improvement pass downgrades to ``finalized_unaccepted``, while a WAIT
    shape that never ran the panel keeps the typed acceptance-bypass verdict from
    `_record_forced_finalization`. Never raises — salvage outranks review."""
    if not str(text or "").strip():
        return
    tools_ctx = tools._ctx
    try:
        from ouroboros.tools.join_ledger import _child_result_sha256

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
        decision = llm_trace.get("acceptance_decision")
        status = str(decision.get("status") or "") if isinstance(decision, dict) else ""
        if status == ACCEPTANCE_REVISION_REQUESTED:
            # A panel DID run and asked for an improvement pass; record the honest
            # terminal state instead of leaving a dangling revision request.
            _loop()._set_acceptance_decision(llm_trace, {
                "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
                "reason": "revision_unavailable_on_forced_rail",
                "source": "forced_finalization",
                "rationale": (
                    "The acceptance panel requested an improvement pass, but the "
                    "forced children_unabsorbed rail cannot take another model round."
                ),
            })
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


def _call_forced_model_once(ctx: _RoundLimitContext) -> str:
    final_msg, _final_cost = _loop().call_llm_with_retry(
        ctx.llm,
        ctx.messages,
        ctx.active_model,
        None,
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
    )
    return str((final_msg or {}).get("content") or "").strip()


def _publish_model_forced_candidate(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    full_text: str,
    reason_code: str,
) -> Optional[DeliveryCandidate]:
    """Replace the retained answer and invalidate any verdict for the old SHA."""

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
    candidate.degraded_reason = reason_code
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
    """Preserve useful old text without pretending it absorbed newer evidence."""

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
    # The host-added disclosure is current, but the substantive answer it
    # qualifies is not. Preserve the answer's original evidence provenance so
    # every projection remains conservative instead of laundering unchanged
    # text onto the newer fingerprint.
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
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Return one exact candidate; reuse only current unchanged full text."""

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
        composed = _loop()._compose_delivery_suffix(candidate.full_text, suffix)
        if composed != candidate.full_text:
            candidate = _publish_model_forced_candidate(
                ctx, llm_trace, composed, reason_code,
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
            reason_code=reason_code,
        )
        # The preserved candidate is a previously model-produced complete answer.
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
            ctx.accumulated_usage["_best_effort_extracted"] = True
            _loop()._record_forced_finalization(
                ctx,
                llm_trace,
                reason_code=reason_code,
                source=f"{source}_stale_evidence_resume_required",
                candidate=candidate,
            )
            return candidate.full_text, ctx.accumulated_usage, llm_trace

    composed = _loop()._compose_delivery_suffix(fallback_text, suffix)
    candidate = _publish_model_forced_candidate(
        ctx, llm_trace, composed, reason_code,
    )
    if fallback_is_retained_model_text:
        ctx.accumulated_usage["_best_effort_extracted"] = True
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
) -> Tuple[str, str]:
    """PURE, no-retry delivery-control resolution for the forced rail.

    While the latch is armed, the one forced answer may legitimately be the
    protocol object ``{"delivery_control": ...}`` — shipped raw it leaked
    protocol JSON into the owner's chat and the durable result. Resolve it
    before suffix composition, never re-looping (``_resolve_delivery_control``
    can inject a repair round, which a hard forced stop must never do): valid
    ``keep`` = the retained candidate's full text, valid ``replace`` =
    ``full_answer``, malformed/duplicate/invalid = the retained candidate with
    the typed degraded reason. Armed protocol intent is ANY parsed object with
    the ``delivery_control`` key AND any JSON-looking text that fails to parse
    (the model was told to answer with the object, so that is a mangled
    control, never the answer). JSON while NOT armed passes through untouched.
    Disclosed residual: armed PROSE stands as-is. Clears the latch. Returns
    ``(resolved_text, degraded_reason)``."""
    if tools_ctx is None or not extracted:
        return extracted, ""
    candidate = getattr(tools_ctx, "_delivery_candidate", None)
    candidate = candidate if isinstance(candidate, _loop().DeliveryCandidate) else None
    armed = bool(getattr(tools_ctx, "_delivery_control_required", False)) or (
        candidate is not None and _loop()._delivery_replace_required(candidate)
    )
    if not armed:
        return extracted, ""
    tools_ctx._delivery_control_required = False
    parsed, duplicate_protocol_key = _loop()._parse_delivery_control_object(extracted)
    # Protocol intent: any parsed object with the protocol key (unknown verb =
    # broken control, never prose), or JSON-looking text that fails to parse (a
    # mangled protocol attempt under the armed latch — the candidate is the answer).
    protocol_intent = duplicate_protocol_key or (
        ("delivery_control" in parsed)
        if isinstance(parsed, dict)
        else extracted.lstrip().startswith("{")
    )
    if not protocol_intent:
        # An ordinary prose answer under an armed latch: the fresh text stands.
        return extracted, ""
    selected = str(parsed.get("delivery_control") or "") if isinstance(parsed, dict) else ""
    if selected == "replace" and set(parsed) == {"delivery_control", "full_answer"}:
        replacement = parsed.get("full_answer")
        if isinstance(replacement, str) and replacement.strip():
            return replacement, ""
    elif selected == "keep" and set(parsed) == {"delivery_control"} and candidate is not None:
        return candidate.full_text, ""
    # Malformed/duplicate/invalid control: preserve the retained candidate (or,
    # with none retained, let the caller's fallback text stand) and say so.
    return (
        candidate.full_text if candidate is not None else "",
        REASON_DELIVERY_CONTROL_DEGRADED,
    )


def _forced_final_answer(
    ctx: _RoundLimitContext,
    *,
    prompt: str,
    fallback_text: str,
    reason_code: str,
    single_semantic_turn: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Force one tool-less final answer; stamp the typed forced-finalization
    reason code (the best_effort outcome gate reads it downstream).
    ``single_semantic_turn`` (owner-stop rail, CF-03): exactly ONE logical
    model call — the late-owner-directive semantic refresh is disabled because
    steering is fenced while the stop intent is pending."""
    live_trace = getattr(ctx, "llm_trace", None)
    llm_trace = live_trace if isinstance(live_trace, dict) else {}
    _loop()._finalize_forced_services(ctx, llm_trace)
    router_result = _loop()._forced_swarm_router_result(ctx, llm_trace, reason_code)
    if router_result is not None:
        return router_result
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    prompt += _loop()._forced_delegation_note(tools_ctx, llm_trace)
    _loop()._append_or_merge_user_message(ctx.messages, prompt)
    extracted = ""
    for attempt in range(1 if single_semantic_turn else 2):
        try:
            extracted = _call_forced_model_once(ctx)
        except BudgetExceeded:
            _drain_forced_owner_directives(ctx, llm_trace)
            raise
        except Exception:
            log.warning("Failed to get final response after %s", reason_code, exc_info=True)
            extracted = ""
        ctx.accumulated_usage["execution_status"] = "failed"
        ctx.accumulated_usage["reason_code"] = reason_code
        if not _drain_forced_owner_directives(ctx, llm_trace):
            break
        if attempt == 1:
            return _loop()._forced_fallback_result(
                ctx,
                llm_trace,
                (
                    "⚠️ A new owner directive arrived during the forced refresh and could "
                    "not be incorporated safely before the hard stop. Resume the task to "
                    "produce an answer bound to the latest directive."
                ),
                reason_code,
                source="late_owner_directive_requires_resume",
            )
        _loop()._finalize_forced_services(ctx, llm_trace)
        _loop()._append_or_merge_user_message(
            ctx.messages,
            "[FORCED_OWNER_REFRESH] A new typed owner directive arrived while the prior "
            "forced answer was being generated. Discard that stale draft and produce one "
            "new complete answer bound to every owner directive now present.",
        )

    extracted, control_degraded = _resolve_forced_delivery_control(
        getattr(getattr(ctx, "tools", None), "_ctx", None), extracted,
    )
    if extracted:
        # Typed fact for the best_effort outcome gate: a REAL model answer
        # was extracted (host fallback strings never set this).
        ctx.accumulated_usage["_best_effort_extracted"] = True
        tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
        plan_suffix = (
            _loop()._force_plan_disclosure(tool_ctx, llm_trace, forced_reason=reason_code)
            if tool_ctx is not None else ""
        )
        full_text = _loop()._compose_delivery_suffix(
            extracted, plan_suffix + _loop()._forced_orphan_note(ctx),
        )
        candidate = _publish_model_forced_candidate(
            ctx, llm_trace, full_text, reason_code,
        )
        if control_degraded and candidate is not None:
            candidate.degraded_reason = control_degraded
            llm_trace.setdefault("reasoning_notes", []).append(
                "Forced finalization received an invalid delivery-control object; "
                "preserved the retained complete answer."
            )
            if getattr(ctx, "tools", None) is not None:
                _loop()._publish_delivery_candidate(ctx.tools, candidate, llm_trace)
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
    )
