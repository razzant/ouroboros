"""Forced finalization of a task that ran out of road: orphan notes, child claims
and the absorption gate, forced children acceptance, swarm-action enforcement,
forced services and owner-directive drain, the one forced model call, stale and
fallback candidates and the forced final answer.
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
from ouroboros.outcomes import ACCEPTANCE_FINALIZED_UNACCEPTED, REASON_DELIVERY_CONTROL_DEGRADED
from ouroboros.task_finalization import TERMINAL_ORIGIN_HOST_NOTICE, TERMINAL_ORIGIN_HOST_SALVAGE, TERMINAL_ORIGIN_MODEL_FINAL, set_terminal_host_notice
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
            and bool(current_fingerprint) and not binding.get("stale_evidence")
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
    mid-flight — are reported. A settled child whose own terminal row already
    reached THIS reader's chat is left to that row rather than repeated here; a
    child whose disposition was claimed but did not bind is kept whatever its
    row said, because the terminal row does not carry that fact. Never
    raises."""
    try:
        from ouroboros.project_dialogue import canonical_task_summary_reached_chat
        from ouroboros.task_status import FINAL_STATUSES

        children = _loop()._direct_child_results(ctx)
        claimed = _loop()._claimed_child_dispositions(ctx)
        # The chat this note is about to be read in. Every settled task has a
        # receipt, so only the receipt's own chat can say whether THIS reader
        # already saw the child's terminal row; without it the note stays whole.
        tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
        note_chat_id = getattr(tools_ctx, "current_chat_id", None)

        def _undecided(c: Dict[str, Any]) -> bool:
            if _loop()._child_disposition_state(c) in {
                "integrated", "irrelevant", "deferred", "discarded", "cancelled",
            }:
                return False  # explicitly handled
            if (
                canonical_task_summary_reached_chat(c, note_chat_id)
                and str(c.get("task_id") or c.get("id") or "") not in claimed
            ):
                # This reader already has the child's own terminal row, so it was
                # not orphaned SILENTLY and naming it here tells one event twice.
                # A durable fact of the child's, never a scan of chat text. The
                # claimed-but-unbound case stays: that row says the child
                # finished, never that the parent's disposition failed to bind.
                return False
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
                # A child that FAILED or was CANCELLED is neither running nor
                # completed: the two-way clause described it as something it is
                # not, while its own label already carries the real lifecycle.
                "running ones may be incomplete, finished ones (completed, failed "
                "or cancelled) may be UNREAD"
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
            "child-absorption reminder. Produce an honest best-effort final answer now that says "
            "what remains unabsorbed or unfinished; the exact child state is: "
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


def _plan_gate_identity(decision: Dict[str, Any]) -> str:
    """Typed identity of the plan-review gate as the mind was last TOLD it.
    Existing decision fields only — never the rendered English, which cannot tell
    two different waves apart (astra F3)."""
    return "|".join(str(decision.get(key) or "") for key in (
        "status", "allow", "closed", "outcome", "enforcement", "cycles_paid",
        "custody_pending", "reviewer_slots_degraded", "review_late_result_pending",
        "quorum_unreachable", "owner_hurry_local_advisory",
    ))


def _enforce_swarm_actions(
    content: str,
    messages: List[Dict[str, Any]],
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    emit_progress: Callable[[str], None],
) -> bool:
    """Hold normal finalization while blocking plan work is open, and hand the mind
    ONE fact when a hold it was told about was released behind its back (owner Q4=A).

    The marker advances ONLY on the two paths that actually append a message to
    ``messages`` — the hold reminder and the release note — so it records what
    entered the MIND's context, not what this gate observed. Every other gate
    transition already reaches the mind by its own route and must not buy a round:
    a settled reviewer slot arrives as a ``[System task message]`` through the task
    mailbox (``plan_review_collect.announce_released_settlement`` ->
    ``loop_round_limits`` drain), a rail release rides the forced prompt's typed
    facts, and ``closed``/``author_stopped``/``cycles_exhausted`` are the mind's own
    ``plan_task`` results. The one residual is the owner's hurry, which the HQ1
    no-chat contract forbids putting into ``messages``
    (``loop_round_limits.py``) — that is the only release this fires on."""

    decision = _loop()._force_plan_decision(tools._ctx, llm_trace)
    if decision.get("required"):
        llm_trace["force_plan_decision"] = decision
    identity = _plan_gate_identity(decision) if decision.get("required") else ""
    told = str(getattr(tools._ctx, "_plan_gate_told_identity", "") or "")
    if decision.get("allow"):
        if not (
            told and identity and identity != told
            and not decision.get("closed")
            and decision.get("status") != "author_stopped"
            and decision.get("owner_hurry_local_advisory")
            and not getattr(tools._ctx, "_plan_gate_release_told", False)
        ):
            return False
        tools._ctx._plan_gate_told_identity = identity
        tools._ctx._plan_gate_release_told = True
        if content.strip():
            messages.append({"role": "assistant", "content": content})
        _loop()._append_or_merge_user_message(
            messages,
            "[PLAN_REVIEW_RELEASED]\n"
            "The plan-review hold reported to you earlier is released by the owner's hurry "
            "request, and the review is still open.\n" + _FACTS_LEAD + "\n"
            + _plan_gate_facts(decision),
        )
        llm_trace["reasoning_notes"].append("Released plan-review gate reported before final response.")
        emit_progress("Plan-review gate released before final response.")
        return True
    if content.strip():
        messages.append({"role": "assistant", "content": content})
    reminder = _loop()._force_plan_reminder(decision)
    _loop()._append_or_merge_user_message(messages, reminder)
    llm_trace["reasoning_notes"].append(reminder)
    tools._ctx._plan_gate_told_identity = identity
    emit_progress("Plan-review action required before final response.")
    return True


_FORCED_BEST_EFFORT_TAIL = (
    "Produce your best final answer now from the verified work so far; clearly "
    "mark anything unverified or incomplete. An honest best-effort result is the "
    "expected outcome here, not a failure."
)


_FACTS_LEAD = (
    "Typed facts this task already records about this finalization. They stay recorded "
    "whether or not you mention them; your own answer is where they are said, in your own "
    "words — this block is data, not wording to reuse."
)


def _plan_gate_facts(decision: Dict[str, Any]) -> str:
    """One ``key=value`` line for an open plan-review gate, built from the decision
    dict's own typed fields (never ``plan_review_disclosure``'s owner-facing English:
    BIBLE P5/P6 — the host does not put a template in the mind's mouth).

    Only fields that do NOT depend on ``hard_rail`` are emitted, so the same line is
    true on the normal rail and on every forced rail. ``owner_hurry_local_advisory``/
    ``configured_enforcement`` carry the accurate hurry-vs-advisory attribution that
    the recorded notice does not distinguish."""
    if not decision.get("required") or decision.get("closed"):
        return ""
    parts = ["plan_review_open=true"]
    if decision.get("outcome"):
        parts.append(f"plan_review_outcome={decision['outcome']}")
    if decision.get("enforcement"):
        parts.append(f"plan_review_enforcement={decision['enforcement']}")
    if decision.get("decision_authority"):
        parts.append(f"decision_authority={decision['decision_authority']}")
    if decision.get("owner_hurry_local_advisory"):
        parts.append("owner_hurry_local_advisory=true")
        parts.append(f"configured_enforcement={decision.get('configured_enforcement') or 'blocking'}")
    for key in (
        "reviewer_slots_degraded", "custody_pending",
        "review_late_result_pending", "quorum_unreachable",
    ):
        if decision.get(key):
            parts.append(f"{key}=true")
    if decision.get("cycles_paid"):
        parts.append(f"cycles_paid={decision['cycles_paid']}")
    return " ".join(parts)


def _forced_state_facts(ctx: _RoundLimitContext, llm_trace: Dict[str, Any]) -> str:
    """Hand the ONE forced model call the same limitations this rail will record beside
    its answer — as typed facts, not as the owner-facing notice.

    Never raises: a forced answer outranks its own annotation."""
    try:
        tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
        projected = llm_trace.get("force_plan_decision")
        decision = (
            projected if isinstance(projected, dict)
            else (_loop()._force_plan_decision(tools_ctx, llm_trace) if tools_ctx is not None else {})
        )
        lines = [line for line in (_plan_gate_facts(decision),) if line]
        undecided = _undispositioned_children(ctx)
        if undecided:
            lines.append(
                f"children_undecided={len(undecided)}: {_undecided_children_listing(undecided)}"
            )
        deferred = [
            child for child in _direct_child_results(ctx)
            if _child_disposition_state(child) == "deferred"
        ]
        if deferred:
            lines.append(
                f"children_deferred={len(deferred)}: {_undecided_children_listing(deferred)}"
            )
        candidate = _loop()._live_delivery_candidate(ctx)
        current = str(getattr(tools_ctx, "_delivery_evidence_fingerprint", "") or "")
        if (candidate is not None and current and candidate.evidence_fingerprint
                and candidate.evidence_fingerprint != current):
            lines.append("retained_answer_predates_current_evidence=true")
        if not lines:
            return ""
        return "\n\n[TASK_STATE_FACTS]\n" + _FACTS_LEAD + "\n" + "\n".join(lines)
    except Exception:
        log.debug("Forced task-state facts unavailable", exc_info=True)
        return ""


def _prepare_forced_prompt(
    ctx: _RoundLimitContext, prompt: str, llm_trace: Dict[str, Any],
) -> str:
    _loop()._drain_forced_owner_directives(ctx, llm_trace)
    _loop()._finalize_forced_services(ctx, llm_trace)
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    # The typed facts sit BEFORE the acceptance observation so that one-shot block
    # stays the tail of the message (the prompt-cache prefix rule).
    return (
        prompt
        + _loop()._forced_delegation_note(tools_ctx, llm_trace)
        + _forced_state_facts(ctx, llm_trace)
        + _presence_forced_contract(ctx, tools_ctx)
        + _forced_subject_prompt(ctx, llm_trace)
    )


def _presence_forced_contract(ctx: _RoundLimitContext, tools_ctx: Any) -> str:
    """Arm a Presence task's ONE forced call to declare its outward delivery apart from its record.

    The forced answer is the internal record (owner, review, task result); what the
    conversation receives is only the nested ``presence_finish`` declaration. Until a
    valid declaration arrives with a model-final answer the arm stays ``missing``, so
    no untyped internal prose becomes Presence speech (owner Q4). Not a delivery receipt.
    Only a context whose final becomes a Presence result is armed: the host ceiling AND
    the Presence metadata the pipeline keys that result on. A delegated child inherits
    the ceiling with its contract, but it answers its parent, not a conversation.
    """
    contract = getattr(tools_ctx, "task_contract", None)
    metadata = getattr(tools_ctx, "task_metadata", None)
    presence = metadata.get("presence") if isinstance(metadata, dict) else None
    if not (isinstance(contract, dict) and isinstance(contract.get("capability_ceiling"), dict)
            and isinstance(presence, dict)):
        return ""
    tools_ctx._presence_forced_declaration = {"status": "missing", "reason": "no presence_finish declaration"}
    tools_ctx._presence_forced_pending = None
    try:
        from ouroboros.presence_context import presence_send_facts
        from ouroboros.tool_access import canonical_data_root

        # Receipts live on the canonical root; a forked execution drive holds none.
        sent = presence_send_facts(canonical_data_root(tools_ctx), ctx.task_id, presence)
    except Exception:
        sent = "unknown (receipts unreadable)"
    handoff = getattr(tools_ctx, "_swarm_handoff_attempt", None)
    scheduled = isinstance(handoff, dict) and str(handoff.get("status") or "") == "scheduled"
    return (
        "\n\n[PRESENCE_DELIVERY]\n"
        "This task answers a Presence conversation. Your answer is its internal record for the "
        "owner and review; the people in the conversation receive only what you declare. Return "
        "exactly one JSON object and no other text: "
        '{"delivery_control":"replace","full_answer":"<complete internal record>",'
        '"presence_finish":{"outcome":"message","message":"<new text for the conversation>"}}'
        + (' ("delivery_control":"keep" without full_answer keeps the current answer as the record)'
           if _loop()._live_delivery_candidate(ctx) is not None else "")
        + ". Outcomes: message = new useful speech on their subject, an honest partial included; "
        "silent = nothing new needs saying; tool_delivered = the substantive result already reached "
        "them through a transport tool; deferred = acknowledge work that was actually scheduled"
        + (" (it was)" if scheduled else " (none was)") + ". Keep internal facts in full_answer; "
        "the host never forwards that record automatically. You decide what, if anything, to say "
        "in presence_finish.message, including relevant limitations. "
        "An early acknowledgement is not the promised result and an uncertain send may not have "
        "landed. Without a valid presence_finish nothing new is sent. Sends confirmed for this task "
        f"so far: {sent}."
    )


def _read_presence_declaration(tools_ctx: Any, extracted: str) -> None:
    """Record the nested ``presence_finish`` of an armed forced body without rewriting that body.

    The resolver keeps reading the original bytes (``envelope_keys`` admits this one
    key), so the parser's duplicate-key evidence still reaches every rail, the
    acceptance subject included. A declaration is valid only in an envelope that
    repeats no key anywhere. Each read records its non-speaking verdict at once; a
    valid declaration speaks only once its answer becomes the model final.
    """
    from ouroboros.loop_delivery import _parse_delivery_control_object
    from ouroboros.observability import strip_protocol_fence
    from ouroboros.tools.presence import PRESENCE_OUTCOMES

    tools_ctx._presence_forced_pending = None
    tools_ctx._presence_forced_declaration = {"status": "missing", "reason": "no presence_finish declaration"}
    parsed, duplicate = _parse_delivery_control_object(strip_protocol_fence(extracted))
    if not duplicate and (not isinstance(parsed, dict) or "presence_finish" not in parsed):
        return
    value = parsed.get("presence_finish") if isinstance(parsed, dict) else None
    reason = ""
    if duplicate or getattr(parsed, "has_duplicate_keys", False):
        reason = "the forced envelope repeats a key"
    elif not isinstance(value, dict) or not set(value) <= {"outcome", "message"} \
            or value.get("outcome") not in PRESENCE_OUTCOMES or not isinstance(value.get("message", ""), str):
        reason = "presence_finish must be one {outcome, message} object with a known outcome"
    else:
        outcome, message = value["outcome"], value.get("message", "").strip()
        handoff = getattr(tools_ctx, "_swarm_handoff_attempt", None)
        if outcome == "message" and not message:
            reason = "message needs nonblank conversational text"
        elif outcome == "silent" and message:
            reason = "silent carries no text"
        elif outcome == "deferred" and not (isinstance(handoff, dict) and handoff.get("status") == "scheduled"):
            reason = "deferred needs work that was actually scheduled"
    if reason:
        tools_ctx._presence_forced_declaration = {"status": "invalid", "reason": reason}
    else:
        tools_ctx._presence_forced_pending = {
            "status": "declared", "outcome": value["outcome"], "message": value.get("message", "").strip()}


def _forced_subject_prompt(ctx: _RoundLimitContext, llm_trace: Dict[str, Any]) -> str:
    """Capture the source before pricing/sending, never after a reply arrives."""
    from ouroboros.loop_acceptance import capture_acceptance_observation, acceptance_observation_prompt

    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if tools_ctx is None:
        return ""
    observed = capture_acceptance_observation(tools_ctx, llm_trace, ctx.incoming_messages)
    rendered = acceptance_observation_prompt(tools_ctx, observed)
    return "\n\n" + rendered if rendered else ""


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
    from ouroboros.model_slots import task_model_binding
    from ouroboros.model_wait import current_model_wait

    owner_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    waiter = current_model_wait()
    role, account = task_model_binding({"model_role": getattr(ctx, "model_role", ""),
        "task_metadata": getattr(owner_ctx, "task_metadata", {})},
        context_fit_plan=getattr(owner_ctx, "context_fit_plan", None),
        overrides=waiter.overrides if waiter else None)
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
        model_role=role,
        model_account_override=account,
        # A forced final belongs to the loop invocation that is finishing, so it
        # continues that same active turn instead of opening a new one.
        model_turn_state=getattr(owner_ctx, "model_turn_state", None),
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
    """Publish forced model text, retaining an unchanged current answer's binding."""

    tools = getattr(ctx, "tools", None)
    if tools is None:
        return None
    current = _loop()._current_delivery_candidate(ctx, llm_trace)
    if current is not None and current.full_text == sanitize_tool_result_for_log(full_text):
        return _loop()._degrade_retained_delivery_candidate(
            ctx, llm_trace, current, control=f"forced_preserve:{reason_code}",
            reason_code=degraded_reason or reason_code,
        )
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
    current_revision, current_fingerprint = _loop()._delivery_evidence_state(
        tools, ctx, llm_trace,
    )
    disclosure = (
        "\n\n⚠️ STALE-EVIDENCE NOTICE — RESUME REQUIRED (host): The preserved "
        + ("answer above was produced before newer task evidence reached the loop. "
           "It has not been regenerated or accepted against that newer evidence and "
           "does not claim to incorporate it. "
           if current_fingerprint else
           # UNKNOWN evidence: the host could not re-read it, so it is not claimed
           # newer — only unverified. Typed, never a crash or an approval.
           "answer above rests on task evidence the host could no longer read. "
           "It has not been re-verified or accepted against the current evidence and "
           "does not claim to reflect it. ")
        + "Resume the task to produce and review a complete answer against the latest evidence."
    )
    set_terminal_host_notice(ctx.accumulated_usage, suffix, disclosure)
    candidate = _loop()._replace_delivery_candidate(
        tools,
        ctx,
        llm_trace,
        stale_candidate.full_text,
        control=f"forced_stale_preserve:{reason_code}",
    )
    # A host disclosure cannot make the preserved model text current.
    candidate.evidence_revision = stale_candidate.evidence_revision
    candidate.evidence_fingerprint = stale_candidate.evidence_fingerprint
    candidate.owner_source_sha256 = stale_candidate.owner_source_sha256
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
    tool_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    plan_suffix = (
        _loop()._force_plan_disclosure(tool_ctx, llm_trace, forced_reason=reason_code)
        if tool_ctx is not None else ""
    )
    suffix = plan_suffix + _loop()._forced_orphan_note(ctx)
    # Every rail that DISCLOSES types the fact: a host-notice fallback used to state
    # the open review in prose while nothing typed carried it. Absent = not open: a
    # clean result carries no key (pinned usage shapes).
    if plan_suffix:
        ctx.accumulated_usage["terminal_plan_review_open"] = True
    set_terminal_host_notice(ctx.accumulated_usage, suffix)
    live_candidate = _loop()._live_delivery_candidate(ctx)
    fallback_is_retained_model_text = (
        isinstance(live_candidate, _loop().DeliveryCandidate)
        and fallback_text == live_candidate.full_text
    )
    candidate = _loop()._current_delivery_candidate(ctx, llm_trace)
    if candidate is not None:
        composed = (
            candidate.model_text or candidate.full_text if provider_terminal else
            candidate.full_text
        )
        ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
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

    composed = sanitize_tool_result_for_log(fallback_text)
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


def _resolve_forced_delivery_control(
    tools_ctx: Any,
    extracted: str,
    *, ctx: Optional[_RoundLimitContext] = None, llm_trace: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, bool, bool]:
    """Resolve forced control; returns text, degradation, retained, replaced."""
    if tools_ctx is None or not extracted:
        return extracted, "", False, False
    presence_armed = isinstance(getattr(tools_ctx, "_presence_forced_declaration", None), dict)
    if presence_armed:
        _read_presence_declaration(tools_ctx, extracted)
    candidate = getattr(tools_ctx, "_delivery_candidate", None)
    armed = presence_armed or bool(getattr(tools_ctx, "_delivery_control_required", False)) or (
        isinstance(candidate, _loop().DeliveryCandidate)
        and _loop()._delivery_replace_required(candidate)
    )
    resolved, retained, degraded, consumed, replaced = (
        _loop()._resolve_forced_delivery_control_body(
            extracted, candidate, armed=armed,
            envelope_keys=("presence_finish",) if presence_armed else (),
        )
    )
    if presence_armed and isinstance(tools_ctx._presence_forced_pending, dict):
        from ouroboros.loop_delivery import _parse_delivery_control_body

        parsed, _, _ = _parse_delivery_control_body(extracted)
        if degraded or not (isinstance(parsed, dict) and parsed.get("delivery_control") in {"keep", "replace"}):
            # A declaration cannot speak unless its outer control positively chose the final record.
            tools_ctx._presence_forced_pending = None
            tools_ctx._presence_forced_declaration = {
                "status": "invalid", "reason": "the delivery-control envelope was rejected"}
    if consumed:
        tools_ctx._delivery_control_required = False
        from ouroboros.loop_delivery import _parse_delivery_control_body, apply_delivery_subject_decision

        parsed, duplicate, embedded = _parse_delivery_control_body(extracted)
        if (not duplicate and not embedded and isinstance(parsed, dict)
                and "acceptance_subject" in parsed and not degraded):
            applied, error = (
                apply_delivery_subject_decision(ctx.tools, ctx, llm_trace, parsed["acceptance_subject"])
                if ctx is not None and llm_trace is not None else
                (False, "forced subject has no current source observation context")
            )
            if llm_trace is not None:
                llm_trace["forced_acceptance_subject"] = {"applied": applied, "reason": error}
            if not applied:
                degraded = True
                if ctx is not None and llm_trace is not None and isinstance(candidate, _loop().DeliveryCandidate):
                    candidate.acceptance_binding = _loop()._forced_unaccepted_binding(
                        ctx.tools, candidate, REASON_DELIVERY_CONTROL_DEGRADED,
                    )
                    tools_ctx._task_acceptance_reviewed = False
                    _loop()._set_acceptance_decision(llm_trace, {
                        "status": ACCEPTANCE_FINALIZED_UNACCEPTED, "reason": REASON_DELIVERY_CONTROL_DEGRADED,
                        "source": "forced_acceptance_subject", "rationale": error,
                    })
    return (
        resolved,
        REASON_DELIVERY_CONTROL_DEGRADED if degraded else "",
        retained,
        replaced,
    )


def _send_admitted_forced_candidate(
    ctx: _RoundLimitContext, initial_messages: Any, admitted_request: Any, reason_code: str,
) -> str:
    """Send the admitted wrap-up; one that drifted from its pricing is sent once more, unpredicated.

    The identity predicate refuses BEFORE a byte leaves and the refused
    reservation is released, so nothing was paid and nothing is sent twice. Ending
    the task there cost the owner the whole final answer three times in one night
    (a wire key the send had grown and the priced copy had not), while the money
    that predicate guards is guarded again by the ledger fence, which prices the
    send it actually sees. So the drift is recorded as a typed fact — the refused
    attempt's row and sealed candidate carry the actual identity — and the answer
    is asked for once more the ordinary way. A closed dispatch window is a
    deadline, not drift, and keeps its own rail."""
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    try:
        return _loop()._call_forced_model_once(
            ctx, initial_messages=initial_messages, admitted_request=admitted_request)
    except PhysicalDispatchInterrupted:
        raise
    except PhysicalAttemptPreconditionFailed as refusal:
        log.warning("Admitted %s wrap-up candidate drifted from its pricing; sending it unpredicated", reason_code)
        _loop()._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
            "checkpoint_kind": "forced_candidate_drift",
            "reason_code": reason_code,
            "refused_attempt_id": str(getattr(refusal, "attempt_id", "") or ""),
            "admitted": {key: getattr(admitted_request, key, None) for key in (
                "model", "provider", "candidate_raw_sha256", "candidate_raw_size_bytes")},
        })
        return _loop()._call_forced_model_once(ctx)


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
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    _loop()._append_or_merge_user_message(ctx.messages, prompt)
    extracted = ""
    response_meta: Dict[str, Any] = {}
    for attempt in range(1 if single_semantic_turn else 2):
        try:
            ctx.accumulated_usage.pop("_forced_response_meta", None)
            if attempt == 0 and _admitted_request is not None:
                forced = _send_admitted_forced_candidate(
                    ctx, _initial_messages, _admitted_request, reason_code)
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
        if single_semantic_turn or attempt == 1:
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
            "[FORCED_OWNER_REFRESH] Answer all current directives; ignore the stale draft."
            + _forced_subject_prompt(ctx, llm_trace),
        )

    # Control resolution runs BEFORE the incomplete branch: a retained candidate
    # recovered from a control body must not be discarded as a truncated draft,
    # and a stale-evidence retention keeps its own reason (#447/issue-449).
    incomplete = bool(extracted) and forced_response_is_incomplete(response_meta)
    extracted, control_degraded, retained, replaced = _resolve_forced_delivery_control(
        tools_ctx, extracted, ctx=ctx, llm_trace=llm_trace,
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
        if plan_suffix:  # absent = not open: a clean result carries no key (pinned usage shapes)
            ctx.accumulated_usage["terminal_plan_review_open"] = True
        set_terminal_host_notice(ctx.accumulated_usage, plan_suffix, _loop()._forced_orphan_note(ctx))
        full_text = extracted
        ctx.accumulated_usage["terminal_origin"] = TERMINAL_ORIGIN_MODEL_FINAL
        if isinstance(getattr(tools_ctx, "_presence_forced_pending", None), dict):
            tools_ctx._presence_forced_declaration = tools_ctx._presence_forced_pending
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
