"""The task-acceptance fence and its obligations: eligibility, begin/end/supersede,
subtree snapshots, the final-answer latch, the typed decision vocabulary and the
obligation ledger. Extracted from loop.py (v7 L-B split); loop.py re-exports every name."""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from ouroboros.acceptance_settlement import forced_rail_panel_verdict
from ouroboros.review_cycles import REASON_REVIEW_CYCLES_EXHAUSTED
from ouroboros.review_projection import publish_acceptance_checkpoint
from ouroboros.outcomes import ACCEPTANCE_ACCEPTED, ACCEPTANCE_BYPASS_REASONS, ACCEPTANCE_BYPASS_REASON_BY_RAIL, ACCEPTANCE_DECISION_STATUSES, ACCEPTANCE_FINALIZED_UNACCEPTED, ACCEPTANCE_REVISION_REQUESTED, REASON_ACCEPTANCE_PREPARATION_FAILED, REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE, REASON_DELIVERY_CONTROL_DEGRADED, REASON_IDENTICAL_ACCEPTANCE_REFUSED, extract_final_answer, turn_has_reviewable_effects
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import truncate_review_artifact

if TYPE_CHECKING:  # annotation-only names; lazy under future annotations, never imported at runtime
    from ouroboros.loop_delivery import DeliveryCandidate
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


def _task_acceptance_eligible(
    mode: str,
    llm_trace: Dict[str, Any],
    is_direct_chat: bool,
    *,
    is_root_task: bool = True,
    task_contract: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Return ``(host_should_review, trigger_reason)``.

    Both modes review observable effects and typed deliverables/criteria.
    ``auto`` also honors an agent's explicit review-tool request, so read-only
    research remains reviewable without classifying its prose or tool counts.
    Queue membership alone qualifies only in ``required``. Child reviews stay
    advisory, and ``off`` never reviews.
    Eligibility uses typed contracts and runtime facts, never message content.
    """
    if mode == "off":
        return False, "off"
    if not is_root_task:
        return False, "skipped_child_advisory"
    if mode in {"auto", "required"}:
        prefix = "required" if mode == "required" else "auto"
        if turn_has_reviewable_effects(llm_trace):
            return True, f"{prefix}_effect"
        if mode == "required" and not is_direct_chat:
            return True, f"{prefix}_nondirect"
        contract = task_contract if isinstance(task_contract, dict) else {}
        if (
            str(contract.get("expected_output") or "").strip()
            or bool(contract.get("acceptance_criteria"))
            or bool(contract.get("success_criteria"))
            or bool(contract.get("acceptance_claims"))
        ):
            return True, f"{prefix}_contract"
        if any(
            isinstance(call, dict) and call.get("tool") == "task_acceptance_review"
            for call in (llm_trace.get("tool_calls") or [])
        ):
            return True, f"{prefix}_agent_request"
        return False, "skipped_conversation"
    return False, "skipped_unknown_mode"


from ouroboros.loop_messages import (  # noqa: F401 — shared owner-source surface
    _acceptance_observation_state,
    capture_acceptance_observation,
    acknowledge_acceptance_observation,
    acceptance_observation_prompt,
    queue_inspection_unknown,
)


@dataclass(frozen=True)
class FenceOutcome:
    """Typed answer to one queue-fence request; truthy iff the supervisor said yes.

    ``refused``: it answered no (``reason``). ``unknown``: no answer arrived within
    ``waited_sec`` — a gap, never a refusal, a verdict or an owner message.
    """

    status: str = "ok"
    op: str = ""
    reason: str = ""
    waited_sec: float = 0.0

    def __bool__(self) -> bool:
        return self.status == "ok"


def _root_task_id(ctx: Any, task_id: str) -> str:
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    return str(meta.get("root_task_id") or getattr(ctx, "root_task_id", "") or task_id)


def _settle_fence_outcome(ctx: Any, outcome: FenceOutcome) -> FenceOutcome:
    """Expose the outcome on ctx; a refusal or a gap leaves ONE durable worker-side row."""
    ctx._task_acceptance_fence_outcome = outcome
    if not outcome:
        from ouroboros import task_pacing
        from ouroboros.utils import append_jsonl, utc_now_iso

        task_id = str(getattr(ctx, "task_id", "") or "")
        try:
            append_jsonl(task_pacing.acceptance_timing_events_path(ctx), {
                "ts": utc_now_iso(), "type": "supervisor_ack_unavailable", "task_id": task_id,
                "root_task_id": _root_task_id(ctx, task_id), "op": outcome.op, "outcome": outcome.status,
                "reason": outcome.reason, "waited_sec": outcome.waited_sec,
            })
        except Exception:
            log.warning("supervisor_ack_unavailable row could not be written for %s", task_id, exc_info=True)
    return outcome


def _fence_request(ctx: Any, op: str, callback: Callable[..., Any], **kwargs: Any) -> tuple[FenceOutcome, Any]:
    """One request to the queue-owned fence: ok | refused(reason) | unknown(waited_sec)."""
    started = time.monotonic()
    try:
        response = callback(**kwargs)
        if not (isinstance(response, dict) and not response.get("ok", True)):
            return _settle_fence_outcome(ctx, FenceOutcome(op=op)), response
        outcome = FenceOutcome("refused", op, str(response.get("error") or response.get("status") or ""))
    except RuntimeError as exc:
        outcome = FenceOutcome("refused", op, str(exc))
    except Exception as exc:
        outcome = FenceOutcome("unknown", op, type(exc).__name__, round(time.monotonic() - started, 3))
    return _settle_fence_outcome(ctx, outcome), None


def _drop_fence_binding(ctx: Any) -> None:
    """The queue owns the fence; a token whose state is unproven is never retained."""
    ctx._task_acceptance_fence_token = None
    ctx._task_acceptance_fence_generation = None
    ctx._task_acceptance_queue_descendants = []


def _begin_task_acceptance_fence(ctx: Any, task_id: str) -> tuple[FenceOutcome, Any]:
    """Optional seam implemented by the supervisor under its queue lock."""
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if admission_lock is not None and admission_agent is not None:
        with admission_lock:
            ctx._task_acceptance_owner_generation = int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
    existing = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    if existing is not None and not callable(inspect):
        return FenceOutcome(op="begin"), existing
    if existing is not None:
        outcome, refreshed = _fence_request(ctx, "inspect", inspect, token=str(existing))
        if outcome:
            ctx._task_acceptance_queue_descendants = []
        if outcome and isinstance(refreshed, dict):
            ctx._task_acceptance_queue_descendants = list(refreshed.get("queue_descendants") or [])
            ctx._task_acceptance_fence_generation = int(refreshed.get("owner_message_generation") or 0)
        if outcome or outcome.status == "unknown":
            return outcome, existing  # no answer is a gap: the binding and its known generation stay
        _drop_fence_binding(ctx)  # refused: the row is gone — rebind through the idempotent begin
    callback = getattr(ctx, "begin_acceptance_fence", None)
    if not callable(callback):
        return FenceOutcome(op="begin"), None  # one-minor/direct-context compatibility
    outcome, response = _fence_request(
        ctx, "begin", callback, root_task_id=_root_task_id(ctx, task_id), task_id=str(task_id))
    answer = response if isinstance(response, dict) else {"token": response}
    if outcome and answer.get("token") in (None, False, ""):
        outcome = _settle_fence_outcome(ctx, FenceOutcome("refused", "begin", "no_token"))
    if not outcome:
        return outcome, None
    ctx._task_acceptance_queue_descendants = list(answer.get("queue_descendants") or [])
    # Never None: ``end`` omits ``expected_generation`` for None, and a seal without the queue's
    # compare-and-seal is the blind seal (#406). A fresh fence starts at 0; a wrong 0 refuses.
    ctx._task_acceptance_fence_generation = int(answer.get("owner_message_generation") or 0)
    ctx._task_acceptance_fence_token = answer["token"]
    return outcome, answer["token"]


def _end_task_acceptance_fence(ctx: Any, *, outcome: str, admission_locked: bool = False) -> FenceOutcome:
    if getattr(ctx, "_acceptance_review_only", False) and outcome != "revision":
        outcome = "revision"  # Early feedback never closes the root's future work.
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    if token is None and str(outcome) == "revision":
        token = getattr(ctx, "_task_acceptance_sealed_fence_token", None)
    callback = getattr(ctx, "end_acceptance_fence", None)
    admission_lock = getattr(ctx, "owner_message_admission_lock", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    acquired = False
    try:
        if admission_lock is not None and admission_agent is not None and not admission_locked:
            admission_lock.acquire()
            acquired = True
        expected_owner_generation = getattr(ctx, "_task_acceptance_owner_generation", None)
        from ouroboros.loop_messages import owner_source_sha256
        from ouroboros.loop_transport import _owner_signal_pending

        def owner_mail_pending() -> bool:
            return _owner_signal_pending(
                getattr(ctx, "_acceptance_observation_incoming", None), getattr(ctx, "drive_root", None),
                str(getattr(ctx, "task_id", "") or ""), getattr(ctx, "_loop_mailbox_seen_ids", None),
                getattr(ctx, "task_attempt", None) or 1, owner_authority_only=True,
            )

        acknowledged_source = getattr(ctx, "_acceptance_ack_source_sha256", "")
        generation_mismatch = bool(
            (acknowledged_source and (acknowledged_source != owner_source_sha256(ctx) or owner_mail_pending()))
            or (expected_owner_generation is not None and admission_agent is not None
                and int(getattr(admission_agent, "_owner_message_generation", 0) or 0) != int(expected_owner_generation))
        )
        effective_outcome = "revision" if generation_mismatch else str(outcome)
        if token is None or not callable(callback):
            ctx._task_acceptance_fence_generation_mismatch = generation_mismatch
            return FenceOutcome(op="end")
        expected_queue_generation = getattr(ctx, "_task_acceptance_fence_generation", None)
        result, response = _fence_request(
            ctx, "end", callback, token=token, outcome=effective_outcome,
            **({} if expected_queue_generation is None else {"expected_generation": int(expected_queue_generation)}),
        )
        status = str(response.get("status") or "") if isinstance(response, dict) else ""
        generation_mismatch = generation_mismatch or bool(isinstance(response, dict) and response.get("generation_mismatch"))
        if result and status == "released" and effective_outcome != "revision" and not generation_mismatch:
            # Only ``sealed`` is a seal. An unexplained release may echo a lost ``released +
            # generation_mismatch`` answer: owner mail is durably written before the generation
            # moves, so the local mailbox decides here, never a blind seal.
            generation_mismatch = owner_mail_pending()
    finally:
        if acquired:
            admission_lock.release()
    _drop_fence_binding(ctx)  # also after a refusal or a gap: the next begin re-adopts or reopens
    ctx._task_acceptance_fence_generation_mismatch = generation_mismatch  # local owner facts stand whatever the queue answered
    if result:
        sealed = status == "sealed" or (not status and effective_outcome != "revision")
        ctx._task_acceptance_sealed_fence_token = token if sealed else None
    return result


def _supersede_delivery_acceptance_binding(
    tools: ToolRegistry,
    llm_trace: Dict[str, Any],
    candidate: DeliveryCandidate,
    *,
    reason: str,
) -> bool:
    """Invalidate the exact host verdict bound to a changed delivery candidate.

    The run remains in ``review_runs`` as audit evidence, but neither the
    candidate nor ``review_decision`` may keep pointing at it after answer text
    or answer-invalidating evidence changes.  Negative superseded verdicts stay
    available to the outcome reducer's fail-closed path. An honoured author stop
    binds no candidate or evidence, so it is never superseded here (TZ-2 C4).
    """
    from ouroboros.review_records import recorded_author_stop

    if recorded_author_stop(llm_trace.get("acceptance_decision")):
        return False
    decision = (
        dict(llm_trace.get("review_decision") or {})
        if isinstance(llm_trace.get("review_decision"), dict)
        else {}
    )
    candidate_binding = (
        dict(candidate.acceptance_binding or {})
        if isinstance(candidate.acceptance_binding, dict)
        else {}
    )
    exact_bindings = {
        (str(panel_id), str(binding_hash))
        for panel_id, binding_hash in (
            (candidate_binding.get("panel_id"), candidate_binding.get("binding_hash")),
            (decision.get("panel_id"), decision.get("binding_hash")),
        )
        if panel_id and binding_hash
    }
    run_record: Optional[Dict[str, Any]] = None
    if exact_bindings:
        for run in reversed(llm_trace.get("review_runs") or []):
            if not isinstance(run, dict):
                continue
            if run.get("authority") != "host_root" or run.get("superseded_by_revision"):
                continue
            run_candidate = str(
                run.get("candidate_hash") or run.get("candidate_sha256") or ""
            )
            run_binding = (
                str(run.get("panel_id") or ""),
                str(run.get("binding_hash") or ""),
            )
            if run_candidate != candidate.content_sha256 or run_binding not in exact_bindings:
                continue
            run_record = run
            break

    decision_was_bound = bool(decision.get("panel_id") and decision.get("binding_hash"))
    candidate_was_bound = bool(exact_bindings)
    if run_record is None and not decision_was_bound and not candidate_was_bound:
        return False
    if run_record is not None:
        run_record["superseded_by_revision"] = True
        run_record["superseded_reason"] = reason
        run_record["enforcement_impact"] = "requires_revision"

    for key in ("panel_id", "binding_hash", "panel_reused"):
        decision.pop(key, None)
    decision.update({
        "eligibility": "pending_delivery_acceptance",
        "trigger": reason,
    })
    llm_trace["review_decision"] = decision
    candidate_binding.update({
        "acceptance_status": "unaccepted",
        "authoritative": False,
        "panel_id": "",
        "binding_hash": "",
    })
    candidate_binding.pop("review_evidence_revision", None)
    candidate.acceptance_binding = candidate_binding
    tools._ctx._task_acceptance_reviewed = False
    llm_trace.pop("root_phase_checkpoint", None)
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "delivery_binding_superseded",
        "source": "delivery_candidate_binding",
        "rationale": (
            "The delivery candidate or its evidence binding changed after host "
            "acceptance; the prior panel is retained only as superseded audit evidence."
        ),
    })
    publish_acceptance_checkpoint(tools._ctx, llm_trace)
    return True


def _supersede_task_acceptance_for_owner_followup(
    ctx: Any,
    llm_trace: Dict[str, Any],
    *,
    admission_locked: bool = False,
) -> bool:
    """Release finalization for Main to consume new input, retaining paid evidence."""
    released = _loop()._end_task_acceptance_fence(
        ctx, outcome="revision", admission_locked=admission_locked,
    )
    ctx._task_acceptance_reviewed = False
    ctx._task_acceptance_fence_generation_mismatch = False
    # The panel in flight reviewed the answer to the OLD requirements: it stays
    # custodied and its verdicts still arrive as advice, but the answer Main writes
    # for the follow-up is not a delivery under it — the ordinary path decides.
    ctx._task_acceptance_pending = ""
    llm_trace.pop("root_phase_checkpoint", None)
    llm_trace["review_decision"] = {**dict(llm_trace.get("review_decision") or {}),
        "eligibility": "pending_owner_followup",
        "trigger": "owner_followup_after_acceptance",
    }
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "owner_followup",
        "source": "owner_followup",
        "rationale": "Main must consume the owner follow-up and decide whether the retained review subject changed.",
    })
    publish_acceptance_checkpoint(ctx, llm_trace)
    return released


def _task_acceptance_owner_generation_changed(ctx: Any) -> bool:
    """Check direct and queue-owned owner generations without closing the fence."""

    from ouroboros.loop_messages import owner_source_sha256

    known_source = (getattr(ctx, "_acceptance_ack_source_sha256", "")
                    or getattr(getattr(ctx, "_delivery_candidate", None), "owner_source_sha256", ""))
    if known_source and known_source != owner_source_sha256(ctx):
        return True
    expected_owner = getattr(ctx, "_task_acceptance_owner_generation", None)
    admission_agent = getattr(ctx, "owner_message_admission_agent", None)
    if (
        expected_owner is not None
        and admission_agent is not None
        and int(getattr(admission_agent, "_owner_message_generation", 0) or 0)
        != int(expected_owner)
    ):
        return True
    expected_queue = getattr(ctx, "_task_acceptance_fence_generation", None)
    token = getattr(ctx, "_task_acceptance_fence_token", None)
    inspect = getattr(ctx, "inspect_acceptance_fence", None)
    if expected_queue is None or token is None or not callable(inspect):
        return False
    try:
        state = inspect(token=str(token))
        return bool(
            isinstance(state, dict)
            and int(state.get("owner_message_generation") or 0) != int(expected_queue)
        )
    except Exception as exc:
        # Unknown queue state is not evidence that a new message arrived. Keep
        # the existing acceptance candidate; the inspection failure is disclosed
        # by the shared stamp and must not manufacture a semantic owner change.
        queue_inspection_unknown(ctx, exc)
        return False


def _supersede_task_acceptance_for_evidence_change(
    ctx: Any,
    llm_trace: Dict[str, Any],
    run_record: Optional[Dict[str, Any]],
    reason: str,
    messages: List[Dict[str, Any]],
    emit_progress: Callable[[str], None],
) -> None:
    """Invalidate an acceptance boundary when frozen evidence changes before delivery.

    An honoured author stop is no such boundary: only the author's next decision
    or owner input reopens it (TZ-2 C4)."""
    from ouroboros.review_records import recorded_author_stop

    if recorded_author_stop(llm_trace.get("acceptance_decision")):
        return
    if isinstance(run_record, dict):
        run_record["superseded_by_revision"] = True
        run_record["superseded_reason"] = reason
        run_record["enforcement_impact"] = "requires_revision"
    _loop()._end_task_acceptance_fence(ctx, outcome="revision")
    ctx._task_acceptance_reviewed = False
    ctx._task_acceptance_fence_generation_mismatch = False
    llm_trace.pop("root_phase_checkpoint", None)
    llm_trace["review_decision"] = {
        "eligibility": "pending_evidence_refresh",
        "trigger": reason,
    }
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_REVISION_REQUESTED,
        "reason": "evidence_refresh",
        "source": "host_acceptance_evidence_refresh",
        "rationale": (
            "Task or child evidence changed after acceptance evidence was frozen; "
            "the prior boundary was superseded before it could authorize delivery."
        ),
    })
    _loop()._append_or_merge_user_message(
        messages,
        "[TASK ACCEPTANCE REFRESH] Task or child evidence changed after acceptance "
        "evidence was frozen. Re-read the latest evidence and produce one complete "
        "replacement answer before the next host acceptance review.",
    )
    emit_progress(
        "Task acceptance review superseded: task or child evidence changed before delivery."
    )
    publish_acceptance_checkpoint(ctx, llm_trace)


def _task_acceptance_subtree_snapshot(
    ctx: Any, drive_root: Optional[pathlib.Path], task_id: str,
) -> tuple[bool, List[Dict[str, Any]]]:
    """Return recursive terminal/quiescent state using the existing task SSOT."""
    if drive_root is None:
        try:
            drive_root = pathlib.Path(getattr(ctx, "drive_root"))
        except (TypeError, OSError, ValueError):
            return False, []
    try:
        from ouroboros.task_status import SETTLED_STATUSES, find_child_tasks
        from ouroboros.depth_evidence import task_depth_provenance
        from ouroboros.tools.join_ledger import _child_result_sha256

        meta = getattr(ctx, "task_metadata", {})
        status_root = pathlib.Path(str(
            (meta.get("budget_drive_root") if isinstance(meta, dict) else "")
            or getattr(ctx, "budget_drive_root", "")
            or drive_root
        ))
        rows = find_child_tasks(
            status_root,
            parent_task_id=str(task_id),
            root_task_id=_root_task_id(ctx, task_id),
            exclude_task_id=str(task_id),
            scope="subtree",
        )
        compact = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_task_id = str(row.get("task_id") or row.get("id") or "")
            status = str(row.get("status") or "unknown")
            projected = {
                "task_id": row_task_id,
                "parent_task_id": str(row.get("parent_task_id") or ""),
                "status": status,
                "artifact_status": str(row.get("artifact_status") or ""),
            }
            if depth_provenance := task_depth_provenance(row):
                projected["depth_provenance"] = depth_provenance
            if status in SETTLED_STATUSES:
                projected["child_result_sha256"] = _child_result_sha256(row)
            compact.append(projected)
        # Acceptance needs true quiescence: SETTLED statuses only. A child
        # with a pending durable cancel intent stays non-quiescent until
        # custody settles it (guaranteed by the cancel-intent watchdog).
        queue_rows = [
            {
                "task_id": str(row.get("task_id") or ""),
                "parent_task_id": "",
                "status": str(row.get("status") or "running"),
                "artifact_status": "",
                "source": "supervisor_queue",
            }
            for row in (getattr(ctx, "_task_acceptance_queue_descendants", None) or [])
            if isinstance(row, dict)
        ]
        return (
            not queue_rows and all(row["status"] in SETTLED_STATUSES for row in compact),
            compact + queue_rows,
        )
    except Exception:
        log.debug("Unable to establish task-acceptance subtree quiescence", exc_info=True)
        return False, []


def _mark_root_acceptance_checkpoint(
    ctx: Any, llm_trace: Dict[str, Any], *, status: str, pass_index: int = 0,
) -> None:
    """Minimal in-result phase checkpoint; no parallel acceptance journal."""
    from ouroboros.loop_acceptance_review import _resolve_ctx_lineage

    if not _resolve_ctx_lineage(ctx)["is_root_task"]:
        return
    llm_trace["root_phase_checkpoint"] = {
        "phase": "task_acceptance",
        "status": str(status),
        "pass_index": max(0, int(pass_index)),
        "post_task_synthesis": "pending_once",
    }


def _latch_final_answer_marker(
    llm_trace: Dict[str, Any],
    content: str | None,
    current_tool_calls: list | None = None,
) -> None:
    """Anytime capture for explicit FINAL ANSWER markers.

    Marker-only: do not mine prose. The tool-call count stamp preserves the
    existing stale-answer invariant: later grounding invalidates this fallback
    unless the model emits a newer marker.
    """
    # Opt-in CANDIDATES latch (v6.54.4): an explicit block ("CANDIDATES:" on
    # its own line, "- " items) latches candidate interpretations beside the
    # final answer so the acceptance reviewer can adjudicate ambiguity.
    # Marker-only, like FINAL ANSWER — no prose mining; no block = unchanged.
    text = content or ""
    try:
        lines = text.splitlines()
        marker_idx = next(
            (i for i, line in enumerate(lines) if line.strip() == "CANDIDATES:"),
            None,
        )
        if marker_idx is not None:
            # Marker-only, like FINAL ANSWER (adversarial r2 #4): the block
            # is the "- " items IMMEDIATELY after the marker line; the first
            # non-item line ends it. No substring or distant-bullet harvest.
            candidates: list = []
            for line in lines[marker_idx + 1:]:
                if line.strip().startswith("- "):
                    candidates.append(line.strip()[2:].strip()[:300])
                else:
                    break
            if candidates:
                llm_trace["candidate_answers"] = candidates[:8]
    except Exception:
        pass
    answer = extract_final_answer(text)
    if not answer:
        return
    llm_trace["best_valid_final_answer"] = answer
    del current_tool_calls
    llm_trace["best_valid_final_answer_tools"] = len(llm_trace.get("tool_calls") or [])


def _server_web_allowed_by_task(ctx: Any) -> bool:
    contract = getattr(ctx, "task_contract", {}) if isinstance(getattr(ctx, "task_contract", {}), dict) else {}
    resources = contract.get("allowed_resources") if isinstance(contract.get("allowed_resources"), dict) else {}
    forbidden_names = {"web", "allow_web", "network", "allow_network", "internet", "external_network"}
    return not any(resources.get(name) is False for name in forbidden_names)


ACCEPTANCE_REASON_UNSPECIFIED = "unspecified"


ACCEPTANCE_DECISION_REASONS = (
    "clean_pass",
    "clean_pass_obligations_closed",
    "no_actionable_changes",
    "delivery_binding_superseded",
    "owner_followup",
    "evidence_refresh",
    "improvement_capsule",
    "dialogue_terminal",
    "open_obligations",
    "capsule_spent",
    "improvement_window_closed",
    "reviewer_fail_no_capsule",
    "review_degraded",
    "fence_reopen_failed",
    "infra_failure",
    "author_finish",
    "author_stop",
    "review_outcome_received",
    # #1223: the host could not assemble the acceptance evidence locally, before
    # any reviewer existed. Its own cause, never a reviewer verdict or rework.
    REASON_ACCEPTANCE_PREPARATION_FAILED,
    "admission_close_unconfirmed",  # owner 2A: blocking, clean PASS, the supervisor never confirmed the close
    # An explicit author stop can retain the wallet's exhausted-cycle reason.
    REASON_REVIEW_CYCLES_EXHAUSTED,
    # A-material (2026-08-30): the resubmit carried no changed candidate and no new
    # obligation disposition, so the recorded verdict was replayed for free.
    REASON_IDENTICAL_ACCEPTANCE_REFUSED,
    REASON_DELIVERY_CONTROL_DEGRADED,
    # Owner Q2A: the forced children_unabsorbed rail runs the panel but cannot
    # grant a requested improvement pass; the dangling revision terminalizes.
    "revision_unavailable_on_forced_rail",
    REASON_ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE,
    # Forced-rail acceptance bypass (closed set, outcomes.py SSOT): stamped by
    # `_record_forced_acceptance_bypass` when the panel was owed but a rail fired.
    *sorted(ACCEPTANCE_BYPASS_REASONS),
    ACCEPTANCE_REASON_UNSPECIFIED,
)


def _set_acceptance_decision(llm_trace: Dict[str, Any], decision: Dict[str, Any]) -> None:
    """The ONLY merge point for the host acceptance decision (v6.78.0, owner Q23).

    Every host exit funnels here and leaves in one of the three canonical
    owner-facing states (``ACCEPTANCE_DECISION_STATUSES``) plus a typed
    ``reason`` naming WHICH exit. A status outside the trio fails closed to
    ``finalized_unaccepted`` with its raw token surviving as ``reason`` — no
    fourth state, no lost token. The author's historical stance survives;
    owner/evidence supersession consumes its controlling finish intent and act,
    so a later host exit never re-reads an earlier stop as its own (TZ-2 C4)."""
    previous = llm_trace.get("acceptance_decision") if isinstance(llm_trace.get("acceptance_decision"), dict) else {}
    merged = dict(decision)
    merged.setdefault("enforcement", _loop().get_review_enforcement())
    status = str(merged.get("status") or "")
    reason = str(merged.get("reason") or "")
    if status not in ACCEPTANCE_DECISION_STATUSES:
        merged["status"] = ACCEPTANCE_FINALIZED_UNACCEPTED
        reason = reason or status or ACCEPTANCE_REASON_UNSPECIFIED
    merged["reason"] = reason
    superseded = reason in {"owner_followup", "evidence_refresh"}
    for key in ("agent_disposition", "agent_rationale", "author_disposition") + (() if superseded else ("author_action",)):
        if previous.get(key) and not merged.get(key):
            merged[key] = previous.get(key)
    if not superseded and reason not in {"author_finish", "author_stop"} and not (
        reason == "delivery_binding_superseded" and previous.get("reason") == "author_finish"
    ) and not merged.get("author_disposition") and previous.get("agent_finish_intent"):
        merged["agent_finish_intent"] = previous["agent_finish_intent"]
    # #1224: while the host's own LOCAL acceptance preparation is unresolved, every
    # decision carries that fact, so a rail's cause and the local failure are both
    # stated instead of one replacing the other. A resolved incident stops riding
    # along (its history stays on the trace), and an explicit one always wins.
    if not merged.get("acceptance_incident"):
        from ouroboros.acceptance_preparation import STATUS_FAILED, incident_projection

        record = llm_trace.get("acceptance_preparation")
        if isinstance(record, dict) and str(record.get("status") or "") == STATUS_FAILED:
            merged["acceptance_incident"] = incident_projection(record)
    incident = merged.get("acceptance_incident") or {}
    if incident.get("status") == "failed" and incident.get("stage") == "preparation":
        from ouroboros.acceptance_preparation import LOCAL_PREPARATION_ORIGIN

        merged.setdefault("origin", LOCAL_PREPARATION_ORIGIN)
    llm_trace["acceptance_decision"] = merged
    from ouroboros.acceptance_preparation import TASK_ONLY_ORIGINS

    if merged.get("origin") in TASK_ONLY_ORIGINS:
        return  # A host processing failure is no panel's applied reviewer decision.
    # A full applied-review source includes the host's actual decision, not
    # only the provider's earlier response. The decision above stays authority.
    for run in reversed(llm_trace.get("review_runs") or []):
        if isinstance(run, dict) and run.get("authority") == "host_root":
            author = merged.get("author_disposition")
            if reason == "author_finish" and isinstance(author, dict) and author.get("subject_hash") != run.get("binding_hash"):
                break  # A revised author subject is a task fact, not this older panel's decision.
            run["applied_decision"] = dict(merged)
            break


def merge_agent_acceptance_stance(trace: Dict[str, Any], decision: dict, ctx: Any) -> None:
    """Record one explicit stance after feedback, separately from the host verdict."""
    previous = trace.get("acceptance_decision")
    merged = dict(previous) if isinstance(previous, dict) else {}
    merged.setdefault("source", "agent_task_acceptance_review_tool")
    merged["agent_disposition"] = str(decision.get("disposition") or "")
    merged["agent_rationale"] = truncate_review_artifact(str(decision.get("rationale") or ""), limit=500)
    action = str(decision.get("author_action") or "finish")
    merged["author_action"] = action
    merged.pop("agent_finish_intent", None)
    feedback = next((run for run in reversed(trace.get("review_runs") or [])
                     if isinstance(run, dict) and run.get("authority") == "host_root"
                     and run.get("feedback_delivered")), None)
    outcome = trace.get("acceptance_review_outcome") or {}
    if not feedback and outcome.get("feedback_delivered"):
        feedback = outcome
    explicit = ctx is not None and decision.get("explicit_finish") is True and bool(merged["agent_rationale"].strip())
    from ouroboros.acceptance_preparation import STATUS_FAILED, preparation_exposed

    incident = trace.get("acceptance_preparation")
    exposed_incident = (isinstance(incident, dict) and str(incident.get("status") or "") == STATUS_FAILED
                        and preparation_exposed(incident))
    if explicit and exposed_incident:
        # An exposed LOCAL preparation failure binds the stance to the MATERIAL
        # and the ATTEMPT it was informed about: there is no reviewer binding to
        # bind to, and the author must not be routed back through the broken
        # builder — or the same failing evidence fingerprint — in order to
        # finish or stop. Its own branch, so no fallible computation sits here.
        from ouroboros.loop_messages import owner_source_sha256

        merged["agent_finish_intent"] = {
            "author_action": action,
            "preparation_identity": str(incident.get("source_identity") or ""),
            "incident_id": str(incident.get("incident_id") or ""),
            "incident_attempt": int(incident.get("attempts") or 0),
            "candidate_sha256": str(getattr(getattr(ctx, "_delivery_candidate", None), "content_sha256", "")),
            "owner_source_sha256": owner_source_sha256(ctx),
        }
    elif explicit and (feedback or action == "stop"):
        from ouroboros.loop_delivery import observed_delivery_evidence

        merged["agent_finish_intent"] = {
            "review_binding_hash": str((feedback or {}).get("binding_hash") or ""),
            "author_action": action,
            "tool_count": len(trace.get("tool_calls") or []),
            "owner_directives": len(getattr(ctx, "_owner_directives", []) or []),
            # UNKNOWN ("") binds to nothing: the host pass compares it against a
            # fresh read, so an unverifiable stance is never honoured as ready.
            "evidence_fingerprint": observed_delivery_evidence(ctx, trace),
        }
    from ouroboros.review_records import recorded_author_stop

    if merged.get("agent_finish_intent") and recorded_author_stop(previous):
        ctx._task_acceptance_reviewed = False  # the author's next decision reopens an honoured stop
    trace["acceptance_decision"] = merged


def _collect_acceptance_obligations(llm_trace: Dict[str, Any], result: Any) -> None:
    """Typed PER-TASK obligations from critical contributing findings (v6.54.4).

    Blocking path after review eligibility. Each critical finding WITH a concrete
    recommendation becomes one open obligation in llm_trace (never the durable
    commit review_state — a separate SSOT). Clean finalization asks for an
    agent disposition per obligation (v6.54.0); time/pass gates and the
    forced-finalization escape hatches bound the loop, so a deadline never
    hangs here. v6.60.0 widening (S1-lite, owner quiz 18b): when the AGGREGATE
    verdict itself is failing — signal FAIL, or worst tier
    blocked_with_evidence — contributing reviewers' HIGH findings with a
    concrete recommendation also become obligations (the PB incident). On a
    PASS (incl. with-dissent) the bar stays critical-only, so the blocking
    lane cannot creep into taxing clean runs with hygiene items."""
    import hashlib

    from ouroboros.review_substrate import _contributing_actors, aggregate_outcome_tier

    contributing = {str(a.get("slot_id", "")) for a in _contributing_actors(result)}
    obligations = llm_trace.setdefault("acceptance_obligations", [])
    by_id = {str(o.get("id")): o for o in obligations if isinstance(o, dict)}
    # No contributing actors (all parse-degraded / no quorum) => no
    # authoritative verdict: manufacture NO blocking obligations — else one
    # parse-degraded slot's critical finding would gate finalization, the
    # class the capsule refuses (r1); obligations ride CONTRIBUTING slots.
    if not contributing:
        return
    _agg_failing = (
        str(getattr(result, "aggregate_signal", "") or "").upper() == "FAIL"
        or aggregate_outcome_tier(result) == "blocked_with_evidence"
    )
    _obligation_severities = {"critical", "high"} if _agg_failing else {"critical"}
    # Ids already created or reopened by THIS panel pass: slots of one panel
    # routinely raise the same finding (typed re_raise copies the catalog
    # id); without it the second slot's dupe would falsely bump
    # reopened_count and overwrite reviewer_rebuttal_response (fable r1 #1).
    touched_this_pass: set[str] = set()
    for finding in (getattr(result, "parsed_findings", None) or []):
        if not isinstance(finding, dict):
            continue
        if str(finding.get("severity") or "").strip().lower() not in _obligation_severities:
            continue
        if str(finding.get("slot_id", "")) not in contributing:
            continue
        recommendation = " ".join(str(finding.get("recommendation") or "").split()).strip()
        if not recommendation:
            continue
        item = str(finding.get("item") or "finding").strip()
        # v6.74.0 (A3): obligation identity is reviewer-authored. A re_raise
        # MUST name an existing catalog id (the host validates existence
        # only); a missing/unknown id fails closed to `new` with a disclosed
        # note — a reworded re-raise cannot mint a fresh hash id.
        kind = str(finding.get("disposition_kind") or "").strip().lower()
        claimed_id = str(finding.get("obligation_id") or "").strip()
        unbound_note = ""
        if kind == "re_raise":
            row = by_id.get(claimed_id)
            if row is not None:
                if claimed_id not in touched_this_pass:
                    touched_this_pass.add(claimed_id)
                    _reopen_obligation_row(row, finding)
                continue
            unbound_note = f"re_raise_unbound:{claimed_id or 'missing_id'}"
        oid = "ob-" + hashlib.sha256(
            json.dumps([item, recommendation], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        if oid in by_id:
            # Reviewer-authored identity (triad r2, sol): only an UNTYPED
            # legacy finding may reopen via byte-identical text (v6.71.1
            # compat). A typed "new"/unbound "re_raise" matching a settled row
            # must NOT resurrect the settled rebuttal — sloppiness DISCLOSED.
            row = by_id[oid]
            if not kind and oid not in touched_this_pass:
                touched_this_pass.add(oid)
                _reopen_obligation_row(row, finding)
            elif kind:
                notes = row.setdefault("notes", [])
                note = unbound_note or f"typed_new_matched_existing:{oid}"
                if note not in notes:
                    notes.append(note)
            continue
        row = {
            "id": oid,
            "item": item,
            "recommendation": recommendation,
            "status": "open",
            "disposition": "",
            "disposition_reason": "",
        }
        if unbound_note:
            row["notes"] = [unbound_note]
        by_id[oid] = row
        touched_this_pass.add(oid)
        obligations.append(row)


def _reopen_obligation_row(row: Dict[str, Any], finding: Dict[str, Any]) -> None:
    """Reopen a re-raised obligation WITHOUT wiping the agent's argument (A3).

    The prior disposition/reason survive as ``previous_disposition`` /
    ``previous_reason`` and ``reopened_count`` increments, so the agent can see
    its rebuttal was overruled (previously indistinguishable from a fresh
    finding) and the next reviewer receives the prior argument to adjudicate.
    The reviewer's stated reason for maintaining the finding rides along."""
    if str(row.get("disposition") or "").strip() or str(row.get("status") or "") == "agent_disposed":
        row["previous_disposition"] = str(
            row.get("disposition") or row.get("status") or ""
        )
        row["previous_reason"] = str(row.get("disposition_reason") or "")
    row["reopened_count"] = int(row.get("reopened_count") or 0) + 1
    row["disposition"] = ""
    row["disposition_reason"] = ""
    row["status"] = "open"
    reviewer_response = " ".join(str(finding.get("evidence") or "").split()).strip()
    if reviewer_response:
        row["reviewer_rebuttal_response"] = truncate_review_artifact(
            reviewer_response, limit=600,
        )


def _open_acceptance_obligations(llm_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    # An agent-filed disposition (status="agent_disposed") is a
    # CLAIM/rebuttal, not a settlement: the row is pending until a host panel
    # adjudicates (PASS settles; re-raise reopens). SSOT: review_evidence.
    from ouroboros.review_evidence import obligation_is_pending

    return [
        o for o in (llm_trace.get("acceptance_obligations") or [])
        if obligation_is_pending(o)
    ]


def _dispose_obligations_on_clean_pass(
    llm_trace: Dict[str, Any],
    result: Any,
    open_obligations: List[Dict[str, Any]],
    dissent_noted: bool,
) -> bool:
    """If the re-review is a CLEAN PASS (aggregate PASS and not degraded), close
    the open obligations as disposed_by_re_review and record the accepted verdict;
    return True. A DEGRADED/no-quorum run proves nothing → returns False, leaving
    the honest best-effort labeling to the caller."""
    if not open_obligations:
        return False
    from ouroboros.review_substrate import task_acceptance_is_clean

    if not task_acceptance_is_clean(result):
        return False
    for ob in open_obligations:
        if str(ob.get("status") or "") == "agent_disposed":
            # The clean panel ACCEPTED the agent's filed disposition (a
            # rebuttal it chose not to re-raise): keep that disposition/reason
            # as provenance, record the host settlement distinctly (r6) —
            # never rewrite a rejected rebuttal into "addressed by revision".
            ob["status"] = "disposed_rebuttal_accepted"
            continue
        ob["disposition"] = "addressed"
        ob["disposition_reason"] = "resolved by revision: the clean re-review returned no findings"
        ob["status"] = "disposed_by_re_review"
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_ACCEPTED,
        "reason": "clean_pass_obligations_closed",
        "source": "task_acceptance_review",
        "rationale": "Clean PASS re-review; open obligations closed by the revision (dissent, if any, stays advisory).",
        "dissent_noted": dissent_noted,
    })
    return True


def _format_obligations_clause(open_obligations: List[Dict[str, Any]]) -> str:
    # v6.74.0 (A4): disagreement is recorded ONLY via
    # obligation_dispositions — the old "or address them directly" prose read
    # as a third channel; fixing the work just makes the next panel clean.
    if not open_obligations:
        return ""
    lines = [
        "",
        "OPEN OBLIGATIONS (blocking review policy). Either FIX the work so the next review "
        "panel finds it clean, or record your disagreement via the task_acceptance_review "
        "tool's obligation_dispositions (addressed / rejected / deferred + reason) — "
        "dispositions are the ONLY channel the reviewer adjudicates:",
    ]
    for o in open_obligations[:5]:
        line = f"  {o.get('id')}: {o.get('item')} — {o.get('recommendation')}"
        reopened = int(o.get("reopened_count") or 0)
        if reopened > 0:
            line += f" [re-raised ×{reopened}"
            if str(o.get("previous_disposition") or "").strip():
                line += (
                    f"; your '{o.get('previous_disposition')}' rebuttal was overruled"
                )
                response = str(o.get("reviewer_rebuttal_response") or "").strip()
                if response:
                    line += f" — reviewer: {response}"
            line += "]"
        lines.append(line)
    if len(open_obligations) > 5:
        lines.append(f"  (+{len(open_obligations) - 5} more in the task record)")
    return "\n".join(lines)


def terminalize_dangling_revision(llm_trace: Dict[str, Any], *, rail: str) -> bool:
    """Close a dangling ``revision_requested`` when a forced rail fires.

    A forced rail cannot take another model round, so a recorded revision
    request would otherwise promise a pass that never comes. The prior reason is
    named in the rationale because "the panel requested an improvement pass" is
    false for the superseded-binding shape. Returns True when it terminalized.

    No bypass reason is stamped: the panel really ran. ``accepted`` and
    ``finalized_unaccepted`` decisions are never overwritten, and the
    reason/status pair stays outside the blocked-terminal set, so the objective
    remains best_effort.
    """
    decision = llm_trace.get("acceptance_decision")
    decision = decision if isinstance(decision, dict) else {}
    if str(decision.get("status") or "") != ACCEPTANCE_REVISION_REQUESTED:
        return False
    prior = str(decision.get("reason") or "") or ACCEPTANCE_REASON_UNSPECIFIED
    # A revision that came from the HOST's own local preparation failure is not a
    # reviewer's rework request, so it must not terminalize as one: the record
    # would read "the requested rework never happened" with no reviewer in it
    # (#1224). It keeps its own cause, and the rail is named beside it — the
    # rationale here, and the owner-facing clause through `acceptance_incident`.
    local_failure = prior == REASON_ACCEPTANCE_PREPARATION_FAILED
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "reason": REASON_ACCEPTANCE_PREPARATION_FAILED if local_failure
                  else "revision_unavailable_on_forced_rail",
        **({"origin": "local_acceptance_preparation"} if local_failure else {}),
        **({"acceptance_incident": decision["acceptance_incident"]}
           if local_failure and decision.get("acceptance_incident") else {}),
        "source": "forced_finalization",
        "rationale": (
            f"Acceptance evidence could not be assembled locally; the forced {rail} rail "
            "then ended the task. No reviewer rework was requested."
            if local_failure else
            f"The acceptance decision was {prior}; the forced {rail} rail cannot "
            "take another model round."
        ),
    })
    return True


def _record_forced_acceptance_bypass(
    ctx: _RoundLimitContext,
    llm_trace: Dict[str, Any],
    reason_code: str,
) -> None:
    """Typed acceptance record on a forced rail — a LEDGER write, never a gate.

    Forced exits used to leave the review axis at {skipped, not_eligible,
    run_count:0} — indistinguishable from "no panel warranted". Stamp the
    terminal truth instead: eligibility is evaluated PURE against the live
    trace (no fence begin, quiescence wait, panel, model round, or prompt text
    — forced exits are the v6.29 honesty/salvage shelf, byte-identical), and
    the turn's own panel is COLLECTED at $0 before anything is recorded, so a
    closed-enum bypass reason (`ACCEPTANCE_BYPASS_REASON_BY_RAIL`) lands only
    on a turn whose owed panel never ran (`forced_rail_panel_verdict` owns what
    a collected one says). Reason tokens stay ledger-only (v6.61.4
    token-parroting class). Never raises."""
    rail_reason = ACCEPTANCE_BYPASS_REASON_BY_RAIL.get(str(reason_code or ""))
    if rail_reason is None:
        return
    # A rail that deliberately cleared the failure state (a confirmed swarm routing
    # handoff) terminalized nothing reviewable here — the admitted managed task gets
    # its own acceptance lifecycle.
    if not str(ctx.accumulated_usage.get("reason_code") or ""):
        return
    tools_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    if tools_ctx is None:
        return
    # A recorded host decision (canonical status, NOT the status-less agent
    # stance merged on a deferral) wins; the bypass record exists only for the
    # no-host-verdict shape; `_set_acceptance_decision` stamps.
    decision = llm_trace.get("acceptance_decision")
    if isinstance(decision, dict) and str(decision.get("status") or "") in ACCEPTANCE_DECISION_STATUSES:
        # A revision request is not a terminal host decision: this rail cannot
        # take the pass it promised, so close it instead of leaving it dangling.
        terminalize_dangling_revision(llm_trace, rail=str(reason_code or ""))
        publish_acceptance_checkpoint(tools_ctx, llm_trace, task_id=ctx.task_id)
        return
    if getattr(tools_ctx, "_task_acceptance_reviewed", False):
        return
    trigger = f"bypassed_{reason_code}"
    try:
        from ouroboros.loop_acceptance_review import _resolve_ctx_lineage

        lineage = _resolve_ctx_lineage(tools_ctx, str(ctx.task_id or ""))
        eligible, probe_trigger = _loop()._task_acceptance_eligible(
            _loop().get_task_review_mode(),
            llm_trace,
            bool(getattr(tools_ctx, "is_direct_chat", False)),
            is_root_task=bool(lineage["is_root_task"]),
            task_contract=(
                tools_ctx.task_contract
                if isinstance(getattr(tools_ctx, "task_contract", None), dict)
                else {}
            ),
        )
    except Exception:
        # A mid-round dying trace may not support the probe; record the honest
        # unknown instead of crashing the salvage path.
        log.debug("Forced acceptance-bypass eligibility probe failed", exc_info=True)
        llm_trace["review_decision"] = {"eligibility": "unknown", "trigger": trigger}
        return
    if not eligible:
        # Explicitly "no panel warranted" — now distinguishable from "not evaluated".
        llm_trace["review_decision"] = {"eligibility": "not_eligible", "trigger": probe_trigger}
        return
    llm_trace["review_decision"] = {"eligibility": "eligible", "trigger": trigger}
    _loop()._set_acceptance_decision(llm_trace, {
        "status": ACCEPTANCE_FINALIZED_UNACCEPTED,
        "source": "forced_finalization",
        **forced_rail_panel_verdict(tools_ctx, llm_trace, rail_reason),
    })
