"""Atomic custody and zero-run decisions before delegated transport."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Tuple

from ouroboros import delegate_custody as custody


def claimed_start_request(
    drive: pathlib.Path, *, actor_ctx: Any = None, initial_bootstrap: bool = False,
    serving_store_id: Any = "",
    **request_row: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Record an independent invocation without limiting other runs or captures.

    A replay of an already recorded invocation (``retry_of``) goes only to the engine store
    it was submitted to (``delegate_pending.replay_store_unproven``). The row records the
    handshake's store as ``handshake_store_id``, an observation and never admission proof;
    a fresh start's ``daemon_store_id`` stays empty until its POST receipt names a store.

    A start that continues IN a predecessor's snapshot (``capture_id``) claims
    under that snapshot's one disposition lock, so no apply/reject of the
    predecessor can interleave with its hand-over (``delegate_continuation``).
    A start bound to a managed work engine whose request names a path that
    engine does not see is refused before its row (``execution_domain``).
    """
    from ouroboros.execution_domain import is_work_ref, request_paths_refusal
    from ouroboros.usage_ledger import BillingAuthorityUnavailable

    if is_work_ref(request_row.get("engine_ref")) and (
            unseen := request_paths_refusal(request_row.get("request") or {})):
        from ouroboros.delegate_invocation import invocation_record

        fresh = invocation_record(drive, str(request_row.get("invocation_id") or "")) is None
        return False, {"reason": "work_path_unmapped", "detail": unseen, **({"definitely_unrun": True} if fresh else {})}
    # This is only the identity observed during the current handshake. It
    # distinguishes a legacy identity-less engine from a named engine that
    # changed identity between attempts; it is never admitting-store proof.
    request_row = dict(request_row)
    request_row.setdefault("handshake_store_id", str(serving_store_id or ""))
    request_row.setdefault("daemon_store_id", "")
    request_row.setdefault("store_identity_unproven", True)
    try:
        if request_row.get("capture_id") and request_row.get("continuation_of"):
            from ouroboros.delegate_continuation import snapshot_handover

            with snapshot_handover(drive, request_row) as refusal:
                if refusal:
                    return False, refusal
                return _claimed(drive, actor_ctx, initial_bootstrap, request_row, serving_store_id)
        return _claimed(drive, actor_ctx, initial_bootstrap, request_row, serving_store_id)
    except BillingAuthorityUnavailable as exc:
        # Refused before any START row or POST; not a budget limit.
        return False, {"reason": "billing_authority_unavailable", "billing_authority": exc.kind, "detail": (
            f"{exc}. No delegated session was started. "
            + ("This task's billing lineage is missing or not its own; neither waiting nor a budget change repairs it."
               if exc.kind == "refused" else
               "The authority could not be read; a later start rereads it once the cause is fixed."))}


def _claimed(drive: pathlib.Path, actor_ctx: Any, initial_bootstrap: bool,
             request_row: Dict[str, Any], serving_store_id: Any = "") -> Tuple[bool, Dict[str, Any]]:
    task_id = str(request_row.get("task_id") or "").strip()
    try:
        with custody.actor_decision_lock(drive, task_id):
            if custody.custody_log_unreadable(drive):
                return False, {
                    "reason": "replacement_custody_unknown",
                    "custody_log_unreadable": True,
                    "detail": "The host cannot read the actor's custody authority.",
                }
            bootstrap = (
                getattr(actor_ctx, "_configured_actor_bootstrap", None)
                if actor_ctx is not None else None
            )
            if isinstance(bootstrap, dict):
                from ouroboros.subagent_bootstrap import _durable_zero_run_receipt

                gaps: set[str] = set()
                zero_run = _durable_zero_run_receipt(actor_ctx, gap_reasons=gaps)
                if zero_run:
                    return False, {
                        "reason": "zero_run_already_recorded",
                        "detail": (
                            "The actor recorded a terminal delegation_zero_run decision "
                            "before this physical start could be claimed."
                        ),
                        "zero_run_decision": str(
                            zero_run.get("zero_run_decision") or "unknown"
                        ),
                    }
                if gaps:
                    return False, {
                        "reason": "zero_run_evidence_unavailable",
                        "detail": (
                            "The host cannot prove whether the actor already recorded "
                            "a terminal delegation_zero_run decision."
                        ),
                        "zero_run_evidence_status": "unknown",
                        "zero_run_evidence_gaps": sorted(gaps),
                    }
            prior = custody.invocation_record(drive, str(request_row.get("invocation_id") or ""))
            # The new claim's empty daemon_store_id describes this POST's
            # unproven admission, not the store that admitted a prior retry.
            # Compare the prior durable facts with the current handshake only.
            if prior and (unproven := custody.replay_store_unproven(prior, serving_store_id)):
                return False, {"reason": "invocation_store_unproven", "detail": (
                    f"Nothing was re-sent: {unproven}. The recorded start stays pending and its outcome "
                    "unknown; this refusal is no evidence that its first POST started no run.")}
            if initial_bootstrap:
                from ouroboros.delegate_bootstrap_identity import claim_initial_invocation

                refusal = claim_initial_invocation(drive, task_id, request_row)
                if refusal:
                    return False, refusal
            return custody.record_start_requested(drive, **request_row), {}
    except (OSError, TimeoutError, ValueError) as exc:
        return False, {
            "reason": "replacement_custody_unknown",
            "detail": f"actor decision claim unavailable: {type(exc).__name__}: {exc}. This call sent no POST.",
        }
