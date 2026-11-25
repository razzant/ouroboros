"""Recovery and pre-dispatch custody helpers for delegated review sessions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)
_SESSION_ANSWER = "review_session_answer"


def owned_started_review_custody(
    custody: Any,
    custody_drive: Any,
    record: Dict[str, Any],
    claimant_task_id: str,
) -> tuple[str, Any]:
    """Return existing run custody only after exact owner corroboration."""
    from ouroboros.review_execution import ReviewRouteUnavailable

    run_id = str(record.get("run_id") or "")
    ownership, found = custody.lookup(custody_drive, claimant_task_id, run_id)
    invocation_owner = str(record.get("task_id") or "")
    custody_owner = str(getattr(found, "task_id", "") or "")
    if (
        ownership != custody.OWNED
        or found is None
        or invocation_owner != claimant_task_id
        or custody_owner != claimant_task_id
    ):
        raise ReviewRouteUnavailable(
            "delegated review started-run recovery could not corroborate "
            f"ownership for run {run_id} (lookup={ownership}, "
            f"claimant={claimant_task_id!r}, invocation_owner={invocation_owner!r}, "
            f"custody_owner={custody_owner!r})",
            code="review_recovery_ownership_unverified",
        )
    return run_id, found


def recorded_review_shape(run_request: Any) -> Optional[Any]:
    """The run shape a stored review request recorded, or None when no review runs it.

    Ask/readonly is every request recorded before configured reviewer rights and
    every read-only seat. A seat with configured ``workspace_write``/``full``
    access recorded Agent mode with its private checking copy as the delegated
    live ``workspaceRoot``. Recovery replays exactly this shape — never today's
    configuration — so an old read-only invocation is never relabelled.
    """
    from ouroboros.subagents import DelegatedRunShape

    mode, access = run_request.get("mode"), run_request.get("access")
    execution = run_request.get("execution")
    if mode == "ask" and access == "readonly" and "execution" not in run_request:
        return DelegatedRunShape(access="readonly", mode="ask")
    if mode == "agent" and access in {"workspace_write", "full"} and (run_request.get("_use_thread") is True and (
            "execution" not in run_request) or isinstance(execution, dict) and execution.get("isolation") == "live"
            and execution.get("delegated") is True and str(execution.get("workspaceRoot") or "")):
        return DelegatedRunShape(access=access, mode="agent", isolation="live", delegated=True,
                                 investigation=True)
    return None


def review_recovery_facts(
    record: Dict[str, Any],
    run_request: Any,
    started_custody: Any,
    *,
    prompt: str,
    root: str,
    claimant_task_id: str,
    claimant_surface: str,
    claimant_slot_id: str,
    claimant_operation_id: str,
) -> tuple[Any, str, str, str, bool]:
    """Validate one stored request and restore immutable delivery facts."""
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.subagents import DelegationRoute

    if not isinstance(run_request, dict) or not run_request:
        raise ReviewRouteUnavailable(
            "delegated review recovery has no canonical stored request; "
            "the existing invocation is not re-derived",
            code="review_recovery_request_missing",
        )
    invocation_owner = str(record.get("task_id") or "")
    if invocation_owner != claimant_task_id:
        raise ReviewRouteUnavailable(
            "delegated review pending-invocation recovery could not corroborate "
            f"ownership (claimant={claimant_task_id!r}, "
            f"invocation_owner={invocation_owner!r})",
            code="review_recovery_ownership_unverified",
        )
    stored_binding = (
        str(record.get("surface") or ""), str(record.get("slot_id") or ""),
        str(record.get("operation_id") or ""),
    )
    claimant_binding = (
        str(claimant_surface or ""), str(claimant_slot_id or ""),
        str(claimant_operation_id or ""),
    )
    if stored_binding != claimant_binding:
        raise ReviewRouteUnavailable(
            "delegated review retry token belongs to a different surface, slot, "
            "or physical operation; the recorded invocation is not replayed",
            code="review_recovery_request_mismatch",
        )
    scope = run_request.get("scope")
    route_id = str(run_request.get("primaryHarness") or "")
    if (
        not isinstance(scope, dict)
        or str(scope.get("kind") or "") != "project"
        or not str(scope.get("root") or "")
        or not route_id
        or run_request.get("harnesses") != [route_id]
        or str(record.get("route") or "") != route_id
        or str(run_request.get("authPreference") or "") != "subscription"
        or recorded_review_shape(run_request) is None
    ):
        raise ReviewRouteUnavailable(
            "delegated review recovery has an invalid canonical stored request; "
            "the durable retry token remains untouched and nothing was re-posted",
            code="review_recovery_request_invalid",
        )
    stored_root = str(scope.get("root") or "")
    if str(run_request.get("prompt") or "") != prompt or stored_root != root:
        raise ReviewRouteUnavailable(
            "delegated review retry replays a recorded invocation whose "
            f"{'prompt' if stored_root == root else 'session root'} differs from "
            "this call's; the durable retry token remains untouched and the "
            "recorded invocation is not replayed against a different review",
            code="review_recovery_request_mismatch",
        )
    if started_custody is not None:
        route_id = str(started_custody.route_id or "")
        model = str(started_custody.model or "")
        project_id = str(started_custody.project_id or "")
        project_owned = bool(started_custody.project_owned)
        key = str(started_custody.idempotency_key or "")
    else:
        model = str(run_request.get("model") or "")
        project_id = str(record.get("project_id") or "")
        project_owned = bool(record.get("project_owned"))
        key = str(record.get("idempotency_key") or "")
    route = DelegationRoute(
        route_id=route_id,
        model=model,
        effort=str(run_request.get("effort") or ""),
        profile_id=str(run_request.get("credentialProfileId") or ""),
    )
    existing_project = "" if project_owned else project_id
    return route, project_id, existing_project, key, "outputSchema" in run_request


def checkpoint_pending_invocation(
    *,
    checkpoint: Any,
    invocation_id: str,
    state: Dict[str, Any],
    on_failure: Callable[[], None],
) -> None:
    """Bind a pending invocation to its durable slot before provider POST.

    The binding waits out a busy review-state lock itself. A refusal names its cause: a reserved
    slot that cannot take this token keeps its own reason (``review_custody_slot_refused``); a
    review state that could not be written keeps its lock or I/O facts
    (``review_custody_checkpoint_unwritable``)."""
    if not callable(checkpoint):
        return
    try:
        checkpoint(invocation_id)
    except Exception as exc:
        from ouroboros.review_execution import ReviewRouteUnavailable
        from ouroboros.gateways.claudexor import run_failure_cause
        from ouroboros.review_state import ReservedReviewSlotRefused

        cause = str(getattr(exc, "reported_cause", "") or f"{type(exc).__name__}: {exc}")
        try:
            on_failure()
        except Exception as cleanup_error:
            cause += f"; cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
        state.pop("pending_invocation_id", None)
        if isinstance(exc, ReservedReviewSlotRefused):
            refusal = ReviewRouteUnavailable(
                f"the delegated review invocation was refused by its reserved commit-review slot ({exc}); "
                "no run was started", code="review_custody_slot_refused")
        else:
            refusal = ReviewRouteUnavailable(
                "the delegated review invocation could not be bound to its durable "
                "commit-review slot before dispatch; no run was started",
                code="review_custody_checkpoint_unwritable",
            )
        refusal.reported_cause = run_failure_cause({"safeMessage": cause})
        raise refusal from exc


def exact_recovery_join(executor: Any, custody_root: Any, retry_state: Optional[Dict[str, Any]],
                        operation_id: str, slot: Any, *, replay: bool) -> bool:
    """Is this seat an exact delegated recovery? When it only collects, restore its RAW request.

    Such a recovery has already crossed the paid boundary: a session seat carrying
    both its pending invocation token and its operation id. A collection (``replay``)
    sends nothing of its own: the RAW request recorded before the POST is the
    transport source, never a prompt rebuilt from mutable history or the redacted,
    overwritable ``{operation}_prompt`` record (#1582).
    """
    from ouroboros.delegate_custody import invocation_record

    token = str((retry_state or {}).get("pending_invocation_id") or "").strip()
    session = str(getattr(slot.route, "value", slot.route) or "") == "agent_session"
    if not (token and str(operation_id or "").strip() and session):
        return False
    stored = (invocation_record(custody_root, token) or {}).get("request") if replay else None
    if isinstance(stored, dict) and isinstance(stored.get("prompt"), str):
        executor._session_prompt = stored["prompt"]
    return True


def validate_producer_invocation(root: Any, token: str, expected: dict, payload: dict,
                                outcome: dict, *, error_source: bool) -> None:
    """Session-specific proof behind a complete, already-bound producer outcome.

    Only positive preparation provenance needs no START_REQUESTED row. An absent
    token, contradictory start facts or a legacy source never establish that proof.
    """
    from ouroboros.delegate_custody import invocation_record

    usage = payload.get("usage") or {}
    if (error_source and usage.get("delegated_start_unrequested") is True
            and outcome.get("status") == "error" and bool(outcome.get("error"))
            and not (token or usage.get("delegated_run_id") or usage.get("delegated_run_started"))):
        return
    invocation = invocation_record(root, token) if token else None
    if invocation is not None and invocation.get("state") == "legacy_gap":
        raise ValueError(f"legacy_gap: {invocation['reason']}; completed producer custody needs repair")
    # A binding names the facts its request carried: a request without a reconciliation
    # identity binds no root, which is an unbound fact, never a contradiction.
    if not invocation or any(str(invocation.get(k) or "") != str(expected[k] or "")
                             for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")
                             if k in expected):
        raise ValueError("completed producer has no exact delegated invocation")
    run_id = str(usage.get("delegated_run_id") or "")
    if run_id and run_id != str(invocation.get("run_id") or ""):
        if not review_continuation_descends_from(root, run_id, invocation):
            raise ValueError("completed producer delegated run mismatch")


def review_continuation_descends_from(root: Any, run_id: str, anchor: dict) -> bool:
    """Prove that a collected successor belongs to the slot's original invocation."""
    from ouroboros import delegate_custody as custody

    seen = set()
    while run_id and run_id not in seen:
        if run_id == str(anchor.get("run_id") or ""):
            return True
        seen.add(run_id)
        ownership, entry = custody.lookup(root, anchor["task_id"], run_id)
        if ownership != custody.OWNED or entry is None or not entry.continuation_of:
            return False
        record = custody.invocation_record(root, entry.invocation_id)
        if not record or any(record.get(k) != anchor.get(k)
                             for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")):
            return False
        if record.get("request") != {"continueFrom": entry.continuation_of}:
            return False
        run_id = entry.continuation_of
    return False


def join_recorded_review_continuation(gateway: Any, root: Any, entry: Any, detail: dict, invocation: Any,
                                      *, seen: set[str], checking: Optional[Dict[str, Any]] = None) -> Any:
    """Join a successor an older host recorded automatically under this seat's anchor; never create one.

    Continuation is now the author's explicit choice (``review_continuation``): a new operation
    collected through the existing exact recovery. A legacy successor already recorded under
    ``("review_continue", invocation, run)`` is joined when started, or read through the engine's
    noncreating lookup using its recorded body and key. An early 4xx can belong to an accepted
    command with no native run yet: its job id binds just like a late 202 handle. A missing,
    refused, unavailable or foreign-store lookup leaves the start pending and the predecessor's
    original stop untouched; it never licenses a creating POST, even after the owner ended.
    A bound successor whose checking copy is missing or foreign is still returned for observation: the gap
    rides ``checking["legacy_capture_gap"]`` (``_legacy_capture``), which then answers its evidence.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_accepted import bind_started, locate
    from ouroboros.delegate_collection import record_terminal_money
    from ouroboros.review_execution import ReviewRouteUnavailable

    token = custody.idempotency_key("review_continue", entry.invocation_id, entry.run_id)
    record = custody.invocation_record(root, token)
    body = {"continueFrom": entry.run_id}
    if not record or record["state"] == "failed_definite":
        return None
    if (record.get("request") != body or record.get("task_id") != entry.task_id
            or record.get("surface") != invocation.surface or record.get("slot_id") != invocation.slot_id
            or record.get("operation_id") != invocation.operation_id):
        raise ReviewRouteUnavailable("Review continuation custody does not match this seat",
                                     code="review_recovery_request_mismatch")
    if record["state"] == "started":
        run_id, successor = owned_started_review_custody(custody, root, record, entry.task_id)
        if run_id in seen or successor.continuation_of != entry.run_id:
            return None
        return _legacy_capture(root, entry, invocation, successor, checking)
    located = locate(gateway, {**record, "invocation_id": token})
    if not located.bindable or located.run_id in seen:
        return None  # Reading accepted work needs no generation window or permission to start another run.
    run_id = located.run_id
    record_terminal_money(root, entry, detail)  # the stopped run's money; the final settlement takes the successor's
    successor = custody.RunCustody(run_id=run_id, invocation_id=token, idempotency_key=token,
        continuation_of=entry.run_id, ledger_root=str(root), daemon_store_id=located.store_id,
        engine_ref=entry.engine_ref, access=entry.access, mode=entry.mode, isolation=entry.isolation,
        delegated=entry.delegated, max_seconds=entry.max_seconds,
        **{key: getattr(entry, key) for key in ("task_id", "route_id", "model", "profile_id", "project_id",
            "project_owned", "project_persistent", "root_task_id", "parent_task_id", "category", "source",
            "review_skill", "review_wave_id", "review_slot_id", "authority_source", "execution_root", "target_root")})
    successor, landed, _adopted = bind_started(root, successor, exclusive=True, shape={
            "accepted_start_lookup": located.outcome,
            "access": entry.access, "mode": entry.mode, "isolation": entry.isolation,
            "delegated": entry.delegated, "max_seconds": entry.max_seconds,
            "surface": invocation.surface, "slot_id": invocation.slot_id})
    if successor is None or not landed:
        raise ReviewRouteUnavailable("Review successor start remains pending in custody",
                                     code="review_continuation_checkpoint_unwritable")
    if successor.run_id in seen or successor.continuation_of != entry.run_id:
        return None
    return _legacy_capture(root, entry, invocation, successor, checking)


def _legacy_capture(root: Any, entry: Any, invocation: Any, successor: Any, checking: Optional[Dict[str, Any]]) -> Any:
    """Advance the capture generation for a bound legacy successor, or name why it cannot.

    The successor is this seat's own bound run, so it is observed and settled whatever its copy's state. A
    missing or foreign copy (``CheckingAreaUnavailable``) never re-materializes or takes over a copy and never
    reuses the predecessor's capture receipt: the seat's evidence becomes this typed gap instead, and every
    earlier capture stays as it is. A later successor whose generation advances replaces the gap."""
    from ouroboros.review_checking import CheckingAreaUnavailable, prepare_legacy_successor_capture

    try:
        prepare_legacy_successor_capture(root, entry.task_id, invocation.operation_id, run_id=successor.run_id,
                                         execution_root=successor.execution_root, access=successor.access)
    except CheckingAreaUnavailable as exc:
        if checking is None:
            raise
        log.warning("legacy review successor %s is observed without a capture: %s", successor.run_id, exc)
        checking["legacy_capture_gap"] = {
            "status": "capture_failed", "capture_gap": "legacy_successor_checking_copy_unavailable",
            "run_id": successor.run_id, "operation_id": str(invocation.operation_id or ""),
            "area": str(successor.execution_root or ""), "error": f"CheckingAreaUnavailable: {exc}"}
        return successor
    if checking is not None:
        checking.pop("legacy_capture_gap", None)
    return successor


def rejoin_recorded_review(invocation: Any, data_root: Any, gateway: Any, record: Dict[str, Any],
                           run_id: str, started: Any) -> tuple[str, Any, Optional[float]]:
    """A recovered seat's first step on its recorded engine: ``(run_id, started custody, generation end)``.

    A pending start (its key: the seat's own token, ``retry_state["pending_invocation_id"]``) is first
    read for the work the engine already accepted (#3138, ``delegate_accepted.recover``); found, it is
    bound once and joined like a started one. The panel owns this read (the sweep leaves review starts
    to it), after ``review_recovery_facts`` proved the seat owns the record. It precedes the checking-copy restoration
    and the generation window (``prepare_continuation_send``, ``continuation_generation_end``), which
    bound only a new POST: accepted work is collected after its window closed or its released checkout
    went. A read that forbids any replay refuses typed with the start kept; any other miss leaves the
    send exactly as before, an explicit continuation's first send included."""
    if not run_id and record.get("state") == "pending":
        from ouroboros.delegate_accepted import NO_REPLAY, recover

        token = str((getattr(invocation, "retry_state", None) or {}).get("pending_invocation_id") or "")
        held, read = recover(data_root, gateway, {**record, "invocation_id": token})
        if held is not None:
            run_id, started = held.run_id, held
        elif read["reason"] in NO_REPLAY:
            from ouroboros.review_execution import ReviewRouteUnavailable

            raise ReviewRouteUnavailable(f"the engine answered for this review's recorded start ({read['lookup']}) but no "
                                         "run could be bound to it now; nothing was re-sent and its outcome stays unknown",
                                         code=read["reason"])
    prepare_continuation_send(invocation, data_root, record, run_id)  # every continuation POST, replays too
    return run_id, started, continuation_generation_end(invocation, record, run_id)


def continuation_send(invocation: Any) -> bool:
    """Is this send an explicit continuation's successor (``review_continuation``), whose start its decision recorded?"""
    from ouroboros.review_continuation import PREFIX

    return str(getattr(invocation, "operation_id", "") or "").startswith(PREFIX)


def prepare_continuation_send(invocation: Any, data_root: Any, record: Optional[Dict[str, Any]], run_id: str) -> None:
    """Before EVERY POST of an explicit continuation's recorded start (no run bound yet), the first send and a crash or
    restart replay alike: the released read-only checkout its START names is put back from the START row's exact
    mapping (``review_continuation.RESTORE_KEY``, ``review_subject.restore_released_subject``). A failure raises before
    any POST, so the start stays pending, decided and unsent, for a later collection to repair; a start with no
    mapping (a writable seat, a live root, an older row) is untouched."""
    from ouroboros.review_continuation import RESTORE_KEY

    mapping = (record or {}).get(RESTORE_KEY)
    if run_id or not continuation_send(invocation) or not isinstance(mapping, dict) or not mapping.get("root"):
        return
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.tools.review_subject import restore_released_subject

    try:
        restore_released_subject(data_root, mapping)
    except ValueError as exc:
        raise ReviewRouteUnavailable(f"the reviewed checkout this continuation reads is unavailable: {exc}; its "
                                     "recorded start was not POSTed", code="review_subject_unavailable") from exc


def continuation_generation_end(invocation: Any, record: Optional[Dict[str, Any]], run_id: str) -> Optional[float]:
    """When an explicit continuation's successor must stop generating, on this process's monotonic clock: the instant
    its START recorded (``generation_end_at``: the original window narrowed by the owner deadline and task window
    when the projection read it, ``review_continuation._start_requested``), never rebased on the START's own later
    timestamp, whatever cap its immutable bytes carry. The executor polls no longer than that and cancels; custody's
    settlement margin collects receipts only. Past it, a start with no run bound is never POSTed, because that POST
    could begin new work: its outcome stays unknown (``custody_lost``), its key and bytes kept. ``None`` for any
    other send, or a start recorded with no window. Read on custody's clock (``review_custody.utc_now``)."""
    import time

    from ouroboros import review_custody
    from ouroboros.deadline_utils import parse_deadline_ts

    closes = parse_deadline_ts(str((record or {}).get("generation_end_at") or ""))
    if not continuation_send(invocation) or closes is None:
        return None
    end = time.monotonic() + (closes - review_custody.utc_now()).total_seconds()
    if not run_id:
        _refuse_past(end)
    return end


def continuation_send_admitted(invocation: Any, end: Optional[float], request: Dict[str, Any], model: str) -> None:
    """Admit the POST of an explicit continuation's start (``continuation_send``) in the send callback itself, after
    its gateway, START, checkpoint and launch steps, since any POST of a start with no run bound may begin new paid
    work: known spend is below every budget fence, as every review wave is admitted (``review_wave_budget_gate``:
    the global, root and original-group fences, fail-open on an unknown), and its window (``end``,
    ``continuation_generation_end``) has not closed. A refusal keeps its key and bytes: its outcome stays UNKNOWN,
    never a definite absence. Every other send, and every known run's collection, passes untouched."""
    from types import SimpleNamespace

    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_substrate import review_usage_category
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    if not continuation_send(invocation):
        return
    admission = review_wave_budget_gate(
        SimpleNamespace(task_id=invocation.task_id), surface=invocation.surface, models=[model],
        prompt_chars=[len(str(request.get("prompt") or ""))], categories=[review_usage_category(invocation.surface)],
        slot_ids=[invocation.slot_id], extra={"continuation": True, "at_send": True})
    if admission is not None:
        fence, remedy = review_wave_binding_fence(admission)
        raise ReviewRouteUnavailable(f"known spend has reached the {fence}, so this continuation's recorded start was "
                                     "not POSTed: its outcome stays unknown, its key and bytes kept, and no collection "
                                     f"re-posts it. To review again, {remedy} first.", code="continuation_budget_insufficient")
    _refuse_past(end)


def _refuse_past(end: Optional[float]) -> None:
    """Refuse the POST of a continuation start at or after ``end``: UNKNOWN, its key and bytes kept."""
    import time

    from ouroboros.review_execution import ReviewRouteUnavailable

    if end is not None and end <= time.monotonic():
        raise ReviewRouteUnavailable("this continuation's original review window ended before its recorded start was "
                                     "sent; a POST now could begin new work, so its outcome stays unknown and it is "
                                     "never re-posted", code="continuation_window_ended")


def full_session_text(gateway: Any, run_id: str, detail: Dict[str, Any], *, checkpoint: Optional[tuple] = None) -> str:
    """The session's final answer from the verified FULL primary output (D7).

    The resolver fetches and verifies the full artifact when the engine reports truncation; an unresolvable full text
    refuses rather than judging a head-cut transcript. ``checkpoint`` (root, entry, binding; the binding None where
    the seat is bound to no operation) also keeps the complete textual primary output, masked, before any parsing;
    the engine's own summary, which stands in as the text for a run with no primary output, never is kept."""
    from ouroboros.delegate_output import _resolve_full_primary_output

    primary, full_ok, disclosure = _resolve_full_primary_output(gateway, run_id, detail.get("primaryOutput"))
    if not full_ok:
        raise RuntimeError(
            f"delegated review session {run_id} produced a truncated primary "
            f"output whose full artifact could not be matched to the size or the "
            f"preview the run reported ({(disclosure or {}).get('reason')}); a "
            "verdict is never read from a preview"
        )
    text = str(primary.get("text") or "") if isinstance(primary, dict) else "" if primary is None else str(primary)
    if checkpoint and checkpoint[2] and (answer := complete_session_primary_text(primary)):
        _checkpoint_session_answer(*checkpoint, detail, answer, delivered_answer(primary, answer))
    if not text.strip():
        final_summary = detail.get("finalSummary")
        text = final_summary if isinstance(final_summary, str) else ""
    return text


def complete_session_primary_text(primary: Any) -> str:
    """A complete textual primary output, not a preview or engine summary."""
    if isinstance(primary, dict):
        primary = primary.get("text") if primary.get("truncated", False) is False else None
    return primary if isinstance(primary, str) and primary.strip() else ""


def answer_identity(text: str) -> Dict[str, Any]:
    """The identity of a text, as the standard readers measure it: its UTF-8 bytes' sha256 and its characters. The
    delivery binds it before any stored copy is masked; a retained copy names its own."""
    import hashlib

    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "chars": len(text)}


def delivered_answer(primary: Any, text: str) -> Dict[str, Any]:
    """The identity of the text the engine delivered and the check its own report of the output passed, under the check's
    existing name: ``size`` or ``preview_prefix`` for a fetched full artifact
    (``delegate_output._resolve_full_primary_output``), ``untruncated`` for the engine's whole inline text. The engine
    serves text through ``redactSecrets`` and publishes no hash of the file, so a matching length, a shared prefix or an
    untruncated flag compare the text with the engine's description of the output, never with the source's bytes: the
    identity binds this delivery only (``answer_words``). A truncated preview is no delivered answer."""
    fetched = isinstance(primary, dict) and primary.get("full_fetched") is True
    if isinstance(primary, dict) and primary.get("truncated") is True and not fetched:
        raise ValueError("a truncated preview is not a delivered answer")
    checked_by = str(primary.get("verified_by") or "") if fetched else "untruncated"
    return {**answer_identity(text), "checked_by": checked_by or "unrecorded"}


def _text_masks(rules: Any) -> list:
    """The masking rules a stored projection recorded on the answer's text."""
    return sorted({str(rule.get("rule") or "") for rule in rules or []
                   if isinstance(rule, dict) and rule.get("path") == "$.text"})


def answer_words(delivered: Any, shown: str, masked: list) -> Dict[str, Any]:
    """The one reading of what a retained answer's words are, from its checkpoint or from an older run's output alike.
    What the engine delivered is a ``redacted_copy`` (the retained role ``redacted_run_answer``): the source's bytes
    are never confirmed, so ``words_identity`` stays ``unknown`` and no ``sha256``/``chars`` is offered as the run's own.
    ``copy_sha256``/``copy_chars`` name exactly the bytes shown and retained, after the host's own masking (``masked``:
    the stored projection's rules, empty when it changed nothing), which the retained ref and ``read_source`` repeat.
    ``checked_by`` keeps the delivery's check. A delivery bound to no recorded check, or shown words it does not explain
    (changed, no recorded masking), prove nothing."""
    delivered = delivered if isinstance(delivered, dict) else {}
    bound = {"sha256": delivered.get("sha256"), "chars": delivered.get("chars")}
    if not isinstance(bound["sha256"], str) or not bound["sha256"] or not delivered.get("checked_by"):
        raise ValueError("the answer's delivered words are not bound to a recorded check")
    copy = answer_identity(shown)
    unchanged = copy == bound
    if not (unchanged or masked):
        raise ValueError("the shown words differ from the run's delivered answer and no recorded masking explains it")
    return {"provenance": "redacted_copy", "masked": [] if unchanged else list(masked),
            "checked_by": delivered["checked_by"], "words_identity": "unknown",
            "copy_sha256": copy["sha256"], "copy_chars": copy["chars"]}


_PREVIEW_ONLY = ("only a truncated preview of this run's answer is retained: its full text is unavailable, "
                 "so none of its words are shown")


def output_answer_words(primary: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """A run's retained primary output read as its answer without a checkpoint (an older producer's run):
    ``(text, words)`` with the text masked as ``persist_call(keep_raw=False)`` stores it, before anything shows or keeps
    it, and ``answer_words`` over its delivery. A truncated preview is ``("", unavailable)``: the gap is named and
    nothing of the preview is kept. No textual output is None."""
    from ouroboros.redaction import redact_projection

    raw = complete_session_primary_text(primary)
    if not raw:
        truncated = isinstance(primary, dict) and primary.get("truncated") is True
        return ("", {"provenance": "unavailable", "reason": _PREVIEW_ONLY}) if truncated else None
    masked = redact_projection({"text": raw})
    text = masked.value["text"]
    return text, answer_words(delivered_answer(primary, raw), text, _text_masks(masked.manifest()["rules"]))


def _checkpoint_session_answer(root: Any, entry: Any, binding: Optional[dict], detail: dict, text: str,
                               delivered: Dict[str, Any]) -> None:
    """Persist this run's full answer, not a completed review producer outcome. The stored payload is masked (no raw
    copy), so the manifest binds the delivery's identity and check first (``read_session_answer_checkpoint``)."""
    from ouroboros.delegate_custody_usage import run_names, summary_of
    from ouroboros.observability import persist_call

    summary = summary_of(detail)
    if (not binding or not binding.get("operation_id") or not entry.invocation_id
            or entry.task_id != binding.get("task_id") or entry.run_id not in run_names(detail)
            or summary.get("state") != "succeeded"):
        return
    try:
        if read_session_answer_checkpoint(root, binding) is not None:
            return  # A rejoin cannot replace the original words with a later observation.
        persist_call(root, task_id=entry.task_id, call_id=f"{binding['operation_id']}_{_SESSION_ANSWER}",
                     call_type=_SESSION_ANSWER, keep_raw=False,
                     manifest={"review_operation_binding": binding, "verified_answer": delivered},
                     payload={"run_id": entry.run_id, "invocation_id": entry.invocation_id, "text": text,
                              "conformance_passed": str(summary.get("outputConformance") or "").strip().lower() == "passed"})
    except Exception:
        log.warning("Original review session answer could not be retained: %s", entry.run_id, exc_info=True)


def read_session_answer_checkpoint(root: Any, binding: dict) -> Optional[dict]:
    """Read the original-answer CAS, independently of parser settlement, with what its words are (``words``:
    ``answer_words`` over the delivery the manifest bound before masking and the masks it recorded). Missing is None;
    a foreign binding, an unbound delivery, or a change no masking records raise."""
    from ouroboros.observability import read_call_payload

    try:
        manifest, answer, _ = read_call_payload(root, task_id=binding["task_id"],
                                              call_id=f"{binding['operation_id']}_{_SESSION_ANSWER}")
    except FileNotFoundError:
        return None
    if (manifest.get("call_type") != _SESSION_ANSWER or manifest.get("review_operation_binding") != binding
            or not isinstance(answer, dict) or not all(isinstance(answer.get(k), str) and answer[k].strip()
                                                       for k in ("text", "run_id", "invocation_id"))):
        raise ValueError("original session answer binding or payload mismatch")
    masked = _text_masks((manifest.get("redaction") or {}).get("rules"))
    return {**answer, "words": answer_words(manifest.get("verified_answer"), answer["text"], masked)}


def stopped_review_failure(gateway: Any, run_id: str, detail: Dict[str, Any], summary: Dict[str, Any]) -> Any:
    """The typed failure of a run that did not succeed, with its stop facts and whole partial output.

    ``failure_details`` keeps the engine's own failure plus ``resumable`` and ``outcomeFacts`` (what an
    explicit continuation is offered from); ``review_partial_output`` is the run's complete primary
    output when it is readable: retained evidence, never parsed into a verdict."""
    from ouroboros.gateways.claudexor import run_failure_error
    from ouroboros.observability import redact_projection

    run_state = str(summary.get("state") or "")
    failure = run_failure_error(run_id, run_state, summary.get("failure"))
    failure.review_run_state = run_state
    stop = {key: summary.get(key) or detail.get(key) for key in ("resumable", "outcomeFacts")
            if isinstance(summary.get(key) or detail.get(key), dict)}
    if stop:
        failure.failure_details = {**dict(failure.failure_details or {}), **redact_projection(stop).value}
    try:
        from ouroboros.review_execution import _full_session_text

        failure.review_partial_output = _full_session_text(gateway, run_id, detail)
    except Exception as exc:  # an unreadable output is a named gap beside the typed failure
        failure.review_partial_output_gap = f"{type(exc).__name__}: {exc}"
    return failure


def failed_start_seat_evidence(custody_drive: Any, task_id: str, operation_id: str, exc: BaseException,
                               checking: Dict[str, Any], invocation_id: str, *, unrequested: bool,
                               posted: bool) -> Dict[str, Any]:
    """A writable seat's evidence once its start failed, never raising: the attempt's own typed failure stays its
    outcome. An observed terminal is captured as ever. The copy this attempt made (``checking['allocation']``) is
    captured, then released unless kept, only on the CURRENT attempt's proof that its start never ran: no
    START_REQUESTED write (``unrequested``), or ``_start_never_ran``. Every other failure keeps the copy: an entered
    POST whatever the engine answered (a 4xx too), which may hold an accepted command (``accepted_after_refusal``)."""
    from ouroboros.review_checking import terminal_seat_evidence

    if checking.get("legacy_capture_gap"):  # a bound legacy successor without its copy: never the old receipt
        return dict(checking["legacy_capture_gap"])
    try:
        proven = unrequested or _start_never_ran(custody_drive, task_id, operation_id, checking, invocation_id,
                                                 posted=posted)
        return terminal_seat_evidence(custody_drive, task_id, operation_id, getattr(exc, "checking_terminal_detail", None),
                                      unsent=checking.get("allocation") if proven else None)
    except Exception as error:  # e.g. an unreadable install identity at the release: the copy stays for the prune
        log.warning("checking evidence after the failed start of %s failed", operation_id, exc_info=True)
        return {"status": "capture_failed", "operation_id": operation_id, "error": f"{type(error).__name__}: {error}"}


def _start_never_ran(custody_drive: Any, task_id: str, operation_id: str, checking: Dict[str, Any],
                     invocation_id: str, *, posted: bool) -> bool:
    """This attempt's own recorded start provably never ran: its POST was never entered (``posted`` false: a launch
    fence or any refusal after the START_REQUESTED write attempt), and ``invocation_id`` now reads back
    ``failed_definite``, with no run, for this task, operation, copy and reviewed root. An entered POST proves
    nothing whatever the engine answered: a typed body may be synthesized from a non-JSON one, and an accepted
    command can end before execution and answer its start with a 4xx. A pending fate (the one a replay may still
    send), a started one (a late STARTED keeps precedence), a legacy, absent or unreadable one prove nothing too."""
    from ouroboros import delegate_custody as custody

    if not invocation_id or posted:
        return False
    try:
        record = custody.invocation_record(custody_drive, invocation_id) or {}
    except Exception:
        log.debug("review start %s: its custody fate is unreadable; the copy stays", invocation_id, exc_info=True)
        return False
    request, bound = record.get("request") or {}, checking.get("custody") or {}
    return (record.get("state") == "failed_definite" and not record.get("run_id")
            and (record.get("task_id"), record.get("operation_id")) == (task_id, operation_id)
            and bool(bound.get("execution_root"))
            and (request.get("execution") or {}).get("workspaceRoot") == bound["execution_root"]
            and (request.get("scope") or {}).get("root") == bound.get("target_root"))
