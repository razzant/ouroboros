"""A saved task's explicit decision about one ordinary start with a lost reply.

The request stays in open custody. Only the selected task's explicit Resume reads the decision
as permission. Every create handoff for the invocation shares its starter's short decision lock.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from urllib.parse import quote

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros.owner_pause import OwnerPauseRefused

LOOKUP_ONLY = "delegated_start_lookup_only"
BUSY, UNREADABLE = "delegated_start_custody_busy", "delegated_start_custody_unreadable"
ACTIVE = frozenset({"queued", "running"})
_RESUME = ContextVar("delegated_start_resume", default=None)


def address(request):
    member = str(request.get("task_id") or "")
    tree = str(request.get("root_task_id") or member)
    invocation = str(request.get("invocation_id") or "")
    if (not member or not invocation or request.get("surface") not in (None, "", "delegate")
            or (request.get("source") or "delegated_subagent") != "delegated_subagent"
            or (request.get("category") or "subagent") != "subagent"
            or any(request.get(k) for k in ("operation_id", "slot_id", *REVIEW_ATTRIBUTION_KEYS))):
        return {}
    target = "/".join(quote(part, safe="") for part in (tree, member, invocation))
    return {"root_task_id": tree, "task_id": member, "invocation_id": invocation,
            "choice_id": "delegate_start:" + target}


def _selection(selected):
    from ouroboros.control_actor import resume_author

    scope = _RESUME.get()
    return scope if scope and scope["selected"] == selected and scope["author"] == resume_author() else None


def decisions_in_use(selected):
    scope = _selection(selected)
    return list(scope["decisions"].values()) if scope else []


def with_resume_decisions(selected, resume, *, child=False):
    """Run the addressed Resume once. Its grant owner judges the complete, authorized refusal."""
    from ouroboros.control_actor import accepted_unknown, accepting_unknown, resume_author

    author = resume_author()
    if not (author.get("door") == "api_task_resume" and author.get("actor") == "owner"
            or author.get("actor") == "ouroboros" and author.get("task_id")):
        return resume()
    choice = accepted_unknown()
    scope = {"selected": selected, "author": author, "choice": choice, "decisions": {}}
    token = _RESUME.set(scope)
    try:
        # A child's predecessor is never the subject of its selection. A named start
        # likewise cannot answer the separate Continue/held-predecessor decision.
        with accepting_unknown(()) if child or names_start() else nullcontext():
            result = resume()
        risks = decisions_in_use(selected)
        return {**result, "accepted_start_risks": risks} if risks else result
    finally:
        _RESUME.reset(token)


def deciding_resume():
    return _RESUME.get() is not None


def names_start():
    scope = _RESUME.get()
    return bool(scope and any(key.startswith("delegate_start:") for key in scope["choice"].get("ids", ())))


def decide(root, tree, selected, blockers, *, fence_id, pause_id="", tree_actor=False,
           allow_starts=True, wait_id="", error="owner_pause_effects_unsettled", action="wait_for_effect_settlement"):
    """Take choices only after ordinary grant authority, against one set of current blockers."""
    from ouroboros.work_stop import acceptable_unknown

    scope = _selection(selected)
    held, offers = [], {}
    wanted = set(scope["choice"].get("ids", ())) if scope else set()
    basis = {"fence_id": fence_id, "pause_id": pause_id}
    if wait_id:
        basis["wait_id"] = wait_id
    for row in blockers:
        start = row.get("ordinary_start") or {}
        member = row.get("task_id")
        if not tree_actor and member != selected:
            # This writer can veto a release, but cannot be accepted by the selected child.
            return {"ok": False, "error": error, "blockers": blockers, "action": action,
                    "selection_scope_mismatch": True}
        if (allow_starts and scope and start and start.get("root_task_id") == tree and start.get("task_id") == member
                and start.get("invocation_id") == row.get("invocation_id") and not row.get("run_id")
                and row.get("stop_outcome") == "pending_invocation_unbound"):
            saved = (selected, fence_id, pause_id) + ((wait_id,) if wait_id else ())
            choice = "/".join([start["choice_id"], *(quote(v, safe="") for v in saved)])
            decision = (row.get("resume_decisions") or {}).get(selected) or {}
            if (decision.get("choice_id") == choice and decision.get("creation") == "lookup_only"
                    and decision.get("resuming_task_id") == selected and all(decision.get(k) == v for k, v in basis.items())):
                scope["decisions"][choice] = decision
                continue
            offers[choice] = start
            row = {**row, "choice_id": choice}
        held.append(row)
    # A failed grant can leave the choice and its notice committed. Confirm it
    # against current custody, but record only choices that are still owed.
    if scope:
        wanted -= scope["decisions"].keys()
    offered = set(acceptable_unknown(held))
    if wanted and not wanted <= offered:
        if not held or offered:
            return {"ok": False, "error": "selection_authority_changed"}
    elif wanted and (not wanted.intersection(offers) or scope["choice"].get("text")):
        accepted = set()
        for key in sorted(wanted.intersection(offers)):
            decision = _record(root, offers[key], selected, scope["author"], scope["choice"]["text"],
                               choice_id=key, **basis)
            if decision:
                scope["decisions"][key] = decision
                accepted.add(key)
        local = wanted - offers.keys()
        if local and accepted == wanted.intersection(offers):
            # Bind each id to the holder that actually offered it in this refusal.
            for member in {row.get("task_id") for row in held}:
                ids = local.intersection(acceptable_unknown([r for r in held if r.get("task_id") == member]))
                if ids:
                    from ouroboros.store import acquisition_budget
                    from supervisor import queue
                    from supervisor.continuation_admission import accept_local_choice
                    with acquisition_budget(0):
                        accepted.update(accept_local_choice(queue, root, tree, {member}, ids, scope["choice"]))
        held = [r for r in held if not accepted.intersection(acceptable_unknown([r]))]
    if held:
        return {"ok": False, "error": error, "blockers": held, "action": action}
    return None


def _record(root, start, selected, author, reason, *, choice_id=None, fence_id="", pause_id="", wait_id=""):
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from ouroboros.delegate_invocation import _read

    iid, member = start["invocation_id"], start["task_id"]
    try:
        with custody.actor_decision_lock(root, member, wait=False), store.acquisition_budget(0):
            with store.open_store(root, write=False) as txn:
                record = _read(txn, iid) or {}
            if record.get("state") != "pending" or address(record.get("request") or {}) != start:
                return None
            key = choice_id or start["choice_id"]
            prior = (record.get("resume_decisions") or {}).get(selected) or {}
            if prior.get("choice_id") == key:
                return prior
            decision = {**start, "choice_id": key, "fence_id": fence_id, "pause_id": pause_id,
                        **({"wait_id": wait_id} if wait_id else {}),
                        "resuming_task_id": selected, "by": author, "reason": reason, "outcome": "unknown",
                        "external_effects": "unknown", "spend": "unknown", "creation": "lookup_only"}
            if custody.emit(root, custody.START_RISK_ACCEPTED, {**start, "resuming_task_id": selected, "decision": decision}):
                return decision
    except (custody.ActorDecisionLockUnavailable, store.StoreError):
        return None
    return None


class StartHeld(OwnerPauseRefused):
    """This handoff lost custody. Its earlier request and any remote work stay intact."""


@contextmanager
def _admit_start(root, task_id, iid):
    """Hold the actor's decision lock across the handoff, after re-reading this request.

    It is taken before the root's launch admission, the order a START_REQUESTED claim
    already uses, so a sibling's claim delays this start and never refuses it. Contention
    past the lock's bound or a busy store is ``custody_busy``; any other failure is
    ``custody_unreadable``. Both leave the request pending and send nothing.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from ouroboros.delegate_invocation import _read

    with ExitStack() as admission:
        try:
            admission.enter_context(custody.actor_decision_lock(root, task_id))
            with store.open_store(root, write=False) as txn:
                record = _read(txn, iid) or {}
        except (custody.ActorDecisionLockUnavailable, store.StoreBusy) as exc:
            busy = getattr(exc, "reason", "") == "contention" or isinstance(exc, store.StoreBusy)
            raise StartHeld(BUSY if busy else UNREADABLE) from exc
        except Exception as exc:
            raise StartHeld(UNREADABLE) from exc
        if record.get("resume_decisions"):
            raise StartHeld(LOOKUP_ONLY)
        if record.get("state") != "pending" or (record.get("request") or {}).get("task_id") != task_id:
            raise StartHeld("delegated_start_custody_changed")
        yield


def checked_start(root, task_id, iid, submit):
    from ouroboros.owner_pause import CheckedSubmission

    return CheckedSubmission(submit, partial(_admit_start, root, task_id, iid))


def close_unsent_start(root, member, iid, reason):
    """A fresh sender stopped before handoff must close its claim before retiring its resources."""
    from ouroboros import delegate_custody as custody

    try:
        with _admit_start(root, member, iid):
            return custody.emit(root, custody.START_UNSENT, {"invocation_id": iid, "task_id": member,
                "reason": reason, "definite": True})
    except StartHeld:
        return False


def risk_words(decision):
    by = decision["by"]
    who = "The owner" if by["actor"] == "owner" else f"Ouroboros (task {by['task_id']})"
    return (f"{who} accepted the unknown outcome of start {decision['invocation_id']} from task "
            f"{decision['task_id']} in tree {decision['root_task_id']} for Resume of {decision['resuming_task_id']}. "
            f"The work may still be running, changing files or spending money. Retries can only look up this start. "
            f"Reason: {decision['reason']}")


def _update_task(txn, task_id, change):
    from ouroboros.obligations import pending_drive_rule
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema

    current = txn.load(task_id)
    if not current:
        raise ValueError("saved task unavailable")
    require_writable_task_result_schema(current)
    fields = change(current)
    if fields:
        updated = stamp_task_result_schema({**current, **fields})
        txn.store_task(updated, current=current, pending_drive=pending_drive_rule(txn, current, updated))


def owe_risk_notice(txn, event):
    """The choice and its addressed local timeline notice commit together."""
    def notice(current):
        identity = f"pause:{event['resuming_task_id']}:{event['decision']['choice_id']}"
        owed = dict(current.get("pause_notices") or {})
        owed[identity] = {"reason": "delegated_start_risk", "confirmed_at": event["ts"], "decision": event["decision"]}
        return {"pause_notices": owed}

    _update_task(txn, event["resuming_task_id"], notice)


class AnswerHeld(Exception):
    """A late writer won before any answer was owed, sent or stored."""

    def park(self, ctx):
        from ouroboros.budget_pause import RAIL_CONTROL_AUTHORITY, enter_owner_pause, request_pause
        from ouroboros.model_wait import ModelWaitInterrupted
        from ouroboros.loop_round_limits import _handle_model_wait_control

        limit, candidate, control_required = ctx._selected_answer_context
        ctx._delivery_candidate, ctx._delivery_control_required = candidate, control_required
        try:
            if self.args and self.args[0] != "owner_pause":
                raise ModelWaitInterrupted(self.args[0])
            # The publication boundary parks COLD even where a warm park is available: a resumed
            # stack could only re-emit the retained text, while the owner's words accepted under
            # this pause must reach the model. The ordinary Resume re-enters the loop (resume_point
            # phase "candidate"), where the owner-input check holds the saved candidate for a
            # revision round when new words arrived, and delivers it unchanged when none did.
            enter_owner_pause(limit, delivery_candidate=True, allow_warm=False)
        except ModelWaitInterrupted as control:
            if control.control_reason == "owner_pause_authority_unreadable":
                request_pause(limit, rail=RAIL_CONTROL_AUTHORITY, scope="task",
                              reason_text=control.control_reason, delivery_candidate=True)
                raise
            return _handle_model_wait_control(limit, control)
        finally:
            ctx._delivery_candidate, ctx._delivery_control_required = None, False


def finish_selected_answer(ctx, emit, result, events, worker_stopped):
    if not getattr(ctx, "_selected_answer_context", None):
        emit(*result)
        return list(events)
    from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait

    try:
        while True:
            try:
                emit(*result)
                return list(events)
            except AnswerHeld as held:
                forced = held.park(ctx)  # a cold hold unwinds to the worker's usual pause event
                if forced:
                    result, ctx._selected_answer_context = forced, None
    except ModelWaitInterrupted as control:
        if worker_stopped(control, current_model_wait()):
            return []
        raise
    finally:
        ctx._selected_answer_context = None


def selected_answer_context(limit):
    """Keep the prepared answer until its first handoff. A sibling's first choice can still arrive.

    Returns ``(context, waited)``. ``waited`` says this stack stood under an owner Pause before the
    fence opened (``enter_owner_pause``): its caller then re-enters its owner-input check, so my
    human's words accepted meanwhile hold the answer for a round instead of leaving with it.
    """
    from ouroboros.budget_pause import enter_owner_pause
    from ouroboros.owner_pause import _member_coordinates

    waited = enter_owner_pause(limit, delivery_candidate=True)
    ctx = getattr(getattr(limit, "tools", None), "_ctx", None)
    if all(_member_coordinates(ctx)):
        return (limit, getattr(ctx, "_delivery_candidate", None), getattr(ctx, "_delivery_control_required", False)), waited
    return None, waited


def answer_publication(ctx):
    """Order the first actual publication with the same tree lock as the active-start observation."""
    if not getattr(ctx, "_selected_answer_context", None):
        return None
    from ouroboros._usage_wait import retry_presend
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.owner_pause import LaunchBusy, _member_coordinates, launch_lock, member_fence
    from ouroboros.store import StoreBusy, acquisition_budget

    root, tree, _ = _member_coordinates(ctx)
    committed = False

    def publish(write, *args, wait=False, **kwargs):
        if committed:
            return write(*args, **kwargs)

        def commit():
            nonlocal committed
            with launch_lock(root, tree, wait=False), acquisition_budget(0):
                if member_fence(ctx):
                    raise AnswerHeld()
                value = write(*args, **kwargs)  # only the ready outbox rename or the local result transaction
                committed = True
            ctx._selected_answer_context = None
            return value

        if not wait:
            return commit()  # the outbox writer already holds its own lock; a refusal uses the result path
        try:
            return retry_presend(commit, check=False)  # waits after all locks unwind, only on proven contention
        except PhysicalDispatchInterrupted as control:
            raise AnswerHeld(control.control_reason) from control
        except (LaunchBusy, StoreBusy) as exc:
            if getattr(exc, "rollback_confirmed", None) is False:
                raise
            raise AnswerHeld("owner_pause_authority_unreadable") from exc
    return publish


def still_pending(txn, event):
    """A late definite failure or binding cannot acquire a decision in the write transaction."""
    from ouroboros.delegate_invocation import _read
    from ouroboros.owner_pause import owner_stopped

    decision = event["decision"]
    tree, selected = event["root_task_id"], event["resuming_task_id"]
    tree_row = txn.load(tree) or {}
    member = tree_row if selected == tree else (txn.load(selected) or {})
    if not member or (member.get("root_task_id") and member["root_task_id"] != tree):
        return False
    if owner_stopped(tree_row) or owner_stopped(member) or member.get("_owner_hold"):
        return False
    if str((tree_row.get("owner_pause") or {}).get("fence_id") or "") != decision["fence_id"]:
        return False
    pause = member.get("budget_pause") or {}
    if decision["pause_id"] and (pause.get("pause_id") != decision["pause_id"]
                                 or pause.get("state") not in {"pausing", "paused"}):
        return False
    if decision.get("wait_id"):
        wait = member.get("owner_wait") or {}
        if (wait.get("wait_id") != decision["wait_id"] or wait.get("reason") != "owner_pause"
                or wait.get("state") != "waiting"):
            return False
    record = _read(txn, event["invocation_id"]) or {}
    return (record.get("state") == "pending" and address(record.get("request") or {}) == address(event)
            and (record.get("request") or {}).get("task_id") == event["task_id"])


@contextmanager
def publication_lock(root, event):
    """An accepted writer's first proof of activity is ordered against every tree handoff."""
    from ouroboros.owner_pause import launch_lock

    if event.get("resume_start_state") in ACTIVE:
        from ouroboros.store import acquisition_budget
        with launch_lock(root, event.get("root_task_id") or event["task_id"]), acquisition_budget(0):
            yield
    else:
        yield


def hold_active_start(txn, event):
    """Called in the live custody commit only. Imports rebuild observations and never issue controls."""
    from ouroboros.delegate_invocation import INVOCATIONS
    from ouroboros.owner_pause import _resume_outstanding, fence_closed, next_fence, owner_stopped

    record = txn.get_row(INVOCATIONS, event["invocation_id"]) or {}
    start = address(record.get("request") or {})
    if not record.get("resume_decisions") or not start:
        return
    tree = start["root_task_id"]
    if tree != (event.get("root_task_id") or event["task_id"]) or start["task_id"] != event["task_id"]:
        raise ValueError("recovered start belongs to a different task tree")
    run = txn.custody_row(event["run_id"]) or {}
    if run.get("task_id") != start["task_id"] or run.get("invocation_id") != start["invocation_id"]:
        raise ValueError("observed run does not hold the accepted invocation")
    if run.get("settled") or run.get("terminal_state") or run.get("resume_start_state") not in ACTIVE:
        return
    def hold(current):
        if owner_stopped(current):
            return None
        actor = current
        if current.get("superseded_by") or current.get("retry_task_id"):
            from ouroboros.cancel_intents import _validated_single_cancel_target

            actor = txn.load(_validated_single_cancel_target(txn.root, tree, txn=txn)) or {}
            if owner_stopped(actor):
                return None
        old = current.get("owner_pause") or {}
        request_id = "active:" + start["choice_id"]
        if request_id in (old.get("requests") or {}):
            return None
        activity = {**start, "run_id": event["run_id"], "state": run["resume_start_state"]}
        fence = next_fence(tree, old, request_id, {"active_delegated_start": activity})
        if not fence_closed(old) or old.get("observe_only") or _resume_outstanding(actor, old):
            fence["observe_only"] = True
        if fence_closed(old):
            fence.update({key: old[key] for key in ("by", "ended_member_runs", "membership_gaps") if key in old})
        return {"owner_pause": fence}

    _update_task(txn, tree, hold)
    return tree


def recovered_writer_changed(root, tree):
    """Wake the addressed tree's settlement pass if this process owns its queue."""
    from supervisor import queue
    from supervisor.owner_pause_control import mark_tree_changed

    if not queue.owns_live_queue(root):
        return False
    mark_tree_changed(tree, recovered_drive=queue.DRIVE_ROOT)
    return True


def restore_recovered_writer_hints(root):
    """Rebuild the supervisor's settlement hints from their authority, not a retired run's receipt."""
    from ouroboros.store_hot_facts import recovered_owner_pause_roots

    for tree in recovered_owner_pause_roots(root):
        recovered_writer_changed(root, tree)


def retain_start_receipt(txn, entry, task_id, task_row):
    """A worker keeps its closed fence address until the live supervisor can look; boot reads the fence itself."""
    from ouroboros.owner_pause import fence_closed

    if entry.resume_start_state not in ACTIVE:
        return False
    tree = entry.root_task_id or entry.task_id
    if recovered_writer_changed(txn.root, tree):
        return False
    tree_row = task_row if tree == task_id else txn.load(tree)
    fence = (tree_row or {}).get("owner_pause") or {}
    return fence_closed(fence) and bool(fence.get("active_delegated_start"))


def observe_bound_state(root, entry, state):
    """Use the first actual status after binding. False means the write failed and remains owed."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run
    from ouroboros.store import StoreError

    if (getattr(entry, "resume_start_state", None) == "unknown" and not entry.terminal_state
            and state in ACTIVE | custody.TERMINAL_STATES):
        if not custody.emit(root, custody.START_STATE_OBSERVED, {
            "root_task_id": entry.root_task_id or entry.task_id, "task_id": entry.task_id,
            "invocation_id": entry.invocation_id, "run_id": entry.run_id, "resume_start_state": state}):
            return False
        # An observer in another process may have published first. Adopt the committed state.
        try:
            observed = run(root, entry.run_id)
        except StoreError:
            return False  # unreadable observation does not retract the commit
        if observed is None or observed.invocation_id != entry.invocation_id or observed.task_id != entry.task_id:
            return False
        entry.resume_start_state = observed.resume_start_state
        memo = custody._CUSTODY.get(entry.run_id)
        if memo and memo.invocation_id == entry.invocation_id and memo.task_id == entry.task_id:
            memo.resume_start_state = observed.resume_start_state
        return True
    return None


def recovered_cancel_store(gateway, entry):
    from ouroboros.execution_domain import same_engine

    return (same_engine(gateway, entry) and bool(entry.daemon_store_id)
            and entry.daemon_store_id == getattr(gateway, "store_identity", ""))


def recovered_cancel_detail(gateway, entry, detail):
    """A recovered run's terminal and active facts must name its run in its store."""
    from ouroboros.delegate_custody_usage import run_names

    return (recovered_cancel_store(gateway, entry) and entry.run_id in run_names(detail)
            and detail.get("storeId") in (None, "", entry.daemon_store_id))


def fold_start_control(entry, row):
    """Reduce one row of a recovered start onto its run (``delegate_custody._apply``, replay and memo alike).

    A lookup-bound start's first engine state replaces ``unknown`` once, for its own task and
    invocation, never after a terminal. An ordinary run's cancel outcome is not folded. A handoff
    without its outcome keeps its identity and its sender (``sender``: the image that reserved it)
    as an unresolved fault; the outcome that follows drops the sender, who has then returned.
    """
    from ouroboros import delegate_custody as c

    if row["type"] == c.START_STATE_OBSERVED:
        if (entry.resume_start_state == "unknown" and not entry.terminal_state
                and entry.invocation_id == row.get("invocation_id") and entry.task_id == row.get("task_id")):
            entry.resume_start_state = row["resume_start_state"]
        return
    action = getattr(entry, "_cancel_action", {})
    if row["type"] == c.CANCEL_HANDOFF:
        entry._recovered_cancel = c.CANCEL_CONTAINMENT_FAULT
        entry._cancel_action = {"control_id": row.get("stop_control_id") or "", "unknown_at": row.get("ts"),
                                "unknown_event": row.get("event_id"), "sender": row.get("sender") or {}}
        return
    if entry.resume_start_state is None:
        return
    outcome = row.get("outcome")
    entry._recovered_cancel = ("not_issued" if row.get("control_owed") else c.CANCEL_REQUESTED
                               if row.get("accepted") and outcome != c.CANCEL_CONFIRMED else outcome or "unknown")
    entry._cancel_action = {"control_id": action.get("control_id", "")}
    if entry._recovered_cancel == c.CANCEL_CONTAINMENT_FAULT:
        entry._cancel_action.update(unknown_at=row.get("ts"), unknown_event=row.get("event_id"))


def _prior_sender_returned(action):
    """Whether the earlier handoff's sender can no longer deliver it: it recorded its outcome (no
    ``sender`` left), or its image is positively gone (``local_custody_repair.owner_ended``: an absent
    pid, a different kernel birth, or this pid under an earlier custody session). A sender without a
    recorded identity, or one that may still run, keeps every later press read-only."""
    sender = action.get("sender")
    if sender is None:
        return True
    from ouroboros.local_custody_repair import owner_ended

    return owner_ended(sender.get("pid"), "", sender)


def can_reserve_cancel(entry, event):
    """One sender per handoff. A new one is reserved when no control stands (none issued, or a
    recorded failure), or once after an unresolved fault: by a distinct owner Stop requested after
    that fault and naming it (``after_unknown_event``, which only ``prepare_recovered_cancel`` sets)."""
    from ouroboros import delegate_custody as c

    if entry._recovered_cancel in ("", "not_issued", c.CANCEL_FAILED):
        return True
    old = getattr(entry, "_cancel_action", {})
    control_id = event.get("stop_control_id")
    return (entry._recovered_cancel == c.CANCEL_CONTAINMENT_FAULT and bool(control_id)
            and control_id != old.get("control_id") and bool(old.get("unknown_event"))
            and event.get("after_unknown_event") == old["unknown_event"]
            and bool(old.get("unknown_at")) and (event.get("stop_requested_at") or "") > old["unknown_at"])


def prepare_recovered_cancel(root, gateway, entry, reason, owner_stop):
    """Control follows the exact run/store, not the success of publishing its first state.

    GET can avoid an unnecessary POST or corroborate identity. An unreadable GET alone
    cannot veto an authorized Stop. The existing handoff admits one sender; after a crash
    or ambiguous return, the same duty only reads back unless non-delivery was proved.
    A distinct later owner Stop names that fault, and so may reserve once more, only when
    this read shows the exact run active in its store and the earlier sender has returned
    (``_prior_sender_returned``). The same press and automatic reconciliation never do.
    """
    from ouroboros.tool_custody import owner_proof
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def blocked(reason, explanation, *, owed=False):
        return {"outcome": custody.CANCEL_CONTAINMENT_FAULT, "accepted": False, "control_status": "",
                "state": "", "fault_reason": reason, "detail": explanation, "control_owed": owed}

    fresh = run(root, entry.run_id)
    if fresh is None and custody._CUSTODY.get(entry.run_id) is entry and recovered_cancel_store(gateway, entry):
        # The starter kept the exact physical handle when STARTED failed. Rejoin its invocation,
        # not a new start, before reserving Stop; another engine name is not a match here.
        bound = custody.bind_started(root, entry, exclusive=True, accepted_only=True)
        if bound.durable and bound.custody.run_id == entry.run_id:
            fresh = bound.custody
    if not fresh or (fresh.task_id, fresh.invocation_id, fresh.daemon_store_id) != (
            entry.task_id, entry.invocation_id, entry.daemon_store_id):
        return blocked("custody_owner_mismatch", "The run's custody identity is unavailable or changed.", owed=True)
    prior = fresh._recovered_cancel
    accepted = prior == custody.CANCEL_REQUESTED
    owed = prior in ("", "not_issued", custody.CANCEL_FAILED)

    def result(outcome, state="", explanation="", terminal=None):
        if outcome == custody.CANCEL_CONFIRMED:
            return custody._cancel_result(root, fresh, outcome, accepted=accepted,
                                          control_status="", state=state, terminal_detail=terminal)
        fault = "cancel_unverified" if outcome == custody.CANCEL_CONTAINMENT_FAULT else ""
        if fault and not fresh.containment_fault:
            custody.record_containment_fault(root, fresh, fault, explanation)
        return {"outcome": outcome, "accepted": accepted, "control_status": "", "state": state,
                "detail": explanation, "fault_reason": fault, "control_owed": owed}

    if fresh.settled:
        return result(custody.CANCEL_CONFIRMED, "settled")
    if not recovered_cancel_store(gateway, fresh):
        return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The run's owning store was not verified.")
    read_error = ""
    try:
        detail = gateway.get_run(entry.run_id)
    except ClaudexorUnavailable as exc:
        if custody.daemon_says_absent(exc, gateway=gateway, run=fresh):
            custody.close_absent_run(root, gateway, fresh, "get_run_absent", absence=exc)
            return result(custody.CANCEL_CONFIRMED, "absent")
        if exc.status_code == 404:
            return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The 404 did not prove owning-store absence.")
        read_error, detail = str(exc), {}
    if (custody.names_another_run(detail, entry.run_id)
            or custody.admitted_store_id(detail) not in ("", fresh.daemon_store_id)):
        return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The read-back names another run or store.")
    state = str(custody.summary_of(detail).get("state") or "") if recovered_cancel_detail(gateway, fresh, detail) else ""
    if state in custody.TERMINAL_STATES:
        custody.settle_run(root, gateway, fresh, detail)
        return result(custody.CANCEL_CONFIRMED, state, terminal=detail)
    if accepted:
        return result(prior, state, "The earlier cancel was accepted. Termination is not verified.")
    if state in ACTIVE:
        observe_bound_state(root, fresh, state)  # owed publication is not a veto over the exact Stop
    action = owner_stop or {}
    event = {"task_id": fresh.task_id, "run_id": fresh.run_id, "invocation_id": fresh.invocation_id,
             "daemon_store_id": fresh.daemon_store_id, "reason": reason,
             "stop_control_id": action.get("control_id", ""), "stop_requested_at": action.get("requested_at", ""),
             "sender": owner_proof()}
    earlier = getattr(fresh, "_cancel_action", {})
    if state in ACTIVE and _prior_sender_returned(earlier):  # never on a penumbral GET or beside a live sender
        event["after_unknown_event"] = earlier.get("unknown_event")
    if (recovered_cancel_store(gateway, fresh) and can_reserve_cancel(fresh, event)
            and custody.emit(root, custody.CANCEL_HANDOFF, event)):
        if recovered_cancel_store(gateway, fresh):
            return None
        return custody._cancel_result(root, fresh, custody.CANCEL_CONTAINMENT_FAULT, accepted=False,
                                      control_status="", state=state, fault_reason="cancel_unverified",
                                      detail="Engine identity changed before control delivery.", control_owed=True)
    return result(custody.CANCEL_CONTAINMENT_FAULT, state, read_error or "The cancellation handoff is unconfirmed.")
