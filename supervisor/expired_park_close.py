"""Exact Close of an expired park: durable cancel intent, without the live Stop rails.

``observe`` is the one eligibility reading: /api/state publishes its offer and Close accepts only that offer.
Only a positively identified parked queue row may enter pending-file capture; it stays queued (snapshot-visible)
until its cancelled write commits, and the commit removes it under Q. Root requests freeze their parked membership
in the token, then use the same capture for each exact descendant, deepest first. Replay never adds another member:
a member or reservation that appears later refuses the root's write (partial), and a replay after the cancelled write
finishes that write's runs and settlement. A direct parent's request is durable before its acknowledgement.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import NamedTuple

from ouroboros.cancel_intents import (
    PARKED_CLOSE_SOURCE as SOURCE, active_intent, claim_still_owned,
    conditional_park_close as conditional, request_cancel, withdraw_intent,
)
from ouroboros.owner_pause import launch_lock
from ouroboros.parked_deadline import close_withheld, deadline_offer, park_limit, parked_binding, valid_token
from ouroboros.task_results import load_task_result
from ouroboros.task_status import SETTLED_STATUSES
from ouroboros.work_stop import parked_stop_view, queue_control
from supervisor.cancel_publication import CANCEL_ALREADY_SETTLED, CANCEL_CANCELLED, CANCEL_FAILED, CANCEL_NOT_FOUND

log = logging.getLogger(__name__)
CHANGED = "expired_park_changed"
PARTIAL = "expired_park_partly_closed"
NOTICE = "_expired_park_notice"  # the reminder's own mark on the row; no part of the park
_reminders_in_flight: set[tuple[str, str]] = set()
WITHHELD_RECHECK_SEC = 120.0  # a withheld park's reminder re-reads it at most this often (its row's NOTICE)


def observe_due_child(q, row):
    """Existing timeout scan, under Q; submit only the observed row, never work on its behalf."""
    from functools import partial
    from supervisor.workspace_preparation import ADMISSION_LANE, current_generation

    parent, task_id = str(row.get("parent_task_id") or ""), str(row.get("id") or "")
    point, hold = row.get("_budget_pause") or {}, row.get("_budget_pause_hold") or {}
    tree = q.BUDGET_ROOT_FENCES.get(str(row.get("root_task_id") or task_id)) or {}
    marker = [row.get("_attempt"), point.get("paused_at"), hold.get("held_at"), tree.get("fence_id")]
    notice = row.get(NOTICE)
    if not (parent and task_id) or notice == marker or (isinstance(notice, dict) and notice.get("quiet") == marker
                                                        and 0 <= notice.get("until", 0) - time.time() <= WITHHELD_RECHECK_SEC):
        return
    generation = current_generation()
    key = (str(q.DRIVE_ROOT), task_id)
    if generation is None or generation.launches_closed() or key in _reminders_in_flight:
        return
    _reminders_in_flight.add(key)
    if not generation.submit(partial(_remind, q, task_id, row, parent, key, marker), ADMISSION_LANE):
        _reminders_in_flight.discard(key)


def _parent_actor(q, parent):
    """The parent's live queue row; else a live inline Presence turn's own record, only as the exact mailbox
    ``forward_to_worker`` admits (``peer_roster.independent_message_target``: running, source-bound, observed as
    Presence), never any authority over its work; or None: a settled, absent or drained parent hears nothing."""
    from ouroboros.peer_roster import independent_message_target
    from ouroboros.task_status import load_effective_task_result

    with q._queue_lock:
        actor = (q.RUNNING.get(parent) or {}).get("task") or next((p for p in q.PENDING if p.get("id") == parent), None)
    if actor is not None:
        return actor
    data = load_effective_task_result(q.DRIVE_ROOT, parent) or {}
    if (independent_message_target(Path(q.DRIVE_ROOT), parent, data) or {}).get("target_kind") != "inline_presence":
        return None
    drive = data.get("child_drive_root") or data.get("headless_child_drive_root") or data.get("drive_root")
    return {**(load_task_result(q.DRIVE_ROOT, parent, strict=True) or {}), "id": parent,
            **({"child_drive_root": drive} if drive else {})}


def _remind(q, task_id, original, parent, key, marker):
    from ouroboros.owner_mailbox import mailbox_drain_ended, write_task_message

    settled, quiet = False, False  # done (written, or nobody can receive it); or withheld, re-read later
    try:
        seen = observe(q, task_id)
        offer, binding = seen.get("offer") or {}, seen.get("binding") or {}
        if not offer:  # no positive park, not due, or work Close would not take: a later pass observes it again
            quiet = True
            return
        stored = load_task_result(q.DRIVE_ROOT, parent, strict=True) or {}
        actor = _parent_actor(q, parent)
        with q._queue_lock:
            retained = any(item is original for item in q.PENDING)
        if not retained:
            return
        if (binding.get("parent_task_id") != parent or not actor or not stored
                or stored.get("status") in SETTLED_STATUSES
                or str(actor.get("root_task_id") or parent) != binding.get("root_task_id")):
            settled = True  # proven: this park has no live direct parent to tell
            return
        drive = q._task_drive_for_task(actor, parent)
        if mailbox_drain_ended(drive, parent):
            settled = True
            return
        token = offer["close_expired"]
        words = (f"The deadline expired while your direct child {task_id} was paused. It cannot Resume. "
                 f"If you still need its result, you can start a new separate task instead. "
                 f"To close this saved park and keep its materials, call cancel_task(task_id={task_id!r}, "
                 f"close_expired={token!r}). If this exact request is refused, do not retry it as ordinary Stop.")
        settled = bool(write_task_message(drive, words, parent, source_task_id=task_id, provenance="system",
                                          msg_id=f"expired-park:{token}", work_attention={
                                              "kind": "deadline_expired", "task_id": task_id,
                                              "waiting_for_task_id": task_id, "episode_id": token}))
    except Exception:
        log.debug("The expired park reminder of %s was not confirmed", task_id, exc_info=True)
    finally:
        with q._queue_lock:
            if (settled or quiet) and any(item is original for item in q.PENDING):  # a gone row forgets it
                original[NOTICE] = marker if settled else {"quiet": marker, "until": time.time() + WITHHELD_RECHECK_SEC}
            _reminders_in_flight.discard(key)


def _live_rows(q):
    with q._queue_lock:
        return ([(dict(r), False) for r in q.PENDING if isinstance(r, dict)]
                + [({**(meta.get("task") or {}), "id": tid}, True) for tid, meta in q.RUNNING.items()])


def same_episode(stored: dict, row: dict) -> bool:
    """The queue row is the stored attempt itself, or its restart/Panic successor: recovery re-queues attempt N as
    N+1 held for Resume and names its source (``restart_retention._recovery_source``: ``source_task_id`` and
    ``from_attempt``) while the stored record keeps N. Without that named lineage an adjacent attempt is not this row."""
    ran, queued = stored.get("task_attempt"), row.get("_attempt", 1)
    if ran is None or ran == queued:
        return True
    recovery = row.get("_working_recovery") if isinstance(row.get("_working_recovery"), dict) else {}
    return (type(ran) is int and type(queued) is int and queued == ran + 1
            and recovery.get("from_attempt") == ran and recovery.get("source_task_id") == row.get("id"))


def _parent_of(q):
    def parent_of(task_id):
        return str((load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}).get("parent_task_id") or "")
    return parent_of


def _reaper_expires(row: dict, *, nested=False) -> bool:
    """The #939 reaper's own decision (``queue_timeouts.pending_deadline_expirable``): a row it expires without dispatch
    once due is no park Close takes. A root's frozen member counts only when its own deadline is already due (without
    one, or before it, only the root's limit binds it). Unknown withholds the offer; the reaper decides alone."""
    from ouroboros.deadline_utils import task_deadline

    if nested and ((own := task_deadline(row)) is None or own.timestamp() > time.time()):
        return False
    return _reaper_would_expire(row)


def _reaper_would_expire(row: dict) -> bool:
    """The reaper's predicate alone, without its clock: whether this row is its to expire once its own limit is due."""
    from supervisor.queue_timeouts import pending_deadline_expirable

    try:
        return pending_deadline_expirable(row, strict=True)
    except Exception:
        log.debug("The reaper's reading of %s failed; its Close is withheld", row.get("id"), exc_info=True)
        return True


class _QueueView(NamedTuple):
    """One copy of the live queue under Q, indexed once: rows by physical id and by tree, and the root fences."""
    by_id: dict
    by_tree: dict
    fences: dict


def _queue_view(q, memo=None) -> _QueueView:
    """A Close reads afresh; a publication's ``memo`` keeps one indexed copy for all its parks."""
    view = memo.get(None) if isinstance(memo, dict) else None
    if isinstance(view, _QueueView):
        return view
    with q._queue_lock:
        live, fences = _live_rows(q), dict(q.BUDGET_ROOT_FENCES)
    by_id: dict = {}
    by_tree: dict = {}
    for item in live:
        row = item[0]
        by_id.setdefault(str(row.get("id") or ""), []).append(item)
        by_tree.setdefault(str(row.get("root_task_id") or row.get("id") or ""), []).append(item)
    view = _QueueView(by_id, by_tree, fences)
    if isinstance(memo, dict):
        memo[None] = view
    return view


def _observation(q, task_id, *, late=False, nested=False, owned=None, memo=None):
    """Read one current row. A missing park is not authority to use the miss lane.

    A helper with live descendants of its own is no Close target (Close never alters them); only a root's
    frozen membership (``nested``) observes members whose own descendants it closes first. Live ownership (a review
    operation, a registered actor) is no park; only ``observe`` passes ``owned`` to still read the park's own hard
    limit for display, recording the owner there instead of refusing. ``memo``: one publication's queue copy."""
    from supervisor.queue_transitions import _task_ownership_reason
    from supervisor.worker_assignment import _direct_actor_still_registered
    from supervisor.queue_snapshot import saved_queue_unowned
    from supervisor.restart_retention import panic_recorded

    if saved_queue_unowned() or panic_recorded(q.DRIVE_ROOT):
        return {}, {}, {}
    view = _queue_view(q, memo)
    fences, rows = view.fences, view.by_id.get(task_id, [])
    if len(rows) > 1 or any(running for _r, running in rows):
        return {}, {}, {}
    row = dict(rows[0][0]) if rows else {}
    stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
    if not stored or stored.get("superseded_by") or str(stored.get("retry_task_id") or task_id) != task_id:
        return {}, {}, {}
    if _task_ownership_reason(task_id) or _direct_actor_still_registered(task_id):
        if owned is None:
            return {}, {}, {}
        owned.append(task_id)
    if not row:
        if not late or stored.get("status") not in SETTLED_STATUSES or stored.get("parent_task_id"):
            return {}, {}, {}
        row = {**stored, "id": task_id}
        view = parked_stop_view(q.DRIVE_ROOT, row, task_id, fences.get(task_id) or {})
        return row, parked_binding(q.DRIVE_ROOT, row, view, late_phase=True), stored
    row.pop(NOTICE, None)
    if (stored.get("status") in SETTLED_STATUSES or row.get("parent_task_id") != stored.get("parent_task_id")
            or not same_episode(stored, row)
            or str(stored.get("root_task_id") or task_id) != str(row.get("root_task_id") or task_id)):
        return {}, {}, {}
    control = queue_control(q.DRIVE_ROOT, row, fences)
    if control["phase"] != "budget_paused" or control["owner_resumed"]:
        return {}, {}, {}
    binding = parked_binding(q.DRIVE_ROOT, row, control["view"])
    if not binding:
        return {}, {}, {}
    if not nested:
        tree = view.by_tree.get(binding["root_task_id"], [])  # a helper's descendants live in its own tree
        if binding["parent_task_id"] and close_withheld(binding, tree, parent_of=_parent_of(q)):
            return {}, {}, {}
    if _reaper_expires(row, nested=nested):  # its own due deadline is the reaper's: a root's Close waits for it
        return {}, {}, {}
    return row, binding, stored


def _membership(q, binding, memo=None):
    """Current live tree members, keyed by their immutable physical ids."""
    target = binding["task_id"]
    if binding["parent_task_id"]:
        return []
    root = binding["root_task_id"]
    members = {r["id"]: r for r, _running in _queue_view(q, memo).by_tree.get(root, []) if r.get("id")
               and r["id"] != target and (r.get("root_task_id") == root or r.get("id") == root)}
    # Queue metadata are the same lineage the ordinary cascade uses. The chain may
    # contain ended intermediate parents; the recorded depth, if present, supplies order.
    def depth(task_id):
        row, seen = members[task_id], {task_id}
        while (parent := row.get("parent_task_id")) in members and parent not in seen:
            seen.add(parent)
            row = members[parent]
        value = members[task_id].get("depth")
        return max(len(seen), value) if type(value) is int else len(seen)
    return sorted(members, key=lambda member: (-depth(member), member))


def _tails_hold(q, binding, members, closures=None) -> bool:
    """Delegated work Close would not take (``close_withheld``): a pending start, or a run that may be live or is
    owned outside the target and its frozen members. One store read per tree (``closures`` memoizes it for one
    publication); an unreadable store raises (custody pending)."""
    from ouroboros.delegate_custody_current import tree_closure

    closures = {} if closures is None else closures
    tree = binding["root_task_id"]
    if tree not in closures:
        closures[tree] = tree_closure(Path(q.DRIVE_ROOT), tree)
    owners = set() if binding["late_phase"] else {binding["task_id"], *members}
    return close_withheld(binding, [], closures[tree], owners=owners)


def _preparing_child(task_id, token):
    """The live child preparation holding this reservation (``workspace_preparation._LIVE``: token -> claim, set before
    its reservation and kept until its job ends), as its lineage; ``None`` when no claim holds it. Read without a lock
    (``dict()`` copies whole); an unreadable map raises (withheld)."""
    from supervisor import workspace_preparation

    claim = dict(workspace_preparation._LIVE).get(str(token or ""))
    if claim is None or str(getattr(claim, "task_id", "")) != task_id:
        return None
    route = getattr(claim, "route", None) or {}
    return {"root_task_id": str(route.get("root_task_id") or ""), "parent_task_id": str(route.get("parent_id") or "")}


def _reservation_under(load, reserved, binding) -> bool:
    """An admission reserved under Q that Close would orphan: any of a root's tree, or, for a helper, one not proven a
    sibling. Only a durable parent chain through records of this tree up to its root proves a sibling; a missing or
    foreign record, a cycle or a chain past 64 links withholds. ``reserved`` is the map id -> token. A reservation
    without a record of its own is a child only while a live preparation claim holds its token
    (``task_admission.claim_requested_child``), read by that claim's lineage; with none it is a fresh id that
    ``reserve_task_admission`` admitted only because no record of it existed: a new root (API, continuation, a chat
    routed to a project), so no tree's member. ``load`` reads strictly; an unreadable record raises (withheld)."""
    tree, target = binding["root_task_id"], binding["task_id"]
    for tid, token in (reserved.items() if isinstance(reserved, dict) else ((tid, "") for tid in reserved)):
        node, seen = load(tid), {tid}
        if not node:
            node = _preparing_child(tid, token)
            if node is None:
                continue
            if not node["root_task_id"]:
                return True
        if str(node.get("root_task_id") or "") != tree:
            continue
        if not binding["parent_task_id"]:
            return True
        while (parent := str(node.get("parent_task_id") or "")) != tree:
            if parent == target or parent in seen or len(seen) >= 64:
                return True
            if not parent:  # a root attempt of this tree (its physical retry, say): the chain reached the top
                break
            seen.add(parent)
            node = load(parent)
            if not node or str(node.get("root_task_id") or "") != tree:
                return True
    return False


def _reserved(q, binding) -> bool:
    with q._queue_lock:
        reserved = dict(q.ADMISSION_RESERVATIONS)
    return _reservation_under(lambda tid: load_task_result(q.DRIVE_ROOT, tid, strict=True), reserved, binding)


def _late_settled(q, binding, row) -> bool:
    """An answered root's late phase may close only when ``owner_pause_control._late_work_settled`` says nothing of its
    own is left (synthesis, review); otherwise the expiry shows and Close waits, before any member is taken."""
    from supervisor.owner_pause_control import _late_work_settled

    return not binding["late_phase"] or _late_work_settled(Path(row.get("budget_drive_root") or q.DRIVE_ROOT),
                                                           binding["task_id"])


def observe(q, task_id, *, late=False, now=None, closures=None) -> dict:
    """The one eligibility reading the census publishes and Close accepts, on the same rows and checks.

    ``{}``: no positive park. Otherwise ``row``, ``binding`` and ``limit``; past the limit, ``offer`` is the exact
    Close (its token binds the park, limit and frozen member bindings) or ``withheld`` names work Close would not
    take (a live or unbound member, an unsettled delegated run, a reservation, open late work); ``unread`` adds that
    the tree was not read (a transient failure, no known blocker). Ordinary Stop stays. ``closures``: one
    publication's memo (custody per tree, one queue copy); without one each call reads afresh, once for all members."""
    closures = {} if closures is None else closures
    owned: list = []
    row, binding, stored = _observation(q, task_id, late=late, owned=owned, memo=closures)
    if not binding:
        return {}
    limit = park_limit(row, binding)
    seen = {"row": row, "binding": binding, "stored": stored, "limit": limit}
    now = time.time() if now is None else now
    if limit is None or now < limit:
        return {**seen, **({"withheld": True} if owned else {})}
    if owned:  # a positively read owner (a late review, a live actor) holds it: the expiry shows, Close waits
        return {**seen, "withheld": True}
    members = []
    for member in _membership(q, binding, closures):
        item = _observation(q, member, nested=True, memo=closures)[1]
        if not item or item["root_task_id"] != binding["root_task_id"]:
            return {**seen, "withheld": True}
        members.append(item)
    try:
        if (_tails_hold(q, binding, [item["task_id"] for item in members], closures)
                or _reserved(q, binding) or not _late_settled(q, binding, row)):
            return {**seen, "withheld": True}
    except Exception:
        log.debug("The tree under %s was not read; its Close is withheld", task_id, exc_info=True)
        return {**seen, "withheld": True, "unread": True}
    return {**seen, "members": members, "offer": deadline_offer(row, binding, now, members)[1]}


def _stands(q, task_id, binding, limit, *, nested=False) -> bool:
    """The same bound park, still observing the same hard limit, now past (``None``: a member, bound by its group's)."""
    row, current, _ = _observation(q, task_id, late=binding["late_phase"], nested=nested)
    return current == binding and _due(row, current, limit) and _late_settled(q, binding, row)


def _due(row, binding, limit) -> bool:
    """The bound limit is unchanged and the clock is still past it: a clock moved back keeps Resume, not a Close."""
    return limit is None or park_limit(row, binding) == limit and time.time() >= limit


def _settled_close(stored, token, group=""):
    """The durable record of a Close this exact offer (or group) already finished."""
    closed = stored.get("pause_at_close") or {}
    return stored.get("status") in SETTLED_STATUSES and (
        (group and closed.get("deadline_close") == group) or (token and closed.get("close_token") == token))


def _kept_late(stored, token, group=""):
    """An answered root's late-work fence that this exact Close released: the release wrote its request and offer onto
    the fence in that same CAS (``_close_late_root``). An ordinary Resume or Pause release names no Close."""
    fence = stored.get("owner_pause") or {}
    return stored.get("status") in SETTLED_STATUSES and (
        (group and fence.get("deadline_close") == group) or (token and fence.get("close_token") == token))


def _read(seen: dict) -> dict:
    """An observation whose tree was not read proves no change: the caller's outcome stays unconfirmed (retry)."""
    if seen.get("unread"):
        raise RuntimeError("the tree under the expired park was not read")
    return seen


def request_close(q, task_id, token, *, parent_id="") -> dict:
    """Record the validated physical choice: exactly the offer ``observe`` publishes now, or its own open request."""
    if not valid_token(token):
        return {"ok": False, "error": "invalid_close_expired"}
    # An exact retry (or the owner pressing what the parent already asked) joins its own still-open request.
    open_close = active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True) or {}
    if conditional(open_close) and (open_close.get("parked_close") or {}).get("token") == token:
        return {"ok": True, "request_id": open_close["request_id"]}
    seen = _read(observe(q, task_id, late=not parent_id))
    if (seen.get("offer") or {}).get("close_expired") != token:
        retained = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
        if not open_close and _settled_close(retained, token):  # only a finished Close reports its result
            return {"ok": True, "settled": CANCEL_CANCELLED,
                    "request_id": str((retained.get("pause_at_close") or {}).get("deadline_close") or "")}
        if not open_close and not parent_id and _kept_late(retained, token):
            return {"ok": True, "settled": CANCEL_ALREADY_SETTLED,
                    "request_id": str((retained.get("owner_pause") or {}).get("deadline_close") or "")}
        return {"ok": False, "error": CHANGED}
    binding = seen["binding"]
    if parent_id and parent_id != binding["parent_task_id"]:
        return {"ok": False, "error": CHANGED}
    with launch_lock(Path(seen["row"].get("budget_drive_root") or q.DRIVE_ROOT), binding["root_task_id"], wait=False):
        again = _read(observe(q, task_id, late=binding["late_phase"]))
        if (again.get("offer") or {}).get("close_expired") != token:
            return {"ok": False, "error": CHANGED}
        intent = request_cancel(q.DRIVE_ROOT, task_id, reason="the deadline expired", source=SOURCE,
                                requested_by=parent_id, allow_settled_target=binding["late_phase"],
                                parked_close={"binding": binding, "members": again["members"], "token": token,
                                              "limit": again["limit"]})
    if intent.get("close_conflict") or intent.get("already_settled"):
        return {"ok": False, "error": CHANGED}
    return {"ok": True, "request_id": intent["request_id"]}


def _withdraw(q, task_id, claim, outcome=CHANGED):
    """Retire only this conditional request. A parent hears its outcome first, durably: an unconfirmed message keeps
    the request for the watchdog (the message id makes the retry idempotent). An ordinary Stop that replaced the Close
    owns the intent now: its claim returns at once, without waiting for any Close mail (its own record is the source)."""
    from supervisor.task_lifecycle import _release_intent_claim

    current = active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True) or {}
    if not conditional(current) or current.get("request_id") != claim.get("request_id"):
        _release_intent_claim(q, task_id, error="Another Stop owns custody", intent=claim)
        return CANCEL_FAILED
    spec = claim.get("parked_close") or {}
    if not (spec.get("binding") or {}).get("late_phase"):  # an answered root's record is always settled: no reading
        stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
        if stored.get("status") in SETTLED_STATUSES and not _settled_close(stored, "", spec.get("group_request")
                                                                             or claim.get("request_id")):
            return _kept_alone(q, task_id, claim, stored)  # it ended by itself meanwhile: that, not "changed"
    if not _owed(q, task_id, claim, _NOT_CLOSED):
        return CANCEL_FAILED
    joined = withdraw_intent(q.DRIVE_ROOT, task_id, withdrawable=conditional, detail=outcome,
                             expected_generation=claim.get("generation"), request_id=claim.get("request_id", ""))
    if joined is not None:  # The distinct, explicit Stop keeps its own custody.
        _release_intent_claim(q, task_id, error="Another Stop owns custody", intent=claim)
        return CANCEL_FAILED
    return outcome


def _requester(claim) -> str:
    """The direct parent whose own request this claim drives (never a member of a root's group)."""
    spec = claim.get("parked_close") or {}
    return "" if spec.get("group_request") else str(spec.get("requested_by") or "")


def _drop_residual(q, task_id, stored) -> list:
    with q._queue_lock:
        residual = [r for r in q.PENDING if r.get("id") == task_id and same_episode(stored, r)]
        for row in residual:
            q.PENDING.remove(row)  # the durable verdict stands: a restored copy of that attempt never runs
    return residual


def _owed(q, task_id, claim, words) -> bool:
    """Tell the requesting parent before the request retires; an unconfirmed message keeps it (its claim returns
    for the watchdog's retry, which reuses the same message id)."""
    from supervisor.task_lifecycle import _release_intent_claim

    parent = _requester(claim)
    if not parent or _tell_parent(q, parent, task_id, claim["request_id"], words):
        return True
    _release_intent_claim(q, task_id, error="Close outcome not yet delivered", intent=claim)
    return False


def _kept_alone(q, task_id, claim, stored):
    """The parked attempt ended by itself first: its own result stands and this Close changed nothing."""
    from supervisor.task_lifecycle import SETTLED_ALREADY, _settle_intent

    residual = _drop_residual(q, task_id, stored)
    if residual:
        q.persist_queue_snapshot(reason="cancel_pending")
    if not _owed(q, task_id, claim, _KEPT):
        return CANCEL_FAILED
    _settle_intent(q, task_id, outcome=SETTLED_ALREADY, detail=str(stored.get("status") or ""), intent=claim)
    return CANCEL_ALREADY_SETTLED


def _finish_replay(q, task_id, claim, stored):
    """The cancelled write committed before an interruption: finish what follows it, once, from that record."""
    from supervisor.ended_work_helpers import stop_closed_runs, stored_helper_lineage
    from supervisor.task_lifecycle import _settle_intent

    residual = _drop_residual(q, task_id, stored)
    stop_closed_runs(q, task_id)
    if not _owed(q, task_id, claim, _CLOSED):
        if residual:
            q.persist_queue_snapshot(reason="cancel_pending")
        return CANCEL_FAILED
    _settle_intent(q, task_id, outcome=CANCEL_CANCELLED, detail="expired deadline closed", intent=claim)
    q._emit_cancel_task_done(residual[0] if residual else {**stored_helper_lineage(stored), "id": task_id}, task_id)
    if residual:
        q.persist_queue_snapshot(reason="cancel_pending")
    return CANCEL_CANCELLED


def _tree_admitted(txn, q, binding) -> bool:
    """Under Q: any live member of the root's tree, or an admission reserved for it, after its frozen members closed."""
    root, target = binding["root_task_id"], binding["task_id"]
    if any(r.get("id") != target and r.get("root_task_id") == root for r, _ in _live_rows(q)):
        return True
    return _reservation_under(txn.load, dict(q.ADMISSION_RESERVATIONS), binding)


def _close_one(q, binding, claim, group, *, token="", limit=None):
    """Capture and cancel the one queued row of this exact park. The row stays in PENDING (and in every snapshot)
    until the cancelled write commits; ``commit_guard`` removes it under Q in that same step."""
    from supervisor.task_lifecycle import _finish_captured_pending

    task_id = binding["task_id"]
    spec = claim.get("parked_close") or {}
    group_root = spec.get("group_root")
    stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
    if _settled_close(stored, "", group):
        return _finish_replay(q, task_id, claim, stored)
    if binding["late_phase"] and _kept_late(stored, "", group):  # this request's release committed before a crash
        from supervisor.task_lifecycle import _settle_intent

        _settle_intent(q, task_id, outcome=CANCEL_ALREADY_SETTLED, detail="kept the existing answer", intent=claim)
        return CANCEL_ALREADY_SETTLED
    if _requester(claim) and stored.get("status") in SETTLED_STATUSES:
        return _kept_alone(q, task_id, claim, stored)
    row, current, observed = _observation(q, task_id, late=binding["late_phase"], nested=bool(group_root))
    if current != binding or not _due(row, current, limit):
        return _withdraw(q, task_id, claim)

    def root_request(active=None):
        """The root's own Close request still open: no ordinary Stop replaced it (``_close_replaced``)."""
        active = (active_intent(q.DRIVE_ROOT, group_root, strict=True, include_close=True) or {}) if active is None else active
        return conditional(active) and active.get("request_id") == group

    def parent_park_stands():
        if not group_root:
            return True
        active = active_intent(q.DRIVE_ROOT, group_root, strict=True, include_close=True) or {}
        expected = (active.get("parked_close") or {}).get("binding") or {}
        return root_request(active) and bool(expected) and _stands(q, group_root, expected, spec.get("group_limit"))

    if not parent_park_stands():
        return _withdraw(q, task_id, claim)
    if binding["late_phase"]:
        return _close_late_root(q, binding, claim, row, limit, group, token)

    # A root's physical retry carries its logical root's fence and launch lock (#1730), never its own.
    logical = binding["root_task_id"]
    root_record = load_task_result(q.DRIVE_ROOT, logical, strict=True) or {}
    original_fence = root_record.get("owner_pause") or {}
    original_budget_fence = dict(q.BUDGET_ROOT_FENCES.get(logical) or {})
    is_root = not binding["parent_task_id"]
    bound = limit if limit is not None else spec.get("group_limit")
    with q._queue_lock:
        original = next((item for item in q.PENDING if item.get("id") == task_id), None)
    if original is None or {k: v for k, v in original.items() if k != NOTICE} != row:
        return _withdraw(q, task_id, claim)

    def kept():
        """Under Q: the captured row is still the only queued copy, unchanged, and no actor took it."""
        from supervisor import workers

        queued = [item for item in q.PENDING if item.get("id") == task_id]
        return (len(queued) == 1 and queued[0] is original and {k: v for k, v in original.items() if k != NOTICE} == row
                and task_id not in q.RUNNING and not any(w.busy_task_id == task_id for w in workers.WORKERS.values()))

    reaper: dict = {}

    def verify():
        with q._queue_lock:
            if not kept():
                return False
        # The reaper's own predicate, read here (never under Q): the final write refuses once its clock is due.
        reaper["expires"] = _reaper_would_expire(row)
        return (_stands(q, task_id, binding, limit, nested=bool(group_root))
                and conditional(active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True))
                and parent_park_stands()
                and not _tails_hold(q, binding, [m["task_id"] for m in spec.get("members") or []]))

    refused: dict = {}

    def commit_guard(write, *args, wait=False, **kwargs):
        from contextlib import nullcontext
        from ouroboros.cancel_intents import cancellation_projection_lock
        from ouroboros.store import acquisition_budget

        # write_task_result calls this after preparing unread mail. Only its final transaction holds a
        # root's launch lock (admission's order, §10.25: before Q, never waited for), Q and the cancel
        # projection, without waiting on either store. The committed cancel leaves PENDING in the same hold.
        launch = launch_lock(Path(row.get("budget_drive_root") or q.DRIVE_ROOT), logical, wait=False) if is_root else nullcontext()
        with launch, q._queue_lock, cancellation_projection_lock(q.DRIVE_ROOT, wait=False), acquisition_budget(0):
            document = write(*args, **kwargs)
            if not refused and _settled_close(document or {}, "", group):
                q.PENDING[:] = [item for item in q.PENDING if item is not original]
            return document

    def last_word(txn, before, after):
        from ouroboros.deadline_utils import task_deadline

        point = before.get("budget_pause") or {}
        base_point = observed.get("budget_pause") or {}
        root_record = before if logical == task_id else txn.load(logical) or {}
        if (before.get("task_attempt") != observed.get("task_attempt") or before.get("status") in SETTLED_STATUSES
                or before.get("superseded_by") or before.get("retry_task_id") != observed.get("retry_task_id")
                or str(before.get("parent_task_id") or "") != binding["parent_task_id"]
                or task_deadline(before) != task_deadline(observed)
                or point != base_point or (root_record.get("owner_pause") or {}) != original_fence
                or (q.BUDGET_ROOT_FENCES.get(binding["root_task_id"]) or {}) != original_budget_fence
                or not kept() or bound is not None and time.time() < bound
                or reaper.get("expires", True) and (own := task_deadline(row)) is not None
                and time.time() >= own.timestamp()  # #939: past its own limit it is the reaper's, even mid-capture
                or (is_root and _tree_admitted(txn, q, binding))
                or (not is_root and (close_withheld(binding, _live_rows(q), parent_of=lambda tid: str(
                    (txn.load(tid) or {}).get("parent_task_id") or ""))
                    or _reservation_under(txn.load, dict(q.ADMISSION_RESERVATIONS), binding)))
                or group_root and not root_request()
                or not conditional(active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True))
                or not claim_still_owned(q.DRIVE_ROOT, task_id, claim)):
            refused["refused"] = True
            return None
        return after

    alone: dict = {}

    def announce(outcome) -> bool:
        """Before the request retires: its parent hears this request's own outcome (``_owed``). Unconfirmed after the
        cancelled write: the snapshot drops the written row now, and the replay delivers, settles and publishes."""
        alone["seen"] = outcome != CANCEL_CANCELLED
        told = _owed(q, task_id, claim, _CLOSED if outcome == CANCEL_CANCELLED else _KEPT)
        if not told and outcome == CANCEL_CANCELLED:
            q.persist_queue_snapshot(reason="cancel_pending")
        return told

    outcome = _finish_captured_pending(task_id, original, intent=claim, exact_park={
        "verify": verify, "last_word": last_word, "commit_guard": commit_guard, "refused": refused,
        "group": group, "token": token,  # the offer this target was closed by; a member keeps only the group
        "withdraw": lambda: _withdraw(q, task_id, claim), "announce": announce,
    })
    if outcome == CANCEL_ALREADY_SETTLED or alone:  # it ended by itself first: its own result stands, the copy is a ghost
        with q._queue_lock:
            q.PENDING[:] = [item for item in q.PENDING if item is not original]
    return outcome


def _close_late_root(q, binding, claim, row, limit, group, token):
    """Retire the exact late-work fence only when nothing behind it can start; the answer is kept. The release names
    this request and offer on the fence in its own CAS (``_kept_late``): a replay or retry reads that, never a guess."""
    from ouroboros.cancel_intents import cancellation_projection_lock
    from ouroboros.owner_pause import FENCE_RELEASED, read_fence
    from ouroboros.store import acquisition_budget
    from supervisor.owner_pause_control import _release_settled_late_work
    from supervisor.task_lifecycle import _settle_intent

    root, task_id = Path(row.get("budget_drive_root") or q.DRIVE_ROOT), binding["task_id"]
    with launch_lock(root, task_id, wait=False):
        if not _stands(q, task_id, binding, limit) or _membership(q, binding) or _tails_hold(q, binding, []):
            return _withdraw(q, task_id, claim)
        fence = read_fence(root, task_id)
        # Admission reserves under Q: no member or reservation may slip past the release. Nothing waits on a store
        # under Q (§10.25): contention raises, the claim is released and the watchdog retries the same request.
        with q._queue_lock, cancellation_projection_lock(q.DRIVE_ROOT, wait=False), acquisition_budget(0):
            # After every read: the clock is still past the bound limit, this Close still owns its open request (the
            # projection lock keeps an ordinary Stop from replacing it before the CAS), and the release itself is a
            # CAS on this exact fence id and state (``set_fence_state``).
            changed = bool(_membership(q, binding) or _reserved(q, binding)
                           or limit is None or time.time() < limit or fence.get("state") != "paused"
                           or (binding["control"].get("fence_id") or fence.get("fence_id")) != fence.get("fence_id")
                           or not conditional(active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True))
                           or not claim_still_owned(q.DRIVE_ROOT, task_id, claim))
            if not changed and _release_settled_late_work(q, root, task_id, fence, deadline_close=group,
                                                          close_token=token) != FENCE_RELEASED:
                raise RuntimeError("Close could not retire the exact late-work fence")
    if changed:
        return _withdraw(q, task_id, claim)
    _settle_intent(q, task_id, outcome=CANCEL_ALREADY_SETTLED, detail="kept the existing answer", intent=claim)
    return CANCEL_ALREADY_SETTLED


_NOT_CLOSED = "not closed: the saved park changed and nothing was closed. Do not retry it as ordinary Stop"
_CLOSED = "closed; its materials are kept"
_KEPT = "not closed: it had already ended by itself and its own result is kept"


def _drive_requested(q, task_id, claim, spec):
    """A direct parent's durable request: bind its exact park now or withdraw it. The request was the parent's own
    while it was the direct parent, so a parent that has since ended does not void it; it only hears nothing."""
    parent, token, group = spec["requested_by"], spec["token"], claim["request_id"]
    stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
    if _settled_close(stored, "", group):
        return _finish_replay(q, task_id, claim, stored)
    if stored.get("status") in SETTLED_STATUSES:  # it ended by itself after the request: its own result stands
        return _kept_alone(q, task_id, claim, stored)
    seen = _read(observe(q, task_id))  # unread: the claim returns and the watchdog retries this same request
    if ((seen.get("offer") or {}).get("close_expired") != token
            or seen["binding"]["parent_task_id"] != parent):
        return _withdraw(q, task_id, claim)
    # Every outcome reaches the parent before the request retires (``_owed``, ``_withdraw``).
    return _close_one(q, seen["binding"], claim, group, token=token, limit=seen["limit"])


def _tell_parent(q, parent, task_id, group, words) -> bool:
    """True when the parent has the outcome or nobody can receive it; False when unconfirmed."""
    from ouroboros.owner_mailbox import mailbox_drain_ended, write_task_message

    try:
        actor = _parent_actor(q, parent)
        if actor is None:
            return True
        drive = q._task_drive_for_task(actor, parent)
        if mailbox_drain_ended(drive, parent):
            return True
        return bool(write_task_message(drive, f"Close of {task_id}: {words}.", parent, source_task_id=task_id,
                                       provenance="system", msg_id=f"close-outcome:{group}", work_attention={
                                           "kind": "deadline_expired", "task_id": task_id,
                                           "waiting_for_task_id": task_id, "episode_id": group}))
    except Exception:
        log.warning("The Close outcome of %s did not reach %s", task_id, parent, exc_info=True)
        return False


def _group_mark(q, member_id, group) -> str:
    """This group's own durable mark on a frozen member, read open first: ``open`` while a request of this group is
    open on it (a driver may be committing it), ``closed`` once its cancelled write names the group, else ``""``. The
    member's request settles only after that write, so a commit between the two reads is never missed."""
    current = active_intent(q.DRIVE_ROOT, member_id, strict=True, include_close=True) or {}
    if conditional(current) and (current.get("parked_close") or {}).get("group_request") == group:
        return "open"
    return "closed" if _settled_close(load_task_result(q.DRIVE_ROOT, member_id, strict=True) or {}, "", group) else ""


def _exact_claim(q, task_id, matches):
    """Take the claim of this exact open Close only. ``_close_replaced`` keeps the request id and generation on an
    ordinary Stop that replaced it, so the claim write itself requires a conditional Close with this frozen spec
    (``claim_intent(only_if=...)``): anything else is never claimed. ``None``: this caller holds no claim."""
    from ouroboros.cancel_intents import claim_intent

    try:
        taken = claim_intent(q.DRIVE_ROOT, task_id, owner="cancel_task_custody",
                             only_if=lambda row: conditional(row) and matches(row))
    except Exception:
        log.warning("Close of %s: its claim was not taken", task_id, exc_info=True)
        return None
    return None if not taken or taken.get("claim_refused") else taken


def drive_request(q, task_id, request_id):
    """Drive one exact Close request (its HTTP door): ``None`` when this caller took no claim of it (it is no longer
    open, an ordinary Stop replaced it, or another custody holds it), so it never drives another action."""
    try:
        current = active_intent(q.DRIVE_ROOT, task_id, strict=True, include_close=True) or {}
    except Exception:
        log.debug("Close of %s: its request was not read", task_id, exc_info=True)
        return None
    if not (conditional(current) and current.get("request_id") == request_id):
        return None
    spec = current.get("parked_close")
    claim = _exact_claim(q, task_id, lambda row: row.get("request_id") == request_id and row.get("parked_close") == spec)
    return None if claim is None else drive(q, task_id, claim)


def drive(q, task_id, claim) -> str:
    """Run only the frozen park scope. A partial close refuses a wider retry."""
    from supervisor.task_lifecycle import _release_intent_claim

    try:
        spec = claim.get("parked_close") or {}
        binding, token, limit = spec.get("binding") or {}, spec.get("token") or "", spec.get("limit")
        if not valid_token(token):
            return _withdraw(q, task_id, claim)
        if not binding and spec.get("requested_by"):
            return _drive_requested(q, task_id, claim, spec)
        if binding.get("task_id") != task_id:
            return _withdraw(q, task_id, claim)
        group = spec.get("group_request") or claim["request_id"]
        if spec.get("group_request"):
            return _close_one(q, binding, claim, group)
        stored = load_task_result(q.DRIVE_ROOT, task_id, strict=True) or {}
        if _settled_close(stored, "", group) or binding.get("late_phase") and _kept_late(stored, "", group):
            return _close_one(q, binding, claim, group, token=token, limit=limit)  # its own write committed: replay
        members = spec.get("members") or []
        closed = {m["task_id"] for m in members if _group_mark(q, m["task_id"], group) == "closed"}
        already = bool(closed)  # a replay may find this exact group's members closed before a later refusal

        def refuse():
            """Before this request retires, recount the group's own marks: a competing driver (the watchdog on a
            member's own request) may have committed a member since the reads above, or may still be committing one."""
            marks = [_group_mark(q, m["task_id"], group) for m in members]
            if "open" in marks:  # a member may still commit: never retire as no change, the watchdog retries
                _release_intent_claim(q, task_id, error="Parked member custody pending", intent=claim)
                return CANCEL_FAILED
            return _withdraw(q, task_id, claim, PARTIAL if already or "closed" in marks else CHANGED)

        if (not _stands(q, task_id, binding, limit)
                or set(_membership(q, binding)) - {m["task_id"] for m in members}):
            return refuse()
        remaining = []
        for member in members:
            member_id = member["task_id"]
            if member_id in closed:
                continue
            if _observation(q, member_id, nested=True)[1] != member:
                if _group_mark(q, member_id, group) != "closed":
                    return refuse()
                already = True  # another driver of this group closed it meanwhile: complete, not changed
                continue
            remaining.append(member)
        if _tails_hold(q, binding, [m["task_id"] for m in members]) or _reserved(q, binding):
            return refuse()
        for member in remaining:
            member_id = member["task_id"]
            # Root must remain the same expired park before any further member is taken.
            if not _stands(q, task_id, binding, limit):
                return refuse()
            asked = request_cancel(q.DRIVE_ROOT, member_id, reason="the work's deadline expired", source=SOURCE,
                                   parked_close={"binding": member, "token": token, "group_request": group,
                                                 "group_root": task_id, "group_limit": limit})
            if asked.get("already_settled") or asked.get("close_conflict"):
                if _group_mark(q, member_id, group) != "closed":
                    return refuse()
                already = True
                continue
            child_claim = _exact_claim(q, member_id, lambda row: (
                row.get("request_id") == asked.get("request_id")
                and (row.get("parked_close") or {}).get("group_request") == group
                and (row.get("parked_close") or {}).get("binding") == member))
            if child_claim is None:
                _release_intent_claim(q, task_id, error="Parked member custody pending", intent=claim)
                return CANCEL_FAILED
            try:
                changed = _close_one(q, member, child_claim, group)
            except Exception:
                _release_intent_claim(q, member_id, error="Parked member capture pending", intent=child_claim)
                raise
            if changed not in {CANCEL_CANCELLED, CANCEL_ALREADY_SETTLED}:
                if changed == CHANGED:
                    return refuse()
                _release_intent_claim(q, task_id, error="Parked member capture pending", intent=claim)
                return CANCEL_FAILED
            already = already or changed == CANCEL_CANCELLED  # a member that ended by itself was not closed by us
        outcome = _close_one(q, binding, claim, group, token=token, limit=limit)
        return PARTIAL if already and outcome in {CHANGED, CANCEL_NOT_FOUND} else outcome
    except Exception:
        log.warning("Expired park custody pending for %s", task_id, exc_info=True)
        _release_intent_claim(q, task_id, error="Expired park custody pending", intent=claim)
        return CANCEL_FAILED
