"""After Panic only my human's own action resumes operation (BIBLE, Emergency Stop Invariant). Their press in the
app's list returns one work (``gateway.tasks.api_task_resume``); their ``/bg start`` from any owner door turns
consciousness back on and hands over no choice (the owner 27.11, as in 7.12.1). This module owns their desktop
confirmation that also hands this run's wakes the choice of Panic-stopped work: at that Turn on, or—when
consciousness is already on in a later run while Panic-stopped work still waits—without toggling it off and on.

**The stop's reason** while consciousness is off by Panic, or off in a way the state cannot say
(``start_needs_proof``; a later Off keeps that reason, ``consciousness._persist_stop``), is what the confirmation is
about. The mind's own Start never lifts it (``consciousness._persist_start``); the owner's plain Start lifts it with
no choice, and every hold stays until the owner's Resume or a choice confirmed here.

**Where the press comes from.** The press is accepted only over the desktop shell's closed private control pipe:
``launcher.MainApi.request_panic_consciousness_start`` -> ``launcher_background.request_panic_consciousness_start``
reads the passive preview ``GET /api/consciousness?switch=on``, shows one native OS confirmation dialog with those
lines, and on confirmation writes ``panic_start <basis>`` to the launcher's non-inheritable private stdin pipe
(``server_process.watch_launcher_stop`` -> ``receive_launcher_start``). Chat, ``POST /api/command``, WebSocket frames,
browser JS, mobile/remote clients, older desktop shells without that bridge method, agent tools, skills and letters
cannot reach that pipe; ``confirm_start`` consumes ``basis`` once only while what stands in this process epoch still
matches what was shown (mode, allowance, Panic receipt and holds included), every Off invalidates any earlier ``basis``
(``withdraw``), and the irreversible Panic latch (``panic_termination_requested`` / ``installation_panic``) blocks new
confirmations and grants from the reader side without Panic entering this module. The standard ``command_reply`` row in
Main and state reads report the outcome.

**The choice.** Where the dialog said so (the Panic receipt and its holds read as trusted controls and a fresh queue
snapshot, not Advises, an allowance above $0), the live controller records those holds in this server run's memory
only (``_CHOICE``, never persisted to ``state.json``; ``worker_view`` carries a session-stamped display copy for a
spawned wake worker's context and never grants authority). Every new launch, Restart or Update begins with no Panic
choice and opens no dialog on its own; if consciousness is already on and Panic-stopped work still waits, the owner may
confirm that handover on desktop in the same way (``why == "held"``), also after an ordinary Off and On in this run
(``note_start``: the earlier choice never comes back by itself). A root wake of consciousness's own clock may then
return one shown work, a helper too, by a request that says so from the start:
``resume_task(task_id, chosen_hold)`` with the hold its own Drive state showed (``wake_fact``). Only that request
reaches ``resume_chosen``, and it always answers itself: knowing an id is no right,
so the host checks this run's choice, the wake, the live row's hold and the expected one, again under the queue lock
where the grant is made (``choice_stands``), and releases through that hold's conditional Resume
(``expected_panic_hold_id``) as Ouroboros, never as the owner, and never its tree (no cascade). A gone, changed or
no longer chosen stop refuses before any effect and never becomes an ordinary Resume; a resume_task without
``chosen_hold`` is the ordinary one, unchanged. A separate Stop or Pause, the owner's Pause on the tree, a deadline,
money (the tree's included), an unknown start, the wake's own tree and, for a helper, a parent still held, paused,
stopped or unreadable (``_parent_refusal``) refuse; a parent that runs again or finished without a Stop does not. Off
(in memory at once, saved or not), a server exit/restart and the next Panic (via the Panic latch, unreceived flag or
persisted Panic controls) end the choice; a grant already past its last check under the queue lock is written before
an Off returns (``_GRANT_LOCK``), none is admitted after, and nothing written is taken back. A launch, a Restart, a
setting, a task's own act and a letter create no choice; Panic itself never waits on this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from typing import Any

log = logging.getLogger(__name__)

SWITCH_CONTROL = "consciousness_toggle"  # the app's one switch (web/modules/consciousness_switch.js)
NOT_APPLIED = "This Turn on was not applied: "
NOT_AS_SHOWN = NOT_APPLIED + ("it did not match what the switch last showed. Look at the switch again: it shows what "
                              "stands now.")
CHANGED = {"ok": False, "error": "panic_selection_changed"}

_PROCESS_NONCE = secrets.token_hex(16)
_EPOCH = [0]
_OFF_HERE = [False]  # this process took an Off since the last choice it recorded (the stop's own write may have failed)
_CHOICE: list[dict | None] = [None]  # live controller authority for THIS server run only; never persisted on disk
_LOCK = threading.Lock()
_GRANT_LOCK = threading.Lock()  # a wake's grant, from its last check under Q to its write (``grant_lock``)
_START_HANDLER: list = [None]
_PENDING_PIPE: list = []
_DRAINING = [False]


def _is_hex32(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(ch in "0123456789abcdef" for ch in value)


def last_panic(state: Any) -> int | None:
    """The last Panic this installation received, named by its input's write time in nanoseconds, from one state
    view; ``None``: none received. ``LookupError`` when the receipt is not a known control (an unconfirmed recovery,
    an unreadable state): never read as no Panic, never as the one a backup remembers."""
    from supervisor.state import control_value

    known, receipt = control_value(state, "panic_ingress_receipt")
    if not known:
        raise LookupError("the Panic receipt is not a confirmed control")
    stamp = receipt.get("mtime_ns") if isinstance(receipt, dict) else None
    return stamp if type(stamp) is int and stamp > 0 else None


def _panic_latched(root: Any = None) -> bool:
    """Whether Panic termination is latched in this process or an unreceived ``panic_stop.flag`` stands on disk."""
    from ouroboros.server_control import panic_termination_requested

    if panic_termination_requested():
        return True
    if root is not None:
        from ouroboros.installation_stop import installation_panic

        return installation_panic(root)
    return False


def active_choice() -> dict | None:
    """This server run's live controller choice, or ``None`` when none was confirmed here or Off/Panic ended it."""
    if _OFF_HERE[0] or _panic_latched():
        return None
    chosen = _CHOICE[0]
    return ({"panic": chosen.get("panic"), "holds": dict(chosen.get("holds") or {})}
            if isinstance(chosen, dict) else None)


def note_start(shown: dict | None) -> None:
    """A saved Start: after an Off taken here it clears that Off without bringing back the choice it ended, and no
    earlier ``basis`` applies; only a confirmed ``shown`` that offered the choice records its holds, in this server
    run's live controller memory only. A Start of a clock already on changes nothing here."""
    if _panic_latched():
        return
    with _LOCK:
        if _panic_latched():
            return
        if shown and shown.get("choice"):
            _CHOICE[0] = {"panic": int(shown["panic"]), "holds": dict(shown.get("holds") or {})}
        elif _OFF_HERE[0]:
            _CHOICE[0] = None
            _EPOCH[0] += 1
            _PENDING_PIPE.clear()
        _OFF_HERE[0] = False


def reset_session() -> None:
    """Begin a fresh controller run with zero prior Panic choice and a new confirmation epoch."""
    _CHOICE[0] = None
    _OFF_HERE[0] = False
    with _LOCK:
        _CHOICE[0] = None
        _EPOCH[0] += 1
        _PENDING_PIPE.clear()


def worker_view() -> dict | None:
    """Session-stamped display snapshot for a spawned wake worker's context; never read by grant decisions."""
    from ouroboros.process_custody import current_custody_session_id

    chosen = active_choice()
    if not chosen:
        return None
    return {"session": current_custody_session_id(), "panic": chosen["panic"], "holds": dict(chosen["holds"])}


def start_needs_proof(state: Any) -> str:
    """Why a stop is more than an ordinary one, over one state view (a projection or the live write): ``"panic"``
    (Panic turned consciousness off), ``"unknown"`` (it is not known on and who stopped it cannot be read), else
    ``""`` (on, or off by a known ordinary stop). The desktop confirmation is about it; a later Off keeps it."""
    from supervisor.state import control_value

    known, enabled = control_value(state, "bg_consciousness_enabled")
    if known and enabled is True:
        return ""
    source_known, source = control_value(state, "bg_consciousness_stop_source")
    return "unknown" if not (known and source_known) or source == "unknown" else "panic" if source == "panic" else ""


def _lineage(task_id: str, record: Any) -> dict:
    from ouroboros.task_results import resolve_task_lineage

    record = record if isinstance(record, dict) else {}
    return resolve_task_lineage(task_id, **{key: record.get(key) for key in (
        "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})


def _held(root: Any) -> dict | None:
    """``{task_id: panic_hold_id}`` of the pending works their own Panic hold keeps, past no deadline, as the fresh
    published queue snapshot says; ``None`` when no fresh snapshot can be read. No queue lock is taken."""
    from ouroboros.parked_deadline import expired_park
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_status import queue_snapshot_observation
    from ouroboros.work_stop import panic_hold_id

    try:
        snapshot = load_queue_document(root)
        if not queue_snapshot_observation(snapshot)["fresh"]:
            return None
        rows = [{**(outer.get("task") or {}), "id": outer.get("id"), "_attempt": outer.get("attempt")}
                for outer in snapshot.get("pending") or () if isinstance(outer, dict)]
        return {str(row["id"]): hold for row in rows if (hold := panic_hold_id(row)) and not expired_park(row)}
    except Exception:
        log.warning("Works held by Panic could not be read from the queue snapshot", exc_info=True)
        return None


def _live_queue_has_no_panic_holds(root: Any) -> bool:
    """In the process that owns ``root``'s live queue, ``PENDING`` is in memory: when no pending row carries a
    Panic hold, ordinary polls need no ``queue_snapshot.json`` or ``queue.alive`` read."""
    from ouroboros.work_stop import panic_hold_id
    from supervisor import queue as q

    if not q.owns_live_queue(root):
        return False
    return not any(panic_hold_id(row) for row in list(q.PENDING))


def _unchosen_holds(panic: int | None, holds: dict | None) -> bool:
    if not panic or not holds:
        return False
    chosen = active_choice()
    covered = (chosen.get("holds") if isinstance(chosen, dict) and chosen.get("panic") == panic
               and isinstance(chosen.get("holds"), dict) else {})
    return any(covered.get(tid) != hid for tid, hid in holds.items())


def standing(root: Any) -> dict:
    """What an owner's desktop confirmation would be about now: ``{}`` when there is nothing for it to add, else
    ``{"why", "panic", "holds", "choice"}`` where ``why`` is ``"panic"``, ``"unknown"`` or ``"held"`` (consciousness is
    already on, and Panic-stopped work not yet handed over in this run still waits)."""
    from ouroboros.runtime_limits import get_consciousness_autonomy, get_consciousness_daily_usd
    from supervisor.state import control_is, read_state

    if _panic_latched(root):
        return {"why": "panic", "panic": None, "holds": None, "choice": False}
    state = read_state(root).projection()
    why = start_needs_proof(state)
    if not why:
        if (_OFF_HERE[0] or not control_is(state, "bg_consciousness_enabled", True)
                or _live_queue_has_no_panic_holds(root)):
            return {}
        try:
            panic = last_panic(state)
        except LookupError:
            panic = None
        holds = _held(root) if panic else None
        if not _unchosen_holds(panic, holds):
            return {}
        choice = get_consciousness_autonomy() != "observe" and get_consciousness_daily_usd() != 0
        return {"why": "held", "panic": panic, "holds": holds, "choice": choice}
    try:
        panic = last_panic(state) if why == "panic" else None
    except LookupError:
        panic = None
    holds = _held(root) if panic else None
    choice = bool(holds) and get_consciousness_autonomy() != "observe" and get_consciousness_daily_usd() != 0
    return {"why": why, "panic": panic, "holds": holds, "choice": choice}


def handover_available(root: Any) -> bool:
    """Whether consciousness is already on and unchosen Panic-stopped tasks can be handed to this run's wakes."""
    from ouroboros.runtime_limits import get_consciousness_autonomy, get_consciousness_daily_usd

    if (_OFF_HERE[0] or _live_queue_has_no_panic_holds(root) or _panic_latched(root)
            or get_consciousness_autonomy() == "observe" or get_consciousness_daily_usd() == 0):
        return False
    try:
        shown = standing(root)
        return bool(shown.get("why") == "held" and shown.get("choice"))
    except Exception:
        return False


def standing_basis(shown: Any, *, epoch: int | None = None) -> str:
    """32-hex digest binding what ``shown`` and current limits mean in this process epoch."""
    from ouroboros.runtime_limits import get_consciousness_autonomy, get_consciousness_daily_usd

    if not isinstance(shown, dict) or not shown.get("why"):
        return ""
    with _LOCK:
        ep = _EPOCH[0] if epoch is None else int(epoch)
    raw = json.dumps({
        "nonce": _PROCESS_NONCE, "epoch": ep,
        "why": shown["why"], "panic": shown.get("panic"), "holds": shown.get("holds"),
        "choice": bool(shown.get("choice")), "autonomy": get_consciousness_autonomy(),
        "zero_allowance": get_consciousness_daily_usd() == 0,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def preview_start(root: Any) -> dict:
    """Passive post-Panic preview facts for ``GET /api/consciousness?switch=on``: ``{}`` when ``standing`` is empty,
    else ``{"why", "choice", "held", "basis"}`` (``basis == ""`` when state could not be read: fails closed)."""
    with _LOCK:
        ep = _EPOCH[0]
    try:
        shown = standing(root)
    except Exception:
        log.warning("What a Turn on after Panic would be about could not be read", exc_info=True)
        return {"why": "unknown", "choice": False, "held": None, "basis": ""}
    if not shown:
        return {}
    basis = "" if _panic_latched(root) else standing_basis(shown, epoch=ep)
    return {"why": shown["why"], "choice": shown["choice"],
            "held": None if shown["holds"] is None else len(shown["holds"]), "basis": basis}


def withdraw(*, wait: bool = True) -> None:
    """An ordinary Off, as this process takes it and before any write: any open confirmation ends, every earlier
    ``basis`` is spent and this run's choice is cleared in memory at once. The clock takes it under its own lock
    (``wait=False``), the lock its Start holds from ``confirm_start`` through ``note_start``, so either that Start's
    choice is recorded first and this clears it, or this comes first and that Start finds the Off; then it waits
    outside that lock for a wake's grant already being written (``settle_grants``). Panic's own latch ends the choice:
    under it this does nothing and waits on nothing."""
    if _panic_latched():
        return
    with _LOCK:
        _OFF_HERE[0] = True
        _CHOICE[0] = None
        _EPOCH[0] += 1
        _PENDING_PIPE.clear()
    if wait:
        settle_grants()


def settle_grants() -> None:
    """After ``withdraw``: a wake's grant already past its last check is written first, none is admitted after
    (``grant_lock``). Never under Panic."""
    if not _panic_latched():
        with _GRANT_LOCK:
            pass


def confirm_start(root: Any, basis: Any, *, epoch: int | None = None) -> dict | None:
    """Consume ``basis`` once if it matches this process's current epoch and what stands now."""
    if not _is_hex32(basis) or _panic_latched(root):
        return None
    with _LOCK:
        now_ep = _EPOCH[0]
        if epoch is not None and epoch != now_ep:
            return None
    try:
        shown = standing(root)
        if _panic_latched(root) or not (shown and hmac.compare_digest(standing_basis(shown, epoch=now_ep), basis)):
            return None
    except Exception:
        log.warning("What a confirmed Turn on is about could not be read again", exc_info=True)
        return None
    with _LOCK:
        if _EPOCH[0] != now_ep or _panic_latched(root):
            return None
        _EPOCH[0] += 1
        return {**shown, "_epoch": _EPOCH[0]}


def still_valid_at_write(live: Any, shown: dict) -> bool:
    """Re-verify under the state lock that no Off, Panic latch/receipt or limit change raced after ``confirm_start``."""
    from ouroboros.runtime_limits import get_consciousness_autonomy, get_consciousness_daily_usd
    from supervisor.state import control_is

    if _panic_latched():
        return False
    with _LOCK:
        if shown.get("_epoch") is not None and _EPOCH[0] != shown["_epoch"]:
            return False
    expected_why = shown.get("why")
    if expected_why == "held":
        if _OFF_HERE[0] or start_needs_proof(live) or not control_is(live, "bg_consciousness_enabled", True):
            return False
        try:
            panic = last_panic(live)
        except LookupError:
            panic = None
    else:
        why = start_needs_proof(live)
        if not why or why != expected_why:
            return False
        try:
            panic = last_panic(live) if why == "panic" else None
        except LookupError:
            panic = None
    if panic != shown.get("panic"):
        return False
    choice = bool(shown.get("holds")) and get_consciousness_autonomy() != "observe" and get_consciousness_daily_usd() != 0
    return choice == bool(shown.get("choice"))


def bind_launcher_start(handler: Any) -> None:
    """Bind (or clear with ``None``) the current server generation's ``BackgroundConsciousness`` accessor."""
    with _LOCK:
        _START_HANDLER[0] = handler
        if handler is None:
            _PENDING_PIPE.clear()


def _run_launcher_start(basis: str, epoch: int) -> None:
    from ouroboros.consciousness import OWNER_STOP, owner_chat_of
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.server_control import panic_termination_requested
    from ouroboros.server_process import _exit_signalled, _restart_requested, _supervisor_stop
    from supervisor.message_bus import send_with_budget, try_get_bridge
    from supervisor.state import load_state

    if _exit_signalled.is_set() or _supervisor_stop.is_set() or _restart_requested.is_set() or panic_termination_requested():
        return
    with _LOCK:
        getter = _START_HANDLER[0]
    clock = getter() if callable(getter) else getter
    if clock is None:
        result = NOT_APPLIED + "the task supervisor is not ready yet. Try again in a moment."
    else:
        result = clock.start(source=OWNER_STOP, basis=basis, epoch=epoch)
    if _exit_signalled.is_set() or panic_termination_requested() or try_get_bridge() is None:
        return
    try:
        bound = owner_chat_of(load_state())
        chat_id = WEB_UI_CHAT_ID if bound is None else int(bound)
    except Exception:
        chat_id = WEB_UI_CHAT_ID
    try:
        send_with_budget(chat_id, f"🧠 {result}", role="system", system_type="command_reply")
    except Exception:
        log.warning("Launcher post-Panic consciousness reply was not delivered", exc_info=True)


def _drain_launcher_starts() -> None:
    while True:
        with _LOCK:
            if not _PENDING_PIPE:
                _DRAINING[0] = False
                return
            basis, epoch = _PENDING_PIPE.pop()
        try:
            _run_launcher_start(basis, epoch)
        except Exception:
            log.warning("Launcher post-Panic consciousness start failed", exc_info=True)


def receive_launcher_start(basis: Any) -> bool:
    """Record one private-pipe confirmation and hand off without blocking the launcher stop watcher."""
    if not _is_hex32(basis) or _panic_latched():
        return False
    with _LOCK:
        if _panic_latched():
            return False
        _PENDING_PIPE[:] = [(basis, _EPOCH[0])]
        if _DRAINING[0]:
            return True
        _DRAINING[0] = True
    try:
        threading.Thread(target=_drain_launcher_starts, name="launcher-panic-start", daemon=True).start()
    except RuntimeError:
        with _LOCK:
            _DRAINING[0] = False
        return False
    return True


def choice_decision(chosen: Any, *, wake: bool, enabled: Any, autonomy: str, last_panic: Any, task_id: str,
                    hold_id: str) -> str:
    """``"none"`` (this call stands on no choice: the ordinary door answers), ``"granted"`` or ``"changed"``, from
    facts read now. ``wake``: the caller is a root wake of consciousness's own clock above Observe. A work the saved
    choice names is granted only while consciousness is on, not Advises, the Panic chosen about is still the last one
    received and the work's hold is the one shown."""
    holds = chosen.get("holds") if isinstance(chosen, dict) else None
    if not (wake and isinstance(holds, dict) and task_id in holds):
        return "none"
    return "granted" if (enabled is True and autonomy != "observe" and chosen.get("panic") is not None
                         and chosen["panic"] == last_panic and hold_id and holds[task_id] == hold_id) else "changed"


def _decide(q, task: Any, task_id: str, wake: bool) -> str:
    """``choice_decision`` over this run's controller choice, the switch, the mode and the receipt as they read now,
    and the live row's own hold; an unreadable fact is ``changed`` for a wake, never a grant."""
    from ouroboros.runtime_limits import get_consciousness_autonomy
    from ouroboros.work_stop import panic_hold_id
    from supervisor.state import control_is

    try:
        if _panic_latched(q.DRIVE_ROOT):
            return "none"
        state = q.load_state()
        chosen = None if start_needs_proof(state) == "panic" else active_choice()
        return choice_decision(chosen, wake=wake, enabled=control_is(state, "bg_consciousness_enabled", True),
                               autonomy=get_consciousness_autonomy(), last_panic=last_panic(state),
                               task_id=task_id, hold_id=panic_hold_id(task))
    except Exception:
        log.warning("The owner's consciousness choice could not be read for a Panic return", exc_info=True)
        return "changed" if wake else "none"


def choice_stands(q, task: Any) -> bool:
    """Under the queue lock, where the grant is made: the choice this call was released on still covers ``task``."""
    return _decide(q, task, str((task or {}).get("id") or ""), True) == "granted"


def grant_lock() -> Any:
    """Held by a wake's grant under Q from its last check of the choice through its write
    (``budget_resume.panic_selection_lock``); ``withdraw`` waits it out, so an Off once taken admits no grant."""
    return _GRANT_LOCK


def parent_refusal(q, task: Any) -> dict | None:
    """Under Q, for a helper chosen alone: each ancestor up to its root is shown to run again or to have finished
    without a Stop, the root read as its physical actor (``cancel_intents._validated_single_cancel_target``, a timeout
    retry included). Running again: a live row (``RUNNING``, or a direct turn) not parked warm under a Pause
    (``owner_pause_control.warm_paused_member``), or a queued row with no stop (``current_stop``; a sleep is a wait).
    Finished: a terminal record other than cancelled. Stopped, held, paused or cut off answers
    ``selection_owner_held``; anything not shown either way, ``selection_authority_unavailable``: the helper stays.
    The callers' own tree checks (``budget_resume._panic_member_refusal``: the root's owner fence and money latch)
    never prove an ancestor live, so this is the one place that does."""
    import pathlib

    from ouroboros.cancel_intents import _validated_single_cancel_target, cancel_pending
    from ouroboros.task_results import (STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, STATUS_INTERRUPTED,
                                        _TRULY_TERMINAL_STATUSES, load_task_result)
    from ouroboros.work_stop import continuation_refusal, current_stop
    from supervisor.owner_pause_control import _warm_paused_direct_turn, warm_paused_member
    from supervisor.workers import direct_chat_turn

    task = task if isinstance(task, dict) else {}
    drive = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    held = {"ok": False, "error": "selection_owner_held"}
    unknown = {"ok": False, "error": "selection_authority_unavailable"}
    root, parent = str(task.get("root_task_id") or ""), str(task.get("parent_task_id") or "")
    seen = {str(task.get("id") or "")}
    try:
        while parent:
            stored = None if parent in seen or len(seen) > 64 else load_task_result(drive, parent, strict=True)
            if not isinstance(stored, dict):
                return unknown
            seen.add(parent)
            actor = _validated_single_cancel_target(drive, parent) if parent == root else parent
            if any(cancel_pending(drive, tid, strict=True) for tid in {parent, actor}):
                return held
            meta = q.RUNNING.get(actor)
            row = next((item for item in q.PENDING if str(item.get("id") or "") == actor), None)
            if isinstance(meta, dict) or row is None and direct_chat_turn(actor) is not None:
                if warm_paused_member(meta) if isinstance(meta, dict) else _warm_paused_direct_turn(drive, actor):
                    return held
            elif row is not None:
                if current_stop(row) or continuation_refusal(row):
                    return held
            else:
                status = str(((stored if actor == parent else load_task_result(drive, actor, strict=True)) or {})
                             .get("status") or "")
                if status in {STATUS_CANCELLED, STATUS_CANCEL_REQUESTED, STATUS_INTERRUPTED}:
                    return held
                if status not in _TRULY_TERMINAL_STATUSES:
                    return unknown
            parent = str(stored.get("parent_task_id") or "")
    except Exception:
        return unknown
    return None


def _own_wake(task_id: str, record: Any) -> str:
    """The logical root of a root wake of consciousness's own clock above Observe that ``record`` (a task's stored
    result or its queue row) shows; "" for any other task."""
    from ouroboros.consciousness_authority import is_observe_origin, is_own_clock_wake

    record = record if isinstance(record, dict) else {}
    lineage = _lineage(task_id, record)
    wake = (task_id and lineage["is_root_task"] and is_own_clock_wake(record.get("metadata"))
            and not is_observe_origin(record.get("metadata")))
    return str(lineage["root_task_id"] or task_id) if wake else ""


def _wake_choice_view(task: dict) -> dict | None:
    if _panic_latched():
        return None
    if os.environ.get("OUROBOROS_IN_WORKER") == "1":
        from ouroboros.process_custody import current_custody_session_id

        view = task.get("_panic_choice_view")
        return view if isinstance(view, dict) and view.get("session") == current_custody_session_id() else None
    return active_choice()


def wake_fact(root: Any, task: Any) -> str:
    """For an own-clock root wake's context: the fact that my human left it the choice, and the works that choice
    would be granted for now, by name. "" for every other reader and when none would be."""
    from ouroboros.runtime_limits import get_consciousness_autonomy
    from ouroboros.task_results import load_task_result
    from supervisor.state import control_is, read_state

    try:
        task = task if isinstance(task, dict) else {}
        if _panic_latched(root) or not _own_wake(str(task.get("id") or ""), task):
            return ""
        state = read_state(root).projection()
        if start_needs_proof(state) == "panic":
            return ""
        chosen, live = _wake_choice_view(task), _held(root) or {}
        facts = dict(wake=True, enabled=control_is(state, "bg_consciousness_enabled", True),
                     autonomy=get_consciousness_autonomy(), last_panic=last_panic(state))
        works = [work for work in (chosen.get("holds") if isinstance(chosen, dict) else None) or ()
                 if choice_decision(chosen, task_id=work, hold_id=live.get(work, ""), **facts) == "granted"]
        stored = {work: load_task_result(root, work) or {} for work in works}
        names = [f"«{str(stored[work].get('title') or 'untitled')[:80]}» (task_id {work}, chosen_hold {live[work]}"
                 + (f", helper of {parent}" if (parent := str(stored[work].get('parent_task_id') or '')) else "")
                 + ")" for work in works]
    except Exception:
        log.debug("The consciousness choice was not read for a wake's context", exc_info=True)
        return ""
    return ("After Panic my human left me the choice of the work it stopped. Still stopped and mine to decide: "
            f"{'; '.join(names)}. resume_task(task_id, chosen_hold) with the hold shown here returns one, alone; none "
            "is owed, and one of them may be what the Panic was pressed for. If its stop changed since, the request "
            "is refused and nothing else is resumed. A helper returns alone, once its parent runs again or finished "
            "without a Stop. A separate Stop or Pause, a deadline and money still apply. "
            "Off, a restart or a new Panic ends this choice.") if names else ""


def resume_chosen(task_id: str, actor_id: str, expected: str) -> dict:
    """A wake's conditional Resume of ``task_id`` on my human's Turn on, asked for by ``chosen_hold`` (``expected``:
    the hold its Drive state showed). It always answers itself and never falls through to the ordinary Resume: the
    caller is not a wake of the clock, names its own tree or stands on no choice of this run (the owner's Resume is
    needed); the stop is not the one expected and chosen (``panic_selection_changed``); else that hold's conditional
    Resume, as Ouroboros."""
    from ouroboros.control_actor import panic_release
    from ouroboros.task_results import load_task_result
    from ouroboros.work_stop import panic_hold_id, resume_outcome
    from supervisor import queue as q
    from supervisor.queue_transitions import resume_budget_paused_task

    with q._queue_lock:
        row = next((dict(item) for item in q.PENDING if str(item.get("id") or "") == task_id), None)
    try:
        own_root = _own_wake(actor_id, load_task_result(q.DRIVE_ROOT, actor_id, strict=True))
        mine = own_root in (task_id, _lineage(task_id, row)["root_task_id"])  # its own tree is not its call
    except Exception:
        own_root, mine = "", True
    decision = _decide(q, row, task_id, True) if own_root and task_id != actor_id and not mine else "none"
    if decision == "none":
        return resume_outcome({"ok": False, "error": "panic_requires_owner_resume"})
    if decision != "granted" or not expected or panic_hold_id(row) != expected:
        return resume_outcome(dict(CHANGED))
    basis = {"kind": "consciousness_choice", "task_id": task_id}
    with panic_release(basis):
        return {**resume_budget_paused_task(task_id, expected_panic_hold_id=expected), "panic_release": basis}
