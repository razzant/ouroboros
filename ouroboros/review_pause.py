"""A paid review operation its author leaves: owner Pause, or a wait-only exit.

Pause (owner 2026-10-08, full variant): the author is truly paused while reviewers
it ALREADY launched
finish the same check separately and visibly; Resume does not wait for them and
later collects their result without another purchase; nothing new starts and
the author commits, applies or publishes nothing. A wait-only exit
(``left_wait_only``: addressed mail, a selected helper or reviewer outcome, the
wake time, a departed author) is not Pause: the author bound by ``receptive_author``
leaves only its own wait (it can answer at once instead of waiting deaf), every
launched or still-unsent reviewer continues over the frozen subject, the same
unchanged request rejoins for free, and ``watch_left_review`` keeps the wait
intention and one settlement notice (none for a departed author). This module is
that seam between the review custody
(``review_custody``/``review_operation``) and the author (``owner_pause``):

- the author's synchronous drain detaches on Pause, including while a slot
  prepares to launch. Under the physical handoff's same root launch lock, an
  unarmed slot latches a first-launch refusal; its still-live preparation may
  unwind later but cannot send, even if Resume came first. An armed reviewer
  (``owner_pause.review_episode``) finishes its already-started episode;
- the row the author takes with it is PENDING (``in_flight`` for a launched
  episode, ``pending_dispatch`` for a retained unsent preparation,
  ``late_result_pending``), never a verdict; the physical operation stays live
  and settles late into this process's custody, so the same unchanged request
  after Resume relays or replays it;
- the operation is listed as detached until it CLOSES (its workers settled and
  published, ``ReviewOperation.maybe_close``), which keeps the author's process
  parked warm; its state is what this process knows — running, closed here, or
  unknown — never "settled" from absence.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import logging
import pathlib
import threading
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ouroboros.owner_pause import episode_armed
from ouroboros.review_operation import _CLOSED_DETACHED, _LIVE, _LOCK, START_UNBOUND, ReviewOperation

log = logging.getLogger(__name__)

# How often the author's drain re-reads its own owner fence while its review
# operation runs: a poll interval of the existing wait, not a bound.
DETACH_POLL_SEC = 1.0
# The author's drain mark outlives any panel window; the supervisor bounds its lease by the task.
_DRAIN_MARK_SEC = 6 * 3600.0
ADDRESSED_EVENT = "addressed_event"
HELPER_EVENT = "helper_result"
REVIEW_EVENT = "review_result"
TIME_EVENT = "wake_at"
# The author never waits at all (a Presence completion: its speech is never held): its drain leaves at once,
# no mailbox notice follows, and the wave settles as an unwaited one (``owe_unwaited_review``).
DEPARTED_EVENT = "author_departed"
# A lost seat's cause, written only by the settlement that runs after its author's task ended.
AUTHOR_ENDED_CAUSE = "no retained outcome when its author's task ended"
# This process's identity: a left-review wait restored from a checkpoint names another
# process, whose live custody and settlement notice died with it.
PROCESS_TOKEN = uuid.uuid4().hex
# The interactive author whose change-review drain hears its addressed mail, bound
# BY VALUE at the panel's dispatch: drains and workers never read the live context.
_RECEPTIVE: contextvars.ContextVar = contextvars.ContextVar("ouroboros_receptive_review_author", default=None)


def receptive_binding(ctx: Any, wake_on: Any = None, *, state: Any = None) -> Optional[Dict[str, Any]]:
    """An author with a continuation owner hears mail while it waits; a script or a
    caller without that owner keeps its synchronous wait (``None``)."""
    if not (callable(getattr(ctx, "owner_wait_callback", None)) or callable(getattr(ctx, "review_wait_callback", None))):
        return None
    drive_root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if not drive_root or not task_id:
        return None
    from ouroboros.owner_mailbox import OwnerMailboxPeek
    from ouroboros.review_waiting import select_helpers

    return {"drive_root": pathlib.Path(drive_root), "task_id": task_id, "attempt": getattr(ctx, "task_attempt", None) or 1,
            "task_attempt": getattr(ctx, "task_attempt", None) or 1,
            "_loop_mailbox_seen_ids": frozenset(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()),
            "_sleep_interaction_seen": copy.deepcopy(getattr(ctx, "_sleep_interaction_seen", None) or {}),
            "seen": frozenset(getattr(ctx, "_loop_mailbox_seen_ids", None) or ()), "peek": OwnerMailboxPeek(),
            "budget_drive_root": getattr(ctx, "budget_drive_root", None), **(copy.deepcopy(state) if state is not None else select_helpers(ctx, wake_on))}


def rejoin_binding(ctx: Any, record_id: str) -> Optional[Dict[str, Any]]:
    """Restore the original selection before draining a repeated paid call."""
    from ouroboros.review_ledger import ledger_root, load_record

    binding = _RECEPTIVE.get()
    if not binding:
        return binding
    record = load_record(ledger_root(ctx), record_id) or {}
    state = record.get("waiting") or {}
    selected = {key: copy.deepcopy(state.get(key) or ({} if key == "selection" else []))
                for key in ("selection", "seen_helpers", "seen_outcomes")}
    ctx._review_wait_selection = selected
    return {**binding, **selected}


def departed_binding(ctx: Any) -> Optional[Dict[str, Any]]:
    """The author of a wave nobody waits for: the drain detaches at its first slice and keeps the workers."""
    drive_root, task_id = getattr(ctx, "drive_root", None), str(getattr(ctx, "task_id", "") or "")
    if not drive_root or not task_id:
        return None
    return {"drive_root": pathlib.Path(drive_root), "task_id": task_id, "departed": True,
            "attempt": getattr(ctx, "task_attempt", None) or 1, "seen": frozenset(), "peek": None}


@contextmanager
def receptive_author(binding: Optional[Dict[str, Any]]):
    """Bind (or, with ``None``, explicitly unbind) the receptive author for one wave."""
    token = _RECEPTIVE.set(binding)
    from ouroboros.swarm_wait_facts import review_started, review_woke
    review_started(binding)
    try:
        yield binding
    except BaseException:
        if binding:
            binding["_swarm_wait_outcome"] = "unwound"
        raise
    finally:
        review_woke(binding, (binding or {}).get("_swarm_wait_outcome", "whole_panel_returned"))
        _RECEPTIVE.reset(token)


def receptive_active() -> bool:
    return bool(_RECEPTIVE.get())


def author_hears_mail() -> bool:
    """An unread addressed message waits for the bound author: the loop's own
    non-destructive peek; delivery and acknowledgement stay with its drain."""
    binding = _RECEPTIVE.get()
    if not binding:
        return False
    try:
        return bool(binding["peek"].pending(binding["drive_root"], binding["task_id"], set(binding["seen"]),
                                            binding["attempt"]))
    except Exception:
        log.debug("receptive mailbox peek failed; the author keeps waiting", exc_info=True)
        return False


@contextmanager
def slot_episode(entry: Any, scope: Any):
    """Bind one physical review attempt's Pause exception for its own worker and
    for the author's drain, which reads whether it launched (``author_detaches``)."""
    from ouroboros.owner_pause import _member_coordinates, launch_lock, review_episode

    root_drive, root_task_id, _ = _member_coordinates(scope)
    with review_episode(root_task_id, entry.operation) as marker:
        with launch_lock(root_drive, root_task_id):
            # The drain may detach before this worker even binds its marker.
            # Carry that refusal into the one marker, never replace an armed one.
            marker["pause_before_launch"] = entry.pause_before_launch
            entry.episode = marker
        yield marker


def drain_slice(operation: Any, remaining: float) -> float:
    """While its author may pause, the drain re-reads the author's fence each slice."""
    return min(remaining, DETACH_POLL_SEC) if operation is not None else remaining


def _author_coordinates(parent: Any) -> Tuple[str, str, str]:
    """``(root_drive, root_task_id, task_id)`` of the author wait an operation serves."""
    from ouroboros.owner_pause import _member_coordinates

    source = getattr(parent, "tool_context", None)
    if source is None:
        task = parent.task if isinstance(getattr(parent, "task", None), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        source = SimpleNamespace(
            task_id=parent.task_id,
            root_task_id=str(task.get("root_task_id") or metadata.get("root_task_id") or parent.task_id),
            budget_drive_root=parent.canonical_root)
    return _member_coordinates(source)


def author_fence_closed(parent: Any) -> bool:
    """A NEW panel under the author's accepted (or unreadable) Pause never starts.

    An early $0 refusal only: no allowance is minted here. Each physical attempt
    earns its own Pause exception at its first admitted handoff
    (``owner_pause.review_episode``).
    """
    from ouroboros.owner_pause import member_fence

    root_drive, root_task_id, task_id = _author_coordinates(parent)
    if not root_drive or not root_task_id:
        return False
    source = getattr(parent, "tool_context", None) or SimpleNamespace(
        task_id=task_id, root_task_id=root_task_id, budget_drive_root=root_drive)
    return bool(member_fence(source))


def author_paused(operation: ReviewOperation) -> bool:
    """Whether the owner's Pause holds this operation's author now.

    Read directly (never through ``member_fence``, which answers "open" for an
    armed reviewer): the author's root fence is closed and the author is not a
    selected member. Unreadable authority reads as not paused: the author's
    drain keeps waiting, which is the pre-existing behavior.
    """
    from ouroboros.owner_pause import fence_closed, read_fence

    root_drive, root_task_id, task_id = _author_coordinates(operation.wait)
    try:
        fence = read_fence(root_drive or str(operation.result_root), root_task_id)
    except Exception:
        log.debug("author pause fence unreadable during the review drain", exc_info=True)
        return False
    return bool(fence_closed(fence) and task_id not in (fence.get("selected_members") or {}))


def author_detaches(operation: Any, pending_entries: Iterable[Any]) -> bool:
    """Detach promptly, retaining workers. Pause refuses every still-unsent slot;
    a wait-only exit (``left_wait_only``) refuses none — it ends only the author's own wait.

    Serialize with the first physical handoff: either that handoff armed the
    same episode first and it finishes, or Pause's refusal wins and survives
    Resume. No worker is reported ended merely because the author left.
    """
    from ouroboros.owner_pause import launch_lock

    if not author_paused(operation):
        departed = bool((_RECEPTIVE.get() or {}).get("departed"))
        from ouroboros.review_waiting import wake_reason

        binding = _RECEPTIVE.get() or {}
        observed = wake_reason(SimpleNamespace(**binding), binding, entries=pending_entries) if binding and not departed else ""
        reason = (DEPARTED_EVENT if departed else HELPER_EVENT if observed.startswith(("task:", "run:"))
                  else TIME_EVENT if observed == "timeout" else REVIEW_EVENT if observed == REVIEW_EVENT
                  else ADDRESSED_EVENT if observed else "")
        if not reason:
            return False
        binding["_swarm_wait_outcome"] = observed or reason
        for entry in pending_entries:
            if entry is not None:  # read by ``detached_actor`` of this drain
                entry.author_left_for = reason
        return True
    root_drive, root_task_id, _ = _author_coordinates(operation.wait)
    with launch_lock(root_drive, root_task_id):
        if not author_paused(operation):
            return False
        if _RECEPTIVE.get() is not None:
            _RECEPTIVE.get()["_swarm_wait_outcome"] = "owner_pause"
        for entry in pending_entries:
            if entry is not None:
                entry.author_left_for = ""  # this detach is Pause's, whatever an earlier wait left for
            if entry is None or episode_armed(entry.episode) or entry.event.is_set():
                continue
            # Fresh local work has started_at even before its first POST; its
            # write-ahead token proves no handoff. Exact recovery leaves it empty.
            if not entry.started_at and any(entry.retry_state.get(key) for key in ("pending_invocation_id", "delegated_run_id")):
                continue  # A recovered remote invocation is not an unsent local preparation.
            entry.pause_before_launch = True
            if isinstance(entry.episode, dict):
                entry.episode["pause_before_launch"] = True
        return True


def left_wait_only(entry: Any) -> bool:
    """The author left only its own wait over this seat, never Pause: an unsent preparation still sends.
    The one definition the drain's PENDING row (``detached_actor``) and the seat's own start checkpoint
    (``review_state_custody``) read."""
    return (not getattr(entry, "pause_before_launch", False) and getattr(entry, "author_left_for", "")
            in {ADDRESSED_EVENT, HELPER_EVENT, REVIEW_EVENT, TIME_EVENT, DEPARTED_EVENT})


def detached_actor(slot: Any, entry: Any, error_actor: Callable[..., Any], *, usage_ctx: Any,
                   request: Any, operation: Any) -> Any:
    """The PENDING row the author takes with it when Pause or a wait-only exit (``left_wait_only``) detaches it.

    The physical operation stays live and settles late into this process's
    custody (``_review_settled_attempts``, keyed by the attempt identity), so the
    same request after Resume, or after a wait-only exit, replays it at $0.
    Never a verdict, never a stop.
    """
    from ouroboros.review_custody import _ACTIVE_LOCK
    from ouroboros.review_operation import stamp_review_controller

    with _ACTIVE_LOCK:
        if entry.event.is_set() and entry.actor is not None:
            try:
                return copy.deepcopy(entry.actor)
            except Exception:
                return entry.actor
        entry.detached = True
        if operation is not None:
            operation.author_detached = True
    preparing = bool(entry.pause_before_launch)
    left_for = getattr(entry, "author_left_for", "")
    yielded = left_wait_only(entry)
    unsent = yielded and not episode_armed(entry.episode)
    message = (
        "Nobody waits for this reviewer (the author's speech is never held): it finishes the same check "
        "separately and its outcome settles into the record and a late mark in the author's room. No verdict "
        "yet; nothing was committed or applied."
        if yielded and left_for == DEPARTED_EVENT else
        ("A selected outcome or wake time was ready while this reviewer was " if left_for in {HELPER_EVENT, REVIEW_EVENT, TIME_EVENT} else
         "An addressed message reached the author while this reviewer was ")
        + ("preparing its first launch, which still sends" if unsent else "running")
        + ": the author left only its own wait and the reviewer finishes the same check separately. No verdict "
        "yet; nothing was committed or applied. The same unchanged request rejoins this attempt in this process; "
        "nothing is sent again automatically."
        if yielded else
        "The owner paused the author before this reviewer's first launch. Its preparation is still owned "
        "by this operation; no reviewer was sent. That preparation cannot send after Resume and must "
        "finish before its final refusal is known. No verdict yet; nothing was committed or applied."
        if preparing else
        "The owner paused the author while this reviewer was already running: it finishes the "
        "same check separately. No verdict yet; nothing was committed or applied. After Resume the "
        "same unchanged request rejoins this attempt in this process; nothing is sent again automatically.")
    actor = error_actor(slot, message, entry.operation_id, "pending_dispatch" if preparing or unsent else "in_flight")
    actor.late_result_pending = True
    actor.awaiting_since = entry.started_at
    usage = dict(getattr(actor, "usage", None) or {})
    if yielded:
        usage["author_left_for"] = left_for
    else:
        usage["owner_pause_detached"] = True
    if preparing:
        usage["owner_pause_preparing"] = True
    usage.update({key: str(entry.retry_state.get(key) or "") for key in ("pending_invocation_id", "delegated_run_id")
                  if entry.retry_state.get(key)})
    actor.usage = usage
    stamp_review_controller(actor, entry)
    if usage_ctx is not None:
        detached = getattr(usage_ctx, "_review_detached_operations", None)
        if not isinstance(detached, list):
            detached = []
            setattr(usage_ctx, "_review_detached_operations", detached)
        detached.append({"owner_id": getattr(operation, "owner_id", ""),
                         "surface": str(getattr(request, "surface", "") or ""),
                         "retry_key": str(getattr(request, "retry_key", "") or ""),
                         "slot_id": str(getattr(slot, "slot_id", "") or ""),
                         "operation_id": entry.operation_id, "attempt_key": entry.key,
                         "reason": left_for if yielded else "owner_pause"})
    return actor


def review_operation_state(owner_id: str) -> str:
    """``running`` while this process owns the operation live, ``closed`` once it
    closed here after its workers settled and published, else ``unknown``.

    A process-local fact for the resumed author's notice; absence is never read
    as settlement (the operation may have lived in a process that is gone).
    """
    owner_id = str(owner_id or "")
    with _LOCK:
        operation = _LIVE.get(owner_id)
        if operation is not None and not operation.closed:
            return "running"
        return "closed" if owner_id in _CLOSED_DETACHED else "unknown"


def live_detached_operations(task_id: str) -> List[Dict[str, Any]]:
    """The open review operations of ``task_id`` its paused author detached from.

    Their launched reviewers keep running in THIS process, so the author parks
    warm here and the process stays until each operation closes (after its
    settlement publication), never released on the settle event alone. Facts
    only: owner id, surface, paid identity.
    """
    task_id = str(task_id or "")
    with _LOCK:
        operations = [op for op in _LIVE.values()
                      if op.task_id == task_id and not op.closed and op.author_detached]
    return [{"owner_id": op.owner_id, "surface": op.surface, "retry_key": op.retry_key,
             "controller": dict(op.identity)} for op in operations]


@contextmanager
def author_drain(parent: Any, operation: Any):
    """The author's synchronous drain of ``operation``: its own call holds the slot
    waiting on this panel (``model_sleep.declare_in_slot_wait``), so while it lasts
    the reply it prepares is the panel's work (``panel_activity``, DESIGN §9 typing)."""
    ctx = getattr(parent, "tool_context", None)
    if ctx is None or operation is None:
        yield
        return
    from ouroboros import model_sleep

    key = f"review:{operation.owner_id}:{uuid.uuid4().hex}"
    model_sleep.declare_in_slot_wait(ctx, key, time.time() + _DRAIN_MARK_SEC)
    try:
        yield
    finally:
        model_sleep.end_in_slot_wait(ctx, key)


def local_panel_seats(task_id: str) -> Tuple[set, set]:
    """This process's open panels of ``task_id``: ``(live seats, seats waiting for model
    access)``, each ``(owner_id, slot_id)`` as their wait rows name it. A settled seat is
    not live, whatever its panel's roster says."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    with _LOCK:
        operations = {id(op): op for op in _LIVE.values() if op.task_id == str(task_id) and not op.closed}
    with _ACTIVE_LOCK:
        entries = list(_ACTIVE.values())
    live = {(operations[id(entry.operation)].owner_id, str(getattr(entry.assignment[1], "slot_id", "") or ""))
            for entry in entries if id(entry.operation) in operations and entry.assignment and not entry.event.is_set()}
    waiting = {(op.owner_id, slot) for op in operations.values() for slot in op.wait.waiting_slots()}
    return live, waiting


def panel_activity(live: Iterable[tuple], waiting: Iterable[tuple], *, owner_id: str = "") -> Optional[bool]:
    """Whether review work prepares the author's answer now: a live seat (of ``owner_id``'s
    panel, else of any) that is not waiting for model access. A drained panel with no live seat
    of its own is unknown, never work: the moments between its seats and a seat whose facts
    are missing or unattributed look the same."""
    seats = {seat for seat in live if not owner_id or seat[0] == owner_id}
    return bool(seats - set(waiting)) if seats else None if owner_id else False


def notify_when_settled(rows: List[Dict[str, Any]], notify: Callable[[int], None], *,
                        settled_keys: Iterable[str] = ()) -> bool:
    """Run ``notify(unknown)`` once EVERY seat these rows name has settled: the wave's own
    recorded roster its author left, never a single-slot request's quorum. A seat already
    outside live custody counts as settled only when this process retained its settled
    outcome (``settled_keys``); otherwise it stays unknown.
    Process-local: a lost process loses this notice, never the paid outcome."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    keys = [str(row.get("attempt_key") or "") for row in rows if row.get("attempt_key")]
    if not keys:
        return False
    settled = set(settled_keys)
    with _ACTIVE_LOCK:
        entries = [(key, _ACTIVE.get(key)) for key in dict.fromkeys(keys)]
    unknown = sum(entry is None and key not in settled for key, entry in entries)

    def run() -> None:
        for _key, entry in entries:
            if entry is not None:
                entry.event.wait()
        try:
            notify(unknown)
        except Exception:
            log.warning("settlement notice for a left review failed", exc_info=True)

    threading.Thread(target=run, name="review-left-settlement", daemon=True).start()
    return True


def watch_left_review(ctx: Any, *, surface: str, retry_key: str, reissue: str, attempt: int = 0,
                      record_id: str = "") -> Optional[Dict[str, Any]]:
    """The author left this wave for mail, a selected outcome or a wake time: keep the wait intention on
    ``ctx._left_review_wait`` (the continuation serializer carries it; a new turn resets it),
    persist the wave's room duty before anything process-local can be lost (``_owe_left_review``),
    and, once the wave's whole recorded roster settles, send ONE system notice to the author's own
    mailbox, or hand the value-bound record to ``ended_left_review`` when no live drain reads that
    mailbox. ``None`` when no seat was left for one of these wait-only reasons."""
    # Every seat THIS call left (a seat Pause latched earlier included), once one has a wait-only reason.
    rows = [row for row in (getattr(ctx, "_review_detached_operations", None) or []) if row.get("retry_key") == retry_key]
    if not retry_key or not any(row.get("reason") in {ADDRESSED_EVENT, HELPER_EVENT, REVIEW_EVENT, TIME_EVENT} for row in rows):
        return None
    from ouroboros.tool_access_paths import canonical_data_root

    task_id, drive_root = str(getattr(ctx, "task_id", "") or ""), pathlib.Path(ctx.drive_root)
    identity = f"{surface}\0{task_id}\0{retry_key}\0{int(attempt or 0)}\0{record_id}"
    wait = {"surface": surface, "binding": f"{surface}-review:{retry_key}", "retry_key": retry_key,
            "attempt": int(attempt or 0), "record_id": str(record_id or ""), "task_id": task_id,
            "reissue": reissue, "owner": PROCESS_TOKEN, "done": False, "unknown_seats": 0,
            "msg_id": f"{surface}-review-settled:" + hashlib.sha256(identity.encode()).hexdigest()[:24]}
    from ouroboros.review_waiting import save_wait

    wait.update(copy.deepcopy(getattr(ctx, "_review_wait_selection", None) or {}))
    if (wait.get("selection") or {}).get("reviewers"):
        wait["selection"]["record_id"] = str(record_id or "")
    wait["review_reference"] = {"record_id": str(record_id), "surface": surface, "task_id": task_id}
    wait["duty_id"] = _left_mark_id(wait)
    if record_id:
        saved = save_wait(ctx, record_id, wait)
        if saved is None:
            return None  # No notice can claim a revision that was never made durable.
    ctx._left_review_wait = wait
    named = f"attempt #{wait['attempt']}" if wait["attempt"] else f"record {wait['record_id'] or retry_key[-12:]}"
    record = {key: wait[key] for key in ("surface", "task_id", "attempt", "record_id", "retry_key", "msg_id",
                                         "selection", "seen_helpers", "seen_outcomes", "review_reference") if key in wait}
    # The late mark's root and the author's drive, bound by value; the duty lands before this call returns.
    record.update(unknown_seats=0, drive_root=str(drive_root), canonical_root=str(canonical_data_root(ctx)))
    _owe_left_review(record)

    def notify(unknown: int) -> None:
        from ouroboros.owner_mailbox import mailbox_drain_ended, write_task_message

        record.update(unknown_seats=int(unknown))
        verdict = _settled_aggregate(record)
        if not verdict:
            unknown = max(1, unknown)
        status = "[Review custody] The local wait ended for" if unknown else "[Review settled] The reviewers finished"
        text = (f"{status} the paid {surface} review ({named}) you left while a chosen event was ready"
                + (f"; durable revision {record.get('revision')} reads {verdict}" if verdict else "") + ". "
                f"That review call committed or applied nothing. Re-issue {reissue} to collect its outcome at $0 "
                "(no second dispatch), or choose another course.")
        if unknown:
            text += (f" {unknown} seat(s) had already left this process's live custody without a retained "
                     "outcome; they stay unknown until that call collects what was recorded.")
        if mailbox_drain_ended(drive_root, task_id):
            wait.update(done=True, noticed=False, unknown_seats=int(unknown), notice=text)
            ended_left_review(record)
            return
        feedback = {key: record[key] for key in ("surface", "task_id", "attempt", "record_id", "retry_key",
                                                 "msg_id", "unknown_seats") if key in record}
        feedback.update({key: record[key] for key in ("revision", "review_reference") if key in record})
        written = write_task_message(drive_root, text, task_id, source_task_id=task_id, provenance="system",
                                     msg_id=wait["msg_id"], review_feedback={**feedback, "settled": not unknown})
        # Done only once the notice is in the mailbox (a park starting now still wakes on it);
        # an unwritten notice is carried as ``notice`` for the author's next round instead. Reading it is
        # no release: the owed duty keeps the outcome for the room once the author's task ended (#1738).
        wait.update(done=True, noticed=bool(written), unknown_seats=int(unknown), notice=text)

    notify_when_settled(rows, notify, settled_keys=(getattr(ctx, "_review_settled_attempts", None) or {}).keys())
    return wait


def owe_unwaited_review(ctx: Any, *, surface: str, retry_key: str, record_id: str, attempt: int = 0) -> bool:
    """A wave returned to its author with seats still running and no addressed message left behind it: the
    author no longer waits for those seats, so the wave's record settles once they finish, at the latest when
    the author's task ended (``ended_left_review``), whether or not the same call is re-issued. The duty is
    the left wave's own row (``_owe_left_review``), written before this call returns."""
    from ouroboros.tool_access_paths import canonical_data_root

    if not retry_key or not record_id:
        return False
    task_id, drive_root = str(getattr(ctx, "task_id", "") or ""), pathlib.Path(ctx.drive_root)
    identity = f"{surface}\0{task_id}\0{retry_key}\0{int(attempt or 0)}\0{record_id}"
    return _owe_left_review({
        "surface": surface, "task_id": task_id, "attempt": int(attempt or 0), "record_id": str(record_id),
        "retry_key": retry_key, "unknown_seats": 0, "left": "unwaited",
        "msg_id": f"{surface}-review-settled:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        "drive_root": str(drive_root), "canonical_root": str(canonical_data_root(ctx))})


def left_review_custody_note(wait: Dict[str, Any]) -> str:
    """What an author restored in another process learns of the review it left."""
    return (f"[Review custody] The {wait.get('surface') or 'change'} review you left for an addressed message is "
            "no longer in this process's live custody (a restart or new execution). Nothing was committed or "
            f"applied. Re-issue {wait.get('reissue') or 'the same unchanged call'} to collect what its reviewers "
            "recorded at $0; a seat without a recorded outcome stays unknown and is never sent again. "
            f"Original reference: {json.dumps(wait.get('review_reference') or {'record_id': wait.get('record_id')})}. "
            f"Selected wake sources: {json.dumps(wait.get('selection') or {})}.")


def wake_restored_left_review(ctx: Any) -> bool:
    """A cold-restored wait whose left review lived in the ended process: its settlement
    notice can no longer come, so the custody note goes to the author's mailbox, which also
    ends that wait at once instead of a park nothing would end. A settled wave's own notice
    already says it; a wait of this process is left alone."""
    wait = getattr(ctx, "_left_review_wait", None)
    if not isinstance(wait, dict) or wait.get("owner") == PROCESS_TOKEN:
        return False
    if wait.get("done") and wait.get("noticed"):
        wait["owner"] = PROCESS_TOKEN
        return False
    from ouroboros.owner_mailbox import write_task_message

    task_id = str(getattr(ctx, "task_id", "") or "")
    note = left_review_custody_note(wait)
    custody_msg_id = f"{wait.get('msg_id') or 'left-review'}:custody-ended"
    written = write_task_message(pathlib.Path(ctx.drive_root), note, task_id,
                                 source_task_id=task_id, provenance="system",
                                 msg_id=custody_msg_id)
    wait.update(owner=PROCESS_TOKEN, done=True, noticed=bool(written), notice=note, msg_id=custody_msg_id,
                wake_reason="review_custody_ended")
    return written


_LEFT_TOOLS = {"commit": "commit_reviewed", "change": "review_change", "result": "review_result", "skill": "review_skill"}
# the mark's writer surface (late_review_marks)
_LEFT_SURFACES = {"commit": "commit_gate", "change": "change", "result": "result", "skill": "skill"}
_SEAT_FACTS = ("model", "status", "operation_id", "operation_state", "late_result_pending", "raw_text", "error",
               "failure_code", "reported_cause", "transport_status", "prompt_ref", "response_ref", "usage",
               "cost_usd", "cost_estimate", "cost_kind", "tokens_in", "tokens_out", "outcome", "outcome_id", "usage_ref")


def _left_mark_id(record: Dict[str, Any]) -> str:
    from ouroboros.late_review_marks import late_review_mark_id  # D06->D15 is lazy-only

    surface, attempt_no = str(record["surface"]), int(record.get("attempt") or 0)
    return late_review_mark_id(_LEFT_SURFACES.get(surface, surface), str(record["task_id"]),
                               f"{surface}-late:{record.get('record_id') or record['retry_key']}:{attempt_no}")


def _owe_left_review(record: Dict[str, Any]) -> bool:
    """The left wave's room duty as one owed row keyed by its mark id, written once, when the author leaves
    (its live controller, this process, may still settle it) and before anything process-local can be lost.
    Nothing rewrites it on settlement: the replay reads settlement from the wave's own records
    (``settle_left_review``), so a refused write cannot strand answers already paid for. The mark that
    discharges it follows the author's task end."""
    from ouroboros.late_review_marks import owe_after_end  # D06->D15 is lazy-only
    from ouroboros.review_operation import controller_identity

    return owe_after_end(record.get("canonical_root") or record["drive_root"], _left_mark_id(record), {
        "left_review": dict(record), "controller": controller_identity()})


def settle_left_review(drive_root: Any, row: Dict[str, Any]) -> str:
    """The owed duty of a left wave whose author's task ended: its mark (``ended_left_review``) once every
    seat of the wave's recorded roster has a terminal outcome on disk or a completed producer nothing live can
    still settle (``_left_roster_settled``), or its controller is proven dead, else ``waiting`` (a live
    controller may still settle a seat that has none)."""
    from ouroboros.review_operation import controller_state

    record = dict(row["left_review"])
    if controller_state(row.get("controller")) != "dead" and not _left_roster_settled(record):
        return "waiting"
    return ended_left_review(record)


def _custody_root(record: Dict[str, Any]) -> pathlib.Path:
    """Where the wave's reviewer assignments and producer outcomes were written: the canonical custody
    root (``review_custody.custody_root``) for every surface, never the child drive of a headless or
    forked author (where an old wave's attempt row may still live, ``_left_attempt``). The reader falls
    back canonical→child."""
    return pathlib.Path(record.get("canonical_root") or record["drive_root"])


def _left_attempt(record: Dict[str, Any]) -> Any:
    """The left wave's own attempt row, the whole roster its author left; ``LookupError`` when it is gone."""
    from ouroboros.review_state import load_state

    surface, task_id, retry_key = str(record["surface"]), str(record["task_id"]), str(record["retry_key"])
    attempt_no = int(record.get("attempt") or 0)
    # Change attempts moved to canonical custody. Still read a pre-move legacy
    # row if this left wave has not yet re-entered through its author's context.
    roots = dict.fromkeys((_left_review_root(record), pathlib.Path(record["drive_root"])))
    attempt = next((row for root in roots for row in reversed(load_state(root).attempts)
                    if row.tool_name == _LEFT_TOOLS.get(surface) and row.task_id == task_id
                    and row.review_retry_key == retry_key and attempt_no in (0, int(row.attempt or 0))), None)
    if attempt is None:
        raise LookupError("its attempt row is not in the review state")
    return attempt


def _left_review_root(record: Dict[str, Any]) -> pathlib.Path:
    """Where the wave's attempt row lives: canonical custody for ``review_change`` waves (change,
    result and skill, ``review_change_custody.review_state_root``), the author's own drive for commit waves."""
    return pathlib.Path((record.get("canonical_root") if record.get("surface") in ("change", "result", "skill") else "")
                        or record["drive_root"])


def _left_rows(attempt: Any) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """``(surface, slot_id, row)`` of every seat the attempt row records, frozen as the gate reconciles them."""
    from ouroboros.review_custody import prepare_frozen_review_reconciliation

    holder = SimpleNamespace()
    prepare_frozen_review_reconciliation(holder, attempt)
    for kind, rows in holder._review_frozen_rows.items():
        for slot_id, row in rows.items():
            yield kind, slot_id, row


def _left_roster_settled(record: Dict[str, Any]) -> bool:
    """Whether every seat of the left wave has a terminal outcome on disk: settled in its attempt row, or the
    producer's own retained CAS under its recorded assignment (``recover_review_producer``). A completed
    producer whose delegated start is still in flight leaves nothing for a live controller to settle once no
    review operation of the task is live (``task_has_live_review_operation``: none can still bind or rejoin
    that start); the final collection then reads it attach-only, where a run still running stays pending.
    Local reads only, never a send or a remote poll; an unreadable record proves no settlement."""
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_operation import (
        recorded_review_assignment,
        recover_review_producer,
        task_has_live_review_operation,
    )

    drive = _custody_root(record)
    try:
        attempt = _left_attempt(record)
        for kind, _slot_id, row in _left_rows(attempt):
            if not _row_is_pending(row):
                continue
            request, slot = recorded_review_assignment(
                drive, task_id=str(record["task_id"]), surface=kind, row=row, retry_key=str(record["retry_key"]),
                root_task_id=str(attempt.root_task_id or ""), review_contract=str(attempt.review_contract_fingerprint or ""))
            actor = recover_review_producer(drive, request, slot, row)
            if actor is None or (actor.operation_state == "in_flight"
                                 and task_has_live_review_operation(drive, str(record["task_id"]))):
                return False
        return True
    except Exception:
        log.debug("left review %s: settlement unreadable; its duty keeps waiting", record.get("retry_key"), exc_info=True)
        return False


def ended_left_review(record: Dict[str, Any]) -> str:
    """Keep a left review's outcome in its author's room once the author's task ended (#1738).

    ``record`` is value-bound (surface, task_id, attempt, record_id, retry_key, unknown_seats, msg_id,
    drive_root, canonical_root; ``left`` is ``unwaited`` for a wave returned with seats still running
    and no addressed message). Its source is the wave's own attempt row (the whole recorded roster);
    every seat still owed there is collected at $0 through the established collector over its recorded
    assignment (``review_operation.collect_recorded_review_actor``), never the live index, today's
    roster or a new send. The wave's own ledger record then settles from those seats
    (``_settle_left_record``): the verdict is a fact of the journal, never a decision, and nothing is
    committed or applied. A seat without a retained outcome stays unknown and counts as failed. An
    unwaited wave whose author already collected it while running needs no mark. Publication discharges
    the owed duty (the same mark id); a failure keeps it owed. Never raises.
    """
    from ouroboros.late_review_marks import publish_late_review  # D06->D15 is lazy-only

    try:
        attempt, seats, gap = _left_seats(record)
        prior_state = _record_state(record)
        settled = None if gap else _settle_left_record(record, seats, final=True)
        if settled and settled.get("state") == "pending":
            return "waiting"
        if record.get("left") == "unwaited" and prior_state == "settled":
            return "collected"  # its author's own re-issue collected it while the task ran
        return publish_late_review(record.get("canonical_root") or record["drive_root"],
                                   _left_review_mark(record, attempt, seats, gap, settled)).status
    except Exception:
        log.warning("left review %s of task %s keeps no mark in view", record.get("retry_key"), record.get("task_id"),
                    exc_info=True)
        return "failed"


def _left_seats(record: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]], str]:
    """``(attempt row, seats, gap)`` of one left wave: every seat of its recorded roster with its own
    recorded outcome, a still-owed seat collected at $0; ``gap`` names an unreadable record."""
    from ouroboros.review_custody import _row_is_pending
    from ouroboros.review_operation import collect_recorded_review_actor

    task_id, retry_key = str(record["task_id"]), str(record["retry_key"])
    drive = _custody_root(record)
    attempt, seats, gap = None, [], ""
    try:
        attempt = _left_attempt(record)
        for kind, slot_id, row in _left_rows(attempt):
            actor = (collect_recorded_review_actor(
                drive, task_id=task_id, surface=kind, row=row, retry_key=retry_key,
                root_task_id=str(attempt.root_task_id or ""),
                review_contract=str(attempt.review_contract_fingerprint or "")) if _row_is_pending(row) else row)
            parts = (actor.get("outcome") or {}).get("parts") or row.get("parts") or (row.get("review_plan") or {}).get("parts") or ()
            seats.append({"surface": kind, "slot_id": slot_id, "parts": list(parts),
                          **{key: actor.get(key) for key in _SEAT_FACTS},
                          "model": actor.get("model") or actor.get("model_id"),
                          "raw_text": actor.get("raw_text") or actor.get("text") or ""})
    except Exception as exc:  # an unreadable record: every seat stays unknown, said in the mark itself
        gap = f"{type(exc).__name__}: {exc}"
    return attempt, seats, gap


def _seat_unknown(seat: Dict[str, Any]) -> bool:
    """No retained outcome: in flight, unsent or lost custody (never sent again)."""
    return (str(seat.get("operation_state") or "") in {"pending_dispatch", "in_flight", "custody_lost"}
            or bool(seat.get("late_result_pending")))


def _record_state(record: Dict[str, Any]) -> str:
    """The wave's own ledger record state (``pending``/``settled``), "" when it names none or is unreadable."""
    from ouroboros import review_ledger as ledger

    record_id = str(record.get("record_id") or "")
    try:
        payload = ledger.load_record(record.get("canonical_root") or record["drive_root"], record_id) if record_id else None
    except ValueError:
        payload = None
    return str((payload or {}).get("state") or "")


def _settle_left_record(record: Dict[str, Any], seats: List[Dict[str, Any]], *, final: bool) -> Optional[Dict[str, Any]]:
    """Settle the wave's own ledger record from the seats it recorded: each answer is read by the gate's own
    parser (``triad_review.parse_seat_answers``) and the verdict reduced by the ledger
    (``review_ledger.settle_from_collected``). A seat in flight or unsent keeps the record pending. A lost seat
    is a failed seat with its own recorded cause, except an unbound start that its author's own collection may
    still replay under its original key (``START_UNBOUND``): while that author lives it keeps the record
    pending. ``final`` (its author ended and no live drain will collect it) fails every lost seat and names
    that cause. ``None`` when the wave names no record or it cannot be settled; never raises."""
    from ouroboros import review_ledger as ledger
    from ouroboros.triad_review import parse_seat_answers

    record_id = str(record.get("record_id") or "")
    if not record_id:
        return None
    results, row_parts = [], {}
    for seat in seats:
        slot = str(seat.get("slot_id") or "")
        unknown = _seat_unknown(seat)
        lost = str(seat.get("operation_state") or "") == "custody_lost"
        if unknown and (not lost or (not final and (seat.get("usage") or {}).get("custody_state") == START_UNBOUND)):
            continue  # its answer may still arrive, or its live author's collection may still bind its start
        row_parts[slot] = tuple(seat.get("parts") or ("change",))
        result = {key: seat[key] for key in _SEAT_FACTS if seat.get(key) is not None}
        usage = seat.get("usage") or {}
        cost = next((v for v in (seat.get("cost_usd"), seat.get("cost_estimate"), usage.get("cost")) if v is not None), None)
        result.update(slot_id=slot, cost_estimate=cost,
                      tokens_in=seat.get("tokens_in") or usage.get("prompt_tokens", 0),
                      tokens_out=seat.get("tokens_out") or usage.get("completion_tokens", 0))
        raw = str(seat.get("raw_text") or "")
        if unknown:
            cause = (AUTHOR_ENDED_CAUSE if final
                     else seat.get("reported_cause") or seat.get("error") or "no retained outcome")
            result.update(verdict="ERROR", text="", operation_state="custody_lost",
                          failure_code=str(seat.get("failure_code") or "review_custody_lost"),
                          reported_cause=str(cause))
        elif "not_dispatched" in (str(seat.get("operation_state") or ""), str(seat.get("status") or "")):
            result.update(operation_state="not_dispatched", text="")
        elif not raw.strip() or str(seat.get("status") or "") == "error":
            result.update(verdict="ERROR", text=str(seat.get("error") or "no answer"),
                          reported_cause=" ".join(str(seat.get("error") or "no answer").split())[:400])
        else:
            result["text"] = raw
        results.append(result)
    try:
        parsed = parse_seat_answers({"results": results}, row_parts)
        return ledger.settle_from_collected(record.get("canonical_root") or record["drive_root"], record_id,
                                            [actor.to_dict() for actor in parsed.actor_records])
    except Exception:
        log.warning("left review %s: its record %s stays as it was", record.get("retry_key"), record_id, exc_info=True)
        return None


def _settled_aggregate(record: Dict[str, Any]) -> str:
    """Settle a left wave's record once every seat settled in this process; its verdict word, "" when none."""
    attempt, seats, gap = _left_seats(record)
    settled = None if gap else _settle_left_record(record, seats, final=False)
    if not settled or settled.get("state") != "settled":
        return ""
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    settle_collected_attempt(_left_review_root(record), attempt, seats, settled)
    record["revision"] = settled["revision"]
    return str((settled.get("verdict") or {}).get("aggregate") or "")


def _left_review_mark(record: Dict[str, Any], attempt: Any, seats: List[Dict[str, Any]], gap: str,
                      settled: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The late-review record of one left wave, composed from its attempt row and retained source."""
    from ouroboros.artifacts import store_actor_source_bytes

    surface, task_id, retry_key = str(record["surface"]), str(record["task_id"]), str(record["retry_key"])
    drive, attempt_no = pathlib.Path(record["drive_root"]), int(record.get("attempt") or 0)
    source = {"kind": "left_review_settlement", "gap": gap, "seats": seats,
              **{key: record.get(key) for key in ("surface", "task_id", "attempt", "record_id", "retry_key",
                                                  "unknown_seats", "msg_id", "left")},
              "attempt_row": {key: getattr(attempt, key, None) for key in (
                  "repo_key", "tool_name", "attempt", "review_record_id", "pre_review_fingerprint", "phase",
                  "status", "block_reason")} if attempt is not None else None}
    ref = store_actor_source_bytes(
        record.get("canonical_root") or drive, task_id, category="context_checkpoints",
        source_id=f"left-review-{surface}-{hashlib.sha256(retry_key.encode()).hexdigest()[:16]}",
        data=json.dumps(source, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"), extension="json")
    named = f"attempt #{attempt_no}" if attempt_no else f"record {record.get('record_id') or retry_key[-12:]}"
    unknown = [seat for seat in seats if _seat_words(seat).startswith("outcome unknown")]
    how = ("that its author stopped waiting for" if record.get("left") == "unwaited"
           else "that its author left for an addressed message")
    verdict = str(((settled or {}).get("verdict") or {}).get("aggregate") or "")
    # Lifecycle facts only: the author's task ended; whether a seat answered is each seat's own line.
    head = (f"The paid {surface} review ({named}) {how}, kept here because its author's task ended. "
            + (f"Its record now reads {verdict}; nothing was committed or applied; "
               if verdict and (settled or {}).get("state") == "settled"
               else "No panel verdict was formed for it and nothing was committed or applied; ")
            + (f"its record could not be read ({gap}), so every seat's outcome stays unknown."
               if gap else "each seat's recorded state follows."))
    if unknown:
        head += f" {len(unknown)} seat(s) have no retained outcome and stay unknown; nothing is sent again."
    return {"surface": _LEFT_SURFACES.get(surface, surface), "task_id": task_id, "source_ref": ref,
            "publication_id": f"{surface}-late:{record.get('record_id') or retry_key}:{attempt_no}",
            "lines": [head, *(f"- {seat.get('model') or 'reviewer'} (seat {seat['slot_id']}): {_seat_words(seat)}"
                              for seat in seats)],
            "subject": {"retry_key": retry_key, "attempt": attempt_no, "record_id": str(record.get("record_id") or ""),
                        "reviewed_fingerprint": str(getattr(attempt, "pre_review_fingerprint", "") or "")}}


def _seat_words(seat: Dict[str, Any]) -> str:
    """One seat's own outcome in words, per question it was asked (the gate's own seat parser), never a verdict."""
    from ouroboros.triad_review import parse_two_part_answer

    state, status = str(seat.get("operation_state") or ""), str(seat.get("status") or "")
    # Composed once no live drain of its author can collect it: a seat with no terminal record yet is unknown.
    if state in {"pending_dispatch", "in_flight", "custody_lost"} or seat.get("late_result_pending"):
        return "outcome unknown (no retained outcome; never sent again)"
    if state == "not_dispatched" or status == "not_dispatched":
        return "not sent"
    raw = str(seat.get("raw_text") or "")
    if not raw.strip():
        return "failed: " + (" ".join(str(seat.get("error") or "").split()) or "no answer")
    parts = tuple(seat.get("parts") or ())
    if seat.get("outcome"):
        answers = seat["outcome"].get("answers")
    elif parts == ("result",):  # a result seat answers the findings array (contract A), not the two-part object
        from ouroboros.triad_review import parse_seat_answers

        answers = (parse_seat_answers({"results": [{**seat, "text": raw}]}, {str(seat.get("slot_id") or ""): parts})
                   .actor_records[0].answers or None)
    else:
        answers = parse_two_part_answer(raw, parts) if parts else None
    if not answers:
        return "answered; its full answer is in the retained source"
    return "; ".join(f"{part} {answer.get('verdict') or answer.get('status')}"
                     + (f" ({answer['critical']} critical)" if answer.get("critical") else "")
                     for part, answer in answers.items())
