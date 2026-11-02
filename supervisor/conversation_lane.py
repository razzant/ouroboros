"""The conversation lane: pool slots that carry the mind's foci, their reserved spare, and admission.

Owner Main/room turns, consciousness wakes, Presence bodies and the auto-resume
turn execute in worker processes like any task, but they are admitted by routing
(never through PENDING), hold no project lease and sit outside ``MAX_WORKERS``,
the crash-storm capture and the readiness-exhaustion fence. Their slots live in
the same ``WORKERS`` table as project workers with ``lane="conversation"``, so
health, custody, Panic and quiescence retirement iterate every slot alike; the
three project fences filter by lane.

Admission (``admit_turn``, under Q and the repo-writer gate lock the caller holds)
writes the RUNNING row first — ``execution_owner.kind="direct"`` is the origin,
``lane="conversation"`` and ``executor`` the carrier — persists it, and hands the
turn to the ready spare when one is idle; otherwise the row awaits a carrier and
``reconcile_conversation_lane`` dispatches it on the tick that has one. The carrier
runs the turn through the same ``handle_task`` path as any task; its frames carry
chat and lineage by value (``log_addressing``), its phases reach the census as
``direct_turn_phase`` frames, and its owner waits use the pooled protocol.

Capacity (owner decision 4A): the pool's one quiescent spare is the lane's. This
module keeps exactly one idle READY carrier warm beyond the turns still awaiting
one, replenishes it when a turn claims it, retires acknowledged surplus through
the existing quiescence probe and never spawns while the repo-writer gate is
closed (a fresh carrier would import a conflicted tree; admission then takes the
disclosed degraded executor). No number bounds concurrent carriers: the bound is
spawn success, and the vitals show the carriers and each one's own RSS, reported
on its turn's heartbeat (``carrier_vitals``). Called from
``reconcile_worker_capacity`` under the lifecycle serializer, before the project
pool's own growth.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
from supervisor.queue import _queue_lock

log = logging.getLogger(__name__)

# The lane keeps its spare only for a STARTED pool generation: ``spawn_workers`` arms it for the
# table it filled and a whole-table stop (Panic, Restart, a managed update) disarms it, so a
# stopped pool never grows a carrier behind its stop and a table built by hand starts no child.
_ARMED_TABLE: Any = None
PREPARE_KEY = "_conversation_prepare"  # the carrier's first-step facts, consumed by prepare_conversation_task
# The last carrier spawn this generation could not make (the exception's name), cleared by a
# carrier that started: the physical fact the degraded executor reads (owner decision 2A).
_SPAWN_REFUSED = [""]


def spawn_refusal() -> str:
    return _SPAWN_REFUSED[0]


def arm(table: Any) -> None:
    """Arm replenishment for ``table`` (the live ``WORKERS`` dict); ``None`` disarms.

    A generation starts with no spawn refusal on record: the fact belongs to the generation that observed it."""
    global _ARMED_TABLE
    _ARMED_TABLE = id(table) if table is not None else None
    _SPAWN_REFUSED[0] = ""


def armed() -> bool:
    return _ARMED_TABLE is not None and _ARMED_TABLE == id(_pool().WORKERS)


def _pool():
    """The parent module, read at call time (the owner-approved D18/D33 mechanical exception)."""
    from supervisor import workers

    return workers


def conversation_row(meta: Any) -> bool:
    """A RUNNING row the conversation lane admitted (its carrier may be awaited, a worker or inline)."""
    return isinstance(meta, dict) and meta.get("lane") == LANE_CONVERSATION


def lane_carriers(pool: Any = None) -> list:
    """Every conversation carrier that is a custodied worker process."""
    from supervisor.worker_pool_lifecycle import is_worker_slot, lane_slots

    return [slot for slot in lane_slots(pool) if is_worker_slot(slot)]


def idle_carriers() -> list:
    """Under Q: the READY idle lane carriers nothing else holds, in slot order."""
    from supervisor.worker_pool_lifecycle import worker_ready

    return [slot for slot in lane_carriers()
            if slot.busy_task_id is None and worker_ready(slot) and not slot.retiring
            and getattr(slot, "active_capacity", True) and not getattr(slot, "resume_hold", "")]


def carrier_rss() -> Dict[str, Any]:
    """In a carrier: its own resident size now, measured in-process (``update_generation_facts._rss``:
    psutil's current figure where it ships, else the lifetime peak, labelled); {} when unobserved."""
    try:
        from ouroboros.update_generation_facts import _rss

        rss, measure = _rss()
    except Exception:
        return {}
    return {"rss_bytes": int(rss), "rss_measure": measure} if rss is not None else {}


def note_carrier_rss(slot: Any, sample: Any) -> None:
    """In the server: a lane carrier's own report (``carrier_rss``, riding its turn's heartbeat) stays on its slot."""
    if (isinstance(sample, dict) and type(sample.get("rss_bytes")) is int
            and getattr(slot, "lane", "") == LANE_CONVERSATION):
        slot.rss = {"rss_bytes": sample["rss_bytes"], "rss_measure": str(sample.get("rss_measure") or "")}


def carrier_vitals() -> Dict[str, Any]:
    """The lane's carriers for the vitals (owner decision 4A: no number bounds them; the vitals show their
    count and RSS): how many, how many idle, and each one's last own report in MiB, in slot order (None
    before its first turn's heartbeat). A section of the per-minute ``supervisor_load`` row; memory only."""
    carriers = lane_carriers()
    reports = [getattr(slot, "rss", None) or {} for slot in carriers]
    return {"carriers": len(carriers), "idle": sum(1 for slot in carriers if slot.busy_task_id is None),
            "rss_mib": [report["rss_bytes"] // 1048576 if report else None for report in reports],
            "rss_measure": sorted({report["rss_measure"] for report in reports if report})}


def owner_turn(task: Dict[str, Any]) -> bool:
    """A turn the reserved spare is for: the owner's Main or room turn (and its auto-resume) and a Presence
    turn a correspondent waits on (FINAL_PLAN §3: the spare is the conversation lane's). A consciousness wake
    is not (owner decision 4A, packet A6): it takes another idle carrier or starts its own."""
    from ouroboros.consciousness_authority import wake_cause

    return not wake_cause(task.get("metadata"))


def ready_idle_carrier(*, for_owner: bool = True) -> Any:
    """Under Q: a READY idle lane carrier for this turn, or None.

    The owner's turn takes any idle carrier, the reserved spare included. Another focus (a wake,
    a Presence turn) takes an idle carrier only while one remains for the owner: the last idle
    carrier is the spare and stays, so the owner's next turn never waits for a boot behind a wake.
    """
    idle = idle_carriers()
    if not idle:
        return None
    if for_owner or len(idle) > 1:
        return idle[0]
    return None


def awaiting_turns() -> List[Dict[str, Any]]:
    """Under Q: the lane's RUNNING rows no carrier holds yet, oldest first."""
    rows = [meta for meta in _pool().RUNNING.values()
            if conversation_row(meta) and meta.get("worker_id") is None
            and not isinstance(meta.get("_terminal_file_recovery"), dict)]
    return sorted(rows, key=lambda meta: float(meta.get("started_at") or 0.0))


def _held_by_a_stop(rows: List[Dict[str, Any]]) -> set:
    """The awaiting rows a Stop owns: one read of the durable cancel-intent projection (custody settles them).

    Read only while turns await a carrier; an unreadable projection holds every row for this pass (an unknown
    is never a start), a quarantined intent row holds its own."""
    ids = {str(meta["task"].get("id") or "") for meta in rows}
    if not ids:
        return set()
    from ouroboros.cancel_intents import read_active_intents

    try:
        intents, unseen = read_active_intents(_pool().DRIVE_ROOT, strict=True)
    except Exception:
        log.warning("cancel intents unreadable: the awaiting conversation turns wait a pass", exc_info=True)
        return ids
    return ids & (set(intents) | set(unseen or ()))


def _unpublished(rows: List[Dict[str, Any]]) -> set:
    """The rows whose admission barrier failed: one barrier now covers them all, else they stay held this pass."""
    pending = [meta for meta in rows if meta.get("_unpublished")]
    if not pending:
        return set()
    from supervisor import queue

    if queue.publish_queue_snapshot(reason="conversation_turn_admitted") is False:
        return {str(meta["task"].get("id") or "") for meta in pending}
    for meta in pending:
        meta.pop("_unpublished", None)
    return set()


def take_awaiting_turn(task_id: str) -> Any:
    """Under Q: a Stop's custody takes a lane turn no carrier holds yet out of RUNNING, as it takes a PENDING row.

    No process exists, so custody settles it typed ``cancelled`` with its pre-start cost and publishes its
    ``task_done``; the turn never reaches a carrier. None for a held row or no lane row (the kill path's)."""
    meta = _pool().RUNNING.get(str(task_id))
    if (not conversation_row(meta) or meta.get("worker_id") is not None
            or isinstance(meta.get("_terminal_file_recovery"), dict)):
        return None
    return _pool().RUNNING.pop(str(task_id))


def live_turn_count() -> int:
    """Under Q: the lane's live rows plus the inline executors' registered actors (the census at entry)."""
    from supervisor.active_activity import get_direct_activity_registry

    live = {tid for tid, meta in _pool().RUNNING.items() if conversation_row(meta)}
    live.update(entry.activity_id for entry in get_direct_activity_registry().actors())
    return len(live)


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------

def admit_turn(task: Dict[str, Any], *, chat_id: int, client_message_id: str = "", quiet: bool = False,
               reply_expected: bool = False, hold_carrier: bool = False) -> Dict[str, Any]:
    """Under Q and the repo-writer gate lock (the caller's): the RUNNING row, then the carrier.

    Durable intent before effects: the row is written and published before the
    turn reaches any process. ``task`` is the admitted seed (id, chat, text, origin
    facts); the carrier layers attachments, Project binding, authority, contract
    and tool profile on it as its first step (``worker_chat_lane.prepare_conversation_task``).
    Returns the RUNNING meta; ``worker_id`` is None while the row awaits a carrier.
    ``hold_carrier``: the caller binds the row's executor itself (an inline executor).
    """
    from ouroboros.task_status import execution_owner_record
    from supervisor import queue

    pool = _pool()
    now = time.time()
    task_id = str(task["id"])
    task.setdefault("root_task_id", task_id)
    origin = "presence" if task.get("_presence_turn") else "direct"
    task["_execution_owner"] = {**execution_owner_record(pool.DRIVE_ROOT, task, origin),
                                "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}
    task[PREPARE_KEY] = {
        "admitted_at": now, "quiet": bool(quiet), "reply_expected": bool(reply_expected),
        "client_message_id": str(client_message_id or ""),
        "census": {"direct_turns_at_entry": live_turn_count()},
    }
    # The host-only fresh-identity mark rides the dispatched copy, never the published row.
    row_task = {key: value for key, value in task.items() if key != "_fresh_identity"}
    meta = _awaiting_row(row_task, attempt=1, client_message_id=client_message_id, reply_expected=reply_expected,
                         fresh_identity=task.get("_fresh_identity"))
    pool.RUNNING[task_id] = meta
    # The admission barrier, not a mark: the row is on disk before any carrier can hold the turn. A failed
    # publication keeps the row visibly unstarted (held from every carrier until a later barrier lands).
    if queue.publish_queue_snapshot(reason="conversation_turn_admitted") is False:
        meta["_unpublished"] = True
        log.warning("Conversation turn %s admitted but its row is not on disk yet: no carrier holds it", task_id)
        return meta
    slot = None if hold_carrier else ready_idle_carrier(for_owner=owner_turn(task))
    if slot is not None:
        hand_to_carrier(meta, slot)
    return meta


def _awaiting_row(task: Dict[str, Any], *, attempt: int, client_message_id: Any, reply_expected: Any,
                  fresh_identity: Any) -> Dict[str, Any]:
    """The RUNNING meta of a lane turn no carrier holds yet (an admission, or a dead carrier's retry)."""
    now = time.time()
    return {"task": task, "worker_id": None, "started_at": now, "last_heartbeat_at": now, "last_progress_at": now,
            "budget_paused_sec": 0.0, "soft_sent": False, "attempt": int(attempt), "lane": LANE_CONVERSATION,
            "client_message_id": str(client_message_id or ""), "reply_expected": bool(reply_expected),
            "fresh_identity": bool(fresh_identity)}


def readmit_after_crash(task: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """Under Q: a dead carrier's retry (the same id, its next attempt, saved work attached) returns to the lane.

    Never PENDING: the conversation's recovery is independent of the project pool's capacity and crash
    fuse (FINAL_PLAN §1). The row awaits a carrier again with the turn's own lane facts, published before
    any carrier can hold it; a ready idle carrier takes it now, the disclosed degraded executor on a physical
    fact, otherwise the lane's reconcile hands it the next carrier (the dead slot's replacement among them).
    """
    from supervisor import queue

    task["_execution_owner"] = {**dict(task.get("_execution_owner") or {}), "lane": LANE_CONVERSATION,
                                "executor": EXECUTOR_WORKER}
    meta = _awaiting_row(task, attempt=int(task.get("_attempt") or 1), client_message_id=prior.get("client_message_id"),
                         reply_expected=prior.get("reply_expected"), fresh_identity=prior.get("fresh_identity"))
    _pool().RUNNING[str(task["id"])] = meta
    queue.publish_queue_snapshot(reason="conversation_turn_retried")
    dispatch_awaiting_turns()
    return meta


def hand_to_carrier(meta: Dict[str, Any], slot: Any) -> bool:
    """Under Q: put the turn on the carrier's queue, then bind the row and slot to each other.

    The put is the handoff boundary (as ``worker_assignment._claim_worker_launch``):
    from there the carrier may hold the turn, so the binding and its snapshot follow
    it. A put that raises leaves the row awaiting the next ready carrier.
    """
    from supervisor import queue

    task = meta["task"]
    dispatched = dict(task)
    if meta.get("fresh_identity"):
        dispatched["_fresh_identity"] = True
    slot.quiescence_token = ""
    slot.quiescence_confirmed = False
    try:
        slot.in_q.put(dispatched)
    except Exception:
        log.exception("Conversation turn %s could not reach carrier %s; it awaits the next one",
                      task.get("id"), getattr(slot, "wid", "?"))
        return False
    slot.busy_task_id = str(task["id"])
    meta["worker_id"] = slot.wid
    meta["dispatched_at"] = time.time()
    queue.persist_queue_snapshot(reason="conversation_turn_dispatched")
    return True


def dispatch_awaiting_turns() -> int:
    """Under Q: hand every awaiting turn to a ready idle carrier, oldest first; returns how many moved.

    When no carrier can start at all (``inline_carrier.refused_carrier_reason``: a
    physical fact, never "busy"), the turns still waiting take the disclosed degraded
    executor instead of waiting for a carrier that will not come.
    """
    from supervisor.inline_carrier import EXECUTOR_INLINE_DEGRADED, refused_carrier_reason, start_inline_turn

    waiting = awaiting_turns()
    # A Stop's custody settles its rows; a row not yet on disk waits for its publication: no carrier takes them.
    held = _held_by_a_stop(waiting) | _unpublished(waiting)
    moved = 0
    for meta in waiting:  # a wake waiting for the spare never holds up the owner's turn behind it
        if str(meta["task"].get("id") or "") in held:
            continue
        slot = ready_idle_carrier(for_owner=owner_turn(meta["task"]))
        if slot is not None and hand_to_carrier(meta, slot):
            moved += 1
    waiting = [meta for meta in awaiting_turns() if str(meta["task"].get("id") or "") not in held]
    if waiting and not any(w.booting and w.busy_task_id is None for w in lane_carriers()):
        reason = refused_carrier_reason()
        if reason:
            for meta in waiting:
                try:
                    start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason=reason)
                    moved += 1
                except Exception:
                    log.exception("Degraded executor could not start for %s", meta["task"].get("id"))
    return moved


# ---------------------------------------------------------------------------
# capacity
# ---------------------------------------------------------------------------

def reconcile_conversation_lane() -> None:
    """Dispatch awaiting turns; keep one ready idle carrier beyond them; retire surplus.

    Lifecycle serializer held by the caller; Q is taken only for the census and
    the dispatch. Nothing spawns in a marker window (the admission discloses the
    degraded executor there). A carrier that exhausted its readiness attempts is
    parked until Restart like a project slot, and no second carrier starts
    beside it: an awaiting turn then takes the degraded executor too.
    """
    from supervisor.worker_pool_lifecycle import (
        _spawn_worker_slot, request_worker_quiescence, retire_quiescent_worker, worker_ready,
    )

    pool = _pool()
    with _queue_lock:
        if not armed() or not pool.WORKERS:
            return
        # A marker window forbids a spawn, never a waiting turn's carrier (a spare that finished booting)
        # or its degraded executor: the turns are looked at first, the window decides growth below.
        dispatch_awaiting_turns()
        if pool.repo_writer_admission_closed():
            return
        carriers = lane_carriers()
        if any(getattr(slot, "readiness_exhausted", False) for slot in carriers):
            return
        idle = [w for w in carriers if w.busy_task_id is None and worker_ready(w) and not w.retiring]
        booting = [w for w in carriers if w.busy_task_id is None and w.booting and not w.retiring]
        wanted = 1 + len(awaiting_turns())  # the reserved spare, plus one carrier per turn still waiting
        missing = max(0, wanted - len(idle) - len(booting))
        if len(idle) > 1:
            # Ask every idle carrier: a resident with late duties is not the spare. Only
            # acknowledged surplus beyond the one spare retires, through the reaper.
            for slot in idle:
                request_worker_quiescence(slot)
            quiet = [w for w in idle if w.quiescence_confirmed]
            for slot in quiet[1:]:
                retire_quiescent_worker(slot)
        next_id = max(pool.WORKERS, default=-1) + 1
    for wid in range(next_id, next_id + missing):
        try:
            if not _spawn_worker_slot(wid, lane=LANE_CONVERSATION):
                break
            _SPAWN_REFUSED[0] = ""
        except Exception as exc:
            _SPAWN_REFUSED[0] = f"carrier_spawn_refused:{type(exc).__name__}"
            log.warning("The conversation lane's carrier could not be started", exc_info=True)
            break


# ---------------------------------------------------------------------------
# census and frames
# ---------------------------------------------------------------------------

def replenish_after_claim() -> None:
    """A turn was admitted: reconcile the lane now, not at the next tick (owner decision 4A) — the spare a
    claim took is replaced, a waiting turn's own carrier is started.

    Outside Q (the caller's admission transaction has ended). The lifecycle serializer is taken
    without waiting: a tick or a stop that holds it reconciles the lane itself.
    """
    from supervisor.worker_pool_lifecycle import _WORKER_LIFECYCLE_LOCK

    if not armed() or not _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        return
    try:
        reconcile_conversation_lane()
    except Exception:
        log.exception("Conversation lane replenishment after a claim failed")
    finally:
        _WORKER_LIFECYCLE_LOCK.release()


def lane_turn_rows() -> List[Dict[str, Any]]:
    """``active_direct_turns`` rows for the lane's RUNNING turns (the registry's shape), by value.

    ``phase`` is ``preparing`` until a carrier holds the turn, then ``working``;
    ``turn_phase`` follows the carrier's own start facts (``direct_turn_phase``
    frames) and ``model_activity`` its observed call, both frame-fed.
    """
    rows: List[Dict[str, Any]] = []
    with _queue_lock:
        snapshot = [(tid, meta) for tid, meta in _pool().RUNNING.items() if conversation_row(meta)]
        for tid, meta in snapshot:
            task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            owner = task.get("_execution_owner") if isinstance(task.get("_execution_owner"), dict) else {}
            attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
            row: Dict[str, Any] = {
                "activity_id": tid,
                "chat_id": int(task.get("chat_id") or 0),
                "project_id": str(task.get("project_id") or ""),
                "client_message_id": str(meta.get("client_message_id") or metadata.get("client_message_id") or ""),
                # The census's one shape (``ActiveChatActivity``): a Host-admitted Presence turn the lane carries is
                # ``presence`` there, as the in-process author census lists one (#2566); the owner's turns stay direct.
                "kind": "presence" if task.get("_presence_turn") else "direct_chat",
                "phase": "working" if meta.get("worker_id") is not None else "preparing",
                "started_at": float(meta.get("started_at") or 0.0),
                "task_attempt": attempt,
                "executor": str(owner.get("executor") or EXECUTOR_WORKER),
                "carrier": "awaiting" if meta.get("worker_id") is None else "worker",
            }
            if meta.get("turn_phase"):
                row["turn_phase"] = str(meta["turn_phase"])
            active = meta.get("active_llm_call")
            if isinstance(active, dict) and active.get("task_attempt") == attempt and active.get("model_activity"):
                row["model_activity"] = dict(active["model_activity"])
            rows.append(row)
    return rows


def handle_direct_turn_phase(evt: Dict[str, Any], ctx: Any) -> None:
    """A carrier's start fact (``TURN_PHASES``) lands on its RUNNING row for the census."""
    from supervisor.active_activity import TURN_PHASES

    task_id, phase = str(evt.get("task_id") or ""), str(evt.get("phase") or "")
    if not task_id or phase not in TURN_PHASES:
        return
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        if not isinstance(meta, dict):
            return
        try:
            if int(evt.get("task_attempt") or 1) != int(meta.get("attempt") or 1):
                return
        except (TypeError, ValueError):
            return
        current = str(meta.get("turn_phase") or TURN_PHASES[0])
        if TURN_PHASES.index(phase) >= TURN_PHASES.index(current):  # facts only move forward
            meta["turn_phase"] = phase


BOUND_FACTS = ("_project_admission", "project_id")  # the carrier's first-step facts a row rebuilds from


def handle_direct_turn_bound(evt: Dict[str, Any], ctx: Any) -> None:
    """The carrier bound its turn (the Project basis of a room turn): the RUNNING row's task learns the
    facts everything rebuilt from the row needs — the snapshot's running rows, the pooled park, restore."""
    from supervisor.queue_snapshot import persist_queue_snapshot

    task_id, facts = str(evt.get("task_id") or ""), evt.get("facts")
    if not task_id or not isinstance(facts, dict):
        return
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        if not isinstance(meta, dict) or not isinstance(meta.get("task"), dict):
            return
        try:
            if int(evt.get("task_attempt") or 1) != int(meta.get("attempt") or 1):
                return
        except (TypeError, ValueError):
            return
        for key in BOUND_FACTS:
            if key in facts and facts[key] is not None:
                meta["task"][key] = facts[key]
        persist_queue_snapshot(reason="conversation_turn_bound")


def handle_task_named(evt: Dict[str, Any], ctx: Any) -> None:
    """A carrier coined a title (its namer or a wake's cause): the bridge broadcasts it as the server's own did."""
    task_id, name = str(evt.get("task_id") or ""), str(evt.get("suggested_name") or "")
    if not task_id or not name:
        return
    try:
        ctx.bridge.broadcast({"type": "task_named", "task_id": task_id, "suggested_name": name})
    except Exception:
        log.debug("task_named broadcast failed for %s", task_id, exc_info=True)
