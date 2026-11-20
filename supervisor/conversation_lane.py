"""The conversation lane: pool slots that carry the mind's foci, their reserved spare, and admission.

Owner Main/room turns, consciousness wakes, Presence bodies and the auto-resume
turn execute in worker processes like any task, but they are admitted by routing
(never through PENDING), hold no project lease and sit outside the owner's cap
(``OUROBOROS_MAX_WORKERS``), the memory admission of project processes and the
readiness-exhaustion fence. Their slots live in the same ``WORKERS``
table as project workers with ``lane="conversation"``, so health, custody, Panic
and quiescence retirement iterate every slot alike; the project fences filter by lane.
My human's own turn that a stop door, a Resume or its own sleep returns to the queue
(``resumes_on_lane``) is assigned from PENDING like queued work, onto this lane's carrier.

Admission (``admit_turn``, under Q and the repo-writer gate lock the caller holds)
writes the RUNNING row first — ``execution_owner.kind="direct"`` is the origin,
``lane="conversation"`` and ``executor`` the carrier — and marks the queue; the row
is ``_unpublished`` until its admission barrier lands. Q spans memory only: the
caller releases its admission transaction, then ``dispatch_admitted_turn`` runs the
barrier outside Q (``publish_awaiting_rows``) and, under Q again, hands the turn to
the ready spare when one is idle, else to the disclosed degraded executor on a
physical fact; otherwise the row awaits a carrier and ``reconcile_conversation_lane``
dispatches it on the tick that has one. No carrier holds a row that is not on disk.
A row admitted ``reserved`` (a live call's: ``inline_carrier.run_inline_turn``) awaits no
carrier at all: it is ``_inline_bind`` from its admission until ``start_inline_turn``
binds it, and no dispatch door passes it to a carrier or the degraded executor meanwhile.
The carrier runs the turn through the same ``handle_task`` path as any task; its
frames carry chat and lineage by value (``log_addressing``), its phases reach the
census as ``direct_turn_phase`` frames, and its owner waits use the pooled protocol.

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
import pathlib
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.task_carrier import EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, LANE_CONVERSATION
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
# The returned turns the last assignment pass found dispatchable with no idle carrier (``note_returns_waiting``).
_RETURNS_WAITING = [0]


def spawn_refusal() -> str:
    return _SPAWN_REFUSED[0]


def arm(table: Any) -> None:
    """Arm replenishment for ``table`` (the live ``WORKERS`` dict); ``None`` disarms.

    A generation starts with no spawn refusal or waiting return on record: both belong to the generation
    that observed them."""
    global _ARMED_TABLE
    _ARMED_TABLE = id(table) if table is not None else None
    _SPAWN_REFUSED[0] = ""
    _RETURNS_WAITING[0] = 0


def armed() -> bool:
    return _ARMED_TABLE is not None and _ARMED_TABLE == id(_pool().WORKERS)


def inline_resume_available(drive_root: Any) -> bool:
    """Whether a Presence thread in this process can hand its pause to this root's running pool."""
    from supervisor.queue import owns_live_queue

    return armed() and owns_live_queue(drive_root)


def _pool():
    """The parent module, read at call time (the owner-approved D18/D33 mechanical exception)."""
    from supervisor import workers

    return workers


def conversation_row(meta: Any) -> bool:
    """A RUNNING row the conversation lane admitted (its carrier may be awaited, a worker or inline)."""
    return isinstance(meta, dict) and meta.get("lane") == LANE_CONVERSATION


def host_presence_row(meta: Any) -> bool:
    """A Host owns its reserved or thread-bound Presence turn, including after a running pool fails to refill."""
    if not conversation_row(meta) or not (meta.get("task") or {}).get("_presence_run"):
        return False
    wid = meta.get("worker_id")
    return bool(meta.get("_inline_bind")) or (isinstance(wid, int) and wid < -1)


def host_inline_meta(task_id: Any) -> Optional[Dict[str, Any]]:
    """Under Q: the lane's RUNNING meta of ``task_id`` when its Host owns the turn inline, else None.

    Reads the lane's own table — the map ``admit_turn`` writes. The queue facade's ``RUNNING`` is that
    same object only once ``workers.init`` bound it (``init_queue_refs``); a standalone Host answers the
    Pause door with the two maps distinct, so the Pause contract reads the row that actually exists."""
    meta = _pool().RUNNING.get(str(task_id or ""))
    return meta if host_presence_row(meta) else None


def host_inline_drive_roots(default_root: Any = None) -> Dict[str, str]:
    """Under Q: every RUNNING row its Host owns inline (``host_presence_row``) mapped to the drive root
    the row names (``default_root`` when it names none): the census's Pause-availability domain, from the
    same authoritative table as ``host_inline_meta``."""
    return {str(tid): str((meta.get("task") or {}).get("drive_root") or default_root or "")
            for tid, meta in _pool().RUNNING.items() if host_presence_row(meta)}


def lane_carriers(pool: Any = None) -> list:
    """Every conversation carrier that is a custodied worker process."""
    from supervisor.worker_pool_lifecycle import is_worker_slot, lane_slots

    return [slot for slot in lane_slots(pool) if is_worker_slot(slot)]


def idle_carriers() -> list:
    """Under Q: the READY idle lane carriers nothing else holds, in slot order."""
    from supervisor.worker_pool_lifecycle import worker_ready

    return [slot for slot in lane_carriers()
            if slot.busy_task_id is None and worker_ready(slot) and not slot.retiring
            and getattr(slot, "active_capacity", True)]


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
    """In the server: a worker's own report (``carrier_rss``, riding every task's heartbeat, both lanes) stays on its
    slot: the lane's vitals read it, and memory admission's typical worker size (``pool_admission.typical_size``)."""
    if slot is not None and isinstance(sample, dict) and type(sample.get("rss_bytes")) is int:
        slot.rss = {"rss_bytes": sample["rss_bytes"], "rss_measure": str(sample.get("rss_measure") or "")}


def carrier_vitals() -> Dict[str, Any]:
    """The lane's carriers for the vitals (owner decision 4A: no number bounds them; the vitals show their
    count and RSS): how many, how many idle, and each one's last own report in MiB, in slot order (None
    before its first turn's heartbeat). A section of the per-minute ``supervisor_load`` row; memory only."""
    carriers = lane_carriers(dict(_pool().WORKERS))  # one copy: a spawn on another thread never breaks the read
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
    """Under Q: the lane's RUNNING rows awaiting a carrier (none holds them yet), oldest first.

    A row reserved for the inline executor that admitted it (``_inline_bind``) awaits none: no door here
    hands it out, counts it as wanting a carrier or starts the degraded executor for it."""
    rows = [meta for meta in _pool().RUNNING.values()
            if conversation_row(meta) and meta.get("worker_id") is None and not meta.get("_inline_bind")
            and not isinstance(meta.get("_terminal_file_recovery"), dict)]
    return sorted(rows, key=lambda meta: float(meta.get("started_at") or 0.0))


_STOPS_READ: Dict[str, Any] = {"identity": None, "read": ({}, [])}


def _held_by_a_stop(rows: List[Dict[str, Any]]) -> set:
    """The awaiting rows a Stop owns (custody settles them), read only while turns await a carrier.

    The cancel-intent projection is read as every behaviour read is (the watchdog's enforcement read): fail-soft
    with disclosure. An intent and a quarantined intent row hold their own task; an unreadable projection, which
    no Stop can be minted into, holds none and is disclosed (``disclose_corruption``, and a warning here) rather
    than holding every turn silently until someone repairs it. Parsed once per file identity: a quiet pass stats.
    """
    ids = {str(meta["task"].get("id") or "") for meta in rows}
    if not ids:
        return set()
    from ouroboros.cancel_intents import _intents_path_read, read_active_intents
    from ouroboros.utils import file_identity

    path = _intents_path_read(_pool().DRIVE_ROOT)
    try:
        identity = (str(path), file_identity(path.stat()))
    except OSError:
        identity = (str(path), None)  # absent: no intent, nothing read
    if _STOPS_READ["identity"] != identity:
        read = read_active_intents(_pool().DRIVE_ROOT, disclose_corruption=True)
        if read[1] is None:
            log.warning("cancel intents unreadable: awaiting conversation turns start without the Stops it held")
        _STOPS_READ.update(identity=identity, read=read)
    intents, unseen = _STOPS_READ["read"]
    return ids & (set(intents) | set(unseen or ()))


def publish_awaiting_rows() -> bool:
    """Outside Q: one admission barrier for every lane row not on disk yet (a carrier awaited, or its own
    executor's bind); True when none stays unpublished.

    The capture happens under Q inside the barrier and covers every row marked before it, so only the rows
    collected before the barrier lose their mark; a row admitted meanwhile keeps it for the next pass.
    A failed barrier (False: the documents on disk are the previous generation's) keeps them all held.
    """
    from supervisor import queue

    with _queue_lock:
        pending = [meta for meta in _pool().RUNNING.values() if conversation_row(meta) and meta.get("_unpublished")]
    if not pending:
        return True
    if queue.publish_queue_snapshot(reason="conversation_turn_admitted") is False:
        log.warning("Conversation turns %s admitted but their rows are not on disk yet: no carrier holds them",
                    [str(meta["task"].get("id") or "") for meta in pending])
        return False
    with _queue_lock:
        for meta in pending:
            meta.pop("_unpublished", None)
    return True


def dispatch_admitted_turn(meta: Dict[str, Any]) -> None:
    """Outside Q, once the admission transaction released it: the barrier, then this turn's carrier.

    Under Q again the row takes a ready idle carrier, else the disclosed degraded executor on a physical
    fact (``worker_chat_lane._degrade_if_no_carrier_can_start``; ``InlineExecutorUnavailable`` propagates
    to the caller when its thread cannot start); a row a Stop took meanwhile, one already held or reserved for
    its own executor, or one whose barrier failed is left as it is (the lane's next pass looks again).
    """
    from supervisor.worker_chat_lane import _degrade_if_no_carrier_can_start

    publish_awaiting_rows()
    task_id = str(meta["task"].get("id") or "")
    withdrawals: list = []
    with _queue_lock:
        if (_pool().RUNNING.get(task_id) is not meta or meta.get("worker_id") is not None
                or meta.get("_inline_bind")):
            return
        slot = None if meta.get("_unpublished") else ready_idle_carrier(for_owner=owner_turn(meta["task"]))
        if not _presence_handoff_closed(meta, withdrawals) and not meta.get("_unpublished"):
            if slot is not None and hand_to_carrier(meta, slot):
                return
            if not _presence_handoff_closed(meta, withdrawals):
                _degrade_if_no_carrier_can_start(meta)
    _finish_presence_withdrawals(withdrawals)


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
    """Under Q: the owner's live conversation turns (the census at entry, the auto-resume gate): the lane's rows and
    the inline executors' registered actors, as base's registered direct actors were (its turns and wakes). A
    correspondent's Presence turn is its place's conversation, never the owner's."""
    from ouroboros.task_carrier import execution_kind
    from supervisor.active_activity import get_direct_activity_registry

    live = {tid for tid, meta in _pool().RUNNING.items()
            if conversation_row(meta) and execution_kind(meta.get("task") or {}) != "presence"}
    live.update(entry.activity_id for entry in get_direct_activity_registry().actors() if entry.kind != "presence")
    return len(live)


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------

def admit_turn(task: Dict[str, Any], *, chat_id: int, client_message_id: str = "", quiet: bool = False,
               reply_expected: bool = False, reserved: bool = False) -> Dict[str, Any]:
    """Under Q and the repo-writer gate lock (the caller's): the RUNNING row, marked, awaiting its barrier.

    Durable intent before effects: the row is written here and published by ``dispatch_admitted_turn``
    (the caller's, outside Q) before the turn reaches any process; until then it is ``_unpublished``
    and no carrier or executor holds it. ``task`` is the admitted seed (id, chat, text, origin
    facts); the carrier layers attachments, Project binding, authority, contract
    and tool profile on it as its first step (``worker_chat_lane.prepare_conversation_task``).
    Returns the RUNNING meta; ``worker_id`` is None while the row awaits a carrier.
    ``reserved``: the caller binds the turn to its own inline executor (``inline_carrier.start_inline_turn``,
    which clears the mark); until then the row is ``_inline_bind``, awaiting no carrier, so the tick's lane
    pass in the barrier window hands it to none.
    """
    from ouroboros.task_status import execution_owner_record
    from supervisor import queue

    pool = _pool()
    now = time.time()
    task_id = str(task["id"])
    task.setdefault("root_task_id", task_id)
    origin = "presence" if task.get("_presence_turn") else "direct"
    host_presence = reserved and origin == "presence"
    root = (task.get("drive_root") or pool.DRIVE_ROOT) if origin == "presence" else pool.DRIVE_ROOT
    task["_execution_owner"] = {**execution_owner_record(root, task, origin),
                                "lane": LANE_CONVERSATION,
                                "executor": EXECUTOR_INLINE_DEGRADED if host_presence else EXECUTOR_WORKER}
    task[PREPARE_KEY] = {
        "admitted_at": now, "quiet": bool(quiet), "reply_expected": bool(reply_expected),
        "client_message_id": str(client_message_id or ""),
        "census": {"direct_turns_at_entry": live_turn_count()},
    }
    # The host-only fresh-identity mark rides the dispatched copy, never the published row.
    row_task = {key: value for key, value in task.items() if key != "_fresh_identity"}
    meta = _awaiting_row(row_task, attempt=1, client_message_id=client_message_id, reply_expected=reply_expected,
                         fresh_identity=task.get("_fresh_identity"))
    meta["_unpublished"] = True  # on disk only once its barrier lands (``dispatch_admitted_turn``, outside Q)
    if reserved:
        meta["_inline_bind"] = True
    pool.RUNNING[task_id] = meta
    # The unarmed Host owns its reserved Presence row's synchronous barrier and withdrawal.
    if not host_presence:
        queue.persist_queue_snapshot(reason="conversation_turn_admitted")
    return meta


def _awaiting_row(task: Dict[str, Any], *, attempt: int, client_message_id: Any, reply_expected: Any,
                  fresh_identity: Any) -> Dict[str, Any]:
    """The RUNNING meta of a lane turn no carrier holds yet (an admission, or a dead carrier's retry)."""
    now = time.time()
    return {"task": task, "worker_id": None, "started_at": now, "last_heartbeat_at": now, "last_progress_at": now,
            "budget_paused_sec": 0.0, "soft_sent": False, "attempt": int(attempt), "lane": LANE_CONVERSATION,
            "client_message_id": str(client_message_id or ""), "reply_expected": bool(reply_expected),
            # Fence buffered frames while the row is still waiting for a carrier.  A
            # physical handoff replaces this with another token immediately before
            # the task copy enters the worker queue.
            "fresh_identity": bool(fresh_identity), "phase_generation": uuid.uuid4().hex}


def readmit_after_crash(task: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """Under Q: a dead carrier's retry (the same id, its next attempt, saved work attached) returns to the lane.

    Never PENDING: the conversation's recovery is independent of the project pool's capacity and crash
    fuse (FINAL_PLAN §1). The row awaits a carrier again with the turn's own lane facts, marked here and
    published by the caller outside Q (``dispatch_admitted_turn``: a ready idle carrier takes it then, the
    disclosed degraded executor on a physical fact, otherwise the lane's reconcile hands it the next carrier,
    the dead slot's replacement among them). Until that barrier lands the retry is visibly unstarted, held
    from every carrier, as an admission's (``admit_turn``).
    """
    from supervisor import queue

    task["_execution_owner"] = {**dict(task.get("_execution_owner") or {}), "lane": LANE_CONVERSATION,
                                "executor": EXECUTOR_WORKER}
    meta = _awaiting_row(task, attempt=int(task.get("_attempt") or 1), client_message_id=prior.get("client_message_id"),
                         reply_expected=prior.get("reply_expected"), fresh_identity=prior.get("fresh_identity"))
    if "_presence_execution" in prior:
        meta["_presence_execution"] = prior["_presence_execution"]  # the same Host request awaits its retried body
    meta["_unpublished"] = True
    _pool().RUNNING[str(task["id"])] = meta
    queue.persist_queue_snapshot(reason="conversation_turn_retried")
    return meta


def resumes_on_lane(task: Any) -> bool:
    """A queued row this lane carries again: my human's own Main or room turn, returned by a stop door, a Resume
    or its own sleep (owner choice of 9 November 2026: the conversation path survives every permitted return).

    The facts the host stamped, as ``conversation_lease.owner_conversation_key`` reads them for Main: the direct
    origin with the owner door's ingress (``dialogue_provenance.run_origin``: a wake or the auto-resume has none),
    this lane as its recorded carrier, any executor but a live call's. A Presence turn, a turn the mind sent to the
    pool (``detached_at``) and pooled work keep their project slot."""
    from ouroboros.dialogue_provenance import is_presence_task, run_origin
    from ouroboros.task_carrier import EXECUTOR_INLINE_CALL, carrier_executor, execution_kind, execution_owner

    if not isinstance(task, dict):
        return False
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return bool(execution_kind(task) == "direct" and execution_owner(task).get("lane") == LANE_CONVERSATION
                and carrier_executor(task) != EXECUTOR_INLINE_CALL and run_origin(task)["owner_ingress"]
                and not is_presence_task(task) and not (task.get("detached_at") or metadata.get("detached_at")))


def saved_carrier(task: Dict[str, Any], drive_root: Any) -> Dict[str, Any]:
    """The carrier record a saved queued row lacks, from the same task's own result; else ``{}``. Never raises.

    A release before #2870 saved PENDING rows without ``_execution_owner``: my human's returned turn waiting in the
    queue then came back pooled. Only the row's own facts qualify it (direct, the owner door's ingress, never
    Presence or detached), and only a positive record restores it: this task's conversation-lane direct carrier,
    undetached and not terminal, for the attempt the row carries (its saved work's or seed's, else its own; a first
    direct launch records 0 for 1, unmarked). No label or source is read as a carrier."""
    import pathlib

    from ouroboros.dialogue_provenance import is_presence_task, run_origin
    from ouroboros.task_carrier import EXECUTORS
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_CANCEL_REQUESTED, load_task_result
    from supervisor.task_admission import SEED_RETURN

    task_id = str(task.get("id") or "")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if ("_execution_owner" in task or task.get("_is_direct_chat") is not True or not task_id or is_presence_task(task)
            or task.get("detached_at") or metadata.get("detached_at") or not run_origin(task)["owner_ingress"]):
        return {}
    source = next((v for v in (task.get("_working_recovery"), task.get(SEED_RETURN)) if isinstance(v, dict)), None)
    saved = task.get("_attempt") if source is None else source.get("from_attempt")
    try:
        stored = load_task_result(pathlib.Path(task.get("budget_drive_root") or drive_root), task_id, strict=True) or {}
    except Exception:
        log.warning("The saved carrier of %s is unreadable; its row returns as saved", task_id, exc_info=True)
        return {}
    owner, recorded = stored.get("execution_owner"), stored.get("task_attempt")
    if (not isinstance(owner, dict) or stored.get("task_id") != task_id or owner.get("task_id") != task_id
            or (source or {}).get("source_task_id", task_id) != task_id or stored.get("detached_at")
            or (owner.get("kind"), owner.get("lane")) != ("direct", LANE_CONVERSATION)
            or owner.get("executor") not in EXECUTORS or type(saved) is not int or type(recorded) is not int
            or owner.get("task_attempt") != recorded
            or stored.get("status") in _TRULY_TERMINAL_STATUSES | {STATUS_CANCEL_REQUESTED}):
        return {}
    first_direct = saved == 1 and recorded == 0 and "admitted_dispatch_attempt" not in stored
    if not (first_direct or recorded == saved and stored.get("admitted_dispatch_attempt") == saved):
        return {}
    return {"_execution_owner": dict(owner)}


def returned_row_facts(task: Dict[str, Any]) -> Dict[str, Any]:
    """Under Q: a returned turn's RUNNING facts once a carrier holds it (``worker_assignment._bind_running``): the
    lane's row shape (``_awaiting_row``) with its admission's client id and the owner door's reply fact. The fresh
    identity of its first admission does not return: it grants a later attempt nothing."""
    from supervisor.active_activity import reply_expected
    from supervisor.worker_chat_lane import _client_message_id

    facts = task.get(PREPARE_KEY) if isinstance(task.get(PREPARE_KEY), dict) else {}
    return {"lane": LANE_CONVERSATION, "fresh_identity": False, "dispatched_at": time.time(),
            "client_message_id": str(facts.get("client_message_id") or _client_message_id(task.get("metadata"))),
            "reply_expected": reply_expected(task)}


def note_returns_waiting(count: int) -> None:
    """Under Q: the returned turns an assignment pass found dispatchable with no idle carrier; the lane starts one
    carrier for each, as for a turn awaiting one (``reconcile_conversation_lane``)."""
    _RETURNS_WAITING[0] = max(0, int(count))


def hand_to_carrier(meta: Dict[str, Any], slot: Any) -> bool:
    """Under Q: put the turn on the carrier's queue, then bind the row and slot to each other.

    The put is the handoff boundary (as ``worker_assignment._claim_worker_launch``):
    from there the carrier may hold the turn, so the binding and its snapshot follow
    it. A put that raises leaves the row awaiting the next ready carrier.
    """
    from supervisor import queue

    task = meta["task"]
    dispatched = dict(task)
    # Attempt and worker slot can both be reused by a supported Resume.  A
    # buffered phase frame from the previous physical dispatch must not claim
    # the fresh row, so stamp the row and the carrier copy at the handoff.
    phase_generation = uuid.uuid4().hex
    meta["phase_generation"] = phase_generation
    meta.pop("turn_execution", None)
    meta.pop("turn_phase", None)
    dispatched["_phase_generation"] = phase_generation
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


def _take_unstarted_presence(meta: dict) -> Any:
    """Under Q: take this exact, unbound first admission for the Host's retryable outcome."""
    turn_id = meta["task"]["id"]
    expected = meta.get("_presence_execution")
    if expected is None or meta.get("attempt") != 1 or _pool().RUNNING.get(turn_id) is not meta:
        return None  # a crash retry keeps the same waiter for a turn a previous carrier already held
    from ouroboros.presence_executions import _find_execution

    registry, execution = _find_execution(turn_id, expected)
    if registry is not None and execution is not None and take_awaiting_turn(turn_id) is not None:
        return registry, execution
    return None


def _withdraw_unstarted_presence(meta: dict, cause: BaseException) -> None:
    """Under Q, during thread-start rollback: only a first admission can tell the Host nothing ran."""
    if taken := _take_unstarted_presence(meta):
        from ouroboros.presence_executions import PresenceTurnNotStarted

        registry, execution = taken
        error = PresenceTurnNotStarted("presence turn thread could not start")
        error.__cause__ = cause
        registry.settle_from_frame(execution, error=error)


def _presence_handoff_closed(meta: dict, withdrawals: list, *, closing: bool = False) -> bool:
    """Under Q, at the fresh Presence claim: an observed closure forbids the handoff.

    The latch read orders this claim against ordinary server shutdown; a later Stop
    cannot undo a successful put. An explicit task Stop keeps its own settlement.
    """
    handoff = meta.get("_presence_handoff")
    if handoff is None:
        return False
    from supervisor import queue
    from supervisor.queue_publication import publication_identity

    stopped, identity = handoff
    same_generation = publication_identity() == identity
    if same_generation and not closing and not stopped():
        return False
    # A failed supervisor startup can be retried in the same process with the
    # same RUNNING map. Release only our exact unbound row, never a different
    # drive's row; its old publication identity grants no write in the new one.
    if ((same_generation or queue.owns_live_queue(meta["task"].get("drive_root")))
            and not _held_by_a_stop([meta]) and (taken := _take_unstarted_presence(meta))):
        withdrawals.append((identity, taken))
    return True


def _finish_presence_withdrawals(withdrawals: list) -> None:
    """Outside Q: publish the removed admissions before releasing their live Host waiters."""
    if not withdrawals:
        return
    from supervisor import queue
    from supervisor.queue_publication import publication_identity

    with _queue_lock:
        current = publication_identity()
    published = False
    try:
        if any(identity == current for identity, _ in withdrawals):
            published = queue.publish_queue_snapshot(reason="presence_turn_not_started",
                                                     expected_identity=current) is True
    except Exception:
        log.warning("Could not publish withdrawn Presence admissions", exc_info=True)
    finally:
        settle_presence_withdrawals(withdrawals, current if published else None)


def settle_presence_withdrawals(withdrawals: list, published_identity: Any) -> None:
    """Outside Q: a removed admission and a confirmed publication are separate facts."""
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted

    for identity, (registry, execution) in withdrawals:
        error = (PresenceTurnNotStarted("the Presence carrier is closing; retry the event")
                 if identity == published_identity else PresenceTurnInterrupted(
                     "the Presence withdrawal was not confirmed; resend the same event for its outcome"))
        registry.settle_from_frame(execution, error=error)


def take_closing_presence() -> list:
    """Under Q: withdraw fresh pooled admissions before generic pool teardown."""
    withdrawals: list = []
    for meta in awaiting_turns():
        _presence_handoff_closed(meta, withdrawals, closing=True)
    return withdrawals


def take_closing_reserved_presence(withdrawals: list) -> None:
    """Under Q: withdraw the unbound Host-inline reservations an ordinary pool stop owns.

    The stop's partition has already copied each unbound reserved row aside for settlement
    (``workers._partition_slots_for_stop``), so its publisher can bind only the original
    object and the body never starts. A pooled admission in that position is withdrawn by
    ``take_closing_presence``; the reserved first message gets the same closing: without a
    recorded owner Stop — whose custody keeps the row, its exact source and its retained
    refusal — a never-started admission must stay retryable, never become the generic
    teardown's terminal ``cancelled`` that carries neither a Stop nor the event identity (a
    retry of that event is then refused forever). A row whose waiter already settled — its
    publisher refused the fenced copy — is only taken out of RUNNING: the waiter heard its
    outcome and the stop's snapshot publishes the removal. Retried bodies (``attempt``
    above 1), rows without this process's first admission and rows a Stop holds keep the
    settlement that already owns them.
    """
    from supervisor.queue_publication import publication_identity

    identity = publication_identity()
    for meta in list(_pool().RUNNING.values()):
        if not host_presence_row(meta) or meta.get("worker_id") is not None:
            continue
        if (meta.get("attempt") != 1 or meta.get("_presence_execution") is None
                or _held_by_a_stop([meta])):
            continue
        if (taken := _take_unstarted_presence(meta)) is not None:
            withdrawals.append((identity, taken))
        else:
            take_awaiting_turn(str(meta["task"]["id"]))


def pool_stop_basis(eligible: bool) -> Any:
    """Under Q: bind the closing claims to the existing shutdown barrier's generation."""
    from supervisor.queue_publication import publication_identity

    withdrawals = take_closing_presence() if eligible else []
    if eligible:
        take_closing_reserved_presence(withdrawals)
    return (withdrawals, publication_identity()) if withdrawals else None


def publish_pool_stop_snapshot(reason: str, basis: Any) -> bool | None:
    """Publish the existing shutdown snapshot, which confirms its saved parks, and settle any claims it carried."""
    from supervisor import queue, restart_retention as retention

    if basis is None:
        return retention.publish_pending_parks(queue, reason, final=True)
    withdrawals, identity = basis
    published = None
    try:
        published = retention.publish_pending_parks(queue, reason, final=True, expected_identity=identity)
        return published
    finally:
        settle_presence_withdrawals(withdrawals, identity if published is True else None)


def close_presence_after_failed_startup() -> None:
    """End unbound admissions; a failed, partially armed pool has no next tick."""
    try:
        with _queue_lock:
            if armed():
                arm(None)  # preserve the observed spawn refusal of a pool that never armed
            withdrawals = take_closing_presence()
        _finish_presence_withdrawals(withdrawals)
    except Exception:
        log.warning("Could not close awaiting Presence admissions after supervisor startup failed", exc_info=True)


def dispatch_awaiting_turns(*, require_armed: bool = False) -> int:
    """Claim under Q, then complete any shutdown withdrawals outside it."""
    withdrawals: list = []
    try:
        with _queue_lock:
            if require_armed and (not armed() or not _pool().WORKERS):
                return 0
            return _dispatch_awaiting_turns(withdrawals)
    finally:
        _finish_presence_withdrawals(withdrawals)


def _dispatch_awaiting_turns(withdrawals: list) -> int:
    """Under Q: hand every awaiting turn to a ready idle carrier, oldest first; returns how many moved.

    When no carrier can start at all (``inline_carrier.refused_carrier_reason``: a
    physical fact, never "busy"), the turns still waiting take the disclosed degraded
    executor instead of waiting for a carrier that will not come.
    """
    from supervisor.inline_carrier import EXECUTOR_INLINE_DEGRADED, refused_carrier_reason, start_inline_turn

    waiting = awaiting_turns()
    # A Stop's custody settles its rows; a row not yet on disk waits for its barrier (``publish_awaiting_rows``,
    # outside Q, before this pass): no carrier takes them.
    held = _held_by_a_stop(waiting)
    moved = 0
    for meta in waiting:  # a wake waiting for the spare never holds up the owner's turn behind it
        if str(meta["task"].get("id") or "") in held:
            continue
        slot = None if meta.get("_unpublished") else ready_idle_carrier(for_owner=owner_turn(meta["task"]))
        if _presence_handoff_closed(meta, withdrawals) or meta.get("_unpublished"):
            continue
        if slot is not None and hand_to_carrier(meta, slot):
            moved += 1
    waiting = [meta for meta in awaiting_turns()
               if str(meta["task"].get("id") or "") not in held and not meta.get("_unpublished")]
    if waiting and not any(w.booting and w.busy_task_id is None for w in lane_carriers()):
        reason = refused_carrier_reason()
        if reason:
            for meta in waiting:
                if _presence_handoff_closed(meta, withdrawals):
                    continue
                try:
                    start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason=reason,
                                      on_not_started=_withdraw_unstarted_presence)
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
    the dispatch; the admission barrier of rows not on disk yet runs before it,
    outside Q. Nothing spawns in a marker window (the admission discloses the
    degraded executor there). A carrier that exhausted its readiness attempts is
    parked until Restart like a project slot, and no second carrier starts
    beside it: an awaiting turn then takes the degraded executor too.
    """
    from supervisor.worker_pool_lifecycle import (
        _spawn_worker_slot, request_worker_quiescence, retire_quiescent_worker, worker_ready,
    )

    pool = _pool()
    if not armed() or not pool.WORKERS:
        return
    publish_awaiting_rows()
    # A marker window forbids a spawn, not a waiting turn's carrier (or its degraded executor).
    dispatch_awaiting_turns(require_armed=True)
    with _queue_lock:
        if not armed() or not pool.WORKERS:
            return
        if pool.repo_writer_admission_closed():
            return
        carriers = lane_carriers()
        if any(getattr(slot, "readiness_exhausted", False) for slot in carriers):
            return
        idle = [w for w in carriers if w.busy_task_id is None and worker_ready(w) and not w.retiring]
        booting = [w for w in carriers if w.busy_task_id is None and w.booting and not w.retiring]
        # The reserved spare, plus one carrier per turn still waiting: admitted, or returned and dispatchable.
        wanted = 1 + len(awaiting_turns()) + _RETURNS_WAITING[0]
        missing = max(0, wanted - len(idle) - len(booting))
        if len(idle) > 1:
            # Ask every idle carrier: a resident with late duties is not the spare. Only
            # acknowledged surplus beyond the one spare retires, through the reaper.
            for slot in idle:
                request_worker_quiescence(slot)
            quiet = [w for w in idle if w.quiescence_confirmed]
            for slot in quiet[1:]:
                retire_quiescent_worker(slot)
        next_id = max(0, max(pool.WORKERS, default=-1) + 1)
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


def reconcile_under_project_fence() -> None:
    """Reconcile the lane while the project pool is fenced: the crash fuse that fences project slots does
    not fence the conversation lane, and the fenced ``assign_tasks`` returns before ``reconcile_worker_capacity``,
    the lane's other caller. A waiting turn is dispatched, a refused spawn noticed, the degraded executor
    started. Lifecycle serializer held by the caller."""
    try:
        reconcile_conversation_lane()
    except Exception:
        log.exception("Conversation lane reconcile under the project fence failed")


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
            if isinstance(sent := meta.get("context_attempt"), dict) and sent.get("task_attempt") == attempt:
                row["context_attempt"] = dict(sent)
            from ouroboros.lead_fact import lead_processing_fields

            row.update(lead_processing_fields(meta.get("lead"), {**task, "_lead_request_id": meta.get("_lead_request_id"),
                                                                 "processing_last": meta.get("processing_last")}))
            rows.append(row)
    return rows


def handle_direct_turn_phase(evt: Dict[str, Any], ctx: Any) -> None:
    """A task's start fact (``TURN_PHASES``, ``worker_chat_lane.TurnPhaseFrames``) lands on its RUNNING row for the
    census, in either lane. A row is fresh for each dispatch and names the slot holding it (``worker_id``): a frame
    of another slot moves nothing, and on the slot the first execution of its attempt to report owns it
    (``turn_execution``, a pooled task's own first frame), so a late frame of another execution (a Resume that kept
    the attempt) moves nothing either."""
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
        slot = evt.get("worker_id")
        if slot is not None and meta.get("worker_id") is not None and str(slot) != str(meta.get("worker_id")):
            return
        generation = str(evt.get("phase_generation") or "")
        current_generation = str(meta.get("phase_generation") or "")
        # A phase frame from a previous physical handoff may arrive first after Resume.
        # Once admission stamped a generation, frames without that exact token cannot
        # claim the fresh execution, even when attempt and worker slot are reused.
        if current_generation and generation != current_generation:
            return
        execution = str(evt.get("execution_id") or "")
        if execution and meta.setdefault("turn_execution", execution) != execution:
            return
        current = str(meta.get("turn_phase") or TURN_PHASES[0])
        if TURN_PHASES.index(phase) >= TURN_PHASES.index(current):  # facts only move forward
            meta["turn_phase"] = phase


BOUND_FACTS = ("_project_admission", "project_id")  # the carrier's first-step facts a row rebuilds from


def handle_direct_turn_bound(evt: Dict[str, Any], ctx: Any) -> None:
    """The carrier bound its turn (the Project basis of a room turn, its admitted tier): the RUNNING row's task learns
    the facts everything rebuilt from the row needs — the snapshot's running rows, the pooled park, restore."""
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
        from ouroboros.route_choice import PROCESSING_KEY, processing_snapshot

        # The admitted tier, for its card; an owner tier act already on the row is newer than any admission frame.
        if (carrier := processing_snapshot(facts.get(PROCESSING_KEY))) and not meta["task"].get("owner_processing"):
            meta["task"]["metadata"] = {**(meta["task"].get("metadata") or {}), PROCESSING_KEY: carrier}
        if meta["task"].get("_project_admission") is not None:
            meta["task"].pop("_project_scope_none", None)  # a real basis replaces attested absence
        persist_queue_snapshot(reason="conversation_turn_bound")


def _queue_row(ctx: Any, task_id: str) -> Any:
    """Under Q: the row ``task_id`` names, its RUNNING meta's task, else its PENDING row (None when it has none)."""
    meta = (getattr(ctx, "RUNNING", None) or {}).get(task_id)
    if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
        return meta["task"]
    return next((row for row in getattr(ctx, "PENDING", None) or [] if str(row.get("id") or "") == task_id), None)


def handle_task_named(evt: Dict[str, Any], ctx: Any) -> None:
    """A carrier coined a title (its namer or a wake's cause): the bridge broadcasts it as the server's own did.

    A queue row admitted with a name (a late input's head, ``worker_chat_lane._late_input_naming``) follows the
    task's chosen name by value, running or already parked, so its receipts, works row, park, detach and restore never
    fall back to the admission name: the name its result holds, the frame's only where it holds none. That result is the
    record owning the name, as the replica reducer reads it (``canonical_name_fields``): a titled canonical record's own,
    else the copy its carrier writes on a split drive. The namer writes
    that result before its frame, while a frame can be older than it: a resumed attempt's first step re-announces the
    name the Resume dispatched, and two carriers' frames can cross. The result is read off Q (ARCHITECTURE §5) and
    never waits on a busy store; re-taken Q finds the row by its task id again. A result that cannot be read (busy,
    unreadable or refused) leaves the row's own name standing, logged, never the frame's. The bridge broadcasts the
    name the row then carries. A row without one keeps reading its durable result."""
    from ouroboros.post_task_checkpoint import canonical_name_fields
    from ouroboros.store import acquisition_budget
    from ouroboros.task_results import load_task_result
    from supervisor.queue_snapshot import persist_queue_snapshot

    task_id, name = str(evt.get("task_id") or ""), str(evt.get("suggested_name") or "")
    if not task_id or not name:
        return
    with _queue_lock:
        row = _queue_row(ctx, task_id)
        named = isinstance(row, dict) and bool(row.get("suggested_name"))
        if named:
            canonical = pathlib.Path(row.get("budget_drive_root") or _pool().DRIVE_ROOT)
            execution = pathlib.Path(row.get("drive_root") or _pool().DRIVE_ROOT)  # where its carrier writes the result
    if named:
        try:
            with acquisition_budget(0):
                record = load_task_result(canonical, task_id, strict=True) or {}
                if execution != canonical and "suggested_name" not in canonical_name_fields(record):
                    record = load_task_result(execution, task_id, strict=True) or {}
                chosen = str(record.get("suggested_name") or "") or name
        except Exception as exc:
            chosen = ""
            log.warning("task_named %r for %s not applied: its result could not be read (%r); "
                        "the queue row keeps its name", name, task_id, exc)
        with _queue_lock:
            row = _queue_row(ctx, task_id)
            if isinstance(row, dict) and row.get("suggested_name"):
                if chosen and row["suggested_name"] != chosen:
                    row["suggested_name"] = chosen
                    persist_queue_snapshot(reason="conversation_turn_named")
                chosen = row["suggested_name"]
        name = chosen or name
    try:
        ctx.bridge.broadcast({"type": "task_named", "task_id": task_id, "suggested_name": name})
    except Exception:
        log.debug("task_named broadcast failed for %s", task_id, exc_info=True)
