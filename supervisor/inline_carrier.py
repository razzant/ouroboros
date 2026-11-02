"""The two disclosed inline executors: the worker task body on a thread of the server process.

One execution implementation, two carriers. A conversation turn normally runs in a
lane worker (``conversation_lane``). When a carrier physically cannot start — a
spawn refused, or the repo-writer gate closed in an assisted-update marker window
(``executor=inline_degraded``, owner decision 2A) — or when the turn is a live
realtime call whose host port is process-local (``executor=inline_call``, owner
decision 3A), the SAME task body (``worker_process.run_worker_task``) runs on a
daemon thread of this process over in-memory queues: the same frames on the
server's bus, the same RUNNING row and census, the same mailbox controls, the
pooled owner-wait protocol. Never chosen because something is busy.

The carrier registers a thread-backed pseudo-slot in ``WORKERS`` (``lane``
``conversation``, ``executor`` named) so the worker protocol reaches it: owner-wait
grants and the ``task_done`` release address the slot; Stop's pooled custody finds
it busy, asks it to stop (``terminate`` is the cooperative request: the finalize
intent is already written) and confirms its death by the thread's end. The slot
has no pid: nothing ledgers, tree-kills or respawns it, and a whole-table stop that
cannot confirm it reports it unconfirmed. Its registered actor keeps the in-process
seams a live call needs (routing's admission lock, the hub's close).

Every inline execution is disclosed: a ``conversation_inline_executor`` journal row,
the owner's notice for the degraded case, and ``inline_vitals`` for the Runtime
vitals; in normal operation the degraded count is 0.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

from ouroboros.observability import stamp_finalization_enqueue
from ouroboros.task_carrier import EXECUTOR_INLINE_CALL, EXECUTOR_INLINE_DEGRADED, LANE_CONVERSATION
from ouroboros.utils import utc_now_iso
from supervisor.log_addressing import TurnEventQueue as _TurnEventQueue
from supervisor.queue import _queue_lock
from supervisor.state import append_jsonl

log = logging.getLogger(__name__)

INLINE_KEPT = 50  # the disclosed executions this process remembers for the vitals: a memory bound, never a judgment
_INLINE: Deque[Dict[str, Any]] = deque(maxlen=INLINE_KEPT)
_INLINE_LOCK = threading.Lock()
_DEGRADED_TOTAL = [0]


def _pool():
    from supervisor import workers

    return workers


class InlineCarrier:
    """A thread-backed stand-in for a worker's ``Process``: alive while the body runs, no pid.

    ``terminate``/``kill`` request a cooperative stop (the body reads its controls at
    every boundary); ``join`` waits for the thread. ``_ouroboros_stop_request`` answers
    Panic's owner-stop protocol with an honest unconfirmed receipt.
    """

    pid = None
    daemon = True

    def __init__(self, target: Callable[[], None], name: str, on_stop: Optional[Callable[[], None]] = None) -> None:
        self._thread = threading.Thread(target=target, name=name, daemon=True)
        self._started = False
        self.exitcode: Optional[int] = None
        self.stop_requested = False
        self._on_stop = on_stop

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def is_alive(self) -> bool:
        return self._started and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._started:
            self._thread.join(timeout)
            if not self._thread.is_alive() and self.exitcode is None:
                self.exitcode = 0

    def terminate(self) -> None:
        """Cooperative: arm the turn's finalize control once; the body ends at its next control boundary."""
        first, self.stop_requested = not self.stop_requested, True
        if first and self._on_stop is not None and self.is_alive():
            try:
                self._on_stop()
            except Exception:
                log.debug("inline executor stop request failed", exc_info=True)

    kill = terminate

    def _ouroboros_stop_request(self) -> Dict[str, Any]:
        self.terminate()
        return {"pid": None, "requested": True, "scope": "inline_executor", "confirmation": "unconfirmed",
                "error": "", "limit": "a thread of the server process: stops at its next control boundary"}


def inline_vitals() -> Dict[str, Any]:
    """The disclosed inline executions this process ran (newest last) and the degraded total."""
    with _INLINE_LOCK:
        return {"inline_executions": [dict(row) for row in _INLINE], "degraded_total": _DEGRADED_TOTAL[0]}


def _disclose(task: Dict[str, Any], executor: str, reason: str) -> None:
    row = {"ts": utc_now_iso(), "type": "conversation_inline_executor", "task_id": str(task.get("id") or ""),
           "chat_id": int(task.get("chat_id") or 0), "executor": executor, "reason": reason}
    with _INLINE_LOCK:
        _INLINE.append(row)
        if executor == EXECUTOR_INLINE_DEGRADED:
            _DEGRADED_TOTAL[0] += 1
    try:
        from supervisor.state import append_jsonl

        append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", row)
    except Exception:
        log.debug("inline executor row not written", exc_info=True)
    if executor == EXECUTOR_INLINE_DEGRADED:
        try:
            _pool().send_with_budget(
                int(task.get("chat_id") or 0),
                f"⚠️ No conversation carrier could start ({reason}); continuing this turn in direct-chat "
                "mode inside the server process.",
                is_progress=True, task_id=str(task.get("id") or ""),
                progress_meta={"task_incident": "conversation_degraded_executor",
                               "toast_once": f"{task.get('id')}:conversation_degraded_executor"},
                role="system", system_type="worker_failure")
        except Exception:
            log.debug("degraded executor notice failed", exc_info=True)


def start_inline_turn(meta: Dict[str, Any], *, executor: str, reason: str, agent: Any = None,
                      realtime_turn: Any = None, on_failure: Optional[Callable[[Any, BaseException], bool]] = None) -> Any:
    """Under Q: bind the admitted RUNNING row to a fresh pseudo-slot and start its body thread.

    ``meta`` is the lane's RUNNING meta (``conversation_lane.admit_turn``), still
    awaiting a carrier. The actor is this process's fresh native agent (the
    pooled owner wait over the pseudo-slot's queue), registered in the direct
    activity registry for the in-process seams; a live call gets its turn attached.
    Returns the installed slot.
    """
    import queue as stdqueue
    from functools import partial

    from ouroboros.owner_wait import worker_owner_wait
    from supervisor import queue
    from supervisor.active_activity import get_direct_activity_registry

    pool = _pool()
    task = meta["task"]
    task_id = str(task["id"])
    owner = dict(task.get("_execution_owner") or {})
    owner.update(lane=LANE_CONVERSATION, executor=executor)
    task["_execution_owner"] = owner
    wid = max(pool.WORKERS, default=-1) + 1
    in_q: Any = stdqueue.Queue()
    out_q = pool.get_event_q()
    if agent is None:
        agent = pool._get_chat_agent()
    agent.owner_wait_callback = partial(worker_owner_wait, wid, in_q, out_q)
    if realtime_turn is not None:
        agent._realtime_turn = realtime_turn
    dispatched = dict(task)
    if meta.get("fresh_identity"):
        dispatched["_fresh_identity"] = True
    registry = get_direct_activity_registry()
    entry = registry.register(
        task_id, int(task.get("chat_id") or 0), client_message_id=str(meta.get("client_message_id") or ""),
        project_id=str(task.get("project_id") or ""), kind="direct_chat",
        origin_message_ref=task.get("origin_message_ref"), actor=agent,
        reply_expected=bool(meta.get("reply_expected")),
    )
    entry.executor = executor
    if entry.execution is not None:
        entry.execution.advance("started")
    slot = pool.Worker(wid=wid, proc=None, in_q=in_q, busy_task_id=task_id, lane=LANE_CONVERSATION, executor=executor)
    slot.booting, slot.reaping = False, False

    def body() -> None:
        from supervisor.worker_process import run_worker_task

        try:
            run_worker_task(agent, dispatched, wid=wid, out_q=out_q, repo_dir=str(pool.REPO_DIR),
                            drive_root=str(pool.DRIVE_ROOT), on_failure=on_failure)
            slot.proc.exitcode = 0
        except Exception:
            slot.proc.exitcode = 1
            log.warning("Inline executor %s for %s ended with an error", executor, task_id, exc_info=True)
        finally:
            registry.unregister(task_id)
            with _queue_lock:
                if pool.WORKERS.get(wid) is slot:
                    pool.WORKERS.pop(wid)

    slot.proc = InlineCarrier(body, name=f"{executor}-{task_id}", on_stop=lambda: arm_inline_stop(task_id))
    slot.process_birth = f"inline:{task_id}"
    slot.ready_identity = None
    pool.WORKERS[wid] = slot
    meta["worker_id"] = wid
    meta["dispatched_at"] = time.time()
    meta["executor"] = executor
    _disclose(task, executor, reason)
    queue.persist_queue_snapshot(reason="conversation_turn_inline")
    slot.proc.start()
    return slot


def arm_inline_stop(task_id: str) -> None:
    """Write the finalize control of a live inline turn once, quietly (a pool teardown or a managed
    update stopping the executor's thread); the body reads it at its next control boundary."""
    from supervisor import queue as q
    from supervisor import workers
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from supervisor.task_reaper import request_finalization_grace

    turn = workers.direct_chat_turn(task_id)
    if turn is None or turn.get("stop_control_msg_id"):
        return
    workers.arm_direct_chat_turn(task_id, lambda live: request_finalization_grace(
        pathlib.Path(q.DRIVE_ROOT), task_id, REASON_OWNER_STOPPED_DIRECT_TURN,
        chat_id=int(live.get("chat_id") or 0), stamp=int(time.time()), toast_text="", quiet=True))


def refused_carrier_reason() -> str:
    """The physical fact that forbids a fresh carrier right now, or "" (never "busy").

    No started pool generation (``conversation_lane.armed``: a Host serving without a pool),
    a marker window (``worker_chat_lane.marker_window_closed``), a lane carrier parked after
    exhausting its readiness attempts, or a spawn the lane could not make.
    """
    from supervisor.conversation_lane import armed, lane_carriers, spawn_refusal
    from supervisor.worker_chat_lane import marker_window_closed

    if not armed():
        return "pool_not_started"  # no pool generation keeps a carrier for the lane: a Host without a pool
    if marker_window_closed():
        return "repo_writer_gate_closed"
    if any(getattr(slot, "readiness_exhausted", False) for slot in lane_carriers()):
        return "carrier_readiness_exhausted"
    return spawn_refusal()


# ---------------------------------------------------------------------------
# the inline executors (a live call; the degraded carrier)
# ---------------------------------------------------------------------------

def run_inline_turn(
    chat_id: int, text: str, image_data: Any, task_constraint: Optional[dict], task_metadata: Optional[dict],
    *, realtime_turn: Any = None, executor: str = EXECUTOR_INLINE_DEGRADED, execution: Any = None,
) -> Optional[Dict[str, Any]]:
    """Admit one turn to the lane and run its body on a thread of THIS process, disclosed by executor.

    The same RUNNING row and the same worker task body as a carrier
    (``inline_carrier.start_inline_turn``); a live call attaches its process-local
    turn to the actor. Returns the RUNNING meta, or None when refused or deferred.
    """
    from ouroboros.server_restart import agent_restart_pending
    from supervisor import queue
    from supervisor.active_activity import reply_expected
    from supervisor.conversation_lane import admit_turn
    from supervisor.worker_chat_lane import (
        _announce_turn_start, _client_message_id, _defer_chat_for_restart, _host_operation_failure,
        _report_direct_chat_error, _seed_task, owner_conversation_admitted,
    )

    client_msg_id = _client_message_id(task_metadata)
    task = _seed_task(chat_id, text, image_data, task_constraint, task_metadata)
    agent = None
    try:
        if realtime_turn is not None:
            from ouroboros.live_conversation import LiveConversationTurn

            if not isinstance(realtime_turn, LiveConversationTurn):
                raise ValueError("call requires a host-admitted request")
            task["model"] = realtime_turn.model
            task["use_local_model"] = False
        with queue._queue_lock, _pool()._repo_writer_gate_lock:
            if not owner_conversation_admitted(chat_id, client_msg_id):
                return None
            if agent_restart_pending() and queue.RETURNS_NAMED:
                _pool().send_with_budget(chat_id, "Restart intake is closed. This turn was not accepted; send it after Restart.",
                    **_host_operation_failure(task_metadata), role="system", system_type="task_not_started")
                return None
            if agent_restart_pending():
                deferred = True
            else:
                deferred = False
                meta = admit_turn(task, chat_id=chat_id, client_message_id=client_msg_id, quiet=False,
                                  reply_expected=reply_expected({"metadata": task_metadata or {}}),
                                  hold_carrier=True)
                admitted = {"task": task, "chat_id": chat_id, "client_msg_id": client_msg_id, "kind": "direct_chat",
                            "task_metadata": task_metadata, "realtime_turn": realtime_turn, "agent": None}
                try:
                    start_inline_turn(
                        meta, executor=executor, reason="live_call" if realtime_turn is not None else "degraded",
                        realtime_turn=realtime_turn,
                        on_failure=lambda runner, error: _inline_failure({**admitted, "agent": runner}, error),
                    )
                except Exception:
                    # The body never started here and no carrier may take a process-local turn:
                    # the row leaves with the refusal (reported below), never awaiting a carrier.
                    _pool().RUNNING.pop(str(task.get("id") or ""), None)
                    queue.persist_queue_snapshot(reason="conversation_turn_inline_refused")
                    raise
        if deferred:
            _defer_chat_for_restart(task, chat_id, task_metadata)
            return None
        _announce_turn_start({"task": task, "chat_id": chat_id, "client_msg_id": client_msg_id, "kind": "direct_chat"})
        return meta
    except Exception as e:
        _report_direct_chat_error({"task": task, "chat_id": chat_id, "client_msg_id": client_msg_id,
                                   "task_metadata": task_metadata, "realtime_turn": realtime_turn, "agent": agent}, e)
        return None


def _inline_failure(admitted: Dict[str, Any], error: BaseException) -> bool:
    """A live call's failure inside the inline body: close its hub and answer its refusal as the
    turn's terminal (claimed: no exception terminal follows); every other failure takes the body's own path."""
    refusal = _close_failed_call(admitted, error)
    if refusal is None:
        return False
    _report_call_context_refusal(admitted, error, refusal)
    return True


def _close_failed_call(admitted: Dict[str, Any], e: BaseException):
    """Notify the admitted call independently of result-store recovery."""
    if admitted.get("realtime_turn") is not None:
        from ouroboros.call_peers import close_hub
        from ouroboros.context_fit import MinimumViewUnfit
        from ouroboros.contracts.realtime import RealtimeSetupError
        from ouroboros.live_conversation import LiveConversationTurn
        from ouroboros.realtime_setup import CONTEXT_CAPACITY_DETAIL
        from ouroboros.store import StoreBusy

        turn = admitted["realtime_turn"]
        if isinstance(turn, LiveConversationTurn):
            # The live run normally closes its hub. Preparation can fail before it owns that duty.
            # Close before logging too: a busy store must not strand the admitted call.
            if isinstance(e, MinimumViewUnfit):
                with turn._lock:
                    already_closed, turn._closed = turn._closed, True
                close_hub(turn.spec.call_id, "context_capacity", detail=CONTEXT_CAPACITY_DETAIL)
                if not already_closed:
                    turn.publish({"type": "conversation_finished", "chat_id": turn.spec.chat_id,
                                  "call_id": turn.spec.call_id, "task_id": admitted["task"]["id"],
                                  "reason": "context_capacity"})
                return RealtimeSetupError("context_capacity", CONTEXT_CAPACITY_DETAIL)
            busy = isinstance(e, StoreBusy) or isinstance(e.__cause__, StoreBusy)
            close_hub(turn.spec.call_id, "store_busy" if busy else "call_start_failed",
                      detail=("The local store is busy. Please try the call again." if busy else
                              "The call could not start. Please try again."))


def _report_call_context_refusal(admitted, unfit, refusal) -> None:
    """Finish an admitted call whose ordinary context build failed before the host existed."""
    from dataclasses import asdict
    from ouroboros.agent_task_pipeline import emit_task_results

    task, agent, turn = admitted["task"], admitted["agent"], admitted["realtime_turn"]
    measurement = {key: getattr(unfit, key) for key in (
        "fixed_input_tokens", "memory_view_tokens", "estimated_input_tokens", "raw_input_tokens",
        "measurement_density", "response_reserve_tokens", "work_room_tokens", "work_room_deficit_tokens")}
    measurement.update(capacity_total_tokens=unfit.window_tokens, capacity_deficit_tokens=unfit.deficit_tokens)
    event = {"type": "call.fit_refused", "ts": utc_now_iso(), "task_id": task["id"],
             "call_id": turn.spec.call_id, "reason": refusal.code, "stage": "task_preparation",
             "fit": {"measurement": measurement}, "physical": unfit.physical,
             "deficit_tokens": unfit.deficit_tokens, "route_facts": asdict(turn.catalog_model.facts),
             "units": turn.catalog_model.facts.basis}
    append_jsonl(agent.env.drive_root / "logs" / "events.jsonl", event)
    events = []
    event_queue = _TurnEventQueue(_pool().get_event_q(), task["id"], admitted["chat_id"],
                                initiator=str((admitted.get("task_metadata") or {}).get("initiator") or ""))
    emit_task_results(
        agent.env, agent.memory, agent.llm, events, task, str(refusal),
        {"terminal_origin": "host_notice", "execution_status": "infra_failed", "reason_code": refusal.code},
        {"reasoning_notes": [], "tool_calls": [], "fit_refusal": event},
        getattr(agent, "_task_started_ts", time.time()), agent.env.drive_root / "logs",
        ctx=getattr(agent.tools, "_ctx", None), event_queue=event_queue)
    for event in events:
        _pool().get_event_q().put(stamp_finalization_enqueue(event_queue.stamp(event)))
    if admitted.get("activity") is not None:
        admitted["activity"].reply_handed_off = True




# ---------------------------------------------------------------------------
# Stop of an inline executor's turn (a lane row is stopped by pooled custody)
# ---------------------------------------------------------------------------

DIRECT_TURN_STOP_GONE = "gone"        # the turn had already ended before the stop could be armed
DIRECT_TURN_STOP_ENDED = "ended"      # armed, and the turn reached its boundary within the wait
DIRECT_TURN_STOP_LIVE = "live"        # armed, still inside a step (the sweep retries custody)


def stop_direct_chat_turn(task_id: str, turn: Dict[str, Any], *, deliver: bool = True) -> str:
    """Stop an INLINE executor's turn COOPERATIVELY; a typed outcome.

    Only the two disclosed inline executors own a registered actor inside the
    supervisor; a conversation-lane turn is a RUNNING row whose carrier the pooled
    custody kills after the same finalize intent. The lane writes the typed
    ``finalize_now`` control (``REASON_OWNER_STOPPED_DIRECT_TURN``) to the canonical
    drive's owner mailbox — the one the turn's loop drains at every round boundary,
    where it ends the turn with ZERO further model calls — then waits the short
    config-owned bound for the turn to reach that boundary (the same custody pass
    runs on the supervisor sweep, so the wait stays short).

    Two things the write must never do: arm a turn that is already gone (a
    turn that ended between custody's ownership read and this write would get
    a false owner toast over an answer that already landed, and an orphaned
    control in a mailbox the settled-cleanup may already have pruned) — so
    liveness is re-read immediately before the write and ``GONE`` names that
    lane; and re-arm on a retry: the control is written ONCE per turn (the
    turn's stamp is the latch, as the RUNNING row is for a pooled task) and a
    custody pass that finds the stamp already there answers WITHOUT waiting
    (the supervisor tick must not spend the bound on every pass).
    ``deliver=False`` (a cascade sweep, which speaks for the tree once)
    suppresses the owner toast.
    """
    from ouroboros.config import get_direct_turn_stop_wait_sec
    from supervisor import queue as q
    from supervisor import workers
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN
    from supervisor.task_reaper import request_finalization_grace

    if turn.get("stop_control_msg_id"):
        return DIRECT_TURN_STOP_LIVE if workers.direct_chat_turn(task_id) is not None else DIRECT_TURN_STOP_ENDED

    def _write_control(live_turn: Dict[str, Any]) -> str:
        # The canonical drive: an inline turn runs on the main data root, and
        # its loop drains that root's owner mailbox (the same root custody
        # settles the intent on).
        return request_finalization_grace(
            pathlib.Path(q.DRIVE_ROOT), task_id, REASON_OWNER_STOPPED_DIRECT_TURN,
            chat_id=int(live_turn.get("chat_id") or 0), stamp=int(time.time()),
            toast_text=(
                f"⏹ The owner stopped chat turn {task_id}; it ends at its next "
                "step without further work."
            ) if deliver else "",
            quiet=not deliver,
        )

    # Atomic against the turn's own completion (its admission lock): a turn
    # that ended first gets no control and no toast.
    if workers.arm_direct_chat_turn(task_id, _write_control) is None:
        return DIRECT_TURN_STOP_GONE
    deadline = time.monotonic() + float(get_direct_turn_stop_wait_sec())
    while workers.direct_chat_turn(task_id) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    return DIRECT_TURN_STOP_LIVE if workers.direct_chat_turn(task_id) is not None else DIRECT_TURN_STOP_ENDED


__all__ = ["DIRECT_TURN_STOP_ENDED", "DIRECT_TURN_STOP_GONE", "DIRECT_TURN_STOP_LIVE", "EXECUTOR_INLINE_CALL",
           "EXECUTOR_INLINE_DEGRADED", "InlineCarrier", "arm_inline_stop", "inline_vitals", "refused_carrier_reason",
           "run_inline_turn", "start_inline_turn", "stop_direct_chat_turn"]
