"""A Host-admitted Presence turn's body in a conversation-lane carrier (owner decision 1A).

The Host (``gateway/host_service_places``) keeps authentication, capture, dedup, its
in-flight reservation and ``OUROBOROS_PRESENCE_MAX_ACTIVE``; ``dispatch_presence_turn``
admits the body as a lane RUNNING row (origin ``presence``, carrier ``conversation``) and
``carry_presence_turn`` runs it in whichever process carries the row: it re-admits from
the durable facts (``admit_presence_turn``, ``presence_turn_event``), acquires the
``PresenceTurnLease`` itself (version 2 never waits: a busy resource is a typed retry),
runs ``presence_runner.run_presence_turn`` and reports ONE ``presence_turn_settled`` frame —
the result, or the error by value — that resolves the Host's ``PresenceTurnExecution``
future and releases the row and the slot; a continuing author's initial envelope is a
``presence_turn_initial`` frame (``FrameExecution``). A carrier that ends without that frame
(a Stop's kill, a crash) leaves its waiter to the server's published terminal
(``settle_ended_turn``, from the ``task_done`` dispatch). Retry identity and ``heard_replay``
dedup stay the gate's, inside the body.

A turn the owner moved to a Project (``moved_text_turn``) takes that work's lifecycle with no
generic owner-wait capability: its exact pause on the owner and money/authority rails parks the
SAME id in the queue as the turn's own record (``saved_turn_record``), its waiting transport hears
the park as soon as that custody is durable (``answer_parked_turn``) while the carrier keeps the
conversation and its slot until its own effects settled (``park_moved_turn``), and a Resume or a
restart's return runs the saved work in the worker the queue gives it (``carry_saved_turn``),
which takes the conversation and a slot before any restore or call. Each settlement frame names
its sender's slot, process, attempt and park, so a late one never touches a newer execution.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_events import PresenceTurnError, PresenceTurnEvent
from ouroboros.presence_executions import (
    PresenceTurnExecution, PresenceTurnInterrupted, PresenceTurnNotStarted, _find_execution,
)
from ouroboros.presence_runner import (
    PresenceTurnLease, PresenceTurnResult, _configured_gate, _task_id, presence_chat_id, presence_event_identity,
    run_presence_turn,
)

log = logging.getLogger(__name__)

PRESENCE_RUN_KEY = "_presence_run"  # the Host's admission facts the carrier re-admits from (durable bindings)
_RUNNERS: dict = {}  # data root -> a runner injected in this process (``host_service.create_host_service_app``), tests only


def set_presence_runner(drive_root: Any, runner: Any) -> None:
    """Replace the body's runner for turns of ``drive_root`` in THIS process (a Host test's seam); None restores
    the module's own. Keyed by root: Hosts of different roots in one process never share it."""
    key = str(Path(drive_root).resolve())
    if runner is None:
        _RUNNERS.pop(key, None)
    else:
        _RUNNERS[key] = runner


def _runner(drive_root: Any) -> Any:
    return _RUNNERS.get(str(Path(drive_root).resolve())) or run_presence_turn


class FrameExecution:
    """The carrier's stand-in for the Host's ``PresenceTurnExecution``: an initial envelope published
    here becomes a ``presence_turn_initial`` frame the Host resolves its future from."""

    def __init__(self, turn_id: str, sink: Any) -> None:
        self.turn_id, self._sink = turn_id, sink
        self.initial = concurrent.futures.Future()
        self.initial.set_running_or_notify_cancel()

    def publish_initial(self, value: Any, *, return_capacity: bool = True) -> bool:
        # Two-phase like the Host's own execution: an author that answered early (``return_capacity=False``)
        # sets the envelope here and returns its capacity only at its first real park, whose own
        # ``publish_initial`` must still reach the Host to release it, though the envelope is already out.
        first = not self.initial.done()
        if first:
            self.initial.set_result(value)
        if first or return_capacity:
            self._sink.put({"type": "presence_turn_initial", "turn_id": self.turn_id, "task_id": self.turn_id,
                            "result": _result_payload(value), "return_capacity": bool(return_capacity)})
        return first

    def return_capacity(self, fallback: Any = None) -> None:
        pass


def _result_payload(result: Any) -> dict[str, Any]:
    """The result by value: every field of ``PresenceTurnResult``; a runner that states fewer (a double)
    leaves the dataclass's defaults for the rest."""
    defaults = PresenceTurnResult(**{key: getattr(result, key) for key in ("outcome", "text", "task_id")
                                     if hasattr(result, key)})
    payload = {key: getattr(result, key, getattr(defaults, key)) for key in PresenceTurnResult.__dataclass_fields__}
    if payload.get("reply_anchor") is not None:
        payload["reply_anchor"] = dict(payload["reply_anchor"])
    return payload


def _error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, PresenceTurnError):
        return {"kind": "presence", "code": exc.code, "field": exc.field, "turn_ref": exc.turn_ref,
                "work_ref": exc.work_ref, "disposition": exc.disposition,
                "delivery_reporting_version": exc.delivery_reporting_version,
                "attachment_manifest": list(exc.attachment_manifest),
                "retry_after": getattr(exc, "retry_after", None), "custody": getattr(exc, "custody", "")}
    return {"kind": "exception", "type": type(exc).__name__, "message": str(exc)[:2000]}


def _error_from_payload(payload: Mapping[str, Any]) -> BaseException:
    if payload.get("kind") == "presence":
        error = PresenceTurnError(str(payload.get("code") or "presence_turn_failed"), str(payload.get("field") or ""),
                                  attachment_manifest=payload.get("attachment_manifest") or (),
                                  turn_ref=str(payload.get("turn_ref") or ""), work_ref=str(payload.get("work_ref") or ""),
                                  disposition=str(payload.get("disposition") or ""),
                                  delivery_reporting_version=int(payload.get("delivery_reporting_version") or 0))
        if payload.get("retry_after") is not None:
            error.retry_after = payload["retry_after"]
        if payload.get("custody"):
            error.custody = str(payload["custody"])
        return error
    return RuntimeError(f"{payload.get('type') or 'Exception'}: {payload.get('message') or ''}")


def _capacity_full(busy: str) -> PresenceTurnError:
    """Version 2's typed refusal of a busy gate: the busy resource named; a conversation retries after 1 s,
    the active-turn cap after 5 s."""
    error = PresenceTurnError("presence_capacity_full", busy, disposition="retry")
    error.retry_after = 1 if busy == "conversation" else 5
    return error


def presence_seed_task(*, admission: PresenceAdmission, event: PresenceTurnEvent, skill_name: str,
                       event_payload: Mapping[str, Any], staged_files: Sequence[Any], gate_wait: bool) -> dict[str, Any]:
    """The lane's admitted seed of a Host turn: identity, address, the correspondent's words, and the
    facts the carrier re-admits from (the authenticated skill, the binding, the validated event payload).

    The carrier rebuilds the admission from the durable bindings (``admit_presence_turn``) and the
    event from the payload (``presence_turn_event``): nothing process-local travels.
    """
    identity = presence_event_identity(admission.binding_id, event)
    payload = {key: value for key, value in dict(event_payload).items() if key != "identity"}
    return {
        "id": _task_id(admission, event), "type": "presence", "source": "presence",
        "chat_id": presence_chat_id(event.conversation_key), "text": str(event.text or "").strip(),
        "_is_direct_chat": True, "_presence_turn": True,
        "metadata": {"source": "presence", "client_message_id": event.source_event_id,
                     "presence_event_identity": identity, "presence_request_waits": True,
                     "presence": {"binding_id": admission.binding_id, "transport_skill": admission.transport_skill,
                                  "event": {"source_event_id": event.source_event_id, "provider": event.provider,
                                            "account_id": event.account_id, "conversation_id": event.conversation_id,
                                            "thread_id": event.thread_id, "conversation_key": event.conversation_key}}},
        PRESENCE_RUN_KEY: {
            "skill_name": str(skill_name), "binding_id": admission.binding_id, "event": payload,
            "identity": dict(event.identity) if isinstance(event.identity, Mapping) else None,
            "staged_files": [str(path) for path in staged_files],
            "vouched_facts": list(event.facts or ()),  # the Host admitted these facts (its declared signals too)
            "delivery_reporting_version": int(event.delivery_reporting_version or 0),
            "continuation_version": int(event.continuation_version or 0),
            "allowance": dict(admission.allowance) if isinstance(admission.allowance, Mapping) else None,
            "gate_wait": bool(gate_wait), "request_waits": True,  # a Host request waits on this turn's result
        },
    }


def dispatch_presence_turn(execution: PresenceTurnExecution, *, drive_root: Path, skill_name: str,
                           admission: PresenceAdmission, event: PresenceTurnEvent, event_payload: Mapping[str, Any],
                           staged_files: Sequence[Any], gate_wait: bool) -> None:
    """Admit a Host turn's body to the conversation lane (under Q); the carrier runs it.

    The Host keeps authentication, capture, dedup and its in-flight reservation; the gate
    (conversation lock, ``OUROBOROS_PRESENCE_MAX_ACTIVE`` slot) is acquired by the carrier.
    The row is admitted under Q and the gate lock, its barrier runs outside both, the carrier
    takes it under Q again (``dispatch_admitted_turn``). A server whose pool has not armed starts
    nothing: ``PresenceTurnNotStarted`` (a typed retry), no row over an unrestored saved queue
    (#1952); a Host without a queue of its own keeps the disclosed degraded executor (2A).
    """
    from supervisor import queue, workers
    from supervisor.conversation_lane import LANE_CONVERSATION, admit_turn, armed, dispatch_admitted_turn
    from supervisor.conversation_lane import replenish_after_claim, spawn_refusal
    from supervisor.inline_carrier import InlineExecutorUnavailable

    # A server whose carrier spawn failed keeps the disclosed degraded executor (the observed physical fact).
    if not armed() and queue.owns_live_queue(drive_root) and not spawn_refusal():
        raise PresenceTurnNotStarted("the worker pool of this server has not started yet; retry the event")
    if not gate_wait:
        # Version 2 never waits in the gate, and a busy gate costs no lane admission (no row, no barrier, no
        # spare claimed, no carrier spawned): one non-blocking pass of the carrier's own acquisition, released.
        probe, busy = _configured_gate(Path(drive_root)).try_admit(event.conversation_key)
        if probe is None:
            raise _capacity_full(busy)
        probe.release()
    task = presence_seed_task(admission=admission, event=event, skill_name=skill_name, event_payload=event_payload,
                              staged_files=staged_files, gate_wait=gate_wait)
    # The Host's data root is this turn's authority, whichever carrier runs it: a pooled worker carries the
    # server's own root (the same one), and the in-process degraded executor must re-admit and write against
    # the Host's root, never the pool's global DRIVE_ROOT (unset, or another root, in a standalone Host).
    task["drive_root"] = str(drive_root)
    with queue._queue_lock, workers._repo_writer_gate_lock:
        held = workers.RUNNING.get(str(task["id"]))
        if any(str(row.get("id") or "") == str(task["id"]) for row in queue.PENDING) or (
                isinstance(held, dict) and held.get("lane") != LANE_CONVERSATION):
            # The queue keeps this id's saved work (a moved turn's park, its Resume or a restart's return): the
            # retry is answered from that custody, never a second lane row or author for the same id.
            refusal = PresenceTurnError("presence_turn_parked", "source_event_id", turn_ref=str(task["id"]),
                                        disposition="retained")
            refusal.custody = "queued"
            raise refusal
        meta = admit_turn(task, chat_id=int(task["chat_id"]), client_message_id=event.source_event_id,
                          quiet=True, reply_expected=False)
    try:
        dispatch_admitted_turn(meta)  # the barrier outside Q, then the carrier (or the degraded executor)
    except InlineExecutorUnavailable as exc:
        # Nothing ran and no carrier will come to this process: the row leaves now and the Host answers
        # a typed retry; its reservation returns with the settlement, the gate was never taken.
        with queue._queue_lock:
            if workers.RUNNING.get(str(task["id"])) is meta:
                workers.RUNNING.pop(str(task["id"]))
            queue.persist_queue_snapshot(reason="presence_turn_not_started")
        raise PresenceTurnNotStarted("presence turn thread could not start") from exc
    replenish_after_claim()  # the spare the turn took is replaced now, or its own carrier started now (4A)


def carry_presence_turn(task: dict[str, Any], out_q: Any, repo_dir: Any, drive_root: Any, *,
                        owner_wait_callback: Any = None, worker_id: Any = None) -> list:
    """The carrier's body for a Host turn: re-admit from durable facts, take the gate, run, report.

    Every outcome leaves as ONE ``presence_turn_settled`` frame (the result, or the error by
    value); the Host resolves its waiter from it and releases the RUNNING row and the slot. A
    turn's saved record (no admission facts: a moved turn's park, Resume or restart return) is
    ``carry_saved_turn``'s, whose events go back to ``run_worker_task``.
    """
    from ouroboros.owner_wait import governing_continuation

    if task.get(PRESENCE_RUN_KEY) is None or (
            governing_continuation(task) and _authorized_continuation(task)
            and moved_text_turn(drive_root, str(task.get("id") or ""), task.get("metadata"))):
        return carry_saved_turn(task, out_q, repo_dir, drive_root, owner_wait_callback=owner_wait_callback,
                                worker_id=worker_id)
    from dataclasses import replace

    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_events import presence_event_origin, presence_turn_event

    root = Path(drive_root)
    facts = task.pop(PRESENCE_RUN_KEY, None) or {}
    turn_id = str(task.get("id") or "")
    carrier_owner = task.get("_execution_owner") if isinstance(task.get("_execution_owner"), dict) else None
    physical_id, result, error = turn_id, None, None
    try:
        admission = admit_presence_turn(drive_root=root, authenticated_transport_skill=str(facts.get("skill_name") or ""),
                                        binding_id=str(facts.get("binding_id") or ""),
                                        **presence_event_origin(facts.get("event")))
        if facts.get("allowance") is not None:
            admission = replace(admission, allowance=dict(facts["allowance"]))
        event = presence_turn_event(facts.get("event"), admission,
                                    reporting_version=int(facts.get("delivery_reporting_version") or 0),
                                    continuing=int(facts.get("continuation_version") or 0),
                                    vouched_facts=facts.get("vouched_facts") or ())
        if facts.get("identity") is not None:
            event = replace(event, identity=dict(facts["identity"]))
        gate = _configured_gate(root)
        if facts.get("gate_wait", True):
            from ouroboros.cancel_intents import has_active_intent

            def stopped() -> bool:
                try:
                    return bool(has_active_intent(root, turn_id, strict=False))
                except Exception:
                    return False

            resources = gate.acquire_resources(event.conversation_key, stopped)
            if resources is None:
                raise PresenceTurnError("presence_turn_cancelled", "conversation_key", turn_ref=turn_id)
            lease = PresenceTurnLease(event.conversation_key, resources, gate)
        else:  # version 2 never waits: a slot taken since the Host's probe is the same typed refusal
            lease, busy = gate.try_admit(event.conversation_key)
            if lease is None:
                raise _capacity_full(busy)
        lease.execution = FrameExecution(turn_id, out_q)
        try:
            def factory(**kwargs):
                from ouroboros.agent import make_agent

                return make_agent(**kwargs)  # no generic owner wait: Pause, budget pause, detach stay closed

            runner = _runner(root)
            # The turn facts, never the physical carrier's. An injected runner (a test's seam, as the
            # retired unarmed path's runner) supplies its own repo/drive/factory; only the module runner
            # takes this carrier's, so an injected wrapper is never handed a second repo_dir/drive_root.
            turn_kwargs = dict(admission=admission, event=event,
                               staged_files=[Path(p) for p in facts.get("staged_files") or []],
                               event_queue=out_q, admitted=lease, carrier_owner=carrier_owner,
                               event_sink=lambda rows: [out_q.put(dict(r)) for r in rows],
                               request_waits=bool(facts.get("request_waits", True)))
            if runner is run_presence_turn:
                # This carrier's own runner: its review/private-question park lends the slot through the carrier's
                # pooled wait, bound to the lease (an injected runner keeps its own in-process wait, as before).
                lease.review_wait = owner_wait_callback
                result = runner(repo_dir=Path(repo_dir), drive_root=root, agent_factory=factory, **turn_kwargs)
            else:
                result = runner(**turn_kwargs)
            physical_id = str(result.task_id or turn_id)
        finally:
            lease.release()
    except BaseException as exc:
        error = exc
        physical_id = str(getattr(exc, "turn_ref", "") or turn_id)
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            out_q.put({"type": "presence_turn_settled", "turn_id": turn_id, "task_id": turn_id,
                       "physical_id": physical_id,
                       "result": _result_payload(result) if result is not None else None,
                       "error": _error_payload(error) if error is not None else None,
                       **_frame_identity(task, worker_id, error)})
        except Exception:
            log.warning("presence turn %s could not report its settlement", turn_id, exc_info=True)
    return []


def _frame_identity(task: Mapping[str, Any], worker_id: Any, error: Any) -> dict[str, Any]:
    """Who sends a settlement: the slot, its process and the attempt, and the park it ends with (a pause id;
    ``before_task_start`` for a start that never ran its body). An inline caller with no slot names none."""
    if worker_id is None:
        return {}
    return {"worker_id": worker_id, "pid": os.getpid(), "attempt": int(task.get("_attempt") or 1),
            "parked": str(getattr(error, "parked", "") or "")}


def handle_presence_turn_initial(evt: dict[str, Any], ctx: Any) -> None:
    """A carrier published a continuing author's initial envelope: the Host's waiter takes it."""
    _registry, execution = _find_execution(str(evt.get("turn_id") or ""))
    if execution is None or not isinstance(evt.get("result"), dict):
        return
    execution.publish_initial(PresenceTurnResult(**evt["result"]), return_capacity=bool(evt.get("return_capacity", True)))


def handle_presence_turn_settled(evt: dict[str, Any], ctx: Any) -> None:
    """A carrier's turn ended: resolve the Host's waiter, release the lane row and its slot.

    A frame naming its sender (``_frame_identity``) acts on that sender's own execution only: its row
    when RUNNING still binds the same slot, attempt and process, and its slot while that slot is the same
    live process and no newer row of the id is bound to it. A parked frame's custody already moved to the
    queue (``events_budget.install_exact_budget_pause``), so it pops no row and closes no intake: a
    later execution of the same id may own both. A frame naming none keeps the older by-id release.
    """
    from supervisor import queue
    from supervisor.queue import _queue_lock

    turn_id = str(evt.get("turn_id") or "")
    registry, execution = _find_execution(turn_id)
    if execution is not None and registry is not None:
        error_payload = evt.get("error")
        if isinstance(error_payload, dict):
            registry.settle_from_frame(execution, error=_error_from_payload(error_payload))
        elif isinstance(evt.get("result"), dict):
            registry.settle_from_frame(execution, result=PresenceTurnResult(**evt["result"]))
        else:
            registry.settle_from_frame(execution, error=PresenceTurnNotStarted("presence turn settled without a result"))
    named = "worker_id" in evt
    with _queue_lock:
        current = ctx.RUNNING.get(turn_id)
        if not named:
            meta = ctx.RUNNING.pop(turn_id, None)
            wid = meta.get("worker_id") if isinstance(meta, dict) else None
        else:
            wid = evt.get("worker_id")
            sender = ctx.WORKERS.get(wid) if wid is not None else None  # a respawned slot is another process's
            own = (isinstance(current, dict) and not evt.get("parked") and current.get("worker_id") == wid
                   and int(current.get("attempt") or 1) == int(evt.get("attempt") or 1)
                   and (sender is None or _same_process(sender, evt.get("pid"))))
            meta = ctx.RUNNING.pop(turn_id, None) if own else None
        slot = ctx.WORKERS.get(wid) if wid is not None else None
        newer = ctx.RUNNING.get(turn_id)
        if (slot is not None and slot.busy_task_id == turn_id and not getattr(slot, "reaping", False)
                and (not named or _same_process(slot, evt.get("pid")))
                and not (isinstance(newer, dict) and newer.get("worker_id") == wid)):
            slot.busy_task_id = None
    if meta is not None or not named:
        try:  # the row ended here: a remote Host offers nothing more to this author, whatever its carrier did
            from ouroboros.conversation_lease import close_author_intake

            close_author_intake(Path(getattr(ctx, "DRIVE_ROOT", None) or queue.DRIVE_ROOT), turn_id)
        except Exception:
            log.debug("author intake of %s not closed at settlement", turn_id, exc_info=True)
    if meta is not None:
        try:
            queue.persist_queue_snapshot(reason="presence_turn_settled")
        except Exception:
            log.debug("presence settlement snapshot failed for %s", turn_id, exc_info=True)


def _same_process(slot: Any, pid: Any) -> bool:
    """The slot is still the process that sent the frame (a respawned slot is another one); an inline executor's
    pseudo-slot has no pid of its own and is its server's."""
    from supervisor.worker_pool_lifecycle import is_worker_slot

    try:
        return not is_worker_slot(slot) or int(getattr(slot.proc, "pid", 0) or 0) == int(pid or 0)
    except Exception:
        return False


def settle_ended_turn(turn_id: str) -> bool:
    """The server published this turn's terminal and released its row (``task_done``): a Host waiter still live
    heard nothing from the carrier (a Stop's kill, a crash, a pool stop never send ``presence_turn_settled``).

    It settles now with ``PresenceTurnInterrupted``: its in-flight reservation returns, retries stop joining a
    dead execution, and the transport's resend reads the turn's durable row (a replay, or the retained refusal).
    A settlement the carrier sent before it ended rides the same control lane ahead of this terminal and
    already retired the execution, so it is never overwritten. Called after the row left RUNNING, so a resend
    admits a fresh row. Never raises."""
    try:
        registry, execution = _find_execution(str(turn_id or ""))
        if execution is None or registry is None:
            return False
        registry.settle_from_frame(execution, error=PresenceTurnInterrupted(
            f"presence turn {turn_id} ended in its carrier without a settlement; resend the same event"))
        return True
    except Exception:
        log.warning("presence turn %s: its Host waiter could not be settled at its terminal", turn_id, exc_info=True)
        return False


# --- a moved turn's Project lifecycle (#2566) --------------------------------------------------------------

_TERMINAL = frozenset({"completed", "failed", "cancelled"})
# A park's hold reads the author's process scope (``tool_custody.task_scope_blockers``). Once a second: one identity
# read of the members it knows, the whole-table census (on macOS one ``ps -E`` of every process) only once they ended.
_HOLD_POLL_SEC = 1.0
_RUNS_RECHECK_SEC = 5.0
# Queue facts a saved record keeps from the row it replaces: the attempt, the park/grant/recovery locators.
_QUEUE_FACTS = ("_attempt", "_budget_pause", "_budget_pause_resume", "_working_recovery", "_owner_wait_resume",
                "_budget_pause_hold", "_resume_selection", "_continuation_prepared", "queued_at", "priority",
                "_queue_seq", "root_task_id", "drive_root", "budget_drive_root", "detached_at", "detached_by",
                "_execution_handoff_sec", "_execution_consumed_sec", "execution_timeout_sec", "admitted_dispatch",
                "budget_resumed_at")
# What the durable row the turn's own start wrote holds of its built task (``agent._persist_running_record``).
_STORED_FIELDS = ("chat_id", "actor_id", "project_id", "workspace_root", "workspace_mode", "memory_mode",
                  "task_contract", "metadata", "origin_message_ref", "origin_message_text", "model",
                  "use_local_model", "root_task_id")
_CARRIER: contextvars.ContextVar = contextvars.ContextVar("presence_saved_carrier", default=None)


def moved_text_turn(drive_root: Any, task_id: str, metadata: Any = None, *, unreadable: bool = False) -> bool:
    """A Presence turn the owner moved to a Project (a binding that is not ``origin_bound``) and carried as text:
    its saved work takes that work's lifecycle. An unmoved turn and a live call (its ``execution_transport``) keep
    their own interruption contract; an unreadable binding is not a move, unless the caller says what it should read
    as (``unreadable``: custody asks whether it may be one). Never raises."""
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("execution_transport") is not None or not isinstance(metadata.get("presence"), Mapping):
        return False
    try:
        from ouroboros.projects_registry import project_binding_for_task

        binding = project_binding_for_task(Path(drive_root), str(task_id or ""), strict=True) or {}
    except Exception:
        return unreadable
    return bool(binding.get("project_id")) and not binding.get("origin_bound")


def saves_exact_pause(ctx: Any, rail: str) -> bool:
    """Whether a carried Presence author saves this rail's exact pause with no generic owner-wait capability:
    a moved text turn, on the owner's Pause and the money/authority rails, never the model's own sleep or
    detach. Read at the request, not at the carrier's start: the owner can move a turn while it runs."""
    from ouroboros.budget_pause import RAIL_MODEL_SLEEP

    if rail == RAIL_MODEL_SLEEP:
        return False
    from ouroboros.tool_access import canonical_data_root

    return moved_text_turn(canonical_data_root(ctx), str(getattr(ctx, "task_id", "") or ""),
                           getattr(ctx, "task_metadata", None))


def owner_allowance_lift(task: Mapping[str, Any], drive_root: Any) -> dict[str, Any]:
    """The owner's choice of 2026-11-02 (quiz f43a393a, «Мой явный Resume переводит работу на обычный бюджет
    проекта»): an explicit Resume through the owner's own door (``control_actor.owner_door``) moves an originally
    autonomous moved Presence turn off ITS producer allowance onto the ordinary project budget, once. The record
    to keep, else ``{}``: a mind's or automatic Resume, an unbound call, owner work at entry (C8), an unmoved or
    call turn, or no allowance left to lift. Spend so far, the tree and group caps, the shared Total budget,
    deadlines, Stop/Panic and unknown outcomes are the grant's own checks and stay."""
    from ouroboros.consciousness_authority import WORK_KIND_OWNER, work_kind
    from ouroboros.control_actor import owner_door_bound
    from ouroboros.task_pacing import recorded_producer_ceiling

    door, metadata = owner_door_bound(), task.get("metadata")
    ceiling = recorded_producer_ceiling(dict(task)) if isinstance(metadata, Mapping) else None
    if not (door and ceiling is not None and work_kind(metadata) != WORK_KIND_OWNER
            and moved_text_turn(drive_root, str(task.get("id") or task.get("task_id") or ""), metadata)):
        return {}
    return {"by": door, "ceiling_usd": ceiling}


def lift_allowance(task: dict[str, Any], lift: Mapping[str, Any]) -> dict[str, Any]:
    """Keep ``lift`` (``owner_allowance_lift``) on a queue row: its allowance moves into the record, so every later
    start, restore, member and Continue reads no producer ceiling. Returns the prior fields a rollback restores."""
    if not lift:
        return {}
    prior = {key: copy.deepcopy(task[key]) for key in ("metadata", "root_cost_ceiling_usd") if key in task}
    task.pop("root_cost_ceiling_usd", None)
    metadata = task["metadata"] = dict(task.get("metadata") or {})
    metadata.pop("root_cost_ceiling_usd", None)
    metadata["producer_allowance_lifted"] = dict(lift)
    return prior


def lifted_record(row: Mapping[str, Any], lift: Mapping[str, Any]) -> dict[str, Any]:
    """A durable row's copy with ``lift`` kept in place of its allowance (``lift_allowance``)."""
    record = dict(row)
    lift_allowance(record, lift)
    return record


def kept_allowance_lift(row: Mapping[str, Any]) -> dict[str, Any]:
    """The owner's decision a row already keeps, else ``{}``."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    kept = metadata.get("producer_allowance_lifted")
    return dict(kept) if isinstance(kept, Mapping) else {}


def owner_resume_decision(task: Mapping[str, Any], drive_root: Any) -> dict[str, Any]:
    """The owner's Resume releasing a queue row with no grant to record the decision on (a held row, a never-started
    root, a start that never ran its body): the same decision, judged on the record its start wrote when the row is
    a lane seed (``saved_turn_record`` rebuilds it with this lift)."""
    from ouroboros.control_actor import owner_door_bound

    if not owner_door_bound():
        return {}
    record: Mapping[str, Any] = task
    if not _saved_record(task):
        from ouroboros.task_results import load_task_result

        record = load_task_result(Path(drive_root), str(task.get("id") or "")) or {}
    return owner_allowance_lift(record, drive_root)


def owner_resume_lift(task: dict[str, Any], drive_root: Any) -> dict[str, Any]:
    """That decision kept before its release, when only that publication can still refuse and take it back
    (``budget_resume.take_back_owner_resume_lift``): the durable row every Stop, Continue, member and restart reads
    takes it first, then the queue row. A store that refuses moves nothing and returns the door's refusal; else ``{}``."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document

    def keep(row: dict[str, Any]) -> Any:
        require_writable_task_result_schema(row)
        return stamp_task_result_schema(lifted_record(row, lift)) if row else None

    try:
        lift = owner_resume_decision(task, drive_root)
        if lift:
            patch_document(Path(drive_root), str(task.get("id") or ""), keep)
    except Exception as exc:
        return {"ok": False, "error": "grant_not_recorded", "detail": str(exc)[:200]}
    lift_allowance(task, lift)
    return {}


def lift_live_allowance(ctx: Any, lift: Any) -> None:
    """A warm Resume's lift where the stack continues: its own task metadata, then its ceiling read again by the restore
    rule (a producer's number goes, disclosed in the basis). The loop decides on this ceiling from that boundary on; the
    records a restart or a Continue rebuild from took the decision with the release (``budget_resume``)."""
    from dataclasses import asdict

    from ouroboros.task_pacing import CostCeiling, restore_cost_ceiling

    metadata = getattr(ctx, "task_metadata", None)
    if isinstance(lift, Mapping) and lift and isinstance(metadata, dict) and not metadata.get("producer_allowance_lifted"):
        metadata.pop("root_cost_ceiling_usd", None)
        metadata["producer_allowance_lifted"] = dict(lift)
        held = getattr(ctx, "_cost_ceiling", None)
        ctx._cost_ceiling = restore_cost_ceiling(ctx, asdict(held) if isinstance(held, CostCeiling) else None)


def _identity(row: Any) -> str:
    metadata = row.get("metadata") if isinstance(row, Mapping) and isinstance(row.get("metadata"), Mapping) else {}
    return str(metadata.get("presence_event_identity") or "")


def saved_turn_record(task: Mapping[str, Any], drive_root: Any, *, carried: Any = None,
                      stored: Any = None) -> dict[str, Any] | None:
    """The queue's record of a Presence turn's saved work: the turn's own built task (contract and ceiling, folder,
    attachments, metadata), never the sparse lane seed or its admission facts. ``carried`` is the record its own pause
    event carries (``budget_pause.pause_event``), else the durable row its start wrote (``stored``: one its caller read
    strictly, not read again). The source room's ``chat_id`` stays (every room reader resolves the binding first) and
    a folder its Place chose stays the explicit resource readmission pins. None when neither is a complete record."""
    task_id, identity = str(task.get("id") or ""), _identity(task)
    base: dict[str, Any] | None = None
    if identity and isinstance(carried, Mapping) and str(carried.get("id") or "") == task_id \
            and _identity(carried) == identity:
        base = copy.deepcopy(dict(carried))
    elif identity:
        if stored is None:
            try:
                from ouroboros.task_results import load_task_result

                stored = load_task_result(Path(drive_root), task_id, strict=True) or {}
            except Exception:
                stored = {}
        if _identity(stored) == identity and isinstance(stored.get("task_contract"), Mapping):
            base = {key: copy.deepcopy(stored[key]) for key in _STORED_FIELDS if stored.get(key) not in (None, "")}
            presence = (base.get("metadata") or {}).get("presence") or {}
            base["text"] = str(presence.get("observed_text") or task.get("text") or "")
    if base is None:
        return None
    for key in (PRESENCE_RUN_KEY, "_conversation_prepare", "_execution_owner", "execution_owner", "worker_id"):
        base.pop(key, None)
    base.update({key: copy.deepcopy(task[key]) for key in _QUEUE_FACTS if key in task})
    base.update(id=task_id, type="presence", source="presence", _is_direct_chat=True, _presence_turn=True,
                chat_id=task.get("chat_id", base.get("chat_id")), context_requires_development=False)
    metadata = base["metadata"] = dict(base.get("metadata") or {})
    metadata["presence_request_waits"] = False  # no request waits on saved work: its poll and replay read the row
    if str(base.get("workspace_root") or "").strip() and not metadata.get("resource_intent"):
        metadata["resource_intent"] = {"kind": "explicit_resource"}
    if kept := kept_allowance_lift(task):  # the owner's lift the queue row keeps
        lift_allowance(base, kept)
    return base


def _saved_record(task: Mapping[str, Any]) -> bool:
    """Whether a queue row already is a turn's saved record: ``saved_turn_record`` alone writes this fact. A lane seed
    carries a default task contract of its own, so a contract proves nothing about the turn's built capabilities."""
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    return metadata.get("presence_request_waits") is False and isinstance(task.get("task_contract"), Mapping)


def adopt_saved_records(rows: Any, drive_root: Any) -> None:
    """Recovery keeps the complete record: a restored moved text turn's row is still the seed its admission wrote, so
    before any gate reads it (its Project's readmission pins the folder from these facts, its start reads the allowance
    and ceiling) it becomes the turn's own built record, or holds typed when none is readable. A row already complete
    may be older than the owner's decision on its durable row (a crash after a release, before the snapshot carrying
    it): it takes that decision. Other rows stay. Never raises."""
    from ouroboros.task_results import load_task_result

    for task in rows or ():
        try:
            if not (isinstance(task, dict) and task.get("_presence_turn")):
                continue
            if _saved_record(task):
                root = Path(task.get("budget_drive_root") or drive_root)
                stored = load_task_result(root, str(task.get("id") or "")) or {}
                if _identity(stored) == _identity(task):
                    lift_allowance(task, kept_allowance_lift(stored))
            elif moved_text_turn(drive_root, str(task.get("id") or ""), task.get("metadata")):
                record = park_record(task, drive_root)
                for key in (PRESENCE_RUN_KEY, "_conversation_prepare", "_execution_owner", "execution_owner", "worker_id"):
                    task.pop(key, None)
                task.update(record)
        except Exception:
            log.warning("Presence turn %s: its saved record was not adopted at restore", task.get("id"), exc_info=True)


def park_record(task: Mapping[str, Any], drive_root: Any, carried: Any = None) -> dict[str, Any]:
    """The row a Presence turn's park or restart keeps in the queue: its saved record, else (no complete
    record) the row it had under a typed hold that never dispatches, so no seed runs wider than admitted."""
    record = saved_turn_record(task, drive_root, carried=carried)
    if record is not None:
        return record
    from supervisor.events_budget import hold_budget_row

    held = dict(task)
    hold_budget_row(held, reason="presence_record_unavailable", result_root=Path(drive_root),
                    detail="the turn's own saved record could not be read; it is held, never run from its seed")
    return held


def answer_parked_turn(drive_root: Any, task_id: str) -> None:
    """The queue now durably keeps this moved turn (its park's snapshot is on disk): the transport waiting on it
    hears so at once, while its carrier still holds the conversation for its own effects. Versions 1 and 2 take
    the write-once initial envelope, every other waiter the typed retained answer. Nothing settles, returns
    capacity or releases here. Never raises."""
    try:
        _registry, execution = _find_execution(str(task_id or ""))
        if execution is None:
            return
        from ouroboros.presence_recovery import CUSTODY_QUEUED, custody_refusal
        from ouroboros.task_results import load_task_result

        stored = load_task_result(Path(drive_root), str(task_id)) or {}
        record = stored.get("presence_continuation") if isinstance(stored.get("presence_continuation"), dict) else {}
        initial = record.get("initial") if isinstance(record.get("initial"), dict) else None
        if initial is not None:
            presence = (stored.get("metadata") or {}).get("presence") or {}
            execution.publish_initial(PresenceTurnResult(
                outcome=str(initial.get("outcome") or "deferred"), text=str(initial.get("text") or ""),
                task_id=str(task_id), work_ref=str(initial.get("work_ref") or ""),
                delivery_reporting_version=int(presence.get("delivery_reporting_version") == 1), status="continuing",
                continuation_ref=str(task_id), output_ref=str(initial.get("output_ref") or ""),
                continuation_version=int(record.get("continuation_version") or 1)), return_capacity=False)
        execution.publish_retained(custody_refusal(stored, str(task_id), CUSTODY_QUEUED))
    except Exception:
        log.warning("Parked Presence turn %s: its waiting transport was not answered early", task_id, exc_info=True)


def _record_yield(root: Path, task: Mapping[str, Any], ctx: Any, lease: Any, event: Any, identity: str) -> None:
    """The yield every continuing author records (``presence_continuation``): its open turn on the conversation
    pointer, its write-once initial envelope and the chat cursor its continuation reads facts from."""
    from ouroboros.presence_continuation import ReviewWaitBinding, _chat_cursor, _handoff_ref, _persist, _released
    from ouroboros.presence_runner import record_continuing_turn
    from ouroboros.utils import utc_now_iso

    task_id, lent_at = str(task["id"]), utc_now_iso()
    work_ref = _handoff_ref(ctx) if ctx is not None else ""
    spoken = next((row for row in reversed(_released(ctx)) if row.get("text")), {}) if ctx is not None else {}
    try:
        cursor = _chat_cursor(root)
        record_continuing_turn(root, str(event.conversation_key), task_id, outcome=str(spoken.get("outcome") or "deferred"),
                               message=str(spoken.get("text") or ""), work_ref=work_ref, lent_at=lent_at)
        _persist(ReviewWaitBinding(lease, root, task_id, identity, event), work_ref, "", lent_at, cursor=cursor)
    except Exception:
        log.warning("Moved Presence turn %s could not record its yield; its waiters hear the retained answer",
                    task_id, exc_info=True)


def _task_runs_open(root: Path, task_id: str) -> bool:
    from ouroboros.external_runs import STOP_POLICY_OBSERVE, observe_task_runs, task_owned_runs_open

    try:
        return task_owned_runs_open(observe_task_runs(root, task_id, reason="presence_park_hold",
                                                      stop_policy=STOP_POLICY_OBSERVE))
    except Exception:
        return True  # unreadable custody is open, never none


def _hold_until_settled(root: Path, task_id: str, pause_id: str, reason: str) -> None:
    """Keep the conversation and this carrier's slot until the park's own effects settled: the same pause's
    settlement no longer reports its sent work running (an owner Pause: a Resume granted meanwhile is no
    settlement of it, a newer pause is not this one), a money park's own runs observed stopped, and none of the
    author's tools, processes, services or task-scope members running. A terminal (a Stop) ends the hold.
    Critics it started finish on their own; an unreadable read keeps the hold."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES, budget_pause_row
    from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING
    from ouroboros.presence_continuation import _custody_blockers
    from ouroboros.task_results import load_task_result

    found: dict = {}
    runs_open, looked = reason == "budget", 0.0
    while True:
        try:
            if str((load_task_result(root, task_id, strict=True) or {}).get("status") or "") in _TERMINAL:
                return
            row = budget_pause_row(root, task_id) if pause_id else {}
            sent = bool(row and str(row.get("pause_id") or "") == pause_id and row.get("state") in LIVE_PAUSE_STATES
                        and row.get("settlement") == SETTLEMENT_EXTERNAL_RUNNING)
            if runs_open and time.monotonic() - looked >= _RUNS_RECHECK_SEC:
                looked, runs_open = time.monotonic(), _task_runs_open(root, task_id)
            if not sent and not runs_open and not _custody_blockers(root, task_id, scope=True, found=found):
                return
        except Exception:
            log.debug("Moved Presence %s: its park custody is unreadable; it stays held", task_id, exc_info=True)
        time.sleep(_HOLD_POLL_SEC)


def park_moved_turn(drive_root: Any, task: Mapping[str, Any], events: list, *, agent: Any, lease: Any, event: Any,
                    identity: str, sink: Any) -> Exception | None:
    """After a moved author's ``handle_task``: None when it saved no pause; otherwise park it and return the typed
    retained answer its carrier settles with. The yield is recorded first (the initial envelope its transport takes),
    then the pause event goes to the queue's park owner WITHOUT a worker id (this carrier's slot stays busy, so
    no Resume dispatches beside it), then the author holds the conversation until its own effects settled."""
    pause = next((row for row in events or () if isinstance(row, dict) and row.get("type") == "budget_pause"), None)
    if pause is None or sink is None:
        return None
    root, task_id = Path(drive_root), str(task["id"])
    limit = pause.get("resource_limit") if isinstance(pause.get("resource_limit"), dict) else {}
    pause_id = str((limit.get("checkpoint") or {}).get("pause_id") or "")  # none before the body ran
    _record_yield(root, task, getattr(getattr(agent, "tools", None), "_ctx", None), lease, event, identity)
    events.remove(pause)
    sink([{key: value for key, value in pause.items() if key != "worker_id"}])
    _hold_until_settled(root, task_id, pause_id, str(limit.get("reason") or ""))
    from ouroboros.presence_recovery import CUSTODY_QUEUED, CUSTODY_TERMINAL, custody_refusal
    from ouroboros.task_results import load_task_result

    try:
        stored = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        stored = {}
    refusal = custody_refusal(stored, task_id, CUSTODY_TERMINAL if str(stored.get("status") or "") in _TERMINAL
                              else CUSTODY_QUEUED)
    refusal.parked = pause_id or "before_task_start"
    return refusal


def _authorized_continuation(task: Mapping[str, Any]) -> bool:
    """A continuation an owner Resume or an acknowledged restart authorized: the exact grant, a planned restart's
    owner-wait handoff, saved work a stop door returned or the owner released, or the owner's Resume of a start
    that never ran its body. A crash retry (``worker_crash``) is none: the shared crash policy keeps the retained
    unknown outcome and adds no automatic continuation."""
    if isinstance(task.get("_budget_pause_resume"), Mapping):
        return True
    handoff = task.get("_owner_wait_resume")
    if isinstance(handoff, Mapping):
        return bool(handoff.get("restart_transaction_id"))
    recovery = task.get("_working_recovery")
    if isinstance(recovery, Mapping):
        return str(recovery.get("cause") or "") != "worker_crash"
    return bool(task.get("budget_resumed_at")) and int(task.get("_attempt") or 1) == 1


def _end_refused_attempt(root: Path, turn_id: str) -> dict[str, Any]:
    """A continuation nothing authorized ends this attempt, and the row must say so before the attempt leaves RUNNING and
    its slot: the pooled owner its assignment stamped would otherwise read as the queue's custody, with no producer left
    (``presence_recovery.queue_kept``). The ended, unknown-outcome terminal the orphan reconciler projects, durable; a
    terminal already there stays. A failed write raises: nothing is released and the crash policy keeps the row."""
    from ouroboros.delegate_recovery import reconcile_unrecoverable_task
    from ouroboros.task_results import STATUS_FAILED, STATUS_INTERRUPTED, STATUS_RUNNING, write_task_result
    from ouroboros.task_status import PRESENCE_ATTEMPT_LOST, PRESENCE_ATTEMPT_LOST_NOTICE, _project_lost_attempt

    def ended(current: dict, _incoming: dict) -> dict | None:
        if str(current.get("status") or "") in _TERMINAL:
            return None
        lost = {**current, "result": ""}  # the row's "Task is running." is no record of this end
        prior = STATUS_INTERRUPTED if current.get("status") == STATUS_INTERRUPTED else STATUS_RUNNING
        _project_lost_attempt(lost, PRESENCE_ATTEMPT_LOST, prior, PRESENCE_ATTEMPT_LOST_NOTICE)
        return {key: value for key, value in lost.items() if value != current.get(key)}

    row = write_task_result(root, turn_id, STATUS_FAILED, _field_projector=ended, _terminal_observed=True)
    reconcile_unrecoverable_task(root, turn_id)  # as the crash terminal does: no successor takes the runs a handoff kept
    return row


def _saved_event(task: Mapping[str, Any], continuation_version: int) -> PresenceTurnEvent:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), Mapping) else {}
    source = presence.get("event") if isinstance(presence.get("event"), Mapping) else {}
    facts = source.get("facts")
    return PresenceTurnEvent(
        source_event_id=str(source.get("source_event_id") or ""), provider=str(source.get("provider") or ""),
        account_id=str(source.get("account_id") or ""), conversation_id=str(source.get("conversation_id") or ""),
        thread_id=str(source.get("thread_id") or ""), conversation_key=str(source.get("conversation_key") or ""),
        actor=dict(source.get("actor") or {}), conversation=dict(source.get("conversation") or {}),
        message=dict(source.get("message") or {}), text=str(presence.get("observed_text") or ""),
        delivery_reporting_version=int(presence.get("delivery_reporting_version") or 0),
        continuation_version=int(continuation_version or 0), facts=tuple(facts) if isinstance(facts, list) else None)


class _SavedCarrier:
    """One saved moved turn a pool worker continues: its gate lease, author binding and reentry facts."""

    def __init__(self, task: Mapping[str, Any], root: Path, lease: Any, binding: Any, agent: Any, event: Any) -> None:
        self.task_id, self.root, self.lease, self.binding, self.agent, self.event = (
            str(task["id"]), root, lease, binding, agent, event)
        # A restored question/review park reads its own facts at its reentry; an exact pause or saved work here.
        self.restored_wait = isinstance(task.get("_owner_wait_resume"), Mapping)
        self.parked_by = "pause" if (task.get("_budget_pause_resume") or task.get("budget_resumed_at")) else "restart"
        self.entered = self.intake_open = False
        self.live_key: tuple | None = None
        self.own_start = 0

    def bind(self) -> None:
        from ouroboros.conversation_lease import open_author_intake
        from ouroboros.presence_runner import _LIVE_LOCK, _LIVE_PRESENCE_TASKS, _live_task_rows

        self.live_key = (str(self.root.resolve()), self.task_id)
        with _LIVE_LOCK:
            _LIVE_PRESENCE_TASKS.add(self.live_key)
        self.lease.bind_author(self.task_id)
        self.lease.intake_agent = self.agent
        self.own_start = len(_live_task_rows(self.root, self.task_id, self.event.conversation_key))
        try:
            self.intake_open = open_author_intake(self.root, self.task_id, self.event.conversation_key)
        except Exception:
            log.debug("author intake fact not written for %s", self.task_id, exc_info=True)

    def reentered(self, ctx: Any, messages: list) -> None:
        """The conversation's facts since the park, appended once its saved state is restored."""
        from ouroboros.presence_continuation import conversation_lost, reentry_note

        if self.restored_wait or not self.lease.held() or conversation_lost(ctx):
            return
        note = reentry_note(self.binding, ctx, parked_by=self.parked_by)
        lift = (getattr(ctx, "task_metadata", None) or {}).get("producer_allowance_lifted")
        if isinstance(lift, Mapping):  # said, not implied (#2566)
            note += (f"\nThe owner's own Resume moved this turn off its ${float(lift.get('ceiling_usd') or 0):.2f} "
                     "producer allowance: it runs on the ordinary project budget now. Its tree cap, the shared Total "
                     "budget and its deadline still bind, and the spend so far still counts.")
        messages.append({"role": "user", "content": note})
        self.binding.restored = False

    def leave(self) -> None:
        from ouroboros.conversation_lease import close_author_intake
        from ouroboros.presence_runner import _LIVE_LOCK, _LIVE_PRESENCE_TASKS

        try:
            if self.intake_open:
                close_author_intake(self.root, self.task_id)
        except Exception:
            log.debug("author intake fact not closed for %s", self.task_id, exc_info=True)
        finally:
            if self.live_key is not None:
                with _LIVE_LOCK:
                    _LIVE_PRESENCE_TASKS.discard(self.live_key)


def reenter(ctx: Any) -> _SavedCarrier | None:
    """The loop's first step for a saved moved turn (``loop._resume_continuation``): before any restore, grant
    consumption or call, take the conversation and a Presence slot in admission order and wait out a process an
    earlier holder left; a Stop meanwhile holds nothing and ends on the zero-call rail. Then bind the author and
    open its intake. Any other task passes untouched."""
    state = _CARRIER.get()
    if state is None or state.entered or state.task_id != str(getattr(ctx, "task_id", "") or ""):
        return None
    state.entered = True
    from ouroboros.presence_continuation import _custody_blockers, _hard_stop_check

    stop = _hard_stop_check(ctx)
    if not state.lease.reacquire(stop):
        ctx._presence_conversation_lost = stop.reason or "cancelled"
        return state
    while _custody_blockers(state.root, state.task_id, scope=True) and not stop():
        time.sleep(_HOLD_POLL_SEC)
    if stop.reason:
        ctx._presence_conversation_lost = stop.reason
        return state
    state.bind()
    return state


def carry_saved_turn(task: dict[str, Any], out_q: Any, repo_dir: Any, drive_root: Any, *,
                     owner_wait_callback: Any = None, worker_id: Any = None) -> list:
    """A moved turn's saved work in the pool worker the queue dispatched it to (an owner's Resume, a restart's
    return): the same factory with no generic owner wait, the review/private-question park through this worker's
    pooled wait, the gate taken at ``reenter``, the turn's own end under its lock (``finish_turn``). Its events
    go back to ``run_worker_task`` (``task_done`` releases the row); a new park answers with one parked
    settlement frame after its hold. A record no owner act or restart authorized (a crash retry, an unmoved turn)
    is refused retained, building no agent, as its lost attempt always was: its ended terminal is published first
    (``_end_refused_attempt``) and its ``task_done`` ends the work like any task's."""
    from ouroboros.presence_continuation import bind_review_wait
    from ouroboros.presence_owner_contact import flush_owner_events
    from ouroboros.presence_runner import _configured_gate, _refused_turn, _stored_turn, finish_turn

    root, turn_id = Path(drive_root), str(task.get("id") or "")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    identity = _identity(task)

    def settle(error: Exception) -> list:
        out_q.put({"type": "presence_turn_settled", "turn_id": turn_id, "task_id": turn_id, "physical_id": turn_id,
                   "result": None, "error": _error_payload(error), **_frame_identity(task, worker_id, error)})
        return []

    # A row that cannot be read raises into ``run_worker_task``: its failure rail and the dead worker's recovery keep
    # this attempt's RUNNING row and slot, never a frame releasing them over a row no producer writes any more.
    stored = _stored_turn(root, turn_id, identity)
    built = task if _saved_record(task) else saved_turn_record(task, root, stored=stored)
    refused = not (identity and built and _authorized_continuation(task) and moved_text_turn(root, turn_id, metadata))
    if refused and not stored:  # no row to end: the frame alone releases this attempt
        return settle(_refused_turn(root, turn_id, stored, "presence_attempt_outcome_unknown"))
    if refused:
        ended = _end_refused_attempt(root, turn_id)
        settle(_refused_turn(root, turn_id, ended, "presence_attempt_outcome_unknown"))
        return [{"type": "task_done", "task_id": turn_id, "task_type": "presence", "chat_id": task.get("chat_id"),
                 "_is_direct_chat": True, "status": str(ended.get("status") or ""),
                 "reason_code": str(ended.get("reason_code") or ""), "outcome_axes": ended.get("outcome_axes") or {}}]
    if built is not task:  # a seed recovery could not adopt: the turn's own built fields from the row its start wrote
        task = {**task, **{key: built[key] for key in (*_STORED_FIELDS, "text", "context_requires_development")
                           if key in built}}
    task.pop(PRESENCE_RUN_KEY, None)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    record = stored.get("presence_continuation") if isinstance(stored.get("presence_continuation"), dict) else {}
    event = _saved_event(task, int(record.get("continuation_version") or metadata.get("presence_continuation_version") or 0))
    lease = PresenceTurnLease(event.conversation_key, None, _configured_gate(root))  # taken at ``reenter``
    lease.review_wait = owner_wait_callback

    def factory(**kwargs):
        from ouroboros.agent import make_agent

        return make_agent(**kwargs)  # no generic owner wait: sleep and detach stay closed

    agent = factory(repo_dir=str(repo_dir), drive_root=str(root), event_queue=out_q)
    binding = bind_review_wait(agent, lease=lease, event=event, task_id=turn_id, identity=identity, drive_root=root,
                               cursor=record.get("cursor") if isinstance(record.get("cursor"), dict) else None)
    state = _SavedCarrier(task, root, lease, binding, agent, event)
    token = _CARRIER.set(state)
    try:
        events = agent.handle_task(task)
    except BaseException:
        lease.release()
        raise
    finally:
        _CARRIER.reset(token)
        state.leave()
    try:
        flush_owner_events(root, events, sink=lambda rows: [out_q.put(dict(row)) for row in rows])
        parked = park_moved_turn(root, task, events, agent=agent, lease=lease, event=event, identity=identity,
                                 sink=lambda rows: [out_q.put(dict(row)) for row in rows])
        if parked is not None:
            lease.release()
            return settle(parked)
        if state.entered and lease.held():
            try:
                finish_turn(root, task, turn_id, identity, event, events, own_start=state.own_start)
            except PresenceTurnError:
                pass  # a refused or unknown terminal is its row's own answer; no request waits on saved work
    finally:
        lease.release()
    return [dict(row) for row in events if isinstance(row, dict) and row.get("type") != "presence_result"]


__all__ = ["PRESENCE_RUN_KEY", "FrameExecution", "adopt_saved_records", "answer_parked_turn", "carry_presence_turn",
           "carry_saved_turn", "dispatch_presence_turn", "handle_presence_turn_initial",
           "handle_presence_turn_settled", "kept_allowance_lift", "lift_allowance", "lift_live_allowance", "lifted_record",
           "moved_text_turn", "owner_allowance_lift", "owner_resume_decision", "owner_resume_lift", "park_moved_turn",
           "park_record", "presence_seed_task",
           "reenter", "saved_turn_record", "saves_exact_pause", "set_presence_runner", "settle_ended_turn"]
