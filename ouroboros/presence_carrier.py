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
``presence_turn_initial`` frame (``FrameExecution``). Retry identity and ``heard_replay``
dedup stay the gate's, inside the body.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_events import PresenceTurnError, PresenceTurnEvent
from ouroboros.presence_executions import PresenceTurnExecution, PresenceTurnNotStarted, _find_execution
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
                "retry_after": getattr(exc, "retry_after", None)}
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
        return error
    return RuntimeError(f"{payload.get('type') or 'Exception'}: {payload.get('message') or ''}")


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
    """
    from supervisor import queue, workers
    from supervisor.conversation_lane import admit_turn, replenish_after_claim
    from supervisor.inline_carrier import InlineExecutorUnavailable
    from supervisor.worker_chat_lane import _degrade_if_no_carrier_can_start

    task = presence_seed_task(admission=admission, event=event, skill_name=skill_name, event_payload=event_payload,
                              staged_files=staged_files, gate_wait=gate_wait)
    # The Host's data root is this turn's authority, whichever carrier runs it: a pooled worker carries the
    # server's own root (the same one), and the in-process degraded executor must re-admit and write against
    # the Host's root, never the pool's global DRIVE_ROOT (unset, or another root, in a standalone Host).
    task["drive_root"] = str(drive_root)
    with queue._queue_lock, workers._repo_writer_gate_lock:
        meta = admit_turn(task, chat_id=int(task["chat_id"]), client_message_id=event.source_event_id,
                          quiet=True, reply_expected=False)
        try:
            _degrade_if_no_carrier_can_start(meta)
        except InlineExecutorUnavailable as exc:
            # Nothing ran and no carrier will come to this process: the row leaves now and the Host answers
            # a typed retry; its reservation returns with the settlement, the gate was never taken.
            workers.RUNNING.pop(str(task["id"]), None)
            queue.persist_queue_snapshot(reason="presence_turn_not_started")
            raise PresenceTurnNotStarted("presence turn thread could not start") from exc
    replenish_after_claim()  # the spare the turn took is replaced now, or its own carrier started now (4A)


def carry_presence_turn(task: dict[str, Any], out_q: Any, repo_dir: Any, drive_root: Any, *,
                        owner_wait_callback: Any = None) -> None:
    """The carrier's body for a Host turn: re-admit from durable facts, take the gate, run, report.

    Every outcome leaves as ONE ``presence_turn_settled`` frame (the result, or the error by
    value); the Host resolves its waiter from it and releases the RUNNING row and the slot.
    """
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
                                    continuing=int(facts.get("continuation_version") or 0))
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
        else:
            lease = gate.try_acquire(event.conversation_key)
            if lease is None:
                error_busy = PresenceTurnError("presence_capacity_full", "conversation", disposition="retry")
                error_busy.retry_after = 1
                raise error_busy
        lease.execution = FrameExecution(turn_id, out_q)
        try:
            def factory(**kwargs):
                from ouroboros.agent import make_agent

                agent = make_agent(**kwargs)
                if owner_wait_callback is not None:
                    agent.owner_wait_callback = owner_wait_callback
                return agent

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
                       "error": _error_payload(error) if error is not None else None})
        except Exception:
            log.warning("presence turn %s could not report its settlement", turn_id, exc_info=True)


def handle_presence_turn_initial(evt: dict[str, Any], ctx: Any) -> None:
    """A carrier published a continuing author's initial envelope: the Host's waiter takes it."""
    _registry, execution = _find_execution(str(evt.get("turn_id") or ""))
    if execution is None or not isinstance(evt.get("result"), dict):
        return
    execution.publish_initial(PresenceTurnResult(**evt["result"]), return_capacity=bool(evt.get("return_capacity", True)))


def handle_presence_turn_settled(evt: dict[str, Any], ctx: Any) -> None:
    """A carrier's turn ended: resolve the Host's waiter, release the lane row and its slot."""
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
    with _queue_lock:
        meta = ctx.RUNNING.pop(turn_id, None)
        wid = meta.get("worker_id") if isinstance(meta, dict) else None
        slot = ctx.WORKERS.get(wid) if wid is not None else None
        if slot is not None and slot.busy_task_id == turn_id and not getattr(slot, "reaping", False):
            slot.busy_task_id = None
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


__all__ = ["PRESENCE_RUN_KEY", "FrameExecution", "carry_presence_turn", "dispatch_presence_turn", "set_presence_runner",
           "handle_presence_turn_initial", "handle_presence_turn_settled", "presence_seed_task"]
