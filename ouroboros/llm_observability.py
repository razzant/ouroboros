"""Model-call evidence and event transport, independent of admission and retry decisions.

The Main call publishes lifecycle and optional context through this existing observation
owner; the provider loop chooses what happened, never how the public event is shaped. A
round's ``llm_round`` also carries what the host did to prepare it (``round_window``).
"""

from __future__ import annotations

import contextlib
import pathlib
import logging
import queue
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

from ouroboros.context_timing import STEPS as _STEPS, carry_spans, host_step, measured_sections, section_spans, timed
from ouroboros.observability import new_call_id, persist_call
from ouroboros.anthropic_native_custody import public_custody_projection
from ouroboros.utils import emit_cognitive_operation_event, sanitize_tool_result_for_log
from ouroboros.physical_attempt_context import attempt_fact


def emit_main_call_state(event_queue: Optional[queue.Queue], identity: Tuple[Any, ...],
                        phase: str, activity: Any = None, capture: Any = None) -> None:
    """Exact-call lifecycle and display observations; only terminal facts close the idle lease. Once this round's
    attempt is dispatched, its frames carry its context (``physical_attempt_context.attempt_fact`` of ``capture``).
    Event delivery failure cannot change the provider loop's send or retry decision."""
    task_id, task_attempt, llm_call_id, execution_id, round_id, call_attempt = identity
    if phase in {"finished", "failed"}:
        timed("event_put", emit_cognitive_operation_event, event_queue, task_id=task_id, operation_id=llm_call_id,
              phase=phase, kind="llm", task_attempt=task_attempt, execution_id=execution_id, round_id=round_id)
    fact = attempt_fact(capture, str(round_id or ""))  # ``dispatched``: the ledger's own dispatch, while it waits
    if event_queue is None or (phase == "observed" and not isinstance(activity, dict)) or (phase == "dispatched" and not fact):
        return
    try:
        timed("event_put", event_queue.put, {
            "type": "main_llm_call_state", "task_id": str(task_id or ""), "task_attempt": task_attempt,
            "llm_call_id": str(llm_call_id or ""), "execution_id": str(execution_id or ""),
            "round_id": str(round_id or ""), "call_attempt": int(call_attempt), "phase": phase,
            **({"model_activity": activity} if phase == "observed" else {}),
            **({"context_attempt": fact} if fact else {}),
        })
    except Exception:
        logging.getLogger(__name__).debug("Failed to emit main LLM call state event", exc_info=True)


_RUSAGE = ("ru_nvcsw", "ru_nivcsw", "ru_majflt", "ru_oublock")


def _rusage() -> Dict[str, int]:
    """The process's voluntary and involuntary context switches, major page faults and block writes so far: waiting,
    being preempted, paging, writing. Nothing where the platform keeps no such counters (Windows has no ``resource``)."""
    try:
        import resource
    except ImportError:
        return {}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {name[3:]: int(getattr(usage, name)) for name in _RUSAGE}


def round_starts(slot: Any) -> None:
    """A round's preparation starts now, at a batch's end: the next ``round_window`` on ``slot`` times its first step,
    ``round_top`` (the loop's own work up to the model call), and its ``rusage`` from here."""
    slot._round_prep_since = (time.perf_counter(), time.thread_time(), threading.get_ident(), _rusage())


@contextlib.contextmanager
def round_window(slot: Any = None) -> Iterator[None]:
    """One round's model call as a ``context_timing`` window, receipted on its ``llm_round`` (``round_receipt``): the
    counted work, each marked section's wall time, the steps between ``host_step`` marks (``round_top`` from
    ``round_starts`` on ``slot``, used once) and the process's ``_rusage`` since the round's start. When that call
    fails the loop opens another, without ``slot``, around the recovery (a configured fallback, an owner-wait resend),
    whose answer carries the recovery's own facts. Facts for the log only: nothing reads them to decide."""
    since, prologue = getattr(slot, "_round_prep_since", None), getattr(slot, "_round_prologue", None)
    if since is not None:
        slot._round_prep_since = None
    if prologue is not None:  # a turn's restore, route rebind and tools, measured before its first round (``loop``)
        slot._round_prologue = None
    now = (time.perf_counter(), time.thread_time(), threading.get_ident())
    with measured_sections():
        carry_spans("prologue", prologue)
        token = _STEPS.set({"steps": {}, "mark": since[:3] if since else now, "rusage": since[3] if since else _rusage()})
        try:
            if since is not None:
                host_step("round_top")
            yield
        finally:
            _STEPS.reset(token)


def round_receipt() -> Dict[str, Any]:
    """``work`` and ``host_steps`` of the open round window for its ``llm_round``, or nothing outside one:
    ``steps`` as ``[wall ms, thread CPU ms or None]``, the sections' wall ms (``total``: the window so far) and ``rusage``.
    The line lands in an archive kept for good, so it holds about a kilobyte, one entry per name."""
    spans, state = section_spans(), _STEPS.get()
    if not spans or state is None:
        return {}
    rusage = _rusage()
    return {"work": spans["work"], "host_steps": {
        "steps": {name: [round(span[0] * 1000), None if span[1] is None else round(span[1] * 1000)]
                  for name, span in state["steps"].items()},
        "sections_ms": spans["sections_ms"],
        **({"rusage": {name: count - state["rusage"].get(name, count) for name, count in rusage.items()}}
           if rusage else {})}}


def retain_cancelled_response(request, response, capture, *, control_reason: str = "caller_cancelled") -> dict:
    """Keep a received physical response before cancellation unwinds its caller.

    Reuse private CAS and the ordinary call reader. This records the provider
    object (model_dump for SDK responses), not a completed task/review verdict.
    Native transports continue to own any original wire-byte receipt separately.
    """
    from dataclasses import asdict

    body = response.model_dump() if hasattr(response, "model_dump") else response
    return persist_call(
        request.drive_root, task_id=request.task_id or "llm",
        call_id=f"physical_{capture.attempt_id}_response", call_type="physical_response",
        payload={"response": body, "physical_attempt_capture": asdict(capture)}, keep_raw=True,
        manifest={"attempt_id": capture.attempt_id, "model": capture.model,
                  "provider": capture.provider, "state": capture.state,
                  "status": "received", "control_reason": control_reason},
    )


def persist_observed_call(root: Any, *, payload: Any, writer: Any = None, **identity: Any) -> dict:
    """Best-effort public trace: opaque continuation never enters this projection.

    Failure to write observability cannot discard an already useful response or
    authorize another provider call. Private exact-byte transport custody uses
    persist_call directly and retains its own stronger acknowledgement contract.
    """
    try:
        return (writer or persist_call)(root, payload=public_custody_projection(payload), **identity)
    except Exception:
        logging.getLogger(__name__).warning("Failed to persist LLM observability payload", exc_info=True)
        return {}


def _root(drive_root: Any) -> pathlib.Path:
    try:
        return pathlib.Path(drive_root)
    except TypeError:
        return pathlib.Path("../data")


def _base_manifest(call_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "call_type": call_type,
        "model": kwargs.get("model"),
        "reasoning_effort": kwargs.get("reasoning_effort"),
        "max_tokens": kwargs.get("max_tokens"),
        "use_local": kwargs.get("use_local"),
    }


def chat_observed(
    llm: Any,
    *,
    drive_root: Any,
    task_id: str = "",
    call_type: str = "llm_call",
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run ``llm.chat`` while preserving request/response/error payloads."""

    root = _root(drive_root)
    call_id = new_call_id(call_type)
    traced_kwargs = dict(kwargs)
    persist_observed_call(
        root,
        task_id=task_id or call_type,
        call_id=f"{call_id}_request",
        call_type=f"{call_type}_request",
        payload={"kwargs": traced_kwargs},
        manifest=_base_manifest(call_type, kwargs),
    )
    try:
        msg, usage = llm.chat(**kwargs)
    except Exception as exc:
        safe = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")
        persist_observed_call(
            root,
            task_id=task_id or call_type,
            call_id=f"{call_id}_error",
            call_type=f"{call_type}_error",
            payload={"error": f"{type(exc).__name__}: {exc}", "kwargs": traced_kwargs},
            manifest={**_base_manifest(call_type, kwargs), "status": "error", "error": safe},
        )
        raise
    try:
        from ouroboros.openai_chat_dispatch import CUSTOM_RECEIPTS_USAGE_KEY

        public_usage = dict(usage)
        public_usage.pop(CUSTOM_RECEIPTS_USAGE_KEY, None)
        persist_call(
            root,
            task_id=task_id or call_type,
            call_id=f"{call_id}_response",
            call_type=f"{call_type}_response",
            payload={
                "message": public_custody_projection(msg),
                "usage": public_usage,
            },
            manifest={**_base_manifest(call_type, kwargs), "status": "ok"},
        )
    except Exception:
        pass
    return msg, usage
