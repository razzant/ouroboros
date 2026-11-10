"""Model-call evidence and event transport, independent of admission and retry decisions.

The Main call publishes lifecycle and optional context through this existing observation
owner; the provider loop chooses what happened, never how the public event is shaped.
"""

from __future__ import annotations

import pathlib
import logging
import queue
from typing import Any, Dict, Optional, Tuple

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
        emit_cognitive_operation_event(event_queue, task_id=task_id, operation_id=llm_call_id, phase=phase,
                                       kind="llm", task_attempt=task_attempt, execution_id=execution_id, round_id=round_id)
    fact = attempt_fact(capture, str(round_id or ""))  # ``dispatched``: the ledger's own dispatch, while it waits
    if event_queue is None or (phase == "observed" and not isinstance(activity, dict)) or (phase == "dispatched" and not fact):
        return
    try:
        event_queue.put({
            "type": "main_llm_call_state", "task_id": str(task_id or ""), "task_attempt": task_attempt,
            "llm_call_id": str(llm_call_id or ""), "execution_id": str(execution_id or ""),
            "round_id": str(round_id or ""), "call_attempt": int(call_attempt), "phase": phase,
            **({"model_activity": activity} if phase == "observed" else {}),
            **({"context_attempt": fact} if fact else {}),
        })
    except Exception:
        logging.getLogger(__name__).debug("Failed to emit main LLM call state event", exc_info=True)


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
