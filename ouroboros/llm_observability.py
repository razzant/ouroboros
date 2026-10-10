"""Observed LLM call helpers for non-loop decision surfaces."""

from __future__ import annotations

import pathlib
import logging
from typing import Any, Dict, Tuple

from ouroboros.observability import new_call_id, persist_call
from ouroboros.anthropic_native_custody import public_custody_projection
from ouroboros.utils import sanitize_tool_result_for_log


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
    traced_kwargs = {key: value for key, value in kwargs.items() if key != "model_context_observer"}
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
