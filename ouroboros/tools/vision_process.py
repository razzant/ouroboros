"""VLM child IPC: typed custody crosses the existing tracked process boundary.

The parent owns waiting and cancellation. The child owns one LLM call and its
ordinary physical-attempt ledger; no wait controller or credentials cross IPC.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any

from jsonschema import Draft202012Validator

from ouroboros import config
from ouroboros.observability import new_call_id
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture, PhysicalAttemptContext, adopt_physical_attempt_capture,
    capture_attempt_ids, current_usage_scope, last_physical_attempt_capture,
)
from ouroboros.utils import atomic_write_json

log = logging.getLogger(__name__)


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_STRING = {"type": "string"}
_DICT = {"type": "object"}
_NULL_INT = {"type": ["integer", "null"]}
_CONTEXT = _object({
    "profile": {"enum": ["owner_max", "owner_low", "owner_nano", "task_local_low"]},
    "rendered_mode": {"enum": ["max", "low", "nano"]},
    "measurement_basis": {"enum": ["fresh_route_usage", "fresh_model_usage", "cold_estimate"]},
    "route_fp": _STRING, "round_id": _STRING,
    "target_total_tokens": _NULL_INT, "capacity_total_tokens": _NULL_INT,
    "context_target_miss": {"type": "boolean"}, "automatic_pass_used": {"type": "boolean"},
    "memory_view": {"type": ["object", "null"], "additionalProperties": False,
        "required": ["status", "chars", "utf8_bytes", "estimated_tokens", "projection_target_miss", "basis"],
        "properties": {"status": {"enum": ["observed", "unobserved", "ambiguous"]},
            **{key: {"type": ["integer", "null"], "minimum": 0}
               for key in ("chars", "utf8_bytes", "estimated_tokens")},
            "projection_target_miss": {"type": ["boolean", "null"]},
            "basis": {"const": "host_physical_candidate"},
            "sections": {"type": "integer", "minimum": 1},
            "token_estimate_basis": {"const": "chars_div_4"}}},
})
_CAPTURE = _object({
    **{key: _STRING for key in ("attempt_id", "model", "provider", "provider_code", "provider_error_type", "provider_error")},
    "state": {"enum": ["reserved", "released", "dispatched", "settled", "unresolved"]},
    "candidate_measurement_kind": {"enum": ["canonical_json_v1", "opaque"]},
    "max_completion_tokens": {"type": "integer", "minimum": 0},
    **{key: {"type": ["string", "null"]} for key in ("candidate_raw_sha256", "candidate_context_sha256")},
    **{key: _NULL_INT for key in ("candidate_raw_size_bytes", "candidate_context_size_bytes", "provider_status_code")},
    "candidate_manifest_ref": {"type": ["object", "null"]},
    "physical_context": {"anyOf": [_CONTEXT, {"type": "null"}]},
    "route_is_loopback": {"type": "boolean"},
    "processing_preference": _STRING,
    "submitted_processing_mode": _STRING,
    "processing_basis": {"type": ["object", "null"]},
    "effort": {"type": ["object", "null"]},
})
_CUSTODY = _object({
    "operation_id": _STRING, "invocation_id": {"type": "string", "minLength": 1},
    "request_ref": _DICT, "request_manifest_ref": _DICT,
})
_RECEIPT = _object({
    "receipt_id": {"type": "string", "minLength": 1},
    "custody": {"anyOf": [_CUSTODY, {"type": "null"}]},
    "capture": {"anyOf": [_CAPTURE, {"type": "null"}]},
})
_RESULT = _object({
    **_RECEIPT["properties"],
    "kind": {"enum": ["success", "model", "not_dispatched", "interrupted", "error"]},
    "text": _STRING, "usage": _DICT, "ledger_attempt_ids": {"type": "array", "items": _STRING},
    "error": _STRING, "problem": _DICT, "operation_id": _STRING,
    "model_role": _STRING, "route": _DICT, "unknown": {"type": "boolean"},
    "control_reason": _STRING, "model_result": {"type": ["object", "null"]},
})


def _read_receipt(value: Any, receipt_id: str, *, terminal: bool = False) -> dict:
    Draft202012Validator(_RESULT if terminal else _RECEIPT).validate(value)
    if value["receipt_id"] != receipt_id:
        raise ValueError("VLM child receipt belongs to another invocation")
    capture, custody = value["capture"], value["custody"]
    if custody and capture and custody["invocation_id"] != capture["attempt_id"]:
        raise ValueError("VLM child operation and physical attempt disagree")
    if terminal and value["kind"] in {"model", "not_dispatched"}:
        problem = value["problem"]
        if not isinstance(problem.get("code"), str) or not problem["code"]:
            raise ValueError("VLM typed model error has no problem code")
        if not isinstance(problem.get("context", {}), dict):
            raise ValueError("VLM typed problem context must be an object")
        status = problem.get("context", {}).get("httpStatus", 0)
        if type(status) is not int:
            raise ValueError("VLM typed HTTP status must be an integer")
        if value["kind"] == "not_dispatched" and (value["unknown"] or capture is None or capture["state"] != "released"):
            raise ValueError("VLM non-dispatch requires its released physical attempt receipt")
        if custody and custody["operation_id"] and custody["operation_id"] != value["operation_id"]:
            raise ValueError("VLM terminal error belongs to another model operation")
    return value


def _restore_capture(raw: dict | None) -> PhysicalAttemptCapture | None:
    if raw is None:
        return None
    values = dict(raw)
    if values["physical_context"] is not None:
        values["physical_context"] = PhysicalAttemptContext(**values["physical_context"])
    return PhysicalAttemptCapture(**values)


def _private_json(path: Path, value: dict) -> None:
    # The enclosing 0700 directory protects the first atomic publication too.
    # Never publish an empty placeholder: the concurrent child would interpret
    # it as unreadable control before the shared writer replaces it with JSON.
    atomic_write_json(path, value)
    if os.name != "nt":
        path.chmod(0o600)


def child_main(payload_path: str) -> None:
    """Entrypoint imported by the tracked child, never a second wait owner."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.usage_accounting import UsageScope, usage_scope

    payload_file = Path(payload_path)
    kwargs = json.loads(payload_file.read_text(encoding="utf-8"))
    receipt_id = kwargs.pop("_receipt_id")
    raw_scope = kwargs.pop("_usage_scope", None)
    subscription = kwargs.pop("_subscription")
    # Existing timeout regression fixture; never a provider/auth bypass.
    sleep_for = float(kwargs.pop("_test_sleep_sec", 0) or 0)
    if sleep_for > 0:
        import time
        time.sleep(sleep_for)
    custody = None

    def observe(value):
        nonlocal custody
        Draft202012Validator(_CUSTODY).validate(value)
        custody = value
        capture = last_physical_attempt_capture()
        _private_json(payload_file.with_name("receipt.json"), {
            "receipt_id": receipt_id, "custody": custody,
            "capture": asdict(capture) if capture else None,
        })

    def control():
        try:
            value = json.loads(payload_file.with_name("control.json").read_text(encoding="utf-8"))
            if set(value) != {"receipt_id", "reason"} or value["receipt_id"] != receipt_id or not isinstance(value["reason"], str):
                return "child_control_invalid"
            return value["reason"] or None
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            return "child_control_unreadable"

    if subscription:
        kwargs.update(model_poll_control=control, model_operation_observer=observe)
    envelope = {"receipt_id": receipt_id, "custody": None, "capture": None,
                "kind": "success", "text": "", "usage": {}, "ledger_attempt_ids": [],
                "error": "", "problem": {}, "operation_id": "", "model_role": "",
                "route": {}, "unknown": False, "control_reason": "", "model_result": None}
    restored = UsageScope(**raw_scope) if raw_scope is not None else None
    scope = usage_scope(restored) if restored is not None else contextlib.nullcontext()
    with scope, capture_attempt_ids() as attempts:
        try:
            envelope["text"], envelope["usage"] = LLMClient().vision_query(**kwargs)
        except BaseException as exc:
            envelope["kind"] = "error"
            envelope["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (ClaudexorModelError, ModelWaitInterrupted)):
                cause = getattr(exc, "previous_error", None)
                cause = cause if isinstance(cause, ClaudexorModelError) else exc
                envelope.update(kind="interrupted" if isinstance(exc, ModelWaitInterrupted) else
                                "not_dispatched" if isinstance(exc, ClaudexorModelNotDispatched) else "model",
                                problem=getattr(cause, "problem", {}), operation_id=getattr(cause, "operation_id", ""),
                                model_role=exc.model_role, route=getattr(cause, "route", {}),
                                unknown=getattr(cause, "code", "") == "model_outcome_unknown",
                                control_reason=getattr(exc, "control_reason", ""),
                                usage=getattr(exc, "usage", {}), model_result=getattr(exc, "model_result", None))
            capture = getattr(exc, "physical_attempt_capture", None)
            envelope["capture"] = asdict(capture) if isinstance(capture, PhysicalAttemptCapture) else None
            envelope["ledger_attempt_ids"] = list(getattr(exc, "ledger_attempt_ids", []))
        else:
            capture = last_physical_attempt_capture()
            envelope["capture"] = asdict(capture) if capture else None
        envelope["ledger_attempt_ids"] = list(dict.fromkeys([*attempts, *envelope["ledger_attempt_ids"]]))
    envelope["custody"] = custody
    # Stdout is the terminal channel of our own tracked child. A failed sidecar
    # write must not replace or discard an already paid model result.
    print(json.dumps(envelope), flush=True)


def _unknown(receipt_id: str, receipt_path: Path, role: str, reason: str):
    from ouroboros.llm_claudexor import ClaudexorModelError

    receipt = {}
    try:
        receipt = _read_receipt(json.loads(receipt_path.read_text(encoding="utf-8")), receipt_id)
    except Exception as exc:
        # No checkpoint is no evidence of non-dispatch. Never invent an account,
        # provider receipt, operation id, or zero-cost physical capture.
        log.debug("VLM child checkpoint unavailable: %s", type(exc).__name__)
    custody = receipt.get("custody") or {}
    error = ClaudexorModelError({"code": "vlm_child_unsettled", "message": reason},
                               model_role=role, operation_id=custody.get("operation_id", ""), unknown=True)
    error.model_operation_custody = custody
    capture = _restore_capture(receipt.get("capture"))
    if capture is not None:
        error.physical_attempt_capture = capture
        error.ledger_attempt_ids = [capture.attempt_id]
        adopt_physical_attempt_capture(capture)
    return error


def _decode_terminal(value: dict, receipt_id: str) -> tuple[str, dict]:
    from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorModelNotDispatched
    from ouroboros.model_wait import ModelWaitInterrupted

    data = _read_receipt(value, receipt_id, terminal=True)
    capture = _restore_capture(data["capture"])
    adopt_physical_attempt_capture(capture)
    if data["kind"] == "success":
        attempts = list(dict.fromkeys([
            *data["ledger_attempt_ids"], *data["usage"].get("ledger_attempt_ids", []),
        ]))
        if attempts or "ledger_attempt_ids" in data["usage"]:
            data["usage"]["ledger_attempt_ids"] = attempts
        return data["text"], data["usage"]
    if data["kind"] == "error":
        error = RuntimeError(data["error"])
    else:
        problem = data["problem"]
        cls = ClaudexorModelNotDispatched if data["kind"] == "not_dispatched" else ClaudexorModelError
        error = cls(problem, model_role=data["model_role"], operation_id=data["operation_id"],
                    route=data["route"], unknown=data["unknown"])
        if data["kind"] == "interrupted":
            error = ModelWaitInterrupted(data["control_reason"], role=data["model_role"], cause=error)
            error.problem, error.operation_id, error.route = problem, data["operation_id"], data["route"]
        error.model_operation_custody = data["custody"]
        error.control_reason = data["control_reason"]
    if capture is not None:
        error.physical_attempt_capture = capture
    error.ledger_attempt_ids = data["ledger_attempt_ids"]
    error.usage = data["usage"]
    if data["model_result"] is not None:
        error.model_result = data["model_result"]
    raise error


def run_vision_child(*, child_timeout: float, subscription: bool, **kwargs: Any) -> tuple[str, dict]:
    """Run exactly one killable child; every retry remains the parent's decision."""
    from ouroboros.tools.shell import _tracked_subprocess_run

    poll = kwargs.pop("model_poll_control", None)
    # A callback is process-local; the child's own observer writes typed IPC.
    kwargs.pop("model_operation_observer", None)
    role = str(kwargs.get("model_role") or "vision")
    receipt_id = new_call_id("vision_child")
    payload = {**kwargs, "_receipt_id": receipt_id, "_subscription": subscription}
    scope = current_usage_scope()
    if scope is not None:
        payload["_usage_scope"] = {**vars(scope), "drive_root": str(scope.drive_root) if scope.drive_root else None}
    stopped = threading.Event()
    with tempfile.TemporaryDirectory(prefix="ouro-vlm-") as directory:
        payload_path = Path(directory) / "payload.json"
        receipt_path = Path(directory) / "receipt.json"
        _private_json(payload_path, payload)

        def transport_control():
            while not stopped.is_set():
                try:
                    reason = poll()
                    if reason:
                        _private_json(Path(directory) / "control.json", {"receipt_id": receipt_id, "reason": str(reason)})
                        return
                except Exception as exc:
                    log.warning("VLM parent control unavailable: %s", type(exc).__name__)
                stopped.wait(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC)

        watcher = None
        if subscription and poll is not None:
            context = contextvars.copy_context()
            watcher = threading.Thread(target=context.run, args=(transport_control,), daemon=True)
            watcher.start()
        try:
            executable = sys.executable or os.environ.get("OUROBOROS_AGENT_PYTHON") or "python3"
            script = "from ouroboros.tools.vision_process import child_main; import sys; child_main(sys.argv[1])"
            result = _tracked_subprocess_run([executable, "-c", script, str(payload_path)],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, timeout=child_timeout, env=config.runtime_environ())
            try:
                lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
                value = json.loads(lines[-1]) if lines else None
                _read_receipt(value, receipt_id, terminal=True)
            except Exception as exc:
                if subscription:
                    raise _unknown(receipt_id, receipt_path, role, "VLM child returned no valid terminal receipt") from None
                raise RuntimeError("VLM child returned no valid terminal receipt") from exc
            if subscription and value["kind"] == "error":
                error = _unknown(receipt_id, receipt_path, role, value["error"] or "VLM child failed")
                capture = _restore_capture(value["capture"])
                if capture is not None:
                    error.physical_attempt_capture = capture
                    adopt_physical_attempt_capture(capture)
                error.ledger_attempt_ids = value["ledger_attempt_ids"]
                error.usage = value["usage"]
                raise error
            return _decode_terminal(value, receipt_id)
        except subprocess.TimeoutExpired as exc:
            if subscription:
                raise _unknown(receipt_id, receipt_path, role, "VLM child exceeded its task execution window") from None
            raise TimeoutError(f"VLM query child did not settle within {child_timeout:g}s") from exc
        finally:
            stopped.set()
            if watcher is not None:
                watcher.join()
