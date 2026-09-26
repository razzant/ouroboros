"""Read-only receipt of the official Cowork evaluator output for one task dump.

Standard library only, so the same reader stays usable inside the benchmark's own
evaluator environment. It reads exactly ``eval_res.json`` and, for a ``pass: null``
decline, exactly the ``traj_log.json`` that the upstream evaluator gated on. It never
globs, never writes either file and never copies the evaluator payload: the receipt
keeps the path, exact byte count and SHA-256, the literal verdict and a bounded cause.

``official_eval_status`` vocabulary produced here:

* ``completed``  — ``pass`` is a literal JSON boolean; ``false`` stays a negative verdict.
* ``declined``   — ``pass: null`` proven to be the upstream status gate: the linked log
  is non-success and the file is exactly that gate's two-key payload.
* ``unknown``    — ``pass: null`` without that proof.
* ``invalid``    — a JSON object whose ``pass`` is missing or not boolean/null.
* ``unreadable`` — I/O, UTF-8, JSON or non-object failure.
* ``unreported`` — no file: the evaluator left no receipt, which is not proof it never ran.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

RECEIPT_SCHEMA = "ouroboros.cowork.official_receipt.v1"
RESULT_NAME = "eval_res.json"
# ``TaskConfig.log_file`` default under the task dump; the evaluator's status gate reads it.
GATE_LOG_NAME = "traj_log.json"
UPSTREAM_SUCCESS = "success"
_CAUSE_LIMIT = 200


def _gate_details(status: str) -> str:
    # utils/evaluation/evaluator.py::TaskEvaluator.evaluate_one at the pinned benchmark commit.
    return f"Task status: {status}, only SUCCESS counts as pass; pass is null"


def _cause(text: str) -> str:
    return text[:_CAUSE_LIMIT]


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"non_json_constant:{value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _read_object(path: pathlib.Path) -> tuple[dict[str, Any], Any]:
    """Return ``(facts, value)``; ``facts['state']`` is ``absent``, ``unreadable`` or ``parsed``."""
    facts: dict[str, Any] = {"path": str(path), "bytes": None, "sha256": None}
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {**facts, "state": "absent"}, None
    except OSError as exc:
        return {**facts, "state": "unreadable", "cause": _cause(f"os_error:{type(exc).__name__}")}, None
    facts.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {**facts, "state": "unreadable", "cause": _cause(f"utf8_error:byte_{exc.start}")}, None
    try:
        value = json.loads(text, parse_constant=_reject_non_json_constant,
                           object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError) as exc:
        where = f"line_{exc.lineno}_col_{exc.colno}" if isinstance(exc, json.JSONDecodeError) else type(exc).__name__
        return {**facts, "state": "unreadable", "cause": _cause(f"json_error:{where}")}, None
    if not isinstance(value, dict):
        return {**facts, "state": "unreadable", "cause": _cause(f"not_object:{type(value).__name__}")}, None
    return {**facts, "state": "parsed"}, value


def _status_gate(task_dump: pathlib.Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Evidence that ``pass: null`` came from the evaluator's non-success status gate, else None."""
    log, record = _read_object(task_dump / GATE_LOG_NAME)
    if log["state"] != "parsed":
        return None
    status = record.get("status")
    if not isinstance(status, str) or not status or status == UPSTREAM_SUCCESS:
        return None
    if set(payload) != {"pass", "details"} or payload["details"] != _gate_details(status):
        return None
    return {"log_path": log["path"], "log_bytes": log["bytes"], "log_sha256": log["sha256"],
            "log_status": status, "rule": "upstream_status_gate"}


def read_official_receipt(task_dump: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return ``(receipt, verdict_payload)``; the payload is returned only for a literal verdict.

    The caller decides what, if anything, of the payload to retain; the receipt itself is
    always safe to persist and publish."""
    facts, value = _read_object(task_dump / RESULT_NAME)
    receipt = {"schema": RECEIPT_SCHEMA, "path": facts["path"], "bytes": facts["bytes"],
               "sha256": facts["sha256"], "pass": None, "cause": facts.get("cause", "")}
    if facts["state"] == "absent":
        return {**receipt, "official_eval_status": "unreported", "cause": "file_absent"}, None
    if facts["state"] == "unreadable":
        return {**receipt, "official_eval_status": "unreadable"}, None
    if "pass" not in value:
        return {**receipt, "official_eval_status": "invalid", "cause": "pass_missing"}, None
    verdict = value["pass"]
    if verdict is True or verdict is False:
        return {**receipt, "official_eval_status": "completed", "pass": verdict}, value
    if verdict is not None:
        return {**receipt, "official_eval_status": "invalid",
                "cause": _cause(f"pass_not_boolean:{type(verdict).__name__}")}, None
    gate = _status_gate(task_dump, value)
    if gate is None:
        return {**receipt, "official_eval_status": "unknown", "cause": "pass_null_unlinked"}, None
    return {**receipt, "official_eval_status": "declined", "cause": "status_gate", "gate": gate}, None


def read_linked_runtime_result(task_dump: pathlib.Path, summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the summary-named exported task, never select an arbitrary JSON file."""
    task_id = summary.get("ouroboros_task_id")
    if (not isinstance(task_id, str) or not task_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in task_id)):
        return {}, {"state": "unavailable", "cause": "task_id_unavailable"}
    facts, value = _read_object(task_dump / "ouroboros" / f"{task_id}.json")
    if facts["state"] != "parsed":
        return {}, facts
    if value.get("task_id", value.get("id")) != task_id:
        return {}, {**facts, "state": "unavailable", "cause": "task_id_mismatch"}
    return value, {**facts, "state": "linked", "task_id": task_id}
