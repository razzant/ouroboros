"""Native inspection and maintenance tools over the same service as Accounts.

No installer, account action, arbitrary control path or private job ledger lives
here. A returned operation id is custody to inspect, never a completed update.
"""
from __future__ import annotations

import json

from ouroboros import harness_maintenance as service
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result


def _result(ctx, call):
    try:
        body = call()
        status, code = "ok", "OK"
    except (ClaudexorUnavailable, ValueError) as exc:
        http_status, body = service.maintenance_problem(exc)
        status, code = (("blocked", "ACCESS_BLOCKED") if http_status == 403 else
                        ("error", "TOOL_ARG_ERROR") if http_status == 400 else
                        ("unavailable", "CAPABILITY_UNAVAILABLE"))
    return _publish_tool_result(ctx, ToolResult(status=status, code=code,
                                text=json.dumps(body, ensure_ascii=False, allow_nan=False)))


def _inspect_harness(ctx: ToolContext, harnesses: list | None = None, operation_id: str = "",
                     fresh: bool = False, check_latest: bool = False) -> str:
    def inspect():
        if operation_id:
            if harnesses or fresh or check_latest:
                raise ValueError("An operation_id read cannot also inspect harness installations")
            return service.inspect_operation(operation_id, ctx=ctx)
        return service.inspect_harnesses(harnesses or [], fresh=fresh, check_latest=check_latest, ctx=ctx)
    return _result(ctx, inspect)


def _maintain_harness(ctx: ToolContext, action: str, harness: str = "", target: str = "latest",
                      version: str = "", request_id: str = "", operation_id: str = "") -> str:
    def maintain():
        if action == "cancel":
            return service.cancel_maintenance(operation_id, ctx=ctx)
        if action != "update" or not harness or operation_id:
            raise ValueError("Use update with a harness, or cancel with operation_id")
        if target not in {"latest", "version", "previous", "baseline"}:
            raise ValueError("Unknown maintenance target")
        if (target == "version") != bool(version):
            raise ValueError("Supply version only for target='version'")
        requested = {"kind": target, **({"version": version} if version else {})}
        return service.start_maintenance({"harness": harness, "target": requested}, request_id, ctx=ctx)
    return _result(ctx, maintain)


def maintenance_tool_entries() -> list[ToolEntry]:
    return [
        ToolEntry("inspect_harness", {
            "name": "inspect_harness",
            "description": (
                "Inspect the selected vendor CLI, update capabilities or one retained maintenance operation. "
                "Read this when a harness may need an update or a previous update has an unconfirmed result. "
                "It never starts an engine, installs a CLI, logs in or generates model output. "
                "check_latest explicitly asks the engine for a vendor version check; missing evidence stays unknown. "
                "Use operation_id alone to recover an existing operation, including a failed or cancelled update."
            ),
            "parameters": {"type": "object", "properties": {
                "harnesses": {"type": "array", "items": {"type": "string"}, "description": "Exact harness ids from engine discovery; omit for all."},
                "operation_id": {"type": "string", "description": "Read this retained operation; omit other selectors."},
                "fresh": {"type": "boolean", "default": False},
                "check_latest": {"type": "boolean", "default": False},
            }, "additionalProperties": False},
        }, _inspect_harness),
        ToolEntry("maintain_harness", {
            "name": "maintain_harness",
            "description": (
                "Request the engine's declared CLI update/return capability, or cancel an operation. "
                "Inspect first: supported targets and ownership come from the engine, never the harness name. "
                "This changes installed program files in place; new native starts may fail, and returning a version "
                "may require the registry. It never changes accounts or starts a login. "
                "Update requires a stable request_id: retain it and retry the SAME body and id after a lost reply, "
                "never generate a new id to recover it. Acceptance is not success; inspect_harness(operation_id=...) "
                "reads progress and terminal evidence. Cancel may return while termination remains unconfirmed. "
                "The task's existing authority and Pause apply; no extra per-update owner confirmation is required."
            ),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["update", "cancel"]},
                "harness": {"type": "string", "description": "Required for update: exact engine harness id."},
                "target": {"type": "string", "enum": ["latest", "version", "previous", "baseline"], "default": "latest"},
                "version": {"type": "string", "description": "Required for target=version: exact version to install. Omit for other targets."},
                "request_id": {"type": "string", "description": "Required for update. Stable id, at most 256 printable ASCII characters; reuse with the same body after lost contact."},
                "operation_id": {"type": "string", "description": "Required for cancel; inspect reads status separately."},
            }, "required": ["action"], "additionalProperties": False},
        }, _maintain_harness),
    ]
