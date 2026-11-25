"""Host facts bound to one canonical tool-result occurrence, never wire metadata.

Provider call IDs may repeat. Keep the existing invocation identity and the
producer's typed facts on the actual result row; checkpoint serialization then
retains them without a second ledger or a history lookup. Absence is unknown.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

TOOL_RESULT_RECORD_KEY = "_tool_result_record"


def logical_tool_result_text(content: Any) -> str | None:
    """Text-only cache wrappers preserve bytes; opaque content is not a text view."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
        isinstance(block, dict) and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and not (block.keys() - {"type", "text", "cache_control"})
        for block in content
    ):
        return "".join(block["text"] for block in content)
    return None


def _facts(value: Mapping[str, Any] | None) -> dict:
    facts = copy.deepcopy(dict(value or {}))
    facts["status"] = facts.get("status") or "unknown"
    facts["code"] = facts.get("code") or None
    facts["is_error"] = facts.get("is_error") if isinstance(facts.get("is_error"), bool) else None
    facts.setdefault("exit_code", None)
    return facts


def make_tool_result_record(
    exec_result: Mapping[str, Any], delivered_text: str, *,
    facts: Mapping[str, Any] | None = None, source_ref: Mapping[str, Any] | None = None,
) -> dict:
    """Stamp facts supplied by this producer, after the final visible text exists.

    The caller supplies typed facts, not parsed stdout. Invocation identity is
    copied from execution, never manufactured here. Missing identity is retained
    as a gap; the reader will not promote it to an occurrence-bound outcome.
    """
    from ouroboros.tool_call_log import invocation_fields

    return {
        "version": 1,
        "content_sha256": hashlib.sha256(delivered_text.encode("utf-8")).hexdigest(),
        "invocation": copy.deepcopy(invocation_fields(dict(exec_result))),
        "facts": _facts(facts),
        "trace_ref": copy.deepcopy(exec_result.get("trace_ref") or {}),
        "source_ref": copy.deepcopy(dict(source_ref or {})),
    }


def make_not_started_tool_result_record(
    tool_call_id: str, delivered_text: str, *,
    facts: Mapping[str, Any] | None = None, source_ref: Mapping[str, Any] | None = None,
) -> dict:
    """Stamp host-known non-execution (e.g. a cut batch that started zero tools) on the result row.

    No invocation or trace is fabricated: this binds non-execution to the call id,
    the final visible text digest, and the raw reply reference when available.
    """
    return {
        "version": 1,
        "kind": "not_started",
        "tool_call_id": str(tool_call_id),
        "content_sha256": hashlib.sha256(delivered_text.encode("utf-8")).hexdigest(),
        "facts": _facts({**dict(facts or {}), "status": "not_started", "is_error": False}),
        "source_ref": copy.deepcopy(dict(source_ref or {})),
    }


def read_tool_result_record(message: Mapping[str, Any]) -> dict:
    """Read this row's facts, or disclose an unknown outcome without blocking work.

    ``state == "recorded"`` is strictly reserved for an occurrence-bound tool invocation;
    host-known non-execution returns ``state == "not_started"`` without fabricating an invocation.
    """
    record = message.get(TOOL_RESULT_RECORD_KEY)
    reason = "record_missing"
    if isinstance(record, Mapping):
        text = logical_tool_result_text(message.get("content"))
        kind = record.get("kind")
        state, call_id = None, None
        if record.get("version") != 1 or not isinstance(record.get("facts"), Mapping):
            reason = "record_unrecognized"
        elif kind == "not_started":
            call_id = record.get("tool_call_id")
            if (record["facts"].get("status") != "not_started"
                    or record.get("invocation") is not None or record.get("trace_ref") is not None):
                reason = "record_unrecognized"
            elif not isinstance(call_id, str) or not call_id.strip():
                reason = "tool_result_mismatch"
            else:
                state = "not_started"
        elif kind is not None:
            reason = "record_unrecognized"
        else:
            invocation = record.get("invocation")
            if not isinstance(invocation, Mapping) or not invocation.get("invocation_id"):
                reason = "invocation_unrecorded"
            else:
                state, call_id = "recorded", invocation.get("tool_call_id")
        if state is not None:
            if message.get("role") != "tool" or call_id != message.get("tool_call_id"):
                reason = "tool_result_mismatch"
            elif (not isinstance(text, str)
                  or record.get("content_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()):
                reason = "content_mismatch"
            else:
                return {**copy.deepcopy(dict(record)), "state": state, "facts": _facts(record["facts"])}
    return {"state": "unknown", "reason": reason, "facts": _facts(None)}
