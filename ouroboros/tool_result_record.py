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


def read_tool_result_record(message: Mapping[str, Any]) -> dict:
    """Read this row's facts, or disclose an unknown outcome without blocking work."""
    record = message.get(TOOL_RESULT_RECORD_KEY)
    reason = "record_missing"
    if isinstance(record, Mapping):
        invocation = record.get("invocation")
        text = logical_tool_result_text(message.get("content"))
        if record.get("version") != 1 or not isinstance(record.get("facts"), Mapping):
            reason = "record_unrecognized"
        elif not isinstance(invocation, Mapping) or not invocation.get("invocation_id"):
            reason = "invocation_unrecorded"
        elif message.get("role") != "tool" or invocation.get("tool_call_id") != message.get("tool_call_id"):
            reason = "tool_result_mismatch"
        elif not isinstance(text, str) or record.get("content_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            reason = "content_mismatch"
        else:
            return {**copy.deepcopy(dict(record)), "state": "recorded", "facts": _facts(record["facts"])}
    return {"state": "unknown", "reason": reason, "facts": _facts(None)}
