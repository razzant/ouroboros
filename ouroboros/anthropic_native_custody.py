"""Exact direct-Anthropic assistant-content custody.

The canonical transcript stays provider neutral for execution, but a direct
Anthropic tool turn also carries one private receipt containing the complete
native assistant ``content`` list.  Only the same provider/endpoint/API/model
route may replay it.  Public projections drop the receipt; checkpoints retain
it so a crash cannot destroy an unfinished assistant/tool-result unit.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import hashlib
import json
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ouroboros.request_wire_contract import canonical_sha256, normalize_endpoint

ANTHROPIC_NATIVE_RECEIPT_KEY = "_anthropic_native_content"
ANTHROPIC_CONSUMED_RECEIPTS_KEY = "_anthropic_consumed_receipts"
ANTHROPIC_OPAQUE_PROJECTION_KEY = "_anthropic_native_opaque"
ANTHROPIC_CUSTODY_PROJECTION_KEY = "_anthropic_native_custody"
_RECEIPT_VERSION = 1
_REPLAYED_RECEIPTS: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
    "ouroboros_anthropic_replayed_receipts", default=(),
)


def _content_json(content: Any) -> str:
    return json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _route_identity(target: Mapping[str, Any]) -> Dict[str, str]:
    endpoint = normalize_endpoint(target.get("base_url"))
    provider = str(target.get("provider") or "").strip().lower()
    model = str(target.get("resolved_model") or "").strip()
    if provider != "anthropic" or not endpoint or not model:
        raise ValueError("direct Anthropic custody requires an exact route")
    return {
        "provider": provider,
        "endpoint_sha256": canonical_sha256(endpoint),
        "api_surface": "messages",
        "model": model,
    }


def _native_tool_ids(content: Sequence[Any]) -> Tuple[str, ...]:
    ids: List[str] = []
    for block in content:
        if not isinstance(block, Mapping) or str(block.get("type") or "") != "tool_use":
            continue
        call_id = block.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id != call_id.strip():
            raise ValueError("native Anthropic tool_use requires an exact id")
        ids.append(call_id)
    if len(ids) != len(set(ids)):
        raise ValueError("native Anthropic content contains duplicate tool ids")
    return tuple(ids)


def _canonical_tools_match_native(
    message: Mapping[str, Any],
    content: Sequence[Mapping[str, Any]],
) -> bool:
    native = [block for block in content if block.get("type") == "tool_use"]
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != len(native):
        return False
    for call, block in zip(calls, native):
        function = call.get("function") if isinstance(call, Mapping) else None
        arguments = function.get("arguments") if isinstance(function, Mapping) else None
        if not isinstance(arguments, str):
            return False
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return False
        if not (
            call.get("type") == "function"
            and call.get("id") == block.get("id")
            and function.get("name") == block.get("name")
            and canonical_sha256(parsed) == canonical_sha256(block.get("input"))
        ):
            return False
    return True


def retain_native_assistant_content(
    message: Mapping[str, Any],
    native_content: Any,
    target: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach the complete native list when it contains a tool use."""
    out = copy.deepcopy(dict(message))
    if not isinstance(native_content, list):
        return out
    tool_ids = _native_tool_ids(native_content)
    if not tool_ids:
        return out
    encoded = _content_json(native_content)
    out[ANTHROPIC_NATIVE_RECEIPT_KEY] = {
        "version": _RECEIPT_VERSION,
        "route": _route_identity(target),
        "content_sha256": _sha256_bytes(encoded),
        "content_json": encoded,
        "tool_use_ids": list(tool_ids),
    }
    return out


def _validated_receipt(
    message: Mapping[str, Any],
    target: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], Tuple[str, ...]]]:
    raw = message.get(ANTHROPIC_NATIVE_RECEIPT_KEY)
    if not isinstance(raw, Mapping) or set(raw) != {
        "version", "route", "content_sha256", "content_json", "tool_use_ids",
    }:
        return None
    if type(raw.get("version")) is not int or raw.get("version") != _RECEIPT_VERSION:
        return None
    route = raw.get("route")
    if not isinstance(route, Mapping) or set(route) != {
        "provider", "endpoint_sha256", "api_surface", "model",
    }:
        return None
    if target is not None:
        try:
            if dict(route) != _route_identity(target):
                return None
        except ValueError:
            return None
    encoded = raw.get("content_json")
    digest = raw.get("content_sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        return None
    if _sha256_bytes(encoded) != digest:
        return None
    try:
        content = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    if not isinstance(content, list) or any(not isinstance(item, dict) for item in content):
        return None
    try:
        ids = _native_tool_ids(content)
    except ValueError:
        return None
    stored_ids = raw.get("tool_use_ids")
    if not isinstance(stored_ids, list) or tuple(stored_ids) != ids:
        return None
    if not _canonical_tools_match_native(message, content):
        return None
    return dict(raw), content, ids


def native_reasoning_blocks(message: Mapping[str, Any], *, retire: bool = False) -> int:
    """Count bound blocks; optionally rewrite their receipt, preserving all other custody.

    The new digest is unconsumed until the next answered request, so its tool
    unit becomes active again. Server blocks and their original order survive.
    """
    validated = _validated_receipt(message)
    if validated is None:
        return 0
    receipt, content, _ids = validated
    kept = [block for block in content if block.get("type") not in {"thinking", "redacted_thinking"}]
    count = len(content) - len(kept)
    if retire and count:
        encoded = _content_json(kept)
        receipt.update(content_json=encoded, content_sha256=_sha256_bytes(encoded))
        message[ANTHROPIC_NATIVE_RECEIPT_KEY] = receipt
    return count


@contextlib.contextmanager
def anthropic_replay_scope() -> Iterator[None]:
    token = _REPLAYED_RECEIPTS.set(())
    try:
        yield
    finally:
        _REPLAYED_RECEIPTS.reset(token)


def anthropic_replay_scoped(function: Any) -> Any:
    @functools.wraps(function)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with anthropic_replay_scope():
            return function(*args, **kwargs)
    return _wrapped


def native_content_for_replay(
    message: Mapping[str, Any],
    target: Mapping[str, Any],
    matching_tool_result_ids: Sequence[str],
) -> Optional[List[Dict[str, Any]]]:
    """Return an exact detached replay only for the matching same-route unit."""
    validated = _validated_receipt(message, target)
    if validated is None:
        return None
    receipt, content, tool_ids = validated
    if tuple(matching_tool_result_ids) != tool_ids:
        return None
    digest = str(receipt["content_sha256"])
    current = _REPLAYED_RECEIPTS.get()
    if digest not in current:
        _REPLAYED_RECEIPTS.set((*current, digest))
    return copy.deepcopy(content)


def is_replayed_native_content(content: Any) -> bool:
    """Whether this exact physical block list came from a replay receipt."""
    if not isinstance(content, list):
        return False
    try:
        digest = _sha256_bytes(_content_json(content))
    except (TypeError, ValueError):
        return False
    return digest in _REPLAYED_RECEIPTS.get()


def mark_replayed_receipts_consumed(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind successful continuation custody to its new assistant response."""
    out = copy.deepcopy(dict(message))
    consumed = tuple(dict.fromkeys(_REPLAYED_RECEIPTS.get()))
    if consumed:
        out[ANTHROPIC_CONSUMED_RECEIPTS_KEY] = list(consumed)
    return out


def scrub_native_custody(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Remove direct-route custody before another provider/API/model dispatch."""
    cleaned = copy.deepcopy(list(messages))
    for message in cleaned:
        if isinstance(message, dict):
            message.pop(ANTHROPIC_NATIVE_RECEIPT_KEY, None)
            message.pop(ANTHROPIC_CONSUMED_RECEIPTS_KEY, None)
    return cleaned


def custody_private_key(value: Any) -> bool:
    """Keys withheld from public/summarizer projections, not transport copies."""
    return str(value) in {
        ANTHROPIC_NATIVE_RECEIPT_KEY,
        ANTHROPIC_CONSUMED_RECEIPTS_KEY,
        "nativeContinuation",
    }


def public_custody_projection(value: Any) -> Any:
    """Drop opaque custody recursively from summarizer/observability payloads."""
    if isinstance(value, Mapping):
        return {
            str(key): public_custody_projection(item)
            for key, item in value.items()
            if not custody_private_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [public_custody_projection(item) for item in value]
    return value


def _receipt_projection(receipt: Any) -> Dict[str, Any]:
    encoded = receipt.get("content_json") if isinstance(receipt, Mapping) else None
    if not isinstance(encoded, str):
        return {"block_types": [], "size_bytes": 0, "sha256": ""}
    try:
        content = json.loads(encoded)
    except (TypeError, ValueError):
        content = []
    block_types = [
        str(block.get("type") or "opaque")
        for block in content
        if isinstance(block, Mapping)
    ] if isinstance(content, list) else []
    return {
        "block_types": block_types,
        "size_bytes": len(encoded.encode("utf-8")),
        "sha256": _sha256_bytes(encoded),
    }


def observability_custody_projection(value: Any) -> Any:
    """Replace private replay receipts with type/order/size/digest metadata."""
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if str(key) == ANTHROPIC_NATIVE_RECEIPT_KEY:
                projected[ANTHROPIC_CUSTODY_PROJECTION_KEY] = _receipt_projection(item)
            elif str(key) != ANTHROPIC_CONSUMED_RECEIPTS_KEY:
                projected[str(key)] = observability_custody_projection(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [observability_custody_projection(item) for item in value]
    return value


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return "unknown"


def _opaque_fields(block: Mapping[str, Any], safe: frozenset[str]) -> List[Dict[str, Any]]:
    projected = []
    for field, value in block.items():
        if str(field) in safe:
            continue
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False, default=str,
        )
        projected.append({
            "field": str(field),
            "value_type": _json_value_type(value),
            "size_bytes": len(encoded.encode("utf-8")),
            "sha256": _sha256_bytes(encoded),
        })
    return projected


def _physical_native_block_projection(block: Any) -> Any:
    if not isinstance(block, Mapping):
        encoded = json.dumps(block, ensure_ascii=False, allow_nan=False, default=str)
        return {
            "type": "opaque",
            ANTHROPIC_OPAQUE_PROJECTION_KEY: [{
                "field": "value",
                "value_type": _json_value_type(block),
                "size_bytes": len(encoded.encode("utf-8")),
                "sha256": _sha256_bytes(encoded),
            }],
        }
    block_type = str(block.get("type") or "").strip()
    safe = {
        "text": frozenset({"type", "text", "cache_control"}),
        "tool_use": frozenset({"type", "id", "name", "input"}),
    }.get(block_type, frozenset({"type"}))
    out = {
        str(field): public_custody_projection(value)
        for field, value in block.items()
        if str(field) in safe
    }
    opaque = _opaque_fields(block, safe)
    if opaque:
        out[ANTHROPIC_OPAQUE_PROJECTION_KEY] = opaque
    return out


def physical_custody_projection(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Project direct-Anthropic physical replay without exposing opaque values."""
    projected = public_custody_projection(value)
    messages = projected.get("messages") if isinstance(projected, dict) else None
    if not isinstance(messages, list):
        return projected
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, Mapping) and block.get("type") == "tool_use"
            for block in content
        ):
            message["content"] = [
                _physical_native_block_projection(block) for block in content
            ]
    return projected


def context_custody_proxy(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Represent opaque replay bytes by size, without exposing their values."""
    projected = public_custody_projection(value)
    receipt = value.get(ANTHROPIC_NATIVE_RECEIPT_KEY)
    encoded = receipt.get("content_json") if isinstance(receipt, Mapping) else None
    if isinstance(encoded, str):
        projected["_anthropic_native_opaque_bytes"] = "#" * len(encoded.encode("utf-8"))
    return projected


def anthropic_tool_unit_active(
    messages: Sequence[Mapping[str, Any]],
    assistant_index: int,
    result_end_index: int,
) -> bool:
    """True until a later successful assistant response consumes the replay."""
    if not (0 <= assistant_index < len(messages)):
        return False
    validated = _validated_receipt(messages[assistant_index])
    if validated is None:
        return False
    digest = str(validated[0]["content_sha256"])
    for later in messages[result_end_index + 1:]:
        consumed = later.get(ANTHROPIC_CONSUMED_RECEIPTS_KEY)
        if isinstance(consumed, list) and digest in consumed:
            return False
    return True
