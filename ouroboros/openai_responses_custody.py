"""Exact direct-OpenAI Responses output custody.

A Responses reply is an ordered list of output items: reasoning (opaque ``encrypted_content`` under
``store: false``), assistant messages with their ``phase``, and function calls with their ``call_id``.
The canonical transcript keeps the visible text and calls; one private receipt beside them keeps the
items exactly as returned. The next request on the SAME connection, endpoint, API and model replays
a completed turn whole, so reasoning continuity, phases and call identity survive tool rounds,
commentary rounds, owner follow-ups, checkpoints and restores.

A receipt binds that route and the canonical text and calls it was produced with. An edited, folded,
incomplete or route-changed turn projects canonically, without item ids, never as a partial native
replay, and no item reaches another provider, connection or model (``llm_messages`` scrubs every
receipt from other routes' send copies). Whether the server used replayed reasoning is its own report
(``input_transformations`` on the reply), not something this receipt asserts. Public projections drop
the receipt; checkpoints keep it. A replayed turn stays active for compaction until a later answered
request consumed it, as the native Anthropic unit does (``responses_tool_unit_active``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.request_wire_contract import canonical_sha256, normalize_endpoint

RESPONSES_RECEIPT_KEY = "_openai_responses_items"
RESPONSES_CONSUMED_KEY = "_openai_responses_consumed"
RESPONSES_CUSTODY_PROJECTION_KEY = "_openai_responses_custody"
RESPONSES_PRIVATE_KEYS = frozenset({RESPONSES_RECEIPT_KEY, RESPONSES_CONSUMED_KEY})
# Per-call target key: the receipt digests one request replayed whole, read back by the reply normalizer.
REPLAYED_TARGET_KEY = "_openai_responses_replayed"
_VERSION = 1
_RECEIPT_FIELDS = frozenset({"version", "route", "status", "items_sha256", "items_json", "call_ids"})


def _items_json(items: Sequence[Any]) -> str:
    return json.dumps(list(items), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def route_identity(target: Mapping[str, Any]) -> Dict[str, str]:
    """The exact route a receipt may replay on: provider, endpoint, API, connection and model."""
    endpoint = normalize_endpoint(target.get("base_url"))
    model = str(target.get("resolved_model") or "").strip()
    if (str(target.get("provider") or "") != "openai" or target.get("api_surface") != "responses"
            or not endpoint or not model):
        raise ValueError("Responses custody requires an exact direct OpenAI Responses route")
    return {"provider": "openai", "endpoint_sha256": canonical_sha256(endpoint), "api_surface": "responses",
            "connection": str(target.get("connection") or ""), "model": model}


def message_text(item: Mapping[str, Any]) -> str:
    """The visible text of one assistant ``message`` item (its ``output_text`` parts, in order)."""
    parts = item.get("content") if isinstance(item.get("content"), list) else []
    return "".join(str(part.get("text") or "") for part in parts
                   if isinstance(part, Mapping) and part.get("type") == "output_text")


def visible_text(items: Sequence[Any]) -> str:
    """The canonical ``content`` of a reply: its assistant messages' text, a blank line between messages."""
    texts = [message_text(item) for item in items if isinstance(item, Mapping) and item.get("type") == "message"]
    return "\n\n".join(text for text in texts if text)


def function_calls(items: Sequence[Any]) -> List[Dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, Mapping) and item.get("type") == "function_call"]


def retain_items(message: Mapping[str, Any], items: Sequence[Any], target: Mapping[str, Any],
                 status: str) -> Dict[str, Any]:
    """The canonical reply with its exact output items attached, whatever its status. Only a completed turn
    whose canonical text and calls equal the items ever replays; any other one is retained, never re-sent."""
    out = copy.deepcopy(dict(message))
    if not items:
        return out
    encoded = _items_json(items)
    out[RESPONSES_RECEIPT_KEY] = {
        "version": _VERSION, "route": route_identity(target), "status": str(status or ""),
        "items_sha256": _sha256(encoded), "items_json": encoded,
        "call_ids": [str(call.get("call_id") or "") for call in function_calls(items)],
    }
    return out


def _canonical_matches(message: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> bool:
    """The canonical row still says what the items said: the same text and the same calls in order."""
    if str(message.get("content") or "") != visible_text(items):
        return False
    native = function_calls(items)
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or len(calls) != len(native):
        return False
    for call, item in zip(calls, native):
        function = call.get("function") if isinstance(call, Mapping) else None
        if not (isinstance(function, Mapping) and call.get("id") == item.get("call_id")
                and function.get("name") == item.get("name") and function.get("arguments") == item.get("arguments")):
            return False
    return True


def validated_items(message: Mapping[str, Any],
                    target: Optional[Mapping[str, Any]] = None) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """``(receipt, items)`` of an intact completed receipt still bound to its canonical row (and to ``target``'s
    exact route when given), else None."""
    raw = message.get(RESPONSES_RECEIPT_KEY) if isinstance(message, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != _RECEIPT_FIELDS or raw.get("version") != _VERSION:
        return None
    if raw.get("status") != "completed":
        return None
    if target is not None:
        try:
            if dict(raw.get("route") or {}) != route_identity(target):
                return None
        except ValueError:
            return None
    encoded = raw.get("items_json")
    if not isinstance(encoded, str) or _sha256(encoded) != raw.get("items_sha256"):
        return None
    try:
        items = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        return None
    if [str(call.get("call_id") or "") for call in function_calls(items)] != list(raw.get("call_ids") or []):
        return None
    if not _canonical_matches(message, items):
        return None
    return dict(raw), items


def native_items_for_replay(message: Mapping[str, Any], target: Dict[str, Any],
                            matching_result_ids: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    """The exact items of a completed same-route turn whose following tool results are exactly its calls;
    None sends the canonical projection instead. Records the replayed digest on the per-call ``target``."""
    validated = validated_items(message, target)
    if validated is None:
        return None
    receipt, items = validated
    if list(matching_result_ids) != list(receipt["call_ids"]):
        return None
    replayed = target.setdefault(REPLAYED_TARGET_KEY, [])
    if receipt["items_sha256"] not in replayed:
        replayed.append(receipt["items_sha256"])
    return copy.deepcopy(items)


def mark_consumed(message: Dict[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind the receipts this request replayed to its answered reply (releases them for compaction)."""
    consumed = [digest for digest in target.get(REPLAYED_TARGET_KEY) or () if isinstance(digest, str)]
    if consumed:
        message[RESPONSES_CONSUMED_KEY] = list(dict.fromkeys(consumed))
    return message


def responses_tool_unit_active(messages: Sequence[Mapping[str, Any]], assistant_index: int,
                               result_end_index: int) -> bool:
    """True for a replayable turn until a later answered request consumed its replay."""
    if not (0 <= assistant_index < len(messages)):
        return False
    validated = validated_items(messages[assistant_index])
    if validated is None:
        return False
    digest = validated[0]["items_sha256"]
    for later in messages[result_end_index + 1:]:
        consumed = later.get(RESPONSES_CONSUMED_KEY) if isinstance(later, Mapping) else None
        if isinstance(consumed, list) and digest in consumed:
            return False
    return True


def receipt_projection(receipt: Any) -> Dict[str, Any]:
    """Type, order, size and digest of a receipt's items, never their values."""
    encoded = receipt.get("items_json") if isinstance(receipt, Mapping) else None
    if not isinstance(encoded, str):
        return {"item_types": [], "size_bytes": 0, "sha256": ""}
    try:
        items = json.loads(encoded)
    except (TypeError, ValueError):
        items = []
    return {"item_types": [str(item.get("type") or "opaque") for item in items if isinstance(item, Mapping)]
            if isinstance(items, list) else [], "status": str(receipt.get("status") or ""),
            "size_bytes": len(encoded.encode("utf-8")), "sha256": _sha256(encoded)}


def receipt_size_bytes(message: Mapping[str, Any]) -> int:
    """Bytes of the items a receipt would replay, for the context estimator's proxy."""
    raw = message.get(RESPONSES_RECEIPT_KEY) if isinstance(message, Mapping) else None
    encoded = raw.get("items_json") if isinstance(raw, Mapping) else None
    return len(encoded.encode("utf-8")) if isinstance(encoded, str) else 0
