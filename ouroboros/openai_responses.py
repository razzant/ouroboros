"""Direct OpenAI Responses: one connection's API under the shared client.

A connection whose ``api_surface`` is ``responses`` (``provider_models.openai_api_surface``) sends through the
same ``LLMClient``, physical-attempt accounting, wire recovery and normalization as Chat; only the physical
body and its reading differ. The canonical transcript stays Chat-shaped and provider neutral:
``responses_payload`` projects it to Responses input items (a completed same-route turn replays its exact
items, ``openai_responses_custody``), and ``completion_view`` reads one Response as the canonical normalizer
reads a completion, with the facts only Responses carries (terminal status, phases, withheld calls, the
server's ``input_transformations``) attached by ``attach_reply_facts``.

The reply's usability is its terminal fact, never its words (``_usage_response.reply_usability``): a completed
reply ending in ``commentary`` is healthy intermediate progress, and one carrying the model's typed ``refusal`` part
and no call ends as that refusal (finish ``refusal``, its words kept); ``incomplete`` keeps its documented reason
(``max_output_tokens`` stays an output cut; ``content_filter``, ``max_messages``, ``steered`` or no reason make
the reply no result); a failed or other non-terminal reply is a provider body error. Calls of any reply that
did not complete are withheld, disclosed, never executable.

Nothing here chooses a model, an API or a retry: no server Conversations, ``previous_response_id``,
background jobs or hosted tools; ``store`` is always false (no Responses state kept on the server, not a
promise about the provider's other retention). ``strict`` omitted on a function is sent ``false`` so optional
and missing arguments keep working; an authored value is sent as written.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ouroboros.openai_responses_custody import (
    mark_consumed,
    message_text,
    native_items_for_replay,
    retain_items,
    visible_text,
)

RESPONSES = "responses"
# Responses ``call_id`` is at most 64 characters; a longer canonical id from another route is projected
# deterministically on both its call and its result, so the pair stays matched inside one request.
_CALL_ID_MAX = 64
_IMAGE_TYPES = frozenset({"image_url", "image"})
# Failed-response codes a re-send can outlive; every other code is the provider's permanent verdict.
_TRANSIENT_ERROR_CODES = {"server_error": 500, "rate_limit_exceeded": 429, "vector_store_timeout": 504}
# Host-owned reply facts this adapter writes; a provider usage extension never supplies them.
REPLY_FACT_KEYS = ("response_status", "response_incomplete_reason", "response_withheld_output",
                   "response_input_transformations", "response_reasoning_context", "response_unexpected_items")


class ResponsesProjectionError(ValueError):
    """A canonical row this API cannot carry: refused before any send, never silently dropped."""


def responses_route(target: Mapping[str, Any]) -> bool:
    """Whether this per-call target is a direct OpenAI connection on Responses."""
    return str(target.get("provider") or "") == "openai" and target.get("api_surface") == RESPONSES


def response_body(response: Any) -> Dict[str, Any]:
    """The Response exactly as the wire carried it: unset SDK fields stay absent, so replayed items
    never gain the nulls a full dump would add."""
    try:
        body = response.model_dump(exclude_unset=True)
    except TypeError:
        body = response.model_dump()
    return body if isinstance(body, dict) else {}


# --- request ------------------------------------------------------------------------------------------

def _call_id(value: Any) -> str:
    text = str(value or "")
    return text if len(text) <= _CALL_ID_MAX else "call_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def _image_part(block: Mapping[str, Any]) -> Dict[str, Any]:
    from ouroboros.vision_routing import _image_url_from_block

    url = _image_url_from_block(dict(block))
    if not url:
        raise ResponsesProjectionError("an image part carries no image URL or bytes")
    nested = block.get("image_url")
    detail = nested.get("detail") if isinstance(nested, Mapping) else block.get("detail")
    return {"type": "input_image", "image_url": url, **({"detail": detail} if detail else {})}


def _content_parts(content: Any) -> Any:
    """One input message's or tool result's content: text as written, images with their bytes and detail."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesProjectionError(f"content of type {type(content).__name__} has no Responses form")
    parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"type": "input_text", "text": block})
            continue
        kind = str(block.get("type") or "") if isinstance(block, Mapping) else ""
        if kind in {"text", "input_text"}:
            parts.append({"type": "input_text", "text": str(block.get("text") or "")})
        elif kind in _IMAGE_TYPES or kind == "input_image":
            parts.append(_image_part(block))
        else:
            raise ResponsesProjectionError(f"content part {kind or type(block).__name__!r} has no Responses input form")
    return parts


def _assistant_text(content: Any) -> str:
    """Visible assistant text of a canonical row; another provider's thinking blocks stay on their route."""
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text") or "") for block in content or ()
                   if isinstance(block, Mapping) and block.get("type") in {"text", "output_text"})


def _assistant_items(message: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """A canonical assistant turn without item ids or phase: Chat-era, other-route, folded or edited history."""
    items: List[Dict[str, Any]] = []
    text = _assistant_text(message.get("content"))
    if text:
        items.append({"role": "assistant", "content": text})
    for call in message.get("tool_calls") or ():
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise ResponsesProjectionError("a canonical tool call carries no function")
        arguments = function.get("arguments")
        items.append({"type": "function_call", "call_id": _call_id(call.get("id")), "name": str(function.get("name") or ""),
                      "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})})
    return items


def input_items(target: Dict[str, Any], messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The canonical transcript as Responses input, every role in its own place and order."""
    items: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        if role == "assistant":
            results = []
            for later in messages[index + 1:]:
                if str(later.get("role") or "") != "tool":
                    break
                results.append(str(later.get("tool_call_id") or ""))
            native = native_items_for_replay(message, target, results)
            items.extend(native if native is not None else _assistant_items(message))
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": _call_id(message.get("tool_call_id")),
                          "output": _content_parts(message.get("content"))})
        elif role in {"system", "developer", "user"}:
            items.append({"role": role, "content": _content_parts(message.get("content"))})
        else:
            raise ResponsesProjectionError(f"canonical role {role!r} has no Responses input item")
    return items


def _function_tool(tool: Mapping[str, Any]) -> Dict[str, Any]:
    function = tool["function"]
    return {"type": "function", "name": function["name"], "description": function.get("description") or "",
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            "strict": function["strict"] if "strict" in function else False}


def _tool_choice(choice: Any) -> Any:
    if isinstance(choice, Mapping) and isinstance(choice.get("function"), Mapping):
        return {"type": "function", "name": str(choice["function"].get("name") or "")}
    return choice


def _text_format(response_format: Mapping[str, Any]) -> Dict[str, Any]:
    """Chat's ``response_format`` in Responses' ``text.format``: a JSON schema's members move up one level."""
    if response_format.get("type") == "json_schema" and isinstance(response_format.get("json_schema"), Mapping):
        return {"type": "json_schema", **dict(response_format["json_schema"])}
    return dict(response_format)


def responses_payload(target: Dict[str, Any], messages: Sequence[Mapping[str, Any]], *, effort: str,
                      max_tokens: int, tool_choice: Any, temperature: Optional[float],
                      tools: Sequence[Mapping[str, Any]], response_format: Optional[Mapping[str, Any]],
                      cache_identity: str, stream: bool) -> Dict[str, Any]:
    """One Responses body from canonical rows and the shared request options (``tools`` already sanitized)."""
    payload: Dict[str, Any] = {"model": str(target.get("resolved_model") or ""), "input": input_items(target, messages),
                               "max_output_tokens": max_tokens, "store": False, "reasoning": {"effort": effort}}
    if tools:
        payload["tools"] = [_function_tool(tool) for tool in tools]
        payload["tool_choice"] = _tool_choice(tool_choice)
    if temperature is not None:
        payload["temperature"] = temperature
    if response_format:
        payload["text"] = {"format": _text_format(response_format)}
    if cache_identity:
        payload["prompt_cache_key"] = cache_identity
    if stream:
        payload["stream"] = True
    return payload


# --- reply --------------------------------------------------------------------------------------------

def _withheld(item: Mapping[str, Any]) -> Dict[str, Any]:
    """A visible output item the canonical message does not carry, as returned minus opaque reasoning."""
    if item.get("type") == "message":
        return {"type": "message", "phase": item.get("phase"), "status": item.get("status"), "text": message_text(item)}
    if item.get("type") == "function_call":
        return {key: item.get(key) for key in ("type", "call_id", "name", "arguments", "status")}
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    return {"type": str(item.get("type") or "opaque"), "id": item.get("id"), "size_bytes": len(encoded.encode("utf-8")),
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def completion_view(body: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(completion, facts)``: the Response as the canonical normalizer reads a chat completion, and the
    Responses-only facts ``attach_reply_facts`` adds after it. Usage stays the provider's own numbers:
    input includes cached input and output includes reasoning; an absent count stays absent, never zero."""
    status = str(body.get("status") or "")
    items = [dict(item) for item in body.get("output") or () if isinstance(item, Mapping)]
    completed = status == "completed"
    calls = [item for item in items if completed and item.get("type") == "function_call"
             and item.get("status") in (None, "completed") and isinstance(item.get("arguments"), str)
             and all(isinstance(item.get(key), str) and item[key].strip() for key in ("call_id", "name"))]
    known = {"message", "function_call", "reasoning"}
    text = visible_text(items)
    refusal = "".join(str(part.get("refusal") or "") for item in items if item.get("type") == "message"
                      for part in item.get("content") or () if isinstance(part, Mapping) and part.get("type") == "refusal")
    phases = [item.get("phase") for item in items if item.get("type") == "message" and message_text(item)]
    incomplete = body.get("incomplete_details") if isinstance(body.get("incomplete_details"), Mapping) else {}
    error = body.get("error") if isinstance(body.get("error"), Mapping) else None
    terminal_failure = status not in {"completed", "incomplete"}
    if completed:  # a typed refusal outranks any commentary before it; complete calls keep their own finish
        finish = ("tool_calls" if calls else "refusal" if refusal.strip()
                  else "commentary" if phases and phases[-1] == "commentary" else "stop")
    elif status == "incomplete":
        finish = str(incomplete.get("reason") or "") or None
    else:
        finish = None
    message: Dict[str, Any] = {"role": "assistant", "content": "" if terminal_failure else text}
    if refusal and not terminal_failure:
        message["refusal"] = refusal
    if calls:
        message["tool_calls"] = [{"id": str(call.get("call_id") or ""), "type": "function", "function": {
            "name": str(call.get("name") or ""), "arguments": call.get("arguments") if isinstance(call.get("arguments"), str)
            else json.dumps(call.get("arguments") or {})}} for call in calls]
    completion: Dict[str, Any] = {"id": body.get("id"), "choices": [{"message": message, **(
        {"finish_reason": finish} if status in {"completed", "incomplete"} else {})}], "model": body.get("model")}
    for key in ("service_tier", "total_cost_usd"):  # the reported charge, zero included, as the ledger reads it
        if key in body:
            completion[key] = body[key]
    if isinstance(body.get("_stream_receipt"), dict):
        completion["_stream_receipt"] = body["_stream_receipt"]
    if terminal_failure:  # the provider's error object kept whole; its code moves to ``type`` under a status class
        code = str((error or {}).get("code") or f"response_{status or 'without_status'}")
        completion["error"] = {**{key: value for key, value in (error or {}).items() if key not in {"code", "type"}},
                               "code": _TRANSIENT_ERROR_CODES.get(code, 400), "type": code,
                               "message": str((error or {}).get("message") or f"Responses reply ended {status or 'without a status'}")}
    raw_usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else None
    if raw_usage is not None:
        usage = copy.deepcopy(dict(raw_usage))
        for source, name in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
            if isinstance(usage.get(source), int) and not isinstance(usage.get(source), bool):
                usage[name] = usage[source]
        details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), Mapping) else {}
        for name in ("cached_tokens", "cache_write_tokens"):
            if isinstance(details.get(name), int) and not isinstance(details.get(name), bool) and details[name] > 0:
                usage[name] = details[name]  # a positive count is lifted; a zero or absent one stays as reported
        completion["usage"] = usage
    shown = {id(call) for call in calls} | (set() if terminal_failure else {
        id(item) for item in items if item.get("type") == "message"})
    withheld = [_withheld(item) for item in items if item.get("type") in {"message", "function_call"} and id(item) not in shown]
    facts: Dict[str, Any] = {"response_status": status, "items": items}
    if status == "incomplete":
        facts["response_incomplete_reason"] = incomplete.get("reason")
    if withheld:
        facts["response_withheld_output"] = withheld
    if isinstance(body.get("input_transformations"), list):
        facts["response_input_transformations"] = copy.deepcopy(body["input_transformations"])
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), Mapping) else {}
    if reasoning.get("context"):
        facts["response_reasoning_context"] = reasoning["context"]
    unexpected = [str(item.get("type") or "opaque") for item in items if item.get("type") not in known]
    if unexpected:
        facts["response_unexpected_items"] = unexpected
    return completion, facts


def attach_reply_facts(message: Dict[str, Any], usage: Dict[str, Any], facts: Mapping[str, Any],
                       target: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The normalized reply with its exact items retained and the Responses-only facts in usage."""
    for key in REPLY_FACT_KEYS:
        usage.pop(key, None)
        if key in facts:
            usage[key] = copy.deepcopy(facts[key])
    message = retain_items(message, facts.get("items") or (), target, str(facts.get("response_status") or ""))
    if facts.get("response_status") in {"completed", "incomplete"}:
        message = mark_consumed(message, target)
    return message, usage
