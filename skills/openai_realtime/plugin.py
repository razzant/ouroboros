"""OpenAI wire layout, catalog and usage mapping. No host content selection."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from ouroboros.contracts.realtime import (
    CatalogModel, CatalogVoice, ProviderCatalog, RealtimeRoute, RealtimeSetupError, RouteFacts, canonical_json,
)

# One address for both the connection and the declared route identity.
ENDPOINT = "wss://api.openai.com/v1/realtime"
ROUTE = RealtimeRoute(accounting_provider="openai_realtime", endpoint=ENDPOINT, model_prefix="openai::")
# One text probe was accepted and answered at 133,554 input tokens on 20.10.2026, above the
# documented 128,000. By the owner's rule (20.10) a successful request is the window's lower
# bound, never a ceiling below it, so the route states that measurement; spoken retention is
# not yet measured and a larger accepted input in a real call is recorded as evidence.
CATALOG = ProviderCatalog(
    label="OpenAI", credential_settings=("OPENAI_API_KEY",),
    privacy="Call audio and the frames you show go to OpenAI.",
    models=(CatalogModel(
        "gpt-realtime-2.1", "gpt-realtime-2.1 \u2014 OpenAI\u2019s realtime voice model",
        RouteFacts(conversation_window_tokens=133_554, accepted_input_tokens=133_554, acceptance_can_raise=True, working_reserve_tokens=10_000,
                   basis="provider", source="accepted and answered at 133,554 input tokens (text probe 20.10.2026; "
                   "documented context 128,000); spoken retention not yet measured",
                   observed="2026-10-20"),
        default=True, vision=True,
        voices=tuple(CatalogVoice(name, name.capitalize()) for name in (
            "alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage", "shimmer", "verse")),
        thinking_levels=("minimal", "low", "medium", "high", "xhigh"),
        honours=frozenset({"pause", "interruptions", "languages", "thinking"}),
        notes=(("search", "OpenAI Realtime has no built-in search here."),)),))

# Host bookkeeping is retained in RealtimeInput, not model text. Unknown fields
# (including name/tool_calls) refuse instead of quietly losing their semantics.
_MESSAGE_METADATA = {
    "_stable_prefix_blocks", "_host_context_kind", "_host_text_sha256",
    "_review_history_source", "_review_context_index",
}
# Session and response ``max_output_tokens`` accept an integer 1-4096 or "inf",
# the model's maximum (API reference, Realtime, read 2026-10-20). That is field
# encoding, not the model's output capability, which stays route evidence.
_OUTPUT_FIELD_MAX = 4096
# Observed at this endpoint on 2026-10-20: conversation.item.create rejected a
# 299039-character text with string_above_max_length, maximum 256000. This is
# field packaging, not a memory/token limit; preserve every character and role.
_ITEM_TEXT_MAX = 256000


def _parts(message):
    if not isinstance(message, dict) or set(message) - {"role", "content", *_MESSAGE_METADATA}:
        raise RealtimeSetupError("message_shape", "unsupported message fields")
    role = message.get("role")
    if role not in {"system", "user", "assistant"}:
        raise RealtimeSetupError("role", f"cannot preserve role {role!r} in this setup slice")
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or not content:
        raise RealtimeSetupError("content", "expected nonempty text content")
    parts = []
    for part in content:
        if (not isinstance(part, dict) or set(part) - {"type", "text", "cache_control"}
                or part.get("type") != "text" or not isinstance(part.get("text"), str)):
            raise RealtimeSetupError("content", "this setup supports text only; no attachment or block may be dropped")
        # No strip(), join(), summarization, size ceiling or character slicing.
        parts.append({"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]})
    return role, parts


def _tools(raw):
    if not isinstance(raw, list):
        raise RealtimeSetupError("tools", "expected the host-selected tool list")
    result = []
    for tool in raw:
        if not isinstance(tool, dict) or set(tool) - {"type", "function", "cache_control"} or tool.get("type") != "function":
            raise RealtimeSetupError("tools", "only function schemas can be mapped losslessly")
        function = tool.get("function")
        if (not isinstance(function, dict) or set(function) - {"name", "description", "parameters"}
                or not isinstance(function.get("name"), str) or not function["name"]):
            raise RealtimeSetupError("tools", "unsupported function schema fields")
        result.append({"type": "function", **function})
    return result


def layout(selected):
    """Preserve selected roles/text, fragmenting only at observed field limits."""
    messages = json.loads(selected.messages_json)
    if not isinstance(messages, list) or not messages:
        raise RealtimeSetupError("messages", "the complete selected message list is required")
    parsed = [_parts(message) for message in messages]  # validate ALL before opening a socket
    first_role, first_parts = parsed[0]
    if first_role != "system":
        raise RealtimeSetupError("instructions", "the ordinary leading system message is required")
    tools = _tools(json.loads(selected.tools_json))
    # Initial input is acknowledged before VAD or a host-requested response starts.
    session = {
        "type": "realtime", "instructions": first_parts[0]["text"], "tools": tools,
        "truncation": "disabled", "output_modalities": ["text"],
        "max_output_tokens": "inf",
        "audio": {"input": {"turn_detection": None}},
    }
    events = [{"type": "session.update", "event_id": "setup_session", "session": session}]

    def append_item(role, parts):
        if any(len(part["text"]) > _ITEM_TEXT_MAX for part in parts):
            for part in parts:
                text = part["text"]
                for start in range(0, len(text), _ITEM_TEXT_MAX) or (0,):
                    append_item(role, [{**part, "text": text[start:start + _ITEM_TEXT_MAX]}])
            return
        index = len(events)
        events.append({
            "type": "conversation.item.create", "event_id": f"setup_event_{index}",
            "previous_item_id": events[-1]["item"]["id"] if "item" in events[-1] else "root",
            "item": {"id": f"setup_item_{index}", "type": "message", "role": role, "content": parts},
        })

    for part in first_parts[1:]:
        append_item("system", [part])
    for role, parts in parsed[1:]:
        append_item(role, parts)
    wire = canonical_json({"model": selected.model, "input_sha256": selected.sha256, "events": events})
    return events, wire


def _usage(response):
    """Reported axes plus unbilled video; totals do not witness a modality split."""
    raw = response.get("usage") or {}
    inputs, outputs = raw.get("input_token_details") or {}, raw.get("output_token_details") or {}
    values = {"text_input_tokens": inputs.get("text_tokens"),
              "audio_input_tokens": inputs.get("audio_tokens"),
              "image_input_tokens": inputs.get("image_tokens"),
              "video_input_tokens": 0,  # Frames are billed as images, never video.
              "cached_input_tokens": inputs.get("cached_tokens"),
              "text_output_tokens": outputs.get("text_tokens"),
              "audio_output_tokens": outputs.get("audio_tokens"),
              "thinking_tokens": 0}  # Reasoning is already billed inside text output; raw usage retains it.
    return {axis: value for axis, value in values.items() if type(value) is int and value >= 0}


def register(api):
    @asynccontextmanager
    async def connect_openai(model):
        # No reads during registration, preview or injected-transport tests.
        key = api.get_settings(["OPENAI_API_KEY"]).get("OPENAI_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise RealtimeSetupError("credential", "OpenAI Realtime requires a granted OPENAI_API_KEY")
        from websockets.legacy.client import connect

        async with connect(
            ENDPOINT + "?" + urlencode({"model": model}),
            extra_headers={"Authorization": "Bearer " + key},
            open_timeout=10, close_timeout=3, max_size=None,
        ) as ws:
            yield ws

    from .live import OpenAISession

    def live(selected, options, *, observe, connect=None):
        return OpenAISession(selected, options, observe=observe, connect=connect or connect_openai)

    api.register_realtime_provider("openai", route=ROUTE, session=live, catalog=CATALOG)
