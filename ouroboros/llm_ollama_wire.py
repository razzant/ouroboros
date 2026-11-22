"""Lossless supported chat shapes at Ollama's native JSON/NDJSON boundary.

Remote media and unsupported blocks fail before dispatch: fetching arbitrary URLs
or quietly dropping an attachment would change the caller's input contract.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import copy
import json
import re
import uuid

import httpx

from ouroboros.llm_stream import (
    IncompleteProviderStream, ProviderStreamError, RejectedProviderStream, _tool_call_problem, retain_stream,
)
from ouroboros.transport_custody import ProviderNotDispatched


# Only the native /api/chat boundary types these protocol errors. Generic
# OpenAI-compatible prose is still classified solely by context_budget.
_NATIVE_CONTEXT_ERROR = re.compile(
    r"request \(\d+ tokens\) exceeds the available context size \(\d+ tokens\), try increasing it"
    r"|the prompt is longer than the context length currently available to the model"
    r"(?:; shorten the prompt, adjust the context length in settings, or use a model with a longer context length)?")


def native_error_body(body):
    """Type native Ollama's exact size verdict; leave other bodies and words intact.

    No numeric extraction here: provider_answer reads the normalized body and
    size_refusal extracts counts only after upstream recognizes its type.
    """
    message = body.get("error") if isinstance(body, dict) else None
    if isinstance(message, str) and _NATIVE_CONTEXT_ERROR.fullmatch(message):
        return {**body, "error": {"code": 400, "type": "ollama_context_length_exceeded", "message": message}}
    return body


class OllamaIntegrityError(ValueError):
    """A native protocol value could not be used.

    ``retryable`` is read by ``loop_llm_call.classify_llm_exception``: the same
    request to the same server gives the same verdict, so only the classes whose
    cause can change by itself (a busy store, a transient discovery read, the
    nondeterministic length of a generation) ask for a repeat."""
    code, kind, retryable = "ollama_invalid_response", "provider_error", False


class OllamaRouteUnsettledError(OllamaIntegrityError, ProviderNotDispatched):
    """Discovery or its durable record could not finish; nothing was sent to /v1."""
    code, kind, retryable = "ollama_route_unsettled", "provider_transient", True


class OllamaUnsupportedRequest(OllamaIntegrityError, ProviderNotDispatched):
    """A request shape the native API cannot carry without losing part of it."""
    code, kind = "ollama_unsupported_request", "bad_request"


def _content_rows(role, content, image_urls):
    if content is None or isinstance(content, str):
        return [{"role": role, "content": content or ""}]
    if not isinstance(content, list):
        raise OllamaUnsupportedRequest("Unsupported Ollama message content")
    parts = []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"role": role, "content": block["text"]})
        elif kind == "image_url":
            source = block.get("image_url")
            url = source.get("url") if isinstance(source, dict) else source
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise OllamaUnsupportedRequest("Ollama native chat requires inline base64 images")
            image_urls.append(url)
            parts.append({"role": role, "content": "", "images": [url.split(";base64,", 1)[1]]})
        else:
            raise OllamaUnsupportedRequest(f"Unsupported Ollama message block: {kind}")
    # Ollama's own compatible reader splits mixed content into same-role messages:
    # one image array next to concatenated text cannot say which caption belongs where.
    if any("images" in part for part in parts):
        return parts
    separator = "" if role == "tool" else "\n"
    return [{"role": role, "content": separator.join(part["content"] for part in parts)}]


def native_messages(messages, image_urls=None):
    from ouroboros.llm_messages import project_wire_messages

    result, names = [], {}
    if image_urls is None:
        image_urls = []
    for message in project_wire_messages(messages):
        parts = _content_rows(message["role"], message.get("content"), image_urls)
        row = parts[-1]
        if message.get("reasoning"):
            row["thinking"] = message["reasoning"]
        if message.get("tool_calls"):
            row["tool_calls"] = []
            for call in message["tool_calls"]:
                function = copy.deepcopy(call["function"])
                args = function.get("arguments", {})
                try:
                    function["arguments"] = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError as error:
                    raise OllamaUnsupportedRequest("Ollama tool arguments must be valid JSON") from error
                if not isinstance(function["arguments"], dict):
                    raise OllamaUnsupportedRequest("Ollama tool arguments must be a JSON object")
                row["tool_calls"].append({"id": call["id"], "function": function})
                names[call["id"]] = function["name"]
        if message["role"] == "tool":
            name = message.get("tool_name") or message.get("name") or names.get(message.get("tool_call_id"))
            if not name:
                raise OllamaUnsupportedRequest("Ollama tool result has no matching call name")
            for part in parts:
                part["tool_name"] = name
                if message.get("tool_call_id"):
                    part["tool_call_id"] = message["tool_call_id"]
        result.extend(parts)
    return result


def image_context(payload, urls):
    """Put the exact source URLs back on the final image rows for sizing and refusal attribution."""
    if not urls:
        return None
    rows = list(payload["messages"])
    positions = (i for i, row in enumerate(rows) if "images" in row)
    for i, url in zip(positions, urls, strict=True):
        row = dict(rows[i])
        if row.pop("images") != [url.split(";base64,", 1)[1]]:
            raise OllamaIntegrityError("Ollama image changed during request preparation")
        parts = [{"type": "text", "text": row["content"]}] if row["content"] else []
        row["content"] = [*parts, {"type": "image_url", "image_url": {"url": url}}]
        rows[i] = row
    return {**payload, "messages": rows}


def payload_for(target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                *, stream, response_format, window, image_urls=None):
    from ouroboros.llm_attempt import bound_reply_allowance
    from ouroboros.openai_chat_dispatch import sanitize_function_tools
    logical = {"model": target["resolved_model"], "messages": messages, "max_tokens": max_tokens}
    if tools and tool_choice != "none":
        if tool_choice != "auto":
            raise OllamaUnsupportedRequest("Ollama native chat does not support forced tool_choice")
        logical["tools"] = sanitize_function_tools(tools, description_normalizer=lambda value: str(value or ""))
    logical = bound_reply_allowance(target, logical)
    payload = {"model": logical["model"], "messages": native_messages(messages, image_urls), "stream": bool(stream),
               "truncate": False, "shift": False, "options": {"num_predict": logical["max_tokens"]}}
    if window:
        payload["options"]["num_ctx"] = window
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    mode = target.get("ollama_thinking_mode")
    if reasoning_effort == "none":
        payload["think"] = False
    elif reasoning_effort and mode:
        # v0.40.16 openai.ThinkingFromReasoningEffort. A descriptor may name its own levels.
        legacy = {"minimal": "low", "xhigh": "max", "ultra": "max"}.get(reasoning_effort, reasoning_effort)
        known_level = legacy in {"low", "medium", "high", "max"}
        if mode == "legacy":
            if not known_level:
                raise OllamaUnsupportedRequest(f"Unsupported Ollama reasoning effort: {reasoning_effort}")
            payload["think"] = legacy
        else:
            payload["think"] = True if mode == "boolean" and known_level else reasoning_effort
    if "tools" in logical:
        payload["tools"] = logical["tools"]
    if response_format is not None:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        if kind == "json_object":
            payload["format"] = "json"
        elif kind == "json_schema":
            spec = response_format.get("json_schema")
            if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
                raise OllamaUnsupportedRequest("Ollama json_schema requires a schema object")
            payload["format"] = spec["schema"]
        elif kind != "text":
            raise OllamaUnsupportedRequest("Unsupported Ollama response_format")
    return payload


class Reply:
    """Accumulate native frames; a missing terminal receipt is never a success."""
    def __init__(self):
        self.text, self.thinking, self.calls = [], [], []
        self.final = self.usage = None
        self.assembled = None  # the settled reply; set only with its retained receipt

    @contextmanager
    def stream(self, *, ndjson=True):
        """Keep received bytes even on a socket error between two native frames."""
        from ouroboros.usage_accounting import last_physical_attempt_capture
        raw = bytearray()
        start = 0

        def feed(chunk):
            nonlocal start
            search = len(raw)
            raw.extend(chunk)
            if not ndjson:
                return False  # one JSON document, possibly pretty-printed
            while (end := raw.find(b"\n", search)) >= 0:
                line = raw[start:end]
                start = search = end + 1
                if line.strip():
                    self.add_line(line)
                    if self.final is not None:
                        return True
            return False

        capture = last_physical_attempt_capture()
        facts = {"attempt_id": str(getattr(capture, "attempt_id", "") or "")}
        reset = False
        try:
            try:
                yield feed
            except httpx.RequestError:
                reset = True
                # A reset can follow a complete last object without its optional final newline.
                # Parse any tail once. A partial frame or missing done still holds the attempt.
                if self.final is None and start == len(raw):
                    raise
            if self.final is None and start < len(raw):
                line = raw[start:]
                if line.strip():
                    try:
                        frame = json.loads(line)
                    except ValueError as exc:
                        raise IncompleteProviderStream("Ollama response ended within a JSON object") from exc
                    start = len(raw)
                    self.add(frame)
            if self.final is None:
                raise IncompleteProviderStream("Ollama response ended without done:true")
            self.assembled = self.response()
        except BaseException as exc:
            if not raw and isinstance(exc, httpx.HTTPError):
                raise  # the transport or HTTP refusal owns the outcome before a reply body
            if isinstance(exc, asyncio.CancelledError) and self._settle_buffered_terminal(raw, start, facts):
                raise  # the paid answer is settled and retained; the cancellation still reaches its caller
            received = any(self.text) or any(self.thinking) or bool(self.calls)
            unfinished = self.final is None and bool(raw[start:].strip()) and not isinstance(exc, ProviderStreamError)
            error = exc
            if self.final is not None and isinstance(exc, OllamaIntegrityError):
                error = RejectedProviderStream(str(exc), usage=self.usage)
            elif self.final is None and (isinstance(exc, httpx.RequestError)
                    or received and isinstance(exc, (OllamaIntegrityError, ProviderStreamError))):
                error = IncompleteProviderStream("Ollama stream ended before a valid terminal frame")
            if (received or unfinished) and isinstance(error, IncompleteProviderStream):
                error.retain_partial_reply = True
            partial = {"content": "".join(self.text), "thinking": "".join(self.thinking), "tool_calls": self.calls}
            retain_stream(raw, partial, {**facts, "complete": self.final is not None,
                          "model_output_received": received or unfinished}, error)
            if error is not exc:
                raise error from exc
            raise
        if not ndjson and not reset:
            return  # Ordinary JSON already settles through the physical attempt.
        self._retain_terminal(raw, facts)

    def _retain_terminal(self, raw, facts):
        receipt = retain_stream(raw, self.assembled, {**facts, "complete": True})
        self.assembled["_stream_receipt"] = receipt
        if receipt.get("manifest_ref"):
            self.assembled["usage"]["cost_evidence"] = {"stream_receipt": receipt["manifest_ref"]}

    def _settle_buffered_terminal(self, raw, start, facts) -> bool:
        """Settle a complete terminal reply that a caller cancellation overtook.

        A Pause/Stop can land after the terminal bytes were buffered but before
        the transport reported EOF (nonstream) or while the response was closing
        (NDJSON). When the buffer holds one complete, valid terminal reply, the
        original attempt keeps exactly that paid answer and its usage, retained
        under this attempt's receipt; a partial byte prefix or an invalid
        terminal stays under unknown-outcome custody. No network read happens
        here, so the caller's cancellation is never delayed."""
        try:
            if self.final is None:
                tail = raw[start:]
                frame = json.loads(tail) if tail.strip() else None
                if not (isinstance(frame, dict) and frame.get("done") is True):
                    return False  # genuinely partial: the attempt stays uncertain
                self.add(frame)
            self.assembled = self.response()
        except Exception:
            self.assembled = None
            return False
        self._retain_terminal(raw, facts)
        return True

    def add_line(self, line):
        try:
            frame = json.loads(line)
        except ValueError as error:
            raise OllamaIntegrityError("Ollama sent invalid NDJSON; no complete response could be read") from error
        self.add(frame)

    def add(self, frame):
        if not isinstance(frame, dict):
            raise OllamaIntegrityError("Ollama response frame must be a JSON object")
        if self.final is not None:
            raise OllamaIntegrityError("Ollama sent data after its terminal frame")
        if frame.get("error"):
            normalized = native_error_body(frame)
            if normalized is frame:
                normalized = {"error": {"message": str(frame["error"]), "type": "ollama_stream_error"}}
            raise ProviderStreamError(normalized)
        if frame.get("done") is True:
            self.final = frame
            self._terminal_usage(frame)
        message = frame.get("message", {})
        if not isinstance(message, dict):
            raise OllamaIntegrityError("Ollama message is not an object")
        text, thinking, calls = message.get("content", ""), message.get("thinking", ""), message.get("tool_calls", [])
        if not isinstance(text, str) or not isinstance(thinking, str) or not isinstance(calls, list):
            raise OllamaIntegrityError("Ollama message has invalid content, thinking or tool calls")
        self.text.append(text)
        self.thinking.append(thinking)
        self.calls.extend(calls)
    def _terminal_usage(self, frame):
        def count(key):
            value = frame.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise OllamaIntegrityError(f"Ollama omitted or invalidated {key}")
            return value

        prompt, output = count("prompt_eval_count"), count("eval_count")
        self.usage = {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}
        if "prompt_eval_cached_count" in frame:
            cached = count("prompt_eval_cached_count")
            if cached > prompt:
                raise OllamaIntegrityError("Ollama cache count exceeds its total input count")
            self.usage["prompt_tokens_details"] = {"cached_tokens": cached}

    def response(self):
        if self.final is None:
            raise OllamaIntegrityError("Ollama connection ended without a complete response")
        message = {"role": "assistant", "content": "".join(self.text)}
        if any(self.thinking):
            message["reasoning"] = "".join(self.thinking)
        if self.calls:
            message["tool_calls"] = []
            for call in self.calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
                    raise OllamaIntegrityError("Ollama returned invalid tool arguments")
                identity = call.get("id", "")
                normalized = {"id": identity if identity != "" else "call_" + uuid.uuid4().hex,
                              "type": "function", "function": {
                                  **function, "arguments": json.dumps(function["arguments"], ensure_ascii=False)}}
                if problem := _tool_call_problem(normalized):
                    raise OllamaIntegrityError(f"Ollama tool call: {problem}")
                message["tool_calls"].append(normalized)
        reason = self.final.get("done_reason")
        if reason not in ("stop", "length"):
            raise OllamaIntegrityError(f"Ollama chat has no valid finish reason: {reason!r}")
        if reason == "stop" and self.calls:
            reason = "tool_calls"
        return {"model": self.final.get("model"), "choices": [{"message": message, "finish_reason": reason}],
                "usage": self.usage}
