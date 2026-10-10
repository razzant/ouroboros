"""Wire assembly for completed Chat Completions and native Messages SSE replies.

Doctrine: strict about completeness, tolerant about form. Terminal framing, a
finish reason on every choice and structurally complete tool calls are the only
hard requirements; identity, index and shape irregularities are forgiven
first-wins/skip, and every forgiven fact is disclosed in the stream receipt as
``anomalies``. ``tools/search.py`` (the Responses SSE consumer behind
``web_search``) is the house precedent: it ignores unknown events, settles only
on ``response.completed`` and never retries on form. Form forgiveness is the Chat
assembler's doctrine; the native Messages assembler forgives only post-terminal
shapes and still reports a malformed mid-stream frame as an unknown outcome — a
deliberate boundary (Anthropic's SSE is stable; no recorded case exists), not an
oversight.

Consumption belongs inside the physical send closure. Only protocol-complete
assemblies leave it as responses; partial bytes stay in private observability
custody and never become an assistant message. Received monetary evidence
survives independently of the response's completeness or cancellation.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from typing import Any, Callable


class IncompleteProviderStream(RuntimeError):
    """The wire ended before its terminal framing: the provider outcome is unknown."""

    code = "model_outcome_unknown"
    stream_incomplete = True


class RejectedProviderStream(RuntimeError):
    """Terminal framing arrived, but the assembled body is unusable.

    A deterministic local verdict on complete wire: it carries no ``code`` (it
    never reads as an unknown outcome), keeps ``stream_incomplete`` (wire
    recovery must not resend) and hands the usage frame it read to the ledger.
    """

    stream_rejected = True
    stream_incomplete = True

    def __init__(self, message: str, *, usage: Any = None, anomalies: Any = None):
        super().__init__(message)
        self.stream_usage = copy.deepcopy(usage) if isinstance(usage, dict) and usage else None
        self.anomalies = list(anomalies or [])


class ProviderStreamError(IncompleteProviderStream):
    """An explicit SSE error, distinct from an EOF or a socket failure.

    A provider fact, not an unknown outcome: no ``code``; the body's numeric
    ``error.code`` becomes ``status_code`` so the classifier files it through
    its ordinary status ladder, and a frame without one is ``stream_rejected``.
    """

    code = ""

    def __init__(self, body: dict, *, usage: Any = None):
        self.body = copy.deepcopy(body)
        error = body.get("error")
        error = error if isinstance(error, dict) else {}
        code = error.get("code")
        numeric = isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599
        self.status_code = code if numeric else 200
        # Without an HTTP-shaped code the frame is the provider's own terminal
        # verdict on this stream (an overload/api_error shape, a finish_reason of
        # "error"): filed like a rejected body — provider_error, no same-request
        # repeat, the cross-model chain eligible — never an unknown outcome.
        self.stream_rejected = not numeric
        self.type = str(error.get("type") or error.get("code") or "provider_stream_error")
        self.provider_message = str(error.get("message") or "")
        self.stream_usage = copy.deepcopy(usage) if isinstance(usage, dict) and usage else None
        # Keep producer facts in body/private evidence. Text-only pre-routing
        # classifiers must not turn an HTTP-200 SSE failure into a free rejection.
        super().__init__("Provider reported an SSE error after stream dispatch")


class AssembledResponse:
    """The two existing response readers share this detached, non-iterator value."""

    def __init__(self, body: dict):
        self.body = body

    def model_dump(self):
        return copy.deepcopy(self.body)

    def json(self):
        return self.model_dump()


_IDENTITY_KEYS = frozenset({"type", "role", "format", "id"})
_ANOMALY_KEEP = 16


def _valid_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _index(value: Any) -> int:
    if not _valid_index(value):
        raise IncompleteProviderStream("Stream item lacks a non-negative integer index")
    return value


def _adjacent(last: Any, item: dict) -> bool:
    """Continuation of the last record: ``type`` and ``id`` absent on either side or equal.

    Discreteness of opaque payloads (``reasoning.encrypted`` data, signatures)
    rides on the provider's ``id``: two id-less records of one type would fuse.
    Every recorded encrypted record carries an id; text records must stay
    id-less-mergeable, which is the #856 fix itself.
    """
    return isinstance(last, dict) and all(
        last.get(key) is None or item.get(key) is None or last.get(key) == item.get(key)
        for key in ("type", "id"))


def _snapshot(target: dict, update: dict) -> None:
    """Usage counters and metadata are cumulative snapshots, never token deltas."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _snapshot(target[key], value)
        elif value is not None:
            target[key] = copy.deepcopy(value)


def _delta(target: dict, update: dict, note: Callable[[str], None], *,
           chat_path: tuple | None = None) -> None:
    """Fold a delta; disclose shape/identity conflicts without raising.

    Only Chat messages supply ``chat_path`` (choice, index, then wire fields),
    so call-name compatibility cannot change unrelated names or native deltas.
    """
    for key, value in update.items():
        current = target.get(key)
        if value is None:
            target.setdefault(key, None)
        elif isinstance(value, dict):
            if current is None:
                current = target[key] = {}
            if not isinstance(current, dict):
                note(f"{key}: object delta onto {type(current).__name__}; kept first shape")
                continue
            _delta(current, value, note, chat_path=(*chat_path, key) if chat_path is not None else None)
        elif isinstance(value, list):
            if current is None:
                current = target[key] = []
            if not isinstance(current, list):
                note(f"{key}: list delta onto {type(current).__name__}; kept first shape")
                continue
            for item in value:
                _merge_list_item(key, current, item, note,
                                 chat_path=(*chat_path, key) if chat_path is not None else None)
        elif isinstance(value, str) and key not in _IDENTITY_KEYS:
            if current is not None and not isinstance(current, str):
                note(f"{key}: text delta onto {type(current).__name__}; kept first shape")
                continue
            if key == "name" and current and value and chat_path is not None and (
                    (len(chat_path) == 3 and chat_path[2] == "function_call")
                    or (len(chat_path) == 5 and chat_path[2] == "tool_calls"
                        and chat_path[4] in {"function", "custom"})):
                # Chat call names alone tolerate a repeat of the accumulated value.
                # This compatibility choice interprets a,a as a rather than aa;
                # other fragments still append, without guessing a callable name.
                address = ".".join(str(part) for part in (*chat_path, key))
                if current == value:
                    note(f"{address}: equal name value not appended")
                    continue
                note(f"{address}: different name values concatenated")
            target[key] = (current or "") + value
        elif key in _IDENTITY_KEYS:
            if current is None:
                target[key] = copy.deepcopy(value)
            elif current != value:
                note(f"{key}: {current!r} then {value!r}; kept first")
        elif isinstance(current, (dict, list, str)):
            note(f"{key}: {type(value).__name__} delta onto {type(current).__name__}; kept first shape")
        else:
            target[key] = copy.deepcopy(value)


def _merge_list_item(key: str, current: list, item: Any, note: Callable[[str], None], *,
                     chat_path: tuple | None = None) -> None:
    """List identity is a property of the wire field: ``reasoning_details`` records are
    reassembled by type transition (every delta repeats a frame-local ``index`` that is
    not a record key, as the wire's own reference client documents); every other list
    keys on ``index``, and an index-less tool-call fragment continues the last call."""
    if not isinstance(item, dict):
        current.append(copy.deepcopy(item))
        return
    match = None
    if key == "reasoning_details":
        if "type" not in item:
            note(f"{key}: item without type")
        if current and _adjacent(current[-1], item):
            match = current[-1]
    else:
        index = item.get("index")
        if "index" in item and not _valid_index(index):
            note(f"{key}: index {index!r} is not a non-negative integer; treated as absent")
            index = None
        if index is not None:
            match = next((row for row in current if isinstance(row, dict) and row.get("index") == index), None)
            if match is None:
                match = {"index": index}
                current.append(match)
        elif key == "tool_calls":
            if current and _adjacent(current[-1], item):
                match = current[-1]
            note(f"{key}: item without index; {'merged into the last call' if match else 'appended'}")
    if match is None:
        current.append(copy.deepcopy(item))
    else:
        _delta(match, item, note,
               chat_path=(*chat_path, match.get("index", "last")) if chat_path is not None else None)


def _tool_call_problem(call: Any) -> str:
    """Why a tool call cannot be executed; empty when it is structurally complete."""
    if not isinstance(call, dict):
        return f"is {type(call).__name__}, not an object"
    if not isinstance(call.get("id"), str) or not call["id"]:
        return "id missing"
    # Structural completeness is id + name + arguments; a fragment set that never
    # carried ``type`` is the same shape the non-stream path executes.
    kind = call.get("type") or next((k for k in ("function", "custom") if isinstance(call.get(k), dict)), None)
    if kind not in {"function", "custom"}:
        return f"type {kind!r}"
    payload = call.get(kind)
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str) or not payload["name"]:
        return f"{kind}.name missing"
    field = "arguments" if kind == "function" else "input"
    if not isinstance(payload.get(field), str):
        return f"{kind}.{field} is not text"
    return ""


class _Accumulator:
    """Shared ledger of forgiven wire irregularities: the first few verbatim plus a total."""

    def __init__(self) -> None:
        self.anomalies: list[str] = []
        self.anomaly_count = 0
        self.done = False

    def _note(self, text: str) -> None:
        self.anomaly_count += 1
        if len(self.anomalies) < _ANOMALY_KEEP:
            self.anomalies.append(text)

    def anomaly_facts(self) -> dict:
        return {"count": self.anomaly_count, "first": list(self.anomalies)}


class ChatAccumulator(_Accumulator):
    def __init__(self, expected_choices: int = 1, on_generation_id: Callable | None = None):
        super().__init__()
        self.body: dict = {"object": "chat.completion"}
        self.choices: dict[int, dict] = {}
        self.expected_choices = expected_choices
        self.on_generation_id = on_generation_id

    def accept(self, event: str, data: str) -> None:
        if self.done:
            self._note("data after [DONE]; ignored")
            return
        if data == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            self._note("chunk is not JSON; skipped")
            return
        if not isinstance(chunk, dict):
            self._note(f"chunk is {type(chunk).__name__}, not an object; skipped")
            return
        if self.on_generation_id is not None:
            self.on_generation_id(chunk.get("id"))
        if isinstance(chunk.get("error"), dict):
            _snapshot(self.body, {key: value for key, value in chunk.items()
                                  if key not in {"choices", "usage", "object"}})
            raise ProviderStreamError(chunk, usage=chunk.get("usage") or self.body.get("usage"))
        for key, value in chunk.items():
            if key in {"choices", "object", "obfuscation"} or value is None:
                continue
            if key in {"id", "model"} and self.body.get(key) not in (None, value):
                self._note(f"{key}: {self.body[key]!r} then {value!r}; kept first")
                continue
            _snapshot(self.body, {key: value})
        choices = chunk.get("choices")
        if choices is None:
            return  # A usage/metadata frame: its envelope keys were snapshotted above.
        if not isinstance(choices, list):
            self._note(f"choices is {type(choices).__name__}, not a list; skipped")
            return
        for position, update in enumerate(choices):
            self._accept_choice(position, update)

    def _accept_choice(self, position: int, update: Any) -> None:
        if not isinstance(update, dict):
            self._note(f"choice at position {position} is not an object; skipped")
            return
        index = update.get("index")
        if not _valid_index(index) or (self.expected_choices == 1 and index != 0):
            index = 0 if self.expected_choices == 1 else position
            self._note(f"choice at position {position}: index {update.get('index')!r}; used {index}")
        delta = update.get("delta")
        if not isinstance(delta, dict):
            # A choice may legitimately carry only its terminal facts (a
            # finish_reason frame without a delta); forgive the shape and keep
            # judging completeness from what the frame does say.
            self._note(f"choice {index}: delta is {type(delta).__name__}, not an object; treated as empty")
            delta = {}
        choice = self.choices.setdefault(index, {"index": index, "message": {"role": "assistant", "content": None}})
        # OpenRouter's final usage frame repeats finish_reason and an empty
        # delta. It updates accounting without creating a second answer.
        substantive = any(value not in (None, "", [], {}) for key, value in delta.items() if key != "role")
        if choice.get("finish_reason") and substantive:
            self._note(f"choice {index}: content after finish_reason {choice['finish_reason']!r}; accepted")
        _delta(choice["message"], delta, self._note, chat_path=("choice", index))
        logprobs = update.get("logprobs")
        if isinstance(logprobs, dict):
            _delta(choice.setdefault("logprobs", {}), logprobs, self._note)
        elif logprobs is not None:
            self._note(f"choice {index}: logprobs is {type(logprobs).__name__}, not an object; skipped")
        finish = update.get("finish_reason")
        if finish is not None:
            if not isinstance(finish, str) or not finish:
                self._note(f"choice {index}: finish_reason {finish!r} is not a non-empty string; ignored")
            elif finish == "error":
                raise ProviderStreamError({"error": {"type": "stream_finish_error"}}, usage=self.body.get("usage"))
            elif choice.get("finish_reason") not in (None, finish):
                self._note(f"choice {index}: finish_reason {choice['finish_reason']!r} then {finish!r}; kept first")
            else:
                choice["finish_reason"] = finish
        _snapshot(choice, {key: value for key, value in update.items()
                           if key not in {"delta", "logprobs", "index", "finish_reason"}})

    def result(self) -> dict:
        """Judge once, after terminal framing: unknown outcome vs. unusable body."""
        if not self.done:
            # A body the provider closed cleanly after every choice finished is
            # terminal framing too ([DONE] is the other witness); a close before
            # that is the one unknown outcome this assembler still reports.
            if (set(self.choices) != set(range(self.expected_choices))
                    or any(not choice.get("finish_reason") for choice in self.choices.values())):
                raise IncompleteProviderStream("Stream ended without complete terminal framing")
            self._note("stream closed without [DONE] after every choice finished")
            self.done = True
        if set(self.choices) != set(range(self.expected_choices)):
            self._reject(f"choices {sorted(self.choices)} present, {self.expected_choices} expected")
        body = self.partial()
        for choice in body["choices"]:
            path = f"choice {choice['index']}"
            if not choice.get("finish_reason"):
                self._reject(f"{path}: no finish_reason after the terminal frame")
            message = choice["message"]
            calls = message.get("tool_calls")
            if calls is not None and not isinstance(calls, list):
                self._reject(f"{path}: tool_calls is {type(calls).__name__}, not a list")
            for position, call in enumerate(calls or []):
                problem = _tool_call_problem(call)
                if problem:
                    self._reject(f"{path} tool call {position}: {problem}")
            legacy = message.get("function_call")
            if legacy is not None and (not isinstance(legacy, dict) or not isinstance(legacy.get("name"), str)
                                       or not legacy["name"] or not isinstance(legacy.get("arguments"), str)):
                self._reject(f"{path} function_call: name or arguments incomplete")
        return body

    def _reject(self, detail: str) -> None:
        raise RejectedProviderStream(f"Stream rejected after terminal framing: {detail}",
                                     usage=self.body.get("usage"), anomalies=self.anomalies)

    def partial(self) -> dict:
        body = copy.deepcopy(self.body)
        body["choices"] = [copy.deepcopy(self.choices[key]) for key in sorted(self.choices)]
        for choice in body["choices"]:
            calls = choice["message"].get("tool_calls")
            if isinstance(calls, list) and calls:
                if all(isinstance(call, dict) and _valid_index(call.get("index")) for call in calls):
                    calls.sort(key=lambda call: call["index"])
                for call in calls:
                    if isinstance(call, dict):
                        call.pop("index", None)
        return body


class AnthropicAccumulator(_Accumulator):
    def __init__(self):
        super().__init__()
        self.body: dict = {}
        self.blocks: dict[int, dict] = {}
        self.open_blocks: set[int] = set()
        self.usage_final = False  # message_delta folded its final counters into body["usage"]
        self.inputs: dict[int, str] = {}

    def accept(self, event: str, data: str) -> None:
        if self.done:
            self._note("data after message_stop; ignored")
            return
        try:
            chunk = json.loads(data)
        except ValueError:
            raise IncompleteProviderStream("Native stream chunk is not JSON") from None
        if not isinstance(chunk, dict):
            raise IncompleteProviderStream("Native stream chunk is not an object")
        kind = chunk.get("type")
        if event and event != kind:
            raise IncompleteProviderStream("Native SSE event differs from payload type")
        if kind == "error":
            # message_start's usage is a lower-bound snapshot (final output tokens
            # arrive in message_delta): an aborted native stream keeps its
            # unresolved upper bound rather than settling on an understatement.
            raise ProviderStreamError(chunk)
        if kind == "ping":
            return
        if kind == "message_start":
            if self.body or not isinstance(chunk.get("message"), dict):
                raise IncompleteProviderStream("Invalid native message start")
            self.body = copy.deepcopy(chunk["message"])
            if self.body.get("content"):
                raise IncompleteProviderStream("Native message start contains unexpected blocks")
        elif kind == "content_block_start":
            index = _index(chunk.get("index"))
            if (not self.body or self.body.get("stop_reason") or index in self.blocks
                    or not isinstance(chunk.get("content_block"), dict)):
                raise IncompleteProviderStream("Invalid native block start")
            self.blocks[index] = copy.deepcopy(chunk["content_block"])
            self.open_blocks.add(index)
        elif kind in {"content_block_delta", "content_block_stop"}:
            index = _index(chunk.get("index"))
            if index not in self.open_blocks:
                raise IncompleteProviderStream("Native block delta/stop without an open block")
            block = self.blocks[index]
            if kind == "content_block_stop":
                self.open_blocks.remove(index)
                if index in self.inputs:
                    try:
                        block["input"] = json.loads(self.inputs[index])
                    except ValueError:
                        self._note(f"block {index}: tool input is not JSON; left for the terminal verdict")
                        block["input"] = self.inputs[index]
                return
            delta = chunk.get("delta")
            if not isinstance(delta, dict):
                raise IncompleteProviderStream("Native block lacks delta")
            delta_type = delta.get("type")
            if delta_type == "input_json_delta":
                if block.get("type") not in {"tool_use", "server_tool_use"}:
                    raise IncompleteProviderStream("Native input delta belongs to a non-tool block")
                fragment = delta.get("partial_json")
                if not isinstance(fragment, str):
                    raise IncompleteProviderStream("Native input delta is not text")
                self.inputs[index] = self.inputs.get(index, "") + fragment
            elif delta_type == "citations_delta":
                block.setdefault("citations", []).append(copy.deepcopy(delta["citation"]))
            else:
                _delta(block, {key: value for key, value in delta.items() if key != "type"}, self._note)
        elif kind == "message_delta":
            if not self.body or self.open_blocks:
                raise IncompleteProviderStream("Native message delta before blocks finished")
            _snapshot(self.body, chunk.get("delta") or {})
            _snapshot(self.body.setdefault("usage", {}), chunk.get("usage") or {})
            self.usage_final = True
        elif kind == "message_stop":
            self.done = True
        # Future non-content events are retained in the exact wire evidence.

    def result(self) -> dict:
        """Judge once, after ``message_stop``: unknown outcome vs. unusable body."""
        if not self.done:
            raise IncompleteProviderStream("Native stream ended without complete terminal framing")
        if not self.body:
            self._reject("message_stop without a message_start")
        if self.open_blocks:
            self._reject(f"blocks {sorted(self.open_blocks)} never stopped")
        if not self.body.get("stop_reason"):
            self._reject("no stop_reason after message_stop")
        if set(self.blocks) != set(range(len(self.blocks))):
            self._reject(f"block indices {sorted(self.blocks)} are not contiguous")
        for index, block in self.blocks.items():
            kind = block.get("type")
            if kind in {"tool_use", "server_tool_use"} and (
                    not isinstance(block.get("id"), str) or not block["id"]
                    or not isinstance(block.get("name"), str) or not block["name"]
                    or not isinstance(block.get("input"), dict)):
                self._reject(f"block {index}: tool block lacks id, name or an object input")
            if kind == "thinking" and (not isinstance(block.get("thinking"), str)
                                       or not isinstance(block.get("signature"), str) or not block["signature"]):
                self._reject(f"block {index}: thinking block lacks its complete signature")
        return self.partial()

    def _reject(self, detail: str) -> None:
        # message_start's usage is a lower-bound snapshot: without message_delta the
        # attempt keeps its unresolved upper bound (same policy as the error branch).
        raise RejectedProviderStream(f"Native stream rejected after message_stop: {detail}",
                                     usage=self.body.get("usage") if self.usage_final else None,
                                     anomalies=self.anomalies)

    def partial(self) -> dict:
        return {**copy.deepcopy(self.body), "content": [copy.deepcopy(self.blocks[key]) for key in sorted(self.blocks)]}


class _SSEFrames:
    """Incremental UTF-8 SSE framing; only CR/LF are line delimiters."""

    def __init__(self):
        self.buffer = b""
        self.data: list[str] = []
        self.event = ""
        self.first_line = True

    def feed(self, chunk: bytes, *, final: bool = False):
        self.buffer += chunk
        while match := re.search(b"\r\n|\r|\n", self.buffer):
            if not final and match.group() == b"\r" and match.end() == len(self.buffer):
                break  # CRLF may straddle network chunks.
            line = self.buffer[:match.start()].decode("utf-8")
            self.buffer = self.buffer[match.end():]
            if self.first_line:
                line = line.removeprefix("\ufeff")
                self.first_line = False
            if not line:
                data, event = self.data, self.event
                self.data, self.event = [], ""
                if data:
                    yield event, "\n".join(data)
            elif not line.startswith(":"):
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "data":
                    self.data.append(value)
                elif field == "event":
                    self.event = value


class _StreamAssembly:
    def __init__(self, response: Any, native: bool, expected_choices: int, *, defer_binding: bool = False):
        from ouroboros.usage_accounting import (
            AttemptReservation, current_physical_attempt_drive_root, last_physical_attempt_capture,
        )

        capture, root = last_physical_attempt_capture(), current_physical_attempt_drive_root()
        self.reservation = (AttemptReservation(capture.attempt_id, root, capture.model, capture.provider, None)
                            if capture is not None and root is not None else None)
        self.defer_binding = defer_binding
        self.pending_generations = []
        self.generation_id = ""
        self.generation_conflict = False
        self.conflicting_generation_id = ""
        self.generation_binding_error = ""
        self.generation_bound = False
        self.accumulator = (AnthropicAccumulator() if native else
                            ChatAccumulator(expected_choices, self.observe_generation))
        self.frames = _SSEFrames()
        self.raw: list[bytes] = []
        headers = getattr(response, "headers", {})
        self.observe_generation(headers.get("x-generation-id"))

    def observe_generation(self, generation_id: Any) -> None:
        """Bind once at first observation; preserve one conflicting observation too."""
        if (not isinstance(generation_id, str) or not generation_id
                or generation_id == self.generation_id or self.generation_conflict):
            return
        if self.generation_id:
            self.generation_conflict = True
            self.conflicting_generation_id = generation_id
        else:
            self.generation_id = generation_id
        self.pending_generations.append(generation_id)
        # The prior ID's write cannot certify a newly observed conflict.
        # Parsing can raise before the async caller drains this pending write.
        self.generation_bound = False
        if not self.defer_binding:
            self.bind_pending_generations()

    def bind_pending_generations(self) -> None:
        """One transaction per newly observed ID; async callers join off-loop."""
        pending, self.pending_generations = self.pending_generations, []
        self.generation_bound = False
        try:
            from ouroboros.usage_accounting import bind_provider_generation

            for generation_id in pending:
                bind_provider_generation(generation_id, reservation=self.reservation)
            self.generation_bound = bool(self.reservation) and not self.generation_binding_error
        except Exception as exc:
            # Evidence retention must not turn a usable answer into a retry.
            self.generation_binding_error = type(exc).__name__

    def preserve_failure_usage(self, error: BaseException) -> None:
        """Money survives read/close/cancel errors independently of reply completeness."""
        from ouroboros._usage_response import provider_cost_value

        usage = self.accumulator.body.get("usage")
        usage = copy.deepcopy(usage) if isinstance(usage, dict) else {}
        complete_usage = (self.accumulator.usage_final if isinstance(self.accumulator, AnthropicAccumulator)
                          else self.accumulator.done)
        if not complete_usage:
            # message_start and unfinished Chat streams can carry partial counters.
            # An explicit price remains authoritative even on a partial message.
            if not any(provider_cost_value(usage.get(key)) is not None for key in ("cost", "total_cost")):
                usage = {}
        received = getattr(error, "stream_usage", None)
        if isinstance(received, dict):
            cost = next((value for source in (received, usage) for key in ("cost", "total_cost")
                         if (value := provider_cost_value(source.get(key))) is not None), None)
            _snapshot(usage, received)
            if cost is not None:
                usage["cost"] = cost
        if usage:
            error.stream_usage = usage

    def feed(self, chunk: bytes) -> bool:
        self.raw.append(chunk)
        # The socket's per-phase bound was narrowed at dispatch. A late physical
        # terminal still settles its original attempt; no cadence or wall-clock
        # watchdog abandons an in-flight stream to return its wrapper sooner.
        for event, data in self.frames.feed(chunk):
            self.accumulator.accept(event, data)
        return self.accumulator.done

    def retain(self, *, complete: bool, error: BaseException | None = None) -> dict:
        if error is not None:
            self.preserve_failure_usage(error)
        from ouroboros import config
        from ouroboros.observability import persist_call, write_blob
        from ouroboros.usage_accounting import current_usage_scope, last_physical_attempt_capture, current_physical_attempt_drive_root

        capture = last_physical_attempt_capture()
        scope = current_usage_scope()
        attempt_id = str(getattr(capture, "attempt_id", "") or "")
        facts = {"attempt_id": attempt_id, "complete": complete,
                 "generation_id": self.generation_id or self.accumulator.body.get("id", ""),
                 "anomalies": self.accumulator.anomaly_facts()}
        if self.generation_conflict:
            facts["generation_conflict"] = True
            facts["conflicting_generation_id"] = self.conflicting_generation_id
        if self.generation_bound:
            facts["generation_bound"] = True
        if self.generation_binding_error:
            facts["generation_binding_error"] = self.generation_binding_error
        evidence = {"wire_base64": base64.b64encode(b"".join(self.raw)).decode("ascii"),
                    "partial_assembly": self.accumulator.partial(), **facts}
        try:
            if not attempt_id:
                raise RuntimeError("stream has no physical attempt identity")
            root = current_physical_attempt_drive_root() or (scope.drive_root if scope is not None else config.DATA_DIR)
            # Raw frames can contain private native signatures. Only the blob
            # reference and structural facts enter the ordinary public projection.
            raw_ref = write_blob(root, evidence, kind="json")
            retained = persist_call(root, task_id=str(getattr(scope, "task_id", "") or "llm"),
                                    call_id=f"physical_{attempt_id}_stream", call_type="physical_stream",
                                    payload={**facts, "private_wire_ref": raw_ref}, manifest=facts)
            facts["manifest_ref"] = retained["manifest_ref"]
        except Exception as retention_error:
            facts["retention_error"] = type(retention_error).__name__
            if error is not None:
                error.stream_evidence = evidence
        if error is not None:
            error.stream_receipt = facts
            error.stream_incomplete = True
        return facts

    def result(self) -> dict:
        for event, data in self.frames.feed(b"", final=True):
            self.accumulator.accept(event, data)
        return self.accumulator.result()


def consume_stream(stream: Any, *, native: bool = False, expected_choices: int = 1) -> AssembledResponse:
    response = stream if native else getattr(stream, "response", stream)
    assembly = _StreamAssembly(response, native, expected_choices)
    try:
        chunks = response.iter_content(chunk_size=8192) if native else response.iter_bytes()
        for chunk in chunks:
            if assembly.feed(chunk):
                break
        body = assembly.result()
        response.close()
    except BaseException as exc:
        # A rejected body that reached its terminal frame is complete wire.
        assembly.retain(complete=assembly.accumulator.done, error=exc)
        try:
            response.close()
        except BaseException:
            pass  # Cleanup cannot replace the original stream/cancellation cause.
        raise
    body["_stream_receipt"] = assembly.retain(complete=True)
    return AssembledResponse(body)


async def consume_stream_async(stream: Any, *, expected_choices: int = 1) -> AssembledResponse:
    from ouroboros._usage_wait import presend_off_loop

    response = getattr(stream, "response", stream)
    assembly = _StreamAssembly(response, False, expected_choices, defer_binding=True)
    try:
        if assembly.pending_generations:
            await presend_off_loop(assembly.bind_pending_generations)
        async for chunk in response.aiter_bytes():
            done = assembly.feed(chunk)
            if assembly.pending_generations:
                await presend_off_loop(assembly.bind_pending_generations)
            if done:
                break
        body = assembly.result()
        if assembly.pending_generations:
            await presend_off_loop(assembly.bind_pending_generations)
        await response.aclose()
    except BaseException as exc:
        assembly.retain(complete=assembly.accumulator.done, error=exc)
        try:
            await response.aclose()
        except BaseException:
            pass
        raise
    body["_stream_receipt"] = assembly.retain(complete=True)
    return AssembledResponse(body)
