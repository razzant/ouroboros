"""OpenAI Realtime audio/tool session, sharing the measured initial layout.

Wire: https://developers.openai.com/api/docs/guides/realtime-conversations
Read 2026-10-21. Automatic truncation stays disabled. Only a playback-correlated
host interruption may truncate newly generated audio; seeded input is immutable.
"""
from __future__ import annotations

import base64
import asyncio
import json
from dataclasses import replace

from ouroboros.contracts.live_conversation import Audio, ContextUpdate, Interrupt, LiveEvent, Respond, ToolOutputs, Video
from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json
from ouroboros.realtime_session import JsonSession
from . import _OUTPUT_FIELD_MAX, _tools, _usage, layout


def items(message):
    """Native tool history/results and images; text uses the measured fragmenter."""
    role = message.get("role")
    if role == "tool":
        if not isinstance(message.get("content"), str) or not message.get("tool_call_id"):
            raise RealtimeSetupError("tool_result", "OpenAI tool output must be text with its call id")
        return [{"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}]
    calls = message.get("tool_calls", [])
    clean = {key: value for key, value in message.items() if key != "tool_calls"}
    content = clean.get("content")
    result = []
    if content or not calls:
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for part in parts:
            if part.get("type") == "image_url" and role == "user":
                url = part["image_url"]
                result.append({"type": "message", "role": role, "content": [
                    {"type": "input_image", "image_url": url["url"] if isinstance(url, dict) else url}]})
            elif part.get("type") == "text" and isinstance(part.get("text"), str):
                text = part["text"]
                for start in range(0, len(text), 256000) or (0,):
                    result.append({"type": "message", "role": role, "content": [{
                        "type": "output_text" if role == "assistant" else "input_text", "text": text[start:start + 256000]}]})
            else:
                raise RealtimeSetupError("content", "OpenAI cannot map this block without loss")
    for call in calls:
        function = call["function"]
        result.append({"type": "function_call", "call_id": call["id"], "name": function["name"],
                       "arguments": function["arguments"]})
    if role not in {"system", "user", "assistant"}:
        raise RealtimeSetupError("role", "unsupported OpenAI history role")
    return result


class OpenAISession(JsonSession):
    def __init__(self, selected, options, *, connect, observe):
        super().__init__(selected, options, connect=connect, observe=observe)
        messages = json.loads(selected.messages_json)
        additions = [options.instructions] if options.instructions else []
        if len(options.languages) > 1:
            additions.append("Speech recognition should also expect: " + ", ".join(options.languages[1:]) + ".")
        if additions:
            position = 0
            while position < len(messages) and messages[position]["role"] == "system":
                position += 1
            messages.insert(position, {"role": "system", "content": "\n\n".join(additions)})
        # The first system block remains the measured setup layout; all following
        # native items retain their order, tool ids and full text.
        first = replace(selected, messages_json=canonical_json(messages[:1]))
        events, _ = layout(first)
        for message in messages[1:]:
            for item in items(message):
                index = len(events)
                item = {**item, "id": f"setup_item_{index}"}
                events.append({"type": "conversation.item.create", "event_id": f"setup_event_{index}",
                    "previous_item_id": events[-1].get("item", {}).get("id", "root"), "item": item})
        session = events[0]["session"]
        session["output_modalities"] = ["audio"]
        allowance = selected.output_allowance_tokens
        session["max_output_tokens"] = allowance if allowance <= _OUTPUT_FIELD_MAX else "inf"
        # "inf" selects the model maximum, not the host's allowance or an
        # unlimited response. Only the host's dated route evidence can name it.
        self.output_bound_tokens = options.model_output_bound_tokens
        if session["max_output_tokens"] != "inf":
            self.output_bound_tokens = (min(allowance, self.output_bound_tokens)
                                        if self.output_bound_tokens is not None else allowance)
        session["audio"] = {"input": {"format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "gpt-4o-transcribe"}, "turn_detection": None},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}}}
        if options.languages:
            session["audio"]["input"]["transcription"]["language"] = options.languages[0].split("-")[0].lower()
        self.turn_detection = {"type": "server_vad", "create_response": False,
                               "interrupt_response": options.interruptions,
                               "silence_duration_ms": {"short": 300, "normal": 500, "long": 1000}[options.pause]}
        if options.thinking:
            session["reasoning"] = {"effort": options.thinking}
        if options.voice:
            session["audio"]["output"]["voice"] = options.voice
        self.events = events
        self.wire = {"model": selected.model, "input_sha256": selected.sha256, "events": events}
        self.pending = []
        self.text = {}
        self.responses = set()
        self.cancelled = set()
        self.pending_cancel = ""
        self.tools = tuple(json.loads(selected.tools_json))
        self.active_response = ""
        self.truncations = {}
        self.number = 0
        self.outstanding = {}
        self.tool_update = None
        self.context_queue = []
        self.context_lock = asyncio.Lock()
        self.response_requested = self.want_response = self.speech_active = False
        self.waiting_tools = set()
        self.input_transcripts, self.output_transcripts = set(), set()
        self.ready = False

    @property
    def renewal_ready(self):
        # The feature separately waits for playback ACKs and its caption writer.
        return self.ready and not (self.speech_active or self.input_transcripts or self.output_transcripts
            or self.active_response or self.response_requested or self.want_response or self.context_queue
            or self.outstanding or self.tool_update is not None or self.waiting_tools or self.truncations)

    def final_captions(self):
        events = [LiveEvent("caption", {"speaker": "ouroboros", "item_id": item, "text": text,
                  "final": True, "replace": True, "incomplete": True}) for item, text in self.text.items() if text]
        self.text.clear()
        return events

    async def open(self):
        await self.connect_wire()
        for outgoing in self.events:
            await self.write(outgoing)
            while True:
                event = await self.read()
                kind = event.get("type")
                if kind == "session.updated" and outgoing["type"] == "session.update":
                    actual, expected = event.get("session", {}), outgoing["session"]
                    for field in ("instructions", "tools", "truncation", "output_modalities", "max_output_tokens"):
                        if actual.get(field) != expected[field]:
                            raise RealtimeSetupError("session_mismatch", f"OpenAI changed {field}")
                    if actual.get("audio", {}).get("input", {}).get("turn_detection") is not None:
                        raise RealtimeSetupError("session_mismatch", "OpenAI started VAD before complete setup")
                    break
                if kind == "conversation.item.done" and outgoing.get("item", {}).get("id") == event.get("item", {}).get("id"):
                    actual, sent = event["item"], outgoing["item"]
                    if any(actual.get(k) != v for k, v in sent.items()):
                        raise RealtimeSetupError("item_mismatch", "OpenAI changed selected input")
                    expected = outgoing["previous_item_id"]
                    if event.get("previous_item_id") not in ({None, "root"} if expected == "root" else {expected}):
                        raise RealtimeSetupError("item_order", "OpenAI changed selected item order")
                    break
                if kind and (kind.startswith("response.") or kind in {"conversation.item.deleted", "conversation.item.truncated"}):
                    raise RealtimeSetupError("setup_mutated", "OpenAI responded or changed input before setup completed")
        await self.write({"type": "session.update", "session": {"type": "realtime", "audio": {"input": {
            "turn_detection": self.turn_detection}}}})
        while True:
            event = await self.read()
            if event.get("type") == "session.updated":
                effective = event.get("session", {})
                if effective.get("truncation") != "disabled":
                    raise RealtimeSetupError("session_mismatch", "OpenAI enabled truncation")
                self.check_output_bound(effective)
                self.check_turn_detection(effective)
                break
            if event.get("type", "").startswith("response.") or event.get("type") in {"conversation.item.deleted", "conversation.item.truncated"}:
                raise RealtimeSetupError("setup_mutated", "OpenAI responded or changed input before setup completed")
        self.ready = True
        await self.advance_context()
        self.pending.append(LiveEvent("ready", {"input_rate": 24000, "output_rate": 24000, "vision": True}))

    async def receive(self):
        if self.pending:
            events, self.pending = self.pending, []
            return events
        events = self.convert(await self.read())
        await self.advance_context()
        return events

    def is_benign_error(self, event):
        error = event.get("error")
        if (event.get("type") != "error" or not isinstance(error, dict) or not self.pending_cancel
                or error.get("code") != "response_cancel_not_active"
                or error.get("event_id") not in (None, "cancel_" + self.pending_cancel)
                or error.get("response_id") not in (None, self.pending_cancel)
                or event.get("response_id") not in (None, self.pending_cancel)):
            return False
        # The probe's error has no client event id and follows response.done.
        # Allow one such reply to our cancel, only until the next response starts.
        self.pending_cancel = ""
        return True

    def check_output_bound(self, effective):
        if effective.get("max_output_tokens") != self.events[0]["session"]["max_output_tokens"]:
            raise RealtimeSetupError("session_mismatch", "OpenAI changed max_output_tokens")

    def check_turn_detection(self, effective):
        actual = effective.get("audio", {}).get("input", {}).get("turn_detection", {})
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in self.turn_detection.items()):
            raise RealtimeSetupError("session_mismatch", "OpenAI changed turn_detection")

    async def advance_context(self):
        """Only this provider owns the ordering of item ACKs and responses."""
        async with self.context_lock:
            if not self.ready or self.active_response or self.response_requested or self.outstanding or self.tool_update is not None:
                return
            while self.context_queue and not self.outstanding and self.tool_update is None:
                command = self.context_queue.pop(0)
                outgoing = [item for message in command.messages for item in items(message)]
                if isinstance(command, ToolOutputs) and command.tools != self.tools:
                    self.tools = command.tools
                    self.tool_update = _tools(list(command.tools))
                    await self.write({"type": "session.update", "session": {"type": "realtime", "tools": self.tool_update}})
                for item in outgoing:
                    self.number += 1
                    item = {**item, "id": f"host_item_{self.number}"}
                    self.outstanding[item["id"]] = item
                    if item["type"] == "function_call_output":
                        self.waiting_tools.discard(item["call_id"])
                    await self.write({"type": "conversation.item.create", "item": item})
                self.want_response |= any(item["type"] != "function_call_output" or
                    not isinstance(command, ToolOutputs) or item["call_id"] not in command.silent for item in outgoing)
            if (self.want_response and not self.context_queue and not self.outstanding
                    and self.tool_update is None and not self.speech_active and not self.waiting_tools):
                self.want_response, self.response_requested = False, True
                await self.write({"type": "response.create"})

    def convert(self, event):
        kind, item = event.get("type", ""), event.get("item_id", "")
        if kind == "session.updated":
            actual = event.get("session", {})
            if actual.get("truncation") != "disabled":
                raise RealtimeSetupError("provider_truncation", "OpenAI changed truncation policy")
            self.check_output_bound(actual)
            self.check_turn_detection(actual)
            if self.tool_update is not None:
                if actual.get("tools") != self.tool_update:
                    raise RealtimeSetupError("tools_mismatch", "OpenAI changed the ordinary tool catalog")
                self.tool_update = None
        if kind == "conversation.item.done":
            actual = event.get("item", {})
            sent = self.outstanding.pop(actual.get("id"), None)
            if sent and any(actual.get(k) != v for k, v in sent.items()):
                raise RealtimeSetupError("item_mismatch", "OpenAI changed host context or tool output")
        if kind == "conversation.item.deleted":
            raise RealtimeSetupError("provider_truncation", "OpenAI deleted conversation input")
        if kind == "conversation.item.truncated":
            if self.truncations.pop(item, None) != event.get("audio_end_ms"):
                raise RealtimeSetupError("provider_truncation", "unrequested OpenAI truncation")
        if kind == "response.created":
            self.pending_cancel = ""
            self.response_requested = False
            self.active_response = event["response"]["id"]
            return [LiveEvent("response_started", {"response_id": self.active_response})]
        if kind == "input_audio_buffer.speech_started":
            self.speech_active = True
            self.input_transcripts.add(item)
            if self.options.interruptions and self.active_response:
                # interrupt_response=true makes VAD cancel before notifying us.
                # Keep the response active until its final usage receipt arrives.
                self.cancelled.add(self.active_response)
            return [LiveEvent("input_started")] + ([LiveEvent("speech_started", {"item_id": item})] if self.options.interruptions else [])
        if kind == "input_audio_buffer.speech_stopped":
            self.speech_active = False
            return [LiveEvent("input_done")]
        if kind == "input_audio_buffer.committed":
            self.input_transcripts.add(item)
            self.want_response = True
        if kind == "response.output_audio.delta":
            self.output_transcripts.add(item)
            if (event.get("response_id") or self.active_response) in self.cancelled:
                return []
            return [LiveEvent("audio", {"item_id": item, "rate": 24000,
                "pcm": base64.b64decode(event["delta"], validate=True)})]
        if kind == "response.output_audio_transcript.delta":
            self.output_transcripts.add(item)
            self.text[item] = self.text.get(item, "") + event["delta"]
            return [LiveEvent("caption", {"speaker": "ouroboros", "item_id": item, "text": event["delta"], "final": False})]
        if kind in {"response.output_audio_transcript.done", "conversation.item.input_audio_transcription.completed"}:
            speaker = "ouroboros" if kind.startswith("response.") else "owner"
            text = event.get("transcript", self.text.pop(item, ""))
            self.text.pop(item, None)
            (self.output_transcripts if speaker == "ouroboros" else self.input_transcripts).discard(item)
            return [LiveEvent("caption", {"speaker": speaker, "item_id": item, "text": text, "final": True})]
        if kind == "conversation.item.input_audio_transcription.failed":
            self.input_transcripts.discard(item)
            return []  # This caption failed; the model still received the audio.
        if kind == "response.done":
            response = event["response"]
            rid = response["id"]
            if rid in self.responses:
                return []
            self.responses.add(rid)
            if self.active_response == rid:
                self.active_response = ""
            self.response_requested = False
            status = response.get("status", "incomplete")
            if status == "cancelled" and (rid in self.cancelled or (response.get("status_details") or {}).get("reason") == "turn_detected"):
                status = "interrupted"
            if response.get("status") == "cancelled":
                self.cancelled.add(rid)
            events = [LiveEvent("response_done", {"usage": _usage(response), "raw_usage": response.get("usage") or {},
                "response_id": rid, "status": status, "continues": False})]
            # A failed response still reports usage before the host stops it.
            if response.get("status") not in {"completed", "cancelled"}:
                events.append(LiveEvent("closed", {"reason": response.get("status"), "resumable": False}))
            for output in response.get("output", []) if response.get("status") == "completed" else ():
                if output.get("type") == "function_call":
                    self.waiting_tools.add(output["call_id"])
                    events.append(LiveEvent("tool_call", {"call_id": output["call_id"], "name": output["name"],
                                                          "arguments": output["arguments"]}))
            return events
        return []

    async def send(self, command):
        if isinstance(command, Audio):
            if command.rate != 24000 or len(command.pcm) % 2:
                raise RealtimeSetupError("audio_format", "OpenAI expects PCM16 mono at 24000 Hz")
            await self.write({"type": "input_audio_buffer.append", "audio": base64.b64encode(command.pcm).decode()})
        elif isinstance(command, Video):
            # Frames are observations. They do not trigger a response by themselves.
            self.number += 1
            item = {"id": f"host_item_{self.number}", "type": "message", "role": "user",
                "content": [{"type": "input_image", "image_url": "data:" + command.mime + ";base64,"
                            + base64.b64encode(command.data).decode()}]}
            self.outstanding[item["id"]] = item
            await self.write({"type": "conversation.item.create", "item": item})
        elif isinstance(command, Interrupt):
            if self.active_response and self.active_response not in self.cancelled:
                self.cancelled.add(self.active_response)
                self.pending_cancel = self.active_response
                await self.write({"type": "response.cancel", "response_id": self.active_response,
                                  "event_id": "cancel_" + self.active_response})
            if command.item_id:
                self.truncations[command.item_id] = command.played_ms
                await self.write({"type": "conversation.item.truncate", "item_id": command.item_id,
                                  "content_index": 0, "audio_end_ms": command.played_ms})
        elif isinstance(command, Respond):
            self.want_response = True
            await self.advance_context()
        elif isinstance(command, (ToolOutputs, ContextUpdate)):
            self.context_queue.append(command)
            await self.advance_context()
        else:
            raise RealtimeSetupError("command", "unsupported OpenAI live command")
