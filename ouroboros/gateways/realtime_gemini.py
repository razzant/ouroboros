"""Gemini Live I/O with checkpoint-bound tool activation and turn accounting.

The wire is documented at https://ai.google.dev/api/live and
https://ai.google.dev/gemini-api/docs/live-api/tools. One fixture is synthetic from
those documents; the other is gemini-3.8-live's own wire from a probed call
(``devtools/live_call_probe.py``, redacted), replayed in the tests. Gemini has no
response/audio item IDs: local playback IDs are labeled, while function IDs and
resume handles remain provider-owned. Context waits for idle because clientContent
interrupts generation even when turnComplete is false.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
from uuid import uuid4

from ouroboros.gateways.realtime_engine import (
    AudioIn, AudioOut, Caption, ContextItem, EngineError, GoAway, Interrupt, Ready,
    ResponseDone, ResponseStarted, ResumeCheckpoint, SpeechStarted, ToolCall,
    ToolCallCancelled, ToolResult, ToolsApplied, UpdateTools, USAGE_KEYS, register_engine,
)
from ouroboros.gateways.realtime_engine import GOOGLE_INPUT_RATE as INPUT_RATE, GOOGLE_OUTPUT_RATE as OUTPUT_RATE
from ouroboros.gateways.realtime_transport import (
    EVENT_QUEUE_SIZE, LiveEngineRefusal, RealtimeTransport, check_audio,
    function_schemas, image_part, provider_sentence, transport_error,
)


def normalize_usage(raw):
    """Keep modality totals separate; cached counts are a subset of input totals."""
    usage = dict.fromkeys(USAGE_KEYS, 0)
    for field, direction in (("promptTokensDetails", "input"), ("responseTokensDetails", "output")):
        for detail in raw.get(field) or ():
            key = f"{detail['modality'].lower()}_{direction}_tokens"
            if key in usage:
                usage[key] += detail.get("tokenCount", 0)
    usage["cached_input_tokens"] = raw.get("cachedContentTokenCount", sum(
        detail.get("tokenCount", 0) for detail in raw.get("cacheTokensDetails") or ()))
    return usage


def compression(config):
    """Sliding-window compression starting at the driver's watermarks (provider defaults without them).

    Gemini keeps the system instruction and slides out the oldest spoken turns; the
    defaults (80% of the window, keep half of that) would sit below a large call view.
    """
    if config.compress_at_tokens <= 0:
        return {"slidingWindow": {}}
    window = {"targetTokens": config.compress_to_tokens} if config.compress_to_tokens > 0 else {}
    return {"triggerTokens": config.compress_at_tokens, "slidingWindow": window}


def declaration(tool):
    """A registry function as Gemini's declaration: the whole JSON Schema rides as ``parametersJsonSchema``.

    Gemini's older ``parameters`` field takes only an OpenAPI subset and refuses the session
    over keywords such as ``additionalProperties`` or a list ``type`` (close 1007, observed live
    on gemini-3.8-live 2026-10-17); ``parametersJsonSchema`` takes the schema as written.
    """
    shaped = {key: tool[key] for key in ("name", "description") if key in tool}
    if "parameters" in tool:
        shaped["parametersJsonSchema"] = tool["parameters"]
    return {**shaped, "behavior": "NON_BLOCKING"}


def setup_message(config):
    declarations = [declaration(tool) for tool in function_schemas(config.tools)]
    tools = [{"functionDeclarations": declarations}] if declarations else []
    if config.native_search:
        tools.append({"googleSearch": {}})
    generation = {"responseModalities": ["AUDIO"]}
    if config.route.voice:
        generation["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.route.voice}}}
    return {"setup": {
        "model": config.route.model if config.route.model.startswith("models/") else f"models/{config.route.model}",
        "systemInstruction": {"parts": [{"text": config.instructions}]},
        "tools": tools, "generationConfig": generation,
        "inputAudioTranscription": {}, "outputAudioTranscription": {},
        "contextWindowCompression": compression(config),
        "sessionResumption": {"handle": config.resume_handle} if config.resume_handle else {},
    }}


def history_turn(row):
    """Host history is context; earlier tool calls are never replayed as effects."""
    content = row.get("content", "")
    role = row.get("role", "user")
    if role not in ("user", "assistant", "model"):
        content = f"[{role}] {content}" if isinstance(content, str) else json.dumps(row)
    if isinstance(content, str):
        parts = [{"text": content}]
    else:
        parts = []
        for part in content:
            if part.get("type") in ("text", "input_text", "output_text"):
                parts.append({"text": part["text"]})
            elif part.get("type") == "image_url":
                url = part["image_url"]
                url = url["url"] if isinstance(url, dict) else url
                if not url.startswith("data:") or ";base64," not in url:
                    raise LiveEngineRefusal("history_image", "Gemini live history needs inline base64 images.")
                mime, data = url[5:].split(";base64,", 1)
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            else:
                raise LiveEngineRefusal("history_content", "Gemini live history requires text or inline images.")
    if row.get("tool_calls"):
        # As the replay bound weighs them (``live_session_view``): words unescaped, Cyrillic as itself.
        parts.append({"text": "[Earlier tool calls] " + json.dumps(row["tool_calls"], ensure_ascii=False)})
    return {"role": "model" if role in ("assistant", "model") else "user", "parts": parts}


class GeminiLiveEngine(RealtimeTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._local_id = "local:gemini:" + uuid4().hex
        self._turn_number = 0
        self._owner_number = 0
        self._item_id = ""
        self._response_id = ""
        self._usage = {}
        self._interrupted = False
        self._discard_audio = False
        self._transcripts = {"owner": "", "ouroboros": ""}
        self._transcript_ids = {}
        self._calls = {}
        self._context = []
        self._pending_tools = None
        self._activating_tools = None
        self._checkpoint = ""
        self._initial_setup = True

    async def open(self, config):
        if config.route.engine != "google":
            raise LiveEngineRefusal("route", "Gemini requires a google live route.")
        setup_message(config)  # Refuse malformed tools before a socket is opened.
        if not config.resume_handle:
            for row in config.history:
                history_turn(row)
        self._begin(config)
        try:
            await self._connect_session()
            self._spawn(self._read())
        except asyncio.CancelledError:
            await self.close("cancelled")
            raise
        except LiveEngineRefusal:
            await self.close("configuration")
            raise
        except Exception as exc:
            await self._fail(transport_error(exc))

    async def _connect_session(self):
        # The key travels as the ``x-goog-api-key`` request header, where Google's own
        # Live clients put it (go-genai ``live.go`` and the js-genai change that moved it
        # out of the query string, read 2026-10-16), so it stays out of URLs that
        # connection logs and proxies record. The WebSocket get-started page still
        # shows the ``?key=`` spelling; both are the same server's accepted forms.
        url = ("wss://generativelanguage.googleapis.com/ws/"
               "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
        await self._connect(url, headers={"x-goog-api-key": self._api_key})
        await self._write(setup_message(self._config))

    async def _write(self, payload):
        # A token predating an outbound message is no longer a safe tool-update
        # checkpoint. A later server checkpoint, never a timer, permits reopening.
        self._checkpoint = ""
        await super()._write(payload)

    async def _reopen_tools(self):
        if self._pending_tools is None or not self._checkpoint or self._item_id or self._calls:
            return
        self._ready.clear()
        self._activating_tools = self._pending_tools
        self._pending_tools = None
        self._config = replace(self._config, tools=self._activating_tools,
                               resume_handle=self._checkpoint, history=())
        self._checkpoint = ""
        await self._ws.close()
        await self._connect_session()

    async def _send_command(self, command):
        if isinstance(command, AudioIn):
            check_audio(command, INPUT_RATE)
            await self._write({"realtimeInput": {"audio": {
                "mimeType": f"audio/pcm;rate={INPUT_RATE}", "data": base64.b64encode(command.pcm).decode("ascii")}}})
        elif isinstance(command, ContextItem):
            if self._item_id:
                if len(self._context) >= EVENT_QUEUE_SIZE:
                    raise LiveEngineRefusal("context_backpressure", "Live context queue is full; retain this item for redelivery.")
                self._context.append(command)
            else:
                await self._send_context(command)
        elif isinstance(command, ToolResult):
            schedules = {"when_idle": "WHEN_IDLE", "silent": "SILENT"}
            if command.scheduling not in schedules:
                raise LiveEngineRefusal("scheduling", "Live tool scheduling requires when_idle or silent.")
            if command.call_id not in self._calls:
                raise LiveEngineRefusal("tool_call", "The Gemini function call is unknown or cancelled.")
            images = [image_part(data) for data in command.images]
            # The FunctionResponse wire schema places scheduling alongside
            # response. A key inside the arbitrary result object is model data.
            response = {"id": command.call_id, "name": self._calls[command.call_id],
                        "response": {"result": command.output}, "scheduling": schedules[command.scheduling]}
            if images:
                response["parts"] = [{"inlineData": {"mimeType": mime, "data": data}} for mime, data in images]
            await self._write({"toolResponse": {"functionResponses": [response]}})
            del self._calls[command.call_id]
        elif isinstance(command, Interrupt):
            self._discard_audio = bool(self._item_id)  # Gemini handles truncation server-side.
        elif isinstance(command, UpdateTools):
            tuple(function_schemas(command.tools))
            self._pending_tools = command.tools
            # The caller answers its activation tool with a pending ToolResult
            # first. Outstanding calls prevent a resume that would lose that result.
            await self._reopen_tools()
        else:
            raise LiveEngineRefusal("command", "Unsupported live engine command.")

    async def _send_context(self, command):
        await self._write({"clientContent": {"turns": [{"role": "user", "parts": [
            {"text": f"[{command.source}] {command.text}"}]}], "turnComplete": False}})

    def _start_response(self, message):
        if self._item_id:
            return
        self._turn_number += 1
        content = message.get("serverContent") or {}
        model_turn = content.get("modelTurn") or {}
        self._item_id = model_turn.get("id") or f"{self._local_id}:turn:{self._turn_number}"
        self._response_id = message.get("responseId", "")
        self._emit(ResponseStarted(self._item_id, self._response_id))

    def _transcription(self, speaker, transcript):
        if not self._transcripts[speaker]:
            if speaker == "owner":
                self._owner_number += 1
            self._transcript_ids[speaker] = transcript.get("id") or (
                f"{self._local_id}:owner:{self._owner_number}" if speaker == "owner" else self._item_id)
        text = transcript.get("text", "")
        self._transcripts[speaker] += text
        if text:
            self._emit(Caption(speaker, text, False, self._transcript_ids[speaker]))
        if transcript.get("finished"):
            self._finish_transcript(speaker)

    def _finish_transcript(self, speaker):
        text = self._transcripts[speaker]
        if text:
            self._emit(Caption(speaker, text, True, self._transcript_ids[speaker]))
            self._transcripts[speaker] = ""

    async def _handle_message(self, message):
        if "setupComplete" in message:
            if self._initial_setup and not self._config.resume_handle and self._config.history:
                await self._write({"clientContent": {"turns": [history_turn(row) for row in self._config.history],
                                                     "turnComplete": False}})
            self._ready.set()
            if self._initial_setup:
                self._initial_setup = False
                self._emit(Ready({"model": self._config.route.model, "input_rate": INPUT_RATE, "output_rate": OUTPUT_RATE,
                                  "native_search": self._config.native_search, **message["setupComplete"]}))
            elif self._activating_tools is not None:
                self._emit(ToolsApplied(tuple(tool["name"] for tool in function_schemas(self._activating_tools))))
                self._activating_tools = None
        if "usageMetadata" in message:
            # These are cumulative snapshots of the same response, not deltas.
            self._usage.update(message["usageMetadata"])
        content = message.get("serverContent") or {}
        calls = (message.get("toolCall") or {}).get("functionCalls") or []
        if content or calls or "toolCallCancellation" in message or "usageMetadata" in message:
            self._checkpoint = ""  # New server content makes an earlier token stale too.
        if "groundingMetadata" in content:
            self._usage["groundingMetadata"] = content["groundingMetadata"]
        if content.get("modelTurn") or content.get("outputTranscription") or calls:
            self._start_response(message)
        for part in (content.get("modelTurn") or {}).get("parts", ()):
            if "inlineData" in part:
                audio = part["inlineData"]
                if audio.get("mimeType", "").split(";")[0] != "audio/pcm":
                    raise ValueError("unsupported output audio format")
                if not self._discard_audio:
                    self._emit(AudioOut(base64.b64decode(audio["data"], validate=True), OUTPUT_RATE, self._item_id))
            elif "text" in part and not part.get("thought"):
                self._transcription("ouroboros", {"text": part["text"]})
        for key, speaker in (("inputTranscription", "owner"), ("outputTranscription", "ouroboros")):
            if key in content:
                self._transcription(speaker, content[key])
        for call in calls:
            if call["id"] not in self._calls:
                self._calls[call["id"]] = call["name"]
                self._emit(ToolCall(call["id"], call["name"], json.dumps(call.get("args", {}), ensure_ascii=False)))
        if "toolCallCancellation" in message:
            ids = tuple(message["toolCallCancellation"].get("ids", ()))
            for call_id in ids:
                self._calls.pop(call_id, None)
            self._emit(ToolCallCancelled(ids))
        if content.get("interrupted"):
            self._interrupted = True
            self._discard_audio = True
            self._emit(SpeechStarted())
        if content.get("turnComplete"):
            for speaker in self._transcripts:
                self._finish_transcript(speaker)
            if self._item_id:
                self._emit(ResponseDone(self._item_id, "interrupted" if self._interrupted else "completed",
                                        normalize_usage(self._usage), dict(self._usage), self._response_id))
                self._item_id = ""
                self._response_id = ""
                self._usage = {}
            self._interrupted = self._discard_audio = False
            for command in self._context:
                await self._send_context(command)
            self._context.clear()
        if "sessionResumptionUpdate" in message:
            checkpoint = message["sessionResumptionUpdate"]
            resumable = checkpoint.get("resumable", False)
            self._checkpoint = checkpoint.get("newHandle", "") if resumable else ""
            self._emit(ResumeCheckpoint(checkpoint.get("newHandle", ""), resumable))
            await self._reopen_tools()
        if "goAway" in message:
            duration = message["goAway"].get("timeLeft", "0s")
            seconds = float(duration.removesuffix("s")) if isinstance(duration, str) else (
                float(duration.get("seconds", 0)) + float(duration.get("nanos", 0)) / 1e9)
            self._emit(GoAway(seconds))
        if "error" in message:
            code = message["error"].get("code")
            kind = "authentication" if code in (401, 403) else "provider_error"
            # An access refusal keeps the host's words: its provider text can echo credentials.
            words = "" if kind == "authentication" else provider_sentence(message["error"].get("message"))
            await self._fail(EngineError(kind, words or "Gemini rejected a live event.", code in (408, 429, 500, 503),
                                         provider_words=words))


register_engine("google", GeminiLiveEngine)
