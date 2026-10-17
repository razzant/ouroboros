"""Bounded websocket lifetime of the Gemini Live projection.

The LLM layer supplies credentials; this leaf owns I/O and cleanup only. A full
consumer queue closes explicitly instead of losing speech or blocking Stop behind
an unread event. Provider bodies and websocket debug frames never reach logs.
"""
from __future__ import annotations

import asyncio
import json
import logging

from websockets.asyncio.client import connect

from ouroboros.deadline_utils import llm_transport_timeout_sec
from ouroboros.gateways.realtime_engine import Close, Closed, EngineError, PlayedAudio
from ouroboros.net_transport import trust_ssl_context

# Memory/transport bounds, independent of dialogue policy or model capabilities.
EVENT_QUEUE_SIZE = 128
MAX_WIRE_BYTES = 4 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024


class LiveEngineRefusal(ValueError):
    """A command cannot be represented by this transport; no substitute is sent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def provider_sentence(text: object) -> str:
    """A provider's own close or error sentence, bounded, with anything key- or URL-shaped removed."""
    import re

    words = " ".join(str(text or "").split())
    words = re.sub(r"https?://\S+", "[link]", words)
    words = re.sub(r"\b(?:AIza|sk-)[A-Za-z0-9_\-]{8,}", "[key]", words)
    return words[:300]


def transport_error(exc: Exception) -> EngineError:
    """Classify status and close codes without echoing URLs, keys or audio.

    A server close that names its reason (Gemini closes a session over quota with
    1011 and the sentence "You exceeded your current quota…") keeps that sentence and
    is not retried: reopening cannot change it, and the owner reads the provider's own
    words. A server error close (1011) without a reason ends the same way, with the
    owner's Try again: automatic reopening would repeat a paid session the server just
    failed. A connection lost without a close frame (1006) or closed normally resumes.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    received = getattr(exc, "rcvd", None)
    code = getattr(received, "code", None)
    sentence = provider_sentence(getattr(received, "reason", ""))
    if status in (401, 403) or code == 1008:
        return EngineError("authentication", "Live provider refused access.")  # its text can echo credentials
    if status == 429:
        return EngineError("rate_limit", "Live provider rate limit reached.", True)
    if status is not None and 400 <= status < 500 and status != 408:
        return EngineError("invalid_request", "Live provider rejected the session configuration.")
    if code == 1007 and sentence:
        # Gemini refuses a setup it cannot hold (e.g. an instruction above the window) this way,
        # in its own words; reopening the same session cannot change them.
        return EngineError("invalid_request", sentence, provider_words=sentence)
    if code in (1002, 1003, 1007, 1009):
        return EngineError("protocol", "Live provider rejected the websocket payload.")
    if code is not None and code not in (1000, 1001, 1006) and sentence:
        return EngineError("provider_closed", sentence, provider_words=sentence)
    if code == 1011:
        return EngineError("provider_closed", "The live provider ended the session with an internal error.")
    return EngineError("transport", "Live provider connection lost; delivery may be incomplete.", True)


def check_audio(command, rate: int) -> None:
    if command.rate != rate:
        raise LiveEngineRefusal("audio_rate", f"This live route requires PCM16 mono at {rate} Hz.")
    if len(command.pcm) % 2 or len(command.pcm) > MAX_AUDIO_BYTES:
        raise LiveEngineRefusal("audio_size", "PCM16 audio must contain whole samples and fit 64 KiB.")


def function_schemas(tools):
    """Unwrap canonical functions without changing their argument schemas."""
    for tool in tools:
        if tool.get("type", "function") != "function":
            raise LiveEngineRefusal("tool_schema", "Live tools require function schemas.")
        function = dict(tool.get("function", tool))
        if not function.get("name"):
            raise LiveEngineRefusal("tool_schema", "Live function schemas require a name.")
        yield {key: function[key] for key in ("name", "description", "parameters") if key in function}


def image_part(data: bytes) -> tuple[str, str]:
    """Keep actual image bytes and MIME; unknown formats refuse before dispatch."""
    import base64
    from ouroboros.gateways.image_api import sniff_image_mime

    mime = sniff_image_mime(data)
    if not mime:
        raise LiveEngineRefusal("image_format", "Live tool images require PNG, JPEG, WebP or GIF bytes.")
    return mime, base64.b64encode(data).decode("ascii")


def wire_text(payload) -> str:
    """Check a whole command before dispatch, including multi-message results."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_WIRE_BYTES:
        raise LiveEngineRefusal("wire_size", "Live command exceeds the 4 MiB transport limit.")
    return encoded


class RealtimeTransport:
    """One event consumer, one reader, serialized commands, interruptible closure."""

    def __init__(self, *, api_key: str = "", connector=None):
        self._api_key = api_key
        self._connector = connector or connect
        self._ws = None
        self._events = asyncio.Queue(maxsize=EVENT_QUEUE_SIZE)
        self._available = asyncio.Event()
        self._ready = asyncio.Event()
        self._command_lock = asyncio.Lock()
        self._tasks = set()
        self._closed_reason = None
        self._close_finished = asyncio.Event()
        self._failure = None
        self._opened = False
        self._logger = logging.Logger("ouroboros.live_wire")
        self._logger.disabled = True

    def _begin(self, config):
        if self._opened:
            raise LiveEngineRefusal("already_open", "Create a fresh engine for another session.")
        if not self._api_key:
            raise LiveEngineRefusal("missing_credential", "Configure the selected live provider's API key.")
        self._opened = True
        self._config = config

    async def _connect(self, url, headers=None):
        options = dict(max_size=MAX_WIRE_BYTES, max_queue=16, write_limit=64 * 1024,
                       open_timeout=llm_transport_timeout_sec(), close_timeout=5,
                       logger=self._logger)
        if headers:
            options["additional_headers"] = headers
        context = trust_ssl_context()
        if context is not None:
            options["ssl"] = context
        ws = await self._connector(url, **options)
        if self._closed_reason is not None:
            await ws.close()
            return
        self._ws = ws

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _emit(self, event):
        self._events.put_nowait(event)
        self._available.set()

    async def _write(self, payload):
        if self._closed_reason is not None:
            raise LiveEngineRefusal("closed", "The live session is closed.")
        await self._ws.send(wire_text(payload))

    async def _read(self):
        try:
            while self._closed_reason is None:
                raw = await self._ws.recv()
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected an event object")
                async with self._command_lock:
                    await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except asyncio.QueueFull:
            await self._fail(EngineError("backpressure", "Live event queue is full; delivery is incomplete."))
        except (ValueError, TypeError, KeyError):
            await self._fail(EngineError("protocol", "Live provider sent an invalid event."))
        except Exception as exc:
            await self._fail(transport_error(exc))

    async def send(self, command):
        if isinstance(command, Close):
            await self.close(command.reason)
            return
        if not self._opened:
            raise LiveEngineRefusal("not_open", "Open the live session before sending commands.")
        try:
            while True:
                await self._ready.wait()
                async with self._command_lock:
                    if self._closed_reason is not None:
                        raise LiveEngineRefusal("closed", "The live session is closed.")
                    if not self._ready.is_set():
                        continue  # Tool activation can reopen while this command waits.
                    if not isinstance(command, PlayedAudio):
                        await self._send_command(command)
                    return
        except LiveEngineRefusal:
            raise
        except asyncio.CancelledError:
            await self.close("cancelled")
            raise
        except Exception as exc:
            await self._fail(transport_error(exc))

    async def events(self):
        while True:
            self._available.clear()
            while not self._events.empty():
                yield self._events.get_nowait()
            if self._closed_reason is not None:
                await self._close_finished.wait()
                if self._failure is not None:
                    yield self._failure
                yield Closed(self._closed_reason)
                return
            await self._available.wait()

    async def _fail(self, error):
        if self._closed_reason is None:
            self._failure = error
            await self.close(error.kind)

    async def close(self, reason):
        if self._closed_reason is not None:
            return
        self._closed_reason = reason
        self._ready.set()
        self._available.set()
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass  # Closure remains explicit even when the peer has already gone.
        self._close_finished.set()
        self._available.set()
