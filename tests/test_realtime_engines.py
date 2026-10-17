"""Live provider wire replay and the credential/transport consumer boundaries."""

import asyncio
import base64
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.gateways import realtime_gemini as gemini
from ouroboros.gateways.realtime_engine import (
    AudioIn, AudioOut, Caption, Close, Closed, ContextItem, EngineError, EngineRoute,
    GoAway, Interrupt, PlayedAudio, Ready, ResponseDone, ResponseStarted, ResumeCheckpoint,
    SessionConfig, ToolCallCancelled,
    ToolResult, ToolsApplied, UpdateTools, USAGE_KEYS,
)
from ouroboros.gateways.realtime_transport import LiveEngineRefusal, transport_error

WIRE = Path(__file__).parent / "fixtures" / "realtime_wire"
TOOL = {"type": "function", "function": {"name": "lookup", "description": "Return the fixture value.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
NEW_TOOL = {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
PCM = b"\x00\x00\x01\x00"


def run_async(test):
    @functools.wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))
    return run


async def eventually(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(poll(), 2)


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.received = 0
        self.closed = False

    async def recv(self):
        value = await self.incoming.get()
        self.received += 1
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)

    async def send(self, value):
        assert not self.closed
        self.sent.append(json.loads(value))

    async def close(self):
        self.closed = True


class Connection:
    def __init__(self):
        self.sockets = []
        self.options = []

    async def __call__(self, url, **kwargs):
        self.options.append((url, kwargs))
        socket = Socket()
        self.sockets.append(socket)
        return socket


async def feed(engine, socket, message):
    count = socket.received + 1
    socket.incoming.put_nowait(message)
    await eventually(lambda: socket.received == count)
    async with engine._command_lock:
        pass
    await asyncio.sleep(0)


async def opened(provider, *, config=None, collect=True):
    connection = Connection()
    engine = gemini.GeminiLiveEngine(api_key="test-key-never-log", connector=connection)
    config = config or SessionConfig(EngineRoute(provider, "unseen-model", voice="test-voice"), "Whole instructions", (TOOL,))
    events = []

    async def consume():
        async for event in engine.events():
            events.append(event)

    collector = asyncio.create_task(consume()) if collect else None
    await engine.open(config)
    ready = {"setupComplete": {}}
    await feed(engine, connection.sockets[0], ready)
    return engine, connection, events, collector


async def finished(engine, collector):
    await engine.close("test_finished")
    if collector:
        await asyncio.wait_for(collector, 2)
    assert not engine._tasks


def selected(events, kind):
    return [event for event in events if isinstance(event, kind)]


@run_async
async def test_documented_gemini_exchange_both_directions():
    fixture = json.loads((WIRE / "gemini" / "documented_exchange.json").read_text())
    assert fixture["header"]["provenance"] == "synthetic_from_docs"
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    for row in fixture["events"][1:]:
        message = row["message"]
        if row["direction"] == "server":
            await feed(engine, socket, message)
        else:
            if "realtimeInput" in message:
                await engine.send(AudioIn(PCM, 16000))
            elif "toolResponse" in message:
                await engine.send(ToolResult("synthetic-call-1", "42"))
            else:
                await engine.send(ContextItem("A typed message.", "room"))
            assert socket.sent[-1] == message
    await finished(engine, collector)
    assert [e.pcm for e in selected(events, AudioOut)] == [PCM, b"\x02\x00\x03\x00"]
    assert len(selected(events, ResponseStarted)) == len(selected(events, ResponseDone)) == 1
    done = selected(events, ResponseDone)[0]
    assert done.item_id.startswith("local:gemini:") and done.response_id == ""
    assert done.usage == dict(zip(USAGE_KEYS, (10, 6, 3, 5, 2, 6)))
    assert done.raw_usage["cacheTokensDetails"][1]["tokenCount"] == 2
    assert selected(events, ToolCallCancelled) == [ToolCallCancelled(("synthetic-call-2",))]
    assert selected(events, GoAway) == [GoAway(12.5)]
    assert [e.text for e in selected(events, Caption) if e.final] == ["Look up fixture.", "I will check."]
    assert not selected(events, EngineError)


@run_async
async def test_a_recorded_gemini_exchange_replays_into_the_events_the_driver_reads():
    """What gemini-3.8-live sent in one probed call on a fresh install (redacted by devtools/live_call_probe.py):
    a greeting, a web search, a link sent to the chat, an interruption and a typed message."""
    from ouroboros.gateways.realtime_engine import ToolCall

    fixture = json.loads((WIRE / "gemini" / "recorded_exchange.json").read_text(encoding="utf-8"))
    assert fixture["header"]["provenance"] == "recorded_live"
    setup = next(row["message"]["setup"] for row in fixture["events"] if "setup" in row["message"])
    declarations = setup["tools"][0]["functionDeclarations"]
    assert declarations and all("parametersJsonSchema" in d and "parameters" not in d for d in declarations)
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    for row in fixture["events"]:
        if row["direction"] == "server" and "setupComplete" not in row["message"]:
            await feed(engine, socket, row["message"])
    await finished(engine, collector)
    assert not selected(events, EngineError)
    done = selected(events, ResponseDone)
    # Gemini counts each prompt's text apart from its audio: the density witness and the price read these.
    assert done and all(d.usage["text_input_tokens"] > 0 for d in done)
    assert any(d.usage["audio_input_tokens"] > 0 for d in done) and any(d.usage["audio_output_tokens"] > 0 for d in done)
    assert [call.name for call in selected(events, ToolCall)] == ["web_search", "send_user_message"]
    finals = [c for c in selected(events, Caption) if c.final]
    assert any(c.speaker == "owner" for c in finals) and any(c.speaker == "ouroboros" for c in finals)
    assert selected(events, AudioOut) and selected(events, ResumeCheckpoint)


@pytest.mark.parametrize("provider,rate", [("google", 16000)])
@run_async
async def test_session_configuration_audio_limits_and_history(provider, rate):
    config = SessionConfig(EngineRoute(provider, "custom-live", voice="chosen-voice"), "Full core", (TOOL,),
                           ({"role": "assistant", "content": "Earlier reply"}, {"role": "tool", "content": "Retained result"}), True, "")
    engine, connection, events, collector = await opened(provider, config=config)
    socket = connection.sockets[0]
    setup = socket.sent[0]
    setup = setup["setup"]
    assert setup["systemInstruction"]["parts"] == [{"text": "Full core"}]
    assert setup["tools"][0]["functionDeclarations"][0]["behavior"] == "NON_BLOCKING"
    # The registry's JSON Schema rides whole as parametersJsonSchema: Gemini's OpenAPI-subset
    # `parameters` field refuses keywords such as additionalProperties (close 1007, live).
    assert setup["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"] == TOOL["function"]["parameters"]
    assert "parameters" not in setup["tools"][0]["functionDeclarations"][0]
    assert setup["tools"][1] == {"googleSearch": {}}
    assert setup["sessionResumption"] == {}
    assert setup["contextWindowCompression"] == {"slidingWindow": {}}
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "chosen-voice"
    assert socket.sent[1]["clientContent"]["turnComplete"] is False
    assert selected(events, Ready)[0].effective["input_rate"] == rate
    await engine.send(AudioIn(PCM, rate))
    before = len(socket.sent)
    for command in (AudioIn(PCM, 48000), AudioIn(b"x", rate), AudioIn(bytes(65538), rate)):
        with pytest.raises(LiveEngineRefusal):
            await engine.send(command)
    await engine.send(PlayedAudio("played-item", 100))
    assert len(socket.sent) == before
    assert connection.options[0][1]["logger"].disabled
    await finished(engine, collector)


@run_async
async def test_gemini_tool_activation_waits_for_result_then_fresh_checkpoint():
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "old", "resumable": True}})
    await feed(engine, socket, {"toolCall": {"functionCalls": [{"id": "get-tools", "name": "lookup", "args": {}}]}})
    await engine.send(UpdateTools((TOOL, NEW_TOOL)))
    assert len(connection.sockets) == 1 and not selected(events, ToolsApplied)
    await engine.send(ToolResult("get-tools", "Activation pending", scheduling="silent"))
    assert socket.sent[-1]["toolResponse"]["functionResponses"][0]["scheduling"] == "SILENT"
    await feed(engine, socket, {"serverContent": {"turnComplete": True}})
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "", "resumable": False}})
    assert len(connection.sockets) == 1
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "fresh", "resumable": True}})
    assert socket.closed and len(connection.sockets) == 2
    resumed = connection.sockets[1]
    assert resumed.sent[0]["setup"]["sessionResumption"] == {"handle": "fresh"}
    assert len(resumed.sent[0]["setup"]["tools"][0]["functionDeclarations"]) == 2
    audio = asyncio.create_task(engine.send(AudioIn(PCM, 16000)))
    await asyncio.sleep(0)
    assert not audio.done() and not selected(events, ToolsApplied)
    await feed(engine, resumed, {"setupComplete": {}})
    await audio
    assert selected(events, ToolsApplied) == [ToolsApplied(("lookup", "read_file"))]
    assert len(selected(events, Ready)) == 1
    await finished(engine, collector)


@run_async
async def test_gemini_context_waits_for_idle_interrupt_is_local_and_cancellation_refuses_results():
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    content = {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AAABAA=="}}]}}}
    await feed(engine, socket, content)
    before = len(socket.sent)
    await engine.send(ContextItem("Text from the room", "owner typed"))
    await engine.send(Interrupt(selected(events, AudioOut)[0].item_id, 10))
    assert len(socket.sent) == before
    await feed(engine, socket, content)
    assert len(selected(events, AudioOut)) == 1
    await feed(engine, socket, {"serverContent": {"interrupted": True}})
    await feed(engine, socket, {"serverContent": {"turnComplete": True}, "usageMetadata": {"responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 7}]}})
    assert selected(events, ResponseDone)[0].status == "interrupted"
    assert selected(events, ResponseDone)[0].usage["audio_output_tokens"] == 7
    assert socket.sent[-1]["clientContent"]["turnComplete"] is False
    await feed(engine, socket, {"toolCall": {"functionCalls": [{"id": "cancel-me", "name": "lookup"}]}})
    await feed(engine, socket, {"toolCallCancellation": {"ids": ["cancel-me"]}})
    with pytest.raises(LiveEngineRefusal, match="cancelled"):
        await engine.send(ToolResult("cancel-me", "late result"))
    await finished(engine, collector)


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_transport_loss_is_retryable_and_close_is_final(provider):
    engine, connection, events, collector = await opened(provider)
    await feed(engine, connection.sockets[0], ConnectionError("secret URL and audio must not be echoed"))
    await asyncio.wait_for(collector, 2)
    assert selected(events, EngineError)[0].kind == "transport"
    assert selected(events, EngineError)[0].retryable
    assert "secret" not in selected(events, EngineError)[0].message
    assert isinstance(events[-1], Closed)
    with pytest.raises(LiveEngineRefusal):
        await engine.send(AudioIn(PCM, 24000))
    await finished(engine, collector)


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_full_event_queue_fails_explicitly_and_stop_never_waits_for_consumer(provider):
    engine, connection, _, _ = await opened(provider, collect=False)
    socket = connection.sockets[0]
    message = {"sessionResumptionUpdate": {"resumable": False}}
    for _ in range(engine._events.maxsize):
        socket.incoming.put_nowait(message)
    await eventually(lambda: engine._closed_reason is not None)
    await asyncio.wait_for(engine.close("stop"), 1)
    events = [event async for event in engine.events()]
    assert events[-2].kind == "backpressure"
    assert isinstance(events[-1], Closed)
    assert socket.closed


@pytest.mark.parametrize("status,retryable,kind", [(401, False, "authentication"), (403, False, "authentication"),
    (400, False, "invalid_request"), (429, True, "rate_limit"), (503, True, "transport")])
def test_handshake_error_classification(status, retryable, kind):
    error = RuntimeError("credential-bearing URL")
    error.response = SimpleNamespace(status_code=status)
    event = transport_error(error)
    assert (event.kind, event.retryable) == (kind, retryable)
    assert "credential" not in event.message


def test_usage_normalization_never_maps_undifferentiated_totals_to_text():
    assert gemini.normalize_usage({"promptTokenCount": 300, "responseTokenCount": 20}) == dict.fromkeys(USAGE_KEYS, 0)
    raw = {"promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}, {"modality": "AUDIO", "tokenCount": 20},
                                   {"modality": "IMAGE", "tokenCount": 3}], "cachedContentTokenCount": 7,
           "responseTokensDetails": [{"modality": "TEXT", "tokenCount": 4}, {"modality": "AUDIO", "tokenCount": 6}]}
    assert gemini.normalize_usage(raw) == dict(zip(USAGE_KEYS, (10, 20, 3, 7, 4, 6)))


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_llm_entry_point_resolves_captured_credentials_and_optional_open(provider, monkeypatch):
    from ouroboros import llm_live
    from ouroboros.gateways import realtime_transport
    from ouroboros.llm import LLMClient

    connection = Connection()
    monkeypatch.setattr(realtime_transport, "connect", connection)
    monkeypatch.setattr(llm_live, "runtime_setting", lambda key, default="": "google-key" if key == "GOOGLE_API_KEY" else default)
    client = LLMClient()
    route = EngineRoute(provider, "custom-model", voice="chosen-voice")
    engine = client.open_live(route)
    assert not connection.sockets
    assert engine.route.input_rate == 16000
    await engine.close("unused")
    engine = await client.open_live(route, SessionConfig(route, "Instructions"))
    assert len(connection.sockets) == 1
    url, options = connection.options[0]
    # The key is a request header, as in Google's own Live clients, never part of the URL.
    assert options["additional_headers"] == {"x-goog-api-key": "google-key"}
    assert "google-key" not in url and "?" not in url
    await engine.close("stop")
    with pytest.raises(LiveEngineRefusal) as other:
        client.open_live(EngineRoute("openai", "gpt-realtime-2.1"))
    assert other.value.code == "engine"  # calls use Google Gemini Live only
    monkeypatch.setattr(llm_live, "runtime_setting", lambda *args: "")
    with pytest.raises(LiveEngineRefusal) as missing:
        client.open_live(route)
    assert missing.value.code == "missing_credential"


@run_async
async def test_gemini_old_checkpoint_never_discards_new_server_or_client_content():
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "old", "resumable": True}})
    await feed(engine, socket, {"serverContent": {"modelTurn": {"parts": [{"text": "New response"}]}}})
    await feed(engine, socket, {"serverContent": {"turnComplete": True}})
    await engine.send(UpdateTools((NEW_TOOL,)))
    assert len(connection.sockets) == 1
    await engine.send(ContextItem("Fresh room context", "room"))
    assert not engine._checkpoint
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "current", "resumable": True}})
    assert connection.sockets[1].sent[0]["setup"]["sessionResumption"] == {"handle": "current"}
    await finished(engine, collector)


@run_async
async def test_gemini_resume_skips_history_and_search_and_voice_are_opt_in():
    config = SessionConfig(EngineRoute("google", "models/custom"), "instructions", history=({"role": "tool", "content": "old effect"},), resume_handle="resume-me")
    engine, connection, events, collector = await opened("google", config=config)
    setup = connection.sockets[0].sent[0]["setup"]
    assert setup["model"] == "models/custom"
    assert setup["sessionResumption"] == {"handle": "resume-me"}
    assert "speechConfig" not in setup["generationConfig"] and not setup["tools"]
    assert len(connection.sockets[0].sent) == 1
    await finished(engine, collector)


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_tool_images_preserve_bytes_and_invalid_images_send_nothing(provider):
    engine, connection, events, collector = await opened(provider)
    socket = connection.sockets[0]
    if provider == "google":
        await feed(engine, socket, {"toolCall": {"functionCalls": [{"id": "image-call", "name": "lookup"}]}})
    image = b"\x89PNG\r\n\x1a\nfixture"
    before = len(socket.sent)
    with pytest.raises(LiveEngineRefusal) as refused:
        await engine.send(ToolResult("image-call", "image", images=(b"invalid",)))
    assert refused.value.code == "image_format" and len(socket.sent) == before
    await engine.send(ToolResult("image-call", "image", images=(image,)))
    if provider == "google":
        encoded = socket.sent[-1]["toolResponse"]["functionResponses"][0]["parts"][0]["inlineData"]["data"]
    else:
        encoded = socket.sent[-1]["item"]["content"][1]["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == image
    await finished(engine, collector)


@run_async
async def test_oversized_result_images_refuse_before_partial_tool_output(monkeypatch):
    from ouroboros.gateways import realtime_transport

    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    await feed(engine, socket, {"toolCall": {"functionCalls": [{"id": "image-call", "name": "lookup", "args": {}}]}})
    before = len(socket.sent)
    monkeypatch.setattr(realtime_transport, "MAX_WIRE_BYTES", 300)
    with pytest.raises(LiveEngineRefusal) as refused:
        await engine.send(ToolResult("image-call", "result", images=(b"\x89PNG\r\n\x1a\n" + bytes(300),)))
    assert refused.value.code == "wire_size" and len(socket.sent) == before
    await finished(engine, collector)


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_rejected_setup_wakes_waiting_sends_and_never_echoes_provider_body(provider):
    connection = Connection()
    engine = gemini.GeminiLiveEngine(api_key="fixture-key", connector=connection)
    await engine.open(SessionConfig(EngineRoute(provider, "model"), "instructions"))
    pending = asyncio.create_task(engine.send(ContextItem("context", "room")))
    error = {"error": {"code": 403, "message": "fixture-key"}}
    await feed(engine, connection.sockets[0], error)
    with pytest.raises(LiveEngineRefusal):
        await asyncio.wait_for(pending, 1)
    events = [event async for event in engine.events()]
    assert events[-2].kind == "authentication" and not events[-2].retryable
    assert "fixture-key" not in repr(events)
    assert isinstance(events[-1], Closed)


@pytest.mark.parametrize("provider", ["google"])
@run_async
async def test_stop_cleans_up_before_ready_or_during_pending_activation(provider):
    connection = Connection()
    engine = gemini.GeminiLiveEngine(api_key="fixture-key", connector=connection)
    await engine.open(SessionConfig(EngineRoute(provider, "model"), "instructions"))
    pending = asyncio.create_task(engine.send(AudioIn(PCM, 16000)))
    await engine.send(Close("stop"))
    with pytest.raises(LiveEngineRefusal):
        await pending
    assert [event async for event in engine.events()] == [Closed("stop")]
    assert connection.sockets[0].closed
    assert not engine._tasks


@run_async
async def test_gemini_cancelled_tools_allow_next_checkpoint_activation_and_usage_resets():
    engine, connection, events, collector = await opened("google")
    socket = connection.sockets[0]
    for index in range(2):
        await feed(engine, socket, {"serverContent": {"modelTurn": {"parts": [{"text": str(index)}]}},
                                    "usageMetadata": {"responseTokensDetails": [{"modality": "TEXT", "tokenCount": index + 1}]}})
        await feed(engine, socket, {"serverContent": {"turnComplete": True}})
    assert [e.usage["text_output_tokens"] for e in selected(events, ResponseDone)] == [1, 2]
    assert len({e.item_id for e in selected(events, ResponseDone)}) == 2
    await feed(engine, socket, {"toolCall": {"functionCalls": [{"id": "pending", "name": "lookup"}]}})
    await engine.send(UpdateTools((NEW_TOOL,)))
    await feed(engine, socket, {"toolCallCancellation": {"ids": ["pending"]}})
    await feed(engine, socket, {"serverContent": {"turnComplete": True}})
    await feed(engine, socket, {"sessionResumptionUpdate": {"newHandle": "after-cancel", "resumable": True}})
    assert len(connection.sockets) == 2
    await finished(engine, collector)


@pytest.mark.parametrize("configured", [False, True])
def test_google_credential_uses_its_own_settings(configured, monkeypatch):
    from ouroboros import provider_models as providers

    values = {"OPENROUTER_API_KEY": "router-test-key", "GOOGLE_API_KEY": "google-test-key" if configured else ""}
    monkeypatch.setattr(providers, "runtime_setting", lambda key, default="": values.get(key, default))
    assert providers.provider_has_credentials("google") is configured
    assert providers.provider_has_credentials_in_settings("google", values) is configured
    assert providers.PROVIDER_CREDENTIAL_GROUPS["google"] == ("GOOGLE_API_KEY",)


def test_google_secret_has_every_core_boundary():
    from ouroboros.colab_bootstrap import _SECRET_KEYS
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
    from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS
    from ouroboros.safety import _PROVIDER_KEY_ENV, _REMOTE_PROVIDER_KEYS
    from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS, looks_masked_settings_secret
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    for keys in (_SECRET_KEYS, FORBIDDEN_SKILL_SETTINGS, MODEL_PROVIDER_CREDENTIAL_KEYS,
                 _REMOTE_PROVIDER_KEYS, MASKED_SECRET_SETTING_KEYS, SETTINGS_DEFAULTS):
        assert "GOOGLE_API_KEY" in keys
    assert _PROVIDER_KEY_ENV["google"] == "GOOGLE_API_KEY"
    assert looks_masked_settings_secret("GOOGLE_API_KEY", "***set***")
    assert not looks_masked_settings_secret("GOOGLE_API_KEY", "actual-test-value")


def test_colab_collects_google_without_treating_live_as_text_access(monkeypatch):
    from ouroboros import colab_bootstrap as colab

    requested = []

    def secret(name, *, required=True):
        requested.append((name, required))
        return "google-test-key" if name == "GOOGLE_API_KEY" else ""

    monkeypatch.setattr(colab, "get_colab_secret", secret)
    assert colab.collect_colab_secrets()["GOOGLE_API_KEY"] == "google-test-key"
    assert ("GOOGLE_API_KEY", False) in requested
    assert ("OPENROUTER_API_KEY", True) in requested


def test_a_named_server_close_keeps_its_sentence_and_is_not_retried():
    from types import SimpleNamespace

    from ouroboros.gateways.realtime_transport import transport_error

    quota = SimpleNamespace(rcvd=SimpleNamespace(code=1011, reason=(
        "You exceeded your current quota, please check your plan https://ai.google.dev/x key AIzaSyFAKEFAKEFAKE12")))
    error = transport_error(quota)
    assert error.kind == "provider_closed" and not error.retryable
    assert error.message.startswith("You exceeded your current quota") and "AIza" not in error.message
    assert error.provider_words == error.message  # Google's own sentence, which the owner may read as Google's
    assert "https://" not in error.message
    failed = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1011, reason="")))
    assert failed.kind == "provider_closed" and not failed.retryable  # a server error without words: Try again
    assert failed.provider_words == ""  # the host's description is never quoted as Google's
    lost = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1006, reason="")))
    assert lost.kind == "transport" and lost.retryable  # a connection lost without a close frame resumes
    normal = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1000, reason="bye")))
    assert normal.kind == "transport" and normal.retryable
    too_big = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1007, reason=(
        "user system instruction has 140000 tokens, the sum is above the 131072 limit"))))
    assert too_big.kind == "invalid_request" and not too_big.retryable and "140000 tokens" in too_big.message
    bare = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1007, reason="")))
    assert bare.kind == "protocol"
    denied = transport_error(SimpleNamespace(rcvd=SimpleNamespace(code=1008, reason="key fixture-key is invalid")))
    assert denied.kind == "authentication" and "fixture-key" not in denied.message and denied.provider_words == ""


def test_gemini_compression_starts_at_the_drivers_watermarks():
    base = SessionConfig(EngineRoute("google", "gemini-3.8-live"), "core")
    assert gemini.setup_message(base)["setup"]["contextWindowCompression"] == {"slidingWindow": {}}
    marked = SessionConfig(EngineRoute("google", "gemini-3.8-live"), "core",
                           compress_at_tokens=122_880, compress_to_tokens=106_000)
    assert gemini.setup_message(marked)["setup"]["contextWindowCompression"] == {
        "triggerTokens": 122_880, "slidingWindow": {"targetTokens": 106_000}}
