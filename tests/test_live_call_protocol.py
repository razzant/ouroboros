"""The live-call vocabulary: frames, control messages, the call specification, engines."""

from __future__ import annotations

import pytest

from ouroboros import call_protocol as cp
from ouroboros.gateways import realtime_engine as re_


def test_audio_frame_round_trips_and_refuses_foreign_bytes():
    frame = cp.AudioFrame(cp.FRAME_AUDIO_IN, epoch=3, seq=41, rate=16000, stream=0, pcm=b"\x01\x00" * 160)
    raw = cp.encode_frame(frame)
    assert len(raw) == cp.HEADER_SIZE + 320
    assert cp.decode_frame(raw) == frame
    with pytest.raises(ValueError):
        cp.decode_frame(b"XX" + raw[2:])
    with pytest.raises(ValueError):
        cp.encode_frame(cp.AudioFrame(cp.FRAME_AUDIO_OUT, 0, 0, 24000, 0, b"\x00"))  # odd PCM16 length


def test_control_messages_accept_only_declared_types():
    text = cp.encode_control({"type": "played", "item_id": "it_1", "played_ms": 1200})
    assert cp.decode_control(text) == {"type": "played", "item_id": "it_1", "played_ms": 1200}
    with pytest.raises(ValueError):
        cp.encode_control({"type": "speaker_guess", "name": "x"})
    with pytest.raises(ValueError):
        cp.decode_control('{"type": "exec", "cmd": "rm"}')


def test_call_spec_survives_task_metadata():
    spec = cp.CallSpec(call_id="c1", chat_id=1, engine=re_.ENGINE_GOOGLE, model="gemini-3.8-live",
                       voice="marin", max_minutes=None)
    assert cp.CallSpec.from_metadata(spec.as_metadata()) == spec
    presence = cp.CallSpec(call_id="c2", chat_id=1099511627776, engine=re_.ENGINE_GOOGLE,
                           model="gemini-3.8-live", opened_by=cp.OPENED_BY_SKILL,
                           presence_binding_id="b" * 32, max_minutes=90)
    assert cp.CallSpec.from_metadata(presence.as_metadata()) == presence


def test_engine_registry_refuses_unknown_and_returns_fresh_adapters():
    made = []

    class _Fake:
        async def open(self, config): ...
        async def send(self, command): ...
        def events(self): return iter(())
        async def close(self, reason): ...

    def factory():
        made.append(_Fake())
        return made[-1]

    re_.register_engine("fake-test-engine", factory)
    try:
        first = re_.create_engine(re_.EngineRoute("fake-test-engine", "m"))
        second = re_.create_engine(re_.EngineRoute("fake-test-engine", "m"))
        assert first is not second and len(made) == 2
    finally:
        re_._FACTORIES.pop("fake-test-engine", None)
    with pytest.raises(KeyError):
        re_.create_engine(re_.EngineRoute("nonexistent", "m"))
