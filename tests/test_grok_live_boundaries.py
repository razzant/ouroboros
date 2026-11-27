"""Observed xAI representations through the loaded adapter, synthetic wires only."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
from pathlib import Path

import pytest

from tests.test_grok_openai_protocol import providers as providers, deny_network as deny_network, Wire
from ouroboros.contracts.live_conversation import ContextUpdate, Interrupt, LiveOptions
from ouroboros.contracts.realtime import RealtimeInput, RealtimeSetupError, canonical_json


@pytest.fixture
def captured(request):
    records = []
    yield records
    root = Path(os.environ["OUROBOROS_TEST_EVIDENCE_ROOT"]) / "grok-boundaries"
    root.mkdir(parents=True, exist_ok=True)
    (root / (request.node.name + ".json")).write_text(json.dumps(records, indent=2, default=str))


def make(providers, wire, captured, messages=None):
    selected = RealtimeInput("synthetic-grok", canonical_json(messages or [
        {"role": "system", "content": "Synthetic ordinary rules."},
        {"role": "user", "content": "Synthetic owner words."},
    ]), "[]", 256, "nano", "core", "view")
    return providers["grok"].session(
        selected, LiveOptions(), connect=wire.connect,
        observe=lambda direction, raw: captured.append({"direction": direction, "raw": raw}))


class AddedWire(Wire):
    fault = ""

    def __init__(self):
        super().__init__("grok")

    def put(self, value):
        value = copy.deepcopy(value)
        if value.get("type") in {"conversation.item.done", "conversation.item.added"}:
            value["type"] = "conversation.item.added"
            item = value["item"]
            if self.fault != "absent":
                item["status"] = (None if self.fault == "null" else
                                  "in_progress" if self.fault == "in_progress" else "completed")
            if self.fault == "id":
                item["id"] = "wrong-id"
            elif self.fault == "role":
                item["role"] = "assistant"
            elif self.fault == "text":
                item["content"][0]["text"] = "Changed words"
        super().put(value)


@pytest.mark.parametrize("phase", ["setup", "live"])
@pytest.mark.parametrize("fault", ["", "absent", "in_progress", "null", "id", "role", "text"])
def test_added_ack_requires_exact_item_and_no_incomplete_status(providers, captured, phase, fault):
    async def run():
        wire = AddedWire()
        adapter = make(providers, wire, captured)
        opening = None
        blocked = fault not in {"", "absent"}
        try:
            if phase == "setup":
                wire.fault = fault
                opening = asyncio.create_task(adapter.open())
                if fault in {"in_progress", "null", "id"}:
                    done, _ = await asyncio.wait((opening,), timeout=.05)
                    assert not done
                    assert not any(row.get("session", {}).get("turn_detection") for row in wire.sent)
                elif blocked:
                    with pytest.raises(RealtimeSetupError, match="item_mismatch"):
                        await asyncio.wait_for(opening, 1)
                else:
                    await asyncio.wait_for(opening, 1)
                    assert (await adapter.receive())[0].kind == "ready"
            else:
                await adapter.open()
                await adapter.receive()
                wire.fault = fault
                await adapter.send(ContextUpdate(({"role": "user", "content": "Exact new words"},)))
                if fault in {"role", "text"}:
                    with pytest.raises(RealtimeSetupError, match="item_mismatch"):
                        await adapter.receive()
                else:
                    await adapter.receive()
                    assert bool(adapter.outstanding) is blocked
                    assert any(row.get("type") == "response.create" for row in wire.sent) is (not blocked)
            if blocked:
                assert not any(row.get("type") == "response.create" for row in wire.sent)
        finally:
            if opening is not None and not opening.done():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("fault,code", [
    ("response.created", "setup_response"),
    ("conversation.item.deleted", "provider_truncation"),
    ("conversation.item.truncated", "provider_truncation"),
    ("disabled", "session_mismatch"),
])
def test_vad_activation_keeps_admission_guards(providers, captured, fault, code):
    class VadWire(AddedWire):
        def put(self, value):
            if value.get("type") == "session.updated" and value.get("session", {}).get("turn_detection"):
                value = ({"type": "session.updated", "session": {"turn_detection": None}}
                         if fault == "disabled" else {"type": fault})
            super().put(value)

    async def run():
        adapter = make(providers, VadWire(), captured)
        try:
            with pytest.raises(RealtimeSetupError, match=code):
                await asyncio.wait_for(adapter.open(), 1)
        finally:
            await adapter.close()
    asyncio.run(run())


def test_cancel_before_audio_drains_usage_and_allows_next_response(providers, captured):
    async def run():
        wire = AddedWire()
        adapter = make(providers, wire, captured)
        try:
            await adapter.open()
            adapter.convert({"type": "response.created", "response": {"id": "cancelled"}})
            await adapter.send(Interrupt())
            audio = {"type": "response.audio.delta", "response_id": "cancelled", "item_id": "late",
                     "delta": base64.b64encode(b"\x00\x00" * 20).decode()}
            assert adapter.convert(audio) == []
            usage = adapter.convert({"type": "response.done", "usage": {"audio_seconds": 4},
                                     "response": {"id": "cancelled", "status": "cancelled"}})
            assert usage[0].data["response_id"] == "cancelled"
            # Interrupt is a barge-in even before any PCM arrives. The native
            # cancelled receipt still drains, normalized to interrupted.
            assert usage[0].data["status"] == "interrupted"
            assert usage[0].data["raw_usage"] == {"audio_seconds": 4}
            adapter.convert({"type": "response.created", "response": {"id": "next"}})
            assert adapter.convert({**audio, "response_id": "next"})[0].kind == "audio"
            captured.append({"usage": usage})
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["archived", "live"])
def test_system_image_refuses_before_any_wire_write(providers, captured, phase):
    image = {"role": "system", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}
    wire = AddedWire()
    if phase == "archived":
        with pytest.raises(RealtimeSetupError, match="system_content"):
            make(providers, wire, captured, [{"role": "system", "content": "rules"},
                                 {"role": "user", "content": "history"}, image])
    else:
        adapter = make(providers, wire, captured)
        with pytest.raises(RealtimeSetupError, match="system_content"):
            asyncio.run(adapter.send(ContextUpdate((image,))))
    assert not wire.sent


@pytest.mark.parametrize("mutate", [False, True])
def test_parent_echo_representation_preserves_schema_checks(providers, captured, mutate):
    module = providers["grok"].module
    requested = {
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}},
    }
    actual = {
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        "audio": {"input": {"transport": "json", "format": {"type": "audio/pcm", "rate": 24000}}},
    }
    if mutate:
        actual["tools"][0]["function"]["parameters"] = {"type": "string"}
        with pytest.raises(RealtimeSetupError, match="session_mismatch"):
            module.check_session_echo(requested, actual)
    else:
        module.check_session_echo(requested, actual)
    captured.append({"requested": requested, "actual": actual})


def test_a_message_of_several_text_parts_is_one_part_on_the_wire():
    """xAI joins a message's text parts into one in its acknowledgement; several parts failed the exact echo check
    and closed both Grok calls of 27.10 right after a screenshot (its caption row has two text parts)."""
    from ouroboros.realtime_item_echo import item_matches
    from skills.grok_realtime.plugin import item_rows
    [sent] = item_rows({'role': 'user', 'content': [{'type': 'text', 'text': '[image: a.png]'},
                                                     {'type': 'text', 'text': '[Image not shown]'}]})
    assert sent['content'] == [{'type': 'input_text', 'text': '[image: a.png]\n\n[Image not shown]'}]
    assert item_matches(sent, {**sent, 'status': 'completed', 'content': [dict(sent['content'][0])]})
    [single] = item_rows({'role': 'user', 'content': 'Exact words'})
    assert single['content'] == [{'type': 'input_text', 'text': 'Exact words'}]


# xAI's session.updated echo as recorded by the paid probe of 27.11.2026 (sprint call-fit-20261125,
# probes/grok/sessions/g01-mem-75k-r1/events.jsonl, t=3.601 and t=3.947). Instructions and tools are
# left to the fake wire's own echo; these are the fields the recording holds verbatim.
RECORDED_TUNING = {"threshold": 0.85, "silence_duration_ms": 500, "prefix_padding_ms": 333}
RECORDED_FIRST_ECHO = {
    "turn_detection": {"type": "client_controlled", **RECORDED_TUNING},
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transport": "json",
                        "turn_detection": {"type": "client_controlled", **RECORDED_TUNING}},
              "output": {"format": {"type": "audio/pcm", "rate": 24000}, "transport": "json", "voice": "Ara"}},
    "reasoning": {"effort": "none"}, "voice": "Ara", "model": "grok-voice-3.0",
}
RECORDED_SECOND_ECHO = {
    **RECORDED_FIRST_ECHO,
    "turn_detection": {"type": "server_vad", **RECORDED_TUNING},
    "audio": {"input": {**RECORDED_FIRST_ECHO["audio"]["input"],
                        "turn_detection": {"type": "server_vad", **RECORDED_TUNING}},
              "output": RECORDED_FIRST_ECHO["audio"]["output"]},
}


def _changed(echo, path, value):
    echo = copy.deepcopy(echo)
    *parents, last = path
    node = echo
    for key in parents:
        node = node[key]
    node[last] = value
    return echo


class _Drop:
    def __deepcopy__(self, memo):
        return self


DROP = _Drop()  # an overlay value that removes the key from the fake's echo
WHOLE = "whole"  # an answer ("whole", value) replaces the session value itself; value DROP omits it


class RecordedEchoWire(AddedWire):
    """Answers each session.update with the fake's echo overlaid by the recorded xAI fields.

    An answer of None leaves the fake's exact echo of the request, which the code before the
    27.11 fix accepted too, so a test can reach the update it is about on either version."""

    def __init__(self, first=RECORDED_FIRST_ECHO, second=RECORDED_SECOND_ECHO, tools=None):
        super().__init__()
        self.answers = [first, second]
        self.tools = tools

    def put(self, value):
        if value.get("type") == "session.updated" and self.answers:
            answer = copy.deepcopy(self.answers.pop(0))
            if isinstance(answer, tuple) and answer[0] == WHOLE:
                value = {"type": "session.updated"}
                if answer[1] is not DROP:
                    value["session"] = answer[1]
            else:
                session = {**value["session"], **(answer or {})}
                session = {k: v for k, v in session.items() if v is not DROP}
                if self.tools is not None and "tools" in session:
                    session["tools"] = self.tools
                value = {"type": "session.updated", "session": session}
        super().put(value)


def test_recorded_echo_opens_a_session(providers, captured):
    async def run():
        adapter = make(providers, RecordedEchoWire(), captured)
        try:
            await asyncio.wait_for(adapter.open(), 1)
            assert (await adapter.receive())[0].kind == "ready"
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("first,second,changed", [
    (_changed(RECORDED_FIRST_ECHO, ("audio", "input", "format", "rate"), 16000), RECORDED_SECOND_ECHO, "audio"),
    (_changed(RECORDED_FIRST_ECHO, ("audio", "output", "format", "rate"), 16000), RECORDED_SECOND_ECHO, "audio"),
    (_changed(RECORDED_FIRST_ECHO, ("audio", "input", "format", "type"), "audio/pcmu"), RECORDED_SECOND_ECHO, "audio"),
    (_changed(RECORDED_FIRST_ECHO, ("audio", "output", "transport"), "binary"), RECORDED_SECOND_ECHO, "audio"),
    (_changed(RECORDED_FIRST_ECHO, ("turn_detection", "type"), "server_vad"), RECORDED_SECOND_ECHO, "turn_detection"),
    (_changed(RECORDED_FIRST_ECHO, ("audio", "input", "turn_detection", "type"), "server_vad"),
     RECORDED_SECOND_ECHO, "audio"),
    # A requested direction missing from a present audio echo refuses.
    (_changed(RECORDED_FIRST_ECHO, ("audio", "output"), DROP), RECORDED_SECOND_ECHO, "audio"),
    (RECORDED_FIRST_ECHO, RECORDED_SECOND_ECHO, "tools"),
    # Second-update rows: the first answer is the exact echo every version accepts, so the
    # refusal they pin is the VAD activation's own.
    (None, _changed(RECORDED_SECOND_ECHO, ("turn_detection", "type"), "client_controlled"), "turn_detection"),
    (None, _changed(RECORDED_SECOND_ECHO, ("turn_detection",), None), "turn_detection"),
    (None, _changed(RECORDED_SECOND_ECHO, ("audio", "input", "turn_detection", "type"), "client_controlled"),
     "turn_detection"),
    (None, _changed(_changed(RECORDED_SECOND_ECHO, ("audio", "input", "turn_detection", "type"), "client_controlled"),
                    ("turn_detection",), DROP), "turn_detection"),
])
def test_recorded_echo_dialect_still_refuses_a_changed_session(providers, captured, first, second, changed):
    if isinstance(first, dict) and first["audio"]["output"] is DROP:
        first = {**first, "audio": {"input": first["audio"]["input"]}}
    async def run():
        tools = [{"type": "function", "name": "invented_tool", "parameters": {}}] if changed == "tools" else None
        wire = RecordedEchoWire(first, second, tools)
        adapter = make(providers, wire, captured)
        try:
            with pytest.raises(RealtimeSetupError, match=f"session_mismatch.*Grok changed {changed}"):
                await asyncio.wait_for(adapter.open(), 1)
            if first is None:
                assert wire.sent[-1] == {"type": "session.update", "session": {"turn_detection": {"type": "server_vad"}}}
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("second", [
    RECORDED_SECOND_ECHO,
    # Both representations of turn detection absent: a sparse acknowledgement.
    _changed(_changed(RECORDED_SECOND_ECHO, ("turn_detection",), DROP), ("audio",), DROP),
    ("whole", {}),
    ("whole", DROP),
])
def test_vad_activation_accepts_the_recorded_and_sparse_answers(providers, captured, second):
    async def run():
        adapter = make(providers, RecordedEchoWire(None, second), captured)
        try:
            await asyncio.wait_for(adapter.open(), 1)
            assert (await adapter.receive())[0].kind == "ready"
        finally:
            await adapter.close()
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["first", "second"])
@pytest.mark.parametrize("value", [[], ""])
def test_a_session_that_is_not_an_object_refuses(providers, captured, phase, value):
    async def run():
        answers = ((WHOLE, value), None) if phase == "first" else (None, (WHOLE, value))
        adapter = make(providers, RecordedEchoWire(*answers), captured)
        try:
            with pytest.raises(RealtimeSetupError, match="session_mismatch"):
                await asyncio.wait_for(adapter.open(), 1)
        finally:
            await adapter.close()
    asyncio.run(run())


# The fixed adapter's own open() against xAI with a requested voice, paid proof of 27.11.2026
# (sprint call-fit-20261125, probes/grok/ADAPTER_OPEN_PROOF.txt, p3-fixed-eve t=2.245/2.455 and
# p4-fixed-xai_ara t=1.518/1.739): the requested id comes back at the top level and under audio.output.
def _voice_echo(echo, voice):
    return _changed(_changed(echo, ("voice",), voice), ("audio", "output", "voice"), voice)


PROOF_ECHOES = {voice: (_voice_echo(RECORDED_FIRST_ECHO, voice), _voice_echo(RECORDED_SECOND_ECHO, voice))
                for voice in ("eve", "xai_ara")}
assert PROOF_ECHOES["eve"][0]["audio"]["output"] == {
    "format": {"type": "audio/pcm", "rate": 24000}, "transport": "json", "voice": "eve"}


def _open_with_voice(providers, captured, voice, wire):
    selected = RealtimeInput("synthetic-grok", canonical_json([
        {"role": "system", "content": "Synthetic ordinary rules."},
        {"role": "user", "content": "Synthetic owner words."}]), "[]", 256, "nano", "core", "view")
    adapter = providers["grok"].session(selected, LiveOptions(voice=voice), connect=wire.connect,
                                        observe=lambda direction, raw: captured.append({"raw": raw}))
    async def run():
        try:
            await asyncio.wait_for(adapter.open(), 1)
            assert (await adapter.receive())[0].kind == "ready"
        finally:
            await adapter.close()
    return run


@pytest.mark.parametrize("voice,first", [
    ("eve", PROOF_ECHOES["eve"][0]),
    ("xai_ara", PROOF_ECHOES["xai_ara"][0]),
    # Sparse: no nested voice, or no top-level voice, or neither.
    ("eve", _changed(PROOF_ECHOES["eve"][0], ("audio", "output", "voice"), DROP)),
    ("eve", _changed(PROOF_ECHOES["eve"][0], ("voice",), DROP)),
    ("eve", _changed(_changed(PROOF_ECHOES["eve"][0], ("voice",), DROP), ("audio", "output", "voice"), DROP)),
    # No voice requested: the provider's default under audio.output stays accepted.
    (None, RECORDED_FIRST_ECHO),
])
def test_requested_voice_echo_opens(providers, captured, voice, first):
    first = copy.deepcopy(first)
    output = first["audio"]["output"]
    if output.get("voice") is DROP:
        del output["voice"]
    second = PROOF_ECHOES[voice][1] if voice else RECORDED_SECOND_ECHO
    asyncio.run(_open_with_voice(providers, captured, voice, RecordedEchoWire(first, second))())


@pytest.mark.parametrize("top", ["eve", DROP])
def test_a_contradictory_output_voice_refuses(providers, captured, top):
    first = _changed(_changed(PROOF_ECHOES["eve"][0], ("audio", "output", "voice"), "xai_ara"), ("voice",), top)
    run = _open_with_voice(providers, captured, "eve", RecordedEchoWire(first, PROOF_ECHOES["eve"][1]))
    with pytest.raises(RealtimeSetupError, match="session_mismatch.*Grok changed audio"):
        asyncio.run(run())


def test_recorded_echo_at_the_check_seam(providers):
    module = providers["grok"].module
    requested = {"turn_detection": None, "reasoning": {"effort": "none"},
                 "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}},
                           "output": {"format": {"type": "audio/pcm", "rate": 24000}}}}
    module.check_session_echo(requested, RECORDED_FIRST_ECHO)
    module.check_session_echo({"turn_detection": {"type": "server_vad"}}, RECORDED_SECOND_ECHO)
    with pytest.raises(RealtimeSetupError, match="session_mismatch.*turn_detection"):
        module.check_session_echo(requested, {**RECORDED_FIRST_ECHO, "turn_detection": RECORDED_SECOND_ECHO["turn_detection"]})
    with pytest.raises(RealtimeSetupError, match="session_mismatch.*audio"):
        module.check_session_echo(requested, {**RECORDED_FIRST_ECHO, "audio": RECORDED_SECOND_ECHO["audio"]})
    with pytest.raises(RealtimeSetupError, match="session_mismatch.*audio"):
        module.check_session_echo(requested, _changed(RECORDED_FIRST_ECHO, ("audio", "output"), {"transport": "json"}))
