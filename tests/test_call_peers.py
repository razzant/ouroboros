"""Audio transport checks use real sample clocks and concurrent producers."""

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import json
from ouroboros import call_protocol as cp
from ouroboros.call_peers import CallPeerError, LinearResampler, LiveCallHub, close_hub, get_hub, open_hub


def pcm(values):
    import struct
    return struct.pack(f"<{len(values)}h", *values)


def samples(data):
    import struct
    return list(struct.unpack(f"<{len(data) // 2}h", data))


def spec(call_id="test-call"):
    return cp.CallSpec(call_id, 1, "google", "test-model")


def hub_at_24k(value=None, **kwargs):
    """A hub whose engine runs at 24 kHz both ways, so the 24 kHz peers below mix and play without resampling."""
    return LiveCallHub(value or spec(), rates=(24000, 24000), **kwargs)


def join(hub, peer_id="owner", *, role="owner", mic=True, speaker=True, rate=24000):
    hub.join(cp.PeerInfo(peer_id, role, mic, speaker), rate)
    assert isinstance(hub.next_event(0), cp.PeerJoined)
    assert cp.decode_control(hub.next_outgoing(peer_id))["type"] == "state"
    assert cp.decode_control(hub.next_outgoing(peer_id))["type"] == "flush"


def send(hub, values, peer_id="owner", rate=24000, seq=0, epoch=None, stream=0):
    return hub.receive_audio(peer_id, cp.AudioFrame(cp.FRAME_AUDIO_IN, hub.epoch if epoch is None else epoch,
                                                  seq, rate, stream, pcm(values)))


@pytest.mark.parametrize("rates", [(48000, 16000), (48000, 24000), (44100, 24000), (32000, 16000)])
def test_stateful_resampling_chunk_identity_and_round_trip_error(rates):
    source_rate, wire_rate = rates
    original = [round(12000 * math.sin(2 * math.pi * 440 * i / source_rate)) for i in range(source_rate // 5)]
    raw = pcm(original)
    complete = LinearResampler(source_rate, wire_rate).feed(raw)
    converter = LinearResampler(source_rate, wire_rate)
    chunked = b"".join(converter.feed(raw[i:i + 638]) for i in range(0, len(raw), 638))
    assert complete == chunked
    returned = samples(LinearResampler(wire_rate, source_rate).feed(complete))
    assert len(original) - len(returned) < 8
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(original, returned)) / len(returned))
    assert rms < 45, (rates, rms)


def test_mixing_clips_once_and_keeps_conservative_provenance():
    hub = hub_at_24k()
    join(hub)
    join(hub, "meeting", role="meeting")
    send(hub, [30000, -30000, 100], "owner")
    send(hub, [30000, -30000, -50], "meeting")
    event = hub.next_event(0.1)
    assert (event.peer_id, event.role) == ("mixed", "meeting")
    assert samples(event.pcm) == [32767, -32768, 50]
    assert hub.next_event(0) is None
    send(hub, [10], seq=1)
    assert hub.next_event(0.1) == cp.PeerAudio("owner", "owner", pcm([10]))


def test_tracks_and_different_device_rates_share_one_engine_timeline():
    hub = hub_at_24k()
    hub.set_engine_rates(16000, 24000)
    join(hub, rate=48000)
    join(hub, "meeting", role="meeting", rate=16000)
    send(hub, [1000] * 960, rate=48000)
    send(hub, [2000] * 320, "meeting", rate=16000)
    event = hub.next_event(0.1)
    assert samples(event.pcm) == [3000] * 320
    hub.play(pcm([1000] * 480), "item")
    owner = cp.decode_frame(hub.next_outgoing("owner"))
    meeting = cp.decode_frame(hub.next_outgoing("meeting"))
    assert owner.rate == 48000 and meeting.rate == 16000
    assert set(samples(owner.pcm)) == set(samples(meeting.pcm)) == {1000}


def test_flush_discards_queued_audio_and_resampler_tail_but_keeps_captions():
    hub = hub_at_24k()
    join(hub, rate=48000)
    hub.send_caption("ouroboros", "Hello", True, "item")
    hub.play(pcm([1234] * 480), "item")
    assert hub.flush() == 1
    assert cp.decode_control(hub.next_outgoing("owner")) == {"type": "flush", "epoch": 1}
    assert cp.decode_control(hub.next_outgoing("owner"))["type"] == "caption"
    assert hub.next_outgoing("owner") is None
    hub.play(pcm([0] * 480), "next")
    frame = cp.decode_frame(hub.next_outgoing("owner"))
    assert frame.epoch == 1 and set(samples(frame.pcm)) == {0}
    assert not send(hub, [1], rate=48000, epoch=0)
    assert send(hub, [1], rate=48000, epoch=1)
    hub._epoch = 65535
    assert hub.flush() == 0


def test_audio_queue_bounds_drop_oldest_with_counters_and_sequence_wrap():
    hub = hub_at_24k(None, audio_queue_frames=2, output_queue_frames=2)
    join(hub)
    send(hub, [1] * 480, seq=0xFFFFFFFF)
    send(hub, [2] * 480, seq=0)
    send(hub, [3] * 480, seq=1)
    assert not send(hub, [4], seq=1)
    assert hub.stats()["input_dropped_samples"] == 480
    assert samples(hub.next_event(0.1).pcm) == [2] * 480
    assert samples(hub.next_event(0.1).pcm) == [3] * 480
    for value in (1, 2, 3):
        hub.play(pcm([value] * 480), "item")
    assert hub.stats()["output_dropped_frames"] == 1
    assert samples(cp.decode_frame(hub.next_outgoing("owner")).pcm) == [2] * 480
    assert samples(cp.decode_frame(hub.next_outgoing("owner")).pcm) == [3] * 480


def test_output_absorbs_a_recorded_realtime_burst_without_changing_input_bound():
    hub = hub_at_24k()
    join(hub)
    # The real provider emitted 4.15 seconds in 112 ms: the input capture's
    # one-second bound must not cut this normal outbound burst to one second.
    blocks = [bytes(19200)] * 10 + [bytes(7200)]
    for block in blocks:
        hub.play(block, "spoken")
    received = bytearray()
    while (frame := hub.next_outgoing("owner")) is not None:
        if isinstance(frame, bytes):
            received.extend(cp.decode_frame(frame).pcm)
    assert len(received) == sum(map(len, blocks))
    assert hub.stats()["output_dropped_frames"] == 0
    assert hub._audio_limit == 50
    with pytest.raises(ValueError, match="positive"):
        hub_at_24k(None, output_queue_frames=0)


def test_mute_direction_checks_and_peer_lifecycle():
    hub = hub_at_24k()
    join(hub)
    send(hub, [100])
    hub.receive_control("owner", {"type": "mute", "muted": True})
    assert isinstance(hub.next_event(0), cp.PeerControl)
    assert hub.next_event(0) is None
    assert not send(hub, [200], seq=1)
    hub.receive_control("owner", {"type": "mute", "muted": False})
    hub.next_event(0)
    assert send(hub, [300], seq=2)
    assert samples(hub.next_event(0.1).pcm) == [300]
    join(hub, "speaker", mic=False)
    with pytest.raises(CallPeerError, match="direction"):
        send(hub, [1], "speaker")
    hub.leave("owner", "bye")
    assert hub.next_event(0) == cp.PeerLeft("owner", "bye")
    hub.leave("owner")
    assert hub.next_event(0) is None


def test_control_pressure_refuses_instead_of_dropping_unread_facts():
    hub = hub_at_24k(None, control_queue_messages=2)
    join(hub)
    for value in (True, False):
        hub.receive_control("owner", {"type": "mute", "muted": value})
    with pytest.raises(CallPeerError, match="control_queue_full"):
        hub.receive_control("owner", {"type": "context", "text": "cannot fit"})
    hub.leave("owner", "control_queue_full")
    assert [type(hub.next_event(0)) for _ in range(3)] == [cp.PeerControl, cp.PeerControl, cp.PeerLeft]
    assert hub.stats()["control_overflows"] == 1


def test_registry_concurrency_close_wakes_waiter_and_revokes_grants():
    value = spec("concurrent-call")
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            hubs = list(pool.map(open_hub, [value] * 32))
        hub = hubs[0]
        assert all(item is hub for item in hubs)
        assert get_hub(value.call_id) is hub
        hub.grant_peer("meeting", "token-digest", mic=True, speaker=False)
        with pytest.raises(CallPeerError, match="direction_mismatch"):
            hub.grant_peer("meeting", "token-digest", mic=True, speaker=True)
        with pytest.raises(CallPeerError, match="not_admitted"):
            hub.peer_grant("other", "token-digest")
        result = []
        thread = threading.Thread(target=lambda: result.append(hub.next_event(10)))
        thread.start()
        close_hub(value.call_id)
        thread.join(1)
        assert not thread.is_alive() and result == [None]
        assert get_hub(value.call_id) is None
        with pytest.raises(CallPeerError, match="call_closed"):
            hub.peer_grant("meeting", "token-digest")
    finally:
        close_hub(value.call_id)


def test_mixer_waits_one_audio_tick_for_asynchronous_microphones(monkeypatch):
    from ouroboros import call_peers

    clock = [100.0]
    monkeypatch.setattr(call_peers.time, "monotonic", lambda: clock[0])
    hub = hub_at_24k()
    join(hub)
    join(hub, "meeting", role="meeting")
    send(hub, [1000] * 480)
    assert hub.next_event(0) is None
    clock[0] += 0.005
    send(hub, [2000] * 480, "meeting")
    assert hub.next_event(0) is None
    clock[0] += 0.016
    assert samples(hub.next_event(0).pcm) == [3000] * 480
    assert hub.next_event(0) is None


def test_engine_rate_change_resets_buffers_and_notifies_peers():
    hub = hub_at_24k()
    join(hub, rate=48000)
    send(hub, [100] * 960, rate=48000)
    hub.play(pcm([100] * 480), "item")
    hub.set_engine_rates(16000, 24000)
    assert hub.stats()["input_dropped_samples"] == 480
    assert cp.decode_control(hub.next_outgoing("owner"))["epoch"] == hub.epoch
    assert hub.next_event(0) is None
    hub.set_engine_rates(16000, 24000)
    assert hub.next_outgoing("owner") is None
    send(hub, [200] * 960, rate=48000, seq=1)
    assert samples(hub.next_event(.1).pcm) == [200] * 320


def test_peer_and_stream_capacity_have_accept_and_refusal_paths(monkeypatch):
    from ouroboros import call_peers

    monkeypatch.setattr(call_peers, "MAX_PEERS", 2)
    monkeypatch.setattr(call_peers, "MAX_STREAMS", 2)
    hub = hub_at_24k()
    join(hub)
    join(hub, "second")
    with pytest.raises(CallPeerError, match="peer_capacity_full"):
        hub.join(cp.PeerInfo("third", "owner", True, True), 24000)
    hub.leave("second")
    hub.next_event(0)
    join(hub, "third")
    send(hub, [1], stream=0)
    send(hub, [2], seq=1, stream=1)
    with pytest.raises(CallPeerError, match="stream_capacity_full"):
        send(hub, [3], seq=2, stream=2)
    for token in ("a", "b"):
        hub.grant_peer("meeting", token, mic=True, speaker=False)
    with pytest.raises(CallPeerError, match="peer_capacity_full"):
        hub.grant_peer("meeting", "c", mic=True, speaker=False)


@pytest.mark.parametrize("rate", [0, -1, 7999, 192001, True, 16000.5])
def test_unsupported_rates_refuse_before_join(rate):
    hub = hub_at_24k()
    with pytest.raises(CallPeerError, match="unsupported_audio_rate"):
        hub.join(cp.PeerInfo("owner", "owner", True, True), rate)
    assert hub.peers() == ()


def test_speaker_direction_and_slow_control_consumer():
    hub = hub_at_24k(None, control_queue_messages=2)
    join(hub, speaker=False)
    hub.play(pcm([12] * 480), "item")
    assert hub.next_outgoing("owner") is None
    with pytest.raises(CallPeerError, match="audio_direction_forbidden"):
        hub.receive_control("owner", {"type": "played", "item_id": "item", "played_ms": 10})
    for value in ("a", "b", "c"):
        hub.send_caption("ouroboros", value, True)
    with pytest.raises(CallPeerError, match="control_queue_full"):
        hub.next_outgoing("owner")
    assert hub.stats()["control_overflows"] == 1
    hub.close("finished")
    with pytest.raises(CallPeerError, match="call_closed"):
        send(hub, [1])


def test_registry_refuses_conflicting_spec_and_closed_hub():
    value = spec("registry-conflict")
    hub = open_hub(value)
    try:
        with pytest.raises(CallPeerError, match="call_id_conflict"):
            open_hub(cp.CallSpec(value.call_id, 9, "google", "another-model"))
        hub.close("finished")
        with pytest.raises(CallPeerError, match="call_id_conflict"):
            open_hub(value)
    finally:
        close_hub(value.call_id)


def test_a_hub_is_born_preparing_and_remembers_an_owner_bye():
    spec = cp.CallSpec("born-preparing", 1, "google", "m")
    hub = LiveCallHub(spec)
    hub.join(cp.PeerInfo("p1", cp.PEER_OWNER, True, True), 24000)
    first = json.loads(hub.next_outgoing("p1"))
    assert first == {"type": "state", "state": "preparing", "reason": "", "engine": "google", "phase": "memory"}
    assert not hub.ended_by_owner
    hub.receive_control("p1", {"type": "bye"})
    assert hub.ended_by_owner
    meeting = LiveCallHub(cp.CallSpec("meeting-bye", 1, "google", "m"))
    meeting.join(cp.PeerInfo("m1", cp.PEER_MEETING, True, True), 24000)
    meeting.receive_control("m1", {"type": "bye"})
    assert not meeting.ended_by_owner  # only the owner's surface ends a call this way


def test_a_closed_state_keeps_the_drivers_words_and_close_adds_them_only_when_missing():
    hub = LiveCallHub(cp.CallSpec("words", 1, "google", "m"))
    hub.join(cp.PeerInfo("p1", cp.PEER_OWNER, True, True), 24000)
    hub.next_outgoing("p1"), hub.next_outgoing("p1")
    hub.send_activity("working", "web_search")
    assert json.loads(hub.next_outgoing("p1")) == {"type": "activity", "activity": "working", "label": "web_search"}
    hub.send_state(cp.STATE_CLOSED, "provider_closed", detail="You exceeded your current quota.", action="retry")
    hub.close("provider_closed")
    states = []
    while (message := hub.next_outgoing("p1")) is not None:
        if isinstance(message, str) and json.loads(message)["type"] == "state":
            states.append(json.loads(message))
    assert states == [{"type": "state", "state": "closed", "reason": "provider_closed", "engine": "google",
                       "detail": "You exceeded your current quota.", "action": "retry"}]
    other = LiveCallHub(cp.CallSpec("bare", 1, "google", "m"))
    other.join(cp.PeerInfo("p2", cp.PEER_OWNER, True, True), 24000)
    other.next_outgoing("p2"), other.next_outgoing("p2")
    other.close("call_not_started", action="retry")
    tail = [json.loads(m) for m in iter(lambda: other.next_outgoing("p2"), None) if isinstance(m, str)]
    assert tail[-1] == {"type": "state", "state": "closed", "reason": "call_not_started", "engine": "google",
                        "action": "retry"}
