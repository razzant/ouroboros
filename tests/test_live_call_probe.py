"""Probe verdicts require observations, and the owner simulator preserves pacing."""
import asyncio
import base64
import json
from unittest.mock import AsyncMock, Mock

import pytest

from devtools import live_call_probe as probe
from ouroboros.call_peers import LiveCallHub
from ouroboros.call_protocol import CallSpec, PeerAudio
from ouroboros.gateways.realtime_engine import AudioIn, EngineError, EngineRoute, SessionConfig, ToolResult


def test_output_requires_new_external_temp_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "REPO", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    with pytest.raises(ValueError, match="outside"):
        probe.output_root(str(tmp_path / "repo" / "output"))
    with pytest.raises(ValueError, match="already exists"):
        probe.output_root(str(tmp_path))
    root = probe.output_root(str(tmp_path / "fresh"))
    assert root.is_dir()
    with pytest.raises(ValueError, match="already exists"):
        probe.output_root(str(root))


def test_a_data_copy_never_carries_the_source_daemon_home_or_its_request_dumps(tmp_path):
    source = tmp_path / "live"
    (source / "claudexor" / "daemon").mkdir(parents=True)
    (source / "claudexor" / "daemon" / "control-api.json").write_text("{}")
    (source / "observability" / "calls").mkdir(parents=True)
    (source / "state" / "cx" / "3.24.0").mkdir(parents=True)
    (source / "state" / "queue_snapshot.json").write_text("{}")
    (source / "memory").mkdir()
    (source / "memory" / "identity.md").write_text("me")
    (source / "settings.json").write_text("{}")
    (tmp_path / "run").mkdir()
    copied = tmp_path / "run" / "data"
    probe.copy_data(source, copied)
    assert (copied / "memory" / "identity.md").read_text() == "me" and (copied / "settings.json").exists()
    assert not (copied / "claudexor").exists() and not (copied / "observability").exists()
    assert (copied / "state" / "queue_snapshot.json").exists() and not (copied / "state" / "cx").exists()


def test_a_running_installs_own_data_is_refused_and_a_copy_of_it_is_taken(tmp_path):
    live = tmp_path / "Ouroboros"
    (live / "data" / "memory").mkdir(parents=True)
    (live / "ouroboros.pid").write_text("123")
    with pytest.raises(ValueError, match="refusing a live install"):
        probe.data_copy_source(str(live / "data"))
    versioned = tmp_path / "versioned"
    (versioned / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing a live install"):
        probe.data_copy_source(str(versioned))
    copy = tmp_path / "copy" / "data"
    copy.mkdir(parents=True)
    assert probe.data_copy_source(str(copy)) == copy.resolve()


def test_the_recorded_wire_labels_resume_handles_both_ways_and_keeps_no_audio_or_instruction():
    handles = {}
    instruction = "The whole call view. " * 20
    received = probe.redacted_wire({"sessionResumptionUpdate": {"newHandle": "token-1", "resumable": True}}, handles)
    sent = probe.redacted_wire({"setup": {"systemInstruction": {"parts": [{"text": instruction}]},
                                          "sessionResumption": {"handle": "token-1"}}}, handles)
    audio = probe.redacted_wire({"serverContent": {"modelTurn": {"parts": [
        {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(b"\x01\x00" * 480).decode()}},
        {"text": "Привет"}]}}}, handles)
    # A reopen sends back the handle it received: one label in both directions, the token in neither.
    assert sent["setup"]["sessionResumption"]["handle"] == received["sessionResumptionUpdate"]["newHandle"] == "recorded-handle-1"
    assert "token-1" not in json.dumps([received, sent])
    assert sent["setup"]["systemInstruction"] == {"recorded_chars": len(instruction),
                                                  "sha256": probe.hashlib.sha256(instruction.encode()).hexdigest()}
    parts = audio["serverContent"]["modelTurn"]["parts"]
    assert parts[0]["inlineData"] == {"mimeType": "audio/pcm;rate=24000", "recorded_bytes": 960, "data": probe.TINY_PCM}
    assert parts[1] == {"text": "Привет"}  # a fresh run's words stay: they are the probe's own synthetic speech


def test_a_data_copys_older_ledger_neither_counts_as_the_runs_money_nor_spends_its_budget(tmp_path, monkeypatch):
    from ouroboros import usage_accounting

    rows = [{"attempt_id": "old", "ts_reserved": "2026-10-16T18:58:40.041002+00:00", "cost_usd": "legacy"},
            {"attempt_id": "new", "ts_reserved": "2026-10-17T02:03:37.710432+00:00", "cost_usd": 0.01}]
    assert [r["attempt_id"] for r in probe.attempts_since(rows, "2026-10-17T02:03:20.000000+00:00")] == ["new"]
    assert len(probe.attempts_since(rows, "2026-10-01T00:00:00.000000+00:00")) == 2  # a fresh run keeps everything
    monkeypatch.setattr(usage_accounting, "usage_projection", lambda data, **_: {"settled_usd": 1234.5})
    assert probe.probe_budget(tmp_path) == "1244.5"  # ten dollars beyond what the copy has already spent


def test_only_a_fresh_run_writes_a_replay_fixture():
    wire = [{"direction": "server", "message": {"setupComplete": {}}}]
    record = probe.recorded_exchange(wire, "gemini-3.8-live", data_copy=False)
    assert record["header"]["provenance"] == "recorded_live" and record["events"] == wire
    assert probe.recorded_exchange(wire, "gemini-3.8-live", data_copy=True) is None  # a copy's wire is its memory
    assert probe.recorded_exchange([], "gemini-3.8-live", data_copy=False) is None


def test_scenario_selection_rejects_typos():
    assert probe.select_scenarios("opening,money") == {"opening", "money"}
    assert probe.select_scenarios("all") == set(probe.SCENARIOS)
    with pytest.raises(ValueError, match="unknown"):
        probe.select_scenarios("openning")


def attempts():
    counts = {"text_input_tokens": 12, "audio_input_tokens": 44}
    evidence = {"response_key": "response", "usage": counts, "raw_usage": {"input_tokens": 56},
                "tariff_quote": {"url": "https://catalog", "fetched_at": "2026-10-16", "sha256": "digest", "entry": {"rate": 1}}}
    rows = [{"attempt_id": "response-attempt", "source": "live_response", "state": "settled",
             "cost_evidence": evidence, "prompt_tokens": 12, "cost_usd": .1,
             "ts_reserved": "start", "ts_dispatched": "sent", "ts_final": "settled"},
            {"attempt_id": "fixture-attempt", "source": "probe_speech_fixture", "state": "settled",
             "cost_evidence": {**evidence, "response_key": "fixture"}, "prompt_tokens": 12, "cost_usd": .02}]
    events = [{"kind": "ResponseStarted", "response_id": "response"}, {"kind": "ResponseDone", "response_id": "response"}]
    return rows, events


def test_money_matches_response_id_and_keeps_audio_separate():
    rows, events = attempts()
    ok, facts = probe.money_evidence(rows, events)
    assert ok and facts["total_cost_usd"] == pytest.approx(.12)
    rows[0]["prompt_tokens"] += 44
    ok, facts = probe.money_evidence(rows, events)
    assert not ok and any("audio included" in e for e in facts["errors"])


@pytest.mark.parametrize("change", ["duplicate", "missing_done", "unknown_price", "no_quote", "unresolved", "no_reservation"])
def test_money_cannot_pass_incomplete_or_double_billed_evidence(change):
    rows, events = attempts()
    if change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "missing_done":
        events.pop()
    elif change == "unknown_price":
        rows[0]["cost_usd"] = None
    elif change == "no_quote":
        rows[0]["cost_evidence"]["tariff_quote"] = None
    elif change == "no_reservation":
        rows[0].pop("ts_reserved")
    else:
        rows[0]["state"] = "unresolved"
    ok, facts = probe.money_evidence(rows, events)
    assert not ok and facts["errors"]
    if change == "unknown_price":
        assert facts["total_cost_usd"] is None
        assert facts["known_estimated_usd"] == .02


def test_a_grounded_response_is_a_disclosed_unknown_price_and_any_other_unknown_still_fails():
    rows, events = attempts()
    rows[0]["cost_usd"] = None
    rows[0]["cost_evidence"]["unpriced"] = ["groundingMetadata"]  # Gemini's search charge has no catalog tariff
    ok, facts = probe.money_evidence(rows, events)
    assert ok and facts["total_cost_usd"] is None and facts["known_estimated_usd"] == .02
    assert facts["grounded_unknown_priced_attempts"] == ["response-attempt"]
    rows[0]["cost_evidence"]["unpriced"] = ["groundingMetadata", "input_cost_per_audio_token"]
    ok, facts = probe.money_evidence(rows, events)
    assert not ok and facts["grounded_unknown_priced_attempts"] == []


def test_observer_forwards_real_commands_and_records_only_audio_lengths(tmp_path):
    class Engine:
        _write = AsyncMock()
        _handle_message = AsyncMock()
        open = AsyncMock()
        send = AsyncMock()
        close = AsyncMock()

        async def events(self):
            yield EngineError("invalid_request", "Rejected instructions")

    async def run():
        engine = Engine()
        evidence = probe.Evidence(tmp_path)
        observed = probe.ObservedEngine(engine, evidence, object())
        audio, result = AudioIn(b"\x01\x00" * 480, 24000), ToolResult("call", "weather result")
        await observed.send(audio)
        await observed.send(result)
        assert engine.send.call_args_list[0].args == (audio,)
        assert engine.send.call_args_list[1].args == (result,)
        assert len(evidence.rows) == 1 and evidence.rows[0]["output"] == "weather result"
        config = SessionConfig(EngineRoute("google", "test"), "A complete instruction")
        await observed.open(config)
        assert engine.open.call_args.args == (config,)
        assert (tmp_path / "instructions-1.txt").read_text() == config.instructions
        assert evidence.of("session_config")[0]["instruction_tokens"] > 0
        assert [e.kind async for e in observed.events()] == ["invalid_request"]
        assert evidence.of("EngineError")[0]["error_kind"] == "invalid_request"

    asyncio.run(run())


def test_owner_audio_uses_real_codec_and_engine_rate(tmp_path):
    async def run():
        hub = LiveCallHub(CallSpec("call", 1, "google", "test"))
        hub.set_engine_rates(probe.RATE, 24000)  # an engine at the device's rate: frames pass through unchanged
        peer = probe.OwnerPeer(hub, probe.Evidence(tmp_path))
        # Drain the genuine mixer concurrently, as the driver does.
        received = bytearray()
        async def drain():
            while not hub.closed:
                event = await asyncio.to_thread(hub.next_event, .025)
                if isinstance(event, PeerAudio):
                    received.extend(event.pcm)
        task = asyncio.create_task(drain())
        await peer.speak(b"\x01\x00" * 480, "fixture")
        for _ in range(200):
            if len(received) == 960 + probe.RATE * 2:
                break
            await asyncio.sleep(.01)
        hub.close("done")
        await task
        assert received == b"\x01\x00" * 480 + bytes(probe.RATE * 2)
        assert hub.stats()["input_dropped_samples"] == 0
        start = peer.evidence.of("speech_start")[0]["at"]
        end = peer.evidence.of("speech_end")[0]["at"]
        assert end - start >= .019
    asyncio.run(run())


def test_failed_open_keeps_unexecuted_scenarios_not_run(tmp_path):
    runner = probe.Probe(tmp_path, set(probe.SCENARIOS), 1)
    runner.verdict("opening", False, error="rejected by provider")
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["scenarios"]["opening"]["verdict"] == "FAIL"
    assert report["scenarios"]["tool"]["verdict"] == "NOT_RUN"
    assert report["total_cost_usd"] is None


def test_speech_sse_done_marker_preserves_real_usage_and_pcm(tmp_path, monkeypatch):
    import requests
    from ouroboros import live_usage, pricing_catalog, usage_accounting

    usage = {"input_tokens": 16, "output_tokens": 150}
    pcm = b"\x01\x00" * 480
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def raise_for_status(self):
            pass
        def iter_lines(self):
            yield b"event: speech.audio.delta"
            yield b"data: " + json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(pcm).decode()}).encode()
            yield b"data: " + json.dumps({"type": "speech.audio.done", "usage": usage}).encode()
            yield b"data: [DONE]"
    monkeypatch.setattr(requests, "post", lambda *a, **k: Response())
    monkeypatch.setattr(pricing_catalog, "get_catalog_quote", lambda *a, **k: {"source": "test"})
    settlement = Mock(return_value=({"cost_evidence": {"raw_usage": usage}}, .01))
    monkeypatch.setattr(live_usage, "_settlement", settlement)
    for method in ("reserve_attempt", "mark_dispatched", "settle_attempt"):
        monkeypatch.setattr(usage_accounting, method, Mock())
    assert probe.synthesize(tmp_path, "russian_reply", "test-key") == pcm
    assert settlement.call_args.args[2] == {"text_input_tokens": 16, "audio_output_tokens": 150}
    assert json.loads((tmp_path / "input-russian_reply.json").read_text())["usage"] == usage


def test_fixture_reuse_requires_matching_text_and_retains_provenance(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    probe.write_json(source / "input-tool.json", {"text": probe.SPEECH["tool"]})
    (source / "input-tool.pcm").write_bytes(b"\x01\x00")
    assert probe.speech_fixture(output, "tool", "unused", str(source)) == b"\x01\x00"
    assert json.loads((output / "input-tool.json").read_text())["reused_from"].endswith("input-tool.json")
    probe.write_json(source / "input-tool.json", {"text": "different request"})
    with pytest.raises(ValueError, match="does not match"):
        probe.speech_fixture(output, "tool", "unused", str(source))


def test_owner_microphone_keeps_sending_quiet_frames_between_utterances(tmp_path):
    async def run():
        hub = LiveCallHub(CallSpec("call", 1, "google", "test"))
        evidence = probe.Evidence(tmp_path)
        peer = probe.OwnerPeer(hub, evidence)
        evidence.note("Ready")
        pump = asyncio.create_task(peer.pump())
        received = bytearray()
        try:
            async def collect():
                while len(received) < 3 * probe.FRAME_BYTES:
                    event = await asyncio.to_thread(hub.next_event, .025)
                    if isinstance(event, PeerAudio):
                        received.extend(event.pcm)
            await asyncio.wait_for(collect(), 3)
            assert not any(received) and not evidence.of("speech_start")
        finally:
            hub.close("test_done")
            await pump
    asyncio.run(run())


def test_the_isolated_child_receives_the_settings_the_parent_resolved(tmp_path, monkeypatch):
    install = tmp_path / "install-data"
    install.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(install))
    started: list = []
    monkeypatch.setattr(probe, "daemon_list", lambda: [])
    monkeypatch.setattr(probe.subprocess, "call", lambda command, **kwargs: started.append((command, kwargs["env"])) or 0)
    assert probe.main(["--output-dir", str(tmp_path / "run")]) == 0
    (command, env), = started
    assert command[command.index("--settings-source") + 1] == str((install / "settings.json").resolve())
    assert command[-1] == "--isolated-child" and env["OUROBOROS_DATA_DIR"] != str(install)
