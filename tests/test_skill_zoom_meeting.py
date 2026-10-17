"""skills/zoom_meeting: the Zoom Meeting SDK companion, its tools and its call peer.

A fake ``zoom_meeting_sdk`` (``tests/_zoom_meeting_fakes.py``, injected through
``sys.modules``) mirrors the 0.0.27 binding surface Hope's code uses, so a misspelt SDK
call fails here instead of on a server. The Host door is a real Starlette app on a
loopback port, the peer socket a real WebSocket, and frames are checked against the
core's own codec.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

from ouroboros import call_protocol as cp
from ouroboros.contracts.skill_manifest import VALID_SKILL_PERMISSIONS, parse_skill_manifest_text
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint, save_presence_binding
from tests._zoom_meeting_fakes import (
    BINDING,
    CLIENT_ID,
    CLIENT_SECRET,
    HOST_TOKEN,
    MEETING,
    OBF,
    PASSCODE,
    FakeApi,
    FakeCompanion,
    FakePacer,
    FakePeer,
    FakeZoom,
    ManualLoop,
    _libc,
    drive_to_audio,
    join_args,
    open_call_via_audio,
    serve_app,
    start_fake_door,
)

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "zoom_meeting"
SCRIPTS = SKILL_DIR / "scripts"
_SCRIPT_MODULES = ("companion_link", "glib_loop", "raw_audio", "zoom_session", "meeting_peer", "zoom_bot", "probe")


_LIVE_BOTS: list = []


@pytest.fixture(autouse=True)
def _close_bots():
    """Every companion a test built ends its call threads and cleans its SDK up."""
    yield
    while _LIVE_BOTS:
        _LIVE_BOTS.pop().close(timeout=3)


@pytest.fixture
def zoom(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in _SCRIPT_MODULES:
        sys.modules.pop(name, None)
    yield types.SimpleNamespace(**{name: importlib.import_module(name) for name in _SCRIPT_MODULES})
    for name in _SCRIPT_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture
def plugin():
    name = "zoom_meeting_skill_under_test"
    spec = importlib.util.spec_from_file_location(name, SKILL_DIR / "plugin.py",
                                                  submodule_search_locations=[str(SKILL_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    for key in [key for key in sys.modules if key == name or key.startswith(name + ".")]:
        sys.modules.pop(key, None)


def make_bot(zoom, tmp_path, fake, *, door=None, refuse=None, **kwargs):
    loop = ManualLoop()
    peers, bodies = [], []

    def open_call(url, token, body):
        bodies.append(body)
        if refuse:
            raise zoom.meeting_peer.CallOpenRefused(*refuse)
        return {"call_id": "call-7", "peer_path": "/call/call-7/peer"}

    def attach(url, token, path, hello, **callbacks):
        peers.append(FakePeer(hello, callbacks))
        return peers[-1]

    options = dict(open_call=open_call, attach_peer=attach, identity_name=lambda url, token: "Aika",
                   pacer_factory=FakePacer)
    options.update(kwargs)
    bot = zoom.zoom_bot.ZoomBot(state_dir=tmp_path, loop=loop, sdk=fake.m, client_id=CLIENT_ID,
                                client_secret=CLIENT_SECRET, host_url=door or "http://127.0.0.1:9",
                                host_token=HOST_TOKEN, clock=lambda: loop.now, **options)
    _LIVE_BOTS.append(bot)
    return bot, loop, peers, bodies


# ---- manifest -------------------------------------------------------------------------

def test_manifest_parses_and_preflight_is_clean(tmp_path, monkeypatch):
    from ouroboros.skill_dependencies import normalize_declared_dependency_specs
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    manifest = parse_skill_manifest_text((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))
    assert (manifest.name, manifest.type, manifest.runtime, manifest.entry, manifest.os) == (
        "zoom_meeting", "extension", "python3", "plugin.py", "linux")
    assert set(manifest.permissions) == {"tool", "presence", "call_peer", "companion_process", "net", "read_settings"}
    assert manifest.env_from_settings == ["ZOOM_CLIENT_ID", "ZOOM_CLIENT_SECRET"]
    assert "presence" not in manifest.raw_extra  # a transport: the behavior is the install's own profile
    assert manifest.companion_processes == [{"name": "zoom_bot", "command": ["python3", "scripts/zoom_bot.py"],
                                             "runtime": "python3", "restart_policy": "on_failure",
                                             "max_restarts": 5}]
    assert set(manifest.model_experience) == {"what_model_sees", "token_effect"}
    expected = [] if "call_peer" in VALID_SKILL_PERMISSIONS else [
        w for w in manifest.validate() if "'call_peer'" in w]  # until the audio lane's permission lands
    assert manifest.validate() == expected
    auto, manual, warnings = normalize_declared_dependency_specs(manifest.raw_extra["install_specs"])
    assert (manual, warnings) == ([], [])
    assert [spec["package"] for spec in auto] == ["PyJWT==2.13.0", "zoom-meeting-sdk==0.0.27"]
    assert all(spec["platforms"] == ["linux-x86_64"] for spec in auto)
    assert auto[-1]["check"] == {"argv": ["python3", "-c", "import jwt, zoom_meeting_sdk"]}

    skills_root = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, skills_root / "zoom_meeting", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    (tmp_path / "repo").mkdir()
    (tmp_path / "drive").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "drive")
    payload = json.loads(_handle_skill_preflight(ctx, skill="zoom_meeting"))
    assert payload["ok"] is True, payload
    assert payload["files_failed"] == 0 and payload["files_checked"] >= 7


# ---- meeting references and bindings ----------------------------------------------------

@pytest.mark.parametrize("meeting, expected", [
    ("https://us02web.zoom.us/j/85012345678?pwd=AbC123.1", (MEETING, "AbC123.1")),
    ("zoom.us/j/85012345678", (MEETING, "")),
    ("850 1234 5678", (MEETING, "")),
    ("850-1234-5678", (MEETING, "")),
    ("https://company.zoom.us/wc/join/85012345678?pwd=x9", (MEETING, "x9")),
    ("https://zoom.us/wc/85012345678/join", (MEETING, "")),
    ("https://zoom.us/s/85012345678", (MEETING, "")),
    ("https://zoom.us/w/85012345678?tk=abc", (MEETING, "")),
    ("https://zoom.com/join?confno=85012345678", (MEETING, "")),
    ("zoommtg://zoom.us/join?action=join&confno=85012345678&pwd=p1", (MEETING, "p1")),
])
def test_meeting_links_and_numbers_parse_by_url_structure(plugin, meeting, expected):
    assert plugin.parse_meeting(meeting) == expected


@pytest.mark.parametrize("meeting, fragment", [
    ("", "empty"), ("https://zoom.us/my/anton", "personal link"), ("https://example.com/j/85012345678", "not a Zoom"),
    ("https://zoom.us.evil.example/j/85012345678", "not a Zoom"), ("https://agency.zoomgov.com/j/1", "ZoomGov"),
    ("https://zoom.us/j/", "no meeting number"), ("123abc", "not a Zoom"), ("0", "not a Zoom meeting number"),
])
def test_meeting_references_that_are_not_zoom_meetings_are_refused(plugin, meeting, fragment):
    with pytest.raises(plugin.ToolRefusal) as refused:
        plugin.parse_meeting(meeting)
    assert refused.value.code == "ZOOM_MEETING_INVALID" and fragment in refused.value.message


def _binding(zoom_account, *, binding_id=BINDING, transport="zoom_meeting", account=None, conversation="*",
             behavior="meeting_persona", enabled=True):
    return PresenceBinding(binding_id=binding_id, transport_skill=transport, behavior_skill=behavior,
                           origin=PresenceEndpoint("zoom", account or zoom_account, conversation),
                           destination=PresenceEndpoint("zoom", account or zoom_account, MEETING), enabled=enabled)


def test_binding_selection_uses_the_argument_or_the_only_zoom_binding(plugin, zoom):
    account = zoom.companion_link.app_account_id(CLIENT_ID)
    one, other = _binding(account), _binding(account, binding_id="c" * 32, behavior="other_persona")
    foreign = _binding(account, binding_id="d" * 32, transport="telegram")
    pick = lambda bindings, requested="": plugin.select_binding(  # noqa: E731
        bindings, requested=requested, account_id=account, meeting_number=MEETING)
    assert pick([one, foreign]) is one
    assert pick([one, other], requested="c" * 32) is other
    assert pick([_binding(account, conversation=MEETING)]).origin.conversation_id == MEETING
    for bindings, requested, code in (
            ([foreign], "", "ZOOM_BINDING_MISSING"), ([one, other], "", "ZOOM_BINDING_REQUIRED"),
            ([one], "e" * 32, "ZOOM_BINDING_UNKNOWN"), ([_binding(account, enabled=False)], "", "ZOOM_BINDING_MISSING"),
            ([_binding(account, account="zoom-sdk-app-0000000000000000")], "", "ZOOM_BINDING_MISMATCH"),
            ([_binding(account, conversation="99999999999")], "", "ZOOM_BINDING_MISMATCH")):
        with pytest.raises(plugin.ToolRefusal) as refused:
            pick(bindings, requested)
        assert refused.value.code == code
    with pytest.raises(plugin.ToolRefusal) as refused:
        pick([foreign])
    assert account in refused.value.message and "call mode" in refused.value.message


# ---- JWT, the join sequence and refusals --------------------------------------------------

def test_sdk_jwt_carries_exactly_the_app_claims(zoom):
    import jwt

    token = zoom.zoom_session.build_sdk_jwt(CLIENT_ID, CLIENT_SECRET, now=1_800_000_000)
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    claims = jwt.decode(token, CLIENT_SECRET, algorithms=["HS256"], options={"verify_exp": False, "verify_iat": False})
    assert claims == {"appKey": CLIENT_ID, "iat": 1_800_000_000, "exp": 1_800_007_200, "tokenExp": 1_800_007_200}


def test_join_sequence_follows_hopes_order_when_raw_recording_is_allowed(zoom, tmp_path):
    import jwt

    fake = FakeZoom()
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake)
    reply = bot.handle_control("join", join_args(zoom))
    assert reply["ok"] and reply["state"] == "joining" and reply["display_name"] == "Aika"
    drive_to_audio(fake, loop)
    assert fake.calls == [
        "InitSDK", "CreateAuthService", "AuthService.SetEvent", "SDKAuth", "CreateMeetingService",
        "MeetingService.SetEvent", "CreateSettingService", "EnableAutoJoinAudio", "Join", "GetMeetingInfo",
        "JoinVoip", "UnMuteAudio", "GetAudioRawdataHelper", "setExternalAudioSource", "chat.SetEvent",
        "audio.SetEvent", "recording.SetEvent", "CanStartRawRecording", "StartRawRecording", "subscribe"]
    assert fake.init_param == {"strWebDomain": "https://zoom.us", "enableGenerateDump": False}
    assert jwt.decode(fake.jwt, CLIENT_SECRET, algorithms=["HS256"], options={"verify_exp": False,
                                                                               "verify_iat": False})["appKey"] == CLIENT_ID
    param = fake.join_param
    assert param["userType"] == fake.m.SDK_UT_WITHOUT_LOGIN
    assert (param["meetingNumber"], param["userName"], param["psw"], param["onBehalfToken"]) == (
        int(MEETING), "Aika", PASSCODE, OBF)
    assert (param["isVideoOff"], param["isAudioOff"], param["isMyVoiceInMix"], param["isAudioRawDataStereo"]) == (
        True, False, False, False)
    assert param["eAudioRawdataSamplingRate"] == fake.m.AudioRawdataSamplingRate_32K
    assert bot.state == "audio_ready" and bot.snapshot()["topic"] == "Weekly sync"


def test_privilege_request_waits_for_the_host_and_starts_once_on_approval(zoom, tmp_path):
    fake = FakeZoom()
    fake.codes["CanStartRawRecording"] = "SDKERR_NO_PERMISSION"
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake)
    bot.handle_control("join", join_args(zoom))
    drive_to_audio(fake, loop, raw_allowed=False)
    assert fake.calls[-2:] == ["CanStartRawRecording", "RequestLocalRecordingPrivilege"]
    assert bot.snapshot()["message"] == "in the meeting, waiting for the host to allow Aika's audio"
    loop.advance(30)
    assert "StartRawRecording" not in fake.calls
    fake.answer("Granted")
    fake.fire("recording", "onRecordPrivilegeChangedCallback", True)
    loop.advance(2)
    assert fake.calls.count("StartRawRecording") == 1 and fake.calls[-1] == "subscribe"
    assert bot.state == "audio_ready"


def test_quiet_stdout_keeps_the_bindings_join_printout_off_the_companion_stdout(zoom, tmp_path, capfd):
    fake = FakeZoom()
    fake.print_on_join = True
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake)
    bot.handle_control("join", join_args(zoom))
    loop.run_ready()
    fake.auth()
    loop.run_ready()
    assert "Join" in fake.calls
    _libc().fflush(None)
    _libc().puts(b"after the join")
    _libc().fflush(None)
    out = capfd.readouterr().out
    assert "after the join" in out and PASSCODE not in out


@pytest.mark.parametrize("path", ["leave", "ended_by_host", "removed_by_host", "join_refused", "auth_refused",
                                  "privilege_denied", "privilege_timeout", "call_refused", "call_ended",
                                  "shutdown"])
def test_every_exit_path_ends_the_call_leaves_and_cleans_up_once(zoom, tmp_path, path):
    fake = FakeZoom()
    bot, loop, peers, _bodies = make_bot(
        zoom, tmp_path, fake,
        refuse=("presence_call_not_enabled", "call mode is off for this binding") if path == "call_refused" else None)
    bot.handle_control("join", join_args(zoom))
    expect_leave, expect_peer = True, path in ("leave", "ended_by_host", "removed_by_host", "call_ended", "shutdown")
    if path == "auth_refused":
        loop.run_ready()
        fake.auth("AUTHRET_JWTTOKENWRONG")
        loop.run_ready()
        outcome, expect_leave = "Zoom refused the app credentials: AUTHRET_JWTTOKENWRONG", False
    elif path == "join_refused":
        loop.run_ready()
        fake.auth()
        loop.run_ready()
        fake.status("MEETING_STATUS_FAILED", fake.m.MEETING_FAIL_PASSWORD_ERR)
        loop.run_ready()
        outcome, expect_leave = "the meeting refused the join: MEETING_FAIL_PASSWORD_ERR", False
    elif path.startswith("privilege"):
        fake.codes["CanStartRawRecording"] = "SDKERR_NO_PERMISSION"
        drive_to_audio(fake, loop, raw_allowed=False)
        fake.answer("Denied" if path == "privilege_denied" else "Timeout")
        loop.advance(2)
        outcome = ("the host declined Aika's audio request; Aika left the meeting" if path == "privilege_denied"
                   else "the host did not answer Aika's audio request in time; Aika left the meeting")
    else:
        drive_to_audio(fake, loop)
        fake.mixed(b"\x01\x00" * 320)
        if path == "call_refused":
            loop.pump(lambda: "Leave" in fake.calls)
            loop.advance(2)
            outcome = "the call could not open: call mode is off for this binding; Aika left the meeting"
        else:
            loop.pump(lambda: bot.state == "in_call")
            if path == "leave":
                assert bot.handle_control("leave", {})["left"] is True
                outcome = "left the meeting"
            elif path in ("ended_by_host", "removed_by_host"):
                fake.end("EndMeetingReason_EndByHost" if path == "ended_by_host" else "EndMeetingReason_KickByHost")
                expect_leave = False
                outcome = ("the host ended the meeting" if path == "ended_by_host"
                           else "the host removed Aika from the meeting")
            elif path == "call_ended":
                peers[0].callbacks["on_closed"]("budget")
                outcome = "the call ended (budget); Aika left the meeting"
            else:
                bot.request_shutdown()
                assert bot.heartbeat() is False
                outcome = "the Zoom companion stopped; Aika left the meeting"
            loop.advance(2)
    assert bot.state == "idle" and bot.last_outcome["message"] == outcome
    assert fake.calls.count("CleanUPSDK") == 1 and fake.calls[-1] == "CleanUPSDK"
    assert ("Leave" in fake.calls) is expect_leave
    if expect_leave:
        assert fake.leave_cmd == fake.m.LEAVE_MEETING
        assert fake.calls.index("Leave") < fake.calls.index("CleanUPSDK")
    if "subscribe" in fake.calls and path not in ("ended_by_host", "removed_by_host"):
        assert fake.calls.index("unSubscribe") < fake.calls.index("CleanUPSDK")
    if expect_peer:
        assert peers[0].closed == outcome
    if path == "shutdown":
        assert loop.quit_called
    assert zoom.companion_link.read_status(tmp_path)["state"] == "idle"


def test_join_refusals_from_the_companion_itself(zoom, tmp_path):
    fake = FakeZoom()
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake)
    assert bot.handle_control("join", join_args(zoom, account_id="zoom-sdk-app-other"))["code"] == "ZOOM_COMPANION_STALE"
    assert bot.handle_control("join", join_args(zoom, binding_id="nope"))["code"] == "ZOOM_BINDING_INVALID"
    assert bot.handle_control("join", join_args(zoom))["ok"]
    busy = bot.handle_control("join", join_args(zoom))
    assert busy["code"] == "ZOOM_BUSY" and MEETING in busy["error"]
    bare = zoom.zoom_bot.ZoomBot(state_dir=tmp_path / "bare", loop=None, sdk=None, sdk_error="no module named zoom")
    refused = bare.handle_control("join", join_args(zoom))
    assert refused["code"] == "ZOOM_SDK_UNAVAILABLE" and "no module named zoom" in refused["error"]
    assert bare.snapshot()["message"] == "this host cannot run the Zoom Meeting SDK: no module named zoom"
    assert bot.handle_control("teleport", {})["code"] == "ZOOM_UNKNOWN_OP"


def test_a_restarted_companion_starts_idle_and_never_rejoins(zoom, tmp_path):
    zoom.companion_link.write_status(tmp_path, {"state": "in_call", "message": "in the call", "pid": -1,
                                                "meeting_number": MEETING, "join_id": "j1"})
    fake = FakeZoom()
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake)
    loop.advance(60)
    assert bot.state == "idle" and fake.calls == []
    assert "did not re-join" in bot.last_outcome["message"] and MEETING in bot.last_outcome["message"]


# ---- the call: door, peer socket, pacing ------------------------------------------------

def test_call_opens_through_a_real_door_with_the_true_rate_and_carries_audio_both_ways(zoom, tmp_path, door):
    fake = FakeZoom()
    bot, loop, _peers, _bodies = make_bot(
        zoom, tmp_path, fake, door=door.url, open_call=zoom.meeting_peer.open_call,
        attach_peer=zoom.meeting_peer.PeerSession.attach, identity_name=zoom.meeting_peer.identity_name,
        pacer_factory=lambda send, rate: zoom.meeting_peer.OutboundPacer(send, rate=rate))
    assert bot.handle_control("join", join_args(zoom))["display_name"] == "Aika"  # from the door's /identity
    drive_to_audio(fake, loop)
    open_call_via_audio(fake, loop, bot)
    body, headers = door.bodies[0], door.headers[0]
    assert headers["x-skill-token"] == HOST_TOKEN
    assert set(body) == {"binding_id", "role", "event", "label", "mic", "speaker"}
    assert (body["binding_id"], body["role"], body["mic"], body["speaker"]) == (BINDING, "meeting", True, True)
    event = body["event"]
    assert set(event) == {"source_event_id", "provider", "account_id", "conversation_id", "thread_id",
                          "conversation_key", "actor", "conversation", "message", "text"}
    account = zoom.companion_link.app_account_id(CLIENT_ID)
    assert (event["provider"], event["account_id"], event["conversation_id"], event["thread_id"]) == (
        "zoom", account, MEETING, "")
    assert event["conversation_key"] == f"zoom:{account}:{MEETING}:0"
    assert event["source_event_id"] == f"zoom-join-{bot.snapshot()['join_id']}"
    assert event["actor"]["id"] == "task:t-1" and event["text"] == "Joined the Zoom meeting Weekly sync"
    assert door.hello == {"type": "hello", "protocol": 1, "role": "meeting", "mic": True, "speaker": True,
                          "rate": 32000, "label": "Zoom meeting Weekly sync"}
    assert bot.snapshot()["call_id"] == "call-1"

    fake.mixed(b"\x02\x00" * 320)
    fake.mixed(b"\x03\x00" * 320)
    loop.pump(lambda: len(door.frames) >= 2)
    up = [cp.decode_frame(frame) for frame in door.frames[:2]]
    assert [(f.kind, f.rate, f.pcm) for f in up] == [(cp.FRAME_AUDIO_IN, 32000, b"\x02\x00" * 320),
                                                    (cp.FRAME_AUDIO_IN, 32000, b"\x03\x00" * 320)]
    assert up[1].seq == up[0].seq + 1

    door.send(cp.encode_frame(cp.AudioFrame(cp.FRAME_AUDIO_OUT, 0, 1, 32000, 0, b"\x04\x00" * 1600)))
    loop.pump(lambda: len(fake.sent) >= 3)
    assert [(len(data), rate, channel) for data, rate, channel in fake.sent] == [
        (1280, 32000, fake.m.ZoomSDKAudioChannel_Mono)] * 3
    assert fake.sent[2][0] == b"\x04\x00" * 320 + b"\x00" * 640  # the utterance tail, padded to one frame

    door.send(cp.encode_frame(cp.AudioFrame(cp.FRAME_AUDIO_OUT, 0, 2, 32000, 0, b"\x05\x00" * 30000)))
    door.send(cp.encode_control({"type": "flush", "epoch": 1}))
    loop.pump(lambda: bot.snapshot()["stats"].get("voice_bytes_flushed", 0) > 0)
    time.sleep(0.2)
    assert len(fake.sent) < 3 + 20  # far fewer than the 47 frames that were queued

    fake.one_way(42)
    loop.advance(0.2)
    loop.pump(lambda: {"type": "participant", "label": "Anna", "speaking": True} in door.controls)
    loop.advance(1.0)
    loop.pump(lambda: {"type": "participant", "label": "Anna", "speaking": False} in door.controls)
    fake.chat(7, "the numbers are on slide 4")
    fake.host_mute()
    loop.pump(lambda: len([c for c in door.controls if c["type"] == "context"]) == 2)
    chat, mute = [c for c in door.controls if c["type"] == "context"]
    assert chat == {"type": "context", "source": "zoom_chat",
                    "text": "Zoom chat from Boris (to everyone): the numbers are on slide 4"}
    assert mute["source"] == "zoom" and "muted you" in mute["text"]

    bot.handle_control("leave", {})
    loop.pump(lambda: door.closed)
    assert door.controls[-1] == {"type": "bye", "reason": "left the meeting"}
    loop.advance(2)
    assert bot.state == "idle" and fake.calls[-3:] == ["Leave", "unSubscribe", "CleanUPSDK"]


def test_a_zoom_reconnect_while_the_call_opens_keeps_the_call(zoom, tmp_path):
    fake = FakeZoom()
    release, asked = threading.Event(), []

    def slow_open(url, token, body):
        asked.append(body)
        assert release.wait(5)
        return {"call_id": "call-7", "peer_path": "/call/call-7/peer"}

    bot, loop, peers, _bodies = make_bot(zoom, tmp_path, fake, open_call=slow_open)
    bot.handle_control("join", join_args(zoom))
    drive_to_audio(fake, loop)
    fake.mixed(b"\x01\x00" * 320)  # the first meeting audio opens the call
    loop.pump(lambda: bool(asked))
    fake.status("MEETING_STATUS_RECONNECTING")
    fake.status("MEETING_STATUS_RECONNECTING")  # a repeated notice keeps the state to return to
    loop.run_ready()
    assert bot.state == "reconnecting"
    release.set()
    loop.pump(lambda: bot.snapshot()["call_id"] == "call-7")
    assert bot.state == "reconnecting" and peers[0].closed is None
    fake.status("MEETING_STATUS_INMEETING")  # Zoom's reconnect finished
    loop.run_ready()
    assert bot.state == "in_call" and peers[0].closed is None and len(asked) == 1
    fake.mixed(b"\x02\x00" * 320)
    assert peers[0].audio == [(b"\x02\x00" * 320, 32000)]


def test_hello_and_voice_pacing_follow_the_rate_zoom_actually_delivers(zoom, tmp_path):
    fake = FakeZoom()
    pacers = []
    bot, loop, peers, _bodies = make_bot(zoom, tmp_path, fake,
                                         pacer_factory=lambda send, rate: pacers.append(FakePacer(send, rate))
                                         or pacers[-1])
    bot.handle_control("join", join_args(zoom))
    drive_to_audio(fake, loop)
    fake.mixed(b"\x01\x00" * 480, rate=48000)
    loop.pump(lambda: bot.state == "in_call")
    assert peers[0].hello["rate"] == 48000 and pacers[0].rate == 48000
    fake.mixed(b"\x02\x00" * 480, rate=48000)
    assert peers[0].audio == [(b"\x02\x00" * 480, 48000)]


def test_server_hang_up_ends_the_meeting_and_a_refusal_never_attaches(zoom, tmp_path, door):
    fake = FakeZoom()
    bot, loop, _peers, _bodies = make_bot(zoom, tmp_path, fake, door=door.url, open_call=zoom.meeting_peer.open_call,
                                          attach_peer=zoom.meeting_peer.PeerSession.attach)
    bot.handle_control("join", join_args(zoom))
    drive_to_audio(fake, loop)
    open_call_via_audio(fake, loop, bot)
    door.send(cp.encode_control({"type": "state", "state": "closed", "reason": "owner stop"}))
    door.close()
    loop.pump(lambda: "Leave" in fake.calls)
    loop.advance(2)
    assert bot.last_outcome["message"] == "the call ended (owner stop); Aika left the meeting"

    door.refuse = ("presence_call_busy", "2 Presence turns are active")
    bot.handle_control("join", join_args(zoom))
    drive_to_audio(fake, loop)
    fake.mixed(b"\x01\x00" * 320)
    loop.pump(lambda: bot.state == "idle")
    assert bot.last_outcome["message"] == ("the call could not open: 2 Presence turns are active; "
                                           "Aika left the meeting")
    assert door.peers == 1


def test_pacer_sends_exact_20ms_frames_in_real_time_and_pads_only_the_tail(zoom):
    now, sent = [0.0], []
    stop = threading.Event()

    def sleep(seconds):
        now[0] += seconds
        if len(sent) == 3:
            stop.set()

    pacer = zoom.meeting_peer.OutboundPacer(lambda frame: sent.append((round(now[0], 6), frame)), rate=32000,
                                            clock=lambda: now[0], sleep=sleep)
    assert (pacer.frame_bytes, pacer.frame_sec) == (1280, 0.02)
    speech = bytes(range(256)) * 11 + bytes(184)  # 3,000 bytes: two full frames and a 440-byte tail
    pacer.feed(speech, 0)
    pacer.run(stop)
    assert [t for t, _frame in sent] == [0.0, 0.02, 0.04]
    assert [frame for _t, frame in sent] == [speech[:1280], speech[1280:2560], speech[2560:] + b"\x00" * 840]


def test_pacer_flush_drops_queued_voice_and_stale_epochs(zoom):
    peer_mod = zoom.meeting_peer
    pacer = peer_mod.OutboundPacer(lambda frame: None, rate=32000)
    pacer.feed(b"\x01\x00" * 6400, 0)
    assert len(pacer.take_frame()) == 1280
    assert pacer.flush(1) == 11520 and pacer.take_frame() is None
    pacer.feed(b"\x02\x00" * 640, 0)
    assert pacer.take_frame() is None and pacer.stale_frames == 1
    pacer.feed(b"\x03\x00" * 640, 1)
    assert pacer.flush(0) == 0  # an older epoch's late flush changes nothing
    assert pacer.take_frame() == b"\x03\x00" * 640
    pacer.feed(b"\x04\x00" * 700, 1)
    pacer.feed(b"\x05\x00" * 640, 2)  # a newer epoch supersedes what was queued
    assert pacer.take_frame() == b"\x05\x00" * 640
    assert peer_mod.epoch_is_newer(0, 0xFFFF) and not peer_mod.epoch_is_newer(0xFFFF, 0)
    assert peer_mod.OutboundPacer(lambda frame: None, rate=16000).frame_bytes == 640


def test_a_failing_microphone_send_is_logged_once_per_streak_with_its_count(zoom, caplog):
    now, attempts, stop = [0.0], [], threading.Event()
    outcomes = [False, False, False, True, False, False]  # three fail, one goes through, two fail until stop

    def send(frame):
        attempts.append(frame)
        if len(attempts) == len(outcomes):
            stop.set()
        if not outcomes[len(attempts) - 1]:
            raise OSError("the SDK refused the audio frame")

    pacer = zoom.meeting_peer.OutboundPacer(send, rate=32000, clock=lambda: now[0],
                                            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    pacer.feed(b"\x01\x00" * 640 * len(outcomes), 0)  # one exact 20 ms frame per attempt
    with caplog.at_level(logging.WARNING, logger="zoom_bot.peer"):
        pacer.run(stop)
    assert [(record.levelname, record.getMessage(), record.exc_info is not None)
            for record in caplog.records if record.name == "zoom_bot.peer"] == [
        ("ERROR", "virtual microphone send failed", True),
        ("WARNING", "virtual microphone send recovered after 3 failed frames", False),
        ("ERROR", "virtual microphone send failed", True),
        ("WARNING", "virtual microphone send still failing when the voice stopped: 2 failed frames", False)]
    assert len(attempts) == 6 and pacer.sent_frames == 1


def test_meeting_audio_keeps_reaching_the_real_hub_across_interruptions(zoom, tmp_path):
    """The real Host door and hub: Zoom attaches after the call's epoch moved, and its
    microphone still reaches the call after each interruption, the 16-bit wrap included."""
    from ouroboros.call_peers import close_hub, open_hub
    from ouroboros.gateway.host_service import create_host_service_app
    from tests.test_call_gateway import seed_skill

    seed_skill(tmp_path, name="zoom_meeting", token=HOST_TOKEN)
    hub = open_hub(cp.CallSpec(uuid.uuid4().hex, 1, "openai", "test-model"))
    while hub.epoch != 0xFFFF:  # interrupted before Zoom attached; the next flush wraps to 0
        hub.flush()
    url, stop = serve_app(create_host_service_app(tmp_path), "zoom-real-door")
    flushes, closed, peer = [], [], None

    def wait_for(condition, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not condition():
            assert time.monotonic() < deadline, "condition not reached"
            time.sleep(0.01)

    def heard(timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if isinstance(hub.next_event(0.05), cp.PeerAudio):
                return True
        return False

    try:
        opened = zoom.meeting_peer.open_call(url, HOST_TOKEN, {"call_id": hub.call_id, "role": "meeting",
                                                               "mic": True, "speaker": True})
        hello = {"type": "hello", "protocol": 1, "role": "meeting", "mic": True, "speaker": True,
                 "rate": 32000, "label": "Zoom meeting Weekly sync"}
        peer = zoom.meeting_peer.PeerSession.attach(url, HOST_TOKEN, opened["peer_path"], hello,
                                                    on_audio=lambda pcm, epoch: None, on_flush=flushes.append,
                                                    on_closed=closed.append)
        wait_for(lambda: flushes == [0xFFFF])
        peer.send_audio(b"\x01\x00" * 320, 32000)
        assert heard()
        for expected in (0, 1):  # a barge-in flushes the call; the wrap first
            assert hub.flush() == expected
            wait_for(lambda: flushes[-1:] == [expected])
            peer.send_audio(b"\x02\x00" * 320, 32000)
            assert heard()
        assert hub.stats()["stale_frames"] == 0 and closed == []
    finally:
        if peer is not None:
            peer.close("left the meeting")
            peer.join()
        close_hub(hub.call_id)
        stop()


def test_a_flush_drops_uplink_of_the_old_epoch_and_meeting_facts_stay_bounded(zoom):
    peer_mod = zoom.meeting_peer
    flushes = []
    peer = peer_mod.PeerSession(None, rate=32000, on_audio=lambda *_: None, on_flush=flushes.append,
                                on_closed=lambda *_: None)
    peer.send_audio(b"\x01\x00" * 320, 32000)
    peer.send_audio(b"\x01\x00" * 320, 32000)
    assert [cp.decode_frame(frame).epoch for frame in peer._audio] == [0, 0]
    peer._handle(cp.encode_control({"type": "flush", "epoch": 0}))  # a fresh call's attach flush
    assert len(peer._audio) == 2 and peer.stale_uplink == 0
    peer._handle(cp.encode_control({"type": "flush", "epoch": 7}))
    assert not peer._audio and peer.stale_uplink == 2 and flushes == [0, 7]
    peer.send_audio(b"\x02\x00" * 320, 32000)
    frame = cp.decode_frame(peer._audio[-1])
    assert (frame.epoch, frame.seq) == (7, 3)
    peer._handle(json.dumps({"type": "flush", "epoch": "8"}))  # not the hub's shape: no epoch change
    assert flushes == [0, 7] and cp.decode_frame(peer._audio[-1]).epoch == 7
    for index in range(peer_mod.MAX_QUEUED_CONTROLS + 5):
        peer.send_control({"type": "context", "text": f"fact {index}", "source": "zoom_chat"})
    assert len(peer._controls) == peer_mod.MAX_QUEUED_CONTROLS and peer.dropped_controls == 5
    assert json.loads(peer._controls[0])["text"] == "fact 5"


def test_frames_match_the_frozen_core_codec_byte_for_byte(zoom):
    peer_mod = zoom.meeting_peer
    for kind, epoch, seq, rate, stream, pcm in ((cp.FRAME_AUDIO_IN, 0, 1, 32000, 0, b"\x01\x02" * 320),
                                                (cp.FRAME_AUDIO_OUT, 0x1FFFF, 0x1FFFFFFFF, 32000, 3, b""),
                                                (cp.FRAME_AUDIO_OUT, 7, 9, 48000, 0, b"\x00" * 65536)):
        raw = peer_mod.encode_frame(kind, epoch, seq, rate, stream, pcm)
        assert raw == cp.encode_frame(cp.AudioFrame(kind, epoch, seq, rate, stream, pcm))
        decoded = cp.decode_frame(raw)
        assert peer_mod.decode_frame(raw) == (decoded.kind, decoded.epoch, decoded.seq, decoded.rate,
                                              decoded.stream, decoded.pcm)
    assert peer_mod.HEADER_SIZE == cp.HEADER_SIZE and peer_mod.MAX_CONTROL_BYTES == cp.MAX_CONTROL_BYTES
    for bad in (b"XX" + raw[2:], raw[:10], raw + b"\x00"):
        with pytest.raises(ValueError):
            peer_mod.decode_frame(bad)


def test_control_messages_fit_the_protocol_limit_with_a_named_omission(zoom):
    peer_mod = zoom.meeting_peer
    fitted = peer_mod.fit_control({"type": "context", "text": "é" * 20000, "source": "zoom_chat"})
    assert len(fitted.encode("utf-8")) <= cp.MAX_CONTROL_BYTES
    message = cp.decode_control(fitted)
    assert message["source"] == "zoom_chat" and message["text"].endswith("more characters omitted]")
    small = {"type": "participant", "label": "Anna", "speaking": True}
    assert json.loads(peer_mod.fit_control(small)) == small
    assert peer_mod.fit_control({"type": "participant", "label": "x" * 20000, "speaking": True}) is None


# ---- the control channel and the tools -------------------------------------------------

@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_control_channel_needs_the_token_and_reports_a_missing_companion(zoom, tmp_path, monkeypatch, transport):
    link = zoom.companion_link
    if transport == "tcp":
        monkeypatch.setattr(link, "UNIX_PATH_LIMIT", 0)
    elif len(os.fsencode(str(tmp_path / link.SOCKET_FILE))) >= link.UNIX_PATH_LIMIT:
        pytest.skip("this temp path is too long for an AF_UNIX socket; the tcp case covers the channel")
    with pytest.raises(link.ControlUnavailable):
        link.control_request(tmp_path, "status")
    seen = []
    server = link.ControlServer(tmp_path, lambda op, args: seen.append((op, args)) or {"ok": True, "op": op})
    server.start()
    try:
        assert server.address["transport"] == transport
        assert oct((tmp_path / link.CONTROL_FILE).stat().st_mode & 0o777) == "0o600"
        assert link.control_request(tmp_path, "status", {"x": 1}) == {"ok": True, "op": "status"}
        info = json.loads((tmp_path / link.CONTROL_FILE).read_text())
        if transport == "tcp":
            conn = socket.create_connection(("127.0.0.1", info["port"]), timeout=5)
        else:
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(info["path"])
        with conn:
            conn.sendall(json.dumps({"token": "guess", "op": "leave"}).encode() + b"\n")
            assert json.loads(conn.makefile().readline())["code"] == "ZOOM_CONTROL_DENIED"
        assert seen == [("status", {"x": 1})]
    finally:
        server.close()
    assert not (tmp_path / link.CONTROL_FILE).exists()
    with pytest.raises(link.ControlUnavailable):
        link.control_request(tmp_path, "status")


@pytest.fixture
def tools(plugin, zoom, tmp_path, monkeypatch):
    state_dir, data_dir = tmp_path / "state", tmp_path / "data"
    state_dir.mkdir()
    (data_dir / "state").mkdir(parents=True)
    account = zoom.companion_link.app_account_id(CLIENT_ID)
    save_presence_binding(data_dir, _binding(account))
    api = FakeApi(state_dir, data_dir, {"ZOOM_CLIENT_ID": CLIENT_ID, "ZOOM_CLIENT_SECRET": CLIENT_SECRET})
    plugin.register(api)
    companion = FakeCompanion(account)
    server = zoom.companion_link.ControlServer(state_dir, companion)
    server.start()
    monkeypatch.setattr(plugin, "_sleep", lambda seconds: None)
    ctx = types.SimpleNamespace(task_id="t-1", task_metadata={"source": "web"}, current_task_type="task",
                                drive_root=data_dir)
    call = lambda name, **kwargs: api.tools[name].handler(ctx, **kwargs)  # noqa: E731
    yield types.SimpleNamespace(api=api, call=call, companion=companion, server=server, ctx=ctx, account=account,
                                data_dir=data_dir, state_dir=state_dir)
    server.close()


def test_plugin_registers_the_companion_and_three_tools(tools):
    assert tools.api.companions == ["zoom_bot"]
    assert sorted(tools.api.tools) == ["zoom_join", "zoom_leave", "zoom_status"]
    schema = tools.api.tools["zoom_join"].schema
    assert schema["required"] == ["meeting"]
    assert set(schema["properties"]) == {"meeting", "passcode", "display_name", "binding_id", "on_behalf_token"}
    assert "OBF" in schema["properties"]["on_behalf_token"]["description"]


def test_zoom_join_hands_over_the_join_and_never_echoes_a_secret(tools):
    result = tools.call("zoom_join", meeting=f"https://us02web.zoom.us/j/{MEETING}?pwd={PASSCODE}",
                        on_behalf_token=OBF)
    assert result.splitlines()[0] == "in the meeting, waiting for the host to allow Aika's audio"
    op, args = tools.companion.requests[-2]
    assert op == "join" and tools.companion.requests[-1][0] == "status"
    assert (args["meeting_number"], args["passcode"], args["on_behalf_token"], args["binding_id"]) == (
        MEETING, PASSCODE, OBF, BINDING)
    assert args["account_id"] == tools.account
    assert args["requested_by"] == {"id": "task:t-1", "display_name": "", "requested_via": "web", "task_id": "t-1"}
    for secret in (PASSCODE, OBF, CLIENT_SECRET, CLIENT_ID):
        assert secret not in result
    tools.companion.state = "idle"
    tools.ctx.task_metadata = {"presence": {"event": {"provider": "telegram", "actor": {
        "platform_actor_id": "u77", "display_name": "Lena"}}}}
    tools.call("zoom_join", meeting=MEETING, passcode="explicit-1")
    args = tools.companion.requests[-2][1]
    assert args["passcode"] == "explicit-1"
    assert args["requested_by"]["id"] == "telegram:u77" and args["requested_by"]["display_name"] == "Lena"
    assert tools.call("zoom_status").splitlines()[0] == "in the meeting, waiting for the host to allow Aika's audio"
    assert tools.call("zoom_leave").splitlines()[0] == "not in a Zoom meeting"


def test_zoom_join_refusals_are_typed_and_say_what_to_do(tools, plugin, zoom):
    assert tools.call("zoom_join", meeting="https://zoom.us/my/anton").startswith("⚠️ ZOOM_MEETING_INVALID:")
    tools.api.settings = {"ZOOM_CLIENT_ID": CLIENT_ID}
    assert tools.call("zoom_join", meeting=MEETING) == (
        "⚠️ ZOOM_CREDENTIALS_MISSING: Zoom app credentials are not set: save the Meeting SDK app's ZOOM_CLIENT_ID "
        "and ZOOM_CLIENT_SECRET in Settings → Secrets and grant them to zoom_meeting")
    tools.api.settings = {"ZOOM_CLIENT_ID": CLIENT_ID, "ZOOM_CLIENT_SECRET": CLIENT_SECRET}
    save_presence_binding(tools.data_dir, _binding(tools.account, binding_id="c" * 32, behavior="other_persona"))
    assert tools.call("zoom_join", meeting=MEETING).startswith("⚠️ ZOOM_BINDING_REQUIRED:")
    tools.companion.sdk = False
    refused = tools.call("zoom_join", meeting=MEETING, binding_id=BINDING)
    assert refused.startswith("⚠️ ZOOM_SDK_UNAVAILABLE:") and "not importable" in refused
    tools.companion.sdk, tools.companion.account = True, "zoom-sdk-app-stale"
    assert tools.call("zoom_join", meeting=MEETING, binding_id=BINDING).startswith("⚠️ ZOOM_COMPANION_STALE:")
    tools.companion.account, tools.companion.state = tools.account, "waiting_for_audio"
    busy = tools.call("zoom_join", meeting=MEETING, binding_id=BINDING)
    assert busy.startswith("⚠️ ZOOM_BUSY:") and "zoom_leave" in busy
    tools.companion.state, tools.companion.after_join = "idle", "idle"
    tools.companion.join_outcome = {"message": "the meeting refused the join: MEETING_FAIL_PASSWORD_ERR",
                                    "join_id": "j-9", "at": "now"}
    assert tools.call("zoom_join", meeting=MEETING, binding_id=BINDING) == (
        "⚠️ ZOOM_JOIN_FAILED: the meeting refused the join: MEETING_FAIL_PASSWORD_ERR")
    tools.server.close()
    zoom.companion_link.write_status(tools.state_dir, {"state": "in_call", "message": "in the call",
                                                       "meeting_number": MEETING})
    down = tools.call("zoom_status")
    assert down.startswith("⚠️ ZOOM_COMPANION_UNAVAILABLE:") and "Last published state: in the call" in down
    assert all(op != "join" or args["binding_id"] == BINDING for op, args in tools.companion.requests)


# ---- secrets, the probe and the GLib loop ---------------------------------------------------

def test_no_secret_reaches_a_log_line_the_status_file_or_a_reply(zoom, tmp_path):
    fake = FakeZoom()
    redactor = zoom.zoom_bot.SecretRedactor()
    for value in (CLIENT_SECRET, HOST_TOKEN):
        redactor.add(value)
    stream = __import__("io").StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(redactor)
    logger = logging.getLogger("zoom_bot")
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        bot, loop, peers, bodies = make_bot(zoom, tmp_path, fake, register_secret=redactor.add)
        replies = [bot.handle_control("join", join_args(zoom))]
        drive_to_audio(fake, loop)
        open_call_via_audio(fake, loop, bot)
        replies += [bot.handle_control("status", {}), bot.handle_control("leave", {})]
        loop.advance(2)
        logging.getLogger("zoom_bot.session").error("unexpected: %s", fake.jwt)
        try:
            raise RuntimeError(f"wrapped {PASSCODE} and {OBF}")
        except RuntimeError:
            logging.getLogger("zoom_bot.peer").exception("traceback carries %s", CLIENT_SECRET)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    written = stream.getvalue() + (tmp_path / "status.json").read_text() + json.dumps(replies) + json.dumps(
        bodies) + json.dumps(peers[0].hello)
    assert "[redacted]" in stream.getvalue()
    for secret in (CLIENT_SECRET, PASSCODE, OBF, HOST_TOKEN, fake.jwt):
        assert secret and secret not in written


def test_probe_reports_pass_fail_and_not_run_without_printing_a_secret(zoom, capsys, monkeypatch, tmp_path):
    probe = zoom.probe
    run_probe = probe.run_probe
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "zoomus.conf").write_text("system.audio.type=default\n", encoding="utf-8")
    (tmp_path / "run" / "pulse").mkdir(parents=True)
    (tmp_path / "run" / "pulse" / "native").write_text("", encoding="utf-8")
    env = {"ZOOM_CLIENT_ID": CLIENT_ID, "ZOOM_CLIENT_SECRET": CLIENT_SECRET, "HOME": str(tmp_path),
           "XDG_RUNTIME_DIR": str(tmp_path / "run")}

    def run(fake, **kwargs):
        loop = ManualLoop()
        if fake is not None and fake.on_sdkauth is None:
            fake.on_sdkauth = lambda: loop.idle_add(lambda: fake.auth())
        options = dict(env=env, import_sdk=lambda: (fake.m, "zoom-meeting-sdk 0.0.27") if fake else (None, "ImportError"),
                       make_loop=lambda: loop, platform_name=lambda: "linux-x86_64", clock=lambda: 1_800_000_000)
        options.update(kwargs)
        return {step["step"]: (step["status"], step["detail"]) for step in run_probe(**options)}

    fake = FakeZoom()
    steps = run(fake)
    assert {name: status for name, (status, _detail) in steps.items()} == dict.fromkeys(probe.STEPS, "PASS")
    assert fake.calls == ["InitSDK", "CreateAuthService", "AuthService.SetEvent", "SDKAuth", "CleanUPSDK"]
    assert "Join" not in fake.calls and zoom.companion_link.app_account_id(CLIENT_ID) in steps["credentials"][1]
    missing = run(FakeZoom(), env={"HOME": str(tmp_path / "nowhere")}, platform_name=lambda: "darwin-arm64",
                  uid=1001, user_runtime_root=tmp_path / "run-user")
    assert missing["platform"][0] == "FAIL" and missing["credentials"][0] == "FAIL"
    assert missing["audio_raw"] == ("FAIL", (
        f"{tmp_path / 'nowhere' / '.config' / 'zoomus.conf'} does not exist (it needs system.audio.type=default); "
        f"PULSE_SERVER and XDG_RUNTIME_DIR are not set, and {tmp_path / 'run-user' / '1001'} does not exist "
        "(this user has no login session or lingering user manager)"))
    assert (missing["init_sdk"][0], missing["authenticate"][0], missing["cleanup"][0]) == ("PASS", "NOT_RUN", "PASS")
    absent = run(None)
    assert absent["import_sdk"][0] == "FAIL" and {absent[s][0] for s in ("glib", "init_sdk", "authenticate",
                                                                         "cleanup")} == {"NOT_RUN"}
    refused = FakeZoom()
    refused.on_sdkauth = lambda: None
    timed_out = run(refused, auth_timeout_ms=30000)
    assert timed_out["authenticate"] == ("FAIL", "no answer from Zoom within 30 s")
    wrong = FakeZoom()
    loop_box = {}
    wrong.on_sdkauth = lambda: loop_box["loop"].idle_add(lambda: wrong.auth("AUTHRET_KEYORSECRETWRONG"))
    steps = run(wrong, make_loop=lambda: loop_box.setdefault("loop", ManualLoop()))
    assert steps["authenticate"] == ("FAIL", "AUTHRET_KEYORSECRETWRONG")

    monkeypatch.setattr(probe, "run_probe", lambda **kwargs: [
        {"step": s, "status": "PASS", "detail": d} for s, (_st, d) in run(FakeZoom()).items()])
    assert probe.main(["--json"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["result"] == "PASS"
    assert CLIENT_SECRET not in printed and CLIENT_ID not in printed


def test_raw_audio_names_each_missing_piece_and_adopts_the_user_runtime_dir(zoom, tmp_path):
    raw = zoom.raw_audio
    home, runtime_root = tmp_path / "home", tmp_path / "run" / "user"
    conf, socket_path = home / ".config" / "zoomus.conf", runtime_root / "1001" / "pulse" / "native"
    env = {}
    assert raw.check(env, home=str(home), uid=1001, user_runtime_root=runtime_root) == ("FAIL", (
        f"{conf} does not exist (it needs system.audio.type=default); PULSE_SERVER and XDG_RUNTIME_DIR are not "
        f"set, and {runtime_root / '1001'} does not exist (this user has no login session or lingering user manager)"))
    assert env == {}
    conf.parent.mkdir(parents=True)
    conf.write_text("system.audio.type=pulse\n", encoding="utf-8")
    (runtime_root / "1001").mkdir(parents=True)
    # A lingering user manager's runtime dir is adopted for the SDK, and its missing socket named.
    assert raw.check(env, home=str(home), uid=1001, user_runtime_root=runtime_root) == ("FAIL", (
        f"{conf} does not set system.audio.type=default; PULSE_SERVER is not set and {socket_path} does not exist "
        "(no PulseAudio or PipeWire pulse server runs for this user)"))
    assert env == {"XDG_RUNTIME_DIR": str(runtime_root / "1001")}
    conf.write_text("[General]\nsystem.audio.type=default\n", encoding="utf-8")
    socket_path.parent.mkdir()
    socket_path.write_text("", encoding="utf-8")
    assert raw.check(env, home=str(home), uid=1001, user_runtime_root=runtime_root) == (
        "PASS", f"{conf} sets system.audio.type=default; PulseAudio server: {socket_path}")
    # PULSE_SERVER wins, as in libpulse; an XDG_RUNTIME_DIR the environment set is never replaced.
    named = {"PULSE_SERVER": "unix:/srv/aika/pulse/native", "XDG_RUNTIME_DIR": str(tmp_path / "elsewhere")}
    assert raw.check(named, home=str(home), uid=1001, user_runtime_root=runtime_root)[1].endswith(
        "PulseAudio server: PULSE_SERVER (unix:/srv/aika/pulse/native)")
    explicit = {"XDG_RUNTIME_DIR": str(tmp_path / "elsewhere")}
    assert raw.check(explicit, home=str(home), uid=1001, user_runtime_root=runtime_root) == ("FAIL", (
        f"PULSE_SERVER is not set and {tmp_path / 'elsewhere' / 'pulse' / 'native'} does not exist "
        "(no PulseAudio or PipeWire pulse server runs for this user)"))
    assert explicit == {"XDG_RUNTIME_DIR": str(tmp_path / "elsewhere")}


def test_the_companion_publishes_its_own_audio_check_and_the_tools_show_a_failure(zoom, plugin, tmp_path):
    fake = FakeZoom()
    detail = "PULSE_SERVER and XDG_RUNTIME_DIR are not set, and /run/user/1001 does not exist"
    bot, _loop, _peers, _bodies = make_bot(zoom, tmp_path, fake, audio={"status": "FAIL", "detail": detail})
    status = zoom.companion_link.read_status(tmp_path)
    assert status["audio"] == bot.snapshot()["audio"] == {"status": "FAIL", "detail": detail}
    assert plugin.render_status(status).splitlines()[-1] == (
        f"meeting audio may not arrive here: {detail} (SKILL.md, Setup, says how to provide it)")
    assert "meeting audio" not in plugin.render_status({**status, "audio": {"status": "PASS", "detail": "ok"}})


def test_off_linux_the_companion_stays_idle_and_refuses_typed_without_a_posix_lock(zoom, tmp_path, monkeypatch):
    bot_mod = zoom.zoom_bot
    assert bot_mod.platform_refusal("linux") == ""
    reason = bot_mod.platform_refusal("win32")
    assert reason == "the Zoom Meeting SDK runs on Linux x86_64 only; this host is win32"
    monkeypatch.setattr(bot_mod, "platform_refusal", lambda platform=None: reason)
    monkeypatch.setattr(bot_mod, "acquire_singleton", lambda *a, **k: pytest.fail("a POSIX lock off Linux"))
    monkeypatch.setattr(bot_mod, "load_sdk", lambda: pytest.fail("an SDK import off Linux"))
    monkeypatch.setattr(bot_mod.raw_audio, "check", lambda *a, **k: pytest.fail("an audio check off Linux"))
    monkeypatch.setattr(bot_mod, "watch_parent", lambda on_orphaned: None)
    monkeypatch.setattr(bot_mod.signal, "signal", lambda *args: None)
    monkeypatch.setattr(bot_mod.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(tmp_path))
    replies = []
    monkeypatch.setattr(bot_mod.ZoomBot, "wait_for_shutdown", lambda bot: replies.append(
        zoom.companion_link.control_request(tmp_path, "join", join_args(zoom))))
    assert bot_mod.main() == 0
    assert (replies[0]["ok"], replies[0]["code"], replies[0]["error"]) == (False, "ZOOM_SDK_UNAVAILABLE", reason)
    status = zoom.companion_link.read_status(tmp_path)
    assert status["state"] == "idle" and status["sdk"] == {"available": False, "detail": reason, "loop": ""}


def test_ctypes_glib_loop_runs_callbacks_on_its_own_thread_and_releases_them(zoom):
    glib = zoom.glib_loop
    library = glib.find_libglib()
    if library is None:
        pytest.skip("libglib-2.0 is not installed on this host; the ctypes loop is exercised where it is")
    loop = glib.make_loop(prefer_gi=False)
    assert isinstance(loop, glib.CtypesGLibLoop) and loop.backend == "ctypes"
    here, seen, ticks = threading.get_ident(), [], []

    def from_other_thread():
        worker = threading.Thread(target=lambda: loop.idle_add(lambda: seen.append(("cross", threading.get_ident()))))
        worker.start()
        worker.join()
        return False

    def boom():
        raise RuntimeError("a failing callback is removed, the loop goes on")

    loop.idle_add(lambda: seen.append(("idle", threading.get_ident())))
    loop.idle_add(boom)
    loop.timeout_add(5, lambda: ticks.append(1) or len(ticks) < 3)
    loop.timeout_add(1, from_other_thread)
    loop.timeout_add(150, lambda: loop.quit())
    watchdog = threading.Timer(5, loop.quit)
    watchdog.start()
    try:
        loop.run()
    finally:
        watchdog.cancel()
    assert ticks == [1, 1, 1] and ("idle", here) in seen and ("cross", here) in seen
    assert loop.pending() == 0 and not loop.is_running()
    with pytest.raises(glib.GLibUnavailable):
        glib.make_loop(prefer_gi=False, library="/nonexistent/libglib-2.0.so.0")


def test_find_libglib_reports_absence(zoom, monkeypatch):
    monkeypatch.setattr(zoom.glib_loop.ctypes.util, "find_library", lambda name: None)
    assert zoom.glib_loop.find_libglib(candidates=["/nonexistent/libglib-2.0.so.0"]) is None


# ---- the fake Host door -----------------------------------------------------------------------

@pytest.fixture
def door():
    state, stop = start_fake_door()
    try:
        yield state
    finally:
        stop()
