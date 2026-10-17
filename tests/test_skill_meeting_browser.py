"""The experimental meeting_browser skill: Ouroboros's own browser window as a participant.

Playwright, Chrome and the call door are faked for the companion tests; the page
script is exercised for real twice: under Node with fake Web Audio objects (its
PCM arithmetic in both the AudioWorklet and the ScriptProcessor paths), and in
bundled headless Chromium (``ui_browser``) on a local page with two
RTCPeerConnections, proving the replaced microphone carries fed audio across
WebRTC and the remote tap returns the other side's audio. No real meeting,
account or camera is used.
"""

from __future__ import annotations

import base64
import http.server
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import threading
from array import array
from types import SimpleNamespace

import pytest

from tests.test_skill_meeting_audio import (
    FakeAPI,
    FakeDoor,
    RacingPeer,
    out_frame,
    point_lifeline,
    serve_fake_door,
    until,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "meeting_browser"
SCRIPTS = SKILL_DIR / "scripts"
PAGE_SCRIPT = SCRIPTS / "participant_page.js"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import meeting_participant  # noqa: E402
import participant_browser  # noqa: E402
import probe_vendor  # noqa: E402
import vendors  # noqa: E402


def tool_payload(text):
    """A tool reply's JSON; a refusal leads with its typed ``⚠️ CODE: message`` line."""
    if text.startswith("⚠️ "):
        line, _, body = text.partition("\n")
        payload = json.loads(body)
        assert line == f"⚠️ {payload['code']}: {payload['message']}"
        return payload
    return json.loads(text)

CALL_META = {"live_call": {"call_id": "call-1", "chat_id": 7, "engine": "openai", "model": "m"}}
MEET = "https://meet.google.com/abc-defg-hij"


def load_plugin():
    import importlib
    import types

    package = types.ModuleType("meeting_browser_under_test")
    package.__path__ = [str(SKILL_DIR)]
    sys.modules["meeting_browser_under_test"] = package
    return importlib.import_module("meeting_browser_under_test.plugin")


@pytest.fixture
def fake_door():
    with serve_fake_door() as door:
        yield door


def pcm_b64(samples) -> str:
    return base64.b64encode(array("h", samples).tobytes()).decode("ascii")


# ---- manifest, preflight, plugin, shared modules ---------------------------------------------

def test_manifest_declares_the_participant_surface():
    from ouroboros.contracts.skill_manifest import VALID_SKILL_PERMISSIONS, parse_skill_manifest_text
    from ouroboros.skill_dependencies import _manifest_declared_raw, normalize_declared_dependency_specs

    manifest = parse_skill_manifest_text((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))
    assert (manifest.name, manifest.type, manifest.entry) == ("meeting_browser", "extension", "plugin.py")
    assert set(manifest.permissions) == {"tool", "call_peer", "companion_process", "subprocess", "net"}
    [companion] = manifest.companion_processes
    assert companion["command"] == ["python3", "scripts/meeting_participant.py"]
    warnings = manifest.validate()
    if "call_peer" in VALID_SKILL_PERMISSIONS:
        assert warnings == []
    else:
        assert warnings and all("call_peer" in warning for warning in warnings)
    auto, manual, dependency_warnings = normalize_declared_dependency_specs(_manifest_declared_raw(manifest))
    assert [spec["package"] for spec in auto] == ["playwright==1.61.0"] and manual == [] == dependency_warnings


def test_skill_preflight_is_clean_including_the_page_script(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    skills_root = tmp_path / "skills"
    shutil.copytree(SKILL_DIR, skills_root / "meeting_browser", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    (tmp_path / "repo").mkdir()
    (tmp_path / "drive").mkdir()
    result = json.loads(_handle_skill_preflight(
        ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "drive"), skill="meeting_browser"))
    assert result["ok"] is True, result
    assert "scripts/participant_page.js" in {row["path"] for row in result["files"]}


def test_the_call_peer_and_control_channel_are_the_same_code_as_meeting_audio():
    audio = REPO_ROOT / "skills" / "meeting_audio" / "scripts"
    for name in ("meeting_peer.py", "meeting_control.py"):
        assert (SCRIPTS / name).read_bytes() == (audio / name).read_bytes(), f"{name} copies drifted"


def test_plugin_registers_participant_tools_and_refuses_outside_a_call(tmp_path, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setattr(plugin, "request", lambda *a, **k: pytest.fail("no companion call outside a call"))
    api = FakeAPI(tmp_path)
    plugin.register(api)
    assert set(api.tools) == {"meeting_join", "meeting_leave", "meeting_status"}
    assert api.companions == ["meeting_participant"]
    reply = tool_payload(api.tools["meeting_join"]["handler"](SimpleNamespace(task_metadata={}), url=MEET))
    assert reply["code"] == "MEETING_NO_CALL" and "press Call" in reply["message"]


def test_join_inside_a_call_reaches_the_companion(tmp_path, monkeypatch):
    plugin = load_plugin()
    seen = []
    monkeypatch.setattr(plugin, "request", lambda state, op, args, **k: seen.append((op, args)) or {"ok": True})
    api = FakeAPI(tmp_path)
    plugin.register(api)
    api.tools["meeting_join"]["handler"](SimpleNamespace(task_metadata=CALL_META), url=f" {MEET} ")
    assert seen == [("join", {"call_id": "call-1", "url": MEET})]


# ---- vendors and probes --------------------------------------------------------------------------

def test_vendor_comes_from_the_link_host():
    assert vendors.vendor_for(MEET) == ("meet", "https://meet.google.com")
    assert vendors.vendor_for("https://us02web.zoom.us/j/123?pwd=x") == ("zoom", "https://us02web.zoom.us")
    assert vendors.vendor_for("https://teams.microsoft.com/l/meetup-join/x")[0] == "teams"
    for link, code in (("http://meet.google.com/x", "MEETING_LINK_INVALID"),
                       ("https://meet.google.com.evil.example/x", "MEETING_VENDOR_UNKNOWN"),
                       ("https://example.com/zoom.us", "MEETING_VENDOR_UNKNOWN")):
        with pytest.raises(vendors.VendorRefused) as refused:
            vendors.vendor_for(link)
        assert refused.value.code == code
    assert vendors.belongs_to("teams", "https://teams.live.com/meet/1") is True
    assert vendors.belongs_to("teams", "https://login.example.com/") is False


def test_vendors_ship_unprobed_and_every_join_reports_what_the_probe_showed(tmp_path):
    table = vendors.probe_table(tmp_path)
    assert {vendor: row["result"] for vendor, row in table.items()} == {
        "zoom": "NOT_RUN", "meet": "NOT_RUN", "teams": "NOT_RUN"}
    assert "CAPTCHA" in table["zoom"]["known"] and "admits" in table["meet"]["known"]
    assert vendors.probe_note("meet", tmp_path).startswith("Experimental: Google Meet has not been probed")
    vendors.record_probe(tmp_path, "meet", vendors.FAIL, {}, "no remote meeting audio arrived")
    note = vendors.probe_note("meet", tmp_path)
    assert "failed its participant probe" in note and "no remote meeting audio" in note and "listen-only" in note
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {"published": True})
    assert vendors.probe_note("meet", tmp_path) == ""
    assert vendors.probe_table(tmp_path)["meet"]["where"] == "this computer"


def test_probe_report_needs_no_companion(tmp_path, capsys):
    assert probe_vendor.main(["--state-dir", str(tmp_path), "--report"]) == 0
    assert json.loads(capsys.readouterr().out)["teams"]["result"] == "NOT_RUN"


def test_presence_is_described_from_page_facts_only():
    speak = meeting_participant.describe_presence("Google Meet", {"published": True, "micLive": True})
    assert speak[0] == "speaking"
    muted = meeting_participant.describe_presence("Google Meet", {"published": True, "micLive": False})
    assert muted[0] == "listen-only" and "muted" in muted[1]
    taken = meeting_participant.describe_presence("Zoom web", {"microphoneRequests": 1})
    assert taken[0] == "listen-only" and "refused the replaced microphone" in taken[1]
    assert meeting_participant.describe_presence("Zoom web", {})[0] == "listen-only"


# ---- the browser session against a fake Playwright -----------------------------------------------------

class FakeFrame:
    def __init__(self):
        self.evaluated = []

    def evaluate(self, script, argument=None):
        self.evaluated.append((script, argument))


class FakePage:
    def __init__(self, context):
        self.context = context
        self.handlers = {}
        self.visited = []
        self.url = ""
        self.status = {"mode": "worklet", "microphoneRequests": 0, "published": False, "tappedTracks": 0}

    def on(self, event, handler):
        self.handlers[event] = handler

    def goto(self, url, **kwargs):
        self.visited.append((url, kwargs))
        self.url = url

    def evaluate(self, script, argument=None):
        if script == participant_browser.STATUS_SCRIPT:
            return dict(self.status)
        if "outboundAudioBytes" in script:
            return self.context.outbound
        raise AssertionError(script)

    def title(self):
        return "Meet: abc-defg-hij"

    def wait_for_timeout(self, ms):
        import time

        time.sleep(ms / 1000)

    def is_closed(self):
        return False


class FakeContext:
    def __init__(self, profile, kwargs):
        self.profile, self.kwargs = profile, kwargs
        self.granted, self.bindings, self.scripts, self.handlers = [], {}, [], {}
        self.pages = []
        self.closed = False
        self.outbound = 0

    def grant_permissions(self, permissions, origin=None):
        self.granted.append((tuple(permissions), origin))

    def expose_binding(self, name, callback):
        self.bindings[name] = callback

    def add_init_script(self, path=None, script=None):
        self.scripts.append(path)

    def on(self, event, handler):
        self.handlers[event] = handler

    def new_page(self):
        page = FakePage(self)
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True


class FakePlaywright:
    def __init__(self, failure=None):
        self.contexts = []
        self.failure = failure
        self.chromium = SimpleNamespace(launch_persistent_context=self.launch)

    def launch(self, profile, **kwargs):
        if self.failure:
            raise RuntimeError(self.failure)
        self.contexts.append(FakeContext(profile, kwargs))
        return self.contexts[-1]

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def open_session(tmp_path, playwright, remote, closed=None, url=MEET):
    session = participant_browser.BrowserSession(
        tmp_path, url, "https://meet.google.com", on_remote_pcm=lambda pcm, stream: remote.append((pcm, stream)),
        on_closed=(closed or []).append, belongs=lambda address: vendors.belongs_to("meet", address),
        playwright_factory=playwright)
    session.open()
    return session


def test_session_opens_a_visible_chrome_window_with_a_private_profile(tmp_path):
    playwright, remote = FakePlaywright(), []
    session = open_session(tmp_path, playwright, remote)
    try:
        [context] = playwright.contexts
        assert context.profile == str(tmp_path / participant_browser.PROFILE_DIR)
        assert context.kwargs["channel"] == "chrome" and context.kwargs["headless"] is False
        assert "--mute-audio" in context.kwargs["args"]
        assert "--autoplay-policy=no-user-gesture-required" in context.kwargs["args"]
        assert context.granted == [(("microphone",), "https://meet.google.com")]
        assert context.scripts == [str(PAGE_SCRIPT)]
        assert set(context.bindings) == {"__ouroborosRemotePcm", "__ouroborosParticipantEvent"}
        assert context.pages[0].visited[0][0] == MEET
        # A redirect inside the vendor gets the permission too; another site never does.
        context.pages[0].handlers["framenavigated"](SimpleNamespace(url="https://workspace.meet.google.com/x"))
        context.pages[0].handlers["framenavigated"](SimpleNamespace(url="https://accounts.example.com/login"))
        assert until(lambda: len(context.granted) == 2)
        assert context.granted[1] == (("microphone",), "https://workspace.meet.google.com")
    finally:
        session.close()
    assert context.closed


def test_session_feeds_only_frames_that_took_the_microphone_and_honors_flush(tmp_path):
    playwright, remote = FakePlaywright(), []
    session = open_session(tmp_path, playwright, remote)
    try:
        context = playwright.contexts[0]
        top, other = FakeFrame(), FakeFrame()
        context.bindings["__ouroborosParticipantEvent"]({"frame": top}, "microphone_requested")
        session.flush(4)
        session.feed(b"\x01\x00" * 2, 3)  # an older epoch still in flight
        session.feed(b"\x02\x00" * 2, 4)
        assert until(lambda: len(top.evaluated) >= 2)
        assert top.evaluated[0] == ("() => window.__ouroborosFlush()", None)
        assert top.evaluated[1][1] == base64.b64encode(b"\x02\x00" * 2).decode("ascii")
        assert other.evaluated == []
        context.bindings["__ouroborosRemotePcm"]({"frame": top}, pcm_b64([1, 2, 3]), 48000)
        context.bindings["__ouroborosRemotePcm"]({"frame": other}, pcm_b64([4]), 48000)
        context.bindings["__ouroborosRemotePcm"]({"frame": top}, pcm_b64([5]), 44100)  # wrong rate: dropped
        assert remote == [(array("h", [1, 2, 3]).tobytes(), 0), (array("h", [4]).tobytes(), 1)]
    finally:
        session.close()


def test_the_window_has_no_real_capture_device(tmp_path):
    """Chrome's fake capture devices: a frame or API the page script did not replace still gets no real mic."""
    playwright = FakePlaywright()
    session = open_session(tmp_path, playwright, [])
    try:
        assert "--use-fake-device-for-media-stream" in playwright.contexts[0].kwargs["args"]
    finally:
        session.close()


def test_missing_chrome_is_a_typed_refusal(tmp_path):
    playwright = FakePlaywright("BrowserType.launch_persistent_context: Chromium distribution 'chrome' is not "
                                "found at /Applications/Google Chrome.app")
    with pytest.raises(participant_browser.SessionFailed) as failed:
        open_session(tmp_path, playwright, [])
    assert failed.value.code == "MEETING_BROWSER_CHROME_MISSING"


# ---- the participant companion against the fake call door ---------------------------------------------

class FakeSession:
    instances = []

    def __init__(self, state_dir, url, origin, *, on_remote_pcm, on_closed, belongs):
        self.url, self.origin = url, origin
        self.on_remote_pcm, self.on_closed, self.belongs = on_remote_pcm, on_closed, belongs
        self.fed, self.flushes, self.closed, self.alive = [], [], "", False
        self.page_status, self.title, self.page_url = {}, "Meet", url
        FakeSession.instances.append(self)

    def open(self):
        self.alive = True

    def feed(self, pcm, epoch):
        self.fed.append((pcm, epoch))

    def flush(self, epoch):
        self.flushes.append(epoch)

    def call(self, function, timeout=30.0):
        return function(SimpleNamespace(evaluate=lambda script: 2048))

    def close(self, reason="closed"):
        self.closed, self.alive = reason, False


def participant(tmp_path, door):
    FakeSession.instances = []
    return meeting_participant.ParticipantCompanion(tmp_path, door.url, FakeDoor.token, session_factory=FakeSession)


@pytest.mark.serial
def test_join_refuses_without_a_call_or_an_unknown_link(tmp_path, fake_door):
    companion = participant(tmp_path, fake_door)
    assert companion.join("", MEET)["code"] == "MEETING_NO_CALL"
    assert companion.join("call-1", "https://example.com/meeting")["code"] == "MEETING_VENDOR_UNKNOWN"
    assert fake_door.opened == [] and FakeSession.instances == []


@pytest.mark.serial
def test_an_unprobed_vendor_joins_as_experimental(tmp_path, fake_door):
    companion = participant(tmp_path, fake_door)
    reply = companion.join("call-1", MEET)
    assert reply["state"] == "joining" and reply["experimental"].startswith("Experimental: Google Meet")
    assert "only listens" in reply["message"]
    companion.leave()


@pytest.mark.serial
def test_participant_carries_audio_both_ways_and_leaves_when_the_call_ends(tmp_path, fake_door):
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {"published": True})
    companion = participant(tmp_path, fake_door)
    reply = companion.join("call-1", MEET)
    assert reply["ok"] is True and reply["state"] == "joining" and "Google Meet" in reply["message"]
    assert fake_door.opened == [{"role": "meeting", "call_id": "call-1", "mic": True, "speaker": True}]
    assert fake_door.hellos[0] == {"type": "hello", "protocol": 1, "role": "meeting", "mic": True, "speaker": True,
                                   "rate": 48000, "label": "Ouroboros in Google Meet"}
    [session] = FakeSession.instances
    assert session.flushes == [7]
    session.on_remote_pcm(array("h", [3] * 960).tobytes(), 1)
    assert until(lambda: fake_door.frames)
    assert (fake_door.frames[0]["rate"], fake_door.frames[0]["stream"], fake_door.frames[0]["epoch"]) == (48000, 1, 7)
    fake_door.send(out_frame(7, b"\x09\x00" * 4))
    assert until(lambda: session.fed)
    assert session.fed == [(b"\x09\x00" * 4, 7)]
    session.page_status = {"microphoneRequests": 1, "published": True, "micLive": True, "tappedTracks": 2}
    status = companion.status()
    assert status["state"] == "speaking" and status["remote_audio_tracks"] == 2
    fake_door.end_call("finished")
    assert until(lambda: session.closed)
    assert "the call ended" in session.closed and companion.status()["state"] == "idle"


@pytest.mark.serial
def test_leave_and_a_closed_window_end_the_participation(tmp_path, fake_door):
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {})
    companion = participant(tmp_path, fake_door)
    assert companion.join("call-1", MEET)["ok"]
    assert companion.leave()["message"] == "left Google Meet and closed the window"
    assert until(lambda: fake_door.controls and fake_door.controls[-1]["type"] == "bye")
    assert companion.join("call-1", MEET)["ok"]
    second = FakeSession.instances[-1]
    second.on_closed("the browser window was closed")
    assert until(lambda: companion.status()["state"] == "idle")
    assert "the browser window was closed" in companion.status()["message"]


@pytest.mark.serial
def test_a_participant_that_drops_out_tells_the_call_and_a_requested_leave_does_not(tmp_path, fake_door):
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {})
    companion = participant(tmp_path, fake_door)

    def sent(kind):
        return [control for control in fake_door.controls if control.get("type") == kind]

    assert companion.join("call-1", MEET)["ok"]
    FakeSession.instances[-1].on_closed("the browser window was closed")  # closed or crashed by itself
    assert until(lambda: len(sent("bye")) == 1)  # a context goes before the bye on the same socket
    assert sent("context") == [{"type": "context", "text": "Ouroboros left Google Meet: the browser window was closed.",
                                "source": "meeting_browser"}]
    assert companion.join("call-1", MEET)["ok"]
    companion.leave()  # the mind asked for it and its tool result says so
    assert until(lambda: len(sent("bye")) == 2)
    assert len(sent("context")) == 1
    assert companion.join("call-1", MEET)["ok"]
    companion.shutdown()  # the skill was disabled or the host stopped the companion
    assert until(lambda: len(sent("bye")) == 3)
    assert sent("context")[-1]["text"] == "Ouroboros left Google Meet: the participant companion stopped."


@pytest.mark.serial
def test_a_browser_that_fails_to_open_closes_the_peer_it_was_granted(tmp_path, fake_door):
    peers = []

    def recording_peer(*args, **kwargs):
        peers.append(meeting_participant.DoorPeer(*args, **kwargs))
        return peers[-1]

    class FailingSession(FakeSession):
        def open(self):
            raise participant_browser.SessionFailed("MEETING_BROWSER_CHROME_MISSING", "Chrome is not installed.")

    companion = meeting_participant.ParticipantCompanion(tmp_path, fake_door.url, FakeDoor.token,
                                                         session_factory=FailingSession, peer_factory=recording_peer)
    reply = companion.join("call-1", MEET)
    assert reply == {"ok": False, "code": "MEETING_BROWSER_CHROME_MISSING", "message": "Chrome is not installed."}
    assert fake_door.opened == [{"role": "meeting", "call_id": "call-1", "mic": True, "speaker": True}]
    [peer] = peers
    with pytest.raises(meeting_participant.PeerRefused) as refused:
        peer.attach(speaker=True)  # the failed join's direction grant is never used by this peer again
    assert refused.value.code == "peer_closed"
    assert fake_door.hellos == [] and companion.status()["state"] == "idle"


def test_join_leaves_nothing_behind_when_the_call_closes_while_attaching(tmp_path):
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {})
    FakeSession.instances = []
    companion = meeting_participant.ParticipantCompanion(tmp_path, "http://127.0.0.1:9", "token",
                                                         session_factory=FakeSession, peer_factory=RacingPeer)
    reply = companion.join("call-1", MEET)
    assert reply["code"] == "MEETING_CALL_GONE" and "the call ended (finished)" in reply["message"]
    [session] = FakeSession.instances
    assert session.closed and session.alive is False
    assert companion.status()["state"] == "idle" and companion.leave()["message"] == "not in a meeting"


@pytest.mark.serial
def test_a_dropped_participant_socket_is_not_called_the_end_of_the_call(tmp_path, fake_door):
    vendors.record_probe(tmp_path, "meet", vendors.PASS, {})
    companion = participant(tmp_path, fake_door)
    assert companion.join("call-1", MEET)["ok"]
    fake_door.drop_peer("peer_send_timeout")
    assert until(lambda: companion.status()["state"] == "idle")
    message = companion.status()["message"]
    assert "closed the participant's call socket (peer_send_timeout)" in message and "call ended" not in message
    assert FakeSession.instances[-1].closed


def test_a_closed_window_takes_no_more_call_audio(tmp_path):
    session = participant_browser.BrowserSession(
        tmp_path, MEET, "https://meet.google.com", on_remote_pcm=lambda *a: None, on_closed=lambda reason: None,
        belongs=lambda address: True, playwright_factory=FakePlaywright())
    session.close("left")
    session.feed(b"\x01\x00" * 4, 1)
    assert session._commands.qsize() == 1  # only the close command; the audio was not queued


def test_the_host_token_stays_in_the_companion_and_never_reaches_playwright_or_chrome(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("HOST_SERVICE_URL", "http://127.0.0.1:8770")
    monkeypatch.setenv("HOST_SERVICE_TOKEN", "skill-secret")
    started = []

    class StoppedCompanion:
        def __init__(self, state_dir, host_url, host_token):
            started.append((host_url, host_token))
            self.stop_event = threading.Event()
            self.stop_event.set()  # only main's start-up is under test

        def start(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(meeting_participant, "ParticipantCompanion", StoppedCompanion)
    monkeypatch.setattr(meeting_participant.signal, "signal", lambda *args: None)
    assert meeting_participant.main() == 0
    assert started == [("http://127.0.0.1:8770", "skill-secret")]
    assert [key for key in os.environ if key.upper().startswith("HOST_SERVICE_")] == []
    child = subprocess.run(
        [sys.executable, "-I", "-c", "import os; print(sorted(k for k in os.environ if k.startswith('HOST_SERVICE_')))"],
        capture_output=True, text=True, timeout=60)
    assert child.stdout.strip() == "[]", "Playwright's driver and Chrome start from this environment"


def test_a_pid1_server_is_a_live_parent_and_a_reparented_orphan_is_not(monkeypatch):
    point_lifeline(monkeypatch, meeting_participant, start=1, named=1)  # a container's server is PID 1
    assert meeting_participant._parent_alive(1), "a PID-1 server that started the companion is a live parent"
    assert meeting_participant._lifeline_parent() == 1
    ppid = point_lifeline(monkeypatch, meeting_participant, start=4321, named=4321)  # a desktop server
    parent = meeting_participant._lifeline_parent()
    assert parent == 4321 and meeting_participant._parent_alive(parent)
    ppid["now"] = 1  # the server died and init adopted the companion
    assert not meeting_participant._parent_alive(parent)
    point_lifeline(monkeypatch, meeting_participant, start=1, named=4321)  # orphaned before it started
    assert meeting_participant._lifeline_parent() == 0 and not meeting_participant._parent_alive(0)
    point_lifeline(monkeypatch, meeting_participant, start=1, named=None)  # a PID 1 nobody named
    assert meeting_participant._lifeline_parent() == 0


def test_probe_records_pass_only_when_every_fact_held(tmp_path, monkeypatch):
    monkeypatch.setattr(meeting_participant, "PROBE_CHUNK_SEC", 0.01)
    monkeypatch.setattr(meeting_participant, "MIN_PROBE_SEC", 0.05)
    FakeSession.instances = []

    def session_with(facts, remote_samples):
        def factory(*args, **kwargs):
            session = FakeSession(*args, **kwargs)
            session.page_status = facts
            original_open = session.open

            def open_and_hear():
                original_open()
                session.on_remote_pcm(array("h", remote_samples).tobytes(), 0)

            session.open = open_and_hear
            return session

        return factory

    good = meeting_participant.ParticipantCompanion(tmp_path, "", "", session_factory=session_with(
        {"microphoneRequests": 1, "published": True, "micLive": True}, [0, 5, -5]))
    result = good.probe(MEET, 0.05)
    assert result["result"] == "PASS" and result["facts"]["outbound_audio_bytes"] == 2048
    assert result["facts"]["remote_nonzero_samples"] == 2
    assert FakeSession.instances[-1].fed, "the probe speaks a test tone into the microphone"
    assert vendors.probe_table(tmp_path)["meet"]["result"] == "PASS"
    silent = meeting_participant.ParticipantCompanion(tmp_path, "", "", session_factory=session_with(
        {"microphoneRequests": 1, "published": False}, [0, 0]))
    failed = silent.probe(MEET, 0.05)
    assert failed["result"] == "FAIL"
    assert "never sent the microphone" in failed["reason"] and "no remote meeting audio" in failed["reason"]
    assert vendors.probe_table(tmp_path)["meet"]["result"] == "FAIL"


# ---- the page script under Node: its PCM path in both audio modes ---------------------------------------------

NODE_HARNESS = r"""
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const source = fs.readFileSync(process.env.PAGE_SCRIPT, 'utf8');

function fixture(options) {
  const calls = { remote: [], events: [] };
  const workletSources = [];
  const urls = new Map();
  class Track {
    constructor(kind) { this.kind = kind; this.enabled = true; this.readyState = 'live'; this.listeners = {}; }
    clone() { return new Track(this.kind); }
    addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); }
  }
  class Stream {
    constructor(tracks) { this.tracks = tracks ? tracks.slice() : []; }
    getAudioTracks() { return this.tracks.filter((t) => t.kind === 'audio'); }
    getVideoTracks() { return this.tracks.filter((t) => t.kind === 'video'); }
    addTrack(track) { this.tracks.push(track); }
  }
  class Node { constructor(kind) { this.kind = kind; this.connections = []; } connect(n) { this.connections.push(n); return n; } disconnect() {} }
  class Context {
    constructor(opts) {
      this.sampleRate = opts.sampleRate; this.state = 'running'; this.destination = new Node('destination');
      this.nodes = [];
      this.audioWorklet = { addModule: async (url) => {
        if (options.workletRefused) throw new Error('refused by the page content policy');
        workletSources.push(urls.get(url));
      } };
      fixture.context = this;
    }
    createGain() { const n = new Node('gain'); n.gain = { value: 1 }; this.nodes.push(n); return n; }
    createMediaStreamDestination() { const n = new Node('destination-stream'); n.stream = new Stream([new Track('audio')]); this.nodes.push(n); return n; }
    createMediaStreamSource(stream) { const n = new Node('source'); n.stream = stream; this.nodes.push(n); return n; }
    createScriptProcessor(size, inputs) { const n = new Node('script'); n.inputs = inputs; this.nodes.push(n); return n; }
  }
  class WorkletNode extends Node {
    constructor(ctx, name) { super('worklet'); this.name = name; this.port = { messages: [], onmessage: null, postMessage(m) { this.messages.push(m); } }; ctx.nodes.push(this); }
  }
  class PeerConnection {
    constructor() { this.listeners = {}; }
    addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); }
    emit(name, event) { (this.listeners[name] || []).forEach((fn) => fn(event)); }
    addTrack(track) { return { track }; }
    addTransceiver() { return {}; }
    getStats() { return Promise.resolve(new Map([['a', { type: 'outbound-rtp', kind: 'audio', bytesSent: 640 }]])); }
  }
  function Sender() {}
  Sender.prototype.replaceTrack = function () { return Promise.resolve(); };
  class MediaDevices {
    async getUserMedia() { throw new Error('the native microphone must never be opened for audio'); }
    async enumerateDevices() { return [{ kind: 'audioinput', deviceId: 'real' }, { kind: 'audiooutput', deviceId: 'speakers' }]; }
  }
  const scope = {
    atob: (text) => Buffer.from(text, 'base64').toString('binary'),
    btoa: (text) => Buffer.from(text, 'binary').toString('base64'),
    AudioContext: Context, AudioWorkletNode: WorkletNode, MediaStream: Stream,
    Blob: class { constructor(parts) { this.text = parts.join(''); } },
    URL: { createObjectURL(blob) { const id = 'blob:' + urls.size; urls.set(id, blob.text); return id; } },
    RTCPeerConnection: PeerConnection, RTCRtpSender: Sender, setInterval: () => 0,
    document: { createElement: () => ({ play: () => Promise.resolve(), getContext: () => ({ fillRect() {}, fillText() {} }),
      captureStream: () => new Stream([new Track('video')]) }) },
    MediaDevices, navigator: { mediaDevices: new MediaDevices() },
    __ouroborosRemotePcm: (text, rate) => { calls.remote.push({ text, rate }); return Promise.resolve(); },
    __ouroborosParticipantEvent: (name) => { calls.events.push(name); return Promise.resolve(); },
  };
  new Function('window', source)(scope);
  return { scope, calls, workletSources, Track };
}

function pcm(values) { return Buffer.from(Int16Array.from(values).buffer).toString('base64'); }
function ints(text) { const bytes = Buffer.from(text, 'base64'); return Array.from(new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2)); }
const tick = () => new Promise((resolve) => setImmediate(resolve));

test('worklet path: fed call audio becomes the microphone, flush drops it, the tap returns remote audio', async () => {
  const { scope, calls, workletSources, Track } = fixture({});
  const stream = await scope.navigator.mediaDevices.getUserMedia({ audio: true, video: true });
  assert.strictEqual(stream.getAudioTracks().length, 1);
  assert.strictEqual(stream.getVideoTracks().length, 1);
  assert.deepStrictEqual(calls.events, ['microphone_requested']);
  const registered = {};
  class AudioWorkletProcessor { constructor() { this.port = { posted: [], postMessage(m) { this.posted.push(m); } }; } }
  new Function('AudioWorkletProcessor', 'registerProcessor', workletSources[0])(
    AudioWorkletProcessor, (name, cls) => { registered[name] = cls; });
  const nodes = fixture.context.nodes.filter((n) => n.kind === 'worklet');
  const feedNode = nodes.find((n) => n.name === 'ouroboros-feed');
  const tapNode = nodes.find((n) => n.name === 'ouroboros-tap');
  const feed = new registered['ouroboros-feed']();
  assert.strictEqual(scope.__ouroborosFeed(pcm([16384, -16384, 32767])), true);
  feed.port.onmessage({ data: feedNode.port.messages.shift() });
  let out = new Float32Array(5);
  feed.process([], [[out]]);
  assert.deepStrictEqual(Array.from(out), [0.5, -0.5, 32767 / 32768, 0, 0]);
  scope.__ouroborosFeed(pcm([1000, 1000]));
  feed.port.onmessage({ data: feedNode.port.messages.shift() });
  scope.__ouroborosFlush();
  feed.port.onmessage({ data: feedNode.port.messages.shift() });
  out = new Float32Array(2);
  feed.process([], [[out]]);
  assert.deepStrictEqual(Array.from(out), [0, 0]);

  const tap = new registered['ouroboros-tap']();
  tap.process([[new Float32Array(128).fill(0.5), new Float32Array(128).fill(-0.25)]]);
  tapNode.port.onmessage({ data: tap.port.posted.shift() });
  assert.strictEqual(calls.remote.length, 0, 'no remote track yet: nothing is carried');
  const connection = new scope.RTCPeerConnection();
  connection.emit('track', { track: new Track('audio') });
  connection.emit('track', { track: stream.getAudioTracks()[0] });
  await tick(); await tick();
  assert.strictEqual(scope.__ouroborosParticipant.status().tappedTracks, 1, 'own microphone is never tapped');
  for (let i = 0; i < 15; i += 1) {
    tap.process([[new Float32Array(128).fill(0.5), new Float32Array(128).fill(-0.25)]]);
    tapNode.port.onmessage({ data: tap.port.posted.shift() });
  }
  assert.strictEqual(calls.remote.length, 1);
  assert.strictEqual(calls.remote[0].rate, 48000);
  const samples = ints(calls.remote[0].text);
  assert.strictEqual(samples.length, 1920);
  assert.ok(samples.every((value) => value === Math.round(0.125 * 32767)));
  connection.addTrack(stream.getAudioTracks()[0]);
  const status = scope.__ouroborosParticipant.status();
  assert.strictEqual(status.mode, 'worklet');
  assert.strictEqual(status.published, true);
  assert.strictEqual(status.micLive, true);
  assert.strictEqual(await scope.__ouroborosParticipant.outboundAudioBytes(), 640);
});

test('fallback path: a refused worklet keeps the same arithmetic on a ScriptProcessor', async () => {
  const { scope, calls, Track } = fixture({ workletRefused: true });
  await scope.navigator.mediaDevices.getUserMedia({ audio: true });
  const [feed, tap] = fixture.context.nodes.filter((n) => n.kind === 'script');
  assert.strictEqual(scope.__ouroborosParticipant.status().mode, 'script');
  assert.match(scope.__ouroborosParticipant.status().errors[0], /content policy/);
  scope.__ouroborosFeed(pcm([8192, -8192]));
  const out = new Float32Array(3);
  feed.onaudioprocess({ outputBuffer: { getChannelData: () => out } });
  assert.deepStrictEqual(Array.from(out), [0.25, -0.25, 0]);
  const input = { length: 1920, numberOfChannels: 2,
    getChannelData: (channel) => new Float32Array(1920).fill(channel ? -1 : 1) };
  tap.onaudioprocess({ inputBuffer: input });
  assert.strictEqual(calls.remote.length, 0);
  new scope.RTCPeerConnection().emit('track', { track: new Track('audio') });
  await tick(); await tick();
  tap.onaudioprocess({ inputBuffer: input });
  assert.deepStrictEqual(ints(calls.remote[0].text).slice(0, 3), [0, 0, 0]);
  const devices = await scope.navigator.mediaDevices.enumerateDevices();
  assert.deepStrictEqual(devices.map((d) => [d.kind, d.label || d.deviceId]),
    [['audioinput', 'Ouroboros'], ['audiooutput', 'speakers']]);
});

test('the prototype method gives the replaced microphone too, never the native one', async () => {
  const { scope, calls } = fixture({});
  const viaPrototype = await scope.MediaDevices.prototype.getUserMedia.call(scope.navigator.mediaDevices, { audio: true });
  assert.strictEqual(viaPrototype.getAudioTracks().length, 1);
  const viaAnotherObject = await new scope.MediaDevices().getUserMedia({ audio: true });
  assert.strictEqual(viaAnotherObject.getAudioTracks().length, 1);
  assert.deepStrictEqual(calls.events, ['microphone_requested', 'microphone_requested']);
});
"""


def node_binary() -> str:
    from ouroboros.node_runtime import select_skill_node_runtime

    path, _provenance = select_skill_node_runtime()
    return path or shutil.which("node") or ""


def test_page_script_pcm_path_under_node(tmp_path):
    node = node_binary()
    if not node:
        pytest.skip("no Node runtime on this host for the page script test")
    harness = tmp_path / "participant_page.test.js"
    harness.write_text(NODE_HARNESS, encoding="utf-8")
    result = subprocess.run([node, "--test", "--test-reporter=tap", str(harness)], capture_output=True, text=True,
                            timeout=120,
                            env={"PATH": "", "PAGE_SCRIPT": str(PAGE_SCRIPT)}, cwd=str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# pass 3" in result.stdout and "# fail 0" in result.stdout


# ---- real Chromium: the replaced microphone carries fed audio, the tap returns remote audio ---------------------

LOOPBACK_PAGE = """<!doctype html><html><body><script>
window.results = {};
// Non-trickle exchange: each side waits for its candidates before handing over its description.
const gathered = (pc) => new Promise((resolve) => {
  if (pc.iceGatheringState === 'complete') return resolve();
  pc.addEventListener('icegatheringstatechange', () => { if (pc.iceGatheringState === 'complete') resolve(); });
});
(async () => {
  const ours = new RTCPeerConnection();
  const meeting = new RTCPeerConnection();
  window.connections = [ours, meeting];
  const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
  mic.getTracks().forEach((track) => ours.addTrack(track, mic));
  await ours.setLocalDescription(await ours.createOffer());
  await gathered(ours);
  await meeting.setRemoteDescription(ours.localDescription);
  const speaker = new AudioContext();
  const tone = speaker.createOscillator();
  tone.frequency.value = 880;
  const out = speaker.createMediaStreamDestination();
  tone.connect(out);
  tone.start();
  out.stream.getTracks().forEach((track) => meeting.addTrack(track, out.stream));
  await meeting.setLocalDescription(await meeting.createAnswer());
  await gathered(meeting);
  await ours.setRemoteDescription(meeting.localDescription);
  window.results.ready = true;
})().catch((error) => { window.results.error = String(error); });
</script></body></html>"""


def tone_power(pcm: bytes, frequency: float, rate: int = 48000) -> float:
    samples = array("h", pcm)
    coefficient = 2 * math.cos(2 * math.pi * frequency / rate)
    previous = before = 0.0
    for sample in samples:
        previous, before = sample + coefficient * previous - before, previous
    return (previous * previous + before * before - coefficient * previous * before) / max(1, len(samples)) ** 2


@pytest.mark.ui_browser
def test_replaced_microphone_carries_fed_audio_and_the_tap_returns_remote_audio(tmp_path):
    from playwright.sync_api import sync_playwright

    (tmp_path / "index.html").write_text(LOOPBACK_PAGE, encoding="utf-8")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), lambda *a, **k: Quiet(*a, directory=str(tmp_path), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chunks, events = [], []
    try:
        with sync_playwright() as playwright:
            # The participant window's own flags (its fake capture devices included); host candidates stay
            # plain IP addresses: a sandboxed headless browser may not resolve mDNS names.
            browser = playwright.chromium.launch(args=[*participant_browser.LAUNCH_ARGS,
                                                       "--disable-features=WebRtcHideLocalIpsWithMdns"])
            try:
                context = browser.new_context()
                context.expose_binding("__ouroborosRemotePcm",
                                       lambda source, text, rate: chunks.append((base64.b64decode(text), rate)))
                context.expose_binding("__ouroborosParticipantEvent",
                                       lambda source, name, detail="": events.append(name))
                context.add_init_script(path=str(PAGE_SCRIPT))
                page = context.new_page()
                page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html")
                page.wait_for_function("window.results.ready || window.results.error", timeout=20000)
                assert page.evaluate("window.results.error") is None
                page.wait_for_timeout(2500)  # the meeting side speaks; the microphone is fed nothing yet
                before = b"".join(pcm for pcm, _rate in chunks)
                chunks.clear()
                frames = 48000 * 3
                fed = struct.pack(f"<{frames}h", *(int(12000 * math.sin(2 * math.pi * 440 * index / 48000))
                                                   for index in range(frames)))
                assert page.evaluate("(text) => window.__ouroborosFeed(text)", base64.b64encode(fed).decode())
                page.wait_for_timeout(2500)
                after = b"".join(pcm for pcm, _rate in chunks)
                status = page.evaluate("window.__ouroborosParticipant.status()")
                via_prototype = page.evaluate("""async () => {
                  const stream = await MediaDevices.prototype.getUserMedia.call(navigator.mediaDevices, { audio: true });
                  return [stream.getAudioTracks().length, window.__ouroborosParticipant.status().microphoneRequests];
                }""")
            finally:
                browser.close()
    finally:
        server.shutdown()
    assert {rate for _pcm, rate in chunks} == {48000}
    assert status["mode"] == "worklet" and status["published"] is True and status["microphoneRequests"] == 1
    assert status["tappedTracks"] == 2  # the meeting's tone at ours, our microphone at the meeting
    assert via_prototype == [1, 2], "the prototype method is the replaced microphone in a real browser too"
    assert events[:1] == ["microphone_requested"] and "microphone_published" in events
    assert len(before) > 48000 and len(after) > 48000
    # The tap returns the meeting's 880 Hz; the fed 440 Hz appears only once fed through WebRTC.
    assert tone_power(before, 880) > 20 * tone_power(before, 440)
    assert tone_power(after, 440) > 20 * tone_power(before, 440)
