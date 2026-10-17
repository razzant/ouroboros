"""Platform capture backends of the meeting_audio skill, each against fakes of what it drives.

Linux runs against a stateful fake of pactl / pw-dump / pw-link and fake helper
processes, macOS against fake pyobjc modules, Windows against fake process-loopback
streams and a fake PyAudioWPatch. No device, permission prompt or audio server is
touched.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "skills" / "meeting_audio" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import capture_common  # noqa: E402
import capture_linux  # noqa: E402
import capture_macos  # noqa: E402
import capture_windows  # noqa: E402
import meeting_apps  # noqa: E402
import meeting_capture  # noqa: E402
import probe_capture  # noqa: E402
from meeting_apps import RunningApp  # noqa: E402

from tests.test_skill_meeting_audio import until  # noqa: E402


def test_pcm_conversions_scale_clip_and_average():
    from array import array

    floats = array("f", [0.0, 0.5, -0.5, 1.5, -1.0, 0.25]).tobytes()
    assert array("h", capture_common.float32_mono(floats)).tolist() == [0, 16384, -16384, 32767, -32767, 8192]
    planar = array("f", [1.0, 0.0, 0.0, 1.0]).tobytes()  # two channels, two frames each
    assert array("h", capture_common.float32_mono(planar, 2, planar=True)).tolist() == [16384, 16384]
    stereo = array("h", [100, 300, -50, -150]).tobytes()
    assert array("h", capture_common.int16_mono(stereo, 2)).tolist() == [200, -100]


# ---- Linux backend -------------------------------------------------------------------------

class FakePactl:
    """A stateful fake of pactl/pw-dump/pw-link for one desktop session."""

    def __init__(self, *, pipewire=False, graph=None, sink_inputs="", sinks="1\talsa_output.pci\tmodule\n",
                 modules=None, link_ok=True, server_name="pulseaudio"):
        self.pipewire = pipewire
        self.server_name = server_name  # what `pactl info` calls the Pulse server
        self.graph = graph or []
        self.sink_inputs = sink_inputs
        self.sinks = sinks
        self.sources = ""
        self.modules = dict(modules or {})
        self.next_id = 536870913
        self.calls = []
        self.link_ok = link_ok
        self.server_down = False  # pactl reaches no server
        self.unload_refused = set()  # module ids whose unload fails

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        assert kwargs["env"]["LC_ALL"] == "C"
        ok = lambda out="": subprocess.CompletedProcess(argv, 0, out, "")  # noqa: E731
        if argv[0] == "pw-dump":
            return ok(json.dumps(self.graph)) if self.pipewire else subprocess.CompletedProcess(argv, 1, "", "no pw")
        if argv[0] == "pw-link":
            return ok() if self.link_ok else subprocess.CompletedProcess(argv, 1, "", "failed to link ports")
        if argv[0] == "pactl" and self.server_down:
            return subprocess.CompletedProcess(argv, 1, "", "Connection failure: Connection refused")
        if argv[:2] == ["pactl", "info"]:
            return ok(f"Server String: /run/user/1000/pulse/native\nServer Name: {self.server_name}\n"
                      "Server Version: 15.99.1\n")
        if argv[:3] == ["pactl", "list", "sink-inputs"]:
            return ok(self.sink_inputs)
        if argv[:4] == ["pactl", "list", "sinks", "short"]:
            return ok(self.sinks + self._devices("module-null-sink", "sink_name="))
        if argv[:4] == ["pactl", "list", "sources", "short"]:
            return ok(self.sources + self._devices("module-remap-source", "source_name="))
        if argv[:4] == ["pactl", "list", "modules", "short"]:
            return ok("".join(f"{mid}\t{name}\t{args}\t\n" for mid, (name, args) in self.modules.items()))
        if argv[:2] == ["pactl", "load-module"]:
            module_id, self.next_id = self.next_id, self.next_id + 1
            self.modules[module_id] = (argv[2], " ".join(argv[3:]))
            return ok(f"{module_id}\n")
        if argv[:2] == ["pactl", "unload-module"]:
            if int(argv[2]) in self.unload_refused:
                return subprocess.CompletedProcess(argv, 1, "", "Failure: Invalid argument")
            return ok() if self.modules.pop(int(argv[2]), None) else subprocess.CompletedProcess(argv, 1, "", "no")
        raise AssertionError(f"unexpected command {argv}")

    def _devices(self, module, key):
        """The devices loaded modules create, as `pactl list ... short` rows."""
        rows = []
        for module_id, (name, args) in self.modules.items():
            for arg in args.split():
                if name == module and arg.startswith(key):
                    rows.append(f"{module_id}\t{arg[len(key):]}\t{module}.c\n")
        return "".join(rows)


class FakeProc:
    def __init__(self, argv, stdout_bytes=b"", pid=4242):
        self.argv = argv
        self.pid = pid
        self.stdout = __import__("io").BytesIO(stdout_bytes)
        self.stderr = __import__("io").BytesIO(b"")
        self.stdin = __import__("io").BytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def linux_backend(tmp_path, fake, tools, procs, stdout=b""):
    def popen(argv, **kwargs):
        proc = FakeProc(argv, stdout)
        procs.append(proc)
        return proc

    return capture_linux.LinuxBackend(tmp_path, which=lambda name: f"/usr/bin/{name}" if name in tools else None,
                                      run=fake.run, popen=popen, env={"PATH": "/usr/bin"})


def test_linux_names_the_missing_tools(tmp_path):
    backend = linux_backend(tmp_path, FakePactl(), set(), [])
    with pytest.raises(capture_common.CaptureFailed) as failed:
        backend.running_apps()
    assert failed.value.code == "MEETING_AUDIO_TOOLS_MISSING"
    assert "pw-record" in failed.value.message and "parec" in failed.value.message


PW_GRAPH = [
    {"id": 50, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Stream/Output/Audio", "application.name": "ZOOM VoiceEngine",
        "application.process.binary": "zoom", "application.process.id": 900, "node.name": "ZOOM VoiceEngine"}}},
    {"id": 51, "type": "PipeWire:Interface:Port", "info": {"direction": "output", "props": {"node.id": 50}}},
    {"id": 52, "type": "PipeWire:Interface:Port", "info": {"direction": "output", "props": {"node.id": 50}}},
    {"id": 60, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Stream/Output/Audio", "application.name": "Firefox", "application.process.binary": "firefox"}}},
    {"id": 70, "type": "PipeWire:Interface:Node", "info": {"props": {
        "media.class": "Stream/Input/Audio", "node.name": capture_linux.NODE_NAME, "application.process.id": 4242}}},
    {"id": 71, "type": "PipeWire:Interface:Port", "info": {"direction": "input", "props": {"node.id": 70}}},
]


def test_linux_pipewire_links_only_the_meeting_apps_streams(tmp_path):
    fake, procs = FakePactl(pipewire=True, graph=PW_GRAPH), []
    pcm = b"\x01\x00" * 960 * 2
    backend = linux_backend(tmp_path, fake, {"pw-record", "pw-dump", "pw-link"}, procs, stdout=pcm)
    apps = backend.running_apps()
    assert [(app.name, app.ident, app.handle) for app in apps] == [
        ("ZOOM VoiceEngine", "zoom", "50"), ("Firefox", "firefox", "60")]
    selection = meeting_apps.choose(apps)
    assert selection.label == "Zoom"
    capture = backend.open_capture(selection)
    received = []
    capture.start(lambda chunk, stream=0: received.append((chunk, stream)), lambda reason: None)
    try:
        assert procs[0].argv[:12] == ["pw-record", "--raw", "--target", "0", "--rate", "48000",
                                      "--channels", "1", "--format", "s16", "--latency", "20ms"]
        assert json.loads(procs[0].argv[13])["node.name"] == capture_linux.NODE_NAME
        assert [call for call in fake.calls if call[0] == "pw-link"] == [["pw-link", "51", "71"],
                                                                         ["pw-link", "52", "71"]]
        assert until(lambda: len(received) == 2)
        assert received[0] == (pcm[:1920], 0)
        assert capture.describe()["source"] == "application streams"
        assert capture.disclosures() == []
    finally:
        capture.stop()


SINK_INPUTS = """Sink Input #42
\tDriver: protocol-native.c
\tSink: 1
\tSample Specification: s16le 2ch 48000Hz
\tProperties:
\t\tmedia.name = "Playback"
\t\tapplication.name = "ZOOM VoiceEngine"
\t\tapplication.process.id = "900"
\t\tapplication.process.binary = "zoom"
Sink Input #43
\tSink: 1
\tProperties:
\t\tapplication.name = "Ouroboros voice"
"""


def test_linux_pulse_records_the_sink_monitor_and_discloses_it(tmp_path):
    fake, procs = FakePactl(sink_inputs=SINK_INPUTS), []
    backend = linux_backend(tmp_path, fake, {"parec", "pactl"}, procs)
    apps = backend.running_apps()
    assert [(app.name, app.handle) for app in apps] == [("ZOOM VoiceEngine", "42")]
    capture = backend.open_capture(meeting_apps.choose(apps, "zoom"))
    capture.start(lambda chunk, stream=0: None, lambda reason: None)
    try:
        assert procs[0].argv == ["parec", "--raw", "--device=alsa_output.pci.monitor", "--format=s16le",
                                 "--rate=48000", "--channels=1", "--latency-msec=20",
                                 "--client-name=Ouroboros meeting capture"]
        assert capture.describe()["includes_own_voice"] is True
        [note] = capture.disclosures()
        assert "everything playing on alsa_output.pci" in note and "own whispered hints" in note
    finally:
        capture.stop()


@pytest.mark.parametrize("server_name,stack,recorder", [
    ("pulseaudio", "pulse", "parec"),  # Ubuntu 22.04: PulseAudio plays, a PipeWire daemon runs for screen sharing
    ("PulseAudio (on PipeWire 0.3.48)", "pipewire", "pw-record"),  # PipeWire's own Pulse layer answers
])
def test_linux_listens_through_the_server_that_plays_the_sound(tmp_path, server_name, stack, recorder):
    fake, procs = FakePactl(pipewire=True, graph=PW_GRAPH, sink_inputs=SINK_INPUTS, server_name=server_name), []
    backend = linux_backend(tmp_path, fake, {"pw-record", "pw-dump", "pw-link", "parec", "pactl"}, procs)
    assert backend.stack() == stack, "a working pw-dump alone must not pick PipeWire"
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps(), "zoom"))
    capture.start(lambda *a: None, lambda reason: None)
    try:
        assert procs[0].argv[0] == recorder and capture.describe()["backend"] == stack
    finally:
        capture.stop()


def test_linux_uses_pipewire_when_no_pulse_server_answers(tmp_path):
    fake = FakePactl(pipewire=True, graph=PW_GRAPH)
    fake.server_down = True
    backend = linux_backend(tmp_path, fake, {"pw-record", "pw-dump", "pw-link", "parec", "pactl"}, [])
    assert backend.stack() == "pipewire"
    fake.pipewire = False  # and with neither answering, the refusal names the unreachable server
    with pytest.raises(capture_common.CaptureFailed) as failed:
        backend.stack()
    assert failed.value.code == "MEETING_AUDIO_SERVER_UNREACHABLE" and "Connection refused" in failed.value.message


def test_audio_helpers_never_inherit_the_host_service_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOST_SERVICE_URL", "http://127.0.0.1:8770")
    monkeypatch.setenv("HOST_SERVICE_TOKEN", "skill-secret")
    fake, envs = FakePactl(sink_inputs=SINK_INPUTS), []

    def run(argv, **kwargs):
        envs.append(kwargs["env"])
        return fake.run(argv, **kwargs)

    def popen(argv, **kwargs):
        envs.append(kwargs["env"])
        return FakeProc(argv)

    tools = {"parec", "pactl", "pacat"}
    backend = capture_linux.LinuxBackend(tmp_path, which=lambda name: f"/usr/bin/{name}" if name in tools else None,
                                         run=run, popen=popen)  # no env given: the companion's own is the base
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps(), "zoom"))
    capture.start(lambda *a: None, lambda reason: None)
    capture.stop()
    voice = backend.open_voice(48000)
    voice.start()
    voice.close()
    assert len(envs) > 5 and all(env["LC_ALL"] == "C" for env in envs)
    leaked = [key for env in envs for key in env if key.upper().startswith("HOST_SERVICE_")]
    assert leaked == [], "the token stays in the companion; no recorder, player or pactl run gets it"


def test_virtual_microphone_unloads_exactly_the_modules_it_loaded(tmp_path):
    fake, procs = FakePactl(modules={1: ("module-device-restore", "")}), []
    backend = linux_backend(tmp_path, fake, {"pactl", "pacat"}, procs)
    voice = backend.open_voice(48000)
    voice.start()
    record = json.loads((tmp_path / capture_linux.MODULES_FILE).read_text(encoding="utf-8"))
    loaded = [entry["id"] for entry in record["modules"]]
    assert [entry["name"] for entry in record["modules"]] == ["module-null-sink", "module-remap-source"]
    assert fake.modules[loaded[1]] == ("module-remap-source", "master=ouroboros_voice.monitor "
                                       "source_name=ouroboros_mic source_properties=device.description=Ouroboros")
    assert procs[0].argv[:4] == ["pacat", "--playback", "--raw", "--device=ouroboros_voice"]
    assert "--rate=48000" in procs[0].argv
    assert voice.close() == list(reversed(loaded))
    assert set(fake.modules) == {1}
    assert not (tmp_path / capture_linux.MODULES_FILE).exists()


def test_crash_leftover_modules_are_unloaded_only_when_still_ours(tmp_path):
    null_args = "sink_name=ouroboros_voice sink_properties=device.description=Ouroboros-voice"
    (tmp_path / capture_linux.MODULES_FILE).write_text(json.dumps({"modules": [
        {"id": 25, "name": "module-null-sink", "argument": null_args},
        {"id": 26, "name": "module-remap-source", "argument": "master=ouroboros_voice.monitor source_name=x"},
    ], "pid": 1}), encoding="utf-8")
    # The server restarted since: id 26 now belongs to somebody else's module.
    fake = FakePactl(modules={25: ("module-null-sink", null_args), 26: ("module-loopback", "latency_msec=1")})
    backend = linux_backend(tmp_path, fake, {"pactl"}, [])
    assert backend.cleanup_leftovers() == [25]
    assert set(fake.modules) == {26}
    assert not (tmp_path / capture_linux.MODULES_FILE).exists()
    assert [call for call in fake.calls if call[:2] == ["pactl", "unload-module"]] == [
        ["pactl", "unload-module", "25"]]


def test_virtual_microphone_leaves_a_same_named_device_alone(tmp_path):
    fake = FakePactl(sinks="5\touroboros_voice\tmodule-null-sink.c\n")
    backend = linux_backend(tmp_path, fake, {"pactl", "pacat"}, [])
    with pytest.raises(capture_linux.VoiceFailed) as failed:
        backend.open_voice(48000).start()
    assert failed.value.code == "MEETING_VOICE_NAME_TAKEN"
    assert not any(call[:2] in (["pactl", "load-module"], ["pactl", "unload-module"]) for call in fake.calls)


def test_voice_drops_audio_of_older_epochs_on_flush(tmp_path):
    voice = capture_linux.LinuxVoice(linux_backend(tmp_path, FakePactl(), {"pacat"}, []), 24000)
    voice.play(b"\x01\x00" * 4, 24000, 1)
    voice.play(b"\x02\x00" * 4, 24000, 1)
    voice.flush(2)
    assert list(voice._queue) == [] and voice.stats["dropped_bytes"] == 16
    voice.play(b"\x03\x00" * 4, 24000, 1)  # a frame of the old epoch still in flight
    voice.play(b"\x04\x00" * 4, 24000, 2)
    voice.play(b"\x05\x00" * 4, 16000, 2)  # wrong rate is never played
    assert [item[0] for item in voice._queue] == [2]


def test_linux_capture_stop_is_final(tmp_path):
    fake, procs = FakePactl(sink_inputs=SINK_INPUTS), []
    backend = linux_backend(tmp_path, fake, {"parec", "pactl"}, procs)
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps(), "zoom"))
    capture.stop()  # the call ended before the capture started
    with pytest.raises(capture_common.CaptureFailed) as stopped:
        capture.start(lambda *a: None, lambda reason: None)
    assert stopped.value.code == "MEETING_CAPTURE_STOPPED" and procs == []


def test_linux_pipewire_falls_back_to_the_sink_monitor_and_says_so_after_start(tmp_path):
    graph = [obj for obj in PW_GRAPH] + [
        {"id": 30, "type": "PipeWire:Interface:Metadata", "props": {"metadata.name": "default"},
         "metadata": [{"subject": 0, "key": "default.audio.sink", "value": {"name": "alsa_output.analog"}}]},
    ]
    fake, procs = FakePactl(pipewire=True, graph=graph, link_ok=False), []
    backend = linux_backend(tmp_path, fake, {"pw-record", "pw-dump", "pw-link"}, procs)
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    assert capture.disclosures() == []  # before start: application streams are still the plan
    capture.start(lambda *a: None, lambda reason: None)
    try:
        assert len(procs) == 2 and procs[0].returncode is not None  # the linked recorder was stopped
        assert procs[1].argv[3] == "alsa_output.analog"
        assert json.loads(procs[1].argv[13])["stream.capture.sink"] is True
        assert capture.describe()["source"] == "device monitor"
        [note] = capture.disclosures()
        assert "everything playing on alsa_output.analog" in note
    finally:
        capture.stop()


def test_module_records_survive_a_silent_audio_server(tmp_path):
    null_args = "sink_name=ouroboros_voice sink_properties=device.description=Ouroboros-voice"
    record = {"modules": [{"id": 25, "name": "module-null-sink", "argument": null_args}], "pid": 1}
    (tmp_path / capture_linux.MODULES_FILE).write_text(json.dumps(record), encoding="utf-8")
    fake = FakePactl(modules={25: ("module-null-sink", null_args)})
    fake.server_down = True
    backend = linux_backend(tmp_path, fake, {"pactl"}, [])
    assert backend.cleanup_leftovers() == []
    assert json.loads((tmp_path / capture_linux.MODULES_FILE).read_text(encoding="utf-8")) == record
    fake.server_down = False
    assert backend.cleanup_leftovers() == [25] and fake.modules == {}


def test_a_refused_unload_stays_recorded_and_the_next_start_heals_it(tmp_path):
    fake, procs = FakePactl(), []
    backend = linux_backend(tmp_path, fake, {"pactl", "pacat"}, procs)
    voice = backend.open_voice(48000)
    voice.start()
    sink_id, source_id = sorted(fake.modules)
    fake.unload_refused = {source_id}
    assert voice.close() == [sink_id]
    kept = json.loads((tmp_path / capture_linux.MODULES_FILE).read_text(encoding="utf-8"))["modules"]
    assert [entry["id"] for entry in kept] == [source_id]
    fake.unload_refused = set()
    healed = backend.open_voice(48000)
    healed.start()  # its own recorded leftover goes first (it still carries our name), then fresh modules
    assert source_id not in fake.modules and len(fake.modules) == 2
    healed.close()


def test_a_dead_player_is_reported_once_and_stops_taking_audio(tmp_path):
    class BrokenPipe(__import__("io").BytesIO):
        def write(self, data):
            raise BrokenPipeError("the player exited")

    procs, errors = [], []

    def popen(argv, **kwargs):
        proc = FakeProc(argv)
        proc.stdin, proc.returncode = BrokenPipe(), 1
        procs.append(proc)
        return proc

    backend = capture_linux.LinuxBackend(tmp_path, which=lambda name: f"/usr/bin/{name}",
                                         run=FakePactl().run, popen=popen, env={"PATH": "/usr/bin"})
    voice = backend.open_voice(24000)
    voice.start(on_error=errors.append)
    voice.play(b"\x01\x00" * 480, 24000, 0)
    assert until(lambda: errors)
    assert errors == ["the pacat player stopped (exit 1)"] and voice.failure == errors[0]
    voice.play(b"\x01\x00" * 480, 24000, 0)
    assert list(voice._queue) == [] and voice.stats["dropped_bytes"] == 960
    voice.close()


# ---- macOS backend (fake pyobjc) --------------------------------------------------------------

class FakeNSObject:
    def __init_subclass__(cls, protocols=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.protocols = protocols

    @classmethod
    def alloc(cls):
        return cls.__new__(cls)

    def init(self):
        return self


class FakeConfig(FakeNSObject):
    def __init__(self):
        self.calls = {}

    def init(self):
        self.calls = {}
        return self

    def __getattr__(self, name):
        if name.startswith("set") and name.endswith("_"):
            return lambda value: self.calls.__setitem__(name, value)
        raise AttributeError(name)


class FakeFilter(FakeNSObject):
    def initWithDisplay_includingApplications_exceptingWindows_(self, display, apps, windows):
        self.display, self.apps, self.windows = display, list(apps), list(windows)
        return self


class FakeStream(FakeNSObject):
    created = []

    def initWithFilter_configuration_delegate_(self, content_filter, config, delegate):
        self.filter, self.config, self.delegate, self.outputs, self.started = content_filter, config, delegate, [], False
        FakeStream.created.append(self)
        return self

    def addStreamOutput_type_sampleHandlerQueue_error_(self, output, kind, queue_, error):
        self.outputs.append((output, kind, queue_))
        return True, None

    def startCaptureWithCompletionHandler_(self, completion):
        self.started = True
        completion(None)

    def stopCaptureWithCompletionHandler_(self, completion):
        self.started = False
        completion(None)


class FakeSCApp:
    def __init__(self, bundle, pid):
        self._bundle, self._pid = bundle, pid

    def bundleIdentifier(self):
        return self._bundle

    def processID(self):
        return self._pid


class FakeContentSource:
    content = None
    error = None
    calls = 0

    @classmethod
    def getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(cls, a, b, completion):
        cls.calls += 1
        completion(cls.content, cls.error)


class FakeRunningApp:
    def __init__(self, name, bundle, pid, policy=0):
        self._v = (name, bundle, pid, policy)

    def localizedName(self):
        return self._v[0]

    def bundleIdentifier(self):
        return self._v[1]

    def processIdentifier(self):
        return self._v[2]

    def activationPolicy(self):
        return self._v[3]


class FakeAccess:
    def __init__(self, granted):
        self.granted_value, self.requests = granted, 0

    def granted(self):
        return self.granted_value

    def request(self):
        self.requests += 1
        return False


def fake_frameworks(running, copy_fills_buffer=False):
    FakeStream.created = []
    FakeContentSource.calls = 0
    FakeContentSource.content = SimpleNamespace(
        applications=lambda: [FakeSCApp("us.zoom.xos", 10), FakeSCApp("com.google.chrome", 11)],
        displays=lambda: ["display-1"])
    FakeContentSource.error = None
    capture_macos._HANDLER_CLASS.clear()
    workspace = SimpleNamespace(runningApplications=lambda: running)
    core_media = SimpleNamespace(
        CMTimeMake=lambda value, scale: (value, scale),
        CMSampleBufferGetDataBuffer=lambda buffer: buffer["block"],
        CMBlockBufferGetDataLength=lambda block: len(block),
        CMBlockBufferCopyDataBytes=(
            (lambda block, offset, length, out: out.__setitem__(slice(0, length), block[offset:offset + length]) or 0)
            if copy_fills_buffer else (lambda block, offset, length, out: (0, block[offset:offset + length]))),
        CMSampleBufferGetFormatDescription=lambda buffer: "description",
        CMAudioFormatDescriptionGetStreamBasicDescription=lambda description: SimpleNamespace(
            mSampleRate=48000.0, mFormatFlags=capture_macos.K_AUDIO_FORMAT_FLAG_IS_FLOAT | (1 << 5),
            mChannelsPerFrame=1))
    return SimpleNamespace(
        objc=SimpleNamespace(protocolNamed=lambda name: f"protocol:{name}"),
        AppKit=SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace)),
        Foundation=SimpleNamespace(NSObject=FakeNSObject),
        CoreMedia=core_media,
        SCK=SimpleNamespace(SCShareableContent=FakeContentSource, SCContentFilter=FakeFilter,
                            SCStreamConfiguration=FakeConfig, SCStream=FakeStream))


MAC_RUNNING = [FakeRunningApp("zoom.us", "us.zoom.xos", 10), FakeRunningApp("Google Chrome", "com.google.chrome", 11),
               FakeRunningApp("Dock helper", "com.apple.dock.extra", 12, policy=2)]


def test_macos_lists_regular_apps_and_requires_macos_13(tmp_path):
    backend = capture_macos.MacBackend(tmp_path, frameworks=fake_frameworks(MAC_RUNNING),
                                       access=FakeAccess(True), macos_version="14.5")
    assert [app.ident for app in backend.running_apps()] == ["us.zoom.xos", "com.google.chrome"]
    old = capture_macos.MacBackend(tmp_path, frameworks=fake_frameworks(MAC_RUNNING), access=FakeAccess(True),
                                   macos_version="12.7")
    with pytest.raises(capture_common.CaptureFailed) as failed:
        old.permission()
    assert failed.value.code == "MEETING_PLATFORM_UNSUPPORTED"


def test_macos_missing_consent_shows_the_prompt_once_then_points_to_settings(tmp_path):
    access = FakeAccess(False)
    backend = capture_macos.MacBackend(tmp_path, frameworks=fake_frameworks(MAC_RUNNING), access=access,
                                       macos_version="15.1")
    companion = meeting_capture.MeetingCompanion(tmp_path, "http://127.0.0.1:1", "token", backend)
    reply = companion.listen_start("call-1", "")
    assert reply["code"] == "MEETING_PERMISSION_PENDING" and "Screen & System Audio Recording" in reply["message"]
    assert access.requests == 1 and FakeContentSource.calls == 0
    again = companion.listen_start("call-1", "")
    assert again["code"] == "MEETING_PERMISSION_DENIED" and "shows its prompt only once" in again["message"]
    assert access.requests == 1  # macOS would not show the prompt again; no request pretends it does
    access.granted_value = True
    assert backend.permission() == ("granted", "")


def test_macos_capture_stop_is_final(tmp_path):
    fw = fake_frameworks(MAC_RUNNING)
    backend = capture_macos.MacBackend(tmp_path, frameworks=fw, access=FakeAccess(True), macos_version="14.5")
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    capture.stop()
    with pytest.raises(capture_common.CaptureFailed) as stopped:
        capture.start(lambda *a: None, lambda reason: None)
    assert stopped.value.code == "MEETING_CAPTURE_STOPPED" and FakeStream.created == []


@pytest.mark.parametrize("copy_fills_buffer", [False, True], ids=["returns-copy", "fills-buffer"])
def test_macos_capture_includes_only_the_meeting_app_and_converts_audio(tmp_path, copy_fills_buffer):
    from array import array

    fw = fake_frameworks(MAC_RUNNING, copy_fills_buffer)
    backend = capture_macos.MacBackend(tmp_path, frameworks=fw, access=FakeAccess(True), macos_version="14.5")
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    received, errors = [], []
    capture.start(lambda pcm, stream=0: received.append((pcm, stream)), errors.append)
    [stream] = FakeStream.created
    assert [app.bundleIdentifier() for app in stream.filter.apps] == ["us.zoom.xos"]
    assert stream.filter.display == "display-1" and stream.filter.windows == []
    assert stream.config.calls["setCapturesAudio_"] is True
    assert stream.config.calls["setExcludesCurrentProcessAudio_"] is True
    assert (stream.config.calls["setSampleRate_"], stream.config.calls["setChannelCount_"]) == (48000, 1)
    assert [kind for _, kind, _ in stream.outputs] == [capture_macos.SC_STREAM_OUTPUT_AUDIO,
                                                       capture_macos.SC_STREAM_OUTPUT_SCREEN]
    assert stream.delegate.protocols == ["protocol:SCStreamOutput", "protocol:SCStreamDelegate"]
    samples = array("f", [0.0, 0.5, -1.0]).tobytes()
    stream.delegate.stream_didOutputSampleBuffer_ofType_(stream, {"block": samples},
                                                         capture_macos.SC_STREAM_OUTPUT_AUDIO)
    stream.delegate.stream_didOutputSampleBuffer_ofType_(stream, {"block": samples},
                                                         capture_macos.SC_STREAM_OUTPUT_SCREEN)
    assert [array("h", pcm).tolist() for pcm, _ in received] == [[0, 16384, -32767]]
    assert errors == []
    capture.stop()
    assert stream.started is False and stream.delegate.sink is None


def test_macos_reports_an_app_screencapturekit_does_not_offer(tmp_path):
    fw = fake_frameworks(MAC_RUNNING)
    backend = capture_macos.MacBackend(tmp_path, frameworks=fw, access=FakeAccess(True), macos_version="14.5")
    capture = backend.open_capture(meeting_apps.Selection((RunningApp("Teams", "com.microsoft.teams2", 99),),
                                                         "Microsoft Teams", False))
    with pytest.raises(capture_common.CaptureFailed) as failed:
        capture.start(lambda *a: None, lambda reason: None)
    assert failed.value.code == "MEETING_APP_NOT_CAPTURABLE" and FakeStream.created == []


def test_capture_attribution_probe_classifies_without_prompting(tmp_path, monkeypatch):
    paths = {300: "/Applications/Ouroboros.app/Contents/Resources/python-standalone/bin/python3",
             200: "/Applications/Ouroboros.app/Contents/Resources/python-standalone/bin/python3",
             100: "/Applications/Ouroboros.app/Contents/MacOS/Ouroboros", 1: "/sbin/launchd"}
    parents = {300: 200, 200: 100, 100: 1}
    record = probe_capture.attribution(300, path_of=paths.get, parent_of=lambda pid: parents.get(pid, 0),
                                       responsible_of=lambda pid: 100, granted=lambda: False)
    assert [row["pid"] for row in record["chain"]] == [300, 200, 100, 1]
    assert record["responsible"] == {"pid": 100, "path": paths[100], "app_bundle": "/Applications/Ouroboros.app",
                                     "kind": "app_bundle"}
    assert record["preflight_granted"] is False
    assert probe_capture.classify("/usr/local/bin/python3.12") == "python"
    monkeypatch.setattr(probe_capture, "attribution", lambda: dict(record))
    monkeypatch.setattr(probe_capture.sys, "platform", "darwin")
    assert probe_capture.main(["--state-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / probe_capture.RECORD_FILE).read_text(encoding="utf-8"))
    assert saved["real_capture"] == "NOT_RUN"


# ---- Windows backend (fake PyAudioWPatch and process loopback) -------------------------------------

def test_process_loopback_structures_match_the_windows_abi():
    params, variant = capture_windows.activation_params(1234)
    assert ctypes.sizeof(capture_windows.AUDIOCLIENT_ACTIVATION_PARAMS) == 12
    assert ctypes.sizeof(capture_windows.WAVEFORMATEX) == 18
    assert ctypes.sizeof(capture_windows.PROPVARIANT) == (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16)
    assert (variant.vt, variant.blob.cbSize) == (capture_windows.VT_BLOB, 12)
    assert params.ProcessLoopbackParams.TargetProcessId == 1234
    assert params.ProcessLoopbackParams.ProcessLoopbackMode == 0  # include the target tree only
    assert bytes(capture_windows.IID_IAUDIO_CLIENT).hex() == "4cadb91cfadb324cb178c2f568a703b2"
    fmt = capture_windows.capture_format()
    assert (fmt.nSamplesPerSec, fmt.nChannels, fmt.wBitsPerSample, fmt.nBlockAlign) == (48000, 2, 16, 4)


def test_windows_lists_one_root_per_program_tree():
    rows = [(1000, 1, "chrome.exe"), (1001, 1000, "chrome.exe"), (1002, 1000, "chrome.exe"),
            (2000, 50, "Zoom.exe"), (4, 0, "System"), (3000, 2000, "zWebview2Agent.exe")]
    roots = capture_windows.root_processes(rows)
    assert [(app.ident, app.pid) for app in roots] == [("chrome.exe", 1000), ("zoom.exe", 2000),
                                                       ("zwebview2agent.exe", 3000)]
    assert meeting_apps.choose(roots).label == "Zoom"


class FakeLoopStream:
    created = []

    def __init__(self, pid, stream, failure=""):
        self.pid, self.stream, self._failure, self.failure = pid, stream, failure, ""
        self.started = self.stopped = False
        FakeLoopStream.created.append(self)

    def prepare(self):
        self.failure = self._failure

    def start(self, on_pcm, on_error):
        self.started = True

    def stop(self):
        self.stopped = True


class FakePyAudio:
    paInt16, paContinue = 8, 0
    instances = []

    class PyAudio:
        def __init__(self):
            self.opened, self.terminated = [], False
            FakePyAudio.instances.append(self)

        def get_default_wasapi_loopback(self):
            return {"index": 7, "name": "Speakers [Loopback]", "defaultSampleRate": 44100.0, "maxInputChannels": 2}

        def open(self, **kwargs):
            self.opened.append(kwargs)
            return SimpleNamespace(start_stream=lambda: None, stop_stream=lambda: None, close=lambda: None)

        def terminate(self):
            self.terminated = True


def windows_backend(tmp_path, *, build=22631, failure=""):
    FakePyAudio.instances = []
    return capture_windows.WindowsBackend(
        tmp_path, processes=lambda: [(2000, 50, "Zoom.exe"), (2001, 2000, "Zoom.exe")],
        pyaudio_loader=lambda: FakePyAudio, build=build,
        stream_factory=lambda pid, stream: FakeLoopStream(pid, stream, failure))


def test_windows_prefers_process_loopback_of_the_meeting_tree(tmp_path):
    backend = windows_backend(tmp_path)
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    assert isinstance(capture, capture_windows.ProcessLoopbackCapture)
    assert [(stream.pid, stream.stream) for stream in capture.streams] == [(2000, 0)]
    assert capture.describe()["includes_own_voice"] is False and capture.disclosures() == []
    assert not (tmp_path / capture_windows.ATTEMPT_FILE).exists()
    capture.start(lambda *a: None, lambda reason: None)
    capture.stop()
    assert capture.streams[0].started and capture.streams[0].stopped
    assert FakePyAudio.instances == []


def test_windows_falls_back_to_the_device_loopback_and_says_it_hears_the_whisper(tmp_path):
    from array import array

    FakeLoopStream.created = []
    backend = windows_backend(tmp_path, failure="activation refused (HRESULT 0x80070057)")
    capture = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    assert isinstance(capture, capture_windows.EndpointLoopbackCapture)
    assert [stream.stopped for stream in FakeLoopStream.created] == [True]  # its thread is released
    facts = capture.describe()
    assert facts["includes_own_voice"] is True and "0x80070057" in facts["fallback_reason"]
    assert capture.rate == 44100
    [note] = capture.disclosures()
    assert "Speakers [Loopback]" in note and "own whispered hints" in note
    received = []
    capture.start(lambda pcm, stream=0: received.append(pcm), lambda reason: None)
    [opened] = FakePyAudio.instances[0].opened
    assert (opened["format"], opened["channels"], opened["rate"], opened["input_device_index"],
            opened["start"]) == (8, 2, 44100, 7, False)
    assert opened["stream_callback"](array("h", [10, 30, -10, -30]).tobytes(), 2, None, 0) == (None, 0)
    assert array("h", received[0]).tolist() == [20, -20]
    capture.stop()
    assert FakePyAudio.instances[0].terminated


def test_windows_capture_stops_are_final(tmp_path):
    backend = windows_backend(tmp_path)
    process = backend.open_capture(meeting_apps.choose(backend.running_apps()))
    process.stop()
    with pytest.raises(capture_common.CaptureFailed):
        process.start(lambda *a: None, lambda reason: None)
    assert process.streams[0].started is False
    endpoint = windows_backend(tmp_path / "endpoint", failure="refused").open_capture(
        meeting_apps.choose(backend.running_apps()))
    endpoint.stop()
    endpoint.stop()  # idempotent: PyAudio is terminated once
    with pytest.raises(capture_common.CaptureFailed) as stopped:
        endpoint.start(lambda *a: None, lambda reason: None)
    assert stopped.value.code == "MEETING_CAPTURE_STOPPED" and FakePyAudio.instances[0].opened == []


def test_windows_crash_during_activation_turns_process_loopback_off(tmp_path):
    (tmp_path / capture_windows.ATTEMPT_FILE).write_text("attempt")
    backend = windows_backend(tmp_path)
    usable, reason = backend.process_loopback_state()
    assert usable is False and "ended the companion" in reason
    assert isinstance(backend.open_capture(meeting_apps.choose(backend.running_apps())),
                      capture_windows.EndpointLoopbackCapture)
    old = windows_backend(tmp_path / "old", build=19045)
    assert old.process_loopback_state()[0] is False
