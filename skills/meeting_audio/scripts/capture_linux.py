"""Linux meeting capture and speaking mode over PipeWire or PulseAudio, user space only.

Capture. With PipeWire, a ``pw-record`` node that links nowhere by itself is
connected with ``pw-link`` to the output ports of the meeting application's own
playback streams, so only that application is heard; new streams of the same
application are linked as they appear. Where that is not possible (no
``pw-link``, or linking refused) and on plain PulseAudio, the capture records the
monitor of the output device the application plays to, and says so, because that
monitor also carries everything else playing there.

Speaking (Linux only, never a driver). ``pactl`` loads a null sink and a remapped
source over its monitor: a microphone the owner can pick as "Ouroboros" in the
meeting application; ``pacat`` (or ``pw-play``) plays the call's output into it.
The module ids this companion loads are written to the state directory before
anything else happens, and only those ids, still carrying the same module name
and arguments, are ever unloaded: on stop, on exit, and at the next start after a
crash. Nothing is removed by name.

Every helper is an existing system binary; a missing one is a typed refusal that
names it. ``LC_ALL=C`` keeps their output parseable, and the systemd default
``/run/user/<uid>`` stands in for ``XDG_RUNTIME_DIR`` when the host did not pass it.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence

from capture_common import CHUNK_MS, CaptureFailed, Helper, OnError, OnPcm, PipeReader, stop_process
from meeting_apps import RunningApp, Selection, display_name

CAPTURE_RATE = 48000
NODE_NAME = "ouroboros_meeting_capture"
VOICE_SINK = "ouroboros_voice"
VOICE_SOURCE = "ouroboros_mic"
MODULES_FILE = "speak_modules.json"
# Waiting for our own capture node to appear in the graph, and the period of the
# re-link scan that picks up new streams: PipeWire graph timing bounds.
NODE_WAIT_SEC = 3.0
RELINK_SEC = 2.0
# Playback lead kept inside the player, so a flush silences within this bound; the queue
# bound is memory only: engines stream speech faster than real time, so it holds minutes.
VOICE_LEAD_SEC = 0.08
VOICE_QUEUE_SEC = 120.0
COMMAND_TIMEOUT_SEC = 5.0

Run = Callable[..., subprocess.CompletedProcess]


def audio_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The helpers' environment: the companion's own without the Host Service address and token.

    The token belongs to the companion process alone; no audio helper needs it.
    """
    env = {key: value for key, value in (os.environ if base is None else base).items()
           if not key.upper().startswith("HOST_SERVICE_")}
    env["LC_ALL"] = "C"
    if not env.get("XDG_RUNTIME_DIR") and hasattr(os, "getuid"):
        candidate = pathlib.Path(f"/run/user/{os.getuid()}")
        if candidate.is_dir():
            env["XDG_RUNTIME_DIR"] = str(candidate)
    return env


def _int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


# ---- PipeWire graph (pw-dump JSON) ------------------------------------------------

def pw_objects(text: str) -> List[Dict[str, Any]]:
    data = json.loads(text or "[]")
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _props(obj: Dict[str, Any]) -> Dict[str, Any]:
    info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
    return info.get("props") if isinstance(info.get("props"), dict) else {}


def _is_own(props: Dict[str, Any]) -> bool:
    names = (str(props.get("node.name") or ""), str(props.get("media.name") or ""),
             str(props.get("application.name") or ""))
    return any(name.lower().startswith("ouroboros") for name in names)


def pw_app_streams(objects: Sequence[Dict[str, Any]]) -> List[RunningApp]:
    apps = []
    for obj in objects:
        props = _props(obj)
        if obj.get("type") != "PipeWire:Interface:Node" or props.get("media.class") != "Stream/Output/Audio":
            continue
        if _is_own(props):
            continue
        apps.append(RunningApp(
            name=str(props.get("application.name") or props.get("node.name") or ""),
            ident=str(props.get("application.process.binary") or ""),
            pid=_int(props.get("application.process.id")),
            handle=str(obj.get("id")),
        ))
    return apps


def pw_ports(objects: Sequence[Dict[str, Any]], node_id: int, direction: str) -> List[int]:
    ports = []
    for obj in objects:
        info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
        if obj.get("type") != "PipeWire:Interface:Port" or info.get("direction") != direction:
            continue
        if _int(_props(obj).get("node.id")) == node_id:
            ports.append(_int(obj.get("id")))
    return sorted(ports)


def pw_links(objects: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    links = []
    for obj in objects:
        info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
        if obj.get("type") == "PipeWire:Interface:Link":
            links.append({key: _int(info.get(key)) for key in
                          ("output-node-id", "output-port-id", "input-node-id", "input-port-id")})
    return links


def pw_node(objects: Sequence[Dict[str, Any]], node_id: int) -> Dict[str, Any]:
    for obj in objects:
        if obj.get("type") == "PipeWire:Interface:Node" and _int(obj.get("id")) == node_id:
            return _props(obj)
    return {}


def pw_own_node(objects: Sequence[Dict[str, Any]], pid: int) -> int:
    for obj in objects:
        props = _props(obj)
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        if props.get("node.name") == NODE_NAME or (pid and _int(props.get("application.process.id")) == pid):
            return _int(obj.get("id"))
    return 0


def pw_default_sink(objects: Sequence[Dict[str, Any]]) -> str:
    for obj in objects:
        # pw-dump keeps a Metadata object's props at its top level, not under "info".
        props = obj.get("props") if isinstance(obj.get("props"), dict) else _props(obj)
        if obj.get("type") != "PipeWire:Interface:Metadata" or props.get("metadata.name") != "default":
            continue
        for entry in obj.get("metadata") or []:
            if isinstance(entry, dict) and entry.get("key") == "default.audio.sink":
                value = entry.get("value")
                if isinstance(value, dict) and value.get("name"):
                    return str(value["name"])
    return ""


# ---- PulseAudio text (pactl with LC_ALL=C) ----------------------------------------

def parse_sink_inputs(text: str) -> List[Dict[str, Any]]:
    """``pactl list sink-inputs`` blocks as {index, sink, props}."""
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    in_props = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if raw.startswith("Sink Input #"):
            current = {"index": _int(raw.split("#", 1)[1]), "sink": -1, "props": {}}
            entries.append(current)
            in_props = False
        elif current is None:
            continue
        elif line.startswith("Sink:") and not in_props:
            current["sink"] = _int(line.split(":", 1)[1])
        elif line == "Properties:":
            in_props = True
        elif in_props and " = " in line:
            key, value = line.split(" = ", 1)
            current["props"][key.strip()] = value.strip().strip('"')
        elif in_props and line and not raw.startswith(("\t\t", "        ")):
            in_props = False
    return entries


def parse_short(text: str) -> List[List[str]]:
    return [line.split("\t") for line in (text or "").splitlines() if line.strip()]


def pulse_server_name(info: str) -> str:
    """The ``Server Name`` of ``pactl info``; PipeWire's Pulse layer answers "PulseAudio (on PipeWire ...)"."""
    for line in (info or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "Server Name":
            return value.strip()
    return ""


class LinuxBackend:
    """Capture targets, capture and speaking on one Linux desktop session."""

    platform = "linux"
    speaking_supported = True

    def __init__(self, state_dir: pathlib.Path, *, which: Callable[[str], Optional[str]] = shutil.which,
                 run: Run = subprocess.run, popen: Callable[..., Any] = subprocess.Popen,
                 env: Optional[Dict[str, str]] = None):
        self.state_dir = pathlib.Path(state_dir)
        self.which = which
        self._run = run
        self.popen = popen
        self.env = audio_env(env)

    # -- helpers ------------------------------------------------------------------
    def run(self, argv: Sequence[str], timeout: float = COMMAND_TIMEOUT_SEC) -> subprocess.CompletedProcess:
        try:
            return self._run(list(argv), capture_output=True, text=True, timeout=timeout, env=self.env)
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(list(argv), 127, "", str(exc))

    def has(self, *names: str) -> bool:
        return all(self.which(name) for name in names)

    def stack(self) -> str:
        """``pipewire`` or ``pulse``: the server that really plays this desktop's sound.

        A PipeWire daemon also runs on desktops whose sound is PulseAudio's (Ubuntu
        22.04 starts it for screen sharing), so a working ``pw-dump`` says nothing
        about audio. The Pulse server names itself: PipeWire is used when PipeWire's
        own Pulse layer answers, or when no Pulse server answers at all; otherwise
        PulseAudio. A typed refusal names what is missing.
        """
        info = self.run(["pactl", "info"]) if self.has("pactl") else None
        pulse = pulse_server_name(info.stdout) if info is not None and info.returncode == 0 else None
        if (pulse is None or "on pipewire" in pulse.lower()) and self.has("pw-record", "pw-dump"):
            if self.run(["pw-dump"]).returncode == 0:
                return "pipewire"
        if pulse is not None and self.has("parec"):
            return "pulse"
        if info is not None and info.returncode != 0 and self.has("parec"):
            raise CaptureFailed("MEETING_AUDIO_SERVER_UNREACHABLE",
                                "Could not reach the desktop audio server: "
                                + ((info.stderr or "").strip()[-300:] or "pactl info failed"))
        missing = [name for name in ("pw-record", "pw-dump", "parec", "pactl") if not self.which(name)]
        raise CaptureFailed(
            "MEETING_AUDIO_TOOLS_MISSING",
            "Listening needs PipeWire's pw-record and pw-dump, or PulseAudio's parec and pactl; "
            f"not found: {', '.join(missing)}.",
        )

    def graph(self) -> List[Dict[str, Any]]:
        result = self.run(["pw-dump"])
        if result.returncode != 0:
            raise CaptureFailed("MEETING_AUDIO_SERVER_UNREACHABLE",
                                "pw-dump failed: " + (result.stderr or "").strip()[-300:])
        try:
            return pw_objects(result.stdout)
        except ValueError:
            raise CaptureFailed("MEETING_AUDIO_SERVER_UNREACHABLE", "pw-dump output was unreadable") from None

    def sink_inputs(self) -> List[Dict[str, Any]]:
        result = self.run(["pactl", "list", "sink-inputs"])
        if result.returncode != 0:
            raise CaptureFailed("MEETING_AUDIO_SERVER_UNREACHABLE",
                                "pactl list sink-inputs failed: " + (result.stderr or "").strip()[-300:])
        return parse_sink_inputs(result.stdout)

    # -- the backend surface the companion uses -----------------------------------
    def permission(self) -> tuple:
        return "granted", ""

    def request_permission(self) -> None:
        return None

    def running_apps(self) -> List[RunningApp]:
        """Applications playing sound right now (the meeting app plays while in a call)."""
        if self.stack() == "pipewire":
            return pw_app_streams(self.graph())
        apps = []
        for entry in self.sink_inputs():
            props = entry["props"]
            if _is_own(props):
                continue
            apps.append(RunningApp(name=str(props.get("application.name") or ""),
                                   ident=str(props.get("application.process.binary") or ""),
                                   pid=_int(props.get("application.process.id")),
                                   handle=str(entry["index"])))
        return apps

    def open_capture(self, selection: Selection) -> "LinuxCapture":
        return LinuxCapture(self, selection, self.stack())

    def cleanup_leftovers(self) -> List[int]:
        return VirtualMicrophone(self).cleanup()

    def open_voice(self, rate: int) -> "LinuxVoice":
        return LinuxVoice(self, rate)

    def run_main_loop(self, stop_event: threading.Event) -> None:
        stop_event.wait()


class LinuxCapture:
    """One capture of the selected applications at ``CAPTURE_RATE`` mono.

    ``stop`` is final and waits for a ``start`` in progress, so a call that ends
    while the capture is starting never leaves a recorder running.
    """

    rate = CAPTURE_RATE

    def __init__(self, backend: LinuxBackend, selection: Selection, stack: str):
        self.backend = backend
        self.selection = selection
        self.stack = stack
        self.source = "application streams" if stack == "pipewire" and backend.has("pw-link") else "device monitor"
        self.device = ""
        self._helper: Optional[Helper] = None
        self._reader: Optional[PipeReader] = None
        self._linked: set = set()
        self._stop = threading.Event()
        self._relinker: Optional[threading.Thread] = None
        self._node = 0
        self._lifecycle = threading.Lock()
        self._closed = False

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.stack, "source": self.source, "device": self.device,
                "includes_own_voice": self.source == "device monitor" or self.selection.browser}

    def disclosures(self) -> List[str]:
        if self.source == "device monitor":
            where = self.device or "the default output device"
            return [f"This meeting audio is everything playing on {where}, not only "
                    f"{self.selection.label}; if Ouroboros's own whispered hints play there too, they are "
                    "in it, and speech in Ouroboros's own voice here is not a meeting participant."]
        return []

    # -- start/stop ---------------------------------------------------------------
    def start(self, on_pcm: OnPcm, on_error: OnError) -> None:
        with self._lifecycle:
            if self._closed:
                raise CaptureFailed("MEETING_CAPTURE_STOPPED", "listening stopped before the capture started")
            if self.source == "application streams":
                try:
                    self._start_linked(on_pcm, on_error)
                    return
                except CaptureFailed:
                    self._teardown()
                    self.source = "device monitor"
            self._start_monitor(on_pcm, on_error)

    def _record_argv(self, target: str, properties: Dict[str, Any]) -> List[str]:
        return ["pw-record", "--raw", "--target", target, "--rate", str(self.rate), "--channels", "1",
                "--format", "s16", "--latency", f"{CHUNK_MS}ms", "-P", json.dumps(properties), "-"]

    def _start_linked(self, on_pcm: OnPcm, on_error: OnError) -> None:
        argv = self._record_argv("0", {"node.name": NODE_NAME, "media.name": "Ouroboros meeting capture"})
        self._helper = Helper(argv, popen=self.backend.popen, env=self.backend.env)
        deadline = time.monotonic() + NODE_WAIT_SEC
        while not self._node:
            self._node = pw_own_node(self.backend.graph(), self._helper.proc.pid)
            if self._node or time.monotonic() > deadline:
                break
            time.sleep(0.1)
        if not self._node:
            raise CaptureFailed("MEETING_CAPTURE_FAILED", "the capture node did not appear in PipeWire")
        if not self._link_new():
            raise CaptureFailed("MEETING_CAPTURE_FAILED",
                                f"could not link {self.selection.label}'s audio to the capture")
        self._reader = PipeReader(self._helper, self.rate, on_pcm, on_error, name="pw-record")
        self._reader.start()
        self._relinker = threading.Thread(target=self._relink_loop, args=(self._stop,), name="meeting-relink",
                                          daemon=True)
        self._relinker.start()

    def _matches(self, app: RunningApp) -> bool:
        wanted = {display_name(chosen).lower() for chosen in self.selection.apps}
        idents = {chosen.ident.lower() for chosen in self.selection.apps if chosen.ident}
        return display_name(app).lower() in wanted or (bool(app.ident) and app.ident.lower() in idents)

    def _link_new(self) -> int:
        objects = self.backend.graph()
        inputs = pw_ports(objects, self._node, "input")
        for app in pw_app_streams(objects):
            if not self._matches(app):
                continue
            for out_port in pw_ports(objects, _int(app.handle), "output"):
                for in_port in inputs:
                    if (out_port, in_port) in self._linked:
                        continue
                    result = self.backend.run(["pw-link", str(out_port), str(in_port)])
                    if result.returncode == 0:
                        self._linked.add((out_port, in_port))
        return len(self._linked)

    def _relink_loop(self, stop: threading.Event) -> None:
        while not stop.wait(RELINK_SEC):
            try:
                self._link_new()
            except CaptureFailed:
                continue

    def _monitor_target(self) -> str:
        """The output device the selected application plays to (or the default one)."""
        if self.stack == "pipewire":
            objects = self.backend.graph()
            handles = {_int(app.handle) for app in self.selection.apps}
            for link in pw_links(objects):
                if link["output-node-id"] in handles:
                    props = pw_node(objects, link["input-node-id"])
                    if props.get("node.name"):
                        self.device = str(props.get("node.description") or props["node.name"])
                        return str(props["node.name"])
            default = pw_default_sink(objects)
            self.device = default
            return default
        sinks = {_int(row[0]): row[1] for row in parse_short(self.backend.run(
            ["pactl", "list", "sinks", "short"]).stdout) if len(row) > 1}
        handles = {_int(app.handle) for app in self.selection.apps}
        for entry in self.backend.sink_inputs():
            if entry["index"] in handles and entry["sink"] in sinks:
                self.device = sinks[entry["sink"]]
                return f"{sinks[entry['sink']]}.monitor"
        self.device = "the default output device"
        return "@DEFAULT_MONITOR@"

    def _start_monitor(self, on_pcm: OnPcm, on_error: OnError) -> None:
        target = self._monitor_target()
        if self.stack == "pipewire":
            if not target:
                raise CaptureFailed("MEETING_CAPTURE_FAILED", "PipeWire reports no output device to listen to")
            argv = self._record_argv(target, {"stream.capture.sink": True, "media.name": "Ouroboros meeting capture"})
        else:
            argv = ["parec", "--raw", f"--device={target}", "--format=s16le", f"--rate={self.rate}",
                    "--channels=1", f"--latency-msec={CHUNK_MS}", "--client-name=Ouroboros meeting capture"]
        self._helper = Helper(argv, popen=self.backend.popen, env=self.backend.env)
        self._reader = PipeReader(self._helper, self.rate, on_pcm, on_error, name="monitor")
        self._reader.start()

    def stop(self) -> None:
        with self._lifecycle:
            self._closed = True
            self._teardown()

    def _teardown(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.stop()
        if self._helper is not None:
            self._helper.stop()
        if self._reader is not None:
            self._reader.join()
        self._helper, self._reader, self._node = None, None, 0
        self._linked.clear()
        self._stop = threading.Event()  # the next attempt's relinker gets its own stop


class VoiceFailed(CaptureFailed):
    """Speaking mode could not start; nothing it created is left behind."""


class VirtualMicrophone:
    """The user-space "Ouroboros" microphone: two pactl modules tracked by their own ids.

    A record leaves ``speak_modules.json`` only once its module is unloaded, or once
    the audio server shows that id no longer carries that module (a restarted server
    reuses ids). When the server does not answer, the record is kept for the next try,
    so a module this companion created is never forgotten while it may still exist.
    """

    def __init__(self, backend: LinuxBackend):
        self.backend = backend
        self.path = backend.state_dir / MODULES_FILE
        self.records: List[Dict[str, Any]] = []

    def _persist(self) -> None:
        if not self.records:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps({"modules": self.records, "pid": os.getpid()}), encoding="utf-8")
        os.replace(str(tmp), str(self.path))

    def _names_taken(self) -> List[str]:
        taken = []
        for kind, name in (("sinks", VOICE_SINK), ("sources", VOICE_SOURCE)):
            rows = parse_short(self.backend.run(["pactl", "list", kind, "short"]).stdout)
            if any(len(row) > 1 and row[1] == name for row in rows):
                taken.append(name)
        return taken

    def _load(self, module: str, args: List[str]) -> int:
        result = self.backend.run(["pactl", "load-module", module, *args])
        module_id = _int(result.stdout) if result.returncode == 0 else 0
        if not module_id and (result.stdout or "").strip() != "0":
            raise VoiceFailed("MEETING_VOICE_FAILED", f"pactl could not load {module}: "
                              + ((result.stderr or "").strip()[-300:] or f"exit {result.returncode}"))
        self.records.append({"id": module_id, "name": module, "argument": " ".join(args)})
        self._persist()
        return module_id

    def create(self) -> None:
        if not self.backend.has("pactl"):
            raise VoiceFailed("MEETING_AUDIO_TOOLS_MISSING",
                              "Speaking mode needs pactl (PulseAudio or PipeWire's pipewire-pulse); not found.")
        self.cleanup()  # this companion's own leftovers go first, matched by id, name and arguments
        taken = self._names_taken()
        if taken:
            raise VoiceFailed(
                "MEETING_VOICE_NAME_TAKEN",
                f"An audio device named {', '.join(taken)} already exists and was not created by this run; "
                "it is left untouched. Remove it (pactl list modules short; pactl unload-module <id>) "
                "to let speaking mode create its own.")
        try:
            self._load("module-null-sink", [f"sink_name={VOICE_SINK}",
                                            "sink_properties=device.description=Ouroboros-voice"])
            self._load("module-remap-source", [f"master={VOICE_SINK}.monitor", f"source_name={VOICE_SOURCE}",
                                               "source_properties=device.description=Ouroboros"])
        except VoiceFailed:
            self.remove()
            raise

    def remove(self) -> List[int]:
        """Unload this companion's modules, newest first; a failed unload stays recorded."""
        unloaded = []
        for record in list(reversed(self.records)):
            if self.backend.run(["pactl", "unload-module", str(record["id"])]).returncode == 0:
                unloaded.append(record["id"])
                self.records.remove(record)
                self._persist()
        return unloaded

    def cleanup(self) -> List[int]:
        """Unload what an earlier run recorded and left, if those ids are still those modules."""
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8")).get("modules") or []
        except (OSError, ValueError, AttributeError):
            saved = []
        saved = [record for record in saved if isinstance(record, dict)]
        if not saved or not self.backend.has("pactl"):
            return []
        listing = self.backend.run(["pactl", "list", "modules", "short"])
        if listing.returncode != 0:  # the server did not answer: keep every record for the next try
            return []
        loaded = {}
        for row in parse_short(listing.stdout):
            if len(row) >= 2:
                loaded[_int(row[0])] = (row[1], " ".join((row[2] if len(row) > 2 else "").split()))
        self.records = [record for record in saved
                        if loaded.get(_int(record.get("id"))) == (record.get("name"),
                                                                  " ".join(str(record.get("argument") or "").split()))]
        self._persist()  # ids that no longer carry our modules are dropped here
        return self.remove()


class LinuxVoice:
    """Speaking mode: the virtual microphone plus a paced player of the call's output."""

    def __init__(self, backend: LinuxBackend, rate: int):
        self.backend = backend
        self.rate = int(rate)
        self.mic = VirtualMicrophone(backend)
        self.epoch: Optional[int] = None
        self._helper: Optional[Helper] = None
        self._queue: deque = deque()
        self._queued = 0  # bytes in the queue
        self._cond = threading.Condition()
        self._stopping = False
        self._writer: Optional[threading.Thread] = None
        self._on_error: Optional[OnError] = None
        self._player = ""
        self.failure = ""
        self.stats = {"played_bytes": 0, "dropped_bytes": 0}

    def player_argv(self) -> List[str]:
        if self.backend.has("pacat"):
            return ["pacat", "--playback", "--raw", f"--device={VOICE_SINK}", "--format=s16le",
                    f"--rate={self.rate}", "--channels=1", "--latency-msec=60",
                    "--client-name=Ouroboros voice"]
        if self.backend.has("pw-play"):
            return ["pw-play", "--raw", "--target", VOICE_SINK, "--rate", str(self.rate), "--channels", "1",
                    "--format", "s16", "--latency", "60ms", "-"]
        raise VoiceFailed("MEETING_AUDIO_TOOLS_MISSING", "Speaking mode needs pacat or pw-play; neither was found.")

    def start(self, on_error: Optional[OnError] = None) -> None:
        """Create the microphone and its player; ``on_error`` hears it if the player dies."""
        self._on_error = on_error
        argv = self.player_argv()
        self._player = argv[0]
        self.mic.create()
        try:
            self._helper = Helper(argv, popen=self.backend.popen, env=self.backend.env, stdin=True)
        except OSError as exc:
            self.mic.remove()
            raise VoiceFailed("MEETING_VOICE_FAILED", f"could not start {argv[0]}: {exc}") from None
        self._writer = threading.Thread(target=self._write_loop, name="meeting-voice", daemon=True)
        self._writer.start()

    def play(self, pcm: bytes, rate: int, epoch: int) -> None:
        if rate != self.rate or self.failure:
            self.stats["dropped_bytes"] += len(pcm)
            return
        with self._cond:
            if self.epoch is not None and epoch != self.epoch:
                self.stats["dropped_bytes"] += len(pcm)
                return
            self._queue.append((epoch, pcm))
            self._queued += len(pcm)
            limit = int(self.rate * VOICE_QUEUE_SEC) * 2
            while self._queued > limit and len(self._queue) > 1:
                dropped = len(self._queue.popleft()[1])
                self._queued -= dropped
                self.stats["dropped_bytes"] += dropped
            self._cond.notify_all()

    def flush(self, epoch: int) -> None:
        with self._cond:
            self.epoch = epoch
            kept = deque(item for item in self._queue if item[0] == epoch)
            remaining = sum(len(item[1]) for item in kept)
            self.stats["dropped_bytes"] += self._queued - remaining
            self._queue, self._queued = kept, remaining
            self._cond.notify_all()

    def _write_loop(self) -> None:
        started, written = 0.0, 0
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    if not self._cond.wait(0.5):
                        started = 0.0
                        if self._helper is not None and self._helper.proc.poll() is not None:
                            break  # the player died while idle
                if self._stopping:
                    return
                item = self._queue.popleft() if self._queue else None
                if item is not None:
                    self._queued -= len(item[1])
            if item is None:
                self._player_died()
                return
            epoch, pcm = item
            if not started:
                started, written = time.monotonic(), 0
            ahead = written / self.rate - (time.monotonic() - started)
            if ahead > VOICE_LEAD_SEC:
                with self._cond:
                    self._cond.wait(ahead - VOICE_LEAD_SEC)
                    if self._stopping:
                        return
                    if self.epoch is not None and epoch != self.epoch:
                        continue
            elif ahead < -VOICE_LEAD_SEC:
                started, written = time.monotonic(), 0
            try:
                stream = self._helper.proc.stdin
                stream.write(pcm)
                stream.flush()
            except (OSError, ValueError, AttributeError):
                self._player_died()
                return
            written += len(pcm) // 2
            self.stats["played_bytes"] += len(pcm)

    def _player_died(self) -> None:
        with self._cond:
            if self._stopping:
                return
            helper = self._helper
            code = helper.proc.poll() if helper is not None else None
            tail = helper.tail() if helper is not None else ""
            self.failure = f"the {self._player} player stopped (exit {code}){': ' + tail if tail else ''}"
            self._queue.clear()
            self._queued = 0
        if self._on_error is not None:
            self._on_error(self.failure)

    def close(self) -> List[int]:
        with self._cond:
            self._stopping = True
            self._queue.clear()
            self._queued = 0
            self._cond.notify_all()
        if self._writer is not None and self._writer is not threading.current_thread():
            self._writer.join(timeout=2.0)
        if self._helper is not None:
            stop_process(self._helper.proc)
        return self.mic.remove()
