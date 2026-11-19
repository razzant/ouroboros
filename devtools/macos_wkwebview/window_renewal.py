#!/usr/bin/env python3
"""An ordinary Restart and Update in the real macOS launcher window, a baseline build beside a candidate.

See window_renewal.md for the frozen two-arm, keyless workflow and its evidence boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a script under -I: this instrument's own checkout
    sys.path.insert(0, str(REPO))

from devtools.macos_wkwebview import arm_source  # noqa: E402
from devtools.macos_wkwebview.arm_source import GIT_IDENTITY, sha256  # noqa: E402
from devtools.macos_wkwebview.native_window import (  # noqa: E402
    CAPTURE_BRIDGE, OSLOG_RESULTS, RoomWindow, launcher_channel, launcher_diagnostics, public_reason, same_view, same_window, visible_key_view)

SCHEMA = "ouroboros.macos-window-renewal/6"
EXIT_CODES = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3, "UNSUPPORTED": 4}
TASKS = ("live", "paused")
FIXTURE_BUTTON = "Open the sample page"
TRANSITIONS = ("restart", "update", "handoff")
PROJECT_UPDATES = ("update", "handoff")
PROJECT_CHECKS = ("document_ready", "project_active_before", "project_active_after", "project_editable",
                  "project_reply_restored", "project_reattach_notice", "project_typed", "project_typed_pixels",
                  "project_window_typed_pixels", "project_window_pixels", "project_sent_once")
COMMON = ("generation_new", "client_health", "view_new", "installed", "live_connected", "peer_visible", "old_draft", "draft_restored",
          "reply_restored", "project_draft", "reattach_notice", "file_count_only", "attachment_restaged", "pending_owner",
          "no_replay", "accepted_once", "wire_once", "typed", "typed_pixels", "window_typed_pixels", "window_pixels",
          "live_continued", "pause_kept")
PHASES = {
    "restart": COMMON + ("hidden_kept", "reopened", "size_kept", "zoom_kept", "confirm_cancelled", "diagnostic_lookup_recorded"),
    "update": COMMON + PROJECT_CHECKS + ("update_landed", "fullscreen_kept", "reopened", "size_kept", "zoom_kept", "confirm_cancelled", "diagnostic_lookup_recorded"),
    "handoff": COMMON + PROJECT_CHECKS + ("update_landed", "source_loaded"),
    "session": ("live_finished", "ordinary_quit", "draft_gone", "no_send_after_quit", "pause_kept_after_quit", "help_after_quit"),
    "fixture": ("generation_new", "installed", "view_new", "sample_document_opened", "menu_drives_view"),
}
CONTROL = ("restart.view_new", "update.view_new", "fixture.view_new")
SAME_OPERATION = tuple(p + ".generation_new" for p in (*TRANSITIONS, "fixture")) + (
    "restart.installed", "update.installed", "fixture.installed", "update.project_active_before", "update.update_landed",
    "handoff.project_active_before", "handoff.update_landed", "handoff.source_loaded")
CALIBRATED = tuple(f"{phase}.{name}" for phase in TRANSITIONS for name in (
    "client_health", "live_connected", "peer_visible", "typed", "typed_pixels", "window_typed_pixels", "window_pixels", "live_continued", "pause_kept"))
# --- judgement (pure) ---------------------------------------------------------------------------------------------

def flatten(arm: dict) -> dict:
    checks = arm.get("checks") or {}
    return {f"{phase}.{name}": (checks.get(phase) or {}).get(name) for phase, names in PHASES.items() for name in names}


def judge(report: dict) -> dict:
    """PASS only when every candidate check holds and the baseline, after the same operations, kept its view."""
    arms = report.get("arms") or {}
    unsupported = [reason for reason in [report.get("unsupported")] + [arm.get("unsupported") for arm in arms.values()]
                   if reason]
    if unsupported:
        return {"verdict": "UNSUPPORTED", "failures": [], "gaps": unsupported, "comparison": {}}
    base, cand = flatten(arms.get("baseline") or {}), flatten(arms.get("candidate") or {})
    failures, gaps = [], []
    for key, value in cand.items():
        if value is None:
            gaps.append(f"candidate {key} was not established")
        elif value is False and key in CALIBRATED and base[key] is not True:
            gaps.append(f"candidate {key} is false, but the baseline's probe did not hold either ({base[key]!r})")
        elif value is False:
            failures.append(f"candidate {key} is false (baseline: {base[key]!r})")
    gaps += [f"baseline {key} is {base[key]!r}, not False: this run cannot tell a kept view from a new one"
             for key in CONTROL if base[key] is not False]
    gaps += [f"baseline {key} is {base[key]!r}: the arms did not run the same operation"
             for key in SAME_OPERATION if base[key] is not True]
    gaps += [f"baseline {key} was not established" for key in CALIBRATED if base[key] is not True]
    for name, arm in arms.items():
        gaps += [f"{name}: {gap}" for gap in arm.get("gaps", [])]
    gaps += list(report.get("gaps", []))
    return {"verdict": "FAIL" if failures else "INCONCLUSIVE" if gaps else "PASS", "failures": failures,
            "gaps": gaps, "comparison": {key: {"baseline": base[key], "candidate": cand[key]} for key in cand}}


def fixture_page(port: int, tag: str, page_path: str | None = None) -> str:
    path = page_path or f"http://127.0.0.1:{port}/p-{tag}"
    target = f"{path}?k-{tag}=q-{tag}#f-{tag}"
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Window renewal navigation"
            "</title><style>body{font:18px system-ui;margin:48px}button{font-size:20px;padding:16px 28px}"
            "</style></head><body><h1>Window renewal navigation</h1>"
            f"<button type=\"button\" onclick=\"this.disabled=true;location.assign('{target}')\">"
            f"{FIXTURE_BUTTON}</button></body></html>\n")


# --- host, files and processes ---------------------------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def command(*argv: str) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def ps(pid: int, column: str) -> str:
    return command("/bin/ps", "-o", f"{column}=", "-p", str(pid)) or ""


def alive(generation: dict) -> bool | None:
    """A zombie, a missing PID or a different birth ended this generation. A failed read is unknown."""
    done = subprocess.run(["/bin/ps", "-o", "stat=,lstart=", "-p", str(generation["pid"])],
                          capture_output=True, text=True, timeout=5)
    if done.returncode == 1 and not done.stdout.strip() and not done.stderr.strip():
        return False
    parts = done.stdout.strip().split(None, 1)
    if done.returncode or len(parts) != 2:
        return None
    return not parts[0].startswith("Z") and parts[1].strip() == generation["birth"]


def fixture_ports():
    with socket.socket() as main, socket.socket() as host:
        for probe in (main, host):
            probe.bind(("127.0.0.1", 0))
        return main.getsockname()[1], host.getsockname()[1]


def http(port: int, method: str, route: str, body=None, timeout: float = 15.0):
    """One owner HTTP request from this separate client: (status, parsed JSON or None); 0 when nothing answered."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - this run's own loopback core
            status, raw = response.status, response.read(16 << 20)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(1 << 20)
    except (OSError, ValueError, HTTPException):
        return 0, None
    try:
        return status, json.loads(raw or b"null")
    except ValueError:
        return status, None


def wait_for(read, seconds: float, interval: float = 0.5):
    deadline = time.monotonic() + seconds
    while True:
        value = read()
        if value or time.monotonic() >= deadline:
            return value
        time.sleep(interval)


def host_facts() -> dict:
    facts = {"platform": sys.platform, "macOS": command("/usr/bin/sw_vers", "-productVersion") if sys.platform == "darwin" else None,
             "python": sys.version.split()[0]}
    try:
        from importlib.metadata import version
        facts["pywebview"] = version("pywebview")
    except Exception:  # noqa: BLE001 - a fact, never a reason to stop
        facts["pywebview"] = None
    return facts


def instrument_identity() -> dict:
    names = ("window_renewal.py", "arm_source.py", "native_window.py", "composer_clearance.py")
    return {**{name: sha256(Path(__file__).with_name(name)) for name in names}, CAPTURE_BRIDGE.name: sha256(CAPTURE_BRIDGE)}


# --- loopback model ------------------------------------------------------------------------------------------------

def journey_model(workspaces: dict, tag: str, binding: Path):
    """The keyless loopback model (tests/system_e2e/harness.py): each tagged task reads its probe file for real,
    then every later request of that task is held until released, so a restart cuts the round and the restored
    worker sends the same saved read again. Untagged calls (Main) answer at once with the words to reply to."""
    from tests.system_e2e.harness import ScriptedStubModel, classify_call, default_lineage_binder, message_text, scripted_completion

    class JourneyModel(ScriptedStubModel):
        def __init__(self):
            self.tokens = {key: f"{key.upper()}-{tag}" for key in TASKS}
            for key, token in self.tokens.items():
                (workspaces[key] / f"probe-{key}.txt").write_text(token + "\n", encoding="utf-8")
            self.held = {key: {} for key in TASKS}
            self.read_requests = dict.fromkeys(TASKS, 0)
            self.release = {key: threading.Event() for key in TASKS}
            self.count_lock = threading.Lock()
            super().__init__(gate=self._gate, final_answer=f"QUOTE-{tag}: a synthetic answer to reply to.", select_retained=True)

        def _key(self, body):
            if classify_call(body) not in ("agent", "finalization"):
                return ""
            for key in TASKS:
                if self._read(body, key):
                    return key
            key = default_lineage_binder(body)
            return key if key in TASKS else ""

        def _read(self, body, key):
            results = "\n".join(message_text(m) for m in body.get("messages", []) if m.get("role") == "tool")
            return self.tokens[key] in results

        def _gate(self, body):
            key = self._key(body)
            if key and self._read(body, key):
                try:
                    server = json.loads(binding.read_text())["main"]
                    generation = (int(server["pid"]), server["fingerprint"]["start_time"])
                except (OSError, ValueError, KeyError, TypeError):
                    generation = None
                with self.count_lock:
                    counts = self.held[key]
                    counts[generation] = counts.get(generation, 0) + 1
                if not self.release[key].wait(3600):
                    raise TimeoutError(f"the {key} request was never released")

        def _answer(self, body, seq):
            key = self._key(body)
            if not key:
                return super()._answer(body, seq)
            def next_read(_):
                if self._read(body, key):
                    return None
                with self.count_lock:
                    self.read_requests[key] += 1
                return {"tool": "read_file", "arguments": {"root": "active_workspace", "path": f"probe-{key}.txt"}}
            return scripted_completion(body, seq, next_read,
                                       f"Finished: {self.tokens[key]}", select_retained=True)

        def count(self, key, generation=None):
            with self.count_lock:
                if generation is None:
                    return sum(self.held[key].values())
                return self.held[key].get((generation["pid"], generation["birth"]), 0)

    return JourneyModel()


# --- one arm: real launcher sessions, their transitions and checks -------------------------------------------------

FRAME_FIELDS = ("type", "content", "client_message_id", "sender_session_id", "chat_id", "project_id", "reply_to")


def same_frame(wire, entries):
    if not wire or not wire.get("client_message_id") or not wire.get("sender_session_id"):
        return None
    frames = [entry["frame"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("frame"), dict)
              and entry["frame"].get("client_message_id") == wire["client_message_id"]]
    return len(frames) == 1 and {k: frames[0][k] for k in FRAME_FIELDS if k in frames[0]} == {
        k: wire[k] for k in FRAME_FIELDS if k in wire}


def receipt(history, wire, *, room=None):
    """Count both a repeat of the original id and a repeat under a new id. Never infer identity from text."""
    if not wire or not wire.get("client_message_id") or not wire.get("sender_session_id") or not history or (
            history.get("window") or {}).get("complete") is not True:
        return None
    rows = [r for r in history["messages"] if r.get("role") == "user"
            and (r.get("client_message_id") == wire["client_message_id"] or r.get("text") == wire["content"])]
    if room is not None and len(rows) == 1:
        row = rows[0]
        if not all(row.get(k) for k in ("row_address", "client_message_id", "sender_session_id")) or row.get("ingress_accepted") is None:
            return None
    exact = [r for r in rows if r.get("client_message_id") == wire["client_message_id"]
             and r.get("sender_session_id") == wire["sender_session_id"] and r.get("text") == wire["content"]
             and r.get("ingress_accepted") is True and r.get("row_address")]
    if room is not None:
        def addressed(row):
            reply = row.get("reply_to") or {}
            return (row["row_address"].startswith(f"row:{room}@") and reply.get("address") == wire.get("reply_to")
                    and (not reply or (reply.get("chat_id") == room and not reply.get("unresolved"))))
        exact = [row for row in exact if addressed(row)]
    return {"count": len(rows), "addresses": [r["row_address"] for r in exact]}


def send_writes(initial, current):
    if initial != [1, 0] or not isinstance(current, list) or len(current) != 2:
        return None
    if current[0] > 1 or current[1] > 0:
        return False
    return True if current == initial else None  # a missing positive control cannot prove silence


class Arm:
    def __init__(self, name, sha, parent, out, checkpoint):
        self.name, self.sha = name, sha
        self.root = Path(tempfile.mkdtemp(prefix=name[0], dir=parent))
        self.out = out / name
        self.out.mkdir()
        self.tag = secrets.token_hex(4)
        self.marker = "Private-fixture-content-" + self.tag
        self.app, self.data = self.root / "app", self.root / "data"
        self.workspaces = {key: self.root / key for key in TASKS}
        for path in self.workspaces.values():
            path.mkdir()
        self.checks = {phase: dict.fromkeys(names) for phase, names in PHASES.items()}
        self.facts, self.gaps, self.task_ids, self.servers = {}, [], {}, []
        self.tree, self.project = {}, {}
        self.ack = self.held = self.accepted = self.answer = self.project_reply = None
        self.launcher = self.model = self.win = self.generation = None
        self.quit_requested = False
        self.settled = True
        self.checkpoint = checkpoint
        self.phase = "prepare"

    def observe(self, name, read):
        try:
            return read()
        except Exception as exc:
            # No page, log, foreign exception text, environment or command output is a public artifact.
            gap = f"{name} ({public_reason(exc)})"
            if gap not in self.gaps:
                self.gaps.append(gap)
            return None

    def prepare(self):
        from ouroboros.test_environment import isolated_environment
        from tests.system_e2e.harness import (PROXY_ENV_KEYS, STRIPPED_PROVIDER_ENV_KEYS, assert_settings_keyless,
                                              keyless_settings, write_settings_file)

        self.tree = arm_source.prepare(REPO, self.sha, self.root)
        self.env = isolated_environment(self.root, self.app / "repo")
        for key in STRIPPED_PROVIDER_ENV_KEYS | PROXY_ENV_KEYS:
            self.env.pop(key, None)
        self.model = journey_model(self.workspaces, self.tag, self.data / "state/server_port.bindings.json").__enter__()
        port, host_port = fixture_ports()
        self.env.update(OUROBOROS_SERVER_HOST="127.0.0.1", OUROBOROS_SERVER_PORT=str(port),
                        OUROBOROS_PID_FILE=str(self.app / "ouroboros.pid"), OUROBOROS_PORT_FILE=str(self.data / "state/server_port"),
                        OUROBOROS_HOST_SERVICE_PORT=str(host_port), PIP_NO_INDEX="1", GIT_ALLOW_PROTOCOL="file", GIT_TERMINAL_PROMPT="0",
                        PIP_DISABLE_PIP_VERSION_CHECK="1", GIT_AUTHOR_NAME=GIT_IDENTITY[0], GIT_AUTHOR_EMAIL=GIT_IDENTITY[1],
                        GIT_COMMITTER_NAME=GIT_IDENTITY[0], GIT_COMMITTER_EMAIL=GIT_IDENTITY[1])
        cfg = keyless_settings(self.model, OUROBOROS_MAX_WORKERS=3, OUROBOROS_UPDATE_CHANNEL="development",
                               OUROBOROS_SERVER_PORT=port, OUROBOROS_HOST_SERVICE_PORT=host_port, OUROBOROS_DESKTOP_KEEP_RUNNING="true")
        assert_settings_keyless(cfg)
        write_settings_file(self.data / "settings.json", cfg)

    def start_session(self, entry):
        channel, child, extra = launcher_channel(self.root, self.tree["src"] / "launcher.py",
                                                  self.app / "repo/launcher.py", self.out, self.tag)
        self.win = RoomWindow(channel, wait_for, self.out, self.observe)
        self.quit_requested = False
        with child:
            self.launcher = subprocess.Popen([sys.executable, str(entry), "--seed-bundle", str(self.tree["seed"])],
                cwd=entry.parent, env={**self.env, **extra}, pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.settled = False
        self.generation = self.wait_generation(None)
        if (self.app / "ouroboros.pid").read_text().strip() != str(self.launcher.pid):
            raise RuntimeError("launcher PID lock mismatch")
        self.facts.setdefault("launches", []).append({"launcher_pid": self.launcher.pid, "server": self.generation})
        self.checkpoint()
        self.observe("open room", self.open_main)

    def open_main(self):
        current = self.settle(None)
        if not self.active_view("room", current) or not self.win.main() or not self.win.until(
                lambda: self.win.page("return window.__ouroWs?.isConnected() === true"), 30):
            raise RuntimeError("main room unavailable")
        return True

    def active_view(self, label, current):
        if not (current and current["installed"] and current["loopback"] and current["port"] == self.generation["port"]):
            raise RuntimeError("the launcher's installed loopback view was not observed")
        facts = {"stage": label, "before_key": current["key"], "before_visible": current["visible"], "window": current["window"]}
        self.facts.setdefault("focus", []).append(facts)
        if not (current["visible"] and current["key"]):
            self.win.action("reopen")
        ready = self.win.until(lambda: (s := self.win.state()) and same_view(current, s) and s["visible"] and s["key"] and s)
        facts["ready_key"] = bool(ready)
        return ready

    def settle(self, old):
        def current():
            state = self.win.state()
            return state if state and state["loopback"] and state["port"] == self.generation["port"] else None
        return wait_for(lambda: (s := current()) and (not old or s["view"] != old["view"]) and s,
                        25 if old else 75) or current()

    def quit(self):
        if self.quit_requested or self.launcher.poll() is not None:
            return None
        self.quit_requested = True
        # A lost observer reply can be the consequence of Quit itself. Observe the handle; never request it twice.
        try:
            self.win.action("quit")
        except (EOFError, OSError, TimeoutError):
            pass
        code = self.launcher.wait(100)
        self.settled = bool(wait_for(self.servers_gone, 45))
        return code == 0 and self.settled

    def servers_gone(self) -> bool:
        return all(alive(generation) is False for generation in self.servers)

    def api(self, method: str, route: str, body=None, timeout: float = 15.0):
        return http(self.generation["port"], method, route, body, timeout)

    def read_generation(self):
        try:
            binding = json.loads((self.data / "state" / "server_port.bindings.json").read_text())["main"]
            pid, port = int(binding["pid"]), int(binding["port"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        birth = ps(pid, "lstart")  # the launcher's own child, never another process at a recorded pid
        ours = birth and birth == binding.get("fingerprint", {}).get("start_time") and ps(pid, "ppid") == str(self.launcher.pid)
        return {"pid": pid, "port": port, "birth": birth} if ours else None

    def wait_generation(self, previous) -> dict:
        """The launcher's next server: its own child with a new pid or birth, bound to its port and ready."""
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if self.launcher.poll() is not None:
                raise RuntimeError(f"the launcher exited {self.launcher.returncode} while its server was awaited")
            current = self.read_generation()
            if current and (previous is None or (current["pid"], current["birth"]) != (previous["pid"],
                                                                                        previous["birth"])):
                if current not in self.servers:  # every generation seen must be gone at the end, ready or not
                    self.servers.append(current)
                status, body = http(current["port"], "GET", "/api/health")
                boot = (body or {}).get("boot") or {}
                if status == 200 and boot.get("phase") == "ready":
                    return {**current, "health": {k: boot.get(k) for k in ("phase", "dialogue_ready", "tasks_ready")}}
            time.sleep(0.5)
        raise TimeoutError("no new server generation answered ready")

    def wait_gone(self, generation: dict, seconds: float = 300) -> bool:
        if not wait_for(lambda: alive(generation) is False, seconds, 0.25):
            raise TimeoutError(f"the old server {generation['pid']} did not exit")
        return True

    def trigger_restart(self) -> int:
        status, body = self.api("POST", "/api/command", {"cmd": "/restart", "ui_control": {"name": "restart"}}, 60)
        if status and not 200 <= status < 300:
            raise RuntimeError(f"Restart returned HTTP {status}")
        return status  # even if the reply was lost, observe this operation once

    def history(self, chat_id=1):
        status, body = self.api("GET", f"/api/chat/history?chat_id={chat_id}&n_human=1500&n_progress=600")
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return None
        return body

    def task_facts(self) -> dict:
        state = self.api("GET", "/api/state")[1] or {}
        activities = {row.get("activity_id"): row for row in state.get("active_chat_activities") or []
                      if isinstance(row, dict)}
        facts = {}
        for key, task in self.task_ids.items():
            detail, activity = self.api("GET", f"/api/tasks/{task}")[1] or {}, activities.get(task) or {}
            facts[key] = {"task_id": detail.get("task_id"), "activity_id": activity.get("activity_id"),
                          "status": detail.get("status"), "phase": activity.get("phase"),
                          "pause_cause": activity.get("pause_cause"), "held_requests": self.model.count(key),
                          "activities_complete": state.get("active_chat_activities_complete"),
                          "requests_in_generation": self.model.count(key, self.generation), "read_requests": self.model.read_requests[key]}
        return facts

    def pause_kept(self) -> bool | None:
        paused = self.task_facts().get("paused") or {}
        if not paused.get("phase") and paused.get("activities_complete") is not True:
            return None
        original = self.facts.get("tasks_before", {}).get("paused")
        return (paused.get("activity_id") == self.task_ids["paused"] and paused.get("phase") == "budget_paused"
                and paused.get("pause_cause") == "owner" and (not original or
                    (paused["held_requests"] == original["held_requests"] and paused["requests_in_generation"] == 0)))

    def start_tasks(self) -> None:
        for key in TASKS:
            status, body = self.api("POST", "/api/tasks", {
                "description": f"[E2E-LINEAGE:{key}] Read probe-{key}.txt and retain its exact content.",
                "memory_mode": "forked", "workspace_mode": "external", "workspace_root": str(self.workspaces[key]),
                "source": "cli"})
            if not 200 <= status < 300 or not (body or {}).get("task_id") or body["task_id"] in self.task_ids.values():
                raise RuntimeError(f"the {key} task was refused (HTTP {status})")
            self.task_ids[key] = body["task_id"]
            if not wait_for(lambda: self.model.count(key, self.generation) >= 1, 240, 1.0):
                raise TimeoutError(f"the {key} task never read its probe and asked again")
        status, _body = self.api("POST", f"/api/tasks/{self.task_ids['paused']}/pause",  # the Pause button's request
                                 {"request_id": f"window-renewal-{self.tag}", "actor": "owner"})
        self.facts["owner_pause_http"] = status
        if (status and not 200 <= status < 300) or not wait_for(self.pause_kept, 120, 1.0):
            raise RuntimeError(f"the owner Pause did not settle (HTTP {status})")
        self.facts["tasks_before"] = self.task_facts()

    def create_project(self, name):
        status, body = self.api("POST", "/api/projects", {"name": name, "id": name, "no_folder": True})
        if not 200 <= status < 300 or not (body or {}).get("project", {}).get("chat_id"):
            raise RuntimeError("project creation unconfirmed")
        return body["project"]

    def peer_project(self, phase, current):
        w, checks = self.win, self.checks[phase]
        name = f"peer-{phase}-{self.tag}"
        listening = self.observe(phase + " listen for project", lambda: w.watch_project(name))
        try:
            peer = self.create_project(name)
            expected = {"type": "projects_changed", "project_id": peer["id"], "chat_id": peer["chat_id"]}
            event = self.facts[phase + "_peer"] = {"listening": listening, "expected": expected, "received": None}
            def heard():
                if not listening:
                    return listening
                event["received"] = w.until(lambda: w.page("return window.__renewalPeer.frame"), 10)
                return event["received"] == expected
            checks["live_connected"] = self.observe(phase + " incoming project event", heard)
            checks["peer_visible"] = w.shows_project(peer, current, in_list=True)
        finally:
            if listening:
                w.page("window.__renewalPeer?.stop(); delete window.__renewalPeer")

    def seed_exchange(self):
        self.open_main()
        self.ack = self.win.send_once("Hello-" + self.tag, False)
        if not self.ack or not self.ack.get("client_message_id") or not self.ack.get("sender_session_id"):
            raise RuntimeError("first Send had no observed wire identity")
        def accepted():
            result = receipt(self.history(), self.ack)
            return result["addresses"][0] if result and result["count"] == len(result["addresses"]) == 1 else None
        self.accepted = wait_for(accepted, 60)
        def answered():
            history = self.history()
            return next((row.get("row_address") for row in (history or {}).get("messages", [])
                         if row.get("role") == "assistant" and f"QUOTE-{self.tag}" in row.get("text", "")), None)
        self.answer = wait_for(answered, 90)
        if not self.accepted or not self.answer:
            raise RuntimeError("the canonical exchange was not observed")
        self.facts["accepted"] = {"id": self.ack["client_message_id"], "sender": self.ack["sender_session_id"],
                                  "row": self.accepted, "assistant_row": self.answer}

    def pending(self):
        if not self.answer or not self.win.quote(self.answer) or not self.win.follows_quote(self.answer):
            raise RuntimeError("the pending message's original reply could not be selected")
        held = self.win.send_once("Held-" + self.tag, True)
        if not held or not held.get("client_message_id") or not held.get("sender_session_id") or held.get("reply_to") != self.answer:
            raise RuntimeError("original frame identity unavailable")
        if not wait_for(lambda: self.win.bubble(held["client_message_id"])["count"] == 1 and
                        same_frame(held, self.slots()["frame"]), 15):
            raise RuntimeError("the original view did not retain the held frame")
        if receipt(self.history(), held) != {"count": 0, "addresses": []}:
            raise RuntimeError("the held message's absence was not established")
        self.held = held
        self.facts["held"] = {k: held[k] for k in FRAME_FIELDS if k != "content" and k in held}
        self.facts["original_writes"] = self.wire_counts()

    def wire_counts(self, text=None):
        return self.win.page("return window.__renewalWireCounts?.(a.text)", text=text)

    def project_send(self, phase, target):
        """Click once. A lost observer reply never causes another Send."""
        project, room = self.project["id"], self.project["chat_id"]
        text = f"Project-{phase}-{self.tag}"
        reply = None if phase == "seed" else self.project_reply
        evidence = self.facts[phase + "_project_send"] = {"project": project, "room": room, "text": text, "reply": reply}
        target = self.win.key_window(target)
        if not target or target["port"] != self.generation["port"] or (phase != "seed" and not reply):
            return
        prefix = "#pchat-" + project
        doc = self.win.draft(prefix + "-input")
        if not doc or doc["names"] or doc["input_files"]:
            return  # a text-only Send cannot be measured with a live attachment
        from_draft = phase != "seed"
        if from_draft and doc["text"] != text:
            return
        evidence["wire"] = self.win.send_once(text, False, prefix, target, from_draft=from_draft)

    def project_delivery(self, phase):
        evidence = self.facts.get(phase + "_project_send") or {}
        wire = evidence.get("wire")
        if not wire:
            return None
        evidence["once"] = None
        room, text, reply = evidence["room"], evidence["text"], evidence["reply"]
        addressed = (wire.get("type") == "chat" and wire.get("content") == text and wire.get("project_id") == evidence["project"]
                     and wire.get("chat_id") == room and wire.get("reply_to") == reply)
        def delivered():
            saved = receipt(self.history(room), wire, room=room)
            main = receipt(self.history(), wire)
            writes = self.wire_counts(text)
            evidence.update(saved=saved, main=main, writes=writes)
            if not addressed or (main and main["count"]) or (writes is not None and writes > 1) or (
                    saved and saved["count"] and (saved["count"] != 1 or len(saved["addresses"]) != 1)):
                evidence["once"] = False
            elif saved and saved["addresses"] and main and writes == 1:
                evidence["once"] = True
            return evidence["once"] is not None or not (wire.get("client_message_id") and wire.get("sender_session_id"))
        wait_for(delivered, 60)
        if phase == "seed" and evidence["once"]:
            self.project_reply = evidence["saved"]["addresses"][0]
        return evidence["once"]

    def slots(self):
        return self.win.custody(self.marker, self.project.get("chat_id"))

    def file_count_only(self, slots, project_file):
        from ouroboros.chat_uploads import uploads_dir

        if not slots or "files" not in slots:
            return None
        uploads = uploads_dir(self.data)
        try:
            uploaded = any(uploads.iterdir())
        except FileNotFoundError:
            uploaded = False
        return (slots.get("files") == "1" and (not project_file or slots.get("project_files") == "1")
                and slots.get("contains_fixture_bytes") is False and not uploaded)

    def stage(self, phase):
        tag, leave_project = f"{phase}-{self.tag}", phase in PROJECT_UPDATES
        staged = {"main": {"prefix": "#chat", "text": "Draft-" + tag, "reply": self.answer, "filename": f"Note-{tag}.txt"},
                  "project": {"prefix": "#pchat-" + self.project.get("id", ""), "text": "Project-" + tag,
                              "reply": self.project_reply if leave_project else None,
                              "quote_text": "Project-seed-" + self.tag if leave_project else None,
                              "filename": f"Project-{tag}.txt" if leave_project else None}}
        before = self.facts[phase + "_before"] = {"main_room": 1, "project_id": self.project.get("id"),
                                                   "project_room": self.project.get("chat_id")}
        for room in (("main", "project") if leave_project else ("project", "main")):
            draft = staged[room]
            self.observe(f"{phase} prepare {room}", lambda: self.win.prepare_draft(
                draft, self.marker, self.project if room == "project" else None))
            before[room] = {k: v for k, v in draft.items() if isinstance(v, bool)}
        staged["count_only"] = self.observe(phase + " file count", lambda:
            wait_for(lambda: self.file_count_only(self.slots(), leave_project), 6) if "files" in self.slots() else None)
        before["count_only"] = staged["count_only"]
        return staged

    def apply(self, phase):
        target = self.tree["handoff" if phase == "handoff" else "target"]
        if phase == "handoff":
            arm_source.offer(self.tree, target)  # the launcher.py child is published only when its turn comes
        status, body = self.api("POST", "/api/update/preflight", {}, 240)
        plan = (body or {}).get("merge_plan") or {}
        base = self.tree["target" if phase == "handoff" else "base"]
        offered = {"available": plan.get("available") is True, "clean": plan.get("kind") == "clean",
                   "base_matches": plan.get("base_sha") == base, "target_matches": plan.get("target_sha") == target}
        self.facts[phase + "_preflight"] = {"http": status, **offered}
        if status != 200 or not all(offered.values()):
            raise RuntimeError(f"the clean pinned update was not offered (HTTP {status}, {offered})")
        status, applied = self.api("POST", "/api/update/apply", {"strategy": "auto_merge",
                                   "expected_base_sha": base, "expected_target_sha": target}, 300)
        self.facts[phase + "_request"] = {"base": base, "target": target, "http": status}
        if status and not (200 <= status < 300 and (applied or {}).get("restarting") is True):
            raise RuntimeError(f"managed update did not request restart (HTTP {status})")

    def change(self, name, kind, condition, **args):
        """One native window action, then the state that shows it within 12 s: False when it did not show,
        None when the action or its observation failed."""
        def act():
            self.win.action(kind, **args)
            return wait_for(lambda: (s := self.win.state()) and condition(s) and s, 12, 0.2) or False
        return self.observe(name, act)

    def reopened(self, name, target):
        state = self.change(name, "reopen", lambda s: visible_key_view(target, s))
        return None if state is None else bool(state)

    def presentation(self, phase):
        old = self.win.state()
        if not old or phase == "handoff":
            return {}, old
        width, height = (870, 540) if phase == "restart" else (930, 580)
        resized = self.change(phase + " resize", "resize", lambda s: s["frame"][2:] != old["frame"][2:],
                              width=width, height=height)
        zoom = self.change(phase + " View menu", "zoom", lambda s: s["zoom"] != old["zoom"], title="Zoom In")
        observed = {"frame": resized and resized["frame"], "zoom": zoom and zoom["zoom"]}
        if phase == "restart":
            observed["hide"] = self.change("restart hide", "hide", lambda s: not s["visible"])
        else:
            observed["fullscreen"] = self.change("update fullscreen", "fullscreen", lambda s:
                same_view(old, s) and s["fullscreen"] and s["fullscreen_phase"] == "full")
        return observed, self.win.state()

    def after_presentation(self, phase, observed, current):
        w, checks = self.win, self.checks[phase]
        if phase == "handoff" or not current:
            return
        if observed.get("hide"):
            checks["hidden_kept"] = not current["visible"]
            checks["reopened"] = self.reopened(phase + " reopen", current)
        if phase == "update":
            def windowed(s):
                return same_view(current, s) and not s["fullscreen"] and s["fullscreen_phase"] == "windowed"
            ready = windowed(current)
            if observed.get("fullscreen"):
                checks["fullscreen_kept"] = current["fullscreen"]
                if current["fullscreen"] and current["fullscreen_phase"] == "full":
                    ready = self.change("leave full screen", "fullscreen", windowed)
            if ready and self.change("hide after update", "hide", lambda s: same_view(current, s) and not s["visible"]):
                checks["reopened"] = self.reopened("reopen after update", current)
        state = w.state()
        if observed.get("zoom") and state:
            checks["zoom_kept"] = abs(state["zoom"] - observed["zoom"]) < 0.001
        if observed.get("frame"):
            frame = observed["frame"]
            checks["size_kept"] = bool(wait_for(lambda: (s := w.state()) and same_view(current, s) and not s["fullscreen"] and
                all(abs(a-b) <= 2 for a, b in zip(s["frame"][2:], frame[2:])), 6, 0.2))

    def sends(self, history):
        """(the accepted Send is still one row at its address, the held Send is still absent, its receipt);
        None where the Send never happened or the history read was incomplete."""
        ack = receipt(history, self.ack) if self.accepted else None
        held = receipt(history, self.held) if self.held else None
        return (None if ack is None else ack == {"count": 1, "addresses": [self.accepted]},
                None if held is None else held["count"] == 0, held)

    def inspect_sends(self, phase):
        checks = self.checks[phase]
        checks["accepted_once"], checks["no_replay"], held = self.sends(self.history())
        if self.held:
            def restored():
                value = self.win.bubble(self.held["client_message_id"])
                return value if value.get("retry") else None
            bubble = wait_for(restored, 8) or self.win.bubble(self.held["client_message_id"])
            checks["pending_owner"] = bubble == {"count": 1, "retry": True} and same_frame(self.held, self.slots()["frame"])
            self.facts[phase + "_send"] = {"receipt": held, "held_bubble": bubble}
        counts = self.wire_counts()
        self.facts[phase + "_writes"] = counts
        checks["wire_once"] = send_writes(self.facts.get("original_writes"), counts)

    def generation_step(self, phase):
        before = self.generation
        if phase in ("update", "handoff"):
            self.apply(phase)
        else:
            self.facts[phase + "_request"] = {"http": self.trigger_restart()}
        gone = self.observe(phase + " previous server exit", lambda: self.wait_gone(before))
        self.generation = self.wait_generation(before)
        checks = self.checks[phase]
        checks["generation_new"] = gone
        self.facts[phase + "_server"] = {"old": before, "new": self.generation}
        if "client_health" in checks:
            checks["client_health"] = (all(self.generation["health"][key] is True for key in ("dialogue_ready", "tasks_ready"))
                                       and self.api("GET", "/api/state")[0] == 200)
        if phase in ("update", "handoff"):
            sha = wait_for(lambda: (self.api("GET", "/api/update/status")[1] or {}).get("running_sha"), 25)
            if sha:
                checks["update_landed"] = sha == self.facts[phase + "_request"]["target"]

    def transition(self, phase):
        checks, w = self.checks[phase], self.win
        stage = self.stage(phase)
        project = stage["project"]
        field = project["prefix"] + "-input"
        if all(stage[room].get(k) for room in ("main", "project") for k in ("typed", "quoted", "file")):
            checks["old_draft"] = True
        held = self.model.count("live", self.generation)
        prepared = self.observe(phase + " presentation", lambda: self.presentation(phase))
        observed, old = prepared if prepared else ({}, None)
        window = self.facts[phase + "_window"] = {"before": old}
        if phase in PROJECT_UPDATES and project.get("typed"):
            checks["project_active_before"] = self.observe(phase + " original Project", lambda:
                w.shows_project(self.project, old) and w.visible(field, editable=True)) or None
        self.checkpoint()
        request_started = time.time()
        self.generation_step(phase)  # exactly once, after all staging and observations in the old document
        self.checkpoint()
        current = self.observe(phase + " observe view", lambda: self.settle(old))
        window["after"] = current
        settled = self.observe(phase + " document", lambda: w.document(self.generation["port"]))
        trace = self.facts[phase + "_diagnostics"] = launcher_diagnostics(self.root, request_started, time.time())
        if "diagnostic_lookup_recorded" in checks and trace["process_log_lookup"] in OSLOG_RESULTS:
            checks["diagnostic_lookup_recorded"] = True
        if current and old:
            checks["view_new"] = current["view"] != old["view"]
            checks["installed"] = current["installed"] and old["installed"] and (
                current["shell"] != old["shell"] if phase == "handoff" else same_window(old, current))
            if phase == "handoff":
                rev = self.observe("loaded source", lambda: w.until(lambda: w.page(
                    "const api = window.pywebview?.api; return api?.shell_info ? (await api.shell_info()).source_revision : null")))
                checks["source_loaded"] = rev == self.tree["handoff"] and current["shell"] != old["shell"] if rev else None
        if phase in PROJECT_UPDATES:
            checks["document_ready"] = True if settled else None
            if settled and checks.get("project_active_before"):
                checks["project_active_after"] = self.observe(phase + " restored Project", lambda:
                    w.shows_project(self.project, current, origin=settled["origin"]))
        self.observe(phase + " presentation after", lambda: self.after_presentation(phase, observed, current))
        if checks.get("project_active_after"):
            checks["project_editable"] = self.observe(phase + " Project input", lambda: w.until(lambda: w.visible(field, editable=True)))
        if stage["main"].get("file") and stage["count_only"] is not None:
            checks["file_count_only"] = stage["count_only"] and self.observe(phase + " restored file count", lambda:
                w.until(lambda: self.file_count_only(self.slots(), phase in PROJECT_UPDATES), 6))
        if checks.get("project_editable"):
            self.observe(phase + " Project draft", lambda: w.inspect_draft(project, checks, self.marker))
            if project.get("quoted"):
                self.observe(phase + " Project Send", lambda: self.project_send(phase, current))
            self.observe(phase + " Project typing", lambda: w.typing(current, field, phase + "-project", checks,
                                                                     self.facts.setdefault(phase + "_project_pixels", {})))
            checks["project_sent_once"] = self.observe(phase + " Project receipt", lambda: self.project_delivery(phase))
        self.observe(phase + " Main draft", lambda: w.main() and w.inspect_draft(stage["main"], checks, self.marker))
        if phase == "restart" and project.get("typed"):
            self.observe("restart read Project", lambda: w.open_project(self.project) and w.inspect_draft(project, checks, self.marker))
            self.observe("restart return to Main", w.main)
        self.observe(phase + " new project", lambda: self.peer_project(phase, current))
        self.observe(phase + " send custody", lambda: self.inspect_sends(phase))
        if phase != "handoff":
            checks["confirm_cancelled"] = self.observe(phase + " Cancel", lambda: w.confirm_cancel("Fixture confirmation " + phase))
        self.observe(phase + " native typing", lambda: w.typing(current, "#chat-input", phase, checks,
                                                               self.facts.setdefault(phase + "_pixels", {})))
        if "live" in self.task_ids and held:
            resumed = wait_for(lambda: self.model.count("live", self.generation) > 0, 180)
            facts = self.observe(phase + " task facts", self.task_facts) or {}
            live = facts.get("live") or {}
            original_reads = self.facts.get("tasks_before", {}).get("live", {}).get("read_requests")
            if original_reads and live.get("status") and live.get("task_id"):
                checks["live_continued"] = bool(resumed and live["task_id"] == self.task_ids["live"]
                                                and live["read_requests"] == original_reads
                                                and live["status"] not in ("failed", "cancelled", "completed"))
            self.facts[phase + "_tasks"] = facts
        if "tasks_before" in self.facts:
            checks["pause_kept"] = self.observe(phase + " paused task", self.pause_kept)
        end = self.observe(phase + " document after", lambda: w.document(self.generation["port"]))
        self.facts[phase + "_document"] = {"settled": settled, "end": end}
        last = self.observe(phase + " final view", w.state)
        if not same_view(current, last) or not settled or not end or end["origin"] != settled["origin"]:
            self.gaps.append(f"{phase}: a single installed successor document could not be established")
        self.checkpoint()

    def help(self, label):
        w = self.win
        before = self.active_view(label, w.state())
        if not before or before["builtin_setup"] or w.visible("#onboarding-appearance-label"):
            raise RuntimeError("Help needs the active window on a different page")
        w.action("help")
        return bool(w.until(lambda: (s := w.state()) and same_view(before, s) and s["loopback"]
                            and s["port"] == before["port"] and s["builtin_setup"] and s["visible"]
                            and w.visible("#onboarding-appearance-label"), 18))

    def fixture(self):
        checks, w = self.checks["fixture"], self.win
        old = self.observe("fixture initial view", w.state)
        ui = self.root / "custom-ui"
        ui.mkdir()
        sample = f"sample-{self.tag}.html"
        (ui / sample).write_text('<!doctype html><title>Sample</title><h1 id="sample">Sample page</h1>', encoding="utf-8")
        route = "/static/" + sample
        (ui / "index.html").write_text(fixture_page(self.generation["port"], self.tag, route), encoding="utf-8")
        if self.api("POST", "/api/settings", {"OUROBOROS_UI_DIR": str(ui)})[0] != 200:
            raise RuntimeError("custom UI selection unconfirmed")
        self.generation_step("fixture")
        state = self.settle(old)
        if old and state:
            checks["installed"] = same_window(old, state) and old["installed"] and state["installed"]
            checks["view_new"] = old["view"] != state["view"]
        if not state or not (state["installed"] and state["loopback"] and state["home"] and state["port"] == self.generation["port"]):
            raise RuntimeError("the selected custom interface was not observed at the bound server")
        if not w.until(lambda: w.page("return document.querySelector('button')?.textContent === a.label", label=FIXTURE_BUTTON)):
            raise RuntimeError("custom interface did not open")
        if w.click("button"):
            checks["sample_document_opened"] = bool(w.until(lambda: same_view(state, w.state()) and w.visible("#sample")
                and w.page("return location.pathname === a.path && location.port === a.port", path=route, port=str(self.generation["port"]))))
        checks["menu_drives_view"] = self.observe("Help menu in custom interface", lambda: self.help("custom Help"))

    def relaunch(self, staged):
        w, checks = self.win, self.checks["session"]
        self.open_main()
        accepted, absent, _ = self.sends(self.history())
        writes = send_writes(self.facts.get("original_writes"), self.wire_counts())
        if None not in (accepted, absent, writes):
            checks["no_send_after_quit"] = writes and accepted and absent
        doc = w.draft()
        project = self.observe("relaunch project", lambda: w.open_project(self.project) and w.draft(staged["project"]["prefix"] + "-input"))
        if not w.main():
            raise RuntimeError("return from the relaunched Project was not observed")
        slots = self.slots()
        if doc and project and all(staged[r].get(k) for r in ("main", "project") for k in ("typed", "quoted", "file")):
            persisted = w.page("""
                return Object.values(localStorage).some(value => a.markers.some(text => value.includes(text)))
            """, markers=[staged["main"]["text"], staged["project"]["text"], self.marker])
            checks["draft_gone"] = (not any(doc.values()) and not any(project.values()) and not any(
                slots.get(k) for k in ("text", "project", "reply", "files", "project_files", "frame")) and persisted is False)
        # Help first opens from the relaunched built-in document, then independently from the custom one.
        checks["help_after_quit"] = self.help("built-in Help")
        if checks["help_after_quit"]:
            w.page("setTimeout(() => location.replace('/'), 0); return true")
            self.open_main()

    def run(self):
        from tests.system_e2e.harness import ArtifactOracle, wait_durable_result

        session = self.checks["session"]
        try:
            self.prepare()
            self.phase = "first launch"
            self.start_session(self.tree["src"] / "launcher.py")
            self.project = self.observe("create initial project", lambda: self.create_project("room-" + self.tag)) or {}
            self.observe("first exchange", self.seed_exchange)
            self.observe("hold original Send", self.pending)
            self.observe("seed a Project reply target", lambda: self.win.open_project(self.project) and self.project_send("seed", self.win.state()))
            self.observe("saved Project reply target", lambda: self.project_delivery("seed"))
            self.checkpoint()
            self.observe("start independent tasks", self.start_tasks)
            for phase in TRANSITIONS:
                self.phase = phase
                self.transition(phase)
            self.phase = "release live task and Quit"
            self.model.release["live"].set()
            if "live" in self.task_ids:
                stored = self.observe("durable live result", lambda: wait_durable_result(ArtifactOracle(self.data), self.task_ids["live"]))
                if stored:
                    answer = str(stored.get("final_answer") or stored.get("result") or "")
                    has_probe = self.model.tokens["live"] in answer
                    session["live_finished"] = (stored.get("task_id") == self.task_ids["live"] and stored.get("status") == "completed"
                                                and has_probe)
                    self.facts["live_final"] = {"id": stored.get("task_id"), "status": stored.get("status"), "probe_in_result": has_probe}
            staged = self.stage("quit")
            session["ordinary_quit"] = self.quit()
            self.checkpoint()
            if session["ordinary_quit"]:
                self.phase = "cold relaunch"
                self.win.channel.conn.close()
                self.start_session(self.app / "repo/launcher.py")
                self.observe("actual Quit and reopen", lambda: self.relaunch(staged))
                if "tasks_before" in self.facts:
                    session["pause_kept_after_quit"] = self.observe("paused task after Quit", lambda: wait_for(self.pause_kept, 40))
                self.observe("navigation fixture", self.fixture)
        except Exception as exc:
            self.gaps.append(f"{self.phase} stopped ({public_reason(exc)})")
        finally:
            if self.launcher is not None and not self.quit_requested and self.launcher.poll() is None:
                self.observe("final Quit", self.quit)
            if not self.settled and self.launcher is not None and self.launcher.poll() is not None:
                self.settled = bool(self.servers and self.observe("server exits", self.servers_gone))
            if self.win:
                self.win.channel.conn.close()
            if self.model:
                for release in self.model.release.values():
                    release.set()
                self.model.__exit__(None, None, None)
            if self.settled:
                shutil.rmtree(self.root)
            else:
                from ouroboros.test_environment import retain_tree
                retain_tree(self.root, "fixture exit not confirmed")
                self.gaps.append("fixture exit not confirmed; directory retained by safe_test")
        self.checkpoint()

    def result(self):
        return {"sha": self.sha, "phase": self.phase, "checks": self.checks, "facts": self.facts, "gaps": self.gaps,
                **{k: self.tree.get(k) for k in ("tree", "base", "target", "handoff", "version")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.baseline == args.candidate or not all(re.fullmatch("[a-f0-9]{40}", x) for x in (args.baseline, args.candidate)):
        parser.error("two distinct full commit IDs are required")
    out = args.out.resolve()
    if out == REPO or REPO in out.parents or (out.exists() and any(out.iterdir())):
        parser.error("output must be new or empty and outside the checkout")
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "started_at": now(), "host": host_facts(), "instrument": instrument_identity(), "arms": {},
              "judgement": {"verdict": "INCONCLUSIVE", "gaps": ["the native run has not reached its final observation"]}}
    parent = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if sys.platform != "darwin" or not parent:
        report["unsupported"] = "requires macOS and a safe_test disposable root"
    else:
        for arm, sha in (("baseline", args.baseline), ("candidate", args.candidate)):
            def checkpoint():
                report["arms"][arm] = case.result()
                write_json(out / "report.json", report)
            case = Arm(arm, sha, Path(parent), out, checkpoint)
            checkpoint()
            case.run()
            if not case.settled:  # a surviving window or server of this arm could take the next arm's key window
                report.setdefault("gaps", []).append(f"the arms after {arm} did not run: its exit was not confirmed")
                break
    report.update(judgement=judge(report), finished_at=now())
    write_json(out / "report.json", report)
    verdict = report["judgement"]["verdict"]
    print(f"macOS window renewal {verdict}")
    return EXIT_CODES[verdict]


if __name__ == "__main__":
    sys.exit(main())
