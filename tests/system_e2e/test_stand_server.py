"""S43, the stand's real half: E1, E5, E7, E9 and E10 on a real server with real workers, the scripted model at a fixed latency, no spend.

The root is the production shape's standing set (tests/_stand.py ``build_root``: restored paused trees and pending rows, held runs, schedules,
markers, rooms and a story, a wake soon due), with the fake engine answering for the owned daemon. Two servers per shape and journal record what
happened (``_Run``), and each law then reads its own part of the record, so a law that fails names itself. ``load``: an owner turn alone; real
tasks, concurrent owner turns (each in a conversation of its own, as in tests/_stand.py) and a Presence burst at once, held at the model; the
door, the controls and one more turn answered under them; the
browser's Panic frame ending all of it. ``recovery``, on a fresh root (Panic ended the first): every project worker killed, then a hello; no
process able to start, then a hello; an update's conflict markers in the tree, then a hello; Stop of a held turn. The reduced shape runs in the keyless mock lane; the full one (50 real workers, tens of
GiB) is its own lane, ``OUROBOROS_E2E_DEEP=stand_full``. Linux's forkserver is the mock lane's own start method (``supervisor.workers``), so E10's second half
is this file on that runner. Windows has no runner for this file: there the counted half (tests/test_stand_counts.py) runs in the default lane, and
every law of this file stays unmeasured.

A law the tree cannot meet yet is a strict xfail naming what breaks it. Wall-clock figures are facts (printed, and into
``OUROBOROS_TEST_EVIDENCE_ROOT``); a clock that decides is only ever a transport bound, so that a broken run fails instead of hanging.

Named elsewhere, not repeated here: a turn's saved work across a planned restart, returned and never replayed, through the real kill, snapshot and
boot restore (tests/test_restart_saved_work.py, tests/system_e2e/test_direct_project_turn_restart.py); ``disable_exhausted_worker_pool`` sparing a
lane carrier (LA's tests/test_conversation_lane_pool.py); E6's count (tests/test_stand_counts.py).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import signal
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List

import pytest

from tests._stand import OTHER_CHAT, SHAPES
from tests._store_shared import as_install
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ScriptedStubModel,
    clone_repo,
    keyless_settings,
    message_text,
    require_lane,
    wait_durable_result,
    wait_until,
    write_settings_file,
    ws_url,
)
from tests.system_e2e.interfaces import FakeClaudexorDaemon
from tests.system_e2e.test_ordinary_conversation_capabilities import _received_turn
from tests.system_e2e.test_presence_resilience import (
    LIVE_DEFAULT_PORTS,
    _call,
    _event_body,
    _HookedServer,
    _seed_behavior,
    _seed_transport,
    _Turn,
)
from tests.system_e2e.test_system_scenarios_w6 import _WsFrames
from tests._stand_cache import child

_TIMEOUT_METHOD = "signal" if hasattr(signal, "SIGALRM") else "thread"
pytestmark = [pytest.mark.integration, pytest.mark.timeout(3600, method=_TIMEOUT_METHOD)]

LANE_FULL = "stand_full"  # the full shape's own lane: 50 real workers
HOOK = pathlib.Path(__file__).resolve().parents[1] / "_stand_hook.py"
STORM_WORKERS = 3  # three busy project workers killed at once: each task gets its one retry, no fence (ARCHITECTURE §5)
RESTORED_MARKER = "live work "  # the input of every row the root's queue restores at boot (tests/_live_set.py ``_task``)
TURN_MARKER = "[STAND-TURN]"
READ_WINDOW_SEC = 60.0  # a transport bound for one answer under load
SETTLE_SEC = 300.0  # a transport bound for one turn to its end
BOOT_SEC = 900.0  # a transport bound for the pool to take its first rows (a spawn takes up to a minute on a loaded Mac)
QUIET_SEC = 5.0  # how long the worker ledger stays empty before no spawn in flight can still land in it
DEGRADED = "inline_degraded"
NO_CARRIER = ("blocked", "window")  # a refused spawn; an assisted update's conflict markers in the tree a spawn would import
COGNITION_DOORS = {"handle_task", "build_llm_messages", "_context_facts", "render_wake_message", "observe_wake", "run_presence_turn"}


def _own_input(body: Dict[str, Any]) -> str:
    """The first user message of a model call: the input of the turn it serves. The system prompt lists everything else at work,
    and the host's later user rows name other roots."""
    users = [message_text(message) for message in body.get("messages") or [] if isinstance(message, dict) and message.get("role") == "user"]
    return users[0] if users else ""


class _HoldAll:
    """Every model call whose own input carries ``marker`` waits here until ``release``: the work that stays live while the scenario
    looks. Bounded, so a broken scenario fails by name."""

    def __init__(self, marker: str, *, timeout: float = 1500.0) -> None:
        self.marker, self.timeout = marker, timeout
        self.release = threading.Event()
        self.arrived = 0
        self._lock = threading.Lock()

    def __call__(self, body: Dict[str, Any]) -> None:
        if not body.get("tools") or self.marker not in _own_input(body):
            return
        with self._lock:
            self.arrived += 1
        if not self.release.wait(self.timeout):
            raise TimeoutError(f"nobody released {self.marker}")


class _Gates:
    def __init__(self, *gates: Any) -> None:
        self.gates = gates

    def __call__(self, body: Dict[str, Any]) -> None:
        for gate in self.gates:
            gate(body)


class ScenarioBroke(Exception):
    """The scenario itself could not finish: never one of the laws' assertions, so no strict xfail may absorb it."""


def _strict_gap(reason: str) -> Any:
    """A law the tree cannot meet yet: only its own assertion may fail, so a broken scenario errors by name."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


def _doors(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _worker_pids(data_root: pathlib.Path) -> Dict[str, int]:
    """The pool's members, by the custody ledger: each worker is ledgered by measured birth."""
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_custody import ledger_path

    pids: Dict[str, int] = {}
    path = ledger_path(data_root)
    for line in (path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else ()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        purpose, pid = str(row.get("purpose") or ""), int(row.get("pid") or 0)
        if purpose.startswith("worker:") and pid and pid_is_alive(pid):
            pids[purpose] = pid
    return pids


def _carrier_pid(oracle: ArtifactOracle, data_root: pathlib.Path, task_id: str) -> int:
    """The process that runs a live turn, by its RUNNING row's slot and that slot's ledger row; 0: no custodied process runs it."""
    for row in oracle.queue_snapshot().get("running") or []:
        if row.get("id") == task_id and row.get("worker_id") is not None:
            return _worker_pids(data_root).get(f"worker:{row['worker_id']}", 0)
    return 0


class _Run:
    """One server's part of the scenario: every step records into ``facts``; nothing is asserted before the record is whole."""

    def __init__(self, candidate: Any, home: pathlib.Path, shape: str, journal: str, *, workers: int = 0) -> None:
        self.candidate, self.home, self.shape, self.journal = candidate, home, SHAPES[shape], journal
        self.shape_name, self.workers = shape, workers or SHAPES[shape].real_tasks
        self.data_root = home / "data"
        self.door_log, self.refuse_flag, self.window_flag = home / "doors.jsonl", home / "refuse-spawn", home / "marker-window"
        self.sample_log = home / "samples.json"
        self.tree = f"stand-tree-{uuid.uuid4().hex[:10]}"
        self.facts: Dict[str, Any] = {"shape": shape, "journal": journal, "timings": {}}
        self.holds, self.turns = _HoldAll(RESTORED_MARKER), _HoldAll(TURN_MARKER)
        self.stub = ScriptedStubModel(gate=_Gates(self.holds, self.turns), latency_sec=0.2,
                                      final_answer="The stand's scripted answer.")
        self.server: Any = None
        self.oracle: Any = None
        self.ws: Any = None
        self.frames: Any = None

    # ---- the root ----------------------------------------------------------------------------------------------------------
    def build(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True)
        (self.home / "repo").mkdir()
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(self.data_root), "OUROBOROS_SETTINGS_PATH": str(self.data_root / "settings.json"),
               "OUROBOROS_REPO_DIR": str(self.home / "repo"), "OUROBOROS_HOST_SERVICE_PORT": "0",
               "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(self.home / "worktrees"), "COST_STORE_JOURNAL": self.journal}
        started = time.monotonic()
        child("tests._stand", [str(self.data_root), self.shape_name, str(self.home / "built.json")], env, self.home / "build.log")
        self.facts["timings"]["build_sec"] = round(time.monotonic() - started, 1)
        self.facts["built"] = json.loads((self.home / "built.json").read_text(encoding="utf-8"))["laid"]

    def start(self) -> None:
        settings = keyless_settings(self.stub, OUROBOROS_MAX_WORKERS=self.workers, OUROBOROS_BG_WAKEUP_MIN=60,  # a wake within a minute of boot
                                    OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT=4, OUROBOROS_CONSCIOUSNESS_SAFETY_MODE="off",
                                    OUROBOROS_MODEL_MAX_CONCURRENCY=0,  # one mock route: its cap of three would queue every call behind the held ones
                                    OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(self.home / "worktrees"))
        write_settings_file(self.data_root / "settings.json", settings)
        _seed_behavior(self.data_root)
        self.transport = _seed_transport(self.data_root, "stand-t0")
        hook_dir = self.home / "hook"
        hook_dir.mkdir()
        shutil.copyfile(HOOK, hook_dir / "sitecustomize.py")
        (self.home / "home").mkdir()
        self.server = _HookedServer(self.candidate, self.data_root, self.data_root / "settings.json", extra_env={
            "HOME": str(self.home / "home"), "PYTHONPATH": str(hook_dir), "STAND_DOOR_LOG": str(self.door_log),
            "STAND_REFUSE_SPAWN": str(self.refuse_flag), "STAND_MARKER_WINDOW": str(self.window_flag),
            "STAND_JOURNAL": self.journal, "STAND_SAMPLE_LOG": str(self.sample_log),
            "STAND_TREE": self.tree})
        started = time.monotonic()
        self.server.start(ready_timeout=900)
        self.facts["timings"]["boot_sec"] = round(time.monotonic() - started, 1)
        self.facts["server_pid"] = self.server.proc.pid
        self.oracle = ArtifactOracle(self.data_root)

    # ---- the owner's chat --------------------------------------------------------------------------------------------------
    def send(self, text: str, chat_id: int = 1) -> str:
        message_id = "stand-" + uuid.uuid4().hex
        self.ws.send(json.dumps({"type": "chat", "content": text, "client_message_id": message_id, "chat_id": chat_id}))
        return message_id

    def turn_of(self, message_id: str) -> Dict[str, Any]:
        """The turn its carrier received: a carrier may have to boot first, a minute on a loaded Mac, so this bound is a turn's."""
        return wait_until(lambda: _received_turn(self.oracle, message_id), SETTLE_SEC) or {}

    def finished_turn(self, text: str) -> Dict[str, Any]:
        """One owner turn to its stored end, with the wall clock it took (a fact)."""
        started = time.monotonic()
        task = self.turn_of(self.send(text))
        stored = wait_durable_result(self.oracle, task["id"], timeout=SETTLE_SEC) if task.get("id") else {}
        return {"task_id": task.get("id", ""), "status": stored.get("status"), "seconds": round(time.monotonic() - started, 1),
                "execution_owner": stored.get("execution_owner") or {}, "reason_code": stored.get("reason_code"),
                "answer": str(stored.get("result") or "")[:400]}

    def receipt(self, task_id: str) -> Dict[str, Any]:
        """A turn's receipt: its row lands about when the turn's result does, in either order."""
        rows = wait_until(lambda: [row for row in self.oracle.supervisor_rows("direct_turn") if row.get("task_id") == task_id], READ_WINDOW_SEC)
        return rows[-1] if rows else {}

    def held_turn(self, text: str, chat_id: int = 1) -> Dict[str, Any]:
        """One owner turn held at the model, with the process that carries it (0: no custodied process does)."""
        before = self.turns.arrived
        task = self.turn_of(self.send(f"{TURN_MARKER} {text}", chat_id))
        wait_until(lambda: self.turns.arrived > before, SETTLE_SEC)
        task_id = task.get("id", "")
        return {"task_id": task_id, "carrier": _carrier_pid(self.oracle, self.data_root, task_id)}

    def probe_door(self, path: str, method: str = "GET", payload: Any = None) -> Dict[str, Any]:
        answer = _call(self.server.port, method, path, payload, timeout=READ_WINDOW_SEC)
        return {"status": answer["status"], "sec": answer["sec"], "error": answer["error"], "reason_code": answer["body"].get("reason_code")}

    def pause_door(self, task_id: str) -> Dict[str, Any]:
        """The owner's Pause as its contract says to send it: one request id, again after a 503 (the door reads the store with no wait,
        so that it never stalls behind it). How many answers it took is a fact; the law is that the Pause is taken."""
        refused: List[Any] = []
        deadline = time.monotonic() + READ_WINDOW_SEC
        while True:
            answer = self.probe_door(f"/api/tasks/{task_id}/pause", "POST", {"request_id": "stand-pause"})
            if answer["status"] != 503 or time.monotonic() >= deadline:
                return {**answer, "refused_first": refused}
            refused.append(answer["reason_code"])
            time.sleep(0.5)

    def wake_seen(self) -> bool:
        return any(row.get("door") == "_launch" for row in _doors(self.door_log))

    # ---- load: everything at once, the door and the controls under it ----------------------------------------------------------
    def at_work(self, count: int) -> List[str]:
        """The restored rows the pool took, each held at the model: real workers, real tasks, the rest of the queue behind them."""
        def rows() -> List[str]:  # a wake or a turn may be at work too, and a wake's own text may quote these rows
            return [row["id"] for row in self.oracle.queue_snapshot().get("running") or []
                    if row.get("worker_id") is not None and str(row["id"]).startswith("live-")]

        wait_until(lambda: self.holds.arrived >= count and len(rows()) >= count, BOOT_SEC)
        return rows()

    def under_load(self) -> None:
        tasks = self.at_work(self.shape.real_tasks)
        # Two owner messages in one chat are one turn while it reads: the held turns are other chats', the measured ones Main's.
        held = [self.held_turn(f"a held owner turn {index}: answer in one line.", OTHER_CHAT + index) for index in range(self.shape.turns)]
        snapshot = self.oracle.queue_snapshot()
        self.facts["held_tasks"], self.facts["held_turns"] = tasks, held
        self.facts["arrivals"] = {"tasks": self.holds.arrived, "turns": self.turns.arrived, "pending": snapshot.get("pending_count")}
        burst = [_Turn(self.server.host_service_port, self.transport,
                       _event_body(self.transport, f"stand-event-{uuid.uuid4().hex}", f"stand-room-{index}", f"Stand event {index}: one sentence."),
                       f"e{index}") for index in range(self.shape.burst)]
        for turn in burst:
            turn.start()
        doors = {"health": self.probe_door("/api/health"), "contract": self.probe_door("/api/contract"), "state": self.probe_door("/api/state"),
                 "pause": self.pause_door(tasks[-1])}
        self.facts["doors"] = doors
        beside = self.finished_turn("An owner turn beside the others: answer in one line.")
        self.facts["beside"] = {**beside, "receipt": self.receipt(beside["task_id"])}
        for turn in burst:
            turn.join(SETTLE_SEC)
        self.facts["burst"] = [{"status": turn.result.get("status"), "outcome": (turn.result.get("body") or {}).get("outcome"),
                                "sec": turn.result.get("sec")} for turn in burst]

    def panic_frame(self) -> None:
        """E5 and E9: the browser's Panic frame with everything still held: answered, the tree gone, the turn's carrier among it."""
        from ouroboros.config import PANIC_EXIT_CODE
        from ouroboros.platform_layer import pid_is_alive
        from ouroboros.process_containment import pids_with_env_marker
        from ouroboros.process_scope import installation_marker

        assert not {self.server.port, self.server.host_service_port} & LIVE_DEFAULT_PORTS, "Panic sweeps its own ports only"
        assert self.oracle.server_port() == self.server.port, "the bound port is not provably this server's"
        carriers = [turn["carrier"] for turn in self.facts["held_turns"]]
        workers = list(_worker_pids(self.data_root).values())
        # Panic's census is the installation's scope, read from the kernel: who it would end, measured before it runs. Its own
        # journal row races the hard exit by design, so it can only be a fact.
        census = set(pids_with_env_marker(installation_marker(self.data_root, create=False)) or ())
        started = time.monotonic()
        self.ws.send(json.dumps({"type": "command", "cmd": "/panic"}))
        accepted = wait_until(lambda: self.frames.find(type="panic_accepted"), READ_WINDOW_SEC)
        answered_sec = round(time.monotonic() - started, 3)
        exited = wait_until(lambda: self.server.proc.poll() is not None, READ_WINDOW_SEC)
        self.facts["panic"] = {"accepted": bool(accepted), "answered_sec": answered_sec, "exited": bool(exited), "code": self.server.proc.poll(),
                               "expected_code": PANIC_EXIT_CODE, "flag": (self.data_root / "state" / "panic_stop.flag").is_file(),
                               "carriers": carriers, "carriers_gone": bool(carriers) and all(pid and not pid_is_alive(pid) for pid in carriers),
                               "carriers_in_census": bool(carriers) and all(pid in census for pid in carriers),
                               "workers": workers, "workers_in_census": bool(workers) and all(pid in census for pid in workers),
                               "census_rows": self.oracle.supervisor_rows("panic_scope_census"),
                               "tree_gone": bool(wait_until(lambda: pids_with_env_marker(self.tree) == [], 30))}

    def load(self) -> None:
        self.facts["wake_door_seen"] = bool(wait_until(self.wake_seen, 300))
        alone = self.finished_turn("An owner turn alone: answer in one line.")
        self.facts["alone"] = {**alone, "receipt": self.receipt(alone["task_id"])}
        self.under_load()
        self.facts["doors_seen"] = _doors(self.door_log)
        self.panic_frame()

    # ---- recovery: the conversation with no pool ------------------------------------------------------------------------------
    def storm(self) -> None:
        """E7: three project workers at work, kill -9 at once, then the owner says hello: the lane answers while each
        killed task takes its one retry; no crash-storm fence freezes the pool (ARCHITECTURE §5)."""
        pids = [_carrier_pid(self.oracle, self.data_root, task_id) for task_id in self.at_work(STORM_WORKERS)]
        self.facts["storm_workers"] = sum(bool(pid) for pid in pids)
        for pid in pids:
            if pid:
                os.kill(pid, signal.SIGKILL)
        dead_rows = wait_until(lambda: len(self.oracle.supervisor_rows("worker_dead_detected")) >= STORM_WORKERS
                               and self.oracle.supervisor_rows("worker_dead_detected"), SETTLE_SEC)
        hello = self.finished_turn("Hello after the storm: answer in one line.")
        self.facts["storm"] = {**hello, "dead_rows": len(dead_rows or []),
                               "fence": self.oracle.queue_snapshot().get("worker_pool_disabled_reason") or "",
                               "pending_ids": [row.get("id") for row in self.oracle.queue_snapshot().get("pending") or []]}

    def no_worker_left(self) -> None:
        """Every worker killed, and the one a spawn had in flight when the fact began (a turn may rightly take a carrier that came
        through). Quiet for ``QUIET_SEC`` of the ledger means none is coming: the fact stands, and it forbids any new one."""
        quiet_since = None
        deadline = time.monotonic() + READ_WINDOW_SEC
        while time.monotonic() < deadline:
            pids = list(_worker_pids(self.data_root).values())
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            quiet_since = None if pids else (quiet_since or time.monotonic())
            if quiet_since and time.monotonic() - quiet_since >= QUIET_SEC:
                return
            time.sleep(0.2)

    def no_carrier_can_start(self, flag: pathlib.Path, text: str) -> Dict[str, Any]:
        """E7: the physical fact holds and no process is left to take the turn: the owner is answered all the same, and told how."""
        flag.write_text("closed", encoding="utf-8")
        self.no_worker_left()
        turn = self.finished_turn(text)
        flag.unlink()
        wait_until(lambda: _worker_pids(self.data_root), SETTLE_SEC)  # the fact is over only when a process stands again: the next step's carrier is coming
        task_id = turn["task_id"]
        return {**turn, "disclosed_row": [row for row in self.oracle.supervisor_rows("conversation_inline_executor")
                                          if row.get("task_id") == task_id and row.get("executor") == DEGRADED],
                "notice": [frame for frame in self.frames.frames if frame.get("task_id") == task_id and frame.get("task_incident")],
                "calls_when_answered": len(self.stub.calls)}

    def blocked_spawn(self) -> None:
        self.facts["blocked"] = self.no_carrier_can_start(self.refuse_flag, "Hello while no process can start: answer in one line.")

    def marker_window(self) -> None:
        self.facts["window"] = self.no_carrier_can_start(self.window_flag, "Hello while an update's conflict markers are in the tree: answer in one line.")

    def later_requests_name_it(self) -> None:
        """The Runtime vitals the mind reads: one more turn, whose request carries each degraded turn."""
        self.finished_turn("One more owner turn: answer in one line.")
        for step in NO_CARRIER:
            later = self.stub.calls[self.facts[step]["calls_when_answered"]:]
            row = f"{DEGRADED} turn {self.facts[step]['task_id']}"  # the vitals' own line; the bare word is in every request's documents
            self.facts[step]["later_requests"] = len(later)
            self.facts[step]["vitals"] = any(row in json.dumps(body) for _kind, body in later)

    def stop_a_turn(self) -> None:
        """E9: Stop of a held turn: the carrier by custody, its death measured."""
        from ouroboros.platform_layer import pid_is_alive

        held = self.held_turn("a turn the owner stops.")
        answer = self.probe_door(f"/api/tasks/{held['task_id']}/cancel", "POST", {"cascade": True})
        gone = bool(held["carrier"]) and bool(wait_until(lambda: not pid_is_alive(held["carrier"]), SETTLE_SEC))
        # Finalize, then kill: at the moment the carrier is gone its row is already terminal.
        at_death = self.oracle.task_result(held["task_id"]).get("status") if gone else ""
        self.facts["stop"] = {**held, "answer": answer, "carrier_gone": gone, "status_at_death": at_death}

    def recovery(self) -> None:
        self.storm()
        self.blocked_spawn()
        self.marker_window()
        self.stop_a_turn()
        self.later_requests_name_it()
        self.facts["doors_seen"] = _doors(self.door_log)

    # ---- one server ---------------------------------------------------------------------------------------------------------
    def run(self, steps: Callable[[], None]) -> Dict[str, Any]:
        from websockets.sync.client import connect

        # This process reads a root its own processes stamped; the fake engine answers for the owned daemon, whose descriptor
        # the server's discovery prefers: nothing is provisioned, nothing leaves the machine.
        with self.stub, FakeClaudexorDaemon() as engine, as_install(self.data_root):
            self.build()
            engine.install(self.data_root / "claudexor")
            self.start()
            try:
                with connect(ws_url(self.server), open_timeout=60, proxy=None) as self.ws, _WsFrames(self.ws) as self.frames:
                    steps()
            finally:
                self.holds.release.set()
                self.turns.release.set()
                self.facts["samples"] = json.loads(self.sample_log.read_text(encoding="utf-8")) if self.sample_log.exists() else {}
                self.server.stop()
                self.facts["supervisor_load"] = _row_sizes(self.oracle.supervisor_rows("supervisor_load"))
        return self.facts


def _row_sizes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What a minute's row weighs in its journal: a fact for that journal's reader."""
    sizes = [len(json.dumps(row)) for row in rows]
    return {"rows": len(rows), "max_bytes": max(sizes, default=0), "mean_bytes": sum(sizes) // max(1, len(sizes))}


def _report_facts(facts: Dict[str, Any]) -> None:
    """Timings decide nothing: they are the receipt a reader compares with the owner's numbers."""
    target = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT", "")  # the launcher keeps this folder past the run
    reported = {phase: {**{key: value for key, value in part.items() if key != "doors_seen"},
                        "cognition_in_the_server": sorted({f"{row['door']} on {row['thread'].split('-')[0]}, from {row['caller']}" for row in _server_rows(part)})}
                for phase, part in facts.items()}
    print(json.dumps(reported, indent=1, default=str)[:20_000])
    if target:
        name = f"stand-server-{facts['load']['journal']}-{facts['load']['shape']}.json"
        (pathlib.Path(target) / name).write_text(json.dumps(reported, indent=1, default=str), encoding="utf-8")


@pytest.fixture(scope="module", params=[pytest.param(("WAL", "reduced"), id="wal-reduced"), pytest.param(("PERSIST", "reduced"), id="persist-reduced"),
                                        pytest.param(("WAL", "full"), id="wal-full")])
def scenario(request, tmp_path_factory) -> Dict[str, Any]:
    journal, shape = request.param
    require_lane(LANE_MOCK if shape == "reduced" else LANE_FULL)
    home = tmp_path_factory.mktemp("stand")
    clone = clone_repo(home)
    load = _Run(clone, home / "load", shape, journal)
    recovery = _Run(clone, home / "recovery", "reduced", journal, workers=STORM_WORKERS)
    try:
        facts = {"load": load.run(load.load), "recovery": recovery.run(recovery.recovery)}
    except AssertionError as broke:  # the harness's own waits
        raise ScenarioBroke(str(broke)) from broke
    _report_facts(facts)
    return facts


# ---- E1: placement ---------------------------------------------------------------------------------------------------------

def _server_rows(part: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Doors of cognition that ran in the server's own process; the state endpoint's panel reads the wake's facts (a display)."""
    return [row for row in part["doors_seen"] if row["pid"] == part["server_pid"] and row["door"] in COGNITION_DOORS
            and row.get("caller") != "status_snapshot"]


def test_s43_e1_no_server_stack_ran_cognition_under_load(scenario):
    """Normal operation: owner turns, a wake and a Presence burst, and no door of cognition in the server's process."""
    rows = _server_rows(scenario["load"])
    assert not rows, sorted({(row["door"], row["thread"].split("-")[0], row["caller"]) for row in rows})[:12]


def test_s43_e1_only_the_disclosed_row_ran_in_the_server_during_recovery(scenario):
    """With no process able to start, each such turn runs on a server thread and says so; nothing else does."""
    recovery = scenario["recovery"]
    degraded = {recovery[step]["task_id"] for step in NO_CARRIER if recovery[step]["execution_owner"].get("executor") == DEGRADED}
    rows = [row for row in _server_rows(recovery) if row.get("task_id") not in degraded]
    assert not rows, sorted({(row["door"], row["thread"].split("-")[0], row["caller"]) for row in rows})[:12]


def test_s43_e1_the_spy_saw_the_doors_where_they_ran(scenario):
    """The laws above read nothing unless the spy works: real tasks ran ``handle_task`` in workers, turns built their context."""
    load = scenario["load"]
    rows = load["doors_seen"]
    assert [row for row in rows if row["door"] == "handle_task" and row["pid"] != load["server_pid"]], rows[:6]
    assert [row for row in rows if row["door"] == "build_llm_messages"], rows[:6]


def test_s43_e1_a_due_wake_was_met(scenario):
    """The stand's alarm falls due within a minute of its boot, so a wake's doors are in the record above."""
    assert scenario["load"]["wake_door_seen"]


# ---- E5: the door and the controls ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("door", ["health", "contract", "state", "pause"])
def test_s43_e5_the_door_and_the_controls_answer_under_load(scenario, door):
    answer = scenario["load"]["doors"][door]
    assert 200 <= answer["status"] < 300, (door, answer, scenario["load"]["arrivals"])


def test_s43_e5_the_load_was_really_there(scenario):
    load = scenario["load"]
    shape = SHAPES[load["shape"]]
    # A wake due meanwhile quotes what it observes, so its request may wait at a gate too: the arrivals are at least these.
    assert len(load["held_tasks"]) == shape.real_tasks <= load["arrivals"]["tasks"], load["arrivals"]
    assert load["arrivals"]["turns"] >= shape.turns and all(turn["task_id"] for turn in load["held_turns"]), load["held_turns"]
    assert load["arrivals"]["pending"] >= shape.live.pending, load["arrivals"]


def test_s43_e5_the_burst_was_answered(scenario):
    load = scenario["load"]
    answered = [row for row in load["burst"] if row["status"] == 200]
    assert len(answered) == SHAPES[load["shape"]].burst, load["burst"]


def test_s43_e5_panic_is_answered_under_load_and_the_whole_tree_is_gone(scenario):
    panic = scenario["load"]["panic"]
    assert panic["accepted"] and panic["exited"], panic
    assert panic["code"] == panic["expected_code"] and panic["flag"] and panic["tree_gone"], panic


# ---- E6, as facts here (its counted law: tests/test_stand_counts.py) ----------------------------------------------------------

@pytest.mark.parametrize("turn", ["alone", "beside"])
def test_s43_e6_a_receipt_names_what_the_owner_waited(scenario, turn):
    part = scenario["load"][turn]
    assert part["status"] == "completed", part
    assert part["receipt"].get("type") == "direct_turn" and part["receipt"].get("dispatch"), part["receipt"]


# ---- E7: the recovery channel ----------------------------------------------------------------------------------------------

def test_s43_e7_after_a_worker_storm_the_next_hello_is_answered_and_never_waits_in_pending(scenario):
    recovery = scenario["recovery"]
    storm = recovery["storm"]
    assert recovery["storm_workers"] == STORM_WORKERS and storm["dead_rows"] >= STORM_WORKERS, (recovery["storm_workers"], storm)
    assert storm["fence"] == "", storm  # three deaths fence nothing: the pool recovers each task once
    assert storm["status"] == "completed", storm
    assert storm["task_id"] not in storm["pending_ids"], storm


def test_s43_e7_the_hello_after_a_storm_was_admitted_to_the_lane(scenario):
    assert scenario["recovery"]["storm"]["execution_owner"].get("lane") == "conversation", scenario["recovery"]["storm"]


@pytest.mark.parametrize("step", NO_CARRIER)
def test_s43_e7_with_no_carrier_able_to_start_the_owner_is_still_answered(scenario, step):
    assert scenario["recovery"][step]["status"] == "completed", scenario["recovery"][step]


@pytest.mark.parametrize("step", NO_CARRIER)
@pytest.mark.parametrize("surface", ["receipt", "journal", "chat", "vitals"])
def test_s43_e7_a_degraded_turn_is_disclosed(scenario, step, surface):
    part = scenario["recovery"][step]
    told = {"receipt": part["execution_owner"].get("executor") == DEGRADED, "journal": bool(part["disclosed_row"]),
            "chat": any(frame.get("task_incident") == "conversation_degraded_executor" for frame in part["notice"]),
            "vitals": part["later_requests"] > 0 and part["vitals"]}
    assert told[surface], (surface, part)


def test_s43_e7_nothing_spawns_in_a_marker_window(scenario):
    starts = [row for row in scenario["recovery"]["doors_seen"] if row["door"] == "spawn_in_marker_window"]
    assert not starts, sorted({row["caller"] for row in starts})


# ---- E9: Stop and Panic ----------------------------------------------------------------------------------------------------

def test_s43_e9_stop_is_answered_while_the_turn_waits_on_its_model(scenario):
    stop = scenario["recovery"]["stop"]
    assert 200 <= stop["answer"]["status"] < 300, stop


def test_s43_e9_stop_finalizes_then_kills_the_turn_s_carrier(scenario):
    stop = scenario["recovery"]["stop"]
    assert stop["carrier"] and stop["carrier_gone"] and stop["status_at_death"] == "cancelled", stop


def test_s43_e9_panic_s_census_covers_every_worker(scenario):
    """The census law's instrument, on what exists today: the pool's workers are all in the scope Panic ends."""
    panic = scenario["load"]["panic"]
    assert panic["workers"] and panic["workers_in_census"], panic


def test_s43_e9_panic_kills_each_turn_s_carrier_by_measured_identity(scenario):
    panic = scenario["load"]["panic"]
    assert panic["carriers"] and all(panic["carriers"]) and panic["carriers_in_census"] and panic["carriers_gone"], panic


# ---- the instrument ---------------------------------------------------------------------------------------------------------

def test_the_hook_is_loadable_on_its_own():
    """It runs in the server tree before anything of the tree is importable, so it must stand alone."""
    source = HOOK.read_text(encoding="utf-8")
    compile(source, str(HOOK), "exec")
    assert "ouroboros" not in {line.split()[1].split(".")[0] for line in source.splitlines() if line.startswith(("import ", "from "))}
    assert sys.version_info >= (3, 10)
