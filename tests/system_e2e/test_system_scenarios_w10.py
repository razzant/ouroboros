"""Wave 10 — S40: the owner's Pause, Resume and Stop hold one live task on its own id (#1729 / #1730).

A keyless real server runs from a byte-faithful checkout of the tree under test (S33's
``s33_candidate``: working-tree edits included, its process tree held until proven gone; a
HEAD clone only for a clean tree) with ONE real worker process. The owner acts only through the
HTTP doors the UI posts to (Pause, Resume, Stop); the queue ``put``, assignment, the exact pause,
its park and Resume, the launch gates and cancellation custody are the shipped ones.

Doubled, and only these: the model is the loopback stub, and the round the owner catches is held
by an event-gated ``ModelGate`` (never a timed race).

No retry is involved: since #1376 silence never ends an attempt, and the reaper ends only a hard
bound, without a successor, so the task keeps its one id from submission to Stop.
``after_owner_resume``: the owner Pauses, the attempt parks on its exact pause, the owner Resumes,
and the SAME id runs again, consumes its grant and continues its saved work (#1729).
``ordinary``: without that round trip. Both then Pause the RUNNING task while its model round is
held. A write before the Pause lands. The Pause names the logical root and fences it. A write the
model asks for after the acceptance never lands. One Resume reopens the tree, and Stop ends it
Cancelled with no late effect (#1730).

S44 (#3396): a server started in the launcher's shape (stdin the quit pipe, stdout and stderr
one pipe, the launcher's mark); once it is ready, output that piled up while the launcher stopped
reading fills that pipe and nobody reads it until the server has exited. A task still completes
through the worker and Quit exits 0. On the base tree the same task never reaches a terminal
status: the run behind that full pipe is the negative control.

Not covered here, by design:
- crash recovery: the same-id crash rail is ``tests/test_worker_crash_retry.py`` and
  ``tests/test_working_recovery_relay.py``;
- the silence rail's negative control (a silent live attempt is attention, never a kill or a
  successor): ``tests/test_silence_attention.py::test_heartbeat_only_never_finalizes_or_reaps``
  and ``tests/test_acceptance_fence_attempts.py::test_live_silence_keeps_the_exact_active_fence_and_queues_no_reap``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
import time
import uuid

import pytest

from devtools.benchmarks.common.server_runner import IsolatedServer, _api_status
from tests.candidate_checkout import assert_served_candidate, verify_checkout
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    KeylessIsolatedServer,
    ModelGate,
    ScriptedStubModel,
    keyless_settings,
    require_lane,
    start_server,
    submit_running,
    wait_durable_result,
    wait_until,
    write_settings_file,
)
from tests.system_e2e.test_presence_resilience import s33_candidate  # noqa: F401 - pytest fixture

KEEPALIVE = {"tool": "list_files", "arguments": {"path": "."}}
CONTROL = {"tool": "write_file", "arguments": {"root": "task_drive", "path": "s40-control.txt",
                                               "content": "written before the Pause"}}
SENTINEL = {"tool": "write_file", "arguments": {"root": "task_drive", "path": "s40-sentinel.txt",
                                                "content": "a task effect after the accepted Pause"}}
SETTINGS = {"OUROBOROS_MAX_WORKERS": 1}


class _PauseStub(ScriptedStubModel):
    """Keepalive rounds; the scenario queues one-shot steps and records when each was served."""

    def __init__(self) -> None:
        super().__init__(latency_sec=0.2)
        self.steps, self.served = [], []

    def _next_step(self, _body):
        if self.steps:
            step = self.steps.pop(0)
            self.served.append((time.time(), step["arguments"]["path"]))
            return step
        return dict(KEEPALIVE)


def _post(server, path: str, body: dict | None = None) -> dict:
    return _api_status(server.base_url, "POST", path, body or {}, timeout=60)


def _pause_row(data, task_id: str) -> dict:
    from ouroboros.budget_pause import budget_pause_row

    try:
        return budget_pause_row(data, task_id) or {}
    except Exception as exc:  # an unreadable row is a fact for the forensics, not a crash of the reader
        return {"unreadable": repr(exc)}


def _rows(oracle) -> list:
    snapshot = oracle.queue_snapshot()
    return [dict(row.get("task") or {}, id=row.get("id"), queue=state)
            for state in ("pending", "running") for row in snapshot.get(state) or [] if isinstance(row, dict)]


def _foreign_rows(oracle, origin: str) -> list:
    """Queue rows of any other id: a successor, a copy, a retry. The task owns exactly one id."""
    return [row for row in _rows(oracle) if row["id"] != origin]


def _found(data, name: str) -> list:
    return sorted(str(path) for path in data.rglob(name))


def _first_agent_round_since(stub, start: int) -> dict:
    with stub._lock:
        return next((body for _kind, body in stub.calls[start:] if body.get("tools")), {})


def _facts(oracle, data, origin: str, stub, **extra) -> dict:
    from ouroboros.owner_pause import read_fence

    keys = ("id", "queue", "_attempt", "root_task_id", "_budget_pause", "_budget_pause_resume",
            "_budget_pause_hold", "timeout_retry_from")
    return {
        "rows": [{key: row.get(key) for key in keys if row.get(key) is not None} for row in _rows(oracle)],
        "pause_row": {key: _pause_row(data, origin).get(key) for key in (
            "state", "reason", "pause_id", "task_attempt", "unreadable")},
        "fence": read_fence(data, origin),
        "result": {key: oracle.task_result(origin).get(key) for key in ("status", "reason_code")},
        "stub_calls": len(stub.calls), "stub_served": stub.served, **extra,
    }


@pytest.mark.integration
@pytest.mark.parametrize("history", ["after_owner_resume", "ordinary"])
def test_s40_the_owner_pause_resume_and_stop_hold_one_task_on_its_own_id(s33_candidate, tmp_path_factory,  # noqa: F811
                                                                           history):
    require_lane(LANE_MOCK)
    from ouroboros.budget_pause import STATE_PAUSED, STATE_RESUMED
    from ouroboros.owner_pause import read_fence

    root = tmp_path_factory.mktemp(f"s40_{history}")
    gate = None
    with _PauseStub() as stub:
        server = start_server(s33_candidate, root, keyless_settings(stub, **SETTINGS))
        data, oracle = server.data_root, ArtifactOracle(server.data_root)
        facts = {"history": history, "tree": getattr(s33_candidate, "identity", str(s33_candidate))}
        try:
            origin = submit_running(server, "Keep listing the repository files. [S40]")
            facts["origin"] = origin
            assert wait_until(lambda: len(stub.calls) >= 3, 60), stub.kinds()
            if history == "after_owner_resume":
                paused = _post(server, f"/api/tasks/{origin}/pause", {"request_id": f"s40-{uuid.uuid4().hex}"})
                assert paused["status"] in (200, 202), paused
                # Settled: the exact row, the durable fence and the queue all say paused.
                assert wait_until(lambda: _pause_row(data, origin).get("state") == STATE_PAUSED
                                  and read_fence(data, origin).get("state") == "paused"
                                  and origin not in oracle.running_ids(), 120), _facts(oracle, data, origin, stub)
                at_pause = len(stub.calls)
                granted = _post(server, f"/api/tasks/{origin}/resume")
                assert granted["status"] == 200 and granted["body"].get("ok"), granted
                # #1729: the SAME id runs again, its loop consumes the grant, and it continues the saved work.
                assert wait_until(lambda: origin in oracle.running_ids() and _first_agent_round_since(stub, at_pause),
                                  120), _facts(oracle, data, origin, stub)
                assert wait_until(lambda: _pause_row(data, origin).get("state") == STATE_RESUMED, 60), \
                    _facts(oracle, data, origin, stub)
                body = _first_agent_round_since(stub, at_pause)
                earlier = {f"call_{seq}" for seq in range(1, at_pause + 1)}
                history_ids = {call.get("id") for message in body.get("messages") or [] if isinstance(message, dict)
                               for call in message.get("tool_calls") or []} | {
                    message.get("tool_call_id") for message in body.get("messages") or [] if isinstance(message, dict)}
                assert history_ids & earlier, "the Resume restarted instead of continuing the saved work"
                assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)

            # #1730: a write before the Pause lands (the door is real), then the round is held.
            stub.steps.append(dict(CONTROL))
            facts["control"] = wait_until(lambda: _found(data, "s40-control.txt"), 60)
            assert facts["control"], _facts(oracle, data, origin, stub)
            gate = stub.gate = ModelGate(lambda body: bool(body.get("tools")), timeout=120)
            assert gate.arrived.wait(60), _facts(oracle, data, origin, stub)
            accepted = _post(server, f"/api/tasks/{origin}/pause", {"request_id": f"s40-{uuid.uuid4().hex}"})
            assert accepted["status"] in (200, 202), accepted
            facts["pause"] = accepted["body"]
            stub.steps.append(dict(SENTINEL))  # the model asks for a task effect after the acceptance
            gate.release.set()
            # Resume observes settled authority: requested -> paused during its census must refuse.
            parked = wait_until(lambda: (_pause_row(data, origin).get("state") == STATE_PAUSED
                                        and read_fence(data, origin).get("state") == "paused")
                                or _found(data, "s40-sentinel.txt"), 90)
            facts.update(after_pause=_facts(oracle, data, origin, stub))
            assert any(path == "s40-sentinel.txt" for _ts, path in stub.served), "the write was never asked for"
            assert not _found(data, "s40-sentinel.txt"), facts
            assert accepted["body"].get("root_task_id") == origin, accepted
            assert parked and origin not in oracle.running_ids(), facts
            assert read_fence(data, origin).get("state") == "paused" and not _foreign_rows(oracle, origin), facts

            # One Resume reopens the logical tree on the same id; Stop ends it.
            resumed = _post(server, f"/api/tasks/{origin}/resume")
            assert resumed["status"] == 200 and resumed["body"].get("root_task_id") == origin, resumed
            assert wait_until(lambda: read_fence(data, origin).get("state") == "released", 60), \
                _facts(oracle, data, origin, stub)
            assert wait_until(lambda: origin in oracle.running_ids(), 120), _facts(oracle, data, origin, stub)
            assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)
            stopped = server.cancel_task(origin)
            assert stopped["status"] in (200, 202), stopped
            facts["terminal"] = wait_durable_result(oracle, origin, timeout=180).get("status")
            assert facts["terminal"] == "cancelled", facts
            assert not _found(data, "s40-sentinel.txt"), "no task effect of the paused attempt appeared later"
            assert not _foreign_rows(oracle, origin), _facts(oracle, data, origin, stub)
        finally:
            if gate is not None:
                gate.release.set()
            print("S40_FACTS", json.dumps(facts, default=str))
            server.stop()


S44_READY_SEC = 300.0  # a transport bound: a spawn on a loaded Mac takes up to a minute
S44_TASK_SEC = 180.0  # a transport bound for one scripted task
# The launcher's mark, set by the server process itself: ProcessContainer keeps that authority from
# every contained child, and without OUROBOROS_MANAGED_REPO_DIR the checkout is no managed bootstrap target.
_LAUNCHER_SHAPE = ("import os, sys, runpy, multiprocessing.process\n"
                   "os.environ['OUROBOROS_MANAGED_BY_LAUNCHER'] = '1'\n"
                   "start = multiprocessing.process.BaseProcess.start\n"
                   "def diagnostic_start(process, *args, **kwargs):\n"
                   "    sys.stdout.write('S44 buffered diagnostic before Process.start')\n"
                   "    return start(process, *args, **kwargs)\n"
                   "multiprocessing.process.BaseProcess.start = diagnostic_start\n"
                   "runpy.run_path('server.py', run_name='__main__')\n")
# A single ordinary diagnostic exercises the incident's real _flush_std_streams
# consumer. An otherwise silent boot can finish even without the relay, depending
# on whether startup warnings happened before or after the pipe filled.
# Output that piled up while the launcher stopped reading: more than any pipe holds, so it waits.
_PILED_UP = "import os\nos.write(1, b'piled up ' * 8192)\n"


class _LauncherShapedServer(KeylessIsolatedServer):
    """A keyless server started as the desktop launcher starts its own (``launcher.start_agent``):
    stdin the private quit pipe, stdout and stderr one pipe (``output``, whose read end the
    scenario holds), the launcher's mark."""

    def __init__(self, *args, output: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output = output

    def start(self, ready_timeout: float = S44_READY_SEC):
        from ouroboros.process_containment import ProcessContainer

        if self.candidate is not None:
            verify_checkout(self.clone, self.candidate)
        self._patch_settings_ports()
        env = self._env()
        env["OUROBOROS_LAUNCHER_STOP_STDIN"] = "1"
        self._candidate_container, self._released = ProcessContainer(), False
        if self.candidate is not None:
            self.candidate.hold()
        self.proc = self._candidate_container.spawn(
            [sys.executable, "-c", _LAUNCHER_SHAPE], cwd=self.clone, env=env,
            stdin=subprocess.PIPE, stdout=self.output, stderr=self.output)
        try:
            self._wait_ready(ready_timeout)
            if self.candidate is not None:
                assert_served_candidate(self.base_url, self.clone, self.data_root, self.proc.pid, self.candidate)
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self) -> None:
        container, self._candidate_container = self._candidate_container, None
        try:
            IsolatedServer.stop(self)
        finally:
            error = ""
            if container is not None:
                try:
                    error = container.reap()
                finally:
                    container.close()
            if error:
                from ouroboros.test_environment import retain_tree

                retain_tree(self.data_root, error)
                raise AssertionError(f"S44 process custody unconfirmed: {error}")
            if self.candidate is not None and not getattr(self, "_released", True):
                self._released = True
                self.candidate.release()


def _unread_bytes(fd: int) -> int:
    import array
    import fcntl
    import termios

    count = array.array("i", [0])
    fcntl.ioctl(fd, termios.FIONREAD, count)
    return count[0]


def _run_s44(tree, root: pathlib.Path) -> dict:
    """One launcher-shaped server; once it is ready its pipe is full and nobody reads it until the
    end, as after the launcher stopped. A task and Quit must still complete. Returns the facts."""
    import os

    from ouroboros.process_containment import ProcessContainer
    from ouroboros.runtime_limits import LAUNCHER_STOP_GRACE_SEC

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    facts: dict = {"tree": getattr(tree, "identity", str(tree))}
    output = bytearray()
    read_end, write_end = os.pipe()
    filler = ProcessContainer()
    with ScriptedStubModel(final_answer="Final answer: S44 complete.") as stub:
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=1,
                                    OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(root / "worktrees"))
        write_settings_file(data / "settings.json", settings)
        server = _LauncherShapedServer(tree, data, data / "settings.json", output=write_end)
        started = time.monotonic()
        try:
            server.start()
            facts["ready_after"] = round(time.monotonic() - started, 1)
            piled = filler.spawn([sys.executable, "-I", "-c", _PILED_UP], stdout=write_end, stderr=subprocess.DEVNULL)
            os.close(write_end)
            write_end = -1
            # The pipe holds no more (macOS 16 KiB, Linux 64 KiB) and the filler waits on it.
            facts["pipe_full"] = bool(wait_until(lambda: _unread_bytes(read_end) >= 16000, 30)) and piled.poll() is None
            facts["unread_bytes"] = _unread_bytes(read_end)
            task_id = server.submit("Answer in one line. [S44]")
            facts["task"] = wait_durable_result(ArtifactOracle(data), task_id, timeout=S44_TASK_SEC).get("status")
            server.proc.stdin.write(b"quit\n")
            server.proc.stdin.flush()
            quit_at = time.monotonic()
            facts["exit"] = server.proc.wait(timeout=LAUNCHER_STOP_GRACE_SEC + 20)
            facts["exit_after"] = round(time.monotonic() - quit_at, 1)
        except BaseException as exc:
            facts["failed"] = f"{type(exc).__name__}: {exc}"[:600]
            raise
        finally:
            if write_end >= 0:
                os.close(write_end)

            def drain():  # the launcher reads again: whatever the pipe still holds
                while chunk := os.read(read_end, 65536):
                    output.extend(chunk)

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                filler.reap()
            finally:
                filler.close()
                server.stop()
                reader.join(30)
                os.close(read_end)
            facts["read_after_exit"] = len(output)
            from tests._stability_evidence import retain_server_evidence
            retain_server_evidence(data, facts, output)
            print("S44_FACTS", json.dumps(facts, default=str))
    return facts


@pytest.mark.integration
def test_s44_an_unread_launcher_pipe_holds_neither_the_server_nor_its_worker(s33_candidate, tmp_path):  # noqa: F811
    require_lane(LANE_MOCK)
    for generation in range(2):
        # Each launch reopens the same persisted data after positive Quit, with
        # its own children and a fresh full pipe. No live-install restart.
        facts = _run_s44(s33_candidate, tmp_path / "s44")
        assert facts["pipe_full"] and facts["task"] == "completed" and facts["exit"] == 0, (generation, facts)
        assert facts["read_after_exit"] > 0, facts


@pytest.mark.integration
def test_s44_disabling_the_buffer_holds_the_same_real_task(s33_candidate, tmp_path, monkeypatch):  # noqa: F811
    """Matched ablation on the candidate: only the logging bootstrap is disabled."""
    require_lane(LANE_MOCK)
    class HeldByOutput(Exception):
        pass

    def observe(oracle, task_id, **kwargs):
        time.sleep(20)
        record = oracle.task_result(task_id)
        assert record.get("status") == "scheduled", record
        print("S44_ABLATION", json.dumps({"task_id": task_id, "status": "scheduled", "seconds": 20}))
        raise HeldByOutput("buffer disabled: real task still pending behind full pipe")

    monkeypatch.setattr(sys.modules[__name__], "wait_durable_result", observe)
    name = "ouroboros/process_logging.py"
    original = (s33_candidate.path / name).read_bytes()
    changed = original.replace(b"    global _relay\n", b"    return False  # negative control: disable only the relay bootstrap\n    global _relay\n", 1)
    assert changed != original and not s33_candidate.unproven
    s33_candidate.overlay[name] = (original, (s33_candidate.path / name).stat().st_mode & 0o777)
    try:
        s33_candidate.rewrite(name, changed)
        root = tmp_path / "negative"
        (root / "data").mkdir(parents=True)
        (root / "data" / "ablation-process_logging.py").write_bytes(changed)
        print("S44_ABLATION_SOURCE", s33_candidate.checkout_identity)
        with pytest.raises(HeldByOutput):
            _run_s44(s33_candidate, root)
    finally:
        if not s33_candidate.unproven:
            s33_candidate.rewrite(name, original)
            del s33_candidate.overlay[name]
