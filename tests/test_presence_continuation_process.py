"""Real child Host death/exec, manual follow-up and controls with production agent bootstrap.

This is a TCP Host consumer, not a Slack provider or full server/supervisor boot.
Inference and reviewer transports are local scripts; lifecycle owners are real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

import httpx
import pytest

from ouroboros.test_environment import isolated_environment

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "lifecycle-fixture-token"
HEADERS = {"X-Skill-Token": TOKEN}
ANSWER = "The checked status report is ready."
REVISED_ANSWER = "The checked status report is ready; the independent report remains scheduled."
LATE_CHILD = "work-promoted-after-first-yield"
MANUAL = "I checked the prior delivery. Continue the interrupted report as a new turn."
pytestmark = [pytest.mark.serial, pytest.mark.skipif(
    os.name != "posix", reason="This fixture qualifies POSIX SIGKILL/process-group custody only")]


@pytest.fixture
def tmp_path(tmp_path):
    """Keep private multiprocessing sockets short, independent of runner basetemp.

    macOS limits AF_UNIX addresses to 104 bytes. The runner's nested pytest
    root can consume that before multiprocessing appends its own socket name.
    Retain this test's separate root and leave its address in pytest's tree,
    just like the other isolated session trees; never discard failure evidence.
    """
    root = Path(tempfile.mkdtemp(prefix="ob-pres-", dir="/tmp"))
    (tmp_path / "presence-root.txt").write_text(str(root), encoding="utf-8")
    yield root
    print(f"Presence fixture tree retained: {root}")


def eventually(read, *, seconds=45):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            value = read()
        except httpx.TransportError:
            value = None  # the socket may be bound before the server starts serving
        if value:
            return value
        time.sleep(.05)
    raise AssertionError("child Host condition did not settle")


class Host:
    def __init__(self, root, generation, mode, scenario="incoming"):
        self.root, self.generation = root, generation
        self.ready = root / f"ready-{generation}.json"
        env = isolated_environment(root, ROOT)
        self.log = (root / f"host-{generation}.log").open("w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tests.presence_lifecycle_host", str(root), str(generation), mode, scenario],
            cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.client = None
        try:
            def ready():
                assert self.proc.poll() is None, (root / f"host-{generation}.log").read_text()
                return json.loads(self.ready.read_text()) if self.ready.exists() else None
            self.info = eventually(ready)
            self.client = httpx.Client(base_url=self.info["url"], headers=HEADERS,
                                       trust_env=False, timeout=60)
            eventually(lambda: self.client.get("/identity").status_code == 200)
        except BaseException:
            self.close()
            raise

    def payload(self, event="original", text="Prepare the report.", version=1):
        return {
            "binding_id": self.info["binding"], "continuation_version": version,
            "event": {"source_event_id": event, "provider": "telegram", "account_id": "bot-1",
                      "conversation_id": "room-1", "thread_id": "topic-1",
                      "conversation_key": "ignored-by-host",
                      "actor": {"platform_actor_id": "user-7"}, "conversation": {},
                      "message": {"message_id": event}, "text": text}}

    def turn(self, event="original", text="Prepare the report.", version=1):
        return self.client.post("/presence/turn", json=self.payload(event, text, version))

    def work(self, ref):
        return self.client.get(f"/presence/work/{ref}", params={"binding_id": self.info["binding"]})

    def inputs(self):
        path = self.root / "model-inputs.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def restart(self):
        assert self.client.post("/fixture/restart").status_code == 200
        self.client.close()
        self.generation += 1
        self.ready = self.root / f"ready-{self.generation}.json"

        def ready():
            assert self.proc.poll() is None, "direct restart lost the original PID"
            return json.loads(self.ready.read_text()) if self.ready.exists() else None

        self.info = eventually(ready)
        self.client = httpx.Client(base_url=self.info["url"], headers=HEADERS, trust_env=False, timeout=60)
        eventually(lambda: self.client.get("/identity").status_code == 200)

    def close(self):
        if self.proc.poll() is None:
            if self.client is not None:
                try:
                    self.client.post("/fixture/shutdown")
                except httpx.HTTPError:
                    pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        if self.client is not None:
            self.client.close()
        self.log.close()
        # The child starts a private process group; no descendants may outlive it.
        try:
            os.killpg(self.proc.pid, 0)
        except ProcessLookupError:
            group_gone = True
        else:
            os.killpg(self.proc.pid, signal.SIGKILL)
            group_gone = False
        (self.root / f"exit-{self.generation}.json").write_text(json.dumps({
            "pid": self.proc.pid, "returncode": self.proc.returncode, "process_group_gone": group_gone}))
        assert group_gone, "child exited with a live descendant in its private group"


def test_host_close_rejects_a_live_descendant_after_the_parent_exits(tmp_path):
    """Support-process cleanup must not turn an actual orphan into a pass."""
    host = Host.__new__(Host)
    host.root, host.generation, host.client = tmp_path, 1, None
    host.log = (tmp_path / "host-1.log").open("w")
    host.proc = subprocess.Popen([
        sys.executable, "-c",
        "import os, subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "os._exit(0)",
    ], cwd=ROOT, env=isolated_environment(tmp_path, ROOT), stdout=host.log,
        stderr=subprocess.STDOUT, start_new_session=True)
    try:
        assert host.proc.wait(timeout=10) == 0
        with pytest.raises(AssertionError, match="live descendant"):
            host.close()
        assert json.loads((tmp_path / "exit-1.json").read_text()) == {
            "pid": host.proc.pid, "returncode": 0, "process_group_gone": False}
    finally:
        # Only the private group created above is ours. Reap even if the
        # negative control fails before Host.close reaches its own cleanup.
        try:
            os.killpg(host.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        host.proc.wait(timeout=10)
        host.log.close()

        def gone():
            try:
                os.killpg(host.proc.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                # macOS can briefly retain an unsignalable exiting group.
                # It still must disappear; EPERM is not evidence of cleanup.
                return False
            return False

        eventually(gone, seconds=10)


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_fixture_joins_owned_multiprocessing_support_before_exit(tmp_path, method):
    """Force the bus to exist; test cleanup independently of late-notice timing."""
    script = r'''
import json, os, sys
from pathlib import Path
os.environ["OUROBOROS_WORKER_START_METHOD"] = sys.argv[2]
from supervisor import queue
from supervisor.workers import get_event_q
from multiprocessing import forkserver, resource_tracker
from tests.presence_lifecycle_host import shutdown_fixture_processes
root = Path(sys.argv[1])
try:
    bus = get_event_q()
    bus.put({"fixture": True})
    assert bus.get(timeout=5) == {"fixture": True}
    shutdown_fixture_processes(root, 1)
    assert forkserver._forkserver._forkserver_pid is None
    record = json.loads((root / "process-cleanup-1.json").read_text())
    assert record["event_bus_children"] and record["resource_tracker_pid"], record
    assert record["remaining_children"] == [] and record["resource_tracker_joined"], record
    assert record["forkserver_joined"], record
    assert bool(record["forkserver_pid"]) == (sys.argv[2] == "forkserver"), record
finally:
    # A failing old fixture still must not leave its support processes behind.
    forkserver._forkserver._stop()
    resource_tracker._resource_tracker._stop()
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), method],
                            cwd=ROOT, env=isolated_environment(tmp_path, ROOT),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
def test_direct_exec_restart_interrupts_same_pid_author_without_replaying_effects(tmp_path, mode):
    from ouroboros.presence_continuation import continuation_status
    from ouroboros.review_operation import controller_identity, controller_state

    host = Host(tmp_path, 1, mode)
    try:
        initial = host.turn().json()
        assert initial["status"] == "continuing"
        ref = initial["continuation_ref"]
        record_path = tmp_path / "data" / "task_results" / f"{ref}.json"
        before = json.loads(record_path.read_text())
        original = before["presence_continuation"]["author_controller"]
        assert original == host.info["controller"] and original["birth"] and original["session"]
        # A genuine live controller in another process remains pending despite
        # its different custody session and absence from this process's registry.
        assert original["session"] != controller_identity()["session"]
        assert controller_state(original) == "alive" and continuation_status(before) == "running"
        assert host.work(ref).status_code == 202 and len(host.inputs()) == 2

        host.restart()
        successor = host.info["controller"]
        assert successor["pid"] == original["pid"] == host.proc.pid
        assert successor["birth"] == original["birth"]
        assert successor["session"] and successor["session"] != original["session"]
        assert host.client.get("/fixture/live").json() == [] and len(host.inputs()) == 2
        # Only the successor can prove replacement of its own local controller;
        # an unrelated observer's PID/birth probe still sees a live process.
        assert controller_state(original) == "alive"
        interrupted = host.work(ref)
        facts = {"mode": mode, "original_controller": original, "successor_controller": successor,
                 "initial": initial, "poll_status": interrupted.status_code, "poll": interrupted.json(),
                 "model_calls_after_exec": len(host.inputs()), "foreign_observer_state": controller_state(original)}
        (tmp_path / "exec-facts.json").write_text(json.dumps(facts, indent=2))
        assert interrupted.status_code == 200, interrupted.text
        assert interrupted.json()["status"] == "interrupted"
        assert interrupted.json()["text"] == "" and interrupted.json()["output_ref"] == ""
        assert interrupted.json()["outputs"] == before["presence_continuation"].get("outputs", [])
        for _ in range(2):
            assert host.turn().json() == initial
            assert host.work(ref).json() == interrupted.json()
        assert len(host.inputs()) == 2
        after = json.loads(record_path.read_text())
        assert after["status"] == before["status"] == "running"
        assert after["presence_continuation"] == before["presence_continuation"]

        manual = host.turn("manual-followup", MANUAL).json()
        assert manual["status"] == "completed" and manual["turn_ref"] != ref
        assert len(host.inputs()) == 3
        prompt = str(host.inputs()[-1]["messages"])
        assert ref in prompt and "interrupted" in prompt and MANUAL in prompt
        assert host.turn().json() == initial and len(host.inputs()) == 3
        facts.update(manual=manual, model_calls_after_manual=3, original_replay_unchanged=True)
        (tmp_path / "exec-facts.json").write_text(json.dumps(facts, indent=2))
    finally:
        host.close()


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
@pytest.mark.parametrize("death", ["sigkill", "panic"])
def test_killed_author_stays_interrupted_and_explicit_new_event_can_continue(tmp_path, mode, death):
    first_host = Host(tmp_path, 1, mode)
    try:
        first = first_host.turn().json()
        assert first.get("status") == "continuing", first
        if mode == "advisory":
            assert first["outcome"] == "message" and first["text"] == ANSWER and first["output_ref"]
        else:
            assert (first["outcome"], first["text"], first["output_ref"]) == ("deferred", "", "")
        ref = first["continuation_ref"]
        assert first_host.work(ref).status_code == 202
        assert len(first_host.inputs()) == 2
        record_path = tmp_path / "data" / "task_results" / f"{ref}.json"
        # The initial response is published before direct_owner_wait parks.
        def parked_record():
            record = json.loads(record_path.read_text())
            return record if record.get("owner_wait", {}).get("state") == "waiting" else None

        record = eventually(parked_record)
        assert record["execution_owner"]["kind"] == "presence"
        assert record["owner_wait"]["state"] == "waiting"
        continuation = record["presence_continuation"]
        from ouroboros.presence_continuation import continuation_status
        assert continuation["author_controller"]["pid"] == first_host.proc.pid
        assert continuation_status(record) == "running"  # another process sees the real owner alive
        if death == "sigkill":
            first_host.proc.kill()
            assert first_host.proc.wait(timeout=10) == -signal.SIGKILL
        else:
            try:
                first_host.client.post("/fixture/panic")
            except httpx.HTTPError:
                pass  # real hard exit may beat the acknowledgement
            assert first_host.proc.wait(timeout=15) == 73
            assert (tmp_path / "data" / "state" / "panic_stop.flag").exists()
    finally:
        first_host.close()

    second_host = Host(tmp_path, 2, mode)
    try:
        assert second_host.info["pid"] != first_host.info["pid"]
        assert second_host.info["reconciled"] == 0  # inline owners are not falsely pooled orphans
        assert second_host.turn().json() == first
        interrupted = second_host.work(ref)
        assert interrupted.status_code == 200, interrupted.text
        assert interrupted.json()["status"] == "interrupted"
        assert interrupted.json()["text"] == "" and interrupted.json()["output_ref"] == ""
        assert interrupted.json()["outputs"] == continuation["outputs"]
        assert len(second_host.inputs()) == 2  # no automatic author and no original event regeneration
        legacy = second_host.turn(version=0)
        assert legacy.status_code == 409 and legacy.json()["code"] == "presence_attempt_outcome_unknown"
        assert len(second_host.inputs()) == 2

        # This is a fresh, explicit conversational request, never a resumed dead stack/API.
        manual = second_host.turn("manual-continuation", MANUAL).json()
        assert manual["status"] == "completed" and manual["text"] == "Continued after the explicit request."
        assert manual["turn_ref"] != ref
        assert manual["output_ref"] and manual["output_ref"] != first["output_ref"]
        inputs = second_host.inputs()
        assert len(inputs) == 3 and inputs[-1]["pid"] == second_host.info["pid"]
        assert ref in str(inputs[-1]["messages"]) and "interrupted" in str(inputs[-1]["messages"])
        assert all("send_message" not in tool for row in inputs for tool in row["tools"])
        assert second_host.turn().json() == first
        assert second_host.work(ref).json() == interrupted.json()
        assert json.loads(record_path.read_text())["presence_continuation"] == continuation
        (tmp_path / "consumer-facts.json").write_text(json.dumps({
            "death": death, "mode": mode, "initial": first, "interrupted": interrupted.json(),
            "manual": manual, "original_model_calls": 2, "manual_model_calls": 1,
            "original_event_replay_unchanged": True, "no_direct_send_tool": True}, indent=2))
    finally:
        second_host.close()


def test_stop_ends_parked_full_agent_without_another_model_call(tmp_path):
    host = Host(tmp_path, 1, "blocking")
    try:
        first = host.turn().json()
        assert first.get("status") == "continuing", first
        assert host.client.post("/fixture/stop", json={"task_id": first["turn_ref"]}).status_code == 200
        final = eventually(lambda: (reply if (reply := host.work(first["turn_ref"])).status_code == 200 else None))
        assert final.json()["text"] == "" and not final.json()["outputs"]
        assert len(host.inputs()) == 2
        # The durable terminal is readable before handle_task's finally closes its
        # model receiver (the race the teardown-order test exposes): require closure.
        eventually(lambda: host.client.get("/fixture/live").json() == [])
        (tmp_path / "consumer-facts.json").write_text(json.dumps({"stop": final.json(), "model_calls": 2}))
    finally:
        host.close()


def second_park_with_late_child(host):
    """Drive two actual reviewer waits, with promotion only after the first yield."""
    response = host.turn()
    assert response.status_code == 200, response.text
    initial = response.json()
    assert initial["status"] == "continuing" and initial["work_ref"] == ""
    ref = initial["continuation_ref"]
    record_path = host.root / "data" / "task_results" / f"{ref}.json"

    def parked():
        record = json.loads(record_path.read_text())
        return record if record.get("owner_wait", {}).get("state") == "waiting" else None

    before = eventually(parked)
    first_continuation = before["presence_continuation"]
    assert first_continuation["initial"]["work_ref"] == ""
    assert not (host.root / "data" / "task_results" / f"{LATE_CHILD}.json").exists()
    assert host.work(ref).status_code == 202 and len(host.inputs()) == 2
    assert host.client.post("/fixture/release-review").status_code == 200

    def second_park():
        record = parked()
        if not record:
            return None
        continuation = record["presence_continuation"]
        state = host.client.get("/fixture/review-state").json()
        return record if (continuation["review_binding"] != first_continuation["review_binding"]
                          and state == {"reviews": 2, "model_calls": 4, "held": True}) else None

    after = eventually(second_park)
    assert after["presence_continuation"]["initial"] == first_continuation["initial"]
    assert after["presence_continuation"]["author_controller"] == first_continuation["author_controller"]
    assert host.client.get("/fixture/live").json() == [ref]
    assert host.turn().json() == initial and len(host.inputs()) == 4
    return initial, after, record_path


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
@pytest.mark.parametrize("ending", ["complete", "stop"])
def test_second_park_exposes_late_child_without_changing_initial_replay(tmp_path, mode, ending):
    host = Host(tmp_path, 1, mode, "late_child")
    try:
        initial, record, record_path = second_park_with_late_child(host)
        ref = initial["continuation_ref"]
        pending = host.work(ref)
        assert pending.status_code == 202, pending.text
        child_ref = pending.json().get("child_work_ref")
        assert child_ref == LATE_CHILD, pending.text
        assert record["presence_continuation"]["child_work_ref"] == child_ref
        child = host.work(child_ref)
        assert child.status_code == 202 and child.json()["status"] == "pending"
        assert child.json()["work_ref"] == child_ref and "continuation_ref" not in child.json()
        if ending == "stop":
            assert host.client.post("/fixture/stop", json={"task_id": ref}).status_code == 200
        else:
            assert host.client.post("/fixture/release-review").status_code == 200
        final = eventually(lambda: (reply if (reply := host.work(ref)).status_code == 200 else None))
        assert final.json()["child_work_ref"] == child_ref, final.text
        assert final.json()["status"] in {"completed", "cancelled"}
        assert len(host.inputs()) == (4 if ending == "stop" else 5)
        # Polling reads the durable terminal before handle_task's finally closes
        # its model receiver. Require actual closure too; terminal is not its ACK.
        eventually(lambda: host.client.get("/fixture/live").json() == [])
        assert host.turn().json() == initial
        assert json.loads(record_path.read_text())["presence_continuation"]["initial"] == record["presence_continuation"]["initial"]
        assert host.work(child_ref).json() == child.json()  # independent work survives the author's end
        (tmp_path / "late-child-facts.json").write_text(json.dumps({
            "mode": mode, "ending": ending, "initial": initial, "second_park": pending.json(),
            "final": final.json(), "child": child.json(), "model_calls": len(host.inputs()),
            "promotion_dispatch_scripted": True, "original_replay_unchanged": True}, indent=2))
    finally:
        host.close()
    cleanup = json.loads((tmp_path / "process-cleanup-1.json").read_text())
    assert cleanup["remaining_children"] == [] and cleanup["resource_tracker_joined"]
    if ending == "stop":
        # Releasing the pending reviewer after Stop emits a late notice through
        # the real manager-backed event bus. Both support processes must be
        # joined by the fixture, before the parent's group assertion runs.
        assert cleanup["event_bus_children"] and cleanup["resource_tracker_pid"], cleanup


def test_terminal_poll_precedes_author_teardown_but_cleanup_must_finish(tmp_path):
    """Deterministically expose the stop/advisory race; never infer cleanup from HTTP 200."""
    host = Host(tmp_path, 1, "advisory", "late_child_close_barrier")
    try:
        initial, record, _ = second_park_with_late_child(host)
        ref = initial["continuation_ref"]
        assert host.client.post("/fixture/stop", json={"task_id": ref}).status_code == 200
        marker = tmp_path / "before-author-close.json"
        blocked = eventually(lambda: json.loads(marker.read_text()) if marker.exists() else None)
        assert blocked["task_id"] == ref and blocked["closed"] is False
        assert blocked["status"] in {"completed", "cancelled"}
        final = host.work(ref)
        assert final.status_code == 200 and final.json()["status"] == blocked["status"]
        assert final.json()["child_work_ref"] == LATE_CHILD
        assert host.client.get("/fixture/live").json() == [ref]
        assert len(host.inputs()) == 4
        assert host.client.post("/fixture/release-close").status_code == 200
        eventually(lambda: host.client.get("/fixture/live").json() == [])
        assert host.work(ref).json() == final.json()
        assert host.turn().json() == initial and len(host.inputs()) == 4
        assert host.work(LATE_CHILD).status_code == 202
        (tmp_path / "teardown-order-facts.json").write_text(json.dumps({
            "blocked": blocked, "terminal": final.json(), "initial": record["presence_continuation"]["initial"],
            "receiver_closed_after_release": True, "model_calls": 4}, indent=2))
    finally:
        host.close()
    cleanup = json.loads((tmp_path / "process-cleanup-1.json").read_text())
    assert cleanup["remaining_children"] == [] and cleanup["resource_tracker_joined"]
    assert cleanup["event_bus_children"] and cleanup["resource_tracker_pid"], cleanup


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
def test_killed_second_park_author_preserves_late_child_for_tcp_consumer(tmp_path, mode):
    first_host = Host(tmp_path, 1, mode, "late_child")
    try:
        initial, record, record_path = second_park_with_late_child(first_host)
        ref = initial["continuation_ref"]
        # Physical process death drops the author stack and all in-memory refs.
        first_host.proc.kill()
        assert first_host.proc.wait(timeout=10) == -signal.SIGKILL
    finally:
        first_host.close()

    second_host = Host(tmp_path, 2, mode, "late_child")
    try:
        interrupted = second_host.work(ref)
        assert interrupted.status_code == 200 and interrupted.json()["status"] == "interrupted"
        child_ref = interrupted.json().get("child_work_ref")
        assert child_ref == LATE_CHILD, interrupted.text
        assert interrupted.json()["text"] == "" and interrupted.json()["output_ref"] == ""
        assert interrupted.json()["outputs"] == record["presence_continuation"]["outputs"]
        assert record["presence_continuation"]["child_work_ref"] == child_ref
        child = second_host.work(child_ref)
        assert child.status_code == 202 and child.json()["status"] == "pending"
        assert child.json()["work_ref"] == child_ref and "continuation_ref" not in child.json()
        for _ in range(2):
            assert second_host.turn().json() == initial
            assert second_host.work(ref).json() == interrupted.json()
            assert second_host.work(child_ref).json() == child.json()
        assert len(second_host.inputs()) == 4  # no original author restart or event regeneration
        assert second_host.client.get("/fixture/live").json() == []
        assert json.loads(record_path.read_text())["presence_continuation"] == record["presence_continuation"]
        (tmp_path / "late-child-facts.json").write_text(json.dumps({
            "mode": mode, "death": "SIGKILL", "initial": initial,
            "interrupted": interrupted.json(), "independent_child": child.json(),
            "original_model_calls": 4, "recycled_model_calls": 0,
            "promotion_dispatch_scripted": True, "original_replay_unchanged": True}, indent=2))
    finally:
        second_host.close()


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
def test_v1_tcp_disconnect_before_first_park_keeps_one_author(tmp_path, mode):
    from ouroboros.presence_runner import presence_turn_task_id

    host = Host(tmp_path, 1, mode, "disconnect")
    try:
        body = json.dumps(host.payload()).encode()
        endpoint = urlsplit(host.info["url"])
        # Raw TCP gives a physical disconnect with no HTTP retry by the client.
        with socket.create_connection((endpoint.hostname, endpoint.port), timeout=10) as connection:
            connection.sendall((
                f"POST /presence/turn HTTP/1.1\r\nHost: {endpoint.netloc}\r\n"
                f"X-Skill-Token: {TOKEN}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body)
            eventually(lambda: (tmp_path / "before-first-inference.json").exists())
            ref = presence_turn_task_id(host.info["binding"], "original")
            record_path = tmp_path / "data" / "task_results" / f"{ref}.json"
            before = json.loads(record_path.read_text())
            assert not before.get("presence_continuation")
            assert len(host.inputs()) == 1
            connection.shutdown(socket.SHUT_RDWR)
        # Only now may the first inference return and later reach the review park.
        assert host.client.post("/fixture/release-inference").status_code == 200
        eventually(lambda: json.loads(record_path.read_text()).get("presence_continuation"))
        initial = host.turn().json()
        assert initial["status"] == "continuing" and initial["turn_ref"] == ref
        assert host.work(ref).status_code == 202 and len(host.inputs()) == 2
        assert host.turn().json() == initial and len(host.inputs()) == 2
        assert host.client.post("/fixture/release-review").status_code == 200
        final = eventually(lambda: (reply if (reply := host.work(ref)).status_code == 200 else None)).json()
        assert final["status"] == "completed" and len(host.inputs()) == 3
        if mode == "advisory":
            assert initial["text"] == ANSWER and initial["output_ref"]
            assert final["text"] == "" and final["output_ref"] == ""
            assert final["outputs"] == [{"outcome": "message", "text": ANSWER,
                                          "output_ref": initial["output_ref"]}]
        else:
            assert initial["text"] == "" and initial["output_ref"] == ""
            assert final["text"] == ANSWER and final["output_ref"]
        assert host.turn().json() == initial and len(host.inputs()) == 3
        assert host.client.get("/fixture/live").json() == []
        (tmp_path / "consumer-facts.json").write_text(json.dumps({
            "mode": mode, "tcp_disconnected_before_first_inference_returned": True,
            "continuation_absent_at_disconnect": True, "initial": initial, "final": final,
            "model_calls": 3, "original_replay_unchanged": True}, indent=2))
    finally:
        host.close()


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
def test_proactive_caller_process_death_and_recycle_preserve_interrupted_work(tmp_path, mode):
    first_host = Host(tmp_path, 1, mode, "proactive")
    try:
        initial = first_host.client.post("/fixture/proactive").json()
        assert initial["status"] == "continuing" and initial["continuation_ref"] == initial["turn_ref"]
        ref = initial["turn_ref"]
        record_path = tmp_path / "data" / "task_results" / f"{ref}.json"
        record = json.loads(record_path.read_text())
        original_continuation = record["presence_continuation"]
        assert original_continuation["author_controller"]["pid"] == first_host.proc.pid
        assert record["metadata"]["presence"]["event"]["actor"]["kind"] == "proactive_initiation"
        assert len(first_host.inputs()) == 2 and first_host.work(ref).status_code == 202
        # The actual initiating tool and retained author run in this process;
        # killing it must not be confused with merely timing out its caller wait.
        first_host.proc.kill()
        assert first_host.proc.wait(timeout=10) == -signal.SIGKILL
    finally:
        first_host.close()
    second_host = Host(tmp_path, 2, mode, "proactive")
    try:
        interrupted = second_host.work(ref)
        assert interrupted.status_code == 200 and interrupted.json()["status"] == "interrupted"
        assert interrupted.json()["text"] == "" and interrupted.json()["output_ref"] == ""
        assert second_host.client.post("/fixture/proactive").json() == initial
        assert second_host.client.post("/fixture/proactive").json() == initial
        assert len(second_host.inputs()) == 2
        assert second_host.client.get("/fixture/live").json() == []
        assert json.loads(record_path.read_text())["presence_continuation"] == original_continuation
        (tmp_path / "consumer-facts.json").write_text(json.dumps({
            "mode": mode, "initial": initial, "interrupted": interrupted.json(),
            "caller_pid": first_host.proc.pid, "recycled_pid": second_host.proc.pid,
            "proactive_dedupe_replay_unchanged": True, "model_calls": 2,
            "author_shared_callers_process": True}, indent=2))
    finally:
        second_host.close()


@pytest.mark.parametrize("mode", ["blocking", "advisory"])
def test_tool_delivered_host_replay_and_poll_never_offer_duplicate_speech(tmp_path, mode):
    host = Host(tmp_path, 1, mode, "tool_delivered")
    try:
        initial = host.turn().json()
        assert initial["status"] == "continuing"
        ref = initial["turn_ref"]
        # An isolated transport fixture reports an already delivered tool effect.
        # No actual provider is contacted and the model has no direct send tool.
        receipt = {"schema_version": 1, "delivery_id": "fixture-tool-effect", "part_id": "0",
            "state": "delivered", "provider": "telegram", "account_id": "bot-1",
            "conversation_id": "room-1", "thread_id": "topic-1", "text": ANSWER,
            "format": "plain", "message": {"message_id": "fixture-tool-message"},
            "origin": {"kind": "tool", "task_id": ref, "source_event_id": "original"}}
        for _ in range(2):
            reply = host.client.post("/presence/delivery", json=receipt)
            assert reply.status_code == 200, reply.text
        assert initial["text"] == "" and initial["output_ref"] == ""
        assert host.work(ref).status_code == 202
        assert host.client.post("/fixture/release-review").status_code == 200
        final = eventually(lambda: (reply if (reply := host.work(ref)).status_code == 200 else None)).json()
        assert final["status"] == "completed" and final["outcome"] == "tool_delivered"
        assert final["text"] == "" and final["output_ref"] == "" and final["outputs"] == []
        assert host.turn().json() == initial and host.work(ref).json() == final
        rows = [json.loads(line) for line in (tmp_path / "data" / "logs" / "chat.jsonl").read_text().splitlines()]
        deliveries = [row for row in rows if row.get("type") == "presence_delivery"]
        assert len(deliveries) == 1 and deliveries[0]["text"] == ANSWER
        assert len(host.inputs()) == 3
        assert all("send_message" not in tool for row in host.inputs() for tool in row["tools"])
        (tmp_path / "consumer-facts.json").write_text(json.dumps({
            "mode": mode, "initial": initial, "final": final,
            "fixture_delivery_receipt_rows": len(deliveries), "model_calls": 3,
            "provider_transport_scripted": True}, indent=2))
    finally:
        host.close()
