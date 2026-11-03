"""Which process a review controller record names (#2382): ``review_operation.controller_state``.

A record carries the ps-spelled ``birth`` and, from this release, ``kernel_birth``
(``process_identity``). Kernel births of one validated kind decide whatever zone or locale
spelled ``ps -o lstart=``; malformed, missing or cross-kind ones prove nothing and leave the
spelling to decide. Each row also states what the same record without its kernel birth gets
(``legacy``): an older release's record.

Without comparable births, an equal spelling is alive; a different
one proves the end only as one validated kind (Linux ticks, Windows FILETIME), never as ps text
another zone or locale spells differently, which is unknown. Every consumer keeps an unknown
controller's custody waiting. The accepted price: an old record on a reused pid stays unknown,
its pointer kept, until that process ends; no timer or manager retires it. Our own pid keeps
its session rules, and worker readiness its own comparison.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros import review_operation
from ouroboros.review_operation import controller_state

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PID = 2_000_000_001  # never this process: every read about it below is the fixture's
_KERNEL, _REUSED = "darwin-start:1793143271.251840", "darwin-start:1793143271.251841"
_HERE, _THERE = "Wed Oct 28 14:38:55 2026", "Wed Oct 28 03:53:55 2026"  # one start, two ps zones
_LINUX, _LINUX_OTHER = "812." + "0123456789abcdef" * 2, "811." + "0123456789abcdef" * 2
_FILETIME, _FILETIME_OTHER = "win-filetime:134058912345678901", "win-filetime:134058912345678902"


@pytest.fixture
def live(monkeypatch):
    """``_PID`` is alive; what this reader sees of its ps spelling and kernel birth is the test's."""
    from ouroboros import platform_layer, process_containment, process_identity

    seen = {"birth": _HERE, "kernel": _KERNEL}
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: pid == _PID)
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: seen["birth"])
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda _pid: seen["kernel"])
    return seen


@pytest.mark.parametrize("record,seen,now,legacy", [
    ({"birth": _THERE, "kernel_birth": _KERNEL}, (_HERE, _KERNEL), "alive", "unknown"),
    ({"birth": _HERE, "kernel_birth": _KERNEL}, (_HERE, _KERNEL), "alive", "alive"),
    ({"birth": _HERE, "kernel_birth": _REUSED}, (_HERE, _KERNEL), "dead", "alive"),
    ({"birth": "", "kernel_birth": _KERNEL}, (_HERE, _KERNEL), "alive", "unknown"),
    ({"birth": _THERE, "kernel_birth": _KERNEL}, (_HERE, None), "unknown", "unknown"),
    ({"birth": _HERE, "kernel_birth": _KERNEL}, (_HERE, None), "alive", "alive"),
    ({"birth": _HERE, "kernel_birth": _KERNEL}, ("", None), "unknown", "unknown"),
    ({"birth": _THERE, "kernel_birth": "darwin-start:1793143271"}, (_HERE, _KERNEL), "unknown", "unknown"),
    ({"birth": _HERE, "kernel_birth": _THERE}, (_HERE, _KERNEL), "alive", "alive"),
    ({"birth": _THERE, "kernel_birth": _FILETIME}, (_HERE, _KERNEL), "unknown", "unknown"),
    ({"birth": _THERE}, (_HERE, _KERNEL), "unknown", "unknown"),
    ({"birth": _HERE}, (_HERE, _KERNEL), "alive", "alive"),
    ({"birth": ""}, (_HERE, _KERNEL), "unknown", "unknown"),
    ({"birth": _LINUX, "kernel_birth": _LINUX}, (_LINUX, _LINUX), "alive", "alive"),
    ({"birth": _LINUX, "kernel_birth": _LINUX}, (_LINUX_OTHER, _LINUX_OTHER), "dead", "dead"),
    ({"birth": _FILETIME, "kernel_birth": _FILETIME}, (_FILETIME, _FILETIME), "alive", "alive"),
    ({"birth": _FILETIME, "kernel_birth": _FILETIME}, (_FILETIME_OTHER, _FILETIME_OTHER), "dead", "dead"),
    ({"birth": "812"}, ("811", None), "dead", "dead"),
], ids=["new_other_zone", "new_same_zone", "reused_same_second", "kernel_without_spelling",
        "live_kernel_unread_other_zone", "live_kernel_unread_same_zone", "live_unreadable",
        "malformed_other_zone", "ps_text_as_kernel", "two_kinds", "old_other_zone", "old_same_zone",
        "old_without_birth", "linux_same", "linux_reused", "windows_same", "windows_reused", "old_ticks_reused"])
def test_another_process_is_judged_by_comparable_births_and_spelling_alone_proves_no_end(
        live, record, seen, now, legacy):
    live["birth"], live["kernel"] = seen
    identity = {"pid": _PID, "session": "this-server", **record}
    older = {key: value for key, value in identity.items() if key != "kernel_birth"}

    assert controller_state(identity) == now
    assert controller_state(older) == legacy


def test_no_live_pid_is_dead_whatever_births_it_carries(live, monkeypatch):
    from ouroboros import process_containment

    assert controller_state({"pid": _PID + 1, "birth": _HERE, "kernel_birth": _KERNEL, "session": "s"}) == "dead"
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: True)
    assert controller_state({"pid": _PID, "birth": _HERE, "kernel_birth": _KERNEL, "session": "s"}) == "dead"


@pytest.mark.parametrize("change,now,legacy", [
    ({}, "local", "local"),
    ({"birth": _THERE}, "local", "dead"),
    ({"kernel_birth": _REUSED}, "dead", "local"),
    ({"session": "before-exec"}, "dead", "dead"),
    ({"birth": _THERE, "session": "before-exec"}, "dead", "dead"),
    ({"session": ""}, "unknown", "unknown"),
    ({"birth": "", "kernel_birth": ""}, "unknown", "unknown"),
], ids=["own", "own_other_zone", "earlier_holder_of_this_pid", "same_pid_and_birth_after_exec",
        "after_exec_other_zone", "no_session", "no_birth"])
def test_this_process_judges_its_own_records_and_exec_by_session(monkeypatch, change, now, legacy):
    """One pid and kernel birth survive exec: only the custody session tells the pre-exec controller.
    This process mints its identity once, so another spelling of its own pid names another image."""
    monkeypatch.setattr(review_operation, "_SELF", {os.getpid(): {"pid": os.getpid(), "birth": _HERE,
                                                                  "kernel_birth": _KERNEL}})
    identity = {**review_operation.controller_identity(), **change}
    older = {key: value for key, value in identity.items() if key != "kernel_birth"}

    assert controller_state(identity) == now
    assert controller_state(older) == legacy


_MINT = ("import json, sys; from ouroboros.review_operation import controller_identity; "
         "print(json.dumps(controller_identity()), flush=True); sys.stdin.read()")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS ps spells lstart in the reader's zone")
def test_a_live_controller_minted_in_another_zone_keeps_its_pointer(tmp_path, monkeypatch):
    """A real controller identity minted by an owned child in Pacific/Chatham, read here in UTC
    (TZ of these two processes only; no OS setting changes). Its child exits on stdin EOF."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_identity import process_kernel_birth
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import load_task_result, write_task_result

    path = os.pathsep.join(part for part in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if part)
    child = subprocess.Popen([sys.executable, "-c", _MINT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, cwd=REPO_ROOT,
                             env={**os.environ, "PYTHONPATH": path, "TZ": "Pacific/Chatham"})
    try:
        identity = json.loads(child.stdout.readline())
        monkeypatch.setenv("TZ", "UTC")
        monkeypatch.setattr(review_operation, "_SELF", {})  # no identity of this process cached in UTC
        assert identity["pid"] == child.pid and identity["kernel_birth"] == process_kernel_birth(child.pid)
        assert identity["birth"] != process_start_time(child.pid), "precondition: two zones spell one birth"
        legacy = {key: value for key, value in identity.items() if key != "kernel_birth"}
        assert (controller_state(identity), controller_state(legacy)) == ("alive", "unknown")

        prepare_startup_state(tmp_path)
        pointers = {owner: {"state": "dispatched", "surface": "plan_review", "retry_key": f"plan_review:{owner}",
                            "controller": controller} for owner, controller in (("op-new", identity),
                                                                                 ("op-old", legacy))}
        write_task_result(tmp_path, "t-controller", "completed", result="done", review_operations=pointers)
        report = review_operation.retire_dead_review_pointers(tmp_path)
        assert report["retired"] == [], report  # neither live controller's pointer is retired
        assert set(load_task_result(tmp_path, "t-controller")["review_operations"]) == {"op-new", "op-old"}
        assert review_operation.task_has_live_review_operation(tmp_path, "t-controller")
    finally:
        for stream in (child.stdin, child.stdout):
            stream.close()
        child.kill()
        child.wait(10)


_OTHER, _GONE = _PID + 2, _PID + 3
_ROWS = {  # one ended task's open pointers: newer and older records, live and ended controllers
    "op-new": {"pid": _PID, "birth": _THERE, "kernel_birth": _KERNEL},  # alive: the kernel births match
    "op-old-zone": {"pid": _PID, "birth": _THERE},  # unknown: ps text another zone spelled
    "op-new-reused": {"pid": _PID, "birth": _HERE, "kernel_birth": _REUSED},  # dead: kernel births differ
    "op-old-reused": {"pid": _OTHER, "birth": _LINUX},  # dead: comparable older births differ
    "op-old-gone": {"pid": _GONE, "birth": _THERE},  # dead: no live pid
}


@pytest.fixture
def processes(monkeypatch):
    """Live pids and what this reader sees of each: ``pid -> (ps spelling, kernel birth)``."""
    from ouroboros import platform_layer, process_containment, process_identity

    seen = {_PID: (_HERE, _KERNEL), _OTHER: (_LINUX_OTHER, _LINUX_OTHER)}
    own = {"pid": os.getpid(), "birth": _HERE, "kernel_birth": ""}  # this process, never a row below
    monkeypatch.setattr(review_operation, "_SELF", {os.getpid(): own})
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: pid in seen)
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda pid: seen[pid][0])
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda pid: seen[pid][1])
    return seen


def test_mixed_pointers_retire_only_on_proof_and_an_unknown_one_waits_for_its_process(processes, tmp_path):
    """The maintenance pass retires proven ends only; the older row stays until its pid's process exits."""
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import load_task_result, write_task_result

    prepare_startup_state(tmp_path)
    pointers = {owner: {"state": "dispatched", "surface": "plan_review", "retry_key": f"plan_review:{owner}",
                        "controller": {**controller, "session": "earlier-server"}}
                for owner, controller in _ROWS.items()}
    write_task_result(tmp_path, "t-mixed", "completed", result="done", review_operations=pointers)
    assert {owner: controller_state(row["controller"]) for owner, row in pointers.items()} == {
        "op-new": "alive", "op-old-zone": "unknown", "op-new-reused": "dead", "op-old-reused": "dead",
        "op-old-gone": "dead"}
    report = review_operation.retire_dead_review_pointers(tmp_path)
    assert sorted(row["owner_id"] for row in report["retired"]) == ["op-new-reused", "op-old-gone", "op-old-reused"]
    assert set(load_task_result(tmp_path, "t-mixed")["review_operations"]) == {"op-new", "op-old-zone"}
    assert review_operation.task_has_live_review_operation(tmp_path, "t-mixed")
    del processes[_PID]  # the accepted price ends with that process: the next pass has its proof
    report = review_operation.retire_dead_review_pointers(tmp_path)
    assert sorted(row["owner_id"] for row in report["retired"]) == ["op-new", "op-old-zone"]
    assert not review_operation.task_has_live_review_operation(tmp_path, "t-mixed")


def test_an_unknown_controller_ends_no_presence_attempt_and_settles_no_left_wave(processes, tmp_path, monkeypatch):
    from ouroboros import review_pause
    from ouroboros.presence_recovery import attempt_ended

    unknown, gone = _ROWS["op-old-zone"], _ROWS["op-old-gone"]

    def attempt(controller):
        return {"task_id": "presence-1", "status": "running", "execution_owner": {"controller": controller}}

    assert not attempt_ended(attempt(unknown), tmp_path) and attempt_ended(attempt(gone), tmp_path)
    monkeypatch.setattr(review_pause, "_left_roster_settled", lambda _record: False)  # a seat has no outcome yet
    monkeypatch.setattr(review_pause, "ended_left_review", lambda _record: "ended")
    assert review_pause.settle_left_review(tmp_path, {"left_review": {}, "controller": unknown}) == "waiting"
    assert review_pause.settle_left_review(tmp_path, {"left_review": {}, "controller": gone}) == "ended"


def test_a_decision_reaches_another_process_only_through_a_controller_proven_alive(processes, monkeypatch):
    """``review_operation_controller`` refuses an unknown controller (``task_not_live`` upstream): it is
    neither addressed as live nor declared dead."""
    from ouroboros import process_custody, task_results

    def stored(controller):
        block = {"owner_id": "op-1", "controller": controller, "retry_key": "rk", "slot_id": "s1", "chat_id": 7}
        return {"model_waits": {"w1": {"state": "waiting", "review_operation": block, "model_wait_owner_id": "op-1",
                                       "task_attempt": 1}},
                "review_operations": {"op-1": {"state": "dispatched", "controller": controller, "retry_key": "rk",
                                               "operations": {"s1": {}}, "task_attempt": 1}}}

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "this-server")
    monkeypatch.setattr(review_operation, "operation_stop_state", lambda *_a: "")
    for owner, routed in (("op-new", True), ("op-old-zone", False)):
        identity = {**_ROWS[owner], "session": "this-server"}
        monkeypatch.setattr(task_results, "load_task_result", lambda *_a, identity=identity, **_k: stored(identity))
        assert (review_operation.review_operation_controller("root", "t-1", "w1") is not None) is routed, owner
