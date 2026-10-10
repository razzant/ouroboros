"""#1554: an addressed action retires a positively ENDED local owner's capability.

Two kinds of positive evidence reach the existing retirement writers: a retained
witness (the producer knew the owner ended but its write failed) and a
platform-qualified absence of the recorded owner. Passive readers write
nothing; access denial, EPERM, unreadable or differently formatted identity,
generation, terminality and age prove nothing; independent effects and money
stay exactly as they were.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ouroboros import local_custody_repair as repair
from ouroboros import platform_layer
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _install_queue

pytestmark = pytest.mark.serial


def _dead_process_identity():
    """A real child that started, recorded its birth, and was reaped: positively gone."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    birth = platform_layer.process_start_time(child.pid)
    child.kill()
    child.wait(10)
    return child.pid, birth


def _claim(tmp_path, task_id, pid, birth, *, holder=None, attempt=1):
    holder = holder or task_id
    row = load_task_result(tmp_path, holder) or {}
    claims = dict(row.get("launch_handoffs") or {})
    claims[f"op-{task_id}"] = {"tool": "delegate_wait", "task_id": task_id, "root_task_id": "root",
                               "state": "claimed", "claimed_at": "2026-10-06T11:32:45+00:00",
                               "local_owner": {"pid": pid, "process_birth": birth, "task_attempt": attempt}}
    write_task_result(tmp_path, holder, row.get("status") or "cancelled", root_task_id="root", launch_handoffs=claims)


def _witnesses(tmp_path):
    path = tmp_path / "state" / "obligations" / f"{repair.WITNESS_SET}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def test_a_positively_dead_owner_is_retired_only_by_the_addressed_action(tmp_path, monkeypatch):
    from supervisor.continuation_admission import action_writers, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    pid, birth = _dead_process_identity()
    write_task_result(tmp_path, "helper", "cancelled", root_task_id="root")
    _claim(tmp_path, "helper", pid, birth)
    before = (tmp_path / "task_results" / "helper.json").read_bytes()
    passive = conflicting_writers(queue, "root")
    assert [b["kind"] for b in passive] == ["tool_handoff"], "the passive census still holds it"
    assert (tmp_path / "task_results" / "helper.json").read_bytes() == before, "passive reads write nothing"

    assert [b for b in action_writers(queue, "root") if b["kind"] == "tool_handoff"] == []
    retired = load_task_result(tmp_path, "helper")["retired_tool_invocations"]["op-helper"]
    assert retired["state"] == "owner_dead" and retired["effect_outcome"] == "unknown"
    assert retired["replay_authorized"] is False


@pytest.mark.parametrize("case", ["live_self", "unreadable_birth", "other_format_birth", "no_identity"])
def test_a_live_unknown_or_unattributable_owner_stays_held(tmp_path, monkeypatch, case):
    from supervisor.continuation_admission import action_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    write_task_result(tmp_path, "helper", "cancelled", root_task_id="root")
    own = platform_layer.process_start_time(os.getpid())
    # Exercise a DIFFERENT identity kind on Windows as well as POSIX; another
    # valid Windows FILETIME would correctly prove that the recorded owner ended.
    foreign_birth = "812" if own.startswith("win-filetime:") else "win-filetime:1234567"
    pid, birth = {"live_self": (os.getpid(), own), "unreadable_birth": (os.getpid(), ""),
                  "other_format_birth": (os.getpid(), foreign_birth),
                  "no_identity": (0, own)}[case]
    _claim(tmp_path, "helper", pid, birth)
    assert repair.owner_ended(pid, birth) is False
    assert [b["kind"] for b in action_writers(queue, "root")] == ["tool_handoff"]


_BOOT = "0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("live,recorded,ended", [
    (f"812.{_BOOT}", f"811.{_BOOT}", True),            # boot-qualified ticks of another process
    (f"812.{_BOOT}", "811." + "f" * 32, True),          # another boot: the recorded owner died with it
    ("win-filetime:1300", "win-filetime:1200", True),  # UTC FILETIME
    ("812.", "811", True),                               # boot-relative ticks, both spellings
    ("812.", "812", False),                              # one identity, two spellings
    (f"812.{_BOOT}", "811", False),                      # different kinds prove nothing
    ("Thu Oct  8 17:18:45 2026", "Thu Oct  8 14:18:45 2026", False),  # wall clock: never proof
    ("Thu Oct  8 17:18:45 2026", "812", False),
])
def test_only_a_validated_birth_identity_can_prove_a_different_process(monkeypatch, live, recorded, ended):
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: live)
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda _pid: False)
    assert repair.owner_ended(os.getpid(), recorded) is ended
    assert repair.owner_ended(4242, recorded) is ended


@pytest.mark.skipif(sys.platform == "win32", reason="ps lstart is the POSIX wall-clock fallback")
def test_two_timezone_readings_of_this_living_process_are_not_its_death(monkeypatch):
    """Parent repro 2026-10-08: ``ps -o lstart=`` of ONE living pid under TZ=UTC and
    TZ=Europe/Moscow gives two different strings; neither proves the other dead."""
    def lstart(tz):
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(os.getpid())], capture_output=True,
                             text=True, timeout=5, env={**os.environ, "TZ": tz})
        return out.stdout.strip()

    utc, moscow = lstart("UTC"), lstart("Europe/Moscow")
    if not utc or utc == moscow:
        pytest.skip("this ps does not render a timezone-dependent lstart")
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: moscow)
    assert repair.owner_ended(os.getpid(), utc) is False


@pytest.mark.parametrize("state,gone", [("no_such_pid", True), ("exited", True), ("denied", False),
                                        ("error", False), ("unreadable", False), ("running", False)])
def test_windows_absence_is_only_no_such_pid_or_a_read_exit_code(monkeypatch, state, gone):
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "_windows_process_state", lambda _pid: state)
    assert platform_layer.pid_provably_gone(4242) is gone
    assert platform_layer.pid_is_alive(4242) is (state in {"running", "denied", "unreadable"})


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-zero semantics")
def test_posix_eperm_is_not_absence(monkeypatch):
    def denied(_pid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(platform_layer.os, "kill", denied)
    assert platform_layer.pid_provably_gone(4242) is False
    assert platform_layer.pid_is_alive(4242) is True


def test_a_closed_receiver_whose_retirement_write_failed_keeps_a_witness_the_next_action_discharges(
        tmp_path, monkeypatch):
    from ouroboros import model_wait, usage_accounting as ua
    from supervisor.continuation_admission import action_writers, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                    event_queue=None, worker_slot_held=True)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
        with model_wait.operation_wait_scope(owner):
            with pytest.raises(RuntimeError):
                ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="t", reservation_usd=.2),
                                            lambda: (_ for _ in ()).throw(RuntimeError("response unknown")))
    money = ua.read_usage_records(tmp_path, final_only=True)
    real_update = model_wait.update_json_locked
    monkeypatch.setattr(model_wait, "update_json_locked",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")))
    owner.close()  # a closed receiver on a live host; its retirement write fails
    witness = next(iter(_witnesses(tmp_path).values()))
    assert witness["model_consumers"] == {owner.answer_consumer_id: 1}
    assert witness["producer"] == "model_wait" and witness["task_id"] == "root"
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    assert any(b["kind"] == "model_handoff" for b in conflicting_writers(queue, "root"))
    assert _witnesses(tmp_path), "a passive census discharges nothing"

    monkeypatch.setattr(model_wait, "update_json_locked", real_update)
    assert not any(b["kind"] == "model_handoff" for b in action_writers(queue, "root"))
    assert _witnesses(tmp_path) == {}, "discharged only by the writer that landed"
    assert owner.answer_consumer_id in load_task_result(tmp_path, "root")["retired_model_consumers"]
    assert ua.read_usage_records(tmp_path, final_only=True) == money, "unknown money stays unknown"


def test_a_witness_that_cannot_be_written_either_leaves_an_honest_hold(tmp_path, monkeypatch):
    from ouroboros import obligations

    monkeypatch.setattr(obligations, "add", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("lock")))
    assert repair.retain_witness(tmp_path, "t", reason="x", model_consumers={"c": 1}) is False


def test_a_member_claim_parked_on_its_roots_row_is_retired_where_it_lives(tmp_path, monkeypatch):
    from supervisor.continuation_admission import action_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    pid, birth = _dead_process_identity()
    _claim(tmp_path, "unpublished-child", pid, birth, holder="root")
    assert not [b for b in action_writers(queue, "root") if b["kind"] == "tool_handoff"]
    assert "op-unpublished-child" in load_task_result(tmp_path, "root")["retired_tool_invocations"]


def test_independent_effects_survive_the_repair(tmp_path, monkeypatch):
    from supervisor.continuation_admission import action_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    pid, birth = _dead_process_identity()
    write_task_result(tmp_path, "helper", "cancelled", root_task_id="root",
                      merge_receipts=[{"receipt_id": "merge-1", "launch_operation_ids": ["op-helper"],
                                       "state": "submitted"}])
    _claim(tmp_path, "helper", pid, birth)
    kinds = [b["kind"] for b in action_writers(queue, "root")]
    assert "tool_handoff" not in kinds and "merge_operation" in kinds, "a merge receipt keeps its own custody"


def test_confirmed_worker_death_whose_write_failed_retains_its_death_witness(tmp_path, monkeypatch):
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    import ouroboros.utils as utils

    real = utils.update_json_locked
    monkeypatch.setattr(utils, "update_json_locked", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")))
    assert repair.retire_ended_owner(tmp_path, "root", "root", pid=99999, birth="b", attempt=1,
                                     producer="worker_death") is False
    monkeypatch.setattr(utils, "update_json_locked", real)
    witness = next(iter(_witnesses(tmp_path).values()))
    assert witness["local_owner"] == {"pid": 99999, "process_birth": "b", "task_attempt": 1}
    assert witness["producer"] == "worker_death"


def test_addressed_retry_discharges_a_root_held_child_witness_after_retirement_write_failure(tmp_path, monkeypatch):
    from ouroboros import tool_custody, usage_accounting as ua
    from supervisor.continuation_admission import action_writers, conflicting_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    write_task_result(tmp_path, "root", "failed", root_task_id="root",
                      merge_receipts=[{"receipt_id": "merge-child", "launch_operation_ids": ["op-child"], "state": "submitted"}])
    pid, birth = _dead_process_identity()
    _claim(tmp_path, "child", pid, birth, holder="root")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="child", root_task_id="root")):
        with pytest.raises(RuntimeError, match="unknown cost"):
            ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.2),
                                        lambda: (_ for _ in ()).throw(RuntimeError("unknown cost")))
    money = ua.read_usage_records(tmp_path, final_only=True)
    retire = tool_custody.retire_tool_invocations
    monkeypatch.setattr(tool_custody, "retire_tool_invocations",
                        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("retirement disk failure")))
    assert repair.retire_ended_owner(tmp_path, "child", "root", pid=pid, birth=birth, attempt=1,
                                     producer="confirmed_worker_death", holder="root") is False
    assert _witnesses(tmp_path)
    monkeypatch.setattr(tool_custody, "retire_tool_invocations", retire)
    # The next action has only the positive retained witness; a new process
    # probe is unavailable. The unpublished child is absent from tree members.
    monkeypatch.setattr(repair, "owner_ended", lambda *_a: False)
    assert "tool_handoff" in [b["kind"] for b in conflicting_writers(queue, "root")]
    assert _witnesses(tmp_path), "passive census never discharges the witness"
    kinds = [b["kind"] for b in action_writers(queue, "root")]
    assert "tool_handoff" not in kinds and "merge_operation" in kinds
    assert _witnesses(tmp_path) == {}
    retired = load_task_result(tmp_path, "root")["retired_tool_invocations"]["op-child"]
    assert retired["effect_outcome"] == "unknown" and retired["replay_authorized"] is False
    assert load_task_result(tmp_path, "child") is None, "repair does not invent the unpublished child's row"
    assert ua.read_usage_records(tmp_path, final_only=True) == money


def test_holder_witness_from_a_different_tree_is_not_discharged(tmp_path, monkeypatch):
    from supervisor.continuation_admission import action_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root")
    repair.retain_witness(tmp_path, "root", reason="write_failed", producer="confirmed_worker_death",
                          local_owner={"pid": 4242, "process_birth": "old-birth", "task_attempt": 1},
                          root_task_id="foreign-tree", holder="root")
    before = _witnesses(tmp_path)
    action_writers(queue, "root")
    assert _witnesses(tmp_path) == before, "matching the holder name is insufficient without its tree"


def test_a_malformed_neighbor_claim_does_not_erase_or_prevent_exact_retirement(tmp_path, monkeypatch):
    from supervisor.continuation_admission import action_writers

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "failed", root_task_id="root", launch_handoffs={"malformed": "unreadable"})
    pid, birth = _dead_process_identity()
    _claim(tmp_path, "root", pid, birth)
    blockers = action_writers(queue, "root")
    row = load_task_result(tmp_path, "root")
    assert "op-root" in row.get("retired_tool_invocations", {})
    assert row["launch_handoffs"] == {"malformed": "unreadable"}
    assert "tree_census_unreadable" in [b["kind"] for b in blockers], "the malformed independent claim stays held"
