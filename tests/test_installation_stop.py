"""Installation inputs are received once; tasks keep their own current stops."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery as dr
from ouroboros.installation_stop import installation_panic, panic_input, PANIC_RECEIPT
from ouroboros.work_stop import current_stop
from tests._budget_pause_exact_helpers import _install_queue
from tests._source_scan import parse


def test_received_panic_survives_failed_unlink_and_a_new_panic_is_new(tmp_path, monkeypatch):
    from supervisor import worker_chat_lane
    from ouroboros.launcher_bootstrap import automatic_launch_allowed
    from tests.test_model_sleep import _cold_park
    from tests.test_state_authority import _prior

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=3600)
    assert q.persist_queue_snapshot(reason="before_panic")
    flag = tmp_path / "state/panic_stop.flag"
    flag.write_text("panic", encoding="utf-8")
    first = panic_input(tmp_path)
    assert installation_panic(tmp_path)
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    assert current_stop(workers.PENDING[0])["release"] == "owner"
    real_unlink = Path.unlink

    def failed_unlink(path, *args, **kwargs):
        if path == flag:
            raise PermissionError("injected unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failed_unlink)
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()[PANIC_RECEIPT] == first and flag.exists()
    assert not installation_panic(tmp_path)
    assert automatic_launch_allowed("automatic", tmp_path, SimpleNamespace(info=lambda *a: None))
    assert q.resume_budget_paused_task("sleeper")["ok"]
    state.update_state(lambda live: live.update(bg_consciousness_enabled=True))
    assert q.persist_queue_snapshot(reason="resumed")
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()["bg_consciousness_enabled"] is True
    assert current_stop(workers.PENDING[0]).get("release") != "owner"
    monkeypatch.setattr(Path, "unlink", real_unlink)
    flag.unlink()
    flag.write_text("panic", encoding="utf-8")
    assert panic_input(tmp_path) != first and installation_panic(tmp_path)
    assert q.persist_queue_snapshot(reason="before_new_panic")
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    assert current_stop(workers.PENDING[0])["release"] == "owner"
    worker_chat_lane.auto_resume_after_restart()
    assert not flag.exists() and state.load_state()["bg_consciousness_enabled"] is False


@pytest.mark.parametrize("ack", [False, True])
def test_boot_consumes_only_a_prior_generations_door_independent_of_ack(tmp_path, monkeypatch, ack):
    from supervisor import worker_chat_lane

    q, _state, _workers = _install_queue(tmp_path, monkeypatch)
    row = dr.prepare_owner_restart(tmp_path)
    assert not dr.pending_owner_restart(tmp_path)
    row.update(generation="previous", status="normal_exit_acknowledged" if ack else "prepared")
    dr._write_restart_transaction(tmp_path, row)
    assert dr.pending_owner_restart(tmp_path)["transaction_id"] == row["transaction_id"]
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kw: False)
    worker_chat_lane.auto_resume_after_restart()
    assert not dr.active_restart_transaction(tmp_path)["door_applied_at"]
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **kw: True)
    worker_chat_lane.auto_resume_after_restart()
    assert dr.active_restart_transaction(tmp_path)["door_applied_at"]
    assert not dr.pending_owner_restart(tmp_path)


def test_worker_reads_only_its_generation_and_cannot_ensure_a_daemon(tmp_path, monkeypatch):
    import multiprocessing
    from ouroboros import claudexor_daemon, config, budget_pause, model_wait
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_operation import operation_stop_state

    row = dr.prepare_owner_restart(tmp_path)
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: SimpleNamespace(pid=row["supervisor_pid"]))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda: pytest.fail("must not ensure a daemon"))
    assert dr.owner_restart_in_progress(tmp_path)
    with pytest.raises(ClaudexorUnavailable, match="restarting"):
        claudexor_daemon.ensure_owned_gateway()
    assert operation_stop_state(tmp_path, "task") == "owner_restart"
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
    assert budget_pause._hold_control_reason(SimpleNamespace(drive_root=tmp_path, task_id="task")) == ""
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: SimpleNamespace(pid=row["supervisor_pid"] + 1))
    assert not dr.owner_restart_in_progress(tmp_path)


def _worker_restart_probe(root, output):
    from unittest.mock import patch
    from ouroboros import claudexor_daemon, config
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    config.DATA_DIR = root
    with patch.object(claudexor_daemon, "get_owned_daemon", side_effect=AssertionError("daemon reached")):
        try:
            claudexor_daemon.ensure_owned_gateway()
        except ClaudexorUnavailable as exc:
            output.put((dr.owner_restart_in_progress(root), exc.code))


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_a_real_worker_recognizes_its_parent_restart_before_daemon_access(tmp_path, method):
    import multiprocessing

    if method not in multiprocessing.get_all_start_methods():
        pytest.skip("start method is unavailable on this host")
    dr.prepare_owner_restart(tmp_path)
    ctx = multiprocessing.get_context(method)
    output = ctx.Queue()
    worker = ctx.Process(target=_worker_restart_probe, args=(tmp_path, output))
    worker.start()
    try:
        assert output.get(timeout=20) == (True, "owner_restart")
        worker.join(timeout=20)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=20)
        output.close()
        output.join_thread()


@pytest.mark.parametrize("kind", ["flag", "byte", "panic_and_restart"])
def test_old_build_inputs_become_a_boot_door_never_panic(tmp_path, kind):
    from ouroboros.delegate_recovery import boot_door

    state = tmp_path / "state"
    state.mkdir()
    if kind != "byte":
        (state / "owner_restart_no_resume.flag").write_text("owner_restart", encoding="utf-8")
    if kind != "flag":
        (state / "panic_stop.flag").write_text(
            "panic" if kind == "panic_and_restart" else "owner_restart_no_resume", encoding="utf-8")
    assert boot_door(tmp_path) == ("panic" if kind == "panic_and_restart" else "owner_restart")
    assert installation_panic(tmp_path) is (kind == "panic_and_restart")
    assert dr.pending_owner_restart(tmp_path)["door"] == "owner_restart"
    assert not (state / "owner_restart_no_resume.flag").exists()


def test_failed_skeleton_publication_restores_the_previous_pointer(tmp_path, monkeypatch):
    before = dr.prepare_owner_restart(tmp_path)
    pointer = dr._active_restart_transaction_path(tmp_path)
    raw = pointer.read_bytes()
    write = dr.atomic_write_json

    def fail(path, value):
        if path == pointer:
            raise OSError("injected publication failure")
        return write(path, value)

    monkeypatch.setattr(dr, "atomic_write_json", fail)
    with pytest.raises(OSError):
        dr.prepare_owner_restart(tmp_path)
    assert pointer.read_bytes() == raw
    assert dr.active_restart_transaction(tmp_path)["transaction_id"] == before["transaction_id"]
    assert len(list(pointer.parent.glob("*.json"))) == 2


@pytest.mark.parametrize("status", ["prepared", "normal_exit_acknowledged"])
def test_legacy_flag_cannot_rename_an_unconsumed_update_transaction(tmp_path, status):
    from supervisor.restart_retention import prepare_restart_returns

    prepare_restart_returns(tmp_path, {}, [], transaction_id="update")
    transaction = dr.active_restart_transaction(tmp_path)
    transaction.update(status=status, return_ids=["saved"], task_ids=["saved"])
    dr._write_restart_transaction(tmp_path, transaction)
    path = dr._restart_transaction_path(tmp_path, "update")
    before = path.read_bytes()
    (tmp_path / "state/owner_restart_no_resume.flag").write_text("owner_restart", encoding="utf-8")
    assert dr.boot_door(tmp_path) == ""
    assert path.read_bytes() == before
    assert not (tmp_path / "state/owner_restart_no_resume.flag").exists()


def test_owner_update_keeps_sleep_readiness_and_returns_saved_work_after_ack(tmp_path, monkeypatch):
    import os
    import threading
    from ouroboros import server_process, server_restart
    from ouroboros.task_results import write_task_result
    from supervisor.restart_retention import prepare_restart_returns
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _cold_park, _mail
    from tests.test_restart_saved_work import _working_run

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    write_task_result(tmp_path, "peer", "running")
    _cold_park(tmp_path, monkeypatch, workers, senders=["peer"])
    _working_run(tmp_path, workers, "working")
    prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="update")
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
    owner = threading.Event()
    monkeypatch.setattr(server_restart, "_owner_restart_requested", owner)
    monkeypatch.setattr(server_process, "_owner_restart_requested", owner)
    monkeypatch.setattr(server_process, "_restart_requested", threading.Event())
    server_process._request_restart_exit(owner=True)
    assert owner.is_set()
    in_progress = dr.owner_restart_in_progress(tmp_path)
    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Managed update.",
                                stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True,
                                archive_service_logs=False, reconcile_delegate_custody=False)
    assert dr.acknowledge_observed_restart_exit(tmp_path, supervisor_pid=os.getpid(), exit_code=42)
    owner.clear()  # the successor has fresh process latches; only durable stops survive
    workers.PENDING.clear()
    workers.RUNNING.clear()
    assert q.restore_pending_from_snapshot() == 2
    restored = {t["id"]: t for t in workers.PENDING}
    assert not current_stop(restored["working"])
    assert not current_stop(restored["sleeper"]), restored["sleeper"].get("_budget_pause_hold")
    _mail(tmp_path, "sleeper", "ready", sender="peer")
    assert wake_ready_sleepers(q)[0]["ok"]
    assert not in_progress


def test_restart_flag_names_exist_only_in_the_boot_compatibility_reader():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    # Include launchers, Android and container/bootstrap scripts, not only Python owners.
    matches = subprocess.check_output([
        "git", "grep", "-l", "owner_restart_no_resume", "--", ":!tests", ":!docs", ":!*.md",
    ], cwd=root, text=True).splitlines()
    assert matches == ["ouroboros/delegate_recovery.py"]
    for directory in (root / "ouroboros", root / "supervisor"):
        for path in directory.rglob("*.py"):
            if path.name == "delegate_recovery.py":
                continue
            source = path.read_text(encoding="utf-8")
            assert "owner_restart_no_resume" not in source, path
            if path.name not in {"installation_stop.py", "server_control.py"}:
                assert '"panic_stop.flag"' not in source, path
    assert "owner_restart_no_resume" not in (root / "server.py").read_text(encoding="utf-8")


def test_removed_history_fields_are_not_runtime_inputs_or_outputs():
    import ast

    removed = {"prior_hold", "released_reason", "stale_root_grant_id", "prior_hold_reason",
               "rebound_at", "replaced_fence_marker", "dispatchable"}
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "ouroboros", root / "supervisor"):
        for path in directory.rglob("*.py"):
            names = {node.value for node in ast.walk(parse(path.read_text(encoding="utf-8")))
                     if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            assert not names & removed, (path, names & removed)


def test_captured_switch_uses_the_same_panic_receipt(tmp_path):
    import runpy
    from ouroboros import body_switch, installation_stop

    state = tmp_path / "state"
    state.mkdir()
    (state / "panic_stop.flag").write_text("panic", encoding="utf-8")
    # The captured reader has only the stdlib, exactly as before product import.
    reader = runpy.run_path(installation_stop.__file__)
    assert reader["installation_panic"](tmp_path)
    (state / "state.json").write_text(json.dumps({PANIC_RECEIPT: panic_input(tmp_path)}), encoding="utf-8")
    assert not reader["installation_panic"](tmp_path)
    assert not body_switch.installation_panic(tmp_path)


def test_unreadable_panic_content_retains_the_existing_installation_veto(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(evolution_lifecycle, "_STOP_LATCH", {"stopped": False})
    (tmp_path / "state").mkdir()
    (tmp_path / "state/panic_stop.flag").write_bytes(b"\xff")
    assert installation_panic(tmp_path)
    assert evolution_lifecycle.evolution_stop_reason({}) == "panic_flag_unknown"


def test_boot_panic_holds_need_the_owners_word_after_that_panic_or_his_press(tmp_path, monkeypatch):
    """Emergency Stop Invariant, read 21 Nov 2026: after Panic a mind releases a hold of either class (``panic_hold``
    for sleeping work, ``saved_work_hold{door: panic}`` for working) only with the owner's own message written after
    that Panic began, cited and verified by author and time; the owner's press works as before. No cite, or a cite
    from before the Panic, refuses; a second Panic voids the first cite. The Panic moment is taken when it starts
    and kept across the restart (``panic_started_at``)."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import server_control
    from ouroboros.control_actor import citing_owner, mind_control
    from supervisor import worker_chat_lane
    from tests.test_model_sleep import _cold_park
    from tests.test_restart_saved_work import _working_run
    from tests.test_state_authority import _prior

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=3600)
    _working_run(tmp_path, workers, "working")
    assert q.persist_queue_snapshot(reason="before_panic")
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    monkeypatch.setattr(server_control, "_PANIC_STARTED_AT", started.timestamp())
    server_control._persist_panic_controls(tmp_path)  # the stopping process records when its Panic began
    (tmp_path / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
    workers.PENDING.clear()
    workers.RUNNING.clear()
    assert q.restore_pending_from_snapshot() == 2
    rows = {task["id"]: task for task in workers.PENDING}
    assert current_stop(rows["sleeper"])["release"] == "owner"
    assert current_stop(rows["working"])["door"] == "panic"
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()["panic_started_at"] == started.isoformat(), "the start, not the boot, survives"
    assert "panic_pending_started_at" not in state.load_state()
    before, after = (started - timedelta(seconds=1)).isoformat(), (started + timedelta(seconds=1)).isoformat()
    for task_id in ("sleeper", "working"):
        with mind_control("mind", "continue saved work"):
            assert q.resume_budget_paused_task(task_id)["error"] == "panic_requires_owner_resume", task_id
            with citing_owner(before):
                assert q.resume_budget_paused_task(task_id)["error"] == "panic_requires_owner_resume", task_id
    # A second Panic voids a cite written after the first one only.
    state.update_state(lambda st: st.update(panic_started_at=(started + timedelta(minutes=1)).isoformat()))
    with mind_control("mind", "continue saved work"), citing_owner(after):
        assert q.resume_budget_paused_task("working")["error"] == "panic_requires_owner_resume"
    state.update_state(lambda st: st.update(panic_started_at=started.isoformat()))
    with mind_control("mind", "continue saved work"), citing_owner(after):
        assert q.resume_budget_paused_task("working")["ok"]
    assert q.resume_budget_paused_task("sleeper")["ok"], "the owner's own press works as before"


def test_a_launcher_panic_with_no_stopping_server_dates_the_panic_at_the_boot_that_takes_it(tmp_path, monkeypatch):
    """No server recorded a start (a tray or launcher Panic): the boot's receipt time stands, before which no owner
    message can have reached the host; a cite older than it never releases a Panic hold."""
    from ouroboros.utils import utc_now_iso
    from supervisor import worker_chat_lane
    from tests.test_state_authority import _prior

    q, state, _workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    (tmp_path / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
    earlier = utc_now_iso()
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()["panic_started_at"] >= earlier


def test_the_panic_start_is_recorded_before_the_barrier_from_any_doors_ingress_and_never_goes_stale(tmp_path,
                                                                                                     monkeypatch):
    """The boot that takes a Panic records its start before the holds' barrier, so a failed barrier leaves no stale
    start; until the input is taken every mind release refuses, a cite after the Panic included. The start is the
    ingress time of any door (``panic_source.json``, the tray's too) when newer than the Panic before; an older one
    (a record this Panic failed to rewrite) gives the boot's own time. A stopping server's time no Panic followed
    is dropped."""
    import json
    from datetime import datetime, timedelta, timezone

    from ouroboros.control_actor import citing_owner, mind_release_refusal
    from supervisor import queue, worker_chat_lane
    from tests.test_state_authority import _prior

    _q, state, _workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    tray = datetime.now(timezone.utc) - timedelta(minutes=3)
    (tmp_path / "state/panic_source.json").write_text(json.dumps({"time": tray.isoformat()}), encoding="utf-8")
    (tmp_path / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda **_kw: False)
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()["panic_started_at"] == tray.isoformat(), "recorded although the barrier failed"
    assert installation_panic(tmp_path), "the input is kept"
    with citing_owner((tray + timedelta(minutes=1)).isoformat()):
        assert mind_release_refusal({"cause": "pause", "requested_at": tray.isoformat()}), "not yet taken: blocked"
    monkeypatch.undo()
    _q, state, _workers = _install_queue(tmp_path, monkeypatch)
    worker_chat_lane.auto_resume_after_restart()
    assert not installation_panic(tmp_path) and state.load_state()["panic_started_at"] == tray.isoformat()
    with citing_owner((tray + timedelta(minutes=1)).isoformat()):
        assert mind_release_refusal({"cause": "panic"}) is None, "taken: a later word releases"
        assert mind_release_refusal({}) is None, "no stop at all is never refused"
    with citing_owner((tray - timedelta(seconds=1)).isoformat()):
        assert mind_release_refusal({"cause": "panic"}), "a word from before it does not"
    # A second Panic whose ingress record was not rewritten: the old record is older, the boot's time stands.
    (tmp_path / "state/panic_stop.flag").write_text("panic", encoding="utf-8")
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state()["panic_started_at"] > tray.isoformat()
    state.update_state(lambda st: st.update(panic_pending_started_at=tray.isoformat()))
    worker_chat_lane.auto_resume_after_restart()
    assert "panic_pending_started_at" not in state.load_state(), "no Panic input followed: dropped"


def test_android_reader_cli_recognizes_the_receipt_without_product_import(tmp_path):
    import subprocess
    import sys
    from ouroboros import installation_stop

    state = tmp_path / "state"
    state.mkdir()
    (state / "panic_stop.flag").write_text("panic", encoding="utf-8")
    command = [sys.executable, "-I", "-S", installation_stop.__file__, str(tmp_path)]
    assert subprocess.run(command, check=False).returncode == 0
    (state / "state.json").write_text(json.dumps({PANIC_RECEIPT: panic_input(tmp_path)}), encoding="utf-8")
    assert subprocess.run(command, check=False).returncode == 1
    assert subprocess.run(command[:-1], check=False).returncode == 2
