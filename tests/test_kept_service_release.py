"""What a kept service owes until it is captured, end to end through real processes and real retirement.

A successful capture releases the cached retirement hold; a repeated start of a running launch changes nothing about
its declaration; an owed declaration costs its readers only what is owed; a session scratch holding its outputs waits
for the capture, whichever pass reaches it; a declaration or directory that cannot be read stops the retirement reading
it, deleting nothing; a local launch is one row under the id its declaration names, on every platform (the Windows
branch here is controlled: the platform flags are set, the processes are this host's)."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from ouroboros import headless, maintenance, maintenance_jobs, process_custody, session_scratch, task_custody
from ouroboros import workspace_executor
from ouroboros.artifacts import registered_task_artifact
from ouroboros.launch_custody import stop_kept_service
from ouroboros.task_results import write_task_result
from ouroboros.tools import kept_service_outputs as kept, services, shell_outputs
from ouroboros.tools.kept_service_outputs import (
    KeptDeclarationUnreadable, capture_kept_service_outputs, declaration_path, owed_declarations, owed_output_hold,
    unconsumed_declaration,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.services import stop_task_services
from ouroboros.workspace_preflight import release_finished_scratch
from tests._cost_meter import CostMeter, tagged
from tests._launch_stop_shared import _PATIENCE, _end, _executor_ctx, write_scripts
from tests._store_shared import seed_result
from tests.test_launch_stop import EXITING_WRITER, _kept_view, _queue_view, _registry, _start_late_writer, _wait_report
from tests.test_session_scratch_release import OVER, _record, _started

REPORT = b"report written after its task ended\n"
READY = {"log_contains": "READY", "timeout_sec": 20}


def test_capture_completion_releases_the_retirement_walks_cached_hold(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch, task_id="producer")
    _started, task_drive, (pid, birth) = _start_late_writer(registry, write_scripts(tmp_path), tmp_path)
    drive, rested, rest = tmp_path / "data", {}, {}
    real_copy = shell_outputs.copy_file_to_task_artifacts

    def walk():
        return headless.prune_task_drives(drive, retention_days=0, now=time.time() + 60,
                                          live=lambda _task: False, rested=rested, rest=rest)

    def refuse(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    try:
        assert [item["lifecycle"] for item in stop_task_services(registry._ctx)] == ["kept"]
        write_task_result(drive, "producer", "completed")
        (tmp_path / "go").write_text("go", encoding="utf-8")
        _wait_report(task_drive)
        (service,) = _kept_view(drive)["services"]
        monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", refuse)
        stopped = stop_kept_service(drive, service["launch_id"], reason="done", actor="owner:test")
        assert stopped["outcome"] == "stopped" and stopped["outputs"]["state"] == "incomplete", stopped
        first = walk()
        assert any(row.get("reason") == "kept_service_capture_owed" for row in first["skipped"]), first
        second = walk()  # the ordinary maintenance consumer may now remember the hold
        monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", real_copy)
        captured = stop_kept_service(drive, service["launch_id"], reason="retry capture", actor="owner:test")
        assert captured["ok"] and captured["outputs"]["state"] == "captured", captured
        artifact = registered_task_artifact(drive, "producer", "report.txt")
        assert Path(artifact["path"]).read_bytes() == b"report written after its task ended\n"
        final = walk()
        print(json.dumps({"before": first, "unchanged": second, "after_capture": final}))
        assert not final["rested"], "the consumed declaration must invalidate an owed-capture rest"
        assert all(row.get("retained") != "kept_service_capture_owed" and
                   row.get("reason") != "kept_service_capture_owed" for row in final["skipped"]), final
    finally:
        monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", real_copy)
        registry.execute("stop_service", {"name": "reporter"})
        _end(pid, birth)


def _writer(tmp_path, name="writer.py"):
    script = tmp_path / name
    script.write_text(EXITING_WRITER, encoding="utf-8")
    return script


def _wait_ended(drive, launch_id):
    deadline = time.monotonic() + _PATIENCE
    while [item["state"] for item in _kept_view(drive)["services"] if item["launch_id"] == launch_id] != ["capture_owed"]:
        assert time.monotonic() < deadline, "its exit was never read as ended"
        time.sleep(0.05)


def test_a_repeated_start_of_a_running_kept_service_keeps_its_declaration_and_its_capture(tmp_path, monkeypatch):
    """The second start of the same name while the launch runs starts nothing (``already_running``): what that
    launch's start declared stands, byte for byte, so after its producer ended and it exited the outside stop captures
    the original report instead of reading ``none_declared`` from an empty declaration."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    (tmp_path / "system").mkdir(exist_ok=True)
    drive, go = tmp_path / "data", tmp_path / "go"
    cmd = [sys.executable, str(_writer(tmp_path)), str(go)]
    first = json.loads(services._start_service(ctx, name="reporter", cmd=cmd, cwd=str(workspace), outputs=["report.txt"],
                                               keep_alive=True, readiness=READY))
    assert first["executor"]["type"] == "local" and "kept_declaration_problem" not in first, first
    (service,) = _kept_view(drive)["services"]
    declaration = declaration_path(drive, "producer", service["launch_id"])
    recorded = declaration.read_bytes()
    assert json.loads(recorded)["outputs"] == ["report.txt"] and json.loads(recorded)["before_outputs"]
    for repeat in ({}, {"outputs": ["decoy.txt"]}):  # outputs omitted, then different ones: the launch started once
        again = json.loads(services._start_service(ctx, name="reporter", cmd=cmd, cwd=str(workspace), keep_alive=True,
                                                   **repeat))
        assert again["note"] == "already_running" and again["outputs"] == ["report.txt"], again
        assert declaration.read_bytes() == recorded, "a repeated start must not rewrite the running launch's declaration"
    assert not declaration_path(drive, "producer", service["launch_id"], owed=False).exists(), "nor write a second one"
    try:
        assert [item["lifecycle"] for item in stop_task_services(ctx)] == ["kept"]
        write_task_result(drive, "producer", "completed")
        _queue_view(drive)
        go.write_text("go", encoding="utf-8")
        _wait_report(workspace, REPORT)
        _wait_ended(drive, service["launch_id"])
        receipt = stop_kept_service(drive, service["launch_id"], reason="collect the report it left", actor="owner:test")
        assert receipt["outcome"] == "not_found" and receipt["ok"] is True, receipt
        assert receipt["outputs"]["state"] == "captured" and receipt["outputs"]["registered"] is True, receipt
        assert Path(registered_task_artifact(drive, "producer", "report.txt")["path"]).read_bytes() == REPORT
        assert not declaration.exists(), "a complete capture consumes it"
    finally:
        services._stop_service(ctx, name="reporter")


def test_a_true_new_launch_declares_its_own_outputs_and_the_old_launchs_debt_stands(tmp_path, monkeypatch):
    """A start after the old launch exited is a new launch: its declaration is its own; the old one, still owed, is
    not overwritten and can still be captured."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    (tmp_path / "system").mkdir(exist_ok=True)
    drive, first_go, second_go = tmp_path / "data", tmp_path / "go-1", tmp_path / "go-2"
    script = _writer(tmp_path)
    json.loads(services._start_service(ctx, name="reporter", cmd=[sys.executable, str(script), str(first_go)],
                                       cwd=str(workspace), outputs=["report.txt"], keep_alive=True, readiness=READY))
    (old,) = _kept_view(drive)["services"]
    old_path = declaration_path(drive, "producer", old["launch_id"])
    old_bytes = old_path.read_bytes()
    first_go.write_text("go", encoding="utf-8")
    _wait_report(workspace, REPORT)
    _wait_ended(drive, old["launch_id"])
    new_cmd = [sys.executable, str(script), str(second_go)]
    second = json.loads(services._start_service(ctx, name="reporter", cmd=new_cmd, cwd=str(workspace), keep_alive=True,
                                                outputs=["other.txt"], readiness=READY))
    try:
        assert second.get("note") != "already_running", second
        by_id = {item["launch_id"]: item for item in _kept_view(drive)["services"]}
        new_id = next(launch for launch in by_id if launch != old["launch_id"])
        assert by_id[old["launch_id"]]["state"] == "capture_owed" and old_path.read_bytes() == old_bytes
        assert json.loads(declaration_path(drive, "producer", new_id).read_text(encoding="utf-8"))["outputs"] == ["other.txt"]
    finally:
        second_go.write_text("go", encoding="utf-8")
        services._stop_service(ctx, name="reporter")
    receipt = stop_kept_service(drive, old["launch_id"], reason="the first report", actor="owner:test")
    assert receipt["ok"] is True and receipt["outputs"]["state"] == "captured", receipt
    assert Path(registered_task_artifact(drive, "producer", "report.txt")["path"]).read_bytes() == REPORT


def test_the_kept_readers_cost_what_is_owed_not_the_installations_history(tmp_path):
    """Hot reads (the Runtime fact's inventory and every retirement's gate) enumerate only outstanding declarations:
    finished tasks, their logs and no-output declarations (beside the logs, in child drives too) cost them nothing."""
    from ouroboros.launch_custody import kept_services

    counted = {}
    for history in (3, 60):
        data = tmp_path / f"k{history:03d}"
        for index in range(history):  # what an installation's past leaves behind, on the layouts the old scan walked
            task = f"task-{index:04d}"
            child = data / "state" / "headless_tasks" / task / "data"
            for root in (data, child, data / "task_drives" / task):
                (root / "services" / task).mkdir(parents=True, exist_ok=True)
                (root / "services" / task / "preview.log").write_text("done\n", encoding="utf-8")
                declaration_path(root, task, f"launch-{index:032x}", owed=False).write_text(
                    json.dumps({"schema": 1, "launch_id": f"launch-{index:032x}", "task_id": task, "outputs": []}),
                    encoding="utf-8")
        owed = declaration_path(data, "producer", "launch-" + "a" * 32)
        owed.parent.mkdir(parents=True)
        owed.write_text(json.dumps({"schema": 1, "launch_id": "launch-" + "a" * 32, "task_id": "producer", "cwd": str(data),
                                    "outputs": ["report.txt"], "name": "reporter"}), encoding="utf-8")
        with CostMeter(data) as meter:
            tagged("read", lambda: (owed_declarations(data), kept_services(data),
                                    owed_output_hold(data, data / "task_drives" / "task-0001")))
        counted[history] = meter.snapshot("read")["counts"]
    small, large = counted[3], counted[60]
    for key in ("scandir", "listdir", "dir_entries", "open", "json_loads"):
        assert small.get(key, 0) == large.get(key, 0), (key, small, large)
    assert len(owed_declarations(data)) == 1


def test_one_retirement_pass_reads_the_kept_debts_once(tmp_path, monkeypatch):
    """Every candidate of a pass reads the same selection: the drive walk and the scratch passes' cost is one read of
    what is owed, not one per candidate (the first gate; an actual move re-reads under its own lock)."""
    calls = []
    real = kept.owed_declarations
    monkeypatch.setattr(kept, "owed_declarations", lambda root: calls.append(root) or real(root))
    data = tmp_path / "data"
    outputs = []
    for index in range(3):
        task = f"task-{index}"
        drive = data / "task_drives" / task
        drive.mkdir(parents=True)
        write_task_result(data, task, "completed")
        out = declaration_path(data, task, f"launch-{index:032x}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"schema": 1, "launch_id": f"launch-{index:032x}", "task_id": task, "cwd": str(drive),
                                   "outputs": ["report.txt"]}), encoding="utf-8")
        outputs.append(out)
    settled = headless.prune_task_drives(data, retention_days=0, now=time.time() + 60, live=lambda _task: False)
    assert all(row.get("reason") == "kept_service_capture_owed" for row in settled["skipped"]), settled
    assert len(calls) == 1, "the first gate of every drive shares the pass's one read"
    calls.clear()
    task_custody.remove_custody_trash(data)
    release_finished_scratch(data, OVER)
    assert len(calls) == 2, "one read each, however many entries or notes the pass finds"


def test_owed_outputs_in_an_allocated_scratch_wait_for_their_capture_and_then_release(tmp_path, monkeypatch):
    """A real kept service of an external workspace writes its declared report into the session scratch allocated for
    the task and exits; the task is proven over. The custody pass and the trash remover keep the scratch and its
    record with the visible hold, Storage says why, and the retried capture still has the bytes; only then does the
    next pass release it. The control captures first: release is not delayed by a debt that was paid."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    data, scratch = _started(tmp_path)  # task-1's allocated scratch; its row is already terminal
    (tmp_path / "system").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=data, task_id="task-1", workspace_root=scratch,
                      workspace_mode="external")
    go = tmp_path / "go"
    started = json.loads(services._start_service(ctx, name="reporter", cmd=[sys.executable, str(_writer(tmp_path)), str(go)],
                                                 cwd=str(scratch), outputs=["report.txt"], keep_alive=True, readiness=READY))
    assert "kept_declaration_problem" not in started and started["cwd_root"] == "active_workspace", started
    (service,) = _kept_view(data)["services"]
    declaration = declaration_path(data, "task-1", service["launch_id"])
    try:
        assert [item["lifecycle"] for item in stop_task_services(ctx)] == ["kept"]
        go.write_text("go", encoding="utf-8")
        _wait_report(scratch, REPORT)
        _wait_ended(data, service["launch_id"])

        # The custody pass over the finished task: nothing is handed to the trash, the hold is said.
        passed = release_finished_scratch(data, OVER)
        assert passed["released"] == [] and passed["errors"] == [], passed
        assert [row["class"] for row in passed["holds"]] == ["kept_service_capture_owed"], passed
        assert (scratch / "report.txt").read_bytes() == REPORT and _record(data).exists()
        records = {row["key"]: row for row in session_scratch.scratch_records(data, OVER)}
        assert records["snap-1"]["hold"] == "kept_service_capture_owed", records

        # The drive's settlement moves the record into the trash whole (this hold is about the scratch, not the
        # drive): the remover keeps the scratch and hands its record to its own entry, the rest of the drive goes.
        settled = task_custody.settle_child_drive(data, "task-1", data / "task_drives" / "task-1", live=OVER,
                                                  admission_rollback=True)
        assert settled["status"] == "removed", settled
        monkeypatch.setattr(maintenance, "JOBS", {"custody_trash": maintenance.JOBS["custody_trash"]})
        assert maintenance.run_job(data, "custody_trash")["state"] == "done"
        job = maintenance_jobs.job(data, "custody_trash")
        [hold] = job["holds"]
        assert hold["class"] == "kept_service_capture_owed" and str(scratch) in hold["example"], hold
        assert (scratch / "report.txt").read_bytes() == REPORT, "the owed bytes outlive the release"
        [entry] = (data / "state" / "custody_trash").iterdir()
        assert entry.name.startswith("task-1+scratch-"), entry

        # The retried capture sees the original bytes, then the debt is paid and the next pass releases the scratch.
        receipt = stop_kept_service(data, service["launch_id"], reason="collect the report", actor="owner:test")
        assert receipt["ok"] is True and receipt["outputs"]["state"] == "captured", receipt
        artifact = registered_task_artifact(data, "task-1", "report.txt")
        assert Path(artifact["path"]).read_bytes() == REPORT and not declaration.exists()
        assert maintenance.run_job(data, "custody_trash")["state"] == "done"
        assert not scratch.exists() and maintenance_jobs.job(data, "custody_trash")["holds"] == []
    finally:
        services._stop_service(ctx, name="reporter")


def test_capturing_first_lets_the_same_scratch_release_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    data, scratch = _started(tmp_path)
    (tmp_path / "system").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=data, task_id="task-1", workspace_root=scratch,
                      workspace_mode="external")
    go = tmp_path / "go"
    json.loads(services._start_service(ctx, name="reporter", cmd=[sys.executable, str(_writer(tmp_path)), str(go)],
                                       cwd=str(scratch), outputs=["report.txt"], keep_alive=True, readiness=READY))
    (service,) = _kept_view(data)["services"]
    go.write_text("go", encoding="utf-8")
    _wait_report(scratch, REPORT)
    _wait_ended(data, service["launch_id"])
    receipt = stop_kept_service(data, service["launch_id"], reason="collect it first", actor="owner:test")
    assert receipt["ok"] is True and receipt["outputs"]["state"] == "captured", receipt
    passed = release_finished_scratch(data, OVER)
    assert passed["holds"] == [] and len(passed["released"]) == 1, passed
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()
    assert Path(registered_task_artifact(data, "task-1", "report.txt")["path"]).read_bytes() == REPORT


def test_a_local_launch_is_one_row_under_its_declaration_id_on_the_windows_branch(tmp_path, monkeypatch):
    """Controlled, not native: the platform flags say Windows while this host's real process and ledger stand in. The
    durable marker still names one launch: one row, the id the declaration names, both custody records stopped."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    (tmp_path / "system").mkdir(exist_ok=True)
    drive = tmp_path / "data"
    go = tmp_path / "go"
    started = json.loads(services._start_service(ctx, name="windowsy", cmd=[sys.executable, str(_writer(tmp_path)), str(go)],
                                                 cwd=str(workspace), outputs=["report.txt"], keep_alive=True, readiness=READY))
    assert started["executor"]["type"] == "local", started
    record = workspace_executor._SERVICES["producer:windowsy"]
    assert record.scope_marker and record.durable_record_path.exists()
    pid = int(started["backend_pid"])
    monkeypatch.setattr(workspace_executor, "IS_WINDOWS", True)
    monkeypatch.setattr(process_custody, "IS_WINDOWS", True)
    reaper_stop = threading.Event()  # POSIX keeps an unreaped child as a zombie; Windows has none to reap
    reaper = threading.Thread(target=lambda: [record.local_proc.poll() or reaper_stop.wait(0.02)
                                              for _ in iter(lambda: reaper_stop.is_set(), True)], daemon=True)
    reaper.start()
    try:
        assert not workspace_executor._recorded_launch(workspace_executor._load_process_record(record.durable_record_path))
        view = _kept_view(drive)
        (service,) = view["services"]  # never a second row under the executor record's service-<hex> id
        launch_id = service["launch_id"]
        assert launch_id.startswith("launch-") and service["backend"] == "local_executor", service
        assert declaration_path(drive, "producer", launch_id).exists(), "the declaration names the same launch"
        receipt = stop_kept_service(drive, launch_id, reason="windows branch stop", actor="owner:test")
        assert receipt["outcome"] == "stopped", receipt
        assert not record.durable_record_path.exists(), "the executor record's stop still ran"
        from ouroboros.platform_layer import pid_is_alive

        assert not pid_is_alive(pid), "its process ended"
    finally:
        reaper_stop.set()
        reaper.join()
        monkeypatch.setattr(workspace_executor, "IS_WINDOWS", False)
        monkeypatch.setattr(process_custody, "IS_WINDOWS", False)
        services._stop_service(ctx, name="windowsy")
        _end(pid, "")


def test_the_ui_types_name_every_state_the_kept_contract_returns():
    types = (Path(__file__).resolve().parents[1] / "web" / "modules" / "task_activity_types.js").read_text(encoding="utf-8")
    assert "'capture_owed'" in types, "a capture-owed launch is listed"
    assert "|'')=} backend" in types, "an ended launch has no backend to name"
    assert "not_found capture-only" in types, "a capture-only retry can be ok"


def test_unreadable_owed_declaration_does_not_release_its_producers_scratch(tmp_path, monkeypatch):
    """A temporary declaration read failure cannot turn an outstanding capture into disposable scratch."""
    data, scratch = _started(tmp_path)
    report = scratch / "report.txt"
    report.write_bytes(REPORT)
    declaration = declaration_path(data, "task-1", "launch-" + "e" * 32)
    declaration.parent.mkdir(parents=True)
    declaration.write_text(json.dumps({"schema": 1, "task_id": "task-1", "outputs": ["report.txt"],
                                        "cwd": str(scratch)}), encoding="utf-8")
    read = Path.read_text

    def unavailable(path, *args, **kwargs):
        if path == declaration:
            raise OSError("temporary declaration read failure")
        return read(path, *args, **kwargs)

    try:
        with monkeypatch.context() as failed_read:
            failed_read.setattr(Path, "read_text", unavailable)
            with pytest.raises(KeptDeclarationUnreadable):
                release_finished_scratch(data, OVER)
            with pytest.raises(KeptDeclarationUnreadable):
                task_custody.remove_custody_trash(data)
            assert report.exists() and report.read_bytes() == REPORT
    finally:
        # This test manufactured the debt and has no live service; retire only this fixture.
        declaration.unlink(missing_ok=True)
        release_finished_scratch(data, OVER)
        task_custody.remove_custody_trash(data)


def _declare_in_outstanding(data, task, launch, body):
    path = declaration_path(data, task, launch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body) if not isinstance(body, str) else body, encoding="utf-8")
    return path


def _read_fails(context, declaration):
    """Only this declaration's read fails, and only inside ``context``: the temporary case the operations must survive."""
    read = Path.read_text

    def unavailable(path, *args, **kwargs):
        if path == declaration:
            raise OSError("temporary declaration read failure")
        return read(path, *args, **kwargs)

    context.setattr(Path, "read_text", unavailable)


def test_a_kept_service_in_another_tasks_scratch_keeps_its_bytes_while_its_declaration_is_unreadable(tmp_path, monkeypatch):
    """The service runs and writes in another task's allocated scratch, so no owner fact of that scratch names the debt;
    only the declaration does. While it cannot be read the retirement stops at its own error boundary and no source bytes
    are released, in either task's scratch. Readable again, the same launch id captures the original bytes into the
    producer's store with nothing run again, and then the ordinary passes release both scratches."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    data, own = _started(tmp_path)
    _other_data, other = _started(tmp_path, task_id="task-2")
    (tmp_path / "system").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "system", drive_root=data, task_id="task-1", workspace_root=other,
                      workspace_mode="external")
    go = tmp_path / "go"
    started = json.loads(services._start_service(ctx, name="reporter", cmd=[sys.executable, str(_writer(tmp_path)), str(go)],
                                                 cwd=str(other), outputs=["report.txt"], keep_alive=True, readiness=READY))
    assert "kept_declaration_problem" not in started and started["cwd_root"] == "active_workspace", started
    (service,) = _kept_view(data)["services"]
    declaration = declaration_path(data, "task-1", service["launch_id"])
    try:
        assert [item["lifecycle"] for item in stop_task_services(ctx)] == ["kept"]
        go.write_text("go", encoding="utf-8")
        _wait_report(other, REPORT)
        _wait_ended(data, service["launch_id"])
        with monkeypatch.context() as failed_read:
            _read_fails(failed_read, declaration)
            with pytest.raises(KeptDeclarationUnreadable):
                release_finished_scratch(data, OVER)
            with pytest.raises(KeptDeclarationUnreadable):
                task_custody.remove_custody_trash(data)
            view = _kept_view(data)
            assert [(row["launch_id"], row["state"]) for row in view["services"]] == [(service["launch_id"], "capture_owed")]
            assert (other / "report.txt").read_bytes() == REPORT and (own / "env" / "pyvenv.cfg").exists()
            assert _record(data).exists() and _record(data, "task-2").exists(), "neither record was handed on"

        receipt = stop_kept_service(data, service["launch_id"], reason="the declaration reads again", actor="owner:test")
        assert receipt["outcome"] == "not_found" and receipt["ok"] is True, receipt
        assert receipt["outputs"]["state"] == "captured", receipt
        artifact = registered_task_artifact(data, "task-1", "report.txt")
        assert Path(artifact["path"]).read_bytes() == REPORT and not declaration.exists()
        assert len(_kept_view(data)["services"]) == 0, "the capture ran nothing again"
        released = release_finished_scratch(data, OVER)
        assert released["holds"] == [] and released["errors"] == [] and sorted(released["released"]) == sorted(
            [str(_record(data).parent), str(_record(data, "task-2").parent)]), released
        task_custody.remove_custody_trash(data)
        assert not own.exists() and not other.exists()
    finally:
        services._stop_service(ctx, name="reporter")


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs a directory the OS refuses this process")
@pytest.mark.parametrize("level", ["outstanding", "producer"])
def test_a_debt_directory_that_cannot_be_listed_stops_the_retirement(tmp_path, level):
    """``Path.glob`` drops a refused directory, so a retirement would read "no debts" and free what they name. The listing
    is strict: each operation stops at its own error boundary with the ``PermissionError`` and the source stays; once it
    can be listed again nothing needs repair and the debt still holds what it names."""
    from ouroboros.tools.services import prune_service_logs

    data, scratch = _started(tmp_path)
    (scratch / "report.txt").write_bytes(REPORT)
    launch = "launch-" + "a" * 32
    declaration = _declare_in_outstanding(data, "producer", launch, {"schema": 1, "launch_id": launch, "task_id": "producer",
                                                                      "cwd": str(scratch), "outputs": ["report.txt"]})
    refused = declaration.parent if level == "producer" else declaration.parent.parent
    refused.chmod(0)
    try:
        for read in (lambda: owed_declarations(data), lambda: unconsumed_declaration(data, launch),
                     lambda: release_finished_scratch(data, OVER), lambda: task_custody.remove_custody_trash(data),
                     lambda: prune_service_logs(data, retention_days=0, now=time.time() + 1)):
            with pytest.raises(PermissionError):
                read()
        assert (scratch / "report.txt").read_bytes() == REPORT and _record(data).exists()
        from ouroboros.launch_custody import kept_services

        view = kept_services(data)
        assert view["complete"] is False and "custody could not be read" in view["note"], view
        receipt = stop_kept_service(data, launch, reason="while it is refused", actor="owner:test")
        assert receipt["outcome"] == "unknown" and "custody could not be read: PermissionError" in receipt["detail"], receipt
    finally:
        refused.chmod(0o700)
    assert [path for path, _declaration in owed_declarations(data)] == [declaration]
    assert unconsumed_declaration(data, launch)[1] == "producer"
    held = release_finished_scratch(data, OVER)
    assert held["released"] == [] and [row["class"] for row in held["holds"]] == ["kept_service_capture_owed"], held
    declaration.unlink()
    assert release_finished_scratch(data, OVER)["released"] == [str(_record(data).parent)]
    task_custody.remove_custody_trash(data)
    assert not scratch.exists()


def test_a_readable_debt_holds_only_what_it_names_and_an_unrelated_scratch_releases(tmp_path):
    """The normal case stays exact: no directory yet is no debt; a readable debt holds the scratch it names, and another
    finished task's scratch goes in the same pass."""
    data, held_scratch = _started(tmp_path)
    assert owed_declarations(data) == []
    _other_data, free = _started(tmp_path, task_id="task-2")
    launch = "launch-" + "b" * 32
    _declare_in_outstanding(data, "task-1", launch, {"schema": 1, "launch_id": launch, "task_id": "task-1",
                                                     "cwd": str(held_scratch), "outputs": ["report.txt"]})
    passed = release_finished_scratch(data, OVER)
    assert passed["released"] == [str(_record(data, "task-2").parent)] and passed["errors"] == [], passed
    assert [row["class"] for row in passed["holds"]] == ["kept_service_capture_owed"], passed
    task_custody.remove_custody_trash(data)
    assert not free.exists() and held_scratch.exists()


def test_an_unreadable_declaration_keeps_a_drive_at_the_settlements_own_error_boundary(tmp_path):
    """The settlement names why it kept the drive; with the declaration gone the same drive settles."""
    data = tmp_path / "data"
    drive = data / "task_drives" / "task-1"
    drive.mkdir(parents=True)
    seed_result(data, "task-1", {"status": "completed"})
    declaration = _declare_in_outstanding(data, "retry-1", "launch-" + "c" * 32, "{ not json")
    settled = task_custody.settle_child_drive(data, "task-1", drive, live=OVER, admission_rollback=True)
    assert settled["status"] == "retained" and settled["reason"] == "custody_error: KeptDeclarationUnreadable", settled
    assert drive.is_dir()
    declaration.unlink()
    settled = task_custody.settle_child_drive(data, "task-1", drive, live=OVER, admission_rollback=True)
    assert settled["reason"] != "custody_error: KeptDeclarationUnreadable", settled


def test_an_unreadable_declaration_keeps_every_copy_its_read_decides(tmp_path, monkeypatch):
    """A delegated snapshot's own proof fails closed; a body candidate's verdict raises to its pass's boundary."""
    from ouroboros import body_candidate, delegate_custody, owner_keep, subagent_worktrees

    data = tmp_path / "data"
    _declare_in_outstanding(data, "task-1", "launch-" + "e" * 32, "{ not json")
    monkeypatch.setattr(subagent_worktrees, "_task_proven_over", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(owner_keep, "is_kept", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(delegate_custody, "custody_log_unreadable", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(delegate_custody, "open_snapshot_ids", lambda *_args, **_kwargs: set())

    for task in ("task-1", "task-2"):  # whoever's copy it is: the debts are not known whole
        entry = {"snapshot_id": f"snap-{task}", "task_id": task, "path": str(tmp_path / f"copy-{task}")}
        assert not subagent_worktrees.snapshot_released(entry, set(), data_dir=data, live=OVER)
    with pytest.raises(KeptDeclarationUnreadable):
        body_candidate._capture_owed({"path": str(tmp_path / "body"), "task_id": "task-2"}, data)


MALFORMED = [
    {"schema": 1, "launch_id": "L", "task_id": "task-1", "outputs": []},  # empty outputs in the outstanding directory
    {"schema": 1, "launch_id": "L", "task_id": "task-1"},  # no outputs, no working directory
    {"schema": 1, "launch_id": "L", "task_id": "task-1", "outputs": ["report.txt"]},  # outputs, no working directory
    {"schema": 1, "launch_id": "L", "task_id": "task-1", "cwd": "/nowhere", "outputs": "report.txt"},  # not a list
    {"schema": 1, "launch_id": "L", "task_id": "task-1", "cwd": 7, "outputs": ["report.txt"]},  # not a path
    ["a", "list"],
]


@pytest.mark.parametrize("malformed", MALFORMED, ids=["empty-outputs", "no-outputs", "no-cwd", "outputs-string", "cwd-number",
                                                      "not-a-mapping"])
def test_anything_malformed_in_the_outstanding_directory_stops_the_retirement_and_stays_owed(tmp_path, malformed):
    """Only a start naming outputs writes this directory, so a mapping there that cannot say what it holds is damage,
    not "owes nothing" (a true no-output declaration lies beside its log): the retirement stops, retention keeps and
    counts it, the inventory lists it and a capture refuses to read it, each by its existing path."""
    from ouroboros.tools.services import prune_service_logs

    data, scratch = _started(tmp_path)
    (scratch / "report.txt").write_bytes(REPORT)
    launch = "launch-" + "f" * 32
    if isinstance(malformed, dict):
        malformed = {**malformed, "launch_id": launch}
    declaration = _declare_in_outstanding(data, "task-1", launch, malformed)
    for read in (lambda: owed_declarations(data), lambda: release_finished_scratch(data, OVER),
                 lambda: task_custody.remove_custody_trash(data)):
        with pytest.raises(KeptDeclarationUnreadable):
            read()
    assert (scratch / "report.txt").read_bytes() == REPORT and declaration.exists() and _record(data).exists()

    assert [row["launch_id"] for row in _kept_view(data)["services"]] == [launch]
    report = prune_service_logs(data, retention_days=0, now=time.time() + 1)
    assert declaration.exists() and report["capture_owed"] == [launch], report
    captured = capture_kept_service_outputs(data, "task-1", launch, actor="owner:test", reason="try the damaged one")
    assert captured["state"] == "unavailable" and declaration.exists(), captured
    declaration.unlink()
    assert release_finished_scratch(data, OVER)["released"] == [str(_record(data).parent)]
    task_custody.remove_custody_trash(data)
    assert not scratch.exists()
