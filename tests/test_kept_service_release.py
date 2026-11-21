"""What a kept service owes until it is captured, end to end through real processes and real retirement.

A successful capture releases the cached retirement hold; a repeated start of a running launch changes nothing about
its declaration; an owed declaration costs its readers only what is owed; a session scratch holding its outputs waits
for the capture, whichever pass reaches it; a declaration or directory that cannot be read stops the retirement reading
it, deleting nothing; a local launch is one row under the id its declaration names, on every platform (the Windows
branch here is controlled: the platform flags are set, the processes are this host's)."""
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

from ouroboros import (
    headless,
    maintenance,
    maintenance_jobs,
    process_custody,
    session_scratch,
    task_custody,
    workspace_executor,
)
from ouroboros.artifacts import registered_task_artifact, service_output_version_digest, task_artifact_dir_path
from ouroboros.launch_custody import stop_kept_service
from ouroboros.task_results import write_task_result
from ouroboros.tools import kept_service_outputs as kept
from ouroboros.tools import services, shell_outputs
from ouroboros.tools.kept_service_outputs import (
    KeptDeclarationUnreadable,
    capture_kept_service_outputs,
    declaration_path,
    owed_declarations,
    owed_output_hold,
    unconsumed_declaration,
)
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.services import stop_task_services
from ouroboros.workspace_preflight import release_finished_scratch
from tests._cost_meter import CostMeter, tagged
from tests._launch_stop_shared import _PATIENCE, _end, _executor_ctx, write_scripts
from tests._store_shared import seed_result
from tests.test_launch_stop import EXITING_WRITER, _kept_view, _queue_view, _registry, _start_late_writer, _wait_report
from tests.test_log_forwarding import _ctx, production_sink  # noqa: F401 -- the production sink fixture, reused
from tests.test_session_scratch_release import OVER, _record, _started
from tests.test_session_service_directories import _synthetic_temp_parent  # noqa: F401 -- this run's own scratch

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
        assert again["note"] == "already_running" and again["launch_id"] == first["launch_id"], again
        assert "outputs" not in again and "backend_log_path" not in again
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


@pytest.mark.parametrize(("backend", "failure_at"), [("local", ""), ("local", "replacement"), ("host", "explicit_stop")])
def test_a_true_new_launch_captures_the_owed_bytes_before_its_writer_starts(tmp_path, monkeypatch, backend, failure_at):
    """A failed capture keeps the previous bytes and declaration. Retrying start captures them before reuse."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    if backend == "host":
        ctx.executor_ref = {}
    (tmp_path / "system").mkdir(exist_ok=True)
    drive, first_go = tmp_path / "data", tmp_path / "go-1"
    script = _writer(tmp_path)
    json.loads(services._start_service(ctx, name="reporter", cmd=[sys.executable, str(script), str(first_go)],
                                       cwd=str(workspace), outputs=["report.txt"], keep_alive=True, readiness=READY))
    (old,) = _kept_view(drive)["services"]
    old_path = declaration_path(drive, "producer", old["launch_id"])
    old_bytes = old_path.read_bytes()
    first_go.write_text("go", encoding="utf-8")
    _wait_report(workspace, REPORT)
    _wait_ended(drive, old["launch_id"])
    def start_again():
        cmd = [sys.executable, "-c", "import time; from pathlib import Path; Path('report.txt').write_bytes(b'second'); "
               "print('READY', flush=True); time.sleep(60)"]
        return services._start_service(ctx, name="reporter", cmd=cmd, cwd=str(workspace), keep_alive=True,
                                       outputs=["report.txt"], readiness=READY)

    if failure_at:
        def refuse(*args, **kwargs):
            raise OSError("synthetic artifact store failure")

        with monkeypatch.context() as patch:
            patch.setattr(shell_outputs, "copy_file_to_task_artifacts", refuse)
            if failure_at == "explicit_stop":
                assert "ARTIFACT_OUTPUT_ERROR" in services._stop_service(ctx, name="reporter")
            refused = start_again()
            assert "SERVICE_START_ERROR" in refused and "uncaptured outputs" in refused, refused
            # The confirmed stop is recorded in the same declaration (``finish_own_outputs``): what a retry needs
            # (the declared paths, their baseline, the producer's context) is unchanged, and so are the old bytes.
            held = json.loads(old_path.read_text(encoding="utf-8"))
            inputs = {key: value for key, value in json.loads(old_bytes).items() if key != "declared_at"}
            assert {key: held[key] for key in inputs} == inputs and (workspace / "report.txt").read_bytes() == REPORT
            assert held["stop"]["process_cleanup"]["exit_confirmed"] is True
            assert held["stop"]["output_outcomes"] == [{"index": 1, "status": "unconfirmed"}]
            assert [item["launch_id"] for item in _kept_view(drive)["services"]] == [old["launch_id"]]
    try:
        second = json.loads(start_again())
        assert second["ready"] and (workspace / "report.txt").read_bytes() == b"second"
        [new] = _kept_view(drive)["services"]
        assert not old_path.exists() and new["launch_id"] != old["launch_id"]
        assert Path(registered_task_artifact(drive, "producer", "report.txt")["path"]).read_bytes() == REPORT
        assert json.loads(declaration_path(drive, "producer", new["launch_id"]).read_text(encoding="utf-8"))["outputs"] == ["report.txt"]
    finally:
        services._stop_service(ctx, name="reporter")


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
        assert hold["class"] == "kept_service_capture_owed" and str(scratch.parent) in hold["example"], hold
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


def _ended_report(ctx, workspace, launch="launch-" + "b" * 32, outputs=("report.txt", "second.txt")):
    return _declare_in_outstanding(ctx.drive_root, ctx.task_id, launch, {
        "schema": 1, "launch_id": launch, "task_id": ctx.task_id, "name": "reporter",
        "cwd": str(workspace), "cwd_base": str(workspace), "cwd_root": "active_workspace",
        "outputs": list(outputs), "context": {
            "repo_dir": str(ctx.repo_dir), "workspace_root": str(workspace), "workspace_mode": "external",
            "drive_root": str(ctx.drive_root), "tool_profile": "advanced"},
    })


def _confirmed_report(ctx, workspace, launch="launch-" + "b" * 32):
    """An ended launch's outstanding declaration carrying its confirmed stop: what a late by-name
    capture reads — the saved outcome names the original declared file."""
    (workspace / "report.txt").write_bytes(REPORT)
    return _declare_in_outstanding(ctx.drive_root, ctx.task_id, launch, {
        "schema": 2, "launch_id": launch, "task_id": ctx.task_id, "name": "reporter",
        "cwd": str(workspace), "cwd_base": str(workspace), "cwd_root": "active_workspace",
        "outputs": ["report.txt"], "before_outputs": {}, "keep_alive": False,
        "context": {"repo_dir": str(ctx.repo_dir), "workspace_root": str(workspace),
                    "workspace_mode": "external", "drive_root": str(ctx.drive_root), "tool_profile": "advanced"},
        "stop": {"task_id": ctx.task_id, "launch_id": launch, "name": "reporter", "state": "exited",
                 "process_cleanup": {"exit_confirmed": True}, "log_outcome": "saved",
                 "output_outcomes": [{"index": 1, "status": "unconfirmed"}]}})


@pytest.mark.parametrize("error", ["missing", "permission", "copy", "receipt"])
def test_outside_stop_settles_only_recorded_capture_and_positive_absence(tmp_path, monkeypatch, error):
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _ended_report(ctx, workspace, launch)
    (workspace / "report.txt").write_bytes(REPORT)
    other = workspace / "second.txt"
    if error != "missing":
        other.write_bytes(b"second")
    if error == "permission":
        original = Path.stat

        def read(path, *args, **kwargs):
            if path == other:
                raise PermissionError("unreadable output")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "stat", read)
    elif error == "copy":
        original = shell_outputs.copy_file_to_task_artifacts
        monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts",
                            lambda ctx, src, **kw: None if src == other else original(ctx, src, **kw))
    elif error == "receipt":
        monkeypatch.setattr(kept, "append_jsonl", lambda *_args: False)
    receipt = stop_kept_service(ctx.drive_root, launch, reason="collect", actor="owner:test")
    assert receipt["outcome"] == "not_found" and not receipt["ok"]
    assert receipt["outputs"]["registered"]
    assert receipt["outputs"]["state"] == ("missing" if error == "missing" else "incomplete")
    assert path.exists() == (error != "missing")
    artifact = registered_task_artifact(ctx.drive_root, ctx.task_id, "report.txt")
    assert Path(artifact["path"]).read_bytes() == REPORT
    first, second = receipt["outputs"]["output_outcomes"]
    version_ref = first["artifacts"][0]["version_ref"]
    assert service_output_version_digest(version_ref["path"]) == artifact["sha256"]
    assert (task_artifact_dir_path(ctx.drive_root, ctx.task_id) / version_ref["path"]).read_bytes() == REPORT
    assert first == {"index": 1, "status": "saved", "registered": True,
                     "artifacts": [{"name": "report.txt", "sha256": artifact["sha256"], "version_ref": version_ref}]}
    assert second["index"] == 2 and second["status"] == {
        "missing": "absent", "permission": "unconfirmed", "copy": "not_saved", "receipt": "saved"}[error]
    if error == "missing":
        assert "missing output: second.txt" in receipt["outputs"]["detail"]
        assert unconsumed_declaration(ctx.drive_root, launch) is None


@pytest.mark.parametrize("invalid", [{}, {"cmd": ["writer"], "extra_option": True},
                                     {"cmd": ["writer"], "admitted": None}])
def test_invalid_direct_start_never_enters_replacement(tmp_path, monkeypatch, invalid):
    ctx, workspace = _executor_ctx(tmp_path)
    monkeypatch.setattr(workspace_executor, "_custody_drive_root", lambda *args: pytest.fail("accessed custody"))
    monkeypatch.setattr(services, "_admit_named_start", lambda *a, **k: pytest.fail("entered admission"))
    with pytest.raises(TypeError):
        workspace_executor.start_service(ctx, name="reporter", host_cwd=workspace, cwd_root="active_workspace",
                                         readiness={}, outputs=[], before_outputs={}, **invalid)


@pytest.mark.parametrize(("backend", "durable_retry"), [("host", False), ("local_executor", False), ("host", True)])
def test_unwritten_kept_declaration_holds_old_bytes_until_producer_can_register(tmp_path, monkeypatch, backend, durable_retry):
    from ouroboros import process_scope, utils
    from ouroboros.tools import shell
    from tests.test_service_finalization_failure import _ended_service

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    record, _, entries, _ = _ended_service(ctx.drive_root, monkeypatch, backend, task=ctx.task_id)
    record.keep_alive, record.record_id = True, "launch-" + "d" * 32
    record.cwd_root, record.cwd_base = "active_workspace", str(workspace)
    if backend == "host":
        record.cwd = str(workspace)
    else:
        record.host_cwd, record.executor = workspace, workspace_executor.executor_ref_from_ctx(ctx)
    output, sibling = workspace / "out.txt", workspace / "sibling.txt"
    sibling.write_bytes(b"first output")
    record.outputs.append(sibling.name)
    output.write_bytes(REPORT)
    declaration = declaration_path(ctx.drive_root, ctx.task_id, record.record_id)
    write = utils.atomic_write_json

    def unavailable(path, *args, **kwargs):
        if path == declaration:
            raise OSError("declaration unavailable")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", unavailable)
    # Exactly the start-time handoff after spawn: writing the declaration fails, the service retains its reservation.
    with kept.writer_reservation(ctx.drive_root, ctx.task_id, record.name, True) as writer:
        assert "kept_declaration_problem" in kept.declare_kept(ctx, record, writer)
    assert entries[record.service_id] is record and not record.finalizing

    def start():
        return workspace_executor.start_service(ctx, name=record.name, cmd=["writer"], host_cwd=workspace,
            cwd_root="active_workspace", readiness={}, outputs=[], before_outputs={})

    def successor(*args, **options):
        saved = registered_task_artifact(ctx.drive_root, ctx.task_id, output.name)
        assert Path(saved["path"]).read_bytes() == REPORT
        output.write_bytes(b"successor")
        return {"state": "running"}

    monkeypatch.setattr(workspace_executor, "_launch_service", successor)
    monkeypatch.setattr(shell, "_register_process_outputs", shell_outputs._register_process_outputs)
    copy = shell_outputs.copy_file_to_task_artifacts

    def partial_store(ctx, source, **options):
        if source == output:
            raise OSError("output store unavailable")
        return copy(ctx, source, **options)

    with monkeypatch.context() as failure:
        failure.setattr(shell_outputs, "copy_file_to_task_artifacts", partial_store)
        _, stopped, registered = services._stop_by_name(ctx, record.name)
        assert stopped["artifact_output_owed"] and "kept_declaration_problem" in stopped and registered
        assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, sibling.name)["path"]).read_bytes() == b"first output"
        assert entries[record.service_id] is record and not declaration.exists()
        failure.setattr(process_scope, "end_held_processes", lambda *a, **k: pytest.fail("stopped twice"))
        with pytest.raises(kept.KeptNamesake, match="uncaptured outputs"):
            start()
        # A contender with no access to the producer's memory must also leave the name alone.
        with failure.context() as foreign:
            foreign.delitem(entries, record.service_id)
            with pytest.raises(kept.KeptCaptureBusy):
                start()
        assert output.read_bytes() == REPORT
        if durable_retry:
            failure.setattr(utils, "atomic_write_json", write)
            with pytest.raises(kept.KeptNamesake, match="uncaptured outputs"):
                start()
            assert declaration.is_file() and entries[record.service_id] is record  # by-name retry still identifies it

    # A replacement's roots may differ. The retry still resolves the original output as its producer did.
    ctx.system_repo_dir = workspace
    monkeypatch.setattr(process_scope, "end_held_processes", lambda *a, **k: pytest.fail("stopped twice"))
    assert start()["state"] == "running"
    assert record.service_id not in entries and output.read_bytes() == b"successor"


def test_direct_restart_reports_missing_output_and_registers_existing_bytes_before_reuse(tmp_path, monkeypatch):
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    declaration = _ended_report(ctx, workspace)
    report = workspace / "report.txt"
    report.write_bytes(REPORT)

    def start(*_args, **_kwargs):
        assert not declaration.exists()
        saved = registered_task_artifact(ctx.drive_root, ctx.task_id, report.name)
        assert Path(saved["path"]).read_bytes() == REPORT
        report.write_bytes(b"successor")
        return {"state": "running"}

    monkeypatch.setattr(workspace_executor, "_launch_service", start)
    answer = workspace_executor.start_service(ctx, name="reporter", cmd=["writer"], host_cwd=workspace,
        cwd_root="active_workspace", readiness={}, outputs=[], before_outputs={})
    assert answer == {"state": "running", "previous_service": {"state": "stopped", "artifact_output_failed": True}}


def test_late_by_name_stop_cannot_capture_after_activity_consumed_the_launch(tmp_path):
    from types import SimpleNamespace

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    _ended_report(ctx, workspace, launch)
    report = workspace / "report.txt"
    report.write_bytes(REPORT)
    receipt = stop_kept_service(ctx.drive_root, launch, actor="owner:test", reason="collect")
    assert receipt["outputs"]["registered"] and receipt["outputs"]["state"] == "missing"
    record = SimpleNamespace(task_id=ctx.task_id, service_id=f"{ctx.task_id}:reporter", name="reporter",
        record_id=launch, drive_root=ctx.drive_root, outputs=["report.txt"], keep_alive=True, kept_declared=True)
    report.write_bytes(b"next writer")
    assert not kept.finish_own_outputs(ctx, record, {}, lambda _: pytest.fail("read a successor's output"))
    assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, report.name)["path"]).read_bytes() == REPORT


def test_by_name_stop_retry_still_reports_the_original_declared_files(tmp_path, monkeypatch):
    """The positive pair of the read failure: no current process on either backend, but a readable
    confirmed-stop declaration — the by-name stop captures what is still owed and reports the ended
    launch's own declared files (its saved outcome), consuming the declaration."""
    import hashlib

    registry = _registry(tmp_path, monkeypatch, task_id="producer")
    ctx = registry._ctx
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    workspace = tmp_path / "work"
    workspace.mkdir()
    launch = "launch-" + "b" * 32
    path = _confirmed_report(ctx, workspace, launch)
    other = _declare_in_outstanding(ctx.drive_root, ctx.task_id, "launch-" + "a" * 32, "{ unreadable")
    entries = kept._entries
    monkeypatch.setattr(kept, "_entries", lambda directory: sorted(entries(directory), key=lambda e: e.name))
    answer = json.loads(registry.execute("stop_service", {"name": "reporter"}))
    assert answer["output_retry"] is True and answer["lifecycle"] == "outputs_updated"
    [outcome] = answer["output_outcomes"]
    [artifact] = outcome["artifacts"]
    assert (outcome["index"], outcome["status"], outcome["registered"]) == (1, "saved", True)
    assert (artifact["name"], artifact["sha256"]) == ("report.txt", hashlib.sha256(REPORT).hexdigest())
    version = artifact["version_ref"]
    assert service_output_version_digest(version["path"]) == artifact["sha256"]
    assert (task_artifact_dir_path(ctx.drive_root, ctx.task_id) / version["path"]).read_bytes() == REPORT
    assert not path.exists() and other.exists()  # Only the identified declaration was consumed.


@pytest.mark.parametrize("unreadable", ["declaration", "directory"])
def test_by_name_stop_read_failure_is_not_a_stop_fact(tmp_path, monkeypatch, unreadable):
    """An unreadable saved input cannot prove a launch's name, exit or file outcome."""
    registry = _registry(tmp_path, monkeypatch, task_id="producer")
    ctx = registry._ctx
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    launch = "launch-" + "b" * 32
    if unreadable == "declaration":
        path = _declare_in_outstanding(ctx.drive_root, ctx.task_id, launch, "{ not the declaration")
        expected = {"name": "reporter", "detail": "KeptDeclarationUnreadable"}
    else:
        def unlistable(*_args):
            raise OSError("cannot list")
        monkeypatch.setattr(kept, "_entries", unlistable)
        path = None
        expected = {"name": "reporter", "detail": "OSError"}
    result = registry.execute_result("stop_service", {"name": "reporter"})
    assert (result.status, result.code) == ("unavailable", "LEGACY_UNAVAILABLE")
    assert result.meta.get("late_capture_unavailable") is True
    assert result.text.startswith("⚠️ SERVICE_CAPTURE_UNAVAILABLE")
    assert "SERVICE_STOP_ERROR" not in result.text and "stop_failed" not in result.text
    assert "Nothing was stopped or signalled" in result.text
    assert json.loads(result.text.split("\n\n", 1)[1]) == expected
    if unreadable == "declaration":
        assert path.exists()  # unreadable: the declaration is kept


def test_capture_before_reuse_reports_the_declaration_it_consumed(tmp_path):
    """A clean capture and no predecessor give different facts to the new start."""
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _confirmed_report(ctx, workspace, launch)
    assert kept.capture_before_reuse(ctx.drive_root, ctx.task_id, "reporter") == {}
    assert kept.capture_before_reuse(ctx.drive_root, ctx.task_id, "reporter") is None
    assert not path.exists()  # consumed by the capture


def test_the_retained_first_end_publishes_one_frame_to_the_producers_room(production_sink, monkeypatch):  # noqa: F811 -- the imported fixture, used by name
    """The first confirmed end of a retained launch speaks as a stop checkpoint of the producer's own
    room: one durable row and one live frame per publish (the log sink's copy of the row is suppressed,
    never a second frame), the supervisor's handler routes the frame by the producer's address, and a
    replay of the same stop keeps one card row (the same card_row_id). Synthetic: no product process runs."""
    from ouroboros.service_finalization import publish_retained_outputs
    from supervisor import events as ev
    from supervisor import worker_chat_lane

    data = production_sink.drive_root
    seed_result(data, "producer", {"status": "completed", "chat_id": 999001})
    production_sink.frames.clear()  # the fresh store's own first-touch import row is no frame of this publish
    observation = {"task_id": "producer", "launch_id": "launch-" + "a" * 32, "name": "reporter",
                   "lifecycle": "stopped", "stop_source": "retained", "state": "exited",
                   "process_cleanup": {"exit_confirmed": True}, "log_outcome": "not_attempted",
                   "output_outcomes": []}
    bus = queue.Queue()
    monkeypatch.setattr(worker_chat_lane, "_WORKER_BUS", None)  # this process's binding returns with the test
    worker_chat_lane.bind_worker_bus(bus)  # as ``worker_main`` names its bus
    publish_retained_outputs(data, observation)
    assert production_sink.frames == [], "the durable row's sink copy is suppressed: no second frame"
    envelope = bus.get_nowait()
    assert envelope["type"] == "log_event" and bus.empty(), "one publish queues exactly one frame"
    frame = envelope["data"]
    assert (frame["type"], frame["task_id"], frame["chat_id"]) == ("task_checkpoint", "producer", 999001), \
        "the frame is addressed to the producer's room, never the caller's"
    assert frame["checkpoint_kind"] == "services_stopped" and "Background service stopped: reporter" in frame["text"]
    ev._handle_log_event(envelope, _ctx(production_sink))
    [pushed] = production_sink.frames
    assert pushed["chat_id"] == 999001 and pushed["card_row_id"] == frame["card_row_id"]
    rows = [json.loads(line) for line in
            (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["type"] for row in rows] == ["task_checkpoint"]
    assert rows[0]["ts"] == frame["ts"] and rows[0]["card_row_id"] == frame["card_row_id"], "row and frame are one event"

    publish_retained_outputs(data, observation)  # a replay of the same stop
    replay = bus.get_nowait()["data"]
    assert replay["card_row_id"] == frame["card_row_id"] and replay["checkpoint_kind"] == "services_stopped"
    ev._handle_log_event({"type": "log_event", "data": replay}, _ctx(production_sink))
    assert len(production_sink.frames) == 2, "one frame per publish; the sink replayed neither row"
    assert {row["card_row_id"] for row in production_sink.frames} == {frame["card_row_id"]}, \
        "the card keeps a single stop row: no repeated stop"


def test_the_retained_frame_rides_the_servers_bridge_where_no_worker_bus_is_bound(production_sink, monkeypatch):  # noqa: F811 -- the imported fixture, used by name
    """In the server process (the owner's Activity Stop) no worker bus is bound: the same durable row and one
    frame, pushed on the bridge and addressed to the producer's room."""
    from ouroboros.service_finalization import publish_retained_outputs
    from supervisor import message_bus, worker_chat_lane

    data = production_sink.drive_root
    seed_result(data, "producer", {"status": "completed", "chat_id": 999001})
    production_sink.frames.clear()  # the fresh store's own first-touch import row is no frame of this publish
    monkeypatch.setattr(worker_chat_lane, "_WORKER_BUS", None)
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: production_sink.bridge)
    publish_retained_outputs(data, {
        "task_id": "producer", "launch_id": "launch-" + "a" * 32, "name": "reporter", "lifecycle": "outputs_updated",
        "stop_source": "retained", "output_retry": True, "log_outcome": "saved", "output_outcomes": [
            {"index": 1, "status": "saved", "artifacts": [{"name": "report.txt", "sha256": "a" * 64}]}]})
    [frame] = production_sink.frames
    assert (frame["type"], frame["task_id"], frame["chat_id"]) == ("task_checkpoint", "producer", 999001)
    assert frame["checkpoint_kind"] == "service_outputs_updated"
    assert frame["text"].startswith("Background service outputs updated: reporter") and "output 1 saved: report.txt" in frame["text"]
    [row] = [json.loads(line) for line in
             (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert (row["type"], row["ts"], row["card_row_id"]) == ("task_checkpoint", frame["ts"], frame["card_row_id"])


def _service_checkpoints(drive):
    """The checkpoint rows recorded in ``drive``'s events, in order: what the producer's room and Logs replay."""
    log = Path(drive) / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.exists() else []
    return [(row["checkpoint_kind"], row["text"]) for row in rows if row.get("type") == "task_checkpoint"]


def test_a_retained_launch_says_its_end_once_and_then_only_changed_files(tmp_path, monkeypatch):
    """The outside capture that first observes an ended kept launch (its own complete custody read, no stop
    snapshot yet) says the stop once, beside the file it could not save, and the declaration remembers it. The
    retry that saves the file says only that changed file result; a repeat finds nothing and says nothing."""
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _ended_report(ctx, workspace, launch, outputs=("report.txt",))
    seed_result(ctx.drive_root, "producer", {"status": "completed", "chat_id": 999001})  # its room, for the rows
    (workspace / "report.txt").write_bytes(REPORT)

    def refuse(*_args, **_kwargs):
        raise OSError("synthetic artifact store failure")

    with monkeypatch.context() as failed:
        failed.setattr(shell_outputs, "copy_file_to_task_artifacts", refuse)
        first = stop_kept_service(ctx.drive_root, launch, reason="collect", actor="owner:test")
    assert first["outcome"] == "not_found" and not first["ok"] and first["outputs"]["state"] == "incomplete", first
    assert first["outputs"]["observation"]["lifecycle"] == "stopped"
    remembered = kept.confirmed_output_stop(ctx.drive_root, ctx.task_id, launch)
    assert remembered["state"] == "exited" and remembered["process_cleanup"] == {"exit_confirmed": True}
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "services_stopped" and text.startswith("Background service stopped: reporter"), text

    second = stop_kept_service(ctx.drive_root, launch, reason="retry the capture", actor="owner:test")
    assert second["ok"] and second["outputs"]["state"] == "captured" and not path.exists(), second
    assert second["outputs"]["observation"]["lifecycle"] == "outputs_updated"
    assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, "report.txt")["path"]).read_bytes() == REPORT
    kinds = [kind for kind, _text in _service_checkpoints(ctx.drive_root)]
    assert kinds == ["services_stopped", "service_outputs_updated"], "never a second stop"
    assert "output 1 saved: report.txt" in _service_checkpoints(ctx.drive_root)[-1][1]

    assert "outputs" not in stop_kept_service(ctx.drive_root, launch, reason="repeat", actor="owner:test")
    assert [kind for kind, _text in _service_checkpoints(ctx.drive_root)] == kinds


def test_a_settled_capture_keeps_its_record_until_its_room_is_told_and_holds_no_drive(tmp_path, monkeypatch):
    """Every declared file is saved, but the room's row cannot be stored. The launch record stays with that exact row,
    holds no drive and is no listed debt; retention's own visit of the record says the same row again and only then
    retires it. No second stop, and the row is one row however often it was said."""
    from ouroboros import dialogue_store
    from ouroboros.tools.services import prune_service_logs

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _confirmed_report(ctx, workspace, launch)
    seed_result(ctx.drive_root, "producer", {"status": "completed", "chat_id": 999001})

    def refuse(*_args, **_kwargs):
        raise OSError("synthetic dialogue store failure")

    with monkeypatch.context() as failed:
        failed.setattr(dialogue_store, "append", refuse)
        first = capture_kept_service_outputs(ctx.drive_root, "producer", launch, actor="owner:test", reason="collect")
        assert first["state"] == "captured" and "not confirmed told" in first["detail"], first
        record = json.loads(path.read_text(encoding="utf-8"))
        owed = record["unspoken"]
        assert owed["observation"]["row_revision"] == 2 and record["stop"]["output_outcomes"][0]["status"] == "saved"
        assert Path(registered_task_artifact(ctx.drive_root, "producer", "report.txt")["path"]).read_bytes() == REPORT
        assert _service_checkpoints(ctx.drive_root) == [], "its room was told nothing yet"
        assert not owed_output_hold(ctx.drive_root, workspace), "saved files hold no drive for a row"
        assert _kept_view(ctx.drive_root)["services"] == [], "nothing is left to capture"
        again = prune_service_logs(ctx.drive_root, retention_days=0, now=time.time() + 1)
        assert path.exists() and "capture_owed" not in again, "still untold: the record stays, and it is no file debt"
        assert json.loads(path.read_text(encoding="utf-8"))["unspoken"] == owed

    report = prune_service_logs(ctx.drive_root, retention_days=0, now=time.time() + 1)  # the visit that already reads it
    assert not path.exists() and not report["errors"], report
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "service_outputs_updated" and text.startswith("Background service outputs updated: reporter"), text
    assert text.endswith(" · ended · log saved · output 1 saved: report.txt"), "the launch's whole record, one row"
    rows = [json.loads(line) for line in (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [told] = [row for row in rows if row.get("type") == "task_checkpoint"]
    assert (told["ts"], told["chat_id"], told["card_row_revision"]) == (owed["ts"], 999001, 2)
    assert dialogue_store.append(ctx.drive_root, "progress", told, idempotent=True).duplicate is True
    assert [row["state"] for row in rows if row.get("type") == "kept_service_outputs"] == ["captured"], "one capture"
    assert "outputs" not in stop_kept_service(ctx.drive_root, launch, reason="repeat", actor="owner:test")


def test_the_retained_row_door_answers_what_it_stored(tmp_path, monkeypatch):
    """The door a launch record depends on says what happened: no room at all, a row it could not store, a stored
    row. Saying the stored row again with its own time adds no second row to the room."""
    from ouroboros import dialogue_store
    from ouroboros.service_finalization import publish_retained_outputs

    from ouroboros import task_results

    data = tmp_path / "data"
    data.mkdir()
    seed_result(data, "producer", {"status": "completed", "chat_id": 999001})
    observation = {"task_id": "producer", "launch_id": "launch-" + "a" * 32, "name": "reporter", "lifecycle": "stopped",
                   "stop_source": "retained", "state": "exited", "process_cleanup": {"exit_confirmed": True},
                   "log_outcome": "not_attempted", "output_outcomes": []}
    stamp = "2026-11-20T10:00:00+00:00"
    # No room is a proven absence: the store was read whole and holds no such producer, and no project binds it.
    assert publish_retained_outputs(data, {**observation, "task_id": "never-a-task"}, ts=stamp) == "no_room"

    def unreadable(*_args, **_kwargs):
        raise OSError("synthetic task store failure")

    with monkeypatch.context() as blind:  # a producer that cannot be read is unknown, never "no room"
        blind.setattr(task_results, "load_task_result", unreadable)
        assert publish_retained_outputs(data, {**observation, "task_id": "never-a-task"}, ts=stamp) == "unknown"

    def refuse(*_args, **_kwargs):
        raise OSError("synthetic dialogue store failure")

    with monkeypatch.context() as failed:
        failed.setattr(dialogue_store, "append", refuse)
        assert publish_retained_outputs(data, observation, ts=stamp) == "unknown"
    assert _service_checkpoints(data) == [], "neither the room nor Logs holds an unconfirmed row"

    assert publish_retained_outputs(data, observation, ts=stamp) == "recorded"
    [(kind, text)] = _service_checkpoints(data)
    assert kind == "services_stopped" and text.startswith("Background service stopped: reporter")
    [row] = [json.loads(line) for line in (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (row["ts"], row["chat_id"], row["card_row_revision"]) == (stamp, 999001, 1)
    assert publish_retained_outputs(data, observation, ts=stamp) == "recorded"  # said again: the same row
    assert dialogue_store.append(data, "progress", row, idempotent=True).duplicate is True


@pytest.mark.parametrize("failure", ["logs", "generation_close"])
def test_retained_checkpoint_delivers_the_durable_card_row_before_logs_repair(tmp_path, monkeypatch, failure):
    from ouroboros import dialogue_store, service_finalization, task_custody, utils

    data = tmp_path / "data"
    data.mkdir()
    seed_result(data, "producer", {"status": "completed", "chat_id": 999001})
    observation = {"task_id": "producer", "launch_id": "launch-" + "a" * 32, "name": "reporter",
                   "lifecycle": "stopped", "state": "exited", "log_outcome": "not_attempted"}
    ts = "2026-11-20T10:00:00+00:00"
    frames, appends = [], []
    monkeypatch.setattr(service_finalization, "_push_retained_frame", lambda root, frame: frames.append(frame))
    with monkeypatch.context() as inject:
        closed = False
        append = dialogue_store.append

        def room_landed(*args, **kwargs):
            nonlocal closed
            receipt = append(*args, **kwargs)
            closed = failure == "generation_close"
            return receipt

        inject.setattr(dialogue_store, "append", room_landed)
        append_log = utils.append_jsonl
        inject.setattr(utils, "append_jsonl", lambda path, row, **kw: (appends.append(row) or False)
                       if row.get("card_row_id") else append_log(path, row, **kw))
        with task_custody.publication_fence(lambda: closed):
            assert service_finalization.publish_retained_outputs(data, observation, ts=ts) == "unknown"
    assert len(appends) == (failure == "logs")
    [frame] = frames
    assert frame["chat_id"] == 999001 and frame["card_row_revision"] == 1
    assert service_finalization.publish_retained_outputs(data, observation, ts=ts) == "recorded"
    assert frames == [frame] and len(_service_checkpoints(data)) == 1


@pytest.mark.parametrize("sink", ["dialogue", "logs"])
def test_retained_checkpoint_rechecks_the_fence_after_the_sink_lock(tmp_path, monkeypatch, sink):
    from contextlib import contextmanager

    from ouroboros import dialogue_store, platform_layer, task_custody, utils

    data = tmp_path / "data"
    data.mkdir()
    task, launch = "producer", "launch-" + "a" * 32
    seed_result(data, task, {"status": "completed", "chat_id": 999001})
    checkpoint = {"task_id": task, "launch_id": launch, "name": "reporter", "lifecycle": "stopped",
                  "state": "exited", "log_outcome": "not_attempted", "output_outcomes": []}
    declaration = {**checkpoint, "outputs": [], "context": {"drive_root": str(data)}, "stop": checkpoint,
                   "unspoken": {"ts": "2026-11-20T10:00:00+00:00", "observation": checkpoint}}
    sidecar = declaration_path(data, task, launch, owed=False)
    sidecar.parent.mkdir(parents=True)
    original = json.dumps(declaration)
    sidecar.write_text(original, encoding="utf-8")
    logs = data / "logs" / "events.jsonl"
    closed = False
    with monkeypatch.context() as locked:
        if sink == "dialogue":
            transaction = dialogue_store.transaction

            @contextmanager
            def admit(root, *, write=False, **kwargs):
                nonlocal closed
                with transaction(root, write=write, **kwargs) as connection:
                    closed |= write
                    yield connection

            locked.setattr(dialogue_store, "transaction", admit)
        else:
            acquire = platform_layer.acquire_exclusive_file_lock

            def admit(path, **kwargs):
                nonlocal closed
                result = acquire(path, **kwargs)
                if path == utils.jsonl_append_lock_path(logs):
                    closed = True
                return result

            locked.setattr(platform_layer, "acquire_exclusive_file_lock", admit)
        with task_custody.publication_fence(lambda: closed):
            assert kept.say_unspoken(sidecar, declaration, late=True) == declaration
    assert closed and sidecar.read_text(encoding="utf-8") == original
    if sink == "dialogue":
        assert not [item for item in dialogue_store.rows_by_identity(data, 999001, declaration["unspoken"]["ts"],
                                                                    stream="progress") if item.row.get("card_row_id")]
    assert not logs.exists() or not any(row.get("card_row_id") for row in map(json.loads, logs.read_text().splitlines()))
    assert kept.say_unspoken(sidecar, declaration, late=True) is None and not sidecar.exists()
    assert len(_service_checkpoints(data)) == 1


@pytest.mark.parametrize("gap", ["unknown_target", "no_group"])
def test_an_end_inspection_that_cannot_be_made_creates_no_exit(tmp_path, monkeypatch, gap):
    """Asking to verify observes nothing. A kept Docker launch custody does not list (its record was never
    published) whose capture target cannot be identified, or names no group to probe, is not said to have ended:
    no stop row, and no stop snapshot a later capture would trust. Once its group answers the probe on its
    recorded identity, the same capture says the end, once, even while its files stay unreadable."""
    from ouroboros import workspace_executor_paths as paths

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch, container = "service-" + "c" * 32, "d" * 64
    docker = {"type": "docker_exec", "id": "lab", "network": "host", "container_name": container,
              "path_mappings": [{"host_path": str(workspace), "backend_path": "/work"}]}

    def declared(executor_ref, **facts):
        return {"schema": 2, "launch_id": launch, "task_id": ctx.task_id, "name": "reporter", "keep_alive": True,
                "cwd": str(workspace), "cwd_base": str(workspace), "cwd_root": "active_workspace",
                "outputs": ["report.txt"], "before_outputs": {}, "custody_drive": str(ctx.drive_root),
                "context": {"repo_dir": str(ctx.repo_dir), "workspace_root": str(workspace),
                            "workspace_mode": "external", "drive_root": str(ctx.drive_root),
                            "tool_profile": "advanced", "executor_ref": executor_ref}, **facts}

    path = _declare_in_outstanding(ctx.drive_root, ctx.task_id, launch,
                                   declared({} if gap == "unknown_target" else docker))
    recorded = path.read_bytes()
    seed_result(ctx.drive_root, "producer", {"status": "completed", "chat_id": 999001})  # its room, for the rows

    def unreadable(_argv):
        raise paths.OutputResolutionUnavailable("no container answers in this test")

    monkeypatch.setattr(paths, "_docker_output_probe", unreadable)
    monkeypatch.setattr(workspace_executor, "_docker_pid_state", lambda *_: pytest.fail("no group was named to probe"))
    receipt = stop_kept_service(ctx.drive_root, launch, reason="collect", actor="owner:test")
    assert receipt["outcome"] == "unknown" and not receipt["ok"] and receipt["outputs"]["state"] == "not_attempted", receipt
    assert "observation" not in receipt["outputs"]
    with pytest.raises(kept.KeptNamesake):
        kept.capture_before_reuse(ctx.drive_root, ctx.task_id, "reporter")
    assert path.read_bytes() == recorded, "no stop snapshot was written"
    assert kept.confirmed_output_stop(ctx.drive_root, ctx.task_id, launch) is None
    assert _service_checkpoints(ctx.drive_root) == [], "nothing was said in the producer's room"

    path.write_text(json.dumps(declared(docker, backend_pid="42")), encoding="utf-8")
    probed = []
    monkeypatch.setattr(workspace_executor, "_docker_pid_state",
                        lambda name, pid: probed.append((name, pid)) or "exited")
    answered = stop_kept_service(ctx.drive_root, launch, reason="its group answered", actor="owner:test")
    assert probed == [(container, "42")], "probed once, on the launch's recorded identity"
    assert answered["outputs"]["state"] == "incomplete" and answered["outputs"]["observation"]["lifecycle"] == "stopped"
    assert kept.confirmed_output_stop(ctx.drive_root, ctx.task_id, launch)["process_cleanup"] == {"exit_confirmed": True}
    assert [kind for kind, _text in _service_checkpoints(ctx.drive_root)] == ["services_stopped"]

    again = stop_kept_service(ctx.drive_root, launch, reason="still unreadable", actor="owner:test")
    assert again["outputs"]["state"] == "incomplete" and probed == [(container, "42")]
    assert [kind for kind, _text in _service_checkpoints(ctx.drive_root)] == ["services_stopped"], \
        "the end is remembered: no second stop, and unchanged file facts are no news"


def test_a_stop_whose_capture_cannot_confirm_the_end_is_not_reported_stopped(tmp_path, monkeypatch):
    """The stop's own read after the signal found nothing, but the capture's identity-checked read (the launch's
    Docker group) did not confirm that end: the receipt says ``unknown`` with the capture's reason, never ``stopped``.
    Synthetic: the custody reads, the signal and the capture are scripted, no process runs."""
    from ouroboros import launch_custody, owned_shutdown

    ctx, _workspace = _executor_ctx(tmp_path, task_id="producer")
    ctx.drive_root.mkdir(parents=True, exist_ok=True)
    launch = "launch-" + "e" * 32
    listed = ({"launch_id": launch, "name": "reporter", "task_id": "producer", "state": "running", "log_path": ""},
              [{"record_id": launch, "drive_root": str(ctx.drive_root)}])
    reads = iter([([listed], True, ""), ([listed], True, ""), ([], True, "")])  # found, re-selected, gone after the stop
    signalled = []
    monkeypatch.setattr(launch_custody, "_kept_selection", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(owned_shutdown, "_stamp", lambda *_a, **_k: None)
    monkeypatch.setattr(owned_shutdown, "_stop_records", lambda _root, entries, _deadline: signalled.append(entries))
    monkeypatch.setattr(kept, "capture_kept_service_outputs", lambda *_a, **_k: {
        "task_id": "producer", "state": "not_attempted", "detail": "the original Docker group is not confirmed ended"})
    receipt = stop_kept_service(ctx.drive_root, launch, reason="stop it", actor="owner:test")
    assert len(signalled) == 1 and receipt["outputs"]["state"] == "not_attempted"
    assert receipt["outcome"] == "unknown" and not receipt["ok"], receipt
    assert receipt["detail"] == "the original Docker group is not confirmed ended"


@pytest.mark.parametrize("door", ["registry", "executor"])
def test_replacement_waits_for_activity_capture_before_the_new_writer(tmp_path, monkeypatch, door):
    registry = _registry(tmp_path, monkeypatch, task_id="producer")
    ctx = registry._ctx
    workspace = tmp_path / "work"
    workspace.mkdir()
    ctx.workspace_root, ctx.workspace_mode = workspace, "external"
    launch = "launch-" + "b" * 32
    path = _ended_report(ctx, workspace, launch)
    report = workspace / "report.txt"
    report.write_bytes(REPORT)
    entered, release, contending, spawned = (threading.Event() for _ in range(4))
    receipts = {}
    real_copy = shell_outputs.copy_file_to_task_artifacts

    def slow_copy(ctx, source, **options):
        if source == report:
            entered.set()
            assert release.wait(5)
        return real_copy(ctx, source, **options)

    def launch_new(*args, **options):
        assert not path.exists()
        report.write_bytes(b"new writer")
        spawned.set()
        return {"state": "running"}

    def restart():
        if door == "registry":
            receipts["new"] = registry.execute_result("start_service", {
                "cmd": ["writer"], "name": "reporter", "cwd": str(workspace)})
        else:
            receipts["new"] = workspace_executor.start_service(ctx, name="reporter", cmd=["writer"], host_cwd=workspace,
                cwd_root="active_workspace", readiness={}, outputs=[], before_outputs={})

    real_lock = services.service_name_hold

    def reserve(*args):
        contending.set()
        return real_lock(*args)

    monkeypatch.setattr(services, "service_name_hold", reserve)
    monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", slow_copy)
    monkeypatch.setattr(workspace_executor, "_launch_service", launch_new)
    monkeypatch.setattr(services, "_executor_can_run_cwd", lambda *_: True)
    outside = threading.Thread(target=lambda: receipts.update(old=stop_kept_service(
        ctx.drive_root, launch, reason="collect", actor="owner:test")))
    contender = threading.Thread(target=restart)
    outside.start()
    try:
        assert entered.wait(5)
        contender.start()
        assert contending.wait(5) and not spawned.is_set() and report.read_bytes() == REPORT
    finally:
        release.set()
        outside.join(10)
        if contender.ident is not None:
            contender.join(10)
    assert receipts["old"]["outputs"]["state"] == "missing" and spawned.is_set()
    assert not outside.is_alive() and not contender.is_alive()
    assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, report.name)["path"]).read_bytes() == REPORT
    assert report.read_bytes() == b"new writer"
    assert "outputs" not in stop_kept_service(ctx.drive_root, launch, reason="repeat", actor="owner:test")


@pytest.mark.parametrize("door", ["tool", "executor"])
def test_executor_start_releases_name_after_registration_before_readiness(tmp_path, monkeypatch, door):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from ouroboros.service_outcomes import launch_facts

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    ctx.repo_dir.mkdir(exist_ok=True)
    events = []
    register = workspace_executor._register_service_process

    @contextmanager
    def hold(drive, task, name):
        assert (Path(drive).resolve(), task, name) == (ctx.drive_root.resolve(), ctx.task_id, "reporter")
        with kept.service_name_hold(drive, task, name):
            events.append("hold")
            yield
        events.append("released")

    def spawn(_cmd, *, on_spawn, **_options):
        events.append("spawn")
        proc = SimpleNamespace(pid=4242, poll=lambda: None)
        on_spawn(proc)
        return proc

    def register_launch(*args):
        events.append("register")
        return register(*args)

    def readiness(record, _probe):
        events.append("readiness")
        assert record.durable_record_path.is_file()
        launch = launch_facts(record)["launch_id"]
        assert declaration_path(ctx.drive_root, ctx.task_id, launch, owed=False).is_file()

    monkeypatch.setattr(services, "service_name_hold", hold)
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_process_command_sha256", lambda _pid: "")
    monkeypatch.setattr(workspace_executor, "_register_service_process", register_launch)
    monkeypatch.setattr(workspace_executor, "_wait_readiness", readiness)
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    params = dict(name="reporter", cmd=["writer"], keep_alive=True, readiness={})
    if door == "tool":
        started = json.loads(services._start_service(ctx, cwd=str(workspace), **params))
    else:
        started = workspace_executor.start_service(ctx, host_cwd=workspace, cwd_root="active_workspace",
                                                   outputs=[], before_outputs={}, **params)
    assert started["state"] == "running"
    assert events == ["hold", "spawn", "register", "released", "readiness"]


def test_namesake_guard_reads_only_its_producer_and_probes_its_launch(tmp_path, monkeypatch):
    from ouroboros import launch_custody

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    unrelated = _declare_in_outstanding(ctx.drive_root, "another-task", "launch-" + "c" * 32, "incomplete")
    _ended_report(ctx, workspace, launch)
    selected = []

    def selection(_drive, *, launch_id, task_id):
        selected.append((task_id, launch_id))
        return [({"state": "unknown", "launch_id": launch_id}, [])], True, ""
    monkeypatch.setattr(launch_custody, "_kept_selection", selection)
    monkeypatch.setattr(kept, "_outstanding_files", lambda *_: pytest.fail("scanned all declarations"))
    with pytest.raises(kept.KeptNamesake) as held:
        kept.capture_before_reuse(ctx.drive_root, ctx.task_id, "reporter")
    assert held.value.state == "unknown" and selected == [(ctx.task_id, launch)] and unrelated.is_file()


def test_a_by_name_capture_leaves_its_row_in_the_record_for_the_callers_pass(tmp_path, monkeypatch):
    """No current process on either backend, a readable confirmed-stop declaration, and a caller whose completion pass
    will tell what the capture finds: the row waits in the launch record under the stop's own name for it, never in
    memory alone. The pass says it with the task's own writer and settles the record. (With no pass to tell it the
    same call consumes the declaration: ``test_by_name_stop_retry_still_reports_the_original_declared_files``.)"""
    from types import SimpleNamespace

    from ouroboros import service_finalization, service_outcomes

    registry = _registry(tmp_path, monkeypatch, task_id="producer")
    ctx = registry._ctx
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    workspace = tmp_path / "work"
    workspace.mkdir()
    launch = "launch-" + "b" * 32
    path = _confirmed_report(ctx, workspace, launch)
    seed_result(ctx.drive_root, "another-task", {"status": "completed"})  # the task store exists and names no room for ours
    ctx._execution_trace = {}  # a tool call of a loop: its completion pass collects this stop

    _backend, stopped, registered = services._stop_by_name(ctx, "reporter")
    assert registered and stopped["lifecycle"] == "outputs_updated" and stopped["output_outcomes"][0]["status"] == "saved"
    owed = stopped[service_outcomes.ROW_OWED]
    held = json.loads(path.read_text(encoding="utf-8"))
    assert (held["unspoken"]["ts"], held["unspoken"]["pass"], owed["drive"]) == (owed["ts"], True, str(ctx.drive_root))
    assert held["stop"]["output_outcomes"][0]["status"] == "saved" and held["stop"]["row_revision"] == 2
    assert held["unspoken"]["observation"]["row_revision"] == 2 and not owed_output_hold(ctx.drive_root, workspace)
    assert ctx._execution_trace["service_stop_observations"] == [stopped], "handed to the pass, and kept on disk for it"
    again = capture_kept_service_outputs(ctx.drive_root, "producer", launch, actor="owner:test", reason="collect")
    assert again["only_told"] and "not confirmed told" in again["detail"] and path.exists(), "the pass's row is the pass's"

    completion = SimpleNamespace(drive_root=ctx.drive_root, task_id=ctx.task_id, tools=SimpleNamespace(_ctx=ctx),
                                 drive_logs=ctx.drive_root / "logs", event_queue=None, llm_trace=ctx._execution_trace)
    assert service_finalization.finalize(completion, completion.llm_trace)
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "service_outputs_updated" and " · requested stop" not in text and "output 1 saved: report.txt" in text
    assert not path.exists(), "no room is owed this row: the record is settled"
    assert not service_finalization.finalize(completion, completion.llm_trace)


def test_a_process_record_that_cannot_be_removed_keeps_the_name_and_the_end_is_told_all_the_same(tmp_path, monkeypatch):
    """The capture proved the end and took every file, but the ended launch's old process record cannot be removed.
    That is the record's fate alone: the end and the file answers are remembered and told once, and the declaration
    keeps the name (a successor could reuse that group's PID) until a later capture removes the record and says nothing."""
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _ended_report(ctx, workspace, launch, outputs=("report.txt",))
    seed_result(ctx.drive_root, "producer", {"status": "completed", "chat_id": 999001})  # its room, for the rows
    (workspace / "report.txt").write_bytes(REPORT)
    stale = tmp_path / "stale-process-record.json"
    stale.write_text("{}", encoding="utf-8")
    target = kept._capture_executor
    monkeypatch.setattr(kept, "_capture_executor", lambda *args: (target(*args)[0], stale, True))
    monkeypatch.setattr(workspace_executor, "_forget_process", lambda _record: False)

    first = capture_kept_service_outputs(ctx.drive_root, "producer", launch, actor="owner:test", reason="collect",
                                         verify_exit=True)
    assert first["state"] == "incomplete" and "still holds its name" in first["detail"], first
    assert first["observation"]["lifecycle"] == "stopped" and not first["observation"]["artifact_output_owed"]
    held = json.loads(path.read_text(encoding="utf-8"))
    assert held["name_held"] is True and "unspoken" not in held, "told, and kept only for the name"
    assert held["stop"]["process_cleanup"] == {"exit_confirmed": True}
    assert [row["status"] for row in held["stop"]["output_outcomes"]] == ["saved"]
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "services_stopped" and text.endswith("output 1 saved: report.txt"), text
    assert owed_output_hold(ctx.drive_root, workspace) == "kept_service_capture_owed", "a held name is not settled"
    with pytest.raises(kept.KeptNamesake) as refused:
        kept.capture_before_reuse(ctx.drive_root, "producer", "reporter")
    assert refused.value.state == "owed" and path.exists() and stale.exists()

    monkeypatch.setattr(workspace_executor, "_forget_process", lambda record: record.unlink() is None)
    assert kept.capture_before_reuse(ctx.drive_root, "producer", "reporter") == {}
    assert not path.exists() and not stale.exists()
    assert [kind for kind, _text in _service_checkpoints(ctx.drive_root)] == ["services_stopped"], "nothing new was said"
    assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, "report.txt")["path"]).read_bytes() == REPORT


@pytest.mark.parametrize("spawned", ["confirmed", "nothing", "unknown"])
def test_a_kept_launch_is_declared_before_its_spawn_and_only_a_confirmed_start_stops_being_unconfirmed(
        tmp_path, monkeypatch, spawned):
    """The declaration exists, under the launch id its marker already names and with the producer's frozen context,
    before any process does. A start that spawned nothing takes it back as its admission ends; one that could not
    finish after a process existed keeps it unconfirmed; only a confirmed start is written without that mark."""
    from types import SimpleNamespace

    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    ctx.repo_dir.mkdir(exist_ok=True)
    folder = ctx.drive_root / "services" / ctx.task_id
    seen = {}

    def spawn(_cmd, *, on_spawn, scope_marker, **_options):
        [path] = folder.glob("*" + kept.KEPT_DECLARATION)
        seen.update(path=path, before=json.loads(path.read_text(encoding="utf-8")),
                    launch="launch-" + scope_marker.removeprefix(LAUNCH_SCOPE_PREFIX))
        if spawned == "nothing":
            raise OSError("spawn refused")
        proc = SimpleNamespace(pid=4242, poll=lambda: None)
        on_spawn(proc)
        if spawned == "unknown":
            raise RuntimeError("spawned process could not enter durable custody")
        return proc

    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_process_command_sha256", lambda _pid: "")
    monkeypatch.setattr(workspace_executor, "_wait_readiness", lambda *_a: None)
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)

    def start():
        return workspace_executor.start_service(ctx, host_cwd=workspace, cwd_root="active_workspace", outputs=[],
                                                before_outputs={}, name="reporter", cmd=["writer"], keep_alive=True,
                                                readiness={})

    if spawned == "confirmed":
        assert start()["launch_id"] == seen["launch"]
    else:
        with pytest.raises(OSError if spawned == "nothing" else RuntimeError):
            start()
    before = seen["before"]
    assert (before["start"], before["launch_id"], before["name"], before["keep_alive"]) == (
        "unconfirmed", seen["launch"], "reporter", True)
    assert before["context"]["drive_root"] == str(ctx.drive_root), "the producer's context, before any process of it"
    if spawned == "nothing":
        assert not seen["path"].exists(), "nothing was spawned: the declaration names no launch"
    else:
        after = json.loads(seen["path"].read_text(encoding="utf-8"))
        assert ("start" in after) is (spawned == "unknown") and after["launch_id"] == seen["launch"]


def test_ordinary_launch_outputs_are_prepared_and_recovered_after_worker_loss(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import owned_shutdown, process_scope, utils
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX
    from ouroboros.tools import shell

    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    seed_result(ctx.drive_root, ctx.task_id, {"status": "running", "chat_id": 6})
    seen = {}
    output = workspace / "report.txt"

    def spawn(_cmd, *, on_spawn, scope_marker, **_options):
        launch = "launch-" + scope_marker.removeprefix(LAUNCH_SCOPE_PREFIX)
        path = declaration_path(ctx.drive_root, ctx.task_id, launch)
        prepared = json.loads(path.read_text())
        assert (prepared["launch_id"], prepared["start"], prepared["keep_alive"]) == (launch, "unconfirmed", False)
        assert prepared["outputs"] == [output.name] and prepared["before_outputs"] == {output.name: [False, 0, ""]}
        assert prepared["context"]["drive_root"] == str(ctx.drive_root)
        seen.update(launch=launch, path=path, marker=scope_marker)
        proc = SimpleNamespace(pid=4242, returncode=0, poll=lambda: 0)
        on_spawn(proc)
        return proc

    def start():
        return workspace_executor.start_service(ctx, name="reporter", cmd=["synthetic"], host_cwd=workspace,
            cwd_root="active_workspace", readiness={}, outputs=[output.name], before_outputs={output.name: (False, 0, "")})

    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_wait_readiness", lambda *_a: None)
    monkeypatch.setattr(workspace_executor, "_process_command_sha256", lambda _pid: "")
    monkeypatch.setattr(process_custody, "spawn_supervised", spawn)
    with monkeypatch.context() as refused:
        write = utils.atomic_write_json

        def refuse_declaration(path, value, **opts):
            if path.name.endswith(kept.KEPT_DECLARATION):
                raise OSError("capture context could not be saved")
            return write(path, value, **opts)

        refused.setattr(utils, "atomic_write_json", refuse_declaration)
        with pytest.raises(OSError, match="declaration could not be saved") as error:
            start()
        assert error.value.process_not_started and not seen

    assert start()["launch_id"] == seen["launch"]
    declaration = seen["path"]
    assert "start" not in json.loads(declaration.read_text())
    with monkeypatch.context() as live:
        from ouroboros import task_status

        live.setattr(task_status, "task_execution_observation", lambda *_a: {"state": "active", "status": "running"})
        assert _kept_view(ctx.drive_root)["services"] == []
        assert stop_kept_service(ctx.drive_root, seen["launch"], reason="recover", actor="owner:test")["outcome"] == "refused"
        assert declaration.is_file() and not registered_task_artifact(ctx.drive_root, ctx.task_id, output.name)
    output.write_bytes(REPORT)
    workspace_executor._SERVICES.clear()  # worker lost before its ordinary Stop; only durable custody survives
    write_task_result(ctx.drive_root, ctx.task_id, "failed")
    assert owed_output_hold(ctx.drive_root, workspace) == "kept_service_capture_owed"
    assert [(row["launch_id"], row["state"]) for row in _kept_view(ctx.drive_root)["services"]] == [(seen["launch"], "unknown")]
    monkeypatch.setattr(owned_shutdown, "_stop_records", lambda *_a: pytest.fail("recovery signalled a process"))
    monkeypatch.setattr(services, "_working_tree_changes", lambda _cwd: set())
    monkeypatch.setattr(shell, "_register_process_outputs", shell_outputs._register_process_outputs)
    monkeypatch.setattr(process_custody, "IS_WINDOWS", False)
    state = "unknown"

    def observe(targets, **_opts):
        [target] = targets
        assert target["key"] == seen["launch"] and target["markers"] == [seen["marker"]]
        return {target["key"]: state}

    monkeypatch.setattr(process_scope, "observe_owned_processes", observe)
    unknown = stop_kept_service(ctx.drive_root, seen["launch"], reason="recover", actor="owner:test")
    assert unknown["outcome"] == "unknown" and declaration.is_file()
    assert not registered_task_artifact(ctx.drive_root, ctx.task_id, output.name)
    state = "settled"
    recovered = stop_kept_service(ctx.drive_root, seen["launch"], reason="recover", actor="owner:test")
    assert recovered["ok"] and recovered["outputs"]["state"] == "captured", recovered
    assert Path(registered_task_artifact(ctx.drive_root, ctx.task_id, output.name)["path"]).read_bytes() == REPORT
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "services_stopped" and text.endswith("output 1 saved: report.txt")
    assert not declaration.exists() and not owed_output_hold(ctx.drive_root, workspace)


def test_a_start_nobody_confirmed_is_never_read_as_an_end(tmp_path):
    """No process of the launch is found, but its start was never confirmed: that proves no process ran and ended.
    The inventory says ``unknown``, never ``capture_owed``; the capture takes the declared file and tells only that
    file result, with no stop row and no end in the record's words."""
    ctx, workspace = _executor_ctx(tmp_path, task_id="producer")
    launch = "launch-" + "b" * 32
    path = _ended_report(ctx, workspace, launch, outputs=("report.txt",))
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "start": "unconfirmed"}), encoding="utf-8")
    seed_result(ctx.drive_root, "producer", {"status": "completed", "chat_id": 999001})  # its room, for the rows
    (workspace / "report.txt").write_bytes(REPORT)
    assert [(row["launch_id"], row["state"]) for row in _kept_view(ctx.drive_root)["services"]] == [(launch, "unknown")]

    answer = capture_kept_service_outputs(ctx.drive_root, "producer", launch, actor="owner:test", reason="collect",
                                          verify_exit=True)
    assert answer["state"] == "captured" and answer["registered"] and not path.exists(), answer
    assert answer["observation"]["lifecycle"] == "outputs_updated" and "state" not in answer["observation"]
    [(kind, text)] = _service_checkpoints(ctx.drive_root)
    assert kind == "service_outputs_updated" and text.endswith("output 1 saved: report.txt"), text
    assert "ended" not in text and "stopped" not in text, "a file result only: no end was proven"
