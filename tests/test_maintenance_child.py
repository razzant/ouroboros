"""The maintenance child's named jobs: the rest law, durable resume, and no history walk in the server.

Each job's second run over unchanged inputs opens no history file, decodes no task body and
commits nothing; a run killed after k items resumes at k from its durable row; a fresh server
process (a new dispatcher object, no module memory) finds every job where the rows left it.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import pathlib

import pytest

from ouroboros import maintenance, maintenance_jobs, store
from ouroboros import projects_registry as pr


class _Killed(BaseException):
    """A hard stop between two items: no handler of the job runs (a SIGKILL's shape)."""


def _write_rows(path, chat_id, *, count=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"chat_id": chat_id, "text": f"row {index}"}) + "\n")


def _archive(data, segments, chat_id=7):
    """SEGMENTS chat and progress segments of unrelated chats (none mentions a candidate)."""
    for index in range(segments):
        _write_rows(data / "archive" / f"chat_20260101T{index:06d}.jsonl", chat_id)
        _write_rows(data / "archive" / f"progress_20260101T{index:06d}.jsonl", chat_id)


@pytest.fixture
def opened(monkeypatch):
    """Every path opened through ``open``/``io.open``/``os.open`` while the fixture is active."""
    seen = []
    real_open, real_io, real_os = builtins.open, io.open, os.open

    def record(path):
        try:
            seen.append(pathlib.Path(os.fsdecode(path)).resolve(strict=False))
        except TypeError:
            pass  # a file descriptor

    def py_open(file, *args, **kwargs):
        record(file)
        return real_open(file, *args, **kwargs)

    def io_open(file, *args, **kwargs):
        record(file)
        return real_io(file, *args, **kwargs)

    def os_open(path, *args, **kwargs):
        record(path)
        return real_os(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", py_open)
    monkeypatch.setattr(io, "open", io_open)
    monkeypatch.setattr(os, "open", os_open)
    return seen


def _under(paths, directory):
    directory = pathlib.Path(directory).resolve()
    return [path for path in paths if path.is_relative_to(directory)]


def _decodes(monkeypatch):
    """Task-body reads (single documents and batched fetches)."""
    from ouroboros import task_store

    reads = []
    body_text, decode = store.Txn.body_text, task_store._decode

    def text(txn, task_id):
        reads.append(task_id)
        return body_text(txn, task_id)

    def batch(rows, handoffs):
        reads.extend(row["task_id"] for row in rows)
        return decode(rows, handoffs)

    monkeypatch.setattr(store.Txn, "body_text", text)
    monkeypatch.setattr(task_store, "_decode", batch)
    return reads


def _revision(data):
    with store.open_store(data, write=False) as txn:
        return txn.revision()


def _statistics(data):
    """The planner tables as stored: the ``sqlite_stat1`` rows and the number of ``sqlite_stat4`` samples."""
    with store.open_store(data, write=False) as txn:
        stat1 = sorted(tuple(row) for row in txn.conn.execute("SELECT tbl, idx, stat FROM sqlite_stat1"))
        stat4 = txn.conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_stat4'").fetchone()
        return stat1, txn.conn.execute("SELECT COUNT(*) FROM sqlite_stat4").fetchone()[0] if stat4 else 0


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return root


def test_reconcile_and_room_reads_open_no_archive_after_dialogue_import(data, opened):
    from tests._dialogue import import_all

    pr.create_project(data, "never-active", origin="reconcile")
    _archive(data, 5)
    import_all(data)
    from ouroboros import dialogue_store
    assert dialogue_store.counts(data) == {"chat": 10, "progress": 10}
    opened.clear()
    pr.reconcile_projects(data)
    pr.reconcile_projects(data)
    [room] = pr.projects_summary(data)
    assert room["has_thread_activity"] is False
    assert _under(opened, data / "archive") == []
    assert "activity_backfill" not in maintenance.JOBS


def _promotion_member(parent, task):
    """A settled child whose call inventory misses one blob: a deterministic refusal to remember."""
    from ouroboros import headless, observability
    from ouroboros.task_results import write_task_result

    child = headless.prepare_task_drive(parent, task, "empty")
    call = observability.persist_call(child, task_id=task, call_id="broken", call_type="tool_call",
                                      payload={"tool": "read_file", "result": f"original of {task}"})
    manifest = json.loads(pathlib.Path(call["manifest_ref"]["path"]).read_text(encoding="utf-8"))
    pathlib.Path(manifest["full_payload_ref"]["path"]).unlink()
    write_task_result(child, task, "completed", result="answer", artifact_status="ready")
    headless.copy_child_task_result(parent, {"id": task, "drive_root": str(child)})
    return child


def test_child_ref_promotion_rests_on_its_second_run(tmp_path, opened, monkeypatch):
    """The remembered refusal is a durable row: the next run (any process; the memo is read
    from the store, never from module memory) reads no evidence and no body and commits nothing."""
    parent = tmp_path / "canonical"
    child = _promotion_member(parent, "member")
    assert "child_ref_promotion" in maintenance.due(parent)
    assert maintenance.run_job(parent, "child_ref_promotion")["state"] == "done"
    assert "child_ref_promotion" not in maintenance.due(parent)
    before, decodes = _revision(parent), _decodes(monkeypatch)
    opened.clear()
    assert maintenance.run_job(parent, "child_ref_promotion") is None  # at rest: only its facts were read
    assert decodes == [] and _revision(parent) == before
    assert _under(opened, parent / "observability") == [] and _under(opened, child / "observability") == []


def test_child_ref_promotion_killed_after_k_members_resumes_at_k(tmp_path, monkeypatch):
    from ouroboros import observability

    parent = tmp_path / "canonical"
    for task in ("t1", "t2", "t3"):
        _promotion_member(parent, task)
    walk, walked = observability._retry_pending_child_ref_promotion, []

    def killed_on_the_third(parent_root, child, task_id, result, **kwargs):
        if len(walked) == 2:
            raise _Killed()
        walked.append(task_id)
        return walk(parent_root, child, task_id, result, **kwargs)

    monkeypatch.setattr(observability, "_retry_pending_child_ref_promotion", killed_on_the_third)
    with pytest.raises(_Killed):
        maintenance.run_job(parent, "child_ref_promotion")
    assert sorted(maintenance_jobs.job(parent, "child_ref_promotion")["observed"]) == walked
    monkeypatch.setattr(observability, "_retry_pending_child_ref_promotion",
                        lambda parent_root, child, task_id, result, **kw: walked.append(task_id) or walk(
                            parent_root, child, task_id, result, **kw))
    assert maintenance.run_job(parent, "child_ref_promotion")["state"] == "done"
    assert sorted(walked) == ["t1", "t2", "t3"], "a member remembered before the kill was walked again"


def test_planner_statistics_once_per_store_and_again_after_tenfold_growth(data):
    from ouroboros.task_results import write_task_result

    for index in range(3):
        write_task_result(data, f"task-{index}", "completed", result="answer")
    assert "sqlite_analyze" not in maintenance.due(data), "never by itself a reason to spawn the child"
    assert maintenance.run_job(data, "sqlite_analyze") == {"state": "done", "items": 3}
    with store.open_store(data, write=False) as txn:
        assert txn.conn.execute("SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl='tasks'").fetchone()[0] > 0
    assert _statistics(data)[1] == 0, "row counts only, whatever the SQLite build"
    before = _revision(data)
    assert maintenance.run_job(data, "sqlite_analyze") is None and _revision(data) == before
    for index in range(3, 30):
        write_task_result(data, f"task-{index}", "completed", result="answer")
    assert maintenance.run_job(data, "sqlite_analyze") == {"state": "done", "items": 30}
    assert _statistics(data)[1] == 0


def test_a_store_analyzed_before_keeps_row_counts_only_from_the_childs_next_run(data):
    """The job's plain ``ANALYZE`` of before, on a build with ENABLE_STAT4, left index samples that every new
    connection loads with the schema. The child's next pass, whatever job it runs for, drops them and keeps the
    ``sqlite_stat1`` rows and the tenfold base as they were; the pass after it rests. The server's look never
    spawns the child for it."""
    import contextlib
    import sqlite3

    from ouroboros.task_results import write_task_result

    for index in range(3):
        write_task_result(data, f"task-{index}", "completed", result="answer")
    with store.open_store(data, write=True) as txn:  # what the job wrote before its row named its statistics
        txn.conn.execute("ANALYZE")
        maintenance_jobs.record_in(txn, "sqlite_analyze", state="done", tasks=3)
    for index in range(3, 9):  # grown, short of tenfold
        write_task_result(data, f"task-{index}", "completed", result="answer")
    stat1, samples = _statistics(data)
    with contextlib.closing(sqlite3.connect(":memory:")) as probe:
        sampling = "ENABLE_STAT4" in {row[0] for row in probe.execute("PRAGMA compile_options")}
    assert (samples > 0) == sampling, (sqlite3.sqlite_version, samples)
    assert "sqlite_analyze" not in maintenance.due(data), "never by itself a reason to spawn the child"
    report = maintenance.run(data)
    assert report["jobs"]["sqlite_analyze"] == {"state": "done", "items": 0}, report
    assert _statistics(data) == (stat1, 0)
    assert maintenance_jobs.job(data, "sqlite_analyze")["tasks"] == 3
    before = _revision(data)
    assert maintenance.run_job(data, "sqlite_analyze") is None and _revision(data) == before
    for index in range(9, 30):
        write_task_result(data, f"task-{index}", "completed", result="answer")
    assert maintenance.run_job(data, "sqlite_analyze") == {"state": "done", "items": 30}
    assert _statistics(data)[1] == 0


def test_samples_an_older_body_analyzed_back_under_this_bodys_row_go_at_the_childs_next_run(data):
    """A Restore to a body whose job was a plain ``ANALYZE`` and a merge into its row: the store this body analyzed
    grows tenfold there and is analyzed again, so a build with ENABLE_STAT4 has its samples back while the row still
    names ``stat1``, beside that body's tenfold base. Back on this body neither its boot nor the server's look moves
    them; the child's next pass, whatever job it runs for, drops them and keeps the ``sqlite_stat1`` rows and the base,
    and the pass after it rests. Without samples the row's word holds and that pass leaves the job alone."""
    import contextlib
    import sqlite3

    from ouroboros import store_import
    from ouroboros.task_results import write_task_result

    def older_run():  # the job's run before its row named its statistics, verbatim
        with store.open_store(data, write=True) as txn:
            tasks = int(txn.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
            txn.conn.execute("ANALYZE")
        maintenance_jobs.record(data, "sqlite_analyze", state="done", tasks=tasks)
        return {"state": "done", "items": tasks}

    this_body, older_body = "this-body", "older-body"
    for index in range(3):
        write_task_result(data, f"task-{index}", "completed", result="answer")
    store_import.ensure_current_store(data, with_children=False, body_sha=this_body)
    assert maintenance.run_job(data, "sqlite_analyze") == {"state": "done", "items": 3}
    store.close_connections()
    store_import.ensure_current_store(data, with_children=False, body_sha=older_body)  # Restore: the older body boots
    for index in range(3, 30):  # tenfold: due by the older body's own rule
        write_task_result(data, f"task-{index}", "completed", result="answer")
    assert older_run() == {"state": "done", "items": 30}
    stat1, samples = _statistics(data)
    with contextlib.closing(sqlite3.connect(":memory:")) as probe:
        sampling = "ENABLE_STAT4" in {row[0] for row in probe.execute("PRAGMA compile_options")}
    assert (samples > 0) == sampling, (sqlite3.sqlite_version, samples)
    row = maintenance_jobs.job(data, "sqlite_analyze")
    assert (row["statistics"], row["tasks"]) == ("stat1", 30), "the merge kept this body's word"
    store.close_connections()
    store_import.ensure_current_store(data, with_children=False, body_sha=this_body)  # the update back
    assert _statistics(data) == (stat1, samples), "the boot leaves the statistics as they are"
    assert "sqlite_analyze" not in maintenance.due(data), "never by itself a reason to spawn the child"
    report = maintenance.run(data)
    assert report["jobs"].get("sqlite_analyze") == ({"state": "done", "items": 0} if sampling else None), report
    assert _statistics(data) == (stat1, 0)
    row = maintenance_jobs.job(data, "sqlite_analyze")
    assert (row["statistics"], row["tasks"]) == ("stat1", 30)
    before = _revision(data)
    assert maintenance.run_job(data, "sqlite_analyze") is None and _revision(data) == before


def test_legacy_reconcile_runs_only_when_scheduled_and_tolerates_a_body_without_it(data, monkeypatch):
    from ouroboros import store_import

    for name in ("pending_invocations", "ended_plan_asks", "runtime_retirement", "record_compression"):
        maintenance.run_job(data, name)
    calls = []
    monkeypatch.setattr(store_import, "run_legacy_reconcile",
                        lambda root, **kw: calls.append(kw) or maintenance_jobs.record(root, "legacy_reconcile",
                                                                                       state="done", done=4),
                        raising=False)
    assert maintenance.run_job(data, "legacy_reconcile") is None and calls == [], "never scheduled: never run"
    maintenance_jobs.record(data, "legacy_reconcile", state="pending", reason="a store-unaware body ran")
    assert maintenance.due(data) == ["legacy_reconcile"], "a scheduled reconcile is a reason to spawn"
    assert maintenance.run_job(data, "legacy_reconcile") == {"state": "done", "items": 4}
    assert set(calls[0]) == {"should_stop"} and maintenance.due(data) == []
    monkeypatch.delattr(store_import, "run_legacy_reconcile")
    maintenance_jobs.record(data, "legacy_reconcile", state="pending")
    assert maintenance.run_job(data, "legacy_reconcile") == {"state": "pending", "items": 0}


def test_a_retained_drive_rests_until_its_inputs_change(data, monkeypatch):
    """Drive settlement stays in the server (the queue lock and the liveness probe are its), but
    a drive retained for a reason a later pass would meet again is not decoded, copied or hashed
    on every pass: its verdict and the cursor are the ``drive_settlement`` row. A row change
    re-attempts it, and the settlement then succeeds."""
    from ouroboros import headless, server_maintenance as sm, task_custody
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import load_task_result, write_task_result

    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr("ouroboros.retention.age_cutoff", lambda *a, **k: 4_000_000_000)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    drive = headless.prepare_task_drive(data, "held", "empty")
    write_task_result(data, "held", "completed", result="answer", child_drive_root=str(drive),
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    sm._run_drive_custody_pass()
    assert sm.settlement_verdicts(data)["held"]["reason"] == "post_work_open" and drive.is_dir()
    assert "retained" not in maintenance_jobs.job(data, "drive_settlement"), "the verdicts have their own row"

    settle, attempts = task_custody.settle_child_drive, []
    with monkeypatch.context() as patch:
        patch.setattr(task_custody, "settle_child_drive",
                      lambda *a, **k: attempts.append(a[1]) or settle(*a, **k))
        decodes = _decodes(patch)
        sm._run_drive_custody_pass()  # compares the verdict, attempts nothing: records that the layout rests
        assert attempts == [] and decodes == [], "a resting verdict was re-attempted"
        before = _revision(data)
        sm._run_drive_custody_pass()  # the layout's inputs stand: nothing read, nothing written
        assert attempts == [] and decodes == [] and _revision(data) == before, "a resting layout was walked"
    # Only the probes go; this install stays bound while the post-work closes.
    write_task_result(data, "held", "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    sm._run_drive_custody_pass()
    assert not drive.exists() and "held" not in sm.settlement_verdicts(data)
    assert load_task_result(data, "held")["status"] == "completed"


def _setup_job(name, data, monkeypatch):
    """Synthetic inputs that give job NAME real work on its first run."""
    import time

    from ouroboros import store_import
    from ouroboros.task_results import write_task_result

    if name == "delegate_capture":
        from tests.test_delegate_capture_child import _queued

        assert _queued(data.parent, monkeypatch)[0] == data
    elif name == "subagent_metrics":
        from ouroboros.subagent_metrics import EVENT
        from ouroboros.utils import append_jsonl
        from tests.test_subagent_metrics import event
        append_jsonl(data / "logs/events.jsonl", event(type=EVENT))
    elif name == "legacy_reconcile":
        monkeypatch.setattr(store_import, "run_legacy_reconcile", lambda root, **kw: maintenance_jobs.record(
            root, "legacy_reconcile", state="done", done=1), raising=False)
        maintenance_jobs.record(data, "legacy_reconcile", state="pending")
    elif name == "child_ref_promotion":
        _promotion_member(data, "member")
    elif name == "custody_trash":
        moved = data / "state" / "custody_trash" / "task-0123456789ab" / "data"
        moved.mkdir(parents=True)
        (moved / "result.json").write_text("{}", encoding="utf-8")
    elif name == "temp_sweep":
        orphan = data / "state" / ".state.json.tmp.1.2.abc"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("x", encoding="utf-8")
        os.utime(orphan, (time.time() - 7200,) * 2)
    elif name == "sqlite_analyze":
        write_task_result(data, "task", "completed", result="answer")
    elif name == "evidence_packing":
        from tests.test_evidence_packs import _task

        _task(data, "settled", blobs=1)
    elif name == "record_compression":
        write_task_result(data, "settled", "completed", result="answer")
        evidence = data / "task_results" / "artifacts" / "settled" / "delegated_runs" / "run" / "evidence" / "out.log"
        evidence.parent.mkdir(parents=True)
        evidence.write_bytes(b"a finished record compresses where the filesystem can\n" * 4096)
    elif name == "dialogue_import":
        from ouroboros import dialogue_import
        from tests._dialogue import seed

        seed(data, archives={"chat_20261001T000000.jsonl": [{"ts": "2026-10-01T00:00:00Z", "direction": "in",
                                                             "chat_id": 1, "text": "archived"}]})
        dialogue_import.cutover(data)  # plans the archive segment; the job imports it
    elif name == "memory_tail":
        from ouroboros import dialogue_store
        from ouroboros.chronicle_store import ChronicleStore

        assert ChronicleStore(data).ensure_activated()["kind"] == "activation"  # its frontier at the stream's start
        dialogue_store.append(data, "chat", {"chat_id": 1, "direction": "in", "text": "after the frontier"})
    elif name == "room_appearances":
        from ouroboros import dialogue_store

        # An older body's notice: no appearance keys, the derivation mark left behind it.
        dialogue_store.append(data, "chat", {"chat_id": 1, "direction": "out", "type": "main_notice", "task_id": "t",
                                             "text": "an older body's notice"}, feed=True, appearances=None)
    elif name == "hot_facts":
        from ouroboros import store

        write_task_result(data, "older", "completed", result="answer")
        with store.open_store(data, write=True) as txn:  # an older writer left the hot columns unset
            txn.conn.execute("UPDATE tasks SET owner_pause_state=NULL WHERE task_id='older'")


@pytest.mark.parametrize("name", maintenance.JOB_NAMES)
def test_every_job_rests_on_unchanged_inputs(name, data, opened, monkeypatch):
    """The rest law, for every named job: once its work is done, another run over unchanged
    inputs reads only its facts (no history file opened, no task body decoded, nothing
    committed), and a fresh server process's dispatcher finds nothing due."""
    _setup_job(name, data, monkeypatch)
    outcomes = [maintenance.run_job(data, name) for _ in range(3)]  # a packing pass that unlinked owes one look
    assert outcomes[0] is not None and outcomes[-1] is None, outcomes
    before, decodes = _revision(data), _decodes(monkeypatch)
    opened.clear()
    assert maintenance.run_job(data, name) is None
    assert decodes == [] and _revision(data) == before
    for history in ("archive", "observability", "task_results", "state/headless_tasks"):
        assert _under(opened, data / history) == [], history
    assert name not in maintenance.due(data)


def test_no_history_sized_duty_runs_in_the_server_and_the_child_spawns_under_custody(data, monkeypatch):
    """Both cadences of the supervisor's maintenance tick run with every moved duty armed to fail
    in this process: none runs, no thread carries one's name, and the dispatcher spawns the one
    maintenance child through the custodied spawn (own module, session custody, own group)."""
    import threading

    from ouroboros import evidence_packer, observability, process_custody, server_maintenance as sm, task_custody, temp_sweep
    from tests.test_evidence_packs import REPO, _FakeChild

    ran = []

    def refuse(name):
        def run(*_args, **_kwargs):
            ran.append(name)
            raise AssertionError(f"{name} ran in the server process")
        return run

    monkeypatch.setattr(evidence_packer, "packing_pending", refuse("evidence discovery"))
    monkeypatch.setattr(evidence_packer, "run_pass", refuse("evidence packing"))
    monkeypatch.setattr(observability, "retry_pending_child_ref_promotions", refuse("child-ref promotion"))
    monkeypatch.setattr(task_custody, "remove_custody_trash", refuse("trash removal"))
    sweep = temp_sweep.sweep_stale_temp_files
    monkeypatch.setattr(temp_sweep, "sweep_stale_temp_files", lambda root, **kw: (
        refuse("atomic temp walk")() if kw.get("atomic_temps", True) else sweep(root, **kw)))
    started, start = [], threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", lambda thread: started.append(thread) or start(thread))
    spawned = []
    report = json.dumps({"status": "completed", "jobs": {}}).encode()
    monkeypatch.setattr(process_custody, "spawn_supervised", lambda cmd, **kw: spawned.append((cmd, kw)) or _FakeChild(report))
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "session")
    monkeypatch.setattr("ouroboros.config.REPO_DIR", REPO)
    unit = maintenance.MaintenanceChild()
    monkeypatch.setattr(maintenance, "child", unit)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", lambda **_kw: None)
    maintenance_jobs.record(data, "legacy_reconcile", state="pending")  # a scheduled job: a reason to spawn
    pr.create_project(data, "never-active", origin="reconcile")
    _archive(data, 2)
    (data / "state" / "custody_trash" / "moved-0123456789ab").mkdir(parents=True)

    sm._periodic_supervisor_maintenance([0.0], [0.0])  # the 600 s and 300 s cadences are due too
    while any(thread.is_alive() for thread in started):  # every pass and the dispatch it ends with
        for thread in list(started):
            thread.join(20)

    assert ran == []
    names = {thread.name for thread in started}
    assert "maintenance-child" in names and "evidence-packer" not in names
    # The process's first look and the reconcile pass's look each spawn while the scheduled job
    # stays due (this stub child does no work); every spawn is the custodied one.
    assert 1 <= len(spawned) <= 2
    for cmd, kwargs in spawned:
        assert cmd[1:3] == ["-m", "ouroboros.maintenance"] and cmd[cmd.index("--data-root") + 1] == str(data)
        assert kwargs["purpose"] == "maintenance" and kwargs["scope"] == "session"
        assert kwargs.get("new_process_group", True) and kwargs["env"]["OUROBOROS_DATA_DIR"] == str(data)


def test_the_real_child_process_runs_the_due_jobs_and_reports(data):
    """End to end, one real ``python -m ouroboros.maintenance`` on a synthetic root: the custodied
    spawn, the child's own facts, its jobs and its bounded report in ``logs/supervisor.jsonl``."""
    from tests.test_evidence_packs import REPO

    pr.create_project(data, "never-active", origin="reconcile")
    _archive(data, 2)
    (data / "state" / "custody_trash" / "moved-0123456789ab").mkdir(parents=True)
    (data / "logs").mkdir(exist_ok=True)
    from ouroboros import dialogue_import

    dialogue_import.cutover(data)  # as the boot does before the dialogue opens: the archive is planned
    assert maintenance.due(data) == ["subagent_metrics", "pending_invocations", "dialogue_import",
                                     "ended_plan_asks", "custody_trash", "runtime_retirement", "record_compression"]
    unit = maintenance.MaintenanceChild()
    assert unit.tick(data, REPO)
    unit._thread.join(240)
    assert not unit._thread.is_alive() and unit.live_children() == []
    rows = [json.loads(line) for line in (data / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    finished = [row for row in rows if row.get("type") == "maintenance" and row["phase"] != "started"]
    assert [row["phase"] for row in finished] == ["completed"], finished
    assert finished[0]["exit_code"] == 0
    assert "activity_backfill" not in finished[0]["jobs"]
    assert finished[0]["jobs"]["custody_trash"] == {"state": "done", "items": 1}
    assert finished[0]["jobs"]["dialogue_import"]["state"] == "done"
    assert pr.projects_summary(data)[0]["has_thread_activity"] is False
    assert maintenance.due(data) == [], "the child left every job at rest"


def test_the_dispatchers_look_costs_the_same_whatever_the_server_and_the_ledger_hold(data, monkeypatch):
    """``maintenance.due`` decodes only its own jobs' rows and probes the import ledger by index: the
    verdict map the server keeps for its retained drives and a ledger of imported segments, both sized
    by history, add nothing to the look (JSON decoded, statements, VM steps equal at 1 and 30 units)."""
    from ouroboros import dialogue_store as ds, server_maintenance as sm
    from ouroboros.startup_migrations import prepare_startup_state
    from tests._cost_meter import BOUNDARY, CostMeter

    looks = []
    for scale in (1, 30):
        root = data.parent / f"root{scale:02}"
        prepare_startup_state(root)
        with store.open_store(root, write=True) as txn:
            txn.set_meta(sm.SETTLEMENT_VERDICTS, {f"drive-{index}": {
                "layout": "headless_tasks", "reason": "canonical_not_settled",
                "basis": [[index, "2025-01-01T00:00:00+00:00"], [index, 96, 1], None, "FileNotFoundError", "7.8.0"]}
                for index in range(20 * scale)})
            maintenance_jobs.record_in(txn, "drive_settlement", cursor={"headless": "", "direct": ""}, rest={})
        ds.create(root)
        with ds.transaction(root, write=True) as conn:
            conn.executemany("INSERT INTO import_ledger (stream, name, base, size, state) VALUES ('chat', ?, ?, 1, 'done')",
                             [(f"chat_{index:06d}.jsonl", index) for index in range(20 * scale)])
        store.close_connections()
        with CostMeter(root) as meter:
            token = BOUNDARY.set("look")
            try:
                due = maintenance.due(root)
            finally:
                BOUNDARY.reset(token)
        counts = meter.snapshot("look")["counts"]
        looks.append((due, {key: counts.get(key, 0) for key in ("json_chars", "json_loads", "statements", "vm_steps", "open", "scandir")}))
    assert looks[0] == looks[1], looks


def test_a_layout_whose_inputs_stand_is_not_listed_and_each_input_wakes_it(data, monkeypatch):
    """After a whole pass finds nothing to settle, the next passes read the layout's basis and stop: the
    drives are not listed, no verdict is compared, no body decoded, nothing written. A task completing
    elsewhere costs one indexed read of the rows written since and leaves the rest standing; a drive's
    own closed task written again, and a drive its creator published, are looked at alone, without a
    listing. A drive added without a publication, a changed retention setting, a new server generation
    and a young drive reaching its retention instant each make the next pass walk again; a pass stopped
    by the generation token records no rest."""
    from ouroboros import headless, maintenance_retirement, server_maintenance as sm, task_custody
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import write_task_result

    now, day = 1_800_000_000.0, 86_400.0
    clock = [now]
    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "1")
    monkeypatch.setattr("ouroboros.retention.time.time", lambda: clock[0])
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    listings, compared = [], []
    names = headless._drive_names
    monkeypatch.setattr(headless, "_drive_names", lambda base: listings.append(base.name) or names(base))
    verdict = task_custody.rested_verdict
    monkeypatch.setattr(task_custody, "rested_verdict", lambda rested, parent, task_id, *a: compared.append(task_id) or verdict(rested, parent, task_id, *a))
    old = "2026-01-01T00:00:00+00:00"  # ten days before the clock: past retention
    drive = headless.prepare_task_drive(data, "held", "empty")
    write_task_result(data, "held", "completed", result="answer", child_drive_root=str(drive), ts=old,
                      root_phase_checkpoint={"post_task_synthesis": "running"})

    def rest():
        return (maintenance_jobs.job(data, "drive_settlement") or {}).get("rest", {}).get("headless") or {}

    def walks_then_rests(why):
        listings.clear()
        sm._run_drive_custody_pass()
        assert "headless_tasks" in listings, why  # the changed input made the pass walk (no direct layout here)
        for _pass in range(3):
            if rest():
                return
            sm._run_drive_custody_pass()
        raise AssertionError(f"{why}: the lap never came to rest")

    sm._run_drive_custody_pass()  # attempts the drive: retained post_work_open, its verdict recorded
    assert sm.settlement_verdicts(data)["held"]["reason"] == "post_work_open" and not rest()
    sm._run_drive_custody_pass()  # walks, compares the verdict, attempts nothing: the layout rests
    assert rest()["basis"]["directory"] and rest()["youngest"] is None
    listings.clear(), compared.clear()
    before, decodes = _revision(data), _decodes(monkeypatch)
    for _pass in range(2):
        sm._run_drive_custody_pass()
    assert listings == [] and compared == [] and decodes == [] and _revision(data) == before, "a resting layout was walked"
    assert drive.is_dir()

    fresh = headless.prepare_task_drive(data, "fresh", "empty")  # the creator publishes the arrival
    write_task_result(data, "fresh", "completed", result="answer", child_drive_root=str(fresh), ts=old,
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    listings.clear(), compared.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and compared == ["fresh"], "a drive its creator published is looked at alone, without a listing"
    assert sm.settlement_verdicts(data)["fresh"]["reason"] == "post_work_open" and not rest()
    walks_then_rests("the whole pass after the published drive was attempted")
    (data / "state" / "headless_tasks" / "copied" / "data").mkdir(parents=True)  # a drive nobody published
    walks_then_rests("a drive added without a publication")
    write_task_result(data, "elsewhere", "completed", result="an ordinary finished turn")  # no drive of this layout
    listings.clear(), compared.clear(), decodes.clear()
    counter = rest()["basis"]["terminal"]
    sm._run_drive_custody_pass()
    assert listings == [] and compared == [] and decodes == [], "a completion elsewhere walked the layout"
    assert rest()["basis"]["terminal"] > counter, "the rest stands under the moved counter"
    write_task_result(data, "held", "completed", note="written again")  # a closed task written again: its drive alone
    listings.clear(), compared.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and compared == ["held"], "a drive's own row written is looked at alone, without a listing"
    walks_then_rests("the whole pass after a drive's own row changed")
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "2")
    walks_then_rests("the retention setting changed")
    maintenance_retirement.record_startup(data, gaps=False)
    walks_then_rests("a new server generation")
    young = headless.prepare_task_drive(data, "young", "empty")  # published: looked at alone, its instant kept
    write_task_result(data, "young", "completed", result="answer", child_drive_root=str(young),
                      ts="2027-01-14T16:00:00+00:00")  # a day before the clock: inside the two-day retention
    listings.clear(), compared.clear(), decodes.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and compared == [] and decodes == [], "a published young drive is skipped alone: no verdict decoded"
    assert rest()["youngest"] is not None, "the rest stands and keeps the young drive's instant"
    sm._run_drive_custody_pass()
    assert listings == [], "a young drive is not a reason to walk before its instant"
    clock[0] = now + 2 * day
    walks_then_rests("the young drive reached its retention instant")
    assert not young.is_dir(), "it settled when its instant came"
    kept: dict = {}
    report = headless.prune_headless_task_drives(data, live=lambda _task: False, stop=lambda: True, rested={}, rest=kept)
    assert report["deferred"] and kept == {}, "a stopped pass records no rest"


def test_a_published_drive_arrival_is_looked_at_alone_consumed_once_and_best_effort(data, monkeypatch):
    """The drive creator publishes the new drive (``task_custody.record_drive_arrival`` at the end of
    ``headless.prepare_task_drive``): the next pass looks at that drive alone, with no listing and no
    other verdict compared, leaves the rest standing under the moved inputs, and forgets the publication
    in the transaction that records the pass; a publication landing during a walking pass survives it
    and is looked at by the next. The publication is best-effort: a root nothing imported gets no store
    built for it, and a store that cannot take the write loses nothing (the creator still returns the
    drive; the layout directory's moved mtime makes the next pass walk whole, as every pass did before)."""
    from ouroboros import headless, server_maintenance as sm, task_custody
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import write_task_result

    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    listings, compared = [], []
    names = headless._drive_names
    monkeypatch.setattr(headless, "_drive_names", lambda base: listings.append(base.name) or names(base))
    verdict = task_custody.rested_verdict
    monkeypatch.setattr(task_custody, "rested_verdict",
                        lambda rested, parent, task_id, *a: compared.append(task_id) or verdict(rested, parent, task_id, *a))
    layout = data / "state" / "headless_tasks"

    def rest():
        return (maintenance_jobs.job(data, "drive_settlement") or {}).get("rest", {}).get("headless") or {}

    def arrivals():
        with store.open_store(data, write=False) as txn:
            return sorted(txn.set_rows(task_custody.DRIVE_ARRIVALS))

    def at_work(task_id):
        drive = headless.prepare_task_drive(data, task_id, "empty")
        write_task_result(data, task_id, "running", child_drive_root=str(drive))
        return drive

    layout.mkdir(parents=True)
    sm._run_drive_custody_pass()  # an empty layout: listed once, the lap rests
    assert rest() and "headless_tasks" in listings and rest()["basis"]["arrivals"] is None
    at_work("atwork")
    assert arrivals() == ["headless_tasks/atwork"]
    listings.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and set(compared) <= {"atwork"} and arrivals() == [], "the published drive is looked at alone and forgotten"
    assert rest()["basis"]["arrivals"] == 1, "the rest stands under the moved inputs"
    revision = _revision(data)
    sm._run_drive_custody_pass()
    assert listings == [] and _revision(data) == revision, "and the layout rests again"

    (layout / "copied" / "data").mkdir(parents=True)  # a drive nobody published: the next pass walks whole ...
    published = []

    def listing_with_an_arrival(base):  # ... and while it walks, a creator publishes another drive
        if not published:
            published.append(at_work("late"))
        return names(base)
    monkeypatch.setattr(headless, "_drive_names", listing_with_an_arrival)
    sm._run_drive_custody_pass()
    assert published and arrivals() == ["headless_tasks/late"] and rest(), "a publication during the walk survives it"
    monkeypatch.setattr(headless, "_drive_names", lambda base: listings.append(base.name) or names(base))
    listings.clear(), compared.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and set(compared) <= {"late"} and arrivals() == [] and rest(), "the next pass looks at it alone"

    bare = data.parent / "bare"  # a root nothing imported: the creator builds no store to publish into
    drive = headless.prepare_task_drive(bare, "orphan", "empty")
    assert drive.is_dir() and not (bare / "state" / "current.sqlite").exists()
    opening = store.open_store

    def refusing(root, *, write, **kwargs):  # a store that cannot take the publication's write
        if write:
            raise store.StoreBusy("held elsewhere", reason="contention")
        return opening(root, write=write, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(store, "open_store", refusing)
        assert task_custody.record_drive_arrival(data, layout, "unpublished") is False
        drive = headless.prepare_task_drive(data, "unpublished", "empty")
    assert drive.is_dir() and arrivals() == []
    write_task_result(data, "unpublished", "running", child_drive_root=str(drive))
    listings.clear()
    sm._run_drive_custody_pass()
    assert "headless_tasks" in listings and rest(), "an unpublished drive moved the directory: the pass walked whole and rests"


def test_the_arrival_publication_never_waits_for_a_busy_store_and_lands_on_a_free_one(data, monkeypatch):
    """Task creation publishes the new drive's arrival under a zero acquisition budget: a store another
    writer holds costs the creator no wait (the drive exists, no row lands, and the next pass walks the
    layout whole and rests, as every pass did before the publication existed); a free store still takes
    the row, and the next pass looks at the drive alone. With the budget removed the creation spends the
    store's whole wait (``store.STORE_WAIT_SEC``) on every task admitted while the store is held."""
    import contextlib
    import sqlite3
    import time

    from ouroboros import headless, server_maintenance as sm, task_custody
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import write_task_result

    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", lambda _task: False)
    listings = []
    names = headless._drive_names
    monkeypatch.setattr(headless, "_drive_names", lambda base: listings.append(base.name) or names(base))
    (data / "state" / "headless_tasks").mkdir(parents=True)

    def arrivals():
        with store.open_store(data, write=False) as txn:
            return sorted(txn.set_rows(task_custody.DRIVE_ARRIVALS))

    sm._run_drive_custody_pass()  # the lap rests on the empty layout
    with contextlib.closing(sqlite3.connect(store.store_path(data), isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")  # another writer holds the store throughout the creation
        try:
            started = time.monotonic()
            held = headless.prepare_task_drive(data, "held", "empty")
            elapsed = time.monotonic() - started
        finally:
            holder.execute("ROLLBACK")
    assert held.is_dir() and arrivals() == [], "the creator keeps its drive; no row landed"
    assert elapsed < 1, f"a busy store cost the creator {elapsed:.1f} s: the publication waited for it"
    write_task_result(data, "held", "running", child_drive_root=str(held))
    listings.clear()
    sm._run_drive_custody_pass()
    assert "headless_tasks" in listings, "the unpublished drive moved the directory: the pass walked whole"

    free = headless.prepare_task_drive(data, "free", "empty")
    assert free.is_dir() and arrivals() == ["headless_tasks/free"], "a free store still takes the publication"
    write_task_result(data, "free", "running", child_drive_root=str(free))
    listings.clear()
    sm._run_drive_custody_pass()
    assert listings == [] and arrivals() == [], "the next pass looks at the published drive alone"
