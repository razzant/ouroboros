"""Source-bound repairs of the body candidate / adoption seams (#1539 countercheck).

Each test drives the real owner — the switch helper, the candidate registry, the
server exit tail, the launcher's relaunch rule, the engine-pin reader, the
review substrate, the commit attribution baseline — through both the
preserving and the refusing branch of the repaired seam. Real Git throughout.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import body_adoption, body_candidate, body_switch
from tests.body_candidate_support import (
    candidate_commit, git, isolate, make_ctx, make_serving, restart_receipt, rich_commit,
)

EXITED = {"doomed": [4001], "dead": [4001], "unconfirmed": [], "cleanup_ok": True, "snapshot_ok": True}


@pytest.fixture
def body(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    return serving, data


def _armed(serving, data, ctx, cand, reason="adopt it"):
    body_adoption.authorize(ctx, cand, reason=reason)
    restart_receipt(data, cand, reason)
    assert body_adoption.bind_restart(lambda **_kw: (True, "ok"), data, reason)()[0]
    assert body_adoption.arm(data, worker_exits=EXITED, live_children=[], owned_stop={"state": "completed",
                             "unconfirmed": []}, owner_restart=False) == "armed"
    return body_adoption.read(data)


# --------------------------------------------------------------------------- #
# 1. A return to the old tree from a half-switched start needs a fresh process
# --------------------------------------------------------------------------- #
def test_return_to_old_hands_over_only_when_this_process_began_on_a_half_switched_tree(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-return")
    body_candidate.prepare(ctx)
    cand = rich_commit(ctx.repo_dir)
    body_candidate.record_reviewed_commit(ctx, cand)
    handoff = _armed(serving, data, ctx, cand)
    root, rows, helper = str(serving), handoff["switch"], str(body_adoption.helper_dir(data))
    handovers = []
    monkeypatch.setattr(body_switch, "_handover", lambda: handovers.append(True))
    tracked = body_switch._tracks_filemode(root)

    # Wholly old tree + a foreign file: returned in place, boot continues in this process.
    foreign = next(row for row in rows if row["old_sha"] and row["new_sha"])
    (serving / foreign["path"]).write_text("owner edit\n", encoding="utf-8")
    body_switch._switch(root, helper, body_switch.read_handoff(helper))
    assert body_adoption.read(data)["phase"] == "abandoned" and handovers == []
    assert (serving / foreign["path"]).read_text() == "owner edit\n"
    assert git(serving, "rev-parse", "HEAD") == handoff["old"]

    # The same refusal from a tree one switch path of which was ALREADY at the new side when
    # this process started (a resumed attempt): the old tree is restored, then handed over.
    git(serving, "checkout", "--", foreign["path"])
    handoff = _armed(serving, data, ctx, cand, reason="again")
    rows = handoff["switch"]
    first = next(row for row in rows if row["new_sha"] and row["path"] != foreign["path"])
    body_switch._put(root, first["path"], first["new_mode"], first["new_sha"])
    (serving / foreign["path"]).write_text("owner edit\n", encoding="utf-8")
    body_switch._switch(root, helper, body_switch.read_handoff(helper))
    assert body_adoption.read(data)["phase"] == "abandoned" and handovers == [True]
    files = body_switch._states(root, rows, tracked)
    assert files[first["path"]] == body_switch._side(first, "old", tracked)


# --------------------------------------------------------------------------- #
# 8. A directory at a switch path that holds only switch paths is not foreign
# --------------------------------------------------------------------------- #
def test_file_to_directory_transitions_and_bytecode_are_not_foreign_content(body):
    serving, data = body
    ctx = make_ctx(serving, data, "root-dir")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    # Old side: ouroboros/pkg.py (a module); new side: ouroboros/pkg/__init__.py (a package).
    (candidate / "ouroboros" / "pkg.py").write_text("OLD = 1\n")
    git(candidate, "add", "-A")
    git(candidate, "commit", "-q", "-m", "module")
    base = git(candidate, "rev-parse", "HEAD")
    git(serving, "fetch", "-q", str(candidate), "HEAD")
    git(serving, "reset", "-q", "--hard", base)
    (serving / "ouroboros" / "__pycache__").mkdir(exist_ok=True)
    (serving / "ouroboros" / "__pycache__" / "pkg.cpython-310.pyc").write_bytes(b"stale")
    cand = candidate_commit(candidate, "package", files={"ouroboros/pkg/__init__.py": "NEW = 1\n"},
                            delete=("ouroboros/pkg.py",))
    body_candidate.record_reviewed_commit(ctx, cand)
    row = body_candidate.find("root-dir")
    body_candidate._update_row(row, lambda entry: entry.__setitem__("base_sha", base))
    handoff = _armed(serving, data, ctx, cand)
    root, rows = str(serving), handoff["switch"]
    tracked = body_switch._tracks_filemode(root)
    paths = sorted(row["path"] for row in rows)
    assert paths == ["ouroboros/pkg.py", "ouroboros/pkg/__init__.py"]
    assert body_switch._foreign_paths(rows, body_switch._states(root, rows, tracked),
                                      body_switch._index_states(root, rows), tracked) == []
    body_switch._converge(root, rows, "new", body_switch._states(root, rows, tracked), tracked)
    assert (serving / "ouroboros" / "pkg" / "__init__.py").read_text() == "NEW = 1\n"
    assert not (serving / "ouroboros" / "__pycache__" / "pkg.cpython-310.pyc").exists()
    # Back: the package directory holds only the switch path, so it reads as absent, not foreign.
    files = body_switch._states(root, rows, tracked)
    assert body_switch._foreign_paths(rows, files, body_switch._index_states(root, rows), tracked) == []
    body_switch._converge(root, rows, "old", files, tracked)
    assert (serving / "ouroboros" / "pkg.py").read_text() == "OLD = 1\n"
    assert not (serving / "ouroboros" / "pkg").exists()
    # A directory AT a switch path with anyone else's file in it IS foreign.
    (serving / "ouroboros" / "pkg.py").unlink()
    (serving / "ouroboros" / "pkg.py").mkdir()
    (serving / "ouroboros" / "pkg.py" / "notes.txt").write_text("mine\n")
    files = body_switch._states(root, rows, tracked)
    assert files["ouroboros/pkg.py"] == ("directory", "directory")
    assert body_switch._foreign_paths(rows, files, body_switch._index_states(root, rows), tracked) == ["ouroboros/pkg.py"]


# --------------------------------------------------------------------------- #
# 3 + 12. Boot settlement and the exit tail read the real owners' facts
# --------------------------------------------------------------------------- #
def test_boot_settles_an_adoption_as_unready_when_the_bootstrap_failed(body, monkeypatch):
    import server
    import threading

    serving, data = body
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "REPO_DIR", serving)
    seen = []
    monkeypatch.setattr(body_adoption, "settle_on_boot", lambda *a, **kw: seen.append(kw["supervisor_ready"]) or {})
    monkeypatch.setattr("supervisor.update_merge.finalize_managed_update_on_boot", lambda **kw: {})
    monkeypatch.setattr("supervisor.update_merge.active_update_tx", lambda: None)
    monkeypatch.setattr("supervisor.git_ops.compute_managed_update_status", lambda *a, **kw: {})
    done = threading.Event()
    done.set()
    monkeypatch.setattr(server, "_supervisor_init_done", done)
    monkeypatch.setattr(server, "_supervisor_error", None)
    for outcome, expected in ((True, True), (False, False), (None, False)):
        monkeypatch.setattr(server, "_bootstrap_ok", outcome)
        server._boot_managed_update_tasks()
    assert seen == [True, False, False]


@pytest.mark.serial  # real cold entries
@pytest.mark.parametrize("fails", ["pointer", "armed_record"])
def test_arming_interrupted_on_either_side_of_the_pointer_settles_unapplied_and_never_holds_the_checkout(
        body, monkeypatch, fails):
    """`armed` is recorded only after the pointer is published: an unwritable Git dir leaves an
    unarmed record, and a death between the two leaves a pointer to a record the helper ignores."""
    from tests.body_candidate_support import run_entry

    serving, data = body
    ctx = make_ctx(serving, data, "root-arm-interrupted")
    body_candidate.prepare(ctx)
    cand = rich_commit(ctx.repo_dir)
    body_candidate.record_reviewed_commit(ctx, cand)
    old = git(serving, "rev-parse", "HEAD")
    body_adoption.authorize(ctx, cand, reason="adopt it")
    restart_receipt(data, cand, "adopt it")
    assert body_adoption.bind_restart(lambda **_kw: (True, "ok"), data, "adopt it")()[0]
    pointer = pathlib.Path(body_switch.pointer_path(str(serving)))
    real_record = body_switch.record

    real_write = pathlib.Path.write_text

    def git_dir_not_writable(path, *args, **kwargs):
        if path.parent == pointer.parent:
            raise PermissionError("Git dir not writable")
        return real_write(path, *args, **kwargs)

    def dies_before_armed(helper, handoff, phase, detail=""):
        if phase == "armed":
            raise KeyboardInterrupt  # stands for the process dying here
        return real_record(helper, handoff, phase, detail)

    with monkeypatch.context() as patch:
        if fails == "pointer":
            patch.setattr(pathlib.Path, "write_text", git_dir_not_writable)
            assert body_adoption.arm(data, worker_exits=EXITED, live_children=[], owner_restart=False,
                                     owned_stop={"state": "completed", "unconfirmed": []}) == ""
        else:
            patch.setattr(body_switch, "record", dies_before_armed)
            with pytest.raises(KeyboardInterrupt):
                body_adoption.arm(data, worker_exits=EXITED, live_children=[], owner_restart=False,
                                  owned_stop={"state": "completed", "unconfirmed": []})
    assert body_adoption.read(data)["phase"] == "authorized" and pointer.exists() == (fails == "armed_record")

    started = run_entry(serving)  # the next cold start: the real hook, then boot settlement
    assert started.returncode == 0 and json.loads(started.stdout)["a"] == "GEN_OLD", started.stderr
    assert body_adoption.finalize_on_boot(data, serving, supervisor_ready=True)["outcome"] == "not_applied"
    assert not pointer.exists() and body_adoption.read(data) == {} and git(serving, "rev-parse", "HEAD") == old
    assert run_entry(serving).returncode == 0  # nothing left for a later start to refuse on
    assert body_adoption.authorize(ctx, cand, reason="adopt again")["phase"] == "authorized"


def test_kill_workers_publishes_a_pid_census_the_exit_tail_arms_on(tmp_path, monkeypatch):
    from supervisor import workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "_LAST_WORKER_EXIT_CENSUS", None)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **kw: True)
    assert workers.last_worker_exit_census() is None  # before any kill: no census, never an inferred one
    workers.WORKERS.clear()
    assert workers.kill_workers(force=True, archive_service_logs=False) is True
    census = workers.last_worker_exit_census()
    assert census["doomed"] == [] and census["dead"] == [] and census["unconfirmed"] == [] and census["ts"]
    assert census["cleanup_ok"] is True and census["snapshot_ok"] is True
    monkeypatch.setattr(workers, "_LAST_WORKER_EXIT_CENSUS", {**EXITED, "unconfirmed": [4002]})
    assert workers.last_worker_exit_census()["unconfirmed"] == [4002]


# --------------------------------------------------------------------------- #
# 4. The launcher re-executes itself when the landed checkout changed its own modules
# --------------------------------------------------------------------------- #
def test_launcher_relaunches_only_when_a_loaded_module_changed_in_the_checkout(tmp_path, monkeypatch):
    from ouroboros import launcher_bootstrap as launcher

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    (repo / "launcher.py").write_text("print('launcher')\n")
    (repo / "other.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    monkeypatch.setattr(launcher, "_LOADED_CHECKOUT_SHA", "")
    loaded = launcher.remember_loaded_checkout(repo)
    assert loaded == git(repo, "rev-parse", "HEAD") == launcher.remember_loaded_checkout(tmp_path)  # set once
    assert launcher.launcher_sources_changed(repo, loaded) is False  # same commit
    (repo / "other.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "unrelated")
    assert launcher.launcher_sources_changed(repo, loaded) is False  # HEAD moved, loaded modules untouched
    (repo / "launcher.py").write_text("print('launcher v2')\n")
    git(repo, "commit", "-qam", "launcher")
    assert launcher.launcher_sources_changed(repo, loaded) is True
    assert launcher.launcher_sources_changed(repo) is True  # the remembered commit is the default base
    assert launcher.launcher_sources_changed(repo, "") is False  # unknown import-time commit: never a guess
    assert launcher.checkout_sha(tmp_path / "not-a-repo") == ""


def test_packaged_launcher_counts_helpers_it_loaded_from_its_bundle(tmp_path, monkeypatch):
    """A frozen launcher imported its helpers from the bundle, not the checkout:
    PyInstaller presents them as ``<bundle>/<pkg>/<module>.pyc`` (not a native app run)."""
    from ouroboros import launcher_bootstrap as launcher
    from ouroboros import launcher_server_reaper as helper

    repo, bundle = tmp_path / "repo", tmp_path / "bundle"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    for rel in ("launcher.py", "ouroboros/launcher_server_reaper.py", "ouroboros/other.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("GEN = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    loaded = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(helper, "__file__", str(bundle / "ouroboros" / "launcher_server_reaper.pyc"))
    (repo / "ouroboros/other.py").write_text("GEN = 2\n")
    git(repo, "commit", "-qam", "unrelated")
    assert launcher.launcher_sources_changed(repo, loaded, bundle_dir=bundle) is False
    (repo / "ouroboros/launcher_server_reaper.py").write_text("GEN = 2\n")
    git(repo, "commit", "-qam", "helper")
    assert launcher.launcher_sources_changed(repo, loaded) is False  # the checkout alone never names it
    assert launcher.launcher_sources_changed(repo, loaded, bundle_dir=bundle) is True


# --------------------------------------------------------------------------- #
# 5. The planned restart compares the engine the NEXT generation selects
# --------------------------------------------------------------------------- #
def test_next_generation_pin_comes_from_the_bound_candidate_when_the_switch_changes_it(monkeypatch):
    from ouroboros import server_restart
    from ouroboros.claudexor_runtime import _PIN_FILENAME

    calls = []
    monkeypatch.setattr("ouroboros.claudexor_runtime.load_runtime_pin",
                        lambda path=None: calls.append(path) or SimpleNamespace(path=path))
    monkeypatch.setattr(body_adoption, "bound_candidate_file", lambda data_dir, rel: None)
    assert server_restart._next_generation_pin().path is None  # the landed checkout's own pin
    monkeypatch.setattr(body_adoption, "bound_candidate_file",
                        lambda data_dir, rel: b'{"schema_version": 1, "release": null}' if rel.endswith(_PIN_FILENAME) else None)
    pin = server_restart._next_generation_pin()
    assert pin.path is not None and pin.path.name == _PIN_FILENAME and calls[-1] == pin.path


# --------------------------------------------------------------------------- #
# 6 + 7. The bound candidate is the review subject and never runs with the serving env
# --------------------------------------------------------------------------- #
def test_bound_candidate_is_the_review_subject_even_under_a_workspace_root(body, tmp_path):
    from ouroboros.review_substrate import review_repo_dirs_for

    serving, data = body
    project = tmp_path / "project"
    project.mkdir()
    ctx = make_ctx(serving, data, "root-review", workspace_root=project, workspace_mode="external")
    governance, subject = review_repo_dirs_for(ctx)
    assert (governance, subject) == (serving.resolve(), project.resolve())
    body_candidate.prepare(ctx)
    governance, subject = review_repo_dirs_for(ctx)
    assert governance == serving.resolve() and subject == pathlib.Path(ctx.repo_dir).resolve() != project.resolve()


def test_a_process_inside_the_candidate_is_refused_rather_than_started_with_the_serving_environment(body, monkeypatch):
    from ouroboros.tools import shell_process

    serving, data = body
    ctx = make_ctx(serving, data, "root-env")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    assert body_candidate.process_environment(ctx, serving) is None  # other cwd: the ordinary rule

    def broken(*_a, **_kw):
        raise OSError("env root unavailable")

    monkeypatch.setattr("ouroboros.test_environment.isolated_environment", broken)
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate.process_environment(ctx, candidate / "ouroboros")
    assert refused.value.code == "CANDIDATE_ENVIRONMENT_UNAVAILABLE"
    with pytest.raises(body_candidate.CandidateRefused):
        shell_process._shell_env_for_cwd(ctx, candidate)  # the shell owner does not fall back to the live env


# --------------------------------------------------------------------------- #
# 10. Transfer and GC need quiescent custody and a verdict that still holds
# --------------------------------------------------------------------------- #
def test_transfer_waits_for_the_previous_owners_processes(body, monkeypatch):
    serving, data = body
    first = make_ctx(serving, data, "root-first")
    body_candidate.prepare(first)
    row = body_candidate.find("root-first")
    monkeypatch.setattr(body_candidate, "_recorded_terminal", lambda task_id, root: True)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda root, task_id: [4242])
    second = make_ctx(serving, data, "root-second")
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate._transfer(second, row, "root-second", via="test")
    assert refused.value.code == "CANDIDATE_OWNER_PROCESSES_LIVE" and body_candidate.find("root-first")
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda root, task_id: [])
    moved = body_candidate._transfer(second, row, "root-second", via="test")
    assert moved["task_id"] == "root-second" and moved["owners"] == ["root-first", "root-second"]


def test_live_lineage_processes_reads_the_ownership_set_and_treats_unreadable_as_live(body, monkeypatch):
    serving, data = body
    records = [{"host_pid": 11, "owner_task": "root-a", "ledger_entry": {"pid": 11}},
               {"host_pid": 12, "owner_task": "root-b", "ledger_entry": {"pid": 12}}]
    monkeypatch.setattr("ouroboros.owned_shutdown.owned_records", lambda root, *, strict=False: records)
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda entry: entry["pid"] == 11)
    assert body_candidate.live_lineage_processes(data, "root-a") == [11]
    assert body_candidate.live_lineage_processes(data, "root-b") == []
    monkeypatch.setattr("ouroboros.owned_shutdown.owned_records", lambda root, *, strict=False: (_ for _ in ()).throw(OSError("x")))
    assert body_candidate.live_lineage_processes(data, "root-a") == [-1]


def test_candidate_custody_reads_real_owned_records_and_child_lineage(body, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros.task_results import write_task_result

    serving, data = body
    first = make_ctx(serving, data, "custody-root")
    bound = body_candidate.prepare(first)
    child = "custody-child"
    write_task_result(data, child, "completed", root_task_id="custody-root", result="done")
    record_path = data / "task_drives" / child / "state" / "workspace_executor_processes" / "service-1.json"
    record_path.parent.mkdir(parents=True)
    executor = {"id": "service-1", "schema_version": 1, "owner": "ouroboros_workspace_executor",
                "record_type": "service", "executor_type": "local", "task_id": child,
                "host_pid": 4321, "created_at": "today"}
    record_path.write_text(json.dumps(executor))
    ledger = {"pid": 4322, "purpose": "service:child", "scope": "task", "owner_task": child,
              "fingerprint": {"start_time": "today"}}
    entries = {}
    monkeypatch.setattr(owned_shutdown, "_publish", lambda root, entry: entries.update({entry["record_id"]: entry}) or True)
    assert owned_shutdown.record_executor_process(record_path, executor)
    assert owned_shutdown.record_ledgered_process(data, ledger)
    document = {"schema_version": 1, "records": entries}
    path = owned_shutdown.owned_processes_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda row: row.get("pid") == 4322)
    monkeypatch.setattr("ouroboros.workspace_executor._host_pid_matches_record", lambda row: row.get("host_pid") == 4321)
    monkeypatch.setattr("ouroboros.platform_layer.pid_provably_gone", lambda pid: False)
    write_task_result(data, "custody-root", "completed", result="done")
    assert 4321 in body_candidate.live_lineage_processes(data, "custody-root")
    assert 4322 in body_candidate.live_lineage_processes(data, "custody-root")
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate.prepare(make_ctx(serving, data, "new-root"), resume=bound["candidate_id"])
    assert refused.value.code == "CANDIDATE_OWNER_PROCESSES_LIVE"


def test_unreadable_owned_document_does_not_authorize_transfer_or_gc(body):
    from ouroboros import owned_shutdown
    from ouroboros import subagent_worktrees
    from ouroboros.task_results import write_task_result

    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "unreadable-owner"))
    write_task_result(data, "unreadable-owner", "completed", result="done")
    path = owned_shutdown.owned_processes_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json")
    assert body_candidate.live_lineage_processes(data, "unreadable-owner") == [-1]
    with pytest.raises(body_candidate.CandidateRefused) as refused:
        body_candidate.prepare(make_ctx(serving, data, "new-owner"), resume=bound["candidate_id"])
    assert refused.value.code == "CANDIDATE_OWNER_PROCESSES_LIVE"
    assert subagent_worktrees.prune_orphans(data_dir=data, retention_days=0)["kept"] == 1
    assert pathlib.Path(bound["path"]).is_dir()


def test_docker_executor_custody_uses_backend_outcome_not_host_pid_zero(body, monkeypatch):
    from ouroboros import owned_shutdown, workspace_executor

    _serving, data = body
    path = data / "task_drives" / "docker-root" / "state" / "workspace_executor_processes" / "service-1.json"
    path.parent.mkdir(parents=True)
    record = {"id": path.stem, "owner": workspace_executor._PROCESS_RECORD_OWNER,
              "schema_version": workspace_executor._PROCESS_RECORD_SCHEMA_VERSION,
              "record_type": "service", "executor_type": "docker_exec", "task_id": "docker-root",
              "host_pid": 0, "container_name": "task-container", "backend_pid": "99"}
    path.write_text(json.dumps(record))
    entry = owned_shutdown._executor_entry(path, record)
    document = owned_shutdown.owned_processes_path(data)
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(json.dumps({"schema_version": 1, "records": {entry["record_id"]: entry}}))
    monkeypatch.setattr(workspace_executor, "_docker_pid_state", lambda container, pid: "unknown")
    assert body_candidate.live_lineage_processes(data, "docker-root") == [-1]
    assert body_candidate.live_lineage_processes(data, "unrelated-root") == []
    monkeypatch.setattr(workspace_executor, "_docker_pid_state", lambda container, pid: "running")
    assert body_candidate.live_lineage_processes(data, "docker-root") == [-1]
    monkeypatch.setattr(workspace_executor, "_docker_pid_state", lambda container, pid: "exited")
    assert body_candidate.live_lineage_processes(data, "docker-root") == []


def test_removal_rechecks_new_custody_after_clean_verdict(body, monkeypatch):
    from ouroboros import owned_shutdown, subagent_worktrees
    from ouroboros.task_results import write_task_result

    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "late-process"))
    write_task_result(data, "late-process", "completed", result="done")
    verdict = body_candidate.retention_verdict(body_candidate.find("late-process"), expired=True, data_dir=data)
    assert verdict["reason"] == "no_unique_work"
    path = owned_shutdown.owned_processes_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"record_id": "pid-8", "kind": "supervised", "host_pid": 8,
             "owner_task": "late-process", "ledger_entry": {"pid": 8}}
    path.write_text(json.dumps({"schema_version": 1, "records": {"pid-8": entry}}))
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda row: True)
    assert not body_candidate.verdict_holds(body_candidate.find("late-process"), verdict, data_dir=data)
    monkeypatch.setattr(body_candidate, "retention_verdict", lambda *a, **kw: verdict)
    assert subagent_worktrees.prune_orphans(data_dir=data, retention_days=0)["kept"] == 1
    assert pathlib.Path(bound["path"]).is_dir()
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda row: False)
    assert body_candidate.verdict_holds(body_candidate.find("late-process"), verdict, data_dir=data)
    assert subagent_worktrees.prune_orphans(data_dir=data, retention_days=0)["removed"] == 1


def test_same_path_bytes_change_after_preservation_keeps_the_original_checkout(body, monkeypatch):
    from ouroboros import subagent_worktrees
    from ouroboros.task_results import write_task_result

    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "late-rewrite"))
    candidate = pathlib.Path(bound["path"])
    target = candidate / "untracked.bin"
    target.write_bytes(b"first\0")
    write_task_result(data, "late-rewrite", "completed", result="done")
    verdict = body_candidate.retention_verdict(body_candidate.find("late-rewrite"), expired=True, data_dir=data)
    assert verdict["reason"] == "preserved_on_pin"
    target.write_bytes(b"second\0")
    assert not body_candidate.verdict_holds(body_candidate.find("late-rewrite"), verdict)
    monkeypatch.setattr(body_candidate, "retention_verdict", lambda *a, **kw: verdict)
    assert subagent_worktrees.prune_orphans(data_dir=data, retention_days=0)["kept"] == 1
    assert target.read_bytes() == b"second\0"


def test_preserved_staged_version_change_after_verdict_keeps_original(body):
    from ouroboros.task_results import write_task_result

    serving, data = body
    bound = body_candidate.prepare(make_ctx(serving, data, "staged-rewrite"))
    candidate = pathlib.Path(bound["path"])
    target = candidate / "ouroboros" / "mod_a.py"
    target.write_text("GEN = 'FIRST_STAGED'\n")
    git(candidate, "add", "ouroboros/mod_a.py")
    target.write_text("GEN = 'SAME_ON_DISK'\n")
    write_task_result(data, "staged-rewrite", "completed", result="done")
    verdict = body_candidate.retention_verdict(body_candidate.find("staged-rewrite"), expired=True, data_dir=data)
    assert verdict["reason"] == "preserved_on_pin"
    target.write_text("GEN = 'SECOND_STAGED'\n")
    git(candidate, "add", "ouroboros/mod_a.py")
    target.write_text("GEN = 'SAME_ON_DISK'\n")
    assert not body_candidate.verdict_holds(body_candidate.find("staged-rewrite"), verdict, data_dir=data)


def test_gc_applies_a_removing_verdict_only_while_it_still_describes_the_checkout(body, monkeypatch):
    from ouroboros import subagent_worktrees as wt

    serving, data = body
    ctx = make_ctx(serving, data, "root-gc")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    row = body_candidate.find("root-gc")
    monkeypatch.setattr(body_candidate, "_recorded_terminal", lambda *_a: True)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_a: [])
    verdict = body_candidate.retention_verdict(row, expired=True, data_dir=data)
    assert verdict["remove"] and verdict["reason"] == "no_unique_work"
    assert body_candidate.verdict_holds(body_candidate.find("root-gc"), verdict)
    (candidate / "late.txt").write_text("written after the verdict\n")  # a write between verdict and lock
    assert not body_candidate.verdict_holds(body_candidate.find("root-gc"), verdict)
    monkeypatch.setattr(body_candidate, "retention_verdict", lambda *a, **kw: verdict)
    assert wt.prune_orphans(data_dir=data, retention_days=0)["removed"] == 0
    assert candidate.is_dir() and (candidate / "late.txt").exists() and body_candidate.find("root-gc")
    (candidate / "late.txt").unlink()
    assert body_candidate.verdict_holds(body_candidate.find("root-gc"), verdict)
    assert wt.prune_orphans(data_dir=data, retention_days=0)["removed"] == 1 and not candidate.exists()


# --------------------------------------------------------------------------- #
# 11. Retention keeps a staged-only version and an author's top-level file
# --------------------------------------------------------------------------- #
def test_preserve_keeps_the_staged_version_distinct_from_the_file_on_disk(body, monkeypatch):
    serving, data = body
    ctx = make_ctx(serving, data, "root-staged")
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    (candidate / "ouroboros" / "mod_a.py").write_text("GEN = 'STAGED'\n")
    git(candidate, "add", "ouroboros/mod_a.py")
    (candidate / "ouroboros" / "mod_a.py").write_text("GEN = 'ON_DISK'\n")
    (candidate / "env").write_text("AUTHOR=1\n")  # a FILE named like an excluded directory: the author's
    row = body_candidate.find("root-staged")
    work = body_candidate.unique_work(row)
    assert "env" in work["loose"] and "ouroboros/mod_a.py" in work["tracked"]
    monkeypatch.setattr(body_candidate, "_recorded_terminal", lambda task_id, root: True)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda root, task_id: [])
    verdict = body_candidate.retention_verdict(row, expired=True, data_dir=data)
    assert verdict["reason"] == "preserved_on_pin"
    pin = verdict["pin"]
    assert git(candidate, "show", f"{pin}:ouroboros/mod_a.py") == "GEN = 'ON_DISK'"
    assert git(candidate, "show", f"{pin}^:ouroboros/mod_a.py") == "GEN = 'STAGED'"
    assert git(candidate, "show", f"{pin}:env") == "AUTHOR=1"
    assert git(candidate, "rev-parse", f"{pin}^^") == git(candidate, "rev-parse", "HEAD")


# --------------------------------------------------------------------------- #
# 13. A bound candidate is attributed through the lineage's own baseline
# --------------------------------------------------------------------------- #
def test_binding_appends_the_candidate_to_the_mutation_baseline_and_staging_consults_it(body):
    from ouroboros.mutation_attribution import capture_mutation_baseline, load_task_result
    from ouroboros.tools import git as git_tools

    serving, data = body
    capture_mutation_baseline(data, "root-attr", [{"surface_type": "system_repo", "host_root": str(serving)}],
                              owner_kind="task_root", owner_id="root-attr")
    ctx = make_ctx(serving, data, "root-attr", budget_drive_root=data)
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir).resolve()
    surfaces = load_task_result(data, "root-attr")["mutation_evidence"]["baseline"]["surfaces"]
    assert {pathlib.Path(s["canonical_root"]).resolve() for s in surfaces} == {serving.resolve(), candidate}
    (candidate / "ouroboros" / "mod_a.py").write_text("GEN = 'MINE'\n")
    selected, attribution, error, evidence = git_tools._task_attributed_commit_paths(ctx, None)
    assert error == "" and evidence == (data, "root-attr")
    assert selected == ["ouroboros/mod_a.py"] and attribution is not None


def test_retry_kept_service_retains_original_candidate(body, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros.task_results import write_task_result

    serving, data = body
    body_candidate.prepare(make_ctx(serving, data, "original-owner"))
    write_task_result(data, "original-owner", "failed", result="interrupted",
                      root_task_id="original-owner", superseded_by="retry-owner")
    write_task_result(data, "retry-owner", "completed", result="done", root_task_id="retry-owner")
    path = owned_shutdown.owned_processes_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"record_id": "retry-service", "kind": "supervised", "host_pid": 8,
             "owner_task": "retry-owner", "ledger_entry": {"pid": 8}}
    path.write_text(json.dumps({"schema_version": 1, "records": {"retry-service": entry}}), encoding="utf-8")
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda row: True)
    monkeypatch.setattr("ouroboros.process_custody._service_group_survives_leader", lambda row: False)
    assert body_candidate.live_lineage_processes(data, "original-owner") == [8]
    assert not body_candidate.retention_verdict(
        body_candidate.find("original-owner"), expired=True, data_dir=data)["remove"]
    monkeypatch.setattr("ouroboros.process_custody._fingerprint_matches", lambda row: False)
    assert body_candidate.live_lineage_processes(data, "original-owner") == []
    assert body_candidate.retention_verdict(
        body_candidate.find("original-owner"), expired=True, data_dir=data)["remove"]
