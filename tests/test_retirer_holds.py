"""Every server retirer releases what it may and reports what it keeps, in one shape (O5 / O10).

The five retirers of disposable planes (``docs/PERSISTENCE.md`` ``[disposable: ...]`` rows, named by
``tests/test_persistence_lifetimes.py::SERVER_RETIRERS``) are each witnessed here in both directions: one
fixture it can release goes (``released``), one it must keep stays and is reported as a typed hold
(``holds: [{class, count, example, since}]``, the engine's ``GcHold`` shape, through
``maintenance_jobs.record_pass`` as its job row); a second identical report writes nothing. A retirer without
a witness here fails the registry test: a remover that is named but never fires is what #2710 looked like.
Isolated roots only; the supervisor's probe is a stub that proves one task absent and one live.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Dict

from ouroboros import maintenance_jobs, server_maintenance, store_import
from ouroboros import subagent_worktrees as trees
from ouroboros.maintenance_jobs import record_pass
from tests._store_shared import seed_result
from tests.test_claudexor_runtime_retention import cx  # noqa: F401 -- the engine runtime fixture
from tests.test_session_service_directories import _synthetic_temp_parent  # noqa: F401 -- isolate session scratch parent

WITNESSES: Dict[str, str] = {}  # retirer (dotted) -> the test here that releases one fixture and holds another
OLD = "2026-10-01T00:00:00+00:00"


def witness(dotted: str) -> Callable:
    def bind(test: Callable) -> Callable:
        WITNESSES[dotted] = test.__name__
        return test
    return bind


def _probe(live_task: str):
    return lambda task_id: task_id == live_task


# ---- drive settlement: the custody pass's own hold building (server_maintenance) ----------------------------

@witness("ouroboros.task_custody.settle_child_drive")
def test_drive_settlement_releases_a_settled_drive_and_holds_a_live_one(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    store_import.ensure_current_store(data)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "1")
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", _probe("live"))
    for task in ("done", "live"):
        drive = data / "task_drives" / task
        drive.mkdir(parents=True)
        (drive / "scratch.bin").write_bytes(b"s" * 64)
        seed_result(data, task, {"status": "completed", "ts": OLD, "child_drive_root": str(drive)})

    server_maintenance._run_drive_custody_pass()
    row = maintenance_jobs.job(data, "drive_settlement")
    assert row["released"] == 1 and not (data / "task_drives" / "done").exists() and (data / "task_drives" / "live").is_dir()
    assert [(hold["class"], hold["count"], hold["example"]) for hold in row["holds"]] == [("task_live", 1, "live")]
    assert row["holds"][0]["since"] == row["pass_at"]

    server_maintenance._run_drive_custody_pass()
    again = maintenance_jobs.job(data, "drive_settlement")
    assert (again["holds"], again["pass_at"], again["released"]) == (row["holds"], row["pass_at"], 1), \
        "a second pass with the same hold set and nothing released writes no pass"


def test_drive_settlement_does_not_count_a_durable_verdict_twice(tmp_path, monkeypatch):
    """A skipped row already written to the durable verdict map is one hold; a fresh skipped
    row remains one hold, proving the duplicate guard is selective."""
    from ouroboros import headless

    data = tmp_path / "data"
    data.mkdir()
    store_import.ensure_current_store(data)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    monkeypatch.setattr(server_maintenance, "settlement_verdicts",
                        lambda _root: {"held": {"reason": "task_live"}})
    empty = {"pruned": [], "errors": [], "cursor": "", "arrivals": []}
    def headless_report(*args, **kwargs):
        kwargs["rested"]()
        return {**empty, "skipped": [{"task_id": "held", "reason": "task_live"}]}

    def direct_report(*args, **kwargs):
        kwargs["rested"]()
        return {**empty, "skipped": [{"task_id": "fresh", "reason": "store_busy"}]}

    monkeypatch.setattr(headless, "prune_headless_task_drives", headless_report)
    monkeypatch.setattr(headless, "prune_task_drives", direct_report)
    server_maintenance._run_drive_custody_pass()
    holds = {row["class"]: row["count"] for row in maintenance_jobs.job(data, "drive_settlement")["holds"]}
    assert holds == {"store_busy": 1, "task_live": 1}


def test_drive_settlement_holds_a_busy_read_for_the_next_pass_and_an_unreadable_row_as_unreadable(tmp_path, monkeypatch):
    """A busy store is contention, no fact about the row: the drive is held as ``store_busy`` and that pass
    leaves the layout no rest, so the next pass reads it again and settles it; a row the store cannot serve
    holds its drive as ``task_result_unreadable``."""
    from ouroboros import headless
    from ouroboros.store import StoreBusy, StoreError

    data = tmp_path / "data"
    data.mkdir()
    store_import.ensure_current_store(data)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "1")
    monkeypatch.setattr("supervisor.queue.task_settlement_liveness", _probe("none"))
    for task in ("busy", "torn"):
        drive = data / "task_drives" / task
        drive.mkdir(parents=True)
        seed_result(data, task, {"status": "completed", "ts": OLD, "child_drive_root": str(drive)})
    real, busy = headless.load_task_result, {"busy"}

    def strict_read(root, task_id, *args, **kwargs):  # the pass's strict row read, which decides eligibility
        if kwargs.get("strict") and task_id in busy:
            raise StoreBusy("contended", reason="lock_timeout")
        if kwargs.get("strict") and task_id == "torn":
            raise StoreError("simulated unreadable row: torn")
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(headless, "load_task_result", strict_read)
    server_maintenance._run_drive_custody_pass()
    row = maintenance_jobs.job(data, "drive_settlement")
    assert [(hold["class"], hold["example"]) for hold in row["holds"]] == [("store_busy", "busy"), ("task_result_unreadable", "torn")]
    assert (data / "task_drives" / "busy").is_dir() and not row["rest"]["direct"], "a busy read leaves the layout no rest"
    busy.clear()  # the contention passed
    server_maintenance._run_drive_custody_pass()
    row = maintenance_jobs.job(data, "drive_settlement")
    assert not (data / "task_drives" / "busy").exists() and row["released"] == 1
    assert [(hold["class"], hold["example"]) for hold in row["holds"]] == [("task_result_unreadable", "torn")]


# ---- private copies ---------------------------------------------------------------------------------------------

@witness("ouroboros.subagent_worktrees.prune_orphans")
def test_copy_retirement_releases_a_finished_copy_and_holds_a_live_one(tmp_path, monkeypatch):
    from tests.test_store import document
    from tests.test_storage_free import _init_repo

    root, repo, wtroot = tmp_path / "data", tmp_path / "repo", tmp_path / "wtroot"
    _init_repo(repo)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(wtroot))
    root.mkdir()
    store_import.ensure_current_store(root)
    paths = {}
    for task in ("done", "live"):
        paths[task] = trees.provision_worktree(repo_dir=repo, task_id=task, worktree_root=wtroot, data_dir=root).path
        seed_result(root, task, document(task, "completed"))
    reasons: Dict[str, str] = {}
    report = trees.prune_orphans(worktree_root=wtroot, data_dir=root, retention_days=0, live=_probe("live"), reasons=reasons)
    holds = maintenance_jobs.Holds()
    for path, reason in sorted(reasons.items()):
        holds.note(reason, path)
    assert report["removed"] == 1 and not pathlib.Path(paths["done"]).exists() and pathlib.Path(paths["live"]).is_dir()
    # The copies' proof is one predicate (terminal row, negative probe, no lineage process): its hold is that predicate's name.
    assert [(row["class"], row["count"], row["example"]) for row in holds.rows()] == [("not_proven_over", 1, str(paths["live"]))]
    first = record_pass(root, "copy_retirement", maintenance_jobs.pass_report(report["removed"], holds))
    assert first["released"] == 1 and first["holds"][0]["class"] == "not_proven_over"
    assert record_pass(root, "copy_retirement", maintenance_jobs.pass_report(0, holds)) is None


# ---- review checkouts --------------------------------------------------------------------------------------------

@witness("ouroboros.review_checking_prune.prune_review_checkouts")
def test_review_checkout_prune_releases_a_settled_seat_and_holds_an_open_run(tmp_path, monkeypatch):
    from ouroboros import review_checking_prune as rp
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.review_subject import ReviewSubjectSpec, checkout_token, isolated_checkout
    from tests.test_review_checking_prune import _custody
    from tests.test_review_checkout_retirement import _repo
    from tests.test_review_checkout_lifecycle import _frozen, _seat
    from types import SimpleNamespace

    root, source = _frozen(tmp_path)
    done, running = _seat(root, source, "op-done"), _seat(root, source, "op-running")
    store_import.ensure_current_store(root)
    _custody(monkeypatch, running=[running], settled=[done])

    prepare_startup_state(root)
    repo, system = _repo(tmp_path / "checkout-subject"), _repo(tmp_path / "checkout-system")
    ctx = SimpleNamespace(repo_dir=str(system), drive_root=str(root), task_id="", task_metadata=None)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(repo), kind="base..head", base="HEAD~1",
                             head="HEAD", governance_root=str(system), surface="change")

    def retained(task_id):
        retry_key = f"review:{task_id}"
        ctx.task_id = task_id
        with isolated_checkout(ctx, spec, retain=lambda: {"seats": ["late seat"]},
                               token=lambda _identity: checkout_token(retry_key, task_id=task_id)) as frozen:
            entry = pathlib.Path(frozen.checkout).parent
        write_task_result(root, task_id, "completed", result="done")
        return entry

    checkout_done = retained("checkout-done")
    checkout_running = retained("checkout-running")
    live_checkouts = {"checkout-running"}
    report = rp.prune_review_checkouts(root, live_checkouts)
    assert report["released"] == 1 and not done.exists() and running.is_dir()
    assert report["checkouts_removed"] == 1 and not checkout_done.exists() and checkout_running.is_dir()
    assert [(hold["class"], hold["count"]) for hold in report["holds"]] == [("run_open", 1), ("task_live", 1)]
    first = record_pass(root, "review_checkout_prune", report, event="review_checkout_prune")
    assert first["released"] == 1 and first["holds"][0]["class"] == "run_open" and first["holds"][1]["class"] == "task_live"
    second = rp.prune_review_checkouts(root, live_checkouts)
    assert record_pass(root, "review_checkout_prune", second, event="review_checkout_prune") is None, \
        "nothing released, the same hold: silent"
    assert "partial" not in second, "every checkout was judged"
    from ouroboros import review_checking as rc
    with rc._seat_locked(running.parent):  # a worker holds a seat: its pass judges no checkout
        busy = rp.prune_review_checkouts(root, live_checkouts)
    assert busy["partial"] is True and checkout_running.is_dir()
    assert record_pass(root, "review_checkout_prune", busy, event="review_checkout_prune") is None
    assert maintenance_jobs.job(root, "review_checkout_prune")["holds"] == first["holds"], \
        "the checkouts' holds stand while a pass could not judge them"


# ---- session scratch ----------------------------------------------------------------------------------------------

@witness("ouroboros.workspace_preflight.release_finished_scratch")
def test_session_scratch_release_releases_a_finished_tasks_scratch_and_holds_a_live_tasks(tmp_path):
    from ouroboros.workspace_preflight import release_finished_scratch
    from tests.test_session_scratch_release import _started

    data, finished = _started(tmp_path, task_id="task-done")
    _, live = _started(tmp_path, task_id="task-live", key="snap-2")
    store_import.ensure_current_store(data)
    report = release_finished_scratch(data, _probe("task-live"))
    assert len(report["released"]) == 1 and [hold["class"] for hold in report["holds"]] and report["errors"] == []
    assert report["holds"][0]["count"] == 1 and "task-live" in report["holds"][0]["example"]
    first = record_pass(data, "session_scratch_release", report, event="session_scratch_release")
    assert first["released"] == 1 and len(first["holds"]) == 1
    second = release_finished_scratch(data, _probe("task-live"))
    assert second["released"] == [] and second["holds"] == report["holds"]
    assert record_pass(data, "session_scratch_release", second, event="session_scratch_release") is None


# ---- engine program copies --------------------------------------------------------------------------------------

@witness("ouroboros.claudexor_runtime_retention.prune_unpinned_runtimes")
def test_engine_runtime_prune_releases_an_unpinned_archive_and_holds_a_tree_it_could_not_remove(cx, tmp_path, monkeypatch):  # noqa: F811
    from ouroboros import claudexor_runtime_retention as retention, force_remove

    root = cx
    store_import.ensure_current_store(tmp_path)
    monkeypatch.setattr(force_remove, "force_rmtree", lambda *_a, **_k: None)  # a tree the platform keeps open
    report = retention.prune_unpinned_runtimes()
    assert "claudexor-3.9.0.tar.gz" in report["removed"] and report["released"] == len(report["removed"]) >= 1
    assert [hold["class"] for hold in report["holds"]] == ["removal_failed"] and report["holds"][0]["count"] >= 1
    assert "survived its removal" in report["holds"][0]["example"] and report["errors"] == []
    assert not (root / "cache" / "claudexor-3.9.0.tar.gz").exists()
    first = record_pass(tmp_path, "engine_runtime_prune", report, event="engine_runtime_prune")
    assert first["released"] == report["released"] and first["holds"][0]["class"] == "removal_failed"
    assert record_pass(tmp_path, "engine_runtime_prune", {**report, "released": 0, "removed": []}, event="engine_runtime_prune") is None


# ---- passes that judged part of their plane, or threw ----------------------------------------------------

def test_drive_settlement_evidence_at_rest_reports_every_retained_verdict(tmp_path, monkeypatch):
    """Both layouts at rest decode no verdicts; a pass that is evidence all the same (a mailbox it removed) reads
    them before it writes, so its holds are the retained drives' verdicts, never the part this pass looked at."""
    from ouroboros import headless, owner_mailbox

    data = tmp_path / "data"
    data.mkdir()
    store_import.ensure_current_store(data)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    verdicts = {"a": {"reason": "child_result_unadopted"}, "b": {"reason": "child_result_unadopted"}}
    monkeypatch.setattr(server_maintenance, "settlement_verdicts", lambda _root: dict(verdicts))
    rested = lambda *args, **kwargs: {"pruned": [], "errors": [], "skipped": [], "cursor": "", "arrivals": [], "rested": True}  # noqa: E731
    monkeypatch.setattr(headless, "prune_headless_task_drives", rested)
    monkeypatch.setattr(headless, "prune_task_drives", rested)
    monkeypatch.setattr(owner_mailbox, "sweep_settled_owner_mailboxes", lambda *_a, **_k: {"removed": ["t-mail"]})
    server_maintenance._run_drive_custody_pass()
    row = maintenance_jobs.job(data, "drive_settlement")
    assert [(hold["class"], hold["count"]) for hold in row["holds"]] == [("child_result_unadopted", 2)] and row["held"] == 2
    del verdicts["b"]  # one retained drive went: the next evidence reads the verdicts as they stand now
    server_maintenance._run_drive_custody_pass()
    assert [(hold["class"], hold["count"]) for hold in maintenance_jobs.job(data, "drive_settlement")["holds"]] == [
        ("child_result_unadopted", 1)]


def test_a_retirer_pass_that_throws_leaves_a_fact_until_one_returns(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    store_import.ensure_current_store(data)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", data)
    monkeypatch.setattr(trees, "prune_orphans", lambda **_kwargs: (_ for _ in ()).throw(OSError("store moved")))
    server_maintenance._prune_worktrees()
    row = maintenance_jobs.job(data, "copy_retirement")
    assert row["last_error"]["class"] == "OSError", "the throw is on the retirer's row, not only in a debug log"
    monkeypatch.setattr(trees, "prune_orphans", lambda **_kwargs: {"removed": [], "kept": []})
    server_maintenance._prune_worktrees()
    row = maintenance_jobs.job(data, "copy_retirement")
    assert "last_error" not in row and row["state"] == "done", "a pass that returns clears it"


def test_every_witness_names_a_test_of_this_module():
    for dotted, name in WITNESSES.items():
        assert name.startswith("test_") and callable(globals()[name]), dotted
    assert len(WITNESSES) == 5
