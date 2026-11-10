"""The one historical walk (#1180 with #2821): old finished reviewer threads archived, verified and released once,
then the engine registrations Ouroboros minted for its own one-shot folders retired through the engine's own removal.

The engines here are in-memory MOCKS of the Claudexor seams (``tests/test_review_thread_retention.Engine`` for threads;
a project registry answering ``GET /v2/projects`` and ``DELETE /v2/projects/:id`` like ``store/projects.ts``: a
non-purged thread or a live run refuses with a typed 409, an unknown id is a 404). The host's own stores, Keep, task
records and receipts are real. Fixture paths are neutral temporary folders."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import delegate_continuation, delegate_custody_current, maintenance_jobs, owner_keep, task_status
from ouroboros import historical_retirement as hr
from ouroboros import review_thread_retention as rt
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.maintenance_retirement import JOB
from tests import _dialogue as dialogue_test
from tests._store_shared import seed_result
from tests.test_delegate_result_taken import owner_chat  # noqa: F401 -- the real owner-chat fixture
from tests.test_review_thread_retention import KEEP, TASK, THREAD, world  # noqa: F401 -- the real thread world
from tests.test_store import document

OLD = "2000-01-01T00:00:00+00:00"


def _walked(root):
    return (maintenance_jobs.job(root, JOB) or {}).get(hr.WALK) or {}


def _upgraded(root):
    """The step as an installation activated it before this walk shipped: its row holds no ``historical_walk``."""
    from ouroboros import store

    with store.open_store(root, write=True) as txn:
        maintenance_jobs.forget_in(txn, JOB, hr.WALK)


def _review_custody(monkeypatch, rows):
    """``delegate_custody_current.runs`` over ROWS (the custody store's own query shapes the walk uses)."""
    def runs(_root, where="", params=()):
        if where == "project_id=?":
            return {row.run_id: row for row in rows if row.project_id == params[0]}
        if where == "project_id<>?":
            return {row.run_id: row for row in rows if row.project_id}
        return {row.run_id: row for row in rows}

    monkeypatch.setattr(delegate_custody_current, "runs", runs)


@pytest.fixture
def old_thread(world, monkeypatch):  # noqa: F811
    """The thread world with its host run older than activation and the task's review seat in custody."""
    world.custody["runs"]["run-1"].started_ts = OLD
    _review_custody(monkeypatch, [RunCustody(run_id="run-1", task_id=TASK, root_task_id=TASK, settled=True,
                                             terminal_state="succeeded", source="review_substrate")])
    _upgraded(world.root)
    return world


def test_the_ordinary_pass_holds_an_old_thread_and_the_walk_archives_then_releases_it_once(old_thread):
    report = old_thread.run()
    release = old_thread.release()
    assert report["holds"]["predates_activation"] == 1, "the ordinary pass never releases older material"
    assert release["state"] == "accepted" and old_thread.engine.purged
    receipt = rt.read_thread_export(old_thread.root, TASK, release["receipt"])
    assert receipt["verified"] and receipt["coverage"] == "complete", "the whole export was kept before release"
    walked = _walked(old_thread.root)
    assert walked["threads"]["accepted"] == 1 and walked["threads"]["asked"] == 1 and walked["done_at"]
    assert walked["waiting"] == [] and walked["task_holds"] == {}
    text = hr.receipt(walked)
    assert "1 finished reviewer session was archived" in text and "cannot be resumed" in text
    assert [entry for entry in report["observed"] if hr.WALK in entry] == [{hr.WALK: text}]
    calls = list(old_thread.engine.calls)
    old_thread.touch()
    old_thread.run()
    old_thread.run()
    assert [call for call in old_thread.engine.calls[len(calls):] if call[0] in ("export", "retire")] == [], (
        "an accepted thread is never exported again, and the walk is over")


def test_a_kept_task_is_held_once_and_never_waits(old_thread):
    owner_keep.set_owner_keep(old_thread.root, TASK, keep=True, source=KEEP)
    old_thread.run()
    walked = _walked(old_thread.root)
    assert not old_thread.release() and not old_thread.engine.purged
    assert walked["task_holds"] == {TASK: ["owner_keep"]} and walked["waiting"] == [] and walked["done_at"]
    assert "sessions of 1 task stay as they are" in hr.receipt(walked)


def test_work_not_over_waits_in_the_walk_and_is_released_with_the_cutoff_lifted_when_it_ends(old_thread):
    old_thread.continuable["value"] = True
    old_thread.run()
    walked = _walked(old_thread.root)
    assert walked["waiting"] == [TASK] and walked["task_holds"] == {TASK: ["continuable"]}
    assert hr.owed(maintenance_jobs.job(old_thread.root, JOB)), "the step owes the waiting task"
    assert "1 task with sessions waits for work that is not over and is released" in hr.receipt(walked)
    old_thread.continuable["value"] = False
    report = old_thread.run()  # no new terminal write: the walk's own waiting set asks it, historically
    assert old_thread.release()["state"] == "accepted" and report["accepted"] == 1
    walked = _walked(old_thread.root)
    assert walked["waiting"] == [] and walked["task_holds"] == {} and walked["threads"]["accepted"] == 1
    assert not hr.owed(maintenance_jobs.job(old_thread.root, JOB))


def test_a_partial_export_is_held_with_its_receipt_and_nothing_is_released(old_thread):
    old_thread.engine.coverage, old_thread.engine.gaps = "partial", [{"code": "native_unreadable"}]
    old_thread.run()
    release = old_thread.release()
    assert release["state"] == "held" and release["holds"] == ["export_partial"] and not old_thread.engine.purged
    assert not [call for call in old_thread.engine.calls if call[0] in ("item", "retire")]
    assert _walked(old_thread.root)["task_holds"] == {TASK: ["export_partial"]}


def test_a_stop_keeps_the_walk_open_at_its_cursor(old_thread):
    rt.retire_finished_review_threads(
        old_thread.root, engine=old_thread.engine, live=lambda _t: False, census=set(), should_stop=lambda: True)
    assert not old_thread.release() and not _walked(old_thread.root).get("threads_done_at")
    old_thread.run()
    assert old_thread.release()["state"] == "accepted" and _walked(old_thread.root)["done_at"]


# ---- registrations ------------------------------------------------------------------------------------------


class Registry:
    """``GET /v2/projects`` (strict reader) and ``DELETE /v2/projects/:id`` as the engine answers them (projects.ts:
    a random id per registration, a typed 409 or a 404 ``project_not_found`` refusal); thread routes unused."""

    def __init__(self, projects):
        self.store_identity, self.projects, self.deleted, self.refuse = "store-1", dict(projects), [], {}
        self.listed, self.lost, self.malformed = 0, [], False
        self.lose_after_effect = set()  # removed by the engine, but the answer never reaches the host
        self.bad_receipt = set()  # an answer that names no removal
        self.linger = {}  # id -> root the list still shows after a removal (a readback mismatch)

    def thread_retirement_supported(self):
        return True

    def project_registrations(self):
        self.listed += 1
        if self.malformed:
            raise ClaudexorUnavailable("malformed_response", "Claudexor project list is malformed")
        rows = {**self.linger, **self.projects}
        return [{"id": pid, "root": root} for pid, root in sorted(rows.items())]

    def remove_project(self, project_id):
        self.deleted.append(project_id)
        if self.lost:
            raise self.lost.pop(0)
        if project_id in self.refuse:
            raise ClaudexorUnavailable(self.refuse[project_id], "refused", status_code=409)
        if project_id not in self.projects:
            raise ClaudexorUnavailable("project_not_found", "no such project", status_code=404)
        root = self.projects.pop(project_id)
        if project_id in self.lose_after_effect:
            raise ClaudexorUnavailable("daemon_unreachable", "answer lost", status_code=0)
        if project_id in self.bad_receipt:
            return {}
        return {"projectId": project_id, "root": root, "registryRemoved": True, "journalPartitionArchived": True,
                "archivedPartitionPath": f"partition:{project_id}@1", "artifactsRetained": True}

    def close(self):
        pass


LOST = ClaudexorUnavailable("daemon_unreachable", "lost", status_code=0)


@pytest.fixture
def registrations(tmp_path, monkeypatch):
    """An upgraded install on a real store; custody rows name engine projects bound to store-1; real task records."""
    root = tmp_path / "data"
    (root / "state" / "review_checkouts").mkdir(parents=True)
    rows, continuable, waits = [], {"value": False}, set()
    _review_custody(monkeypatch, rows)
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: continuable["value"])
    monkeypatch.setattr(task_status, "task_subtree_has_live_queue_rows", lambda _root, _task: False)
    task = rt._Pass.task
    monkeypatch.setattr(rt._Pass, "task", lambda self, task_id, historical=False: rt.WAIT if task_id in waits
                        else task(self, task_id, historical))

    def minted(name):
        folder = root / "state" / "review_checkouts" / name
        folder.mkdir()
        return str(folder)

    def custody(run_id, project_id, task="task-r1", status="completed", **facts):
        seed_result(root, task, document(task, status))
        rows.append(RunCustody(run_id=run_id, task_id=task, root_task_id=task, project_id=project_id, settled=True,
                               terminal_state="succeeded", **{"daemon_store_id": "store-1", **facts}))

    engine = Registry({})
    rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set())  # activation
    _upgraded(root)
    install = SimpleNamespace(root=root, engine=engine, rows=rows, minted=minted, custody=custody,
                              continuable=continuable, waits=waits)
    install.run = lambda **kw: rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False,
                                                                 census=kw.get("census", set()))
    return install


def _outcomes(report):
    return {entry["historical_registration"]["project_id"]: entry["historical_registration"]
            for entry in report["observed"] if "historical_registration" in entry}


def test_only_proven_minted_registrations_of_this_store_whose_work_ended_are_removed(registrations, tmp_path,
                                                                                     monkeypatch):
    from ouroboros import delegate_custody

    owner_repo = tmp_path / "owner-repo"
    owner_repo.mkdir()
    r = registrations
    r.engine.projects.update({"p-minted": r.minted("c1"), "p-owner": str(owner_repo), "p-foreign": r.minted("c2"),
                              "p-persistent": r.minted("c3"), "p-owned": r.minted("c4"), "p-open": r.minted("c5"),
                              "p-unbound": r.minted("c6"), "p-other": r.minted("c7"), "p-moved": r.minted("c8")})
    r.custody("run-a", "p-minted")
    r.custody("run-b", "p-owner")  # a delegated run on the owner's own project: never ours to retire
    r.custody("run-c", "p-persistent", project_persistent=True)
    r.custody("run-d", "p-owned", project_owned=True)  # still owned: the ordinary sweep's
    r.custody("run-e", "p-open", task="task-r2")
    r.rows[-1].settled = False
    r.custody("run-f", "p-unbound", daemon_store_id="")  # a legacy run never bound to a store
    r.custody("run-g", "p-other", daemon_store_id="store-2")  # the same id another store named
    r.custody("run-h", "p-moved", invocation_id="inv-h")  # its request registered another folder
    elsewhere = r.minted("c9")
    monkeypatch.setattr(delegate_custody, "invocation_record", lambda _root, inv, **_kw: {
        "request": {"scope": {"kind": "project", "root": elsewhere}}} if inv == "inv-h" else None)
    report = r.run()
    outcomes = _outcomes(report)
    assert r.engine.deleted == ["p-minted"], "nothing custody does not prove ours, ended and of this store goes"
    assert outcomes["p-minted"]["outcome"] == "removed" and outcomes["p-minted"]["receipt"]["artifactsRetained"]
    assert {pid: (o["outcome"], o.get("code")) for pid, o in outcomes.items() if pid != "p-minted"} == {
        "p-owner": ("held", "ownership_unproven"), "p-persistent": ("held", "persistent"),
        "p-owned": ("held", "custody_owned"), "p-open": ("held", "run_unsettled"),
        "p-unbound": ("held", "store_unproven"), "p-other": ("held", "store_unproven"),
        "p-moved": ("held", "root_mismatch")}
    assert "p-foreign" not in outcomes, "a project no custody row names is not a candidate at all"
    walked = _walked(r.root)
    assert walked["registrations"] == {"candidates": 8, "removed": 1, "absent": 0, "kept": {}, "listed_after_removal": 0,
                                       "held": {"ownership_unproven": 1, "persistent": 1, "custody_owned": 1,
                                                "run_unsettled": 1, "store_unproven": 2, "root_mismatch": 1}}
    assert walked["done_at"] and "removed_ids" not in walked and "pending" not in walked
    text = hr.receipt(walked)
    assert "1 engine project Ouroboros had registered for its own temporary folders was removed" in text
    assert "7 projects stay registered" in text and "not bound to this engine store" in text
    listed, deleted = r.engine.listed, list(r.engine.deleted)
    r.run()
    r.run()
    assert (r.engine.listed, r.engine.deleted) == (listed, deleted), "once: an over walk asks the engine nothing"


def test_the_engines_typed_answers_decide_and_a_kept_or_live_task_holds(registrations):
    r = registrations
    r.engine.projects.update({"p-threads": r.minted("c1"), "p-run": r.minted("c2"), "p-kept": r.minted("c3"),
                              "p-live": r.minted("c4"), "p-gone": r.minted("c5")})
    r.engine.refuse.update({"p-threads": "project_has_threads", "p-run": "project_has_active_run"})
    r.custody("run-a", "p-threads")
    r.custody("run-b", "p-run")
    r.custody("run-c", "p-kept", task="task-kept")
    owner_keep.set_owner_keep(r.root, "task-kept", keep=True, source=KEEP)
    r.custody("run-d", "p-unlisted", task="task-gone")  # the engine no longer lists it: not asked
    r.custody("run-e", "p-live", task="task-live", status="running")
    r.custody("run-f", "p-gone")
    r.engine.linger["p-gone"] = r.engine.projects.pop("p-gone")  # listed (its native minted path), then 404 on DELETE
    r.run()
    walked = _walked(r.root)["registrations"]
    assert walked["kept"] == {"project_has_threads": 1, "project_has_active_run": 1} and walked["absent"] == 1
    assert walked["held"] == {"owner_keep": 1, "task_not_proven_over": 1} and walked["removed"] == 0
    assert sorted(r.engine.deleted) == ["p-gone", "p-run", "p-threads"]
    text = hr.receipt(_walked(r.root))
    assert "the engine kept 2 projects" in text and "still hold sessions" in text and "already gone" in text


@pytest.mark.parametrize("change", ["keep", "persistent", "continuable"])
def test_a_proof_lost_between_the_check_and_the_effect_sends_nothing(registrations, monkeypatch, change):
    """Keep, persistence or a Continue offer that lands after the first check and before the removal: every proof
    is made again under the project lock, and Keep's own gate is held across the effect."""
    from ouroboros import delegate_custody_usage

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    lock = delegate_custody_usage.project_retirement_lock

    def racing(root, project_id):
        if change == "keep":
            owner_keep.set_owner_keep(root, "task-r1", keep=True, source=KEEP)
        elif change == "persistent":
            r.rows[0].project_persistent = True
        else:
            r.continuable["value"] = True
        return lock(root, project_id)

    monkeypatch.setattr(delegate_custody_usage, "project_retirement_lock", racing)
    r.run()
    expected = {"keep": "owner_keep", "persistent": "persistent", "continuable": "continuable"}[change]
    assert r.engine.deleted == [] and _walked(r.root)["registrations"]["held"] == {expected: 1}


def test_keep_saved_while_the_gate_check_runs_is_read_inside_the_gate(registrations, monkeypatch):
    """The last check before the request is Keep's own gate: a Keep saved after every other proof still stops it."""
    from ouroboros import owner_keep as keep_module

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    gate = keep_module.gate

    def gate_after_keep(root, task_id, **kw):
        keep_module.set_owner_keep(root, task_id, keep=True, source=KEEP)
        return gate(root, task_id, **kw)

    monkeypatch.setattr(keep_module, "gate", gate_after_keep)
    r.run()
    assert r.engine.deleted == [] and _walked(r.root)["registrations"]["held"] == {"owner_keep": 1}


def test_a_busy_task_gate_sends_nothing_and_holds_after_a_second_pass(registrations):
    from ouroboros.task_custody import task_custody_lock

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    with task_custody_lock(r.root, "task-r1", timeout_sec=5) as locked:  # another holder of the task's custody
        assert locked
        r.run()
        assert r.engine.deleted == [] and not _walked(r.root).get("done_at")
        r.run()
    assert r.engine.deleted == [] and _walked(r.root)["registrations"]["held"] == {"busy": 1}


def test_a_malformed_list_proves_nothing_and_a_valid_empty_list_is_an_answer(registrations):
    r = registrations
    r.custody("run-a", "p-1")
    r.engine.malformed = True
    report = r.run()
    assert report["holds"].get("historical_projects_unread") == 1 and not _walked(r.root).get("done_at")
    r.engine.malformed = False
    r.run()  # valid and empty: custody's project is not in this store's list
    assert _walked(r.root)["done_at"] and r.engine.deleted == [] and _walked(r.root)["registrations"]["candidates"] == 0


def test_an_unverified_receipt_is_unknown_and_reconciled_from_the_list(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.bad_receipt.add("p-1")  # the engine removed it but answered with no receipt
    r.run()
    walked = _walked(r.root)
    assert walked["pending"]["project_id"] == "p-1" and walked["registrations"]["removed"] == 0
    r.run()
    walked = _walked(r.root)
    assert r.engine.deleted == ["p-1"], "an unknown effect is never repeated before the list answers"
    assert walked["registrations"]["gone_after_unknown"] == 1 and walked["registrations"]["removed"] == 0
    assert "whether that request removed it is unknown" in hr.receipt(walked) and walked["done_at"]


def test_a_lost_answer_after_the_effect_is_gone_after_unknown_never_removed(registrations):
    r = registrations
    folder = r.minted("c1")  # the native spelling the host minted and the engine lists, recorded as it is
    r.engine.projects["p-1"] = folder
    r.custody("run-a", "p-1")
    r.engine.lose_after_effect.add("p-1")
    r.run()
    assert _walked(r.root)["pending"] == {"project_id": "p-1", "root": folder, "store_id": "store-1"}
    r.run()
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["registrations"]["gone_after_unknown"] == 1


def test_a_still_listed_project_after_a_lost_answer_is_held_not_asked_again(registrations):
    r = registrations
    r.engine.projects.update({"p-1": r.minted("c1"), "p-2": r.minted("c2")})
    r.custody("run-a", "p-1")
    r.custody("run-b", "p-2")
    r.engine.lost = [LOST]
    r.run()
    listed = r.engine.listed
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["pending"]["project_id"] == "p-1"
    r.engine.linger["p-2"] = r.engine.projects["p-2"]  # removed, yet the next listing still names it
    r.run()  # a read can precede the first request's late effect; it does not prove non-execution
    walked = _walked(r.root)["registrations"]
    assert r.engine.deleted == ["p-1", "p-2"] and r.engine.listed > listed
    assert walked["held"] == {"engine_unanswered": 1}
    assert walked["removed"] == 1 and walked["listed_after_removal"] == 1
    assert "still appeared in the engine's list" in hr.receipt(_walked(r.root))
    r.run()
    assert r.engine.deleted == ["p-1", "p-2"], "the one-time walk never replays the unanswered request"


def test_a_404_that_is_not_the_projects_own_absence_proves_nothing(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [ClaudexorUnavailable("route_not_found", "no such route", status_code=404)]
    r.run()
    walked = _walked(r.root)
    assert walked["pending"]["project_id"] == "p-1" and walked["registrations"]["absent"] == 0
    r.run()  # the same root does not prove that the original request has finished without effect
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["registrations"]["removed"] == 0
    assert _walked(r.root)["registrations"]["held"] == {"engine_unanswered": 1}


def test_a_root_changed_after_a_lost_answer_is_held_without_a_second_request(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    r.engine.projects["p-1"] = r.minted("c-relinked")  # the engine relinked the same id to another folder
    r.run()
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["registrations"]["held"] == {"root_changed": 1}


def test_another_stores_empty_list_never_resolves_the_old_unknown_request(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    assert _walked(r.root)["pending"]["store_id"] == "store-1"
    r.engine.store_identity = "store-2"
    r.engine.projects.clear()
    r.run()
    counts = _walked(r.root)["registrations"]
    assert counts.get("gone_after_unknown", 0) == 0
    assert counts["held"] == {"store_changed": 1} and r.engine.deleted == ["p-1"]


def test_failed_final_readback_remains_owed_after_custody_rows_are_gone(registrations, monkeypatch):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    reader = r.engine.project_registrations
    calls = []

    def readback_fails_once():
        calls.append(True)
        if len(calls) == 2:
            raise LOST
        return reader()

    monkeypatch.setattr(r.engine, "project_registrations", readback_fails_once)
    r.run()
    assert _walked(r.root)["removed_ids"] == ["p-1"]
    assert hr.owed(maintenance_jobs.job(r.root, JOB))
    r.rows.clear()
    r.run()
    assert "removed_ids" not in _walked(r.root)
    assert not hr.owed(maintenance_jobs.job(r.root, JOB))
    assert r.engine.deleted == ["p-1"], "readback is retried, never the successful deletion"


def test_a_project_kept_by_the_walks_own_pending_release_is_deferred_then_retired(registrations, monkeypatch):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1", task="task-slow")
    custody_runs, scans = delegate_custody_current.runs, []
    monkeypatch.setattr(delegate_custody_current, "runs", lambda root, where="", params=(): (
        scans.append(where), custody_runs(root, where, params))[1])
    maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"threads": {}, "task_holds": {}, "waiting": ["task-slow"],
                                                      "threads_done_at": "2026-11-09T00:00:00+00:00"}})
    r.waits.add("task-slow")  # its admitted thread release still waits
    report = r.run()
    walked = _walked(r.root)
    assert r.engine.deleted == [] and walked["deferred"] == {"p-1": ["task-slow"]} and walked["done_at"]
    text = [entry[hr.WALK] for entry in report["observed"] if hr.WALK in entry][0]
    assert "1 project waits until its sessions' release settles and is not cleaned up yet" in text
    assert "removed through the engine" not in text, "no completed cleanup is claimed for it"
    assert hr.owed(maintenance_jobs.job(r.root, JOB))
    listed, scans[:] = r.engine.listed, []
    r.run()
    assert r.engine.deleted == [], "still waiting: nothing is asked"
    assert r.engine.listed == listed and "project_id<>?" not in scans, "an unchanged wait reads no custody or engine list"
    r.waits.discard("task-slow")
    r.run()  # the release settled: the walk asks its deferred project, nothing else
    walked = _walked(r.root)
    assert r.engine.deleted == ["p-1"] and walked["deferred"] == {} and walked["registrations"]["removed"] == 1
    assert not hr.owed(maintenance_jobs.job(r.root, JOB))
    r.custody("run-new", "p-new")
    r.engine.projects["p-new"] = r.minted("c-new")
    r.run()
    assert r.engine.deleted == ["p-1"], "a registration that appears after the walk is never touched by it"


def test_an_unknown_outcome_left_by_the_walk_is_counted_once_whoever_settles_it(registrations, monkeypatch):
    """The deferred wait above, then an unanswered removal: the next pass reads the list (no resend)."""
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1", task="task-slow")
    maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"threads": {}, "task_holds": {}, "waiting": ["task-slow"],
                                                      "threads_done_at": "2026-11-09T00:00:00+00:00"}})
    r.waits.add("task-slow")
    r.run()
    r.waits.discard("task-slow")
    r.engine.lose_after_effect.add("p-1")
    r.run()
    assert _walked(r.root)["pending"]["project_id"] == "p-1" and r.engine.deleted == ["p-1"]
    listed = r.engine.listed
    r.run()  # the unknown intent still progresses after done_at: read back, never sent again
    assert r.engine.listed > listed and r.engine.deleted == ["p-1"]
    assert _walked(r.root)["registrations"]["gone_after_unknown"] == 1 and "pending" not in _walked(r.root)


@pytest.mark.parametrize("settler", ["ordinary_pass", "the_walk"])
def test_an_acceptance_the_walk_left_bound_is_counted_once_whoever_reads_it(old_thread, settler):
    """The walk's release answer is lost (``bound``: requested, not accepted). The step's ordinary pass may read the
    operation's acceptance first (a later terminal write of the task) or the walk's own ask does; either way it is
    the walk's one acceptance, counted once across passes, and the owner's receipt names it."""
    old_thread.engine.faults.add("lost_after")  # the engine accepted; the answer never arrived
    old_thread.run()
    walked = _walked(old_thread.root)
    assert old_thread.release()["state"] == "bound" and walked["waiting"] == [TASK]
    assert old_thread.release()["historical"] is True, "the release says itself that this walk made it"
    assert walked["threads"]["accepted"] == 0 and walked["threads"]["unknown"] == 1
    assert "archived" not in hr.receipt(walked), "a bound request is no acceptance"
    if settler == "ordinary_pass":
        old_thread.touch()
    report = old_thread.run()
    assert old_thread.release()["state"] == "accepted" and report["accepted"] == 1
    walked = _walked(old_thread.root)
    assert walked["threads"]["accepted"] == 1 and walked["waiting"] == [] and "counted" not in walked
    assert "1 finished reviewer session was archived" in hr.receipt(walked)
    for _ in range(2):  # later passes and the step's restarts read the same records: no second count, no resend
        old_thread.touch()
        old_thread.run()
    assert _walked(old_thread.root)["threads"]["accepted"] == 1
    assert [call for call in old_thread.engine.calls if call[0] == "retire"] == [call for call in old_thread.engine.calls
                                                                                if call[0] == "retire"][:1]


def test_a_crash_after_the_release_settled_recovers_the_count_from_the_release_record(old_thread, monkeypatch):
    """The engine accepted and the task record says so, then the process dies before the step's job row lands. The next
    pass re-asks the task from the persisted cursor: the attributed release is counted once, the owner's receipt names
    it, and the engine is never asked to retire that thread again."""
    real = maintenance_jobs.record

    def died(root, job, **fields):
        if hr.WALK in fields:
            raise RuntimeError("the process died before the job row landed")
        return real(root, job, **fields)

    monkeypatch.setattr(maintenance_jobs, "record", died)
    with pytest.raises(RuntimeError, match="died before the job row"):
        old_thread.run()
    assert old_thread.release()["state"] == "accepted" and old_thread.release()["historical"] is True
    assert hr.WALK not in (maintenance_jobs.job(old_thread.root, JOB) or {}), "nothing of the walk's pass landed"
    retires = [call for call in old_thread.engine.calls if call[0] == "retire"]
    assert len(retires) == 1
    monkeypatch.setattr(maintenance_jobs, "record", real)
    report = old_thread.run()  # the restarted maintenance child: same cursor, the attributed record answers
    assert report["accepted"] == 0, "no new settlement happened"
    walked = _walked(old_thread.root)
    assert walked["threads"]["accepted"] == 1 and walked["threads"]["asked"] == 1 and walked["done_at"], walked
    assert "1 finished reviewer session was archived" in hr.receipt(walked)
    assert [call for call in old_thread.engine.calls if call[0] == "retire"] == retires, "no second deletion request"
    old_thread.touch()
    old_thread.run()
    assert _walked(old_thread.root)["threads"]["accepted"] == 1, "counted once, however often it is read back"


def test_a_waiting_tasks_counted_release_is_not_counted_again_when_it_is_asked_again(monkeypatch):
    """A task still waiting keeps the operations already counted: each ask adds only what is new."""
    class Ask:
        stopped, root = False, "data-root"

        def __init__(self, answers):
            self.answers, self.report, self.walk_states = answers, {"holds": {}, "errors": [], "observed": []}, {}

        def task(self, _task, historical=False):
            verdict, self.walk_states = self.answers.pop(0)
            return verdict

    monkeypatch.setattr(hr, "_walk_states", lambda _root, _task: {})  # the records read before the ask name nothing settled
    run, record = Ask([("wait", {"op-1": "accepted"}), ("wait", {"op-1": "accepted", "op-2": "refused"}),
                       ("done", {"op-1": "accepted", "op-2": "refused"})]), hr._copy({})
    assert hr._ask(run, record, "t-1") == "wait"
    assert record["threads"]["accepted"] == 1 and record["counted"] == {"t-1": ["op-1"]}
    assert hr._ask(run, record, "t-1") == "wait"
    assert (record["threads"]["accepted"], record["threads"]["refused"]) == (1, 1)
    assert record["counted"] == {"t-1": ["op-1", "op-2"]}
    assert hr._ask(run, record, "t-1") == "done"
    assert (record["threads"]["accepted"], record["threads"]["refused"]) == (1, 1) and "counted" not in record


def test_the_walks_count_reads_only_its_own_releases_from_the_task_records(monkeypatch):
    """``historical`` on the record is the attribution: an ordinary pass's settled release is not the walk's."""
    from ouroboros import task_store

    records = {"thread-1": {"operation_id": "op-1", "state": "accepted", "historical": True},
               "thread-2": {"operation_id": "op-2", "state": "accepted"}, "thread-3": "not a record",
               "thread-4": {"state": "accepted", "historical": True}}
    monkeypatch.setattr(task_store, "read_document", lambda _root, _task: {rt.RELEASES: records})
    assert hr._walk_states("data-root", "t-1") == {"op-1": "accepted"}
    monkeypatch.setattr(task_store, "read_document", lambda _root, _task: {rt.RELEASES: "not a mapping"})
    assert hr._walk_states("data-root", "t-1") == {}


def test_an_ordinary_passes_release_the_walk_happens_to_settle_is_not_the_walks_count(world, monkeypatch):  # noqa: F811
    """The ordinary pass bound the release of a thread younger than activation and lost its answer. The walk asks the
    same task and reads the settlement; the receipt counts only what this walk released."""
    world.custody["runs"]["run-1"].source = "review_substrate"
    _review_custody(monkeypatch, [RunCustody(run_id="run-1", task_id=TASK, root_task_id=TASK, settled=True,
                                             terminal_state="succeeded", source="review_substrate")])
    _upgraded(world.root)
    world.engine.faults.add("lost_after")
    world.run()  # the ordinary pass binds it and loses the answer; the walk's own ask reads the settlement
    release = world.release()
    assert release["state"] in ("bound", "accepted") and "historical" not in release
    world.touch()
    world.run()
    assert world.release()["state"] == "accepted"
    walked = _walked(world.root)
    assert int(walked["threads"].get("accepted") or 0) == 0, walked
    assert "archived" not in hr.receipt(walked), "the walk did not release it"


def test_a_saved_acceptance_needs_no_second_task_read_to_reach_the_receipt(old_thread, monkeypatch):
    from ouroboros import task_store

    reconcile, read = rt._Pass.reconcile, task_store.read_document
    accepted = []

    def settle(self, task_id, record, **kwargs):
        result = reconcile(self, task_id, record, **kwargs)
        if task_id == TASK and self.report["accepted"]:
            accepted.append(record["operation_id"])

            def temporarily_unreadable(root, target, *args, **kwargs):
                if target == TASK:
                    raise OSError("task read failed after the acceptance was saved")
                return read(root, target, *args, **kwargs)

            monkeypatch.setattr(task_store, "read_document", temporarily_unreadable)
        return result

    monkeypatch.setattr(rt._Pass, "reconcile", settle)
    report = old_thread.run()
    assert accepted and report["accepted"] == 1
    assert _walked(old_thread.root)["threads"]["accepted"] == 1
    monkeypatch.setattr(task_store, "read_document", read)
    assert old_thread.release()["state"] == "accepted"
    old_thread.run()
    assert _walked(old_thread.root)["threads"]["accepted"] == 1
    assert len([call for call in old_thread.engine.calls if call[0] == "retire"]) == 1


def test_custody_without_a_readable_request_proves_no_root(registrations, monkeypatch):
    from ouroboros import delegate_custody

    r = registrations
    named = {"p-gap": "inv-gap", "p-none": "inv-none", "p-unread": "inv-unread", "p-thread": "inv-thread",
             "p-match": "inv-match"}
    for index, (project, invocation) in enumerate(sorted(named.items())):
        r.engine.projects[project] = r.minted(f"c{index}")
        r.custody(f"run-{index}", project, invocation_id=invocation)
    records = {"inv-gap": {"state": "legacy_gap", "reason": "no_request_row"},
               "inv-unread": {"state": "started", "request": None},  # its body reference could not be read
               "inv-thread": {"state": "started", "request": {"_thread_id": "thr-9"}},  # a thread turn: no scope
               "inv-match": {"state": "started", "request": {"scope": {"kind": "project",
                                                                       "root": r.engine.projects["p-match"]}}}}
    monkeypatch.setattr(delegate_custody, "invocation_record", lambda _root, inv, **_kw: records.get(inv))
    r.run()
    assert sorted(r.engine.deleted) == ["p-match", "p-thread"]
    assert _walked(r.root)["registrations"]["held"] == {"custody_unreadable": 3}


def test_an_intent_that_was_not_written_sends_nothing_and_leaves_no_unknown(registrations, monkeypatch):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    write, failed = maintenance_jobs.record, []

    def record(root, job, **fields):
        if not failed and (fields.get(hr.WALK) or {}).get("pending") and rt.JOB_FIELD not in fields:
            failed.append(True)
            raise OSError("disk full")  # the intent write alone fails; the pass's final save succeeds
        return write(root, job, **fields)

    monkeypatch.setattr(maintenance_jobs, "record", record)
    report = r.run()
    walked = _walked(r.root)
    assert failed and r.engine.deleted == [] and "pending" not in walked and report["errors"]
    r.run()  # healthy again: the candidate is asked anew, its intent written, one removal sent
    walked = _walked(r.root)
    assert r.engine.deleted == ["p-1"] and walked["registrations"]["removed"] == 1
    assert walked["registrations"].get("gone_after_unknown", 0) == 0 and walked["done_at"]


def test_a_crash_after_the_written_intent_stays_unknown_and_is_never_resent(registrations, monkeypatch):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")

    def crash(project_id):
        r.engine.deleted.append(project_id)
        r.engine.projects.pop(project_id)
        raise RuntimeError("the pass died with the request in flight")

    monkeypatch.setattr(r.engine, "remove_project", crash)
    r.run()
    assert _walked(r.root)["pending"]["project_id"] == "p-1"
    r.run()
    counts = _walked(r.root)["registrations"]
    assert r.engine.deleted == ["p-1"] and counts["gone_after_unknown"] == 1 and counts["removed"] == 0


def test_a_run_started_after_the_check_holds_under_the_retirement_lock(registrations, monkeypatch):
    from ouroboros import delegate_custody_usage

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    lock = delegate_custody_usage.project_retirement_lock

    def racing(root, project_id):  # a sibling's STARTED lands between the decision and the lock
        r.rows.append(RunCustody(run_id="run-late", task_id="task-r9", project_id=project_id, settled=True,
                                 terminal_state="succeeded", daemon_store_id="store-1"))  # only its identity shows it
        return lock(root, project_id)

    monkeypatch.setattr(delegate_custody_usage, "project_retirement_lock", racing)
    r.run()
    assert r.engine.deleted == [] and _walked(r.root)["registrations"]["held"] == {"custody_changed": 1}


def test_no_census_waits_without_asking_the_engine(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.run(census=None)
    assert r.engine.deleted == [] and not _walked(r.root).get("done_at")
    r.run()
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["done_at"]


def test_an_unreadable_private_copy_record_proves_nothing(registrations, tmp_path, monkeypatch):
    from ouroboros import private_copy_read

    r = registrations
    copy = tmp_path / "copies" / "dlg-1"
    copy.mkdir(parents=True)
    r.engine.projects.update({"p-copy": str(copy), "p-unknown": str(tmp_path)})
    r.custody("run-a", "p-copy")
    r.custody("run-b", "p-unknown")
    answers = {str(copy): "registered", str(tmp_path): "unknown"}
    monkeypatch.setattr(private_copy_read, "checkout_registration", lambda path, _data=None: answers.get(str(path), "absent"))
    r.run()
    assert r.engine.deleted == ["p-copy"] and _walked(r.root)["registrations"]["held"] == {"ownership_unproven": 1}


def test_an_install_with_nothing_delegated_closes_the_walk_at_activation(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    engine = Registry({"p-1": str(tmp_path)})
    assert rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set()) == {
        "activated": True}
    assert _walked(root)["done_at"] and not hr.owed(maintenance_jobs.job(root, JOB))


def test_an_upgraded_install_whose_custody_names_nothing_closes_the_walk_without_the_engine(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    engine = Registry({"p-1": str(tmp_path)})
    rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set())
    _upgraded(root)
    assert hr.owed(maintenance_jobs.job(root, JOB)), "an activated step without the walk's record has it open"
    report = rt.retire_finished_review_threads(root, engine=engine, live=lambda _t: False, census=set())
    assert _walked(root)["done_at"] and engine.listed == 0 and engine.deleted == []
    assert hr.receipt(_walked(root)) == "" and not [e for e in report["observed"] if hr.WALK in e]


def test_the_receipt_reaches_the_owners_chat_once(owner_chat):  # noqa: F811
    record = {"threads": {"accepted": 3}, "waiting": [], "task_holds": {"t-1": ["owner_keep"]},
              "registrations": {"removed": 40, "absent": 2, "kept": {"project_has_threads": 1},
                                "gone_after_unknown": 1, "held": {"ownership_unproven": 5}, "listed_after_removal": 0}}
    maintenance_jobs.record(owner_chat.root, JOB, **{hr.WALK: record})
    owner_chat.child_ran()

    def notices():
        return [row for row in dialogue_test.chat_rows(owner_chat.root) if row.get("type") == hr.NOTICE]

    assert notices() == [], "not over yet: nothing to tell"
    maintenance_jobs.record(owner_chat.root, JOB, **{hr.WALK: {**record, "done_at": "2026-11-09T00:00:00+00:00"}})
    owner_chat.child_ran()
    owner_chat.child_ran()
    [row] = notices()
    assert row["direction"] == "system" and row["chat_id"] == 7
    for fact in ("3 finished reviewer sessions were archived", "cannot be resumed", "sessions of 1 task stay",
                 "40 engine projects", "kept their artifacts", "2 such projects were already gone",
                 "the engine kept 1 project (1 because they still hold sessions)",
                 "5 projects stay registered (5 because Ouroboros could not prove it registered them)",
                 "1 project disappeared after a request whose answer was lost"):
        assert fact in row["text"], (fact, row["text"])
    assert "project_has" not in row["text"], "engine codes reach the owner in words"
