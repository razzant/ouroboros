"""The one historical walk (#1180 with #2821): old finished reviewer threads archived, verified and released once,
then the engine registrations Ouroboros minted for its own one-shot folders retired through the engine's own removal.

The engines here are in-memory MOCKS of the Claudexor seams (``tests/test_review_thread_retention.Engine`` for threads;
a project registry with a strict listing, bound inspection and release, and typed refusals). The host's own stores, Keep, task
records and receipts are real. Fixture paths are neutral temporary folders."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import delegate_continuation, delegate_custody_current, maintenance_jobs, owner_keep, task_status
from ouroboros import historical_retirement as hr
from ouroboros import review_thread_retention as rt
from ouroboros.delegate_custody import RunCustody
from ouroboros.delegate_custody_current import runs as stored_runs
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.maintenance_retirement import JOB
from tests import _dialogue as dialogue_test
from tests._bound_registration import BoundRegistry, binding
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
    def runs(_root, where="", params=(), *, transaction=None):
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


class Registry(BoundRegistry):
    """An in-memory registry with guarded release, addressed reads and independent list snapshots."""

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

    def registered_root(self, project_id):
        return self.projects.get(project_id)

    def release_registered(self, project_id):
        self.deleted.append(project_id)
        if self.lost:
            raise self.lost.pop(0)
        if project_id in self.refuse:
            raise ClaudexorUnavailable(self.refuse[project_id], "refused", status_code=409)
        if project_id not in self.projects:
            return "absent"
        self.projects.pop(project_id)
        if project_id in self.lose_after_effect:
            raise ClaudexorUnavailable("daemon_unreachable", "answer lost", status_code=0)
        if project_id in self.bad_receipt:
            return {}
        return "released"

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
        if project_id in engine.projects and not facts.get("invocation_id"):
            facts.setdefault("project_resolution", {**binding("store-1", engine.projects[project_id]),
                                                    "registration_created": True})
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


def _case_variant(folder):
    """The same folder spelled with the first character of its name in the other case."""
    path = Path(folder)
    flipped = path.name[0].swapcase() + path.name[1:]
    assert path.name and flipped != path.name, "a minted folder name has a flippable first character"
    return str(path.with_name(flipped))


def test_a_recorded_root_spelling_the_folder_in_another_case_is_not_a_mismatch(registrations, monkeypatch):
    """The engine's list may spell a folder in another case (a Windows drive letter) than the creator's receipt,
    which names the engine's own record: the walk retires it, asking and releasing at the receipt's exact root,
    never the list's spelling; only a genuinely different folder is kept as root_mismatch."""
    import os

    r = registrations
    folder, other, elsewhere = r.minted("c1"), r.minted("c2"), r.minted("c3")
    receipt = _case_variant(folder)
    r.engine.projects.update({"p-1": receipt, "p-2": other})  # the engine's records, as its POST answered
    r.custody("run-a", "p-1", project_resolution={**binding("store-1", receipt), "registration_created": True})
    r.custody("run-b", "p-2", project_resolution={**binding("store-1", elsewhere), "registration_created": True})
    listing = r.engine.project_registrations
    monkeypatch.setattr(r.engine, "project_registrations", lambda: [
        {**row, "root": folder} if row["id"] == "p-1" else row for row in listing()])
    monkeypatch.setattr(os.path, "normcase", lambda path: path.lower() if isinstance(path, str) else path)
    outcomes = _outcomes(r.run())
    assert outcomes["p-1"]["outcome"] == "removed", "the same folder in another case retires"
    assert r.engine.deleted == ["p-1"] and outcomes["p-1"]["receipt"]["root"] == receipt != folder
    assert (outcomes["p-2"]["outcome"], outcomes["p-2"].get("code")) == ("held", "root_mismatch")


def test_a_recorded_alias_spelling_of_the_same_folder_is_held_never_resolved(registrations):
    """Recorded roots are compared lexically and never resolved per row (the folder may be gone): an alias
    spelling of the same folder stays held as root_mismatch, the fail-safe keep."""
    r = registrations
    real = r.minted("real")
    alias = r.root / "alias"
    alias.symlink_to(real, target_is_directory=True)
    r.engine.projects["p-1"] = real
    r.custody("run-a", "p-1", project_resolution={**binding("store-1", str(alias)),
                                                   "registration_created": True})
    outcomes = _outcomes(r.run())
    assert (outcomes["p-1"]["outcome"], outcomes["p-1"].get("code")) == ("held", "root_mismatch")


def test_only_proven_minted_registrations_of_this_store_whose_work_ended_are_removed(registrations, tmp_path,
                                                                                     monkeypatch):
    from ouroboros import delegate_invocation

    owner_repo = tmp_path / "owner-repo"
    owner_repo.mkdir()
    r = registrations
    r.engine.projects.update({"p-minted": r.minted("c1"), "p-owner": str(owner_repo), "p-foreign": r.minted("c2"),
                              "p-persistent": r.minted("c3"), "p-owned": r.minted("c4"), "p-open": r.minted("c5"),
                              "p-unbound": r.minted("c6"), "p-other": r.minted("c7"), "p-moved": r.minted("c8")})
    r.custody("run-a", "p-minted")
    r.custody("foreign-store", "p-minted", task="foreign", daemon_store_id="store-2", project_owned=True)
    r.custody("foreign-work", "p-minted", task="foreign", project_persistent=True, engine_ref={
        "kind": "work", "descriptor": "/work/engine.json", "store": "store-1"})
    r.rows[-1].settled = False
    owner_keep.set_owner_keep(r.root, "foreign", keep=True, source=KEEP)
    r.custody("run-b", "p-owner")  # a delegated run on the owner's own project: never ours to retire
    r.custody("run-c", "p-persistent", project_persistent=True)
    r.custody("run-d", "p-owned", project_owned=True)  # still owned: the ordinary sweep's
    r.custody("run-e", "p-open", task="task-r2")
    r.rows[-1].settled = False
    r.custody("run-f", "p-unbound", daemon_store_id="")  # a legacy run never bound to a store
    r.custody("run-g", "p-other", daemon_store_id="store-2")  # the same id another store named
    r.custody("run-h", "p-moved", invocation_id="inv-h")  # its request registered another folder
    elsewhere = r.minted("c9")
    monkeypatch.setattr(delegate_invocation, "read", lambda _root, inv, _project: {
        "state": "started", "project_id": "p-moved", "run_id": "run-h", "request": {"scope": {"kind": "project", "root": elsewhere}}} if inv == "inv-h" else None)
    report = r.run()
    outcomes = _outcomes(report)
    assert r.engine.deleted == ["p-minted"], "nothing custody does not prove ours, ended and of this store goes"
    assert outcomes["p-minted"]["outcome"] == "removed" and outcomes["p-minted"]["receipt"]["state"] == "released"
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


@pytest.mark.parametrize("change", ["legacy", "root_changed"])
def test_historical_release_needs_support_and_the_same_binding_at_mutation(registrations, monkeypatch, change):
    r = registrations
    folder = r.engine.projects["p"] = r.minted("old")
    r.custody("ended", "p")
    if change == "legacy":
        monkeypatch.setattr(r.engine, "operations", lambda: [])
    else:
        # Keep the discovery snapshot and the addressed read valid, then move the registration.
        read_or_release = r.engine.bound_registration

        def moved(pid, store, root, *, release=False):
            if release:
                r.engine.projects[pid] = r.minted("new-folder")
            return read_or_release(pid, store, root, release=release)
        monkeypatch.setattr(r.engine, "bound_registration", moved)
    r.run()
    assert not r.engine.deleted and _walked(r.root)["registrations"]["removed"] == 0
    assert "p" in r.engine.projects and (r.engine.projects["p"] == folder) == (change == "legacy")
    assert not delegate_custody_current.removal_status(r.root, "p", {"daemon_store_id": "store-1"})[0]


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
    r.engine.linger["p-gone"] = r.engine.projects.pop("p-gone")  # listed in the snapshot, absent in the addressed read
    r.run()
    walked = _walked(r.root)["registrations"]
    assert walked["kept"] == {"project_has_threads": 1, "project_has_active_run": 1} and walked["absent"] == 1
    assert walked["held"] == {"owner_keep": 1, "task_not_proven_over": 1} and walked["removed"] == 0
    assert sorted(r.engine.deleted) == ["p-run", "p-threads"]
    text = hr.receipt(_walked(r.root))
    assert "the engine kept 2 projects" in text and "still hold sessions" in text and "already gone" in text


@pytest.mark.parametrize(("lane", "kept"), [("custody_kept", True), ("custody", False)])
def test_a_remembered_refusal_is_kept_and_an_unanswered_removal_holds(registrations, lane, kept):
    """The lane the ordinary sweep left behind decides: the engine's proven keep (``custody_kept``) is
    a keep for its threads, never an unanswered removal; a release whose answer never came stays held."""
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    delegate_custody_current.mark_removal(r.root, "p-1", {"daemon_store_id": "store-1"}, lane)
    r.run()
    walked = _walked(r.root)["registrations"]
    assert r.engine.deleted == [] and walked["removed"] == 0
    if kept:
        assert walked["kept"] == {"project_has_threads": 1} and walked["held"] == {}
        assert "the engine kept 1 project" in hr.receipt(_walked(r.root))
    else:
        assert walked["kept"] == {} and walked["held"] == {"engine_unanswered": 1}


@pytest.mark.parametrize(("binding", "holds"), [
    ({}, True),
    ({"daemon_store_id": "store-1"}, True),
    ({"store_identity_unproven": True, "handshake_store_id": "store-2"}, True),
    ({"daemon_store_id": "store-2"}, False),
    ({"engine_ref": {"kind": "work", "descriptor": "different-engine", "store": "store-1"}}, False),
], ids=["unbound", "this-store", "handshake-only", "other-store", "other-engine"])
def test_a_start_still_pending_on_a_registration_holds_it(registrations, binding, holds):
    """A start request naming the project whose POST answer never came (its outcome unknown) may have a run there,
    or land one: the walk holds the registration as it holds one an unsettled run names."""
    from ouroboros import delegate_custody

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    assert delegate_custody.record_start_requested(
        r.root, run_id="", task_id="task-r2", invocation_id="inv-r2", idempotency_key="inv-r2",
        request={"prompt": "p", "scope": {"kind": "project", "root": r.engine.projects["p-1"]}},
        project_id="p-1", route="r", **binding)
    r.run()
    assert r.engine.deleted == ([] if holds else ["p-1"])
    assert _walked(r.root)["registrations"]["held"] == ({"run_unsettled": 1} if holds else {})


@pytest.mark.parametrize("change", ["keep", "persistent", "continuable"])
def test_a_proof_lost_between_the_check_and_the_effect_sends_nothing(registrations, monkeypatch, change):
    """Keep, persistence or a Continue offer that lands after the first check and before the removal: every proof
    is made again under the project lock, and Keep's own gate is held across the effect."""
    from ouroboros import delegate_custody_usage

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    lock = delegate_custody_usage.project_retirement_lock

    def racing(root, project_id, *, engine):
        if change == "keep":
            owner_keep.set_owner_keep(root, "task-r1", keep=True, source=KEEP)
        elif change == "persistent":
            r.rows[0].project_persistent = True
        else:
            r.continuable["value"] = True
        return lock(root, project_id, engine=engine)

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


def test_an_unverified_receipt_is_unknown_and_reconciled_by_address(registrations):
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
    assert _walked(r.root)["pending"] == {"project_id": "p-1", "root": folder, "store_id": "store-1", "shared": True}
    facts = {"daemon_store_id": "store-1"}
    assert delegate_custody_current.removal_status(r.root, "p-1", facts)[0] == "historical"
    r.run()
    assert not delegate_custody_current.removal_status(r.root, "p-1", facts)[0]
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["registrations"]["gone_after_unknown"] == 1


@pytest.mark.parametrize(("lane", "saved"), [("historical_walk", True), ("historical", True), ("historical", False),
                                             ("historical_walk", False)],
                         ids=["older-walk-lane", "carried-historical-mark", "mark-without-intent",
                              "older-walk-mark-without-intent"])
def test_what_no_shared_intent_proves_is_only_read_and_never_sent_again(registrations, lane, saved):
    """The lane alone proves nothing about the request that left: only a ``shared`` intent saved before it does. An older
    walk's id-only DELETE, a ``historical`` mark carried over from that format and a mark whose saved intent is missing
    or already cleared are read: a present registration is held, no second request leaves from the walk or from
    custody, and the walk goes on with its cursor past the project."""
    from ouroboros.delegate_registration_policy import remove_registration

    r = registrations
    root = r.engine.projects["p"] = r.minted("old")
    r.custody("run", "p")
    facts = {"daemon_store_id": "store-1"}
    if saved:
        maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"pending": {"project_id": "p", "root": root, "store_id": "store-1"}}})
    if not (saved and lane == "historical_walk"):  # an older walk's saved intent is itself that lane; else a mark was written
        delegate_custody_current.mark_removal(r.root, "p", facts, lane)
    assert delegate_custody_current.removal_status(r.root, "p", facts)[0] == lane

    def custody_cannot_repeat():
        with pytest.raises(ClaudexorUnavailable, match="unconfirmed"):
            remove_registration(r.root, r.engine, "p", facts, registration_root=root)
        assert r.engine.deleted == []

    custody_cannot_repeat()
    r.run()
    walked = _walked(r.root)
    assert r.engine.deleted == [], "a second request would be of a kind nobody proved the first was"
    assert walked["registrations"]["held"] == {"engine_unanswered": 1} and "pending" not in walked
    assert walked["project_cursor"] == "p" and walked["done_at"] and walked["registrations"]["removed"] == 0
    if saved and lane == "historical_walk":  # carried on as the lane that is only read
        assert delegate_custody_current.removal_status(r.root, "p", facts)[0] == "historical_walk"
    custody_cannot_repeat()


@pytest.mark.parametrize("case", ["proven", "historical_walk-mark", "no-intent", "unshared", "no-root", "other-id",
                                  "other-root", "other-store", "other-engine"])
def test_a_saved_intent_proves_a_repeat_only_for_its_own_store_id_and_root(case):
    """Only the ``historical`` mark with a ``shared`` intent naming exactly this store, id and root proves a repeat."""
    pending = {"project_id": "p-1", "root": "/f", "store_id": "store-1", "shared": True}
    record, project, facts, lane = {"pending": pending}, {"id": "p-1", "root": "/f"}, {"daemon_store_id": "store-1"}, "historical"
    if case == "historical_walk-mark":
        lane = "historical_walk"
    elif case == "no-intent":
        record = {}
    elif case == "unshared":
        record = {"pending": {key: value for key, value in pending.items() if key != "shared"}}
    elif case == "no-root":
        record = {"pending": {**pending, "root": ""}}
    elif case == "other-id":
        project = {**project, "id": "p-2"}
    elif case == "other-root":
        project = {**project, "root": "/g"}
    elif case == "other-store":
        facts = {"daemon_store_id": "store-2"}
    elif case == "other-engine":
        facts = {**facts, "engine_ref": {"kind": "work", "descriptor": "other-engine"}}
    assert hr._repeats(record, project, facts, lane) is (case == "proven")


def test_settling_an_absent_legacy_intent_invalidates_earlier_resolutions(registrations):
    r = registrations
    facts = {"daemon_store_id": "store-1"}
    pending = {"project_id": "p", "root": r.minted("old"), "store_id": "store-1"}
    maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"pending": pending}})
    lane, before = delegate_custody_current.removal_status(r.root, "p", facts)
    assert lane == "historical_walk"
    r.run()
    lane, after = delegate_custody_current.removal_status(r.root, "p", facts)
    assert lane == "" and after and after != before
    assert _walked(r.root)["registrations"]["gone_after_unknown"] == 1
    assert r.engine.deleted == []


def test_legacy_unanswered_release_without_store_remains_unresolved(registrations):
    r = registrations
    pending = {"project_id": "p", "root": r.minted("old")}
    maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"pending": pending}})
    r.run()
    assert _walked(r.root)["registrations"]["held"] == {"store_changed": 1}
    assert delegate_custody_current.removal_status(r.root, "p", {"daemon_store_id": "store-1"})[0] == "store_unproven"
    assert r.engine.deleted == []


def test_a_still_listed_project_after_a_lost_answer_gets_the_same_release_once_more(registrations):
    r = registrations
    r.engine.projects.update({"p-1": r.minted("c1"), "p-2": r.minted("c2")})
    r.custody("run-a", "p-1")
    r.custody("run-b", "p-2")
    r.engine.lost = [TimeoutError("unanswered DELETE")]
    r.run()
    listed = r.engine.listed
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["pending"]["project_id"] == "p-1"
    from ouroboros.delegate_registration_policy import remove_registration

    with pytest.raises(ClaudexorUnavailable, match="unconfirmed"):
        remove_registration(r.root, r.engine, "p-1", {"daemon_store_id": "store-1"})
    r.engine.linger["p-2"] = r.engine.projects["p-2"]  # removed, yet the next listing still names it
    r.run()  # still present: the same bound release again, after the same checks, and this time answered
    walked = _walked(r.root)["registrations"]
    assert r.engine.deleted == ["p-1", "p-1", "p-2"] and r.engine.listed > listed
    assert walked["held"] == {} and walked["removed"] == 2 and walked["listed_after_removal"] == 1
    assert "still appeared in the engine's list" in hr.receipt(_walked(r.root))
    r.run()
    assert r.engine.deleted == ["p-1", "p-1", "p-2"], "answered: the one-time walk sends nothing more"


@pytest.mark.parametrize("listed", [True, False])
def test_route_404_never_proves_absence(registrations, listed):
    r = registrations
    folder = r.minted("old")
    if listed:
        r.engine.projects["p"] = folder
    r.engine.lost = [ClaudexorUnavailable("not_found", "missing", status_code=404)]
    assert hr._release(r.root, r.engine, "p", {"daemon_store_id": "store-1"}, folder)[0] == "unknown"
    assert r.engine.deleted == ["p"] and r.engine.listed == 0


def test_a_404_that_is_not_the_projects_own_absence_proves_nothing(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [ClaudexorUnavailable("route_not_found", "no such route", status_code=404)]
    r.run()
    walked = _walked(r.root)
    assert walked["pending"]["project_id"] == "p-1" and walked["registrations"]["absent"] == 0
    r.run()  # the 404 proved nothing either way: still present, the same release goes again and is answered
    walked = _walked(r.root)["registrations"]
    assert r.engine.deleted == ["p-1", "p-1"] and walked["removed"] == 1 and walked["held"] == {}


def test_an_unanswered_release_waits_at_its_project_while_held_then_goes_again(registrations, monkeypatch):
    """Neither an unreadable readback nor a Keep saved after the lost answer writes the release off: the intent,
    the cursor and done_at wait; once the read answers and the Keep is withdrawn, the same release goes again."""
    r = registrations
    r.engine.projects.update({"p-1": r.minted("c1"), "p-2": r.minted("c2")})
    r.custody("run-a", "p-1")
    r.custody("run-b", "p-2", task="task-r2")
    r.engine.lost = [LOST]
    r.run()
    read = r.engine.bound_registration

    def unreadable(project_id, store_id, root, *, release=False):
        if not release:
            raise ClaudexorUnavailable("daemon_unreachable", "no answer", status_code=0)
        return read(project_id, store_id, root, release=True)

    monkeypatch.setattr(r.engine, "bound_registration", unreadable)
    assert r.run()["holds"]["historical_release_unanswered"] == 1, "a waiting pass says why"
    monkeypatch.setattr(r.engine, "bound_registration", read)
    owner_keep.set_owner_keep(r.root, "task-r1", keep=True, source=KEEP)
    assert r.run()["holds"]["historical_release_unanswered"] == 1
    walked = _walked(r.root)
    assert walked["pending"]["project_id"] == "p-1" and not walked.get("done_at") and "project_cursor" not in walked
    assert r.engine.deleted == ["p-1"] and walked["registrations"]["held"] == {}, "nothing written off, nothing sent"
    owner_keep.set_owner_keep(r.root, "task-r1", keep=False, source=KEEP)
    r.run()
    walked = _walked(r.root)
    assert r.engine.deleted == ["p-1", "p-1", "p-2"] and walked["registrations"]["removed"] == 2 and walked["done_at"]


def test_a_root_changed_after_a_lost_answer_is_held_without_a_second_request(registrations):
    from ouroboros.delegate_workspace_hold import folder_release

    r = registrations
    folder = r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    facts = {"daemon_store_id": "store-1"}
    assert folder_release(r.root, folder, facts) == "p-1", "the folder waits for the answer"
    r.engine.projects["p-1"] = r.minted("c-relinked")  # the engine relinked the same id to another folder
    r.run()
    assert r.engine.deleted == ["p-1"] and _walked(r.root)["registrations"]["held"] == {"root_changed": 1}
    assert not delegate_custody_current.removal_status(r.root, "p-1", facts)[0], "a dead binding ends the mark"
    assert not folder_release(r.root, folder, facts)


def _waits(r, report):
    """The conditional request still waits: its intent, the cursor and ``done_at`` stay, and the pass says why."""
    walked = _walked(r.root)
    assert report["holds"]["historical_release_unanswered"] == 1
    assert walked["pending"]["project_id"] == "p-1" and walked["pending"]["shared"] is True
    assert "project_cursor" not in walked and not walked.get("done_at") and hr.owed(maintenance_jobs.job(r.root, JOB))
    assert walked["registrations"]["held"] == {} and walked["registrations"].get("gone_after_unknown", 0) == 0
    assert delegate_custody_current.removal_status(r.root, "p-1", {"daemon_store_id": "store-1"})[0] == "historical"


def _answered_again(r):
    walked = _walked(r.root)
    assert r.engine.deleted == ["p-1", "p-1"] and walked["registrations"]["removed"] == 1 and walked["done_at"]
    assert "pending" not in walked and not delegate_custody_current.removal_status(r.root, "p-1", {"daemon_store_id": "store-1"})[0]


def test_a_changed_served_base_keeps_a_conditional_request_waiting_for_its_own(registrations, monkeypatch):
    """The base now served is not the one that took the request: that is no addressed answer, and another base's
    empty list is none either. The intent, the cursor and ``done_at`` wait; the original base's return settles it."""
    from ouroboros import delegate_custody_usage
    from ouroboros.delegate_registration_policy import engine_facts

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    assert _walked(r.root)["pending"]["store_id"] == "store-1"
    r.engine.store_identity = "store-2"
    registered = dict(r.engine.projects)
    r.engine.projects.clear()
    locked = []
    retirement_lock = delegate_custody_usage.project_retirement_lock

    def observe(root, project_id, *, engine):
        locked.append((project_id, engine_facts(engine)))
        return retirement_lock(root, project_id, engine=engine)

    monkeypatch.setattr(delegate_custody_usage, "project_retirement_lock", observe)
    _waits(r, r.run())
    assert r.engine.deleted == ["p-1"] and locked == [("p-1", {"daemon_store_id": "store-1"})]
    r.engine.store_identity = "store-1"  # its own base is back and still holds the registration
    r.engine.projects.update(registered)
    r.run()
    _answered_again(r)


def test_an_away_original_store_costs_one_addressed_question_never_the_whole_inventory(registrations):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    listed = r.engine.listed
    r.engine.store_identity = "store-2"
    assert r.run()["holds"]["historical_release_unanswered"] == 1
    _waits(r, r.run())  # a pass that waits rechecks the saved address alone: no custody walk, no inventory read
    assert r.engine.listed == listed
    r.engine.store_identity = "store-1"
    r.run()
    _answered_again(r)
    assert r.engine.listed > listed, "the answered pass resumes the discovery and reads the list"


def test_a_conditional_request_waits_while_the_engine_does_not_offer_the_bound_operations(registrations, monkeypatch):
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    offered = r.engine.operations
    monkeypatch.setattr(r.engine, "operations", lambda: [])  # an engine that cannot be asked or told to release
    _waits(r, r.run())
    assert r.engine.deleted == ["p-1"]
    monkeypatch.setattr(r.engine, "operations", offered)
    r.run()
    _answered_again(r)


def test_another_stores_mismatch_on_the_read_answers_nothing_and_the_intent_waits(registrations, monkeypatch):
    """A 409 ``store_mismatch`` is the answer of the base that serves now, and only that base's word: it cannot
    establish whether the store that took the request applied the release (that base returned later with the
    registration still present). The intent, its marks and the folder fence wait, claiming nothing; the
    original base's return settles the same release again."""
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    offered = r.engine.operations

    def base_swapped_before_the_read():  # after the walk saw the base it asks about, before its addressed read
        r.engine.store_identity = "store-2"
        return offered()

    monkeypatch.setattr(r.engine, "operations", base_swapped_before_the_read)
    _waits(r, r.run())  # another base's refusal is no answer: the pass holds and the intent stays
    r.engine.store_identity = "store-1"  # its own base is back and still holds the registration
    monkeypatch.setattr(r.engine, "operations", offered)
    r.run()
    _answered_again(r)


def test_a_repeat_release_refused_by_another_store_waits_for_the_one_that_took_it(registrations, monkeypatch):
    """The read is answered by the original base, then the base swaps before the repeated release: another
    store's refusal clears no mark and settles nothing — the intent waits, and the original base's return
    answers the same release again."""
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    read = r.engine.bound_registration

    def swapped_base_refuses_only_the_release(project_id, store_id, root, *, release=False):
        if release:  # the base restored between the read and the release: not the one that took the request
            raise ClaudexorUnavailable("store_mismatch", "different store", status_code=409)
        return read(project_id, store_id, root, release=release)

    monkeypatch.setattr(r.engine, "bound_registration", swapped_base_refuses_only_the_release)
    _waits(r, r.run())
    monkeypatch.setattr(r.engine, "bound_registration", read)
    r.run()
    _answered_again(r)


def test_a_repeat_asks_about_the_address_its_intent_saved_never_the_one_the_list_shows_now(registrations, monkeypatch):
    r = registrations
    folder = r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    r.engine.lost = [LOST]
    r.run()
    elsewhere = r.minted("elsewhere")
    monkeypatch.setattr(r.engine, "project_registrations", lambda: [{"id": "p-1", "root": elsewhere}])
    asked, bound = [], r.engine.bound_registration

    def spy(project_id, store_id, root, *, release=False):
        asked.append((root, release))
        return bound(project_id, store_id, root, release=release)

    monkeypatch.setattr(r.engine, "bound_registration", spy)
    r.run()
    assert asked == [(folder, False), (folder, False), (folder, True)], "the saved folder, the read and the release"
    _answered_again(r)


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
    monkeypatch.setattr(delegate_custody_current, "runs", lambda root, where="", params=(), **kw: (
        scans.append(where), custody_runs(root, where, params, **kw))[1])
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


def test_a_deferred_project_missing_from_the_list_is_reported_unconfirmed_never_kept(registrations):
    """A deferred project whose sessions settled while it left the list was never confirmed present: the receipt
    reports that its registration could not be confirmed and claims no retention for it (never "stay registered")."""
    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1", task="task-slow")
    maintenance_jobs.record(r.root, JOB, **{hr.WALK: {"threads": {}, "task_holds": {}, "waiting": ["task-slow"],
                                                      "threads_done_at": "2026-11-09T00:00:00+00:00"}})
    r.waits.add("task-slow")
    r.run()  # deferred: its sessions' release still waits
    r.waits.discard("task-slow")
    r.engine.projects.pop("p-1")  # gone from the list before its release could be asked again
    r.run()
    walked = _walked(r.root)
    assert walked["deferred"] == {} and walked["registrations"]["held"] == {"not_listed": 1}
    text = hr.receipt(walked)
    assert "the registrations of 1 project could not be confirmed" in text
    assert "stay registered" not in text, "an unconfirmed presence is never claimed as retention"


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
    r.run()  # the unknown intent still progresses after done_at: read back addressed, never sent again
    assert r.engine.deleted == ["p-1"] and r.engine.listed == listed, "one addressed readback, no resend, no inventory"
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
    from ouroboros import delegate_invocation

    r = registrations
    named = {"p-gap": "inv-gap", "p-none": "inv-none", "p-unread": "inv-unread", "p-thread": "inv-thread",
             "p-match": "inv-match"}
    for index, (project, invocation) in enumerate(sorted(named.items())):
        r.engine.projects[project] = r.minted(f"c{index}")
        r.custody(f"run-{index}", project, invocation_id=invocation)
    records = {"inv-gap": {"state": "legacy_gap", "reason": "no_request_row"},
               "inv-unread": {"state": "started", "request": None},  # its body reference could not be read
               "inv-thread": {"state": "started", "request": {"_thread_id": "thr-9"}},  # a thread turn: no scope
               "inv-match": {"state": "started", "project_owned": True,
                             "project_resolution": binding("store-1", r.engine.projects["p-match"]),
                             "request": {"scope": {"kind": "project", "root": r.engine.projects["p-match"]}}}}
    for row in r.rows:
        if row.invocation_id in records:
            records[row.invocation_id].update(project_id=row.project_id, run_id=row.run_id)
    monkeypatch.setattr(delegate_invocation, "read", lambda _root, inv, _project: records.get(inv))
    r.run()
    assert r.engine.deleted == ["p-match"]
    assert _walked(r.root)["registrations"]["held"] == {"custody_unreadable": 3, "ownership_unproven": 1}


@pytest.mark.parametrize("proof,removed", [("created", True), ("adopted", False), ("original-owner", True),
    ("old-adopter", False), ("unknown-store", False), ("handshake-only", False), ("run-store-only", False),
    ("already-released", False), ("creator-duty", True)])
def test_historical_minted_folder_requires_original_registration_ownership(registrations, proof, removed):
    r = registrations
    folder = r.engine.projects["p"] = r.minted("old")
    invocation_id = "invoke-original"
    origin = binding("store-1", folder)
    if proof in ("created", "adopted"):
        origin["registration_created"] = proof == "created"
    if proof == "creator-duty":
        origin["creator_invocation"] = invocation_id
    if proof in ("old-adopter", "unknown-store", "handshake-only", "run-store-only"):
        origin = {}  # the legacy event has no registration receipt but keeps its original scope
    r.custody("run", "p", invocation_id=invocation_id, project_resolution=origin,
              registration_outcome="removed" if proof == "already-released" else "kept", request_event={
                  "invocation_id": invocation_id, "project_id": "p",
                  "project_owned": proof in ("original-owner", "unknown-store", "handshake-only", "run-store-only", "already-released"),
                  "handshake_store_id": "store-1" if proof == "handshake-only" else "",
                  "daemon_store_id": "store-1" if proof == "run-store-only" else "",
                  "request": {"scope": {"kind": "project", "root": folder}}})
    result = _outcomes(r.run())["p"]
    assert r.engine.deleted == (["p"] if removed else [])
    assert result["outcome"] == ("removed" if removed else "held")
    if not removed:
        assert result["code"] == "ownership_unproven"


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

    monkeypatch.setattr(r.engine, "release_registered", crash)
    r.run()
    assert _walked(r.root)["pending"]["project_id"] == "p-1"
    r.run()
    counts = _walked(r.root)["registrations"]
    assert r.engine.deleted == ["p-1"] and counts["gone_after_unknown"] == 1 and counts["removed"] == 0


@pytest.mark.parametrize("late_store,removed", [("store-1", False), ("", False), ("store-2", True)])
def test_locked_reread_distinguishes_local_ambiguous_and_foreign_runs(registrations, monkeypatch, late_store, removed):
    from ouroboros import delegate_custody_usage

    r = registrations
    r.engine.projects["p-1"] = r.minted("c1")
    r.custody("run-a", "p-1")
    lock = delegate_custody_usage.project_retirement_lock

    def racing(root, project_id, *, engine):  # a sibling's STARTED lands between the decision and the lock
        r.rows.append(RunCustody(run_id="run-late", task_id="task-r9", project_id=project_id, settled=True,
                                 terminal_state="succeeded", daemon_store_id=late_store))
        return lock(root, project_id, engine=engine)

    monkeypatch.setattr(delegate_custody_usage, "project_retirement_lock", racing)
    r.run()
    assert r.engine.deleted == (["p-1"] if removed else [])
    assert _walked(r.root)["registrations"]["held"] == ({} if removed else {"custody_changed": 1})


def test_start_binding_at_the_locked_snapshot_keeps_its_registration(registrations, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_custody_usage, store

    r = registrations
    folder = r.engine.projects["p"] = r.minted("old")
    r.custody("old-run", "p")
    monkeypatch.setattr(delegate_custody_current, "runs", stored_runs)
    with store.open_store(r.root, write=True) as txn:
        txn.put_custody("old-run", vars(r.rows[0]), state=delegate_custody_current.CLOSED)
    request = {"project_id": "p", "task_id": "sibling", "invocation_id": "i", "daemon_store_id": "store-1"}
    lock = delegate_custody_usage.project_retirement_lock
    snapshot = delegate_custody_current.registration_snapshot
    pending = False

    @contextmanager
    def race(root, project_id, **kw):
        nonlocal pending
        assert custody.emit(root, custody.START_REQUESTED, {**request, "request": {"scope": {"root": folder}}})
        pending = True
        with lock(root, project_id, **kw):
            yield

    def binding_snapshot(root, project_id, engine, **kw):
        nonlocal pending
        if pending:
            # STARTED atomically replaces the pending row just before the walk reads its requests.
            assert custody.emit(root, custody.STARTED, {**request, "run_id": "sibling-run"})
            pending = False
        return snapshot(root, project_id, engine, **kw)

    monkeypatch.setattr(delegate_custody_usage, "project_retirement_lock", race)
    monkeypatch.setattr(delegate_custody_current, "registration_snapshot", binding_snapshot)
    r.run()
    named, requests, _ = snapshot(r.root, "p", request)
    assert not requests and not named["sibling-run"].settled
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
