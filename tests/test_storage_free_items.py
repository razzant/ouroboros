"""The ordinary action on exact objects: complete pages, one object per id, refusals before any effect,
each object's own outcome, and the owner's keep at every retirer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import body_candidate as body
from ouroboros import owner_keep, record_compression, task_custody, test_environment as environment  # noqa: F401
from ouroboros import copy_environments as environments
from ouroboros import subagent_worktrees as sw
from ouroboros.gateway.storage import storage_routes
from ouroboros.gateway.task_owner_keep import api_task_owner_keep
from ouroboros.storage_free import FreeSpace
from ouroboros.task_results import load_task_result, write_task_result
from starlette.routing import Route
from tests._shared import retire_synthetic_retention
from tests._store_shared import seed_result
from tests.test_body_candidate_retention import candidate  # noqa: F401 -- the body fixture
from tests.test_storage_free import _drives, _fake_sizer, _group, _join, _owner, _preview, _run, OLD
from tests.test_storage_free_safety import copies, _copy  # noqa: F401 -- the plain copies fixture
from tests.test_store import document

KEEP = {"basis": "test", "route": "test"}


def _start(owner, env, items=None, groups=None, plan="", ):
    view = owner.projection(env[0])
    return owner.start(env[0], env[1], plan or view["plan"], groups, items)


def _page(owner, env, view, **query):
    plan = query.pop("plan", owner.projection(env[0])["plan"])
    return owner.items(env[0], plan, view, query.get("cursor", 0), query.get("limit", 100), query.get("version"))


def _ids(page):
    return [row["id"] for row in page["items"]]


def _second_copy_of(env, task_id, other):
    """A second copy record of one task: what ``only=[task_id]`` could never tell apart."""
    handle = _copy(env, other)
    sw._put_record({**sw._record_at(handle.path, env[0]), "task_id": task_id}, env[0])
    return handle


def test_one_id_is_one_object_when_a_task_owns_two_copies(copies):  # noqa: F811
    first, second = _copy(copies, "two"), _second_copy_of(copies, "two", "other")
    owner = _owner({"two": False, "other": False})
    view = _preview(owner, copies[0], copies[0].parent)
    pool = _page(owner, copies, "pool")
    assert view["pool"]["count"] == pool["total"] == 2 and view["pool"]["kinds"] == {"working_copy": 2}
    chosen = next(row["id"] for row in pool["items"] if row["path"] == first.path)
    assert _start(owner, copies, items=[chosen]) == {"started": True, "reason": ""}
    _join(owner)
    assert not Path(first.path).exists() and Path(second.path).is_dir()
    outcomes = _page(owner, copies, "outcomes")
    assert [(row["id"], row["outcome"]) for row in outcomes["items"]] == [(chosen, "removed")]


@pytest.mark.parametrize("items,groups,reason", [
    (["nope"], None, "unknown_item"), ([], None, "nothing_selected"), ("x", None, "unknown_item"), ([1], None, "unknown_item"),
    (["any"], ["working_copies"], "selection_conflict"), (None, None, "nothing_selected")])
def test_a_bad_selection_refuses_before_any_effect(copies, items, groups, reason):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    assert _start(owner, copies, items=items, groups=groups) == {"started": False, "reason": reason}
    assert Path(handle.path).is_dir() and owner.projection(copies[0])["state"] == "ready"


def test_a_good_id_beside_a_bad_one_touches_nothing(copies):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    good = _ids(_page(owner, copies, "pool"))
    assert _start(owner, copies, items=[*good, "nope"]) == {"started": False, "reason": "unknown_item"}
    assert Path(handle.path).is_dir()


def test_an_excluded_or_saved_work_item_is_not_offered_to_the_ordinary_action(copies):  # noqa: F811
    root, _repo, wtroot = copies
    (wtroot / "dlg_lost-task_0123456789abcdef").mkdir(parents=True)
    held, live = _copy(copies, "held"), _copy(copies, "live")
    (Path(held.path) / "work.txt").write_text("unsaved\n", encoding="utf-8")
    owner = _owner({"held": False, "live": True, "lost-task": False})
    _preview(owner, root, root.parent)
    assert _page(owner, copies, "pool")["total"] == 0
    excluded = _page(owner, copies, "excluded")
    reasons = {Path(row["path"]).name: row["reason"] for row in excluded["items"]}
    assert reasons == {"held": "capture_unproven", "live": "in_use", "dlg_lost-task_0123456789abcdef": "advanced_group"}
    assert excluded["counts"] == {"advanced_group": 1, "capture_unproven": 1, "in_use": 1}
    for row in excluded["items"]:
        assert _start(owner, copies, items=[row["id"]]) == {"started": False, "reason": "not_offered"}
    assert Path(held.path).is_dir() and Path(live.path).is_dir()


def test_a_copy_used_after_its_preview_stays_as_changed(copies):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    sw._put_record({**sw._record_at(handle.path, copies[0]), "used_at": 123.0}, copies[0])
    assert _start(owner, copies, items=_ids(_page(owner, copies, "pool")))["started"]
    _join(owner)
    [row] = _page(owner, copies, "outcomes")["items"]
    assert (row["outcome"], row["reason"]) == ("kept", "changed") and Path(handle.path).is_dir()
    assert owner.projection(copies[0])["run"]["kept"] == [{"reason": "changed", "count": 1}]


def test_stop_leaves_the_rest_not_started_with_failures_first(copies, monkeypatch):  # noqa: F811
    handles = [_copy(copies, f"t{number}") for number in range(3)]
    owner = _owner({f"t{number}": False for number in range(3)})
    _preview(owner, copies[0], copies[0].parent)
    real, calls = sw.prune_orphans, []

    def stop_after_one(*args, **kwargs):
        outcome = real(*args, **kwargs)
        calls.append(1)
        owner.cancel()
        return outcome

    monkeypatch.setattr(sw, "prune_orphans", stop_after_one)
    assert _start(owner, copies, items=_ids(_page(owner, copies, "pool")))["started"]
    _join(owner)
    run = owner.projection(copies[0])["run"]
    assert (run["selected"], run["removed"], run["not_started"], run["stopped"]) == (3, 1, 2, True)
    assert [row["outcome"] for row in _page(owner, copies, "outcomes")["items"]] == ["not_started", "not_started", "removed"]
    assert sum(Path(handle.path).exists() for handle in handles) == 2


def test_pages_are_complete_and_a_moved_version_is_said_not_skipped(copies):  # noqa: F811
    for number in range(5):
        _copy(copies, f"t{number}")
    owner = _owner({f"t{number}": False for number in range(5)})
    view = _preview(owner, copies[0], copies[0].parent)
    first = _page(owner, copies, "pool", limit=2)
    second = _page(owner, copies, "pool", limit=2, cursor=first["next_cursor"], version=first["version"])
    third = _page(owner, copies, "pool", limit=2, cursor=second["next_cursor"], version=first["version"])
    listed = _ids(first) + _ids(second) + _ids(third)
    assert third["next_cursor"] is None and len(set(listed)) == 5 == first["total"] == view["pool"]["count"]
    assert _page(owner, copies, "pool", version="stale") == {"error": "view_changed"}
    assert _page(owner, copies, "nope") == {"error": "bad_view"}
    assert _page(owner, copies, "pool", limit=0) == {"error": "bad_view"}
    assert _page(owner, copies, "pool", plan="old") == {"error": "preview_outdated"}


def test_a_finished_plan_cannot_run_twice_and_a_new_preview_replaces_its_pages(copies):  # noqa: F811
    _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    plan = owner.projection(copies[0])["plan"]
    items = _ids(_page(owner, copies, "pool"))
    assert _start(owner, copies, items=items)["started"]
    _join(owner)
    assert _start(owner, copies, items=items, plan=plan) == {"started": False, "reason": "preview_outdated"}
    assert _page(owner, copies, "outcomes", plan=plan)["total"] == 1
    _preview(owner, copies[0], copies[0].parent)
    assert _page(owner, copies, "outcomes", plan=plan) == {"error": "preview_outdated"}


def test_an_outcome_page_says_when_the_trash_settled_it(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("done", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    monkeypatch.setattr("ouroboros.maintenance.child.request", lambda: None)
    env = (root, tmp_path)
    owner = _owner(probes)
    _preview(owner, root, tmp_path)
    assert _start(owner, env, items=["direct:done"])["started"]
    _join(owner)
    pending = _page(owner, env, "outcomes")
    assert [row["outcome"] for row in pending["items"]] == ["deleting"]
    task_custody.remove_custody_trash(root)
    assert _page(owner, env, "outcomes", version=pending["version"]) == {"error": "view_changed"}
    assert [row["outcome"] for row in _page(owner, env, "outcomes")["items"]] == ["removed"]


def test_the_routes_answer_pages_and_typed_refusals(copies, monkeypatch):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    client = TestClient(Starlette(routes=storage_routes(copies[0], repo_dir=copies[1], free_owner=owner)))
    assert client.get("/api/storage/free/items", params={"plan": "x", "view": "pool"}).json()["reason_code"] == "preview_outdated"
    _preview(owner, copies[0], copies[0].parent)
    plan = owner.projection(copies[0])["plan"]
    answer = client.get("/api/storage/free/items", params={"plan": plan, "view": "pool"})
    assert answer.status_code == 200 and answer.headers["cache-control"] == "no-store" and answer.json()["total"] == 1
    assert client.get("/api/storage/free/items", params={"plan": plan, "view": "pool", "cursor": "x"}).status_code == 400
    moved = client.get("/api/storage/free/items", params={"plan": plan, "view": "pool", "version": "old"})
    assert (moved.status_code, moved.json()["reason_code"]) == (409, "view_changed")
    item = answer.json()["items"][0]["id"]
    assert client.post("/api/storage/free", json={"plan": plan, "items": ["nope"]}).json()["reason"] == "unknown_item"
    assert client.post("/api/storage/free", json={"plan": plan, "items": [item], "groups": ["x"]}).json()["reason"] == "selection_conflict"
    assert client.post("/api/storage/free", json={"plan": plan, "items": [item]}).json()["started"] is True
    _join(owner)
    assert not Path(handle.path).exists()


def test_a_remove_whose_answer_fails_after_the_start_is_a_500_over_a_run_that_began_and_cannot_start_twice(copies):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    plan = owner.projection(copies[0])["plan"]
    item = _ids(_page(owner, copies, "pool"))[0]
    counter = SimpleNamespace(projection=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("projection failed after start")))
    failing = TestClient(Starlette(routes=storage_routes(copies[0], repo_dir=copies[1], owner=counter, free_owner=owner)),
                         raise_server_exceptions=False)
    assert failing.post("/api/storage/free", json={"plan": plan, "items": [item]}).status_code == 500
    _join(owner)
    assert not Path(handle.path).exists(), "the 500 followed an admitted run: the copy is gone"
    assert owner.projection(copies[0])["run"]["removed"] + owner.projection(copies[0])["run"]["deleting"] == 1
    # The same plan is spent: the owner's original Remove, if it were resent, could only be refused.
    assert owner.start(copies[0], copies[1], plan, None, [item]) == {"started": False, "reason": "preview_outdated"}


def test_group_runs_keep_their_documented_form_and_also_answer_outcomes(copies):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    view = _preview(owner, copies[0], copies[0].parent)
    assert _run(owner, copies[0], copies[0].parent, groups=("working_copies",))["run"]["outcomes_ready"] is True
    assert not Path(handle.path).exists() and _group(view, "working_copies")["count"] == 1
    assert [row["outcome"] for row in _page(owner, copies, "outcomes")["items"]] == ["removed"]


# ---- the owner's keep ------------------------------------------------------------------------


def test_a_kept_copy_stays_through_preview_run_and_the_background_and_goes_once_released(copies):  # noqa: F811
    handle = _copy(copies)
    answer = owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP, note="mine")
    assert (answer["status"], answer["keep"], answer["retained"]) == ("kept", True, [handle.path])
    assert load_task_result(copies[0], "finished")["owner_keep"]["source"] == KEEP
    assert sw.prune_orphans(data_dir=copies[0], worktree_root=copies[2], retention_days=0, live=lambda _t: False)["removed"] == 0
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    assert _page(owner, copies, "pool")["total"] == 0
    assert _page(owner, copies, "excluded")["items"][0]["reason"] == "owner_keep"
    assert Path(handle.path).is_dir()
    assert owner_keep.set_owner_keep(copies[0], "finished", keep=False, source=KEEP)["status"] == "released"
    assert "owner_keep" not in load_task_result(copies[0], "finished")
    assert sw.prune_orphans(data_dir=copies[0], worktree_root=copies[2], retention_days=0, live=lambda _t: False)["removed"] == 1


def test_a_keep_named_after_the_preview_stops_the_run_with_its_reason(copies):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP)
    assert _start(owner, copies, items=_ids(_page(owner, copies, "pool")))["started"]
    _join(owner)
    assert Path(handle.path).is_dir()
    [row] = _page(owner, copies, "outcomes")["items"]
    assert (row["outcome"], row["reason"]) == ("kept", "owner_keep")


def test_a_keep_holds_the_exact_task_not_its_parent_or_child(copies):  # noqa: F811
    parent, child = _copy(copies, "parent"), _copy(copies, "child")
    write_task_result(copies[0], "child", "completed", parent_task_id="parent", root_task_id="parent")
    owner_keep.set_owner_keep(copies[0], "parent", keep=True, source=KEEP)
    assert sw.prune_orphans(data_dir=copies[0], worktree_root=copies[2], retention_days=0, live=lambda _t: False)["removed"] == 1
    assert Path(parent.path).is_dir() and not Path(child.path).exists()


def test_a_kept_drive_is_never_moved_and_a_moved_drive_is_never_called_kept(tmp_path, monkeypatch):
    root = tmp_path / "data"
    probes = _drives(root, [("kept", "direct", "completed", OLD), ("gone", "direct", "completed", OLD)])
    _fake_sizer(monkeypatch)
    monkeypatch.setattr("ouroboros.maintenance.child.request", lambda: None)
    assert owner_keep.set_owner_keep(root, "kept", keep=True, source=KEEP)["status"] == "kept"
    outcome = task_custody.settle_child_drive(root, "kept", root / "task_drives" / "kept", live=lambda _t: False)
    assert (outcome["status"], outcome["reason"]) == ("retained", "owner_keep") and (root / "task_drives" / "kept").is_dir()
    assert task_custody.settle_child_drive(root, "gone", root / "task_drives" / "gone", live=lambda _t: False)["status"] == "removed"
    late = owner_keep.set_owner_keep(root, "gone", keep=True, source=KEEP)
    assert (late["status"], late["keep"], late["retained"]) == ("already_retired", False, [])
    assert "owner_keep" not in load_task_result(root, "gone")


def test_keep_during_a_retirement_says_inflight_and_saves_nothing(copies):  # noqa: F811
    handle = _copy(copies)
    holding, release = threading.Event(), threading.Event()

    def retirer():
        with owner_keep.gate(copies[0], "finished") as why:
            assert why == ""
            holding.set()
            release.wait(10)

    thread = threading.Thread(target=retirer)
    thread.start()
    holding.wait(10)
    try:
        answer = owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP, timeout_sec=0)
    finally:
        release.set()
        thread.join()
    assert (answer["status"], answer["keep"], answer["reason"]) == ("inflight", False, "custody_busy")
    assert "owner_keep" not in load_task_result(copies[0], "finished") and Path(handle.path).is_dir()


def test_a_kept_snapshot_survives_the_release_after_apply_and_the_removal_after_disposition(copies):  # noqa: F811
    root, repo, wtroot = copies
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", "completed"))
    owner_keep.set_owner_keep(root, "finished", keep=True, source=KEEP)
    assert sw.release_snapshot_checkout("snap", worktree_root=wtroot, data_dir=root) is False
    assert sw.remove_execution_snapshot("snap", worktree_root=wtroot, data_dir=root) is False
    assert sw.prune_execution_snapshots(set(), worktree_root=wtroot, data_dir=root, live=lambda _t: False)["removed"] == []
    assert Path(handle.path).is_dir() and sw.find_execution_snapshot("snap", root)
    owner_keep.set_owner_keep(root, "finished", keep=False, source=KEEP)
    assert sw.release_snapshot_checkout("snap", worktree_root=wtroot, data_dir=root) is True


def test_a_kept_snapshot_is_listed_with_the_keep_as_its_reason(copies):  # noqa: F811
    root, repo, wtroot = copies
    sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", "completed"))
    owner_keep.set_owner_keep(root, "finished", keep=True, source=KEEP)
    owner = _owner({"finished": False})
    view = _preview(owner, root, root.parent)
    [row] = _page(owner, copies, "excluded")["items"]
    assert (row["id"], row["reason"]) == ("snapshot:snap", "owner_keep"), "the run would say the same, not in use"
    assert view["pool"]["excluded"] == [{"reason": "owner_keep", "count": 1}]


def test_a_stale_row_decides_nothing_at_release_or_removal(copies, monkeypatch):  # noqa: F811
    root, repo, wtroot = copies
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", "completed"))
    stale = sw.find_execution_snapshot("snap", root)
    sw._drop_record(handle.path, root)
    monkeypatch.setattr(sw, "find_execution_snapshot", lambda *_a, **_k: stale)
    assert sw.release_snapshot_checkout("snap", worktree_root=wtroot, data_dir=root) is False
    assert sw.remove_execution_snapshot("snap", worktree_root=wtroot, data_dir=root) is False
    assert Path(handle.path).is_dir()


def test_lossless_compression_still_reads_a_kept_task_as_over(copies):  # noqa: F811
    _copy(copies)
    owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP)
    assert sw._task_proven_over("finished", copies[0], lambda _t: False) is True


def test_an_unreadable_document_counts_as_kept_and_an_unknown_name_carries_no_fact(copies, monkeypatch):  # noqa: F811
    assert owner_keep.is_kept(copies[0], "not a task id!") is False
    assert owner_keep.is_kept(copies[0], "never-recorded") is False
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("torn")))
    assert owner_keep.is_kept(copies[0], "finished") is True


def test_the_route_stamps_its_source_and_refuses_any_declared_one(copies, monkeypatch):  # noqa: F811
    _copy(copies)
    app = Starlette(routes=[Route("/api/tasks/{task_id}/owner-keep", endpoint=api_task_owner_keep, methods=["POST"])])
    app.state.drive_root = copies[0]
    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.request_drive_root", lambda _request: copies[0])
    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.migrating_error", lambda _root: None)
    client = TestClient(app)
    refused = client.post("/api/tasks/finished/owner-keep", json={"keep": True, "source": {"basis": "owner said so"}})
    assert (refused.status_code, refused.json()["reason_code"]) == (400, "unexpected_fields")
    assert client.post("/api/tasks/nobody/owner-keep", json={"keep": True}).status_code == 404
    answer = client.post("/api/tasks/finished/owner-keep", json={"keep": True, "note": " why "})
    assert answer.status_code == 200 and answer.json()["status"] == "kept"
    fact = load_task_result(copies[0], "finished")["owner_keep"]
    assert fact["source"]["basis"] == "authenticated_gateway_request" and fact["note"] == "why"


# ---- the temporary environment, whole: the real producer through retirement and back ---------


def _gone_copy(candidate_row):
    ctx, data, env_root = candidate_row
    sw._force_rmtree(Path(ctx.repo_dir))  # a released checkout: its record stays, its environment beside it
    return ctx, data, env_root


def test_the_real_producer_names_its_owner_and_its_environment_goes_then_comes_back_fresh(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = _gone_copy(candidate)
    assert environments.scratch_owner(env_root) == {"task_id": ctx.task_id, "copy": str(Path(ctx.repo_dir).resolve())}
    (env_root / "tmp" / "venv").mkdir(parents=True)
    (env_root / "tmp" / "venv" / "installed").write_bytes(b"package")
    live = lambda _task: False  # noqa: E731
    assert environments.environment_verdict(env_root, data, live) == ""
    _fake_sizer(monkeypatch)
    owner = FreeSpace(live=live, offered=("service_folders",))
    _preview(owner, data, data.parent)
    pool = owner.items(data, owner.projection(data)["plan"], "pool", 0, 100, None)
    [row] = pool["items"]
    assert (row["kind"], row["task_id"], row["path"]) == ("environment", ctx.task_id, str(env_root))
    assert owner.start(data, data.parent, owner.projection(data)["plan"], None, [row["id"]])["started"]
    _join(owner)
    assert not env_root.exists()
    # The next process of that work: a fresh declared folder, nothing installed in it.
    Path(ctx.repo_dir).mkdir(parents=True)
    env = body.process_environment(ctx, ctx.repo_dir)
    assert Path(env["HOME"]).parent == env_root and not (env_root / "tmp" / "venv").exists()
    assert environments.scratch_owner(env_root)["task_id"] == ctx.task_id


def test_a_preparation_that_cannot_rebuild_refuses_visibly_instead_of_using_the_serving_environment(candidate, monkeypatch):  # noqa: F811
    ctx, _data, env_root = _gone_copy(candidate)
    sw._force_rmtree(env_root)
    Path(ctx.repo_dir).mkdir(parents=True)
    monkeypatch.setattr(environment, "isolated_environment", lambda *_a, **_k: (_ for _ in ()).throw(OSError(28, "no space left")))
    with pytest.raises(body.CandidateRefused) as refused:
        body.process_environment(ctx, ctx.repo_dir)
    assert refused.value.code == "CANDIDATE_ENVIRONMENT_UNAVAILABLE" and "no space left" in str(refused.value)


def test_the_background_waits_the_shared_term_and_a_kept_or_retained_one_stays(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = _gone_copy(candidate)
    live = lambda _task: False  # noqa: E731
    assert environments.prune_environments(data, live, retention_days=30)["removed"] == []
    assert env_root.is_dir()
    owner_keep.set_owner_keep(data, ctx.task_id, keep=True, source=KEEP)
    assert environments.prune_environments(data, live, retention_days=1)["removed"] == []
    owner_keep.set_owner_keep(data, ctx.task_id, keep=False, source=KEEP)
    environment.retain_tree(env_root / "tmp", "evidence inside")
    assert environments.environment_verdict(env_root, data, live) == "retained_evidence"
    (env_root / "tmp" / environment.RETENTION_MARKER).unlink()
    import os
    old = (env_root / environment.SCRATCH_MARKER).stat().st_mtime - 7 * 86400
    os.utime(env_root / environment.SCRATCH_MARKER, (old, old))
    assert environments.prune_environments(data, live, retention_days=1)["removed"] == [env_root.name]
    assert not env_root.exists()


def test_an_unknown_old_environment_is_never_enrolled_by_its_name(copies, monkeypatch):  # noqa: F811
    root, _repo, wtroot = copies
    old = wtroot / "dlg_old-task_0123456789abcdef.env"
    (old / "venv").mkdir(parents=True)
    (old / environment.SCRATCH_MARKER).write_text("body-process\n", encoding="utf-8")  # declared, but by no one
    live = lambda _task: False  # noqa: E731
    assert environments.environment_verdict(old, root, live) == "undeclared_environment"
    assert environments.prune_environments(root, live, retention_days=1)["removed"] == [] and old.is_dir()
    owner = _owner({"old-task": False})
    _preview(owner, root, root.parent)
    assert _page(owner, copies, "pool")["total"] == 0
    [row] = _page(owner, copies, "excluded")["items"]
    assert row["reason"] == "advanced_group", "the saved group below removes it: it is no item that cannot go"
    assert _run(owner, root, root.parent, groups=("service_folders",))["run"]["removed"] == 1, "the explicit saved-work group stays"


_OLD_ENV = "dlg_old-task_0123456789abcdef.env"
_DECLARATIONS = {"absent": None, "torn": "body-process\n{\"task_id\": \"old-", "ownerless": "body-process\n",
                 "this_task": "body-process\n{\"task_id\": \"old-task\", \"copy\": \"/gone\"}\n",
                 "another_task": "body-process\n{\"task_id\": \"another-task\", \"copy\": \"/gone\"}\n"}


def _old_environment(fixture, place, declaration):
    """A finished task's drive and copy record are gone; only an older environment folder remains."""
    from ouroboros.unindexed import COPIES

    root, wtroot = fixture[0], sw._resolve_root()
    seed_result(root, "old-task", document("old-task", "completed", child_drive_root=str(root / "task_drives" / "old-task")))
    seed_result(root, "another-task", document("another-task", "completed"))
    env = (wtroot if place == "root" else wtroot / COPIES) / _OLD_ENV
    (env / "venv").mkdir(parents=True)
    (env / "venv" / "old.bin").write_bytes(b"older")
    if _DECLARATIONS[declaration] is not None:
        (env / environment.SCRATCH_MARKER).write_text(_DECLARATIONS[declaration], encoding="utf-8")
    if declaration == "torn":
        (env / environment.SCRATCH_MARKER).chmod(0)
    return root, env


@pytest.mark.parametrize("place", ["root", "copies"])
@pytest.mark.parametrize("declaration", ["absent", "torn", "ownerless", "this_task"])
def test_keep_names_an_old_environment_the_saved_group_would_remove_for_that_task(copies, place, declaration):  # noqa: F811
    """The saved group gates an old environment on its declared owner, else the owner its copy's name carries;
    the keep's inventory names the same object, so a keep is saved and that group's run then leaves the folder."""
    root, env = _old_environment(copies, place, declaration)
    assert environments.prune_environments(root, lambda _task: False, retention_days=0)["removed"] == [], "never enrolled"
    owner = _owner({}, offered=("service_folders",))
    _preview(owner, root, root.parent)  # offered before the keep: the run's own gate must read it
    answer = owner_keep.set_owner_keep(root, "old-task", keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["unknown"]) == ("partly_retired", True, [])
    assert str(env) in answer["retained"] and load_task_result(root, "old-task")[owner_keep.OWNER_KEEP]
    assert _run(owner, root, root.parent, groups=("service_folders",))["run"]["removed"] == 0
    assert [(row["path"], row["outcome"], row["reason"]) for row in _page(owner, copies, "outcomes")["items"]] == [
        (str(env), "kept", "owner_keep")]
    assert (env / "venv" / "old.bin").read_bytes() == b"older"
    _preview(owner, root, root.parent)
    assert [(row["path"], row["reason"]) for row in _page(owner, copies, "excluded")["items"]] == [(str(env), "owner_keep")]
    assert owner_keep.set_owner_keep(root, "old-task", keep=False, source=KEEP)["status"] == "released"
    _preview(owner, root, root.parent)
    assert _run(owner, root, root.parent, groups=("service_folders",))["run"]["removed"] == 1 and not env.exists()


@pytest.mark.parametrize("declaration", ["none", "another_task"])
def test_keep_of_a_task_with_no_old_environment_left_is_still_already_retired(copies, declaration):  # noqa: F811
    """Nothing of this task remains: none at all, or a folder whose declaration names another task, which the
    saved group gates on that task and not on the name."""
    root, env = _old_environment(copies, "copies", "another_task")
    if declaration == "none":
        sw._force_rmtree(env)
    answer = owner_keep.set_owner_keep(root, "old-task", keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["retained"]) == ("already_retired", False, [])
    assert owner_keep.OWNER_KEEP not in load_task_result(root, "old-task")


def _lost_copy(fixture, place):
    """A finished task's drive is gone and its copy's record too; the copy's folder remains."""
    from ouroboros.unindexed import COPIES

    root, wtroot = fixture[0], sw._resolve_root()
    seed_result(root, "old-task", document("old-task", "completed", child_drive_root=str(root / "task_drives" / "old-task")))
    seed_result(root, "another-task", document("another-task", "completed"))
    lost = (wtroot if place == "root" else wtroot / COPIES) / _OLD_ENV[:-len(".env")]
    lost.mkdir(parents=True)
    (lost / "f.bin").write_bytes(b"lost")
    return root, lost


@pytest.mark.parametrize("place", ["root", "copies"])
def test_keep_names_a_lost_copy_the_saved_group_would_remove_for_that_task(copies, place):  # noqa: F811
    """A folder no record names goes with the lost-copies group, gated on the task its name carries: the keep names it."""
    root, lost = _lost_copy(copies, place)
    owner = _owner({}, offered=("lost_copies",))
    _preview(owner, root, root.parent)
    answer = owner_keep.set_owner_keep(root, "old-task", keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["unknown"]) == ("partly_retired", True, [])
    assert str(lost) in answer["retained"] and load_task_result(root, "old-task")[owner_keep.OWNER_KEEP]
    assert _run(owner, root, root.parent, groups=("lost_copies",))["run"]["removed"] == 0
    assert [(row["path"], row["outcome"], row["reason"]) for row in _page(owner, copies, "outcomes")["items"]] == [
        (str(lost), "kept", "owner_keep")]
    assert (lost / "f.bin").read_bytes() == b"lost"
    assert owner_keep.set_owner_keep(root, "old-task", keep=False, source=KEEP)["status"] == "released"
    _preview(owner, root, root.parent)
    assert _run(owner, root, root.parent, groups=("lost_copies",))["run"]["removed"] == 1 and not lost.exists()


@pytest.mark.parametrize("fault", ["readable", "unreadable", "unlistable"])
def test_keep_never_claims_by_name_a_folder_a_record_gives_another_task(copies, monkeypatch, fault):  # noqa: F811
    import os

    root, lost = _lost_copy(copies, "copies")
    sw._put_record({"path": str(lost.resolve()), "task_id": "another-task", "kind": "plain"}, root)
    record_path = sw._record_file(lost, root)
    before = record_path.read_bytes()
    parse, listdir = sw._parse_record, os.listdir

    def read_record(path):
        if fault == "unreadable" and path == record_path:
            raise PermissionError("record unavailable")
        return parse(path)

    def list_records(path="."):
        if fault == "unlistable" and Path(path) == sw._records_dir(root):
            raise PermissionError("record directory unavailable")
        return listdir(path)

    monkeypatch.setattr(sw, "_parse_record", read_record)
    monkeypatch.setattr(os, "listdir", list_records)
    answer = owner_keep.set_owner_keep(root, "old-task", keep=True, source=KEEP)
    assert answer["retained"] == [], "unknown registration cannot positively attribute a copy by its old name"
    assert record_path.read_bytes() == before and (lost / "f.bin").read_bytes() == b"lost"
    if fault == "readable":
        assert (answer["status"], answer["keep"], answer["unknown"]) == ("already_retired", False, [])
    else:
        assert answer["keep"] and answer["unknown"] == ["private_copies"]
        assert load_task_result(root, "old-task")[owner_keep.OWNER_KEEP]


REASON = "evidence a failed teardown kept"


def test_a_retained_old_environment_a_saved_group_offers_is_counted_by_that_group(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    (env_root / "tmp").mkdir(parents=True, exist_ok=True)
    assert environment.retain_tree(env_root / "tmp", REASON)
    sw._force_rmtree(Path(ctx.repo_dir))
    live = lambda _task: False  # noqa: E731
    assert environments.environment_verdict(env_root, data, live) == "retained_evidence"
    owner = FreeSpace(live=live, offered=("service_folders",))
    _preview(owner, data, data.parent)
    view = owner.projection(data)
    assert view["pool"]["count"] == 0 and view["pool"]["excluded"] == [{"reason": "advanced_group", "count": 1}]
    assert _group(view, "service_folders")["count"] == 1
    assert retire_synthetic_retention(env_root / "tmp", REASON)


# ---- one name in both copy roots: two objects, two ids ------------------------------------------------


def test_a_press_on_the_new_environment_never_reaches_the_older_one_of_the_same_name(candidate, monkeypatch):  # noqa: F811
    """A declared environment in the new copies' root and an older undeclared one of the same name at the root:
    the pool offers only the first, the older one only its saved group, and pressing either id reaches only it."""
    from ouroboros.unindexed import COPIES

    ctx, data, env_root = _gone_copy(candidate)
    wtroot = sw._resolve_root()
    older = wtroot / env_root.name
    assert env_root.parent == wtroot / COPIES and not older.exists()
    (older / "venv").mkdir(parents=True)
    (older / environment.SCRATCH_MARKER).write_text("body-process\n", encoding="utf-8")  # declared by no one
    (older / "venv" / "old.bin").write_bytes(b"older")
    _fake_sizer(monkeypatch)
    env, owner = (data, data.parent), FreeSpace(live=lambda _task: False, offered=("service_folders",))
    _preview(owner, data, data.parent)
    [pooled] = _page(owner, env, "pool")["items"]
    [saved] = _page(owner, env, "excluded")["items"]
    assert (pooled["path"], saved["path"], saved["reason"]) == (str(env_root), str(older), "advanced_group")
    assert _start(owner, env, items=[pooled["id"]]) == {"started": True, "reason": ""}
    _join(owner)
    assert not env_root.exists() and (older / "venv" / "old.bin").read_bytes() == b"older", "the unselected one stays"
    assert [(row["id"], row["outcome"]) for row in _page(owner, env, "outcomes")["items"]] == [(pooled["id"], "removed")]

    _preview(owner, data, data.parent)
    assert _page(owner, env, "pool")["total"] == 0
    [saved] = _page(owner, env, "excluded")["items"]
    assert _start(owner, env, items=[saved["id"]]) == {"started": False, "reason": "not_offered"}
    assert older.is_dir(), "an ordinary press cannot reach a saved-group object"
    assert _run(owner, data, data.parent, groups=("service_folders",))["run"]["removed"] == 1 and not older.exists()


def test_lost_copies_and_test_runs_of_one_name_in_both_roots_are_named_and_settled_apart(copies, monkeypatch):  # noqa: F811
    """Every id of the plan names one object; a group run reports each object under its own id."""
    from ouroboros.unindexed import COPIES

    root, wtroot = copies[0], sw._resolve_root()
    name = "dlg_lost-task_0123456789abcdef"
    lost = {"older": wtroot / name, "new": wtroot / COPIES / name}
    runs = {"older": wtroot / ".test-evidence" / "run-1", "new": wtroot / COPIES / ".test-evidence" / "run-1"}
    for folder in (*lost.values(), *runs.values()):
        folder.mkdir(parents=True)
        (folder / "f.bin").write_bytes(b"f" * 64)
    for run in runs.values():
        (run / "result.json").write_text("{}", encoding="utf-8")
    owner = _owner({"lost-task": False}, offered=("lost_copies", "service_folders"))
    _preview(owner, root, root.parent)
    rows = [row for view in ("pool", "excluded") for row in _page(owner, copies, view)["items"]]
    assert sorted(row["path"] for row in rows) == sorted(str(path) for path in (*lost.values(), *runs.values()))
    assert len({row["id"] for row in rows}) == len(rows), "one id is one object"
    # The newer lost folder gains a record after the preview: the run keeps it and removes only the older one.
    sw._put_record({"path": str(lost["new"].resolve()), "task_id": "lost-task", "kind": "plain"}, root)
    view = _run(owner, root, root.parent, groups=("lost_copies", "service_folders"))
    rows = _page(owner, copies, "outcomes")["items"]
    assert {row["path"]: (row["outcome"], row["reason"]) for row in rows} == {
        str(lost["older"]): ("removed", None), str(lost["new"]): ("kept", "in_use"),
        str(runs["older"]): ("removed", None), str(runs["new"]): ("removed", None)}
    assert view["run"]["removed"] == 3 and view["run"]["removed_bytes"] == sum(row["bytes"] for row in rows if row["outcome"] == "removed")
    assert not lost["older"].exists() and (lost["new"] / "f.bin").is_file()


# ---- what the run says about a copy it left: the retirer's own reason, never a failed deletion -------


def _outcome(owner, env):
    [row] = _page(owner, env, "outcomes")["items"]
    return row["outcome"], row["reason"]


def _snapshot(env, status="completed"):
    root, repo, wtroot = env
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", status))
    return handle


def test_a_snapshot_live_again_after_its_preview_is_in_use_not_a_failed_deletion(copies):  # noqa: F811
    handle = _snapshot(copies)
    probes = {"finished": False}
    owner = _owner(probes)
    assert _preview(owner, copies[0], copies[0].parent)["pool"]["count"] == 1
    probes["finished"] = True
    answer = _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    assert Path(handle.path).is_dir() and answer["run"]["failed"] == 0
    assert _outcome(owner, copies) == ("kept", "in_use")


def _custody_answers_held(_root):
    return {"snap"}


def _custody_unreadable(_root):
    raise OSError("no")


@pytest.mark.parametrize("custody,reason", [(_custody_answers_held, "in_use"), (_custody_unreadable, "unreadable_record")])
def test_a_snapshot_custody_took_back_or_cannot_answer_for_is_no_failed_deletion(copies, monkeypatch, custody, reason):  # noqa: F811
    from ouroboros import delegate_custody

    handle = _snapshot(copies)
    owner = _owner({"finished": False})
    assert _preview(owner, copies[0], copies[0].parent)["pool"]["count"] == 1
    monkeypatch.setattr(delegate_custody, "open_snapshot_ids", custody)
    _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    assert Path(handle.path).is_dir() and _outcome(owner, copies) == ("kept", reason)


def test_a_snapshot_whose_removal_failed_is_still_a_failed_deletion(copies, monkeypatch):  # noqa: F811
    handle = _snapshot(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    monkeypatch.setattr(sw, "_force_rmtree", lambda _path: None)
    answer = _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    assert Path(handle.path).is_dir() and answer["run"]["failed"] == 1 and _outcome(owner, copies) == ("failed", None)


def test_a_plain_copy_whose_removal_failed_is_a_failed_deletion_not_a_changed_one(copies, monkeypatch):  # noqa: F811
    handle = _copy(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    monkeypatch.setattr(sw, "_force_rmtree", lambda _path: None)
    monkeypatch.setattr(sw, "_remove_paths", lambda *_args, **_kwargs: False)
    answer = _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    assert Path(handle.path).is_dir() and answer["run"]["failed"] == 1 and _outcome(owner, copies) == ("failed", None)


def _held(lock):
    """Hold a lock on another thread while the block runs (the locks are per process and thread alike)."""
    entered, leave = threading.Event(), threading.Event()

    def hold():
        with lock():
            entered.set()
            leave.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(10)
    return leave, thread


@pytest.mark.parametrize("make", ["plain", "snapshot"])
def test_custody_held_by_another_during_the_removal_is_busy_not_a_failed_deletion(copies, make):  # noqa: F811
    handle = _copy(copies) if make == "plain" else _snapshot(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    leave, thread = _held(lambda: task_custody.task_custody_lock(copies[0], "finished"))
    try:
        answer = _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    finally:
        leave.set()
        thread.join(10)
    assert Path(handle.path).is_dir() and answer["run"]["failed"] == 0 and _outcome(owner, copies) == ("kept", "busy")


@pytest.mark.parametrize("make", ["plain", "snapshot"])
def test_a_record_lock_held_by_another_is_busy_and_the_run_goes_on(copies, make, monkeypatch):  # noqa: F811
    handle = _copy(copies) if make == "plain" else _snapshot(copies)
    owner = _owner({"finished": False})
    _preview(owner, copies[0], copies[0].parent)
    monkeypatch.setattr(sw, "_LOCK_TIMEOUT_SEC", 0.2)
    leave, thread = _held(lambda: sw._record_lock(handle.path, copies[0], op="test"))
    try:
        answer = _run(owner, copies[0], copies[0].parent, groups=("working_copies",))
    finally:
        leave.set()
        thread.join(10)
    assert Path(handle.path).is_dir() and answer["run"]["error"] is None and answer["state"] == "done"
    assert _outcome(owner, copies) == ("kept", "busy")


def test_a_candidate_with_retained_evidence_stays_out_of_the_pool_and_the_preview_writes_nothing(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    (env_root / "tmp").mkdir(parents=True, exist_ok=True)
    assert environment.retain_tree(env_root / "tmp", REASON)
    _fake_sizer(monkeypatch)
    record = sw._record_file(ctx.repo_dir, data)
    before = record.read_bytes()
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    view = owner.projection(data)
    assert view["pool"]["count"] == 0 and view["pool"]["excluded"] == [{"reason": "retained_evidence", "count": 1}]
    assert record.read_bytes() == before
    assert retire_synthetic_retention(env_root / "tmp", REASON)


def test_a_candidate_retained_after_its_preview_is_kept_with_that_reason(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    assert owner.projection(data)["pool"]["count"] == 1
    (env_root / "tmp").mkdir(parents=True, exist_ok=True)
    assert environment.retain_tree(env_root / "tmp", REASON)
    _run(owner, data, data.parent, groups=("working_copies",))
    assert Path(ctx.repo_dir).is_dir() and env_root.is_dir()
    assert _outcome(owner, (data, data.parent)) == ("kept", "retained_evidence")
    assert retire_synthetic_retention(env_root / "tmp", REASON)


def test_a_candidate_whose_owner_ran_again_is_in_use(candidate, monkeypatch):  # noqa: F811
    ctx, data, _env_root = candidate
    _fake_sizer(monkeypatch)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    seed_result(data, ctx.task_id, document(ctx.task_id, "running"))
    _run(owner, data, data.parent, groups=("working_copies",))
    assert Path(ctx.repo_dir).is_dir() and _outcome(owner, (data, data.parent)) == ("kept", "in_use")


def test_names_come_from_the_narrow_display_read(copies, monkeypatch):  # noqa: F811
    from ouroboros import task_store

    handle = _copy(copies)
    seed_result(copies[0], "finished", {**document("finished", "completed"), "title": "The saved title", "result": "x" * 4096})

    def full_read(*_args, **_kwargs):
        raise AssertionError("names need no full document")

    owner = _owner({"finished": False})
    monkeypatch.setattr(task_store, "documents_by_ids", full_read)
    _preview(owner, copies[0], copies[0].parent)
    [row] = _page(owner, copies, "pool")["items"]
    assert row["path"] == handle.path and row["name"] == "The saved title"


# ---- one reader of the private-copy records ----------------------------------------------------------


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"path": ""}', b"\xff\xfe", b'"text"'])
def test_passive_and_owning_readers_refuse_the_same_records(copies, raw):  # noqa: F811
    from ouroboros.private_copy_read import checkout_registration, read_records

    handle = _copy(copies)
    file = sw._record_file(handle.path, copies[0])
    file.write_bytes(raw)
    with pytest.raises(sw.SubagentWorktreeRegistryCorrupt):
        sw._read_record(file, data_dir=copies[0])
    assert file.name in read_records(copies[0]).unreadable and checkout_registration(handle.path, copies[0]) == "unknown"


def test_the_passive_reader_follows_the_owners_rule_for_a_record_and_a_legacy_file(copies, monkeypatch):  # noqa: F811
    from ouroboros.private_copy_read import read_records

    handle = _copy(copies)
    root = copies[0]
    monkeypatch.setattr(sw, "_parse_record", lambda _file: (_ for _ in ()).throw(ValueError("a new rule")))
    assert read_records(root).unreadable == {sw._record_file(handle.path, root).name}
    monkeypatch.undo()
    sw._legacy_registry_path(root).write_text(json.dumps({"worktrees": [{"path": "/elsewhere/copy"}, {"path": ""}, 3]}), encoding="utf-8")
    assert "/elsewhere/copy" in {row["path"] for row in read_records(root).rows}
    monkeypatch.setattr(sw, "_legacy_entries", lambda _raw: (_ for _ in ()).throw(ValueError("a new rule")))
    assert read_records(root).legacy_unreadable


def test_the_genesis_owner_imports_one_way_and_the_old_names_still_answer():
    import ast
    import subprocess
    import sys

    from ouroboros import genesis_project

    assert sw.provision_genesis_project is genesis_project.provision_genesis_project
    assert sw.remove_genesis_project is genesis_project.remove_genesis_project
    module_level = [node for node in ast.parse(Path(sw.__file__).read_text(encoding="utf-8")).body
                    if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not [node for node in module_level if "genesis_project" in ast.dump(node)]
    for first in ("genesis_project", "subagent_worktrees"):
        subprocess.run([sys.executable, "-c", f"import ouroboros.{first}; from ouroboros.subagent_worktrees import remove_genesis_project"],
                       check=True, cwd=Path(sw.__file__).parents[1])
