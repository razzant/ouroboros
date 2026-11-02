"""The owner's explicit discard of a finished copy no capture holds: only by exact id with the loss accepted,
re-judged at the effect, and unreachable from the pool, the saved groups and every background pass."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import body_candidate as body
from ouroboros import owner_keep, owner_pause, task_custody, test_environment as environment
from ouroboros import subagent_worktrees as sw
from ouroboros.gateway.storage import storage_routes
from ouroboros.headless import task_artifacts_dir
from ouroboros.storage_free import FreeSpace
from ouroboros.task_results import load_task_result
from tests._shared import retire_synthetic_retention
from tests._store_shared import seed_result
from tests.test_body_candidate_retention import candidate  # noqa: F401 -- the body fixture
from tests.test_storage_free import _fake_sizer, _join, _owner, _preview, _run
from tests.test_storage_free_safety import _capture, _copy, _git, _prune, copies  # noqa: F401 -- the plain copies fixture
from tests.test_store import document

KEEP = {"basis": "test"}
REASON = "evidence a failed teardown kept"


def _held(env, task_id="finished", rel=".venv/bin/python"):
    """A finished copy whose saved capture holds one note and no capture holds ``rel``."""
    handle = _copy(env, task_id)
    path = Path(handle.path)
    (path / "note.txt").write_text("saved\n", encoding="utf-8")
    _capture(env, handle, task_id)
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"only here")
    return handle, path, target


def _ready(env, probes=None):
    owner = _owner({"finished": False, **(probes or {})})
    _preview(owner, env[0], env[0].parent)
    return owner


def _page(owner, env, view="manual"):
    return owner.items(env[0], owner.projection(env[0])["plan"], view, 0, 500, None)


def _discard(owner, env, ids=None, accept=True):
    view = owner.projection(env[0])
    ids = [row["id"] for row in _page(owner, env)["items"]] if ids is None else ids
    answer = owner.start(env[0], env[1], view["plan"], None, None, manual=ids, accept_loss=accept)
    _join(owner)
    return answer


def _outcome(owner, env):
    [row] = _page(owner, env, "outcomes")["items"]
    return row["outcome"], row["reason"]


def _kept_for(owner, env, path, reason):
    assert _discard(owner, env) == {"started": True, "reason": ""}
    assert path.is_dir() and _outcome(owner, env) == ("kept", reason)
    assert owner.projection(env[0])["run"]["failed"] == 0


# ---- what is offered, and to whom --------------------------------------------------------------------


def test_an_unproven_copy_is_listed_with_its_exact_loss_and_counted_apart_from_the_pool(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    pool = owner.projection(copies[0])["pool"]
    assert pool["count"] == 0 and pool["manual"]["count"] == 1 and pool["manual"]["bytes"] > 0
    assert pool["excluded"] == [{"reason": "capture_unproven", "count": 1, "manual_count": 1}]
    [row] = _page(owner, copies)["items"]
    assert row["path"] == str(path) and row["reason"] == "capture_unproven"
    assert row["manual"] == {"not_proven_saved_count": 1, "not_proven_saved_bytes": 9, "not_proven_saved": [{"path": ".venv/bin/python", "bytes": 9}],
                             "omitted_count": 0, "keeps_commits": False}, "the saved note is no loss"
    [excluded] = _page(owner, copies, "excluded")["items"]
    assert excluded["availability"] == row["availability"] == "manual_discard" and "manual" not in excluded


def test_availability_says_how_each_listed_object_can_go(copies):  # noqa: F811
    (copies[2] / "copies.noindex").mkdir(parents=True)
    (copies[2] / "dlg_lost-task_0123456789abcdef").mkdir()
    _copy(copies, "clean")
    _held(copies)
    held_kept = _copy(copies, "kept")
    (Path(held_kept.path) / "scratch.tmp").write_bytes(b"x")
    owner_keep.set_owner_keep(copies[0], "kept", keep=True, source=KEEP)
    live = _copy(copies, "live")
    (Path(live.path) / "scratch.tmp").write_bytes(b"x")
    owner = _ready(copies, {"live": True, "clean": False, "kept": False, "lost-task": False})
    assert {row["availability"] for row in _page(owner, copies, "pool")["items"]} == {"ordinary"}
    seen = {Path(row["path"]).name: row["availability"] for row in _page(owner, copies, "excluded")["items"]}
    assert seen == {"finished": "manual_discard", "live": "none", "kept": "none", "dlg_lost-task_0123456789abcdef": "saved_group"}
    assert owner.projection(copies[0])["pool"]["manual"]["count"] == 1
    assert {Path(row["path"]).name: row["reason"] for row in _page(owner, copies, "excluded")["items"]}["kept"] == "owner_keep"


def test_a_path_a_verified_file_result_holds_is_no_loss(copies):  # noqa: F811
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "blob.bin").write_bytes(bytes(range(256)) * 8)
    _capture(copies, handle)
    (path / "scratch.tmp").write_bytes(b"x")
    [row] = _page(_ready(copies), copies)["items"]
    assert [entry["path"] for entry in row["manual"]["not_proven_saved"]] == ["scratch.tmp"]


def test_only_the_largest_paths_are_named_and_the_count_stays_whole(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    for number in range(60):
        (path / f"s{number:02}.tmp").write_bytes(b"x" * (number + 1))
    [row] = _page(_ready(copies), copies)["items"]
    manual = row["manual"]
    assert manual["not_proven_saved_count"] == 61 and len(manual["not_proven_saved"]) == 50 and manual["omitted_count"] == 11
    assert manual["not_proven_saved"][0] == {"path": "s59.tmp", "bytes": 60}


def test_a_copy_with_no_capture_at_all_is_offered_and_all_its_work_is_named(copies):  # noqa: F811
    handle = _copy(copies)
    (Path(handle.path) / "work.txt").write_text("only here\n", encoding="utf-8")
    [row] = _page(_ready(copies), copies)["items"]
    assert [entry["path"] for entry in row["manual"]["not_proven_saved"]] == ["work.txt"]


def test_nothing_else_reaches_it(copies, monkeypatch):  # noqa: F811
    """Not the background, not the ordinary pool, not a whole group, and nothing unchecked by default."""
    _handle, path, target = _held(copies)
    assert _prune(copies)["removed"] == 0
    owner = _ready(copies)
    [row] = _page(owner, copies)["items"]
    assert owner.start(copies[0], copies[1], owner.projection(copies[0])["plan"], None, [row["id"]]) == {"started": False, "reason": "not_offered"}
    assert _run(owner, copies[0], copies[0].parent, groups=("working_copies", "lost_copies", "service_folders"))["run"]["removed"] == 0
    assert target.read_bytes() == b"only here" and path.is_dir()


def test_the_background_verdict_is_unchanged_by_a_discard_answer(candidate):  # noqa: F811
    ctx, data, _env_root = candidate
    (Path(ctx.repo_dir) / "__pycache__").mkdir()
    (Path(ctx.repo_dir) / "__pycache__" / "a.pyc").write_bytes(b"x")
    row = next(row for row in sw.list_worktrees(data) if row.get("kind") == "body_candidate")
    assert body.retention_verdict(row, expired=True, data_dir=data)["reason"] == "unproven_excluded_inputs"
    assert sw.prune_orphans(retention_days=0, data_dir=data, live=lambda _t: False)["removed"] == 0


@pytest.mark.parametrize("ids,accept,reason", [
    (None, False, "loss_not_accepted"), (None, None, "loss_not_accepted"), (None, "true", "loss_not_accepted"), (None, 1, "loss_not_accepted"),
    ([], True, "nothing_selected"), ("x", True, "unknown_item"), ([1], True, "unknown_item"), (["nope"], True, "unknown_item")])
def test_a_refused_request_touches_nothing(copies, ids, accept, reason):  # noqa: F811
    _handle, path, target = _held(copies)
    owner = _ready(copies)
    assert _discard(owner, copies, ids, accept) == {"started": False, "reason": reason}
    assert target.is_file() and path.is_dir() and owner.projection(copies[0])["state"] == "ready"


def test_an_id_of_another_list_and_a_second_form_are_refused(copies):  # noqa: F811
    _held(copies)
    other = _copy(copies, "live")
    owner = _ready(copies, {"live": True})
    good = [row["id"] for row in _page(owner, copies)["items"]]
    plan = owner.projection(copies[0])["plan"]
    assert len(good) == 1 and Path(other.path).is_dir()
    live = next(row["id"] for row in _page(owner, copies, "excluded")["items"] if row["id"] not in good)
    assert _discard(owner, copies, [*good, live]) == {"started": False, "reason": "not_offered"}
    for form in ({"items": good}, {"groups": ["working_copies"]}):
        answer = owner.start(copies[0], copies[1], plan, form.get("groups"), form.get("items"), manual=good, accept_loss=True)
        assert answer == {"started": False, "reason": "selection_conflict"}


# ---- the effect, and what survives it ----------------------------------------------------------------


def test_the_discard_removes_exactly_that_copy_and_keeps_every_saved_fact(copies):  # noqa: F811
    handle, path, _target = _held(copies)
    _other, other_path, other_target = _held(copies, "second")
    owner = _ready(copies, {"second": False})
    chosen = next(row["id"] for row in _page(owner, copies)["items"] if row["path"] == str(path))
    artifacts = task_artifacts_dir(copies[0], "finished")
    saved = sorted(file.name for file in artifacts.iterdir())
    assert _discard(owner, copies, [chosen]) == {"started": True, "reason": ""}
    assert not path.exists() and sw._record_at(handle.path, copies[0]) is None
    assert _outcome(owner, copies) == ("removed", None)
    assert other_target.read_bytes() == b"only here" and other_path.is_dir()
    assert saved and sorted(file.name for file in artifacts.iterdir()) == saved
    assert load_task_result(copies[0], "finished")["status"] == "completed"
    run = owner.projection(copies[0])["run"]
    assert (run["removed"], run["selected"], run["groups"]) == (1, 1, [])


def test_the_discard_leaves_a_record_of_what_was_lost(copies, monkeypatch):  # noqa: F811
    from ouroboros import server_maintenance

    _held(copies)
    owner = _ready(copies)
    [row] = _page(owner, copies)["items"]
    events = []
    monkeypatch.setattr(server_maintenance, "_prune_event", lambda *args, **kwargs: events.append(kwargs["report"]))
    _discard(owner, copies)
    assert events[0]["manual_discards"] == {row["id"]: {"unsaved_count": 1, "unsaved_bytes": 9}}
    assert row["id"] in events[0]["removed_ids"]


def test_commits_on_its_branch_stay_with_the_branch(copies):  # noqa: F811
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "work.txt")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "work")
    tip = _git(path, "rev-parse", "HEAD").strip()
    (path / "scratch.tmp").write_bytes(b"x")
    owner = _ready(copies)
    [row] = _page(owner, copies)["items"]
    assert row["manual"]["keeps_commits"] is True
    assert _discard(owner, copies)["started"] and not path.exists()
    assert _git(copies[1], "rev-parse", f"refs/heads/{handle.branch}").strip() == tip


def test_commits_alone_lose_nothing_and_the_branch_stays(copies):  # noqa: F811
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "work.txt")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "work")
    tip = _git(path, "rev-parse", "HEAD").strip()
    owner = _ready(copies)
    [row] = _page(owner, copies)["items"]
    assert row["manual"]["not_proven_saved_count"] == 0 and row["manual"]["keeps_commits"] is True
    assert _discard(owner, copies)["started"] and not path.exists()
    assert _git(copies[1], "rev-parse", f"refs/heads/{handle.branch}").strip() == tip


def test_a_branch_with_no_commits_of_its_own_goes_with_the_copy(copies):  # noqa: F811
    handle, path, _target = _held(copies)
    owner = _ready(copies)
    assert _discard(owner, copies)["started"] and not path.exists()
    assert subprocess.run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{handle.branch}"], cwd=copies[1]).returncode != 0


def test_a_detached_head_with_its_own_commits_is_not_offered(copies):  # noqa: F811
    handle = _copy(copies)
    path = Path(handle.path)
    (path / "work.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "work.txt")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "work")
    _git(path, "checkout", "-q", "--detach")
    _git(path, "branch", "-f", handle.branch, handle.base_sha)
    assert _page(_ready(copies), copies)["total"] == 0


def test_a_nested_repository_is_not_offered(copies):  # noqa: F811
    handle = _copy(copies)
    nested = Path(handle.path) / "vendor"
    nested.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
    (nested / "f").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f"], cwd=nested, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "own history"], cwd=nested, check=True)
    assert _page(_ready(copies), copies)["total"] == 0


def test_a_delegated_snapshot_is_never_offered(copies):  # noqa: F811
    root, repo, wtroot = copies
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", "completed"))
    (Path(handle.path) / "scratch.tmp").write_bytes(b"x")
    assert _page(_ready(copies), copies)["total"] == 0


# ---- the preview no longer stands -----------------------------------------------------------------


@pytest.mark.parametrize("change", ["new_file", "new_empty_folder", "edit_same_size", "saved_file_edited", "staged", "commit", "removed"])
def test_any_write_since_the_preview_keeps_the_copy_as_changed(copies, change):  # noqa: F811
    _handle, path, target = _held(copies)
    owner = _ready(copies)
    if change == "new_file":
        (path / "late.txt").write_text("arrived\n", encoding="utf-8")
    elif change == "new_empty_folder":
        (path / "late").mkdir()
    elif change == "edit_same_size":
        target.write_bytes(b"ONLY HERE")
    elif change == "saved_file_edited":
        (path / "note.txt").write_text("other\n", encoding="utf-8")
    elif change == "staged":
        _git(path, "add", "note.txt")
    elif change == "commit":
        _git(path, "add", "note.txt")
        _git(path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "late")
    else:
        target.unlink()
    _kept_for(owner, copies, path, "changed")
    assert sw._record_at(path, copies[0]) is not None


def test_a_record_rewritten_since_the_preview_keeps_the_copy_as_changed(copies):  # noqa: F811
    handle, path, target = _held(copies)
    owner = _ready(copies)
    sw._put_record({**sw._record_at(handle.path, copies[0]), "task_id": "someone-else"}, copies[0])
    _kept_for(owner, copies, path, "changed")
    assert target.read_bytes() == b"only here"


def test_an_unreadable_record_keeps_the_copy(copies):  # noqa: F811
    handle, path, _target = _held(copies)
    owner = _ready(copies)
    sw._record_file(handle.path, copies[0]).write_text("{", encoding="utf-8")
    _kept_for(owner, copies, path, "unreadable_record")


def test_a_record_lock_or_custody_held_by_another_is_busy(copies, monkeypatch):  # noqa: F811
    import threading

    handle, path, _target = _held(copies)
    owner = _ready(copies)
    monkeypatch.setattr(sw, "_LOCK_TIMEOUT_SEC", 0.2)
    entered, leave = threading.Event(), threading.Event()

    def hold():
        with sw._record_lock(handle.path, copies[0], op="test"):
            entered.set()
            leave.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(10)
    try:
        _kept_for(owner, copies, path, "busy")
    finally:
        leave.set()
        thread.join(10)


def test_custody_taken_during_the_removal_is_busy(copies):  # noqa: F811
    import threading

    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    entered, leave = threading.Event(), threading.Event()

    def hold():
        with task_custody.task_custody_lock(copies[0], "finished"):
            entered.set()
            leave.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(10)
    try:
        _kept_for(owner, copies, path, "busy")
    finally:
        leave.set()
        thread.join(10)


# ---- keep, live, pause, custody, evidence ------------------------------------------------------------


def test_a_kept_task_is_not_offered_and_a_keep_after_the_preview_holds(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    assert _page(owner, copies)["total"] == 1
    assert owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP)["keep"] is True
    _kept_for(owner, copies, path, "owner_keep")
    assert _page(_ready(copies), copies)["total"] == 0


def test_a_live_task_is_not_offered_and_one_live_again_after_the_preview_stays(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    assert _page(_ready(copies, {"finished": True}), copies)["total"] == 0
    assert _page(_ready(copies, {"finished": None}), copies)["total"] == 0
    probes = {"finished": False}
    owner = _owner(probes)
    _preview(owner, copies[0], copies[0].parent)
    assert _page(owner, copies)["total"] == 1
    probes["finished"] = True
    _kept_for(owner, copies, path, "in_use")


def test_a_task_running_again_or_with_a_live_process_stays(copies, monkeypatch):  # noqa: F811
    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    seed_result(copies[0], "finished", document("finished", "running"))
    _kept_for(owner, copies, path, "in_use")
    seed_result(copies[0], "finished", document("finished", "completed"))
    monkeypatch.setattr(body, "live_lineage_processes", lambda *_args: [4242])
    assert _page(_ready(copies), copies)["total"] == 0


@pytest.mark.parametrize("state", ["requested", "paused"])
def test_a_paused_tree_is_not_offered_and_a_pause_after_the_preview_holds(copies, monkeypatch, state):  # noqa: F811
    _handle, path, _target = _held(copies)
    seed_result(copies[0], "finished", {**document("finished", "completed"), "root_task_id": "the-root"})
    owner = _ready(copies)
    assert _page(owner, copies)["total"] == 1
    monkeypatch.setattr(owner_pause, "read_fence", lambda _root, task: {"state": state} if task == "the-root" else {})
    _kept_for(owner, copies, path, "in_use")
    assert _page(_ready(copies), copies)["total"] == 0


def test_a_released_pause_does_not_refuse(copies, monkeypatch):  # noqa: F811
    _held(copies)
    monkeypatch.setattr(owner_pause, "read_fence", lambda _root, _task: {"state": "released"})
    assert _page(_ready(copies), copies)["total"] == 1


def _held_run(**facts):
    return {"run": SimpleNamespace(**{"task_id": "other", "snapshot_task_id": "", "execution_root": "", "target_root": "", **facts})}


@pytest.mark.parametrize("facts", [{"task_id": "finished"}, {"snapshot_task_id": "finished"}, {"execution_root": "HERE"}, {"target_root": "HERE/sub"}])
def test_a_held_delegated_run_on_the_task_or_inside_the_copy_stays(copies, monkeypatch, facts):  # noqa: F811
    from ouroboros import delegate_custody_current as current

    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    assert _page(owner, copies)["total"] == 1
    bound = {key: value.replace("HERE", str(path)) for key, value in facts.items()}
    monkeypatch.setattr(current, "held_runs", lambda _root: _held_run(**bound))
    _kept_for(owner, copies, path, "in_use")


def test_a_held_run_of_another_task_elsewhere_does_not_refuse(copies, monkeypatch):  # noqa: F811
    from ouroboros import delegate_custody_current as current

    _held(copies)
    monkeypatch.setattr(current, "held_runs", lambda _root: _held_run(execution_root="/elsewhere/copy"))
    assert _page(_ready(copies), copies)["total"] == 1


def test_unreadable_custody_refuses(copies, monkeypatch):  # noqa: F811
    from ouroboros import delegate_custody_current as current

    def unreadable(_root):
        raise OSError("no")

    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    monkeypatch.setattr(current, "held_runs", unreadable)
    _kept_for(owner, copies, path, "unreadable_record")


def test_retention_inside_the_copy_or_its_environment_is_never_discarded(copies):  # noqa: F811
    _handle, path, target = _held(copies)
    held = path / "kept-evidence"
    held.mkdir()
    assert environment.retain_tree(held, REASON)
    assert _page(_ready(copies), copies)["total"] == 0
    assert retire_synthetic_retention(held, REASON)
    env_root = path.with_name(path.name + ".env")
    (env_root / "tmp").mkdir(parents=True)
    owner_env = env_root / environment.SCRATCH_MARKER
    owner_env.write_text("body-process\n", encoding="utf-8")
    assert environment.retain_tree(env_root / "tmp", REASON)
    assert _page(_ready(copies), copies)["total"] == 0
    assert target.is_file() and retire_synthetic_retention(env_root / "tmp", REASON)


def test_evidence_retained_after_the_preview_stays_with_its_reason(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    held = path / "kept-evidence"
    held.mkdir()
    (held / ".gitkeep").write_text("", encoding="utf-8")
    assert environment.retain_tree(held, REASON)
    _kept_for(owner, copies, path, "retained_evidence")
    assert (held / environment.RETENTION_MARKER).is_file() and retire_synthetic_retention(held, REASON)


# ---- an own-body candidate: its work goes on its pin first -------------------------------------------


def _body_row(data):
    return next(row for row in sw.list_worktrees(data) if row.get("kind") == "body_candidate")


def test_a_candidates_work_is_pinned_and_only_the_named_files_are_lost(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    repo = Path(ctx.repo_dir)
    (repo / "work.txt").write_text("unique\n", encoding="utf-8")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "a.pyc").write_bytes(b"x")
    (repo / ".cache").mkdir()
    (repo / ".cache" / "big.bin").write_bytes(b"c" * 4096)  # excluded by name, and no .gitignore line covers it
    row = _body_row(data)
    env = (data, data.parent)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    [entry] = _page(owner, env)["items"]
    assert entry["reason"] == "unproven_excluded_inputs" and entry["manual"]["keeps_commits"] is True
    assert [item["path"] for item in entry["manual"]["not_proven_saved"]] == [".cache/big.bin", "__pycache__/a.pyc"]
    assert _discard(owner, env)["started"]
    assert not repo.exists() and _outcome(owner, env) == ("removed", None)
    serving = Path(row["repo_dir"])
    pin = f"{body.PIN_PREFIX}{row['candidate_id']}"
    assert _git(serving, "show", f"{pin}:work.txt") == "unique\n"
    for lost in ("__pycache__/a.pyc", ".cache/big.bin"):  # the accepted loss goes; it is not moved into Git
        assert subprocess.run(["git", "cat-file", "-e", f"{pin}:{lost}"], cwd=serving, capture_output=True).returncode != 0, lost
    assert _git(serving, "rev-parse", "--verify", f"refs/heads/{row['branch']}"), "its branch stays, as the row said"


def test_a_candidate_holding_only_excluded_files_says_its_branch_goes_and_it_does(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    repo = Path(ctx.repo_dir)
    (repo / ".cache").mkdir()
    (repo / ".cache" / "big.bin").write_bytes(b"c" * 4096)
    row = _body_row(data)
    env = (data, data.parent)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    [entry] = _page(owner, env)["items"]
    assert entry["manual"]["keeps_commits"] is False, "no other work: no pin, and its adopted branch goes"
    assert _discard(owner, env)["started"] and not repo.exists()
    serving = Path(row["repo_dir"])
    assert subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{row['branch']}"], cwd=serving,
                          capture_output=True).returncode != 0
    assert subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{body.PIN_PREFIX}{row['candidate_id']}"], cwd=serving,
                          capture_output=True).returncode != 0


def test_a_discard_never_writes_the_excluded_cache_into_git_and_pins_every_tracked_and_staged_change(candidate, monkeypatch):  # noqa: F811
    """The accepted loss never becomes a Git object, not even for a moment; a tracked edit, a deletion, a staged new
    file, a staged version edited again and the loose work all reach the pin."""
    import uuid

    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    repo, row, env = Path(ctx.repo_dir), _body_row(data), (data, data.parent)
    serving = Path(row["repo_dir"])
    (repo / "ouroboros" / "mod_a.py").write_text("GEN = 'EDITED'\n", encoding="utf-8")
    (repo / "ouroboros" / "gone.py").unlink()
    (repo / "added.txt").write_text("staged new\n", encoding="utf-8")
    (repo / "ouroboros" / "mod_b.py").write_text("GEN = 'STAGED'\n", encoding="utf-8")
    _git(repo, "add", "added.txt", "ouroboros/mod_b.py")
    (repo / "ouroboros" / "mod_b.py").write_text("GEN = 'ON_DISK'\n", encoding="utf-8")
    (repo / "work.txt").write_text("unique\n", encoding="utf-8")
    cache = repo / ".cache" / "unique.bin"
    cache.parent.mkdir()
    cache.write_bytes(f"excluded and only here {uuid.uuid4()}\n".encode())
    oid = subprocess.run(["git", "hash-object", "--stdin"], cwd=serving, input=cache.read_bytes(),  # no -w: computed only
                         capture_output=True, check=True).stdout.decode().strip()

    def written() -> bool:
        return subprocess.run(["git", "cat-file", "-e", oid], cwd=serving, capture_output=True).returncode == 0

    assert not written()
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    [entry] = _page(owner, env)["items"]
    assert [item["path"] for item in entry["manual"]["not_proven_saved"]] == [".cache/unique.bin"]
    assert _discard(owner, env)["started"] and not repo.exists() and _outcome(owner, env) == ("removed", None)
    assert not written(), "the accepted loss was never written into the shared object store"
    pin = f"{body.PIN_PREFIX}{row['candidate_id']}"
    assert _git(serving, "show", f"{pin}:ouroboros/mod_a.py") == "GEN = 'EDITED'\n"
    assert _git(serving, "show", f"{pin}:ouroboros/mod_b.py") == "GEN = 'ON_DISK'\n"
    assert _git(serving, "show", f"{pin}^:ouroboros/mod_b.py") == "GEN = 'STAGED'\n", "the staged version is its own commit"
    assert _git(serving, "show", f"{pin}:added.txt") == "staged new\n" and _git(serving, "show", f"{pin}:work.txt") == "unique\n"
    tree = _git(serving, "ls-tree", "-r", "--name-only", pin).split()
    assert "ouroboros/gone.py" not in tree and ".cache/unique.bin" not in tree


def test_the_plan_holds_only_the_listed_paths_with_the_whole_count_and_bytes(copies):  # noqa: F811
    from ouroboros.storage_free_view import UNSAVED_LISTED

    path, total = Path(_copy(copies).path), UNSAVED_LISTED + 10
    for number in range(total):
        (path / f"scratch-{number:03}.tmp").write_bytes(b"x" * (number + 1))
    owner = _ready(copies)
    [row] = _page(owner, copies)["items"]
    manual = row["manual"]
    assert (manual["not_proven_saved_count"], manual["omitted_count"]) == (total, 10)
    assert manual["not_proven_saved_bytes"] == sum(range(1, total + 1))
    assert [entry["bytes"] for entry in manual["not_proven_saved"]] == list(range(total, 10, -1))
    [held] = [item for group in owner._plan["groups"] for item in group["excluded"] if item.manual]
    assert len(held.manual["unsaved"]) == UNSAVED_LISTED, "the plan keeps no path it will never show"


def test_a_candidate_edited_after_its_preview_stays_and_its_retained_evidence_is_not_offered(candidate, monkeypatch):  # noqa: F811
    ctx, data, env_root = candidate
    _fake_sizer(monkeypatch)
    repo = Path(ctx.repo_dir)
    (repo / "__pycache__").mkdir()
    pyc = repo / "__pycache__" / "a.pyc"
    pyc.write_bytes(b"x")
    env = (data, data.parent)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    assert _page(owner, env)["total"] == 1
    pyc.write_bytes(b"y")
    assert _discard(owner, env)["started"] and repo.is_dir()
    assert _outcome(owner, env) == ("kept", "changed") and pyc.read_bytes() == b"y"
    (env_root / "tmp").mkdir(parents=True, exist_ok=True)
    assert environment.retain_tree(env_root / "tmp", REASON)
    owner = FreeSpace(live=lambda _task: False, offered=("working_copies",))
    _preview(owner, data, data.parent)
    assert _page(owner, env)["total"] == 0
    assert retire_synthetic_retention(env_root / "tmp", REASON)


# ---- the wire --------------------------------------------------------------------------------------


def test_the_routes_page_the_list_and_require_the_acceptance(copies):  # noqa: F811
    _handle, path, _target = _held(copies)
    owner = _ready(copies)
    client = TestClient(Starlette(routes=storage_routes(copies[0], repo_dir=copies[1], free_owner=owner)))
    plan = owner.projection(copies[0])["plan"]
    page = client.get("/api/storage/free/items", params={"plan": plan, "view": "manual"}).json()
    assert page["view"] == "manual" and page["total"] == 1
    ids = [row["id"] for row in page["items"]]
    refused = client.post("/api/storage/free", json={"plan": plan, "manual": ids})
    assert refused.json()["started"] is False and refused.json()["reason"] == "loss_not_accepted" and path.is_dir()
    accepted = client.post("/api/storage/free", json={"plan": plan, "manual": ids, "accept_loss": True})
    assert accepted.status_code == 202 and accepted.json()["started"] is True
    _join(owner)
    assert not path.exists()
