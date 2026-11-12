"""New generated copies are born where desktop search skips them (``ouroboros/unindexed.py``).

Spotlight skips a folder whose name ends in ``.noindex`` with everything under it, so a new
review checkout, seat area or acting checkout gets such a name. A copy that already exists keeps
its spelling: a rejoin reuses it, an engine thread stays bound to its path (also when a refresh
died after removing it) and the sweeps still retire it. Owner folders (genesis projects) are
never named this way. Real Git, real copies.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate, review_checking as rc, review_checking_prune as rp
from ouroboros import subagent_worktrees as trees
from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.tools.review_subject import ReviewSubjectSpec, checkout_token, isolated_checkout, round_checkout_repo
from ouroboros.unindexed import BYTECODE, COPIES, SUFFIX, bytecode_prefix, copy_path, unindexed
from tests.body_candidate_support import isolate, make_ctx, make_serving
from tests.test_review_checking_prune import _custody
from tests.test_review_checkout_lifecycle import EDIT, TASK, _frozen, _seat, _thread, _work
from tests.test_review_checkout_retirement import RETRY_KEY, _git, _prune, _repo, _retain
from tests.test_review_checkout_retirement import TASK as REVIEWER  # the task `_retain` finishes


def test_a_new_entry_is_born_unindexed_and_an_existing_one_keeps_its_name(tmp_path):
    assert SUFFIX == ".noindex" and COPIES == "copies.noindex"
    assert unindexed(tmp_path, "area") == tmp_path / "area.noindex"
    (tmp_path / "area").mkdir()
    assert unindexed(tmp_path, "area") == tmp_path / "area"
    (tmp_path / "repo").write_text("not a folder\n", encoding="utf-8")  # any entry holds the old name
    assert unindexed(tmp_path, "repo") == tmp_path / "repo"
    assert unindexed(tmp_path, "seat", tmp_path / "seat") == tmp_path / "seat"  # a recorded old name, while absent
    assert unindexed(tmp_path, "seat", tmp_path / "seat.noindex") == tmp_path / "seat.noindex"
    assert copy_path(tmp_path, "dlg_x") == tmp_path / COPIES / "dlg_x"
    (tmp_path / "dlg_x").mkdir()
    assert copy_path(tmp_path, "dlg_x") == tmp_path / "dlg_x"



def test_the_bytecode_prefix_is_born_unindexed_and_takes_over_the_legacy_tree_once(tmp_path):
    """The prefix mirrors the app's own path (``…/Ouroboros.app/…``), which Spotlight lists as an app."""
    data, warm = tmp_path / "data", pathlib.Path("Applications", "Ouroboros.app", "m.cpython-310.pyc")
    state = data / "state"
    (state / "pycache" / warm).parent.mkdir(parents=True)
    (state / "pycache" / warm).write_bytes(b"warm")
    assert BYTECODE == "pycache" + SUFFIX and bytecode_prefix(data) == str(state / BYTECODE)
    assert not (state / "pycache").exists() and (state / BYTECODE / warm).read_bytes() == b"warm", "renamed, still warm"
    (state / "pycache" / warm).parent.mkdir(parents=True)  # an older body wrote the legacy tree again
    (state / "pycache" / warm).write_bytes(b"again")
    assert bytecode_prefix(data) == str(state / BYTECODE)
    assert (state / BYTECODE / warm).read_bytes() == b"warm", "the new tree is never replaced; the pruner retires the old"
    assert bytecode_prefix(data, str(state / "pycache")) == str(state / BYTECODE), "an old launcher's default is replaced"
    assert bytecode_prefix(data, str(tmp_path / "mine")) == str(tmp_path / "mine"), "any other prefix is honoured"


def test_the_launcher_takes_over_an_inherited_legacy_prefix_before_anything_runs(tmp_path):
    data = tmp_path / "data"
    legacy = data / "state" / "pycache"
    legacy.mkdir(parents=True)
    (legacy / "warm.pyc").write_bytes(b"warm")
    repo = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(data), "PYTHONPYCACHEPREFIX": str(legacy), "PYTHONPATH": str(repo)}
    done = subprocess.run([sys.executable, "-B", "-c", "import os, launcher; print(os.environ['PYTHONPYCACHEPREFIX'])"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == str(data / "state" / BYTECODE), "the old setdefault kept the legacy one"
    assert not legacy.exists() and (data / "state" / BYTECODE / "warm.pyc").read_bytes() == b"warm"

@pytest.fixture
def review(tmp_path):
    """A prepared data root, a repository with a proposal and a reviewer task (as in retirement tests)."""
    data = tmp_path / "data"
    prepare_startup_state(data)
    repo, system = _repo(tmp_path / "subject"), _repo(tmp_path / "system")
    ctx = SimpleNamespace(repo_dir=str(system), drive_root=str(data), task_id=REVIEWER, task_metadata=None)
    spec = ReviewSubjectSpec(root_kind="active_workspace", root=str(repo), kind="base..head", base="HEAD~1",
                             head="HEAD", governance_root=str(system), surface="change")
    return SimpleNamespace(data=data.resolve(), repo=repo, ctx=ctx, spec=spec,
                           name=checkout_token(RETRY_KEY, task_id=REVIEWER))


def _same(left, right) -> bool:
    return pathlib.Path(left).resolve() == pathlib.Path(right).resolve()


def test_a_new_review_checkout_is_born_unindexed_and_retired(review):
    entry = _retain(review)
    assert (entry / "repo.noindex").is_dir() and not (entry / "repo").exists()
    assert _same(round_checkout_repo(review.ctx, RETRY_KEY, REVIEWER), entry / "repo.noindex")
    report = _prune(review)
    assert not entry.exists() and report["checkouts_removed"] == 1, report
    assert len(_git(review.repo, "worktree", "list").splitlines()) == 1


def test_a_review_checkout_an_older_body_made_is_reused_in_place_and_retired(review):
    entry = _retain(review)
    legacy = entry / "repo"
    _git(review.repo, "worktree", "move", str(entry / "repo.noindex"), str(legacy))
    with isolated_checkout(review.ctx, review.spec, retain=lambda: {"seats": ["late seat"]},
                           token=lambda _identity: review.name) as again:
        assert _same(again.checkout, legacy)
    assert legacy.is_dir() and not (entry / "repo.noindex").exists()
    assert _same(round_checkout_repo(review.ctx, RETRY_KEY, REVIEWER), legacy)
    report = _prune(review)
    assert not entry.exists() and report["checkouts_removed"] == 1, report
    assert len(_git(review.repo, "worktree", "list").splitlines()) == 1


def test_a_new_seat_copy_is_born_unindexed_and_an_existing_area_is_reused(tmp_path):
    root, source = _frozen(tmp_path)
    fresh = _seat(root, source, "op-new")
    assert fresh == rc.seat_dir(root, TASK, "op-new") / "area.noindex" and (fresh / "check.py").read_bytes() == EDIT
    older = _seat(root, source, "op-old")
    legacy = older.with_name("area")
    older.rename(legacy)  # the copy an older body made for this operation
    assert rc.prepare_seat_area(root, TASK, "op-old", source) == str(legacy)
    assert (legacy / "check.py").read_bytes() == EDIT and not older.exists()


@pytest.mark.parametrize("older_body", [False, True])
def test_a_thread_copy_keeps_one_path_across_turns(tmp_path, older_body):
    """The engine thread is bound to the copy's path, so a refresh never renames it."""
    root, source = _frozen(tmp_path)
    first = _thread(root, source, "turn-1")
    assert first.name == "area.noindex"
    if older_body:
        first = first.rename(first.with_name("area"))
    _work(first)
    assert _thread(root, source, "turn-2") == first
    assert first.is_dir() and not first.with_name("area.noindex" if older_body else "area").exists()


class _Interrupted(BaseException):
    """A process that dies between removing a thread's old copy and renaming the new one into place."""


@pytest.mark.parametrize("older_body", [False, True])
def test_an_interrupted_refresh_keeps_the_path_the_engine_thread_is_bound_to(tmp_path, monkeypatch, older_body):
    """A refresh that dies after removing the old copy is rebuilt at the recorded path, under either name."""
    root, source = _frozen(tmp_path)
    bound = _thread(root, source, "turn-1")
    marker = bound.parent / "checking.json"
    if older_body:  # an older body made and recorded the copy under the old name
        bound = bound.rename(bound.with_name("area"))
        marker.write_text(json.dumps({**json.loads(marker.read_text(encoding="utf-8")), "area": str(bound)}),
                          encoding="utf-8")
    rc.bind_checking_thread(root, TASK, "plan_review:critic", "engine-thread")
    remove = rc._remove_checking_area

    def dies_after_removing(path, *recorded):
        remove(path, *recorded)
        if path == bound:
            raise _Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(rc, "_remove_checking_area", dies_after_removing)
        with pytest.raises(_Interrupted):
            _thread(root, source, "turn-2")
    assert not bound.exists()
    assert _thread(root, source, "turn-2") == bound
    assert json.loads(marker.read_text(encoding="utf-8"))["thread_id"] == "engine-thread"
    assert not bound.with_name("area.noindex" if older_body else "area").exists()


@pytest.mark.parametrize("older_body", [False, True])
def test_the_sweep_releases_a_finished_seat_under_either_name(tmp_path, monkeypatch, older_body):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-done")
    if older_body:
        area = area.rename(area.with_name("area"))
    _custody(monkeypatch, settled=[area])
    report = rp.prune_review_checkouts(root, set())
    assert not area.exists() and (report["released"], report["errors"]) == (1, []), report
    receipt = rc._receipt(root, TASK, "op-done")
    assert (pathlib.Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


def _source(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "t@example.com"), ("config", "user.name", "t")):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    (path / "a.txt").write_bytes(b"base\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True, capture_output=True)
    return path


def test_a_new_acting_checkout_lives_in_the_unindexed_folder_and_an_existing_path_stays(tmp_path, monkeypatch):
    source, data, root = _source(tmp_path / "project"), tmp_path / "data", tmp_path / "copies"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(root))
    fresh = trees.provision_execution_snapshot(target_root=source, task_id="child", snapshot_id="leaf",
                                               worktree_root=root, data_dir=data)
    assert _same(pathlib.Path(fresh.path).parent, root / COPIES)
    assert (pathlib.Path(fresh.path) / "a.txt").read_bytes() == b"base\n"
    stale = root / "dlg_child_again"  # an earlier attempt of this snapshot id, made before the folder
    stale.mkdir()
    (stale / "left.txt").write_text("crash leftover\n", encoding="utf-8")
    again = trees.provision_execution_snapshot(target_root=source, task_id="child", snapshot_id="again",
                                               worktree_root=root, data_dir=data)
    assert _same(again.path, stale) and not (stale / "left.txt").exists()
    assert (stale / "a.txt").read_bytes() == b"base\n" and not (root / COPIES / "dlg_child_again").exists()


def test_a_body_candidate_and_its_environment_share_the_unindexed_folder(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    ctx = make_ctx(serving, data, "root-copy")
    candidate = pathlib.Path(body_candidate.prepare(ctx)["path"])
    assert _same(candidate.parent, tmp_path / "worktrees" / COPIES) and candidate.name == "body_root-copy"
    env_root = pathlib.Path(body_candidate.process_environment(ctx, ctx.repo_dir)["HOME"]).parent
    assert env_root.is_dir() and env_root.name == "body_root-copy.env" and _same(env_root.parent, candidate.parent)


def test_owner_projects_are_never_born_unindexed(tmp_path, monkeypatch):
    system, data, projects = _source(tmp_path / "system"), tmp_path / "data", tmp_path / "projects"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    project = trees.provision_genesis_project(repo_dir=system, task_id="genesis", projects_root=projects,
                                              data_dir=data)
    # The name it is born with: the test's own root may itself be unindexed (scripts/safe_test.py).
    assert _same(pathlib.Path(project.path).parent, projects) and SUFFIX not in pathlib.Path(project.path).name
