"""Commits ahead are history; only content of the checkout's own makes an update a caution (#1718).

Real Git with local remotes only (no network): an official repository, a managed
clone, the production merge planner/builder and lander for the updater's own merges,
and the real status reader on the fetching and the passive path. ``present`` names a
delta since the one shared base, never authorship; several bases, a shallow history or
a failed read stay ``unknown``.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops
from supervisor.git_ops_updates import compute_managed_update_status, local_changes_since_shared_base

pytestmark = pytest.mark.serial  # real Git processes against the module-global REPO_DIR


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, message, **files):
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _manage(local, official):
    (local / ".git" / git_ops.MANAGED_REPO_META_NAME).write_text(json.dumps({
        "managed_remote_name": "managed", "managed_remote_branch": "ouroboros",
        "managed_local_branch": "ouroboros", "managed_remote_url": str(official)}), encoding="utf-8")


@pytest.fixture
def managed(tmp_path, monkeypatch):
    for key in ("GIT_AUTHOR", "GIT_COMMITTER"):
        monkeypatch.setenv(f"{key}_NAME", "Update test")
        monkeypatch.setenv(f"{key}_EMAIL", "update@test.invalid")
    official, local = tmp_path / "official", tmp_path / "local"
    official.mkdir()
    _git(official, "init", "-q", "-b", "ouroboros")
    base = _commit(official, "official 1", **{"BIBLE.md": "# Constitution\n", "app.py": "A = 1\nB = 2\n"})
    _git(tmp_path, "clone", "-q", "--origin", "managed", str(official), str(local))
    _manage(local, official)
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "development")
    monkeypatch.setattr(git_ops, "REPO_DIR", local)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    state = {}
    import supervisor.state as supervisor_state

    monkeypatch.setattr(git_ops, "load_state", lambda: state)
    monkeypatch.setattr(supervisor_state, "update_state", lambda fn: fn(state))
    return SimpleNamespace(official=official, local=local, base=base, state=state, tmp=tmp_path)


def _updater_merge(local_branch="ouroboros"):
    """The production path of a clean update: the planner builds the commit, the lander lands it."""
    from supervisor.update_merge import apply_managed_merge_update, plan_managed_update_merge

    plan = plan_managed_update_merge(fetch=True, build=True)
    assert plan["kind"] == "clean" and plan["merge_commit"], plan
    ok, message = apply_managed_merge_update(local_branch, plan["merge_commit"])
    assert ok, message
    return plan


def _assert_published_status(status):
    """Every real Git observation fits the status shape exposed to independent clients."""
    import typing
    from ouroboros.gateway.schema import contract_document, contract_typed_dict, validate_ingress

    route = next(row for row in contract_document()["http"]
                 if row["method"] == "GET" and row["path"] == "/api/update/status")
    assert "UpdateStatusResponse" in route["responses"]["200"]["schemas"]
    shape = contract_typed_dict("UpdateStatusResponse")
    assert {"local_changes", "local_changes_base", "local_changes_reason"} <= set(typing.get_type_hints(shape))
    # The Git reader is not the HTTP envelope: the gateway adds running/source facts.
    from ouroboros.gateway.control import _managed_update_payload

    from unittest.mock import patch

    # Frame this exact observed result, not a second passive read: a consumed
    # target legitimately changes the cache projection even with unchanged HEAD.
    with patch("supervisor.git_ops.compute_managed_update_status", return_value=status):
        payload = _managed_update_payload(fetch=False, include_tags=False)
    for key in ("local_changes", "local_changes_base", "local_changes_reason"):
        assert payload.get(key) == status.get(key)
    assert validate_ingress(payload, shape) == []
    return status


def _check():
    return _assert_published_status(compute_managed_update_status(fetch=True))


def _passive(monkeypatch):
    """The passive read takes no network and writes nothing durable."""
    import supervisor.state as supervisor_state

    monkeypatch.setattr(git_ops, "git_fetch_bounded", lambda *_a, **_k: pytest.fail("passive read fetched"))
    monkeypatch.setattr(supervisor_state, "update_state", lambda fn: pytest.fail("passive read wrote state"))
    return _assert_published_status(compute_managed_update_status(fetch=False))


def test_successive_updater_merges_without_content_stay_an_ordinary_update(managed, monkeypatch):
    _commit(managed.local, "local note", NOTE="x\n")
    _git(managed.local, "revert", "--no-edit", "HEAD")  # a commit and its revert: history only
    o2 = _commit(managed.official, "official 2", **{"app.py": "A = 1\nB = 3\n"})
    status = _check()
    assert (status["ahead"], status["behind"]) == (2, 1)
    assert status["local_changes"] == "absent" and status["safe_to_apply"] is True
    first = _updater_merge()
    assert first["merge_commit"] != o2 and _git(managed.local, "rev-list", "--count", "--merges", "HEAD") == "1"
    o3 = _commit(managed.official, "official 3", extra="3\n")
    status = _check()
    assert (status["ahead"], status["behind"], status["latest_sha"]) == (3, 1, o3)
    assert status["local_changes"] == "absent" and status["local_changes_base"] == o2
    assert status["safe_to_apply"] is True
    _updater_merge()
    o4 = _commit(managed.official, "official 4", extra="4\n")
    status = _check()
    assert (status["ahead"], status["behind"]) == (4, 1) and _git(managed.local, "rev-list", "--count",
                                                                  "--merges", "HEAD") == "2"
    assert status["local_changes"] == "absent" and status["local_changes_base"] == o3
    assert status["safe_to_apply"] is True
    cached = managed.state["managed_update_cache"]
    assert cached["local_changes"] == "absent" and cached["checked_head_sha"] == status["current_sha"]
    passive = _passive(monkeypatch)
    assert passive["from_cache"] is True and passive["latest_sha"] == o4
    assert passive["local_changes"] == "absent" and passive["safe_to_apply"] is True


def test_a_kept_local_edit_and_a_real_resolution_delta_are_present(managed):
    _commit(managed.local, "owner edit", **{"owner.py": "MINE = 1\n"})
    _commit(managed.official, "official edits B", **{"app.py": "A = 1\nB = 3\n"})
    _updater_merge()  # a clean merge that keeps the owner's file
    _commit(managed.official, "official 3", extra="3\n")
    status = _check()
    assert status["local_changes"] == "present" and status["safe_to_apply"] is False
    # A conflict the resolver settles with a line of its own is content too.
    _commit(managed.local, "owner edit 2", **{"app.py": "A = 11\nB = 3\n"})
    o4 = _commit(managed.official, "official edits A", **{"app.py": "A = 20\nB = 3\n"})
    _git(managed.local, "fetch", "-q", "managed")
    subprocess.run(["git", "merge", "--no-edit", o4], cwd=managed.local, capture_output=True)
    (managed.local / "app.py").write_text("A = 30\nB = 3\n", encoding="utf-8")
    _git(managed.local, "commit", "-q", "-am", f"Merge official Ouroboros update {o4[:12]} (resolved)")
    _commit(managed.official, "official 5", extra="5\n")
    assert _check()["local_changes"] == "present"


def test_a_resolution_that_keeps_the_official_side_leaves_no_local_changes(managed):
    _commit(managed.local, "owner edit", **{"app.py": "A = 10\nB = 2\n"})
    o2 = _commit(managed.official, "official edits A", **{"app.py": "A = 20\nB = 2\n"})
    _git(managed.local, "fetch", "-q", "managed")
    subprocess.run(["git", "merge", "--no-edit", o2], cwd=managed.local, capture_output=True)
    _git(managed.local, "checkout", "--theirs", "app.py")
    _git(managed.local, "commit", "-q", "-am", f"Merge official Ouroboros update {o2[:12]} (resolved)")
    _commit(managed.official, "official 3", extra="3\n")
    status = _check()
    assert status["ahead"] == 2 and status["local_changes"] == "absent" and status["safe_to_apply"] is True


def test_an_edit_upstream_also_carries_is_present_until_an_update_merges_it(managed):
    _commit(managed.local, "feature", feature="same\n")
    _commit(managed.official, "feature upstream", feature="same\n")
    status = _check()  # conservative: no patch-equivalence classifier
    assert status["local_changes"] == "present" and status["safe_to_apply"] is False
    _updater_merge()
    _commit(managed.official, "official 3", extra="3\n")
    status = _check()
    assert status["ahead"] == 2 and status["local_changes"] == "absent" and status["safe_to_apply"] is True


def test_a_rewound_target_is_not_an_update_and_keeps_the_delta_honest(managed):
    o2 = _commit(managed.official, "official 2", extra="2\n")
    plan = _updater_merge()
    assert plan["merge_commit"] == o2  # nothing local: a fast-forward, no merge commit
    _git(managed.official, "reset", "-q", "--hard", managed.base)
    status = _check()
    assert (status["ahead"], status["behind"], status["available"]) == (1, 0, False)
    assert status["local_changes"] == "present" and status["local_changes_base"] == managed.base
    assert status["safe_to_apply"] is False


def test_several_shared_bases_are_unknown(managed):
    l1 = _commit(managed.local, "local", lfile="l\n")
    o2 = _commit(managed.official, "official", ofile="o\n")
    _git(managed.official, "fetch", "-q", str(managed.local), "ouroboros:refs/heads/from-local")
    _git(managed.official, "merge", "-q", "--no-edit", "from-local")  # the official line adopts l1
    _git(managed.local, "fetch", "-q", "managed")
    _git(managed.local, "merge", "-q", "--no-edit", o2)  # and the checkout merges o2: criss-cross
    status = _check()
    assert sorted(_git(managed.local, "merge-base", "--all", "HEAD", status["latest_sha"]).split()) == sorted([l1, o2])
    assert status["local_changes"] == "unknown" and status["local_changes_reason"] == "several_bases"
    assert status["safe_to_apply"] is False


def test_a_shallow_history_is_unknown(managed, monkeypatch):
    shallow = managed.tmp / "shallow"
    _git(managed.tmp, "clone", "-q", "--depth", "1", "--origin", "managed",
         f"file://{managed.official}", str(shallow))
    _manage(shallow, managed.official)
    monkeypatch.setattr(git_ops, "REPO_DIR", shallow)
    _commit(shallow, "local", lfile="l\n")
    _commit(managed.official, "official 2", extra="2\n")
    status = _check()
    assert status["behind"] == 1 and status["ahead"] == 1
    assert status["local_changes"] == "unknown" and status["local_changes_reason"] == "shallow_history"
    assert status["safe_to_apply"] is False


def test_missing_objects_are_unknown(managed):
    head = _git(managed.local, "rev-parse", "HEAD")
    fact = local_changes_since_shared_base(head, "0" * 40, ahead=1)
    assert fact["local_changes"] == "unknown" and fact["local_changes_reason"] == "git_error"


def test_uncommitted_work_stays_a_separate_fact(managed):
    _commit(managed.official, "official 2", extra="2\n")
    (managed.local / "app.py").write_text("A = 1\nB = 2\nC = dirty\n", encoding="utf-8")
    status = _check()
    assert status["dirty"] is True and status["dirty_count"] == 1
    assert status["local_changes"] == "absent" and status["safe_to_apply"] is False


@pytest.mark.parametrize("fetch", [True, False], ids=["fresh", "passive"])
def test_counts_and_content_describe_the_head_the_read_captured(managed, monkeypatch, fetch):
    """HEAD moving between the read's capture and its counts never pairs one tree's identity
    with another tree's counts: a divergent captured HEAD is never reported absent."""
    owner = _commit(managed.local, "owner edit", **{"app.py": "A = 7\nB = 2\n"})
    _commit(managed.official, "official 2", extra="2\n")
    if not fetch:
        _check()  # the cache a passive read overlays
    real, moved = git_ops.git_capture, []

    def capture(cmd, **kwargs):
        result = real(cmd, **kwargs)
        if cmd == ["git", "rev-parse", "HEAD"] and not moved:
            moved.append(_git(managed.local, "reset", "-q", "--hard", managed.base))  # HEAD moves under the read
        return result

    monkeypatch.setattr(git_ops, "git_capture", capture)
    status = compute_managed_update_status(fetch=fetch)
    assert moved and _git(managed.local, "rev-parse", "HEAD") == managed.base
    assert status["current_sha"] == owner and (status["ahead"], status["behind"]) == (1, 1)
    assert status["local_changes"] == "present" and status["local_changes_base"] == managed.base
    assert status["safe_to_apply"] is False


def test_head_and_target_moving_after_the_check_never_reuse_the_cached_fact(managed, monkeypatch):
    from ouroboros.update_letter import official_update_projection

    o2 = _commit(managed.official, "official 2", extra="2\n")
    checked = _check()
    assert checked["local_changes"] == "absent" and checked["safe_to_apply"] is True
    drive = managed.tmp / "data"
    (drive / "state").mkdir(parents=True)
    fact = official_update_projection(checked["current_sha"], drive_root=drive, state=managed.state)
    assert fact["local_changes"] == "absent" and fact["ahead"] == 0
    moved = _commit(managed.local, "owner edit after the check", **{"app.py": "A = 5\nB = 2\n"})
    _commit(managed.official, "official 3 (not fetched)", extra="3\n")
    passive = _passive(monkeypatch)
    assert passive["current_sha"] == moved and passive["latest_sha"] == o2  # the checked target identity
    assert (passive["ahead"], passive["behind"]) == (1, 1)
    assert passive["local_changes"] == "present" and passive["local_changes_base"] == managed.base
    assert passive["safe_to_apply"] is False
    # The runtime fact describes only the HEAD the check saw.
    assert official_update_projection(moved, drive_root=drive, state=managed.state)["local_changes"] == "unknown"
