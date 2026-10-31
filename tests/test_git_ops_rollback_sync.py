"""Restore changes only the local checkout, even with a divergent personal remote."""

import pytest

from tests.test_git_lock_owners import _git, _rows, restore_door  # noqa: F401


@pytest.mark.parametrize("door", ["ui_rollback", "vcs_rollback"])
def test_restore_lands_and_preserves_origin_and_rescue(restore_door, tmp_path, door):  # noqa: F811
    scene = restore_door
    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(scene.repo, "remote", "add", "origin", str(remote))
    _git(scene.repo, "push", "-u", "origin", "ouroboros")
    before = _git(remote, "rev-parse", "refs/heads/ouroboros")
    (scene.repo / "seed.txt").write_text("uncommitted owner work\n", encoding="utf-8")

    status, body = scene.restore(scene.old, door)

    assert status == 200 and body["restarting"] is True
    assert _git(scene.repo, "rev-parse", "HEAD") == scene.old
    assert _git(remote, "rev-parse", "refs/heads/ouroboros") == before == scene.head
    assert "Nothing was pushed" in body["message"]
    [row] = _rows(scene.data, "manual_rollback")
    assert row["remote_pushed"] is False and row["new_sha"] == scene.old
    assert _git(scene.repo, "rev-parse", row["keep_branch"]) == scene.head
    refs = _git(scene.repo, "for-each-ref", "--format=%(refname)", "refs/rescue").splitlines()
    assert len(refs) == 1
    assert _git(scene.repo, "show", refs[0] + ":seed.txt") == "uncommitted owner work"
