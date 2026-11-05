"""S6 C3/C4 — a private copy's record when it cannot be read or written.

Each private copy has ONE durable record file, ``state/private_copies/<name>.json``,
the only thing that names the copy and the ``refs/ouroboros/delegated/*`` pins it
holds. An absent record store is an ordinary empty one. An unreadable record is
skipped by every listing, its bytes are kept, one typed
``private_copy_record_unreadable`` event discloses it, and the direct read of THAT
record refuses: no pass rewrites or deletes it, so the checkout and its pin stay
where they are, and every other copy works as before. (The whole-file registry
this replaces refused every mutation on the machine over one malformed file.)

C4 pins the write half: a failed record write removes the checkout AND the
baseline ref, the symmetry the payload sibling already had
(``tests/test_delegated_skill_payload.py::test_registry_save_failure_leaves_no_orphan_snapshot_dir``).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros import subagent_worktrees as wt
from ouroboros.unindexed import copy_path


MALFORMED = '"not a record"'


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check,
    )


def _seed_target(tmp_path: pathlib.Path) -> pathlib.Path:
    target = tmp_path / "target"
    target.mkdir()
    _git(target, "init")
    (target / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(target, "add", "-A")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    return target


def _break(data, path, payload=MALFORMED) -> pathlib.Path:
    record = wt._record_file(path, data)
    record.write_text(payload, encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# C3 — absent is empty, unreadable is skipped, kept and refused for its own copy
# ---------------------------------------------------------------------------
# C4 — a record write that fails
# ---------------------------------------------------------------------------


def _failing_put(monkeypatch, failing_call):
    real_put, calls = wt._put_record, []

    def put(row, data_dir=None):
        calls.append(row["path"])
        if len(calls) == failing_call:
            raise OSError("record disk full")
        return real_put(row, data_dir)

    monkeypatch.setattr(wt, "_put_record", put)
    return calls


def test_c4_b_the_acting_worktree_cleans_up_on_record_failure(tmp_path, monkeypatch):
    """``provision_worktree`` creates a checkout AND a task branch after its record.
    An unreadable record of that copy refuses before either exists, and a failed write
    after them removes both — otherwise every retry strands one more worktree+branch."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    copy = copy_path(snaps, "acting9")

    # Leg 1: an unreadable record of this copy refuses first; its bytes are kept.
    (data / "state" / "private_copies").mkdir(parents=True)
    snaps.mkdir()
    record = _break(data, copy.resolve())
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.provision_worktree(
            repo_dir=target, task_id="acting9", worktree_root=snaps, data_dir=data)
    assert not copy.exists(), "the refused attempt made nothing"
    assert _git(
        target, "rev-parse", "--verify", f"{wt._BRANCH_PREFIX}acting9", check=False,
    ).returncode != 0, "and no task branch"
    assert record.read_text(encoding="utf-8") == MALFORMED

    # Leg 2: the final record write fails after the branch and checkout exist.
    record.unlink()
    _failing_put(monkeypatch, 2)
    with pytest.raises(OSError, match="record disk full"):
        wt.provision_worktree(
            repo_dir=target, task_id="acting9", worktree_root=snaps, data_dir=data)
    assert not copy.exists()
    assert _git(
        target, "rev-parse", "--verify", f"{wt._BRANCH_PREFIX}acting9", check=False,
    ).returncode != 0
    assert _git(target, "worktree", "list").stdout.count("acting9") == 0
    assert not record.exists()


# ---------------------------------------------------------------------------
# Disclosure — the record store's own shape
# ---------------------------------------------------------------------------
