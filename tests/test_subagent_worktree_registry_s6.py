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

import json
import pathlib
import re
import subprocess

import pytest

from ouroboros import body_candidate
from ouroboros import subagent_worktrees as wt
from ouroboros.task_results import write_task_result
from ouroboros.unindexed import COPIES, copy_path


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


def _events(data_dir: pathlib.Path):
    path = data_dir / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unreadable_events(data_dir: pathlib.Path):
    return [row for row in _events(data_dir) if row.get("type") == "private_copy_record_unreadable"]


def _snapshot(tmp_path, snapshot_id="snapS6"):
    """One recorded delegated execution snapshot; returns (target, snaps, data, handle)."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    handle = wt.provision_execution_snapshot(
        target_root=target, task_id="t1", snapshot_id=snapshot_id,
        worktree_root=snaps, data_dir=data)
    return target, snaps, data, handle


def _break(data, path, payload=MALFORMED) -> pathlib.Path:
    record = wt._record_file(path, data)
    record.write_text(payload, encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# C3 — absent is empty, unreadable is skipped, kept and refused for its own copy
# ---------------------------------------------------------------------------


def test_c3_absent_is_an_ordinary_empty_store(tmp_path):
    """A never-written store is the first-write case, never a refusal."""
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)

    assert wt._records(data) == [] and wt._records(data, strict=True) == []
    assert wt.list_worktrees(data_dir=data) == []
    assert wt.find_execution_snapshot("nothing", data_dir=data) is None
    assert wt._record_at(tmp_path / "copies" / "nothing", data) is None
    assert _unreadable_events(data) == []


@pytest.mark.parametrize(
    "payload", [MALFORMED, '{"path": ""}', '{"path": [', "\x00\x01"],
)
def test_c3_every_unreadable_record_shape_is_skipped_kept_and_refused_by_its_own_read(tmp_path, payload):
    data = tmp_path / "data"
    good, bad = tmp_path / "copies" / "good", tmp_path / "copies" / "bad"
    wt._put_record({"path": str(good), "task_id": "good"}, data)
    (data / "state" / "private_copies").mkdir(parents=True, exist_ok=True)
    record = _break(data, bad, payload)

    assert [row["task_id"] for row in wt._records(data)] == ["good"], "the listing skips it"
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt._records(data, strict=True)  # a strict listing (placement, body candidates) refuses instead
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt._record_at(bad, data)
    assert wt._record_at(good, data)["task_id"] == "good"
    assert record.read_text(encoding="utf-8") == payload, "bytes are kept"
    assert [row["record"] for row in _unreadable_events(data)] == [str(record)], "disclosed once, durably"


def test_c3_a_live_snapshot_with_an_unreadable_record_keeps_its_checkout_and_pin(tmp_path):
    """No read may answer "this copy is gone, its pin may go" from a record it could
    not read: the checkout and its pinned baseline ref stay exactly where they are."""
    target, snaps, data, handle = _snapshot(tmp_path)
    assert wt.find_execution_snapshot("snapS6", data_dir=data) is not None
    record = _break(data, handle.path)

    assert wt.find_execution_snapshot("snapS6", data_dir=data) is None
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt._record_at(handle.path, data)
    assert wt.remove_execution_snapshot("snapS6", worktree_root=snaps, data_dir=data) is False
    assert pathlib.Path(handle.path).is_dir()
    assert _git(target, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha
    assert record.read_text(encoding="utf-8") == MALFORMED


def test_c3_gc_passes_never_touch_a_copy_whose_record_is_unreadable(tmp_path, monkeypatch):
    """A destructive GC over an unknowable copy is exactly what the custody prune already
    fails closed on; a healthy copy beside it is still reclaimed."""
    target, snaps, data, kept = _snapshot(tmp_path, snapshot_id="kept")
    gone = wt.provision_execution_snapshot(target_root=target, task_id="t1", snapshot_id="gone",
                                           worktree_root=snaps, data_dir=data)
    acting = wt.provision_worktree(repo_dir=target, task_id="acting1", worktree_root=snaps, data_dir=data)
    record = _break(data, kept.path)
    acting_record = _break(data, acting.path, '{"path": [')
    write_task_result(data, "t1", "completed")
    write_task_result(data, "acting1", "completed")
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_args: [])

    assert wt.prune_execution_snapshots(set(), worktree_root=snaps, data_dir=data,
                                        live=lambda _task: False) == {"removed": ["gone"], "kept": []}
    assert wt.prune_orphans(worktree_root=snaps, data_dir=data, retention_days=0,
                            live=lambda _task: False) == {"removed": 0, "kept": 0}

    assert not pathlib.Path(gone.path).exists(), "the healthy copy went"
    assert pathlib.Path(kept.path).is_dir() and pathlib.Path(acting.path).is_dir(), "nothing else was removed"
    assert _git(target, "rev-parse", kept.baseline_ref, check=False).returncode == 0
    assert _git(target, "rev-parse", "--verify", acting.branch, check=False).returncode == 0
    assert record.read_text(encoding="utf-8") == MALFORMED, "recovery material"
    assert acting_record.read_text(encoding="utf-8") == '{"path": ['


def test_c3_a_new_snapshot_is_not_blocked_while_the_same_copy_refuses(tmp_path):
    """One unreadable record blocks only its own copy: re-provisioning that copy refuses
    before anything moves, and a different snapshot provisions as usual."""
    target, snaps, data, first = _snapshot(tmp_path, snapshot_id="snapOne")
    record = _break(data, first.path)

    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapOne",
            worktree_root=snaps, data_dir=data)
    assert record.read_text(encoding="utf-8") == MALFORMED
    assert pathlib.Path(first.path).is_dir(), "the recorded snapshot is untouched"
    assert _git(target, "rev-parse", first.baseline_ref).stdout.strip() == first.baseline_sha

    second = wt.provision_execution_snapshot(
        target_root=target, task_id="t1", snapshot_id="snapTwo",
        worktree_root=snaps, data_dir=data)
    assert pathlib.Path(second.path).is_dir()
    assert wt.find_execution_snapshot("snapTwo", data_dir=data)["path"] == second.path


def test_c3_the_healthy_record_paths_are_unchanged(tmp_path):
    """The fix must not cost the ordinary lifecycle anything."""
    target, snaps, data, handle = _snapshot(tmp_path)

    assert wt.find_execution_snapshot("snapS6", data_dir=data)["path"] == handle.path
    assert wt.prune_execution_snapshots(
        {"snapS6"}, worktree_root=snaps, data_dir=data,
    ) == {"removed": [], "kept": ["snapS6"]}
    assert wt.prune_orphans(worktree_root=snaps, data_dir=data) == {"removed": 0, "kept": 1}
    assert wt.remove_execution_snapshot("snapS6", worktree_root=snaps, data_dir=data) is True
    assert wt.list_worktrees(data_dir=data) == []
    assert _unreadable_events(data) == []


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


@pytest.mark.parametrize("failing_call", [1, 2], ids=["provisional", "final"])
def test_c4_a_record_write_failure_leaves_no_worktree_or_ref(tmp_path, monkeypatch, failing_call):
    """A failed record write removes the checkout and deletes the baseline ref it
    pinned, instead of leaving a snapshot nothing can name plus a ref that holds
    its commit against git's own GC."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    calls = _failing_put(monkeypatch, failing_call)

    with pytest.raises(OSError, match="record disk full"):
        wt.provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapFail",
            worktree_root=snaps, data_dir=data)

    assert len(calls) == failing_call
    leftovers = sorted(p.name for p in [*snaps.glob("dlg_*"), *snaps.glob(f"{COPIES}/dlg_*")])
    assert leftovers == [], leftovers
    assert _git(
        target, "rev-parse", "refs/ouroboros/delegated/snapFail", check=False,
    ).returncode != 0, "the baseline ref is gone with the checkout"
    assert _git(target, "worktree", "list").stdout.count("dlg_t1_snapFail") == 0
    assert wt.find_execution_snapshot("snapFail", data_dir=data) is None


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


def test_each_copy_is_one_record_file_told_apart_by_kind(tmp_path):
    """Disclosure: one JSON object per copy, named after the copy's directory plus a
    digest of its full path, with no version field; the two shapes are told apart only
    by ``kind == "delegated_exec"`` (an acting copy's record carries no kind at all)."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    acting = wt.provision_worktree(
        repo_dir=target, task_id="acting1", worktree_root=snaps, data_dir=data)
    snapshot = wt.provision_execution_snapshot(
        target_root=target, task_id="t1", snapshot_id="snapKind",
        worktree_root=snaps, data_dir=data)

    files = sorted((data / "state" / "private_copies").glob("*.json"))
    assert [re.sub(r"-[0-9a-f]{12}\.json$", "", f.name) for f in files] == ["acting1", "dlg_t1_snapKind"]
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    assert [row["path"] for row in rows] == [acting.path, snapshot.path]
    assert [row.get("kind", "") for row in rows] == ["", "delegated_exec"]
    assert not any("schema_version" in row for row in rows), "no version field to dispatch on"
