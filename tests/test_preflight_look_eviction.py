"""A preflight look's freshness outlives the bounded hot-index window (D5-002).

The stale marker speaks only against the checkout's newest ``surface=preflight`` look; a
marker no look precedes is the retired advisory lane's archive (owner decision). The look
used to be found among the 50 newest preflight rows of the hot index, filtered by checkout
AFTER that cut: newer looks of other checkouts, or a rotation of the hot index, hid it. A
real marker then vanished from the task evidence and the review context, the first
mutation after the look marked nothing, and an unreadable record read as no look. Real
ledger writes, real rotation, the ordinary mutation path and the three durable readers.
"""

from __future__ import annotations

import json
import shutil
import types

import pytest

from ouroboros import review_ledger
from ouroboros.review_evidence import collect_review_evidence
from ouroboros.review_state import (
    CommitAttemptRecord,
    invalidate_advisory_after_mutation,
    load_state,
    make_repo_key,
    save_state,
    update_state,
)
from ouroboros.review_status_projection import build_review_projection

_HEADS = ("state", "review_ledger", "preflight_heads.json")
_EVICTIONS = ("newer_foreign_looks", "hot_index_rotation")


def _checkout(tmp_path, name):
    repo = tmp_path / name
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    return repo


def _drive(tmp_path):
    drive = tmp_path / "drive"
    (drive / "state").mkdir(parents=True)
    (drive / "logs").mkdir()
    return drive


def _reviewed(drive, repo):
    """A reviewed attempt on the checkout, so its Review Continuity section renders."""
    update_state(drive, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-09-28T00:01:00+00:00", commit_message="reviewed", status="reviewed",
        repo_key=make_repo_key(repo), tool_name="commit_reviewed", task_id="task-history", attempt=1)))


def _look(drive, repo, task_id="task-look"):
    """The author's preflight look: a settled ``surface=preflight`` ledger record, stamped when written."""
    return review_ledger.write_record(drive, {
        "record_id": review_ledger.new_record_id(), "state": "settled", "surface": "preflight",
        "task_id": task_id, "verdict": {"aggregate": "PASS"}, "subject": {"root": str(repo)},
        "rows": []})["record_id"]


def _in_hot_window(drive, record_id):
    """Whether the bounded read the lookup used (50 newest preflight rows, hot index) still shows the record."""
    return record_id in {row["record_id"] for row in review_ledger.recent_records(
        drive, limit=50, hot_only=True, surface="preflight")}


def _evict(eviction, drive, tmp_path, monkeypatch):
    """Newer looks of other checkouts beyond the 50-row window, or the hot index at its rotation size."""
    if eviction == "newer_foreign_looks":
        others = [_checkout(tmp_path, f"other-{n}") for n in range(3)]
        for n in range(60):
            _look(drive, others[n % len(others)], task_id="task-other")
        return
    hot = review_ledger.index_path(drive)
    with monkeypatch.context() as patch:
        patch.setattr(review_ledger, "INDEX_MAX_BYTES", hot.stat().st_size)  # this index is now full
        _look(drive, _checkout(tmp_path, "rotation-writer"), task_id="task-other")
    assert review_ledger.archived_segments_exist(drive)


def _edit(drive, repo, task_id):
    """The ordinary tool path: ``commit_gate._invalidate_advisory`` with the task's ctx."""
    from ouroboros.tools.commit_gate import _invalidate_advisory

    (repo / "tracked.py").write_text(f"x = {task_id!r}\n", encoding="utf-8")
    ctx = types.SimpleNamespace(drive_root=str(drive), repo_dir=str(repo), task_id=task_id,
                                drive_logs=lambda: drive / "logs")
    _invalidate_advisory(ctx, changed_paths=["tracked.py"], source_tool="edit_text")


def _surfaces(drive, repo, reader="task-reader"):
    """The task evidence's checkout panel, the Review Continuity context and the review_status projection."""
    from ouroboros.agent_task_pipeline import build_review_context

    return (collect_review_evidence(drive, task_id=reader, repo_dir=repo)["current_repo"],
            build_review_context(types.SimpleNamespace(drive_root=drive, repo_dir=repo)),
            build_review_projection(drive, repo_dir=repo, reader_task_id=reader))


def _assert_stale(drive, repo, look_id, editor):
    panel, context, projection = _surfaces(drive, repo)
    assert "edit_text mutated the worktree" in panel["stale_reason"] and panel["stale_ts"], panel
    assert (panel["stale_task_id"], panel["stale_repo_key"]) == (editor, make_repo_key(repo))
    assert "stale_marker=" in context and f"invalidated_by=task {editor}" in context, context
    assert (projection["stale_from_edit"], projection["effective_status"]) == (True, "stale")
    assert (projection["preflight"] or {}).get("record_id") == look_id and projection["preflight"]["marked"]


def _assert_fresh(drive, repo, look_id):
    panel, context, projection = _surfaces(drive, repo)
    assert panel["stale_reason"] == panel["stale_ts"] == panel["stale_task_id"] == "", panel
    assert "stale_marker=" not in context, context
    assert (projection["stale_from_edit"], projection["effective_status"]) == (False, "fresh")
    assert (projection["preflight"] or {}).get("record_id") == look_id


@pytest.mark.parametrize("eviction", _EVICTIONS)
def test_the_marker_of_a_real_look_survives_the_look_leaving_the_hot_window(tmp_path, monkeypatch, eviction):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    look_id = _look(drive, repo)
    _edit(drive, repo, "task-a")
    _assert_stale(drive, repo, look_id, "task-a")

    _evict(eviction, drive, tmp_path, monkeypatch)

    assert not _in_hot_window(drive, look_id), "the scenario: the old bounded read no longer shows the look"
    assert review_ledger.load_record(drive, look_id)["surface"] == "preflight", "its source record is kept"
    _assert_stale(drive, repo, look_id, "task-a")
    # A later mutation keeps the first one's attribution: the marker names the edit that staled the look.
    _edit(drive, repo, "task-b")
    _assert_stale(drive, repo, look_id, "task-a")


@pytest.mark.parametrize("eviction", _EVICTIONS)
def test_the_first_mutation_after_the_look_left_the_hot_window_marks_it_stale(tmp_path, monkeypatch, eviction):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    look_id = _look(drive, repo)
    _evict(eviction, drive, tmp_path, monkeypatch)
    assert not _in_hot_window(drive, look_id)
    _assert_fresh(drive, repo, look_id)  # still the checkout's look, and nothing has moved since it

    _edit(drive, repo, "task-a")

    state = load_state(drive)
    assert (state.last_stale_task_id, state.last_stale_repo_key) == ("task-a", make_repo_key(repo))
    _assert_stale(drive, repo, look_id, "task-a")


@pytest.mark.parametrize("eviction", _EVICTIONS)
def test_a_newer_look_retires_the_marker_even_once_it_left_the_hot_window(tmp_path, monkeypatch, eviction):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    _look(drive, repo)
    _edit(drive, repo, "task-a")
    newer = _look(drive, repo)  # the author looks again after the edit
    _assert_fresh(drive, repo, newer)

    _evict(eviction, drive, tmp_path, monkeypatch)

    assert not _in_hot_window(drive, newer)
    _assert_fresh(drive, repo, newer)  # the older marker stays retired: it never resurfaces as current
    assert load_state(drive).last_stale_task_id == "task-a", "the retired marker itself stays in the file"
    _edit(drive, repo, "task-b")  # the first mutation after the newer look stales it, naming its editor
    _assert_stale(drive, repo, newer, "task-b")


def test_a_marker_no_look_precedes_stays_archived_while_looks_come_and_leave(tmp_path, monkeypatch):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    state = load_state(drive)
    state.last_stale_from_edit_ts = "2026-09-20T00:00:00+00:00"
    state.last_stale_reason = "run_command mutated the worktree; advisory freshness invalidated."
    state.last_stale_repo_key = make_repo_key(repo)
    save_state(drive, state)

    for situation in ("no look of this checkout", "a look after the marker, evicted", "rotated away"):
        panel, context, projection = _surfaces(drive, repo)
        assert panel["stale_reason"] == panel["stale_ts"] == panel["stale_task_id"] == "", situation
        assert "stale_marker=" not in context and "advisory freshness" not in context, situation
        assert projection["stale_from_edit"] is False and projection["stale_reason"] is None, situation
        assert load_state(drive).last_stale_from_edit_ts == "2026-09-20T00:00:00+00:00", "history stays in the file"
        if situation == "no look of this checkout":
            _evict("newer_foreign_looks", drive, tmp_path, monkeypatch)
            _look(drive, repo)
            _evict("newer_foreign_looks", drive, tmp_path, monkeypatch)
        else:
            _evict("hot_index_rotation", drive, tmp_path, monkeypatch)


def test_per_checkout_scope_holds_after_eviction(tmp_path, monkeypatch):
    drive, one, two = _drive(tmp_path), _checkout(tmp_path, "one"), _checkout(tmp_path, "two")
    _reviewed(drive, one)
    _reviewed(drive, two)
    look_one, look_two = _look(drive, one), _look(drive, two)
    _edit(drive, one, "task-a")
    _evict("newer_foreign_looks", drive, tmp_path, monkeypatch)

    _assert_stale(drive, one, look_one, "task-a")
    _assert_fresh(drive, two, look_two)  # the marker scoped to one checkout never stales another

    # A mutation of no single checkout stales every look before it — but not a look taken after it.
    update_state(drive, lambda state: state.on_successful_commit(repo_key=make_repo_key(one)))
    invalidate_advisory_after_mutation(drive, mutation_root=None, changed_paths=[], source_tool="run_shell",
                                       mutating_task_id="task-c")
    look_two_again = _look(drive, two)
    _evict("hot_index_rotation", drive, tmp_path, monkeypatch)

    panel, context, projection = _surfaces(drive, one)
    assert "run_shell mutated the worktree" in panel["stale_reason"] and panel["stale_repo_key"] == ""
    assert "applies to every checkout" in context and projection["preflight"]["record_id"] == look_one
    assert projection["stale_from_edit"] is True
    _assert_fresh(drive, two, look_two_again)


@pytest.mark.parametrize("unreadable", ["look_record", "checkout_heads"])
def test_an_unreadable_look_is_named_and_never_read_as_no_look(tmp_path, unreadable):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    look_id = _look(drive, repo)
    _edit(drive, repo, "task-a")
    target = review_ledger.record_path(drive, look_id) if unreadable == "look_record" else drive.joinpath(*_HEADS)
    original = target.read_bytes()
    target.write_text("{not json", encoding="utf-8")

    panel, context, projection = _surfaces(drive, repo)

    # A read error hides nothing: the marker still shows, and the gap is named beside it.
    assert "edit_text mutated the worktree" in panel["stale_reason"] and panel["stale_task_id"] == "task-a"
    assert "ValueError" in panel["preflight_unreadable"]
    assert "stale_marker=" in context and "preflight_unreadable=ValueError" in context, context
    assert (projection["stale_from_edit"], projection["effective_status"]) == (True, "unknown")
    assert "ValueError" in projection["preflight"]["unreadable"] and projection["stale_task_id"] == "task-a"

    # A mutation while the look cannot be read is still recorded, never dropped as "no look" ...
    update_state(drive, lambda state: state.on_successful_commit(repo_key=make_repo_key(repo)))
    _edit(drive, repo, "task-b")
    assert load_state(drive).last_stale_task_id == "task-b"
    # ... and is measured against the look once the look reads again.
    target.write_bytes(original)
    _assert_stale(drive, repo, look_id, "task-b")


def test_looks_from_before_the_heads_are_seeded_once_and_heads_follow_live_checkouts(tmp_path, monkeypatch):
    drive, repo, gone = _drive(tmp_path), _checkout(tmp_path, "repo"), _checkout(tmp_path, "gone")
    gone_key = make_repo_key(gone)
    _reviewed(drive, repo)
    look_id = _look(drive, repo)
    _look(drive, gone)
    _edit(drive, repo, "task-a")
    heads = drive.joinpath(*_HEADS)
    heads.unlink(missing_ok=True)  # a ledger written before checkout heads were kept
    _assert_stale(drive, repo, look_id, "task-a")  # read as before: the bounded hot-index window
    shutil.rmtree(gone)

    # The first ledger write seeds the heads from that window before its rotation archives the window.
    _evict("hot_index_rotation", drive, tmp_path, monkeypatch)
    _assert_stale(drive, repo, look_id, "task-a")
    _evict("newer_foreign_looks", drive, tmp_path, monkeypatch)
    _assert_stale(drive, repo, look_id, "task-a")

    kept = json.loads(heads.read_text(encoding="utf-8"))["heads"]
    assert kept[make_repo_key(repo)] == look_id and gone_key not in kept, "a removed checkout drops out"
    assert set(kept) == {"", *(make_repo_key(tmp_path / name) for name in
                               ("repo", "rotation-writer", "other-0", "other-1", "other-2"))}


@pytest.mark.parametrize("eviction", _EVICTIONS)
@pytest.mark.parametrize("heads_state", ["missing", "corrupt"])
def test_incomplete_head_import_never_erases_an_archived_modern_marker(tmp_path, monkeypatch, eviction, heads_state):
    """Upgrade/lost-heads paths cannot certify absence from a bounded miss."""
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    look_id = _look(drive, repo)
    _edit(drive, repo, "task-a")
    _evict(eviction, drive, tmp_path, monkeypatch)
    heads = drive.joinpath(*_HEADS)
    if heads_state == "missing":
        heads.unlink()
    else:
        heads.write_text("{broken", encoding="utf-8")
    assert not _in_hot_window(drive, look_id)
    assert review_ledger.load_record(drive, look_id)["surface"] == "preflight"

    def unknown_surfaces():
        panel, context, projection = _surfaces(drive, repo)
        assert panel["stale_reason"] and panel["preflight_unreadable"]
        assert "stale_marker=" in context and "marker unverified" in context
        assert projection["effective_status"] == "unknown" and projection["stale_from_edit"]

    unknown_surfaces()
    _look(drive, _checkout(tmp_path, "new-writer"))
    unknown_surfaces()  # the next write cannot launder an incomplete import into absence
    if heads_state == "corrupt":
        assert heads.read_text(encoding="utf-8") == "{broken", "do not overwrite corrupt history"
    else:
        assert json.loads(heads.read_text())["complete"] is False
        newer = _look(drive, repo)
        _assert_fresh(drive, repo, newer)  # an actual new look resolves this checkout


@pytest.mark.parametrize("eviction", _EVICTIONS)
def test_first_edit_after_incomplete_import_is_recorded_with_a_gap(tmp_path, monkeypatch, eviction):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    _reviewed(drive, repo)
    _look(drive, repo)
    _evict(eviction, drive, tmp_path, monkeypatch)
    drive.joinpath(*_HEADS).unlink()
    _look(drive, _checkout(tmp_path, "new-writer"))
    _edit(drive, repo, "task-a")
    panel, context, projection = _surfaces(drive, repo)
    assert panel["stale_task_id"] == "task-a" and panel["preflight_unreadable"]
    assert "marker unverified" in context and projection["effective_status"] == "unknown"


def test_current_head_read_does_not_replay_the_index(tmp_path, monkeypatch):
    drive, repo = _drive(tmp_path), _checkout(tmp_path, "repo")
    look_id = _look(drive, repo)
    def no_replay(*args, **kwargs):
        raise AssertionError("current checkout lookup must not replay an index")
    monkeypatch.setattr(review_ledger, "recent_records", no_replay)
    monkeypatch.setattr(review_ledger, "archived_segments_exist", no_replay)
    assert review_ledger.latest_preflight_record(drive, repo_key=make_repo_key(repo))["record_id"] == look_id
