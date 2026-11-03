"""``headless.drive_candidates``, the selection the drive prune and the free-space preview share,
and the prune's ``only``/``progress`` knobs the free action uses."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ouroboros import headless
from ouroboros.retention import age_cutoff
from tests.test_store import document, publish

NOW = datetime(2026, 10, 16, tzinfo=timezone.utc).timestamp()
OLD = "2026-10-01T00:00:00+00:00"
YOUNG = "2026-10-15T12:00:00+00:00"


def _layout(root, kind, task_id):
    if kind == "headless":
        drive = headless.prepare_task_drive(root, task_id, "empty")
        return drive.parent, headless.prune_headless_task_drives
    drive = root / "task_drives" / task_id
    drive.mkdir(parents=True)
    return drive, headless.prune_task_drives


def _world(root, kind, extra=()):
    rows = []
    for task_id, status, date in (("old-done", "completed", OLD), ("young-done", "failed", YOUNG),
                                  ("running", "running", OLD), *extra):
        directory, _prune = _layout(root, kind, task_id)
        rows.append(document(task_id, status, child_drive_root=str(directory if kind != "headless" else directory / "data"), ts=date))
    _layout(root, kind, "rowless")
    publish(root, *rows)
    return root / (headless.HEADLESS_TASKS_DIR if kind == "headless" else headless.TASK_DRIVES_DIR)


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_candidates_carry_the_cheap_facts_in_name_order_and_say_who_is_inside_the_wait(tmp_path, kind):
    root = tmp_path / "data"
    base = _world(root, kind)
    rows = list(headless.drive_candidates(base, root, age_cutoff(7, NOW)))
    assert [(task_id, status, inside) for task_id, status, _at, inside, _fact in rows] == [
        ("old-done", "completed", False), ("rowless", "", False), ("running", "running", False), ("young-done", "failed", True),
    ]
    assert rows[0][2] is not None and rows[1][2] is None and rows[1][4] == {}
    zero = list(headless.drive_candidates(base, root, age_cutoff(0, NOW)))
    assert [inside for _t, _s, _a, inside, _f in zero] == [False, False, False, False], "retention 0: nothing is inside the wait"
    wrapped = [task_id for task_id, *_rest in headless.drive_candidates(base, root, age_cutoff(7, NOW), after="rowless")]
    assert wrapped == ["running", "young-done", "old-done", "rowless"]


@pytest.mark.parametrize("kind", ["headless", "direct"])
def test_only_settles_exactly_the_named_folders_and_progress_sees_each_attempt(tmp_path, kind):
    root = tmp_path / "data"
    base = _world(root, kind, extra=(("other-done", "completed", OLD),))
    prune = headless.prune_headless_task_drives if kind == "headless" else headless.prune_task_drives
    seen = []
    report = prune(root, retention_days=0, now=NOW, live=lambda _task: False, only=["old-done", "young-done"],
                   progress=lambda current: seen.append(len(current["pruned"])))
    assert {row["task_id"] for row in report["pruned"]} == {"old-done", "young-done"}
    assert (base / "other-done").is_dir() and (base / "running").is_dir(), "eligible folders outside `only` are untouched"
    assert report["scanned"] == 2 and seen == [1, 2]
    ordinary = prune(root, retention_days=7, now=NOW, live=lambda _task: False)
    assert {row["task_id"] for row in ordinary["pruned"]} == {"other-done"}
    assert not any(row["task_id"] in {"old-done", "young-done"} for row in ordinary["skipped"]), "already gone"
