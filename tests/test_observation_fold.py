"""CPL4-C23 pins: acknowledged observations fold to archive, unACKed never.

An acknowledged enqueue older than GC retention moves — with every ack row
naming it — into ``archive/consciousness_observations_<ts>.jsonl``; pending
rows of any age and fresh acknowledged rows stay in the live inbox verbatim.
Any malformed line or ghost ack skips the whole fold (the same gap classes
that block a live ack), and the surviving inbox replays cleanly.
"""

from __future__ import annotations

import json
import queue
from unittest.mock import MagicMock, patch

from ouroboros.consciousness import compact_acknowledged_observations
from ouroboros.utils import utc_now_iso

_OLD_TS = "2020-01-01T00:00:00+00:00"


def _inbox(tmp_path):
    path = tmp_path / "state" / "consciousness_observations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _enqueue(identifier, ts):
    return {"op": "enqueue", "id": identifier, "source": "s", "kind": "k",
            "time": ts, "payload": f"payload-{identifier}", "ref": "r"}


def _ack(identifier, ts):
    return {"op": "ack", "id": identifier, "time": ts}


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_old_acked_rows_fold_pending_and_fresh_stay(tmp_path):
    path = _inbox(tmp_path)
    now = utc_now_iso()
    rows = [
        _enqueue("old-acked", _OLD_TS), _ack("old-acked", _OLD_TS),
        _enqueue("old-pending", _OLD_TS),
        _enqueue("fresh-acked", now), _ack("fresh-acked", now),
    ]
    _write(path, rows)

    report = compact_acknowledged_observations(tmp_path)

    assert report["folded"] == 1 and not report["skipped"]
    live_ids = [
        (row["op"], row["id"])
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]
    assert live_ids == [
        ("enqueue", "old-pending"),
        ("enqueue", "fresh-acked"), ("ack", "fresh-acked"),
    ]
    segment = tmp_path / "archive" / report["archive"]
    archived = [json.loads(line) for line in segment.read_text(encoding="utf-8").splitlines()]
    assert [(row["op"], row["id"]) for row in archived] == [
        ("enqueue", "old-acked"), ("ack", "old-acked"),
    ]


def test_ghost_ack_or_malformed_line_skips_the_fold(tmp_path):
    path = _inbox(tmp_path)
    _write(path, [_ack("never-enqueued", _OLD_TS), _enqueue("old", _OLD_TS)])
    before = path.read_bytes()
    report = compact_acknowledged_observations(tmp_path)
    assert report["skipped"] == "ghost_ack" and path.read_bytes() == before

    _write(path, [_enqueue("old", _OLD_TS), _ack("old", _OLD_TS)])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{torn line\n")
    before = path.read_bytes()
    report = compact_acknowledged_observations(tmp_path)
    assert report["skipped"] == "malformed_row" and path.read_bytes() == before


def test_survivors_replay_cleanly_after_the_fold(tmp_path):
    from ouroboros.consciousness import BackgroundConsciousness

    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True)
    (tmp_path / "repo").mkdir()
    path = _inbox(drive)
    _write(path, [
        _enqueue("old-acked", _OLD_TS), _ack("old-acked", _OLD_TS),
        _enqueue("still-pending", _OLD_TS),
    ])
    assert compact_acknowledged_observations(drive)["folded"] == 1

    with patch.object(BackgroundConsciousness, "_build_registry", return_value=MagicMock()):
        bc = BackgroundConsciousness(
            drive_root=drive, repo_dir=tmp_path / "repo",
            event_queue=queue.Queue(), owner_chat_id_fn=lambda: None,
        )
    pending = bc._snapshot_pending_observations()
    assert [row.get("id") for row in pending] == ["still-pending"]
    assert bc.status_snapshot().get("observation_gap_count", 0) == 0


def test_startup_prune_sweeps_run_the_fold():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "compact_acknowledged_observations" in inspect.getsource(sm._startup_prune_sweeps)
