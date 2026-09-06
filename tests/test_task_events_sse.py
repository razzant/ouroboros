"""SSE /api/tasks/{id}/events follow-phase tests (v6.90.x P2).

The stream's initial replay is a full archive-aware merge; the follow phase
reads only appended bytes per (root, source) log, re-discovers late-spawned
child roots each tick, and heals mid-stream rotation by reading the newest
archive's unconsumed suffix. seq stays monotonic in-stream and the
cross-reconnect cursor contract (ouroboros/cli.py::_watch_task) is preserved.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from ouroboros.gateway.tasks import api_task_events, iter_task_events
from ouroboros.task_results import write_task_result

OLD_TS = "2026-01-01T00:00:00Z"


def _request(data, task_id, *, cursor=0, wait=8):
    return SimpleNamespace(
        path_params={"task_id": task_id},
        query_params={"cursor": str(cursor), "wait": str(wait)},
        app=SimpleNamespace(state=SimpleNamespace(drive_root=data)),
    )


def _parse_frame(frame):
    if not isinstance(frame, str) or frame.startswith(":"):
        return None
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: "):])
    return None


async def _consume(response, on_event=None):
    events = []
    async for frame in response.body_iterator:
        event = _parse_frame(frame)
        if event is None:
            continue
        events.append(event)
        if on_event is not None:
            on_event(event, events)
    return events


def _seed_running_task(tmp_path, task_id="t1", progress_rows=2):
    data = tmp_path / "data"
    logs = data / "logs"
    logs.mkdir(parents=True)
    lines = [
        json.dumps({
            "ts": f"2026-01-01T00:01:{i:02d}Z",
            "content": f"step-{i}",
            "task_id": task_id,
        })
        for i in range(progress_rows)
    ]
    (logs / "progress.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_task_result(data, task_id, "running", result="working", ts=OLD_TS)
    (data / "state").mkdir(parents=True, exist_ok=True)
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    return data


def _append_progress(data, task_id, content, ts):
    with (data / "logs" / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": ts, "content": content, "task_id": task_id}) + "\n")


def _finalize(data, task_id):
    # ts sorts after every progress row used in these tests, so the terminal
    # task_result row is the LAST merged event (a terminal row sorting earlier
    # is also valid — the stream then synthesizes finality for a late cursor).
    write_task_result(data, task_id, "completed", result="done", ts="2026-01-01T00:10:00Z")
    # The follow loop recomputes the terminal projection only when log offsets
    # advanced or the queue snapshot moved — production terminalization always
    # does one of the two; tests bump the snapshot explicitly.
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []} ', encoding="utf-8")


def test_sse_incremental_follow_appends_with_monotonic_seq(tmp_path):
    data = _seed_running_task(tmp_path)
    fired = {"appended": False, "finalized": False}

    def on_event(event, events):
        if not fired["appended"] and len(events) >= 3:  # 2 progress + running task_result
            _append_progress(data, "t1", "late-step", "2026-01-01T00:02:00Z")
            fired["appended"] = True
        if fired["appended"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "late-step" for e in events
        ):
            _finalize(data, "t1")
            fired["finalized"] = True

    # The stream closes at the wait deadline with a heartbeat; on a loaded
    # macOS runner the default 8 s elapsed before the appended row and the
    # finalization were observed. Finalization still ends the stream early.
    response = asyncio.run(api_task_events(_request(data, "t1", wait=60)))
    events = asyncio.run(_consume(response, on_event))

    contents = [(e.get("data") or {}).get("content") for e in events if e["type"] == "progress"]
    assert contents == ["step-0", "step-1", "late-step"]
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    final = events[-1]
    assert final["type"] == "task_result"
    assert final["data"]["status"] == "completed"


def test_sse_cursor_resume_matches_initial_replay_positions(tmp_path):
    """Reconnecting with cursor=N replays exactly the events after position N —
    the CLI's cross-reconnect contract."""
    data = _seed_running_task(tmp_path, progress_rows=4)
    _finalize(data, "t1")

    first = asyncio.run(_consume(asyncio.run(api_task_events(_request(data, "t1", wait=0)))))
    assert len(first) >= 5  # 4 progress + terminal task_result

    resumed = asyncio.run(
        _consume(asyncio.run(api_task_events(_request(data, "t1", cursor=2, wait=0))))
    )

    assert [e["seq"] for e in resumed] == [e["seq"] for e in first[2:]]
    assert [(e["type"], (e.get("data") or {}).get("content")) for e in resumed] == [
        (e["type"], (e.get("data") or {}).get("content")) for e in first[2:]
    ]


def test_sse_survives_mid_stream_rotation_without_loss_or_duplicates(tmp_path):
    data = _seed_running_task(tmp_path)
    live = data / "logs" / "progress.jsonl"
    fired = {"rotated": False, "finalized": False}

    def on_event(event, events):
        if not fired["rotated"] and len(events) >= 3:
            # An unconsumed row lands just before the rotation…
            _append_progress(data, "t1", "pre-rotation", "2026-01-01T00:02:00Z")
            archive_dir = data / "archive"
            archive_dir.mkdir(exist_ok=True)
            os.replace(live, archive_dir / "progress_20260101T000200.jsonl")
            live.touch()
            # …and a fresh row starts the new live file.
            _append_progress(data, "t1", "post-rotation", "2026-01-01T00:02:01Z")
            fired["rotated"] = True
        if fired["rotated"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "post-rotation" for e in events
        ):
            _finalize(data, "t1")
            fired["finalized"] = True

    response = asyncio.run(api_task_events(_request(data, "t1")))
    events = asyncio.run(_consume(response, on_event))

    contents = [(e.get("data") or {}).get("content") for e in events if e["type"] == "progress"]
    # The archive suffix (pre-rotation) and the new live row both arrive, exactly
    # once each, and the pre-rotation history is not re-emitted.
    assert contents == ["step-0", "step-1", "pre-rotation", "post-rotation"]
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert events[-1]["data"]["status"] == "completed"


def test_sse_discovers_late_spawned_child_root(tmp_path):
    data = _seed_running_task(tmp_path, task_id="p1", progress_rows=1)
    # A row in the PARENT's progress log that matches the lineage only via
    # subagent_task_id, written (and consumed) BEFORE the child is discovered
    # (P2 review, fix 3): only the filter-growth re-merge can recover it.
    with (data / "logs" / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "ts": "2026-01-01T00:02:30Z",
            "content": "pre-spawn-lineage",
            "task_id": "",
            "subagent_task_id": "c1",
        }) + "\n")
    child_drive = tmp_path / "childdrive"
    fired = {"spawned": False, "finalized": False}

    def on_event(event, events):
        if not fired["spawned"] and len(events) >= 2:
            (child_drive / "logs").mkdir(parents=True, exist_ok=True)
            (child_drive / "logs" / "progress.jsonl").write_text(
                json.dumps({
                    "ts": "2026-01-01T00:03:00Z",
                    "content": "child-step",
                    "task_id": "c1",
                }) + "\n",
                encoding="utf-8",
            )
            write_task_result(
                data, "c1", "running",
                delegation_role="subagent",
                parent_task_id="p1",
                root_task_id="p1",
                child_drive_root=str(child_drive),
                ts=OLD_TS,
            )
            fired["spawned"] = True
        if fired["spawned"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "child-step" for e in events
        ):
            _finalize(data, "p1")
            fired["finalized"] = True

    response = asyncio.run(api_task_events(_request(data, "p1")))
    events = asyncio.run(_consume(response, on_event))

    child_rows = [e for e in events if (e.get("data") or {}).get("content") == "child-step"]
    assert len(child_rows) == 1  # the late child's log joined at offset 0
    lineage_rows = [e for e in events if (e.get("data") or {}).get("content") == "pre-spawn-lineage"]
    assert len(lineage_rows) == 1  # recovered by the filter-growth re-merge (fix 3)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert events[-1]["data"]["status"] == "completed"


def test_iter_task_events_reads_progress_archive_chain(tmp_path):
    """The initial replay is archive-aware: rotated progress rows precede live
    rows in the merged, seq-numbered order. The result ts is the task's creation
    floor (P2 review, fix 4), so it must predate the task's own rows — the
    task_result row therefore sorts first here."""
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    (data / "archive").mkdir()
    (data / "archive" / "progress_20260101T000100.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:30Z", "content": "archived", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    (data / "logs" / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:01:30Z", "content": "live", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    write_task_result(data, "t1", "completed", result="done", ts="2026-01-01T00:00:10Z")

    events = iter_task_events(data, "t1")

    progress = [e for e in events if e["type"] == "progress"]
    assert [(e["data"]["content"]) for e in progress] == ["archived", "live"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["type"] == "task_result"


def test_sse_archive_floor_skips_archives_predating_task_creation(tmp_path):
    """Fix 4: an archive whose rotation stamp predates the watched task's raw
    creation ts is never read (bounds the glob to the task lifetime); archives
    stamped after creation are still consulted."""
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    (data / "archive").mkdir()
    # Ancient archive rotated before the task existed — even a task_id-colliding
    # row inside it must not be read.
    (data / "archive" / "progress_20251231T000000.jsonl").write_text(
        json.dumps({"ts": "2025-12-31T00:00:00Z", "content": "ancient", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    (data / "archive" / "progress_20260101T000200.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:01:30Z", "content": "recent-archived", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    (data / "logs" / "progress.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:02:30Z", "content": "live-row", "task_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    write_task_result(data, "t1", "running", result="working", ts="2026-01-01T00:01:00Z")

    events = iter_task_events(data, "t1")

    contents = [(e.get("data") or {}).get("content") for e in events if e["type"] == "progress"]
    assert contents == ["recent-archived", "live-row"]  # "ancient" skipped


def test_sse_rotation_stash_consumed_keeps_new_live_offset(tmp_path):
    """Fix 2: when the inode flip was observed BEFORE the archive became visible
    (offset stashed, new live file partially consumed), the later archive tick
    must keep the current live offset/ino — resetting to 0 would re-emit the
    already-consumed rows of the new live file."""
    from ouroboros.gateway.tasks import _TaskEventFollower

    data = _seed_running_task(tmp_path)
    live = data / "logs" / "progress.jsonl"
    follower = _TaskEventFollower(data, "t1")
    follower.full_merge()  # consumes step-0/step-1

    # An unconsumed row lands, then the live file is replaced while its archive
    # is NOT yet visible (moved aside), and the new live file starts.
    _append_progress(data, "t1", "pre-rotation", "2026-01-01T00:02:00Z")
    side = tmp_path / "rotated_aside.jsonl"
    os.replace(live, side)
    live.touch()
    _append_progress(data, "t1", "post-rotation-1", "2026-01-01T00:02:01Z")

    rows, _ = follower.poll()  # stash tick: consumes part of the NEW live file
    assert [(r.get("data") or {}).get("content") for r in rows if r["type"] == "progress"] == [
        "post-rotation-1"
    ]

    # The archive becomes visible; more rows land on the new live file.
    (data / "archive").mkdir(exist_ok=True)
    os.replace(side, data / "archive" / "progress_20260101T000201.jsonl")
    _append_progress(data, "t1", "post-rotation-2", "2026-01-01T00:02:02Z")

    rows, _ = follower.poll()
    contents = [(r.get("data") or {}).get("content") for r in rows if r["type"] == "progress"]
    # The archive suffix and the new live delta — and NO duplicate of the
    # already-consumed post-rotation-1.
    assert contents == ["pre-rotation", "post-rotation-2"]


def test_sse_older_ts_append_triggers_full_remerge_without_losing_newer_rows(tmp_path):
    """Fix 7a: a mid-stream row with an OLDER ts than the emitted tail forces the
    full re-merge; strictly-newer rows are still delivered exactly once,
    duplicates of already-emitted rows are tolerated, seq stays monotonic, and
    the backdated row itself is dropped for this stream (disclosed cursor
    parity — a from-zero replay recovers it)."""
    data = _seed_running_task(tmp_path)
    fired = {"appended": False, "finalized": False}

    def on_event(event, events):
        if not fired["appended"] and len(events) >= 3:
            _append_progress(data, "t1", "backdated", "2026-01-01T00:00:30Z")
            _append_progress(data, "t1", "fresh", "2026-01-01T00:02:00Z")
            fired["appended"] = True
        if fired["appended"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "fresh" for e in events
        ):
            _finalize(data, "t1")
            fired["finalized"] = True

    response = asyncio.run(api_task_events(_request(data, "t1")))
    events = asyncio.run(_consume(response, on_event))

    contents = [(e.get("data") or {}).get("content") for e in events if e["type"] == "progress"]
    assert contents.count("fresh") == 1
    assert "backdated" not in contents  # sorts before the cursor: dropped, not duplicated
    assert contents.count("step-0") >= 1 and contents.count("step-1") >= 1
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert events[-1]["data"]["status"] == "completed"


def test_sse_double_rotation_between_ticks_delivers_all_rows_once(tmp_path):
    """Fix 7b: two rotations between ticks — the first new archive is read from
    the consumed offset, the second fully, the new live file from 0; every row
    arrives exactly once."""
    data = _seed_running_task(tmp_path)
    live = data / "logs" / "progress.jsonl"
    fired = {"rotated": False, "finalized": False}

    def on_event(event, events):
        if not fired["rotated"] and len(events) >= 3:
            archive_dir = data / "archive"
            archive_dir.mkdir(exist_ok=True)
            _append_progress(data, "t1", "rot1-row", "2026-01-01T00:02:00Z")
            os.replace(live, archive_dir / "progress_20260101T000200.jsonl")
            live.touch()
            _append_progress(data, "t1", "rot2-row", "2026-01-01T00:02:01Z")
            os.replace(live, archive_dir / "progress_20260101T000201.jsonl")
            live.touch()
            _append_progress(data, "t1", "post-row", "2026-01-01T00:02:02Z")
            fired["rotated"] = True
        if fired["rotated"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "post-row" for e in events
        ):
            _finalize(data, "t1")
            fired["finalized"] = True

    response = asyncio.run(api_task_events(_request(data, "t1")))
    events = asyncio.run(_consume(response, on_event))

    contents = [(e.get("data") or {}).get("content") for e in events if e["type"] == "progress"]
    assert contents == ["step-0", "step-1", "rot1-row", "rot2-row", "post-row"]
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert events[-1]["data"]["status"] == "completed"


def test_sse_terminal_merge_row_uses_one_materializing_read(tmp_path, monkeypatch):
    """Fix 5: the terminal task_result row emitted through the merge path is
    replaced with exactly ONE materializing (True) read at emission time; every
    other effective read in the stream stays a False projection."""
    import ouroboros.gateway.tasks as tasks_mod

    data = _seed_running_task(tmp_path, progress_rows=1)
    _finalize(data, "t1")
    real = tasks_mod.load_effective_task_result
    flags = []

    def spy(drive_root, task_id, **kw):
        flags.append(kw.get("materialize_artifacts", True))
        return real(drive_root, task_id, **kw)

    monkeypatch.setattr(tasks_mod, "load_effective_task_result", spy)

    response = asyncio.run(api_task_events(_request(data, "t1", wait=0)))
    events = asyncio.run(_consume(response))

    assert events[-1]["type"] == "task_result"
    assert events[-1]["data"]["status"] == "completed"
    assert flags.count(True) == 1  # the sanctioned terminal-emission read
    assert flags[-1] is True  # ...and it happens at emission, after the projections


def test_sse_torn_child_result_is_retried_and_discovered_next_tick(tmp_path):
    """Scandir name-diff seen-set rule (P2 wave-1, GPT#6): a name is committed
    only after a successful read — a torn/mid-write child file is re-read on the
    next tick and the child is discovered once the write completes."""
    from ouroboros.gateway.tasks import _TaskEventFollower

    data = _seed_running_task(tmp_path, task_id="p1")
    follower = _TaskEventFollower(data, "p1")
    follower.full_merge()

    (data / "task_results" / "c1.json").write_text("{torn", encoding="utf-8")
    follower.poll()
    assert "c1" not in follower.task_filter_ids
    assert "c1.json" not in follower._seen_result_names  # NOT committed

    child_drive = tmp_path / "childdrive"
    child_drive.mkdir()
    (data / "task_results" / "c1.json").write_text(
        json.dumps({
            "task_id": "c1",
            "status": "running",
            "delegation_role": "subagent",
            "parent_task_id": "p1",
            "root_task_id": "p1",
            "child_drive_root": str(child_drive),
            "ts": OLD_TS,
        }),
        encoding="utf-8",
    )
    follower.poll()

    assert "c1" in follower.task_filter_ids
    assert "c1.json" in follower._seen_result_names
    assert follower.filter_grew  # the stream will run the recovery re-merge
    assert any(str(root) == str(child_drive) for root in follower.roots)


def test_sse_nonlineage_result_names_read_once_then_committed(tmp_path, monkeypatch):
    """Successfully-read NON-lineage names are committed to the seen-set too,
    so an unrelated busy store is never re-read on every tick."""
    import ouroboros.gateway.tasks as gateway_tasks

    data = _seed_running_task(tmp_path, task_id="p1")
    follower = gateway_tasks._TaskEventFollower(data, "p1")
    follower.full_merge()

    (data / "task_results" / "other.json").write_text(
        json.dumps({"task_id": "other", "status": "running", "delegation_role": "root", "ts": OLD_TS}),
        encoding="utf-8",
    )
    reads = []
    real = gateway_tasks.read_json_dict
    monkeypatch.setattr(
        gateway_tasks, "read_json_dict", lambda path: reads.append(str(path)) or real(path)
    )

    follower.poll()
    follower.poll()
    follower.poll()

    other_reads = [path for path in reads if path.endswith("other.json")]
    assert len(other_reads) == 1  # committed after ONE successful read
    assert "other" not in follower.task_filter_ids


def test_sse_child_discovered_mid_stream_then_reconnect_replays_once(tmp_path):
    """Reconnect grid: after a child was discovered mid-stream, a fresh stream
    (fresh follower, from-zero replay) re-discovers the child from scratch and
    delivers its rows exactly once with monotonic seq."""
    data = _seed_running_task(tmp_path, task_id="p1", progress_rows=1)
    child_drive = tmp_path / "childdrive"
    fired = {"spawned": False, "finalized": False}

    def on_event(event, events):
        if not fired["spawned"] and len(events) >= 1:
            (child_drive / "logs").mkdir(parents=True, exist_ok=True)
            (child_drive / "logs" / "progress.jsonl").write_text(
                json.dumps({"ts": "2026-01-01T00:03:00Z", "content": "child-step", "task_id": "c1"}) + "\n",
                encoding="utf-8",
            )
            write_task_result(
                data, "c1", "running",
                delegation_role="subagent",
                parent_task_id="p1",
                root_task_id="p1",
                child_drive_root=str(child_drive),
                ts=OLD_TS,
            )
            fired["spawned"] = True
        if fired["spawned"] and not fired["finalized"] and any(
            (e.get("data") or {}).get("content") == "child-step" for e in events
        ):
            _finalize(data, "p1")
            fired["finalized"] = True

    first = asyncio.run(_consume(asyncio.run(api_task_events(_request(data, "p1"))), on_event))
    assert any((e.get("data") or {}).get("content") == "child-step" for e in first)

    resumed = asyncio.run(_consume(asyncio.run(api_task_events(_request(data, "p1", wait=0)))))

    child_rows = [e for e in resumed if (e.get("data") or {}).get("content") == "child-step"]
    assert len(child_rows) == 1
    seqs = [e["seq"] for e in resumed]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert resumed[-1]["data"]["status"] == "completed"


def test_sse_stream_on_running_task_does_zero_artifact_work(tmp_path, monkeypatch):
    """The SSE replay/follow reads are False projections: no artifact collection
    or copy may run while the task is not terminal (materialize_artifacts
    contract; the one True read is reserved for terminal emission)."""
    import ouroboros.artifacts as artifacts_mod

    data = _seed_running_task(tmp_path)
    calls = []
    monkeypatch.setattr(artifacts_mod, "collect_task_artifact_records", lambda *a, **k: calls.append("collect") or [])
    monkeypatch.setattr(artifacts_mod, "copy_file_to_task_artifacts", lambda *a, **k: calls.append("copy") or {})

    response = asyncio.run(api_task_events(_request(data, "t1", wait=0)))
    events = asyncio.run(_consume(response))

    assert any(e["type"] == "progress" for e in events)
    assert calls == []
