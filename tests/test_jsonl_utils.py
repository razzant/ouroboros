from __future__ import annotations

import json
import os

from ouroboros import utils


def test_iter_jsonl_objects_skips_bad_lines_and_non_dict_by_default(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"ok": 1}),
            "not-json",
            json.dumps(["list"]),
            json.dumps({"ok": 2}),
            "",
        ]),
        encoding="utf-8",
    )

    assert list(iter_jsonl_objects(path)) == [{"ok": 1}, {"ok": 2}]
    assert list(iter_jsonl_objects(path, dict_only=False)) == [{"ok": 1}, ["list"], {"ok": 2}]


def test_iter_jsonl_objects_supports_tail_and_max_entries(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"i": i}) + "\n" for i in range(5)]
    path.write_text("".join(lines), encoding="utf-8")

    assert [entry["i"] for entry in iter_jsonl_objects(path, max_entries=2)] == [3, 4]
    tail = len(lines[-1].encode("utf-8")) + 4
    assert [entry["i"] for entry in iter_jsonl_objects(path, tail_bytes=tail)] == [4]
    assert list(iter_jsonl_objects(path, tail_bytes=0)) == []


def test_iter_jsonl_objects_reports_raw_tail_truncation(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"i": 0}),
            "",
            json.dumps({"i": 1}),
            json.dumps({"i": 2}),
        ]) + "\n",
        encoding="utf-8",
    )
    gaps = set()

    rows = list(iter_jsonl_objects(path, max_entries=3, gap_reasons=gaps))

    assert [row["i"] for row in rows] == [1, 2]
    assert gaps == {"max_entries_truncated"}


def test_iter_jsonl_objects_tail_keeps_line_boundary_start(tmp_path):
    from ouroboros.utils import iter_jsonl_objects

    path = tmp_path / "events.jsonl"
    lines = [json.dumps({"i": i}) + "\n" for i in range(3)]
    path.write_bytes("".join(lines).encode("utf-8"))

    tail = len("".join(lines[1:]).encode("utf-8"))
    assert [entry["i"] for entry in iter_jsonl_objects(path, tail_bytes=tail)] == [1, 2]


def test_supervisor_usage_summaries_use_shared_cost_normalizer(tmp_path, monkeypatch):
    from supervisor import state

    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "events.jsonl").write_text(
        "\n".join([
            json.dumps({
                "type": "llm_usage", "category": "task", "task_id": "a",
                "model": "m1", "cost": 1.25, "prompt_tokens": 10,
            }),
            json.dumps({
                "type": "llm_usage", "category": "review", "task_id": "b",
                "model": "m2", "usage": {"cost": 0.5}, "completion_tokens": 4,
            }),
        ]),
        encoding="utf-8",
    )

    from ouroboros import usage_store

    usage_store.migrate_from_journal(tmp_path)  # the lifecycle job the server runs before any display
    assert state.budget_breakdown({}) == {"task": 1.25, "review": 0.5}
    assert state.model_breakdown({})["m2"]["cost"] == 0.5
    assert [item["task_id"] for item in state.per_task_cost_summary(max_tasks=2)] == ["a", "b"]


def _short_writes(monkeypatch):
    """Every ``os.write`` the writers issue lands one byte; the file lock's own metadata is written whole."""
    real_write = os.write

    def one_byte_at_a_time(fd, data):
        if bytes(data).startswith(b"pid="):
            return real_write(fd, data)
        return real_write(fd, bytes(data)[:1])

    monkeypatch.setattr(utils.os, "write", one_byte_at_a_time)


def test_every_writer_lands_every_byte_through_short_os_writes(tmp_path, monkeypatch):
    """Audit #15-11/12, #16-6 and the external-audit lane of 8827fd2c: one bare ``os.write`` whose byte count was
    trusted published a truncated file behind a successful rename (the fsync lanes, ``supervisor.state``) and tore
    the record of every authority JSONL stream (``append_jsonl``). Each writer loops until every byte lands
    (``utils._write_fd_fully``). Line coverage cannot see that loop, which runs on every full write too."""
    from supervisor import state as sup_state

    text, data = "short-write survivor \u2713 text", b"short-write survivor bytes"
    state = '{"a": 1, "b": "' + "x" * 300 + '"}\n'
    with monkeypatch.context() as patch:
        _short_writes(patch)
        utils.write_text_atomic(tmp_path / "out.txt", text, fsync=True)
        utils.write_bytes_atomic(tmp_path / "out.dat", data, fsync=True)
        assert utils.append_jsonl(tmp_path / "logs" / "events.jsonl", {"type": "llm_usage", "cost": 1.5}) is True
        sup_state.atomic_write_text(tmp_path / "state.json", state)
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == text
    assert (tmp_path / "out.dat").read_bytes() == data
    lines = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"type": "llm_usage", "cost": 1.5}]
    assert (tmp_path / "state.json").read_bytes() == state.encode("utf-8")


def test_append_jsonl_reports_a_write_that_dies_mid_record_and_never_replays_it(tmp_path, monkeypatch):
    """A write that dies mid-record reports False and is not retried whole: a replay would duplicate the prefix
    already on disk. Only the open is retried."""
    target = tmp_path / "logs" / "events.jsonl"
    real_write = os.write
    calls = {"n": 0}

    def half_then_die(fd, data):
        if bytes(data).startswith(b"pid="):
            return real_write(fd, data)
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data)[:4])
        raise OSError("device gone")

    with monkeypatch.context() as patch:
        patch.setattr(utils.os, "write", half_then_die)
        assert utils.append_jsonl(target, {"type": "task_done"}) is False
    assert calls["n"] == 2  # one short write, one failure: no whole-record replay
    assert target.read_bytes() == b'{"ty'
