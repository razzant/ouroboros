"""Boot leaves durable phase facts in ``logs/supervisor.jsonl`` (facts, never a gate)."""
from __future__ import annotations

import inspect
import json

from ouroboros import server_liveness


def _rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_readiness_records_the_process_age(tmp_path, monkeypatch):
    monkeypatch.setattr(server_liveness, "DATA_DIR", tmp_path)
    server_liveness.note_supervisor_ready()
    rows = [row for row in _rows(tmp_path) if row.get("type") == "supervisor_ready"]
    assert len(rows) == 1 and rows[0]["ts"]
    assert isinstance(rows[0]["process_age_sec"], float) and rows[0]["process_age_sec"] >= 0


def test_an_unwritable_log_never_blocks_readiness(tmp_path, monkeypatch):
    (tmp_path / "logs").write_text("a file where the directory should be", encoding="utf-8")
    monkeypatch.setattr(server_liveness, "DATA_DIR", tmp_path)
    server_liveness.note_supervisor_ready()  # must not raise: the fact is optional, readiness is not


def test_the_supervisor_records_the_fact_only_after_it_is_ready():
    import server

    source = inspect.getsource(server._run_supervisor)
    ready, note = source.index("_supervisor_ready.set()"), source.index("note_supervisor_ready()")
    assert ready < note < source.index("offset = 0")
