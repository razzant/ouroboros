"""One read of a log's bounded tail per context build: the health block's log check and the Runtime
section's body fact share ``jsonl_tail.cached_tail_objects``, memoized on the file's identity
(device, inode, size, mtime); an append reads again, an unchanged file never does."""
from __future__ import annotations

import json

from ouroboros import context_health, jsonl_tail
from ouroboros.context_runtime_facts import vitals_fact
from tests._context_shared import _make_health_env


def _row(**fields) -> str:
    return json.dumps({"ts": "2026-10-25T18:00:00+00:00", **fields}) + "\n"


def test_one_read_serves_the_health_block_and_the_body_fact_and_an_append_reads_again(tmp_path, monkeypatch):
    monkeypatch.setattr(jsonl_tail, "_TAILS", {})
    reads: list = []
    real = jsonl_tail.iter_jsonl_objects
    monkeypatch.setattr(jsonl_tail, "iter_jsonl_objects", lambda path, **kw: reads.append(str(path)) or real(path, **kw))
    env = _make_health_env(tmp_path, events_lines=[json.dumps({"type": "llm_api_error", "model": "m"})])
    supervisor, events = tmp_path / "logs" / "supervisor.jsonl", tmp_path / "logs" / "events.jsonl"
    supervisor.write_text(_row(type="boot", branch="o", sha="abcdef1234567890"), encoding="utf-8")

    context_health.build_health_invariants(env)
    fact = vitals_fact(tmp_path, observed_at="2026-10-25T20:00:00+00:00")
    assert fact["boot"] == "18:00 o@abcdef123456"
    assert reads.count(str(supervisor)) == 1, reads  # the health check and the body fact: one read
    assert reads.count(str(events)) == 1, reads  # the duplicate-processing and provider-error checks: one read

    with supervisor.open("a", encoding="utf-8") as handle:  # an append changes size and mtime: a fresh read
        handle.write(_row(type="supervisor_load", window_sec=60.0, max_event_lag_sec=0.2, handlers={}, lanes={}))
    fact = vitals_fact(tmp_path, observed_at="2026-10-25T20:00:00+00:00")
    assert fact["busy_minute"].startswith("18:00: lag ≤0.2 s") and reads.count(str(supervisor)) == 2
    vitals_fact(tmp_path, observed_at="2026-10-25T20:00:00+00:00")
    context_health.build_health_invariants(env)
    assert reads.count(str(supervisor)) == 2 and reads.count(str(events)) == 1  # unchanged files: no read
    assert jsonl_tail.cached_tail_objects(tmp_path / "logs" / "absent.jsonl", 256_000) == ()
    assert set(jsonl_tail._TAILS) == {str(supervisor), str(events)}  # one entry per path
