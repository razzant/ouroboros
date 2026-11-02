"""Focused contract proofs for the structural selector diagnostics."""
from __future__ import annotations

import json

from ouroboros import store, store_import
from ouroboros.tools import query_data as query
from ouroboros.tools.registry import ToolContext


def _ctx(tmp_path):
    store_import.ensure_current_store(tmp_path, with_children=False)
    store.update_task(tmp_path, "caller", lambda _old: {
        "task_id": "caller", "status": "completed", "result": "answer",
    })
    repo = tmp_path / "repo"
    repo.mkdir()
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="caller")


def _json(text):
    return json.loads(text)


def test_selector_errors_disclose_structure_never_neighbor_values(tmp_path):
    ctx = _ctx(tmp_path)
    path = tmp_path / "state/queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rows": [{"safe": 1, "neighbor_secret": "VALUE-MUST-NOT-LEAK"}],
    }), encoding="utf-8")

    missing = _json(query._query_data(
        ctx, op="json", kind="queue_snapshot", select="rows[0].missing",
    ))
    assert missing["status"] == "error"
    assert "path 'rows[0]'" in missing["message"] and "segment 'missing'" in missing["message"]
    assert '"neighbor_secret"' in missing["message"]
    assert "VALUE-MUST-NOT-LEAK" not in missing["message"]

    bounds = _json(query._query_data(
        ctx, op="json", kind="queue_snapshot", select="rows[4]",
    ))
    assert "array length=1" in bounds["message"]
    assert "VALUE-MUST-NOT-LEAK" not in bounds["message"]

    wrong_type = _json(query._query_data(
        ctx, op="json", kind="queue_snapshot", select="rows[0].safe.value",
    ))
    assert "expected object, got number" in wrong_type["message"]


def test_missing_filter_field_stays_a_nonmatch_through_query_data(tmp_path):
    ctx = _ctx(tmp_path)
    path = tmp_path / "logs/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"kind":"keep","value":1}\n{"value":2}\n', encoding="utf-8")
    result = _json(query._query_data(
        ctx, op="jsonl", kind="events_log",
        filter={"field": "missing", "op": "eq", "value": 1}, tail=0,
    ))
    assert result["status"] == "ok" and result["data"] == []
