"""#3047: who asked for a Continue is recorded apart from the origin the work inherits.

The real HTTP door, the mind's control door and ``admit_continuation`` write the record; the frame carrier reads it.
A declaration from the app is written down as a declaration: it grants nothing, and nothing proves it.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ouroboros.subagent_messages import initiator_meta
from ouroboros.task_results import load_task_result
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE, _interrupted

APP = {"X-Ouroboros-Surface": "app", "X-Ouroboros-Actor": "owner"}


def _press(task_id, headers):
    from starlette.requests import Request

    from ouroboros.gateway.task_continue import api_task_continue
    from supervisor import queue

    payload = json.dumps({"action_nonce": NONCE}).encode()

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": f"/api/tasks/{task_id}/continue",
                       "path_params": {"task_id": task_id},
                       "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=queue.DRIVE_ROOT))}, receive)
    response = asyncio.run(api_task_continue(request))
    return response.status_code, json.loads(response.body)


def _admission(root, successor):
    return load_task_result(root, successor)["metadata"]["continuation"]["admission"]


def test_the_apps_press_is_recorded_as_a_declared_owner_and_read_apart_from_the_origin(tmp_path, monkeypatch):
    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    status, ack = _press("pred-1", APP)
    assert status == 200 and ack["ok"]
    by = _admission(tmp_path, ack["successor_task_id"])["by"]
    assert (by["actor"], by["declared"], by["surface"]) == ("owner", True, "app") and by["at"]
    assert initiator_meta(load_task_result(tmp_path, ack["successor_task_id"])) == {"continue_requester": "owner"}
    # The same press replayed answers the same admission, whoever retries it.
    status, again = _press("pred-1", {})
    assert status == 200 and again["successor_task_id"] == ack["successor_task_id"]
    assert _admission(tmp_path, ack["successor_task_id"])["by"]["actor"] == "owner"


def test_an_undeclared_post_stays_unknown(tmp_path, monkeypatch):
    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    status, ack = _press("pred-1", {})
    assert status == 200 and ack["ok"]
    assert _admission(tmp_path, ack["successor_task_id"])["by"] == {}
    assert initiator_meta(load_task_result(tmp_path, ack["successor_task_id"])) == {"continue_requester": "unknown"}


def test_a_malformed_declaration_names_nobody(tmp_path, monkeypatch):
    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    status, ack = _press("pred-1", {"X-Ouroboros-Surface": "app", "X-Ouroboros-Actor": "administrator"})
    assert status == 200 and _admission(tmp_path, ack["successor_task_id"])["by"] == {}


def test_a_minds_door_records_its_own_task(tmp_path, monkeypatch):
    from supervisor.events_mind_controls import _apply

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    ack = _apply({"action": "continue_task", "task_id": "pred-1", "actor_task_id": "mind-1", "action_nonce": NONCE,
                  "reason": "carry it on"})
    assert ack["ok"]
    by = _admission(tmp_path, ack["successor_task_id"])["by"]
    assert (by["actor"], by["task_id"]) == ("ouroboros", "mind-1")
    assert "declared" not in by
    assert initiator_meta(load_task_result(tmp_path, ack["successor_task_id"]))["continue_requester"] == "ouroboros"


def test_a_wake_keeps_its_origin_and_gains_a_requester_only_when_continued():
    wake = {"metadata": {"initiator": "consciousness", "work_kind": "consciousness"}}
    assert initiator_meta(wake) == {"initiator": "consciousness"}
    continued = {"metadata": {**wake["metadata"], "continuation": {"admission": {"by": {"actor": "owner", "declared": True}}}}}
    assert initiator_meta(continued) == {"initiator": "consciousness", "continue_requester": "owner"}
    # The metadata record is itself the frame carrier's input on the loop side.
    assert initiator_meta(continued["metadata"]) == {"initiator": "consciousness", "continue_requester": "owner"}
    historical = {"metadata": {"continuation": {"admission": {"by": {}}}}}
    assert initiator_meta(historical) == {"continue_requester": "unknown"}
    assert initiator_meta({"metadata": {"continuation": {"predecessor_task_id": "p"}}}) == {}


_CONTINUED = {"continuation": {"predecessor_task_id": "pred-1", "admission": {"by": {"actor": "owner", "declared": True}}}}


def _pushed_terminal(tmp_path, task_id, evt, *, settle):
    """The live log frames a terminal published, when it is the first frame the card ever sees."""
    from ouroboros.headless import prepare_terminal_task_files
    from supervisor.events import _handle_task_done
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    sent, pushed = [], []
    ctx = _mk_ctx(tmp_path, sent, pushed)
    task = {"id": task_id, "type": "task", "chat_id": 1, "metadata": dict(_CONTINUED)}
    ctx.RUNNING[task_id] = {"task": task}
    if settle:
        from ouroboros.task_results import write_task_result

        write_task_result(tmp_path, task_id, "completed", result="done", chat_id=1, metadata=dict(_CONTINUED))
        assert not prepare_terminal_task_files(tmp_path, task)["error"]
    _handle_task_done({"type": "task_done", "task_id": task_id, "task_type": "task", "worker_id": 0, "chat_id": 1,
                       "ts": "2026-11-10T09:00:00Z", **evt}, ctx)
    return [p for p in pushed if p.get("type") == "task_done" and p.get("task_id") == task_id]


def test_a_continue_that_ends_before_any_progress_frame_still_names_its_requester(tmp_path):
    done = _pushed_terminal(tmp_path, "next-1", {"_files_prepared_attempt": 1}, settle=True)
    assert [row.get("continue_requester") for row in done] == ["owner"]


def test_a_continue_whose_worker_faults_before_any_progress_frame_still_names_its_requester(tmp_path):
    done = _pushed_terminal(tmp_path, "next-2", {"status": "running"}, settle=False)
    assert [(row.get("reason_code"), row.get("continue_requester")) for row in done] == [("task_done_lifecycle_fault", "owner")]


def test_a_terminal_of_work_nobody_continued_names_no_requester(tmp_path, monkeypatch):
    monkeypatch.setitem(_CONTINUED, "continuation", {})
    done = _pushed_terminal(tmp_path, "plain-1", {"_files_prepared_attempt": 1}, settle=True)
    assert done and "continue_requester" not in done[0]


def _history(tmp_path, monkeypatch):
    from ouroboros.gateway import history

    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *a, **k: [])
    return json.loads(history._assemble_history_response(tmp_path, 1, 10, 10))["messages"]


def test_a_history_page_without_a_provenance_row_keeps_the_requester(tmp_path, monkeypatch):
    """The page reads the task through its display view: the requester must survive it."""
    from ouroboros.store_import import ensure_current_store

    ts = "2026-11-10T09:00:00Z"
    for rel, rows in {"logs/chat.jsonl": [{"direction": "out", "task_id": "history-only", "text": "Done.", "ts": ts, "chat_id": 1}],
                      "logs/tools.jsonl": [{"task_id": "history-only", "ts": ts, "type": "tool_call", "tool": "read_file",
                                            "invocation_id": "history-only-call", "is_error": False}],
                      "task_results/history-only.json": [{
                          "_schema_version": 1, "task_id": "history-only", "status": "completed", "chat_id": 1, "reason_code": "finished",
                          "metadata": {"continuation": {"predecessor_task_id": "old", "admission": {"by": {"actor": "owner"}}}}}]}.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ensure_current_store(tmp_path)
    carrier = next(row for row in _history(tmp_path, monkeypatch) if row.get("system_type") == "task_evidence")
    assert carrier["continue_requester"] == "owner" and carrier["task_terminal_status"] == "completed"


def test_the_producers_own_history_row_names_the_requester(tmp_path, monkeypatch):
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.store_import import ensure_current_store

    task = {"id": "host-facts", "chat_id": 1,
            "metadata": {"continuation": {"predecessor_task_id": "old", "admission": {"by": {"actor": "owner"}}}}}
    path = tmp_path / "task_results/host-facts.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**task, "task_id": task["id"], "status": "completed", "reason_code": "finished", "_schema_version": 1}))
    ensure_current_store(tmp_path)
    _record_task_facts(SimpleNamespace(drive_root=tmp_path), task, {},
                       {"tool_calls": [{"tool": "read_file", "is_error": False}]}, tmp_path / "logs")
    row = next(row for row in _history(tmp_path, monkeypatch) if row.get("system_type") == "task_summary")
    assert row["continue_requester"] == "owner" and not row.get("historical_terminal")
