"""Project work cards receive named work, live activity and separate outcome facts."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from ouroboros.gateway import state as gateway_state
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.task_results import load_task_result, write_task_result


def _history(root):
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": "1"})))
    assert response.status_code == 200
    return json.loads(response.body)["messages"]


@pytest.fixture
def activity_queue(tmp_path, monkeypatch):
    from supervisor import queue

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    gateway_state._FINALIZING_MEMO.clear()
    return queue


@pytest.mark.parametrize("status,execution", [("failed", "infra_failed"), ("completed", "ok"), ("completed", "degraded")])
def test_live_census_keeps_known_outcome_separate_from_open_finalization(tmp_path, activity_queue, status, execution):
    from ouroboros.outcomes import normalize_outcome_axes, terminal_outcome_axes

    task = {"id": "root", "root_task_id": "root", "chat_id": 1}
    activity_queue.RUNNING["root"] = {"task": task, "started_at": 1}
    write_task_result(
        tmp_path, "root", status, result="answer not needed by the census",
        reason_code="observed_result",
        outcome_axes=terminal_outcome_axes(lifecycle=status, execution=execution, reason_code="observed_result"),
        root_phase_checkpoint={"post_task_synthesis": "running", "post_task_pause": {"actor_source": "private work"}},
    )

    row = gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])[0]
    assert row["phase"] == "finalizing"
    assert row["status"] == status
    assert row["outcome_axes"] == normalize_outcome_axes(load_task_result(tmp_path, "root"))
    assert row["outcome_axes"]["execution"]["status"] == execution
    assert row["root_phase_checkpoint"] == {"post_task_synthesis": "running"}
    assert row["reason_code"] == "observed_result"
    assert "task_terminal_status" not in row and "result" not in row

    # Once the real activity owner releases it, an old result creates no census row.
    activity_queue.RUNNING.clear()
    assert gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[]) == []


def test_census_result_memo_refreshes_without_inventing_a_missing_outcome(tmp_path, activity_queue, monkeypatch):
    from ouroboros import utils

    activity_queue.RUNNING["root"] = {"task": {"id": "root", "chat_id": 1}, "started_at": 1}
    initial = gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])[0]
    assert initial["phase"] == "working"
    assert "status" not in initial and "outcome_axes" not in initial
    write_task_result(tmp_path, "root", "running")
    assert gateway_state._task_activity_facts(tmp_path, "root")["display"]["status"] == "running"

    original = utils.read_json_dict
    reads = []

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(utils, "read_json_dict", counted)
    gateway_state._task_activity_facts(tmp_path, "root")
    assert reads == []
    write_task_result(tmp_path, "root", "failed", root_phase_checkpoint={"post_task_synthesis": "running"})
    reads.clear()
    row = gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])[0]
    assert row["status"] == "failed" and row["phase"] == "finalizing"
    assert len([path for path in reads if path.name == "root.json"]) == 1


def test_active_successor_exposes_explicit_retry_link_before_result_publication(tmp_path, activity_queue):
    task = {"id": "retry", "root_task_id": "first", "delegation_role": "root", "chat_id": 1,
            "original_task_id": "first", "timeout_retry_from": "first"}
    continuation = {"id": "owner-continue", "chat_id": 1, "root_task_id": "owner-continue",
                    "predecessor_task_id": "first", "metadata": {"continuation": {"predecessor_task_id": "first"}}}
    activity_queue.PENDING.extend([task, continuation])

    rows = {row["activity_id"]: row for row in gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])}
    assert rows["retry"]["phase"] == "queued"
    assert rows["retry"]["timeout_retry_from"] == rows["retry"]["original_task_id"] == "first"
    assert "timeout_retry_from" not in rows["owner-continue"]
    assert "original_task_id" not in rows["owner-continue"]

    # The successor retains its identity when its result arrives after admission.
    activity_queue.PENDING.clear()
    activity_queue.RUNNING["retry"] = {"task": task, "started_at": 2}
    write_task_result(tmp_path, "retry", "running", original_task_id="first", timeout_retry_from="first")
    row = gateway_state._chat_activities_snapshot_safe(tmp_path, direct_turns=[])[0]
    assert row["activity_id"] == "retry" and row["phase"] == "working"
    assert row["timeout_retry_from"] == row["original_task_id"] == "first"


@pytest.mark.parametrize("kind", ["project_started", "project_handoff"])
def test_project_work_name_survives_real_outbox_bus_live_and_history(tmp_path, monkeypatch, kind):
    from ouroboros.project_dialogue import announce_project_started, build_owner_message_ref
    from ouroboros.project_handoff import enqueue_project_handoff
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor import events, events_chat_delivery, message_bus, workers

    project = create_project(tmp_path, "project", name="Project › area")
    origin = {"ref": build_owner_message_ref(chat_id=1, client_message_id="request", ts="2026-10-08T10:00:00Z", text="Do it"),
              "text": "Do it"}
    bind_task_to_project(tmp_path, "root", project["id"], project["chat_id"], origin=origin)
    task = {"id": "root", "title": "**Check › boundary**", "project_id": project["id"]}
    write_task_result(tmp_path, "root", "running", title=task["title"])
    queued, live = [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=queued.append))
    monkeypatch.setattr(events_chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    if kind == "project_started":
        assert announce_project_started(tmp_path, project, "root", task=task)
    else:
        assert enqueue_project_handoff(tmp_path, "root") == "durable"
    event = queued[0]
    assert event["progress_meta"]["task_name"] == "Check › boundary"
    expected_text = "Project › area › Check › boundary" + (" · Started" if kind == "project_started" else "")
    assert event["text"] == expected_text
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, send_with_budget=message_bus.send_with_budget,
                          append_jsonl=lambda *_a, **_k: None)
    events._handle_send_message(event, ctx)
    events._handle_send_message(event, ctx)
    assert len(live) == 1  # delivery id and event provenance keep their original meaning
    saved = json.loads((tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8"))
    history = next(row for row in _history(tmp_path) if row.get("system_type") == kind)
    for row in (live[0], saved, history):
        assert row["task_name"] == "Check › boundary"
        assert row["target_label"] == "Project › area › Check › boundary"
        assert row["task_id"] == "root" and row["project_id"] == "project"
    assert live[0]["system_type"] == saved["type"] == history["system_type"] == kind
    assert live[0]["content"] == saved["text"] == history["text"] == expected_text


@pytest.mark.parametrize("fields,expected", [
    ({"title": "**Current › title**", "suggested_name": "Other title"}, "Current › title"),
    ({"suggested_name": "**Suggested › title**"}, "Suggested › title"),
    ({}, "Task"),
])
def test_legacy_work_name_uses_cached_structured_result_and_preserves_recorded_name(tmp_path, activity_queue, monkeypatch, fields, expected):
    from ouroboros import task_status

    write_task_result(tmp_path, "root", "completed", **fields)
    rows = [
        {"ts": f"2026-10-08T10:00:0{index}Z", "direction": "system", "chat_id": 1,
         "type": kind, "task_id": "root", "project_id": "project", "project_name": "Project",
         "target_label": "Project › Legacy prose is not a task-name field", "text": "Recorded event",
         **({"task_name": "Name recorded at creation"} if index == 3 else {})}
        for index, kind in enumerate(["project_started", "project_handoff", "project_started"], 1)
    ]
    path = tmp_path / "logs" / "chat.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    before = path.read_bytes()
    original = task_status.load_effective_task_result
    calls = []

    def counted(root, task_id, **kwargs):
        calls.append(task_id)
        return original(root, task_id, **kwargs)

    monkeypatch.setattr(task_status, "load_effective_task_result", counted)
    history = [row for row in _history(tmp_path) if row.get("system_type") in {"project_started", "project_handoff"}]
    assert [row["task_name"] for row in history] == [expected, expected, "Name recorded at creation"]
    assert all(row["text"] == "Recorded event" for row in history)
    assert calls == ["root"]
    assert path.read_bytes() == before
