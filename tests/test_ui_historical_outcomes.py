"""Bounded history retains typed lifecycle observations without reviving results."""
import asyncio
import json
from types import SimpleNamespace


from ouroboros.gateway.history import make_chat_history_endpoint, _load_terminal_result
from ouroboros.project_dialogue import append_terminal_task_projection
from ouroboros.task_results import write_task_result, task_results_dir


def test_retained_terminal_is_history_only_until_client_checks_current_activity(tmp_path):
    task = {"id": "past-root", "chat_id": 1, "root_task_id": "past-root", "delegation_role": "root"}
    model_execution = {
        "source": "usable_solve_response", "used_model": "fallback-model",
        "requested_model": "initial-model", "used_local": False,
    }
    result = write_task_result(
        tmp_path, task["id"], "cancelled", result="Preserved work",
        model_execution=model_execution, **{k: v for k, v in task.items() if k != "id"},
    )
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "progress.jsonl").write_text(json.dumps({
        "task_id": task["id"], "chat_id": 1, "content": "Inspecting the source",
        "ts": "2026-09-06T00:00:00Z",
    }) + "\n")
    assert append_terminal_task_projection(tmp_path, task["id"], task, result, {"chat_id": 1})
    path = task_results_dir(tmp_path) / "past-root.json"
    quarantine = tmp_path / "preserved-quarantine.json"
    path.rename(quarantine)
    before = quarantine.read_bytes()
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    rows = json.loads(response.body)["messages"]
    assert len(rows) == 1  # synthetic summary text stays hidden
    assert rows[0]["is_progress"] is True and rows[0]["history_id"].startswith("progress:")
    assert "summary_kind" not in rows[0]  # no redundant hidden evidence beside its annotation
    assert rows[0]["historical_terminal"]["status"] == "cancelled"
    assert rows[0]["historical_terminal"]["model_execution"] == model_execution
    assert "task_terminal_status" not in rows[0]
    assert "outcome_axes" not in rows[0]
    assert quarantine.read_bytes() == before
    assert not path.exists()


def test_terminal_evidence_survives_when_its_narration_is_on_an_older_page(tmp_path):
    task = {"id": "archived-root", "chat_id": 1, "root_task_id": "archived-root", "delegation_role": "root"}
    result = write_task_result(tmp_path, task["id"], "cancelled", result="Preserved work",
                               **{key: value for key, value in task.items() if key != "id"})
    assert append_terminal_task_projection(tmp_path, task["id"], task, result, {"chat_id": 1})
    (task_results_dir(tmp_path) / "archived-root.json").unlink()
    archive = tmp_path / "archive"
    archive.mkdir()
    for index in range(5):
        (archive / f"progress_2026090{index + 1}T000000.jsonl").write_text(json.dumps({
            "task_id": task["id"] if index == 0 else f"newer-{index}", "chat_id": 1,
            "content": "First saved narration" if index == 0 else f"Later work {index}",
            "ts": f"2026-09-0{index + 1}T00:00:00Z",
        }) + "\n")
        # Older pages follow the conversation, so the room has older dialogue to page
        # through while its narration pages alongside (owner decision 2026-10-05).
        (archive / f"chat_2026090{index + 1}T000000.jsonl").write_text(json.dumps({
            "direction": "in", "chat_id": 1, "text": f"Older dialogue {index}",
            "ts": f"2026-09-0{index + 1}T00:00:00Z",
        }) + "\n")
    endpoint = make_chat_history_endpoint(tmp_path)

    def read(cursor=None):
        response = asyncio.run(endpoint(SimpleNamespace(query_params={
            "chat_id": "1", "n_human": "1", "n_progress": "1", **({"cursor": cursor} if cursor else {}),
        })))
        assert response.status_code == 200
        return json.loads(response.body)

    first = read()
    [evidence] = [row for row in first["messages"] if row.get("task_id") == task["id"]]
    assert evidence["text"] == "" and evidence["summary_kind"] == "terminal_root_projection"
    assert evidence["historical_terminal"]["status"] == "cancelled"
    assert not {"task_terminal_status", "outcome_axes", "review_projection"} & evidence.keys()
    later = []
    page = first
    while page["has_more"]:
        page = read(page["next_cursor"])
        later.extend(page["messages"])
    [narration] = [row for row in later if row.get("task_id") == task["id"]]
    assert narration["text"] == "First saved narration"
    assert narration["history_id"] != evidence["history_id"]
    assert "historical_terminal" not in narration  # replay joins the earlier evidence
    assert "task_terminal_status" not in narration


def test_unreadable_effective_result_is_not_proven_absent(tmp_path, monkeypatch):
    from ouroboros import task_status
    path = task_results_dir(tmp_path) / "unreadable.json"
    path.write_text("malformed")
    monkeypatch.setattr(task_status, "load_effective_task_result", lambda *_a, **_k: None)
    assert _load_terminal_result(tmp_path, "unreadable", {}) == {}
    assert _load_terminal_result(tmp_path, "absent", {}) == {"_history_result_absent": True}


def test_partial_activity_keeps_other_positive_source(tmp_path, monkeypatch):
    from ouroboros.gateway import state
    from supervisor import queue, active_activity

    monkeypatch.setattr(queue, "PENDING", [{"id": "queued", "chat_id": 1}])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: (_ for _ in ()).throw(OSError("unavailable")))
    availability = {"complete": True}
    direct = state._direct_turns_snapshot_safe(availability=availability)
    rows = state._chat_activities_snapshot_safe(tmp_path, {}, direct_turns=direct, availability=availability)
    assert availability["complete"] is False
    assert any(row["activity_id"] == "queued" for row in rows)


def test_routing_destination_address_survives_annotation_live_and_history(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.project_dialogue import latest_chat_annotations
    from supervisor import message_bus
    from supervisor.events_project_routing import _emit_routing_receipt

    project = create_project(tmp_path, "destination", name="Destination")
    bind_task_to_project(tmp_path, "destination-task", project["id"], project["chat_id"], origin={"absent": "system"})
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge({})
    frames = []
    bridge._broadcast_fn = frames.append
    event = {"client_message_id": "owner-row", "routing_token": "token", "chat_id": 1}
    receipt = _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge), event,
                                    action="promote_chat_to_task", target="destination-task", status="scheduled")
    assert receipt["persisted"] is True
    annotation = latest_chat_annotations(tmp_path)["owner-row"]
    frame = next(row for row in frames if row.get("type") == "message_annotation")
    message_bus.log_chat("in", 1, 1, "Work in the project", client_message_id="owner-row", drive_root=tmp_path)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "1"})))
    replay = next(row for row in json.loads(response.body)["messages"] if row.get("client_message_id") == "owner-row")
    for row in (annotation, frame, replay["chat_annotation"]):
        assert row["project_id"] == "destination"
        assert row["project_chat_id"] == project["chat_id"]
        assert row["target"] == "destination-task"


def test_existing_malformed_bindings_keep_activity_coverage_unknown(tmp_path):
    from ouroboros.gateway.state import _task_bindings_safe

    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "project_task_bindings.json").write_text("malformed")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))
    availability = {"complete": True}
    assert _task_bindings_safe(request, availability=availability) == {}
    assert availability["complete"] is False


def test_task_get_distinguishes_unreadable_result_from_absence(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    path = task_results_dir(tmp_path) / "unreadable.json"
    path.write_text("preserved unreadable bytes")
    monkeypatch.setattr(tasks, "load_effective_task_result", lambda *_a, **_k: None)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
                              path_params={"task_id": "unreadable"})
    assert asyncio.run(tasks.api_task_get(request)).status_code == 503
    request.path_params["task_id"] = "missing"
    assert asyncio.run(tasks.api_task_get(request)).status_code == 404
    assert path.read_text() == "preserved unreadable bytes"


def test_history_conversion_does_not_need_or_recreate_a_task_result(tmp_path, monkeypatch):
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import project_binding_for_task
    from supervisor import queue

    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    async def body():
        return {"task_id": "historical-only", "name": "Retained history"}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)), json=body)
    response = asyncio.run(api_project_from_task(request))
    assert response.status_code == 200
    assert project_binding_for_task(tmp_path, "historical-only")
    assert not (task_results_dir(tmp_path) / "historical-only.json").exists()
    assert queue.RUNNING == {} and queue.PENDING == []
