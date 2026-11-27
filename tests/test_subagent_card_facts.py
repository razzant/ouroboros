"""Actual live and history frames feed the card's shared Python/JS fixture."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS, requested_executor_route, subagent_message_meta
from ouroboros.subagent_runtime import cognitive_route_billing_basis
from tests._cases import case_folders

CASES = json.loads((Path(__file__).parent / "fixtures/subagent_card_facts.json").read_text())
COORDINATOR_FIELDS = {"parent_cognitive_route", "coordinator_route", "coordinator_billing",
                      "coordinator_provenance", "use_local_model"}


def test_producer_metadata_carries_the_card_facts(subtests):
    for case in CASES:
        with subtests.test(case=case):
            from ouroboros.supervisor_route import cognitive_route_provenance

            frame = subagent_message_meta(case["task"], task_id="child", event="running")
            expected = {key: "" for key in ("subagent_task_id", "root_task_id", "parent_task_id", "delegation_role",
                "subagent_role", "write_surface", "task_group_id", "model_lane", "effective_model_lane", "model",
                "executor_route", "effort_level", "effort_requested", "effort_source")}
            expected.update(subagent_task_id="child", delegation_role="subagent", subagent_event="running", **case["facts"])
            assert frame == expected
            assert set(frame) <= set(SUBAGENT_MESSAGE_FIELDS), "speech persistence must retain the same facts"
            assert subagent_message_meta({"model": "root-model"}) == {}
            assert (frame.get("coordinator_provenance") or cognitive_route_provenance(
                frame.get("parent_cognitive_route"))) == case["provenance"]


def test_history_keeps_only_recorded_card_facts_without_task_result(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for recorded in [True, False]:
        for case in CASES:
            tmp_path = next(case_paths)
            with subtests.test(recorded=recorded, case=case):
                from ouroboros.gateway.contracts import ChatOutbound
                from ouroboros.gateway.history import _load_terminal_result, make_chat_history_endpoint

                frame = subagent_message_meta({**case["task"], "root_task_id": "parent", "parent_task_id": "parent"},
                                              task_id="child", event="running")
                facts = case["facts"] if recorded else {k: v for k, v in case["facts"].items() if k not in COORDINATOR_FIELDS}
                if not recorded:
                    frame = {k: v for k, v in frame.items() if k not in COORDINATOR_FIELDS}
                assert set(frame) <= get_type_hints(ChatOutbound).keys()
                unrelated = {"raw_prompt": "private", "coordinator_account": "not a card fact", "chosen_by": "delegator",
                             "source_role": "fallback", "selected_supervisor_snapshot": {"handle": "unrelated"}}
                logs = tmp_path / "logs"
                logs.mkdir()
                (logs / "chat.jsonl").write_text("", encoding="utf-8")
                rows = [{"ts": "2026-10-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
                        {"ts": "2026-10-22T00:00:01Z", "task_id": "child", "content": "child working", **frame, **unrelated}]
                (logs / "progress.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                assert _load_terminal_result(tmp_path, "child", {}) == {"_history_result_absent": True}
                response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
                assert response.status_code == 200
                replay = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "child")
                assert replay["is_progress"] and {key: replay[key] for key in frame} == frame
                assert {key: replay[key] for key in facts} == facts, "the web fixture is the actual replayed card input"
                assert not unrelated.keys() & replay.keys(), "replay must not broaden its metadata whitelist"
                if not recorded:
                    assert not COORDINATOR_FIELDS & replay.keys(), "replay cannot reconstruct absent facts"


def test_history_names_a_childs_selection_from_its_task_for_rows_of_the_same_attempt(tmp_path):
    """A row recorded before the selection rode the frames takes it from its task's display view on replay:
    the queued, running and terminal card name the model the row selected, its account pin never."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result

    route = {"kind": "agent_session", "target_id": "codex=gpt-5.5"}
    seed_result(tmp_path, "child", {"task_id": "child", "status": "running", "delegation_role": "subagent",
                                    "parent_task_id": "parent", "root_task_id": "parent",
                                    "configured_subagent": {"schema": 1, "selected_subagent_id": "row", "config_fingerprint": "fp",
                                                            "route": {**route, "credential_profile_id": "work-account"}}})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
            {"ts": "2026-11-22T00:00:01Z", "task_id": "child", "content": "child working", "subagent_event": "running",
             "subagent_task_id": "child", "parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"}]
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    replay = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "child")
    assert replay["requested_executor_route"] == route


def test_history_and_display_view_omit_local_api_connection_name(tmp_path):
    """Bounded display views (top-level and metadata) and history replay (hydrated and legacy saved rows)
    drop local API connection names while keeping the authoritative stored route intact for internal routing."""
    from ouroboros import task_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.project_registry_recovery import initialize_new_root
    from tests._store_shared import read_result, seed_result

    initialize_new_root(tmp_path)  # A healthy Main already has its empty Project registry before the history is written.
    raw_route = {"kind": "api_model", "target_id": "openai@demo::model-a"}
    clean_route = {"kind": "api_model", "target_id": "openai::model-a"}
    snapshot = {"schema": 1, "selected_subagent_id": "row", "config_fingerprint": "fp", "route": raw_route}
    seed_result(tmp_path, "child-api", {
        "task_id": "child-api", "status": "running", "delegation_role": "subagent",
        "parent_task_id": "parent", "root_task_id": "parent", "configured_subagent": snapshot,
    })
    seed_result(tmp_path, "child-meta", {
        "task_id": "child-meta", "status": "running", "delegation_role": "subagent",
        "parent_task_id": "parent", "root_task_id": "parent", "metadata": {"configured_subagent": snapshot},
    })
    assert read_result(tmp_path, "child-api")["configured_subagent"]["route"] == raw_route
    assert read_result(tmp_path, "child-meta")["metadata"]["configured_subagent"]["route"] == raw_route
    views = {tid: json.loads(text) for tid, text in task_store.display_documents(
        tmp_path, ["child-api", "child-meta"]).items()}
    assert views["child-api"]["configured_subagent"] == {"route": clean_route}
    assert views["child-meta"]["metadata"]["configured_subagent"] == {"route": clean_route}

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    chat_rows = [
        {"ts": "2026-11-22T00:00:02Z", "direction": "out", "chat_id": 1, "task_id": "legacy-chat",
         "text": "legacy answer", "delegation_role": "subagent", "parent_task_id": "parent",
         "root_task_id": "parent", "requested_executor_route": raw_route},
    ]
    progress_rows = [
        {"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
        {"ts": "2026-11-22T00:00:01Z", "task_id": "child-api", "content": "child hydrated",
         "subagent_event": "running", "subagent_task_id": "child-api", "parent_task_id": "parent",
         "root_task_id": "parent", "delegation_role": "subagent"},
        {"ts": "2026-11-22T00:00:03Z", "task_id": "legacy-progress", "content": "legacy progress",
         "subagent_event": "scheduled", "subagent_task_id": "legacy-progress", "parent_task_id": "parent",
         "root_task_id": "parent", "delegation_role": "subagent",
         "requested_executor_route": {"kind": "api_model", "target_id": "openai@demo::model-a (local)"}},
        {"ts": "2026-11-22T00:00:04Z", "task_id": "legacy-bad", "content": "legacy malformed",
         "subagent_event": "scheduled", "subagent_task_id": "legacy-bad", "parent_task_id": "parent",
         "root_task_id": "parent", "delegation_role": "subagent",
         "requested_executor_route": {"kind": "api_model", "target_id": "openai@demo::"}},
    ]
    (logs / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in chat_rows), encoding="utf-8")
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in progress_rows), encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    assert response.status_code == 200
    messages = {row["task_id"]: row for row in json.loads(response.body)["messages"] if row.get("task_id")}
    assert messages["child-api"]["requested_executor_route"] == clean_route
    assert messages["legacy-chat"]["requested_executor_route"] == clean_route
    assert messages["legacy-progress"]["requested_executor_route"] == {"kind": "api_model", "target_id": "openai::model-a (local)"}
    assert "requested_executor_route" not in messages["legacy-bad"]
    assert "demo" not in response.body.decode("utf-8")


API_ROUTE_CASES = json.loads((Path(__file__).parent / "fixtures/api_route_display.json").read_text())


@pytest.mark.parametrize("case", API_ROUTE_CASES, ids=lambda case: case["raw"])
def test_api_route_display_cases_the_web_card_reads_too(case):
    """The shared table web/tests/review_truth.test.js reads as well: both languages show a named connection's model
    without its local name, and a name the resolver refuses selects nothing, so no model is named for it."""
    raw = {"kind": "api_model", "target_id": case["raw"]}
    want = {"kind": "api_model", "target_id": case["shown"]} if case["shown"] is not None else {}
    assert requested_executor_route({"configured_subagent": {"route": raw}}) == want
    assert requested_executor_route({"requested_executor_route": raw}) == want
    meta = subagent_message_meta({"delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent",
                                  "configured_subagent": {"route": raw}}, task_id="child", event="scheduled")
    assert meta.get("requested_executor_route", {}) == want
    # The route grammar the web readers share (harness_presentation.apiRouteParts): its head's validity and the added key.
    from ouroboros.provider_models import connection_id, parse_model_route

    parts = parse_model_route(case["raw"].removesuffix(" (local)"))
    assert parts.valid is case["valid"]
    named = parts.valid and parts.kind != "local" and parts.connection
    assert (connection_id(parts.kind, parts.connection) if named and not case["raw"].endswith(" (local)") else "") == case["connection"]


def test_an_empty_row_snapshot_yields_to_the_saved_selection_and_a_full_one_keeps_priority():
    """An old record's empty snapshot names no route, so the selection it saved stands; a snapshot that names one wins."""
    saved = {"kind": "api_model", "target_id": "openai::saved"}
    session = {"kind": "agent_session", "target_id": "codex=gpt-5.5"}
    for snapshot in ({}, {"schema": 1}, {"route": {}}, None):
        record = {"configured_subagent": snapshot, "requested_executor_route": saved}
        assert requested_executor_route(record) == saved, snapshot
        meta = subagent_message_meta({"delegation_role": "subagent", "metadata": record}, task_id="child")
        assert meta["requested_executor_route"] == saved, snapshot
    assert requested_executor_route({"configured_subagent": {"route": session}, "requested_executor_route": saved}) == session
    unusable = {"route": {"kind": "shell", "target_id": "x"}}  # a snapshot that names a route is never papered over
    assert requested_executor_route({"configured_subagent": unusable, "requested_executor_route": saved}) == {}


def test_history_keeps_the_selection_an_old_record_saved_beside_an_empty_snapshot(tmp_path):
    """An old record's empty ``configured_subagent`` names no route: its replayed card keeps the model it saved,
    shown without the local connection name even when that copy was saved whole."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result

    shown = {"kind": "api_model", "target_id": "openai::model-a"}
    lineage = {"delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent"}
    saved = {"old-shown": shown, "old-raw": {"kind": "api_model", "target_id": "openai@demo::model-a"}}
    for task_id, route in saved.items():
        seed_result(tmp_path, task_id, {"task_id": task_id, "status": "running", **lineage,
                                        "configured_subagent": {}, "requested_executor_route": route})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "parent working"},
            *({"ts": f"2026-11-22T00:00:0{n}Z", "task_id": task_id, "content": "child working", "subagent_event": "running",
               "subagent_task_id": task_id, **lineage} for n, task_id in enumerate(saved, 1))]
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    replay = {row["task_id"]: row.get("requested_executor_route") for row in json.loads(response.body)["messages"]
              if row.get("task_id") in saved}
    assert replay == {"old-shown": shown, "old-raw": shown}
    assert "demo" not in response.body.decode("utf-8")


SELECTED = {"kind": "agent_session", "target_id": "codex=gpt-5.5"}
SNAPSHOT = {"schema": 1, "selected_subagent_id": "row", "config_fingerprint": "fp",
            "route": {**SELECTED, "credential_profile_id": "work-account"}}
COGNITION = {"model": "openai/gpt-6-astra", "use_local_model": False, "chosen_by": "inherited"}
CHILD_LINEAGE = {"delegation_role": "subagent", "parent_task_id": "parent", "root_task_id": "parent"}


def test_history_admission_notices_keep_only_their_own_selection_and_the_parent_anchor(tmp_path, monkeypatch):
    from ouroboros import task_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result

    evidence = {"delegated_runs_started": 1}
    seed_result(tmp_path, "parent", {"status": "completed", "delegation_role": "subagent", "chat_id": 1,
                                    "parent_task_id": "root", "root_task_id": "root", "model": "parent-model",
                                    "executor_route": "claude=claude-opus-5-5", "coordinator_route": "parent-model",
                                    "subagent_envelope": {"execution_evidence": evidence, "actual_substrate": "harness_used"}})
    child_ids = ["refused-a", "refused-b"]
    for tid in child_ids:
        # Even the same child id and lineage are not a binding to the refused request.
        seed_result(tmp_path, tid, {"status": "completed", "_attempt": 1, "delegation_role": "subagent", "chat_id": 1,
                                   "parent_task_id": "parent", "root_task_id": "root", "configured_subagent": {
                                       "route": {"kind": "agent_session", "target_id": "claude=later-child"}},
                                   "executor_route": "claude=later-child", "model": "later-nanny",
                                   "subagent_envelope": {"execution_evidence": evidence}})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "Parent working."}]
    rows += [{"ts": f"2026-11-22T00:00:0{i}Z", "task_id": "parent", "subagent_task_id": tid,
              "parent_task_id": "parent", "root_task_id": "root", "delegation_role": "subagent",
              "role": "system", "system_type": "subagent_admission_notice", "content": "Request refused.",
              "subagent_event": "rejected", "accepted": False, "status": "failed"} for i, tid in enumerate(child_ids, 1)]
    rows[1].update(requested_executor_route=SELECTED, parent_cognitive_route=COGNITION)
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    batches = []
    display_documents = task_store.display_documents

    def record_batch(root, task_ids, *args, **kwargs):
        batches.append(set(task_ids))
        return display_documents(root, task_ids, *args, **kwargs)

    monkeypatch.setattr(task_store, "display_documents", record_batch)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    assert response.status_code == 200
    replay = json.loads(response.body)["messages"]
    parent = next(row for row in replay if row["text"] == "Parent working.")
    assert parent["executor_route"] == "claude=claude-opus-5-5"
    assert parent["task_terminal_status"] == "completed" and parent["execution_evidence"] == evidence
    notices = [row for row in replay if row.get("system_type") == "subagent_admission_notice"]
    assert [row["subagent_task_id"] for row in notices] == child_ids
    assert notices[0]["requested_executor_route"] == SELECTED and notices[0]["parent_cognitive_route"] == COGNITION
    assert "requested_executor_route" not in notices[1] and "parent_cognitive_route" not in notices[1]
    for notice in notices:
        assert notice["task_id"] == "parent" and notice["status"] == "failed"
        assert not {"executor_route", "model", "coordinator_route", "coordinator_billing", "task_terminal_status",
                    "execution_evidence", "actual_substrate", "tool_evidence"} & notice.keys()
    assert all(not batch.intersection(child_ids) for batch in batches)


@pytest.mark.parametrize(("saved_facts", "expected"), [
    ({"configured_subagent": SNAPSHOT, "requested_executor_route": {"kind": "api_model", "target_id": "openai::stale"}}, SELECTED),
    ({"metadata": {"configured_subagent": SNAPSHOT}}, SELECTED),
    ({"metadata": {"configured_subagent": SNAPSHOT}, "requested_executor_route": {"kind": "api_model", "target_id": "openai::stale"}}, SELECTED),
    ({"configured_subagent": {}, "metadata": {"requested_executor_route": {"kind": "api_model", "target_id": "openai@work::legacy"}}},
     {"kind": "api_model", "target_id": "openai::legacy"}),
    ({"configured_subagent": {"route": {"kind": "api_model", "target_id": "openai@Bad_Name::unusable"}},
      "metadata": {"configured_subagent": SNAPSHOT}, "requested_executor_route": SELECTED}, None),
    # A child whose record keeps its lineage only in legacy ``metadata`` is still a child to its detail, as to history.
    ({"metadata": {"configured_subagent": SNAPSHOT, **CHILD_LINEAGE}}, SELECTED),
])
def test_cancelled_child_uses_the_same_frozen_selection_in_history_and_http_detail(tmp_path, saved_facts, expected):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._store_shared import seed_result
    from tests._task_file_http import task_file_client

    lineage = CHILD_LINEAGE
    top_level = {} if "delegation_role" in saved_facts.get("metadata", {}) else lineage
    seed_result(tmp_path, "child-missed", {"status": "cancelled", "chat_id": 1, **top_level, **saved_facts})
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    rows = [{"ts": "2026-11-22T00:00:00Z", "task_id": "parent", "content": "Parent working."},
            {"ts": "2026-11-22T00:00:01Z", "task_id": "child-missed", "content": "Queued.",
             "subagent_event": "scheduled", "subagent_task_id": "child-missed", **lineage}]
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    row = next(row for row in json.loads(response.body)["messages"] if row["text"] == "Queued.")
    response = task_file_client(tmp_path).get("/api/tasks/child-missed")
    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["status"] == row["task_terminal_status"] == "cancelled"
    assert row.get("requested_executor_route") == expected
    assert detail.get("requested_executor_route") == expected
    for frame in (row, detail):
        assert not any(frame.get(key) for key in ("executor_route", "model", "execution_evidence", "actual_substrate"))


def test_a_root_detail_names_no_helper_selection_and_root_admission_strips_one(tmp_path):
    """Only a helper has a frozen executor selection: a root's detail names none whatever its row carries, and
    ``/api/tasks`` drops a caller's ``metadata.requested_executor_route`` like the other host-owned keys."""
    from ouroboros.gateway.tasks import _RESERVED_METADATA_KEYS
    from tests._store_shared import seed_result
    from tests._task_file_http import task_file_client

    seed_result(tmp_path, "root-claim", {"status": "completed", "chat_id": 1, "delegation_role": "root",
                                         "requested_executor_route": SELECTED,
                                         "metadata": {"requested_executor_route": SELECTED, "configured_subagent": SNAPSHOT}})
    # An explicit top-level role outranks a stale legacy ``metadata`` lineage.
    seed_result(tmp_path, "root-legacy", {"status": "cancelled", "chat_id": 1, "delegation_role": "root", "metadata": {
        "delegation_role": "subagent", "parent_task_id": "other", "configured_subagent": SNAPSHOT}})
    for task_id in ("root-claim", "root-legacy"):
        response = task_file_client(tmp_path).get(f"/api/tasks/{task_id}")
        assert response.status_code == 200, response.text
        assert "requested_executor_route" not in response.json(), task_id
    assert "requested_executor_route" in _RESERVED_METADATA_KEYS


@pytest.mark.parametrize("producer", ["invalid_depth", "project_state", "preflight_unreadable", "preparation_unknown",
                                     "preparation_live", "preparation_recovered", "preparation_write_failed",
                                     "preparation_recovered_write_failed", "preparation_unmarked_write_failed"])
def test_every_rejection_producer_names_the_refused_childs_selection_live_and_on_replay(tmp_path, monkeypatch, producer):
    """Admission and preparation refusals retain the request even when the result cannot answer."""
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.task_results import STATUS_REQUESTED, load_task_result, write_task_result
    from supervisor import events, events_subagent_admission, message_bus, queue, task_admission
    from supervisor import workspace_preparation as prep

    (tmp_path / "logs").mkdir()
    live = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(events_subagent_admission, "_bound_project_chat_id", lambda *_: None)
    for name, value in (("DRIVE_ROOT", tmp_path), ("PENDING", []), ("RUNNING", {}), ("ADMISSION_RESERVATIONS", {})):
        monkeypatch.setattr(queue, name, value)
    write_task_result(tmp_path, "parent", "running", chat_id=1, delegation_role="root", root_task_id="parent")
    tid, token = f"refused-{producer}", "claim-token"
    # The producer's REQUESTED row and event (control_scheduling): both carry the row's whole snapshot.
    facts = {"objective": f"check {tid}", "expected_output": "a note", "role": "checker", "parent_task_id": "parent",
             "root_task_id": "parent", "delegation_role": "subagent", "chat_id": 1, "configured_subagent": SNAPSHOT,
             "parent_cognitive_route": COGNITION}
    marker = {"token": token, "phase": prep.QUEUED, "generation": "gen", "owner": {}, "surface": "",
              "claimed_at": "2026-11-22T00:00:00Z", "updated_at": "2026-11-22T00:00:00Z"}
    write_task_result(tmp_path, tid, STATUS_REQUESTED, description=facts["objective"],
                      **({prep.MARKER: marker} if producer.startswith("preparation") else {}), **facts)
    event = {**facts, "type": "schedule_subagent", "task_id": tid, "description": facts["objective"], "depth": 1}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={}, WORKERS={0: SimpleNamespace(busy_task_id=None)},
                          send_with_budget=message_bus.send_with_budget, load_state=lambda: {"owner_chat_id": 1},
                          persist_queue_snapshot=lambda reason="": None, enqueue_task=lambda task: pytest.fail("enqueued"))
    def inaccessible(*_args, **_kwargs):
        raise OSError("task authority unavailable")

    unrecorded = producer in {"preflight_unreadable", "preparation_unknown", "preparation_write_failed",
                              "preparation_recovered_write_failed", "preparation_unmarked_write_failed"}
    if producer == "invalid_depth":
        events._handle_schedule_task({**event, "depth": "not-a-depth"}, ctx)
    elif producer == "project_state":
        from ouroboros import projects_registry

        def unreadable(*_args, **_kwargs):
            raise RuntimeError("registry unreadable")

        with monkeypatch.context() as fault:  # only admission's read fails; the replay below reads the registry
            fault.setattr(projects_registry, "project_binding_for_task", unreadable)
            events._handle_schedule_task({**event, "project_id": "proj-1"}, ctx)
    elif producer == "preparation_unmarked_write_failed":
        monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", inaccessible)
        monkeypatch.setattr(prep, "_request_ended", lambda *_: prep.REQUEST_ORPHANED)
        assert prep._recover_unmarked(tmp_path, load_task_result(tmp_path, tid, strict=True)) == "retained"
    elif producer == "preflight_unreadable":
        read = task_admission.load_task_result
        with monkeypatch.context() as fault:  # the event's first step reads this id's authority itself (#4317)
            fault.setattr(task_admission, "subagent_schedule_owned", inaccessible)
            fault.setattr(task_admission, "load_task_result", lambda root, task_id, *args, **kwargs: (
                inaccessible() if task_id == tid else read(root, task_id, *args, **kwargs)))
            events._handle_schedule_task(event, ctx)
    else:
        if "recovered" not in producer:
            claim = prep.PreparationClaim(task_id=tid, ctx=ctx, token=token, generation=None, generation_token="gen",
                                          owner={}, event=event, route={"chat_id": 1, "delegation_role": "subagent",
                                          "parent_id": "parent", "role": "checker", "root_task_id": "parent"})
        else:  # after a restart: only the durable row and its marker, no event
            row = load_task_result(tmp_path, tid, strict=True)
            claim = prep.PreparationClaim.recovered(tmp_path, row, row[prep.MARKER])
        if producer == "preparation_unknown":
            claim.disclose_unknown()
        else:
            if unrecorded:
                monkeypatch.setattr(task_admission, "persist_never_admitted_refusal", inaccessible)
            assert claim.refuse("Subagent not scheduled: its workspace preparation failed: boom",
                                "admission_preparation_failed") is (not unrecorded)
    stored = load_task_result(tmp_path, tid, strict=True)
    assert stored["status"] == (STATUS_REQUESTED if unrecorded else "failed")
    assert stored["configured_subagent"] == SNAPSHOT and stored["parent_cognitive_route"] == COGNITION
    notices = [frame for frame in live if frame.get("subagent_event") == "rejected" and frame.get("subagent_task_id") == tid]
    assert len(notices) == 1, live
    assert notices[0]["requested_executor_route"] == SELECTED
    assert notices[0]["parent_cognitive_route"] == COGNITION
    assert "work-account" not in json.dumps(notices[0]), "the row's account pin never travels with the selection"
    if unrecorded:  # The refusal's own progress row must suffice even while its exact authority is unreadable.
        from ouroboros.task_reads import TaskReads

        effective = TaskReads.effective
        monkeypatch.setattr(TaskReads, "effective", lambda reads, task_id: (
            inaccessible() if task_id == tid else effective(reads, task_id)))
        if producer == "preparation_unmarked_write_failed":
            assert notices[0]["error"] == prep._DETAILS[prep.REQUEST_ORPHANED]
        else:
            assert ("unreadable" in notices[0]["error"] or "scheduled_result_authority_unknown" in notices[0]["error"])
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    replays = [row for row in json.loads(response.body)["messages"] if row.get("subagent_task_id") == tid
               and row.get("subagent_event") == "rejected"]
    assert [row.get("requested_executor_route") for row in replays] == [SELECTED]
    assert replays[0]["parent_cognitive_route"] == COGNITION


@pytest.mark.parametrize(("model", "local", "cost", "billing", "price"), [
    ("openai/gpt-5", False, None, "unknown", None),
    ("openai/gpt-5", False, .125, "measured", .001),
    ("future-provider/reviewer", False, 0.0, "measured", None),
    ("local-model", True, None, "unknown", .001),
    ("claudexor::codex=gpt-5", False, None, "subscription", None),
])
def test_billing_basis_ignores_price_cache_and_requires_observed_cost(monkeypatch, price, model, local, cost, billing):
    consulted = []
    def estimate(*_args, **_kwargs):
        consulted.append(True)
        return price
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", estimate)
    route, actual = cognitive_route_billing_basis(model, use_local=local, cost=cost)
    assert route == model + (" (local)" if local else "") and actual == billing
    # A cumulative total names no accepted call: whatever it sums, it labels no route.
    ctx = SimpleNamespace(_accumulated_usage={"cost": cost, "cost_final": True,
                                              "_last_llm_call_meta": {"model": model}})
    assert cognitive_route_billing_basis(model, use_local=local, ctx=ctx) == (
        route, "subscription" if billing == "subscription" else "unknown")
    assert not consulted, "a billing fact cannot depend on the cache or pricing service"


def _call(usage, tmp_path, monkeypatch, model, *, cost=None, final=None, estimate=None, served="", round_idx=1,
          content="usable", use_local=False, error=None):
    """One call through the loop's own producer; ``usage`` then holds the records the billing label reads.

    The route is a synthetic object and an unpriced reply meets a stubbed tariff: no catalog is read."""
    from ouroboros.loop_llm_call import call_llm_with_retry

    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *_args, **_kwargs: estimate)

    class Route:
        def chat(self, **_kwargs):
            if error is not None:
                raise error
            return {"content": content}, {"prompt_tokens": 9, "completion_tokens": 2, "cost": cost,
                                           **({} if final is None else {"cost_final": final}),
                                           "provider": "openai-compatible", "resolved_model": served or model}

    return call_llm_with_retry(Route(), [{"role": "user", "content": "task"}], model, None, "medium", 1,
                               tmp_path / "logs", "billing-basis", round_idx, None, usage, "task", use_local,
                               attempt_cap=1)


@pytest.mark.parametrize(("cost", "final", "estimate", "billing"), [
    (0.125, True, None, "measured"),
    (0.0, True, None, "measured"),  # a legitimate reported zero
    (0.125, False, None, "unknown"),  # an amount its producer did not call final
    (None, None, None, "unknown"),  # unpriced
    (None, None, 0.001, "unknown"),  # a tariff estimate
    ("bogus", True, None, "unknown"),  # an invalid amount
])
def test_the_label_is_the_last_accepted_calls_own_money_whatever_the_history(
        tmp_path, monkeypatch, cost, final, estimate, billing):
    """The nanny route says measured only for money its own last accepted call reported.

    The history before it would decide the other way if the cumulative labelled the route: an unpriced call
    on another route before reported money, exact money on this route before a call that reports none."""
    usage = {}
    if billing == "measured":
        _call(usage, tmp_path, monkeypatch, "other/route")
        assert (usage["cost_partial"], usage["cost_final"]) == (True, False), "the cumulative is unknown from here on"
    else:
        _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.125, final=True)
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=cost, final=final, estimate=estimate, round_idx=2)
    ctx = SimpleNamespace(_accumulated_usage=usage)
    consulted = []
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *_args, **_kwargs: consulted.append(True))
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", billing)
    assert cognitive_route_billing_basis("other/route", ctx=ctx) == ("other/route", "unknown")
    assert not consulted, "the label reads the recorded call, never a price"


@pytest.mark.parametrize(("accepted", "local", "asked", "asked_local"), [
    ("openai/gpt-5", False, "openai/gpt-5.1", False),  # another model
    ("openai/gpt-5", False, "openai::gpt-5", False),  # another provider's spelling of the same served name
    ("openai/gpt-5", False, "served/gpt-5", False),  # the served name itself is not the route that was asked
    ("qwen/qwen3-coder", False, "qwen/qwen3-coder", True),  # the same id on another locality
    ("qwen/qwen3-coder", True, "qwen/qwen3-coder", False),
])
def test_an_accepted_call_labels_only_the_route_it_was_requested_on(
        tmp_path, monkeypatch, accepted, local, asked, asked_local):
    """Exact money names its own route: the configured id and locality its call record requested, no alias."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, accepted, cost=0.125, final=True, served="served/gpt-5", use_local=local)
    ctx = SimpleNamespace(_accumulated_usage=usage)
    label = accepted + (" (local)" if local else "")
    assert cognitive_route_billing_basis(accepted, use_local=local, ctx=ctx) == (label, "measured")
    assert cognitive_route_billing_basis(label, ctx=ctx) == (label, "measured"), "a display suffix is locality"
    assert cognitive_route_billing_basis(asked, use_local=asked_local, ctx=ctx)[1] == "unknown"


def test_a_reply_without_a_usable_answer_accepts_nothing_for_its_route(tmp_path, monkeypatch):
    """Such a reply moves the last call record and writes no accepted call: neither route is then labelled."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.125, final=True)
    accepted = dict(usage["_last_round_cache_usage"])
    reply, _cost = _call(usage, tmp_path, monkeypatch, "fallback/route", cost=0.25, final=True, round_idx=2, content="")
    assert reply is None and usage["_last_round_cache_usage"] == accepted
    assert usage["_last_llm_call_meta"]["model"] == "fallback/route" and usage["_last_llm_call_meta"]["failure_code"]
    assert usage["cost"] == 0.375 and usage.get("cost_final") is not False, (
        "the cumulative still reads as reported money")
    ctx = SimpleNamespace(_accumulated_usage=usage)
    assert cognitive_route_billing_basis("fallback/route", ctx=ctx) == ("fallback/route", "unknown")
    # The earlier accepted call is no longer the one the last call record names: unknown until the next accepted call.
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "unknown")
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.0, final=True, round_idx=3)
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "measured")


def test_a_send_that_raised_leaves_the_last_accepted_call_the_fact(tmp_path, monkeypatch):
    """No reply came back: the call record and the accepted call both stay the earlier call's."""
    usage = {}
    _call(usage, tmp_path, monkeypatch, "openai/gpt-5", cost=0.0, final=True)
    records = (dict(usage["_last_llm_call_meta"]), dict(usage["_last_round_cache_usage"]))
    reply, _cost = _call(usage, tmp_path, monkeypatch, "fallback/route", round_idx=2, error=RuntimeError("no reply"))
    assert reply is None and usage["_last_llm_failure"]["model"] == "fallback/route"
    assert (usage["_last_llm_call_meta"], usage["_last_round_cache_usage"]) == records
    ctx = SimpleNamespace(_accumulated_usage=usage)
    assert cognitive_route_billing_basis("openai/gpt-5", ctx=ctx) == ("openai/gpt-5", "measured")
    assert cognitive_route_billing_basis("fallback/route", ctx=ctx) == ("fallback/route", "unknown")


def test_explicit_cost_and_subscription_keep_their_kind():
    """The caller's own amount is taken as given; a subscription route stays a subscription."""
    assert cognitive_route_billing_basis("openai/gpt-5", cost=0.0) == ("openai/gpt-5", "measured")
    assert cognitive_route_billing_basis("openai/gpt-5", cost=None) == ("openai/gpt-5", "unknown")
    assert cognitive_route_billing_basis("qwen/qwen3-coder", use_local=True, cost=0.0) == (
        "qwen/qwen3-coder (local)", "measured")
    assert cognitive_route_billing_basis("claudexor::codex=gpt-5", cost=0.0) == (
        "claudexor::codex=gpt-5", "subscription")
    zero = SimpleNamespace(_accumulated_usage={"cost": 0.0, "cost_final": True,
                                              "_last_llm_call_meta": {"model": "claudexor::codex=gpt-5"}})
    assert cognitive_route_billing_basis("claudexor::codex=gpt-5", ctx=zero) == (
        "claudexor::codex=gpt-5", "subscription")


def test_subagent_started_stamps_own_attempt_through_live_and_history_without_fabricating_absence(tmp_path, monkeypatch):
    """Worker `subagent_started` carries its own `task_attempt` (0 valid) onto live WS and history replay, never
    takes a newer RUNNING/result/sibling row's ordinal, and `work_status` preserves `queue_attempt` 0..N."""
    import queue as pyqueue
    from ouroboros.agent import OuroborosAgent
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.state import _raw_work_facts
    from ouroboros.gateway.work_status import classify_member, roll_up_work
    from supervisor import events_chat_delivery, message_bus
    from tests._store_shared import seed_result

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    sent: list[dict] = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = sent.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)

    out_q: pyqueue.Queue = pyqueue.Queue()
    agent = object.__new__(OuroborosAgent)
    agent._event_queue = out_q
    agent._current_chat_id = 1

    base = {"delegation_role": "subagent", "parent_task_id": "root", "root_task_id": "root", "role": "reader"}
    agent._emit_subagent_started({**base, "id": "child-zero", "_attempt": 0})
    agent._emit_subagent_started({**base, "id": "child-late", "_attempt": 1})
    agent._emit_subagent_started({**base, "id": "child-unnumbered"})

    # Even if RUNNING already holds attempt 2 when attempt 1's event drains, the frame keeps attempt 1 (and unnumbered stays absent).
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "child-zero": {"task": {**base, "id": "child-zero", "_attempt": 1}, "attempt": 1},
            "child-late": {"task": {**base, "id": "child-late", "_attempt": 2}, "attempt": 2},
            "child-unnumbered": {"task": {**base, "id": "child-unnumbered", "_attempt": 2}, "attempt": 2},
        },
        send_with_budget=message_bus.send_with_budget,
    )
    while not out_q.empty():
        events_chat_delivery._handle_send_message(out_q.get_nowait(), ctx)

    live_by_task = {frame["task_id"]: frame for frame in sent if frame.get("system_type") == "subagent_started"}
    assert live_by_task["child-zero"]["task_attempt"] == 0
    assert live_by_task["child-late"]["task_attempt"] == 1
    assert "task_attempt" not in live_by_task["child-unnumbered"]

    # Seed stored results at attempt 2 + a sibling progress row at attempt 2: replay must NOT backfill child-unnumbered's attempt.
    for tid in ("child-zero", "child-late", "child-unnumbered"):
        seed_result(tmp_path, tid, {"task_id": tid, "status": "running", "task_attempt": 2, **base})
    message_bus.send_with_budget(
        1, "later progress", is_progress=True, task_id="child-unnumbered",
        progress_meta={"subagent_event": "running", "subagent_task_id": "child-unnumbered",
                       "parent_task_id": "root", "root_task_id": "root", "delegation_role": "subagent", "task_attempt": 2},
    )
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "20"})))
    assert response.status_code == 200
    replayed = {
        row["task_id"]: row for row in json.loads(response.body)["messages"]
        if row.get("system_type") == "subagent_started"
    }
    assert replayed["child-zero"]["task_attempt"] == 0
    assert replayed["child-late"]["task_attempt"] == 1
    assert "task_attempt" not in replayed["child-unnumbered"]

    # Census work projection preserves queue_attempt 0 as well as positive ordinals, and omits absent ones.
    root_member = classify_member({"activity_id": "root", "phase": "working", "live": True}, {"queue_attempt": 1})
    for raw, expected in [({"_queue_attempt": 0, "_attempt": 1}, 0), ({"_attempt": 2}, 2), ({}, None)]:
        facts = _raw_work_facts(raw)
        member = classify_member({"activity_id": "c", "phase": "working", "live": True}, facts)
        rolled = roll_up_work("root", [root_member, member], episode={"task_id": "root", "live": True, "state": root_member})
        assert rolled["members"]["states"]["c"].get("queue_attempt") == expected
