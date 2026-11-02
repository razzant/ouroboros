"""Root result pointers remain readable after retiring the spawn predecessor selector."""

import types
import pytest

def _host_ctx(tmp_path, *, pending=None, running=None):
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, PENDING=list(pending or []), RUNNING=dict(running or {}),
    )


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """`_durable_project_of_request` reads the registry through config.DATA_DIR."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)


def test_only_a_root_finalization_moves_the_projects_pointer(tmp_path):
    """The pointer answers "continue from here" for the ROOM; a child that
    finalizes later must not move it onto work no owner ever addressed."""
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.tools.project_journal import record_task_finalization

    create_project(tmp_path, "racer", name="Racer")
    record_task_finalization(
        "racer", {"id": "racer-root", "root_task_id": "racer-root"},
        objective="the root", kind="task", exec_status="completed", drive_root=tmp_path,
    )
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"

    record_task_finalization(
        "racer", {"id": "racer-child", "parent_task_id": "racer-root",
                  "root_task_id": "racer-root", "delegation_role": "subagent"},
        objective="helper work", kind="task", exec_status="completed", drive_root=tmp_path,
    )
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"

    # The door calls a row a child by its parent OR by its role; so does the stamp.
    record_task_finalization(
        "racer", {"id": "racer-role-only", "root_task_id": "racer-root", "delegation_role": "subagent"},
        objective="helper with no parent field", kind="task", exec_status="completed", drive_root=tmp_path,
    )
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"


def test_the_self_heal_scan_never_offers_or_stamps_a_child(tmp_path):
    """The lookup's fallback scan is the pointer's SECOND writer: with no pointer
    yet and a child as the project's newest result, it answers with the newest ROOT
    and stamps that, never the child the hint does not offer."""
    import server
    from ouroboros.projects_registry import create_project, get_project
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "racer-root", "completed", project_id="racer", result="root answer",
                      ts="2026-09-21T10:00:00Z")
    write_task_result(tmp_path, "racer-child", "completed", project_id="racer", result="helper answer",
                      parent_task_id="racer-root", root_task_id="racer-root", delegation_role="subagent",
                      ts="2026-09-21T11:00:00Z")  # the child is the newest row
    assert not get_project(tmp_path, "racer").get("last_task_result_id")

    row = server._latest_project_task_result(_host_ctx(tmp_path), "racer")
    assert row["task_id"] == "racer-root"
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"


def test_a_pointer_a_child_stamped_before_the_rule_heals_onto_the_root(tmp_path):
    """A pointer written before "only a root stamps it" may name a child. That is
    provably wrong (not a copy-back in flight), so the lookup answers with the root
    and repairs the pointer; a pointer naming a ROOT is served as it is."""
    import server
    from ouroboros.projects_registry import create_project, get_project, update_project
    from ouroboros.task_results import write_task_result

    create_project(tmp_path, "racer", name="Racer")
    write_task_result(tmp_path, "racer-root", "completed", project_id="racer", result="root answer")
    write_task_result(tmp_path, "racer-child", "completed", project_id="racer", result="helper answer",
                      parent_task_id="racer-root", root_task_id="racer-root", delegation_role="subagent")
    update_project(tmp_path, "racer", last_task_result_id="racer-child")

    assert server._latest_project_task_result(_host_ctx(tmp_path), "racer")["task_id"] == "racer-root"
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"

    # The quiet direction: a root pointer is one direct fetch and is left alone.
    write_task_result(tmp_path, "racer-root-2", "completed", project_id="racer", result="later root")
    update_project(tmp_path, "racer", last_task_result_id="racer-root")
    assert server._latest_project_task_result(_host_ctx(tmp_path), "racer")["task_id"] == "racer-root"
    assert get_project(tmp_path, "racer")["last_task_result_id"] == "racer-root"


def test_the_room_manifest_addresses_cancel_facts_and_has_an_honest_omission_count(tmp_path):
    import server
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "racer", name="Racer")
    write_task_result(
        tmp_path, "racer-cancelled", "cancelled", project_id="racer",
        objective="stopped work", ts="2026-08-10T00:00:01Z",
        cancel_origin={"source": "owner", "reason": "owner stopped it",
                       "requested_at": "2026-08-10T00:00:00Z"},
    )
    write_task_result(tmp_path, "racer-child", "completed", project_id="racer",
                      objective="helper", parent_task_id="racer-cancelled",
                      root_task_id="racer-cancelled", delegation_role="subagent",
                      ts="2026-08-10T00:00:02Z")
    running = {"racer-live": {"task": {"id": "racer-live", "project_id": "racer",
                                       "title": "Live", "objective": "in flight"},
                              "started_at": "2026-08-10T00:00:03Z"},
               # Another project's live root is none of this room's business.
               "other-live": {"task": {"id": "other-live", "project_id": "other",
                                       "title": "Elsewhere", "objective": "not this room"},
                              "started_at": "2026-08-10T00:00:04Z"}}
    request_cancel(tmp_path, "racer-live", source="owner", reason="stop it")

    manifest = server._decision_turn_metadata(
        _host_ctx(tmp_path, running=running), int(project["chat_id"]), "room-7",
        {"project_id": "racer"},
    )["project_routing_manifest"]

    [row] = manifest["final_results"]
    assert row["task_id"] == "racer-cancelled"
    from tests.test_project_routing_v664 import _read_result_reference
    assert "cancel_origin" not in row and row["status"] == "cancelled"
    assert _read_result_reference(tmp_path, row)["cancel_origin"] == {
        "source": "owner", "reason": "owner stopped it", "requested_at": "2026-08-10T00:00:00Z"}
    [live] = manifest["active_roots"]
    assert live["task_id"] == "racer-live" and live["cancel_state"] == "pending"
    assert manifest["omissions"]["children"] == 1
    assert manifest["omissions"]["final_results"] == 0
    assert manifest["omissions"]["active_roots"] == 0

    # The live list is bounded too, and says how many of this room's roots it left out.
    crowd = {f"racer-live-{index:02d}": {"task": {"id": f"racer-live-{index:02d}", "project_id": "racer"},
                                          "started_at": "2026-08-10T00:00:05Z"} for index in range(43)}
    crowded = server._decision_turn_metadata(
        _host_ctx(tmp_path, running=crowd), int(project["chat_id"]), "room-8", {"project_id": "racer"},
    )["project_routing_manifest"]
    assert len(crowded["active_roots"]) == 40 and crowded["omissions"]["active_roots"] == 3


def test_the_manifest_row_cap_is_one_runtime_limit(tmp_path, monkeypatch):
    """Both lanes read the same bound; no call-site literal decides the window."""
    import server
    from ouroboros import runtime_limits
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "racer", name="Racer")
    for index in range(3):
        write_task_result(tmp_path, f"root{index}", "completed", project_id="racer",
                          objective="work", ts=f"2026-08-10T00:00:0{index}Z")
    monkeypatch.setattr(runtime_limits, "ROUTING_MANIFEST_RESULT_ROWS", 1)

    room = server._decision_turn_metadata(
        _host_ctx(tmp_path), int(project["chat_id"]), "room-8", {"project_id": "racer"},
    )["project_routing_manifest"]
    main = server._main_routing_manifest(_host_ctx(tmp_path))

    assert len(room["final_results"]) == 1 and room["omissions"]["final_results"] == 2
    # Main's scope is the whole store: its window proves more roots exist, it does not count them.
    assert len(main["final_results"]) == 1 and main["omissions"]["final_results"] is None
