"""Situation names use the memory view's role and exact host-bound room identities."""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.consciousness_authority import consciousness_origin_metadata
from ouroboros.memory_view_events import situation_names
from ouroboros.places import connect_place, conversation_chat_id
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_context import focus_metadata
from ouroboros.presence_events import PresenceTurnEvent
from ouroboros.projects_registry import create_project
from ouroboros.tools.knowledge import _knowledge_read, _knowledge_write
from ouroboros.tools.registry import ToolContext


def test_knowledge_receipts_show_the_project_room_in_integrating_foci_and_are_quiet_for_helpers(tmp_path):
    project = create_project(tmp_path, "nova", name="Nova")
    room = project["chat_id"]
    exact = f"This situation's names: room:{room} (Project Nova)"
    fields = {"repo_dir": tmp_path, "drive_root": tmp_path, "current_chat_id": room}
    integrating = ToolContext(**fields, task_id="root")
    child = ToolContext(**fields, task_id="child", task_contract={"lineage": {"delegation_role": "subagent"}})
    nanny = ToolContext(**fields, task_id="nanny", task_metadata={"configured_subagent": {"route": {"kind": "agent_session"}}})

    first = _knowledge_read(integrating, "unwritten", scope="global")
    assert "Topic 'unwritten' not found" in first and first.splitlines().count(exact) == 1
    assert "This situation's names:" not in _knowledge_read(child, "unwritten", scope="global")

    for index, ctx in enumerate((integrating, child, nanny)):
        topic = f"topic-{index}"
        written = _knowledge_write(ctx, topic, content="Room facts.", mode="overwrite", scope="global")
        read = _knowledge_read(ctx, topic, scope="global")
        assert written.startswith("✅") and read.endswith("Room facts.")
        assert written.splitlines().count(exact) == read.splitlines().count(exact) == int(index == 0)
        if index:
            assert "This situation's names:" not in written and "This situation's names:" not in read


def test_place_names_use_exact_carried_components_and_no_thread_name_without_a_thread(tmp_path):
    place = connect_place(tmp_path, connector="fixture", provider="chat", label="Nova Team", place_id="p/%")
    route = {"provider": "chat", "account_id": "A/%:B", "conversation_id": "C/%", "thread_id": "T/%"}
    key = conversation_key(**route)
    chat = conversation_chat_id(key)
    event = PresenceTurnEvent(source_event_id="event-1", conversation_key=key, actor={"id": "member"},
                              conversation={"name": "Nova discussion"}, message={}, text="Status", **route)
    admission = SimpleNamespace(place=place, binding_id=place.place_id, allowance={}, circle=None)
    metadata = focus_metadata(admission, event, "place", chat)
    metadata["presence"] = {"place_id": place.place_id, "event": {**route, "conversation": event.conversation}}
    names = (f"room:{chat}", "place:p%2F%25", "conversation:chat/A%2F%25%3AB/C%2F%25",
             "thread:chat/A%2F%25%3AB/C%2F%25/T%2F%25")
    assert situation_names({"metadata": metadata}, str(chat)) == names
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="place", task_metadata=metadata)
    reply = _knowledge_read(ctx, "unwritten", scope="global")
    assert all(name in reply for name in names)
    assert "place:p%2F%25 (Nova Team)" in reply
    assert "conversation:chat/A%2F%25%3AB/C%2F%25 (Nova discussion)" in reply

    carried = consciousness_origin_metadata(metadata, started_root=True)
    assert "presence" not in carried
    ordinary_task = {"type": "task", "metadata": carried}
    assert situation_names(ordinary_task, "1") == ("room:1", *names[1:])
    in_conversation = {"focus": {**metadata["focus"], "transport": {**route, "thread_id": ""}}}
    assert situation_names(in_conversation, str(chat)) == names[:-1]
    assert situation_names({"type": "task"}, "1") == ("room:1",)
