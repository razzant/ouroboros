"""route_to_project + list_projects (v6.33.0 WS10 LLM-first routing)."""

from __future__ import annotations

import json
import queue
import types

import pytest

from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import create_project
from ouroboros.tools.control import _list_projects, _route_to_project, get_tools


def _ctx(tmp_path, events=None, *, task_metadata=None, **overrides):
    values = dict(
        pending_events=events if events is not None else [],
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata=task_metadata or {},
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _owner_turn_ctx(tmp_path, events, metadata):
    """A direct turn the owner door stamped (``origin_message_ref``): the one shape
    that speaks as an owner turn and is offered the manual-target picker; a client
    id alone never does."""
    ref = build_owner_message_ref(
        chat_id=1, client_message_id=metadata["client_message_id"],
        ts="2026-09-24T00:00:00+00:00", text="owner text",
    )
    return _ctx(tmp_path, events, is_direct_chat=True, task_metadata={
        **metadata, "origin_message_ref": ref, "origin_message_text": "owner text",
    })


def test_a_task_on_a_forked_drive_reads_the_registry_from_the_canonical_root(tmp_path, monkeypatch):
    """A promoted task with a workspace runs on a forked execution drive that never
    carries ``state/projects.json``; the routing verbs read the registry through the
    canonical data root, so such a task lists, routes into and names the owner's
    projects. A context without a canonical root still reads its own drive."""
    from ouroboros.tools.control_routing import _effective_scope_note, _promote_chat_to_task

    create_project(tmp_path, "racer", name="Racer")
    child = tmp_path / "state" / "headless_tasks" / "fork-1" / "data"
    child.mkdir(parents=True)
    forked = _ctx(child, task_metadata={"budget_drive_root": str(tmp_path)}, budget_drive_root=str(tmp_path))

    assert "racer — Racer" in _list_projects(forked)
    out = _route_to_project(forked, "racer", "continue the engine tuning", predecessor_task_id="")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED:"), out
    assert forked.pending_events[0]["project_id"] == "racer"
    assert _effective_scope_note(forked, "racer") == " in project 'Racer' (racer)"

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "racer"},
    )
    promoted = _ctx(child, task_metadata={"budget_drive_root": str(tmp_path)}, budget_drive_root=str(tmp_path))
    out = _promote_chat_to_task(promoted, "tune the engine", project_id="racer", workspace="none", predecessor_task_id="")
    assert out.startswith("OK: task") and "in project 'Racer' (racer)" in out, out

    # The quiet direction: no canonical root at all means the task's own drive is the registry.
    own_drive = _ctx(child)
    assert _list_projects(own_drive).startswith("No projects yet")
    assert "target_not_found" in _route_to_project(own_drive, "racer", "msg", predecessor_task_id="")


def test_route_to_existing_project_emits_event_and_receipt(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    # The origin identity is captured at INGRESS and rides task_metadata by
    # value; the tool never re-derives it from chat-log content (v6.73.0 — the
    # message text may even be an LLM paraphrase and routing still keeps the ref).
    origin_ref = {
        "chat_id": 1,
        "client_message_id": "owner-route-1",
        "ts": "2026-07-14T12:00:00Z",
        "text_sha256": "b" * 64,
    }
    events = []
    ctx = _ctx(tmp_path, events, is_direct_chat=True, task_metadata={
        "client_message_id": "owner-route-1",
        "origin_message_ref": origin_ref,
        "origin_message_text": "continue the engine tuning",
    })
    ctx.task_id = "drafter"
    out = _route_to_project(ctx, "racer", "paraphrased: keep tuning the engine", reason="follow-up", predecessor_task_id="")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED:")
    assert "do not retry automatically" in out.lower()
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "promote_chat_to_task"
    assert evt["project_id"] == "racer"
    assert evt["routed_from_main"] is True
    assert "keep tuning the engine" in evt["objective"]
    assert "routing reason: follow-up" in evt["objective"]
    assert evt["chat_id"] == 1
    assert evt["task_id"]
    assert evt["routing_token"]
    assert evt["source_ref"] == origin_ref
    assert evt["source_text"] == "continue the engine tuning"
    assert evt["objective_author"] == {"kind": "task", "task_id": ctx.task_id}
    assert evt["owner_corpus"] == [{"source": "origin_message", "content": "continue the engine tuning"}]
    assert ctx._typed_routing_action_emitted == "route_to_project"


def test_main_route_to_existing_project_explicitly_selects_predecessor_or_stays_fresh(tmp_path):
    import server

    create_project(tmp_path, "racer", name="Racer")
    predecessor = {
        "task_id": "racer-old", "status": "completed", "project_id": "racer",
        "title": "Racer prototype", "objective": "Build the racer prototype",
        "task_contract": {"objective": "Build the racer prototype", "context": "exact old context"},
    }
    result_dir = tmp_path / "task_results"
    result_dir.mkdir()
    (result_dir / "racer-old.json").write_text(json.dumps({"_schema_version": 1, **predecessor}), encoding="utf-8")
    preview = server._task_result_ground_truth(predecessor)
    metadata = {"main_routing_manifest": {"final_results": [preview]}}
    events = []
    route_tool = next(entry for entry in get_tools() if entry.name == "route_to_project")

    out = route_tool.handler(
        _ctx(tmp_path, events, task_metadata=metadata),
        "racer", "Continue the racer", predecessor_task_id="racer-old",
    )

    assert out.startswith("⚠️ ROUTE_UNCONFIRMED")
    assert events[0]["predecessor_task_id"] == "racer-old"
    assert events[0]["predecessor_authority_source"] == preview["authority_source"]

    fresh_events = []
    route_tool.handler(
        _ctx(tmp_path, fresh_events, task_metadata=metadata),
        "racer", "Start a separate racer experiment", predecessor_task_id="",
    )
    assert "predecessor_authority_source" not in fresh_events[0]
    assert "predecessor_task_id" in route_tool.schema["parameters"]["properties"]


def test_managed_swarm_routes_distinct_work_without_a_conversation_latch(tmp_path, monkeypatch):
    create_project(tmp_path, "racer", name="Racer")
    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "scheduled"},
    )
    events = []
    ctx = _ctx(tmp_path, events, project_id="", task_metadata={
        "client_message_id": "swarm-root-1", "force_plan": True, "force_plan_source": "swarm",
    })
    first = _route_to_project(ctx, "racer", "Audit Racer", predecessor_task_id="")
    second = _route_to_project(ctx, "racer", "Research a separate design", predecessor_task_id="")
    assert "durably scheduled" in first and "durably scheduled" in second
    assert len(events) == 2
    assert events[0]["task_id"] != events[1]["task_id"]
    assert all("force_plan" not in event for event in events)
    assert not hasattr(ctx, "_swarm_handoff_attempt")


def test_managed_swarm_uses_ordinary_project_routing(tmp_path, monkeypatch):
    create_project(tmp_path, "beta", name="Beta")
    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "scheduled"},
    )
    events = []
    ctx = _ctx(tmp_path, events, project_id="alpha", task_metadata={"force_plan": True})
    out = _route_to_project(ctx, "beta", "Audit and fix this in Beta", predecessor_task_id="")
    assert "durably scheduled" in out
    assert events[0]["project_id"] == "beta"
    assert "force_plan" not in events[0]


def test_route_to_missing_project_emits_typed_manual_target(tmp_path):
    events = []
    metadata = {
        "client_message_id": "owner-1",
        "routing_contract": {"manual_options": [{"task_id": "task-1", "title": "Fix it"}]},
    }
    ctx = _owner_turn_ctx(tmp_path, events, metadata)
    out = _route_to_project(ctx, "ghost", "do the thing", predecessor_task_id="")
    assert "ROUTING_UNCONFIRMED" in out
    assert len(events) == 1
    assert events[0]["type"] == "routing_manual_target"
    assert events[0]["routing_token"]
    assert events[0]["chat_id"] == 1
    assert events[0]["client_message_id"] == "owner-1"
    assert events[0]["requested_target"] == "ghost"
    assert events[0]["reason"] == "target_not_found"
    assert events[0]["options"] == [{"task_id": "task-1", "title": "Fix it"}]
    assert ctx._typed_routing_action_emitted == "routing_manual_target"


def test_manual_target_preserves_valid_predecessor_and_rejects_unreadable_one(tmp_path):
    import server

    predecessor = {
        "task_id": "previous", "status": "completed", "project_id": "racer",
        "title": "Previous result", "objective": "Build the previous result",
        "task_contract": {"objective": "Build the previous result"},
    }
    result_dir = tmp_path / "task_results"
    result_dir.mkdir()
    (result_dir / "previous.json").write_text(json.dumps({"_schema_version": 1, **predecessor}), encoding="utf-8")
    preview = server._task_result_ground_truth(predecessor)
    events = []
    metadata = {"client_message_id": "owner-4", "main_routing_manifest": {"final_results": [preview]}}

    out = _route_to_project(
        _owner_turn_ctx(tmp_path, events, metadata),
        "missing-project", "continue it", predecessor_task_id="previous",
    )

    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["type"] == "routing_manual_target"
    assert events[0]["predecessor_task_id"] == "previous"
    assert events[0]["predecessor_authority_source"] == preview["authority_source"]

    unreadable_source = {
        "kind": "task_result", "task_id": "gone", "tool": "get_task_result",
        "arguments": {"task_id": "gone", "include_authority": True},
    }
    unreadable_metadata = {"client_message_id": "owner-4", "main_routing_manifest": {"final_results": [{
        "task_id": "gone", "authority_source": unreadable_source,
    }]}}
    rejected_events = []
    rejected = _route_to_project(
        _owner_turn_ctx(tmp_path, rejected_events, unreadable_metadata),
        "missing-project", "continue it", predecessor_task_id="gone",
    )
    assert "AUTHORITY_SOURCE_UNAVAILABLE" in rejected
    assert "missing or unreadable" in rejected
    assert rejected_events == []


def test_route_rejects_dirty_project_id(tmp_path):
    events = []
    metadata = {"client_message_id": "owner-3"}
    out = _route_to_project(_owner_turn_ctx(tmp_path, events, metadata), "Bad Name!", "msg", predecessor_task_id="")
    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["routing_token"]
    assert events[0]["reason"] == "invalid_project_id"


@pytest.mark.parametrize("drained_owner", [False, True])
def test_a_task_issuer_gets_a_typed_refusal_and_no_owner_picker(tmp_path, drained_owner):
    """7=A: a pooled or Swarm root routing to a missing, malformed or unnamed project gets
    ROUTE_REJECTED in its own result; no manual-target event is emitted, so no picker or ack
    can reach an owner surface under an empty message id (the 14.09 incident's class)."""
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    events = []
    for target, failure in (("ghost", "target_not_found"), ("Bad Name!", "invalid_project_id"), ("", "target_unspecified")):
        ctx = _ctx(tmp_path, events, task_id="root-1", task_metadata={
            "root_task_id": "root-1",
            "routing_contract": {"manual_options": [{"task_id": "task-1", "title": "Fix it"}]},
        })
        if drained_owner:
            write_owner_message(tmp_path, "Continue the work", task_id="root-1",
                                msg_id=f"followup-{failure}", client_message_id="owner-followup")
            _drain_incoming_messages([], queue.Queue(), tmp_path, "root-1", None, set(), owner_ctx=ctx)
            assert ctx.last_owner_delivery["client_message_id"] == "owner-followup"
        out = _route_to_project(ctx, target, "continue the work there", predecessor_task_id="")
        assert out.startswith(f"⚠️ ROUTE_REJECTED ({failure})"), out
        assert "list_projects" in out
        assert not hasattr(ctx, "_typed_routing_action_emitted")
    assert events == []


def test_route_empty_target_is_the_typed_abstention_path(tmp_path):
    events = []
    metadata = {
        "client_message_id": "owner-2",
        "routing_contract": {
            "manual_options": [{"action": "new_task_in_project", "label": "New task in Project"}],
        },
    }
    out = _route_to_project(_owner_turn_ctx(tmp_path, events, metadata), "", "ambiguous follow-up", predecessor_task_id="")
    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["routing_token"]
    assert events[0]["reason"] == "target_unspecified"
    assert events[0]["options"][0]["label"] == "New task in Project"


def test_route_requires_message(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    events = []
    ctx = _ctx(tmp_path, events)
    out = _route_to_project(ctx, "racer", "   ", predecessor_task_id="")
    assert "TOOL_ARG_ERROR" in out
    assert events == []
    assert not hasattr(ctx, "_typed_routing_action_emitted")


def test_list_projects_lists_created_projects(tmp_path):
    create_project(tmp_path, "racer", name="Racer")
    create_project(tmp_path, "site", name="Marketing Site")
    out = _list_projects(_ctx(tmp_path))
    assert "racer" in out and "Racer" in out
    assert "site" in out and "Marketing Site" in out


def test_list_projects_empty(tmp_path):
    out = _list_projects(_ctx(tmp_path))
    assert "No projects yet" in out


def test_route_tool_uncertainty_contract_requires_manual_target():
    tool = next(entry for entry in get_tools() if entry.name == "route_to_project")
    description = tool.schema["description"]
    assert "needs_manual_target" in description
    assert "New task in Project" in description
    assert "answer inline and offer" not in description


def test_a_task_issuer_with_an_unmet_obligation_moves_it_through_route_to_project(tmp_path, monkeypatch):
    """Owner 3=A on the sibling verb: a Swarm root routing new work into an existing
    project carries its unmet planning obligation on the promote event it emits, so
    the new root owes the plan and the sender is told the obligation moved."""
    create_project(tmp_path, "racer", name="Racer")
    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "scheduled", "force_plan_transfer": {"from": "swarm-root", "released": True}},
    )
    monkeypatch.setattr("ouroboros.owner_hurry.release_force_plan_obligation", lambda *_a, **_k: None)
    events = []
    ctx = _ctx(tmp_path, events, task_id="swarm-root", task_metadata={
        "force_plan": True, "force_plan_source": "swarm", "root_task_id": "swarm-root",
    })
    out = _route_to_project(ctx, "racer", "Implement it in Racer", predecessor_task_id="")
    assert "durably scheduled" in out and "planning obligation (force_plan) moved to task" in out
    assert events[0]["type"] == "promote_chat_to_task"
    assert events[0]["force_plan"] is True and events[0]["force_plan_source"] == "swarm"
    assert events[0]["force_plan_transferred_from"] == "swarm-root"



def test_route_refusal_keeps_the_typed_code_and_carries_the_models_words_as_detail(tmp_path):
    """The receipt's machine `reason` stays the host's typed code — the cause
    table reads it — while the model's free-text explanation rides `detail`
    beside it. With no options there is nothing to choose, so the durable row
    carries the host's cause sentence instead of «Choose a target»."""
    from ouroboros.project_dialogue import chat_annotation_receipt
    from supervisor.events import _handle_routing_manual_target

    events = []
    metadata = {"client_message_id": "owner-9", "routing_contract": {"manual_options": []}}
    out = _route_to_project(
        _owner_turn_ctx(tmp_path, events, metadata),
        "ghost", "continue it there", reason="I could not find it", predecessor_task_id="",
    )

    assert "ROUTING_UNCONFIRMED" in out
    assert events[0]["reason"] == "target_not_found"
    assert events[0]["detail"] == "I could not find it"
    assert events[0]["options"] == []

    class _Ctx:
        DRIVE_ROOT = tmp_path

        @staticmethod
        def append_jsonl(path, row):
            pass

    _handle_routing_manual_target(events[0], _Ctx)
    row = chat_annotation_receipt(tmp_path, "owner-9", events[0]["routing_token"])
    assert (row["status"], row["reason"]) == ("needs_manual_target", "target_not_found")
    assert row["detail"] == "I could not find it"
    assert row["cause"] == "Not started: that project does not exist"


def test_route_abstention_without_a_target_leaves_the_receipt_target_empty(tmp_path):
    """An unnamed destination is a typed abstention (`target_unspecified`), never a
    task id: the durable row keeps `target` empty (so no task-label lookup runs on
    a reason code) and the owner line reads the host's sentence."""
    from ouroboros.project_dialogue import chat_annotation_receipt
    from supervisor.events import _handle_routing_manual_target

    events = []
    metadata = {"client_message_id": "owner-10", "routing_contract": {"manual_options": []}}
    _route_to_project(
        _owner_turn_ctx(tmp_path, events, metadata),
        "", "continue it somewhere", reason="", predecessor_task_id="",
    )
    assert events[0]["reason"] == "target_unspecified"
    assert events[0]["requested_target"] == ""

    class _Ctx:
        DRIVE_ROOT = tmp_path

        @staticmethod
        def append_jsonl(path, row):
            pass

    _handle_routing_manual_target(events[0], _Ctx)
    row = chat_annotation_receipt(tmp_path, "owner-10", events[0]["routing_token"])
    assert row["target"] == ""
    assert (row["status"], row["reason"]) == ("needs_manual_target", "target_unspecified")
    assert row["cause"] == "Not started: no destination was chosen"


def test_a_suppressed_origin_still_hands_the_routed_task_the_owners_stamped_instruction(tmp_path):
    """Finding 4: a suppressed (never-logged) owner message puts no ``source_text`` on
    the promote event, and the corpus filter dropped the host-stamped ``initial_user``
    row, so the routed task got ``objective_author=task`` with an EMPTY owner corpus:
    the owner's instruction disappeared. The stamped row now rides the corpus under
    its own label; a first turn nobody typed (``initial_text``) is never laundered."""
    from ouroboros.loop_messages import _initialize_owner_directives

    create_project(tmp_path, "racer", name="Racer")
    events = []
    ctx = _ctx(tmp_path, events, is_direct_chat=True,
               task_metadata={"client_message_id": "owner-route-2", "origin_suppressed": True})
    ctx.task_id = "drafter"
    ctx._owner_directives = [{"source": "initial_user", "content": "continue the engine tuning"}]
    out = _route_to_project(ctx, "racer", "paraphrased objective", reason="follow-up", predecessor_task_id="")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED:"), out
    evt = events[0]
    assert evt["origin_suppressed"] is True and "source_text" not in evt
    assert evt["objective_author"] == {"kind": "task", "task_id": "drafter"}
    assert evt["owner_corpus"] == [{"source": "initial_user", "content": "continue the engine tuning"}]
    routed = types.SimpleNamespace(task_metadata={
        "origin_suppressed": True, "objective_author": evt["objective_author"], "owner_corpus": evt["owner_corpus"]})
    _initialize_owner_directives(routed, [{"role": "user", "content": "paraphrased objective"}])
    assert routed._owner_directives == [{"source": "initial_user", "content": "continue the engine tuning"}]

    unstamped = _ctx(tmp_path, [], task_metadata={"client_message_id": "wake-1"})
    unstamped.task_id = "drafter"
    unstamped._owner_directives = [{"source": "initial_text", "content": "a wake-up nobody typed"}]
    _route_to_project(unstamped, "racer", "objective", reason="r", predecessor_task_id="")
    assert unstamped.pending_events[0]["owner_corpus"] == []
