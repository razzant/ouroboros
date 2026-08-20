"""promote_chat_to_task: the promotion event, its project, and the task it enqueues.

This module owns the promote tool's transport event and its host-scope pinning, the
ephemeral swarm promotions and the one task id an unconfirmed attempt may reuse, the
project names derived from a display name or a title, the first-class task the event
enqueues, the route receipt, and the confined skill-repair promotion.

Project chat routing, task/project binding, chat steering and workspace provisioning
were split verbatim into ``tests/test_project_chat_routing.py``,
``tests/test_project_task_binding.py``, ``tests/test_chat_steering.py`` and
``tests/test_promote_workspace_provisioning.py``; the projects-root isolation fixture
they all apply lives in ``tests/_promote_chat_shared.py``.
"""

from __future__ import annotations

import queue
import types

from tests._promote_chat_shared import _isolated_projects_root  # noqa: F401  (autouse fixture applies on import)



def _confirm_promote(monkeypatch):
    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "scheduled"},
    )


def test_promote_tool_emits_event_with_chat_and_project(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "Build the racer prototype", project_id="racer")
    assert out.startswith("OK: task")
    assert "accepted and durably scheduled" in out
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "promote_chat_to_task"
    assert evt["objective"] == "Build the racer prototype"
    assert evt["project_id"] == "racer"
    assert evt["chat_id"] == 1
    assert evt["task_id"]
    assert ctx._typed_routing_action_emitted == "promote_chat_to_task"


def _swarm_ctx(tmp_path, **overrides):
    values = {
        "pending_events": [],
        "event_queue": None,
        "current_chat_id": 1,
        "drive_root": tmp_path,
        "project_id": "",
        "is_ephemeral_turn": True,
        "task_metadata": {"force_plan": True, "force_plan_source": "swarm"},
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_ephemeral_swarm_promotion_carries_intent_and_pins_host_scope(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    ctx = _swarm_ctx(tmp_path, project_id="alpha")

    out = _promote_chat_to_task(
        ctx,
        "Audit and fix the issue",
        project_id="beta",
        project_name="Injected Project",
        workspace_root="/tmp/foreign",
        workspace="none",
        source="https://example.invalid/repo.git",
    )

    assert out.startswith("OK: task")
    evt = ctx.pending_events[0]
    assert evt["force_plan"] is True
    assert evt["force_plan_source"] == "swarm"
    assert evt["project_id"] == "alpha"
    assert evt["project_name"] == evt["workspace_root"] == evt["workspace"] == evt["source"] == ""
    # The override of an explicit owner input is DISCLOSED, never silent.
    assert "Explicit project 'Injected Project' was ignored" in out
    assert "bound to project 'alpha'" in out
    assert ctx._swarm_handoff_attempt["status"] == "scheduled"


def test_ephemeral_swarm_projectless_room_inherits_explicit_project_name(tmp_path, monkeypatch):
    """Q9-A: in a PROJECTLESS room the router turn INHERITS an explicitly passed
    project_name — room scope wins only on a genuine conflict (room already bound
    to a project). Clearing the name here made the saga's first root run
    projectless and strand its work in an off-registry tree."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    ctx = _swarm_ctx(tmp_path)  # project_id="" — projectless main chat

    out = _promote_chat_to_task(
        ctx,
        "Build the slime lab escape game",
        project_name="Slime Lab Escape",
        workspace_root="/tmp/foreign",
        source="https://example.invalid/repo.git",
    )

    assert out.startswith("OK: task")
    assert "new project 'Slime Lab Escape'" in out
    evt = ctx.pending_events[0]
    assert evt["project_name"] == "Slime Lab Escape"
    assert evt["project_id"] == project_id_from_display_name("Slime Lab Escape")
    # The host still owns the rest of the scope surface on a router turn.
    assert evt["workspace_root"] == evt["workspace"] == evt["source"] == ""


def test_ephemeral_swarm_projectless_room_inherits_explicit_project_id(tmp_path, monkeypatch):
    """Q9-A sibling parameter: in a PROJECTLESS room an explicitly passed
    project_id is honored, not silently dropped (the same saga failure shape as
    the project_name drop)."""
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    ctx = _swarm_ctx(tmp_path)  # project_id="" — projectless main chat

    out = _promote_chat_to_task(ctx, "Continue the racer build", project_id="racer")

    assert out.startswith("OK: task")
    assert "in project 'racer'" in out
    assert "ignored" not in out
    evt = ctx.pending_events[0]
    assert evt["project_id"] == "racer"


def test_ephemeral_swarm_room_scope_override_matrix(tmp_path, monkeypatch):
    """Room=A + explicit project B (id or name): A wins WITH a disclosure
    sentence in the response; explicit input equal to the room binding is not a
    conflict and produces no disclosure."""
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    for kwargs, shown in (
        ({"project_id": "beta"}, "beta"),
        ({"project_name": "Beta Project"}, "Beta Project"),
    ):
        ctx = _swarm_ctx(tmp_path, project_id="alpha")
        out = _promote_chat_to_task(ctx, "Audit the issue", **kwargs)
        assert out.startswith("OK: task")
        assert ctx.pending_events[0]["project_id"] == "alpha"
        assert f"Explicit project {shown!r} was ignored" in out
        assert "bound to project 'alpha'" in out

    ctx = _swarm_ctx(tmp_path, project_id="alpha")
    out = _promote_chat_to_task(ctx, "Audit the issue", project_id="alpha")
    assert out.startswith("OK: task")
    assert ctx.pending_events[0]["project_id"] == "alpha"
    assert "ignored" not in out


def test_promoted_named_project_from_projectless_chat_provisions_workspace(tmp_path, monkeypatch):
    """Q9-A worker side: the promote event carrying the inherited name creates and
    binds the project BEFORE the root launches, and the file-less project gets its
    workspace auto-provisioned (Q10-A) — the root never runs projectless."""
    import pathlib

    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "slime0001",
        "objective": "Build the slime lab escape game",
        "project_id": "slime-lab-escape",
        "project_name": "Slime Lab Escape",
        "chat_id": 0,
    }, ctx)

    assert outcome["status"] == "scheduled"
    project = get_project(tmp_path, "slime-lab-escape")
    assert project is not None and project["name"] == "Slime Lab Escape"
    task = enqueued[0]
    assert task["project_id"] == "slime-lab-escape"
    workspace_root = str(task.get("workspace_root") or "")
    assert workspace_root, "file-less named project must get an auto-provisioned workspace"
    assert (pathlib.Path(workspace_root) / ".git").exists()
    assert str(project.get("working_dir") or "") == workspace_root


def test_ephemeral_swarm_unconfirmed_promotion_reuses_one_task_id(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    ctx = _swarm_ctx(tmp_path)

    first = _promote_chat_to_task(ctx, "Audit and fix the issue")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue")

    assert first == second
    assert first.startswith("PROMOTE_UNCONFIRMED")
    assert len(ctx.pending_events) == 1
    assert ctx._swarm_handoff_attempt["task_id"] == ctx.pending_events[0]["task_id"]


def test_ephemeral_swarm_receipt_error_after_emit_keeps_one_attempt(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt unavailable")),
    )
    event_queue = queue.Queue()
    ctx = _swarm_ctx(tmp_path, event_queue=event_queue)

    first = _promote_chat_to_task(ctx, "Audit and fix the issue")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue")

    assert first == second
    assert first.startswith("PROMOTE_UNCONFIRMED")
    assert event_queue.qsize() == 1
    event = event_queue.get_nowait()
    assert ctx._swarm_handoff_attempt["task_id"] == event["task_id"]
    assert ctx._swarm_handoff_attempt["reason"] == "admission_confirmation_failed"


def test_ephemeral_swarm_rejected_promotion_is_latched_without_event(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        "ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot",
        lambda _ctx: "crash_storm",
    )
    ctx = _swarm_ctx(tmp_path)

    first = _promote_chat_to_task(ctx, "Audit and fix the issue")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue")

    assert first == second
    assert first.startswith("PROMOTE_REJECTED")
    assert ctx.pending_events == []
    assert ctx._swarm_handoff_attempt["status"] == "rejected"


def test_managed_swarm_does_not_recursively_propagate_routing_intent(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    ctx = _swarm_ctx(tmp_path, is_ephemeral_turn=False)

    _promote_chat_to_task(ctx, "A later task chosen during execution")

    assert "force_plan" not in ctx.pending_events[0]
    assert not hasattr(ctx, "_swarm_handoff_attempt")


def test_ephemeral_swarm_rejects_steer_without_emitting_event(tmp_path):
    from ouroboros.tools.control import _steer_task

    ctx = _swarm_ctx(tmp_path)
    out = _steer_task(ctx, "existing-root", "do this there")

    assert "cannot steer an existing task" in out
    assert ctx.pending_events == []


def test_promote_tool_rejects_dirty_project_id(tmp_path):
    from ouroboros.tools.control import _promote_chat_to_task

    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "x", project_id="Bad Name!")
    assert "TOOL_ARG_ERROR" in out
    assert not ctx.pending_events


def test_promote_tool_project_name_creates_named_project_event(tmp_path, monkeypatch):
    """LLM-first 'create a named project and work there' (v6.33.0): project_name
    derives a clean id, carries the human display name, and rides title."""
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(
        ctx, "research everything about the airi institute",
        project_name="Airi Research", title="Airi Research",
    )
    assert out.startswith("OK: task")
    assert "new project 'Airi Research'" in out
    evt = events[0]
    assert evt["project_name"] == "Airi Research"
    assert evt["project_id"] == "airi-research"   # derived, filesystem-clean
    assert evt["title"] == "Airi Research"


def test_project_id_from_display_name_handles_non_ascii():
    """A Cyrillic-only display name must still yield a usable (hash) id, not '' —
    so the named-project feature works for the Russian-speaking owner."""
    from ouroboros.project_facts import project_id_from_display_name

    assert project_id_from_display_name("airi research") == "airi-research"
    assert project_id_from_display_name("Динозавры").startswith("proj_")
    # Deterministic: re-asking for the same name resolves to the same project.
    assert project_id_from_display_name("Динозавры") == project_id_from_display_name("Динозавры")
    assert project_id_from_display_name("") == ""


def test_promote_tool_cyrillic_project_name_still_creates(tmp_path, monkeypatch):
    """promote_chat_to_task(project_name=<cyrillic>) must NOT fail — it derives a
    hash id while keeping the Cyrillic display name (Workflow-caught regression)."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _promote_chat_to_task(ctx, "исследуй динозавров", project_name="динозавры", title="динозавры")
    assert "TOOL_ARG_ERROR" not in out
    assert out.startswith("OK: task")
    evt = events[0]
    assert evt["project_name"] == "динозавры"
    assert evt["project_id"] == project_id_from_display_name("динозавры")
    assert evt["project_id"].startswith("proj_")  # ASCII-clean hash fallback


def test_promote_event_names_project_from_display_name(tmp_path, monkeypatch):
    """The handler creates the project with the human display name (not the bare
    id) and persists the task title (v6.33.0)."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "air01",
        "objective": "research the airi institute",
        "project_id": "airi-research",
        "project_name": "Airi Research",
        "title": "Airi Research",
        "chat_id": 0,
    }, ctx)

    project = get_project(tmp_path, "airi-research")
    assert project is not None
    assert project["name"] == "Airi Research"      # human name, not the bare id
    assert enqueued[0]["title"] == "Airi Research"  # persisted on the task


def test_derive_project_name_prefers_title(tmp_path):
    """_derive_project_name uses the model-coined short title over the objective
    so a converted card never shows a truncated sentence or a bare id (v6.33.0)."""
    from ouroboros.gateway.projects import _derive_project_name
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path, "tt01", STATUS_RUNNING,
        title="Tic-tac-toe game",
        objective="make an html page with a tic-tac-toe game that tracks score",
    )
    assert _derive_project_name(tmp_path, "tt01") == "Tic-tac-toe game"


def test_promote_event_enqueues_first_class_task(tmp_path, monkeypatch):
    """The supervisor handler enqueues a pooled OWNER task (not a subagent),
    registers the project, and carries the chat thread."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    evt = {
        "type": "promote_chat_to_task",
        "task_id": "abc12345",
        "objective": "Research the market",
        "expected_output": "A summary",
        "project_id": "research-1",
        "chat_id": 0,  # falls back to owner chat
    }
    workers.promote_chat_to_task(evt, ctx)

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["id"] == "abc12345"
    assert task["type"] == "task"
    assert task["project_id"] == "research-1"
    assert "delegation_role" not in task
    assert "_is_direct_chat" not in task
    assert "Expected output: A summary" in task["text"]
    # The project got registered as a side effect, and the promoted task runs in
    # the PROJECT thread: its chat_id is the project's deterministic chat_id (not
    # the main/owner fallback), so its live card + owner mailbox route to the panel.
    from ouroboros.contracts.chat_id_policy import project_chat_id
    from ouroboros.projects_registry import get_project

    project = get_project(tmp_path, "research-1")
    assert project is not None
    assert task["chat_id"] == project["chat_id"] == project_chat_id("research-1")
    assert task["chat_id"] != 1  # not the owner-chat fallback
    # P2: the promoted task is BOUND to its project, so /api/state's all_task_bindings
    # surfaces it and the frontend never offers a stray "turn into project" button.
    from ouroboros.projects_registry import all_task_bindings
    assert all_task_bindings(tmp_path).get("abc12345") == project["chat_id"]


def test_promote_worker_persists_swarm_intent_on_managed_root(tmp_path, monkeypatch):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=enqueued.append,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "swarmroot1",
        "objective": "Audit and fix the issue",
        "chat_id": 1,
        "force_plan": True,
        "force_plan_source": "swarm",
    }, ctx)

    assert outcome["status"] == "scheduled"
    assert enqueued[0]["metadata"]["force_plan"] is True
    assert enqueued[0]["metadata"]["force_plan_source"] == "swarm"


def test_route_to_project_event_emits_route_receipt_action(tmp_path, monkeypatch):
    """route_to_project reuses promote admission but must retain its distinct
    host receipt action instead of rendering the task as a fresh promotion."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "racer", name="Racer")
    receipts = []
    enqueued = []

    class Bridge:
        def send_routing_ack(self, *args, **kwargs):
            receipts.append((args, kwargs))

        def broadcast(self, *args, **kwargs):
            pass

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        bridge=Bridge(),
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=lambda *args, **kwargs: None,
    )

    _handle_promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "route01",
        "routing_token": "route-token-01",
        "objective": "Continue the racer",
        "project_id": "racer",
        "chat_id": 1,
        "client_message_id": "owner-route-receipt-1",
        "routed_from_main": True,
    }, ctx)

    assert enqueued and enqueued[0]["project_id"] == "racer"
    assert receipts[-1][1]["action"] == "route_to_project"
    assert receipts[-1][1]["status"] == "scheduled"


def test_promoted_skill_repair_is_canonical_confined_managed_task(tmp_path, monkeypatch):
    import supervisor.workers as workers

    payload = tmp_path / "skills" / "external" / "alpha"
    payload.mkdir(parents=True)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    result = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "repair01",
        "objective": "Repair alpha and re-run review",
        "chat_id": 1,
        "task_constraint": {
            "mode": "skill_repair",
            "skill_name": "alpha",
            "payload_root": "skills/external/alpha",
            "allow_enable": True,
            "allow_review": False,
            "extra_allowlist": ["run_command"],
        },
    }, ctx)

    assert result == {"status": "scheduled", "task_id": "repair01"}
    assert len(enqueued) == 1
    task = enqueued[0]
    assert task.get("_ephemeral_turn") is None
    assert task["task_constraint"] == {
        "mode": "skill_repair",
        "skill_name": "alpha",
        "payload_root": "skills/external/alpha",
        "allow_enable": False,
        "allow_review": True,
    }
    assert task["task_contract"]["objective"] == "Repair alpha and re-run review"


def test_promoted_skill_repair_rejects_missing_payload(tmp_path, monkeypatch):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    result = workers.promote_chat_to_task({
        "task_id": "repair02",
        "objective": "Repair missing alpha",
        "task_constraint": {
            "mode": "skill_repair",
            "skill_name": "alpha",
            "payload_root": "skills/external/alpha",
        },
    }, ctx)

    assert result == {
        "status": "needs_manual_target",
        "reason": "skill_repair_payload_missing",
        "task_id": "repair02",
    }
    assert enqueued == []

    invalid = workers.promote_chat_to_task({
        "task_id": "repair03",
        "objective": "Repair escaped payload",
        "task_constraint": {
            "mode": "skill_repair",
            "skill_name": "alpha",
            "payload_root": "skills/external/alpha/../../memory",
        },
    }, ctx)
    assert invalid == {
        "status": "needs_manual_target",
        "reason": "invalid_skill_repair_constraint",
        "task_id": "repair03",
    }
    assert enqueued == []


def test_promote_route_persists_source_ref_and_fails_closed_on_binding_error(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, project_binding_for_task

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "racer", name="Racer")
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    from ouroboros.project_dialogue import _text_sha256

    owner_text = "continue the engine tuning"
    source_ref = {
        "chat_id": 1,
        "client_message_id": "owner-route-1",
        "ts": "2026-07-14T12:00:00Z",
        "text_sha256": _text_sha256(owner_text),
    }
    result = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "route-ok",
        "objective": "Continue",
        "project_id": "racer",
        "chat_id": 1,
        "routed_from_main": True,
        "source_ref": source_ref,
        "source_text": owner_text,
    }, ctx)
    assert result["status"] == "scheduled"
    binding = project_binding_for_task(tmp_path, "route-ok")
    assert binding["source_ref"] == source_ref
    assert binding["source_text"] == owner_text
    # The origin identity also rides the TASK RECORD for post-hoc conversion.
    assert enqueued[0]["origin_message_ref"] == source_ref
    assert enqueued[0]["origin_message_text"] == owner_text
    assert len(enqueued) == 1

    monkeypatch.setattr(
        "ouroboros.projects_registry.bind_task_to_project",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk failure")),
    )
    failed = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "route-fail",
        "objective": "Continue",
        "project_id": "racer",
        "chat_id": 1,
        "routed_from_main": True,
        "source_ref": source_ref,
        "source_text": owner_text,
    }, ctx)
    assert failed == {
        "status": "needs_manual_target",
        "reason": "project_binding_failed",
        "task_id": "route-fail",
    }
    assert len(enqueued) == 1
