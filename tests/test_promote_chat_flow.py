"""promote_chat_to_task + project chat routing (multi-project, v6.32.0)."""

from __future__ import annotations

import queue
import pathlib
import threading
import types

import pytest

from ouroboros import dialogue_store
from tests import _dialogue as dialogue_test
from tests.test_swarm_host_admission import host as _swarm_host

swarm_host = _swarm_host


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )


def _confirm_promote(monkeypatch, effective_project_id: str = ""):
    monkeypatch.setattr(  # the receipt names where admission actually put the task
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": effective_project_id},
    )


def test_first_main_conversation_promotion_keeps_its_original_message_annotation(swarm_host):
    """No project/active-root facts are needed to transport the ingress id."""
    import server
    from ouroboros.tools.control import _start_task
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_swarm_host_admission import incoming_case, rows

    host = swarm_host
    case = incoming_case(host, "main")
    case.incoming["task_metadata"].update(force_plan=False, force_plan_source="")
    observed = []

    def direct(chat_id, text, _image=None, *, task_metadata, **_kwargs):
        observed.append(task_metadata)
        ctx = types.SimpleNamespace(
            current_chat_id=chat_id, drive_root=host.root, task_metadata=task_metadata,
            pending_events=[], event_queue=types.SimpleNamespace(
                put_nowait=lambda event: _handle_promote_chat_to_task(event, host.ctx)),
        )
        result = _start_task(ctx, text, title="Owner request", workspace="none")
        assert result.startswith("OK: task"), result

    host.ctx.handle_chat_direct = direct
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    assert len(observed) == 1 and not host.attempts
    assert len(host.pending) == 1
    [annotation] = rows(host.root / "logs/chat_annotations.jsonl")
    assert annotation["client_message_id"] == case.client_message_id
    assert annotation["status"] == "scheduled"
    assert annotation["target"] == host.pending[0]["id"]
    assert host.pending[0]["origin_message_ref"] == case.ref


def test_promote_tool_emits_event_with_chat_and_project(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
    )
    out = _start_task(ctx, "Build the racer prototype", title="Racer prototype",
                                project_id="racer")
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


def test_presence_promotion_preserves_ceiling_and_cannot_choose_new_scope(tmp_path, monkeypatch):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, presence_ceiling_payload
    from ouroboros.tools.control import _build_child_subagent_contract, _start_task

    _confirm_promote(monkeypatch)
    capability_ceiling = presence_ceiling_payload(PresenceCapabilityCeiling(
        skill_name="project-helper", skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=8,
        tool_grants=(), resource_grants=(), digest="0" * 64,
    ))
    contract = {
        "capability_ceiling": capability_ceiling,
        "context": "preserve exact owner context",
        "predecessor_authority": {"result": "preserve completed predecessor",
                                  "source": {"task_id": "previous", "tool": "get_task_result"}},
        "attachment_manifest": [{
            "ordinal": 0, "status": "staged", "reason": "", "label": "authority",
            "root": "artifact_store", "relpath": "attachments/authority.txt",
            "abs_path": str(tmp_path / "parent" / "attachments" / "authority.txt"),
        }],
    }
    ctx = types.SimpleNamespace(
        pending_events=[],
        event_queue=None,
        current_chat_id=4242,
        drive_root=tmp_path,
        task_metadata={"presence": {"binding_id": "b" * 32}},
        task_contract=contract,
    )
    out = _start_task(
        ctx,
        "Research the question",
        title="Question research",
        project_name="Injected",
        workspace_root="/tmp/foreign",
        source="https://example.invalid/repo.git",

    )
    assert out.startswith("OK: task")
    event = ctx.pending_events[0]
    assert event["presence"] == {"binding_id": "b" * 32}
    assert event["task_contract"] == contract
    assert event["project_id"] == event["project_name"] == ""
    assert event["workspace_root"] == event["workspace"] == event["source"] == ""
    child = _build_child_subagent_contract({
        "tid": "presence-child", "objective": "Inspect", "expected_output": "Report",
        "parent_contract": event["task_contract"],
        "attachment_manifest": event["task_contract"]["attachment_manifest"],
    })
    for key in ("capability_ceiling", "context", "attachment_manifest"):
        assert child[key] == contract[key]
    assert child["predecessor_authority"]["source"] == contract["predecessor_authority"]["source"]
    assert "result" not in child["predecessor_authority"]
    assert child["predecessor_authority"]["omitted_fields"]["result"] == len("preserve completed predecessor")


def test_real_presence_promotion_rebases_root_and_materializes_all_attachments(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    from ouroboros.presence_runner import _build_task
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools.control import (
        _build_child_subagent_contract,
        _materialize_child_attachment_manifest,
        _start_task,
    )
    from tests.test_presence_runner import _admission, _event

    inherited_source = tmp_path / "presence-input.txt"
    inherited_source.write_text("presence authority bytes", encoding="utf-8")
    upload_source = tmp_path / "promotion-input.txt"
    upload_source.write_text("promotion authority bytes", encoding="utf-8")
    presence_task = _build_task(
        _admission(), _event(), drive_root=tmp_path, staged_files=(inherited_source,),
    )
    presence_task["task_contract"]["context"] = "preserve exact presence context"
    presence_task["task_contract"]["predecessor_authority"] = {
        "result": "preserve predecessor result",
    }
    assert presence_task["attachments"] == presence_task["task_contract"]["attachment_manifest"]

    _confirm_promote(monkeypatch)
    tool_ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=4242,
        drive_root=tmp_path,
        task_metadata={
            **presence_task["metadata"],
            "chat_attachment_uploads": [{"path": str(upload_source), "label": "promotion input"}],
        },
        task_contract=presence_task["task_contract"],
    )
    result = _start_task(
        tool_ctx,
        "Research the new question deeply",
        title="Deep question research",
        expected_output="A grounded report",
        project_name="forbidden scope",
        workspace_root="/tmp/forbidden",

    )
    assert result.startswith("OK: task")
    event = tool_ctx.pending_events[0]

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    worker_ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task) or task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task(event, worker_ctx)

    assert outcome["status"] == "scheduled"
    promoted = enqueued[0]
    contract = promoted["task_contract"]
    assert promoted["type"] == contract["task_type"] == "task"
    assert promoted["objective"] == contract["objective"] == "Research the new question deeply"
    assert promoted["expected_output"] == contract["expected_output"] == "A grounded report"
    assert contract["lineage"]["root_task_id"] == promoted["id"]
    assert contract["context"] == "preserve exact presence context"
    assert contract["predecessor_authority"]["result"] == "preserve predecessor result"
    assert contract["capability_ceiling"] == presence_task["task_contract"]["capability_ceiling"]
    assert promoted["attachments"] == contract["attachment_manifest"]
    assert len(promoted["attachments"]) == 2
    assert all(pathlib.Path(row["abs_path"]).is_file() for row in promoted["attachments"])
    assert all(
        pathlib.Path(row["abs_path"]).is_relative_to(
            tmp_path / "task_results" / "artifacts" / promoted["id"] / "attachments"
        )
        for row in promoted["attachments"]
    )

    child_root = tmp_path / "child-drive"
    child_manifest, attachment_error = _materialize_child_attachment_manifest(
        contract, child_root, "presence-child",
    )
    assert attachment_error == ""
    child = _build_child_subagent_contract({
        "tid": "presence-child", "objective": "Inspect both inputs",
        "expected_output": "Report", "parent_contract": contract,
        "root_task_id": promoted["id"], "parent_task_id": promoted["id"],
        **child_manifest,
    })
    work_order = compile_external_work_order({
        "id": "presence-child", "objective": "Inspect both inputs",
        "expected_output": "Report", "task_contract": child,
        "parent_task_id": promoted["id"], "root_task_id": promoted["id"],
    })
    assert child["capability_ceiling"] == contract["capability_ceiling"]
    assert len(child["attachment_manifest"]) == 2
    assert "presence authority bytes" not in work_order
    assert "presence-input.txt" in work_order and "promotion-input.txt" in work_order


def test_real_presence_promotion_rejection_cleans_promoted_attachment_copy(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    from ouroboros.presence_runner import _build_task
    from ouroboros.tools.control import _start_task
    from tests.test_presence_runner import _admission, _event

    inherited_source = tmp_path / "presence-input.txt"
    inherited_source.write_text("presence authority bytes", encoding="utf-8")
    presence_task = _build_task(
        _admission(), _event(), drive_root=tmp_path, staged_files=(inherited_source,),
    )
    original_path = pathlib.Path(presence_task["attachments"][0]["abs_path"])
    assert original_path.is_file()

    _confirm_promote(monkeypatch)
    tool_ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=4242,
        drive_root=tmp_path,
        task_metadata={
            **presence_task["metadata"],
            "chat_attachment_uploads": [{
                "path": str(tmp_path / "missing-upload.txt"), "label": "missing",
            }],
        },
        task_contract=presence_task["task_contract"],
    )
    assert _start_task(tool_ctx, "Long work", title="Long work").startswith("OK: task")
    event = tool_ctx.pending_events[0]

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    worker_ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task) or task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task(event, worker_ctx)

    # В25c (capinv-447): partial staging is the default — the inherited input
    # rides into the promoted task and the missing upload stays a disclosed row.
    assert outcome["status"] != "needs_manual_target"
    assert enqueued, "promotion must proceed with the staged attachment"
    manifest = enqueued[0]["attachments"]
    assert [row["status"] for row in manifest] == ["staged", "rejected"]
    assert manifest[1]["reason"] == "source_missing"
    assert original_path.is_file()


def _managed_swarm_ctx(tmp_path, **overrides):
    values = {
        "pending_events": [],
        "event_queue": None,
        "current_chat_id": 1,
        "drive_root": tmp_path,
        "project_id": "",
        "task_metadata": {"force_plan": True, "force_plan_source": "swarm"},
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_managed_swarm_can_choose_a_named_project_for_later_work(tmp_path, monkeypatch):
    """The admitted root chooses the ordinary project and source selectors."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch, project_id_from_display_name("Slime Lab Escape"))
    ctx = _managed_swarm_ctx(tmp_path, task_id="swarm-root")  # project_id="" — projectless main chat

    out = _start_task(
        ctx,
        "Build the slime lab escape game",
        title="Slime lab escape game",
        project_name="Slime Lab Escape",
        workspace_root="/tmp/foreign",
        source="https://example.invalid/repo.git",

    )

    assert out.startswith("OK: task")
    assert f"in project '{project_id_from_display_name('Slime Lab Escape')}'" in out
    evt = ctx.pending_events[0]
    assert evt["project_name"] == "Slime Lab Escape"
    assert evt["project_id"] == project_id_from_display_name("Slime Lab Escape")
    assert evt["workspace_root"] == "/tmp/foreign"
    assert evt["source"] == "https://example.invalid/repo.git"
    assert "force_plan_transferred_from" not in evt and "force_plan" not in evt
    assert ctx.task_metadata["force_plan"] is True


def test_managed_swarm_can_choose_an_existing_project_for_later_work(tmp_path, monkeypatch):
    """Q9-A sibling parameter: in a PROJECTLESS room an explicitly passed
    project_id is honored, not silently dropped (the same saga failure shape as
    the project_name drop)."""
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch, "racer")
    ctx = _managed_swarm_ctx(tmp_path)  # project_id="" — projectless main chat

    out = _start_task(ctx, "Continue the racer build", title="Racer build continuation",
                                project_id="racer")

    assert out.startswith("OK: task")
    assert "in project 'racer'" in out
    assert "ignored" not in out
    evt = ctx.pending_events[0]
    assert evt["project_id"] == "racer"


def test_managed_swarm_promotes_in_current_project_without_explicit_target(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    ctx = _managed_swarm_ctx(tmp_path, project_id="alpha", task_id="swarm-root")
    out = _start_task(ctx, "Audit the issue", title="Issue audit")
    assert out.startswith("OK: task")
    assert ctx.pending_events[0]["project_id"] == "alpha"
    assert "force_plan_transferred_from" not in ctx.pending_events[0]
    assert not hasattr(ctx, "_swarm_handoff_attempt")


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


def test_promote_announces_project_started_only_on_real_creation(tmp_path, monkeypatch):
    """B1 seam 1 (owner 2=A): the promote path announces the durable Main
    `project_started` row exactly when create_project actually created the row;
    a second promote into the SAME project (idempotent replay, created=False)
    stays silent — the created gate, not the delivery dedupe, is under test."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_k: queued.append(dict(event)) or True,
    )
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: True,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    evt = {
        "type": "promote_chat_to_task",
        "task_id": "slime0001",
        "objective": "Build the slime lab escape game",
        "project_id": "slime-lab-escape",
        "project_name": "Slime Lab Escape",
        "chat_id": 0,
    }

    assert workers.promote_chat_to_task(evt, ctx)["status"] == "scheduled"
    assert [row["system_type"] for row in queued] == ["project_started"]
    row = queued[0]
    assert row["chat_id"] == 1
    assert row["task_id"] == "slime0001"
    assert row["delivery_id"] == "project-start:slime-lab-escape"
    assert row["progress_meta"]["project_id"] == "slime-lab-escape"
    assert row["progress_meta"]["project_name"] == "Slime Lab Escape"

    queued.clear()
    outcome = workers.promote_chat_to_task(
        {**evt, "task_id": "slime0002"}, ctx,
    )
    assert outcome["status"] == "scheduled"
    assert queued == []  # idempotent re-create (created=False) announces nothing


def test_source_prepared_promote_announces_started_for_flow_created_project(
    tmp_path, monkeypatch,
):
    """B1: a source-bearing promote registers its project OFF-LOOP
    (`resolve_promote_source` in `_prepare_promote_source_off_loop`), so the
    workers-side create_project replay reports created=False — the continuation
    carries `_source_created` and the one announce still fires for this
    agent-initiated creation."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_k: queued.append(dict(event)) or True,
    )
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: True,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    # The off-loop source half already registered the project.
    assert create_project(
        tmp_path, "cloned-repo", name="Cloned Repo", origin="promote_chat_to_task",
    )["created"] is True

    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "clone0001",
        "objective": "Work on the cloned repo",
        "project_id": "cloned-repo",
        "chat_id": 0,
        "_source_prepared": True,
        "_source_created": True,
    }, ctx)
    assert outcome["status"] == "scheduled"
    assert [row["system_type"] for row in queued] == ["project_started"]
    assert queued[0]["delivery_id"] == "project-start:cloned-repo"


def test_worker_admits_promoted_presence_with_same_verified_ceiling(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import (
        PresenceProfileResolution,
        PresenceSelection,
        PresenceToolTarget,
    )
    from tests.frozen_presence_runtime import ResolvedPresenceRuntime

    resolution = PresenceProfileResolution(
        active=(PresenceSelection("1" * 64, PresenceToolTarget("builtin", "chat_history")),),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="helper",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    outcome = workers.promote_chat_to_task(
        {
            "task_id": "presencechild1",
            "objective": "Continue the research",
            "chat_id": 4242,
            "presence": {"binding_id": "e" * 32},
            "task_contract": contract,
        },
        ctx,
    )
    assert outcome["status"] == "scheduled"
    task = enqueued[0]
    assert task["_presence_origin"] is True
    assert task["metadata"]["presence"]["binding_id"] == "e" * 32
    assert task["task_contract"]["capability_ceiling"]["digest"] == ceiling.digest


def test_presence_unconfirmed_promotion_records_handoff_for_reconciliation(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    ctx = _managed_swarm_ctx(tmp_path, task_metadata={"presence": {"binding_id": "presence-binding"}})
    out = _start_task(ctx, "Audit and fix the issue", title="Issue audit and fix")
    assert out.startswith("Accepted, waiting to start:")
    assert len(ctx.pending_events) == 1
    assert ctx._swarm_handoff_attempt["task_id"] == ctx.pending_events[0]["task_id"]
    assert ctx._swarm_handoff_attempt["status"] == "unconfirmed"


def test_promotion_receipt_error_after_emit_propagates(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt unavailable")),
    )
    event_queue = queue.Queue()
    ctx = _managed_swarm_ctx(tmp_path, event_queue=event_queue)
    with pytest.raises(OSError, match="receipt unavailable"):
        _start_task(ctx, "Audit and fix the issue", title="Issue audit and fix")
    assert event_queue.qsize() == 1
    assert event_queue.get_nowait()["type"] == "promote_chat_to_task"
    assert not hasattr(ctx, "_swarm_handoff_attempt")


def test_presence_rejected_promotion_records_handoff_without_event(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr(
        "ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot",
        lambda _ctx: "crash_storm",
    )
    ctx = _managed_swarm_ctx(tmp_path, task_metadata={"presence": {"binding_id": "presence-binding"}})
    out = _start_task(ctx, "Audit and fix the issue", title="Issue audit and fix")
    assert out.startswith("⚠️ PROMOTE_REJECTED")
    assert ctx.pending_events == []
    assert ctx._swarm_handoff_attempt["status"] == "rejected"


def test_managed_swarm_can_steer_through_the_ordinary_receipt_path(tmp_path, monkeypatch):
    from ouroboros.tools.control import _tell_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_routing_annotation",
        lambda *_args, **_kwargs: {"status": "delivered"},
    )
    ctx = _managed_swarm_ctx(tmp_path, task_id="swarm-root")
    out = _tell_task(ctx, "existing-root", "do this there")
    # A Swarm root speaks for itself: its words are WRITTEN as a task message
    # (never the owner's), on the same receipt rail (wave 2 issuer fact).
    assert out.startswith("Message written to existing-root")
    assert ctx.pending_events[0]["type"] == "steer_task"
    assert ctx.pending_events[0]["target_task_id"] == "existing-root"
    assert ctx.pending_events[0]["issuer"] == {
        "kind": "task", "task_id": "swarm-root", "root_task_id": "swarm-root",
    }
    assert not hasattr(ctx, "_swarm_handoff_attempt")


def test_promote_tool_rejects_dirty_project_id(tmp_path):
    from ouroboros.tools.control import _start_task

    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _start_task(ctx, "x", title="Scoped work", project_id="Bad Name!")
    assert "TOOL_ARG_ERROR" in out
    assert not ctx.pending_events


def test_promote_tool_project_name_creates_named_project_event(tmp_path, monkeypatch):
    """LLM-first 'create a named project and work there' (v6.33.0): project_name
    derives a clean id, carries the human display name, and rides title."""
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch, "airi-research")
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _start_task(
        ctx, "research everything about the airi institute",
        project_name="Airi Research", title="Airi Research",

    )
    assert out.startswith("OK: task")
    assert "in project 'airi-research'" in out   # the receipt's destination, not the ask
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
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path,
    )
    out = _start_task(ctx, "исследуй динозавров", project_name="динозавры", title="динозавры")
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
    assert (task["delegation_role"], task["root_task_id"]) == ("root", task["id"])
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


def test_promote_initial_task_defaults_to_partial_on_attachment_rejection(tmp_path, monkeypatch):
    """В25c (capinv-447): a rejected upload becomes a disclosed manifest row;
    the promotion itself proceeds instead of discarding the whole task."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task) or task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    good = tmp_path / "good.txt"
    good.write_text("payload", encoding="utf-8")
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "attach-reject",
        "objective": "must use the input",
        "chat_id": 1,
        "project_id": "attachment-project",
        "project_name": "Attachment Project",
        "attachment_uploads": [
            {"path": str(good), "label": "good"},
            {"path": str(tmp_path / "missing.txt"), "label": "missing"},
        ],
    }, ctx)

    assert outcome["status"] != "needs_manual_target"
    assert enqueued, "promotion must proceed with the rejection disclosed"
    manifest = enqueued[0]["attachments"]
    assert [row["status"] for row in manifest] == ["staged", "rejected"]
    assert manifest[1]["reason"] == "source_missing"


def test_promote_success_relocates_pre_admitted_attachment_to_child_drive(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    enqueued = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task) or task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "attach-success",
        "objective": "use the input",
        "chat_id": 1,
        "project_id": "attachment-success-project",
        "project_name": "Attachment Success Project",
        "attachment_uploads": [{"path": str(source), "label": "input"}],
    }, ctx)

    assert outcome["status"] == "scheduled"
    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["drive_root"] != str(tmp_path)
    staged_path = pathlib.Path(task["attachments"][0]["abs_path"])
    assert staged_path.is_file()
    assert staged_path.is_relative_to(pathlib.Path(task["drive_root"]))
    assert str(staged_path) in task["text"]
    old_staged_path = (
        tmp_path / "task_results" / "artifacts" / "attach-success"
        / "attachments" / source.name
    )
    assert str(old_staged_path) not in task["text"]
    assert not (
        tmp_path / "task_results" / "artifacts" / "attach-success" / "attachments"
    ).exists()


def test_promote_worker_ignores_legacy_swarm_flags(tmp_path, monkeypatch):
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
    assert "force_plan" not in enqueued[0].get("metadata", {})


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


def test_promoted_skill_repair_is_ordinary_managed_task_with_selected_resource(tmp_path, monkeypatch):
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

    assert {key: result[key] for key in ("status", "task_id")} == {
        "status": "scheduled",
        "task_id": "repair01",
    }
    assert result["_admitted_task_contract"] == enqueued[0]["task_contract"]
    assert len(enqueued) == 1
    task = enqueued[0]
    assert task.get("_ephemeral_turn") is None
    assert task["task_constraint"] == {
        "mode": "normal",
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
        "never_admitted": True,
        "_admission_cleanup_manifest": [],
    }
    assert len(enqueued) == 1


def test_promote_chat_to_task_broadcasts_projects_changed(tmp_path, monkeypatch):
    """Backend project creation pushes a projects_changed WS frame carrying the new
    chat_id, so the frontend fan-out learns the project thread IMMEDIATELY (no
    ≤20s window where its live frames misroute into the main chat)."""
    import supervisor.message_bus as mbus
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    broadcasts = []
    fake_bridge = types.SimpleNamespace(broadcast=lambda payload: broadcasts.append(payload))
    monkeypatch.setattr(mbus, "get_bridge", lambda: fake_bridge)
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: None,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "pc1",
        "objective": "Build it",
        "project_id": "proj-x",
        "chat_id": 0,
    }, ctx)

    from ouroboros.contracts.chat_id_policy import project_chat_id

    changed = [b for b in broadcasts if b.get("type") == "projects_changed"]
    assert len(changed) == 1
    assert changed[0]["project_id"] == "proj-x"
    assert changed[0]["chat_id"] == project_chat_id("proj-x")


def test_reserved_project_chat_ids_recognizes_every_project(tmp_path):
    """The isolation SSOT recognizes EVERY registered project's chat_id (regardless
    of sidebar visibility) so its raw chat never re-leaks into the штаб's main
    context / dialogue consolidation / background consciousness (BIBLE P1). Sidebar
    visibility is a separate presentation concern (no project statuses, v6.33.0)."""
    from ouroboros.projects_registry import (
        create_project,
        reserved_project_chat_ids,
        update_project,
    )

    proj = create_project(tmp_path, "old-racer")
    chat_id = int(proj["chat_id"])
    assert chat_id in reserved_project_chat_ids(tmp_path)
    # A rename (or any mutable-field update) never drops it from the isolation set.
    update_project(tmp_path, "old-racer", name="Old Racer (renamed)")
    assert chat_id in reserved_project_chat_ids(tmp_path)


def test_chat_history_tool_spans_all_threads_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the chat_history TOOL is the one mind's
    DELIBERATE recall — it spans the WHOLE conversation (main + ALL project
    threads), only A2A virtual transport excluded. Project-task FOCUS lives in the
    passive default context (the memory view's room), NOT in this recall tool, so
    the one identity can recall anything it chooses (BIBLE P1)."""
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    a = create_project(tmp_path, "alpha")
    b = create_project(tmp_path, "beta")
    ca, cb = int(a["chat_id"]), int(b["chat_id"])
    rows = [
        {"direction": "in", "text": "main-msg", "chat_id": 1},
        {"direction": "in", "text": "alpha-msg", "chat_id": ca},
        {"direction": "in", "text": "beta-msg", "chat_id": cb},
        {"direction": "in", "text": "a2a-noise", "chat_id": -1001},
    ]
    for row in rows:
        dialogue_store.append(tmp_path, "chat", row)
    mem = Memory(drive_root=tmp_path)

    view = mem.chat_history(count=50)
    assert "main-msg" in view and "alpha-msg" in view and "beta-msg" in view  # all threads
    assert "a2a-noise" not in view  # only A2A virtual transport excluded


def test_chat_history_tool_uses_canonical_budget_root_and_archive_pagination(tmp_path):
    from types import SimpleNamespace

    from ouroboros.tools.control import _chat_history

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    child.mkdir()
    dialogue_test.seed(
        canonical,
        archives={"chat_20260820T010000.jsonl": [
            {"direction": "in", "text": f"old-{i}"} for i in range(3)
        ]},
        live={"chat": [{"direction": "in", "text": "new-live"}]},
    )
    dialogue_test.import_all(canonical)
    ctx = SimpleNamespace(
        drive_root=child,
        budget_drive_root=str(canonical),
        task_metadata={},
    )

    first = _chat_history(ctx, count=2)
    second = _chat_history(ctx, count=2, offset=2)

    assert "new-live" in first and "old-2" in first
    assert "Continue with offset=2" in first
    assert "old-0" in second and "old-1" in second


def test_memory_view_main_names_the_bound_room_and_the_project_task_sees_its_rows(tmp_path):
    """Main keeps its words and aggregates the quiet Project with an exact room directory;
    the bound task reads its own room, with original rows available through memory_read."""
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.tools.chronicle import _memory_read
    from tests._memory_view_context import blocks, section
    from tests.test_cache_optimization import _make_env_and_memory

    env, mem = _make_env_and_memory(tmp_path)
    root = mem.drive_root
    proj = create_project(root, "promoted", name="Promoted")
    pchat = int(proj["chat_id"])
    bind_task_to_project(root, "task-7", "promoted", pchat, origin={"absent": "system"})
    dialogue_store.append(root, "chat", {"direction": "in", "text": "plain-main", "chat_id": 1, "ts": "2026-09-01T00:00:00+00:00"})
    dialogue_store.append(root, "chat", {"direction": "out", "text": "bound-task-row", "chat_id": 1,
                                        "task_id": "task-7", "ts": "2026-09-01T00:01:00+00:00"})

    _a, _b, main_ctx, _cap = blocks(env, mem, {"id": "main-turn", "chat_id": 1})
    assert "] plain-main" in section(main_ctx, "## This room (Main)")
    assert "bound-task-row" not in main_ctx
    live = section(main_ctx, "## Live rooms")
    assert "1 open rooms;" in live and "memory_read(room_id='*', start=0)" in live
    assert f"Project Promoted [chat_id={pchat}]" not in live
    reader = types.SimpleNamespace(drive_root=root)
    assert f"- Project Promoted [chat_id={pchat}];" in _memory_read(reader, room_id="*", start=0)

    _a, _b, proj_ctx, _cap = blocks(env, mem, {"id": "task-7", "chat_id": 1})  # bound: its room is the Project
    room = section(proj_ctx, f"## This room (Project Promoted [chat_id={pchat}])")
    # A pre-lineage outgoing row is a host fact of its task, with its address one read away.
    assert "; host; task task-7] " in room and "get_task_result(task_id='task-7')" in room
    assert "plain-main" not in proj_ctx
    history = _memory_read(reader, room_id=str(pchat), rows=True)
    assert "bound-task-row" in history and "plain-main" not in history


def test_restart_drain_defers_then_completes_without_sleeping(tmp_path, monkeypatch):
    """The drain must NOT sleep on the supervisor thread: a restart with live
    tasks defers (returns immediately), and a later loop-tick check completes
    it once tasks drain or the deadline passes."""
    import types

    import server

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    performed = []
    monkeypatch.setattr(server, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()
    # The drain path records its admission intent in the module-global
    # _AGENT_RESTART; a leftover "ready" record from a sibling test makes
    # agent_restart_pending() True and short-circuits the drain before it
    # reaches the live-task check. Reset it so each drain test is isolated.
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})

    now = __import__("time").time()
    ctx = types.SimpleNamespace(
        RUNNING={"t1": {"task": {"id": "t1"}, "last_heartbeat_at": now}},
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )

    # Live task -> defer, do NOT restart inline.
    server._handle_restart_in_supervisor({"reason": "evolution"}, ctx)
    assert performed == []
    assert server._pending_restart  # recorded for the loop tick

    # Tick while still live + before deadline -> keep waiting.
    server._check_pending_restart_drain(ctx)
    assert performed == []

    # Task drained -> the next tick completes the restart.
    ctx.RUNNING = {}
    server._check_pending_restart_drain(ctx)
    assert performed == [True]
    assert not server._pending_restart


def test_restart_drain_no_live_tasks_restarts_immediately(tmp_path, monkeypatch):
    import types

    import server

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    performed = []
    monkeypatch.setattr(server, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})

    ctx = types.SimpleNamespace(
        RUNNING={},
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )
    server._handle_restart_in_supervisor({"reason": "x"}, ctx)
    assert performed == [True]
    assert not server._pending_restart


def test_restart_drain_uses_generic_queue_heartbeat_not_retired_planning_knob(
    tmp_path, monkeypatch
):
    """A stale generic RUNNING heartbeat must not defer restart, even when a
    legacy process environment still carries the removed planning-scout knob."""
    import time
    import types

    import server
    from supervisor.queue import HEARTBEAT_STALE_SEC

    monkeypatch.setenv("OUROBOROS_RESTART_DRAIN_MAX_SEC", "120")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC", "999999")
    performed = []
    monkeypatch.setattr(server, "_perform_supervisor_restart", lambda ctx, **kw: performed.append(True))
    server._pending_restart.clear()
    from ouroboros import server_restart
    monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})

    ctx = types.SimpleNamespace(
        RUNNING={
            "stale": {
                "task": {"id": "stale"},
                "last_heartbeat_at": time.time() - HEARTBEAT_STALE_SEC - 1,
            }
        },
        load_state=lambda: {"owner_chat_id": 0},
        send_with_budget=lambda *a, **k: None,
        DRIVE_ROOT=tmp_path,
    )

    server._handle_restart_in_supervisor({"reason": "x"}, ctx)

    assert performed == [True]
    assert not server._pending_restart


def test_direct_chat_project_thread_skips_letters_home(tmp_path, monkeypatch):
    """A project-thread CONVERSATION (direct chat) is project-scoped for context
    only: it must not block on post-processing or write journal/digest."""
    from ouroboros.project_lease import running_project_ids

    # Sanity: a direct-chat task is never a lease occupant (no project lane),
    # and _is_direct_chat tasks are excluded from letters-home by the pipeline.
    direct = {"id": "d1", "type": "task", "project_id": "racer", "_is_direct_chat": True}
    # The lease only counts top-level project tasks; a direct-chat task still
    # carries project_id but the pipeline gates letters-home on _is_direct_chat.
    assert running_project_ids([{"task": direct}]) == {"racer"}  # context scope is real
    # (full pipeline gating is covered by the agent_task_pipeline branch; this
    # pins the flag the branch reads.)
    assert direct.get("_is_direct_chat") is True


def test_route_project_chat_ignores_non_registered_chat_ids(tmp_path):
    """External-transport chat ids (large, non-project) must NOT be captured as
    project threads — only registered project chat_ids route to a task mailbox."""
    import types

    import server
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    transport_chat = 987654321  # Telegram-style id, NOT a project

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "tp": {"task": {"id": "tp", "chat_id": transport_chat}, "last_heartbeat_at": 1.0},
            "pr": {"task": {"id": "pr", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
        },
    )
    # Transport chat: not a project -> never routed (main free lane preserved).
    assert server._route_project_chat_to_running_task(ctx, transport_chat, "hi") == ""
    # Registered project chat with an active task -> routed to its mailbox.
    assert server._route_project_chat_to_running_task(ctx, project_chat, "steer") == "pr"


def test_route_project_chat_defers_when_multiple_running_tasks(tmp_path, monkeypatch):
    """v6.34.0 WS1/P5: with MORE THAN ONE steerable task in a project room, choosing a
    target is a routing JUDGMENT — code must NOT mechanically steer the first of several.
    The pre-LLM delivery returns "" (the message reaches the decision turn, where the
    agent picks via steer_task) and nothing is mechanically written to a mailbox."""
    import types

    import server
    import ouroboros.owner_mailbox as omb
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])

    delivered = []
    monkeypatch.setattr(
        omb, "write_owner_message", lambda *a, **k: delivered.append(a) or True
    )

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "a": {"task": {"id": "a", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
            "b": {"task": {"id": "b", "chat_id": project_chat}, "last_heartbeat_at": 1.0},
        },
    )
    assert server._route_project_chat_to_running_task(ctx, project_chat, "which one?") == ""
    assert delivered == []  # no mechanical first-of-N steer


def test_route_project_chat_1to1_delivery_is_idempotent(tmp_path, monkeypatch):
    """The 1:1 project-room auto-delivery derives a STABLE msg_id from client_message_id,
    so a WebSocket retry can't double-deliver (drain_owner_entries dedups by msg_id) —
    matching steer_task's idempotency contract."""
    import types

    import server
    import ouroboros.owner_mailbox as omb
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])

    msg_ids = []
    monkeypatch.setattr(omb, "write_owner_message",
                        lambda drive, text, tid, msg_id=None, **k: msg_ids.append(msg_id) or True)

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"pr": {"task": {"id": "pr", "chat_id": project_chat}, "last_heartbeat_at": 1.0}},
    )
    # Same client_message_id retried twice -> identical stable msg_id (dedup), not None (random).
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    server._route_project_chat_to_running_task(ctx, project_chat, "go", "cmid-7")
    assert msg_ids == ["cmid-7:pr", "cmid-7:pr"]


def test_route_project_chat_does_not_confirm_failed_mailbox_write(tmp_path, monkeypatch):
    import types

    import ouroboros.owner_mailbox as omb
    import server
    from ouroboros.projects_registry import create_project

    project_chat = int(create_project(tmp_path, "racer")["chat_id"])
    monkeypatch.setattr(omb, "write_owner_message", lambda *_a, **_k: False)
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "pr": {
                "task": {"id": "pr", "chat_id": project_chat},
                "last_heartbeat_at": 1.0,
            }
        },
    )

    assert (
        server._route_project_chat_to_running_task(
            ctx, project_chat, "must be durable", "owner-msg"
        )
        == ""
    )


def test_busy_project_chat_runs_native_with_routing_context(tmp_path, monkeypatch):
    """WS1/P5 (v6.34.0): a busy PROJECT chat is NOT mechanically auto-enqueued into a
    duplicate pooled task. It runs a native conversation turn (project-scoped, seeing
    current_chat.running_tasks) so the one mind decides steer_task / answer / promote by
    judgment — replacing the old 'Hybrid B+' auto-enqueue fallback."""
    import threading as _threading

    import server
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "market-research")
    project_chat = int(proj["chat_id"])
    enqueued = []
    direct_calls = []
    called = _threading.Event()

    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *a, **k: None)

    class _Bridge:
        def get_updates(self, offset=0, timeout=0):
            return [{
                "update_id": offset,
                "message": {
                    "chat": {"id": project_chat},
                    "from": {"id": 1},
                    "text": "сколько будет 2+2?",
                    "source": "web",
                    "task_metadata": {"project_id": "market-research"},
                },
            }]

    class _Consciousness:
        def inject_observation(self, _text):
            return None

        def pause(self):
            pass

        def resume(self):
            pass

    def _direct(cid, text, image_data, *, task_constraint=None, task_metadata=None):
        direct_calls.append({"chat_id": cid, "text": text, "metadata": task_metadata})
        called.set()

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=True),
        handle_chat_direct=_direct,
        enqueue_task=lambda task: enqueued.append(task),
        send_with_budget=lambda *a, **k: None,
    )

    assert server._process_bridge_updates(_Bridge(), 0, ctx) == 1
    assert called.wait(timeout=3)  # the native conversation turn ran on its own thread
    assert enqueued == []  # NOT auto-enqueued into a duplicate pooled task
    assert len(direct_calls) == 1
    md = direct_calls[0]["metadata"] or {}
    assert str(md.get("project_id") or "")  # project-scoped decision turn
    assert "сколько будет 2+2?" in (direct_calls[0]["text"] or "")


def test_project_from_task_endpoint_creates_binding(tmp_path):
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project, project_binding_for_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "abc123", "id": "task-abc123", "name": "Research thread"}

    resp = asyncio.run(api_project_from_task(_Req()))
    payload = json.loads(resp.body)
    assert resp.status_code == 200
    assert payload["project"]["id"] == "task-abc123"
    assert payload["project"]["name"] == "Research thread"
    assert payload["binding"]["task_id"] == "abc123"
    assert get_project(tmp_path, "task-abc123") is not None
    assert project_binding_for_task(tmp_path, "abc123")["project_id"] == "task-abc123"


def test_project_from_task_auto_names_from_objective(tmp_path):
    """One-click convert (owner P1): with NO name supplied the project name is
    derived from the task's own objective, not the live progress headline, and
    long objectives are collapsed/truncated. No human input, no extra LLM call."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    long_objective = "Собрать конкурентный обзор   рынка облачных GPU\nи свести в таблицу за квартал"
    write_task_result(tmp_path, "obj01", STATUS_RUNNING, objective=long_objective)

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "obj01", "id": "task-obj01"}  # no name → derive

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert "\n" not in name and "  " not in name  # whitespace collapsed
    assert name.startswith("Собрать конкурентный обзор")
    assert len(name) <= 60
    assert name != "task-obj01"  # not the bare id fallback
    assert get_project(tmp_path, "task-obj01")["name"] == name


def test_project_from_task_reuses_explicit_title_without_inline_light_call(tmp_path, monkeypatch):
    """An already-authored title is the Project name; conversion must not buy a
    second naming request while a complete human title already exists."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(
        tmp_path,
        "named01",
        STATUS_RUNNING,
        title="Ракетный стенд 🚀",
        suggested_name="Ненужный запасной вариант",
        objective="Длинное описание задачи",
    )

    async def _forbidden(*_args, **_kwargs):
        raise AssertionError("LIGHT naming must not run for an explicit task title")

    import ouroboros.project_naming as naming
    monkeypatch.setattr(naming, "llm_project_name_async", _forbidden)

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(
                state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
            )

        async def json(self):
            return {"task_id": "named01", "id": "task-named01"}

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    assert payload["project"]["name"] == "Ракетный стенд 🚀"


def test_project_from_task_uses_neutral_name_when_nothing_derivable(tmp_path):
    """Nothing derivable (no title/objective/description) → a NEUTRAL 'New project'
    name, never the bare task id (the owner explicitly rejects task-… names)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "noobj", "id": "task-noobj"}

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    assert payload["project"]["name"] == "New project"
    assert payload["project"]["name"] != "task-noobj"


def test_project_from_task_names_skill_lifecycle_task(tmp_path):
    """A skill-lifecycle (non-human-text) task carries no owner request, so naming
    derives a human label from the synthetic skill_lifecycle_<kind>_<target>_<job>
    id instead of dead-ending at 'New project' (P1)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(
                state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
            )

        async def json(self):
            return {
                "task_id": "skill_lifecycle_install_travel-planner-notion-ai-obsidian_job1",
                "id": "task-skl1",
            }

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name != "New project"
    assert name.startswith("Install skill")
    assert "travel-planner" in name
    assert len(name) <= 60


def test_project_from_task_uses_objective_hint_for_in_progress_direct_chat(tmp_path):
    """A still in-progress DIRECT chat task has no server-side title/objective/queue
    source, so the frontend's objective_hint (the owner's original request) names
    the project — not 'New project' or the bare id (P1, scope-review fix)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "live9", "id": "task-live9",
                    "objective_hint": "исследуй рынок облачных GPU и собери таблицу"}

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name.startswith("исследуй рынок облачных GPU")
    assert name not in ("New project", "task-live9")
    assert len(name) <= 60


def test_project_from_task_auto_names_from_live_queue_snapshot(tmp_path):
    """An in-progress conversion (no task_result objective written yet) derives the
    name from the LIVE queue snapshot, not the bare task id (F1 — fixes the observed
    task-id fallback when converting a still-running card)."""
    import asyncio
    import json

    from ouroboros.gateway.projects import api_project_from_task

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").write_text(
        json.dumps({
            "running": [{"id": "live01", "task": {"id": "live01", "objective": "Изучить рынок облачных GPU и собрать таблицу"}}],
            "pending": [],
        }),
        encoding="utf-8",
    )

    class _Req:
        def __init__(self):
            self.app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path))

        async def json(self):
            return {"task_id": "live01", "id": "task-live01"}  # no name, no task_result

    payload = json.loads(asyncio.run(api_project_from_task(_Req())).body)
    name = payload["project"]["name"]
    assert name.startswith("Изучить рынок облачных GPU")
    assert name != "task-live01"  # not the bare id fallback


def test_all_task_project_bindings_exposes_project_id(tmp_path):
    """F4: the richer binding map carries project_id (not just chat_id) so a bound
    main-chat card can render a pointer that opens the bound project's panel."""
    from ouroboros.projects_registry import (
        all_task_project_bindings,
        bind_task_to_project,
        create_project,
    )

    proj = create_project(tmp_path, "market-thread", name="Market thread")
    bind_task_to_project(tmp_path, "tk9", "market-thread", proj["chat_id"], origin={"absent": "system"})
    mapping = all_task_project_bindings(tmp_path)
    assert mapping["tk9"]["project_id"] == "market-thread"
    assert mapping["tk9"]["chat_id"] == int(proj["chat_id"])


def test_bound_project_history_backfills_task_progress(tmp_path):
    """A task converted into a project after it started keeps its original log
    rows, but project history resolves them through the binding."""
    import asyncio
    import json

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "bound-progress", name="Bound progress")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-1", "bound-progress", project_chat, origin={"absent": "system"})
    dialogue_store.append(tmp_path, "chat", {
        "ts": "2026-01-01T00:00:00Z", "direction": "out", "text": "final answer",
        "chat_id": 1, "task_id": "task-1",
    })
    dialogue_store.append(tmp_path, "chat", {
        "ts": "2026-01-01T00:00:01Z", "direction": "in", "text": "raw project chat",
        "chat_id": project_chat,
    })
    dialogue_store.append(tmp_path, "progress", {
        "ts": "2026-01-01T00:00:02Z", "type": "send_message", "content": "working",
        "text": "working", "is_progress": True, "chat_id": 1, "task_id": "task-1",
        "format": "markdown", "cancelable": True,
    })

    api = make_chat_history_endpoint(tmp_path)

    class _Req:
        def __init__(self, params):
            self.query_params = params

    project_resp = json.loads(asyncio.run(api(_Req({"chat_id": str(project_chat)}))).body)
    project_texts = [m["text"] for m in project_resp["messages"]]
    assert "final answer" in project_texts
    assert "working" in project_texts
    assert "raw project chat" in project_texts
    project_progress = next(m for m in project_resp["messages"] if m["text"] == "working")
    assert "project_mirror" not in project_progress

    main_resp = json.loads(asyncio.run(api(_Req({}))).body)
    main_texts = [m["text"] for m in main_resp["messages"]]
    assert "working" not in main_texts
    assert "raw project chat" not in main_texts
    # The bound task's RAW final-answer row (still stored with main chat_id 1) is
    # project-owned via the binding and must NOT leak into the штаб's main history.
    assert "final answer" not in main_texts


def test_bound_task_heartbeat_routes_to_project_panel(tmp_path):
    """A post-hoc bound task's heartbeat routes to its PROJECT panel: the durable
    binding takes PRECEDENCE over the task's original (main) chat_id, matching the
    send_message/log handlers (UI routing for a "Turn into project" running task)."""
    import time

    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events import _handle_task_heartbeat

    project = create_project(tmp_path, "hb-proj")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-hb", "hb-proj", project_chat, origin={"absent": "system"})

    pushed = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"task-hb": {"task": {"id": "task-hb", "type": "task", "chat_id": 1}, "started_at": time.time()}},
        bridge=types.SimpleNamespace(push_log=lambda payload: pushed.append(payload)),
    )
    _handle_task_heartbeat({"task_id": "task-hb", "phase": "running"}, ctx)
    assert pushed
    assert pushed[0]["chat_id"] == project_chat  # binding precedence, not the original main 1


def test_bound_task_media_routes_to_project_panel(tmp_path):
    """A post-hoc bound task's media (send_photo/send_video) routes to its PROJECT
    panel via the durable binding, not the task's original (main) chat_id —
    same precedence as the send_message/log/heartbeat handlers."""
    import base64

    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events_chat_delivery import _handle_send_photo, _handle_send_video

    project = create_project(tmp_path, "media-proj")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-m", "media-proj", project_chat, origin={"absent": "system"})

    photo_sent, video_sent = [], []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda *a, **k: None,
        bridge=types.SimpleNamespace(
            send_photo=lambda cid, data, caption="", mime="", task_id="": (photo_sent.append((cid, task_id)) or (True, "")),
            send_video=lambda cid, data, caption="", mime="", task_id="": (video_sent.append((cid, task_id)) or (True, "")),
        ),
    )
    blob = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
    _handle_send_photo({"task_id": "task-m", "chat_id": 1, "image_base64": blob, "mime": "image/png"}, ctx)
    _handle_send_video({"task_id": "task-m", "chat_id": 1, "video_base64": blob, "mime": "video/mp4"}, ctx)
    assert photo_sent == [(project_chat, "task-m")]  # binding precedence, not the original main 1
    assert video_sent == [(project_chat, "task-m")]


def test_bound_task_send_message_routes_future_events_to_project(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from supervisor.events import _handle_send_message

    project = create_project(tmp_path, "future-events")
    project_chat = int(project["chat_id"])
    bind_task_to_project(tmp_path, "task-9", "future-events", project_chat, origin={"absent": "system"})
    sent = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
        append_jsonl=lambda *a, **k: None,
    )
    _handle_send_message({
        "chat_id": 1,
        "task_id": "task-9",
        "text": "future progress",
        "is_progress": True,
        "format": "markdown",
    }, ctx)
    assert sent
    assert sent[0][0][0] == project_chat


def test_chat_history_filters_by_thread(tmp_path):
    """api_chat_history returns only the requested thread's rows."""
    import asyncio
    import json

    from ouroboros.gateway.history import make_chat_history_endpoint

    # Register a project so its chat_id partitions out of the main view; a
    # large NON-project chat_id (transport mirror) must STAY in the main view.
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "racer")
    project_chat = int(proj["chat_id"])
    transport_chat = 555000111

    rows = [
        {"ts": "2026-06-13T00:00:01Z", "direction": "in", "text": "main hello", "chat_id": 1},
        {"ts": "2026-06-13T00:00:02Z", "direction": "out", "text": "main reply", "chat_id": 1},
        {"ts": "2026-06-13T00:00:03Z", "direction": "in", "text": "project hello", "chat_id": project_chat},
        {"ts": "2026-06-13T00:00:033Z", "direction": "system", "type": "task_summary", "text": "project summary", "chat_id": project_chat, "task_id": "pt"},
        {"ts": "2026-06-13T00:00:035Z", "direction": "in", "text": "transport mirror", "chat_id": transport_chat},
        {"ts": "2026-06-13T00:00:04Z", "direction": "out", "text": "a2a noise", "chat_id": -1001},
        {"ts": "2026-06-13T00:00:05Z", "direction": "out", "text": "legacy row (no chat_id)"},
    ]
    for row in rows:
        dialogue_store.append(tmp_path, "chat", row)

    api = make_chat_history_endpoint(tmp_path)

    class _Req:
        def __init__(self, params):
            self.query_params = params

    main = json.loads(asyncio.run(api(_Req({}))).body)
    main_texts = [m["text"] for m in main["messages"]]
    assert "main hello" in main_texts and "main reply" in main_texts
    assert "legacy row (no chat_id)" in main_texts  # legacy rows are main-chat
    assert "transport mirror" in main_texts  # non-project transport stays visible
    assert "project hello" not in main_texts  # registered project partitions out
    assert "project summary" not in main_texts
    assert "a2a noise" not in main_texts

    proj_resp = json.loads(asyncio.run(api(_Req({"chat_id": str(project_chat)}))).body)
    proj_texts = [m["text"] for m in proj_resp["messages"]]
    assert proj_texts and "project hello" in proj_texts
    assert "project summary" in proj_texts
    assert "main hello" not in proj_texts
    assert "transport mirror" not in proj_texts


def test_project_media_and_typing_broadcasts_carry_chat_id():
    """Photo/video/typing WS frames must carry chat_id so the client fan-out
    routes project-thread media to its panel (default-to-main would hide them)."""
    from supervisor.message_bus import LocalChatBridge

    bridge = LocalChatBridge()
    frames = []
    bridge._broadcast_fn = lambda payload: frames.append(payload)

    project_chat = 1234  # positive project-range id (not A2A, which is negative)
    bridge.send_chat_action(project_chat, "typing")
    bridge.send_photo(project_chat, b"img-bytes", caption="shot")
    bridge.send_video(project_chat, b"vid-bytes", caption="clip", mime="video/mp4")

    by_type = {f.get("type"): f for f in frames}
    assert by_type["typing"]["chat_id"] == project_chat
    assert by_type["photo"]["chat_id"] == project_chat
    assert by_type["video"]["chat_id"] == project_chat


def test_journal_write_rejects_over_limit_instead_of_truncating(tmp_path, monkeypatch):
    """A durable journal entry is never silently sliced: over-limit writes are
    rejected (the workpad_write contract), so cognitive memory stays whole."""
    import types

    # Project store paths resolve via config.DATA_DIR (NOT ctx.drive_root); isolate
    # it to tmp_path so a plain local pytest run (no OUROBOROS_DATA_DIR set) never
    # writes into the real data dir.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.tools.project_journal import _MAX_TEXT_CHARS, _journal_read, _journal_write

    ctx = types.SimpleNamespace(project_id="journal-reject-test", task_id="t1", drive_root=tmp_path)
    assert _journal_write(ctx, "note", "hello milestone", "").startswith("OK:")
    over = _journal_write(ctx, "note", "Z" * (_MAX_TEXT_CHARS + 50), "")
    assert "TOOL_ARG_ERROR" in over and "exceeds" in over
    body = _journal_read(ctx, "", 30)
    assert "hello milestone" in body
    assert "Z" * 200 not in body  # the rejected over-limit text was never stored


# --- WS1: multi-task chat steering (steer_task + current_chat.running_tasks) ---

def test_steer_task_tool_emits_event_with_target_and_client_id(tmp_path):
    """The agent's steer_task choice emits a transport event (target + message +
    chat + originating message id); the supervisor performs the actual delivery."""
    from ouroboros.tools.control import _tell_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1,
        drive_root=tmp_path, is_direct_chat=True,
        task_metadata={"client_message_id": "cm-42",
                       "origin_message_ref": {"chat_id": 1, "client_message_id": "cm-42"}},
    )
    out = _tell_task(ctx, "abc12345", "also add the benchmarks slide")
    assert out.startswith("TELL_UNCONFIRMED")
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "steer_task"
    assert evt["target_task_id"] == "abc12345"
    assert evt["message"] == "also add the benchmarks slide"
    assert evt["chat_id"] == 1
    assert evt["client_message_id"].startswith("agent-steer:")
    assert evt["quotes"] == []
    # The host-minted issuer fact replaces the routing-contract lane flag: a
    # chat turn is an owner turn, and the lane is the supervisor's registry answer.
    assert evt["issuer"] == {"kind": "owner_turn", "task_id": ""}
    assert "allow_global_root" not in evt
    assert ctx._typed_routing_action_emitted == "steer_task"


def test_tell_keeps_note_without_implicit_origin_substitution(tmp_path):
    from ouroboros.tools.control import _tell_task

    events = []
    exact = "  exact owner text\nwith trailing space  "
    ctx = types.SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        is_direct_chat=True,
        task_metadata={
            "client_message_id": "cm-exact",
            "origin_message_text": exact,
            "origin_message_ref": {"chat_id": 1, "client_message_id": "cm-exact"},
        },
    )

    _tell_task(ctx, "target-task", "model paraphrase")

    assert events[0]["message"] == "model paraphrase"
    assert events[0]["quotes"] == []


def test_main_steer_can_address_project_bound_root_from_host_manifest(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _tell_task
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    emitted = []
    tool_ctx = types.SimpleNamespace(
        pending_events=emitted,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        is_direct_chat=True,
        task_metadata={
            "client_message_id": "main-42",
            "origin_message_ref": {"chat_id": 1, "client_message_id": "main-42"},
            "routing_contract": {"source_lane": "main"},
        },
    )
    _tell_task(tool_ctx, "project-root", "continue from Main")
    assert emitted[0]["issuer"] == {"kind": "owner_turn", "task_id": ""}

    supervisor_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "project-root": {
                "task": {"id": "project-root", "chat_id": 42, "project_id": "racer"},
                "started_at": 1.0,
            },
        },
    )
    _handle_steer_task(emitted[0], supervisor_ctx)
    assert [e["text"] for e in drain_owner_entries(tmp_path, "project-root")] == ["continue from Main"]


def test_busy_direct_main_root_is_manifested_and_steerable_without_promotion(tmp_path):
    import threading

    import server
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools.control import _tell_task
    from supervisor.events import _handle_steer_task

    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-root",
        _current_chat_id=1,
        _current_task_text="Build the AIRI research report",
        _current_task_metadata={"client_message_id": "initial-1"},
        _task_started_ts=10.0,
    )
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(
        direct_agent._current_task_id, direct_agent._current_chat_id, actor=direct_agent,
    )
    routing_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
    )
    metadata = server._decision_turn_metadata(routing_ctx, 1, "followup-1", {})
    root = metadata["main_routing_manifest"]["root_tasks"][0]
    assert root["task_id"] == "direct-root"
    assert root["direct_chat"] is True
    assert root["objective"] == "Build the AIRI research report"

    emitted = []
    tool_ctx = types.SimpleNamespace(
        pending_events=emitted,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        is_direct_chat=True,
        task_metadata={
            "client_message_id": "followup-1",
            "origin_message_ref": {"chat_id": 1, "client_message_id": "followup-1"},
            "routing_contract": metadata["routing_contract"],
        },
    )
    _tell_task(tool_ctx, "direct-root", "Use FusionBrain images too")
    assert [event["type"] for event in emitted] == ["steer_task"]

    event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
    )
    _handle_steer_task(emitted[0], event_ctx)
    assert [e["text"] for e in drain_owner_entries(tmp_path, "direct-root")] == ["Use FusionBrain images too"]


def test_direct_turn_closed_admission_returns_manual_target(tmp_path):
    import threading

    from supervisor.events import _handle_steer_task

    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=False,
        _busy=True,
        _current_task_id="direct-root",
        _current_chat_id=1,
        _current_task_metadata={},
    )
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(
        direct_agent._current_task_id, direct_agent._current_chat_id, actor=direct_agent,
    )
    receipts = []

    class Bridge:
        def send_routing_ack(self, *args, **kwargs):
            receipts.append((args, kwargs))

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
        bridge=Bridge(),
    )
    _handle_steer_task({
        "target_task_id": "direct-root",
        "message": "too late",
        "chat_id": 1,
        "client_message_id": "followup-late",
        "allow_global_root": True,
    }, ctx)
    assert receipts[-1][1]["status"] == "needs_manual_target"


def test_steer_task_tool_requires_args(tmp_path):
    from ouroboros.tools.control import _tell_task

    ctx = types.SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, task_metadata={})
    assert "TOOL_ARG_ERROR" in _tell_task(ctx, "", "msg")
    assert "TOOL_ARG_ERROR" in _tell_task(ctx, "t1", "")
    assert not ctx.pending_events


def test_handle_steer_task_delivers_once_to_running_task(tmp_path, monkeypatch):
    """The handler writes the running task's owner-mailbox on its active drive, and
    a retry with the same client_message_id+target does NOT double-deliver."""
    import supervisor.queue as queue
    from supervisor.events import _handle_steer_task
    from ouroboros.owner_mailbox import drain_owner_entries

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"t1": {"task": {"id": "t1", "chat_id": 1}, "started_at": 1.0}},
        send_with_budget=lambda *a, **k: None,
    )
    evt = {"type": "steer_task", "target_task_id": "t1", "message": "steer me",
           "chat_id": 1, "client_message_id": "cm-1"}
    _handle_steer_task(evt, ctx)
    _handle_steer_task(evt, ctx)  # retry — same client id + target -> stable msg_id
    entries = drain_owner_entries(tmp_path, "t1")  # dedups by msg_id
    assert [e["text"] for e in entries] == ["steer me"]  # delivered exactly once




def test_direct_root_steering_uses_live_human_identity_for_receipt_and_notice(
    tmp_path, monkeypatch,
):
    import supervisor.queue as queue
    from ouroboros.projects_registry import create_project
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    project = create_project(tmp_path, "tower-project", name="Tower Defence")
    attachment = tmp_path / "map.txt"
    attachment.write_text("map", encoding="utf-8")
    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.RLock(),
        _busy=True,
        _accepting_owner_messages=True,
        _current_task_id="direct-live-opaque-id",
        _current_chat_id=project["chat_id"],
        _current_task_metadata={
            "project_id": project["id"],
            "title": "Fix nested delegation",
        },
        _current_task_text="Continue the Tower Defence task",
        _owner_message_generation=0,
    )
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(
        direct_agent._current_task_id, direct_agent._current_chat_id, actor=direct_agent,
    )
    acks = []
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
        bridge=types.SimpleNamespace(
            send_routing_ack=lambda _chat_id, **payload: acks.append(payload),
            send_message=lambda *a, **k: None,
        ),
        send_with_budget=lambda _chat_id, text, *args, **kwargs: notices.append(text),
        persist_queue_snapshot=lambda **_kwargs: None,
    )

    _handle_steer_task(
        {
            "type": "steer_task",
            "target_task_id": "direct-live-opaque-id",
            "message": "Use the attached map",
            "chat_id": project["chat_id"],
            "client_message_id": "cm-direct-title",
            "routing_token": "route-direct-title",
            "allow_global_root": True,
            "attachment_uploads": [{"path": str(attachment), "label": "map"}],
        },
        ctx,
    )

    assert acks[-1]["target"] == "direct-live-opaque-id"
    assert acks[-1]["target_label"] == "Tower Defence › Fix nested delegation"
    assert acks[-1]["status"] == "delivered"
    assert notices == []  # the receipt carries the identity; no staging row in the chat


def test_direct_project_followup_carries_same_live_human_identity(tmp_path):
    import server
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "tower-project", name="Tower Defence")
    attachment = tmp_path / "map.txt"
    attachment.write_text("map", encoding="utf-8")
    direct_agent = types.SimpleNamespace(
        _owner_message_admission_lock=threading.RLock(),
        _busy=True,
        _accepting_owner_messages=True,
        _current_task_id="direct-live-opaque-id",
        _current_chat_id=project["chat_id"],
        _current_task_metadata={
            "project_id": project["id"],
            "title": "Fix nested delegation",
        },
        _current_task_text="Continue the Tower Defence task",
        _owner_message_generation=0,
    )
    from supervisor.active_activity import get_direct_activity_registry

    get_direct_activity_registry().register(
        direct_agent._current_task_id, direct_agent._current_chat_id, actor=direct_agent,
    )
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
        send_with_budget=lambda _chat_id, text, *args, **kwargs: notices.append(text),
    )
    metadata = {
        "project_id": project["id"],
        "chat_attachment_uploads": [{"path": str(attachment), "label": "map"}],
    }

    routed = server._route_project_chat_to_running_task(
        ctx,
        project["chat_id"],
        "Use the attached map",
        "cm-project-direct-title",
        task_metadata=metadata,
    )

    assert routed == "direct-live-opaque-id"
    assert metadata["_routing_target_label"] == "Tower Defence › Fix nested delegation"
    assert [row["status"] for row in metadata["_attachment_manifest"]] == ["staged"]
    assert notices == []  # the receipt recorded from this stash is the whole record


def test_handle_steer_task_stale_target_notifies_visibly(tmp_path, monkeypatch):
    """A target no longer RUNNING (or in another room / a subagent) fails VISIBLY
    with a chat notice and writes NO mailbox — never silently dropped or respawned.
    The issuing chat is a Project ROOM: Main sees the global manifest and may
    address any root, so only a room turn has a "wrong chat" to refuse."""
    import supervisor.queue as queue
    from supervisor.events import _handle_steer_task
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.projects_registry import create_project
    from ouroboros.project_dialogue import routing_refusal_cause

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    room = int(create_project(tmp_path, "issuing-room", name="Issuing Room")["chat_id"])
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "other": {"task": {"id": "other", "chat_id": 999}},  # different chat
            "sub": {"task": {"id": "sub", "chat_id": room, "delegation_role": "subagent"}},
        },
        send_with_budget=lambda cid, text, **kwargs: notices.append((cid, text, kwargs)),
    )
    _handle_steer_task({"target_task_id": "gone", "message": "a", "chat_id": room}, ctx)   # not running
    _handle_steer_task({"target_task_id": "other", "message": "b", "chat_id": room}, ctx)  # wrong chat
    _handle_steer_task({"target_task_id": "sub", "message": "c", "chat_id": room}, ctx)    # subagent
    assert len(notices) == 2
    for (cid, text, kwargs), target, reason in zip(
        notices, ("gone", "sub"), ("target_unknown", "subagent_target"),
    ):
        assert cid == room
        # No bridge handed an ACK over, so this row carries the notice and replies to no owner row.
        assert kwargs == {"role": "system", "system_type": "steer_not_delivered", "task_id": target,
                          "progress_meta": {"routing_notice_carrier": "system", "reply_to": {}}}
        assert routing_refusal_cause("steer_task", "needs_manual_target", reason) in text
        assert "I'll" not in text and "Couldn't steer task" not in text
    assert drain_owner_entries(tmp_path, "gone") == []
    assert [e["text"] for e in drain_owner_entries(tmp_path, "other")] == ["b"]
    assert drain_owner_entries(tmp_path, "sub") == []


def test_chat_running_tasks_lists_pooled_and_restored_direct_roots(tmp_path):
    """The structural snapshot lists the chat's pooled RUNNING root tasks (so the
    decision turn can pick a steer target) and excludes direct/subagent/other-chat."""
    import server

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "a": {"task": {"id": "a", "chat_id": 1, "objective": "build racer"}, "started_at": 1.0},
            "b": {"task": {"id": "b", "chat_id": 1, "title": "Docs", "objective": "write docs"}, "started_at": 2.0},
            "direct": {"task": {"id": "direct", "chat_id": 1, "_is_direct_chat": True}},
            "sub": {"task": {"id": "sub", "chat_id": 1, "delegation_role": "subagent"}},
            "elsewhere": {"task": {"id": "elsewhere", "chat_id": 7}},
        },
    )
    rows = server._chat_running_tasks(ctx, 1)
    assert {r["task_id"] for r in rows} == {"a", "b", "direct"}
    assert all(r["steerable"] for r in rows)
    by_id = {r["task_id"]: r for r in rows}
    assert by_id["a"]["objective"] == "build racer"
    assert by_id["b"]["title"] == "Docs"


def test_decision_turn_metadata_injects_running_tasks_and_client_id(tmp_path):
    """The chat-turn metadata is enriched with current_chat.running_tasks + the
    originating message id, so build_runtime_section can surface them (P5 — state
    only; the agent still chooses)."""
    import server

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"a": {"task": {"id": "a", "chat_id": 1, "objective": "x"}, "started_at": 1.0}},
    )
    md = server._decision_turn_metadata(ctx, 1, "cm-9", {"project_id": "p"})
    assert md["project_id"] == "p"  # preserved
    assert md["client_message_id"] == "cm-9"
    assert md["current_chat"]["chat_id"] == 1
    assert [t["task_id"] for t in md["current_chat"]["running_tasks"]] == ["a"]
    # No running tasks + no client id -> no routing facts at all (the head's
    # context capture builds its own live-works present).
    empty_ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={})
    md = server._decision_turn_metadata(empty_ctx, 1, "", {"k": "v"})
    assert md == {"k": "v"}


# --- Q10=A (owner, 2026-08-08): file-less project promotes auto-provision -----

def _promote_ctx(enqueued, sent=None):
    return types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        # Records every chat send, so a refusal can prove it sent NOTHING.
        send_with_budget=lambda *args, **kwargs: (sent if sent is not None else []).append((args, kwargs)),
    )


def test_promote_fileless_project_autoprovisions_and_binds_workspace(tmp_path, monkeypatch):
    """A project promoted with an EMPTY working_dir gets a genesis workspace via
    the existing ensure_project_workspace seam and the task is BOUND to it
    (external profile, forked memory, lease lane) — the submarine shape fix."""
    import os

    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "fileless1",
        "objective": "Build the submarine game",
        "project_id": "sunken-city",
        "project_name": "Sunken City",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "scheduled"
    task = enqueued[0]
    ws = str(task.get("workspace_root") or "")
    assert ws, "file-less project promote must bind an auto-provisioned workspace"
    projects_root = os.environ["OUROBOROS_SUBAGENT_PROJECTS_ROOT"]
    assert ws.startswith(str(pathlib_resolve(projects_root)))
    assert task["workspace_mode"] == "external"
    assert task["memory_mode"] == "forked"
    assert task["metadata"]["workspace_autoprovisioned"] is True
    assert "[HEADLESS_WORKSPACE]" in task["text"]
    # The registry carries the provisioned working_dir for later waves/promotes.
    assert get_project(tmp_path, "sunken-city")["working_dir"] == ws
    # Idempotency: a second promote reuses the SAME tree (no sunken-city_1 mint).
    enqueued2 = []
    workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "fileless2",
        "objective": "Continue the submarine game",
        "project_id": "sunken-city",
        "chat_id": 1,
    }, _promote_ctx(enqueued2))
    assert enqueued2[0]["workspace_root"] == ws


def pathlib_resolve(p):
    import pathlib

    return pathlib.Path(p).resolve()


def test_promote_workspace_none_still_opts_out_of_autoprovision(tmp_path, monkeypatch):
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "optout1",
        "objective": "Pure research, no folder",
        "project_id": "folderless",
        "workspace": "none",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "scheduled"
    assert not enqueued[0].get("workspace_root")
    # The opt-out means NO provisioning side effect either.
    assert str(get_project(tmp_path, "folderless").get("working_dir") or "") == ""


def test_promote_broken_working_dir_loud_fails_never_blind_ensures(tmp_path, monkeypatch):
    """v6.58.0 invariant preserved: a NON-EMPTY broken working_dir loud-fails;
    auto-provision fires ONLY on the empty string and never papers over a broken
    folder with a fresh empty repo."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project, update_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    create_project(tmp_path, "brokenp", name="BrokenP")
    gone = tmp_path / "gone-folder"
    update_project(tmp_path, "brokenp", working_dir=str(gone))  # never existed

    enqueued, sent = [], []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "broken1",
        "objective": "Continue",
        "project_id": "brokenp",
        "chat_id": 1,
    }, _promote_ctx(enqueued, sent))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_unusable"
    # The repair rides `detail` to the MODEL (PROMOTE_REJECTED renders it);
    # admission itself sends no chat text (Q1=A) and writes no task result —
    # the handler's single rejection writer does.
    assert outcome["detail"].startswith("project 'brokenp' working_dir is unusable")
    assert "Projects → this project" in outcome["detail"] and "workspace='none'" in outcome["detail"]
    assert sent == []
    assert enqueued == []
    # The broken value is preserved for the owner to fix — not overwritten.
    assert get_project(tmp_path, "brokenp")["working_dir"] == str(gone)


def test_promote_provisioning_failure_loud_fails_not_silent_fileless(tmp_path, monkeypatch):
    """Bind-or-fail: if auto-provisioning fails, the promote fails LOUDLY instead
    of silently degrading to a workspace-less self_modification-profile task."""
    import supervisor.workers as workers
    from ouroboros import projects_registry

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(projects_registry, "ensure_project_workspace", lambda *a, **k: "")
    enqueued, sent = [], []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "provfail1",
        "objective": "Build",
        "project_id": "provfail-proj",
        "chat_id": 1,
    }, _promote_ctx(enqueued, sent))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_provisioning_failed"
    assert outcome["detail"].startswith("project 'provfail-proj' has no working folder and auto-provisioning")
    assert "Projects → this project" in outcome["detail"] and "workspace='none'" in outcome["detail"]
    assert sent == []
    assert enqueued == []


def test_swarm_intent_survives_admission_to_the_finalization_read(swarm_host):  # noqa: F811 -- imported fixture
    """Host-admitted Swarm intent survives through the actual efficiency reader."""
    from supervisor.events import _handle_promote_chat_to_task
    from ouroboros.agent_task_pipeline import _build_swarm_efficiency
    from tests.test_swarm_host_admission import incoming_case, promotion_event

    swarm_host.monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", "swarm")
    case = incoming_case(swarm_host, "main")
    _handle_promote_chat_to_task(promotion_event(case), swarm_host.ctx)
    assert swarm_host.attempts == []
    assert len(swarm_host.pending) == 1
    task = swarm_host.pending[0]
    assert "force_plan_source" not in task.get("metadata", {})
    assert task["metadata"]["subagents_mode"] == "swarm"
    swarm_host.monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", "off")  # telemetry keeps admission, runtime reads live
    block = _build_swarm_efficiency(types.SimpleNamespace(drive_root=str(swarm_host.root)), task)
    assert block == {
        "intent_source": "swarm",
        "planned": None,
        "observed_started": 0,
        "status": "no_fanout_observed",
    }


def test_promote_emission_row_carries_the_owner_message_id(tmp_path, monkeypatch):
    """I7: the durable ingress row named the task and the routing token but not the
    owner MESSAGE, so the same message becoming several roots could not be seen in
    the record at all."""
    import json

    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        task_metadata={"client_message_id": "cm-ingress"},
    )

    assert _start_task(
        ctx, "Build the racer prototype", title="Racer prototype", project_id="racer",
    ).startswith("OK: task")

    rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    emitted = next(row for row in rows if row["type"] == "promote_chat_to_task_emitted")
    assert emitted["client_message_id"] == "cm-ingress"
    assert emitted["task_id"] == ctx.pending_events[0]["task_id"]


def _steer_refusal(tmp_path, monkeypatch, *, running: dict, chat_id: int = 1):
    """Drive one refused steer and return (receipt kwargs, owner messages)."""
    import supervisor.events as events_mod
    from supervisor.steering import _handle_steer_task

    receipts: list = []
    sent: list = []
    monkeypatch.setattr(events_mod, "_emit_routing_receipt",
                        lambda ctx, evt, **kwargs: receipts.append(kwargs) or {})
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[],
        get_chat_agent=lambda: None,
        send_with_budget=lambda _chat_id, text, **kwargs: sent.append((text, kwargs)),
    )
    _handle_steer_task(
        {"target_task_id": "target-1", "message": "hurry up", "chat_id": chat_id}, ctx,
    )
    return receipts[-1], sent




@pytest.mark.parametrize("running, reason", [
    ({}, "target_unknown"),
    ({"target-1": {"task": {"id": "target-1", "chat_id": 1, "delegation_role": "subagent",
                            "title": "Review"}}}, "subagent_target"),
])
def test_steer_refusal_keeps_a_distinct_reason_for_every_other_cause(
        tmp_path, monkeypatch, running, reason):
    from ouroboros.project_dialogue import routing_refusal_cause

    receipt, sent = _steer_refusal(tmp_path, monkeypatch, running=running)

    assert receipt["reason"] == reason and receipt["status"] == "needs_manual_target"
    assert len(sent) == 1
    text, kwargs = sent[0]
    assert routing_refusal_cause("steer_task", "needs_manual_target", reason) in text
    assert "I'll" not in text and "Couldn't steer task" not in text
    assert kwargs == {"role": "system", "system_type": "steer_not_delivered", "task_id": "target-1",
                      "progress_meta": {"routing_notice_carrier": "system", "reply_to": {}}}


def test_steer_reaches_a_queue_resident_direct_turn(tmp_path, monkeypatch):
    """A direct turn resumed on a pooled worker under the SAME id after its exact
    budget pause (#1196) is a RUNNING row: the owner's follow-up is delivered to
    that actor's mailbox and is never refused as a finished direct reply."""
    import supervisor.events as events_mod
    from supervisor.steering import _handle_steer_task

    receipts: list = []
    monkeypatch.setattr(events_mod, "_emit_routing_receipt",
                        lambda ctx, evt, **kwargs: receipts.append(kwargs) or {})
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, PENDING=[], get_chat_agent=lambda: None,
        RUNNING={"target-1": {"task": {"id": "target-1", "chat_id": 1, "_is_direct_chat": True,
                                       "title": "Chat"}}},
        send_with_budget=lambda _chat_id, text, **kwargs: None,
    )
    _handle_steer_task({"target_task_id": "target-1", "message": "hurry up", "chat_id": 1}, ctx)

    assert receipts, "a steer must leave a receipt"
    assert receipts[-1].get("reason") != "direct_chat_turn"
    assert receipts[-1].get("status") == "delivered"


def test_the_steer_tool_renders_the_typed_reason_and_still_defaults_without_one(tmp_path, monkeypatch):
    """The fallback belongs to the RENDERER, not to the emitter: every refusal the
    handler emits now carries its own typed reason, and a receipt that carries
    none (another producer, an older row) still renders the documented default."""
    from ouroboros.tools import control_routing, control_relay

    answers = iter([
        ("live", {"status": "needs_manual_target", "reason": "target_finished"}),
        ("live", {"status": "needs_manual_target"}),
    ])
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, _evt: next(answers))
    ctx = types.SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1,
                                task_metadata={}, drive_root=tmp_path)

    assert "target_finished" in control_relay._tell_task(ctx, "t1", "go")
    assert "target_not_steerable" in control_relay._tell_task(ctx, "t1", "go")


def _repair_hint(**kwargs):
    """The MODEL-facing repair for one refused workspace: one line, never empty."""
    from ouroboros.workspace_admission import workspace_repair_hint

    hint = workspace_repair_hint(**kwargs)
    assert hint and "\n" not in hint
    return hint


def test_workspace_repair_hint_follows_the_source_of_the_refused_folder(tmp_path):
    """The request named its own folder, so the project's working folder was never
    read: sending the caller to Projects points at a setting the failure never
    touched. The project folder is named as the way back when it is readable."""
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "roomp", name="RoomP", working_dir=str(tmp_path / "room-tree"))
    hint = _repair_hint(
        ws_error="explicit workspace_root is unusable: not a git checkout.",
        explicit_workspace=str(tmp_path / "asked-for"), project_id="roomp",
        drive_root=tmp_path, system_repo_dir=tmp_path / "system-repo",
    )

    assert hint.startswith("explicit workspace_root is unusable: not a git checkout. ")
    assert "asked for" in hint and str(tmp_path / "asked-for") in hint
    assert "Projects → this project" not in hint
    assert str(tmp_path / "room-tree") in hint
    assert "workspace='none'" in hint


def test_workspace_repair_hint_keeps_the_projects_remedy_for_a_project_folder():
    """A project working_dir failure, an unreadable registry entry and a failed
    auto-provision are fixed exactly where the sentence says: in Projects."""
    for ws_error in (
        "project 'roomp' working_dir is unusable: not a git checkout.",
        "project 'roomp' registry entry is unreadable (OSError: boom) — cannot determine "
        "the task's workspace",
        "project 'roomp' has no working folder and auto-provisioning one failed; "
        "see the supervisor log (ensure_project_workspace)",
    ):
        hint = _repair_hint(ws_error=ws_error)
        assert hint.startswith(ws_error.rstrip("."))
        assert hint.endswith(
            "Fix the project's working folder (Projects → this project) or re-promote with "
            "workspace='none' for a folder-less task."
        )
        assert "asked for" not in hint


def test_workspace_repair_hint_names_a_retired_delegated_run_worktree(tmp_path, monkeypatch):
    """The path the agent passed twice in one minute was a delegated-run worktree
    its own run had already retired; nothing in the old message said so."""
    from ouroboros import config

    worktrees = tmp_path / "subagent_worktrees"
    monkeypatch.setattr(config, "get_subagent_worktree_root", lambda: str(worktrees))
    hint = _repair_hint(
        ws_error="explicit workspace_root is unusable: path does not exist.",
        explicit_workspace=str(worktrees / "dlg_060055d5_x"), project_id="",
    )

    assert "delegated-run worktree" in hint and "removed when its run ends" in hint
    assert "Projects → this project" not in hint and "workspace='none'" in hint


def test_workspace_repair_hint_points_a_repo_subfolder_at_the_empty_default(tmp_path):
    """A SUBFOLDER of the Ouroboros repository can never be a workspace (the exact
    root never reaches admission: the promote tool maps it onto no workspace);
    the repair names the argument shape that works instead."""
    repo = tmp_path / "Ouroboros" / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    hint = _repair_hint(
        ws_error="explicit workspace_root is unusable: workspace_root must not overlap the "
        "Ouroboros system repo",
        explicit_workspace=str(repo / "ouroboros"), system_repo_dir=repo,
    )

    assert "workspace_root must be a folder outside the Ouroboros repository" in hint
    assert "empty (or workspace='none') to work in the repository itself" in hint
    assert "asked for" not in hint


def test_workspace_repair_hint_names_the_presence_profile():
    hint = _repair_hint(
        ws_error="explicit workspace_root is unusable: workspace_root is not a directory: /gone",
        presence=True,
    )

    assert hint.startswith("The folder configured in the Presence profile is unusable: ")
    assert "workspace_root is not a directory: /gone" in hint
    assert hint.endswith("Fix the Presence profile's workspace_root or clear it.")


# --- promote refusals: placement (Q1/Q2=A), the host cause (Q3=A), the repo root (Q4=A) ---

def _host_promote(host, monkeypatch, **overrides):
    """Drive the REAL promote handler (reservation, admission, receipts, the
    publication boundary) for a Main event; ``host.notices`` records every chat
    send with its kwargs and ``host.bridge.acks`` every live receipt."""
    from supervisor.events import _handle_promote_chat_to_task

    evt = {
        "type": "promote_chat_to_task", "task_id": "refused0001", "routing_token": "tok-refused-1",
        "objective": "Audit the GitHub tool", "chat_id": 1, "client_message_id": "cm-refused-1",
        "workspace": "none",
    }
    evt.update(overrides)
    return evt, _handle_promote_chat_to_task(evt, host.ctx)


def test_host_initiated_workspace_refusal_sends_one_typed_row_to_the_owner_chat(swarm_host, monkeypatch):
    """Q2=A: no model turn narrates a host-issued promote, so the owner is told
    ONCE — a typed System row in the chat the owner wrote in (never the project
    room admission just created), bound to the never-started task — and the
    receipt under the message carries the same host sentence (Q3=A)."""
    from tests.test_swarm_host_admission import rows

    host = swarm_host
    announced = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_k: announced.append(dict(event)) or True,
    )
    evt, outcome = _host_promote(
        host, monkeypatch, host_initiated=True, title="Аудит GitHub-инструмента",
        project_id="refused-room", project_name="Refused Room",
        workspace="", workspace_root=str(host.root / "missing-folder"),
    )

    assert (outcome["status"], outcome["reason"]) == ("needs_manual_target", "workspace_unusable")
    assert outcome["detail"].startswith("explicit workspace_root is unusable")
    assert host.pending == []
    [notice] = host.notices
    assert notice["chat_id"] == 1  # the OWNER's chat, not the new project room
    assert notice["text"] == "Аудит GitHub-инструмента · Not started: the working folder can't be used"
    assert notice["role"] == "system" and notice["system_type"] == "task_not_started"
    assert notice["task_id"] == evt["task_id"]
    # R9: the project was created, but its "Started" row is owed only to real work.
    assert announced == []
    row = rows(host.root / "logs/chat_annotations.jsonl")[-1]
    assert (row["status"], row["reason"]) == ("needs_manual_target", "workspace_unusable")
    assert row["cause"] == "Not started: the working folder can't be used"
    assert host.bridge.acks[-1]["status"] == "needs_manual_target"
    assert host.bridge.acks[-1]["cause"] == "Not started: the working folder can't be used"


def test_tool_issued_workspace_refusal_sends_nothing(swarm_host, monkeypatch):
    """Q1=A: a model turn waits on the receipt and narrates; the receipt (with
    its cause) and the failed call are the record — no host chat text at all."""
    host = swarm_host
    evt, outcome = _host_promote(
        host, monkeypatch, workspace="", workspace_root=str(host.root / "missing-folder"),
    )

    assert (outcome["status"], outcome["reason"]) == ("needs_manual_target", "workspace_unusable")
    assert host.notices == []
    assert host.bridge.acks[-1]["cause"] == "Not started: the working folder can't be used"


def test_host_initiated_reservation_refusal_sends_one_row_and_a_replay_sends_nothing(
    swarm_host, monkeypatch,
):
    """R13: the reservation-blocked exit (a disabled worker pool) passes the same
    publication boundary; a replay of the settled admission (same token) is
    silent — the owner was told once. The picker's route (`routed_from_main`)
    wears the same «Not started» prefix as a promote (R14)."""
    import supervisor.workers as workers
    from supervisor.events import _handle_promote_chat_to_task

    host = swarm_host
    monkeypatch.setattr(
        workers, "_worker_pool_execution_state",
        lambda *_a, **_k: {"available": False, "disabled_reason": "maintenance"},
    )
    evt, outcome = _host_promote(host, monkeypatch, host_initiated=True, routed_from_main=True)

    assert (outcome["status"], outcome["reason"]) == ("needs_manual_target", "worker_pool_unavailable")
    [notice] = host.notices
    assert notice["text"] == "Audit the GitHub tool · Not started: no worker is available right now"
    assert notice["system_type"] == "task_not_started" and notice["task_id"] == evt["task_id"]

    replay = _handle_promote_chat_to_task(dict(evt), host.ctx)
    assert replay["replayed"] is True and replay["reason"] == "worker_pool_unavailable"
    assert len(host.notices) == 1


def test_host_initiated_unconfirmed_admission_sends_the_unconfirmed_row(swarm_host, monkeypatch):
    """R14: an admission whose receipt could not be persisted is UNCONFIRMED — the
    row says exactly that (`task_start_unconfirmed`, «Not confirmed»), never
    «Not started»."""
    host = swarm_host
    monkeypatch.setattr("ouroboros.project_dialogue.append_chat_annotation", lambda *_a, **_k: False)
    evt, outcome = _host_promote(host, monkeypatch, host_initiated=True)

    assert (outcome["status"], outcome["reason"]) == ("unconfirmed", "routing_annotation_persist_failed")
    [notice] = host.notices
    assert notice["text"] == "Audit the GitHub tool · Not confirmed: the receipt could not be saved"
    assert notice["role"] == "system" and notice["system_type"] == "task_start_unconfirmed"
    assert notice["task_id"] == evt["task_id"]


def test_promote_maps_the_exact_repo_root_onto_no_workspace(tmp_path, monkeypatch):
    """Q4=A: naming the Ouroboros repository ITSELF names the documented default
    (the ordinary self-modification task): the event carries the existing
    "none" sentinel — in a project room too — and the tool result discloses it."""
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    repo = tmp_path / "Ouroboros" / "repo"
    repo.mkdir(parents=True)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path / "data",
        repo_dir=repo, system_repo_dir=repo,
    )
    out = _start_task(
        ctx, "Audit the GitHub tool", title="GitHub tool audit", project_id="racer", workspace_root=f"{repo}/",

    )

    assert out.startswith("OK: task")
    assert (
        "(workspace_root named the Ouroboros repository itself; started as an ordinary task "
        "over it — no separate workspace)"
    ) in out
    [evt] = events
    assert evt["workspace_root"] == "" and evt["workspace"] == "none"
    assert evt["project_id"] == "racer"


def test_promote_passes_a_repo_subfolder_through_to_admission_unchanged(tmp_path, monkeypatch):
    """A SUBFOLDER is a different ask the shape cannot serve: it reaches admission
    unchanged and is refused there with the repair hint."""
    from ouroboros.tools.control import _start_task

    _confirm_promote(monkeypatch)
    repo = tmp_path / "Ouroboros" / "repo"
    sub = repo / "ouroboros"
    sub.mkdir(parents=True)
    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1, drive_root=tmp_path / "data",
        repo_dir=repo, system_repo_dir=repo,
    )
    out = _start_task(ctx, "Audit the GitHub tool", title="GitHub tool audit",
                                workspace_root=str(sub))

    assert out.startswith("OK: task") and "named the Ouroboros repository itself" not in out
    [evt] = events
    assert evt["workspace_root"] == str(sub) and evt["workspace"] == ""


def test_project_started_is_not_announced_when_workspace_admission_refuses(tmp_path, monkeypatch):
    """R9: the «Project · Started» row is owed only once the task is REALLY in the
    queue — a promote that creates a project and then fails workspace admission
    announces nothing (the project itself stays; the refusal names it)."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_k: queued.append(dict(event)) or True,
    )
    enqueued, sent = [], []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task", "task_id": "doomed0001", "objective": "Build",
        "project_id": "doomed-room", "project_name": "Doomed Room",
        "workspace_root": str(tmp_path / "never-existed"), "chat_id": 1,
    }, _promote_ctx(enqueued, sent))

    assert (outcome["status"], outcome["reason"]) == ("needs_manual_target", "workspace_unusable")
    assert get_project(tmp_path, "doomed-room")["name"] == "Doomed Room"  # the project stays
    assert queued == [] and enqueued == [] and sent == []


def test_host_initiated_refusal_row_names_an_untitled_act_at_a_word_boundary():
    """An untitled host-issued act (a skill-card request has no title) is named by
    its request's first words; a long request is cut at a word boundary with an
    ellipsis, never mid-word, before the « · Not started: …» clause."""
    from supervisor.events_project_routing import _notify_host_initiated_refusal

    sent = []
    ctx = types.SimpleNamespace(send_with_budget=lambda chat, text, **kw: sent.append((chat, text, kw)))
    objective = "Почини скилл stand-missing-skill, его файлы отсутствуют и манифест не читается вообще"
    _notify_host_initiated_refusal(
        ctx,
        {"host_initiated": True, "chat_id": 1, "objective": objective, "task_id": "t1"},
        {"status": "needs_manual_target", "reason": "invalid_skill_repair_constraint", "task_id": "t1"},
    )
    assert len(sent) == 1
    chat, text, kw = sent[0]
    title, _, clause = text.partition(" · ")
    assert title.endswith("…") and not title[:-1].endswith(" ") and len(title) <= 61
    assert objective.startswith(title[:-1]) and objective[len(title) - 1] == " "
    assert clause == "Not started: the skill repair request was invalid"
    assert kw["role"] == "system" and kw["system_type"] == "task_not_started" and kw["task_id"] == "t1"
