"""promote_chat_to_task + project chat routing (multi-project, v6.32.0)."""

from __future__ import annotations

import queue
import pathlib
import threading
import types

import pytest


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    """Q10=A auto-provisions a genesis workspace for file-less project promotes;
    keep it out of the real ~/Ouroboros/projects."""
    monkeypatch.setenv(
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT",
        str(tmp_path_factory.mktemp("projects_root")),
    )


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
    out = _promote_chat_to_task(ctx, "Build the racer prototype", project_id="racer", predecessor_task_id="")
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


def test_cat_router_preview_promote_first_request_and_direct_harness_keep_full_authority(
    tmp_path, monkeypatch,
):
    import json

    import server
    import supervisor.workers as workers
    from ouroboros.agent import Env
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.outcomes import append_verification_receipt
    from ouroboros.projects_registry import create_project
    from ouroboros.subagent_work_order import assignment_instructions, compile_external_work_order
    from ouroboros.tools.control import (
        _build_child_subagent_contract,
        _get_task_result,
        _promote_chat_to_task,
    )
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _confirm_promote(monkeypatch)
    project = create_project(tmp_path, "cat-tower", name="Cat Tower Builder")
    predecessor_id = "cat-old-root"
    tail = "CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3"
    result_tail = "KEEP_EXISTING_TOWER_ASSET; DO_NOT_REBUILD_FROM_SCRATCH"
    artifact_error = "ARTIFACT_CAPTURE_FAILED_AFTER_PARTIAL_COPY"
    artifact_finalized_at = "2026-08-21T00:00:00Z"
    capability_delta = "HARNESS_ROUTE_REDUCED_TO_NATIVE_ONLY"
    delegated_custody = "DELEGATED_RUN_STILL_OPEN"
    verification_ledger = "VERIFICATION_LEDGER_BLOCKER"
    verification_receipt = "CANONICAL_RECEIPT_FAILED"
    future_terminal_fact = "FUTURE_TERMINAL_AUTHORITY_FIELD"
    final_answer = "FINAL_ANSWER_TERMINAL_FACT"
    non_final_rows = "NON_FINAL_ROWS_TERMINAL_FACT"
    mutation_evidence = "MUTATION_EVIDENCE_TERMINAL_FACT"
    plan_review_state = "PLAN_REVIEW_STATE_TERMINAL_FACT"
    process_evidence = (
        "RAW_LOOP_TRANSCRIPT_MUST_NOT_BE_INHERITED",
        "UNRELATED_ROUTING_METADATA_MUST_NOT_BE_INHERITED",
        "RAW_LLM_TRACE_MUST_NOT_BE_INHERITED",
        "RAW_REVIEW_EVIDENCE_MUST_NOT_BE_INHERITED",
        "RAW_REVIEW_PROJECTION_MUST_NOT_BE_INHERITED",
        "RAW_TRACE_REFS_MUST_NOT_BE_INHERITED",
        "RAW_ROOT_PHASE_CHECKPOINT_MUST_NOT_BE_INHERITED",
    )
    excluded_process_fields = (
        "review_evidence", "review_projection", "trace_refs", "root_phase_checkpoint",
    )
    process_padding = "P" * 12_000
    predecessor = {
        "task_id": predecessor_id,
        "status": "cancelled",
        "title": "Cat Tower Builder",
        "objective": "o" * 700 + tail,
        "result": "completed implementation evidence\n" + "r" * 700 + result_tail,
        "artifact_status": "failed",
        "artifact_error": artifact_error,
        "artifact_finalized_at": artifact_finalized_at,
        "capability_delta": {"reduced": True, "detail": capability_delta},
        "delegated_runs_unreconciled": [delegated_custody],
        "verification_ledger": {"entries": [{"evidence": verification_ledger}]},
        "future_terminal_fact": {"detail": future_terminal_fact},
        "final_answer": final_answer,
        "non_final_rows": [{"reason": non_final_rows}],
        "mutation_evidence": {"summary": mutation_evidence},
        "plan_review_state": {
            "schema_version": 2,
            "current_attempt": {"status": "closed", "decision": plan_review_state},
            "waves": [],
        },
        "loop_outcome": {"final_text": process_evidence[0] * 500},
        "metadata": {"main_routing_manifest": {"raw": process_evidence[1]}},
        "llm_trace": {"reasoning_notes": [process_evidence[2]]},
        "review_evidence": {"reasoning_notes": [process_evidence[3] + process_padding]},
        "review_projection": {
            "panels": [{"raw_response": process_evidence[4] + process_padding}],
        },
        "trace_refs": {"tool_log": process_evidence[5] + process_padding},
        "root_phase_checkpoint": {
            "post_task_synthesis": {"raw_output": process_evidence[6] + process_padding},
        },
        "project_id": "cat-tower",
        "task_contract": {
            "objective": "o" * 700 + tail,
            "context": "never use native/API fallback",
            "constraints": "Claudexor harness only",
            "delegation_budget": {"intent_note": "L1 asks L2 to spawn L3"},
        },
    }
    result_dir = tmp_path / "task_results"
    result_dir.mkdir()
    (result_dir / f"{predecessor_id}.json").write_text(
        json.dumps({"_schema_version": 1, **predecessor}), encoding="utf-8",
    )
    assert len(json.dumps(predecessor["plan_review_state"])) < 1_000
    assert len(json.dumps({
        key: predecessor[key]
        for key in (
            "loop_outcome", "metadata", "llm_trace", *excluded_process_fields,
        )
    })) > 40_000
    append_verification_receipt(tmp_path, predecessor_id, {
        "criterion_id": "canonical-receipt", "status": "fail",
        "evidence": verification_receipt,
    })
    preview = server._task_result_ground_truth(predecessor)
    assert preview["objective"].endswith("chars omitted]")
    assert preview["authority_source"]["arguments"] == {
        "task_id": predecessor_id, "include_authority": True,
    }
    router_ctx = _swarm_ctx(
        tmp_path,
        project_id="cat-tower",
        current_chat_id=int(project["chat_id"]),
        task_metadata={
            "force_plan": True,
            "force_plan_source": "swarm",
            "project_last_task_result": preview,
        },
    )

    assert _promote_chat_to_task(
        router_ctx, "Continue the Cat build", predecessor_task_id=predecessor_id,
    ).startswith("OK: task")
    event = router_ctx.pending_events[0]
    assert event["predecessor_authority_source"] == preview["authority_source"]
    fresh_router = _swarm_ctx(
        tmp_path, project_id="cat-tower", current_chat_id=int(project["chat_id"]),
        task_metadata={
            "force_plan": True, "force_plan_source": "swarm",
            "project_last_task_result": preview,
        },
    )
    assert _promote_chat_to_task(fresh_router, "Build a fresh Cat demo", predecessor_task_id="").startswith("OK: task")
    assert "predecessor_authority_source" not in fresh_router.pending_events[0]
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    admitted = []
    outcome = workers.promote_chat_to_task(event, types.SimpleNamespace(
        enqueue_task=lambda task: admitted.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    ))
    assert outcome["status"] == "scheduled"
    task = admitted[0]
    task["budget_drive_root"] = str(tmp_path)
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "prompts" / "SYSTEM.md").write_text("You are Ouroboros.", encoding="utf-8")
    (repo / "BIBLE.md").write_text("# BIBLE", encoding="utf-8")
    (repo / "docs" / "ARCHITECTURE.md").write_text("# Architecture", encoding="utf-8")
    (repo / "docs" / "DEVELOPMENT.md").write_text("# Development", encoding="utf-8")
    env = Env(repo_dir=repo, drive_root=tmp_path, budget_drive_root=tmp_path)

    assert validate_task_authority_sources(env, task) == {}
    attach_task_contract(task)
    assert task["task_contract"]["predecessor_authority"] == task["predecessor_authority"]
    messages, _ = build_llm_messages(env=env, memory=Memory(tmp_path, repo_dir=repo), task=task)
    rendered = json.dumps(messages, ensure_ascii=False)
    tool_ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_contract=task["task_contract"], task_metadata=task["metadata"],
    )
    direct_harness = assignment_instructions(tool_ctx)
    retrieved = _get_task_result(tool_ctx, predecessor_id, include_authority=True)
    child_contract = _build_child_subagent_contract({
        "tid": "cat-nested", "objective": "Inspect the Cat implementation",
        "expected_output": "Report", "constraints": "Use the inherited authority",
        "parent_contract": task["task_contract"], "child_delegation_budget": {
            **task["task_contract"]["delegation_budget"], "depth_remaining": 1,
        },
    })
    nested_work_order = compile_external_work_order({
        "id": "cat-nested", "objective": "Inspect the Cat implementation",
        "task_contract": child_contract,
    })
    assert len(nested_work_order) < 250_000
    assert child_contract["predecessor_authority"] == task["predecessor_authority"]

    surfaces = (rendered, retrieved, direct_harness, nested_work_order)
    for surface in surfaces:
        for marker in (
            tail, result_tail, artifact_error, artifact_finalized_at, capability_delta,
            delegated_custody, verification_ledger, verification_receipt,
            future_terminal_fact, final_answer, non_final_rows,
            mutation_evidence, plan_review_state,
        ):
            assert marker in surface
        for marker in process_evidence:
            assert marker not in surface
        for field in excluded_process_fields:
            assert field not in surface
    assert "never use native/API fallback" in rendered
    assert "L1 asks L2 to spawn L3" in direct_harness
    assert "never use native/API fallback" in nested_work_order


def test_main_promotion_selects_only_manifested_canonical_predecessor(tmp_path, monkeypatch):
    import json

    import server
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    result_dir = tmp_path / "task_results"
    result_dir.mkdir()
    rows = [
        {"_schema_version": 1, "task_id": "old-a", "status": "completed", "title": "First project"},
        {"_schema_version": 1, "task_id": "old-b", "status": "completed", "title": "Chosen project"},
    ]
    for row in rows:
        (result_dir / f"{row['task_id']}.json").write_text(
            json.dumps(row), encoding="utf-8",
        )
    manifest = {"final_results": [server._task_result_ground_truth(row) for row in rows]}
    selected = _swarm_ctx(tmp_path, task_metadata={
        "force_plan": True, "force_plan_source": "swarm",
        "main_routing_manifest": manifest,
    })

    out = _promote_chat_to_task(
        selected, "Continue the chosen work", predecessor_task_id="old-b",
    )

    assert out.startswith("OK: task")
    assert selected.pending_events[0]["predecessor_authority_source"] == (
        manifest["final_results"][1]["authority_source"]
    )

    fresh = _swarm_ctx(tmp_path, task_metadata={
        "force_plan": True, "force_plan_source": "swarm",
        "main_routing_manifest": manifest,
    })
    assert _promote_chat_to_task(fresh, "Start unrelated work", predecessor_task_id="").startswith("OK: task")
    assert "predecessor_authority_source" not in fresh.pending_events[0]

    selected_source = manifest["final_results"][1]["authority_source"]
    forged_sources = [
        manifest["final_results"][0]["authority_source"],
        {**selected_source, "kind": "other"},
        {**selected_source, "tool": "other"},
        {**selected_source, "arguments": {"task_id": "other", "include_authority": True}},
        {**selected_source, "arguments": {"task_id": "old-b", "include_authority": False}},
    ]
    for forged_source in forged_sources:
        forged_manifest = json.loads(json.dumps(manifest))
        forged_manifest["final_results"][1]["authority_source"] = forged_source
        forged = _swarm_ctx(tmp_path, task_metadata={
            "force_plan": True, "force_plan_source": "swarm",
            "main_routing_manifest": forged_manifest,
        })
        refused_forgery = _promote_chat_to_task(
            forged, "Continue mismatched work", predecessor_task_id="old-b",
        )
        assert refused_forgery.startswith("⚠️ AUTHORITY_SOURCE_UNAVAILABLE")
        assert forged.pending_events == []

    missing = _swarm_ctx(tmp_path, task_metadata={
        "force_plan": True, "force_plan_source": "swarm",
        "main_routing_manifest": manifest,
    })
    (result_dir / "old-b.json").unlink()
    refused = _promote_chat_to_task(
        missing, "Continue missing work", predecessor_task_id="old-b",
    )
    assert refused.startswith("⚠️ AUTHORITY_SOURCE_UNAVAILABLE")
    assert missing.pending_events == []


def test_presence_promotion_preserves_ceiling_and_cannot_choose_new_scope(tmp_path, monkeypatch):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, presence_ceiling_payload
    from ouroboros.tools.control import _build_child_subagent_contract, _promote_chat_to_task

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
        "predecessor_authority": {"result": "preserve completed predecessor"},
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
    out = _promote_chat_to_task(
        ctx,
        "Research the question",
        project_name="Injected",
        workspace_root="/tmp/foreign",
        source="https://example.invalid/repo.git",
        predecessor_task_id="",
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
    # Envelope contract (2026-08-30): a bounded legacy body - no nested
    # recursion carrier, no oversized string - passes through byte-identical
    # (exact strings are authority); only the growth carriers get collapsed.
    assert child["predecessor_authority"] == contract["predecessor_authority"]


def test_real_presence_promotion_rebases_root_and_materializes_all_attachments(
    tmp_path, monkeypatch,
):
    import supervisor.workers as workers
    from ouroboros.presence_runner import _build_task
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools.control import (
        _build_child_subagent_contract,
        _materialize_child_attachment_manifest,
        _promote_chat_to_task,
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
    result = _promote_chat_to_task(
        tool_ctx,
        "Research the new question deeply",
        expected_output="A grounded report",
        project_name="forbidden scope",
        workspace_root="/tmp/forbidden",
        predecessor_task_id="",
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
        "attachment_manifest": child_manifest,
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
    from ouroboros.tools.control import _promote_chat_to_task
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
    assert _promote_chat_to_task(tool_ctx, "Long work", predecessor_task_id="").startswith("OK: task")
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
        predecessor_task_id="",
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
        predecessor_task_id="",
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

    out = _promote_chat_to_task(ctx, "Continue the racer build", project_id="racer", predecessor_task_id="")

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
        out = _promote_chat_to_task(ctx, "Audit the issue", predecessor_task_id="", **kwargs)
        assert out.startswith("OK: task")
        assert ctx.pending_events[0]["project_id"] == "alpha"
        assert f"Explicit project {shown!r} was ignored" in out
        assert "bound to project 'alpha'" in out

    ctx = _swarm_ctx(tmp_path, project_id="alpha")
    out = _promote_chat_to_task(ctx, "Audit the issue", project_id="alpha", predecessor_task_id="")
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


def test_ensure_project_scope_announces_started_only_on_real_create(tmp_path, monkeypatch):
    """B1 seam 2 (owner 2=A): a mid-task ensure_project_scope announces the
    `project_started` row only when it REALLY created the project; scoping a
    later task to the existing project stays silent."""
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    queued = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_k: queued.append(dict(event)) or True,
    )
    ctx = types.SimpleNamespace(RUNNING={})

    workers.ensure_project_scope({
        "type": "ensure_project_scope", "task_id": "midtask01",
        "project_id": "research-hub", "project_name": "Research Hub",
    }, ctx)
    assert [row["system_type"] for row in queued] == ["project_started"]
    assert queued[0]["chat_id"] == 1
    assert queued[0]["delivery_id"] == "project-start:research-hub"
    assert queued[0]["progress_meta"]["project_name"] == "Research Hub"

    queued.clear()
    workers.ensure_project_scope({
        "type": "ensure_project_scope", "task_id": "midtask02",
        "project_id": "research-hub",
    }, ctx)
    assert queued == []  # attach to the existing project announces nothing


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
    from ouroboros.presence_authority import (
        build_presence_capability_ceiling,
        presence_ceiling_payload,
    )
    from ouroboros.presence_capabilities import (
        PresenceProfileResolution,
        PresenceSelection,
        PresenceToolTarget,
    )
    from ouroboros.presence_runtime import ResolvedPresenceRuntime

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


def test_ephemeral_swarm_unconfirmed_promotion_reuses_one_task_id(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    monkeypatch.setattr(
        "ouroboros.tools.control_events._wait_for_promotion_admission",
        lambda *_args, **_kwargs: {"status": "unconfirmed", "reason": "confirmation_timeout"},
    )
    ctx = _swarm_ctx(tmp_path)

    first = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")

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

    first = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")

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

    first = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")
    second = _promote_chat_to_task(ctx, "Audit and fix the issue", predecessor_task_id="")

    assert first == second
    assert first.startswith("PROMOTE_REJECTED")
    assert ctx.pending_events == []
    assert ctx._swarm_handoff_attempt["status"] == "rejected"


def test_managed_swarm_does_not_recursively_propagate_routing_intent(tmp_path, monkeypatch):
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    ctx = _swarm_ctx(tmp_path, is_ephemeral_turn=False)

    _promote_chat_to_task(ctx, "A later task chosen during execution", predecessor_task_id="")

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
    out = _promote_chat_to_task(ctx, "x", project_id="Bad Name!", predecessor_task_id="")
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
        predecessor_task_id="",
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
    out = _promote_chat_to_task(ctx, "исследуй динозавров", project_name="динозавры", title="динозавры", predecessor_task_id="")
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


def test_promote_post_stage_lookup_failure_cleans_attachment(tmp_path, monkeypatch):
    import ouroboros.projects_registry as projects_registry
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    monkeypatch.setattr(
        projects_registry,
        "get_reserved_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("registry unavailable")),
    )
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: task,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    )

    outcome = workers.promote_chat_to_task({
        "task_id": "attach-lookup-fail",
        "objective": "use input",
        "project_id": "lookup-project",
        "attachment_uploads": [{"path": str(source), "label": "input"}],
    }, ctx)

    assert outcome["reason"] == "project_routing_fence_lookup_failed"
    assert not (
        tmp_path / "task_results" / "artifacts" / "attach-lookup-fail"
    ).exists()


@pytest.mark.parametrize("failure", ["enqueue", "snapshot"])
def test_promote_queue_failure_cleans_pre_staged_attachment(
    tmp_path, monkeypatch, failure,
):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    source = tmp_path / f"{failure}.txt"
    source.write_text("input", encoding="utf-8")
    captured = []

    def enqueue(task):
        captured.append(task)
        if failure == "enqueue":
            return {"_admission_blocked": "project_routing_fence"}
        return task

    ctx = types.SimpleNamespace(
        enqueue_task=enqueue,
        persist_queue_snapshot=lambda **_kwargs: failure != "snapshot",
        load_state=lambda: {"owner_chat_id": 1},
    )
    tid = f"attach-{failure}-fail"

    outcome = workers.promote_chat_to_task({
        "task_id": tid,
        "objective": "use input",
        "workspace": "none",
        "attachment_uploads": [{"path": str(source), "label": "input"}],
    }, ctx)

    expected = "project_routing_fence" if failure == "enqueue" else "queue_snapshot_persist_failed"
    assert outcome["reason"] == expected
    assert captured
    staged_path = pathlib.Path(captured[0]["attachments"][0]["abs_path"])
    assert not staged_path.exists()
    assert not (tmp_path / "task_results" / "artifacts" / tid).exists()


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

    assert {key: result[key] for key in ("status", "task_id")} == {
        "status": "scheduled",
        "task_id": "repair01",
    }
    assert result["_admitted_task_contract"] == enqueued[0]["task_contract"]
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


def test_registered_project_chat_ids_recognizes_every_project(tmp_path):
    """The isolation SSOT recognizes EVERY registered project's chat_id (regardless
    of sidebar visibility) so its raw chat never re-leaks into the штаб's main
    context / dialogue consolidation / background consciousness (BIBLE P1). Sidebar
    visibility is a separate presentation concern (no project statuses, v6.33.0)."""
    from ouroboros.projects_registry import (
        create_project,
        registered_project_chat_ids,
        update_project,
    )

    proj = create_project(tmp_path, "old-racer")
    chat_id = int(proj["chat_id"])
    assert chat_id in registered_project_chat_ids(tmp_path)
    # A rename (or any mutable-field update) never drops it from the isolation set.
    update_project(tmp_path, "old-racer", name="Old Racer (renamed)")
    assert chat_id in registered_project_chat_ids(tmp_path)


def test_chat_history_tool_spans_all_threads_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the chat_history TOOL is the one mind's
    DELIBERATE recall — it spans the WHOLE conversation (main + ALL project
    threads), only A2A virtual transport excluded. Project-task FOCUS lives in the
    passive default context (build_recent_sections), NOT in this recall tool, so
    the one identity can recall anything it chooses (BIBLE P1)."""
    import json

    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    a = create_project(tmp_path, "alpha")
    b = create_project(tmp_path, "beta")
    ca, cb = int(a["chat_id"]), int(b["chat_id"])
    rows = [
        {"direction": "in", "text": "main-msg", "chat_id": 1},
        {"direction": "in", "text": "alpha-msg", "chat_id": ca},
        {"direction": "in", "text": "beta-msg", "chat_id": cb},
        {"direction": "in", "text": "a2a-noise", "chat_id": -1001},
    ]
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mem = Memory(drive_root=tmp_path)

    view = mem.chat_history(count=50)
    assert "main-msg" in view and "alpha-msg" in view and "beta-msg" in view  # all threads
    assert "a2a-noise" not in view  # only A2A virtual transport excluded


def test_chat_history_tool_uses_canonical_budget_root_and_archive_pagination(tmp_path):
    import json
    from types import SimpleNamespace

    from ouroboros.tools.control import _chat_history

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    (canonical / "logs").mkdir(parents=True)
    (canonical / "archive").mkdir()
    child.mkdir()
    (canonical / "archive" / "chat_20260820T010000.jsonl").write_text(
        "\n".join(json.dumps({"direction": "in", "text": f"old-{i}"}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    (canonical / "logs" / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "text": "new-live"}) + "\n",
        encoding="utf-8",
    )
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


def test_recent_context_full_awareness_and_project_focus_with_bindings(tmp_path):
    """Passive context (v6.32.0): the one identity's MAIN recent context sees
    EVERYTHING, including a post-hoc bound task's rows (one mind, BIBLE P1). A
    PROJECT task's recent context is FOCUSED on its own thread + rows of tasks
    bound to it; unrelated main chat is left out of the focused working view
    (focus in the passive default, not isolation)."""
    import json

    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import bind_task_to_project, create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    proj = create_project(tmp_path, "promoted")
    pchat = int(proj["chat_id"])
    bind_task_to_project(tmp_path, "task-7", "promoted", pchat, origin={"absent": "system"})
    rows = [
        {"direction": "in", "text": "plain-main", "chat_id": 1},
        {"direction": "out", "text": "bound-task-row", "chat_id": 1, "task_id": "task-7"},
    ]
    (logs / "chat.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    mem = Memory(drive_root=tmp_path)

    # Main passive context: full awareness sees everything.
    main_ctx = "\n".join(build_recent_sections(mem, env=None))
    assert "plain-main" in main_ctx and "bound-task-row" in main_ctx

    # Project task passive context: focused on its own thread + bound-task rows.
    proj_ctx = "\n".join(build_recent_sections(mem, env=None, thread_chat_id=pchat))
    assert "bound-task-row" in proj_ctx
    assert "plain-main" not in proj_ctx


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


def test_busy_project_chat_routes_to_ephemeral_decision_turn(tmp_path, monkeypatch):
    """WS1/P5 (v6.34.0): a busy PROJECT chat is NOT mechanically auto-enqueued into a
    duplicate pooled task. It runs the ephemeral decision turn (project-scoped, seeing
    current_chat.running_tasks) so the one mind decides steer_task / answer / promote by
    judgment — replacing the old 'Hybrid B+' auto-enqueue fallback."""
    import threading as _threading

    import server
    from ouroboros.projects_registry import create_project

    proj = create_project(tmp_path, "market-research")
    project_chat = int(proj["chat_id"])
    enqueued = []
    ephemeral_calls = []
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

    def _ephemeral(cid, text, image_data, *, task_constraint=None, task_metadata=None):
        ephemeral_calls.append({"chat_id": cid, "text": text, "metadata": task_metadata})
        called.set()

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=_Consciousness(),
        get_chat_agent=lambda: types.SimpleNamespace(_busy=True),
        handle_chat_direct=lambda *a, **k: (_ for _ in ()).throw(AssertionError("direct lane must not run when busy")),
        handle_chat_ephemeral=_ephemeral,
        enqueue_task=lambda task: enqueued.append(task),
        send_with_budget=lambda *a, **k: None,
    )

    assert server._process_bridge_updates(_Bridge(), 0, ctx) == 1
    assert called.wait(timeout=3)  # the ephemeral decision turn ran on its own thread
    assert enqueued == []  # NOT auto-enqueued into a duplicate pooled task
    assert len(ephemeral_calls) == 1
    md = ephemeral_calls[0]["metadata"] or {}
    assert str(md.get("project_id") or "")  # project-scoped decision turn
    assert "сколько будет 2+2?" in (ephemeral_calls[0]["text"] or "")


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
    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    with open(logs / "chat.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-01-01T00:00:00Z", "direction": "out", "text": "final answer", "chat_id": 1, "task_id": "task-1"}) + "\n")
        fh.write(json.dumps({"ts": "2026-01-01T00:00:01Z", "direction": "in", "text": "raw project chat", "chat_id": project_chat}) + "\n")
    with open(logs / "progress.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-01-01T00:00:02Z", "type": "send_message", "content": "working", "text": "working", "is_progress": True, "chat_id": 1, "task_id": "task-1", "format": "markdown", "cancelable": True}) + "\n")

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

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    rows = [
        {"ts": "2026-06-13T00:00:01Z", "direction": "in", "text": "main hello", "chat_id": 1},
        {"ts": "2026-06-13T00:00:02Z", "direction": "out", "text": "main reply", "chat_id": 1},
        {"ts": "2026-06-13T00:00:03Z", "direction": "in", "text": "project hello", "chat_id": project_chat},
        {"ts": "2026-06-13T00:00:033Z", "direction": "system", "type": "task_summary", "text": "project summary", "chat_id": project_chat, "task_id": "pt"},
        {"ts": "2026-06-13T00:00:035Z", "direction": "in", "text": "transport mirror", "chat_id": transport_chat},
        {"ts": "2026-06-13T00:00:04Z", "direction": "out", "text": "a2a noise", "chat_id": -1001},
        {"ts": "2026-06-13T00:00:05Z", "direction": "out", "text": "legacy row (no chat_id)"},
    ]
    with open(logs / "chat.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

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
    from ouroboros.tools.control import _steer_task

    events = []
    ctx = types.SimpleNamespace(
        pending_events=events, event_queue=None, current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={"client_message_id": "cm-42"},
    )
    out = _steer_task(ctx, "abc12345", "also add the benchmarks slide")
    assert out.startswith("⚠️ STEER_UNCONFIRMED")
    assert len(events) == 1
    evt = events[0]
    assert evt["type"] == "steer_task"
    assert evt["target_task_id"] == "abc12345"
    assert evt["message"] == "also add the benchmarks slide"
    assert evt["chat_id"] == 1
    assert evt["client_message_id"] == "cm-42"
    assert evt["allow_global_root"] is False
    assert ctx._typed_routing_action_emitted == "steer_task"


def test_steer_task_uses_exact_ingress_owner_text(tmp_path):
    from ouroboros.tools.control import _steer_task

    events = []
    exact = "  exact owner text\nwith trailing space  "
    ctx = types.SimpleNamespace(
        pending_events=events,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={
            "client_message_id": "cm-exact",
            "origin_message_text": exact,
        },
    )

    _steer_task(ctx, "target-task", "model paraphrase")

    assert events[0]["message"] == exact


def test_main_steer_can_address_project_bound_root_from_host_manifest(tmp_path, monkeypatch):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    emitted = []
    tool_ctx = types.SimpleNamespace(
        pending_events=emitted,
        event_queue=None,
        current_chat_id=1,
        drive_root=tmp_path,
        task_metadata={
            "client_message_id": "main-42",
            "routing_contract": {"source_lane": "main"},
        },
    )
    _steer_task(tool_ctx, "project-root", "continue from Main")
    assert emitted[0]["allow_global_root"] is True

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
    assert drain_owner_messages(tmp_path, "project-root") == ["continue from Main"]


def test_busy_direct_main_root_is_manifested_and_steerable_without_promotion(tmp_path):
    import threading

    import server
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
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
        task_metadata={
            "client_message_id": "followup-1",
            "routing_contract": metadata["routing_contract"],
        },
    )
    _steer_task(tool_ctx, "direct-root", "Use FusionBrain images too")
    assert [event["type"] for event in emitted] == ["steer_task"]

    event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
    )
    _handle_steer_task(emitted[0], event_ctx)
    assert drain_owner_messages(tmp_path, "direct-root") == ["Use FusionBrain images too"]


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
    from ouroboros.tools.control import _steer_task

    ctx = types.SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, task_metadata={})
    assert "TOOL_ARG_ERROR" in _steer_task(ctx, "", "msg")
    assert "TOOL_ARG_ERROR" in _steer_task(ctx, "t1", "")
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


@pytest.mark.parametrize("chat_id", [1, 0, None])
def test_steering_delivers_text_and_identical_typed_attachment_report(
    tmp_path, monkeypatch, chat_id,
):
    import supervisor.queue as queue
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.project_dialogue import latest_chat_annotations
    from supervisor.events import _handle_steer_task

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    good = tmp_path / "good.txt"
    good.write_text("ok", encoding="utf-8")
    notices = []
    acks = []
    bridge = types.SimpleNamespace(
        send_routing_ack=lambda _chat_id, **payload: acks.append(payload),
    )
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={"t-report": {"task": {"id": "t-report", "chat_id": chat_id or 0}}},
        send_with_budget=lambda _cid, text, *a, **k: notices.append(text),
        bridge=bridge,
    )
    exact = "urgent owner text  \n"
    evt = {
        "type": "steer_task",
        "target_task_id": "t-report",
        "message": exact,
        "chat_id": chat_id,
        "client_message_id": "cm-report",
        "routing_token": "route-report",
        "attachment_uploads": [
            {"path": str(good), "label": "good"},
            {"path": str(tmp_path / "missing.txt"), "label": "missing"},
        ],
    }

    _handle_steer_task(evt, ctx)

    delivered = drain_owner_entries(tmp_path, "t-report")[0]["text"]
    assert delivered.startswith(exact)
    report = delivered.split("[ATTACHMENTS]\n", 1)[1].split("\n[END_ATTACHMENTS]", 1)[0]
    if chat_id is None:
        assert notices == []
    else:
        assert report in notices[-1]
    annotation = latest_chat_annotations(tmp_path)["cm-report"]
    assert annotation["attachment_manifest"] == acks[-1]["attachment_manifest"]
    assert [row["status"] for row in annotation["attachment_manifest"]] == [
        "staged", "rejected",
    ]
    assert annotation["detail"] == report


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
    acks = []
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={},
        PENDING=[],
        get_chat_agent=lambda: direct_agent,
        bridge=types.SimpleNamespace(
            send_routing_ack=lambda _chat_id, **payload: acks.append(payload),
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
    assert notices[-1].splitlines()[0] == (
        "📎 Attachment staging report for Tower Defence › Fix nested delegation:"
    )


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
    assert notices[-1].splitlines()[0] == (
        "📎 Attachment staging report for Tower Defence › Fix nested delegation:"
    )


def test_handle_steer_task_stale_target_notifies_visibly(tmp_path, monkeypatch):
    """A target no longer RUNNING (or in another chat / a subagent) fails VISIBLY
    with a chat notice and writes NO mailbox — never silently dropped or respawned."""
    import supervisor.queue as queue
    from supervisor.events import _handle_steer_task
    from ouroboros.owner_mailbox import drain_owner_entries

    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    notices = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "other": {"task": {"id": "other", "chat_id": 999}},  # different chat
            "sub": {"task": {"id": "sub", "chat_id": 1, "delegation_role": "subagent"}},
        },
        send_with_budget=lambda cid, text, *a, **k: notices.append(text),
    )
    _handle_steer_task({"target_task_id": "gone", "message": "a", "chat_id": 1}, ctx)   # not running
    _handle_steer_task({"target_task_id": "other", "message": "b", "chat_id": 1}, ctx)  # wrong chat
    _handle_steer_task({"target_task_id": "sub", "message": "c", "chat_id": 1}, ctx)    # subagent
    assert len(notices) == 3 and all("Couldn't steer task" in n for n in notices)
    assert drain_owner_entries(tmp_path, "gone") == []
    assert drain_owner_entries(tmp_path, "other") == []
    assert drain_owner_entries(tmp_path, "sub") == []


def test_chat_running_tasks_lists_same_chat_pooled_only(tmp_path):
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
    assert {r["task_id"] for r in rows} == {"a", "b"}
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
    # No running tasks + no client id -> metadata returned unchanged.
    empty_ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={})
    assert server._decision_turn_metadata(empty_ctx, 1, "", {"k": "v"}) == {"k": "v"}


# --- Q10=A (owner, 2026-08-08): file-less project promotes auto-provision -----

def _promote_ctx(enqueued):
    return types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
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

    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "broken1",
        "objective": "Continue",
        "project_id": "brokenp",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_unusable"
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
    enqueued = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "provfail1",
        "objective": "Build",
        "project_id": "provfail-proj",
        "chat_id": 1,
    }, _promote_ctx(enqueued))

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "workspace_provisioning_failed"
    assert enqueued == []


def test_swarm_intent_survives_admission_to_the_finalization_read(tmp_path, monkeypatch):
    """rc-phaseC propagation pin: `force_plan_source="swarm"` attached at chat
    admission rides the promote event into the admitted root's task["metadata"]
    and is the exact fact build_swarm_efficiency reads at finalization — so a
    Swarm-button root that fanned out nothing finalizes with the
    no_fanout_observed block instead of a silent None."""
    import supervisor.workers as workers
    from ouroboros.agent_task_pipeline import _build_swarm_efficiency
    from ouroboros.tools.control import _promote_chat_to_task

    _confirm_promote(monkeypatch)
    router_ctx = _swarm_ctx(tmp_path)
    assert _promote_chat_to_task(router_ctx, "Build it with a swarm", predecessor_task_id="").startswith("OK: task")
    evt = router_ctx.pending_events[0]
    assert evt["force_plan_source"] == "swarm"

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    admitted = []
    outcome = workers.promote_chat_to_task(evt, _promote_ctx(admitted))
    assert outcome["status"] == "scheduled"
    task = admitted[0]
    assert task["metadata"]["force_plan_source"] == "swarm"

    # Deliberately NO logs/events.jsonl: a fresh drive root has none, and the
    # zero-fanout block must still be returned (the reader is fail-soft).
    block = _build_swarm_efficiency(types.SimpleNamespace(drive_root=str(tmp_path)), task)
    assert block == {
        "intent_source": "swarm",
        "planned": None,
        "observed_started": 0,
        "status": "no_fanout_observed",
    }
