"""Focused Phase-1B runtime tests for configured recursive actors."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest


def _settings(*rows):
    return {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)}),
    }


def _api_row(row_id="api-builder", model="openai/gpt-5.6-sol", effort="high"):
    return {
        "subagent_id": row_id,
        "name": "API builder",
        "recommended_use": "Exact recursive API actor.",
        "route": {"kind": "api_model", "target_id": model},
        "effort": effort,
    }


def _session_row(row_id="session-builder", target="codex=gpt-5.6-sol", effort="high"):
    return {
        "subagent_id": row_id,
        "name": "Session builder",
        "recommended_use": "Subscription-backed implementation.",
        "route": {
            "kind": "agent_session",
            "target_id": target,
            "credential_profile_id": "profile-1",
        },
        "effort": effort,
    }


def _snapshot(settings, row_id):
    from ouroboros.subagent_runtime import select_subagent_snapshot

    return select_subagent_snapshot(settings, subagent_id=row_id)[0]


def test_public_schema_is_id_only_and_real_registry_allows_hidden_d23_conflict(
    monkeypatch, tmp_path,
):
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    settings = _settings(_api_row())
    monkeypatch.setattr(control, "load_settings", lambda: settings)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path))
    schema = registry.get_schema_by_name("schedule_subagent")["function"]["parameters"]
    assert "subagent_id" in schema["properties"]
    assert "model_lane" not in schema["properties"]
    assert "executor" not in schema["properties"]
    assert {"subagent_id", "objective", "expected_output"} <= set(schema["required"])

    # Reaches the handler through ToolRegistry.execute: public validation did not
    # advertise the legacy arg, but the bounded hidden seam carried it for D23.
    result = registry.execute("schedule_subagent", {
        "subagent_id": "api-builder",
        "objective": "Implement it",
        "expected_output": "Patch",
        "executor": "native",
    })
    assert "subagent_selector_conflict" in result
    assert "unsupported" not in result


def test_d23_maps_one_migrated_row_and_refuses_ambiguous_auto(monkeypatch):
    from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot

    settings = {
        "OUROBOROS_SUBAGENT_HARNESS": "codex=gpt-5.6-sol:high",
        "OUROBOROS_SUBAGENT_PROFILE": "profile-1",
    }
    snapshot, legacy = select_subagent_snapshot(
        settings,
        legacy_executor="harness",
        legacy_executor_supplied=True,
    )
    assert legacy is True
    assert snapshot["selected_subagent_id"] == "legacy-session"
    assert snapshot["route"] == {
        "kind": "agent_session",
        "target_id": "codex=gpt-5.6-sol",
        "credential_profile_id": "profile-1",
    }
    try:
        select_subagent_snapshot(
            settings,
            legacy_executor="auto",
            legacy_executor_supplied=True,
        )
    except SubagentSelectionError as exc:
        assert exc.code == "subagent_selection_required"
    else:  # pragma: no cover - the assertion explains the invariant
        raise AssertionError("legacy auto must never select by order")


def test_unsaved_roster_is_off_but_d23_can_map_exact_owner_legacy(monkeypatch):
    from ouroboros.subagent_runtime import (
        SubagentSelectionError,
        effective_runtime_subagent_settings,
        select_subagent_snapshot,
    )

    runtime_keys = (
        "OUROBOROS_SUBAGENT_HARNESS", "OUROBOROS_SUBAGENT_PROFILE",
        "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL",
        "USE_LOCAL_HEAVY", "USE_LOCAL_LIGHT", "USE_LOCAL_MAIN", "LOCAL_MODEL_SOURCE",
    )
    for key in runtime_keys:
        monkeypatch.delenv(key, raising=False)
    effective = effective_runtime_subagent_settings({
        # A shipped value still present on disk was cleared by provider
        # normalization at task start, represented by its absent env projection.
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.5-flash",
    })
    assert effective["OUROBOROS_MODEL_HEAVY"] == ""
    with pytest.raises(SubagentSelectionError) as refused:
        select_subagent_snapshot(effective, subagent_id="legacy-heavy")
    assert refused.value.code == "subagent_configuration_unsaved"

    legacy = {"OUROBOROS_MODEL_HEAVY": "owner/custom-heavy"}
    with pytest.raises(SubagentSelectionError) as hidden_from_new_surface:
        select_subagent_snapshot(legacy, subagent_id="legacy-heavy")
    assert hidden_from_new_surface.value.code == "subagent_configuration_unsaved"
    snapshot, used_d23 = select_subagent_snapshot(
        legacy, legacy_model_lane="heavy", legacy_model_lane_supplied=True,
    )
    assert used_d23 is True
    assert snapshot["selected_subagent_id"] == "legacy-heavy"
    assert snapshot["route"]["target_id"] == "owner/custom-heavy"


def test_snapshot_is_immutable_across_queue_persistence(monkeypatch, tmp_path):
    from supervisor import queue as task_queue

    original = _settings(_api_row(model="openai/model-a"))
    snapshot = _snapshot(original, "api-builder")
    task = {
        "id": "child1",
        "type": "task",
        "chat_id": 1,
        "text": "work",
        "delegation_role": "subagent",
        "configured_subagent": snapshot,
        "parent_cognitive_route": {
            "model": "openai/parent",
            "effort": "high",
            "use_local_model": False,
        },
        "_attempt": 1,
    }
    pending, running = [task], {}
    task_queue.init(tmp_path)
    task_queue.init_queue_refs(pending, running, {"value": 0})
    assert task_queue.persist_queue_snapshot(reason="test") is True
    pending.clear()
    assert task_queue.restore_pending_from_snapshot() == 1
    restored = pending[0]
    assert restored["configured_subagent"] == snapshot
    assert restored["configured_subagent"]["route"]["target_id"] == "openai/model-a"
    assert restored["parent_cognitive_route"]["model"] == "openai/parent"


def test_api_actor_dispatch_is_exact_recursive_route_without_slot_substitution(monkeypatch):
    from ouroboros.subagents import resolve_subagent_dispatch
    import ouroboros.provider_models as provider_models

    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    snapshot = _snapshot(_settings(_api_row(model="openai/exact-model", effort="xhigh")), "api-builder")
    dispatch = resolve_subagent_dispatch({
        "id": "child1",
        "type": "task",
        "delegation_role": "subagent",
        "configured_subagent": snapshot,
        "task_constraint": {},
    }, task_type="task")
    assert dispatch.executor == "native"
    assert dispatch.route == "openai/exact-model"
    assert dispatch.lane.model == "openai/exact-model"
    assert dispatch.effort == "xhigh"
    assert dispatch.availability["route_kind"] == "api_model"


def test_session_dispatch_pins_exact_route_and_inherits_parent_cognition(monkeypatch):
    import ouroboros.subagents as subagents
    import ouroboros.claudexor_daemon as daemon

    class Gateway:
        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    seen = {}

    def health(_gateway, route_id, _shape, *, route_model="", pinned_profile=""):
        seen.update(route_id=route_id, model=route_model, profile=pinned_profile)
        return "", ""

    monkeypatch.setattr(subagents, "route_health", health)
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    dispatch = subagents.resolve_subagent_dispatch({
        "id": "child1",
        "type": "task",
        "delegation_role": "subagent",
        "configured_subagent": snapshot,
        "parent_cognitive_route": {
            "model": "anthropic/parent-exact",
            "effort": "max",
            "use_local_model": False,
        },
        "task_constraint": {},
    }, task_type="task")
    assert dispatch.executor == "harness"
    assert dispatch.lane.model == "anthropic/parent-exact"
    assert dispatch.effort == "max"
    assert seen == {"route_id": "codex", "model": "gpt-5.6-sol", "profile": "profile-1"}


def test_selected_session_visibility_failure_blocks_without_native_substitution(monkeypatch):
    import ouroboros.agent as agent_module
    import ouroboros.subagents as subagents

    route = subagents.DelegationRoute(route_id="codex", model="gpt-5.6-sol")
    dispatch = subagents.SubagentDispatch(
        lane=subagents.SubagentLaneResolution(
            requested_lane="auto", effective_lane="main", model="openai/parent",
            resolved_from="main", provenance="configured_subagent",
        ),
        effort="high", executor="harness", route="codex", profile="local_readonly_subagent",
        delta=subagents.CapabilityDelta(
            derived_effort="high", effective_effort="high",
            requested_executor="harness", effective_executor="harness",
        ),
        executor_resolution=subagents.SubagentExecutorResolution(
            "harness", "harness", route, "harness_ready",
        ),
    )
    monkeypatch.setattr(agent_module, "envelope_from_task", lambda *_a, **_k: {})
    monkeypatch.setattr(
        subagents, "preflight_native_fallback_dispatch",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("a selected row must never host-fallback to native")
        ),
    )
    task = {"id": "child1", "configured_subagent": {"schema": 1}}
    amended, changed = agent_module.preflight_delegate_visibility(
        SimpleNamespace(available_tools=lambda: []), task, dispatch,
    )
    assert changed is True
    assert amended.executor == "blocked"
    assert amended.executor_resolution.reason == "delegate_tools_invisible"


@pytest.mark.parametrize(
    ("interactive", "expected_reason"),
    [
        (True, ""),
        (False, "work_order_source_channel_unavailable"),
    ],
)
def test_over_budget_bootstrap_uses_only_a_live_interaction_channel(
    monkeypatch, tmp_path, interactive, expected_reason,
):
    from ouroboros import delegate_custody as custody
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagent_runtime as runtime
    from ouroboros.subagent_work_order import work_order_fingerprint

    class Gateway:
        def harnesses(self):
            return [{
                "id": "codex",
                "manifest": {"capabilities": {"interactive": interactive}},
            }]

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    calls = []
    monkeypatch.setattr(runtime, "exact_start", lambda ctx, prompt, spec: (
        calls.append((prompt, spec))
        or json.dumps({"status": "started", "run_id": "run-source", "custody_durable": True})
    ))
    import ouroboros.delegate_supervision as supervision

    monkeypatch.setattr(
        supervision, "supervised_wait",
        lambda *_a, **_kw: pytest.fail("the host must not wait inside bootstrap (owner 1=A)"),
    )
    snapshot = _snapshot(_settings(_session_row(target="codex=gpt-5.6-sol")), "session-builder")
    dispatch = SimpleNamespace(
        executor="harness", blocked=False,
        executor_resolution=SimpleNamespace(route=SimpleNamespace(route_id="codex")),
    )
    ctx = SimpleNamespace(
        task_id="child-source", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_metadata={},
    )
    task = {
        "id": "child-source",
        "objective": ("THIS MUST NOT BE SENT AS A PREFIX " + ("x" * 250_100)),
        "configured_subagent": snapshot,
        "task_contract": {"objective": "THIS MUST NOT BE SENT AS A PREFIX " + ("x" * 250_100)},
    }
    full_sha = work_order_fingerprint(task)
    # Charter D1: the host pre-starts the leaf during bootstrap, through the
    # same wrapper the model's delegate_start(prompt="") uses — and does NOT
    # wait on it (owner 1=A). With a live interactive channel the oversized
    # order rides the source-request lens; without one, the definite refusal
    # ends the child unrun and typed at $0.
    raw = bootstrap.bootstrap_before_context(ctx, task, dispatch)
    custody_rows = [
        json.loads(line)
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    ]

    if interactive:
        out = json.loads(raw)
        assert out["status"] == "configured_session_started"
        assert out["startup"]["status"] == "started"
        assert out["startup"]["run_id"] == "run-source"
        assert len(calls) == 1
        prompt, spec = calls[0]
        assert "WORK ORDER SOURCE REQUEST" in prompt
        assert "THIS MUST NOT BE SENT AS A PREFIX" not in prompt
        assert spec["compiled_work_order"] is True
        assert spec["work_order_fingerprint"] == full_sha
        assert spec["work_order_source_request"]["complete_sha256"] == full_sha
        assert custody_rows[-1]["type"] == "configured_subagent_work_order_source_request"
        assert custody_rows[-1]["status"] == "attempted"
        assert custody_rows[-1]["source_channel"] == {
            "status": "available",
            "reason": "interactive",
            "route": "codex",
        }
    else:
        assert raw == ""
        assert ctx._configured_startup_refusal["reason"] == expected_reason
        assert calls == []
        assert custody_rows[-1]["type"] == "delegate_run_start_blocked"
        assert custody_rows[-2]["type"] == "configured_subagent_work_order_refused"
        assert custody_rows[-2]["reason"] == expected_reason


@pytest.mark.parametrize(
    ("bootstrap_interactive", "start_interactive", "expected_status", "expected_reason"),
    [
        (True, False, "refused", "work_order_source_channel_unavailable"),
        (None, True, "started", ""),
        (True, None, "refused", "work_order_source_channel_unverified"),
    ],
)
def test_over_budget_start_reprobes_live_interaction_capability(
    monkeypatch, tmp_path, bootstrap_interactive, start_interactive,
    expected_status, expected_reason,
):
    from ouroboros import delegate_custody as custody
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagent_runtime as runtime

    observations = iter([bootstrap_interactive, start_interactive])
    closed = []

    class Gateway:
        def harnesses(self):
            interactive = next(observations)
            capabilities = (
                {} if interactive is None else {"interactive": interactive}
            )
            return [{
                "id": "codex",
                "manifest": {"capabilities": capabilities},
            }]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(daemon, "ensure_owned_gateway", Gateway)
    starts = []
    monkeypatch.setattr(runtime, "exact_start", lambda _ctx, prompt, spec: (
        starts.append((prompt, spec))
        or json.dumps({"status": "started", "run_id": "run-live-probe"})
    ))
    import ouroboros.delegate_supervision as supervision

    monkeypatch.setattr(
        supervision, "supervised_wait",
        lambda *_a, **_kw: pytest.fail("the host must not wait inside bootstrap (owner 1=A)"),
    )
    snapshot = _snapshot(
        _settings(_session_row(target="codex=gpt-5.6-sol")),
        "session-builder",
    )
    dispatch = SimpleNamespace(
        executor="harness", blocked=False,
        executor_resolution=SimpleNamespace(route=SimpleNamespace(route_id="codex")),
    )
    ctx = SimpleNamespace(
        task_id="child-live-probe",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={},
    )
    task = {
        "id": "child-live-probe",
        "objective": "x" * 250_100,
        "configured_subagent": snapshot,
        "task_contract": {"objective": "x" * 250_100},
    }

    # Charter D1: both observations happen inside the bootstrap now — the
    # cached channel probe at authority-freeze time, then the LIVE re-probe
    # inside the pre-start's delegate_start_entry. The (True→False) row proves
    # the cached "available" observation is context only, never start
    # authority: the live probe overrides it into a typed $0 refusal.
    raw = bootstrap.bootstrap_before_context(ctx, task, dispatch)
    custody_rows = [
        json.loads(line)
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    ]

    assert len(closed) == 2
    assert ctx._configured_actor_bootstrap["source_channel"]["route"] == "codex"
    if expected_reason:
        assert raw == ""
        assert ctx._configured_startup_refusal["reason"] == expected_reason
        assert starts == []
        # The refusal row plus the D5 attempt fact: a pre-custody refusal is
        # still a durable delegate_start ATTEMPT (triad 2026-08-30).
        assert custody_rows[-1]["type"] == "delegate_run_start_blocked"
        assert custody_rows[-1]["reason"] == "configured_work_order_source_refused"
        refused = custody_rows[-2]
        assert refused["route"] == "codex"
        assert refused["type"] == "configured_subagent_work_order_refused"
        assert refused["reason"] == expected_reason
        assert refused["source_channel"]["reason"] == (
            "interactive_unsupported"
            if start_interactive is False
            else "interactive_capability_missing"
        )
    else:
        out = json.loads(raw)
        assert out["status"] == "configured_session_started"
        assert out["startup"]["status"] == expected_status
        assert len(starts) == 1
        assert "WORK ORDER SOURCE REQUEST" in starts[0][0]
        assert custody_rows[-1]["route"] == "codex"
        assert custody_rows[-1]["type"] == "configured_subagent_work_order_source_request"
        assert custody_rows[-1]["status"] == "attempted"
        assert custody_rows[-1]["source_channel"] == {
            "status": "available",
            "reason": "interactive",
            "route": "codex",
        }


def test_pending_over_budget_recovery_replays_compact_body_and_full_fingerprint(
    monkeypatch, tmp_path,
):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery
    import ouroboros.subagent_work_order as work_order
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child-source-recovery",
        "_attempt": 2,
        "configured_subagent": snapshot,
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {"objective": "x" * 250_100},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    full_fingerprint = work_order.work_order_fingerprint(task)
    source_request = {
        "schema": 1,
        "kind": "complete_work_order",
        "coverage": "partial",
        "complete_sha256": full_fingerprint,
    }
    compact_prompt = "WORK ORDER SOURCE REQUEST\ncoverage=partial"
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id=task["id"],
        invocation_id="inv-source-recovery",
        idempotency_key="inv-source-recovery",
        max_seconds=60,
        request={"prompt": compact_prompt},
        project_id="",
        project_owned=False,
        route="codex",
        selected_subagent_id=snapshot["selected_subagent_id"],
        config_fingerprint=snapshot["config_fingerprint"],
        work_order_fingerprint=full_fingerprint,
        authority_fingerprint=authority,
        work_order_source_request=source_request,
    )
    handoff = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1, new_attempt=2, worker_id=1, exitcode=1,
    )
    assert handoff["pending_invocation_id"] == "inv-source-recovery"

    monkeypatch.setattr(
        work_order, "compile_external_work_order",
        lambda _task: (_ for _ in ()).throw(
            AssertionError("pending recovery must replay the stored prompt")
        ),
    )
    calls = []
    monkeypatch.setattr(
        delegate, "exact_start",
        lambda _ctx, prompt, spec: (
            calls.append((prompt, spec))
            or json.dumps({"status": "started", "run_id": "run-source-recovered"})
        ),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)
    out = recovery.adopt_handoff(ctx, task)

    assert out["status"] == "adopted"
    assert calls == [(
        compact_prompt,
        {
            "retry_of": "inv-source-recovery",
            "work_order_source_request": source_request,
        },
    )]
    from ouroboros.delegate_source_coverage import prepare_work_order_start_binding

    rebound = prepare_work_order_start_binding(
        ctx, tmp_path, "inv-source-recovery", "", compact_prompt, None,
    )
    assert rebound["fingerprint"] == full_fingerprint
    custody._CUSTODY.clear()


def test_real_task_context_bootstraps_before_context_and_any_llm(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.delegate_supervision as supervision
    import ouroboros.subagent_runtime as runtime
    import ouroboros.subagents as subagents
    from ouroboros.agent import Env, OuroborosAgent

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    order = []

    class Gateway:
        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setattr(
        subagents, "route_health",
        lambda *_a, **_k: ("", ""),
    )
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    # Charter D1 + owner 1=A: the host pre-starts the exact leaf during
    # bootstrap, before the context build and any model call — and does NOT
    # wait on it: the first round arrives immediately with the live receipt.
    monkeypatch.setattr(runtime, "exact_start", lambda _ctx, _prompt, _spec: (
        order.append("physical_start")
        or json.dumps({"status": "started", "run_id": "run-pre-start"})
    ))
    monkeypatch.setattr(
        supervision, "supervised_wait",
        lambda *_a, **_kw: pytest.fail("the host must not wait inside bootstrap (owner 1=A)"),
    )
    def build_context(**_kwargs):
        order.append("context_build")
        return [], {}

    monkeypatch.setattr(agent_module, "build_llm_messages", build_context)
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    agent.tools.available_tools = lambda: ["delegate_start", "delegate_wait", "delegate_cancel"]
    agent.llm.chat = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("_prepare_task_context must not call the model")
    )
    _ctx, messages, _caps = agent._prepare_task_context({
        "id": "child1", "type": "task", "chat_id": 1, "text": "Build",
        "delegation_role": "subagent", "configured_subagent": snapshot,
        "parent_cognitive_route": {
            "model": "openai/parent", "effort": "high", "use_local_model": False,
        },
        "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
        "drive_root": str(drive), "budget_drive_root": str(drive),
    })
    assert order == ["physical_start", "context_build"]
    assert any("CONFIGURED SESSION STARTUP / WAKE RECEIPT" in item["content"] for item in messages)
    receipt = next(item["content"] for item in messages if "CONFIGURED SESSION STARTUP / WAKE RECEIPT" in item["content"])
    assert "configured_session_started" in receipt
    assert "run-pre-start" in receipt
    assert "Waiting is your decision" in receipt
    assert _ctx._configured_actor_bootstrap["selected_subagent_id"] == "session-builder"
    assert _ctx._configured_actor_bootstrap["canonical_work_order"]
    assert _ctx._configured_actor_bootstrap["physical_started"] is True
    # The physical run exists before the first round, so the pacing baseline
    # starts seeded — burn is measured from the live delegated activity.
    assert _ctx._nanny_delegate_baseline == {"round": 0, "cost": 0.0}


def test_actor_first_delegate_start_binds_snapshot_and_canonical_work_order(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    calls = []

    monkeypatch.setattr(
        runtime, "exact_start",
        lambda ctx, prompt, spec: calls.append((prompt, spec)) or json.dumps({
            "status": "started", "run_id": "run-actor-first",
        }),
    )
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "snapshot": snapshot,
            "selected_subagent_id": "session-builder",
            "canonical_work_order": "OBJECTIVE\nBuild the patch",
            "work_order_fingerprint": "full-work-order-sha",
            "work_order_chars": 23,
        },
    )

    out = json.loads(runtime.delegate_start_entry(ctx, ""))
    assert out["status"] == "started"
    assert calls == [(
        "OBJECTIVE\nBuild the patch",
        {
            "snapshot": snapshot,
            "compiled_work_order": True,
            "work_order_fingerprint": "full-work-order-sha",
            "_coordination_context": "",
        },
    )]


def test_selectorless_fresh_start_outside_actor_first_is_refused(tmp_path):
    import ouroboros.subagent_runtime as runtime
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "ordinary-root"

    out = json.loads(runtime.delegate_start_entry(ctx, ""))

    assert out["status"] == "refused"
    assert out["reason"] == "subagent_selection_required"


def test_actor_first_start_marks_physical_activity_and_closes_zero_run(monkeypatch, tmp_path):
    """A started (including uncustodied) leaf cannot later be reported as zero-run."""
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    ctx = SimpleNamespace(
        task_id="child-start-marker", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "snapshot": snapshot,
            "selected_subagent_id": "session-builder",
            "canonical_work_order": "OBJECTIVE\nBuild the patch",
            "work_order_fingerprint": "full-work-order-sha",
            "physical_started": False,
            "exact_start_pending": True,
        },
    )
    monkeypatch.setattr(delegate, "_delegate_start", lambda *_a, **_k: json.dumps({
        "status": "started_uncustodied", "run_id": "run-live",
    }))

    started = json.loads(runtime.exact_start(
        ctx,
        "OBJECTIVE\nBuild the patch",
        {"snapshot": snapshot, "compiled_work_order": True},
    ))
    assert started["status"] == "started_uncustodied"
    assert ctx._configured_actor_bootstrap["physical_started"] is True
    assert ctx._configured_actor_bootstrap["exact_start_pending"] is False
    assert ctx._nanny_physical_activity_seed is True

    # The opposite contradiction is guarded by exact_start as well: once a
    # durable zero-run decision exists, a later physical invocation is refused.
    ctx._configured_actor_bootstrap["physical_started"] = False
    ctx._configured_actor_bootstrap["exact_start_pending"] = True
    ctx._configured_actor_bootstrap["zero_run_receipt_recorded"] = True
    refused = json.loads(runtime.exact_start(
        ctx,
        "OBJECTIVE\nBuild the patch",
        {"snapshot": snapshot, "compiled_work_order": True},
    ))
    assert refused["status"] == "refused"
    assert refused["reason"] == "zero_run_already_recorded"


def test_actor_first_exact_start_hydrates_terminal_zero_run_receipt(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.outcomes import append_verification_receipt

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task_id = "hydrated-zero-run"
    assert append_verification_receipt(tmp_path, task_id, {
        "status": "declared",
        "contract_kind": "delegation_zero_run",
        "zero_run": True,
        "zero_run_decision": "unknown",
        "zero_run_basis": "No physical leaf was started.",
        "physical_run_started": False,
    })
    ctx = SimpleNamespace(
        task_id=task_id,
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "physical_started": False,
            "exact_start_pending": True,
        },
    )
    monkeypatch.setattr(delegate, "_delegate_start", lambda *_a, **_k: pytest.fail("must not start"))
    refused = json.loads(runtime.exact_start(
        ctx,
        "OBJECTIVE\nBuild the patch",
        {"snapshot": snapshot, "compiled_work_order": True},
    ))
    assert refused["reason"] == "zero_run_already_recorded"
    assert ctx._configured_actor_bootstrap["zero_run_decision"] == "unknown"


@pytest.mark.parametrize("gap_kind", ["malformed", "unreadable", "invalid_schema"])
def test_actor_first_exact_start_blocks_unknown_zero_run_evidence(
    monkeypatch, tmp_path, gap_kind,
):
    import ouroboros.outcome_receipt_store as receipt_store
    import ouroboros.subagent_bootstrap as bootstrap
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.outcomes import verification_receipts_path

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task_id = f"unknown-zero-run-{gap_kind}"
    path = verification_receipts_path(tmp_path, task_id, create=True)
    if gap_kind == "invalid_schema":
        path.write_text(json.dumps({
            "status": "declared",
            "contract_kind": "delegation_zero_run",
            "zero_run": "false",
            "zero_run_decision": "complete",
            "zero_run_basis": "malformed boolean",
            "physical_run_started": False,
        }) + "\n")
    else:
        path.write_text('{"contract_kind":"delegation_zero_run","zero_run":true\n')
    if gap_kind == "unreadable":
        monkeypatch.setattr(
            receipt_store,
            "iter_jsonl_objects",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        )
    ctx = SimpleNamespace(
        task_id=task_id,
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={},
    )
    task = {
        "id": task_id,
        "objective": "Inspect the evidence",
        "expected_output": "A truthful result",
        "configured_subagent": snapshot,
        "task_contract": {
            "objective": "Inspect the evidence",
            "expected_output": "A truthful result",
        },
    }
    startup = json.loads(bootstrap._prepare_actor_first_bootstrap(
        ctx, task, SimpleNamespace(blocked=False, executor_resolution=None),
    ))
    assert startup["startup"]["zero_run_evidence_status"] == "unknown"
    assert startup["startup"]["exact_start_pending"] is False
    monkeypatch.setattr(
        delegate, "_delegate_start", lambda *_a, **_k: pytest.fail("must not start"),
    )

    refused = json.loads(runtime.exact_start(
        ctx,
        "OBJECTIVE\nBuild the patch",
        {"snapshot": snapshot, "compiled_work_order": True},
    ))

    assert refused["status"] == "refused"
    assert refused["reason"] == "zero_run_evidence_unavailable"
    assert ctx._configured_actor_bootstrap["zero_run_evidence_status"] == "unknown"
    assert ctx._configured_actor_bootstrap["exact_start_pending"] is False


def test_valid_zero_run_wins_over_unrelated_malformed_receipt(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.outcomes import append_verification_receipt, verification_receipts_path

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task_id = "valid-zero-run-with-gap"
    path = verification_receipts_path(tmp_path, task_id, create=True)
    path.write_text("not-json\n", encoding="utf-8")
    assert append_verification_receipt(tmp_path, task_id, {
        "status": "declared",
        "contract_kind": "delegation_zero_run",
        "zero_run": True,
        "zero_run_decision": "complete",
        "zero_run_basis": "No physical leaf was started.",
        "physical_run_started": False,
    })
    ctx = SimpleNamespace(
        task_id=task_id,
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "physical_started": False,
            "exact_start_pending": True,
        },
    )
    monkeypatch.setattr(
        delegate, "_delegate_start", lambda *_a, **_k: pytest.fail("must not start"),
    )

    refused = json.loads(runtime.exact_start(
        ctx,
        "OBJECTIVE\nBuild the patch",
        {"snapshot": snapshot, "compiled_work_order": True},
    ))

    assert refused["reason"] == "zero_run_already_recorded"
    assert ctx._configured_actor_bootstrap["zero_run_decision"] == "complete"
    assert "zero_run_evidence_status" not in ctx._configured_actor_bootstrap


def test_actor_first_delegate_start_rejects_alternate_snapshot(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "snapshot": snapshot,
            "selected_subagent_id": "session-builder",
            "canonical_work_order": "OBJECTIVE\nBuild the patch",
            "work_order_fingerprint": "full-work-order-sha",
            "work_order_chars": 23,
        },
    )
    out = json.loads(runtime.delegate_start_entry(
        ctx, "retarget me", subagent_id="other-session",
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "configured_actor_route_mismatch"
    assert out["host_fallback"] is False


def test_actor_first_retry_cannot_turn_coordination_prompt_into_work_order_prefix(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.claudexor_daemon as daemon

    class Gateway:
        def harnesses(self):
            return [{"id": "codex", "manifest": {"capabilities": {}}}]

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", Gateway)
    snapshot = _snapshot(_settings(_session_row()), "session-builder")

    ctx = SimpleNamespace(
        task_id="child-over-budget", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "snapshot": snapshot,
            "selected_subagent_id": "session-builder",
            "canonical_work_order": "",
            "source_request": {"kind": "complete_work_order", "sha256": "f" * 64},
            "source_channel": {"status": "unverified", "route": "codex"},
            "work_order_fingerprint": "f" * 64,
            "work_order_chars": 250001,
        },
    )
    monkeypatch.setattr(
        runtime, "exact_start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an over-budget retry must refuse before exact_start")
        ),
    )
    out = json.loads(runtime.delegate_start_entry(
        ctx, "coordination text is not the canonical assignment", retry_of="inv-1",
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "work_order_source_channel_unverified"


def test_actor_first_over_budget_retry_reprobes_source_channel(monkeypatch, tmp_path):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagent_runtime as runtime

    class Gateway:
        def harnesses(self):
            return [{
                "id": "codex",
                "manifest": {"capabilities": {"interactive": True}},
            }]

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", Gateway)
    starts = []
    monkeypatch.setattr(runtime, "exact_start", lambda _ctx, prompt, spec: (
        starts.append((prompt, spec))
        or json.dumps({"status": "started", "run_id": "run-retry"})
    ))
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    ctx = SimpleNamespace(
        task_id="child-over-budget-retry",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "snapshot": snapshot,
            "selected_subagent_id": "session-builder",
            "canonical_work_order": "",
            "source_prompt": "WORK ORDER SOURCE REQUEST\ncoverage=partial",
            "source_request": {"kind": "complete_work_order", "sha256": "f" * 64},
            "source_channel": {"status": "unavailable", "route": "cursor"},
            "work_order_fingerprint": "f" * 64,
            "work_order_chars": 250001,
        },
    )

    out = json.loads(runtime.delegate_start_entry(ctx, "ignored", retry_of="inv-1"))

    assert out["status"] == "started"
    assert len(starts) == 1
    assert starts[0][1]["retry_of"] == "inv-1"
    assert ctx._configured_actor_bootstrap["source_channel"]["route"] == "codex"


def test_actor_first_bootstrap_adopts_existing_handoff_without_new_start(monkeypatch, tmp_path):
    import ouroboros.delegate_recovery as recovery
    import ouroboros.subagent_runtime as runtime
    from ouroboros.subagent_bootstrap import bootstrap_before_context

    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    monkeypatch.setattr(recovery, "adopt_handoff", lambda _ctx, _task: {
        "status": "adopted", "run_id": "run-recovered",
    })
    monkeypatch.setattr(
        "ouroboros.delegate_supervision.supervised_wait",
        lambda *_a, **_kw: pytest.fail("the host must not wait inside bootstrap (owner 1=A)"),
    )
    dispatch = SimpleNamespace(
        blocked=False,
        executor="harness",
        executor_resolution=SimpleNamespace(reason="harness_ready"),
    )
    ctx = SimpleNamespace(task_id="child-recovery", drive_root=tmp_path, budget_drive_root=str(tmp_path))
    out = json.loads(bootstrap_before_context(
        ctx,
        {
            "id": "child-recovery",
            "objective": "Recover",
            "configured_subagent": snapshot,
            "task_contract": {"objective": "Recover"},
        },
        dispatch,
    ))
    # Owner 1=A: an adopted live run is handed to the model's first round
    # immediately; waiting is the model's own delegate_wait decision.
    assert out["status"] == "configured_session_started"
    assert out["recovery"]["run_id"] == "run-recovered"
    assert out["run_id"] == "run-recovered"
    assert ctx._configured_actor_bootstrap["selected_subagent_id"] == "session-builder"
    assert ctx._configured_actor_bootstrap["canonical_work_order"]
    assert ctx._configured_actor_bootstrap["physical_started"] is True
    mismatch = json.loads(runtime.delegate_start_entry(
        ctx, "switch route", subagent_id="another-session",
    ))
    assert mismatch["reason"] == "configured_actor_route_mismatch"


def test_atomic_start_attempt_does_not_trigger_legacy_no_delegate_nudges(tmp_path):
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.loop import _forced_delegation_note, _nanny_finalization_message
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child1")
    ctx.budget_drive_root = str(tmp_path)
    ctx._nanny_route_dispatched = True
    record_start_blocked(ctx, "child1", "configured_route_unavailable")
    tools = SimpleNamespace(
        _ctx=ctx,
        available_tools=lambda: ["delegate_start", "delegate_wait", "delegate_cancel"],
    )
    assert _nanny_finalization_message(tools, tmp_path, "child1") == ""
    assert _forced_delegation_note(ctx, {"tool_calls": []}) == ""


def test_quiet_windows_renew_and_terminal_plus_mailbox_coalesce(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_supervision import supervised_wait
    from ouroboros.owner_mailbox import write_task_message

    ctx = SimpleNamespace(
        task_id="child1",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_metadata={"configured_subagent": {"config_fingerprint": "fp"}},
    )
    calls = []

    def wait_once(_ctx, run_id, _sec, since):
        calls.append((run_id, since))
        if len(calls) < 3:
            return json.dumps({"status": "no_progress", "run_id": run_id, "last_seq": len(calls)})
        write_task_message(
            tmp_path, "new direction", "child1", source_task_id="parent1",
            provenance="ancestor_task", msg_id="m1",
        )
        return json.dumps({"status": "completed", "run_id": run_id, "last_seq": 3})

    out = json.loads(supervised_wait(ctx, "run-1", wait_once=wait_once))
    assert len(calls) == 3
    assert out["status"] == "completed"
    [message] = out["wake_events"]
    assert message["msg_id"] == "m1"
    assert message["provenance"] == "ancestor_task"
    assert message["source_task_id"] == "parent1"
    assert message["text"] == "new direction"
    assert out["supervision_wake_id"]
    event_types = {
        json.loads(line)["type"]
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    }
    assert {
        "delegate_supervision_wait_entered",
        "delegate_supervision_wait_renewed",
        "delegate_supervision_wake_pending",
    } <= event_types


def test_pending_wake_replays_until_post_injection_ack(tmp_path):
    from ouroboros.delegate_supervision import acknowledge_pending_wake, supervised_wait
    from ouroboros.owner_mailbox import (
        acknowledged_task_message_ids, write_task_message,
    )
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_metadata={"configured_subagent": {"config_fingerprint": "fp"}},
    )
    assert write_task_message(
        tmp_path, "full durable direction", "child1", source_task_id="parent1",
        provenance="ancestor_task", msg_id="m1",
    )
    first = json.loads(supervised_wait(
        ctx, "run-1",
        wait_once=lambda *_a, **_k: json.dumps({
            "status": "no_progress", "run_id": "run-1", "last_seq": 4,
        }),
    ))
    assert first["wake_events"][0]["text"] == "full durable direction"
    assert acknowledged_task_message_ids(tmp_path, "child1") == set()
    replay = json.loads(supervised_wait(
        ctx, "run-1",
        wait_once=lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("an unacknowledged wake must replay before another poll")
        ),
    ))
    assert replay == first
    assert acknowledge_pending_wake(ctx, replay)
    assert acknowledged_task_message_ids(tmp_path, "child1") == {"m1"}


def test_one_shot_checkpoint_is_reasoned_and_consumed(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    import ouroboros.delegate_supervision as supervision

    now = [100.0]
    monkeypatch.setattr(supervision.time, "time", lambda: now[0])
    contract = build_task_contract({})
    write_task_result(
        tmp_path, "child1", STATUS_RUNNING, root_task_id="child1",
        delegation_role="root", task_contract=contract,
    )
    ctx = SimpleNamespace(
        task_id="child1", drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_contract=contract,
        task_metadata={
            "configured_subagent": {"config_fingerprint": "fp"},
            "root_task_id": "child1", "delegation_role": "root",
            "task_contract": contract, "budget_drive_root": str(tmp_path),
        },
    )

    def wait_once(_ctx, run_id, _sec, _since):
        now[0] += 2
        return json.dumps({"status": "no_progress", "run_id": run_id, "last_seq": 0})

    out = json.loads(supervision.supervised_wait(
        ctx, "run-1", checkpoint_after_sec=1,
        checkpoint_reason="inspect a promised artifact", wait_once=wait_once,
    ))
    wake_id = out.pop("supervision_wake_id")
    assert wake_id
    coordination_context = out.pop("coordination_context")
    assert coordination_context["root_task_id"] == "child1"
    assert coordination_context["parent_intent"]["state"] == "absent"
    assert coordination_context["time"]["state"] == "not_set"
    assert coordination_context["review_capacity"]["state"] == "available"
    assert out == {
        "status": "inspection_checkpoint",
        "run_id": "run-1",
        "reason": "inspect a promised artifact",
        "last_seq": 0,
    }
    state = supervision.supervision_checkpoint(ctx)
    assert state["checkpoint"]["consumed"] is True
    assert state["checkpoint"]["consumed_by"] == "scheduled_checkpoint"
    event_types = [
        json.loads(line)["type"]
        for line in custody.event_log_path(tmp_path).read_text().splitlines()
    ]
    assert "delegate_supervision_checkpoint_scheduled" in event_types
    assert "delegate_supervision_checkpoint_consumed" in event_types


def test_task_message_provenance_never_spoofs_owner(tmp_path):
    from ouroboros.loop import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_task_message

    assert write_task_message(
        tmp_path, "peer finding", "child1", source_task_id="parent1",
        provenance="peer_via_ancestor", relayed_from_task_id="peer2", msg_id="m1",
    )
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "child1", None, set())
    rendered = json.dumps(messages, ensure_ascii=False)
    assert "Message from task peer2, relayed by ancestor parent1" in rendered
    assert "Message from my human" not in rendered


def test_ancestor_can_relay_to_a_true_grandchild_without_owner_spoof(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
    from ouroboros.tools.core import _forward_to_worker
    from ouroboros.tools.registry import ToolContext
    child_drive = tmp_path / "target-drive"
    child_drive.mkdir()
    write_task_result(
        tmp_path, "middle", STATUS_RUNNING, parent_task_id="parent",
        root_task_id="parent", result="running",
    )
    write_task_result(
        tmp_path, "target", STATUS_RUNNING, parent_task_id="middle",
        root_task_id="parent", child_drive_root=str(child_drive), result="running",
    )
    write_task_result(
        tmp_path, "peer", STATUS_COMPLETED, parent_task_id="parent",
        root_task_id="parent", result="peer evidence",
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="parent")
    ctx.budget_drive_root = str(tmp_path)
    assert _forward_to_worker(
        ctx, "target", "peer evidence", relayed_from_task_id="peer",
    ) == "Message forwarded to task target"
    [entry] = drain_owner_entries(child_drive, "target", seen_ids=set())
    assert entry["kind"] == "task_message"
    assert entry["provenance"] == "peer_via_ancestor"
    assert entry["source_task_id"] == "parent"
    assert entry["relayed_from_task_id"] == "peer"

    # A shared root label plus a parent-cycle is not a descendant proof.
    write_task_result(
        tmp_path, "cycle-a", STATUS_RUNNING, parent_task_id="cycle-b",
        root_task_id="parent", child_drive_root=str(child_drive), result="running",
    )
    write_task_result(
        tmp_path, "cycle-b", STATUS_RUNNING, parent_task_id="cycle-a",
        root_task_id="parent", result="running",
    )
    assert "TASK_FORBIDDEN" in _forward_to_worker(ctx, "cycle-a", "must not deliver")


def test_replacement_is_refused_before_gateway_or_post(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext
    import ouroboros.claudexor_daemon as daemon

    custody._CUSTODY.clear()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "codex=gpt-5.6-sol:high")
    old = custody.RunCustody(
        run_id="run-old", task_id="child1", route_id="codex", snapshot_id="snap-old",
    )
    custody.record_started(tmp_path, old)
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-old", "task_id": "child1"})
    custody.record_patch_captured(tmp_path, old)
    custody._CUSTODY.clear()
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: (_ for _ in ()).throw(
        AssertionError("gateway must not open before settlement guard")))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "child1"
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    out = json.loads(delegate.exact_start(
        ctx, "replacement work", {"snapshot": snapshot},
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "replacement_requires_settlement"
    assert out["undisposed_patch_run_ids"] == ["run-old"]


def test_replacement_refuses_unreadable_custody_before_fail_soft_scan(
    monkeypatch, tmp_path,
):
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_recovery
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: True)
    monkeypatch.setattr(
        delegate_recovery,
        "unsettled_start_ids",
        lambda *_a, **_k: pytest.fail("unreadable custody must stop before scan"),
    )
    monkeypatch.setattr(
        daemon,
        "ensure_owned_gateway",
        lambda: pytest.fail("unreadable custody must stop before gateway work"),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "child-unknown"
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    out = json.loads(delegate.exact_start(
        ctx, "replacement work", {"snapshot": snapshot},
    ))
    assert out["status"] == "refused"
    assert out["reason"] == "replacement_custody_unknown"


def test_terminal_boundary_reaudits_durable_pending_starts(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_terminal
    from ouroboros.task_results import load_task_result
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id="child1", invocation_id="inv-1",
        idempotency_key="inv-1", max_seconds=30, request={"prompt": "complete brief"},
        project_id="project-1", project_owned=False, route="codex",
    )
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: [])
    audit = delegate_terminal.terminal_reconcile_task(
        tmp_path, "child1", trigger="test_terminal",
    )
    assert audit["pending_invocation_ids"] == ["inv-1"]
    assert audit["unreconciled"] == ["invocation:inv-1"]
    delegate_terminal.record_terminal_reconciliation(tmp_path, "child1", audit)
    result = load_task_result(tmp_path, "child1")
    assert result["delegated_runs_unreconciled"] == ["invocation:inv-1"]


def test_worker_crash_mismatch_vetoes_without_post_and_cause_matrix(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.tools.registry import ToolContext

    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1",
        "_attempt": 2,
        "configured_subagent": snapshot,
        "task_constraint": {},
        "task_contract": {},
        "drive_root": str(tmp_path),
    }
    authority = recovery.authority_fingerprint_from_task(task)
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=authority,
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    handoff = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1, new_attempt=2, worker_id=1, exitcode=1,
    )
    assert handoff["run_id"] == "run-1"
    assert "child1" in recovery.recoverable_task_ids(tmp_path)
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: [])
    bad = {**task, "configured_subagent": {**snapshot, "config_fingerprint": "different"}}
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "child1"
    ctx.budget_drive_root = str(tmp_path)
    result = recovery.adopt_handoff(ctx, bad)
    assert result == {"status": "recovery_required", "reason": "successor_binding_mismatch"}
    assert recovery._read(tmp_path, "child1")["status"] == "vetoed"

    planned = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART,
        old_attempt=1, new_attempt=2,
    )
    assert planned["cause"] == recovery.CAUSE_PLANNED_SELF_RESTART
    assert "child1" in recovery.recoverable_task_ids(tmp_path)


def test_same_run_crash_adoption_uses_exact_binding_and_never_posts(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.delegate_recovery as recovery
    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.tools.registry import ToolContext

    custody._CUSTODY.clear()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1", "_attempt": 2,
        "configured_subagent": snapshot,
        "workspace_root": str(workspace), "workspace_mode": "workspace_write",
        "drive_root": str(tmp_path), "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
    }
    ctx = ToolContext(
        repo_dir=workspace, drive_root=tmp_path, workspace_root=workspace,
        workspace_mode="workspace_write", task_id="child1",
        task_metadata={
            "workspace_root": str(workspace), "workspace_mode": "workspace_write",
            "drive_root": str(tmp_path), "configured_subagent": snapshot,
        },
        task_constraint=normalize_task_constraint({}),
        task_contract=task["task_contract"],
    )
    ctx.budget_drive_root = str(tmp_path)
    authority = recovery.authority_fingerprint_from_context(ctx)
    assert authority == recovery.authority_fingerprint_from_task(task)
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=authority,
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    handoff = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1, new_attempt=2, worker_id=1, exitcode=1,
    )
    assert handoff["run_id"] == "run-1"

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-1"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setattr(delegate, "exact_start", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("a verified live run must be adopted without a second POST")
    ))
    assert recovery.adopt_handoff(ctx, task) == {
        "status": "adopted", "run_id": "run-1", "cause": recovery.CAUSE_WORKER_CRASH,
    }


def test_settled_terminal_wake_survives_worker_crash_without_a_new_post(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery
    import ouroboros.tools.delegate as delegate
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.tools.registry import ToolContext
    from ouroboros.utils import atomic_write_json
    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1", "_attempt": 2, "configured_subagent": snapshot,
        "drive_root": str(tmp_path), "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=authority,
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    terminal = {
        "status": "completed", "run_id": "run-1", "result": "full terminal result",
        "supervision_wake_id": "wake-terminal",
    }
    wait_path = tmp_path / "state" / "delegate_supervision" / "child1.json"
    wait_path.parent.mkdir(parents=True)
    atomic_write_json(wait_path, {
        "schema": 1, "run_id": "run-1", "status": "wake_pending",
        "pending_wake": {
            "wake_id": "wake-terminal", "payload": terminal,
            "mailbox_ids": [], "interaction_ids": [],
        },
    })
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-1", "task_id": "child1"})
    custody._CUSTODY.clear()
    row = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1, new_attempt=2, worker_id=1, exitcode=1,
    )
    assert row["settled_terminal"] is True
    monkeypatch.setattr(delegate, "exact_start", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("settled terminal recovery must never issue another POST")
    ))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child1")
    out = recovery.adopt_handoff(ctx, task)
    assert out["status"] == "settled_recovered"
    assert out["wake"] == terminal


def test_planned_restart_selectively_restores_sleeping_leaf_and_wait_state(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.delegate_recovery as recovery
    import ouroboros.delegate_interactions as interactions
    import ouroboros.tools.delegate as delegate
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.subagent_work_order import work_order_fingerprint
    from ouroboros.tools.registry import ToolContext
    from ouroboros.utils import atomic_write_json

    custody._CUSTODY.clear()
    interactions._REPORTED_INTERACTIONS.clear()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1", "_attempt": 1, "configured_subagent": snapshot,
        "workspace_root": str(workspace), "workspace_mode": "workspace_write",
        "drive_root": str(tmp_path), "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=authority,
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    wait_path = tmp_path / "state" / "delegate_supervision" / "child1.json"
    wait_path.parent.mkdir(parents=True)
    atomic_write_json(wait_path, {
        "schema": 1, "run_id": "run-1", "status": "wake_pending", "journal_cursor": 7,
        "mailbox_acknowledged_ids": ["m-old"],
        "interaction_acknowledged_ids": [],
        "pending_wake": {
            "wake_id": "wake-1",
            "payload": {
                "status": "waiting_on_user", "supervision_wake_id": "wake-1",
                "pending_interactions": [{"interaction_id": "quiz-1", "question": "Choose"}],
            },
            "mailbox_ids": [], "interaction_ids": ["quiz-1"],
        },
        "checkpoint": {"reason": "inspect artifact", "consumed": False},
    })
    [preserved] = recovery.prepare_planned_restart_handoffs(
        tmp_path, {"child1": {"task": task, "attempt": 1, "worker_id": 3}},
    )
    assert preserved == "child1"
    handoff = recovery._read(tmp_path, "child1")
    assert handoff["no_resume_veto_causes"] == list(recovery.NO_RESUME_CAUSES)

    old_pid = handoff["supervisor_pid"]
    assert not recovery.acknowledge_observed_restart_exit(
        tmp_path, supervisor_pid=old_pid + 1, exit_code=42,
    )
    assert not recovery.acknowledge_observed_restart_exit(
        tmp_path, supervisor_pid=old_pid, exit_code=1,
    )
    assert recovery.acknowledge_observed_restart_exit(
        tmp_path, supervisor_pid=old_pid, exit_code=42,
    )
    monkeypatch.setattr(recovery.os, "getpid", lambda: old_pid + 100_000)
    monkeypatch.setattr(recovery, "_pid_alive", lambda _pid: False)

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-1"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    successor = {**task, "_attempt": 2}
    assert recovery.pre_adopt_planned_handoffs(tmp_path, [successor]) == {"child1"}
    ctx = ToolContext(
        repo_dir=workspace, drive_root=tmp_path, workspace_root=workspace,
        workspace_mode="workspace_write", task_id="child1",
        task_metadata={"drive_root": str(tmp_path)},
        task_constraint=normalize_task_constraint({}), task_contract=task["task_contract"],
    )
    monkeypatch.setattr(delegate, "exact_start", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("planned handoff must adopt, never POST")
    ))
    adopted = recovery.adopt_handoff(ctx, successor)
    assert adopted["status"] == "adopted"
    assert adopted["wake"]["pending_interactions"][0]["question"] == "Choose"
    restored = json.loads(wait_path.read_text(encoding="utf-8"))
    assert restored["journal_cursor"] == 7
    assert restored["mailbox_acknowledged_ids"] == ["m-old"]
    assert restored["checkpoint"]["reason"] == "inspect artifact"
    assert "run-1" not in interactions._REPORTED_INTERACTIONS
    from ouroboros.delegate_supervision import acknowledge_pending_wake
    assert acknowledge_pending_wake(ctx, adopted["wake"])
    assert interactions._REPORTED_INTERACTIONS["run-1"] == frozenset({"quiz-1"})


def test_only_approved_restart_causes_reserve_and_abrupt_gap_vetoes(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery
    from ouroboros.subagent_work_order import work_order_fingerprint

    custody._CUSTODY.clear()
    snapshot = _snapshot(_settings(_session_row()), "session-builder")
    task = {
        "id": "child1", "_attempt": 1, "configured_subagent": snapshot,
        "drive_root": str(tmp_path), "task_constraint": {}, "task_contract": {},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-1", task_id="child1", route_id="codex",
        selected_subagent_id="session-builder",
        config_fingerprint=snapshot["config_fingerprint"],
        authority_fingerprint=authority,
        work_order_fingerprint=work_order_fingerprint(task),
    ))
    for cause in recovery.NO_RESUME_CAUSES:
        assert recovery.prepare_handoff(
            tmp_path, task, cause=cause, old_attempt=1, new_attempt=2,
        ) == {}
    row = recovery.prepare_handoff(
        tmp_path, task, cause=recovery.CAUSE_PLANNED_SELF_RESTART,
        old_attempt=1, new_attempt=2,
    )
    assert recovery.recoverable_task_ids(tmp_path) == {"child1"}
    monkeypatch.setattr(recovery, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(recovery.os, "getpid", lambda: int(row["supervisor_pid"]) + 1)
    assert recovery.recoverable_task_ids(tmp_path) == set()
    monkeypatch.setattr(custody, "reconcile_task_runs", lambda *_a, **_k: [])
    assert recovery.pre_adopt_planned_handoffs(tmp_path, []) == set()
    assert recovery._read(tmp_path, "child1")["veto_reason"] == "restart_transaction_missing"
