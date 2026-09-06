"""Focused regressions for the bounded synthesis runtime review fixes."""

from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _clear_delegate_custody_memo():
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.clear()
    yield
    custody._CUSTODY.clear()


def _session_snapshot(subagent_id: str, config_fingerprint: str) -> dict:
    return {
        "schema": 1,
        "selected_subagent_id": subagent_id,
        "config_fingerprint": config_fingerprint,
        "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol"},
        "effort": "high",
    }


@pytest.mark.parametrize("has_initial_actor", [True, False])
def test_d24_adopts_unique_explicit_replacement_or_root_direct_holder(
    monkeypatch, tmp_path, has_initial_actor,
):
    """The current durable leaf, not the nanny's initial actor, owns recovery.

    The two configured rows intentionally share one list fingerprint.  ``session-b``
    is a later explicit same-nanny replacement when ``has_initial_actor`` is true,
    and an explicit root-direct leaf otherwise.
    """

    from ouroboros import delegate_custody as custody
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.delegate_recovery as recovery
    from ouroboros.tools.registry import ToolContext
    from ouroboros.utils import atomic_write_json

    workspace = tmp_path / "workspace"
    private_tree = tmp_path / "private-tree"
    workspace.mkdir()
    private_tree.mkdir()
    config_fingerprint = "configured-list-fingerprint"
    task = {
        "id": "nanny-1",
        "_attempt": 2,
        "workspace_root": str(workspace),
        "workspace_mode": "workspace_write",
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {"objective": "Initial assignment", "expected_output": "Patch"},
    }
    if has_initial_actor:
        task["configured_subagent"] = _session_snapshot("session-a", config_fingerprint)
    authority = recovery.authority_fingerprint_from_task(task)
    replacement_work_order = "b" * 64
    assert custody.record_started(
        tmp_path,
        custody.RunCustody(
            run_id="run-b",
            task_id="nanny-1",
            route_id="codex",
            selected_subagent_id="session-b",
            config_fingerprint=config_fingerprint,
            authority_fingerprint=authority,
            work_order_fingerprint=replacement_work_order,
            snapshot_id="snapshot-b",
            execution_root=str(private_tree),
            baseline_sha="abc123",
            target_root=str(workspace),
        ),
    )
    supervision = tmp_path / "state" / "delegate_supervision" / "nanny-1.json"
    supervision.parent.mkdir(parents=True)
    atomic_write_json(
        supervision,
        {"schema": 1, "run_id": "run-b", "status": "sleeping", "journal_cursor": 9},
    )

    handoff = recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=3,
        exitcode=1,
    )

    assert handoff["selected_subagent_id"] == "session-b"
    assert handoff["config_fingerprint"] == config_fingerprint
    assert handoff["work_order_fingerprint"] == replacement_work_order
    assert handoff["actor_binding_source"] == "current_custody_holder"
    assert handoff["snapshot_id"] == "snapshot-b"
    assert handoff["execution_root"] == str(private_tree)

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-b"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    ctx = ToolContext(repo_dir=workspace, drive_root=tmp_path, task_id="nanny-1")
    ctx.budget_drive_root = str(tmp_path)
    assert recovery.adopt_handoff(ctx, task) == {
        "status": "adopted",
        "run_id": "run-b",
        "cause": recovery.CAUSE_WORKER_CRASH,
    }


def test_d24_still_refuses_multiple_current_custody_holders(tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery

    task = {
        "id": "nanny-1",
        "_attempt": 2,
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    for suffix in ("a", "b"):
        assert custody.record_started(
            tmp_path,
            custody.RunCustody(
                run_id=f"run-{suffix}",
                task_id="nanny-1",
                route_id="codex",
                selected_subagent_id=f"session-{suffix}",
                config_fingerprint="configured-list-fingerprint",
                authority_fingerprint=authority,
                work_order_fingerprint=suffix * 64,
            ),
        )
    assert recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=3,
        exitcode=1,
    ) == {}


def test_d24_does_not_post_a_pending_replacement(monkeypatch, tmp_path):
    from ouroboros import delegate_custody as custody
    import ouroboros.delegate_recovery as recovery

    task = {
        "id": "nanny-1",
        "_attempt": 2,
        "configured_subagent": _session_snapshot("session-a", "configured-list-fingerprint"),
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {"objective": "Initial assignment", "expected_output": "Patch"},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    replacement_prompt = "Continue from the settled first leaf with this corrected direction."
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id="nanny-1",
        invocation_id="invocation-b",
        idempotency_key="invocation-b",
        max_seconds=300,
        request={"prompt": replacement_prompt, "primaryHarness": "codex"},
        project_id="project-1",
        project_owned=False,
        route="codex",
        selected_subagent_id="session-b",
        config_fingerprint="configured-list-fingerprint",
        authority_fingerprint=authority,
        work_order_fingerprint="b" * 64,
    )
    monkeypatch.setattr(
        "ouroboros.tools.delegate.exact_start",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not POST")),
    )
    assert recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=3,
        exitcode=1,
    ) == {}


def test_terminal_custody_outcome_reads_the_canonical_budget_root(tmp_path):
    from ouroboros.agent_task_pipeline import _apply_terminal_custody_outcome
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    child_drive = tmp_path / "child-drive"
    budget_root = tmp_path / "canonical-drive"
    child_drive.mkdir()
    write_task_result(
        budget_root,
        "child-1",
        STATUS_RUNNING,
        delegated_runs_unreconciled=["run-open"],
    )
    original = {"reason_code": "completed", "outcome_axes": {}}
    actual = _apply_terminal_custody_outcome(
        SimpleNamespace(drive_root=child_drive),
        {"id": "child-1", "budget_drive_root": str(budget_root)},
        original,
    )
    assert actual["reason_code"] == "delegated_custody_unreconciled"
    # An undisposed own patch is a DEBT, not an infrastructure failure: the debt
    # is ADDED on the objective axis and the derived axes are left alone.
    assert actual["outcome_axes"]["objective"]["warning"] == (
        "delegated_custody_unreconciled")
    assert actual["outcome_axes"].get("execution", {}).get("status") != "infra_failed"


def test_custody_debt_overlay_preserves_the_paid_verdicts_and_is_idempotent(tmp_path):
    from ouroboros.agent_task_pipeline import _apply_terminal_custody_outcome
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "t-1", STATUS_RUNNING,
                      delegated_runs_unreconciled=["run-open"])
    env = SimpleNamespace(drive_root=tmp_path)
    task = {"id": "t-1", "budget_drive_root": str(tmp_path)}
    derived = {"reason_code": "completed", "outcome_axes": {
        "execution": {"status": "ok", "reason_code": ""},
        "objective": {"status": "degraded", "source": "review"},
        "review": {"status": "changes_requested", "trigger": "acceptance"},
        "artifacts": {"status": "present"},
    }}
    once = _apply_terminal_custody_outcome(env, task, derived)
    axes = once["outcome_axes"]
    assert axes["execution"] == {"status": "ok", "reason_code": ""}
    assert axes["objective"]["status"] == "degraded"
    assert axes["review"] == {"status": "changes_requested", "trigger": "acceptance"}
    assert axes["artifacts"] == {"status": "present"}
    assert axes["objective"]["warnings"] == ["delegated_custody_unreconciled"]
    # The derived axes the caller passed in are NOT mutated in place.
    assert "warning" not in derived["outcome_axes"]["objective"]
    twice = _apply_terminal_custody_outcome(env, task, once)
    assert twice["outcome_axes"]["objective"]["warnings"] == [
        "delegated_custody_unreconciled"]


def test_custody_debt_does_not_launder_a_derived_infra_failure(tmp_path):
    from ouroboros.agent_task_pipeline import _apply_terminal_custody_outcome
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "t-2", STATUS_RUNNING,
                      delegated_runs_unreconciled=["run-open"])
    derived = {"reason_code": "provider_unavailable", "outcome_axes": {
        "execution": {"status": "infra_failed", "reason_code": "provider_unavailable"},
    }}
    actual = _apply_terminal_custody_outcome(
        SimpleNamespace(drive_root=tmp_path),
        {"id": "t-2", "budget_drive_root": str(tmp_path)}, derived)
    assert actual["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert actual["outcome_axes"]["objective"]["warning"] == (
        "delegated_custody_unreconciled")
    # Disclosed residual: the devtools-only infra codes are not in the runtime
    # best-effort set, so the custody code still takes the one Reason line.
    assert actual["reason_code"] == "delegated_custody_unreconciled"


def test_custody_debt_preserves_a_truncation_rail_reason_code(tmp_path):
    from ouroboros.agent_task_pipeline import _apply_terminal_custody_outcome
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    write_task_result(tmp_path, "t-3", STATUS_RUNNING,
                      delegated_runs_unreconciled=["run-open"])
    derived = {"reason_code": "round_limit", "outcome_axes": {
        "execution": {"status": "ok", "reason_code": "round_limit"},
    }}
    actual = _apply_terminal_custody_outcome(
        SimpleNamespace(drive_root=tmp_path),
        {"id": "t-3", "budget_drive_root": str(tmp_path)}, derived)
    assert actual["reason_code"] == "round_limit"
    assert actual["outcome_axes"]["objective"]["warning"] == (
        "delegated_custody_unreconciled")


def test_custody_debt_task_stores_as_done_with_warnings(tmp_path):
    """Owner-visible consequence: the card reads Done with warnings, and the
    review mirror keeps the REAL review axis instead of a skipped stand-in."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.task_results import (
        STATUS_COMPLETED, STATUS_RUNNING, load_task_result, write_task_result,
    )

    write_task_result(tmp_path, "t-4", STATUS_RUNNING,
                      delegated_runs_unreconciled=["run-open"])
    env = SimpleNamespace(drive_root=tmp_path)
    loop_outcome = {"reason_code": "completed", "outcome_axes": {
        "execution": {"status": "ok", "reason_code": ""},
        "objective": {"status": "met", "source": "review"},
        "review": {"status": "approved", "trigger": "acceptance"},
        "artifacts": {"status": "present"},
    }}
    _store_task_result(
        env, {"id": "t-4", "budget_drive_root": str(tmp_path)},
        "the answer", {}, {}, loop_outcome=loop_outcome,
    )
    stored = load_task_result(tmp_path, "t-4")
    assert stored["status"] == STATUS_COMPLETED
    assert stored["reason_code"] == "delegated_custody_unreconciled"
    axes = stored["outcome_axes"]
    assert axes["objective"]["warning"] == "delegated_custody_unreconciled"
    assert axes["review"]["status"] == "approved"
    assert stored.get("review_status") == axes["review"]


@pytest.mark.parametrize("mode", ["local_readonly_subagent", "acting_subagent"])
def test_descendant_forwarding_is_visible_and_executable_for_both_subagent_profiles(
    tmp_path, mode,
):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo = tmp_path / "repo"
    canonical = tmp_path / "canonical"
    child_drive = tmp_path / "child"
    workspace = tmp_path / "workspace"
    for path in (repo, canonical, child_drive, workspace):
        path.mkdir()
    constraint = TaskConstraint(
        mode=mode,
        allow_enable=False,
        surface="external_workspace" if mode == "acting_subagent" else "",
        write_root=str(workspace) if mode == "acting_subagent" else "",
    )
    write_task_result(
        canonical,
        "child-1",
        STATUS_RUNNING,
        parent_task_id="parent-1",
        root_task_id="parent-1",
        child_drive_root=str(child_drive),
        result="running",
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=canonical)
    registry.set_context(
        ToolContext(
            repo_dir=repo,
            drive_root=canonical,
            workspace_root=workspace,
            workspace_mode="external",
            task_id="parent-1",
            task_metadata={"budget_drive_root": str(canonical)},
            task_constraint=constraint,
        )
    )
    assert registry.get_schema_by_name("forward_to_worker") is not None
    assert "Message forwarded" in registry.execute(
        "forward_to_worker", {"task_id": "child-1", "message": "inspect the new evidence"}
    )
    mailbox = child_drive / "memory" / "owner_mailbox" / "child-1.jsonl"
    assert "inspect the new evidence" in mailbox.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("target", "expected_provider"),
    [
        ("openai::gpt-5.5", "openai"),
        ("anthropic::claude-fable-5", "anthropic"),
        ("minimax::MiniMax-M2.5", "minimax"),
        ("cloudru::deepseek-v3.1", "cloudru"),
        ("gigachat::GigaChat-2-Max", "gigachat"),
        ("openai-compatible::custom-model", "openai-compatible"),
    ],
)
def test_api_actor_preserves_direct_provider_executable_route(target, expected_provider):
    from ouroboros.llm import LLMClient
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    dispatch = resolve_configured_actor_dispatch(
        {
            "id": "api-child",
            "configured_subagent": {
                "schema": 1,
                "selected_subagent_id": "api-builder",
                "config_fingerprint": "configured-list-fingerprint",
                "route": {"kind": "api_model", "target_id": target},
                "effort": "high",
            },
            "task_constraint": {},
        },
        task_type="research",
    )
    assert dispatch.lane.model == target
    assert LLMClient._resolve_remote_target(LLMClient(), dispatch.lane.model)["provider"] == expected_provider


def test_api_actor_strips_only_the_canonical_local_marker():
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    dispatch = resolve_configured_actor_dispatch(
        {
            "id": "local-child",
            "configured_subagent": {
                "schema": 1,
                "selected_subagent_id": "local-scout",
                "config_fingerprint": "configured-list-fingerprint",
                "route": {"kind": "api_model", "target_id": "qwen3:8b (local)"},
                "effort": "low",
            },
            "task_constraint": {},
        },
        task_type="research",
    )
    assert dispatch.lane.model == "qwen3:8b"
    assert dispatch.lane.use_local_model is True


@pytest.mark.parametrize("route_kind", ["api_model", "agent_session"])
def test_configured_actor_preserves_requested_effort_until_request_wire(
    monkeypatch, route_kind,
):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagents as subagents
    from ouroboros.llm import LLMClient
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    nanny_model = f"openai/review-fix-{route_kind}"
    monkeypatch.setitem(LLMClient._EFFORT_CEILING_CACHE, nanny_model, "low")
    if route_kind == "api_model":
        monkeypatch.setattr(
            "ouroboros.provider_models.model_has_credentials", lambda _model: True,
        )
        route = {"kind": route_kind, "target_id": f"openai::review-fix-{route_kind}"}
        cognitive = {}
    else:
        route = {"kind": route_kind, "target_id": "codex=gpt-5.6-sol"}
        cognitive = {"model": nanny_model, "effort": "max"}

        class Gateway:
            def close(self):
                pass

        monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
        monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))

    dispatch = resolve_configured_actor_dispatch(
        {
            "id": f"{route_kind}-child",
            "configured_subagent": {
                "schema": 1,
                "selected_subagent_id": f"{route_kind}-builder",
                "config_fingerprint": "configured-list-fingerprint",
                "route": route,
                "effort": "max",
            },
            "parent_cognitive_route": cognitive,
            "task_constraint": {},
        },
        task_type="research",
    )
    assert dispatch.delta.derived_effort == "max"
    assert dispatch.delta.effective_effort == "max"
    assert dispatch.delta.reduction_reasons == ()
    assert dispatch.delta.reason == ""
    assert dispatch.delta.reduced is False


def test_delegate_start_recipes_match_the_fresh_start_schema():
    from ouroboros.tools import delegate

    schema = next(
        entry.schema for entry in delegate.get_tools() if entry.name == "delegate_start"
    )["parameters"]
    assert schema["required"] == ["prompt"]
    assert not ({"anyOf", "oneOf", "allOf"} & schema.keys())
    assert "Required for a fresh start" in schema["properties"]["subagent_id"]["description"]
    assert "supplying both selectors is a typed conflict" in (
        schema["properties"]["retry_of"]["description"]
    )

    repo = pathlib.Path(__file__).parents[1]
    recipe_paths = (
        "prompts/SYSTEM.md",
        "docs/ARCHITECTURE.md",
        "docs/DEVELOPMENT.md",
        # #447 stage 3: the standing disclosures (which carry the delegate_start
        # recipe) moved to the binding archive; its recipes must stay
        # schema-valid too. The live checklist has none TODAY, so it is scanned
        # tolerantly below rather than dropped from coverage.
        "docs/CHECKLISTS_ARCHIVE.md",
        # v7 D01 split: the dispatch-note pair (and its delegate_start recipe
        # strings) moved to the agent_dispatch leaf; sdn re-exports the pair.
        "ouroboros/agent_dispatch.py",
        "ouroboros/tools/control.py",
        "ouroboros/tools/delegate.py",
        "ouroboros/tools/delegate_integration.py",
    )
    # The live checklist may legitimately have zero recipes (they moved to the
    # archive), but any it GAINS must stay schema-valid. prompts/SYSTEM.md is
    # tolerant for the same reason since the prompt audit: the recipe lives in
    # the delegate_start schema (the SSOT sent every round), and the prompt
    # only names the lane — a copy there would be the duplication class the
    # audit removed. Any recipe the prompt GAINS must still be schema-valid.
    tolerant = {"docs/CHECKLISTS.md", "prompts/SYSTEM.md"}
    for relative in (*recipe_paths, "docs/CHECKLISTS.md"):
        text = (repo / relative).read_text(encoding="utf-8")
        recipes = re.findall(r"\bdelegate_start\(([^)]*)\)", text, flags=re.DOTALL)
        assert recipes or relative in tolerant, (
            f"expected at least one delegate_start recipe in {relative}"
        )
        for recipe in recipes:
            assert re.search(r"\bprompt\s*=", recipe), (relative, recipe)
            if not re.search(r"\bretry_of\s*=", recipe):
                direct_selector = re.search(r"\bsubagent_id\s*=", recipe)
                actor_first_snapshot = re.fullmatch(
                    r"\s*prompt\s*=\s*(['\"])\1\s*", recipe,
                )
                assert direct_selector or actor_first_snapshot, (relative, recipe)


def test_delegate_recovery_uses_platform_pid_probe(monkeypatch):
    import ouroboros.delegate_recovery as recovery
    import ouroboros.platform_layer as platform_layer

    seen = []
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: seen.append(pid) or True)
    assert recovery._pid_alive(4312) is True
    assert seen == [4312]
