"""Focused regressions for the bounded synthesis runtime review fixes."""

from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import pytest
from tests._governance_docs_shared import governance_doc_text


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
        "settled": False,
        "terminal_state": "",
    }


def test_d24_reserves_multiple_current_custody_holders_independently(tmp_path):
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
    handoff = recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=3,
        exitcode=1,
    )
    assert handoff["schema"] == 2
    assert [(row["run_id"], row["selected_subagent_id"], row["status"])
            for row in handoff["holders"]] == [
                ("run-a", "session-a", "reserved"), ("run-b", "session-b", "reserved")]


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
    handoff = recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=3,
        exitcode=1,
    )
    assert len(handoff["holders"]) == 1
    holder = handoff["holders"][0]
    assert holder["pending_invocation_id"] == "invocation-b"
    assert holder["selected_subagent_id"] == "session-b"
    assert holder["work_order_fingerprint"] == "b" * 64
    assert holder["status"] == "reserved"


_CHILD = {"parent_task_id": "root-1", "root_task_id": "root-1", "delegation_role": "subagent"}
_OPEN_AUDIT = {"trigger": "loop_exit", "audit_status": "ok", "unreconciled": ["run-open"],
               "open_run_ids": ["run-open"], "pending_invocation_ids": [], "undisposed_patch_run_ids": [],
               "terminal_runs": []}


@pytest.mark.parametrize(("reason", "execution", "headline", "healed"), [
    ("completed", {"status": "ok", "reason_code": ""}, "Done with warnings", "Done"),
    ("provider_unavailable", {"status": "infra_failed", "reason_code": "provider_unavailable"}, "Failed", "Failed"),
    ("round_limit", {"status": "best_effort", "reason_code": "round_limit"}, "Done with warnings", "Done with warnings"),
], ids=["ordinary", "provider-failure", "truncation-rail"])
def test_an_open_custody_debt_stores_beside_the_cause_it_never_replaces(tmp_path, reason, execution, headline, healed):
    """#1583: an unreconciled own run is a DEBT beside the task's cause, never its cause.

    Through the real store path, the row keeps the derived reason and every earned
    axis byte-for-byte (no custody warning is added, nothing is laundered into an
    infrastructure failure, a truncation rail keeps its code), while the current
    audit lands as the row's own list and envelope, the facts that heal."""
    import copy

    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES, _completion_verdict, completion_status_label
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    write_task_result(tmp_path, "t-debt", STATUS_RUNNING)
    derived = {"reason_code": reason, "outcome_axes": {
        "execution": dict(execution), "objective": {"status": "met", "source": "review"},
        "review": {"status": "changes_requested", "trigger": "acceptance"}, "artifacts": {"status": "present"}}}
    before = copy.deepcopy(derived)
    _store_task_result(SimpleNamespace(drive_root=tmp_path), {**_CHILD, "id": "t-debt", "budget_drive_root": str(tmp_path)},
                       "the answer", {"delegate_terminal_reconciliation": dict(_OPEN_AUDIT)}, {}, loop_outcome=derived)
    stored = load_task_result(tmp_path, "t-debt")
    assert derived == before, "the caller's derived outcome is not mutated"
    assert stored["reason_code"] == reason
    axes = stored["outcome_axes"]
    earned = normalize_outcome_axes({"outcome_axes": before["outcome_axes"]})
    assert {key: axes[key] for key in ("objective", "review")} == {key: earned[key] for key in ("objective", "review")}
    assert "warnings" not in axes["objective"], "no custody warning is added to what the task earned"
    assert axes["execution"]["status"] == execution["status"]
    assert axes["execution"].get("reason_code", "") == execution["reason_code"]
    assert stored["delegated_runs_unreconciled"] == ["run-open"]
    assert stored["delegate_terminal_reconciliation"]["open_run_ids"] == ["run-open"]
    assert completion_status_label(stored, {}) == headline
    assert _completion_verdict(stored, {}).endswith(TASK_CAUSE_PHRASES["delegated_custody_unreconciled"])
    # The write-side heal rewrites only the list; the earned cause and axes then speak alone.
    write_task_result(tmp_path, "t-debt", stored["status"], delegated_runs_unreconciled=[])
    healed_row = load_task_result(tmp_path, "t-debt")
    assert healed_row["reason_code"] == reason and healed_row["outcome_axes"] == axes
    assert completion_status_label(healed_row, {}) == healed
    assert TASK_CAUSE_PHRASES["delegated_custody_unreconciled"] not in _completion_verdict(healed_row, {})


def test_custody_debt_task_stores_as_done_with_warnings(tmp_path):
    """Owner-visible consequence: the card reads Done with warnings from the row's
    own list, the stored reason stays the loop's own, and the review mirror keeps
    the REAL review axis instead of a skipped stand-in."""
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.project_dialogue import completion_status_label
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
        env, {**_CHILD, "id": "t-4", "budget_drive_root": str(tmp_path)},
        "the answer", {}, {}, loop_outcome=loop_outcome,
    )
    stored = load_task_result(tmp_path, "t-4")
    assert stored["status"] == STATUS_COMPLETED
    assert stored["reason_code"] == "completed"
    assert stored["delegated_runs_unreconciled"] == ["run-open"]
    axes = stored["outcome_axes"]
    assert "warning" not in axes["objective"] and "warnings" not in axes["objective"]
    assert completion_status_label(stored, {}) == "Done with warnings"
    assert axes["review"]["status"] == "approved"
    assert stored.get("review_status") == axes["review"]


@pytest.mark.parametrize("mode", ["local_readonly_subagent", "acting_subagent"])
def test_descendant_forwarding_is_visible_and_executable_for_both_subagent_profiles(
    tmp_path, monkeypatch, mode,
):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")

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
    monkeypatch, tmp_path, route_kind,
):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagents as subagents
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch
    from tests._shared import write_legacy_capability_rows

    nanny_model = f"openai/review-fix-{route_kind}"
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    write_legacy_capability_rows(tmp_path, effort_ceilings={nanny_model: "low"})
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
    assert schema["required"] == []  # retry_of may omit prompt (#1614); recipes below still name it
    assert not ({"anyOf", "oneOf", "allOf"} & schema.keys())
    assert "required for direct starts including continue_from" in schema["properties"]["subagent_id"]["description"]
    assert "Omit subagent_id or receive a typed conflict" in (
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
    # The two reference books are map-level: whether they carry a recipe at all
    # is prose, not contract, so they are tolerant too (decision 7A); any recipe
    # they do carry must still be schema-valid.
    tolerant = {
        "docs/CHECKLISTS.md", "prompts/SYSTEM.md",
        "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md",
    }
    for relative in (*recipe_paths, "docs/CHECKLISTS.md"):
        text = governance_doc_text(relative, repo)
        recipes = re.findall(r"\bdelegate_start\(([^)]*)\)", text, flags=re.DOTALL)
        assert recipes or relative in tolerant, (
            f"expected at least one delegate_start recipe in {relative}"
        )
        for recipe in recipes:
            assert re.search(r"\bprompt\s*=", recipe), (relative, recipe)
            if not re.search(r"\bretry_of\s*=", recipe):
                # continue_from is the schema's third start selector: a NEW run
                # bound to a finished predecessor.
                assert "continue_from" in schema["properties"]
                direct_selector = re.search(
                    r"\b(?:subagent_id|continue_from)\s*=", recipe)
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
