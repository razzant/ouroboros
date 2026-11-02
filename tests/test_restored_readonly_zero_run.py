"""Restored configured read-only actors retain only the typed zero-run receipt."""
import pytest

from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines
from tests.test_configured_session_restore import _known_custody, _restored_registry, _task

pytestmark = pytest.mark.serial


def _restored_actor(tmp_path, monkeypatch, carrier):
    from ouroboros import subagent_bootstrap
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.contracts.task_constraint import TaskConstraint

    registry = _restored_registry(tmp_path, monkeypatch, carrier)
    ctx = registry._ctx
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    task = _task(ctx, carrier)
    _known_custody(tmp_path, task, "pending")
    receipt = subagent_bootstrap.bootstrap_before_context(ctx, task, resolve_dispatch_axes(task))
    assert receipt and ctx._configured_actor_bootstrap["exact_start_pending"] is False
    assert ctx._configured_actor_bootstrap["physical_started"] is False
    # The restored child of an owner task carries the owner work kind its root's admission minted;
    # an unmarked context is consciousness work, which the Supervisor holds when no checker can run.
    ctx.task_metadata = {**(ctx.task_metadata or {}), "work_kind": "owner"}
    return registry, task


def _receipt_args(decision="incomplete"):
    return {"contract_kind": "delegation_zero_run", "zero_run_decision": decision,
            "zero_run_basis": "The retained invocation received a definite no-start refusal."}


@pytest.mark.parametrize("carrier", ["working", "exact_pause", "owner_wait"])
@pytest.mark.parametrize("decision", ["incomplete", "unknown"])
def test_restored_readonly_definite_unrun_can_record_only_its_receipt(
        tmp_path, monkeypatch, carrier, decision, stub_delegate_engines):
    from ouroboros import delegate_custody as custody
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tool_policy import initial_tool_schemas

    registry, task = _restored_actor(tmp_path, monkeypatch, carrier)
    ctx = registry._ctx
    assert custody.emit(tmp_path, custody.START_FAILED, {
        "task_id": ctx.task_id, "invocation_id": "invoke-one", "definite": True})
    assert not custody.open_runs(tmp_path) and not custody.pending_invocations(tmp_path)
    schemas = {row["function"]["name"]: row["function"] for row in initial_tool_schemas(registry)}
    properties = schemas["verify_and_record"]["parameters"]["properties"]
    assert properties["contract_kind"]["enum"] == ["delegation_zero_run"]
    assert "check" not in properties and "task_id" not in properties
    for args in ({"contract_kind": "explicit_command", "check": ["must-never-execute"]},
                 {"contract_kind": "artifact_observation", "artifact_paths": ["foreign/file"]}):
        result = registry.execute("verify_and_record", args)
        assert "local_readonly_subagent" in result
        assert read_verification_receipts(tmp_path, ctx.task_id) == []
    result = registry.execute("verify_and_record", _receipt_args(decision))
    assert decision.upper() in result and "typed host receipt recorded" in result
    receipts = read_verification_receipts(tmp_path, task["id"])
    assert len(receipts) == 1 and receipts[0]["zero_run_decision"] == decision
    assert receipts[0]["physical_run_started"] is False
    assert read_verification_receipts(tmp_path, "foreign") == []
    assert registry.get_schema_by_name("verify_and_record") is None
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute("verify_and_record", _receipt_args())
    assert not stub_delegate_engines


@pytest.mark.parametrize("blocker", ["pending", "live", "custody_unreadable", "already_receipted"])
def test_restored_readonly_receipt_still_proves_empty_custody(tmp_path, monkeypatch, blocker):
    from ouroboros import delegate_custody as custody
    from ouroboros.outcomes import read_verification_receipts

    registry, task = _restored_actor(tmp_path, monkeypatch, "working")
    if blocker == "live":
        assert custody.emit(tmp_path, custody.STARTED, {
            "task_id": task["id"], "invocation_id": "invoke-one", "run_id": "late-live", "route": "codex"})
    elif blocker in {"custody_unreadable", "already_receipted"}:
        assert custody.emit(tmp_path, custody.START_FAILED, {
            "task_id": task["id"], "invocation_id": "invoke-one", "definite": True})
        if blocker == "custody_unreadable":
            monkeypatch.setattr(custody, "custody_log_unreadable", lambda *_: True)
        else:
            assert "typed host receipt recorded" in registry.execute("verify_and_record", _receipt_args())
            # Simulate a stale schema/marker; the handler must still read durable proof.
            registry._ctx._configured_actor_bootstrap["zero_run_receipt_recorded"] = False
    before = read_verification_receipts(tmp_path, task["id"])
    assert registry.get_schema_by_name("verify_and_record") is not None
    result = registry.execute("verify_and_record", _receipt_args())
    expected = {"pending": "zero_run_requires_settlement", "live": "zero_run_requires_settlement",
                "custody_unreadable": "zero_run_custody_unknown", "already_receipted": "already"}[blocker]
    assert expected in result
    assert read_verification_receipts(tmp_path, task["id"]) == before


@pytest.mark.parametrize("context", ["unconfigured", "foreign_task", "physical_started"])
def test_readonly_receipt_requires_this_actors_host_bootstrap(tmp_path, monkeypatch, context):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.registry import ToolContext

    registry, task = _restored_actor(tmp_path, monkeypatch, "working")
    if context == "physical_started":
        registry._ctx._configured_actor_bootstrap["physical_started"] = True
    else:
        # Task metadata, including another actor's configured snapshot, is not
        # the private marker that authorizes a typed receipt for this context.
        registry.set_context(ToolContext(
            repo_dir=tmp_path, drive_root=tmp_path,
            task_id="foreign" if context == "foreign_task" else task["id"],
            task_metadata=task if context == "foreign_task" else {},
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False)))
    assert registry.get_schema_by_name("verify_and_record") is None
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute("verify_and_record", _receipt_args())
    assert read_verification_receipts(tmp_path, task["id"]) == []
    assert read_verification_receipts(tmp_path, "foreign") == []
