"""Independent invocations under one focus preserve the per-invocation floors."""
from types import SimpleNamespace
import json

import pytest

from ouroboros import delegate_custody as custody, subagent_runtime as runtime
from ouroboros.delegate_shared import delegate_payload, delegate_result
from ouroboros.delegate_start_claims import claimed_start_request
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_builtin_refusal_results import _delegate_registry
from tests.test_delegate_retry_replay import _Engine, PROMPT


def test_identical_deliberate_starts_coexist_and_one_retry_keeps_one_identity(tmp_path, monkeypatch):
    from ouroboros.tools import delegate

    registry = _delegate_registry(tmp_path, monkeypatch)
    engine = _Engine(tmp_path, monkeypatch, lose=1)
    first = delegate_payload(delegate._delegate_start(registry._ctx, PROMPT, max_seconds=120))
    second = delegate_payload(delegate._delegate_start(registry._ctx, PROMPT, max_seconds=120))
    assert first["pending_invocation_id"] != second["invocation_id"]
    assert second["status"] == "started"
    token = first["pending_invocation_id"]
    resumed = delegate_payload(runtime.exact_start(registry._ctx, "", {"retry_of": token}))
    assert resumed["status"] == "started" and resumed["invocation_id"] == token
    assert engine.keys == [token, second["invocation_id"], token]
    assert engine.bodies[0] == engine.bodies[2]
    repeated = delegate_payload(runtime.exact_start(registry._ctx, "", {"retry_of": token}))
    assert repeated["reason"] == "invocation_already_started" and repeated["run_id"] == resumed["run_id"]
    assert len(engine.keys) == 3 and len(custody.open_runs(tmp_path)) == 2


@pytest.mark.parametrize("mode", ["advisory", "blocking", "cyber"])
def test_a_nanny_can_choose_two_rows_and_the_same_row_again(tmp_path, monkeypatch, mode):
    from ouroboros.tools import delegate
    from tests.test_available_subagents_runtime import _session_row, _settings, _snapshot

    settings = _settings(_session_row(), {**_session_row(), "subagent_id": "other", "name": "Other",
                                        "route": {"kind": "agent_session", "target_id": "codex=other"}})
    monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: settings)
    snapshot = _snapshot(settings, "session-builder")
    ctx = SimpleNamespace(task_id="nanny", drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_metadata={"review_mode": mode}, _configured_actor_bootstrap={
                              "snapshot": snapshot, "selected_subagent_id": "session-builder",
                              "canonical_work_order": "INITIAL", "work_order_fingerprint": "initial"})
    captured = []

    def start(_ctx, prompt, *_args, **_kwargs):
        selected = dict(runtime._EXACT_START_SELECTION.get()["snapshot"])
        captured.append((prompt, selected))
        return delegate_result({"status": "started", "run_id": f"r{len(captured)}"})

    monkeypatch.setattr(delegate, "_delegate_start", start)
    for selected in ("session-builder", "other", "other"):
        result = runtime.delegate_start_entry(ctx, "independent", subagent_id=selected)
        assert delegate_payload(result)["status"] == "started"
    assert [row[0] for row in captured] == ["independent"] * 3
    assert [row[1]["selected_subagent_id"] for row in captured] == ["session-builder", "other", "other"]
    assert ctx._configured_actor_bootstrap["snapshot"] == snapshot
    refusal = delegate_payload(runtime.delegate_start_entry(ctx, "independent", retry_of="prior", subagent_id="other"))
    assert refusal["reason"] == "retry_selector_conflict" and len(captured) == 3


@pytest.mark.parametrize("kind", ["open", "pending", "undisposed"])
def test_other_obligations_do_not_limit_new_start_claims(tmp_path, kind):
    if kind == "pending":
        assert custody.record_start_requested(tmp_path, task_id="task", invocation_id="older", request={"prompt": "old"})
    else:
        row = custody.RunCustody(run_id="older", task_id="task", snapshot_id="capture" if kind == "undisposed" else "",
                                 access="workspace_write", mode="agent", isolation="live")
        assert custody.record_started(tmp_path, row)
        if kind == "undisposed":
            assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "older", "task_id": "task", "state": "succeeded"})
    accepted, refusal = claimed_start_request(tmp_path, actor_ctx=SimpleNamespace(), task_id="task",
                                              invocation_id="new", request={"prompt": "new work"})
    assert accepted and not refusal
    assert custody.invocation_record(tmp_path, "new")["state"] == "pending"


@pytest.mark.parametrize("blocked", [False, True])
def test_initial_claim_keeps_zero_run_race_check(tmp_path, monkeypatch, blocked):
    from ouroboros import subagent_bootstrap

    ctx = SimpleNamespace(_configured_actor_bootstrap={})
    monkeypatch.setattr(subagent_bootstrap, "_durable_zero_run_receipt",
                        lambda *_a, **_k: {"zero_run_decision": "incomplete"} if blocked else None)
    accepted, refusal = claimed_start_request(tmp_path, actor_ctx=ctx, initial_bootstrap=True,
                                              task_id="task", invocation_id="initial", request={"prompt": "assigned"})
    assert accepted is not blocked
    if blocked:
        assert refusal["reason"] == "zero_run_already_recorded"
        assert custody.invocation_record(tmp_path, "initial") is None
    else:
        second, fault = claimed_start_request(tmp_path, actor_ctx=ctx, initial_bootstrap=True,
                                              task_id="task", invocation_id="accidental", request={"prompt": "assigned"})
        assert not second and fault["reason"] == "initial_start_already_claimed"
        assert custody.invocation_record(tmp_path, "accidental") is None


@pytest.mark.parametrize("unreadable", [False, True])
def test_custody_authority_still_gates_a_new_claim(tmp_path, monkeypatch, unreadable):
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: unreadable)
    accepted, refusal = claimed_start_request(tmp_path, task_id="task", invocation_id="new")
    assert accepted is not unreadable
    if unreadable:
        assert refusal["reason"] == "replacement_custody_unknown"


def test_restart_observes_initial_identity_without_another_start(tmp_path, monkeypatch):
    from ouroboros import subagent_bootstrap as bootstrap
    from tests.test_available_subagents_runtime import _session_row, _settings, _snapshot

    task = {"id": "nanny", "configured_subagent": _snapshot(_settings(_session_row()), "session-builder"),
            "task_contract": {"objective": "Initial assignment"}}
    ctx = SimpleNamespace(task_id="nanny", task_metadata=task, drive_root=tmp_path, budget_drive_root=tmp_path)
    dispatch = SimpleNamespace(blocked=False)
    calls = []

    def start(actual_ctx, prompt, spec):
        calls.append((prompt, spec))
        accepted, refusal = claimed_start_request(tmp_path, actor_ctx=actual_ctx, initial_bootstrap=spec["_initial_bootstrap"],
            task_id="nanny", invocation_id="initial", request={"prompt": prompt})
        assert accepted and not refusal
        return delegate_result({"status": "refused", "reason": "lost_response", "pending_invocation_id": "initial"})

    monkeypatch.setattr(runtime, "exact_start", start)
    bootstrap.bootstrap_before_context(ctx, task, dispatch)
    restored = SimpleNamespace(task_id="nanny", task_metadata=task, drive_root=tmp_path, budget_drive_root=tmp_path)
    receipt = json.loads(bootstrap.bootstrap_before_context(restored, task, dispatch))
    assert len(calls) == 1 and "Initial assignment" in calls[0][0]
    assert receipt["startup"]["invocation_id"] == "initial"
    assert receipt["startup"]["physical_start"] == "not_repeated"


@pytest.mark.parametrize("initial_state", ["failed", "succeeded"])
def test_later_success_does_not_hide_the_initial_execution(tmp_path, initial_state):
    from ouroboros import subagent_bootstrap as bootstrap
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation

    ctx = SimpleNamespace(task_id="nanny", drive_root=tmp_path, budget_drive_root=tmp_path,
                          _configured_actor_bootstrap={"physical_started": True}, task_metadata={})
    with custody.actor_decision_lock(tmp_path, "nanny"):
        assert not claim_initial_invocation(tmp_path, "nanny", {"invocation_id": "initial"})
    for name, state in (("initial", initial_state), ("later", "succeeded")):
        assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id=name, invocation_id=name))
        assert custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": name, "state": state})
    fact = bootstrap.actor_first_unresolved_fact(ctx, task_id="nanny", drive_root=tmp_path)
    if initial_state == "failed":
        assert fact["reason"] == "initial_execution_unresolved" and fact["initial_execution"]["state"] == "failed"
        assert "Start the exact assigned session now" not in bootstrap.configured_actor_finalization_message(
            ctx, task_id="nanny", fallback_root=tmp_path)
    else:
        assert fact is None
    assert custody.record_start_requested(tmp_path, task_id="nanny", invocation_id="pending", request={"prompt": "other"})
    fact = bootstrap.actor_first_unresolved_fact(ctx, task_id="nanny", drive_root=tmp_path)
    assert fact["delegated_obligations"]["pending_invocation_ids"] == ["pending"]


def test_initial_receipt_observes_nothing_without_a_root_and_names_a_corrupt_marker(tmp_path):
    """No data root: nothing durable to observe (a test double, never a live context). A real
    root without a claim: the first start is still to come. A claimed identity is observed
    without a second start; a corrupt marker is named, never read as «no bootstrap»."""
    from types import SimpleNamespace
    import pytest
    from ouroboros import delegate_custody
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation, initial_receipt

    assert initial_receipt(SimpleNamespace(task_id="mock"), {}) == {}
    ctx = SimpleNamespace(task_id="bootstrap-task", drive_root=tmp_path)
    assert initial_receipt(ctx, {}) == {}
    request = {"invocation_id": "inv-initial", "selected_subagent_id": "sub", "config_fingerprint": "cfg",
               "work_order_fingerprint": "work", "authority_fingerprint": "auth"}
    with delegate_custody.actor_decision_lock(tmp_path, ctx.task_id):
        assert claim_initial_invocation(tmp_path, ctx.task_id, request) == {}
    receipt = initial_receipt(ctx, {})
    assert receipt["invocation_id"] == "inv-initial" and receipt["physical_start"] == "not_repeated"
    (tmp_path / "state" / "delegate_bootstrap" / "bootstrap-task.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        initial_receipt(ctx, {})
