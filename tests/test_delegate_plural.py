"""Independent invocations under one focus preserve the per-invocation floors."""
from types import SimpleNamespace
import json

import pytest

from ouroboros import delegate_custody as custody, subagent_runtime as runtime
from ouroboros.delegate_shared import delegate_payload, delegate_result
from ouroboros.delegate_start_claims import claimed_start_request
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_builtin_refusal_results import _delegate_registry
from tests._shared import full_disk_at
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
    monkeypatch.setattr("ouroboros.settings_integrity.runtime_catalog_settings", lambda *, settings_reader=None: settings)
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


@pytest.mark.parametrize("cause", ["full_disk", "contention"])
def test_an_unavailable_actor_lock_keeps_its_platform_cause(tmp_path, monkeypatch, cause):
    """The lock owner keeps the platform's outcome; contention claims no OS cause (#2647)."""
    import errno
    import os

    from ouroboros import platform_layer

    if cause == "full_disk":
        full_disk_at(monkeypatch, "delegate_actor_claims")
    else:
        real = platform_layer.acquire_exclusive_file_lock

        def held(path, **kwargs):  # what the platform reports when the holder outlives the bound
            if "delegate_actor_claims" not in str(path):
                return real(path, **kwargs)
            kwargs["outcome"].update(reason="contention", errno=None)
            return None
        monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", held)
    with pytest.raises(custody.ActorDecisionLockUnavailable) as caught:
        with custody.actor_decision_lock(tmp_path, "task"):
            pytest.fail("the lock was not taken")
    assert isinstance(caught.value, TimeoutError)  # the callers' existing contract
    assert (caught.value.reason, caught.value.errno) == (
        ("unknown", errno.ENOSPC) if cause == "full_disk" else ("contention", None))
    accepted, refusal = claimed_start_request(tmp_path, task_id="task", invocation_id="new", request={"prompt": "w"})
    assert not accepted and refusal["reason"] == "replacement_custody_unknown"
    assert (os.strerror(errno.ENOSPC) in refusal["detail"]) is (cause == "full_disk")
    assert custody.invocation_record(tmp_path, "new") is None


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


@pytest.mark.parametrize("initial_state, sentence", [
    ("failed", "First coding attempt failed."),
    ("succeeded", "First coding session finished."),
    ("interrupted", "First coding attempt interrupted."),
])
def test_later_success_does_not_hide_the_initial_execution(tmp_path, initial_state, sentence):
    from ouroboros import subagent_bootstrap as bootstrap
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation
    from ouroboros.outcomes import _apply_actor_first_terminal_projection
    from ouroboros.project_dialogue import _completion_verdict

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
    elif initial_state == "interrupted":
        assert fact["reason"] == "initial_execution_display_only"
        assert fact["initial_execution"]["state"] == "interrupted" and fact["initial_execution"]["run_id"] == "initial"
        outcome = {"reason_code": "final_message", "finish_reason": "final_message",
                   "outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"},
                                    "objective": {"status": "pass", "source": "author_acceptance"}}}
        projected = _apply_actor_first_terminal_projection(outcome, {"actor_first_terminal": fact})
        assert (projected["reason_code"], projected["finish_reason"]) == ("final_message", "final_message")
        assert "failure" not in projected and projected["outcome_axes"]["objective"]["status"] == "pass"
        clean = {"status": "completed", "reason_code": "final_message", "outcome_axes": projected["outcome_axes"]}
        assert clean["outcome_axes"]["execution"]["status"] == "ok"
        assert _completion_verdict(clean, {}) == sentence
        assert _completion_verdict(clean, {}, include_review_details=False) == sentence
    else:
        assert fact is None
    assert custody.record_start_requested(tmp_path, task_id="nanny", invocation_id="pending", request={"prompt": "other"})
    fact = bootstrap.actor_first_unresolved_fact(ctx, task_id="nanny", drive_root=tmp_path)
    assert fact["delegated_obligations"]["pending_invocation_ids"] == ["pending"]
    # The producer records the task's own unknown (delegated_run_unsettled) beside the FIRST attempt's typed
    # initial_execution; the card reads that fact alone, whatever later runs or obligations remain.
    record = {"status": "completed", "reason_code": "configured_actor_unknown", "outcome_axes": {"execution": {
        "status": "degraded", "reason_code": "configured_actor_unknown", "actor_first_terminal": fact}}}
    assert _completion_verdict(record, {}) == sentence
    assert record["reason_code"] == "configured_actor_unknown"


def test_an_observed_terminal_failure_without_settlement_still_states_a_known_first_attempt(tmp_path):
    """R6 (1): the first run's engine terminal observation ("failed") sits in its bound custody row while
    its monetary settlement never landed, and a later attempt succeeded. The aggregate axes (started,
    settled, succeeded, failed, failure states) and the open obligation stay exactly as they were — only
    the FIRST attempt's typed state becomes the KNOWN failure it proves, so the wake/card sentence stops
    saying the outcome could not be determined."""
    from ouroboros import subagent_bootstrap as bootstrap
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation
    from ouroboros.delegate_evidence import task_execution_evidence
    from ouroboros.project_dialogue import _completion_verdict

    ctx = SimpleNamespace(task_id="nanny", drive_root=tmp_path, budget_drive_root=tmp_path,
                          _configured_actor_bootstrap={"physical_started": True}, task_metadata={})
    with custody.actor_decision_lock(tmp_path, "nanny"):
        assert not claim_initial_invocation(tmp_path, "nanny", {"invocation_id": "initial"})
    assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id="initial",
                                                              invocation_id="initial"))
    assert custody.emit(tmp_path, custody.TERMINAL_OBSERVED, {
        "task_id": "nanny", "run_id": "initial", "state": "failed",
        "observation_seq": 1, "source_ref": {"sha256": "a" * 64}})
    assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id="later",
                                                              invocation_id="later"))
    assert custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": "later", "state": "succeeded"})

    evidence = task_execution_evidence(tmp_path, "nanny")
    assert evidence["initial_execution"]["state"] == "failed"        # KNOWN (was "unknown")
    assert evidence["initial_execution"]["run_id"] == "initial"
    assert (evidence["delegated_runs_started"], evidence["delegated_runs_settled"]) == (2, 1)
    assert (evidence["delegated_runs_succeeded"], evidence["delegated_runs_failed"]) == (1, 0)
    assert evidence["delegated_run_failure_states"] == []            # the unsettled row never enters the aggregate

    fact = bootstrap.actor_first_unresolved_fact(ctx, task_id="nanny", drive_root=tmp_path)
    assert fact["reason"] == "delegated_run_unsettled"                # the open obligation stands unchanged
    assert fact["initial_execution"]["state"] == "failed"
    record = {"status": "completed", "reason_code": "configured_actor_unknown", "outcome_axes": {"execution": {
        "status": "degraded", "reason_code": "configured_actor_unknown", "actor_first_terminal": fact}}}
    assert _completion_verdict(record, {}) == "First coding attempt failed."
    assert _completion_verdict(record, {}, include_review_details=False) == "First coding attempt failed."


def test_an_unsettled_terminal_success_still_states_no_confirmed_first_attempt(tmp_path):
    """R6 control (the paired synthetic custody case): a terminal observation of ``succeeded`` on the
    bound row whose settlement never landed proves no confirmed success — the display witness keeps
    the prior success qualification, and the aggregate axes stay exactly as they were."""
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation
    from ouroboros.delegate_evidence import task_execution_evidence

    with custody.actor_decision_lock(tmp_path, "nanny"):
        assert not claim_initial_invocation(tmp_path, "nanny", {"invocation_id": "initial"})
    assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id="initial",
                                                              invocation_id="initial"))
    assert custody.emit(tmp_path, custody.TERMINAL_OBSERVED, {
        "task_id": "nanny", "run_id": "initial", "state": "succeeded",
        "observation_seq": 1, "source_ref": {"sha256": "a" * 64}})
    assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id="later",
                                                              invocation_id="later"))
    assert custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": "later", "state": "succeeded"})

    evidence = task_execution_evidence(tmp_path, "nanny")
    assert evidence["initial_execution"]["state"] != "succeeded"    # no confirmed success before settlement
    assert (evidence["delegated_runs_started"], evidence["delegated_runs_settled"]) == (2, 1)
    assert (evidence["delegated_runs_succeeded"], evidence["delegated_runs_failed"]) == (1, 0)


@pytest.mark.parametrize("initial_state, sentence", [
    ("failed", "First coding attempt failed."),
    ("cancelled", "First coding attempt cancelled."),
])
def test_the_card_words_the_initial_attempt_only_even_when_a_later_run_succeeded(tmp_path, initial_state, sentence):
    """The stored fact (not a bare reason code) picks the words; the later success changes neither the code nor the line."""
    from ouroboros import subagent_bootstrap as bootstrap
    from ouroboros.delegate_bootstrap_identity import claim_initial_invocation
    from ouroboros.project_dialogue import _completion_verdict

    ctx = SimpleNamespace(task_id="nanny", drive_root=tmp_path, budget_drive_root=tmp_path,
                          _configured_actor_bootstrap={"physical_started": True}, task_metadata={})
    with custody.actor_decision_lock(tmp_path, "nanny"):
        assert not claim_initial_invocation(tmp_path, "nanny", {"invocation_id": "initial"})
    for name, state in (("initial", initial_state), ("later", "succeeded")):
        assert custody.record_started(tmp_path, custody.RunCustody(task_id="nanny", run_id=name, invocation_id=name))
        assert custody.emit(tmp_path, custody.SETTLED, {"task_id": "nanny", "run_id": name, "state": state})
    fact = bootstrap.actor_first_unresolved_fact(ctx, task_id="nanny", drive_root=tmp_path)
    record = {"status": "completed", "reason_code": "configured_actor_incomplete", "outcome_axes": {"execution": {
        "status": "degraded", "reason_code": "configured_actor_incomplete", "actor_first_terminal": fact}}}
    assert _completion_verdict(record, {}) == sentence
    assert _completion_verdict(record, {}, include_review_details=False) == sentence  # the wake's and the notice's reading
    assert record["reason_code"] == "configured_actor_incomplete"


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
