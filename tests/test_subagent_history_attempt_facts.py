"""Actual attempt/custody producers keep requested, observed and unknown separate."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.outcomes import collect_trace_refs
from ouroboros.subagent_history import record_task_execution, subagent_last_delegation
from tests._store_shared import result_exists


TARGET = "claudexor::codex-models=gpt-fixture"


def _task(pin=""):
    return {"id": "history-actor", "type": "task", "configured_subagent": {
        "schema": 1, "selected_subagent_id": "worker", "config_fingerprint": "fixture",
        "route": {"kind": "api_model", "target_id": TARGET, "credential_profile_id": pin},
        "effort": "high", "processing_preference": "standard"}}


def _call(tmp_path, usage, *, pin="", observed="account-a", broken=False, model=TARGET, provider="claudexor"):
    route = {"source": "codex-models", "model": "observed-model", "credentialProfileId": observed} if observed else {}

    class Provider:
        def chat(self, **kwargs):
            assert kwargs["model_account_override"] == pin
            if broken:
                error = RuntimeError("insufficient_quota")
                error.route = route
                raise error
            return {"content": "Useful response"}, {"prompt_tokens": 1, "completion_tokens": 1,
                "cost": 0.0, "provider": provider, "resolved_model": model,
                "claudexor": {"route": route, "outcome": "succeeded"}}

    message, _ = call_llm_with_retry(Provider(), [{"role": "user", "content": "work"}], model,
        None, "high", 0, tmp_path / "logs", "history-actor", 1, None, usage, model_account_override=pin)
    if message:
        usage["_last_llm_call_meta"]["usable_solve_response"] = True
    route.update(model="mutated-later", credentialProfileId="unrelated")
    return message


@pytest.mark.parametrize("pin", ["", "account-a"])
def test_actual_auto_and_pinned_attempt_reach_result_and_context(tmp_path, monkeypatch, pin):
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    usage = {}
    assert _call(tmp_path, usage, pin=pin)
    _store_task_result(SimpleNamespace(drive_root=tmp_path), _task(pin), "Done", usage, {"tool_calls": []})
    row = subagent_last_delegation(tmp_path)
    assert row["requested_profile"] == pin
    assert row["applied_profile"] == "account-a"
    assert row["applied_model"] == "observed-model"
    assert row["observed_route"]["model"] == "observed-model"
    assert row["attempt_id"] == usage["llm_call_refs"][0]["llm_call_id"]
    # The context keeps the recorded actor and execution identities; the exact
    # reader holds the detailed account/model evidence without copying it inline.
    handle = f"{TARGET}/high" + (f"/@{pin}" if pin else "")
    fact = _delegation_capability_fact()
    shown = fact["subagents_last_executions"][0]
    assert shown["selected_subagent_id"] == handle and shown["task_id"] == "history-actor"
    assert shown["attempt_id"] == row["attempt_id"] and shown["outcome"] == row["outcome"]
    assert "applied_profile" not in shown and "observed_route" not in shown
    reader = next(item for item in fact["readers"] if item["arguments"]["path"].endswith("subagent_last_delegation.json"))
    assert json.loads((tmp_path / reader["arguments"]["path"]).read_text(encoding="utf-8")) == row


@pytest.mark.parametrize("provider", ["claudexor", "openai"])
def test_success_with_no_reported_route_does_not_invent_account_or_model(tmp_path, provider):
    usage = {}
    _call(tmp_path, usage, pin="account-a", observed="", provider=provider)
    record_task_execution(_task("account-a"), usage, drive_root=tmp_path)
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "succeeded"
    assert row["requested_profile"] == "account-a"
    assert row["applied_profile"] == row["applied_model"] == ""


@pytest.mark.parametrize("fallback_model", [TARGET, "claudexor::codex-models=other"])
def test_sibling_account_or_model_fallback_does_not_certify_selected_failure(tmp_path, fallback_model):
    usage = {}
    assert _call(tmp_path, usage, pin="account-a", broken=True) is None
    failed_id = usage["llm_call_refs"][0]["llm_call_id"]
    assert _call(tmp_path, usage, pin="account-b", observed="account-b", model=fallback_model)
    record_task_execution(_task("account-a"), usage, drive_root=tmp_path)
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "failed" and row["attempt_id"] == failed_id
    assert row["requested_profile"] == row["applied_profile"] == "account-a"
    assert row["applied_model"] == ""
    assert row["observed_route"]["model"] == "observed-model"
    assert row["fallback"]["applied_profile"] == "account-b"
    assert _call(tmp_path, usage, pin="account-a")
    record_task_execution(_task("account-a"), usage, drive_root=tmp_path)
    assert subagent_last_delegation(tmp_path)["outcome"] == "succeeded"


def test_failed_attempt_history_survives_successful_served_trace_projection(tmp_path):
    from devtools.benchmarks.cybergym.cybergym_wire import _served_telemetry, ExecutorFailure
    usage = {}
    _call(tmp_path, usage, broken=True)
    _call(tmp_path, usage)
    refs = collect_trace_refs(usage, {})
    assert len(usage["llm_call_refs"]) == 2
    assert usage["llm_call_refs"][0]["failure_code"] == "quota_exhausted"
    assert len(refs["llm_call_refs"]) == 1
    assert _served_telemetry({"trace_refs": refs})["observed_provider"] == "claudexor"
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("type") == "llm_api_error" for row in events)
    broken = {**refs["llm_call_refs"][0], "provider": None}
    with pytest.raises(ExecutorFailure, match="incomplete served-call identity"):
        _served_telemetry({"trace_refs": {"llm_call_refs": [broken]}})
    mixed = {**refs["llm_call_refs"][0], "resolved_model": "different-model"}
    with pytest.raises(ExecutorFailure, match="mixed served models"):
        _served_telemetry({"trace_refs": {"llm_call_refs": [refs["llm_call_refs"][0], mixed]}})


def _no_history_scan(*args, **kwargs):
    pytest.fail("history projection must not scan invocation archives")


def test_started_options_survive_replay_and_known_empty_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(custody, "_CUSTODY", {})
    entry = custody.RunCustody(run_id="run", selected_subagent_id="worker", task_id="task",
        route_id="codex", model="fixture", profile_id="account-a", effort="", processing_preference="")
    assert custody.record_started(tmp_path, entry)
    assert custody.record_started(tmp_path, replace(entry, effort="high", processing_preference="fast"))
    monkeypatch.setattr(custody, "_CUSTODY", {})
    replayed = custody.replay(tmp_path)["run"]
    assert replayed.effort == replayed.processing_preference == ""
    monkeypatch.setattr(custody, "_iter_rows", _no_history_scan)
    assert custody.settle_run(tmp_path, None, replayed, {"summary": {
        "state": "succeeded", "spendUsd": 0, "finishedAt": "2099-01-01T00:00:00Z"}})["settled"]
    identity = subagent_last_delegation(tmp_path)["identity"]
    assert identity["effort"] == identity["processing_preference"] == ""


@pytest.mark.parametrize("selected", ["worker", ""])
def test_legacy_missing_options_remain_unknown_without_terminal_archive_read(tmp_path, monkeypatch, selected):
    monkeypatch.setattr(custody, "_CUSTODY", {})
    entry = custody.RunCustody(run_id="legacy", route_id="codex", selected_subagent_id=selected)
    custody.record_started(tmp_path, entry, shape={"effort": "high"})
    replayed = custody.replay(tmp_path)["legacy"]
    assert replayed.effort == "high" and replayed.processing_preference is None
    monkeypatch.setattr(custody, "_iter_rows", _no_history_scan)
    custody.settle_run(tmp_path, None, replayed, {"summary": {"state": "failed", "spendUsd": 0}})
    row = subagent_last_delegation(tmp_path)
    assert row["identity"]["effort"] == "high" and "processing_preference" not in row["identity"]


def test_actual_start_failure_uses_captured_source_without_lookup(tmp_path, monkeypatch):
    from ouroboros.subagent_history import session_request_facts
    from ouroboros.tools.delegate import _retire_orphaned_registration
    from ouroboros.tools.registry import ToolContext
    from ouroboros.store_import import ensure_current_store
    facts = session_request_facts({"model": "fixture", "credentialProfileId": "original", "effort": "", "access": "full"},
        selected_subagent_id="worker", task_id="task", route="codex", processing={"requested": "economy"})
    ensure_current_store(tmp_path)  # the one chain fold the first-touch import owns, before the probes
    monkeypatch.setattr(custody, "invocation_record", _no_history_scan)
    monkeypatch.setattr(custody, "_iter_rows", _no_history_scan)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task")
    _retire_orphaned_registration(ctx, None, "", definite_refusal=True, reason="auth_required",
                                  invocation_id="inv", history_facts=facts)
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "not_started" and row["requested_profile"] == "original"
    assert row["identity"]["effort"] == "" and row["identity"]["processing_preference"] == "economy"


def test_history_error_does_not_claim_successful_result_store_failed(tmp_path, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise OSError("fixture history failure")
    monkeypatch.setattr("ouroboros.subagent_history.record_task_execution", fail)
    _store_task_result(SimpleNamespace(drive_root=tmp_path), _task(), "Done", {}, {"tool_calls": []})
    assert result_exists(tmp_path, "history-actor")
    assert "Task result stored; subagent history unavailable" in caplog.text
    assert "Failed to store task result" not in caplog.text


@pytest.mark.parametrize("pin", ["", "account-a"])
@pytest.mark.parametrize("failed", [False, True])
def test_transport_request_account_survives_when_caller_uses_role_default(tmp_path, pin, failed):
    from ouroboros.llm_claudexor import _ModelInvocation
    invocation = _ModelInvocation({"usage_model": TARGET},
        {"account": {"mode": "pin", "profileId": pin} if pin else {"mode": "auto"}}, {"timeout": 1})
    result = {"outcome": "completed", "message": {"content": "Done"},
              "route": {"source": "codex-models", "model": "served-model", "credentialProfileId": "account-a"}}
    if failed:
        result.update(outcome="failed", problem={"code": "quota_exhausted", "message": "fixture"})
    provider = SimpleNamespace(chat=lambda **kwargs: invocation.finish(result))
    usage = {}
    message, _ = call_llm_with_retry(provider, [{"role": "user", "content": "work"}], TARGET, None, "high", 0,
                                    tmp_path / "logs", "history-actor", 1, None, usage)
    assert bool(message) is not failed
    if message:
        usage["_last_llm_call_meta"]["usable_solve_response"] = True
    assert usage["llm_call_refs"][0]["requested_profile"] == pin
    record_task_execution(_task(pin), usage, drive_root=tmp_path)
    row = subagent_last_delegation(tmp_path)
    assert row["applied_profile"] == "account-a"
    assert row["outcome"] == ("failed" if failed else "succeeded")


@pytest.mark.parametrize("refused", [False, True])
def test_pending_recovery_reuses_held_original_request_without_history_scan(tmp_path, monkeypatch, refused):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    record = {"invocation_id": "inv-recover", "task_id": "task", "route": "codex", "project_id": "",
        "project_owned": False, "idempotency_key": "key", "selected_subagent_id": "worker",
        "request": {"model": "original-model", "credentialProfileId": "original-account", "effort": "",
                    "access": "readonly"}, "processing": {"requested": "economy", "submitted": None}}
    monkeypatch.setattr(custody, "_CUSTODY", {})
    custody.record_start_requested(tmp_path, **record)
    monkeypatch.setattr(custody, "invocation_record", _no_history_scan)
    monkeypatch.setattr(custody, "_iter_rows", _no_history_scan)

    class Gateway:
        def start_run(self, body, *, idempotency_key):
            assert body == record["request"] and idempotency_key == "inv-recover"
            if refused:
                raise ClaudexorUnavailable("auth_required", "fixture refusal", status_code=403)
            return {"runId": "recovered"}

        def get_run(self, run_id):
            return {"summary": {"state": "succeeded", "spendUsd": 0, "finishedAt": "2099-01-01T00:00:00Z"}}

    custody._recover_pending_invocation(tmp_path, Gateway(), record)
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == ("not_started" if refused else "succeeded")
    assert row["requested_profile"] == "original-account"
    assert row["identity"]["effort"] == "" and row["identity"]["processing_preference"] == "economy"
    assert row["applied_profile"] == ""


@pytest.mark.parametrize("capture_available", [True, False])
def test_received_empty_response_keeps_served_identity(tmp_path, monkeypatch, capture_available):
    from devtools.benchmarks.cybergym.cybergym_wire import _served_telemetry
    from ouroboros import loop_llm_call
    from tests.test_loop_compaction import _ctx
    from tests.test_task_model_execution import dispatch, Model

    if not capture_available:
        monkeypatch.setattr(loop_llm_call, "persist_observed_call", lambda *a, **k: {})
    ctx = _ctx(tmp_path)
    dispatch(ctx, "stable")
    ctx.round_idx += 1
    assert dispatch(ctx, "stable", message={"role": "assistant", "content": "", "tool_calls": []})[0] is None
    loop_llm_call.call_llm_with_retry(
        Model({"role": "assistant", "content": "wrap up"}, {}), ctx.messages,
        "stable", [], "high", 1, ctx.drive_logs, ctx.task_id, 3, None,
        ctx.accumulated_usage, attempt_cap=1)
    raw = ctx.accumulated_usage["llm_call_refs"]
    assert len(raw) == 3 and raw[1]["failure_code"] == "provider_incomplete_response"
    refs = collect_trace_refs(ctx.accumulated_usage, {})
    assert [row["llm_call_id"] for row in refs["llm_call_refs"]] == [row["llm_call_id"] for row in raw]
    assert bool(refs["llm_call_refs"][1]["response_ref"]) is capture_available
    telemetry = _served_telemetry({"trace_refs": refs})
    assert telemetry["authoritative_identity"] and telemetry["trace_call_count"] == 3
    assert telemetry["provider_distribution"] == {"openrouter": 3}
    assert raw[1]["failure_code"] == "provider_incomplete_response"
