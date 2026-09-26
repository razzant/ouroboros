"""Real terminal producers feed dated disclosure without becoming admission policy."""

import json
from types import SimpleNamespace

from ouroboros import delegate_custody as custody
from ouroboros.context_runtime_facts import _delegation_capability_fact
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.subagent_history import record_task_execution, record_last_delegation, subagent_last_delegation, session_request_facts
from ouroboros.subagent_runtime import resolve_configured_actor_dispatch


def _task(kind="api_model", target="openai::fixture"):
    return {"id": "task-history", "type": "task", "configured_subagent": {
        "schema": 1, "config_fingerprint": "irrelevant-list-hash", "selected_subagent_id": "worker",
        "route": {"kind": kind, "target_id": target, "credential_profile_id": ""},
        "effort": "high", "processing_preference": "standard"}}


def test_api_failure_fallback_and_retry_reach_next_task_context(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    task = _task()
    usage = {}

    class Provider:
        broken = True

        def chat(self, **_kwargs):
            if self.broken:
                raise RuntimeError("insufficient_quota")
            return {"content": "Useful reply"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}

    provider = Provider()
    args = (provider, [{"role": "user", "content": "work"}], "openai::fixture", None, "high", 0,
            tmp_path / "logs", task["id"], 1, None, usage)
    assert call_llm_with_retry(*args)[0] is None
    failure = usage["llm_call_refs"][-1]
    assert failure["failure_code"] == "quota_exhausted"
    # Another model's usable response does not erase this route's own incident.
    usage["llm_call_refs"].append({"model": "openai::fallback", "usable_solve_response": True})
    _store_task_result(SimpleNamespace(drive_root=tmp_path), task, "Fallback answered", usage, {"tool_calls": []})
    assert (tmp_path / "task_results" / "task-history.json").is_file()
    row = _delegation_capability_fact()["subagents_last_executions"][0]
    assert row["outcome"] == "failed" and row["occurred_at"] == failure["ts"]
    assert row["task_id"] == task["id"] and row["fallback"]["model"] == "openai::fallback"
    assert row["applied_model"] == ""
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda *_: True)
    assert resolve_configured_actor_dispatch(task, task_type="task").executor == "native"
    provider.broken = False
    assert call_llm_with_retry(*args)[0]["content"] == "Useful reply"
    usage["_last_llm_call_meta"]["usable_solve_response"] = True
    usage["execution_status"] = "failed"  # a later code/test failure is not a provider failure
    record_task_execution(task, usage, drive_root=tmp_path)
    row = _delegation_capability_fact()["subagents_last_executions"][0]
    assert row["outcome"] == "succeeded" and "failure_code" not in row
    before = subagent_last_delegation(tmp_path)
    record_task_execution(task, usage, drive_root=tmp_path)
    assert subagent_last_delegation(tmp_path) == before


def test_recovery_settlement_and_refused_start_share_history(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    request = {"model": "fixture", "effort": "high", "credentialProfileId": "account-a", "access": "full"}
    custody.record_start_requested(tmp_path, invocation_id="invoke", selected_subagent_id="worker",
                                  route="codex", request=request, task_id="task-history")
    facts = session_request_facts(request, selected_subagent_id="worker", task_id="task-history",
                                  route="codex", processing={"requested": "standard"})
    custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "invoke", "definite": False,
                                                "reason": "transport_unavailable", **facts})
    uncertain = subagent_last_delegation(tmp_path)
    assert uncertain["outcome"] == "unknown"
    custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "invoke", "definite": True,
                                                "reason": "quota_exhausted", **facts})
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "not_started" and row["requested_profile"] == "account-a"
    assert row["identity"]["access"] == "full"
    assert row["observed_at"] == uncertain["observed_at"]
    entry = custody.RunCustody(run_id="run-new", task_id="task-history", route_id="codex", model="fixture",
                               selected_subagent_id="worker", profile_id="account-a")
    custody.record_started(tmp_path, entry)
    detail = {"summary": {"state": "succeeded", "spendUsd": 0,
                          "finishedAt": "2099-09-18T12:00:00Z"}}
    assert custody.settle_run(tmp_path, SimpleNamespace(), entry, detail)["settled"]
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "succeeded" and row["applied_model"] == ""
    assert row["occurred_at"] == "2099-09-18T12:00:00Z"
    assert custody.settle_run(tmp_path, SimpleNamespace(), entry, detail)["settled"]
    assert subagent_last_delegation(tmp_path) == row
    old = custody.RunCustody(run_id="old", task_id="old-task", route_id="codex", model="fixture",
                             selected_subagent_id="worker")
    custody.record_started(tmp_path, old)
    detail = {"summary": {"state": "failed", "spendUsd": 0,
                          "finishedAt": "2001-09-18T12:00:00Z", "failure": {"code": "quota_exhausted"}}}
    custody.settle_run(tmp_path, SimpleNamespace(), old, detail)
    assert subagent_last_delegation(tmp_path) == row


def test_first_post_upgrade_write_preserves_legacy_subagent_row(tmp_path):
    path = tmp_path / "state" / "subagent_last_delegation.json"
    path.parent.mkdir()
    legacy = {
        "ts": "2026-09-17T12:00:00Z",
        "observed_at": "2026-09-17T12:00:00Z",
        "occurred_at": "2026-09-17T12:00:00Z",
        "route": "api_model",
        "requested_model": "openai::legacy",
        "applied_model": "openai::legacy",
        "requested_profile": "",
        "applied_profile": "",
        "selected_subagent_id": "legacy-worker",
        "run_id": "legacy-run",
        "outcome": "succeeded",
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    record_last_delegation(
        route="api_model",
        requested_model="openai::new",
        applied_model="openai::new",
        run_id="new-run",
        selected_subagent_id="new-worker",
        drive_root=tmp_path,
        occurred_at="2026-09-20T12:00:00Z",
        outcome="succeeded",
    )

    row = subagent_last_delegation(tmp_path)
    assert row["selected_subagent_id"] == "new-worker"
    assert row["latest_by_subagent"]["new-worker"]["run_id"] == "new-run"
    assert row["latest_by_subagent"]["legacy-worker"]["run_id"] == "legacy-run"
    assert "latest_by_subagent" not in row["latest_by_subagent"]["legacy-worker"]


def test_old_corrupt_missing_and_unknown_time_are_not_health(tmp_path):
    assert subagent_last_delegation(tmp_path) == {}
    path = tmp_path / "state" / "subagent_last_delegation.json"
    path.parent.mkdir()
    path.write_text('{"run_id":"legacy","ts":"2001-01-01T00:00:00Z"}', encoding="utf-8")
    assert subagent_last_delegation(tmp_path)["run_id"] == "legacy"
    path.write_text("corrupt", encoding="utf-8")
    record_last_delegation(route="codex", requested_model="fixture", applied_model="", run_id="unknown",
                           selected_subagent_id="worker", drive_root=tmp_path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["outcome"] == "unknown" and row["occurred_at"] == ""
    assert row["observed_at"] and row["latest_by_subagent"]["worker"]["occurred_at"] == ""


def test_pre_invocation_session_refusal_is_dated_at_its_existing_producer(tmp_path, monkeypatch):
    from ouroboros.subagent_bootstrap import _record_startup_refusal
    monkeypatch.setattr("ouroboros.subagent_runtime.current_subagent_alternatives", lambda *_: [])
    task = _task("agent_session", "codex=fixture")
    task["configured_subagent"]["access"] = "workspace_write"
    ctx = SimpleNamespace()
    _record_startup_refusal(ctx, task, reason="route_disabled")
    _store_task_result(SimpleNamespace(drive_root=tmp_path), task, "Could not start", {}, {"tool_calls": []})
    row = subagent_last_delegation(tmp_path)
    assert row["outcome"] == "not_started" and row["failure_code"] == "route_disabled"
    assert row["occurred_at"] == task["subagent_availability"]["observed_at"]
    assert row["task_id"] == task["id"]
    assert row["identity"]["access"] == "workspace_write"
