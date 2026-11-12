"""Actual attempt/custody producers keep requested, observed and unknown separate."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.outcomes import collect_trace_refs
from ouroboros.subagent_history import record_last_delegation, record_task_execution, subagent_last_delegation
from tests._store_shared import result_exists
from tests._usage_store_testing import ledger_rows, root, write_journal  # noqa: F401 -- the fixture


TARGET = "claudexor::codex-models=gpt-fixture"


def _task(pin=""):
    return {"id": "history-actor", "type": "task", "configured_subagent": {
        "schema": 1, "selected_subagent_id": "worker", "config_fingerprint": "fixture",
        "route": {"kind": "api_model", "target_id": TARGET, "credential_profile_id": pin},
        "effort": "high", "processing_preference": "standard"}}


PAID = "openrouter::openai/gpt-fixture"  # the roster's spelling; the ledger names the same model as its route sends it


def _paid_task(target=PAID):
    task = _task()
    task["configured_subagent"]["route"]["target_id"] = target
    return task


def _settled(index, *, cost, model="openai/gpt-fixture", task_id="history-actor", source="agent.task", provider="openrouter",
             tokens=900):
    """One physical attempt of a task as the usage ledger keeps it; ``cost=None`` is a route without a price."""
    base = dict(attempt_id=f"first-{index}", kind="attempt", root_task_id="r", task_id=task_id, provider=provider,
                model=model, category="task", source=source, reservation_upper_bound_usd=5.0,
                pricing_known=cost is not None, ts=f"2026-11-12T10:{index:02d}:00Z")
    return [{**base, "state": "reserved"}, {**base, "state": "dispatched"},
            {**base, "state": "settled", "cost_usd": cost, "cost_final": cost is not None, "prompt_tokens": tokens}]


def _call(tmp_path, usage, *, pin="", observed="account-a", broken=False, model=TARGET, provider="claudexor", served="",
          ledger=""):
    route = {"source": "codex-models", "model": "observed-model", "credentialProfileId": observed} if observed else {}

    class Provider:
        def chat(self, **kwargs):
            assert kwargs["model_account_override"] == pin
            if broken:
                error = RuntimeError("insufficient_quota")
                error.route = route
                raise error
            return {"content": "Useful response"}, {"prompt_tokens": 1, "completion_tokens": 1,
                "cost": 0.0, "provider": provider, "resolved_model": served or model,
                "claudexor": {"route": route, "outcome": "succeeded"}, **({"ledger_attempt_ids": [ledger]} if ledger else {})}

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


@pytest.mark.parametrize(("failed", "pin"), [(False, "account-a"), (True, "")])
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
    # Binding the replayed start re-reads this one invocation's addressed row under its decision lock;
    # a history scan stays forbidden.
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


def _rings(data_root):
    """The samples as the projection file keeps them, by row; the receipt every other reader gets carries none."""
    assert "first_call_costs" not in subagent_last_delegation(data_root)
    return json.loads((data_root / "state" / "subagent_last_delegation.json").read_text(encoding="utf-8")).get("first_call_costs")


def _first_calls(data_root):
    return (_rings(data_root) or {}).get("worker", {}).get("calls")


def test_a_paid_rows_first_call_price_is_the_ledgers_first_attempt_on_its_model(root, monkeypatch):  # noqa: F811
    """The sample is the task's FIRST settled attempt on the row's own model: a call another model made first
    (a tool's own model call) and the row's later, larger calls are not it. The roster and the ledger spell the
    model differently, and the measured price reaches the context in the review seats' words."""
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    write_journal(root, [*_settled(0, cost=9.0, model="openai/gpt-other"),  # another model of the task answered a tool
                         *_settled(1, cost=8.0, source="safety_check"),  # a check's own call on the same model
                         *_settled(2, cost=7.0, tokens=0),  # an attempt that came back without input counted
                         *_settled(3, cost=0.0432), *_settled(4, cost=0.9), *_settled(5, cost=7.0, task_id="another-task")])
    ledger_rows(root)  # the server's one-time import; a display read never imports
    usage = {}
    assert _call(root, usage, model=PAID, provider="openrouter")
    record_task_execution(_paid_task(), usage, drive_root=root)
    (sample,) = _first_calls(root)
    assert (sample["task_id"], sample["usd"], _rings(root)["worker"]["of"]) == ("history-actor", 0.0432, [PAID, "standard"])
    shown = _delegation_capability_fact()["subagents_last_executions"][0]
    assert shown["first_call"] == "≈$0.04 per first call (1 measured)" and shown["outcome"] == "succeeded"
    record_task_execution(_paid_task(), usage, drive_root=root)  # the same task observed again is one sample
    assert len(_first_calls(root)) == 1


def test_the_ledgers_own_name_for_the_first_answer_finds_its_attempt(root):  # noqa: F811
    """A route may send a model under a name no spelling rule derives from the row's target. The first answer's
    record carries that name, so its attempt is still found; an attempt under any other name is not the row's."""
    write_journal(root, [*_settled(0, cost=0.07, model="gateway/alias-2026"), *_settled(1, cost=0.5, model="gateway/other")])
    ledger_rows(root)
    usage = {}
    assert _call(root, usage, model=PAID, provider="openrouter", served="gateway/alias-2026")
    record_task_execution(_paid_task(), usage, drive_root=root)
    assert [item["usd"] for item in _first_calls(root)] == [0.07]


def test_an_unpriced_first_call_is_cost_unknown_never_zero(root, monkeypatch):  # noqa: F811
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    target = "openai::gpt-fixture"  # a direct key: the ledger settles its calls without a price
    # An earlier priced attempt of the same task through OpenRouter carries the same model name: another route's
    # money never prices this row.
    write_journal(root, [*_settled(0, cost=9.0), *_settled(1, cost=None, provider="openai")])
    ledger_rows(root)
    usage = {}
    assert _call(root, usage, model=target, provider="openai")
    record_task_execution(_paid_task(target), usage, drive_root=root)
    (sample,) = _first_calls(root)
    assert sample["usd"] is None
    assert _delegation_capability_fact()["subagents_last_executions"][0]["first_call"] == "cost unknown"


@pytest.mark.parametrize("case", ["fallback_answered_first", "subscription_engine", "ledger_unreadable", "no_attempt"])
def test_no_first_call_sample_without_the_rows_own_paid_first_answer(root, monkeypatch, case):  # noqa: F811
    """A fallback's first answer is not the row's price, the subscription engine states no per-call cash, and a
    ledger that cannot be read (or holds no such attempt) is no sample: the execution record is written all the same."""
    from ouroboros import usage_store
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    usage, target, provider = {}, PAID, "openrouter"
    if case == "subscription_engine":  # its attempt is in the ledger under the row's own name, with cash: only the route keeps it out
        target, provider = TARGET, "claudexor"
    write_journal(root, _settled(0, cost=0.05, model=TARGET if case == "subscription_engine" else "openai/gpt-fixture",
                                 provider=provider, task_id="another-task" if case == "no_attempt" else "history-actor"))
    ledger_rows(root)
    if case == "fallback_answered_first":
        assert _call(root, usage, model="openrouter::openai/gpt-other", provider="openrouter")
    if case == "ledger_unreadable":
        def unreadable(*args, **kwargs):
            raise RuntimeError("fixture: the usage store is busy")
        monkeypatch.setattr(usage_store, "read", unreadable)
    assert _call(root, usage, model=target, provider=provider)
    record_task_execution(_paid_task(target), usage, drive_root=root)
    assert subagent_last_delegation(root)["outcome"] == "succeeded" and _first_calls(root) is None
    assert "first_call" not in _delegation_capability_fact()["subagents_last_executions"][0]


def test_first_call_samples_are_bounded_per_row_and_follow_its_price_class(tmp_path, monkeypatch):
    """The projection keeps a row's newest 20 samples of its recorded price class, so its size has a bound
    (26 rows x 20 samples). A late observation of an older run neither erases nor displaces them. A row moved
    to another model or processing tier starts over: the other tier's price per token is not this one's.
    A row that leaves the projection takes its samples."""
    from ouroboros.configured_subagents import MAX_CONFIGURED_SUBAGENTS
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    from ouroboros.review_price import RECENT_REVIEWS
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)

    def run(index, *, actor="worker", target=PAID, processing="standard", usd=None, outcome="succeeded", measured=True, task=None):
        record_last_delegation(route="api_model", requested_model=target, applied_model="", run_id=f"run-{actor}-{index}",
            selected_subagent_id=actor, drive_root=tmp_path, occurred_at=f"2026-11-12T10:{index:02d}:00+00:00",
            outcome=outcome, task_id=f"task-{actor}-{task or index}",
            first_call={"usd": index / 10 if usd is None else usd} if measured else None,
            identity={"kind": "api_model", "target_id": target, "processing_preference": processing})

    def shown():
        return _delegation_capability_fact()["subagents_last_executions"][0]["first_call"]

    def kept():
        return [item["usd"] for item in _first_calls(tmp_path)]

    for index in range(4, 29):
        run(index)
    newest = [index / 10 for index in range(9, 29)]
    assert len(kept()) == RECENT_REVIEWS == 20 and kept() == newest
    assert shown() == "≈$1.85 per first call (median of 20 measured, $0.90–$2.80)"
    run(1, usd=0.01)  # an older run observed late is older than every kept sample: it displaces none
    run(2, processing="priority", usd=9.0)  # ...and an older run on another tier is not this row's price
    run(29, task=28, usd=7.0)  # ...and a task already measured, continued as a new run, keeps its first sample
    assert kept() == newest and shown() == "≈$1.85 per first call (median of 20 measured, $0.90–$2.80)"
    run(30, processing="priority", usd=4.0)  # the same model on another processing tier is another price
    assert kept() == [4.0] and shown() == "≈$4.00 per first call (1 measured)"
    run(31, target="openrouter::openai/gpt-next", usd=0.5)  # the row now names another model: its price starts over
    assert kept() == [0.5] and shown() == "≈$0.50 per first call (1 measured)"
    run(32, target="openrouter::openai/gpt-next", outcome="failed", measured=False)
    assert kept() == [0.5] and shown() == "≈$0.50 per first call (1 measured)", "a later record without a sample keeps the measured ones"
    for index in range(MAX_CONFIGURED_SUBAGENTS):  # newer rows fill the projection; the oldest row and its samples leave
        run(33 + index % 20, actor=f"newer-{index}")
    assert "worker" not in subagent_last_delegation(tmp_path)["latest_by_subagent"]
    assert "worker" not in _rings(tmp_path) and len(_rings(tmp_path)) == MAX_CONFIGURED_SUBAGENTS


def test_a_damaged_sample_never_takes_the_history_out_of_the_context(tmp_path, monkeypatch):
    """The samples sit in a file a person or a crash can damage. A value that is no price is not counted, a ring
    that cannot be read prices nothing, and the rows' history is shown either way."""
    from ouroboros.context_runtime_facts import _delegation_capability_fact
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    for index, actor in enumerate(("worker", "broken")):
        record_last_delegation(route="api_model", requested_model=PAID, applied_model="", run_id=f"run-{actor}",
            selected_subagent_id=actor, drive_root=tmp_path, occurred_at=f"2026-11-12T10:0{index}:00+00:00",
            outcome="succeeded", task_id=f"task-{actor}", first_call={"usd": 0.5},
            identity={"kind": "api_model", "target_id": PAID, "processing_preference": "standard"})
    path = tmp_path / "state" / "subagent_last_delegation.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["first_call_costs"]["worker"]["calls"] += [{"task_id": name, "ts": "2026-11-12T10:00:00+00:00", "usd": usd}
                                                       for name, usd in (("inf", float("inf")), ("nan", float("nan")),
                                                                         ("text", "0.5"), ("negative", -1), ("huge", 10 ** 400))]
    stored["first_call_costs"]["broken"] = {"of": [PAID, "standard"], "calls": 5}
    path.write_text(json.dumps(stored), encoding="utf-8")
    shown = {row["task_id"]: row.get("first_call") for row in _delegation_capability_fact()["subagents_last_executions"]}
    assert shown == {"task-worker": None, "task-broken": None}, "a number too large to be a price hides its row's price, not the rows"
    stored["first_call_costs"]["worker"]["calls"].pop()
    path.write_text(json.dumps(stored), encoding="utf-8")
    shown = {row["task_id"]: row.get("first_call") for row in _delegation_capability_fact()["subagents_last_executions"]}
    assert shown == {"task-worker": "≈$0.50 per first call (1 measured)", "task-broken": None}


def test_a_helpers_measured_first_call_price_changes_no_cached_block(tmp_path, monkeypatch):
    """A helper's first-call price moves whenever one of its tasks ends. It rides in the changing part
    (``subagents_last_executions`` of the runtime facts), so the cache-marked blocks and the leading system
    items keep their bytes on both send paths; in a cached block every helper's task end would re-send it."""
    from ouroboros import context
    from ouroboros.llm_messages import split_leading_system_prefix
    from tests.test_doc_context import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", env.drive_root)
    sends = []
    for index, first_call in enumerate((None, {"usd": 0.9})):  # the row's second task brings its first measured price
        record_last_delegation(route="api_model", requested_model=PAID, applied_model="", run_id=f"run-{index}",
            selected_subagent_id="worker", drive_root=env.drive_root, occurred_at=f"2026-11-12T10:0{index}:00+00:00",
            outcome="succeeded", task_id=f"task-{index}", first_call=first_call,
            identity={"kind": "api_model", "target_id": PAID, "processing_preference": "standard"})
        messages, _ = context.build_llm_messages(env=env, memory=memory, task={"id": "root", "type": "task", "text": "w"})
        marked = [block["text"] for block in messages[0]["content"] if block.get("cache_control")]
        wire, moved = split_leading_system_prefix(messages)
        assert moved > 0
        sends.append((marked, [row for row in wire if row["role"] == "system"],
                      json.dumps([row for row in wire if row["role"] != "system"], ensure_ascii=False)))
    (marked_before, system_before, tail_before), (marked_after, system_after, tail_after) = sends
    assert len(marked_before) >= 2 and marked_before == marked_after, "the cache-marked blocks keep their bytes"
    assert system_before == system_after, "the leading system items keep their bytes"
    assert "per first call" not in tail_before and "≈$0.90 per first call (1 measured)" in tail_after
    assert "per first call" not in json.dumps([marked_after, system_after], ensure_ascii=False)


def test_the_first_answer_is_priced_by_the_ledger_attempt_that_carried_it(root, monkeypatch):  # noqa: F811
    """The answer's own ledger attempt prices it: an earlier billed attempt of the same route (an empty glitch
    retried, a tool's call on the same model) is not its price, and neither is a later round."""
    write_journal(root, [*_settled(0, cost=0.002), *_settled(1, cost=0.18), *_settled(2, cost=0.9)])
    ledger_rows(root)
    usage = {}
    assert _call(root, usage, model=PAID, provider="openrouter", ledger="first-1")
    assert usage["llm_call_refs"][0]["physical_attempt_id"] == "first-1"
    record_task_execution(_paid_task(), usage, drive_root=root)
    assert [item["usd"] for item in _first_calls(root)] == [0.18]
