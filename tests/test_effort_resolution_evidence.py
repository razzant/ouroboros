"""The shared engine fixture reaches model/session consumers without false echoes."""
import json
from pathlib import Path

import pytest

from ouroboros.effort_evidence import validated_effort_resolution
from ouroboros.gateways.claudexor import final_attempt_facts
from ouroboros.llm_claudexor import _ModelInvocation
from tests._review_session_route_shared import (
    FakeLLM,
    _agent_request,
    _agent_slot,
    _terminal_detail,
)
from tests._review_session_route_shared import (
    _owned_gateway_uses_each_test_transport as _owned_fixture,
)
from tests._review_session_route_shared import (
    fake_route as _route_fixture,
)
from tests.test_claudexor_observed_attempt import _write_telemetry
from tests.test_llm_claudexor import MODEL, ledger
from tests.test_llm_claudexor import result as _model_result
from tests.test_llm_claudexor import setup as _model_fixture
from tests._usage_store_testing import ledger_rows

_owned_gateway_uses_each_test_transport = _owned_fixture
fake_route = _route_fixture
model_setup = _model_fixture

SHARED = json.loads((Path(__file__).parent / "fixtures/claudexor_effort_resolution.json").read_text())
VARIANTS = json.loads((Path(__file__).parent / "fixtures/claudexor_effort_resolution_variants.json").read_text())
# Claudexor owns the ordered proof, including future tokens and native orders
# that differ from the host preference scale. These are engine claims, not echoes.
REPORTS = {"paired_downward": SHARED, **VARIANTS}


def _finish(report, applied=None):
    invocation = _ModelInvocation(
        {"usage_model": "claudexor::codex=model", "requested_reasoning_effort": "ultra"},
        {"options": {"reasoningEffort": "ultra"}}, {})
    return invocation.finish({"outcome": "completed", "message": {"content": "done"},
                              "effortResolution": report, "appliedOptions": applied})[1]


@pytest.mark.parametrize("applied", [None, {}, {"cacheKey": "unrelated"}])
@pytest.mark.parametrize("report", REPORTS.values(), ids=REPORTS)
def test_paired_fixture_is_preparation_evidence_not_dispatch_or_provider_echo(applied, report):
    usage = _finish(report, applied)
    assert usage["effort_resolution"] == report
    assert usage["effort"]["requested"] == "ultra"
    assert usage["effort"]["sent"] == {"options.reasoningEffort": "ultra"}
    assert usage["effort"]["reported"] is None
    assert usage["claudexor"]["applied_options"] == applied


def test_provider_response_echo_remains_observation_when_resolution_has_none():
    usage = _finish(SHARED, {"reasoningEffort": "xhigh"})
    assert usage["effort"]["reported"] == "xhigh"
    assert usage["effort"]["report_source"] == "provider_applied_options"
    assert usage["claudexor"]["applied_options_source"] == "provider_response"
    assert usage["claudexor"]["applied_options"] == {"reasoningEffort": "xhigh"}


def test_observation_reporter_is_separate_from_resolution_authority():
    report = {**SHARED, "observed": "high", "observedSource": "provider_response"}
    usage = _finish(report)
    assert usage["effort"]["reported"] == "high"
    assert usage["effort"]["report_source"] == "provider_response"
    assert usage["effort_resolution"]["source"] == "account_catalog"


@pytest.mark.parametrize("report", [None, {}, [], {"requested": "ultra"},
    {**SHARED, "submitted": None}, {**SHARED, "parameter": None},
    {**SHARED, "requested": None}, {**SHARED, "resolution": "exact"},
    {**SHARED, "resolution": "floor", "requested": None},
    {**SHARED, "resolution": "floor", "submitted": None},
    {**SHARED, "resolution": "omitted"}, {**SHARED, "resolution": "unverifiable"},
    {**SHARED, "resolution": "rejected"},
    {**SHARED, "resolution": "future-resolution"},
    {**SHARED, "resolution": []}, {**SHARED, "source": "unknown"},
    {**SHARED, "source": []}, {**SHARED, "observed": "xhigh"},
    {**SHARED, "observedSource": "provider_response"},
    {**SHARED, "reason": []},
    {**SHARED, "resolution": "rejected", "submitted": None, "parameter": None,
     "observed": "high", "observedSource": "provider_response"},
    *[{key: value for key, value in SHARED.items() if key != missing} for missing in SHARED],
    *[{**SHARED, field: invalid} for field in ("requested", "submitted", "parameter", "observed", "observedSource")
      for invalid in (" ", 1, False, [], {})],
    {**VARIANTS["known_parameter_unverifiable"], "parameter": " "},
])
def test_malformed_or_incoherent_reports_stay_unknown_in_both_consumers(tmp_path, report):
    assert validated_effort_resolution(report) is None
    usage = _finish(report)
    assert usage["effort_resolution"] is None
    assert usage["effort"]["reported"] is None
    assert usage["claudexor"]["applied_options"] is None
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "effort_resolution": report}])
    assert final_attempt_facts(detail, "run-fixture")["effort_resolution"] is None


@pytest.mark.parametrize("report", REPORTS.values(), ids=REPORTS)
def test_paired_fixture_binds_only_to_unique_final_session_attempt(tmp_path, report):
    detail = _write_telemetry(tmp_path, [
        {"attempt_id": "a01", "effort_resolution": {**SHARED, "observed": "wrong", "observedSource": "earlier"}},
        {"attempt_id": "a02", "effort_resolution": report},
    ])
    assert final_attempt_facts(detail, "run-fixture")["effort_resolution"] == report
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a01", "effort_resolution": report}, {"attempt_id": "a02"}])
    assert "effort_resolution" not in final_attempt_facts(detail, "run-fixture")
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "effort_resolution": report}, {"attempt_id": "a02"}])
    assert final_attempt_facts(detail, "run-fixture") == {}


@pytest.mark.parametrize("token", ["future-token", "none"])
def test_exact_tokens_are_not_reranked_and_literal_none_is_not_omission(token):
    report = {**SHARED, "requested": token, "submitted": token, "resolution": "exact"}
    assert validated_effort_resolution(report) == report


@pytest.mark.parametrize("resolution", ["omitted", "unverifiable", "rejected"])
@pytest.mark.parametrize("parameter", [None, "model_reasoning_effort"])
def test_absent_knob_is_null_and_never_confirms_a_tier(resolution, parameter):
    report = {**SHARED, "submitted": None, "parameter": parameter, "resolution": resolution}
    assert validated_effort_resolution(report) == report
    assert _finish(report)["effort"]["reported"] is None


@pytest.mark.parametrize("report", REPORTS.values(), ids=REPORTS)
def test_final_session_report_reaches_review_ledger_and_last_execution(tmp_path, monkeypatch, fake_route, report):
    from types import SimpleNamespace

    from ouroboros.review_substrate import run_review_request
    from ouroboros.reviewer_slot_config import record_reviewer_slot_executions, reviewer_slot_last_executions

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "canonical-data")
    fake_route.detail = _terminal_detail('[]')
    fake_route.telemetry = {"run_id": "run-1", "final_attempt_id": "a02", "attempts": [
        {"attempt_id": "a02", "harness_id": "fake-review", "observed_model": "fake-small",
         "effort_resolution": report},
    ]}
    slot = _agent_slot()
    result = run_review_request(_agent_request(), slots=[slot], drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert actor["usage"]["effort_resolution"] == report
    record_reviewer_slot_executions("scope_review", [SimpleNamespace(**actor)], {slot.slot_id: slot})
    effective = reviewer_slot_last_executions()[slot.slot_id]["effective"]
    assert effective["effort_resolution"] == report
    assert "effort" not in effective  # Prepared effort must not become an applied scalar.
    rows = ledger_rows(tmp_path)
    assert any(row.get("effort_resolution") == report for row in rows)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reported", [None, "xhigh"])
@pytest.mark.parametrize("report", REPORTS.values(), ids=REPORTS)
def test_shared_model_fixture_survives_real_driver_and_monetary_settlement(model_setup, asynchronous, reported, report):
    import asyncio

    root, gateway, client = model_setup
    response = _model_result(cash=0.25, knowledge="exact")
    response.update(effortResolution=report,
                    appliedOptions={"reasoningEffort": reported} if reported else {})
    gateway.results = [response]
    args = ([{"role": "user", "content": "hi"}], MODEL)
    if asynchronous:
        _, usage = asyncio.run(client.chat_async(*args, reasoning_effort="ultra"))
    else:
        _, usage = client.chat(*args, reasoning_effort="ultra")
    assert usage["effort_resolution"] == report
    assert usage["effort"]["requested"] == "ultra"
    assert usage["effort"]["reported"] == reported
    assert usage["effort"]["report_source"] == ("provider_applied_options" if reported else None)
    assert usage["claudexor"]["applied_options"] == response["appliedOptions"]
    rows = ledger(root)
    assert [row["state"] for row in rows] == ["settled"]
    assert rows[-1]["effort_resolution"] == report
    assert rows[-1]["effort"] == usage["effort"]
    assert rows[-1]["cost_usd"] == 0.25


def test_old_model_result_without_resolution_keeps_effort_unknown(model_setup):
    root, gateway, client = model_setup
    response = _model_result(cash=0.25, knowledge="exact")
    response.pop("appliedOptions")
    gateway.results = [response]
    _, usage = client.chat([{"role": "user", "content": "hi"}], MODEL, reasoning_effort="ultra")
    assert "effort_resolution" not in usage
    assert usage["effort"]["reported"] is None
    assert usage["effort"]["report_source"] is None
    assert usage["claudexor"]["applied_options"] is None
    assert not ledger(root)[-1].get("effort_resolution")


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("evidence", ["missing", "prepared", "applied", "observed", "malformed", "malformed_applied"])
def test_late_model_result_keeps_only_exact_attempt_effort_evidence(tmp_path, monkeypatch, online, evidence):
    from types import SimpleNamespace

    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import effort_request_facts
    from ouroboros.llm_claudexor import recover_model_attempt
    from ouroboros.observability import persist_call

    target = {"usage_model": MODEL, "requested_reasoning_effort": "ultra"}
    payload = {"options": {"reasoningEffort": "ultra"}}
    facts = effort_request_facts(target, payload)
    result = _model_result(cash=0.25, knowledge="exact")
    result.pop("appliedOptions", None)
    if evidence != "missing":
        result["effortResolution"] = {} if evidence == "malformed" else dict(SHARED)
    if evidence in {"applied", "observed"}:
        result["appliedOptions"] = {"reasoningEffort": "xhigh"}
    if evidence == "observed":
        result["effortResolution"].update(observed="high", observedSource="provider_response")
    if evidence == "malformed_applied":
        result["appliedOptions"] = {"reasoningEffort": ["xhigh"]}
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="late")):
        reservation = ua.reserve_attempt(ua.AttemptRequest(model=MODEL, provider="claudexor", effort=facts))
        ua.mark_dispatched(reservation)
        ua._transition(reservation, "settled", settle_reason="abandoned", cost_usd=None, cost_final=False,
                       effort={**facts, "reported": "stale", "report_source": "older_receipt"},
                       effort_resolution={**SHARED, "observed": "stale", "observedSource": "older_receipt"})
        row = ledger(tmp_path)[-1]
        attempt = reservation.attempt_id
        persist_call(tmp_path, task_id="late", call_id=f"{attempt}_model_request",
                     call_type="llm_claudexor_request", payload=payload, keep_raw=True,
                     manifest={"invocation_id": attempt, "operation_id": "exact-operation"})
        raw = json.dumps(result)
        calls = []

        def get_operation(operation, **kwargs):
            assert operation == "exact-operation"
            calls.append("read_operation")
            return {"state": "succeeded", "dispatch": {"state": "response_received"},
                    "response": {"state": "ready", "ref": {"sha256": "fixture-ref"}}}

        def get_result(operation, **kwargs):
            assert operation == "exact-operation" and kwargs["expected_ref"] == {"sha256": "fixture-ref"}
            calls.append("read_result")
            return raw.encode()

        def factory():
            assert online, "retained terminal evidence needs no gateway"
            return SimpleNamespace(get_model_operation=get_operation, get_model_result=get_result,
                                   acknowledge_model_result=lambda *args: calls.append("ack"),
                                   close=lambda: calls.append("close"))

        if not online:
            persist_call(tmp_path, task_id="late", call_id=f"{attempt}_model_response",
                         call_type="llm_claudexor_response", payload={"result_json_utf8": raw}, keep_raw=True,
                         manifest={"invocation_id": attempt, "operation_id": "exact-operation",
                                   "operation_state": "succeeded", "dispatch_state": "response_received"})
        monkeypatch.setattr("ouroboros.llm_claudexor.read_owned_gateway", factory)
        state, usage, cost, final = recover_model_attempt(tmp_path, row)
        assert state == "settled"
        live = _ModelInvocation(target, payload, {}).extract_usage(result)[0]
        assert usage.get("effort_resolution") == live.get("effort_resolution")
        assert usage["effort"] == live["effort"]
        ua.settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
    settled = ledger(tmp_path)[-1]
    assert settled["effort"] == live["effort"]
    assert settled.get("effort_resolution") == live.get("effort_resolution")
    assert settled["cost_usd"] == 0.25 and settled["cost_final"] is True
    assert settled["settle_reason"] == "late_receipt"
    assert len({entry["attempt_id"] for entry in ledger(tmp_path)}) == 1
    assert calls == (["read_operation", "read_result", "ack", "close"] if online else [])
