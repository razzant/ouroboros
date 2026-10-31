"""Incoming prices preserve their knowledge through the real accounting consumers."""

import copy
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import usage_accounting as ua
from ouroboros._usage_response import usage_from_response
from tests import _review_session_route_shared as session_fixtures
from tests._usage_store_testing import ledger_rows

FakeGateway = session_fixtures.FakeGateway
fake_route = session_fixtures.fake_route
_owned_gateway_uses_each_test_transport = session_fixtures._owned_gateway_uses_each_test_transport

MISSING = object()
CASES = [
    pytest.param(MISSING, None, id="absent"),
    pytest.param(None, None, id="null"),
    pytest.param(0, 0.0, id="zero"),
    pytest.param("1.25", 1.25, id="numeric-string"),
    pytest.param(2.5, 2.5, id="positive"),
    pytest.param(True, None, id="true"),
    pytest.param(False, None, id="false"),
    pytest.param(-1, None, id="negative"),
    pytest.param(float("nan"), None, id="nan"),
    pytest.param(float("inf"), None, id="infinity"),
    pytest.param(float("-inf"), None, id="negative-infinity"),
    pytest.param(10**1000, None, id="overflow"),
    pytest.param("garbage", None, id="garbage"),
    pytest.param("", None, id="empty-string"),
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                        lambda *args: pytest.fail("unexpected real HTTP"))


def _missing(value):
    return value is MISSING or value is None


def _reserve(root, *, provider="openrouter"):
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        "vendor/model", provider, drive_root=root, task_id="task", root_task_id="task",
        reservation_usd=0.5))
    ua.mark_dispatched(reservation)
    return reservation


@pytest.mark.parametrize("raw,known", CASES)
def test_public_settlement_rejects_malformed_without_estimating(tmp_path, monkeypatch, raw, known):
    calls = []
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: calls.append(k) or 0.75)
    reservation = _reserve(tmp_path)
    calls.clear()
    ua.settle_attempt(reservation, {"prompt_tokens": 5},
                      cost_usd=None if raw is MISSING else raw, cost_final=True)
    row, = ledger_rows(tmp_path)
    assert row["cost_usd"] == (0.75 if _missing(raw) else known)
    assert row["cost_final"] is (known is not None)
    assert len(calls) == int(_missing(raw))
    assert bool(row.get("cost_invalid")) is (known is None and not _missing(raw))


@pytest.mark.parametrize("alternate", [0, "1.25"])
def test_alternate_provider_evidence_survives_public_settlement(tmp_path, monkeypatch, alternate):
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: pytest.fail("unexpected estimate"))
    normalized, cost, final = usage_from_response({"usage": {"cost": True, "total_cost": alternate}})
    ua.settle_attempt(_reserve(tmp_path), normalized, cost_usd=cost, cost_final=final)
    row, = ledger_rows(tmp_path)
    assert row["cost_usd"] == float(alternate) and row["cost_final"]
    assert not row.get("cost_invalid")


def test_settlement_keeps_normalized_invalid_unknown_and_local_free(tmp_path, monkeypatch):
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: pytest.fail("unexpected estimate"))
    ua.settle_attempt(_reserve(tmp_path), {"prompt_tokens": 5, "cost_invalid": True})
    ua.settle_attempt(_reserve(tmp_path, provider="local"), {"cost_invalid": True}, cost_usd=True)
    remote, local = ledger_rows(tmp_path)
    assert remote["cost_usd"] is None and not remote["cost_final"]
    assert local["cost_usd"] == 0 and local["cost_final"]


@pytest.mark.parametrize("raw,known", CASES)
def test_subscription_boundary_validates_and_replay_never_reprices(tmp_path, raw, known):
    kwargs = {} if raw is MISSING else {"spend_usd": raw}
    attempt = ua.record_subscription_session("session", drive_root=tmp_path, route="fixture", **kwargs)
    before = ledger_rows(tmp_path)
    row, = before
    assert row["cost_usd"] == known and row["cost_final"] is (known is not None)
    assert row["reservation_upper_bound_usd"] == known
    assert bool(row.get("cost_invalid")) is (known is None and not _missing(raw))
    assert ua.record_subscription_session("session", drive_root=tmp_path, route="fixture",
                                          model="later", spend_usd=8) == attempt
    assert ledger_rows(tmp_path) == before


@pytest.mark.parametrize("raw,known", CASES)
def test_terminal_collection_and_nanny_validate_legacy_summary(tmp_path, raw, known):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_collection import record_terminal_money
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload

    summary = {"state": "succeeded"}
    if raw is not MISSING:
        summary["spendUsd"] = raw
    detail = {"summary": summary, "primaryOutput": {"text": "usable output"}}
    entry = custody.RunCustody(run_id="run", task_id="task", route_id="fixture", ledger_root=str(tmp_path))
    assert record_terminal_money(tmp_path, entry, detail)["settled"]
    row, = ledger_rows(tmp_path)
    payload = _terminal_payload("run", detail, DelegatedRunShape("readonly", "ask", "live", False))
    assert payload["cost"]["cost_usd"] == row["cost_usd"] == known
    assert payload["cost"]["cost_final"] is row["cost_final"] is (known is not None)
    assert record_terminal_money(tmp_path, entry, detail)["settled"]
    assert ledger_rows(tmp_path) == [row]


@pytest.mark.parametrize("queued", [True, False], ids=["queued", "direct"])
@pytest.mark.parametrize("raw,known", CASES)
def test_safety_verdict_and_telemetry_keep_price_knowledge(tmp_path, monkeypatch, queued, raw, known):
    payload = {"provider": "openrouter", "prompt_tokens": 5}
    if raw is not MISSING:
        payload["cost"] = raw
    usage, emitted, calls = _safety_usage(tmp_path, monkeypatch, payload, queued=queued)
    expected = 0.75 if _missing(raw) else known
    assert usage["cost"] == expected
    assert len(calls) == int(_missing(raw))
    assert bool(usage.get("cost_estimated")) is _missing(raw)
    if expected is None or _missing(raw):
        assert usage["cost_final"] is False
    if queued:
        assert emitted["cost"] == expected
        assert emitted["cost_known"] is (expected is not None)
        assert emitted["cost_invalid"] is (known is None and not _missing(raw))
        assert emitted["cost_estimated"] is _missing(raw)


def _safety_usage(root, monkeypatch, payload, *, queued=True, local=False):
    from ouroboros import safety
    from supervisor.events_budget import _handle_llm_usage
    from tests.test_safety_rate_limit import _DriveCtx, _patch_llm_client, _ScriptedLLMClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "vendor/model")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "full")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true" if local else "false")
    calls = []
    def estimate(*args, **kwargs):
        calls.append(kwargs)
        return 0.75

    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", estimate)
    if hasattr(safety, "estimate_cost_optional"):  # same tariff for the original-base negative control
        monkeypatch.setattr(safety, "estimate_cost_optional", estimate)
    stub = _ScriptedLLMClient([('{"status":"SAFE","reason":"ok"}', payload)])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(root)
    captured = []
    if queued:
        ctx.event_queue = queue.Queue()
    else:
        ctx.update_budget_from_usage = lambda value: captured.append(copy.deepcopy(value))
    assert safety.check_safety("github_write", {"method": "create_issue", "title": "fixture"}, ctx=ctx) == (True, "")
    assert len(stub.calls) == 1
    if not queued:
        return captured[0], None, calls
    events = list(ctx.event_queue.queue)
    event, = [item for item in events if item["type"] == "llm_usage"]
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=root, bridge=SimpleNamespace(push_log=captured.append)))
    stored = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text().splitlines()]
    persisted, = [item for item in stored if item["type"] == "llm_usage"]
    assert persisted == captured[0]
    return event["usage"], persisted, calls


@pytest.mark.parametrize("local", [False, True])
def test_safety_normalized_invalid_never_estimates_but_local_is_free(tmp_path, monkeypatch, local):
    usage, event, calls = _safety_usage(tmp_path, monkeypatch,
        {"prompt_tokens": 5, "cost": None, "cost_invalid": True, "cost_final": False}, local=local)
    assert usage["cost"] == (0 if local else None)
    assert usage["cost_final"] is local
    assert event["cost_invalid"] is (not local)
    assert calls == []


@pytest.mark.parametrize("raw,known", CASES)
def test_raw_usage_events_do_not_turn_malformed_into_known_money(tmp_path, raw, known):
    from supervisor.events_budget import _handle_llm_usage

    rows = []
    event = {"type": "llm_usage", "usage": {}}
    if raw is not MISSING:
        event["cost"] = raw
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=rows.append)))
    row, = rows
    assert row["cost"] == known and row["cost_known"] is (known is not None)
    assert row["cost_invalid"] is (known is None and not _missing(raw))


@pytest.mark.parametrize("last_cost", [0, 3.0, None, True], ids=["zero", "positive", "unknown", "invalid"])
def test_review_continuation_uses_each_runs_typed_cash_and_collects_once(tmp_path, monkeypatch, fake_route, last_cost):
    from ouroboros.delegate_custody_usage import observe_session_usage
    from tests.test_review_session_continuation import _engine, _run

    _engine(monkeypatch)
    original = FakeGateway.get_run
    evidence = {}

    def detail(self, run_id, **kw):
        value = original(self, run_id, **kw)
        amount = 2.0 if run_id == "run-1" else last_cost
        value["summary"]["spendUsd"] = 99
        value["attemptExecution"] = [{"attemptId": f"{run_id}-a", "usageCost": {
            "cashUsd": amount, "cashKnowledge": "exact"}}]
        evidence[run_id] = copy.deepcopy(value["attemptExecution"])
        return value

    monkeypatch.setattr(FakeGateway, "get_run", detail)
    checkpoints, stamps = [], []
    facts = _run(tmp_path, {}, checkpoints, stamps)
    rows = ledger_rows(tmp_path)
    expected = 2.0 + (last_cost if type(last_cost) in (int, float) else 0)
    partial = last_cost is None or last_cost is True
    assert facts["text"] == "[]" and facts["spend"] == expected
    assert facts["spend_estimated"] is False  # Exact known cash, even when the total is incomplete.
    assert facts["usage"]["cost_partial"] is partial
    assert facts["usage"]["cost_disclosed_usd"] == expected
    assert facts["usage"]["cost_final"] is (not partial)
    assert len(rows) == 2 and sum(row["cost_usd"] or 0 for row in rows) == expected
    assert [row["attempt_execution"] for row in rows] == list(evidence.values())
    per_run = facts["usage"]["session_usage_events"]
    assert [part["attempt_execution"] for part in per_run] == list(evidence.values())
    assert [part["cost_disclosed_usd"] for part in per_run] == [2, None if partial else last_cost]
    monkeypatch.setattr("ouroboros.delegate_custody_usage._EMITTED_SESSION_USAGE", {})
    observed = []
    observe_session_usage(observed.append, facts["usage"])
    observe_session_usage(observed.append, facts["usage"])
    assert observed == per_run
    replay = _run(tmp_path, {"pending_invocation_id": checkpoints[0]}, [], [], reconcile=True)
    assert replay["spend"] == expected and ledger_rows(tmp_path) == rows


@pytest.mark.parametrize("amounts,expected", [([2, None], 2), ([0, None], 0), ([1e308, 1e308], None)])
def test_typed_partial_and_overflow_keep_the_exact_attempt_evidence(tmp_path, amounts, expected):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_collection import record_terminal_money

    attempts = [{"usageCost": {"cashUsd": amount, "cashKnowledge": "exact"}} for amount in amounts]
    detail = {"summary": {"state": "succeeded", "spendUsd": 99}, "attemptExecution": attempts}
    entry = custody.RunCustody(run_id="run", task_id="task", route_id="fixture", ledger_root=str(tmp_path))
    assert record_terminal_money(tmp_path, entry, detail)["settled"]
    row, = ledger_rows(tmp_path)
    assert row["cost_usd"] == expected and not row["cost_final"]
    if None in amounts:
        assert row["cost_partial"] is True
        assert row["cost_estimated"] is False
        assert row["reservation_upper_bound_usd"] is None
    assert row["attempt_execution"] == attempts


def test_review_actor_reports_the_typed_cash_the_collector_settled(tmp_path, fake_route):
    from ouroboros import review_substrate
    from tests._review_session_route_shared import FakeLLM, _agent_request, _agent_slot

    attempts = [{"usageCost": {"cashUsd": "2.75", "cashKnowledge": "exact"}}]
    FakeGateway.detail["summary"]["spendUsd"] = True
    FakeGateway.detail["attemptExecution"] = attempts
    result = review_substrate.run_review_request(
        _agent_request(session_root=str(tmp_path), task_id="review-task"),
        slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM())
    actor, = result.actors
    assert actor["status"] == "ok"
    assert actor["usage"]["cost"] == actor["usage"]["cost_disclosed_usd"] == 2.75
    assert actor["usage"]["attempt_execution"] == attempts
    session, = [row for row in ledger_rows(tmp_path) if row["kind"] == "subscription_session"]
    assert session["cost_usd"] == 2.75 and session["cost_final"]


def test_review_failed_successor_keeps_prior_evidence_on_its_original_run(tmp_path, monkeypatch, fake_route):
    from ouroboros import review_execution
    from tests.test_review_session_continuation import _engine, _run

    _engine(monkeypatch)
    original_detail = FakeGateway.get_run
    attempts = [{"usageCost": {"cashUsd": 2, "cashKnowledge": "exact"}}]

    def detail(self, run_id, **kw):
        value = original_detail(self, run_id, **kw)
        value["attemptExecution"] = attempts
        return value

    monkeypatch.setattr(FakeGateway, "get_run", detail)
    poll = review_execution._poll_session_terminal

    def fail_successor(gateway, custody, drive, entry, run_id, seconds):
        if run_id == "run-2":
            raise OSError("fixture observation unavailable")
        return poll(gateway, custody, drive, entry, run_id, seconds)

    monkeypatch.setattr(review_execution, "_poll_session_terminal", fail_successor)
    with pytest.raises(OSError) as error:
        _run(tmp_path, {}, [], [])
    usage = error.value.delegated_usage
    first, last = usage["session_usage_events"]
    assert first["delegated_run_id"] == "run-1" and first["attempt_execution"] == attempts
    assert first["cost_disclosed_usd"] == 2
    assert last["delegated_run_id"] == "run-2" and last["cost_disclosed_usd"] is None
    assert "attempt_execution" not in last


@pytest.mark.parametrize("raw,known", CASES)
def test_image_result_survives_price_validation_without_text_tariff(tmp_path, monkeypatch, raw, known):
    from ouroboros.gateways import image_api
    from ouroboros.tools import imagegen
    from ouroboros.tools.registry import ToolContext
    from tests.test_imagegen_tool import BODY, PNG

    result = copy.deepcopy(BODY)
    result["usage"] = {"prompt_tokens": 500, "completion_tokens": 600}
    if raw is not MISSING:
        result["usage"]["cost"] = raw
    monkeypatch.setattr(image_api, "resolve_image_target", lambda model: {
        "provider": "openai", "model": "fixture-image", "base_url": "https://fixture.invalid/v1",
        "api_key": "fixture-key", "headers": {}})
    monkeypatch.setattr(image_api, "send_image_request", lambda *a: result)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: pytest.fail("no image text tariff"))
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="image-task")
    text = imagegen._generate_image(ctx, "fixture image", "openai::fixture-image")
    assert text.startswith("OK:")
    summary = json.loads(text.split("\n", 1)[1])
    assert summary["generated"] == 1 and Path(summary["images"][0]["absolute_path"]).read_bytes() == PNG
    assert summary["actual_cost_usd"] == known and summary["cost_final"] is (known is not None)
    row, = ledger_rows(ctx.drive_root)
    assert row["cost_usd"] == known and row["cost_final"] is (known is not None)
    assert row["cost_basis"] == "image_generation"
    assert bool(row.get("cost_invalid")) is (known is None and not _missing(raw))
