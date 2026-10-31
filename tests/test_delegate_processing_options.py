"""Captured processing intent changes only a supported physical request option."""
from types import SimpleNamespace

import pytest

from ouroboros.tools.delegate import _processing_start_request
from tests._delegated_transport_shared import (  # noqa: F401
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests._review_session_route_shared import (  # noqa: F401
    FakeGateway,
    _agent_slot,
    _terminal_detail,
    fake_route as fake_route,
)
from tests._usage_store_testing import ledger_rows
from tests.test_review_operation_collection import forbid_effects  # noqa: F401


@pytest.mark.parametrize("preference", ["standard", "fast", "economy"])
def test_captured_processing_preference_uses_exact_route_capability(preference):
    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [
        {"id": "other", "processingPreferences": []},
        {"id": "chosen", "processingPreferences": ["standard", "fast", "economy"]}]})
    body = {"instructions": "captured instructions", "model": "exact-model", "effort": "xhigh"}
    result, facts = _processing_start_request(body, {"processing_preference": preference}, gateway,
                                       SimpleNamespace(route_id="chosen"))
    assert result == {"instructions": "captured instructions", "model": "exact-model", "effort": "xhigh",
                      "processingPreference": preference}
    assert facts["submitted"] == preference and facts["observed"] == "unknown"


def test_old_engine_does_not_receive_unsupported_wire_field():
    gateway = SimpleNamespace(agent_capabilities=lambda: {"harnesses": [{"id": "chosen"}]})
    body = {"instructions": "captured", "model": "exact-model", "effort": "xhigh"}
    result, facts = _processing_start_request(body, {"processing_preference": "economy"}, gateway,
                                       SimpleNamespace(route_id="chosen"))
    assert "processingPreference" not in result
    assert result["instructions"] == "captured"
    assert facts["reason"] == "processing_not_submitted" and facts["submitted"] is None
    assert result["model"] == "exact-model" and result["effort"] == "xhigh"


def test_no_preference_preserves_legacy_body_without_catalog_read():
    body = {"instructions": "legacy", "model": "exact-model"}
    assert _processing_start_request(body, {}, None, None) == (body, {})


@pytest.mark.parametrize("costs,expected,final", [
    ([{"cashUsd": 0, "cashKnowledge": "exact", "valuationUsd": 8}], 0.0, True),
    ([{"cashUsd": None, "cashKnowledge": "unknown", "valuationUsd": 8}], None, False),
    ([{"cashUsd": 1.25, "cashKnowledge": "exact"}, {"cashKnowledge": "unknown"}], 1.25, False),
    ([{"cashUsd": 1.25, "cashKnowledge": "exact"}, {"cashUsd": 0.5, "cashKnowledge": "exact"}], 1.75, True),
])
def test_terminal_and_existing_ledger_use_attempt_cash_not_valuation(tmp_path, costs, expected, final):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload

    attempts = [{"attemptId": str(index), "usageCost": cost} for index, cost in enumerate(costs)]
    detail = {"summary": {"state": "succeeded", "spendUsd": 99}, "attemptExecution": attempts}
    entry = custody.RunCustody(run_id="cash-test", task_id="task", route_id="session",
                              project_persistent=True, ledger_root=str(tmp_path))
    result = custody.settle_run(tmp_path, None, entry, detail)
    assert result["ledger_recorded"]
    rows = ledger_rows(tmp_path)
    row = next(row for row in rows if row.get("kind") == "subscription_session")
    payload = _terminal_payload("cash-test", detail, DelegatedRunShape("readonly", "ask", "live", False))
    assert payload["cost"]["cost_usd"] == row["cost_usd"] == expected
    assert payload["cost"]["cost_final"] == row["cost_final"] == final
    assert payload["attempt_execution"] == attempts


# --------------------------------------------------------------------------- native session prompt_cache_ttl (#1705)


@pytest.mark.parametrize("ttl,harnesses,control_keys,expected_wire,expected_reason", [
    ("default", [{"id": "chosen", "promptCacheTtls": ["5m", "1h"]}], ["promptCacheTtl"], None, None),
    ("5m", [{"id": "chosen", "promptCacheTtls": ["5m", "1h"]}], ["promptCacheTtl"], "5m", None),
    ("1h", [{"id": "chosen", "promptCacheTtls": ["5m", "1h"]}], ["promptCacheTtl"], "1h", None),
    # Gated by exact advertised value, never by harness name:
    ("1h", [{"id": "claude", "promptCacheTtls": ["5m"]}], ["promptCacheTtl"], None, "harness_unsupported"),
    ("1h", [{"id": "claude", "promptCacheTtls": []}], ["promptCacheTtl"], None, "harness_unsupported"),
    ("1h", [{"id": "custom-harness", "promptCacheTtls": ["1h"]}], ["promptCacheTtl"], "1h", None),
    # Older strict engine omits promptCacheTtls and runControlKeys entry:
    ("1h", [{"id": "chosen"}], [], None, "engine_capability_missing"),
])
def test_session_prompt_cache_ttl_negotiates_by_exact_capability(
    monkeypatch, ttl, harnesses, control_keys, expected_wire, expected_reason,
):
    from ouroboros.effort_evidence import session_prompt_cache_ttl_request

    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", ttl)
    calls = []

    def capabilities():
        calls.append(1)
        return {"harnesses": harnesses, "runControlKeys": control_keys}

    gateway = SimpleNamespace(agent_capabilities=capabilities)
    body = {"prompt": "check", "instructions": "inst"}
    facts = session_prompt_cache_ttl_request(body, route_id="chosen" if harnesses[0]["id"] == "chosen" else harnesses[0]["id"],
                                             gateway=gateway)
    if expected_wire is None:
        assert "promptCacheTtl" not in body
    else:
        assert body["promptCacheTtl"] == expected_wire
    if ttl == "default":
        assert calls == []
        assert facts == {
            "requested": None, "submitted": None, "reason": None,
            "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
        }
    else:
        assert facts == {
            "requested": ttl, "submitted": expected_wire, "reason": expected_reason,
            "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
        }


def test_session_prompt_cache_ttl_capability_read_failure_preserves_run(monkeypatch):
    from ouroboros.effort_evidence import session_prompt_cache_ttl_request

    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")

    def broken_caps():
        raise RuntimeError("daemon capability read timed out")

    body = {"prompt": "work"}
    facts = session_prompt_cache_ttl_request(body, route_id="claude", gateway=SimpleNamespace(agent_capabilities=broken_caps))
    assert "promptCacheTtl" not in body
    assert facts == {
        "requested": "1h", "submitted": None, "reason": "capabilities_read_failed",
        "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
    }


@pytest.mark.parametrize("valid_report", [
    {"requested": "1h", "submitted": "1h", "reason": None, "observed": "1h", "write_1h_tokens": 1200, "write_5m_tokens": 0},
    {"requested": "5m", "submitted": "5m", "reason": None, "observed": "5m", "write_1h_tokens": 0, "write_5m_tokens": 450},
    {"requested": "1h", "submitted": "1h", "reason": None, "observed": "mixed", "write_1h_tokens": 800, "write_5m_tokens": 200},
    {"requested": "1h", "submitted": "1h", "reason": None, "observed": "none", "write_1h_tokens": 0, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "reason": "cli_below_2_1_242", "observed": "5m", "write_1h_tokens": 0, "write_5m_tokens": 300},
    {"requested": "1h", "submitted": None, "reason": "harness_unsupported", "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None},
    {"requested": None, "submitted": None, "reason": None, "observed": "5m", "write_1h_tokens": 0, "write_5m_tokens": 100},
])
def test_final_attempt_prompt_cache_ttl_binds_unique_final_attempt(tmp_path, valid_report):
    from ouroboros.effort_evidence import validated_prompt_cache_ttl
    from ouroboros.gateways.claudexor import final_attempt_facts
    from tests.test_claudexor_observed_attempt import _write_telemetry

    assert validated_prompt_cache_ttl(valid_report) == valid_report
    earlier = {"requested": "5m", "submitted": "5m", "reason": None, "observed": "5m", "write_1h_tokens": 0, "write_5m_tokens": 999}
    detail = _write_telemetry(tmp_path, [
        {"attempt_id": "a01", "harness_id": "claude", "observed_model": "early", "prompt_cache_ttl": earlier},
        {"attempt_id": "a02", "harness_id": "claude", "observed_model": "final", "prompt_cache_ttl": valid_report},
        {"attempt_id": "a03", "harness_id": "claude", "observed_model": "late", "prompt_cache_ttl": earlier},
    ])
    facts = final_attempt_facts(detail, "run-fixture")
    assert facts["prompt_cache_ttl"] == valid_report


@pytest.mark.parametrize("malformed", [
    None, {}, [], "1h",
    {"requested": "default", "submitted": None, "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None},
    {"requested": "2h", "submitted": "2h", "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None},
    {"requested": "1h", "submitted": "5m", "observed": "1h", "write_1h_tokens": 100, "write_5m_tokens": 0},
    {"requested": None, "submitted": "1h", "observed": "1h", "write_1h_tokens": 100, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "1h", "write_1h_tokens": 0, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "none", "write_1h_tokens": 10, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "5m", "write_1h_tokens": 10, "write_5m_tokens": 10},
    {"requested": "1h", "submitted": "1h", "observed": "mixed", "write_1h_tokens": 10, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "unknown", "write_1h_tokens": 0, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "1h", "write_1h_tokens": 100, "write_5m_tokens": None},
    {"requested": "1h", "submitted": "1h", "observed": "1h", "write_1h_tokens": True, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "observed": "1h", "write_1h_tokens": -1, "write_5m_tokens": 0},
    {"requested": "1h", "submitted": "1h", "reason": 123, "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None},
])
def test_malformed_prompt_cache_ttl_telemetry_never_fabricates_observation(tmp_path, malformed):
    from ouroboros.effort_evidence import terminal_prompt_cache_ttl, validated_prompt_cache_ttl
    from ouroboros.gateways.claudexor import final_attempt_facts
    from tests.test_claudexor_observed_attempt import _write_telemetry

    assert validated_prompt_cache_ttl(malformed) is None
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "prompt_cache_ttl": malformed}])
    observed = final_attempt_facts(detail, "run-fixture")
    assert observed["prompt_cache_ttl"] is None
    assert terminal_prompt_cache_ttl(observed) is None
    preserved = terminal_prompt_cache_ttl(observed, {"requested": "1h", "submitted": "1h", "reason": None})
    assert preserved == {
        "requested": "1h", "submitted": None, "reason": "malformed_engine_telemetry",
        "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
    }


def test_task_captured_ttl_ignores_live_settings_file_and_measures_before_custody(tmp_path, monkeypatch):
    import json
    from ouroboros import config
    from ouroboros.review_session_preparation import prepare_review_session_request
    from ouroboros.settings_integrity import task_settings_snapshot

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"OUROBOROS_PROMPT_CACHE_TTL": "5m"}), encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_file)
    # Live settings say 1h (also the shipped default); only the task's capture says 5m.
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")
    invocation = SimpleNamespace(
        timeout_sec=120, use_thread=False, output_schema=None,
        instructions="review inst", access="readonly", task_id="t-1", source_delivery={},
    )
    route = SimpleNamespace(route_id="claude", model="claude-opus-5-5", effort="high", profile_id="")
    caps = {"harnesses": [{"id": "claude", "promptCacheTtls": ["5m", "1h"]}], "runControlKeys": ["promptCacheTtl"]}

    def prepare(delivery):
        return prepare_review_session_request(
            invocation, route, prompt="review diff", root=str(tmp_path),
            thread_id="", schema_asked=False, delivery_facts=delivery, capabilities=caps,
        )

    snapshot = task_settings_snapshot(
        {"OUROBOROS_PROMPT_CACHE_TTL": "5m"},
        {"OUROBOROS_PROMPT_CACHE_TTL": "5m"},
    )
    with config.task_settings_scope(snapshot):
        # Mutate live settings.json on disk after task start:
        settings_file.write_text(json.dumps({"OUROBOROS_PROMPT_CACHE_TTL": "1h"}), encoding="utf-8")
        assert config.load_settings()["OUROBOROS_PROMPT_CACHE_TTL"] == "1h"
        assert config.resolve_prompt_cache_ttl() == "5m"

        delivery_with_ttl = {}
        req_with_ttl = prepare(delivery_with_ttl)
        assert req_with_ttl["promptCacheTtl"] == "5m"
        serialized = json.dumps(req_with_ttl, ensure_ascii=False)
        assert delivery_with_ttl["first_send_chars"] == len(serialized)
        assert delivery_with_ttl["first_send_bytes"] == len(serialized.encode("utf-8"))
        assert delivery_with_ttl["prompt_cache_ttl"] == {
            "requested": "5m", "submitted": "5m", "reason": None,
            "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
        }

        # Falsifying check: without promptCacheTtl the measured serialized size is strictly smaller.
        without_ttl = dict(req_with_ttl)
        without_ttl.pop("promptCacheTtl")
        assert delivery_with_ttl["first_send_bytes"] > len(json.dumps(without_ttl, ensure_ascii=False).encode("utf-8"))

    # Outside the task's scope the same producer reads the live 1h: the capture, not a default, chose 5m.
    assert config.resolve_prompt_cache_ttl() == "1h"
    live_delivery = {}
    assert prepare(live_delivery)["promptCacheTtl"] == "1h"
    assert live_delivery["prompt_cache_ttl"]["requested"] == live_delivery["prompt_cache_ttl"]["submitted"] == "1h"


def test_delegate_start_retry_and_continuation_preserve_ttl_intent_without_settings_reread(tmp_path, monkeypatch):
    import json
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor as gw
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools import delegate
    from ouroboros.tools.delegate_terminal_evidence import resolved_terminal_payload
    from tests.test_claudexor_observed_attempt import _write_telemetry

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude=claude-opus-5-5:high")
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")

    posts = []
    fail_next = [True]

    class _Stub:
        engine_version = "4.5.0"
        def handshake(self, **_kw): return {}
        def agent_capabilities(self, **_kw):
            return {
                "runControlKeys": ["continueFrom", "promptCacheTtl"],
                "harnesses": [{
                    "id": "claude", "enabled": True, "status": "ok",
                    "accessProfilesSupported": ["readonly", "workspace_write"],
                    "promptCacheTtls": ["5m", "1h"],
                }],
            }
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-1"
        def register_project(self, root): return "prj-1"
        def remove_project(self, project_id): return {}
        def start_run(self, request, *, idempotency_key=""):
            posts.append((dict(request), idempotency_key))
            if fail_next[0]:
                raise gw.ClaudexorUnavailable("daemon_unreachable", "lost response", status_code=503)
            return {"runId": f"run-{len(posts)}", "runDir": str(tmp_path / f"run-{len(posts)}")}
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-ttl")

    # 1. Initial start records START_REQUESTED with promptCacheTtl='1h' before transport fails:
    first_res = json.loads(delegate._delegate_start(ctx, "inspect repo").text)
    pending_id = first_res["pending_invocation_id"]
    assert posts[0][0]["promptCacheTtl"] == "1h"

    # 2. Change live setting to '5m'; retry_of must replay the exact '1h' request bytes and start facts:
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "5m")
    fail_next[0] = False
    retry_res = json.loads(delegate._delegate_start(ctx, "inspect repo", retry_of=pending_id).text)
    assert retry_res["status"] == "started" and retry_res["idempotent_recovery"] is True
    assert posts[1][0] == posts[0][0] and posts[1][0]["promptCacheTtl"] == "1h"
    assert retry_res["prompt_cache_ttl"]["requested"] == "1h" and retry_res["prompt_cache_ttl"]["submitted"] == "1h"

    # 3. Settle run-2 and verify terminal payload merges final_attempt_id telemetry:
    run_id = retry_res["run_id"]
    run_dir = tmp_path / run_id
    report = {"requested": "1h", "submitted": "1h", "reason": None, "observed": "1h", "write_1h_tokens": 2048, "write_5m_tokens": 0}
    detail = _write_telemetry(run_dir, [
        {"attempt_id": "a01", "harness_id": "claude", "observed_model": "claude-opus-5-5",
         "prompt_cache_ttl": {"requested": "1h", "submitted": "1h", "reason": None, "observed": "5m", "write_1h_tokens": 0, "write_5m_tokens": 100}},
        {"attempt_id": "a02", "harness_id": "claude", "observed_model": "claude-opus-5-5", "prompt_cache_ttl": report},
    ], run_id=run_id)
    detail["summary"].update(
        state="failed", spendUsd=0.0, spendEstimated=False,
        outcomeFacts={"lifecycle": "failed", "reason": "subscription_window_exhausted"},
        inputTokenUsage={"total_tokens": 3000, "cache_read_tokens": 952, "cache_write_tokens": 2048},
    )
    monkeypatch.setattr("ouroboros.install_identity.require_owned_root", lambda *_a: None)
    drive = custody.custody_root(ctx)
    entry = custody.replay(drive)[run_id]
    assert custody.settle_run(drive, _Stub(), entry, detail)["settled"]
    term_payload, _, _ = resolved_terminal_payload(
        run_id, detail, DelegatedRunShape("readonly", "ask", "live", False), entry=custody.replay(drive)[run_id],
    )
    assert term_payload["prompt_cache_ttl"] == report
    assert term_payload["observed_attempt"]["prompt_cache_ttl"] == report
    # Existing 3-key input_token_usage and non-string _usage_rows.ttls remain unpolluted:
    session_row = [r for r in ledger_rows(drive) if r.get("kind") == "subscription_session"][-1]
    assert "input_token_usage" in session_row, session_row
    assert session_row["input_token_usage"] == {
        "total_tokens": 3000, "cache_read_tokens": 952, "cache_write_tokens": 2048,
    }
    assert not session_row.get("prompt_cache_ttl")

    # 4. Continuation omits promptCacheTtl on wire so engine inherits predecessor intent, ignoring current '5m':
    cont_res = json.loads(delegate._delegate_start(ctx, "continue work", continue_from=run_id).text)
    assert cont_res["status"] == "started"
    assert posts[2][0]["continueFrom"] == run_id
    assert "promptCacheTtl" not in posts[2][0]
    assert cont_res["prompt_cache_ttl"] == {
        "requested": "1h", "submitted": None, "reason": "inherited_from_continuation",
        "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
    }


def test_review_thread_and_continuation_carry_ttl_on_turns_and_keep_exact_continue_from_body(
    tmp_path, monkeypatch, fake_route,
):
    from ouroboros.review_execution import SessionInvocation, run_delegated_review_session
    from ouroboros.review_session_custody import review_continuation_descends_from
    from ouroboros.reviewer_slot_config import record_reviewer_slot_executions, reviewer_slot_last_executions
    from tests.test_claudexor_observed_attempt import _write_telemetry

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "canonical-data")
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")
    monkeypatch.setattr(
        FakeGateway, "agent_capabilities",
        lambda self: {
            "harnesses": [{**FakeGateway.catalog_entry, "promptCacheTtls": ["5m", "1h"]}],
            "runControlKeys": ["continueFrom", "promptCacheTtl"],
        },
    )

    # 1. Thread review turn: thread creation has no promptCacheTtl; turn start carries promptCacheTtl='1h'.
    created_threads, started_turns = [], []
    monkeypatch.setattr(
        FakeGateway, "create_thread",
        lambda self, req, *, idempotency_key="": (created_threads.append(dict(req)) or {"id": "thr-1"}),
        raising=False,
    )
    def _turn(self, tid, req, *, idempotency_key=""):
        started_turns.append((tid, dict(req)))
        return {"threadId": tid, "turnId": "turn-1", "runId": "run-1"}
    monkeypatch.setattr(FakeGateway, "start_thread_turn", _turn, raising=False)
    monkeypatch.setattr(
        FakeGateway, "get_thread",
        lambda self, tid: {"id": tid, "turns": [{"id": "turn-1", "runId": "run-1"}]},
        raising=False,
    )

    report = {"requested": "1h", "submitted": "1h", "reason": None, "observed": "1h", "write_1h_tokens": 512, "write_5m_tokens": 0}
    run_dir = tmp_path / "run-1"
    _write_telemetry(run_dir, [
        {"attempt_id": "a02", "harness_id": "fake-review", "observed_model": "fake-small", "prompt_cache_ttl": report},
    ], run_id="run-1")
    detail = _terminal_detail("[]")
    detail["summary"]["runDir"] = str(run_dir)
    monkeypatch.setattr(FakeGateway, "get_run", lambda self, rid, **_kw: detail)

    delivery = {}
    inv = SessionInvocation(
        task_id="t-rev", surface="review", slot_id="slot-1", timeout_sec=60,
        use_thread=True, thread_id="", source_delivery=delivery,
    )
    facts = run_delegated_review_session(prompt="review it", root=str(tmp_path), custody_drive=tmp_path, invocation=inv)
    assert len(created_threads) == 1 and "promptCacheTtl" not in created_threads[0]
    assert len(started_turns) == 1 and started_turns[0][1]["promptCacheTtl"] == "1h"
    assert delivery["prompt_cache_ttl"]["submitted"] == "1h"
    assert facts["observed_attempt"]["prompt_cache_ttl"] == report
    # Session usage never puts a dict in string usage['prompt_cache_ttl']:
    assert "prompt_cache_ttl" not in facts["usage"]

    slot = _agent_slot()
    actor = SimpleNamespace(
        slot_id=slot.slot_id, status="ok",
        usage={**facts["usage"], "observed_attempt": facts["observed_attempt"], "review_source_delivery": delivery},
    )
    record_reviewer_slot_executions("review", [actor], {slot.slot_id: slot})
    assert reviewer_slot_last_executions()[slot.slot_id]["effective"]["prompt_cache_ttl"] == report

    # 2. Review continuation body remains EXACT {"continueFrom": run_id}:
    from ouroboros import delegate_custody as custody

    anchor = custody.replay(tmp_path)["run-1"]
    custody.record_start_requested(
        tmp_path, run_id="", task_id="t-rev", idempotency_key="k-exact", invocation_id="inv-exact",
        operation_id="", max_seconds=60, request={"continueFrom": "run-1"},
        project_id="", project_owned=False, route="fake-review", surface="review", slot_id="slot-1",
        continuation_of="run-1",
    )
    custody.record_started(
        tmp_path,
        custody.RunCustody(
            run_id="run-exact", task_id="t-rev", route_id="fake-review",
            invocation_id="inv-exact", continuation_of="run-1",
        ),
        shape={"surface": "review", "slot_id": "slot-1"},
    )
    custody.record_start_requested(
        tmp_path, run_id="", task_id="t-rev", idempotency_key="k-extra", invocation_id="inv-extra",
        operation_id="", max_seconds=60, request={"continueFrom": "run-1", "promptCacheTtl": "1h"},
        project_id="", project_owned=False, route="fake-review", surface="review", slot_id="slot-1",
        continuation_of="run-1",
    )
    custody.record_started(
        tmp_path,
        custody.RunCustody(
            run_id="run-extra", task_id="t-rev", route_id="fake-review",
            invocation_id="inv-extra", continuation_of="run-1",
        ),
        shape={"surface": "review", "slot_id": "slot-1"},
    )
    anchor_dict = {
        "run_id": anchor.run_id, "task_id": "t-rev", "root_task_id": "",
        "surface": "review", "slot_id": "slot-1", "operation_id": "",
    }
    assert review_continuation_descends_from(tmp_path, "run-exact", anchor_dict)
    assert not review_continuation_descends_from(tmp_path, "run-extra", anchor_dict)


@pytest.mark.parametrize("malformed", [False, True])
def test_terminal_ttl_requires_engine_evidence_not_host_transmission(tmp_path, malformed):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload
    from tests.test_claudexor_observed_attempt import _write_telemetry

    attempt = {"attempt_id": "a02", "harness_id": "claude"}
    if malformed:
        attempt["prompt_cache_ttl"] = {"requested": "1h"}
    detail = _write_telemetry(tmp_path, [attempt])
    detail["summary"].update(state="failed", failure={"code": "auth_failed"})
    start = {"requested": "1h", "submitted": "1h", "reason": None}
    entry = custody.RunCustody(run_id="run-fixture", task_id="ttl-test", prompt_cache_ttl=start)
    payload = _terminal_payload(
        "run-fixture", detail, DelegatedRunShape("readonly", "ask", "live", False), entry)
    assert payload["state"] == "failed" and payload["failure"]["code"] == "auth_failed"
    assert payload["prompt_cache_ttl"] == {
        "requested": "1h", "submitted": None,
        "reason": "malformed_engine_telemetry" if malformed else "missing_engine_telemetry",
        "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
    }
    assert entry.prompt_cache_ttl == start == {"requested": "1h", "submitted": "1h", "reason": None}


@pytest.mark.parametrize("predecessor,reason", [
    ({"requested": "1h", "submitted": "1h", "reason": None}, "inherited_from_continuation"),
    ({"requested": "1h", "submitted": None, "reason": "inherited_from_continuation"}, "inherited_from_continuation"),
    ({"requested": "1h", "submitted": None, "reason": "capabilities_read_failed"}, "capabilities_read_failed"),
    ({"requested": "1h", "submitted": None, "reason": "harness_unsupported"}, "harness_unsupported"),
    ({"requested": None, "submitted": None, "reason": None}, "inherited_from_continuation"),
])
def test_continuation_inherits_only_a_submitted_ttl_and_keeps_engine_default_receipt(
    tmp_path, monkeypatch, predecessor, reason,
):
    from ouroboros import delegate_custody as custody
    from ouroboros.effort_evidence import session_prompt_cache_ttl_request
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload
    from tests.test_claudexor_observed_attempt import _write_telemetry

    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "5m")  # a bare continuation never re-reads settings
    body = {"continueFrom": "run-pred"}
    facts = session_prompt_cache_ttl_request(body, route_id="claude", continuation_from=custody.RunCustody(
        run_id="run-pred", task_id="ttl-test", prompt_cache_ttl=predecessor))
    assert body == {"continueFrom": "run-pred"}
    assert facts == {"requested": predecessor["requested"], "submitted": None, "reason": reason,
                     "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None}
    # The engine inherits only its accepted request, so a never-submitted 1h leaves a valid Default receipt.
    default = {"requested": None, "submitted": None, "reason": None, "observed": "5m",
               "write_1h_tokens": 0, "write_5m_tokens": 64}
    detail = _write_telemetry(tmp_path, [{"attempt_id": "a02", "harness_id": "claude", "prompt_cache_ttl": default}])
    detail["summary"].update(state="succeeded")
    entry = custody.RunCustody(run_id="run-fixture", task_id="ttl-test", prompt_cache_ttl=facts)
    payload = _terminal_payload("run-fixture", detail, DelegatedRunShape("readonly", "ask", "live", False), entry)
    failed = reason != "inherited_from_continuation"
    assert payload["prompt_cache_ttl"] == ({**default, "requested": "1h", "reason": reason} if failed else default)


@pytest.mark.parametrize("supported", [False, True])
def test_failed_reviewer_keeps_ttl_intent_through_failure_projection(tmp_path, monkeypatch, fake_route, supported):
    from ouroboros.review_execution import SessionInvocation, run_delegated_review_session
    from ouroboros.review_session_usage import session_failure_identity
    from ouroboros.reviewer_slot_config import record_reviewer_slot_executions, reviewer_slot_last_executions

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "canonical-data")
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "1h")
    if supported:
        monkeypatch.setattr(FakeGateway, "agent_capabilities", lambda self: {
            "harnesses": [{**FakeGateway.catalog_entry, "promptCacheTtls": ["5m", "1h"]}],
            "runControlKeys": ["promptCacheTtl"],
        })
    monkeypatch.setattr(FakeGateway, "detail", _terminal_detail("", state="failed"))
    delivery = {}
    slot = _agent_slot()
    inv = SessionInvocation(task_id="ttl-failure", surface="review", slot_id=slot.slot_id,
                            timeout_sec=60, source_delivery=delivery)
    with pytest.raises(Exception) as raised:
        run_delegated_review_session(prompt="review it", root=str(tmp_path), custody_drive=tmp_path, invocation=inv)
    assert raised.value.delegated_run_id == "run-1", "must reach a real terminal, not a preparation error"
    requests = [r for gateway in FakeGateway.instances for r in gateway.start_requests]
    assert len(requests) == 1 and requests[0].get("promptCacheTtl") == ("1h" if supported else None)
    usage = session_failure_identity(raised.value.delegated_usage)
    assert usage["review_source_delivery"] == delivery
    actor = SimpleNamespace(slot_id=slot.slot_id, status="error", usage=usage)
    record_reviewer_slot_executions("review", [actor], {slot.slot_id: slot})
    actual = reviewer_slot_last_executions()[slot.slot_id]["effective"]["prompt_cache_ttl"]
    assert actual == {
        "requested": "1h", "submitted": None,
        "reason": "missing_engine_telemetry" if supported else "engine_capability_missing",
        "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None,
    }



@pytest.mark.usefixtures("forbid_effects")
@pytest.mark.parametrize("receipt,start_reason", [
    ("missing", "harness_unsupported"), ("malformed", "harness_unsupported"),
    ("engine_default", "capabilities_read_failed"), ("missing", None),
])
def test_late_collected_reviewer_keeps_recorded_ttl_intent_without_new_reads(
    tmp_path, monkeypatch, receipt, start_reason,
):
    from ouroboros import delegate_custody as custody
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.reviewer_slot_config import reviewer_slot_execution_rows
    from tests.test_claudexor_observed_attempt import _write_telemetry
    from tests.test_review_operation_collection import PASS, SURFACE, TASK, _collect, _ctx, _observe, _session_run

    default = {"requested": None, "submitted": None, "reason": None, "observed": "5m",
               "write_1h_tokens": 0, "write_5m_tokens": 64}
    attempt = {"attempt_id": "a02", "harness_id": "codex", "observed_model": "codex-model",
               **({"prompt_cache_ttl": {"requested": "1h"}} if receipt == "malformed" else {}),
               **({"prompt_cache_ttl": default} if receipt == "engine_default" else {})}
    detail = _write_telemetry(tmp_path / "run", [attempt], run_id="run-late")
    detail["summary"].update(state="succeeded", outputConformance="")
    detail["primaryOutput"] = {"text": PASS, "truncated": False}
    calls = _observe(monkeypatch, detail)
    start = {"requested": "1h", "submitted": None, "reason": start_reason,
             "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None}
    custody.emit(tmp_path, custody.START_REQUESTED, {
        "invocation_id": "inv-op-session", "task_id": TASK, "operation_id": "op-session", "slot_id": "s",
        "surface": SURFACE, "root_task_id": TASK, "request": {"prompt": "review", "harnesses": ["codex"]},
        **({"prompt_cache_ttl": start} if start_reason else {})})  # a legacy row carries no TTL facts
    custody.emit(tmp_path, custody.STARTED, {"run_id": "run-late", "invocation_id": "inv-op-session",
                                             "task_id": TASK, "surface": SURFACE})
    run = _session_run()
    actor, fact = _collect(run, tmp_path, _ctx(tmp_path))
    assert fact == "collected" and actor["usage"]["collection"] == "attach_only_observation"
    assert calls == [("handshake",), ("get_run", "run-late"), ("close",)], "the recorded facts need no new read"
    roster = run["slot_roster"][0]
    slot = ReviewSlot(**{**roster, "route": ReviewRouteKind(roster["route"])})
    effective = reviewer_slot_execution_rows(SURFACE, [SimpleNamespace(**actor)], {slot.slot_id: slot})["s"]["effective"]
    if start_reason is None:
        assert "prompt_cache_ttl" not in effective, "absent start facts stay unknown, never fabricated"
    else:
        observed = default if receipt == "engine_default" else {
            "observed": "unknown", "write_1h_tokens": None, "write_5m_tokens": None}
        assert effective["prompt_cache_ttl"] == {**observed, "requested": "1h", "submitted": None,
                                                 "reason": start_reason}
