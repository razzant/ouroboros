"""#2136: native run totals reach events, not another charge or a physical request."""

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.review_session_usage import session_usage_facts
from tests._usage_store_testing import ledger_rows
from tests._review_session_route_shared import (
    FakeLLM, _agent_request, _agent_slot, _terminal_detail,
    fake_route as fake_route,
    _owned_gateway_uses_each_test_transport as _transport_fixture,
)

_owned_gateway_uses_each_test_transport = _transport_fixture
pytestmark = pytest.mark.serial
SPLIT = {"total_tokens": 20488, "cache_read_tokens": 12738, "cache_write_tokens": 7746}


@pytest.fixture(autouse=True)
def _synthetic_root_lineage(tmp_path):
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "root", "running", root_task_id="root")


def events(root):
    path = root / "logs/events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def terminal(split=SPLIT, state="succeeded"):
    # Legacy axes deliberately disagree with the normalized inclusive total.
    return {"summary": {"state": state, "inputTokens": 4, "cachedInputTokens": 20484,
                        "outputTokens": 142, "inputTokenUsage": copy.deepcopy(split),
                        "spendUsd": 0.0}, "lastSeq": 3}


def entry(root, run_id="run-input", source="delegated_subagent"):
    return custody.RunCustody(run_id=run_id, task_id="task", root_task_id="root",
                              parent_task_id="parent", route_id="claude", source=source,
                              ledger_root=str(root))


def collect(root, run=None, detail=None):
    from ouroboros.delegate_collection import record_terminal_money
    run = run or entry(root)
    return run, record_terminal_money(root, run, detail or terminal())


def persist_review(root, usage, model="last-observed-model"):
    from ouroboros.tools.review_helpers import emit_review_usage
    from supervisor.events_budget import _handle_llm_usage
    ctx = SimpleNamespace(task_id="review", pending_events=[])
    emit_review_usage(ctx, model=model, usage=usage, source="scope_review")
    host = SimpleNamespace(DRIVE_ROOT=root)
    for row in ctx.pending_events:
        _handle_llm_usage(row, host)
    return events(root)[-1]


def test_ordinary_collection_emits_run_split_once_without_repricing(tmp_path, monkeypatch):
    from ouroboros.gateways import claudexor
    observed = {"attempt_id": "a2", "harness_id": "claude", "model": "final-model", "profile_id": "last-account"}
    monkeypatch.setattr(claudexor, "final_attempt_facts", lambda *_: observed)
    run, settled = collect(tmp_path)
    assert settled["ledger_recorded"] and settled["settled"]
    usage = [row for row in events(tmp_path) if row["type"] == "llm_usage"]
    assert len(usage) == 1
    row = usage[0]
    assert row["input_token_usage"] == SPLIT
    assert row["claudexor"]["run_id"] == run.run_id
    assert row["claudexor"]["usage_scope"] == "run"
    assert row["claudexor"]["observed_attempt"] == observed
    assert row["model"] == ""  # run totals are not the final account/model's own request
    assert row["task_id"] == "task" and row["root_task_id"] == "root"
    assert row["parent_task_id"] == "parent"
    assert row["accounting_authority"] == "physical_attempt_ledger"
    before = ledger_rows(tmp_path)
    collect(tmp_path, run)
    # A fresh collector after a crash can repeat telemetry; it cannot rewrite money.
    collect(tmp_path, entry(tmp_path))
    assert ledger_rows(tmp_path) == before and len(before) == 1
    assert before[0]["input_token_usage"] == SPLIT
    # The ledger's input axes are the split (a direct call's meaning); the event keeps the harness's own.
    assert (before[0]["prompt_tokens"], before[0]["cached_tokens"], before[0]["cache_write_tokens"]) == (20488, 12738, 7746)
    assert before[0]["harness_tokens"] == {"prompt_tokens": 4, "cached_tokens": 20484}
    assert (row["prompt_tokens"], row["cached_tokens"]) == (4, 20484)
    assert before[0]["cost_usd"] == 0.0
    assert {row["ledger_attempt_ids"][0] for row in events(tmp_path) if row["type"] == "llm_usage"} == {before[0]["attempt_id"]}


@pytest.mark.parametrize("split", [SPLIT, dict.fromkeys(SPLIT), {**SPLIT, "cache_write_tokens": None}])
def test_review_split_survives_both_emit_and_persistence_filters(tmp_path, split):
    facts = session_usage_facts("run-review", requested_route="claude", summary=terminal(split)["summary"])
    assert facts["input_token_usage"] == split
    row = persist_review(tmp_path, facts)
    assert row["input_token_usage"] == split
    assert row["model"] == "" and row["claudexor"]["usage_scope"] == "run"
    assert row["prompt_tokens"] == 4 and row["cached_tokens"] == 20484
    assert row["record_contract_version"] == 2


@pytest.mark.parametrize("split", [None, {}, {"total_tokens": 3}, {**SPLIT, "extra": 1},
                                  {**SPLIT, "cache_read_tokens": True}, {**SPLIT, "total_tokens": -1},
                                  {**SPLIT, "total_tokens": 1.2}, {**SPLIT, "total_tokens": "20488"}])
def test_invalid_split_is_not_repaired_or_derived_from_legacy(tmp_path, split):
    facts = session_usage_facts("run-review", summary=terminal(split)["summary"])
    assert "input_token_usage" not in facts
    row = persist_review(tmp_path, {**facts, "input_token_usage": split})
    assert "input_token_usage" not in row
    collect(tmp_path, detail=terminal(split))
    ordinary = [row for row in events(tmp_path) if row["type"] == "llm_usage" and row["category"] != "review"]
    assert not ordinary  # the existing custody event already discloses the unknown session
    assert "input_token_usage" not in ledger_rows(tmp_path)[0]


def test_supervisor_validates_its_direct_usage_input(tmp_path):
    from supervisor.events_budget import _handle_llm_usage
    host = SimpleNamespace(DRIVE_ROOT=tmp_path)
    _handle_llm_usage({"type": "llm_usage", "usage": {"input_token_usage": {"total_tokens": 3}}}, host)
    assert "input_token_usage" not in events(tmp_path)[0]


def test_no_new_ordinary_event_for_review_owned_or_unstarted_or_live_run(tmp_path):
    collect(tmp_path, entry(tmp_path, source="review_substrate"))
    assert not [row for row in events(tmp_path) if row["type"] == "llm_usage"]
    with pytest.raises(ValueError, match="live execution"):
        collect(tmp_path, entry(tmp_path, "run-live"), terminal(state="running"))
    assert len(ledger_rows(tmp_path)) == 1


def test_failed_but_started_run_still_emits_disclosed_split(tmp_path):
    run, result = collect(tmp_path, detail=terminal(state="failed"))
    assert result["settled"]
    assert next(row for row in events(tmp_path) if row["type"] == "llm_usage")["input_token_usage"] == SPLIT


def test_event_failure_never_repeats_money_or_blocks_settlement(tmp_path, monkeypatch, caplog):
    import ouroboros.utils as utils
    original = utils.append_jsonl
    def fail_usage(path, row, *args, **kwargs):
        if row.get("type") == "llm_usage":
            raise OSError("event write unavailable")
        return original(path, row, *args, **kwargs)
    monkeypatch.setattr(utils, "append_jsonl", fail_usage)
    run, result = collect(tmp_path)
    assert result["settled"] and result["ledger_recorded"]
    before = ledger_rows(tmp_path)
    collect(tmp_path, run)
    assert ledger_rows(tmp_path) == before
    assert "usage event" in caplog.text


def test_ledger_failure_never_emits_usage_or_claims_settled(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    def fail(*args, **kwargs):
        raise OSError("ledger unavailable")
    monkeypatch.setattr(usage_accounting, "record_subscription_session", fail)
    _, result = collect(tmp_path)
    assert not result["ledger_recorded"] and not result["settled"]
    assert not [row for row in events(tmp_path) if row["type"] == "llm_usage"]


def test_real_review_route_keeps_session_extraction_and_collection_separate(tmp_path, monkeypatch, fake_route):
    from ouroboros import delegate_custody_usage
    from ouroboros.review_substrate import run_review_request
    from supervisor.events_budget import _handle_llm_usage
    monkeypatch.setattr(delegate_custody_usage, "_EMITTED_SESSION_USAGE", {})
    fake_route.detail = _terminal_detail("The review is complete; I found no problems.")
    fake_route.detail["summary"].update(terminal()["summary"])
    ctx = SimpleNamespace(task_id="t-agent", pending_events=[])
    llm = FakeLLM()
    result = run_review_request(_agent_request(), slots=[_agent_slot()], drive_root=tmp_path,
                                llm=llm, usage_ctx=ctx)
    rows = [row for row in ctx.pending_events if row["type"] == "llm_usage"]
    assert len(rows) == 1 and len(llm.calls) == 1
    for row in rows:
        _handle_llm_usage(row, SimpleNamespace(DRIVE_ROOT=tmp_path))
    session_rows = [row for row in events(tmp_path) if row["type"] == "llm_usage"]
    assert len(session_rows) == 1  # custody must NOT emit the same reviewer again
    assert session_rows[0]["input_token_usage"] == SPLIT
    assert session_rows[0]["model"] == ""
    assert session_rows[0]["claudexor"]["observed_attempt"]["model"] == "fake-small"
    assert result.actors[0]["usage"]["extraction"]["prompt_tokens"] == 5
    assert result.actors[0]["usage"]["input_token_usage"] == SPLIT


def test_review_chain_observation_is_per_run_not_final_chain_sum(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_usage as dcu
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment, _combined_session_usage
    monkeypatch.setattr(dcu, "_EMITTED_SESSION_USAGE", {})
    # The fold reads engine run details (summary and attempts), as review_execution passes them.
    first_detail, final_detail = terminal(), terminal({**SPLIT, "total_tokens": None})
    first, final = first_detail["summary"], final_detail["summary"]
    _, _, combined = _combined_session_usage(custody, [first_detail], final_detail)
    assert combined["inputTokens"] == 8  # existing actor aggregate stays unchanged
    assert "inputTokenUsage" not in combined  # never attach final-run split to chain totals
    parts = [session_usage_facts(run, requested_route="claude", summary=value)
             for run, value in (("run-first", first), ("run-final", final))]
    executor = AgentSessionReviewExecutor(ReviewAssignment(_agent_request(), _agent_slot(), custody_root=tmp_path))
    def fake_session():
        executor._run_id = "run-final"
        executor._session_usage = {**session_usage_facts("run-final", summary=combined),
                                   "session_usage_events": parts}
    monkeypatch.setattr(executor, "_run_session", fake_session)
    monkeypatch.setattr(executor, "_verdict_result", lambda **_: "verdict")
    rows = []
    executor.usage_observer = rows.append
    assert executor.execute() == "verdict"
    assert [row["claudexor"]["run_id"] for row in rows] == ["run-first", "run-final"]
    assert [row["input_token_usage"] for row in rows] == [SPLIT, final["inputTokenUsage"]]
    for row in rows:
        persist_review(tmp_path, row)
    assert [row["input_token_usage"] for row in events(tmp_path)] == [SPLIT, final["inputTokenUsage"]]
    executor.execute()
    assert len(rows) == 2  # repeated extraction/collection never re-emits a native run


@pytest.mark.parametrize("final_failed", [False, True])
def test_actual_review_continuations_emit_each_run_once(tmp_path, monkeypatch, fake_route, final_failed):
    from ouroboros import delegate_custody_usage as dcu
    from ouroboros.review_substrate import run_review_request
    from tests.test_review_session_continuation import _engine
    from tests._review_session_route_shared import FakeGateway
    from supervisor.events_budget import _handle_llm_usage
    monkeypatch.setattr(dcu, "_EMITTED_SESSION_USAGE", {})
    accepted, posts = _engine(monkeypatch, final_failed=final_failed)
    get_run = FakeGateway.get_run
    splits = {"run-1": SPLIT, "run-2": {**SPLIT, "cache_write_tokens": None}}
    def detail(self, run_id, **kwargs):
        result = get_run(self, run_id, **kwargs)
        result["summary"].update(inputTokenUsage=splits[run_id], inputTokens=4,
                                  outputTokens=142, cachedInputTokens=20484)
        return result
    monkeypatch.setattr(FakeGateway, "get_run", detail)
    ctx = SimpleNamespace(task_id="t-review", pending_events=[])
    result = run_review_request(_agent_request(session_root=str(tmp_path), task_id="t-review", retry_key="split-chain"),
                                slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx)
    usage = [row for row in ctx.pending_events if row["type"] == "llm_usage"]
    assert len(accepted) == len(posts) == 1
    assert [row["usage"]["claudexor"]["run_id"] for row in usage] == ["run-1", "run-2"]
    assert [row["usage"]["input_token_usage"] for row in usage] == list(splits.values())
    for row in usage:
        _handle_llm_usage(row, SimpleNamespace(DRIVE_ROOT=tmp_path))
    assert len([row for row in events(tmp_path) if row["type"] == "llm_usage"]) == 2
    assert len([row for row in ledger_rows(tmp_path) if row["kind"] == "subscription_session"]) == 2
    if not final_failed:
        assert result.actors[0]["usage"]["prompt_tokens"] == 8
        assert "input_token_usage" not in result.actors[0]["usage"]
