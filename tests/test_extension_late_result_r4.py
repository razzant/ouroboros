"""Small pure checks for late-result tier, closing clock, and remaining-time debit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import extension_late_result as elr
from ouroboros.model_wait import EXECUTION_CLOSE_KEY, execution_close
from ouroboros.owner_continue import recorded_execution_seconds


def _ctx(route: dict) -> SimpleNamespace:
    return SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(_late_report_ref="r1", _late_result_route=route)))


def test_processing_preference_economy_and_default_vs_unknown():
    base = {"model": "openai::gpt-5", "physical_provider": "openai", "physical_model": "openai/gpt-5",
            "dispatch_account": "conn:openai", "use_local": False}
    econ = elr._captured_route(SimpleNamespace(_current_llm_call_meta={**base, "processing_preference": "economy"}))
    default = elr._captured_route(SimpleNamespace(_current_llm_call_meta={**base, "processing_preference": ""}))
    assert econ["processing_preference"] == "economy" and "submitted_processing_mode" not in econ
    assert default["processing_preference"] == "" and "submitted_processing_mode" not in default
    assert elr._captured_route(SimpleNamespace(_current_llm_call_meta=dict(base))) == {}
    assert elr._captured_route(SimpleNamespace(_current_llm_call_meta={**base, "processing_preference": "default"})) == {}
    assert elr.return_processing_preference(_ctx(econ), default="fast") == "economy"
    assert elr.return_processing_preference(_ctx(default), default="fast") == ""
    for bad in (dict(base), {**base, "processing_preference": "default"}):
        with pytest.raises(elr.LateResultUnavailable):
            elr.return_processing_preference(_ctx(bad), default="fast")


def test_execution_close_direct_first_turn_vs_other_attempts():
    first = {"id": "t1", "_is_direct_chat": True, "execution_timeout_sec": 60, "_execution_owner": {"task_id": "t1", "kind": "direct", "task_attempt": 0}}
    closed = execution_close(first, {"attempt": 1, "started_at": 100.0, "task": first}, now=112.5)
    assert closed == {EXECUTION_CLOSE_KEY: {"sec": 12.5, "attempt": 0}}
    assert recorded_execution_seconds({"task_id": "t1", "task_attempt": 0, **closed}) == pytest.approx(12.5)
    assert recorded_execution_seconds({"task_id": "t1", "task_attempt": 1, **closed}) is None
    retry = {"id": "t2", "_is_direct_chat": True, "_attempt": 2, "execution_timeout_sec": 60, "_execution_owner": {"task_id": "t2", "kind": "direct", "task_attempt": 2}}
    assert execution_close(retry, {"attempt": 2, "started_at": 10.0}, now=15.0) == {EXECUTION_CLOSE_KEY: {"sec": 5.0, "attempt": 2}}
    assert execution_close({"id": "p1", "execution_timeout_sec": 60}, {"attempt": 1, "started_at": 20.0}, now=27.0) == {EXECUTION_CLOSE_KEY: {"sec": 7.0, "attempt": 1}}


def test_debit_return_execution_no_fresh_seconds_on_unknown():
    source = {"task_id": "src-1", "status": "completed", elr._SHARED_EXECUTION: {
        "spent": {"src-1": 5.0}, "active": {"task_id": "ret-1", "report_ref": "r1", "scopes": ["src-1"], "seconds": 35.0}}}
    settled = {"task_id": "ret-1", "status": "completed", "task_attempt": 0, "admission_author": {"late_report_ref": "r1"}}
    spent, left = elr.debit_return_execution(source, {"src-1": 40.0}, {**settled, EXECUTION_CLOSE_KEY: {"sec": 12.5, "attempt": 0}})
    assert spent == {"src-1": pytest.approx(17.5)} and left == pytest.approx(22.5)
    with pytest.raises(elr.LateResultUnavailable):
        elr.debit_return_execution(source, {"src-1": 40.0}, settled)
    with pytest.raises(elr.LateResultUnavailable):
        elr.debit_return_execution(source, {"src-1": 40.0}, {**settled, EXECUTION_CLOSE_KEY: {"sec": 35.0, "attempt": 0}})
