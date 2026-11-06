"""Q1A capability preflight (2026-08-10 amendments): a harness dispatch whose
materialized toolset does not carry the delegate verbs must never pay for a
single LLM round under a dispatch record that says "harness".

The audited incident (task e9108a09c6574184, "slime games research PDF"): nine
children recorded effective_executor=harness / executor_route=codex, made ZERO
delegated runs and burned ~$29-54 of metered API, while their results said
execution=ok and capability_delta.reduced=false. The Phase A allowlist widening
closed one instance of the class; this preflight closes the class at the last
zero-cost moment — toolset materialization. Only a configured session row
dispatches onto the harness, so every failed check is the pin's typed block.
"""

from types import SimpleNamespace

from ouroboros.agent import (
    _blocked_executor_terminal,
    executor_blocked_outcome,
    preflight_delegate_visibility,
)
from ouroboros.subagents import (
    CapabilityDelta,
    SubagentDispatch,
    SubagentExecutorResolution,
    SubagentLaneResolution,
)


def _dispatch(requested_executor="harness", executor="harness"):
    lane = SubagentLaneResolution(
        requested_lane="auto", effective_lane="main", model="test-model",
        resolved_from="main",
    )
    delta = CapabilityDelta(
        requested_lane="auto", resolved_lane="main", effective_lane="main",
        derived_effort="low", effective_effort="low",
        requested_executor=requested_executor, effective_executor=executor,
        reason="", reduced=False,
    )
    resolution = SubagentExecutorResolution(
        requested=requested_executor, executor=executor, route=None,
        reason="harness_ready" if executor == "harness" else "requested_native",
    )
    return SubagentDispatch(
        lane=lane, effort="low", executor=executor,
        route="claude" if executor == "harness" else "",
        profile="local_readonly_subagent", delta=delta,
        executor_resolution=resolution,
    )


def _task(requested_executor="harness"):
    return {
        "id": "child-1",
        "delegation_role": "subagent",
        "requested_executor": requested_executor,
        "effective_executor": "harness",
        "executor_route": "claude",
    }


def _tools(available):
    return SimpleNamespace(available_tools=lambda: list(available))


def test_visible_verbs_leave_the_dispatch_untouched():
    dispatch = _dispatch()
    task = _task()
    result, amended = preflight_delegate_visibility(
        _tools(["delegate_start", "delegate_wait", "delegate_cancel", "read_file"]),
        task, dispatch)
    assert result is dispatch
    assert amended is False
    assert task["executor_route"] == "claude"  # nothing re-stamped


def test_partial_verb_set_is_still_broken():
    # A child that can start a run but not wait on it cannot honor the contract.
    dispatch = _dispatch()
    task = _task()
    result, amended = preflight_delegate_visibility(
        _tools(["delegate_start", "delegate_cancel", "read_file"]), task, dispatch)
    assert amended is True
    assert result.executor == "blocked"
    assert "delegate_tools_invisible" in task["capability_delta"]["reason"]


def test_explicit_harness_pin_with_invisible_verbs_blocks_with_zero_spend():
    dispatch = _dispatch(requested_executor="harness")
    task = _task(requested_executor="harness")
    result, amended = preflight_delegate_visibility(
        _tools(["read_file"]), task, dispatch)
    assert amended is True
    assert result.blocked is True
    # The recorded dispatch fields and the typed delta no longer say "harness".
    assert task["effective_executor"] == "blocked" and task["executor_route"] == ""
    assert task["subagent_envelope"]["effective_executor"] == "blocked"
    assert task["capability_delta"]["reduced"] is True
    # The existing blocked terminal carries the distinct typed reason.
    text, usage = executor_blocked_outcome(result.executor_resolution)
    assert usage["reason_code"] == "delegate_tools_invisible"
    assert usage["execution_status"] == "infra_failed"
    assert "not visible" in text
    assert "NOT run and spent nothing" in text
    # And the cap_info seam _prepare_task_context feeds rebuilds the same outcome.
    cap_info = {
        "executor_blocked_reason": result.executor_resolution.reason,
        "executor_blocked_requested": result.executor_resolution.requested,
        "executor_blocked_reset_at": result.executor_resolution.reset_at,
    }
    terminal_text, terminal_usage, _trace = _blocked_executor_terminal(cap_info)
    assert terminal_usage["reason_code"] == "delegate_tools_invisible"
    assert "delegate_start" in terminal_text


def _broken_tools():
    def _boom():
        raise RuntimeError("registry exploded")
    return SimpleNamespace(available_tools=_boom)


def test_broken_introspection_with_pinned_harness_fails_closed():
    # A probe that cannot prove visibility cannot prove the pinned contract is
    # executable: the typed blocked outcome, zero spend — under the HONEST
    # reason (visibility is unknown, not disproven).
    dispatch = _dispatch(requested_executor="harness")
    task = _task(requested_executor="harness")
    result, amended = preflight_delegate_visibility(_broken_tools(), task, dispatch)
    assert amended is True
    assert result.blocked is True
    assert task["effective_executor"] == "blocked"
    text, usage = executor_blocked_outcome(result.executor_resolution)
    assert usage["reason_code"] == "delegate_visibility_unverified"
    assert "could not be verified" in text
    assert "NOT run and spent nothing" in text


def test_native_and_undispatched_children_are_not_probed():
    probe_forbidden = SimpleNamespace(
        available_tools=lambda: (_ for _ in ()).throw(AssertionError("must not probe")))
    native = _dispatch(executor="native")
    assert preflight_delegate_visibility(probe_forbidden, _task(), native) == (native, False)
    assert preflight_delegate_visibility(probe_forbidden, _task(), None) == (None, False)
