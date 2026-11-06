"""Delivered answers, tool-call facts, and acceptance remain separate axes."""

import copy

import pytest

from ouroboros.outcomes import derive_loop_outcome, normalize_outcome_axes
from ouroboros.tool_capabilities import PURE_READ_TOOLS
from tests._cases import plain_rows


TOOL_ERRORS = [
    pytest.param({"tool": "run_command", "args": {"cmd": ["node", "--version"]},
                  "is_error": True, "status": "non_zero_exit", "exit_code": -9,
                  "signal": "SIGKILL", "result": "SHELL_EXIT_ERROR"}, id="signal-death"),
    pytest.param({"tool": "search_code", "args": {}, "is_error": True,
                  "status": "error", "result": "SEARCH_ERROR: query is required"}, id="missing-query"),
    pytest.param({"tool": "run_command", "args": {"cmd": ["false"]}, "is_error": True,
                  "status": "non_zero_exit", "exit_code": 1, "result": "SHELL_EXIT_ERROR"}, id="plain-exit"),
]


def test_tool_errors_do_not_decide_delivery_or_acceptance(subtests):
    for tool_error in plain_rows(TOOL_ERRORS):
        for review in [None, "PASS", "FAIL", "DEGRADED"]:
            for text, usage, extra, execution, reason in [("Delivered answer.", {}, {}, "ok", "final_message"), ("", {}, {}, "failed", "empty_final_text"), ("Retained answer", {"execution_status": "infra_failed", "reason_code": "provider_unavailable"}, {}, "infra_failed", "provider_unavailable"), ("Retained answer", {"execution_status": "failed", "reason_code": "task_exception"}, {}, "failed", "task_exception"), ("Retained answer", {"execution_status": "failed", "reason_code": "deadline_local",
                               "_best_effort_extracted": True}, {}, "best_effort", "deadline_local"), ("Retained answer", {}, {"delivery_candidate": {
   "degraded": True, "degraded_reason": "invalid_delivery_control_after_repair"}}, "degraded", "invalid_delivery_control_after_repair"), ("Retained answer", {}, {"child_result_dispositions": {"deferred_count": 1}}, "degraded", "child_results_deferred"), ("Retained answer", {}, {"verification_events": [{"services": [
   {"name": "result", "artifact_output_failed": True}]}]}, "degraded", "tool_failure")]:
                with subtests.test(tool_error=tool_error, review=review, text=text, usage=usage, extra=extra, execution=execution, reason=reason):
                    trace = {"tool_calls": [copy.deepcopy(tool_error)], **copy.deepcopy(extra)}
                    if review:
                        trace.update(review_runs=[{"authority": "host_root", "aggregate_signal": review}],
                                     review_decision={"eligibility": "eligible", "trigger": "host_root_effects"})
                    outcome = derive_loop_outcome(text, usage, trace)
                    axes = outcome["outcome_axes"]
                    assert axes["execution"]["status"] == execution
                    assert axes["execution"]["reason_code"] == reason
                    judged = {None: "not_evaluated", "PASS": "pass", "FAIL": "fail", "DEGRADED": "degraded"}[review]
                    if "child_result_dispositions" in extra and review != "FAIL":
                        judged = "best_effort"
                    elif "delivery_candidate" in extra and review is None:
                        judged = "degraded"
                    assert axes["objective"]["status"] == judged
                    # A failed read stays in the details (owner 06.11, A); a command warns when nothing judged it.
                    loud = tool_error["tool"] not in PURE_READ_TOOLS
                    assert bool(axes["objective"].get("warning")) == (judged == "not_evaluated" and loud)
                    errors = axes["execution"]["unresolved_tool_errors"] or axes["execution"]["cosmetic_tool_errors"]
                    assert errors[0]["status"] == tool_error["status"]
                    if tool_error.get("signal"):
                        assert errors[0]["signal"] == "SIGKILL"
                        assert errors[0]["exit_code"] == -9
                        assert not axes["execution"]["cosmetic_tool_errors"]
                    if execution == "ok":
                        assert outcome["failure"] is None
                    if review and judged == review.lower():
                        assert axes["objective"]["source"] == "task_acceptance_review"


def test_advisory_review_cannot_clear_unreviewed_tool_error_warning():
    trace = {"tool_calls": [{"tool": "write_file", "is_error": True, "status": "error"}],
             "review_runs": [{"authority": "agent_advisory", "aggregate_signal": "PASS"}]}
    axes = derive_loop_outcome("Delivered answer.", {}, trace)["outcome_axes"]
    assert axes["execution"]["status"] == "ok"
    assert axes["execution"]["unresolved_tool_errors"]
    assert axes["objective"]["status"] == "not_evaluated"
    assert axes["objective"]["source"] == "none"
    assert axes["objective"]["warning"] == "residual_tool_errors_without_review"


@pytest.mark.parametrize(("call", "loud"), [
    pytest.param({"tool": "read_file", "status": "argument_error"}, False, id="failed-read"),
    pytest.param({"tool": "search_code", "status": "error"}, False, id="failed-search"),
    pytest.param({"tool": "browse_page", "status": "error"}, False, id="failed-page-read"),
    pytest.param({"tool": "github_read", "status": "error"}, False, id="failed-github-read"),
    pytest.param({"tool": "write_file", "status": "error"}, True, id="unfulfilled-write"),
    pytest.param({"tool": "run_command", "status": "non_zero_exit", "exit_code": 1}, True, id="failed-command"),
    pytest.param({"tool": "mcp_gitlab__get_commit", "status": "mcp_error"}, True, id="unknown-kind"),
])
def test_only_an_unfulfilled_effect_a_failed_command_or_an_unknown_tool_warns_without_review(call, loud):
    """Owner 06.11 («A — изменения и команды»): with no review, Done turns into Done with warnings only for an
    effect or command that failed or a tool of unknown kind; a failed read or search is still recorded, in the details."""
    axes = derive_loop_outcome("Delivered answer.", {}, {"tool_calls": [{**call, "is_error": True, "args": {}}]})["outcome_axes"]
    assert axes["objective"]["status"] == "not_evaluated" and axes["execution"]["status"] == "ok"
    assert bool(axes["objective"].get("warning")) == loud
    stored = normalize_outcome_axes({"status": "completed", "outcome_axes": axes})  # a stored result read back
    assert bool(stored["objective"].get("warning")) == loud
    [error] = axes["execution"]["unresolved_tool_errors"] or axes["execution"]["cosmetic_tool_errors"]
    assert error["tool"] == call["tool"] and bool(error.get("read_only")) == (not loud)


@pytest.mark.parametrize(("extra", "execution", "reason", "exit_code", "signal", "bucket"), [
    pytest.param({}, "ok", "final_message", -9, "SIGKILL", "unresolved_tool_errors", id='extra0-ok-final_message--9-SIGKILL-unresolved_tool_errors'),
    pytest.param({"delivery_candidate": {"degraded": True, "degraded_reason": "invalid_delivery_control_after_repair"}}, "degraded", "invalid_delivery_control_after_repair", 1, "", "cosmetic_tool_errors", id='extra1-degraded-invalid_delivery_control_after_repair-1--cosmetic_tool_errors'),
    pytest.param({"child_result_dispositions": {"deferred_count": 1}}, "degraded", "child_results_deferred", -9, "SIGKILL", "unresolved_tool_errors", id='extra2-degraded-child_results_deferred--9-SIGKILL-unresolved_tool_errors'),
])
def test_persisted_unjudged_objective_keeps_warning_after_normalization(
    tmp_path, exit_code, signal, bucket, extra, execution, reason,
):
    from types import SimpleNamespace

    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import load_task_result

    call = {"tool": "run_command", "args": {"cmd": ["false"]}, "is_error": True,
            "status": "non_zero_exit", "exit_code": exit_code, "result": "SHELL_EXIT_ERROR"}
    if signal:
        call["signal"] = signal
    _store_task_result(
        env=SimpleNamespace(drive_root=tmp_path),
        task={"id": "normalized-warning", "type": "task", "text": "Produce an answer"},
        text="Retained answer.", usage={"rounds": 2, "cost": 0},
        llm_trace={"tool_calls": [call], **copy.deepcopy(extra)}, review_evidence={},
    )
    saved = load_task_result(tmp_path, "normalized-warning")
    original = copy.deepcopy(saved)
    public = public_task_result(saved)
    for record in (saved, public, public_task_result(public)):
        axes = record["outcome_axes"]
        assert axes["objective"]["status"] == "not_evaluated"
        assert axes["objective"]["warnings"].count("residual_tool_errors_without_review") == 1
        assert axes["execution"]["status"] == execution
        assert axes["execution"]["reason_code"] == reason
        assert axes["execution"][bucket][0]["exit_code"] == exit_code
    assert saved == original


@pytest.mark.parametrize("verdict", ["pass", "fail", "degraded"])
def test_normalization_does_not_warn_over_authoritative_acceptance(verdict):
    from ouroboros.outcomes import normalize_outcome_axes

    record = {"status": "completed", "outcome_axes": {
        "objective": {"status": verdict, "source": "task_acceptance_review"},
        "execution": {"status": "ok", "unresolved_tool_errors": [{"exit_code": -9}]},
    }}
    original = copy.deepcopy(record)
    assert normalize_outcome_axes(record)["objective"] == original["outcome_axes"]["objective"]
    assert record == original
