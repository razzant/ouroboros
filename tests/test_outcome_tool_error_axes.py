"""Delivered answers, tool-call facts, and acceptance remain separate axes."""

import copy

import pytest

from ouroboros.outcomes import derive_loop_outcome, normalize_outcome_axes
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
                    # A failed read (owner 06.11, A) and a command the task went past (owner 09.11, A) stay in the
                    # details; a signal-killed command warns when nothing judged the result.
                    loud = bool(tool_error.get("signal"))
                    assert bool(axes["objective"].get("warning")) == (judged == "not_evaluated" and loud)
                    errors = axes["execution"]["unresolved_tool_errors"] or axes["execution"]["cosmetic_tool_errors"]
                    assert errors[0]["status"] == tool_error["status"]
                    assert bool(errors[0].get("read_only") or errors[0].get("details_only")) == (not loud)
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


@pytest.mark.parametrize(("call", "loud", "flag"), [
    pytest.param({"tool": "read_file", "status": "argument_error"}, False, "read_only", id="failed-read"),
    pytest.param({"tool": "search_code", "status": "error"}, False, "read_only", id="failed-search"),
    pytest.param({"tool": "browse_page", "status": "error"}, False, "read_only", id="failed-page-read"),
    pytest.param({"tool": "github_read", "status": "error"}, False, "read_only", id="failed-github-read"),
    pytest.param({"tool": "write_file", "status": "error"}, True, "", id="unfulfilled-write"),
    pytest.param({"tool": "github_write", "status": "safety_violation"}, True, "", id="send-stopped-by-safety"),
    pytest.param({"tool": "run_command", "status": "non_zero_exit", "exit_code": 1}, False, "details_only", id="failed-command"),
    pytest.param({"tool": "run_script", "status": "timeout"}, False, "details_only", id="timed-out-script"),
    pytest.param({"tool": "run_command", "status": "safety_violation"}, True, "", id="command-stopped-by-safety"),
    pytest.param({"tool": "run_script", "status": "safety_violation"}, True, "", id="script-stopped-by-safety"),
    pytest.param({"tool": "run_script", "status": "run_script_error"}, True, "", id="script-run-error"),
    pytest.param({"tool": "run_command", "status": "artifact_output_error"}, True, "", id="command-output-unregistered"),
    pytest.param({"tool": "run_command", "status": "non_zero_exit", "exit_code": -9, "signal": "SIGKILL"}, True, "",
                 id="signal-killed-command"),
    pytest.param({"tool": "edit_batch", "status": "argument_error"}, False, "details_only", id="rejected-write-call"),
    pytest.param({"tool": "enable_tools", "status": "unknown_tool"}, False, "details_only", id="call-named-no-tool"),
    pytest.param({"tool": "mcp_gitlab__get_commit", "status": "mcp_error"}, True, "", id="unknown-kind"),
    pytest.param({"tool": "skill_exec", "status": "error"}, True, "", id="unknown-builtin-kind-that-ran"),
    pytest.param({"tool": "github_write", "status": "safety_returned"}, False, "", id="write-safety-returned"),
    pytest.param({"tool": "run_command", "status": "safety_unavailable"}, False, "", id="command-safety-could-not-read"),
])
def test_only_a_loud_leftover_warns_without_review(call, loud, flag):
    """Owner 06.11 («A — изменения и команды») revised 09.11 («Q1: A»): with no review, Done turns into Done with
    warnings only for a loud leftover — an effect that did not land, a send or command Safety stopped, a signal-killed
    command, a command whose declared output never registered, or a tool of unknown kind that ran and failed. A failed
    read (``read_only``), a command that ran and failed or timed out and a call the tool refused before running
    (``details_only``) are still recorded, in the details."""
    axes = derive_loop_outcome("Delivered answer.", {}, {"tool_calls": [{**call, "is_error": True, "args": {}}]})["outcome_axes"]
    assert axes["objective"]["status"] == "not_evaluated" and axes["execution"]["status"] == "ok"
    assert bool(axes["objective"].get("warning")) == loud
    stored = normalize_outcome_axes({"status": "completed", "outcome_axes": axes})  # a stored result read back
    assert bool(stored["objective"].get("warning")) == loud
    [error] = axes["execution"]["unresolved_tool_errors"] or axes["execution"]["cosmetic_tool_errors"]
    assert error["tool"] == call["tool"]
    assert {key for key in ("read_only", "details_only") if error.get(key)} == ({flag} if flag else set())


@pytest.mark.parametrize(("call", "clause"), [
    ({"tool": "github_write", "status": "safety_returned"}, "A GitHub write was returned by the Safety Supervisor."),
    ({"tool": "run_command", "status": "safety_unavailable"},
     "A command was not run: the Safety Supervisor could not read it."),
])
def test_a_call_safety_held_is_no_failure_and_stays_named_on_done(call, clause):
    """A call Safety returned or could not read counts as no tool failure and alone never makes a task Done with
    warnings; while nothing replaced it, the Done card names it as held. A later call through its tool replaces it,
    and an old record's ``safety_violation`` stays a loud failure."""
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase

    def card(calls):
        trace = {"tool_calls": [{"args": {}, **item} for item in calls]}
        outcome = derive_loop_outcome("Delivered answer.", {}, trace)
        record = {"status": "completed", "reason_code": "final_message", "outcome_axes": outcome["outcome_axes"]}
        ledger = build_verification_ledger(task={}, loop_outcome=outcome, llm_trace=trace, artifact_bundle={})
        return outcome_phase(record, {}), _completion_verdict(record, {}), ledger["summary"]["has_failures"], outcome

    held = {**call, "is_error": True}
    phase, line, failed, outcome = card([held])
    assert (phase, line, failed, outcome["failure"]) == ("done", clause, False, None)
    assert not outcome["outcome_axes"]["objective"].get("warning")
    assert card([held, {"tool": call["tool"], "status": "ok"}])[:3] == ("done", "", False)
    phase, line, failed, _ = card([{**held, "status": "safety_violation"}])
    assert phase == "warn" and "returned" not in line and failed


def test_a_record_written_before_the_flags_stays_loud():
    """A stored error without ``read_only`` or ``details_only`` (written before either flag) keeps the warning: the
    presentation never guesses that an old command error was harmless."""
    from ouroboros.outcomes import normalize_outcome_axes

    legacy = {"status": "completed", "outcome_axes": {
        "execution": {"status": "ok", "cosmetic_tool_errors": [{"tool": "run_command", "status": "non_zero_exit", "exit_code": 1}]},
        "objective": {"status": "not_evaluated", "source": "none"},
    }}
    assert normalize_outcome_axes(legacy)["objective"]["warning"] == "residual_tool_errors_without_review"


@pytest.mark.parametrize(("extra", "execution", "reason", "exit_code", "signal", "bucket"), [
    pytest.param({}, "ok", "final_message", -9, "SIGKILL", "unresolved_tool_errors", id='extra0-ok-final_message--9-SIGKILL-unresolved_tool_errors'),
    pytest.param({"delivery_candidate": {"degraded": True, "degraded_reason": "invalid_delivery_control_after_repair"}}, "degraded", "invalid_delivery_control_after_repair", -9, "SIGKILL", "unresolved_tool_errors", id='extra1-degraded-invalid_delivery_control_after_repair--9-SIGKILL-unresolved_tool_errors'),
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
