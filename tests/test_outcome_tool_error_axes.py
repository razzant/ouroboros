"""Delivered answers, tool-call facts, and acceptance remain separate axes."""

import copy

import pytest

from ouroboros.outcomes import derive_loop_outcome, normalize_outcome_axes
from tests._cases import plain_rows
from tests._safety_fixtures import supervisor as _supervisor
from tests.test_place_guard import REAL_CALL_PLACE_FACTS, REAL_EFFECT_OF, places as _places
from tests.test_tool_effects import published_tool as _published_tool

places, published_tool, supervisor = _places, _published_tool, _supervisor


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
    pytest.param({"tool": "run_command", "status": "safety_unavailable"}, False, "",
                 id="command-safety-could-not-read"),
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


@pytest.mark.parametrize("call", [{"tool": "github_write", "status": "safety_returned"},
                                  {"tool": "run_command", "status": "safety_unavailable"}])
def test_a_call_safety_held_is_no_failure_and_no_host_sentence_names_it(call):
    """A call Safety returned or could not read counts as no tool failure and alone never makes a task Done with
    warnings, and no host sentence names it, with review details or without: only the open Done card does (owner
    11.11, A), while a real warning beside it keeps its own sentence. The record keeps it unresolved while its
    identical call (same tool and target) has not got past Safety. That call replaces it once it got past Safety and
    then succeeded or failed in its handler, a failure being recorded as itself; a refusal the registry returns before
    the Safety check (or a status a handler shares with one) and another call through the tool leave it held, and an
    old record's ``safety_violation`` stays a loud failure."""
    from ouroboros.outcomes import build_verification_ledger
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase

    def card(calls):
        trace = {"tool_calls": [{"args": {}, **item} for item in calls]}
        outcome = derive_loop_outcome("Delivered answer.", {}, trace)
        record = {"status": "completed", "reason_code": "final_message", "outcome_axes": outcome["outcome_axes"]}
        ledger = build_verification_ledger(task={}, loop_outcome=outcome, llm_trace=trace, artifact_bundle={})
        lines = [_completion_verdict(record, {}, include_review_details=details) for details in (True, False)]
        assert not [line for line in lines if "Safety Supervisor" in line], lines
        return outcome_phase(record, {}), lines[0], ledger["summary"]["has_failures"], outcome

    def left(outcome):
        execution = outcome["outcome_axes"]["execution"]
        return [e["status"] for e in execution["unresolved_tool_errors"] + execution["cosmetic_tool_errors"]]
    held = {**call, "is_error": True}
    phase, line, failed, outcome = card([held])
    assert (phase, line, failed, outcome["failure"], left(outcome)) == ("done", "", False, None, [call["status"]])
    assert not outcome["outcome_axes"]["objective"].get("warning")
    phase, line, failed, outcome = card([held, {"tool": "write_file", "status": "error", "is_error": True}])
    assert (phase, line, failed, left(outcome)) == ("warn", "A file write failed.", True, [call["status"], "error"])
    assert card([held, {"tool": call["tool"], "status": "ok"}])[:3] == ("done", "", False)
    phase, line, failed, _ = card([{**held, "status": "safety_violation"}])
    assert phase == "warn" and "returned" not in line and failed
    probe = {"tool": "run_command", "args": {"cmd": ["echo", "sample"]}}
    command = {**probe, "status": call["status"], "is_error": True}
    assert left(card([command, {**probe, "status": "ok"}])[3]) == []  # its identical call got past Safety
    assert left(card([command, dict(command)])[3]) == [call["status"]] * 2  # held again: neither got past it
    # These results measure a process, not just that the caller stopped waiting for a tool.
    for failure in ({"status": "non_zero_exit", "exit_code": 1}, {"status": "timeout", "timed_out": True},
                    {"status": "timeout", "exit_code": 1}):
        phase, line, failed, outcome = card([command, {**probe, **failure, "is_error": True}])
        assert (line, failed, left(outcome)) == ("", True, [failure["status"]])
        [recovered] = outcome["outcome_axes"]["execution"]["recoveries"]
        assert recovered["recovered_by_call_index"] == 2
    for refused in ("resource_constraint_blocked", "argument_error", "light_mode_blocked", "unavailable", "cwd_blocked",
                    "error"):  # refused before the Safety check, or a status a handler shares with such a refusal
        assert left(card([command, {**probe, "status": refused, "is_error": True}])[3])[0] == call["status"], refused
    other = card([command, {"tool": "run_command", "status": "ok", "args": {"cmd": ["echo", "other"]}}])
    assert other[:3] == ("done", "", False) and left(other[3]) == [call["status"]]  # a different command


@pytest.mark.parametrize("timeout_facts", [
    {}, {"tool_result_code": "TOOL_TIMEOUT", "tool_result_meta": {"timeout_sec": 30}},
], ids=["legacy-status", "caller-wait-ended"])
def test_a_wait_timeout_alone_is_not_recorded_retry_execution(timeout_facts):
    call = {"tool": "run_command", "args": {"cmd": ["echo", "sample"]}, "is_error": True}
    held = {**call, "status": "safety_returned"}
    waited = {**call, "status": "timeout", **timeout_facts}
    axes = derive_loop_outcome("Delivered answer.", {}, {"tool_calls": [held, waited]})["outcome_axes"]
    assert [error["status"] for error in axes["execution"]["unresolved_tool_errors"]] == ["safety_returned", "timeout"]
    assert not axes["execution"]["recoveries"]


@pytest.mark.parametrize("later_code,recovered", [("COGNITIVE_TOOL_REQUIRED", False), ("OK", True)])
def test_a_cognitive_redirect_does_not_clear_a_held_call_on_the_finished_card(later_code, recovered):
    """Only synthetic results: both later calls look non-error/ok, but the redirect did not enter a handler."""
    from ouroboros.late_review_marks import current_outcome
    from ouroboros.loop_tool_execution import _typed_result_metadata
    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.tools.tool_result import ToolResult

    call = {"tool": "write_file", "args": {"root": "runtime_data", "path": "example.txt", "content": "sample"}}
    held = ToolResult("blocked", "SAFETY_RETURNED", "Synthetic recorded return")
    later = ToolResult("ok", later_code, "Same nonempty preview for both cases")
    trace = {"tool_calls": [
        {**call, "is_error": denied, **_typed_result_metadata(call["tool"], result.text, denied, result)}
        for denied, result in [(True, held), (False, later)]
    ]}
    assert trace["tool_calls"][1]["status"] == "ok"
    outcome = derive_loop_outcome("Delivered answer.", {}, trace)
    card = current_outcome({"status": "completed", "reason_code": "final_message", "result": "Delivered answer.",
                            "outcome_axes": outcome["outcome_axes"]})
    assert card["outcome_phase"] == "done" and _completion_verdict(card, {}) == ""
    execution = card["outcome_axes"]["execution"]
    assert execution["status"] == "ok" and not card["outcome_axes"]["objective"].get("warning")
    if recovered:
        assert not execution["unresolved_tool_errors"]
        [recovery] = execution["recoveries"]
        assert recovery["status"] == "safety_returned" and recovery["recovered_by_call_index"] == 2
    else:
        assert [item["status"] for item in execution["unresolved_tool_errors"]] == ["safety_returned"]
        assert not execution["recoveries"]


@pytest.mark.parametrize("tool,held,later", [
    ("github_write", {"issue": 12}, {"issue": 13}),
    ("place_reply", {}, {}),
    ("send", {"name": "", "text": "a"}, {"name": "", "text": "a"}),
    ("send", {"cwd": "/work"}, {"cwd": "/work"}),
    ("send", {"name": "report", "room": "one"}, {"name": "report", "room": "two"}),
    ("send", {"name": "report", "text": "<TRUNCATED:text:5000ch:sha=123>"},
             {"name": "report", "text": "<TRUNCATED:text:5000ch:sha=123>"}),
    ("send", {"name": "report", "token": "*** REDACTED ***"}, {"name": "report", "token": "*** REDACTED ***"}),
    ("send", {"name": "***REDACTED[str:len=4:sha256_8=12345678]***"},
             {"name": "***REDACTED[str:len=4:sha256_8=12345678]***"}),
])
def test_held_action_needs_positive_recorded_identity(tool, held, later):
    from ouroboros.outcomes import _classify_tool_errors

    calls = [{"tool": tool, "args": held, "is_error": True, "status": "safety_returned", "result": "same preview"},
             {"tool": tool, "args": later, "is_error": False, "status": "ok", "result": "same preview"}]
    classified = _classify_tool_errors({"tool_calls": calls})
    assert not classified["recovered"]
    assert [item["status"] for item in classified["unresolved"]] == ["safety_returned"]


def test_a_held_call_is_identified_by_its_recorded_address_and_complete_arguments():
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.outcomes import _classify_tool_errors
    from ouroboros.tool_effects import RETAINED_DECLARATION

    def recovered(first, second):
        result = _classify_tool_errors({"tool_calls": [
            {**first, "is_error": True, "status": "safety_returned"},
            {**second, "is_error": False, "status": "ok"}]})
        assert len(result["unresolved"]) + len(result["recovered"]) == 1
        return bool(result["recovered"])

    pause = {"tool": "pause_task", "args": {"task_id": "reports", "reason": "waiting for input"}}
    assert recovered(pause, pause)
    assert not recovered(pause, {**pause, "args": {**pause["args"], "task_id": "drafts"}})
    gh = {"tool": "github_write", "args": {"method": "comment", "repo": "example/demo", "target": 12, "body": "Reviewed."}}
    assert recovered(gh, gh)
    assert not recovered(gh, {**gh, "args": {**gh["args"], "body": "A different comment."}})

    name = extension_surface_name("example", "announce")
    declaration = {"skill": "example", "extension_generation": "one", "tool": {
        "name": "announce", "effect": "message", "target": "envelope.channel"}}
    send = {"tool": name, "args": {"envelope": {"channel": "reports"}, "text": "ok"}, RETAINED_DECLARATION: declaration}
    assert recovered(send, send)
    assert not recovered(send, {**send, RETAINED_DECLARATION: {**declaration, "extension_generation": "two"}})
    assert not recovered(send, {**send, "args": {**send["args"], "envelope": {"channel": "drafts"}}})
    for unknown in ({**send, "args": {"text": "ok"}}, {k: v for k, v in send.items() if k != RETAINED_DECLARATION}):
        assert not recovered(unknown, unknown)


def test_a_place_paused_after_a_safety_return_keeps_the_return_named(supervisor, published_tool, places, monkeypatch):
    """Safety returns a message to a place, the place is then paused, and the identical call is refused by the place
    guard, which the registry asks before Safety: the record still holds the returned call, which only the open Done
    card names."""
    from dataclasses import replace

    from ouroboros import place_guard, tool_effects
    from ouroboros.loop_tool_execution import _typed_result_metadata
    from ouroboros.project_dialogue import _completion_verdict
    from ouroboros.tools.registry_guards import _capability_resource_guard_result
    from ouroboros.tools.tool_result import ToolResult, safety_refusal

    rows, ctx = places
    ctx._current_llm_call_meta = {"llm_call_id": "round-1"}
    monkeypatch.setattr(tool_effects, "effect_of", REAL_EFFECT_OF)
    monkeypatch.setattr(place_guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    name, descriptor = published_tool("message", target="room", conversation="room", text="text")
    args = {"room": "C2", "text": "Prepared notes."}
    assert _capability_resource_guard_result(ctx, name, args, descriptor) is None
    supervisor.replies.append({"status": "DANGEROUS", "reason": "The standing words keep these notes private."})
    allowed, words = supervisor.module.check_safety(name, args, ctx=ctx, ext_tool=descriptor)
    assert not allowed and len(supervisor.calls) == 1
    rows[1] = replace(rows[1], state="paused")
    refused = _capability_resource_guard_result(ctx, name, args, descriptor)
    assert refused.code == "RESOURCE_CONSTRAINT_BLOCKED"
    trace = {"tool_calls": [{"tool": name, "args": args, "is_error": True,
                             **_typed_result_metadata(name, item.text, True, item)}
                            for item in (ToolResult(*safety_refusal(words), text=words), refused)]}
    outcome = derive_loop_outcome("Prepared the notes.", {}, trace)
    record = {"status": "completed", "reason_code": "final_message", "outcome_axes": outcome["outcome_axes"]}
    unresolved = outcome["outcome_axes"]["execution"]["unresolved_tool_errors"]
    assert [e["tool"] for e in unresolved if e["status"] == "safety_returned"] == [name]
    assert "Safety Supervisor" not in _completion_verdict(record, {})


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


@pytest.mark.parametrize("status, is_error, fails", [
    ("ok_autocorrected", False, False), ("artifact_output_undeclared", False, False), ("untyped", False, False),
    ("artifact_output_error", True, True), ("non_zero_exit", True, True),
    ("review_blocked", False, True), ("git_error", False, True)])  # typed ok codes, yet not a success
def test_the_ledger_lists_a_typed_ok_answer_without_calling_it_a_failure(status, is_error, fails):
    """#3214: a call the execution axis reads as a success (``_OK_TOOL_STATUSES``: a shell regex corrected and run, a
    finished command carrying the undeclared-output note, an untyped dynamic success) stays a listed ``tool_call``
    entry, and the ledger summary's ``has_failures`` no longer counts it; a failed registration, a non-zero exit,
    a review block and a git error still count."""
    from ouroboros.outcomes import build_verification_ledger

    trace = {"tool_calls": [{"tool": "run_command", "status": status, "is_error": is_error, "args": {}}]}
    outcome = derive_loop_outcome("Delivered answer.", {}, trace)
    ledger = build_verification_ledger(task={}, loop_outcome=outcome, llm_trace=trace, artifact_bundle={})
    assert [entry["status"] for entry in ledger["entries"] if entry["kind"] == "tool_call"] == [status]
    assert ledger["summary"]["has_failures"] is fails
