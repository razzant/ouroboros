"""The owner's words for the tool error a residual-warning ending names.

``residual_tool_errors_without_review`` read one sentence whatever had failed. The record already
holds the failures (``outcome_axes.execution.unresolved_tool_errors`` first, else
``cosmetic_tool_errors``; ``outcomes`` keeps at most 20 of each), so the clause names the first LOUD one — not a
failed read (``read_only``, owner 06.11 A) and not a details-only leftover (``details_only``, owner 09.11 A: a
command that ran and failed, a call refused before running; both stay in the details) — in plain words: a noun for
the step and what happened to it, never a tool name, its result text or a path (DESIGN principle 7 "Words serve the next step"; owner decision 16.09 Q5=A). Browser twin:
``web/modules/task_reason_words.js`` (``residualToolErrorsPhrase``); the cases of
``web/tests/fixtures/outcome_phase_parity.json`` pin both.

``verification_red_unreconciled`` read one sentence too, true and misleading where a corrected check had passed since.
``red_check_phrase`` reads the fact the record holds (``objective.verification_red``,
``_outcome_receipts.red_check_fact``): why the check did not pass, in the words of the card's "Check did not pass"
row where it has one, and whether any check passed after it; the command and paths stay in the receipt. Browser twin:
``web/modules/verification_presentation.js`` (``redCheckPhrase``), pinned by the same fixture.

``configured_actor_incomplete`` / ``configured_actor_unknown`` read one sentence whatever the first coding attempt of a
helper had done, and said no run succeeded. ``first_attempt_cause_key`` picks a sentence from the stored
``outcome_axes.execution.actor_first_terminal`` instead, only where its typed witness PROVES the first attempt: an
``initial_execution.state`` out of the closed ``_FIRST_ATTEMPT_STATES`` vocabulary (a settled success whose work-order
coverage was never confirmed carries ``succeeded_source_unverified`` and proves only the ending), or a typed zero-run receipt —
a validated historical ``complete`` included, since it still proves no physical start while the fact stays UNKNOWN.
The reason codes stay as recorded; the producer's status×reason pairing is deliberately NOT re-derived here (a
duplicated matrix is a second exception list that reads every new producer fact as unknown).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ouroboros._outcome_tool_errors import SAFETY_HELD_STATUSES

# The steps that most often stay failed at an ending; any other reads "A tool step".
_TOOL_NOUNS = {
    "run_command": "A command",
    "run_script": "A script",
    "read_file": "A file read",
    "list_files": "A folder listing",
    "query_data": "A data query",
    "github_read": "A GitHub read",
    "query_code": "A code search",
    "search_code": "A code search",
    "pr_merge": "A pull request merge",
    "review_change": "A review request",
    "finish_task": "A finish request",
    # The effects the main view names once command failures stay in the details (owner 09.11, A).
    "write_file": "A file write",
    "edit_text": "A file edit",
    "edit_batch": "A file edit",
    "apply_patch": "A file edit",
    "github_write": "A GitHub write",
    "send_user_message": "A message send",
    "place_reply": "A reply",
    "schedule_subagent": "A subagent start",
    "commit_reviewed": "A commit",
    "vcs_commit_reviewed": "A commit",
    "skill_exec": "A skill run",
}
_LISTED = 20  # outcomes keeps at most this many errors per list


def _errors(execution: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = execution.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _int(value: Any) -> Any:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _loud(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The errors the main view names: a failed read (owner 06.11, A) and a details-only leftover (owner 09.11, A)
    stay in the details; a call Safety held only in the open Done card (owner 11.11, A). Twin of
    ``_outcome_tool_errors.warns_without_review``."""
    return [error for error in errors if not (error.get("read_only") or error.get("details_only")
                                              or error.get("status") in SAFETY_HELD_STATUSES)]


def residual_tool_errors_phrase(execution: Any) -> str:
    """The clause naming the first recorded loud tool error; "" when the record names none."""
    execution = execution if isinstance(execution, dict) else {}
    listed_unresolved, listed_cosmetic = _errors(execution, "unresolved_tool_errors"), _errors(execution, "cosmetic_tool_errors")
    unresolved, cosmetic = _loud(listed_unresolved), _loud(listed_cosmetic)
    errors = unresolved or cosmetic
    if not errors:
        return ""
    first = errors[0]
    noun = _TOOL_NOUNS.get(str(first.get("tool") or ""), "A tool step")
    status, code, signal = str(first.get("status") or ""), _int(first.get("exit_code")), first.get("signal")
    if status == "unknown_tool":
        subject = "A call named a tool that does not exist"
    elif status == "non_zero_exit" and signal not in (None, ""):
        subject = f"{noun} was stopped by signal {signal}"
    elif status == "non_zero_exit":
        subject = f"{noun} returned exit code {code}" if code is not None else f"{noun} returned a non-zero exit code"
    elif status.endswith("timeout"):
        subject = f"{noun} timed out"
    elif status == "argument_error":
        subject = f"{noun} was called with invalid arguments"
    else:
        subject = f"{noun} failed"
    more = len(unresolved) + len(cosmetic) - 1
    # "More" is asserted only when a list the producer capped is loud through and through; otherwise the retained
    # loud errors are counted exactly (details-only entries never stand in for an unseen loud one).
    capped = any(len(listed) >= _LISTED and len(loud) == len(listed)
                 for listed, loud in ((listed_unresolved, unresolved), (listed_cosmetic, cosmetic)))
    tail = (", plus more tool errors" if capped else ", plus 1 more tool error" if more == 1
            else f", plus {more} more tool errors" if more else "")
    return f"{subject}{tail}."


# Why a check did not pass, in the words of the card's "Check did not pass" row (verification_presentation.js).
_MISSED_MATCH = {"substring": "expected text not found", "exact": "output did not match exactly",
                 "exact_line": "no output line matched exactly", "json_equals": "JSON output not equal"}


def check_cause(fact: Dict[str, Any]) -> str:
    code, mode, matched = _int(fact.get("exit_code")), str(fact.get("expected_match") or ""), fact.get("matched")
    if fact.get("kind") == "artifact_observation":  # the host looked for the named files and ran no command
        return "a named file was not found" if fact.get("paths_named") else "no files were named"
    if fact.get("signal"):
        return f"killed by {fact['signal']}"
    if code is None:  # a timeout and a stopped check record none; the receipt does not say which
        return "no exit code"
    if code != 0:
        return f"exit code {code}"
    if mode == "bytes_equal":  # the process exited 0: its comparison is what did not pass
        return "exit code 0; " + ("files not byte-equal" if matched is False else "comparison not confirmed")
    return "exit code 0; " + (_MISSED_MATCH.get(mode, _MISSED_MATCH["substring"]) if matched is False
                              else "criterion not confirmed")


def red_check_phrase(objective: Any) -> str:
    """The detail sentence of the recorded red check; "" when the record holds no such fact (its stored word stands)."""
    fact = objective.get("verification_red") if isinstance(objective, dict) else None
    if not isinstance(fact, dict):
        return ""
    after, more = max(_int(fact.get("passed_after")) or 0, 0), max(_int(fact.get("more")) or 0, 0)
    head = (f"A recorded check did not pass ({check_cause(fact)}); {after} check{'' if after == 1 else 's'} passed "
            "after it, none recorded as a rerun of it" if after
            else f"The last recorded check did not pass ({check_cause(fact)}); no check passed after it")
    tail = f"; {more} more failed check{' has' if more == 1 else 's have'} no recorded passing rerun" if more else ""
    return f"{head}{tail}."


_ACTOR_REASONS = ("configured_actor_incomplete", "configured_actor_unknown")
# The closed vocabulary of the typed first-attempt witness (``initial_execution.state``): the custody
# terminal states that prove what the FIRST coding attempt did. A settled success whose work-order
# coverage stayed unconfirmed carries ``succeeded_source_unverified`` (delegate_evidence): it gets
# its own words, never the succeeded attempt's sentence.
_FIRST_ATTEMPT_STATES = ("failed", "cancelled", "interrupted", "succeeded", "succeeded_source_unverified")


def first_attempt_cause_key(execution: Dict[str, Any]) -> str:
    """The sentence key the stored ``actor_first_terminal`` PROVES about the FIRST coding attempt, else ``""``.

    The typed witness is the single source: an ``initial_execution.state`` out of ``_FIRST_ATTEMPT_STATES``
    with its own run id (a refused start is ``failed_definite`` without one), or a typed zero-run receipt —
    the decision-as-status spelling, or a validated historical ``complete`` receipt (``historical_zero_run_complete``),
    whose ``physical_run_started=False`` validation still proves the leaf never started while the fact's own
    status stays UNKNOWN. The producer's status×reason pairing is not re-checked here. ``physical_leaf_not_started``
    is not proof: it is also what a lost receipt looks like. Malformed values (a non-string reason, a state
    outside the vocabulary) prove nothing. Never a claim about later runs or the answer (a settled success is
    that session's own end). Twin: ``firstAttemptCauseKey`` (``web/modules/task_reason_words.js``)."""
    fact = execution.get("actor_first_terminal") if isinstance(execution, dict) else None
    status, why = (fact.get("status"), fact.get("reason")) if isinstance(fact, dict) else (None, None)
    if status not in ("incomplete", "unknown") or not isinstance(why, str):
        return ""
    if fact.get("zero_run") is True:
        decision = fact.get("zero_run_decision")
        if decision == status and why == f"configured_actor_zero_run_{decision}":
            return "configured_actor_not_started"
        # A validated historical receipt (the pre-charter "complete" decision): the store only
        # accepts it with physical_run_started False, so the absent start is proven; the
        # self-reported completion is not, and this function never touches the fact's status.
        return ("configured_actor_not_started"
                if (status, decision, why) == ("unknown", "complete", "historical_zero_run_complete") else "")
    initial = fact.get("initial_execution")
    if not isinstance(initial, dict):
        return ""
    state, run_id = initial.get("state"), initial.get("run_id")
    if state in _FIRST_ATTEMPT_STATES:
        return f"configured_actor_first_{state}" if isinstance(run_id, str) and run_id.strip() else ""
    no_run = run_id is None or (isinstance(run_id, str) and not run_id.strip())
    return "configured_actor_first_refused" if state == "failed_definite" and no_run else ""


def configured_actor_cause_key(reason: str, execution: Dict[str, Any]) -> str:
    """The sentence key for ``configured_actor_incomplete`` / ``_unknown``: the reason codes stay as recorded.

    The words come from ``first_attempt_cause_key`` only when the stored fact's own status matches the code; anything
    else (a reason-only or legacy record, a reason that is not a code, unknown or unreadable evidence) says the first
    attempt's outcome could not be determined. Twin: ``configuredActorCauseKey`` (``web/modules/task_reason_words.js``)."""
    if not isinstance(reason, str):
        return ""  # a malformed code is never a sentence
    if reason not in _ACTOR_REASONS:
        return reason
    fact = execution.get("actor_first_terminal") if isinstance(execution, dict) else None
    if not isinstance(fact, dict) or fact.get("status") != reason.removeprefix("configured_actor_"):
        return "configured_actor_first_undetermined"
    return first_attempt_cause_key(execution) or "configured_actor_first_undetermined"


def first_attempt_clause_key(reason: str, execution: Dict[str, Any]) -> str:
    """The proven first-attempt key to state BESIDE a different primary reason (a round limit, a refusal ...), else
    ``""``; under the actor's own codes the primary clause already says it. Twin: ``firstAttemptClauseKey``."""
    return "" if not isinstance(reason, str) or reason in _ACTOR_REASONS else first_attempt_cause_key(execution)
