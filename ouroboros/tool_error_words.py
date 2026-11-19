"""The owner's words for the tool error a residual-warning ending names.

``residual_tool_errors_without_review`` read one sentence whatever had failed. The record already
holds the failures (``outcome_axes.execution.unresolved_tool_errors`` first, else
``cosmetic_tool_errors``; ``outcomes`` keeps at most 20 of each), so the clause names the first LOUD one — not a
failed read (``read_only``, owner 06.11 A) and not a details-only leftover (``details_only``, owner 09.11 A: a
command that ran and failed, a call refused before running; both stay in the details) — in plain words: a noun for
the step and what happened to it, never a tool name, its result text or a path (DESIGN "Owner-facing words"; owner decision 16.09 Q5=A). Browser twin:
``web/modules/task_reason_words.js`` (``residualToolErrorsPhrase``); the cases of
``web/tests/fixtures/outcome_phase_parity.json`` pin both.

``verification_red_unreconciled`` read one sentence too, true and misleading where a corrected check had passed since.
``red_check_phrase`` reads the fact the record holds (``objective.verification_red``,
``_outcome_receipts.red_check_fact``): why the check did not pass, in the words of the card's "Check did not pass"
row where it has one, and whether any check passed after it; the command and paths stay in the receipt. Browser twin:
``web/modules/verification_presentation.js`` (``redCheckPhrase``), pinned by the same fixture.
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
