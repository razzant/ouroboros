"""The owner's words for the tool error a residual-warning ending names.

``residual_tool_errors_without_review`` read one sentence whatever had failed. The record already
holds the failures (``outcome_axes.execution.unresolved_tool_errors`` first, else
``cosmetic_tool_errors``; ``outcomes`` keeps at most 20 of each), so the clause names the first LOUD one — not a
failed read (``read_only``, owner 06.11 A) and not a details-only leftover (``details_only``, owner 09.11 A: a
command that ran and failed, a call refused before running; both stay in the details) — in plain words: a noun for
the step and what happened to it, never a tool name, its result text or a path (DESIGN "Owner-facing words"; owner decision 16.09 Q5=A). Browser twin:
``web/modules/task_reason_words.js`` (``residualToolErrorsPhrase``); the cases of
``web/tests/fixtures/outcome_phase_parity.json`` pin both.
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
    "schedule_subagent": "A helper start",
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
