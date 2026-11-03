"""The owner's words for the tool error a residual-warning ending names.

``residual_tool_errors_without_review`` read one sentence whatever had failed. The record already
holds the failures (``outcome_axes.execution.unresolved_tool_errors`` first, else
``cosmetic_tool_errors``; ``outcomes`` keeps at most 20 of each), so the clause names the first in
plain words: a noun for the step and what happened to it, never a tool name, its result text or a
path (DESIGN "Owner-facing words"; owner decision 16.09 Q5=A). Browser twin:
``web/modules/task_reason_words.js`` (``residualToolErrorsPhrase``); the cases of
``web/tests/fixtures/outcome_phase_parity.json`` pin both.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
}
_LISTED = 20  # outcomes keeps at most this many errors per list


def _errors(execution: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = execution.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _int(value: Any) -> Any:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def residual_tool_errors_phrase(execution: Any) -> str:
    """The clause naming the first recorded tool error, or "" when the record names none."""
    execution = execution if isinstance(execution, dict) else {}
    unresolved, cosmetic = _errors(execution, "unresolved_tool_errors"), _errors(execution, "cosmetic_tool_errors")
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
    capped = len(unresolved) >= _LISTED or len(cosmetic) >= _LISTED
    tail = (", plus more tool errors" if capped else ", plus 1 more tool error" if more == 1
            else f", plus {more} more tool errors" if more else "")
    return f"{subject}{tail}; the result has not been evaluated."
