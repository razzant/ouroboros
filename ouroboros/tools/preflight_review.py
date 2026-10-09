"""``preflight_review``: one named row's early look at the worktree (decision 3A).

The tool is ``review_change(subject=worktree, surface=preflight, reviewers=[reviewer])``
on the system repository, the same action ``commit_reviewed(preflight_reviewer=…)``
takes, so a preflight has no pipeline of its own: one ledger record, the record's
reuse, its budget and its ceiling. A ``surface=preflight`` record never answers the
commit panel (the reuse key carries the surface). ``deterministic_only`` keeps the
free release-metadata diagnostics (``commit_gate.release_diagnostics``);
``advisory_review`` stays callable under the old name. ``review_status`` is the
read-only diagnostic of commit attempts, open obligations and commit-readiness debt.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.registry import ToolContext, ToolEntry


def _handle_preflight_review(
    ctx: ToolContext, reviewer: str = "", commit_message: str = "", goal: str = "", scope: str = "",
    review_rebuttal: str = "", paths: Optional[List[str]] = None, deterministic_only: bool = False,
    source: str = "",
) -> str:
    from ouroboros.tools.commit_gate import preflight_reviewer_error, release_diagnostics

    if deterministic_only:
        return json.dumps(release_diagnostics(ctx, paths, source), ensure_ascii=False, indent=2)
    reviewer = str(reviewer or "").strip()
    error = (preflight_reviewer_error(reviewer) if reviewer
             else "reviewer is required: name one enabled catalog row by id or handle")
    if error:
        return argument_refusal(ctx, "TOOL_ARG_ERROR (preflight_review)", [error], effect="No reviewer was dispatched.")
    from ouroboros.tools.review_change import _handle_review_change

    return _handle_review_change(ctx, root="system_repo", subject="worktree", surface="preflight",
                                 reviewers=[reviewer], goal=goal or commit_message, scope=scope,
                                 review_rebuttal=review_rebuttal)


def _next_step(projection: Dict[str, Any]) -> str:
    from ouroboros.tools.review_helpers import review_enforcement_blocks

    obligations = len(projection.get("open_obligations") or [])
    carried = (f" {obligations} open obligation(s) from earlier blocked rounds ride into the next panel's brief."
               if obligations else "")
    if not review_enforcement_blocks("blocking"):
        return ("Cyber Pro: whether and how to commit is your judgment; findings, missing evidence and pending "
                "operations stay recorded, and this is not a PASS." + carried)
    return ("When the edits are complete, run commit_reviewed(commit_message='...'): the deterministic checks, "
            "the tests and the review panel decide; preflight_reviewer='<enabled row>' optionally buys one "
            "early look first." + carried)


def _handle_review_status(
    ctx: ToolContext, repo_key: str = "", tool_name: str = "", task_id: str = "",
    attempt: Optional[int] = None, include_raw: bool = False,
) -> str:
    from ouroboros.review_state import compute_snapshot_hash
    from ouroboros.review_status_projection import build_review_projection, build_review_status_payload

    projection = build_review_projection(
        ctx.drive_root, repo_dir=getattr(ctx, "repo_dir", ""), repo_key=repo_key, tool_name=tool_name,
        task_id=task_id, attempt=attempt, snapshot_hash_fn=compute_snapshot_hash,
        reader_task_id=str(getattr(ctx, "task_id", "") or ""))
    payload = build_review_status_payload(projection, next_step=_next_step(projection), include_raw=include_raw)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _param(param_type: str, description: str, **extra: Any) -> Dict[str, Any]:
    return {"type": param_type, "description": description, **extra}


def _preflight_review_params() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reviewer": _param("string", "One ENABLED catalog row (id or handle), a review-pool member or not. "
                                         "Required unless deterministic_only."),
            "commit_message": _param("string", "The intended commit message; the goal when goal is empty."),
            "goal": _param("string", "What the change is meant to achieve."),
            "scope": _param("string", "What the change deliberately covers."),
            "review_rebuttal": _param("string", "Your answer to the previous findings; buys a new look."),
            "paths": _param("array", "deterministic_only: limit the diagnostics to these paths.",
                            items={"type": "string"}),
            "deterministic_only": _param("boolean", "Only diagnose release metadata, free: no reviewer, staging, "
                                                    "tests or record; requires source.", default=False),
            "source": _param("string", "deterministic_only: worktree reads current files; index reads staged blobs.",
                             enum=["worktree", "index"]),
        },
        "required": [],
    }


_PREFLIGHT_DESCRIPTION = (
    "Optional early look by ONE named enabled catalog row, in the review pool or not. Reads the system "
    "worktree via review_change(subject=worktree, surface=preflight, reviewers=[reviewer]), also used by "
    "commit_reviewed(preflight_reviewer=...). Informational, never a gate or an answer for the commit panel; "
    "unchanged worktree reuses its settled record free. No tests here; commit_reviewed runs them. "
    "deterministic_only=True requires source=worktree|index and returns free release diagnostics instead. "
    "Returns review_change's JSON."
)

_REVIEW_STATUS_DESCRIPTION = (
    "Read-only diagnostic of reviewed commits: the last commit attempt (reviewing/blocked/succeeded/failed) with "
    "its block reason and guidance; open obligations from earlier blocking rounds (they ride into the next panel's "
    "brief); open commit-readiness debt (`commit_readiness_debts`, `commit_readiness_debts_count`; a durable "
    "anti-thrashing signal that `retry_anchor` names, never an admission gate); the checkout's newest "
    "preflight look (`preflight`) and whether the worktree "
    "moved since it (`stale_from_edit`, its editor when a tool recorded the edit); the history of legacy "
    "advisory runs; and a next_step. Pass include_raw=true "
    "for the targeted attempt's full per-actor evidence (triad_raw_results, scope_raw_result)."
)


def get_tools() -> List[ToolEntry]:
    from ouroboros.tools.review_change import _review_change_tool_timeout_sec

    timeout = _review_change_tool_timeout_sec()
    return [
        ToolEntry("preflight_review", {"name": "preflight_review", "description": _PREFLIGHT_DESCRIPTION,
                                       "parameters": _preflight_review_params()},
                  _handle_preflight_review, timeout_sec=timeout),
        # The old public name stays callable for saved prompts and configs, never advertised.
        ToolEntry("advisory_review", {"name": "advisory_review",
                                      "description": "Compatibility alias for `preflight_review`.",
                                      "parameters": _preflight_review_params()},
                  _handle_preflight_review, timeout_sec=timeout, alias_for="preflight_review"),
        ToolEntry("review_status", {"name": "review_status", "description": _REVIEW_STATUS_DESCRIPTION,
                                    "parameters": {"type": "object", "properties": {
                                        "repo_key": _param("string", "Optional repo identity filter."),
                                        "tool_name": _param("string", "Optional tool-name filter (for example "
                                                                      "commit_reviewed)."),
                                        "task_id": _param("string", "Optional task-id filter."),
                                        "attempt": _param("integer", "Optional attempt number within the selected "
                                                                     "repo/tool/task scope."),
                                        "include_raw": _param("boolean", "Append the targeted attempt's full "
                                                                         "per-actor evidence. Default false."),
                                    }, "required": []}},
                  _handle_review_status),
    ]
