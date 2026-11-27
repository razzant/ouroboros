"""``preflight_review``: one named row's early look at the worktree (decision 3A).

The tool is ``review_change(subject=worktree, surface=preflight, reviewers=[reviewer])``
on the system repository, the same action ``commit_reviewed(preflight_reviewer=…)``
takes, so a preflight has no pipeline of its own: one ledger record, the record's
reuse, its budget and the code-review allowance it spends with ``review_change``'s
code waves and the commit panel. A ``surface=preflight`` record never answers the
commit panel (the reuse key carries the surface). ``deterministic_only`` keeps the
free release metadata and labelled worktree book balance (``commit_gate.release_diagnostics``);
``advisory_review`` stays callable under the old name. ``review_status`` is the
read-only diagnostic of commit attempts and the checkout's preflight freshness.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.registry import ToolContext, ToolEntry


def _handle_preflight_review(
    ctx: ToolContext, reviewer: str = "", commit_message: str = "", goal: str = "", scope: str = "",
    review_rebuttal: str = "", paths: Optional[List[str]] = None, deterministic_only: bool = False,
    source: str = "",
) -> str:
    from ouroboros.tools.commit_gate import commit_preflight_choice_error, release_diagnostics

    if deterministic_only:
        return json.dumps(release_diagnostics(ctx, paths, source), ensure_ascii=False, indent=2)
    reviewer = str(reviewer or "").strip()
    error = (commit_preflight_choice_error(reviewer, skipped=False, continuation=False, ctx=ctx,
                                          goal=goal or commit_message, scope=scope, review_rebuttal=review_rebuttal)
             if reviewer else "reviewer is required: name one enabled catalog row by id or handle")
    if error:
        return argument_refusal(ctx, "TOOL_ARG_ERROR (preflight_review)", [error], effect="No reviewer was dispatched.")
    from ouroboros.tools.review_change import _handle_review_change

    return _handle_review_change(ctx, root="system_repo", subject="worktree", surface="preflight",
                                 reviewers=[reviewer], goal=goal or commit_message, scope=scope,
                                 review_rebuttal=review_rebuttal)


def _next_step() -> str:
    from ouroboros.tools.review_helpers import cyber_review_authority

    if cyber_review_authority():
        return ("Cyber Pro: whether and how to commit is your judgment; findings, missing evidence and pending "
                "operations stay recorded, and this is not a PASS.")
    return ("When the edits are complete, run commit_reviewed(commit_message='...'): the deterministic checks, "
            "the tests and the review panel run (the outcome returns before any commit); "
            "preflight_reviewer='<enabled row>' optionally buys one early look first, from the same paid "
            "allowance as the panel (a new panel wave keeps the last one; a matching earlier look is reused free).")


def _predecessor_commit_rows(ctx, canonical, repo_key, task_filter, tool_name=""):
    """``(rows, dropped ids, gap, unread ids)``: this checkout's commit attempts of the tasks the owner's recorded
    Continue chain leads from to this one, each read from that task's own admitted drives
    (``review_collection.commit_attempt_roots``), never from another task's fork. Optional history: a read that fails is
    named once in ``gap`` and leaves this drive's own rows and any rows already proven on higher-priority drives alone.
    ``dropped`` are the known predecessors whose copies on this fork are no source: the verified ones (their rows
    replace the copies) and the ones that could not be read or verified. An empty ``repo_key`` (another checkout's
    filter) lists and reads no drive, so nothing replaces a copy there; only the drive this call reads is asked whether
    it is a predecessor's own source (the shared canonical drive, or its retained drive by the existing placement check,
    ``source_retention.retained_child_root``), and the rows it holds of such a predecessor stay as ``unread`` history,
    never evidence. A tool filter no commit attempt answers walks no chain and no drive."""
    from ouroboros.owner_continue import continuation_chain
    from ouroboros.review_state import _load_state_unlocked
    from ouroboros.review_state_records import _attempt_identity_tuple
    from ouroboros.source_retention import retained_child_root
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.review_collection import commit_attempt_roots

    own, tools = str(getattr(ctx, "task_id", "") or ""), {"commit_reviewed", "vcs_commit_reviewed"}
    if not own or task_filter == own or (tool_name and tool_name not in tools):
        return [], set(), "", set()
    gaps, linked, seen, unreadable_link = [], [], {own}, ""
    current = own
    while current:  # the successor's backward links only suggest candidates; the forward claims below decide
        try:
            result = load_task_result(canonical, current, strict=True) or {}
        except Exception as exc:  # StoreBusy included: an optional read never takes this drive's status down
            unreadable_link = current
            gaps.append(f"the Continue link of {current} could not be read ({type(exc).__name__})")
            break
        if current == task_filter:
            break
        metadata = result.get("metadata")
        continuation = metadata.get("continuation") if isinstance(metadata, dict) else None
        prior = continuation.get("predecessor_task_id") if isinstance(continuation, dict) else None
        if not isinstance(prior, str) or not prior or prior in seen:
            break
        linked.append(prior)
        seen.add(prior)
        current = prior
    verified: set = set()
    for candidate in reversed(linked):  # one verified chain covers every later candidate
        chain = continuation_chain(canonical, candidate, own)
        if chain:
            verified.update(chain[:-1])
            break
    if task_filter and task_filter not in verified:
        verified.update(continuation_chain(canonical, task_filter, own)[:-1])
    unverified = [task for task in linked if task not in verified and task != unreadable_link]
    if task_filter:
        unverified = [task for task in unverified if task == task_filter]
    if unverified:
        gaps.append("the recorded Continue of " + ", ".join(unverified) + " is not verified or could not be read")

    rows, reads, unread = [], {}, set()  # one read per physical drive for the whole call
    drive = pathlib.Path(ctx.drive_root).resolve(strict=False) if getattr(ctx, "drive_root", None) else None
    targets = ([task_filter] if task_filter in verified else ()) if task_filter else sorted(verified)
    for task in targets:
        if not repo_key:  # another checkout's filter lists no drive of the task: only the one this call reads is asked
            if drive is not None:
                try:
                    if (drive == pathlib.Path(canonical).resolve(strict=False)
                            or retained_child_root(canonical, drive, task, strict=True)):
                        unread.add(task)  # that drive is the task's own source, so its rows there are no copies
                except Exception as exc:
                    gaps.append(f"{drive} could not be read ({type(exc).__name__})")
            continue
        held = {}
        try:
            roots = commit_attempt_roots(
                ctx, task, verified_canonical=canonical,
                on_error=lambda path, exc: gaps.append(f"{path} could not be read ({type(exc).__name__})"),
            )
        except Exception as exc:
            gaps.append(f"the drives of {task} could not be listed ({type(exc).__name__})")
            continue
        for root in roots:
            key = str(pathlib.Path(root).resolve(strict=False))
            if key not in reads:
                try:
                    reads[key] = _load_state_unlocked(root, strict_attempt_authority=True).attempts
                except Exception as exc:
                    reads[key] = exc
            if isinstance(reads[key], Exception):
                gaps.append(f"{root} could not be read ({type(reads[key]).__name__})")
                break  # keep rows already proven on higher-priority drives; stop before any staler fallback
            for row in reads[key]:  # this task's own rows only; the first admitted drive holding an identity wins
                if row.task_id == task and row.repo_key == repo_key and row.tool_name in tools:
                    held.setdefault(_attempt_identity_tuple(row), row)
        rows.extend(held.values())
    dropped = verified | set(linked)
    if not repo_key:  # no read replaces a copy here, so a shared drive keeps the rows it holds
        dropped -= unread
    unique_gaps = list(dict.fromkeys(gaps))
    gap = ("Continue predecessor attempts are not all listed: " + "; ".join(unique_gaps) + ".") if unique_gaps else ""
    return rows, dropped, gap, unread


def _handle_review_status(
    ctx: ToolContext, repo_key: str = "", tool_name: str = "", task_id: str = "",
    attempt: Optional[int] = None, include_raw: bool = False,
) -> str:
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state import make_repo_key
    from ouroboros.review_status_projection import build_review_projection, build_review_status_payload

    # ledger_root is review_state_root's canonical root without its move: reading migrates nothing.
    canonical = ledger_root(ctx)
    repo_dir = getattr(ctx, "repo_dir", "")
    checkout = make_repo_key(pathlib.Path(repo_dir)) if repo_dir else ""
    # A filter on this drive does not open a different checkout on a predecessor's drive.
    rows, predecessors, gap, unread = _predecessor_commit_rows(
        ctx, canonical, checkout if not repo_key or repo_key == checkout else "", task_id, tool_name)
    projection = build_review_projection(
        ctx.drive_root, repo_dir=repo_dir, repo_key=repo_key, tool_name=tool_name,
        task_id=task_id, attempt=attempt,
        reader_task_id=str(getattr(ctx, "task_id", "") or ""), canonical_root=canonical,
        predecessor_rows=rows, predecessor_ids=predecessors, predecessor_gap=gap, predecessor_unread=unread)
    selected = projection.get("selected_attempt")
    if selected is not None and selected.tool_name == "commit_reviewed" and selected.status == "reviewing":
        # Read-only repair view of an unfinished projection, never a new handback for a blocked commit.
        from ouroboros.tools.review_change_custody import settled_record
        projection["selected_settled"] = settled_record(ledger_root(ctx), selected) is not None
    payload = build_review_status_payload(projection, next_step=_next_step(), include_raw=include_raw)
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
            "deterministic_only": _param("boolean", "Free release diagnostics and worktree book balance before "
                                                    "buying review; no reviewer, staging, tests or record; requires source.", default=False),
            "source": _param("string", "Release metadata: worktree reads current files; index reads staged blobs and admits neutral contributions. "
                                      "Ordinary serving commits still require a numbered release. Book balance is labelled worktree.",
                             enum=["worktree", "index"]),
        },
        "required": [],
    }


_PREFLIGHT_DESCRIPTION = (
    "Optional paid early look at the system worktree by ONE named enabled catalog row, in the review pool or "
    "not; deterministic_only is the free diagnostic form. Reads the system "
    "worktree via review_change(subject=worktree, surface=preflight, reviewers=[reviewer]), also used by "
    "commit_reviewed(preflight_reviewer=...). Informational, never a gate or an answer for the commit panel; "
    "unchanged worktree reuses its settled record free; a new look spends the paid code-review allowance "
    "review_change and the commit panel share. No tests here; commit_reviewed runs them. "
    "deterministic_only=True requires source=worktree|index and returns free release diagnostics plus a separately "
    "labelled worktree book balance instead. Returns a review record or the free diagnostic JSON."
)

_REVIEW_STATUS_DESCRIPTION = (
    "Read-only diagnostic of reviewed commits: attempts for this checkout, including recorded Continue predecessors "
    "(`predecessor_attempts_gap` names any not read), with the last matching attempt's block reason and guidance; the newest preflight look (`preflight`) and whether the worktree "
    "moved since it (`stale_from_edit`, its editor when a tool recorded the edit); where this drive keeps the "
    "retired advisory lane's and obligation layer's rows (`advisory_run_archive`, `obligation_archive`, history "
    "only); and a next_step. Pass include_raw=true for the targeted attempt's full per-actor evidence "
    "(triad_raw_results, scope_raw_result)."
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
