"""One read-only present for new heads, derived from the independent-root roster.

Focus is the model's authored account of its work. It is deliberately distinct
from the web card's latest visible narration and never supplies routing authority.
"""
from __future__ import annotations

import pathlib


def root_present_fields(root, task_id, task, stored, *, settled=False):
    """Enrich the roster using the task document it has already read."""
    from ouroboros.project_admission import project_hold_fact
    from ouroboros.projects_registry import get_project, project_binding_for_task

    from ouroboros.store import StoreError

    fields = {}
    if root is not None:
        try:
            binding = project_binding_for_task(root, task_id) or {}
            if binding:
                fields.update(project_id=binding["project_id"], chat_id=binding.get("project_chat_id"))
            project = get_project(root, fields.get("project_id") or task.get("project_id") or "") or {}
            fields["project_name"] = str(project.get("name") or "")
        except StoreError as exc:
            fields["unavailable"] = {"project": type(exc).__name__}
    if settled:
        fields["recorded_status"] = stored["status"]
        return fields
    quizzes = stored.get("owner_quiz") or {}
    questions = [{"quiz_id": quiz_id, **{key: quiz[key] for key in (
        "question", "options", "asked_at", "wait_for_answer", "state") if key in quiz}}
        for quiz_id, quiz in quizzes.items() if isinstance(quiz, dict) and quiz.get("state", "open") == "open"]
    if questions:
        fields["open_questions"] = questions
    hold = project_hold_fact({**stored, **task})
    if hold:
        fields["project_admission_hold"] = hold
    return fields


# One neutral sentence of understanding, not a rule: what the list holds and that tidying it is ordinary work.
_HYGIENE_NOTE = ("Every live, waiting, paused or held work in your rooms, with what it waits for, since when and "
                 "what of it still runs (tails); "
                 "other_live_work is work whose own root ended while its helpers or external runs go on. Keeping this "
                 "list free of work nobody needs any more is ordinary hygiene: close it, ask the owner, or leave it, "
                 "as you judge.")


def live_works_present(root, *, exclude=""):
    """Project every host-listed root; stale/missing source observations stay explicit."""
    from ouroboros.peer_roster import independent_roots

    roster = independent_roots(pathlib.Path(root))
    works = []
    for row in roster["roots"]:
        if row["task_id"] == exclude:
            continue
        focus = row.get("focus") or {}
        works.append({"task_id": row["task_id"], "title": row.get("title") or row.get("suggested_name") or "",
            "room": {key: row.get(key) for key in ("chat_id", "project_id", "project_name")},
            "state": row["status"], "started_at": row.get("started_at"),
            "open_questions": row.get("open_questions") or [], "waiting": row.get("waiting") or [],
            "last_line": focus.get("text", ""), "last_line_at": focus.get("authored_at"),
            **{key: row[key] for key in ("focus", "project_admission_hold", "recorded_status", "tails", "unavailable", "stop", "release", "refusal")
               if key in row}})
    from ouroboros.peer_roster import unlisted_live_work

    other = unlisted_live_work(pathlib.Path(root), {row["task_id"] for row in roster["roots"]})
    return {"works": works, "last_line_source": "model_authored_focus",
            "incomplete": roster["incomplete"] or any(row.get("unavailable") for row in other),
            **({"other_live_work": other} if other else {}), "note": _HYGIENE_NOTE,
            **{key: roster[key] for key in ("queue_snapshot", "direct_roots")}}


def context_present(env, task):
    """Capture at head construction, including queued heads with older ingress metadata."""
    root = task.get("budget_drive_root") or getattr(env, "budget_drive_root", None) or env.drive_root
    return live_works_present(root, exclude=str(task.get("id") or ""))


def compact_other_live_work(rows):
    """Count known foreign work; keep unknowns, holds and control facts whole.

    These are not the addressable independent roots. Their exact titles, rooms and
    run IDs remain in the retained runtime capture, before this form can be shown.
    """
    from collections import Counter

    if not isinstance(rows, list):
        return rows
    statuses, tails, exceptions = Counter(), Counter(), []
    running = waiting = 0
    fields = {'work_id', 'title', 'room', 'root_status', 'helpers_running', 'helpers_waiting', 'tails'}
    tail_fields = {'kind', 'state', 'code', 'text', 'count', 'ids'}
    for row in rows:
        if (not isinstance(row, dict) or set(row) - fields or not row.get('work_id')
                or row.get('root_status') not in ('completed', 'failed', 'cancelled', 'running')
                or not isinstance(row.get('tails', []), list)
                or any(not isinstance(row.get(key, 0), int) for key in ('helpers_running', 'helpers_waiting'))):
            exceptions.append(row)
            continue
        statuses[row['root_status']] += 1
        running += row.get('helpers_running', 0)
        waiting += row.get('helpers_waiting', 0)
        unknown = []
        for tail in row.get('tails') or ():
            if (not isinstance(tail, dict) or set(tail) - tail_fields or tail.get('state') != 'live'
                    or tail.get('kind') not in ('helper', 'run', 'review')
                    or not isinstance(tail.get('count', 1), int)
                    or tail.get('code') not in ('delegated_running', 'helpers_running', 'review_work_finishing')):
                unknown.append(tail)
            else:
                tails[tail['code']] += tail.get('count', 1)
        if unknown:
            exceptions.append({'work_id': row['work_id'], 'root_status': row['root_status'], 'tails': unknown})
    return {'count': len(rows), 'counted_root_statuses': dict(statuses),
            'counted_helpers_running': running, 'counted_helpers_waiting': waiting,
            'confirmed_tails': dict(tails), 'unabridged': exceptions,
            'fact': 'Other live work is outside the independent-root roster; an ended root does not confirm its tails ended.',
            'details': 'retained full runtime section: live_works_present.other_live_work'}
