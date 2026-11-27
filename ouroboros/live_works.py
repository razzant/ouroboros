"""One read-only present for new heads, derived from the independent-root roster.

Focus is the model's authored account of its work. It is deliberately distinct
from the web card's latest visible narration and never supplies routing authority.

``works`` holds one row per listed root: its id, title, room, state, ``started_at``,
open questions, waiting and hold facts, ``tails`` (what of it still runs) and focus.
``other_live_work`` is a sibling of ``works``, present only when not empty: ended
roots whose helpers or runs go on (``peer_roster.unlisted_live_work``, from queue and recorded task facts) and unlisted live place turns. It is never a row
field, and ``works`` can be empty while it lists running helpers.

The Runtime section's shorter form (``context_sections.runtime_host_section``) states each work as one line,
``work_line``; the late roster note leaves to the head what its list in view already carries (``head_carried``).
"""
from __future__ import annotations

import json
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
                 "what of it still runs (tails); focus is the work's own dated line, model-authored; "
                 "other_live_work is work whose own root ended while its helpers or external runs go on, and each "
                 "live place conversation turn (kind inline_presence), busy with its own conversation. Keeping this "
                 "list free of work nobody needs any more is ordinary hygiene: close it, ask the owner, or leave it, "
                 "as you judge.")


def live_place_turns(listed, rows, facts):
    """Every live inline Presence turn no listed root is (a moved one is listed), here or in another process, from
    the roster's reading (``resolved_presence_turns``): ``kind="inline_presence"`` facts, not a roster row or direct turn.
    ``tell_task`` can reach a local accepting author. Marks its tree's row."""
    by_id, turns = {row["work_id"]: row for row in rows}, []
    for fact in facts:
        task_id, stored = fact["task_id"], fact["row"]
        if task_id in listed:
            continue
        if task_id in by_id:
            by_id[task_id]["kind"] = "inline_presence"
            continue
        status = stored.get("status") or ("preparing" if fact["state"] == "preparing" else "unknown")
        turns.append({"work_id": task_id, "kind": "inline_presence", "title": str(stored.get("suggested_name") or ""),
                      "room": {"chat_id": stored.get("chat_id"), "project_id": str(stored.get("project_id") or "")},
                      "root_status": str(status), "helpers_running": 0, "helpers_waiting": 0, "tails": [],
                      **({"unavailable": {"task_result": fact["error"]}} if "error" in fact else {})})
    return turns


# In a stop's ``requests``, the value of the entry that only repeats that stop (``_stop_shown``).
OWN_PRESS = "same by, fence_id and generation as this stop"


def _stop_shown(stop):
    """A stop as a head states it: the press that made it, once. ``owner_pause.next_fence`` records that press on the
    fence and again under ``requests[request_id]``; where the fence declares its author and that entry is exactly the
    fence's own ``by``, ``fence_id`` and ``generation``, the entry keeps its key and says so (``OWN_PRESS``) instead
    of repeating them. Anything else stays as recorded: a fence that declares no ``by``, an own entry that differs or
    is absent, every other press with its author, words and date."""
    if not isinstance(stop, dict):
        return stop
    shown, requests, own = dict(stop), stop.get("requests"), stop.get("request_id")
    stated = {key: stop.get(key) for key in ("by", "fence_id", "generation")}
    if (isinstance(requests, dict) and isinstance(own, str) and isinstance(stated["by"], dict) and stated["by"]
            and None not in stated.values() and requests.get(own) == stated):
        shown["requests"] = {**requests, own: OWN_PRESS}
    if isinstance(stop.get("member_stop"), dict):
        shown["member_stop"] = _stop_shown(stop["member_stop"])
    return shown


def work_entry(row):
    """One roster row as ``works`` lists it (the head's capture and the late roster note compare these)."""
    return {"task_id": row["task_id"], "title": row.get("title") or row.get("suggested_name") or "",
            "room": {key: row.get(key) for key in ("chat_id", "project_id", "project_name")},
            "state": row["status"], "started_at": row.get("started_at"),
            "open_questions": row.get("open_questions") or [], "waiting": row.get("waiting") or [],
            **{key: _stop_shown(row[key]) if key == "stop" else row[key]
               for key in ("focus", "project_admission_hold", "recorded_status", "tails", "unavailable", "stop", "release", "refusal")
               if key in row}}


def live_works_present(root, *, exclude=""):
    """Project every host-listed root; stale/missing source observations stay explicit. A root's dated ``focus``
    stands once, whole (no second copy of its text and date beside it), and so does the press of its ``stop``,
    in ``works`` and in ``other_live_work`` alike."""
    from ouroboros.peer_roster import independent_roots

    roster = independent_roots(pathlib.Path(root))
    works = [work_entry(row) for row in roster["roots"] if row["task_id"] != exclude]
    from ouroboros.peer_roster import unlisted_live_work

    listed = {row["task_id"] for row in roster["roots"]}
    # Copies: ``live_place_turns`` marks a row, and the dicts the producer returned stay as it returned them.
    other = [{**row, "stop": _stop_shown(row["stop"])} if "stop" in row else dict(row)
             for row in unlisted_live_work(pathlib.Path(root), listed)]
    other += live_place_turns(listed | {exclude}, other, roster.get("presence_turns") or ())
    return {"works": works,
            "incomplete": roster["incomplete"] or any(row.get("unavailable") for row in other),
            **({"other_live_work": other} if other else {}), "note": _HYGIENE_NOTE,
            "queue_snapshot": roster["queue_snapshot"]}


_HANDLE_FIXED = {"kind": "task_source", "root": "artifact_store"}  # ``focus._safe_handle`` admits no other values
# The shorter Runtime form states, once, how to read the lines that follow its JSON object.
WORKS_LINES_HEADING = "Running works (live_works_present.works), one line each"
WORKS_LINES_LEGEND = (": `- <task_id>` and its facts as key=JSON. chat, project_id and project_name are its room (an "
                      "empty project is left out), started its start; an absent open_questions or waiting is empty; "
                      "focus is its dated focus whole (focus_at its date, focus_source the reader it names), and "
                      "focus_sha256 with focus_size the source it retained, read with get_task_result(task_id=<task_id>, "
                      "include_focus_source=True, focus_source_sha256=<focus_sha256>); every other key stands as the "
                      "row records it.")


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def work_line(work):
    """Every fact of one ``works`` entry on one line, the Runtime section's shorter form of it (owner 26.11.2026,
    D2: the last form the host chooses). Only what the legend restores leaves: the field names' JSON, the room's
    empty project, empty lists, the focus author (always this work: ``peer_roster._compact_root``) and its retained
    source's fixed kind and root and its path, which ``focus_source_sha256`` selects through the reader the legend
    names (the whole row stays in the retained full section). Unknown fields stay whole."""
    facts = dict(work)
    task_id, room, focus = facts.pop("task_id"), facts.pop("room", None), facts.pop("focus", None)
    pairs = [(key, facts.pop(key)) for key in ("state", "title") if key in facts]
    if isinstance(room, dict) and set(room) == {"chat_id", "project_id", "project_name"}:
        pairs += [("chat", room["chat_id"])] + [(key, room[key]) for key in ("project_id", "project_name") if room[key] != ""]
    elif "room" in work:
        pairs.append(("room", room))
    pairs += [("started", facts.pop("started_at"))] if "started_at" in facts else []
    pairs += [(key, value) for key, value in facts.items() if not (key in ("open_questions", "waiting") and value == [])]
    if isinstance(focus, dict):
        focus = dict(focus)
        pairs += [(name, focus.pop(key)) for key, name in (("text", "focus"), ("authored_at", "focus_at"),
                                                           ("source_ref", "focus_source")) if key in focus]
        if focus.get("author_task_id") == task_id:
            focus.pop("author_task_id")
        handle = focus.get("source_handle")
        if (isinstance(handle, dict) and set(handle) == {*_HANDLE_FIXED, "path", "size", "sha256"}
                and all(handle[key] == value for key, value in _HANDLE_FIXED.items())):
            focus.pop("source_handle")
            pairs += [("focus_sha256", handle["sha256"]), ("focus_size", handle["size"])]
        pairs += [(f"focus_{key}", value) for key, value in focus.items()]
    elif "focus" in work:
        pairs.append(("focus", focus))
    return f"- {task_id} " + " ".join(f"{key}={_json(value)}" for key, value in pairs)


def works_lines(works):
    """The text after the shorter Runtime object: the legend once, then one line per work; nothing for none, or
    when an entry is not a work's row (the list then stays whole in the object)."""
    if not works or not all(isinstance(work, dict) and isinstance(work.get("task_id"), str) for work in works):
        return ""
    return f"\n\n{WORKS_LINES_HEADING}{WORKS_LINES_LEGEND}\n" + "\n".join(map(work_line, works))


def head_works(messages):
    """The list of running works the head in view carries, as the mind reads it there: ``(its exact text, {task_id:
    whole entry or line})``, or None when no row in view holds a Runtime section that lists works."""
    marker = "## Runtime context\n\n"
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        for text in [content] if isinstance(content, str) else [
                block.get("text") or "" for block in content or () if isinstance(block, dict)]:
            if marker not in text:
                continue
            start = text.index(marker) + len(marker)
            try:
                runtime, end = json.JSONDecoder().raw_decode(text, start)
            except ValueError:
                return None
            works = (runtime.get("live_works_present") or {}).get("works") if isinstance(runtime, dict) else None
            if isinstance(works, list):
                return _json(works), {work["task_id"]: work for work in works if isinstance(work, dict) and "task_id" in work}
            block = text[end:].split(f"\n\n{WORKS_LINES_HEADING}", 1)
            if not isinstance(works, str) or len(block) < 2:
                return None
            lines = [line for line in block[1].split("\n\n", 1)[0].split("\n")[1:] if line.startswith("- ")]
            return "\n".join(lines), {line.split(" ", 2)[1]: line for line in lines}
    return None


def _owner_held(work):
    """An owner's question, a stop or hold (with its author and words): never left to the head alone."""
    return bool(work["open_questions"] or work.get("stop") or work.get("refusal") or work.get("project_admission_hold")
                or any(isinstance(wait, dict) and wait.get("kind") in ("owner", "owner_pause", "hold")
                       for wait in work["waiting"]))


# The first roster note after a head whose list of running works is in view (``head_base``).
HEAD_INTRO = (" Active independent works beside the list of running works in this context's Runtime context: a root "
              "it lists with the same facts is not repeated (a line here names only what it does not hold), each "
              "other row replaces its entry there; an owner question, stop or hold is always whole.")


def head_base(messages, roster, exclude, shown):
    """The late roster note's base while no note is in view: ``(the head's list text, the displayed rows ``shown``
    (task_id -> rendered row) of the roots that list carries exactly as the roster says now, plus "" for each of its
    roots no longer listed, {carried task_id: its line of what only the note says})``, or None without a visible head
    list (``peer_roster.maybe_append_roster_note``). A work with an owner question, stop or hold is never carried."""
    head = head_works(messages)
    if head is None:
        return None
    text, listed = head
    rows = {row["task_id"]: row for row in roster.get("roots") or () if row["task_id"] != exclude}
    same = {task_id for task_id, row in rows.items() if task_id in listed and not _owner_held(work := work_entry(row))
            and listed[task_id] == (work_line(work) if isinstance(listed[task_id], str) else json.loads(_json(work)))}
    return (text, {**{task_id: shown_row for task_id, shown_row in shown.items() if task_id in same},
                   **dict.fromkeys(set(listed) - set(rows), "")},
            {task_id: f"- {task_id} · {facts}" for task_id in same if task_id in shown
             and (facts := note_only_facts(rows[task_id]))})


def note_only_facts(row):
    """What a roster note says of a root that the head's ``works`` entry does not hold, or ``""``."""
    facts = [f"{key}={_json(row[key])}" for key in ("origin", "lane_wait") if row.get(key)]
    if not row.get("title") and row.get("suggested_name"):
        facts.append(f"suggested_name={_json(row['suggested_name'])}")
    lane = ("live direct conversation" if row.get("direct_chat")
            else "live place conversation turn" if row.get("inline_presence") else "")
    return " · ".join(facts + ([lane] if lane else []))


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
