"""Structural reader facts shared by room intake and addressed question answers.

Queued work remains addressable explicitly but cannot consume ordinary room
input. A live actor waiting for its own answer is still reading; on the question
path so is an unheld cold sleep, which web room intake never treats as a reader.
A place conversation's own work differs (owner's choice): new text there ends its
ordinary wait, a cold sleep's included. No delivery here releases an owner, budget,
restored-process or project admission hold.
"""
from __future__ import annotations

from pathlib import Path


def is_reading(root, task_id: str, task: dict, *, running: bool, fences=None, drain_ended=None, canonical=None) -> bool:
    """``drain_ended``: the caller's own read of the actor's settlement, else read here. ``root`` is the actor's
    drive (its mailbox, its settlement); ``canonical`` the data root whose owner Pause governs it, when they differ."""
    if not running or task.get("_owner_hold") or task.get("_project_admission_hold"):
        return False
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.queue_transitions import budget_pause_fact
    if budget_pause_fact(task, fences) is not None:
        return False
    if fence_closed(read_fence(canonical or root, str(task.get("root_task_id") or task_id))):
        return False
    return not (mailbox_drain_ended(root, task_id) if drain_ended is None else drain_ended)


def unheld_cold_sleep(root, task_id: str, task: dict, *, fences=None, canonical=None) -> bool:
    """A cold sleep, or its own grant awaiting a worker, under no hold, owner Pause or money pause.

    Its question mail wakes it through sleep readiness, whose vetoes still apply;
    a sleep pause without selectors waits for Resume like any interrupted pause.
    """
    pause = task.get("_budget_pause")
    if not isinstance(pause, dict):
        grant = task.get("_budget_pause_resume")
        pause = grant.get("pause") if isinstance(grant, dict) else None
    if not isinstance(pause, dict) or pause.get("reason") != "sleep" or not isinstance(pause.get("sleep"), dict):
        return False
    if task.get("_owner_hold") or task.get("_project_admission_hold"):
        return False
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.owner_pause import fence_closed, read_fence
    from supervisor.queue_transitions import budget_pause_fact
    if budget_pause_fact({k: v for k, v in task.items() if k != "_budget_pause"}, fences) is not None:
        return False  # a hold, or a money fence over its tree
    if fence_closed(read_fence(canonical or root, str(task.get("root_task_id") or task_id))):
        return False
    return not mailbox_drain_ended(root, task_id)


def _published_fences(document: dict) -> dict:
    if document.get("_snapshot_invalid") or any(not isinstance(document.get(key, []), list) for key in ("running", "pending")):
        raise OSError("the published queue cannot be read")
    return {str(row.get("root_task_id") or ""): row for row in document.get("budget_root_fences") or [] if isinstance(row, dict)}


def conversation_works(root, conversation_key: str, binding_ids) -> list:
    """Every live root answering this exact conversation for these bindings, reading or not, each with
    the drive its mailbox lives on: ``{task_id, task, drive, running, attempt, root, fences}`` (``attempt``: the
    dispatcher's published attempt of a running row).

    Read from the published queue document, so a Host and a focus's own worker see the same set without
    the queue lock; every door rechecks ``work_reads`` under that work's mail lock. A snapshot that cannot
    be read proves no absence and raises ``OSError`` (the transport retries; a listing says it is unknown).
    """
    from ouroboros.conversation_lease import root_conversation_key
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.queue_observation import load_queue_document
    from supervisor.queue_timeouts import _task_drive_for_task

    bindings = {str(item) for item in binding_ids if item}
    if not conversation_key or not bindings:
        return []
    document = load_queue_document(root)
    fences = _published_fences(document)
    works = []
    for running, rows in ((True, document.get("running") or []), (False, document.get("pending") or [])):
        for row in rows:
            task = row.get("task") if isinstance(row, dict) else None
            task_id = str(row.get("id") or "") if isinstance(row, dict) else ""
            if (task_id and isinstance(task, dict) and presence_record_binding(task) in bindings
                    and root_conversation_key(task) == conversation_key):
                works.append({"task_id": task_id, "task": task, "running": running, "root": root, "fences": fences,
                              "attempt": int(row.get("attempt") or 1) if running else None,
                              "drive": _task_drive_for_task(task, task_id, canonical=Path(root))})
    return works


def _controlled(work: dict) -> bool:
    """Stop, owner Pause, a hold or money: what governs the work on the canonical root, never its own drive."""
    from ouroboros.cancel_intents import cancel_pending

    task_id, task, root = work["task_id"], work["task"], work["root"]
    if cancel_pending(root, task_id):
        return True
    if work["running"]:
        return not is_reading(work["drive"], task_id, task, running=True, fences=work["fences"], canonical=root)
    return not unheld_cold_sleep(work["drive"], task_id, task, fences=work["fences"], canonical=root)


def work_reads(work: dict, conversation_key: str) -> bool:
    """Whether this live root takes its conversation's new text now: the dispatcher's current attempt's loop with
    its own durable intake open (no fact: preparing or finishing), or an unheld cold sleep, whose vetoes still
    decide its wake. Queued, cancelling, held, owner-paused and money-paused work reads nothing and resumes for nothing."""
    from ouroboros.conversation_lease import author_intake_accepting

    if _controlled(work):
        return False
    return not work["running"] or author_intake_accepting(work["drive"], work["task_id"], conversation_key,
                                                          attempt=work["attempt"])


def work_reads_now(work: dict, conversation_key: str, binding_ids) -> bool:
    """Admission's own read, run inside the work's mail lock: its row as published NOW (attempt, holds, fences), never
    the census that chose it. Only as current as the last publication: a dispatcher's change not yet published is unseen."""
    current = next((row for row in conversation_works(work["root"], conversation_key, binding_ids)
                    if row["task_id"] == work["task_id"]), None)
    return current is not None and current["drive"] == work["drive"] and work_reads(current, conversation_key)


def lent_author_reads(root, author: str, conversation_key: str, binding_ids) -> bool:
    """An author of this conversation parked for its review or private question whose mail wakes it now. All are positive
    facts: its stack's witness (a cold turn with no live author has none), its open intake, its RUNNING row naming this
    conversation and a real yield, and no Stop, owner Pause, hold or money over it. Local and remote alike."""
    from ouroboros.conversation_lease import author_intake_accepting, live_presence_roots
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.task_results import load_task_result

    if (author not in {fact["task_id"] for fact in live_presence_roots(root)["roots"]}
            or not author_intake_accepting(root, author, conversation_key)):
        return False
    row = load_task_result(root, author, strict=True) or {}
    continued = row.get("presence_continuation") if isinstance(row.get("presence_continuation"), dict) else {}
    if (row.get("status") != "running" or not continued.get("lent_at") or continued.get("conversation_key") != conversation_key
            or presence_record_binding(row) not in {str(item) for item in binding_ids if item}):
        return False
    return not lent_author_controlled(root, author)


def lent_author_controlled(root, author: str) -> bool:
    """Stop, owner Pause, a hold or money over a yielded author: it takes nothing, and nothing stands in for it."""
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_results import load_task_result

    if cancel_pending(root, author):
        return True
    row = load_task_result(root, author, strict=True) or {}
    return not is_reading(root, author, {**row, "id": author}, running=True, fences=_published_fences(load_queue_document(root)))


def lent_authors(root, conversation_key: str, exclude: str = "") -> list:
    """The conversation's yielded authors by its own pointer (``open_turns``): candidates only, read by ``lent_author_reads``."""
    from ouroboros.presence_runner import current_previous_turn

    previous, open_turns = current_previous_turn(Path(root), conversation_key, exclude_task_id=exclude)
    named = [str(previous["task_id"])] if (previous or {}).get("continuing") and previous.get("task_id") else []
    return list(dict.fromkeys(item for item in [*named, *(str(row["task_id"]) for row in open_turns if row.get("status") == "running")]
                              if item != exclude))


def work_not_reading_now(work: dict) -> bool:
    """A running, uncontrolled root whose loop is not reading this moment: preparing, or past its final close.
    Its conversation's new text is not plain-heard for that (``place_intake``); queued or controlled work is."""
    return bool(work["running"]) and not _controlled(work)
