"""A published data root whose current facts are constant and whose unrelated history scales with ``k``.

The cost contract (tests/test_cost_contract.py) boots and queries two such roots, ``k=1`` and
``k=30``, and requires every public boundary to do the same work on both. ``build(root, k)``
must therefore keep everything a boundary may legitimately read identical for every ``k``
(ids, sizes, timestamps of the current set) and grow only history nothing interactive needs.

Current facts (constant): three running tasks (a root with a child, and a task of the active
project) as the previous process left them in the queue snapshot; a pending tree of two tasks
(``CURRENT["stop_root"]``, the tree the Stop boundary stops); a paused root whose owner fence is
closed and which holds one open owner question; Main's own recent rows; two project rooms: the
active one with its rows in the OLDEST archive segment, and one bound later to an old task
(``CURRENT["late_task"]``) whose rows are just as old. A finished author keeps one expired,
still-answerable question; its already-delivered result grows with ``k`` while its question
stays fixed, so reading the full body for that question cannot pass the state contract.

Unrelated history (per unit of ``k``): ``LEGACY_PER_K`` frozen ``task_results/*.json`` files
imported at the cutover by the real boot preparation (``startup_migrations.prepare_startup_state``,
which also stamps the one-time migrations as a previous boot did), ``SETTLED_PER_K`` settled tasks with 20-200 KB bodies written through
the store, ``CHILDREN_PER_K`` child drives under ``state/headless_tasks/`` with their own
``current.sqlite``, archive segments of chat/progress/tools/events (``SEGMENTS_PER_K``) whose
rows belong to OTHER rooms and are NEWER than the requested room's rows (a LIMIT cannot hide a
walk), ``CUSTODY_PER_K`` closed delegated-run custody records, one settled task per unit that
still carries an answered, fully delivered owner question. Undelivered answers are CURRENT
facts under ``quiz_needs_retention`` and must not scale with history. ``NEVER_ACTIVE_PROJECTS``
stays constant: a registered project is a current fact (every ``/api/state`` lists it), and one
project that never had a row already sends a boot through the whole growing archive.

The duties' own shapes (the 22.10 class, counted from the owner's snapshot, never census-sized):
CURRENT and constant, three held candidates whose captures await their owner's apply or reject,
each a private copy (a linked worktree under ``<home>/worktrees/exec``, outside the data root, of a
fixture source under ``<home>/sources``) whose named evidence — a read-only Git object, a log and a
symlinked entry (the recorded gap) — was retained: two copies whose records date from before the
one-primitive update keep their checkouts until disposition (the rest law guards real checkouts),
one current copy's checkout went once its capture was saved (its result lives as the patch and
the pinned commits). An evidence-only investigation no longer stays held: it is disposed at its
capture, so its shape is the closed history below. Plus one copy with no custody yet
(``COMPANION_RUN``, whose owner task has no record, so no GC may take it: the driver's positive
companion starts it) and ``PENDING_INVOCATIONS`` start requests whose bodies live in observability
blobs; history per unit
of ``k``: ``CLOSED_RUNS_PER_K`` closed runs that once held such evidence, ``ABSENT_OWNERS_PER_K``
usage owners whose task row is gone (``result_absent``), ``PACKED_PER_K`` settled tasks whose
evidence the real packer packed (sidecars, pack index), ``PLAN_ASKS_PER_K`` ended tasks with a
plan ask still pending a seat, one more day of consciousness money (``CONSCIOUSNESS_PER_K`` own
wakes and the step of a started root still at work today, through the retired usage journal the
store imports once) while the allowance's own last-24-hours window stays CURRENT and constant;
and the subagent metrics cursor is ABSENT while the event chain holds every unit's suffix.

Run in its own interpreter with ``OUROBOROS_DATA_DIR`` set to the root (the writers that use
the process data root then write there): ``python -m tests._scale_root ROOT K OUT.json``. The
build names ``<home>/worktrees`` the process's worktree root, as the measured boot must too; the
meter's execution root is the copies' own folder inside it, so resolving the configured root on a
pass is no walk of a checkout.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

LEGACY_PER_K = 10
SETTLED_PER_K = 10
CHILDREN_PER_K = 3
CUSTODY_PER_K = 5
SEGMENTS_PER_K = {"chat": 1, "progress": 2, "tools": 2, "events": 2}
ROWS_PER_SEGMENT = 150
# A registered project is a current fact (the sidebar lists it on every /api/state), so their
# number stays constant; one never-active project already sends a boot through the archive.
NEVER_ACTIVE_PROJECTS = 1
HELD_RUNS = (("cur-legacy-a", "legacy"), ("cur-legacy-b", "legacy"), ("cur-cand-a", "candidate"))
COMPANION_RUN = "cur-inv-late"  # its copy exists; no custody row until the driver's companion starts it
PENDING_INVOCATIONS = 2
CLOSED_RUNS_PER_K = 2
ABSENT_OWNERS_PER_K = 5
PACKED_PER_K = 2
PLAN_ASKS_PER_K = 2
CONSCIOUSNESS_PER_K = 2  # own wakes per day of history
LONG_ROOT = "cur-long-root"  # a started root at work today whose steps reach back k days
BODY_SIZES = (20_000, 40_000, 80_000, 200_000)
MAIN = 1
FOREIGN_CHATS = (MAIN, 424242, 515151)  # Main and two transport mirrors: never a project room
OLD = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
NEW = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)

CURRENT: Dict[str, Any] = {
    "running": ("cur-run-root", "cur-run-child", "cur-run-p1"),
    "stop_root": "cur-pend-root",
    "pending": ("cur-pend-root", "cur-pend-child"),
    "paused_root": "cur-paused-root",
    "question": ("cur-paused-root", "quiz-open-1"),
    "retained_question": ("cur-finished-author", "quiz-expired-1"),
    "active_project": "p-active",
    "late_project": "p-late",
    "late_task": "hist-late-0",
    "main_rows": 6,
    "room_rows": 24,
}


def _iso(moment: datetime.datetime) -> str:
    return moment.isoformat()


def _stamp(moment: datetime.datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%S")


def _body(size: int, tag: str) -> str:
    unit = f"{tag} result line with ordinary prose about the finished work. "
    return (unit * (size // len(unit) + 1))[:size]


def _write_segment(root: pathlib.Path, prefix: str, moment: datetime.datetime, rows: List[dict]) -> pathlib.Path:
    path = root / "archive" / f"{prefix}_{_stamp(moment)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def _chat_row(moment: datetime.datetime, chat_id: int, index: int, *, task_id: str = "", inbound: bool) -> dict:
    return {"ts": _iso(moment), "session_id": "fixture", "direction": "in" if inbound else "out",
            "chat_id": chat_id, "user_id": 7 if inbound else 0,
            "text": f"message {index} of room {chat_id} with a few words of ordinary conversation",
            "format": "", "source": "web" if inbound else "", "sender_label": "", "sender_session_id": "",
            "client_message_id": f"cm-{chat_id}-{index}-{_stamp(moment)}" if inbound else "",
            "transport": {}, "task_id": task_id}


def _progress_row(moment: datetime.datetime, chat_id: int, task_id: str, index: int) -> dict:
    text = f"progress {index} of {task_id}: reading files and writing the next step"
    return {"ts": _iso(moment), "type": "send_message", "task_id": task_id, "is_progress": True,
            "direction": "out", "chat_id": chat_id, "user_id": 0, "text": text, "content": text,
            "format": "", "role": "assistant", "system_type": "", "narration": False}


def _history_rows(prefix: str, moment: datetime.datetime, unit: int, part: int) -> List[dict]:
    rows = []
    for index in range(ROWS_PER_SEGMENT):
        at = moment + datetime.timedelta(seconds=index)
        chat_id = FOREIGN_CHATS[index % len(FOREIGN_CHATS)]
        task_id = f"hist-settled-{unit}-{index % SETTLED_PER_K}"
        if prefix == "chat":
            rows.append(_chat_row(at, chat_id, index, inbound=index % 2 == 0))
        elif prefix == "progress":
            rows.append(_progress_row(at, chat_id, task_id, index))
        elif prefix == "tools":
            rows.append({"ts": _iso(at), "task_id": task_id, "tool": "read_file",
                         "args": {"path": f"src/module_{index}.py"}, "result_preview": "x" * 120})
        elif index < 2 * CUSTODY_PER_K and part == 0:
            kind = "delegate_run_started" if index % 2 == 0 else "delegate_run_settled"
            rows.append({"ts": _iso(at), "type": kind, "event_id": f"ev-{unit}-{index}",
                         "run_id": f"run-hist-{unit}-{index // 2}", "task_id": task_id,
                         **({"state": "succeeded"} if index % 2 else {})})
        else:
            rows.append({"ts": _iso(at), "type": "llm_round", "task_id": task_id, "round": index,
                         "prompt_tokens": 1000 + index, "completion_tokens": 100, "part": part})
    return rows


def _legacy_document(task_id: str, size: int, moment: datetime.datetime) -> dict:
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION

    return {SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": task_id, "status": "completed",
            "chat_id": FOREIGN_CHATS[len(task_id) % len(FOREIGN_CHATS)], "ts": _iso(moment),
            "updated_at": _iso(moment), "result": _body(size, task_id), "cost_usd": 0.01}


def _legacy_files(root: pathlib.Path, k: int) -> int:
    folder = root / "task_results"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for unit in range(k):
        for index in range(LEGACY_PER_K):
            task_id = f"hist-legacy-{unit}-{index}"
            document = _legacy_document(task_id, BODY_SIZES[index % len(BODY_SIZES)],
                                        OLD + datetime.timedelta(days=unit, minutes=index))
            (folder / f"{task_id}.json").write_text(json.dumps(document), encoding="utf-8")
            count += 1
    return count


def _discharged(root: pathlib.Path, task_id: str) -> Dict[str, Any]:
    """A settled task as history holds it: its terminal projection was published (``attempt``-less
    marker), so it owes no synthesis and no projection; the real writer stores the marker."""
    from ouroboros.task_store import patch_document, read_document

    patch_document(root, task_id, lambda current: {**current, "canonical_terminal_projection": {
        "published_at": current.get("updated_at") or _iso(NEW)}})
    return read_document(root, task_id)


def _settled(root: pathlib.Path, k: int) -> Dict[str, int]:
    """``SETTLED_PER_K`` settled tasks per unit: one template through the real lifecycle writer,
    the rest stored through the store's own transaction with the template's shape (one write
    transaction per unit keeps k=30 fast); child drives are copies of one real child store."""
    import shutil

    from ouroboros import owner_quiz, store
    from ouroboros.task_results import write_task_result

    write_task_result(root, "hist-template", "completed", chat_id=MAIN, result=_body(20_000, "template"),
                      ts=_iso(NEW))
    template = _discharged(root, "hist-template")
    child_template = root / "state" / "headless_tasks" / "hist-template" / "data"
    child_template.mkdir(parents=True, exist_ok=True)
    write_task_result(child_template, "hist-template-sub", "completed", result=_body(20_000, "sub"),
                      parent_task_id="hist-template", root_task_id="hist-template")
    _discharged(child_template, "hist-template-sub")
    children = 1
    for unit in range(k):
        with store.open_store(root, write=True) as txn:
            for index in range(SETTLED_PER_K):
                task_id = f"hist-settled-{unit}-{index}"
                at = _iso(NEW + datetime.timedelta(days=unit, minutes=index))
                txn.store_task({**template, "task_id": task_id, "chat_id": FOREIGN_CHATS[index % len(FOREIGN_CHATS)],
                                "result": _body(BODY_SIZES[index % len(BODY_SIZES)], task_id),
                                "ts": at, "updated_at": at, "canonical_terminal_projection": {"published_at": at}},
                               pending_drive=False)
        for index in range(CHILDREN_PER_K):
            drive = root / "state" / "headless_tasks" / f"hist-settled-{unit}-{index}" / "data"
            shutil.copytree(child_template, drive)
            children += 1
        answered = f"hist-asked-{unit}"
        write_task_result(root, answered, "running", chat_id=MAIN)
        owner_quiz.record_asked(root, answered, quiz_id=f"quiz-{unit}", question="Which one?",
                                options=["A", "B"], chat_id=MAIN)
        owner_quiz.record_publication(root, answered, f"quiz-{unit}", MAIN)
        owner_quiz.record_answered(root, answered, quiz_id=f"quiz-{unit}", option_index=0,
                                   request_id=f"answer-{unit}")
        owner_quiz.mark_discharged(root, answered, f"quiz-{unit}")
        write_task_result(root, answered, "completed", result="Answered and finished.")
        _discharged(root, answered)
    return {"settled": k * (SETTLED_PER_K + 1) + 1, "child_drives": children}


def _custody(root: pathlib.Path, k: int) -> int:
    """Closed delegated-run custody: one run through the real emitter (its closing receipt retired
    as the terminal acknowledgement does), the rest stored as copies of its closed record."""
    from ouroboros import delegate_custody as custody
    from ouroboros import obligations, store

    custody.emit(root, custody.STARTED, {"run_id": "run-hist-template", "task_id": "hist-template"})
    custody.emit(root, custody.SETTLED, {"run_id": "run-hist-template", "task_id": "hist-template",
                                         "state": "succeeded"})
    obligations.remove(root, "delegated_runs", "hist-template")
    count = 0
    with store.open_store(root, write=True) as txn:
        closed = txn.custody_row("run-hist-template")
        for unit in range(k):
            for index in range(CUSTODY_PER_K):
                run_id, task_id = f"run-hist-{unit}-{index}", f"hist-settled-{unit}-{index % SETTLED_PER_K}"
                txn.put_custody(run_id, {**closed, "run_id": run_id, "task_id": task_id}, state="closed")
                count += 1
    return count


def _segments(root: pathlib.Path, k: int) -> Dict[str, int]:
    """Foreign rooms' rows, newer than every row of the requested rooms."""
    written: Dict[str, int] = {}
    for unit in range(k):
        for prefix, per_k in SEGMENTS_PER_K.items():
            for part in range(per_k):
                moment = NEW + datetime.timedelta(days=unit, hours=part * 2 + 1)
                _write_segment(root, prefix, moment, _history_rows(prefix, moment, unit, part))
                written[prefix] = written.get(prefix, 0) + 1
    return written


def _rooms(root: pathlib.Path) -> Dict[str, Any]:
    """The requested rooms' own rows: the oldest segments of the chain (constant for every k)."""
    from ouroboros.contracts.chat_id_policy import project_chat_id

    active_chat = project_chat_id(CURRENT["active_project"])
    late_task = CURRENT["late_task"]
    chat_rows, progress_rows = [], []
    for index in range(CURRENT["room_rows"]):
        at = OLD + datetime.timedelta(minutes=index)
        chat_rows.append(_chat_row(at, active_chat, index, inbound=index % 2 == 0))
        progress_rows.append(_progress_row(at, MAIN, late_task, index))
    _write_segment(root, "chat", OLD, chat_rows)
    _write_segment(root, "progress", OLD, progress_rows)
    return {"active_chat_id": active_chat, "late_chat_id": project_chat_id(CURRENT["late_project"])}


def _projects(root: pathlib.Path, k: int) -> Dict[str, Any]:
    from ouroboros import projects_registry as registry
    from ouroboros.task_results import write_task_result

    registry.create_project(root, CURRENT["active_project"], name="Active project", origin="owner_ui")
    write_task_result(root, CURRENT["late_task"], "completed", chat_id=MAIN, result=_body(20_000, "late"),
                      ts=_iso(OLD))
    _discharged(root, CURRENT["late_task"])
    registry.create_project(root, CURRENT["late_project"], name="Late project", origin="owner_ui")
    registry.bind_task_to_project(root, CURRENT["late_task"], CURRENT["late_project"],
                                  origin={"absent": "post_hoc_unresolved"})
    quiet = NEVER_ACTIVE_PROJECTS
    for index in range(quiet):
        registry.create_project(root, f"p-quiet-{index}", name=f"Quiet {index}")
    return {"never_active_projects": quiet}


def _current(root: pathlib.Path, k: int) -> Dict[str, Any]:
    from ouroboros import budget_pause, owner_pause, owner_quiz
    from ouroboros import projects_registry as registry
    from ouroboros.contracts.chat_id_policy import project_chat_id
    from ouroboros.task_results import write_task_result
    from supervisor import queue, state

    state.update_state(lambda st: st.update(owner_chat_id=MAIN))
    queue.init(root)
    run_root, run_child, run_p1 = CURRENT["running"]
    pend_root, pend_child = CURRENT["pending"]
    tasks = {
        run_root: {"id": run_root, "type": "task", "chat_id": MAIN, "text": "running root", "root_task_id": run_root},
        run_child: {"id": run_child, "type": "task", "chat_id": MAIN, "text": "running child",
                    "parent_task_id": run_root, "root_task_id": run_root, "delegation_role": "subagent", "depth": 1},
        run_p1: {"id": run_p1, "type": "task", "chat_id": project_chat_id(CURRENT["active_project"]),
                 "text": "running in the active project", "root_task_id": run_p1,
                 "project_id": CURRENT["active_project"]},
        pend_root: {"id": pend_root, "type": "task", "chat_id": MAIN, "text": "pending root", "root_task_id": pend_root},
        pend_child: {"id": pend_child, "type": "task", "chat_id": MAIN, "text": "pending child",
                     "parent_task_id": pend_root, "root_task_id": pend_root, "delegation_role": "subagent",
                     "depth": 1},
    }
    registry.bind_task_to_project(root, run_p1, CURRENT["active_project"], origin={"absent": "system"})
    for task_id in CURRENT["running"]:
        task = tasks[task_id]
        write_task_result(root, task_id, "running", chat_id=task["chat_id"], text=task["text"],
                          **{key: task[key] for key in ("parent_task_id", "root_task_id", "project_id") if key in task})
    for task_id in CURRENT["pending"]:
        task = tasks[task_id]
        write_task_result(root, task_id, "scheduled", chat_id=MAIN, text=task["text"],
                          **{key: task[key] for key in ("parent_task_id", "root_task_id") if key in task})
    paused = CURRENT["paused_root"]
    write_task_result(root, paused, "scheduled", chat_id=MAIN, text="paused root", root_task_id=paused)
    fence, _created = owner_pause.install_fence(root, paused, request_id="fixture-pause")
    owner_pause.set_fence_state(root, paused, fence_id=fence["fence_id"], state="paused")
    budget_pause.set_budget_pause(root, paused, {"pause_id": "fixture-pause", "state": "paused",
                                                 "reason": "owner", "rail": owner_pause.RAIL_OWNER_PAUSE})
    asker, quiz_id = CURRENT["question"]
    owner_quiz.record_asked(root, asker, quiz_id=quiz_id, question="Which layout do you prefer?",
                            options=["Rows", "Grid"], chat_id=MAIN, recommended_index=0)
    owner_quiz.record_publication(root, asker, quiz_id, MAIN)
    asker, quiz_id = CURRENT["retained_question"]
    write_task_result(root, asker, "completed", chat_id=MAIN,
                      result=_body(20_000 * k, "already-delivered result"), ts=_iso(OLD))
    _discharged(root, asker)
    owner_quiz.record_asked(root, asker, quiz_id=quiz_id, question="Which follow-up format?",
                            options=["PDF", "HTML"], chat_id=MAIN)
    owner_quiz.record_publication(root, asker, quiz_id, MAIN)
    owner_quiz.mark_published(root, asker, quiz_id)
    owner_quiz.reconcile_terminal(root, asker)
    with queue._queue_lock:
        queue.PENDING[:] = [dict(tasks[pend_root], priority=0, queued_at=time.time()),
                            dict(tasks[pend_child], priority=0, queued_at=time.time())]
        queue.RUNNING.clear()
        for slot, task_id in enumerate(CURRENT["running"], start=1):
            queue.RUNNING[task_id] = {"task": dict(tasks[task_id]), "worker_id": slot, "attempt": 1,
                                      "started_at": time.time()}
    if not queue.persist_queue_snapshot(reason="fixture"):
        raise RuntimeError("the fixture queue snapshot was not published")
    with queue._queue_lock:
        queue.PENDING.clear()
        queue.RUNNING.clear()
    return {"tasks": sorted(tasks), "paused_root": paused}


def _main_rows(root: pathlib.Path) -> None:
    from supervisor import message_bus

    for index in range(CURRENT["main_rows"]):
        message_bus.log_chat("in" if index % 2 == 0 else "out", MAIN, 7 if index % 2 == 0 else 0,
                             f"recent main message {index}", source="web",
                             client_message_id=f"main-recent-{index}" if index % 2 == 0 else "",
                             drive_root=root, require_write=True)


# ---- the duties' own shapes --------------------------------------------------------------------

def _git(area: pathlib.Path, *args: str) -> str:
    done = subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", *args],
                          cwd=area, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def worktree_root(root: pathlib.Path) -> pathlib.Path:
    """The configured worktree root: beside the data root, as an install's is."""
    return pathlib.Path(root).resolve().parent / "worktrees"


def execution_root(root: pathlib.Path) -> pathlib.Path:
    """Where the delegated copies live, inside the configured worktree root."""
    return worktree_root(root) / "exec"


def checkout(exec_root: pathlib.Path, name: str, *, legacy: bool = False) -> Tuple[pathlib.Path, str]:
    """A private copy of a fixture source (``snap-<name>``, owned by ``<name>-task``) whose declared
    evidence folder holds a read-only Git object (copied as Git writes them, 0444), a log, and a
    symlinked entry (a recorded gap). ``legacy``: its record carries ``file_baseline``, as a copy made
    before the one-primitive update does, so its checkout stays until disposition."""
    from ouroboros import subagent_worktrees as worktrees
    from ouroboros.delegate_start_instructions import EVIDENCE_DIRNAME

    source = exec_root.parent.parent / "sources" / name  # beside the worktree root, never inside it
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    (source / "README.md").write_text("fixture checkout\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-q", "-m", "baseline")
    handle = worktrees.provision_execution_snapshot(target_root=source, task_id=f"{name}-task",
                                                    snapshot_id="snap-" + name, worktree_root=exec_root)
    if legacy:
        worktrees._update_record(handle.path, lambda row: {**row, "file_baseline": {}}, op="scale_root")
    area = pathlib.Path(handle.path)
    evidence = area / EVIDENCE_DIRNAME
    loose = next(path for path in (source / ".git" / "objects").glob("??/*") if path.is_file())
    kept = evidence / "objects" / loose.parent.name / loose.name
    kept.parent.mkdir(parents=True)
    shutil.copy2(loose, kept)
    os.chmod(kept, 0o444)
    (evidence / "check.log").write_text("ran the checks\n" * 50, encoding="utf-8")
    os.symlink(os.path.join("..", "README.md"), evidence / "linked-readme")
    return area, handle.baseline_sha


def terminal_detail(run_id: str) -> dict:
    text = f"complete answer of {run_id}"
    return {"summary": {"runId": run_id, "state": "succeeded", "spendUsd": 0.25, "spendEstimated": False,
                        "effectiveAccess": "readonly"}, "lastSeq": 4,
            "primaryOutput": {"text": text, "bytes": len(text.encode()), "truncated": False}}


def held_run(root: pathlib.Path, run_id: str, kind: str, area: pathlib.Path, sha: str) -> Dict[str, Any]:
    """One terminal candidate the previous generation collected: output retained, money settled, patch
    captured, named evidence retained (the symlink a recorded gap), undisposed, so it stays HELD until its
    owner applies or rejects it. A legacy copy keeps its checkout; a current one is released."""
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_collection import collect_terminal
    from ouroboros.subagent_worktrees import find_execution_snapshot

    entry = custody.RunCustody(
        run_id=run_id, task_id=f"{run_id}-task", root_task_id=f"{run_id}-task", invocation_id="inv-" + run_id,
        route_id="test", model="test", access="readonly", mode="ask", ledger_root=str(root),
        snapshot_id="snap-" + run_id, execution_root=str(area), baseline_sha=sha,
        target_root=str(find_execution_snapshot("snap-" + run_id)["target_root"]), authority_source="preflight_head")
    if not custody.record_started(root, entry):
        raise RuntimeError(f"custody of {run_id} did not land")
    collected = collect_terminal(root, run_id, detail=terminal_detail(run_id))
    row = current.run(root, run_id)
    capture = collected.get("workspace_capture") or {}
    named = capture.get("named_evidence") or {}
    released = kind != "legacy"
    if not (row.terminal_source and row.settled and row.output_complete and row.patch_captured
            and not row.patch_disposed and named.get("status") == "retained" and named.get("gaps")
            and bool(capture.get("checkout_released")) is released and area.exists() is not released):
        raise RuntimeError(f"{run_id} is not a held {kind} candidate: {collected}")
    return {"run_id": run_id, "task_id": entry.task_id, "kind": kind, "area": str(area), "baseline_sha": sha}


def _settled_task(root: pathlib.Path, task_id: str, **fields: Any) -> None:
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, "completed", chat_id=MAIN, result=_body(20_000, task_id), **fields)
    _discharged(root, task_id)


GLOBAL_LIMIT_USD = 100.0  # the k=1 and k=30 roots reserve under this limit (measured so; unchanged)
LIMIT_REFERENCE_K = 30


def spend_limit_usd(k: int) -> float:
    """The global limit the synthetic money is reserved under: the k=30 root's $100, or proportionally more
    for a larger root (every unit settles ``ABSENT_OWNERS_PER_K`` attempts: the known spend grows with the
    shape, and a fixed limit refused the k=100 root's reservations once it reached $100)."""
    return GLOBAL_LIMIT_USD * max(1.0, k / LIMIT_REFERENCE_K)


def money(root: pathlib.Path, task_id: str, *, category: str = "", cost: float = 0.4,
          limit: float = GLOBAL_LIMIT_USD) -> None:
    """One settled attempt of ``task_id`` through the real ledger: the task becomes a dirty usage owner
    and the ledger's publication marker moves (also the driver's ``money`` activity, tests/_cost_driver.py)."""
    from ouroboros import usage_accounting as usage

    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="test", provider="openai", drive_root=root, task_id=task_id, root_task_id=task_id,
        reservation_usd=1.0, global_limit_usd=limit, category=category))
    usage.mark_dispatched(reservation)
    usage.settle_attempt(reservation, {"prompt_tokens": 7}, cost_usd=cost, cost_final=True)


def _delegated(root: pathlib.Path, k: int) -> Dict[str, Any]:
    """The held runs (current), their closed predecessors (history) and the pending start requests."""
    from ouroboros import delegate_custody as custody
    from ouroboros import store

    from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY

    exec_root = execution_root(root)
    os.environ["OUROBOROS_SUBAGENT_WORKTREE_ROOT"] = str(worktree_root(root))  # copies release only inside it
    held = []
    for run_id, kind in HELD_RUNS:
        _settled_task(root, f"{run_id}-task")
        held.append(held_run(root, run_id, kind, *checkout(exec_root, run_id, legacy=kind == "legacy")))
    # No task record for the companion's owner: an owner not proven over keeps its copy from every GC.
    area, sha = checkout(exec_root, COMPANION_RUN)
    companion = {"run_id": COMPANION_RUN, "task_id": f"{COMPANION_RUN}-task", "area": str(area), "baseline_sha": sha}
    pending = []
    for index in range(PENDING_INVOCATIONS):
        task_id = f"cur-pending-inv-{index}-task"
        _settled_task(root, task_id)
        invocation_id = f"inv-pending-{index}"
        if not custody.record_start_requested(
                root, invocation_id=invocation_id, task_id=task_id, root_task_id=task_id, surface="delegate",
                route="test", category="subagent", source="delegated_subagent", idempotency_key=f"idem-{index}",
                request={"prompt": f"pending start {index} " + "p" * 4000, "route": "test"}):
            raise RuntimeError("the pending start request did not land")
        pending.append(invocation_id)
    with store.open_store(root, write=True) as txn:
        template = txn.custody_row(HELD_RUNS[0][0])
        count = 0
        for unit in range(k):
            for index in range(CLOSED_RUNS_PER_K):
                run_id = f"run-closed-{unit}-{index}"
                txn.put_custody(run_id, {**template, "run_id": run_id, "task_id": f"hist-settled-{unit}-{index}",
                                         "root_task_id": f"hist-settled-{unit}-{index}",
                                         "authority_source": INVESTIGATION_AUTHORITY,
                                         "patch_disposed": custody.EVIDENCE_RETAINED}, state="closed")
                count += 1
    return {"execution_root": str(exec_root), "held_runs": held, "companion": companion,
            "pending_invocations": pending, "closed_evidence_runs": count}


def _usage_owners(root: pathlib.Path, k: int) -> Dict[str, int]:
    """Per unit: owners whose task row is gone (``result_absent``), and the settled rows of the day's wakes."""
    from tests._store_shared import delete_result

    absent = 0
    for unit in range(k):
        for index in range(ABSENT_OWNERS_PER_K):
            task_id = f"hist-absent-{unit}-{index}"
            _settled_task(root, task_id)
            money(root, task_id, limit=spend_limit_usd(k))
            delete_result(root, task_id)
            absent += 1
        for index in range(CONSCIOUSNESS_PER_K):
            _settled_task(root, f"hist-wake-{unit}-{index}", ts=_iso(OLD))  # history: older than current work
    return {"absent_owners": absent}


def _journal_rows(attempt_id: str, ts: float, root_task_id: str, task_id: str, category: str, cost: float = 0.01) -> List[dict]:
    """One settled attempt in the retired journal format the store's one-time import reads."""
    base = {"attempt_id": attempt_id, "kind": "attempt", "model": "test", "provider": "openai", "category": category,
            "source": "test", "task_id": task_id, "root_task_id": root_task_id, "parent_task_id": "",
            "ts": datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()}
    return [{**base, "state": "reserved", "reservation_upper_bound_usd": "1"}, {**base, "state": "dispatched"},
            {**base, "state": "settled", "cost_usd": str(cost), "cost_final": True, "pricing_known": True}]


def _consciousness_journal(root: pathlib.Path, k: int) -> int:
    """Consciousness money through the retired journal the usage store imports once, when it is first
    opened (before any money is settled on this root). CURRENT and constant: the last 24 hours — two own
    wakes, one started root, the long-lived root's row of today. Per unit of ``k``, one day further back
    (outside the allowance's window): ``CONSCIOUSNESS_PER_K`` own wakes and the long-lived root's step of
    that day. The allowance read sums the window; the older days are history it must not visit."""
    from tests._usage_store_testing import write_journal

    now, day, rows, roots = time.time(), 86_400.0, [], 0
    for wake in range(2):
        rows += _journal_rows(f"cur-wake-{wake}", now - 3600 * (wake + 1), f"cur-wake-{wake}", f"cur-wake-{wake}", "consciousness")
    rows += _journal_rows("cur-started", now - 7200, "cur-started", "cur-started", "consciousness_task")
    rows += _journal_rows("cur-long-today", now - 1800, LONG_ROOT, LONG_ROOT, "consciousness_task")
    for unit in range(k):
        at = now - (unit + 1) * day - 3 * 3600
        for index in range(CONSCIOUSNESS_PER_K):
            wake = f"hist-wake-{unit}-{index}"
            rows += _journal_rows(wake, at - 60 * index, wake, wake, "consciousness")
            roots += 1
        rows += _journal_rows(f"hist-long-{unit}-0", at - 10, LONG_ROOT, f"hist-long-step-{unit}", "consciousness_task")
        rows += _journal_rows(f"hist-long-{unit}-1", at - 20, LONG_ROOT, f"hist-long-step-{unit}", "task")
    write_journal(root, rows)
    return roots


def _packed(root: pathlib.Path, k: int) -> Dict[str, Any]:
    """Per unit: settled tasks with call evidence and a checkpoint handle, packed by the real packer."""
    from ouroboros import evidence_packer
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.observability import write_blob

    for unit in range(k):
        for index in range(PACKED_PER_K):
            task_id = f"hist-packed-{unit}-{index}"
            calls = root / "observability" / "calls" / task_id
            calls.mkdir(parents=True, exist_ok=True)
            for call in range(2):
                ref = write_blob(root, {"message": {"content": f"answer {task_id} {call} " + "x" * 4000}}, kind="json")
                (calls / f"llm_{call}_response.json").write_text(json.dumps(
                    {"call_id": f"llm_{call}", "task_id": task_id, "full_payload_ref": ref}), encoding="utf-8")
            store_actor_source_bytes(root, task_id, category="context_checkpoints", source_id=f"focus_{task_id}",
                                     data=b"# focus\n" + b"y" * 3000, extension="md")
            _settled_task(root, task_id, ts=_iso(OLD))  # history: older than current work
    report = evidence_packer.run_pass(root)
    return {"packed": report["packed"], "faults": report["faults"]}


def _plan_asks(root: pathlib.Path, k: int) -> int:
    """Per unit: ended tasks whose plan record still holds a pending seat (the reconcile pass's collector)."""
    from ouroboros import store
    from ouroboros.plan_record import PLAN_RECORD_KEY, PLAN_RECORD_VERSION
    from ouroboros.task_store import read_document

    template = read_document(root, "hist-template")
    count = 0
    with store.open_store(root, write=True) as txn:
        for unit in range(k):
            for index in range(PLAN_ASKS_PER_K):
                task_id = f"hist-plan-{unit}-{index}"
                at = _iso(NEW + datetime.timedelta(days=unit, hours=12, minutes=index))
                record = {"schema_version": PLAN_RECORD_VERSION, "attempts": [],
                          "revisions": [{"n": 1, "goal": "finish", "plan": "do it", "done_when": [], "affected_paths": [],
                                         "ts": at}],
                          "asks": [{"ask_id": f"ask-{unit}-{index}", "revision": 1, "ts": at, "participants": [
                              {"seat_id": "seat-1", "status": "pending", "operation_id": f"op-{unit}-{index}",
                               "owner_marked": True, "dispatched": True, "model": "test", "route": "api_chat"}]}]}
                txn.store_task({**template, "task_id": task_id, "chat_id": MAIN, "ts": at, "updated_at": at,
                                "canonical_terminal_projection": {"published_at": at}, PLAN_RECORD_KEY: record},
                               pending_drive=False)
                count += 1
    return count


def _forget_metrics_cursor(root: pathlib.Path) -> None:
    """The producer boundary's replay shape: no cursor while the chain holds every unit's suffix."""
    from ouroboros import store
    from ouroboros.subagent_metrics import CURSOR

    with store.open_store(root, write=True) as txn:
        txn.conn.execute("DELETE FROM meta WHERE key=?", (CURSOR,))


def build(root: Any, k: int) -> Dict[str, Any]:
    """Build the published root (see the module docstring); returns its facts."""
    from ouroboros.startup_migrations import prepare_startup_state
    from supervisor import state

    started = time.monotonic()
    root = pathlib.Path(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "state").mkdir(parents=True, exist_ok=True)
    state.init(root, 100.0)
    state.init_state()  # a first boot's state, before any history exists
    # No fallback model: the turn law (B10) ends a turn at its first model request, which the driver refuses.
    (root / "settings.json").write_text(json.dumps({"TOTAL_BUDGET": 100, "OUROBOROS_MAX_WORKERS": 2,
                                                    "OUROBOROS_MODEL_FALLBACKS": []}), encoding="utf-8")
    facts: Dict[str, Any] = {"k": k}
    steps: Dict[str, float] = {}
    facts["consciousness_roots"] = _consciousness_journal(root, k)  # before the first money write opens the store

    def step(name: str, call):
        begun = time.monotonic()
        value = call()
        steps[name] = round(time.monotonic() - begun, 2)
        return value

    facts["legacy_files"] = step("legacy_files", lambda: _legacy_files(root, k))
    # The real boot preparation (import, publication, migration stamps), as the install's previous boots ran it.
    facts["import"] = step("import", lambda: prepare_startup_state(root)["status"])
    facts.update(step("settled", lambda: _settled(root, k)))
    facts["custody_runs"] = step("custody", lambda: _custody(root, k))
    facts["segments"] = step("segments", lambda: _segments(root, k))
    facts.update(step("rooms", lambda: _rooms(root)))
    facts.update(step("projects", lambda: _projects(root, k)))
    facts.update(step("current", lambda: _current(root, k)))
    step("main_rows", lambda: _main_rows(root))
    facts.update(step("delegated", lambda: _delegated(root, k)))
    facts.update(step("usage_owners", lambda: _usage_owners(root, k)))
    facts["packs"] = step("packed", lambda: _packed(root, k))
    facts["plan_asks"] = step("plan_asks", lambda: _plan_asks(root, k))
    step("metrics_cursor", lambda: _forget_metrics_cursor(root))
    facts["build_seconds"] = round(time.monotonic() - started, 2)
    facts["build_steps"] = steps
    return facts


def main(argv: List[str]) -> int:
    root, k, out = pathlib.Path(argv[0]), int(argv[1]), pathlib.Path(argv[2])
    if pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR", "")).resolve() != root.resolve():
        raise SystemExit("OUROBOROS_DATA_DIR must name the root being built")
    facts = build(root, k)
    out.write_text(json.dumps(facts, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
