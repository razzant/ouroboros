"""The live set: what a server answers for at one moment, grown on a booted root, and every recurring pass over it.

The cost contract's first axis is history (tests/_scale_root.py, k=1 vs k=30). Its second, here, is the live set:
running roots, owner-paused trees, pending rows, held custody runs, schedule rows, reconcile markers. The saturation of
01.11 grew along this axis while every law held: 28 running, 20 paused trees and 2,252 markers cost the server's one
interpreter the whole of a core, on a fixture with no fence in the queue at all. tests/_cost_driver.py runs ``measure``
in the booted driver process (``plan: live_set``), with the supervisor held before its first tick, and
tests/test_cost_live_set.py holds the counts it leaves.

``measure`` lays the smallest set (``STAGES[0]``: one of each), measures every pass over it, then grows one axis at a
time and measures every pass again: each stage's counts differ from the stage before by that axis alone. A pass is
one duty as the supervisor tick starts it (``tick_pass``: its due condition and its latch are production's, the tick's
start runs it here on this thread), so a pass the tick no longer starts costs the tick's look at its condition; the
lifespan's reconcile pickup; the delegated body's tick; one root's heartbeat as the drain handles it; ``GET /api/state``;
one queue publication; the assignment tick's queue hygiene; the timeout pass. Every pass runs until it writes nothing
(``SETTLE_PASSES``), then once more: ``rest``, the pass over an unchanged live set that the law holds. Everything is
laid through the real writers: an owner Pause is ``request_owner_pause``, a marker is ``request_extension_reconcile``,
a held run is a stored custody row of the shape the build's real collection left. Only a RUNNING row has no process
behind it (the driver spawns no worker), as in the turn law's breadth step (tests/_turn_shapes.py).

The six companions the duty table once named as not built (``companions``) are here too, because each of them needs
what only a live set has: a fence, a saved cold request, a held Continue, a due schedule, an owed notice, an owed
synthesis.
"""

from __future__ import annotations

import pathlib
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, NamedTuple

from tests._cost_driver import _body_pass, _duty_rows, _event_ctx, _measured_pass, _summary, refused, tagged

MAIN = 1
SETTLE_PASSES = 3  # a pass whose first run moved its own inputs: bounded by this transport, never a judgment
# One owner message by value on every row, as the 46,277-byte message rode 20 rows of the 01.11 queue document (R3 §C.2).
OWNER_NEEDLE = "the owner's one long message of this live set"  # once in the text, so its copies can be counted
OWNER_TEXT = (OWNER_NEEDLE + ": " + "the long brief for the campaign, with every issue named in full. " * 800)[:46_277]
BODY_BYTES = 40_000  # a task document's size class (75-106 KB on the owner's install, rounded down to stay quick)


class Shape(NamedTuple):
    running: int  # roots at work
    paused: int  # owner-paused trees
    members: int  # members of those trees, roots included
    pending: int  # rows waiting, outside the paused trees
    held: int  # held custody runs ...
    settled: int  # ... of which settled, awaiting their disposition
    schedules: int  # schedule rows, five of them enabled
    markers: int  # reconcile markers over ``SKILLS`` skills


SKILLS = 2  # the markers of a reload name few skills, many times (2,252 markers over 51 skills on 01.11)
SMALLEST = Shape(running=1, paused=1, members=4, pending=1, held=1, settled=1, schedules=1, markers=SKILLS)
# The owner's numbers of 01.11 (PLAN_BASE §(e)), except the markers: the law needs their growth, not their 2,252.
AXIS = Shape(running=50, paused=20, members=86, pending=61, held=122, settled=113, schedules=35, markers=120)
FULL = AXIS._replace(markers=2_000)  # the stand's production shape
STAGES = ("smallest", "running", "paused", "pending", "held", "schedules", "markers")  # each grows one axis over the last


def _body(tag: str) -> str:
    unit = f"{tag} worked on the campaign and recorded what it found. "
    return (unit * (BODY_BYTES // len(unit) + 1))[:BODY_BYTES]


def _task(task_id: str, **fields: Any) -> Dict[str, Any]:
    return {"id": task_id, "type": "task", "chat_id": MAIN, "text": f"live work {task_id}", "root_task_id": task_id,
            "priority": 0, "queued_at": time.time(), "metadata": {"origin_message_text": OWNER_TEXT}, **fields}


def grow_running(root: pathlib.Path, have: int, want: int) -> Dict[str, Any]:
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    for index in range(have, want):
        task_id = f"live-run-{index:03d}"
        write_task_result(root, task_id, "running", chat_id=MAIN, text=f"live work {task_id}", root_task_id=task_id,
                          result=_body(task_id))
        with queue._queue_lock:
            queue.RUNNING[task_id] = {"task": _task(task_id), "worker_id": 1000 + index, "attempt": 1,
                                      "started_at": time.time(), "last_heartbeat_at": time.time()}
            queue.persist_queue_snapshot(reason="live_set")
    return {"running": want}


def grow_paused(root: pathlib.Path, have: int, want: int, members: int) -> Dict[str, Any]:
    """Whole trees through the owner's own door: a root and a child still pending, the rest of the tree finished
    (its strict bodies are what the census reads), then ``request_owner_pause``, which settles what it can."""
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause

    states, trees, laid = {}, want - have, 0
    per_tree, more = divmod(members, trees)
    for index in range(have, want):
        root_id = f"live-paused-{index:03d}"
        size = per_tree + (index - have < more)
        laid += size
        write_task_result(root, root_id, "scheduled", chat_id=MAIN, text=f"paused root {root_id}", root_task_id=root_id,
                          result=_body(root_id))
        child = f"{root_id}-child-0"
        write_task_result(root, child, "scheduled", chat_id=MAIN, text=f"paused child of {root_id}", root_task_id=root_id,
                          parent_task_id=root_id, delegation_role="subagent")
        for number in range(1, size - 1):
            write_task_result(root, f"{root_id}-child-{number}", "completed", chat_id=MAIN, result=_body(root_id),
                              root_task_id=root_id, parent_task_id=root_id, delegation_role="subagent")
        with queue._queue_lock:
            queue.PENDING.extend([_task(root_id), _task(child, root_task_id=root_id, parent_task_id=root_id,
                                                        delegation_role="subagent", depth=1)])
        answer = request_owner_pause(root_id, request_id=f"live-pause-{index:03d}")
        if not answer.get("ok"):
            raise RuntimeError(f"the owner's Pause of {root_id} was refused: {answer}")
        states[root_id] = refresh_owner_pause_tree(root_id)
    with queue._queue_lock:
        fences = sum(fence.get("cause") == "owner_pause" for fence in queue.BUDGET_ROOT_FENCES.values())
    return {"trees": trees, "members": laid, "fences_in_queue": fences, "states": sorted(set(states.values()))}


def grow_pending(root: pathlib.Path, have: int, want: int) -> Dict[str, Any]:
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    for index in range(have, want):
        task_id = f"live-pending-{index:03d}"
        write_task_result(root, task_id, "scheduled", chat_id=MAIN, text=f"live work {task_id}", root_task_id=task_id,
                          result=_body(task_id))
        with queue._queue_lock:
            queue.PENDING.append(_task(task_id))
            queue.persist_queue_snapshot(reason="live_set")
    return {"pending": want}


def grow_held(root: pathlib.Path, have: int, want: int, settled: int, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Held custody runs: copies of a real held candidate's row (the build's collection wrote it, settled, its patch
    awaiting disposition, its collection attempt recorded over these inputs, so the body has nothing owed), each with
    its finished owner task, and runs still open, whose rows the real emitter writes."""
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from tests._scale_root import _settled_task

    template = facts["held_runs"][0]["run_id"]
    closed = [f"live-held-{index:03d}" for index in range(have, min(want, settled))]
    for run_id in closed:
        _settled_task(root, f"{run_id}-task")
    with store.open_store(root, write=True) as txn:
        row = txn.custody_row(template)
        attempt = row.get("collection_attempt") or {}
        for run_id in closed:
            txn.put_custody(run_id, {
                **row, "run_id": run_id, "task_id": f"{run_id}-task", "root_task_id": f"{run_id}-task",
                "collection_attempt": {**attempt, "inputs": {**(attempt.get("inputs") or {}), "run_id": run_id}}}, state="held")
    for index in range(max(have, settled), want):
        run_id = f"live-open-{index:03d}"
        if not custody.emit(root, custody.STARTED, {"run_id": run_id, "invocation_id": f"inv-{run_id}",
                                                    "task_id": f"{run_id}-task", "root_task_id": f"{run_id}-task",
                                                    "route": "route", "source": "delegate_start", "access": "readonly"}):
            raise RuntimeError(f"the open run {run_id} did not land")
    return {"held": want, "settled": min(want, settled)}


def schedule_rows(have: int, want: int) -> List[Dict[str, Any]]:
    """Idle rows of the table's shape: the last five enabled and far from due, the rest finished, their outcome cached."""
    rows = []
    for index in range(have, want):
        schedule_id, last = f"live-schedule-{index:03d}", f"live-schedule-run-{index:03d}"
        rows.append({"id": schedule_id, "name": f"Follow-up {index}", "enabled": want - index <= 5,
                     "source": "task_followup", "trigger": {"type": "cron", "expr": "0 0 1 1 *"},
                     "next_run_at": "2099-01-01T00:00:00+00:00", "last_task_id": last,
                     "followup_origin": {"task_id": "origin", "root_task_id": "origin"},
                     "last_outcome": {"task_id": last, "state": "completed", "quiet": False, "notice": False,
                                      "ended_at": None, "line": ""},
                     "task": {"type": "task", "text": f"continue {schedule_id}", "chat_id": MAIN, "metadata": {}}})
    return rows


def grow_schedules(root: pathlib.Path, have: int, want: int) -> Dict[str, Any]:
    from supervisor.queue_schedules import _write_scheduled_tasks, load_schedule_store, schedule_transaction

    with schedule_transaction(root):
        data = load_schedule_store(root)
        data["tasks"] = [*(data.get("tasks") or []), *schedule_rows(have, want)]
        _write_scheduled_tasks(data, root)
    return {"schedules": want}


def grow_markers(root: pathlib.Path, want: int) -> Dict[str, Any]:
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests, request_extension_reconcile

    for index in range(want):
        request_extension_reconcile(root, f"live-skill-{index % SKILLS}", reason="live set", source="worker")
    return {"requested": want, "markers": len(list_extension_reconcile_requests(root))}


def grow(stage: str, root: pathlib.Path, shape: Shape, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Lay one stage: the smallest set, or one axis up to its size in ``shape``."""
    if stage == "smallest":
        return {"running": grow_running(root, 0, SMALLEST.running),
                "paused": grow_paused(root, 0, SMALLEST.paused, SMALLEST.members),
                "pending": grow_pending(root, 0, SMALLEST.pending),
                "held": grow_held(root, 0, SMALLEST.held, SMALLEST.settled, facts),
                "schedules": grow_schedules(root, 0, SMALLEST.schedules),
                "markers": grow_markers(root, SMALLEST.markers)}
    return {"running": lambda: grow_running(root, SMALLEST.running, shape.running),
            "paused": lambda: grow_paused(root, SMALLEST.paused, shape.paused, shape.members - SMALLEST.members),
            "pending": lambda: grow_pending(root, SMALLEST.pending, shape.pending),
            "held": lambda: grow_held(root, SMALLEST.held, shape.held, shape.settled, facts),
            "schedules": lambda: grow_schedules(root, SMALLEST.schedules, shape.schedules),
            "markers": lambda: grow_markers(root, shape.markers)}[stage]()


# ---- the passes ----------------------------------------------------------------------------------------------------

def tick_pass(sm: Any, name: str, stop: threading.Event) -> None:
    """One supervisor maintenance tick with every cadence due, which starts ``name`` on this thread if its own due
    condition says so; every other duty it would start is left: its latch goes back, its thread does not start."""
    def start(latch, started, target, args):
        if started == name:
            target(*args)  # the pass releases its own latch
        else:
            latch.release()
        return True

    real = sm._start_maintenance_thread, sm._tick_maintenance_child
    sm._start_maintenance_thread = start
    if name != "maintenance-child":
        sm._tick_maintenance_child = lambda: None
    sm._LAST_CANCEL_INTENT_SWEEP[0] = sm._LAST_COLD_CUSTODY_SWEEP[0] = 0.0
    try:
        sm._periodic_supervisor_maintenance([0.0], [0.0], stop_event=stop)
    finally:
        sm._start_maintenance_thread, sm._tick_maintenance_child = real


def _passes(meter: Any, srv: Any, client: Any, root: pathlib.Path, stubs: Dict[str, Any],
            stop: threading.Event) -> Dict[str, Callable[[], Any]]:
    from ouroboros import server_maintenance as sm
    from ouroboros import utils
    from ouroboros.config import load_settings
    from ouroboros.delegate_collection import body_runtime
    from ouroboros.extension_reconcile_queue import process_extension_reconcile_requests
    from supervisor import queue
    from supervisor.events_worker_reports import _handle_task_heartbeat
    from supervisor.queue_timeouts import enforce_task_timeouts
    from supervisor.worker_assignment import assign_tasks

    # A file read within a clock tick of its write cannot prove its identity (``utils._RACY_NS``): bounded by time, not by the
    # set, it would make a rest depend on the milliseconds since the stage's last write. The stages run without that window.
    utils._RACY_NS = 0
    runtime, park = body_runtime(root), stubs["park"]
    ctx = _event_ctx(srv)
    probe = "live-run-000"

    def state() -> None:
        from tests._cost_meter import tag_of

        client.get("/api/state", headers={"x-cost-boundary": tag_of()})

    table: Dict[str, Callable[[], Any]] = {
        duty.name: (lambda duty=duty: tick_pass(sm, duty.name, stop)) for duty in _duty_rows(sm)}
    table.update({
        "delegated-body": lambda: _body_pass(runtime, park.tick),
        "extension-reconcile": lambda: process_extension_reconcile_requests(root, load_settings),
        "heartbeat": lambda: _handle_task_heartbeat(_beat(probe), ctx),
        "state": state,
        "publication": lambda: _publish_as_the_tick_does(queue),
        "assign": assign_tasks,
        "timeouts": lambda: enforce_task_timeouts(pending_events=()),
    })
    return table


def _beat(task_id: str) -> Dict[str, Any]:
    """A worker's heartbeat as it reaches the drain: stamped when it was spoken."""
    from ouroboros.utils import utc_now_iso

    return {"task_id": task_id, "phase": "running", "root_task_id": task_id, "ts": utc_now_iso()}


def _publish_as_the_tick_does(queue: Any) -> Any:
    """The tick's one publication: its publisher where queue changes are published by revision, else its call."""
    publish = getattr(queue, "publish_queue_changes", None)
    return publish() if publish is not None else queue.persist_queue_snapshot(reason="main_loop")


def _before_rest(name: str, stubs: Dict[str, Any]) -> None:
    """What makes the pass due again without changing its inputs: the body's retry clock, a requested child look, the
    census's per-root recheck window (a pass inside that window reads nothing, whatever it would read after it)."""
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.maintenance import child
    from supervisor import owner_pause_control

    if name == "delegated-body":
        stubs["clock"].now += BODY_MAINTENANCE_INTERVAL_SEC + 1.0
    elif name == "maintenance-child":
        child.request()
    elif name == "owner_pause_settlement":
        getattr(owner_pause_control, "_LAST_SETTLE_CHECK", {}).clear()


def _one_stage(meter: Any, stage: str, table: Dict[str, Callable[[], Any]], root: pathlib.Path, stubs: Dict[str, Any],
               stop: threading.Event, result: Dict[str, Any]) -> Dict[str, Any]:
    """Every pass at ``stage``: its first run, the settling runs while it wrote, its rest. ``tags`` names each run in
    order: what a run's maintenance child reported stays under its tag (``child_reports``, ``_drain``), and a law that
    reads the count reads that too (tests/test_cost_live_set.py ``judged_pass``)."""
    from ouroboros import server_maintenance as sm

    measured: Dict[str, Any] = {}
    for name, one in table.items():
        started = time.monotonic()
        tags = [f"L:{stage}:{name}:first"]
        _before_rest(name, stubs)
        first = _measured_pass(meter, tags[-1], one, sm, root, stop, result)
        settled = first
        for attempt in range(SETTLE_PASSES):
            if not settled["counts"].get("writes"):
                break
            _before_rest(name, stubs)
            tags.append(f"L:{stage}:{name}:settle{attempt}")
            settled = _measured_pass(meter, tags[-1], one, sm, root, stop, result)
        _before_rest(name, stubs)
        tags.append(f"L:{stage}:{name}:rest")
        rest = _measured_pass(meter, tags[-1], one, sm, root, stop, result)
        measured[name] = {"rest": rest, "first": first, "seconds": round(time.monotonic() - started, 3), "tags": tags}
    if stage == STAGES[0]:
        # The smallest set's rest again, once every pass has met it: a pass the table runs before another's first
        # run (the skills' state the extension reconcile writes, the schedule's wake) would count that run's
        # effect as the next axis's growth. Its first runs stay the markers law's baseline; its rest is this one,
        # which answers for every run before it.
        for name, entry in _one_stage(meter, f"{stage}-steady", table, root, stubs, stop, result).items():
            measured[name]["rest"] = entry["rest"]
            measured[name]["tags"] += entry["tags"]
    return measured


def _previous_process_work_gone(root: pathlib.Path) -> None:
    """The tasks the build left ``running`` have no process in this measurement: their orphan reconcile would fall due
    by the clock, at whichever stage's pass the grace ran out in, and its owed answer would read as that stage's growth.
    They are no part of the set this axis grows."""
    from tests._scale_root import CURRENT
    from tests._store_shared import delete_result

    for task_id in CURRENT["running"]:
        delete_result(root, task_id)


def _child_jobs_done(meter: Any, root: pathlib.Path, result: Dict[str, Any]) -> None:
    """What the install's maintenance child would have run by now, so that no stage's pass pays for it. Each run's own
    report is read as the cost contract's preparation reads it (``refused``): a run that refused did nothing, and every
    stage would then be measured on an install whose maintenance never ran (#2465), so the driver fails by name."""
    from ouroboros import maintenance

    for _attempt in range(3):
        if not maintenance.due(root):
            break
        report = tagged("L:child-jobs", lambda: maintenance.run(root, should_stop=lambda: False))
        if refused(report):
            result["errors"].append(f"L:child-jobs refused: {refused(report)}")
    result["live_set"]["child_jobs"] = _summary(meter, "L:child-jobs")["counts"]


def _published_change(meter: Any, root: pathlib.Path, stage: str, result: Dict[str, Any], stop: threading.Event) -> Dict[str, Any]:
    """One real queue change published as the tick publishes it: what the owner's controls wait behind (``under_q``), and
    what lands. Each publication's files are named in the order they were replaced."""
    from ouroboros import server_maintenance as sm
    from supervisor import queue

    task_id = f"live-change-{stage}"

    def change() -> None:
        with queue._queue_lock:
            queue.PENDING.append(_task(task_id))
            queue.persist_queue_snapshot(reason="live_change")
        _publish_as_the_tick_does(queue)

    return _measured_pass(meter, f"L:{stage}:publication-change", change, sm, root, stop, result)


def _crash_after_each_replacement(root: pathlib.Path) -> List[Dict[str, Any]]:
    """E4 under a kill: the publication dies after its n-th file replacement, for each n until one publishes whole. What is on
    disk then (a kill leaves it so; here a refused replacement stops the rest) is what a restart and a reader meet."""
    import os

    from ouroboros.queue_observation import load_queue_document, queue_stamp
    from supervisor import queue

    real, facts = os.replace, []
    for allowed in range(0, 8):
        done: List[str] = []

        def replace(src, dst, *args, allowed=allowed, done=done, **kwargs):
            if len(done) >= allowed and os.path.basename(os.fspath(dst)) in _QUEUE_FILES:
                raise OSError("killed here")
            if os.path.basename(os.fspath(dst)) in _QUEUE_FILES:
                done.append(os.path.basename(os.fspath(dst)))
            return real(src, dst, *args, **kwargs)

        os.replace = replace
        try:
            with queue._queue_lock:
                queue.PENDING.append(_task(f"live-crash-{allowed}"))
                queue.persist_queue_snapshot(reason="live_crash")
            published = _publish_as_the_tick_does(queue)
        except Exception:
            published = False
        finally:
            os.replace = real
        document = load_queue_document(root)
        stamp = queue_stamp(document)
        dispatch = load_queue_document(root, "queue_dispatch.json")
        facts.append({"allowed": allowed, "replaced": done, "published": published, "certifies": bool(stamp),
                      "documents_agree": bool(stamp) and bool(queue_stamp(dispatch))
                      and f"live-crash-{allowed}" in str(document.get("pending"))})
        if published:
            break
    return facts


_QUEUE_FILES = frozenset({"queue_snapshot.json", "queue_dispatch.json", "queue.alive"})


def _reader_sees_owner_text(root: pathlib.Path) -> bool:
    from ouroboros.queue_observation import load_queue_document

    rows = {row.get("id"): row for row in load_queue_document(root).get("running") or []}
    return ((rows.get("live-run-000") or {}).get("task") or {}).get("metadata", {}).get("origin_message_text") == OWNER_TEXT


def _heartbeat_facts(meter: Any, srv: Any, client: Any, result: Dict[str, Any], root: pathlib.Path,
                     stop: threading.Event) -> Dict[str, Any]:
    """E3: the stamp a heartbeat leaves, and what it costs and sends with no client and with two."""
    from ouroboros import server_maintenance as sm
    from supervisor import queue
    from supervisor.events_worker_reports import _handle_task_heartbeat

    ctx, probe = _event_ctx(srv), "live-run-000"
    with queue._queue_lock:
        queue.RUNNING[probe]["last_heartbeat_at"] = 0.0
    alone = _measured_pass(meter, "L:heartbeat:no-clients", lambda: _handle_task_heartbeat(_beat(probe), ctx), sm, root, stop, result)
    stamped = queue.RUNNING[probe]["last_heartbeat_at"] > 0.0

    def read_frames(socket: Any, frames: List[int]) -> None:
        while True:
            if "task_heartbeat" in socket.receive_text():
                frames.append(1)
                return

    counted: List[List[int]] = [[], []]
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
        readers = [threading.Thread(target=read_frames, args=(socket, frames), daemon=True)
                   for socket, frames in zip((first, second), counted)]
        for reader in readers:
            reader.start()
        watched = _measured_pass(meter, "L:heartbeat:two-clients", lambda: _handle_task_heartbeat(_beat(probe), ctx), sm, root, stop, result)
        for reader in readers:
            reader.join(30.0)
    return {"stamp_moved": stamped, "frames_per_client": [sum(frames) for frames in counted],
            "no_clients": alone["counts"], "two_clients": watched["counts"]}


def _state_change_facts(client: Any, root: pathlib.Path) -> Dict[str, Any]:
    """E2's positive half for the state: one more pending row, published as the tick publishes it, is in the next answer."""
    from supervisor import queue

    before = client.get("/api/state").json()["pending_count"]
    grow_pending(root, 1_000, 1_001)
    _publish_as_the_tick_does(queue)
    return {"before": before, "after": client.get("/api/state").json()["pending_count"]}


def measure(meter: Any, srv: Any, client: Any, result: Dict[str, Any], root: pathlib.Path, spec: Dict[str, Any],
            stubs: Dict[str, Any]) -> None:
    facts = spec["facts"]
    shape = Shape(**spec.get("shape", AXIS._asdict()))
    stop = threading.Event()
    out = result.setdefault("live_set", {"stages": {}, "laid": {}, "shape": shape._asdict()})
    _previous_process_work_gone(root)
    _child_jobs_done(meter, root, result)
    # This thread stands for the held supervisor's tick: where queue changes are published by revision, a mark then waits
    # for the publisher (``_publish_as_the_tick_does``) instead of writing under the mark's own lock.
    from supervisor import queue

    getattr(queue, "tick_publisher_started", lambda: None)()
    table = None
    change: Dict[str, Any] = {}
    for stage in STAGES:
        started = time.monotonic()
        out["laid"][stage] = {**grow(stage, root, shape, facts), "seconds": round(time.monotonic() - started, 3)}
        if table is None:
            table = _passes(meter, srv, client, root, stubs, stop)
        out["stages"][stage] = _one_stage(meter, stage, table, root, stubs, stop, result)
    for stage in (STAGES[0], STAGES[-1]):
        change[stage] = _published_change(meter, root, stage, result, stop)
    largest = change[STAGES[-1]]
    change["order"] = [name.split(" ", 1)[1] for name in largest["names"] if name.startswith("replace ")]
    document = (root / "state" / "queue_snapshot.json").read_text(encoding="utf-8")
    change.update(row_published=f"live-change-{STAGES[-1]}" in document, owner_text_copies=document.count(OWNER_NEEDLE),
                  reader_sees_owner_text=_reader_sees_owner_text(root))
    out["change"] = change
    out["crashes"] = _crash_after_each_replacement(root)
    out["heartbeat"] = _heartbeat_facts(meter, srv, client, result, root, stop)
    out["state_change"] = _state_change_facts(client, root)


# ---- the six companions ----------------------------------------------------------------------------------------------

def companions(root: pathlib.Path) -> Dict[str, Callable[[], str]]:
    """One addressed input change per duty, after which its next pass has work (tests/_cost_driver.py ``_companions``):
    each returns what it changed, through the writer production uses for it."""
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    def fence() -> str:
        """A Pause still ``requested`` whose tree the duty finds settled (the request's own census ran before the
        last member parked, as a member at work leaves it): the fence turns ``paused`` in this pass."""
        from ouroboros.owner_pause import install_fence
        from supervisor.events_budget import _set_root_budget_pause_locked

        root_id = "cur-owner-paused-late"
        write_task_result(root, root_id, "scheduled", chat_id=MAIN, text="paused while its member parked",
                          root_task_id=root_id)
        accepted, _created = install_fence(root, root_id, request_id="companion-pause")
        with queue._queue_lock:
            queue.PENDING.append(_task(root_id))
            _set_root_budget_pause_locked(root_id, {"fence_id": str(accepted["fence_id"]), "cause": "owner_pause",
                                                    "paused_at": str(accepted.get("requested_at") or "")})
        return "an owner Pause still requested over a tree that can settle"

    def cold_request() -> str:
        """A saved cold request of an actor in the queue, with no caretaker to take it (the daemon is absent here):
        the pass records the typed refusal, the one write an unavailable body owes."""
        from ouroboros import budget_pause, sleep_custody
        from ouroboros.artifacts import store_actor_source_bytes

        task_id = "cur-cold-sleeper"
        write_task_result(root, task_id, "running", chat_id=MAIN, text="asks to sleep cold", root_task_id=task_id)
        ref = store_actor_source_bytes(root, task_id, category="context_checkpoints", source_id="cold-sleep",
                                       data=b"the saved request", extension="md")
        row = {"pause_id": "cold-1", "state": "pausing", "reason": "authority", "task_attempt": 1, "root_task_id": task_id,
               "source_ref": ref}
        budget_pause.set_budget_pause(root, task_id, {**row, "external_custody": {
            "state": "requested", "runs": [["run-cold", "invoke-cold"]], "binding": sleep_custody._binding(task_id, row)}})
        with queue._queue_lock:
            queue.RUNNING[task_id] = {"task": _task(task_id), "worker_id": 999, "attempt": 1, "started_at": time.time()}
        return "a saved cold request of a running actor"

    def due_schedule() -> str:
        """A due once-schedule: the pass claims its occurrence and admits one task under the queue's locks."""
        due = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        queue.upsert_scheduled_task(
            {"id": "live-due-once", "name": "Due now", "enabled": True, "source": "task_followup",
             "trigger": {"type": "once", "run_at": due},
             "task": {"type": "task", "text": "continue the due schedule", "chat_id": MAIN,
                      "metadata": {"resource_intent": {"kind": "system_repo"}}}},
            writer_kind="owner", writer_metadata={},
            host_followup={"followup_relation": {"kind": "independent"}, "followup_origin": {"task_id": "", "root_task_id": ""}})
        return "a once-schedule that came due"

    def held_continue() -> str:
        """A Continue admitted under its predecessor's writer: a delegated run whose custody is still open. Its custody
        then closes with no member reaching a terminal: the pass selects the held row, once."""
        from ouroboros import delegate_custody as custody
        from supervisor.continuation_admission import admit_continuation

        pred = "cur-continued"
        write_task_result(root, pred, "running", chat_id=MAIN, project_id="", root_task_id=pred, title="Friday report",
                          origin_message_text="Write the Friday report", deadline_at="2099-01-01T00:00:00+00:00",
                          origin_message_ref={"chat_id": MAIN, "client_message_id": "m-continued"},
                          billing_group={"billing_group_id": pred, "billing_group_limit_usd": 20.0,
                                         "billing_group_limit_source": "initial_task_admission",
                                         "billing_group_limit_revision": "admission-1"})
        write_task_result(root, pred, "cancelled", result="interrupted",
                          cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
        if not custody.record_started(root, custody.RunCustody(
                run_id="run-held-continue", task_id=pred, route_id="route", model="model", snapshot_id="snap-held-continue",
                authority_source="active_workspace", ledger_root=str(root))):
            raise RuntimeError("the writer's custody did not land")
        admitted = admit_continuation(pred, action_nonce="press-0001-abcdef")
        if not (admitted.get("ok") and admitted.get("held")):
            raise RuntimeError(f"the Continue was not held under its writer: {admitted}")
        custody.emit(root, custody.SETTLED, {"run_id": "run-held-continue", "task_id": pred, "route": "route",
                                             "model": "model", "state": "completed"})
        return "a writer-held Continue whose writer settled"

    def owed_notice() -> str:
        from ouroboros.budget_pause import set_budget_pause

        write_task_result(root, "cur-owes-notice", "running", chat_id=MAIN, text="paused on money")
        set_budget_pause(root, "cur-owes-notice", {"pause_id": "notice-1", "state": "paused", "reason": "budget"})
        return "a confirmed pause owing its local notice"

    def owed_synthesis() -> str:
        """A root's synthesis still owed after its owner's process went: the pass dispatches it, the model refuses here
        (the driver refuses every call), and the checkpoint says so."""
        write_task_result(root, "cur-owes-synthesis", "completed", chat_id=MAIN, root_task_id="cur-owes-synthesis",
                          objective="finish", accounted_upper_bound_usd=0.25, result="done",
                          root_phase_checkpoint={"phase": "task_acceptance", "status": "pass",
                                                 "post_task_synthesis": "pending_once"})
        return "a root synthesis owed with no live owner"

    return {"owner_pause_settlement": fence, "cold-custody": cold_request, "schedule-admission": due_schedule,
            "held_continuation_recheck": held_continue, "pause_notice_replay": owed_notice,
            "post_task_synthesis_recovery": owed_synthesis}
