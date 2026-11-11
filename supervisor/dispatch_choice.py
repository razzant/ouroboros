"""Choosing queued work for free capacity: one predicate read in cost order, and a pass that stops at k.

A queued row starts only when every gate passes, and the chooser reads the launch's own gates with the launch's own
functions, so it never takes a row the launch would refuse (S01: a row the launch refused by its own facts was chosen
and refused about once a tick for days, three SQLite reads a look and forty looks a tick). Gates, cheapest first:

1. memory (``worker_assignment._candidate_ineligible``, under Q): claims and holds, depth, the root's latch and
   cancellation fence, the direct actor and the Presence carrier, the per-root child limit and the one-writer lane;
   its stat reads (a moved room's binding before the lane, a copied root's verdict, Off/Panic, the writer gate, the
   paid route at the limit) only for a row every memory gate before them passed;
2. store (``_screen``, outside Q): the agent restart's holds, an evolution's campaign authority, a schedule's control
   (``followup_policy.scheduled_start``), the owner Pause and the row's own sleep (``owner_pause._validate_launch``),
   read in one short transaction per store (``store_hot_facts.read_ahead``) beside the reads only schedules,
   evolution and a restart need;
3. C2 (``events_budget.budget_resume_dispatch_allowed``), decided under Q from what step 2 read
   (``store_hot_facts.serving``, strict: nothing under Q opens the store).

C2 is also the chooser's repair of a superseded selection (#1196, Q9: it returns to its exact pause or to the root's
hold), so, as before, it is decided for every row the walk passes that carries a selection and is not otherwise held,
whatever a later gate says; the decision and its repair come from a full read outside Q (R3 item 17). A full read's
admission is kept while the in-memory identities it rested on stand (``_identity``: the row's carriers and stops, its
root's latch, the queue facts of its root and of the root's physical actor, that actor's direct turn and
``store_hot_facts.CONTROL_WRITES``), so an unchanged root is not read again; only an admission is kept.

Nothing here authorizes a start: the launch (``worker_assignment._launch_attempt``) takes the owner's launch lock, the
schedule's table hold and the restart boundary, reads again, and revalidates memory and C2 at the put. Only C2's
refusal at the put (``resume_admits``) forgets the chooser's kept admission, so the next pass can repair the
selection; the launch's other refusals keep it (``run_pass``).

``run_pass``: ``worker_pool_lifecycle.admissible_starts`` says how many new processes the pass may start; with no idle
worker and no start it chooses nothing and reads nothing. Otherwise one walk in queue order (``queue._queue_sort_key``:
type priority, then ``_queue_seq``) stops at the rows it can use: one per idle worker (the owner's cap on acting tasks,
when set, counts them) and, while processes may start, as many as the booting slots and those starts can carry. With
no idle worker a start is one probe row until the walk chose work that needs a new process; only then is the machine
read (``pool_admission``, once, outside Q), and a further walk collects what the starts it admitted may carry. A
refusal is not kept as a hold: the next pass asks again. C2's kept admissions are bounded by the queue and forgotten
with their row (``_forget``).

A walk reads one lane's rows (``lane``): the project pool's, or my human's returned turns, which the conversation
lane's carriers take with the same predicate (``conversation_lane.resumes_on_lane``,
``worker_assignment._assign_returned_turns``).

Disclosed residual (§10 invariant 24): with free capacity the memory gates still walk the held prefix, O(P) a pass,
and a superseded selection beyond the rows a pass reaches waits for a pass that reaches it; the tick's cancel hygiene
keeps one ``statuses`` SELECT, read before Q (``store_hot_facts.statuses_ahead``), and its cancel-intent projection read.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.task_carrier import LANE_CONVERSATION, LANE_PROJECT

log = logging.getLogger(__name__)

# The walk; a second only when its rows were refused after it (store gates, a refused claim); one more when the
# pass's reading of the machine admitted more starts than the one probe row the walk was sized with.
_ROUNDS = 3
_ADMITTED: Dict[str, Tuple[tuple, str]] = {}  # row id -> (identity, actor): C2 admitted it under this identity
_CONTROL_KEYS = ("_attempt", "_budget_pause", "_budget_pause_resume", "_resume_selection", "_budget_pause_hold",
                 "_owner_hold", "_owner_wait_resume", "_working_recovery", "_continuation_prepared")


def _pool():
    from supervisor import workers

    return workers


def _assignment():
    from supervisor import worker_assignment

    return worker_assignment


def _tid(row: Dict[str, Any]) -> str:
    return str(row.get("id") or "")


def _root(row: Dict[str, Any]) -> str:
    return str(row.get("root_task_id") or row.get("id") or "")


def _drive(queue: Any, row: Dict[str, Any]) -> pathlib.Path:
    return pathlib.Path(row.get("budget_drive_root") or queue.DRIVE_ROOT)


def _carrier(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("_budget_pause_resume"), dict) or isinstance(row.get("_resume_selection"), dict)


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _controls(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in _CONTROL_KEYS if key in row}


def _identity(queue: Any, row: Dict[str, Any], index: tuple, actor: str,
              memo: Optional[Dict[tuple, str]] = None) -> tuple:
    """Under Q: what C2's verdict for ``row`` rests on that this process sees without a read (``memo``: one pass's
    root parts, shared by the root's members)."""
    from ouroboros.store_hot_facts import CONTROL_WRITES

    root = _root(row)
    key = (root, actor)
    part = memo.get(key) if memo is not None else None
    if part is None:
        view = _narrow(queue, index, row, actor)
        named = [(id(item), _controls(item)) for item in [*view.PENDING, *(m["task"] for m in view.RUNNING.values())]]
        part = _json([queue.BUDGET_ROOT_FENCES.get(root), named,
                      bool(actor) and _pool().direct_chat_turn(actor) is not None])
        if memo is not None:
            memo[key] = part
    return id(row), _json(_controls(row)), part, CONTROL_WRITES[0]


def _index(queue: Any) -> tuple:
    """Under Q: queued rows and running metas by task id (built once a walk, only when a row carries a selection)."""
    pending: Dict[str, list] = {}
    running: Dict[str, dict] = {}
    for item in queue.PENDING:
        if isinstance(item, dict):
            pending.setdefault(_tid(item), []).append(item)
    for task_id, meta in queue.RUNNING.items():
        if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
            running.setdefault(_tid(meta["task"]), {})[task_id] = meta
    return pending, running


def _narrow(queue: Any, index: tuple, row: Dict[str, Any], actor: str) -> SimpleNamespace:
    """Under Q: the queue as C2 reads it for ``row``: it looks only at the rows of the row's root and of the root's
    actor, so this view answers exactly as the whole queue would, without a pass over every queued row."""
    names, (pending, running) = dict.fromkeys((_root(row), actor)), index
    return SimpleNamespace(DRIVE_ROOT=queue.DRIVE_ROOT, BUDGET_ROOT_FENCES=queue.BUDGET_ROOT_FENCES,
                           PENDING=[item for name in names for item in pending.get(name, ())],
                           RUNNING={tid: meta for name in names for tid, meta in running.get(name, {}).items()})


def _kept(queue: Any, row: Dict[str, Any], index: tuple, memo: Dict[tuple, str]) -> bool:
    kept = _ADMITTED.get(_tid(row))
    return bool(kept and kept[0] == _identity(queue, row, index, kept[1], memo))


def _repairable(row: Dict[str, Any]) -> bool:
    """A selection C2 decides and, when superseded, repairs: no other hold owns the row (the chooser's condition)."""
    from supervisor.events_budget import budget_hold_fact

    return bool(_carrier(row) and not row.get("_owner_hold") and not row.get("_project_admission_restore_hold")
                and not row.get(_assignment()._LAUNCH_CLAIM) and not isinstance(row.get("_budget_pause"), dict)
                and budget_hold_fact(row) is None)


def _drives(queue: Any) -> Dict[str, Any]:
    """Outside Q, from copies: where each queued and running task keeps its money rows (C2 reads its actor's there)."""
    pending, running = list(queue.PENDING), list(queue.RUNNING.values())
    return {_tid(item): item.get("budget_drive_root")
            for item in [*pending, *(meta.get("task") or {} for meta in running if isinstance(meta, dict))]}


def _forget(pending: List[Dict[str, Any]]) -> None:
    """Under Q: a kept admission lives only while its row is queued."""
    if _ADMITTED:
        live = {_tid(row) for row in pending}
        for task_id in [task_id for task_id in _ADMITTED if task_id not in live]:
            _ADMITTED.pop(task_id, None)


def _restart_holds(row: Dict[str, Any], held: Dict[int, bool]) -> bool:
    """Outside Q: whether the agent's pending Restart holds this row (a store read for a continuation; once a pass)."""
    from ouroboros.server_restart import agent_restart_holds, agent_restart_pending

    if id(row) not in held:
        held[id(row)] = bool(agent_restart_pending() and _assignment()._unsent(row) is None
                             and agent_restart_holds(row))
    return held[id(row)]


def _own_refusal(queue: Any, row: Dict[str, Any], held: Dict[int, bool]) -> str:
    """Outside Q: the launch's refusals that read beyond the row's pause and fence, in cost order ("" when none)."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if _restart_holds(row, held):
        return "agent_restart"
    error = _pool()._evolution_assignment_error(row)  # "" for any other row, read only for an evolution
    if error:
        return "evolution_authority:" + error
    if metadata.get("schedule_id"):
        from supervisor.followup_policy import scheduled_start
        from supervisor.queue_schedules import automatic_acquisitions

        with automatic_acquisitions(), scheduled_start(queue.DRIVE_ROOT, row) as allowed:
            if not allowed:
                return "schedule_control"
    return ""


def _actor_reads(queue: Any, row: Dict[str, Any], drives: Dict[str, Any],
                 actors: Optional[Dict[tuple, str]] = None) -> Tuple[str, list]:
    """Outside Q: the physical actor of the row's root (its retry lineage: one body read a hop, once a pass per root
    through ``actors``) and the rows C2 reads for the row; ``drives`` is a copy of where tasks keep their money rows."""
    from ouroboros.cancel_intents import _validated_single_cancel_target

    drive, root = _drive(queue, row), _root(row)
    own = [(drive, _tid(row)), (drive, root)]  # its own pause and its tree's fence (a sleep-readiness grant)
    if root == _tid(row):
        return root, own
    key = (os.path.abspath(drive), root)
    if actors is None or key not in actors:
        actor = _validated_single_cancel_target(drive, root)
        if actors is not None:
            actors[key] = actor
    actor = actors[key] if actors is not None else actor
    drives_read = {drive, pathlib.Path(drives.get(actor) or drive)}
    return actor, own + [(where, name) for where in drives_read for name in (actor, root)]


def _screen(queue: Any, rows: List[Dict[str, Any]], undecided: List[Dict[str, Any]],
            plans: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Outside Q: each candidate's store gates, and the facts C2 is decided from under Q for every undecided row."""
    from ouroboros import store_hot_facts
    from ouroboros.owner_pause import OwnerPauseRefused, _validate_launch
    from ouroboros.store import StoreBusy

    refused, actors, requests, open_rows, held = {}, {}, [], [], {}
    drives: Optional[Dict[str, Any]] = None
    lineage: Dict[tuple, str] = {}
    for row in undecided:
        try:
            if _restart_holds(row, held):
                continue  # a row the pending Restart holds is neither decided nor repaired, as the chooser skipped it
            drives = _drives(queue) if drives is None else drives
            actors[id(row)], wanted = _actor_reads(queue, row, drives, lineage)
            requests += wanted
        except Exception:
            log.debug("Retry lineage of %s unreadable; its selection waits this pass", row.get("id"), exc_info=True)
    for row in rows:
        try:
            reason = _own_refusal(queue, row, held)
        except StoreBusy:
            reason = "deferred"  # contention: the same row is still demand, asked again next pass
        except Exception:
            log.debug("Store gates of %s unreadable; it waits this pass", row.get("id"), exc_info=True)
            reason = "unreadable"
        if reason:
            refused[id(row)] = reason
            continue
        requests += [(_drive(queue, row), _root(row)), (_drive(queue, row), _tid(row))]
        open_rows.append(row)
    facts = store_hot_facts.read_ahead(requests) if requests else {}
    with store_hot_facts.serving(facts):
        for row in open_rows:
            source = SimpleNamespace(task_id=_tid(row), root_task_id=row.get("root_task_id"),
                                     budget_drive_root=row.get("budget_drive_root") or queue.DRIVE_ROOT)
            try:
                _validate_launch(source, plans[id(row)]["root_resume"])
            except OwnerPauseRefused as exc:
                refused[id(row)] = f"owner_pause:{exc}"
            except Exception:
                log.debug("Launch fences of %s unreadable; it waits this pass", row.get("id"), exc_info=True)
                refused[id(row)] = "unreadable"
    return {"refused": refused, "actors": actors, "facts": facts}


def _repair(queue: Any, row: Dict[str, Any]) -> None:
    """Under Q, outside any read-ahead: a selection C2 found stale from a full read goes back to its exact pause or to
    the root's hold (#1196, Q9), as the chooser always repaired it; nothing else here writes."""
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, hold_budget_row

    if isinstance(row.get("_budget_pause_resume"), dict):
        from supervisor.budget_resume import revoke_exact_budget_resume

        revoke_exact_budget_resume(row, "root_resume_generation_stale")
    else:
        from ouroboros.task_results import resolve_task_lineage

        row.pop("_resume_selection", None)
        lineage = resolve_task_lineage(row.get("id"), **{key: row.get(key) for key in (
            "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")})
        if not lineage["is_root_task"]:
            root_id = lineage["root_task_id"]
            latch = queue.BUDGET_ROOT_FENCES.get(root_id) or {}
            hold_budget_row(row, reason=HOLD_ROOT_FENCE_LIFTED, detail="root_resume_generation_stale",
                            extra={"root_task_id": root_id, "fence_id": str(latch.get("fence_id") or "")},
                            result_root=pathlib.Path(row.get("budget_drive_root") or _pool().DRIVE_ROOT))
    queue.persist_queue_snapshot(reason="stale_child_resume_held")


def _root_resume(queue: Any, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Under Q: the root resume the launch passes its fence check (``_launch_attempt``'s own rule)."""
    from supervisor.events_budget import budget_fence_selected

    # A monetary latch hands over only an explicit selection against it (an
    # owner-resumed cold sleeper must start), never sleep readiness.
    root = _root(row)
    latch, resume = queue.BUDGET_ROOT_FENCES.get(root) or {}, row.get("_budget_pause_resume")
    selected = bool(_tid(row) != root and isinstance(resume, dict)
                    and (latch.get("cause") == "owner_pause" or budget_fence_selected(row, latch))
                    and resume.get("root_fence_id") == latch.get("fence_id"))
    return resume if not latch or selected else None


def _collect(queue: Any, remaining: float, invalid_ids: set, limit: int, excluded: set,
             decided: set, carried: Any = (), lane: str = LANE_PROJECT) -> Optional[tuple]:
    """Under Q: up to ``limit`` rows of ``lane`` the memory gates (then their few stat reads) admit, in queue order,
    each seen by the lanes and child limits of the rows before it and of the rows an earlier walk of this pass chose
    for growth (``carried``), and every row of ``lane`` it passes whose selection C2 must decide (not yet this pass,
    ``decided``, and no admission kept); None when a malformed depth could not be terminalized."""
    from ouroboros.project_lease import lane_holders, running_project_ids
    from supervisor.conversation_lane import resumes_on_lane

    wa = _assignment()
    picked: List[Dict[str, Any]] = []
    undecided: List[Dict[str, Any]] = []
    plans: Dict[int, Dict[str, Any]] = {}
    with queue._queue_lock:
        running, chosen = wa._started_projection() + [{"task": row} for row in carried], set()
        leased, holders = running_project_ids(running), lane_holders(running, _pool().PENDING)
        index, memo = None, {}
        for row in list(_pool().PENDING):
            if len(picked) >= limit:
                break
            if resumes_on_lane(row) != (lane == LANE_CONVERSATION):
                continue  # the other lane's row: its own walk decides and repairs it
            if id(row) not in decided and _repairable(row):
                index = _index(queue) if index is None else index
                decided.add(id(row))
                if not _kept(queue, row, index, memo):
                    undecided.append(row)
            if wa._candidate_ineligible(queue, row, remaining, running, selected=chosen, refused=excluded,
                                        holders=holders, leased=leased, unresolved_invalid_id_set=invalid_ids):
                continue
            depth_error = _pool()._normalize_pending_task_depth(row)
            if depth_error:
                if not _pool()._terminalize_invalid_pending_depth(row, depth_error):
                    log.error("Assignment blocked: invalid task depth could not be terminalized for %s", row.get("id"))
                    return None
                _pool().PENDING[:] = [item for item in _pool().PENDING if item is not row]
                queue.persist_queue_snapshot(reason="invalid_task_depth")
                continue
            plans[id(row)] = {"root_resume": _root_resume(queue, row)}
            picked.append(row)
            chosen.add(_tid(row))
            running.append({"task": row})
            leased, holders = running_project_ids(running), lane_holders(running, _pool().PENDING)
    return picked, undecided, plans


def _decide(queue: Any, rows: List[Dict[str, Any]], undecided: List[Dict[str, Any]], screened: Dict[str, Any],
            remaining: float, invalid_ids: set, carried: Any = ()) -> List[Dict[str, Any]]:
    """Under Q: C2 for every undecided row, from step 2's read (an admission kept, a superseded selection repaired
    after the read-ahead is left); then the candidates still queued, still admitted by memory, unrefused by step 2
    and admitted by C2. An evolution without its campaign authority is cancelled, as the chooser cancelled it."""
    from ouroboros import store_hot_facts
    from supervisor.events_budget import budget_resume_dispatch_allowed

    wa = _assignment()
    live, admitted, stale, chosen = {id(item) for item in _pool().PENDING}, [], [], set()
    index = _index(queue) if undecided or any(_carrier(row) for row in rows) else ({}, {})
    memo: Dict[tuple, str] = {}
    running = wa._started_projection() + [{"task": row} for row in carried]
    with store_hot_facts.serving(screened["facts"], strict=True):
        for row in undecided:
            actor = screened["actors"].get(id(row))
            if id(row) not in live or actor is None or not _repairable(row):
                continue  # gone, held since, or its lineage unread: decided by a later pass
            verdict = budget_resume_dispatch_allowed(_narrow(queue, index, row, actor), row, actor_id=actor)
            if verdict is True:
                _ADMITTED[_tid(row)] = (_identity(queue, row, index, actor, memo), actor)
            elif verdict is False:
                stale.append((row, ""))
        for row in rows:
            reason = screened["refused"].get(id(row), "")
            if id(row) not in live or any(row is item for item, _why in stale):
                continue
            if wa._candidate_ineligible(queue, row, remaining, running, selected=chosen,
                                        unresolved_invalid_id_set=invalid_ids, stat=False):
                continue
            if reason or _carrier(row) and not _kept(queue, row, index, memo):
                if reason.startswith("evolution_authority:"):
                    stale.append((row, reason))
                continue
            admitted.append(row)
            chosen.add(_tid(row))
            running.append({"task": row})
    for row, reason in stale:
        if reason:
            if _pool()._cancel_unauthorized_evolution(row, reason.split(":", 1)[1]):
                _pool().PENDING[:] = [item for item in _pool().PENDING if item is not row]
                queue.persist_queue_snapshot(reason="evolution_authority_rejected")
        else:
            _repair(queue, row)
    return admitted


def choose(queue: Any, remaining: float, invalid_ids: Any = (), *, limit: int = 1,
           excluded: Any = (), lane: str = LANE_PROJECT) -> Optional[List[Dict[str, Any]]]:
    """Up to ``limit`` queued rows of ``lane`` every gate admits now, in queue order, claiming nothing (selections it
    passes are decided and repaired as in a pass); None when the pass must stop (a malformed depth whose terminal could
    not be written). The caller holds no lock."""
    collected = _collect(queue, remaining, set(invalid_ids), limit, set(excluded), set(), lane=lane)
    if collected is None:
        return None
    rows, undecided, plans = collected
    if not rows and not undecided:
        return []
    screened = _screen(queue, rows, undecided, plans)
    with queue._queue_lock:
        return _decide(queue, rows, undecided, screened, remaining, set(invalid_ids))


def _capacity(queue: Any, reading: Any) -> Tuple[list, int, int]:
    """Under Q: the idle project workers this pass may hand rows to (within the owner's cap on acting tasks, when one
    is set), the booting slots, and how many rows growth may carry: booting slots and new starts, only while starts >
    0. Before the pass reads the machine (``reading`` None) a start is one probe row, offered only when no worker is
    idle, to learn whether chosen work needs a new process at all; after the reading ``admissible_starts`` says."""
    from supervisor.pool_admission import PROBE
    from supervisor.worker_pool_lifecycle import acting_room, admissible_starts, idle_project_slots, usable_booting

    idle, room = idle_project_slots(), acting_room()
    idle = idle if room is None else idle[:room]
    booting = usable_booting()
    starts = admissible_starts(PROBE if reading is None and not idle else reading)
    return idle, booting, booting + starts if starts else 0


def run_pass(queue: Any, remaining: float, invalid_ids: Any) -> Optional[Tuple[Callable[[], int], Any]]:
    """One assignment pass over free capacity: reuse every idle worker, count what growth may carry. Returns the demand
    ``reconcile_worker_capacity`` grows on (a callable it reads under Q) with the pass's reading of the machine (None
    when it read none), or None when the pass was blocked.

    The machine is read at most once a pass, outside Q, and only when the walk chose more work than the idle and
    booting processes carry (the probe row): a pass whose work fits reads nothing. The reading then sizes growth, and
    one more walk collects what further starts it admitted may carry."""
    from supervisor import pool_admission

    wa = _assignment()
    invalid_ids = set(invalid_ids)
    reading = None
    with queue._queue_lock:
        idle, booting, growth_room = _capacity(queue, reading)
        if not _pool().PENDING or not idle and not growth_room:
            return (lambda: 0), None  # nothing to reuse and nothing to start: no choice is made and nothing is read
        _forget(_pool().PENDING)
    wanted: List[Dict[str, Any]] = []
    excluded: set = set()
    deferred: set = set()
    decided: set = set()
    for _round in range(_ROUNDS):
        with queue._queue_lock:
            idle = _capacity(queue, reading)[0]
        limit = len(idle) + max(0, growth_room - len(wanted))
        collected = _collect(queue, remaining, invalid_ids, limit, excluded | {_tid(row) for row in wanted}, decided,
                             wanted)
        if collected is None:
            return None
        rows, undecided, plans = collected
        if not rows and not undecided:
            break
        screened = _screen(queue, rows, undecided, plans)
        with queue._queue_lock:
            admitted = _decide(queue, rows, undecided, screened, remaining, invalid_ids, wanted)
            refused = [row for row in rows if not any(row is item for item in admitted)]
            excluded |= {_tid(row) for row in refused if screened["refused"].get(id(row)) != "deferred"}
            wanted += [row for row in refused if screened["refused"].get(id(row)) == "deferred"]
            idle = _capacity(queue, reading)[0]
            for row in admitted:
                if not idle:
                    if len(wanted) < growth_room:
                        wanted.append(row)
                    continue
                outcome = wa._claim_worker_launch(queue, row, idle[0], deferred=deferred)
                if outcome is False:
                    refused.append(row)
                    if _tid(row) in deferred:
                        wanted.append(row)  # contention deferred it: the same work retries, still demand
                    else:
                        excluded.add(_tid(row))
                    continue
                if outcome is True:  # the inline path put it; bind it here under the same Q hold
                    wa._bind_running(queue, row, idle[0])
                idle.pop(0)
        grew = False
        if reading is None and not idle and len(wanted) > booting:  # chosen work needs a new process: read, once
            reading = pool_admission.read()
            with queue._queue_lock:
                idle, booting, room = _capacity(queue, reading)
            grew, growth_room = room > growth_room, room
        if not grew and (not refused or not idle and len(wanted) >= growth_room):
            break

    def residual() -> int:
        """Under Q: the chosen rows no idle worker took that are still queued and unclaimed (growth's demand)."""
        live = {id(item) for item in _pool().PENDING}
        return sum(1 for row in wanted if id(row) in live and not row.get(wa._LAUNCH_CLAIM)
                   and _pool().repo_writer_task_allowed(row))

    return residual, reading


def resume_read(queue: Any, row: Dict[str, Any]) -> Optional[Tuple[str, Dict[Any, Any]]]:
    """Outside Q, before a launch's put: the actor and facts C2 reads for a row carrying a selection, else None."""
    from ouroboros import store_hot_facts

    if not _carrier(row):
        return None
    try:
        actor, requests = _actor_reads(queue, row, _drives(queue))
    except Exception:
        log.debug("Retry lineage of %s unreadable at the put", _root(row), exc_info=True)
        return None
    return actor, store_hot_facts.read_ahead(requests)


def resume_admits(queue: Any, row: Dict[str, Any], read: Optional[Tuple[str, Dict[Any, Any]]]) -> bool:
    """Under Q at the put: C2 from ``resume_read``'s facts; a row without a selection passes, one read without fails.
    A refusal forgets the chooser's kept admission: the put saw a write its identity cannot see (another process's),
    so the next pass decides C2 from a full read and repairs the selection instead of choosing it again."""
    from ouroboros import store_hot_facts
    from supervisor.events_budget import budget_resume_dispatch_allowed

    if not _carrier(row):
        return True
    admitted = False
    if read is not None:
        with store_hot_facts.serving(read[1], strict=True):
            admitted = budget_resume_dispatch_allowed(queue, row, actor_id=read[0]) is True
    if not admitted:
        _ADMITTED.pop(_tid(row), None)
    return admitted
