"""Server observations for the work projection (``gateway/work_status.py``).

``works_snapshot`` turns the census builder's full activity set (``decision_owners``:
roots, children, answered roots finishing, post-task waiters) plus the queue facts the
census rows drop into ``/api/state`` ``works``/``rooms``/``work_summary``. It adds the
members no queue row names: delegated runs still working and answer upkeep a direct turn
still runs after its actor left. Reads are display reads; a failed read is a disclosed
gap, never an empty room. Nothing here writes, repairs or wakes anything.
"""
from __future__ import annotations

import logging
import pathlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros.gateway import work_status as ws

log = logging.getLogger(__name__)

_CONTINUE_PREFIX = "Continue: "
# The census's execution facts for a member (gateway/state.py), never a work's identity.
_EXECUTION_FACTS = frozenset({"running", "owner_pause_warm", "owner_wait"})


def _strip_continue(name: str) -> str:
    name = str(name or "").strip()
    while name.startswith(_CONTINUE_PREFIX):
        name = name[len(_CONTINUE_PREFIX):].strip()
    return name


def _first_name(*sources: Mapping[str, Any]) -> str:
    for field in ("suggested_name", "title"):
        for source in sources:
            name = _strip_continue(str((source or {}).get(field) or ""))
            if name:
                return name
    return ""


class _Observations:
    """Per-snapshot memo of the few stored reads the projection needs."""

    def __init__(self, drive_root: Any):
        self.drive_root = drive_root
        self.results: Dict[str, Optional[dict]] = {}
        self.fences: Dict[str, dict] = {}
        self.stops: Dict[str, frozenset] = {}
        self.reviews: Dict[str, bool] = {}
        self.models: Dict[str, list] = {}
        self.gaps: List[Dict[str, Any]] = []

    def result(self, task_id: str) -> Optional[dict]:
        if task_id not in self.results:
            try:
                from ouroboros.task_results import load_task_result

                row = load_task_result(self.drive_root, task_id)
                self.results[task_id] = row if isinstance(row, dict) else None
            except Exception:
                log.debug("Work projection could not read %s", task_id, exc_info=True)
                self.results[task_id] = None
        return self.results[task_id]

    def fence(self, root_id: str, drive_root: Any = None) -> dict:
        if root_id not in self.fences:
            try:
                from ouroboros.owner_pause import read_fence

                self.fences[root_id] = read_fence(pathlib.Path(drive_root or self.drive_root), root_id)
            except Exception:
                self.fences[root_id] = {}
        return self.fences[root_id]

    def sent_review(self, task_id: str) -> bool:
        """Whether a review the task already sent still runs (``review_operation``), read from its stored result."""
        if task_id not in self.reviews:
            try:
                from ouroboros.review_operation import task_has_live_review_operation

                self.reviews[task_id] = bool(task_has_live_review_operation(
                    self.drive_root, task_id, sent_only=True, result_loader=lambda tid: self.result(tid) or {}))
            except Exception:
                log.debug("Work projection could not read the review of %s", task_id, exc_info=True)
                self.reviews[task_id] = False  # an unreadable operation proves no review
        return self.reviews[task_id]

    def model_handoffs(self, root_id: str, drive_root: Any = None) -> list:
        if root_id not in self.models:
            from supervisor.continuation_admission import model_handoffs
            self.models[root_id] = model_handoffs(drive_root or self.drive_root, root_id)
        return self.models[root_id]

    def stop_unknown(self, task_id: str, drive_root: Any = None, owner: Optional[Mapping[str, Any]] = None
                     ) -> frozenset:
        if task_id not in self.stops:
            self.stops[task_id] = unconfirmed_stops(drive_root or self.drive_root, task_id, owner)
        return self.stops[task_id]


def unconfirmed_stops(drive_root: Any, task_id: str, owner: Optional[Mapping[str, Any]] = None) -> frozenset:
    """The runs whose stop ``task_id``'s live Pause could not confirm (``external_runs.EXTERNAL_STOP_UNKNOWN``): its
    cold park's pause row, or the record its warm park keeps on its stored result (``owner``, waiting under the
    owner's Pause). The server's and the roster's one reading; an unreadable record proves no trouble, so those
    runs read as before."""
    from ouroboros.external_runs import EXTERNAL_STOP_UNKNOWN
    from supervisor.owner_pause_control import warm_paused_member

    def unknown(external: Any) -> set:
        runs = (external.get("runs") or ()) if isinstance(external, dict) else ()
        return {str(run["run_id"]) for run in runs
                if isinstance(run, dict) and run.get("run_id") and run.get("state") == EXTERNAL_STOP_UNKNOWN}

    stops = set()
    if isinstance(owner, Mapping) and warm_paused_member({"owner_wait": owner.get("owner_wait")}):
        stops |= unknown((owner["owner_wait"].get("owner_pause") or {}).get("external_runs"))
    try:
        from ouroboros.budget_pause import LIVE_PAUSE_STATES, budget_pause_row

        row = budget_pause_row(pathlib.Path(drive_root), str(task_id)) if task_id else {}
        if row.get("state") in LIVE_PAUSE_STATES:
            stops |= unknown(row.get("external_runs"))
    except Exception:
        log.debug("Work projection could not read the pause of %s", task_id, exc_info=True)
    return frozenset(stops)


def _continuation(source: Mapping[str, Any]) -> dict:
    if isinstance(source.get("continuation"), dict):
        return source["continuation"]
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return metadata.get("continuation") if isinstance(metadata.get("continuation"), dict) else {}


def _work_of(root_id: str, raw: Mapping[str, Mapping[str, Any]], obs: _Observations,
             memo: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
    """(work_id, work name) of one episode root: its own stamp, else its recorded predecessor edges (a Continue's
    predecessor, a timeout retry's original)."""
    if root_id in memo:
        return memo[root_id]
    chain: List[str] = []
    current, work_id, name = root_id, "", ""
    while current and current not in chain:
        chain.append(current)
        if current in memo:
            work_id, name = memo[current]
            break
        facts = raw.get(current) or {}
        # A census entry that only says whether the episode runs carries no identity: its stored result does.
        source = facts if set(facts) - _EXECUTION_FACTS else obs.result(current) or {}
        continuation = _continuation(source)
        if continuation.get("work_id"):
            work_id, name = str(continuation["work_id"]), str(continuation.get("work_name") or "")
            break
        predecessor = str(continuation.get("predecessor_task_id") or source.get("original_task_id") or "")
        if not predecessor:
            work_id, name = current, _first_name(source)
            break
        if not (raw.get(predecessor) or obs.result(predecessor)):
            # The edge names a record that cannot be read: the last known episode stands for the work.
            obs.gaps.append({"reason": "episode_record_unavailable", "task_id": predecessor})
            work_id, name = current, _first_name(source)
            break
        current = predecessor
    else:
        work_id = work_id or root_id
        obs.gaps.append({"reason": "episode_chain_cycle", "task_id": root_id})
    for task_id in chain:
        memo[task_id] = (work_id or root_id, name)
    return memo[root_id]


def _episode_row(row: Mapping[str, Any], raw: Mapping[str, Any], root_id: str) -> bool:
    """A live episode root: a non-child row that is its root, or a retry attempt standing for that root."""
    if row.get("parent_task_id") or raw.get("parent_task_id"):
        return False
    original = str(row.get("original_task_id") or raw.get("original_task_id") or "")
    return str(row.get("activity_id") or "") == root_id or original == root_id


def _root_of(row: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    task_id = str(row.get("activity_id") or "")
    if row.get("parent_task_id") or raw.get("parent_task_id"):
        return str(row.get("root_task_id") or raw.get("root_task_id") or task_id)
    return str(row.get("original_task_id") or raw.get("original_task_id") or task_id)


def _member_raw(row: Mapping[str, Any], raw: Mapping[str, Any], obs: _Observations, root_id: str) -> dict:
    hold = raw.get("_budget_pause_hold") if isinstance(raw.get("_budget_pause_hold"), dict) else {}
    pause = raw.get("_budget_pause") if isinstance(raw.get("_budget_pause"), dict) else {}
    facts: Dict[str, Any] = {"hold": hold, "paused_at": pause.get("paused_at") or pause.get("requested_at"),
                             "running": bool(raw.get("running", row.get("live"))), **dict(raw)}
    facts["execution_transport"] = (raw.get("metadata") or {}).get("execution_transport")
    if row.get("phase") in {"budget_pausing", "budget_paused"} and (raw.get("owner_pause_warm") or row.get("kind") == "direct_chat"):
        # Parked warm under a Pause still settling: the review it already sent still runs beside it.
        facts["sent_review"] = obs.sent_review(str(row.get("activity_id") or ""))
    if row.get("phase") in {"budget_paused", "budget_pausing"}:
        from ouroboros.control_actor import pause_author

        fence = obs.fence(root_id, raw.get("budget_drive_root"))
        # The Pause is dated from its press: a settling fence restamps ``paused_at``.
        facts["paused_at"] = fence.get("requested_at") or fence.get("paused_at") or facts["paused_at"]
        facts["pause_by"] = pause_author(fence)
        facts["fences"] = {"owner": fence}
        facts["model_handoffs"] = obs.model_handoffs(root_id, raw.get("budget_drive_root"))
        facts["parked"] = [str(task) for task in fence.get("parked_members") or () if task]
    return facts


def _inflight_upkeep(drive_root: Any, known: set) -> List[Dict[str, Any]]:
    """Answer upkeep this process still runs for a task no census row names (a direct turn after its answer)."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK

    root = str(pathlib.Path(drive_root).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        task_ids = [task for (path, task) in POST_TASK_SYNTHESIS_INFLIGHT if path == root and task not in known]
    rows = []
    for task_id in task_ids:
        rows.append({"activity_id": task_id, "phase": "finalizing", "kind": "direct_chat"})
    return rows


def _open_runs(drive_root: Any) -> Dict[str, List[Any]]:
    """Delegated runs with a durable start and no settlement, by their tree's root: members of that root's work,
    which the cascade Stop of that root cancels. A settled run custody still holds (an undisposed patch) runs
    nothing and is not a member; nor is a run the engine already reported finished while its settlement is owed."""
    from ouroboros.delegate_custody_reconcile import open_runs, run_tree_id

    held: Dict[str, List[Any]] = {}
    for entry in open_runs(drive_root):
        root_id = run_tree_id(entry)
        if root_id and not str(getattr(entry, "terminal_state", "") or ""):
            held.setdefault(root_id, []).append(entry)
    return held


def _typed_run(entry: Any, obs: _Observations, raw: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """One open run as a work tail (``work_status.run_tail``): review work, a live run, or a run nobody can confirm,
    from the run's ``source``, its own task's stored outcome and that task's live Pause."""
    task_id = str(entry.task_id or "")
    owner = obs.result(task_id) if task_id else None
    stops = obs.stop_unknown(task_id, (raw.get(task_id) or {}).get("budget_drive_root"), owner) if task_id else frozenset()
    kind, state = ws.run_tail(getattr(entry, "source", ""), owner, stop_unknown=str(entry.run_id) in stops)
    return {"run_id": str(entry.run_id), "kind": kind, "state": state}


def _since(row: Mapping[str, Any]) -> Optional[float]:
    """The work's start in epoch seconds: a census row's number, or a stored result's ISO time."""
    value = row.get("started_at")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def works_snapshot(drive_root: Any, owners: Iterable[Mapping[str, Any]], raw_facts: Mapping[str, Mapping[str, Any]],
                   *, complete: bool = True) -> Dict[str, Any]:
    """``works``, ``rooms``, ``work_summary``, ``works_complete`` and ``works_gaps`` for ``/api/state``. Never raises."""
    try:
        return _snapshot(drive_root, list(owners or ()), dict(raw_facts or {}), complete)
    except Exception:
        log.debug("Work projection unavailable for /api/state", exc_info=True)
        return {"works": [], "rooms": {}, "work_summary": ws.roll_up_summary({}, [], complete=False),
                "works_complete": False, "works_gaps": [{"reason": "work_projection_unavailable"}]}


def _snapshot(drive_root: Any, owners: List[Mapping[str, Any]], raw: Dict[str, Mapping[str, Any]],
              complete: bool) -> Dict[str, Any]:
    obs = _Observations(drive_root)
    rows: Dict[str, Dict[str, Any]] = {}
    for row in owners:
        task_id = str(row.get("activity_id") or "")
        if task_id and task_id not in rows:
            rows[task_id] = dict(row)
    try:
        for row in _inflight_upkeep(drive_root, set(rows)):
            facts = obs.result(row["activity_id"]) or {}
            row.update({key: facts[key] for key in ("chat_id", "project_id") if facts.get(key)})
            row["kind"] = "direct_chat" if facts.get("_is_direct_chat", True) else "managed_task"
            rows[row["activity_id"]] = row
    except Exception:
        complete = False
        obs.gaps.append({"reason": "answer_upkeep_unavailable"})
    try:
        held = _open_runs(drive_root)
    except Exception:
        held, complete = {}, False
        obs.gaps.append({"reason": "delegated_custody_unavailable"})
    memo: Dict[str, Tuple[str, str]] = {}
    groups: Dict[str, Dict[str, Any]] = {}

    def group(work_id: str) -> Dict[str, Any]:
        return groups.setdefault(work_id, {"rows": [], "roots": set(), "runs": [], "name": ""})

    for task_id, row in rows.items():
        facts = raw.get(task_id) or {}
        root_id = _root_of(row, facts)
        work_id, name = _work_of(root_id, raw, obs, memo)
        entry = group(work_id)
        entry["rows"].append((row, facts, root_id))
        entry["roots"].add(root_id)
        entry["name"] = entry["name"] or name
    for root_id, runs in held.items():
        work_id, name = _work_of(root_id, raw, obs, memo)
        entry = group(work_id)
        entry["runs"].extend(_typed_run(run, obs, raw) for run in runs)
        entry["roots"].add(root_id)
        entry["name"] = entry["name"] or name
    works: List[Dict[str, Any]] = []
    for work_id, entry in groups.items():
        members, live_roots, owners_by_root = [], [], {}
        for row, facts, root_id in entry["rows"]:
            member = ws.classify_member(row, _member_raw(row, facts, obs, root_id))
            members.append(member)
            owners_by_root.setdefault(root_id, []).append(member)
            if _episode_row(row, facts, root_id):
                live_roots.append((row, facts, member, root_id))
        current_row = current_state = None
        current_root = ""
        if live_roots:
            row, facts, member, current_root = max(live_roots, key=lambda item: float(item[0].get("started_at") or 0.0))
            episode = {"task_id": row["activity_id"], "state": member}
            outcome = ws.shown_outcome(row) if row.get("phase") == "finalizing" else ""
            room_row, current_row, current_state = row, row, member
        else:
            episode_id = sorted(entry["roots"])[-1] if entry["roots"] else work_id
            stored = obs.result(episode_id) or {}
            episode = {"task_id": episode_id}
            outcome = ws.shown_outcome(stored)
            room_row = next((item[0] for item in entry["rows"]), stored)
            facts = raw.get(episode_id) or stored
        if not members and not entry["runs"]:
            continue
        name = entry["name"] or _first_name(facts)
        work = ws.roll_up_work(work_id, members, episode=episode, name=name, room=room_row.get("chat_id"),
                               focus=(facts or {}).get("focus"), runs=entry["runs"], outcome=outcome)
        current_id = str(episode["task_id"])
        # Every root grouped here owns a live queue or direct row of its tree (each member is one) or an open
        # delegated run: exactly what the Stop door reaches (queue_transitions.task_subtree_is_live), so each root
        # takes Stop. A writer the door cannot reach (an unbound start, a handoff, a process) is no member and adds
        # no root, so no control is offered for it.
        # Every other root is the earlier attempt; a retry attempt's own original stands for the same episode.
        earlier = sorted(root for root in entry["roots"] if root not in {current_id, current_root})
        # Stop ends the whole work: a retry's logical original carries its tree too (never an earlier attempt).
        logical = [current_root] if current_root and current_root != current_id else []
        ended = [] if current_state is not None else [current_id]
        controls = ws.controls_for(current_state, current_row, earlier_live=earlier, live_roots=ended + logical + earlier)
        if current_state is None and members and any(m["activity"] in {"working", "queued", "waiting"} for m in members):
            controls.insert(0, {"action": "pause", "target": current_id})
        if current_state is None:
            # An ended author is never run again. Each held helper has its own
            # addressed Resume in Activity; a work-level press selects none.
            controls.extend({"action": "resume", "target": member["task_id"], "scope": "single"}
                            for member in members if member["activity"] == "paused" and member.get("resumable")
                            and member.get("stop", {}).get("scope") != "tree")
        work.update(kind="direct" if str(room_row.get("kind") or "") == "direct_chat" else "task",
                    project_id=str(room_row.get("project_id") or ""), since=_since(room_row), controls=controls)
        cmid = str(room_row.get("client_message_id") or "")
        if cmid:
            work["client_message_id"] = cmid
        works.append(work)
    works.sort(key=lambda work: (str(work.get("room") or ""), ws._rank(work["activity"]),
                                 -float(work.get("since") or 0.0), work["work_id"]))
    by_room: Dict[str, List[Dict[str, Any]]] = {}
    for work in works:
        by_room.setdefault(str(work.get("room") or 0), []).append(work)
    rooms = {chat: ws.roll_up_room(items, complete=complete) for chat, items in by_room.items()}
    return {"works": works, "rooms": rooms, "work_summary": ws.roll_up_summary(rooms, works, complete=complete),
            "works_complete": bool(complete), "works_gaps": obs.gaps}
