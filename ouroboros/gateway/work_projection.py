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
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros.external_runs import unbound_starts
from ouroboros.gateway import work_status as ws
from ouroboros.owner_pause import owner_stopped
from ouroboros.project_naming import without_continue, work_shown_name
from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

log = logging.getLogger(__name__)

# The last projection's stored results per data root, each beside the row signature it was read at
# (``(write_seq, updated_at)``, the key every census reader memoizes on): the next projection re-reads only a
# document whose row moved. Replaced whole by each projection, so it keeps one projection's working set, the
# documents that projection held anyway; the results are shared read-only.
_RESULTS: Dict[str, Dict[str, Tuple[tuple, dict]]] = {}
_RESULTS_LOCK = threading.Lock()
# The census's execution facts for a member (gateway/state.py), never a work's identity.
_EXECUTION_FACTS = frozenset({"running", "owner_pause_warm", "owner_wait", "cancel_pending", "nothing_to_stop",
                              "tail_unknown"})


def _first_name(*sources: Mapping[str, Any]) -> str:
    for field in ("suggested_name", "title"):
        for source in sources:
            name = without_continue((source or {}).get(field))
            if name:
                return name
    return ""


def _root_key(drive_root: Any) -> str:
    return str(pathlib.Path(drive_root).resolve(strict=False))


def _signature(drive_root: Any, task_id: str) -> Optional[tuple]:
    """The task row's signature from its hot row (no body; batched inside ``store_hot_facts.snapshot``)."""
    from ouroboros.store_hot_facts import read

    try:
        row = read(drive_root, task_id)
    except Exception:
        return None
    return (row["write_seq"], row["updated_at"]) if row else None


class _Observations:
    """Per-snapshot memo of the few stored reads the projection needs."""

    def __init__(self, drive_root: Any):
        self.drive_root = drive_root
        self.results: Dict[str, Optional[dict]] = {}
        # Effective results may span the canonical task document and child-drive
        # replicas.  Keep that composition behind TaskReads, the existing batched
        # observation owner, rather than opening the effective loader directly for
        # every finalizing projection rebuild.
        from ouroboros.task_reads import TaskReads

        self.task_reads = TaskReads(drive_root)
        self.effective_results: Dict[str, Dict[str, Any]] = {}
        # Child stores touched by effective-result reads.  The state publisher
        # probes these roots too; a split-drive result can change without the
        # canonical budget store changing.
        self.dependency_roots = set()
        self.signatures: Dict[str, tuple] = {}
        with _RESULTS_LOCK:
            self.previous = _RESULTS.get(_root_key(drive_root), {})
        self.fences: Dict[str, dict] = {}
        self.stops: Dict[str, frozenset] = {}
        self.reviews: Dict[str, bool] = {}
        self.models: Dict[str, list] = {}
        self.gaps: List[Dict[str, Any]] = []

    def result(self, task_id: str) -> Optional[dict]:
        """The task's stored result: the last projection's when its row has not moved since, else one read."""
        if task_id not in self.results:
            signature = _signature(self.drive_root, task_id)
            held = self.previous.get(task_id)
            if signature is not None and held is not None and held[0] == signature:
                self.results[task_id] = held[1]
                self.signatures[task_id] = signature
                return held[1]
            try:
                from ouroboros.task_results import load_task_result

                row = load_task_result(self.drive_root, task_id)
                self.results[task_id] = row if isinstance(row, dict) else None
            except Exception:
                log.debug("Work projection could not read %s", task_id, exc_info=True)
                self.results[task_id] = None
            if signature is not None and self.results[task_id] is not None:
                self.signatures[task_id] = signature
        return self.results[task_id]

    def effective(self, task_id: str) -> Dict[str, Any]:
        """One effective result through the shared canonical/replica reader, memoized for this projection only.

        It never crosses projections: effective truth also folds the queue snapshot, cancel intents,
        worker-restart events and freshness, which no stored signature names. Every child store it
        reads joins ``dependency_roots``, one whose read failed included, so publication probes it.
        """
        task_id = str(task_id)
        if task_id in self.effective_results:
            return self.effective_results[task_id]

        try:
            value = self.task_reads.effective(task_id) or {}
        finally:
            # _stamps contains roots even when their read was unavailable or
            # too fresh to cache.  Publication must retain those dependencies
            # so recovery gets another probe rather than serving the failure.
            self.dependency_roots.update(self.task_reads._stamps)
        self.effective_results[task_id] = value
        return value

    def keep(self) -> None:
        """Hand this projection's signed results to the next one (only results read at a known signature)."""
        held = {task_id: (signature, self.results[task_id]) for task_id, signature in self.signatures.items()}
        with _RESULTS_LOCK:
            _RESULTS[_root_key(self.drive_root)] = held

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
                self.reviews[task_id] = bool(tree_review_operation(
                    self.drive_root, task_id, sent_only=True, result_loader=lambda tid: self.result(tid) or {}))
            except Exception:
                log.debug("Work projection could not read the review of %s", task_id, exc_info=True)
                self.reviews[task_id] = False  # an unreadable operation proves no review
        return self.reviews[task_id]

    def prime_model_handoffs(self, root_ids: Iterable[str]) -> None:
        """Every paused tree's model recipients over ONE read of their unanswered sends (``trees_model_handoffs``)."""
        wanted = sorted({root for root in root_ids if root and root not in self.models})
        if wanted:
            from supervisor.continuation_admission import trees_model_handoffs

            self.models.update(trees_model_handoffs(self.drive_root, wanted))

    def model_handoffs(self, root_id: str, drive_root: Any = None) -> list:
        if root_id not in self.models:
            self.models[root_id] = tree_model_handoffs(drive_root or self.drive_root, root_id)
        return self.models[root_id]

    def resume_attempt(self, task_id: str, drive_root: Any = None) -> dict:
        """The task's latest Resume attempt (``work_stop.ATTEMPT_KEY``) from the census's own body facts
        (``gateway.state._task_activity_facts``: its body is read again only once its row moves). An unreadable
        one is a disclosed gap, never an absent attempt."""
        from ouroboros.gateway.state import _task_activity_facts

        facts = _task_activity_facts(drive_root or self.drive_root, task_id)
        if facts.get("unavailable"):
            self.gaps.append({"reason": "resume_attempt_unavailable", "task_id": task_id})
        attempt = facts.get("resume_attempt")
        return attempt if isinstance(attempt, dict) else {}

    def stop_unknown(self, task_id: str, drive_root: Any = None, owner: Optional[Mapping[str, Any]] = None
                     ) -> frozenset:
        if task_id not in self.stops:
            self.stops[task_id] = unconfirmed_stops(drive_root or self.drive_root, task_id, owner)
        return self.stops[task_id]


def tree_model_handoffs(drive_root: Any, root_id: str) -> list:
    """A paused tree's unresolved model recipients: with ``tree_review_operation`` the tree census ``/api/state``
    reads, so the owner-pause census memo (keyed by the tree's attempt/custody/fence revision) replaces it here."""
    from supervisor.continuation_admission import model_handoffs

    return model_handoffs(drive_root, root_id)


def tree_review_operation(drive_root: Any, task_id: str, *, sent_only: bool = False, result_loader=None) -> bool:
    """Whether a review ``task_id`` launched still runs (``sent_only``: one it already sent): the tree census's
    review read for ``/api/state``, beside ``tree_model_handoffs``."""
    from ouroboros import review_operation

    return review_operation.task_has_live_review_operation(drive_root, task_id, sent_only=sent_only,
                                                           result_loader=result_loader)


def tree_review_reads(drive_root: Any, root_id: str):
    """The review questions the census asks of one paused root (sent or not) over ONE strict stored read."""
    from ouroboros.task_results import load_task_result

    results: Dict[str, dict] = {}
    answers: Dict[bool, bool] = {}

    def load(task_id: str) -> dict:
        if task_id not in results:
            results[task_id] = load_task_result(drive_root, task_id, strict=True) or {}
        return results[task_id]

    def ask(sent_only: bool = False) -> bool:
        if sent_only not in answers:
            answers[sent_only] = tree_review_operation(drive_root, root_id, sent_only=sent_only, result_loader=load)
        return answers[sent_only]

    return ask


def unconfirmed_stops(drive_root: Any, task_id: str, owner: Optional[Mapping[str, Any]] = None) -> frozenset:
    """The runs whose stop ``task_id``'s live Pause could not confirm (``external_runs.EXTERNAL_STOP_UNKNOWN``): its
    cold park's pause row, the record its warm park keeps on its stored result (``owner``, waiting under the
    owner's Pause), or, once it ended, the record its root's closed fence keeps for it. The server's and the
    roster's one reading; an unreadable record proves no trouble, so those runs read as before."""
    from ouroboros.external_runs import ENDED_MEMBER_RUNS, EXTERNAL_STOP_UNKNOWN
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.store_hot_facts import lineage_root
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
        if isinstance(owner, Mapping) and owner.get("status") in _TRULY_TERMINAL_STATUSES:  # only an ended owner has one
            fence = read_fence(pathlib.Path(drive_root), lineage_root({"task_id": task_id, **dict(owner)}))
            if fence_closed(fence):
                stops |= unknown(((fence.get(ENDED_MEMBER_RUNS) or {}).get(task_id) or {}).get("external_runs"))
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
            work_id = str(continuation["work_id"])
            name = work_shown_name(obs.result(work_id)) or str(continuation.get("work_name") or "")
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
        facts["resume_attempt"] = obs.resume_attempt(str(row.get("activity_id") or ""), raw.get("budget_drive_root"))
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
                   *, complete: bool = True, bindings: Optional[Mapping[str, Any]] = None,
                   publication_roots: Optional[set] = None,
                   questions: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    """``works``, ``rooms``, ``work_summary``, ``works_complete`` and ``works_gaps`` for ``/api/state``;
    ``bindings`` is the census's task-to-Project map (``_bound_room``); ``questions`` are Main's current questions
    (``main_decisions``), which the owner's turn of each room counts as Main's pill does. Never raises."""
    try:
        return _snapshot(drive_root, list(owners or ()), dict(raw_facts or {}), complete, dict(bindings or {}),
                         publication_roots, list(questions or ()))
    except Exception:
        log.debug("Work projection unavailable for /api/state", exc_info=True)
        return {"works": [], "rooms": {}, "work_summary": ws.roll_up_summary({}, [], complete=False),
                "works_complete": False, "works_gaps": [{"reason": "work_projection_unavailable"}]}


def durable_binding(row: Mapping[str, Any], task_id: Any, bindings: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The binding that places a task: its own, then its parent's or root's durable one (the census's one rule; an
    origin-bound row is a convert-gate fact only and re-homes nothing)."""
    for key in (task_id, row.get("parent_task_id"), row.get("root_task_id")):
        binding = bindings.get(str(key)) if key else None
        if isinstance(binding, dict) and not binding.get("origin_bound"):
            return binding
    return None


def _named(value: Any) -> bool:
    """Whether a stored row names an id at all. A chat id is a value, never a truth value: 0 is the hidden partition
    (``chat_id_policy.HIDDEN_CHAT_ID``), a real room; only a missing id (None, or the empty string) is absent."""
    return value is not None and value != ""


def _bound_room(row: Mapping[str, Any], task_id: str, bindings: Mapping[str, Any]) -> Mapping[str, Any]:
    """A stored row's room by ``durable_binding``. Unbound, the row's own room stands."""
    binding = durable_binding(row, task_id, bindings)
    if binding is None or not _named(binding.get("chat_id")):
        return row
    return {**row, "chat_id": binding["chat_id"], "project_id": str(binding.get("project_id") or "")}


def _snapshot(drive_root: Any, owners: List[Mapping[str, Any]], raw: Dict[str, Mapping[str, Any]],
              complete: bool, bindings: Mapping[str, Any], publication_roots: Optional[set] = None,
              questions: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    from ouroboros.store_hot_facts import snapshot

    obs = _Observations(drive_root)
    # The signatures of every result this projection may read, in one short transaction (the hot rows only).
    known = {str(row.get("activity_id") or "") for row in owners} | set(raw) | set(obs.previous)
    try:
        with snapshot([(drive_root, task_id) for task_id in known if task_id]):
            projected = _project(drive_root, owners, raw, complete, obs, bindings, questions)
    finally:
        if publication_roots is not None:
            publication_roots.update(obs.dependency_roots)
    obs.keep()
    return projected


def _project(drive_root: Any, owners: List[Mapping[str, Any]], raw: Dict[str, Mapping[str, Any]],
             complete: bool, obs: _Observations, bindings: Mapping[str, Any],
             questions: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in owners:
        task_id = str(row.get("activity_id") or "")
        if task_id and task_id not in rows:
            rows[task_id] = dict(row)
    try:
        for row in _inflight_upkeep(drive_root, set(rows)):
            facts = obs.result(row["activity_id"]) or {}
            room = _bound_room(facts, row["activity_id"], bindings)  # a moved task's upkeep runs in its Project
            row.update({key: room[key] for key in ("chat_id", "project_id") if _named(room.get(key))})
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
    obs.prime_model_handoffs(root_id for entry in groups.values() for row, facts, root_id in entry["rows"]
                             if row.get("phase") in {"budget_paused", "budget_pausing"}
                             and not facts.get("budget_drive_root"))
    for root_id, runs in held.items():
        work_id, name = _work_of(root_id, raw, obs, memo)
        entry = group(work_id)
        entry["runs"].extend(_typed_run(run, obs, raw) for run in runs)
        entry["roots"].add(root_id)
        entry["name"] = entry["name"] or name
    works: List[Dict[str, Any]] = []
    # The census rows' own names: what a sleep waits for is said by them, never by an id or a new read.
    names = {task_id: _first_name(facts) for task_id, facts in raw.items()}
    for work_id, entry in groups.items():
        members, live_roots, owners_by_root, starts = [], [], {}, set()
        for row, facts, root_id in entry["rows"]:
            member = ws.classify_member(row, _member_raw(row, facts, obs, root_id), names)
            members.append(member)
            owners_by_root.setdefault(root_id, []).append(member)
            if _episode_row(row, facts, root_id):
                live_roots.append((row, facts, member, root_id))
            if row.get("phase") in {"budget_paused", "budget_pausing"}:
                starts.update(unbound_starts(obs.fence(root_id, facts.get("budget_drive_root"))))
        # A start no run is bound to yet: the fence names it, no custody row can, and nobody can confirm it.
        entry["runs"].extend({"run_id": start, "kind": "run", "state": "unknown"} for start in sorted(starts))
        current_row = current_state = None
        current_root = ""
        if live_roots:
            row, facts, member, current_root = max(live_roots, key=lambda item: float(item[0].get("started_at") or 0.0))
            episode = {"task_id": row["activity_id"], "state": member}
            outcome = ""
            if row.get("phase") == "finalizing":
                # A split-drive root's census row stays its canonical hot row (running) until copy-back: its
                # settled word is the effective result's, as its card reads it; neither known says none.
                try:
                    effective = obs.effective(str(row["activity_id"]))
                except Exception:
                    log.debug("Work projection could not read %s", row.get("activity_id"), exc_info=True)
                    effective = {}
                outcome = next((word for word in (ws.shown_outcome(effective or {}), ws.shown_outcome(row))
                                if word in _TRULY_TERMINAL_STATUSES), "")
            room_row, current_row, current_state = row, row, member
        else:
            episode_id = sorted(entry["roots"])[-1] if entry["roots"] else work_id
            stored = obs.result(episode_id) or {}
            episode = {"task_id": episode_id}
            outcome = ws.shown_outcome(stored)
            # A tail-only work (open runs, review or upkeep, no member row) lives where its binding puts it.
            room_row = next((item[0] for item in entry["rows"]), None) or _bound_room(stored, episode_id, bindings)
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
        controls = ws.controls_for(current_state, current_row, earlier_live=earlier, live_roots=ended + logical + earlier,
                                   presence_turn=bool((raw.get(current_id) or {}).get("_presence_turn")))
        from ouroboros.owner_model import eligibility

        durable = obs.result(current_id) if current_row is not None else None
        if (durable and current_state["activity"] in {"working", "queued", "waiting"}
                and not eligibility(durable) and durable.get("status") not in _TRULY_TERMINAL_STATUSES):
            controls.append({"action": "model", "target": current_id})
        if (raw.get(current_id) or {}).get("nothing_to_stop"):
            controls = [control for control in controls if control["action"] != "stop"]  # its door has nothing to reach
        # An ended author's tree takes the owner's Pause while a helper lives, or an ordinary run its members left
        # may still execute (the door reads the same ended task: ``owner_pause_control._ended_tail``).
        stored_root = obs.result(current_id) or {} if current_state is None else {}
        if current_state is None and (any(m["activity"] in {"working", "queued", "waiting"} for m in members) or (
                any(run["kind"] == "run" for run in entry["runs"])
                and stored_root.get("status") in _TRULY_TERMINAL_STATUSES and not owner_stopped(stored_root))):
            controls.insert(0, {"action": "pause", "target": current_id})
        if current_state is None:
            # An ended author is never run again. Each held helper has its own
            # addressed Resume in Activity; a work-level press selects none.
            controls.extend({"action": "resume", "target": member["task_id"], "scope": "single"}
                            for member in members if member["activity"] == "paused" and member.get("resumable")
                            and member.get("stop", {}).get("scope") != "tree")
        work.update(kind="direct" if str(room_row.get("kind") or "") == "direct_chat" else "task",
                    project_id=str(room_row.get("project_id") or ""), since=_since(room_row), controls=controls)
        if current_row is not None and current_row.get("lead"):
            work["lead"] = current_row["lead"]
        cmid = str(room_row.get("client_message_id") or "")
        if cmid:
            work["client_message_id"] = cmid
        works.append(work)
    works.sort(key=lambda work: (str(work.get("room") or ""), ws._rank(work["activity"]),
                                 -float(work.get("since") or 0.0), work["work_id"]))
    # What waits for the owner is one set: a current question is its room's turn, in its work's when one holds it.
    loose = ws.place_questions(works, questions)
    by_room: Dict[str, List[Dict[str, Any]]] = {}
    for work in works:
        by_room.setdefault(str(work.get("room") or 0), []).append(work)
    rooms = {chat: ws.roll_up_room(by_room.get(chat, []), complete=complete, questions=loose.get(chat, ()))
             for chat in [*by_room, *(chat for chat in loose if chat not in by_room)]}
    return {"works": works, "rooms": rooms, "work_summary": ws.roll_up_summary(rooms, works, complete=complete),
            "works_complete": bool(complete), "works_gaps": obs.gaps}
