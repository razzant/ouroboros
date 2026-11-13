"""One tree's writer census, its durable facts read once per change of the tree's own rows.

The settle pass, Pause and Resume (``conflicting_writers``) and the display and
roster readers (``model_handoffs``) all ask the same store the same question about one tree. They share
this memo. It holds INPUTS, never verdicts: the member rows' custody fields and each member's run
observation, keyed by the row sequences that produced them. Every blocker is still derived per call from these
inputs plus what no revision can name (the queue, a process that exited, a receiver's owner, a tool's
effect), so a hit never answers with a stale physical fact. The model requests are read fresh by the
caller: their read IS their revision, and a tree with none needs no member at all.

The revision is read BEFORE its facts, so a write landing between the two leaves newer facts under an
older key and only costs the next reader a re-read; it can never serve older facts under a newer key.
Nothing that failed to read is kept: unknown stays unknown and is asked again.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import logging
import pathlib
import threading
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from ouroboros.utils import machine_json
from ouroboros.work_stop import pause_marker

log = logging.getLogger(__name__)

# What a census reads from a member's document; a held row keeps these, never the body it was decoded from.
MEMBER_FIELDS = ("launch_handoffs", "merge_receipts", "retired_model_consumers", "review_operations")
# What a census reads from an observed run.
RUN_FIELDS = ("run_id", "invocation_id", "review_owned", "state", "stop_outcome", "source", "independent_review")
# A physical memory bound, never a judgment: an evicted tree is read again by whoever asks next.
MEMO_TREES = 256

_LOCK = threading.Lock()
_MEMO: "collections.OrderedDict[Tuple[str, str], _Facts]" = collections.OrderedDict()


class Revision(NamedTuple):
    members: tuple  # the tree's rows and their sequences, the tree's custody tasks, its requests
    runs: Dict[str, tuple]  # per asked id: its runs' sequences and its open requests


class _Facts:
    __slots__ = ("key", "links", "tree_ids", "tree_error", "rows", "unresolved", "census_error", "observations")

    def __init__(self, key: Any, links: frozenset) -> None:
        self.key, self.links = key, links
        self.tree_ids: List[str] = []
        self.tree_error = ""
        self.rows: Dict[str, dict] = {}
        self.unresolved: Dict[str, dict] = {}
        self.census_error = ""
        self.observations: Dict[str, tuple] = {}


class WriterFacts:
    """The durable half of one census: what ``conflicting_writers`` and ``model_handoffs`` read."""

    def __init__(self, root: pathlib.Path, facts: _Facts, revision: Optional[Revision]) -> None:
        self._root, self._facts, self._revision = root, facts, revision

    tree_ids = property(lambda self: self._facts.tree_ids)
    tree_error = property(lambda self: self._facts.tree_error)
    rows = property(lambda self: self._facts.rows)
    unresolved = property(lambda self: self._facts.unresolved)
    census_error = property(lambda self: self._facts.census_error)

    def observed(self, member: str) -> Dict[str, Any]:
        """``member``'s open runs as a census reads them; a custody store that cannot answer is not kept."""
        from ouroboros.budget_pause import observe_task_runs

        key = self._revision.runs.get(member) if self._revision is not None else None
        held = self._facts.observations.get(member)
        if key is not None and held is not None and held[0] == key:
            return held[1]
        full = observe_task_runs(self._root, member, reason="continuation_writer_check", request_stop=False)
        observed = {"custody_read": full.get("custody_read"),
                    "runs": [{name: run.get(name) for name in RUN_FIELDS} for run in full.get("runs") or []
                             if isinstance(run, dict)]}
        if key is not None and observed["custody_read"] == "ok":
            self._facts.observations[member] = (key, observed)
        return observed


def read_revision(root: pathlib.Path, tree: str, asked: Iterable[str] = ()) -> Optional[Revision]:
    """One read transaction naming everything a census of ``tree`` would read from the current store, without
    decoding a body. ``None`` for a store below the hot-fact columns: no memo, the census reads as before."""
    from ouroboros import task_store
    from ouroboros.delegate_custody_current import _LEGACY_TREE, _TREE, _selected
    from ouroboros.store_hot_facts import VERSION, tree_requests

    with task_store.open_store(root, write=False) as txn:
        if int(txn.meta("schema_version") or VERSION) < VERSION:
            return None
        rows = sorted((row["task_id"], row["write_seq"]) for row in txn.conn.execute(
            "SELECT task_id,write_seq FROM tasks WHERE root_task_id=? UNION ALL "
            "SELECT task_id,write_seq FROM tasks INDEXED BY tasks_underived_root WHERE root_task_id IS NULL", (tree,)))
        where, params = _selected(txn, _TREE, _LEGACY_TREE, (tree, tree))
        custody = sorted({row["task_id"] for row in txn.conn.execute(
            f"SELECT DISTINCT task_id FROM custody_runs WHERE {where}", params)})
        requested = tree_requests(txn, tree, refused=True)
        ids = sorted({tree, *map(str, asked), *custody, *(task_id for task_id, _seq in rows),
                      *(str((facts.get("request") or {}).get("task_id") or "") for _n, _i, facts in requested)} - {""})
        selected = json.dumps(ids)
        runs: Dict[str, list] = {task_id: [] for task_id in ids}
        for row in txn.conn.execute("SELECT task_id,run_id,write_seq FROM custody_runs "
                                    "WHERE task_id IN (SELECT value FROM json_each(?))", (selected,)):
            runs[row["task_id"]].append((row["run_id"], row["write_seq"]))
        opened: Dict[str, list] = {task_id: [] for task_id in ids}
        for row in txn.conn.execute(
                "SELECT id,facts,CAST(json_extract(facts,'$.request.task_id') AS TEXT) AS owner FROM obligations "
                "WHERE set_name='custody_open' AND CAST(json_extract(facts,'$.request.task_id') AS TEXT) "
                "IN (SELECT value FROM json_each(?))", (selected,)):
            opened[row["owner"]].append((row["id"], _digest(row["facts"])))
    tree_part = (tuple(rows), tuple(custody), tuple((name, identity, _digest(machine_json(facts)))
                                                    for name, identity, facts in requested))
    return Revision(tree_part, {task_id: (tuple(sorted(runs[task_id])), tuple(sorted(opened[task_id])))
                                for task_id in ids})


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _read_facts(root: pathlib.Path, tree: str, linked: Iterable[str], key: Any) -> _Facts:
    """Each source answers on its own: a failed body census still leaves custody and queue members observed."""
    from ouroboros.delegate_custody_current import tree_task_ids
    from ouroboros.owner_pause import tree_member_results

    links = frozenset(map(str, linked))
    facts = _Facts(key, links)
    try:
        # A member whose result has been collected can still own a remote start.
        facts.tree_ids = list(tree_task_ids(root, tree))  # the store answers or raises
    except Exception as exc:
        facts.tree_error = str(exc)[:200]
    try:
        census = tree_member_results(root, tree, linked={*links, *facts.tree_ids})
        facts.rows = {task_id: {name: row[name] for name in MEMBER_FIELDS if name in row}
                      for task_id, row in census.members.items()}
        facts.unresolved = dict(census.unresolved)
    except Exception as exc:
        facts.census_error = str(exc)[:200]
    return facts


def writer_facts(custody_root: Any, tree: str, *, linked: Optional[Iterable[str]] = None) -> WriterFacts:
    """The durable facts of ``tree``, from the memo while its rows stand. ``linked``: the ids the caller's queue
    and model requests tie to this tree. It only decides whether a DAMAGED row is this tree's (``task_store._tied``),
    so it keys the memo only when such a row exists; a reader that names none takes any record of this revision."""
    root, tree = pathlib.Path(custody_root), str(tree)
    asked = frozenset(map(str, linked)) if linked is not None else None
    try:
        revision = read_revision(root, tree, asked or ())
    except Exception:
        log.debug("Tree revision of %s unreadable; its census reads every source", tree, exc_info=True)
        revision = None
    key = (str(root), tree)
    with _LOCK:
        held = _MEMO.get(key)
        if held is not None:
            _MEMO.move_to_end(key)
    if (held is not None and revision is not None and held.key == revision.members
            and (asked is None or not held.unresolved or asked == held.links)):
        return WriterFacts(root, held, revision)
    facts = _read_facts(root, tree, asked or (), revision.members if revision is not None else None)
    if revision is not None and not facts.tree_error and not facts.census_error:
        with _LOCK:
            _MEMO[key] = facts
            _MEMO.move_to_end(key)
            while len(_MEMO) > MEMO_TREES:
                _MEMO.popitem(last=False)
    return WriterFacts(root, facts, revision)


def member_result_loader(custody_root: Any, tree: str) -> Any:
    """A ``result_loader`` for the review readers right after a census: the rows it just read, else the
    member's own strict read. A row a census kept is never the whole document, only what those readers use."""
    from ouroboros.task_results import load_task_result

    root = pathlib.Path(custody_root)
    with _LOCK:
        held = _MEMO.get((str(root), str(tree)))
    rows = held.rows if held is not None else {}

    def load(task_id: str) -> dict:
        return rows[task_id] if task_id in rows else load_task_result(root, task_id, strict=True) or {}

    return load


# The ONE explicit purpose that may leave an independent review out of the
# writer census: the owner's Continue. Every other consumer (Pause, Resume,
# money, sleep, restart) keeps the whole observation.
CONTINUE_PURPOSE = "owner_continue"


def conflicting_writers(q: Any, predecessor: str, *, drive_root: Any = None, owner_pause_fence_id: str = "",
                        purpose: str = "", gaps: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """Observe all retained tree custody, regardless of the member's lifecycle.

    Queue authority is copied briefly; custody observations never cancel runs.
    Unknown files or unbound starts keep the same Continue action held. Each
    source answers on its own: a failed body census still leaves the queue,
    custody and model-request members observed. A member whose own record is
    unreadable holds as ``member_record_unreadable``; ``gaps`` receives the rows
    the census could tie to no tree (``tree_member_results``), never a writer,
    and only once every link source answered: until then such a row holds too.
    ``owner_pause_fence_id`` is the owner Pause's own census: a queued admitted
    dispatch that only that exact accepted Pause's latch holds is not running
    work (its earlier effects are still observed below). Continue never passes
    it: that row would dispatch again once the Pause is resumed. Callers pass the
    root's closed fence they just read; while its queue latch is not projected
    (``latch_pending``, or a park that could not read it) that durable fence alone
    refuses every launch of the tree, so it stands in for the latch. The answered
    predecessor's own late phase is custody too while it runs or its state is
    unreadable; a Pause ends it at once and saves nothing that could write again.
    ``purpose=CONTINUE_PURPOSE`` is the owner's Continue (#1572): a delegated
    run or unbound start the observation projects as ``independent_review``
    (a review surface registered it and it has no candidate-apply possibility),
    and a model attempt under a reviewer slot's (or skill review's) custody
    attribution, read the predecessor's work but cannot write it, so they do
    not hold the Continue; a legitimately running
    panel still finishes and attaches its verdict. Missing or unknown
    provenance, ordinary delegations, author research copies, author model
    attempts, tool/process/merge/executor custody and unreadable custody keep
    holding. Retained raised sends without a receiver are not writers, even
    with unresolved cost; this pure classification applies to every consumer.
    The empty default is the complete census for every other consumer.
    """
    from supervisor.queue_transitions import budget_pause_fact, queued_admitted_dispatch

    custody_root = pathlib.Path(drive_root or q.DRIVE_ROOT)
    for_continue = purpose == CONTINUE_PURPOSE
    blockers: List[Dict[str, Any]] = []
    members = {predecessor}
    from supervisor.owner_pause_control import warm_paused_member

    with q._queue_lock:
        running = [(str(tid), dict(meta.get("task") or {}),
                    warm_paused_member(meta) if owner_pause_fence_id else False)
                   for tid, meta in q.RUNNING.items() if isinstance(meta, dict)]
        pending = [dict(task) for task in q.PENDING if isinstance(task, dict)]
        fences = {root: dict(row) for root, row in q.BUDGET_ROOT_FENCES.items() if isinstance(row, dict)}
    latch = fences.get(predecessor) or {}
    if owner_pause_fence_id and not latch:
        latch = fences[predecessor] = {"root_task_id": predecessor, "fence_id": str(owner_pause_fence_id),
                                       "cause": "owner_pause", "status": "paused"}
    owner_latched = bool(owner_pause_fence_id and latch.get("cause") == "owner_pause"
                         and str(latch.get("status") or "") in {"active", "paused"}
                         and str(latch.get("fence_id") or "") == str(owner_pause_fence_id))
    for task_id, task, warm in running:
        if str(task.get("root_task_id") or task_id) == predecessor:
            members.add(task_id)
            if warm:
                continue  # saved warm under the owner's Pause: its stack waits, it writes nothing
            blockers.append({"kind": "running_member", "task_id": task_id})
    from ouroboros.post_task_checkpoint import late_phase_state

    late = late_phase_state(custody_root, predecessor)
    if late:
        blockers.append({"kind": f"late_phase_{late}", "task_id": predecessor})
    for task in pending:
        tid = str(task.get("id") or "")
        if str(task.get("root_task_id") or tid) == predecessor:
            members.add(tid)
            if owner_latched and queued_admitted_dispatch(task):
                continue
            if budget_pause_fact(task, fences) is None or (task.get("admitted_dispatch") == "possible"
                                                   and not pause_marker(task)):
                blockers.append({"kind": "dispatchable_member", "task_id": tid})
    links_known = True  # a damaged row is called unlinked only once every link source answered
    try:
        from ouroboros.owner_pause import tree_model_requests
        attempts = tree_model_requests(custody_root, predecessor)  # read once for both uses
    except Exception:
        attempts, links_known = None, False  # model_handoffs below reads again and names the unreadable custody
    # The tree's durable facts come from the one memo; what depends on the moment is derived below.
    facts = writer_facts(custody_root, predecessor, linked={
        *members, *(str(attempt.get("task_id") or "") for attempt in attempts or ())})
    members.update(facts.tree_ids)
    if facts.tree_error:
        links_known = False
        blockers.append({"kind": "tree_census_unreadable", "detail": facts.tree_error})
    try:
        if facts.census_error:
            raise ValueError(facts.census_error)
        from ouroboros.tool_custody import retained_tool_custody
        for task_id, row in facts.rows.items():
            members.add(task_id)
            # A blocker carries its claim outward: the held row stays this census's own.
            blockers.extend(retained_tool_custody(custody_root, task_id, copy.deepcopy(row)))
        for task_id, fact in facts.unresolved.items():
            if fact["linked"] or not links_known:  # its runs and processes are still observed by id below
                members.add(task_id)
                blockers.append({"kind": "member_record_unreadable", "task_id": task_id, "detail": fact["reason"]})
            elif gaps is not None:
                gaps.append({"task_id": task_id, "reason": fact["reason"]})
    except Exception as exc:
        blockers.append({"kind": "tree_census_unreadable", "detail": str(exc)[:200]})
    for member in sorted(members):
        observed = facts.observed(member)
        if observed.get("custody_read") != "ok":
            blockers.append({"kind": "custody_unreadable", "task_id": member})
        else:
            blockers.extend({"kind": "delegated_run", "task_id": member,
                             "run_id": str(run.get("run_id") or ""),
                             "invocation_id": str(run.get("invocation_id") or ""),
                             # The owner Pause's consumers read it; Continue's hold does not.
                             "review_owned": bool(run.get("review_owned")),
                             # The observer's own facts (#1721): running, or a start/stop outcome unknown.
                             **{key: str(run.get(key) or "") for key in ("state", "stop_outcome", "source")}}
                            for run in observed.get("runs") or [] if isinstance(run, dict)
                            and not (for_continue and run.get("independent_review") is True))
    try:
        from ouroboros.launch_custody import unsettled_rows
        from ouroboros.tool_custody import task_process_blockers
        blockers.extend(task_process_blockers(custody_root, members))
        for row, state in unsettled_rows(custody_root, members):
            pid = int(row.get("pid") or 0)
            blockers.append({"kind": "owned_process", "task_id": row.get("owner_task"), "pid": pid}
                            if state == "live" else
                            {"kind": "process_custody_unreadable", "detail": f"launch of process {pid} unobserved"})
    except Exception as exc:
        blockers.append({"kind": "process_custody_unreadable", "detail": str(exc)[:200]})
    blockers.extend(model_handoffs(custody_root, predecessor, member_results=facts.rows,
                                   for_continue=for_continue, attempts=attempts))
    return blockers


def model_handoffs(custody_root, predecessor, *, member_results=None, for_continue=False, attempts=None):
    """Read the same unresolved model recipients for admission and display. Never repairs.
    ``attempts``: the root's open attempts the caller already read (``None``: read here). A tree with
    none reads no member; one with some reads them from the census's memo (``tree_census``)."""
    blockers = []
    try:
        from ouroboros.local_custody_repair import ended_unbound_model_request, receiver_reason, retired_unbound_send
        from ouroboros.owner_pause import tree_model_requests
        if attempts is None:  # the root's open attempts only (the store's open-set index)
            attempts = tree_model_requests(custody_root, predecessor)
        if attempts and member_results is None:
            facts = writer_facts(custody_root, predecessor)
            if facts.census_error:
                raise ValueError(facts.census_error)
            member_results = facts.rows
        for attempt in attempts:
            if attempt.get("state") in {"dispatched", "unresolved"}:
                consumer = attempt.get("local_answer_consumer_id")
                retirements = (member_results.get(str(attempt.get("task_id") or ""), {})
                               .get("retired_model_consumers") or {})
                retired = retirements.get(consumer) if consumer else None
                if (isinstance(retired, dict) and retired.get("retired_at")
                        and type(attempt.get("local_answer_task_attempt")) is int
                        and retired.get("task_attempt") == attempt["local_answer_task_attempt"]):
                    continue
                if ended_unbound_model_request(attempt) or retired_unbound_send(attempt, retirements):
                    continue
                # A reviewer's own send (its slot/skill attribution) is review
                # custody: the owner Pause's readers list it as still finishing.
                review_owned = bool(attempt.get("review_slot_id") or attempt.get("review_skill"))
                # Only the review substrate's per-slot scope binds that attribution,
                # and every send there reads the candidate without writing it, whatever
                # ``source`` its route stamps (llm.anthropic, web_search.*, extraction).
                # A review ``source`` without it has no current producer and holds.
                if for_continue and review_owned:
                    continue
                blockers.append({"kind": "model_handoff", "attempt_id": attempt.get("attempt_id"),
                                 "task_id": attempt.get("task_id"), "review_owned": review_owned,
                                 "reason": receiver_reason(attempt)})
    except Exception as exc:
        blockers.append({"kind": "attempt_custody_unreadable", "detail": str(exc)[:200]})
    return blockers


def trees_model_handoffs(custody_root: Any, roots: Any) -> Dict[str, List[Dict[str, Any]]]:
    """``model_handoffs`` of many trees over ONE read of their dispatched/unresolved attempts, the only states
    it names: the display reader's seam over every paused root. Only a tree with such an attempt touches its
    members, from the memo. An unreadable attempt store answers each tree as unreadable, as a single read
    would: unknown, never none."""
    from ouroboros.owner_pause import trees_model_requests

    roots = [str(root) for root in roots]
    if not roots:
        return {}
    try:
        attempts = trees_model_requests(custody_root, roots)
    except Exception as exc:
        unreadable = {"kind": "attempt_custody_unreadable", "detail": str(exc)[:200]}
        return {root: [dict(unreadable)] for root in roots}
    return {root: model_handoffs(custody_root, root, attempts=attempts.get(root) or []) for root in roots}


_RUNNING_KINDS = frozenset({"running_member", "dispatchable_member", "late_phase_running", "owned_process"})


def only_running_writers(blockers: Any) -> bool:
    """Whether every blocker a census named is a positively observed running writer (#1721).

    Only then is "wait for it or stop it" true advice. An unbound start, an
    unknown stop, an outstanding model send or unreadable custody is reported as
    that fact, also beside a running writer: Stop may end spending without
    settling it, and only new facts do.
    """
    rows = list(blockers or ())
    return bool(rows) and all(isinstance(b, dict) and (
        b.get("kind") in _RUNNING_KINDS or b.get("kind") == "delegated_run" and b.get("state") == "running")
        for b in rows)


def writers_unsettled_refusal(blockers: Any) -> Dict[str, Any]:
    """A held Continue's Resume refusal (#1721): the whole census, which the UI reduces
    to a few facts and a truthful remainder count, and wait-or-stop advice only when
    every one of its rows is a running writer."""
    rows = list(blockers or ())
    return {"ok": False, "error": "predecessor_writers_unsettled", "blockers": rows,
            **({"action": "wait_or_stop_the_previous_work"} if only_running_writers(rows) else {})}


def forget(custody_root: Any = None) -> None:
    """Drop what is held (one data root's, or all): a store replaced under this process."""
    with _LOCK:
        if custody_root is None:
            _MEMO.clear()
            return
        for key in [key for key in _MEMO if key[0] == str(pathlib.Path(custody_root))]:
            del _MEMO[key]
