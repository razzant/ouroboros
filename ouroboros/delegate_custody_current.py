"""The store half of delegated custody: one row per run, reduced by the existing reducer.

``custody_runs`` is the authority for every run (indexed by task, project and invocation;
``state`` is the custody hold: ``held`` while the run is open, owned or holds an undisposed
patch, ``closed`` after). A pending start is the owner-held ``custody_open`` row
``invocation:<id>`` until its STARTED or definite refusal; a definite refusal moves it to
``custody_refused`` so the id's fate stays answerable. The task-scoped facts the evidence
reader needs without a run (the nudge stamp, a blocked attempt, patch verdicts) are the
``custody_marks`` row of the task. Closing a run also owes an addressed terminal-result
refresh; its receipt travels in ``delegated_runs`` until that refresh lands. A retained output
that lands after its run settled owes that receipt again (#1573): the receipt publishes it.

``publish`` is ONE ``BEGIN IMMEDIATE``: the reduced rows and the event's outbox row commit
together with its append ordinal and first-request/last-start invocation projection;
the chain append follows in commit order, and the outbox row is deleted only once the append
confirmed; a re-owed receipt is then refreshed outside any transaction. ``fold`` is the
importer's single pass over the retained chain.
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from json import loads as json_loads
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional

from ouroboros import store
from ouroboros.delegate_documents import DOCUMENT_IMPORT_TYPES, fold_document_import
from ouroboros.utils import append_jsonl

log = logging.getLogger(__name__)

HELD, CLOSED = "held", "closed"
# Task-scoped rows on the marks row: patch verdicts kept whole, document imports folded by input.
MARK_TYPES = ("delegate_run_patch_verdict", *DOCUMENT_IMPORT_TYPES)
# Task-scoped attempt facts without a run: the evidence reader's ``nanny_nudge_recorded`` and
# ``delegate_start_attempted`` (the constants live beside that reader).
_NUDGE = "delegate_run_nanny_nudge_injected"
_ATTEMPTS = ("delegate_run_start_requested", "delegate_run_start_blocked", "delegate_run_configured_startup_fault")


def _decode(raw):
    from ouroboros.delegate_custody import RunCustody
    raw = deepcopy(raw)  # Reducer mutations must not also mutate the comparison snapshot.
    entry = RunCustody(**{f.name: raw[f.name] for f in fields(RunCustody) if f.name in raw})
    entry.verified_source_ranges = [tuple(pair) for pair in entry.verified_source_ranges]
    entry.output_reader_receipts = tuple(tuple(pair) for pair in entry.output_reader_receipts)
    for key, value in raw.items():
        if key.startswith("_"):
            setattr(entry, key, value)
    return entry


def collection_pending(entry, root=None) -> bool:
    """Collection debt, never retry eligibility. Only custody facts, even with a root."""
    from ouroboros.delegate_capture_attempt import capture_required, named_required

    if entry is None or entry.review_owned:
        return False
    if entry.terminal_source and (not entry.ledger_recorded or not entry.settled
        or not entry.output_source or not entry.output_complete
        or entry.output_terminal_sha != entry.terminal_source.get("sha256")
        or (capture_required(entry) and not entry.patch_captured)
        or (named_required(entry) and entry.named_evidence.get("status") != "retained")):
        return True
    return False


def _held(entry) -> bool:
    return (not entry.settled or entry.project_owned or collection_pending(entry)
            or (entry.mode != "plan" and (entry.snapshot_id or (entry.resource_ref.get("workspace_kind") == "directory"
                                      and entry.resource_ref.get("strategy") == "copy"))
                and not entry.patch_disposed))


def _decode_all(rows: Dict[str, Dict[str, Any]]):
    return {rid: _decode(facts) for rid, facts in rows.items()}


# Keep this superset in sync with delegate_custody.settled_output_unread/undisposed_patches,
# _held here, and tools.delegate_integration._payload_delegation_busy; Python decides eligibility.
# ``machine_json`` writes NaN/Infinity, which Python reads but ``json_valid`` rejects (and newer
# SQLite reads NaN as NULL). Non-ordinary JSON passes, including invalid JSON's decode error.
# Valid JSON with malformed field shapes outside the selection is not globally validated here,
# nor are duplicate keys (SQLite reads the first, Python the last; writers store dicts).
# ``_MAYBE_UNREAD`` drops only a falsy ``output_complete`` (absent, null, false, numeric zero) or
# an ``output_consumed`` of true/numeric one; ``_MAY_HOLD_PAYLOAD`` drops only a payload run whose
# ``settled`` is true/numeric one AND whose ``patch_disposed`` is a non-empty string. Any other
# value (text, arrays, 2, ...) passes and Python decides. An absent field reads as its ``_decode``
# default, so every JSON term is coalesced: a SQL NULL would drop the row, not let Python decide.
_MAYBE_UNREAD = ("CASE WHEN json_valid(facts) THEN (COALESCE(json_extract(facts, '$.output_complete'), 0)<>0 "
                 "AND COALESCE(json_extract(facts, '$.output_consumed'), 0)<>1) ELSE 1 END")
_MAY_HOLD_PAYLOAD = ("CASE WHEN json_valid(facts) THEN "
                     "(COALESCE(json_extract(facts, '$.authority_source'), '')='skill_payload' "
                     "AND NOT (COALESCE(json_extract(facts, '$.settled'), 0)=1 "
                     "AND COALESCE(json_type(facts, '$.patch_disposed'), '')='text' "
                     "AND COALESCE(json_extract(facts, '$.patch_disposed'), '')<>'')) ELSE 1 END")


def _selected(txn, typed: str, legacy: str, params: tuple = ()):
    """Indexed current facts plus only the partial index's underived legacy rows."""
    from ouroboros.store_hot_facts import VERSION

    if int(txn.meta("schema_version") or VERSION) < VERSION:
        return legacy, params  # the import owner has not prepared this store's columns yet
    return ("run_id IN (SELECT run_id FROM custody_runs WHERE facts_derived=1 AND (" + typed + ") "
            "UNION SELECT run_id FROM custody_runs INDEXED BY custody_underived_hot "
            "WHERE facts_derived IS NULL AND (" + legacy + "))", params + params)


_TREE = "root_task_id=? OR (root_task_id='' AND task_id=?)"
_LEGACY_TREE = "json_extract(facts, '$.root_task_id')=? OR (json_extract(facts, '$.root_task_id')='' AND task_id=?)"


def runs(root, where: str = "", params: tuple = ()):
    """Custody selected by typed columns (all runs when unfiltered)."""
    with store.open_store(root, write=False) as txn:
        raw = txn.custody_raw_rows(where, params)
    return _decode_all({rid: json_loads(text) for rid, text in raw.items()})


# The held rows are a few current runs among every closed one. Under the planner statistics the maintenance
# child's ANALYZE writes, SQLite would rather walk every run in run_id order than sort these few (always before
# 3.46: tests/test_cost_contract.py B7); ``unlikely`` states the selection is rare, so the read stays on its index.
HELD_WHERE = "unlikely(state=?)"


def held_runs(root, also: str = "", params: tuple = ()):
    """The held runs, narrowed by ALSO (a predicate over the row with PARAMS)."""
    return runs(root, f"{HELD_WHERE} AND ({also})" if also else HELD_WHERE, (HELD, *params))


def _any_of(*terms: str) -> str:
    """A disjunction read as the union of its terms, each on its own index. Under the same statistics
    SQLite 3.46+ plans ``state=? OR output_unread=1`` (two-valued columns) as a scan of every run."""
    return "run_id IN (" + " UNION ".join(f"SELECT run_id FROM custody_runs WHERE {term}" for term in terms) + ")"


def health_runs(root, task_id: str = ""):
    """The rows the context's delegated-run lines can name, in ONE read and ``run_id`` order:
    the reader's own runs, every held run (an undisposed patch is held) and every run whose
    output may be unread. A superset: the callers' own predicates decide, as over ``runs``."""
    terms, legacy, params = [HELD_WHERE, "unlikely(output_unread=1)"], f"state=? OR {_MAYBE_UNREAD}", (HELD,)
    if task_id:
        terms, legacy, params = ["task_id=?", *terms], f"task_id=? OR {legacy}", (str(task_id), HELD)
    with store.open_store(root, write=False) as txn:
        raw = txn.custody_raw_rows(*_selected(txn, _any_of(*terms), legacy, params))
    return _decode_all({rid: json_loads(text) for rid, text in raw.items()})


def run(root, run_id: str):
    with store.open_store(root, write=False) as txn:
        facts = txn.custody_row(str(run_id or ""))
    return None if facts is None else _decode(facts)


def tree_task_ids(root, tree_id: str) -> List[str]:
    """Every task of a tree that ever delegated (a bound run, a pending or a refused request) —
    the census the root's sleep check needs for members whose result row is gone (one SQL pass
    over the run facts, no Python decode, plus the request rows)."""
    from ouroboros.store_hot_facts import tree_requests

    tree = str(tree_id or "")
    with store.open_store(root, write=False) as txn:
        where, params = _selected(txn, _TREE, _LEGACY_TREE, (tree, tree))
        ids = {record["task_id"] for record in txn.conn.execute(
            f"SELECT DISTINCT task_id FROM custody_runs WHERE task_id<>'' AND {where}", params)}
        for _name, _identity, facts in tree_requests(txn, tree, refused=True):
            request = facts.get("request") if isinstance(facts.get("request"), dict) else {}
            tid = str(request.get("task_id") or "")
            if tid:
                ids.add(tid)
    return sorted(ids)


def tree_billing_group(root, tree_id: str) -> Optional[Dict[str, Any]]:
    """The admission bucket a tree's delegated starts carried (a bound run's request, else a
    pending request): the late settlement's lineage when no execution scope is active. An
    older start's ``unavailable:`` stand-in names no bucket and is skipped."""
    from ouroboros.usage_admission import UNAVAILABLE_GROUP_PREFIX
    from ouroboros.store_hot_facts import tree_requests

    tree = str(tree_id or "")
    with store.open_store(root, write=False) as txn:
        where, params = _selected(txn, f"billing_group_id<>'' AND ({_TREE})",
            f"json_extract(facts, '$.billing_group.billing_group_id')<>'' AND ({_LEGACY_TREE})", (tree, tree))
        carried = [json_loads(record["facts"]) for record in txn.conn.execute(
            f"SELECT facts FROM custody_runs WHERE {where} ORDER BY updated_at, run_id", params)]
        requests = [facts["request"] for _name, key, facts in tree_requests(txn, tree) if key.startswith("invocation:")]
    for facts in carried:
        group = facts.get("billing_group") or {}
        if "billing_group_limit_usd" in group and not str(group["billing_group_id"]).startswith(UNAVAILABLE_GROUP_PREFIX):
            return dict(group)
    for request in requests:
        group = request.get("billing_group") or {}
        if (str(request.get("root_task_id") or request.get("task_id") or "") == tree
                and group.get("billing_group_id") and "billing_group_limit_usd" in group
                and not str(group["billing_group_id"]).startswith(UNAVAILABLE_GROUP_PREFIX)):
            return dict(group)
    return None


def marks(root, task_id: str) -> Dict[str, Any]:
    with store.open_store(root, write=False) as txn:
        return txn.get_row("custody_marks", str(task_id or "")) or {}


def _pending_rows(txn) -> List[Dict[str, Any]]:
    return [facts["request"] for key, facts in txn.set_rows("custody_open").items() if key.startswith("invocation:")]


def pending(root):
    from ouroboros.delegate_pending import current_rows

    with store.open_store(root, write=False) as txn:
        rows = txn.set_rows("custody_open")
    return current_rows(root, [facts for key, facts in rows.items() if key.startswith("invocation:")])


def task_snapshot(root, task_id: str = "", *, root_task_id: str = "", wait_sec=store.STORE_WAIT_SEC):
    """One transaction's task runs and pending invocations; STARTED crosses no read gap.

    Runs use the task index. The request predicate scans only custody_open:
    closed requests leave that set, so cost follows open requests, not history.
    Only this task's typed pending facts are decoded; no request source is opened.
    """
    from ouroboros.delegate_pending import current_rows
    from ouroboros.store_hot_facts import tree_requests

    mine = str(task_id or "")
    with store.open_store(root, write=False, wait_sec=wait_sec) as txn:
        if root_task_id:
            where, params = _selected(txn, _TREE, _LEGACY_TREE, (root_task_id, root_task_id))
            raw = txn.custody_raw_rows(where, params)
            requests = [facts for _, _, facts in tree_requests(txn, root_task_id)]
        else:
            raw = txn.custody_raw_rows("task_id=?", (mine,))
            requests = [json_loads(row["facts"]) for row in txn.conn.execute(
                "SELECT facts FROM obligations WHERE set_name='custody_open' "
                "AND CAST(json_extract(facts, '$.request.task_id') AS TEXT)=? ORDER BY rowid", (mine,))]
    runs_by_id = {rid: json_loads(text) for rid, text in raw.items()}
    if root_task_id and mine:
        runs_by_id = {rid: facts for rid, facts in runs_by_id.items() if facts.get("task_id") == mine}
        requests = [facts for facts in requests if facts["request"].get("task_id") == mine]
    return {"runs": _decode_all(runs_by_id), "pending": current_rows(root, requests)}


def payload_snapshot(root):
    """ONE transaction's view of who may hold a skill payload: the payload runs not surely
    settled and disposed, and every pending invocation (a holder whose REQUESTED becomes
    STARTED between two reads would be in neither). Target, owner and terminality checks
    belong to the caller, after this transaction closes."""
    from ouroboros.delegate_pending import current_rows

    with store.open_store(root, write=False) as txn:
        raw = txn.custody_raw_rows(*_selected(txn, "payload_held=1", _MAY_HOLD_PAYLOAD))
        requests = [facts for key, facts in txn.set_rows("custody_open").items() if key.startswith("invocation:")]
    held = {rid: json_loads(text) for rid, text in raw.items()}
    return {"runs": _decode_all(held), "pending": current_rows(root, requests)}


def invocation(root, invocation_id: str, request_facts: Callable) -> Optional[Dict[str, Any]]:
    """The first request and last STARTED, with an explicit path for legacy rows."""
    from ouroboros.delegate_invocation import read

    return read(root, invocation_id, request_facts)


def unreadable(root) -> bool:
    """Whether the store cannot answer custody (the ``custody_log_unreadable`` of the store era)."""
    try:
        with store.open_store(root, write=False) as txn:
            txn.phase()
        return False
    except store.StoreError:
        return True


def snapshot(root, closed: bool = True):
    """One read: held runs and pending invocations, optionally including owed closing receipts."""
    with store.open_store(root, write=False) as txn:
        debts = txn.set_rows("delegated_runs") if closed else {}
        held = txn.custody_rows(HELD_WHERE, (HELD,))
        invocations = [facts for key, facts in txn.set_rows("custody_open").items() if key.startswith("invocation:")]
    from ouroboros.delegate_pending import current_rows

    state = {}
    for facts in debts.values():
        state.update({rid: _decode(raw) for rid, raw in facts.get("closed_runs", {}).items()})
    state.update(_decode_all(held))
    return {"state": state, "pending": current_rows(root, invocations), "current": True}



def _reduce(txn, event: Dict[str, Any], state: Dict[str, Any], marks_by_task: Dict[str, Dict[str, Any]],
            pending=None) -> str:
    """Apply ONE event to the rows it touches (``state`` holds the decoded runs the caller
    loaded; a PROJECT_RETIRED needs its project's siblings, a STARTED its invocation row).
    Returns the owner whose closing receipt a late retained output re-owed ("" otherwise)."""
    from ouroboros import delegate_custody as c
    from ouroboros import delegate_invocation as inv

    event_id = str(event.get("event_id") or "")
    if event_id and txn.get_row(inv.EVENTS, event_id) is not None:
        return ""  # A repeated append/import cannot become the invocation's last start.
    ordinal = inv.event_ordinal(txn, event)
    txn.set_meta(inv.ORDINAL, ordinal)
    if event_id:
        txn.put_row(inv.EVENTS, event_id, {"ordinal": ordinal})
    invocation_row = inv.reduce_event(txn, event, ordinal)
    kind, rid, iid = str(event.get("type") or ""), str(event.get("run_id") or ""), str(event.get("invocation_id") or "")
    tid = str(event.get("task_id") or "")
    if kind == c.START_REQUESTED and iid:
        key = f"invocation:{iid}"
        if (txn.get_row("custody_open", key) is None and txn.get_row("custody_refused", iid) is None
                and not inv.bound_runs(txn, iid)):  # a replayed request of an answered invocation re-opens nothing
            txn.put_row("custody_open", key, {"request": event, **({"pending": pending} if pending is not None else {})})
    before = {run_id: deepcopy(vars(entry)) for run_id, entry in state.items()}
    c._apply(state, event)
    if kind == c.STARTED and rid in state:
        state[rid]._invocation_started_ordinal = ordinal
        request = (invocation_row or {}).get("request")
        if request is not None:
            state[rid].request_event = dict(request)
            group = request.get("billing_group")
            state[rid].billing_group = dict(group) if isinstance(group, dict) else {}
    # A retained output (its canonical ``source``) staged after the run settled: only the closing
    # receipt publishes it, and an earlier refresh may already have acknowledged that receipt.
    late = rid if kind == c.OUTPUT_SPILLED and event.get("staged") and isinstance(event.get("source"), dict) else ""
    owed = ""
    for run_id, entry in state.items():
        facts = vars(entry)
        changed = facts != before.get(run_id)
        if changed:
            txn.put_custody(run_id, facts, state=HELD if _held(entry) else CLOSED)
        if entry.settled and entry.task_id and (changed or run_id == late):
            # The closing receipt and the custody row commit together; a later fact on a
            # settled run (its patch disposition, its output ack) refreshes the receipt, and a
            # late retained output owes it again (an identical re-wait included).
            debt = txn.get_row("delegated_runs", entry.task_id)
            if debt is not None or (run_id == rid and kind in (c.SETTLED, c.CLOSED_ABSENT)) or run_id == late:
                debt = dict(debt or {})
                receipts = {**debt.get("closed_runs", {}), run_id: facts}
                txn.put_row("delegated_runs", entry.task_id, {**debt, "task_id": entry.task_id, "closed_runs": receipts})
            if run_id == late:
                owed = entry.task_id
    if iid and (kind == c.STARTED or (kind == c.START_FAILED and event.get("definite") is True)):
        request = txn.get_row("custody_open", f"invocation:{iid}")
        txn.delete_row("custody_open", f"invocation:{iid}")
        if kind == c.START_FAILED and request is not None:
            txn.put_row("custody_refused", iid, {"request": request["request"], "failure": event})
    if tid and (kind in MARK_TYPES or (not rid and kind in (_NUDGE, *_ATTEMPTS))):
        mark = marks_by_task.setdefault(tid, txn.get_row("custody_marks", tid) or {"task_id": tid})
        _mark(mark, event)
        if set(mark) - {"task_id"}:  # a review-owned attempt marks nothing: no empty record row
            txn.put_row("custody_marks", tid, mark)
    for target in _cost_targets(event, state, invocation_row):
        debt = txn.get_row("delegated_runs", target) or {"task_id": target}
        txn.put_row("delegated_runs", target, {**debt, "cost_revision": ordinal})
    return owed


def _cost_targets(event, state, invocation_row):
    """Custody-only changes owe own/root cost refresh, even without a money row."""
    from ouroboros import delegate_custody as c

    if event.get("type") not in (c.START_REQUESTED, c.STARTED, c.START_FAILED, c.TERMINAL_OBSERVED,
                                 c.LEDGER_RECORDED, c.SETTLED, c.CLOSED_ABSENT):
        return set()
    entry = state.get(str(event.get("run_id") or ""))
    request = (invocation_row or {}).get("request") or event
    owner = entry.task_id if entry else str(request.get("task_id") or event.get("task_id") or "")
    tree = (entry.root_task_id if entry else str(request.get("root_task_id") or "")) or owner
    return {owner, tree} - {""}


def _mark(mark: Dict[str, Any], event: Dict[str, Any]) -> None:
    from ouroboros import delegate_custody as c

    kind = str(event.get("type") or "")
    if kind == _NUDGE:
        mark["nanny_nudge"] = True
    elif kind in _ATTEMPTS:
        if not c.review_owned_source(event.get("source")):
            mark["start_attempted"] = True
    elif kind in DOCUMENT_IMPORT_TYPES:
        fold_document_import(mark, event)
    elif kind in MARK_TYPES:
        verdicts = list(mark.get("patch_verdicts", []))
        verdicts.append({key: value for key, value in event.items() if key != "event_id"})
        mark["patch_verdicts"] = verdicts


def _loaded(txn, event: Dict[str, Any]) -> Dict[str, Any]:
    """The runs ONE event may touch: its own, plus its project's siblings for a retirement."""
    from ouroboros import delegate_custody as c

    rid = str(event.get("run_id") or "")
    state = {}
    if str(event.get("type") or "") == c.PROJECT_RETIRED:
        project_id = str(event.get("project_id") or "")
        if rid and not project_id:
            own = txn.custody_row(rid)
            project_id = str((own or {}).get("project_id") or "")
        if project_id:
            state.update(_decode_all(txn.custody_rows("project_id=?", (project_id,))))
    if rid and rid not in state:
        facts = txn.custody_row(rid)
        if facts is not None:
            state[rid] = _decode(facts)
    if str(event.get("type") or "") == c.STARTED:
        prior = str(event.get("continuation_of") or "")
        if prior and prior not in state:
            facts = txn.custody_row(prior)
            if facts is not None:
                state[prior] = _decode(facts)  # ``_supersede`` hands the snapshot over
    return state


def publish(root, event: Dict[str, Any], *, propagate_contention: bool = False) -> bool:
    """Returns whether custody + outbox COMMITTED; append failure does not undo custody.
    A failed append keeps the outbox for the next flush, so the chain never leads the store.
    Transaction/append ordering: module docstring."""
    from ouroboros import delegate_invocation as inv

    owed = ""
    from ouroboros.delegate_custody import START_REQUESTED
    from ouroboros.delegate_pending import pending_invocations
    # Keep the canonical reference, never a replay-authorizing copy of its body.
    pending = pending_invocations(root, [event], materialize=False) if event.get("type") == START_REQUESTED else None
    try:
        while True:
            with store.open_store(root, write=True) as txn:
                if int(txn.meta("schema_version") or 1) >= store.SCHEMA_VERSION:
                    if not event.get("event_id") or txn.get_row(inv.EVENTS, str(event["event_id"])) is None:
                        event = {**event, "custody_ordinal": int(txn.meta(inv.ORDINAL) or 0) + 1}
                        owed = _reduce(txn, event, _loaded(txn, event), {}, pending)
                        txn.outbox_put(event)
                    break
            # Prepare an old child before reducing anything, like store.update_task.
            from ouroboros.store_import import upgrade_current_schema
            upgrade_current_schema(root)
    except store.StoreError as exc:
        if (propagate_contention and isinstance(exc, store.StoreBusy) and exc.reason == "contention"
                and getattr(exc, "rollback_confirmed", None) is not False):
            exc.custody_not_committed = True  # Only this pre-COMMIT boundary grants retry.
            raise
        log.warning("Custody %s not recorded under %s (%s)", event.get("type"), root, type(exc).__name__)
        return False
    flush_outbox(root)
    if owed:
        _refresh_owed_receipt(root, owed)
    return True


def _refresh_owed_receipt(root, task_id: str) -> None:
    """The addressed refresh of a receipt a late output re-owed, after COMMIT and outside any
    transaction: a terminal owner publishes the output now; a running owner keeps the receipt
    until its terminal write. A failure leaves the committed receipt owed for the sweep and the
    boot backfill, and never turns the committed custody fact into a failed emit."""
    from ouroboros.delegate_terminal import refresh_terminal_reconciliation

    try:
        refresh_terminal_reconciliation(root, task_id, trigger="output_spilled")
    except Exception:
        log.warning("Custody post-commit bookkeeping failed for %s; its closing receipt stays owed",
                    task_id, exc_info=True)


CHAIN_BOUNDARY = "custody_chain_offset"  # meta: the chain's length after this store's last own row
# meta: the live segment that length was last measured on (device, inode, size, the archive directory's
# mtime) and the offset it stood at, so the next flush advances the boundary by the live file's growth
# and lists the rotated segments only after a rotation or a removed segment changed what they add up to.
CHAIN_LIVE = "custody_chain_live"


def flush_outbox(root) -> int:
    """Append every parked event to the chain, oldest first; delete each row once its append
    confirmed. Idempotent: a replayed append carries the same ``event_id`` the fold dedups. The
    chain is measured between the append and the delete, under no store transaction (a listing of
    the archive never runs inside the store's write lock); the delete's transaction records it."""
    from ouroboros.delegate_custody import event_log_path

    path = event_log_path(root)
    try:
        with store.open_store(root, write=False) as txn:
            # BEGIN IMMEDIATE serializes insertion; clocks and event ids do not.
            # rowid also preserves pending legacy events written before ordinals.
            parked = [json_loads(row["event"]) for row in
                      txn.conn.execute("SELECT event FROM custody_outbox ORDER BY rowid")]
            known = txn.meta(CHAIN_LIVE)
    except store.StoreError:
        return 0
    appended = 0
    for event in parked:
        try:
            live = _append_durable(path, event)
            if not live:
                break
            measured = _chain_live(path, live, known)
            with store.open_store(root, write=True) as txn:
                txn.outbox_delete(str(event.get("event_id") or ""))
                if "ino" in measured:
                    txn.set_meta(CHAIN_LIVE, measured)
                # This store's own rows end here: P6' applies only what follows.
                txn.set_meta(CHAIN_BOUNDARY, max(int(txn.meta(CHAIN_BOUNDARY) or 0), measured["offset"]))
        except (store.StoreError, OSError):
            log.warning("custody outbox flush interrupted under %s", root, exc_info=True)
            break
        if "ino" in measured:
            known = measured
        appended += 1
    return appended


def _archive_stamp(path) -> Optional[int]:
    """The archive directory's mtime: a rotation into it or a segment removed from it moves it."""
    try:
        return os.stat(path.parent.parent / "archive").st_mtime_ns
    except OSError:
        return None


def _chain_live(path, live: Dict[str, int], known: Any) -> Dict[str, Any]:
    """The ``CHAIN_LIVE`` record of the chain right after the append that left its live segment at
    ``live`` (device, inode, size): ``known``, the previous flush's record, advanced by the live
    file's growth while the closed segments cannot have changed (the same live inode, the same
    archive directory, a live file that only grew); otherwise — the first flush, a rotation, a
    removed segment — one listing (``chain_measure``) with the live stat it was read with. Only
    ``offset`` when the live file is gone between its append and the measurement (nothing to
    advance from, no record to keep). A pure measurement under no transaction: a record another
    flusher wrote meanwhile agrees with this one (``offset - size`` is the same on the same inode and
    archive), so a stale ``known`` costs at most one listing, never a wrong boundary."""
    from ouroboros.delegate_custody import chain_measure

    stamp = _archive_stamp(path)
    if (isinstance(known, dict) and isinstance(known.get("size"), int) and isinstance(known.get("offset"), int)
            and [known.get("dev"), known.get("ino"), known.get("archive")] == [live["dev"], live["ino"], stamp]
            and live["size"] >= known["size"]):
        return {**live, "archive": stamp, "offset": known["offset"] + live["size"] - known["size"]}
    offset, measured = chain_measure(path)
    if measured is None:
        return {"offset": offset}
    return {"dev": measured.st_dev, "ino": measured.st_ino, "size": measured.st_size,
            "archive": _archive_stamp(path), "offset": offset}


def _append_durable(path, event: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """The chain append an outbox delete may follow: under the chain's append lock (never the
    unlocked fallback), the line boundary restored when the file ends mid-line, the bytes
    fsynced — and the directory on create — before the live segment's identity and size after
    the append are returned. A death after that leaves at most a replayed ``event_id`` the fold
    dedups; a death before it (``None``) leaves the outbox row. The general ``append_jsonl``
    keeps its fast path for the observational logs."""
    created = not path.exists()
    if not append_jsonl(path, event, ensure_record_boundary=True, require_lock=True):
        return None
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_APPEND | getattr(os, "O_BINARY", 0))
        try:
            os.fsync(fd)
            written = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError:
        log.warning("custody chain append not durable under %s", path.parent, exc_info=True)
        return None
    if created:
        store.fsync_directory(path.parent)
    return {"dev": written.st_dev, "ino": written.st_ino, "size": written.st_size}


def outbox_depth(root) -> int:
    with store.open_store(root, write=False) as txn:
        return len(txn.outbox_rows())



def fold(rows) -> Dict[str, Any]:
    """Fold a retained chain ONCE into every custody table's rows: ``runs`` (facts by run
    id, with the hold state), ``open``, ``refused``, ``marks`` and invocation/order facts."""
    from ouroboros import delegate_custody as c
    from ouroboros import delegate_invocation as inv

    state: Dict[str, Any] = {}
    requests: Dict[str, Dict[str, Any]] = {}
    pending_ids: List[str] = []
    refused: Dict[str, Dict[str, Any]] = {}
    marks_by_task: Dict[str, Dict[str, Any]] = {}
    invocations, operations, event_ordinals, cost_revisions = {}, {}, {}, {}
    ordinal = 0
    for row in c._unique_rows(rows):
        ordinal = max(ordinal + 1, int(row.get("custody_ordinal") or 0))
        if row.get("event_id"):
            event_ordinals[str(row["event_id"])] = {"ordinal": ordinal}
        kind, rid, iid = str(row.get("type") or ""), str(row.get("run_id") or ""), str(row.get("invocation_id") or "")
        tid = str(row.get("task_id") or "")
        if kind == c.START_REQUESTED:
            key = inv.operation_key(str(row.get("operation_id") or ""), str(row.get("slot_id") or ""))
            operations.setdefault(key, {"invocation_id": iid, "request_ordinal": ordinal})
        if iid and kind in (c.START_REQUESTED, c.STARTED, c.START_FAILED):
            invocations[iid] = inv.advance(invocations.get(iid), row, ordinal)
        if kind == c.START_REQUESTED and iid and iid not in requests:
            requests[iid] = row
            pending_ids.append(iid)
        c._apply(state, row)
        if kind == c.STARTED and rid in state:
            state[rid]._invocation_started_ordinal = ordinal
        if kind == c.STARTED and rid in state and iid in requests:
            state[rid].request_event = dict(requests[iid])
            group = requests[iid].get("billing_group")
            state[rid].billing_group = dict(group) if isinstance(group, dict) else {}
        if iid and iid in requests and (kind == c.STARTED or (kind == c.START_FAILED and row.get("definite") is True)):
            if iid in pending_ids:
                pending_ids.remove(iid)
                if kind == c.START_FAILED:
                    refused[iid] = {"request": requests[iid], "failure": row}
        if tid and (kind in MARK_TYPES or (not rid and kind in (_NUDGE, *_ATTEMPTS))):
            _mark(marks_by_task.setdefault(tid, {"task_id": tid}), row)
        for target in _cost_targets(row, state, invocations.get(iid)):
            cost_revisions[target] = ordinal
    # A replay recovers only still-open monetary custody, not a notification
    # for every closed run in history. Existing exact debts survive write_fold.
    open_owners = {target for entry in state.values()
                   if not entry.settled and not (entry.terminal_state in c.TERMINAL_STATES and entry.ledger_recorded)
                   for target in (entry.task_id, entry.root_task_id or entry.task_id)}
    open_owners.update(target for iid in pending_ids for target in
                       (requests[iid].get("task_id"), requests[iid].get("root_task_id")))
    return {
        "runs": {rid: (vars(entry), HELD if _held(entry) else CLOSED) for rid, entry in state.items()},
        "open": {f"invocation:{iid}": {"request": requests[iid]} for iid in pending_ids},
        "refused": refused,
        "marks": {tid: mark for tid, mark in marks_by_task.items() if set(mark) - {"task_id"}},
        "invocations": invocations, "operations": operations, "event_ordinals": event_ordinals,
        "ordinal": ordinal, "cost_revisions": {tid: rev for tid, rev in cost_revisions.items() if tid in open_owners},
    }


def write_fold(txn, folded: Dict[str, Any], *, replace: bool) -> None:
    """Write a fold's rows on an open transaction (the importer's build, the explicit rebuild)."""
    from ouroboros import delegate_invocation as inv

    if replace:
        txn.conn.execute("DELETE FROM custody_runs")
    for rid, (facts, hold) in folded["runs"].items():
        txn.put_custody(rid, facts, state=hold)
    for name in ("refused", "marks"):
        for identity, facts in folded[name].items():
            txn.put_row(f"custody_{name}", identity, facts)
    for name, rows in ((inv.INVOCATIONS, folded["invocations"]), (inv.OPERATIONS, folded["operations"]),
                       (inv.EVENTS, folded["event_ordinals"])):
        if replace:
            txn.replace_set(name, rows)
        else:
            for identity, facts in rows.items():
                txn.put_row(name, identity, facts)
    txn.set_meta(inv.ORDINAL, max(int(txn.meta(inv.ORDINAL) or 0), folded["ordinal"]))
    txn.set_meta(inv.COMPLETE, True)
    for tid, revision in folded.get("cost_revisions", {}).items():
        debt = txn.get_row("delegated_runs", tid) or {"task_id": tid}
        txn.put_row("delegated_runs", tid, {**debt, "cost_revision": max(revision, debt.get("cost_revision", 0))})
    held = txn.set_rows("custody_open")
    merged = dict(folded["open"]) if replace else {**folded["open"], **held}
    for key in set(held) - set(merged):
        txn.delete_row("custody_open", key)
    for key, facts in merged.items():
        if held.get(key) != facts:
            txn.put_row("custody_open", key, facts)
    if any(key.startswith("invocation:") and "pending" not in facts for key, facts in merged.items()):
        txn.set_meta("job:pending_invocations", {"state": "pending", "requested_revision": txn.revision() + 1})


def rebuild(root, *, replace=False) -> int:
    """Explicit migration/repair only: fold the retained chain once. The strict read precedes
    the write, so a segment that cannot be listed, opened or read raises before any retained
    row is replaced. Offline: a writer appending meanwhile is not coordinated with the fold."""
    from ouroboros import delegate_custody as c
    from ouroboros.obligations import ObligationsUnavailable

    try:
        rows = tuple(c._iter_rows(c.event_log_path(root), strict=True))
    except OSError as exc:
        raise ObligationsUnavailable(f"custody event chain is unreadable: {exc}") from exc
    folded = fold(rows)
    with store.open_store(root, write=True) as txn:
        write_fold(txn, folded, replace=replace)
        debts = txn.set_rows("delegated_runs")
        for tid, facts in debts.items():
            if not facts.get("closed_runs"):
                continue  # Cost-only debt does not re-owe historical cleanup receipts.
            receipts = {rid: run_facts for rid, (run_facts, _hold) in folded["runs"].items()
                        if run_facts.get("task_id") == tid and run_facts.get("settled")}
            if receipts:
                txn.put_row("delegated_runs", tid, {**facts, "closed_runs": {**facts.get("closed_runs", {}), **receipts}})
    return len(folded["runs"]) + len(folded["open"])
