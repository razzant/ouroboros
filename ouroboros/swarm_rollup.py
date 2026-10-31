"""Addressed fan-out facts with one-time journal catch-up for legacy parents.

The producer commits one immutable event in the task's existing row-backed fields
before best-effort diagnostic publication. A lost log append cannot lose counts;
a retry of that event id cannot double them. No new task is created for telemetry.
Like the old log append, a failed durable recording never prevents task launch.
An exact base witness survives older writers separately from their mutable rollup.
Only legacy first reads or invalidated witnesses inspect the shared journal.
"""

import copy
import hashlib
import json
import pathlib
from json import dumps, loads

from ouroboros.jsonl_tail import JsonlChainSnapshot
from ouroboros.utils import jsonl_chain_handles

_HISTORY_FIELDS = ("fanout_count", "requested_count", "fanout_interval_sec_total",
                   "child_ids", "lanes_requested", "included_event_ids")


def initial_swarm_rollup(drive_root):
    """A newly admitted task has no prior emissions; no journal inspection is needed."""
    return {"version": 2, "fanout_count": 0, "requested_count": 0,
            "fanout_interval_sec_total": 0, "child_ids": [], "lanes_requested": []}


def _state(row):
    return {"base": row.get("swarm_fanout_rollup"), "witness": row.get("swarm_fanout_basis")}


def _identity(state):
    return hashlib.sha256(dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _proven(state):
    base, witness = state["base"], state["witness"]
    return (isinstance(base, dict) and base.get("version") == 2 and
            ((isinstance(witness, dict) and "conflicts" not in witness and witness.get("base") == base)
             or (witness is None and base == initial_swarm_rollup(None))))


def _empty(state):
    base = state["base"] or {}
    if not isinstance(base, dict):
        return False
    legacy_empty = (state["witness"] is None and base.get("version") != 2
                    and {"source_root", "through", "child_ids", "lanes_requested"} <= base.keys())
    return (_proven(state) or legacy_empty) and not any(base.get(key) for key in _HISTORY_FIELDS)


def retain_swarm_basis(txn, document, *, current=None):
    """One selection for lifecycle writes, copy-back, import and reconciliation.

    An older writer changes the rollup but preserves the separate witness. Only
    a repair of that EXACT state may replace it; cursor size is not ancestry.
    Empty proven bases contribute nothing in either direction. Independent
    nonempty histories retain both sources as unavailable, never guessed sums.
    Reuse the pristine document held by update_task's write transaction; direct
    upserts read their current body, whose basis counts only while it decodes to
    this task's object: a torn or foreign body is what the upsert replaces. No
    journal I/O or cross-write cache.
    """
    if current is None:
        task_id = str(document["task_id"])
        record = txn.conn.execute("SELECT body FROM task_bodies WHERE task_id=?", (task_id,)).fetchone()
        try:
            current = loads(record[0]) if record else {}
        except ValueError:
            current = {}
        if not isinstance(current, dict) or str(current.get("task_id") or "") != task_id:
            current = {}
    held, incoming = _state(current), _state(document)
    held_proof = held["witness"] if isinstance(held["witness"], dict) else {}
    incoming_proof = incoming["witness"] if isinstance(incoming["witness"], dict) else {}
    if incoming["base"] is None and incoming["witness"] is None:
        chosen = held
    elif (held["base"] is None and held["witness"] is None) or held == incoming:
        chosen = incoming
    elif held["witness"] == incoming["witness"] and (
            held["witness"] is not None or
            (isinstance(held["base"], dict) and isinstance(incoming["base"], dict)
             and held["base"].get("source_root") == incoming["base"].get("source_root"))):
        chosen = incoming  # old loader advances its mutable base; the witness stays invalidated
    elif _identity(held) in incoming_proof.get("replaces", []):
        chosen = incoming
    elif _identity(incoming) in held_proof.get("replaces", []):
        chosen = held
    elif _empty(held):
        chosen = incoming
    elif _empty(incoming):
        chosen = held
    else:
        sources = {}
        for state, proof in ((held, held_proof), (incoming, incoming_proof)):
            for source in proof.get("conflicts", [state]):
                sources[_identity(source)] = source
        chosen = {"base": held["base"], "witness": {"conflicts": list(sources.values())}}
    if chosen["base"] is not None:
        document["swarm_fanout_rollup"] = chosen["base"]
    if chosen["witness"] is not None:
        document["swarm_fanout_basis"] = chosen["witness"]
    elif chosen["base"] == initial_swarm_rollup(None):
        document["swarm_fanout_basis"] = {"base": copy.deepcopy(chosen["base"])}
    else:
        document.pop("swarm_fanout_basis", None)


def record_swarm_fanout(drive_root, event):
    """Commit this host-minted emission id once, on its existing parent row.

    The full event survives with the document across copy-back/import/export.
    Whole-document writes union these immutable rows; only task deletion retires
    them. A retry carries the same id and bytes, while another emission gets a
    fresh id even when it requests the same children or group.
    """
    from ouroboros.task_store import patch_handoff

    def retain(current):
        if current is not None:
            if current != event:
                raise ValueError("swarm emission id already holds different facts")
            return None
        return event

    return patch_handoff(drive_root, str(event["parent_task_id"]), "swarm_fanouts", str(event["event_id"]), retain)


def rollup_from_document(row):
    """Fold only this task's addressed emissions over its admitted/repaired base.

    Missing/legacy authority is unavailable, never evidence of zero fan-out.
    The rollup's existing counters describe observed emissions, not successful
    children, dispatch lanes or telemetry that failed to be durably recorded.
    """
    prior = row.get("swarm_fanout_rollup")
    if not _proven(_state(row)):
        raise ValueError("swarm history requires explicit repair")
    facts = copy.deepcopy(prior)
    children, lanes = set(facts.get("child_ids", [])), list(facts.get("lanes_requested", []))
    included = set(facts.get("included_event_ids", []))
    for identity, event in (row.get("swarm_fanout_basis") or {}).get("includes", {}).items():
        if identity in (row.get("swarm_fanouts") or {}) and row["swarm_fanouts"][identity] != event:
            raise ValueError("swarm emission id has inconsistent history")
    for identity, event in (row.get("swarm_fanouts") or {}).items():
        if identity not in included:
            _accumulate(facts, children, lanes, event, str(row["task_id"]))
    facts.update(child_ids=sorted(children), lanes_requested=lanes)
    return facts


def load_swarm_rollup(drive_root, task_id):
    """Read addressed facts, catching up a legacy base once before using them.

    Missing or damaged history and concurrent repair/emission stay unavailable;
    no retry loop hides them. A later observation can retry that same catch-up.
    """
    from ouroboros.task_results import load_task_result

    row = load_task_result(drive_root, task_id, strict=True) or {}
    if not _proven(_state(row)):
        return repair_swarm_rollup(drive_root, task_id)
    return rollup_from_document(row)


def repair_swarm_rollup(drive_root, task_id):
    """Legacy catch-up, at the first read or an explicit repair request.

    Addressed ids already counted by an old cursor stay in the repaired base.
    A concurrent emitter or repair invalidates this snapshot instead of losing
    or double-counting an emission. Torn/shortened/unreadable chains publish
    nothing. A raced repair raises; only a later observation retries it.
    """
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document

    root = pathlib.Path(drive_root)
    row = load_task_result(root, task_id, strict=True) or {}
    prior = row.get("swarm_fanout_rollup")
    if _proven(_state(row)):
        return rollup_from_document(row)
    witness = row.get("swarm_fanout_basis") or {}
    if not isinstance(witness, dict) or (prior is not None and not isinstance(prior, dict)):
        raise ValueError("swarm legacy basis is unreadable")
    if "conflicts" in witness:
        raise ValueError("swarm legacy sources have unproven overlap")
    # The baseline loader copies unknown top-level fields, even when it resets
    # a foreign cursor. Resume the witnessed base, not its changed counters:
    # addressed diagnostics after that cursor may have been counted repeatedly.
    saved = witness.get("base", prior)
    for identity, event in witness.get("includes", {}).items():
        if identity in (row.get("swarm_fanouts") or {}) and row["swarm_fanouts"][identity] != event:
            raise ValueError("swarm emission id has inconsistent history")
    if not witness and isinstance(prior, dict) and prior.get("version") == 2:
        raise ValueError("swarm v2 base has no inclusion witness")
    if isinstance(prior, dict) and prior.get("source_root") not in (None, str(root.resolve())):
        if any(prior.get(key) for key in _HISTORY_FIELDS):
            raise ValueError("swarm legacy source is unavailable at this root")
    if isinstance(saved, dict) and saved.get("source_root") != str(root.resolve()):
        if any(saved.get(key) for key in _HISTORY_FIELDS):
            raise ValueError("swarm legacy source is unavailable at this root")
        saved = None
    facts = copy.deepcopy(saved) if isinstance(saved, dict) else {}
    children, lanes = set(facts.get("child_ids", [])), list(facts.get("lanes_requested", []))
    events = row.get("swarm_fanouts") or {}
    source = JsonlChainSnapshot(root / "logs/events.jsonl")
    if not source.entries:
        raise ValueError("swarm event history is absent")
    prefix = facts.get("through", 0)
    if isinstance(prior, dict) and prior.get("source_root") == str(root.resolve()):
        if prior.get("through", 0) > source.upper or prior.get("through", 0) < prefix:
            raise ValueError("swarm event chain shortened")
    if prefix and events and not witness:
        included = set()

        def overlap(event):
            identity = event.get("event_id")
            if identity in events:
                if event != events[identity]:
                    raise ValueError("swarm emission id has inconsistent history")
                if identity in included:
                    raise ValueError("swarm legacy base repeats an addressed emission")
                included.add(identity)

        # Only this one-time legacy conversion reads its consumed prefix. An old
        # writer can have counted a new producer's log already; prove those ids
        # rather than subtract guessed amounts or double-fold their task rows.
        if fold_event_suffix(root, 0, overlap, source=source, max_bytes=prefix) != prefix:
            raise ValueError("swarm event prefix is incomplete")
        facts["included_event_ids"] = sorted(included)
    through = fold_event_suffix(root, facts.get("through", 0), lambda event: (
        _accumulate(facts, children, lanes, event, task_id) if event.get("event_id") not in events else None), source=source)
    if through != source.upper:
        raise ValueError("swarm event history is incomplete")
    facts.update(version=2, source_root=str(root.resolve()), through=through,
                 child_ids=sorted(children), lanes_requested=lanes)
    if not row:
        # Legacy/unbound readers can observe an intact journal without creating
        # a task result. An empty observation is not complete receipt coverage.
        return facts
    proof = {"base": copy.deepcopy(facts),
             "includes": {**witness.get("includes", {}),
                          **{identity: events[identity] for identity in facts.get("included_event_ids", [])
                             if identity in events}},
             "replaces": sorted(set(witness.get("replaces", [])) | {_identity(_state(row))})}

    def publish(current):
        if not current or _state(current) != _state(row) or (current.get("swarm_fanouts") or {}) != events:
            raise ValueError("swarm facts changed during repair; retry repair")
        return {**current, "swarm_fanout_rollup": facts, "swarm_fanout_basis": proof}

    return rollup_from_document(patch_document(root, task_id, publish))


def fold_event_suffix(root, through, consume, *, source=None, max_bytes=None):
    """Visit complete unconsumed lines; the caller commits effects and cursor together."""
    source = source or JsonlChainSnapshot(pathlib.Path(root) / "logs/events.jsonl")
    if through > source.upper:
        raise ValueError("swarm event chain shortened")
    start = through
    for index in range(len(source.entries)):
        _, end = source.segment(index)
        if through >= end:
            continue
        with jsonl_chain_handles(source.path, strict=True, start_offset=through, snapshot=source.snapshot) as handles:
            handle = handles[0][1]
            while through < end:
                # A memory bound on a batch, never a partial JSON row. One oversized
                # event is still consumed whole, so a cursor always makes progress.
                if max_bytes is not None and through - start >= max_bytes:
                    return through
                raw = handle.readline(end - through)
                if not raw.endswith(b"\n"):
                    if source.entries[index][2]:
                        break
                    raise ValueError("swarm event history is incomplete")
                through += len(raw)
                if not raw.strip():
                    continue
                event = json.loads(raw)
                if isinstance(event, dict):
                    consume(event)
    return through


def _accumulate(facts, children, lanes, event, task_id):
    if event.get("type") != "swarm_fanout" or str(event.get("parent_task_id") or event.get("task_id") or "") != task_id:
        return
    facts["fanout_count"] = facts.get("fanout_count", 0) + 1
    for key, value, convert in (("requested_count", event.get("requested_count"), int),
                                 ("fanout_interval_sec_total", event.get("fanout_interval_sec", event.get("inter_wave_latency_sec")), float)):
        try:
            facts[key] = facts.get(key, 0) + convert(value or 0)
        except (ValueError, TypeError):
            pass
    children.update(str(tid) for tid in event.get("task_ids") or [] if str(tid or "").strip())
    lane = str(event.get("requested_model_lane") or "").strip()
    if lane and lane not in lanes:
        lanes.append(lane)
