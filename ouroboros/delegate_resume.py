"""A saved task's explicit decision about one ordinary start with a lost reply.

The request stays in open custody. Only the selected task's explicit Resume reads the decision
as permission. Every create handoff for the invocation shares its starter's short decision lock.
A never-started Continue may take that decision for its predecessor's start only when custody
proves the start had no write authority (owner, 24 Nov); the start then joins the Continue's own
custody (``CARRIED``), and the first proof of its activity holds that Continue again.
"""
from __future__ import annotations

import json
import logging
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from urllib.parse import quote

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros.owner_pause import OwnerPauseRefused

log = logging.getLogger(__name__)

LOOKUP_ONLY = "delegated_start_lookup_only"
BUSY, UNREADABLE = "delegated_start_custody_busy", "delegated_start_custody_unreadable"
ACTIVE = frozenset({"queued", "running"})
_RESUME = ContextVar("delegated_start_resume", default=None)
# The successor's own record of the predecessor starts it accepted: ``{invocation_id: its tree, member, invocation}``.
CARRIED = "accepted_predecessor_starts"
_RUN_CHAIN_MAX_LINKS = 16  # separate bound on engine continueFrom evidence, not task episodes
# The wire shapes with no write grant: an omitted or empty group, or the ordinary read-only replacement.
_READ_ONLY_EXECUTION = ({}, {"isolation": "envelope", "delegated": False},
                        {"isolation": "envelope", "delegated": False, "auxiliaryWritableRoots": []})
# Every other wire key (a thread or plan turn, a project, workspace, environment or resource input) proves nothing.
_READ_ONLY_BODY_KEYS = frozenset({"prompt", "instructions", "authPreference", "mode", "scope", "harnesses",
                                  "primaryHarness", "access", "execution", "model", "effort", "credentialProfileId",
                                  "maxSeconds", "continueFrom", "continueCarrier", "processingPreference",
                                  "promptCacheTtl"})
_WRITE_BINDINGS = ("execution_root", "snapshot_id", "execution_binding_fingerprint", "capture_id", "snapshot_task_id",
                   "thread_id", "resource_ref", "promotion", "authority_ceiling")


def address(request):
    member = str(request.get("task_id") or "")
    tree = str(request.get("root_task_id") or member)
    invocation = str(request.get("invocation_id") or "")
    if (not member or not invocation or request.get("surface") not in (None, "", "delegate")
            or (request.get("source") or "delegated_subagent") != "delegated_subagent"
            or (request.get("category") or "subagent") != "subagent"
            or any(request.get(k) for k in ("operation_id", "slot_id", *REVIEW_ATTRIBUTION_KEYS))):
        return {}
    target = "/".join(quote(part, safe="") for part in (tree, member, invocation))
    return {"root_task_id": tree, "task_id": member, "invocation_id": invocation,
            "choice_id": "delegate_start:" + target}


def _selection(selected):
    from ouroboros.control_actor import resume_author

    scope = _RESUME.get()
    if scope and not scope["panic_selection"] and scope["selected"] == selected and scope["author"] == resume_author():
        return scope
    return None


def decisions_in_use(selected):
    scope = _selection(selected)
    return list(scope["decisions"].values()) if scope else []


def with_resume_decisions(selected, resume, *, child=False, panic_selection=False):
    """Run the addressed Resume once. Its grant owner judges the complete, authorized refusal."""
    from ouroboros.control_actor import accepted_unknown, accepting_unknown, resume_author

    author = resume_author()
    if not (author.get("door") == "api_task_resume" and author.get("actor") == "owner"
            or author.get("actor") == "ouroboros" and author.get("task_id")):
        return resume()
    choice = accepted_unknown()
    scope = {"selected": selected, "author": author, "choice": choice, "decisions": {},
             "panic_selection": panic_selection}
    token = _RESUME.set(scope)
    try:
        # A child's predecessor is never the subject of its selection. A named start
        # likewise cannot answer the separate Continue/held-predecessor decision.
        with accepting_unknown(()) if child or panic_selection or names_start() else nullcontext():
            result = resume()
        risks = decisions_in_use(selected)
        return {**result, "accepted_start_risks": risks} if risks else result
    finally:
        _RESUME.reset(token)


def deciding_resume():
    return _RESUME.get() is not None


def selects(selected):
    """Addressed Resume alone: neither host release, mind list, nor a row selected on the Panic sheet."""
    from ouroboros.control_actor import list_row_call

    return _selection(selected) is not None and not list_row_call()


def names_start():
    scope = _RESUME.get()
    return bool(scope and any(key.startswith("delegate_start:") for key in scope["choice"].get("ids", ())))


def decide(root, tree, selected, blockers, *, fence_id, pause_id="", tree_actor=False,
           allow_starts=True, wait_id="", error="owner_pause_effects_unsettled", action="wait_for_effect_settlement",
           proven=None, hold_id=""):
    """Take choices only after ordinary grant authority, against one set of current blockers. ``proven``: the
    starts this never-started Continue may name, each with its tree (``predecessor_starts``), only in a Resume
    addressed alone (``selects``): no list row offers, records or reuses them. ``hold_id`` binds that choice to the
    exact queue hold it releases as well (checked under the queue lock by its caller)."""
    from ouroboros.work_stop import acceptable_unknown

    scope = _selection(selected)
    held, offers = [], {}
    wanted = set(scope["choice"].get("ids", ())) if scope else set()
    basis = {"fence_id": fence_id, "pause_id": pause_id}
    if wait_id:
        basis["wait_id"] = wait_id
    if hold_id:
        basis["hold_id"] = hold_id
    alone = bool(proven) and selects(selected)
    for row in blockers:
        start = row.get("ordinary_start") or {}
        member = row.get("task_id")
        carried = bool(alone and start.get("root_task_id")
                       and proven.get(start.get("invocation_id")) == start["root_task_id"])
        if not tree_actor and member != selected and not carried:
            # This writer can veto a release, but cannot be accepted by the selected child.
            return {"ok": False, "error": error, "blockers": blockers, "action": action,
                    "selection_scope_mismatch": True}
        if (allow_starts and scope and start and (start.get("root_task_id") == tree or carried)
                and start.get("task_id") == member
                and start.get("invocation_id") == row.get("invocation_id") and not row.get("run_id")
                and row.get("stop_outcome") == "pending_invocation_unbound"):
            saved = (selected, fence_id, pause_id) + ((wait_id,) if wait_id else ()) + ((hold_id,) if hold_id else ())
            choice = "/".join([start["choice_id"], *(quote(v, safe="") for v in saved)])
            decision = (row.get("resume_decisions") or {}).get(selected) or {}
            if (decision.get("choice_id") == choice and decision.get("creation") == "lookup_only"
                    and decision.get("resuming_task_id") == selected and all(decision.get(k) == v for k, v in basis.items())):
                scope["decisions"][choice] = decision
                continue
            offers[choice] = (start, start["root_task_id"] if carried else "")
            # ``read_only``: positive custody proof, so the owner is not told it may change files.
            row = {**row, "choice_id": choice, **({"read_only": True} if carried else {})}
        held.append(row)
    # A failed grant can leave the choice and its notice committed. Confirm it
    # against current custody, but record only choices that are still owed.
    if scope:
        wanted -= scope["decisions"].keys()
    offered = set(acceptable_unknown(held))
    if wanted and not wanted <= offered:
        if not held or offered:
            return {"ok": False, "error": "selection_authority_changed"}
    elif wanted and (not wanted.intersection(offers) or scope["choice"].get("text")):
        accepted = set()
        for key in sorted(wanted.intersection(offers)):
            start, successor_of = offers[key]
            decision = _record(root, start, selected, scope["author"], scope["choice"]["text"],
                               choice_id=key, successor_of=successor_of, **basis)
            if decision:
                scope["decisions"][key] = decision
                accepted.add(key)
        local = wanted - offers.keys()
        if local and accepted == wanted.intersection(offers):
            # Bind each id to the holder that actually offered it in this refusal.
            for member in {row.get("task_id") for row in held}:
                ids = local.intersection(acceptable_unknown([r for r in held if r.get("task_id") == member]))
                if ids:
                    from ouroboros.store import acquisition_budget
                    from supervisor import queue
                    from supervisor.continuation_admission import accept_local_choice
                    with acquisition_budget(0):
                        accepted.update(accept_local_choice(queue, root, tree, {member}, ids, scope["choice"]))
        held = [r for r in held if not accepted.intersection(acceptable_unknown([r]))]
    if held:
        return {"ok": False, "error": error, "blockers": held, **({"action": action} if action else {})}
    return None


def _record(root, start, selected, author, reason, *, choice_id=None, fence_id="", pause_id="", wait_id="",
            successor_of="", hold_id=""):
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from ouroboros.delegate_invocation import _read

    iid, member = start["invocation_id"], start["task_id"]
    try:
        with custody.actor_decision_lock(root, member, wait=False), store.acquisition_budget(0):
            with store.open_store(root, write=False) as txn:
                record = _read(txn, iid) or {}
            if record.get("state") != "pending" or address(record.get("request") or {}) != start:
                return None
            key = choice_id or start["choice_id"]
            prior = (record.get("resume_decisions") or {}).get(selected) or {}
            if prior.get("choice_id") == key:
                return prior
            decision = {**start, "choice_id": key, "fence_id": fence_id, "pause_id": pause_id,
                        **({"wait_id": wait_id} if wait_id else {}),
                        **({"successor_of": successor_of} if successor_of else {}),
                        **({"hold_id": hold_id} if hold_id else {}),
                        "resuming_task_id": selected, "by": author, "reason": reason, "outcome": "unknown",
                        "external_effects": "unknown", "spend": "unknown", "creation": "lookup_only"}
            if custody.emit(root, custody.START_RISK_ACCEPTED, {**start, "resuming_task_id": selected, "decision": decision}):
                return decision
    except (custody.ActorDecisionLockUnavailable, store.StoreError):
        return None
    return None


class StartHeld(OwnerPauseRefused):
    """This handoff lost custody. Its earlier request and any remote work stay intact."""


@contextmanager
def _admit_start(root, task_id, iid):
    """Hold the actor's decision lock across the handoff, after re-reading this request.

    It is taken before the root's launch admission, the order a START_REQUESTED claim
    already uses, so a sibling's claim delays this start and never refuses it. Contention
    past the lock's bound or a busy store is ``custody_busy``; any other failure is
    ``custody_unreadable``. Both leave the request pending and send nothing.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from ouroboros.delegate_invocation import _read

    with ExitStack() as admission:
        try:
            admission.enter_context(custody.actor_decision_lock(root, task_id))
            with store.open_store(root, write=False) as txn:
                record = _read(txn, iid) or {}
        except (custody.ActorDecisionLockUnavailable, store.StoreBusy) as exc:
            busy = getattr(exc, "reason", "") == "contention" or isinstance(exc, store.StoreBusy)
            raise StartHeld(BUSY if busy else UNREADABLE) from exc
        except Exception as exc:
            raise StartHeld(UNREADABLE) from exc
        if record.get("resume_decisions"):
            raise StartHeld(LOOKUP_ONLY)
        if record.get("state") != "pending" or (record.get("request") or {}).get("task_id") != task_id:
            raise StartHeld("delegated_start_custody_changed")
        yield


def checked_start(root, task_id, iid, submit):
    from ouroboros.owner_pause import CheckedSubmission

    return CheckedSubmission(submit, partial(_admit_start, root, task_id, iid))


def close_unsent_start(root, member, iid, reason):
    """A fresh sender stopped before handoff must close its claim before retiring its resources."""
    from ouroboros import delegate_custody as custody

    try:
        with _admit_start(root, member, iid):
            return custody.emit(root, custody.START_UNSENT, {"invocation_id": iid, "task_id": member,
                "reason": reason, "definite": True})
    except StartHeld:
        return False


def risk_words(decision):
    by = decision["by"]
    who = "The owner" if by["actor"] == "owner" else f"Ouroboros (task {by['task_id']})"
    effect = ("The earlier request was read-only, without permission to change files; its work may still be running "
              "or spending money." if decision.get("successor_of") else
              "The work may still be running, changing files or spending money.")
    return (f"{who} accepted the unknown outcome of start {decision['invocation_id']} from task "
            f"{decision['task_id']} in tree {decision['root_task_id']} for Resume of {decision['resuming_task_id']}. "
            f"{effect} Retries can only look up this start. Reason: {decision['reason']}")


def _update_task(txn, task_id, change):
    from ouroboros.obligations import pending_drive_rule
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema

    current = txn.load(task_id)
    if not current:
        raise ValueError("saved task unavailable")
    require_writable_task_result_schema(current)
    fields = change(current)
    if fields:
        updated = stamp_task_result_schema({**current, **fields})
        txn.store_task(updated, current=current, pending_drive=pending_drive_rule(txn, current, updated))


def owe_risk_notice(txn, event):
    """The choice and its addressed local timeline notice commit together. A successor's choice over its
    predecessor's start makes that start its own custody in the same write (``CARRIED``, read by its census)."""
    decision = event["decision"]

    def notice(current):
        identity = f"pause:{event['resuming_task_id']}:{decision['choice_id']}"
        owed = dict(current.get("pause_notices") or {})
        owed[identity] = {"reason": "delegated_start_risk", "confirmed_at": event["ts"], "decision": decision}
        if not decision.get("successor_of"):
            return {"pause_notices": owed}
        carried = dict(current.get(CARRIED) or {})
        carried[decision["invocation_id"]] = {key: decision[key] for key in ("root_task_id", "task_id", "invocation_id")}
        return {"pause_notices": owed, CARRIED: carried}

    _update_task(txn, event["resuming_task_id"], notice)


def verified_successor(prior, row, predecessor, successor):
    """``row`` is the Continue ``predecessor``'s own hash-bound claim names, admitted under that same binding."""
    from ouroboros.owner_continue import binding_sha, continued_by_binding

    claim = continued_by_binding(prior or {})
    admission = (row or {}).get("continuation_admission")
    return bool(claim and claim.get("predecessor_task_id") == predecessor and claim.get("successor_task_id") == successor
                and isinstance(admission, dict) and admission.get("binding") == claim
                and admission.get("binding_sha256") == binding_sha(claim))


def _chain(load, tree, selected):
    """Verified admitted episodes up to ``selected``, or a distinct observation gap."""
    from ouroboros.owner_continue import WORK_CHAIN_MAX_HOPS, continued_by_binding

    prior = load(tree)
    if not prior:
        return [], {}, "episode_record_unavailable"
    roots, rows = [tree], {tree: prior}
    while roots[-1] != selected:
        predecessor = roots[-1]
        nxt = continued_by_binding(rows[predecessor]).get("successor_task_id")
        if not nxt:
            return [], {}, "episode_record_unavailable"
        row = rows.get(nxt) or load(nxt)
        if not verified_successor(rows[predecessor], row, predecessor, nxt):
            return [], {}, "episode_record_unavailable"
        if nxt in rows:
            return [], {}, "episode_chain_cycle"
        if len(roots) >= WORK_CHAIN_MAX_HOPS:
            return [], {}, "episode_chain_depth_exceeded"
        roots.append(nxt)
        rows[nxt] = row
    return roots, rows, ""


def _owner_ended(row):
    """``owner_pause.owner_stopped`` by its class (``owner_continue``): my human's Stop or Panic ends a chain's root for
    every later choice, while the technical interruption a Continue resumes (Restart, restore, shutdown) does not,
    though its cancel intent recorded an id just the same."""
    from ouroboros.owner_continue import OWNER_STOP_SOURCES, TECHNICAL_CANCEL_SOURCES
    from ouroboros.owner_pause import owner_stopped

    origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
    source = str(origin.get("source") or "")
    return source in OWNER_STOP_SOURCES or owner_stopped(row) and not (
        source in TECHNICAL_CANCEL_SOURCES or origin.get("reason") == "server_shutdown")


def _read_only_link(record, body, granted):
    """The exact request and its custody both say ask/readonly with no writable resource."""
    scope = body.get("scope") if isinstance(body, dict) else None
    return bool(
        isinstance(body, dict)
        and (body.get("access"), body.get("mode")) == ("readonly", "ask")
        and set(body) <= _READ_ONLY_BODY_KEYS
        and body.get("execution", {}) in _READ_ONLY_EXECUTION
        and isinstance(scope, dict) and scope.get("kind") == "project"
        and set(scope) <= {"kind", "root", "ephemeral"}
        and (body.get("continueFrom") or "") == (record.get("continuation_of") or "")
        and granted == ("readonly", "ask")
        and (record.get("effective_access", "readonly"), record.get("execution_mode", "ask")) == granted
        and address(record)
        and (record.get("authority_source") or "") in ("", "readonly")
        and not any(record.get(key) for key in _WRITE_BINDINGS)
    )


def _read_only_chain(root, tree, invocation_id):
    """Prove no writes even if every earlier execution group was inherited.

    Claudexor 6.3.0 (0ff3df71, run-continuation-start.ts) replaces the whole execution group on
    key presence. START_REQUESTED does not prove the exact serving build across an ambiguous
    send/replay. Neither the current pin nor an owned-engine address is that proof: an explicit
    group therefore does not end the walk over an earlier writer or an unreadable source.
    """
    from ouroboros import store
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.delegate_invocation import _read, bound_runs
    from ouroboros.delegate_pending import request_body
    from ouroboros.delegate_registration_policy import review_owned_source
    from ouroboros.execution_domain import engine_key
    from ouroboros.owner_continue import recorded_continuation

    evidence, seen = [], set()
    with store.open_store(root, write=False) as txn:
        if bound := bound_runs(txn, invocation_id):
            return False
        indexed = _read(txn, invocation_id, bound=bound) or {}
        request = indexed.get("request") or {}
        store_id = request.get("daemon_store_id") or request.get("handshake_store_id")
        engine = engine_key(request)
        if indexed.get("state") != "pending" or indexed.get("run_id") or not isinstance(store_id, str) or not store_id:
            return False
        if ((request.get("root_task_id") or request.get("task_id")) != tree
                or (engine and (not engine[0] or engine[1] != store_id))):
            return False
        granted = (request.get("effective_access"), request.get("execution_mode"))
        for _ in range(_RUN_CHAIN_MAX_LINKS):
            if (request.get("invocation_id") != invocation_id or invocation_id in seen
                    or engine_key(request) != engine or (request.get("engine_ref") and not engine)
                    or not (request.get("daemon_store_id") or request.get("handshake_store_id"))
                    or any(request.get(key) not in (None, "", store_id)
                           for key in ("daemon_store_id", "handshake_store_id"))):
                return False
            seen.add(invocation_id)
            evidence.append((request, granted))
            head = request.get("continuation_of")
            if not head:
                break
            prior = txn.custody_row(head) or {}
            prior_id = prior.get("task_id")
            invocation_id = prior.get("invocation_id")
            if (not prior_id or not invocation_id or prior.get("daemon_store_id") != store_id
                    or engine_key(prior) != engine or (prior.get("engine_ref") and not engine)
                    or prior.get("terminal_state") not in TERMINAL_STATES
                    or review_owned_source(prior.get("source"))
                    or any(prior.get(key) for key in _WRITE_BINDINGS)):
                return False
            member = request.get("task_id")
            prior_tree = prior.get("root_task_id") or prior_id
            # The original launch's owner-Continue binder, not a same-room/tree guess or a new retry policy.
            if member != prior_id and not recorded_continuation(root, prior_tree, member, load=txn.load):
                return False
            bound = bound_runs(txn, invocation_id)
            if set(bound) != {head}:
                return False
            indexed = _read(txn, invocation_id, bound=bound) or {}
            request = indexed.get("request") or {}
            if (indexed.get("state") != "started" or indexed.get("run_id") != head
                    or request.get("task_id") != prior_id
                    or (request.get("root_task_id") or prior_id) != prior_tree
                    or (request.get("continuation_of") or "") != (prior.get("continuation_of") or "")):
                return False
            granted = (prior.get("access"), prior.get("mode"))
        else:
            return False
    # CAS reads are not made under the store transaction or the supervisor queue lock.
    return all(_read_only_link(row, request_body(root, row), grant) for row, grant in evidence)


def predecessor_starts(root, predecessor, successor, blockers):
    """``{invocation_id: its tree}`` that ``successor`` (a never-started Continue) may name when its predecessor census
    ``blockers`` is solely unbound ordinary starts, each proven no-write (``_read_only_chain``) and reached from its tree
    by the verified chain (``_chain``) through ``predecessor``: the predecessor's own, or one the predecessor carried by
    its own recorded choice (``successor_of``), offered afresh and never inherited. Otherwise ``{}``: the refusal stays."""
    from ouroboros.task_results import load_task_result

    starts = {}
    try:
        for row in blockers or ():
            start = row.get("ordinary_start") if isinstance(row, dict) else None
            if (not isinstance(start, dict) or row.get("kind") != "delegated_run" or row.get("run_id")
                    or row.get("stop_outcome") != "pending_invocation_unbound"
                    or start.get("task_id") != row.get("task_id") or start.get("invocation_id") != row.get("invocation_id")):
                return {}
            tree, accepted = str(start.get("root_task_id") or ""), (row.get("resume_decisions") or {}).get(predecessor) or {}
            if tree != predecessor and (accepted.get("successor_of"), accepted.get("creation")) != (tree, "lookup_only"):
                return {}
            starts[start["invocation_id"]] = tree
        load = partial(load_task_result, root, strict=True)
        for tree in set(starts.values()):
            roots, rows, gap = _chain(load, tree, successor)
            if gap:
                log.debug("Read-only predecessor choice for %s: %s", successor, gap)
                return {}
            if roots[-2:-1] != [predecessor] or any(_owner_ended(rows[tid]) for tid in roots[:-1]):
                return {}
        return starts if all(_read_only_chain(root, tree, iid) for iid, tree in starts.items()) else {}
    except Exception:
        log.warning("Start evidence of %s's predecessor is unreadable; nothing is offered", successor, exc_info=True)
        return {}


def carried_start_writers(root, carried, *, unbound):
    """An accepted start's bound run until settled; only its own Continue sees the unbound start too."""
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.delegate_invocation import decision_facts, snapshot
    from ouroboros.delegate_registration_policy import review_owned_source

    if not carried:
        return []
    try:
        current = snapshot(root, sorted(carried))
    except Exception:
        current = {}  # each saved address below still has unknown custody
    rows = []
    for iid, start in carried.items():
        member = (start.get("task_id") or "") if isinstance(start, dict) else ""
        indexed, bound = current.get(iid, (None, {}))
        indexed = indexed or {}
        request = indexed.get("request") or {}
        fate, run_id = indexed.get("state"), indexed.get("run_id") or ""
        run = bound.get(run_id) or {}
        own = address(request)
        if (not member or not own or own.get("invocation_id") != iid or start.get("invocation_id") != iid
                or own.get("task_id") != member or own.get("root_task_id") != start.get("root_task_id")
                or fate not in ("pending", "failed_definite", "started")
                or bool(run_id) != (fate == "started")
                or set(bound) != ({run_id} if fate == "started" else set())
                or fate == "started" and (run.get("task_id") != member or run.get("invocation_id") != iid
                                          or (run.get("root_task_id") or member) != start.get("root_task_id")
                                          or run.get("settled") not in (None, False, True))):
            rows.append({"kind": "custody_unreadable", "task_id": member, "invocation_id": iid})
        elif fate == "pending" and unbound:
            rows.append({"kind": "delegated_run", "task_id": member, "run_id": "", "invocation_id": iid,
                         "ordinary_start": own, "review_owned": False, "source": request.get("source") or "",
                         "state": "stop_unknown", "stop_outcome": "pending_invocation_unbound", **decision_facts(indexed)})
        elif fate == "started" and not run.get("settled"):
            first = run.get("resume_start_state")
            observed = (run.get("terminal_state") or first) if first is not None else ""
            if observed in TERMINAL_STATES:
                state, outcome = "terminal", "awaiting_collection"
            elif observed == "unknown":
                state, outcome = "stop_unknown", "accepted_activity_unknown"
            else:
                state, outcome = "running", ""
            rows.append({"kind": "delegated_run", "task_id": member, "run_id": run_id, "invocation_id": iid,
                         "review_owned": review_owned_source(run.get("source")), "source": run.get("source") or "",
                         "state": state, "stop_outcome": outcome})
    return rows


class AnswerHeld(Exception):
    """A late writer won before any answer was owed, sent or stored."""

    def park(self, ctx):
        from ouroboros.budget_pause import RAIL_CONTROL_AUTHORITY, enter_owner_pause, request_pause
        from ouroboros.model_wait import ModelWaitInterrupted
        from ouroboros.loop_round_limits import _handle_model_wait_control

        limit, candidate, control_required = ctx._selected_answer_context
        ctx._delivery_candidate, ctx._delivery_control_required = candidate, control_required
        try:
            if self.args and self.args[0] != "owner_pause":
                raise ModelWaitInterrupted(self.args[0])
            # The publication boundary parks COLD even where a warm park is available: a resumed
            # stack could only re-emit the retained text, while the owner's words accepted under
            # this pause must reach the model. The ordinary Resume re-enters the loop (resume_point
            # phase "candidate"), where the owner-input check holds the saved candidate for a
            # revision round when new words arrived, and delivers it unchanged when none did.
            enter_owner_pause(limit, delivery_candidate=True, allow_warm=False)
        except ModelWaitInterrupted as control:
            if control.control_reason == "owner_pause_authority_unreadable":
                request_pause(limit, rail=RAIL_CONTROL_AUTHORITY, scope="task",
                              reason_text=control.control_reason, delivery_candidate=True)
                raise
            return _handle_model_wait_control(limit, control)
        finally:
            ctx._delivery_candidate, ctx._delivery_control_required = None, False


def finish_selected_answer(ctx, emit, result, events, worker_stopped):
    if not getattr(ctx, "_selected_answer_context", None):
        emit(*result)
        return list(events)
    from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait

    try:
        while True:
            try:
                emit(*result)
                return list(events)
            except AnswerHeld as held:
                forced = held.park(ctx)  # a cold hold unwinds to the worker's usual pause event
                if forced:
                    result, ctx._selected_answer_context = forced, None
    except ModelWaitInterrupted as control:
        if worker_stopped(control, current_model_wait()):
            return []
        raise
    finally:
        ctx._selected_answer_context = None


def selected_answer_context(limit):
    """Keep the prepared answer until its first handoff. A sibling's first choice can still arrive.

    Returns ``(context, waited)``. ``waited`` says this stack stood under an owner Pause before the
    fence opened (``enter_owner_pause``): its caller then re-enters its owner-input check, so my
    human's words accepted meanwhile hold the answer for a round instead of leaving with it.
    """
    from ouroboros.budget_pause import enter_owner_pause
    from ouroboros.owner_pause import _member_coordinates

    waited = enter_owner_pause(limit, delivery_candidate=True)
    ctx = getattr(getattr(limit, "tools", None), "_ctx", None)
    if all(_member_coordinates(ctx)):
        return (limit, getattr(ctx, "_delivery_candidate", None), getattr(ctx, "_delivery_control_required", False)), waited
    return None, waited


def answer_publication(ctx):
    """Order the first actual publication with the same tree lock as the active-start observation."""
    if not getattr(ctx, "_selected_answer_context", None):
        return None
    from ouroboros._usage_wait import retry_presend
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.owner_pause import LaunchBusy, _member_coordinates, launch_lock, member_fence
    from ouroboros.store import StoreBusy, acquisition_budget

    root, tree, _ = _member_coordinates(ctx)
    committed = False

    def publish(write, *args, wait=False, **kwargs):
        if committed:
            return write(*args, **kwargs)

        def commit():
            nonlocal committed
            with launch_lock(root, tree, wait=False), acquisition_budget(0):
                if member_fence(ctx):
                    raise AnswerHeld()
                value = write(*args, **kwargs)  # only the ready outbox rename or the local result transaction
                committed = True
            ctx._selected_answer_context = None
            return value

        if not wait:
            return commit()  # the outbox writer already holds its own lock; a refusal uses the result path
        try:
            return retry_presend(commit, check=False)  # waits after all locks unwind, only on proven contention
        except PhysicalDispatchInterrupted as control:
            raise AnswerHeld(control.control_reason) from control
        except (LaunchBusy, StoreBusy) as exc:
            if getattr(exc, "rollback_confirmed", None) is False:
                raise
            raise AnswerHeld("owner_pause_authority_unreadable") from exc
    return publish


def still_pending(txn, event):
    """A late definite failure or binding cannot acquire a decision in the write transaction. A successor's decision
    (``successor_of``) binds that Continue and its own stop (its fence, or none still closed): its chain from the start's
    tree verified in this write (``_chain``), no earlier root of it ended by my human (``_owner_ended``; a Restart does
    not), and a start reached past the tree's own successor still carried by its predecessor's own recorded choice."""
    from ouroboros.delegate_invocation import _read
    from ouroboros.owner_pause import fence_closed, owner_stopped

    decision = event["decision"]
    tree, selected = event["root_task_id"], event["resuming_task_id"]
    successor_of = decision.get("successor_of")
    record = _read(txn, event["invocation_id"]) or {}
    if successor_of:
        if successor_of != tree or tree == selected:
            return False
        roots, rows, gap = _chain(txn.load, tree, selected)
        if gap:
            return False
        member = holder = rows.get(selected) or {}
        carrier = (((record.get("resume_decisions") or {}).get(roots[-2]) or {}) if len(roots) > 2
                   else {"successor_of": tree, "creation": "lookup_only"})
        if (not roots or str(member.get("root_task_id") or selected) != selected
                or any(_owner_ended(rows[root_id]) for root_id in roots[:-1])
                or (carrier.get("successor_of"), carrier.get("creation")) != (tree, "lookup_only")):
            return False
    else:
        tree_row = txn.load(tree) or {}
        member, holder = (tree_row if selected == tree else (txn.load(selected) or {})), tree_row
        if not member or (member.get("root_task_id") and member["root_task_id"] != tree) or owner_stopped(tree_row):
            return False
    if owner_stopped(member) or member.get("_owner_hold"):
        return False
    fence = holder.get("owner_pause") or {}
    if (str(fence.get("fence_id") or "") if fence_closed(fence) or not successor_of else "") != decision["fence_id"]:
        return False
    pause = member.get("budget_pause") or {}
    if decision["pause_id"] and (pause.get("pause_id") != decision["pause_id"]
                                 or pause.get("state") not in {"pausing", "paused"}):
        return False
    if decision.get("wait_id"):
        wait = member.get("owner_wait") or {}
        if (wait.get("wait_id") != decision["wait_id"] or wait.get("reason") != "owner_pause"
                or wait.get("state") != "waiting"):
            return False
    return (record.get("state") == "pending" and address(record.get("request") or {}) == address(event)
            and (record.get("request") or {}).get("task_id") == event["task_id"])


def resume_holders(record, tree):
    """The saved successors whose own Resume accepted this start of ``tree`` (its ``successor_of`` decisions)."""
    return sorted({str(decision.get("resuming_task_id") or "") for decision in (record.get("resume_decisions") or {}).values()
                   if isinstance(decision, dict) and decision.get("successor_of") == tree} - {"", str(tree)})


@contextmanager
def publication_lock(root, event):
    """An accepted writer's first proof of activity is ordered against every handoff of its tree, then of each saved
    successor that accepted it (sorted), all taken before the write opens. Yields those successors, or ``None`` while
    they cannot be read: unknown, never none, so the write refuses while any exists (``hold_active_start``) and stays
    owed, as when a successor's lock is busy: it is never waited for (``prepare_recovered_cancel``'s Stop or Panic
    read-back publishes here). The set is fixed: a decision needs its invocation pending under the starter's decision
    lock, which a binding STARTED holds, and every later activity row follows that STARTED."""
    from ouroboros.owner_pause import launch_lock

    if event.get("resume_start_state") not in ACTIVE:
        yield ()
        return
    from ouroboros import store
    from ouroboros.delegate_invocation import INVOCATIONS

    tree = event.get("root_task_id") or event["task_id"]
    with ExitStack() as held:
        held.enter_context(launch_lock(root, tree))
        try:
            with store.acquisition_budget(0), store.open_store(root, write=False) as txn:
                holders = resume_holders(txn.get_row(INVOCATIONS, event["invocation_id"]) or {}, tree)
        except store.StoreError:
            holders = None
        for holder in holders or ():
            held.enter_context(launch_lock(root, holder, wait=False))
        held.enter_context(store.acquisition_budget(0))
        yield holders


def _tree_works(txn, tree, row):
    """The ordinary Pause late-work rules as this custody commit observes them."""
    from ouroboros.external_runs import tree_owned_custody
    from ouroboros.owner_pause import answered_phase_work
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    if any(answered_phase_work(txn.root, row.get("task_id") or tree, row, txn=txn)):
        return True
    children = txn.conn.execute(
        "SELECT 1 FROM tasks WHERE root_task_id=? AND task_id<>? AND status NOT IN (SELECT value FROM json_each(?))",
        (tree, tree, json.dumps(sorted(_TRULY_TERMINAL_STATUSES))),
    ).fetchone()
    return bool(children or tree_owned_custody(txn.root, tree, transaction=txn))


def _hold_fence(txn, root_id, request_id, activity, *, ended_holds=True):
    """Close ``root_id``'s fence once per activity request: a new launch generation over recovered work. A Stop owns
    what remains; ``ended_holds=False`` (a successor) leaves an actor that ended unfenced only once nothing of its tree
    runs (``_tree_works``): its own Continue is then held by the census instead (``carried_start_writers``), while an
    answered one whose members or delegated work still run is held, its children known to the settlement as any Pause's."""
    from ouroboros.owner_pause import _resume_outstanding, fence_closed, next_fence, owner_stopped
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    def hold(current):
        if owner_stopped(current):
            return None
        actor = current
        if current.get("superseded_by") or current.get("retry_task_id"):
            from ouroboros.cancel_intents import _validated_single_cancel_target

            actor = txn.load(_validated_single_cancel_target(txn.root, root_id, txn=txn)) or {}
            if owner_stopped(actor):
                return None
        if not ended_holds and actor.get("status") in _TRULY_TERMINAL_STATUSES and not _tree_works(txn, root_id, actor):
            return None
        old = current.get("owner_pause") or {}
        if request_id in (old.get("requests") or {}):
            return None
        fence = next_fence(root_id, old, request_id, {"active_delegated_start": activity})
        if not fence_closed(old) or old.get("observe_only") or _resume_outstanding(actor, old):
            fence["observe_only"] = True
        if fence_closed(old):
            fence.update({key: old[key] for key in ("by", "ended_member_runs", "membership_gaps") if key in old})
        return {"owner_pause": fence}

    _update_task(txn, root_id, hold)


def hold_active_start(txn, event, holders=()):
    """Called in the live custody commit only. Imports rebuild observations and never issue controls. Holds the
    start's tree when one of its own decisions accepted it (never for a successor's choice alone), and each saved
    successor that accepted it (``holders``: their launch locks ``publication_lock`` took); a successor's fence names
    its own holder beside the start's own address. Returns the roots whose fences it may have closed."""
    from ouroboros.delegate_invocation import INVOCATIONS

    record = txn.get_row(INVOCATIONS, event["invocation_id"]) or {}
    start = address(record.get("request") or {})
    if not record.get("resume_decisions") or not start:
        return []
    tree = start["root_task_id"]
    if tree != (event.get("root_task_id") or event["task_id"]) or start["task_id"] != event["task_id"]:
        raise ValueError("recovered start belongs to a different task tree")
    run = txn.custody_row(event["run_id"]) or {}
    if run.get("task_id") != start["task_id"] or run.get("invocation_id") != start["invocation_id"]:
        raise ValueError("observed run does not hold the accepted invocation")
    if run.get("settled") or run.get("terminal_state") or run.get("resume_start_state") not in ACTIVE:
        return []
    successors = resume_holders(record, tree)
    if not set(successors) <= set(holders or ()):
        if holders is None:  # unread under the tree's lock: unknown, never none; nothing lands and the write stays owed
            from ouroboros.store import StoreBusy
            raise StoreBusy("the successors holding this start were unread under its tree's lock", reason="unknown")
        raise ValueError("a successor that accepted this start is not under its launch lock")
    activity = {**start, "run_id": event["run_id"], "state": run["resume_start_state"]}
    own = [decision for selected, decision in record["resume_decisions"].items() if selected not in successors]
    if own:
        _hold_fence(txn, tree, "active:" + start["choice_id"], activity)
    for holder in successors:
        _hold_fence(txn, holder, "active:" + record["resume_decisions"][holder]["choice_id"],
                    {**activity, "holder_task_id": holder}, ended_holds=False)
    return [*([tree] if own else []), *successors]


def settled_holders(txn, entry):
    """The roots a recovered start's settlement wakes: its tree and each saved successor that accepted it."""
    from ouroboros.delegate_invocation import INVOCATIONS

    tree = entry.root_task_id or entry.task_id
    return [tree, *resume_holders(txn.get_row(INVOCATIONS, entry.invocation_id) or {}, tree)]


def recovered_writer_changed(root, tree):
    """Wake the addressed tree's settlement pass if this process owns its queue."""
    from supervisor import queue
    from supervisor.owner_pause_control import mark_tree_changed

    if not queue.owns_live_queue(root):
        return False
    mark_tree_changed(tree, recovered_drive=queue.DRIVE_ROOT)
    return True


def recovered_writers_changed(root, entries):
    """``recovered_writer_changed`` for recovered starts' trees and each saved successor holding one, the successors read
    once by exact invocation. A failed read still wakes the trees and orders the deferred fence discovery
    (``owner_pause_control.defer_failed_recovery_discovery``), which finds a successor's fence by its holder."""
    from ouroboros import store

    trees = dict.fromkeys(entry.root_task_id or entry.task_id for entry in entries)
    try:
        with store.open_store(root, write=False) as txn:
            for entry in entries:
                trees.update(dict.fromkeys(settled_holders(txn, entry)))
    except Exception:
        log.warning("Successors of recovered starts %s unreadable; their fences are discovered again",
                    [entry.invocation_id for entry in entries], exc_info=True)
        from supervisor import queue
        if queue.owns_live_queue(root):
            from supervisor.owner_pause_control import defer_failed_recovery_discovery
            defer_failed_recovery_discovery(queue.DRIVE_ROOT)
    for tree in trees:
        recovered_writer_changed(root, tree)


def restore_recovered_writer_hints(root):
    """Rebuild the supervisor's settlement hints from their authority, not a retired run's receipt."""
    from ouroboros.store_hot_facts import recovered_owner_pause_roots

    for tree in recovered_owner_pause_roots(root):
        recovered_writer_changed(root, tree)


def retain_start_receipt(txn, entry, task_id, task_row):
    """A worker keeps its closed fence address until the live supervisor can look; boot reads the fence itself."""
    from ouroboros.owner_pause import fence_closed

    if entry.resume_start_state not in ACTIVE:
        return False
    trees = settled_holders(txn, entry)
    if recovered_writer_changed(txn.root, trees[0]):
        for successor in trees[1:]:
            recovered_writer_changed(txn.root, successor)
        return False
    for tree in trees:
        fence = ((task_row if tree == task_id else txn.load(tree)) or {}).get("owner_pause") or {}
        if fence_closed(fence) and fence.get("active_delegated_start"):
            return True
    return False


def observe_bound_state(root, entry, state):
    """Use the first actual status after binding. False means the write failed and remains owed."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run
    from ouroboros.store import StoreError

    if (getattr(entry, "resume_start_state", None) == "unknown" and not entry.terminal_state
            and state in ACTIVE | custody.TERMINAL_STATES):
        if not custody.emit(root, custody.START_STATE_OBSERVED, {
            "root_task_id": entry.root_task_id or entry.task_id, "task_id": entry.task_id,
            "invocation_id": entry.invocation_id, "run_id": entry.run_id, "resume_start_state": state}):
            return False
        # An observer in another process may have published first. Adopt the committed state.
        try:
            observed = run(root, entry.run_id)
        except StoreError:
            return False  # unreadable observation does not retract the commit
        if observed is None or observed.invocation_id != entry.invocation_id or observed.task_id != entry.task_id:
            return False
        entry.resume_start_state = observed.resume_start_state
        memo = custody._CUSTODY.get(entry.run_id)
        if memo and memo.invocation_id == entry.invocation_id and memo.task_id == entry.task_id:
            memo.resume_start_state = observed.resume_start_state
        return True
    return None


def _recovered_peer_binding(gateway, entry, ref, *, for_stop=False):
    """Verify a recovered peer letter against its genuine PeerGateway, exact link binding and run id."""
    from ouroboros import peer_channel, peer_links
    from ouroboros.execution_domain import same_engine
    from ouroboros.gateways.peer import PeerGateway

    place_id, link_id = ref.get("place_id"), ref.get("link_id")
    run_id, invocation_id = getattr(entry, "run_id", ""), getattr(entry, "invocation_id", "")
    if (not isinstance(gateway, PeerGateway) or not same_engine(gateway, entry)
            or not isinstance(place_id, str) or not place_id.strip()
            or not isinstance(link_id, str) or not link_id.strip()
            or not isinstance(run_id, str) or not run_id.strip()
            or (invocation_id and invocation_id != run_id)
            or getattr(entry, "daemon_store_id", "") or getattr(gateway, "store_identity", "")):
        return False
    link = getattr(gateway, "_link", None)
    if (not isinstance(link, peer_links.OutgoingLink)
            or link.place_id != place_id or link.link_id != link_id):
        return False
    now = peer_links.outgoing_link(link.place_id, drive_root=link.drive_root)
    if now is None and for_stop:
        now = peer_links.stopping_link(link.place_id, drive_root=link.drive_root)
    return (now is not None
            and peer_channel.link_identity(now) == peer_channel.link_identity(link)
            and peer_channel.link_target_valid(now)[0])


def stop_read_run(gateway, run_id):
    """Operational Stop read-back: on a linked Ouroboros, uses the Pause-safe Stop read channel."""
    from ouroboros.gateways.peer import PeerGateway

    if isinstance(gateway, PeerGateway):
        return gateway.get_run(run_id, for_stop=True)
    return gateway.get_run(run_id)


def recovered_cancel_store(gateway, entry, *, for_stop=False):
    from ouroboros.execution_domain import is_peer_ref, recorded_ref, same_engine

    ref = recorded_ref(entry)
    if is_peer_ref(ref):
        return _recovered_peer_binding(gateway, entry, ref, for_stop=for_stop)
    return (same_engine(gateway, entry) and bool(entry.daemon_store_id)
            and entry.daemon_store_id == getattr(gateway, "store_identity", ""))


def recovered_cancel_detail(gateway, entry, detail, *, for_stop=False):
    """A recovered run's terminal and active facts must name its run in its store."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_usage import run_names
    from ouroboros.execution_domain import is_peer_ref, recorded_ref

    if not isinstance(detail, dict):
        return False
    if is_peer_ref(recorded_ref(entry)):
        if not recovered_cancel_store(gateway, entry, for_stop=for_stop):
            return False
        if detail.get("storeId") not in (None, ""):
            return False
        if not for_stop and custody.summary_of(detail).get("contentWithheld"):
            return False
        return entry.run_id in run_names(detail) and not custody.names_another_run(detail, entry.run_id)
    return (recovered_cancel_store(gateway, entry) and entry.run_id in run_names(detail)
            and detail.get("storeId") in (None, "", entry.daemon_store_id))


def fold_start_control(entry, row):
    """Reduce one row of a recovered start onto its run (``delegate_custody._apply``, replay and memo alike).

    A lookup-bound start's first engine state replaces ``unknown`` once, for its own task and
    invocation, never after a terminal. An ordinary run's cancel outcome is not folded. A handoff
    without its outcome keeps its identity and its sender (``sender``: the image that reserved it)
    as an unresolved fault; the outcome that follows drops the sender, who has then returned.
    """
    from ouroboros import delegate_custody as c

    if row["type"] == c.START_STATE_OBSERVED:
        if (entry.resume_start_state == "unknown" and not entry.terminal_state
                and entry.invocation_id == row.get("invocation_id") and entry.task_id == row.get("task_id")):
            entry.resume_start_state = row["resume_start_state"]
        return
    action = getattr(entry, "_cancel_action", {})
    if row["type"] == c.CANCEL_HANDOFF:
        entry._recovered_cancel = c.CANCEL_CONTAINMENT_FAULT
        entry._cancel_action = {"control_id": row.get("stop_control_id") or "", "unknown_at": row.get("ts"),
                                "unknown_event": row.get("event_id"), "sender": row.get("sender") or {}}
        return
    if entry.resume_start_state is None:
        return
    outcome = row.get("outcome")
    entry._recovered_cancel = ("not_issued" if row.get("control_owed") else c.CANCEL_REQUESTED
                               if row.get("accepted") and outcome != c.CANCEL_CONFIRMED else outcome or "unknown")
    entry._cancel_action = {"control_id": action.get("control_id", "")}
    if entry._recovered_cancel == c.CANCEL_CONTAINMENT_FAULT:
        entry._cancel_action.update(unknown_at=row.get("ts"), unknown_event=row.get("event_id"))


def _prior_sender_returned(action):
    """Whether the earlier handoff's sender can no longer deliver it: it recorded its outcome (no
    ``sender`` left), or its image is positively gone (``local_custody_repair.owner_ended``: an absent
    pid, a different kernel birth, or this pid under an earlier custody session). A sender without a
    recorded identity, or one that may still run, keeps every later press read-only."""
    sender = action.get("sender")
    if sender is None:
        return True
    from ouroboros.local_custody_repair import owner_ended

    return owner_ended(sender.get("pid"), "", sender)


def can_reserve_cancel(entry, event):
    """One sender per handoff. A new one is reserved when no control stands (none issued, or a
    recorded failure), or once after an unresolved fault: by a distinct owner Stop requested after
    that fault and naming it (``after_unknown_event``, which only ``prepare_recovered_cancel`` sets)."""
    from ouroboros import delegate_custody as c

    if entry._recovered_cancel in ("", "not_issued", c.CANCEL_FAILED):
        return True
    old = getattr(entry, "_cancel_action", {})
    control_id = event.get("stop_control_id")
    return (entry._recovered_cancel == c.CANCEL_CONTAINMENT_FAULT and bool(control_id)
            and control_id != old.get("control_id") and bool(old.get("unknown_event"))
            and event.get("after_unknown_event") == old["unknown_event"]
            and bool(old.get("unknown_at")) and (event.get("stop_requested_at") or "") > old["unknown_at"])


def prepare_recovered_cancel(root, gateway, entry, reason, owner_stop):
    """Control follows the exact run/store, not the success of publishing its first state.

    GET can avoid an unnecessary POST or corroborate identity. An unreadable GET alone
    cannot veto an authorized Stop. The existing handoff admits one sender; after a crash
    or ambiguous return, the same duty only reads back unless non-delivery was proved.
    A distinct later owner Stop names that fault, and so may reserve once more, only when
    this read shows the exact run active in its store and the earlier sender has returned
    (``_prior_sender_returned``). The same press and automatic reconciliation never do.
    """
    from ouroboros.execution_domain import recorded_ref
    from ouroboros.tool_custody import owner_proof
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def blocked(reason, explanation, *, owed=False):
        return {"outcome": custody.CANCEL_CONTAINMENT_FAULT, "accepted": False, "control_status": "",
                "state": "", "fault_reason": reason, "detail": explanation, "control_owed": owed}

    fresh = run(root, entry.run_id)
    if (fresh is None and custody._CUSTODY.get(entry.run_id) is entry
            and recovered_cancel_store(gateway, entry, for_stop=True)):
        # The starter kept the exact physical handle when STARTED failed. Rejoin its invocation,
        # not a new start, before reserving Stop; another engine name is not a match here.
        bound = custody.bind_started(root, entry, exclusive=True, accepted_only=True)
        if bound.durable and bound.custody.run_id == entry.run_id:
            fresh = bound.custody
    if not fresh or (fresh.task_id, fresh.invocation_id, fresh.daemon_store_id, recorded_ref(fresh)) != (
            entry.task_id, entry.invocation_id, entry.daemon_store_id, recorded_ref(entry)):
        return blocked("custody_owner_mismatch", "The run's custody identity is unavailable or changed.", owed=True)
    prior = fresh._recovered_cancel
    accepted = prior == custody.CANCEL_REQUESTED
    owed = prior in ("", "not_issued", custody.CANCEL_FAILED)

    def result(outcome, state="", explanation="", terminal=None):
        if outcome == custody.CANCEL_CONFIRMED:
            return custody._cancel_result(root, fresh, outcome, accepted=accepted,
                                          control_status="", state=state, terminal_detail=terminal)
        fault = "cancel_unverified" if outcome == custody.CANCEL_CONTAINMENT_FAULT else ""
        if fault and not fresh.containment_fault:
            custody.record_containment_fault(root, fresh, fault, explanation)
        return {"outcome": outcome, "accepted": accepted, "control_status": "", "state": state,
                "detail": explanation, "fault_reason": fault, "control_owed": owed}

    if fresh.settled:
        return result(custody.CANCEL_CONFIRMED, "settled")
    if not recovered_cancel_store(gateway, fresh, for_stop=True):
        return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The run's owning store was not verified.")
    read_error = ""
    try:
        detail = stop_read_run(gateway, entry.run_id)
    except ClaudexorUnavailable as exc:
        if custody.daemon_says_absent(exc, gateway=gateway, run=fresh):
            custody.close_absent_run(root, gateway, fresh, "get_run_absent", absence=exc)
            return result(custody.CANCEL_CONFIRMED, "absent")
        if exc.status_code == 404:
            return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The 404 did not prove owning-store absence.")
        read_error, detail = str(exc), {}
    if (custody.names_another_run(detail, entry.run_id)
            or custody.admitted_store_id(detail) not in ("", fresh.daemon_store_id)):
        return result(custody.CANCEL_CONTAINMENT_FAULT, explanation="The read-back names another run or store.")
    state = (str(custody.summary_of(detail).get("state") or "")
             if recovered_cancel_detail(gateway, fresh, detail, for_stop=True) else "")
    if state in custody.TERMINAL_STATES:
        custody.settle_run(root, gateway, fresh, detail)
        return result(custody.CANCEL_CONFIRMED, state, terminal=detail)
    if accepted:
        return result(prior, state, "The earlier cancel was accepted. Termination is not verified.")
    if state in ACTIVE:
        observe_bound_state(root, fresh, state)  # owed publication is not a veto over the exact Stop
    action = owner_stop or {}
    event = {"task_id": fresh.task_id, "run_id": fresh.run_id, "invocation_id": fresh.invocation_id,
             "daemon_store_id": fresh.daemon_store_id, "reason": reason,
             "stop_control_id": action.get("control_id", ""), "stop_requested_at": action.get("requested_at", ""),
             "sender": owner_proof()}
    earlier = getattr(fresh, "_cancel_action", {})
    if state in ACTIVE and _prior_sender_returned(earlier):  # never on a penumbral GET or beside a live sender
        event["after_unknown_event"] = earlier.get("unknown_event")
    if (recovered_cancel_store(gateway, fresh, for_stop=True) and can_reserve_cancel(fresh, event)
            and custody.emit(root, custody.CANCEL_HANDOFF, event)):
        if recovered_cancel_store(gateway, fresh, for_stop=True):
            return None
        return custody._cancel_result(root, fresh, custody.CANCEL_CONTAINMENT_FAULT, accepted=False,
                                      control_status="", state=state, fault_reason="cancel_unverified",
                                      detail="Engine identity changed before control delivery.", control_owed=True)
    return result(custody.CANCEL_CONTAINMENT_FAULT, state, read_error or "The cancellation handoff is unconfirmed.")
