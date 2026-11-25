"""Chosen wake sources and delivery cursors of one retained review wait.

Readiness never consumes a fact. A complete transcript insertion stages its
cursor; the working checkpoint makes that acknowledgement durable. A cursor
never discharges a room duty. ``seen_outcome_states`` retains rendered feedback
separately in the same ledger wait: it ends unread duties here and lets the
terminal collector distinguish a received unwaited outcome from mere settlement.

By default a wait wakes on the whole wave plus mail and controls; counted reviewer IDs
are pinned to wave, operation and slot.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import pathlib

log = logging.getLogger(__name__)


def _refresh_reviewer_pins(record, pins):
    rows = {str(row.get("seat_id") or ""): row for row in (record or {}).get("rows") or ()
            if isinstance(row, dict) and not row.get("additional")}
    wave = str((record or {}).get("review_wave_id") or "")
    refreshed = []
    for pin in pins or ():
        if not isinstance(pin, dict):
            continue
        row = rows.get(str(pin.get("slot_id") or ""))
        if (row and (not wave or pin.get("wave_id") == wave) and row.get("operation_id")
                and pin.get("operation_id") in (row.get("superseded_operation_ids") or ())):
            pin = {**pin, "operation_id": row["operation_id"]}
        if pin not in refreshed:
            refreshed.append(dict(pin))
    return refreshed


def _wave_selection(selection, record=None):
    chosen = selection if isinstance(selection, dict) else {}
    pins = _refresh_reviewer_pins(record, chosen.get("reviewers"))
    slots = list(chosen.get("reviewer_slots") or [])
    if not (pins or slots):
        return {}
    record_id = str((record or {}).get("record_id") or chosen.get("record_id") or "")
    return {"senders": [], "tasks": [], "runs": [],
            "any_mail": bool(chosen.get("any_mail", not (pins or slots))),
            "reviewer_slots": slots, "reviewers": pins,
            **({"record_id": record_id} if record_id else {})}


def _wait_owner(record, waiting=None):
    """Who a non-empty wait state belongs to: its explicit ``task_id``, else the addressed record's ``task_id``."""
    wait = (waiting if waiting is not None else (record or {}).get("waiting")) or {}
    if not wait:
        return ""
    ref = wait.get("review_reference") if isinstance(wait.get("review_reference"), dict) else {}
    return str(wait.get("task_id") or ref.get("task_id") or (record or {}).get("task_id") or "")


def _listener_wait(ctx, record):
    waiting = (record or {}).get("waiting") or {}
    if not waiting:
        return {}
    own = str(getattr(ctx, "task_id", "") or "")
    holder = _wait_owner(record, waiting)
    if holder in ("", own):
        state = copy.deepcopy(waiting)
        if isinstance(state.get("selection"), dict) and state["selection"].get("reviewers"):
            state["selection"]["reviewers"] = _refresh_reviewer_pins(record, state["selection"]["reviewers"])
        return state
    return {"task_id": own,
            "attempt_task_id": str((record or {}).get("task_id") or waiting.get("attempt_task_id") or ""),
            "record_id": str((record or {}).get("record_id") or waiting.get("record_id") or ""),
            "selection": _wave_selection(waiting.get("selection"), record),
            "seen_helpers": [], "seen_outcomes": [], "seen_outcome_states": []}


def saved_selection(ctx, record_id):
    from ouroboros.review_ledger import ledger_root, load_record

    record = load_record(ledger_root(ctx), record_id) or {}
    state = _listener_wait(ctx, record)
    return {key: copy.deepcopy(state.get(key) or ({} if key == "selection" else []))
            for key in ("selection", "seen_helpers", "seen_outcomes")}


def select_helpers(ctx, wake_on=None):
    """Retain this listener's saved selection; a new wait selects only explicitly named sources."""
    from ouroboros.model_sleep import selectors, _owned_runs
    from ouroboros.review_ledger import ledger_root, load_record

    own = str(getattr(ctx, "task_id", "") or "")
    prior = getattr(ctx, "_left_review_wait", None)
    record, record_id = {}, ""
    if isinstance(prior, dict) and prior:
        sel = prior.get("selection") if isinstance(prior.get("selection"), dict) else {}
        ref = prior.get("review_reference") if isinstance(prior.get("review_reference"), dict) else {}
        record_id = str(prior.get("record_id") or sel.get("record_id") or ref.get("record_id") or "")
        if record_id and (not prior.get("task_id") or str(prior.get("task_id") or "") != own or sel.get("reviewers")):
            try:
                record = load_record(ledger_root(ctx), record_id)
            except Exception:
                record = None
    holder = _wait_owner(record, prior) if isinstance(prior, dict) and prior else ""
    foreign = bool(isinstance(prior, dict) and prior and ((record is None and record_id and not prior.get("task_id"))
                                                          or holder not in ("", own)))
    if isinstance(prior, dict) and not foreign and (prior or wake_on is None):
        # Legacy waits mean the whole wave, not a newly inferred helper roster.
        state = {"task_id": own, **{key: copy.deepcopy(prior.get(key) or ([] if key != "selection" else {}))
                                    for key in ("selection", "seen_helpers", "seen_outcomes")}}
        if record and (state["selection"] or {}).get("reviewers"):
            state["selection"]["reviewers"] = _refresh_reviewer_pins(record, state["selection"]["reviewers"])
    elif foreign and wake_on is None:
        state = {"task_id": own, "selection": _wave_selection(prior.get("selection"), record or {}),
                 "seen_helpers": [], "seen_outcomes": []}
        ctx._left_review_wait = ({**copy.deepcopy(state), "record_id": record_id}
                                 if record_id and state["selection"] else None)
    else:
        if foreign:
            ctx._left_review_wait = None
        if wake_on is not None and not isinstance(wake_on, dict):
            raise ValueError("wake_on must be an object of explicit source ids")
        args = dict(wake_on or {})
        unknown = set(args) - {"senders", "tasks", "runs", "reviewers", "wake_at"}
        if unknown:
            raise ValueError("wake_on: unknown selectors " + ", ".join(sorted(unknown)))
        chosen = selectors(ctx, **args)
        owned = _owned_runs(ctx) if chosen["runs"] else {}
        if any(owned[rid].review_owned for rid in chosen["runs"]):
            raise ValueError("wake_on.runs names ordinary owned runs; select review seats with reviewers")
        state = {"task_id": own, "selection": chosen, "seen_helpers": [], "seen_outcomes": []}
    ctx._review_wait_selection = state
    return {key: copy.deepcopy(state[key]) for key in ("selection", "seen_helpers", "seen_outcomes")}


def bind_panel(ctx, panel):
    """Resolve row handles once against this ask's counted seats, before dispatch."""
    from ouroboros.reviewer_slot_config import catalog_review_row
    from ouroboros.review_pause import _RECEPTIVE

    state = getattr(ctx, "_review_wait_selection", None) or {}
    chosen = state.get("selection") or {}
    slots = {seat.slot.slot_id for seat in panel.seats if not seat.additional}
    resolved = []
    for name in chosen.get("reviewer_slots") or ():
        identity = name if name in slots else catalog_review_row(None, name).subagent_id
        if identity not in slots:
            raise ValueError(f"wake_on.reviewers: {name!r} is not a counted seat of this wave")
        if identity not in resolved:
            resolved.append(identity)
    chosen["reviewer_slots"] = resolved
    if _RECEPTIVE.get() is not None:
        _RECEPTIVE.get().update(copy.deepcopy(state))


def bind_roster(ctx, rows):
    """Freeze the selected copies to their wave/operation/slot before workers start."""
    from ouroboros.review_pause import _RECEPTIVE

    state = getattr(ctx, "_review_wait_selection", None) or {}
    chosen = state.get("selection") or {}
    selected = set(chosen.get("reviewer_slots") or ())
    chosen["reviewers"] = [{"wave_id": str(getattr(ctx, "_current_review_retry_key", "") or ""),
                            "operation_id": row["operation_id"], "slot_id": row["slot_id"]}
                           for row in rows if row.get("slot_id") in selected and row.get("operation_id")
                           and not (row.get("review_plan") or {}).get("additional")]
    if _RECEPTIVE.get() is not None:
        _RECEPTIVE.get().update(copy.deepcopy(state))


def _outcome_token(identity):
    """A seat outcome's token on delivery cursors and receipts: its full identity, digested. That identity spells
    out the wave (with the reviewed root's path), the operation and the seat, so a wave's worth outgrows the bounded
    metadata a receipt rides in (``ToolResult``); readers only compare tokens."""
    return "review:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def review_terminals(ctx, chosen, seen=(), entries=()):
    """Only a selected, durably finalized producer outcome can end a review wait."""
    from ouroboros.observability import read_call_payload
    from ouroboros.review_ledger import ledger_root, load_record

    ready, errors = [], []
    own, record, seats, pins = str(ctx.task_id), None, set(), list(chosen.get("reviewers") or ())
    if chosen.get("record_id"):
        try:  # the current listener of this exact record reads its wave's seat outcomes only
            record = load_record(ledger_root(ctx), chosen["record_id"]) or {}
            if not _current_listener(ctx, record):
                return []
            pins = _refresh_reviewer_pins(record, pins)
            if record.get("task_id") != own:
                seats = {(record.get("review_wave_id"), row.get("operation_id"), row.get("seat_id"))
                         for row in record.get("rows") or ()}
        except Exception as exc:
            record = None
            errors.append(exc)
    live = {entry.operation_id: entry for entry in entries if entry is not None}
    for pin in pins:
        operation, slot, wave = (pin[key] for key in ("operation_id", "slot_id", "wave_id"))
        reader = str(record["task_id"]) if (wave, operation, slot) in seats else own
        actor, entry = None, live.get(operation)
        if entry is not None:
            if entry.event.is_set() and entry.actor is not None:
                actor = {"review_wave_id": entry.actor.review_wave_id, "outcome": entry.actor.outcome}
        else:
            for suffix in ("response", "error"):
                try:
                    manifest, payload, _ = read_call_payload(ledger_root(ctx), task_id=reader,
                                                           call_id=f"{operation}_{suffix}")
                    if manifest.get("producer_complete"):
                        actor = payload.get("producer_outcome")
                        break
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    errors.append(exc)
        if actor is None and record:
            try:
                if record.get("task_id") == reader and record.get("review_wave_id") == wave:
                    actor = next(({**row, "review_wave_id": wave} for row in record.get("rows") or ()
                                  if row.get("operation_id") == operation and row.get("seat_id") == slot), None)
            except Exception as exc:
                errors.append(exc)
        outcome = (actor or {}).get("outcome") or {}
        identity = f"review:{wave}:{operation}:{slot}:{outcome.get('outcome_id', '')}"
        token = _outcome_token(identity)
        if (outcome.get("outcome_id") and outcome.get("status") and outcome["status"] != "pending" and not outcome.get("additional")
                and outcome.get("operation_id") == operation and outcome.get("slot_id") == slot
                and (actor or {}).get("review_wave_id") == wave
                and token not in seen and identity not in seen):  # a cursor saved before tokens holds the identity
            ready.append({"id": token, "reason": "review_result", "actor_id": f"review:{operation}:{slot}",
                          **pin, "outcome": copy.deepcopy(outcome)})
    if not ready and errors:
        raise errors[0]
    return ready


def ready_helpers(ctx, state, *, entries=()):
    from ouroboros.model_sleep import terminal_readiness
    from ouroboros.store import acquisition_budget

    try:
        with acquisition_budget(0):
            return terminal_readiness(ctx, state.get("selection") or {},
                                      [*(state.get("seen_helpers") or ()), *(state.get("seen_outcomes") or ())],
                                      ordinary_only=True, review_entries=entries)
    except Exception:
        log.debug("Review terminal readiness unavailable", exc_info=True)
        return []


def wake_reason(ctx, state, *, entries=()):
    """Attention precedes selected terminals; a peek never advances any cursor."""
    from ouroboros import model_sleep
    from ouroboros.store import acquisition_budget

    chosen = {"any_mail": True, **(state.get("selection") or {})}
    observed = {}
    state.pop("waking_actor_id", None)
    try:
        with acquisition_budget(0):
            reason = model_sleep.wake_reason(ctx, chosen, seen=[*(state.get("seen_helpers") or ()),
                *(state.get("seen_outcomes") or ())], review_entries=entries, wake_fact=observed)
        if observed.get("actor_id"):
            state["waking_actor_id"] = observed["actor_id"]
        return reason
    except Exception:
        log.debug("Review readiness unavailable; retaining custody", exc_info=True)
        return ""


def wake_on_schema():
    return {"type": "object", "additionalProperties": False,
            "description": "End only this wait on selected mail, task/run terminals, counted reviewer outcomes or wake_at. "
                           "Explicit ids only; omitted/empty means the whole wave plus mail/control attention. "
                           "Owner input, responsible-child questions and host-marked waking place arrivals bypass selectors; context-only place arrivals stay unread until a genuine wake. Reviewers continue; waking grants no landing. "
                           "Panel waves only; subject=system retains its synchronous report wait.",
            "properties": {**{key: {"type": "array", "items": {"type": "string"}, "description": description}
                              for key, description in {
                                  "senders": "Task ids whose mail wakes you.", "tasks": "Readable task ids whose terminal wakes you.",
                                  "runs": "Your ordinary delegated run ids whose terminal wakes you.",
                                  "reviewers": "Counted row ids/handles in this ask; answer or terminal unavailability wakes you."}.items()},
                           "wake_at": {"type": "string", "description": "Absolute ISO-8601 time with timezone."}}}


def save_wait(ctx, record_id, state):
    from ouroboros.review_ledger import RecordUnchanged, ledger_root, revise_record

    own = str(getattr(ctx, "task_id", "") or "")
    accepted = False

    def merge(record):
        nonlocal accepted
        if not _current_listener(ctx, record):
            raise RecordUnchanged()
        accepted = True
        origin = str(record.get("task_id") or "")
        old = record.get("waiting") or {}
        old_owner = _wait_owner(record, old) or origin
        same_owner = not old or old_owner == own
        by_task = copy.deepcopy(old.get("outcome_states_by_task") or {})
        if old.get("seen_outcome_states") and old_owner:
            by_task[old_owner] = sorted(set(by_task.get(old_owner) or ()) | set(old["seen_outcome_states"]))
        incoming = copy.deepcopy(state)
        incoming.setdefault("task_id", own)
        if own != origin:
            incoming.setdefault("attempt_task_id", origin)
        if not same_owner and incoming.get("duty_id") == old.get("duty_id"):
            incoming.pop("duty_id", None)
        merged = {**incoming, **{key: sorted((set(old.get(key) or ()) if same_owner else set())
                                             | set(incoming.get(key) or ()))
                                 for key in ("seen_helpers", "seen_outcomes", "seen_outcome_states")}}
        if merged.get("seen_outcome_states") and own:
            by_task[own] = sorted(set(by_task.get(own) or ()) | set(merged["seen_outcome_states"]))
        if by_task:
            merged["outcome_states_by_task"] = by_task
        record["waiting"] = merged
        return record

    record = revise_record(ledger_root(ctx), record_id, merge)
    if not accepted or not record:
        return None
    if (record.get("waiting") or {}).get("duty_id"):
        from ouroboros.late_review_marks import update_wait_cursors
        update_wait_cursors(ledger_root(ctx), record["waiting"]["duty_id"], record["waiting"])
    return record


def _closed_outcomes(record):
    """``(identity, status)`` of each seat whose answer can no longer arrive, counted or added, in row order."""
    from ouroboros.review_ledger import seat_open

    for row in record.get("rows") or ():
        if seat_open(row):
            continue
        canonical = row.get("outcome") or {}
        status = str(canonical.get("status") or row.get("status") or "")
        if canonical.get("outcome_id") and record.get("review_wave_id"):
            yield f"review:{record['review_wave_id']}:{row.get('operation_id')}:{row['seat_id']}:{canonical['outcome_id']}", status
            continue
        outcome = (row.get("outcome") or {}).get("outcome_id") or row.get("outcome_id") or hashlib.sha256(json.dumps(
            {key: row.get(key) for key in ("status", "answers", "source_refs")}, sort_keys=True).encode()).hexdigest()
        yield f"{record['record_id']}:{row.get('operation_id') or row['seat_id']}:{outcome}", status


def outcome_ids(record):
    return [identity for identity, _status in _closed_outcomes(record)]


def outcome_states(record):
    """What a delivery of this record carries, ``<outcome token>#<status>`` per closed seat: a later answer of the
    same operation (lost, then answered) is a new state; a revision that records cost or custody is not."""
    return [f"{_outcome_token(identity)}#{status}" for identity, status in _closed_outcomes(record)]


def shown_outcomes(record, rendered_rows=None):
    """One bounded identity of the outcome states a response carried (``outcome_states``; ``rendered_rows`` when it
    renders fewer than the record holds): the reference an author decides with quotes it, so a cost or custody revision
    keeps a decision standing and any added or changed outcome does not. It proves delivery, never a read receipt."""
    shown = record if rendered_rows is None else {**record, "rows": list(rendered_rows)}
    return hashlib.sha256("\n".join(sorted(outcome_states(shown))).encode("utf-8")).hexdigest()[:32]


def delivery_ready(ctx, state):
    """Facts exposed by this return, including the existing absolute time selector."""
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now

    ready = ready_helpers(ctx, state)
    stamp = (state.get("selection") or {}).get("wake_at")
    when = parse_deadline_ts(stamp or "")
    if when is not None and utc_now() >= when and f"time:{stamp}" not in (state.get("seen_helpers") or ()):
        ready.append({"id": f"time:{stamp}", "reason": "wake_at"})
    return ready


def publish_delivery(ctx, text, record, rendered_rows=None):
    """Publish exact receipt metadata beside the result; producing it is not an ACK.

    A receipt names only what this response carries. ``rendered_rows`` are the record's rows as ``text`` renders
    them (a review payload's own ``rows``): their outcomes go on the wait's cursor and their states are certified,
    never what a later read of ``record`` holds, so a seat that settled after the text was rendered stays
    unreceived. A response that renders no ledger rows (a commit's own report) certifies no outcome state, and its
    cursor keeps naming ``record``'s outcomes."""
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, _published_tool_result, _classify_legacy_text

    record_id = str(record.get("record_id") or "")
    if not record_id:
        return text
    own = str(getattr(ctx, "task_id", "") or "")
    left, mem = getattr(ctx, "_left_review_wait", None), getattr(ctx, "_review_wait_selection", None)
    own_left = left if isinstance(left, dict) and left and _wait_owner(record, left) in ("", own) else None
    own_mem = mem if isinstance(mem, dict) and mem and _wait_owner(record, mem) in ("", own) else None
    state = own_left or own_mem or _listener_wait(ctx, record)
    helpers = delivery_ready(ctx, state) if not record.get("task_id") or _current_listener(ctx, record) else []
    from ouroboros.model_sleep import question_sources
    questions = question_sources(ctx)
    shown = record if rendered_rows is None else {**record, "rows": list(rendered_rows)}  # identity stays the record's
    receipt = {"record_id": record_id, "helpers": [row["id"] for row in helpers if not row["id"].startswith("review:")],
               "outcomes": sorted({_outcome_token(identity) for identity in outcome_ids(shown)}
                                  | {row["id"] for row in helpers if row["id"].startswith("review:")}),
               "questions": questions, "outcome_states": [] if rendered_rows is None else outcome_states(shown)}
    suffix = "\n\n[Selected results] " + json.dumps(helpers) if helpers else ""
    if questions:
        suffix += "\n\n[Responsible child questions] " + json.dumps(questions)
    prior = _published_tool_result(ctx, None)
    if not isinstance(prior, ToolResult) or prior.text != text:
        status, code, meta = _classify_legacy_text(text)
        prior = ToolResult(status=status, code=code, text=text, meta=meta)
    return _publish_tool_result(ctx, dataclasses.replace(prior, text=text + suffix,
                                meta={**prior.meta, "review_delivery": receipt}))


def _carries_whole(record, receipt):
    """Whether this receipt carried all of ``record`` as it stands now: the wave settled with no open additional seat,
    and every current outcome state is one its response rendered (``publish_delivery``). Cursor ids prove nothing here:
    a helper notice names them and a report that renders no rows certifies none, and a seat that settled after the
    render leaves it short."""
    from ouroboros.review_ledger import open_additional_seats

    current = set(outcome_states(record))
    return (record.get("state") == "settled" and not open_additional_seats(record)
            and bool(current) and current <= set(receipt.get("outcome_states") or ()))


def received_whole(record, prior=None, *, task_id=""):
    source = (prior if prior is not None else record) or {}
    waiting = source.get("waiting") or {}
    owner = str(task_id or (record or {}).get("task_id") or source.get("task_id") or "")
    holder = _wait_owner(source or record, waiting)
    states = (waiting.get("seen_outcome_states") if waiting and holder in ("", owner)
              else (waiting.get("outcome_states_by_task") or {}).get(owner))
    return _carries_whole(record or {}, {"outcome_states": states})


def _verified_listener(ctx, record):
    """A Continue successor for this commit record on the same checkout, never behind a newer listener (#2901). One
    fresh read of the recorded chain from the record's task to this one answers both questions: only that task and its
    verified successors save a wait, so its holder lies on the same path and is no newer listener exactly when the
    chain passes through it."""
    from ouroboros.owner_continue import continuation_chain
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state import make_repo_key

    own, origin = str(getattr(ctx, "task_id", "") or ""), str(record.get("task_id") or "")
    waiting, root = record.get("waiting") or {}, str((record.get("subject") or {}).get("root") or "")
    holder, attempt_owner = _wait_owner(record, waiting), str(waiting.get("attempt_task_id") or "")
    repo = getattr(ctx, "repo_dir", None)
    if not (bool(own and origin) and own != origin and record.get("surface") == "commit_gate"
            and attempt_owner in ("", origin) and bool(root and repo)
            and make_repo_key(pathlib.Path(root)) == make_repo_key(pathlib.Path(repo))):
        return False
    chain = continuation_chain(ledger_root(ctx), origin, own)
    return bool(chain) and (not holder or holder == own or holder in chain)


def _current_listener(ctx, record):
    own, origin = str(getattr(ctx, "task_id", "") or ""), str((record or {}).get("task_id") or "")
    if not record or not own or not origin:
        return False
    return _wait_owner(record) in ("", own) if own == origin else _verified_listener(ctx, record)


def acknowledge_delivery(ctx, receipt):
    """Called after complete transcript insertion, never from a readiness peek. Once a saved state holds it
    (``working_checkpoint``), the cursor is durable and a whole receipt of a result record ends that record's unread
    duty; a crash or a failed save before then acknowledges nothing."""
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.working_checkpoint import defer_content_ack

    record_id = str(receipt.get("record_id") or "")
    try:
        record = load_record(ledger_root(ctx), record_id)
    except (OSError, ValueError):
        log.warning("Review delivery cursor stays unacknowledged", exc_info=True)
        return
    if not _current_listener(ctx, record):
        return
    if receipt.get("questions"):
        from ouroboros.sleep_custody import record_question_notice
        record_question_notice(ctx, receipt["questions"])
    own, origin = str(ctx.task_id), str(record.get("task_id") or "")
    mem = getattr(ctx, "_review_wait_selection", None)
    own_mem = copy.deepcopy(mem) if isinstance(mem, dict) and mem and _wait_owner(record, mem) in ("", own) else {}
    state = _listener_wait(ctx, record) or copy.deepcopy(own_mem)
    state.update(task_id=own, record_id=record_id, **({"attempt_task_id": origin} if own != origin else {}))
    if not state.get("selection") and own_mem.get("selection"):
        state["selection"] = copy.deepcopy(own_mem["selection"])
    for target, source in (("seen_helpers", "helpers"), ("seen_outcomes", "outcomes"),
                           ("seen_outcome_states", "outcome_states")):
        state[target] = sorted(set(state.get(target) or ()) | set(own_mem.get(target) or ())
                               | set(receipt.get(source) or ()))
    ctx._review_wait_selection = state
    wait = getattr(ctx, "_left_review_wait", None)
    if isinstance(wait, dict) and wait.get("record_id") == record_id and _wait_owner(record, wait) in ("", own):
        wait.update({key: copy.deepcopy(state[key]) for key in ("selection", "seen_helpers", "seen_outcomes", "seen_outcome_states")
                     if key in state})

    def commit():
        saved = save_wait(ctx, record_id, state)
        if saved and saved.get("surface") == "result" and received_whole(saved, saved, task_id=own):
            from ouroboros.review_pause import retire_unread_review

            retire_unread_review(ctx, surface="result", record_id=record_id)

    defer_content_ack(ctx, commit)


def append_helper_notice(ctx, messages):
    from ouroboros.model_sleep import question_sources
    from ouroboros.sleep_custody import record_question_notice

    wait = getattr(ctx, "_left_review_wait", None) or {}
    ready = delivery_ready(ctx, wait)
    if ready:
        messages.append({"role": "user", "content": "[Selected results] " + json.dumps(ready)})
        acknowledge_delivery(ctx, {"record_id": wait.get("record_id"),
                                  "helpers": [row["id"] for row in ready if not row["id"].startswith("review:")],
                                  "outcomes": [row["id"] for row in ready if row["id"].startswith("review:")]})
    if questions := question_sources(ctx):
        messages.append({"role": "user", "content": "[Responsible child questions] " + json.dumps(questions)})
        record_question_notice(ctx, questions)


def _call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def _reference_gap(reason, *, reader=None):
    return {"status": "reference_unavailable", "reason": reason, **({"reader": reader} if reader else {}),
            "note": "Read the original review's tool result or settlement notice if retained; do not start a replacement review."}


def _plan_wait_redirect(ctx, entry, operation):
    from ouroboros.plan_record import load_plan_record
    from ouroboros.tools.plan_ask import state_location

    root, task_id = state_location(ctx)
    record = load_plan_record(root, task_id)
    matches = [ask for ask in record.get("asks", ()) if ask.get("retry_key") == entry.review_wave_id and any(
        seat.get("seat_id") == entry.review_slot_id and seat.get("operation_id") == operation
        for seat in ask.get("participants", ()))]
    reader = _call("plan_task")  # no ask/continuation: the existing task's free plan read collects recorded pending seats
    if len(matches) != 1:
        return _reference_gap("plan_ask_binding_unavailable", reader=reader)
    return {"status": "read_existing", "surface": "intention", "ask_id": matches[0]["ask_id"], **reader}


def _recorded_wait_wave(ctx, drive, entry, operation):
    """Joined by exact task/wave/seat/operation, never by the latest review or the shape of a run id."""
    from ouroboros.review_ledger import load_record, task_record_page

    def matching(record_id):
        record = load_record(drive, record_id) or {}
        if (record.get("record_id") == record_id and record.get("task_id") == entry.task_id
                and record.get("review_wave_id") == entry.review_wave_id and any(
                    isinstance(row, dict) and row.get("seat_id") == entry.review_slot_id
                    and row.get("operation_id") == operation for row in record.get("rows") or ())):
            return record
        return None

    left = getattr(ctx, "_left_review_wait", None) or {}
    if (isinstance(left, dict) and (left.get("attempt_task_id") or left.get("task_id")) == entry.task_id
            and left.get("retry_key") == entry.review_wave_id
            and left.get("record_id") and (found := matching(left["record_id"]))):
        return found  # exact checkpoint address, even after rotation
    # Reuse the task-card's bounded hot projection; never search the archival index or attempt history.
    page, _ = task_record_page(drive, entry.task_id, hot_only=True)
    matches = [record for hint in page if hint.get("review_wave_id") == entry.review_wave_id
               and (record := matching(hint["record_id"])) is not None]
    return matches[0] if len(matches) == 1 else None


def _review_wait_redirect(ctx, entry):
    from pathlib import Path

    from ouroboros import delegate_custody
    from ouroboros.review_ledger import ledger_root
    from ouroboros.store import StoreError
    from ouroboros.tools.review_collection import reference_for

    try:
        root = delegate_custody.custody_root(ctx)
        if entry.ledger_root and Path(entry.ledger_root).resolve() != root.resolve():
            return _reference_gap("invocation_store_mismatch")
        invocation = delegate_custody.invocation_record(root, entry.invocation_id) or {}
        wave, slot = entry.review_wave_id, entry.review_slot_id
        operation, surface = invocation.get("operation_id"), invocation.get("surface")
        if not (wave and slot and operation and surface and invocation.get("state") == "started"
                and invocation.get("task_id") == entry.task_id and invocation.get("run_id") == entry.run_id
                and invocation.get("review_wave_id") == wave and invocation.get("review_slot_id") == slot
                and invocation.get("slot_id") == slot and invocation.get("review_skill", "") == entry.review_skill):
            return _reference_gap("review_invocation_binding_unavailable")
        if surface == "plan_review":
            return _plan_wait_redirect(ctx, entry, operation)
        if surface not in {"multi_model_review", "scope_review", "skill_review"}:
            return _reference_gap("review_surface_has_no_reference_reader")
        record = _recorded_wait_wave(ctx, ledger_root(ctx), entry, operation)
    except (OSError, StoreError, TypeError, ValueError):
        return _reference_gap("review_evidence_unreadable")
    if record is None:
        return _reference_gap("review_wave_binding_unavailable",
                              reader=_call("review_status", task_id=entry.task_id, include_raw=True))
    stored = _call("query_data", op="json", kind="review_ledger", ids=[record["record_id"]])
    kind = record.get("surface")
    if kind == "skill":
        # skill_review(reference-only) currently enters normal review, NOT the continuation collector.
        return {"status": "read_existing", "surface": "skill", **stored,
                "note": "This reads the saved skill review; it does not collect a still-running seat."}
    if kind not in {"commit_gate", "change", "preflight", "result"}:
        return _reference_gap("review_surface_has_no_reference_collector", reader=stored)
    reference = reference_for(record)
    if kind == "commit_gate":
        # This is the original handback's minimal shape. The collector still resolves the original attempt
        # and repository, and refuses rather than dispatches when those facts are no longer available.
        reference["surface"] = "commit"
        wait = record.get("waiting") or {}
        if (isinstance(wait, dict) and wait.get("record_id") == record["record_id"]
                and (wait.get("attempt_task_id") or wait.get("task_id")) == entry.task_id
                and wait.get("retry_key") == wave and type(wait.get("attempt")) is int and wait["attempt"] > 0):
            reference["attempt"] = wait["attempt"]
    return {"status": "collect_existing", "surface": kind, **_call(
        "commit_reviewed" if kind == "commit_gate" else "review_change", review_reference=reference),
        "reader": stored}


def review_wait_refusal(ctx, tool, entries):
    """Refuse a selected owned reviewer before the ordinary sleep; never collect or change its panel here."""
    from ouroboros.delegate_shared import _fail

    return _fail(tool, "review_panel_owns_run", (
        "No selected source was waited on. Review runs belong to their panel, not ordinary supervision. "
        "Use the reported collector or reader for the existing review; wait again on ordinary sources separately."),
        slept=False, review_runs=[{"run_id": run.run_id, "collection": _review_wait_redirect(ctx, run)}
                                  for run in entries if run.review_owned and run.task_id == str(ctx.task_id)])
