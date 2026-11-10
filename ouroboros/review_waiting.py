"""Chosen wake sources and delivery cursors of one retained review wait.

Readiness never consumes a fact. A complete transcript insertion stages its
cursor; the working checkpoint makes that acknowledgement durable. A cursor
never discharges a room duty. ``seen_outcome_states`` retains rendered feedback
separately in the same ledger wait: it ends unread duties here and lets the
terminal collector distinguish a received unwaited outcome from mere settlement.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging

log = logging.getLogger(__name__)


def select_helpers(ctx, wake_on=None):
    """Retain a saved selection; a new wait selects only explicitly named sources."""
    from ouroboros.model_sleep import selectors, _owned_runs
    prior = getattr(ctx, "_left_review_wait", None)
    if isinstance(prior, dict):
        # Legacy waits mean the whole wave, not a newly inferred helper roster.
        state = {key: copy.deepcopy(prior.get(key) or ([] if key != "selection" else {}))
                 for key in ("selection", "seen_helpers", "seen_outcomes")}
    else:
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
        state = {"selection": chosen, "seen_helpers": [], "seen_outcomes": []}
    ctx._review_wait_selection = state
    return copy.deepcopy(state)


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
    record = None
    live = {entry.operation_id: entry for entry in entries if entry is not None}
    for pin in chosen.get("reviewers") or ():
        operation, slot, wave = (pin[key] for key in ("operation_id", "slot_id", "wave_id"))
        actor, entry = None, live.get(operation)
        if entry is not None:
            if entry.event.is_set() and entry.actor is not None:
                actor = {"review_wave_id": entry.actor.review_wave_id, "outcome": entry.actor.outcome}
        else:
            for suffix in ("response", "error"):
                try:
                    manifest, payload, _ = read_call_payload(ledger_root(ctx), task_id=ctx.task_id,
                                                           call_id=f"{operation}_{suffix}")
                    if manifest.get("producer_complete"):
                        actor = payload.get("producer_outcome")
                        break
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    errors.append(exc)
        if actor is None and chosen.get("record_id"):
            try:
                if record is None:
                    record = load_record(ledger_root(ctx), chosen["record_id"]) or {}
                if record.get("task_id") == ctx.task_id and record.get("review_wave_id") == wave:
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
    from ouroboros.review_ledger import ledger_root, revise_record

    def merge(record):
        old = record.get("waiting") or {}
        record["waiting"] = {**copy.deepcopy(state), **{key: sorted(set(old.get(key) or ()) | set(state.get(key) or ()))
                             for key in ("seen_helpers", "seen_outcomes", "seen_outcome_states")}}
        return record
    record = revise_record(ledger_root(ctx), record_id, merge)
    if record and (record.get("waiting") or {}).get("duty_id"):
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
    state = (getattr(ctx, "_left_review_wait", None) or getattr(ctx, "_review_wait_selection", None)
             or record.get("waiting") or {})
    helpers = delivery_ready(ctx, state)
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
    """Whether this receipt carried all of ``record`` as it stands now: the wave settled and every current outcome
    state is one its response rendered (``publish_delivery``). Cursor ids prove nothing here: a helper notice names
    them and a report that renders no rows certifies none, and a seat that settled after the render leaves it short."""
    current = set(outcome_states(record))
    return record.get("state") == "settled" and bool(current) and current <= set(receipt.get("outcome_states") or ())


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
    if not record or record.get("task_id") != str(ctx.task_id):
        return
    if receipt.get("questions"):
        from ouroboros.sleep_custody import record_question_notice
        record_question_notice(ctx, receipt["questions"])
    state = copy.deepcopy(record.get("waiting") or getattr(ctx, "_review_wait_selection", None) or {})
    for target, source in (("seen_helpers", "helpers"), ("seen_outcomes", "outcomes"),
                           ("seen_outcome_states", "outcome_states")):
        state[target] = sorted(set(state.get(target) or ()) | set(receipt.get(source) or ()))
    ctx._review_wait_selection = state
    wait = getattr(ctx, "_left_review_wait", None)
    if isinstance(wait, dict) and wait.get("record_id") == record_id:
        wait.update({key: copy.deepcopy(state[key]) for key in ("selection", "seen_helpers", "seen_outcomes", "seen_outcome_states")
                     if key in state})

    def commit():
        saved = save_wait(ctx, record_id, state)
        if (saved and saved.get("surface") == "result"
                and _carries_whole(saved, {"outcome_states": saved["waiting"]["seen_outcome_states"]})):
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
